# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Layer an exogenous electrified-heat load onto an electricity-only network.

Fork-specific; not part of upstream PyPSA-Eur. Produces the network that PyPSARO's ARO/C&CG
resilience workflow consumes, so that PyPSARO itself carries no content assumptions -- it only
points `run.network` at this rule's output.

Heat is added as *loads on the existing AC buses*, not as a modelled heat sector: no heat buses,
no heat-pump Links, no thermal storage. That keeps the outage-disconnection paths and design
transfer in PyPSARO untouched, at the cost of heat being inflexible during an outage (a
deliberate, documented limitation -- see the parent repo's `memory/aro-study-limitations.md`).

The demand construction is *imported* from `prepare_sector_network` rather than reimplemented:
`build_heat_demand` handles the sector/use decomposition, the final-energy-to-useful-heat
efficiency conversion, and the subtraction of today's already-electrified heat from the metered
load. Only `add_heat`'s bus/link creation is replaced, by dividing each heat system's demand by
that system's own COP.
"""

import logging

import pandas as pd
import pypsa
import xarray as xr

from scripts._helpers import configure_logging, set_scenario_config
from scripts.definitions.heat_system import HeatSystem
from scripts.prepare_sector_network import build_heat_demand, get

logger = logging.getLogger(__name__)

# Existing electricity-only networks leave `Load.carrier` as the empty string, whereas
# `build_heat_demand` locates the loads to correct via `carrier == "electricity"`. Left as-is it
# would subtract from nothing, silently. Naming them also makes the six heat loads below
# distinguishable per node (memory: pypsa-eur-sector-run-gotchas).
ELECTRICITY_CARRIER = "electricity"

HEAT_PUMP_CARRIER = "{heat_system} heat pump electricity"
RESISTIVE_CARRIER = "{heat_system} resistive heater electricity"
CARRIER_COLORS = {
    "rural heat pump electricity": "#2b8cbe",
    "urban decentral heat pump electricity": "#7bccc4",
    "urban central heat pump electricity": "#084081",
    "rural resistive heater electricity": "#fdbb84",
    "urban decentral resistive heater electricity": "#fc8d59",
    "urban central resistive heater electricity": "#d7301f",
}


def cop_heat_system(heat_system: HeatSystem) -> str:
    """
    Map a `HeatSystem` onto the coarser `heat_system` coordinate of the COP profiles.

    `HeatSystem` has five members (residential/services x rural/urban-decentral, plus urban
    central) but `cop_profiles` carries only three curves -- both sector variants of a given
    system share one COP. Stripping the sector prefix is the mapping.
    """
    return heat_system.value.replace("residential ", "").replace("services ", "")


def resolve_shares(country: str, params: dict) -> tuple[float, float]:
    """Heat-pump and resistive shares for one country, falling back to the default."""
    shares = params["shares"].get(country, params["default_shares"])
    return float(shares["heat_pump"]), float(shares["resistive"])


def to_network_resolution(df: pd.DataFrame, snapshots: pd.Index) -> pd.DataFrame:
    """
    Aggregate an hourly frame onto the network's (typically coarser) snapshots.

    By **mean**, not by sampling: the electricity load these join was itself averaged when
    pypsa-eur applied `clustering.temporal.resolution_elec`, and point-sampling a 3-hourly
    network from an hourly profile would overstate the peak.
    """
    freq = pd.infer_freq(snapshots)
    if freq is None:
        logger.warning(
            "Could not infer a frequency from the network snapshots; falling back to "
            "reindexing the heat profile without averaging."
        )
        return df.reindex(snapshots)
    return df.resample(freq).mean().reindex(snapshots)


def add_heat_load(
    n: pypsa.Network,
    heat_demand: pd.DataFrame,
    cop_profiles_file: str,
    district_heat_share_file: str,
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    params: dict,
    sector_params: dict,
    investment_year: int,
) -> None:
    """Convert heat demand into electricity loads on the AC buses, per heat system."""
    cop = xr.open_dataset(cop_profiles_file)
    # NB the data variable is named `temperature` but holds COP -- a mislabel upstream, not the
    # wrong file (values sit at 2.4-5.4, not degrees). See memory: pypsa-eur-sector-run-gotchas.
    cop = cop[next(iter(cop.data_vars))]

    # Both of the following mirror `add_heat` and are NOT optional detail: skipping the space
    # reduction overstates heat demand by ~25% (space is ~86% of it, cut by 29% at 2050), and
    # skipping the district-heating loss understates urban-central load by 15%. Verified against
    # the reference run's own saved config and its "Assumed space heat reduction of 29.00%" log.
    if sector_params["reduce_space_heat_exogenously"]:
        dE = get(sector_params["reduce_space_heat_exogenously_factor"], investment_year)
        logger.info(f"Assumed space heat reduction of {dE:.2%}")
        for sector in {hs.sector.value for hs in HeatSystem if hs.sector is not None}:
            heat_demand[f"{sector} space"] = (1 - dE) * heat_demand[f"{sector} space"]
    dh_loss = sector_params["district_heating"]["district_heating_loss"]

    district_heat_info = pd.read_csv(district_heat_share_file, index_col=0)
    dist_fraction = district_heat_info["district fraction of node"]
    urban_fraction = district_heat_info["urban fraction"]

    sources = params["heat_pump_sources"]
    # The column is float64 on disk, so this cast is a runtime no-op; it exists so the
    # per-system lookup below yields a float rather than pandas' broad `Scalar` union.
    efficiencies = costs["efficiency"].astype(float)

    # A `shares` key that matches no country here is silently ignored by the lookup below, so
    # a typo ("de", "DEU") would quietly leave that country on `default_shares` and produce
    # subtly wrong loads with no signal. Warn rather than raise: listing countries this run
    # does not cover is legitimate (one shared config driving per-country scenario runs).
    unknown = sorted(set(params["shares"]) - set(pop_layout["ct"].unique()))
    if unknown:
        logger.warning(
            f"aro.heat.shares lists {unknown}, which match no country in this network "
            f"({sorted(pop_layout['ct'].unique())}). Those entries are ignored. If a country "
            "was meant to be covered, check the code's spelling and case (ISO-2, upper-case); "
            "otherwise it is simply not part of this run and will use default_shares."
        )

    # Accumulate per heat system ("rural" / "urban decentral" / "urban central"), for BOTH
    # technologies. That split is an exogenous assumption -- demand is allocated across the three
    # systems by urban_fraction and dist_fraction (the latter driven by
    # sector.district_heating.potential), never chosen by an optimiser -- so it is exactly the
    # axis someone will want to interrogate. Keeping both technologies decomposable along it lets
    # a district-heating sensitivity be read off the network directly.
    heat_pump_load: dict[str, pd.DataFrame] = {}
    resistive_load: dict[str, pd.DataFrame] = {}

    for heat_system in HeatSystem:
        if heat_system == HeatSystem.URBAN_CENTRAL:
            nodes = dist_fraction.index[dist_fraction > 0]
        else:
            nodes = pop_layout.index
        if nodes.empty:
            continue

        factor = heat_system.heat_demand_weighting(
            urban_fraction=urban_fraction[nodes], dist_fraction=dist_fraction[nodes]
        )
        if heat_system == HeatSystem.URBAN_CENTRAL:
            demand = heat_demand.T.groupby(level=1).sum().T[nodes]
        else:
            sector = heat_system.sector.value
            demand = (
                heat_demand[[f"{sector} water", f"{sector} space"]]
                .T.groupby(level=1)
                .sum()
                .T[nodes]
            )
        # District heating pipe losses apply only to the central system, as in `add_heat`.
        if heat_system == HeatSystem.URBAN_CENTRAL:
            factor = factor * (1 + dh_loss)
        demand = demand.multiply(factor)

        group = cop_heat_system(heat_system)
        source = sources[group]
        # dims after .sel are (time, name), so to_pandas() is already time-indexed -- no transpose.
        curve = cop.sel(heat_system=group, heat_source=source).to_pandas()[nodes]
        # Not every (system, source) pair exists: pypsa-eur offers ground only for rural, so
        # e.g. "urban central: ground" selects an all-NaN slice. Without this guard that NaN
        # flows through the division and gets filled with zero downstream, silently deleting
        # this heat system's entire heat-pump load rather than failing.
        if curve.isna().any().any():
            available = [
                str(src)
                for src in cop.heat_source.values
                if not cop.sel(heat_system=group, heat_source=src).isnull().all()
            ]
            raise ValueError(
                f"COP profile for heat system '{group}' with source '{source}' contains NaN. "
                f"pypsa-eur defines only {available} for this heat system (compare "
                "sector.heat_pump_sources). Correct aro.heat.heat_pump_sources -- treating "
                "these as zero would silently drop this system's heat-pump load entirely."
            )

        # `ct` is itself just `pop.index.str[:2]` (build_clustered_population_layouts), so this
        # is not a more correct derivation than slicing here -- it routes through the interface
        # five other upstream scripts already use, so a future change to how country is derived
        # is inherited rather than silently diverged from. `.loc` also raises for a node absent
        # from pop_layout, where slicing would invent a country code for it.
        countries = pop_layout.loc[nodes, "ct"]
        hp_share = countries.map(lambda ct: resolve_shares(ct, params)[0])
        res_share = countries.map(lambda ct: resolve_shares(ct, params)[1])

        hp = demand.multiply(hp_share, axis=1).div(curve.loc[demand.index])
        heat_pump_load[group] = heat_pump_load.get(group, 0.0) + hp

        # Read from the cost table, keyed exactly as `add_heat` does. Resistive carries no
        # COP, but its efficiency is NOT 1.0 and differs by heat system (0.99 central, 0.90
        # decentral in the 2050 table) -- hardcoding 1.0 understates the decentral electricity
        # draw by ~11%, in the optimistic direction for a resilience study.
        key = f"{heat_system.central_or_decentral} resistive heater"
        resistive_efficiency = efficiencies.at[key]
        resistive_load[group] = (
            resistive_load.get(group, 0.0)
            + demand.multiply(res_share, axis=1) / resistive_efficiency
        )

    for template, loads in (
        (HEAT_PUMP_CARRIER, heat_pump_load),
        (RESISTIVE_CARRIER, resistive_load),
    ):
        for group, load in loads.items():
            carrier = template.format(heat_system=group)
            _add_loads(n, load, carrier, f" {carrier}")


def _add_loads(n: pypsa.Network, load: pd.DataFrame, carrier: str, suffix: str) -> None:
    """Add one Load per node on its existing AC bus, dropping all-zero nodes."""
    load = to_network_resolution(load, n.snapshots).fillna(0.0)
    load = load.loc[:, load.abs().sum() > 0]
    if load.empty:
        logger.info(f"No {carrier} load to add.")
        return

    n.add(
        "Carrier",
        carrier,
        color=CARRIER_COLORS.get(carrier, "#999999"),
        nice_name=carrier,
    )
    n.add(
        "Load",
        load.columns.tolist(),
        suffix=suffix,
        bus=load.columns,
        carrier=carrier,
        p_set=load,
    )
    logger.info(
        f"Added {len(load.columns)} '{carrier}' loads "
        f"({(load.sum().sum() * n.snapshot_weightings.objective.iloc[0]) / 1e6:.1f} TWh/a)."
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_aro_network",
            clusters=20,
            opts="",
            planning_horizons="2050",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params = snakemake.params.aro_heat
    n = pypsa.Network(snakemake.input.network)

    if not params["enable"]:
        logger.info(
            "aro.heat.enable is false -- writing the network through unchanged."
        )
        n.export_to_netcdf(snakemake.output.network)
    else:
        # Name the existing loads before build_heat_demand runs; it finds what to correct via
        # `carrier == "electricity"`, which an electricity-only network does not otherwise set.
        unnamed = n.loads.index[n.loads.carrier == ""]
        if len(unnamed):
            logger.info(
                f"Setting carrier '{ELECTRICITY_CARRIER}' on {len(unnamed)} unnamed loads so "
                "the existing-electric-heat subtraction can find them."
            )
            n.loads.loc[unnamed, "carrier"] = ELECTRICITY_CARRIER

        pop_weighted_energy_totals = pd.read_csv(
            snakemake.input.pop_weighted_energy_totals, index_col=0
        )
        # `prepare_sector_network` overwrites the space-heating columns with heat-specific
        # totals before calling build_heat_demand. Skipping this does not just rescale demand,
        # it redistributes it between nodes -- verified: with the update the allocation
        # reproduces add_heat's heat loads exactly (ratio 1.0000 per node); without it the
        # per-node ratio scatters between 0.42 and 1.17.
        pop_weighted_energy_totals.update(
            pd.read_csv(snakemake.input.pop_weighted_heat_totals, index_col=0)
        )
        year = int(snakemake.params.energy_totals_year)
        heating_efficiencies = pd.read_csv(
            snakemake.input.heating_efficiencies, index_col=[1, 0]
        ).loc[year]
        pop_layout = pd.read_csv(snakemake.input.pop_layout, index_col=0)
        # processed costs are wide: technology index, parameter columns.
        costs = pd.read_csv(snakemake.input.costs, index_col=0)

        before = n.loads_t.p_set.sum().sum()
        # Subtracts today's already-electrified heat from the metered load in place. This is
        # unconditional by design: skipping it double-counts, so it is logged rather than
        # exposed as a toggle whose "off" position is simply wrong.
        heat_demand = build_heat_demand(
            n,
            snakemake.input.hourly_heat_demand_total,
            pop_weighted_energy_totals,
            heating_efficiencies,
        )
        after = n.loads_t.p_set.sum().sum()
        weight = n.snapshot_weightings.objective.iloc[0]
        logger.info(
            f"Subtracted existing electric heating from the metered load: "
            f"{(before - after) * weight / 1e6:.1f} TWh/a."
        )

        add_heat_load(
            n,
            heat_demand,
            snakemake.input.cop_profiles,
            snakemake.input.district_heat_share,
            pop_layout,
            costs,
            params,
            snakemake.params.sector,
            int(snakemake.wildcards.planning_horizons),
        )

        n.export_to_netcdf(snakemake.output.network)
