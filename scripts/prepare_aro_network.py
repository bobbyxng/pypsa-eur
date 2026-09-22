# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Prepare an electricity-only network for the ARO/C&CG workflow.

Two independent layers, both fork-specific and neither part of upstream PyPSA-Eur:

1. an exogenous electrified-heat load, added as Loads on the existing AC buses
   (`aro.heat.enable`, described below);
2. the H2 cavern-vs-tank split that `attach_stores` omits on the electricity path
   (`cap_hydrogen_storage`, gated on `sector.hydrogen_underground_storage`).

(2) runs unconditionally so the no-heat baseline and the heat run differ in exactly one thing.

Produces the network that PyPSARO's ARO/C&CG resilience workflow consumes, so that PyPSARO
itself carries no content assumptions -- it only points `run.network` at this rule's output.

Heat is added as *loads on the existing AC buses*, not as a modelled heat sector: no heat buses,
no heat-pump Links, no thermal storage. That keeps the outage-disconnection paths and design
transfer in PyPSARO untouched, at the cost of heat being inflexible during an outage (a
deliberate, documented limitation -- see the parent repo's `memory/aro-study-limitations.md`).

The demand construction is *imported* from `prepare_sector_network` rather than reimplemented:
`build_heat_demand` handles the sector/use decomposition, the final-energy-to-useful-heat
efficiency conversion, and the subtraction of today's already-electrified heat from the metered
load. Only `add_heat`'s bus/link creation is replaced, by multiplying each heat system's demand
by that system's own 1/COP.

Heat demand and 1/COP are each aggregated onto the network's snapshots *before* being combined,
which reproduces `add_heat` exactly rather than approximately -- see
`memory: aro-heat-cop-aggregation-order` for why upstream's order is the one to match.
"""

import logging
from itertools import product

import pandas as pd
import pypsa
import xarray as xr

from scripts._helpers import (
    configure_logging,
    get,
    get_temporal_resolution,
    set_scenario_config,
)
from scripts.definitions.heat_sector import HeatSector
from scripts.definitions.heat_system import HeatSystem
from scripts.prepare_sector_network import build_heat_demand

logger = logging.getLogger(__name__)

# `build_heat_demand` locates the loads to correct via `carrier == "electricity"`. Upstream's
# `add_electricity` sets that carrier itself since the streamlined workflow (#1838), so the
# naming pass in `__main__` is a no-op on any network built by the current rules and its log
# line will not fire -- do NOT read that silence as "the subtraction found nothing". The pass
# is kept because pre-#1838 networks left `Load.carrier` empty, where it was load-bearing:
# without it the subtraction matched no loads and silently did nothing. Naming also keeps the
# six heat loads below distinguishable per node (memory: pypsa-eur-sector-run-gotchas).
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
    # `plotting.default.yaml` cannot supply these: `sanitize_carriers` runs inside
    # `compose_network`, before this script adds them. OCGT keeps upstream's `H2 turbine`
    # purple; the CCGT is a darker shade of it.
    "H2 OCGT": "#991f83",
    "H2 CCGT": "#6b1459",
}

# Carrier that `add_electricity.attach_stores` gives the H2 store, bus and links. Upstream's
# sector path uses `H2 Store`; the electricity path reuses the bus carrier verbatim.
H2_CARRIER = "H2"
CAVERN_TECH = "hydrogen storage underground"
TANK_TECH = "hydrogen storage tank type 1 including compressor"
# Both thresholds are upstream's, from `prepare_sector_network.add_storage_and_grids`: sites
# below 2 TWh are dropped as too small to develop, and no single site may exceed 1000 TWh.
CAVERN_MIN_POTENTIAL_TWH = 2.0
CAVERN_MAX_PER_SITE_MWH = 1e9


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


def to_network_resolution(
    df: pd.DataFrame, snapshots: pd.Index, temporal: dict
) -> pd.DataFrame:
    """
    Aggregate an hourly frame onto the network's (typically coarser) snapshots.

    Mirrors `set_temporal_aggregation`'s own operator rather than guessing one, because which
    operator is correct depends on the configured method:

    - `averaging` and `segmentation` group every fine timestamp onto the closest previous
      snapshot and take the **mean**. Grouping rather than `resample(freq)` is what makes this
      right for segmentation's variable-length segments, and for a `drop_leap_day` index whose
      Feb 29 gap has no single frequency to infer.
    - `representative` keeps `n.snapshots[::value]`, i.e. it point-samples. Averaging there
      would NOT match upstream: the coarse value genuinely is the fine value at that hour.

    This used to call `pd.infer_freq` and fall back to an unaveraged reindex when that returned
    None -- which is exactly what a leap year with `enable.drop_leap_day` produces. The fallback
    silently point-sampled the added heat load: only -0.2% on annual energy at 3h, but +5.2% on
    the 3-hourly PEAK, which is the quantity this study turns on. Reading the method from config
    removes the inference entirely. memory: aro-heat-temporal-alignment
    """
    resolution = get_temporal_resolution(temporal)
    if resolution is not None and resolution[0] == "representative":
        return df.reindex(snapshots)

    # `method="ffill"` yields, for each fine timestamp, the position of the closest snapshot at
    # or before it -- the same mapping `set_temporal_aggregation` builds via `get_indexer` plus
    # `ffill`. -1 marks fine timestamps preceding the first snapshot, which belong to no window.
    positions = snapshots.get_indexer(df.index, method="ffill")
    inside = positions >= 0
    grouped = df[inside].groupby(snapshots[positions[inside]]).mean()
    return grouped.reindex(snapshots)


def electric_heat_supply(
    hourly_heat_demand_file: str, pop_weighted_energy_totals: pd.DataFrame
) -> pd.DataFrame:
    """
    Today's already-electrified heat, hourly, per (sector, use) and node.

    Deliberately re-derives what `build_heat_demand` computes internally, because that function
    subtracts it from `n.loads_t.p_set` **in place** and never returns it.

    Upstream can subtract hourly: `add_heat` runs before `set_temporal_aggregation`, so its
    `p_set` is still hourly and the two indices line up. Here the network arrives already
    aggregated (`compose_network` folds the aggregation in), so an hourly subtraction misaligns
    -- pandas widens to the union index and the write-back silently keeps only each window's
    first hour. Because the mean is linear, upstream's answer is just
    `p_coarse - mean_window(supply)`, which is what `__main__` applies instead.

    Re-deriving these three lines is the price of not reaching into upstream's local scope; if
    upstream changes how `electric_heat_supply` is built this diverges silently, which is what
    `test_metered_load_matches_add_heat` exists to catch.
    memory: aro-heat-temporal-alignment
    """
    shape = xr.open_dataset(hourly_heat_demand_file).to_dataframe().unstack(level=1)
    supply = {}
    for sector, use in product([s.value for s in HeatSector], ["water", "space"]):
        name = f"{sector} {use}"
        supply[name] = (shape[name] / shape[name].sum()).multiply(
            pop_weighted_energy_totals[f"electricity {sector} {use}"]
        ) * 1e6
    return pd.concat(supply, axis=1)


def add_heat_load(
    n: pypsa.Network,
    heat_demand: pd.DataFrame,
    cop_profiles_file: str,
    district_heat_share_file: str,
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    params: dict,
    sector_params: dict,
    temporal: dict,
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
    # Aggregate onto the network's snapshots HERE -- before heat is ever combined with COP.
    # That ordering is the whole point: pypsa-eur cannot divide hourly, because how much heat
    # its heat-pump Link serves in each hour is a decision variable. It can only carry 1/COP as
    # a Link `efficiency` parameter, aggregate that, and let the solver multiply. So it computes
    # mean(heat) * mean(1/COP) and we match it exactly, rather than the mean(heat / COP) this
    # script could compute from its exogenous shares. The two differ by the within-window
    # covariance of heat demand and 1/COP -- small (0.02% on totals at 24h averaging) but not
    # zero, and matching upstream is worth more here than the marginal accuracy.
    # memory: aro-heat-cop-aggregation-order
    heat_demand = to_network_resolution(heat_demand, n.snapshots, temporal)

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

        # Invert first, then aggregate: 1/COP is the quantity pypsa-eur stores on the Link
        # (its `efficiency`, per the reversed bus wiring in
        # memory: pypsa-eur-sector-run-gotchas) and therefore the quantity its time aggregation
        # averages. Aggregating COP and inverting afterwards would give the arithmetic mean
        # where upstream has the harmonic mean -- 2.2086 vs 2.2070 on the config this was
        # measured against.
        inv_cop = to_network_resolution(1.0 / curve, n.snapshots, temporal)

        # `ct` is itself just `pop.index.str[:2]` (build_clustered_population_layouts), so this
        # is not a more correct derivation than slicing here -- it routes through the interface
        # five other upstream scripts already use, so a future change to how country is derived
        # is inherited rather than silently diverged from. `.loc` also raises for a node absent
        # from pop_layout, where slicing would invent a country code for it.
        countries = pop_layout.loc[nodes, "ct"]
        hp_share = countries.map(lambda ct: resolve_shares(ct, params)[0])
        res_share = countries.map(lambda ct: resolve_shares(ct, params)[1])

        hp = demand.multiply(hp_share, axis=1).mul(inv_cop[demand.columns])
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
    # Already on the network's snapshots: add_heat_load aggregates heat demand and 1/COP
    # separately, before combining them, to match pypsa-eur's own aggregation order.
    load = load.reindex(n.snapshots).fillna(0.0)
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


def cap_hydrogen_storage(
    n: pypsa.Network,
    h2_cavern_file: str,
    costs: pd.DataFrame,
    options: dict,
) -> None:
    """
    Re-cost the H2 stores as geologically capped caverns or uncapped tanks.

    `add_electricity.attach_stores` gives carrier `H2` the *underground* cost
    (`hydrogen storage underground`) with no `e_nom_max`, at every node. The geological limit
    pypsa-eur does implement (`build_salt_cavern_potentials`) is wired only into
    `prepare_sector_network`, so an electricity-only run gets unlimited salt caverns in
    Bavaria at 1/25th the tank cost.

    Why this is not cosmetic: the store is `e_cyclic`, so a net-zero system builds TWh-scale
    *seasonal* caverns wherever it likes, which makes the NOMINAL system cheaper and more
    resilient and so moves the baseline the resilience premium is measured against -- in both
    numerator and denominator. `dump/todos.md` sec -3 has the full argument; it also notes the
    cap barely moves the outage response itself, where the converter (~206k EUR/MW/a to
    discharge) dominates the store (~16k EUR/MWh/a).

    The >2 TWh floor, the TWh->MWh conversion and the 1000 TWh per-site clip reproduce
    `prepare_sector_network.add_storage_and_grids` rather than importing it: that logic is
    inline in a ~600-line function taking 20+ arguments, not a helper, and this fork keeps its
    changes additive so they re-apply across upstream merges (see AGENTS.md). Everything else
    IS reused -- the rule, the dataset, and the two `sector.hydrogen_underground_storage*`
    config keys, which `validate_config` accepts with `sector.enabled: false`.

    Setting `sector.hydrogen_underground_storage: false` puts every node on tanks, which is
    the H2-vs-H2-tank sensitivity `dump/todos.md` sec -3 leaves open -- a deliberate scenario,
    not a broken network, which is why this is gated on an existing meaningful flag rather
    than on a new enable/disable toggle whose "off" position would be knowingly wrong.

    Matched deliberately to upstream: a cavern node gets NO tank alternative, so it is hard
    capped at its own potential. Harmless at DE-8, where the smallest retained potential
    (116 TWh) is ~90x that node's annual electricity demand, but it would bind on a network
    clustered finely enough to isolate a small-potential site.

    Verified against upstream on de-heat-8 with the default options: identical cavern/tank
    node sets and identical `e_nom_max` to the last decimal.

    ONE DELIBERATE DIVERGENCE, in the `hydrogen_underground_storage: false` branch. Upstream
    rebinds `h2_caverns` to the filtered Series only *inside* its `if`, then computes
    `nodes_overground = h2_caverns.index.symmetric_difference(nodes)` outside it -- so with the
    flag off, `h2_caverns` is still the raw DataFrame and its index is the CSV's node list.
    The symmetric difference then yields only the nodes ABSENT from the CSV, and every node
    the CSV does list gets no H2 store at all: 7 of 8 on de-heat-8, silently. Here every node
    gets a tank instead, which is what the flag is meant to express. `symmetric_difference`
    also puts a store on a non-existent bus if the CSV names a node the network lacks; keying
    off the network's own stores rather than the CSV's index makes that unreachable.
    Locked in by `test_cavern_split_disabled_gives_all_tanks` -- do not "restore parity" here.
    """
    stores = n.stores.index[n.stores.carrier == H2_CARRIER]
    if stores.empty:
        logger.info(
            f"No '{H2_CARRIER}' stores in the network -- nothing to re-cost. This is expected "
            "only if electricity.extendable_carriers.Store omits H2."
        )
        return

    # Store -> AC node via the H2 bus's `location`, which `attach_stores` sets. Going through
    # `location` rather than stripping a " H2" suffix keeps this independent of how
    # `attach_stores` happens to name buses and stores.
    node = n.stores.loc[stores, "bus"].map(n.buses["location"])

    caverns = pd.read_csv(h2_cavern_file, index_col=0)
    cavern_types = [
        c
        for c in options["hydrogen_underground_storage_locations"]
        if c in caverns.columns
    ]
    if options["hydrogen_underground_storage"] and not caverns.empty and cavern_types:
        potential = caverns[cavern_types].sum(axis=1)
        potential = potential[potential > CAVERN_MIN_POTENTIAL_TWH] * 1e6  # TWh -> MWh
        potential = potential.clip(upper=CAVERN_MAX_PER_SITE_MWH)
    else:
        potential = pd.Series(dtype=float)
        logger.info(
            "Hydrogen underground storage disabled or unavailable "
            f"(hydrogen_underground_storage={options['hydrogen_underground_storage']}, "
            f"matched cavern types={cavern_types}) -- every node gets tank storage."
        )

    is_cavern = node.isin(potential.index)
    for mask, tech, cap in (
        (is_cavern, CAVERN_TECH, node[is_cavern].map(potential)),
        (~is_cavern, TANK_TECH, None),
    ):
        names = stores[mask.to_numpy()]
        if names.empty:
            continue
        n.stores.loc[names, "capital_cost"] = costs.at[tech, "capital_cost"]
        n.stores.loc[names, "lifetime"] = costs.at[tech, "lifetime"]
        n.stores.loc[names, "e_nom_max"] = (
            float("inf") if cap is None else cap.to_numpy()
        )
        logger.info(
            f"{len(names)} H2 store(s) priced as '{tech}' at "
            f"{costs.at[tech, 'capital_cost']:.1f} EUR/MWh/a"
            + (
                " (uncapped)"
                if cap is None
                else f" (capped at {cap.sum() / 1e6:.0f} TWh total)"
            )
            + f": {', '.join(sorted(node[mask]))}"
        )


def add_h2_turbines(n: pypsa.Network, costs: pd.DataFrame, options: dict) -> None:
    """
    Offer gas turbines burning stored hydrogen alongside the fuel cell.

    `add_electricity.STORE_LOOKUP` hardcodes `fuel cell` as carrier H2's only discharger, so
    an electricity-only network re-electrifies hydrogen only at ~206k EUR/MW_el/a -- which
    the CCGT beats on both axes (126k, 0.60 against 0.50), so the fuel cell is dominated the
    moment anything else is offered. Which turbine wins is not something to assume: the
    marginal unit for a rare multi-day blackout runs at a capacity factor where cheap MW
    beats efficient MW, while a unit sized for seasonal operation may not. Offering both
    leaves that to the optimiser, and the split it picks is a result.

    Both are literature choices, not inventions: the sector-coupled path exposes exactly this
    as `hydrogen_turbine` at OCGT cost, and Brown & Hampp (Joule 2023) re-electrify hydrogen
    through a CCGT. Their code calls that a `hydrogen_turbine` too, which is why these links
    are named for the machine rather than inheriting the ambiguous upstream carrier.

    Cost expressions follow `prepare_sector_network.add_h2_gas_infrastructure`, including its
    open TODO that gas-turbine costs stand in for hydrogen-specific ones. Note `VOM`, not
    `marginal_cost`: the latter includes the gas these rows were costed for.
    """
    technologies = options["technologies"]
    if not technologies:
        return

    missing = [t for t in technologies if t not in costs.index]
    if missing:
        raise ValueError(
            f"aro.h2_turbine.technologies names {missing}, which the cost table does not "
            f"carry. Available gas turbines: {sorted(set(costs.index) & {'OCGT', 'CCGT'})}."
        )

    # Pair on the existing discharger rather than a name-mangled string: its bus0 IS the H2
    # bus and its bus1 the AC bus it feeds, so this is exact whatever the naming convention.
    fuel_cells = n.links.index[n.links.carrier == "H2 Fuel Cell"]
    if fuel_cells.empty:
        logger.warning(
            f"aro.h2_turbine.technologies is {technologies} but the network has no "
            "'H2 Fuel Cell' links, so carrier H2 is not present as a Store. Not adding "
            "hydrogen turbines."
        )
        return

    for tech in technologies:
        carrier = f"H2 {tech}"
        if carrier not in n.carriers.index:
            n.add("Carrier", [carrier], color=CARRIER_COLORS.get(carrier, "#999999"))

        n.add(
            "Link",
            n.links.bus0[fuel_cells].values,
            suffix=f" {tech}",
            bus0=n.links.bus0[fuel_cells].values,
            bus1=n.links.bus1[fuel_cells].values,
            carrier=carrier,
            p_nom_extendable=True,
            efficiency=costs.at[tech, "efficiency"],
            # NB: these costs are per MW_el while p_nom sits on bus0, the H2 side.
            capital_cost=costs.at[tech, "capital_cost"] * costs.at[tech, "efficiency"],
            marginal_cost=costs.at[tech, "VOM"] * costs.at[tech, "efficiency"],
            lifetime=costs.at[tech, "lifetime"],
        )
        logger.info(
            f"Added {len(fuel_cells)} {carrier} link(s) at "
            f"{costs.at[tech, 'capital_cost']:.0f} EUR/MW_el/a and efficiency "
            f"{costs.at[tech, 'efficiency']:.2f}, against the fuel cell's "
            f"{costs.at['fuel cell', 'capital_cost']:.0f} at "
            f"{costs.at['fuel cell', 'efficiency']:.2f}."
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_aro_network",
            horizon="2050",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params = snakemake.params.aro_heat
    temporal = snakemake.params.clustering_temporal
    n = pypsa.Network(snakemake.input.network)

    # Since #1838 one `composed_{horizon}.nc` covers both network kinds and only
    # `sector.enabled` distinguishes them, so pointing this rule at a sector-coupled network is
    # a plausible mistake with no natural signal: `add_heat` would already have subtracted the
    # baseline and built heat buses, and this script would subtract it a second time and add a
    # second, exogenous copy of the same demand on the AC buses. Check the network rather than
    # the config, so a network built under a different config is caught too.
    heat_buses = n.buses.index[n.buses.carrier.str.contains("heat", na=False)]
    if len(heat_buses):
        raise ValueError(
            f"{snakemake.input.network} already carries {len(heat_buses)} heat buses, so it was "
            "composed with sector.enabled: true. prepare_aro_network layers heat onto an "
            "ELECTRICITY-ONLY network; running it here would double-count heat demand silently. "
            "Point this rule at a run composed with sector.enabled: false."
        )

    # Deliberately OUTSIDE the aro.heat branch, so the no-heat baseline and the heat run
    # differ in exactly one thing. Gated on `sector.hydrogen_underground_storage` rather than
    # on `aro.heat.enable`, which is about a different sector entirely.
    # processed costs are wide: technology index, parameter columns.
    costs = pd.read_csv(snakemake.input.costs, index_col=0)
    cap_hydrogen_storage(n, snakemake.input.h2_cavern, costs, snakemake.params.sector)
    add_h2_turbines(n, costs, snakemake.params.aro_h2_turbine)

    if not params["enable"]:
        logger.info(
            "aro.heat.enable is false -- writing the network through with no heat load."
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

        # Scale the *annual* totals down to the modelled period before they reach
        # build_heat_demand, exactly as prepare_sector_network.main() does. build_heat_demand
        # normalises by the profile's own sum (`shape / shape.sum()`), so it spreads whatever
        # total it is handed across the modelled snapshots regardless of how long they are.
        # Omitting this put a full year of heat into a one-week run -- 52x too much, which
        # drove every metered electricity load negative via the subtraction below. Only
        # invisible on a full-year run, where nyears == 1.
        # memory: aro-heat-nyears-scaling
        nyears = n.snapshot_weightings.objective.sum() / 8760.0
        pop_weighted_energy_totals = (
            pd.read_csv(snakemake.input.pop_weighted_energy_totals, index_col=0)
            * nyears
        )
        # `prepare_sector_network` overwrites the space-heating columns with heat-specific
        # totals before calling build_heat_demand. Skipping this does not just rescale demand,
        # it redistributes it between nodes -- verified: with the update the allocation
        # reproduces add_heat's heat loads exactly (ratio 1.0000 per node); without it the
        # per-node ratio scatters between 0.42 and 1.17.
        pop_weighted_energy_totals.update(
            pd.read_csv(snakemake.input.pop_weighted_heat_totals, index_col=0) * nyears
        )
        year = int(snakemake.params.energy_totals_year)
        heating_efficiencies = pd.read_csv(
            snakemake.input.heating_efficiencies, index_col=[1, 0]
        ).loc[year]
        pop_layout = pd.read_csv(snakemake.input.pop_layout, index_col=0)

        electric_nodes = n.loads.index[n.loads.carrier == ELECTRICITY_CARRIER]
        metered = n.loads_t.p_set[electric_nodes].copy()

        # Called for its return value only. It ALSO subtracts today's already-electrified heat
        # from `n.loads_t.p_set` in place, but at hourly resolution against an index this
        # network no longer has, so that write is discarded and redone below. See
        # `electric_heat_supply` for why it cannot simply be reused in place.
        heat_demand = build_heat_demand(
            n,
            snakemake.input.hourly_heat_demand_total,
            pop_weighted_energy_totals,
            heating_efficiencies,
        )

        # Subtracting today's electric heat is unconditional by design: skipping it
        # double-counts, so it is logged rather than exposed as a toggle whose "off" position is
        # simply wrong. What IS conditional is the window operator, which must match whatever
        # `set_temporal_aggregation` used -- see `to_network_resolution`.
        baseline = to_network_resolution(
            electric_heat_supply(
                snakemake.input.hourly_heat_demand_total, pop_weighted_energy_totals
            )
            .T.groupby(level=1)
            .sum()
            .T,
            n.snapshots,
            temporal,
        )
        # Assigning `metered - baseline` in one step both discards build_heat_demand's
        # misaligned write and applies the correct one.
        n.loads_t.p_set[electric_nodes] = metered - baseline[electric_nodes]

        weight = n.snapshot_weightings.objective.iloc[0]
        subtracted = baseline[electric_nodes].sum().sum() * weight
        logger.info(
            f"Subtracted existing electric heating from the metered load: "
            f"{subtracted / 1e6:.1f} TWh/a."
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
            temporal,
            int(snakemake.wildcards.horizon),
        )

        n.export_to_netcdf(snakemake.output.network)
