# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Prepare an electricity-only network for the ARO/C&CG workflow.

Independent layers, fork-specific additions to upstream PyPSA-Eur:

1. building heat (`aro.heat.enable`), in one of two modes (`aro.heat.mode`):
   - `exogenous` (default): fixed heat-pump/resistive electricity Loads on the existing AC
     buses, split by country shares (described below);
   - `endogenous`: heat buses (`decentral`, `urban central`) with heat Loads served by
     optimised air heat pumps, resistive heaters, water tanks and, in district heating, an
     H2 CHP -- mirroring `add_heat` (see `add_endogenous_heat`);
2. the H2 cavern-vs-tank split that `attach_stores` omits on the electricity path
   (`cap_hydrogen_storage`, gated on `sector.hydrogen_underground_storage`);
3. optional H2 turbines (`aro.h2_turbine.technologies`);
4. optional purchased renewable methanol, local tanks and turbines (`aro.methanol`).

(2) runs unconditionally so the no-heat baseline and the heat run differ in exactly one thing.

Produces the network that PyPSARO's ARO/C&CG resilience workflow consumes, so that PyPSARO
itself carries no content assumptions -- it only points `run.network` at this rule's output.

In exogenous mode heat is added as *loads on the existing AC buses*, not as a modelled heat
sector: no heat buses,
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

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from scripts.definitions.heat_sector import HeatSector
from scripts.definitions.heat_system import HeatSystem

# Import the workflow/sector builders only at their call sites below: standalone
# investment helpers are also tested in PyPSARO's leaner environment, which does
# not install the full geospatial and powerplantmatching preparation stack.
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

# IRENA / Methanol Institute (2021), Table 10, p. 57: 15.8 GJ/m3 LHV.
# Keep all inventories and fuel flows on the LHV basis; costs below arrive already
# annualised AND scaled to the composed network's modelled horizon.
METHANOL_MWH_PER_M3 = 15.8 / 3.6
TANK_TECHNOLOGY = "General liquid hydrocarbon storage (product)"


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
    from scripts._helpers import get_temporal_resolution

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


def heat_system_inputs(
    n: pypsa.Network,
    heat_demand: pd.DataFrame,
    cop_profiles_file: str,
    district_heat_share_file: str,
    pop_layout: pd.DataFrame,
    heat_pump_sources: dict,
    sector_params: dict,
    temporal: dict,
    investment_year: int,
) -> list[tuple[HeatSystem, pd.DataFrame, pd.DataFrame]]:
    """
    Heat demand and heat-pump 1/COP per `HeatSystem`, both at the network's resolution.

    Shared by the exogenous (`add_heat_load`) and endogenous (`add_endogenous_heat`) heat
    variants, so both read the identical demand split: same space reduction, same
    district-heating weighting and losses, same COP curve and aggregation order. Returns one
    `(heat_system, demand, inv_cop)` per system with nodes; `demand` and `inv_cop` are
    time x node frames over the same nodes.
    """
    from scripts._helpers import get

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

    systems = []
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
        source = heat_pump_sources[group]
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
        systems.append((heat_system, demand, inv_cop))
    return systems


def add_heat_load(
    n: pypsa.Network,
    systems: list[tuple[HeatSystem, pd.DataFrame, pd.DataFrame]],
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    params: dict,
) -> None:
    """Convert heat demand into electricity loads on the AC buses, per heat system."""
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

    for heat_system, demand, inv_cop in systems:
        group = cop_heat_system(heat_system)
        nodes = demand.columns

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


# Endogenous heat: two systems only. `decentral` merges pypsa-eur's `rural` and
# `urban decentral` (both individual heating); `urban central` keeps pypsa-eur's name. Cost keys
# are pypsa-eur's `decentral ...` / `central ...` rows either way.
DECENTRAL = "decentral"
CENTRAL = "urban central"
AIR_SOURCES = {"rural": "air", "urban decentral": "air", "urban central": "air"}
# H2-ready engine/turbine CHP for district heating, priced with the gas CHP row: pypsa-eur has
# no H2 CHP, and its unused "central hydrogen CHP" row is a copy of the fuel cell.
H2_CHP_PROXY = "central gas CHP"
ENDOGENOUS_COLORS = {
    "decentral heat": "#cc1f1f",
    "decentral air heat pump": "#7bccc4",
    "decentral resistive heater": "#fc8d59",
    # Charger and discharger get distinct shades: both show up in the heat balance, one as a
    # withdrawal and one as a supply, and a shared colour made them indistinguishable.
    "decentral water tanks": "#e69487",
    "decentral water tanks charger": "#f4c7bf",
    "decentral water tanks discharger": "#c0574a",
    "decentral heat vent": "#aaaaaa",
    "urban central heat": "#9d0208",
    "urban central air heat pump": "#084081",
    "urban central resistive heater": "#d7301f",
    "urban central water tanks": "#f7b7a3",
    "urban central water tanks charger": "#fde0d6",
    "urban central water tanks discharger": "#b5523b",
    "urban central heat vent": "#aaaaaa",
    "urban central H2 CHP": "#d9b8d3",
    # Glaum's min-MeOH-economy colour for the same carrier.
    "urban central methanol CHP": "#A17666",
}


def merge_heat_systems(
    systems: list[tuple[HeatSystem, pd.DataFrame, pd.DataFrame]],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, str]]:
    """
    Collapse pypsa-eur's five heat systems into `decentral` and `urban central`.

    Demand is summed. The merged 1/COP is the heat-weighted mean of the members' curves, which
    keeps the heat-pump electricity exact if every member's heat went through one heat pump;
    pypsa-eur's rural and urban-decentral air curves are identical anyway, so in practice it
    is just that curve. Returns `{system: (demand, inv_cop, "central"|"decentral")}`.
    """
    grouped: dict[str, list] = {}
    for heat_system, demand, inv_cop in systems:
        key = CENTRAL if heat_system == HeatSystem.URBAN_CENTRAL else DECENTRAL
        grouped.setdefault(key, []).append((demand, inv_cop))

    merged = {}
    for key, members in grouped.items():
        nodes = members[0][0].columns
        for demand, _ in members[1:]:
            nodes = nodes.union(demand.columns)
        demand = sum(d.reindex(columns=nodes, fill_value=0.0) for d, _ in members)
        weighted = sum(
            (d * c).reindex(columns=nodes, fill_value=0.0) for d, c in members
        )
        plain = sum(c.reindex(columns=nodes) for _, c in members) / len(members)
        # Where a node has no heat in a snapshot the weights vanish; any member curve is then
        # as good as another, and the plain mean keeps the Link efficiency well defined.
        inv_cop = (weighted / demand).where(demand > 0, plain)
        merged[key] = (demand, inv_cop, "central" if key == CENTRAL else "decentral")
    return merged


def add_dc_losses(n: pypsa.Network, efficiencies: dict) -> None:
    """
    Make every DC link lossy and bidirectional, exactly as a sector run does.

    Reuses `prepare_sector_network.lossy_bidirectional_links`: the link becomes forward-only
    with efficiency `efficiency_static * efficiency_per_1000km ** (length / 1000)`, and a
    `-reversed` twin (capital_cost 0, `reversed: True`) carries the opposite direction. That
    also repairs the transmission-project links, which `new_links.csv` adds with PyPSA's
    default p_min_pu = 0, i.e. one-directional (memory: bidirectional-links-unsupported).
    The twins' p_nom must be tied to the forward links in the optimisation: PyPSARO's
    `add_bidirectional_link_constraints` does that in the investment LP and the C&CG master.
    """
    if (
        n.links.get("reversed", pd.Series(False, index=n.links.index))
        .fillna(False)
        .any()
    ):
        raise ValueError(
            "Network already has '-reversed' links; splitting again would double them."
        )
    from scripts.prepare_sector_network import lossy_bidirectional_links

    lossy_bidirectional_links(n, "DC", efficiencies)


def add_endogenous_heat(
    n: pypsa.Network,
    systems: list[tuple[HeatSystem, pd.DataFrame, pd.DataFrame]],
    costs: pd.DataFrame,
    options: dict,
    sector_params: dict,
    methanol: dict | None = None,
) -> None:
    """
    Model building heat endogenously: heat buses, loads and the technologies serving them.

    Mirrors `prepare_sector_network.add_heat` (wiring, cost keys, overdimensioning, TES
    convention) for a reduced technology set, so the investment and dispatch of heating are
    decided by the optimiser instead of fixed shares:

    - decentral: air heat pump, resistive heater, water tanks, heat vent;
    - urban central: the same, plus an H2 CHP where the node has an H2 bus, and a methanol
      CHP where `aro.methanol` added a local methanol bus (`_add_methanol_chp`).

    Only electricity and stored H2 can make heat -- no boilers, solar thermal, pits or
    biomass -- matching the fully electrified heat world of the ARO network. Heat buses are
    local, so a transmission outage never disconnects them; only the electricity feeding the
    heat pumps and resistive heaters goes.
    """
    overdim = sector_params["overdimension_heat_generators"]
    vent_cost = sector_params["marginal_cost_heat_vent"]
    # technology-data VOM per MWh of heat, which upstream's add_heat never applies; off by
    # default for parity. memory: heat-variable-costs-degeneracy
    vom = costs["VOM"].fillna(0.0) * float(options.get("variable_costs", False))
    ac_nodes = n.buses.index[n.buses.carrier == "AC"]

    for system, (demand, inv_cop, c_or_d) in merge_heat_systems(systems).items():
        demand = demand.loc[:, demand.sum() > 0]
        nodes = demand.columns
        missing = nodes.difference(ac_nodes)
        if len(missing):
            raise ValueError(
                f"Heat demand for '{system}' at nodes without an AC bus: {list(missing)}."
            )
        if nodes.empty:
            continue
        heat_bus = nodes + f" {system} heat"
        tank_bus = nodes + f" {system} water tanks"
        factor = overdim[c_or_d]

        for carrier in [f"{system} heat", f"{system} water tanks"] + [
            f"{system} {tech}"
            for tech in (
                "air heat pump",
                "resistive heater",
                "water tanks charger",
                "water tanks discharger",
                "heat vent",
            )
        ]:
            if carrier not in n.carriers.index:
                n.add(
                    "Carrier",
                    carrier,
                    color=ENDOGENOUS_COLORS.get(carrier, "#999999"),
                    nice_name=carrier,
                )

        n.add("Bus", heat_bus, location=nodes, carrier=f"{system} heat", unit="MWh_th")
        n.add(
            "Load",
            heat_bus,
            bus=heat_bus,
            carrier=f"{system} heat",
            p_set=demand.reindex(n.snapshots).fillna(0.0).set_axis(heat_bus, axis=1),
        )

        # Reversed wiring as upstream: bus0 is the heat bus and the link runs negative, so
        # p_nom is heat output (MW_th) and |p1| = |p0| / COP is the electricity drawn.
        # memory: pypsa-eur-sector-run-gotchas
        hp_key = f"{c_or_d} air-sourced heat pump"
        n.add(
            "Link",
            nodes + f" {system} air heat pump",
            bus0=heat_bus,
            bus1=nodes,
            carrier=f"{system} air heat pump",
            efficiency=inv_cop.reindex(n.snapshots)[nodes].set_axis(
                nodes + f" {system} air heat pump", axis=1
            ),
            capital_cost=costs.at[hp_key, "capital_cost"] * factor,
            # p0 is heat and runs NEGATIVE (reversed wiring), so the per-MWh_th VOM enters
            # with a minus sign to be a cost.
            marginal_cost=-vom.at[hp_key],
            p_min_pu=-1.0,
            p_max_pu=0.0,
            p_nom_extendable=True,
            lifetime=costs.at[hp_key, "lifetime"],
        )

        res_key = f"{c_or_d} resistive heater"
        n.add(
            "Link",
            nodes + f" {system} resistive heater",
            bus0=nodes,
            bus1=heat_bus,
            carrier=f"{system} resistive heater",
            efficiency=costs.at[res_key, "efficiency"],
            capital_cost=costs.at[res_key, "efficiency"]
            * costs.at[res_key, "capital_cost"]
            * factor,
            # VOM is per MWh_th; p0 is electricity in.
            marginal_cost=vom.at[res_key] * costs.at[res_key, "efficiency"],
            p_nom_extendable=True,
            lifetime=costs.at[res_key, "lifetime"],
        )

        # Water tanks exactly as upstream: own bus, charger/discharger, cost on the Store, and
        # the energy-to-power ratio on the charger, which PyPSARO's
        # `add_tes_energy_to_power_constraints` enforces (and its charger/discharger pairing
        # ties the two links) -- both match on these carrier names.
        tank_key = f"{c_or_d} water tank storage"
        n.add(
            "Bus",
            tank_bus,
            location=nodes,
            carrier=f"{system} water tanks",
            unit="MWh_th",
        )
        n.add(
            "Link",
            nodes + f" {system} water tanks charger",
            bus0=heat_bus,
            bus1=tank_bus,
            carrier=f"{system} water tanks charger",
            efficiency=costs.at[f"{c_or_d} water tank charger", "efficiency"],
            marginal_cost=costs.at["water tank charger", "marginal_cost"],
            p_nom_extendable=True,
            lifetime=costs.at[tank_key, "lifetime"],
        )
        n.add(
            "Link",
            nodes + f" {system} water tanks discharger",
            bus0=tank_bus,
            bus1=heat_bus,
            carrier=f"{system} water tanks discharger",
            efficiency=costs.at[f"{c_or_d} water tank discharger", "efficiency"],
            # The tank's VOM, per MWh leaving the store (p0).
            marginal_cost=vom.at[tank_key],
            p_nom_extendable=True,
            lifetime=costs.at[tank_key, "lifetime"],
        )
        n.links.loc[
            nodes + f" {system} water tanks charger", "energy to power ratio"
        ] = costs.at[tank_key, "energy to power ratio"]
        n.add(
            "Store",
            tank_bus,
            bus=tank_bus,
            carrier=f"{system} water tanks",
            e_cyclic=True,
            e_nom_extendable=True,
            # %/hour in the cost table, a per-unit hourly loss on the Store.
            standing_loss=costs.at[tank_key, "standing_losses"] / 100,
            capital_cost=costs.at[tank_key, "capital_cost"],
            lifetime=costs.at[tank_key, "lifetime"],
        )

        # Dumps surplus heat (full tanks, CHP heat beyond demand) at a small cost, as upstream;
        # without it a must-run CHP heat output could make the outage dispatch infeasible.
        n.add(
            "Generator",
            nodes + f" {system} heat vent",
            bus=heat_bus,
            carrier=f"{system} heat vent",
            p_nom_extendable=True,
            p_max_pu=0.0,
            p_min_pu=-1.0,
            marginal_cost=-vent_cost,
        )

        if system == CENTRAL and options.get("h2_chp", True):
            _add_h2_chp(n, nodes, costs)
        if system == CENTRAL and options.get("methanol_chp", True):
            _add_methanol_chp(n, nodes, costs, methanol or {})

        weight = n.snapshot_weightings.objective.iloc[0]
        logger.info(
            f"Endogenous {system} heat: {len(nodes)} nodes, "
            f"{demand.sum().sum() * weight / 1e6:.1f} TWh_th/a."
        )


def _add_h2_chp(n: pypsa.Network, nodes: pd.Index, costs: pd.DataFrame) -> None:
    """H2 CHP (H2 -> electricity + district heat) at central nodes that have an H2 bus."""
    h2_buses = n.buses[n.buses.carrier == H2_CARRIER]
    h2_of = pd.Series(h2_buses.index, index=h2_buses["location"])
    chp_nodes = nodes.intersection(h2_of.index)
    if chp_nodes.empty:
        logger.warning(
            "aro.heat.endogenous.h2_chp: no urban-central node has an H2 bus."
        )
        return
    carrier = f"{CENTRAL} H2 CHP"
    if carrier not in n.carriers.index:
        n.add("Carrier", carrier, color=ENDOGENOUS_COLORS[carrier], nice_name=carrier)
    eff = costs.at[H2_CHP_PROXY, "efficiency"]
    # Upstream's gas-CHP convention: costs per MW_el, converted to the fuel-input p_nom by the
    # electric efficiency; heat output fixed at eff / c_b (back-pressure, no c_v envelope).
    n.add(
        "Link",
        chp_nodes + f" {carrier}",
        bus0=h2_of.loc[chp_nodes].values,
        bus1=chp_nodes,
        bus2=chp_nodes + f" {CENTRAL} heat",
        carrier=carrier,
        p_nom_extendable=True,
        capital_cost=costs.at[H2_CHP_PROXY, "capital_cost"] * eff,
        marginal_cost=costs.at[H2_CHP_PROXY, "VOM"] * eff,
        efficiency=eff,
        efficiency2=eff / costs.at[H2_CHP_PROXY, "c_b"],
        lifetime=costs.at[H2_CHP_PROXY, "lifetime"],
    )


def _add_methanol_chp(
    n: pypsa.Network, nodes: pd.Index, costs: pd.DataFrame, methanol: dict
) -> None:
    """
    Methanol CHP (methanol -> electricity + district heat) at central nodes with a methanol bus.

    A third consumer of the SAME local tank as the methanol OCGT/CCGT of `add_methanol`, with
    the same access rule: under `aro.methanol.reserve_only` it is blocked in normal operation
    (`p_max_pu = 0`), tagged `reserve_only` so PyPSARO opens it only inside the outage window,
    and its fuel is priced at replacement cost like the turbines'. PyPSARO's fuel-inventory
    constraint counts every link leaving the methanol bus as a consumer, so no engine change.

    Technology data as Glaum et al.'s min-MeOH-economy (`urban central methanol CHP`): the
    `central gas CHP` row for every non-biomass CHP fuel. One deliberate difference: Glaum's
    marginal cost carries no CHP VOM (fuel VOM + distribution only); this keeps upstream
    pypsa-eur's gas-CHP convention, `VOM x efficiency` (VOM is per MWh_el), as the methanol
    turbines here do. No CO2 port: the purchased methanol is externally CO2-neutral, as for the
    turbines (memory: purchased-methanol-backstop); Glaum tracks its carbon instead.
    """
    if not methanol.get("enable", False):
        return
    fuel_buses = n.buses[n.buses.carrier == "methanol"]
    fuel_of = pd.Series(fuel_buses.index, index=fuel_buses["location"])
    chp_nodes = nodes.intersection(fuel_of.index)
    if chp_nodes.empty:
        logger.warning(
            "aro.heat.endogenous.methanol_chp: no urban-central node has a methanol bus."
        )
        return
    reserve_only = methanol.get("reserve_only", False)
    fuel_price = methanol["fuel_price"]
    carrier = f"{CENTRAL} methanol CHP"
    if carrier not in n.carriers.index:
        n.add(
            "Carrier",
            carrier,
            co2_emissions=0,
            color=ENDOGENOUS_COLORS[carrier],
            nice_name=carrier,
        )
    eff = costs.at[H2_CHP_PROXY, "efficiency"]
    names = chp_nodes + f" {carrier}"
    n.add(
        "Link",
        names,
        bus0=fuel_of.loc[chp_nodes].values,
        bus1=chp_nodes,
        bus2=chp_nodes + f" {CENTRAL} heat",
        carrier=carrier,
        p_nom_extendable=True,
        p_max_pu=0.0 if reserve_only else 1.0,
        capital_cost=costs.at[H2_CHP_PROXY, "capital_cost"] * eff,
        marginal_cost=costs.at[H2_CHP_PROXY, "VOM"] * eff
        + (fuel_price if reserve_only else 0.0),
        efficiency=eff,
        efficiency2=eff / costs.at[H2_CHP_PROXY, "c_b"],
        lifetime=costs.at[H2_CHP_PROXY, "lifetime"],
    )
    n.links.loc[names, "reserve_only"] = float(reserve_only)


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


def add_methanol(n: pypsa.Network, costs: pd.DataFrame, options: dict) -> None:
    """
    Add independent tank energy and turbine power investment at every AC bus.

    Tagged local suppliers are disabled by PyPSARO during every outage.
    Reserve-only turbines are enabled only inside the outage window.
    Independent local fuel buses prevent bypassing local inventories. Cyclic Stores
    make every consumed MWh payable; they do not supply a free initial stock. Outage foresight remains unchanged.
    memory: purchased-methanol-backstop
    """
    if not options.get("enable", False):
        return
    # Validated by AroMethanolConfig at the workflow boundary. Keep the network
    # builder independent of the full PyPSA-Eur config registry so the parent ARO
    # environment can exercise it in cross-workflow tests.
    fuel_price = options["fuel_price"]
    technologies = options.get("technologies", ["OCGT", "CCGT"])
    reserve_only = options.get("reserve_only", False)
    nodes = n.buses.index[n.buses.carrier == "AC"]
    if nodes.empty:
        raise ValueError("aro.methanol requires at least one AC bus")
    # A sector network may already carry a pooled methanol/carbon system. Mixing it
    # with externally neutral fuel would silently erase its accounting boundary.
    carriers = ["methanol", "methanol supply"] + [
        f"{tech} methanol" for tech in technologies
    ]
    if n.carriers.index.isin(carriers).any():
        raise ValueError("aro.methanol requires a network without methanol carriers")
    requirements = {TANK_TECHNOLOGY: ["capital_cost", "lifetime"]}
    requirements.update(
        {
            tech: ["capital_cost", "efficiency", "VOM", "lifetime"]
            for tech in technologies
        }
    )
    for tech, columns in requirements.items():
        if tech not in costs.index or not set(columns).issubset(costs.columns):
            raise ValueError(
                f"Missing processed methanol cost data for {tech}: {columns}"
            )
        values = costs.loc[tech, columns].astype(float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or values["lifetime"] <= 0
        ):
            raise ValueError(f"Invalid processed methanol cost data for {tech}")
        if "efficiency" in columns and not 0 < values["efficiency"] <= 1:
            raise ValueError(f"Invalid methanol turbine efficiency for {tech}")

    fuel_buses = nodes + " methanol"
    suppliers = nodes + " methanol supply"
    names = {
        "Bus": fuel_buses,
        "Store": nodes + " methanol tank",
        "Generator": suppliers,
        "Link": pd.Index(
            [f"{node} {tech} methanol" for tech in technologies for node in nodes]
        ),
    }
    for component, added in names.items():
        if not added.intersection(n.components[component].static.index).empty:
            raise ValueError(f"aro.methanol would overwrite existing {component} names")

    # Zero is NET externally assumed CO2, not combustion chemistry. No atmospheric
    # withdrawal/capture credit is added: this layer cannot offset fossil emissions.
    n.add(
        "Carrier",
        carriers,
        co2_emissions=0,
        color=["#d95f02", "#e6ab02"]
        + ["#e7298a" if t == "OCGT" else "#7570b3" for t in technologies],
    )
    n.add(
        "Bus",
        fuel_buses,
        carrier="methanol",
        location=nodes,
        unit="MWh_LHV",
        x=n.buses.loc[nodes, "x"].to_numpy(),
        y=n.buses.loc[nodes, "y"].to_numpy(),
    )
    # A finite normalization avoids inf * 0 during outages. The actual upper
    # dispatch bound is infinite: there is no delivery-capacity investment or cap.
    n.add(
        "Generator",
        suppliers,
        bus=fuel_buses,
        carrier="methanol supply",
        p_nom=1.0,
        p_max_pu=0.0 if reserve_only else np.inf,
        marginal_cost=fuel_price,
    )
    n.generators.loc[suppliers, "outage_supply"] = 1.0
    n.add(
        "Store",
        nodes + " methanol tank",
        bus=fuel_buses,
        carrier="methanol",
        e_nom_extendable=True,
        e_cyclic=not reserve_only,
        capital_cost=costs.at[TANK_TECHNOLOGY, "capital_cost"] / METHANOL_MWH_PER_M3,
        lifetime=costs.at[TANK_TECHNOLOGY, "lifetime"],
    )
    if reserve_only:
        tanks = nodes + " methanol tank"
        n.stores.loc[tanks, "reserve_stock"] = 1.0
        n.stores.loc[tanks, "reserve_inventory_cost"] = (
            fuel_price
            * options.get("reserve_inventory_carrying_rate", 0.07)
            * n.snapshot_weightings.objective.sum()
            / 8760
        )
        n.stores.loc[tanks, "reserve_inventory_fixed"] = 0.0
    for tech in technologies:
        efficiency = costs.at[tech, "efficiency"]
        # As in prepare_sector_network.add_methanol_to_power: p_nom is MW of
        # fuel input, whereas turbine capex and VOM are quoted per electrical MW/MWh.
        # Using the gas row's marginal_cost would buy fossil gas a second time.
        n.add(
            "Link",
            nodes + f" {tech} methanol",
            bus0=fuel_buses,
            bus1=nodes,
            carrier=f"{tech} methanol",
            p_nom_extendable=True,
            p_max_pu=0.0 if reserve_only else 1.0,
            efficiency=efficiency,
            capital_cost=costs.at[tech, "capital_cost"] * efficiency,
            marginal_cost=costs.at[tech, "VOM"] * efficiency
            + (fuel_price if reserve_only else 0.0),
            lifetime=costs.at[tech, "lifetime"],
        )
        n.links.loc[nodes + f" {tech} methanol", "reserve_only"] = float(reserve_only)
    logger.info(
        "Added local CO2-neutral purchased methanol at %s EUR/MWh_LHV to %s nodes; deliveries stop automatically during outages.",
        fuel_price,
        len(nodes),
    )


if __name__ == "__main__":
    from scripts._helpers import configure_logging, set_scenario_config

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
    add_methanol(n, costs, snakemake.params.aro_methanol)
    if snakemake.params.aro_dc_losses:
        add_dc_losses(n, snakemake.params.sector["transmission_efficiency"]["DC"])

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
        from scripts.prepare_sector_network import build_heat_demand

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

        endogenous = params["mode"] == "endogenous"
        systems = heat_system_inputs(
            n,
            heat_demand,
            snakemake.input.cop_profiles,
            snakemake.input.district_heat_share,
            pop_layout,
            # Endogenous heat offers air-source heat pumps only (no ground source).
            AIR_SOURCES if endogenous else params["heat_pump_sources"],
            snakemake.params.sector,
            temporal,
            int(snakemake.wildcards.horizon),
        )
        if endogenous:
            add_endogenous_heat(
                n,
                systems,
                costs,
                params["endogenous"],
                snakemake.params.sector,
                methanol=snakemake.params.aro_methanol,
            )
        else:
            add_heat_load(n, systems, pop_layout, costs, params)

        n.export_to_netcdf(snakemake.output.network)
