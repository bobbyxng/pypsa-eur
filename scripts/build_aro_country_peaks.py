# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Coincident peak per country (electricity + electrical heat load) that sizes the outage-island
partition: k per country and the electricity/heat weights of its load grid.

Built from raw inputs, the way the pre-run builds its load, on regions independent of the
islands (`aro.partition.peak_regions`, NUTS2 by default):

- electricity: the hourly series per country (`electricity_demand.csv`) x `load.scaling_factor`,
  averaged onto the run's snapshots as `compose_network` does;
- heat: upstream's heat chain run on those regions (`build_aro_peaks_*` rules), turned into
  heat-pump and resistive-heater electricity with `prepare_aro_network`'s own functions (heat
  shares, 1/COP, district heating and its losses, space-heat reduction, resistive
  efficiencies), and today's electric heating subtracted from the electricity load.

Nothing is reimplemented, so the result is the pre-run's load summed per country. What depends
on the regions is the heat-pump electricity: heat demand x 1/COP is summed over regions, and
cold regions have both more heat and a lower COP. One region per country therefore understates
the peak where climate varies within a country (vs the NUTS1/2-clustered europe-pre-run: FI
-10%, GB -7%, GR -5%; GB dropped from k = 8 to 7). On NUTS2 every country is within 2% of
that pre-run and every k is the same (2026-11-16); NUTS2 is also close to the island resolution
of the pre-run that uses the partition.
The peak is the 3-hourly (`clustering.temporal`) maximum of the sum, both parts reported at
that snapshot.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    get_temporal_resolution,
    set_scenario_config,
)
from scripts.prepare_aro_network import (
    ELECTRICITY_CARRIER,
    add_heat_load,
    electric_heat_supply,
    heat_system_inputs,
    to_network_resolution,
)

logger = logging.getLogger(__name__)


def model_snapshots(
    hourly: pd.DatetimeIndex, temporal: dict, drop_leap_day: bool
) -> pd.Series:
    """Snapshots and their weights as `time_aggregation` builds them (averaging only)."""
    resolution = get_temporal_resolution(temporal)
    ones = pd.Series(1.0, index=hourly)
    if resolution is None:
        return ones
    method, offset = resolution
    if method != "averaging":
        raise NotImplementedError(
            f"clustering.temporal {method}: only averaging (or none) is supported here"
        )
    w = pd.concat(
        [ones[hourly.year == y].resample(offset).sum() for y in hourly.year.unique()]
    )
    w = w[w > 0]
    leap = w.index[(w.index.month == 2) & (w.index.day == 29)]
    if drop_leap_day and len(leap):
        for year in leap.year.unique():
            w.loc[pd.Timestamp(year, 3, 1)] = w.loc[leap[leap.year == year]].sum()
        w = w.drop(leap).sort_index()
    return w


def country_peaks(load: pd.DataFrame) -> pd.DataFrame:
    """
    Electricity and electrical heat load (GW) per country at its coincident peak.

    `load` has columns (part, country), part in {"electricity", "heat"}, in MW.
    """
    el, ht = (
        load["electricity"],
        load["heat"].reindex(columns=load["electricity"].columns),
    )
    ht = ht.fillna(0.0)
    hour = (el + ht).idxmax()
    return pd.DataFrame(
        {
            "electricity_GW": [el.at[hour[c], c] / 1e3 for c in el.columns],
            "heat_GW": [ht.at[hour[c], c] / 1e3 for c in el.columns],
            "peak_snapshot": hour.values,
        },
        index=pd.Index(el.columns, name="country"),
    ).round({"electricity_GW": 3, "heat_GW": 3})


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_aro_country_peaks")
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    params = snakemake.params
    temporal = params.clustering_temporal
    countries = list(params.countries)

    hourly = get_snapshots(params.snapshots, params.drop_leap_day)
    weights = model_snapshots(hourly, temporal, params.drop_leap_day)

    el = pd.read_csv(snakemake.input.electricity_demand, index_col=0, parse_dates=True)
    missing = sorted(set(countries) - set(el.columns))
    if missing:
        raise ValueError(f"no electricity demand for {missing}")
    el = el.reindex(hourly)[countries] * params.load_scaling
    if el.isna().any().any():
        raise ValueError("electricity demand has gaps within the snapshots")

    pop_layout = pd.read_csv(snakemake.input.pop_layout, index_col=0)
    regions = pop_layout.index
    region_country = pop_layout["ct"]

    # one bus per region, named like the heat chain's nodes (pop_layout's index), so
    # prepare_aro_network's functions line up; the electricity load stays per country
    n = pypsa.Network()
    n.set_snapshots(weights.index)
    n.snapshot_weightings.loc[:, :] = np.repeat(weights.values[:, None], 3, axis=1)
    n.add("Bus", regions, country=region_country.values, carrier="AC")

    # as prepare_aro_network's __main__ (memory: aro-heat-nyears-scaling)
    nyears = n.snapshot_weightings.objective.sum() / 8760.0
    pwet = pd.read_csv(snakemake.input.pop_weighted_energy_totals, index_col=0) * nyears
    pwet.update(
        pd.read_csv(snakemake.input.pop_weighted_heat_totals, index_col=0) * nyears
    )
    heating_efficiencies = pd.read_csv(
        snakemake.input.heating_efficiencies, index_col=[1, 0]
    ).loc[int(params.energy_totals_year)]
    costs = pd.read_csv(snakemake.input.costs, index_col=0)

    from scripts.prepare_sector_network import build_heat_demand

    # no electricity Loads on n yet, so build_heat_demand's in-place subtraction is a no-op;
    # it is applied per country below, as prepare_aro_network applies it per node
    heat_demand = build_heat_demand(
        n, snakemake.input.hourly_heat_demand_total, pwet, heating_efficiencies
    )
    baseline = to_network_resolution(
        electric_heat_supply(snakemake.input.hourly_heat_demand_total, pwet)
        .T.groupby(level=1)
        .sum()
        .T,
        n.snapshots,
        temporal,
    )
    electricity = to_network_resolution(el, n.snapshots, temporal) - baseline.T.groupby(
        region_country
    ).sum().T.reindex(columns=countries, fill_value=0.0)

    systems = heat_system_inputs(
        n,
        heat_demand,
        snakemake.input.cop_profiles,
        snakemake.input.district_heat_share,
        pop_layout,
        params.aro_heat["heat_pump_sources"],
        params.sector,
        temporal,
        int(params.horizon),
    )
    add_heat_load(n, systems, pop_layout, costs, params.aro_heat)
    heat_loads = n.loads.index[n.loads.carrier != ELECTRICITY_CARRIER]
    heat = (
        n.loads_t.p_set[heat_loads]
        .T.groupby(n.loads.bus[heat_loads].map(region_country))
        .sum()
        .T.reindex(columns=countries, fill_value=0.0)
    )

    load = pd.concat({"electricity": electricity, "heat": heat}, axis=1)
    peaks = country_peaks(load)
    peaks.to_csv(snakemake.output.peaks)
    logger.info(
        "coincident peaks (GW):\n"
        + peaks.assign(peak_GW=peaks.electricity_GW + peaks.heat_GW)
        .sort_values("peak_GW", ascending=False)
        .to_string()
    )
