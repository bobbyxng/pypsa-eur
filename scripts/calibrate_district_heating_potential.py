# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Calibrate `sector: district_heating: potential` to Fallahnejad et al. (2024).

Fork-specific; not part of upstream PyPSA-Eur. Consumes `retrieve_fallahnejad_dh` and
writes a JSON whose `potential` block is meant to be **copied by hand** into the config.
Deliberately not wired into the DAG: the config stays a static, reviewable artifact, and
no ordinary run depends on a 400 MB download.

Why a calibration is needed at all: PyPSA-Eur's default `potential: 0.6` is a uniform cap
on *urban* heat demand, so the only thing varying it between countries is urbanisation.
That inverts reality -- the Netherlands (4% district heating today) ends up above Poland
(19% today, with the actual networks). Fallahnejad's per-country values instead come from
GIS heat-density mapping with distribution-cost ceilings and explicit connection rates.

Three things this has to get right, each of which produced a wrong answer first:

1. **The denominator is the raster, not `demand_end`.** The summary's `demand_end` column
   counts only demand *inside* prospective DH areas, so `dhPot_2050 / demand_end` is the
   connection rate (0.70-0.90), not the market share. Read that way the EU total comes out
   at 76% instead of 31%.
2. **The urban fraction must be heat-weighted.** The model applies `potential x
   urban_fraction` node by node, so the country's realised share is the heat-weighted mean
   of those. An unweighted node mean puts GB at 0.41 instead of 0.85 and would deliver
   roughly twice the intended British share.
3. **Invert the model's formula, don't divide.** `build_district_heat_share` floors each
   node at today's share and caps it at `urban_fraction`, so a plain division can silently
   clip. Bisecting on the realised share handles both and reports which countries floor
   (Denmark does: its target sits below today's share, so its value is inert -- see the
   `floored` branch for why it is still reported as the implied value, not 0).

Outputs
-------
- `resources/{run}/dh_potential_calibrated_{horizon}.json`: `potential` per country, the
  `floored` and `missing` country lists, and a `validation` block.

Notes
-----
The `validation` block reproduces three figures published in the paper (2020 demand
3128 TWh, 2050 demand 1709 TWh, 2050 district-heating share 31%). They are the guard
rail: if an upstream change to `pop_layout` or the energy totals ever shifts the
derivation, these stop matching and the JSON says so rather than failing silently.

Overrides are deliberately *not* applied here -- countries outside the study's EU-27
scope, and any country whose published value is a model artifact rather than a
projection, are judgement calls that belong in the config where a reader can see and
challenge them. This script only reports them in `missing`.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from scripts._helpers import configure_logging, get, set_scenario_config

logger = logging.getLogger(__name__)

SECTORS = ("residential", "services")
USES = ("water", "space")
# Must match the scenario `retrieve_fallahnejad_dh` fetched; named here only for the
# diagnostic raised when the validation anchors fail.
SCENARIO_NAME = "RES-H Best Case"
# Retrieved because they have demand rasters, but outside the paper's EU-27 scope, so they
# must be excluded when checking the totals against the published figures.
NON_EU27 = {"GB", "IS", "LI"}


def raster_sum(path: Path) -> float:
    """Total demand in a national raster, in TWh, summed blockwise in float64."""
    with rasterio.open(path) as src:
        total = 0.0
        for _, window in src.block_windows(1):
            block = src.read(1, window=window).astype("float64")
            if src.nodata is not None:
                block = np.where(block == src.nodata, 0.0, block)
            total += float(np.nansum(block))
    # Cells are MWh/a, so 1e6 converts to TWh. Confirmed by the validation block: the
    # EU-27 totals come out at 3128.8 and 1709.1 TWh against the paper's 3128 and 1709.
    return total / 1e6


def annual_heat_per_node(
    totals: pd.DataFrame, efficiencies: pd.DataFrame, space_reduction: float
) -> pd.Series:
    """
    Annual useful heat demand per node, mirroring `build_heat_demand`.

    Only the annual total is needed, never the hourly profile: `build_heat_demand`
    normalises the shape by its own sum, so the profile cancels out of the yearly figure.
    """
    heat = pd.Series(0.0, index=totals.index)
    for sector in SECTORS:
        for use in USES:
            efficiency = totals.index.str[:2].map(
                efficiencies[f"total {sector} {use} efficiency"]
            )
            demand = totals[f"total {sector} {use}"] * efficiency
            if use == "space":
                demand = demand * (1 - space_reduction)
            heat = heat + demand
    return heat


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "calibrate_district_heating_potential", horizon="2050"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    horizon = int(snakemake.wildcards.horizon)
    sector_params = snakemake.params.sector
    data = Path(snakemake.input.fallahnejad).parent

    # --- PyPSA-Eur side: nodal heat demand, urban fraction, today's district heat -------
    pop_layout = pd.read_csv(snakemake.input.pop_layout, index_col=0)
    ct = pop_layout["ct"]

    totals = pd.read_csv(snakemake.input.pop_weighted_energy_totals, index_col=0)
    totals.update(pd.read_csv(snakemake.input.pop_weighted_heat_totals, index_col=0))
    year = int(snakemake.params.energy_totals_year)
    efficiencies = pd.read_csv(
        snakemake.input.heating_efficiencies, index_col=[1, 0]
    ).loc[year]
    space_reduction = (
        get(sector_params["reduce_space_heat_exogenously_factor"], horizon)
        if sector_params["reduce_space_heat_exogenously"]
        else 0.0
    )
    heat = annual_heat_per_node(totals, efficiencies, space_reduction)

    # `build_district_heat_share`'s own construction, reproduced so the inversion below
    # solves the formula the model will actually apply.
    national_share = pd.read_csv(snakemake.input.district_heat_share, index_col=0)[
        str(year)
    ]
    nodal_share = national_share.reindex(ct).fillna(0)
    nodal_share.index = pop_layout.index
    ct_urban = pop_layout.urban.groupby(ct).sum()
    urban_ct_fraction = pop_layout.urban / ct.map(ct_urban.get)
    urban_fraction = pop_layout.urban / pop_layout[["rural", "urban"]].sum(axis=1)
    today = nodal_share * urban_ct_fraction / pop_layout["fraction"]
    progress = get(sector_params["district_heating"]["progress"], horizon)

    def realised_share(country: str, potential: float) -> float:
        """District-heating share of that country's total heat at a given `potential`."""
        nodes = ct.index[ct == country]
        capped = pd.concat([urban_fraction[nodes], today[nodes]], axis=1).max(axis=1)
        fraction = (
            today[nodes] + (capped * potential - today[nodes]).clip(lower=0) * progress
        )
        return float((fraction * heat[nodes]).sum() / heat[nodes].sum())

    def heat_weighted_urban_fraction(country: str) -> float:
        """The constant linking `potential` to a country's realised share of total heat."""
        nodes = ct.index[ct == country]
        return float((urban_fraction[nodes] * heat[nodes]).sum() / heat[nodes].sum())

    def solve(
        country: str, target: float, tol: float = 1e-6
    ) -> tuple[float | None, float]:
        """Smallest `potential` reproducing `target`; None if even 1.0 falls short."""
        if realised_share(country, 1.0) < target - tol:
            return None, realised_share(country, 1.0)
        if realised_share(country, 0.0) > target + tol:
            # Today's share already exceeds the target, and `build_district_heat_share` never
            # goes below today, so no `potential` reaches it -- the country is pinned at
            # today's share for anything up to today_share/urban_fraction.
            #
            # Report the value the target *implies* rather than 0.0. Both give an identical
            # result while the floor stands, but 0.0 reads as "no district-heating potential"
            # (absurd for Denmark, which leads Europe) and would send the country to *zero*
            # district heating if that floor ever changed upstream. This degrades gracefully.
            return (
                round(target / heat_weighted_urban_fraction(country), 3),
                realised_share(country, 0.0),
            )
        low, high = 0.0, 1.0
        for _ in range(60):
            mid = (low + high) / 2
            if realised_share(country, mid) < target:
                low = mid
            else:
                high = mid
        return round(high, 3), realised_share(country, high)

    # --- Fallahnejad side: potential over national demand -------------------------------
    potential_twh = {
        path.stem: pd.read_csv(path)["dhPot_2050 [GWh]"].sum() / 1e3
        for path in sorted((data / "summaries").glob("*.csv"))
    }
    demand_twh = {
        year_: {
            path.stem.split("_")[0]: raster_sum(path)
            for path in sorted((data / "demand").glob(f"*_{year_}.tif"))
        }
        for year_ in ("2020", "2050")
    }

    eu27 = [c for c in demand_twh["2050"] if c not in NON_EU27]
    validation = {
        "demand_2020_TWh": round(sum(demand_twh["2020"].values()), 1),
        "demand_2050_TWh_eu27": round(sum(demand_twh["2050"][c] for c in eu27), 1),
        "dh_share_2050_eu27": round(
            sum(potential_twh.values()) / sum(demand_twh["2050"][c] for c in eu27), 4
        ),
        "published": {
            "demand_2020_TWh": 3128,
            "demand_2050_TWh_eu27": 1709,
            "dh_share_2050_eu27": 0.31,
        },
    }
    logger.info(
        f"Validation against Fallahnejad et al. (2024): "
        f"2020 demand {validation['demand_2020_TWh']} TWh (published 3128), "
        f"2050 demand {validation['demand_2050_TWh_eu27']} TWh (published 1709), "
        f"2050 DH share {validation['dh_share_2050_eu27']:.2%} (published 31%)"
    )

    # These anchors are computed purely from the retrieved archive -- they do not touch
    # pop_layout or the energy totals -- so they are stable across clusterings and config
    # changes, and a hard failure here always means the *source reading* is wrong. The
    # three ways it can be wrong all show up loudly: taking `demand_end` (demand inside DH
    # areas) as the denominator puts the share at ~76%, picking the `sEEnergies BL2050`
    # tree puts it at ~41%, and an incomplete retrieval drags the totals down. Reporting
    # these without asserting them would let a plausible-looking but wrong `potential`
    # block reach the config, which is exactly the failure this script exists to prevent.
    # 2% absorbs the published figures' rounding (31% is two significant figures) while
    # still catching every mistake above by a wide margin.
    deviations = {
        key: abs(validation[key] - validation["published"][key])
        / validation["published"][key]
        for key in validation["published"]
    }
    off = {key: dev for key, dev in deviations.items() if dev > 0.02}
    if off:
        raise ValueError(
            "Calibration does not reproduce Fallahnejad et al. (2024): "
            + ", ".join(
                f"{key} = {validation[key]} vs published "
                f"{validation['published'][key]} ({dev:.1%} off)"
                for key, dev in off.items()
            )
            + ". Check that the denominator is the national demand raster (not the "
            "summaries' `demand_end`, which counts only demand inside district-heating "
            f"areas), that only the '{SCENARIO_NAME}' scenario was retrieved, and that "
            "the retrieval completed."
        )
    validation["max_deviation"] = round(max(deviations.values()), 4)

    # --- calibrate ----------------------------------------------------------------------
    modelled = sorted(ct.unique())
    calibrated, floored, achieved = {}, [], {}
    for country in modelled:
        if country not in potential_twh or country not in demand_twh["2050"]:
            continue
        target = potential_twh[country] / demand_twh["2050"][country]
        value, delivered = solve(country, target)
        if value is None:
            logger.warning(
                f"{country}: target share {target:.1%} is unreachable -- even "
                f"potential 1.0 delivers only {delivered:.1%} (urban fraction too low)."
            )
            continue
        calibrated[country] = value
        achieved[country] = round(delivered, 4)
        if abs(delivered - target) > 1e-6:
            floored.append(country)

    missing = [c for c in modelled if c not in calibrated]
    if floored:
        logger.info(
            f"Floored at today's share (target below it, so `potential` is inactive): "
            f"{floored}"
        )
    if missing:
        logger.warning(
            f"No published value for {missing} -- these need an explicit entry or a "
            "`default` in the config; this script does not invent one."
        )

    Path(snakemake.output.potential).write_text(
        json.dumps(
            {
                "source": "doi:10.5281/zenodo.7455894",
                "reference": "Fallahnejad et al. (2024), Applied Energy 353, 122154",
                "scenario": "RES-H Best Case",
                "horizon": horizon,
                "nodes": len(pop_layout),
                "validation": validation,
                "potential": calibrated,
                "achieved_share_of_total_heat": achieved,
                "floored": floored,
                "missing": missing,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(
        f"Calibrated {len(calibrated)} countries; copy the `potential` block into "
        "`sector: district_heating: potential`."
    )
