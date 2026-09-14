# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Regression test for `prepare_aro_network`.

Checks that the exogenous heat load it puts on the AC buses is exactly what `add_heat` would
have served, rather than merely that the rule runs. That distinction matters: every bug found
while developing this script (a missing `pop_weighted_heat_totals` update, a missing space-heat
reduction, a missing district-heating loss, a hardcoded resistive efficiency) produced a network
that built cleanly and was silently wrong.

Ground truth is the heat `Load` components in the *prepared* sector-coupled network, which no
solve is needed to produce. Expected electricity is reconstructed from those loads using the
same COP profiles and cost table the script reads, so the comparison is independent of the
script's own arithmetic rather than a restatement of it.

Requires both networks to exist:

    snakemake -call prepare_aro_networks prepare_sector_networks
        --configfile config/test/config.prepare_aro.yaml
"""

import os

import pandas as pd
import pypsa
import pytest
import xarray as xr
import yaml

RUN = "test-prepare-aro"
CLUSTERS = 5
HORIZON = 2030
RESOURCES = f"resources/{RUN}"

# Must match config/test/config.prepare_aro.yaml.
HP_SHARE = 0.85
RESISTIVE_SHARE = 0.05
SOURCES = {"rural": "ground", "urban decentral": "air", "urban central": "air"}
SYSTEMS = ["rural", "urban decentral", "urban central"]

# Floating point only -- the two sides do the same multiplications in a different order.
TOL = 1e-9


def _require(path):
    if not os.path.exists(path):
        pytest.skip(
            f"{path} not built; run the snakemake command in this module's docstring"
        )
    return path


@pytest.fixture(scope="module")
def networks():
    aro = pypsa.Network(
        _require(f"{RESOURCES}/networks/base_s_{CLUSTERS}_elec__aro_{HORIZON}.nc")
    )
    sector = pypsa.Network(
        _require(f"{RESOURCES}/networks/base_s_{CLUSTERS}___{HORIZON}.nc")
    )
    return aro, sector


def _truth_heat(sector, system):
    """Heat demand `add_heat` placed on one heat system's buses, keyed by node."""
    loads = sector.loads[sector.loads.carrier == f"{system} heat"]
    p_set = sector.get_switchable_as_dense("Load", "p_set")[loads.index]
    return p_set.rename(
        columns={
            i: sector.loads.at[i, "bus"].replace(f" {system} heat", "")
            for i in loads.index
        }
    )


def _emitted(aro, system, technology):
    loads = aro.loads[aro.loads.carrier == f"{system} {technology} electricity"]
    if loads.empty:
        return pd.DataFrame(index=aro.snapshots)
    return aro.loads_t.p_set[loads.index].rename(
        columns={i: aro.loads.at[i, "bus"] for i in loads.index}
    )


def test_carriers_present(networks):
    """All six heat carriers are added, on AC buses, with no NaN."""
    aro, _ = networks
    for system in SYSTEMS:
        for technology in ("heat pump", "resistive heater"):
            carrier = f"{system} {technology} electricity"
            loads = aro.loads[aro.loads.carrier == carrier]
            if loads.empty:
                continue  # legitimately dropped when the system's weight is zero everywhere
            assert aro.buses.loc[loads.bus, "carrier"].eq("AC").all(), (
                f"{carrier} is not on AC buses"
            )
            assert not aro.loads_t.p_set[loads.index].isna().any().any()
            assert carrier in aro.carriers.index, (
                f"{carrier} not registered as a Carrier"
            )


def test_heat_load_matches_add_heat(networks):
    """The emitted electricity equals heat demand / COP, reconstructed independently."""
    aro, sector = networks
    cop_ds = xr.open_dataset(f"{RESOURCES}/cop_profiles_base_s_{CLUSTERS}_{HORIZON}.nc")
    cop = cop_ds[next(iter(cop_ds.data_vars))]
    costs = pd.read_csv(f"{RESOURCES}/costs_{HORIZON}_processed.csv", index_col=0)[
        "efficiency"
    ].astype(float)

    checked = 0
    for system in SYSTEMS:
        heat = _truth_heat(sector, system)
        if heat.empty:
            continue
        curve = cop.sel(heat_system=system, heat_source=SOURCES[system]).to_pandas()
        central = "central" if system == "urban central" else "decentral"
        resistive_efficiency = costs.at[f"{central} resistive heater"]

        # Reconstruct expected electricity from ground-truth heat, independently of the
        # script's own arithmetic: heat pumps divide by COP, resistive by a flat efficiency.
        for technology, expected_full in (
            ("heat pump", heat.div(curve.reindex(heat.index)[heat.columns]) * HP_SHARE),
            ("resistive heater", heat * RESISTIVE_SHARE / resistive_efficiency),
        ):
            expected = expected_full.reindex(aro.snapshots)
            got = _emitted(aro, system, technology)
            if got.empty:
                assert expected.abs().max().max() == pytest.approx(0, abs=TOL)
                continue
            cols = got.columns.intersection(expected.columns)
            assert len(cols), f"no overlapping nodes for {system} {technology}"
            deviation = (got[cols] - expected[cols]).abs().max().max()
            scale = expected[cols].abs().max().max()
            assert deviation <= TOL * max(scale, 1.0), (
                f"{system} {technology}: max deviation {deviation:.3e} "
                f"(scale {scale:.3e}) exceeds tolerance"
            )
            checked += len(cols)

    assert checked > 0, "no node/technology pairs were compared"


def test_config_shares_are_applied(networks):
    """Belgium's explicit override, not the default, drives the emitted load."""
    aro, sector = networks
    cfg = yaml.safe_load(open("config/test/config.prepare_aro.yaml"))
    shares = cfg["aro"]["heat"]["shares"]["BE"]
    assert shares["heat_pump"] == HP_SHARE and shares["resistive"] == RESISTIVE_SHARE, (
        "test constants have drifted from config/test/config.prepare_aro.yaml"
    )
    assert aro.loads.index.str.startswith("BE").any()
