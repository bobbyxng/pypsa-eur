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

Ground truth is the heat `Load` components in the *composed* sector-coupled network, which no
solve is needed to produce. Expected electricity is reconstructed from those loads using the
same COP profiles and cost table the script reads, so the comparison is independent of the
script's own arithmetic rather than a restatement of it.

Since upstream's streamlined workflow (#1838) there is one `composed_{horizon}.nc` per run and
`sector.enabled` decides what is in it, so the two sides come from two runs -- an
electricity-only one that `prepare_aro_network` actually operates on, and a sector-coupled
twin that exists only to produce the reference heat loads:

    snakemake -call prepare_aro_networks --configfile config/test/config.prepare_aro.yaml
    snakemake -call compose_networks --configfile config/test/config.prepare_aro_sector.yaml
"""

import os

import pandas as pd
import pypsa
import pytest
import xarray as xr
import yaml

RUN = "test-prepare-aro"
SECTOR_RUN = f"{RUN}-sector"
HORIZON = 2030
RESOURCES = f"resources/{RUN}"
SECTOR_RESOURCES = f"resources/{SECTOR_RUN}"

# Must match config/test/config.prepare_aro.yaml.
HP_SHARE = 0.85
RESISTIVE_SHARE = 0.05
SOURCES = {"rural": "ground", "urban decentral": "air", "urban central": "air"}
SYSTEMS = ["rural", "urban decentral", "urban central"]

# Floating point only -- the two sides do the same multiplications in a different order.
TOL = 1e-9

# Heat pumps match to TOL as well, and are held there deliberately. The script aggregates heat
# demand and 1/COP separately and multiplies at network resolution -- pypsa-eur's own order,
# which it has no choice about (the hourly heat its Link serves is a decision variable, so COP
# can only be an aggregated parameter). Doing the division hourly instead would be marginally
# more accurate but would leave a covariance gap of ~2.4e-3 against `add_heat`; exact agreement
# was chosen over that. memory: aro-heat-cop-aggregation-order
#
# So a heat-pump deviation above TOL now means the two have genuinely diverged -- most likely
# that one side aggregates COP where the other aggregates 1/COP (arithmetic vs harmonic mean,
# ~7e-4), or that the aggregation moved back across the multiplication.
HP_TOL = TOL
HP_ENERGY_TOL = TOL


def _to_snapshots(df, snapshots):
    """
    Mean-aggregate an hourly frame onto the network's snapshots.

    Deliberately written out here rather than imported from `prepare_aro_network`, so this file
    stays an independent check of the script's arithmetic rather than a restatement of it.

    Groups on the snapshot index itself instead of `pd.infer_freq`, which returns None for a
    `drop_leap_day` year and used to send both this test and the script down an unaveraged,
    point-sampling path. memory: aro-heat-temporal-alignment
    """
    positions = snapshots.get_indexer(df.index, method="ffill")
    inside = positions >= 0
    return df[inside].groupby(snapshots[positions[inside]]).mean().reindex(snapshots)


def _reconstructed_baseline(snapshots, nyears, aggregate):
    """
    Today's already-electrified heat per node, from the same raw inputs the script reads.

    `aggregate` selects the window operator, so the same reconstruction can produce both the
    correct (mean) and the formerly-applied (point-sampled) baseline.
    """
    totals = (
        pd.read_csv(f"{RESOURCES}/pop_weighted_energy_totals.csv", index_col=0) * nyears
    )
    totals.update(
        pd.read_csv(f"{RESOURCES}/pop_weighted_heat_totals.csv", index_col=0) * nyears
    )
    shape = (
        xr.open_dataset(f"{RESOURCES}/hourly_heat_demand_total.nc")
        .to_dataframe()
        .unstack(level=1)
    )
    supply = {}
    for sector in ("residential", "services"):
        for use in ("water", "space"):
            name = f"{sector} {use}"
            supply[name] = (shape[name] / shape[name].sum()).multiply(
                totals[f"electricity {sector} {use}"]
            ) * 1e6
    per_node = pd.concat(supply, axis=1).T.groupby(level=1).sum().T
    return aggregate(per_node, snapshots)


def _require(path):
    if not os.path.exists(path):
        pytest.skip(
            f"{path} not built; run the snakemake command in this module's docstring"
        )
    return path


@pytest.fixture(scope="module")
def networks():
    aro = pypsa.Network(_require(f"{RESOURCES}/networks/composed_aro_{HORIZON}.nc"))
    sector = pypsa.Network(
        _require(f"{SECTOR_RESOURCES}/networks/composed_{HORIZON}.nc")
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
    cop_ds = xr.open_dataset(f"{RESOURCES}/cop_profiles_{HORIZON}.nc")
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

        # Aggregate 1/COP onto the network's snapshots, which is what pypsa-eur itself stores
        # on the heat-pump Link (its `efficiency` is 1/COP, per the reversed bus wiring in
        # memory: pypsa-eur-sector-run-gotchas) and what its time aggregation averages.
        # Point-sampling COP at each snapshot instead -- which this test used to do -- reads
        # the value at midnight rather than the day's mean, understating COP by ~2.5% and
        # inflating expected electricity to match.
        inv_cop = _to_snapshots(1.0 / curve, aro.snapshots)

        # Reconstruct expected electricity from ground-truth heat, independently of the
        # script's own arithmetic: heat pumps multiply by 1/COP, resistive by a flat efficiency.
        for technology, expected_full, tol in (
            ("heat pump", heat.mul(inv_cop[heat.columns]) * HP_SHARE, HP_TOL),
            ("resistive heater", heat * RESISTIVE_SHARE / resistive_efficiency, TOL),
        ):
            expected = expected_full.reindex(aro.snapshots)
            got = _emitted(aro, system, technology)
            if got.empty:
                assert expected.abs().max().max() == pytest.approx(0, abs=tol)
                continue
            cols = got.columns.intersection(expected.columns)
            assert len(cols), f"no overlapping nodes for {system} {technology}"
            deviation = (got[cols] - expected[cols]).abs().max().max()
            scale = expected[cols].abs().max().max()
            assert deviation <= tol * max(scale, 1.0), (
                f"{system} {technology}: max deviation {deviation:.3e} "
                f"(scale {scale:.3e}) exceeds tolerance {tol:g}"
            )
            # Energy over the horizon, where the aggregation covariance cancels out. A
            # scaling error (the class of bug this test exists for) shows up here even if
            # it somehow slipped past the per-snapshot bound.
            got_energy, expected_energy = (
                got[cols].values.sum(),
                expected[cols].values.sum(),
            )
            energy_tol = TOL if technology == "resistive heater" else HP_ENERGY_TOL
            assert got_energy == pytest.approx(expected_energy, rel=energy_tol), (
                f"{system} {technology}: total energy {got_energy:.6e} != "
                f"{expected_energy:.6e} (rel tol {energy_tol:g})"
            )
            checked += len(cols)

    assert checked > 0, "no node/technology pairs were compared"


def test_metered_load_matches_add_heat(networks):
    """
    The baseline subtraction uses the network's own window operator, not an hourly point sample.

    `build_heat_demand` subtracts today's already-electrified heat from `n.loads_t.p_set` in
    place, hourly. That is correct upstream -- `add_heat` runs before
    `set_temporal_aggregation`, so p_set is still hourly and the indices line up.
    `prepare_aro_network` receives an already-aggregated network, so the same call misaligns:
    pandas widens to the union index and the write-back silently keeps only each window's first
    hour. Every other test here looks only at the heat loads, so this went unnoticed.
    memory: aro-heat-temporal-alignment

    Ground truth is deliberately NOT the sector run, unlike the heat-load tests above. The
    sector run applies two further scalings that an electricity-only network must not have --
    a distribution-loss deduction in `insert_electricity_distribution_grid` and the removal of
    today's industrial electricity, both in `prepare_sector_network` -- which together make its
    metered loads ~1.8x smaller here and legitimately incomparable. The right reference is the
    *pre-ARO* composed network from this same run: the subtraction is the only change
    `prepare_aro_network` makes to the metered load, so the difference between the two files
    isolates it exactly.
    """
    aro, _ = networks
    composed = pypsa.Network(_require(f"{RESOURCES}/networks/composed_{HORIZON}.nc"))
    nodes = aro.loads.index[aro.loads.carrier == "electricity"]
    assert len(nodes), "no electricity loads in the ARO network"
    assert nodes.isin(composed.loads.index).all(), (
        "ARO network has electricity loads absent from its own input network"
    )

    nyears = aro.snapshot_weightings.objective.sum() / 8760.0
    mean_baseline = _reconstructed_baseline(aro.snapshots, nyears, _to_snapshots)
    point_baseline = _reconstructed_baseline(
        aro.snapshots, nyears, lambda df, snapshots: df.reindex(snapshots)
    )

    # Guard against a vacuous pass: the two candidate window operators must actually be
    # distinguishable on this fixture, or the assertion below proves nothing. At 24h averaging
    # they differ by ~20% of the baseline peak.
    scale = mean_baseline.abs().max().max()
    assert scale > 0, (
        "reconstructed baseline is all zero; fixture has no electric heating"
    )
    spread = (mean_baseline - point_baseline).abs().max().max()
    assert spread > 0.05 * scale, (
        f"mean and point-sampled baselines differ by only {spread / scale:.2%} on this "
        "fixture, so this test cannot detect the aggregation bug it exists for"
    )

    got = aro.loads_t.p_set[nodes]
    base = composed.loads_t.p_set[nodes]
    expected = base - mean_baseline[nodes]
    deviation = (got - expected).abs().max().max()
    assert deviation <= TOL * max(base.abs().max().max(), 1.0), (
        f"metered load deviates from (input load - mean-aggregated baseline) by "
        f"{deviation:.3e} MW. The mean/point spread is {spread:.3e} MW, so a deviation of "
        "that order means the subtraction is point-sampling the hourly profile again."
    )


def test_config_shares_are_applied(networks):
    """Belgium's explicit override, not the default, drives the emitted load."""
    aro, sector = networks
    cfg = yaml.safe_load(open("config/test/config.prepare_aro.yaml"))
    shares = cfg["aro"]["heat"]["shares"]["BE"]
    assert shares["heat_pump"] == HP_SHARE and shares["resistive"] == RESISTIVE_SHARE, (
        "test constants have drifted from config/test/config.prepare_aro.yaml"
    )
    assert aro.loads.index.str.startswith("BE").any()


# ---------------------------------------------------------------------------------------------
# H2 cavern/tank split (`cap_hydrogen_storage`)
#
# Built on a synthetic network rather than the fixtures above, so these run without either
# snakemake build. The logic under test is a re-expression of upstream's inline cavern filter
# in `prepare_sector_network.add_storage_and_grids`, and the thresholds (>2 TWh floor, 1000 TWh
# clip) are the part most likely to drift if upstream changes them.
# ---------------------------------------------------------------------------------------------

CAVERN_COSTS = pd.DataFrame(
    {"capital_cost": [112.4, 2797.5], "lifetime": [100.0, 30.0]},
    index=[
        "hydrogen storage underground",
        "hydrogen storage tank type 1 including compressor",
    ],
)
CAVERN_OPTIONS = {
    "hydrogen_underground_storage": True,
    "hydrogen_underground_storage_locations": ["onshore", "nearshore"],
}


def _h2_network(nodes):
    """Minimal stand-in for what `add_electricity.attach_stores` leaves behind."""
    n = pypsa.Network()
    n.add("Bus", nodes, carrier="AC")
    n.add("Bus", [f"{b} H2" for b in nodes], location=nodes, carrier="H2")
    n.add(
        "Store",
        [f"{b} H2" for b in nodes],
        bus=[f"{b} H2" for b in nodes],
        carrier="H2",
        e_nom_extendable=True,
        capital_cost=112.4,
        lifetime=100.0,
    )
    n.add(
        "Link",
        [f"{b} H2 Fuel Cell" for b in nodes],
        bus0=[f"{b} H2" for b in nodes],
        bus1=nodes,
        carrier="H2 Fuel Cell",
        p_nom_extendable=True,
    )
    return n


@pytest.fixture
def cavern_csv(tmp_path):
    """Potentials in TWh: one clipped, one ordinary, one below the floor, one absent."""
    path = tmp_path / "salt_cavern_potentials.csv"
    pd.DataFrame(
        {
            "nearshore": [1200.0, 10.0, 0.5],
            "offshore": [
                9999.0,
                9999.0,
                9999.0,
            ],  # never summed: not in the locations list
            "onshore": [300.0, 15.0, 0.4],
        },
        index=["N0", "N1", "N2"],
    ).rename_axis("name").to_csv(path)
    return str(path)


def test_cavern_split_costs_and_caps(cavern_csv):
    """Above the floor -> capped cavern; below or absent -> uncapped tank."""
    from scripts.prepare_aro_network import cap_hydrogen_storage

    n = _h2_network(["N0", "N1", "N2", "N3"])
    cap_hydrogen_storage(n, cavern_csv, CAVERN_COSTS, CAVERN_OPTIONS)
    s = n.stores

    # N0: 1200 + 300 = 1500 TWh, clipped to the 1000 TWh per-site limit. N1: 25 TWh, kept whole.
    assert s.at["N0 H2", "e_nom_max"] == pytest.approx(1e9)
    assert s.at["N1 H2", "e_nom_max"] == pytest.approx(25e6)
    for node in ["N0", "N1"]:
        assert s.at[f"{node} H2", "capital_cost"] == pytest.approx(112.4)
        assert s.at[f"{node} H2", "lifetime"] == pytest.approx(100.0)

    # N2 is 0.9 TWh, below the 2 TWh floor; N3 is absent from the dataset entirely.
    for node in ["N2", "N3"]:
        assert s.at[f"{node} H2", "e_nom_max"] == float("inf")
        assert s.at[f"{node} H2", "capital_cost"] == pytest.approx(2797.5)
        assert s.at[f"{node} H2", "lifetime"] == pytest.approx(30.0)


def test_cavern_split_offshore_excluded(cavern_csv):
    """Only the configured locations are summed; offshore alone must not qualify a node."""
    from scripts.prepare_aro_network import cap_hydrogen_storage

    n = _h2_network(["N2"])
    cap_hydrogen_storage(n, cavern_csv, CAVERN_COSTS, CAVERN_OPTIONS)
    assert n.stores.at["N2 H2", "e_nom_max"] == float("inf"), (
        "a 9999 TWh offshore potential qualified a node whose onshore+nearshore is 0.9 TWh"
    )


def test_cavern_split_disabled_gives_all_tanks(cavern_csv):
    """
    `hydrogen_underground_storage: false` is the all-tank sensitivity, not a no-op.

    Deliberately NOT upstream's behaviour: `prepare_sector_network` rebinds `h2_caverns` to
    the filtered Series only inside its `if`, so with the flag off it takes the symmetric
    difference against the raw CSV index and leaves every node the CSV lists with no H2 store
    at all (7 of 8 on de-heat-8). This asserts the intended reading of the flag -- see
    `cap_hydrogen_storage`'s docstring before changing it back.
    """
    from scripts.prepare_aro_network import cap_hydrogen_storage

    n = _h2_network(["N0", "N1"])
    cap_hydrogen_storage(
        n,
        cavern_csv,
        CAVERN_COSTS,
        {**CAVERN_OPTIONS, "hydrogen_underground_storage": False},
    )
    assert n.stores.capital_cost.tolist() == pytest.approx([2797.5, 2797.5])
    assert (n.stores.e_nom_max == float("inf")).all()


def test_cavern_split_without_h2_stores_is_a_noop(cavern_csv):
    """A network built with `Store: [battery]` must pass through, not raise."""
    from scripts.prepare_aro_network import cap_hydrogen_storage

    n = pypsa.Network()
    n.add("Bus", ["N0"], carrier="AC")
    cap_hydrogen_storage(n, cavern_csv, CAVERN_COSTS, CAVERN_OPTIONS)
    assert n.stores.empty


H2_TURBINE_COSTS = pd.DataFrame(
    {
        "capital_cost": [57123.0, 126432.0, 205583.0],
        "efficiency": [0.43, 0.60, 0.50],
        "VOM": [6.0, 5.3, 0.0],
        "lifetime": [25.0, 25.0, 10.0],
    },
    index=["OCGT", "CCGT", "fuel cell"],
)


def test_h2_turbines_mirror_the_fuel_cell():
    """One link per technology per fuel cell, same buses, that row's cost on the H2 side."""
    from scripts.prepare_aro_network import add_h2_turbines

    n = _h2_network(["N0", "N1"])
    add_h2_turbines(n, H2_TURBINE_COSTS, {"technologies": ["OCGT", "CCGT"]})

    fc = n.links[n.links.carrier == "H2 Fuel Cell"]
    for tech, capex, eff in [("OCGT", 57123.0, 0.43), ("CCGT", 126432.0, 0.60)]:
        t = n.links[n.links.carrier == f"H2 {tech}"]
        assert len(t) == len(fc) == 2
        assert sorted(zip(t.bus0, t.bus1)) == sorted(zip(fc.bus0, fc.bus1))
        assert t.p_nom_extendable.all()
        assert t.efficiency.eq(eff).all()
        # p_nom sits on bus0, so the per-MW_el cost is capital_cost / efficiency.
        assert (t.capital_cost / t.efficiency).round(0).eq(capex).all()


def test_h2_turbines_empty_or_without_h2_is_a_noop():
    """No technologies, or no H2 discharger to pair with, leaves the network alone."""
    from scripts.prepare_aro_network import add_h2_turbines

    n = _h2_network(["N0"])
    add_h2_turbines(n, H2_TURBINE_COSTS, {"technologies": []})
    assert n.links.carrier.eq("H2 Fuel Cell").all()

    m = pypsa.Network()
    m.add("Bus", ["N0"], carrier="AC")
    add_h2_turbines(m, H2_TURBINE_COSTS, {"technologies": ["OCGT"]})
    assert m.links.empty


def test_h2_turbines_reject_unknown_technology():
    """A technology the cost table lacks must fail loud, not silently add nothing."""
    from scripts.prepare_aro_network import add_h2_turbines

    n = _h2_network(["N0"])
    with pytest.raises(ValueError, match="cost table does not carry"):
        add_h2_turbines(n, H2_TURBINE_COSTS, {"technologies": ["OCGT", "Allam"]})
