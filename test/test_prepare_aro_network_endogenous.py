# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Regression test for `prepare_aro_network` with `aro.heat.mode: endogenous`.

Ground truth is again the sector-coupled twin build: `add_heat`'s own heat loads and its own
heat-pump, resistive-heater and water-tank components. The endogenous variant must serve the
SAME heat demand (rural + urban decentral merged into `decentral`) with components priced and
wired exactly like upstream's, so any difference between the exogenous and endogenous ARO
results comes from the modelling choice, not from a drifted input.

    snakemake -call prepare_aro_networks --configfile config/test/config.prepare_aro_endogenous.yaml
    snakemake -call compose_networks --configfile config/test/config.prepare_aro_sector.yaml
"""

import os

import pandas as pd
import pypsa
import pytest

HORIZON = 2030
RESOURCES = "resources/test-prepare-aro-endogenous"
SECTOR_RESOURCES = "resources/test-prepare-aro-sector"
TOL = 1e-9
SYSTEMS = {
    "decentral": ["rural", "urban decentral"],
    "urban central": ["urban central"],
}
# Upstream component whose parameters the endogenous one must reproduce.
UPSTREAM = {"decentral": "urban decentral", "urban central": "urban central"}


def _require(path):
    if not os.path.exists(path):
        pytest.skip(
            f"{path} not built; run the snakemake commands in this module's docstring"
        )
    return path


@pytest.fixture(scope="module")
def networks():
    aro = pypsa.Network(_require(f"{RESOURCES}/networks/composed_aro_{HORIZON}.nc"))
    sector = pypsa.Network(
        _require(f"{SECTOR_RESOURCES}/networks/composed_{HORIZON}.nc")
    )
    return aro, sector


def _by_node(n, component, carrier, attr):
    """Static or time-varying attribute of one carrier's components, keyed by location node."""
    c = n.components[component]
    idx = c.static.index[c.static.carrier == carrier]
    bus = "bus" if component in ("Load", "Store", "Generator") else "bus0"
    loc = c.static.loc[idx, bus].map(n.buses.location)
    dynamic = c.dynamic.get(attr)
    if dynamic is not None and idx.isin(dynamic.columns).any():
        frame = n.get_switchable_as_dense(component, attr)[idx]
        return frame.rename(columns=loc.to_dict())
    return c.static.loc[idx, attr].rename(index=loc.to_dict())


def test_heat_demand_matches_add_heat(networks):
    aro, sector = networks
    for system, members in SYSTEMS.items():
        got = _by_node(aro, "Load", f"{system} heat", "p_set")
        truth = sum(_by_node(sector, "Load", f"{m} heat", "p_set") for m in members)
        truth = truth.loc[:, truth.sum() > 0]
        pd.testing.assert_frame_equal(
            got.sort_index(axis=1),
            truth.reindex(columns=got.columns).sort_index(axis=1),
            rtol=TOL,
            atol=TOL,
            check_names=False,
        )


def test_technology_set(networks):
    aro, _ = networks
    touches_heat = aro.links[["bus0", "bus1", "bus2"]].apply(
        lambda col: col.fillna("").str.contains("heat|water tanks")
    )
    heat_links = set(aro.links.carrier[touches_heat.any(axis=1)])
    assert heat_links == {
        f"{s} {t}"
        for s in SYSTEMS
        for t in (
            "air heat pump",
            "resistive heater",
            "water tanks charger",
            "water tanks discharger",
        )
    } | {"urban central H2 CHP"}
    assert not aro.links.carrier.str.contains("ground").any()
    assert not aro.links.carrier.str.contains("boiler").any()


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_heat_pump_matches_upstream(networks, system):
    aro, sector = networks
    up = UPSTREAM[system]
    for attr in ("capital_cost", "lifetime", "p_max_pu"):
        got = _by_node(aro, "Link", f"{system} air heat pump", attr)
        truth = _by_node(sector, "Link", f"{up} air heat pump", attr)
        pd.testing.assert_series_equal(
            got.sort_index(), truth.reindex(got.index).sort_index(), check_names=False
        )
    # 1/COP, aggregated onto the snapshots: rural and urban-decentral air curves are identical
    # upstream, so the merged decentral curve must equal upstream's urban-decentral one.
    got = _by_node(aro, "Link", f"{system} air heat pump", "efficiency")
    truth = _by_node(sector, "Link", f"{up} air heat pump", "efficiency")
    pd.testing.assert_frame_equal(
        got.sort_index(axis=1),
        truth.reindex(columns=got.columns).sort_index(axis=1),
        rtol=TOL,
        atol=TOL,
        check_names=False,
    )


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize(
    "component, tech, attrs",
    [
        ("Link", "resistive heater", ("efficiency", "capital_cost", "lifetime")),
        (
            "Link",
            "water tanks charger",
            ("efficiency", "marginal_cost", "energy to power ratio"),
        ),
        ("Link", "water tanks discharger", ("efficiency",)),
        (
            "Store",
            "water tanks",
            ("capital_cost", "standing_loss", "e_cyclic", "lifetime"),
        ),
    ],
)
def test_components_match_upstream(networks, system, component, tech, attrs):
    aro, sector = networks
    for attr in attrs:
        got = _by_node(aro, component, f"{system} {tech}", attr)
        truth = _by_node(sector, component, f"{UPSTREAM[system]} {tech}", attr)
        assert not got.empty
        pd.testing.assert_series_equal(
            got.sort_index(), truth.reindex(got.index).sort_index(), check_names=False
        )


def test_h2_chp(networks):
    aro, _ = networks
    chp = aro.links[aro.links.carrier == "urban central H2 CHP"]
    assert not chp.empty
    assert (aro.buses.loc[chp.bus0, "carrier"] == "H2").all()
    assert (aro.buses.loc[chp.bus1, "carrier"] == "AC").all()
    assert (aro.buses.loc[chp.bus2, "carrier"] == "urban central heat").all()
    # Only at nodes that have district heating.
    central = set(aro.buses.location[aro.buses.carrier == "urban central heat"])
    assert set(aro.buses.loc[chp.bus1].index) <= central
    assert (chp.efficiency2 > 0).all() and chp.p_nom_extendable.all()


def test_exogenous_heat_loads_absent(networks):
    aro, _ = networks
    assert (
        not aro.loads.carrier.str.endswith("electricity")
        .drop(aro.loads.index[aro.loads.carrier == "electricity"])
        .any()
    )
