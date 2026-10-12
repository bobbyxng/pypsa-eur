# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""Compare the ARO adapter against the actual sector-network methanol helpers."""

from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

from scripts.prepare_aro_network import (
    METHANOL_MWH_PER_M3,
    TANK_TECHNOLOGY,
    add_methanol,
)
from scripts.prepare_sector_network import add_carrier_buses, add_methanol_to_power


@pytest.fixture
def networks():
    """Build both technology menus from the same processed cost table."""
    costs = pd.DataFrame(
        {
            "capital_cost": [30.0, 57000.0, 126000.0, 0.0],
            "lifetime": [30.0, 25.0, 25.0, 30.0],
            "efficiency": [1.0, 0.43, 0.60, 1.0],
            "VOM": [0.0, 6.0, 5.0, 0.0],
            "carbondioxide-input": [0.0, 0.0, 0.0, 0.248],
        },
        index=[TANK_TECHNOLOGY, "OCGT", "CCGT", "methanolisation"],
    )
    nodes = pd.Index(["a", "b"])
    upstream = pypsa.Network()
    upstream.add("Carrier", "AC")
    upstream.add("Bus", nodes, carrier="AC")
    adapted = upstream.copy()
    spatial = SimpleNamespace(
        methanol=SimpleNamespace(nodes=["EU methanol"], locations=["EU"])
    )
    upstream.add("Bus", "co2 atmosphere")
    add_carrier_buses(upstream, "methanol", costs, spatial, {"fossil_fuels": False})
    add_methanol_to_power(
        upstream,
        costs,
        pd.DataFrame(index=nodes),
        spatial,
        {},
        types={"allam": False, "ocgt": True, "ccgt": True, "ccgt_cc": False},
    )
    add_methanol(adapted, costs, {"enable": True, "fuel_price": 100.0})
    return upstream, adapted


def test_tanks_match_upstream_storage_physics(networks):
    """Local tanks retain upstream's investment and intertemporal semantics."""
    upstream, adapted = networks
    tank = upstream.stores.loc["EU methanol Store"]
    for attribute in (
        "e_nom_extendable",
        "e_cyclic",
        "e_nom_min",
        "e_nom_max",
        "e_initial",
        "e_min_pu",
        "e_max_pu",
        "standing_loss",
        "marginal_cost",
    ):
        assert (adapted.stores[attribute] == tank[attribute]).all(), attribute
    # IRENA's rounded volumetric LHV differs slightly from the sector helper's
    # mass-specific LHV times density. This is the only tank COST difference.
    upstream_mwh_per_m3 = 5.54 * 791 * 1e-3
    expected = tank.capital_cost * upstream_mwh_per_m3 / METHANOL_MWH_PER_M3
    assert adapted.stores.capital_cost.tolist() == pytest.approx([expected, expected])
    assert adapted.stores.lifetime.tolist() == [30.0, 30.0]
    assert set(adapted.stores.bus) == {"a methanol", "b methanol"}


@pytest.mark.parametrize("technology", ["OCGT", "CCGT"])
def test_turbine_costs_and_efficiency_match_upstream(networks, technology):
    """Only fuel locality and the explicit external-neutrality boundary differ."""
    upstream, adapted = networks
    for node in ("a", "b"):
        name = f"{node} {technology} methanol"
        for attribute in (
            "p_nom_extendable",
            "capital_cost",
            "marginal_cost",
            "efficiency",
            "lifetime",
            "p_min_pu",
            "p_max_pu",
        ):
            assert (
                adapted.links.at[name, attribute] == upstream.links.at[name, attribute]
            )
        assert adapted.links.at[name, "bus0"] == f"{node} methanol"
        assert upstream.links.at[name, "bus0"] == "EU methanol"
        assert adapted.links.at[name, "bus1"] == upstream.links.at[name, "bus1"]
        assert upstream.links.at[name, "bus2"] == "co2 atmosphere"
        assert "bus2" not in adapted.links or adapted.links.at[name, "bus2"] == ""
