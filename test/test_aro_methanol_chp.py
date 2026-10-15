# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
The endogenous-heat methanol CHP (`aro.heat.endogenous.methanol_chp`).

Upstream's generic non-biomass CHP (`add_heat`, `options["chp"]["fuel"]`) prices every fuel
with the `central gas CHP` row; the methanol CHP must reproduce that formula exactly, and use
the methanol tank with the same access rule as the methanol turbines of `add_methanol`.
"""

import pandas as pd
import pypsa
import pytest

from scripts.prepare_aro_network import (
    CENTRAL,
    H2_CHP_PROXY,
    TANK_TECHNOLOGY,
    _add_methanol_chp,
    add_methanol,
)

CARRIER = f"{CENTRAL} methanol CHP"
FUEL_PRICE = 100.0


@pytest.fixture
def costs():
    """Processed cost rows for the tank, the turbines and the gas-CHP proxy."""
    return pd.DataFrame(
        {
            "capital_cost": [30.0, 57000.0, 126000.0, 90000.0],
            "lifetime": [30.0, 25.0, 25.0, 25.0],
            "efficiency": [1.0, 0.43, 0.60, 0.43],
            "VOM": [0.0, 6.0, 5.0, 4.0],
            "c_b": [float("nan")] * 3 + [1.0],
        },
        index=[TANK_TECHNOLOGY, "OCGT", "CCGT", H2_CHP_PROXY],
    )


def _network(costs, methanol):
    """Two AC nodes, district heating only at `a`, methanol layer from `add_methanol`."""
    n = pypsa.Network()
    n.add("Carrier", "AC")
    n.add("Bus", ["a", "b"], carrier="AC", location=["a", "b"], x=0.0, y=0.0)
    n.add("Bus", f"a {CENTRAL} heat", carrier=f"{CENTRAL} heat", location="a")
    add_methanol(n, costs, methanol)
    _add_methanol_chp(n, pd.Index(["a"]), costs, methanol)
    return n


@pytest.mark.parametrize("reserve_only", [False, True])
def test_matches_upstream_generic_chp(costs, reserve_only):
    """Wiring and techno-economics as upstream's `urban central {fuel} CHP`, minus the CO2 port."""
    n = _network(
        costs, {"enable": True, "fuel_price": FUEL_PRICE, "reserve_only": reserve_only}
    )
    chp = n.links[n.links.carrier == CARRIER]
    assert list(chp.index) == [f"a {CARRIER}"]
    link = chp.iloc[0]
    assert (link.bus0, link.bus1, link.bus2) == ("a methanol", "a", f"a {CENTRAL} heat")
    proxy = costs.loc[H2_CHP_PROXY]
    assert link.capital_cost == pytest.approx(proxy.capital_cost * proxy.efficiency)
    assert link.efficiency == pytest.approx(proxy.efficiency)
    assert link.efficiency2 == pytest.approx(proxy.efficiency / proxy.c_b)
    assert link.lifetime == proxy.lifetime
    assert link.p_nom_extendable
    # Same access rule and fuel pricing as the methanol turbines on the same tank.
    turbine = n.links.loc["a OCGT methanol"]
    assert link.p_max_pu == turbine.p_max_pu
    assert link.reserve_only == turbine.reserve_only == float(reserve_only)
    vom = proxy.VOM * proxy.efficiency
    assert link.marginal_cost == pytest.approx(
        vom + (FUEL_PRICE if reserve_only else 0.0)
    )


def test_inert_without_methanol(costs):
    """No methanol layer, no methanol CHP."""
    n = _network(costs, {"enable": False, "fuel_price": FUEL_PRICE})
    assert not (n.links.carrier == CARRIER).any()
