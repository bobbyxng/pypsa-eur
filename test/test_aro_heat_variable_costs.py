# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
`aro.heat.endogenous.variable_costs`: technology-data VOM on the endogenous heat links.

A toy network, no build needed. Checks the sign on the heat pump's reversed wiring (p0 is heat
and runs negative, so a cost needs a negative coefficient), the per-MWh_th scaling on the
resistive heater (p0 is electricity), the tank VOM on the discharger, and that the default
(off) leaves every heat link at zero -- parity with upstream's add_heat.
"""

import pandas as pd
import pypsa
import pytest

from scripts.definitions.heat_system import HeatSystem
from scripts.prepare_aro_network import add_endogenous_heat

NODE = "DE0 0"
VOM = {
    "central air-sourced heat pump": 3.57,
    "central resistive heater": 1.34,
    "decentral water tank storage": 1.55,
}


def _costs() -> pd.DataFrame:
    rows = {}
    for cd in ("central", "decentral"):
        rows[f"{cd} air-sourced heat pump"] = dict(capital_cost=1e5, lifetime=20)
        rows[f"{cd} resistive heater"] = dict(
            efficiency=0.9, capital_cost=1e4, lifetime=20
        )
        rows[f"{cd} water tank charger"] = dict(efficiency=1.0)
        rows[f"{cd} water tank discharger"] = dict(efficiency=1.0)
        rows[f"{cd} water tank storage"] = {
            "capital_cost": 300.0,
            "lifetime": 30,
            "energy to power ratio": 10.0,
            "standing_losses": 0.1,
        }
    rows["water tank charger"] = dict(marginal_cost=0.03)
    costs = pd.DataFrame(rows).T
    costs["VOM"] = pd.Series(VOM)  # missing rows stay NaN, as in technology-data
    return costs


def _network(variable_costs: bool) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", NODE, carrier="AC")
    demand = pd.DataFrame({NODE: [1.0, 2.0]}, index=n.snapshots)
    inv_cop = pd.DataFrame({NODE: [0.3, 0.3]}, index=n.snapshots)
    systems = [
        (HeatSystem.URBAN_CENTRAL, demand, inv_cop),
        (HeatSystem.RESIDENTIAL_RURAL, demand, inv_cop),
    ]
    sector = {
        "overdimension_heat_generators": {"central": 1.0, "decentral": 1.0},
        "marginal_cost_heat_vent": 0.02,
    }
    options = {"h2_chp": False, "methanol_chp": False, "variable_costs": variable_costs}
    add_endogenous_heat(n, systems, _costs(), options, sector)
    return n


def _mc(n, carrier):
    return n.links.loc[n.links.carrier == carrier, "marginal_cost"].iloc[0]


def test_variable_costs_signs_and_scaling():
    n = _network(True)
    # Reversed wiring: p0 <= 0 is heat, so -VOM * p0 = +VOM per MWh_th.
    assert _mc(n, "urban central air heat pump") == pytest.approx(-3.57)
    # p0 is electricity in: VOM per MWh_th times efficiency.
    assert _mc(n, "urban central resistive heater") == pytest.approx(1.34 * 0.9)
    assert _mc(n, "decentral water tanks discharger") == pytest.approx(1.55)
    # No VOM in technology-data -> zero, not NaN.
    assert _mc(n, "decentral air heat pump") == 0
    assert _mc(n, "decentral resistive heater") == 0
    assert _mc(n, "urban central water tanks discharger") == 0


def test_default_off_keeps_heat_links_free():
    n = _network(False)
    heat = n.links[~n.links.carrier.str.endswith("charger")]
    assert (heat.marginal_cost == 0).all()
    # The charger's custom tie-breaker is untouched either way.
    assert _mc(n, "decentral water tanks charger") == pytest.approx(0.03)
