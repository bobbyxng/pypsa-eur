# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
`aro.fix_cross_border_transmission`: interconnectors frozen, lines inside a country free.

Toy network, no build needed: two German buses and one French one, every line and DC link
extendable as `transmission_limit: vopt` leaves them, the DC links already split by
`add_dc_losses` so the `-reversed` twins must be frozen too.
"""

import pypsa

from scripts.prepare_aro_network import add_dc_losses, fix_cross_border_transmission

EFF = {"efficiency_static": 0.98, "efficiency_per_1000km": 0.977}


def _network() -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", ["DE0 1", "DE0 2", "FR0 1"], country=["DE", "DE", "FR"], carrier="AC")
    n.add(
        "Line",
        ["inner", "border"],
        bus0=["DE0 1", "DE0 2"],
        bus1=["DE0 2", "FR0 1"],
        x=0.1,
        s_nom=[1000.0, 2000.0],
        s_nom_extendable=True,
    )
    n.add(
        "Link",
        ["dc_inner", "dc_border"],
        bus0=["DE0 1", "DE0 1"],
        bus1=["DE0 2", "FR0 1"],
        carrier="DC",
        length=300.0,
        p_nom=[500.0, 700.0],
        p_nom_extendable=True,
    )
    return n


def test_borders_frozen_inner_lines_free():
    n = _network()
    add_dc_losses(n, EFF)
    fix_cross_border_transmission(n)
    assert n.lines.loc["inner", "s_nom_extendable"]
    assert not n.lines.loc["border", "s_nom_extendable"]
    assert n.lines.loc["border", "s_nom"] == 2000.0
    for name in ["dc_inner", "dc_inner-reversed"]:
        assert n.links.loc[name, "p_nom_extendable"]
    for name in ["dc_border", "dc_border-reversed"]:
        assert not n.links.loc[name, "p_nom_extendable"]
        assert n.links.loc[name, "p_nom"] == 700.0


def test_single_country_network_unchanged():
    n = _network()
    n.remove("Line", "border")
    n.remove("Link", "dc_border")
    before = (n.lines.s_nom_extendable.copy(), n.links.p_nom_extendable.copy())
    fix_cross_border_transmission(n)
    assert n.lines.s_nom_extendable.equals(before[0])
    assert n.links.p_nom_extendable.equals(before[1])
