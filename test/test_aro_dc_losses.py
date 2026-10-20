# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
`aro.dc_losses`: lossy, bidirectional DC links on an electricity-only ARO network.

Toy network, no build needed. The transmission-project link enters one-directional
(p_min_pu = 0, as `new_links.csv` leaves it), which is the case the split must repair.
"""

import pypsa
import pytest

from scripts.prepare_aro_network import add_dc_losses

EFF = {"efficiency_static": 0.98, "efficiency_per_1000km": 0.977}


def _network() -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", ["a", "b"], carrier="AC")
    n.add(
        "Link",
        "TYNDP_x",
        bus0="a",
        bus1="b",
        carrier="DC",
        length=500.0,
        p_min_pu=0.0,
        p_nom_extendable=True,
        capital_cost=100.0,
    )
    return n


def test_split_gives_both_directions_with_losses():
    n = _network()
    add_dc_losses(n, EFF)
    fwd, rev = n.links.loc["TYNDP_x"], n.links.loc["TYNDP_x-reversed"]
    expected = 0.98 * 0.977**0.5
    assert fwd.efficiency == pytest.approx(expected)
    assert rev.efficiency == pytest.approx(expected)
    assert (fwd.bus0, fwd.bus1) == ("a", "b")
    assert (rev.bus0, rev.bus1) == ("b", "a")
    assert fwd.p_min_pu == 0
    assert rev.p_min_pu == 0
    # Capacity is paid once; the twin is tied to it by PyPSARO's sync constraint.
    assert rev.capital_cost == 0
    assert bool(rev.reversed)
    assert rev.p_nom_extendable


def test_refuses_to_split_twice():
    n = _network()
    add_dc_losses(n, EFF)
    with pytest.raises(ValueError, match="reversed"):
        add_dc_losses(n, EFF)
