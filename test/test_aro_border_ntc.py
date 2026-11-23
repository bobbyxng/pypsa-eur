# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
`aro.border_ntc` and `aro.transit_countries`: TYNDP NTCs on cross-border links.

Toy inputs, no build needed. The reference-grid rows mimic the TYNDP 2026 layout: offshore
nodes, zone borders inside a country, Poland's one-directional virtual nodes, and a pair
split into sub-borders.
"""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.build_aro_border_ntc import border_ntc
from scripts.prepare_aro_network import (
    NTC_CARRIER,
    add_dc_losses,
    bus_islands,
    fix_cross_border_transmission,
    mark_transit_countries,
    replace_cross_border_with_ntc_links,
)

EFF = {"efficiency_static": 0.98, "efficiency_per_1000km": 0.977}


def _rows(records):
    return pd.DataFrame(
        records, columns=["tyndp_border", "node0", "node1", "p01", "p10"]
    )


def _subborders(records):
    return pd.DataFrame(
        records, columns=["border", "subborder", "tyndp_borders", "select", "action"]
    )


NO_SUB = _subborders([])


# ---- build_aro_border_ntc -------------------------------------------------------------


def test_offshore_and_inner_zone_rows_dropped():
    rows = _rows(
        [
            ("DE00-FR00", "DE00", "FR00", 3000, 2000),
            ("DE00-DKNS_OFF", "DE00", "DKNS_OFF", 2000, 2000),  # hybrid leg
            ("SE03-SE04", "SE03", "SE04", 5000, 5000),  # zone border inside SE
            ("UK00-UKNI", "UK00", "UKNI", 500, 500),  # Moyle, inside GB
        ]
    )
    out = border_ntc(rows, NO_SUB)
    assert set(out.border) == {"DE-FR"}
    v = out.set_index(["from", "to"]).p_nom
    assert v[("DE", "FR")] == 3000 and v[("FR", "DE")] == 2000


def test_directions_never_summed_across():
    # Poland's virtual nodes carry one direction each
    rows = _rows(
        [
            ("PL00E-CZ00", "PL00E", "CZ00", 800, 0),
            ("CZ00-PL00I", "CZ00", "PL00I", 600, 0),
            ("PL00I-PL00", "PL00I", "PL00", 3000, 0),
        ]
    )
    v = border_ntc(rows, NO_SUB).set_index(["from", "to"]).p_nom
    assert v[("PL", "CZ")] == 800 and v[("CZ", "PL")] == 600


def test_uk_is_gb_and_zones_fold_into_country():
    rows = _rows(
        [
            ("IE00-UK00", "IE00", "UK00", 1750, 1750),
            ("NOS1-SE03", "NOS1", "SE03", 2000, 2100),
            ("NOM1-SE02", "NOM1", "SE02", 600, 1000),
        ]
    )
    v = border_ntc(rows, NO_SUB).set_index(["border", "from", "to"]).p_nom
    assert v[("GB-IE", "IE", "GB")] == 1750
    assert v[("NO-SE", "NO", "SE")] == 2600 and v[("NO-SE", "SE", "NO")] == 3100


def test_subborders_kept_apart():
    rows = _rows(
        [
            ("DE00-DKE1", "DE00", "DKE1", 600, 585),
            ("DE00-DKW1", "DE00", "DKW1", 3500, 3500),
        ]
    )
    sub = _subborders(
        [
            ("DE-DK", "east", "DE00-DKE1", "island:DK 1", "scale"),
            ("DE-DK", "west", "DKW1-DE00", "rest", "scale"),  # node order ignored
        ]
    )
    v = border_ntc(rows, sub).set_index(["subborder", "from", "to"]).p_nom
    assert v[("east", "DK", "DE")] == 585 and v[("west", "DE", "DK")] == 3500


def test_row_of_split_border_without_subborder_refused():
    rows = _rows(
        [
            ("DE00-DKE1", "DE00", "DKE1", 600, 585),
            ("DE00-DKW1", "DE00", "DKW1", 3500, 3500),
        ]
    )
    sub = _subborders([("DE-DK", "east", "DE00-DKE1", "island:DK 1", "scale")])
    with pytest.raises(ValueError, match="matches none of them"):
        border_ntc(rows, sub)


# ---- replace_cross_border_with_ntc_links ---------------------------------------------


def _network() -> pypsa.Network:
    """DE 1, DE 2, FR 1, DK 1 (east), DK 2 (west); borders DE-FR (2 pairs), DE-DK (2)."""
    n = pypsa.Network()
    n.add(
        "Bus",
        ["DE 1AC", "DE 2AC", "FR 1AC", "DK 1AC", "DK 2AC"],
        country=["DE", "DE", "FR", "DK", "DK"],
        carrier="AC",
    )
    n.add(
        "Line",
        ["inner", "defr_a", "defr_b", "defr_c", "dedk_w"],
        bus0=["DE 1AC", "DE 1AC", "FR 1AC", "DE 2AC", "DE 2AC"],
        bus1=["DE 2AC", "FR 1AC", "DE 1AC", "FR 1AC", "DK 2AC"],
        x=0.1,
        s_nom=[1000.0, 2000.0, 1000.0, 1000.0, 5000.0],
        s_max_pu=0.7,
        capital_cost=[1.0, 10.0, 10.0, 20.0, 5.0],
        length=[100.0, 50.0, 50.0, 80.0, 60.0],
    )
    n.add(
        "Link",
        ["kontek", "dc_inner"],
        bus0=["DK 1AC", "DE 1AC"],
        bus1=["DE 1AC", "DE 2AC"],
        carrier="DC",
        length=[170.0, 300.0],
        p_nom=[600.0, 500.0],
        capital_cost=[30.0, 30.0],
    )
    return n


NTC = pd.DataFrame(
    [
        ("DE-FR", "", "DE", "FR", 3000.0, "DE00-FR00"),
        ("DE-FR", "", "FR", "DE", 1500.0, "DE00-FR00"),
        ("DE-DK", "east", "DE", "DK", 600.0, "DE00-DKE1"),
        ("DE-DK", "east", "DK", "DE", 585.0, "DE00-DKE1"),
        ("DE-DK", "west", "DE", "DK", 3500.0, "DE00-DKW1"),
        ("DE-DK", "west", "DK", "DE", 3500.0, "DE00-DKW1"),
        ("DE-GR", "", "DE", "GR", 2870.0, "DE00-GR00"),  # GR not modelled here
    ],
    columns=["border", "subborder", "from", "to", "p_nom", "tyndp_borders"],
)
SUB = _subborders(
    [
        ("DE-DK", "east", "DE00-DKE1", "island:DK 1", "scale"),
        ("DE-DK", "west", "DE00-DKW1", "rest", "scale"),
    ]
)


def _replace(n, ntc=NTC, sub=SUB):
    add_dc_losses(n, EFF)
    return replace_cross_border_with_ntc_links(n, ntc, sub, efficiency=0.999)


def _flow_limit(n, frm_country, to_country, subborder=None):
    c = n.buses.country
    links = n.links[n.links.carrier == NTC_CARRIER]
    if subborder is not None:
        links = links[links.subborder == subborder]
    sel = (links.bus0.map(c) == frm_country) & (links.bus1.map(c) == to_country)
    return links.loc[sel, "p_nom"].sum()


def test_no_ac_line_crosses_and_inner_elements_untouched():
    n = _network()
    _replace(n)
    c = n.buses.country
    assert (n.lines.bus0.map(c) == n.lines.bus1.map(c)).all()
    assert list(n.lines.index) == ["inner"]
    assert {"dc_inner", "dc_inner-reversed"} <= set(n.links.index)
    assert "kontek" not in n.links.index and "kontek-reversed" not in n.links.index


def test_one_link_per_bus_pair_and_direction():
    n = _network()
    _replace(n)
    ntc = n.links[n.links.carrier == NTC_CARRIER]
    # DE1-FR1 (defr_a + defr_b), DE2-FR1, DE2-DK2, DE1-DK1 -> 4 pairs, 8 links
    assert len(ntc) == 8
    assert ntc.index.str.endswith("-reversed").sum() == 4
    pair = ntc[ntc.replaces.str.contains("defr_a")]
    assert set(pair.replaces) == {"defr_a defr_b"}


def test_border_totals_equal_tyndp_per_direction():
    n = _network()
    report = _replace(n)
    assert _flow_limit(n, "DE", "FR") == pytest.approx(3000.0)
    assert _flow_limit(n, "FR", "DE") == pytest.approx(1500.0)
    assert _flow_limit(n, "DE", "DK", "east") == pytest.approx(600.0)
    assert _flow_limit(n, "DK", "DE", "east") == pytest.approx(585.0)
    assert _flow_limit(n, "DE", "DK", "west") == pytest.approx(3500.0)
    scaled = report[report.action == "scale"]
    np.testing.assert_allclose(scaled.model_mw, scaled.tyndp_mw)


def test_shared_by_todays_capacity():
    n = _network()
    _replace(n)
    links = n.links[n.links.carrier == NTC_CARRIER]
    # DE-FR today: DE1-FR1 0.7*(2000+1000)=2100, DE2-FR1 0.7*1000=700 -> 3:1
    a = links[links.replaces == "defr_a defr_b"]
    c_ = links[links.replaces == "defr_c"]
    de = n.buses.country
    fwd_a = a.loc[a.bus0.map(de) == "DE", "p_nom"].sum()
    fwd_c = c_.loc[c_.bus0.map(de) == "DE", "p_nom"].sum()
    assert fwd_a == pytest.approx(2250.0) and fwd_c == pytest.approx(750.0)


def test_forward_is_larger_direction_and_carries_cost():
    n = _network()
    _replace(n)
    links = n.links[n.links.carrier == NTC_CARRIER]
    fwd = links[~links.index.str.endswith("-reversed")]
    rev = links.loc[[f"{i}-reversed" for i in fwd.index]]
    assert (fwd.p_nom.values >= rev.p_nom.values).all()
    assert (rev.capital_cost == 0).all()
    assert (fwd.efficiency == 0.999).all() and (rev.efficiency == 0.999).all()
    assert not links.p_nom_extendable.any()
    # DE1-FR1: capex today (10*2000 + 10*1000) over 2100 MW, times the forward 2250 MW
    a = fwd[fwd.replaces == "defr_a defr_b"].iloc[0]
    assert a.capital_cost * a.p_nom == pytest.approx(30000.0 / 2100.0 * 2250.0)


def test_keep_subborder_keeps_todays_capacity():
    n = _network()
    sub = _subborders(
        [
            ("DE-DK", "east", "DE00-DKE1", "island:DK 1", "keep"),
            ("DE-DK", "west", "DE00-DKW1", "rest", "scale"),
        ]
    )
    _replace(n, sub=sub)
    assert _flow_limit(n, "DE", "DK", "east") == pytest.approx(600.0)
    assert _flow_limit(n, "DK", "DE", "east") == pytest.approx(600.0)


def test_border_without_tyndp_value_refused():
    n = _network()
    with pytest.raises(ValueError, match="no TYNDP NTC"):
        _replace(n, ntc=NTC[NTC.border != "DE-FR"])


def test_unknown_island_in_selector_refused():
    n = _network()
    sub = SUB.replace({"select": {"island:DK 1": "island:DK 9"}})
    with pytest.raises(ValueError, match="no bus of the network"):
        _replace(n, sub=sub)


def test_tyndp_border_without_element_reported():
    n = _network()
    n.add("Bus", "GR 1AC", country="GR", carrier="AC")
    report = _replace(n)
    gr = report[report.border == "DE-GR"]
    assert (gr.action == "no element").all() and gr.tyndp_mw.max() == 2870.0


def test_fix_cross_border_keeps_ntc_links_fixed():
    n = _network()
    _replace(n)
    fix_cross_border_transmission(n)
    assert not n.links.loc[n.links.carrier == NTC_CARRIER, "p_nom_extendable"].any()


def test_bus_islands_strip_carrier():
    n = _network()
    assert bus_islands(n)["DK 1AC"] == "DK 1"


# ---- transit countries ---------------------------------------------------------------


def test_transit_marked_on_buses_and_meta():
    n = _network()
    n.add("Bus", ["BA 1AC", "BA 1 H2"], country="BA", carrier=["AC", "H2"])
    mark_transit_countries(n, ["BA"])
    assert n.buses.loc["BA 1AC", "transit"] and not n.buses.loc["DE 1AC", "transit"]
    assert n.meta["aro_transit_countries"] == ["BA"]


def test_transit_country_with_two_ac_buses_refused():
    n = _network()
    with pytest.raises(ValueError, match="exactly one"):
        mark_transit_countries(n, ["DE"])
