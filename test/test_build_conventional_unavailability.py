# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Unit tests for `build_conventional_unavailability`, on synthetic ENTSO-E outage notices.

What they pin is what would silently bias availability: superseded revisions and cancelled
notices must not count, a plant reporting at both unit levels must not count twice,
overlapping notices of one unit must be capped at its nominal power, and an outage spanning
a month boundary must be split by the hours in each month.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.build_conventional_unavailability import (
    add_fallback,
    apply_closures,
    clean_outages,
    climatology,
    lost_energy,
    merge_overlaps,
    monthly_capacity,
    monthly_energy,
    plant_headroom,
    reference_months,
    reference_years,
    reported_years,
    verified_capacity,
    verified_loss_share,
)


def _notice(mrid, start, end, avail=0.0, nominal=500.0, revision=1, **kw):
    return {
        "mrid": mrid,
        "revision": revision,
        "docstatus": kw.get("docstatus"),
        "businesstype": kw.get("businesstype", "Planned maintenance"),
        "plant_type": kw.get("plant_type", "Nuclear"),
        "production_resource_id": kw.get("plant", "P1"),
        "production_resource_psr_name": kw.get("unit", "G1")
        if kw.get("level", "generation") == "generation"
        else None,
        "nominal_power": nominal,
        "avail_qty": avail,
        "start": pd.Timestamp(start, tz="UTC"),
        "end": pd.Timestamp(end, tz="UTC"),
        "level": kw.get("level", "generation"),
        "country": kw.get("country", "XX"),
    }


def _units(mw_by_year, unit="P1"):
    return pd.DataFrame(
        [
            {
                "carrier": "nuclear",
                "country": "XX",
                "production_resource_id": unit,
                "year": y,
                "installed_capacity": mw,
            }
            for y, mw in mw_by_year.items()
        ]
    )


def _capacity(mw_by_year, closures=None):
    return monthly_capacity(
        _units(mw_by_year), pd.Series(closures or {}, dtype="datetime64[ns, UTC]")
    )


def _clean(*notices, outage_types=("planned", "forced")):
    return clean_outages(pd.DataFrame(notices), ["nuclear"], list(outage_types))


def test_latest_revision_and_void_notices():
    df = _clean(
        _notice("a", "2024-01-01", "2024-01-11", revision=1),
        _notice("a", "2024-01-01", "2024-01-03", revision=2),
        _notice("b", "2024-02-01", "2024-02-11", docstatus="Cancelled"),
        _notice("c", "2024-03-01", "2024-03-02", plant_type="Fossil Gas"),
    )
    assert list(df.mrid) == ["a"]
    assert df.end.iloc[0] == pd.Timestamp("2024-01-03", tz="UTC")


def test_outage_types_filter():
    df = _clean(
        _notice("a", "2024-01-01", "2024-01-02"),
        _notice("b", "2024-01-05", "2024-01-06", businesstype="Unplanned outage"),
        outage_types=["planned"],
    )
    assert list(df.outage_type) == ["planned"]


def test_plant_levels_combined_not_dropped_or_doubled():
    # Ringhals pattern: refuelling reported for the production unit (1000 MW, 10 days), a
    # generator trip inside it (500 MW) and one outside it (500 MW, 1 day).
    df = _clean(
        _notice(
            "a", "2024-01-01", "2024-01-11", nominal=1000, level="production", unit=None
        ),
        _notice("b", "2024-01-03", "2024-01-04", nominal=500, unit="G1"),
        _notice("c", "2024-01-20", "2024-01-21", nominal=500, unit="G2"),
    )
    energy = monthly_energy(
        merge_overlaps(df, ["carrier", "country"]), ["carrier", "country"]
    )
    assert energy.sum() == pytest.approx(1000 * 10 * 24 + 500 * 24)


def test_overlaps_capped_at_nominal():
    # A forced outage during planned maintenance: 500 MW down, not 900.
    df = _clean(
        _notice("a", "2024-01-01", "2024-01-11", avail=0),
        _notice(
            "b", "2024-01-05", "2024-01-06", avail=100, businesstype="Unplanned outage"
        ),
    )
    seg = merge_overlaps(df, ["carrier", "country"])
    assert seg.unavailable.max() == pytest.approx(500)
    energy = monthly_energy(seg, ["carrier", "country"])
    assert energy.sum() == pytest.approx(500 * 10 * 24)


def test_month_boundary_split():
    df = _clean(_notice("a", "2024-01-31 12:00", "2024-02-01 12:00", avail=250))
    energy = monthly_energy(
        merge_overlaps(df, ["carrier", "country"]), ["carrier", "country"]
    )
    assert energy.loc[("nuclear", "XX", pd.Period("2024-01"))] == pytest.approx(
        250 * 12
    )
    assert energy.loc[("nuclear", "XX", pd.Period("2024-02"))] == pytest.approx(
        250 * 12
    )


def test_climatology_averages_years_per_calendar_month():
    # 1000 MW installed; January 2023 fully out, January 2024 half out: mean 0.25 available.
    df = _clean(
        _notice("a", "2023-01-01", "2023-02-01", nominal=1000),
        _notice("b", "2024-01-01", "2024-02-01", nominal=1000, avail=500),
    )
    capacity = _capacity({2023: 1000.0, 2024: 1000.0})
    months = reference_months(
        {"include": ["2023", "2024"], "exclude": []}, [2023, 2024]
    )
    clim = climatology(df, capacity, months).set_index(["country", "period"])
    assert clim.at[("XX", 1), "p_max_pu"] == pytest.approx(0.25)
    assert clim.at[("XX", 1), "planned"] == pytest.approx(0.75)
    assert clim.at[("XX", 2), "p_max_pu"] == pytest.approx(1.0)
    assert clim.at[("XX", 1), "years"] == 2


def test_reference_periods():
    assert reference_years({"include": [], "exclude": ["2022"]}, last_year=2025) == [
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2023,
        2024,
        2025,
    ]
    months = reference_months(
        {"include": ["2023", "2024-02"], "exclude": ["2023-07"]}, [2023, 2024]
    )
    assert len(months) == 12
    assert pd.Period("2023-07") not in months
    assert pd.Period("2024-02") in months


def test_median_fallback():
    df = pd.DataFrame(
        {
            "carrier": "nuclear",
            "country": ["AA", "BB", "CC"],
            "unit": "",
            "period": 1,
            "p_max_pu": [0.6, 0.8, 0.9],
        }
    )
    out = add_fallback(df, ["AA", "BB", "CC", "PL"], ["nuclear"], "median")
    pl = out[out.country == "PL"]
    assert pl.p_max_pu.item() == pytest.approx(0.8)
    assert pl.source.item() == "median"
    assert np.isclose(
        add_fallback(df, ["PL"], ["nuclear"], "static").p_max_pu, [0.6, 0.8, 0.9]
    ).all()


def test_min_coverage_drops_underreported_years(tmp_path):
    # Table says 10% lost in both years; 2023 reports 9.9%, 2024 reports 0.4% (one week of a
    # 500 MW unit on a 10 GW fleet, the Slovak 2019 pattern).
    df = _clean(
        _notice("a", "2023-01-01", "2023-02-07 04:48", nominal=10000),
        _notice("b", "2024-01-01", "2024-01-08", nominal=500),
    )
    capacity = _capacity({2023: 10000.0, 2024: 10000.0})
    table = tmp_path / "p_max_pu.csv"
    pd.DataFrame({"country": ["XX"], "2023": [0.9], "2024": [0.9]}).to_csv(
        table, index=False
    )
    used, check = reported_years(df, capacity, {"nuclear": str(table)}, 0.5)
    assert list(used.get_level_values("year")) == [2023]
    assert check.at[("nuclear", "XX", 2024), "coverage"] < 0.1
    used, _ = reported_years(df, capacity, {"nuclear": str(table)}, None)
    assert len(used) == 2


def test_duplicate_notices_count_once():
    # Paks pattern: one refuelling published five times under different mRIDs.
    df = _clean(*[_notice(m, "2024-04-01", "2024-05-01") for m in "abcde"])
    assert len(df) == 1


def test_closed_unit_stops_counting():
    # A unit closed on 1 July still reports a year-long outage (the British pattern): only
    # the first half counts, against first-half capacity, so the year is fully available.
    closures = pd.Series({"P1": pd.Timestamp("2024-07-01", tz="UTC")})
    df = _clean(_notice("a", "2024-07-01", "2025-01-01", nominal=1000))
    units = _units({2024: 1000.0})
    df, units = apply_closures(df, units, closures)
    assert df.empty
    capacity = monthly_capacity(units, closures)
    assert capacity.loc[("nuclear", "XX", pd.Period("2024-08"))] == 0
    assert capacity.loc[("nuclear", "XX", pd.Period("2024-06"))] == pytest.approx(
        1000 * 30 * 24
    )


def test_notices_of_unlisted_units_dropped():
    # Fessenheim pattern: no longer in the capacity list, still publishing notices.
    df = _clean(_notice("a", "2024-01-01", "2024-02-01", plant="GONE"))
    df, _ = apply_closures(
        df, _units({2024: 1000.0}), pd.Series(dtype="datetime64[ns, UTC]")
    )
    assert df.empty


def test_output_bounds_reported_loss():
    # A notice says 1000 MW off for 10 h, but the plant produced 600 MW in hours 5-9: it
    # cannot have lost more than 400 MW then.
    df = _clean(_notice("a", "2024-01-01 00:00", "2024-01-01 10:00", nominal=1000))
    hours = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    output = [0.0] * 5 + [600.0] * 5
    generation = pd.DataFrame(
        {
            "unit_code": "C1",
            "unit_name": "G1",
            "country": "XX",
            "time": hours,
            "output": output,
        }
    )
    units = _units({2024: 1000.0}).assign(production_resource_name="Plant")
    headroom = plant_headroom(generation, df, units)
    assert lost_energy(df, ["carrier", "country"]).sum() == pytest.approx(10_000)
    assert lost_energy(df, ["carrier", "country"], headroom).sum() == pytest.approx(
        5 * 1000 + 5 * 400
    )


def test_generator_maps_to_plant_by_name_prefix():
    # Forsmark pattern: notices only per plant, output per generator "<plant> G21".
    df = _clean(
        _notice(
            "a", "2024-01-01", "2024-01-02", level="production", plant="P1", unit=None
        )
    )
    generation = pd.DataFrame(
        {
            "unit_code": "C1",
            "unit_name": "Forsmark block 2 G21",
            "country": "XX",
            "time": pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"),
            "output": 500.0,
        }
    )
    units = _units({2024: 1000.0}).assign(production_resource_name="Forsmark block 2")
    headroom = plant_headroom(generation, df, units)
    assert set(headroom.index.get_level_values("unit")) == {"P1"}
    assert headroom.iloc[0] == pytest.approx(500)


def test_only_hours_with_output_data_count():
    # Output data for the first 4 of 10 outage hours only (the British pattern): the other
    # 6 hours drop out of both the loss and the capacity, and the year fails the check
    # because only 40% of its loss is verifiable.
    df = _clean(_notice("a", "2024-01-01 00:00", "2024-01-01 10:00", nominal=1000))
    generation = pd.DataFrame(
        {
            "unit_code": "C1",
            "unit_name": "G1",
            "country": "XX",
            "time": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "output": 0.0,
        }
    )
    units = _units({2024: 1000.0}).assign(production_resource_name="Plant")
    headroom = plant_headroom(generation, df, units)
    assert lost_energy(df, ["carrier", "country"], headroom).sum() == pytest.approx(
        4000
    )
    no_closures = pd.Series(dtype="datetime64[ns, UTC]")
    assert verified_capacity(units, no_closures, headroom).sum() == pytest.approx(4000)
    assert verified_loss_share(df, headroom).item() == pytest.approx(0.4)


def test_attach_unavailability_by_calendar_month():
    import pypsa

    from scripts.add_electricity import attach_unavailability

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", "2013-12-31 21:00", freq="3h"))
    n.add("Bus", ["FR0", "XX0"], country=["FR", "XX"])
    n.add(
        "Generator",
        ["FR0 nuclear", "XX0 nuclear"],
        bus=["FR0", "XX0"],
        carrier="nuclear",
        p_max_pu=0.8,
    )
    table = pd.DataFrame(
        {
            "carrier": "nuclear",
            "country": "FR",
            "unit": "",
            "period": range(1, 13),
            "p_max_pu": [0.9] + [0.7] * 10 + [0.85],
        }
    )
    attach_unavailability(n, table)
    p = n.get_switchable_as_dense("Generator", "p_max_pu")
    assert p.loc["2013-01-15", "FR0 nuclear"].unique().tolist() == [0.9]
    assert p.loc["2013-07-15", "FR0 nuclear"].unique().tolist() == [0.7]
    # No row for XX: the static value stays.
    assert (p["XX0 nuclear"] == 0.8).all()
