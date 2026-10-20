# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Unit tests for `build_aro_nuclear_fleet`, on a synthetic ppm table.

What they pin is what would silently move results: kept units must leave ppm untouched
(same coordinates, so the same bus), retiring units must be absent (custom rows are only
ever added), France must sum to the E3 target, and new units must sit on their sites.
"""

import pandas as pd
import pytest

from scripts.build_aro_nuclear_fleet import (
    COUNTRY_CODES,
    FR_EXISTING_MW,
    KEEP,
    NEW,
    build_fleet,
)

NAMES_TO_COUNTRY = {v: k for k, v in COUNTRY_CODES.items()}


def _unit(name, country, capacity, lat, lon, date_out=float("nan")):
    return {
        "Name": name,
        "Fueltype": "Nuclear",
        "Technology": "Steam Turbine",
        "Set": "PP",
        "Country": NAMES_TO_COUNTRY[country],
        "Capacity": capacity,
        "DateIn": 1985.0,
        "DateOut": date_out,
        "lat": lat,
        "lon": lon,
    }


@pytest.fixture
def ppm():
    rows = [
        _unit(name, country, 1000.0 + i, 40.0 + i / 10, 10.0 + i / 10)
        for country, (names, _) in KEEP.items()
        for i, name in enumerate(names)
    ]
    # Units that must retire, and one long-closed one.
    rows += [
        _unit("Olkiluoto 1", "FI", 920.0, 61.2, 21.4),
        _unit("Kozloduy 5", "BG", 1040.0, 43.7, 23.8),
        _unit("Almaraz 1", "ES", 1049.0, 39.8, -5.7),
        _unit("Torness 1", "GB", 682.0, 55.9, -2.4),
        _unit("Fessenheim 1", "FR", 920.0, 47.9, 7.6, date_out=2020.0),
    ]
    # Operating French units, including the sites the EPR2 rows are placed at.
    rows += [
        _unit("Penly 1", "FR", 1382.0, 49.98, 1.21),
        _unit("Gravelines 1", "FR", 951.0, 51.01, 2.13),
        _unit("Bugey 2", "FR", 945.0, 45.80, 5.27),
        _unit("Cattenom 1", "FR", 1362.0, 49.42, 6.22),
    ]
    rows += [
        _unit("Hinkley Point C1", "GB", 3440.0, 51.2063, -3.1423, date_out=float("nan"))
    ]
    df = pd.DataFrame(rows)
    df.loc[df.Name == "Hinkley Point C1", "DateIn"] = 2029.0
    return df


def test_kept_units_are_copied_unchanged(ppm):
    fleet = build_fleet(ppm).set_index("Name")
    for country, (names, _) in KEEP.items():
        for name in names:
            src = ppm.set_index("Name").loc[name]
            got = fleet.loc[name]
            assert got.Country == country
            assert got.Status == "existing"
            assert (got.Capacity, got.lat, got.lon) == (src.Capacity, src.lat, src.lon)


def test_retiring_units_are_absent(ppm):
    names = set(build_fleet(ppm).Name)
    assert not names & {
        "Olkiluoto 1",
        "Kozloduy 5",
        "Almaraz 1",
        "Torness 1",
        "Fessenheim 1",
    }
    # ppm's single combined HPC row is replaced by the two units, not kept alongside them.
    assert "Hinkley Point C1" not in names


def test_france_existing_scaled_to_target_at_unchanged_sites(ppm):
    fleet = build_fleet(ppm)
    fr = fleet.query("Country == 'FR' and Status == 'existing'").set_index("Name")
    assert fr.Capacity.sum() == pytest.approx(FR_EXISTING_MW)
    src = ppm.set_index("Name").loc[fr.index]
    ratio = fr.Capacity / src.Capacity
    assert ratio.max() == pytest.approx(ratio.min())
    assert (fr.lat == src.lat).all() and (fr.lon == src.lon).all()


def test_new_units_sit_on_their_sites_and_pending_is_flagged(ppm):
    fleet = build_fleet(ppm).set_index("Name")
    by_name = ppm.set_index("Name")
    for name, country, mw, _, site, status, _ in NEW:
        row = fleet.loc[name]
        assert (row.Country, row.Capacity, row.Status) == (country, mw, status)
        lat, lon = by_name.loc[site, ["lat", "lon"]] if isinstance(site, str) else site
        assert (row.lat, row.lon) == (lat, lon)
    assert set(fleet.query("Status == 'pending'").Country) == {"FR"}


def test_missing_kept_unit_raises(ppm):
    with pytest.raises(ValueError, match="units not in ppm"):
        build_fleet(ppm[ppm.Name != "Kozloduy 6"])


def test_missing_availability_gets_per_year_median_of_modelled_countries():
    from scripts.build_aro_nuclear_fleet import fill_missing_availability

    p_max_pu = pd.DataFrame(
        {"2013": [0.70, 0.80, 0.95, 0.10], "2024": [0.60, None, 0.90, 0.10]},
        index=pd.Index(["FR", "GB", "FI", "UA"], name="country"),
    )
    filled = fill_missing_availability(p_max_pu, ["FR", "PL"], ["FR", "GB", "FI", "PL"])
    # UA is outside the run: its 0.10 must not pull the median down.
    assert filled.loc["PL", "2013"] == pytest.approx(0.80)
    # A year with a gap takes the median of the modelled countries that have a value.
    assert filled.loc["PL", "2024"] == pytest.approx(0.75)
    # Existing rows are untouched and no other country is added.
    pd.testing.assert_frame_equal(filled.loc[p_max_pu.index], p_max_pu)
    assert list(filled.index) == ["FR", "GB", "FI", "UA", "PL"]
