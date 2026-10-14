# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Unit tests for `extract_electrified_heat_shares`'s pure share arithmetic.

The end-to-end anchor (DE 0.845/0.040, ES 0.946/0.054 on the old europe-pre-run network)
needs a solved sector network and is not run here; these pin the classification and the
denominator, which is where a silent error would change every country's share.
"""

import pandas as pd
import pytest

from scripts.extract_electrified_heat_shares import classify, compute_shares


@pytest.mark.parametrize(
    "carrier, expected",
    [
        ("urban decentral air heat pump", "heat_pump"),
        ("rural ground heat pump", "heat_pump"),
        ("urban central resistive heater", "resistive"),
        ("urban central gas CHP", "combustion"),
        ("rural gas boiler", "combustion"),
        ("rural solar thermal", "solar_thermal"),
        ("urban central water pits discharger", "storage"),
        ("rural water tanks discharger", "storage"),
        ("H2 Electrolysis", "ptx_waste_heat"),
        ("Sabatier", "ptx_waste_heat"),
    ],
)
def test_classify(carrier, expected):
    assert classify(carrier) == expected


def test_compute_shares_excludes_storage_and_ptx_from_denominator():
    supply = pd.Series(
        {
            ("DE", "rural air heat pump"): 80.0,
            ("DE", "rural resistive heater"): 5.0,
            ("DE", "rural gas boiler"): 15.0,
            # Must not enter the denominator, or the shares sum to less than they should.
            ("DE", "rural water tanks discharger"): 50.0,
            ("DE", "H2 Electrolysis"): 30.0,
            ("ES", "urban central air heat pump"): 9.0,
            ("ES", "urban central resistive heater"): 1.0,
        }
    )
    result = compute_shares(supply)

    assert result["shares"]["DE"] == {"heat_pump": 0.8, "resistive": 0.05}
    assert result["shares"]["ES"] == {"heat_pump": 0.9, "resistive": 0.1}
    assert result["heat_pump_fraction_of_electric"]["DE"] == pytest.approx(
        80 / 85, abs=1e-3
    )
    # Supply-weighted aggregate, not a mean of country shares.
    assert result["default_shares"] == {"heat_pump": 0.809, "resistive": 0.055}


def test_compute_shares_drops_countries_without_heat_supply():
    supply = pd.Series(
        {
            ("DE", "rural air heat pump"): 1.0,
            ("LU", "rural water tanks discharger"): 1.0,
        }
    )
    assert set(compute_shares(supply)["shares"]) == {"DE"}
