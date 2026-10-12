# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""Validate the purchased-methanol modelling boundary."""

import pytest
from pydantic import ValidationError

from scripts.lib.validation.config.aro import AroMethanolConfig


@pytest.mark.parametrize(
    "options",
    [
        {"enable": True},
        {"fuel_price": -1},
        {"fuel_price": float("nan")},
        {"reserve_inventory_carrying_rate": -0.1},
        {"reserve_inventory_carrying_rate": float("inf")},
        {"delivery_capacity": float("inf")},
        {"delivery_capacity": 0},
        {"technologies": []},
        {"technologies": ["OCGT", "OCGT"]},
        {"technologies": ["fuel cell"]},
    ],
)
def test_invalid_config_is_rejected(options):
    """Missing prices, invalid quantities and unsupported converters must fail."""
    with pytest.raises(ValidationError):
        AroMethanolConfig.model_validate(options)


def test_optional_and_explicit_price():
    """Existing configs stay disabled; enabled configs require a deliberate price."""
    assert not AroMethanolConfig().enable
    assert not AroMethanolConfig().reserve_only
    assert AroMethanolConfig(
        enable=True, fuel_price=100, reserve_only=True
    ).reserve_only
    cfg = AroMethanolConfig(enable=True, fuel_price=100)
    assert cfg.technologies == ["OCGT", "CCGT"]
    assert cfg.fuel_price == 100
