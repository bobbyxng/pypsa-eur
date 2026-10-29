# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Conventional generators configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#conventional_cf
"""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator

from scripts.lib.validation.config._base import ConfigModel

_Period = Annotated[str, Field(pattern=r"^\d{4}(-(0[1-9]|1[0-2]))?$")]


class _ReferencePeriodsConfig(ConfigModel):
    """Configuration for `conventional.unavailability.reference_periods` settings."""

    include: list[_Period] = Field(
        default_factory=list,
        description="Calendar years (`YYYY`) or months (`YYYY-MM`) of historical outage data to use. Empty uses every year from 2015, the start of the ENTSO-E Transparency Platform, to the last complete calendar year.",
    )
    exclude: list[_Period] = Field(
        default_factory=list,
        description="Calendar years (`YYYY`) or months (`YYYY-MM`) of historical outage data to leave out; takes precedence over `include`.",
    )

    @field_validator("include", "exclude", mode="before")
    @classmethod
    def _periods_as_strings(cls, v):
        # An unquoted 2022 in YAML arrives as an int.
        return [str(p) for p in v] if isinstance(v, list) else v


class _UnavailabilityConfig(ConfigModel):
    """Configuration for `conventional.unavailability` settings."""

    enable: bool = Field(
        False,
        description="Derive time-varying availability (`p_max_pu`) of conventional generators from reported outages on the ENTSO-E Transparency Platform (unavailability of production and generation units, 15.1.A-D), instead of the static values under `conventional: <carrier>: p_max_pu`.",
    )
    carriers: list[str] = Field(
        default_factory=lambda: ["nuclear"],
        description="Carriers whose availability is derived from outages.",
    )
    outage_types: list[Literal["planned", "forced"]] = Field(
        default_factory=lambda: ["planned", "forced"],
        description="Outage types that reduce availability: planned maintenance and/or forced (unplanned) outages.",
    )
    method: Literal["climatology", "historical"] = Field(
        "climatology",
        description="'climatology' averages each country's availability per calendar month over `reference_periods` and applies it to any modelled year. 'historical' applies the outages of the modelled snapshots themselves, per power plant where it maps to an ENTSO-E unit and per country otherwise; the snapshots must lie within the retrieved years.",
    )
    reference_periods: _ReferencePeriodsConfig = Field(
        default_factory=_ReferencePeriodsConfig,
        description="Historical periods of outage data used to build the availability profile ('climatology' only). Independent of the modelled snapshots.",
    )
    min_coverage: float | None = Field(
        0.5,
        ge=0,
        description="Reporting on the Transparency Platform is incomplete in some countries and years (e.g. Slovak nuclear only from 2021), and an unreported outage reads as availability. Where the carrier has a static `p_max_pu` table (IAEA PRIS energy availability factors for nuclear), a country-year is used only if the output lost to reported outages is at least this share of the loss the table implies, (1 - availability from outages) / (1 - table value). Country-years without any reported outage are never used. null skips the comparison.",
    )
    fallback: Literal["median", "static"] = Field(
        "median",
        description="Availability for countries without outage data: 'median' takes the median over the countries with data (per calendar month, or per snapshot for 'historical'); 'static' keeps the values under `conventional: <carrier>: p_max_pu`.",
    )


class ConventionalConfig(ConfigModel):
    """Configuration for `conventional` settings."""

    model_config = ConfigDict(extra="allow")

    unit_commitment: bool = Field(
        False,
        description="Allow the overwrite of ramp_limit_up, ramp_limit_start_up, ramp_limit_shut_down, p_min_pu, min_up_time, min_down_time, and start_up_cost of conventional generators. Refer to the CSV file 'unit_commitment.csv'.",
    )
    dynamic_fuel_price: bool = Field(
        False,
        description="Consider the monthly fluctuating fuel prices for each conventional generator. Refer to the CSV file 'data/validation/monthly_fuel_price.csv'.",
    )
    fuel_price_rolling_window: int = Field(
        6,
        description="Monthly rolling mean window for fossil fuel prices smoothing.",
        ge=1,
    )
    unavailability: _UnavailabilityConfig = Field(
        default_factory=_UnavailabilityConfig,
        description="Time-varying availability of conventional generators from reported outages.",
    )
    nuclear: dict[str, str | float] = Field(
        default_factory=lambda: {"p_max_pu": "data/nuclear_p_max_pu.csv"},
        description="For any carrier/technology overwrite attributes as listed below.",
    )
