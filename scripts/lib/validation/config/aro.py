# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
ARO configuration -- fork-specific, consumed only by `prepare_aro_network`.

Not part of upstream PyPSA-Eur. It carries the exogenous heat-electrification assumptions
that PyPSARO's ARO/C&CG resilience workflow needs layered onto an electricity-only network,
so that PyPSARO itself stays content-agnostic. See the parent repo's
`memory/heat-electrification-design.md` for why heat is a load rather than a modelled sector.
"""

from typing import Literal

from pydantic import Field, model_validator

from scripts.lib.validation.config._base import ConfigModel


class AroHeatSharesConfig(ConfigModel):
    """Fractions of a country's heat demand served by each electric technology."""

    heat_pump: float = Field(
        0.90,
        ge=0.0,
        le=1.0,
        description="Fraction of residential+services heat demand (space+water) served by heat pumps.",
    )
    resistive: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Fraction served by resistive heaters. Drawn at COP 1, so it costs ~3x the electricity per unit heat and bites hardest in cold snaps when heat-pump COP bottoms out.",
    )

    @model_validator(mode="after")
    def check_shares_sum(self):
        """Whatever is left over is non-electric heat, deliberately outside the model."""
        total = self.heat_pump + self.resistive
        if total > 1.0:
            raise ValueError(
                f"aro.heat shares sum to {total:.3f}, which exceeds the heat demand available. "
                "heat_pump + resistive must be <= 1.0; the remainder is non-electric heat "
                "(biomass, district heat from other sources, residual gas) that is deliberately "
                "not represented in the electricity model."
            )
        return self


class AroHeatConfig(ConfigModel):
    """Configuration for `aro.heat` settings."""

    enable: bool = Field(
        False,
        description="Add the exogenous heat-pump/resistive electricity load to the network. Off by default so existing electricity-only runs are unchanged; the no-heat baseline is just this flag set to false.",
    )
    shares: dict[str, AroHeatSharesConfig] = Field(
        default_factory=dict,
        description="Per-country technology split, e.g. {'DE': {'heat_pump': 0.845, 'resistive': 0.040}}. These are shares of heat SUPPLY, not electrification shares -- see memory/heat-electrification-design.md. Countries absent from this mapping fall back to `default_shares`.",
    )
    default_shares: AroHeatSharesConfig = Field(
        default_factory=AroHeatSharesConfig,
        description="Fallback split for countries with no explicit entry in `shares`. The default 90/5 is the EU-wide demand-weighted technology split from the sector-coupled reference run, leaving 5% non-electric -- which matches the EU-wide gas boiler + CHP share independently. Deliberately NOT the literature 80% electrification figure: that means a different quantity (share electrified, with a non-electric remainder) and mixing it with model-derived supply splits would be incoherent.",
    )
    heat_pump_sources: dict[str, Literal["air", "ground"]] = Field(
        default_factory=lambda: {
            # Mirrors what the sector-coupled reference run actually built: ground-source is
            # only offered in rural, and takes 99.3% of it there because its COP is far better.
            "rural": "ground",
            "urban decentral": "air",
            "urban central": "air",
        },
        description="Heat-pump source per heat system, selecting which COP curve from `cop_profiles` applies. Keys must match the `heat_system` coordinate of the COP file.",
    )


class AroConfig(ConfigModel):
    """Configuration for top-level `aro` settings."""

    heat: AroHeatConfig = Field(
        default_factory=AroHeatConfig,
        description="Exogenous electrified-heat load added by `prepare_aro_network`.",
    )
