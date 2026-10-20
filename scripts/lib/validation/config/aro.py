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


class AroHeatEndogenousConfig(ConfigModel):
    """Configuration for `aro.heat.endogenous` settings (used only with `mode: endogenous`)."""

    h2_chp: bool = Field(
        True,
        description="Offer an H2 CHP (H2 -> electricity + district heat) at urban-central nodes that have an H2 bus. Priced with pypsa-eur's 'central gas CHP' row as an H2-ready engine/turbine proxy; pypsa-eur has no H2 CHP of its own.",
    )
    variable_costs: bool = Field(
        False,
        description="Charge technology-data VOM on the heat technologies, which pypsa-eur's add_heat leaves at zero: central air heat pump 3.57 and central resistive heater 1.34 EUR/MWh_th, decentral water-tank discharge 1.55 EUR/MWh (decentral heat pump and resistive heater have no VOM in technology-data). Real O&M that also breaks the near-free heat-pump/resistive/tank ties that stall barrier. Off (default) keeps parity with add_heat.",
    )
    methanol_chp: bool = Field(
        True,
        description="Offer a methanol CHP (methanol -> electricity + district heat) at urban-central nodes, drawing on the local methanol tank of `aro.methanol` with the same access rule as its turbines (backup-only under `reserve_only`). Inert unless `aro.methanol.enable`. 'central gas CHP' data, as Glaum et al.'s min-MeOH-economy.",
    )


class AroHeatConfig(ConfigModel):
    """Configuration for `aro.heat` settings."""

    enable: bool = Field(
        False,
        description="Add building heat to the network. Off by default so existing electricity-only runs are unchanged; the no-heat baseline is just this flag set to false.",
    )
    mode: Literal["exogenous", "endogenous"] = Field(
        "exogenous",
        description="'exogenous': heat as fixed heat-pump/resistive electricity Loads on the AC buses, split by `shares`. 'endogenous': heat buses ('decentral', 'urban central') with heat Loads served by optimised air heat pumps, resistive heaters, water tanks (and an H2 CHP in district heating), mirroring pypsa-eur's add_heat; `shares` and `heat_pump_sources` are then ignored.",
    )
    endogenous: AroHeatEndogenousConfig = Field(
        default_factory=AroHeatEndogenousConfig,
        description="Options for `mode: endogenous`.",
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


class AroH2TurbineConfig(ConfigModel):
    """Configuration for `aro.h2_turbine` settings."""

    technologies: list[str] = Field(
        default_factory=list,
        description="Gas turbines offered alongside the fuel cell for H2 re-electrification, named by their cost-table row, e.g. ['OCGT', 'CCGT']. `attach_stores` hardcodes `fuel cell` as carrier H2's only discharger, so an electricity-only network re-electrifies hydrogen solely at ~206k EUR/MW_el/a, which a CCGT beats on both cost and efficiency. Which turbine wins depends on capacity factor, so offer both and let the optimiser split. Empty by default so existing runs are unchanged.",
    )


class AroMethanolConfig(ConfigModel, extra="forbid"):
    """Purchased renewable methanol, local tanks and electricity generation."""

    enable: bool = Field(
        False,
        description="Add purchased CO2-neutral methanol at each AC node; no endogenous synthesis or carbon capture.",
    )
    fuel_price: float = Field(
        130.0,
        ge=0,
        allow_inf_nan=False,
        description="Delivered renewable methanol price in EUR/MWh_LHV. Default 130 is a literature-informed 2050 study assumption, not an exact inflation-adjusted IRENA estimate; monetary-year harmonisation remains open. IRENA 2021 quotes USD/t: divide by 19.9/3.6 MWh/t and explicitly convert currency/cost year. Neutrality is an external fuel-sourcing assumption, not zero stack emissions.",
    )
    technologies: list[Literal["OCGT", "CCGT"]] = Field(
        default_factory=lambda: ["OCGT", "CCGT"],
        min_length=1,
        description="Competing methanol turbines; reuse the processed gas-turbine efficiency, capital_cost and VOM, excluding fossil gas fuel costs.",
    )
    reserve_only: bool = Field(
        False,
        description="Use dedicated emergency methanol stock: invest in shared ready inventory and tanks/turbines, permit generation only during outages, and forbid fuel purchases in recourse. Requires enumeration; normal operation remains part of the robust cost comparison.",
    )

    reserve_inventory_carrying_rate: float = Field(
        0.07,
        ge=0,
        allow_inf_nan=False,
        description="Annual financing rate for permanent emergency methanol inventory, assuming unused fuel retains its value. Reserve-only mode charges fuel_price times ready inventory times this rate (scaled to the model horizon), plus consumed fuel as post-event replenishment cost. No fuel purchases are allowed in dispatch.",
    )

    @model_validator(mode="after")
    def check_methanol(self):
        """Reject duplicate turbine components."""
        if len(set(self.technologies)) != len(self.technologies):
            raise ValueError("aro.methanol.technologies must not contain duplicates")
        return self


class AroConfig(ConfigModel):
    """Configuration for top-level `aro` settings."""

    heat: AroHeatConfig = Field(
        default_factory=AroHeatConfig,
        description="Exogenous electrified-heat load added by `prepare_aro_network`.",
    )
    h2_turbine: AroH2TurbineConfig = Field(
        default_factory=AroH2TurbineConfig,
        description="Second H2 discharger added by `prepare_aro_network`.",
    )
    methanol: AroMethanolConfig = Field(
        default_factory=AroMethanolConfig,
        description="Optional purchased renewable methanol, local storage and turbines.",
    )
