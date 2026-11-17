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
        True,
        description="Charge technology-data VOM on the heat technologies, which pypsa-eur's add_heat leaves at zero: central air heat pump 3.57 and central resistive heater 1.34 EUR/MWh_th, decentral water-tank discharge 1.55 EUR/MWh (decentral heat pump and resistive heater have no VOM in technology-data). Real O&M that also breaks the near-free heat-pump/resistive/tank ties that stall barrier. On by default since 2026-10-20; off keeps marginal-cost parity with add_heat (runs before then).",
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


class AroPartitionConfig(ConfigModel, extra="forbid"):
    """
    Outage-island partition (`build_aro_partition` and its per-country rules).

    PyPSARO's `memory: outage-island-partitioning` documents the method and every default here.
    The per-country peaks are built from raw inputs by `build_aro_country_peaks`.
    """

    exclude: list[str] = Field(
        [],
        description="Countries left out of the partition. BA, ME and XK (no LAU, thin sub-transmission data in OSM) are no longer modelled at all, so nothing is left out by default.",
    )
    peak_regions: Literal["country", "nuts1", "nuts2", "nuts3"] = Field(
        "nuts2",
        description="Regions the heat chain of `build_aro_country_peaks` runs on (independent of the islands). Heat-pump electricity is heat x 1/COP summed over regions, and cold regions have more heat and a lower COP, so one region per country understates the peak where climate varies (vs the NUTS1/2-clustered pre-run: FI -10%, GB -7%). NUTS2 is close to the islands' own resolution.",
    )
    reference_country: str = Field(
        "DE",
        description="Country whose peak, divided by `k_max`, gives the target island size.",
    )
    k_max: int = Field(
        14,
        ge=1,
        description="Islands per country at most; 14 = Germany's TSO black-start procurement regions.",
    )
    k_min: int = Field(
        3,
        ge=1,
        description="Islands per country at least (a regional structure beyond national self-sufficiency).",
    )
    size_bounds: list[float] = Field(
        [0.75, 1.33],
        min_length=2,
        max_length=2,
        description="Every island's load within these multiples of the country mean, checked on the grid-based islands. The least cut drifts to whichever bound has room, so the bounds decide the balance: with Litgrid's operator-planned range [0.5, 1.5] (0.49-1.52) Spain's mainland came out 17/17/7/7 GW, Switzerland 8/4/4 and Berlin stood alone at 0.50; [0.75, 1.33] (largest/smallest at most 1.77) gives comparable outage units with mostly coherent islands (2026-11-16). [0.5, 1.5] is the sensitivity: looser bounds let the least cut follow weak links.",
    )
    voltage_threshold: float = Field(
        220.0,
        description="kV; the outage cuts the grid at or above it, islands are formed from the grid below it.",
    )
    separate_land_mass_km2: float = Field(
        50.0,
        description="A land mass across the sea from the mainland (water wider than ~2 km; no island spans a sea crossing, whatever its voltage: Crete's 150 kV cable, Sicily's 380 kV Messina crossing) becomes its own island only if it covers at least this area, holds at least `separate_land_mass_min_gw` and has a link the model represents (a bus at or above `voltage_threshold`, a DC landing, or a link in `data/transmission_projects/aro`: Crete's Ariadne); otherwise it joins the nearest island (Zeeuws-Vlaanderen, on 150 kV cables only). It counts within k if it holds at least `size_bounds[0]` x the country's mean island (East Denmark), else it comes on top (Balearics, Corsica, Crete, Sicily).",
    )
    separate_land_mass_min_gw: float = Field(
        0.25,
        ge=0.0,
        description="Peak (GW) a land mass needs to be its own island; smaller ones join the nearest island of their country (Gotland 0.13 GW, Shetland 0.04 GW). Corsica (0.64 GW) stays separate.",
    )
    snap_to_lau: bool = Field(
        True,
        description="Snap the grid-based island outlines to municipal borders (GISCO LAU 2019); membership stays grid-based. Off: the grid-based outlines are the islands.",
    )
    lau_respect_bounds: bool = Field(
        True,
        description="After snapping, move border municipalities between neighbouring islands until every island is within `size_bounds` again (islands kept in one piece). Coarse municipalities otherwise shifted islands from within the bounds to 0.5-1.9 x mean (Slovenia, Latvia, Great Britain).",
    )
    border_min_m: float = Field(
        200.0,
        description="Supply areas are neighbours if they share more than this length of border (m).",
    )
    water_closing_m: float = Field(
        1000.0,
        description="The country outline is made 'dry' before partitioning: inland holes filled and water narrower than twice this (m) closed, so rivers, lakes and estuaries never form island borders. Wider sea straits stay open.",
    )
    land_mass_buffer_m: float = Field(
        2000.0,
        description="Buses within this distance (m) of the main land piece count as on the mainland when looking for separate land masses.",
    )
    unlinked_neighbour_weight: float = Field(
        0.01,
        description="Link weight of touching supply areas without a line below the threshold between them, as a share of the median link: nearly free to cut, so islands keep sub-transmission networks whole and only join separate ones where the size bounds require it.",
    )
    lau_border_min_m: float = Field(
        50.0,
        description="Municipalities are neighbours if they share more than this length of border (m); used to keep snapped islands in one piece.",
    )
    enumeration_max: int = Field(
        20,
        description="Enumerate every grouping exactly up to this many supply areas; METIS candidates above.",
    )
    selection: Literal["least_cut", "near_optimal_balanced", "near_optimal_medoid"] = (
        Field(
            "least_cut",
            description="Which grouping within the size bounds is taken. `least_cut`: the least cut, purely topological; balance comes from `size_bounds`. `near_optimal_balanced`: among the groupings within `near_optimal_tolerance` of the least cut, the one whose islands deviate least from the mean; with [0.5, 1.5] it fixed little at 5% (Spain 0.55-1.46) and at 10% glued electrically separate areas into one Spanish island (50% of its load outside its connected core), since candidates are checked for geographic contiguity only. `near_optimal_medoid`: the medoid of that set (the one moving the least load against the others); in Germany it halved how much load changes island between METIS seeds (5.6% vs 11.4%).",
        )
    )
    near_optimal_tolerance: float = Field(
        0.05,
        ge=0.0,
        description="Groupings within this share of the least cut count as equally good: `selection` picks the most balanced or the medoid among them, and either way the border uncertainty is reported, i.e. how much load they put in another island than the pick. In Germany the ten best groupings were within 2-6% of the least cut but put 10-27% of the load in other islands.",
    )
    metis_samples: int = Field(
        10000,
        description="METIS runs with perturbed edge weights per country. Germany: the least cut stopped improving between 3,000 and 5,000 runs; 10,000 take ~4 min.",
    )
    metis_imbalance: list[int] = Field(
        [5, 10, 30, 50, 100, 150, 200, 300, 500, 700, 1000],
        description="METIS ufactor values cycled through the runs: the heaviest part may be at most (1 + ufactor/1000) x the mean. Strict values give balanced but costlier cuts, loose ones cheap but uneven cuts; the size bounds then filter.",
    )
    metis_weight_perturbation: float = Field(
        0.5,
        ge=0.0,
        lt=1.0,
        description="After one unperturbed run per imbalance value, every link weight is scaled by a uniform random factor in [1 - this, 1 + this], so the runs explore different near-optimal cuts.",
    )
    seed: int = Field(
        0,
        description="Seed of the weight perturbation and of METIS, for reproducible islands.",
    )


class AroConfig(ConfigModel):
    """Configuration for top-level `aro` settings."""

    partition: AroPartitionConfig = Field(
        default_factory=AroPartitionConfig,
        description="Outage-island partition built on an OSM network with lower voltage levels.",
    )

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
    dc_losses: bool = Field(
        True,
        description="Split every DC link into a lossy forward link and a '-reversed' twin, as pypsa-eur's prepare_sector_network.lossy_bidirectional_links does for sector runs, with `sector.transmission_efficiency.DC` (default 2% static + 2.3% per 1000 km). Electricity-only builds never call it, so their DC links stay lossless -- and links from `transmission_projects` (new_links.csv carries no p_min_pu) stay ONE-DIRECTIONAL. The reversed twin has capital_cost 0; PyPSARO's add_bidirectional_link_constraints ties its p_nom to the forward link. On by default since 2026-10-20; off reproduces networks built before then.",
    )
    fix_cross_border_transmission: bool = Field(
        True,
        description="Make every line and DC link between two countries non-extendable at its input capacity (today + `transmission_projects`), while lines inside a country keep whatever `transmission_limit` gives them (e.g. extendable under `vopt`). The study's grid is exogenous at the borders, where the European pre-run is cut into country runs. No effect on single-country networks. Added 2026-10-20.",
    )
