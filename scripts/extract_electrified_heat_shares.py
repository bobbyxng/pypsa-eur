# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Extract per-country heat-pump / resistive shares from a solved sector-coupled network.

Fork-specific; not part of upstream PyPSA-Eur. Reads the solved heat reference run
(`europe-heat-reference.pypsa-eur.yaml` in PyPSARO) and writes a JSON whose `shares` block
is meant to be **copied by hand** into `aro.heat.shares`, like the district-heating
calibration: the config stays a static, reviewable artifact.

The metric is the share of building heat SUPPLY, `tech / (heat pump + resistive + boilers +
CHP + solar thermal)`, read with `n.statistics.supply` on the three building-heat bus
carriers. It never touches the demand side, so it is immune to whatever else draws from the
heat buses (agriculture heat, thermal-storage chargers, DAC). Three things it gets right,
each of which produced a wrong answer first (memory: pypsa-eur-sector-run-gotchas):

1. **Supply, not demand.** "heat-pump output / building heat demand" came out at 418% for
   Denmark, because the heat buses serve more than the building loads.
2. **No hand-rolled p0/p1.** Heat-pump links are wired heat -> electricity, resistive heaters
   electricity -> heat, so one sign convention for both silently gives negative heat-pump
   output. `n.statistics` handles the direction per link.
3. **Storage is not supply.** Water-tank / pit dischargers only shift heat in time and are
   excluded; PtX waste heat (electrolysis, fuel cell, Sabatier, ...) is reported but kept
   out of the denominator, matching how the DE/ES shares were first measured.

Outputs
-------
- `results/{run}/heat_shares/electrified_heat_shares_{horizon}.json`:
  - `shares`: {country: {heat_pump, resistive}}, ready for `aro.heat.shares`;
  - `default_shares`: the supply-weighted aggregate over all countries;
  - `heat_pump_fraction_of_electric`: HP / (HP + resistive), the split that survives when
    the electrified share is taken from literature instead (memory:
    heat-electrification-design);
  - `breakdown_twh`, `effective_cop`, and `warnings`.

Regression anchor: on the old `europe-pre-run` network this reproduces DE 0.845/0.040,
ES 0.946/0.054 and the EU aggregate 0.901/0.049.
"""

import json
import logging

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

HEAT_BUS_CARRIERS = ["rural heat", "urban decentral heat", "urban central heat"]
ELECTRIC_BUS_CARRIERS = ["AC", "low voltage"]

# Sanity band for the realised annual COP of heat pumps. The DE/ES reference values are
# 2.81 and 3.23; ground-source rural units push a country up, cold air-source ones down.
COP_RANGE = (2.0, 4.5)
PTX_WASTE_HEAT = (
    "H2 Electrolysis",
    "H2 Fuel Cell",
    "Haber-Bosch",
    "Sabatier",
    "Fischer-Tropsch",
    "methanolisation",
)


def classify(carrier: str) -> str:
    """Map a heat-supplying carrier to the category the metric uses."""
    if "heat pump" in carrier:
        return "heat_pump"
    if "resistive heater" in carrier:
        return "resistive"
    if "discharger" in carrier or "charger" in carrier:
        return "storage"
    if carrier.startswith(PTX_WASTE_HEAT) or carrier in PTX_WASTE_HEAT:
        return "ptx_waste_heat"
    if "solar thermal" in carrier:
        return "solar_thermal"
    # Boilers (gas, biomass, oil, H2) and every CHP: the non-electric supply.
    return "combustion"


def compute_shares(supply: pd.Series) -> dict:
    """
    Turn annual heat supply per (country, carrier) into the shares JSON.

    `supply` is in MWh, indexed by (country, carrier). Pure function so it can be tested
    without a solved network.
    """
    categories = supply.groupby(
        [
            supply.index.get_level_values(0),
            supply.index.get_level_values(1).map(classify),
        ]
    ).sum()
    table = categories.unstack(fill_value=0.0)
    for col in [
        "heat_pump",
        "resistive",
        "combustion",
        "solar_thermal",
        "ptx_waste_heat",
        "storage",
    ]:
        if col not in table:
            table[col] = 0.0

    denominator = table[["heat_pump", "resistive", "combustion", "solar_thermal"]].sum(
        axis=1
    )
    table = table[denominator > 0]
    denominator = denominator[denominator > 0]

    shares = {
        ct: {
            "heat_pump": round(float(table.at[ct, "heat_pump"] / denominator[ct]), 3),
            "resistive": round(float(table.at[ct, "resistive"] / denominator[ct]), 3),
        }
        for ct in table.index
    }
    total = table.sum()
    total_denominator = denominator.sum()
    default = {
        "heat_pump": round(float(total["heat_pump"] / total_denominator), 3),
        "resistive": round(float(total["resistive"] / total_denominator), 3),
    }
    electric = table["heat_pump"] + table["resistive"]
    hp_fraction = {
        ct: round(float(table.at[ct, "heat_pump"] / electric[ct]), 3)
        for ct in table.index
        if electric[ct] > 0
    }
    breakdown = (table / 1e6).round(3).to_dict(orient="index")
    return {
        "shares": shares,
        "default_shares": default,
        "heat_pump_fraction_of_electric": hp_fraction,
        "breakdown_twh": breakdown,
    }


def contamination_warnings(n: pypsa.Network, table: dict) -> list[str]:
    """Flag what the docstring's metric is not immune to, so a reader can judge."""
    warnings = []
    dac = n.links.index[n.links.carrier == "DAC"]
    if len(dac) and n.links_t.p0.reindex(columns=dac).abs().sum().sum() > 0:
        warnings.append(
            "DAC dispatches in this network: it draws heat from the same buses as building "
            "heat and shifts the supply mix (memory: pypsa-eur-sector-run-gotchas)."
        )
    if n.loads.carrier.str.contains("agriculture", na=False).any():
        warnings.append("Agriculture loads are present (sector.agriculture not off).")
    combustion = {
        ct: b["combustion"] for ct, b in table.items() if b.get("combustion", 0) > 0.05
    }
    if combustion:
        # Expected when biomass is on (biomass/biogas boilers and CHP); a red flag only if the
        # run was meant to be all-electric. Reported, not judged.
        warnings.append(
            "Non-electric (boiler/CHP) heat supply present, so shares sum below 1; use "
            f"heat_pump_fraction_of_electric for the electric split: {combustion} TWh."
        )
    return warnings


def effective_cop(n: pypsa.Network) -> dict:
    """Annual heat delivered per unit of electricity consumed by heat pumps, per country."""
    heat_pumps = n.links.carrier[
        n.links.carrier.str.contains("heat pump", na=False)
    ].unique()
    if len(heat_pumps) == 0:
        return {}
    heat = n.statistics.supply(
        bus_carrier=HEAT_BUS_CARRIERS, groupby=["country", "carrier"], nice_names=False
    )
    elec = n.statistics.withdrawal(
        bus_carrier=ELECTRIC_BUS_CARRIERS,
        groupby=["country", "carrier"],
        nice_names=False,
    )
    heat = heat[heat.index.get_level_values("carrier").isin(heat_pumps)]
    elec = elec[elec.index.get_level_values("carrier").isin(heat_pumps)]
    heat = heat.groupby(level="country").sum()
    elec = elec.groupby(level="country").sum()
    cop = (heat / elec).dropna()
    return {ct: round(float(v), 2) for ct, v in cop.items()}


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("extract_electrified_heat_share", horizon="2050")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    supply = n.statistics.supply(
        bus_carrier=HEAT_BUS_CARRIERS,
        groupby=["country", "carrier"],
        aggregate_across_components=True,
        nice_names=False,
    )
    if supply.empty:
        raise ValueError(
            "No heat supply on the building-heat buses -- this rule needs a solved "
            "sector-coupled network (sector.enabled and sector.heating true)."
        )

    result = compute_shares(supply)
    result["effective_cop"] = effective_cop(n)
    result["warnings"] = contamination_warnings(n, result["breakdown_twh"])

    for ct, cop in result["effective_cop"].items():
        if not COP_RANGE[0] <= cop <= COP_RANGE[1]:
            result["warnings"].append(
                f"{ct}: effective heat-pump COP {cop} outside {COP_RANGE}; check the "
                "heat-pump link direction before trusting its share."
            )
    for w in result["warnings"]:
        logger.warning(w)

    result["metric"] = (
        "share of building heat supply: tech / (heat pump + resistive + boilers + CHP + "
        "solar thermal); storage dischargers and PtX waste heat excluded"
    )
    result["network"] = snakemake.input.network

    with open(snakemake.output.shares, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    logger.info(
        f"Wrote shares for {len(result['shares'])} countries; aggregate "
        f"{result['default_shares']}."
    )
