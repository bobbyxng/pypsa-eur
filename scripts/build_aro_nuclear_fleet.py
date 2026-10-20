# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build the fixed 2050 nuclear fleet of PyPSARO's ARO runs as a custom powerplant list.

Fork-specific; not part of upstream PyPSA-Eur. The fleet is an exogenous input, not an
investment decision: no extendable nuclear, and every country's 2050 fleet is today's fleet
minus retirements plus CONFIRMED new build, decided unit by unit with the rules below. The
decision record (per-country status, primary sources, why each rule) is PyPSARO's
`memory/nuclear-treatment.md`; this script is its executable form.

Rules (adopted 2026-10-20):

- **E1** existing unit with an end date (licence or law): retires then if before 2050,
  unless an extension is formally committed (application, investment decision, signed
  agreement, law). Periodic 10-year licences (RO, BG) are not end dates; the stated
  technical life is.
- **E2** existing unit without an end date: the operator's stated operating horizon.
- **E3** no per-unit horizon at all (FR): the national planning trajectory, applied by
  scaling every unit's capacity (RTE "médiane", ~40 GW of the existing fleet in 2050).
- **N** new build: under construction, EPC signed, FID with financing secured, or EU
  state-aid approval for the project. Judged on the freeze date (the day the pre-run is
  built); projects one step short are written with `Status == "pending"` so a single
  `custom_powerplants` query switches them.

Why a custom list and not a filter on powerplantmatching (ppm): `build_powerplants`
APPENDS custom rows to the filtered ppm list, it never replaces them, and nothing after it
retires a unit by year (`DateOut` only acts inside `powerplants_filter`). So a run using
this file must also drop ppm's own nuclear rows (`powerplants_filter: ... and Fueltype !=
'Nuclear'`), and the file must hold the whole fleet. Kept units are copied from ppm
unchanged, coordinates included, so they map to exactly the buses they had before.

Availability: `add_electricity` caps nuclear at a per-country historical availability
(`conventional: nuclear: p_max_pu`, upstream's `data/nuclear_p_max_pu.csv`) and silently
leaves a country WITHOUT a row at p_max_pu = 1.0. A newcomer (PL) would thus be the most
available nuclear in Europe, which flatters exactly the island premium we measure. Every
fleet country missing from the table gets, per year column, the median over the run's
MODELLED countries that have a value for that year: a neutral value for a country without a
history (2013, our weather year: 0.860). Countries outside the run (UA) are excluded; modelled
countries without a 2050 fleet (DE, BE, ES, CH, SI, LT) still count, their load factors are
history of the same European regulatory environment.

Outputs
-------
- `data/aro_nuclear_fleet_2050.csv`: ppm columns plus `Status` (existing / new / pending)
  and `Rule`. Committed, so the fleet is reviewable row by row and frozen against later ppm
  releases.
- `data/aro_nuclear_p_max_pu.csv`: upstream's availability table plus the median rows,
  for `conventional: nuclear: p_max_pu`.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging

logger = logging.getLogger(__name__)

COUNTRY_CODES = {
    "Belgium": "BE",
    "Bulgaria": "BG",
    "Czechia": "CZ",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Hungary": "HU",
    "Italy": "IT",
    "Lithuania": "LT",
    "Netherlands": "NL",
    "Romania": "RO",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "United Kingdom": "GB",
}

# Existing units still running in 2050, by ppm name; every other ppm nuclear unit in these
# countries retires. Countries absent here (DE, IT, LT, BE, ES, SI, CH) keep nothing.
KEEP = {
    "BG": (["Kozloduy 6"], "E1: technical life to 2051; Kozloduy 5 ends 2049"),
    "CZ": (
        [
            "Dukovany 1",
            "Dukovany 2",
            "Dukovany 3",
            "Dukovany 4",
            "Temelin 1",
            "Temelin 2",
        ],
        "E2: ČEZ (Apr 2026) heads Dukovany for 80 y; Temelín into the 2060s",
    ),
    "FI": (
        ["Loviisa 1", "Loviisa 2", "Olkiluoto 3"],
        "E1: Loviisa licensed to end-2050; OL1/2 only to 2038",
    ),
    "GB": (
        ["Sizewell B"],
        "E1: extension to 2055 (heads of terms, Jul 2026); the AGR stations end 2030",
    ),
    "HU": (
        ["Paks 1", "Paks 2", "Paks 3", "Paks 4"],
        "E1: +20 y by parliament resolution 56/2022 (→ 2052-57)",
    ),
    "NL": (
        ["Borssele 1"],
        "E2: 2033 end date removed by law (Sep 2026); operator plans ~2053",
    ),
    "RO": (["Cernavoda 1"], "E1: refurbishment 2027-29, +30 y; Unit 2 not refurbished"),
    "SE": (
        [
            "Forsmark 1",
            "Forsmark 2",
            "Forsmark 3",
            "Oskarshamn 3",
            "Ringhals 3",
            "Ringhals 4",
        ],
        "E2: owners aim at 80 y",
    ),
    "SK": (
        ["Mochovce 1", "Mochovce 2", "Mochovce 3", "Mochovce 4"],
        "E2: Mochovce past 2050; Bohunice V2 ends 2044/45",
    ),
}

# E3: every operating French unit (no DateOut in ppm) is kept and scaled to this total.
FR_EXISTING_MW = 40_000.0
FR_RULE = "E3: RTE médiane extension trajectory (~40 GW in 2050), all units scaled"

# New build: (name, country, MW, first year, ppm unit giving the site coordinates or
# (lat, lon), status, rule). Unit ratings are the developers' figures.
NEW = [
    (
        "Hinkley Point C 1",
        "GB",
        1630.0,
        2030,
        "Hinkley Point C1",
        "new",
        "N: under construction",
    ),
    (
        "Hinkley Point C 2",
        "GB",
        1630.0,
        2031,
        "Hinkley Point C1",
        "new",
        "N: under construction",
    ),
    ("Sizewell C 1", "GB", 1630.0, 2039, "Sizewell B", "new", "N: FID Jul 2025"),
    ("Sizewell C 2", "GB", 1630.0, 2039, "Sizewell B", "new", "N: FID Jul 2025"),
    ("Dukovany 5", "CZ", 1000.0, 2036, "Dukovany 1", "new", "N: EPC signed Jun 2025"),
    ("Dukovany 6", "CZ", 1000.0, 2037, "Dukovany 1", "new", "N: EPC signed Jun 2025"),
    ("Paks 5", "HU", 1200.0, 2032, "Paks 1", "new", "N: under construction (Feb 2026)"),
    ("Paks 6", "HU", 1200.0, 2033, "Paks 1", "new", "N: under construction (Feb 2026)"),
    # Site point from Global Energy Monitor (approximate, the same for all three units).
    *[
        (
            f"Lubiatowo-Kopalino {i}",
            "PL",
            1250.0,
            2035 + i,
            (54.8036, 17.7881),
            "new",
            "N: EU state aid approved Dec 2025",
        )
        for i in (1, 2, 3)
    ],
    *[
        (
            f"{site} EPR2 {i}",
            "FR",
            1670.0,
            2038,
            ppm_site,
            "pending",
            "N: PPE3 decree; enters once EU state aid is approved",
        )
        for site, ppm_site in [
            ("Penly", "Penly 1"),
            ("Gravelines", "Gravelines 1"),
            ("Bugey", "Bugey 2"),
        ]
        for i in (1, 2)
    ],
]


def build_fleet(ppm: pd.DataFrame) -> pd.DataFrame:
    """
    Return the 2050 nuclear fleet from the raw ppm table (country names, not codes).
    """
    nuclear = ppm.query("Fueltype == 'Nuclear' and Country in @COUNTRY_CODES").copy()
    nuclear["Country"] = nuclear.Country.map(COUNTRY_CODES)
    by_name = nuclear.set_index("Name")
    if not by_name.index.is_unique:
        raise ValueError(
            "ppm nuclear unit names are not unique; selection by name breaks"
        )

    rows = []
    for country, (names, rule) in KEEP.items():
        missing = set(names) - set(by_name.index)
        if missing:
            raise ValueError(f"{country}: units not in ppm: {sorted(missing)}")
        kept = by_name.loc[names].reset_index()
        if not (kept.Country == country).all():
            raise ValueError(f"{country}: a kept unit belongs to another country")
        rows.append(kept.assign(Status="existing", Rule=rule))

    france = nuclear.query("Country == 'FR' and DateOut != DateOut").copy()
    france["Capacity"] *= FR_EXISTING_MW / france.Capacity.sum()
    rows.append(france.assign(Status="existing", Rule=FR_RULE))

    for name, country, mw, year, site, status, rule in NEW:
        lat, lon = by_name.loc[site, ["lat", "lon"]] if isinstance(site, str) else site
        rows.append(
            pd.DataFrame(
                [
                    {
                        "Name": name,
                        "Fueltype": "Nuclear",
                        "Technology": "Steam Turbine",
                        "Set": "PP",
                        "Country": country,
                        "Capacity": mw,
                        "DateIn": float(year),
                        "lat": lat,
                        "lon": lon,
                        "Status": status,
                        "Rule": rule,
                    }
                ]
            )
        )

    fleet = pd.concat(rows, ignore_index=True)
    # The fleet is valid for 2050 by construction; ppm's DateOut (e.g. Dukovany 2037,
    # Borssele 2033) would contradict the rules applied above, and nothing downstream reads
    # it for an overnight 2050 build -- clear it so the file states one thing.
    fleet["DateOut"] = float("nan")
    return fleet[[*ppm.columns, "Status", "Rule"]]


def fill_missing_availability(
    p_max_pu: pd.DataFrame, countries, modelled_countries
) -> pd.DataFrame:
    """
    Add a row for every country in `countries` without one: per-year median over the rows of
    `modelled_countries` (countries outside the run do not inform the value).
    """
    missing = sorted(set(countries) - set(p_max_pu.index))
    median = (
        p_max_pu[p_max_pu.index.isin(modelled_countries)].median(skipna=True).round(3)
    )
    added = pd.DataFrame(
        [median] * len(missing), index=pd.Index(missing, name=p_max_pu.index.name)
    )
    return pd.concat([p_max_pu, added])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_aro_nuclear_fleet")
    configure_logging(snakemake)

    ppm = pd.read_csv(snakemake.input.powerplants, index_col=0)
    fleet = build_fleet(ppm)

    totals = fleet.groupby(["Country", "Status"]).Capacity.sum().unstack(fill_value=0.0)
    logger.info("2050 nuclear fleet (MW):\n%s", totals.round(0).to_string())
    logger.info(
        "Total %.1f GW without pending units, %.1f GW with them",
        fleet.query("Status != 'pending'").Capacity.sum() / 1e3,
        fleet.Capacity.sum() / 1e3,
    )

    fleet.to_csv(snakemake.output.fleet, index=False)

    p_max_pu = pd.read_csv(snakemake.input.p_max_pu, index_col=0)
    filled = fill_missing_availability(
        p_max_pu, fleet.Country.unique(), snakemake.params.countries
    )
    for country in filled.index.difference(p_max_pu.index):
        logger.info(
            "Nuclear availability for %s set to the median: %s",
            country,
            filled.loc[country].to_dict(),
        )
    filled.to_csv(snakemake.output.p_max_pu, float_format="%.3f")
