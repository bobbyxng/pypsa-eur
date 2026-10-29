# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Extract the hourly actual output of the generation units of the configured carriers from
the ENTSO-E Transparency Platform's monthly files (actual generation output per generation
unit, 16.1.A, File Library), for one calendar year.

The monthly files cover every production type in Europe (about 270 MB each), so they are
read in chunks and filtered on the way. A unit is published under each area it belongs to
(Swedish reactors under the control area SE and the bidding zone SE3, German ones under a
control area and DE_LU), with identical values, so units are deduplicated by their EIC code.
Resolutions differ between units (15, 30 and 60 min); everything is averaged to hours.

Outputs
-------
``entsoe_actual_generation_{year}.csv``: one row per unit and hour with ``time`` (UTC),
``unit_code``, ``unit_name``, ``plant_type``, ``country`` and ``output`` [MW].
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_conventional_unavailability import ENTSOE_PLANT_TYPES

logger = logging.getLogger(__name__)

COLUMNS = {
    "DateTime(UTC)": "time",
    "AreaMapCode": "area",
    "GenerationUnitCode": "unit_code",
    "GenerationUnitName": "unit_name",
    "GenerationUnitType": "plant_type",
    "ActualGenerationOutput[MW]": "output",
}


def read_month(path: str, types: set[str]) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(
        path, sep="\t", usecols=list(COLUMNS), chunksize=500_000, quoting=3
    ):
        chunk = chunk[chunk["GenerationUnitType"].isin(types)].rename(columns=COLUMNS)
        if not chunk.empty:
            frames.append(chunk)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(COLUMNS.values()))
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_entsoe_actual_generation", year="2024")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    types = {
        t for c in snakemake.params.carriers for t in ENTSOE_PLANT_TYPES.get(c, [])
    }
    frames = []
    for path in snakemake.input:
        df = read_month(path, types)
        logger.info(f"{path}: {len(df)} rows of {sorted(types)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df.time, utc=True)
    df["country"] = df.area.str[:2]
    df = df.drop_duplicates(["unit_code", "time"])
    df["time"] = df.time.dt.floor("h")
    df = (
        df.groupby(["unit_code", "time"], as_index=False)
        .agg(
            unit_name=("unit_name", "first"),
            plant_type=("plant_type", "first"),
            country=("country", "first"),
            output=("output", "mean"),
        )
        .sort_values(["unit_code", "time"])
    )
    df.to_csv(snakemake.output[0], index=False)
    logger.info(f"{df.unit_code.nunique()} units, {len(df)} unit-hours.")
