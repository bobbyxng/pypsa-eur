# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build the availability of conventional generators from outages reported on the ENTSO-E
Transparency Platform (unavailability of production and generation units, 15.1.A-D).

The cleaning is shared by both methods:

1. keep the configured carriers and outage types (planned maintenance, forced outages);
2. drop cancelled and withdrawn notices and keep only the latest revision of each outage;
3. turn every notice into unavailable MW (nominal minus available power) and merge
   overlapping notices of one unit into non-overlapping segments, clipped at its nominal
   power;
4. combine the two reporting levels per plant (production unit): at any time the plant's
   unavailable MW is the larger of its production-unit notices and the sum over its
   generation units. Plants mix the levels (Ringhals 3 and 4 report refuelling for the
   production unit and short trips per generator), so dropping either level loses outages
   and adding them counts some twice.

Then, with ``conventional: unavailability: method``,

- ``climatology``: unavailable energy per country and calendar month divided by the
  country's installed capacity (14.1.B) gives a monthly availability per year, which is
  averaged per calendar month over ``reference_periods``. Applied to any modelled year.
- ``historical``: the availability of each modelled snapshot, per power plant where an
  ENTSO-E unit maps to one (codes stored by powerplantmatching, completed by
  ``data/entsoe_unit_mapping.csv``) and per country otherwise.

Countries without outage data get the median over the countries with data
(``fallback: median``) or keep their static ``p_max_pu`` (``fallback: static``).

Outputs
-------
``conventional_unavailability.csv`` in long format with columns ``carrier``, ``country``,
``unit`` (power plant index for plant-level rows, empty otherwise), ``period`` (calendar
month 1-12 for ``climatology``, snapshot for ``historical``), ``p_max_pu``, ``planned``
and ``forced`` (the availability lost to each outage type, ``climatology`` only) and
``source`` (``outages`` or ``median``).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

# pypsa-eur carrier -> powerplantmatching `Fueltype` (to decide which countries to retrieve)
PPM_FUELTYPES = {
    "nuclear": ["Nuclear"],
    "CCGT": ["Natural Gas"],
    "OCGT": ["Natural Gas"],
    "coal": ["Hard Coal"],
    "lignite": ["Lignite"],
    "oil": ["Oil"],
    "biomass": ["Solid Biomass"],
}

# pypsa-eur carrier -> ENTSO-E production type (`plant_type` in the outage data)
ENTSOE_PLANT_TYPES = {
    "nuclear": ["Nuclear"],
    "CCGT": ["Fossil Gas"],
    "OCGT": ["Fossil Gas"],
    "coal": ["Fossil Hard coal"],
    "lignite": ["Fossil Brown coal/Lignite"],
    "oil": ["Fossil Oil"],
    "biomass": ["Biomass"],
}

OUTAGE_TYPES = {"Planned maintenance": "planned", "Unplanned outage": "forced"}
VOID_STATUSES = {"Cancelled", "Withdrawn", "A09", "A13"}
FIRST_YEAR = 2015  # start of the ENTSO-E Transparency Platform


def reference_years(periods: dict, last_year: int | None = None) -> list[int]:
    """Calendar years to retrieve for `reference_periods` (fully excluded years dropped)."""
    if last_year is None:
        last_year = pd.Timestamp.now().year - 1
    include = periods.get("include") or [
        str(y) for y in range(FIRST_YEAR, last_year + 1)
    ]
    excluded = {p for p in periods.get("exclude", []) if len(p) == 4}
    return sorted({int(p[:4]) for p in include if p[:4] not in excluded})


def reference_months(periods: dict, years: list[int]) -> pd.PeriodIndex:
    """Year-months selected by `reference_periods`, exclude winning over include."""
    include = periods.get("include") or [str(y) for y in years]
    months = set()
    for p in include:
        months |= set(
            pd.period_range(p, p, freq="M")
            if len(p) == 7
            else pd.period_range(f"{p}-01", f"{p}-12", freq="M")
        )
    for p in periods.get("exclude", []):
        months -= set(
            pd.period_range(p, p, freq="M")
            if len(p) == 7
            else pd.period_range(f"{p}-01", f"{p}-12", freq="M")
        )
    return pd.PeriodIndex(sorted(months), freq="M")


def load_outages(files: dict[str, list[str]]) -> pd.DataFrame:
    frames = []
    for country, paths in files.items():
        for path in paths:
            df = pd.read_csv(path, parse_dates=["start", "end"])
            if not df.empty:
                frames.append(df.assign(country=country))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ["start", "end"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def clean_outages(
    df: pd.DataFrame, carriers: list[str], outage_types: list[str]
) -> pd.DataFrame:
    """Steps 1-3 of the module docstring; returns one row per notice with unavailable MW."""
    types = {t: c for c in carriers for t in ENTSOE_PLANT_TYPES.get(c, [])}
    df = df[df.plant_type.isin(types)].assign(carrier=lambda d: d.plant_type.map(types))
    df = df.assign(outage_type=df.businesstype.map(OUTAGE_TYPES))
    df = df[df.outage_type.isin(outage_types)]
    df = df[~df.docstatus.isin(VOID_STATUSES)]

    latest = df.groupby("mrid").revision.transform("max")
    df = df[df.revision == latest]
    # Operators republish the same outage under new mRIDs (Paks has five identical active
    # notices for one 2015 refuelling of one generator); identical content is one event.
    df = df.drop_duplicates(
        [
            "level",
            "production_resource_id",
            "production_resource_psr_name",
            "start",
            "end",
            "avail_qty",
            "businesstype",
        ]
    )

    df = df.assign(
        unit=np.where(
            df.level == "generation",
            df.production_resource_id
            + "|"
            + df.production_resource_psr_name.fillna(""),
            df.production_resource_id,
        ),
        unavailable=(df.nominal_power - df.avail_qty).clip(lower=0),
    )
    df["unavailable"] = df.unavailable.clip(upper=df.nominal_power)
    return df[df.unavailable > 0]


def merge_overlaps(
    df: pd.DataFrame, by: list[str], per_plant: bool = True
) -> pd.DataFrame:
    """
    Non-overlapping segments of unavailable MW per unit (and `by`), clipped at nominal power.

    Two notices of one unit covering the same hours (e.g. a forced outage during planned
    maintenance) must not count twice: their unavailable MW is summed per segment between
    consecutive start/end times and capped at the unit's nominal power.
    """
    keys = ["unit", *by]
    events = pd.concat(
        [
            df[[*keys, "start", "unavailable"]].rename(columns={"start": "time"}),
            df[[*keys, "end", "unavailable"]]
            .rename(columns={"end": "time"})
            .assign(unavailable=lambda d: -d.unavailable),
        ]
    )
    events = events.groupby([*keys, "time"], as_index=False).unavailable.sum()
    events = events.sort_values([*keys, "time"])
    events["level"] = events.groupby(keys).unavailable.cumsum().round(6)
    events["until"] = events.groupby(keys).time.shift(-1)
    segments = events.dropna(subset=["until"])
    segments = segments[segments.level > 0]

    nominal = df.groupby("unit").nominal_power.max()
    segments = segments.assign(
        unavailable=np.minimum(segments.level, segments.unit.map(nominal))
    )
    segments = segments.rename(columns={"time": "start", "until": "end"})[
        [*keys, "start", "end", "unavailable"]
    ]
    return combine_levels(segments, by) if per_plant else segments


def combine_levels(segments: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """
    Step 4: per plant, the larger of production-unit and summed generation-unit MW.

    Unit keys are "<plant>" for production units and "<plant>|<generator>" for generation
    units, so the plant and the level follow from the key.
    """
    seg = segments.assign(
        plant=segments.unit.str.split("|").str[0],
        level=np.where(
            segments.unit.str.contains("|", regex=False), "generation", "production"
        ),
    )
    keys = ["plant", *by]
    events = pd.concat(
        [
            seg[[*keys, "level", "start", "unavailable"]].rename(
                columns={"start": "time"}
            ),
            seg[[*keys, "level", "end", "unavailable"]]
            .rename(columns={"end": "time"})
            .assign(unavailable=lambda d: -d.unavailable),
        ]
    )
    events = (
        events.pivot_table(
            index=[*keys, "time"], columns="level", values="unavailable", aggfunc="sum"
        )
        .reindex(columns=["generation", "production"])
        .fillna(0)
        .sort_index()
    )
    levels = events.groupby(level=keys).cumsum().round(6)
    out = levels.max(axis=1).rename("unavailable").reset_index()
    out["end"] = out.groupby(keys).time.shift(-1)
    out = out.dropna(subset=["end"])
    out = out[out.unavailable > 0].rename(columns={"time": "start", "plant": "unit"})
    return out[["unit", *by, "start", "end", "unavailable"]]


def monthly_energy(segments: pd.DataFrame, by: list[str]) -> pd.Series:
    """Unavailable MWh per `by` and calendar year-month, splitting segments at month ends."""
    seg = segments.assign(
        first=segments.start.dt.tz_localize(None).dt.to_period("M"),
        last=(segments.end - pd.Timedelta("1ns"))
        .dt.tz_localize(None)
        .dt.to_period("M"),
    )
    seg = seg.assign(
        month=[
            pd.period_range(f, l, freq="M") for f, l in zip(seg["first"], seg["last"])
        ]
    ).explode("month")
    month_start = seg.month.dt.start_time.dt.tz_localize("UTC")
    month_end = (seg.month + 1).dt.start_time.dt.tz_localize("UTC")
    hours = (
        np.minimum(seg.end, month_end) - np.maximum(seg.start, month_start)
    ).dt.total_seconds() / 3600
    return (seg.unavailable * hours).groupby([*[seg[b] for b in by], seg.month]).sum()


def hourly_segments(segments: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean unavailable MW per hour (and `keys`) from segments, by overlap with each hour."""
    start = segments.start.dt.floor("h")
    n = ((segments.end - start).dt.total_seconds() // 3600).astype(int) + 1
    idx = np.repeat(segments.index.values, n.values)
    offset = (
        np.concatenate([np.arange(k) for k in n.values])
        if len(n)
        else np.array([], int)
    )
    hours = start.loc[idx].values + offset.astype("timedelta64[h]")
    seg = segments.loc[idx]
    hour = pd.DatetimeIndex(hours, tz="UTC")
    lo = np.maximum(seg.start.values, hour.values)
    hi = np.minimum(seg.end.values, (hour + pd.Timedelta("1h")).values)
    frac = np.clip((hi - lo) / np.timedelta64(1, "h"), 0, None)
    out = pd.DataFrame(
        {
            **{k: seg[k].values for k in keys},
            "hour": hour,
            "lost": seg.unavailable.values * frac,
        }
    )
    out = out[out.lost > 0]
    return out.groupby([*keys, "hour"], as_index=False).lost.sum()


# A country-year counts only if this share of its reported loss falls into plant-hours
# with output data: where data goes missing during outages (British units 2016-2021), the
# remaining hours over-represent the units that ran, and the year would read as too
# available. A country-month counts only if this share of its capacity has output data.
MIN_VERIFIED_LOSS = 0.9
MIN_VERIFIED_CAPACITY = 0.5
# Reported loss above the static table's by more than this is a reporting error.
MAX_COVERAGE = 1.1


def lost_energy(
    outages: pd.DataFrame, by: list[str], headroom: pd.Series | None = None
) -> pd.Series:
    """
    Unavailable MWh per `by` and calendar year-month.

    With `headroom` (MW a plant could still have produced, its capacity minus its actual
    output, per plant and hour), only plant-hours with output data count, and in each the
    reported loss is capped by the headroom: a plant cannot have lost more than it did not
    produce. This catches notices left at their planned end after the plant returned early
    (Leibstadt 2019) and a reporting level that overstates the other (Paks reports its
    outages per plant and per generator, and the plant notices double the loss).
    """
    segments = merge_overlaps(outages, by)
    if headroom is None:
        return monthly_energy(segments, by)
    hourly = hourly_segments(segments, ["unit", *by])
    room = headroom.reindex(
        pd.MultiIndex.from_arrays([hourly.unit, hourly.hour])
    ).to_numpy(dtype=float)
    hourly = hourly[~np.isnan(room)].assign(
        lost=np.minimum(hourly.lost, room)[~np.isnan(room)]
    )
    hourly["month"] = hourly.hour.dt.tz_localize(None).dt.to_period("M")
    return hourly.groupby([*by, "month"]).lost.sum()


def verified_loss_share(outages: pd.DataFrame, headroom: pd.Series) -> pd.Series:
    """Share of reported loss (before the output cap) in plant-hours with output data, per year."""
    hourly = hourly_segments(
        merge_overlaps(outages, ["carrier", "country"]), ["unit", "carrier", "country"]
    )
    verified = pd.MultiIndex.from_arrays([hourly.unit, hourly.hour]).isin(
        headroom.index
    )
    year = hourly.hour.dt.year.rename("year")
    total = hourly.groupby(["carrier", "country", year]).lost.sum()
    return (
        hourly[verified].groupby(["carrier", "country", year[verified]]).lost.sum()
        / total
    ).reindex(total.index, fill_value=0)


def plant_headroom(
    generation: pd.DataFrame, outages: pd.DataFrame, units: pd.DataFrame
) -> pd.Series:
    """
    Capacity minus actual output per plant (production unit) and hour, clipped at zero.

    Generation units map to plants by (1) the generator name reported in the outage
    notices, (2) a name equal to the plant's, or (3) a name starting with the plant's
    ("Forsmark block 2 G21" -> "Forsmark block 2"), within one country. Plants with any
    unmapped hour of output data get no bound in that hour.
    """
    named = outages[outages.level == "generation"].drop_duplicates(
        ["country", "production_resource_psr_name"]
    )
    by_generator = dict(
        zip(
            zip(named.country, named.production_resource_psr_name),
            named.production_resource_id,
        )
    )
    plants = units.drop_duplicates("production_resource_id")
    by_name = dict(
        zip(
            zip(plants.country, plants.production_resource_name.str.strip()),
            plants.production_resource_id,
        )
    )

    def plant_of(country, name):
        if (country, name) in by_generator:
            return by_generator[(country, name)]
        if (country, name.strip()) in by_name:
            return by_name[(country, name.strip())]
        prefixes = [
            (len(n), pid)
            for (c, n), pid in by_name.items()
            if c == country and name.startswith(n + " ")
        ]
        return max(prefixes)[1] if prefixes else None

    names = generation.drop_duplicates("unit_code")
    mapping = {
        code: plant_of(c, n)
        for code, c, n in zip(names.unit_code, names.country, names.unit_name)
    }
    gen = generation.assign(plant=generation.unit_code.map(mapping)).dropna(
        subset=["plant"]
    )
    unmapped = names[names.unit_code.map(mapping).isna()]
    if len(unmapped):
        logger.info(
            f"Output of {len(unmapped)} generation units not mapped to a plant: {sorted(unmapped.unit_name)[:20]}"
        )
    output = gen.groupby(["plant", "time"]).output.sum()
    cap = units.set_index(["production_resource_id", "year"]).installed_capacity
    years = output.index.get_level_values("time").year
    capacity = cap.reindex(
        pd.MultiIndex.from_arrays([output.index.get_level_values("plant"), years])
    ).to_numpy()
    headroom = pd.Series(
        np.clip(capacity - output.to_numpy(), 0, None), index=output.index
    )
    headroom.index.names = ["unit", "hour"]
    return headroom.dropna()


def unit_capacity(files: dict[str, list[str]], carriers: list[str]) -> pd.DataFrame:
    """Installed MW per production unit and year from 14.1.B (`<year>_capacity.csv`)."""
    types = {t: c for c in carriers for t in ENTSOE_PLANT_TYPES.get(c, [])}
    frames = []
    for country, paths in files.items():
        for path in paths:
            cap = pd.read_csv(path)
            cap = cap[cap.plant_type.isin(types)]
            frames.append(
                cap.assign(
                    carrier=cap.plant_type.map(types),
                    country=country,
                    year=int(Path(path).name[:4]),
                )
            )
    columns = [
        "carrier",
        "country",
        "production_resource_id",
        "production_resource_name",
        "year",
        "installed_capacity",
    ]
    return (
        pd.concat(frames, ignore_index=True)[columns]
        if frames
        else pd.DataFrame(columns=columns)
    )


def apply_closures(
    outages: pd.DataFrame, units: pd.DataFrame, closures: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove closed units from the outages, and outages of units not listed that year.

    A closed unit still listed on the platform (the British list has not been updated
    since 2021, so Dungeness B, Hinkley Point B and Hunterston B appear as year-long
    outages) would count as lost output; a closed unit only absent from the capacity list
    (Fessenheim from 2021) would still add its notices. So notices end at the unit's
    closure date (`closures`: production unit -> last day of operation, from
    `data/entsoe_unit_closures.csv`), and notices of a unit missing from that year's
    capacity list are dropped.
    """
    closures = pd.to_datetime(closures, utc=True)
    date_out = pd.to_datetime(
        closures.reindex(outages.production_resource_id).to_numpy(), utc=True
    )
    date_out = pd.Series(date_out, index=outages.index)
    out = outages.assign(
        end=outages.end.where(date_out.isna() | (outages.end <= date_out), date_out)
    )
    out = out[out.end > out.start]
    listed = set(zip(units.production_resource_id, units.year))
    out = out[
        [
            (u, y) in listed
            for u, y in zip(out.production_resource_id, out.start.dt.year)
        ]
    ]
    return out, units


def monthly_capacity(units: pd.DataFrame, closures: pd.Series) -> pd.Series:
    """
    Capacity energy (MWh) per carrier, country and calendar year-month: each unit's
    installed MW times its hours before closure in that month.
    """
    closures = pd.to_datetime(closures, utc=True)
    rows = []
    for unit in units.itertuples():
        date_out = closures.get(unit.production_resource_id)
        for month in pd.period_range(f"{unit.year}-01", f"{unit.year}-12", freq="M"):
            start = month.start_time.tz_localize("UTC")
            end = (month + 1).start_time.tz_localize("UTC")
            if pd.notna(date_out):
                end = min(end, date_out)
            hours = max((end - start).total_seconds() / 3600, 0)
            rows.append(
                (unit.carrier, unit.country, month, unit.installed_capacity * hours)
            )
    df = pd.DataFrame(rows, columns=["carrier", "country", "month", "capacity"])
    return df.groupby(["carrier", "country", "month"]).capacity.sum()


def verified_capacity(
    units: pd.DataFrame, closures: pd.Series, headroom: pd.Series
) -> pd.Series:
    """Capacity energy (MWh) per carrier, country and year-month in hours with output data."""
    closures = pd.to_datetime(closures, utc=True)
    hours = headroom.index.to_frame(index=False)
    date_out = pd.Series(
        pd.to_datetime(closures.reindex(hours.unit).to_numpy(), utc=True),
        index=hours.index,
    )
    hours = hours[date_out.isna() | (hours.hour < date_out)]
    hours = hours.assign(
        year=hours.hour.dt.year, month=hours.hour.dt.tz_localize(None).dt.to_period("M")
    )
    count = (
        hours.groupby(["unit", "year", "month"]).size().rename("hours").reset_index()
    )
    count = count.merge(
        units.rename(columns={"production_resource_id": "unit"})[
            ["unit", "year", "carrier", "country", "installed_capacity"]
        ],
        on=["unit", "year"],
    )
    count["capacity"] = count.hours * count.installed_capacity
    return count.groupby(["carrier", "country", "month"]).capacity.sum()


def reported_years(
    outages: pd.DataFrame,
    capacity: pd.Series,
    static: dict[str, str],
    min_coverage: float | None,
    headroom: pd.Series | None = None,
) -> tuple[pd.MultiIndex, pd.DataFrame]:
    """
    Country-years reported completely enough to use.

    Reporting starts late or is patchy in some countries (Slovak nuclear only from 2021,
    Slovenia about one notice a year), and an unreported year would read as full
    availability. A country-year is kept if it has at least one outage and, where the
    carrier has a static availability table (`conventional: <carrier>: p_max_pu`, IAEA PRIS
    energy availability factors for nuclear), the reported loss is between `min_coverage`
    and `MAX_COVERAGE` of the loss the table implies. A ratio, not a difference: with
    availabilities around 0.9, even a year without any report is only 0.1 off the table.
    With `headroom`, `capacity` must be the capacity in hours with output data, and at
    least `MIN_VERIFIED_LOSS` of the reported loss must fall into such hours.
    """

    def annual(series):
        out = series.groupby(
            [
                series.index.get_level_values(0),
                series.index.get_level_values(1),
                series.index.get_level_values(-1).year,
            ]
        ).sum()
        out.index.names = ["carrier", "country", "year"]
        return out

    lost = annual(lost_energy(outages, ["carrier", "country"], headroom))
    possible = annual(capacity)
    check = pd.DataFrame(
        {"capacity": possible / 8760, "lost": lost.reindex(possible.index)}
    )
    check = check[check.capacity > 0]
    check["availability"] = 1 - check.lost / possible.reindex(check.index)
    check["static"] = np.nan
    for carrier, fn in static.items():
        table = pd.read_csv(fn, index_col=0)
        table.columns = table.columns.astype(int)
        for c, k, y in check.index:
            if c == carrier and k in table.index and y in table.columns:
                check.loc[(c, k, y), "static"] = table.at[k, y]
    check["coverage"] = (1 - check.availability) / (1 - check.static)
    keep = check.lost > 0
    if min_coverage is not None:
        keep &= check.static.isna() | check.coverage.between(min_coverage, MAX_COVERAGE)
    if headroom is not None:
        check["verified_loss"] = verified_loss_share(outages, headroom).reindex(
            check.index
        )
        keep &= check.verified_loss >= MIN_VERIFIED_LOSS
    check["used"] = keep
    return check.index[keep], check


def climatology(
    outages: pd.DataFrame,
    capacity: pd.Series,
    months: pd.PeriodIndex,
    headroom: pd.Series | None = None,
    total_capacity: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Availability per carrier, country and calendar month, averaged over `months`.
    `capacity` is the capacity energy per carrier, country and year-month (in hours with
    output data when `headroom` is given; a month then counts only if that is at least
    `MIN_VERIFIED_CAPACITY` of `total_capacity`).
    """
    total = lost_energy(outages, ["carrier", "country"], headroom)
    per_type = lost_energy(outages, ["carrier", "country", "outage_type"]).unstack(
        "outage_type"
    )
    # The output bound applies to the plant's total loss; split what remains by type in
    # proportion to the reported loss of each type.
    scale = (total / per_type.sum(axis=1).reindex(total.index)).fillna(0)
    per_type = per_type.reindex(total.index).mul(scale, axis=0)

    cap = capacity[capacity.index.get_level_values("month").isin(months)]
    cap = cap[cap > 0]
    if total_capacity is not None:
        share = cap / total_capacity.reindex(cap.index)
        cap = cap[share >= MIN_VERIFIED_CAPACITY]
    lost = pd.DataFrame(
        {
            "p_max_pu": total.reindex(cap.index, fill_value=0),
            **{
                t: per_type.get(t, pd.Series(dtype=float)).reindex(
                    cap.index, fill_value=0
                )
                for t in ["planned", "forced"]
            },
        }
    ).div(cap, axis=0)
    monthly = lost.assign(p_max_pu=1 - lost.p_max_pu).clip(0, 1)
    monthly["period"] = monthly.index.get_level_values("month").month
    clim = monthly.groupby(["carrier", "country", "period"]).mean()
    clim["years"] = monthly.groupby(["carrier", "country", "period"]).size()
    return clim.reset_index()


def historical(
    outages: pd.DataFrame,
    capacity: pd.Series,
    snapshots: pd.DatetimeIndex,
    plant_units: pd.DataFrame,
) -> pd.DataFrame:
    """
    Availability per snapshot, per power plant where mapped (`plant_units`: unit -> plant
    index and the unit's share of that plant) and per country otherwise.
    """
    segments = merge_overlaps(outages, ["carrier", "country"])
    unit_segments = merge_overlaps(outages, ["carrier", "country"], per_plant=False)
    snaps = pd.DatetimeIndex(snapshots).tz_localize("UTC")
    step = (
        pd.Series(snaps[1:] - snaps[:-1]).mode().iloc[0]
        if len(snaps) > 1
        else pd.Timedelta("1h")
    )
    bounds = snaps.append(pd.DatetimeIndex([snaps[-1] + step]))

    def averaged(seg):
        # Mean unavailable MW over each snapshot's interval.
        out = np.zeros(len(snaps))
        for start, end, mw in zip(seg.start, seg.end, seg.unavailable):
            lo, hi = np.maximum(bounds[:-1], start), np.minimum(bounds[1:], end)
            overlap = np.clip((hi - lo).total_seconds(), 0, None)
            out += mw * overlap / step.total_seconds()
        return out

    rows = []
    for (carrier, country), seg in segments.groupby(["carrier", "country"]):
        year = capacity[
            (capacity.index.get_level_values("carrier") == carrier)
            & (capacity.index.get_level_values("country") == country)
            & (capacity.index.get_level_values("month").year == snaps[0].year)
        ]
        cap = (
            year.sum() / (year.index.get_level_values("month").days_in_month.sum() * 24)
            if len(year)
            else np.nan
        )
        if not cap > 0:
            continue
        p = np.clip(1 - averaged(seg) / cap, 0, 1)
        rows.append(
            pd.DataFrame(
                {
                    "carrier": carrier,
                    "country": country,
                    "unit": "",
                    "period": snapshots,
                    "p_max_pu": p,
                }
            )
        )
    mapped = plant_units[plant_units.unit.isin(unit_segments.unit)]
    for plant, units in mapped.groupby("plant"):
        seg = unit_segments[unit_segments.unit.isin(units.unit)]
        nominal = (
            outages[outages.unit.isin(units.unit)].groupby("unit").nominal_power.max()
        )
        p = np.clip(1 - averaged(seg) / nominal.sum(), 0, 1)
        first = seg.iloc[0]
        rows.append(
            pd.DataFrame(
                {
                    "carrier": first.carrier,
                    "country": first.country,
                    "unit": str(plant),
                    "period": snapshots,
                    "p_max_pu": p,
                }
            )
        )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["carrier", "country", "unit", "period", "p_max_pu"])
    )


def map_units_to_plants(
    outages: pd.DataFrame, ppl: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """
    ENTSO-E units -> power plant index of `ppl` (the clustered powerplantmatching list).

    Two layers: the curated `mapping` (ENTSO-E production unit code and, for generation
    units, the reported unit name -> GEM id of the powerplantmatching unit), then the
    ENTSO-E codes powerplantmatching stores in `projectID`, used only where a code belongs
    to exactly one power plant (a code shared by several reactors of one site cannot say
    which reactor an outage hit).
    """
    project = ppl.projectID.fillna("")
    gem = project.str.extract(r"'GEM': \{'(G\d+)'")[0]
    plant_of_gem = pd.Series(gem.index, index=gem.values).dropna()
    plant_of_gem = plant_of_gem[~plant_of_gem.index.duplicated(keep=False)]

    units = outages.drop_duplicates("unit")[
        ["unit", "production_resource_id", "production_resource_psr_name", "level"]
    ]
    rows = []
    if not mapping.empty:
        key = mapping.production_resource_id + "|" + mapping.psr_name.fillna("")
        key = key.where(mapping.psr_name.notna(), mapping.production_resource_id)
        curated = pd.DataFrame(
            {"unit": key, "plant": mapping.gem_id.map(plant_of_gem)}
        ).dropna()
        rows.append(curated)
    done = set(rows[0].unit) if rows else set()

    codes = project.str.extractall(r"'ENTSOE': \{([^}]*)\}")[0].str.findall(
        r"'([^']+)'"
    )
    code_rows = [(code, idx[0]) for idx, found in codes.items() for code in found]
    plant_of_code = pd.DataFrame(code_rows, columns=["code", "plant"]).drop_duplicates()
    plant_of_code = (
        plant_of_code[~plant_of_code.code.duplicated(keep=False)]
        .set_index("code")
        .plant
    )
    auto = units[~units.unit.isin(done)]
    auto = pd.DataFrame(
        {"unit": auto.unit, "plant": auto.production_resource_id.map(plant_of_code)}
    ).dropna()
    # A plant code that several generation units report under can only be mapped when
    # powerplantmatching also has that site as one unit.
    rows.append(auto)
    out = pd.concat(rows, ignore_index=True)
    out["plant"] = out.plant.astype(str)
    return out


def add_fallback(
    df: pd.DataFrame, countries: list[str], carriers: list[str], fallback: str
) -> pd.DataFrame:
    """Median over the countries with data for countries without (`fallback: median`)."""
    df = df.assign(source="outages")
    if fallback != "median" or df.empty:
        return df
    national = df[df.unit.fillna("") == ""] if "unit" in df else df
    median = national.groupby(["carrier", "period"], as_index=False).p_max_pu.median()
    extra = []
    for carrier in carriers:
        have = set(national.loc[national.carrier == carrier, "country"])
        for country in sorted(set(countries) - have):
            extra.append(
                median[median.carrier == carrier].assign(
                    country=country, source="median"
                )
            )
    return pd.concat([df, *extra], ignore_index=True) if extra else df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_conventional_unavailability")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params = snakemake.params.unavailability
    carriers, method = params["carriers"], params["method"]
    countries, years = snakemake.params.countries, snakemake.params.years

    def per_country(kind):
        return {
            c: [f for f in snakemake.input[kind] if f"/{c}/" in f] for c in countries
        }

    outages = clean_outages(
        load_outages(per_country("outages")), carriers, params["outage_types"]
    )
    units = unit_capacity(per_country("capacity"), carriers)
    closures = pd.read_csv(snakemake.input.closures, comment="#")
    closures = pd.Series(
        pd.to_datetime(closures.date_out, utc=True).values,
        index=closures.production_resource_id,
    )
    outages, units = apply_closures(outages, units, closures)
    capacity = monthly_capacity(units, closures)
    generation = pd.concat(
        [pd.read_csv(f, parse_dates=["time"]) for f in snakemake.input.generation],
        ignore_index=True,
    )
    generation["time"] = pd.to_datetime(generation.time, utc=True)
    headroom = plant_headroom(generation, outages, units)
    logger.info(
        f"Output data for {headroom.index.get_level_values('unit').nunique()} plants bounds "
        "the reported losses."
    )
    logger.info(
        f"{len(outages)} outage notices of {outages.unit.nunique()} units in "
        f"{outages.country.nunique()} countries; capacity for {len(capacity)} country-months."
    )

    static = {
        c: fn
        for c, fn in snakemake.params.static.items()
        if str(fn).startswith("data/")
    }
    total_capacity = capacity
    capacity = verified_capacity(units, closures, headroom)
    used, check = reported_years(
        outages, capacity, static, params["min_coverage"], headroom
    )
    with pd.option_context("display.width", 200, "display.max_rows", None):
        logger.info(f"Reporting check per country-year:\n{check.round(3)}")
    month_year = pd.MultiIndex.from_arrays(
        [
            capacity.index.get_level_values("carrier"),
            capacity.index.get_level_values("country"),
            capacity.index.get_level_values("month").year,
        ]
    )
    capacity = capacity[month_year.isin(used)]

    if method == "climatology":
        months = reference_months(params["reference_periods"], years)
        df = climatology(outages, capacity, months, headroom, total_capacity).assign(
            unit=""
        )
    else:
        snapshots = pd.DatetimeIndex(snakemake.params.snapshots)
        ppl = pd.read_csv(snakemake.input.powerplants, index_col=0)
        mapping = pd.read_csv(snakemake.input.unit_mapping, comment="#")
        plant_units = map_units_to_plants(outages, ppl, mapping)
        nominal = outages.groupby("unit").nominal_power.max()
        mapped = (
            nominal[nominal.index.isin(plant_units.unit)]
            .groupby(outages.drop_duplicates("unit").set_index("unit").country)
            .sum()
            / nominal.groupby(
                outages.drop_duplicates("unit").set_index("unit").country
            ).sum()
        )
        logger.info(
            f"Share of reporting capacity mapped to power plants:\n{mapped.round(2)}"
        )
        df = historical(outages, capacity, snapshots, plant_units)

    df = add_fallback(df, countries, carriers, params["fallback"])
    df.to_csv(snakemake.output[0], index=False)

    if method == "climatology":
        table = df.pivot_table(
            index=["carrier", "country"], columns="period", values="p_max_pu"
        )
        with pd.option_context("display.width", 200, "display.max_rows", None):
            logger.info(f"Monthly availability:\n{table.round(2)}")
