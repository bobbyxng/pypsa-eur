# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieve reported outages and installed capacities per unit from the ENTSO-E Transparency
Platform for one country and calendar year.

Queries the unavailability of generation units (15.1.A/B) and of production units
(15.1.C/D), and the installed capacity per production unit (14.1.B), with the token in the
environment variable ``ENTSOE_API_TOKEN``. All production types are kept: the API cannot
filter by type, and keeping the raw data carrier-independent means changing
``conventional: unavailability: carriers`` needs no new download.

The API is slow rather than large (about 15 s per request of at most 200 outage documents),
so a country-year without any unit of the configured carriers in the power plant list is
skipped without a request and written as empty files.

Outputs
-------
- ``outages``: one row per outage period, start and end clipped to the calendar year, with
  ``level`` = "generation" or "production" for the queried unit level.
- ``capacity``: installed capacity per production unit for the year.
"""

import logging
import os
import time

import pandas as pd
import pycountry
from bs4 import BeautifulSoup
from entsoe import EntsoePandasClient, EntsoeRawClient
from entsoe.exceptions import NoMatchingDataError
from entsoe.mappings import PSRTYPE_MAPPINGS
from requests.exceptions import HTTPError, RequestException

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_conventional_unavailability import PPM_FUELTYPES

logger = logging.getLogger(__name__)

# Countries whose outages are reported per bidding zone rather than under the country code
# (DE alone returns nothing; DE_AT_LU was the zone until 2018-09-30). Zones of one country may
# report the same outage twice, so outages are deduplicated after concatenation.
AREAS = {
    "DE": ["DE_LU", "DE_AT_LU"],
    "DK": ["DK_1", "DK_2"],
}

OUTAGE_COLUMNS = [
    "mrid",
    "revision",
    "docstatus",
    "businesstype",
    "plant_type",
    "biddingzone_domain",
    "production_resource_id",
    "production_resource_name",
    "production_resource_psr_name",
    "nominal_power",
    "avail_qty",
    "start",
    "end",
    "level",
]


def with_retries(func, *args, attempts=4, **kwargs):
    """
    Call an API query, retrying transient HTTP and connection errors with backoff.

    A rate-limit response (HTTP 429) or a server error gets logged, so a stalling
    download is visible in the log rather than only as a slow job.
    """
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except NoMatchingDataError:
            raise
        except (HTTPError, RequestException) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if attempt == attempts:
                raise
            wait = 30 * attempt
            logger.warning(
                f"{func.__name__} failed (attempt {attempt}/{attempts}, "
                f"HTTP {status}): {e}. Retrying in {wait} s."
            )
            time.sleep(wait)


# entsoe-py pages through at most 25 x 200 = 5000 documents per query and then stops
# without an error, so a busy area-year (DE, GB, ES report 4000-5000 outages a year across
# all plant types) comes back silently truncated. A window close to the cap is split (a
# year into months, anything shorter in half) and queried again, recursively, so only
# busy area-years pay for the extra requests.
DOCUMENT_CAP = 4800


def query_window(query, area, start, end):
    # The number of documents is not visible in the parsed frame (one document can repeat
    # an mrid and revision, or hold several periods), so the row count, never below the
    # document count, is the conservative trigger: a capped query returns 5000 rows.
    df = with_retries(query, area, start=start, end=end)
    if len(df) >= DOCUMENT_CAP and end - start > pd.Timedelta("1D"):
        logger.warning(
            f"{area} {start:%Y-%m-%d}..{end:%Y-%m-%d}: {len(df)} outage periods, near "
            "the API's cap of 5000 documents; splitting the window."
        )
        # A capped year goes straight to months: halving a busy year usually caps both
        # halves too and fetches their 2 x 5000 documents for nothing.
        bounds = pd.date_range(start, end, freq="MS")
        if len(bounds) < 3 or bounds[0] != start or bounds[-1] != end:
            bounds = pd.DatetimeIndex([start, start + (end - start) / 2, end])
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            try:
                parts.append(query_window(query, area, a, b))
            except NoMatchingDataError:
                pass
        return pd.concat(parts) if parts else df.iloc[:0]
    return df


def query_outages(client, area, start, end):
    frames = []
    for level, query in [
        ("generation", client.query_unavailability_of_generation_units),
        ("production", client.query_unavailability_of_production_units),
    ]:
        t = time.time()
        parts = []
        try:
            parts.append(query_window(query, area, start, end))
        except NoMatchingDataError:
            pass
        if not parts:
            logger.info(f"{area} {level}: no outages ({time.time() - t:.0f} s)")
            continue
        df = pd.concat(parts, ignore_index=True)
        logger.info(
            f"{area} {level}: {len(df)} outage periods ({time.time() - t:.0f} s)"
        )
        frames.append(df.assign(level=level))
    return frames


def parse_capacity_per_unit(xml):
    """
    Parse 14.1.B ourselves: entsoe-py's parser fails on current responses (it looks up a
    tag the documents no longer carry), while the documents are flat enough to read
    directly.
    """
    # The raw client decodes the UTF-8 response as Latin-1 ("NovÃ¡ky" for "Nováky").
    try:
        xml = xml.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    soup = BeautifulSoup(xml, "html.parser")
    rows = []
    for ts in soup.find_all("timeseries"):
        psr = ts.find("psrtype").text
        rows.append(
            {
                "production_resource_id": ts.find("registeredresource.mrid").text,
                "production_resource_name": ts.find("registeredresource.name").text,
                "plant_type": PSRTYPE_MAPPINGS.get(psr, psr),
                "biddingzone_domain": ts.find("inbiddingzone_domain.mrid").text,
                "installed_capacity": float(ts.find("quantity").text),
                "valid_from": ts.find("timeinterval").find("start").text,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "production_resource_id",
            "production_resource_name",
            "plant_type",
            "biddingzone_domain",
            "installed_capacity",
            "valid_from",
        ],
    )


def has_carrier_units(powerplants, country, year, carriers):
    """Whether the power plant list has a unit of the carriers operating in `year`."""
    fueltypes = {f for c in carriers for f in PPM_FUELTYPES.get(c, [])}  # noqa: F841 (query)
    ppl = powerplants.query("Fueltype in @fueltypes")
    # ppm uses country names; Kosovo is the one pycountry cannot look up.
    codes = ppl.Country.map(
        lambda n: "XK" if n == "Kosovo" else pycountry.countries.lookup(n).alpha_2
    )
    ppl = ppl[codes == country]
    operating = (ppl.DateIn.fillna(0) <= year) & (ppl.DateOut.fillna(9999) >= year)
    return bool(operating.any())


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_entsoe_unavailability", country="SK", year="2024"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    country = snakemake.wildcards.country
    year = int(snakemake.wildcards.year)
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

    powerplants = pd.read_csv(snakemake.input.powerplants, index_col=0)
    if not has_carrier_units(powerplants, country, year, snakemake.params.carriers):
        logger.info(
            f"No {snakemake.params.carriers} units in {country} in {year}; skipping."
        )
        pd.DataFrame(columns=OUTAGE_COLUMNS).to_csv(
            snakemake.output.outages, index=False
        )
        parse_capacity_per_unit("").to_csv(snakemake.output.capacity, index=False)
        raise SystemExit

    token = os.environ.get("ENTSOE_API_TOKEN")
    assert token, (
        "ENTSOE API token must be provided (environment variable ENTSOE_API_TOKEN)!"
    )
    client = EntsoePandasClient(api_key=token)
    raw_client = EntsoeRawClient(api_key=token)

    outages, capacity = [], []
    for area in AREAS.get(country, [country]):
        outages += query_outages(client, area, start, end)
        try:
            xml = with_retries(
                raw_client.query_installed_generation_capacity_per_unit,
                area,
                start=start,
                end=end,
            )
            capacity.append(parse_capacity_per_unit(xml))
        except NoMatchingDataError:
            logger.info(f"{area}: no installed capacity per unit")

    if outages:
        df = pd.concat(outages, ignore_index=True)
        for col in ["start", "end"]:
            df[col] = pd.to_datetime(df[col], utc=True)
        df["start"] = df.start.clip(lower=start)
        df["end"] = df.end.clip(upper=end)
        df = df[df.end > df.start].reindex(columns=OUTAGE_COLUMNS)
        df = df.drop_duplicates(["mrid", "revision", "start", "end", "level"])
    else:
        df = pd.DataFrame(columns=OUTAGE_COLUMNS)
    df.to_csv(snakemake.output.outages, index=False)

    cap = (
        pd.concat(capacity, ignore_index=True).drop_duplicates("production_resource_id")
        if capacity
        else parse_capacity_per_unit("")
    )
    cap.to_csv(snakemake.output.capacity, index=False)
    logger.info(
        f"{country} {year}: {len(df)} outage periods, "
        f"{len(cap)} production units with installed capacity."
    )
