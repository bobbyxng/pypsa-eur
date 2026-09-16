# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieve the district-heating potential data of Fallahnejad et al. (2024).

Fork-specific; not part of upstream PyPSA-Eur. Supplies the inputs for
`calibrate_district_heating_potential`, which turns them into per-country values for
`sector: district_heating: potential`.

Source: https://doi.org/10.5281/zenodo.7455894 (CC-BY-4.0), the data behind
Fallahnejad et al. (2024), Applied Energy 353, 122154. Only the *RES-H Best Case*
scenario is retrieved -- that is the one whose totals reproduce the paper's headline
figures (3128 TWh in 2020, 1709 TWh and a 31% district-heating share in 2050). The
archive also contains an `sEEnergies BL2050` scenario, which is a different (current
policy) demand pathway and gives 41%; mixing the two silently would be wrong.

The archive is 5.1 GB but only ~400 MB of it is needed, so this fetches individual ZIP
members over HTTP range requests rather than downloading the whole thing. Zenodo serves
`accept-ranges: bytes`, and each member's byte span is computable from its local header,
so one GET per file suffices. Do NOT "simplify" this into handing a seekable HTTP file to
`zipfile.read()`: that seeks backwards while inflating, which refetches and gave a 7x read
amplification and an effective stall in testing.

Relevant Settings
-----------------

```yaml
# No configuration; the source is a fixed, versioned archive.
```

Inputs
------
- None (downloads from Zenodo)

Outputs
-------
- `data/fallahnejad/manifest.json`: what was retrieved; the rule's declared output
- `data/fallahnejad/summaries/{country}.csv`: per-DH-area results, incl. `dhPot_2050 [GWh]`
- `data/fallahnejad/demand/{country}_2020.tif`: 2020 heat demand, 100 m, EPSG:3035
- `data/fallahnejad/demand/{country}_2050.tif`: 2050 heat demand, same grid

The manifest, rather than the directory, is the declared output on purpose. Snakemake
deletes a `directory()` output wholesale before re-running the job, which would throw away
382 MB every time the rule is retried and defeat the resume below.

Notes
-----
Filenames use PyPSA-Eur's country codes, not the archive's: the two Eurostat spellings
that differ (`EL`, `UK`) are translated to `GR` and `GB` on the way out, so downstream
consumers never see the source convention.

Re-running skips files already present, so an interrupted retrieval resumes.
"""

import io
import json
import logging
import re
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

URL = (
    "https://zenodo.org/records/7455894/files/"
    "Data%20set%20on%20district%20heating%20potentials%20in%20EU-27%20countries.zip"
)
SCENARIO = "Outputs according to the RES-H Best Case Scenario"
# The archive uses Eurostat's codes; PyPSA-Eur uses ISO-3166-1 alpha-2 throughout, so
# translate here rather than leaving every downstream consumer to remember it.
RENAME = {"EL": "GR", "UK": "GB"}
WANTED = {
    "summary.csv": ("summaries", "{ct}.csv"),
    "Energy_TOTAL_2020.tif": ("demand", "{ct}_2020.tif"),
    "Energy_TOTAL_2050.tif": ("demand", "{ct}_2050.tif"),
}


def _get(start: int, end: int, attempts: int = 5) -> bytes:
    """
    One HTTP range request, inclusive bounds, retried on transient failures.

    Zenodo intermittently answers with 504s under load, and a full retrieval is a few
    hundred requests, so without this a single blip loses the whole job. Retrying here
    rather than leaning on Snakemake's `retries:` keeps the already-fetched files and
    costs seconds instead of restarting the archive listing.
    """
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts:
                raise
            delay = 2**attempt
            logger.warning(
                f"{error} on bytes {start}-{end}; retry {attempt}/{attempts - 1} "
                f"in {delay}s"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


class _Tail(io.RawIOBase):
    """
    Minimal seekable reader, so `zipfile` can parse the central directory.

    Used for listing only -- `zipfile` reads just the archive tail to build the member
    index, which is a handful of requests. Member *contents* are fetched by `_member`.
    """

    def __init__(self):
        head = urllib.request.Request(URL, method="HEAD")
        with urllib.request.urlopen(head, timeout=120) as response:
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        data = _get(self.pos, self.pos + n - 1)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _member(info: zipfile.ZipInfo) -> bytes:
    """Fetch and decompress one member in a single range request."""
    # The local header is 30 fixed bytes plus a name and an extra field, whose lengths
    # live at offsets 26 and 28. The central directory's copy of those lengths can differ,
    # so read the local one rather than trusting `info`.
    header = _get(info.header_offset, info.header_offset + 29)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    raw = _get(start, start + info.compress_size - 1)
    return (
        raw if info.compress_type == zipfile.ZIP_STORED else zlib.decompress(raw, -15)
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_fallahnejad_dh")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    manifest_path = Path(snakemake.output.manifest)
    root = manifest_path.parent
    archive = zipfile.ZipFile(io.BufferedReader(_Tail(), buffer_size=1 << 20))

    targets = []
    for info in archive.infolist():
        parts = info.filename.split("/")
        if parts[0] != SCENARIO or parts[-1] not in WANTED:
            continue
        # Country folders are numbered, e.g. "7DE", "11ES", "10EL".
        match = re.fullmatch(r"[0-9]+([A-Z]{2})", parts[1])
        if match:
            country = RENAME.get(match.group(1), match.group(1))
            subdir, pattern = WANTED[parts[-1]]
            targets.append((info, root / subdir / pattern.format(ct=country)))

    # Smallest first: a couple of national rasters are pathologically large (mostly empty
    # cells), and this way a failure late on still leaves most of the data cached.
    targets.sort(key=lambda t: t[0].file_size)
    total = sum(i.file_size for i, _ in targets) / 1e6
    logger.info(f"Retrieving {len(targets)} files ({total:.0f} MB) from {URL}")

    for n, (info, path) in enumerate(targets, start=1):
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_member(info))
        logger.info(
            f"[{n}/{len(targets)}] {path.relative_to(root)} "
            f"({info.file_size / 1e6:.1f} MB)"
        )

    manifest_path.write_text(
        json.dumps(
            {
                "source": URL,
                "doi": "10.5281/zenodo.7455894",
                "scenario": SCENARIO,
                "files": sorted(str(p.relative_to(root)) for _, p in targets),
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(f"Fallahnejad et al. (2024) data ready in {root}")
