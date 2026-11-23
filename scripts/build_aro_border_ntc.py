# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Turn the TYNDP 2026 reference grid into NTCs per country border, sub-border and direction.

Input is ENTSO-E's `ReferenceGrid_Electricity.xlsx` (TYNDP 2026 scenario/CBA input data,
`Line-data.zip`), sheet `Year_<year>`: one row per bidding-zone border `A-B` with the 2025
NTC, one column pair per project and `Summary Direction 1/2` = the 2025 NTC plus the
reference-grid projects of that year, for A->B and B->A.

Kept: borders between two countries. Dropped: rows with an offshore node (`*_OFF`, the
hybrid interconnector legs and radial offshore connections) and zone borders inside a
country (SE3-SE4, the Italian and Norwegian zones, DK1-DK2, Poland's virtual PL00E/PL00I
import/export nodes against PL00). A country is the node's first two letters, UK -> GB, so
LU's virtual zones, DK1/DK2 etc. fold into their country; TYNDP's RS00 contains Kosovo.

`data/aro_border_ntc_subborders.csv` names the country pairs whose TYNDP rows are kept apart
(e.g. DE-DK East vs West); every TYNDP row of such a pair must belong to exactly one of its
sub-borders. Rows of all other pairs are summed per direction. Direction sums are taken per
direction, never across: PL00E-CZ00 carries PL->CZ only and CZ00-PL00I CZ->PL only.

Output, one row per (border, subborder, direction): `border` (sorted country pair),
`subborder` ("" for a whole border), `from`, `to`, `p_nom` (MW), `tyndp_borders`.
PyPSARO's `memory: crossborder-capacity-vs-tyndp` documents the data and every rule here.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

OFFSHORE_SUFFIX = "_OFF"


def node_country(node: str) -> str:
    """TYNDP node code -> ISO country as pypsa-eur uses it (UK00, UKNI -> GB)."""
    code = node[:2]
    return {"UK": "GB"}.get(code, code)


def border_key(a: str, b: str) -> str:
    return "-".join(sorted([a, b]))


def read_reference_grid(path: str, year: int) -> pd.DataFrame:
    """Rows of the year's sheet: tyndp_border, node0, node1, p01, p10 (MW)."""
    raw = pd.read_excel(path, sheet_name=f"Year_{year}")
    raw = raw[raw["Border"].astype(str).str.contains("-")]
    nodes = raw["Border"].str.split("-", n=1, expand=True)
    return pd.DataFrame(
        {
            "tyndp_border": raw["Border"].values,
            "node0": nodes[0].str.strip().values,
            "node1": nodes[1].str.strip().values,
            "p01": pd.to_numeric(raw["Summary Direction 1"]).fillna(0.0).values,
            "p10": pd.to_numeric(raw["Summary Direction 2"]).fillna(0.0).values,
        }
    )


def border_ntc(rows: pd.DataFrame, subborders: pd.DataFrame) -> pd.DataFrame:
    """NTC per border, sub-border and direction from the reference-grid rows."""
    offshore = rows.node0.str.endswith(OFFSHORE_SUFFIX) | rows.node1.str.endswith(
        OFFSHORE_SUFFIX
    )
    rows = rows[~offshore].copy()
    rows["c0"] = rows.node0.map(node_country)
    rows["c1"] = rows.node1.map(node_country)
    rows = rows[rows.c0 != rows.c1]
    rows["border"] = [border_key(a, b) for a, b in zip(rows.c0, rows.c1)]

    # TYNDP border string -> sub-border, order of the two nodes ignored
    def canon(tyndp_border: str) -> str:
        return border_key(*tyndp_border.split("-", 1))

    assigned = {}
    for _, s in subborders.iterrows():
        for tb in str(s.tyndp_borders).split():
            key = canon(tb)
            if key in assigned:
                raise ValueError(
                    f"TYNDP border {tb} is listed for two sub-borders "
                    f"({assigned[key]} and {s.border}/{s.subborder})"
                )
            assigned[key] = (s.border, s.subborder)

    split = set(subborders.border)
    sub = []
    for _, r in rows.iterrows():
        hit = assigned.get(canon(r.tyndp_border))
        if r.border in split:
            if hit is None or hit[0] != r.border:
                raise ValueError(
                    f"TYNDP row {r.tyndp_border} belongs to {r.border}, which is split into "
                    "sub-borders in the sub-border table, but matches none of them"
                )
            sub.append(hit[1])
        else:
            if hit is not None:
                raise ValueError(
                    f"Sub-border table assigns {r.tyndp_border} to {hit[0]}, "
                    f"but its countries make it {r.border}"
                )
            sub.append("")
    rows["subborder"] = sub
    for (border, subborder), _ in subborders.groupby(["border", "subborder"]):
        if not ((rows.border == border) & (rows.subborder == subborder)).any():
            raise ValueError(
                f"Sub-border {border}/{subborder} matches no row of the reference grid"
            )

    directed = pd.concat(
        [
            rows.assign(**{"from": rows.c0, "to": rows.c1, "p_nom": rows.p01}),
            rows.assign(**{"from": rows.c1, "to": rows.c0, "p_nom": rows.p10}),
        ]
    )
    out = (
        directed.groupby(["border", "subborder", "from", "to"])
        .agg(
            p_nom=("p_nom", "sum"),
            tyndp_borders=("tyndp_border", lambda x: " ".join(sorted(set(x)))),
        )
        .reset_index()
    )
    return out


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_aro_border_ntc")

    from scripts._helpers import configure_logging, set_scenario_config

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    year = snakemake.params.border_ntc["year"]
    rows = read_reference_grid(snakemake.input.reference_grid, year)
    subborders = pd.read_csv(snakemake.input.subborders, comment="#")
    ntc = border_ntc(rows, subborders)
    logger.info(
        f"TYNDP 2026 reference grid {year}: {ntc.border.nunique()} country borders, "
        f"{ntc.p_nom.sum() / 1e3:.1f} GW summed over both directions."
    )
    ntc.to_csv(snakemake.output.border_ntc, index=False)
