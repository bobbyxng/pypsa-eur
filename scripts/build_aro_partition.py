# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Collect the snapped outage islands of all countries into one file and a summary table.

`islands.geojson` is the candidate for `data/busshapes/` (read by a pre-run with
`clustering: mode: custom_busshapes` and `exact_busshapes: true`); promoting it is a
deliberate manual step. Its islands tile `country_shapes` exactly -- clipped to them, the
gaps (lakes such as Peipus, coastline slivers, where the LAU outlines differ) given to the
island sharing the longest border -- so the pre-run's regions, cut along the islands, cover
each country as its offshore regions expect, and every map shows the same borders. The summary
has per country: the electricity and heat peaks, k by the rule and as built, island load range (x mean, separate land masses excluded) and both
copperplate columns (load outside the islands' connected cores, all voltages / below the
threshold).
"""

import logging

import geopandas as gpd
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_aro_islands import islands_per_country

logger = logging.getLogger(__name__)


def conform_to_countries(isl, shapes):
    """
    Islands clipped to the country shapes, gaps given to the island sharing the longest
    border with them (the nearest if none touches). Logs the area changed per country.
    """
    out = []
    for c, g in isl.groupby("country"):
        g = g.to_crs(3035).copy()
        shape = shapes.to_crs(3035).at[c, "geometry"].buffer(0)
        before = g.geometry.make_valid().buffer(0)
        g["geometry"] = before.intersection(shape)
        gaps = shape.difference(g.geometry.union_all())
        pieces = [p for p in getattr(gaps, "geoms", [gaps]) if p.area > 1.0]
        add = {i: [] for i in g.index}
        for p in pieces:
            shared = g.geometry.intersection(p.buffer(1.0)).area
            i = shared.idxmax() if shared.max() > 0 else g.distance(p).idxmin()
            add[i].append(p)
        for i, ps in add.items():
            if ps:
                g.at[i, "geometry"] = gpd.GeoSeries(
                    [g.at[i, "geometry"], *ps]
                ).union_all()
        cut = (before.area - before.intersection(shape).area).sum() / 1e6
        filled = sum(p.area for p in pieces) / 1e6
        logger.info(
            f"{c}: {cut:,.0f} km2 outside the country shape cut off, "
            f"{filled:,.0f} km2 of gaps filled ({len(pieces)} pieces)"
        )
        # valid polygons only (make_valid can leave slivers as lines; buffer(0) drops them),
        # in the output CRS too: reprojecting can make a valid polygon self-touch
        g["geometry"] = g.geometry.make_valid().buffer(0)
        g = g.to_crs(isl.crs)
        g["geometry"] = g.geometry.make_valid().buffer(0)
        out.append(g)
    return gpd.GeoDataFrame(pd.concat(out), crs=isl.crs)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_aro_partition")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    isl = pd.concat(
        [gpd.read_file(f) for f in snakemake.input.islands], ignore_index=True
    )
    isl = gpd.GeoDataFrame(isl, geometry="geometry", crs=4326)
    isl["name"] = isl.country + " " + isl.island.astype(str)
    shapes = gpd.read_file(snakemake.input.country_shapes).set_index("name")
    isl = conform_to_countries(isl, shapes)
    isl[
        [
            "name",
            "country",
            "island",
            "load_share",
            "peak_GW",
            "separate_land_mass",
            "geometry",
        ]
    ].to_file(snakemake.output.islands)

    cp = pd.concat([pd.read_csv(f) for f in snakemake.input.copperplate])
    k, X = islands_per_country(
        snakemake.input.peaks, snakemake.params.countries, snakemake.params.partition
    )
    rows = []
    for c, g in isl.groupby("country"):
        main = g[~g.separate_land_mass.fillna(False).astype(bool)]
        share = main.load_share / main.load_share.sum() * len(main)
        cc = cp[cp.country == c]
        rows.append(
            dict(
                country=c,
                electricity_GW=k.at[c, "electricity_GW"],
                heat_GW=k.at[c, "heat_GW"],
                peak_GW=round(k.at[c, "peak_GW"], 2),
                k_rule=int(k.at[c, "k"]),
                k=len(g),
                separate_land_masses=len(g) - len(main),
                island_load_min=round(share.min(), 2),
                island_load_max=round(share.max(), 2),
                copperplate_all=round(cc.outside_share_all.sum(), 4) + 0.0,  # no "-0.0"
                copperplate_below=round(cc.outside_share_below.sum(), 4)
                + 0.0,  # no "-0.0"
            )
        )
    summary = pd.DataFrame(rows).sort_values("peak_GW", ascending=False)
    summary.to_csv(snakemake.output.summary, index=False)
    logger.info(
        f"{len(isl)} islands in {len(summary)} countries (target island size {X:.2f} GW)"
    )
