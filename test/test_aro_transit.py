# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Fork additions around left-out and transit countries.

- `base_network.build_custom_admin_shapes`: a transit country without a custom bus shape
  becomes one shape `<country> 1` cut from its country shape.
- `build_transmission_projects.remove_projects_ending_outside_countries`: a project ending
  in a country that is not modelled is dropped, not snapped to a neighbour's bus.

Toy shapes, no build needed.
"""

import geopandas as gpd
import pandas as pd
import pypsa
import pytest
from shapely.geometry import LineString, box

from scripts.base_network import build_custom_admin_shapes
from scripts.build_transmission_projects import (
    remove_projects_ending_outside_countries,
)

# three unit-degree countries in a row around 15E 45N: HR | BA | RS
HR, BA, RS = box(14, 44, 15, 45), box(15, 44, 16, 45), box(16, 44, 17, 45)


def _shapes(tmp_path, names, countries, geoms, fname):
    path = tmp_path / fname
    gpd.GeoDataFrame(
        {"name": names, "country": countries}, geometry=geoms, crs=4326
    ).to_file(path)
    return str(path)


def _network(countries):
    n = pypsa.Network()
    n.add(
        "Bus",
        [f"{c} bus" for c in countries],
        x=[{"HR": 14.5, "BA": 15.5, "RS": 16.5}[c] for c in countries],
        y=44.5,
        country=countries,
        carrier="AC",
    )
    n.crs = "EPSG:4326"
    return n


def test_transit_country_becomes_one_shape(tmp_path):
    busshapes = _shapes(
        tmp_path, ["HR 1", "RS 1"], ["HR", "RS"], [HR, RS], "bs.geojson"
    )
    country = _shapes(
        tmp_path, ["HR", "BA", "RS"], [None] * 3, [HR, BA, RS], "cs.geojson"
    )
    n = _network(["HR", "BA", "RS"])
    shapes = build_custom_admin_shapes(
        n, busshapes, country_shapes=country, transit_countries=["BA"]
    )
    assert "BA 1" in shapes.index
    assert shapes.loc["BA 1", "country"] == "BA"
    assert shapes.loc["BA 1", "geometry"].equals(BA)


def test_country_without_shape_and_not_transit_refused(tmp_path):
    busshapes = _shapes(
        tmp_path, ["HR 1", "RS 1"], ["HR", "RS"], [HR, RS], "bs.geojson"
    )
    n = _network(["HR", "BA", "RS"])
    with pytest.raises(ValueError, match="no shapes for"):
        build_custom_admin_shapes(n, busshapes)


def test_transit_country_with_partition_shape_refused(tmp_path):
    busshapes = _shapes(
        tmp_path,
        ["HR 1", "BA 1", "RS 1"],
        ["HR", "BA", "RS"],
        [HR, BA, RS],
        "bs.geojson",
    )
    country = _shapes(
        tmp_path, ["HR", "BA", "RS"], [None] * 3, [HR, BA, RS], "cs.geojson"
    )
    n = _network(["HR", "BA", "RS"])
    with pytest.raises(ValueError, match="not partitioned"):
        build_custom_admin_shapes(
            n, busshapes, country_shapes=country, transit_countries=["BA"]
        )


def test_project_ending_in_unmodelled_country_dropped():
    countries = gpd.GeoDataFrame({"name": ["HR", "RS"]}, geometry=[HR, RS], crs=4326)
    offshore = gpd.GeoDataFrame({"country": []}, geometry=[], crs=4326)
    lines = pd.DataFrame(
        {
            "geometry": [
                LineString([(14.5, 44.5), (16.5, 44.5)]),  # HR -> RS, both modelled
                LineString([(16.5, 44.5), (15.5, 44.5)]),  # RS -> BA, BA not modelled
            ]
        },
        index=["ok", "into_ba"],
    )
    kept = remove_projects_ending_outside_countries(lines, countries, offshore)
    assert list(kept.index) == ["ok"]


def test_project_end_just_across_a_border_kept():
    # coordinate noise: an end 1 km outside HR (inside the unmodelled BA) stays within the
    # 5 km buffer and is kept
    countries = gpd.GeoDataFrame({"name": ["HR", "RS"]}, geometry=[HR, RS], crs=4326)
    offshore = gpd.GeoDataFrame({"country": []}, geometry=[], crs=4326)
    lines = pd.DataFrame(
        {"geometry": [LineString([(14.5, 44.5), (15.012, 44.5)])]}, index=["edge"]
    )
    kept = remove_projects_ending_outside_countries(lines, countries, offshore)
    assert list(kept.index) == ["edge"]
