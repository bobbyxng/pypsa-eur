# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Outage islands of one country, formed from the grid below the transmission level.

The ARO outage cuts the lines between islands; inside an island the model is a copperplate.
Islands are drawn where the sub-transmission grid is weakest, so that each one is a coherent
supply region, and are balanced by peak load (PyPSARO: memory: outage-island-partitioning).
One pass per country:

0. Islands per country: k = clip(round(peak / X), k_min, k_max), X = the reference country's
   peak / k_max (Germany: 14 TSO black-start regions); peak = electricity + heat.
1. Load on a 1 km grid, spread the way the model spreads it, otherwise an island sized here
   would hold a different load in the pre-run: electricity as `build_electricity_demand_base`
   (JRC Energy Atlas raster for EU27, DESNZ local authorities for Great Britain with Northern
   Ireland from NUTS3, NUTS3 GDP/population elsewhere), heat by NUTS3 population spread by area
   (what `build_population_layouts` does), weighted by the electricity and heat peaks.
2. Sub-transmission graph: buses below `voltage_threshold`; edges are lines below it (capacity
   s_nom, length) and transformers between two such buses (length 0).
3. Hubs: buses below the threshold with a transformer from at or above it; one per station
   (the build's station: `way/123-110` and `way/123-380` are station `way/123`).
4. Supply areas: every bus to the nearest hub along the graph (by length), unreachable buses to
   the nearest hub in a straight line. Outlines are Voronoi cells of the stations, split into land
   pieces and made contiguous (each area keeps the piece holding its hub). The country outline
   is made "dry" (inland holes filled, water narrower than ~2 km closed): rivers never form
   borders.
5. Separate land masses (no AC link to the mainland, a bus at or above the threshold or a DC
   converter landing, at least `separate_land_mass_km2`) are islands of their own, counted within k only if they hold
   at least `size_bounds[0]` x the country's mean island (East Denmark), on top of k if
   smaller (Balearics).
6. Neighbours: supply areas sharing more than `border_min_m` of border; link weight = capacity of
   the lines below the threshold between them, 1% of the median link if none.
7. Candidates: every grouping into k connected islands up to `enumeration_max` supply areas,
   METIS with perturbed weights above (its "contiguous" is best effort, every candidate is
   checked), all kept only with every island within `size_bounds` x the country mean. Pick
   (`selection`): the least cut (default), or among the groupings within
   `near_optimal_tolerance` of it the most balanced or their medoid. Reported with it: the border uncertainty, i.e. how much
   load the other groupings in that set put in another island (a flat optimum).
8. Outlines: supply-area cells merged, enclaves and exclaves moved to the neighbour they border
   most.
9. With `snap_to_lau`: each municipality (GISCO LAU 2019) to the island holding most of its
   load, then kept in one piece on the map; with `lau_respect_bounds`, border municipalities
   then moved between neighbouring islands until every island is within `size_bounds` again.
   A city is never split between islands.
10. Copperplate check at substation level: the load outside each island's connected core, with
    the island's own lines at all voltages (the paper's threat model, only lines between
    islands cut) and below the threshold only (every line at or above it lost too).
"""

import logging

import country_converter as coco
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pymetis
import rasterio as rio
import shapely
from matplotlib.lines import Line2D
from rasterio.windows import from_bounds
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon
from unidecode import unidecode

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)
CRS = 3035
log_lines = []


def say(s=""):
    logger.info(s)
    log_lines.append(s)


def read_release(path):
    return pd.read_csv(path, quotechar="'")


def dry_outline(shape, closing):
    """Fill inland holes and close water narrower than 2 x closing (m); straits stay open."""
    closed = shape.buffer(closing).buffer(-closing).union(shape)
    parts = getattr(closed, "geoms", [closed])
    return shapely.make_valid(
        MultiPolygon([Polygon(g.exterior) for g in parts if g.geom_type == "Polygon"])
    )


def station_of(bus_ids):
    """
    The build's station of each bus: the bus id without its voltage suffix.

    A station's buses sit on a 15 m circle around the station point, one per voltage, so a
    station cannot be found from coordinates: snapping to a 100 m grid split 39% of the
    multi-bus stations into two sites (two hubs, two supply areas) and merged 1,280 pairs of
    separate stations.
    """
    return pd.Series(bus_ids.str.rsplit("-", n=1).str[0], index=bus_ids)


def station_points(buses, station):
    """One point per station: the centroid of its buses."""
    return buses.geometry.groupby(station).agg(lambda g: g.union_all().centroid)


class LoadGrid:
    """Share of the country's load inside polygons, from the 1 km load grid."""

    def __init__(self, g):
        self.points = gpd.GeoDataFrame(
            g[["load"]], geometry=gpd.points_from_xy(g.x, g.y), crs=CRS
        )

    def of(self, gdf, key):
        j = gpd.sjoin(self.points, gdf[[key, "geometry"]], predicate="within")
        j = j[~j.index.duplicated()]
        return j.groupby(key)["load"].sum().reindex(gdf[key]).fillna(0.0)


def vstyle(v, ehv):
    if v < 50:
        return "30-50 kV", "#bdbdbd", 0.25
    if v < 100:
        return "50-100 kV", "#969696", 0.35
    if v < ehv:
        return f"100-{ehv:g} kV", "#525252", 0.5
    if v < 300:
        return "220-275 kV", "#2ca02c", 0.9
    if v < 350:
        return "300-330 kV", "#e377c2", 1.1
    return "380-750 kV", "#d62728", 1.2


def enumerate_groupings(n, edges, k):
    """Every grouping of n nodes into k connected parts (bitmask enumeration)."""
    nbr = [0] * n
    for p, q, _ in edges:
        nbr[p] |= 1 << q
        nbr[q] |= 1 << p

    def ncomp(mask):
        c = 0
        while mask:
            c += 1
            front = mask & -mask
            seen = front
            while front:
                nb, m = 0, front
                while m:
                    low = m & -m
                    nb |= nbr[low.bit_length() - 1]
                    m ^= low
                front = nb & mask & ~seen
                seen |= front
            mask &= ~seen
        return c

    def csub(v, R):
        def grow(S, ext, excl):
            yield S
            while ext:
                u = ext & -ext
                ext ^= u
                newS = S | u
                yield from grow(
                    newS, (ext | nbr[u.bit_length() - 1]) & R & ~newS & ~excl, excl
                )
                excl |= u

        yield from grow(1 << v, nbr[v] & R, 0)

    def parts(R, kk):
        if kk == 1:
            if ncomp(R) == 1:
                yield [R]
            return
        v = (R & -R).bit_length() - 1
        for S in csub(v, R):
            rest = R & ~S
            if not rest or bin(rest).count("1") < kk - 1 or ncomp(rest) > kk - 1:
                continue
            for p in parts(rest, kk - 1):
                yield [S] + p

    for part in parts((1 << n) - 1, k):
        lab = np.zeros(n, dtype=int)
        for gi, m in enumerate(part):
            while m:
                low = m & -m
                lab[low.bit_length() - 1] = gi
                m ^= low
        yield lab


def metis_groupings(n, edges, k, node_weights, samples, ufs, perturbation, seed):
    """METIS groupings with perturbed edge weights and a range of imbalance factors."""
    rng = np.random.default_rng(seed)
    adj = {i: [] for i in range(n)}
    for p, q, w in edges:
        adj[p].append((q, w))
        adj[q].append((p, w))
    wmax = max(w for *_, w in edges)
    vw = [max(1, int(round(x * 1e6))) for x in node_weights]
    seen = set()
    for run in range(samples):
        f = {}
        xadj, adjncy, ew = [0], [], []
        for i in range(n):
            for q, w in adj[i]:
                key = (min(i, q), max(i, q))
                f.setdefault(
                    key,
                    1.0
                    if run < len(ufs)
                    else rng.uniform(1 - perturbation, 1 + perturbation),
                )
                adjncy.append(q)
                ew.append(max(1, int(round(w / wmax * 1000 * f[key]))))
            xadj.append(len(adjncy))
        try:
            _, pp = pymetis.part_graph(
                k,
                xadj=xadj,
                adjncy=adjncy,
                vweights=vw,
                eweights=ew,
                contiguous=True,
                options=pymetis.Options(ufactor=ufs[run % len(ufs)], seed=seed + run),
            )
        except Exception:
            continue
        lab = np.array(pp)
        if len(set(pp)) < k or lab.tobytes() in seen:
            continue
        seen.add(lab.tobytes())
        yield lab


def moved_load(a, b, weights, k):
    """Share of the load in a different island between groupings a and b (labels matched)."""
    M = np.zeros((k, k))
    np.add.at(M, (a, b), weights)
    r, c = linear_sum_assignment(-M)
    return 1 - M[r, c].sum() / weights.sum()


def border_uncertainty(ok, best, weights, tolerance, k, cap=200):
    """
    Share of the load that equally good groupings put in another island than `best`.

    Least cut sits on a flat optimum: in Germany the ten best groupings were within 2-6% of
    the least cut but put 10-27% of the load in other islands, and which one wins depends on
    details (two hubs more, the METIS sample sequence). Reported for either `selection`: the
    mean share of load in another island over the groupings within `tolerance` of the least
    cut (at most `cap` of them, cheapest first).
    """
    least = min(s[2] for s in ok)
    near = sorted(
        (s for s in ok if s[2] <= least * (1 + tolerance)), key=lambda s: s[2]
    )[: cap + 1]
    others = [s for s in near if s is not best]
    if not others:
        return 0.0, len(near)
    w = weights / weights.sum()
    moved = [moved_load(best[3], s[3], w, k) for s in others]
    return float(np.mean(moved)), len(near)


def near_optimal_balanced(ok, tolerance):
    """
    The most balanced grouping within `tolerance` of the least cut: the smallest largest
    deviation of an island from the mean (additive, as `size_bounds`), the lower cut on a tie.

    The least cut drifts to whichever size bound has room when cheap borders exist: Spain's
    mainland came out 1.46/1.43/0.59/0.53 x mean, Switzerland 1.46/0.78/0.76, Berlin alone at
    0.50, with 16 (ES) other groupings within 5% of that cut. A cut within that tolerance is
    as good as the least by the measure we already report (border uncertainty), so the tie
    is broken by balance; without near-equals the least cut stays.
    """
    least = min(s[2] for s in ok)
    near = [s for s in ok if s[2] <= least * (1 + tolerance)]
    return min(near, key=lambda s: (max(1 - s[0], s[1] - 1), s[2]))


def near_optimal_medoid(ok, weights, tolerance, k, cap=200):
    """
    The grouping within `tolerance` of the least cut that moves the least load on average
    against the others in that set (at most `cap`, cheapest first).

    The most typical of the equally good groupings: in Germany it halved how much load changes
    island between METIS seeds (5.6% vs 11.4% for least cut), at a cut a few percent above it.
    """
    least = min(s[2] for s in ok)
    near = sorted(
        (s for s in ok if s[2] <= least * (1 + tolerance)), key=lambda s: s[2]
    )[:cap]
    if len(near) < 3:
        return near[0]
    w = weights / weights.sum()
    D = np.zeros((len(near), len(near)))
    for i in range(len(near)):
        for j in range(i + 1, len(near)):
            D[i, j] = D[j, i] = moved_load(near[i][3], near[j][3], w, k)
    return near[int(D.sum(axis=1).argmin())]


def islands_per_country(peaks_path, countries, cfg):
    """K = clip(round(peak / X), k_min, k_max), X = reference peak / k_max."""
    peaks = pd.read_csv(peaks_path, index_col="country")
    missing = sorted(set(countries) - set(peaks.index))
    if missing:
        raise ValueError(f"no peak for {missing} in {peaks_path}")
    peaks = peaks.loc[countries]
    peak = peaks["electricity_GW"] + peaks["heat_GW"]
    X = peak[cfg["reference_country"]] / cfg["k_max"]
    k = np.clip(np.round(peak / X), cfg["k_min"], cfg["k_max"]).astype(int)
    return peaks.assign(peak_GW=peak, k=k, island_GW=peak / k), X


def point_shares(points, polygons, value, id_col):
    """
    Spread `value` of each polygon evenly over the points inside it (equal-area cells).

    Points outside every polygon (coastal mismatch) go to the nearest one; a polygon with no
    point (smaller than a cell) puts its value on its nearest point.
    """
    joined = gpd.sjoin_nearest(points, polygons[[id_col, "geometry"]], how="left")
    owner = joined.loc[~joined.index.duplicated(), id_col].reindex(points.index)
    count = owner.value_counts()
    out = owner.map(value / count).fillna(0.0)
    empty = value.index.difference(count.index)
    if len(empty):
        reps = polygons.set_index(id_col).loc[empty].geometry.representative_point()
        nearest = points.sindex.nearest(reps, return_all=False)[1]
        for pid, idx in zip(empty, nearest):
            out.iloc[idx] += value[pid]
    return out


def build_load_grid(C, shape, inputs, key, el_gw, heat_gw):
    """Points of 1 km (cell centres of the Energy Atlas raster) with shares el, heat, load."""
    with rio.open(inputs.raster) as r:
        window = (
            from_bounds(*shape.bounds, transform=r.transform)
            .round_offsets()
            .round_lengths()
        )
        band = r.read(1, window=window, boundless=True, fill_value=0)
        tf = r.window_transform(window)
    rows, cols = np.indices(band.shape)
    xs, ys = rio.transform.xy(tf, rows.ravel(), cols.ravel())
    xs, ys = np.asarray(xs), np.asarray(ys)
    inside = shapely.contains_xy(shape, xs, ys)
    points = gpd.GeoDataFrame(
        {"x": xs[inside], "y": ys[inside], "raster": band.ravel()[inside]},
        geometry=gpd.points_from_xy(xs[inside], ys[inside]),
        crs=CRS,
    ).reset_index(drop=True)
    points["raster"] = points["raster"].clip(lower=0)

    nuts = gpd.read_file(inputs.nuts3_shapes)
    nuts = nuts.loc[nuts["country"] == C].to_crs(CRS).rename(columns={"index": "nuts3"})
    norm = lambda s: s / s.sum()  # noqa: E731
    nuts_el = key["gdp"] * norm(nuts["gdp"]) + key["population"] * norm(nuts["pop"])
    nuts_el.index = nuts["nuts3"]
    nuts_pop = norm(nuts["pop"])
    nuts_pop.index = nuts["nuts3"]
    nuts = nuts.reset_index(drop=True)

    eu27 = set(coco.CountryConverter().EU27as("ISO2").ISO2)
    if C in eu27 and points["raster"].sum() > 0:
        el = points["raster"]
        source = "JRC Energy Atlas raster"
    elif C == "GB":
        # as build_electricity_demand_base.gb_distribution_keys, plus Northern Ireland from
        # NUTS3 at its share of the UK, renormalised over the UK as there
        df = pd.read_excel(inputs.gb_excel, skiprows=4, sheet_name="2019")
        df = df.loc[
            ~df["Local authority"].isin(["All local authorities", "Unallocated"])
        ]
        lad = gpd.read_file(inputs.gb_geojson).to_crs(CRS)
        lad = lad.rename(columns={"LAD24CD": "Code"}).merge(df, on="Code")
        attr = "Total consumption\n(GWh):\nAll meters"
        lad_val = norm(lad.set_index("Code")[attr].astype(float))
        ni = nuts["nuts3"].str.startswith("UKN")
        in_gb = points.within(lad.union_all().buffer(2000))
        el = pd.Series(0.0, index=points.index)
        el[in_gb] = point_shares(points[in_gb], lad.reset_index(), lad_val, "Code")
        if ni.any():
            el[~in_gb] = point_shares(
                points[~in_gb], nuts[ni], nuts_el[ni.values], "nuts3"
            )
        source = "DESNZ local authorities (Northern Ireland: NUTS3)"
    else:
        el = point_shares(points, nuts, nuts_el, "nuts3")
        source = f"NUTS3 ({key['gdp']:g} GDP + {key['population']:g} population)"
    heat = point_shares(points, nuts, nuts_pop, "nuts3")
    points["el"] = norm(el)
    points["heat"] = norm(heat)
    points["load"] = (el_gw * points["el"] + heat_gw * points["heat"]) / (
        el_gw + heat_gw
    )
    say(
        f"load grid: {len(points)} points of 1 km; electricity from {source}, heat by NUTS3 "
        f"population; weights {el_gw:.1f} GW electricity, {heat_gw:.1f} GW heat"
    )
    return points[["x", "y", "el", "heat", "load"]]


def relative_load(share, separate):
    """Island load / mean over the mainland islands (separate land masses excluded)."""
    main = share[~np.asarray(separate, dtype=bool)]
    return main / main.sum() * len(main)


def rebalance_lau(lau, G, L, separate, bounds):
    """
    Move border municipalities between neighbouring islands until every mainland island is
    within `bounds` x the mainland mean, or no move gets closer.

    Snapping by majority moves whole municipalities, and coarse ones (British districts,
    Latvian and Slovenian municipalities, a city as one unit) shifted islands from within the
    bounds to 0.5-1.9 x mean. Each step takes, among the moves that reduce the total
    violation, the municipality with the largest share of its grid load already in the
    receiving island (one the grid border runs through); others only when none is left.
    Taking the largest reduction instead moved whole Latvian municipalities across network
    borders (load outside the islands' connected cores 0% -> 33%). A move must not split the
    donor into more pieces than it already has; separate land masses are left alone.
    """
    lo, hi = bounds
    isl = lau["island"].to_dict()
    load = lau["load"].to_dict()
    main = sorted(i for i in set(isl.values()) if i not in separate)
    S = {i: sum(load[n] for n in lau.index if isl[n] == i) for i in main}
    mean = sum(S.values()) / len(main)

    def v(x):
        return max(lo - x / mean, x / mean - hi, 0.0)

    edges = [(u, w) for u, w in G.edges() if u in isl and w in isl]
    moves, moved_load = 0, 0.0
    for _ in range(len(lau)):
        if sum(v(x) for x in S.values()) <= 1e-9:
            break
        cand = {}
        for u, w in edges:
            a, b = isl[u], isl[w]
            if a == b or a not in S or b not in S:
                continue
            for n, src, dst in ((u, a, b), (w, b, a)):
                d = v(S[src] - load[n]) + v(S[dst] + load[n]) - v(S[src]) - v(S[dst])
                if d < -1e-12:
                    tot = L.loc[n].sum() if n in L.index else 0.0
                    ov = L.at[n, dst] / tot if tot > 0 and dst in L.columns else 0.0
                    cand[(n, dst)] = (ov <= 0, -ov, d)
        done = False
        for (n, dst), _ in sorted(cand.items(), key=lambda kv: kv[1]):
            src = isl[n]
            members = [m for m in lau.index if isl[m] == src]
            rest = [m for m in members if m != n]
            # the donor must not fall apart further: islands with sea islands (Greece's
            # Aegean, Croatia's coast) are in several pieces already, which a plain
            # is_connected check took as "every move splits" and blocked rebalancing
            if rest and nx.number_connected_components(
                G.subgraph(rest)
            ) <= nx.number_connected_components(G.subgraph(members)):
                isl[n] = dst
                S[src] -= load[n]
                S[dst] += load[n]
                moves += 1
                moved_load += load[n]
                done = True
                break
        if not done:
            break
    lau["island"] = pd.Series(isl)
    return moves, moved_load / max(sum(S.values()), 1e-12)


def snap_to_lau(C, isl, lau_zip, loads, border_min, bounds=None):
    """
    Each municipality to the island holding most of its load; islands kept in one piece;
    with `bounds`, border municipalities then moved until the islands respect them.
    """
    lau = gpd.read_file(f"zip://{lau_zip}!LAU_RG_01M_2019_3035.geojson")
    lau = lau[lau.CNTR_CODE == {"GR": "EL", "GB": "UK"}.get(C, C)]
    lau = lau[["GISCO_ID", "LAU_NAME", "geometry"]].to_crs(CRS).reset_index(drop=True)
    # plain Latin letters in every output (Lithuanian, Greek, Cyrillic names transliterated)
    lau["LAU_NAME"] = lau["LAU_NAME"].astype(str).map(unidecode)
    inter = gpd.overlay(
        lau.reset_index().rename(columns={"index": "lau"})[["lau", "geometry"]],
        isl[["island", "geometry"]],
    )
    inter["a"] = inter.geometry.area
    j = gpd.sjoin(
        loads.points, inter[["lau", "island", "geometry"]], predicate="within"
    )
    j = j[~j.index.duplicated()]
    A = inter.groupby(["lau", "island"]).a.sum().unstack(fill_value=0)
    L = j.groupby(["lau", "island"])["load"].sum().unstack(fill_value=0)
    L = L.reindex(index=A.index, columns=A.columns, fill_value=0)
    assign = pd.Series(
        np.where(L.sum(axis=1) > 0, L.idxmax(axis=1), A.idxmax(axis=1)), index=A.index
    )
    lau = lau.loc[assign.index].copy()
    lau["island"] = assign.values
    lau["load"] = L.sum(axis=1).values

    # keep each island one piece on the map: municipalities linked if they share > border_min
    G = nx.Graph()
    G.add_nodes_from(lau.index)
    blen = {}
    sidx = lau.sindex
    for i, geom in zip(lau.index, lau.geometry):
        for jj in sidx.query(geom, predicate="intersects"):
            jj = lau.index[jj]
            if jj > i:
                bl = geom.intersection(lau.geometry[jj]).length
                if bl > border_min:
                    G.add_edge(i, jj)
                    blen[(i, jj)] = bl
    reassigned = 0
    for _ in range(50):
        changed = False
        for isl_id, grp in lau.groupby("island"):
            comps = sorted(
                nx.connected_components(G.subgraph(grp.index)),
                key=lambda c: -lau.loc[list(c), "load"].sum(),
            )
            for comp in comps[1:]:
                votes = {}
                for i in comp:
                    for jj in G.neighbors(i):
                        if jj not in comp and lau.at[jj, "island"] != isl_id:
                            o = lau.at[jj, "island"]
                            votes[o] = votes.get(o, 0) + blen[(min(i, jj), max(i, jj))]
                if votes:  # an enclave or exclave bordering another island
                    lau.loc[list(comp), "island"] = max(votes, key=votes.get)
                    reassigned += len(comp)
                    changed = True
        if not changed:
            break
    if bounds is not None:
        separate = set(isl.loc[isl.separate_land_mass.astype(bool), "island"])
        n_moves, share = rebalance_lau(lau, G, L, separate, bounds)
        say(
            f"LAU rebalancing to [{bounds[0]:g}, {bounds[1]:g}] x mean: {n_moves} "
            f"municipalities moved ({share:.1%} of the load)"
        )
    total = L.sum().sum()
    moved = (total - sum(L.at[i, a] for i, a in lau["island"].items())) / total
    snapped = lau.dissolve("island", aggfunc={"load": "sum"}).reset_index()
    snapped = snapped.rename(columns={"load": "load_share"})
    snapped["load_share"] /= snapped["load_share"].sum()
    say(
        f"LAU snapping: {len(lau)} municipalities; {moved:.1%} of the load changes island "
        f"({reassigned} municipalities reassigned to keep islands in one piece)"
    )
    return snapped, lau, moved


def copperplate(C, cb, lines, t, ehv, isl, load_points):
    """Load outside each island's connected core: all voltages and below the threshold."""
    sub = cb[cb.voltage < ehv]

    def graph(ids):
        G = nx.Graph()
        G.add_nodes_from(sorted(ids))
        ll = lines[lines.bus0.isin(ids) & lines.bus1.isin(ids)]
        G.add_edges_from(zip(ll.bus0, ll.bus1))
        tt = t[t.bus0.isin(ids) & t.bus1.isin(ids)]
        G.add_edges_from(zip(tt.bus0, tt.bus1))
        return G

    G_sub, G_all = graph(set(sub.index)), graph(set(cb.index))
    # load per substation: the load-grid points of its Voronoi cell (nearest site), split
    # among the buses at that site
    site = station_of(sub.index)
    site_pt = station_points(sub, site)
    sites = pd.Index(site_pt.index)
    _, j = cKDTree([[q.x, q.y] for q in site_pt]).query(load_points[["x", "y"]].values)
    site_load = pd.Series(load_points["load"].values).groupby(sites[j]).sum()
    load = (site.map(site_load).fillna(0) / site.map(site.value_counts())).reindex(
        cb.index
    )
    load = load.fillna(0.0)
    own = gpd.sjoin_nearest(cb[["geometry"]], isl[["island", "geometry"]])["island"]
    own = own[~own.index.duplicated()].reindex(cb.index)

    def outside_core(graph_, members):
        comps = sorted(
            nx.connected_components(graph_.subgraph(members)),
            key=lambda c: -load.reindex(list(c)).sum(),
        )
        return len(comps), load.reindex(members).sum() - load.reindex(
            list(comps[0])
        ).sum()

    rows = []
    for i, members in own.groupby(own).groups.items():
        members = list(members)
        members_sub = [m for m in members if m in G_sub]
        tot = load.reindex(members).sum()
        n_all, out_all = outside_core(G_all, members)
        n_sub, out_sub = outside_core(G_sub, members_sub) if members_sub else (0, 0.0)
        rows.append(
            dict(
                country=C,
                island=i,
                load_share=tot,
                networks_all=n_all,
                outside_share_all=out_all,
                outside_of_island_all=out_all / tot if tot > 0 else 0.0,
                networks_below=n_sub,
                outside_share_below=out_sub,
                outside_of_island_below=out_sub / tot if tot > 0 else 0.0,
            )
        )
    df = pd.DataFrame(rows).sort_values("island")
    say(
        f"copperplate check (load outside its island's connected core): all voltages "
        f"{df.outside_share_all.sum():.1%} (worst island {df.outside_of_island_all.max():.0%}), "
        f"below {ehv:g} kV {df.outside_share_below.sum():.1%} "
        f"(worst {df.outside_of_island_below.max():.0%})"
    )
    return df


def plot_islands(path, title, shape, isl, lin, hubs, ehv, inner=None, outline=None):
    """
    Map of the islands with all lines and hubs; `inner`: thin boundaries (supply areas or
    municipalities); `outline`: grid-based island outlines (white dashed) under snapped ones.
    """
    minx, miny, maxx, maxy = shape.bounds
    asp = (maxy - miny) / (maxx - minx)
    fig, ax = plt.subplots(figsize=(11, max(7, min(18, 11 * asp))))
    gpd.GeoSeries([shape], crs=CRS).plot(ax=ax, color="#f7f7f7", edgecolor="#333333")
    cmap = plt.get_cmap("tab20" if len(isl) > 8 else "Set2")
    isl.plot(ax=ax, color=[cmap((int(i) - 1) % cmap.N) for i in isl.island], alpha=0.45)
    handles = []
    if inner is not None:
        inner_gdf, inner_style, inner_label = inner
        inner_gdf.boundary.plot(ax=ax, **inner_style)
        if inner_label:
            handles.append(Line2D([], [], label=inner_label, **inner_style))
    for lbl, grp in lin.groupby(
        lin.voltage.map(lambda v: vstyle(v, ehv)[0]), sort=False
    ):
        _, col, lw = vstyle(grp.voltage.iloc[0], ehv)
        grp.plot(ax=ax, color=col, lw=lw, zorder=3)
        handles.append(
            Line2D([], [], color=col, lw=max(lw, 1.5), label=f"{lbl} ({len(grp)})")
        )
    if outline is not None:
        halo = [pe.Stroke(linewidth=2.2, foreground="#555555"), pe.Normal()]
        outline.boundary.plot(
            ax=ax, color="white", lw=1.2, linestyle="--", zorder=4, path_effects=halo
        )
        handles.append(
            Line2D(
                [],
                [],
                color="white",
                lw=1.2,
                ls="--",
                path_effects=halo,
                label="grid-based outline",
            )
        )
    isl.boundary.plot(
        ax=ax, color="black", lw=1.4 if outline is not None else 2.2, zorder=5
    )
    hubs.plot(ax=ax, color="crimson", marker="*", markersize=30, zorder=6)
    for x in isl.itertuples():
        p = x.geometry.representative_point()
        ax.annotate(
            f"{x.island}\n{x.peak_GW:.1f} GW",
            (p.x, p.y),
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )
    handles += [
        Line2D([], [], color="black", lw=1.6, label=f"islands ({len(isl)})"),
        Line2D([], [], color="crimson", marker="*", ls="", label=f"hubs ({len(hubs)})"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    ax.set_xlim(minx - 2e4, maxx + 2e4)
    ax.set_ylim(miny - 2e4, maxy + 2e4)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_aro_islands", country="LT")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    C = snakemake.wildcards.country
    cfg = snakemake.params.partition
    EHV = cfg["voltage_threshold"]
    BORDER = cfg["border_min_m"]
    tmin, tmax = cfg["size_bounds"]
    kt, X = islands_per_country(snakemake.input.peaks, snakemake.params.countries, cfg)
    K_total, peak_gw = int(kt.at[C, "k"]), float(kt.at[C, "peak_GW"])

    b = read_release(snakemake.input.buses)
    lines = read_release(snakemake.input.lines)
    t = read_release(snakemake.input.transformers)
    conv = read_release(snakemake.input.converters)
    b = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b.x, b.y), crs=4326).to_crs(CRS)
    dc = b["dc"].astype(str).str.lower().isin(["t", "true"])
    cb = b[(b.country == C) & ~dc].set_index("bus_id")
    sub = cb[cb.voltage < EHV]
    shapes = gpd.read_file(snakemake.input.country_shapes)
    shape_raw = shapes.loc[shapes["name"] == C].to_crs(CRS).union_all()
    shape = dry_outline(shape_raw, cfg["water_closing_m"])
    say(
        f"# {C}: k = {K_total} (peak {peak_gw:.1f} GW, target island size {X:.2f} GW), "
        f"voltages below {EHV:g} kV"
    )
    load_points = build_load_grid(
        C,
        shape_raw,
        snakemake.input,
        snakemake.params.distribution_key,
        kt.at[C, "electricity_GW"],
        kt.at[C, "heat_GW"],
    )
    loads = LoadGrid(load_points)
    say(f"buses below {EHV:g} kV: {len(sub)}")

    # ---- 1 graph
    ids = set(sub.index)
    l = lines[
        (lines.voltage < EHV)
        & lines.bus0.isin(ids)
        & lines.bus1.isin(ids)
        & (lines.bus0 != lines.bus1)
    ]
    tr = t[t.bus0.isin(ids) & t.bus1.isin(ids)]
    G = nx.Graph()
    G.add_nodes_from(
        sorted(ids)
    )  # sorted: set order would break shortest-path ties randomly
    for r in l.itertuples():
        d = G.get_edge_data(r.bus0, r.bus1, {"km": np.inf, "s": 0.0})
        G.add_edge(r.bus0, r.bus1, km=min(d["km"], r.length / 1e3), s=d["s"] + r.s_nom)
    for r in tr.itertuples():
        d = G.get_edge_data(r.bus0, r.bus1, {"s": 0.0})
        G.add_edge(r.bus0, r.bus1, km=0.0, s=d["s"] + r.s_nom)
    say(f"lines below {EHV:g} kV: {len(l)}; transformers between them: {len(tr)}")

    # ---- 2 hubs (one per station)
    ehv_ids = set(cb.index[cb.voltage >= EHV])
    # a DC converter's AC side below the threshold feeds the network like a transformer from
    # it: without it Corsica (90 kV, fed only by SACOI) had no hub and its buses went to the
    # nearest French hub across the sea, merging it into the Marseille island
    landing = set(conv.bus0) | set(conv.bus1)
    landing = {x for x in landing if x in ids}
    hub_bus = sorted(
        set(t.bus1[t.bus0.isin(ehv_ids) & t.bus1.isin(ids)])
        | set(t.bus0[t.bus1.isin(ehv_ids) & t.bus0.isin(ids)])
        | landing
    )
    if not hub_bus:
        raise ValueError(f"{C}: no hub (transformer from >= {EHV:g} kV) in the network")
    site_of = station_of(sub.index)
    site_pt = station_points(sub, site_of)
    hub_site = site_of[hub_bus]
    hub_names = {s: f"H{i + 1:03d}" for i, s in enumerate(sorted(set(hub_site.values)))}
    say(f"hubs: {len(hub_bus)} buses at {len(hub_names)} stations")

    # ---- 3 supply areas
    _, path = nx.multi_source_dijkstra(G, hub_bus, weight="km")
    area = pd.Series(
        {n: hub_names[hub_site[p[0]]] for n, p in path.items()}, dtype=object
    ).reindex(sub.index)
    miss = area.index[area.isna()]
    if len(miss):
        hub_xy = np.array([[site_pt[s].x, site_pt[s].y] for s in hub_names])
        _, j = cKDTree(hub_xy).query(
            np.c_[sub.loc[miss].geometry.x, sub.loc[miss].geometry.y]
        )
        area[miss] = [list(hub_names.values())[i] for i in j]
    say(f"buses reached along lines: {1 - len(miss) / len(sub):.0%}")

    sites = site_of.unique()
    vor = shapely.voronoi_polygons(
        shapely.MultiPoint([site_pt[s] for s in sites]),
        extend_to=shape.buffer(3e5),
    )
    cells = gpd.GeoDataFrame(geometry=list(vor.geoms), crs=CRS)
    cells = gpd.sjoin(
        cells,
        gpd.GeoDataFrame(
            {"site": list(sites)}, geometry=[site_pt[s] for s in sites], crs=CRS
        ),
        predicate="contains",
    ).drop(columns="index_right")
    cells["geometry"] = cells.intersection(shape)
    # A cell clipped by water can fall into land pieces on both banks; each piece on its own,
    # else the piece on the far bank jumps islands with its load (Hamburg, 2.7 GW).
    cells = cells.explode(index_parts=False).reset_index(drop=True)
    cells = cells[~cells.is_empty].reset_index(drop=True)
    site_area = area.groupby(site_of).agg(lambda s: s.mode().iloc[0])
    cells["area"] = cells.site.map(site_area)

    # Contiguous supply areas, graph-based (a move-the-pieces loop oscillated along Norway's
    # fjords): cells linked if they share > border_min_m of border; each area keeps its
    # component holding its hub (or its largest), released cells grow onto adjacent kept areas,
    # longest shared border first; a cell touching nothing keeps its area.
    cg = nx.Graph()
    cg.add_nodes_from(cells.index)
    blen = {}
    sidx = cells.sindex
    for i_, g_ in zip(cells.index, cells.geometry):
        for j_ in sidx.query(g_, predicate="intersects"):
            j_ = cells.index[j_]
            if j_ > i_:
                bl = g_.intersection(cells.geometry[j_]).length
                if bl > BORDER:
                    cg.add_edge(i_, j_)
                    blen[(i_, j_)] = bl
    hub_pt = {name: site_pt[site] for site, name in hub_names.items()}
    hub_cell = {}
    for ar_, pt_ in hub_pt.items():
        hit = cells.index[
            (cells["area"] == ar_) & cells.geometry.buffer(1).contains(pt_)
        ]
        if len(hit):
            hub_cell[ar_] = hit[0]
    orig = cells["area"].copy()
    free = set()
    for ar_, grp in cells.groupby("area"):
        comps = list(nx.connected_components(cg.subgraph(grp.index)))
        if len(comps) <= 1:
            continue
        keep = next((c for c in comps if hub_cell.get(ar_) in c), None)
        if keep is None:
            keep = max(comps, key=lambda c: cells.geometry[list(c)].area.sum())
        for c in comps:
            if c is not keep:
                free |= c
    cells.loc[list(free), "area"] = None
    n_fixed = 0
    while True:
        grabbed = {}
        for i_ in free:
            votes = {}
            for j_ in cg.neighbors(i_):
                ar_ = cells.at[j_, "area"]
                if isinstance(ar_, str):
                    votes[ar_] = votes.get(ar_, 0) + blen[(min(i_, j_), max(i_, j_))]
            if votes:
                grabbed[i_] = max(votes, key=votes.get)
        if not grabbed:
            break
        for i_, ar_ in grabbed.items():
            cells.at[i_, "area"] = ar_
        free -= set(grabbed)
        n_fixed += len(grabbed)
    for i_ in free:
        cells.at[i_, "area"] = orig[i_]
    site_area = (
        cells.groupby("site")["area"]
        .agg(lambda s: s.mode().iloc[0])
        .combine_first(site_area)
    )
    area = site_of.map(site_area)
    say(f"contiguous supply areas: {n_fixed} detached cells reassigned")
    pd.DataFrame(
        {
            "area": area,
            "x": sub.geometry.x,
            "y": sub.geometry.y,
            "voltage": sub.voltage,
            "reached_along_lines": ~sub.index.isin(miss),
            "hub": sub.index.isin(hub_bus),
        }
    ).to_csv(snakemake.output.supply_areas)

    atoms_all = sorted(cells["area"].unique())
    area_shp = cells.dissolve("area").reset_index()
    aload_all = loads.of(area_shp, "area")
    aload_all.index = area_shp["area"]
    aload_all = aload_all.reindex(atoms_all).fillna(0)

    # ---- 4 separate land masses: AC graph at ALL voltages (a 380 kV cable keeps Sicily
    # attached); off the main land piece, a bus at or above the threshold or a DC landing
    # (both are model buses: Corsica's SACOI converter survives a >= 220 kV build), large
    # enough. Smaller land masses stay with the supply area they were assigned to.
    GA = nx.Graph()
    GA.add_nodes_from(cb.index)
    lall = lines[lines.bus0.isin(cb.index) & lines.bus1.isin(cb.index)]
    GA.add_edges_from(zip(lall.bus0, lall.bus1))
    tall = t[t.bus0.isin(cb.index) & t.bus1.isin(cb.index)]
    GA.add_edges_from(zip(tall.bus0, tall.bus1))
    land = sorted(getattr(shape, "geoms", [shape]), key=lambda g: g.area, reverse=True)
    on_main = cb.geometry.within(land[0].buffer(cfg["land_mass_buffer_m"]))
    sep_comp = {}
    for ci, comp in enumerate(nx.connected_components(GA)):
        comp = list(comp)
        km2 = (
            cb.loc[comp]
            .geometry.union_all()
            .convex_hull.buffer(cfg["land_mass_buffer_m"])
            .area
            / 1e6
        )
        if (
            (~on_main[comp]).mean() > 0.5
            and ((cb.loc[comp, "voltage"] >= EHV).any() or landing.intersection(comp))
            and km2 >= cfg["separate_land_mass_km2"]
        ):
            for bb in comp:
                sep_comp[bb] = ci
    separate = {
        hub_names[hub_site[bb]]: sep_comp[bb] for bb in hub_bus if bb in sep_comp
    }
    sea_islands = {}
    for ar, ci in separate.items():
        sea_islands.setdefault(ci, []).append(ar)
    atoms = [x for x in atoms_all if x not in separate]
    # a separate land mass takes one of the k islands only if it is a fair island itself (at
    # least the lower size bound x the country's mean island); a small one comes on top, or
    # the Balearics (0.9 GW) left Spain 4 mainland islands of 12 GW for a 10.6 GW target
    big = [
        ci for ci, ars in sea_islands.items() if aload_all[ars].sum() >= tmin / K_total
    ]
    K = max(1, K_total - len(big))
    if sea_islands:
        say(
            f"separate land masses: {len(sea_islands)}, load "
            f"{[f'{aload_all[ars].sum():.1%}' for ars in sea_islands.values()]}, "
            f"{len(big)} counted within k (at least {tmin / K_total:.1%}); "
            f"{K} islands on the mainland, {K + len(sea_islands)} in total"
        )
    aload = aload_all.reindex(atoms)
    area_shp = area_shp[area_shp["area"].isin(atoms)]

    # ---- 5 neighbours and link weights
    W = {}
    for u, v, d in G.edges(data=True):
        p, q = area[u], area[v]
        if p != q:
            key = tuple(sorted((p, q)))
            W[key] = W.get(key, 0.0) + d["s"]
    touch = {}
    sidx = area_shp.sindex
    aa = list(area_shp["area"])
    for i, g in enumerate(area_shp.geometry):
        for j in sidx.query(g, predicate="intersects"):
            if j > i:
                bl = g.intersection(area_shp.geometry.iloc[j]).length
                if bl > BORDER:
                    touch[tuple(sorted((aa[i], aa[j])))] = bl
    A = nx.Graph()
    A.add_nodes_from(atoms)
    for key in touch:
        if key in W:
            A.add_edge(*key, w=W[key])
    nets = list(
        nx.connected_components(A)
    )  # networks below the threshold, for diagnostics
    say(
        f"neighbours: {A.number_of_edges()} pairs touch and share a line below {EHV:g} kV, "
        f"{sum(1 for key in touch if key not in W)} touch without one"
    )
    med = (
        float(np.median([d["w"] for *_, d in A.edges(data=True)]))
        if A.number_of_edges()
        else 1.0
    )
    for key in touch:
        if not A.has_edge(*key):
            A.add_edge(*key, w=cfg["unlinked_neighbour_weight"] * med)

    aidx = {x: i for i, x in enumerate(atoms)}
    L = aload.values.copy()
    total = L.sum()
    shp_area = area_shp.set_index("area").geometry
    # supply areas off the main body join the nearest island afterwards (linking them
    # artificially made pieces jump between islands); their load counts there already
    tcomps = sorted(
        nx.connected_components(A), key=lambda c: -sum(L[aidx[x]] for x in c)
    )
    main_atoms = sorted(tcomps[0])
    detached = sorted(x for c in tcomps[1:] for x in c)
    say(
        f"supply areas off the main body: {len(detached)} "
        f"({sum(L[aidx[x]] for x in detached) / total:.1%} of the load)"
    )
    if K > len(main_atoms):
        say(
            f"note: only {len(main_atoms)} supply areas on the main body; k reduced from {K}"
        )
        K = len(main_atoms)
    mean = total / K
    near = {
        x: min(main_atoms, key=lambda y: shp_area[x].distance(shp_area[y]))
        for x in detached
    }
    L_grouping = L.copy()
    for x, y in near.items():
        L_grouping[aidx[y]] += L_grouping[aidx[x]]

    # ---- 6 pick within the size bounds: least cut or near-optimal medoid
    labels = np.zeros(len(atoms), dtype=int)
    if K > 1:
        li = {x: i for i, x in enumerate(main_atoms)}
        edges = [
            (li[p], li[q], d["w"])
            for p, q, d in A.subgraph(main_atoms).edges(data=True)
        ]
        Ls = np.array([L_grouping[aidx[x]] for x in main_atoms])
        if len(main_atoms) <= cfg["enumeration_max"]:
            cands = enumerate_groupings(len(main_atoms), edges, K)
        else:
            cands = metis_groupings(
                len(main_atoms),
                edges,
                K,
                Ls,
                cfg["metis_samples"],
                cfg["metis_imbalance"],
                cfg["metis_weight_perturbation"],
                cfg["seed"],
            )
        E_ = np.array([[p, q] for p, q, _ in edges])
        W_ = np.array([w for *_, w in edges])
        As = A.subgraph(main_atoms)
        scored, n_split, n_all = [], 0, 0
        for lab in cands:
            n_all += 1
            if any(
                not nx.is_connected(
                    As.subgraph([x for x, g in zip(main_atoms, lab) if g == gi])
                )
                for gi in set(lab.tolist())
            ):
                n_split += 1
                continue
            lo = np.bincount(lab, weights=Ls, minlength=K)
            cut = W_[lab[E_[:, 0]] != lab[E_[:, 1]]].sum()
            scored.append((lo.min() / mean, lo.max() / mean, cut, lab))
        if n_split:
            say(
                f"dropped {n_split} of {n_all} groupings with an island in several pieces"
            )
        if not scored:
            raise ValueError(f"{C}: no connected grouping into {K} islands")
        ok = [s for s in scored if s[0] >= tmin and s[1] <= tmax]
        if ok:
            least = min(s[2] for s in ok)
            tol = cfg["near_optimal_tolerance"]
            if cfg["selection"] == "near_optimal_medoid":
                best = near_optimal_medoid(ok, Ls, tol, K)
            elif cfg["selection"] == "near_optimal_balanced":
                best = near_optimal_balanced(ok, tol)
            else:
                best = min(ok, key=lambda s: s[2])
            uncertain, n_near = border_uncertainty(ok, best, Ls, tol, K)
            say(
                f"{len(ok)} of {len(scored)} groupings within [{tmin:g}, {tmax:g}] x mean; "
                f"{cfg['selection']} taken: islands {best[0]:.2f}-{best[1]:.2f} x mean, cut "
                f"+{best[2] / least - 1:.1%} over the least; border uncertainty: "
                f"{n_near - 1} other groupings within +{tol:.0%} of the least cut put on "
                f"average {uncertain:.1%} of the load in another island"
            )
        else:
            best = min(scored, key=lambda s: max(tmin - s[0], s[1] - tmax, 0))
            say(
                f"none of {len(scored)} groupings within bounds; closest taken "
                f"({best[0]:.2f}-{best[1]:.2f} x mean)"
            )
        for x, g in zip(main_atoms, best[3]):
            labels[aidx[x]] = g
    for x, y in near.items():
        labels[aidx[x]] = labels[aidx[y]]

    # copperplate check at the supply-area level (the substation-level check, at all voltages
    # and below the threshold, is check_aro_islands)
    net_of = {x: i for i, c in enumerate(nets) for x in c}
    outside = 0.0
    for g in range(K):
        by_net = {}
        for x in atoms:
            if labels[aidx[x]] == g:
                by_net[net_of[x]] = by_net.get(net_of[x], 0) + L[aidx[x]]
        if by_net:
            outside += sum(by_net.values()) - max(by_net.values())
    say(
        f"supply areas outside their island's main network below {EHV:g} kV: "
        f"{outside / total:.1%} of the mainland load"
    )

    # ---- 7 island outlines
    lab_of = dict(zip(atoms, labels))
    for i, ars in enumerate(sea_islands.values()):
        for ar in ars:
            lab_of[ar] = K + i
    cells["island"] = cells["area"].map(lab_of)
    isl = cells.dissolve("island").reset_index()[["island", "geometry"]]
    isl["geometry"] = isl.geometry.buffer(20).buffer(-20).intersection(shape)
    moved = 0
    for _ in range(10):
        parts = isl.explode(index_parts=False).reset_index(drop=True)
        parts["a"] = parts.geometry.area
        main = parts.groupby("island").a.transform("max") == parts.a
        changed = False
        for i, pt in parts[~main].iterrows():
            nb = parts[parts.island != pt.island]
            shared = nb.geometry.intersection(pt.geometry.buffer(1)).length
            if shared.max() > 0:
                parts.loc[i, "island"] = nb.loc[shared.idxmax(), "island"]
                moved += 1
                changed = True
        isl = parts.dissolve("island").reset_index()[["island", "geometry"]]
        if not changed:
            break
    il = loads.of(isl, "island")
    il.index = isl["island"]
    order = {o: n_ + 1 for n_, o in enumerate(il.sort_values(ascending=False).index)}
    isl["island"] = isl.island.map(order)
    il.index = il.index.map(order)
    isl["load_share"] = isl.island.map(il)
    isl["peak_GW"] = isl.load_share * peak_gw
    isl["separate_land_mass"] = isl.island.isin(
        [order[K + i] for i in range(len(sea_islands))]
    )
    lc = relative_load(isl.load_share, isl.separate_land_mass)
    say(
        f"outlines: {moved} pieces moved; island load {lc.min():.2f}-{lc.max():.2f} x mean"
    )
    for x in isl.sort_values("island").itertuples():
        say(f"  {x.island}: {x.peak_GW:5.1f} GW, {x.geometry.area / 1e6:,.0f} km2")
    isl.insert(0, "country", C)
    isl.to_crs(4326).to_file(snakemake.output.islands_grid)

    lin = lines[lines.bus0.isin(cb.index) | lines.bus1.isin(cb.index)].copy()
    lin = gpd.GeoDataFrame(
        lin, geometry=gpd.GeoSeries.from_wkt(lin.geometry.str.strip("'")), crs=4326
    ).to_crs(CRS)
    hubs = gpd.GeoDataFrame(geometry=[site_pt[s_] for s_ in hub_names], crs=CRS)
    plot_islands(
        snakemake.output.map_grid,
        f"{C}: {len(isl)} outage islands from the grid below {EHV:g} kV "
        f"(peak {peak_gw:.1f} GW, islands {lc.min():.2f}-{lc.max():.2f} x mean)",
        shape,
        isl,
        lin,
        hubs,
        EHV,
        inner=(
            area_shp,
            dict(color="0.5", lw=0.4, ls=":"),
            f"supply areas ({len(atoms_all)})",
        ),
    )

    # ---- 9 municipal outlines
    if cfg["snap_to_lau"]:
        snapped, lau, moved = snap_to_lau(
            C,
            isl,
            snakemake.input.lau,
            loads,
            cfg["lau_border_min_m"],
            (tmin, tmax) if cfg["lau_respect_bounds"] else None,
        )
        snapped["peak_GW"] = snapped.load_share * peak_gw
        snapped = snapped.merge(
            isl[["island", "separate_land_mass"]], on="island", how="left"
        )
        snapped.insert(0, "country", C)
        lau[["GISCO_ID", "LAU_NAME", "island", "load"]].to_csv(
            snakemake.output.lau_assignment, index=False
        )
        after = relative_load(
            snapped.load_share, snapped.separate_land_mass.fillna(False)
        )
        say(f"after snapping: island load {after.min():.2f}-{after.max():.2f} x mean")
        final = snapped
        plot_islands(
            snakemake.output.map,
            f"{C}: {len(snapped)} islands snapped to municipalities; peak {peak_gw:.1f} GW; "
            f"island load {after.min():.2f}-{after.max():.2f} x mean\nload changing island "
            f"through snapping: {moved:.1%}",
            shape,
            snapped,
            lin,
            hubs,
            EHV,
            inner=(lau, dict(color="white", lw=0.12), None),
            outline=isl,
        )
    else:
        final = isl
        pd.DataFrame(columns=["GISCO_ID", "LAU_NAME", "island", "load"]).to_csv(
            snakemake.output.lau_assignment, index=False
        )
        plot_islands(
            snakemake.output.map,
            f"{C}: {len(isl)} outage islands (grid-based outlines)",
            shape,
            isl,
            lin,
            hubs,
            EHV,
        )
    final.to_crs(4326).to_file(snakemake.output.islands)

    # ---- 10 copperplate check
    copperplate(C, cb, lines, t, EHV, final, load_points).to_csv(
        snakemake.output.copperplate, index=False
    )
    with open(snakemake.output.summary, "w") as f:
        f.write("\n".join(log_lines) + "\n")
