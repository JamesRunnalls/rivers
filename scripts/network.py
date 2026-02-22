"""Extract the river network downstream of headwater points.

Reads:
  - geodata/headwaters.geojson  — points defining major tributary headwaters
  - geodata/lakes.geojson       — lake polygons
  - swissTLM3D FlowingWater shapefile — full directed river network

Outputs:
  - public/geodata/rivers.geojson — filtered network, lake-interior segments removed (WGS84)
  - public/geodata/sinks.geojson  — network termination points with type property:
        "outlet"      — true river outlet (end of network, not in a lake)
        "lake_entry"  — where a river flows into a lake
        "lake_exit"   — where a river flows out of a lake (lake source)
        "lake_source" — headwater whose snap node lies inside / near a lake

Run from the project root:
  python scripts/network.py
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths relative to project root
ROOT = Path(__file__).parent.parent
HEADWATERS_PATH = ROOT / "public/geodata/inputs/headwaters.geojson"
LAKES_PATH = ROOT / "public/geodata/inputs/lakes.geojson"
RIVERS_PATH = (
    ROOT
    / "external/swisstlm3d_2025-03_2056_5728.shp/TLM_GEWAESSER"
    / "swissTLM3D_TLM_FLIESSGEWAESSER.shp"
)
OUTPUT_PATH = ROOT / "public/geodata/outputs/rivers.geojson"
HEADWATERS_OUT_PATH = ROOT / "public/geodata/outputs/natural_sources.geojson"
LAKE_SOURCES_PATH = ROOT / "public/geodata/outputs/lake_sources.geojson"
SINKS_PATH = ROOT / "public/geodata/outputs/sinks.geojson"

TARGET_CRS = "EPSG:2056"
NODE_PRECISION = 0.1   # metres — coordinates rounded to this grid for node matching
MAX_SNAP_DISTANCE = 300  # metres — headwaters snapped to nearest segment within this distance
LAKE_BUFFER_M = 50  # metres — a point within this distance of a lake is "in" the lake


def round_coord(coord, precision=NODE_PRECISION):
    """Round coordinate to a fixed grid so nearby points map to the same node."""
    factor = 1.0 / precision
    return tuple(round(c * factor) / factor for c in coord[:2])


def extract_lines(geom):
    """Return a flat list of non-empty LineStrings from any Shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty]
    if hasattr(geom, "geoms"):  # GeometryCollection
        result = []
        for g in geom.geoms:
            result.extend(extract_lines(g))
        return result
    return []


def build_graph(rivers):
    """Build a directed NetworkX graph from river segment geometries.

    Tags each edge with `druckstollen=True` for pressure-tunnel segments so the
    tracer can prefer surface-water routes and only fall back to tunnels when
    there is no other way forward.

    Returns (G, edge_geom) where edge_geom maps (u, v, key) → LineString.
    """
    G = nx.MultiDiGraph()
    edge_geom = {}

    obj_vals = rivers["OBJEKTART"].values if "OBJEKTART" in rivers.columns else None
    geoms = rivers.geometry.values
    n = len(geoms)
    log_step = max(1, n // 10)

    for ridx, line in enumerate(geoms):
        if ridx % log_step == 0:
            logger.info("  Building graph: %d/%d segments", ridx, n)

        if line is None or line.is_empty:
            continue

        is_druckstollen = obj_vals is not None and obj_vals[ridx] == "Druckstollen"
        sub_lines = list(line.geoms) if isinstance(line, MultiLineString) else [line]

        for sub_line in sub_lines:
            coords = list(sub_line.coords)
            if len(coords) < 2:
                continue
            u = round_coord(coords[0])
            v = round_coord(coords[-1])
            if u == v:
                continue
            key = G.add_edge(
                u, v,
                river_idx=ridx,
                length=sub_line.length,
                druckstollen=is_druckstollen,
            )
            edge_geom[(u, v, key)] = sub_line

    return G, edge_geom


def snap_headwaters(headwaters, rivers):
    """Snap each headwater point to the upstream end of the nearest river segment.

    Returns:
        snap_nodes   : list of graph-node coordinate tuples (one per matched headwater)
        snap_points  : list of the original headwater Points in TARGET_CRS (same order)
    """
    sindex = rivers.sindex
    snap_nodes = []
    snap_points = []
    n_failed = 0

    for idx, hw in headwaters.iterrows():
        pt = hw.geometry
        if pt is None or pt.is_empty:
            n_failed += 1
            continue

        try:
            result = sindex.nearest(pt, max_distance=MAX_SNAP_DISTANCE, return_all=True)
            candidates = list(result[1])
        except Exception:
            candidates = []

        if not candidates:
            logger.warning(
                "Headwater %d: no segment within %dm — skipped", idx, MAX_SNAP_DISTANCE
            )
            n_failed += 1
            continue

        # Pick the geometrically closest segment
        best_dist = np.inf
        best_ridx = None

        for cidx in candidates:
            line = rivers.geometry.iloc[cidx]
            if line is None or line.is_empty:
                continue
            seg = list(line.geoms)[0] if isinstance(line, MultiLineString) else line
            d = seg.distance(pt)
            if d < best_dist:
                best_dist = d
                best_ridx = cidx

        if best_ridx is None:
            n_failed += 1
            continue

        line = rivers.geometry.iloc[best_ridx]
        seg = list(line.geoms)[0] if isinstance(line, MultiLineString) else line
        snap_nodes.append(round_coord(list(seg.coords)[0]))
        snap_points.append(pt)

    logger.info(
        "Snapped %d/%d headwaters (failed: %d)",
        len(snap_nodes), len(headwaters), n_failed,
    )
    return snap_nodes, snap_points


def trace_all_downstream(G, snap_nodes):
    """Trace downstream from every snap node, collecting visited edge tuples.

    At each node the tracer prefers surface-water edges over Druckstollen and,
    among those, follows the longest outgoing edge (main-channel heuristic).
    Druckstollen are used only when there is no surface-water alternative.

    Returns a set of (u, v, key) edge tuples.
    """
    all_visited = set()

    for i, start_node in enumerate(snap_nodes):
        if i % 50 == 0:
            logger.info(
                "  Tracing headwater %d/%d — %d edges collected so far",
                i, len(snap_nodes), len(all_visited),
            )

        if start_node not in G:
            continue

        current = start_node
        local_visited = set()

        for _ in range(200_000):  # safety limit against infinite loops
            out_edges = list(G.out_edges(current, keys=True, data=True))
            if not out_edges:
                break

            unvisited = [
                (u, v, k, d)
                for u, v, k, d in out_edges
                if (u, v, k) not in local_visited
            ]
            if not unvisited:
                break

            # Prefer surface water; fall back to Druckstollen only if necessary
            surface = [(u, v, k, d) for u, v, k, d in unvisited if not d.get("druckstollen")]
            pool = surface if surface else unvisited

            u, v, k, data = max(pool, key=lambda x: x[3].get("length", 0))
            local_visited.add((u, v, k))
            all_visited.add((u, v, k))
            current = v

    return all_visited


def merge_network(visited_edges, G, edge_geom):
    """Merge chains of edges between junction nodes into single LineStrings.

    A junction node is any node that is not a simple pass-through (i.e. it has
    in-degree ≠ 1 or out-degree ≠ 1 in the visited subgraph). Edges between
    two consecutive junction nodes are concatenated into one LineString.

    Returns a list of Shapely LineString / MultiLineString geometries.
    """
    # Build subgraph of visited edges only
    H = nx.MultiDiGraph()
    for u, v, k in visited_edges:
        data = G.edges[u, v, k]
        H.add_edge(u, v, key=k, **data)

    # Junction nodes: sources, sinks, confluences, bifurcations
    junction_nodes = {
        node for node in H.nodes()
        if H.in_degree(node) != 1 or H.out_degree(node) != 1
    }

    logger.info(
        "Subgraph: %d nodes, %d edges, %d junction nodes",
        H.number_of_nodes(), H.number_of_edges(), len(junction_nodes),
    )

    output_geoms = []
    output_ridxs = []  # parallel list: river_idx values for each merged reach
    seen_edges = set()

    for start_node in junction_nodes:
        for u, v, k in H.out_edges(start_node, keys=True):
            if (u, v, k) in seen_edges:
                continue

            # Walk the chain from this edge until the next junction node or dead end
            chain = []
            chain_ridxs = []
            cu, cv, ck = u, v, k

            while True:
                seen_edges.add((cu, cv, ck))
                geom = edge_geom.get((cu, cv, ck))
                if geom is not None:
                    chain.append(geom)
                ridx = H.edges[cu, cv, ck].get("river_idx")
                if ridx is not None:
                    chain_ridxs.append(ridx)

                if cv in junction_nodes:
                    break

                # Follow the single outgoing edge from this pass-through node
                next_edges = list(H.out_edges(cv, keys=True))
                if not next_edges or next_edges[0] in seen_edges:
                    break
                cu, cv, ck = next_edges[0]

            if not chain:
                continue

            merged = linemerge(chain) if len(chain) > 1 else chain[0]
            output_geoms.append(merged)
            output_ridxs.append(chain_ridxs)

    return output_geoms, output_ridxs


def _extract_points(geom):
    """Recursively extract all Point geometries from a Shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom]
    if hasattr(geom, "geoms"):
        result = []
        for g in geom.geoms:
            result.extend(_extract_points(g))
        return result
    return []


def process_lakes(merged_geoms, lake_union):
    """Clip merged river geometries against lake polygons.

    River segments that pass through a lake are split at the lake boundary:
    the lake-interior portions are removed and their endpoints are recorded
    as lake entry / exit points.

    Returns:
        clipped_geoms  : river segments outside lakes
        lake_entries   : Points (EPSG:2056) where rivers enter lakes
        lake_exits     : Points (EPSG:2056) where rivers exit lakes
    """
    clipped_geoms = []
    lake_entries = []
    lake_exits = []

    for geom in merged_geoms:
        if not geom.intersects(lake_union):
            clipped_geoms.append(geom)
            continue

        # Keep the parts outside the lake
        outside = geom.difference(lake_union)
        outside_lines = extract_lines(outside)
        clipped_geoms.extend(outside_lines)

        if not outside_lines:
            # Line is entirely inside the lake — no boundary crossing, so no
            # entry/exit points to record (the segment is simply dropped).
            continue

        # Line crosses the lake boundary: find the exact boundary crossing points
        # by intersecting with the lake boundary rather than using segment endpoints.
        # This guarantees the points sit on the boundary, not inside the lake.
        crossings = geom.intersection(lake_union.boundary)
        cross_pts = _extract_points(crossings)
        if not cross_pts:
            continue

        # Sort crossings by distance along the line, then classify as entry/exit
        # by tracking whether we are currently inside or outside the lake.
        cross_pts.sort(key=lambda p: geom.project(p))

        # Determine state at the very start of the line (interpolate 1 cm in to
        # avoid boundary ambiguity when the line begins exactly on the shore).
        probe = geom.interpolate(min(0.01, geom.length * 0.01))
        inside_lake = lake_union.contains(probe)

        for pt in cross_pts:
            if inside_lake:
                lake_exits.append(pt)
            else:
                lake_entries.append(pt)
            inside_lake = not inside_lake

    logger.info(
        "Lake clipping: %d lines → %d clipped, %d entries, %d exits",
        len(merged_geoms), len(clipped_geoms), len(lake_entries), len(lake_exits),
    )
    return clipped_geoms, lake_entries, lake_exits


def main():
    # --- Load data ---
    logger.info("Loading headwaters from %s", HEADWATERS_PATH)
    headwaters = gpd.read_file(HEADWATERS_PATH)
    if headwaters.crs is None or headwaters.crs.to_epsg() != 2056:
        logger.info("Reprojecting headwaters to EPSG:2056")
        headwaters = headwaters.to_crs(TARGET_CRS)
    logger.info("Loaded %d headwater points", len(headwaters))

    logger.info("Loading lakes from %s", LAKES_PATH)
    lakes = gpd.read_file(LAKES_PATH, on_invalid='fix')
    if lakes.crs is None or lakes.crs.to_epsg() != 2056:
        lakes = lakes.to_crs(TARGET_CRS)
    lake_union = lakes.geometry.union_all()
    logger.info("Loaded %d lake polygons", len(lakes))

    logger.info("Loading river network from %s", RIVERS_PATH)
    rivers = gpd.read_file(RIVERS_PATH)
    logger.info("Loaded %d river segments", len(rivers))

    # --- Build directed graph (all segments; Druckstollen tagged but not removed) ---
    logger.info("Building directed graph...")
    G, edge_geom = build_graph(rivers)
    logger.info(
        "Graph complete: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )

    # --- Snap headwaters ---
    snap_nodes, snap_points = snap_headwaters(headwaters, rivers)

    # --- Trace downstream ---
    logger.info("Tracing downstream from %d snapped headwaters...", len(snap_nodes))
    visited_edges = trace_all_downstream(G, snap_nodes)
    logger.info("Collected %d unique edges", len(visited_edges))

    # --- Merge chains between junctions ---
    logger.info("Merging edge chains between junction nodes...")
    merged_geoms, merged_ridxs = merge_network(visited_edges, G, edge_geom)
    logger.info("Merged into %d lines", len(merged_geoms))

    # --- Compute modal river name for each merged reach ---
    name_vals = rivers["NAME"].values if "NAME" in rivers.columns else None

    def modal_name(ridxs):
        """Return the most common non-null NAME among the given river_idx values."""
        if name_vals is None:
            return None
        counts = {}
        for ridx in ridxs:
            n = name_vals[ridx]
            if n and str(n).strip() and str(n).lower() != "nan":
                counts[n] = counts.get(n, 0) + 1
        return max(counts, key=counts.get) if counts else None

    reach_names = [modal_name(ridxs) for ridxs in merged_ridxs]

    # --- Clip against lakes, carrying names through in a single pass ---
    logger.info("Clipping river lines against lake polygons...")
    clipped_with_names = []
    lake_entries = []
    lake_exits = []

    for geom, name in zip(merged_geoms, reach_names):
        if not geom.intersects(lake_union):
            clipped_with_names.append((geom, name))
            continue

        outside_lines = extract_lines(geom.difference(lake_union))
        for seg in outside_lines:
            clipped_with_names.append((seg, name))

        if not outside_lines:
            continue  # entirely inside lake — drop silently, no entry/exit points

        # Find exact boundary crossing points and classify as entry/exit
        crossings = geom.intersection(lake_union.boundary)
        cross_pts = _extract_points(crossings)
        if not cross_pts:
            continue

        cross_pts.sort(key=lambda p: geom.project(p))
        probe = geom.interpolate(min(0.01, geom.length * 0.01))
        inside_lake = lake_union.contains(probe)

        for pt in cross_pts:
            if inside_lake:
                lake_exits.append(pt)
            else:
                lake_entries.append(pt)
            inside_lake = not inside_lake

    logger.info(
        "Lake clipping: %d reaches → %d segments, %d entries, %d exits",
        len(merged_geoms), len(clipped_with_names), len(lake_entries), len(lake_exits),
    )

    # --- Export rivers ---
    logger.info("Reprojecting and writing rivers...")
    river_gdf = gpd.GeoDataFrame(
        geometry=[g for g, _ in clipped_with_names], crs=TARGET_CRS
    ).to_crs("EPSG:4326")

    river_features = []
    for (_, name), geom in zip(clipped_with_names, river_gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        mls = geom if isinstance(geom, MultiLineString) else MultiLineString([geom])
        river_features.append(
            {"type": "Feature", "properties": {"name": name}, "geometry": mls.__geo_interface__}
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"type": "FeatureCollection", "features": river_features}, f)
    logger.info("Saved %d river features to %s", len(river_features), OUTPUT_PATH)

    # --- Build and classify all point features ---
    # Graph-level sinks: nodes that are destinations but never sources in visited set
    graph_sources = {u for u, v, k in visited_edges}
    graph_sinks = {v for u, v, k in visited_edges} - graph_sources

    sink_pts_2056 = []       # true outlets + lake entries
    lake_source_pts_2056 = []  # lake exits + headwaters starting in a lake

    for node in graph_sinks:
        pt = Point(node)
        if lake_union.distance(pt) < LAKE_BUFFER_M:
            # Skip — this node is inside or on the lake boundary; the geometry-level
            # lake_entries (clipped at the exact lake boundary) represent this better.
            continue
        sink_pts_2056.append(pt)

    # Geometry-level lake entries (at the lake boundary, from process_lakes clipping)
    sink_pts_2056.extend(lake_entries)
    # Geometry-level lake exits
    lake_source_pts_2056.extend(lake_exits)

    # Headwater snap points that lie inside / near a lake → lake sources
    hw_pts_2056 = []
    for pt in snap_points:
        if lake_union.distance(pt) < LAKE_BUFFER_M:
            lake_source_pts_2056.append(pt)
        else:
            hw_pts_2056.append(pt)

    def pts_to_geojson(pts_2056):
        """Reproject a list of EPSG:2056 Points and return a GeoJSON FeatureCollection."""
        if not pts_2056:
            return {"type": "FeatureCollection", "features": []}
        gdf = gpd.GeoDataFrame(geometry=pts_2056, crs=TARGET_CRS).to_crs("EPSG:4326")
        features = [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "Point", "coordinates": [g.x, g.y]}}
            for g in gdf.geometry if g is not None and not g.is_empty
        ]
        return {"type": "FeatureCollection", "features": features}

    # --- Write the four output files ---
    with open(HEADWATERS_OUT_PATH, "w") as f:
        gc = pts_to_geojson(hw_pts_2056)
        json.dump(gc, f)
    logger.info("Saved %d headwater points to %s", len(gc["features"]), HEADWATERS_OUT_PATH)

    with open(LAKE_SOURCES_PATH, "w") as f:
        gc = pts_to_geojson(lake_source_pts_2056)
        json.dump(gc, f)
    logger.info("Saved %d lake-source points to %s", len(gc["features"]), LAKE_SOURCES_PATH)

    with open(SINKS_PATH, "w") as f:
        gc = pts_to_geojson(sink_pts_2056)
        json.dump(gc, f)
    logger.info("Saved %d sink points to %s", len(gc["features"]), SINKS_PATH)

    print(f"\nDone.")
    print(f"  {len(river_features):>6} river reaches  → {OUTPUT_PATH}")
    print(f"  {len(hw_pts_2056):>6} headwaters     → {HEADWATERS_OUT_PATH}")
    print(f"  {len(lake_source_pts_2056):>6} lake sources   → {LAKE_SOURCES_PATH}")
    print(f"  {len(sink_pts_2056):>6} sinks          → {SINKS_PATH}")


if __name__ == "__main__":
    main()
