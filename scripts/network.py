"""Extract the river network downstream of headwater points.

Reads:
  - geodata/headwaters.geojson  — points defining major tributary headwaters
  - swissTLM3D FlowingWater shapefile — full directed river network

Outputs:
  - public/geodata/rivers.geojson — filtered network (WGS84 GeoJSON)

Run from the project root:
  python scripts/network.py
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import MultiLineString
from shapely.ops import linemerge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths relative to project root
ROOT = Path(__file__).parent.parent
HEADWATERS_PATH = ROOT / "geodata/headwaters.geojson"
RIVERS_PATH = (
    ROOT
    / "geodata/swisstlm3d_2025-03_2056_5728.shp/TLM_GEWAESSER"
    / "swissTLM3D_TLM_FLIESSGEWAESSER.shp"
)
OUTPUT_PATH = ROOT / "public/geodata/rivers.geojson"
SINKS_PATH = ROOT / "public/geodata/sinks.geojson"

TARGET_CRS = "EPSG:2056"
NODE_PRECISION = 0.1   # metres — coordinates rounded to this grid for node matching
MAX_SNAP_DISTANCE = 300  # metres — headwaters snapped to nearest segment within this distance


def round_coord(coord, precision=NODE_PRECISION):
    """Round coordinate to a fixed grid so nearby points map to the same node."""
    factor = 1.0 / precision
    return tuple(round(c * factor) / factor for c in coord[:2])


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

    Returns a list of graph nodes (rounded coordinate tuples) — one per
    successfully snapped headwater.
    """
    sindex = rivers.sindex
    snap_nodes = []
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

    logger.info(
        "Snapped %d/%d headwaters (failed: %d)",
        len(snap_nodes), len(headwaters), n_failed,
    )
    return snap_nodes


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
    seen_edges = set()

    for start_node in junction_nodes:
        for u, v, k in H.out_edges(start_node, keys=True):
            if (u, v, k) in seen_edges:
                continue

            # Walk the chain from this edge until the next junction node or dead end
            chain = []
            cu, cv, ck = u, v, k

            while True:
                seen_edges.add((cu, cv, ck))
                geom = edge_geom.get((cu, cv, ck))
                if geom is not None:
                    chain.append(geom)

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

    return output_geoms


def main():
    # --- Load data ---
    logger.info("Loading headwaters from %s", HEADWATERS_PATH)
    headwaters = gpd.read_file(HEADWATERS_PATH)
    if headwaters.crs is None or headwaters.crs.to_epsg() != 2056:
        logger.info("Reprojecting headwaters to EPSG:2056")
        headwaters = headwaters.to_crs(TARGET_CRS)
    logger.info("Loaded %d headwater points", len(headwaters))

    logger.info("Loading river network from %s", RIVERS_PATH)
    rivers = gpd.read_file(RIVERS_PATH)
    logger.info("Loaded %d river segments", len(rivers))

    # --- Build directed graph (all segments, Druckstollen tagged but not removed) ---
    logger.info("Building directed graph...")
    G, edge_geom = build_graph(rivers)
    logger.info(
        "Graph complete: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )

    # --- Snap headwaters ---
    snap_nodes = snap_headwaters(headwaters, rivers)

    # --- Trace downstream ---
    logger.info("Tracing downstream from %d snapped headwaters...", len(snap_nodes))
    visited_edges = trace_all_downstream(G, snap_nodes)
    logger.info("Collected %d unique edges", len(visited_edges))

    # --- Merge chains between junctions ---
    logger.info("Merging edge chains between junction nodes...")
    merged_geoms = merge_network(visited_edges, G, edge_geom)
    logger.info("Merged into %d lines", len(merged_geoms))

    # --- Reproject and export ---
    logger.info("Reprojecting and writing output...")
    merged_gdf = gpd.GeoDataFrame(geometry=merged_geoms, crs=TARGET_CRS)
    merged_gdf = merged_gdf.to_crs("EPSG:4326")

    features = []
    for geom in merged_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        mls = geom if isinstance(geom, MultiLineString) else MultiLineString([geom])
        features.append({"type": "Feature", "properties": {}, "geometry": mls.__geo_interface__})

    geojson = {"type": "FeatureCollection", "features": features}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(geojson, f)

    logger.info("Saved %d features to %s", len(features), OUTPUT_PATH)

    # --- Export sink nodes (outlets: nodes reached but never departed from) ---
    sources_in_visited = {u for u, v, k in visited_edges}
    sinks = {v for u, v, k in visited_edges} - sources_in_visited
    logger.info("Found %d sink nodes", len(sinks))

    from shapely.geometry import Point
    sink_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in sinks],
        crs=TARGET_CRS,
    ).to_crs("EPSG:4326")

    sink_features = [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]}}
        for geom in sink_gdf.geometry
        if geom is not None and not geom.is_empty
    ]
    with open(SINKS_PATH, "w") as f:
        json.dump({"type": "FeatureCollection", "features": sink_features}, f)

    logger.info("Saved %d sink points to %s", len(sink_features), SINKS_PATH)
    print(f"\nDone. {len(features)} river reaches → {OUTPUT_PATH}")
    print(f"      {len(sink_features)} sink points  → {SINKS_PATH}")


if __name__ == "__main__":
    main()
