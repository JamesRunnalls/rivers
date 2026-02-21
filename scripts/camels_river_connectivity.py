"""
CAMELS-CH River Connectivity Tool
===================================
Snaps CAMELS-CH gauge locations to the swissTLM3D FlowingWater network,
builds a directed river graph, and traces downstream paths between gauges.

Requirements:
    pip install geopandas networkx shapely rtree pyproj matplotlib

Input data (you must supply these files):
    1. CAMELS-CH gauge shapefile (points)
       Download from: https://doi.org/10.5281/zenodo.7784632
    2. swissTLM3D Hydrographical network — FlowingWater layer
       Download from: https://opendata.swiss/en/dataset/swisstlm3d
       (available as Shapefile or GeoPackage)

Both datasets use EPSG:2056 (CH1903+ / LV95).

Usage:
    # 1. Edit the file paths in the CONFIG section below
    # 2. Run the script:
    python camels_river_connectivity.py

    # Or import and use programmatically:
    from camels_river_connectivity import RiverConnectivity
    rc = RiverConnectivity(gauges_path, rivers_path)
    rc.snap_gauges()
    rc.build_graph()
    path_gdf = rc.trace_downstream(gauge_id=2044)
    path_gdf.to_file("downstream_from_2044.gpkg", driver="GPKG")
"""

import logging
from pathlib import Path
from collections import defaultdict

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring

# ---------------------------------------------------------------------------
# CONFIG — edit these paths to match your local data
# ---------------------------------------------------------------------------
GAUGES_PATH = "camels_ch/gauges.shp"            # CAMELS-CH gauge points
RIVERS_PATH = "swissTLM3D/FlowingWater.shp"     # or .gpkg with layer name
RIVERS_LAYER = None                              # set layer name if using GeoPackage

# If your CAMELS-CH gauge file has a different column name for gauge IDs,
# change it here. The paper uses "gauge_id".
GAUGE_ID_COL = "gauge_id"

# Coordinate precision for node matching (metres). Nodes within this distance
# are treated as the same node. 0.1m is safe for TLM's sub-metre accuracy.
NODE_PRECISION = 0.1  # metres

# Maximum distance (metres) to snap a gauge to a river segment.
# Gauges further away are flagged as unmatched.
MAX_SNAP_DISTANCE = 100  # metres

# Target CRS — both datasets should already be in this CRS
TARGET_CRS = "EPSG:2056"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Helper functions
# ===========================================================================

def round_coord(coord: tuple, precision: float = NODE_PRECISION) -> tuple:
    """Round a coordinate tuple to a given precision for node matching.

    We round to a grid defined by `precision` so that coordinates that are
    nearly identical (within the precision) map to the same graph node.
    """
    factor = 1.0 / precision
    return tuple(round(c * factor) / factor for c in coord[:2])  # drop Z if present


def split_line_at_point(line: LineString, point: Point) -> tuple:
    """Split a LineString at the point nearest to `point`.

    Returns (line_before, line_after, projected_point) where:
    - line_before: segment from line start to projected point
    - line_after: segment from projected point to line end
    - projected_point: the actual point on the line

    Either segment may be None if the point coincides with a line endpoint.
    """
    dist_along = line.project(point)
    projected = line.interpolate(dist_along)

    # If the point is at the very start or end, no split needed
    if dist_along <= NODE_PRECISION:
        return None, line, projected
    if dist_along >= line.length - NODE_PRECISION:
        return line, None, projected

    line_before = substring(line, 0, dist_along)
    line_after = substring(line, dist_along, line.length)

    return line_before, line_after, projected


# ===========================================================================
# Main class
# ===========================================================================

class RiverConnectivity:
    """Connects CAMELS-CH gauges to the swissTLM3D river network."""

    def __init__(
        self,
        gauges_path: str,
        rivers_path: str,
        rivers_layer: str = None,
        gauge_id_col: str = GAUGE_ID_COL,
        max_snap_distance: float = MAX_SNAP_DISTANCE,
    ):
        """Load gauge and river data.

        Parameters
        ----------
        gauges_path : str
            Path to CAMELS-CH gauge shapefile or GeoPackage.
        rivers_path : str
            Path to swissTLM3D FlowingWater shapefile or GeoPackage.
        rivers_layer : str, optional
            Layer name if rivers_path is a GeoPackage.
        gauge_id_col : str
            Column name for gauge IDs in the gauge file.
        max_snap_distance : float
            Maximum distance (m) to snap a gauge to the nearest river segment.
        """
        logger.info("Loading gauge data from %s", gauges_path)
        self.gauges = gpd.read_file(gauges_path)

        logger.info("Loading river data from %s", rivers_path)
        if rivers_layer:
            self.rivers = gpd.read_file(rivers_path, layer=rivers_layer)
        else:
            self.rivers = gpd.read_file(rivers_path)

        self.gauge_id_col = gauge_id_col
        self.max_snap_distance = max_snap_distance

        # Ensure consistent CRS
        if self.gauges.crs and self.gauges.crs.to_epsg() != 2056:
            logger.info("Reprojecting gauges to EPSG:2056")
            self.gauges = self.gauges.to_crs(TARGET_CRS)
        if self.rivers.crs and self.rivers.crs.to_epsg() != 2056:
            logger.info("Reprojecting rivers to EPSG:2056")
            self.rivers = self.rivers.to_crs(TARGET_CRS)

        logger.info(
            "Loaded %d gauges and %d river segments",
            len(self.gauges), len(self.rivers)
        )

        # Will be populated by snap_gauges() and build_graph()
        self.snap_results = None  # DataFrame of snapping results
        self.G = None             # networkx DiGraph
        self._gauge_nodes = {}    # gauge_id → graph node coordinate
        self._edge_geom = {}      # (from_node, to_node, key) → LineString

    # -------------------------------------------------------------------
    # Phase 1: Snap gauges to river network
    # -------------------------------------------------------------------

    def snap_gauges(self) -> gpd.GeoDataFrame:
        """Snap each gauge to the nearest FlowingWater segment.

        Returns a GeoDataFrame with columns:
            gauge_id, original_geom, snapped_geom, river_idx,
            snap_distance, matched
        """
        logger.info("Building spatial index for river segments...")
        sindex = self.rivers.sindex

        results = []

        for idx, gauge in self.gauges.iterrows():
            gid = gauge[self.gauge_id_col]
            pt = gauge.geometry

            if gauge.type == "lake":
                results.append({
                    self.gauge_id_col: gid,
                    "original_geom": pt,
                    "snapped_geom": None,
                    "river_idx": None,
                    "snap_distance": np.inf,
                    "matched": False,
                })
                continue

            if pt is None or pt.is_empty:
                logger.warning("Gauge %s has no geometry, skipping", gid)
                results.append({
                    self.gauge_id_col: gid,
                    "original_geom": pt,
                    "snapped_geom": None,
                    "river_idx": None,
                    "snap_distance": np.inf,
                    "matched": False,
                })
                continue

            # Query spatial index: find nearest segment
            # sindex.nearest returns shape (2, n): [input_indices, tree_indices]
            try:
                result = sindex.nearest(
                    pt, max_distance=self.max_snap_distance, return_all=True
                )
                candidates = list(result[1])  # tree (river) indices
            except Exception:
                candidates = []

            if len(candidates) == 0:
                logger.warning(
                    "Gauge %s: no river segment within %dm",
                    gid, self.max_snap_distance
                )
                results.append({
                    self.gauge_id_col: gid,
                    "original_geom": pt,
                    "snapped_geom": None,
                    "river_idx": None,
                    "snap_distance": np.inf,
                    "matched": False,
                })
                continue

            # Find the closest segment among candidates
            best_dist = np.inf
            best_idx = None
            best_proj = None

            for cidx in candidates:
                line = self.rivers.geometry.iloc[cidx]
                if line is None or line.is_empty:
                    continue
                d = line.distance(pt)
                if d < best_dist:
                    best_dist = d
                    best_idx = cidx
                    best_proj = line.interpolate(line.project(pt))

            matched = best_dist <= self.max_snap_distance

            if not matched:
                logger.warning(
                    "Gauge %s: nearest segment is %.1fm away (> %dm threshold)",
                    gid, best_dist, self.max_snap_distance
                )

            results.append({
                self.gauge_id_col: gid,
                "original_geom": pt,
                "snapped_geom": best_proj,
                "river_idx": int(best_idx) if best_idx is not None else None,
                "snap_distance": best_dist,
                "matched": matched,
            })

        self.snap_results = gpd.GeoDataFrame(
            results, geometry="snapped_geom", crs=self.gauges.crs
        )

        n_matched = self.snap_results["matched"].sum()
        logger.info(
            "Snapping complete: %d/%d gauges matched (within %dm)",
            n_matched, len(self.snap_results), self.max_snap_distance
        )

        # Report statistics
        matched_dists = self.snap_results.loc[
            self.snap_results["matched"], "snap_distance"
        ]
        if len(matched_dists) > 0:
            logger.info(
                "Snap distances — median: %.1fm, 95th pct: %.1fm, max: %.1fm",
                matched_dists.median(),
                matched_dists.quantile(0.95),
                matched_dists.max(),
            )

        return self.snap_results

    # -------------------------------------------------------------------
    # Phase 2: Build directed river graph
    # -------------------------------------------------------------------

    def build_graph(self) -> nx.MultiDiGraph:
        """Build a directed graph from FlowingWater segments and insert gauges.

        Each river segment becomes one or two directed edges. Gauge locations
        are inserted as nodes by splitting the segment they snap to.

        Uses a MultiDiGraph because parallel edges can exist (e.g., braided
        rivers or split channels around islands).
        """
        if self.snap_results is None:
            raise RuntimeError("Call snap_gauges() before build_graph()")

        n_rivers = len(self.rivers)
        logger.info("Building directed graph from %d river segments...", n_rivers)

        G = nx.MultiDiGraph()
        edge_geom = {}

        # --- Identify which segments need splitting at gauge locations ---
        splits = defaultdict(list)

        for _, row in self.snap_results.iterrows():
            if row["matched"] and row["river_idx"] is not None:
                ridx = int(row["river_idx"])
                line = self.rivers.geometry.iloc[ridx]
                dist_along = line.project(row["snapped_geom"])
                splits[ridx].append({
                    "gauge_id": row[self.gauge_id_col],
                    "point": row["snapped_geom"],
                    "dist_along": dist_along,
                })

        for ridx in splits:
            splits[ridx].sort(key=lambda x: x["dist_along"])

        logger.info(
            "%d river segments will be split at gauge locations", len(splits)
        )

        # --- Add edges: vectorized extraction of start/end coords ---
        # For the vast majority of segments (no split), we can batch-process
        geoms = self.rivers.geometry.values  # array of shapely geometries

        n_added = 0
        n_skipped = 0
        log_interval = max(1, n_rivers // 10)

        for ridx in range(n_rivers):
            if ridx > 0 and ridx % log_interval == 0:
                logger.info(
                    "  Progress: %d/%d segments (%.0f%%)",
                    ridx, n_rivers, 100 * ridx / n_rivers
                )

            line = geoms[ridx]
            if line is None or line.is_empty:
                n_skipped += 1
                continue

            # Handle MultiLineString
            if isinstance(line, MultiLineString):
                sub_lines = list(line.geoms)
            else:
                sub_lines = [line]

            for single_line in sub_lines:
                if ridx in splits:
                    self._add_split_edges(
                        G, edge_geom, single_line, ridx, splits[ridx]
                    )
                else:
                    # Fast path: extract coords directly (no split needed)
                    coords = single_line.coords
                    start = round_coord(coords[0])
                    end = round_coord(coords[-1])

                    if start == end:
                        n_skipped += 1
                        continue

                    key = G.add_edge(
                        start, end, river_idx=ridx, length=single_line.length,
                    )
                    edge_geom[(start, end, key)] = single_line
                    n_added += 1

        self.G = G
        self._edge_geom = edge_geom

        logger.info(
            "Graph built: %d nodes, %d edges (%d segments skipped)",
            G.number_of_nodes(), G.number_of_edges(), n_skipped
        )

        # Build gauge_id → node lookup
        self._gauge_nodes = {}
        for _, row in self.snap_results.iterrows():
            if row["matched"] and row["snapped_geom"] is not None:
                node = round_coord(
                    (row["snapped_geom"].x, row["snapped_geom"].y)
                )
                self._gauge_nodes[row[self.gauge_id_col]] = node

        logger.info(
            "%d gauges registered as graph nodes", len(self._gauge_nodes)
        )

        return G

    def _add_edge(self, G, edge_geom, line, ridx, gauge_id=None):
        """Add a single directed edge to the graph."""
        coords = list(line.coords)
        start = round_coord(coords[0])
        end = round_coord(coords[-1])

        if start == end:
            return  # skip zero-length loops

        key = G.add_edge(
            start, end,
            river_idx=ridx,
            length=line.length,
        )
        edge_geom[(start, end, key)] = line

        # Mark nodes as gauge nodes if applicable
        if gauge_id is not None:
            G.nodes[start]["gauge_id"] = gauge_id

    def _add_split_edges(self, G, edge_geom, line, ridx, split_points):
        """Split a river segment at gauge locations and add sub-edges."""
        # Build list of cut distances (including start=0 and end=length)
        cuts = [0.0]
        cut_gauge_ids = [None]  # gauge_id at each cut point (None for start)

        for sp in split_points:
            d = sp["dist_along"]
            # Avoid cuts too close to start or end
            if d > NODE_PRECISION and d < line.length - NODE_PRECISION:
                cuts.append(d)
                cut_gauge_ids.append(sp["gauge_id"])

        cuts.append(line.length)
        cut_gauge_ids.append(None)

        # Create sub-edges between consecutive cut points
        for i in range(len(cuts) - 1):
            sub_line = substring(line, cuts[i], cuts[i + 1])
            if sub_line.is_empty or sub_line.length < NODE_PRECISION:
                continue

            sub_coords = list(sub_line.coords)
            start = round_coord(sub_coords[0])
            end = round_coord(sub_coords[-1])

            if start == end:
                continue

            key = G.add_edge(
                start, end,
                river_idx=ridx,
                length=sub_line.length,
            )
            edge_geom[(start, end, key)] = sub_line

            # Tag the start node with its gauge_id if it's a gauge cut point
            if cut_gauge_ids[i] is not None:
                G.nodes[start]["gauge_id"] = cut_gauge_ids[i]
            # Tag end node for last segment if end is a gauge
            if i == len(cuts) - 2 and cut_gauge_ids[i + 1] is not None:
                G.nodes[end]["gauge_id"] = cut_gauge_ids[i + 1]

    # -------------------------------------------------------------------
    # Phase 3: Trace downstream
    # -------------------------------------------------------------------

    def trace_downstream(
        self,
        gauge_id: int,
        stop_at_gauge: bool = True,
    ) -> gpd.GeoDataFrame:
        """Trace downstream from a gauge, collecting river geometry.

        Parameters
        ----------
        gauge_id : int
            CAMELS-CH gauge_id to start from.
        stop_at_gauge : bool
            If True, stop when reaching another gauge node.
            If False, continue to the end of the network.

        Returns
        -------
        GeoDataFrame with columns:
            segment_order, river_idx, length, geometry
            Plus metadata: start_gauge_id, end_gauge_id (or None), total_length
        """
        if self.G is None:
            raise RuntimeError("Call build_graph() before trace_downstream()")

        if gauge_id not in self._gauge_nodes:
            raise ValueError(
                f"Gauge {gauge_id} not found in graph. "
                f"Available gauges: {sorted(self._gauge_nodes.keys())[:20]}..."
            )

        start_node = self._gauge_nodes[gauge_id]
        gauge_node_set = set(self._gauge_nodes.values()) - {start_node}

        logger.info("Tracing downstream from gauge %s at node %s", gauge_id, start_node)

        # Walk downstream: at each node follow outgoing edges
        # If there are multiple outgoing edges (distributary), pick the longest
        segments = []
        visited_edges = set()
        current = start_node
        end_gauge_id = None

        for _ in range(500_000):  # safety limit
            # Get outgoing edges
            out_edges = list(self.G.out_edges(current, keys=True, data=True))

            if not out_edges:
                logger.info("Reached end of network (no outgoing edges)")
                break

            # Filter out already-visited edges (avoid loops)
            unvisited = [
                (u, v, k, d) for u, v, k, d in out_edges
                if (u, v, k) not in visited_edges
            ]

            if not unvisited:
                logger.info("No unvisited outgoing edges — stopping")
                break

            # Pick the edge with the greatest length (main channel heuristic)
            u, v, k, data = max(unvisited, key=lambda x: x[3].get("length", 0))

            visited_edges.add((u, v, k))

            geom = self._edge_geom.get((u, v, k))
            segments.append({
                "segment_order": len(segments),
                "river_idx": data.get("river_idx"),
                "length": data.get("length", 0),
                "geometry": geom,
            })

            current = v

            # Check if we've reached another gauge
            if stop_at_gauge and current in gauge_node_set:
                # Find which gauge_id this node belongs to
                node_data = self.G.nodes[current]
                end_gauge_id = node_data.get("gauge_id")
                if end_gauge_id is None:
                    # Look up from our mapping
                    for gid, gnode in self._gauge_nodes.items():
                        if gnode == current:
                            end_gauge_id = gid
                            break
                logger.info(
                    "Reached gauge %s after %d segments",
                    end_gauge_id, len(segments)
                )
                break

        if not segments:
            logger.warning("No downstream path found from gauge %s", gauge_id)
            return gpd.GeoDataFrame(
                columns=["segment_order", "river_idx", "length", "geometry"],
                crs=self.rivers.crs,
            )

        gdf = gpd.GeoDataFrame(segments, geometry="geometry", crs=self.rivers.crs)

        total_length = gdf["length"].sum()
        gdf.attrs["start_gauge_id"] = gauge_id
        gdf.attrs["end_gauge_id"] = end_gauge_id
        gdf.attrs["total_length_m"] = total_length

        logger.info(
            "Downstream path: %d segments, %.1f km, from gauge %s to %s",
            len(gdf),
            total_length / 1000,
            gauge_id,
            end_gauge_id if end_gauge_id else "end of network",
        )

        return gdf

    def trace_downstream_to_all_gauges(
        self,
        gauge_id: int,
        max_gauges: int = 50,
    ) -> list[gpd.GeoDataFrame]:
        """Trace downstream through multiple gauges, returning one GDF per reach.

        Returns a list of GeoDataFrames, one for each gauge-to-gauge reach.
        """
        results = []
        current_gauge = gauge_id
        visited_gauges = {gauge_id}

        for _ in range(max_gauges):
            gdf = self.trace_downstream(current_gauge, stop_at_gauge=True)
            if len(gdf) == 0:
                break

            results.append(gdf)
            end_gauge = gdf.attrs.get("end_gauge_id")

            if end_gauge is None or end_gauge in visited_gauges:
                break

            visited_gauges.add(end_gauge)
            current_gauge = end_gauge

        return results

    # -------------------------------------------------------------------
    # Phase 4: Visualization
    # -------------------------------------------------------------------

    def plot_downstream(
        self,
        gauge_id: int,
        path_gdf: gpd.GeoDataFrame = None,
        ax=None,
        figsize=(12, 8),
        buffer_m: float = 2000,
    ):
        """Plot the downstream path from a gauge with context.

        Parameters
        ----------
        gauge_id : int
            Starting gauge.
        path_gdf : GeoDataFrame, optional
            Pre-computed path (from trace_downstream). Computed if not given.
        ax : matplotlib Axes, optional
        figsize : tuple
        buffer_m : float
            Buffer around the path extent for showing context rivers.
        """
        if path_gdf is None:
            path_gdf = self.trace_downstream(gauge_id)

        if len(path_gdf) == 0:
            logger.warning("Nothing to plot — empty path")
            return

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Determine plot extent from path + buffer
        path_bounds = path_gdf.total_bounds  # (minx, miny, maxx, maxy)
        extent = (
            path_bounds[0] - buffer_m,
            path_bounds[1] - buffer_m,
            path_bounds[2] + buffer_m,
            path_bounds[3] + buffer_m,
        )

        # Plot context rivers (light blue, thin)
        from shapely.geometry import box
        extent_box = box(*extent)
        context_rivers = self.rivers[self.rivers.intersects(extent_box)]
        if len(context_rivers) > 0:
            context_rivers.plot(ax=ax, color="#b0d4f1", linewidth=0.3, zorder=1)

        # Plot the traced path (bold blue)
        path_gdf.plot(ax=ax, color="#0066cc", linewidth=2.5, zorder=2)

        # Plot all gauge points in the area
        if self.snap_results is not None:
            gauges_in_view = self.snap_results[
                self.snap_results["snapped_geom"].apply(
                    lambda g: g is not None
                    and g.x >= extent[0] and g.x <= extent[2]
                    and g.y >= extent[1] and g.y <= extent[3]
                )
            ]
            if len(gauges_in_view) > 0:
                # Plot other gauges as grey dots
                other_gauges = gauges_in_view[
                    gauges_in_view[self.gauge_id_col] != gauge_id
                ]
                if len(other_gauges) > 0:
                    ax.scatter(
                        [g.x for g in other_gauges["snapped_geom"]],
                        [g.y for g in other_gauges["snapped_geom"]],
                        c="grey", s=40, zorder=3, edgecolors="white",
                        linewidths=0.5, label="Other gauges"
                    )
                    for _, g in other_gauges.iterrows():
                        ax.annotate(
                            str(g[self.gauge_id_col]),
                            (g["snapped_geom"].x, g["snapped_geom"].y),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(4, 4), textcoords="offset points",
                            color="grey",
                        )

                # Highlight start gauge (red)
                start_row = gauges_in_view[
                    gauges_in_view[self.gauge_id_col] == gauge_id
                ]
                if len(start_row) > 0:
                    pt = start_row.iloc[0]["snapped_geom"]
                    ax.scatter(
                        pt.x, pt.y, c="red", s=80, zorder=4,
                        edgecolors="darkred", linewidths=1,
                        label=f"Start: {gauge_id}", marker="^",
                    )

                # Highlight end gauge (green) if we stopped at one
                end_gauge_id = path_gdf.attrs.get("end_gauge_id")
                if end_gauge_id is not None:
                    end_row = gauges_in_view[
                        gauges_in_view[self.gauge_id_col] == end_gauge_id
                    ]
                    if len(end_row) > 0:
                        pt = end_row.iloc[0]["snapped_geom"]
                        ax.scatter(
                            pt.x, pt.y, c="green", s=80, zorder=4,
                            edgecolors="darkgreen", linewidths=1,
                            label=f"End: {end_gauge_id}", marker="v",
                        )

        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_aspect("equal")
        ax.legend(loc="upper right")

        total_km = path_gdf.attrs.get("total_length_m", 0) / 1000
        end_label = path_gdf.attrs.get("end_gauge_id", "end of network")
        ax.set_title(
            f"Downstream from gauge {gauge_id} → {end_label}\n"
            f"{len(path_gdf)} segments, {total_km:.1f} km",
            fontsize=11,
        )
        ax.set_xlabel("Easting (m, LV95)")
        ax.set_ylabel("Northing (m, LV95)")

        plt.tight_layout()
        return ax

    # -------------------------------------------------------------------
    # Convenience: summary of all gauge connections
    # -------------------------------------------------------------------

    def get_snap_summary(self) -> gpd.GeoDataFrame:
        """Return the snap results with useful diagnostics."""
        if self.snap_results is None:
            raise RuntimeError("Call snap_gauges() first")

        df = self.snap_results.copy()

        # Add river name if available in TLM attributes
        name_cols = [c for c in self.rivers.columns if "name" in c.lower()]
        if name_cols and "river_idx" in df.columns:
            col = name_cols[0]
            df["river_name"] = df["river_idx"].apply(
                lambda i: self.rivers.iloc[int(i)][col]
                if i is not None and not np.isnan(i) else None
            )

        return df


# ===========================================================================
# CLI entrypoint
# ===========================================================================

def main():
    """Example usage with file paths from CONFIG."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Trace downstream river path between CAMELS-CH gauges"
    )
    parser.add_argument(
        "--gauges", default=GAUGES_PATH,
        help="Path to CAMELS-CH gauge shapefile"
    )
    parser.add_argument(
        "--rivers", default=RIVERS_PATH,
        help="Path to swissTLM3D FlowingWater shapefile/geopackage"
    )
    parser.add_argument(
        "--rivers-layer", default=RIVERS_LAYER,
        help="Layer name if rivers file is a GeoPackage"
    )
    parser.add_argument(
        "--gauge-id", type=int, default=2044,
        help="Gauge ID to trace downstream from (default: 2044, Thur-Andelfingen)"
    )
    parser.add_argument(
        "--output", default="downstream_path.gpkg",
        help="Output file for the traced path"
    )
    parser.add_argument(
        "--snap-output", default="snap_results.gpkg",
        help="Output file for gauge snapping results"
    )
    parser.add_argument(
        "--plot", default="downstream_plot.png",
        help="Output plot file"
    )

    args = parser.parse_args()

    # Initialize and run
    rc = RiverConnectivity(
        gauges_path=args.gauges,
        rivers_path=args.rivers,
        rivers_layer=args.rivers_layer,
    )

    # Step 1: Snap gauges
    snap_df = rc.snap_gauges()
    snap_df.to_file(args.snap_output, driver="GPKG")
    logger.info("Snap results saved to %s", args.snap_output)

    # Step 2: Build graph
    rc.build_graph()

    # Step 3: Trace downstream
    path_gdf = rc.trace_downstream(args.gauge_id, stop_at_gauge=True)

    if len(path_gdf) > 0:
        path_gdf.to_file(args.output, driver="GPKG")
        logger.info("Downstream path saved to %s", args.output)

        # Step 4: Plot
        rc.plot_downstream(args.gauge_id, path_gdf)
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", args.plot)
    else:
        logger.warning("No downstream path found — nothing to export")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Gauges loaded:   {len(rc.gauges)}")
    print(f"Gauges matched:  {snap_df['matched'].sum()}")
    print(f"Gauges unmatched:{(~snap_df['matched']).sum()}")
    print(f"Graph nodes:     {rc.G.number_of_nodes()}")
    print(f"Graph edges:     {rc.G.number_of_edges()}")
    if len(path_gdf) > 0:
        print(f"Path segments:   {len(path_gdf)}")
        print(f"Path length:     {path_gdf.attrs.get('total_length_m', 0)/1000:.1f} km")
        print(f"Start gauge:     {path_gdf.attrs.get('start_gauge_id')}")
        print(f"End gauge:       {path_gdf.attrs.get('end_gauge_id', 'end of network')}")


if __name__ == "__main__":
    main()