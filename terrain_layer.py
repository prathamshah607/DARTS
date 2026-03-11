"""
Module 2 — Terrain & Flood Risk Layer
Computes Relative Elevation Model (REM), NDWI flags, slope risk,
and bridge/tunnel annotations for the graph.
"""

from typing import Optional

import networkx as nx
import numpy as np
from scipy.interpolate import griddata

from config import LANDSLIDE_SLOPE_THRESHOLD


def compute_rem_for_nodes(
    G: nx.MultiDiGraph,
    water_node_ids: Optional[set[int]] = None,
) -> nx.MultiDiGraph:
    """
    Compute a Relative Elevation Model (REM) value for each node.

    REM = node_elevation − interpolated_water_surface_elevation

    Parameters
    ----------
    G : graph with 'elevation' on every node
    water_node_ids : set of node IDs near water bodies (from OSM waterway tags).
                     If None, water nodes are estimated as the lowest 5% of nodes.
    """
    nodes = list(G.nodes(data=True))
    if not nodes:
        return G

    coords = np.array([[d.get("y", 0), d.get("x", 0)] for _, d in nodes])
    elevations = np.array([d.get("elevation", 0) for _, d in nodes])

    # Identify water reference nodes
    if water_node_ids:
        water_mask = np.array([nid in water_node_ids for nid, _ in nodes])
    else:
        # Heuristic: lowest 5% of nodes are near water
        threshold = np.percentile(elevations, 5)
        water_mask = elevations <= threshold

    if water_mask.sum() < 3:
        # Not enough water reference points — fall back to min elevation
        water_surface = np.full(len(elevations), elevations.min())
    else:
        water_coords = coords[water_mask]
        water_elevs = elevations[water_mask]
        # IDW interpolation of water surface across all nodes
        water_surface = griddata(
            water_coords, water_elevs, coords, method="linear", fill_value=elevations.min()
        )

    rem_values = elevations - water_surface

    for i, (nid, _) in enumerate(nodes):
        G.nodes[nid]["rem_value"] = float(rem_values[i])

    return G


def flag_flood_risk_nodes(
    G: nx.MultiDiGraph, flood_depth_threshold: float = 2.0
) -> nx.MultiDiGraph:
    """
    Mark nodes with rem_value < flood_depth_threshold as flooded.
    Edges connecting two flooded nodes get flood_risk_score = infinity.
    """
    for nid, data in G.nodes(data=True):
        rem = data.get("rem_value", 999)
        data["is_flooded"] = rem < flood_depth_threshold
        data["flood_risk_score"] = max(0.0, flood_depth_threshold - rem) if rem < flood_depth_threshold else 0.0

    for u, v, k, data in G.edges(keys=True, data=True):
        u_flooded = G.nodes[u].get("is_flooded", False)
        v_flooded = G.nodes[v].get("is_flooded", False)
        if u_flooded and v_flooded:
            data["flood_risk_score"] = float("inf")
        elif u_flooded or v_flooded:
            rem_u = G.nodes[u].get("rem_value", 999)
            rem_v = G.nodes[v].get("rem_value", 999)
            data["flood_risk_score"] = max(0, flood_depth_threshold - min(rem_u, rem_v))
        else:
            data["flood_risk_score"] = 0.0

    return G


def flag_slope_risk(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Flag edges with grade (slope) exceeding the landslide threshold.
    Used primarily in Earthquake mode.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        grade = abs(data.get("grade", 0)) * 100  # grade is fractional, convert to %
        grade_deg = np.degrees(np.arctan(grade / 100)) if grade else 0
        data["slope_degrees"] = grade_deg
        data["landslide_risk"] = grade_deg > LANDSLIDE_SLOPE_THRESHOLD

    return G


def flag_bridges_and_tunnels(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Annotate edges that are bridges or tunnels (from OSM tags).
    - Bridges: elevated risk during earthquakes
    - Tunnels: flagged during floods
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        data["is_bridge"] = data.get("bridge", "") in ("yes", True, "viaduct")
        data["is_tunnel"] = data.get("tunnel", "") in ("yes", True, "building_passage")

    return G


def get_water_nodes_from_osm(
    bbox: tuple[float, float, float, float],
) -> set[tuple[float, float]]:
    """
    Download water body geometries from OSM and return centroids
    as (lat, lon) tuples for REM computation.
    """
    import osmnx as ox

    north, south, east, west = bbox
    try:
        water_gdf = ox.features_from_bbox(
            bbox=(north, south, east, west),
            tags={"natural": ["water"], "waterway": True},
        )
        points = set()
        for _, row in water_gdf.iterrows():
            centroid = row.geometry.centroid
            points.add((centroid.y, centroid.x))
        return points
    except Exception:
        return set()


def apply_terrain_layer(
    G: nx.MultiDiGraph,
    flood_threshold: float = 2.0,
    water_node_ids: Optional[set[int]] = None,
) -> nx.MultiDiGraph:
    """
    Full terrain analysis pipeline:
    1. Compute REM
    2. Flag flood risk nodes/edges
    3. Flag slope risk
    4. Flag bridges/tunnels
    """
    G = compute_rem_for_nodes(G, water_node_ids)
    G = flag_flood_risk_nodes(G, flood_threshold)
    G = flag_slope_risk(G)
    G = flag_bridges_and_tunnels(G)
    return G
