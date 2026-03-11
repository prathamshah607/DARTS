"""
Module 1 — Street Graph Extraction (OSMnx)
Downloads road networks, attaches elevation data, filters by disaster mode.
"""

import hashlib
import os

import networkx as nx
import numpy as np
import osmnx as ox

from config import (
    ALLOWED_HIGHWAYS,
    GRAPH_CACHE_DIR,
    ROAD_CLASS_WEIGHTS,
)


def _bbox_hash(bbox: tuple[float, float, float, float]) -> str:
    """Deterministic hash for a bounding box to use as cache key."""
    raw = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def download_graph(
    bbox: tuple[float, float, float, float],
    network_type: str = "drive",
    use_cache: bool = True,
) -> nx.MultiDiGraph:
    """
    Download the road network for a bounding box (north, south, east, west).
    Returns an OSMnx MultiDiGraph with basic attributes.
    """
    cache_key = _bbox_hash(bbox)
    cache_path = os.path.join(GRAPH_CACHE_DIR, f"{cache_key}.graphml")

    if use_cache and os.path.exists(cache_path):
        return ox.load_graphml(cache_path)

    north, south, east, west = bbox
    G = ox.graph_from_bbox(
        bbox=(north, south, east, west),
        network_type=network_type,
        simplify=True,
        retain_all=False,
    )

    if use_cache:
        ox.save_graphml(G, cache_path)

    return G


def bbox_from_point(
    lat: float, lon: float, buffer_km: float = 5.0
) -> tuple[float, float, float, float]:
    """Compute a bounding box (north, south, east, west) from a centre point."""
    delta_lat = buffer_km / 111.0
    delta_lon = buffer_km / (111.0 * np.cos(np.radians(lat)))
    return (lat + delta_lat, lat - delta_lat, lon + delta_lon, lon - delta_lon)


def add_elevation_data(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Attach elevation (metres) to every node using free SRTM data
    via the `elevation` + `rasterio` packages (no API key needed).

    Falls back to Open-Elevation public API, then to zero-fill.
    """
    try:
        # Try SRTM via the `elevation` CLI / rasterio pipeline
        import elevation as elev_pkg
        import rasterio
        from rasterio.transform import rowcol

        nodes = list(G.nodes(data=True))
        lats = [d.get("y", 0) for _, d in nodes]
        lons = [d.get("x", 0) for _, d in nodes]

        min_lat, max_lat = min(lats) - 0.01, max(lats) + 0.01
        min_lon, max_lon = min(lons) - 0.01, max(lons) + 0.01
        bounds = (min_lon, min_lat, max_lon, max_lat)

        dem_path = os.path.join(
            os.path.dirname(__file__), ".cache", "srtm_dem.tif"
        )
        os.makedirs(os.path.dirname(dem_path), exist_ok=True)
        elev_pkg.clip(bounds=bounds, output=dem_path)

        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            for nid, data in nodes:
                row, col = rowcol(src.transform, data.get("x", 0), data.get("y", 0))
                row = min(max(row, 0), dem_data.shape[0] - 1)
                col = min(max(col, 0), dem_data.shape[1] - 1)
                G.nodes[nid]["elevation"] = float(dem_data[row, col])

    except Exception:
        # Fallback: Open-Elevation public API (no key needed)
        try:
            _add_elevation_open_api(G)
        except Exception:
            # Last resort: zero-fill
            for node in G.nodes:
                if "elevation" not in G.nodes[node]:
                    G.nodes[node]["elevation"] = 0.0

    G = ox.elevation.add_edge_grades(G)
    return G


def _add_elevation_open_api(G: nx.MultiDiGraph) -> None:
    """
    Use the free Open-Elevation API (no key) to batch-query node elevations.
    https://open-elevation.com
    """
    import requests

    nodes = list(G.nodes(data=True))
    # API accepts batches of up to 200
    BATCH = 200
    for i in range(0, len(nodes), BATCH):
        batch = nodes[i : i + BATCH]
        locations = [
            {"latitude": d.get("y", 0), "longitude": d.get("x", 0)}
            for _, d in batch
        ]
        resp = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": locations},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        for (nid, _), r in zip(batch, results):
            G.nodes[nid]["elevation"] = float(r.get("elevation", 0))


def _get_highway_type(edge_data: dict) -> str:
    """Extract the primary highway type from an edge's data dict."""
    hw = edge_data.get("highway", "unclassified")
    if isinstance(hw, list):
        hw = hw[0]
    return hw


def filter_graph_by_disaster_mode(
    G: nx.MultiDiGraph,
    mode: str,
    flood_threshold: float = 2.0,
) -> nx.MultiDiGraph:
    """
    Filter the graph based on the active disaster mode.

    - Earthquake: keep only allowed highway classes
    - Flood: drop edges whose average node elevation < flood_threshold
    - Cyclone / General: no structural filtering (handled by weight engine)
    """
    G = G.copy()

    if mode == "Earthquake":
        allowed = set(ALLOWED_HIGHWAYS.get("Earthquake", []))
        edges_to_remove = []
        for u, v, k, data in G.edges(keys=True, data=True):
            hw = _get_highway_type(data)
            if hw not in allowed:
                edges_to_remove.append((u, v, k))
        G.remove_edges_from(edges_to_remove)

    elif mode == "Flood":
        edges_to_remove = []
        for u, v, k, data in G.edges(keys=True, data=True):
            elev_u = G.nodes[u].get("elevation", 999)
            elev_v = G.nodes[v].get("elevation", 999)
            avg_elev = (elev_u + elev_v) / 2.0
            if avg_elev < flood_threshold:
                edges_to_remove.append((u, v, k))
        G.remove_edges_from(edges_to_remove)

    # Remove isolated nodes after filtering
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G


def enrich_edge_attributes(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Add computed attributes to every edge:
    - elevation_avg: mean elevation of endpoint nodes
    - road_class_weight (γ): weight coefficient from config
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        elev_u = G.nodes[u].get("elevation", 0)
        elev_v = G.nodes[v].get("elevation", 0)
        data["elevation_avg"] = (elev_u + elev_v) / 2.0

        hw = _get_highway_type(data)
        data["road_class_weight"] = ROAD_CLASS_WEIGHTS.get(hw, 1.5)

        # Ensure length exists
        if "length" not in data:
            data["length"] = 0.0

    return G


def get_graph_stats(G: nx.MultiDiGraph) -> dict:
    """Return summary statistics for the graph."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "strongly_connected": nx.is_strongly_connected(G) if G.number_of_nodes() > 0 else False,
    }
