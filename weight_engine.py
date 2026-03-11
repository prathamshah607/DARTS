"""
Module 4 — Dynamic Edge Weight Engine
Computes the final composite weight for every edge in the graph:

    w_edge = d_road × α_terrain × β_incident × γ_road_class

Where:
    d_road      = raw road length (metres)
    α_terrain   = exp(k · max(0, T_flood − elev))
    β_incident  ∈ {1.0, 2.5, ∞}
    γ_road_class = highway preference coefficient
"""

import math

import networkx as nx

from config import FLOOD_PENALTY_K


def compute_terrain_alpha(
    elevation_avg: float,
    flood_threshold: float,
    k: float = FLOOD_PENALTY_K,
    mode: str = "Flood",
) -> float:
    """
    Exponential flood-terrain penalty.

    α = exp(k · max(0, T_flood − elevation))

    For non-flood modes, returns 1.0 (no terrain penalty from elevation).
    """
    if mode != "Flood":
        return 1.0
    deficit = max(0.0, flood_threshold - elevation_avg)
    if deficit == 0:
        return 1.0
    return math.exp(k * deficit)


def compute_edge_weight(
    length: float,
    alpha_terrain: float,
    beta_incident: float,
    gamma_road_class: float,
) -> float:
    """
    Composite edge weight.
    Returns infinity if any factor is infinity.
    """
    if math.isinf(alpha_terrain) or math.isinf(beta_incident):
        return float("inf")
    return length * alpha_terrain * beta_incident * gamma_road_class


def assign_weights(
    G: nx.MultiDiGraph,
    mode: str = "General",
    flood_threshold: float = 2.0,
) -> nx.MultiDiGraph:
    """
    Compute and assign the composite 'weight' attribute for every edge
    in the graph based on terrain, incident, and road-class factors.

    Parameters
    ----------
    G : MultiDiGraph with enriched attributes from Modules 1-3
    mode : disaster mode string
    flood_threshold : flood depth threshold in metres

    Returns
    -------
    G with 'weight' set on every edge
    """
    blocked_count = 0

    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length", 0.0)
        elevation_avg = data.get("elevation_avg", 999)
        gamma = data.get("road_class_weight", 1.0)
        beta = data.get("incident_penalty", 1.0)

        # Module 2 flood risk — hard block if infinite
        flood_risk = data.get("flood_risk_score", 0.0)
        if math.isinf(flood_risk):
            data["weight"] = float("inf")
            data["weight_breakdown"] = "BLOCKED: flood risk infinite"
            blocked_count += 1
            continue

        # Earthquake-specific: block tunnels, flag high slopes
        if mode == "Earthquake":
            if data.get("is_tunnel", False):
                data["weight"] = float("inf")
                data["weight_breakdown"] = "BLOCKED: tunnel in earthquake mode"
                blocked_count += 1
                continue
            if data.get("landslide_risk", False):
                gamma *= 3.0  # heavy penalty for landslide-risk slopes

        # Cyclone-specific: block bridges above wind threshold
        if mode == "Cyclone":
            if data.get("is_bridge", False):
                # Bridge penalty handled externally via incident or config;
                # here we add a default 2x penalty
                gamma *= 2.0

        alpha = compute_terrain_alpha(elevation_avg, flood_threshold, mode=mode)

        w = compute_edge_weight(length, alpha, beta, gamma)
        data["weight"] = w

        # Store breakdown for AI explanation
        data["weight_breakdown"] = (
            f"len={length:.0f}m × α={alpha:.2f} × β={beta:.1f} × γ={gamma:.2f} = {w:.0f}"
        )

        if math.isinf(w):
            blocked_count += 1

    return G


def get_blocked_edges(G: nx.MultiDiGraph) -> list[dict]:
    """Return summary of all blocked (weight=inf) edges."""
    blocked = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if math.isinf(data.get("weight", 0)):
            blocked.append({
                "u": u,
                "v": v,
                "reason": data.get("weight_breakdown", "unknown"),
                "incident": data.get("incident_reason", ""),
            })
    return blocked


def get_weight_statistics(G: nx.MultiDiGraph) -> dict:
    """Summary statistics of edge weights for dashboard display."""
    weights = [
        d.get("weight", 0)
        for _, _, _, d in G.edges(keys=True, data=True)
        if not math.isinf(d.get("weight", 0))
    ]
    if not weights:
        return {"min": 0, "max": 0, "mean": 0, "blocked_edges": G.number_of_edges()}

    return {
        "min": min(weights),
        "max": max(weights),
        "mean": sum(weights) / len(weights),
        "blocked_edges": G.number_of_edges() - len(weights),
        "total_edges": G.number_of_edges(),
    }
