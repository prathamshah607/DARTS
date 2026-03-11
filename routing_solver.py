"""
Module 5 — Routing Solvers
Single-pair Dijkstra/A* routing and multi-stop TSP/VRP via Google OR-Tools.
"""

import math
from dataclasses import dataclass, field

import networkx as nx
import osmnx as ox
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ── Data Structures ───────────────────────────────────────────

@dataclass
class RouteResult:
    """Result of a single-pair routing computation."""
    route_id: str
    label: str  # FASTEST / SAFEST / HIGHWAY-PREFERRED
    node_path: list[int] = field(default_factory=list)
    coords: list[tuple[float, float]] = field(default_factory=list)  # [(lat, lon), ...]
    total_distance_m: float = 0.0
    total_weight: float = 0.0
    risk_score: float = 0.0  # 0-100
    blocked_roads: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TSPResult:
    """Result of a multi-stop TSP/VRP solve."""
    vehicle_routes: list[list[int]] = field(default_factory=list)  # node indices per vehicle
    vehicle_coords: list[list[tuple[float, float]]] = field(default_factory=list)
    total_distance_m: float = 0.0
    total_weight: float = 0.0
    stop_order: list[int] = field(default_factory=list)  # ordered stop indices
    unreachable_stops: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Single-Pair Routing ──────────────────────────────────────

def find_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Snap a lat/lon to the nearest graph node."""
    return ox.distance.nearest_nodes(G, lon, lat)


def single_route(
    G: nx.MultiDiGraph,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    weight: str = "weight",
    route_id: str = "R001",
    label: str = "SAFEST",
) -> RouteResult:
    """
    Compute the shortest path between origin and destination
    using Dijkstra with the specified weight attribute.
    """
    orig_node = find_nearest_node(G, origin_lat, origin_lon)
    dest_node = find_nearest_node(G, dest_lat, dest_lon)

    result = RouteResult(route_id=route_id, label=label)

    try:
        path = nx.shortest_path(G, orig_node, dest_node, weight=weight)
    except nx.NetworkXNoPath:
        result.warnings.append(
            f"NO PATH from ({origin_lat},{origin_lon}) to ({dest_lat},{dest_lon})"
        )
        result.risk_score = 100
        return result

    result.node_path = path
    result.coords = [(G.nodes[n].get("y", 0), G.nodes[n].get("x", 0)) for n in path]

    # Compute totals along the path
    total_dist = 0.0
    total_wt = 0.0
    max_flood_risk = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        # Get the edge with minimum weight among parallel edges
        edge_data = _best_edge(G, u, v, weight)
        dist = edge_data.get("length", 0)
        wt = edge_data.get(weight, dist)
        total_dist += dist
        if not math.isinf(wt):
            total_wt += wt
        else:
            total_wt = float("inf")

        flood = edge_data.get("flood_risk_score", 0)
        if not math.isinf(flood):
            max_flood_risk = max(max_flood_risk, flood)

        # Collect warnings
        if edge_data.get("landslide_risk", False):
            result.warnings.append(
                f"Landslide risk at segment {i}: slope {edge_data.get('slope_degrees', 0):.1f}°"
            )
        reason = edge_data.get("incident_reason", "")
        if reason:
            result.warnings.append(f"Incident at segment {i}: {reason}")

    result.total_distance_m = total_dist
    result.total_weight = total_wt
    # Risk score: 0-100 based on total weight relative to raw distance
    if total_dist > 0 and not math.isinf(total_wt):
        ratio = total_wt / total_dist
        result.risk_score = min(100, max(0, (ratio - 1.0) * 50))
    elif math.isinf(total_wt):
        result.risk_score = 100

    return result


def compute_top3_routes(
    G: nx.MultiDiGraph,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> list[RouteResult]:
    """
    Compute up to 3 Pareto-optimal routes:
    1. SAFEST — using composite 'weight'
    2. FASTEST — using raw 'length'
    3. HIGHWAY-PREFERRED — using 'road_class_weight' bias
    """
    routes = []

    # 1. SAFEST (composite weight)
    r1 = single_route(G, origin_lat, origin_lon, dest_lat, dest_lon,
                       weight="weight", route_id="R001", label="SAFEST")
    routes.append(r1)

    # 2. FASTEST (raw distance)
    r2 = single_route(G, origin_lat, origin_lon, dest_lat, dest_lon,
                       weight="length", route_id="R002", label="FASTEST")
    if r2.node_path != r1.node_path:
        routes.append(r2)

    # 3. HIGHWAY-PREFERRED (road class weight only)
    r3 = single_route(G, origin_lat, origin_lon, dest_lat, dest_lon,
                       weight="road_class_weight", route_id="R003", label="HIGHWAY-PREFERRED")
    if r3.node_path not in (r1.node_path, r2.node_path):
        routes.append(r3)

    return routes


def _best_edge(G: nx.MultiDiGraph, u: int, v: int, weight: str = "weight") -> dict:
    """Among parallel edges u→v, return the data dict with lowest weight."""
    edges = G.get_edge_data(u, v)
    if not edges:
        return {}
    best = min(edges.values(), key=lambda d: d.get(weight, float("inf")))
    return best


# ── Multi-Stop TSP / VRP (OR-Tools) ──────────────────────────

def build_distance_matrix(
    G: nx.MultiDiGraph,
    node_ids: list[int],
    weight: str = "weight",
) -> list[list[float]]:
    """
    Build an NxN distance matrix between node_ids using shortest path lengths.
    Unreachable pairs get a very large value.
    """
    n = len(node_ids)
    LARGE = 10**9
    matrix = [[LARGE] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
                continue
            try:
                dist = nx.shortest_path_length(G, node_ids[i], node_ids[j], weight=weight)
                matrix[i][j] = int(dist) if not math.isinf(dist) else LARGE
            except nx.NetworkXNoPath:
                matrix[i][j] = LARGE

    return matrix


def solve_tsp(
    G: nx.MultiDiGraph,
    depot_lat: float,
    depot_lon: float,
    stops: list[dict],
    vehicle_count: int = 1,
    time_limit_seconds: int = 30,
) -> TSPResult:
    """
    Solve TSP (single vehicle) or VRP (multiple vehicles) using OR-Tools.

    Parameters
    ----------
    G : weighted graph
    depot_lat, depot_lon : origin (reserve) location
    stops : list of dicts with keys 'lat', 'lon', 'label', 'priority', 'time_window_minutes'
    vehicle_count : number of vehicles (1 = TSP, >1 = VRP)
    time_limit_seconds : solver time limit

    Returns
    -------
    TSPResult with ordered routes per vehicle
    """
    result = TSPResult()

    # Snap depot and stops to graph nodes
    depot_node = find_nearest_node(G, depot_lat, depot_lon)
    stop_nodes = [find_nearest_node(G, s["lat"], s["lon"]) for s in stops]

    all_nodes = [depot_node] + stop_nodes  # index 0 = depot
    n = len(all_nodes)

    if n < 2:
        result.warnings.append("Need at least one stop besides the depot.")
        return result

    # Build distance matrix
    dist_matrix = build_distance_matrix(G, all_nodes, weight="weight")

    # Check for unreachable stops
    LARGE = 10**9
    for i in range(1, n):
        if dist_matrix[0][i] >= LARGE or dist_matrix[i][0] >= LARGE:
            result.unreachable_stops.append(i - 1)  # stop index
            result.warnings.append(
                f"Stop '{stops[i-1].get('label', i-1)}' is UNREACHABLE from depot."
            )

    # OR-Tools routing model
    manager = pywrapcp.RoutingIndexManager(n, vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add distance dimension for tracking
    routing.AddDimension(
        transit_callback_index,
        0,       # slack
        LARGE,   # max distance per vehicle
        True,    # start cumul to zero
        "Distance",
    )

    # Set priority-based penalties for dropping nodes (if unreachable)
    for i in range(1, n):
        penalty = LARGE
        routing.AddDisjunction([manager.NodeToIndex(i)], penalty)

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit_seconds

    # Solve
    solution = routing.SolveWithParameters(search_params)

    if not solution:
        result.warnings.append("OR-Tools could not find a solution. All stops may be unreachable.")
        return result

    # Extract solution routes
    total_distance = 0
    for vehicle_id in range(vehicle_count):
        route_nodes = []
        route_coords = []
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            nid = all_nodes[node_index]
            route_nodes.append(nid)
            route_coords.append((G.nodes[nid].get("y", 0), G.nodes[nid].get("x", 0)))
            index = solution.Value(routing.NextVar(index))
        # Add final node (return to depot)
        node_index = manager.IndexToNode(index)
        nid = all_nodes[node_index]
        route_nodes.append(nid)
        route_coords.append((G.nodes[nid].get("y", 0), G.nodes[nid].get("x", 0)))

        result.vehicle_routes.append(route_nodes)
        result.vehicle_coords.append(route_coords)

    result.total_weight = solution.ObjectiveValue()
    # Compute actual distance along solution
    for vehicle_route in result.vehicle_routes:
        for i in range(len(vehicle_route) - 1):
            edge_data = _best_edge(G, vehicle_route[i], vehicle_route[i + 1])
            total_distance += edge_data.get("length", 0)
    result.total_distance_m = total_distance

    # Stop order (flatten across vehicles, exclude depot)
    for vr in result.vehicle_routes:
        for nid in vr:
            if nid in stop_nodes:
                result.stop_order.append(stop_nodes.index(nid))

    return result
