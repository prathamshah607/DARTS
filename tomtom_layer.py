"""
Module 3 — TomTom Real-Time Incident Layer
Fetches live traffic incidents, snaps them to graph nodes, and applies
hard blocks or weight penalties.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import osmnx as ox
import requests

from config import (
    HARD_BLOCK_TYPES,
    INCIDENT_TYPE_LABELS,
    PENALTY_TYPES,
    TOMTOM_POLL_INTERVAL_SECONDS,
)


@dataclass
class Incident:
    """Parsed TomTom traffic incident."""
    incident_id: str
    type_code: int
    type_label: str
    lat: float
    lon: float
    description: str = ""
    severity: int = 0
    delay_seconds: int = 0
    from_street: str = ""
    to_street: str = ""
    is_hard_block: bool = False
    penalty_multiplier: float = 1.0
    snapped_node: Optional[int] = None
    raw: dict = field(default_factory=dict)


def fetch_incidents(
    api_key: str,
    bbox: tuple[float, float, float, float],
) -> list[Incident]:
    """
    Call TomTom Traffic Incidents API for a bounding box.

    Parameters
    ----------
    api_key : TomTom developer API key
    bbox : (north, south, east, west) matching OSMnx convention

    Returns
    -------
    List of parsed Incident objects
    """
    if not api_key:
        return []

    north, south, east, west = bbox
    # TomTom bbox format: minLat,minLon,maxLat,maxLon
    bbox_str = f"{south},{west},{north},{east}"

    url = (
        f"https://api.tomtom.com/traffic/services/5/incidentDetails"
        f"?key={api_key}"
        f"&bbox={bbox_str}"
        f"&fields={{incidents{{type,geometry{{type,coordinates}},properties{{id,iconCategory,"
        f"magnitudeOfDelay,startTime,endTime,from,to,length,delay,roadNumbers,events{{description,"
        f"code,iconCategory}}}}}}}}"
        f"&language=en-US"
        f"&timeValidityFilter=present"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    incidents = []
    for item in data.get("incidents", []):
        geom = item.get("geometry", {})
        props = item.get("properties", {})

        # Extract coordinates (first point of the incident geometry)
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        # Geometry can be LineString [[lon,lat],...] or Point [lon,lat]
        if isinstance(coords[0], list):
            lon, lat = coords[0][0], coords[0][1]
        else:
            lon, lat = coords[0], coords[1]

        type_code = props.get("iconCategory", 0)
        type_label = INCIDENT_TYPE_LABELS.get(type_code, f"Unknown({type_code})")

        inc = Incident(
            incident_id=str(props.get("id", "")),
            type_code=type_code,
            type_label=type_label,
            lat=lat,
            lon=lon,
            description=_extract_description(props),
            severity=props.get("magnitudeOfDelay", 0),
            delay_seconds=props.get("delay", 0),
            from_street=props.get("from", ""),
            to_street=props.get("to", ""),
            is_hard_block=type_code in HARD_BLOCK_TYPES,
            penalty_multiplier=PENALTY_TYPES.get(type_code, 1.0),
            raw=item,
        )
        incidents.append(inc)

    return incidents


def _extract_description(props: dict) -> str:
    """Pull human-readable description from incident properties."""
    events = props.get("events", [])
    if events and isinstance(events, list):
        return events[0].get("description", "")
    return ""


def snap_incidents_to_graph(
    G: nx.MultiDiGraph,
    incidents: list[Incident],
) -> list[Incident]:
    """
    Snap each incident's lat/lon to the nearest node in the graph.
    """
    if not incidents:
        return incidents

    lats = [inc.lat for inc in incidents]
    lons = [inc.lon for inc in incidents]
    nearest = ox.distance.nearest_nodes(G, lons, lats)

    for inc, node_id in zip(incidents, nearest):
        inc.snapped_node = node_id

    return incidents


def apply_incidents_to_graph(
    G: nx.MultiDiGraph,
    incidents: list[Incident],
) -> nx.MultiDiGraph:
    """
    Modify edge weights based on snapped incidents.

    - Hard blocks (type 8, 11): set all edges of the snapped node weight = inf
    - Penalty types (type 9, 4, 6): multiply edge weights by the penalty multiplier
    """
    G = G.copy()

    for inc in incidents:
        if inc.snapped_node is None:
            continue
        node = inc.snapped_node

        # Get all edges connected to this node
        out_edges = list(G.out_edges(node, keys=True, data=True))
        in_edges = list(G.in_edges(node, keys=True, data=True))

        if inc.is_hard_block:
            for u, v, k, data in out_edges:
                data["incident_penalty"] = float("inf")
                data["incident_reason"] = f"{inc.type_label}: {inc.description}"
            for u, v, k, data in in_edges:
                data["incident_penalty"] = float("inf")
                data["incident_reason"] = f"{inc.type_label}: {inc.description}"
        elif inc.penalty_multiplier > 1.0:
            for u, v, k, data in out_edges:
                existing = data.get("incident_penalty", 1.0)
                if existing != float("inf"):
                    data["incident_penalty"] = max(existing, inc.penalty_multiplier)
                    data["incident_reason"] = f"{inc.type_label}: {inc.description}"
            for u, v, k, data in in_edges:
                existing = data.get("incident_penalty", 1.0)
                if existing != float("inf"):
                    data["incident_penalty"] = max(existing, inc.penalty_multiplier)
                    data["incident_reason"] = f"{inc.type_label}: {inc.description}"

    return G


def incidents_to_geojson(incidents: list[Incident]) -> dict:
    """Convert incidents to GeoJSON FeatureCollection for map overlay."""
    features = []
    for inc in incidents:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [inc.lon, inc.lat],
            },
            "properties": {
                "id": inc.incident_id,
                "type": inc.type_label,
                "type_code": inc.type_code,
                "description": inc.description,
                "severity": inc.severity,
                "is_hard_block": inc.is_hard_block,
                "from": inc.from_street,
                "to": inc.to_street,
            },
        })
    return {"type": "FeatureCollection", "features": features}


class IncidentPoller:
    """
    Background thread that periodically re-fetches TomTom incidents.
    """

    def __init__(self, api_key: str, bbox: tuple, interval: int = TOMTOM_POLL_INTERVAL_SECONDS):
        self.api_key = api_key
        self.bbox = bbox
        self.interval = interval
        self.incidents: list[Incident] = []
        self.last_fetch_time: float = 0
        self._timer: Optional[threading.Timer] = None
        self._running = False

    def start(self):
        self._running = True
        self._poll()

    def stop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()

    def _poll(self):
        if not self._running:
            return
        self.incidents = fetch_incidents(self.api_key, self.bbox)
        self.last_fetch_time = time.time()
        self._timer = threading.Timer(self.interval, self._poll)
        self._timer.daemon = True
        self._timer.start()
