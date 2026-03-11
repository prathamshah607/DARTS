"""
DAIRS — Disaster-Aware Intelligent Routing System
Central configuration file.
"""

import os

# ── API Keys (load from environment or .env) ──────────────────
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Default Map Parameters ────────────────────────────────────
DEFAULT_LOCATION = (13.3409, 74.7421)  # Udupi, Karnataka
DEFAULT_BBOX_BUFFER_KM = 5.0  # km radius around origin for graph download

# ── Disaster Mode Constants ───────────────────────────────────
DISASTER_MODES = ["Flood", "Earthquake", "Cyclone", "General"]

FLOOD_THRESHOLD_DEFAULT = 2.0  # metres REM
LANDSLIDE_SLOPE_THRESHOLD = 30.0  # degrees
CYCLONE_COAST_BUFFER_KM = 5.0  # km
CYCLONE_WIND_BRIDGE_THRESHOLD = 80.0  # km/h

# ── Road Class Weight Coefficients (γ) ────────────────────────
ROAD_CLASS_WEIGHTS = {
    "motorway": 0.8,
    "motorway_link": 0.85,
    "trunk": 0.9,
    "trunk_link": 0.95,
    "primary": 1.0,
    "primary_link": 1.05,
    "secondary": 1.2,
    "secondary_link": 1.25,
    "tertiary": 1.4,
    "tertiary_link": 1.45,
    "residential": 1.6,
    "unclassified": 1.8,
    "service": 2.0,
    "track": 3.0,
    "path": 4.0,
}

# Road classes allowed per disaster mode
ALLOWED_HIGHWAYS = {
    "Earthquake": ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"],
    "Flood": None,  # all allowed; filtering done by elevation
    "Cyclone": None,
    "General": None,
}

# ── TomTom Incident Type Mapping ──────────────────────────────
INCIDENT_TYPE_LABELS = {
    1: "Accident",
    2: "Fog",
    3: "DangerousConditions",
    4: "Rain",
    5: "Ice",
    6: "Jam",
    7: "LaneClosed",
    8: "RoadClosed",
    9: "RoadWorks",
    10: "Wind",
    11: "Flooding",
    14: "BrokenDownVehicle",
}

HARD_BLOCK_TYPES = {8, 11}  # RoadClosed, Flooding
PENALTY_TYPES = {9: 2.5, 4: 1.5, 6: 1.8}  # type → weight multiplier

# ── Edge Weight Parameters ────────────────────────────────────
FLOOD_PENALTY_K = 1.5  # exponential penalty coefficient for flood risk

# ── TSP / VRP Defaults ────────────────────────────────────────
TSP_TIME_LIMIT_SECONDS = 30
TSP_DEFAULT_HEURISTIC = "PATH_CHEAPEST_ARC"
TSP_DEFAULT_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"

# ── Satellite Imagery (OSM static tiles — no API key needed) ─
SAT_TILE_ZOOM = 16
SAT_WATER_COVERAGE_THRESHOLD = 0.20  # 20% water pixels → warning

# ── TomTom API Refresh ────────────────────────────────────────
TOMTOM_POLL_INTERVAL_SECONDS = 300  # 5 minutes

# ── Cache Directories ─────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
GRAPH_CACHE_DIR = os.path.join(CACHE_DIR, "graphs")
SAT_CACHE_DIR = os.path.join(CACHE_DIR, "satellite")

for _d in (CACHE_DIR, GRAPH_CACHE_DIR, SAT_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)
