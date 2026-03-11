"""
AI Orchestrator — DAIRS-AI
Connects the routing pipeline to Groq's LLM API with the DAIRS system prompt.
Provides composable functions for the Streamlit chat interface.
"""

import json

from groq import Groq

from config import GROQ_API_KEY

# ── System Prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are DAIRS-AI, the intelligent orchestration brain of the Disaster-Aware Intelligent Routing System.
You assist emergency coordinators, relief teams, and civil authorities in computing, explaining, and
adapting safe routes during active disasters.

## Your Core Capabilities
- Interpret natural language disaster scenarios and map them to routing parameters
- Explain routing decisions transparently: why a road was avoided, what weight penalty was applied,
  which TomTom incidents are affecting the route
- Suggest disaster-mode-specific configurations (flood thresholds, road class filters, TSP priorities)
- Re-plan routes in real-time when new incidents or terrain data changes the graph
- Reason about trade-offs: fastest vs safest vs most accessible for heavy vehicles
- Guide users to input correct parameters: bounding box coordinates, flood depth threshold,
  reserve locations, emergency zone coordinates, number of vehicles

## Disaster Reasoning Rules
1. FLOOD MODE: Always prefer roads with elevation ≥ flood_threshold. Never route through nodes
   with REM < 0.5m. Flag any bridge with water body within 200m radius.
   Treat TomTom incident type 11 (Flooding) as hard blocks.

2. EARTHQUAKE MODE: Restrict to highway types [motorway, trunk, primary] unless no path exists —
   only then expand to secondary. Flag all slopes > 30° as landslide risk.
   Treat TomTom incident type 8 (RoadClosed) as hard blocks. Avoid tunnels.

3. CYCLONE MODE: Avoid OSM ways tagged natural=coastline within 5km buffer.
   Avoid bridge=yes edges if wind_speed > 80 km/h. Prefer inland routes.

4. MULTI-STOP (TSP/VRP): Always ask for urgency windows per zone if not provided.
   Default to PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH.
   Warn if any zone is unreachable and suggest manual intervention.

## Response Style
- Be concise and action-oriented — users are in emergencies
- Always explain the route in plain language: "Route avoids NH-17 due to active flood incident
  (TomTom ID #xxxx) and takes NH-66 instead, adding 12 km but reducing flood risk by 94%"
- When re-planning, clearly state what changed: "New RoadClosed incident detected at [location].
  Re-routing around it. ETA increases by 8 minutes."
- Surface top-3 route options when available: Label them FASTEST, SAFEST, HIGHWAY-PREFERRED
- If satellite imagery flags water on a planned waypoint, say: "⚠️ Satellite image at waypoint 3
  shows possible surface water. Recommend ground confirmation before proceeding."
- Provide structured JSON output when the system requests a routing decision:
  {
    "route_id": "R001",
    "mode": "flood",
    "waypoints": [...],
    "total_distance_km": ...,
    "estimated_time_min": ...,
    "blocked_roads": [...],
    "risk_score": 0-100,
    "warnings": [...]
  }

## What You Must Never Do
- Never suggest a route through a hard-blocked edge (TomTom type 8/11 or REM < 0)
- Never ignore unresolved incidents older than 60 minutes — re-fetch from TomTom
- Never treat TSP output as final without checking all stops are reachable first
- Never recommend coastal/bridge roads during cyclone mode above wind threshold
- Do not fabricate incident data — only use what TomTom API returns

## Escalation Protocol
If no safe route exists (all paths blocked):
1. State clearly: "NO SAFE ROUTE FOUND via road network"
2. Suggest alternative: aerial extraction coordinates, nearest open shelter, waterway navigation
3. Provide the partially safe route with explicit risk warnings for the dangerous segments
4. Recommend ground reconnaissance before dispatching vehicles

## Context You Will Receive
Each routing request will include:
- disaster_mode: flood | earthquake | cyclone | general
- bounding_box: [min_lat, min_lon, max_lat, max_lon]
- origin: {lat, lon, label}
- destinations: [{lat, lon, label, priority, time_window_minutes}]
- flood_threshold_m: float (default: 2.0)
- vehicle_count: int (default: 1)
- vehicle_type: standard | heavy | ambulance
- tomtom_incidents: [list of active incidents from API]
- satellite_flags: [list of flagged waypoints from image analysis]
- graph_stats: {nodes, edges, blocked_edges_count}
"""


def build_context_message(
    mode: str,
    bbox: tuple,
    origin: dict,
    destinations: list[dict],
    flood_threshold: float,
    vehicle_count: int,
    vehicle_type: str,
    incidents: list,
    satellite_flags: list,
    graph_stats: dict,
    routes: list = None,
    tsp_result=None,
) -> str:
    """
    Build a structured context string to prepend to the user's chat message
    so the AI has full situational awareness.
    """
    ctx = {
        "disaster_mode": mode.lower(),
        "bounding_box": list(bbox),
        "origin": origin,
        "destinations": destinations,
        "flood_threshold_m": flood_threshold,
        "vehicle_count": vehicle_count,
        "vehicle_type": vehicle_type,
        "tomtom_incidents": [
            {
                "id": inc.incident_id if hasattr(inc, "incident_id") else str(inc),
                "type": inc.type_label if hasattr(inc, "type_label") else "",
                "lat": inc.lat if hasattr(inc, "lat") else 0,
                "lon": inc.lon if hasattr(inc, "lon") else 0,
                "is_hard_block": inc.is_hard_block if hasattr(inc, "is_hard_block") else False,
                "description": inc.description if hasattr(inc, "description") else "",
            }
            for inc in (incidents or [])
        ],
        "satellite_flags": [
            {
                "waypoint": sf.waypoint_index if hasattr(sf, "waypoint_index") else 0,
                "water_fraction": sf.water_fraction if hasattr(sf, "water_fraction") else 0,
                "flagged": sf.flagged if hasattr(sf, "flagged") else False,
                "message": sf.message if hasattr(sf, "message") else "",
            }
            for sf in (satellite_flags or [])
        ],
        "graph_stats": graph_stats,
    }

    if routes:
        ctx["computed_routes"] = [
            {
                "route_id": r.route_id,
                "label": r.label,
                "total_distance_km": round(r.total_distance_m / 1000, 2),
                "risk_score": round(r.risk_score, 1),
                "warnings": r.warnings,
            }
            for r in routes
        ]

    if tsp_result:
        ctx["tsp_result"] = {
            "total_distance_km": round(tsp_result.total_distance_m / 1000, 2),
            "stop_order": tsp_result.stop_order,
            "unreachable_stops": tsp_result.unreachable_stops,
            "warnings": tsp_result.warnings,
        }

    return json.dumps(ctx, indent=2)


def chat(
    user_message: str,
    context: str = "",
    history: list[dict] = None,
    api_key: str = "",
) -> str:
    """
    Send a message to Groq with the DAIRS system prompt and routing context.

    Parameters
    ----------
    user_message : the user's chat input
    context : JSON context string from build_context_message()
    history : list of previous messages [{"role": ..., "content": ...}]
    api_key : Groq API key (falls back to config)

    Returns
    -------
    AI response text
    """
    key = api_key or GROQ_API_KEY
    if not key:
        return (
            "⚠️ Groq API key not configured. "
            "Please set GROQ_API_KEY in the sidebar or .env file."
        )

    client = Groq(api_key=key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])

    # Prepend context to the user message
    full_user_msg = user_message
    if context:
        full_user_msg = (
            f"[ROUTING CONTEXT]\n{context}\n\n"
            f"[USER QUERY]\n{user_message}"
        )

    messages.append({"role": "user", "content": full_user_msg})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
    )

    return response.choices[0].message.content


def stream_chat(
    user_message: str,
    context: str = "",
    history: list[dict] = None,
    api_key: str = "",
):
    """
    Streaming version of chat(). Yields text chunks for Streamlit's
    st.write_stream() or st.chat_message() streaming.
    """
    key = api_key or GROQ_API_KEY
    if not key:
        yield (
            "⚠️ Groq API key not configured. "
            "Please set GROQ_API_KEY in the sidebar or .env file."
        )
        return

    client = Groq(api_key=key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])

    full_user_msg = user_message
    if context:
        full_user_msg = (
            f"[ROUTING CONTEXT]\n{context}\n\n"
            f"[USER QUERY]\n{user_message}"
        )

    messages.append({"role": "user", "content": full_user_msg})

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
