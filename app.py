"""
DAIRS — Disaster-Aware Intelligent Routing System
Streamlit Frontend Application
"""

import math
import os

import folium
import networkx as nx
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

from config import (
    DEFAULT_LOCATION,
    DISASTER_MODES,
    FLOOD_THRESHOLD_DEFAULT,
    TOMTOM_POLL_INTERVAL_SECONDS,
)
from graph_extraction import (
    add_elevation_data,
    bbox_from_point,
    download_graph,
    enrich_edge_attributes,
    filter_graph_by_disaster_mode,
    get_graph_stats,
)
from orchestrator import build_context_message, stream_chat
from routing_solver import (
    RouteResult,
    compute_top3_routes,
    solve_tsp,
)
from satellite_analysis import SatelliteFlag, analyse_waypoints
from terrain_layer import apply_terrain_layer
from tomtom_layer import (
    apply_incidents_to_graph,
    fetch_incidents,
    incidents_to_geojson,
    snap_incidents_to_graph,
)
from weight_engine import assign_weights, get_blocked_edges, get_weight_statistics

load_dotenv()

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DAIRS — Disaster-Aware Intelligent Routing",
    layout="wide",
    page_icon="🚨",
)

# ── Session State Initialization ──────────────────────────────
DEFAULTS = {
    "graph": None,
    "routes": [],
    "tsp_result": None,
    "incidents": [],
    "satellite_flags": [],
    "chat_history": [],
    "graph_stats": {},
    "bbox": None,
    "weight_stats": {},
    "blocked_edges": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar Controls ─────────────────────────────────────────
st.sidebar.title("🚨 DAIRS Control Panel")
st.sidebar.markdown("**Disaster-Aware Intelligent Routing System**")

disaster_mode = st.sidebar.selectbox("Disaster Mode", DISASTER_MODES, index=0)
flood_threshold = st.sidebar.slider(
    "Flood Threshold (metres REM)", 0.5, 10.0, FLOOD_THRESHOLD_DEFAULT, 0.5
)
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Standard", "Heavy", "Ambulance"])
vehicle_count = st.sidebar.number_input("Number of Vehicles (VRP)", 1, 10, 1)

st.sidebar.markdown("---")

st.sidebar.subheader("🔑 API Keys")
tomtom_api_key = st.sidebar.text_input(
    "TomTom API Key",
    value=os.getenv("TOMTOM_API_KEY", ""),
    type="password",
)
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password",
)

st.sidebar.markdown("---")

st.sidebar.subheader("📍 Reserve (Origin)")
origin_lat = st.sidebar.number_input("Origin Latitude", value=DEFAULT_LOCATION[0], format="%.6f")
origin_lon = st.sidebar.number_input("Origin Longitude", value=DEFAULT_LOCATION[1], format="%.6f")

st.sidebar.subheader("🎯 Emergency Zones")
zones_input = st.sidebar.text_area(
    "lat, lon, label, priority (one per line)",
    value="13.3500,74.7500,Hospital,1\n13.3200,74.7300,Shelter,2",
    height=120,
)

buffer_km = st.sidebar.slider("Map Radius (km)", 1.0, 20.0, 5.0, 0.5)

run_btn = st.sidebar.button("🗺️ Compute Routes", type="primary", use_container_width=True)
tsp_btn = st.sidebar.button("🔄 Solve TSP/VRP", use_container_width=True)


# ── Helper Functions ──────────────────────────────────────────

def parse_zones(text: str) -> list[dict]:
    """Parse zone text input into list of dicts."""
    zones = []
    for line in text.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            zones.append({
                "lat": float(parts[0]),
                "lon": float(parts[1]),
                "label": parts[2],
                "priority": int(parts[3]) if len(parts) > 3 else 1,
                "time_window_minutes": int(parts[4]) if len(parts) > 4 else 60,
            })
    return zones


def route_color(risk_score: float) -> str:
    """Map risk score to colour: green/yellow/red."""
    if risk_score < 25:
        return "#22c55e"  # green
    elif risk_score < 60:
        return "#eab308"  # yellow
    else:
        return "#ef4444"  # red


def build_map(
    origin: tuple[float, float],
    routes: list[RouteResult],
    incidents_geojson: dict | None = None,
    zones: list[dict] | None = None,
    satellite_flags: list[SatelliteFlag] | None = None,
) -> folium.Map:
    """Build a Folium map with routes, incidents, and zone markers."""
    m = folium.Map(
        location=origin,
        zoom_start=13,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    )

    # Add OpenStreetMap as an alternative layer
    folium.TileLayer("openstreetmap", name="OpenStreetMap").add_to(m)
    folium.LayerControl().add_to(m)

    # Origin marker
    folium.Marker(
        origin,
        popup="🏥 Reserve (Origin)",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    # Zone markers
    for z in (zones or []):
        folium.Marker(
            (z["lat"], z["lon"]),
            popup=f"🎯 {z['label']} (P{z['priority']})",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

    # Route lines
    dash_patterns = {
        "SAFEST": None,  # solid
        "FASTEST": "10 5",
        "HIGHWAY-PREFERRED": "5 10",
    }
    for route in routes:
        if not route.coords:
            continue
        color = route_color(route.risk_score)
        folium.PolyLine(
            route.coords,
            color=color,
            weight=5,
            opacity=0.85,
            dash_array=dash_patterns.get(route.label),
            popup=(
                f"<b>{route.label}</b><br>"
                f"Distance: {route.total_distance_m/1000:.1f} km<br>"
                f"Risk: {route.risk_score:.0f}/100"
            ),
        ).add_to(m)

    # Incident markers
    if incidents_geojson and incidents_geojson.get("features"):
        for feat in incidents_geojson["features"]:
            coords = feat["geometry"]["coordinates"]
            props = feat["properties"]
            icon_color = "darkred" if props.get("is_hard_block") else "orange"
            folium.CircleMarker(
                (coords[1], coords[0]),
                radius=8,
                color=icon_color,
                fill=True,
                popup=f"⚠️ {props['type']}: {props.get('description', '')}",
            ).add_to(m)

    # Satellite flags
    for sf in (satellite_flags or []):
        if sf.flagged:
            folium.CircleMarker(
                (sf.lat, sf.lon),
                radius=12,
                color="#3b82f6",
                fill=True,
                fill_opacity=0.5,
                popup=sf.message,
            ).add_to(m)

    return m


# ── Main Pipeline ─────────────────────────────────────────────

if run_btn:
    zones = parse_zones(zones_input)
    bbox = bbox_from_point(origin_lat, origin_lon, buffer_km)
    st.session_state["bbox"] = bbox

    with st.spinner("📥 Downloading road network from OpenStreetMap..."):
        G = download_graph(bbox)
        G = add_elevation_data(G)
        G = enrich_edge_attributes(G)

    with st.spinner(f"🌊 Applying terrain analysis ({disaster_mode} mode)..."):
        G = apply_terrain_layer(G, flood_threshold=flood_threshold)
        G = filter_graph_by_disaster_mode(G, disaster_mode, flood_threshold)

    if tomtom_api_key:
        with st.spinner("🚦 Fetching live TomTom incidents..."):
            incidents = fetch_incidents(tomtom_api_key, bbox)
            incidents = snap_incidents_to_graph(G, incidents)
            G = apply_incidents_to_graph(G, incidents)
            st.session_state["incidents"] = incidents
    else:
        st.session_state["incidents"] = []

    with st.spinner("⚖️ Computing dynamic edge weights..."):
        G = assign_weights(G, mode=disaster_mode, flood_threshold=flood_threshold)

    st.session_state["graph"] = G
    st.session_state["graph_stats"] = get_graph_stats(G)
    st.session_state["weight_stats"] = get_weight_statistics(G)
    st.session_state["blocked_edges"] = get_blocked_edges(G)

    # Single-pair route to first zone
    if zones:
        with st.spinner("🧭 Computing routes..."):
            routes = compute_top3_routes(
                G, origin_lat, origin_lon, zones[0]["lat"], zones[0]["lon"]
            )
            st.session_state["routes"] = routes

        # Satellite analysis on safest route waypoints (free OSM tiles)
        if routes and routes[0].coords:
            with st.spinner("🛰️ Analysing satellite imagery for route waypoints..."):
                # Sample every 10th waypoint to limit tile fetches
                sample_coords = routes[0].coords[::10]
                sat_flags = analyse_waypoints(sample_coords)
                st.session_state["satellite_flags"] = sat_flags
        else:
            st.session_state["satellite_flags"] = []

    st.success("✅ Routes computed successfully!")

if tsp_btn and st.session_state["graph"] is not None:
    zones = parse_zones(zones_input)
    G = st.session_state["graph"]

    with st.spinner("🔄 Solving TSP/VRP with OR-Tools..."):
        tsp_result = solve_tsp(
            G,
            depot_lat=origin_lat,
            depot_lon=origin_lon,
            stops=zones,
            vehicle_count=vehicle_count,
        )
        st.session_state["tsp_result"] = tsp_result

    if tsp_result.warnings:
        for w in tsp_result.warnings:
            st.warning(w)
    else:
        st.success(
            f"✅ TSP solved! Total distance: {tsp_result.total_distance_m/1000:.1f} km"
        )


# ── Main Layout ───────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Live Route Map")

    zones = parse_zones(zones_input)
    incidents_gj = incidents_to_geojson(st.session_state["incidents"]) if st.session_state["incidents"] else None

    fmap = build_map(
        origin=(origin_lat, origin_lon),
        routes=st.session_state["routes"],
        incidents_geojson=incidents_gj,
        zones=zones,
        satellite_flags=st.session_state["satellite_flags"],
    )
    st_folium(fmap, height=550, use_container_width=True)


with col2:
    st.subheader("📊 Route Analysis")

    if st.session_state["routes"]:
        for r in st.session_state["routes"]:
            color = route_color(r.risk_score)
            risk_label = "LOW" if r.risk_score < 25 else ("MEDIUM" if r.risk_score < 60 else "HIGH")
            st.markdown(
                f"**{r.label}** ({r.route_id})  \n"
                f"Distance: `{r.total_distance_m/1000:.1f} km` · "
                f"Risk: <span style='color:{color}'>{r.risk_score:.0f}/100 ({risk_label})</span>",
                unsafe_allow_html=True,
            )
            if r.warnings:
                for w in r.warnings:
                    st.caption(f"  ⚠️ {w}")
            st.markdown("---")
    else:
        st.info("Click **Compute Routes** to see route analysis.")

    # Graph stats
    if st.session_state["graph_stats"]:
        stats = st.session_state["graph_stats"]
        wstats = st.session_state["weight_stats"]
        st.subheader("📈 Graph Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes", stats.get("nodes", 0))
        c2.metric("Edges", stats.get("edges", 0))
        c3.metric("Blocked", wstats.get("blocked_edges", 0))

    # Active incidents
    st.subheader("⚠️ Active Incidents")
    if st.session_state["incidents"]:
        for inc in st.session_state["incidents"][:10]:
            icon = "🔴" if inc.is_hard_block else "🟡"
            st.caption(f"{icon} **{inc.type_label}**: {inc.description or 'No details'}")
    else:
        st.caption("No live incidents (TomTom key required).")

    # Satellite flags
    st.subheader("🛰️ Satellite Flags")
    if st.session_state["satellite_flags"]:
        for sf in st.session_state["satellite_flags"]:
            if sf.flagged:
                st.warning(sf.message)
                if sf.image_path and os.path.exists(sf.image_path):
                    st.image(sf.image_path, width=200, caption=f"Waypoint {sf.waypoint_index}")
    else:
        st.caption("No satellite analysis yet.")

    # TSP result
    if st.session_state["tsp_result"] is not None:
        tsp = st.session_state["tsp_result"]
        st.subheader("🔄 TSP/VRP Result")
        st.metric("Total Distance", f"{tsp.total_distance_m/1000:.1f} km")
        if tsp.stop_order:
            zone_names = [z["label"] for z in parse_zones(zones_input)]
            ordered = [zone_names[i] if i < len(zone_names) else f"Stop {i}" for i in tsp.stop_order]
            st.write("**Visit order:** " + " → ".join(ordered))
        if tsp.unreachable_stops:
            st.error(f"⚠️ Unreachable stops: {tsp.unreachable_stops}")


# ── AI Chat Panel ─────────────────────────────────────────────

st.markdown("---")
st.subheader("🤖 DAIRS-AI Assistant (Groq)")

# Display chat history
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Ask about routes, incidents, or re-planning...")

if user_msg:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_msg)
    st.session_state["chat_history"].append({"role": "user", "content": user_msg})

    # Build context
    zones = parse_zones(zones_input)
    context = build_context_message(
        mode=disaster_mode,
        bbox=st.session_state.get("bbox", (0, 0, 0, 0)) or (0, 0, 0, 0),
        origin={"lat": origin_lat, "lon": origin_lon, "label": "Reserve"},
        destinations=zones,
        flood_threshold=flood_threshold,
        vehicle_count=vehicle_count,
        vehicle_type=vehicle_type.lower(),
        incidents=st.session_state["incidents"],
        satellite_flags=st.session_state["satellite_flags"],
        graph_stats=st.session_state["graph_stats"] or {},
        routes=st.session_state["routes"],
        tsp_result=st.session_state["tsp_result"],
    )

    # Stream AI response
    with st.chat_message("assistant"):
        response_chunks = stream_chat(
            user_message=user_msg,
            context=context,
            history=[
                m for m in st.session_state["chat_history"]
                if m["role"] in ("user", "assistant")
            ][:-1],  # exclude current message, already added
            api_key=groq_api_key,
        )
        full_response = st.write_stream(response_chunks)

    st.session_state["chat_history"].append(
        {"role": "assistant", "content": full_response}
    )
