# DAIRS — Disaster-Aware Intelligent Routing System

A multi-modal, AI-assisted emergency routing system that fuses terrain analysis, live incident feeds, satellite imagery, and combinatorial optimization to compute safe routes during disasters.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│   (Disaster Mode Selector, Map Input, Route Visualizer)     │
└─────────────────┬───────────────────────────────────────────┘
                  │
     ┌────────────▼─────────────┐
     │     Orchestrator (AI)    │  ← System Prompt (Groq / Llama 3.3)
     │   Route Planner Engine   │
     └──┬──────┬──────┬─────────┘
        │      │      │
   ┌────▼──┐ ┌─▼────┐ ┌▼──────────┐
   │OSMnx  │ │TomTom│ │ TSP/VRP   │
   │+Elev. │ │ API  │ │ (OR-Tools)│
   └───────┘ └──────┘ └───────────┘
        │      │
   ┌────▼──────▼───────────┐
   │  Graph Weight Engine  │  ← merges terrain + incidents
   └───────────────────────┘
```

## Modules

| Module | File | Purpose |
|--------|------|---------|
| 1. Graph Extraction | `graph_extraction.py` | OSMnx road network download, elevation, road-class filtering |
| 2. Terrain Layer | `terrain_layer.py` | REM computation, flood risk, slope/bridge/tunnel flags |
| 3. TomTom Incidents | `tomtom_layer.py` | Live traffic incidents, snapping, hard blocks & penalties |
| 4. Weight Engine | `weight_engine.py` | Composite edge weight: `d × α_terrain × β_incident × γ_road` |
| 5. Routing Solver | `routing_solver.py` | Dijkstra single-pair + OR-Tools TSP/VRP |
| 6. Satellite Analysis | `satellite_analysis.py` | Free Esri World Imagery tiles + NDWI water detection |
| AI Orchestrator | `orchestrator.py` | Groq (Llama 3.3) system prompt + context injection + streaming chat |
| Configuration | `config.py` | All constants, thresholds, API key loading |
| Frontend | `app.py` | Streamlit UI with Folium map, controls, chat panel |

## Disaster Modes

| Mode | Key Behaviour |
|------|---------------|
| **Flood** | REM-based edge blocking, elevation preference, TomTom flood hard blocks |
| **Earthquake** | Motorway/trunk/primary only, tunnel avoidance, slope > 30° flagged |
| **Cyclone** | Coastal road avoidance, bridge penalties, inland route preference |
| **General** | TomTom hard blocks only, standard shortest-path |

## Edge Weight Formula

$$w_{edge} = d_{road} \times \alpha_{terrain} \times \beta_{incident} \times \gamma_{road\_class}$$

- $\alpha_{terrain} = e^{k \cdot \max(0,\, T_{flood} - elev)}$ — exponential flood penalty
- $\beta_{incident} \in \{1.0,\; 2.5,\; \infty\}$ — TomTom incident multiplier
- $\gamma_{road\_class}$ — motorway=0.8, primary=1.0, residential=1.6, track=3.0

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required keys:
- **TomTom API Key** — for live traffic incidents ([developer.tomtom.com](https://developer.tomtom.com/))
- **Groq API Key** — for the DAIRS-AI chat assistant ([console.groq.com](https://console.groq.com/))

No Google Maps API key is needed — the system uses free SRTM elevation data, Open-Elevation API, and Esri World Imagery tiles.

### 3. Run the application

```bash
streamlit run app.py
```

## Usage

1. Select a **Disaster Mode** from the sidebar
2. Set the **origin** (reserve/hospital) coordinates
3. Add **emergency zones** (one per line: `lat,lon,label,priority`)
4. Click **Compute Routes** for single-destination routing
5. Click **Solve TSP/VRP** for multi-stop optimization
6. Use the **DAIRS-AI chat** to ask questions about routes, incidents, or re-planning

## Key Features

- **Satellite base map** via Esri World Imagery on Folium (free, no API key)
- **3 route options**: SAFEST / FASTEST / HIGHWAY-PREFERRED with risk colour coding
- **Live TomTom incident overlay** with auto-refresh
- **Satellite imagery water detection** at route waypoints
- **OR-Tools TSP/VRP** with guided local search metaheuristic
- **AI chat assistant** (Groq / Llama 3.3 70B) with full routing context injection

## Project Alignment (CSS 2203 IAI)

- **AI Domain**: Geospatial AI + Combinatorial Optimization
- **Problem Statement**: Safe multi-modal emergency routing under real-time disaster constraints
- **Methodology**: Graph theory (OSMnx/NetworkX) + Heuristic optimization (TSP/VRP via OR-Tools) + Live data fusion (TomTom) + Computer vision (satellite imagery classification)
- **Novel Hypothesis**: Dynamic edge weight re-computation fusing terrain REM scores with live incident streams produces safer evacuation routes vs. static map routing
