# DARTS — Dynamic Agent-based Relief Transshipment System
### CSS 2203 IAI Project | Team README & Onboarding Document

> **One-line summary:** A two-tier AI system where a command LLM decides *who gets what and from where*, and vehicle agents independently navigate Udupi's real road network to deliver it — all in real-time, during a simulated coastal disaster.

---

## Table of Contents
1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [Why Is This Novel?](#2-why-is-this-novel)
3. [System Overview (The Big Picture)](#3-system-overview)
4. [The Two-Tier Architecture](#4-the-two-tier-architecture)
5. [Node Types in Our World](#5-node-types-in-our-world)
6. [Commodities and Supply Model](#6-commodities-and-supply-model)
7. [Full Pipeline — Step by Step](#7-full-pipeline--step-by-step)
8. [The Simulation Scenario](#8-the-simulation-scenario)
9. [How Each Module Works](#9-how-each-module-works)
10. [Data Flow Between Modules](#10-data-flow-between-modules)
11. [Task Split — 6 People](#11-task-split--6-people)
12. [Running the Project](#12-running-the-project)
13. [Paper Sections (First Draft)](#13-paper-sections-first-draft)

---

## 1. What Problem Are We Solving?

During a coastal disaster (cyclone, flood) in Udupi district, Karnataka:

- Multiple isolated villages, hospitals, and NDRF staging camps **simultaneously** call for help
- There are **3 supply depots** across the district, each with **limited, different stocks** of food, medicine, and equipment
- Roads get **closed mid-operation** due to flooding or debris
- **No single depot can serve everyone** — some have excess food but no medicine; others have ambulances but no communications gear

Standard GPS routing (Google Maps, TomTom's own app) tells you *how to get somewhere*. It does not tell you *which vehicle should go where, carrying what, from which depot, in what priority order*.

Standard VRP (Vehicle Routing Problem) algorithms compute an optimal static plan *before deployment*. They cannot react when a new distress call arrives at T=18min or a road closes at T=12min.

**DARTS solves this by combining:**
- An LLM (large language model) that acts as a real-time incident commander, deciding dispatch priorities
- Vehicle agents that independently navigate the real Udupi road network using live TomTom traffic data
- A live inventory tracker that monitors what each depot has left and triggers inter-depot transfers when needed

---

## 2. Why Is This Novel?

| What everyone else does | What DARTS does |
|---|---|
| One depot, one vehicle, static roads | Multiple depots, multiple vehicles, live road closures |
| All info in one place | Hard information wall: command knows demand, vehicles know roads |
| Offline pre-computed plan | Real-time replanning at every timestep |
| No explanation for decisions | LLM generates plain-English reason for every dispatch |
| Synthetic grid maps | Real OSMnx graph of Udupi, Karnataka |
| Vehicles and cargo are separate | Ambulances are both a vehicle *and* a commodity (duality) |

The **information asymmetry** is the key insight: the command LLM cannot see road conditions. The vehicle agents cannot see which node is most critical. This mirrors how real Indian NDRF Incident Command System (ICS) actually works — and no prior paper in humanitarian logistics has modelled this separation formally.

---

## 3. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DARTS — top-level view                       │
│                                                                 │
│  Distress calls ──►  COMMAND LLM  ◄──  Depot inventory state   │
│  (node requests)     (Groq API)         (live stockpile counts) │
│         │                │                                      │
│         │          Dispatch orders                              │
│         │         (what · where · why)                          │
│         ▼                ▼                                      │
│  ┌─────────────────────────────────────┐                        │
│  │      VEHICLE ROUTING AGENTS         │                        │
│  │  OSMnx Udupi graph + TomTom API     │                        │
│  │  A* pathfinding · dynamic replan    │                        │
│  └─────────────────────────────────────┘                        │
│         │                                                       │
│         ▼                                                       │
│  NetworkX + Matplotlib animation  ·  Explanation log            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. The Two-Tier Architecture

This is the most important design decision. There are **two completely separate agents** that communicate through a strict interface, like a military command chain.

### Tier 1 — Strategic / Command (the LLM)

The command LLM acts as the **Incident Commander**. It:

- Receives structured JSON "distress call" packets from nodes every simulation timestep
- Sees the current inventory at every depot (how many MREs, medicines, comms kits, ambulances remain)
- Decides: which vehicle from which depot goes to which node, carrying what commodity and how much
- Outputs a structured dispatch order + a plain English explanation of its reasoning
- **Does NOT know**: road conditions, which roads are closed, how long the journey will take

### Tier 2 — Tactical / Navigation (the vehicle agents)

Each vehicle is an autonomous routing agent. It:

- Receives a single instruction from Tier 1: "go to node X"
- Uses the OSMnx road graph of Udupi to find the shortest path
- Queries TomTom Traffic API at each timestep — if a road segment is closed or congested, it recalculates the path using A* on the updated graph
- Reports back its position and arrival status to the command tier
- **Does NOT know**: why it was sent there, what other vehicles are doing, which nodes are most critical

### The interface between tiers

```
Tier 1 (LLM)  ──[dispatch_order JSON]──►  Tier 2 (vehicle agent)
Tier 2 (agent) ──[status_update JSON]──►  Tier 1 (LLM)
```

A dispatch order looks like:
```json
{
  "vehicle_id": "AMB-02",
  "from_depot": "DEPOT-B",
  "destination_node": "NODE-KMC-HOSPITAL",
  "cargo": {"medicines": 40},
  "priority": "critical",
  "reason": "KMC Hospital flagged critical medicine shortage. DEPOT-B holds 60 units surplus. AMB-02 is nearest available vehicle at 3.2km."
}
```

A status update looks like:
```json
{
  "vehicle_id": "AMB-02",
  "current_position": [13.3409, 74.7421],
  "status": "en_route",
  "eta_steps": 4,
  "rerouted": true,
  "reroute_reason": "coastal_road_NH169_closed"
}
```

---

## 5. Node Types in Our World

We have **four types of nodes** in our Udupi simulation. Each has different roles, different commodity needs, and different urgency levels.

```
NODE TYPE          ROLE                   REQUESTS                  URGENCY
─────────────────────────────────────────────────────────────────────────────
NDRF Base Depot    Supply source          Dispatches everything     N/A (source)
                   (3 in simulation)      Holds all 4 commodity
                                          types + vehicle fleet

Hospital /         Tertiary care          Medicines (critical)      Critical / stable
Tertiary center    (KMC Manipal,          Ambulance transfers
                   Udupi Dist. Hosp.)

NDRF Disaster      Regional staging       MRE meals (high vol.)     High / medium
Staging Camp       (coastal flood camp)   Comms gear
                                          Medicines (moderate)

Isolated           Small hamlet,          MRE meals (immediate)     SOS / high
Distress Point     fishing village        Ambulance (1–5 people)
                   (~2–4 people)          GPS signal only
```

### Real locations used (Udupi district, Karnataka)

| Node ID | Real location | Type |
|---|---|---|
| DEPOT-A | NDRF staging base, Udupi town | NDRF Base Depot |
| DEPOT-B | Kundapur sub-depot (simulated) | NDRF Base Depot |
| DEPOT-C | Karkala relief base (simulated) | NDRF Base Depot |
| NODE-H1 | KMC Manipal Hospital | Hospital |
| NODE-H2 | Udupi District Hospital | Hospital |
| NODE-S1 | Padubidri coastal camp (simulated) | Disaster Staging |
| NODE-S2 | Brahmavar relief point (simulated) | Disaster Staging |
| NODE-D1 | Trasi fishing hamlet (simulated) | Isolated Distress |
| NODE-D2 | Gangolli coast (simulated) | Isolated Distress |
| NODE-D3 | Byndoor interior (simulated) | Isolated Distress |

---

## 6. Commodities and Supply Model

We track **4 commodity types**, each with a finite integer count per depot:

| Commodity | Unit | Vehicle that carries it | Notes |
|---|---|---|---|
| MRE meals | Meal packs | Supply truck | Bulk, high volume |
| Emergency medicines | Medicine kits | Ambulance or truck | Critical, low tolerance for delay |
| Comms gear | Device units (walkie-talkies, radios) | Supply truck | Heavy, limited stock |
| Ambulances | Vehicle count | Self-routing | Both commodity AND vehicle |

### Ambulance duality

Ambulances are unique: they are simultaneously a **resource being dispatched** (the hospital or distress point needs the ambulance itself) and a **vehicle doing the routing**. When the command LLM dispatches `{"cargo": {"ambulances": 1}}` to a hospital, it means one ambulance drives itself to the hospital and stays there. The inventory at the source depot decreases by 1. This is called **commodity-vehicle duality** and is formally novel in the VRP literature.

### Initial inventory (simulation start)

```
              MRE meals   Medicines   Comms gear   Ambulances
DEPOT-A          120          60          15            3
DEPOT-B           60          15           8            2
DEPOT-C           80          80          10            2
```

---

## 7. Full Pipeline — Step by Step

This is the exact sequence of what happens when you run the simulation.

```
STEP 0 — INITIALISATION
│
├── Load OSMnx road graph for Udupi district (osmnx.graph_from_place)
├── Assign GPS coordinates to all 10 nodes
├── Snap nodes to nearest OSMnx graph nodes
├── Initialise depot inventory dicts (commodity counts)
├── Initialise vehicle fleet (position = depot location, cargo = empty)
└── Start simulation clock at T=0

STEP 1 — EVENT GENERATION (runs every timestep)
│
├── Distress event generator fires Poisson-distributed demand events
│   └── Each event: {node_id, commodity_type, quantity, urgency}
├── Road closure simulator applies/lifts TomTom-style edge weight changes
└── New events appended to pending_events queue

STEP 2 — PERCEPT COLLECTION (runs every timestep)
│
├── Collect all pending_events since last timestep
├── Collect current depot inventory snapshot
├── Collect vehicle position + status for all vehicles
└── Package into world_state JSON

STEP 3 — COMMAND LLM DECISION (Groq API call)
│
├── Send world_state JSON to LLM with system prompt
│   System prompt defines:
│     - LLM's role (incident commander)
│     - What it can and cannot see (no road data)
│     - Output format (structured dispatch_orders list)
│     - Priority rules (SOS > critical > high > medium)
│     - Lateral transfer logic (if depot X has shortage, check others)
│
├── LLM returns: list of dispatch_orders + explanation_text
└── Dispatch orders written to dispatch_queue

STEP 4 — INVENTORY UPDATE
│
├── For each dispatch_order, subtract cargo from source depot
├── If depot stock would go negative → LLM is called again with constraint
└── Update depot inventory state

STEP 5 — VEHICLE AGENT ROUTING (runs per vehicle, per timestep)
│
├── Each vehicle with a pending dispatch_order:
│   ├── Check if current route is still valid (query TomTom edge weights)
│   ├── If route blocked → rerun A* on updated graph (dynamic replan)
│   ├── Advance vehicle one step along current route
│   └── If arrived → unload cargo, update node inventory, send status update
│
└── Vehicles without orders: remain idle at current position

STEP 6 — STATUS REPORTING
│
├── All vehicle statuses collected
├── Lateral transfer check: if any depot is below threshold on any commodity,
│   flag for next LLM call
└── Metrics updated: deliveries made, time elapsed, shortfalls remaining

STEP 7 — VISUALISATION UPDATE
│
├── NetworkX graph redrawn with:
│   ├── Vehicle positions (moving dots, colour by type)
│   ├── Node demand bars (height = urgency/demand level)
│   ├── Closed edges (red overlay)
│   └── Depot inventory gauges
└── Explanation log printed to console/file

STEP 8 — REPEAT from STEP 1 until T=T_max or all demand satisfied
```

### Flowchart — one full timestep

```
         ┌──────────────────────┐
         │   New events arrive  │
         │  (distress calls,    │
         │   road closures)     │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Package world_state │
         │  (demand + inventory │
         │   + vehicle status)  │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   LLM Command call   │◄─── system prompt (role, rules, format)
         │   (Groq API)         │
         └──────────┬───────────┘
                    │
              dispatch_orders
                    │
         ┌──────────┴───────────┐
         ▼                      ▼
┌─────────────────┐    ┌──────────────────┐
│ Update depot    │    │  Send order to   │
│ inventory       │    │  vehicle agent   │
└─────────────────┘    └────────┬─────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Vehicle: plan route  │
                    │  A* on OSMnx graph    │
                    │  with TomTom weights  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Road closed?   YES ──►  Replan A*
                    │     NO                │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Advance vehicle 1    │
                    │  step along route     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Arrived at dest?     │
                    │  YES ──► unload cargo │
                    │  NO  ──► continue     │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Update visualisation │
                    │  + metrics + log      │
                    └───────────────────────┘
                                │
                                ▼
                          Next timestep
```

---

## 8. The Simulation Scenario

We run one fixed scenario (the "Udupi Cyclone Scenario") to evaluate the system. This is not random — it is a designed sequence of events that tests all system capabilities.

| Timestep | Event | What system must do |
|---|---|---|
| T=0 | Simulation starts. All depots stocked. No active demand. | Initialise. |
| T=3 | NODE-D1 (Trasi hamlet): SOS — 4 people, needs MRE + ambulance | Command dispatches AMB from DEPOT-A, MRE truck from DEPOT-A |
| T=7 | NODE-H1 (KMC Hospital): critical — needs 40 medicine kits | Command dispatches medicine from DEPOT-A. DEPOT-A has 60 units. |
| T=12 | TomTom: coastal road NH-169 segment closed (flood) | Vehicle en route to NODE-D1 detects closure, rerouts via NH-66. Command not informed. |
| T=15 | NODE-S1 (Padubidri camp): high — 80 MRE, 6 comms kits | Command checks depots. DEPOT-A MRE now low. Dispatches from DEPOT-B. |
| T=18 | NODE-H2 (Udupi Dist. Hospital): critical — 50 medicine kits. DEPOT-B has only 15. | Command triggers lateral transfer: DEPOT-A sends 40 units to DEPOT-B, then DEPOT-B dispatches to H2. |
| T=22 | NODE-D2 (Gangolli): SOS — 2 people | DEPOT-A ambulances now depleted. Command dispatches from DEPOT-C. |
| T=28 | NODE-D3 (Byndoor): high — MRE | Last available stock from DEPOT-C dispatched. |
| T=35 | All demand resolved or vehicles en route. Simulation ends. | Report final metrics. |

---

## 9. How Each Module Works

The codebase has **6 Python modules**, one per team member. They are as independent as possible and communicate only through shared JSON files (no shared state, no imports between modules).

### Module 1: `command_llm.py` — the LLM dispatcher

- Reads `world_state.json` (written by simulation engine)
- Constructs a prompt for Groq API
- Parses LLM response into structured `dispatch_orders.json`
- Writes `explanation_log.txt` (one entry per timestep)
- Has no knowledge of roads. Input is demand + inventory only.

### Module 2: `graph_builder.py` — OSMnx + TomTom integration

- Downloads Udupi road network via `osmnx.graph_from_place("Udupi, Karnataka, India")`
- Stores graph as `udupi_graph.graphml`
- Exposes `update_edge_weights(closure_events)` — applies TomTom-style weight changes
- Writes `current_graph.json` (edge list with current weights) every timestep

### Module 3: `vehicle_agent.py` — routing agent

- Reads `dispatch_orders.json` for its vehicle ID
- Reads `current_graph.json` for current road state
- Runs `networkx.astar_path()` to compute route
- Advances vehicle one hop per timestep
- Detects if next edge weight = ∞ (closed) → reruns A*
- Writes its position + status to `vehicle_states.json`

### Module 4: `simulation_engine.py` — the orchestrator

- Runs the main timestep loop
- Generates distress events (Poisson-distributed)
- Fires road closure events at pre-set timesteps
- Reads vehicle statuses, updates depot inventories
- Writes `world_state.json` every timestep
- Calls modules 1, 2, 3 in sequence each step

### Module 5: `visualiser.py` — animation

- Reads `udupi_graph.graphml`, `vehicle_states.json`, `world_state.json`
- Draws OSMnx graph using Matplotlib
- Animates vehicle positions as moving dots
- Colours closed edges red
- Shows depot inventory as bar charts per node
- Saves animation as `darts_simulation.mp4`

### Module 6: `metrics.py` + paper writing

- Reads final `world_state.json` and `explanation_log.txt`
- Computes: total delivery time, number of shortfall events, number of replans, lateral transfers triggered, LLM call count
- Compares against baseline (static VRP, no replanning)
- Writes results tables for the paper

---

## 10. Data Flow Between Modules

All modules communicate through JSON files in a shared `/data/` folder. No module imports another. This means zero merge conflicts and fully independent development.

```
simulation_engine.py
    │
    ├── writes ──► /data/world_state.json         ◄── read by command_llm.py
    ├── writes ──► /data/closure_events.json       ◄── read by graph_builder.py
    │
    ▼
graph_builder.py
    └── writes ──► /data/current_graph.json        ◄── read by vehicle_agent.py

command_llm.py
    └── writes ──► /data/dispatch_orders.json      ◄── read by vehicle_agent.py
                ── writes ──► /data/explanation_log.txt

vehicle_agent.py
    └── writes ──► /data/vehicle_states.json       ◄── read by simulation_engine.py
                                                   ◄── read by visualiser.py

simulation_engine.py + vehicle_agent.py
    └── write ──► /data/final_metrics.json         ◄── read by metrics.py

visualiser.py
    └── reads all of the above, produces animation
```

### Shared data schemas

**`world_state.json`** (written by engine, read by LLM):
```json
{
  "timestep": 12,
  "depot_inventory": {
    "DEPOT-A": {"mre": 45, "medicine": 20, "comms": 10, "ambulances": 1},
    "DEPOT-B": {"mre": 60, "medicine": 0, "comms": 8, "ambulances": 2}
  },
  "pending_requests": [
    {"node_id": "NODE-H2", "commodity": "medicine", "quantity": 50, "urgency": "critical"}
  ],
  "vehicle_status": [
    {"id": "AMB-01", "status": "en_route", "destination": "NODE-D1", "eta": 4}
  ]
}
```

**`dispatch_orders.json`** (written by LLM, read by vehicles):
```json
{
  "timestep": 12,
  "orders": [
    {
      "vehicle_id": "TRUCK-01",
      "from_depot": "DEPOT-B",
      "destination": "NODE-H2",
      "cargo": {"medicine": 40},
      "priority": "critical",
      "reason": "Udupi District Hospital critical shortage. DEPOT-B nearest with sufficient stock after lateral transfer."
    }
  ]
}
```

---

## 11. Task Split — 6 People

Each section below is written as a **self-contained brief** for one team member. Copy-paste your section into any LLM and it will know exactly what to build.

---

### Person 1 — Command LLM (`command_llm.py`)

**Your job**: Build the brain of the system. The LLM that acts as incident commander.

**What you build**: `command_llm.py`

**Inputs you read**: `/data/world_state.json` (written by Person 4)

**Outputs you write**: `/data/dispatch_orders.json`, `/data/explanation_log.txt`

**Step-by-step**:
1. Install: `pip install groq`
2. Load `world_state.json`
3. Build a prompt. The system prompt should tell the LLM: "You are an NDRF incident commander. You will receive the current inventory at 3 depots and a list of pending distress requests. You must assign vehicles from depots to destinations. You cannot see road conditions — only demand and inventory. Respond in JSON only."
4. The user prompt should be the `world_state.json` content, formatted clearly
5. Call Groq API (`llama-3.3-70b-versatile` or `mixtral-8x7b`)
6. Parse the JSON response into `dispatch_orders.json`
7. Append the `reason` fields to `explanation_log.txt`
8. Handle the case where the LLM tries to dispatch more stock than available (re-prompt with correction)
9. Handle **lateral transfers**: if a depot has `medicine: 0` and another has `medicine: 60`, the LLM should be prompted to create two orders — one depot-to-depot, one depot-to-node

**Test without other modules**: Create a fake `world_state.json` manually and run your module standalone. Check that `dispatch_orders.json` is valid JSON and `explanation_log.txt` is readable.

**Do not touch**: graph, routing, visualisation, simulation loop.

---

### Person 2 — Road Graph + TomTom (`graph_builder.py`)

**Your job**: Build and maintain the real Udupi road network that vehicles navigate.

**What you build**: `graph_builder.py`, `udupi_graph.graphml` (generated once)

**Inputs you read**: `/data/closure_events.json` (written by Person 4)

**Outputs you write**: `/data/current_graph.json`, `udupi_graph.graphml`

**Step-by-step**:
1. Install: `pip install osmnx networkx`
2. Run once to download the graph:
   ```python
   import osmnx as ox
   G = ox.graph_from_place("Udupi, Karnataka, India", network_type="drive")
   ox.save_graphml(G, "udupi_graph.graphml")
   ```
3. Write a function `snap_node_to_graph(lat, lon, G)` that finds the nearest OSMnx node to a GPS coordinate. Use this to attach our 10 simulation nodes to real graph nodes.
4. Write a function `apply_closure(G, edge_u, edge_v)` that sets the weight of an edge to 999999 (effectively infinite — impassable)
5. Write a function `lift_closure(G, edge_u, edge_v)` that resets the weight
6. Every timestep, read `closure_events.json` and apply/lift closures
7. Write the updated edge list to `current_graph.json` — this is what vehicles read for routing

**Closure events format** (what you receive from Person 4):
```json
{"timestep": 12, "closures": [{"u": 123456, "v": 789012, "action": "close"}]}
```

**Test without other modules**: Download the graph, snap the 10 node coordinates listed in Section 5, visualise with `ox.plot_graph(G)`. Confirm all nodes land on real roads.

**Do not touch**: LLM, vehicle movement, simulation loop.

---

### Person 3 — Vehicle Routing Agent (`vehicle_agent.py`)

**Your job**: Each vehicle is an agent that navigates itself. Build that navigation logic.

**What you build**: `vehicle_agent.py`

**Inputs you read**: `/data/dispatch_orders.json` (from Person 1), `/data/current_graph.json` (from Person 2)

**Outputs you write**: `/data/vehicle_states.json`

**Step-by-step**:
1. Install: `pip install networkx osmnx`
2. Define a `Vehicle` class with fields: `id`, `position` (OSMnx node ID), `cargo`, `route` (list of node IDs), `status` (idle/en_route/arrived)
3. At each timestep:
   - Read `dispatch_orders.json`. If your vehicle has a new order, set `self.destination`
   - Load the graph from `udupi_graph.graphml`
   - Apply current edge weights from `current_graph.json`
   - If `self.route` is empty or next step is blocked: run `networkx.astar_path(G, source, target, weight='weight')`
   - Advance vehicle to next node in route
   - If next edge weight = 999999 (closed): set `rerouted = True`, rerun A*
   - If arrived at destination: set `status = arrived`, unload cargo
4. Write all vehicle states to `vehicle_states.json`

**Heuristic for A***: use haversine distance between node coordinates as the heuristic (`ox.distance.great_circle`)

**Fleet to simulate** (hardcode these):
- `AMB-01`, `AMB-02`, `AMB-03` (ambulances — capacity: 2 patients or 20 medicine kits)
- `TRUCK-01`, `TRUCK-02`, `TRUCK-03` (supply trucks — capacity: 100 MRE or 50 comms units)
- `AMB-04`, `AMB-05` (Depot C ambulances)

**Test without other modules**: Create a fake `dispatch_orders.json` sending AMB-01 from DEPOT-A to NODE-H1. Confirm it finds a valid path on the real Udupi graph. Then simulate a closure on the path and confirm it reroutes.

**Do not touch**: LLM, inventory, visualisation.

---

### Person 4 — Simulation Engine (`simulation_engine.py`)

**Your job**: The orchestrator. You run the clock, generate events, and call everything in order.

**What you build**: `simulation_engine.py`, `scenario_config.json`

**Inputs you read**: `/data/vehicle_states.json` (from Person 3)

**Outputs you write**: `/data/world_state.json`, `/data/closure_events.json`

**Step-by-step**:
1. Hardcode the scenario from Section 8 (the Udupi Cyclone Scenario) in `scenario_config.json`
2. Initialise depot inventories (Section 6 table)
3. Main loop from T=0 to T=35:
   a. Read event schedule — fire any events due at this timestep
   b. Update `world_state.json` with current demand + inventory + vehicle statuses
   c. Call `command_llm.py` as a subprocess (or import its main function)
   d. Call `graph_builder.py` to apply any closures
   e. Call `vehicle_agent.py` to advance all vehicles
   f. Read `vehicle_states.json` — update depot inventories for arrivals
   g. Update metrics
   h. Sleep 0.5s (for animation effect)
4. Also build a **lateral transfer check**: after each LLM dispatch, if any depot drops below threshold on any commodity, add a transfer request to the next world_state

**Key rule**: You are the only module that modifies depot inventory counts. No other module changes inventory directly.

**Test without other modules**: Run the loop without calling other modules. Just print `world_state.json` at each timestep and confirm events fire correctly.

**Do not touch**: LLM prompts, routing, graph, visualisation.

---

### Person 5 — Visualiser (`visualiser.py`)

**Your job**: Make it look real. Animate the vehicles moving on the actual Udupi map.

**What you build**: `visualiser.py`, output: `darts_simulation.mp4` or live Matplotlib window

**Inputs you read**: `udupi_graph.graphml`, `/data/vehicle_states.json`, `/data/world_state.json`, `/data/current_graph.json`

**Step-by-step**:
1. Install: `pip install osmnx networkx matplotlib`
2. Load the Udupi graph and plot it as background with `ox.plot_graph`
3. Overlay our 10 nodes as coloured markers:
   - Depots: teal triangles
   - Hospitals: blue circles
   - Staging camps: amber squares
   - Distress points: red stars
4. For each vehicle, draw a dot at its current OSMnx node position:
   - Ambulances: red dot
   - Trucks: orange dot
   - Draw the vehicle's current planned route as a thin dashed line
5. For closed edges: colour them red (weight = 999999 in current_graph.json)
6. For each depot node: draw a small bar chart showing inventory levels (4 bars: MRE, medicine, comms, ambulances)
7. Top-right corner: print the latest LLM explanation from `explanation_log.txt`
8. Use `matplotlib.animation.FuncAnimation` to animate — each frame = one timestep
9. Save with `anim.save("darts_simulation.mp4", writer="ffmpeg")`

**Colour scheme**:
- Green edges: passable roads
- Red edges: closed roads
- Amber: vehicles en route
- Teal: depots
- Coral/red: distress nodes

**Test without other modules**: Load `udupi_graph.graphml` and just render the static map with all 10 nodes placed. Confirm coordinates land on the right streets.

**Do not touch**: LLM, routing logic, inventory.

---

### Person 6 — Paper + Metrics (`metrics.py` + IEEE report)

**Your job**: Write the academic paper and compute the evaluation results.

**What you build**: `metrics.py`, IEEE LaTeX report

**Inputs you read**: `/data/final_metrics.json`, `/data/explanation_log.txt`

**Step-by-step for metrics.py**:
1. After simulation completes, read all logs
2. Compute and print:
   - Total simulation time steps to resolve all demand
   - Number of LLM dispatch calls made
   - Number of lateral transfers triggered
   - Number of vehicle reroutings due to road closures
   - Number of commodity shortfall events (a node needed X but received Y < X)
   - Average delivery latency per node type (distress vs hospital vs staging)
3. Compare against **Baseline A** (static nearest-depot assignment, no replanning) and **Baseline B** (FIFO dispatch, no LLM)
4. Write results tables to `results_table.csv`

**For the paper (IEEE LaTeX, first draft — due tomorrow)**:

Write the following sections. Use the reference paper provided (Path Planning Algorithms for Autonomous Robot Navigation, IEEE AISTS 2025) as your formatting template.

**Abstract** (~150 words):
Summarise: the problem (multi-depot humanitarian logistics in dynamic disaster environments), the approach (two-tier LLM + vehicle routing agent system on real OSMnx graph), key results (X% improvement in delivery latency over static baseline, Y lateral transfers triggered), and conclusion (system mirrors real NDRF ICS structure, first explainable real-time lateral transshipment system in literature).

**Introduction** (~400 words):
- Hook: 2018 Kerala floods, 2023 Udupi coastal flooding — coordination failures cost lives
- Problem statement: static VRP cannot handle dynamic demand, road closures, multi-depot stock imbalances simultaneously
- Research gap: no existing system models real-time lateral transshipment + explainable dispatch + real road networks together
- Contributions: list the 3 novelty claims from Section 2
- Paper structure: "Section II reviews literature..."

**Literature Review** (~600 words, 8–10 references):
Cover these topics with citations (find on Google Scholar):
1. Classical VRP and CVRP (Dantzig & Ramser 1959; Toth & Vigo 2002)
2. Humanitarian logistics VRP (Balcik et al. 2008; Özdamar et al. 2004)
3. Dynamic VRP with road uncertainty (Pillac et al. 2013)
4. Multi-depot VRP (Cordeau et al. 1997)
5. Lateral transshipment in disaster relief (Shao et al. 2024 — Wiley distributional robustness paper)
6. LLM agents for combinatorial optimisation (Liu et al. 2024 — VRPAgent)
7. A* pathfinding in dynamic environments (Hart, Nilsson & Raphael 1968)
8. Indian disaster management systems (NDRF operational framework — NDMA.gov.in)

**Methodology (partial)** (~400 words):
- Problem formulation: define the MDHFCVRP-D (Multi-Depot Heterogeneous Fleet Capacitated VRP with Duality) formally
- System components: describe the two-tier architecture using the diagram in Section 4
- Node taxonomy: Table with 4 node types and their attributes
- Commodity model: Table from Section 6
- State that full methodology (LLM prompt engineering, A* heuristic, baseline comparisons) will be in final draft

---

## 12. Running the Project

### Installation

```bash
pip install osmnx networkx groq matplotlib numpy
```

### Environment variables

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
TOMTOM_API_KEY=your_key_here
```

### File structure

```
darts/
├── command_llm.py          # Person 1
├── graph_builder.py        # Person 2
├── vehicle_agent.py        # Person 3
├── simulation_engine.py    # Person 4
├── visualiser.py           # Person 5
├── metrics.py              # Person 6
├── scenario_config.json    # Person 4 defines this
├── udupi_graph.graphml     # Person 2 generates this (run once)
├── .env                    # API keys (do not commit)
├── data/
│   ├── world_state.json
│   ├── dispatch_orders.json
│   ├── closure_events.json
│   ├── vehicle_states.json
│   ├── current_graph.json
│   └── explanation_log.txt
└── outputs/
    ├── darts_simulation.mp4
    └── results_table.csv
```

### Run order

```bash
# Step 1: Generate the graph (Person 2, run ONCE)
python graph_builder.py --init

# Step 2: Run full simulation
python simulation_engine.py

# Step 3: Generate metrics report
python metrics.py

# Step 4: If you want to replay the animation separately
python visualiser.py --replay
```

### Integration testing without all modules ready

Each module can be run standalone using test fixtures. Each person should create a `test_<module>.py` with hardcoded sample inputs.

---

## 13. Paper Sections (First Draft)

Summary of what must be ready **by 23rd March 2026**:

| Section | Who writes it | Status |
|---|---|---|
| Abstract | Person 6 | Due tomorrow |
| Introduction | Person 6 | Due tomorrow |
| Literature Review | Person 6 | Due tomorrow |
| Methodology (problem formulation + node taxonomy + system architecture) | Person 6 + Person 1 | Partial due tomorrow |
| Methodology (A* routing, LLM prompt design, lateral transfer logic) | Person 3 + Person 1 | Final draft |
| Results | Person 6 using metrics.py | Final draft |
| Discussion + Conclusion | Person 6 | Final draft |

### IEEE formatting reminders
- Template: IEEE conference Word template (A4) or Overleaf IEEE template
- All figures: minimum 300 DPI
- References: IEEE citation style [1], [2], ...
- Similarity index: must be below 15% (final draft)
- Length: 4–6 pages typical for this conference format

---

*Document version: 1.0 | Project: CSS 2203 IAI | System: DARTS*
*Team: 6 members | Submission: 23 March 2026 (first draft), 8 April 2026 (final)*
