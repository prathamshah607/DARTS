"""
schemas.py — DARTS data contracts
==================================
Single source of truth for every JSON structure that command_llm.py
reads or writes. Persons 2–6: do NOT redefine these structures in your
own modules. Import from here, or copy-paste the docstrings as your
reference.

All schemas are documented as plain dicts with inline comments.
"""

# ---------------------------------------------------------------------------
# INBOUND — what command_llm.py reads
# ---------------------------------------------------------------------------

WORLD_STATE_SCHEMA = {
    # Written by: simulation_engine.py  (Person 4)
    # Read by:    command_llm.py        (Person 1)
    #
    "timestep": int,          # simulation clock tick, starts at 0
    "depot_inventory": {
        # key = depot ID string, value = commodity counts (all non-negative ints)
        # DEPOT-A, DEPOT-B, DEPOT-C
        "<depot_id>": {
            "mre":         int,   # MRE meal packs
            "medicine":    int,   # Emergency medicine kits
            "comms":       int,   # Comms devices (walkies, radios)
            "ambulances":  int,   # Ambulance count (vehicle-commodity dual)
        }
    },
    "vehicle_status": [
        # One entry per vehicle currently tracked. Ensure vehicle IDs referenced
        # in dispatch_orders also appear here with status="idle".
        {
            "id":          str,   # e.g. "AMB-01", "TRUCK-02"
            "type":        str,   # "ambulance" or "truck"
            "depot":       str,   # home depot, e.g. "DEPOT-A"
            "status":      str,   # "idle" | "en_route" | "arrived" | "returning"
            "destination": str,   # node_id if en_route, else null
            "cargo":       dict,  # {"mre": 0, "medicine": 40, ...} currently loaded
            "eta_steps":   int,   # estimated timesteps until arrival (0 if idle)
        }
    ],
    "pending_requests": [
        # Distress calls not yet fulfilled. Cleared when dispatch_order issued.
        {
            "request_id":  str,   # unique ID, e.g. "REQ-007"
            "node_id":     str,   # destination, e.g. "NODE-H1"
            "node_type":   str,   # "hospital" | "staging" | "distress" | "depot"
            "commodity":   str,   # "mre" | "medicine" | "comms" | "ambulances"
            "quantity":    int,   # units needed
            "urgency":     str,   # "sos" | "critical" | "high" | "medium" | "low"
            "arrived_at_timestep": int,  # when this request was first registered
        }
    ]
}

EXAMPLE_WORLD_STATE = {
    "timestep": 7,
    "depot_inventory": {
        "DEPOT-A": {"mre": 45, "medicine": 60, "comms": 10, "ambulances": 2},
        "DEPOT-B": {"mre": 60, "medicine": 0,  "comms": 8,  "ambulances": 2},
        "DEPOT-C": {"mre": 80, "medicine": 80, "comms": 10, "ambulances": 2},
    },
    "vehicle_status": [
        {"id": "AMB-01", "type": "ambulance", "depot": "DEPOT-A",
         "status": "en_route", "destination": "NODE-D1",
         "cargo": {"mre": 20, "medicine": 0, "comms": 0, "ambulances": 0},
         "eta_steps": 4},
        {"id": "TRUCK-01", "type": "truck", "depot": "DEPOT-A",
         "status": "idle", "destination": None,
         "cargo": {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0},
         "eta_steps": 0},
        {"id": "AMB-02", "type": "ambulance", "depot": "DEPOT-B",
         "status": "idle", "destination": None,
         "cargo": {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0},
         "eta_steps": 0},
    ],
    "pending_requests": [
        {"request_id": "REQ-003", "node_id": "NODE-H1", "node_type": "hospital",
         "commodity": "medicine", "quantity": 40, "urgency": "critical",
         "arrived_at_timestep": 7},
        {"request_id": "REQ-004", "node_id": "NODE-D2", "node_type": "distress",
         "commodity": "mre", "quantity": 15, "urgency": "sos",
         "arrived_at_timestep": 6},
    ]
}


# ---------------------------------------------------------------------------
# OUTBOUND — what command_llm.py writes
# ---------------------------------------------------------------------------

DISPATCH_ORDERS_SCHEMA = {
    # Written by: command_llm.py  (Person 1)
    # Read by:    vehicle_agent.py (Person 3)
    #             simulation_engine.py (Person 4) for inventory deduction
    #
    "timestep": int,       # same timestep as the world_state this was based on
    "orders": [
        {
            "order_id":     str,   # unique, e.g. "ORD-T007-001"
            "vehicle_id":   str,   # which vehicle to dispatch, e.g. "AMB-02"
            "from_depot":   str,   # source depot, e.g. "DEPOT-A"
            "destination":  str,   # target node, e.g. "NODE-H1"
            "cargo": {
                "mre":        int,   # 0 if not carrying this commodity
                "medicine":   int,
                "comms":      int,
                "ambulances": int,  # 1 if this IS the ambulance being dispatched
            },
            "priority":     str,   # mirrors urgency: "sos"|"critical"|"high"|"medium"|"low"
            "is_lateral_transfer": bool,  # True if depot→depot (not depot→affected node)
            "reason":       str,   # plain English rationale from LLM, 1–3 sentences
        }
    ],
    "deferred_requests": [
        # Requests the LLM could not fulfil this timestep (no stock / no vehicle)
        {
            "request_id": str,
            "reason":     str,   # why it was deferred
        }
    ],
    "llm_summary": str,    # 2–4 sentence overview of this timestep's decisions
}

EXAMPLE_DISPATCH_ORDERS = {
    "timestep": 7,
    "orders": [
        {
            "order_id": "ORD-T007-001",
            "vehicle_id": "TRUCK-01",
            "from_depot": "DEPOT-A",
            "destination": "NODE-H1",
            "cargo": {"mre": 0, "medicine": 40, "comms": 0, "ambulances": 0},
            "priority": "critical",
            "is_lateral_transfer": False,
            "reason": (
                "KMC Hospital flagged critical medicine shortage of 40 units. "
                "DEPOT-A holds 60 units — sufficient after this dispatch. "
                "TRUCK-01 is the only idle vehicle at DEPOT-A."
            ),
        },
        {
            "order_id": "ORD-T007-002",
            "vehicle_id": "AMB-02",
            "from_depot": "DEPOT-B",
            "destination": "NODE-D2",
            "cargo": {"mre": 15, "medicine": 0, "comms": 0, "ambulances": 1},
            "priority": "sos",
            "is_lateral_transfer": False,
            "reason": (
                "SOS signal from Gangolli coastal hamlet, 2 people isolated. "
                "AMB-02 at DEPOT-B is nearest idle ambulance at approximately 5km. "
                "Dispatching with 15 MRE from DEPOT-B stock."
            ),
        },
    ],
    "deferred_requests": [],
    "llm_summary": (
        "Two orders issued at T=7. Critical medicine delivery to KMC Hospital "
        "assigned to TRUCK-01 from DEPOT-A. SOS response to Gangolli hamlet "
        "assigned to AMB-02 from DEPOT-B with MRE and ambulance presence. "
        "No lateral transfers required this timestep."
    ),
}
