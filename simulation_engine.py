import json
import time
import os
import copy

DATA_PATH = "data/"
SCENARIO_FILE = "scenario_config.json"

# ----------------------------
# Load Scenario
# ----------------------------
with open(SCENARIO_FILE) as f:
    scenario = json.load(f)

events = scenario["events"]
start = scenario["simulation"]["start"]
end = scenario["simulation"]["end"]

# ----------------------------
# Initial Inventory (Section 6)
# ----------------------------
inventory = {
    "DEPOT-A": {"mre": 120, "medicine": 60, "comms": 15, "ambulances": 3},
    "DEPOT-B": {"mre": 60, "medicine": 15, "comms": 8, "ambulances": 2},
    "DEPOT-C": {"mre": 80, "medicine": 80, "comms": 10, "ambulances": 2}
}

# ----------------------------
# State
# ----------------------------
pending_requests = []

# ----------------------------
# Thresholds for lateral transfer
# ----------------------------
THRESHOLDS = {
    "mre": 30,
    "medicine": 10,
    "comms": 5,
    "ambulances": 1
}

# ----------------------------
# Helpers
# ----------------------------
def load_vehicle_states():
    try:
        with open(DATA_PATH + "vehicle_states.json") as f:
            return json.load(f)
    except:
        return []

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ----------------------------
# Lateral Transfer Logic
# ----------------------------
def check_lateral_transfer(inventory, pending_requests, timestep):
    for depot, stock in inventory.items():
        for commodity, value in stock.items():

            if value < THRESHOLDS[commodity]:

                # prevent duplicate requests
                already_requested = any(
                    r.get("node_id") == depot and
                    any(req["commodity"] == commodity for req in r.get("requests", []))
                    for r in pending_requests
                )

                if not already_requested:
                    print(f"[LATERAL] {depot} low on {commodity}")

                    pending_requests.append({
                        "timestep": timestep,
                        "type": "request",
                        "node_id": depot,
                        "requests": [
                            {
                                "commodity": commodity,
                                "quantity": THRESHOLDS[commodity] * 2
                            }
                        ],
                        "urgency": "high"
                    })

# ----------------------------
# MAIN LOOP
# ----------------------------
for timestep in range(start, end + 1):

    print(f"\n=== Timestep {timestep} ===")

    # ------------------------
    # (a) Fire Events
    # ------------------------
    closure_events = []

    for event in events:
        if event["timestep"] == timestep:

            if event["type"] == "request":
                pending_requests.append(event)

            elif event["type"] == "closure":
                closure_events.append(event)

    # Write closure events for graph module
    write_json(DATA_PATH + "closure_events.json", {
        "timestep": timestep,
        "closures": closure_events
    })

    # ------------------------
    # (b) Build world_state.json
    # ------------------------
    world_state = {
        "timestep": timestep,
        "depot_inventory": copy.deepcopy(inventory),
        "pending_requests": pending_requests,
        "vehicle_status": load_vehicle_states()
    }

    write_json(DATA_PATH + "world_state.json", world_state)

    # ------------------------
    # TEST MODE (disable external modules first)
    # ------------------------
    print(json.dumps(world_state, indent=2))

    # ------------------------
    # (c) Call LLM (enable later)
    # ------------------------
    # os.system("python command_llm.py")

    # ------------------------
    # (d) Apply road closures (enable later)
    # ------------------------
    # os.system("python graph_builder.py")

    # ------------------------
    # (e) Move vehicles (enable later)
    # ------------------------
    # os.system("python vehicle_agent.py")

    # ------------------------
    # (f) Update inventory (arrival logic placeholder)
    # ------------------------
    vehicle_states = load_vehicle_states()

    for v in vehicle_states:
        if v.get("status") == "arrived":

            destination = v.get("destination")

            # Remove fulfilled requests
            pending_requests = [
                r for r in pending_requests
                if r["node_id"] != destination
            ]

    # ------------------------
    # (g) Lateral transfer check
    # ------------------------
    check_lateral_transfer(inventory, pending_requests, timestep)

    # ------------------------
    # (h) Sleep
    # ------------------------
    time.sleep(0.5)

print("\nSimulation complete.")
