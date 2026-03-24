import json
import time
import os
import copy

from command_llm import run_command_llm

DATA_PATH = "data/"
SCENARIO_FILE = "scenario_config.json"

os.makedirs(DATA_PATH, exist_ok=True)

# ----------------------------
# Load Scenario
# ----------------------------
with open(SCENARIO_FILE) as f:
    scenario = json.load(f)

events = scenario["events"]
start = scenario["simulation"]["start"]
end = scenario["simulation"]["end"]

# ----------------------------
# Initial Inventory
# ----------------------------
inventory = {
    "DEPOT-A": {"mre": 120, "medicine": 60, "comms": 15, "ambulances": 3},
    "DEPOT-B": {"mre": 60, "medicine": 15, "comms": 8, "ambulances": 2},
    "DEPOT-C": {"mre": 80, "medicine": 80, "comms": 10, "ambulances": 2}
}

pending_requests = []
request_counter = 0

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
            data = json.load(f)

            # FIX: extract correct format
            if isinstance(data, dict) and "vehicles" in data:
                return data["vehicles"]

            return data
    except:
        return []

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_node_type(node_id):
    if node_id.startswith("NODE-H"):
        return "hospital"
    elif node_id.startswith("NODE-D"):
        return "distress"
    elif node_id.startswith("NODE-S"):
        return "staging"
    elif node_id.startswith("DEPOT"):
        return "depot"
    return "unknown"

# ----------------------------
# Lateral Transfer
# ----------------------------
def check_lateral_transfer(inventory, pending_requests, timestep):
    global request_counter

    for depot, stock in inventory.items():
        for commodity, value in stock.items():

            if value < THRESHOLDS[commodity]:

                already_requested = any(
                    r["node_id"] == depot and r["commodity"] == commodity
                    for r in pending_requests
                )

                if not already_requested:
                    request_counter += 1

                    print(f"[LATERAL] {depot} low on {commodity}")

                    pending_requests.append({
                        "request_id": f"REQ-{request_counter:03}",
                        "node_id": depot,
                        "node_type": "depot",
                        "commodity": commodity,
                        "quantity": THRESHOLDS[commodity] * 2,
                        "urgency": "high",
                        "arrived_at_timestep": timestep
                    })

# ----------------------------
# MAIN LOOP
# ----------------------------
for timestep in range(start, end + 1):

    print(f"\n=== Timestep {timestep} ===")

    closure_events = []

    # ------------------------
    # (a) Fire Events
    # ------------------------
    for event in events:
        if event["timestep"] == timestep:

            if event["type"] == "request":
                for req in event["requests"]:
                    request_counter += 1

                    pending_requests.append({
                        "request_id": f"REQ-{request_counter:03}",
                        "node_id": event["node_id"],
                        "node_type": get_node_type(event["node_id"]),
                        "commodity": req["commodity"],
                        "quantity": req["quantity"],
                        "urgency": event["urgency"],
                        "arrived_at_timestep": timestep
                    })

            elif event["type"] == "closure":
                closure_events.append(event)

    # ------------------------
    # Write closure events
    # ------------------------
    write_json(DATA_PATH + "closure_events.json", {
        "timestep": timestep,
        "closures": closure_events
    })

    # ------------------------
    # (b) Write world state
    # ------------------------
    world_state = {
        "timestep": timestep,
        "depot_inventory": copy.deepcopy(inventory),
        "vehicle_status": load_vehicle_states(),
        "pending_requests": pending_requests
    }

    write_json(DATA_PATH + "world_state.json", world_state)

    # ------------------------
    # (c) Call Command LLM
    # ------------------------
    dispatch = {"orders": []}

    try:
        run_command_llm()
        with open(DATA_PATH + "dispatch_orders.json") as f:
            dispatch = json.load(f)
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        print("[FALLBACK] Using empty dispatch")

    # ------------------------
    # (d) Apply dispatch → inventory
    # ------------------------
    for order in dispatch.get("orders", []):
        depot = order["from_depot"]
        for k, v in order["cargo"].items():
            inventory[depot][k] -= v

    # ------------------------
    # (e) Remove fulfilled requests
    # ------------------------
    fulfilled_nodes = {o["destination"] for o in dispatch.get("orders", [])}

    pending_requests = [
        r for r in pending_requests
        if r["node_id"] not in fulfilled_nodes
    ]

    # ------------------------
    # (f) Run Vehicle Agent
    # ------------------------
    try:
        os.system("python vehicle_agent.py")
    except Exception as e:
        print(f"[VEHICLE ERROR] {e}")

    # ------------------------
    # (g) Lateral transfer
    # ------------------------
    check_lateral_transfer(inventory, pending_requests, timestep)

    # ------------------------
    # (h) Sleep
    # ------------------------
    time.sleep(0.5)

print("\nSimulation complete.")
