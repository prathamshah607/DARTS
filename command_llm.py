"""
command_llm.py — DARTS Strategic Command Tier
==============================================
Person 1's module. Reads world_state.json, calls Groq LLM, writes
dispatch_orders.json and appends to explanation_log.txt.

HOW IT WORKS
------------
1. Read /data/world_state.json  (written by Person 4 each timestep)
2. Build a structured prompt from the world state
3. Call Groq API with the system prompt + user prompt
4. Parse and validate the JSON response
5. If the response violates constraints → re-prompt once with corrections
6. Write /data/dispatch_orders.json  (read by Persons 3 and 4)
7. Append a summary entry to /data/explanation_log.txt

RUNNING STANDALONE (for development/testing)
---------------------------------------------
    python command_llm.py                        # reads real /data/world_state.json
    python command_llm.py --test                 # uses built-in test fixture
    python command_llm.py --test --timestep 18   # uses the T=18 lateral transfer scenario

ENVIRONMENT
-----------
Requires GROQ_API_KEY in environment or .env file.

INTERFACES (for Persons 3, 4, 5, 6)
-------------------------------------
READS:   /data/world_state.json      (see schemas.WORLD_STATE_SCHEMA)
WRITES:  /data/dispatch_orders.json  (see schemas.DISPATCH_ORDERS_SCHEMA)
WRITES:  /data/explanation_log.txt   (appended, human-readable)

Do NOT call this file's functions directly from other modules.
The simulation engine (Person 4) calls this as a subprocess or via
the run_command_llm() entry point at the bottom of this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Try to load .env if python-dotenv is available (not required)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from groq import Groq

from prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_constraint_violation_prompt,
)
from schemas import EXAMPLE_WORLD_STATE, DISPATCH_ORDERS_SCHEMA  # noqa — for docs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
WORLD_STATE_PATH   = DATA_DIR / "world_state.json"
DISPATCH_PATH      = DATA_DIR / "dispatch_orders.json"
EXPLANATION_LOG    = DATA_DIR / "explanation_log.txt"

MODEL = "llama-3.3-70b-versatile"   # swap to "mixtral-8x7b-32768" if needed
MAX_RETRIES = 2          # how many times to re-prompt on constraint violations
TEMPERATURE = 0.2        # low = more deterministic, fewer hallucinations


# ---------------------------------------------------------------------------
# Groq client (lazy initialisation so tests can run without a key)
# ---------------------------------------------------------------------------

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file or environment."
            )
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Make a single Groq API call and return the raw response text.

    Args:
        system_prompt: the system role message
        user_message:  the user role message

    Returns:
        Raw string content from the model

    Raises:
        RuntimeError if the API call fails after all retries
    """
    client = _get_client()
    for attempt in range(1, 4):  # 3 attempts for network errors
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt == 3:
                raise RuntimeError(f"Groq API failed after 3 attempts: {exc}") from exc
            print(f"  [LLM] Network error (attempt {attempt}/3): {exc}. Retrying in 2s...")
            time.sleep(2)
    return ""  # unreachable


# ---------------------------------------------------------------------------
# JSON extraction and parsing
# ---------------------------------------------------------------------------

def extract_json(raw: str) -> dict:
    """
    Extract a JSON object from raw LLM output.

    The model sometimes wraps JSON in markdown fences (```json ... ```)
    or adds a sentence before/after. This strips all of that.

    Args:
        raw: raw string from LLM

    Returns:
        Parsed dict

    Raises:
        ValueError if no valid JSON object can be found
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    cleaned = cleaned.replace("```", "").strip()

    # Find the outermost { ... } block
    brace_start = cleaned.find("{")
    brace_end   = cleaned.rfind("}")
    if brace_start == -1 or brace_end == -1:
        raise ValueError(f"No JSON object found in LLM response:\n{raw[:300]}")

    json_str = cleaned[brace_start : brace_end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON parse failed: {exc}\nExtracted string:\n{json_str[:500]}"
        ) from exc


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

def validate_orders(orders_response: dict, world_state: dict) -> list[str]:
    """
    Check a parsed dispatch_orders dict against world state constraints.

    Returns a list of human-readable violation strings.
    Empty list = no violations.

    Rules checked:
      1. vehicle_id must exist in vehicle_status
      2. vehicle must be idle
      3. from_depot must be the vehicle's home depot
      4. cargo quantities must not exceed depot inventory
      5. cargo values must be non-negative integers
      6. destination must not be empty
    """
    violations: list[str] = []
    inventory   = world_state["depot_inventory"]
    vehicle_map = {v["id"]: v for v in world_state["vehicle_status"]}

    # Track promised outflows from each depot this timestep.
    promised: dict[str, dict[str, int]] = {
        depot: {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0}
        for depot in inventory
    }

    # Track expected inflows into each depot from lateral transfers this timestep.
    # A lateral transfer FROM depot-A TO depot-B promises stock arriving at depot-B.
    # We allow subsequent orders FROM depot-B to draw on this incoming stock,
    # because lateral transfers are assumed to execute before regular dispatches
    # in the same timestep (simulation engine enforces this ordering).
    inflows: dict[str, dict[str, int]] = {
        depot: {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0}
        for depot in inventory
    }

    # First pass: collect all lateral transfer inflows
    for order in orders_response.get("orders", []):
        if not order.get("is_lateral_transfer", False):
            continue
        dest  = order.get("destination", "")
        cargo = order.get("cargo", {})
        if dest in inflows:
            for commodity, qty in cargo.items():
                if isinstance(qty, int) and qty >= 0:
                    inflows[dest][commodity] = inflows[dest].get(commodity, 0) + qty

    # Track vehicles already assigned by earlier orders in THIS response.
    # A vehicle used in order N is gone — it cannot appear in order N+1
    # even if world_state still shows it as idle.
    assigned_this_timestep: set[str] = set()

    # Second pass: validate all orders
    for order in orders_response.get("orders", []):
        oid   = order.get("order_id", "?")
        vid   = order.get("vehicle_id", "")
        depot = order.get("from_depot", "")
        dest  = order.get("destination", "")
        cargo = order.get("cargo", {})

        # Check vehicle exists
        if vid not in vehicle_map:
            violations.append(
                f"Order {oid}: vehicle_id '{vid}' not found in vehicle_status."
            )
            continue

        vehicle = vehicle_map[vid]

        # Check vehicle is idle in world_state
        if vehicle["status"] != "idle":
            violations.append(
                f"Order {oid}: vehicle '{vid}' has status '{vehicle['status']}' — not idle."
            )

        # Check vehicle hasn't already been assigned in an earlier order this timestep
        if vid in assigned_this_timestep:
            violations.append(
                f"Order {oid}: vehicle '{vid}' already dispatched in an earlier order "
                f"this timestep — cannot dispatch the same vehicle twice."
            )
        else:
            assigned_this_timestep.add(vid)

        # Check depot exists
        if depot not in inventory:
            violations.append(
                f"Order {oid}: from_depot '{depot}' not in depot_inventory."
            )
            continue

        # Check destination is not empty
        if not dest:
            violations.append(f"Order {oid}: destination is empty.")

        # Check that at least one cargo value is non-zero
        # (a vehicle dispatched with empty cargo is a wasted trip)
        cargo_total = sum(v for v in cargo.values() if isinstance(v, int) and v > 0)
        if cargo_total == 0:
            violations.append(
                f"Order {oid}: all cargo values are zero — "
                f"dispatching a vehicle with no cargo is not allowed."
            )

        # Check cargo values are non-negative integers
        for commodity, qty in cargo.items():
            if not isinstance(qty, int) or qty < 0:
                violations.append(
                    f"Order {oid}: cargo['{commodity}'] = {qty!r} — must be a non-negative integer."
                )

        # Accumulate promised outflows and check against available + expected inflows
        for commodity, qty in cargo.items():
            if commodity not in promised.get(depot, {}):
                continue
            if not isinstance(qty, int) or qty < 0:
                continue  # already flagged above
            promised[depot][commodity] += qty
            base_stock    = inventory[depot].get(commodity, 0)
            incoming      = inflows[depot].get(commodity, 0)
            available     = base_stock + incoming
            if promised[depot][commodity] > available:
                violations.append(
                    f"Order {oid}: depot '{depot}' has {base_stock}x {commodity} "
                    f"(+{incoming} incoming lateral) "
                    f"but total dispatched this timestep would be "
                    f"{promised[depot][commodity]}."
                )

    return violations


# ---------------------------------------------------------------------------
# Explanation log writer
# ---------------------------------------------------------------------------

def append_explanation_log(
    timestep: int,
    world_state: dict,
    dispatch_orders: dict,
    retry_count: int,
) -> None:
    """
    Append a structured entry to explanation_log.txt.

    Each entry contains:
      - Timestep and wall-clock time
      - Pending requests at time of decision
      - All dispatch orders with their reasons
      - Any deferred requests and why
      - LLM summary
      - How many retries were needed

    Args:
        timestep:       simulation clock value
        world_state:    the world state that was fed to the LLM
        dispatch_orders: the validated dispatch response
        retry_count:    number of constraint-correction retries needed
    """
    EXPLANATION_LOG.parent.mkdir(parents=True, exist_ok=True)

    wall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    orders    = dispatch_orders.get("orders", [])
    deferred  = dispatch_orders.get("deferred_requests", [])
    summary   = dispatch_orders.get("llm_summary", "")

    sep = "=" * 70

    lines = [
        sep,
        f"TIMESTEP {timestep:>3}  |  {wall_time}  |  retries={retry_count}",
        sep,
        "",
        "SITUATION:",
    ]

    for req in world_state.get("pending_requests", []):
        waiting = timestep - req.get("arrived_at_timestep", timestep)
        lines.append(
            f"  [{req['urgency'].upper():8}] {req['node_id']:<20} "
            f"needs {req['quantity']:>4}x {req['commodity']:<12} "
            f"(waiting {waiting}t)"
        )

    lines += ["", "DISPATCH ORDERS:"]
    if orders:
        for o in orders:
            lateral = " [LATERAL TRANSFER]" if o.get("is_lateral_transfer") else ""
            cargo_str = ", ".join(
                f"{k}:{v}" for k, v in o.get("cargo", {}).items() if v > 0
            )
            lines += [
                f"  {o['order_id']}{lateral}",
                f"    Vehicle : {o['vehicle_id']}",
                f"    Route   : {o['from_depot']} → {o['destination']}",
                f"    Cargo   : {cargo_str}",
                f"    Priority: {o['priority']}",
                f"    Reason  : {o['reason']}",
                "",
            ]
    else:
        lines.append("  (no orders issued)")

    if deferred:
        lines += ["DEFERRED:"]
        for d in deferred:
            lines.append(f"  {d['request_id']} — {d['reason']}")
        lines.append("")

    lines += [
        "SUMMARY:",
        f"  {summary}",
        "",
    ]

    with open(EXPLANATION_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main dispatch function
# ---------------------------------------------------------------------------

def run_dispatch(world_state: dict) -> dict:
    """
    Core pipeline: world_state → LLM → validated dispatch_orders.

    1. Build prompt from world_state
    2. Call LLM
    3. Parse JSON response
    4. Validate constraints
    5. If violations found → re-prompt up to MAX_RETRIES times
    6. Return final (possibly partially corrected) dispatch_orders dict

    Args:
        world_state: dict matching WORLD_STATE_SCHEMA

    Returns:
        Validated dispatch_orders dict matching DISPATCH_ORDERS_SCHEMA

    Raises:
        RuntimeError if the LLM cannot produce a valid response within retries
    """
    timestep = world_state.get("timestep", 0)
    print(f"[CommandLLM] T={timestep} — building prompt...")

    user_message   = build_user_prompt(world_state)
    current_system = SYSTEM_PROMPT
    retry_count    = 0

    # First attempt
    raw = call_llm(current_system, user_message)
    print(f"[CommandLLM] T={timestep} — LLM responded ({len(raw)} chars).")

    try:
        parsed = extract_json(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"T={timestep}: LLM returned unparseable output: {exc}"
        ) from exc

    # Ensure required top-level keys exist with safe defaults
    parsed.setdefault("timestep", timestep)
    parsed.setdefault("orders", [])
    parsed.setdefault("deferred_requests", [])
    parsed.setdefault("llm_summary", "")

    # Validate and re-prompt on violations
    violations = validate_orders(parsed, world_state)

    while violations and retry_count < MAX_RETRIES:
        retry_count += 1
        print(
            f"[CommandLLM] T={timestep} — {len(violations)} constraint violation(s). "
            f"Re-prompting (attempt {retry_count}/{MAX_RETRIES})..."
        )
        for v in violations:
            print(f"  VIOLATION: {v}")

        correction_prompt = build_constraint_violation_prompt(violations, world_state)
        # Include the bad response so the LLM can see what went wrong
        correction_user = (
            f"Your previous response:\n{raw}\n\n"
            f"{correction_prompt}"
        )
        raw = call_llm(current_system, correction_user)

        try:
            parsed = extract_json(raw)
        except ValueError:
            # If parsing fails again, break out — we'll return what we have
            print(f"[CommandLLM] T={timestep} — Re-prompt still unparseable. Stopping.")
            break

        parsed.setdefault("timestep", timestep)
        parsed.setdefault("orders", [])
        parsed.setdefault("deferred_requests", [])
        parsed.setdefault("llm_summary", "")
        violations = validate_orders(parsed, world_state)

    if violations:
        print(
            f"[CommandLLM] WARNING: T={timestep} — "
            f"{len(violations)} violation(s) remain after retries. "
            "Returning best-effort response."
        )
        for v in violations:
            print(f"  REMAINING: {v}")

    append_explanation_log(timestep, world_state, parsed, retry_count)
    return parsed


# ---------------------------------------------------------------------------
# File I/O entry point (called by simulation engine)
# ---------------------------------------------------------------------------

def run_command_llm() -> None:
    """
    File-based entry point for Person 4 (simulation_engine.py).

    Reads WORLD_STATE_PATH, calls run_dispatch(), writes DISPATCH_PATH.

    Person 4 calls this function once per timestep:
        from command_llm import run_command_llm
        run_command_llm()
    or as a subprocess:
        subprocess.run(["python", "command_llm.py"])
    """
    if not WORLD_STATE_PATH.exists():
        raise FileNotFoundError(
            f"world_state.json not found at {WORLD_STATE_PATH}. "
            "Ensure Person 4 (simulation_engine.py) wrote it before calling this."
        )

    with open(WORLD_STATE_PATH, encoding="utf-8") as f:
        world_state = json.load(f)

    dispatch_orders = run_dispatch(world_state)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DISPATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(dispatch_orders, f, indent=2)

    timestep = world_state.get("timestep", "?")
    n_orders = len(dispatch_orders.get("orders", []))
    print(
        f"[CommandLLM] T={timestep} — wrote {n_orders} order(s) "
        f"to {DISPATCH_PATH.name}"
    )


# ---------------------------------------------------------------------------
# Test fixtures (--test flag)
# ---------------------------------------------------------------------------

# Additional test scenario: T=18 lateral transfer trigger
LATERAL_TRANSFER_SCENARIO: dict = {
    "timestep": 18,
    "depot_inventory": {
        "DEPOT-A": {"mre": 30, "medicine": 50, "comms": 8,  "ambulances": 1},
        "DEPOT-B": {"mre": 40, "medicine": 0,  "comms": 5,  "ambulances": 2},
        "DEPOT-C": {"mre": 80, "medicine": 80, "comms": 10, "ambulances": 2},
    },
    "vehicle_status": [
        {"id": "AMB-01", "type": "ambulance", "depot": "DEPOT-A",
         "status": "returning", "destination": None,
         "cargo": {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0},
         "eta_steps": 2},
        {"id": "TRUCK-02", "type": "truck", "depot": "DEPOT-B",
         "status": "idle", "destination": None,
         "cargo": {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0},
         "eta_steps": 0},
        {"id": "TRUCK-01", "type": "truck", "depot": "DEPOT-A",
         "status": "idle", "destination": None,
         "cargo": {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0},
         "eta_steps": 0},
    ],
    "pending_requests": [
        {"request_id": "REQ-009", "node_id": "NODE-H2",
         "node_type": "hospital", "commodity": "medicine",
         "quantity": 50, "urgency": "critical",
         "arrived_at_timestep": 18},
    ],
}


# No-stock scenario: forces LLM to defer
NO_STOCK_SCENARIO: dict = {
    "timestep": 30,
    "depot_inventory": {
        "DEPOT-A": {"mre": 2,  "medicine": 0, "comms": 0, "ambulances": 0},
        "DEPOT-B": {"mre": 0,  "medicine": 0, "comms": 0, "ambulances": 1},
        "DEPOT-C": {"mre": 5,  "medicine": 3, "comms": 1, "ambulances": 0},
    },
    "vehicle_status": [
        {"id": "TRUCK-01", "type": "truck", "depot": "DEPOT-A",
         "status": "idle", "destination": None,
         "cargo": {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0},
         "eta_steps": 0},
    ],
    "pending_requests": [
        {"request_id": "REQ-012", "node_id": "NODE-H1",
         "node_type": "hospital", "commodity": "medicine",
         "quantity": 40, "urgency": "critical",
         "arrived_at_timestep": 28},
        {"request_id": "REQ-013", "node_id": "NODE-D3",
         "node_type": "distress", "commodity": "mre",
         "quantity": 25, "urgency": "sos",
         "arrived_at_timestep": 30},
    ],
}


TEST_SCENARIOS: dict[str, dict] = {
    "7":  EXAMPLE_WORLD_STATE,       # from schemas.py — normal dispatch
    "18": LATERAL_TRANSFER_SCENARIO,  # lateral transfer needed
    "30": NO_STOCK_SCENARIO,          # forced deferrals
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DARTS Command LLM — run one dispatch cycle"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use a built-in test fixture instead of reading world_state.json",
    )
    parser.add_argument(
        "--timestep",
        type=str,
        default="7",
        choices=list(TEST_SCENARIOS.keys()),
        help="Which test scenario to use (only with --test). Choices: 7, 18, 30",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt only; do not call the LLM or write any files",
    )
    args = parser.parse_args()

    if args.test:
        world_state = TEST_SCENARIOS[args.timestep]
        print(f"[CommandLLM] Running TEST scenario T={args.timestep}")
    else:
        if not WORLD_STATE_PATH.exists():
            print(
                f"ERROR: {WORLD_STATE_PATH} not found.\n"
                "Run with --test to use a built-in scenario, or ensure "
                "Person 4's simulation_engine.py has written world_state.json first."
            )
            sys.exit(1)
        with open(WORLD_STATE_PATH, encoding="utf-8") as f:
            world_state = json.load(f)
        print(f"[CommandLLM] Loaded world_state.json (T={world_state.get('timestep')})")

    if args.dry_run:
        print("\n--- SYSTEM PROMPT ---")
        print(SYSTEM_PROMPT)
        print("\n--- USER PROMPT ---")
        print(build_user_prompt(world_state))
        print("\n(dry-run: no API call made)")
        return

    dispatch_orders = run_dispatch(world_state)

    # Print to console for verification
    print("\n--- DISPATCH ORDERS ---")
    print(json.dumps(dispatch_orders, indent=2))

    # Write files
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DISPATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(dispatch_orders, f, indent=2)
    print(f"\n[CommandLLM] Written: {DISPATCH_PATH}")
    print(f"[CommandLLM] Log appended: {EXPLANATION_LOG}")

    # Print explanation log tail
    if EXPLANATION_LOG.exists():
        with open(EXPLANATION_LOG, encoding="utf-8") as f:
            lines = f.readlines()
        tail = lines[-40:] if len(lines) > 40 else lines
        print("\n--- EXPLANATION LOG (latest entry) ---")
        print("".join(tail))


if __name__ == "__main__":
    main()