"""
test_command_llm.py — Unit tests for command_llm.py
====================================================
All tests run WITHOUT a Groq API key by mocking the LLM call.
Run with:  python test_command_llm.py
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from command_llm import extract_json, validate_orders, run_dispatch
from prompts import build_user_prompt, build_constraint_violation_prompt
from schemas import EXAMPLE_WORLD_STATE, EXAMPLE_DISPATCH_ORDERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_world_state(**overrides) -> dict:
    """Return a copy of EXAMPLE_WORLD_STATE with optional field overrides."""
    import copy
    ws = copy.deepcopy(EXAMPLE_WORLD_STATE)
    ws.update(overrides)
    return ws


VALID_LLM_RESPONSE = json.dumps(EXAMPLE_DISPATCH_ORDERS)

RESPONSE_WITH_FENCES = f"```json\n{VALID_LLM_RESPONSE}\n```"

RESPONSE_WITH_PREAMBLE = (
    "Sure! Here are the dispatch orders:\n"
    f"{VALID_LLM_RESPONSE}\n"
    "Hope this helps!"
)


# ---------------------------------------------------------------------------
# Tests: extract_json
# ---------------------------------------------------------------------------

class TestExtractJson(unittest.TestCase):

    def test_plain_json(self):
        result = extract_json(VALID_LLM_RESPONSE)
        self.assertIsInstance(result, dict)
        self.assertIn("orders", result)

    def test_json_with_markdown_fences(self):
        result = extract_json(RESPONSE_WITH_FENCES)
        self.assertIn("orders", result)

    def test_json_with_preamble_and_postamble(self):
        result = extract_json(RESPONSE_WITH_PREAMBLE)
        self.assertIn("orders", result)

    def test_raises_on_empty_string(self):
        with self.assertRaises(ValueError):
            extract_json("")

    def test_raises_on_plain_text(self):
        with self.assertRaises(ValueError):
            extract_json("Sorry, I cannot help with that.")

    def test_raises_on_malformed_json(self):
        with self.assertRaises(ValueError):
            extract_json('{"orders": [}')


# ---------------------------------------------------------------------------
# Tests: validate_orders
# ---------------------------------------------------------------------------

class TestValidateOrders(unittest.TestCase):

    def test_valid_response_no_violations(self):
        violations = validate_orders(EXAMPLE_DISPATCH_ORDERS, EXAMPLE_WORLD_STATE)
        self.assertEqual(violations, [], f"Unexpected violations: {violations}")

    def test_unknown_vehicle_id(self):
        bad = json.loads(VALID_LLM_RESPONSE)
        bad["orders"][0]["vehicle_id"] = "GHOST-99"
        violations = validate_orders(bad, EXAMPLE_WORLD_STATE)
        self.assertTrue(any("GHOST-99" in v for v in violations))

    def test_vehicle_not_idle(self):
        """AMB-01 is en_route in the world state — cannot be dispatched."""
        bad = json.loads(VALID_LLM_RESPONSE)
        bad["orders"][0]["vehicle_id"] = "AMB-01"
        violations = validate_orders(bad, EXAMPLE_WORLD_STATE)
        self.assertTrue(any("not idle" in v for v in violations))

    def test_depot_not_in_inventory(self):
        bad = json.loads(VALID_LLM_RESPONSE)
        bad["orders"][0]["from_depot"] = "DEPOT-Z"
        violations = validate_orders(bad, EXAMPLE_WORLD_STATE)
        self.assertTrue(any("DEPOT-Z" in v for v in violations))

    def test_exceeds_stock(self):
        bad = json.loads(VALID_LLM_RESPONSE)
        # EXAMPLE_WORLD_STATE DEPOT-A has medicine=60; try to dispatch 200
        bad["orders"][0]["cargo"]["medicine"] = 200
        violations = validate_orders(bad, EXAMPLE_WORLD_STATE)
        self.assertTrue(any("medicine" in v for v in violations))

    def test_negative_cargo(self):
        bad = json.loads(VALID_LLM_RESPONSE)
        bad["orders"][0]["cargo"]["mre"] = -5
        violations = validate_orders(bad, EXAMPLE_WORLD_STATE)
        self.assertTrue(any("non-negative" in v for v in violations))

    def test_empty_destination(self):
        bad = json.loads(VALID_LLM_RESPONSE)
        bad["orders"][0]["destination"] = ""
        violations = validate_orders(bad, EXAMPLE_WORLD_STATE)
        self.assertTrue(any("destination" in v for v in violations))

    def test_two_orders_exceed_combined_stock(self):
        """Two orders from DEPOT-A both request medicine — combined exceeds stock."""
        import copy
        two_orders = copy.deepcopy(EXAMPLE_DISPATCH_ORDERS)
        two_orders["orders"] = [
            {
                "order_id": "ORD-T007-X",
                "vehicle_id": "TRUCK-01",
                "from_depot": "DEPOT-A",
                "destination": "NODE-H1",
                "cargo": {"mre": 0, "medicine": 40, "comms": 0, "ambulances": 0},
                "priority": "critical",
                "is_lateral_transfer": False,
                "reason": "test order 1",
            },
            {
                "order_id": "ORD-T007-Y",
                "vehicle_id": "TRUCK-01",  # same vehicle — but testing stock check
                "from_depot": "DEPOT-A",
                "destination": "NODE-H2",
                "cargo": {"mre": 0, "medicine": 40, "comms": 0, "ambulances": 0},
                "priority": "critical",
                "is_lateral_transfer": False,
                "reason": "test order 2",
            },
        ]
        # DEPOT-A medicine=60. Two orders of 40 each = 80 > 60 → violation
        violations = validate_orders(two_orders, EXAMPLE_WORLD_STATE)
        self.assertTrue(any("medicine" in v for v in violations),
                        f"Expected medicine over-dispatch violation, got: {violations}")

    def test_no_orders_is_valid(self):
        empty = {"timestep": 7, "orders": [], "deferred_requests": [], "llm_summary": ""}
        violations = validate_orders(empty, EXAMPLE_WORLD_STATE)
        self.assertEqual(violations, [])

    def test_empty_cargo_order_is_invalid(self):
        """
        T=30 real failure: LLM issued a lateral transfer with all-zero cargo.
        A vehicle dispatched carrying nothing is a wasted trip and must be rejected.
        """
        import copy
        bad = copy.deepcopy(EXAMPLE_DISPATCH_ORDERS)
        bad["orders"] = [{
            "order_id": "ORD-T030-001",
            "vehicle_id": "TRUCK-01",
            "from_depot": "DEPOT-A",
            "destination": "DEPOT-C",
            "cargo": {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0},
            "priority": "high",
            "is_lateral_transfer": True,
            "reason": "Lateral transfer to gather resources.",
        }]
        violations = validate_orders(bad, EXAMPLE_WORLD_STATE)
        self.assertTrue(
            any("zero" in v for v in violations),
            f"Expected empty-cargo violation, got: {violations}"
        )

    def test_double_dispatch_same_vehicle_same_timestep(self):
        """
        Real failure observed at T=7: LLM dispatched AMB-02 twice in the
        same response (orders 001 and 003). This should be caught as a
        violation even though AMB-02 appears idle in world_state.
        """
        import copy
        double = copy.deepcopy(EXAMPLE_DISPATCH_ORDERS)
        # Both orders use AMB-02 (which is idle in world_state)
        double["orders"] = [
            {
                "order_id": "ORD-T007-001",
                "vehicle_id": "AMB-02",
                "from_depot": "DEPOT-B",
                "destination": "NODE-D2",
                "cargo": {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 1},
                "priority": "sos",
                "is_lateral_transfer": False,
                "reason": "First dispatch",
            },
            {
                "order_id": "ORD-T007-002",
                "vehicle_id": "AMB-02",   # same vehicle — should fail
                "from_depot": "DEPOT-B",
                "destination": "NODE-H1",
                "cargo": {"mre": 0, "medicine": 10, "comms": 0, "ambulances": 0},
                "priority": "critical",
                "is_lateral_transfer": False,
                "reason": "Second dispatch of same vehicle",
            },
        ]
        violations = validate_orders(double, EXAMPLE_WORLD_STATE)
        self.assertTrue(
            any("already dispatched" in v for v in violations),
            f"Expected double-dispatch violation, got: {violations}"
        )


# ---------------------------------------------------------------------------
# Tests: build_user_prompt
# ---------------------------------------------------------------------------

class TestBuildUserPrompt(unittest.TestCase):

    def test_contains_timestep(self):
        prompt = build_user_prompt(EXAMPLE_WORLD_STATE)
        self.assertIn("TIMESTEP: 7", prompt)

    def test_contains_all_depots(self):
        prompt = build_user_prompt(EXAMPLE_WORLD_STATE)
        for depot in EXAMPLE_WORLD_STATE["depot_inventory"]:
            self.assertIn(depot, prompt)

    def test_contains_all_requests(self):
        prompt = build_user_prompt(EXAMPLE_WORLD_STATE)
        for req in EXAMPLE_WORLD_STATE["pending_requests"]:
            self.assertIn(req["node_id"], prompt)

    def test_urgency_labels_present(self):
        prompt = build_user_prompt(EXAMPLE_WORLD_STATE)
        self.assertIn("CRITICAL", prompt)  # from NODE-H1 request
        self.assertIn("SOS", prompt)       # from NODE-D2 request

    def test_idle_vehicles_listed(self):
        prompt = build_user_prompt(EXAMPLE_WORLD_STATE)
        # TRUCK-01 is idle at DEPOT-B
        self.assertIn("TRUCK-01", prompt)

    def test_sos_before_critical_ordering(self):
        """SOS requests should appear before critical in the sorted prompt."""
        prompt = build_user_prompt(EXAMPLE_WORLD_STATE)
        sos_pos      = prompt.find("SOS")
        critical_pos = prompt.find("CRITICAL")
        self.assertLess(sos_pos, critical_pos,
                        "SOS requests should appear before CRITICAL in prompt")

    def test_empty_requests(self):
        ws = make_world_state(pending_requests=[])
        prompt = build_user_prompt(ws)
        self.assertIn("None.", prompt)


# ---------------------------------------------------------------------------
# Tests: run_dispatch (LLM mocked)
# ---------------------------------------------------------------------------

class TestRunDispatch(unittest.TestCase):

    @patch("command_llm.call_llm", return_value=VALID_LLM_RESPONSE)
    def test_valid_response_passes_through(self, _mock):
        result = run_dispatch(EXAMPLE_WORLD_STATE)
        self.assertIn("orders", result)
        self.assertEqual(len(result["orders"]), 2)

    @patch("command_llm.call_llm", return_value=VALID_LLM_RESPONSE)
    def test_timestep_preserved(self, _mock):
        result = run_dispatch(EXAMPLE_WORLD_STATE)
        self.assertEqual(result["timestep"], 7)

    @patch("command_llm.call_llm")
    def test_retry_on_constraint_violation(self, mock_llm):
        """
        First call returns over-dispatch, second call still has one violation,
        third call returns valid. Verifies retry loop fires and eventually
        returns the best response it can within MAX_RETRIES=2.
        """
        import json, copy

        # First response: medicine = 999 on order 1 (exceeds stock)
        bad = copy.deepcopy(EXAMPLE_DISPATCH_ORDERS)
        bad["orders"][0]["cargo"]["medicine"] = 999
        bad_json = json.dumps(bad)

        # Second response: still bad but different violation (quantity just over)
        still_bad = copy.deepcopy(EXAMPLE_DISPATCH_ORDERS)
        still_bad["orders"][0]["cargo"]["medicine"] = 100  # still > 60
        still_bad_json = json.dumps(still_bad)

        # Third response: valid
        good_json = VALID_LLM_RESPONSE

        mock_llm.side_effect = [bad_json, still_bad_json, good_json]
        result = run_dispatch(EXAMPLE_WORLD_STATE)
        # Should have called LLM 3 times (initial + 2 retries)
        self.assertEqual(mock_llm.call_count, 3)
        # Final result medicine should not exceed depot stock
        depot_medicine = EXAMPLE_WORLD_STATE["depot_inventory"]["DEPOT-A"]["medicine"]
        result_medicine = result["orders"][0]["cargo"]["medicine"]
        self.assertLessEqual(result_medicine, depot_medicine)

    @patch("command_llm.call_llm", return_value='{"garbage": true}')
    def test_missing_orders_key_defaults_to_empty(self, _mock):
        """LLM returns valid JSON but missing 'orders' key — should default to []."""
        result = run_dispatch(EXAMPLE_WORLD_STATE)
        self.assertEqual(result.get("orders", []), [])

    @patch("command_llm.call_llm", return_value="This is not JSON at all")
    def test_unparseable_raises_runtime_error(self, _mock):
        with self.assertRaises(RuntimeError):
            run_dispatch(EXAMPLE_WORLD_STATE)


# ---------------------------------------------------------------------------
# Tests: lateral transfer scenario
# ---------------------------------------------------------------------------

class TestLateralTransfer(unittest.TestCase):

    LATERAL_WORLD_STATE = {
        "timestep": 18,
        "depot_inventory": {
            "DEPOT-A": {"mre": 30, "medicine": 50, "comms": 8,  "ambulances": 1},
            "DEPOT-B": {"mre": 40, "medicine": 0,  "comms": 5,  "ambulances": 2},
            "DEPOT-C": {"mre": 80, "medicine": 80, "comms": 10, "ambulances": 2},
        },
        "vehicle_status": [
            {"id": "TRUCK-01", "type": "truck", "depot": "DEPOT-A",
             "status": "idle", "destination": None,
             "cargo": {"mre": 0, "medicine": 0, "comms": 0, "ambulances": 0},
             "eta_steps": 0},
            {"id": "TRUCK-02", "type": "truck", "depot": "DEPOT-B",
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

    VALID_LATERAL_RESPONSE = {
        "timestep": 18,
        "orders": [
            {
                "order_id": "ORD-T018-001",
                "vehicle_id": "TRUCK-01",
                "from_depot": "DEPOT-A",
                "destination": "DEPOT-B",
                "cargo": {"mre": 0, "medicine": 40, "comms": 0, "ambulances": 0},
                "priority": "critical",
                "is_lateral_transfer": True,
                "reason": "DEPOT-B has zero medicine. Transferring 40 units from DEPOT-A.",
            },
            {
                "order_id": "ORD-T018-002",
                "vehicle_id": "TRUCK-02",
                "from_depot": "DEPOT-B",
                "destination": "NODE-H2",
                "cargo": {"mre": 0, "medicine": 40, "comms": 0, "ambulances": 0},
                "priority": "critical",
                "is_lateral_transfer": False,
                "reason": "After lateral transfer, DEPOT-B can now serve NODE-H2.",
            },
        ],
        "deferred_requests": [],
        "llm_summary": "Lateral transfer from DEPOT-A to DEPOT-B enables hospital delivery.",
    }

    @patch("command_llm.call_llm",
           return_value=json.dumps(VALID_LATERAL_RESPONSE))
    def test_lateral_transfer_passes_validation(self, _mock):
        violations = validate_orders(self.VALID_LATERAL_RESPONSE, self.LATERAL_WORLD_STATE)
        self.assertEqual(violations, [], f"Violations: {violations}")

    @patch("command_llm.call_llm",
           return_value=json.dumps(VALID_LATERAL_RESPONSE))
    def test_lateral_transfer_flag_preserved(self, _mock):
        result = run_dispatch(self.LATERAL_WORLD_STATE)
        lateral_orders = [o for o in result["orders"] if o.get("is_lateral_transfer")]
        self.assertTrue(len(lateral_orders) >= 1, "Expected at least one lateral transfer order")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)