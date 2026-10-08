from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd
import requests

from src.telemetry import RunTelemetryCounter
from src.tools import EvidenceTools


def risk(name="SignFlow", **changes):
    return {"vendor_name": name, "security_review_status": "approved", "last_review_date": "2026-06-20",
            "processes_personal_data": True, "stores_data_outside_region": False, **changes}


class EvidenceToolTests(unittest.TestCase):
    def setUp(self):
        self.counter = RunTelemetryCounter()
        self.tools = EvidenceTools("REQ-1001", self.counter)
        self.risk = patch("src.tools.get_vendor_risk", return_value=risk()).start()
        self.addCleanup(patch.stopall)

    def test_all_six_tools_have_facts_and_provenance(self):
        self.tools.gather_required()
        self.assertEqual(len(self.tools.results), 6)
        self.assertEqual(self.counter.tool_calls, 6)
        self.assertTrue(all(r.status == "ok" and r.references for r in self.tools.results.values()))
        self.assertEqual(self.tools.results["request_context"].data["requester"]["department"], "Finance")
        self.assertEqual(self.tools.results["department_budget"].data["budget"]["available_usd"], 29000)
        self.assertIn("Manager", self.tools.results["policy_evaluation"].data["required_approvals"])
        self.risk.assert_called_once_with("SignFlow")
        self.assertNotIn("supporting_evidence", self.tools.results["policy_evaluation"].data)

    def test_unknown_tool_and_arguments_never_execute(self):
        for name, args in (("purchase", {"request_id": "REQ-1001"}),
                           ("vendor_risk", {"request_id": "REQ-1001", "vendor_name": "spoof"}),
                           ("request_context", {"request_id": "REQ-1002"}),
                           ("request_context", {"request_id": 1}), ("request_context", [])):
            self.assertEqual(self.tools.execute(name, args).status, "error")
        self.assertEqual(self.counter.tool_calls, 0)
        self.risk.assert_not_called()

    def test_api_failure_and_mismatched_or_incomplete_payload(self):
        for result in (requests.Timeout("private upstream text"), {"vendor_name": "Wrong"},
                       {"vendor_name": "SignFlow"}):
            tools = EvidenceTools("REQ-1001", RunTelemetryCounter())
            self.risk.side_effect = result if isinstance(result, Exception) else None
            self.risk.return_value = result
            data = tools.execute("vendor_risk", {"request_id": "REQ-1001"})
            self.assertIn(data.status, {"unavailable", "error", "unknown"})
            self.assertNotIn("vendor_risk", data.data)
            self.assertNotIn("private upstream text", data.model_dump_json())

    def test_unknown_request_and_missing_records(self):
        unknown = EvidenceTools("not-a-request", RunTelemetryCounter())
        self.assertEqual(unknown.ensure("request_context").status, "unknown")
        with patch("src.tools.data.load_budgets", return_value=pd.DataFrame(columns=["department"])):
            self.assertEqual(self.tools.ensure("department_budget").status, "unknown")

    def test_csv_missing_values_are_none(self):
        tools = EvidenceTools("REQ-1002", RunTelemetryCounter())
        registry = tools.ensure("vendor_registry")
        self.assertIsNone(registry.data["vendor"]["security_review_date"])

    def test_employee_failure_preserves_known_request_and_financial_tier(self):
        with patch("src.tools.data.load_employees", side_effect=OSError("unavailable")):
            self.tools.gather_required()
        context = self.tools.results["request_context"]
        self.assertEqual(context.status, "unavailable")
        self.assertEqual(context.data["request"]["annual_cost_usd"], 800)
        self.assertIn("Manager", self.tools.results["policy_evaluation"].data["required_approvals"])

    def test_overlap_returns_scoped_candidates_without_claiming_unused_seats(self):
        tools = EvidenceTools("REQ-1002", RunTelemetryCounter())
        result = tools.ensure("software_overlap")
        matches = result.data["catalog_matches"]
        self.assertEqual({r["product_name"] for r in matches}, {"PixelCraft Pro", "CreativeSuite"})
        self.assertTrue(all("unused_seats" not in r for r in matches))

    def test_tool_limit_reserves_mandatory_evidence(self):
        tools = EvidenceTools("REQ-1001", RunTelemetryCounter(), max_calls=6)
        tools.ensure("request_context")
        self.assertEqual(tools.execute("request_context", {"request_id": "REQ-1001"}).status, "error")
        tools.gather_required()
        self.assertEqual(len(tools.results), 6)
        self.assertEqual(tools.telemetry.tool_calls, 6)


if __name__ == "__main__":
    unittest.main()
