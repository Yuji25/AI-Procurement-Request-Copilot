"""Offline UI integration: no provider calls, local servers, or credentials."""
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.contracts import EvidenceItem, ProcurementDecision, RunTelemetry


APP = Path(__file__).resolve().parents[1] / "app.py"


def decision(request_id="REQ-1001", **changes):
    return ProcurementDecision(
        request_id=request_id, recommendation="Route for required human reviews before approval",
        required_approvals=["Manager"], next_step="Send evidence to the manager for review.",
        evidence=[EvidenceItem(source="department_budget", finding="Available budget: $29,000",
                               reference="data/department_budgets.csv#Finance")],
        telemetry=RunTelemetry(llm_calls=1, logical_llm_calls=1, tool_calls=6,
                               prompt_tokens=424, completion_tokens=60, cached_tokens=0), **changes)


class ReviewUITests(unittest.TestCase):
    def test_click_only_analysis_result_persistence_and_provenance(self):
        with patch("src.solution.handle_request", return_value=decision()) as handle:
            app = AppTest.from_file(str(APP)).run()
            self.assertEqual(len(app.exception), 0)
            handle.assert_not_called()
            self.assertEqual(len(app.json), 0)
            self.assertEqual(app.sidebar.radio[0].options, ["Architecture A — Single", "Architecture B — Staged"])
            self.assertTrue(any("Humans must review" in item.value for item in app.info))
            app.sidebar.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            handle.assert_called_once_with("REQ-1001", architecture="single")
            self.assertEqual([e.label for e in app.expander], ["Run telemetry", "Decision JSON (debug)"])
            frames = [table.value for table in app.dataframe]
            evidence = next(frame for frame in frames if "Source" in frame.columns)
            self.assertEqual(evidence.iloc[0]["Reference"], "data/department_budgets.csv#Finance")
            telemetry = next(frame for frame in frames if "Metric" in frame.columns)
            self.assertIn("Cached prompt tokens", telemetry["Metric"].tolist())
            app.run()
            handle.assert_called_once()
            self.assertEqual(len(app.json), 1)

    def test_selection_change_hides_old_result_and_staged_dispatches(self):
        with patch("src.solution.handle_request", return_value=decision()) as handle:
            app = AppTest.from_file(str(APP)).run()
            app.sidebar.button[0].click().run()
            app.sidebar.radio[0].set_value("staged").run()
            handle.assert_called_once()
            self.assertEqual(len(app.json), 0)
            app.sidebar.button[0].click().run()
            self.assertEqual(handle.call_args.kwargs["architecture"], "staged")
            app.sidebar.selectbox[0].set_value("REQ-1006").run()
            self.assertEqual(len(app.json), 0)
            self.assertEqual(handle.call_count, 2)
            self.assertEqual(app.metric[0].value, "Not provided")

    def test_missing_information_and_risks_are_visible(self):
        result = decision("REQ-1006", missing_information=["annual_cost_usd"],
                          risk_flags=["missing_information", "prompt_injection_detected"])
        result.telemetry = RunTelemetry(llm_calls=0, logical_llm_calls=0, tool_calls=6)
        with patch("src.solution.handle_request", return_value=result):
            app = AppTest.from_file(str(APP)).run()
            app.sidebar.selectbox[0].set_value("REQ-1006").run()
            app.sidebar.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Required information is missing" in e.value for e in app.error))
            self.assertTrue(any("prompt_injection_detected" in e.value for e in app.text))
            self.assertEqual(len(app.warning), 1)
            frame = next(table.value for table in app.dataframe if "Metric" in table.value.columns)
            self.assertNotIn("Cached prompt tokens", frame["Metric"].tolist())
            self.assertIn("Not reported", frame["Value"].tolist())

    def test_failure_hides_previous_result_and_sanitizes_exception(self):
        secret = "offline-secret-must-not-be-displayed"
        with patch("src.solution.handle_request", side_effect=[decision(), RuntimeError(secret)]):
            app = AppTest.from_file(str(APP)).run()
            app.sidebar.button[0].click().run()
            self.assertEqual(len(app.json), 1)
            app.sidebar.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.json), 0)
            self.assertTrue(any("Analysis could not be completed" in e.value for e in app.error))
            self.assertNotIn(secret, str(app))


if __name__ == "__main__":
    unittest.main()
