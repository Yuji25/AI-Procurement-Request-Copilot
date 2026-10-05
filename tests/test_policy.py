from __future__ import annotations

import unittest
from datetime import timedelta

from src.policy import (REFERENCE_DATE, budget_sufficiency, evaluate_policy,
                        financial_approvals, required_request_information, vendor_review_age)


def request(**changes):
    return dict(requester_id="E004", department="Finance", product_name="Product",
                vendor_name="Vendor", annual_cost_usd=800, user_count=3,
                business_justification="Three additional signing identities",
                data_access_level="internal_documents", requested_integrations=[]) | changes


def vendor(**changes):
    return dict(procurement_status="Approved", security_status="Approved",
                security_review_date="2026-06-20", legal_terms_status="Approved") | changes


def risk(**changes):
    return dict(security_review_status="approved", last_review_date="2026-06-20",
                stores_data_outside_region=False) | changes


def evaluate(req=None, **changes):
    facts = dict(available_budget_usd=29000, vendor=vendor(), vendor_risk=risk()) | changes
    return evaluate_policy(request() if req is None else req, **facts)


class FinancialTests(unittest.TestCase):
    def test_tier_boundaries(self):
        cases = [
            ("1000", ("Manager",)),
            ("1000.01", ("Department Head", "Procurement")),
            ("10000", ("Department Head", "Procurement")),
            ("10000.01", ("Department Head", "Finance", "Procurement")),
            ("25000", ("Department Head", "Finance", "Procurement")),
            ("25000.01", ("Department Head", "Finance", "CFO", "Procurement")),
        ]
        for amount, expected in cases:
            with self.subTest(amount=amount):
                self.assertEqual(financial_approvals(amount), expected)

    def test_budget_equality_and_shortfall(self):
        self.assertEqual(budget_sufficiency("1000.01", "1000.01").status, "pass")
        result = evaluate(request(annual_cost_usd="1000.02"), available_budget_usd="1000.01")
        self.assertIn("budget_insufficient", result.risk_flags)
        self.assertIn("Finance", result.required_approvals)

    def test_missing_cost_has_no_invented_tier(self):
        result = evaluate(request(annual_cost_usd=None, data_access_level="source_code"))
        self.assertIsNone(financial_approvals(None))
        self.assertIn("annual_cost_usd", result.missing_information)
        self.assertIn("financial_approvals", result.unknown_checks)
        self.assertIn("budget", result.unknown_checks)
        self.assertNotIn("Manager", result.required_approvals)
        self.assertIn("Security", result.required_approvals)

    def test_invalid_amounts_are_unknown_not_zero(self):
        for amount in (True, -1, "NaN", "Infinity", "", "not money"):
            with self.subTest(amount=amount):
                self.assertIsNone(financial_approvals(amount))
                self.assertEqual(budget_sufficiency(amount, 1000).status, "unknown")
        self.assertEqual(financial_approvals(0), ("Manager",))
        self.assertEqual(budget_sufficiency(100, None).status, "unknown")


class RequiredInformationTests(unittest.TestCase):
    def test_complete_request_and_explicit_none_integrations(self):
        self.assertEqual(required_request_information(request(data_access_level="none")), ())

    def test_each_required_fact_is_checked(self):
        for field in request():
            with self.subTest(field=field):
                self.assertIn(field, required_request_information(request(**{field: None})))

    def test_department_can_be_resolved_from_employee(self):
        req = request()
        req.pop("department")
        self.assertEqual(required_request_information(req, department="Finance"), ())
        self.assertIn("department", required_request_information(req))

    def test_unknown_and_invalid_values(self):
        for value in (None, "", "unknown", float("nan")):
            self.assertIn("data_access_level", required_request_information(request(data_access_level=value)))
        for count in (0, -1, 1.5, True):
            self.assertIn("user_count", required_request_information(request(user_count=count)))
        for integrations in (None, "SSO", ["unknown"], [""]):
            self.assertIn("requested_integrations", required_request_information(request(requested_integrations=integrations)))


class SecurityReviewTests(unittest.TestCase):
    def test_364_365_366_day_boundaries(self):
        for days, expected in ((364, "current"), (365, "current"), (366, "expired")):
            with self.subTest(days=days):
                reviewed = (REFERENCE_DATE - timedelta(days=days)).isoformat()
                age = vendor_review_age(reviewed)
                self.assertEqual((age.status, age.age_days), (expected, days))
                result = evaluate(
                    vendor=vendor(security_review_date=reviewed), vendor_risk=risk(last_review_date=reviewed))
                self.assertEqual("vendor_review_expired" in result.risk_flags, days == 366)
                self.assertEqual("Security" in result.required_approvals, days == 366)

    def test_missing_invalid_future_dates(self):
        for value in (None, "", float("nan"), "broken", "2026-10-01"):
            with self.subTest(value=value):
                age = vendor_review_age(value)
                self.assertEqual((age.status, age.age_days), ("unknown", None))
                result = evaluate(vendor=vendor(security_review_date=value), vendor_risk=risk(last_review_date=value))
                self.assertIn("Security", result.required_approvals)
                self.assertNotIn("vendor_review_expired", result.risk_flags)

    def test_security_data_access_triggers_despite_approved_vendor(self):
        for access in ("source_code", "confidential_documents", "employee_pii", "customer_pii", "credentials", "secrets"):
            with self.subTest(access=access):
                result = evaluate(request(data_access_level=access))
                self.assertIn("Security", result.required_approvals)
                self.assertIn("security_review_required", result.risk_flags)

    def test_production_and_cloud_integrations(self):
        for integration in ("Production cloud account", "Cloud account", "Production"):
            result = evaluate(request(requested_integrations=[integration]))
            self.assertIn("Security", result.required_approvals)
        self.assertNotIn("Security", evaluate(request(requested_integrations=["SSO"])).required_approvals)
        for flag in ("source_code_access", "production_integration", "cloud_account_integration"):
            self.assertIn("Security", evaluate(request(**{flag: True})).required_approvals)

    def test_incomplete_assessment_and_no_evidence(self):
        result = evaluate(vendor=vendor(security_status="Pending"), vendor_risk=risk(security_review_status="not_completed"))
        self.assertIn("Security", result.required_approvals)
        self.assertNotIn("conflicting_vendor_evidence", result.risk_flags)
        self.assertIn("vendor_security_review", evaluate(vendor=None, vendor_risk=None).unknown_checks)

    def test_unrecognized_access_or_integration_is_unknown(self):
        self.assertIn("security_access", evaluate(request(data_access_level="unclassified")).unknown_checks)
        self.assertIn("security_access", evaluate(request(requested_integrations=["Unclassified connector"])).unknown_checks)

    def test_conflict_preserves_review_requirement(self):
        result = evaluate(vendor_risk=risk(security_review_status="expired"))
        self.assertIn("conflicting_vendor_evidence", result.risk_flags)
        self.assertIn("Security", result.required_approvals)
        result = evaluate(vendor_risk=risk(last_review_date="2026-06-21"))
        self.assertIn("conflicting_vendor_evidence", result.risk_flags)


class PrivacyLegalTests(unittest.TestCase):
    def test_pii_requires_privacy(self):
        for access in ("employee_pii", "customer_pii"):
            self.assertIn("Privacy", evaluate(request(data_access_level=access)).required_approvals)

    def test_sensitive_cross_region_requires_privacy_and_legal(self):
        result = evaluate(request(data_access_level="confidential_documents"), vendor_risk=risk(stores_data_outside_region=True))
        self.assertTrue({"Security", "Privacy", "Legal"}.issubset(result.required_approvals))
        # A vendor's generic PII capability is not proof this request uses PII.
        result = evaluate(vendor_risk=risk(processes_personal_data=True))
        self.assertNotIn("Privacy", result.required_approvals)

    def test_unknown_region_is_not_assumed_local(self):
        result = evaluate(request(data_access_level="confidential_documents"), vendor_risk=risk(stores_data_outside_region=None))
        self.assertIn("privacy", result.unknown_checks)

    def test_new_vendor_legal_boundary(self):
        for amount, expected in (("9999.99", False), ("10000", True), ("10000.01", True)):
            result = evaluate(request(annual_cost_usd=amount), vendor=vendor(procurement_status="New"))
            self.assertEqual("Legal" in result.required_approvals, expected)
        self.assertNotIn("Legal", evaluate(request(annual_cost_usd=10000)).required_approvals)

    def test_nonstandard_and_unknown_legal_terms(self):
        for terms in ("Draft", "Pending", "Non-standard", "Unknown", None):
            result = evaluate(vendor=vendor(legal_terms_status=terms))
            self.assertIn("Legal", result.required_approvals)
        self.assertIn("legal", evaluate(vendor=vendor(legal_terms_status=None)).unknown_checks)

    def test_explicit_material_issues(self):
        for flag in ("material_data_processing_issue", "material_cross_region_issue"):
            self.assertIn("Legal", evaluate(request(**{flag: True})).required_approvals)

    def test_missing_cost_keeps_independent_legal_trigger(self):
        result = evaluate(request(annual_cost_usd=None), vendor=vendor(procurement_status="New", legal_terms_status="Draft"))
        self.assertIn("Legal", result.required_approvals)
        self.assertIn("financial_approvals", result.unknown_checks)

    def test_policy_metadata_and_human_authority(self):
        result = evaluate(request(data_access_level="customer_pii"))
        self.assertTrue(result.human_review_required)
        self.assertEqual(len(result.required_approvals), len(set(result.required_approvals)))
        self.assertTrue(all("procurement_policy.md section" in check.reference for check in result.checks))
        self.assertTrue(all("v2026.09" in check.reference for check in result.checks))


if __name__ == "__main__":
    unittest.main()
