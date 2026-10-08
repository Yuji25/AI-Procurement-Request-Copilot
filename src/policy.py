"""Deterministic rules implementing procurement_policy.md, version 2026.09.

The Markdown is the documented business source of truth. Changes to that policy
require a reviewed code/test update; nothing is parsed at runtime. This module
does not fetch evidence, interpret free text, or recommend/approve purchases.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

POLICY_VERSION = "2026.09"
REFERENCE_DATE = date(2026, 9, 30)
SECURITY_REVIEW_VALID_DAYS = 365
FINANCIAL_TIERS = (
    (Decimal("1000"), ("Manager",)),
    (Decimal("10000"), ("Department Head", "Procurement")),
    (Decimal("25000"), ("Department Head", "Finance", "Procurement")),
)
HIGH_SPEND_APPROVALS = ("Department Head", "Finance", "CFO", "Procurement")
NEW_VENDOR_LEGAL_THRESHOLD = Decimal("10000")

CheckStatus = Literal["pass", "fail", "unknown"]


@dataclass(frozen=True)
class PolicyCheck:
    name: str
    status: CheckStatus
    finding: str
    section: int

    @property
    def reference(self) -> str:
        return f"data/procurement_policy.md section {self.section} (v{POLICY_VERSION})"


@dataclass(frozen=True)
class ReviewAge:
    status: Literal["current", "expired", "unknown"]
    age_days: int | None
    reason: str


@dataclass(frozen=True)
class PolicyResult:
    checks: tuple[PolicyCheck, ...]
    required_approvals: tuple[str, ...]
    risk_flags: tuple[str, ...]
    missing_information: tuple[str, ...]
    human_review_required: bool = True

    @property
    def unknown_checks(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if c.status == "unknown")


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _known_text(value: object) -> bool:
    return bool(_text(value)) and _text(value).lower() not in {"unknown", "nan", "n/a"}


def _normal(value: object) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _amount(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def required_request_information(
    request: Mapping[str, object], department: str | None = None,
) -> tuple[str, ...]:
    missing = []
    for field in ("requester_id", "product_name", "vendor_name", "business_justification"):
        if not _known_text(request.get(field)):
            missing.append(field)
    if not _known_text(department if department is not None else request.get("department")):
        missing.append("department")
    if _amount(request.get("annual_cost_usd")) is None:
        missing.append("annual_cost_usd")
    users = _amount(request.get("user_count"))
    if users is None or users <= 0 or users != users.to_integral_value():
        missing.append("user_count")
    if not _known_text(request.get("data_access_level")):
        missing.append("data_access_level")
    integrations = request.get("requested_integrations")
    if not isinstance(integrations, list) or any(not _known_text(i) for i in integrations):
        missing.append("requested_integrations")
    return tuple(missing)


def financial_approvals(annual_cost_usd: object) -> tuple[str, ...] | None:
    """None means the tier cannot be determined; zero is a known amount."""
    amount = _amount(annual_cost_usd)
    if amount is None:
        return None
    for ceiling, approvals in FINANCIAL_TIERS:
        if amount <= ceiling:
            return approvals
    return HIGH_SPEND_APPROVALS


def budget_sufficiency(annual_cost_usd: object, available_usd: object) -> PolicyCheck:
    amount, available = _amount(annual_cost_usd), _amount(available_usd)
    if amount is None or available is None:
        return PolicyCheck("budget", "unknown", "Annual cost or available budget is missing/invalid.", 2)
    sufficient = amount <= available
    return PolicyCheck("budget", "pass" if sufficient else "fail",
                       f"Annual cost ${amount} compared with available budget ${available}.", 2)


def vendor_review_age(review_date: object) -> ReviewAge:
    try:
        reviewed = date.fromisoformat(_text(review_date))
    except ValueError:
        return ReviewAge("unknown", None, "Security review date is missing or invalid.")
    age = (REFERENCE_DATE - reviewed).days
    if age < 0:
        return ReviewAge("unknown", None, "Security review date is after the assessment reference date.")
    status = "current" if age <= SECURITY_REVIEW_VALID_DAYS else "expired"
    return ReviewAge(status, age, f"Security review is {age} days old at {REFERENCE_DATE}.")


def _security_status(value: object) -> str:
    status = _normal(value)
    return "not_completed" if status in {"pending", "not_completed"} else status


def evaluate_policy(
    request: Mapping[str, object], *, department: str | None = None,
    available_budget_usd: object = None,
    vendor: Mapping[str, object] | None = None,
    vendor_risk: Mapping[str, object] | None = None,
) -> PolicyResult:
    """Check supplied facts only; vendor_risk must be retrieved by the caller.

    Optional structured request facts: source_code_access, production_integration,
    cloud_account_integration, sensitive_data, stores_data_outside_region,
    material_data_processing_issue, material_cross_region_issue (booleans).
    Missing conditional facts are reported as unknown, never assumed false.
    """
    checks: list[PolicyCheck] = []
    approvals: list[str] = []
    flags: list[str] = []
    internal = vendor or {}
    external = vendor_risk or {}

    def add(name: str, status: CheckStatus, finding: str, section: int) -> None:
        checks.append(PolicyCheck(name, status, finding, section))

    def review(role: str, flag: str) -> None:
        approvals.append(role)
        flags.append(flag)

    missing = required_request_information(request, department)
    add("required_information", "fail" if missing else "pass",
        "Missing/invalid: " + ", ".join(missing) if missing else "Required request facts are present.", 1)
    if missing:
        flags.append("missing_information")
    budget = budget_sufficiency(request.get("annual_cost_usd"), available_budget_usd)
    checks.append(budget)
    if budget.status == "fail":
        review("Finance", "budget_insufficient")
    tier = financial_approvals(request.get("annual_cost_usd"))
    add("financial_approvals", "unknown" if tier is None else "pass",
        "Annual cost is missing/invalid." if tier is None else "Minimum approvals: " + ", ".join(tier), 4)
    approvals.extend(tier or ())

    access = _normal(request.get("data_access_level"))
    pii = access in {"employee_pii", "customer_pii"}
    sensitive_levels = {"source_code", "confidential_documents", "employee_pii", "customer_pii",
                        "credentials", "secrets", "credentials/secrets", "credentials_secrets"}
    known_levels = sensitive_levels | {"none", "internal_documents", "internal_marketing", "production_telemetry"}
    integrations = request.get("requested_integrations")
    integration_values = {_normal(i) for i in integrations} if isinstance(integrations, list) else set()
    sensitive_access = access in sensitive_levels
    production_integrations = {"production", "production_cloud_account", "cloud_account",
                               "production_integration", "cloud_account_integration"}
    production_access = bool(integration_values & production_integrations)
    known_integrations = production_integrations | {"sso", "git_repositories", "helpdeskly", "crm", "document_repository"}
    explicit_security = any(request.get(k) is True for k in
                            ("source_code_access", "production_integration", "cloud_account_integration"))
    needs_security = sensitive_access or production_access or explicit_security
    security_unknown = (access not in known_levels or "requested_integrations" in missing
                        or bool(integration_values - known_integrations))
    add("security_access", "fail" if needs_security else "unknown" if security_unknown else "pass",
        "Sensitive access/integration requires Security." if needs_security else
        "Access/integration classification is incomplete." if security_unknown else
        "No explicit sensitive access/integration trigger.", 5)
    if needs_security:
        review("Security", "security_review_required")

    sources = []
    if vendor is not None:
        sources.append(("internal", internal.get("security_status"), internal.get("security_review_date")))
    if vendor_risk is not None:
        sources.append(("external", external.get("security_review_status"), external.get("last_review_date")))
    if not sources:
        add("vendor_security_review", "unknown", "No vendor security evidence was supplied.", 5)
        review("Security", "security_review_required")
    for source, raw_status, raw_date in sources:
        status, age = _security_status(raw_status), vendor_review_age(raw_date)
        expired = age.status == "expired" or status == "expired"
        current = status == "approved" and age.status == "current"
        check_status: CheckStatus = "pass" if current else "fail" if expired or status in {
            "not_completed", "rejected", "denied"} else "unknown"
        add(f"{source}_security_review", check_status,
            f"{source} assessment status: {status or 'unknown'}. {age.reason}", 5)
        if not current:
            review("Security", "security_review_required")
        if expired:
            flags.append("vendor_review_expired")
    if vendor is None or vendor_risk is None:
        add("vendor_evidence_consistency", "unknown", "Both vendor sources are needed to verify agreement.", 5)
    else:
        s1, s2 = _security_status(internal.get("security_status")), _security_status(external.get("security_review_status"))
        d1, d2 = internal.get("security_review_date"), external.get("last_review_date")
        known_statuses = {"approved", "expired", "not_completed", "rejected", "denied"}
        conflict = (s1 in known_statuses and s2 in known_statuses and s1 != s2) or (
            vendor_review_age(d1).age_days is not None and vendor_review_age(d2).age_days is not None and d1 != d2)
        fully_known = s1 in known_statuses and s2 in known_statuses and all(
            vendor_review_age(d).age_days is not None for d in (d1, d2))
        add("vendor_evidence_consistency", "fail" if conflict else "pass" if fully_known else "unknown",
            "Vendor sources conflict; retain both for manual review." if conflict else
            "Vendor evidence agrees." if fully_known else "Vendor evidence is incomplete.", 5)
        if conflict:
            review("Security", "security_review_required")
            flags.append("conflicting_vendor_evidence")

    storage_facts = (request.get("stores_data_outside_region"), external.get("stores_data_outside_region"))
    outside_region = any(v is True for v in storage_facts)
    storage_known = any(type(v) is bool for v in storage_facts)
    sensitive = sensitive_access or request.get("sensitive_data") is True
    privacy_trigger = pii or (sensitive and outside_region)
    privacy_unknown = access not in known_levels or (sensitive and not storage_known)
    add("privacy", "fail" if privacy_trigger else "unknown" if privacy_unknown else "pass",
        "PII or sensitive cross-region storage requires Privacy." if privacy_trigger else
        "Data class/storage region is incomplete." if privacy_unknown else "No explicit Privacy trigger.", 6)
    if privacy_trigger:
        review("Privacy", "privacy_review_required")

    new_status = _normal(internal.get("procurement_status"))
    new_vendor = new_status == "new"
    amount = _amount(request.get("annual_cost_usd"))
    terms = _normal(internal.get("legal_terms_status"))
    terms_known = terms in {"approved", "standard", "draft", "not_approved", "pending", "rejected", "non_standard"}
    terms_trigger = terms_known and terms not in {"approved", "standard"}
    material_issue = any(request.get(k) is True for k in
                         ("material_data_processing_issue", "material_cross_region_issue")) or (sensitive and outside_region)
    legal_trigger = (new_vendor and amount is not None and amount >= NEW_VENDOR_LEGAL_THRESHOLD) or terms_trigger or material_issue
    legal_unknown = (not terms_known or new_status not in {"new", "approved"}
                     or (new_vendor and amount is None) or (sensitive and not storage_known))
    # Unknown legal terms require human verification; they cannot count as approved.
    legal_review = legal_trigger or not terms_known
    add("legal", "fail" if legal_trigger else "unknown" if legal_unknown else "pass",
        "New-vendor spend, legal terms, or material data issue requires Legal." if legal_trigger else
        "Vendor onboarding/spend/terms are incomplete." if legal_unknown else "No explicit Legal trigger.", 7)
    if legal_review:
        review("Legal", "legal_review_required")

    return PolicyResult(tuple(checks), tuple(dict.fromkeys(approvals)),
                        tuple(dict.fromkeys(flags)), missing)
