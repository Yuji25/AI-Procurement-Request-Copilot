"""Compact model input. Full tool records and provenance remain on the host."""
from __future__ import annotations

from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field

from src.tools import EvidenceTools


class PackModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequestFacts(PackModel):
    purpose: str
    product: str
    vendor: str
    category: str | None = None
    annual_cost_usd: Decimal
    users: int
    data_access: str
    integrations: list[str]


class SoftwareMatch(PackModel):
    product: str
    category: str | None = None
    status: str | None = None
    scope: str | None = None
    licensed_seats: int | None = None
    notes: str | None = None


class PurchaseSummary(PackModel):
    relevant_count: int
    same_vendor_count: int
    approved_count: int
    products: list[str]


class InternalVendorFacts(PackModel):
    procurement_status: str | None = None
    security_status: str | None = None
    security_review_date: str | None = None
    legal_terms_status: str | None = None
    notes: str | None = None


class ExternalRiskFacts(PackModel):
    security_review_status: str | None = None
    last_review_date: str | None = None
    stores_data_outside_region: bool | None = None
    processes_personal_data: bool | None = None
    risk_level: str | None = None


class PolicyOutcome(PackModel):
    required_approvals: list[str]
    risk_flags: list[str]
    missing_information: list[str]
    unknown_checks: list[str]


class EvidencePack(PackModel):
    request: RequestFacts
    department: str
    available_budget_usd: Decimal
    software_matches: list[SoftwareMatch]
    purchases: PurchaseSummary
    internal_vendor: InternalVendorFacts
    external_risk: ExternalRiskFacts
    policy: PolicyOutcome
    unknown_evidence: list[str] = Field(default_factory=list)


def build_evidence_pack(tools: EvidenceTools, guard_flags: list[str]) -> EvidencePack:
    """Read gathered state only: never invoke tools or serialize provenance."""
    context = tools.results["request_context"].data
    request = context["request"]
    overlap = tools.results["software_overlap"].data
    history = overlap["purchase_history"]
    policy = tools.results["policy_evaluation"].data
    return EvidencePack(
        request=RequestFacts(purpose=request["business_justification"], product=request["product_name"],
                             vendor=request["vendor_name"], category=request.get("category"),
                             annual_cost_usd=request["annual_cost_usd"], users=request["user_count"],
                             data_access=request["data_access_level"], integrations=request["requested_integrations"]),
        department=context["requester"]["department"],
        available_budget_usd=tools.results["department_budget"].data["budget"]["available_usd"],
        software_matches=[SoftwareMatch(product=r["product_name"], category=r.get("category"),
                                        status=r.get("status"), scope=r.get("scope"),
                                        licensed_seats=r.get("licensed_seats"), notes=r.get("notes"))
                          for r in overlap["catalog_matches"]],
        purchases=PurchaseSummary(relevant_count=len(history),
                                  same_vendor_count=sum(str(r.get("vendor_name", "")).casefold() == request["vendor_name"].casefold() for r in history),
                                  approved_count=sum(str(r.get("status", "")).casefold() == "approved" for r in history),
                                  products=sorted({r["product_name"] for r in history if r.get("product_name")})),
        internal_vendor=InternalVendorFacts(**{k: v for k, v in tools.results["vendor_registry"].data["vendor"].items()
                                              if k in InternalVendorFacts.model_fields}),
        external_risk=ExternalRiskFacts(**{k: v for k, v in tools.results["vendor_risk"].data["vendor_risk"].items()
                                          if k in ExternalRiskFacts.model_fields}),
        policy=PolicyOutcome(required_approvals=policy["required_approvals"],
                             risk_flags=list(dict.fromkeys(list(policy["risk_flags"]) + guard_flags)),
                             missing_information=policy["missing_information"], unknown_checks=policy["unknown_checks"]),
        unknown_evidence=[name for name, result in tools.results.items() if result.status != "ok"],
    )
