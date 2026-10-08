"""Read-only, request-scoped evidence tools. No backing risk-JSON access."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal
from urllib.parse import quote

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src import data_access as data
from src.policy import evaluate_policy
from src.provider import ToolDefinition
from src.telemetry import RunTelemetryCounter
from src.vendor_client import get_vendor_risk


class ToolResult(BaseModel):
    name: str
    status: Literal["ok", "unknown", "unavailable", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
    references: list[str] = Field(default_factory=list)
    issue: str | None = None


class RequestArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1)


DESCRIPTIONS = {
    "request_context": "Read this request, requester, department, and manager.",
    "department_budget": "Read the requester's department budget snapshot.",
    "software_overlap": "Read catalog and purchase-history matches; seats are licensed, not unused capacity.",
    "vendor_registry": "Read the requested vendor's internal onboarding, security, and legal facts.",
    "vendor_risk": "Retrieve the requested vendor's external risk through the mock API.",
    "policy_evaluation": "Apply authoritative deterministic checks to gathered facts; collects missing prerequisites.",
}


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def _normal(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


class EvidenceTools:
    def __init__(self, request_id: str, telemetry: RunTelemetryCounter, max_calls: int = 24):
        if max_calls < len(DESCRIPTIONS):
            raise ValueError("Tool limit must allow the mandatory evidence tools")
        self.request_id, self.telemetry, self.max_calls = request_id, telemetry, max_calls
        self.results: dict[str, ToolResult] = {}

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(ToolDefinition(name, description, RequestArguments.model_json_schema())
                     for name, description in DESCRIPTIONS.items())

    def execute(self, name: str, arguments: object) -> ToolResult:
        if name not in DESCRIPTIONS:
            return ToolResult(name="rejected_tool", status="error", issue="Unknown tool; no execution performed.")
        try:
            args = RequestArguments.model_validate(arguments)
            if args.request_id != self.request_id:
                raise ValueError("Wrong request scope")
        except (ValidationError, ValueError):
            return ToolResult(name=name, status="error", issue="Invalid arguments or request scope; no execution performed.")
        missing = sum(n not in self.results for n in DESCRIPTIONS)
        if self.telemetry.tool_calls >= self.max_calls or (
            name in self.results and self.telemetry.tool_calls >= self.max_calls - missing
        ):
            return ToolResult(name=name, status="error", issue="Tool-call limit reached; no execution performed.")
        self.telemetry.record_tool_call(name)
        if name in self.results:
            return self.results[name]
        try:
            result = getattr(self, "_" + name)()
        except (OSError, KeyError, ValueError, TypeError, requests.RequestException):
            result = ToolResult(name=name, status="unavailable", issue="Evidence source failed; facts could not be verified.")
        self.results[name] = result
        return result

    def ensure(self, name: str) -> ToolResult:
        return self.results.get(name) or self.execute(name, {"request_id": self.request_id})

    def gather_required(self) -> None:
        for name in DESCRIPTIONS:
            self.ensure(name)

    def _request_context(self) -> ToolResult:
        try:
            request = data.get_request(self.request_id)
        except KeyError:
            return ToolResult(name="request_context", status="unknown", issue="Request ID was not found.",
                              references=["data/requests.json"])
        try:
            employees = _records(data.load_employees())
        except (OSError, ValueError, KeyError):
            return ToolResult(name="request_context", status="unavailable",
                              data={"request": request, "requester": None, "manager": None},
                              references=[f"data/requests.json#{self.request_id}", "data/employees.csv"],
                              issue="Request is available but requester/department evidence could not be loaded.")
        matches = [e for e in employees if e["employee_id"] == request.get("requester_id")]
        employee = matches[0] if len(matches) == 1 else None
        manager = next((e for e in employees if employee and e["employee_id"] == employee.get("manager_id")), None)
        return ToolResult(name="request_context", status="ok" if employee else "unknown",
                          data={"request": request, "requester": employee, "manager": manager},
                          references=[f"data/requests.json#{self.request_id}",
                                      f"data/employees.csv#{request.get('requester_id', 'unknown')}"],
                          issue=None if employee else "Requester/department could not be resolved uniquely.")

    def _context(self) -> tuple[dict, dict]:
        context = self.ensure("request_context")
        return context.data.get("request", {}), context.data.get("requester") or {}

    def _department_budget(self) -> ToolResult:
        _, employee = self._context()
        department = employee.get("department")
        rows = [r for r in _records(data.load_budgets()) if r["department"] == department]
        return ToolResult(name="department_budget", status="ok" if len(rows) == 1 else "unknown",
                          data={"budget": rows[0] if len(rows) == 1 else None},
                          references=[f"data/department_budgets.csv#{department or 'unknown'}"],
                          issue=None if len(rows) == 1 else "Department budget could not be resolved uniquely.")

    def _software_overlap(self) -> ToolResult:
        request, employee = self._context()
        catalog = []
        for record in _records(data.load_software_catalog()):
            if not _normal(record.get("status")).startswith("approved"):
                continue
            if record.get("scope") not in {"Company-wide", employee.get("department")}:
                continue
            reasons = [field for field in ("product_name", "vendor_name", "category")
                       if _normal(request.get(field)) and _normal(request.get(field)) == _normal(record.get(field))]
            if reasons:
                catalog.append({**record, "match_fields": reasons})
        history = [r for r in _records(data.load_purchase_history()) if
                   _normal(r.get("vendor_name")) == _normal(request.get("vendor_name")) or
                   (employee.get("department") and r.get("department") == employee["department"])]
        return ToolResult(name="software_overlap", status="ok", data={"catalog_matches": catalog, "purchase_history": history},
                          references=["data/software_catalog.csv", "data/purchase_history.csv"])

    def _vendor_registry(self) -> ToolResult:
        request, _ = self._context()
        name = request.get("vendor_name")
        rows = [r for r in _records(data.load_vendors()) if _normal(r["vendor_name"]) == _normal(name)]
        return ToolResult(name="vendor_registry", status="ok" if len(rows) == 1 else "unknown",
                          data={"vendor": rows[0] if len(rows) == 1 else None},
                          references=[f"data/vendors.csv#{name or 'unknown'}"],
                          issue=None if len(rows) == 1 else "Internal vendor record could not be resolved uniquely.")

    def _vendor_risk(self) -> ToolResult:
        request, _ = self._context()
        name = request.get("vendor_name")
        if not isinstance(name, str) or not name.strip():
            return ToolResult(name="vendor_risk", status="unknown", issue="Vendor name is missing.")
        reference = "/vendor-risk/" + quote(name, safe="")
        try:
            risk = get_vendor_risk(name)
        except (requests.RequestException, ValueError):
            return ToolResult(name="vendor_risk", status="unavailable", references=[reference],
                              issue="Vendor-risk API unavailable or returned invalid JSON.")
        if not isinstance(risk, dict) or _normal(risk.get("vendor_name")) != _normal(name):
            return ToolResult(name="vendor_risk", status="error", references=[reference], issue="Invalid vendor-risk payload or vendor mismatch.")
        status = risk.get("security_review_status")
        if not isinstance(status, str) or any(type(risk.get(k)) is not bool for k in
                                              ("processes_personal_data", "stores_data_outside_region")):
            return ToolResult(name="vendor_risk", status="unknown", references=[reference],
                              issue="Vendor-risk payload is missing valid structured assessment facts.")
        return ToolResult(name="vendor_risk", status="ok", data={"vendor_risk": risk}, references=[reference])

    def _policy_evaluation(self) -> ToolResult:
        request, employee = self._context()
        budget = self.ensure("department_budget").data.get("budget") or {}
        vendor = self.ensure("vendor_registry").data.get("vendor")
        risk = self.ensure("vendor_risk").data.get("vendor_risk")
        self.ensure("software_overlap")
        result = evaluate_policy(request, department=employee.get("department"),
                                 available_budget_usd=budget.get("available_usd"), vendor=vendor, vendor_risk=risk)
        facts = asdict(result)
        facts["unknown_checks"] = result.unknown_checks
        facts["checks"] = [{**asdict(c), "reference": c.reference} for c in result.checks]
        return ToolResult(name="policy_evaluation", status="unknown" if result.unknown_checks else "ok",
                          data=facts, references=list(dict.fromkeys(c.reference for c in result.checks)))
