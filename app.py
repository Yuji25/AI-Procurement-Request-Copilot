"""Read-only procurement review UI. Analysis runs only on an explicit click."""
from __future__ import annotations

import streamlit as st

from src.data_access import load_employees, load_requests
from src.solution import handle_request


def display_value(value: object) -> str:
    return "Not provided" if value is None or value == "" else str(value)


st.set_page_config(page_title="Procurement Request Copilot", layout="wide")
st.title("Procurement Request Copilot")
st.caption("Evidence-based purchase recommendations · Policy reference date: 30 September 2026")
st.info("Recommendations are advisory. Humans must review and approve purchases; this tool does not approve or buy anything.")

requests = load_requests()
by_id = {request["request_id"]: request for request in requests}
st.sidebar.header("Review a request")
request_id = st.sidebar.selectbox(
    "Request", list(by_id),
    format_func=lambda rid: f"{rid} — {by_id[rid]['product_name']}",
)
architecture = st.sidebar.radio(
    "Architecture", ["single", "staged"],
    format_func=lambda value: "Architecture A — Single" if value == "single" else "Architecture B — Staged",
)
st.sidebar.caption("Architecture A is the recommended MVP default.")
run = st.sidebar.button("Run Analysis", type="primary")
request = by_id[request_id]
selection = (request_id, architecture)

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("Purchase request")
    st.caption(request_id)
    st.text(display_value(request.get("product_name")))
    try:
        employees = load_employees()
        matches = employees[employees["employee_id"] == request.get("requester_id")]
        requester = matches.iloc[0].to_dict() if len(matches) == 1 else {}
    except (OSError, ValueError, KeyError):
        requester = {}
    st.dataframe([
        {"Field": "Vendor", "Value": display_value(request.get("vendor_name"))},
        {"Field": "Category", "Value": display_value(request.get("category"))},
        {"Field": "Requester", "Value": display_value(requester.get("name") or request.get("requester_id"))},
        {"Field": "Department", "Value": display_value(requester.get("department"))},
    ], hide_index=True)
    cost, users = st.columns(2)
    annual_cost = request.get("annual_cost_usd")
    cost.metric("Annual cost", f"${annual_cost:,.2f}" if annual_cost is not None else "Not provided")
    users.metric("Users", display_value(request.get("user_count")))
    integrations = request.get("requested_integrations")
    st.dataframe([
        {"Field": "Data access", "Value": display_value(request.get("data_access_level"))},
        {"Field": "Integrations", "Value": ", ".join(integrations) if integrations else ("None requested" if integrations == [] else "Not provided")},
        {"Field": "Urgency", "Value": display_value(request.get("urgency"))},
    ], hide_index=True)
    st.markdown("**Business justification**")
    st.text(display_value(request.get("business_justification")))

with right:
    st.subheader("Review outcome")
    if run:
        st.session_state.pop("analysis", None)
        try:
            with st.spinner("Gathering evidence and preparing the recommendation…"):
                decision = handle_request(request_id, architecture=architecture)
            st.session_state["analysis"] = {"selection": selection, "decision": decision}
        except Exception:
            # Never display raw transport/configuration exception text.
            st.error("Analysis could not be completed. Check local service and provider readiness, then try again. Human review is still required.")

    analysis = st.session_state.get("analysis")
    if analysis and analysis["selection"] == selection:
        decision = analysis["decision"]
        if decision.missing_information or decision.risk_flags:
            st.warning(decision.recommendation)
        else:
            st.info(decision.recommendation)
        st.caption("Human review required" if decision.human_review_required else "Advisory result — purchasing still requires human approval")
        st.markdown("**Required approvals**")
        st.text(" · ".join(decision.required_approvals) or "No approval route identified — ask Procurement")
        if decision.missing_information:
            st.error("Required information is missing")
            st.text("\n".join(decision.missing_information))
        if decision.risk_flags:
            st.markdown("**Risk flags / review considerations**")
            st.text("\n".join(decision.risk_flags))
        st.markdown("**Next step**")
        st.text(decision.next_step)
        st.markdown("**Evidence and provenance**")
        st.dataframe([
            {"Source": item.source, "Finding": item.finding, "Reference": item.reference or "Not provided"}
            for item in decision.evidence
        ], hide_index=True)
        with st.expander("Run telemetry"):
            telemetry = decision.telemetry
            if telemetry is None:
                st.caption("Telemetry not reported")
            else:
                rows = [
                    {"Metric": label, "Value": str(getattr(telemetry, name)) if getattr(telemetry, name) is not None else "Not reported"}
                    for label, name in (("Actual LLM HTTP attempts", "llm_calls"),
                                        ("Logical LLM calls", "logical_llm_calls"),
                                        ("Tool calls", "tool_calls"), ("Prompt tokens", "prompt_tokens"),
                                        ("Completion tokens", "completion_tokens"))
                ]
                if telemetry.cached_tokens is not None:
                    rows.append({"Metric": "Cached prompt tokens", "Value": str(telemetry.cached_tokens)})
                st.dataframe(rows, hide_index=True)
                st.text("Tools: " + ", ".join(telemetry.tool_names))
                st.caption("Token usage is provider-reported; missing usage is not assumed to be zero.")
        with st.expander("Decision JSON (debug)"):
            st.json(decision.model_dump(mode="json"))
    elif not run:
        st.caption("Choose a request and architecture, then select Run Analysis. Analysis runs only when requested.")
