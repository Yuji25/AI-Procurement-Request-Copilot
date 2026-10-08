from __future__ import annotations

from src.contracts import Architecture, ProcurementDecision


def handle_request(request_id: str, architecture: Architecture = "single") -> ProcurementDecision:
    """Assessment adapter.

    Keep this function callable by the public/hidden evaluation harness.
    Your internal implementation may use any framework, modules, agents, tools,
    deterministic checks, or orchestration strategy.
    """
    if architecture == "staged":
        raise NotImplementedError("Architecture B ('staged') is not implemented yet.")
    if architecture != "single":
        raise ValueError("Unknown architecture")
    from src.config import load_runtime_config
    from src.single_agent import run_single

    try:
        config = load_runtime_config()
    except ValueError:
        # run_single surfaces provider configuration failure as manual review.
        return run_single(request_id)
    return run_single(request_id, max_tool_calls=config.agent_max_tool_calls)
