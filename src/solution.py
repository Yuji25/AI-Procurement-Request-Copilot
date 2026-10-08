from __future__ import annotations

from src.contracts import Architecture, ProcurementDecision


def handle_request(request_id: str, architecture: Architecture = "single") -> ProcurementDecision:
    """Assessment adapter.

    Keep this function callable by the public/hidden evaluation harness.
    Your internal implementation may use any framework, modules, agents, tools,
    deterministic checks, or orchestration strategy.
    """
    if architecture not in {"single", "staged"}:
        raise ValueError("Unknown architecture")
    from src.config import load_runtime_config
    from src.single_agent import run_single
    from src.staged_agent import run_staged

    runner = run_single if architecture == "single" else run_staged

    try:
        config = load_runtime_config()
    except ValueError:
        # Both runners surface provider configuration failure as manual review.
        return runner(request_id)
    return runner(request_id, max_tool_calls=config.agent_max_tool_calls)
