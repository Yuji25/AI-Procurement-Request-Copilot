"""Provider/runtime settings only. No business-policy values belong here.

src package initialization loads .env without overriding the OS environment.
Pass an explicit mapping to load settings without consulting the environment.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from collections.abc import Mapping
from urllib.parse import urlsplit


def _url(value: str, name: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{name} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{name} must not contain credentials")
    return value.rstrip("/")


@dataclass(frozen=True)
class RuntimeConfig:
    llm_provider: str = "openai-compatible"
    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 1024
    llm_reasoning_effort: str | None = None
    vendor_risk_api_url: str = "http://127.0.0.1:8001"
    agent_max_tool_calls: int = 24

    def __post_init__(self) -> None:
        if not self.llm_provider.strip():
            raise ValueError("LLM_PROVIDER must not be empty")
        if not math.isfinite(self.llm_timeout_seconds) or self.llm_timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be finite and positive")
        if type(self.llm_max_retries) is not int or self.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES must be a nonnegative integer")
        if not math.isfinite(self.llm_temperature) or not 0 <= self.llm_temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if type(self.llm_max_output_tokens) is not int or self.llm_max_output_tokens <= 0:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be a positive integer")
        if self.llm_base_url:
            _url(self.llm_base_url, "LLM_BASE_URL")
        _url(self.vendor_risk_api_url, "VENDOR_RISK_API_URL")
        if self.llm_reasoning_effort is not None and self.llm_reasoning_effort not in {
            "none", "minimal", "low", "medium", "high", "xhigh", "default"
        }:
            raise ValueError("LLM_REASONING_EFFORT has an unsupported value")
        if type(self.agent_max_tool_calls) is not int or not 6 <= self.agent_max_tool_calls <= 40:
            raise ValueError("AGENT_MAX_TOOL_CALLS must be an integer between 6 and 40")

    def validate_llm_ready(self) -> None:
        """Called by a future adapter, not by offline business checks."""
        for name, value in (("LLM_API_KEY", self.llm_api_key),
                            ("LLM_BASE_URL", self.llm_base_url),
                            ("LLM_MODEL", self.llm_model)):
            if not value.strip():
                raise ValueError(f"{name} is required before calling an LLM")


def load_vendor_risk_api_url(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the service URL independently of LLM readiness/settings."""
    env = os.environ if environ is None else environ
    value = (env.get("VENDOR_RISK_API_URL", "").strip()
             or env.get("VENDOR_RISK_BASE_URL", "").strip()
             or "http://127.0.0.1:8001")
    return _url(value, "VENDOR_RISK_API_URL")


def load_runtime_config(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    env = os.environ if environ is None else environ

    def text(name: str, default: str = "") -> str:
        return env.get(name, default).strip()

    def number(name: str, default: str, cast: type) -> int | float:
        try:
            return cast(text(name, default))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} has an invalid numeric value") from exc

    return RuntimeConfig(
        llm_provider=text("LLM_PROVIDER", "openai-compatible"),
        llm_api_key=text("LLM_API_KEY"),
        llm_base_url=text("LLM_BASE_URL").rstrip("/"),
        llm_model=text("LLM_MODEL"),
        llm_timeout_seconds=number("LLM_TIMEOUT_SECONDS", "30", float),
        llm_max_retries=number("LLM_MAX_RETRIES", "2", int),
        llm_temperature=number("LLM_TEMPERATURE", "0", float),
        llm_max_output_tokens=number("LLM_MAX_OUTPUT_TOKENS", "1024", int),
        llm_reasoning_effort=text("LLM_REASONING_EFFORT", "low") or None,
        vendor_risk_api_url=load_vendor_risk_api_url(env),
        agent_max_tool_calls=number("AGENT_MAX_TOOL_CALLS", "24", int),
    )
