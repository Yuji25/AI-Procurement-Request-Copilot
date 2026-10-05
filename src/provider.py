"""Provider-neutral interface only: no HTTP, SDK, agent, or model execution.

A future compatible-API adapter owns RuntimeConfig and translates these types
into its provider's wire format. Business code must not depend on that format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one model turn; transport errors are adapter responsibilities."""
        ...


class ProviderError(Exception):
    """Sanitized transport failure; attempts counts actual HTTP attempts."""

    def __init__(self, message: str, *, attempts: int = 0, status_code: int | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class ProviderConfigurationError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderResponseError(ProviderError):
    pass


class ProviderUpstreamError(ProviderError):
    pass
