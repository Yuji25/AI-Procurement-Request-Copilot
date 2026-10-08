"""Synchronous, non-streaming Chat Completions transport. No agent behavior.

Uses max_tokens for broad compatible-endpoint support. Optional output_schema
maps to json_schema response_format; the configured model must support it.
No raw requests, response bodies, credentials, or transport exceptions are logged.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import requests

from src.config import RuntimeConfig, load_runtime_config
from src.provider import (
    ModelRequest, ModelResponse, ToolCall, ProviderError,
    ProviderConfigurationError, ProviderRequestError, ProviderTimeoutError,
    ProviderRateLimitError, ProviderAuthenticationError, ProviderResponseError,
    ProviderUpstreamError,
)


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant")


class OpenAICompatibleProvider:
    def __init__(self, config: RuntimeConfig | None = None, *, session: requests.Session | None = None):
        try:
            self._config = config if config is not None else load_runtime_config()
            self._config.validate_llm_ready()
            if self._config.llm_provider.lower() not in {"openai-compatible", "openai", "groq"}:
                raise ValueError("Unsupported provider selector")
            url = urlsplit(self._config.llm_base_url)
            _ = url.port  # Validate malformed ports before attaching credentials.
            if url.query or url.fragment:
                raise ValueError("Base URL cannot contain query parameters or fragments")
            self._endpoint = self._config.llm_base_url.rstrip("/") + "/chat/completions"
        except ValueError:
            raise ProviderConfigurationError("Invalid or incomplete compatible-provider configuration.") from None
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> OpenAICompatibleProvider:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _payload(self, request: ModelRequest) -> dict:
        try:
            if not request.messages:
                raise ValueError("Empty messages")
            messages = []
            for message in request.messages:
                if message.role not in {"system", "user", "assistant", "tool"}:
                    raise ValueError("Invalid role")
                if message.content is not None and not isinstance(message.content, str):
                    raise ValueError("Invalid content")
                item = {"role": message.role, "content": message.content}
                if message.tool_calls:
                    if message.role != "assistant":
                        raise ValueError("Tool calls require assistant role")
                    calls = []
                    for call in message.tool_calls:
                        if not _nonempty(call.call_id) or not _nonempty(call.name) or not isinstance(call.arguments, dict):
                            raise ValueError("Invalid tool call")
                        calls.append({"id": call.call_id, "type": "function", "function": {
                            "name": call.name, "arguments": json.dumps(call.arguments, allow_nan=False)}})
                    item["tool_calls"] = calls
                if message.role == "tool":
                    if not _nonempty(message.tool_call_id) or message.content is None:
                        raise ValueError("Tool result requires call ID and content")
                    item["tool_call_id"] = message.tool_call_id
                elif message.tool_call_id is not None:
                    raise ValueError("Unexpected tool call ID")
                if message.content is None and not message.tool_calls:
                    raise ValueError("Message has no content or calls")
                messages.append(item)
            body = {
                "model": self._config.llm_model, "messages": messages,
                "temperature": self._config.llm_temperature,
                "max_tokens": self._config.llm_max_output_tokens, "stream": False,
            }
            if request.tools:
                tools = []
                for tool in request.tools:
                    if not _nonempty(tool.name) or not isinstance(tool.parameters, dict):
                        raise ValueError("Invalid tool definition")
                    tools.append({"type": "function", "function": {
                        "name": tool.name, "description": tool.description, "parameters": tool.parameters}})
                body["tools"] = tools
                body["tool_choice"] = "auto"
            if request.output_schema is not None:
                if not isinstance(request.output_schema, dict):
                    raise ValueError("Invalid schema")
                body["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "model_output", "schema": request.output_schema}}
            json.dumps(body, allow_nan=False)
            return body
        except (ValueError, TypeError, AttributeError):
            raise ProviderRequestError("Invalid provider-neutral model request.") from None

    @staticmethod
    def _parse(payload: object, attempt: int) -> ModelResponse:
        try:
            if not isinstance(payload, dict):
                raise ValueError("Invalid payload")
            choices = payload.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError("Expected one choice")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("Invalid assistant message")
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("Invalid content")
            raw_calls = message.get("tool_calls")
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list):
                raise ValueError("Invalid tool calls")
            calls, seen = [], set()
            for raw in raw_calls:
                if not isinstance(raw, dict) or raw.get("type") != "function":
                    raise ValueError("Invalid call")
                function = raw.get("function")
                call_id = raw.get("id")
                if not isinstance(function, dict) or not _nonempty(call_id) or call_id in seen:
                    raise ValueError("Invalid function/call ID")
                if not _nonempty(function.get("name")) or not isinstance(function.get("arguments"), str):
                    raise ValueError("Invalid function fields")
                arguments = json.loads(function["arguments"], parse_constant=_reject_constant)
                if not isinstance(arguments, dict):
                    raise ValueError("Arguments must be an object")
                json.dumps(arguments, allow_nan=False)
                calls.append(ToolCall(call_id, function["name"], arguments))
                seen.add(call_id)
            finish = choice.get("finish_reason")
            if finish is not None and not isinstance(finish, str):
                raise ValueError("Invalid finish reason")
            if not _nonempty(content) and not calls:
                if finish == "length":
                    raise ProviderResponseError(
                        "Provider exhausted the output-token limit before returning assistant text.",
                        attempts=attempt)
                raise ValueError("Empty response")
            raw_usage = payload.get("usage")
            if raw_usage is not None and not isinstance(raw_usage, dict):
                raise ValueError("Invalid usage")
            usage = {}
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if raw_usage is not None and name in raw_usage:
                    count = raw_usage[name]
                    if type(count) is not int or count < 0:
                        raise ValueError("Invalid usage count")
                    usage[name] = count
            return ModelResponse(content, tuple(calls), finish, usage, attempts=attempt)
        except (ValueError, TypeError, KeyError):
            raise ProviderResponseError("Malformed provider response.", attempts=attempt) from None

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None) -> float | None:
        """Respect Retry-After; do not retry early when it exceeds 30 seconds."""
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    delay = -1
            if math.isfinite(delay) and delay >= 0:
                return delay if delay <= 30 else None
        return 0.5 * 2 ** min(attempt - 1, 4)

    def complete(self, request: ModelRequest) -> ModelResponse:
        body = self._payload(request)
        for attempt in range(1, self._config.llm_max_retries + 2):
            retry_after = None
            retryable = False
            try:
                response = self._session.post(
                    self._endpoint, json=body,
                    headers={"Authorization": f"Bearer {self._config.llm_api_key}", "Content-Type": "application/json"},
                    timeout=self._config.llm_timeout_seconds, allow_redirects=False,
                )
            except requests.Timeout:
                error = ProviderTimeoutError("Provider request timed out.", attempts=attempt)
                retryable = True
            except requests.RequestException:
                error = ProviderUpstreamError("Provider connection/transport failed.", attempts=attempt)
                retryable = True
            else:
                try:
                    status = response.status_code
                    if 200 <= status < 300:
                        try:
                            payload = response.json()
                        except ValueError:
                            raise ProviderResponseError("Provider returned invalid JSON.", attempts=attempt) from None
                        return self._parse(payload, attempt)
                    error_type = (ProviderAuthenticationError if status in {401, 403} else
                                  ProviderRateLimitError if status == 429 else
                                  ProviderTimeoutError if status == 408 else ProviderUpstreamError)
                    error = error_type(f"Provider HTTP request failed (status {status}).",
                                       attempts=attempt, status_code=status)
                    retryable = status in {408, 429, 500, 502, 503, 504}
                    retry_after = response.headers.get("Retry-After")
                finally:
                    response.close()
            delay = self._retry_delay(attempt, retry_after) if retryable else None
            if not retryable or attempt > self._config.llm_max_retries or delay is None:
                raise error from None
            time.sleep(delay)
        raise ProviderError("Provider request did not complete.")  # Defensive; loop always returns/raises.
