from __future__ import annotations

import io
import json
import logging
import traceback
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import Mock, patch

import requests

from src.config import RuntimeConfig
from src.openai_compatible import OpenAICompatibleProvider
from src.provider import (
    ModelMessage, ModelRequest, ToolCall, ToolDefinition, ModelResponse,
    ProviderAuthenticationError, ProviderConfigurationError, ProviderRateLimitError,
    ProviderRequestError, ProviderResponseError, ProviderTimeoutError, ProviderUpstreamError,
)
from src.provider_smoke import main as smoke_main

KEY = "offline-secret-do-not-expose"
CONFIG = RuntimeConfig(llm_api_key=KEY, llm_base_url="https://provider.example/v1",
                       llm_model="configured-model", llm_timeout_seconds=4.5,
                       llm_max_retries=0, llm_temperature=0.2, llm_max_output_tokens=99)
REQUEST = ModelRequest((ModelMessage("user", "Say OK"),))


def payload(content="OK", calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                      "completion_time": 0.01}}


def call(call_id="call-1", name="lookup", arguments='{"department":"Finance"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def response(status=200, body=None, retry_after=None):
    result = Mock(status_code=status, headers={} if retry_after is None else {"Retry-After": retry_after})
    result.json.return_value = payload() if body is None else body
    return result


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock(spec=requests.Session)
        self.session.post.return_value = response()
        self.provider = OpenAICompatibleProvider(CONFIG, session=self.session)
        self.sleep = patch("src.openai_compatible.time.sleep").start()
        self.addCleanup(patch.stopall)

    def test_successful_text_and_runtime_request_settings(self):
        result = self.provider.complete(REQUEST)
        self.assertEqual(result.content, "OK")
        self.assertEqual(result.tool_calls, ())
        self.assertEqual(result.usage, {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})
        args, kwargs = self.session.post.call_args
        self.assertEqual(args, ("https://provider.example/v1/chat/completions",))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer " + KEY)
        self.assertEqual(kwargs["timeout"], 4.5)
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["json"], {
            "model": "configured-model", "messages": [{"role": "user", "content": "Say OK"}],
            "temperature": 0.2, "max_tokens": 99, "stream": False,
        })
        self.session.post.return_value.close.assert_called_once()

    def test_tool_response(self):
        self.session.post.return_value = response(body=payload(None, [call()], "tool_calls"))
        result = self.provider.complete(REQUEST)
        self.assertEqual(result.tool_calls, (ToolCall("call-1", "lookup", {"department": "Finance"}),))
        self.assertIsNone(result.content)

    def test_optional_reasoning_effort(self):
        self.provider.complete(REQUEST)
        self.assertNotIn("reasoning_effort", self.session.post.call_args.kwargs["json"])
        provider = OpenAICompatibleProvider(replace(CONFIG, llm_reasoning_effort="low"), session=self.session)
        provider.complete(REQUEST)
        self.assertEqual(self.session.post.call_args.kwargs["json"]["reasoning_effort"], "low")

    def test_request_output_limit_preserves_global_default_and_ceiling(self):
        provider = OpenAICompatibleProvider(replace(CONFIG, llm_max_output_tokens=1024), session=self.session)
        provider.complete(ModelRequest(REQUEST.messages, max_output_tokens=256))
        self.assertEqual(self.session.post.call_args.kwargs["json"]["max_tokens"], 256)
        provider.complete(REQUEST)
        self.assertEqual(self.session.post.call_args.kwargs["json"]["max_tokens"], 1024)
        self.provider.complete(ModelRequest(REQUEST.messages, max_output_tokens=256))
        self.assertEqual(self.session.post.call_args.kwargs["json"]["max_tokens"], 99)

    def test_request_retry_limit_zero_stops_rate_limit_after_one_attempt(self):
        provider = OpenAICompatibleProvider(replace(CONFIG, llm_max_retries=2), session=self.session)
        self.session.post.return_value = response(status=429)
        with self.assertRaises(ProviderRateLimitError) as caught:
            provider.complete(ModelRequest(REQUEST.messages, max_retries=0))
        self.assertEqual(caught.exception.attempts, 1)
        self.session.post.assert_called_once()
        self.sleep.assert_not_called()

    def test_invalid_request_overrides_fail_before_http(self):
        for kwargs in ({"max_output_tokens": 0}, {"max_output_tokens": True},
                       {"max_retries": -1}, {"max_retries": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ProviderRequestError):
                self.provider.complete(ModelRequest(REQUEST.messages, **kwargs))
        self.session.post.assert_not_called()

    def test_cached_token_usage(self):
        body = payload()
        body["usage"]["prompt_tokens_details"] = {"cached_tokens": 2}
        self.session.post.return_value = response(body=body)
        self.assertEqual(self.provider.complete(REQUEST).usage["cached_tokens"], 2)
        for details in ([], {"cached_tokens": -1}, {"cached_tokens": True}):
            body["usage"]["prompt_tokens_details"] = details
            self.session.post.return_value = response(body=body)
            with self.subTest(details=details), self.assertRaises(ProviderResponseError):
                self.provider.complete(REQUEST)

    def test_multiple_tool_calls(self):
        self.session.post.return_value = response(body=payload("Checking", [call(), call("call-2", "vendor", '{"name":"Vendor"}')], "tool_calls"))
        result = self.provider.complete(REQUEST)
        self.assertEqual([c.call_id for c in result.tool_calls], ["call-1", "call-2"])
        self.assertEqual(result.tool_calls[1].arguments, {"name": "Vendor"})

    def test_assistant_and_tool_message_roundtrip(self):
        self.session.post.return_value = response(body=payload(None, [call()], "tool_calls"))
        first = self.provider.complete(REQUEST)
        tools = (ToolDefinition("lookup", "Read budget", {"type": "object"}),)
        self.provider.complete(ModelRequest(REQUEST.messages + (
            ModelMessage("assistant", tool_calls=first.tool_calls),
            ModelMessage("tool", '{"available":29000}', tool_call_id="call-1")), tools))
        sent = self.session.post.call_args.kwargs["json"]
        self.assertEqual(sent["messages"][1]["tool_calls"][0], call(arguments=json.dumps({"department": "Finance"})))
        self.assertEqual(sent["messages"][2], {"role": "tool", "content": '{"available":29000}', "tool_call_id": "call-1"})
        self.assertEqual(sent["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(sent["tool_choice"], "auto")

    def test_optional_schema_serialization(self):
        schema = {"type": "object", "properties": {"next_step": {"type": "string"}}}
        self.provider.complete(ModelRequest(REQUEST.messages, output_schema=schema))
        self.assertEqual(self.session.post.call_args.kwargs["json"]["response_format"], {
            "type": "json_schema", "json_schema": {"name": "model_output", "schema": schema}})

    def test_invalid_request_fails_before_http(self):
        for req in (ModelRequest(()), ModelRequest((ModelMessage("tool", "result"),)),
                    ModelRequest((ModelMessage("user", tool_calls=(ToolCall("id", "x", {}),)),)),
                    ModelRequest((ModelMessage("assistant", tool_calls=(ToolCall("id", "x", {"n": float("nan")}),)),))):
            with self.subTest(request=req), self.assertRaises(ProviderRequestError):
                self.provider.complete(req)
        self.session.post.assert_not_called()

    def test_invalid_response_json_not_retried(self):
        self.provider = OpenAICompatibleProvider(replace(CONFIG, llm_max_retries=2), session=self.session)
        self.session.post.return_value.json.side_effect = ValueError(KEY)
        with self.assertRaises(ProviderResponseError) as caught:
            self.provider.complete(REQUEST)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(self.session.post.call_count, 1)
        self.assertNotIn(KEY, str(caught.exception))

    def test_malformed_provider_payloads(self):
        bodies = [[], {}, {"choices": []}, {"choices": [None]},
                  {"choices": [{"message": {"role": "user", "content": "OK"}}]},
                  payload(None), payload(42), payload("OK", {}), payload("OK", "bad"),
                  payload(None, [call(arguments="bad")]), payload(None, [call(arguments="[]")]),
                  payload(None, [call(arguments='{"x":NaN}')]), payload(None, [call(arguments='{"x":1e999}')]),
                  payload(None, [call(), call()]), payload("OK", finish={})]
        bad_usage = payload()
        bad_usage["usage"]["prompt_tokens"] = True
        bodies.append(bad_usage)
        for body in bodies:
            with self.subTest(body=body):
                self.session.post.return_value = response(body=body)
                with self.assertRaises(ProviderResponseError):
                    self.provider.complete(REQUEST)

    def test_timeout(self):
        self.session.post.side_effect = requests.Timeout(KEY)
        with self.assertRaises(ProviderTimeoutError) as caught:
            self.provider.complete(REQUEST)
        self.assertEqual(caught.exception.attempts, 1)

    def test_empty_token_exhaustion_is_explicit_and_not_retried(self):
        provider = OpenAICompatibleProvider(replace(CONFIG, llm_max_retries=2), session=self.session)
        self.session.post.return_value = response(body=payload(None, finish="length"))
        with self.assertRaisesRegex(ProviderResponseError, "output-token limit"):
            provider.complete(REQUEST)
        self.assertEqual(self.session.post.call_count, 1)

    def test_text_with_length_finish_preserves_truncation_indicator(self):
        self.session.post.return_value = response(body=payload("Partial text", finish="length"))
        self.assertEqual(self.provider.complete(REQUEST).finish_reason, "length")

    def test_authentication_never_retried(self):
        provider = OpenAICompatibleProvider(replace(CONFIG, llm_max_retries=2), session=self.session)
        for status in (401, 403):
            self.session.post.reset_mock()
            self.session.post.return_value = response(status=status, body={"error": KEY})
            with self.subTest(status=status), self.assertRaises(ProviderAuthenticationError) as caught:
                provider.complete(REQUEST)
            self.assertEqual(caught.exception.status_code, status)
            self.assertEqual(self.session.post.call_count, 1)
            self.session.post.return_value.json.assert_not_called()

    def test_rate_limit(self):
        self.session.post.return_value = response(status=429)
        with self.assertRaises(ProviderRateLimitError) as caught:
            self.provider.complete(REQUEST)
        self.assertEqual(caught.exception.status_code, 429)

    def test_transient_failures_retry_then_succeed(self):
        provider = OpenAICompatibleProvider(replace(CONFIG, llm_max_retries=1), session=self.session)
        for first in (requests.Timeout(KEY), requests.ConnectionError(KEY), response(status=429, retry_after="1"),
                      response(status=500), response(status=502), response(status=503), response(status=504), response(status=408)):
            self.session.post.reset_mock()
            self.sleep.reset_mock()
            self.session.post.side_effect = [first, response()]
            with self.subTest(first=type(first).__name__):
                result = provider.complete(REQUEST)
                self.assertEqual(result.content, "OK")
                self.assertEqual(result.attempts, 2)
                self.assertEqual(self.session.post.call_count, 2)
                self.sleep.assert_called_once()

    def test_retry_limit_means_additional_attempts(self):
        provider = OpenAICompatibleProvider(replace(CONFIG, llm_max_retries=2), session=self.session)
        self.session.post.side_effect = requests.Timeout(KEY)
        with self.assertRaises(ProviderTimeoutError) as caught:
            provider.complete(REQUEST)
        self.assertEqual(caught.exception.attempts, 3)
        self.assertEqual(self.session.post.call_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [0.5, 1.0])

    def test_retry_after_is_respected_or_stops_early(self):
        provider = OpenAICompatibleProvider(replace(CONFIG, llm_max_retries=1), session=self.session)
        self.session.post.side_effect = [response(status=429, retry_after="2"), response()]
        provider.complete(REQUEST)
        self.sleep.assert_called_once_with(2.0)
        self.session.post.reset_mock()
        self.session.post.side_effect = None
        self.session.post.return_value = response(status=429, retry_after="120")
        with self.assertRaises(ProviderRateLimitError):
            provider.complete(REQUEST)
        self.assertEqual(self.session.post.call_count, 1)

    def test_retry_after_http_date_and_invalid_header(self):
        self.assertEqual(OpenAICompatibleProvider._retry_delay(1, "not a date"), 0.5)
        self.assertIsNone(OpenAICompatibleProvider._retry_delay(1, "Wed, 01 Jan 2099 00:00:00 GMT"))

    def test_generic_upstream_and_redirects_do_not_retry(self):
        provider = OpenAICompatibleProvider(replace(CONFIG, llm_max_retries=2), session=self.session)
        for status in (400, 404, 302, 501):
            self.session.post.reset_mock()
            self.session.post.return_value = response(status=status)
            with self.subTest(status=status), self.assertRaises(ProviderUpstreamError):
                provider.complete(REQUEST)
            self.assertEqual(self.session.post.call_count, 1)

    def test_key_absent_from_repr_errors_tracebacks_and_logs(self):
        self.assertNotIn(KEY, repr(CONFIG))
        self.assertNotIn(KEY, repr(self.provider))
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logging.getLogger().addHandler(handler)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        for problem in (requests.Timeout(KEY), requests.ConnectionError(KEY), response(status=401, body={"error": KEY})):
            self.session.post.side_effect = problem if isinstance(problem, Exception) else None
            self.session.post.return_value = problem
            try:
                self.provider.complete(REQUEST)
            except (ProviderTimeoutError, ProviderUpstreamError, ProviderAuthenticationError) as exc:
                self.assertNotIn(KEY, str(exc))
                self.assertNotIn(KEY, repr(exc))
                self.assertNotIn(KEY, traceback.format_exc())
        self.assertNotIn(KEY, stream.getvalue())

    def test_owned_session_cleanup_and_injected_session_ownership(self):
        with patch("src.openai_compatible.requests.Session") as factory:
            with OpenAICompatibleProvider(CONFIG):
                pass
            factory.return_value.close.assert_called_once()
        self.provider.close()
        self.session.close.assert_not_called()


class ConfigurationAndSmokeTests(unittest.TestCase):
    def test_readiness_errors_are_sanitized_and_no_http_session_created(self):
        configs = [replace(CONFIG, llm_api_key=""), replace(CONFIG, llm_model=""),
                   replace(CONFIG, llm_base_url=""), replace(CONFIG, llm_provider="unsupported"),
                   replace(CONFIG, llm_base_url="https://example.com/v1?secret=" + KEY)]
        with patch("src.openai_compatible.requests.Session") as factory:
            for config in configs:
                with self.subTest(config=config), self.assertRaises(ProviderConfigurationError) as caught:
                    OpenAICompatibleProvider(config)
                self.assertNotIn(KEY, str(caught.exception))
            factory.assert_not_called()

    def test_environment_config_errors_are_wrapped(self):
        with patch("src.openai_compatible.load_runtime_config", side_effect=ValueError(KEY)):
            with self.assertRaises(ProviderConfigurationError) as caught:
                OpenAICompatibleProvider()
            self.assertNotIn(KEY, str(caught.exception))

    def test_manual_smoke_uses_one_attempt_and_small_output_without_printing_response(self):
        output = io.StringIO()
        with patch("src.provider_smoke.load_runtime_config", return_value=CONFIG), \
             patch("src.provider_smoke.OpenAICompatibleProvider") as factory, redirect_stdout(output):
            factory.return_value.__enter__.return_value.complete.return_value = ModelResponse(content="OK " + KEY)
            self.assertEqual(smoke_main(), 0)
            configured = factory.call_args.args[0]
            self.assertEqual(configured.llm_max_retries, 0)
            self.assertEqual(configured.llm_max_output_tokens, 99)
        self.assertIn("SMOKE PASSED", output.getvalue())
        self.assertNotIn(KEY, output.getvalue())

    def test_manual_smoke_failure_is_sanitized(self):
        output = io.StringIO()
        with patch("src.provider_smoke.load_runtime_config", side_effect=ValueError(KEY)), redirect_stdout(output):
            self.assertEqual(smoke_main(), 1)
        self.assertIn("ProviderConfigurationError", output.getvalue())
        self.assertNotIn(KEY, output.getvalue())


if __name__ == "__main__":
    unittest.main()
