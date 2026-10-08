from __future__ import annotations

import unittest

from src.provider import LLMProvider, ModelMessage, ModelRequest, ModelResponse, ToolCall, ToolDefinition


class FakeProvider:
    def __init__(self):
        self.requests = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if request.messages[-1].role == "tool":
            return ModelResponse(content='{"next_step":"Human review"}', finish_reason="stop")
        return ModelResponse(tool_calls=(ToolCall("call-1", "budget", {"department": "Finance"}),))


class ProviderInterfaceTests(unittest.TestCase):
    def test_provider_neutral_tool_roundtrip_with_fake(self):
        provider: LLMProvider = FakeProvider()
        tools = (ToolDefinition("budget", "Read available budget", {"type": "object"}),)
        messages = (ModelMessage("system", "Recommendations only"), ModelMessage("user", "Inspect request"))
        first = provider.complete(ModelRequest(messages, tools))
        call = first.tool_calls[0]
        result = provider.complete(ModelRequest(
            messages + (ModelMessage("assistant", tool_calls=first.tool_calls),
                        ModelMessage("tool", '{"available_usd":29000}', tool_call_id=call.call_id)),
            tools, {"type": "object"},
        ))
        self.assertEqual(call.arguments, {"department": "Finance"})
        self.assertEqual(provider.requests[-1].messages[-1].tool_call_id, "call-1")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIn("Human review", result.content)


if __name__ == "__main__":
    unittest.main()
