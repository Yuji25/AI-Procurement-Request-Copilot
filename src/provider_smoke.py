"""Manual only: python -m src.provider_smoke (one small live HTTP attempt)."""
from __future__ import annotations

from dataclasses import replace

from src.config import load_runtime_config
from src.openai_compatible import OpenAICompatibleProvider
from src.provider import ModelMessage, ModelRequest, ProviderConfigurationError, ProviderError


def main() -> int:
    try:
        try:
            configured = load_runtime_config()
            config = replace(configured, llm_max_retries=0,
                             llm_max_output_tokens=min(configured.llm_max_output_tokens, 256))
        except ValueError:
            raise ProviderConfigurationError("Invalid provider configuration.") from None
        with OpenAICompatibleProvider(config) as provider:
            response = provider.complete(ModelRequest((ModelMessage("user", "Reply with the word OK."),)))
        if not response.content or not response.content.strip() or response.tool_calls:
            print("SMOKE FAILED: expected a nonempty text assistant response.")
            return 1
        # Do not print raw model/provider data, URLs, headers, or credentials.
        print("SMOKE PASSED: configured model returned a text response (one HTTP attempt).")
        return 0
    except ProviderError as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
