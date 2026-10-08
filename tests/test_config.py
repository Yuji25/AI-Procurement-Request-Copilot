from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config import load_runtime_config, load_vendor_risk_api_url


class RuntimeConfigTests(unittest.TestCase):
    def test_offline_defaults_do_not_require_credentials(self):
        config = load_runtime_config({})
        self.assertEqual(config.llm_provider, "openai-compatible")
        self.assertEqual(config.vendor_risk_api_url, "http://127.0.0.1:8001")
        with self.assertRaisesRegex(ValueError, "LLM_API_KEY"):
            config.validate_llm_ready()

    def test_all_runtime_values_are_replaceable(self):
        config = load_runtime_config({
            "LLM_PROVIDER": "groq", "LLM_API_KEY": "offline-test-secret",
            "LLM_BASE_URL": "https://api.groq.com/openai/v1/", "LLM_MODEL": "test-model",
            "LLM_TIMEOUT_SECONDS": "12.5", "LLM_MAX_RETRIES": "0",
            "LLM_TEMPERATURE": "0.3", "LLM_MAX_OUTPUT_TOKENS": "500",
            "VENDOR_RISK_API_URL": "http://localhost:9000/",
        })
        config.validate_llm_ready()
        self.assertEqual((config.llm_provider, config.llm_model), ("groq", "test-model"))
        self.assertEqual(config.llm_base_url, "https://api.groq.com/openai/v1")
        self.assertEqual((config.llm_timeout_seconds, config.llm_max_retries,
                          config.llm_temperature, config.llm_max_output_tokens), (12.5, 0, 0.3, 500))
        self.assertEqual(config.vendor_risk_api_url, "http://localhost:9000")
        self.assertNotIn("offline-test-secret", repr(config))

    def test_missing_endpoint_and_model_fail_only_at_readiness(self):
        for absent in ("LLM_BASE_URL", "LLM_MODEL"):
            env = {"LLM_API_KEY": "test", "LLM_BASE_URL": "https://example.com/v1", "LLM_MODEL": "test"}
            env.pop(absent)
            config = load_runtime_config(env)
            with self.assertRaisesRegex(ValueError, absent):
                config.validate_llm_ready()

    def test_invalid_numeric_settings(self):
        cases = {
            "LLM_TIMEOUT_SECONDS": ["0", "-1", "nan", "inf", "abc"],
            "LLM_MAX_RETRIES": ["-1", "1.5", ""],
            "LLM_TEMPERATURE": ["-0.1", "2.1", "nan", "inf"],
            "LLM_MAX_OUTPUT_TOKENS": ["0", "-1", "2.5"],
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    load_runtime_config({name: value})

    def test_invalid_url_and_empty_provider(self):
        for name, value in (("LLM_BASE_URL", "relative/path"),
                            ("VENDOR_RISK_API_URL", "ftp://example.com"),
                            ("LLM_BASE_URL", "https://user:secret@example.com"),
                            ("LLM_PROVIDER", " ")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                load_runtime_config({name: value})

    def test_new_vendor_url_takes_precedence_over_legacy(self):
        self.assertEqual(load_vendor_risk_api_url({"VENDOR_RISK_API_URL": "http://new:9000",
                                                  "VENDOR_RISK_BASE_URL": "http://old:8001"}), "http://new:9000")
        self.assertEqual(load_vendor_risk_api_url({"VENDOR_RISK_BASE_URL": "http://old:8001/"}), "http://old:8001")

    def test_vendor_settings_are_independent_of_llm_settings(self):
        self.assertEqual(load_vendor_risk_api_url({"LLM_TIMEOUT_SECONDS": "broken"}), "http://127.0.0.1:8001")

    def test_agent_limits_are_configurable_with_hard_caps(self):
        config = load_runtime_config({"AGENT_MAX_MODEL_TURNS": "2", "AGENT_MAX_TOOL_CALLS": "6"})
        self.assertEqual((config.agent_max_model_turns, config.agent_max_tool_calls), (2, 6))
        for name, values in (("AGENT_MAX_MODEL_TURNS", ("0", "9")), ("AGENT_MAX_TOOL_CALLS", ("5", "41"))):
            for value in values:
                with self.assertRaisesRegex(ValueError, name):
                    load_runtime_config({name: value})

    def test_os_environment_and_explicit_mapping(self):
        with patch.dict("os.environ", {"LLM_MODEL": "from-os"}, clear=True):
            self.assertEqual(load_runtime_config().llm_model, "from-os")
            self.assertEqual(load_runtime_config({}).llm_model, "")


if __name__ == "__main__":
    unittest.main()
