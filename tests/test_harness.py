import os
import tomllib
import unittest
from unittest.mock import patch

from lllm2 import harness


class HarnessTests(unittest.TestCase):
    def test_running_engine_metadata(self):
        with patch.object(
            harness.Engine,
            "request",
            side_effect=[
                {"data": [{"id": "served-model"}]},
                {"default_generation_settings": {"n_ctx": 16384}, "total_slots": 4},
            ],
        ):
            base, model, window, slots = harness.served_model()
        self.assertEqual((model, window, slots), ("served-model", 16384, 4))
        self.assertTrue(base.startswith("http://127.0.0.1:"))

    def test_unavailable_or_invalid_engine(self):
        with patch.object(harness.Engine, "request", side_effect=OSError("refused")):
            with self.assertRaisesRegex(RuntimeError, "Start a model"):
                harness.served_model()
        with patch.object(harness.Engine, "request", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "did not report"):
                harness.served_model()

    def test_session_configuration(self):
        model = 'model "quoted" \\ path'

        def call(argv, env):
            self.assertEqual(argv[-2:], ["-p", "prompt with spaces"])
            self.assertEqual(env["PRESERVED"], "yes")
            name = os.path.basename(argv[0])
            if name == "claude":
                self.assertNotIn("ANTHROPIC_API_KEY", env)
                self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:1920")
                for key in (
                    "ANTHROPIC_MODEL",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                    "CLAUDE_CODE_SUBAGENT_MODEL",
                ):
                    self.assertEqual(env[key], model)
                self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "16384")
                self.assertEqual(env["CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS"], "3")
            elif name == "codex":
                config = tomllib.loads("\n".join(argv[2:-2:2]))
                self.assertEqual(config["model"], model)
                provider = config["model_providers"]["lllm2"]
                self.assertEqual(provider["base_url"], "http://127.0.0.1:1920/v1")
                self.assertEqual(provider["wire_api"], "responses")
                self.assertFalse(provider["requires_openai_auth"])
                self.assertEqual(config["model_context_window"], 16384)
            return 17

        with (
            patch.dict(
                os.environ, {"ANTHROPIC_API_KEY": "existing", "PRESERVED": "yes"}
            ),
            patch.object(
                harness.shutil, "which", side_effect=lambda name: "/bin/" + name
            ),
            patch.object(
                harness,
                "served_model",
                return_value=("http://127.0.0.1:1920", model, 16384, 4),
            ),
            patch.object(harness.subprocess, "call", side_effect=call),
        ):
            for name in ("claude", "codex"):
                self.assertEqual(
                    harness.run_harness(name, ["-p", "prompt with spaces"]), 17
                )
                self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "existing")

    def test_missing_cli_and_signal_exit(self):
        with (
            patch.object(harness.shutil, "which", return_value=None),
            patch.object(harness, "served_model") as model,
        ):
            with self.assertRaisesRegex(RuntimeError, "not on PATH"):
                harness.run_harness("codex", [])
            model.assert_not_called()
        with (
            patch.object(harness.shutil, "which", return_value="/bin/codex"),
            patch.object(
                harness, "served_model", return_value=("http://localhost", "m", 4096, 1)
            ),
            patch.object(harness.subprocess, "call", return_value=-15),
        ):
            self.assertEqual(harness.run_harness("codex", []), 143)
