"""Settings recovery uses temporary storage; never open the workstation database."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lllm2.app import App
from lllm2.discovery import host_memory
from lllm2.settings import Settings
from lllm2.store import Store


class MemoryTests(unittest.TestCase):
    def test_reclaimable_cache_is_available(self):
        with patch(
            "pathlib.Path.read_text",
            return_value="MemTotal: 8388608 kB\nMemFree: 1048576 kB\nMemAvailable: 6291456 kB\nCached: 5242880 kB\n",
        ):
            self.assertEqual(
                host_memory(), {"total_gib": 8, "used_gib": 2, "available_gib": 6}
            )

    def test_missing_invalid_or_unreadable_memory_is_unknown(self):
        for data in (
            "MemTotal: 8 kB\n",
            "MemTotal: 8 kB\nMemAvailable: 9 kB",
            "MemTotal: nonsense\nMemAvailable: 0 kB",
            "",
        ):
            with (
                self.subTest(data=data),
                patch("pathlib.Path.read_text", return_value=data),
            ):
                self.assertIsNone(host_memory()["used_gib"])
        with patch("pathlib.Path.read_text", side_effect=OSError):
            self.assertIsNone(host_memory()["total_gib"])


class SettingsRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch("lllm2.config.STATE_DIR", Path(self.temp.name)):
            self.app = App.__new__(App)
            self.app.store = Store()
        self.addCleanup(self.app.store.db.close)
        self.settings = Settings(
            model="/models/example.gguf", engine="/engine", gpu_layers=0
        )
        self.result = {
            "id": "measured",
            "status": "complete",
            "settings": self.settings.dict(),
            "samples": [{"workload": "generate"}],
            "probes": [],
            "recommended_context": 4096,
        }
        self.app.store.put("result", self.result["id"], self.result)

    def test_manual_save_preserves_recoverable_experiment_and_recommendation(self):
        original = self.app.store.get("result", "measured")
        with patch("lllm2.app.launch_args"):
            self.app.action("/api/default/save", {"result_id": "measured"})
            provenance = self.app.store.get(
                "default-evidence", self.app.default_key(self.settings)
            )
            self.assertEqual(provenance["kind"], "benchmark")
            self.app.action(
                "/api/default/save",
                {
                    "settings": Settings(
                        **{**self.settings.dict(), "context": 8192}
                    ).dict()
                },
            )
        saved = self.app.store.get("default", self.app.default_key(self.settings))
        preview = self.app.action("/api/result/preview", {"result_id": "measured"})
        self.assertEqual(preview["settings"]["gpu_layers"], 0)
        self.assertEqual(preview["evidence"]["result_id"], "measured")
        self.assertEqual(self.app.store.get("result", "measured"), original)
        self.assertEqual(
            self.app.store.get("default", self.app.default_key(self.settings)), saved
        )
        recommended = {
            "settings": Settings(gpu_layers=None).dict(),
            "source": "Estimated starting settings",
        }
        with patch("lllm2.app.choose_launch", return_value=recommended):
            self.assertEqual(
                self.app.action(
                    "/api/default/resolve",
                    {"settings": self.settings.dict(), "source": "built-in"},
                ),
                recommended,
            )
        restored = self.app.action(
            "/api/default/resolve",
            {"settings": self.settings.dict(), "source": "saved"},
        )
        self.assertEqual(restored["settings"]["context"], 8192)
        self.assertEqual(restored["settings"]["gpu_layers"], 0)
        self.assertEqual(
            self.app.store.get("default", self.app.default_key(self.settings)), saved
        )

    def test_recommendation_switches_backend_but_saved_defaults_keep_selection(self):
        for backend, context in [("CUDA", 8192), ("Vulkan", 4096)]:
            selected = Settings(
                model=self.settings.model,
                engine="/chosen",
                device=backend + "1",
                backend=backend,
            )
            saved = Settings(
                model=self.settings.model,
                engine="/old",
                device=backend + "0",
                backend=backend,
                context=context,
            )
            self.app.store.put("default", self.app.default_key(selected), saved.dict())
            restored = self.app.action(
                "/api/default/resolve", {"settings": selected.dict(), "source": "saved"}
            )
            self.assertEqual(restored["settings"]["context"], context)
            self.assertEqual(restored["settings"]["engine"], "/chosen")
            self.assertEqual(restored["settings"]["device"], backend + "1")
        recommended = {"settings": self.settings.dict(), "source": "Measured CUDA"}
        with patch("lllm2.app.choose_launch", return_value=recommended) as resolve:
            result = self.app.action(
                "/api/default/resolve",
                {"settings": selected.dict(), "source": "built-in"},
            )
        resolve.assert_called_once_with(model_path=self.settings.model)
        self.assertEqual(result["settings"]["backend"], "CUDA")
        self.assertEqual(
            self.app.store.get("default", self.app.default_key(selected)), saved.dict()
        )

    def test_load_and_save_tested_context_with_optional_headroom(self):
        result = {
            **self.result,
            "settings": {**self.settings.dict(), "slots": 2, "context": 8192},
            "largest_observed_context": 16384,
            "recommended_context": 14592,
        }
        self.app.store.put("result", "measured", result)
        for use_context, headroom, expected, choice in (
            (False, False, 8192, "original"),
            (True, False, 32768, "tested"),
            (True, True, 29184, "headroom"),
        ):
            data = {
                "result_id": "measured",
                "use_context": use_context,
                "reserve_headroom": headroom,
            }
            with self.subTest(choice=choice):
                preview = self.app.action("/api/result/preview", data)
                self.assertEqual(preview["settings"]["context"], expected)
                self.assertEqual(preview["reserve_headroom"], use_context and headroom)
                with patch("lllm2.app.launch_args"):
                    saved = self.app.action("/api/default/save", data)
                self.assertEqual(saved["context"], expected)
                evidence = self.app.store.get(
                    "default-evidence", self.app.default_key(self.settings)
                )
                self.assertEqual(evidence["context"]["loaded_context"], choice)
                self.assertEqual(
                    evidence["context"]["used_headroom_estimate"],
                    use_context and headroom,
                )
                self.assertEqual(self.app.store.get("result", "measured"), result)

    def test_tested_context_does_not_silently_use_headroom(self):
        self.app.store.put(
            "result", "measured", {**self.result, "largest_observed_context": None}
        )
        for route in ("/api/result/preview", "/api/default/save"):
            with (
                self.subTest(route=route),
                self.assertRaisesRegex(ValueError, "context measurement"),
            ):
                self.app.action(route, {"result_id": "measured", "use_context": True})

    def test_discovery_only_result_promotes_its_loaded_context(self):
        result = {
            **self.result,
            "samples": [],
            "largest_observed_context": 65536,
            "recommended_context": 58880,
            "context_confirmed": False,
        }
        self.app.store.put("result", "measured", result)
        data = {"result_id": "measured", "use_context": True}
        preview = self.app.action("/api/result/preview", data)
        self.assertEqual(preview["settings"]["context"], 65536)
        with patch("lllm2.app.launch_args"):
            saved = self.app.action("/api/default/save", data)
        self.assertEqual(saved["context"], 65536)
        # Neither a sample nor a context measurement: nothing to promote.
        self.app.store.put(
            "result", "measured", {**result, "largest_observed_context": None}
        )
        for route in ("/api/result/preview", "/api/default/save"):
            with self.subTest(route=route), self.assertRaises(ValueError):
                self.app.action(route, {"result_id": "measured"})

    def test_ineligible_experiments_cannot_be_loaded_or_saved(self):
        for change in (
            {"measurement_mode": "warm-conversation"},
            {"quality_status": "failed"},
            {"status": "failed"},
        ):
            self.app.store.put("result", "measured", {**self.result, **change})
            for route in ("/api/result/preview", "/api/default/save"):
                with (
                    self.subTest(change=change, route=route),
                    self.assertRaises(ValueError),
                ):
                    self.app.action(route, {"result_id": "measured"})
