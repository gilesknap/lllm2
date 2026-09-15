import unittest
from unittest.mock import patch

from lllm2 import defaults, discovery, recommendations
from lllm2.gpu_tables import (
    GPU_TABLES,
    MODAL_GPUS,
    GpuType,
    gpu_type,
    gpu_types,
    register_gpu_table,
    table_hardware,
)
from lllm2.settings import Settings, build_launch_args

FLAGS = [
    "--model",
    "--host",
    "--port",
    "--ctx-size",
    "--parallel",
    "--device",
    "--gpu-layers",
    "--jinja",
    "--fit",
    "--fit-target",
    "--flash-attn",
    "--cache-type-k",
    "--cache-type-v",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-type-k",
    "--spec-draft-type-v",
    "--chat-template-file",
    "--batch-size",
    "--ubatch-size",
    "--cache-ram",
    "--ctx-checkpoints",
]
ENGINE = {
    "path": "/opt/lllm2/llama-server",
    "flags": FLAGS,
    "help": "--spec-type none, draft-mtp, ngram-simple",
    "devices": ["CUDA0"],
    "error": None,
    "cuda_graph": {"supported": True, "reason": "Probed remotely.", "library": None},
    "cache_kernel": {"reason": "Probed remotely.", "library": None},
    "environment": {},
}


def no_local_probe(*_args, **_kwargs):
    raise AssertionError("Remote defaults must not probe local hardware.")


class GpuTableTests(unittest.TestCase):
    def test_every_modal_gpu_builds_a_single_gpu_hardware_description(self):
        self.assertEqual(gpu_types("modal"), MODAL_GPUS)
        for entry in MODAL_GPUS:
            with self.subTest(gpu=entry.name):
                host = table_hardware("modal", entry.name)
                self.assertEqual(host["source"], "modal")
                self.assertIsNone(host["error"])
                (card,) = host["gpus"]
                self.assertEqual(card["name"], entry.label)
                self.assertEqual(card["total_mib"], entry.total_mib)
                # Nominal decimal gigabytes convert to fewer MiB than the GB figure.
                self.assertLess(card["total_mib"], entry.vram_gb * 1000)
                self.assertGreater(entry.total_mib, entry.vram_gb * 900)
                self.assertAlmostEqual(entry.vram_gib, entry.total_mib / 1024, 0)
                self.assertGreater(entry.usd_per_hour, 0)

    def test_unknown_provider_or_gpu_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown GPU provider"):
            table_hardware("nowhere", "T4")
        with self.assertRaisesRegex(ValueError, "Unknown modal GPU type"):
            gpu_type("modal", "Z9")

    def test_another_provider_can_register_a_table(self):
        table = (GpuType("big", "Example GPU", 48, 1.5),)
        with patch.dict(GPU_TABLES):
            register_gpu_table("other", table)
            host = table_hardware("other", "big")
            with self.assertRaises(ValueError):
                register_gpu_table("local", table)
        self.assertEqual(host["source"], "other")
        self.assertEqual(host["gpus"][0]["total_mib"], 45776)
        self.assertNotIn("other", GPU_TABLES)

    def test_local_hardware_reports_local_source(self):
        row = "0, GPU-1, NVIDIA GeForce RTX 3090, 24576, 100, 580.1"
        with patch.object(discovery, "command", return_value=(0, row)):
            host = discovery.hardware()
        self.assertEqual(host["source"], "local")
        self.assertEqual(host["gpus"][0]["total_mib"], 24576)


class ModalStartingDefaultsTests(unittest.TestCase):
    entry = next(m for m in defaults.CATALOG if m["id"] == "qwen3.5-35b-a3b")

    def resolve(self, gpu):
        selection = Settings(
            model=f"/models/{self.entry['name']}/{self.entry['file']}",
            backend="modal",
            gpu_type=gpu,
        )
        meta = {
            "error": None,
            "context": self.entry["max_ctx"],
            "template": "",
            "mtp": bool(self.entry.get("mtp")),
        }
        with (
            patch.object(defaults, "hardware", side_effect=no_local_probe),
            patch.object(defaults, "command", side_effect=no_local_probe),
            patch.object(defaults, "probe", side_effect=no_local_probe),
            patch.object(defaults, "metadata", side_effect=no_local_probe),
            patch.object(recommendations, "hardware", side_effect=no_local_probe),
            patch.object(recommendations, "probe", side_effect=no_local_probe),
            patch.object(recommendations, "metadata", side_effect=no_local_probe),
            patch.object(recommendations, "fingerprint", side_effect=no_local_probe),
        ):
            result = defaults.starting_defaults(
                selection,
                table_hardware("modal", gpu),
                ENGINE,
                meta,
                {"size": None, "sha256": None},
            )
        settings = Settings.parse(result["settings"])
        args = build_launch_args(settings, 8080, ENGINE, meta, path=lambda v: v)
        return settings, result, args

    def test_each_modal_gpu_gets_launchable_starting_defaults(self):
        planned = {}
        for entry in MODAL_GPUS:
            with self.subTest(gpu=entry.name):
                settings, result, args = self.resolve(entry.name)
                notes = " ".join(result["notes"])
                self.assertIn("No measured built-in profile", notes)
                self.assertEqual(
                    (settings.backend, settings.gpu_type), ("modal", entry.name)
                )
                self.assertEqual(args[args.index("--device") + 1], "CUDA0")
                self.assertEqual(
                    args[args.index("--ctx-size") + 1], str(settings.context)
                )
                if "calibrated planner" in notes:
                    planned[entry.total_mib] = settings.context
                else:
                    self.assertIn("does not fit", notes)
                    self.assertEqual(
                        (settings.slots, settings.speculation), (1, "none")
                    )
        # The smallest card cannot hold the weights; every 40 GB or larger card can.
        self.assertNotIn(gpu_type("modal", "T4").total_mib, planned)
        for entry in MODAL_GPUS:
            if entry.vram_gb >= 40:
                self.assertIn(entry.total_mib, planned)
        contexts = [planned[mib] for mib in sorted(planned)]
        self.assertEqual(contexts, sorted(contexts))


if __name__ == "__main__":
    unittest.main()
