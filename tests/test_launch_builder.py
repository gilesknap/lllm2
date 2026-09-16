import unittest
from unittest.mock import patch

from lllm2.settings import Settings, build_launch_args, launch_args

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
    "--perf",
    "--flash-attn",
    "--cache-type-k",
    "--cache-type-v",
    "--spec-type",
    "--spec-draft-n-max",
]
# A remote engine record: capabilities come from a provider probe, not a local binary.
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
META = {"error": None, "context": 65536, "template": "", "mtp": True}
MODEL = "/models/example/model.gguf"


def no_local_probe(*_args, **_kwargs):
    raise AssertionError("The builder must not probe local files.")


def remote(value):
    return "/vol" + value


class LaunchBuilderTests(unittest.TestCase):
    def build(self, settings, engine=ENGINE, meta=META, **kwargs):
        with (
            patch("lllm2.settings.probe", side_effect=no_local_probe),
            patch("lllm2.settings.metadata", side_effect=no_local_probe),
            patch("lllm2.settings.cuda_graph_support", side_effect=no_local_probe),
            patch("lllm2.settings.cache_kernel_support", side_effect=no_local_probe),
            patch("lllm2.settings.engine_environment", side_effect=no_local_probe),
        ):
            return build_launch_args(settings, 8080, engine, meta, **kwargs)

    def test_supplied_capabilities_and_metadata_produce_a_remote_command(self):
        args = self.build(
            Settings(model=MODEL, engine="/local/unused", context=16384, slots=2),
            host="0.0.0.0",
            path=remote,
        )
        self.assertEqual(args[0], "/opt/lllm2/llama-server")
        for flag, value in [
            ("--model", "/vol" + MODEL),
            ("--host", "0.0.0.0"),
            ("--port", "8080"),
            ("--ctx-size", "16384"),
            ("--parallel", "2"),
            ("--device", "CUDA0"),
            ("--fit", "on"),
            ("--spec-type", "none"),
        ]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--perf", args)

    def test_supplied_flags_decide_which_arguments_are_allowed(self):
        old = {**ENGINE, "flags": [f for f in FLAGS if f not in ("--fit", "--perf")]}
        with self.assertRaisesRegex(ValueError, "Automatic GPU layers require"):
            self.build(Settings(model=MODEL), engine=old, path=remote)
        args = self.build(Settings(model=MODEL, gpu_layers=99), engine=old, path=remote)
        self.assertEqual(args[args.index("--gpu-layers") + 1], "99")
        self.assertNotIn("--perf", args)

    def test_supplied_device_list_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "detected GPU device"):
            self.build(
                Settings(model=MODEL, device="CUDA1"),
                path=remote,
            )

    def test_supplied_metadata_limits_context_and_mtp(self):
        with self.assertRaisesRegex(ValueError, "exceeds checkpoint context"):
            self.build(
                Settings(model=MODEL, context=32768),
                meta={**META, "context": 16384},
                path=remote,
            )
        with self.assertRaisesRegex(ValueError, "Cannot read checkpoint: bad"):
            self.build(Settings(model=MODEL), meta={**META, "error": "bad"})
        with self.assertRaisesRegex(ValueError, "no MTP tensors"):
            self.build(
                Settings(model=MODEL, speculation="draft-mtp"),
                meta={**META, "mtp": False},
                path=remote,
            )
        args = self.build(
            Settings(model=MODEL, speculation="draft-mtp", draft_length=3),
            path=remote,
        )
        self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(args[args.index("--spec-draft-n-max") + 1], "3")

    def test_supplied_drafter_metadata_allows_a_remote_dflash_command(self):
        engine = {
            **ENGINE,
            "flags": [*FLAGS, "--model-draft"],
            "help": ENGINE["help"] + ", draft-dflash",
        }
        s = Settings(
            model=MODEL,
            flash="on",
            speculation="draft-dflash",
            drafter="/models/example/drafter.gguf",
            pair_confirmed=True,
        )
        args = self.build(
            s,
            engine=engine,
            meta={**META, "drafter": {**META, "mtp": False}},
            path=remote,
        )
        self.assertEqual(args[args.index("--spec-type") + 1], "draft-dflash")
        self.assertEqual(
            args[args.index("--model-draft") + 1], "/vol/models/example/drafter.gguf"
        )
        with self.assertRaisesRegex(ValueError, "not a readable GGUF"):
            self.build(
                s,
                engine=engine,
                meta={**META, "drafter": {**META, "error": "bad"}},
                path=remote,
            )

    def test_local_wrapper_matches_builder_with_local_probe_results(self):
        s = Settings(model=MODEL, flash="on", cache="q8_0")
        with (
            patch("lllm2.settings.probe", return_value=ENGINE),
            patch("lllm2.settings.metadata", return_value=META),
        ):
            local = launch_args(s, 1920)
        self.assertEqual(local, build_launch_args(s, 1920, ENGINE, META))
        self.assertEqual(local[local.index("--model") + 1], MODEL)
        self.assertEqual(local[local.index("--host") + 1], "127.0.0.1")
        self.assertEqual(local[local.index("--cache-type-k") + 1], "q8_0")


if __name__ == "__main__":
    unittest.main()
