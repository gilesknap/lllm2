"""The context planner works from the checkpoint, not from the catalogue.

The catalogue lists one quantisation of each model by exact file name. Anything
else -- a Q8_0 build of a catalogued model, or a model found through Find models
-- used to skip planning entirely and keep the dataclass default of 4096 tokens,
which makes a 96 GB card refuse a long prompt mid-generation.
"""

import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lllm2 import defaults, discovery, gguf
from lllm2.settings import Settings

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
]
ENGINE = {
    "path": "/opt/lllm2/llama-server",
    "flags": FLAGS,
    "help": "--spec-type none, draft-mtp, ngram-simple",
    "devices": ["CUDA0"],
    "error": None,
}

# A 96 GB card, the hardware that returned 4096 tokens in production.
BIG_CARD = {
    "gpus": [{"name": "NVIDIA RTX PRO 6000", "total_mib": 98304, "uuid": "gpu"}],
    "error": None,
    "ram": {"available_gib": 128},
}

# The owner's checkpoint: a Q8_0 build of the catalogued 27B, downloaded through
# Find models, so its file name matches no catalogue entry.
Q8_PATH = "/models/Qwen3.8-27B/Qwen3.8-27B-Q8_0.gguf"
Q8_META = {
    "architecture": "qwen3next",
    "name": "Qwen3.8-27B",
    "context": 262144,
    "mtp": True,
    "template": "",
    "size": 29_000_000_000,
    "kv_kib_per_token": 64.0,
    "full_attention_layers": 16,
    "error": None,
}


def gguf_file(path, meta, tensors=()):
    """Write a GGUF whose header declares ``meta`` and names ``tensors``."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for key, value in meta.items():
        out += struct.pack("<Q", len(key)) + key.encode()
        if isinstance(value, str):
            out += struct.pack("<IQ", gguf.STRING, len(value)) + value.encode()
        else:
            out += struct.pack("<II", gguf.U32, value)
    for name in tensors:
        out += struct.pack("<Q", len(name)) + name.encode()
        out += struct.pack("<IQIQ", 1, 4096, 0, 0)
    path = Path(path)
    path.write_bytes(bytes(out))
    return path


class HeaderPlanningFactsTests(unittest.TestCase):
    """What the planner needs comes out of the header the panel already reads."""

    # Qwen3.8-27B: 64 layers plus an MTP head, one in four keeping a KV cache,
    # 8 KV heads of 128+128. That is the catalogue's hand-entered 64 KiB.
    HYBRID = {
        "general.architecture": "qwen3next",
        "qwen3next.block_count": 65,
        "qwen3next.full_attention_interval": 4,
        "qwen3next.attention.head_count": 32,
        "qwen3next.attention.head_count_kv": 8,
        "qwen3next.attention.key_length": 128,
        "qwen3next.attention.value_length": 128,
        "qwen3next.context_length": 262144,
    }

    def test_hybrid_header_reproduces_the_catalogue_cost(self):
        entry = next(m for m in defaults.CATALOG if m["id"] == "qwen3.8-27b")
        self.assertEqual(gguf.cache_layers(self.HYBRID, True), 16)
        self.assertEqual(
            gguf.kv_kib_per_token(self.HYBRID, True), entry["kv_kib_per_token"]
        )

    def test_a_hybrid_without_an_mtp_head_keeps_every_fourth_layer(self):
        """``block_count`` only carries a head when the tensors show one.

        Subtracting one regardless drops a caching layer from a hybrid that has
        no head, which prices the cache too cheaply and plans a context that
        does not fit.
        """
        headless = dict(self.HYBRID, **{"qwen3next.block_count": 48})
        self.assertEqual(gguf.cache_layers(headless, False), 12)
        self.assertEqual(
            gguf.cache_layers(dict(headless, **{"qwen3next.block_count": 49}), True), 12
        )

    def test_dense_header_counts_every_layer_and_splits_the_embedding(self):
        dense = {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
        # 8 KV heads x (128 + 128) x 2 bytes x 32 layers.
        self.assertEqual(gguf.cache_layers(dense), 32)
        self.assertEqual(gguf.kv_kib_per_token(dense), 128)

    def test_a_header_without_a_shape_answers_none(self):
        self.assertIsNone(gguf.kv_kib_per_token({"general.architecture": "mamba"}))
        self.assertIsNone(
            gguf.kv_kib_per_token(
                # Per-layer head counts arrive as an array placeholder.
                {"gemma4.block_count": 48, "gemma4.attention.head_count_kv": "<array>"}
            )
        )

    def test_metadata_carries_size_and_cost_to_the_planner(self):
        with tempfile.TemporaryDirectory() as temp:
            path = gguf_file(
                Path(temp) / "m.gguf", self.HYBRID, ["blk.64.nextn.weight"]
            )
            record = discovery.metadata(str(path))
            size = path.stat().st_size
        self.assertEqual(record["error"], None)
        self.assertEqual(record["size"], size)
        self.assertEqual(record["kv_kib_per_token"], 64)
        self.assertEqual(record["full_attention_layers"], 16)
        self.assertIs(record["mtp"], True)


class UncataloguedPlanTests(unittest.TestCase):
    """A quantisation the catalogue never listed still gets a planned context."""

    def resolve(self, model, meta, host, engine=None):
        engine = engine or ENGINE
        quantised = "--cache-type-k" in engine["flags"]
        caps = {
            "flash": {
                "status": "available" if "--flash-attn" in engine["flags"] else "no",
                "reason": "Probed.",
            },
            "cache": {
                "status": "available" if quantised else "unsupported",
                "reason": "This build has no quantised KV cache.",
            },
            "draft-mtp": {
                "status": "available" if meta.get("mtp") else "unsupported",
                "reason": "This checkpoint carries no MTP head.",
            },
        }
        selection = Settings.parse(
            {"model": model, "engine": engine["path"], "backend": "CUDA"}
        )
        with (
            patch.object(defaults, "hardware", return_value=host),
            patch.object(defaults, "probe", return_value=engine),
            patch.object(defaults, "metadata", return_value=meta),
            patch.object(defaults, "capabilities", return_value=caps),
            patch.object(defaults, "command", return_value=(0, "512")),
            patch.object(defaults, "measured_defaults", return_value=(None, [])),
        ):
            result = defaults.starting_defaults(selection)
        return Settings.parse(result["settings"]), result

    def test_owner_q8_checkpoint_plans_the_full_context(self):
        """29 GB of Q8 weights on a 96 GB card, the case that returned 4096."""
        s, result = self.resolve(Q8_PATH, Q8_META, BIG_CARD)
        self.assertEqual((s.context, s.slots), (1048576, 4))
        self.assertEqual(s.speculation, "draft-mtp")
        self.assertIn("calibrated planner", " ".join(result["notes"]))

    def test_the_catalogued_quantisation_still_plans_the_same(self):
        """The Q4_K_S build of the same model is the reference for the Q8 one."""
        entry = next(m for m in defaults.CATALOG if m["id"] == "qwen3.8-27b")
        path = f"/models/{entry['name']}/{entry['file']}"
        meta = dict(Q8_META, size=None, kv_kib_per_token=None)
        meta["full_attention_layers"] = None
        s, _ = self.resolve(path, meta, BIG_CARD)
        self.assertEqual((s.context, s.slots), (1048576, 4))

    def test_the_catalogue_keeps_the_last_word_on_a_checkpoint_it_lists(self):
        """A header cannot see a sliding window; the hand-checked entry can.

        gpt-oss caches 24 layers by the header and half that in practice, so a
        header that overrode the entry would halve a context the panel plans
        correctly today.
        """
        entry = next(m for m in defaults.CATALOG if m["id"] == "gpt-oss-20b")
        path = f"/models/{entry['name']}/{entry['file']}"
        meta = dict(Q8_META, context=entry["max_ctx"], mtp=False, size=None)
        meta["full_attention_layers"] = None
        catalogued, _ = self.resolve(path, dict(meta, kv_kib_per_token=None), BIG_CARD)
        doubled, _ = self.resolve(path, dict(meta, kv_kib_per_token=48.0), BIG_CARD)
        self.assertEqual(doubled.context, catalogued.context)
        self.assertGreater(catalogued.context, 131072)

    def test_an_uncalibrated_context_still_fits_the_card(self):
        """32K is a starting point, not a promise: a small card gets less.

        This engine cannot quantise the cache, so the calibrated planner will
        not run, but the checkpoint's shape is known and an f16 cache of 32K
        tokens would not fit beside the weights on a 12 GiB card.
        """
        host = {
            "gpus": [
                {"name": "NVIDIA GeForce RTX 3060", "total_mib": 12288, "uuid": "gpu"}
            ],
            "error": None,
            "ram": {"available_gib": 32},
        }
        meta = dict(Q8_META, size=5_030_000_000, kv_kib_per_token=144.0, mtp=False)
        meta["full_attention_layers"] = None
        engine = dict(ENGINE, flags=[f for f in FLAGS if f != "--cache-type-k"])
        s, result = self.resolve(
            "/models/Homemade/Homemade-Q4_K_M.gguf", meta, host, engine
        )
        notes = " ".join(result["notes"])
        self.assertIn("NOT calibrated", notes)
        self.assertIn("q8_0 KV cache", notes)
        self.assertLess(s.context, defaults.UNCALIBRATED_CONTEXT)
        self.assertGreaterEqual(s.context, 8192)
        # Weights plus an f16 cache of that context, inside the card.
        weights = meta["size"] / 2**20
        cache = s.context * meta["kv_kib_per_token"] / 1024
        self.assertLess(weights + cache, host["gpus"][0]["total_mib"] - 3936)

    def test_non_catalogue_model_on_a_local_card_plans_sensibly(self):
        host = {
            "gpus": [
                {"name": "NVIDIA GeForce RTX 3090", "total_mib": 24576, "uuid": "gpu"}
            ],
            "error": None,
            "ram": {"available_gib": 48},
        }
        meta = dict(Q8_META, size=13_000_000_000, mtp=False)
        s, result = self.resolve("/models/Homemade/Homemade-Q4_K_M.gguf", meta, host)
        self.assertGreater(s.context, 32768)
        self.assertLessEqual(s.context, meta["context"] * s.slots)
        self.assertGreaterEqual(s.slots, 1)
        self.assertIn("calibrated planner", " ".join(result["notes"]))

    def test_unplannable_checkpoint_is_clamped_and_says_so(self):
        blind = dict(Q8_META, size=None, kv_kib_per_token=None, context=None)
        blind["full_attention_layers"] = None
        s, result = self.resolve("/models/Mystery/Mystery-Q6_K.gguf", blind, BIG_CARD)
        notes = " ".join(result["notes"])
        self.assertEqual((s.context, s.slots), (defaults.UNCALIBRATED_CONTEXT, 1))
        self.assertIn("NOT calibrated", notes)
        self.assertIn("file size", notes)
        self.assertIn("KV cache cost", notes)
        self.assertNotIn("calibrated planner", notes)

    def test_a_checkpoint_ceiling_bounds_the_clamp(self):
        blind = dict(Q8_META, size=None, kv_kib_per_token=None, context=8192)
        blind["full_attention_layers"] = None
        s, _ = self.resolve("/models/Mystery/Mystery-Q6_K.gguf", blind, BIG_CARD)
        self.assertEqual((s.context, s.slots), (8192, 1))

    def test_no_gpu_is_an_error_not_a_default(self):
        host = {"gpus": [], "error": "No devices were found", "ram": {}}
        with self.assertRaisesRegex(ValueError, "No CUDA GPU is available"):
            self.resolve(Q8_PATH, Q8_META, host)


if __name__ == "__main__":
    unittest.main()
