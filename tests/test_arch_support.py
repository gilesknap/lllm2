"""Find models compares a repository's GGUF architecture with the engine's list."""

import json
import runpy
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import Mock, patch

from lllm2 import config
from lllm2.app import App
from lllm2.catalogue import Finder, arch_support, request_support_url, variants
from lllm2.discovery import supported_architectures
from lllm2.engine_install import _unpack
from lllm2.engine_release import LLAMA_CPP_REF
from lllm2.store import Store

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
architectures = runpy.run_path(str(SCRIPTS / "engine-architectures.py"))[
    "architectures"
]
package_engine = runpy.run_path(str(SCRIPTS / "package-engine.py"))["package_engine"]

ARCH_SOURCE = """
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_CLIP,             "clip"             }, // dummy, only used by llama-quantize
    { LLM_ARCH_LLAMA,            "llama"            },
    { LLM_ARCH_QWEN3,            "qwen3"            },
    { LLM_ARCH_UNKNOWN,          "(unknown)"        },
};

static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_TYPE,                     "general.type"                          },
};
"""


class BuildListTests(unittest.TestCase):
    def test_names_come_from_the_arch_table_without_placeholders(self):
        self.assertEqual(architectures(ARCH_SOURCE), ["llama", "qwen3"])

    def test_a_moved_table_fails_the_build(self):
        with self.assertRaisesRegex(ValueError, "LLM_ARCH_NAMES"):
            architectures(ARCH_SOURCE.replace("LLM_ARCH_NAMES", "ARCH_TABLE"))

    def test_the_list_survives_packaging_and_install_unpack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, installed = root / "source", root / "installed"
            source.mkdir()
            installed.mkdir()
            (source / "llama-server").write_bytes(b"server")
            (source / "lllm2-architectures.json").write_text('["llama", "qwen3"]')
            package_engine(source, root / "engine.tar.gz")
            _unpack(root / "engine.tar.gz", installed)
            self.assertEqual(
                supported_architectures(str(installed / "llama-server")),
                {"llama", "qwen3"},
            )


class EngineListTests(unittest.TestCase):
    def test_missing_or_malformed_lists_are_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "llama-server"
            listing = binary.with_name("lllm2-architectures.json")
            # A custom build or an engine archive built before the list existed.
            self.assertIsNone(supported_architectures(str(binary)))
            for content in ("not json", '{"llama": true}', "[1, 2]"):
                with self.subTest(content=content):
                    listing.write_text(content)
                    self.assertIsNone(supported_architectures(str(binary)))
        for binary in ("", None, 3):
            self.assertIsNone(supported_architectures(binary))


class ComparisonTests(unittest.TestCase):
    entry = {"repo": "test/Future-GGUF", "architecture": "future9"}

    def test_states(self):
        self.assertEqual(
            arch_support(self.entry, frozenset({"future9"})),
            {"arch_support": "supported"},
        )
        self.assertEqual(arch_support(self.entry, None), {"arch_support": "unknown"})
        self.assertEqual(
            arch_support({"repo": "test/Old"}, frozenset({"llama"})),
            {"arch_support": "unknown"},
        )
        unsupported = arch_support(self.entry, frozenset({"llama"}))
        self.assertEqual(unsupported["arch_support"], "unsupported")
        self.assertEqual(
            unsupported["support_url"],
            request_support_url("future9", "test/Future-GGUF"),
        )

    def test_request_support_link_is_a_prefilled_issue(self):
        url = urllib.parse.urlsplit(request_support_url("future9", "test/Future-GGUF"))
        self.assertEqual(
            (url.scheme, url.netloc, url.path),
            ("https", "github.com", "/gilesknap/lllm2/issues/new"),
        )
        query = urllib.parse.parse_qs(url.query)
        self.assertEqual(query["title"], ["Support the future9 model architecture"])
        body = query["body"][0]
        self.assertIn("`future9`", body)
        self.assertIn("https://huggingface.co/test/Future-GGUF", body)
        self.assertIn(f"`{LLAMA_CPP_REF}`", body)


class FinderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for key, value in (("STATE_DIR", root / "state"), ("MODELS_DIR", root / "m")):
            patcher = patch.object(config, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.store = Store()
        self.addCleanup(self.store.db.close)
        self.info = {
            "id": "test/Future-GGUF",
            "sha": "b" * 40,
            "pipeline_tag": "text-generation",
            "gguf": {"architecture": "future9", "context_length": 4096},
            "siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": 100}],
        }
        self.host = {"gpus": [], "ram": {"total_gib": 32}}

    def test_variants_record_the_repository_architecture(self):
        self.assertEqual(variants(self.info)[0]["architecture"], "future9")

    def test_cached_results_are_scored_for_each_engine(self):
        finder = Finder(self.store)
        with patch(
            "lllm2.catalogue.metadata_get",
            side_effect=[[{"id": self.info["id"]}], self.info],
        ) as request:
            [old] = finder.search("q", self.host, False, frozenset({"llama"}))[
                "entries"
            ]
            [new] = finder.search("q", self.host, False, frozenset({"future9"}))[
                "entries"
            ]
            [custom] = finder.search("q", self.host)["entries"]
        self.assertEqual(request.call_count, 2)
        self.assertEqual(old["arch_support"], "unsupported")
        self.assertIn("future9", old["support_url"])
        self.assertEqual(new["arch_support"], "supported")
        self.assertEqual(custom["arch_support"], "unknown")

    def test_results_cached_before_the_architecture_was_kept_are_unknown(self):
        entry = {k: v for k, v in variants(self.info)[0].items() if k != "architecture"}
        result = Finder(self.store).present(
            {"entries": [entry]}, self.host, frozenset({"llama"})
        )
        self.assertEqual(result["entries"][0]["arch_support"], "unknown")


class FindApiTests(unittest.TestCase):
    def test_local_searches_use_the_selected_engine_and_remote_ones_do_not(self):
        app = App.__new__(App)
        app.finder = Mock()
        app.selected_hardware = Mock(return_value={})
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "llama-server"
            binary.with_name("lllm2-architectures.json").write_text(
                json.dumps(["llama"])
            )
            for data, expected in (
                ({"backend": "CUDA", "engine": str(binary)}, {"llama"}),
                ({"backend": "", "engine": str(binary)}, {"llama"}),
                ({"backend": "CUDA"}, None),
                ({"backend": "modal", "engine": str(binary)}, None),
            ):
                with self.subTest(data=data):
                    app.action("/api/models/find", {"query": "q", **data})
                    self.assertEqual(app.finder.search.call_args.args[3], expected)
