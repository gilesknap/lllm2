"""Metadata discovery, catalogue ownership and persistent download queue behaviour."""

import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lllm2 import config, downloads
from lllm2.app import App
from lllm2.catalogue import Catalogue, Finder, local_paths, suitability, variants
from lllm2.store import Store


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for key, value in (
            ("STATE_DIR", self.root / "state"),
            ("MODELS_DIR", self.root / "models"),
        ):
            patcher = patch.object(config, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.store = Store()
        self.addCleanup(self.store.db.close)
        self.catalogue = Catalogue(self.store)
        self.entry = {
            "id": "new",
            "name": "New",
            "repo": "test/New-Instruct",
            "file": "Q4/model-00001-of-00002.gguf",
            "files": [
                "Q4/model-00001-of-00002.gguf",
                "Q4/model-00002-of-00002.gguf",
                "mmproj.gguf",
            ],
            "revision": "a" * 40,
        }
        self.host = {"gpus": [{"total_mib": 8192}], "ram": {"total_gib": 32}}
        for attr, value in (
            ("_downloads", {}),
            ("_store", None),
            ("_queue", queue.Queue()),
        ):
            patcher = patch.object(downloads, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_seed_once_preserves_empty_catalogue(self):
        seeds = self.catalogue.list()
        self.assertTrue(seeds)
        for entry in seeds:
            self.catalogue.remove(entry["id"])
        self.assertEqual(Catalogue(self.store).list(), [])

    def test_add_is_idempotent_and_preserves_seed_tuning(self):
        seed = self.catalogue.list()[0]
        self.catalogue.add({**seed, "id": "duplicate", "revision": "new"})
        self.assertEqual(self.catalogue.get(seed["id"]), seed)
        self.assertFalse(self.store.get("catalogue", "duplicate"))
        self.catalogue.add(self.entry)
        self.assertEqual(Catalogue(self.store).get("new"), self.entry)

    def test_path_traversal_and_external_symlinks_rejected(self):
        for name, file in (
            ("../outside", "x.gguf"),
            ("New", "../x.gguf"),
            ("New", "/tmp/x.gguf"),
        ):
            with self.subTest(name=name, file=file), self.assertRaises(ValueError):
                local_paths({**self.entry, "name": name, "files": [file]})
        config.MODELS_DIR.mkdir()
        (config.MODELS_DIR / "New").symlink_to(self.root / "outside")
        with self.assertRaises(ValueError):
            local_paths(self.entry)

    def test_metadata_only_memory_bands(self):
        self.assertEqual(suitability(4 * 2**30, self.host)["fit_rank"], 0)
        self.assertEqual(suitability(20 * 2**30, self.host)["fit_rank"], 1)
        self.assertEqual(suitability(40 * 2**30, self.host)["fit"], "Too large")
        self.assertEqual(suitability(None, self.host)["fit"], "Unknown")
        self.assertEqual(
            suitability(2**30, {"gpus": [], "ram_gib": 32})["fit"], "Unknown"
        )
        two_gpus = {**self.host, "gpus": [{"total_mib": 8192}] * 2}
        self.assertEqual(suitability(8 * 2**30, two_gpus)["fit_rank"], 1)

    def info(self):
        return {
            "id": "test/Small-Instruct-GGUF",
            "sha": "b" * 40,
            "pipeline_tag": "text-generation",
            "downloads": 20,
            "tags": ["license:mit"],
            "siblings": [
                {"rfilename": "Q4/model-Q4_K_M-00001-of-00002.gguf", "size": 100},
                {"rfilename": "Q4/model-Q4_K_M-00002-of-00002.gguf", "size": 200},
                {"rfilename": "model-Q8_0.gguf", "size": 600},
            ],
        }

    def test_shards_are_one_variant_with_full_size_and_pinned_revision(self):
        found = variants(self.info())
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]["size_bytes"], 300)
        self.assertEqual(len(found[0]["files"]), 2)
        self.assertEqual(found[0]["quant"], "Q4_K_M")
        self.assertEqual(found[0]["revision"], "b" * 40)
        self.assertTrue(found[0]["instruct"])
        self.assertEqual(found[0]["license"], "mit")

    def test_incomplete_shards_and_ambiguous_projector_are_not_addable(self):
        info = self.info()
        info["siblings"].pop(1)
        self.assertIn("Incomplete", variants(info)[0]["issue"])
        info = self.info()
        info["siblings"] += [
            {"rfilename": name, "size": 50}
            for name in ("mmproj-F16.gguf", "mmproj-Q8.gguf")
        ]
        self.assertTrue(all(e["issue"] for e in variants(info)))
        candidate = variants(info)[0]
        self.store.put("hf-search", "x", {"entries": [candidate]})
        with self.assertRaisesRegex(ValueError, "projector"):
            Finder(self.store).candidate(candidate["id"])

    def test_projector_size_included_and_missing_sizes_unknown(self):
        info = self.info()
        info["siblings"].append({"rfilename": "mmproj.gguf", "size": 50})
        found = variants(info)[0]
        self.assertEqual(found["size_bytes"], 350)
        self.assertEqual(found["mmproj"], "mmproj.gguf")
        del info["siblings"][0]["size"]
        self.assertIsNone(variants(info)[0]["size_bytes"])

    def test_gated_non_generation_and_unpinned_repositories_excluded(self):
        for override in (
            {"gated": "auto"},
            {"private": True},
            {"pipeline_tag": "feature-extraction"},
            {"sha": None},
        ):
            with self.subTest(override=override):
                self.assertEqual(variants({**self.info(), **override}), [])

    def test_cache_refresh_and_offline_fallback(self):
        finder = Finder(self.store)
        with patch(
            "lllm2.catalogue.metadata_get",
            side_effect=[[{"id": self.info()["id"]}], self.info()],
        ) as request:
            first = finder.search("small", self.host)
            second = finder.search("small", self.host)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(first, second)
        with patch("lllm2.catalogue.metadata_get", side_effect=OSError("offline")):
            stale = finder.search("small", self.host, refresh=True)
            self.assertIn("cached", stale["warning"])
            self.assertEqual(stale["entries"], first["entries"])
            with self.assertRaisesRegex(ValueError, "unavailable"):
                finder.search("another", self.host)

    def test_removal_keeps_files_unless_explicit_and_leaves_unrelated_files(self):
        paths = local_paths(self.entry)
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"weights")
        part = paths[0].with_suffix(".gguf.part")
        part.write_bytes(b"partial")
        unrelated = paths[0].parent / "other.gguf"
        unrelated.write_bytes(b"keep")
        downloads.remove(self.entry)
        self.assertTrue(all(p.exists() for p in paths))
        downloads.remove(self.entry, True)
        self.assertFalse(any(p.exists() for p in paths))
        self.assertFalse(part.exists())
        self.assertTrue(unrelated.exists())

    def test_running_shared_and_downloading_weights_are_protected(self):
        path = local_paths(self.entry)[0]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "Stop"):
            downloads.remove(self.entry, True, protected=[str(path)])
        with self.assertRaisesRegex(ValueError, "shared"):
            downloads.remove(
                self.entry, True, other_entries=[{**self.entry, "id": "another"}]
            )
        with patch.object(downloads, "ensure_worker"):
            downloads.start(self.entry)
        with self.assertRaisesRegex(ValueError, "Cancel"):
            downloads.remove(self.entry, True)
        self.assertTrue(path.exists())

    def test_queue_persists_zero_byte_jobs_and_cancellation(self):
        downloads.configure(self.store)
        with patch.object(downloads, "ensure_worker"):
            first = downloads.start(self.entry)
            self.assertIs(downloads.start(self.entry), first)
            self.assertEqual(downloads._queue.qsize(), 1)
            self.assertEqual(self.store.get("download-job", "new")["state"], "queued")
            downloads._downloads.clear()
            downloads.configure(self.store)
            self.assertIn("new", downloads._downloads)
            downloads.cancel("new")
            self.assertEqual(downloads._downloads["new"].state, "cancelled")
            self.assertEqual(
                self.store.get("download-job", "new")["state"], "cancelled"
            )
            downloads._downloads.clear()
            downloads.configure(self.store)
            self.assertEqual(downloads._downloads, {})

    def test_single_worker_and_nested_shards(self):
        seen = []

        def fetch(dl, file, target, base):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"GGUF")
            seen.append((file, target))
            dl.done = base + 4
            return True

        with patch.object(downloads, "ensure_worker"):
            dl = downloads.start(self.entry)
        with (
            patch.object(downloads, "_fetch", side_effect=fetch),
            patch.object(downloads, "_size_of", return_value=4),
        ):
            downloads._run(dl)
        self.assertEqual(dl.state, "complete")
        self.assertEqual([p for _, p in seen], local_paths(self.entry))
        self.assertEqual(dl.done, 12)
        # One worker consumes the queue serially; submissions never spawn per-job threads.
        worker = Mock(is_alive=Mock(return_value=True))
        with (
            patch.object(downloads, "_worker", worker),
            patch.object(threading, "Thread") as thread,
        ):
            downloads.ensure_worker()
            thread.assert_not_called()

    def test_api_delete_requires_explicit_boolean_and_keeps_results(self):
        app = App.__new__(App)
        app.store, app.catalogue = self.store, self.catalogue
        app.bench = Mock(lock=threading.RLock(), active=False)
        app.engine = Mock(state=Mock(return_value={"running": False}))
        self.catalogue.add(self.entry)
        path = local_paths(self.entry)[0]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"keep")
        self.store.put("default", "saved", {"model": str(path)})
        app.action("/api/catalogue/remove", {"id": "new", "delete_weights": "false"})
        self.assertTrue(path.exists())
        self.assertIsNotNone(self.store.get("default", "saved"))
        self.assertIsNone(self.store.get("catalogue", "new"))
