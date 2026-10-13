"""Maintainer-requested resume regressions, with tiny offline GGUF fixtures."""

import hashlib
import io
import json
import queue
import struct
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from lllm2 import config, downloads
from lllm2.store import Store


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}


class DownloadFollowupTests(unittest.TestCase):
    OLD = "a" * 40
    NEW = "b" * 40
    DATA = struct.pack("<4sIQQ", b"GGUF", 3, 0, 0) + b"tensor bytes"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "model.gguf"
        self.part = self.root / "model.gguf.part"
        self.manifest = self.root / "model.gguf.part.json"
        self.digest = hashlib.sha256(self.DATA).hexdigest()
        self.metadata = {
            "sha": self.NEW,
            "siblings": [
                {
                    "rfilename": "model.gguf",
                    "size": len(self.DATA),
                    "lfs": {"sha256": self.digest},
                }
            ],
        }
        self.dl = downloads.Download(
            "id", "Model", "owner/model", "model.gguf", self.target
        )
        self.requests = []
        for attribute, value in (("_store", None), ("_downloads", {})):
            patcher = patch.object(downloads, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(downloads, "retry_delay", return_value=0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def seed(self, revision=OLD, digest=True):
        self.part.write_bytes(self.DATA[:26])
        self.manifest.write_text(
            json.dumps(
                {
                    "repo": self.dl.repo,
                    "revision": revision,
                    "file": self.dl.file,
                    "size": len(self.DATA),
                    "sha256": self.digest if digest else None,
                }
            )
        )

    def response(self, request, **kwargs):
        self.requests.append(request)
        ranged = request.get_header("Range")
        offset = int(ranged[6:-1]) if ranged else 0
        headers = {"Content-Length": str(len(self.DATA) - offset)}
        if ranged:
            headers["Content-Range"] = (
                f"bytes {offset}-{len(self.DATA) - 1}/{len(self.DATA)}"
            )
        return Response(self.DATA[offset:], 206 if ranged else 200, headers)

    def run_job(self, metadata=None, opener=None):
        with (
            patch.object(
                downloads,
                "metadata_get",
                side_effect=metadata,
                return_value=self.metadata,
            ) as lookup,
            patch.object(
                downloads.urllib.request, "urlopen", side_effect=opener or self.response
            ),
        ):
            downloads._run(self.dl)
        return lookup

    def test_completed_weights_and_extras_work_offline(self):
        extra = self.root / "projector.gguf"
        self.target.write_bytes(self.DATA)
        extra.write_bytes(self.DATA)
        self.dl.extras = [extra.name]
        lookup = self.run_job(metadata=urllib.error.URLError("offline"))
        self.assertEqual(self.dl.state, "complete", self.dl.detail)
        self.assertEqual(self.dl.total, 2 * len(self.DATA))
        self.assertEqual(self.dl.done, self.dl.total)
        lookup.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_offline_completed_file_still_requires_a_valid_gguf_header(self):
        self.target.write_bytes(b"corrupt file")
        lookup = self.run_job(metadata=urllib.error.URLError("offline"))
        self.assertEqual(self.dl.state, "error")
        self.assertIn("GGUF", self.dl.detail)
        lookup.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b"corrupt file")

    def test_completed_weights_do_not_hide_a_missing_extra(self):
        self.target.write_bytes(self.DATA)
        self.dl.extras = ["projector.gguf"]
        lookup = self.run_job(metadata=urllib.error.URLError("offline"))
        self.assertEqual(self.dl.state, "error")
        self.assertEqual(lookup.call_count, downloads.RETRY_ATTEMPTS)
        self.assertEqual(self.requests, [])

    def test_metadata_retries_connection_reset_and_503_then_resumes(self):
        self.seed(self.NEW)
        error = urllib.error.HTTPError(
            "https://huggingface.co/api/models/owner/model/revision/main?blobs=true",
            503,
            "Unavailable",
            {},
            io.BytesIO(),
        )
        self.addCleanup(error.close)
        lookup = self.run_job(metadata=[ConnectionResetError(), error, self.metadata])
        self.assertEqual(self.dl.state, "complete", self.dl.detail)
        self.assertEqual(lookup.call_count, 3)
        self.assertTrue(error.closed)
        self.assertEqual(self.requests[0].get_header("Range"), "bytes=26-")
        self.assertNotIn("retry", self.dl.detail)

    def test_metadata_retry_exhaustion_preserves_partial_and_manifest(self):
        self.seed()
        original = self.manifest.read_bytes()
        lookup = self.run_job(metadata=urllib.error.URLError("offline"))
        self.assertEqual(lookup.call_count, downloads.RETRY_ATTEMPTS)
        self.assertEqual(self.dl.state, "error")
        self.assertEqual(self.part.read_bytes(), self.DATA[:26])
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(self.requests, [])

    def test_cancel_during_metadata_backoff_keeps_identity(self):
        self.seed()
        original = self.manifest.read_bytes()

        def cancel(_attempt):
            self.dl._cancel.set()
            return 0

        with patch.object(downloads, "retry_delay", side_effect=cancel):
            lookup = self.run_job(metadata=ConnectionResetError())
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(self.dl.state, "cancelled")
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(self.part.read_bytes(), self.DATA[:26])
        self.assertEqual(self.requests, [])

    def test_cancel_after_metadata_response_does_not_touch_partial(self):
        self.seed()
        original = self.manifest.read_bytes()

        def cancel(*_args):
            self.dl._cancel.set()
            return self.metadata

        self.run_job(metadata=cancel)
        self.assertEqual(self.dl.state, "cancelled")
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(self.part.read_bytes(), self.DATA[:26])
        self.assertEqual(self.requests, [])

    def test_http_metadata_error_names_the_actual_api_url_and_is_not_retried(self):
        url = "https://huggingface.co/api/models/owner/model/revision/main?blobs=true"
        error = urllib.error.HTTPError(url, 404, "Missing", {}, io.BytesIO())
        self.addCleanup(error.close)
        lookup = self.run_job(metadata=error)
        self.assertEqual(self.dl.state, "error")
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(self.dl.detail, f"HTTP 404 for {url}")
        self.assertTrue(error.closed)

    def test_exhausted_503_reports_the_actual_api_url(self):
        url = "https://huggingface.co/api/models/owner/model/revision/main?blobs=true"
        error = urllib.error.HTTPError(url, 503, "Unavailable", {}, io.BytesIO())
        self.addCleanup(error.close)
        lookup = self.run_job(metadata=error)
        self.assertEqual(lookup.call_count, downloads.RETRY_ATTEMPTS)
        self.assertEqual(self.dl.detail, f"HTTP 503 for {url}")

    def test_same_content_resumes_across_a_readme_commit(self):
        self.seed()
        self.run_job()
        self.assertEqual(self.dl.state, "complete", self.dl.detail)
        self.assertEqual(self.requests[0].get_header("Range"), "bytes=26-")
        self.assertIn(f"/resolve/{self.NEW}/", self.requests[0].full_url)
        self.assertEqual(self.target.read_bytes(), self.DATA)

    def test_same_content_accepts_case_insensitive_published_hash(self):
        self.seed()
        self.metadata["siblings"][0]["lfs"]["sha256"] = self.digest.upper()
        self.run_job()
        self.assertEqual(self.dl.state, "complete", self.dl.detail)
        self.assertEqual(self.requests[0].get_header("Range"), "bytes=26-")

    def test_same_size_without_a_hash_restarts_across_commits(self):
        self.seed(digest=False)
        self.metadata["siblings"][0].pop("lfs")
        self.run_job()
        self.assertEqual(self.dl.state, "complete", self.dl.detail)
        self.assertIsNone(self.requests[0].get_header("Range"))

    def test_same_revision_without_a_hash_can_resume(self):
        self.seed(self.NEW, digest=False)
        self.metadata["siblings"][0].pop("lfs")
        self.run_job()
        self.assertEqual(self.dl.state, "complete", self.dl.detail)
        self.assertEqual(self.requests[0].get_header("Range"), "bytes=26-")

    def test_content_match_still_requires_the_same_repo_file_and_size(self):
        for field, value in (
            ("repo", "other/repo"),
            ("file", "other.gguf"),
            ("size", 99),
        ):
            with self.subTest(field=field):
                self.seed()
                saved = json.loads(self.manifest.read_text())
                saved[field] = value
                self.manifest.write_text(json.dumps(saved))
                self.run_job()
                self.assertEqual(self.dl.state, "complete", self.dl.detail)
                self.assertIsNone(self.requests[-1].get_header("Range"))
                self.target.unlink()

    def test_pinned_revision_is_persisted_before_transfer_and_restored(self):
        entry = {
            "id": self.dl.id,
            "name": self.dl.name,
            "repo": self.dl.repo,
            "file": self.dl.file,
            "revision": "main",
        }
        with (
            patch.object(config, "STATE_DIR", self.root / "state"),
            patch.object(config, "MODELS_DIR", self.root),
            patch.object(downloads, "_queue", queue.Queue()),
            patch.object(downloads, "ensure_worker"),
        ):
            store = Store()
            self.addCleanup(store.db.close)
            downloads.configure(store)
            self.dl = downloads.start(entry)

            def interrupted(request, **kwargs):
                saved = store.get("download-job", self.dl.id)
                self.assertEqual(saved["entry"]["revision"], self.NEW)
                raise ConnectionResetError("interrupt transfer")

            self.run_job(opener=interrupted)
            self.assertEqual(self.dl.state, "error")
            self.assertEqual(entry["revision"], "main")
            self.assertEqual(
                store.get("download-job", "id")["entry"]["revision"], self.NEW
            )
            downloads._downloads.clear()
            downloads.configure(store)
            restored = downloads._downloads["id"]
            self.assertEqual(restored.revision, self.NEW)
            with patch.object(
                downloads, "metadata_get", return_value=self.metadata
            ) as lookup:
                downloads._pin(restored)
            self.assertEqual(
                lookup.call_args.args[0], f"/owner/model/revision/{self.NEW}"
            )

    def test_pin_persistence_preserves_concurrent_cancellation_state(self):
        with patch.object(config, "STATE_DIR", self.root / "state"):
            store = Store()
        self.addCleanup(store.db.close)
        entry = {"id": "id", "revision": "main"}
        store.put("download-job", "id", {"entry": entry, "state": "cancelled"})
        downloads._store = store
        downloads._downloads["id"] = self.dl
        self.run_job()
        saved = store.get("download-job", "id")
        self.assertEqual(saved["state"], "cancelled")
        self.assertEqual(saved["entry"]["revision"], self.NEW)

    def test_hashing_is_chunked_visible_and_cancellable_without_promotion(self):
        chunks = []
        sha256 = hashlib.sha256
        dl = self.dl

        class CancellingHash:
            def __init__(self):
                self.digest = sha256()

            def update(self, chunk):
                self_test.assertEqual(dl.detail, "verifying")
                chunks.append(chunk)
                self.digest.update(chunk)
                dl._cancel.set()

            def hexdigest(self):
                return self.digest.hexdigest()

        self_test = self
        self.dl.detail = "retry 2 after a reset"
        with (
            patch.object(downloads, "CHUNK", 8),
            patch.object(downloads.hashlib, "sha256", CancellingHash),
        ):
            self.run_job()
        self.assertEqual(chunks, [self.DATA[:8]])
        self.assertEqual(self.dl.state, "cancelled", self.dl.detail)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.part.read_bytes(), self.DATA)
        self.assertEqual(json.loads(self.manifest.read_text())["sha256"], self.digest)
        self.assertFalse(self.root.joinpath("model.gguf.part.json.tmp").exists())
        # The retained complete part can finish after a new process starts.
        self.dl = downloads.Download(
            "id", "Model", "owner/model", "model.gguf", self.target
        )
        self.run_job()
        self.assertEqual(self.dl.state, "complete", self.dl.detail)
        self.assertEqual(self.target.read_bytes(), self.DATA)

    def test_cancel_after_transfer_also_stops_promotion_without_a_hash(self):
        self.metadata["siblings"][0].pop("lfs")

        def completed(*args):
            self.part.write_bytes(self.DATA)
            self.dl.done = len(self.DATA)
            self.dl._cancel.set()
            return True

        with patch.object(downloads, "_attempt", side_effect=completed):
            self.run_job()
        self.assertEqual(self.dl.state, "cancelled")
        self.assertFalse(self.target.exists())
        self.assertTrue(self.part.exists())
        self.assertTrue(self.manifest.exists())

    def test_size_and_hash_failures_both_discard_part_and_manifest(self):
        for data in (self.DATA[:-1], self.DATA[:-1] + b"X"):
            with self.subTest(length=len(data)):
                self.run_job(opener=lambda *_args, body=data, **_kwargs: Response(body))
                self.assertEqual(self.dl.state, "error")
                self.assertFalse(self.target.exists())
                self.assertFalse(self.part.exists())
                self.assertFalse(self.manifest.exists())
        self.run_job()
        self.assertEqual(self.dl.state, "complete", self.dl.detail)
        self.assertIsNone(self.requests[0].get_header("Range"))

    def test_unrelated_malformed_metadata_is_ignored(self):
        self.metadata["siblings"].extend(
            [
                None,
                123,
                "bad",
                {},
                {"rfilename": []},
                {"rfilename": "README.md", "lfs": {"sha256": "invalid"}},
                {"rfilename": "unneeded.gguf", "lfs": ["bad"]},
            ]
        )
        self.run_job()
        self.assertEqual(self.dl.state, "complete", self.dl.detail)
        self.assertEqual(set(self.dl.source), {"model.gguf"})

    def test_invalid_needed_metadata_shapes_fail_without_touching_partials(self):
        sibling = self.metadata["siblings"][0]
        variants = [
            None,
            [],
            {"sha": self.NEW},
            {"sha": self.NEW, "siblings": {}},
            {"sha": self.NEW, "siblings": [None, {}]},
            {"sha": self.NEW, "siblings": [{**sibling, "lfs": []}]},
            {"sha": self.NEW, "siblings": [{**sibling, "lfs": "bad"}]},
            {"sha": self.NEW, "siblings": [sibling, sibling]},
        ]
        for info in variants:
            with self.subTest(info=info):
                self.seed()
                original = self.manifest.read_bytes()
                with patch.object(downloads, "metadata_get", return_value=info):
                    with self.assertRaises(ValueError):
                        downloads._pin(self.dl)
                self.assertEqual(self.part.read_bytes(), self.DATA[:26])
                self.assertEqual(self.manifest.read_bytes(), original)
                self.assertEqual(self.requests, [])

    def test_missing_top_level_size_uses_valid_lfs_size(self):
        sibling = self.metadata["siblings"][0]
        sibling.pop("size")
        sibling["lfs"]["size"] = len(self.DATA)
        self.run_job()
        self.assertEqual(self.dl.state, "complete", self.dl.detail)

    def test_primary_and_extra_transfer_errors_report_the_failing_url(self):
        self.target.write_bytes(self.DATA)
        self.dl.extras = ["projector.gguf"]
        self.metadata["siblings"].append(
            {"rfilename": "projector.gguf", "size": len(self.DATA)}
        )

        def missing(request, **kwargs):
            raise urllib.error.HTTPError(
                request.full_url, 404, "Missing", {}, io.BytesIO()
            )

        self.run_job(opener=missing)
        expected = downloads.url_for(self.dl.repo, "projector.gguf", self.NEW)
        self.assertEqual(self.dl.detail, f"HTTP 404 for {expected}")
