"""Download resume identity and completion checks; no network or model required."""

import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lllm2 import downloads


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}


class DownloadIdentityTests(unittest.TestCase):
    OLD = "a" * 40
    NEW = "b" * 40
    DATA = struct.pack("<4sIQQ", b"GGUF", 3, 0, 0) + b"new tensor bytes"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / "model.gguf"
        self.part = self.target.with_suffix(".gguf.part")
        self.manifest = self.part.with_suffix(".part.json")
        self.metadata = {
            "sha": self.NEW,
            "siblings": [
                {
                    "rfilename": "model.gguf",
                    "size": len(self.DATA),
                    "lfs": {"sha256": hashlib.sha256(self.DATA).hexdigest()},
                }
            ],
        }
        self.requests = []
        patcher = patch.object(downloads, "retry_delay", return_value=0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def seed(self, revision, data=None):
        self.part.write_bytes(self.DATA[:26] if data is None else data)
        self.manifest.write_text(
            json.dumps(
                {
                    "repo": "owner/model",
                    "revision": revision,
                    "file": "model.gguf",
                    "size": len(self.DATA),
                    "sha256": hashlib.sha256(self.DATA).hexdigest(),
                }
            )
        )

    def run_download(self, opener=None, revision="main"):
        dl = downloads.Download(
            "id", "Model", "owner/model", "model.gguf", self.target, revision=revision
        )

        def response(request, **kwargs):
            if request.get_method() == "HEAD":
                return Response(b"", headers={"Content-Length": str(len(self.DATA))})
            self.requests.append(request)
            ranged = request.get_header("Range")
            offset = int(ranged[6:-1]) if ranged else 0
            headers = {"Content-Length": str(len(self.DATA) - offset)}
            if ranged:
                headers["Content-Range"] = (
                    f"bytes {offset}-{len(self.DATA) - 1}/{len(self.DATA)}"
                )
            return Response(self.DATA[offset:], 206 if ranged else 200, headers)

        with (
            patch.object(
                downloads, "metadata_get", return_value=self.metadata, create=True
            ),
            patch.object(
                downloads.urllib.request, "urlopen", side_effect=opener or response
            ),
        ):
            downloads._run(dl)
        return dl

    def test_changed_revision_restarts_instead_of_combining_tensor_bytes(self):
        self.seed(self.OLD, self.DATA[:24] + b"OLD")
        dl = self.run_download()
        self.assertEqual(dl.state, "complete", dl.detail)
        self.assertEqual(self.target.read_bytes(), self.DATA)
        self.assertIsNone(self.requests[0].get_header("Range"))
        self.assertIn(f"/resolve/{self.NEW}/", self.requests[0].full_url)
        self.assertFalse(self.manifest.exists())

    def test_same_revision_resumes_after_a_process_restart(self):
        self.seed(self.NEW)
        dl = self.run_download()
        self.assertEqual(dl.state, "complete", dl.detail)
        self.assertEqual(self.target.read_bytes(), self.DATA)
        self.assertEqual(self.requests[0].get_header("Range"), "bytes=26-")

    def test_legacy_partial_without_identity_is_refetched(self):
        self.part.write_bytes(self.DATA[:24] + b"OLD")
        dl = self.run_download()
        self.assertEqual(dl.state, "complete", dl.detail)
        self.assertEqual(self.target.read_bytes(), self.DATA)
        self.assertIsNone(self.requests[0].get_header("Range"))

    def test_malformed_identity_is_refetched(self):
        self.part.write_bytes(b"old")
        self.manifest.write_text("{")
        dl = self.run_download()
        self.assertEqual(dl.state, "complete", dl.detail)
        self.assertIsNone(self.requests[0].get_header("Range"))

    def test_missing_immutable_revision_does_not_touch_partial_bytes(self):
        self.metadata["sha"] = "main"
        self.seed(self.OLD, b"old")
        dl = self.run_download()
        self.assertEqual(dl.state, "error")
        self.assertIn("immutable", dl.detail)
        self.assertEqual(self.part.read_bytes(), b"old")
        self.assertEqual(self.requests, [])
        self.assertFalse(self.target.exists())

    def test_metadata_cannot_change_an_explicitly_pinned_revision(self):
        self.seed(self.OLD, b"old")
        dl = self.run_download(revision=self.OLD)
        self.assertEqual(dl.state, "error")
        self.assertIn("requested revision", dl.detail)
        self.assertEqual(self.part.read_bytes(), b"old")
        self.assertEqual(self.requests, [])

    def test_missing_size_and_invalid_hash_do_not_start_a_transfer(self):
        sibling = self.metadata["siblings"][0]
        for size, digest in ((0, None), (True, None), (len(self.DATA), "not a hash")):
            with self.subTest(size=size, digest=digest):
                sibling.update(size=size, lfs={"sha256": digest})
                dl = self.run_download()
                self.assertEqual(dl.state, "error")
                self.assertEqual(self.requests, [])
                self.assertFalse(self.target.exists())

    def test_published_hash_mismatch_never_promotes_a_valid_gguf_header(self):
        wrong = self.DATA[:24] + b"BAD tensor bytes"
        self.assertEqual(len(wrong), len(self.DATA))
        dl = self.run_download(lambda *args, **kwargs: Response(wrong))
        self.assertEqual(dl.state, "error")
        self.assertIn("SHA-256", dl.detail)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.part.exists())

    def test_published_size_mismatch_never_promotes_a_valid_gguf_header(self):
        self.metadata["siblings"][0].pop("lfs")
        dl = self.run_download(lambda *args, **kwargs: Response(self.DATA[:-1]))
        self.assertEqual(dl.state, "error")
        self.assertIn("size", dl.detail)
        self.assertFalse(self.target.exists())

    def test_interrupted_transfer_keeps_identity_and_pinned_revision_for_resume(self):
        calls = []

        def interrupted(request, **kwargs):
            if request.get_method() == "HEAD":
                return Response(b"", headers={"Content-Length": str(len(self.DATA))})
            calls.append(request)
            if len(calls) == 1:
                return Response(
                    self.DATA[:26], headers={"Content-Length": str(len(self.DATA))}
                )
            return Response(
                b"",
                206,
                {
                    "Content-Length": str(len(self.DATA) - 26),
                    "Content-Range": f"bytes 26-{len(self.DATA) - 1}/{len(self.DATA)}",
                },
            )

        dl = self.run_download(interrupted)
        self.assertEqual(dl.state, "error")
        self.assertEqual(self.part.read_bytes(), self.DATA[:26])
        self.assertEqual(json.loads(self.manifest.read_text())["revision"], self.NEW)
        self.assertTrue(all(f"/resolve/{self.NEW}/" in r.full_url for r in calls))
        self.requests.clear()
        resumed = self.run_download()
        self.assertEqual(resumed.state, "complete", resumed.detail)
        self.assertEqual(self.requests[0].get_header("Range"), "bytes=26-")


if __name__ == "__main__":
    unittest.main()
