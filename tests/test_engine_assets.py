"""CI reuses exact tarball bytes when the engine pins remain unchanged."""

import hashlib
import io
import json
import runpy
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lllm2 import engine_install, engine_release

reuse = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / ".github/scripts/reuse_engine.py")
)["reuse"]


class EngineAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "output"
        self.asset = engine_release.asset_name("13")
        self.record = {
            "requested_ref": engine_release.LLAMA_CPP_REF,
            "cuda_track": engine_release.CUDA_TRACKS["13"],
            "built_for_lllm2_version": "0.2.0",
            "backend": "cuda",
            "architecture": "x86_64",
            "glibc": "2.28",
        }
        self.release = {
            "tag_name": "0.2.0",
            "assets": [
                {"name": name, "state": "uploaded"}
                for name in (self.asset, self.asset + ".sha256")
            ],
        }
        self.corrupt_checksum = False
        self.downloaded = b""

    def gh(self, command, **kwargs):
        if command[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps([[], [self.release]])
            )
        self.assertEqual(command[:3], ["gh", "release", "download"])
        self.assertEqual(command[3], "0.2.0")
        work = Path(command[command.index("--dir") + 1])
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
            data = json.dumps(self.record).encode()
            member = tarfile.TarInfo("./lllm2-engine.json")
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
        self.downloaded = stream.getvalue()
        (work / self.asset).write_bytes(self.downloaded)
        digest = (
            "0" * 64
            if self.corrupt_checksum
            else hashlib.sha256(self.downloaded).hexdigest()
        )
        (work / (self.asset + ".sha256")).write_text(f"{digest}  {self.asset}\n")
        return subprocess.CompletedProcess(command, 0)

    def test_python_only_release_reuses_identical_bytes_from_older_page(self):
        with (
            patch.object(engine_install, "__version__", "0.9.0"),
            patch.object(subprocess, "run", side_effect=self.gh) as run,
        ):
            self.assertTrue(reuse("13", self.output, "owner/repo"))
        self.assertEqual((self.output / self.asset).read_bytes(), self.downloaded)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(len(list(self.output.iterdir())), 2)

    def test_each_track_is_independent_and_pin_changes_miss(self):
        with patch.object(subprocess, "run", side_effect=self.gh) as run:
            self.assertFalse(reuse("12", self.output, "owner/repo"))
            with patch.dict(engine_release.CUDA_TRACKS, {"13": "13.3.2"}):
                self.assertFalse(reuse("13", self.output, "owner/repo"))
            with patch.object(engine_release, "LLAMA_CPP_REF", "b99999"):
                self.assertFalse(reuse("13", self.output, "owner/repo"))
        self.assertEqual(run.call_count, 3)  # Only lookups, no downloads or builds.
        self.assertFalse(self.output.exists())

    def test_incomplete_asset_pair_is_not_reused(self):
        self.release["assets"].pop()
        with patch.object(subprocess, "run", side_effect=self.gh):
            self.assertFalse(reuse("13", self.output, "owner/repo"))

    def test_corrupt_cached_artifact_fails_instead_of_silently_rebuilding(self):
        self.corrupt_checksum = True
        with (
            patch.object(subprocess, "run", side_effect=self.gh),
            self.assertRaisesRegex(RuntimeError, "Checksum failed"),
        ):
            reuse("13", self.output, "owner/repo")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_wrong_cached_metadata_is_rejected(self):
        self.record["requested_ref"] = "wrong"
        with (
            patch.object(subprocess, "run", side_effect=self.gh),
            self.assertRaisesRegex(RuntimeError, "pins do not match"),
        ):
            reuse("13", self.output, "owner/repo")
        self.assertEqual(list(self.output.iterdir()), [])
