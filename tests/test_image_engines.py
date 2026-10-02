"""The image waits for an engine that is neither built in the run nor published."""

import contextlib
import io
import json
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lllm2.engine_release import CUDA_TRACKS, asset_name

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/image_engines.py"
script = runpy.run_path(str(SCRIPT))

REPOSITORY = "gilesknap/lllm2"
# The image carries only the CUDA 12 engine, which every supported driver runs.
ASSET = asset_name("12")
OTHERS = [asset_name(track) for track in CUDA_TRACKS if track != "12"]


def release(names, draft=False, prerelease=False):
    return {"draft": draft, "prerelease": prerelease, "assets": names}


def with_checksums(*names):
    return [item for name in names for item in (name, name + ".sha256")]


class ImageEnginesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dist = Path(temporary.name) / "engine-dist"
        self.dist.mkdir()
        self.output = Path(temporary.name) / "github_output"

    def run_script(self, releases):
        calls = []

        def run(command, **_kwargs):
            calls.append(command)
            self.assertEqual(
                command[:4], ["gh", "api", "--paginate", f"repos/{REPOSITORY}/releases"]
            )
            return subprocess.CompletedProcess(
                command, 0, stdout="".join(json.dumps(r) + "\n" for r in releases)
            )

        with (
            patch.dict(os.environ, GH_REPO=REPOSITORY, GITHUB_OUTPUT=str(self.output)),
            patch("subprocess.run", side_effect=run),
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            self.assertEqual(script["main"](["image_engines.py", str(self.dist)]), 0)
        return self.output.read_text(), printed.getvalue(), calls

    def build(self, *names):
        for name in with_checksums(*names):
            (self.dist / name).touch()

    def test_an_engine_built_in_this_run_needs_no_release_lookup(self):
        self.build(ASSET)
        output, printed, calls = self.run_script([])
        self.assertEqual(output, "available=true\n")
        self.assertEqual(calls, [])
        self.assertEqual(printed, f"{ASSET}: this run's build\n")

    def test_a_published_engine_is_available(self):
        output, printed, calls = self.run_script(
            [release(["lllm2-0.9.0.tar.gz"]), release(with_checksums(ASSET))]
        )
        self.assertEqual(output, "available=true\n")
        self.assertEqual(len(calls), 1)
        self.assertEqual(printed, f"{ASSET}: a published release\n")

    def test_a_merged_bump_waits_for_its_release(self):
        # The other track, built or published, does not stand in for CUDA 12.
        self.build(*OTHERS)
        output, printed, _ = self.run_script([release(with_checksums(*OTHERS))])
        self.assertEqual(output, "available=false\n")
        self.assertEqual(printed, f"{ASSET}: not built or published yet\n")

    def test_drafts_prereleases_and_missing_checksums_do_not_count(self):
        (self.dist / ASSET).touch()  # A tarball without its checksum.
        output, _, _ = self.run_script(
            [
                release(with_checksums(ASSET), draft=True),
                release(with_checksums(ASSET), prerelease=True),
                release([ASSET]),
            ]
        )
        self.assertEqual(output, "available=false\n")

    def test_a_failed_release_lookup_fails_the_job(self):
        with (
            patch.dict(os.environ, GH_REPO=REPOSITORY),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["gh"]),
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            script["main"](["image_engines.py", str(self.dist)])


if __name__ == "__main__":
    unittest.main()
