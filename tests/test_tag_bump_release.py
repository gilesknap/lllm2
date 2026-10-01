"""Only a merged llama.cpp bump commit is tagged, once, with the next patch."""

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

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/tag_bump_release.py"
script = runpy.run_path(str(SCRIPT))

REPOSITORY = "gilesknap/lllm2"
SHA = "a" * 40


def pull(**changes):
    record = {
        "number": 90,
        "merged_at": "2026-10-01T00:00:00Z",
        "merge_commit_sha": SHA,
        "head": {"ref": "bot/llama-cpp-bump", "repo": {"full_name": REPOSITORY}},
    }
    record.update(changes)
    return record


class TagBumpReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "github_output"
        self.output.touch()

    def run_script(self, pulls, tags, tags_here=()):
        def run(command, **_kwargs):
            if command[:2] == ["gh", "api"]:
                self.assertEqual(command[2], f"repos/{REPOSITORY}/commits/{SHA}/pulls")
                stdout = json.dumps(pulls)
            elif "--points-at" in command:
                self.assertEqual(command[-1], SHA)
                stdout = "\n".join(tags_here)
            else:
                stdout = "\n".join(tags)
            return subprocess.CompletedProcess(command, 0, stdout=stdout)

        with (
            patch.dict(os.environ, GH_REPO=REPOSITORY, GITHUB_OUTPUT=str(self.output)),
            patch("subprocess.run", side_effect=run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            script["main"](["tag_bump_release.py", SHA])
        return dict(line.split("=", 1) for line in self.output.read_text().splitlines())

    def test_next_patch_follows_the_highest_version(self):
        tags = ["0.8.3", "0.10.0", "v0.9.9", "0.10.0rc1", "nightly", "1.0"]
        self.assertEqual(script["next_patch"](tags), "0.10.1")
        with self.assertRaisesRegex(ValueError, "X.Y.Z"):
            script["next_patch"](["nightly"])

    def test_merged_bump_commit_gets_the_next_patch(self):
        result = self.run_script([pull()], ["0.9.0", "0.8.3"])
        self.assertEqual(result, {"tag": "0.9.1"})

    def test_other_commits_are_never_tagged(self):
        for name, pulls in {
            "no pull": [],
            "other branch": [
                pull(head={"ref": "feature", "repo": {"full_name": REPOSITORY}})
            ],
            "fork": [
                pull(
                    head={"ref": "bot/llama-cpp-bump", "repo": {"full_name": "x/lllm2"}}
                )
            ],
            "unmerged": [pull(merged_at=None)],
            "other commit": [pull(merge_commit_sha="b" * 40)],
        }.items():
            with self.subTest(name):
                self.output.write_text("")
                self.assertEqual(self.run_script(pulls, ["0.9.0"]), {"tag": ""})

    def test_a_bump_is_tagged_at_most_once(self):
        result = self.run_script([pull()], ["0.9.0", "0.9.1"], tags_here=["0.9.1"])
        self.assertEqual(result, {"tag": ""})
