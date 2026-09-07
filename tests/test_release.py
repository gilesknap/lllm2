"""Exercise the release guard used by the installed-wheel check."""

import contextlib
import io
import os
import runpy
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

CHECK_DIST = Path(__file__).resolve().parents[1] / ".github/scripts/check_dist.py"


class ReleaseTagTests(unittest.TestCase):
    def check_distribution(self, ref_type, ref_name):
        with (
            patch.dict(os.environ, GITHUB_REF_TYPE=ref_type, GITHUB_REF_NAME=ref_name),
            patch("subprocess.run") as cli,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            runpy.run_path(str(CHECK_DIST), run_name="__main__")
        self.assertEqual(cli.call_count, 2)

    def test_unprefixed_version_tag(self):
        self.check_distribution("tag", version("lllm2"))

    def test_prefixed_version_tag(self):
        self.check_distribution("tag", "v" + version("lllm2"))

    def test_wrong_or_malformed_tag_is_rejected(self):
        for tag in (
            "9.9.999",
            "v9.9.999",
            "release-" + version("lllm2"),
            "vv" + version("lllm2"),
        ):
            with (
                self.subTest(tag=tag),
                self.assertRaisesRegex(AssertionError, "does not match"),
            ):
                self.check_distribution("tag", tag)

    def test_branch_build_does_not_require_a_version_ref(self):
        self.check_distribution("branch", "main")
