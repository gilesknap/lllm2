"""The scheduled bump moves the llama.cpp pin forward and nothing else."""

import contextlib
import io
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lllm2 import engine_release

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/bump_llama_cpp.py"
script = runpy.run_path(str(SCRIPT))


class BumpLlamaCppTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.contract = self.directory / "engine_release.py"
        self.contract.write_text(script["CONTRACT"].read_text())
        self.output = self.directory / "github_output"
        self.output.touch()

    def run_script(self, latest):
        with (
            patch.dict(os.environ, GITHUB_OUTPUT=str(self.output)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            script["main"](["bump_llama_cpp.py", latest, str(self.contract)])
        return dict(line.split("=", 1) for line in self.output.read_text().splitlines())

    def test_reads_the_real_pin(self):
        text = script["CONTRACT"].read_text()
        self.assertEqual(script["current_ref"](text), engine_release.LLAMA_CPP_REF)

    def test_newer_release_rewrites_only_the_pin(self):
        before = self.contract.read_text()
        current = engine_release.LLAMA_CPP_REF
        newer = f"b{script['build_number'](current) + 1}"
        outputs = self.run_script(newer)
        after = self.contract.read_text()
        self.assertEqual(
            outputs, {"current": current, "latest": newer, "changed": "true"}
        )
        self.assertEqual(after, before.replace(f'"{current}"', f'"{newer}"'))
        namespace = runpy.run_path(str(self.contract))
        self.assertEqual(namespace["LLAMA_CPP_REF"], newer)
        self.assertIn(newer, namespace["asset_name"]("13"))

    def test_current_or_older_release_leaves_the_file_alone(self):
        before = self.contract.read_text()
        current = engine_release.LLAMA_CPP_REF
        older = f"b{script['build_number'](current) - 1}"
        for latest in (current, older):
            with self.subTest(latest=latest):
                self.output.write_text("")
                outputs = self.run_script(latest)
                self.assertEqual(outputs["changed"], "false")
                self.assertEqual(self.contract.read_text(), before)

    def test_compares_build_numbers_not_strings(self):
        text = 'LLAMA_CPP_REF = "b9999"\n'
        self.assertEqual(script["bump"](text, "b10000"), 'LLAMA_CPP_REF = "b10000"\n')

    def test_rejects_unexpected_tags(self):
        for tag in ("", "master", "v1.2.3", "b12a", "b10850; rm -rf /"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                script["bump"]('LLAMA_CPP_REF = "b1"\n', tag)

    def test_rejects_a_missing_or_ambiguous_pin(self):
        for text in ("", 'LLAMA_CPP_REF = "b1"\nLLAMA_CPP_REF = "b2"\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                script["bump"](text, "b3")


if __name__ == "__main__":
    unittest.main()
