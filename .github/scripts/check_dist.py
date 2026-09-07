"""Smoke-test the installed wheel with Python's isolated mode (-I)."""

import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

package = files("lllm2")
for name in ("models.json", "recommendations.json"):
    assert json.loads(package.joinpath(name).read_text()), name
for name in (
    "static/index.html",
    "static/panel.js",
    "static/panel.css",
    "templates/qwen3.8-27b.jinja",
):
    assert package.joinpath(name).read_text().strip(), name

installed_version = version("lllm2")
if os.environ.get("GITHUB_REF_TYPE") == "tag":
    tag = os.environ["GITHUB_REF_NAME"]
    assert tag.removeprefix("v") == installed_version, (
        f"Tag {tag} does not match package version {installed_version}"
    )

with tempfile.TemporaryDirectory() as directory:
    subprocess.run(
        [sys.executable, "-I", "-m", "lllm2", "--help"], cwd=directory, check=True
    )
    subprocess.run(
        [str(Path(sys.executable).with_name("lllm2")), "--help"],
        cwd=directory,
        check=True,
    )
print(f"Installed lllm2 {installed_version}: CLI and package assets OK")
