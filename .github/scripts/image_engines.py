"""Say whether the container image can carry this commit's pinned engine.

Usage: image_engines.py DIST

The Dockerfile installs the CUDA 12 engine from DIST, where CI puts the
tarballs this run built, or else from the newest published release that
carries it, as ``lllm2 engines install`` does. A pin that a llama.cpp bump PR
merged to main is in neither place until its release tag builds and publishes
the engines, so main builds no image until then.

Prints where the engine comes from and writes ``available=true`` or
``available=false`` to ``GITHUB_OUTPUT``. Uses the standard library and the
``gh`` command only.
"""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[2] / "src/lllm2/engine_release.py"
# The image's track. The Dockerfile installs the same one.
TRACK = "12"


def asset() -> str:
    """Return the image engine's tarball name for the current pins."""
    return runpy.run_path(str(CONTRACT))["asset_name"](TRACK)


def published(repository: str) -> list[set[str]]:
    """Return the uploaded asset names of each published, final release."""
    result = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repository}/releases",
            "--jq",
            '.[] | {draft, prerelease, assets: [.assets[] | select(.state == "uploaded") | .name]}',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    releases = [json.loads(line) for line in result.stdout.splitlines() if line]
    return [
        set(release["assets"])
        for release in releases
        if not release["draft"] and not release["prerelease"]
    ]


def source(dist: Path, repository: str, name: str) -> str | None:
    """Say where the image gets the tarball ``name``, or None if nowhere."""
    pair = {name, name + ".sha256"}
    if all((dist / item).is_file() for item in pair):
        return "this run's build"
    if any(pair <= names for names in published(repository)):
        return "a published release"
    return None


def main(argv: list[str]) -> int:
    name = asset()
    found = source(Path(argv[1]), os.environ["GH_REPO"], name)
    print(f"{name}: {found or 'not built or published yet'}")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as handle:
            handle.write(f"available={str(found is not None).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
