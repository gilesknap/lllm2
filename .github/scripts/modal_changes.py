"""Say whether a pull request changes code that the GPU smoke test covers.

Usage: modal_changes.py BASE HEAD

The ``gpu-smoke`` CI job costs money on every run, so on an ordinary pull
request it runs only when the PR changes one of ``PATHS``: the code that only
the Modal backend runs, the test and sweep that check it, and the CI that runs
them. Shared modules such as the engine installer and the model downloader also
run on Modal, but their unit tests cover them, and including them would make
most pull requests pay. Engine pin changes arrive as llama.cpp bump PRs, which
always run the test.

Lists the files that differ between the two commits, prints those in ``PATHS``
and writes ``modal=true`` or ``modal=false`` to ``GITHUB_OUTPUT``. A renamed
file counts under both names. Uses the standard library and ``git`` only.
"""

import os
import subprocess
import sys

#: Repository files whose changes run the GPU smoke test on a pull request.
PATHS = (
    "src/lllm2/remote.py",  # The remote engine and the provider interface.
    "src/lllm2/proxy.py",  # Proxies the local engine port to the Modal tunnel.
    "src/lllm2/modal_provider.py",
    "src/lllm2/modal_app.py",
    "tests/test_modal_integration.py",
    ".github/scripts/modal_sweep.py",
    ".github/scripts/modal_changes.py",
    ".github/workflows/ci.yml",  # Defines gpu-smoke and when it runs.
)


def changed(base: str, head: str) -> list[str]:
    """Return every path that differs between two commits."""
    return subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", base, head],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def main(argv: list[str]) -> int:
    base, head = argv[1:]
    covered = [path for path in changed(base, head) if path in PATHS]
    if covered:
        print("Changes that gpu-smoke covers:")
        print("\n".join(f"  {path}" for path in covered))
    else:
        print("No change that gpu-smoke covers.")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as handle:
            handle.write(f"modal={str(bool(covered)).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
