"""Move the llama.cpp pin forward to an upstream release tag, never backward.

Usage: bump_llama_cpp.py LATEST_TAG [CONTRACT_FILE]

Only ``LLAMA_CPP_REF`` changes: the engine asset names, the installer and the
Modal image all derive from it. Uses the standard library only, so the
scheduled workflow runs it without installing the project.
"""

import os
import re
import sys
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[2] / "src/lllm2/engine_release.py"
PIN = re.compile(r'^LLAMA_CPP_REF = "(?P<ref>[^"]*)"$', re.MULTILINE)
TAG = re.compile(r"b(?P<build>[0-9]+)")


def build_number(tag: str) -> int:
    """Return the build number of a llama.cpp release tag such as ``b10850``."""
    match = TAG.fullmatch(tag.strip())
    if match is None:
        raise ValueError(f"Not a llama.cpp release tag: {tag!r}")
    return int(match["build"])


def current_ref(text: str) -> str:
    matches = PIN.findall(text)
    if len(matches) != 1:
        raise ValueError(f"Expected one LLAMA_CPP_REF line, found {len(matches)}")
    return matches[0]


def bump(text: str, latest: str) -> str:
    """Return ``text`` pinned to ``latest``, or unchanged if it is not newer."""
    latest = latest.strip()
    if build_number(latest) <= build_number(current_ref(text)):
        return text
    return PIN.sub(f'LLAMA_CPP_REF = "{latest}"', text, count=1)


def main(argv: list[str]) -> None:
    latest = argv[1]
    contract = Path(argv[2]) if len(argv) > 2 else CONTRACT
    text = contract.read_text()
    current = current_ref(text)
    updated = bump(text, latest)
    changed = updated != text
    if changed:
        contract.write_text(updated)
        print(f"Bumped llama.cpp pin from {current} to {latest.strip()}")
    else:
        print(f"llama.cpp pin {current} is current (latest release {latest.strip()})")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as handle:
            handle.write(f"current={current}\n")
            handle.write(f"latest={latest.strip()}\n")
            handle.write(f"changed={str(changed).lower()}\n")


if __name__ == "__main__":
    main(sys.argv)
