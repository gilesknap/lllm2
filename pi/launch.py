#!/usr/bin/env python3
"""Run the shared Pi container launcher directly from a source checkout."""

import runpy
import sys
from pathlib import Path


def main() -> int:
    source = Path(__file__).resolve().parents[1] / "src/lllm2/pi_container.py"
    # Loading this stdlib-only file directly also works in a fresh checkout,
    # before packaging generates lllm2._version.
    return runpy.run_path(str(source))["main"]()


if __name__ == "__main__":
    sys.exit(main())
