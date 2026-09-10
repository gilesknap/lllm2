#!/usr/bin/env python3
"""Run the shared Pi container launcher directly from a source checkout."""

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lllm2.pi_container import main as launch_main

    return launch_main()


if __name__ == "__main__":
    sys.exit(main())
