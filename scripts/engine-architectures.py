"""Record the GGUF architectures a llama.cpp source tree can load."""

import json
import re
import sys
from pathlib import Path

# Placeholders in the name table that no model file can use.
NOT_MODELS = {"clip", "(unknown)"}


def architectures(source: str) -> list[str]:
    """Read names from the LLM_ARCH_NAMES table in src/llama-arch.cpp."""
    table = re.search(r"LLM_ARCH_NAMES\s*=\s*\{(.*?)\n\};", source, re.S)
    names = re.findall(r'\{\s*LLM_ARCH_\w+\s*,\s*"([^"]+)"\s*\}', table[1] if table else "")
    names = sorted(set(names) - NOT_MODELS)
    # A pin bump that moves the table must fail the build, not ship an empty list.
    if "llama" not in names:
        raise ValueError("Could not read LLM_ARCH_NAMES from llama-arch.cpp")
    return names


if __name__ == "__main__":
    names = architectures(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(json.dumps(names, indent=1) + "\n")
    print(f"Recorded {len(names)} supported architectures.")
