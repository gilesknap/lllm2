"""Check a real release artifact using the panel's engine probe."""

import hashlib
import sys
from pathlib import Path

from lllm2.discovery import probe
from lllm2.engine_install import provenance

binary = Path(sys.argv[1]).resolve()
assert not list(binary.parent.glob("libcuda.*")), "Driver stub must not be shipped"
assert not list(binary.parent.glob("libc.so.*")), "Use the host glibc"
digests = {}
for library in binary.parent.glob("*.so*"):
    if library.is_symlink():
        continue
    with library.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest not in digests, (
        f"Duplicate library payloads: {digests.get(digest)}, {library.name}"
    )
    digests[digest] = library.name
record = probe(binary)
assert not record["error"], record["error"]
for flag in (
    "--model",
    "--host",
    "--port",
    "--ctx-size",
    "--parallel",
    "--device",
    "--gpu-layers",
    "--fit",
    "--fit-target",
    "--jinja",
    "--flash-attn",
    "--batch-size",
    "--ubatch-size",
    "--cache-type-k",
    "--cache-type-v",
):
    assert flag in record["flags"], f"Missing panel flag: {flag}"
assert provenance(binary)["matches_release"], provenance(binary)
print(record["version"])
print("Release metadata and panel engine probe passed.")
