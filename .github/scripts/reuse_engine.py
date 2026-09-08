"""Reuse immutable engine assets by llama.cpp/CUDA pins across lllm2 releases."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from lllm2.engine_install import matches_engine
from lllm2.engine_release import RELEASE_REPOSITORY, asset_name


def reuse(track: str, destination: Path, repository: str) -> bool:
    asset = asset_name(track)
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", f"repos/{repository}/releases"],
        check=True,
        capture_output=True,
        text=True,
    )
    for page in json.loads(result.stdout):
        for release in page:
            names = {
                item["name"]
                for item in release["assets"]
                if item["state"] == "uploaded"
            }
            if not {asset, asset + ".sha256"}.issubset(names):
                continue
            destination.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".reuse-", dir=destination
            ) as temporary:
                work = Path(temporary)
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "download",
                        release["tag_name"],
                        "--repo",
                        repository,
                        "--pattern",
                        asset,
                        "--pattern",
                        asset + ".sha256",
                        "--dir",
                        str(work),
                    ],
                    check=True,
                )
                archive, checksum = work / asset, work / (asset + ".sha256")
                fields = checksum.read_text().split()
                if (
                    len(fields) != 2
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0])
                    or fields[1] != asset
                ):
                    raise RuntimeError(f"Invalid checksum for cached {asset}")
                with archive.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != fields[0].lower():
                    raise RuntimeError(f"Checksum failed for cached {asset}")
                with tarfile.open(archive, "r:gz") as bundle:
                    records = [
                        m
                        for m in bundle.getmembers()
                        if m.name.removeprefix("./") == "lllm2-engine.json"
                    ]
                    if len(records) != 1 or not records[0].isfile():
                        raise RuntimeError(f"Missing engine metadata in cached {asset}")
                    stream = bundle.extractfile(records[0])
                    assert stream is not None
                    with stream:
                        record = json.load(stream)
                if not isinstance(record, dict) or not matches_engine(record, track):
                    raise RuntimeError(f"Engine pins do not match cached {asset}")
                archive.rename(destination / asset)
                checksum.rename(destination / checksum.name)
            print(f"Reused {asset} from release {release['tag_name']} (no compilation)")
            return True
    print(f"No release contains {asset}; this engine pin needs a build")
    return False


if __name__ == "__main__":
    found = reuse(
        sys.argv[1], Path(sys.argv[2]), os.environ.get("GH_REPO", RELEASE_REPOSITORY)
    )
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as handle:
            handle.write(f"found={str(found).lower()}\n")
