"""Download a published CUDA engine matching the installed dependency pins."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from . import __version__, config
from .discovery import engine_environment
from .engine_release import CUDA_TRACKS, LLAMA_CPP_REF, RELEASE_REPOSITORY, asset_name
from .tls import download_context

InstallProgress = Callable[[str, int, int | None], None]


def cuda_track() -> str:
    """Select an artifact supported by both the NVIDIA driver and its GPUs."""
    # PTX targets require a driver supporting the toolkit's major/minor version.
    # Derive the floors from the build pins so future pin bumps stay consistent.
    required = {
        track: tuple(int(part) for part in version.split(".")[:2])
        for track, version in CUDA_TRACKS.items()
    }
    floor = ".".join(str(part) for part in required["12"])
    message = (
        f"Update or install the NVIDIA driver: nvidia-smi must report CUDA {floor} "
        "or newer for the available engine bundles. No CUDA toolkit is required."
    )
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15, check=True
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(message) from error
    match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", result.stdout)
    if not match:
        raise RuntimeError(
            "Could not read the NVIDIA driver's supported CUDA version from nvidia-smi."
        )
    if (int(match[1]), int(match[2])) < required["12"]:
        raise RuntimeError(message)
    if (int(match[1]), int(match[2])) < required["13"]:
        return "12"
    # R580 reports CUDA 13 even on Pascal/Volta. Those GPUs need the CUDA 12
    # artifact: CUDA 13's compiler dropped targets below compute capability 7.5.
    try:
        devices = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "The NVIDIA driver could not report GPU compute capability."
        ) from error
    capabilities = []
    for line in devices.stdout.strip().splitlines():
        capability = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", line)
        if not capability:
            raise RuntimeError(
                "The NVIDIA driver could not report GPU compute capability."
            )
        capabilities.append((int(capability[1]), int(capability[2])))
    if not capabilities:
        raise RuntimeError("The NVIDIA driver could not report GPU compute capability.")
    return "12" if min(capabilities) < (7, 5) else "13"


def matches_engine(record: dict, track: str | None = None) -> bool:
    """Engine identity depends on its pins, independently of Python releases."""
    return (
        record.get("requested_ref") == LLAMA_CPP_REF
        and record.get("backend") == "cuda"
        and record.get("cuda_track") in CUDA_TRACKS.values()
        and (track is None or record.get("cuda_track") == CUDA_TRACKS[track])
        and record.get("architecture") == "x86_64"
        and record.get("glibc") == "2.28"
    )


def provenance(binary: Path) -> dict:
    """Read optional metadata without excluding older or hand-placed engines."""
    try:
        record = json.loads(binary.with_name("lllm2-engine.json").read_text())
        if not isinstance(record, dict):
            return {}
    except (OSError, ValueError):
        return {}
    record["matches_release"] = matches_engine(record)

    return record


def _download(
    url: str, destination: Path, *, progress: InstallProgress | None = None
) -> None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "lllm2"})
        with urllib.request.urlopen(
            request, timeout=60, context=download_context()
        ) as response:
            length = response.headers.get("Content-Length", "")
            total = int(length) if length.isdecimal() and int(length) > 0 else None
            completed = 0
            if progress:
                progress("Downloading engine", completed, total)
            with destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    completed += len(chunk)
                    if progress:
                        progress("Downloading engine", completed, total)
            if total is not None and completed != total:
                raise OSError(
                    f"Download incomplete: received {completed} of {total} bytes"
                )
    except (OSError, urllib.error.URLError) as error:
        if isinstance(error, urllib.error.HTTPError):
            error.close()
        raise RuntimeError(
            f"Could not download release engine from {url}: {error}"
        ) from error


def _release_asset_urls(asset: str) -> tuple[str, str]:
    """Find the newest published release carrying this exact engine pin pair."""
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{RELEASE_REPOSITORY}/releases"
            f"?per_page=100&page={page}"
        )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "lllm2"})
            with urllib.request.urlopen(
                request, timeout=30, context=download_context()
            ) as response:
                releases = json.load(response)
        except (OSError, ValueError) as error:
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            raise RuntimeError(
                f"Could not find published engine releases: {error}"
            ) from error
        if not releases:
            break
        for release in releases:
            if release.get("draft") or release.get("prerelease"):
                continue
            assets = {
                item["name"]: item["browser_download_url"]
                for item in release["assets"]
                if item.get("state") == "uploaded"
            }
            if asset in assets and asset + ".sha256" in assets:
                return assets[asset], assets[asset + ".sha256"]
        page += 1
    raise RuntimeError(f"No published release contains {asset} and its checksum.")


def _unpack(archive: Path, staged: Path) -> None:
    """Extract flat files and validated internal library symlinks."""

    def flat_name(value: str) -> str:
        name = value.removeprefix("./")
        if "/" in name or name in ("", ".", ".."):
            raise RuntimeError(f"Unsafe engine archive path: {value}")
        return name

    with tarfile.open(archive, "r:gz") as bundle:
        members = {}
        for member in bundle.getmembers():
            if member.isdir() and member.name in (".", "./"):
                continue
            name = flat_name(member.name)
            if name in members or not (member.isfile() or member.issym()):
                raise RuntimeError(f"Unsafe engine archive member: {member.name}")
            members[name] = member
        # Validate the entire link graph before writing files. No link may leave
        # the flat engine directory, dangle, form a cycle or point at a special file.
        for name, member in members.items():
            seen = {name}
            while member.issym():
                target = flat_name(member.linkname)
                if target in seen or target not in members:
                    raise RuntimeError(f"Unsafe engine archive link: {name}")
                seen.add(target)
                member = members[target]
        for name, member in members.items():
            if member.issym():
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Missing engine archive member: {member.name}")
            destination = staged / name
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            destination.chmod(0o755 if member.mode & 0o111 else 0o644)
        for name, member in members.items():
            if member.issym():
                (staged / name).symlink_to(flat_name(member.linkname))


def install(
    backend: str,
    *,
    name: str = "",
    root: Path | None = None,
    progress: InstallProgress | None = None,
) -> Path:
    """Verify, stage, probe and atomically publish a release engine."""
    if backend != "cuda":
        raise ValueError(f"Unknown install backend {backend!r}; choose cuda.")
    for label, value in (("ref", LLAMA_CPP_REF), ("name", name or LLAMA_CPP_REF)):
        if value in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(
                f"Engine {label} may contain only letters, digits, dot, underscore and dash."
            )
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("Release CUDA engines require Linux x86_64.")
    report = progress or (lambda _phase, _completed, _total: None)
    report("Checking NVIDIA driver", 0, None)
    track = cuda_track()
    name = name or f"llama-{LLAMA_CPP_REF}-cuda{CUDA_TRACKS[track]}"
    root = (root or config.ENGINE_HOME).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    binary = target / "llama-server"
    if target.exists():
        record = provenance(binary)
        if (
            record.get("matches_release")
            and record.get("cuda_track") == CUDA_TRACKS[track]
            and binary.is_file()
            and os.access(binary, os.X_OK)
        ):
            report("Engine already installed", 0, None)
            return binary
        raise FileExistsError(
            f"Engine already exists: {target}. Choose another --name; existing engines are never overwritten."
        )
    asset = asset_name(track)
    report("Finding published engine", 0, None)
    url, checksum_url = _release_asset_urls(asset)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
        work = Path(temporary)
        archive, checksum, staged = work / asset, work / "checksum", work / "installed"
        report("Downloading checksum", 0, None)
        _download(checksum_url, checksum)
        fields = checksum.read_text().split()
        if (
            len(fields) != 2
            or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0])
            or fields[1] != asset
        ):
            raise RuntimeError("Invalid engine checksum file.")
        report("Downloading engine", 0, None)
        _download(url, archive, progress=progress)
        report("Verifying checksum", 0, None)
        with archive.open("rb") as downloaded:
            digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
        if digest != fields[0].lower():
            raise RuntimeError("Engine checksum verification failed.")
        staged.mkdir()
        report("Extracting engine", 0, None)
        try:
            _unpack(archive, staged)
        except tarfile.TarError as error:
            raise RuntimeError(f"Invalid engine archive: {error}") from error
        candidate = staged / "llama-server"
        record = provenance(candidate)
        if not matches_engine(record, track):
            raise RuntimeError(
                "Engine metadata does not match this lllm2 release and CUDA track."
            )
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError("Engine archive has no executable llama-server.")
        report("Checking engine startup", 0, None)
        try:
            check = subprocess.run(
                [str(candidate), "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=engine_environment(candidate),
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"Downloaded llama-server could not start: {error}"
            ) from error
        if check.returncode:
            raise RuntimeError(
                "Downloaded llama-server could not start: " + check.stderr[-1000:]
            )
        # Release tarballs can be reused unchanged by later Python releases.
        # Keep build provenance distinct from the version doing this installation.
        record.pop("matches_release", None)
        if "built_for_lllm2_version" not in record and "lllm2_version" in record:
            record["built_for_lllm2_version"] = record["lllm2_version"]
        record["lllm2_version"] = __version__
        (staged / "lllm2-engine.json").write_text(json.dumps(record, indent=2) + "\n")
        staged.rename(target)
    report("Engine installed", 0, None)
    return binary
