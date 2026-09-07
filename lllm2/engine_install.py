"""Install isolated llama.cpp builds for both the CLI and future panel actions."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config

REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
BACKENDS = {
    "cuda": ["-DGGML_CUDA=ON", "-DGGML_NATIVE=OFF"],
    "vulkan": ["-DGGML_VULKAN=ON", "-DGGML_NATIVE=OFF"],
}


def missing_build_tools(backend: str) -> list[str]:
    """Return commands required before an unprivileged source build can run."""
    required = ["git", "cmake", "c++"]
    if backend == "vulkan":
        required.append("glslc")
    return [name for name in required if shutil.which(name) is None]


def prerequisite_hint(backend: str) -> str:
    packages = "git cmake build-essential"
    if backend == "vulkan":
        packages += " glslc libvulkan-dev"
    return f"sudo apt install {packages}"


def install(backend: str, *, name: str = "", ref: str = "master",
            jobs: int | None = None, root: Path | None = None,
            cuda_architectures: str = "native") -> Path:
    """Build llama-server in a staging tree, then atomically publish it.

    The build itself never invokes sudo and never modifies an existing engine.
    System prerequisites are deliberately kept as a separate, visible step.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose cuda or vulkan.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", ref):
        raise ValueError("Engine ref may contain only letters, digits, dot, underscore and dash.")
    name = name or f"llama-{ref}-{backend}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError("Engine name may contain only letters, digits, dot, underscore and dash.")
    if jobs is not None and jobs < 1:
        raise ValueError("Build jobs must be at least one.")
    if backend == "cuda" and not re.fullmatch(r"(?:native|all|all-major|[0-9;]+)", cuda_architectures):
        raise ValueError("CUDA architectures must be native, all, all-major, or a semicolon-separated numeric list.")
    missing = missing_build_tools(backend)
    if missing:
        raise RuntimeError("Missing build tools: " + ", ".join(missing) +
                           ". Install prerequisites explicitly with: " + prerequisite_hint(backend))

    root = (root or config.ENGINE_HOME).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        raise FileExistsError(f"Engine already exists: {target}. Choose another --name; existing builds are never overwritten.")
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
        work = Path(temporary)
        source, build, staged = work / "source", work / "build", work / "installed"
        configure = ["cmake", "-S", str(source), "-B", str(build),
                     "-DCMAKE_BUILD_TYPE=Release", *BACKENDS[backend]]
        if backend == "cuda":
            configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}")
        steps = [
            ("clone", ["git", "clone", "--filter=blob:none", "--branch", ref,
                       "--depth", "1", REPOSITORY, str(source)]),
            ("configure", configure),
            ("build", ["cmake", "--build", str(build), "--target", "llama-server",
                       "--parallel", str(jobs or max(1, os.cpu_count() or 1))]),
        ]
        for description, command in steps:
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Engine {description} failed with exit code {error.returncode}.") from error
        staged.mkdir()
        for artifact in (build / "bin").iterdir():
            if artifact.is_file() and (artifact.name == "llama-server" or
                                       artifact.suffix == ".so" or ".so." in artifact.name):
                shutil.copy2(artifact, staged / artifact.name, follow_symlinks=True)
        binary = staged / "llama-server"
        if not binary.is_file():
            raise RuntimeError("Build completed without build/bin/llama-server.")
        binary.chmod(binary.stat().st_mode | 0o111)
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(staged) + (
            os.pathsep + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
        check = subprocess.run([str(binary), "--help"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True, env=environment)
        if check.returncode:
            raise RuntimeError("Built llama-server could not start: " + check.stderr[-1000:])
        revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        (staged / "lllm2-engine.json").write_text(json.dumps({
            "repository": REPOSITORY, "requested_ref": ref, "revision": revision,
            "backend": backend, "cuda_architectures": cuda_architectures if backend == "cuda" else None,
        }, indent=2) + "\n")
        staged.rename(target)
    return target / "llama-server"
