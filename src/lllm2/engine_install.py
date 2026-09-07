"""Install isolated llama.cpp builds for both the CLI and future panel actions."""

from __future__ import annotations

import json
import os
import platform
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


def _fix_server_compatibility(source: Path) -> list[str]:
    """Repair known server build failures, including GCC 8 new-expression CTAD."""
    adjustments = []
    context = source / "tools/server/server-context.cpp"
    if context.is_file():
        content = context.read_text()
        if "std::setw" in content and not re.search(
            r"#\s*include\s*<iomanip>", content
        ):
            context.write_text("#include <iomanip>\n" + content)
            adjustments.append("Include iomanip for server-context std::setw")

    schema = source / "tools/server/server-schema.cpp"
    if schema.is_file():
        content = schema.read_text()
        # Preserve the member's exact type; field_num handles integers and floats.
        # Restrict this workaround to named params members, not arbitrary expressions.
        fixed = re.sub(
            r'new field_num\(("[^"\n]+"), (params(?:\.[A-Za-z_]\w*)+)\)',
            r"new field_num<decltype(\2)>(\1, \2)",
            content,
        )
        if fixed != content:
            schema.write_text(fixed)
            adjustments.append(
                "Use explicit field_num member types for GCC 8 compatibility"
            )
    return adjustments


def _fix_gcc8_filesystem_link(source: Path) -> list[str]:
    """Link GCC 8's separate filesystem library into its consumers."""
    cmake = source / "CMakeLists.txt"
    marker = "# lllm2: GCC 8 filesystem linkage"
    content = cmake.read_text()
    if marker in content:
        return []
    cmake.write_text(
        content
        + "\n"
        + marker
        + """
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    foreach(target ggml llama-common server-context llama-server-impl llama-server)
        if(TARGET ${target})
            # Append a library (not a linker flag) so it follows object files.
            set_property(TARGET ${target} APPEND PROPERTY LINK_LIBRARIES stdc++fs)
        endif()
    endforeach()
endif()
"""
    )
    return ["Link stdc++fs for GNU C++ compilers older than GCC 9"]


def _fix_vulkan_header_target(source: Path) -> list[str]:
    """Repair upstream revisions that find SPIRV-Headers but omit its target.

    Without the imported target, headers outside the compiler's default search
    paths are invisible. Only modify the known broken CMake form in our clone.
    """
    cmake = source / "ggml/src/ggml-vulkan/CMakeLists.txt"
    content = cmake.read_text()
    original = "target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)"
    if (
        "find_package(SPIRV-Headers CONFIG REQUIRED)" not in content
        or "SPIRV-Headers::SPIRV-Headers" in content
        or original not in content
    ):
        return []
    cmake.write_text(
        content.replace(
            original,
            "target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan SPIRV-Headers::SPIRV-Headers)",
            1,
        )
    )
    return ["Link ggml-vulkan to SPIRV-Headers::SPIRV-Headers"]


def missing_build_tools(backend: str) -> list[str]:
    """Return commands required before an unprivileged source build can run."""
    required = ["git", "cmake", "c++"]
    if backend == "vulkan":
        required.append("glslc")
    return [name for name in required if shutil.which(name) is None]


def prerequisite_hint(backend: str) -> str:
    """Suggest system packages using the distribution's ID and ancestry."""
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    distributions = [release.get("ID", ""), *release.get("ID_LIKE", "").split()]
    manager = None
    for distribution in distributions:
        if distribution in {"rhel", "centos", "rocky", "almalinux", "ol", "fedora"}:
            manager = "dnf" if shutil.which("dnf") or not shutil.which("yum") else "yum"
            break
        if distribution in {"debian", "ubuntu"}:
            manager = "apt"
            break
    if manager is None:
        manager = next(
            (tool for tool in ("dnf", "yum", "apt") if shutil.which(tool)), None
        )

    enterprise_linux_8 = release.get("VERSION_ID", "").split(".")[0] == "8" and bool(
        set(distributions) & {"rhel", "centos", "rocky", "almalinux", "ol"}
    )
    if enterprise_linux_8 and backend == "vulkan":
        # glslc was added to RHEL in 9.0; enabling CRB on EL8 does not provide it.
        return f"""sudo {manager} install git cmake make python3.11 gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ vulkan-headers vulkan-loader-devel
# Select the compilers explicitly; setting PATH alone can leave cc/c++ using GCC 8.
# Keep these exports for the engine build too. They work in bash and zsh.
export CC=/opt/rh/gcc-toolset-13/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
export PATH="/opt/rh/gcc-toolset-13/root/usr/bin:$PATH"
# Build glslc locally: RHEL 8 has no glslc package in its standard repositories.
# Upstream instructions: https://github.com/google/shaderc#getting-and-building-shaderc
# Use a fresh directory so existing builds are left intact.
shaderc_work=$(mktemp -d)
git clone https://github.com/google/shaderc.git "$shaderc_work/source" && (
  cd "$shaderc_work/source" && python3.11 utils/git-sync-deps
) && cmake -S "$shaderc_work/source" -B "$shaderc_work/build" -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python3.11)" -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON -DSHADERC_ENABLE_WERROR_COMPILE=OFF &&
cmake --build "$shaderc_work/build" --target glslc_exe --parallel 2 &&
mkdir -p "$HOME/.local/bin" &&
install -m 755 "$shaderc_work/build/glslc/glslc" "$HOME/.local/bin/glslc"
export PATH="$HOME/.local/bin:$PATH"
glslc --version
# Install the matching SPIR-V headers and their CMake package (no compilation).
cmake -S "$shaderc_work/source/third_party/spirv-headers" -B "$shaderc_work/headers-build" -DCMAKE_INSTALL_PREFIX="$HOME/.local" -DSPIRV_HEADERS_ENABLE_TESTS=OFF -DSPIRV_HEADERS_ENABLE_INSTALL=ON &&
cmake --install "$shaderc_work/headers-build"
export CMAKE_PREFIX_PATH="$HOME/.local${{CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}}"
# Now rerun: lllm2 engines install vulkan
# If the selected llama.cpp ref requires newer Vulkan headers, install
# a matching Vulkan SDK as well: https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html"""

    if manager in {"dnf", "yum"}:
        packages = "git cmake gcc gcc-c++ make"
        if backend == "vulkan":
            packages += " glslc vulkan-headers vulkan-loader-devel spirv-headers-devel"
    elif manager == "apt":
        packages = "git cmake build-essential"
        if backend == "vulkan":
            packages += " glslc libvulkan-dev spirv-headers"
    else:
        hint = "Install Git, CMake, a C/C++ compiler and Make using your system package manager."
        if backend == "vulkan":
            hint += " Also install glslc, the Vulkan development headers and loader, and SPIRV-Headers with its CMake package."
        return hint
    hint = f"sudo {manager} install {packages}"
    if manager in {"dnf", "yum"} and backend == "vulkan":
        hint += (
            "\n# If Vulkan packages are unavailable, enable the development repositories "
            "for your distribution or install the Vulkan SDK."
        )
    return hint


def install(
    backend: str,
    *,
    name: str = "",
    ref: str = "master",
    jobs: int | None = None,
    root: Path | None = None,
    cuda_architectures: str = "native",
) -> Path:
    """Build llama-server in a staging tree, then atomically publish it.

    The build itself never invokes sudo and never modifies an existing engine.
    System prerequisites are deliberately kept as a separate, visible step.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose cuda or vulkan.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", ref):
        raise ValueError(
            "Engine ref may contain only letters, digits, dot, underscore and dash."
        )
    name = name or f"llama-{ref}-{backend}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError(
            "Engine name may contain only letters, digits, dot, underscore and dash."
        )
    if jobs is not None and jobs < 1:
        raise ValueError("Build jobs must be at least one.")
    if backend == "cuda" and not re.fullmatch(
        r"(?:native|all|all-major|[0-9;]+)", cuda_architectures
    ):
        raise ValueError(
            "CUDA architectures must be native, all, all-major, or a semicolon-separated numeric list."
        )
    missing = missing_build_tools(backend)
    if missing:
        raise RuntimeError(
            "Missing build tools: "
            + ", ".join(missing)
            + "\n\nInstall prerequisites, then rerun the engine install command:\n\n"
            + prerequisite_hint(backend)
        )

    root = (root or config.ENGINE_HOME).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        raise FileExistsError(
            f"Engine already exists: {target}. Choose another --name; existing builds are never overwritten."
        )
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
        work = Path(temporary)
        source, build, staged = work / "source", work / "build", work / "installed"
        configure = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            *BACKENDS[backend],
        ]
        if backend == "cuda":
            configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}")
        steps = [
            (
                "clone",
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--branch",
                    ref,
                    "--depth",
                    "1",
                    REPOSITORY,
                    str(source),
                ],
            ),
            ("configure", configure),
            (
                "build",
                [
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    "llama-server",
                    "--parallel",
                    str(jobs or max(1, os.cpu_count() or 1)),
                ],
            ),
        ]
        build_adjustments = []
        for description, command in steps:
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"Engine {description} failed with exit code {error.returncode}."
                ) from error
            if description == "clone":
                build_adjustments = _fix_server_compatibility(source)
                build_adjustments.extend(_fix_gcc8_filesystem_link(source))
                if backend == "vulkan":
                    build_adjustments.extend(_fix_vulkan_header_target(source))
        staged.mkdir()
        for artifact in (build / "bin").iterdir():
            if artifact.is_file() and (
                artifact.name == "llama-server"
                or artifact.suffix == ".so"
                or ".so." in artifact.name
            ):
                shutil.copy2(artifact, staged / artifact.name, follow_symlinks=True)
        binary = staged / "llama-server"
        if not binary.is_file():
            raise RuntimeError("Build completed without build/bin/llama-server.")
        binary.chmod(binary.stat().st_mode | 0o111)
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(staged) + (
            os.pathsep + environment["LD_LIBRARY_PATH"]
            if environment.get("LD_LIBRARY_PATH")
            else ""
        )
        check = subprocess.run(
            [str(binary), "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        if check.returncode:
            raise RuntimeError(
                "Built llama-server could not start: " + check.stderr[-1000:]
            )
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (staged / "lllm2-engine.json").write_text(
            json.dumps(
                {
                    "repository": REPOSITORY,
                    "requested_ref": ref,
                    "revision": revision,
                    "backend": backend,
                    "cuda_architectures": cuda_architectures
                    if backend == "cuda"
                    else None,
                    "build_adjustments": build_adjustments,
                },
                indent=2,
            )
            + "\n"
        )
        staged.rename(target)
    return target / "llama-server"
