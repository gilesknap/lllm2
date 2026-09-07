"""Install isolated llama.cpp builds for both the CLI and future panel actions."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from . import config

REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
BACKENDS = {
    "cuda": ["-DGGML_CUDA=ON", "-DGGML_NATIVE=OFF"],
    "vulkan": ["-DGGML_VULKAN=ON", "-DGGML_NATIVE=OFF"],
}
CUDA_ROOT_VARIABLES = ("CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH")
CUDA_DEFAULT_ROOTS = ("/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda")
NVIDIA_REPOSITORY = "https://developer.download.nvidia.com/compute/cuda/repos"
# The first CUDA release whose nvcc can generate code for each compute capability.
# A GPU newer than this table asks for the newest toolkit rather than a version.
CUDA_ARCHITECTURE_MINIMUMS = {
    (7, 0): (9, 0),
    (7, 2): (9, 2),
    (7, 5): (10, 0),
    (8, 0): (11, 0),
    (8, 6): (11, 1),
    (8, 7): (11, 4),
    (8, 9): (11, 8),
    (9, 0): (11, 8),
    (10, 0): (12, 8),
    (12, 0): (12, 8),
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


def _search_prefixes() -> list[Path]:
    """Return installation prefixes that may hold development files."""
    prefixes: list[Path] = []
    for variable in ("VULKAN_SDK", "CMAKE_PREFIX_PATH"):
        prefixes.extend(
            Path(entry)
            for entry in os.environ.get(variable, "").split(os.pathsep)
            if entry
        )
    prefixes.extend((Path.home() / ".local", Path("/usr"), Path("/usr/local")))
    return prefixes


def _header_present(relative: str) -> bool:
    """Report whether a header is visible to the compiler's default search."""
    for entry in os.environ.get("CPATH", "").split(os.pathsep):
        if entry and (Path(entry) / relative).is_file():
            return True
    return any(
        (prefix / "include" / relative).is_file() for prefix in _search_prefixes()
    )


def _library_present(name: str) -> bool:
    """Report whether a linkable library exists in a searched library directory."""
    for entry in os.environ.get("LIBRARY_PATH", "").split(os.pathsep):
        if entry and (Path(entry) / name).is_file():
            return True
    patterns = (f"lib*/{name}", f"lib*/*/{name}")
    return any(
        next(prefix.glob(pattern), None) is not None
        for prefix in _search_prefixes()
        for pattern in patterns
    )


def _cmake_package_present(name: str) -> bool:
    """Report whether a CMake config package is installed under a prefix."""
    directories = {name, name.lower()}
    patterns = [
        template.format(directory=directory)
        for directory in directories
        for template in (
            "lib*/cmake/{directory}/*.cmake",
            "lib*/*/cmake/{directory}/*.cmake",
            "share/cmake/{directory}/*.cmake",
            "share/{directory}/cmake/*.cmake",
            "share/{directory}/*.cmake",
        )
    ]
    return any(
        next(prefix.glob(pattern), None) is not None
        for prefix in _search_prefixes()
        for pattern in patterns
    )


def cuda_toolkit_root() -> Path | None:
    """Return an installed CUDA toolkit root, preferring the newest version.

    CMake finds nvcc through PATH or CUDAToolkit_ROOT, so a toolkit unpacked
    under a versioned prefix is invisible until one of those points at it.
    """
    candidates: list[Path] = []
    for variable in CUDA_ROOT_VARIABLES:
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value))
    # With no root variable set, CMake compiles with the nvcc it finds on PATH.
    # Checking a default prefix first would inspect a toolkit the build ignores.
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        candidates.append(Path(nvcc).resolve().parent.parent)
    for root in CUDA_DEFAULT_ROOTS:
        candidates.append(Path(root))
        parent = Path(root).parent
        versioned = [
            entry
            for entry in parent.glob(Path(root).name + "-*")
            if re.fullmatch(r"[\d.]+", entry.name.split("-", 1)[1])
        ]
        candidates.extend(
            sorted(
                versioned,
                key=lambda entry: [
                    int(part) for part in entry.name.split("-", 1)[1].split(".")
                ],
                reverse=True,
            )
        )
    for candidate in candidates:
        if (candidate / "bin/nvcc").is_file():
            return candidate
    return None


def detected_gpus() -> list[tuple[str, tuple[int, int]]]:
    """Return each GPU's name and compute capability as nvidia-smi reports them.

    The driver provides nvidia-smi, so this answers before any toolkit exists.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        report = subprocess.run(
            [smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if report.returncode:
        return []
    gpus = []
    for line in report.stdout.splitlines():
        name, _, capability = line.partition(",")
        match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", capability)
        if match and name.strip():
            gpus.append((name.strip(), (int(match[1]), int(match[2]))))
    return gpus


def required_toolkit_version() -> tuple[int, int] | None:
    """Return the oldest CUDA release that can target every detected GPU.

    None when no GPU was found, or when one is newer than the table knows.
    """
    minimums = [
        CUDA_ARCHITECTURE_MINIMUMS.get(capability) for _, capability in detected_gpus()
    ]
    if not minimums or None in minimums:
        return None
    return max(minimum for minimum in minimums if minimum is not None)


def toolkit_architectures(root: Path | None = None) -> set[tuple[int, int]] | None:
    """Return the compute capabilities the installed nvcc can generate code for.

    Asking nvcc beats comparing release numbers: it answers for the toolkit that
    will actually run the build. None means no nvcc was available to ask.
    """
    nvcc = str(root / "bin/nvcc") if root is not None else shutil.which("nvcc")
    if nvcc is None or not Path(nvcc).is_file():
        return None
    try:
        report = subprocess.run(
            [nvcc, "--list-gpu-arch"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if report.returncode:
        return None
    # compute_86 is 8.6 and compute_120 is 12.0: the last digit is always the minor.
    found = {
        (int(digits[:-1]), int(digits[-1]))
        for digits in re.findall(r"compute_(\d+)", report.stdout)
    }
    return found or None


def unsupported_gpu_architectures(
    root: Path | None = None,
) -> list[tuple[str, tuple[int, int]]]:
    """Return the detected GPUs this CUDA toolkit is too old to generate code for.

    nvcc rejects an unknown architecture once the build reaches a CUDA
    translation unit, long after the clone and configure steps have run.
    """
    architectures = toolkit_architectures(root)
    if architectures is None:
        return []
    return [gpu for gpu in detected_gpus() if gpu[1] not in architectures]


def _host_compiler_version(compiler: str) -> tuple[int, ...] | None:
    """Return the compiler's GNU version, or None when it is not plain GCC.

    The toolkit guards on __GNUC__, so ask the compiler for the macros it will
    define: the c++ alias does not name GCC in its version banner.
    """
    try:
        report = subprocess.run(
            [compiler, "-E", "-dM", "-x", "c++", "-"],
            input="",
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if report.returncode:
        return None
    macros = dict(re.findall(r"^#define (\S+) (.*)$", report.stdout, re.MULTILINE))
    # Clang defines __GNUC__ as well, but the toolkit's ceiling is about real GCC.
    if "__clang__" in macros or "__GNUC__" not in macros:
        return None
    try:
        return tuple(
            int(macros[name])
            for name in ("__GNUC__", "__GNUC_MINOR__", "__GNUC_PATCHLEVEL__")
            if name in macros
        )
    except ValueError:
        return None


def _supported_host_compiler_ceiling(root: Path) -> int | None:
    """Read the newest GCC major version a CUDA toolkit accepts as host compiler."""
    header = root / "include/crt/host_config.h"
    try:
        content = header.read_text(errors="ignore")
    except OSError:
        return None
    ceilings = [int(match) for match in re.findall(r"__GNUC__\s*>\s*(\d+)", content)]
    return max(ceilings) if ceilings else None


def unsupported_host_compiler() -> tuple[str, tuple[int, ...], int] | None:
    """Return the host compiler, its version and CUDA's ceiling when too new.

    nvcc refuses to compile against unsupported GNU releases, and the refusal
    only surfaces once the build reaches a CUDA translation unit.
    """
    root = cuda_toolkit_root()
    if root is None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            return None
        root = Path(nvcc).resolve().parent.parent
    ceiling = _supported_host_compiler_ceiling(root)
    if ceiling is None:
        return None
    compiler = os.environ.get("CXX") or "c++"
    version = _host_compiler_version(compiler)
    if version is None or version[0] <= ceiling:
        return None
    return compiler, version, ceiling


def missing_build_tools(backend: str) -> list[str]:
    """Return the dependencies required before an unprivileged source build can run.

    Names are dependency identifiers rather than packages; prerequisite_hint
    turns them into the commands for the running distribution.
    """
    missing = [name for name in ("git", "cmake", "c++") if shutil.which(name) is None]
    if backend == "cuda" and shutil.which("nvcc") is None:
        missing.append("nvcc")
    if backend == "vulkan":
        if shutil.which("glslc") is None:
            missing.append("glslc")
        if not _header_present("vulkan/vulkan.h"):
            missing.append("vulkan-headers")
        if not _library_present("libvulkan.so"):
            missing.append("vulkan-loader")
        if not _cmake_package_present("SPIRV-Headers"):
            missing.append("spirv-headers")
    return missing


def _package_manager() -> str | None:
    """Choose a package manager from the distribution's ID and ancestry."""
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    distributions = [release.get("ID", ""), *release.get("ID_LIKE", "").split()]
    for distribution in distributions:
        if distribution in {"rhel", "centos", "rocky", "almalinux", "ol", "fedora"}:
            return "dnf" if shutil.which("dnf") or not shutil.which("yum") else "yum"
        if distribution in {"debian", "ubuntu"}:
            return "apt"
    return next((tool for tool in ("dnf", "yum", "apt") if shutil.which(tool)), None)


def _nvidia_repository_slug() -> str | None:
    """Name NVIDIA's repository directory for this distribution and CPU."""
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        return None
    identifier = release.get("ID", "")
    version = release.get("VERSION_ID", "")
    major = version.split(".")[0]
    if identifier == "ubuntu" and version:
        distribution = "ubuntu" + version.replace(".", "")
    elif identifier in {"debian", "fedora"} and major:
        distribution = identifier + major
    elif identifier in {"rhel", "centos", "rocky", "almalinux", "ol"} and major:
        distribution = "rhel" + major
    else:
        return None
    architecture = {"x86_64": "x86_64", "aarch64": "sbsa"}.get(platform.machine())
    if architecture is None:
        return None
    return f"{distribution}/{architecture}"


def _cuda_repository_hint(
    manager: str | None, wanted: tuple[int, int] | None
) -> list[str]:
    """Give the commands that fetch a toolkit from NVIDIA rather than the distribution.

    Distribution packages lag the hardware: Ubuntu 24.04 still carries CUDA 12.0,
    which cannot target anything newer than Hopper.
    """
    slug = _nvidia_repository_slug()
    if slug is None or manager not in {"apt", "dnf", "yum"}:
        return []
    package = f"cuda-toolkit-{wanted[0]}-{wanted[1]}" if wanted else "cuda-toolkit"
    if manager == "apt":
        lines = [
            f"curl -fsSLO {NVIDIA_REPOSITORY}/{slug}/cuda-keyring_1.1-1_all.deb",
            "sudo dpkg -i cuda-keyring_1.1-1_all.deb",
            "sudo apt update",
            f"sudo apt install {package}",
        ]
    else:
        distribution = slug.split("/")[0]
        lines = [
            f"sudo {manager} config-manager --add-repo "
            f"{NVIDIA_REPOSITORY}/{slug}/cuda-{distribution}.repo",
            f"sudo {manager} install {package}",
        ]
    prefix = f"/usr/local/cuda-{wanted[0]}.{wanted[1]}" if wanted else "/usr/local/cuda"
    lines.append(f'export PATH="{prefix}/bin:$PATH"')
    return lines


def _cuda_hint(manager: str | None) -> str:
    """Explain how to expose or install a CUDA toolkit that can target this GPU."""
    root = cuda_toolkit_root()
    outdated = unsupported_gpu_architectures(root)
    if root is not None and not outdated:
        return f"""# A CUDA toolkit is installed at {root}, but nvcc is not on PATH.
# Export these in the shell that runs the engine build. They work in bash and zsh.
export CUDAToolkit_ROOT={root}
export PATH="{root}/bin:$PATH\""""
    wanted = required_toolkit_version()
    lines = []
    for name, (major, minor) in detected_gpus():
        lines.append(f"# {name} reports compute capability {major}.{minor}.")
    if outdated and root is not None:
        lines.append(f"# The toolkit at {root} cannot generate code for it.")
    if wanted:
        lines.append(
            f"# Install CUDA {wanted[0]}.{wanted[1]} or newer, which provides a usable nvcc:"
        )
    else:
        lines.append("# Install the NVIDIA CUDA toolkit, which provides nvcc:")
    if manager == "apt":
        if wanted:
            lines.append(
                "# Only if the distribution carries a new enough release; check with"
            )
            lines.append("# apt-cache policy nvidia-cuda-toolkit")
        lines.append("sudo apt install nvidia-cuda-toolkit")
    elif manager in {"dnf", "yum"}:
        lines.append(f"sudo {manager} install cuda-toolkit")
    repository = _cuda_repository_hint(manager, wanted)
    if repository:
        lines.append("# Otherwise take the release straight from NVIDIA:")
        lines.extend(repository)
    else:
        lines.append(
            "# The distribution package can lag your driver. For a specific version, or"
        )
        lines.append(
            "# if the package is unavailable, use NVIDIA's installer and repositories:"
        )
        lines.append("# https://developer.nvidia.com/cuda-downloads")
    lines.append(
        "# Environment modules on managed systems usually provide it: module load cuda"
    )
    return "\n".join(lines)


def host_compiler_hint() -> str:
    """Explain how to select a host compiler the installed CUDA toolkit accepts."""
    unsupported = unsupported_host_compiler()
    if unsupported is None:
        return ""
    compiler, version, ceiling = unsupported
    printed = ".".join(str(part) for part in version)
    manager = _package_manager()
    lines = [
        f"# {compiler} is GCC {printed}; this CUDA toolkit supports GCC {ceiling} and older.",
        "# Install a supported compiler, then select it for the build:",
    ]
    # CC and CXX select the C++ compiler; nvcc keeps its own default host
    # compiler unless CUDAHOSTCXX names one, so exporting all three is what
    # actually moves the build onto the compiler installed here.
    if manager == "apt":
        lines.append(f"sudo apt install gcc-{ceiling} g++-{ceiling}")
        lines.append(f"export CC=/usr/bin/gcc-{ceiling}")
        lines.append(f"export CXX=/usr/bin/g++-{ceiling}")
        lines.append(f"export CUDAHOSTCXX=/usr/bin/g++-{ceiling}")
    elif manager in {"dnf", "yum"}:
        toolset = f"/opt/rh/gcc-toolset-{ceiling}/root/usr/bin"
        lines.append(f"sudo {manager} install gcc-toolset-{ceiling}")
        lines.append(f"export CC={toolset}/gcc")
        lines.append(f"export CXX={toolset}/g++")
        lines.append(f"export CUDAHOSTCXX={toolset}/g++")
    else:
        lines.append(
            f"# Install GCC {ceiling} or older, then export CC, CXX and"
            " CUDAHOSTCXX to point at it."
        )
    return "\n".join(lines)


APT_PACKAGES = {
    "git": "git",
    "cmake": "cmake",
    "c++": "build-essential",
    "glslc": "glslc",
    "vulkan-headers": "libvulkan-dev",
    "vulkan-loader": "libvulkan-dev",
    "spirv-headers": "spirv-headers",
}
DNF_PACKAGES = {
    "git": "git",
    "cmake": "cmake",
    "c++": "gcc gcc-c++ make",
    "glslc": "glslc",
    "vulkan-headers": "vulkan-headers",
    "vulkan-loader": "vulkan-loader-devel",
    "spirv-headers": "spirv-headers-devel",
}
GENERIC_DESCRIPTIONS = {
    "git": "Git",
    "cmake": "CMake",
    "c++": "a C/C++ compiler and Make",
    "glslc": "the glslc shader compiler",
    "vulkan-headers": "the Vulkan development headers",
    "vulkan-loader": "the Vulkan loader development package",
    "spirv-headers": "SPIRV-Headers with its CMake package",
}


def prerequisite_hint(backend: str, missing: Sequence[str] | None = None) -> str:
    """Suggest the commands that install exactly the missing dependencies."""
    if missing is None:
        missing = missing_build_tools(backend) or ["git", "cmake", "c++"]
    missing = list(missing)
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    distributions = [release.get("ID", ""), *release.get("ID_LIKE", "").split()]
    manager = _package_manager()

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

    packages = APT_PACKAGES if manager == "apt" else DNF_PACKAGES
    sections = []
    if manager in {"apt", "dnf", "yum"}:
        wanted: list[str] = []
        for name in missing:
            package = packages.get(name)
            for entry in package.split() if package else []:
                if entry not in wanted:
                    wanted.append(entry)
        if wanted:
            sections.append(f"sudo {manager} install {' '.join(wanted)}")
    else:
        descriptions = [
            GENERIC_DESCRIPTIONS[name]
            for name in missing
            if name in GENERIC_DESCRIPTIONS
        ]
        if descriptions:
            sections.append(
                "Install "
                + ", ".join(descriptions)
                + " using your system package manager."
            )
    if manager in {"dnf", "yum"} and {"glslc", "vulkan-headers", "vulkan-loader"} & set(
        missing
    ):
        sections.append(
            "# If Vulkan packages are unavailable, enable the development repositories "
            "for your distribution or install the Vulkan SDK."
        )
    if "nvcc" in missing or (backend == "cuda" and unsupported_gpu_architectures()):
        sections.append(_cuda_hint(manager))
    compiler_hint = host_compiler_hint()
    if compiler_hint:
        sections.append(compiler_hint)
    return "\n".join(sections)


def install(
    backend: str,
    *,
    name: str = "",
    ref: str = "master",
    jobs: int | None = None,
    root: Path | None = None,
    cuda_architectures: str = "native",
    check_prerequisites: bool = True,
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
    if check_prerequisites:
        missing = missing_build_tools(backend)
        unsupported = unsupported_host_compiler() if backend == "cuda" else None
        outdated = (
            unsupported_gpu_architectures()
            if backend == "cuda" and "nvcc" not in missing
            else []
        )
        if missing or unsupported or outdated:
            if missing:
                summary = "Missing build dependencies: " + ", ".join(missing)
            elif outdated:
                names = ", ".join(
                    f"{name} (compute capability {major}.{minor})"
                    for name, (major, minor) in outdated
                )
                summary = (
                    "The installed CUDA toolkit cannot generate code for " + names + "."
                )
            else:
                summary = "The default host compiler is too new for the installed CUDA toolkit."
            raise RuntimeError(
                summary
                + "\n\nRun these, then rerun the engine install command:\n\n"
                + prerequisite_hint(backend, missing)
                + "\n\nIf a dependency is installed somewhere these checks cannot see,"
                + " rerun with --skip-checks."
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
