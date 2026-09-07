import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from lllm2 import engine_install
from lllm2.engine_install import (
    _cmake_package_present,
    _fix_gcc8_filesystem_link,
    _fix_server_compatibility,
    _fix_vulkan_header_target,
    _header_present,
    _library_present,
    _supported_host_compiler_ceiling,
    cuda_toolkit_root,
    detected_gpus,
    install,
    missing_build_tools,
    prerequisite_hint,
    required_toolkit_version,
    toolkit_architectures,
    unsupported_gpu_architectures,
    unsupported_host_compiler,
)


class ServerCompatibilityTests(unittest.TestCase):
    def test_fixed_source_compiles_and_preserves_member_types(self):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            server = source / "tools/server"
            server.mkdir(parents=True)
            context = server / "server-context.cpp"
            context.write_text(
                "#include <sstream>\nvoid format() { std::ostringstream s; s << std::setw(8) << 1; }\n"
            )
            schema = server / "server-schema.cpp"
            schema.write_text("""#include <type_traits>
template <typename T = int> struct field_num { field_num(const char *, T &) {} };
struct { int tokens; struct { float temp; unsigned seed; } sampling; } params;
void check() {
    auto i = new field_num("tokens", params.tokens);
    auto f = new field_num("temperature", params.sampling.temp);
    auto u = new field_num("seed", params.sampling.seed);
    static_assert(std::is_same<decltype(i), field_num<int>*>::value);
    static_assert(std::is_same<decltype(f), field_num<float>*>::value);
    static_assert(std::is_same<decltype(u), field_num<unsigned>*>::value);
    delete i; delete f; delete u;
}
""")
            self.assertEqual(len(_fix_server_compatibility(source)), 2)
            self.assertEqual(_fix_server_compatibility(source), [])
            result = subprocess.run(
                [compiler, "-std=c++17", "-fsyntax-only", str(context), str(schema)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_and_already_fixed_sources_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.assertEqual(_fix_server_compatibility(source), [])
            server = source / "tools/server"
            server.mkdir(parents=True)
            contents = {
                "server-context.cpp": "#include <iomanip>\n// std::setw\n",
                "server-schema.cpp": 'new field_num<int>("tokens", params.tokens);\n',
            }
            for name, content in contents.items():
                (server / name).write_text(content)
            self.assertEqual(_fix_server_compatibility(source), [])
            for name, content in contents.items():
                self.assertEqual((server / name).read_text(), content)


class FilesystemCompatibilityTests(unittest.TestCase):
    def test_links_filesystem_across_shared_libraries(self):
        compiler = shutil.which("g++")
        if not compiler or not shutil.which("cmake") or not shutil.which("make"):
            self.skipTest("CMake and C++ build tools unavailable")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            cmake = source / "CMakeLists.txt"
            cmake.write_text("""cmake_minimum_required(VERSION 3.14)
project(filesystem_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_library(ggml SHARED ggml.cpp)
add_library(llama-common SHARED common.cpp)
add_library(server-context STATIC context.cpp)
set_property(TARGET server-context PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(llama-server-impl SHARED server.cpp)
target_link_libraries(llama-server-impl PRIVATE server-context llama-common ggml)
add_executable(llama-server main.cpp)
target_link_libraries(llama-server PRIVATE llama-server-impl)
""")
            for filename, function in (
                ("ggml", "ggml_path"),
                ("common", "common_path"),
                ("context", "context_path"),
            ):
                (source / (filename + ".cpp")).write_text(
                    "#include <filesystem>\n#include <string>\n"
                    f"std::string {function}() {{ return std::filesystem::current_path().string(); }}\n"
                )
            (source / "server.cpp").write_text("""#include <filesystem>
#include <string>
std::string ggml_path(), common_path(), context_path();
bool check() {
    return std::filesystem::path(ggml_path()).has_parent_path()
        && common_path() == context_path();
}
""")
            (source / "main.cpp").write_text(
                "bool check(); int main() { return check() ? 0 : 1; }\n"
            )
            self.assertTrue(_fix_gcc8_filesystem_link(source))
            self.assertEqual(_fix_gcc8_filesystem_link(source), [])
            build = source / "build"
            for command in (
                [
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    "-G",
                    "Unix Makefiles",
                    "-DCMAKE_CXX_COMPILER=" + compiler,
                ],
                ["cmake", "--build", str(build)],
            ):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class VulkanCompatibilityTests(unittest.TestCase):
    def test_only_repairs_missing_imported_target(self):
        broken = (
            "find_package(SPIRV-Headers CONFIG REQUIRED)\n"
            "target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)\n"
        )
        fixed = broken.replace(
            "Vulkan::Vulkan)", "Vulkan::Vulkan SPIRV-Headers::SPIRV-Headers)"
        )
        older = "target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)\n"
        for content, expected in ((broken, fixed), (fixed, fixed), (older, older)):
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as directory,
            ):
                source = Path(directory)
                cmake = source / "ggml/src/ggml-vulkan/CMakeLists.txt"
                cmake.parent.mkdir(parents=True)
                cmake.write_text(content)
                adjustments = _fix_vulkan_header_target(source)
                self.assertEqual(cmake.read_text(), expected)
                self.assertEqual(bool(adjustments), content == broken)
                self.assertEqual(_fix_vulkan_header_target(source), [])


def _reporting_bin(directory: Path, name: str, output: str) -> Path:
    """Create a stub that reports fixed output, standing in for a queried tool."""
    directory.mkdir(parents=True, exist_ok=True)
    stub = directory / name
    # echo is a shell builtin, so the stub still reports with PATH cleared.
    script = "".join(f"echo '{line}'\n" for line in output.splitlines())
    stub.write_text("#!/bin/sh\n" + script)
    stub.chmod(0o755)
    return directory


def _bin(directory: Path, *names: str) -> Path:
    """Create a directory of executable stubs to stand in for installed tools."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        stub = directory / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    return directory


class PreflightDetectionTests(unittest.TestCase):
    """Every dependency the configure step needs is reported before any download."""

    def test_reports_each_missing_dependency_by_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory)
            environment = {"PATH": str(_bin(empty / "bin"))}
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(Path, "home", staticmethod(lambda: empty)),
                mock.patch.object(engine_install, "_search_prefixes", lambda: [empty]),
            ):
                self.assertEqual(
                    missing_build_tools("cuda"), ["git", "cmake", "c++", "nvcc"]
                )
                self.assertEqual(
                    missing_build_tools("vulkan"),
                    [
                        "git",
                        "cmake",
                        "c++",
                        "glslc",
                        "vulkan-headers",
                        "vulkan-loader",
                        "spirv-headers",
                    ],
                )

    def test_nvcc_is_the_only_cuda_requirement_left_when_tools_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = _bin(root / "bin", "git", "cmake", "c++")
            with mock.patch.dict(os.environ, {"PATH": str(tools)}, clear=True):
                self.assertEqual(missing_build_tools("cuda"), ["nvcc"])
            _bin(tools, "nvcc")
            with mock.patch.dict(os.environ, {"PATH": str(tools)}, clear=True):
                self.assertEqual(missing_build_tools("cuda"), [])

    def test_finds_development_files_in_prefixes_and_search_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            header = prefix / "include/vulkan/vulkan.h"
            header.parent.mkdir(parents=True)
            header.write_text("")
            library = prefix / "lib/x86_64-linux-gnu/libvulkan.so"
            library.parent.mkdir(parents=True)
            library.write_text("")
            package = prefix / "share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake"
            package.parent.mkdir(parents=True)
            package.write_text("")
            with (
                mock.patch.dict(
                    os.environ, {"CMAKE_PREFIX_PATH": str(prefix)}, clear=True
                ),
                # Confine the search: a distribution CUDA toolkit installs
                # cuda_runtime.h into /usr/include, which is searched by default.
                mock.patch.object(engine_install, "_search_prefixes", lambda: [prefix]),
            ):
                self.assertTrue(_header_present("vulkan/vulkan.h"))
                self.assertTrue(_library_present("libvulkan.so"))
                self.assertTrue(_cmake_package_present("SPIRV-Headers"))
                self.assertFalse(_header_present("cuda_runtime.h"))


class CudaToolkitTests(unittest.TestCase):
    """A toolkit that CMake cannot see is distinguished from one that is absent."""

    def _toolkit(self, root: Path, gnuc_ceiling: int | None = None) -> Path:
        _bin(root / "bin", "nvcc")
        if gnuc_ceiling is not None:
            header = root / "include/crt/host_config.h"
            header.parent.mkdir(parents=True, exist_ok=True)
            header.write_text(
                textwrap.dedent(f"""
                    #if __GNUC__ > {gnuc_ceiling}
                    #error -- unsupported GNU version! gcc versions later than {gnuc_ceiling} are not supported!
                    #endif
                    """)
            )
        return root

    def test_locates_a_toolkit_named_by_the_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._toolkit(Path(directory) / "cuda")
            with mock.patch.dict(os.environ, {"CUDA_HOME": str(root)}, clear=True):
                self.assertEqual(cuda_toolkit_root(), root)
            # An empty PATH, not an absent one: shutil.which falls back to
            # os.defpath when PATH is unset, so a /usr/bin/nvcc would be found.
            with mock.patch.dict(os.environ, {"PATH": ""}, clear=True):
                self.assertIsNone(cuda_toolkit_root())

    def test_prefers_the_toolkit_on_path_over_a_default_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            on_path = self._toolkit(root / "path-toolkit")
            default = self._toolkit(root / "default-toolkit")
            with (
                mock.patch.object(
                    engine_install, "CUDA_DEFAULT_ROOTS", (str(default),)
                ),
                mock.patch.dict(os.environ, {"PATH": str(on_path / "bin")}, clear=True),
            ):
                # CMake builds with this nvcc, so the checks must read this toolkit.
                self.assertEqual(cuda_toolkit_root(), on_path)
            with mock.patch.object(
                engine_install, "CUDA_DEFAULT_ROOTS", (str(default),)
            ):
                with mock.patch.dict(os.environ, {"PATH": ""}, clear=True):
                    self.assertEqual(cuda_toolkit_root(), default)

    def test_hint_exports_a_found_toolkit_instead_of_installing_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._toolkit(Path(directory) / "cuda")
            with mock.patch.dict(
                os.environ, {"CUDAToolkit_ROOT": str(root), "PATH": ""}, clear=True
            ):
                hint = prerequisite_hint("cuda", ["nvcc"])
            self.assertIn(f"export CUDAToolkit_ROOT={root}", hint)
            self.assertIn(f'export PATH="{root}/bin:$PATH"', hint)
            self.assertNotIn("apt install", hint)

    def test_hint_installs_a_toolkit_when_none_is_present(self):
        with mock.patch.object(engine_install, "cuda_toolkit_root", lambda: None):
            with mock.patch.object(engine_install, "_package_manager", lambda: "apt"):
                self.assertIn(
                    "sudo apt install nvidia-cuda-toolkit",
                    prerequisite_hint("cuda", ["nvcc"]),
                )
            with mock.patch.object(engine_install, "_package_manager", lambda: "dnf"):
                self.assertIn(
                    "sudo dnf install cuda-toolkit", prerequisite_hint("cuda", ["nvcc"])
                )

    def test_hint_names_only_the_missing_packages(self):
        with mock.patch.object(engine_install, "_package_manager", lambda: "apt"):
            hint = prerequisite_hint("vulkan", ["vulkan-headers", "vulkan-loader"])
        self.assertIn("sudo apt install libvulkan-dev", hint)
        self.assertNotIn("cmake", hint)
        self.assertNotIn("build-essential", hint)

    def test_detects_a_host_compiler_the_toolkit_rejects(self):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        version = tuple(
            int(part)
            for part in subprocess.run(
                [compiler, "-dumpfullversion", "-dumpversion"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .split(".")
        )
        major = version[0]
        with tempfile.TemporaryDirectory() as directory:
            rejecting = self._toolkit(Path(directory) / "old", major - 1)
            accepting = self._toolkit(Path(directory) / "new", major + 1)
            self.assertEqual(_supported_host_compiler_ceiling(rejecting), major - 1)
            environment = {"CUDA_HOME": str(rejecting), "CXX": compiler}
            with mock.patch.dict(os.environ, environment, clear=True):
                self.assertEqual(
                    unsupported_host_compiler(), (compiler, version, major - 1)
                )
                hint = prerequisite_hint("cuda", [])
            self.assertIn(f"supports GCC {major - 1} and older", hint)
            self.assertIn("export CXX=", hint)
            # Without CUDAHOSTCXX, CMAKE_CUDA_HOST_COMPILER stays unset and nvcc
            # keeps the default host compiler the toolkit just rejected.
            self.assertIn("export CUDAHOSTCXX=", hint)
            with mock.patch.dict(
                os.environ, {"CUDA_HOME": str(accepting), "CXX": compiler}, clear=True
            ):
                self.assertIsNone(unsupported_host_compiler())


class GpuArchitectureTests(unittest.TestCase):
    """A toolkit that cannot target the installed GPU is caught before the build."""

    def test_reads_names_and_capabilities_from_nvidia_smi(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = _reporting_bin(
                Path(directory) / "bin",
                "nvidia-smi",
                "NVIDIA RTX PRO 6000 Blackwell Server Edition, 12.0\nNVIDIA A100-SXM4-80GB, 8.0",
            )
            with mock.patch.dict(os.environ, {"PATH": str(tools)}, clear=True):
                self.assertEqual(
                    detected_gpus(),
                    [
                        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0)),
                        ("NVIDIA A100-SXM4-80GB", (8, 0)),
                    ],
                )
                # The oldest release that can target both, not just the newest card.
                self.assertEqual(required_toolkit_version(), (12, 8))
            with mock.patch.dict(os.environ, {"PATH": ""}, clear=True):
                self.assertEqual(detected_gpus(), [])
                self.assertIsNone(required_toolkit_version())

    def test_asks_nvcc_which_architectures_it_can_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cuda"
            _reporting_bin(root / "bin", "nvcc", "compute_86\ncompute_90a\ncompute_120")
            self.assertEqual(toolkit_architectures(root), {(8, 6), (9, 0), (12, 0)})
            self.assertIsNone(toolkit_architectures(Path(directory) / "absent"))

    def test_reports_the_gpu_an_old_toolkit_cannot_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cuda"
            _reporting_bin(root / "bin", "nvcc", "compute_86\ncompute_90")
            blackwell = [("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0))]
            with mock.patch.object(engine_install, "detected_gpus", lambda: blackwell):
                self.assertEqual(unsupported_gpu_architectures(root), blackwell)
            ampere = [("NVIDIA GeForce RTX 3090", (8, 6))]
            with mock.patch.object(engine_install, "detected_gpus", lambda: ampere):
                self.assertEqual(unsupported_gpu_architectures(root), [])

    def test_hint_names_the_release_the_gpu_needs_and_where_to_get_it(self):
        blackwell = [("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0))]
        with (
            mock.patch.object(engine_install, "detected_gpus", lambda: blackwell),
            mock.patch.object(engine_install, "cuda_toolkit_root", lambda: None),
            mock.patch.object(engine_install, "_package_manager", lambda: "apt"),
            mock.patch.object(
                engine_install.platform,
                "freedesktop_os_release",
                lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"},
            ),
            mock.patch.object(engine_install.platform, "machine", lambda: "x86_64"),
        ):
            hint = prerequisite_hint("cuda", ["nvcc"])
        self.assertIn("compute capability 12.0", hint)
        self.assertIn("Install CUDA 12.8 or newer", hint)
        self.assertIn("repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb", hint)
        self.assertIn("sudo apt install cuda-toolkit-12-8", hint)
        self.assertIn('export PATH="/usr/local/cuda-12.8/bin:$PATH"', hint)
        # The distribution package stays, qualified: on 24.04 it is CUDA 12.0.
        self.assertIn("apt-cache policy nvidia-cuda-toolkit", hint)

    def test_install_refuses_when_the_toolkit_predates_the_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "engines"
            blackwell = [("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0))]
            with (
                mock.patch.object(engine_install, "missing_build_tools", lambda _: []),
                mock.patch.object(
                    engine_install,
                    "unsupported_gpu_architectures",
                    lambda *_: blackwell,
                ),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    install("cuda", root=root)
            message = str(raised.exception)
            self.assertIn("cannot generate code for", message)
            self.assertIn("compute capability 12.0", message)
            self.assertFalse(root.exists())


class PreflightRefusalTests(unittest.TestCase):
    """Nothing is downloaded while a prerequisite is missing, unless asked."""

    def test_install_refuses_before_cloning_and_skip_checks_bypasses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "engines"
            with mock.patch.object(
                engine_install, "missing_build_tools", lambda backend: ["nvcc"]
            ):
                with self.assertRaises(RuntimeError) as raised:
                    install("cuda", root=root)
                self.assertIn("Missing build dependencies: nvcc", str(raised.exception))
                self.assertIn("--skip-checks", str(raised.exception))
                self.assertFalse(root.exists())
                with mock.patch.object(
                    engine_install.subprocess,
                    "run",
                    side_effect=AssertionError("build started"),
                ):
                    with self.assertRaises(AssertionError):
                        install("cuda", root=root, check_prerequisites=False)


if __name__ == "__main__":
    unittest.main()
