import tempfile
import unittest
import shutil
import subprocess
from pathlib import Path

from lllm2.engine_install import (
    _fix_gcc8_filesystem_link, _fix_server_compatibility, _fix_vulkan_header_target,
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
            context.write_text('#include <sstream>\nvoid format() { std::ostringstream s; s << std::setw(8) << 1; }\n')
            schema = server / "server-schema.cpp"
            schema.write_text('''#include <type_traits>
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
''')
            self.assertEqual(len(_fix_server_compatibility(source)), 2)
            self.assertEqual(_fix_server_compatibility(source), [])
            result = subprocess.run([compiler, "-std=c++17", "-fsyntax-only", str(context), str(schema)],
                                    capture_output=True, text=True)
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
            cmake.write_text('''cmake_minimum_required(VERSION 3.14)
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
''')
            for filename, function in (("ggml", "ggml_path"), ("common", "common_path"),
                                       ("context", "context_path")):
                (source / (filename + ".cpp")).write_text(
                    '#include <filesystem>\n#include <string>\n'
                    f'std::string {function}() {{ return std::filesystem::current_path().string(); }}\n')
            (source / "server.cpp").write_text('''#include <filesystem>
#include <string>
std::string ggml_path(), common_path(), context_path();
bool check() {
    return std::filesystem::path(ggml_path()).has_parent_path()
        && common_path() == context_path();
}
''')
            (source / "main.cpp").write_text('bool check(); int main() { return check() ? 0 : 1; }\n')
            self.assertTrue(_fix_gcc8_filesystem_link(source))
            self.assertEqual(_fix_gcc8_filesystem_link(source), [])
            build = source / "build"
            for command in (
                ["cmake", "-S", str(source), "-B", str(build), "-G", "Unix Makefiles",
                 "-DCMAKE_CXX_COMPILER=" + compiler],
                ["cmake", "--build", str(build)],
            ):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class VulkanCompatibilityTests(unittest.TestCase):
    def test_only_repairs_missing_imported_target(self):
        broken = ("find_package(SPIRV-Headers CONFIG REQUIRED)\n"
                  "target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)\n")
        fixed = broken.replace("Vulkan::Vulkan)", "Vulkan::Vulkan SPIRV-Headers::SPIRV-Headers)")
        older = "target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)\n"
        for content, expected in ((broken, fixed), (fixed, fixed), (older, older)):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                source = Path(directory)
                cmake = source / "ggml/src/ggml-vulkan/CMakeLists.txt"
                cmake.parent.mkdir(parents=True)
                cmake.write_text(content)
                adjustments = _fix_vulkan_header_target(source)
                self.assertEqual(cmake.read_text(), expected)
                self.assertEqual(bool(adjustments), content == broken)
                self.assertEqual(_fix_vulkan_header_target(source), [])


if __name__ == "__main__":
    unittest.main()
