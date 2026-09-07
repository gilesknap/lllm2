import tempfile
import unittest
from pathlib import Path

from lllm2.engine_install import _fix_vulkan_header_target


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


if __name__ == "__main__":
    unittest.main()
