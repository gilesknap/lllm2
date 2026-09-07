import contextlib
import io
import json
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from lllm2 import cli


class CliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help_does_not_start_services(self):
        with patch.object(cli, "_serve") as serve, patch.object(cli, "_launch") as launch:
            for command in ([], ["panel"], ["models"], ["engines"],
                            ["engines", "list"], ["engines", "install"], ["launch"]):
                for flag in ("--help", "-h"):
                    with self.subTest(command=command, flag=flag):
                        result = self.runner.invoke(cli.app, [*command, flag])
                        self.assertEqual(result.exit_code, 0, result.output)
                        self.assertIn("Usage:", result.output)
            root = self.runner.invoke(cli.app, ["--help"])
            for name in ("panel", "models", "engines", "launch"):
                self.assertIn(name, root.output)
            serve.assert_not_called()
            launch.assert_not_called()

    def test_default_and_explicit_panel(self):
        for args, expected in (([], ("127.0.0.1", 8082)),
                               (["--host", "0.0.0.0", "--port", "9000"], ("0.0.0.0", 9000)),
                               (["panel", "--port", "9001"], ("127.0.0.1", 9001))):
            with self.subTest(args=args), patch.object(cli, "_serve") as serve:
                result = self.runner.invoke(cli.app, args)
                self.assertEqual(result.exit_code, 0, result.output)
                serve.assert_called_once_with(*expected)

    def test_discovery_output(self):
        rows = [{"path": "/example/model.gguf", "name": "example"}]
        for args, function in ((["models"], "installed_models"), (["engines", "list"], "engines")):
            with self.subTest(args=args), patch.object(cli, function, return_value=rows):
                result = self.runner.invoke(cli.app, [*args, "--json"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(json.loads(result.output), rows)
                result = self.runner.invoke(cli.app, args)
                self.assertEqual(result.output.strip(), rows[0]["path"])

    def test_install_arguments(self):
        with patch.object(cli, "install", return_value="/build/llama-server") as install:
            result = self.runner.invoke(cli.app, ["engines", "install", "cuda", "--ref", "b123",
                "--name", "test", "--jobs", "4", "--cuda-architectures", "86;89"])
            self.assertEqual(result.exit_code, 0, result.output)
            install.assert_called_once_with("cuda", ref="b123", name="test", jobs=4,
                                            cuda_architectures="86;89")

    def test_missing_tools_print_instructions_without_building(self):
        for version in ("8.10", "9.0"):
            output = io.StringIO()
            with self.subTest(version=version), \
                    patch("lllm2.engine_install.platform.freedesktop_os_release",
                          return_value={"ID": "rhel", "VERSION_ID": version}), \
                    patch("lllm2.engine_install.shutil.which",
                          side_effect=lambda tool: None if tool == "glslc" else "/usr/bin/" + tool), \
                    patch("lllm2.engine_install.subprocess.run") as run, \
                    contextlib.redirect_stderr(output):
                self.assertEqual(cli.main(["engines", "install", "vulkan"]), 2)
                run.assert_not_called()
            hint = output.getvalue()
            self.assertIn("Missing build tools: glslc", hint)
            self.assertIn("sudo dnf install", hint)
            self.assertIn("vulkan-loader-devel", hint)
            package_command = next(line for line in hint.splitlines() if line.startswith("sudo "))
            if version.startswith("8."):
                self.assertNotIn("glslc", package_command)
                self.assertIn("--target glslc_exe", hint)
                self.assertIn("gcc-toolset-13", hint)
            else:
                self.assertIn("glslc", package_command)

    def test_launch_arguments_and_cancellation(self):
        with patch.object(cli, "_launch", return_value=130) as launch:
            result = self.runner.invoke(cli.app, ["launch", "--model", "/model.gguf",
                "--engine", "/llama-server", "--backend", "CUDA", "--device", "CUDA0", "--timeout", "300"])
            self.assertEqual(result.exit_code, 130, result.output)
            launch.assert_called_once_with("/model.gguf", "/llama-server", "CUDA", "CUDA0", 300)
        with patch.object(cli, "_launch", return_value=0) as launch:
            self.assertEqual(self.runner.invoke(cli.app, ["launch"]).exit_code, 0)
            launch.assert_called_once_with("", "", "", "", 180)

    def test_invalid_arguments(self):
        for args in (["engines", "install", "cpu"], ["engines", "install"],
                     ["engines", "install", "cuda", "--jobs", "0"],
                     ["launch", "--backend", "cpu"], ["launch", "--timeout", "0"],
                     ["panel", "--port", "65536"], ["--unknown"]):
            with self.subTest(args=args):
                self.assertEqual(self.runner.invoke(cli.app, args).exit_code, 2)

    def test_entry_point_errors_and_exit_codes(self):
        output = io.StringIO()
        with patch.object(cli, "install", side_effect=RuntimeError("Build failed")), \
                contextlib.redirect_stderr(output):
            self.assertEqual(cli.main(["engines", "install", "cuda"]), 2)
        self.assertIn("Error: Build failed", output.getvalue())
        with patch.object(cli, "_launch", return_value=130):
            self.assertEqual(cli.main(["launch"]), 130)


if __name__ == "__main__":
    unittest.main()
