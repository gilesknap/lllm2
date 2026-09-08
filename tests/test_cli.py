import contextlib
import io
import json
import unittest
from unittest.mock import ANY, patch

from typer.testing import CliRunner

from lllm2 import cli


class CliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help_does_not_start_services(self):
        with (
            patch.object(cli, "_serve") as serve,
            patch.object(cli, "_launch") as launch,
        ):
            for command in (
                [],
                ["panel"],
                ["models"],
                ["engines"],
                ["engines", "list"],
                ["engines", "install"],
                ["launch"],
                ["claude"],
                ["codex"],
                ["pi"],
            ):
                for flag in ("--help", "-h"):
                    with self.subTest(command=command, flag=flag):
                        result = self.runner.invoke(cli.app, [*command, flag])
                        self.assertEqual(result.exit_code, 0, result.output)
                        self.assertIn("Usage:", result.output)
            root = self.runner.invoke(cli.app, ["--help"])
            for name in (
                "panel",
                "models",
                "engines",
                "launch",
                "claude",
                "codex",
                "pi",
            ):
                self.assertIn(name, root.output)
            serve.assert_not_called()
            launch.assert_not_called()

    def test_version_exits_without_starting_panel(self):
        with patch.object(cli, "_serve") as serve:
            result = self.runner.invoke(cli.app, ["--version"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(result.output.strip(), f"lllm2 {cli.__version__}")
            serve.assert_not_called()

    def test_service_install_arguments(self):
        with patch.object(
            cli, "install_service", return_value="/config/lllm2-panel.service"
        ) as install:
            result = self.runner.invoke(
                cli.app,
                [
                    "service",
                    "install",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8090",
                    "--no-start",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            install.assert_called_once_with(host="127.0.0.1", port=8090, start=False)

    def test_default_and_explicit_panel(self):
        for args, expected in (
            ([], ("127.0.0.1", 8082)),
            (["--host", "0.0.0.0", "--port", "9000"], ("0.0.0.0", 9000)),
            (["panel", "--port", "9001"], ("127.0.0.1", 9001)),
        ):
            with self.subTest(args=args), patch.object(cli, "_serve") as serve:
                result = self.runner.invoke(cli.app, args)
                self.assertEqual(result.exit_code, 0, result.output)
                serve.assert_called_once_with(*expected)

    def test_discovery_output(self):
        rows = [{"path": "/example/model.gguf", "name": "example"}]
        for args, function in (
            (["models"], "installed_models"),
            (["engines", "list"], "engines"),
        ):
            with (
                self.subTest(args=args),
                patch.object(cli, function, return_value=rows),
            ):
                result = self.runner.invoke(cli.app, [*args, "--json"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(
                    json.loads(result.output),
                    [dict(rows[0], provenance={})] if function == "engines" else rows,
                )
                result = self.runner.invoke(cli.app, args)
                self.assertEqual(result.output.strip(), rows[0]["path"])

    def test_install_arguments(self):
        with patch.object(
            cli, "install", return_value="/build/llama-server"
        ) as install:
            result = self.runner.invoke(
                cli.app,
                [
                    "engines",
                    "install",
                    "cuda",
                    "--name",
                    "test",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            install.assert_called_once_with(
                "cuda", name="test", force=False, progress=ANY
            )

    def test_force_install_argument(self):
        with patch.object(
            cli, "install", return_value="/build/llama-server"
        ) as install:
            result = self.runner.invoke(
                cli.app, ["engines", "install", "cuda", "--force"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        install.assert_called_once_with("cuda", name="", force=True, progress=ANY)

    def test_install_progress_keeps_stdout_for_the_installed_path(self):
        def install(_backend, *, name, force, progress):
            progress("Checking NVIDIA driver", 0, None)
            progress("Downloading engine", 0, None)
            progress("Downloading engine", 500_000, 1_000_000)
            progress("Downloading engine", 1_000_000, 1_000_000)
            progress("Verifying checksum", 0, None)
            progress("Extracting engine", 0, None)
            progress("Checking engine startup", 0, None)
            progress("Engine installed", 0, None)
            return "/build/llama-server"

        with patch.object(cli, "install", side_effect=install):
            result = self.runner.invoke(cli.app, ["engines", "install", "cuda"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout, "/build/llama-server\n")
        for message in (
            "Checking NVIDIA driver",
            "Downloading engine",
            "Downloaded 1.0 MB / 1.0 MB (100%)",
            "Verifying checksum",
            "Extracting engine",
            "Checking engine startup",
            "Engine installed",
        ):
            self.assertIn(message, result.stderr)
        self.assertEqual(result.stderr.count("Downloading engine"), 1)
        self.assertNotIn("\x1b[", result.stderr)

    def test_unknown_size_download_has_periodic_log_updates(self):
        def install(_backend, *, name, force, progress):
            progress("Downloading engine", 0, None)
            progress("Downloading engine", 1_000_000, None)
            progress("Downloading engine", 2_000_000, None)
            return "/build/llama-server"

        with (
            patch.object(cli, "install", side_effect=install),
            patch.object(cli.time, "monotonic", side_effect=[0, 0, 11, 12]),
        ):
            result = self.runner.invoke(cli.app, ["engines", "install", "cuda"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Downloaded 1.0 MB", result.stderr)
        self.assertNotIn("Downloaded 2.0 MB", result.stderr)
        self.assertNotIn("%", result.stderr)

    def test_removed_install_options_are_rejected(self):
        for args in (
            ["vulkan"],
            ["cuda", "--ref", "master"],
            ["cuda", "--jobs", "2"],
            ["cuda", "--cuda-architectures", "native"],
        ):
            with self.subTest(args=args):
                self.assertEqual(
                    self.runner.invoke(
                        cli.app, ["engines", "install", *args]
                    ).exit_code,
                    2,
                )

    def test_launch_arguments_and_cancellation(self):
        with patch.object(cli, "_launch", return_value=130) as launch:
            result = self.runner.invoke(
                cli.app,
                [
                    "launch",
                    "--model",
                    "/model.gguf",
                    "--engine",
                    "/llama-server",
                    "--backend",
                    "CUDA",
                    "--device",
                    "CUDA0",
                    "--timeout",
                    "300",
                ],
            )
            self.assertEqual(result.exit_code, 130, result.output)
            launch.assert_called_once_with(
                "/model.gguf", "/llama-server", "CUDA", "CUDA0", 300
            )
        with patch.object(cli, "_launch", return_value=0) as launch:
            self.assertEqual(self.runner.invoke(cli.app, ["launch"]).exit_code, 0)
            launch.assert_called_once_with("", "", "", "", 180)

    def test_invalid_arguments(self):
        for args in (
            ["engines", "install", "cpu"],
            ["engines", "install"],
            ["engines", "install", "cuda", "--jobs", "0"],
            ["launch", "--backend", "cpu"],
            ["launch", "--timeout", "0"],
            ["panel", "--port", "65536"],
            ["--unknown"],
        ):
            with self.subTest(args=args):
                self.assertEqual(self.runner.invoke(cli.app, args).exit_code, 2)

    def test_entry_point_errors_and_exit_codes(self):
        output = io.StringIO()
        with (
            patch.object(cli, "install", side_effect=RuntimeError("Build failed")),
            contextlib.redirect_stderr(output),
        ):
            self.assertEqual(cli.main(["engines", "install", "cuda"]), 2)
        self.assertIn("Error: Build failed", output.getvalue())
        with patch.object(cli, "_launch", return_value=130):
            self.assertEqual(cli.main(["launch"]), 130)

    def test_harness_argument_forwarding(self):
        for name in ("claude", "codex", "pi"):
            for args in (
                ["-p", "a prompt with spaces", "--model", "override"],
                ["exec", "--help"],
                ["--", "--help"],
                ["--unknown=value", "$(literal)", "--", "-x"],
            ):
                with (
                    self.subTest(name=name, args=args),
                    patch.object(cli, "run_harness", return_value=7) as run,
                ):
                    result = self.runner.invoke(cli.app, [name, *args])
                    self.assertEqual(result.exit_code, 7, result.output)
                    run.assert_called_once_with(
                        name, args[1:] if args[0] == "--" else args
                    )


if __name__ == "__main__":
    unittest.main()
