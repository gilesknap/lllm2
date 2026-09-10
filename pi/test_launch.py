"""Run with python3 -m unittest discover -s pi -p 'test_*.py'."""

import argparse
import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import launch


class LauncherTests(unittest.TestCase):
    def test_mounts_isolation_and_literal_arguments(self):
        options = argparse.Namespace(image="example:tag", model_port=1920)
        args = ["-p", "spaces; $(touch /tmp/unsafe)", "--model", "a b"]
        result = launch.command(
            options, args, Path("/project space"), Path("/pi"), True
        )
        self.assertEqual(result[-len(args) :], args)
        self.assertIn("type=bind,src=/project space,dst=/workspaces,rw", result)
        self.assertIn("type=bind,src=/pi,dst=/root/.pi,rw", result)
        self.assertEqual(result.count("--mount"), 2)
        self.assertIn("--env=CLAUDE_SANDBOX_EGRESS_JAIL=1", result)
        self.assertIn("--env=CLAUDE_SANDBOX_LOCAL_MODEL_PORT=1920", result)
        self.assertNotIn("--privileged", result)
        self.assertIn("-t", result)
        self.assertNotIn(
            "-t", launch.command(options, [], Path("/project"), Path("/pi"), False)
        )

    def test_ambiguous_mounts_rejected(self):
        for path in ("/a,b", "/a\nb", '/a"b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                launch.mount(Path(path), "/workspaces")

    def test_rootful_and_remote_engines_rejected(self):
        for host in (
            {"security": {"rootless": False}},
            {"security": {"rootless": True}, "serviceIsRemote": True},
            {},
        ):
            with (
                patch.dict(launch.os.environ, {}, clear=True),
                patch.object(launch.shutil, "which", return_value="/bin/podman"),
                patch.object(launch.subprocess, "run") as run,
            ):
                run.return_value.stdout = json.dumps({"host": host})
                with self.assertRaises(RuntimeError):
                    launch.check_runtime()

    def test_dry_run_does_not_touch_config_or_launch(self):
        with (
            patch.object(launch, "check_runtime") as check,
            patch.object(launch.Path, "mkdir") as mkdir,
            patch.object(launch.subprocess, "call") as call,
            patch.object(launch.Path, "cwd", return_value=Path("/project")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(
                launch.main(["--dry-run", "--model-port", "0", "--", "--help"]), 0
            )
            self.assertIn("LOCAL_MODEL_PORT=0", output.getvalue())
            self.assertTrue(output.getvalue().strip().endswith("--help"))
            check.assert_not_called()
            mkdir.assert_not_called()
            call.assert_not_called()

    def test_exit_code_and_argument_forwarding(self):
        with (
            patch.object(launch, "check_runtime"),
            patch.object(launch.Path, "mkdir"),
            patch.object(launch.Path, "cwd", return_value=Path("/project")),
            patch.object(launch.subprocess, "call", return_value=-15) as call,
        ):
            self.assertEqual(launch.main(["--", "-p", "hello world"]), 143)
            self.assertEqual(call.call_args.args[0][-2:], ["-p", "hello world"])

    def test_pat_is_hidden_not_in_argv_and_temporary_file_removed(self):
        token = "test-only-token-not-a-real-credential"
        secret_paths = []

        def run(argv):
            self.assertNotIn(token, " ".join(argv))
            self.assertIn("--env=CLAUDE_SANDBOX_NO_FORGE=0", argv)
            binding = next(a for a in argv if "dst=/run/secrets/pi-github-token" in a)
            secret = Path(binding.split("src=", 1)[1].split(",", 1)[0])
            self.assertEqual(secret.read_text(), token)
            self.assertEqual(secret.stat().st_mode & 0o777, 0o600)
            secret_paths.append(secret)
            return 0

        with (
            patch.object(launch, "check_runtime"),
            patch.object(launch.Path, "cwd", return_value=Path("/project")),
            patch.object(launch.Path, "mkdir"),
            patch.object(launch.getpass, "getpass", return_value=token) as prompt,
            patch.object(launch.subprocess, "call", side_effect=run),
        ):
            self.assertEqual(launch.main(["--pat"]), 0)
            prompt.assert_called_once_with("GitHub PAT (hidden): ")
        self.assertTrue(secret_paths)
        self.assertFalse(secret_paths[0].exists())

    def test_pat_refuses_echoing_fallback(self):
        with (
            patch.object(launch, "check_runtime"),
            patch.object(launch.Path, "cwd", return_value=Path("/project")),
            patch.object(launch.Path, "mkdir"),
            patch.object(
                launch.getpass, "getpass", side_effect=launch.getpass.GetPassWarning
            ),
            patch.object(launch.subprocess, "call") as run,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(launch.main(["--pat"]), 2)
            run.assert_not_called()

    def test_invalid_port(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.main(["--model-port", "65536"])


if __name__ == "__main__":
    unittest.main()
