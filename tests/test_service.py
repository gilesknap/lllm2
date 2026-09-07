import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lllm2 import service


class ServiceTests(unittest.TestCase):
    def test_install_and_reinstall_managed_unit(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": directory}),
            patch.object(service.shutil, "which", return_value="/usr/bin/systemctl"),
            patch.object(
                service.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
        ):
            unit = service.install_service(port=8090)
            self.assertEqual(unit, Path(directory) / "systemd/user/lllm2-panel.service")
            self.assertIn('"--port" "8090"', unit.read_text())
            self.assertEqual(unit.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                [call.args[0] for call in run.call_args_list],
                [
                    ["systemctl", "--user", "daemon-reload"],
                    ["systemctl", "--user", "enable", service.UNIT],
                    ["systemctl", "--user", "restart", service.UNIT],
                ],
            )
            run.reset_mock()
            service.install_service(port=8091, start=False)
            self.assertIn('"--port" "8091"', unit.read_text())
            self.assertEqual(run.call_count, 2)
            unit.write_text("# My custom service\n")
            run.reset_mock()
            with self.assertRaises(FileExistsError):
                service.install_service()
            self.assertEqual(unit.read_text(), "# My custom service\n")
            run.assert_not_called()

    def test_quotes_paths_and_keeps_virtualenv_interpreter(self):
        with (
            patch.object(
                service.sys, "executable", "/home/me/a b/%env/$venv/bin/python"
            ),
            patch.dict(
                os.environ,
                {
                    "PATH": "/bin:/a b",
                    "LLLM2_MODELS_DIR": '/data/"models"/%literal/$HOME',
                    "SECRET_TOKEN": "private",
                },
                clear=True,
            ),
        ):
            unit = service.render_unit("127.0.0.1", 8082)
        self.assertIn('ExecStart="/home/me/a b/%%env/$$venv/bin/python"', unit)
        self.assertIn(
            'Environment="LLLM2_MODELS_DIR=/data/\\"models\\"/%%literal/$HOME"', unit
        )
        self.assertNotIn("SECRET_TOKEN", unit)
        with self.assertRaises(ValueError):
            service.render_unit("localhost\nExecStart=bad", 8082)

    def test_systemctl_failure_is_actionable_and_preserves_unit(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"XDG_CONFIG_HOME": directory}),
            patch.object(service.shutil, "which", return_value="/usr/bin/systemctl"),
            patch.object(
                service.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 1, "", "Failed to connect to bus"
                ),
            ) as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to connect to bus"):
                service.install_service()
            self.assertTrue((Path(directory) / "systemd/user" / service.UNIT).is_file())
            self.assertEqual(run.call_count, 1)

    def test_missing_systemd_does_not_run_commands(self):
        with (
            patch.object(service.shutil, "which", return_value=None),
            patch.object(service.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires Linux with systemd"):
                service.install_service()
            run.assert_not_called()
