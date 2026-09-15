import contextlib
import io
import json
import signal
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from typer.testing import CliRunner

from lllm2 import cli, config, modal_app
from lllm2.discovery import CATALOG
from lllm2.modal_provider import ModalProvider
from lllm2.remote import GpuProbe, ProbeCache, catalogue_source, companion_names
from lllm2.settings import Settings
from lllm2.store import Store
from test_modal_provider import META, FakeModal

REMOTE_COMMANDS = (
    ["setup"],
    ["probe", "--gpu", "T4"],
    ["list"],
    ["models"],
    ["stop", "--all"],
    ["download", "qwen3-8b"],
    ["remove", "A/a.gguf"],
    ["remove", "qwen3-8b"],
)
# A catalogue entry with a companion multimodal projector file.
COMPANION_ENTRY = next(e for e in CATALOG if e.get("mmproj"))


class ModalError(Exception):
    pass


class AuthError(ModalError):
    pass


class NotFoundError(ModalError):
    pass


class Refusing:
    """A Modal object whose every method fails authentication."""

    def __getattr__(self, name):
        def refuse(*_args, **_kwargs):
            raise AuthError("Token missing")

        return refuse


class StateDict(dict):
    def put(self, key, value):
        self[key] = value


def fake_modal(state):
    """Build the parts of the ``modal`` module that setup and listing touch."""
    return SimpleNamespace(
        exception=SimpleNamespace(
            Error=ModalError, AuthError=AuthError, NotFoundError=NotFoundError
        ),
        volume=SimpleNamespace(FileEntryType=SimpleNamespace(FILE=1)),
        Dict=SimpleNamespace(from_name=lambda *_a, **_k: state),
        Queue=SimpleNamespace(from_name=lambda *_a, **_k: Refusing()),
        Volume=SimpleNamespace(from_name=lambda *_a, **_k: Refusing()),
        Function=SimpleNamespace(
            from_name=lambda *_a, **_k: SimpleNamespace(hydrate=lambda: None)
        ),
        FunctionCall=Refusing(),
    )


def flat(text):
    """Join help text that the terminal renderer wrapped inside a box."""
    return " ".join(text.replace("│", " ").split())


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
                "/model.gguf", "/llama-server", "CUDA", "CUDA0", 300, "", None
            )
        with patch.object(cli, "_launch", return_value=0) as launch:
            self.assertEqual(self.runner.invoke(cli.app, ["launch"]).exit_code, 0)
            launch.assert_called_once_with("", "", "", "", 180, "", None)
        with patch.object(cli, "_launch", return_value=0) as launch:
            result = self.runner.invoke(
                cli.app,
                [
                    "launch",
                    "--backend",
                    "modal",
                    "--gpu",
                    "L40S",
                    "--idle-timeout",
                    "0",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            launch.assert_called_once_with("", "", "modal", "", 180, "L40S", 0)

    def test_saved_launch_settings_override_tuning_but_not_discovery(self):
        selected = Settings(
            model="/models/example.gguf",
            engine="/current/llama-server",
            backend="CUDA",
            device="CUDA0",
            context=4096,
        )
        saved = selected.dict()
        saved.update(
            engine="/old/llama-server",
            device="CUDA9",
            context=32768,
        )
        store = MagicMock()
        store.get.return_value = saved

        with patch.object(cli, "Store", return_value=store):
            resolved, found = cli._saved_launch_settings(selected)

        self.assertTrue(found)
        self.assertEqual(resolved.context, 32768)
        self.assertEqual(resolved.engine, "/current/llama-server")
        self.assertEqual(resolved.device, "CUDA0")
        store.get.assert_called_once_with("default", "/models/example.gguf|CUDA")
        store.db.close.assert_called_once_with()

    def test_saved_launch_settings_leave_running_experiments_alone(self):
        selected = Settings(
            model="/models/example.gguf",
            engine="/current/llama-server",
            backend="CUDA",
            device="CUDA0",
        )
        running = {"id": "live", "status": "running", "samples": [], "probes": []}
        with (
            TemporaryDirectory() as root,
            patch.object(config, "STATE_DIR", Path(root)),
        ):
            panel_store = Store()
            self.addCleanup(panel_store.db.close)
            panel_store.put("result", "live", running)

            resolved, found = cli._saved_launch_settings(selected)

            self.assertEqual(panel_store.get("result", "live")["status"], "running")
            restarted = Store()
            self.addCleanup(restarted.db.close)
            self.assertEqual(restarted.get("result", "live")["status"], "interrupted")

        self.assertFalse(found)
        self.assertIs(resolved, selected)

    def test_saved_launch_settings_allow_an_empty_state_directory(self):
        selected = Settings(
            model="/models/example.gguf",
            engine="/current/llama-server",
            backend="CUDA",
            device="CUDA0",
        )
        with (
            TemporaryDirectory() as root,
            patch.object(config, "STATE_DIR", Path(root) / "empty-state"),
        ):
            resolved, found = cli._saved_launch_settings(selected)
            self.assertTrue((config.STATE_DIR / "workbench.sqlite3").is_file())

        self.assertFalse(found)
        self.assertIs(resolved, selected)

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
        for name in ("claude", "codex"):
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


class ModalCliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        state = TemporaryDirectory()
        self.addCleanup(state.cleanup)
        patcher = patch.object(config, "STATE_DIR", Path(state.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def main(self, args):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            code = cli.main(["modal", *args])
        return code, errors.getvalue()

    def test_remote_commands_name_the_extra_when_modal_is_missing(self):
        with patch.dict(sys.modules, {"modal": None}):
            for args in REMOTE_COMMANDS:
                with self.subTest(args=args):
                    code, errors = self.main(args)
                    self.assertEqual(code, 2)
                    self.assertIn("pip install 'lllm2[modal]'", errors)

    def test_remote_commands_point_at_setup_without_credentials(self):
        def provider():
            return ModalProvider(fake_modal(Refusing()), deploy=AssertionError)

        with patch("lllm2.modal_provider.create_provider", side_effect=provider):
            for args in REMOTE_COMMANDS:
                with self.subTest(args=args):
                    code, errors = self.main(args)
                    self.assertEqual(code, 2)
                    self.assertIn("modal token new", errors)
                    self.assertIn("lllm2 modal setup", errors)

    def test_setup_deploys_once_and_a_rerun_says_nothing_changed(self):
        state, deploys = StateDict(), []
        with (
            patch.object(modal_app, "deployment_version", return_value="1.0+abc"),
            patch(
                "lllm2.modal_provider.create_provider",
                side_effect=lambda: ModalProvider(
                    fake_modal(state), deploy=lambda: deploys.append(1)
                ),
            ),
        ):
            first = self.runner.invoke(cli.app, ["modal", "setup"])
            rerun = self.runner.invoke(cli.app, ["modal", "setup"])
        self.assertEqual(first.exit_code, 0, first.output)
        self.assertIn("Deployed lllm2 app version 1.0+abc.", first.output)
        self.assertEqual(rerun.exit_code, 0, rerun.output)
        self.assertIn("1.0+abc is already deployed; nothing changed.", rerun.output)
        self.assertEqual(deploys, [1])

    def test_probe_prints_the_gpu_and_engine_and_saves_the_probe(self):
        digest = "ab" * 32
        found = GpuProbe(
            "NVIDIA L4",
            23034,
            {"devices": ["CUDA0"], "sha256": digest, "version": "version: 1 (x)"},
        )
        provider = MagicMock()
        provider.name = "modal"
        provider.probe.return_value = found
        with patch.object(cli, "remote_provider", return_value=provider):
            result = self.runner.invoke(cli.app, ["modal", "probe", "--gpu", "L4"])
        self.assertEqual(result.exit_code, 0, result.output)
        provider.probe.assert_called_once_with("L4")
        for text in ("GPU: NVIDIA L4", "VRAM: 23034 MiB", "CUDA0", digest):
            self.assertIn(text, result.output)
        self.assertIn("Check current Modal pricing", flat(result.output))
        saved = ProbeCache(config.STATE_DIR / "remote-probes.json")
        self.assertEqual(saved.get("modal", "L4"), found)

    def test_probe_rejects_an_unknown_gpu_type_before_contacting_modal(self):
        with patch.object(cli, "remote_provider") as provider:
            code, errors = self.main(["probe", "--gpu", "Z9"])
        self.assertEqual(code, 2)
        self.assertIn("Unknown modal GPU type: Z9. Choose one of: T4, L4", errors)
        provider.assert_not_called()

    def modal_client(self, fake):
        """Serve every provider lookup from one fake ``modal`` module."""
        return patch(
            "lllm2.modal_provider.create_provider",
            side_effect=lambda: ModalProvider(
                fake, poll_interval=0, deploy=lambda: None
            ),
        )

    def test_download_shows_progress_and_a_rerun_downloads_nothing(self):
        fake, entry = FakeModal(), COMPANION_ENTRY
        source = catalogue_source(entry)
        stored = [source.name, *companion_names(entry)]

        def download(name, repo, files, revision):
            self.assertEqual((name, files), (source.name, list(source.files)))

            def poll(call):
                fake.state.put(
                    modal_app.download_key(call.object_id),
                    {"file": files[0], "done_bytes": 10**9, "total_bytes": 2 * 10**9},
                )
                if not call.pending:
                    fake.store.files.update(dict.fromkeys(stored, 2 * 10**9))

            return {"pending": 2, "result": dict(META), "on_poll": poll}

        fake.behaviour["download"] = download
        with self.modal_client(fake):
            first = self.runner.invoke(cli.app, ["modal", "download", entry["id"]])
            rerun = self.runner.invoke(cli.app, ["modal", "download", entry["id"]])
            listed = self.runner.invoke(cli.app, ["modal", "models"])
        self.assertEqual(first.exit_code, 0, first.output)
        self.assertIn(f"Downloading {source.files[0]} inside Modal", first.stderr)
        self.assertIn(f"Stored {source.name} on Modal.", first.stdout)
        self.assertEqual(rerun.exit_code, 0, rerun.output)
        self.assertIn("already stored on Modal; nothing downloaded", rerun.stdout)
        self.assertEqual(rerun.stderr, "")
        # The rerun starts no download call.
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(listed.exit_code, 0, listed.output)
        for path in stored:
            self.assertIn(f"{path}  2.0 GB  {entry['id']}", listed.stdout)

    def test_ctrl_c_cancels_the_download_and_keeps_the_partial_file(self):
        fake = FakeModal()
        partial = catalogue_source(COMPANION_ENTRY).name + ".part"

        def download(*_args):
            def poll(_call):
                fake.store.files[partial] = 1000
                signal.raise_signal(signal.SIGINT)

            return {"pending": 100, "on_poll": poll}

        fake.behaviour["download"] = download
        before = signal.getsignal(signal.SIGINT)
        with self.modal_client(fake):
            result = self.runner.invoke(
                cli.app, ["modal", "download", COMPANION_ENTRY["id"]]
            )
        self.assertEqual(result.exit_code, 130, result.output)
        self.assertIn(
            f"rerun `lllm2 modal download {COMPANION_ENTRY['id']}` to resume",
            flat(result.stderr),
        )
        self.assertEqual([call.cancelled for call in fake.calls.values()], [True])
        self.assertFalse(any(str(key).startswith("heartbeat:") for key in fake.state))
        self.assertEqual(fake.store.files, {partial: 1000})
        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_download_names_close_matches_for_an_unknown_id(self):
        with patch.object(cli, "remote_provider") as provider:
            code, errors = self.main(["download", "qwen3-8"])
        self.assertEqual(code, 2)
        self.assertIn("Unknown catalogue id: qwen3-8. Close matches: qwen3-8b", errors)
        provider.assert_not_called()

    def test_remove_takes_a_catalogue_id_or_stored_name_and_deletes_companions(self):
        fake, entry = FakeModal(), COMPANION_ENTRY
        main, (projector,) = catalogue_source(entry).name, companion_names(entry)
        fake.store.files.update(
            {main: 10, projector + ".part": 5, "other/x.gguf": 7, "extra/y.gguf": 3}
        )
        with self.modal_client(fake):
            listed = self.runner.invoke(cli.app, ["modal", "models", "--json"])
            by_id = self.runner.invoke(cli.app, ["modal", "remove", entry["id"]])
            by_name = self.runner.invoke(cli.app, ["modal", "remove", "other/x.gguf"])
            code, errors = self.main(["remove", "other/x.gguf"])
        rows = {row["name"]: row["catalogue_id"] for row in json.loads(listed.stdout)}
        self.assertEqual(
            rows, {main: entry["id"], "other/x.gguf": None, "extra/y.gguf": None}
        )
        self.assertEqual(by_id.exit_code, 0, by_id.output)
        self.assertIn(f"Removed {main} from Modal.", by_id.stdout)
        self.assertEqual(by_name.exit_code, 0, by_name.output)
        self.assertEqual(fake.store.files, {"extra/y.gguf": 3})
        self.assertEqual(code, 2)
        self.assertIn("No catalogue id or stored model named other/x.gguf", errors)
        self.assertIn("Close matches: extra/y.gguf", errors)
        # A stored name that escapes the Volume resolves to nothing and deletes nothing.
        with self.modal_client(fake):
            for name in ("../extra/y.gguf", "/extra/y.gguf", "extra/../extra/y.gguf"):
                code, errors = self.main(["remove", name])
                self.assertEqual(code, 2, name)
                self.assertIn(f"No catalogue id or stored model named {name}", errors)
        self.assertEqual(fake.store.files, {"extra/y.gguf": 3})

    def test_remove_refuses_a_model_that_a_running_call_serves(self):
        fake = FakeModal()
        name = catalogue_source(COMPANION_ENTRY).name
        fake.store.files[name] = 10
        row = {"id": "fc-9", "model": str(config.MODELS_DIR / name)}
        with (
            self.modal_client(fake),
            patch.object(cli, "describe_calls", return_value=[row]),
        ):
            code, errors = self.main(["remove", COMPANION_ENTRY["id"]])
        self.assertEqual(code, 2)
        self.assertIn(f"Serve call fc-9 uses {name}", errors)
        self.assertIn("lllm2 modal stop", errors)
        self.assertEqual(fake.store.files, {name: 10})

    def test_gpu_help_shows_the_pricing_caveat(self):
        for args in (["launch"], ["modal", "list"], ["modal", "probe"]):
            with self.subTest(args=args):
                result = self.runner.invoke(cli.app, [*args, "--help"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Check current Modal pricing", flat(result.output))


if __name__ == "__main__":
    unittest.main()
