"""Remote backend settings, engine factory, bench, results, ownership and CLI.

Every test runs without a local GPU: local probes raise, and the fake provider
serves a fake llama-server process.
"""

import contextlib
import copy
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from fake_remote import ENGINE, FakeProvider, free_port
from lllm2 import cli, config, modal_app, remote
from lllm2.app import App
from lllm2.backends import create_engine, engine_serves
from lllm2.bench import Bench, suite
from lllm2.defaults import starting_defaults
from lllm2.engine import LocalEngine
from lllm2.modal_provider import CREDENTIALS_MESSAGE
from lllm2.recommendations import PROFILES
from lllm2.remote import CallRecords, RemoteEngine, catalogue_sources
from lllm2.settings import Settings, default_key

ENTRY = {
    "id": "example",
    "name": "example",
    "repo": "fake/example-GGUF",
    "file": "model.gguf",
    "max_ctx": 65536,
    "mtp": False,
}


def eventually(check, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


def no_local_probe(*args, **kwargs):
    raise AssertionError("A remote backend must not probe local hardware or files.")


@pytest.fixture
def no_local_gpu():
    """Make every local GPU, binary and checkpoint probe fail loudly."""
    with contextlib.ExitStack() as stack:
        for name in (
            "lllm2.engine.hardware",
            "lllm2.engine.probe",
            "lllm2.engine.metadata",
            "lllm2.settings.probe",
            "lllm2.settings.metadata",
            "lllm2.settings.cache_kernel_support",
            "lllm2.settings.cuda_graph_support",
            "lllm2.settings.engine_environment",
            "lllm2.recommendations.hardware",
            "lllm2.recommendations.probe",
            "lllm2.recommendations.fingerprint",
            "lllm2.defaults.hardware",
            "lllm2.defaults.command",
        ):
            stack.enter_context(patch(name, side_effect=no_local_probe))
        yield


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


@pytest.fixture
def provider(state):
    made = FakeProvider(state / "remote")
    with patch.dict(remote.PROVIDERS, {"fake": lambda: made}):
        yield made
    made.close()


@pytest.fixture
def engines():
    made = []
    yield made
    for engine in made:
        engine.shutdown()


def remote_settings(**values):
    return Settings.parse(
        {
            "model": str(config.MODELS_DIR / "example" / "model.gguf"),
            "backend": "fake",
            "gpu_type": "FAKE-24",
        }
        | values
    )


def make_engine(provider, engines, s=None, **options):
    engine = create_engine(
        s or remote_settings(),
        catalogue=[ENTRY],
        port=free_port(),
        poll_interval=0.05,
        **options,
    )
    engines.append(engine)
    return engine


class TestSettings:
    def test_remote_backend_needs_a_known_gpu_type(self, provider):
        s = remote_settings(idle_timeout_minutes=5)
        assert s.remote and s.gpu_backend == "CUDA"
        assert (s.gpu_type, s.idle_timeout_minutes) == ("FAKE-24", 5)
        assert Settings.parse({"backend": "modal", "gpu_type": "L40S"}).remote
        for values, message in [
            ({"gpu_type": ""}, "Choose a fake GPU type"),
            ({"gpu_type": "H100"}, "Unknown fake GPU type"),
            ({"backend": "modal", "gpu_type": "FAKE-24"}, "Unknown modal GPU type"),
            ({"backend": "elsewhere"}, "Invalid backend"),
        ]:
            with pytest.raises(ValueError, match=message):
                remote_settings(**values)

    def test_local_only_options_are_rejected_remotely(self, provider):
        with pytest.raises(ValueError, match="leave the engine path blank"):
            remote_settings(engine="/opt/llama-server")
        with pytest.raises(ValueError, match="use device CUDA0"):
            remote_settings(device="Vulkan0")
        with pytest.raises(ValueError, match="only to a remote backend"):
            Settings.parse({"backend": "CUDA", "gpu_type": "T4"})

    def test_idle_timeout_is_blank_or_minutes(self, provider):
        assert remote_settings().idle_timeout_minutes == 30
        assert remote_settings(idle_timeout_minutes="").idle_timeout_minutes is None
        assert remote_settings(idle_timeout_minutes=0).idle_timeout_minutes == 0
        for value in (-1, 1441, 1.5, "30"):
            with pytest.raises(ValueError, match="idle_timeout_minutes"):
                remote_settings(idle_timeout_minutes=value)

    def test_saved_defaults_key_on_backend_and_gpu_type(self, provider):
        local = Settings(model="/models/m.gguf")
        assert default_key(local) == "/models/m.gguf|CUDA"
        key = default_key(
            Settings(model="/models/m.gguf", backend="fake", gpu_type="X")
        )
        assert key == "/models/m.gguf|fake|X"


class TestEngineFactory:
    def test_local_settings_get_a_local_engine(self):
        engine = create_engine(Settings())
        assert isinstance(engine, LocalEngine)
        assert isinstance(create_engine(), LocalEngine)
        assert engine_serves(engine, Settings(backend="Vulkan"))
        with pytest.raises(ValueError, match="cannot serve the modal backend"):
            engine.launch_args(Settings(backend="modal", gpu_type="T4"))

    def test_remote_settings_get_the_provider_engine(self, provider, engines):
        engine = make_engine(provider, engines, remote_settings(idle_timeout_minutes=0))
        assert isinstance(engine, RemoteEngine)
        assert engine.provider is provider and engine.gpu == "FAKE-24"
        assert engine.idle_timeout is None
        assert engine_serves(engine, remote_settings())
        assert not engine_serves(engine, Settings())
        other = make_engine(provider, engines, remote_settings(idle_timeout_minutes=2))
        assert other.idle_timeout == 120

    def test_unknown_provider_is_reported(self):
        with pytest.raises(ValueError, match="Unknown remote provider"):
            create_engine(Settings(backend="nowhere", gpu_type="T4"))

    def test_catalogue_models_download_from_their_repository(self, state):
        sources = catalogue_sources([ENTRY])
        catalogued = sources(str(config.MODELS_DIR / "example" / "model.gguf"))
        assert catalogued.name == "example/model.gguf"
        assert (catalogued.repo, catalogued.files) == (ENTRY["repo"], ("model.gguf",))
        other = sources(str(config.MODELS_DIR / "mine" / "own.gguf"))
        assert (other.name, other.repo) == ("mine/own.gguf", None)


class TestEvidence:
    def test_launch_check_needs_no_local_gpu(self, provider, engines, no_local_gpu):
        app = App.__new__(App)
        app.engines = {"local": LocalEngine()}
        app.engine_lock = threading.Lock()
        app.catalogue = Mock(list=Mock(return_value=[ENTRY]))
        app.store = Mock(get=Mock(return_value=None))
        s = remote_settings()
        check = app.action("/api/launch/check", {"settings": s.dict()})
        engines.append(app.engines["fake"])
        assert check["valid"], check["error"]
        assert app.engine_for(s) is app.engines["fake"]
        features = app.action("/api/capabilities", {"settings": s.dict()})
        assert "environment" not in features["engine"]
        for source in ("auto", "built-in"):
            resolved = app.action(
                "/api/default/resolve", {"settings": s.dict(), "source": source}
            )
            assert Settings.parse(resolved["settings"]).gpu_type == "FAKE-24"
        selected = app.action(
            "/api/launch/select",
            {"model": s.model, "backend": "fake", "gpu_type": "FAKE-24"},
        )
        assert Settings.parse(selected["settings"]).backend == "fake"
        # Validation starts no container: no probe and no download.
        assert provider.probes == [] and provider.downloads == []

    def test_unprobed_gpu_uses_the_table_and_the_pinned_release(
        self, provider, engines, no_local_gpu
    ):
        provider.probe = Mock(side_effect=AssertionError("probe started a container"))
        engine = make_engine(provider, engines)
        s = remote_settings(flash="on", cache="q8_0")
        assert engine.hardware(s)["gpus"][0]["name"] == "Fake GPU 24GB"
        record = engine.probe(s)
        assert record["estimated"] and record["sha256"] is None
        features = engine.capabilities(s)
        assert features["flash"]["status"] == "available"
        assert features["cache"]["status"] == "available"
        assert features["ngram-simple"]["status"] == "available"
        assert features["draft-mtp"]["status"] == "missing prerequisites"
        assert features["cuda_graph_opt"]["status"] == "unsupported"
        argv = engine.launch_args(s)
        assert argv[argv.index("--device") + 1] == "CUDA0"
        defaults = starting_defaults(s, **engine.defaults_inputs(s))
        assert Settings.parse(defaults["settings"]).gpu_type == "FAKE-24"
        variants, _ = suite(s, engine)
        assert len(variants) > 1

    def test_a_launch_probe_is_reused_until_the_release_changes(
        self, provider, engines, no_local_gpu, monkeypatch
    ):
        s = remote_settings()
        first = make_engine(provider, engines)
        first.start(s, threading.Event(), timeout=30)
        first.stop()
        assert provider.probes == ["FAKE-24"]
        second = make_engine(provider, engines)
        assert second.hardware(s)["gpus"][0]["name"] == "Fake GPU 24GB (probed)"
        assert second.probe(s)["sha256"] == ENGINE["sha256"]
        second.start(s, threading.Event(), timeout=30)
        second.stop()
        assert provider.probes == ["FAKE-24"]
        monkeypatch.setattr(remote, "LLAMA_CPP_REF", "b0-next")
        third = make_engine(provider, engines)
        assert third.probe(s)["estimated"]
        third.start(s, threading.Event(), timeout=30)
        third.stop()
        assert provider.probes == ["FAKE-24", "FAKE-24"]

    def test_remote_gpu_profiles_match_probed_gpu_and_catalogue_sha(
        self, provider, engines, no_local_gpu
    ):
        engine = make_engine(provider, engines)
        s = remote_settings()
        engine.start(s, threading.Event(), timeout=30)
        engine.stop()
        inputs = engine.defaults_inputs(s)
        profile = copy.deepcopy(PROFILES[0])
        profile.update(
            id="fake-profile",
            backend="fake",
            gpu={"name": "Fake GPU 24GB (probed)", "total_mib": 23028, "driver": None},
            settings={"context": 8192, "slots": 2, "flash": "on", "cache": "q8_0"},
            template={"sha256": hashlib.sha256(b"").hexdigest()},
            context={"validated_recommendation": False},
            limitations=[],
            summary="",
        )
        profile["engine"] = {"build": "fake", "sha256": ENGINE["sha256"]}
        # The launch downloaded the model, so its real metadata is known.
        profile["model"] = dict(
            profile["model"], file="model.gguf", mtp=False, architecture="fake"
        )
        entry = dict(ENTRY, recommendation={"profile": "fake-profile", "rank": 1})
        engine._catalogue = [entry]
        with patch("lllm2.recommendations.PROFILES", [profile]):
            with patch("lllm2.remote.PROFILES", [profile]):
                model = engine.identity(s.model)
            assert model["sha256"] == profile["model"]["sha256"]
            resolved = starting_defaults(s, **(inputs | {"model": model}))
            assert resolved["source"].startswith("Measured built-in baseline · fake")
            settings = Settings.parse(resolved["settings"])
            assert (settings.context, settings.slots, settings.gpu_type) == (
                8192,
                2,
                "FAKE-24",
            )
            profile["gpu"]["name"] = "Another GPU"
            fallback = starting_defaults(s, **(inputs | {"model": model}))
        assert "tested GPU" in " ".join(fallback["notes"])


@pytest.fixture
def bench_run(provider, engines, no_local_gpu):
    """Run a bench submission against a remote engine; return snapshot and result."""
    remote_engine = make_engine(provider, engines)
    store = Mock()
    bench = Bench(LocalEngine(), store)

    def run(data):
        bench.submit(
            {
                "settings": remote_settings().dict(),
                "sweep_prompts": False,
                "prompt_tokens": 128,
                "output_tokens": 16,
                "timeout": 60,
            }
            | data,
            remote_engine,
        )
        assert bench.engine is remote_engine
        assert eventually(lambda: not bench.snapshot()["active"], timeout=120)
        results = [c.args[2] for c in store.put.call_args_list if c.args[0] == "result"]
        assert not remote_engine.alive()
        return bench.snapshot(), results[-1]

    return run


def test_bench_results_record_the_remote_hardware(bench_run, provider):
    snapshot, result = bench_run({"mode": "baseline"})
    assert snapshot["status"] == "complete", result.get("error")
    assert result["status"] == "complete"
    assert result["hardware"]["source"] == "fake"
    assert result["hardware"]["gpu_type"] == "FAKE-24"
    assert result["hardware"]["gpus"][0]["name"] == "Fake GPU 24GB (probed)"
    assert result["model"]["store"] == "fake"
    assert result["model"]["store_name"] == "example/model.gguf"
    assert result["model"]["repo"] == ENTRY["repo"]
    assert result["engine"]["sha256"] == ENGINE["sha256"]
    assert "environment" not in result["engine"]
    assert result["settings"]["backend"] == "fake"
    assert result["argv"][0] == ENGINE["path"]
    sample = result["samples"][0]
    assert sample["input_tokens"] >= 128 and sample["output_tokens"] == 16
    assert sample["peak_engine_rss_mib"] is None
    assert sample["peak_total_gpu_used_mib"] is None
    assert provider.downloads == ["example/model.gguf"]


def test_warm_conversation_runs_on_a_remote_engine(bench_run):
    snapshot, result = bench_run({"mode": "warm-conversation"})
    assert snapshot["status"] == "complete", result.get("error")
    assert result["hardware"]["source"] == "fake"
    assert len(result["samples"]) == 12
    assert all(s["status"] == "complete" for s in result["samples"])
    assert any(s["reuse_observed"] for s in result["samples"])
    assert all(s["peak_total_gpu_used_mib"] is None for s in result["samples"])


@pytest.fixture
def other_session():
    """A live process that stands in for another running panel."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield process.pid
    process.kill()
    process.wait()


def spawn_call(provider):
    call_id = provider.spawn(
        "FAKE-24", ["llama-server", "--port", str(free_port())], "key", {}, {}
    )
    assert eventually(lambda: any(c.id == call_id for c in provider.calls()))
    return call_id


def record(pid, heartbeat, model="/models/example/model.gguf"):
    return {
        "api_key": "key",
        "gpu": "FAKE-24",
        "settings": {"model": model},
        "argv": [],
        "started": time.time(),
        "saved": time.time(),
        "owner": {
            "pid": pid,
            "host": socket.gethostname(),
            "pid_ns": remote.pid_namespace(),
            "heartbeat": heartbeat,
        },
    }


class TestOwnership:
    def test_a_running_sessions_call_is_not_an_orphan(
        self, provider, engines, state, other_session
    ):
        records = CallRecords(config.STATE_DIR / "remote-calls.json")
        active = spawn_call(provider)
        records.put("fake", active, record(other_session, time.time()))
        stale = spawn_call(provider)
        records.put("fake", stale, record(other_session, time.time() - 600))
        engine = make_engine(provider, engines)

        rows = {row["id"]: row for row in engine.remote_calls()}
        assert rows[active]["status"] == "active"
        assert rows[active]["owner"]["pid"] == other_session
        assert rows[stale]["status"] == "orphan"
        assert [o["id"] for o in engine.orphans()] == [stale]
        with pytest.raises(ValueError, match="Another lllm2 session"):
            engine.adopt(active)

    def test_dead_owner_leaves_an_orphan(self, provider, engines, state):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        call_id = spawn_call(provider)
        CallRecords(config.STATE_DIR / "remote-calls.json").put(
            "fake", call_id, record(dead.pid, time.time())
        )
        engine = make_engine(provider, engines)
        assert [o["id"] for o in engine.orphans()] == [call_id]

    def test_new_records_survive_a_lagging_provider_list(self, state):
        records = CallRecords(state / "calls.json")
        records.put("fake", "new", record(os.getpid(), time.time()))
        old = record(os.getpid(), time.time() - 600)
        old["saved"] = time.time() - 600
        records.put("fake", "old", old)
        records.keep("fake", [])
        assert records.get("fake", "new") is not None
        assert records.get("fake", "old") is None

    def test_owning_engine_refreshes_its_heartbeat(
        self, provider, engines, state, monkeypatch
    ):
        monkeypatch.setattr(remote, "OWNER_HEARTBEAT_SECONDS", 0)
        engine = make_engine(provider, engines)
        engine.start(remote_settings(), threading.Event(), timeout=30)
        records = CallRecords(config.STATE_DIR / "remote-calls.json")
        first = records.get("fake", engine.call_id)["owner"]
        assert first["pid"] == os.getpid()
        assert eventually(
            lambda: (
                records.get("fake", engine.call_id)["owner"]["heartbeat"]
                > first["heartbeat"]
            )
        )
        assert engine.remote_calls()[0]["status"] == "owned"


class SetupProvider(FakeProvider):
    def setup(self):
        return "1.2.3-fake"


@pytest.fixture
def modal_cli(state, other_session):
    """Serve `lllm2 modal` from a fake provider with an orphan and an active call."""
    made = SetupProvider(state / "remote")
    records = CallRecords(config.STATE_DIR / "remote-calls.json")
    orphan = spawn_call(made)
    active = spawn_call(made)
    records.put("fake", active, record(other_session, time.time()))
    (made.root / "volume" / "example").mkdir(parents=True)
    (made.root / "volume" / "example" / "model.gguf").write_bytes(bytes(2_000_000))
    with patch.dict(remote.PROVIDERS, {"modal": lambda: made}):
        yield made, orphan, active
    made.close()


def invoke(*args):
    return CliRunner().invoke(cli.app, ["modal", *args])


def test_modal_setup_prints_the_deployed_version(modal_cli):
    result = invoke("setup")
    assert result.exit_code == 0, result.output
    assert "Modal credentials work" in result.output
    assert "1.2.3-fake" in result.output


def test_modal_list_tells_owned_calls_from_orphans(modal_cli):
    _, orphan, active = modal_cli
    result = invoke("list")
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert any(orphan in line and "orphan" in line for line in lines)
    assert any(active in line and "in use by lllm2 pid" in line for line in lines)
    assert "~$" in result.output and "Check current pricing" in result.output
    rows = {r["id"]: r for r in json.loads(invoke("list", "--json").output)}
    assert (rows[orphan]["status"], rows[active]["status"]) == ("orphan", "active")
    assert rows[orphan]["usd_per_hour"] == 3.6


def test_modal_stop_leaves_calls_a_running_session_owns(modal_cli):
    provider, orphan, active = modal_cli
    result = invoke("stop", active)
    assert result.exit_code == 1
    assert "add --force" in result.output
    result = invoke("stop", "--all")
    assert result.exit_code == 0, result.output
    assert f"Stopped {orphan}" in result.output
    assert [c.id for c in provider.calls()] == [active]
    result = invoke("stop", active, "--force")
    assert result.exit_code == 0, result.output
    assert provider.calls() == []
    assert invoke("stop").exit_code == 2
    assert "No lllm2 serve calls" in invoke("list").output


def test_modal_models_and_remove(modal_cli, tmp_path):
    provider, orphan, active = modal_cli
    result = invoke("models")
    assert result.output.strip() == "example/model.gguf  0.0 GB"
    rows = json.loads(invoke("models", "--json").output)
    assert rows == [{"name": "example/model.gguf", "size_bytes": 2_000_000}]
    blocked = invoke("remove", "example/model.gguf")
    assert isinstance(blocked.exception, ValueError)
    assert active in str(blocked.exception)
    invoke("stop", active, "--force")
    result = invoke("remove", "example/model.gguf")
    assert result.exit_code == 0, result.output
    assert provider.models() == []
    assert "No models are stored" in invoke("models").output


def test_modal_errors_name_the_extra_and_the_sign_in_step():
    def missing():
        raise RuntimeError(modal_app.INSTALL_MESSAGE)

    broken = Mock(calls=Mock(side_effect=RuntimeError(CREDENTIALS_MESSAGE)))
    for factory, expected in (
        (missing, "pip install 'lllm2[modal]'"),
        (lambda: broken, "modal token new"),
    ):
        stderr = io.StringIO()
        with (
            patch.dict(remote.PROVIDERS, {"modal": factory}),
            contextlib.redirect_stderr(stderr),
        ):
            assert cli.main(["modal", "list"]) == 2
        assert expected in stderr.getvalue()


def test_remote_launch_needs_a_gpu_type_and_no_local_options():
    for args, message in (
        (["launch", "--backend", "modal"], "--gpu: T4, L4"),
        (["launch", "--backend", "modal", "--gpu", "T4", "--device", "CUDA0"], ""),
        (["launch", "--gpu", "T4"], "only to a remote backend"),
    ):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            assert cli.main(args) == 2
        assert message in stderr.getvalue()
    assert "--engine and --device apply only" in _launch_error(
        "", "", "modal", "CUDA0", 1, "T4"
    )


def _launch_error(*args):
    with pytest.raises(ValueError) as caught:
        cli._launch(*args)
    return str(caught.value)


def test_models_path_is_under_the_managed_directory(state):
    catalogue = [ENTRY | {"recommendation": {"rank": 1}}]
    expected = str(Path(config.MODELS_DIR).resolve() / "example" / "model.gguf")
    assert cli._remote_model_path("", catalogue) == expected
    assert cli._remote_model_path("example", catalogue) == expected
