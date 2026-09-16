"""Panel API contracts for remote backends, served by the fake provider.

Every test uses temporary state and a fake provider registry, so no panel
database, Modal client or GPU is touched. Timing-dependent checks move a fake
clock forward instead of sleeping through the idle timeout.
"""

import contextlib
import json
import os
import socket
import sys
import threading
import time
from unittest.mock import patch

import pytest

from fake_remote import FakeProvider, free_port
from lllm2 import config, modal_app, remote
from lllm2.app import App
from lllm2.bench import Bench
from lllm2.catalogue import Catalogue
from lllm2.engine import LocalEngine
from lllm2.modal_provider import ModalProvider
from lllm2.remote import CallRecords, StoreDownloads, catalogue_source
from lllm2.settings import Settings
from lllm2.store import Store
from test_modal_provider import META, FakeModal

ENTRY = {
    "id": "example",
    "name": "example",
    "display_name": "Example 8B",
    "repo": "fake/example-GGUF",
    "file": "model.gguf",
    "size_gb": 5,
    "max_ctx": 65536,
    "mtp": False,
}


def eventually(check, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


class FakeClock:
    """A monotonic clock that runs with real time plus a movable offset."""

    def __init__(self):
        self.offset = 0.0
        self.real = time.monotonic

    def __call__(self):
        return self.real() + self.offset

    def advance(self, seconds):
        self.offset += seconds


@pytest.fixture
def clock(app):
    fake = FakeClock()
    # The panel's remote engine and its proxy read this clock at call time.
    app.remote_engine("fake").clock = fake
    return fake


@pytest.fixture
def providers(tmp_path, monkeypatch):
    """Register only the fake provider, with state and models in tmp_path."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    made = []

    def factory(**options):
        provider = FakeProvider(tmp_path / "remote", **options)
        made.append(provider)
        return provider

    current = {"options": {}}
    with patch.dict(
        remote.PROVIDERS, {"fake": lambda: factory(**current["options"])}, clear=True
    ):
        yield current
    for provider in made:
        provider.close()


@pytest.fixture
def app(providers, tmp_path):
    """A panel App with a real temporary store and one catalogue entry."""
    panel = App.__new__(App)
    panel.store = Store()
    panel.catalogue = Catalogue(panel.store)
    panel.catalogue.lock = threading.RLock()
    panel.finder = None
    panel.engines = {"local": LocalEngine()}
    panel.engine_lock = threading.Lock()
    panel.bench = Bench(panel.engines["local"], panel.store)
    panel.token = "test"
    panel.start_requests = {}
    panel.store_downloads = StoreDownloads()
    with patch.object(Catalogue, "list", lambda self: [dict(ENTRY)]):
        with patch.object(Catalogue, "get", lambda self, id: dict(ENTRY)):
            yield panel
    panel.bench.cancel.set()
    panel.store_downloads.cancel_all()
    for engine in panel.engines.values():
        with contextlib.suppress(Exception):
            engine.stop()
    panel.store.db.close()


def model_path():
    return str(config.MODELS_DIR / "example" / "model.gguf")


def fake_settings(**values):
    return Settings.parse(
        {"model": model_path(), "backend": "fake", "gpu_type": "FAKE-24"} | values
    ).dict()


def use_free_port(app):
    """Serve remote engines on a free port instead of the engine port."""
    engine = app.remote_engine("fake")
    engine.port = free_port()
    engine.base = f"http://127.0.0.1:{engine.port}"
    engine.poll_interval = 0.05
    return engine


def start(app, **values):
    settings = fake_settings(**values)
    app.action("/api/start", {"settings": settings, "request_id": f"r{time.time()}"})
    return settings


def test_discover_lists_remote_backends_and_their_gpu_types(app):
    with (
        patch("lllm2.app.installed_models", return_value=[]),
        patch("lllm2.app.engines", return_value=[]),
    ):
        data = app.action("/api/discover", {"backend": "fake", "gpu_type": "FAKE-24"})
    (backend,) = data["backends"]
    assert (backend["name"], backend["label"]) == ("fake", "Fake")
    assert backend["gpus"] == [
        {
            "name": "FAKE-24",
            "label": "Fake GPU 24GB",
            "vram_gb": 24,
            "vram_gib": 22.3,
            "usd_per_hour": 3.6,
        }
    ]
    assert "Check current pricing" in backend["caveat"]
    assert backend["idle_timeout_minutes"] == Settings().idle_timeout_minutes
    (entry,) = data["catalog"]
    assert entry["path"] == str(config.MODELS_DIR / "example" / "model.gguf")
    # Suitability compares the model with the selected remote GPU, not local cards.
    assert "fit" in entry


def test_status_hardware_follows_the_selected_backend(app):
    with patch("lllm2.app.hardware", return_value={"gpus": [], "source": "local"}):
        assert app.selected_hardware({})["source"] == "local"
        assert app.selected_hardware({"backend": "CUDA"})["source"] == "local"
        # An unknown GPU type falls back to local hardware instead of failing.
        assert (
            app.selected_hardware({"backend": "fake", "gpu_type": "X"})["source"]
            == "local"
        )
    remote_host = app.selected_hardware({"backend": "fake", "gpu_type": "FAKE-24"})
    assert remote_host["source"] == "fake"
    assert remote_host["gpu_type"] == "FAKE-24"
    assert remote_host["gpus"][0]["name"] == "Fake GPU 24GB"


def test_find_models_uses_the_selected_gpu(app):
    class Finder:
        def search(self, query, host, refresh=False):
            return {"host": host}

    app.finder = Finder()
    found = app.action(
        "/api/models/find", {"query": "", "backend": "fake", "gpu_type": "FAKE-24"}
    )
    assert found["host"]["source"] == "fake"


def test_select_takes_the_gpu_type_and_sizes_defaults_for_it(app, no_probe):
    selected = app.action(
        "/api/launch/select",
        {"model": model_path(), "backend": "fake", "gpu_type": "FAKE-24"},
    )
    settings = Settings.parse(selected["settings"])
    assert (settings.backend, settings.gpu_type) == ("fake", "FAKE-24")
    assert settings.engine == "" and settings.device == "CUDA0"
    # The recommendation comes from the shared defaults logic for that VRAM.
    from lllm2.defaults import starting_defaults

    engine = app.remote_engine("fake")
    s = Settings.parse(fake_settings())
    assert selected == starting_defaults(s, **engine.defaults_inputs(s))
    with pytest.raises(ValueError, match="Choose a fake GPU type"):
        app.action("/api/launch/select", {"model": model_path(), "backend": "fake"})


@pytest.fixture
def no_probe(providers):
    """Fail if validation starts a container."""

    def refuse(self, gpu):
        raise AssertionError("validation started a probe container")

    with patch.object(FakeProvider, "probe", refuse):
        yield


def test_switching_backend_needs_no_restart(app, clock):
    use_free_port(app)
    start(app)
    assert eventually(lambda: app.bench.snapshot()["status"] == "serving")
    state = app.engine.state()
    assert state["ready"] and state["provider"] == "fake"
    assert state["phase"] == "ready" and state["call_id"]
    assert any("fake llama-server listening" in line for line in state["logs"])
    # Switching to a local backend stops the remote call first.
    local = app.engines["local"]
    with patch.object(LocalEngine, "stop") as stop:
        app.bench.use(app.engine_for(Settings(model=model_path())))
    assert app.engine is local and not stop.called
    assert app.engines["fake"].state()["running"] is False
    assert app.action("/api/remote", {"backend": "fake"})["calls"] == []


def test_remote_status_reports_cost_and_idle_countdown(app, clock):
    use_free_port(app)
    start(app, idle_timeout_minutes=10)
    assert eventually(lambda: app.bench.snapshot()["status"] == "serving")
    clock.advance(360)
    state = app.engine.state()
    assert state["elapsed_seconds"] >= 360
    assert state["usd_per_hour"] == 3.6
    assert state["estimated_cost_usd"] == pytest.approx(
        state["elapsed_seconds"] * 3.6 / 3600
    )
    assert state["idle_timeout_seconds"] == 600
    assert 230 <= state["idle_remaining_seconds"] <= 240
    # The panel changes the idle timeout of the running engine without a restart.
    status = app.action("/api/remote/idle", {"minutes": 60})
    assert status["idle_timeout_seconds"] == 3600
    assert app.engine.state()["settings"]["idle_timeout_minutes"] == 60
    status = app.action("/api/remote/idle", {"minutes": ""})
    assert status["idle_timeout_seconds"] is None
    assert status["idle_remaining_seconds"] is None
    with pytest.raises(ValueError, match="0..1440"):
        app.action("/api/remote/idle", {"minutes": 2000})
    app.action("/api/remote/idle", {"minutes": 1})
    clock.advance(120)
    # The next status check stops the idle call and says why.
    assert eventually(lambda: app.engine.state()["phase"] == "idle stopped")
    state = app.engine.state()
    assert not state["running"] and state["call_id"] is None
    assert any("No requests for 60 seconds" in line for line in state["logs"])


def test_stop_during_download_cancels_the_start(app, providers):
    providers["options"] = {"download_seconds": 30}
    use_free_port(app)
    start(app)
    engine = app.engine
    assert eventually(lambda: engine.state()["phase"] == "downloading model")
    assert eventually(lambda: engine.state()["download"] is not None)
    began = time.monotonic()
    app.action("/api/stop", {})
    assert eventually(lambda: app.bench.snapshot()["status"] == "cancelled", 5)
    assert time.monotonic() - began < 5
    assert engine.provider.spawned == [] and engine.provider.downloads == []
    assert not engine.state()["running"]


def test_stop_ends_the_remote_call_immediately(app):
    use_free_port(app)
    start(app)
    assert eventually(lambda: app.bench.snapshot()["status"] == "serving")
    provider = app.engine.provider
    call_id = app.engine.state()["call_id"]
    app.action("/api/stop", {})
    assert all(c.id != call_id for c in provider.calls())
    assert app.engine.state()["phase"] == "stopped"


def spawn_orphan(tmp_path, owner):
    """Start a call as a crashed earlier session would have left it."""
    provider = FakeProvider(tmp_path / "remote")
    port = free_port()
    argv = ["llama-server", "--port", str(port)]
    call_id = provider.spawn("FAKE-24", argv, "key", {}, {})
    # Another provider instance sees the call only once its server listens.
    observer = FakeProvider(tmp_path / "remote")
    assert eventually(lambda: any(c.id == call_id for c in observer.calls()))
    # The crashed session stopped writing its record and its heartbeat to the
    # provider at the same moment.
    provider.beat(call_id, at=owner["heartbeat"])
    CallRecords(config.STATE_DIR / "remote-calls.json").put(
        "fake",
        call_id,
        {
            "api_key": "key",
            "gpu": "FAKE-24",
            "settings": fake_settings(),
            "argv": argv,
            "started": time.time() - 90,
            "saved": time.time() - 600,
            "owner": owner,
        },
    )
    return provider, call_id


def spawn_elsewhere(tmp_path):
    """Start a call as a session with its own state directory would have."""
    provider = FakeProvider(tmp_path / "remote")
    call_id = provider.spawn(
        "FAKE-24", ["llama-server", "--port", str(free_port())], "key", {}, {}
    )
    observer = FakeProvider(tmp_path / "remote")
    assert eventually(lambda: any(c.id == call_id for c in observer.calls()))
    return provider, call_id


def dead_owner():
    return {
        "pid": 2**22 + 7,
        "host": socket.gethostname(),
        "pid_ns": remote.pid_namespace(),
        "heartbeat": time.time() - 600,
    }


def test_orphan_banner_lists_and_stops_an_orphan(app, tmp_path):
    spawner, call_id = spawn_orphan(tmp_path, dead_owner())
    try:
        (row,) = app.action("/api/remote/orphans", {})["orphans"]
        assert (row["id"], row["provider"], row["status"]) == (
            call_id,
            "fake",
            "orphan",
        )
        assert row["adoptable"] and row["model"] == model_path()
        assert row["elapsed_seconds"] >= 0 and row["usd_per_hour"] == 3.6
        assert row["estimated_cost_usd"] is not None
        assert "Check current pricing" in row["caveat"]
        app.action("/api/remote/stop-call", {"backend": "fake", "call_id": call_id})
        assert app.action("/api/remote/orphans", {})["orphans"] == []
        with pytest.raises(ValueError, match="no longer running"):
            app.action("/api/remote/stop-call", {"backend": "fake", "call_id": call_id})
    finally:
        spawner.close()


def test_orphan_banner_ignores_a_call_a_live_session_serves(app, tmp_path):
    """Another session's live call bills that session, not this one, to stop."""
    spawner, call_id = spawn_elsewhere(tmp_path)
    try:
        assert app.action("/api/remote/orphans", {"backend": "fake"})["orphans"] == []
        (row,) = app.action("/api/remote", {"backend": "fake"})["calls"]
        assert row["status"] == "active" and not row["adoptable"]
        with pytest.raises(ValueError, match="another machine"):
            app.action("/api/remote/stop-call", {"backend": "fake", "call_id": call_id})
        # The banner offers the call only once its owner stops heartbeating.
        spawner.silence(call_id)
        (row,) = app.action("/api/remote/orphans", {"backend": "fake"})["orphans"]
        assert row["id"] == call_id and row["status"] == "orphan"
        app.action("/api/remote/stop-call", {"backend": "fake", "call_id": call_id})
        assert spawner.calls() == []
    finally:
        spawner.close()


def test_orphan_can_be_adopted_from_the_panel(app, tmp_path):
    use_free_port(app)
    spawner, call_id = spawn_orphan(tmp_path, dead_owner())
    try:
        app.action("/api/remote/adopt", {"backend": "fake", "call_id": call_id})
        assert eventually(lambda: app.bench.snapshot()["status"] == "serving")
        state = app.engine.state()
        assert state["ready"] and state["call_id"] == call_id
        assert app.bench.snapshot()["settings"]["model"] == model_path()
        assert app.action("/api/remote/orphans", {})["orphans"] == []
        (row,) = app.action("/api/remote", {"backend": "fake"})["calls"]
        assert row["status"] == "owned"
        with pytest.raises(ValueError, match="Use Stop model"):
            app.action("/api/remote/stop-call", {"backend": "fake", "call_id": call_id})
    finally:
        spawner.close()


def test_a_running_model_is_not_replaced_without_confirmation(app, tmp_path):
    use_free_port(app)
    start(app)
    assert eventually(lambda: app.bench.snapshot()["status"] == "serving")
    spawner, call_id = spawn_orphan(tmp_path, dead_owner())
    try:
        with pytest.raises(ValueError, match="adopting replaces it"):
            app.action("/api/remote/adopt", {"backend": "fake", "call_id": call_id})
    finally:
        spawner.close()


def test_two_sessions_cannot_both_adopt_one_orphan(providers, tmp_path):
    spawner, call_id = spawn_orphan(tmp_path, dead_owner())
    records = CallRecords(config.STATE_DIR / "remote-calls.json")
    results = []

    def claim():
        try:
            records.claim("fake", call_id)
            results.append("claimed")
        except ValueError as e:
            results.append(str(e))

    try:
        # A claim by another process holds a fresh heartbeat under the file
        # lock; model that process by a different pid in this namespace.
        with patch("os.getpid", return_value=os.getppid()):
            claim()
        claim()
        assert results[0] == "claimed"
        assert "Another lllm2 session" in results[1]
    finally:
        spawner.close()


def test_heartbeat_decides_ownership_across_pid_namespaces():
    now = time.time()
    other_namespace = {
        "pid": 2**22 + 7,
        "host": socket.gethostname(),
        "pid_ns": "pid:[1]",
        "heartbeat": now,
    }
    # The pid does not exist here, but the owner lives in another namespace.
    assert remote.owner_alive(other_namespace, now)
    assert not remote.owner_alive(other_namespace, now + remote.OWNER_STALE_SECONDS + 1)
    same_namespace = dict(other_namespace, pid_ns=remote.pid_namespace())
    assert not remote.owner_alive(same_namespace, now)
    # Only a pid from this host and namespace may be read as gone: a pid from a
    # container means nothing in this process table, whoever holds it here.
    assert remote.owner_process_gone(same_namespace)
    assert not remote.owner_process_gone(other_namespace)
    assert not remote.owner_process_gone(dict(same_namespace, pid_ns=None))
    assert not remote.owner_process_gone(dict(same_namespace, host="elsewhere"))
    assert not remote.owner_process_gone(dict(same_namespace, pid=os.getpid()))


def test_volume_models_download_and_remove(app, providers):
    view = app.action("/api/remote", {"backend": "fake"})
    assert (view["error"], view["models"], view["stored_ids"]) == (None, [], [])
    row = app.action("/api/remote/download", {"backend": "fake", "id": "example"})
    assert (row["id"], row["store"], row["state"]) == (
        "fake:example",
        "fake",
        "downloading",
    )
    assert eventually(lambda: app.store_downloads.rows()[0]["state"] == "complete"), (
        app.store_downloads.rows()
    )
    view = app.action("/api/remote", {"backend": "fake"})
    assert view["stored_ids"] == ["example"]
    (stored,) = view["models"]
    assert stored == {
        "name": "example/model.gguf",
        "size_bytes": 3000,
        "catalogue_id": "example",
        "display_name": "Example 8B",
        "in_use": [],
    }
    # A second download finds the model and downloads nothing.
    app.action("/api/remote/download", {"backend": "fake", "id": "example"})
    assert eventually(lambda: app.store_downloads.rows()[0]["state"] == "complete")
    assert app.remote_engine("fake").provider.downloads == ["example/model.gguf"]
    view = app.action(
        "/api/remote/models/remove", {"backend": "fake", "name": "example/model.gguf"}
    )
    assert view["models"] == [] and view["stored_ids"] == []


def test_volume_download_reports_progress_and_cancels(app, providers):
    providers["options"] = {"download_seconds": 30}
    app.action("/api/remote/download", {"backend": "fake", "id": "example"})
    assert eventually(lambda: app.store_downloads.rows()[0]["percent"] > 0)
    row = app.store_downloads.rows()[0]
    assert row["total_gb"] == 0.0 and row["file"] == "example/model.gguf"
    with pytest.raises(ValueError, match="Cancel the download"):
        app.action(
            "/api/remote/models/remove",
            {"backend": "fake", "name": "example/model.gguf"},
        )
    with pytest.raises(ValueError, match="still downloading"):
        start(app)
    assert app.action("/api/remote/download/cancel", {"id": "fake:example"})
    assert eventually(lambda: app.store_downloads.rows()[0]["state"] == "cancelled")


def test_a_served_model_cannot_be_removed(app):
    use_free_port(app)
    start(app)
    assert eventually(lambda: app.bench.snapshot()["status"] == "serving")
    (stored,) = app.action("/api/remote", {"backend": "fake"})["models"]
    assert stored["in_use"] == [app.engine.state()["call_id"]]
    with pytest.raises(ValueError, match="Stop it first"):
        app.action(
            "/api/remote/models/remove",
            {"backend": "fake", "name": "example/model.gguf"},
        )


@pytest.fixture
def modal_client(providers):
    """Register the Modal provider over a fake ``modal`` module."""
    fake = FakeModal()
    with patch.dict(
        remote.PROVIDERS,
        {"modal": lambda: ModalProvider(fake, poll_interval=0, deploy=lambda: None)},
    ):
        yield fake


def test_modal_client_lists_downloads_and_removes_volume_models(app, modal_client):
    entry = dict(ENTRY, mmproj="mmproj.gguf")
    fake = modal_client

    def download(name, repo, files, revision):
        assert (name, repo, files) == (
            "example/model.gguf",
            "fake/example-GGUF",
            ["model.gguf", "mmproj.gguf"],
        )

        def poll(call):
            fake.state.put(
                modal_app.download_key(call.object_id),
                {"file": files[-1], "done_bytes": 1000, "total_bytes": 4000},
            )
            if not call.pending:
                fake.store.files.update({f"example/{f}": 4000 for f in files})

        return {"pending": 3, "result": dict(META), "on_poll": poll}

    fake.behaviour["download"] = download
    request = {"backend": "modal", "id": "example"}
    with (
        patch.object(Catalogue, "list", lambda self: [dict(entry)]),
        patch.object(Catalogue, "get", lambda self, id: dict(entry)),
    ):
        view = app.action("/api/remote", {"backend": "modal"})
        assert (view["error"], view["models"], view["stored_ids"]) == (None, [], [])
        row = app.action("/api/remote/download", request)
        assert (row["id"], row["store"]) == ("modal:example", "modal")
        assert eventually(lambda: app.store_downloads.rows()[0]["state"] == "complete")
        # Progress from the download container reached the downloads area.
        (row,) = app.store_downloads.rows()
        assert row["file"] == "mmproj.gguf" and row["detail"] == "Stored in modal"
        view = app.action("/api/remote", {"backend": "modal"})
        assert view["stored_ids"] == ["example"]
        assert [(m["name"], m["size_bytes"]) for m in view["models"]] == [
            ("example/mmproj.gguf", 4000),
            ("example/model.gguf", 4000),
        ]
        # A second download of a complete model starts no download call.
        app.action("/api/remote/download", request)
        assert eventually(lambda: app.store_downloads.rows()[0]["state"] == "complete")
        assert len(fake.calls) == 1
        view = app.action(
            "/api/remote/models/remove",
            {"backend": "modal", "name": "example/model.gguf"},
        )
    assert view["models"] == [] and view["stored_ids"] == []
    assert fake.store.files == {}


def test_missing_client_or_credentials_give_a_clear_message(app):
    from lllm2.modal_app import INSTALL_MESSAGE
    from lllm2.modal_provider import CREDENTIALS_MESSAGE

    def missing():
        raise RuntimeError(INSTALL_MESSAGE)

    with patch.dict(remote.PROVIDERS, {"fake": missing}):
        view = app.action("/api/remote", {"backend": "fake"})
        assert "pip install 'lllm2[modal]'" in view["error"]
        orphans = app.action("/api/remote/orphans", {"backend": "fake"})
        assert orphans["orphans"] == []
        assert "pip install" in orphans["unavailable"][0]["error"]
        # Validation reports the missing client instead of failing the request.
        check = app.action("/api/launch/check", {"settings": fake_settings()})
        assert not check["valid"] and "pip install" in check["error"]
    with patch.object(
        FakeProvider, "models", side_effect=RuntimeError(CREDENTIALS_MESSAGE)
    ):
        view = app.action("/api/remote", {"backend": "fake"})
    assert "modal token new" in view["error"]


def test_saved_remote_result_records_gpu_type_and_source(app, no_local_gpu):
    use_free_port(app)
    app.action(
        "/api/benchmark",
        {
            "settings": fake_settings(),
            "mode": "baseline",
            "sweep_prompts": False,
            "search_context": False,
            "prompt_tokens": 128,
            "output_tokens": 16,
            "timeout": 60,
        },
    )
    assert eventually(lambda: not app.bench.snapshot()["active"], 120)
    (summary,) = app.store.list("summary")
    assert summary["status"] == "complete", summary.get("error")
    assert summary["hardware"]["source"] == "fake"
    assert summary["hardware"]["gpu_type"] == "FAKE-24"
    assert summary["settings"]["backend"] == "fake"
    json.dumps(summary)


@pytest.fixture
def no_local_gpu():
    def refuse(*args, **kwargs):
        raise AssertionError("A remote backend must not probe local hardware.")

    with contextlib.ExitStack() as stack:
        for name in (
            "lllm2.engine.hardware",
            "lllm2.defaults.hardware",
            "lllm2.recommendations.hardware",
        ):
            stack.enter_context(patch(name, side_effect=refuse))
        yield


def test_idle_timeout_environment_variable():
    read = config.idle_timeout_minutes
    assert read({}) == 30
    for value, minutes in (("45", 45), ("0", 0), ("off", 0), ("", 0), (" 5 ", 5)):
        assert read({"LLLM2_IDLE_TIMEOUT_MINUTES": value}) == minutes
    for value in ("-1", "1441", "1.5", "soon"):
        with pytest.raises(ValueError, match="LLLM2_IDLE_TIMEOUT_MINUTES"):
            read({"LLLM2_IDLE_TIMEOUT_MINUTES": value})


def test_orphan_check_contacts_only_selected_or_recorded_providers(app):
    contacted = []

    def provider(name):
        def create():
            contacted.append(name)
            raise RuntimeError(f"{name} contacted")

        return create

    empty = {"orphans": [], "unavailable": []}
    with patch.dict(
        remote.PROVIDERS, {"fake": provider("fake"), "modal": provider("modal")}
    ):
        # A local session with no call records makes no provider request.
        for data in ({}, {"backend": "CUDA"}, {"backend": ""}):
            assert app.action("/api/remote/orphans", data) == empty
        assert contacted == [] and set(app.engines) == {"local"}
        # Selecting a remote backend checks that provider only.
        orphans = app.action("/api/remote/orphans", {"backend": "modal"})
        assert [u["provider"] for u in orphans["unavailable"]] == ["modal"]
        assert contacted == ["modal"]
        # A local call record can be an orphan, so its provider is checked.
        contacted.clear()
        CallRecords(config.STATE_DIR / "remote-calls.json").put(
            "fake", "fc-1", {"gpu": "FAKE-24"}
        )
        orphans = app.action("/api/remote/orphans", {"backend": "CUDA"})
        assert [u["provider"] for u in orphans["unavailable"]] == ["fake"]
        assert contacted == ["fake"]


def test_local_panel_works_without_the_modal_extra(app):
    local = {"gpus": [{"name": "RTX", "total_mib": 24000}], "source": "local"}
    # The real Modal factory, with the modal package hidden.
    with (
        patch.dict(remote.PROVIDERS, {"modal": remote._modal_provider}),
        patch.dict(sys.modules, {"modal": None}),
    ):
        local_panel_without_modal(app, local)


def local_panel_without_modal(app, local):
    with (
        patch("lllm2.app.hardware", return_value=local),
        patch("lllm2.app.installed_models", return_value=[]),
        patch("lllm2.app.engines", return_value=[]),
    ):
        # A CUDA selection reads local hardware and creates no remote engine.
        data = app.action("/api/discover", {"backend": "CUDA"})
        assert app.selected_hardware({"backend": "CUDA"}) == local
        assert set(app.engines) == {"local"}
        assert "modal" in {b["name"] for b in data["backends"]}
        # Selecting Modal shows the table GPU instead of failing the status poll.
        modal_host = app.selected_hardware({"backend": "modal", "gpu_type": "T4"})
        assert (modal_host["source"], modal_host["gpu_type"]) == ("modal", "T4")
    orphans = app.action("/api/remote/orphans", {"backend": "modal"})
    (unavailable,) = [u for u in orphans["unavailable"] if u["provider"] == "modal"]
    assert "lllm2[modal]" in unavailable["error"]
    assert "lllm2[modal]" in app.action("/api/remote", {"backend": "modal"})["error"]
    # The panel answers a Modal start with a client error naming the extra.
    s = Settings.parse({"model": model_path(), "backend": "modal", "gpu_type": "T4"})
    with pytest.raises(remote.ProviderError, match=r"lllm2\[modal\]"):
        app.action("/api/start", {"settings": s.dict()})


def states(app):
    """Map each download row's catalogue id to its state."""
    return {row["catalogue_id"]: row["state"] for row in app.store_downloads.rows()}


def test_a_failed_download_stops_being_reported_once_a_retry_succeeds(app, providers):
    """A retry that stores the model must clear the earlier failure."""
    providers["options"] = {"fail_downloads": 1}
    app.action("/api/remote/download", {"backend": "fake", "id": "example"})
    assert eventually(lambda: app.store_downloads.rows()[0]["state"] == "error")
    assert "ended early" in app.store_downloads.rows()[0]["detail"]
    app.action("/api/remote/download", {"backend": "fake", "id": "example"})
    assert eventually(lambda: app.store_downloads.rows()[0]["state"] == "complete")
    rows = app.store_downloads.rows()
    assert [row["state"] for row in rows] == ["complete"]
    assert not any("ended early" in (row["detail"] or "") for row in rows)
    assert app.action("/api/remote", {"backend": "fake"})["stored_ids"] == ["example"]


def test_a_failure_clears_when_the_model_arrives_by_another_route(app, providers):
    """A launch that stores the model settles the Downloads card's failure."""
    providers["options"] = {"fail_downloads": 1}
    other = dict(ENTRY, id="other", name="Other", file="other.gguf")
    provider = app.remote_engine("fake").provider
    app.action("/api/remote/download", {"backend": "fake", "id": "example"})
    app.store_downloads.start(provider, other)
    assert eventually(
        lambda: states(app) == {"example": "error", "other": "complete"}
    ), app.store_downloads.rows()
    # The failed model arrives through a launch instead of the Downloads card.
    provider.ensure_model(
        catalogue_source(app.catalogue.get("example")),
        lambda _update: None,
        threading.Event(),
    )
    view = app.action("/api/remote", {"backend": "fake"})
    assert sorted(view["stored_ids"]) == ["example"]
    assert [row["catalogue_id"] for row in app.store_downloads.rows()] == ["other"]


def test_a_failure_for_another_model_stays_visible(app, providers):
    """Settling one model leaves another model's failure alone."""
    providers["options"] = {"fail_downloads": 2}
    provider = app.remote_engine("fake").provider
    other = dict(ENTRY, id="other", name="Other", file="other.gguf")
    app.action("/api/remote/download", {"backend": "fake", "id": "example"})
    app.store_downloads.start(provider, other)
    assert eventually(lambda: states(app) == {"example": "error", "other": "error"})
    app.action("/api/remote/download", {"backend": "fake", "id": "example"})
    assert eventually(
        lambda: states(app) == {"example": "complete", "other": "error"}
    ), app.store_downloads.rows()
    # Listing the store settles only the model it holds.
    app.action("/api/remote", {"backend": "fake"})
    assert states(app) == {"example": "complete", "other": "error"}
