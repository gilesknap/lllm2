"""Remote engine lifecycle through its public interface, with a fake provider."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fake_remote import ENGINE, FakeProvider, free_port
from lllm2 import config
from lllm2.engine import Cancelled, ResourceConflict
from lllm2.remote import (
    CALL_STALE_SECONDS,
    DEFAULT_IDLE_TIMEOUT,
    OWNER_HEARTBEAT_SECONDS,
    OWNER_STALE_SECONDS,
    RECORD_GRACE_SECONDS,
    CallRecords,
    RemoteEngine,
    model_source,
    pid_namespace,
    stop_owned_calls,
)
from lllm2.settings import Settings

TESTS = Path(__file__).parent
CHILD = """
import os, signal, sys, threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fake_remote import FakeProvider
from lllm2 import config
from lllm2.remote import RemoteEngine
from lllm2.settings import Settings
root, records, port, models, mode = sys.argv[2:7]
config.MODELS_DIR = Path(models)
engine = RemoteEngine(
    FakeProvider(root),
    "FAKE-24",
    port=int(port),
    poll_interval=0.05,
    records=records,
    probes=records + ".probes",
)
engine.start(Settings(model=models + "/example/model.gguf"), threading.Event(), 30)
print("ready", engine.call_id, flush=True)
if mode == "crash":
    os.kill(os.getpid(), signal.SIGKILL)
"""


def eventually(check, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


class FakeClock:
    """A clock that stands still until a test moves it."""

    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    """A fake monotonic clock for the idle timer and heartbeat intervals."""
    return FakeClock(1000.0)


@pytest.fixture
def wall():
    """A fake wall clock for call records and owner heartbeats."""
    return FakeClock(time.time())


@pytest.fixture
def model(tmp_path, monkeypatch):
    root = tmp_path / "models"
    (root / "example").mkdir(parents=True)
    path = root / "example" / "model.gguf"
    path.write_bytes(b"GGUF")
    monkeypatch.setattr(config, "MODELS_DIR", root)
    return str(path)


@pytest.fixture
def providers(tmp_path):
    made = []

    def make(**kwargs):
        provider = FakeProvider(tmp_path / "remote", **kwargs)
        made.append(provider)
        return provider

    yield make
    for provider in made:
        provider.close()


@pytest.fixture
def engines(tmp_path, providers):
    made = []

    def make(provider=None, **kwargs):
        options = {
            "port": free_port(),
            "poll_interval": 0.05,
            "records": tmp_path / "calls.json",
            "probes": tmp_path / "probes.json",
        } | kwargs
        engine = RemoteEngine(provider or providers(), "FAKE-24", **options)
        made.append(engine)
        return engine

    yield make
    for engine in made:
        engine.shutdown()


def request(engine, path, body=None, token="placeholder"):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        engine.base + path,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def test_start_serves_the_remote_server_on_the_engine_port(model, providers, engines):
    provider = providers()
    engine = engines(provider)
    assert not engine.alive()
    engine.start(Settings(model=model), threading.Event(), timeout=30)

    assert engine.alive()
    state = engine.state()
    assert state["running"] and state["ready"] and state["pid"] is None
    assert state["error"] is None
    assert state["phase"] == "ready"
    assert state["provider"] == "fake" and state["gpu"] == "FAKE-24"
    assert state["call_id"] == engine.call_id
    assert state["settings"]["model"] == model
    assert state["usd_per_hour"] == 3.6
    assert state["elapsed_seconds"] >= 0 and state["estimated_cost_usd"] >= 0
    assert state["idle_timeout_seconds"] == DEFAULT_IDLE_TIMEOUT
    assert 0 < state["idle_remaining_seconds"] <= DEFAULT_IDLE_TIMEOUT
    assert engine.base == f"http://127.0.0.1:{engine.port}"

    # Clients keep their placeholder token; only the proxy holds the key.
    assert request(engine, "/health") == {"status": "ok"}
    assert request(engine, "/tokenize", {"content": "ab"}) == {"tokens": [98, 99]}
    # The server needs the key except on the paths llama-server leaves open.
    direct = f"http://127.0.0.1:{provider.server_port}"
    with urllib.request.urlopen(direct + "/health") as response:
        assert response.status == 200
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(direct + "/v1/models")
    assert refused.value.code == 401
    refused.value.close()
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(
            urllib.request.Request(
                direct + "/completion",
                data=b'{"prompt": "x", "n_predict": 1}',
                headers={"Content-Type": "application/json"},
            )
        )
    assert refused.value.code == 401
    refused.value.close()

    argv = state["argv"]
    assert argv[0] == ENGINE["path"]
    assert argv[argv.index("--model") + 1] == "/volume/example/model.gguf"
    assert argv[argv.index("--port") + 1] == str(provider.server_port)
    assert provider.spawned[0]["gpu"] == "FAKE-24"
    assert not any("LLAMA_API_KEY" in arg or "--api-key" in arg for arg in argv)
    assert engine.logs()[0].startswith("Launching: ")
    assert eventually(lambda: "fake llama-server listening" in engine.logs())

    hardware = engine.hardware()
    assert hardware["source"] == "fake"
    assert hardware["gpus"][0]["name"] == "Fake GPU 24GB (probed)"
    assert hardware["gpus"][0]["total_mib"] == 23028
    assert all(v is None for v in engine.memory_sampler()().values())

    started = time.monotonic()
    engine.stop()
    assert time.monotonic() - started < 5
    assert not engine.alive()
    state = engine.state()
    assert not state["running"] and state["settings"] is None
    assert state["phase"] == "stopped" and state["elapsed_seconds"] is None
    assert provider.calls() == []
    with pytest.raises(urllib.error.URLError):
        request(engine, "/health")


def test_cold_start_reports_phases_and_reuses_probe_and_model(
    model, providers, engines
):
    provider = providers(server_env={"FAKE_LOAD_SECONDS": "0.8"}, download_seconds=0.6)
    engine = engines(provider)
    phases, downloads, done = [], [], threading.Event()

    def watch():
        while not done.is_set():
            status = engine.status()
            if not phases or phases[-1] != status["phase"]:
                phases.append(status["phase"])
            if status["download"]:
                downloads.append(status["download"])
            time.sleep(0.005)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        engine.start(Settings(model=model), threading.Event(), timeout=30)
    finally:
        done.set()
        watcher.join()
    phases.append(engine.status()["phase"])
    expected = iter(["downloading model", "starting container", "loading model"])
    step = next(expected)
    for phase in phases:
        if phase == step:
            step = next(expected, "ready")
    assert step == "ready" and phases[-1] == "ready", phases
    assert downloads, phases
    assert {d["file"] for d in downloads} == {"example/model.gguf"}
    assert all(0 < d["done_bytes"] <= d["total_bytes"] == 3000 for d in downloads)
    assert engine.status()["download"] is None

    engine.start(Settings(model=model, context=8192), threading.Event(), timeout=30)
    assert engine.state()["ready"]
    assert provider.probes == ["FAKE-24"]
    assert provider.downloads == ["example/model.gguf"]
    assert [m.name for m in provider.models()] == ["example/model.gguf"]
    assert len(provider.calls()) == 1


def test_chat_template_file_is_sent_with_the_call(model, providers, engines, tmp_path):
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    provider = providers()
    engine = engines(provider)
    engine.start(
        Settings(model=model, chat_template=str(template)), threading.Event(), 30
    )
    argv = engine.state()["argv"]
    assert argv[argv.index("--chat-template-file") + 1] == "/files/chat.jinja"
    assert provider.spawned[0]["files"] == {"chat.jinja": "{{ messages }}"}


def test_local_weights_are_never_uploaded(model, providers, engines, tmp_path):
    weights = b"GGUF" + os.urandom(1 << 20)
    Path(model).write_bytes(weights)
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    provider = providers()
    engines(provider).start(
        Settings(model=model, chat_template=str(template)), threading.Event(), 30
    )
    (spawned,) = provider.spawned
    # Only small text templates travel with the call; the model stays in the store.
    assert spawned["files"] == {"chat.jinja": "{{ messages }}"}
    argv = spawned["argv"]
    assert argv[argv.index("--model") + 1] == "/volume/example/model.gguf"
    assert provider.downloads == ["example/model.gguf"]
    stored = tmp_path / "remote" / "volume" / "example" / "model.gguf"
    assert stored.read_bytes() != weights


def test_model_outside_the_managed_directory_is_rejected(model, providers, engines):
    provider = providers()
    engine = engines(provider)
    with pytest.raises(ValueError, match="managed model directory"):
        engine.start(Settings(model="/elsewhere/m.gguf"), threading.Event(), 30)
    assert not engine.alive() and provider.spawned == []
    assert model_source(model).name == "example/model.gguf"


def test_remote_exit_during_load_is_reported(model, providers, engines):
    provider = providers(server_env={"FAKE_EXIT": "1"})
    engine = engines(provider)
    with pytest.raises(RuntimeError, match="exited during load"):
        engine.start(Settings(model=model), threading.Event(), timeout=30)
    assert not engine.alive()
    assert "error loading model" in engine.logs()
    assert provider.calls() == []


def test_startup_timeout_cancels_the_call(model, providers, engines):
    provider = providers(server_env={"FAKE_LOAD_SECONDS": "60"})
    engine = engines(provider)
    with pytest.raises(TimeoutError, match="startup exceeded timeout"):
        engine.start(Settings(model=model), threading.Event(), timeout=1)
    assert not engine.alive()
    assert provider.calls() == []


def test_cancel_during_download_spawns_nothing(model, providers, engines):
    provider = providers(download_seconds=5)
    engine = engines(provider)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(Cancelled):
        engine.start(Settings(model=model), cancel, timeout=30)
    assert provider.spawned == [] and not engine.alive()
    # A cancelled start releases the engine port.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", engine.port))


def test_stop_during_load_cancels_the_call(model, providers, engines):
    provider = providers(server_env={"FAKE_LOAD_SECONDS": "60"})
    engine = engines(provider)
    threading.Thread(
        target=lambda: (
            eventually(lambda: engine.status()["phase"] == "loading model")
            and engine.stop()
        )
    ).start()
    with pytest.raises((Cancelled, RuntimeError)):
        engine.start(Settings(model=model), threading.Event(), timeout=30)
    assert not engine.alive()
    assert provider.calls() == []


def test_occupied_engine_port_blocks_launch_before_spending(model, providers, engines):
    provider = providers()
    engine = engines(provider)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", engine.port))
        sock.listen()
        with pytest.raises(ResourceConflict, match="is occupied"):
            engine.start(Settings(model=model), threading.Event(), timeout=30)
    assert provider.probes == [] and provider.spawned == []


def test_call_that_ends_remotely_is_reported(model, providers, engines):
    provider = providers()
    engine = engines(provider)
    engine.start(Settings(model=model), threading.Event(), timeout=30)
    provider.cancel(engine.call_id)
    # Both facts latch, so wait for them together: a loaded machine must not
    # read the engine between it stopping and the reason for it appearing.
    assert eventually(
        lambda: not engine.alive() and engine.state()["phase"] == "exited"
    )
    assert "Remote engine stopped" in engine.state()["error"]


def monitor_rounds():
    """Give the engine's status monitor several polls at the fixture interval."""
    time.sleep(0.3)


def counted_down_to(engine, clock, remaining):
    """Move the fake clock until the idle countdown reads ``remaining``.

    The proxy books a finished request from its own thread, so a request the
    client has already read can still be in flight, or be booked just after
    the clock moves. Either leaves the countdown at the whole timeout however
    far the clock has gone, so advance by what is left rather than once by a
    fixed amount. Pass this to ``eventually``.

    Args:
        engine: The engine whose countdown to move.
        clock: The fake clock the engine reads.
        remaining: The countdown value to stop at, in seconds.

    Returns:
        True when the countdown reads ``remaining``.
    """
    left = engine.status()["idle_remaining_seconds"]
    if left is None or left < remaining:
        return False
    clock.advance(left - remaining)
    return engine.status()["idle_remaining_seconds"] == remaining


def test_idle_engine_stops_its_call(model, providers, engines, clock):
    provider = providers()
    engine = engines(provider, idle_timeout=60, clock=clock)
    engine.start(Settings(model=model), threading.Event(), timeout=30)
    assert engine.status()["idle_remaining_seconds"] == 60
    assert eventually(lambda: counted_down_to(engine, clock, 1))
    monitor_rounds()
    assert engine.alive() and engine.status()["idle_remaining_seconds"] == 1
    clock.advance(1)
    assert eventually(
        lambda: not engine.alive() and engine.state()["phase"] == "idle stopped"
    )
    assert engine.state()["call_id"] is None
    assert any("No requests for 60 seconds" in line for line in engine.logs())
    assert provider.calls() == []


def test_disabling_the_timeout_as_it_expires_leaves_the_monitor_running(
    model, providers, engines, clock
):
    """A disable that lands during an idle stop must not kill the monitor.

    Holding the guard puts the monitor's idle stop behind the disable, the
    order that used to format a None timeout and end the monitor thread with
    an unhandled TypeError, stopping the heartbeats and the reaping with it.
    """
    provider = providers()
    engine = engines(provider, idle_timeout=60, clock=clock)
    engine.start(Settings(model=model), threading.Event(), timeout=30)
    assert eventually(lambda: counted_down_to(engine, clock, 1))
    with engine.guard:
        clock.advance(1)
        # Let the monitor read the countdown and queue behind the guard.
        monitor_rounds()
        engine.set_idle_timeout(None)
    monitor_rounds()
    assert engine.alive() and engine.status()["idle_remaining_seconds"] is None
    # The monitor survived, so a fresh timeout still stops the call.
    engine.set_idle_timeout(1)
    assert eventually(
        lambda: not engine.alive() and engine.state()["phase"] == "idle stopped"
    )
    assert provider.calls() == []


def test_requests_reset_the_idle_timer(model, providers, engines, clock):
    provider = providers()
    engine = engines(provider, idle_timeout=60, clock=clock)
    engine.start(Settings(model=model), threading.Event(), timeout=30)
    # Five requests 50 seconds apart span well over the 60 second timeout.
    for _ in range(5):
        clock.advance(50)
        request(engine, "/health")
        assert engine.status()["idle_remaining_seconds"] == 60
        monitor_rounds()
        assert engine.alive()
    # Settle the last request before the final advance, or a late booking
    # restarts the countdown and the call never reaches its idle stop.
    assert eventually(lambda: counted_down_to(engine, clock, 1))
    clock.advance(1)
    assert eventually(lambda: not engine.alive())


def test_long_stream_holds_off_the_idle_timer(model, providers, engines, clock):
    provider = providers(server_env={"FAKE_TOKEN_SECONDS": "0.05"})
    engine = engines(provider, idle_timeout=60, clock=clock)
    engine.start(Settings(model=model), threading.Event(), timeout=30)
    events = []

    def token(event):
        # Each token takes 10 fake seconds, so the stream lasts 250 seconds.
        events.append(event)
        clock.advance(10)

    final = engine.stream_completion(
        {"prompt": "hello", "n_predict": 25, "stream": True},
        threading.Event(),
        30,
        token,
    )
    assert final["stop"] is True and len(events) == 26
    assert engine.alive()

    def countdown_started():
        # The proxy ends the request just after the client reads the final
        # event. Until then the countdown stays full however far the clock
        # moves; afterwards one step shortens it.
        clock.advance(1)
        return engine.status()["idle_remaining_seconds"] < 60

    assert eventually(countdown_started)
    assert engine.alive()
    clock.advance(60)
    assert eventually(lambda: not engine.alive())


def test_call_records_are_private_and_keys_stay_out_of_status_and_logs(
    model, providers, engines, tmp_path
):
    engine = engines(providers())
    engine.start(Settings(model=model), threading.Event(), timeout=30)
    records = tmp_path / "calls.json"
    assert records.stat().st_mode & 0o777 == 0o600
    key = json.loads(records.read_text())["fake"][engine.call_id]["api_key"]
    assert key not in json.dumps(engine.state())
    assert not any(key in line for line in engine.logs())


def test_disabled_idle_timer_keeps_the_call(model, providers, engines, clock):
    engine = engines(idle_timeout=None, clock=clock)
    engine.start(Settings(model=model), threading.Event(), timeout=30)
    clock.advance(10**6)
    monitor_rounds()
    assert engine.alive()
    status = engine.status()
    assert status["idle_timeout_seconds"] is None
    assert status["idle_remaining_seconds"] is None


def test_shutdown_hook_stops_owned_calls(model, providers, engines):
    provider = providers()
    engine = engines(provider)
    engine.start(Settings(model=model), threading.Event(), timeout=30)
    stop_owned_calls()
    assert not engine.alive()
    assert provider.calls() == []


def run_child(tmp_path, model, mode):
    port = free_port()
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(TESTS),
            str(tmp_path / "remote"),
            str(tmp_path / "calls.json"),
            str(port),
            str(config.MODELS_DIR),
            mode,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ,
    )
    lines = [x for x in child.stdout.splitlines() if x.startswith("ready ")]
    assert lines, child.stderr
    return lines[0].split()[1]


def test_panel_exit_stops_its_call(model, providers, tmp_path):
    run_child(tmp_path, model, "exit")
    assert providers().calls() == []


def test_orphan_from_a_crashed_session_can_be_adopted(
    model, providers, engines, tmp_path
):
    call_id = run_child(tmp_path, model, "crash")
    provider = providers()
    engine = engines(provider)

    orphans = engine.orphans()
    assert [o["id"] for o in orphans] == [call_id]
    assert orphans[0]["adoptable"] and orphans[0]["gpu"] == "FAKE-24"
    assert orphans[0]["model"] == model

    engine.adopt(call_id, timeout=30)
    state = engine.state()
    assert state["ready"] and state["call_id"] == call_id
    assert state["settings"]["model"] == model
    assert request(engine, "/health") == {"status": "ok"}
    assert engine.orphans() == []

    engine.stop()
    assert provider.calls() == []


def test_orphan_from_a_crashed_session_can_be_cancelled(
    model, providers, engines, tmp_path
):
    call_id = run_child(tmp_path, model, "crash")
    provider = providers()
    engine = engines(provider)
    assert [o["id"] for o in engine.orphans()] == [call_id]
    engine.cancel_orphan(call_id)
    assert provider.calls() == [] and engine.orphans() == []
    with pytest.raises(ValueError, match="no longer running"):
        engine.adopt(call_id)


def test_owner_heartbeat_refreshes_the_call_record(
    model, providers, engines, clock, wall, tmp_path
):
    engine = engines(providers(), clock=clock, wall_clock=wall)
    engine.start(Settings(model=model), threading.Event(), timeout=30)

    def heartbeat():
        records = json.loads((tmp_path / "calls.json").read_text())
        return records["fake"][engine.call_id]["owner"]["heartbeat"]

    assert heartbeat() == wall.now
    started = wall.now
    wall.advance(OWNER_HEARTBEAT_SECONDS - 1)
    clock.advance(OWNER_HEARTBEAT_SECONDS - 1)
    monitor_rounds()
    assert heartbeat() == started
    wall.advance(1)
    clock.advance(1)
    assert eventually(lambda: heartbeat() == wall.now)


def test_stale_owner_heartbeat_turns_a_call_into_an_orphan(
    model, providers, engines, wall, tmp_path
):
    provider = providers(clock=wall)
    port = free_port()
    argv = ["llama-server", "--port", str(port)]
    call_id = provider.spawn("FAKE-24", argv, "key", {}, {})
    assert eventually(lambda: provider.calls() != [])
    records = CallRecords(tmp_path / "calls.json")
    record = {
        "api_key": "key",
        "gpu": "FAKE-24",
        "settings": Settings(model=model).dict(),
        "argv": argv,
        "started": wall.now,
        "saved": wall.now,
    }
    # The owner lives in another PID namespace, so only its heartbeat counts.
    owner = {"pid": 1, "host": "elsewhere", "pid_ns": "pid:[1]", "heartbeat": wall.now}
    records.put("fake", call_id, record | {"owner": owner})
    records.put("fake", "ended", record)
    engine = engines(provider, wall_clock=wall)

    (row,) = engine.remote_calls()
    assert (row["id"], row["status"]) == (call_id, "active")
    assert engine.orphans() == []
    with pytest.raises(ValueError, match="Another lllm2 session"):
        engine.adopt(call_id)
    # A recent record survives a provider list that does not show its call.
    assert records.get("fake", "ended") is not None

    wall.advance(max(OWNER_STALE_SECONDS, RECORD_GRACE_SECONDS, CALL_STALE_SECONDS) + 1)
    assert [o["id"] for o in engine.orphans()] == [call_id]
    assert records.get("fake", "ended") is None

    engine.adopt(call_id, timeout=30)
    assert engine.state()["ready"] and engine.call_id == call_id
    assert records.get("fake", call_id)["owner"]["heartbeat"] == wall.now
    assert engine.orphans() == []


def test_a_call_another_live_session_serves_is_never_an_orphan(
    model, providers, engines, wall, tmp_path
):
    """The provider's heartbeat decides, not this workstation's records.

    The owning session here is a panel in another container: it writes its
    heartbeats to the provider, but its call records live in a state directory
    this session cannot see, and its local record here is an old one.
    """
    provider = providers(clock=wall)
    argv = ["llama-server", "--port", str(free_port())]
    call_id = provider.spawn("FAKE-24", argv, "key", {}, {})
    assert eventually(lambda: provider.calls() != [])
    records = CallRecords(tmp_path / "calls.json")
    records.put(
        "fake",
        call_id,
        {
            "api_key": "key",
            "gpu": "FAKE-24",
            "settings": Settings(model=model).dict(),
            "argv": argv,
            "started": wall.now,
            "saved": wall.now,
            "owner": {
                "pid": 1,
                "host": "elsewhere",
                "pid_ns": "pid:[1]",
                "heartbeat": wall.now - 10 * OWNER_STALE_SECONDS,
            },
        },
    )
    engine = engines(provider, wall_clock=wall)

    (row,) = engine.remote_calls()
    assert (row["id"], row["status"], row["heartbeat_age"]) == (call_id, "active", 0)
    # The key is here, but taking the call would pull it from a live session.
    assert not row["adoptable"]
    assert engine.orphans() == []
    with pytest.raises(ValueError, match="Another lllm2 session"):
        engine.adopt(call_id)

    # The other session keeps serving, so its heartbeat stays fresh.
    wall.advance(CALL_STALE_SECONDS - 1)
    provider.beat(call_id)
    wall.advance(CALL_STALE_SECONDS - 1)
    assert engine.orphans() == []

    # It goes away: only silence past the grace makes the call an orphan.
    wall.advance(2)
    (orphan,) = engine.orphans()
    assert (orphan["id"], orphan["adoptable"]) == (call_id, True)
    engine.adopt(call_id, timeout=30)
    assert engine.state()["ready"] and engine.call_id == call_id


def test_a_dead_owner_here_frees_a_call_unless_another_session_took_it_over(
    model, providers, engines, wall, tmp_path
):
    """A record naming a dead process is only evidence while it fits the beats.

    A session that crashed on this workstation stopped writing both its record
    and its heartbeat at once, so its call is an orphan straight away. Once the
    provider's heartbeat moves on without that record, a later session has
    taken the call over and the dead process here says nothing about it.
    """
    provider = providers(clock=wall)
    argv = ["llama-server", "--port", str(free_port())]
    call_id = provider.spawn("FAKE-24", argv, "key", {}, {})
    assert eventually(lambda: provider.calls() != [])
    CallRecords(tmp_path / "calls.json").put(
        "fake",
        call_id,
        {
            "api_key": "key",
            "gpu": "FAKE-24",
            "settings": Settings(model=model).dict(),
            "argv": argv,
            "started": wall.now,
            "saved": wall.now,
            "owner": {
                # A pid above pid_max, so this namespace cannot hold it.
                "pid": 2**22 + 7,
                "host": socket.gethostname(),
                "pid_ns": pid_namespace(),
                "heartbeat": wall.now,
            },
        },
    )
    engine = engines(provider, wall_clock=wall)

    # Both heartbeats stopped together, so the crash frees the call at once.
    wall.advance(OWNER_HEARTBEAT_SECONDS + 1)
    (row,) = engine.remote_calls()
    assert (row["status"], row["adoptable"]) == ("orphan", True)

    # A session elsewhere adopted it and keeps the provider's heartbeat fresh,
    # while the record here still names the process that died.
    wall.advance(CALL_STALE_SECONDS + 1)
    provider.beat(call_id)
    (row,) = engine.remote_calls()
    assert (row["status"], row["adoptable"]) == ("active", False)
    assert engine.orphans() == []
    with pytest.raises(ValueError, match="Another lllm2 session"):
        engine.adopt(call_id)


def test_a_call_the_provider_reports_no_heartbeat_for_ages_into_an_orphan(
    providers, engines, wall
):
    """No heartbeat to read neither pins a call nor makes it stoppable on sight."""
    provider = providers(clock=wall)
    argv = ["llama-server", "--port", str(free_port())]
    call_id = provider.spawn("FAKE-24", argv, "key", {}, {})
    assert eventually(lambda: provider.calls() != [])
    provider.untracked(call_id)
    engine = engines(provider, wall_clock=wall)

    # A call another session has just started is not an orphan to stop.
    (row,) = engine.remote_calls()
    assert (row["status"], row["heartbeat_age"]) == ("active", None)

    wall.advance(CALL_STALE_SECONDS + 1)
    (row,) = engine.remote_calls()
    assert (row["status"], row["heartbeat_age"]) == ("orphan", None)


def test_orphan_without_a_saved_key_can_only_be_cancelled(model, providers, engines):
    provider = providers()
    engine = engines(provider)
    port = str(free_port())
    call_id = provider.spawn("FAKE-24", ["llama-server", "--port", port], "k", {}, {})
    assert eventually(lambda: provider.calls() != [])
    provider.silence(call_id)
    orphans = engine.orphans()
    assert [(o["id"], o["adoptable"]) for o in orphans] == [(call_id, False)]
    with pytest.raises(ValueError, match="cannot be adopted"):
        engine.adopt(call_id)
    engine.cancel_orphan(call_id)
    assert engine.orphans() == []
