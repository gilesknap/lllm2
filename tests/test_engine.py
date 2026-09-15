"""Engine lifecycle through its public interface, using a fake llama-server."""

import contextlib
import io
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from lllm2 import config
from lllm2.engine import (
    Cancelled,
    Engine,
    GPUUnavailable,
    LocalEngine,
    ResourceConflict,
)
from lllm2.settings import Settings

FAKE_SERVER = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"status": "ok"})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.reply({"prompt": "Hello"})

    def log_message(self, *args):
        pass

server = HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
print("fake server listening", flush=True)
server.serve_forever()
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def eventually(check, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.05)
    return check()


@contextlib.contextmanager
def local_launch(tmp_path, script=FAKE_SERVER):
    """Patch the GPU checks and launch command so LocalEngine runs a fake server."""
    port = free_port()
    path = tmp_path / "fake_server.py"
    path.write_text(script)
    with (
        patch.object(config, "ENGINE_PORT", port),
        patch(
            "lllm2.engine.launch_args",
            return_value=[sys.executable, str(path), str(port)],
        ),
        patch("lllm2.engine.hardware", return_value={"gpus": [{}], "error": None}),
        patch("lllm2.engine.command", return_value=(0, "")),
        patch("lllm2.engine.launch_environment", return_value=dict(os.environ)),
    ):
        engine = LocalEngine()
        try:
            yield engine
        finally:
            engine.stop()


def test_local_engine_serves_logs_and_samples_its_process(tmp_path):
    stderr = io.StringIO()
    with local_launch(tmp_path) as engine, contextlib.redirect_stderr(stderr):
        assert not engine.alive()
        engine.start(Settings(model="m.gguf"), threading.Event(), timeout=20)

        assert engine.alive()
        state = engine.state()
        assert state["running"] and state["ready"]
        assert state["pid"] is not None
        assert state["error"] is None
        assert state["settings"]["model"] == "m.gguf"
        assert engine.base.endswith(f":{config.ENGINE_PORT}")
        assert eventually(lambda: "fake server listening" in engine.logs())
        assert engine.logs()[0].startswith("Launching: ")
        assert "[llama-server] fake server listening\n" in stderr.getvalue()

        sample = engine.memory_sampler()()
        assert sample["pid"] == state["pid"]
        # Some sandboxes mount a /proc that hides child processes.
        if Path(f"/proc/{state['pid']}/status").exists():
            assert sample["rss_mib"] is not None
        else:
            assert sample["rss_mib"] is None and "error" in sample

        engine.stop()
        assert not engine.alive()
        assert not engine.state()["running"]
        assert engine.state()["settings"] is None


def test_closed_terminal_does_not_stop_engine_output_drain(tmp_path):
    with (
        local_launch(tmp_path) as engine,
        patch("builtins.print", side_effect=BrokenPipeError),
    ):
        engine.start(Settings(model="m.gguf"), threading.Event(), timeout=20)
        assert eventually(lambda: "fake server listening" in engine.logs())
        assert engine.alive()


def test_engine_that_exits_during_load_is_reported(tmp_path):
    script = "print('error loading model', flush=True)\nraise SystemExit(1)\n"
    with (
        local_launch(tmp_path, script) as engine,
        contextlib.redirect_stderr(io.StringIO()),
    ):
        with pytest.raises(RuntimeError, match="exited during load"):
            engine.start(Settings(model="m.gguf"), threading.Event(), timeout=20)
        assert not engine.alive()
        assert eventually(lambda: "error loading model" in engine.logs())
        sample = engine.memory_sampler()()
        assert sample["rss_mib"] is None and "error" in sample


def test_cancelled_start_launches_nothing(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with local_launch(tmp_path) as engine:
        with pytest.raises(Cancelled):
            engine.start(Settings(model="m.gguf"), cancel, timeout=20)
        assert not engine.alive()
        assert engine.logs() == []


def test_competing_gpu_process_blocks_launch(tmp_path):
    with (
        local_launch(tmp_path) as engine,
        patch("lllm2.engine.command", return_value=(0, "999999, trainer\n")),
    ):
        with pytest.raises(ResourceConflict, match="Other GPU compute processes"):
            engine.start(Settings(model="m.gguf"), threading.Event(), timeout=20)
        assert not engine.alive()
        assert engine.logs() == []


def test_unreadable_gpu_ownership_blocks_launch(tmp_path):
    with (
        local_launch(tmp_path) as engine,
        patch("lllm2.engine.command", return_value=(9, "driver failure")),
    ):
        with pytest.raises(GPUUnavailable, match="Cannot check GPU ownership"):
            engine.start(Settings(model="m.gguf"), threading.Event(), timeout=20)
        assert not engine.alive()


def test_occupied_engine_port_blocks_launch(tmp_path):
    with local_launch(tmp_path) as engine, socket.socket() as sock:
        sock.bind(("127.0.0.1", config.ENGINE_PORT))
        sock.listen()
        with pytest.raises(ResourceConflict, match="is occupied"):
            engine.start(Settings(model="m.gguf"), threading.Event(), timeout=20)
        assert not engine.alive()


def test_engine_that_never_answers_times_out_and_is_stopped(tmp_path):
    script = "import time\nprint('loading', flush=True)\ntime.sleep(60)\n"
    with (
        local_launch(tmp_path, script) as engine,
        contextlib.redirect_stderr(io.StringIO()),
    ):
        with pytest.raises(TimeoutError, match="startup exceeded timeout"):
            engine.start(Settings(model="m.gguf"), threading.Event(), timeout=1)
        assert not engine.alive()
        assert engine.state()["settings"] is None


class ServedEngine(Engine):
    """An engine whose server runs elsewhere, as a remote backend's would."""

    def __init__(self, port):
        with patch.object(config, "ENGINE_PORT", port):
            super().__init__()
        self.running = False

    def start(self, s, cancel, timeout=180):
        self.settings, self.running = s, True
        self.log("Launching: remote")
        self._await_ready(s, cancel, timeout, lambda: self.running)

    def stop(self):
        with self.guard:
            self.ready = self.running = False
            self.settings = None

    def alive(self):
        return self.running

    def hardware(self, s=None):
        return {"gpus": [], "error": None, "source": "served"}

    def probe(self, s):
        raise ValueError("No engine binary to probe.")

    def metadata(self, path):
        raise ValueError("No checkpoint to read.")

    def identity(self, path):
        return {"path": path, "size": None, "mtime_ns": None}

    def launch_args(self, s):
        return []

    def status(self):
        return {
            "running": self.running,
            "ready": self.running and self.ready,
            "pid": None,
            "error": None,
        }


@pytest.fixture
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply({"status": "ok"})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.reply({"prompt": "Hello"})

        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def test_engine_without_a_local_process_shares_readiness_and_skips_memory(
    http_server,
):
    engine = ServedEngine(http_server)
    with contextlib.redirect_stderr(io.StringIO()):
        engine.start(Settings(model="m.gguf"), threading.Event(), timeout=10)
    state = engine.state()
    assert state["ready"] and state["pid"] is None
    assert state["logs"] == ["Launching: remote"]
    assert engine.request("/health") == {"status": "ok"}
    sample = engine.memory_sampler()()
    assert sample["pid"] is None
    assert all(value is None for value in sample.values())
    engine.stop()
    assert not engine.alive() and not engine.state()["ready"]


def test_logs_are_bounded_and_limited():
    engine = ServedEngine(free_port())
    with contextlib.redirect_stderr(io.StringIO()):
        for i in range(Engine.LOG_LINES + 5):
            engine.log(str(i))
    logs = engine.logs()
    assert len(logs) == Engine.LOG_LINES
    assert logs[0] == "5" and logs[-1] == str(Engine.LOG_LINES + 4)
    assert engine.logs(3) == [str(Engine.LOG_LINES + n) for n in (2, 3, 4)]
    assert len(engine.state()["logs"]) == Engine.STATE_LOG_LINES


def test_base_engine_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Engine()  # type: ignore[abstract]
