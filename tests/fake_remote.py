"""A fake remote provider that runs a fake llama-server as a local process.

The provider keeps its calls on disk, so a call outlives the process that
started it, as a real remote container outlives a crashed panel. The fake
server requires the per-launch API key on every request and implements the
endpoints that readiness, bench and warm use, including a streamed
``/completion`` with a single-slot prompt cache.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from lllm2.engine import Cancelled
from lllm2.gpu_tables import GpuType, register_gpu_table
from lllm2.proxy import Upstream
from lllm2.remote import (
    DownloadProgress,
    GpuProbe,
    RemoteCall,
    RemoteProvider,
    ServeStatus,
    StoredModel,
)

FAKE_GPU = GpuType("FAKE-24", "Fake GPU 24GB", 24, 3.6)
register_gpu_table("fake", (FAKE_GPU,))

FLAGS = [
    "--model",
    "--host",
    "--port",
    "--ctx-size",
    "--parallel",
    "--device",
    "--gpu-layers",
    "--jinja",
    "--fit",
    "--fit-target",
    "--perf",
    "--cache-ram",
    "--ctx-checkpoints",
    "--flash-attn",
    "--cache-type-k",
    "--cache-type-v",
    "--chat-template-file",
    "--spec-type",
    "--spec-draft-n-max",
]
ENGINE = {
    "path": "/opt/fake/llama-server",
    "version": "fake",
    "flags": FLAGS,
    "help": "--spec-type none, draft-mtp, ngram-simple",
    "devices": ["CUDA0"],
    "device_output": "CUDA0: Fake GPU 24GB",
    "error": None,
    "sha256": "0" * 64,
    "cuda_graph": {"supported": True, "reason": "Probed remotely.", "library": None},
    "cache_kernel": {"reason": "Probed remotely.", "library": None},
    "environment": {},
}
META = {
    "architecture": "fake",
    "name": "fake",
    "context": 65536,
    "mtp": False,
    "template": "",
    "error": None,
}

FAKE_LLAMA_SERVER = r"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = os.environ["LLAMA_API_KEY"]
LOAD_SECONDS = float(os.environ.get("FAKE_LOAD_SECONDS", "0"))
TOKEN_SECONDS = float(os.environ.get("FAKE_TOKEN_SECONDS", "0"))
if os.environ.get("FAKE_EXIT"):
    print("error loading model", flush=True)
    sys.exit(1)
port = int(sys.argv[sys.argv.index("--port") + 1])
started = time.monotonic()
cache = []
lock = threading.Lock()


def tokens(text):
    return [ord(c) % 5000 + 1 for c in text]


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorised(self):
        if self.headers.get("Authorization") == "Bearer " + KEY:
            return True
        self.reply(401, {"error": {"code": 401, "message": "Invalid API Key"}})
        return False

    def do_GET(self):
        if not self.authorised():
            return
        if time.monotonic() - started < LOAD_SECONDS:
            self.reply(503, {"error": {"code": 503, "message": "Loading model"}})
            return
        self.reply(200, {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self.authorised():
            return
        if self.path == "/apply-template":
            text = "\n".join(m["content"] for m in body["messages"])
            suffix = "<assistant>" if body.get("add_generation_prompt") else ""
            self.reply(200, {"prompt": "<user>" + text + suffix})
        elif self.path == "/tokenize":
            self.reply(200, {"tokens": tokens(body["content"])})
        elif self.path == "/completion":
            self.complete(body)
        else:
            self.reply(404, {"error": {"code": 404, "message": "Not found"}})

    def complete(self, body):
        prompt = body.get("prompt", [])
        if isinstance(prompt, str):
            prompt = tokens(prompt)
        n = int(body.get("n_predict", 16))
        with lock:
            reused = 0
            if body.get("cache_prompt", True):
                for a, b in zip(cache, prompt):
                    if a != b:
                        break
                    reused += 1
                reused = min(reused, max(len(prompt) - 1, 0))
            cache[:] = prompt
        final = {
            "content": "",
            "stop": True,
            "tokens_predicted": n,
            "tokens_evaluated": len(prompt),
            "tokens_cached": len(prompt) + n,
            "truncated": False,
            "timings": {
                "prompt_n": len(prompt) - reused,
                "cache_n": reused,
                "predicted_n": n,
                "prompt_per_second": 1000.0,
                "predicted_per_second": 100.0,
            },
        }
        if not body.get("stream"):
            self.reply(200, final | {"content": "x" * n})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(n):
            time.sleep(TOKEN_SECONDS)
            event = {"content": "x", "tokens": [100 + i], "stop": False}
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
server.daemon_threads = True
print("fake llama-server listening", flush=True)
server.serve_forever()
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeProvider(RemoteProvider):
    """A provider whose serve calls are fake llama-server processes.

    Attributes:
        probes: The GPU types probed, in order.
        downloads: The store names downloaded, in order.
        spawned: One dict per spawn with ``gpu``, ``argv``, ``env`` and ``files``.
    """

    name = "fake"
    server_host = "127.0.0.1"

    def __init__(self, root, server_env=None, download_seconds=0.0):
        """Create a provider that keeps its calls and store under a directory.

        Args:
            root: The directory for call records, logs and the model store.
                Providers sharing a directory see the same calls.
            server_env: Extra environment variables for the fake server, such
                as ``FAKE_LOAD_SECONDS`` or ``FAKE_EXIT``.
            download_seconds: How long a model download takes.
        """
        self.root = Path(root)
        for name in ("calls", "volume", "files"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.script = self.root / "fake_llama_server.py"
        self.script.write_text(FAKE_LLAMA_SERVER)
        self.server_port = free_port()
        self.server_env = dict(server_env or {})
        self.download_seconds = download_seconds
        self.probes = []
        self.downloads = []
        self.spawned = []
        self._processes = {}
        self._cursors = {}
        self._listening = set()
        self._lock = threading.Lock()

    def probe(self, gpu):
        self.probes.append(gpu)
        return GpuProbe(name="Fake GPU 24GB (probed)", total_mib=23028, engine=ENGINE)

    def ensure_model(self, source, progress, cancel):
        target = self.root / "volume" / source.name
        if not target.exists():
            for step in (1, 2, 3):
                if cancel.wait(self.download_seconds / 3):
                    raise Cancelled()
                progress(DownloadProgress(source.name, step * 1000, 3000))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes(3000))
            self.downloads.append(source.name)
        return dict(META)

    def models(self):
        volume = self.root / "volume"
        return [
            StoredModel(p.relative_to(volume).as_posix(), p.stat().st_size)
            for p in sorted(volume.rglob("*.gguf"))
        ]

    def remove_model(self, name):
        (self.root / "volume" / name).unlink()

    def model_path(self, name):
        return "/volume/" + name

    def file_path(self, name):
        return "/files/" + name

    def spawn(self, gpu, argv, api_key, env, files):
        call_id = uuid.uuid4().hex[:12]
        for name, text in files.items():
            (self.root / "files" / name).write_text(text)
        port = int(argv[argv.index("--port") + 1])
        with open(self.root / "calls" / f"{call_id}.log", "wb") as log:
            process = subprocess.Popen(
                [sys.executable, str(self.script), *argv[1:]],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **self.server_env, **env, "LLAMA_API_KEY": api_key},
                start_new_session=True,
            )
        record = {"pid": process.pid, "port": port, "gpu": gpu, "started": time.time()}
        (self.root / "calls" / f"{call_id}.json").write_text(json.dumps(record))
        with self._lock:
            self._processes[call_id] = process
        self.spawned.append(
            {"gpu": gpu, "argv": list(argv), "env": dict(env), "files": dict(files)}
        )
        return call_id

    def poll(self, call_id):
        record = self._record(call_id)
        if record is None:
            return ServeStatus(running=False, error="unknown call")
        lines = self._new_lines(call_id)
        if any("listening" in line for line in lines):
            self._listening.add(call_id)
        running = self._alive(call_id, record)
        return ServeStatus(
            running=running,
            upstream=Upstream("127.0.0.1", record["port"])
            if running and call_id in self._listening
            else None,
            logs=tuple(lines),
            error=None if running else "llama-server exited",
        )

    def cancel(self, call_id):
        record = self._record(call_id)
        if record is None:
            return
        try:
            os.killpg(record["pid"], signal.SIGTERM)
        except OSError:
            pass
        with self._lock:
            process = self._processes.pop(call_id, None)
        if process is not None:
            process.wait(timeout=10)
        else:
            deadline = time.monotonic() + 10
            while self._alive(call_id, record) and time.monotonic() < deadline:
                time.sleep(0.02)
        (self.root / "calls" / f"{call_id}.json").unlink(missing_ok=True)

    def calls(self):
        found = []
        for path in sorted((self.root / "calls").glob("*.json")):
            record = json.loads(path.read_text())
            if self._alive(path.stem, record):
                found.append(RemoteCall(path.stem, record["gpu"], record["started"]))
        return found

    def close(self):
        """Stop every call in the provider directory and reap owned processes."""
        for call in self.calls():
            self.cancel(call.id)
        with self._lock:
            processes = list(self._processes.values())
            self._processes.clear()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)

    def _record(self, call_id):
        try:
            return json.loads((self.root / "calls" / f"{call_id}.json").read_text())
        except OSError:
            return None

    def _alive(self, call_id, record):
        with self._lock:
            process = self._processes.get(call_id)
        if process is not None:
            return process.poll() is None
        # Another process started this server. Some sandboxes hide other
        # processes in /proc and never reap orphans, so check the pid and
        # that the server still accepts connections.
        try:
            os.kill(record["pid"], 0)
            with socket.create_connection(("127.0.0.1", record["port"]), timeout=1):
                return True
        except OSError:
            return False

    def _new_lines(self, call_id):
        with self._lock:
            cursor = self._cursors.get(call_id, 0)
            try:
                with open(self.root / "calls" / f"{call_id}.log", "rb") as log:
                    log.seek(cursor)
                    data = log.read()
            except OSError:
                return []
            end = data.rfind(b"\n") + 1
            self._cursors[call_id] = cursor + end
        return data[:end].decode("utf-8", "replace").splitlines()
