"""A fake engine that serves the llama-server HTTP surface in this process.

``FakeEngine`` implements the ``Engine`` interface without a GPU, a binary or
a child process. Its launch arguments come from the real builder with the fake
provider's engine and checkpoint records, and it logs the startup lines that
results parse, so a panel launch and a measurement exercise the same code as a
real local run.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fake_remote import ENGINE, META, free_port
from lllm2.discovery import EXECUTION_ENV_KEYS
from lllm2.engine import Cancelled, Engine
from lllm2.settings import build_launch_args

GPU = {"index": 0, "name": "Fake GPU 24GB", "total_mib": 23028, "used_mib": 0}
BATCH_LOG = "llama_context: n_batch = 2048"
SAMPLER_LOG = "sampler chain: logits -> temp -> dist"
LOAD_FAILURE_LOG = "ggml_backend_cuda_buffer_type_alloc_buffer: out of memory"


def tokens(text):
    return [ord(c) % 5000 + 1 for c in text]


class FakeEngine(Engine):
    """An engine whose llama-server is an HTTP server thread in this process.

    Attributes:
        port: The loopback port the fake server listens on.
        environment: The environment a real launch would pass to the server.
        fail_load: When True, a start logs a memory failure and never serves.
        requests: The request paths served, in order.
    """

    def __init__(self, environment=None, fail_load=False):
        """Create a stopped fake engine on a free port.

        Args:
            environment: The launch environment to record, or None for one
                that selects the first CUDA device.
            fail_load: Whether starts fail as a model that does not fit.
        """
        super().__init__()
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.environment = (
            {"CUDA_VISIBLE_DEVICES": "0"} if environment is None else environment
        )
        self.fail_load = fail_load
        self.requests = []
        self._server = None
        self._cache = []
        self._cache_lock = threading.Lock()

    def start(self, s, cancel, timeout=180):
        self.attempt_environment = None
        argv = self.launch_args(s)
        if cancel.is_set():
            raise Cancelled()
        self.stop()
        with self.guard:
            if cancel.is_set():
                raise Cancelled()
            self.argv, self.settings = argv, s
            self.log("Launching: " + " ".join(argv))
            env = self.environment
            self.execution_environment = {k: env.get(k) for k in EXECUTION_ENV_KEYS}
            self.attempt_environment = self.execution_environment
            if self.fail_load:
                self.log(LOAD_FAILURE_LOG)
                server = None
            else:
                server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
                server.daemon_threads = True
                threading.Thread(target=server.serve_forever, daemon=True).start()
                for line in (BATCH_LOG, SAMPLER_LOG, "fake llama-server listening"):
                    self.log(line)
            self._server = server
        try:
            self._await_ready(
                s,
                cancel,
                timeout,
                lambda: server is not None and self._server is server,
            )
        except BaseException:
            self.stop()
            raise

    def stop(self):
        with self.guard:
            self.ready = False
            server, self._server = self._server, None
            self.settings = None
        if server is not None:
            server.shutdown()
            server.server_close()

    def alive(self):
        return self._server is not None

    def status(self):
        running = self._server is not None
        return {
            "running": running,
            "ready": running and self.ready,
            "pid": None,
            "error": None,
        }

    def hardware(self, s=None):
        return {"gpus": [dict(GPU)], "error": None, "source": "fake-engine"}

    def gpu_memory(self):
        used = 4096 if self.alive() else 0
        return {"gpus": [dict(GPU, used_mib=used)], "error": None}

    def probe(self, s):
        return dict(ENGINE)

    def metadata(self, path):
        return dict(META)

    def identity(self, path):
        return {"path": path, "size": 3000, "mtime_ns": 0}

    def launch_args(self, s):
        if s.remote:
            raise ValueError("The fake engine serves local backends only.")
        return build_launch_args(s, self.port, self.probe(s), self.metadata(s.model))

    def _handler(self):
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                engine.requests.append(self.path)
                if self.path == "/v1/models":
                    self.reply({"data": [{"id": engine.argv[2]}]})
                elif self.path == "/props":
                    s = engine.settings
                    self.reply(
                        {
                            "default_generation_settings": {"n_ctx": s.context},
                            "total_slots": s.slots,
                        }
                    )
                else:
                    self.reply({"status": "ok"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                engine.requests.append(self.path)
                if self.path == "/apply-template":
                    text = "\n".join(m["content"] for m in body["messages"])
                    suffix = "<assistant>" if body.get("add_generation_prompt") else ""
                    self.reply({"prompt": "<user>" + text + suffix})
                elif self.path == "/tokenize":
                    self.reply({"tokens": tokens(body["content"])})
                elif self.path == "/completion":
                    self.complete(body)
                else:
                    self.send_error(404)

            def complete(self, body):
                prompt = body.get("prompt", [])
                if isinstance(prompt, str):
                    prompt = tokens(prompt)
                n = int(body.get("n_predict", 16))
                with engine._cache_lock:
                    reused = 0
                    if body.get("cache_prompt", True):
                        for a, b in zip(engine._cache, prompt, strict=False):
                            if a != b:
                                break
                            reused += 1
                        reused = min(reused, max(len(prompt) - 1, 0))
                    engine._cache[:] = prompt
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
                    self.reply(final | {"content": "x" * n})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(n):
                    event = {"content": "x", "tokens": [100 + i], "stop": False}
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
                self.wfile.flush()
                self.close_connection = True

            def log_message(self, *args):
                pass

        return Handler
