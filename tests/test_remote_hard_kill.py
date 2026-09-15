"""A hard-killed owner stops billing: the serve side ends once heartbeats stop.

A child process owns a ``RemoteEngine`` over the Modal provider and a fake
``modal`` module. This process runs the Modal serve function with a short
owner grace period. Both sides share the Modal Dict through a file, as the
real ones share it through Modal, so the serve side sees the owner's
heartbeats and their absence.
"""

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from fake_remote import free_port
from lllm2 import modal_app
from test_modal_provider import FakeQueue

TESTS = Path(__file__).parent
GRACE = 1.0
CHILD = r"""
import sys, threading, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fake_remote import ENGINE, META
from test_modal_provider import FakeModal
from test_remote_hard_kill import FileDict
from lllm2 import config
from lllm2.modal_provider import ModalProvider
from lllm2.remote import GpuProbe, RemoteEngine
from lllm2.settings import Settings

state, root, port = sys.argv[2:5]
config.MODELS_DIR = Path(root) / "models"
model = config.MODELS_DIR / "example" / "model.gguf"
model.parent.mkdir(parents=True)
model.write_bytes(b"GGUF")


class Provider(ModalProvider):
    def probe(self, gpu):
        return GpuProbe("NVIDIA T4", 15360, ENGINE)

    def ensure_model(self, source, progress, cancel):
        return dict(META)


fake = FakeModal()
fake.state = FileDict(state)
fake.behaviour["serve"] = lambda *args: {"pending": 10**9}
engine = RemoteEngine(
    Provider(fake, poll_interval=0, deploy=lambda: None, heartbeat=0),
    "T4",
    port=int(port),
    poll_interval=0.05,
    records=root + "/calls.json",
    probes=root + "/probes.json",
)
settings = Settings.parse({"model": str(model), "backend": "modal", "gpu_type": "T4"})
threading.Thread(
    target=engine.start, args=(settings, threading.Event(), 600), daemon=True
).start()
while engine.call_id is None:
    time.sleep(0.01)
print("spawned", engine.call_id, flush=True)
threading.Event().wait()
"""


class FileDict:
    """A Modal Dict stand-in that several processes share through a JSON file."""

    def __init__(self, path):
        self.path = Path(path)

    @contextlib.contextmanager
    def _data(self, write=False):
        with open(self.path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                data = {}
            yield data
            if write:
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps(data))
                os.replace(temporary, self.path)

    def get(self, key, default=None):
        with self._data() as data:
            return data.get(key, default)

    def put(self, key, value, *, skip_if_exists=False):
        with self._data(write=True) as data:
            if skip_if_exists and key in data:
                return False
            data[key] = value
            return True

    def pop(self, key, default=None):
        with self._data(write=True) as data:
            return data.pop(key, default)

    def items(self):
        with self._data() as data:
            return list(data.items())

    def __contains__(self, key):
        with self._data() as data:
            return key in data


def test_serve_side_stops_after_its_owner_is_killed(tmp_path):
    state = FileDict(tmp_path / "state.json")
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            CHILD,
            str(TESTS),
            str(state.path),
            str(tmp_path),
            str(free_port()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result = {}
    server = None
    try:
        line = child.stdout.readline() if child.stdout else ""
        assert line.startswith("spawned "), child.stderr.read() if child.stderr else ""
        call_id = line.split()[1]

        @contextlib.contextmanager
        def forward(port):
            yield SimpleNamespace(tls_socket=("127.0.0.1", free_port()))

        def serve():
            result.update(
                modal_app.run_server(
                    [sys.executable, "-c", "import time; time.sleep(120)"],
                    "key",
                    {},
                    {},
                    call_id=call_id,
                    state=state,
                    logs=FakeQueue(),
                    forward=forward,
                    file_root=str(tmp_path / "files"),
                    heartbeat=0.05,
                    grace=GRACE,
                )
            )

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        # The live owner's status polls keep the call serving past the grace.
        time.sleep(GRACE * 3)
        assert server.is_alive() and child.poll() is None

        # SIGKILL skips every shutdown hook, as a crash or power loss does.
        child.send_signal(signal.SIGKILL)
        child.wait(10)
        server.join(30)
        assert not server.is_alive()
        assert modal_app.OWNER_LOST in result["error"]
        assert modal_app.tunnel_key(call_id) not in state
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(10)
        for stream in (child.stdout, child.stderr):
            if stream:
                stream.close()
