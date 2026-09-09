import collections
import csv
import json
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config
from .discovery import EXECUTION_ENV_KEYS, command, hardware
from .settings import launch_args, launch_environment


class Cancelled(Exception):
    pass


class GPUUnavailable(RuntimeError):
    pass


class ResourceConflict(GPUUnavailable):
    pass


def gpu_processes(output):
    """Separate recognised desktop applications from competing compute workloads.

    Wayland and Electron desktop apps can appear in the compute-process list.
    Match executable identity, not GPU-helper arguments: drivers and /proc may
    report different or shortened command lines for the same desktop process.
    This is a coexistence check, not a security boundary or memory guarantee.
    """
    desktop, competing = [], []
    desktop_apps = {
        "chrome",
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "brave",
        "brave-browser",
        "msedge",
        "firefox",
        "nautilus",
        "gnome-shell",
        "gnome-system-monitor",
        "gnome-terminal-server",
        "xorg",
        "xwayland",
        "xdg-desktop-portal",
        "xdg-desktop-portal-gnome",
        "xdg-desktop-portal-gtk",
        "slack",
        "code",
        "code-insiders",
    }
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        if len(row) < 2 or not row[0].strip().isdigit():
            competing.append("Unrecognised GPU process record")
            continue
        pid, reported = row[0].strip(), ",".join(row[1:]).strip()
        try:
            argv = shlex.split(reported)
        except ValueError:
            argv = [reported]
        executable = argv[0] if argv else reported
        try:
            proc = Path("/proc") / pid
            executable = str((proc / "exe").readlink())
        except OSError:
            pass
        name = Path(executable.removesuffix(" (deleted)")).name
        summary = f"{pid}, {name}"
        if name.lower() in desktop_apps:
            desktop.append(summary)
        else:
            competing.append(summary)
    return desktop, competing


class Engine:
    def __init__(self):
        self.process = None
        self.settings = None
        self.ready = False
        self.argv = []
        self.execution_environment = None
        self.attempt_environment = None
        self.lines = collections.deque(maxlen=1000)
        self.log_lock = threading.Lock()
        self.guard = threading.RLock()
        self.base = f"http://127.0.0.1:{config.ENGINE_PORT}"

    def log(self, message):
        with self.log_lock:
            self.lines.append(message)

    def state(self):
        with self.log_lock:
            logs = list(self.lines)[-200:]
        process, settings = self.process, self.settings
        exit_code = process.poll() if process is not None else None
        running = process is not None and exit_code is None
        return {
            "running": running,
            "ready": running and self.ready and self.process is process,
            "pid": process.pid if process else None,
            "error": f"Engine exited with code {exit_code}. See the engine log, then retry."
            if process is not None and exit_code is not None
            else None,
            "settings": settings.dict() if settings else None,
            "argv": self.argv,
            "logs": logs,
            "execution_environment": self.execution_environment,
        }

    def request(self, path, body=None, timeout=30):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read(4000).decode("utf-8", "replace")
            raise RuntimeError(f"{path}: HTTP {e.code}: {detail}") from e

    def stop(self):
        with self.guard:
            self.ready = False
            p = self.process
            if p is None:
                return
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired as e:
                        raise GPUUnavailable(
                            "Owned engine did not exit after kill; queue halted. Inspect workstation GPU."
                        ) from e
            self.process = None
            self.settings = None

    def _logs(self, p):
        for line in p.stdout:
            self.log(line.rstrip())
        p.stdout.close()

    def start(self, s, cancel, timeout=180):
        self.attempt_environment = None
        argv = launch_args(s, config.ENGINE_PORT)
        if cancel.is_set():
            raise Cancelled()
        hw = hardware()
        if not hw["gpus"]:
            raise GPUUnavailable("NVIDIA GPU unavailable: " + str(hw["error"]))
        rc, processes = command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            4,
        )
        if rc != 0:
            raise GPUUnavailable("Cannot check GPU ownership: " + processes[:300])
        # Exclude only this app's owned process before deciding whether a switch is safe.
        owned_pid = (
            self.process.pid if self.process and self.process.poll() is None else None
        )
        processes = "\n".join(
            line
            for line in processes.splitlines()
            if line.split(",", 1)[0].strip() != str(owned_pid)
        )
        desktop, competing = gpu_processes(processes)
        if competing:
            raise ResourceConflict(
                "Other GPU compute processes detected; stop the old model/server first: "
                + "; ".join(competing)[:500]
            )
        if desktop:
            self.log(
                "Allowing desktop GPU processes: "
                + "; ".join(desktop)
                + ". Their VRAM and activity remain part of this workstation benchmark."
            )
        self.stop()
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", config.ENGINE_PORT))
            except OSError as e:
                raise ResourceConflict(
                    f"Port {config.ENGINE_PORT} is occupied by another process. Stop it yourself or change LLLM2_ENGINE_PORT."
                ) from e
        with self.guard:
            if cancel.is_set():
                raise Cancelled()
            self.argv = argv
            self.settings = s
            self.log("Launching: " + " ".join(argv))
            env = launch_environment(s)
            self.execution_environment = {k: env.get(k) for k in EXECUTION_ENV_KEYS}
            p = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
                env=env,
            )
            self.attempt_environment = self.execution_environment
            self.process = p
            threading.Thread(target=self._logs, args=(p,), daemon=True).start()
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if cancel.wait(0.25):
                    raise Cancelled()
                if p.poll() is not None:
                    raise RuntimeError("Engine exited during load. See engine log.")
                try:
                    self.request("/health", timeout=1)
                    break
                except Exception:
                    pass
            else:
                raise TimeoutError("Engine startup exceeded timeout.")
            body = {"messages": [{"role": "user", "content": "Hello"}]}
            if s.effort != "default":
                body["reasoning_effort"] = s.effort
            self.guarded_request("/apply-template", body, cancel, min(timeout, 30))
            with self.guard:
                if cancel.is_set() or self.process is not p or p.poll() is not None:
                    raise Cancelled()
                self.ready = True
        except BaseException:
            self.stop()
            raise

    def guarded_request(self, path, body, cancel, timeout):
        return self._guarded_operation(
            path, lambda: self.request(path, body, timeout), cancel, timeout
        )

    def stream_completion(self, body, cancel, timeout, on_event):
        """Read native /completion SSE; callback receives parsed events, including final."""
        event_lock = threading.Lock()
        closed = threading.Event()

        def read():
            req = urllib.request.Request(
                self.base + "/completion",
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if "text/event-stream" not in response.headers.get(
                        "Content-Type", ""
                    ):
                        raise RuntimeError(
                            "Engine did not return a completion event stream."
                        )
                    data = []
                    size = 0
                    while True:
                        if closed.is_set():
                            raise Cancelled()
                        line = response.readline(1024 * 1024 + 1)
                        if len(line) > 1024 * 1024:
                            raise RuntimeError(
                                "Completion stream line exceeded size bound."
                            )
                        if not line:
                            raise RuntimeError(
                                "Completion stream ended without a terminal response."
                            )
                        line = line.rstrip(b"\r\n")
                        if line.startswith(b"data:"):
                            part = line[5:].lstrip(b" ")
                            size += len(part)
                            if size > 2 * 1024 * 1024:
                                raise RuntimeError(
                                    "Completion stream event exceeded size bound."
                                )
                            data.append(part)
                        elif not line and data:
                            event = json.loads(b"\n".join(data))
                            data, size = [], 0
                            if not isinstance(event, dict):
                                raise RuntimeError("Invalid completion stream event.")
                            if "error" in event:
                                raise RuntimeError(
                                    "Completion stream error: "
                                    + str(event["error"])[:1000]
                                )
                            with event_lock:
                                if closed.is_set():
                                    raise Cancelled()
                                on_event(event)
                            if event.get("stop") is True:
                                return event
            except urllib.error.HTTPError as e:
                raise RuntimeError(
                    f"/completion: HTTP {e.code}: {e.read(4000).decode('utf-8', 'replace')}"
                ) from e

        try:
            return self._guarded_operation("/completion stream", read, cancel, timeout)
        finally:
            # A delayed reader must never mutate a finalized/cancelled sample.
            with event_lock:
                closed.set()

    def _guarded_operation(self, path, operation, cancel, timeout):
        # A wall-clock bound, even if the peer dribbles bytes indefinitely.
        result = []
        error = []
        done = threading.Event()

        def work():
            try:
                result.append(operation())
            except Exception as e:
                error.append(e)
            finally:
                done.set()

        threading.Thread(target=work, daemon=True).start()
        deadline = time.monotonic() + timeout
        while not done.wait(0.1):
            if cancel.is_set():
                self.stop()
                raise Cancelled()
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(f"{path} exceeded {timeout}s wall-clock timeout.")
        if cancel.is_set():
            raise Cancelled()
        if error:
            if isinstance(error[0], urllib.error.URLError) and isinstance(
                error[0].reason, TimeoutError
            ):
                raise TimeoutError(f"{path} timed out: {error[0].reason}") from error[0]
            raise error[0]
        return result[0]
