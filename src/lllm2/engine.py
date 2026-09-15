import abc
import collections
import csv
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config
from .discovery import (
    EXECUTION_ENV_KEYS,
    command,
    hardware,
    identity,
    metadata,
    probe,
)
from .settings import capabilities, launch_args, launch_environment


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


def process_memory(process):
    """Sample the resident memory of an owned engine process from /proc.

    Args:
        process: The engine's ``subprocess.Popen`` handle, or None.

    Returns:
        A host memory record. Values are None and ``error`` explains why when
        the process is not running or /proc cannot be read.
    """
    result = {
        "pid": process.pid if process else None,
        "rss_mib": None,
        "anonymous_mib": None,
        "swap_mib": None,
        "lifetime_peak_rss_mib": None,
        "available_mib": None,
    }
    try:
        if process is None or process.poll() is not None:
            raise OSError("Owned engine is not running")
        fields = {}
        for line in (
            (Path("/proc") / str(process.pid) / "status").read_text().splitlines()
        ):
            key, _, value = line.partition(":")
            if key in ["VmRSS", "RssAnon", "VmSwap", "VmHWM"]:
                fields[key] = int(value.split()[0]) / 1024
        if process.poll() is not None:
            raise OSError("Owned engine exited during memory sampling")
        for name, key in [
            ("rss_mib", "VmRSS"),
            ("anonymous_mib", "RssAnon"),
            ("swap_mib", "VmSwap"),
            ("lifetime_peak_rss_mib", "VmHWM"),
        ]:
            result[name] = fields.get(key)
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                result["available_mib"] = int(line.split()[1]) / 1024
    except (OSError, ValueError) as e:
        result["error"] = str(e)
    return result


def unsampled_memory():
    """Return an empty host memory record for an engine with no local process.

    Returns:
        A host memory record with every value set to None.
    """
    return {
        "pid": None,
        "rss_mib": None,
        "anonymous_mib": None,
        "swap_mib": None,
        "lifetime_peak_rss_mib": None,
        "available_mib": None,
    }


class Engine(abc.ABC):
    """An llama-server reachable over HTTP at a loopback ``base`` URL.

    The base class holds everything that does not depend on where the server
    runs: the bounded log buffer, the guard lock, the HTTP helpers and the
    readiness check. Subclasses own the server's lifetime. Callers use
    ``alive()``, ``logs()`` and ``memory_sampler()`` instead of reaching into a
    process handle, so they need no backend branches.
    """

    LOG_LINES = 1000
    STATE_LOG_LINES = 200

    def __init__(self):
        self.settings = None
        self.ready = False
        self.argv = []
        self.execution_environment = None
        self.attempt_environment = None
        self._lines = collections.deque(maxlen=self.LOG_LINES)
        self._log_lock = threading.Lock()
        self.guard = threading.RLock()
        self.base = f"http://127.0.0.1:{config.ENGINE_PORT}"

    @abc.abstractmethod
    def start(self, s, cancel, timeout=180):
        """Start a server for the settings and wait until it serves requests.

        Args:
            s: The launch settings.
            cancel: An event that aborts the start when set.
            timeout: The startup bound in seconds.

        Raises:
            Cancelled: The cancel event was set.
            GPUUnavailable: The server cannot get the resources it needs.
            TimeoutError: The server did not become ready in time.
            RuntimeError: The server exited during load.
        """

    @abc.abstractmethod
    def stop(self):
        """Stop the owned server, if any, and forget its settings."""

    @abc.abstractmethod
    def alive(self):
        """Return whether the owned server is running.

        Returns:
            True while a started server has not exited or been stopped.
        """

    @abc.abstractmethod
    def status(self):
        """Return the server's lifecycle fields.

        Returns:
            A dict with ``running``, ``ready``, ``pid`` and ``error``. ``pid``
            is None when the server has no local process.
        """

    @abc.abstractmethod
    def hardware(self, s=None):
        """Describe the GPU that serves, or would serve, the settings.

        Args:
            s: Settings whose backend and GPU type select the hardware, or None
                for the engine's current choice.

        Returns:
            A description in the ``discovery.hardware()`` shape. ``source`` is
            ``"local"`` or the remote provider name.
        """

    @abc.abstractmethod
    def probe(self, s):
        """Return the capability record of the engine binary that serves settings.

        Args:
            s: Settings naming the engine and backend.

        Returns:
            A record in the ``discovery.probe()`` shape.

        Raises:
            ValueError: The engine binary cannot be probed.
        """

    @abc.abstractmethod
    def metadata(self, path):
        """Return GGUF metadata for a model or drafter path.

        Args:
            path: A path as in ``Settings.model``.

        Returns:
            A record in the ``discovery.metadata()`` shape.
        """

    @abc.abstractmethod
    def identity(self, path):
        """Identify a model or drafter file for saved results.

        Args:
            path: A path as in ``Settings.model``.

        Returns:
            A dict with at least ``path``, ``size`` and ``mtime_ns``. A remote
            engine adds store and catalogue identity, including ``sha256``
            when the catalogue pins one.

        Raises:
            OSError: A local file cannot be read.
        """

    @abc.abstractmethod
    def launch_args(self, s):
        """Build and validate the llama-server command line for settings.

        Args:
            s: The launch settings.

        Returns:
            The argument list, binary first.

        Raises:
            ValueError: This engine cannot run the settings.
        """

    def capabilities(self, s):
        """Report which settings this engine and checkpoint support.

        Args:
            s: Settings to check.

        Returns:
            The ``settings.capabilities`` result for this engine's evidence.
        """
        return capabilities(s, self.probe(s), self.metadata(s.model))

    def prepare(self, s):
        """Gather launch evidence that validation leaves out, before a run.

        Validation, capabilities and defaults use cheap evidence. A run calls
        this first so that its saved engine and hardware identity is exact.
        Engines whose evidence is always exact keep this default, which does
        nothing.

        Args:
            s: The settings the run launches.
        """
        return None

    def gpu_memory(self):
        """Sample GPU memory use while a workload runs.

        Engines that cannot observe their GPU keep this default, which returns
        no GPUs, so measurements report unknown GPU memory rather than zero.

        Returns:
            A dict with ``gpus`` in the ``discovery.hardware()`` shape and
            ``error``.
        """
        return {"gpus": [], "error": "This engine does not sample GPU memory."}

    def defaults_inputs(self, s):
        """Return the evidence that starting defaults need for settings.

        Args:
            s: Settings naming the model, backend and GPU type.

        Returns:
            Keyword arguments for ``starting_defaults`` and
            ``saved_qualifications``: ``host``, ``engine``, ``meta`` and
            ``model``. An empty dict lets those functions probe locally.
        """
        return {}

    def memory_sampler(self):
        """Return a callable that samples the server's host memory.

        Engines without a local process keep this default, which samples
        nothing.

        Returns:
            A callable with no arguments that returns a host memory record.
        """
        return unsampled_memory

    def log(self, message):
        """Append a line to the bounded log and echo it to stderr.

        Args:
            message: The log line, without a trailing newline.
        """
        with self._log_lock:
            self._lines.append(message)
        try:
            print(f"[llama-server] {message}", file=sys.stderr, flush=True)
        except OSError:
            # Keep draining the child pipe if the terminal/journal sink closes.
            pass

    def logs(self, limit=None):
        """Return a copy of the retained log lines, oldest first.

        Args:
            limit: The maximum number of most recent lines, or None for all.

        Returns:
            A list of log lines.
        """
        with self._log_lock:
            lines = list(self._lines)
        return lines if limit is None else lines[-limit:]

    def state(self):
        """Return a snapshot for the panel and the CLI.

        Returns:
            The ``status()`` fields plus settings, argv, recent logs and the
            execution environment.
        """
        logs = self.logs(self.STATE_LOG_LINES)
        status = self.status()
        settings = self.settings
        return status | {
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

    def _await_ready(self, s, cancel, timeout, launched):
        """Wait for ``/health``, apply the chat template, then mark the engine ready.

        Args:
            s: The launch settings.
            cancel: An event that aborts the wait when set.
            timeout: The startup bound in seconds.
            launched: A callable that returns whether this launch is still
                the owned, running server.

        Raises:
            Cancelled: The cancel event was set or the launch was replaced.
            RuntimeError: The server exited during load.
            TimeoutError: ``/health`` did not answer within the timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancel.wait(0.25):
                raise Cancelled()
            if not launched():
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
            if cancel.is_set() or not launched():
                raise Cancelled()
            self.ready = True

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


class LocalEngine(Engine):
    """An llama-server child process on this machine's NVIDIA GPU."""

    def __init__(self):
        super().__init__()
        self.process = None

    def alive(self):
        process = self.process
        return process is not None and process.poll() is None

    def hardware(self, s=None):
        return hardware()

    def probe(self, s):
        return probe(s.engine)

    def metadata(self, path):
        return metadata(path)

    def identity(self, path):
        return identity(path)

    def launch_args(self, s):
        if s.remote:
            raise ValueError(
                f"The local engine cannot serve the {s.backend} backend. Select a local backend."
            )
        return launch_args(s, config.ENGINE_PORT)

    def gpu_memory(self):
        h = hardware()
        return {"gpus": h["gpus"], "error": h["error"]}

    def status(self):
        process = self.process
        exit_code = process.poll() if process is not None else None
        running = process is not None and exit_code is None
        return {
            "running": running,
            "ready": running and self.ready and self.process is process,
            "pid": process.pid if process else None,
            "error": f"Engine exited with code {exit_code}. See the engine log, then retry."
            if process is not None and exit_code is not None
            else None,
        }

    def memory_sampler(self):
        # Bind the process now so a restart cannot mix two processes in one sample.
        with self.guard:
            process = self.process
        return lambda: process_memory(process)

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

    def _drain_output(self, p):
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
            threading.Thread(target=self._drain_output, args=(p,), daemon=True).start()
        try:
            self._await_ready(
                s, cancel, timeout, lambda: self.process is p and p.poll() is None
            )
        except BaseException:
            self.stop()
            raise
