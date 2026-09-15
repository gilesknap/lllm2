"""Remote engines: the provider interface, the provider registry and RemoteEngine.

A remote provider runs llama-server on a GPU it rents out. ``RemoteEngine``
drives any provider through the ``RemoteProvider`` interface and serves the
remote server on the local engine port through ``EngineProxy``, so bench, warm
and every other client talk to it exactly as they talk to a local engine.
Providers register a factory with ``register_provider`` and a GPU table with
``gpu_tables.register_gpu_table`` under the same name.
"""

import abc
import atexit
import json
import os
import secrets
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import config
from .catalogue import files as catalogue_files
from .catalogue import local_paths
from .discovery import EXECUTION_ENV_KEYS
from .engine import Cancelled, Engine, ResourceConflict
from .gpu_tables import gpu_type, table_hardware
from .proxy import EngineProxy, Upstream
from .settings import Settings, build_launch_args

DEFAULT_IDLE_TIMEOUT = 30 * 60


@dataclass(frozen=True)
class GpuProbe:
    """What a provider's probe found in a container with one GPU type.

    Attributes:
        name: The GPU model name that the driver reports.
        total_mib: The GPU memory that the driver reports, in MiB.
        engine: The engine capability record in the ``discovery.probe()``
            shape, with ``path`` set to the binary path inside the container.
            It also carries the ``cuda_graph``, ``cache_kernel`` and
            ``environment`` keys that ``settings.capabilities`` accepts.
    """

    name: str
    total_mib: int
    engine: dict


@dataclass(frozen=True)
class ModelSource:
    """A model file for a provider to place in its model store.

    Attributes:
        name: The store-relative path of the main GGUF file, such as
            ``"Qwen3-GGUF/Qwen3-Q4_K_M.gguf"``.
        repo: The Hugging Face repository to download from, or None when the
            file must already be in the store.
        files: The repository files to download, main file first.
        revision: The repository revision, or None for the default branch.
    """

    name: str
    repo: str | None = None
    files: tuple[str, ...] = ()
    revision: str | None = None


@dataclass(frozen=True)
class DownloadProgress:
    """Progress of one model download.

    Attributes:
        file: The file being downloaded.
        done_bytes: Bytes present in the store so far.
        total_bytes: The file size, or None when unknown.
    """

    file: str
    done_bytes: int
    total_bytes: int | None


@dataclass(frozen=True)
class StoredModel:
    """A model file in a provider's model store.

    Attributes:
        name: The store-relative path.
        size_bytes: The file size.
    """

    name: str
    size_bytes: int


@dataclass(frozen=True)
class ServeStatus:
    """The state of a serve call, as one poll reports it.

    Attributes:
        running: True until the call ends for any reason.
        upstream: The address that reaches llama-server, or None before the
            provider has opened it.
        logs: Log lines produced since the previous poll of this call.
        error: Why the call ended, or None.
    """

    running: bool
    upstream: Upstream | None = None
    logs: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class RemoteCall:
    """A running lllm2 serve call in the provider account.

    Attributes:
        id: The provider's call identifier.
        gpu: The GPU type, or None when the provider cannot tell.
        started: The start time as Unix seconds, or None when unknown.
    """

    id: str
    gpu: str | None = None
    started: float | None = None


class RemoteProvider(abc.ABC):
    """A service that runs llama-server on rented GPUs.

    Every method may block on the network. Methods raise ``RuntimeError`` with
    a user-facing message for provider failures, such as missing credentials.

    Attributes:
        name: The registry name. It is also the GPU table name and the
            hardware ``source``.
        server_host: The address llama-server binds inside the container.
        server_port: The port llama-server listens on inside the container.
    """

    name: str = ""
    server_host: str = "0.0.0.0"
    server_port: int = 8080

    @abc.abstractmethod
    def probe(self, gpu: str) -> GpuProbe:
        """Inspect the GPU and the engine binary in a container of one GPU type.

        Args:
            gpu: The provider's GPU type string.

        Returns:
            The probed GPU and engine capabilities.
        """

    @abc.abstractmethod
    def ensure_model(
        self,
        source: ModelSource,
        progress: Callable[[DownloadProgress], None],
        cancel: threading.Event,
    ) -> dict:
        """Make a model present in the store, downloading what is missing.

        Args:
            source: The model to place in the store.
            progress: A callable that receives download progress.
            cancel: An event that aborts the download when set. Partial files
                stay in the store so a later call can resume.

        Returns:
            The GGUF metadata record in the ``discovery.metadata()`` shape.

        Raises:
            Cancelled: The cancel event was set.
        """

    @abc.abstractmethod
    def models(self) -> list[StoredModel]:
        """List the model files in the store.

        Returns:
            The stored models, sorted by name.
        """

    @abc.abstractmethod
    def remove_model(self, name: str) -> None:
        """Delete a model file from the store.

        Args:
            name: The store-relative path.
        """

    @abc.abstractmethod
    def model_path(self, name: str) -> str:
        """Return where a stored model appears inside a serve container.

        Args:
            name: The store-relative path.

        Returns:
            The absolute path inside the container.
        """

    @abc.abstractmethod
    def file_path(self, name: str) -> str:
        """Return where a file passed to ``spawn`` appears inside the container.

        Args:
            name: The file name, without directories.

        Returns:
            The absolute path inside the container.
        """

    @abc.abstractmethod
    def spawn(
        self,
        gpu: str,
        argv: list[str],
        api_key: str,
        env: dict[str, str],
        files: dict[str, str],
    ) -> str:
        """Start a serve call that runs llama-server and opens a tunnel to it.

        The call runs until it is cancelled or llama-server exits.

        Args:
            gpu: The provider's GPU type string.
            argv: The llama-server command line, binary first.
            api_key: The key llama-server must require on every request. The
                provider passes it outside ``argv``, for example as
                ``LLAMA_API_KEY``.
            env: Extra environment variables for llama-server.
            files: Small text files, such as chat templates, keyed by the
                name that ``file_path`` maps.

        Returns:
            The call identifier.
        """

    @abc.abstractmethod
    def poll(self, call_id: str) -> ServeStatus:
        """Report a serve call's state without waiting.

        Args:
            call_id: The call identifier.

        Returns:
            The call state. An unknown call is reported as not running.
        """

    @abc.abstractmethod
    def cancel(self, call_id: str) -> None:
        """Stop a serve call and release its GPU.

        Cancelling a call that already ended does nothing.

        Args:
            call_id: The call identifier.
        """

    @abc.abstractmethod
    def calls(self) -> list[RemoteCall]:
        """List the running lllm2 serve calls in the provider account.

        Returns:
            The running calls, including ones this process did not start.
        """


PROVIDERS: dict[str, Callable[[], RemoteProvider]] = {}


def register_provider(name: str, factory: Callable[[], RemoteProvider]) -> None:
    """Register or replace a remote provider.

    Args:
        name: The provider name. It must match the provider's GPU table name.
        factory: A callable that creates the provider. It runs only when the
            provider is first used, so optional client libraries load lazily.

    Raises:
        ValueError: If the name is empty or ``"local"``.
    """
    if not name or name == "local":
        raise ValueError("A remote provider needs a non-local name.")
    PROVIDERS[name] = factory


def remote_provider(name: str) -> RemoteProvider:
    """Create a registered remote provider.

    Args:
        name: The provider name.

    Returns:
        A new provider instance.

    Raises:
        ValueError: If no provider is registered under the name.
    """
    try:
        factory = PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown remote provider: {name}.") from None
    return factory()


def _modal_provider() -> RemoteProvider:
    from .modal_provider import create_provider

    return create_provider()


register_provider("modal", _modal_provider)


def catalogue_source(entry: dict) -> ModelSource:
    """Describe a catalogue model for a provider to download from Hugging Face.

    Args:
        entry: A catalogue entry with ``name``, ``repo``, ``file`` and the
            optional ``files``, ``mmproj`` and ``revision`` keys.

    Returns:
        A source whose name matches the model's path under the managed model
        directory, so local and remote stores use the same layout.

    Raises:
        ValueError: If the entry has an unsafe directory or file path.
    """
    local_paths(entry)
    files = tuple(catalogue_files(entry))
    return ModelSource(
        name=f"{entry['name']}/{files[0]}",
        repo=entry["repo"],
        files=files,
        revision=entry.get("revision"),
    )


def model_source(path: str) -> ModelSource:
    """Describe a model under the managed model directory by its relative path.

    Args:
        path: A model path, as in ``Settings.model``.

    Returns:
        A source with no download repository. Its name is the path relative to
        the managed model directory.

    Raises:
        ValueError: If the path is outside the managed model directory.
    """
    root = config.MODELS_DIR.expanduser().resolve()
    target = Path(path).expanduser().resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError(
            "A remote launch needs a model under the managed model directory."
        )
    return ModelSource(name=target.relative_to(root).as_posix())


class CallRecords:
    """Owned serve calls saved on disk, so a later session can adopt them.

    A record holds the call's API key, so the file is private to the user.
    """

    def __init__(self, path):
        """Use a record file.

        Args:
            path: The JSON file path. It need not exist.
        """
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self):
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        # The mode above applies only to a new file; a leftover one keeps its own.
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump(data, file)
        os.replace(temporary, self.path)

    def put(self, provider, call_id, record):
        """Save a record for a call.

        Args:
            provider: The provider name.
            call_id: The call identifier.
            record: A JSON-serialisable dict.
        """
        with self._lock:
            data = self._read()
            data.setdefault(provider, {})[call_id] = record
            self._write(data)

    def get(self, provider, call_id):
        """Return a call's record.

        Args:
            provider: The provider name.
            call_id: The call identifier.

        Returns:
            The record, or None when there is none.
        """
        with self._lock:
            return self._read().get(provider, {}).get(call_id)

    def keep(self, provider, call_ids):
        """Delete the records of calls that are no longer running.

        Args:
            provider: The provider name.
            call_ids: The identifiers of the running calls.
        """
        with self._lock:
            data = self._read()
            calls = data.get(provider, {})
            stale = set(calls) - set(call_ids)
            if stale:
                data[provider] = {k: v for k, v in calls.items() if k not in stale}
                self._write(data)

    def remove(self, provider, call_id):
        """Delete a call's record, if any.

        Args:
            provider: The provider name.
            call_id: The call identifier.
        """
        with self._lock:
            data = self._read()
            if call_id in data.get(provider, {}):
                del data[provider][call_id]
                self._write(data)


class _Call:
    """One serve call that an engine owns, with its proxy and monitor state."""

    def __init__(self, call_id, api_key, proxy, started):
        self.id = call_id
        self.api_key = api_key
        self.proxy = proxy
        self.started = time.monotonic() - max(0.0, time.time() - started)
        self.started_unix = started
        self.upstream = None
        self.done = threading.Event()


_ENGINES: "weakref.WeakSet[RemoteEngine]" = weakref.WeakSet()


def stop_owned_calls():
    """Stop every serve call that a remote engine in this process owns.

    The module registers this function to run at interpreter exit. The panel
    and the CLI also stop their engine when they shut down.
    """
    for engine in list(_ENGINES):
        engine.shutdown()


atexit.register(stop_owned_calls)


class RemoteEngine(Engine):
    """An llama-server on a provider GPU, served on the local engine port.

    ``start`` probes the GPU type once, places the model in the provider store,
    builds the command line with ``build_launch_args``, spawns a serve call
    with a new API key and proxies the call's tunnel on the engine port. Local
    GPU ownership and desktop checks do not apply. An idle timer stops the call
    when no request has passed through the proxy for ``idle_timeout`` seconds.

    Attributes:
        provider: The ``RemoteProvider``.
        gpu: The provider's GPU type string.
        idle_timeout: Seconds without requests before a ready engine stops
            itself, or None or 0 to disable the idle timer.
        port: The loopback port the proxy listens on.
    """

    def __init__(
        self,
        provider,
        gpu,
        *,
        idle_timeout=DEFAULT_IDLE_TIMEOUT,
        port=None,
        poll_interval=2.0,
        records=None,
        sources=model_source,
    ):
        """Create an engine for one provider GPU type.

        Args:
            provider: The ``RemoteProvider``.
            gpu: The provider's GPU type string.
            idle_timeout: Seconds without requests before the engine stops its
                call, or None or 0 to disable the idle timer.
            port: The loopback port for the proxy. None uses the engine port.
            poll_interval: Seconds between provider status polls.
            records: The path of the owned call record file. None uses the
                state directory.
            sources: A callable that maps a model or drafter path to a
                ``ModelSource``.
        """
        super().__init__()
        self.provider = provider
        self.gpu = gpu
        self.idle_timeout = idle_timeout
        self.port = config.ENGINE_PORT if port is None else port
        self.base = f"http://127.0.0.1:{self.port}"
        self.poll_interval = poll_interval
        self._records = CallRecords(
            config.STATE_DIR / "remote-calls.json" if records is None else records
        )
        self._sources = sources
        self._probes = {}
        self._generation = 0
        self._call = None
        self._phase = None
        self._download = None
        self._error = None
        _ENGINES.add(self)

    @property
    def call_id(self):
        """Return the identifier of the owned serve call, or None."""
        call = self._call
        return call.id if call is not None else None

    def alive(self):
        return self._call is not None

    def hardware(self):
        """Describe the remote GPU for starting defaults and results.

        Returns:
            A description in the ``discovery.hardware()`` shape with ``source``
            set to the provider name. Probed GPU name and memory replace the
            table figures once a probe has run.

        Raises:
            ValueError: If the provider has no table entry for the GPU type.
        """
        description = table_hardware(self.provider.name, self.gpu)
        probed = self._probes.get(self.gpu)
        if probed is not None:
            description["gpus"][0].update(name=probed.name, total_mib=probed.total_mib)
        return description

    def status(self):
        call = self._call
        elapsed = time.monotonic() - call.started if call is not None else None
        try:
            price = gpu_type(self.provider.name, self.gpu).usd_per_hour
        except ValueError:
            price = None
        return {
            "running": call is not None,
            "ready": call is not None and self.ready,
            "pid": None,
            "error": self._error,
            "provider": self.provider.name,
            "gpu": self.gpu,
            "call_id": call.id if call is not None else None,
            "phase": self._phase,
            "download": self._download,
            "elapsed_seconds": elapsed,
            "usd_per_hour": price,
            "estimated_cost_usd": elapsed * price / 3600
            if elapsed is not None and price is not None
            else None,
            "idle_timeout_seconds": self.idle_timeout or None,
            "idle_remaining_seconds": self._idle_remaining(call),
        }

    def start(self, s, cancel, timeout=180):
        """Start a serve call for the settings and wait until it serves requests.

        Probing and model download are bounded only by ``cancel``. The timeout
        covers the container start and the model load.

        Args:
            s: The launch settings.
            cancel: An event that aborts the start when set.
            timeout: The container start and model load bound in seconds.

        Raises:
            Cancelled: The cancel event was set, or ``stop`` ran meanwhile.
            ResourceConflict: The engine port is in use.
            ValueError: The remote engine or model cannot run these settings.
            TimeoutError: The server did not become ready in time.
            RuntimeError: The serve call ended during start, or the provider
                failed.
        """
        self.attempt_environment = None
        if cancel.is_set():
            raise Cancelled()
        self.stop()
        with self.guard:
            self._generation += 1
            generation = self._generation
            self._error = self._download = None
        proxy = self._bind_proxy()
        call = None

        def check():
            if cancel.is_set() or self._generation != generation:
                raise Cancelled()

        try:
            self._phase = "probing"
            probed = self._probe()
            check()
            self._phase = "downloading model"
            source, meta = self._ensure(s.model, cancel)
            paths = {s.model: self.provider.model_path(source.name)}
            if s.speculation == "draft-dflash" and s.drafter:
                drafter, drafter_meta = self._ensure(s.drafter, cancel)
                meta = {**meta, "drafter": drafter_meta}
                paths[s.drafter] = self.provider.model_path(drafter.name)
            check()
            files = {}
            if s.chat_template:
                template = Path(s.chat_template).expanduser()
                try:
                    files[template.name] = template.read_text()
                except OSError as e:
                    raise ValueError("Chat template file does not exist.") from e
                paths[s.chat_template] = self.provider.file_path(template.name)
            argv = build_launch_args(
                s,
                self.provider.server_port,
                probed.engine,
                meta,
                host=self.provider.server_host,
                path=paths.__getitem__,
            )
            env = {}
            if s.cuda_graph_opt != "default":
                env["GGML_CUDA_GRAPH_OPT"] = "1" if s.cuda_graph_opt == "on" else "0"
            api_key = secrets.token_urlsafe(32)
            with self.guard:
                check()
                self.argv, self.settings = argv, s
                self.execution_environment = {k: env.get(k) for k in EXECUTION_ENV_KEYS}
                self.log("Launching: " + " ".join(argv))
                self._phase = "starting container"
                call_id = self.provider.spawn(self.gpu, argv, api_key, env, files)
                self.attempt_environment = self.execution_environment
                call = _Call(call_id, api_key, proxy, time.time())
                proxy = None
                self._own(call, s)
            self._serve(call, s, cancel, timeout)
        except BaseException:
            if proxy is not None:
                proxy.close()
            if call is not None and self._call is call:
                self.stop()
            elif self._generation == generation and self._phase != "exited":
                self._phase = None
            raise

    def adopt(self, call_id, cancel=None, timeout=180):
        """Serve a running call from an earlier session through this engine.

        The engine takes over the call's settings, GPU type and idle timer. If
        the call does not become ready, the engine cancels it.

        Args:
            call_id: The identifier of a call listed by ``orphans``.
            cancel: An event that aborts the wait when set, or None.
            timeout: The readiness bound in seconds.

        Raises:
            ValueError: The call is not running, or this machine has no record
                of its API key.
            Cancelled: The cancel event was set.
            ResourceConflict: The engine port is in use.
            TimeoutError: The server did not become ready in time.
            RuntimeError: The call ended while adopting it.
        """
        cancel = cancel or threading.Event()
        if all(c.id != call_id for c in self.provider.calls()):
            self._records.remove(self.provider.name, call_id)
            raise ValueError("That remote call is no longer running.")
        record = self._records.get(self.provider.name, call_id)
        if record is None:
            raise ValueError(
                "No saved key for this remote call, so it cannot be adopted. Stop it instead."
            )
        s = Settings.parse(record["settings"])
        self.stop()
        with self.guard:
            self._generation += 1
            self._error = self._download = None
        proxy = self._bind_proxy()
        with self.guard:
            self.gpu = record["gpu"]
            self.argv, self.settings = record["argv"], s
            self.execution_environment = self.attempt_environment = None
            self.log(f"Adopting remote call {call_id}")
            call = _Call(call_id, record["api_key"], proxy, record["started"])
            self._own(call, s)
            self._phase = "loading model"
        try:
            self._serve(call, s, cancel, timeout)
        except BaseException:
            if self._call is call:
                self.stop()
            raise

    def orphans(self):
        """List running serve calls that no engine in this process owns.

        Also deletes saved records of calls that have ended.

        Returns:
            A list of dicts with ``id``, ``gpu``, ``started`` (Unix seconds or
            None), ``model`` (the settings model path or None) and
            ``adoptable`` (True when a saved key allows ``adopt``).
        """
        running = self.provider.calls()
        self._records.keep(self.provider.name, [c.id for c in running])
        owned = {engine.call_id for engine in list(_ENGINES)}
        found = []
        for call in running:
            if call.id in owned:
                continue
            record = self._records.get(self.provider.name, call.id)
            found.append(
                {
                    "id": call.id,
                    "gpu": call.gpu or (record or {}).get("gpu"),
                    "started": call.started,
                    "model": (record or {}).get("settings", {}).get("model"),
                    "adoptable": record is not None,
                }
            )
        return found

    def cancel_orphan(self, call_id):
        """Stop a running serve call that this engine does not own.

        Args:
            call_id: The identifier of a call listed by ``orphans``.

        Raises:
            ValueError: This engine owns the call; use ``stop`` instead.
        """
        if call_id == self.call_id:
            raise ValueError("This engine owns that call. Stop the engine instead.")
        self.provider.cancel(call_id)
        self._records.remove(self.provider.name, call_id)
        self.log(f"Stopped orphaned remote call {call_id}")

    def stop(self):
        """Cancel the owned serve call, if any, and close the proxy.

        Raises:
            RuntimeError: The provider could not cancel the call. The call
                record stays, so ``orphans`` lists the call later.
        """
        with self.guard:
            self._generation += 1
            self.ready = False
            self.settings = None
            self._error = None
            call = self._call
            if call is None:
                if self._phase not in (None, "idle stopped", "exited"):
                    self._phase = "stopped"
                return
            self._phase = "stopped"
            try:
                self._cancel(call)
            finally:
                # Report the call as running until the provider has cancelled it.
                self._call = None

    def shutdown(self):
        """Stop the owned serve call and log, rather than raise, a failure."""
        try:
            self.stop()
        except Exception as e:
            self.log(f"Shutdown could not stop the remote engine: {e}")

    def _bind_proxy(self):
        proxy = EngineProxy(self.port, self.log)
        try:
            proxy.start()
        except OSError as e:
            raise ResourceConflict(
                f"Port {self.port} is occupied by another process. Stop it yourself or change LLLM2_ENGINE_PORT."
            ) from e
        return proxy

    def _probe(self):
        if self.gpu not in self._probes:
            self._probes[self.gpu] = self.provider.probe(self.gpu)
        return self._probes[self.gpu]

    def _ensure(self, path, cancel):
        source = self._sources(path)

        def progress(update):
            self._download = {
                "file": update.file,
                "done_bytes": update.done_bytes,
                "total_bytes": update.total_bytes,
            }

        meta = self.provider.ensure_model(source, progress, cancel)
        self._download = None
        return source, meta

    def _own(self, call, s):
        self._call = call
        self._records.put(
            self.provider.name,
            call.id,
            {
                "api_key": call.api_key,
                "gpu": self.gpu,
                "settings": s.dict(),
                "argv": self.argv,
                "started": call.started_unix,
            },
        )
        threading.Thread(target=self._monitor, args=(call,), daemon=True).start()

    def _serve(self, call, s, cancel, timeout):
        deadline = time.monotonic() + timeout
        while call.upstream is None:
            if cancel.wait(0.05):
                raise Cancelled()
            if self._call is not call:
                raise RuntimeError("Engine exited during load. See engine log.")
            if time.monotonic() > deadline:
                raise TimeoutError("Engine startup exceeded timeout.")
        call.proxy.connect(call.upstream, call.api_key)
        self._phase = "loading model"
        self._await_ready(
            s,
            cancel,
            max(deadline - time.monotonic(), 1),
            lambda: self._call is call,
        )
        with self.guard:
            if self._call is call:
                self._phase = "ready"

    def _monitor(self, call):
        failures = 0
        while not call.done.is_set():
            try:
                state = self.provider.poll(call.id)
            except Exception as e:
                failures += 1
                if failures in (1, 30) or failures % 300 == 0:
                    self.log(f"Remote status check failed: {e}")
                # Failed status checks must not keep an idle call running.
                if self._idle_remaining(call) == 0:
                    self._idle_stop(call)
                    return
                call.done.wait(self.poll_interval)
                continue
            failures = 0
            for line in state.logs:
                self.log(line)
            if call.done.is_set():
                return
            if state.upstream is not None:
                call.upstream = state.upstream
            if not state.running:
                self._ended(call, state.error)
                return
            if self._idle_remaining(call) == 0:
                self._idle_stop(call)
                return
            call.done.wait(self.poll_interval)

    def _ended(self, call, error):
        with self.guard:
            if self._call is not call:
                return
            self._call = None
            self.ready = False
            self._phase = "exited"
            self._error = f"Remote engine stopped: {error or 'the serve call ended'}. See the engine log, then retry."
            call.done.set()
            call.proxy.close()
            self._records.remove(self.provider.name, call.id)

    def _idle_stop(self, call):
        with self.guard:
            if self._call is not call:
                return
            self.log(
                f"No requests for {self.idle_timeout:g} seconds; stopping the remote engine."
            )
            self.ready = False
            try:
                self._cancel(call)
            except RuntimeError as e:
                self._error = str(e)
            finally:
                self._call = None
                self.settings = None
                self._phase = "idle stopped"

    def _cancel(self, call):
        call.done.set()
        call.proxy.close()
        try:
            self.provider.cancel(call.id)
        except Exception as e:
            message = f"Could not cancel remote call {call.id}: {e}"
            self.log(message)
            raise RuntimeError(message + ". Stop it from the provider's tools.") from e
        self._records.remove(self.provider.name, call.id)
        self.log(f"Stopped remote call {call.id}")

    def _idle_remaining(self, call):
        if call is None or not self.ready or not self.idle_timeout:
            return None
        active, last = call.proxy.activity()
        if active:
            return float(self.idle_timeout)
        return max(0.0, self.idle_timeout - (time.monotonic() - last))
