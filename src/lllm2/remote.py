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
import contextlib
import dataclasses
import fcntl
import json
import os
import secrets
import socket
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import __version__, config
from .catalogue import files as catalogue_files
from .catalogue import local_paths
from .discovery import CATALOG, EXECUTION_ENV_KEYS
from .discovery import identity as local_identity
from .discovery import metadata as local_metadata
from .engine import Cancelled, Engine, ResourceConflict
from .engine_release import LLAMA_CPP_REF
from .gpu_tables import gpu_type, table_hardware
from .proxy import EngineProxy, Upstream
from .recommendations import PROFILES
from .settings import DEFAULT_IDLE_TIMEOUT_MINUTES, Settings, build_launch_args

DEFAULT_IDLE_TIMEOUT = DEFAULT_IDLE_TIMEOUT_MINUTES * 60
# An owning engine refreshes its call record this often while the call runs.
OWNER_HEARTBEAT_SECONDS = 30
# A call whose owner has not refreshed its record for this long has no owner.
OWNER_STALE_SECONDS = 120
# A record this recent survives a provider list that does not show its call yet.
RECORD_GRACE_SECONDS = 120


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


class ProviderError(RuntimeError):
    """A provider failure with a message the user can act on.

    Examples are a missing client library, missing credentials or a failed
    provider request. The panel reports these as client errors, while other
    ``RuntimeError`` exceptions stay server errors.
    """


class RemoteProvider(abc.ABC):
    """A service that runs llama-server on rented GPUs.

    Every method may block on the network. Methods raise ``ProviderError`` (a
    ``RuntimeError``) with a user-facing message for provider failures, such as
    missing credentials.

    Attributes:
        name: The registry name. It is also the GPU table name and the
            hardware ``source``.
        server_host: The address llama-server binds inside the container.
        server_port: The port llama-server listens on inside the container.
        engine_path: The llama-server path that an unprobed GPU's static
            engine record reports. A probe replaces it with the real path.
    """

    name: str = ""
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    engine_path: str = "llama-server"

    def setup(self) -> str:
        """Check the credentials and prepare the provider account for lllm2.

        Providers that deploy code, such as Modal, deploy it here. The default
        does nothing.

        Returns:
            A version string for the prepared deployment, or an empty string.
        """
        return ""

    def stored_metadata(self, name: str) -> dict | None:
        """Return cached GGUF metadata for a stored model without downloading.

        Args:
            name: The store-relative path.

        Returns:
            The metadata record in the ``discovery.metadata()`` shape, or None
            when the store has no complete copy or the provider keeps no cache.
        """
        return None

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


def _entries(catalogue):
    if catalogue is None:
        return CATALOG
    return catalogue() if callable(catalogue) else catalogue


def catalogue_entry(path, catalogue=None):
    """Find the catalogue entry whose main model file is at a path.

    Args:
        path: A model path, as in ``Settings.model``.
        catalogue: Catalogue entries, a callable that returns them, or None
            for the bundled catalogue.

    Returns:
        The entry, or None when no entry's managed path matches.
    """
    target = Path(path).expanduser().resolve()
    for entry in _entries(catalogue):
        try:
            paths = local_paths(entry)
        except (KeyError, ValueError):
            continue
        if paths and paths[0].resolve() == target:
            return entry
    return None


def catalogue_sources(catalogue=None):
    """Build a ``RemoteEngine`` source mapper that knows the catalogue.

    Args:
        catalogue: Catalogue entries, a callable that returns them, or None
            for the bundled catalogue.

    Returns:
        A callable that maps a model path to ``catalogue_source`` for a
        catalogue model, so a provider can download it, and to
        ``model_source`` for any other model.
    """

    def sources(path):
        entry = catalogue_entry(path, catalogue)
        return catalogue_source(entry) if entry is not None else model_source(path)

    return sources


# The llama-server options that lllm2 reads. The pinned llama.cpp release
# (engine_release.LLAMA_CPP_REF) accepts every one of them.
RELEASE_FLAGS = (
    "--backend-sampling",
    "--batch-size",
    "--cache-ram",
    "--cache-type-k",
    "--cache-type-v",
    "--chat-template-file",
    "--ctx-checkpoints",
    "--ctx-size",
    "--device",
    "--fit",
    "--fit-target",
    "--flash-attn",
    "--gpu-layers",
    "--host",
    "--jinja",
    "--list-devices",
    "--model",
    "--model-draft",
    "--no-context-shift",
    "--parallel",
    "--perf",
    "--port",
    "--reasoning-effort",
    "--spec-draft-device",
    "--spec-draft-n-max",
    "--spec-draft-type-k",
    "--spec-draft-type-v",
    "--spec-ngram-simple-size-m",
    "--spec-ngram-simple-size-n",
    "--spec-type",
    "--ubatch-size",
)
RELEASE_SPEC_TYPES = (
    "none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, "
    "ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache"
)


def release_key():
    """Return the key of the lllm2 build and engine release that a probe describes.

    Returns:
        The lllm2 version and the pinned llama.cpp release.
    """
    return f"{__version__}|{LLAMA_CPP_REF}"


def static_probe(provider, gpu):
    """Describe an unprobed GPU from the GPU table and the pinned engine release.

    No container starts. The GPU figures are the table's nominal ones, the flag
    list is the release's, and evidence that only the binary can give, such as
    its sha256 and the compiled CUDA streams switch, is unknown.

    Args:
        provider: The ``RemoteProvider``.
        gpu: The provider's GPU type string.

    Returns:
        A ``GpuProbe`` whose engine record has ``estimated`` set to True.

    Raises:
        ValueError: If the provider has no table entry for the GPU type.
    """
    (card,) = table_hardware(provider.name, gpu)["gpus"]
    unprobed = f"Not probed yet; the first {provider.name} launch on {gpu} checks it."
    return GpuProbe(
        name=card["name"],
        total_mib=card["total_mib"],
        engine={
            "path": provider.engine_path,
            "version": f"lllm2 CUDA engine release {LLAMA_CPP_REF} (not probed)",
            "flags": list(RELEASE_FLAGS),
            "help": f"--spec-type {RELEASE_SPEC_TYPES}",
            "devices": ["CUDA0"],
            "device_output": "",
            "error": None,
            "sha256": None,
            "cuda_graph": {"supported": False, "library": None, "reason": unprobed},
            "cache_kernel": {
                "mixed_gpu_kernel": "unknown",
                "runtime_dispatch": "unknown; not observed",
                "library": None,
                "reason": "Independent cache types need a runtime check. " + unprobed,
            },
            "environment": {},
            "estimated": True,
        },
    )


class ProbeCache:
    """Provider GPU probes saved on disk, so validation never starts a container.

    Entries key on the provider, the GPU type and ``release_key()``, so a new
    lllm2 version or engine release probes again.
    """

    def __init__(self, path):
        """Use a cache file.

        Args:
            path: The JSON file path. It need not exist.
        """
        self.path = Path(path)
        self._lock = threading.Lock()

    @staticmethod
    def _key(provider, gpu):
        return f"{provider}|{gpu}|{release_key()}"

    def _read(self):
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, provider, gpu):
        """Return the saved probe for a GPU type, if any.

        Args:
            provider: The provider name.
            gpu: The GPU type string.

        Returns:
            The ``GpuProbe``, or None when this release has no saved probe.
        """
        with self._lock:
            entry = self._read().get(self._key(provider, gpu))
        try:
            return GpuProbe(entry["name"], int(entry["total_mib"]), entry["engine"])
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, provider, gpu, probe):
        """Save a probe for a GPU type.

        Args:
            provider: The provider name.
            gpu: The GPU type string.
            probe: The ``GpuProbe``.
        """
        with self._lock:
            data = self._read()
            data[self._key(provider, gpu)] = {
                "name": probe.name,
                "total_mib": probe.total_mib,
                "engine": probe.engine,
                "probed": time.time(),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps(data))
            os.replace(temporary, self.path)


def pid_namespace():
    """Return this process's PID namespace identifier.

    Returns:
        A string such as ``"pid:[4026531836]"``, or None when the system does
        not expose it.
    """
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def owner_record():
    """Describe this process as the owner of a call record.

    Returns:
        A dict with ``pid``, ``host``, ``pid_ns`` and a ``heartbeat`` of now.
    """
    return {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "pid_ns": pid_namespace(),
        "heartbeat": time.time(),
    }


def owner_alive(owner, now=None):
    """Return whether a call record's owner is a running lllm2 session.

    The heartbeat decides. A process ID means something only in the PID
    namespace that recorded it, so the process check applies only when the
    owner shares this host and PID namespace. A CLI in another container then
    trusts a running panel's heartbeat instead of a PID it cannot see.

    Args:
        owner: The record's ``owner`` dict with ``pid``, ``host``, ``pid_ns``
            and ``heartbeat`` (Unix seconds), or None.
        now: The current Unix time, or None to read the clock.

    Returns:
        True when the heartbeat is recent and, where the process check applies,
        the owner process still exists. A record owned by this process returns
        False, because engines in this process are checked directly.
    """
    if not isinstance(owner, dict):
        return False
    now = time.time() if now is None else now
    heartbeat = owner.get("heartbeat")
    if not isinstance(heartbeat, int | float) or now - heartbeat > OWNER_STALE_SECONDS:
        return False
    namespace = owner.get("pid_ns")
    if (
        owner.get("host") == socket.gethostname()
        and namespace is not None
        and namespace == pid_namespace()
    ):
        pid = owner.get("pid")
        if type(pid) is not int or pid <= 0 or pid == os.getpid():
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
    return True


class CallRecords:
    """Owned serve calls saved on disk, so a later session can adopt them.

    A record holds the call's API key, so the file is private to the user.
    Every read-modify-write holds an advisory lock on a sibling ``.lock``
    file, so the panel and the CLI do not overwrite each other's changes.
    """

    def __init__(self, path):
        """Use a record file.

        Args:
            path: The JSON file path. It need not exist.
        """
        self.path = Path(path)
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def _locked(self):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path.with_name(self.path.name + ".lock"), "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                yield

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
        with self._locked():
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
        with self._locked():
            return self._read().get(provider, {}).get(call_id)

    def keep(self, provider, call_ids, grace=RECORD_GRACE_SECONDS):
        """Delete the records of calls that are no longer running.

        A provider list can lag behind a new call, so a record saved or
        refreshed by its owner within the grace period stays.

        Args:
            provider: The provider name.
            call_ids: The identifiers of the running calls.
            grace: Seconds for which a saved or refreshed record stays.
        """
        now = time.time()
        running = set(call_ids)

        def recent(record):
            if not isinstance(record, dict):
                return False
            owner = record.get("owner")
            times = [
                record.get("saved"),
                owner.get("heartbeat") if isinstance(owner, dict) else None,
            ]
            return any(isinstance(t, int | float) and now - t < grace for t in times)

        with self._locked():
            data = self._read()
            calls = data.get(provider, {})
            stale = {k for k, v in calls.items() if k not in running and not recent(v)}
            if stale:
                data[provider] = {k: v for k, v in calls.items() if k not in stale}
                self._write(data)

    def touch(self, provider, call_id, heartbeat):
        """Refresh the owner heartbeat of a call's record, if it exists.

        Args:
            provider: The provider name.
            call_id: The call identifier.
            heartbeat: The heartbeat time in Unix seconds.
        """
        with self._locked():
            data = self._read()
            record = data.get(provider, {}).get(call_id)
            if isinstance(record, dict) and isinstance(record.get("owner"), dict):
                record["owner"]["heartbeat"] = heartbeat
                self._write(data)

    def claim(self, provider, call_id):
        """Make this process the owner of a call's record, unless a session owns it.

        The check and the write happen under the record file lock, so two
        sessions that adopt the same call at once cannot both succeed.

        Args:
            provider: The provider name.
            call_id: The call identifier.

        Returns:
            The record, with ``owner`` set to this process.

        Raises:
            ValueError: No record holds the call's API key, or another running
                lllm2 session owns the call.
        """
        with self._locked():
            data = self._read()
            record = data.get(provider, {}).get(call_id)
            if not isinstance(record, dict) or "api_key" not in record:
                raise ValueError(
                    "No saved key for this remote call, so it cannot be adopted. Stop it instead."
                )
            owner = record.get("owner")
            if owner_alive(owner):
                raise ValueError(
                    f"Another lllm2 session (pid {owner.get('pid')} on {owner.get('host')}) serves that call. Stop it there."
                )
            record["owner"] = owner_record()
            self._write(data)
            return record

    def remove(self, provider, call_id):
        """Delete a call's record, if any.

        Args:
            provider: The provider name.
            call_id: The call identifier.
        """
        with self._locked():
            data = self._read()
            if call_id in data.get(provider, {}):
                del data[provider][call_id]
                self._write(data)


def model_users(rows, name):
    """Return the listed calls that serve a stored model.

    Args:
        rows: Rows from ``describe_calls``.
        name: The store-relative model path.

    Returns:
        The identifiers of calls whose settings model ends with the name.
    """
    suffix = "/" + name.lstrip("/")
    return [
        row["id"]
        for row in rows
        if row["model"] and Path(row["model"]).as_posix().endswith(suffix)
    ]


def describe_calls(provider, records=None, owned=()):
    """List the provider's running lllm2 serve calls with their ownership.

    Also deletes saved records of calls that have ended, after a grace period.

    Args:
        provider: The ``RemoteProvider``.
        records: The ``CallRecords``, or None for the state directory file.
        owned: The identifiers of calls that engines in this process own.

    Returns:
        A list of dicts, oldest first, with ``id``, ``gpu``, ``started`` (Unix
        seconds or None), ``elapsed_seconds``, ``usd_per_hour``,
        ``estimated_cost_usd``, ``model`` (the settings model path or None),
        ``adoptable`` (True when a saved key allows adoption), ``owner`` (the
        owner's ``pid``, ``host`` and ``heartbeat``, or None) and ``status``:
        ``"owned"`` for this process, ``"active"`` for another running lllm2
        session, or ``"orphan"``.
    """
    if records is None:
        records = CallRecords(config.STATE_DIR / "remote-calls.json")
    running = provider.calls()
    records.keep(provider.name, [c.id for c in running])
    now = time.time()
    rows = []
    for call in running:
        record = records.get(provider.name, call.id) or {}
        owner = record.get("owner") if isinstance(record.get("owner"), dict) else None
        if call.id in owned:
            status = "owned"
        elif owner_alive(owner, now):
            status = "active"
        else:
            status = "orphan"
        gpu = call.gpu or record.get("gpu")
        started = call.started if call.started is not None else record.get("started")
        elapsed = max(0.0, now - started) if started is not None else None
        try:
            price = gpu_type(provider.name, gpu).usd_per_hour if gpu else None
        except ValueError:
            price = None
        rows.append(
            {
                "id": call.id,
                "gpu": gpu,
                "started": call.started,
                "elapsed_seconds": elapsed,
                "usd_per_hour": price,
                "estimated_cost_usd": elapsed * price / 3600
                if elapsed is not None and price is not None
                else None,
                "model": (record.get("settings") or {}).get("model"),
                "adoptable": "api_key" in record,
                "owner": {k: owner.get(k) for k in ("pid", "host", "heartbeat")}
                if owner
                else None,
                "status": status,
            }
        )
    return rows


class StoreDownloads:
    """Catalogue downloads into remote provider model stores.

    Each download runs ``RemoteProvider.ensure_model`` in a thread. Rows have
    the shape of local downloads (``downloads.Download.as_dict``), plus
    ``store`` (the provider name) and ``catalogue_id``, so the panel shows both
    in one downloads area. No weights pass through this machine.
    """

    def __init__(self):
        """Create an empty download list."""
        self._jobs = {}
        self._lock = threading.Lock()

    def start(self, provider, entry):
        """Start downloading a catalogue entry into a provider's store.

        Args:
            provider: The ``RemoteProvider``.
            entry: The catalogue entry.

        Returns:
            The download row. A download of the same entry that is still
            running is returned unchanged.

        Raises:
            ValueError: The entry has an unsafe path.
        """
        source = catalogue_source(entry)
        job_id = f"{provider.name}:{entry['id']}"
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job["state"] in ("queued", "downloading"):
                return dict(job)
            job = {
                "id": job_id,
                "name": entry.get("display_name") or entry["name"],
                "file": source.files[0],
                "target": f"{provider.name} store: {source.name}",
                "state": "downloading",
                "detail": f"Checking the {provider.name} store",
                "percent": 0,
                "done_gb": 0,
                "total_gb": 0,
                "rate_mib_s": 0,
                "store": provider.name,
                "catalogue_id": entry["id"],
                "store_name": source.name,
            }
            cancel = threading.Event()
            self._jobs[job_id] = job | {"_cancel": cancel}
        threading.Thread(
            target=self._run, args=(provider, source, job_id, cancel), daemon=True
        ).start()
        return dict(job)

    def _update(self, job_id, **values):
        with self._lock:
            self._jobs[job_id].update(values)

    def _run(self, provider, source, job_id, cancel):
        sample = [time.monotonic(), 0]

        def progress(update):
            now = time.monotonic()
            rate = 0.0
            if now > sample[0] and update.done_bytes >= sample[1]:
                rate = (update.done_bytes - sample[1]) / (now - sample[0]) / 2**20
            sample[:] = [now, update.done_bytes]
            total = update.total_bytes or 0
            self._update(
                job_id,
                file=update.file,
                detail=f"Downloading {update.file} inside {provider.name}",
                done_gb=round(update.done_bytes / 1e9, 2),
                total_gb=round(total / 1e9, 2),
                percent=round(100 * update.done_bytes / total, 1) if total else 0,
                rate_mib_s=round(rate, 1),
            )

        try:
            provider.ensure_model(source, progress, cancel)
        except Cancelled:
            self._update(job_id, state="cancelled", detail="Cancelled", rate_mib_s=0)
        except Exception as e:
            self._update(job_id, state="error", detail=str(e), rate_mib_s=0)
        else:
            self._update(
                job_id,
                state="complete",
                percent=100,
                rate_mib_s=0,
                detail=f"Stored in {provider.name}",
            )

    def cancel(self, job_id):
        """Cancel a running download. Partial files stay so a retry resumes.

        Args:
            job_id: The row ``id``.

        Returns:
            True when a running download was asked to stop.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] not in ("queued", "downloading"):
                return False
            job["_cancel"].set()
            return True

    def cancel_all(self):
        """Cancel every running download."""
        for row in self.rows():
            self.cancel(row["id"])

    def active(self, provider, store_name):
        """Return whether a download of a stored model is running.

        Args:
            provider: The provider name.
            store_name: The store-relative model path.

        Returns:
            True while such a download runs.
        """
        return any(
            row["store"] == provider
            and row["store_name"] == store_name
            and row["state"] in ("queued", "downloading")
            for row in self.rows()
        )

    def rows(self):
        """Return every download row, oldest first."""
        with self._lock:
            return [
                {k: v for k, v in job.items() if not k.startswith("_")}
                for job in self._jobs.values()
            ]


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
        self.heartbeat = time.monotonic()


_ENGINES: "weakref.WeakSet[RemoteEngine]" = weakref.WeakSet()
# Serialises adoption, so two engines in this process cannot adopt one call.
_ADOPT_LOCK = threading.Lock()


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
        sources=None,
        catalogue=None,
        probes=None,
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
                ``ModelSource``. None uses ``catalogue_sources(catalogue)``.
            catalogue: Catalogue entries, a callable that returns them, or None
                for the bundled catalogue. The engine uses it to find download
                sources and estimated metadata for models not yet stored.
            probes: The path of the GPU probe cache file. None uses the state
                directory.
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
        self._catalogue = catalogue
        self._sources = sources or catalogue_sources(catalogue)
        self._probe_cache = ProbeCache(
            config.STATE_DIR / "remote-probes.json" if probes is None else probes
        )
        self._probes = {}
        self._meta = {}
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

    def hardware(self, s=None):
        """Describe the remote GPU for starting defaults and results.

        Args:
            s: Settings whose GPU type to describe, or None for the engine's
                current GPU type.

        Returns:
            A description in the ``discovery.hardware()`` shape with ``source``
            set to the provider name and ``gpu_type`` set to the GPU type.
            Probed GPU name and memory from a launch, in this or an earlier
            session of the same release, replace the table figures. No
            container starts.

        Raises:
            ValueError: If the provider has no table entry for the GPU type.
        """
        gpu = self._gpu_for(s)
        description = table_hardware(self.provider.name, gpu)
        probed = self._known(gpu)
        description["gpus"][0].update(name=probed.name, total_mib=probed.total_mib)
        return description

    def probe(self, s):
        """Return the engine capability record for the settings' GPU type.

        No container starts. The record comes from the last launch probe of
        this release, or from ``static_probe`` for a GPU type that has not
        launched yet; that record has ``estimated`` set to True.

        Args:
            s: Settings whose GPU type to describe.

        Returns:
            The record in the ``discovery.probe()`` shape.

        Raises:
            ValueError: The settings name a different remote provider, or the
                GPU type is not in the provider's table.
        """
        self._check(s)
        return self._known(self._gpu_for(s)).engine

    def prepare(self, s):
        """Probe the settings' GPU type unless this release already has a probe.

        A run calls this before it records engine and hardware identity. For
        a GPU type that has never launched, the provider's probe starts a
        container.

        Args:
            s: Settings whose GPU type to probe.

        Raises:
            ValueError: The settings name a different remote provider.
            RuntimeError: The provider failed.
        """
        self._check(s)
        self._probe(self._gpu_for(s))

    def metadata(self, path):
        """Return GGUF metadata for a model without downloading it.

        The engine prefers, in order: metadata from this engine's last
        download, the provider's cached metadata, a local copy of the file, and
        an estimate from the catalogue entry marked ``estimated``.

        Args:
            path: A model path under the managed model directory.

        Returns:
            A record in the ``discovery.metadata()`` shape. ``error`` explains
            why a model with no stored copy and no catalogue entry cannot run.

        Raises:
            ValueError: The path is outside the managed model directory.
        """
        source = self._sources(path)
        if source.name in self._meta:
            return self._meta[source.name]
        stored = self.provider.stored_metadata(source.name)
        if stored is not None:
            self._meta[source.name] = stored
            return stored
        if Path(path).expanduser().is_file():
            return local_metadata(path)
        entry = catalogue_entry(path, self._catalogue)
        if entry is not None:
            return {
                "architecture": None,
                "name": entry.get("name"),
                "context": entry.get("max_ctx"),
                "mtp": bool(entry.get("mtp")),
                "template": "",
                "error": None,
                "estimated": True,
            }
        return {
            "architecture": None,
            "context": None,
            "mtp": None,
            "template": "",
            "error": f"The model is not in the {self.provider.name} store and has no catalogue entry to download it from.",
        }

    def identity(self, path):
        """Identify a model by its store name and catalogue identity.

        Args:
            path: A model path under the managed model directory.

        Returns:
            A dict with ``path``, ``size`` and ``mtime_ns`` (from a local copy,
            or None), ``store`` (the provider name), ``store_name``, ``repo``
            and ``revision``. A catalogue model adds ``catalogue_id``, and one
            with a measured profile adds the profile's ``sha256`` with
            ``sha256_source`` set to ``"catalogue"``.

        Raises:
            ValueError: The path is outside the managed model directory.
        """
        source = self._sources(path)
        target = Path(path).expanduser().resolve()
        out = {
            "path": str(target),
            "size": None,
            "mtime_ns": None,
            "store": self.provider.name,
            "store_name": source.name,
            "repo": source.repo,
            "revision": source.revision,
        }
        if target.is_file():
            out.update({k: v for k, v in local_identity(target).items() if k != "path"})
        entry = catalogue_entry(path, self._catalogue)
        if entry is None:
            return out
        out["catalogue_id"] = entry.get("id")
        profile_id = (entry.get("recommendation") or {}).get("profile")
        profile = next(
            (
                r
                for r in PROFILES
                if r["id"] == profile_id and r["model"]["file"] == entry["file"]
            ),
            None,
        )
        if profile is not None and out["size"] in (None, profile["model"]["size"]):
            out.update(
                size=profile["model"]["size"],
                sha256=profile["model"]["sha256"],
                sha256_source="catalogue",
            )
        return out

    def launch_args(self, s):
        """Build the remote command line without probing, downloading or spawning.

        Args:
            s: The launch settings.

        Returns:
            The argument list, starting with the binary path in the container.

        Raises:
            ValueError: The remote engine or model cannot run these settings.
        """
        self._check(s)
        probed = self._known(self._gpu_for(s))
        meta = self.metadata(s.model)
        if s.speculation == "draft-dflash" and s.drafter:
            meta = {**meta, "drafter": self.metadata(s.drafter)}
        return self._command(s, probed, meta)[0]

    def defaults_inputs(self, s):
        # Cached or static evidence only: resolving defaults never starts a GPU.
        return {
            "host": self.hardware(s),
            "engine": self.probe(s),
            "meta": self.metadata(s.model),
            "model": self.identity(s.model),
        }

    def remote_calls(self):
        """List every running lllm2 call of the provider with its ownership.

        Returns:
            The rows of ``describe_calls``.
        """
        owned = {engine.call_id for engine in list(_ENGINES)} - {None}
        return describe_calls(self.provider, self._records, owned)

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
        self._check(s)
        self.stop()
        with self.guard:
            self._generation += 1
            generation = self._generation
            self._error = self._download = None
            if s.remote:
                self.gpu = s.gpu_type or self.gpu
                self.idle_timeout = (
                    s.idle_timeout_minutes * 60 if s.idle_timeout_minutes else None
                )
        proxy = self._bind_proxy()
        call = None

        def check():
            if cancel.is_set() or self._generation != generation:
                raise Cancelled()

        try:
            self._phase = "probing"
            probed = self._probe(self.gpu)
            check()
            self._phase = "downloading model"
            meta = self._ensure(s.model, cancel)
            if s.speculation == "draft-dflash" and s.drafter:
                meta = {**meta, "drafter": self._ensure(s.drafter, cancel)}
            check()
            argv, files = self._command(s, probed, meta)
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
        if record is None or "api_key" not in record:
            raise ValueError(
                "No saved key for this remote call, so it cannot be adopted. Stop it instead."
            )
        owner = record.get("owner")
        if call_id != self.call_id and owner_alive(owner):
            raise ValueError(
                f"Another lllm2 session (pid {owner.get('pid')} on {owner.get('host')}) serves that call. Stop it there."
            )
        s = Settings.parse(record["settings"])
        self.stop()
        with self.guard:
            self._generation += 1
            self._error = self._download = None
        proxy = self._bind_proxy()
        with _ADOPT_LOCK:
            try:
                if any(e.call_id == call_id for e in list(_ENGINES) if e is not self):
                    raise ValueError("This lllm2 session already serves that call.")
                record = self._records.claim(self.provider.name, call_id)
                s = Settings.parse(record["settings"])
            except BaseException:
                proxy.close()
                raise
            with self.guard:
                self.gpu = record["gpu"]
                self.argv, self.settings = record["argv"], s
                self.idle_timeout = (
                    s.idle_timeout_minutes * 60 if s.idle_timeout_minutes else None
                )
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

    def set_idle_timeout(self, minutes):
        """Change the idle timeout, including for a call that is serving now.

        The idle countdown restarts from the last request, so a shorter timeout
        can stop an idle call at the next status check.

        Args:
            minutes: Minutes without requests before the engine stops its call,
                or None or 0 to disable the idle timer.

        Raises:
            ValueError: The value is not blank or an integer in 0..1440.
        """
        if minutes is not None and (
            type(minutes) is not int or not 0 <= minutes <= 1440
        ):
            raise ValueError(
                "The idle timeout must be blank or whole minutes in 0..1440; 0 or blank disables it."
            )
        with self.guard:
            self.idle_timeout = minutes * 60 if minutes else None
            if self.settings is not None:
                self.settings = dataclasses.replace(
                    self.settings, idle_timeout_minutes=minutes
                )
            self.log(
                f"Idle timeout set to {minutes} minutes."
                if minutes
                else "Idle timeout disabled; the remote engine runs until stopped."
            )

    def orphans(self):
        """List running serve calls that no running lllm2 session owns.

        Calls that an engine in this process owns, and calls that another
        running session keeps refreshing, are left out. Also deletes saved
        records of calls that have ended, after a grace period.

        Returns:
            The ``describe_calls`` rows whose ``status`` is ``"orphan"``.
        """
        return [row for row in self.remote_calls() if row["status"] == "orphan"]

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

    def _gpu_for(self, s):
        if s is not None and s.remote and s.gpu_type:
            return s.gpu_type
        return self.gpu

    def _check(self, s):
        if s.remote and s.backend != self.provider.name:
            raise ValueError(
                f"These settings use the {s.backend} backend, but this engine serves {self.provider.name}."
            )

    def _known(self, gpu):
        """Return the best probe for a GPU type without starting a container."""
        probed = self._probes.get(gpu)
        if probed is None:
            probed = self._probe_cache.get(self.provider.name, gpu)
        return probed if probed is not None else static_probe(self.provider, gpu)

    def _probe(self, gpu):
        """Return a real probe for a GPU type, probing once per release."""
        if gpu not in self._probes:
            probed = self._probe_cache.get(self.provider.name, gpu)
            if probed is None:
                probed = self.provider.probe(gpu)
                try:
                    self._probe_cache.put(self.provider.name, gpu, probed)
                except OSError as e:
                    self.log(f"Could not save the GPU probe: {e}")
            self._probes[gpu] = probed
        return self._probes[gpu]

    def _command(self, s, probed, meta):
        paths = {s.model: self.provider.model_path(self._sources(s.model).name)}
        if s.speculation == "draft-dflash" and s.drafter:
            paths[s.drafter] = self.provider.model_path(self._sources(s.drafter).name)
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
        return argv, files

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
        self._meta[source.name] = meta
        return meta

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
                "saved": time.time(),
                "owner": owner_record(),
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
            self._heartbeat(call)
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

    def _heartbeat(self, call):
        now = time.monotonic()
        if now - call.heartbeat < OWNER_HEARTBEAT_SECONDS:
            return
        call.heartbeat = now
        try:
            self._records.touch(self.provider.name, call.id, time.time())
        except OSError as e:
            self.log(f"Could not refresh the remote call record: {e}")

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
            raise ProviderError(message + ". Stop it from the provider's tools.") from e
        self._records.remove(self.provider.name, call.id)
        self.log(f"Stopped remote call {call.id}")

    def _idle_remaining(self, call):
        if call is None or not self.ready or not self.idle_timeout:
            return None
        active, last = call.proxy.activity()
        if active:
            return float(self.idle_timeout)
        return max(0.0, self.idle_timeout - (time.monotonic() - last))
