"""The Modal app that runs llama-server for the Modal remote provider.

The app has three functions. ``probe`` inspects the GPU and the engine in a
GPU container. ``download`` fetches a catalogue model from Hugging Face into
the model Volume. ``serve`` runs llama-server, opens an encrypted Modal tunnel
to it and publishes the tunnel address and logs until the call is cancelled.
The tunnel address is public. llama-server requires the per-launch API key on
every path except ``/health``, ``/v1/health`` and its web UI files. Those paths
reveal only that the server is up.

The functions do their work in plain functions (``probe_container``,
``fetch_model`` and ``run_server``) that receive every Modal object they use,
so tests run them without Modal. The module imports without the ``modal``
package; it defines ``app`` only when the package is installed.

Engine selection: the image installs both CUDA engine tarballs that
``lllm2 engine install`` can choose for this lllm2 version's llama.cpp and CUDA
pins, from the same release lookup and with the same checksum and archive
checks, so the engine sha256 matches a local install. Released and development
versions follow the same rule: the newest published release that carries the
pinned tarball. A container then selects a tarball with ``cuda_track``, the
driver rule the local installer uses.

Image layers: the engine layer comes before the lllm2 source layer, and Modal
keys it only on ``install_engines``'s source text and its arguments, the release
pins. A code change therefore rebuilds only the small source layer; the 1.8 GB
of engine tarballs download again only when a pin changes.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__, discovery, engine_install, gguf
from .downloads import (
    CHUNK,
    RETRY_ATTEMPTS,
    RETRY_POLL,
    USER_AGENT,
    retry_delay,
    short_read_message,
    transient,
    url_for,
)
from .engine_release import CUDA_TRACKS, LLAMA_CPP_REF
from .tls import download_context

try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]

APP_NAME = "lllm2"
VOLUME_NAME = "lllm2-models"
STATE_NAME = "lllm2-state"
LOG_QUEUE_NAME = "lllm2-logs"
PYTHON_VERSION = "3.12"

ENGINE_ROOT = "/opt/lllm2/engines"
MODEL_ROOT = "/models"
FILE_ROOT = "/tmp/lllm2-files"
SERVER_PORT = 8080

PROBE_TIMEOUT = 10 * 60
# Hard caps. The owner heartbeat normally ends an abandoned call much sooner.
DOWNLOAD_TIMEOUT = 2 * 60 * 60
SERVE_TIMEOUT = 12 * 60 * 60
HEARTBEAT_SECONDS = 10.0
OWNER_GRACE_SECONDS = 180.0
COMMIT_SECONDS = 60.0
#: Bytes appended to a part file before the download commits the store. A fast
#: link writes gigabytes between two one-minute commits, and every uncommitted
#: byte dies with the worker, so the byte rule bounds the loss on any link. It
#: also bounds the work: a commit costs the Volume the bytes it has to store.
COMMIT_BYTES = 2 * 1024**3
LOG_BATCH = 200

DEPLOYMENT_KEY = "deployment"
OWNER_LOST = "the local lllm2 process stopped sending heartbeats"
DOWNLOAD_CANCELLED = "Download cancelled; the partial file is kept."

INSTALL_MESSAGE = (
    "The Modal backend needs the Modal client. "
    "Install it with: pip install 'lllm2[modal]'"
)


def call_key(call_id: str) -> str:
    """Return the state key of a serve call's owner record."""
    return "call:" + call_id


def tunnel_key(call_id: str) -> str:
    """Return the state key where a serve call publishes its tunnel."""
    return "tunnel:" + call_id


def download_key(call_id: str) -> str:
    """Return the state key where a download call publishes its progress."""
    return "download:" + call_id


def heartbeat_key(call_id: str) -> str:
    """Return the state key where the owning lllm2 process writes heartbeats."""
    return "heartbeat:" + call_id


def meta_key(name: str) -> str:
    """Return the state key of a stored model's cached GGUF metadata."""
    return "meta:" + name


def deployment_version() -> str:
    """Identify the app code that a deployment must match.

    Returns:
        The lllm2 version with a digest of the package files. A released
        package changes only with its version; a development checkout also
        redeploys when its files change.
    """
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(package).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    return f"{__version__}+{digest.hexdigest()[:12]}"


def store_directory(name: str, files: list[str] | tuple[str, ...]) -> PurePosixPath:
    """Return the store directory for a model's files.

    Args:
        name: The store-relative path of the main file.
        files: The repository files, main file first.

    Returns:
        The store-relative directory that holds every file.

    Raises:
        ValueError: The name does not end with the main file, or a path is
            absolute or leaves the store.
    """
    for value in (name, *files):
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError(f"Unsafe model store path: {value}")
    if not files or not name.endswith("/" + files[0]):
        raise ValueError("The model name must end with its main file.")
    return PurePosixPath(name[: -len(files[0]) - 1])


def install_engines(ref: str, tracks: dict[str, str], root: str) -> None:
    """Install every CUDA engine track into the image.

    Modal keys the image layer on this function's source and arguments, so the
    body imports what it needs instead of using module globals. The installer
    comes from the lllm2 source that Modal mounts during the build and keeps
    its checksum and archive checks.

    Args:
        ref: The pinned llama.cpp release, ``LLAMA_CPP_REF``.
        tracks: The pinned CUDA tracks, ``CUDA_TRACKS``.
        root: The engine home inside the image.

    Raises:
        RuntimeError: The pins differ from the mounted installer's pins.
    """
    from pathlib import Path

    from lllm2 import engine_install, engine_release

    if (ref, tracks) != (engine_release.LLAMA_CPP_REF, engine_release.CUDA_TRACKS):
        raise RuntimeError("Engine layer pins differ from the lllm2 source pins.")
    print(f"Installing llama.cpp {ref} engines for CUDA {sorted(tracks.values())}")
    for track in tracks:
        # The build has no GPU driver to start the binary; probe checks it.
        engine_install.install(
            "cuda", track=track, root=Path(root), check_startup=False
        )


def engine_binary(root: str = ENGINE_ROOT) -> str:
    """Return the llama-server path that suits the container's driver.

    Args:
        root: The engine home inside the image.

    Returns:
        The absolute binary path.
    """
    track = engine_install.cuda_track()
    return f"{root}/llama-{LLAMA_CPP_REF}-cuda{CUDA_TRACKS[track]}/llama-server"


def probe_container(binary: str | None = None) -> dict:
    """Inspect the GPU and the engine in this container.

    Args:
        binary: The llama-server path. None selects it with ``engine_binary``.

    Returns:
        A dict with ``name`` and ``total_mib`` of the first GPU and ``engine``,
        the engine record in the ``discovery.probe()`` shape plus the
        ``cuda_track``, ``requested_ref``, ``cuda_graph``, ``cache_kernel`` and
        ``environment`` keys.

    Raises:
        RuntimeError: nvidia-smi failed.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        name, total = (
            part.strip() for part in result.stdout.splitlines()[0].split(",")
        )
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as error:
        raise RuntimeError(f"nvidia-smi could not describe the GPU: {error}") from error
    binary = binary or engine_binary()
    engine = dict(discovery.probe(binary))
    # Name the build the container really ran. The compiler version in the
    # engine's own version text reads like a CUDA version otherwise.
    record = engine_install.provenance(Path(binary))
    engine["cuda_track"] = record.get("cuda_track")
    engine["requested_ref"] = record.get("requested_ref")
    engine["cuda_graph"] = discovery.cuda_graph_support(binary)
    engine["cache_kernel"] = discovery.cache_kernel_support(binary, "CUDA")
    environment = discovery.engine_environment(binary)
    # Report only the variables that affect the engine, not container secrets.
    engine["environment"] = {
        key: value
        for key, value in environment.items()
        if key.startswith(("GGML_", "CUDA_", "NVIDIA_")) or key == "LD_LIBRARY_PATH"
    }
    return {"name": name, "total_mib": int(total), "engine": engine}


def fetch_model(
    name: str,
    repo: str,
    files: list[str],
    revision: str | None,
    progress: Callable[[str, int, int | None, str | None], None],
    commit: Callable[[], None],
    root: str = MODEL_ROOT,
    cancel: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    opener: Callable[..., Any] = urllib.request.urlopen,
    reload: Callable[[], None] | None = None,
) -> dict:
    """Download a model's files into the store, resuming partial files.

    A file downloads to ``<file>.part`` and is renamed once complete and, for
    a GGUF, once its header parses.

    A tens-of-gigabyte transfer over one connection is regularly cut short by
    the Hugging Face CDN, so a dropped, reset or timed-out connection retries
    from the bytes on disk with a ranged request. Only attempts that add no
    bytes count towards ``RETRY_ATTEMPTS``: a transfer that keeps moving
    survives any number of resets.

    A Modal Volume is not a local disk. Committing while the part file is open
    stores almost nothing of it: measured against a live 30 GB download, each
    commit taken with the file open grew the stored file by a single chunk
    while gigabytes sat in the container, and the stored file only caught up
    at the next commit taken with the file closed. The Volume's own rule points
    the same way, since a commit reloads the Volume and a reload fails while
    the container holds a file open. So the part file is closed before every
    commit and reopened for appending afterwards, every ``COMMIT_BYTES`` or
    ``COMMIT_SECONDS``, whichever comes first. A worker that disappears then
    costs one interval of bytes rather than everything since the last retry.

    A container also mounts the Volume as it stood when the container started
    and sees no later commit until it calls ``reload``, while a Volume file is
    last-write-wins. A download that resumes on another worker therefore
    reloads before it measures the part file: resuming from a stale, shorter
    file would commit that short file over the longer one the Volume holds.

    ``progress`` reports the committed size rather than the bytes written, so
    it never claims bytes a re-scheduled worker would not find. The count
    steps at each commit instead of climbing continuously.

    Args:
        name: The store-relative path of the main file.
        repo: The Hugging Face repository.
        files: The repository files, main file first.
        revision: The repository revision, or None for ``main``.
        progress: A callable that receives the file, committed bytes, total
            bytes (None when unknown) and a retry note (None during normal
            progress). It may raise to stop the download.
        commit: A callable that persists the store, so a cancelled download or
            a lost worker can resume.
        root: The store mount point.
        cancel: An object with ``is_set()`` that stops the download, or None.
        sleep: The sleep callable, replaceable in tests.
        opener: The ``urlopen`` callable, replaceable in tests.
        reload: A callable that brings the store's committed state into this
            container, or None when the store needs no refresh.

    Returns:
        The main file's metadata in the ``discovery.metadata()`` shape.

    Raises:
        ValueError: A path is unsafe or the server returned an unexpected range.
        RuntimeError: Every retry of a file failed to add a byte, or the
            download was cancelled. Completed bytes stay for a later resume.
    """
    directory = Path(root) / store_directory(name, files)
    committed = time.monotonic()
    persisted = 0

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise RuntimeError(DOWNLOAD_CANCELLED)

    def close_part(output) -> None:
        """Flush a part file out of every buffer and close it for a commit."""
        if not output.closed:
            output.flush()
            os.fsync(output.fileno())
            output.close()

    def persist(part: Path) -> None:
        """Commit the store and record the size the Volume then holds.

        The part file must be closed first, or the store keeps next to nothing
        of it. The size on disk is then the stored size, because this
        container wrote every byte of it. A failed commit is reported and
        leaves the recorded size where it was, so progress never claims bytes
        that did not reach the store.

        Args:
            part: The part file whose committed size to read.
        """
        nonlocal committed, persisted
        committed = time.monotonic()
        try:
            commit()
        except Exception as error:
            print(f"lllm2: could not commit the model store: {error}", flush=True)
            return
        persisted = part.stat().st_size if part.is_file() else 0

    def refresh(part: Path) -> None:
        """Take the store's committed state and measure the part file.

        Reload before the first byte of a file and never afterwards: every
        file is committed as it completes, so a reload here drops nothing,
        while a reload during a transfer would drop the bytes since the last
        commit. A failed reload stops the download rather than resuming from a
        stale size, because appending to a short file and committing it would
        replace the longer file the store holds.

        Args:
            part: The part file whose committed size to record.
        """
        nonlocal persisted
        if reload is not None:
            reload()
        persisted = part.stat().st_size if part.is_file() else 0

    def attempt(file: str, part: Path, resume: int) -> tuple[int, int | None]:
        """Read a file once. Returns the bytes on disk and the expected total."""
        nonlocal persisted
        request = urllib.request.Request(
            url_for(repo, file, revision or "main"), headers={"User-Agent": USER_AGENT}
        )
        if resume:
            request.add_header("Range", f"bytes={resume}-")
        try:
            response = opener(request, timeout=60, context=download_context())
        except urllib.error.HTTPError as error:
            error.close()
            # 416 means the part file already holds the whole file.
            if not (resume and error.code == 416):
                raise
            return resume, resume
        with response:
            if resume and response.status != 206:
                resume = 0
            if resume and not response.headers.get("Content-Range", "").startswith(
                f"bytes {resume}-"
            ):
                raise ValueError("Download server returned an unexpected range")
            length = response.headers.get("Content-Length") or ""
            total = resume + int(length) if length.isdecimal() else None
            done = resume
            if not resume:
                # The open below discards the part file, so the bytes the
                # Volume holds for it no longer count as progress.
                persisted = 0
            progress(file, persisted, total, None)
            reported = time.monotonic()
            written = 0
            output = part.open("ab" if resume else "wb")
            try:
                while chunk := response.read(CHUNK):
                    check_cancel()
                    output.write(chunk)
                    done += len(chunk)
                    written += len(chunk)
                    now = time.monotonic()
                    if written >= COMMIT_BYTES or now - committed >= COMMIT_SECONDS:
                        close_part(output)
                        persist(part)
                        written = 0
                        output = part.open("ab")
                        progress(file, persisted, total, None)
                        reported = time.monotonic()
                    elif now - reported >= 1:
                        progress(file, persisted, total, None)
                        reported = now
            finally:
                # However this attempt ends, the bytes it read belong in the
                # Volume, and they get there only once the file is closed.
                close_part(output)
                persist(part)
            progress(file, persisted, total, None)
            return done, total

    def back_off(seconds: float, file: str, done: int, total: int | None, note: str):
        """Wait before a retry, checking the cancel flag and the owner.

        The wait runs in short slices rather than one sleep, so a cancel or a
        dead owner ends it promptly instead of after the whole backoff.
        """
        left = seconds
        while True:
            check_cancel()
            # The progress callback carries the owner heartbeat check, so a
            # dead owner ends the wait rather than the backoff outliving it.
            progress(file, done, total, note)
            if left <= 0:
                return
            pause = min(left, RETRY_POLL)
            sleep(pause)
            left -= pause

    for file in files:
        target = directory / file
        part = target.with_name(target.name + ".part")
        # Another worker may have downloaded more of this model, or all of it,
        # since this container started.
        refresh(part)
        if target.is_file():
            size = target.stat().st_size
            progress(file, size, size, None)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        stalled = tries = 0
        known: int | None = None
        while True:
            check_cancel()
            resume = part.stat().st_size if part.is_file() else 0
            tries += 1
            try:
                done, total = attempt(file, part, resume)
            except Exception as error:
                if not transient(error):
                    raise
                # Keep the size from an earlier attempt, so the bar holds its
                # place instead of dropping back to nothing.
                total, reason = known, str(error) or type(error).__name__
            else:
                known = total if total is not None else known
                if total is None or done >= total:
                    break
                reason = "the connection closed early"
            # The attempt closed the part file and committed it as it ended,
            # so the committed size is what another worker would resume from.
            kept = persisted
            stalled = 0 if kept > resume else stalled + 1
            if stalled >= RETRY_ATTEMPTS:
                raise RuntimeError(short_read_message(file, tries, kept))
            note = f"retry {tries} after {reason}"
            print(f"lllm2: {file}: {note}", flush=True)
            back_off(retry_delay(stalled), file, kept, total, note)
        if file.lower().endswith(".gguf"):
            gguf.header(part)
        part.rename(target)
        # Store the finished file before the next one reloads the store, and
        # fail loudly rather than reload a rename the Volume never took: the
        # reload would drop the whole file and the download would repeat it.
        commit()
        committed = time.monotonic()
        persisted = 0
    return discovery.metadata(str(directory / files[0]))


class OwnerWatch:
    """Detect that the lllm2 process owning a call stopped sending heartbeats.

    The owner writes a new value under ``heartbeat_key`` while it drives the
    call. The watch compares values rather than timestamps, so clock skew
    between the owner's machine and the container does not matter. A missing
    key, an unchanged value or a failed read all count as silence.
    """

    def __init__(
        self,
        state: Any,
        call_id: str,
        grace: float = OWNER_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Start watching from now.

        Args:
            state: A dict-like store with ``get``.
            call_id: The call whose heartbeat to watch.
            grace: Seconds of silence after which the owner counts as lost.
            clock: A monotonic clock, replaceable in tests.
        """
        self.state, self.key, self.grace, self.clock = (
            state,
            heartbeat_key(call_id),
            grace,
            clock,
        )
        self._value: Any = None
        self._changed = clock()

    def lost(self) -> bool:
        """Return True when no new heartbeat arrived within the grace period."""
        try:
            value = self.state.get(self.key)
        except Exception as error:
            print(f"lllm2: could not read the owner heartbeat: {error}", flush=True)
            value = None
        now = self.clock()
        if value is not None and value != self._value:
            self._value, self._changed = value, now
        return now - self._changed > self.grace


class LogPump(threading.Thread):
    """Copy a process's output lines to the container log and a log queue.

    Queue failures drop lines rather than stop llama-server, because nobody
    may be reading the queue of an orphaned call.
    """

    def __init__(self, stream, queue, partition, interval=1.0):
        """Create a pump for one process.

        Args:
            stream: The process's text output.
            queue: An object with ``put_many(values, block, *, partition)``.
            partition: The queue partition, the serve call id.
            interval: Seconds between queue writes.
        """
        super().__init__(daemon=True)
        self.stream = stream
        self.queue = queue
        self.partition = partition
        self.interval = interval
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._done = threading.Event()

    def run(self):
        flusher = threading.Thread(target=self._flush_periodically, daemon=True)
        flusher.start()
        for line in self.stream:
            line = line.rstrip("\n")
            print(line, flush=True)
            with self._lock:
                self._lines.append(line)
        self._done.set()
        flusher.join()
        self.flush()

    def flush(self):
        """Send buffered lines to the queue."""
        with self._lock:
            lines, self._lines = self._lines, []
        for start in range(0, len(lines), LOG_BATCH):
            try:
                self.queue.put_many(
                    lines[start : start + LOG_BATCH], False, partition=self.partition
                )
            except Exception as error:
                print(f"lllm2: dropped log lines: {error}", flush=True)

    def _flush_periodically(self):
        while not self._done.wait(self.interval):
            self.flush()


def run_server(
    argv: list[str],
    api_key: str,
    env: dict[str, str],
    files: dict[str, str],
    *,
    call_id: str,
    state: Any,
    logs: Any,
    forward: Callable[[int], Any],
    port: int = SERVER_PORT,
    file_root: str = FILE_ROOT,
    heartbeat: float = HEARTBEAT_SECONDS,
    grace: float = OWNER_GRACE_SECONDS,
) -> dict:
    """Run llama-server behind a tunnel while its owner sends heartbeats.

    llama-server runs until it exits, the call is cancelled or the owner goes
    silent for ``grace`` seconds. The owner check stops billing when the lllm2 process that spawned the call
    crashes, loses its network or its machine shuts down.

    Args:
        argv: The llama-server command line, binary first.
        api_key: The key llama-server requires, passed as ``LLAMA_API_KEY``.
        env: Extra environment variables for llama-server.
        files: Text files to write under ``file_root``, keyed by file name.
        call_id: The serve call id, used for state keys and the log partition.
        state: A dict-like store with ``put`` and ``pop``.
        logs: A queue with ``put_many(values, block, *, partition)``.
        forward: ``modal.forward`` or a replacement: a context manager factory
            whose tunnel has ``tls_socket``.
        port: llama-server's port.
        file_root: The directory for ``files``.
        heartbeat: Seconds between tunnel record refreshes and owner checks.
        grace: Seconds without an owner heartbeat before llama-server stops.

    Returns:
        A dict with ``exit_code``, llama-server's exit status, and ``error``,
        None or the reason the container stopped llama-server.
    """
    owner = OwnerWatch(state, call_id, grace)
    error = None
    directory = Path(file_root)
    directory.mkdir(parents=True, exist_ok=True)
    for file_name, text in files.items():
        (directory / PurePosixPath(file_name).name).write_text(text)
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env={
            **discovery.engine_environment(argv[0]),
            **env,
            "LLAMA_API_KEY": api_key,
        },
    )
    pump = LogPump(process.stdout, logs, call_id)
    pump.start()
    try:
        with forward(port) as tunnel:
            host, tunnel_port = tunnel.tls_socket
            while process.poll() is None:
                state.put(
                    tunnel_key(call_id),
                    {
                        "host": host,
                        "port": tunnel_port,
                        "tls": True,
                        "heartbeat": time.time(),
                    },
                )
                try:
                    process.wait(heartbeat)
                except subprocess.TimeoutExpired:
                    pass
                if process.poll() is None and owner.lost():
                    error = f"{OWNER_LOST} for {grace:g} seconds"
                    print(f"lllm2: {error}; stopping llama-server", flush=True)
                    break
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        pump.join(10)
        if process.stdout is not None:
            process.stdout.close()
        state.pop(tunnel_key(call_id), None)
    return {"exit_code": process.returncode, "error": error}


def run_download(
    name: str,
    repo: str,
    files: list[str],
    revision: str | None,
    *,
    call_id: str,
    state: Any,
    commit: Callable[[], None],
    grace: float = OWNER_GRACE_SECONDS,
    root: str = MODEL_ROOT,
    cancel: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    opener: Callable[..., Any] = urllib.request.urlopen,
    reload: Callable[[], None] | None = None,
) -> dict:
    """Download a model, publish progress and stop if the owner goes silent.

    Args:
        name: The store-relative path of the main file.
        repo: The Hugging Face repository.
        files: The repository files, main file first.
        revision: The repository revision, or None for ``main``.
        call_id: The download call id, used for state keys.
        state: A dict-like store with ``get``, ``put`` and ``pop``.
        commit: A callable that persists the store.
        grace: Seconds without an owner heartbeat before the download stops.
        root: The store mount point.
        cancel: An object with ``is_set()`` that stops the download, or None.
        sleep: The sleep callable, replaceable in tests.
        opener: The ``urlopen`` callable, replaceable in tests.
        reload: A callable that brings the store's committed state into this
            container, or None when the store needs no refresh.

    Returns:
        The main file's metadata in the ``discovery.metadata()`` shape.

    Raises:
        RuntimeError: The owner stopped sending heartbeats, the download was
            cancelled, or every retry of a file added no bytes. Completed
            bytes stay for a later resume.
        ValueError: A path is unsafe or the server returned an unexpected range.
    """
    owner = OwnerWatch(state, call_id, grace)
    key = download_key(call_id)

    def progress(file, done, total, retry=None):
        if owner.lost():
            # Do not commit here: the part file is open. ``fetch_model``
            # closes it and commits it as this error unwinds the download.
            raise RuntimeError(f"Download stopped: {OWNER_LOST}.")
        record = {"file": file, "done_bytes": done, "total_bytes": total}
        if retry:
            # The panel shows this, so a retry does not read as a stalled bar.
            record["retry"] = retry
        state.put(key, record)

    try:
        return fetch_model(
            name,
            repo,
            files,
            revision,
            progress,
            commit,
            root=root,
            cancel=cancel,
            sleep=sleep,
            opener=opener,
            reload=reload,
        )
    finally:
        state.pop(key, None)


def deploy() -> None:
    """Deploy the app to the Modal workspace of the current credentials.

    Raises:
        RuntimeError: The ``modal`` package is not installed.
    """
    if modal is None:
        raise RuntimeError(INSTALL_MESSAGE)
    app.deploy(name=APP_NAME)


if modal is not None:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("ca-certificates")
        .run_function(
            install_engines, args=(LLAMA_CPP_REF, dict(CUDA_TRACKS), ENGINE_ROOT)
        )
        .add_local_python_source(
            "lllm2", copy=True, ignore=["**/__pycache__", "**/__pycache__/**"]
        )
    )

    def _state():
        return modal.Dict.from_name(STATE_NAME, create_if_missing=True)

    @app.function(
        image=image, gpu="T4", timeout=PROBE_TIMEOUT, single_use_containers=True
    )
    def probe() -> dict:
        """Inspect the GPU and engine; the caller picks the GPU type."""
        return probe_container()

    @app.function(
        image=image,
        volumes={MODEL_ROOT: volume},
        timeout=DOWNLOAD_TIMEOUT,
        single_use_containers=True,
    )
    def download(name: str, repo: str, files: list[str], revision: str | None) -> dict:
        """Download a model into the Volume and publish progress."""
        return run_download(
            name,
            repo,
            files,
            revision,
            call_id=modal.current_function_call_id() or "unknown",
            state=_state(),
            commit=volume.commit,
            reload=volume.reload,
        )

    @app.function(
        image=image,
        gpu="T4",
        volumes={MODEL_ROOT: volume},
        timeout=SERVE_TIMEOUT,
        single_use_containers=True,
    )
    def serve(
        argv: list[str], api_key: str, env: dict[str, str], files: dict[str, str]
    ) -> dict:
        """Run llama-server behind a tunnel until cancelled."""
        return run_server(
            argv,
            api_key,
            env,
            files,
            call_id=modal.current_function_call_id() or "unknown",
            state=_state(),
            logs=modal.Queue.from_name(LOG_QUEUE_NAME, create_if_missing=True),
            forward=modal.forward,
        )
