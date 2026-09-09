"""Background model downloads with real progress.

Deliberately a plain HTTP range-resumable GET rather than the HuggingFace
client: it gives exact byte counts for the panel's progress bar, resumes a part
file after a cancel or a crash, and keeps a 15 GB download out of the request
handler that started it.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from . import config
from .catalogue import files, local_paths
from .tls import download_context

CHUNK = 4 * 1024 * 1024
USER_AGENT = "lllm2/0.1 (+https://github.com/gilesknap/lllm2)"


@dataclass
class Download:
    """State of one in-flight or finished download."""

    id: str
    name: str
    repo: str
    file: str
    target: Path
    directory: Path | None = None
    revision: str = "main"
    total: int = 0
    done: int = 0
    state: str = "queued"  # queued | downloading | complete | error | cancelled
    detail: str = ""
    #: Files that belong with the weights -- a multimodal projector. Fetched
    #: after them, into the same directory, under the same cancel flag.
    extras: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def percent(self) -> float:
        return round(100.0 * self.done / self.total, 1) if self.total else 0.0

    @property
    def rate_mib_s(self) -> float:
        elapsed = max(time.time() - self.started, 0.001)
        return round(self.done / elapsed / (1024 * 1024), 1)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "file": self.file,
            "target": str(self.target),
            "state": self.state,
            "detail": self.detail,
            "percent": self.percent,
            "done_gb": round(self.done / 1e9, 2),
            "total_gb": round(self.total / 1e9, 2),
            "rate_mib_s": self.rate_mib_s,
        }


#: Live and recently finished downloads, keyed by catalogue id.
_downloads: dict[str, Download] = {}
_lock = threading.RLock()
_queue: queue.Queue[Download] = queue.Queue()
_worker = None
_store = None


def url_for(repo: str, file: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{quote(repo, safe='/')}/resolve/{quote(revision, safe='')}/{quote(file, safe='/')}"


def _size_of(repo: str, file: str, revision: str = "main") -> int:
    """Content-Length without fetching the body, so the bar knows the whole job."""
    req = urllib.request.Request(url_for(repo, file, revision), method="HEAD")
    req.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(req, timeout=60, context=download_context()) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _fetch(dl: Download, file: str, target: Path, base: int) -> bool:
    """One file, resuming a part file if there is one. ``base`` is bytes already
    finished in this download, so progress runs across the whole job."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    resume = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url_for(dl.repo, file, dl.revision))
    req.add_header("User-Agent", USER_AGENT)
    if resume:
        req.add_header("Range", f"bytes={resume}-")
    with urllib.request.urlopen(req, timeout=60, context=download_context()) as r:
        # Servers may ignore Range; never append a full response to a partial.
        if resume and r.status != 206:
            resume = 0
        if resume and not r.headers.get("Content-Range", "").startswith(
            f"bytes {resume}-"
        ):
            raise ValueError("Download server returned an unexpected range")
        expected = int(r.headers.get("Content-Length") or 0)
        received = 0
        dl.done = base + resume
        mode = "ab" if resume else "wb"
        with part.open(mode) as f:
            while True:
                if dl._cancel.is_set():
                    dl.state = "cancelled"
                    dl.detail = "cancelled; part file kept for resume"
                    return False
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
                dl.done += len(chunk)
        if expected and received != expected:
            raise ValueError("Download incomplete; partial file retained for resume")
    from . import gguf

    gguf.header(part)
    part.rename(target)
    return True


def _run(dl: Download) -> None:
    jobs = [(dl.file, dl.target)] + [
        (e, (dl.directory or dl.target.parent) / e) for e in dl.extras
    ]
    try:
        with _lock:
            if dl._cancel.is_set():
                dl.state = "cancelled"
                return
            dl.state = "downloading"
            dl.started = time.time()
        # Size the whole job up front: a bar that resets when the projector
        # starts reads as a fault rather than as the second of two files.
        dl.total = sum(_size_of(dl.repo, f, dl.revision) for f, _ in jobs)
        occupied = sum(
            target.stat().st_size
            if target.is_file()
            else target.with_suffix(target.suffix + ".part").stat().st_size
            if target.with_suffix(target.suffix + ".part").is_file()
            else 0
            for _, target in jobs
        )
        directory = dl.directory or dl.target.parent
        directory.mkdir(parents=True, exist_ok=True)
        if dl.total and max(0, dl.total - occupied) > shutil.disk_usage(directory).free:
            raise ValueError("Not enough free disk space for this download.")
        base = 0
        for file, target in jobs:
            if dl._cancel.is_set():
                dl.state = "cancelled"
                return
            if target.exists():
                from . import gguf

                gguf.header(target)
                base += target.stat().st_size
                dl.done = base
                continue
            if not _fetch(dl, file, target, base):
                return
            base = dl.done
        dl.state = "complete"
        dl.detail = f"saved to {dl.target.parent}"
    except urllib.error.HTTPError as e:
        dl.state = "error"
        dl.detail = f"HTTP {e.code} for {url_for(dl.repo, dl.file, dl.revision)}"
    except Exception as e:
        dl.state = "error"
        dl.detail = str(e)


def start(entry: dict) -> Download:
    """Begin downloading a catalogue entry. Returns immediately."""
    with _lock:
        existing = _downloads.get(entry["id"])
        if existing and existing.state in {"queued", "downloading"}:
            return existing
        paths = local_paths(entry)
        target = paths[0]
        dl = Download(
            id=entry["id"],
            name=entry["name"],
            repo=entry["repo"],
            file=entry["file"],
            target=target,
            extras=files(entry)[1:],
            revision=entry.get("revision", "main"),
            directory=config.MODELS_DIR / entry["name"],
        )
        if _store:
            _store.put("download-job", entry["id"], {"entry": entry, "state": "queued"})
        _downloads[entry["id"]] = dl
        _queue.put(dl)
        ensure_worker()
    return dl


def configure(store):
    """Restore pending jobs in insertion order, including jobs with no bytes yet."""
    global _store
    _store = store
    for job in reversed(store.list("download-job")):
        if job["state"] in {"queued", "downloading"}:
            start(job["entry"])


def ensure_worker():
    global _worker
    if _worker and _worker.is_alive():
        return

    def work():
        while True:
            dl = _queue.get()
            try:
                _run(dl)
            finally:
                with _lock:
                    if _store and _downloads.get(dl.id) is dl:
                        saved = _store.get("download-job", dl.id)
                        if saved:
                            _store.put(
                                "download-job", dl.id, {**saved, "state": dl.state}
                            )
                _queue.task_done()

    _worker = threading.Thread(target=work, daemon=True)
    _worker.start()


def cancel(model_id: str) -> bool:
    with _lock:
        dl = _downloads.get(model_id)
        if dl and dl.state in {"queued", "downloading"}:
            dl._cancel.set()
            if dl.state == "queued":
                dl.state = "cancelled"
            if _store:
                saved = _store.get("download-job", model_id)
                if saved:
                    _store.put(
                        "download-job", model_id, {**saved, "state": "cancelled"}
                    )
            return True
        return False


def remove(entry, delete_weights=False, protected=(), other_entries=()):
    """Serialize against queue submissions; never remove another model's files."""
    with _lock:
        dl = _downloads.get(entry["id"])
        if dl and dl.state in {"queued", "downloading"}:
            raise ValueError(
                "Cancel the download and wait for it to stop before removing this model."
            )
        paths = local_paths(entry)
        targets = [
            p for path in paths for p in (path, path.with_suffix(path.suffix + ".part"))
        ]
        if delete_weights:
            resolved = {p.resolve() for p in targets}
            if resolved.intersection(
                Path(p).expanduser().resolve() for p in protected if p
            ):
                raise ValueError(
                    "Stop the running model or experiment before deleting its weights."
                )
            shared = {p.resolve() for e in other_entries for p in local_paths(e)}
            if resolved.intersection(shared):
                raise ValueError(
                    "These weights are shared by another catalogue entry. Remove only the catalogue entry."
                )
            # Validate every path before deleting any; never recursively remove directories.
            for target in targets:
                if target.exists() and not target.is_file():
                    raise ValueError("Expected a model file, not a directory.")
            for target in targets:
                target.unlink(missing_ok=True)
        _downloads.pop(entry["id"], None)
        if _store:
            _store.delete("download-job", entry["id"])


def all_downloads() -> list[dict]:
    with _lock:
        return [d.as_dict() for d in _downloads.values()]


def resume_interrupted(catalog_entries: list[dict]) -> list[str]:
    """Restart downloads left half-finished by a panel restart.

    Downloads are threads, so stopping the panel -- which systemd does on every
    upgrade -- abandons them. The part file survives and the range request
    resumes from it, but nothing was restarting them, so an interrupted download
    sat at 83% until someone noticed and pressed the button again.

    Returns the ids resumed, for logging.
    """

    def interrupted(path: Path) -> bool:
        """A part file with bytes in it, for a file that has not since landed."""
        part = path.with_name(path.name + ".part")
        return not path.exists() and part.exists() and part.stat().st_size > 0

    resumed = []
    for entry in catalog_entries:
        d = config.MODELS_DIR / entry["name"]
        wanted = [entry["file"]] + ([entry["mmproj"]] if entry.get("mmproj") else [])
        # A model whose weights are complete but whose projector was never
        # fetched still serves text -- it just does not see -- so that is not a
        # download to start behind someone's back. Only interruptions resume.
        if not any(interrupted(d / f) for f in wanted):
            continue
        start(entry)
        resumed.append(entry["id"])
    return resumed
