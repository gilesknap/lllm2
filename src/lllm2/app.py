import argparse
import fcntl
import json
import secrets
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__, config, downloads
from .backends import create_engine, engine_kind, engine_serves
from .bench import WORKLOADS, Bench
from .catalogue import Catalogue, Finder, local_paths, suitability
from .defaults import starting_defaults
from .discovery import engines, hardware, supported_architectures
from .engine import Cancelled, LocalEngine
from .gpu_tables import GPU_TABLES, gpu_types, pricing_caveat, table_hardware
from .launch import choose_launch, installed_models
from .recommendations import promotion_provenance, saved_qualifications
from .remote import (
    PROVIDERS,
    CallRecords,
    ProbeCache,
    ProviderError,
    RemoteEngine,
    StoreDownloads,
    catalogue_entry,
    catalogue_source,
    companion_names,
    model_users,
    stored_entries,
)
from .settings import (
    DEFAULT_IDLE_TIMEOUT_MINUTES,
    LOCAL_BACKENDS,
    Settings,
    default_key,
)
from .store import Store
from .update_check import UpdateCheck


def measured(result):
    """A completed result with a speed sample or a usable context measurement."""
    if not result or result.get("status") != "complete":
        return False
    context = result.get("largest_observed_context")
    return bool(result.get("samples")) or (type(context) is int and context > 0)


# Display names for remote providers; others show their capitalised name.
PROVIDER_LABELS = {"modal": "Modal"}


class App:
    def __init__(self):
        self.store = Store()
        self.catalogue = Catalogue(self.store)
        self.finder = Finder(self.store)
        self.engines = {"local": LocalEngine()}
        self.engine_lock = threading.Lock()
        self.bench = Bench(self.engines["local"], self.store)
        self.token = secrets.token_urlsafe(32)
        self.start_requests = {}
        self.store_downloads = StoreDownloads()
        self.update_check = UpdateCheck(self.store)

    @property
    def engine(self):
        """Return the current engine: the one that serves or last served a model."""
        return self.bench.engine

    @engine.setter
    def engine(self, engine):
        self.bench.engine = engine

    def engine_for(self, s):
        """Return the engine that serves settings, creating it on first use.

        The panel keeps one engine per backend kind: the local engine and one
        remote engine per provider. Getting an engine does not make it current;
        starting a model or an experiment does, and stops the previous engine.

        Args:
            s: Launch settings.

        Returns:
            The engine for the settings' backend.

        Raises:
            ValueError: The backend names no registered provider.
            RuntimeError: The provider's client library is missing.
        """
        key = engine_kind(s)
        with self.engine_lock:
            engine = self.engines.get(key)
            if engine is None or not engine_serves(engine, s):
                engine = create_engine(s, catalogue=self.catalogue.list)
                self.engines[key] = engine
        return engine

    def remote_engine(self, backend, gpu_type=""):
        """Return the panel's engine for a remote provider.

        Args:
            backend: The provider name.
            gpu_type: A GPU type for a new engine, or empty for the first type
                in the provider's table. An existing engine keeps its type.

        Returns:
            The ``RemoteEngine``.

        Raises:
            ValueError: The backend is local or names no registered provider.
            RuntimeError: The provider's client library is missing.
        """
        if backend in LOCAL_BACKENDS or backend not in PROVIDERS:
            raise ValueError(f"Unknown remote backend: {backend}.")
        table = gpu_types(backend)
        s = Settings.parse({"backend": backend, "gpu_type": gpu_type or table[0].name})
        return self.engine_for(s)

    def remote_backends(self):
        """Describe the remote backends the panel offers, without network use.

        Returns:
            One dict per registered provider with a GPU table: ``name``,
            ``label``, ``caveat``, ``idle_timeout_minutes`` (the default) and
            ``gpus`` with ``name``, ``label``, ``vram_gb``, ``vram_gib`` and
            ``usd_per_hour``.
        """
        return [
            {
                "name": name,
                "label": PROVIDER_LABELS.get(name, name.capitalize()),
                "caveat": pricing_caveat(name),
                "idle_timeout_minutes": DEFAULT_IDLE_TIMEOUT_MINUTES,
                "gpus": [
                    {
                        "name": g.name,
                        "label": g.label,
                        "vram_gb": g.vram_gb,
                        "vram_gib": g.vram_gib,
                        "usd_per_hour": g.usd_per_hour,
                    }
                    for g in table
                ],
            }
            for name, table in GPU_TABLES.items()
            if name in PROVIDERS
        ]

    def selected_hardware(self, data):
        """Describe the hardware of the backend and GPU type the panel selected.

        Args:
            data: A mapping with optional ``backend`` and ``gpu_type``.

        Returns:
            The local ``hardware()`` for a local or incomplete selection,
            otherwise the remote GPU description, which uses a saved probe when
            one exists. No container starts.
        """
        backend, gpu = data.get("backend") or "", data.get("gpu_type") or ""
        if backend in ("", *LOCAL_BACKENDS) or not gpu:
            return hardware()
        try:
            s = Settings.parse({"backend": backend, "gpu_type": gpu})
        except ValueError:
            return hardware()
        try:
            return self.engine_for(s).hardware(s)
        except RuntimeError:
            return table_hardware(backend, gpu)

    def status(self, query):
        """Build the status poll response.

        The panel polls this every few seconds, so it reads only local state
        and the engines' cached status. It never lists remote calls or
        containers; those refresh through ``/api/remote/containers``.

        Args:
            query: The poll's query, with optional ``backend`` and ``gpu_type``.

        Returns:
            The JSON-serialisable status.
        """
        return {
            "token": self.token,
            "version": __version__,
            "update": self.update_check.notice(),
            "engine": self.engine.state(),
            "job": self.bench.snapshot(),
            "hardware": self.selected_hardware(query),
            "downloads": downloads.all_downloads() + self.store_downloads.rows(),
            "endpoint": self.engine.base + "/v1",
            "paths": {
                "models": str(config.MODELS_DIR),
                "engines": [str(p) for p in config.ENGINE_ROOTS],
            },
            "workloads": WORKLOADS,
        }

    def checked_backends(self, data):
        """Return the remote backends that a listing may contact.

        A provider is contacted only when the panel selects it, already runs
        its engine, or this workstation has used it: local call records or a
        saved GPU probe name it. A local session that has never used a
        provider then makes no request to it.

        Args:
            data: The request body, with an optional ``backend``.

        Returns:
            The provider names, in the order ``remote_backends`` lists them.
        """
        checked = {data.get("backend"), *self.engines}
        checked.update(CallRecords(config.STATE_DIR / "remote-calls.json").providers())
        checked.update(ProbeCache(config.STATE_DIR / "remote-probes.json").providers())
        return [b["name"] for b in self.remote_backends() if b["name"] in checked]

    def running_containers(self, data):
        """List every running container in the checked providers' accounts.

        Args:
            data: The request body: ``backend`` (the selected backend or
                empty) and an optional ``check`` list of provider names to
                contact even when ``checked_backends`` leaves them out.

        Returns:
            A dict with ``providers`` (one group per contacted provider with
            ``provider``, ``label``, ``caveat``, ``complete``, ``error`` and
            the ``describe_containers`` rows under ``containers``) and
            ``unchecked`` (``provider`` and ``label`` of each remote backend
            not contacted).

        Raises:
            ValueError: ``check`` is not a list of provider names.
        """
        extra = data.get("check") or []
        if not isinstance(extra, list) or not all(isinstance(n, str) for n in extra):
            raise ValueError("Name the providers to check as a list.")
        checked = set(self.checked_backends(data)) | set(extra)
        groups, unchecked = [], []
        for backend in self.remote_backends():
            name, label = backend["name"], backend["label"]
            if name not in checked:
                unchecked.append({"provider": name, "label": label})
                continue
            group = {
                "provider": name,
                "label": label,
                "caveat": backend["caveat"],
                "complete": False,
                "error": None,
                "containers": [],
            }
            try:
                group.update(self.remote_engine(name).remote_containers())
            except RuntimeError as e:
                group["error"] = str(e)
            groups.append(group)
        return {"providers": groups, "unchecked": unchecked}

    def stop_containers(self, data):
        """Stop listed containers, asking first for work this panel cannot vouch for.

        This session's model stops as **Stop model** does, and an orphan
        stops at once. Another live session's model and a container lllm2
        tracks no serve call for are skipped unless ``confirm`` is True, which
        is allowed for one target at a time.

        Args:
            data: The request body: ``backend``, ``targets`` (dicts with
                ``container_id`` and ``call_id``, either may be None) and
                ``confirm``.

        Returns:
            A dict with ``stopped``, ``skipped`` (with ``reason``) and
            ``failed`` (with ``error``) lists of targets. A container the
            listing does not show is skipped as ended, unless the listing
            failed: then it is failed, as it may still be running.

        Raises:
            ValueError: The request names no valid targets, or confirms more
                than one.
            RuntimeError: The provider could not list its serve calls.
        """
        targets = data.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or not all(
                isinstance(t, dict)
                and (
                    isinstance(t.get("container_id"), str)
                    or isinstance(t.get("call_id"), str)
                )
                for t in targets
            )
        ):
            raise ValueError("Choose the containers to stop.")
        confirm = data.get("confirm") is True
        if confirm and len(targets) > 1:
            raise ValueError(
                "Confirm one container at a time, so another session's model or "
                "unrecognised work is never stopped in bulk."
            )
        engine = self.remote_engine(data["backend"])
        view = engine.remote_containers()
        rows = view["containers"]
        out = {"stopped": [], "skipped": [], "failed": []}
        for target in targets:
            call_id, container_id = target.get("call_id"), target.get("container_id")
            key = {"container_id": container_id, "call_id": call_id}
            row = next(
                (
                    r
                    for r in rows
                    if (
                        r["id"] == call_id
                        if call_id
                        else r["container_id"] == container_id
                    )
                ),
                None,
            )
            if row is None and call_id is None and not view["complete"]:
                # Serve calls are listed apart from containers, so only a
                # container target is lost when the container listing fails.
                # It may still be running and billing.
                out["failed"].append(
                    key
                    | {
                        "error": "Could not check that it is still running, so it "
                        "was not stopped. "
                        + (view["error"] or "The provider cannot list containers.")
                    }
                )
                continue
            if row is None:
                out["skipped"].append(key | {"reason": "It is no longer running."})
                continue
            if row["status"] in ("active", "unknown") and not confirm:
                out["skipped"].append(key | {"reason": unconfirmed_reason(row)})
                continue
            try:
                if row["status"] == "owned":
                    if self.engine is not engine or engine.call_id != row["id"]:
                        raise ValueError(
                            "Another engine in this panel serves that call."
                        )
                    self.action("/api/stop", {})
                else:
                    engine.stop_listed(row)
            except (ValueError, RuntimeError) as e:
                out["failed"].append(key | {"error": str(e)})
                continue
            out["stopped"].append(key)
        return out

    def remote_view(self, backend):
        """Describe a provider's stored models and running calls for the panel.

        Args:
            backend: The provider name.

        Returns:
            A dict with ``provider``, ``caveat``, ``error`` (a user-facing
            message when the client library or credentials are missing, else
            None), ``models`` (``name``, ``size_bytes``, ``catalogue_id``,
            ``display_name`` and ``in_use`` call identifiers), ``stored_ids``
            (catalogue ids whose main file is stored) and ``calls`` (the
            ``describe_calls`` rows).

        Raises:
            ValueError: The backend names no registered provider.
        """
        out = {
            "provider": backend,
            "caveat": pricing_caveat(backend),
            "error": None,
            "models": [],
            "stored_ids": [],
            "calls": [],
        }
        try:
            engine = self.remote_engine(backend)
            rows = engine.remote_calls()
            stored = engine.provider.models()
        except RuntimeError as e:
            out["error"] = str(e)
            return out
        entries = stored_entries(self.catalogue.list())
        names = {m.name for m in stored}
        # A model that is in the store is proof that any earlier download
        # failure for it is over, whichever route placed it there.
        self.store_downloads.settle(backend, names)
        out["stored_ids"] = [e["id"] for name, e in entries.items() if name in names]
        out["calls"] = rows
        out["models"] = [
            {
                "name": m.name,
                "size_bytes": m.size_bytes,
                "catalogue_id": entries[m.name]["id"] if m.name in entries else None,
                "display_name": (
                    entries[m.name].get("display_name") or entries[m.name]["name"]
                )
                if m.name in entries
                else None,
                "in_use": model_users(rows, m.name),
            }
            for m in stored
        ]
        return out

    def catalogue_view(self, data=None):
        host = self.selected_hardware(data or {})
        entries = self.catalogue.list()
        for entry in entries:
            paths = local_paths(entry)
            entry["path"] = str(paths[0]) if paths else None
            entry.update(
                suitability(
                    entry.get("size_bytes") or (entry.get("size_gb") or 0) * 1e9, host
                )
            )
            entry["installed"] = all(p.is_file() for p in paths)
        return entries

    def result_settings(self, result, data):
        settings = Settings.parse(result["settings"])
        if data.get("use_context"):
            field = (
                "recommended_context"
                if data.get("reserve_headroom") is True
                else "largest_observed_context"
            )
            context = result.get(field)
            if type(context) is not int or context <= 0:
                raise ValueError(
                    "This result has no usable context measurement for that choice."
                )
            settings.context = context * settings.slots
        return settings

    def default_key(self, s):
        return default_key(s)

    def action(self, path, data):
        if path == "/api/files":
            requested = data.get("directory")
            directory = Path(requested).expanduser() if requested else config.MODELS_DIR
            if not requested and not directory.is_dir():
                directory = Path.home()
            # Check before resolving: a strict resolve names only the first
            # missing component, not the folder the user typed.
            if not directory.is_dir():
                raise ValueError(f"No folder found at {directory}.")
            directory = directory.resolve(strict=True)
            entries = []
            for child in directory.iterdir():
                try:
                    is_dir = child.is_dir()
                    if is_dir or (child.is_file() and child.suffix.lower() == ".gguf"):
                        entries.append(
                            {
                                "name": child.name,
                                "path": str(child),
                                "directory": is_dir,
                                "size": None if is_dir else child.stat().st_size,
                            }
                        )
                except OSError:
                    continue
            entries.sort(key=lambda e: (not e["directory"], e["name"].casefold()))
            return {
                "directory": str(directory),
                "parent": str(directory.parent),
                "entries": entries,
                "home": str(Path.home()),
                "models": str(config.MODELS_DIR),
            }
        if path == "/api/discover":
            return {
                "models": installed_models(self.catalogue.list()),
                "engines": engines(),
                "catalog": self.catalogue_view(data),
                "backends": self.remote_backends(),
            }
        if path == "/api/launch/select":
            if data.get("backend", "") not in ("", *LOCAL_BACKENDS):
                # A remote backend has no local engine or device to choose.
                # Without a model, serve the top catalogue recommendation.
                model = data.get("model", "")
                ranked = sorted(
                    (e for e in self.catalogue.list() if e.get("recommendation")),
                    key=lambda e: e["recommendation"].get("rank", 999),
                )
                if not model and not ranked:
                    return {"settings": None, "reason": "Choose a catalogue model."}
                s = Settings.parse(
                    {
                        "model": model or str(local_paths(ranked[0])[0]),
                        "backend": data["backend"],
                        "gpu_type": data.get("gpu_type", ""),
                    }
                )
                return starting_defaults(s, **self.engine_for(s).defaults_inputs(s))
            return choose_launch(
                data.get("model", ""),
                data.get("engine", ""),
                data.get("backend", ""),
                data.get("device", ""),
                catalogue=self.catalogue.list(),
            )
        if path == "/api/launch/check":
            s = Settings.parse(data["settings"])
            error = None
            try:
                engine = self.engine_for(s)
                if not engine.hardware(s)["gpus"]:
                    raise ValueError(
                        "No NVIDIA GPU detected. Check GPU availability before starting."
                    )
                engine.launch_args(s)
            except (OSError, ValueError, RuntimeError) as e:
                error = str(e)
            return {
                "valid": error is None,
                "error": error,
                "saved_exists": self.store.get("default", self.default_key(s))
                is not None,
            }
        if path == "/api/results/delete":
            with self.bench.lock:
                if self.bench.active:
                    raise ValueError(
                        "Wait for the current operation to finish before deleting experiment history."
                    )
                deleted = self.store.delete_results(
                    data.get("result_ids"), data.get("failed_only") is True
                )
            return {"deleted": deleted}
        if path == "/api/result/preview":
            r = self.store.get("result", data["result_id"])
            if not measured(r):
                raise ValueError(
                    "Choose a completed result with samples or a context measurement."
                )
            if (
                r.get("measurement_mode") == "warm-conversation"
                or r.get("quality_status") == "failed"
            ):
                raise ValueError(
                    "This result cannot be used as a general measured configuration."
                )
            s = self.result_settings(r, data)
            return {
                "settings": s.dict(),
                "source": "Experiment result · for next launch",
                "notes": [
                    "Review before starting or saving. Saved preferences are unchanged."
                ],
                "evidence": promotion_provenance(
                    r, s, data.get("use_context"), data.get("reserve_headroom") is True
                ),
                "result_id": r["id"],
                "use_context": bool(data.get("use_context")),
                "reserve_headroom": bool(data.get("use_context"))
                and data.get("reserve_headroom") is True,
            }
        if path == "/api/capabilities":
            s = Settings.parse(data["settings"])
            engine = self.engine_for(s)
            return {
                "features": engine.capabilities(s),
                "engine": {
                    k: v
                    for k, v in engine.probe(s).items()
                    if k not in ("help", "environment")
                },
            }
        if path == "/api/default/resolve":
            s = Settings.parse(data["settings"])
            source = data.get("source", "auto")
            if source not in ["auto", "saved", "built-in"]:
                raise ValueError("Unknown defaults source.")
            saved = self.store.get("default", self.default_key(s))
            if saved and source != "built-in":
                resolved = Settings.parse(saved)
                resolved.engine, resolved.device = s.engine, s.device
                provenance = self.store.get("default-evidence", self.default_key(s))
                if not provenance or provenance.get("settings") != saved:
                    return {
                        "settings": resolved.dict(),
                        "source": "Saved defaults · origin unknown",
                        "notes": [
                            "Legacy saved settings preserved; no measurement provenance was recorded."
                        ],
                    }
                notes = [
                    "Saved preferences take precedence over built-in recommendations."
                ]
                if provenance["kind"] == "benchmark":
                    notes.append(provenance["note"])
                    notes.extend(
                        saved_qualifications(
                            provenance,
                            resolved,
                            **self.engine_for(resolved).defaults_inputs(resolved),
                        )
                    )
                    if provenance["context"]["used_headroom_estimate"]:
                        notes.append(
                            "Saved context uses a headroom estimate, not an observed successful allocation."
                        )
                return {
                    "settings": resolved.dict(),
                    "source": "Saved defaults · "
                    + (
                        "historical benchmark evidence · qualified"
                        if provenance["kind"] == "benchmark"
                        else "manual preferences"
                    ),
                    "notes": notes,
                    "evidence": provenance
                    if provenance["kind"] == "benchmark"
                    else None,
                }
            if source == "saved":
                raise ValueError(
                    "No saved settings for this model and backend yet. Load a completed experiment into Launch or edit the draft, then choose “Save my settings”."
                )
            if source == "built-in" and s.remote:
                return starting_defaults(s, **self.engine_for(s).defaults_inputs(s))
            if source == "built-in":
                resolved = choose_launch(model_path=s.model)
                if resolved.get("reason"):
                    resolved.setdefault("notes", []).append(resolved["reason"])
                if not resolved.get("settings"):
                    raise ValueError(
                        resolved.get("reason") or "No recommendation available."
                    )
                return resolved
            return starting_defaults(s, **self.engine_for(s).defaults_inputs(s))
        if path == "/api/default/load":
            s = Settings.parse(data["settings"])
            return self.store.get("default", self.default_key(s))
        if path == "/api/default/save":
            provenance = None
            if data.get("result_id"):
                r = self.store.get("result", data["result_id"])
                if not measured(r):
                    raise ValueError(
                        "Only a completed measured configuration can be promoted."
                    )
                if r.get("measurement_mode") == "warm-conversation":
                    raise ValueError(
                        "Warm conversation results cannot be promoted as a general cold baseline. Save launch preferences manually if desired."
                    )
                if r.get("quality_status") == "failed":
                    raise ValueError(
                        "Source adherence failed; this result cannot be promoted as a measured baseline."
                    )
                s = self.result_settings(r, data)
                provenance = promotion_provenance(
                    r, s, data.get("use_context"), data.get("reserve_headroom") is True
                )
            else:
                s = Settings.parse(data["settings"])
            self.engine_for(s).launch_args(s)
            self.store.put("default", self.default_key(s), s.dict())
            self.store.put(
                "default-evidence",
                self.default_key(s),
                provenance or {"kind": "manual", "settings": s.dict()},
            )
            return s.dict()
        if path == "/api/benchmark":
            s = Settings.parse(data["settings"])
            return self.bench.submit(data, self.engine_for(s))
        if path in ["/api/cancel", "/api/stop"]:
            with self.bench.lock:
                self.bench.cancel.set()
                self.engine.stop()
                if not self.bench.active:
                    self.bench.progress = {"status": "stopped", "kind": "launch"}
            return {"ok": True}
        if path == "/api/start":
            s = Settings.parse(data["settings"])
            request_id = data.get("request_id")
            if request_id is not None and (
                not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
            ):
                raise ValueError("Invalid start request identity.")
            engine = self.engine_for(s)
            engine.launch_args(s)
            entry = catalogue_entry(s.model, self.catalogue.list) if s.remote else None
            if entry is not None and self.store_downloads.active(
                s.backend, catalogue_source(entry).name
            ):
                raise ValueError(
                    "This model is still downloading into the remote store. Wait for the download or cancel it first."
                )
            with self.bench.lock:
                if request_id in self.start_requests:
                    if self.start_requests[request_id] != s.dict():
                        raise ValueError(
                            "Start request identity already used for different settings."
                        )
                    return {"ok": True}
                if self.bench.active:
                    raise ValueError(
                        "Wait for the current operation or cancel it first."
                    )
                running = self.engine.state()
                if running.get("ready") and running["settings"] == s.dict():
                    return {"ok": True}
                if running["running"]:
                    if (
                        data.get("replace_running") is not True
                        or data.get("expected_pid") != running["pid"]
                    ):
                        raise ValueError(
                            "A model is running or has changed. Refresh status and explicitly switch or restart it."
                        )
                self.bench.use(engine)
                self.bench.active = True
                self.bench.cancel.clear()
                if request_id:
                    self.start_requests[request_id] = s.dict()
                    if len(self.start_requests) > 128:
                        del self.start_requests[next(iter(self.start_requests))]
                self.bench.progress = {
                    "kind": "launch",
                    "status": "starting",
                    "phase": "Loading model",
                    "started_at": time.time(),
                    "settings": s.dict(),
                    "request_id": request_id,
                }

            def start():
                try:
                    engine.start(s, self.bench.cancel)
                    self.bench.update(status="serving", phase="Ready")
                except Cancelled:
                    self.bench.update(status="cancelled")
                except Exception as e:
                    self.bench.update(status="failed", error=str(e))
                finally:
                    with self.bench.lock:
                        self.bench.active = False

            threading.Thread(target=start, daemon=True).start()
            return {"ok": True}
        if path == "/api/models/find":
            # Remote engines are not on this workstation; their list is unknown.
            local = data.get("backend", "") in ("", *LOCAL_BACKENDS)
            return self.finder.search(
                data.get("query", ""),
                self.selected_hardware(data),
                data.get("refresh") is True,
                supported_architectures(data.get("engine")) if local else None,
            )
        if path == "/api/catalogue":
            return {"entries": self.catalogue_view(data)}
        if path == "/api/catalogue/add":
            return self.catalogue.add(self.finder.candidate(data["id"]))
        if path == "/api/catalogue/removal-preview":
            entry = self.catalogue.get(data["id"])
            return {
                "name": entry.get("display_name", entry["name"]),
                "paths": [
                    str(p)
                    for path in local_paths(entry)
                    for p in (path, path.with_suffix(path.suffix + ".part"))
                    if p.exists()
                ],
            }
        if path == "/api/catalogue/remove":
            with self.catalogue.lock, self.bench.lock:
                entry = self.catalogue.get(data["id"])
                running = self.engine.state()
                if data.get("delete_weights") is True and self.bench.active:
                    raise ValueError(
                        "Wait for the current operation before deleting weights."
                    )
                settings = running.get("settings") or {}
                protected = (
                    [settings.get("model"), settings.get("drafter")]
                    if running.get("running")
                    else []
                )
                downloads.remove(
                    entry,
                    data.get("delete_weights") is True,
                    protected,
                    [e for e in self.catalogue.list() if e["id"] != entry["id"]],
                )
                self.catalogue.remove(entry["id"])
            return {"ok": True}
        if path == "/api/download":
            with self.catalogue.lock:
                return downloads.start(self.catalogue.get(data["id"])).as_dict()
        if path == "/api/download/cancel":
            return {"cancelled": downloads.cancel(data["id"])}
        if path.startswith("/api/remote"):
            return self.remote_action(path, data)
        raise ValueError("Unknown action")

    def remote_action(self, path, data):
        """Handle the panel's remote backend requests.

        Args:
            path: The request path, starting with ``/api/remote``.
            data: The request body.

        Returns:
            The JSON-serialisable response.

        Raises:
            ValueError: The request is invalid or conflicts with running work.
            RuntimeError: The provider failed.
        """
        if path == "/api/remote":
            return self.remote_view(data["backend"])
        if path == "/api/remote/containers":
            return self.running_containers(data)
        if path == "/api/remote/stop-containers":
            return self.stop_containers(data)
        if path == "/api/remote/orphans":
            found, unavailable = [], []
            for backend in self.checked_backends(data):
                try:
                    rows = self.remote_engine(backend).orphans()
                except RuntimeError as e:
                    unavailable.append({"provider": backend, "error": str(e)})
                    continue
                caveat = pricing_caveat(backend)
                found += [row | {"provider": backend, "caveat": caveat} for row in rows]
            return {"orphans": found, "unavailable": unavailable}
        if path == "/api/remote/stop-call":
            engine = self.remote_engine(data["backend"])
            row = next(
                (r for r in engine.remote_calls() if r["id"] == data["call_id"]), None
            )
            if row is None:
                raise ValueError("That remote call is no longer running.")
            if row["status"] == "owned":
                raise ValueError("This panel serves that call. Use Stop model instead.")
            if row["status"] == "active":
                owner = row["owner"] or {}
                where = (
                    f" (pid {owner.get('pid')} on {owner.get('host')})"
                    if owner
                    else " on another machine or in another container"
                )
                raise ValueError(
                    f"Another lllm2 session{where} serves that call and keeps its heartbeat fresh. Stop it there."
                )
            engine.cancel_orphan(row["id"])
            return {"ok": True}
        if path == "/api/remote/adopt":
            call_id = data["call_id"]
            engine = self.remote_engine(data["backend"])
            with self.bench.lock:
                if self.bench.active:
                    raise ValueError(
                        "Wait for the current operation or cancel it first."
                    )
                if (
                    self.engine.state()["running"]
                    and data.get("replace_running") is not True
                ):
                    raise ValueError(
                        "A model is running. Stop it, or confirm that adopting replaces it."
                    )
                self.bench.use(engine)
                self.bench.active = True
                self.bench.cancel.clear()
                self.bench.progress = {
                    "kind": "launch",
                    "status": "starting",
                    "phase": "Adopting remote call",
                    "started_at": time.time(),
                    "settings": None,
                    "adopting": call_id,
                }

            def adopt():
                try:
                    engine.adopt(call_id, self.bench.cancel)
                    self.bench.update(
                        status="serving",
                        phase="Ready",
                        settings=engine.state()["settings"],
                    )
                except Cancelled:
                    self.bench.update(status="cancelled")
                except Exception as e:
                    self.bench.update(status="failed", error=str(e))
                finally:
                    with self.bench.lock:
                        self.bench.active = False

            threading.Thread(target=adopt, daemon=True).start()
            return {"ok": True}
        if path == "/api/remote/idle":
            engine = self.engine
            if not isinstance(engine, RemoteEngine) or not engine.alive():
                raise ValueError("No remote engine is running.")
            minutes = data.get("minutes")
            engine.set_idle_timeout(None if minutes in ("", None) else minutes)
            return engine.status()
        if path == "/api/remote/download":
            entry = self.catalogue.get(data["id"])
            engine = self.remote_engine(data["backend"])
            with self.bench.lock:
                if (
                    self.bench.active
                    and self.engine is engine
                    and engine.status()["phase"] == "downloading model"
                ):
                    raise ValueError(
                        "A model start is downloading into this store. Wait for it to finish."
                    )
            return self.store_downloads.start(engine.provider, entry)
        if path == "/api/remote/download/cancel":
            return {"cancelled": self.store_downloads.cancel(data["id"])}
        if path == "/api/remote/models/remove":
            backend, name = data["backend"], data["name"]
            engine = self.remote_engine(backend)
            if self.store_downloads.active(backend, name):
                raise ValueError(
                    "Cancel the download and wait for it to stop before removing this model."
                )
            with self.bench.lock:
                if self.bench.active and self.engine is engine:
                    raise ValueError(
                        "Wait for the current operation before removing stored models."
                    )
                users = model_users(engine.remote_calls(), name)
                if users:
                    raise ValueError(
                        f"Serve call {', '.join(users)} uses {name}. Stop it first."
                    )
                entry = stored_entries(self.catalogue.list()).get(name)
                companions = companion_names(entry) if entry else ()
                engine.provider.remove_model(name, companions)
            return self.remote_view(backend)
        raise ValueError("Unknown action")


def unconfirmed_reason(row):
    """Say why a listed container needs confirmation before it stops.

    Args:
        row: A ``describe_containers`` row whose status is ``"active"`` or
            ``"unknown"``.

    Returns:
        A sentence that names what stopping it would cancel.
    """
    if row["status"] == "active":
        owner = row["owner"] or {}
        where = (
            f" (pid {owner.get('pid')} on {owner.get('host')})"
            if owner
            else " on another machine or in another container"
        )
        return (
            f"Another lllm2 session{where} serves call {row['id']}; confirm to stop it."
        )
    if row["function"]:
        return (
            f"Container {row['container_id']} runs an lllm2 {row['function']} call "
            "that this panel does not track; confirm to stop it."
        )
    return (
        f"Container {row['container_id']} ({row['app'] or 'unknown app'}) is not "
        "a serve call lllm2 tracks; confirm to stop it."
    )


def host_allowed(header, allowed, names=frozenset()):
    """Say whether a request's ``Host`` header names this panel.

    The check stops DNS rebinding: a page on another site cannot point its own
    name at this panel and read the answers. It is not authentication.

    Args:
        header: The request's ``Host`` header.
        allowed: The panel's own ``name:port`` forms, in lower case.
        names: Extra host names, from ``LLLM2_PANEL_ALLOWED_HOSTS``, accepted
            with any port or none, as a proxy or Ingress passes them on.

    Returns:
        True when the header is one of ``allowed``, or one of ``names`` with
        or without a port.
    """
    header = header.lower()
    if header in allowed:
        return True
    name, colon, port = header.partition(":")
    return name in names and (not colon or (port.isascii() and port.isdigit()))


def origin_allowed(origin, host):
    """Say whether a POST's ``Origin`` header is this panel's own page.

    A browser behind a TLS proxy or Ingress sends ``https://`` with the same
    host, so both schemes are accepted. The session token is the CSRF defence;
    this check adds a second one.

    Args:
        origin: The request's ``Origin`` header, or None if it has none.
        host: The request's ``Host`` header, already checked.

    Returns:
        True when there is no ``Origin``, or it is ``http://`` or ``https://``
        followed by ``host``.
    """
    if not origin:
        return True
    host = host.lower()
    return origin.lower() in (f"http://{host}", f"https://{host}")


def serve(host="127.0.0.1", port=8082):
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock = (config.STATE_DIR / "panel.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("Another lllm2 panel owns this state directory.") from error
    app = App()
    downloads.configure(app.store)
    hostnames = {
        "127.0.0.1",
        "localhost",
        socket.gethostname().lower(),
        socket.getfqdn().lower(),
    }
    allowed_hosts = {f"{name}:{port}" for name in hostnames}

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, kind="application/json"):
            raw = json.dumps(body).encode() if kind == "application/json" else body
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def valid_host(self):
            # The accepted socket identifies the local interface used by this
            # request, including LAN addresses on multi-interface workstations.
            local_host = f"{self.connection.getsockname()[0]}:{port}"
            if not host_allowed(
                self.headers.get("Host", ""),
                allowed_hosts | {local_host},
                config.PANEL_ALLOWED_HOSTS,
            ):
                self.send(
                    403,
                    {
                        "error": "Use this workstation’s panel address or "
                        "hostname, or add the name to LLLM2_PANEL_ALLOWED_HOSTS."
                    },
                )
                return False
            return True

        def do_GET(self):
            if not self.valid_host():
                return
            path = urlparse(self.path).path
            if path == "/":
                self.send(
                    200,
                    Path(__file__)
                    .with_name("static")
                    .joinpath("index.html")
                    .read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif path == "/static/panel.js":
                self.send(
                    200,
                    Path(__file__)
                    .with_name("static")
                    .joinpath("panel.js")
                    .read_bytes(),
                    "text/javascript; charset=utf-8",
                )
            elif path == "/static/panel.css":
                self.send(
                    200,
                    Path(__file__)
                    .with_name("static")
                    .joinpath("panel.css")
                    .read_bytes(),
                    "text/css; charset=utf-8",
                )
            elif path == "/api/status":
                query = {
                    k: v[-1] for k, v in parse_qs(urlparse(self.path).query).items()
                }
                self.send(200, app.status(query))
            elif path in ["/api/results", "/api/results/export"]:
                rows = app.store.list("summary" if path == "/api/results" else "result")
                self.send(200, rows)
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.valid_host():
                return
            if self.headers.get("X-LLLM2-Token") != app.token or not origin_allowed(
                self.headers.get("Origin"), self.headers.get("Host", "")
            ):
                self.send(403, {"error": "Reload the panel to renew its session."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2_000_000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                self.send(200, app.action(urlparse(self.path).path, data))
            except (ValueError, KeyError, TypeError, OSError, ProviderError) as e:
                # ProviderError carries provider messages, such as missing
                # credentials, that the user can act on. Other RuntimeError
                # exceptions stay server errors.
                self.send(400, {"error": str(e)})
            except Exception as e:
                app.engine.log("Panel error: " + str(e))
                self.send(500, {"error": str(e)})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)

    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"lllm2: http://127.0.0.1:{port}", flush=True)
    if host != "127.0.0.1":
        print(
            f"LAN panel: http://{socket.gethostname() if host == '0.0.0.0' else host}:{port} (listening on {host})",
            flush=True,
        )
        print(
            "LAN access has no login or TLS: anyone who can reach this port can control the workbench. Use only on a trusted network.",
            flush=True,
        )
    if config.PANEL_ALLOWED_HOSTS:
        print(
            "Panel also answers to: " + ", ".join(sorted(config.PANEL_ALLOWED_HOSTS)),
            flush=True,
        )
    try:
        server.serve_forever()
    finally:
        app.bench.cancel.set()
        app.store_downloads.cancel_all()
        for engine in list(app.engines.values()):
            try:
                engine.stop()
            except Exception as e:
                engine.log(f"Shutdown could not stop the engine: {e}")
        server.server_close()
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="lllm2 local LLM workbench")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IPv4 bind address (default: localhost; use 0.0.0.0 for trusted-LAN access without authentication or TLS)",
    )
    args = parser.parse_args(argv)
    serve(args.host, args.port)
