"""Portable measured starting profiles and compact saved-result provenance."""

import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

from .discovery import (
    cache_kernel_support,
    engine_environment,
    hardware,
    identity,
    metadata,
    probe,
)
from .settings import Settings, build_launch_args

PROFILES = json.loads(Path(__file__).with_name("recommendations.json").read_text())
_hash_lock = threading.Lock()


def _stat_key(path):
    s = path.stat()
    return (str(path), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


@lru_cache(maxsize=32)
def _digest(key):
    with open(key[0], "rb") as f:
        value = hashlib.file_digest(f, "sha256").hexdigest()
    if _stat_key(Path(key[0])) != key:
        raise ValueError(
            "File changed while checking recommendation identity; retry after the write completes."
        )
    return value


def fingerprint(path):
    path = Path(path).expanduser().resolve()
    # The lock also prevents concurrent form requests from hashing the same GGUF twice.
    with _hash_lock:
        return _digest(_stat_key(path))


def measured_defaults(selection, host=None, engine=None, meta=None, model=None):
    """Resolve a measured built-in profile for a checkpoint, engine and GPU.

    A profile matches on backend, checkpoint fingerprint and the GPU's name
    and memory. A different engine sha256 qualifies the result rather than
    rejecting it. A remote engine supplies every evidence argument, so remote
    GPU profiles match on probed GPU facts and the catalogue's sha256.

    Args:
        selection: Settings naming the model, engine, backend and device.
        host: Hardware description in the ``hardware()`` shape, or None to
            detect local hardware.
        engine: Engine capability record in the ``probe()`` shape, or None to
            probe the selected binary locally.
        meta: GGUF metadata record in the ``metadata()`` shape, or None to
            read the checkpoint locally. An ``estimated`` record matches on MTP
            only.
        model: Model identity with ``size`` and ``sha256``, or None to
            fingerprint the local checkpoint.

    Returns:
        A pair: the resolved profile dict or None, and fallback qualifications.
    """
    if not selection.model or not (selection.engine or selection.remote):
        return None, []
    path = Path(selection.model).expanduser()
    notes = []
    try:
        size = path.stat().st_size if model is None else model.get("size")
        candidates = [
            r
            for r in PROFILES
            if r["backend"] == selection.backend
            and (size is None or r["model"]["size"] == size)
        ]
        if not candidates:
            return None, ["No measured built-in profile for this checkpoint/backend."]
        m = metadata(str(path)) if meta is None else meta
        keys = ("mtp",) if m.get("estimated") else ("architecture", "mtp")
        candidates = [
            r
            for r in candidates
            if not m["error"] and all(m.get(k) == r["model"][k] for k in keys)
        ]
        cards = (hardware() if host is None else host)["gpus"]
        candidates = [
            r
            for r in candidates
            if len(cards) == 1
            and cards[0]["name"] == r["gpu"]["name"]
            and cards[0]["total_mib"] == r["gpu"]["total_mib"]
        ]
        if not candidates:
            return None, [
                "Measured built-ins require the exact checkpoint and a single tested GPU of the same model and memory; using fallback guidance."
            ]
        digest = fingerprint(path) if model is None else model.get("sha256")
        record = next(
            (r for r in candidates if digest and r["model"]["sha256"] == digest), None
        )
        if not record:
            return None, [
                "Checkpoint fingerprint differs from measured built-ins; using fallback guidance."
            ]
        p = probe(selection.engine) if engine is None else engine
        values = dict(
            record["settings"],
            model=selection.model,
            engine=selection.engine,
            backend=selection.backend,
            device=selection.device,
            gpu_type=selection.gpu_type,
            idle_timeout_minutes=selection.idle_timeout_minutes,
        )
        devices = [d for d in p["devices"] if d.startswith(selection.gpu_backend)]
        if values["device"] not in devices:
            values["device"] = devices[0] if devices else ""
        template = record["template"]
        if template.get("resource"):
            resource = Path(template["resource"])
            if resource.name != str(resource):
                raise ValueError("Invalid portable template resource.")
            values["chat_template"] = str(
                Path(__file__).with_name("templates") / resource
            )
            template_hash = fingerprint(values["chat_template"])
        else:
            values["chat_template"] = ""
            template_hash = hashlib.sha256(m["template"].encode()).hexdigest()
        changed = []
        if cards[0].get("driver") != record["gpu"].get("driver"):
            changed.append("GPU driver")
        if p["sha256"] != record["engine"]["sha256"]:
            changed.append("engine build")
        library = record["engine"].get("adjacent_cuda_library")
        if library and selection.gpu_backend == "CUDA":
            kernel = (
                p["cache_kernel"]
                if "cache_kernel" in p
                else cache_kernel_support(selection.engine, selection.gpu_backend)
            )
            current_library = kernel.get("library") or {}
            if current_library.get("sha256") != library["sha256"]:
                changed.append("adjacent CUDA library")
        if template_hash != template["sha256"]:
            changed.append("chat template")
        settings = Settings.parse(values)
        if settings.gpu_backend == "CUDA":
            env = (
                p["environment"]
                if "environment" in p
                else engine_environment(settings.engine)
            )
            inherited = [
                key
                for key in ("GGML_CUDA_GRAPH_OPT", "GGML_CUDA_DISABLE_GRAPHS")
                if key in env
                and (
                    key != "GGML_CUDA_GRAPH_OPT" or settings.cuda_graph_opt == "default"
                )
            ]
            if inherited:
                changed.append("inherited CUDA execution environment")
                notes.append(
                    "Inherited "
                    + ", ".join(inherited)
                    + " is preserved. Built-in evidence does not verify these inherited values; execution behavior and performance require revalidation."
                )
        # Includes device, metadata, speculative prerequisites and every launch flag.
        if settings.speculation != "none" and "--spec-draft-n-max" not in p["flags"]:
            raise ValueError("Binary cannot reproduce the measured draft length.")
        build_launch_args(settings, 1920, p, m)
        if changed:
            notes.append(
                "Changed "
                + " and ".join(changed)
                + ": settings pass capability checks, but performance and runtime behavior need revalidation."
            )
            if "engine build" in changed and (
                settings.batch_size is None or settings.ubatch_size is None
            ):
                notes.append(
                    "Omitted batch controls use the selected engine defaults, which may differ from the measured build."
                )
        if record.get("summary"):
            notes.append(record["summary"])
        notes.extend(record["limitations"])
        context = record["context"]
        reserve = context.get("minimum_observed_free_gpu_mib")
        if reserve is not None and context.get("validated_recommendation"):
            notes.append(
                f"Tested recommendation: {settings.context:,} total tokens / {settings.slots} slot(s); at least {reserve:,} MiB GPU memory remained free in these probes, including the running desktop. This is an observed reserve, not a searched maximum or a guarantee for other workloads."
            )
        else:
            notes.append(
                f"Tested starting allocation: {settings.context:,} total tokens / {settings.slots} slot(s). This is not a searched maximum or a headroom recommendation."
            )
        source = (
            "Measured built-in recommendation"
            if context.get("validated_recommendation")
            else "Measured built-in baseline"
        )
        return {
            "settings": settings.dict(),
            "source": source
            + f" · {record['backend']} · tested with {record['engine']['build']}"
            + (" · qualified" if changed else ""),
            "notes": notes,
            "evidence": record,
            "qualification": {"changed": changed, "capability_check": "passed"},
        }, []
    except (OSError, ValueError, KeyError) as e:
        return None, [
            "Measured built-in cannot be applied: "
            + str(e)
            + " Using inherited/generic fallback."
        ]


def promotion_provenance(result, settings, use_context, reserve_headroom=False):
    """Keep source evidence without copying raw prompts, logs or GPU traces."""
    return {
        "kind": "benchmark",
        "settings": settings.dict(),
        "result_id": result["id"],
        "started": result.get("started"),
        "engine": result.get("engine"),
        "model": result.get("model"),
        "hardware": result.get("hardware"),
        "options": result.get("options"),
        "measured_settings": result["settings"],
        "template_identity": result.get("template_identity"),
        "batch_settings": result.get("batch_settings"),
        "execution_settings": result.get("execution_settings"),
        "cache_settings": result.get("cache_settings"),
        "samples": [
            {
                k: s.get(k)
                for k in (
                    "workload",
                    "input_tokens",
                    "output_tokens",
                    "requested_output_budget",
                    "output_budget",
                    "wall_seconds",
                    "peak_engine_rss_mib",
                    "context_per_slot",
                    "slots",
                    "prefill_tok_s",
                    "decode_tok_s",
                    "peak_total_gpu_used_mib",
                    "batch_settings",
                    "execution_settings",
                    "speculative_settings",
                    "cache_settings",
                )
            }
            | {
                "adherence": {
                    k: s["adherence"].get(k)
                    for k in (
                        "status",
                        "exact_payload_match",
                        "output_limit_reached",
                        "expected_sha256",
                        "actual_sha256",
                        "thinking_characters",
                        "policy",
                    )
                }
                if s.get("adherence")
                else None
            }
            for s in result["samples"]
        ],
        "context": {
            "allocated_total": result["settings"]["context"],
            "largest_observed_context": result.get("largest_observed_context"),
            "recommended_context": result.get("recommended_context"),
            "search_status": result.get("context_search_status"),
            "used_headroom_estimate": bool(use_context and reserve_headroom),
            "loaded_context": "headroom"
            if use_context and reserve_headroom
            else "tested"
            if use_context
            else "original",
            "loaded_total": settings.context,
        },
        "note": "A completed sample is execution evidence, not a measured gain. Headroom context, if selected, is an estimate.",
    }


def saved_qualifications(
    provenance, settings, host=None, engine=None, meta=None, model=None
):
    """Qualify historical evidence without modifying or rejecting preferences.

    Args:
        provenance: The saved benchmark provenance.
        settings: The saved settings as they would launch now.
        host: Hardware description, or None to detect local hardware.
        engine: Engine capability record, or None to probe locally.
        meta: GGUF metadata record, or None to read the checkpoint locally.
        model: Model identity, or None to read the local file.

    Returns:
        A list of notes.
    """
    notes = []
    try:
        measured = provenance.get("model") or {}
        current = identity(settings.model) if model is None else model
        if measured.get("sha256") and current.get("sha256"):
            notes.append(
                "Checkpoint fingerprint matches the saved result."
                if measured["sha256"] == current["sha256"]
                else "Checkpoint changed since the saved measurement; that result does not validate the current file."
            )
        elif current.get("mtime_ns") is None or measured.get("mtime_ns") is None:
            notes.append(
                "Checkpoint identity is unknown for the saved result or the current model; compatibility is unverified."
            )
        elif not all(k in measured for k in ("size", "mtime_ns")):
            notes.append(
                "Saved result has no checkpoint identity; current checkpoint compatibility is unknown."
            )
        elif any(current[k] != measured[k] for k in ("size", "mtime_ns")):
            notes.append(
                "Checkpoint changed since the saved measurement; that result does not validate the current file."
            )
        else:
            notes.append(
                "Checkpoint size/time match the saved result; legacy results have no full checkpoint fingerprint."
            )
        engine_identity = provenance.get("engine") or {}
        current_engine = probe(settings.engine) if engine is None else engine
        if (
            not engine_identity.get("sha256")
            or current_engine.get("sha256") != engine_identity["sha256"]
        ):
            notes.append(
                "Selected engine differs from the saved measurement or its identity is unknown; runtime and performance need revalidation."
            )
        old_cards = (provenance.get("hardware") or {}).get("gpus", [])
        new_cards = (hardware() if host is None else host)["gpus"]

        def card_keys(cards):
            return [(c.get("name"), c.get("total_mib"), c.get("driver")) for c in cards]

        if not old_cards or card_keys(old_cards) != card_keys(new_cards):
            notes.append(
                "Current GPU/driver differs from the saved measurement or is unknown; performance needs revalidation."
            )
        template = provenance.get("template_identity")
        if not template or not template.get("sha256"):
            notes.append(
                "Saved result did not fingerprint its chat template; current template equivalence is unknown."
            )
        else:
            digest = (
                fingerprint(settings.chat_template)
                if settings.chat_template
                else hashlib.sha256(
                    (metadata(settings.model) if meta is None else meta)[
                        "template"
                    ].encode()
                ).hexdigest()
            )
            if digest != template["sha256"]:
                notes.append(
                    "Chat template changed since the saved measurement; behavior and performance need revalidation."
                )
    except (OSError, ValueError, KeyError) as e:
        notes.append("Saved evidence compatibility could not be checked: " + str(e))
    return notes
