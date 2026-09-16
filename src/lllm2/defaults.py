"""Starting settings adapted from lllm3090 e56c8d0 (Apache-2.0).

The cache and MTP choices are inherited tuning; context is the old calibrated
planner's estimate on the detected hardware, not a new measured maximum.

The planner budgets from what the checkpoint says about itself wherever the
bundled catalogue does not list it, which is any quantisation of any model,
local or remote. Where nothing can be computed the settings say so in a note
rather than keeping a silent 4096.
"""

from pathlib import Path

from .discovery import CATALOG, command, hardware, metadata, probe
from .recommendations import measured_defaults
from .settings import MTP_MODES, Settings, capabilities

# Built-in tuning was previously benchmarked on an RTX 3090.
SOURCE = "Inherited defaults · estimates"

# The context to start with when nothing can be budgeted: large enough for a
# working session, small enough that an engine which fits placement
# automatically can still load. Also the fallback when the full-GPU estimate
# does not fit.
UNCALIBRATED_CONTEXT = 32768

# The planner's own hard ceiling, applied when nothing declares a smaller one.
MAX_CONTEXT = 1048576


def catalogue_entry(path):
    p = Path(path).expanduser()
    candidates = [m for m in CATALOG if m["file"] == p.name]
    return next(
        (m for m in candidates if m["name"] == p.parent.name),
        candidates[0] if len(candidates) == 1 else None,
    )


def planner_evidence(entry, meta, model=None):
    """Merge what the checkpoint says about itself with what the catalogue says.

    The checkpoint supplies its own size, which is the field the catalogue
    cannot generalise: it lists one quantisation of a model by exact file name,
    so a Q8_0 build of a catalogued model, or a model nobody catalogued, was
    unplannable and fell back to the dataclass default of 4096 tokens.

    The catalogue wins the KV cost and the layer count wherever it has an entry,
    which is wherever the file name matches one exactly. Those figures are
    hand-checked against the engine's actual allocation, and a header alone
    cannot be: it declares full-size attention for layers a sliding window
    keeps small, which would halve the planned context of a catalogued model
    the panel plans correctly today. The header answers for everything else,
    which is every checkpoint the catalogue has never seen.

    The KV cost and the layer count are taken from one source together. They are
    two halves of the same shape, and mixing a declared cost with a header layer
    count would price the draft cache against the wrong divisor.

    Args:
        entry: The catalogue entry for this checkpoint, or None.
        meta: GGUF metadata in the ``metadata()`` shape, or None.
        model: Model identity with ``size``, as ``Engine.identity`` returns it,
            or None.

    Returns:
        A pair: a ``context_plan`` entry, and the names of the fields no source
        could supply. The entry is only usable when that list is empty.
    """
    entry, meta, model = entry or {}, meta or {}, model or {}
    size = meta.get("size") or model.get("size")
    kv = entry.get("kv_kib_per_token")
    layers = entry.get("full_attention_layers")
    if not kv:
        kv, layers = meta.get("kv_kib_per_token"), meta.get("full_attention_layers")
    found = {
        "size_gb": size / 1e9 if size else entry.get("size_gb"),
        "kv_kib_per_token": kv,
        "full_attention_layers": layers,
        "mmproj": entry.get("mmproj"),
        "mmproj_gb": entry.get("mmproj_gb", 0),
    }
    missing = [
        name
        for name in ("size_gb", "kv_kib_per_token")
        if not found[name] or found[name] <= 0
    ]
    return found, missing


def planner_ceiling(entry, meta):
    """Return the largest context any source allows for this checkpoint.

    A GGUF header declares its context however the converter wrote it, so a
    string, a float or an array marker can arrive here. Only a positive whole
    number is a context, and ``MAX_CONTEXT`` always stands, so an entry that
    declares nothing usable still has a ceiling.

    Args:
        entry: The catalogue entry, or None.
        meta: The checkpoint metadata, or None.

    Returns:
        The smallest declared limit, in tokens.
    """
    declared = [(entry or {}).get("max_ctx"), (meta or {}).get("context"), MAX_CONTEXT]
    return min(v for v in declared if type(v) is int and v > 0)


def context_plan(
    plan,
    total_mib,
    backend,
    ceiling,
    *,
    mtp=False,
    draft_cache="q8_0",
    desktop=True,
    driver_reserve=512,
):
    """Plan a context and slot count, given a checkpoint's shape and a GPU.

    Args:
        plan: The checkpoint's planning fields, as ``planner_evidence``
            merges them: ``size_gb``, ``kv_kib_per_token`` and optionally
            ``full_attention_layers``, ``mmproj`` and ``mmproj_gb``.
        total_mib: The GPU's total memory.
        backend: The llama.cpp GPU backend that runs the model.
        ceiling: The largest context this checkpoint allows.
        mtp: Whether the launch speculates with a multi-token prediction head.
        draft_cache: The draft cache type, ``q8_0`` or ``default``.
        desktop: Whether a desktop session also uses this GPU.
        driver_reserve: MiB the driver holds back.

    Returns:
        A pair: the total context across all slots, and the slot count.

    Raises:
        ValueError: The budget leaves no usable context on this GPU.
    """
    # Same reserves, q8 block cost, backend factors and slot policy as the
    # source planner. Retain its conservative vision allowance for these entries.
    budget = total_mib - driver_reserve - 1024 - (2400 if desktop else 0)
    budget -= 230 if backend == "CUDA" else 0
    budget -= 1024 if plan.get("mmproj") else 0
    budget -= (plan["size_gb"] + plan.get("mmproj_gb", 0)) * 1e9 / 2**20
    cost = plan["kv_kib_per_token"] * (34 / 64)
    if mtp and plan.get("full_attention_layers"):
        cost += (
            plan["kv_kib_per_token"]
            / plan["full_attention_layers"]
            * 1.25
            * (34 / 64 if draft_cache == "q8_0" else 1)
        )
    cost *= (1.12 / 2 / (34 / 64)) * (1.136 if backend == "CUDA" else 1)
    pool = min(MAX_CONTEXT, max(0, int(budget * 1024 / cost) // 1024 * 1024))
    if pool < 1024:
        raise ValueError(
            "The inherited memory budget leaves no usable context on this GPU."
        )
    ceiling = min(ceiling, MAX_CONTEXT)

    def window(slots):
        return min(ceiling, pool // slots) // 1024 * 1024

    slots = 1
    if pool >= 1.5 * window(1):
        slots = 2
        while slots < 4:
            here, further = window(slots), window(slots + 1)
            if (
                further <= 1024
                or ((slots + 1) * further) / (slots * here) <= here / further
            ):
                break
            slots += 1
    share = window(slots)
    if share < 512:
        raise ValueError(
            "Checkpoint context metadata is too small for inherited planning."
        )
    return share * slots, slots


def starting_defaults(selection, host=None, engine=None, meta=None, model=None):
    """Resolve starting settings for a checkpoint, engine and GPU.

    A remote engine supplies every evidence argument from its provider probe
    and the catalogue (see ``Engine.defaults_inputs``), so no local probe runs.

    Args:
        selection: Settings naming the model, engine, backend and device, and
            for a remote backend the GPU type.
        host: Hardware description in the ``hardware()`` shape. None detects
            local hardware. A description whose ``source`` is not ``"local"``
            skips local desktop and driver probes.
        engine: Engine capability record in the ``probe()`` shape. None probes
            the selected binary locally.
        meta: GGUF metadata record in the ``metadata()`` shape. None reads the
            selected checkpoint locally.
        model: Model identity with ``size`` and ``sha256``, as
            ``Engine.identity`` returns it. None fingerprints the local file.

    Returns:
        A dict with ``settings``, ``source`` and ``notes``.

    Raises:
        ValueError: The launch needs a GPU and the hardware reports none.
    """
    measured, qualifications = measured_defaults(selection, host, engine, meta, model)
    if measured:
        return measured
    fallback = inherited_defaults(selection, host, engine, meta, model)
    fallback["notes"] = qualifications + fallback["notes"]
    return fallback


def inherited_defaults(selection, host=None, engine=None, meta=None, model=None):
    """Resolve inherited estimated settings; arguments match ``starting_defaults``.

    Raises:
        ValueError: The launch needs a GPU and the hardware reports none.
    """
    # Never carry experiments or another model's drafter/effort into a new model.
    s = Settings(
        model=selection.model,
        engine=selection.engine,
        backend=selection.backend,
        device=selection.device,
        gpu_type=selection.gpu_type,
        idle_timeout_minutes=selection.idle_timeout_minutes,
    )
    notes = []
    if not s.model or not (s.engine or s.remote):
        return {
            "settings": s.dict(),
            "source": "generic fallback",
            "notes": ["Select a checkpoint and engine to load inherited tuning."],
        }
    p = probe(s.engine) if engine is None else engine
    automatic_layers = {"--fit", "--fit-target"}.issubset(p["flags"])
    if automatic_layers:
        notes.append(
            "GPU layers: automatic. At startup the engine fits weights and buffers to available VRAM with a 1024 MiB margin, offloading to system RAM when needed. The engine log records actual placement."
        )
    else:
        s.gpu_layers = 999
    devices = [d for d in p["devices"] if d.startswith(s.gpu_backend)]
    if s.device not in devices:
        s.device = devices[0] if devices else ""
    m = metadata(s.model) if meta is None else meta
    caps = capabilities(s, p, m)
    entry = catalogue_entry(s.model)
    s.flash = "on" if caps["flash"]["status"] == "available" else "auto"
    if caps["cache"]["status"] == "available" and s.flash == "on":
        s.cache = "q8_0"
    else:
        notes.append(
            "This binary cannot apply the inherited q8_0/flash settings; generic cache/context retained."
        )
    host = hardware() if host is None else host
    local = host.get("source", "local") == "local"
    cards = host["gpus"]
    if not cards:
        # Every backend here runs the model on a GPU. Planning a context
        # against no GPU at all would invent a number, so this is an error the
        # caller reports rather than a silent fall through to 4096 tokens.
        raise ValueError(
            f"No {s.gpu_backend} GPU is available for this launch"
            + (
                ". Check that the driver is loaded and the card is visible."
                if local
                else f", so {selection.backend} cannot size its context. Probe the GPU type again."
            )
        )
    low_vram = entry.get("low_vram_defaults") if entry else None
    if (
        low_vram
        and len(cards) == 1
        and 0 < cards[0]["total_mib"] <= low_vram["max_vram_mib"]
    ):
        # A400 / 32 GiB RAM starting point. Keep the GGUF context limit intact
        # for explicit experiments; the all-GPU planner cannot budget CPU weights.
        ceiling = min(entry["max_ctx"], m["context"] or entry["max_ctx"])
        s.context = max(512, min(low_vram["context"], ceiling) // 512 * 512)
        s.slots = 1
        s.speculation = "none"
        for key, flag, value in [
            ("batch_size", "--batch-size", low_vram["batch_size"]),
            ("ubatch_size", "--ubatch-size", low_vram["ubatch_size"]),
            ("cache_ram_mib", "--cache-ram", 0),
            ("context_checkpoints", "--ctx-checkpoints", 0),
        ]:
            if flag in p["flags"]:
                setattr(s, key, value)
            else:
                notes.append(f"Engine lacks {flag}; its built-in default will apply.")
        if not automatic_layers:
            s.gpu_layers = 0
            notes.append(
                "This engine lacks automatic memory fitting. Model layers stay on the CPU (0); use a fitting-capable engine for automatic GPU offloading."
            )
        notes.append(
            "Conservative starting settings for a 4 GiB GPU and 32 GiB system RAM: one conversation, up to 4K context, small prompt batches and speculation off. Prompt-cache RAM and recurrent checkpoints are disabled where supported. This is an unmeasured estimate; most weights stay in system RAM and CPU offloading can slow prompt processing."
        )
        # Q4_K_M weights are about 20.5 GiB. Allow host runtime headroom even
        # if automatic fitting keeps few or no layers on the GPU.
        available = host.get("ram", {}).get("available_gib")
        if available is not None and available < 24:
            notes.append(
                f"Only {available:g} GiB system RAM is available. Aim for about 24 GiB available before loading this checkpoint to leave room for weights and runtime buffers."
            )
        return {
            "settings": s.dict(),
            "source": "Low-VRAM starting settings · estimates",
            "notes": notes,
        }
    if not automatic_layers:
        notes.append(
            "This engine lacks automatic memory fitting. GPU layers request full offload (999); enter a smaller count if the model does not fit."
        )
    s.draft_length = 3
    if caps["draft-mtp"]["status"] == "available":
        s.speculation = "draft-mtp"
    else:
        notes.append("MTP not enabled: " + caps["draft-mtp"]["reason"])
    draft_flags = {"--spec-draft-type-k", "--spec-draft-type-v"}
    s.draft_cache = "q8_0" if draft_flags.issubset(p["flags"]) else "default"
    if s.speculation in MTP_MODES and s.draft_cache == "default":
        notes.append(
            "Draft-cache controls unavailable; context budget accounts for f16 draft cache."
        )
    if entry and entry.get("chat_template"):
        if "--chat-template-file" in p["flags"]:
            s.chat_template = str(
                Path(__file__).with_name("templates") / entry["chat_template"]
            )
        else:
            notes.append("Binary lacks the inherited chat-template override flag.")
    plan, unknown = planner_evidence(entry, m, model)
    ceiling = planner_ceiling(entry, m)
    uncalibrated = planner_obstacle(s, cards, unknown)
    if uncalibrated:
        # Never leave the dataclass default of 4096 tokens standing in silence:
        # on a large GPU it fails a long prompt mid-generation, which reads as a
        # model limit rather than as a setting nobody planned.
        s.context = uncalibrated_context(
            ceiling, plan, cards, s.cache, s.gpu_backend, local
        )
        s.slots = 1
        notes.append(
            f"Context is NOT calibrated for this checkpoint and GPU, because {uncalibrated}. "
            f"Starting with one conversation and {s.context} tokens rather than the placeholder 4096, which is not a fitted estimate either. "
            "Watch the engine log for a failed allocation, and set the context yourself before a long session."
        )
    else:
        # A remote container runs no desktop, and its driver reserve is unknown.
        desktop, reserve = False, 512
        if local:
            rc, state = command(["systemctl", "is-active", "graphical.target"], 5)
            desktop = state.strip() != "inactive"
            rc, reserved = command(
                [
                    "nvidia-smi",
                    "--id=" + cards[0]["uuid"],
                    "--query-gpu=memory.reserved",
                    "--format=csv,noheader,nounits",
                ],
                4,
            )
            try:
                reserve = max(0, int(reserved.strip())) if rc == 0 else 512
            except ValueError:
                reserve = 512
        try:
            s.context, s.slots = context_plan(
                plan,
                cards[0]["total_mib"],
                s.gpu_backend,
                ceiling,
                mtp=s.speculation in MTP_MODES,
                draft_cache=s.draft_cache,
                desktop=desktop,
                driver_reserve=reserve,
            )
            notes.append(
                "Context/slots use the inherited calibrated planner on this GPU; verify with a workload. They are estimates, not new measurements."
            )
        except ValueError as e:
            if automatic_layers:
                s.context = clamp_context(ceiling)
                s.slots = 1
                s.speculation = "none"
                notes.append(
                    "The full-GPU context estimate does not fit. Starting with one conversation, up to 32K context and speculation off; automatic placement may use system RAM. This is an unmeasured starting point; CPU offloading can slow prompt processing."
                )
            else:
                notes.append(
                    str(e) + " Set a smaller model or explicit settings before launch."
                )
    return {"settings": s.dict(), "source": SOURCE, "notes": notes}


def planner_obstacle(s, cards, unknown):
    """Explain why the inherited planner cannot budget this launch, or ``""``.

    Args:
        s: The settings being resolved.
        cards: The GPUs the hardware description reports.
        unknown: Planner fields no source could supply, from
            ``planner_evidence``.

    Returns:
        A clause naming the obstacle, to be read after "because". Empty when
        the planner can run.
    """
    if len(cards) != 1:
        return f"the inherited planner budgets a single GPU and this host reports {len(cards)}"
    if s.cache != "q8_0":
        return "the inherited planner is calibrated for a q8_0 KV cache, which this engine cannot provide"
    if unknown:
        return "neither the checkpoint header nor the bundled catalogue gives its " + (
            " or ".join(
                {"size_gb": "file size", "kv_kib_per_token": "KV cache cost"}[name]
                for name in unknown
            )
        )
    return ""


def clamp_context(ceiling):
    """Return the uncalibrated starting context, aligned and within a ceiling."""
    return max(512, min(UNCALIBRATED_CONTEXT, ceiling) // 512 * 512)


def uncalibrated_context(ceiling, plan, cards, cache, backend, desktop=False):
    """Return a starting context for a launch the calibrated planner refuses.

    32K is the target, but it is only a safe starting point on a card with room
    for it. A small GPU holding a large checkpoint has none, and an engine
    without automatic fitting allocates exactly what it is told and dies at
    load. So where the checkpoint's size and KV cost are known -- which is
    every obstacle except a checkpoint nobody can describe -- bound the target
    by the memory, pricing the cache the engine will actually use rather than
    the q8_0 the planner assumes. This only ever lowers the context: the
    calibrated planner has already declined, so this is a floor to stand on,
    not a fitted estimate.

    Args:
        ceiling: The largest context this checkpoint allows.
        plan: Planning fields from ``planner_evidence``.
        cards: The GPUs the hardware description reports; memory is summed,
            as llama.cpp splits both weights and cache across them.
        cache: The KV cache type the launch will request.
        backend: The llama.cpp GPU backend that runs the model.
        desktop: Whether a desktop session may share the GPU. True reserves for
            one without probing, which a caller in this position cannot do.

    Returns:
        The context, aligned to 512 tokens and at least 512.
    """
    context = clamp_context(ceiling)
    size, kv = plan.get("size_gb"), plan.get("kv_kib_per_token")
    if not size or not kv or not cards:
        return context
    budget = sum(c["total_mib"] for c in cards) - 512 - 1024 - (2400 if desktop else 0)
    budget -= 230 if backend == "CUDA" else 0
    budget -= 1024 if plan.get("mmproj") else 0
    budget -= (size + plan.get("mmproj_gb", 0)) * 1e9 / 2**20
    # The planner's own per-token cost, with its q8_0 factor made explicit so
    # an f16 cache is priced as the twice-the-size thing it is.
    cost = kv * (34 / 64 if cache == "q8_0" else 1) * (1.12 / 2 / (34 / 64))
    cost *= 1.136 if backend == "CUDA" else 1
    fits = max(0, int(budget * 1024 / cost)) // 512 * 512
    return max(512, min(context, fits))
