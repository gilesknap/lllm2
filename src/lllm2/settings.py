import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .discovery import (
    EXECUTION_ENV_KEYS,
    batch_defaults,
    cache_kernel_support,
    cuda_graph_support,
    engine_environment,
    metadata,
    probe,
)
from .gpu_tables import GPU_TABLES, gpu_type

MTP_MODES = ("draft-mtp", "draft-mtp,ngram-simple")
LOCAL_BACKENDS = ("CUDA", "Vulkan")
DEFAULT_IDLE_TIMEOUT_MINUTES = 30
LOOKUP_MODES = ("ngram-simple", "draft-mtp,ngram-simple")


@dataclass
class Settings:
    model: str = ""
    engine: str = ""
    backend: str = "CUDA"
    device: str = "CUDA0"
    context: int = 4096
    slots: int = 1
    gpu_layers: int | None = None
    flash: str = "auto"
    cache: str = "f16"
    cache_k: str | None = None
    cache_v: str | None = None
    speculation: str = "none"
    drafter: str = ""
    pair_confirmed: bool = False
    draft_length: int = 15
    effort: str = "default"
    draft_cache: str = "default"
    chat_template: str = ""
    batch_size: int | None = None
    ubatch_size: int | None = None
    backend_sampling: bool = False
    cuda_graph_opt: str = "default"
    cache_ram_mib: int | None = None
    context_checkpoints: int | None = None
    lookup_ngram_n: int | None = None
    lookup_ngram_m: int | None = None
    gpu_type: str = ""
    idle_timeout_minutes: int | None = DEFAULT_IDLE_TIMEOUT_MINUTES

    @property
    def remote(self):
        """Return whether a remote provider, not this machine, serves the model."""
        return self.backend not in LOCAL_BACKENDS

    @property
    def gpu_backend(self):
        """Return the llama.cpp GPU backend that runs the model.

        Remote providers run the lllm2 CUDA engine release, so a remote backend
        resolves to ``"CUDA"``.
        """
        return "CUDA" if self.remote else self.backend

    @classmethod
    def parse(cls, data):
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown launch setting.")
        s = cls(**data)
        for key in ("cache_k", "cache_v"):
            if getattr(s, key) == "":
                setattr(s, key, None)
            if getattr(s, key) not in (None, "f16", "q8_0", "q4_0"):
                raise ValueError(f"Invalid {key}")
        if type(s.backend_sampling) is not bool:
            raise ValueError("backend_sampling must be boolean")
        if s.cuda_graph_opt not in ("default", "on", "off"):
            raise ValueError("Invalid cuda_graph_opt")
        for key, upper in [
            ("cache_ram_mib", 8192),
            ("context_checkpoints", 32),
            ("lookup_ngram_n", 256),
            ("lookup_ngram_m", 256),
        ]:
            if getattr(s, key) == "":
                setattr(s, key, None)
            value = getattr(s, key)
            if value is not None and (
                type(value) is not int or not 0 <= value <= upper
            ):
                raise ValueError(f"{key} must be blank or an integer in 0..{upper}")
        if (s.lookup_ngram_n is None) != (s.lookup_ngram_m is None):
            raise ValueError("Set both lookup N and M, or leave both blank.")
        if (
            s.lookup_ngram_n is not None
            and not 1 <= s.lookup_ngram_n <= s.lookup_ngram_m
        ):
            raise ValueError(
                "Lookup requires 1 <= N <= M; N greater than M produces no drafts."
            )
        for key in ("batch_size", "ubatch_size"):
            if getattr(s, key) == "":
                setattr(s, key, None)
            value = getattr(s, key)
            if value is not None and (
                type(value) is not int or not 1 <= value <= 1048576
            ):
                raise ValueError(f"{key} must be blank or an integer in 1..1048576")
        if (
            s.batch_size is not None
            and s.ubatch_size is not None
            and s.ubatch_size > s.batch_size
        ):
            raise ValueError("Microbatch must not exceed logical batch size.")
        if s.gpu_layers == "":
            s.gpu_layers = None
        if s.gpu_layers is not None and (
            type(s.gpu_layers) is not int or not 0 <= s.gpu_layers <= 999
        ):
            raise ValueError(
                "GPU layers must be blank (automatic) or an integer in 0..999"
            )
        for key, low, high in [
            ("context", 512, 1048576),
            ("slots", 1, 16),
            ("draft_length", 1, 256),
        ]:
            value = getattr(s, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer in {low}..{high}")
        if s.context % s.slots or s.context // s.slots < 512:
            raise ValueError(
                "Total context must divide evenly into slots, each at least 512 tokens."
            )
        for key, choices in {
            "draft_cache": ["default", "f16", "q8_0", "q4_0"],
            "backend": [*LOCAL_BACKENDS, *(k for k in GPU_TABLES if k != "local")],
            "flash": ["auto", "on", "off"],
            "cache": ["f16", "q8_0", "q4_0"],
            "speculation": [
                "none",
                "draft-mtp",
                "draft-dflash",
                "ngram-simple",
                "draft-mtp,ngram-simple",
            ],
            "effort": ["default", "minimal", "low", "medium", "high", "xhigh", "max"],
        }.items():
            if getattr(s, key) not in choices:
                raise ValueError(f"Invalid {key}")
        if type(s.pair_confirmed) is not bool:
            raise ValueError("pair_confirmed must be boolean")
        if s.idle_timeout_minutes == "":
            s.idle_timeout_minutes = None
        if s.idle_timeout_minutes is not None and (
            type(s.idle_timeout_minutes) is not int
            or not 0 <= s.idle_timeout_minutes <= 1440
        ):
            raise ValueError(
                "idle_timeout_minutes must be blank or an integer in 0..1440; 0 or blank disables the idle timeout."
            )
        if type(s.gpu_type) is not str:
            raise ValueError("gpu_type must be a string")
        if s.remote:
            if not s.gpu_type:
                raise ValueError(f"Choose a {s.backend} GPU type.")
            gpu_type(s.backend, s.gpu_type)
            if s.engine:
                raise ValueError(
                    "A remote backend runs the provider's engine release; leave the engine path blank."
                )
            if s.device != "CUDA0":
                raise ValueError(
                    "A remote backend serves on one CUDA GPU; use device CUDA0."
                )
        elif s.gpu_type:
            raise ValueError("A GPU type applies only to a remote backend.")
        return s

    def dict(self):
        return asdict(self)

    def cache_pair(self):
        return self.cache_k or self.cache, self.cache_v or self.cache


def cache_settings(s, kernel=None):
    """Describe the requested and resolved attention cache types.

    Args:
        s: Settings to describe.
        kernel: Cache kernel evidence in the ``cache_kernel_support`` shape. None
            reads the library next to the local engine binary.

    Returns:
        The requested and resolved types with kernel evidence and caveats.
    """
    k, v = s.cache_pair()
    return {
        "requested": {"cache": s.cache, "cache_k": s.cache_k, "cache_v": s.cache_v},
        "resolved": {"k": k, "v": v},
        "kernel": cache_kernel_support(s.engine, s.gpu_backend)
        if kernel is None
        else kernel,
        "context_evidence": "Inherited planner is calibrated for q8_0/q8_0 only; it is not a measured capacity for this pair.",
        "state_scope": "K/V choices apply to attention cache; hybrid recurrent states retain engine-selected precision. Draft cache is unchanged.",
    }


def capabilities(s, engine=None, meta=None):
    """Report which settings the engine binary and checkpoint support.

    Args:
        s: Settings to check.
        engine: Engine capability record in the ``probe()`` shape: ``path``,
            ``flags``, ``help``, ``devices`` and ``error``. The optional keys
            ``cuda_graph`` (``cuda_graph_support()`` shape), ``cache_kernel``
            (``cache_kernel_support()`` shape) and ``environment`` (the engine's
            process environment) replace evidence that is otherwise read next
            to the local binary. None probes ``s.engine`` locally.
        meta: GGUF metadata record in the ``metadata()`` shape. The optional key
            ``drafter`` holds the DFlash drafter's record in the same shape and
            replaces the local drafter file check. None reads ``s.model``
            locally.

    Returns:
        A dict mapping each feature name to its status and reason.
    """
    p = probe(s.engine) if engine is None else engine
    m = metadata(s.model) if meta is None else meta
    flags, help_text = p["flags"], p["help"]

    def flag_status(flag):
        return (
            "unknown" if p["error"] else "available" if flag in flags else "unsupported"
        )

    out = {}
    sampling_status = (
        flag_status("--backend-sampling") if s.gpu_backend == "CUDA" else "unsupported"
    )
    out["backend_sampling"] = {
        "status": sampling_status,
        "reason": "Experimental target GPU sampling requires CUDA and --backend-sampling. Requests may fall back to CPU; no measured benefit implied. Draft sampling is unchanged.",
    }
    graph = p.get("cuda_graph") or cuda_graph_support(s.engine)
    reason = (
        graph["reason"]
        + " Requires ordinary CUDA Graphs and one visible CUDA device; performance needs measurement."
    )
    status = (
        "available" if s.gpu_backend == "CUDA" and graph["supported"] else "unsupported"
    )
    env = p["environment"] if "environment" in p else engine_environment(s.engine)
    if env.get("GGML_CUDA_DISABLE_GRAPHS") is not None:
        status, reason = (
            "missing prerequisites",
            "Inherited GGML_CUDA_DISABLE_GRAPHS disables ordinary CUDA Graphs; concurrent streams cannot run.",
        )
    if len([d for d in p["devices"] if d.startswith("CUDA")]) > 1:
        status, reason = (
            "missing prerequisites",
            "Concurrent streams require exactly one visible CUDA device.",
        )
    out["cuda_graph_opt"] = {"status": status, "reason": reason, "evidence": graph}
    for name, flag in [
        ("flash", "--flash-attn"),
        ("cache", "--cache-type-k"),
        ("effort", "--reasoning-effort"),
    ]:
        status = flag_status(flag)
        if name == "cache" and "--cache-type-v" not in flags:
            status = "unsupported"
        out[name] = {
            "status": status,
            "reason": f"Binary probe: {flag}. Actual model/backend behavior is validated on launch.",
        }
    pair = cache_settings(s, p.get("cache_kernel"))
    k, v = s.cache_pair()
    out["cache_pair"] = {
        "status": out["cache"]["status"]
        if k == v or out["cache"]["status"] != "available"
        else "experimental",
        "reason": pair["kernel"]["reason"]
        if k != v
        else "Resolved attention cache: K "
        + k
        + ", V "
        + v
        + ". Equal types still require a successful engine/model/backend launch.",
        "evidence": pair,
    }
    defaults = batch_defaults(help_text) if not p["error"] else {}
    for name, flag in [
        ("cache_ram_mib", "--cache-ram"),
        ("context_checkpoints", "--ctx-checkpoints"),
    ]:
        block = re.search(
            r"(?m)^[^\n]*(?<!\S)"
            + re.escape(flag)
            + r"(?=[,\s])[^\n]*(?:\n(?![ \t]*-{1,2}[A-Za-z])[ \t]{10,}[^\n]*)*",
            help_text,
        )
        match = re.search(r"\(default:\s*(\d+)(?=[,\s)])", block[0]) if block else None
        out[name] = {
            "status": flag_status(flag),
            "advertised_default": int(match[1]) if match else None,
            "reason": "Blank preserves engine defaults for normal launches. Zero disables this cache component; finite values do not cap total process memory.",
        }
    for name, flag in [
        ("batch_size", "--batch-size"),
        ("ubatch_size", "--ubatch-size"),
    ]:
        default = defaults.get(name)
        out[name] = {
            "status": flag_status(flag),
            "advertised_default": default,
            "reason": f"Blank leaves the engine default unchanged: {default if default is not None else 'unknown'}. Effective size and performance require a launch measurement.",
        }
    template = (
        Path(s.chat_template).expanduser().read_text()
        if s.chat_template
        else m["template"]
    )
    if out["effort"]["status"] == "available" and "reasoning_effort" not in template:
        out["effort"] = {
            "status": "unknown",
            "reason": "Binary accepts effort; checkpoint template does not explicitly reference reasoning_effort. Verify with a launch.",
        }
    for mode in ["draft-mtp", "draft-dflash", "ngram-simple"]:
        status = flag_status("--spec-type")
        reason = "Advertised by this binary; successful workload run still required."
        if status == "available" and mode not in help_text:
            status, reason = (
                "unsupported",
                "This speculative mode is not advertised by the selected binary.",
            )
        if status == "available" and mode == "draft-mtp" and not m["mtp"]:
            status, reason = (
                ("unknown", "GGUF inspection failed.")
                if m["error"]
                else ("missing prerequisites", "Checkpoint contains no MTP tensors.")
            )
        if status == "available" and mode == "draft-dflash":
            if s.gpu_backend != "CUDA":
                status, reason = (
                    "unsupported",
                    "Initial experimental DFlash integration is CUDA only.",
                )
            elif not s.drafter or (
                "drafter" not in m and not Path(s.drafter).expanduser().is_file()
            ):
                status, reason = (
                    "missing prerequisites",
                    "Provide a target-specific ordinary DFlash GGUF (DFlash 2 is not integrated).",
                )
            elif not s.pair_confirmed:
                status, reason = (
                    "missing prerequisites",
                    "Confirm this drafter was trained and converted for this exact target; filenames alone are insufficient.",
                )
            elif (m["drafter"] if "drafter" in m else metadata(s.drafter))["error"]:
                status, reason = (
                    "missing prerequisites",
                    "Drafter is not a readable GGUF.",
                )
            elif (
                "--model-draft" not in flags
                or "--spec-draft-n-max" not in flags
                or "--flash-attn" not in flags
            ):
                status, reason = (
                    "unsupported",
                    "Binary lacks required draft model, draft length or flash-attention controls.",
                )
            else:
                reason = "Pair declared by user, not yet performance validated. Requires flash attention on. No MTP combination."
        out[mode] = {"status": status, "reason": reason}
    required = {
        "--spec-type",
        "--spec-draft-n-max",
        "--spec-ngram-simple-size-n",
        "--spec-ngram-simple-size-m",
    }
    c = dict(out["draft-mtp"])
    if c["status"] == "available":
        if out["ngram-simple"]["status"] != "available":
            c = dict(out["ngram-simple"])
        elif not required.issubset(flags):
            c = {
                "status": "unsupported",
                "reason": "Combination requires MTP draft width and both lookup N/M controls.",
            }
        else:
            c["reason"] = (
                "Explicit draft-mtp,ngram-simple combination. Build 662a0b0 tries lookup first, then MTP on misses; other builds need execution verification. Blank N/M uses 3/3 for this combination only. No measured benefit implied."
            )
    out["draft-mtp,ngram-simple"] = c
    for key, flag in [
        ("lookup_ngram_n", "--spec-ngram-simple-size-n"),
        ("lookup_ngram_m", "--spec-ngram-simple-size-m"),
    ]:
        out[key] = {
            "status": flag_status(flag),
            "reason": "Optional paired lookup controls; N is match length, M is lookup draft length. Require 1 <= N <= M. Blank omits flags except explicit MTP+lookup uses 3/3.",
        }
    return out


def speculative_settings(s, timings=None, engine=None):
    """Describe the requested speculative decoding and its observed counters.

    Args:
        s: Settings to describe.
        timings: The completion response's ``timings`` record, or None.
        engine: Engine capability record in the ``probe()`` shape. None probes
            ``s.engine`` locally.

    Returns:
        The mode, draft and lookup lengths, method order and draft counters.
    """
    combined = s.speculation == "draft-mtp,ngram-simple"
    p = probe(s.engine) if engine is None else engine
    timings = timings or {}
    return {
        "mode": s.speculation,
        "mtp_draft_length": s.draft_length if s.speculation in MTP_MODES else None,
        "lookup_n": (
            s.lookup_ngram_n
            if s.lookup_ngram_n is not None
            else 3
            if combined
            else None
        )
        if s.speculation in LOOKUP_MODES
        else None,
        "lookup_m": (
            s.lookup_ngram_m
            if s.lookup_ngram_m is not None
            else 3
            if combined
            else None
        )
        if s.speculation in LOOKUP_MODES
        else None,
        "order": "lookup first, MTP fallback"
        if combined and "662a0b0" in p.get("version", "")
        else "not verified for this engine",
        "order_evidence": "llama.cpp 662a0b0 common/speculative.cpp",
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "counter_scope": "HTTP counters aggregate all speculative methods; per-method counts unavailable without trace instrumentation",
    }


def launch_environment(s):
    env = engine_environment(s.engine)
    if s.gpu_layers is None:
        # An inherited explicit layer count prevents llama.cpp's fitter from
        # choosing placement, just as an explicit --gpu-layers argument does.
        env.pop("LLAMA_ARG_N_GPU_LAYERS", None)
    if s.cuda_graph_opt != "default":
        env["GGML_CUDA_GRAPH_OPT"] = "1" if s.cuda_graph_opt == "on" else "0"
    return env


def execution_settings(s, environment=None, logs=(), response=None):
    """Requested options and observed diagnostics; never infer GPU execution from a flag."""
    lines = list(logs)
    starts = [i for i, line in enumerate(lines) if line.startswith("Launching: ")]
    current = lines[starts[-1] + 1 :] if starts else []
    diagnostics = [
        line
        for line in current
        if re.search(
            r"backend sampl|sampler chain|sampl.*(?:fallback|disabl|not compatible|not supported)",
            line,
            re.I,
        )
    ]
    generation = (response or {}).get("generation_settings", {})
    return {
        "requested": {
            "backend_sampling": s.backend_sampling,
            "cuda_graph_opt": s.cuda_graph_opt,
        },
        "child_environment": {k: environment.get(k) for k in EXECUTION_ENV_KEYS}
        if environment is not None
        else None,
        "target_sampling": {
            "response_requested": generation.get("backend_sampling"),
            "offload": "unknown; normal logs do not prove complete target GPU sampling",
        },
        "draft_sampling": {
            "requested": None,
            "policy": "engine default unchanged; enablement not observed",
        },
        "diagnostics": diagnostics[-40:],
    }


def batch_settings(s, logs=(), engine=None):
    """Keep requests and help defaults separate from observed target-context values.

    Args:
        s: Settings to describe.
        logs: Engine log lines, oldest first.
        engine: Engine capability record in the ``probe()`` shape. None probes
            ``s.engine`` locally.

    Returns:
        The requested, advertised and observed batch and microbatch sizes.
    """
    p = probe(s.engine) if engine is None else engine
    defaults = batch_defaults(p["help"]) if not p["error"] else {}
    # Engine logs span restarts. Never attribute a previous launch's values to this one.
    lines = list(logs)
    starts = [i for i, line in enumerate(lines) if line.startswith("Launching: ")]
    current = lines[starts[-1] + 1 :] if starts else []
    out = {}
    for key, flag, runtime in [
        ("batch_size", "--batch-size", "n_batch"),
        ("ubatch_size", "--ubatch-size", "n_ubatch"),
    ]:
        # The target context is initialized before speculative/draft contexts.
        match = next(
            (
                m
                for line in current
                if (
                    m := re.search(
                        r"\bllama_context:\s+" + runtime + r"\s*=\s*(\d+)\b", line
                    )
                )
            ),
            None,
        )
        out[key] = {
            "requested": getattr(s, key),
            "supported": flag in p["flags"] if not p["error"] else None,
            "advertised_default": defaults.get(key),
            "effective": int(match[1]) if match else None,
            "effective_source": "target context startup log"
            if match
            else "unknown; not observed",
        }
    return out


def default_key(s):
    """Return the store key for saved launch defaults.

    Local defaults key on the checkpoint and GPU backend. Remote defaults also
    key on the GPU type, because context and slots depend on its memory.

    Args:
        s: Settings naming the model, backend and GPU type.

    Returns:
        The key string.
    """
    key = str(Path(s.model).expanduser().resolve()) + "|" + s.backend
    return key + "|" + s.gpu_type if s.remote else key


def launch_args(s, port):
    """Build the llama-server command line for a local launch.

    Args:
        s: Settings to launch.
        port: Loopback port for llama-server.

    Returns:
        The argument list, starting with the local binary path.

    Raises:
        ValueError: If the binary or checkpoint cannot run these settings.
    """
    return build_launch_args(s, port, probe(s.engine), metadata(s.model))


def build_launch_args(s, port, engine, meta, *, host="127.0.0.1", path=None):
    """Build the llama-server command line from supplied engine and GGUF facts.

    Local launches supply the local probe results. A remote provider supplies
    its own probe results and catalogue metadata, so both share one builder.

    Args:
        s: Settings to launch.
        port: Port for llama-server to listen on.
        engine: Engine capability record in the ``probe()`` shape. Its ``path``
            is the binary path where the command runs. See ``capabilities`` for
            optional evidence keys.
        meta: GGUF metadata record in the ``metadata()`` shape. A remote
            DFlash launch supplies the drafter's record under ``drafter``.
        host: Address for llama-server to bind.
        path: Function that maps a local model, drafter or template path to the
            path where the command runs. None resolves local paths and checks
            that the chat template file exists.

    Returns:
        The argument list, starting with ``engine["path"]``.

    Raises:
        ValueError: If the binary or checkpoint cannot run these settings.
    """
    p, m = engine, meta

    def target(value):
        if path is None:
            return str(Path(value).expanduser().resolve())
        return path(str(value))

    if s.batch_size is not None or s.ubatch_size is not None:
        defaults = batch_defaults(p["help"]) if not p["error"] else {}
        batch = s.batch_size if s.batch_size is not None else defaults.get("batch_size")
        micro = (
            s.ubatch_size if s.ubatch_size is not None else defaults.get("ubatch_size")
        )
        if batch is not None and micro is not None and micro > batch:
            raise ValueError(
                "Microbatch must not exceed logical batch size (including advertised engine defaults). Set both values explicitly to override defaults."
            )
    if s.device not in p["devices"] or not s.device.startswith(s.gpu_backend):
        raise ValueError(
            "Choose a detected GPU device matching the backend. Check binary --list-devices output."
        )
    if m["error"]:
        raise ValueError("Cannot read checkpoint: " + m["error"])
    if m["context"] and s.context // s.slots > m["context"]:
        raise ValueError(
            "Requested context per slot exceeds checkpoint context metadata."
        )
    caps = capabilities(s, p, m)
    if s.backend_sampling and caps["backend_sampling"]["status"] != "available":
        raise ValueError(caps["backend_sampling"]["reason"])
    if s.cuda_graph_opt != "default":
        if (
            s.gpu_backend != "CUDA"
            or not caps["cuda_graph_opt"]["evidence"]["supported"]
        ):
            raise ValueError(
                "Explicit CUDA streams settings require a CUDA build with the compiled switch."
            )
        if s.cuda_graph_opt == "on" and caps["cuda_graph_opt"]["status"] != "available":
            raise ValueError(caps["cuda_graph_opt"]["reason"])
    args = [p["path"]]

    def add(flag, value=None):
        if flag not in p["flags"]:
            raise ValueError(
                f"This binary does not advertise {flag}. Select a compatible existing build."
            )
        args.append(flag)
        if value is not None:
            args.append(str(value))

    for flag, value in [
        ("--model", target(s.model)),
        ("--host", host),
        ("--port", port),
        ("--ctx-size", s.context),
        ("--parallel", s.slots),
        ("--device", s.device),
    ]:
        add(flag, value)
    if s.gpu_layers is None:
        if not {"--fit", "--fit-target"}.issubset(p["flags"]):
            raise ValueError(
                "Automatic GPU layers require an engine with --fit and --fit-target. Select a compatible engine or enter a GPU layer count."
            )
        # Keep context/slots explicit: fit placement to the requested window,
        # including KV/compute buffers, rather than silently shrinking it.
        add("--fit", "on")
        add("--fit-target", 1024)
    else:
        add("--gpu-layers", s.gpu_layers)
    add("--jinja")
    if s.backend_sampling:
        add("--backend-sampling")
    for flag, value in [
        ("--batch-size", s.batch_size),
        ("--ubatch-size", s.ubatch_size),
        ("--cache-ram", s.cache_ram_mib),
        ("--ctx-checkpoints", s.context_checkpoints),
    ]:
        if value is not None:
            add(flag, value)
    if s.chat_template:
        if path is None and not Path(s.chat_template).expanduser().is_file():
            raise ValueError("Chat template file does not exist.")
        add("--chat-template-file", target(s.chat_template))
    if "--perf" in p["flags"]:
        add("--perf")
    if "--no-context-shift" in p["flags"]:
        add("--no-context-shift")
    if s.flash != "auto":
        add("--flash-attn", s.flash)
    cache_k, cache_v = s.cache_pair()
    if (cache_k, cache_v) != ("f16", "f16"):
        if s.flash != "on":
            raise ValueError("Quantized KV experiments require flash attention on.")
        add("--cache-type-k", cache_k)
        add("--cache-type-v", cache_v)
    elif "--cache-type-k" in p["flags"] and "--cache-type-v" in p["flags"]:
        add("--cache-type-k", "f16")
        add("--cache-type-v", "f16")
    if s.effort != "default":
        add("--reasoning-effort", s.effort)
    if s.speculation != "none":
        c = caps[s.speculation]
        if c["status"] != "available":
            raise ValueError(c["reason"])
        add("--spec-type", s.speculation)
        if s.draft_cache != "default" and s.speculation in (*MTP_MODES, "draft-dflash"):
            add("--spec-draft-type-k", s.draft_cache)
            add("--spec-draft-type-v", s.draft_cache)
        if "--spec-draft-n-max" in p["flags"]:
            add("--spec-draft-n-max", s.draft_length)
        if s.speculation in LOOKUP_MODES:
            for flag, value in [
                ("--spec-ngram-simple-size-n", s.lookup_ngram_n),
                ("--spec-ngram-simple-size-m", s.lookup_ngram_m),
            ]:
                if value is not None or s.speculation == "draft-mtp,ngram-simple":
                    add(flag, 3 if value is None else value)
        if s.speculation == "draft-dflash":
            if s.flash != "on":
                raise ValueError("DFlash requires flash attention on.")
            add("--model-draft", target(s.drafter))
            if "--spec-draft-device" in p["flags"]:
                add("--spec-draft-device", s.device)
    elif "--spec-type" in p["flags"]:
        add("--spec-type", "none")
    return args
