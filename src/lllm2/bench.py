import copy
import hashlib
import math
import re
import threading
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime

from . import config
from .discovery import hardware, identity, metadata, probe
from .engine import Cancelled, GPUUnavailable
from .settings import (
    Settings,
    batch_settings,
    cache_settings,
    capabilities,
    execution_settings,
    launch_args,
    speculative_settings,
)
from .source_workloads import TASKS, adherence, output_budget, source_prompt

WORKLOADS = {
    "generate": "Implement a Python LRU cache with get, put, bounded capacity and O(1) operations. Explain edge cases briefly and provide the code.",
    "edit": "Edit the supplied Python modules to reject negative amounts and use decimal.Decimal for monetary arithmetic. Return the changed code and explain the changes.",
    "long-code": "Review the supplied Python modules. Identify shared design problems, then implement a reusable ledger service with validated transactions and pagination. Return Python code.",
}

WORKLOADS.update(TASKS)


def prompt_sizes(s, opts):
    upper = (
        s.context // s.slots
        - max(
            output_budget(w, opts["output_tokens"])
            for w in opts.get("workloads") or ["long-code"]
        )
        - 32
    )
    if opts.get("sweep_prompts", True):
        sizes = {min(tokens, upper) for tokens in [1024, 16384, 65536]}
    else:
        sizes = {opts["prompt_tokens"]}
    if opts.get("full_window", False):
        sizes.add(upper)
    return sorted(sizes)


def stamp():
    return datetime.now(UTC).isoformat()


MEMORY_EXHAUSTED = re.compile(
    r"out of memory|failed to allocate|allocation of size .* failed|bad_alloc"
    r"|cannot allocate|OutOfMemory|OutOfDeviceMemory|cudaMalloc|failed to initialize the context"
    r"|failed to create context|\bKilled\b|not enough (?:memory|space)",
    re.IGNORECASE,
)


def memory_exhausted(logs, error=""):
    """Whether an engine failure looks like memory exhaustion rather than a fault."""
    return bool(MEMORY_EXHAUSTED.search("\n".join([*logs, error])))


def source_text(index):
    return f"""# module ledger_{index}.py
class Ledger{index}:
    def __init__(self):
        self.entries = []
    def deposit(self, account, amount):
        self.entries.append((account, amount))
    def balance(self, account):
        return sum(amount for owner, amount in self.entries if owner == account)
    def page(self, start, count):
        return self.entries[start:start + count]

"""


def suite(s):
    # The selected (normally inherited or saved) configuration is the baseline.
    # Every candidate changes exactly one setting, and invalid combinations skip.
    base = replace(s)
    caps = capabilities(base)
    variants = [("baseline", base)]
    skipped = []

    def candidate(label, **changes):
        v = replace(base, **changes)
        if v == base:
            return
        try:
            launch_args(v, config.ENGINE_PORT)
        except ValueError as e:
            skipped.append({"option": label, "reason": str(e)})
        else:
            variants.append((label, v))

    for mode in [
        "none",
        "draft-mtp",
        "draft-dflash",
        "ngram-simple",
        "draft-mtp,ngram-simple",
    ]:
        if mode == base.speculation:
            continue
        if mode == "none" or caps[mode]["status"] == "available":
            candidate("speculation-" + mode, speculation=mode)
        else:
            skipped.append({"option": mode, "reason": caps[mode]["reason"]})
    if caps["cache"]["status"] == "available":
        for cache in ["f16", "q8_0", "q4_0"]:
            if base.cache_pair() != (cache, cache):
                candidate("cache-" + cache, cache=cache, cache_k=None, cache_v=None)
    if caps["flash"]["status"] == "available":
        candidate(
            "flash-" + ("off" if base.flash == "on" else "on"),
            flash="off" if base.flash == "on" else "on",
        )
    if caps["effort"]["status"] == "available":
        for effort in ["default", "low", "high"]:
            candidate("effort-" + effort, effort=effort)
    return variants, skipped


class Bench:
    def __init__(self, engine, store):
        self.engine, self.store = engine, store
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.active = False
        self.progress = {"status": "idle"}

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.progress) | {"active": self.active}

    def update(self, **values):
        with self.lock:
            self.progress.update(values)

    def submit(self, data):
        s = Settings.parse(data["settings"])
        mode = data.get("mode", "baseline")
        opts = {
            "sweep_prompts": data.get("sweep_prompts", True),
            "full_window": data.get("full_window", False),
            "workloads": data.get("workloads", ["long-code"]),
            "prompt_tokens": data.get("prompt_tokens", 1024),
            "output_tokens": data.get("output_tokens", 256),
            "search_context": data.get("search_context", False),
            "max_context": data.get(
                "max_context", metadata(s.model)["context"] or 131072
            ),
            "timeout": data.get("timeout", 900),
            "context_timeout": data.get("context_timeout", 900),
            "repeats": data.get("repeats", 1),
        }
        if not isinstance(opts["workloads"], list) or any(
            w not in WORKLOADS for w in opts["workloads"]
        ):
            raise ValueError("Select coding workloads.")
        if not opts["workloads"] and not data.get("search_context", False):
            raise ValueError("Select coding workloads or discover usable context.")
        for k, lo, hi in [
            ("prompt_tokens", 128, 131072),
            ("output_tokens", 16, 4096),
            ("max_context", 512, 1048576),
            ("timeout", 10, 86400),
            ("context_timeout", 10, 86400),
            ("repeats", 1, 5),
        ]:
            if type(opts[k]) is not int or not lo <= opts[k] <= hi:
                raise ValueError(f"{k} must be in {lo}..{hi}")
        for k in ["search_context", "sweep_prompts", "full_window"]:
            if type(opts[k]) is not bool:
                raise ValueError(f"{k} must be boolean")
        if mode == "warm-conversation":
            if s.slots != 1 or any(
                opts[k] for k in ["search_context", "sweep_prompts", "full_window"]
            ):
                raise ValueError(
                    "Warm conversations require one slot, a fixed prompt size and no context search or full-window sweep."
                )
            if opts["prompt_tokens"] + 2 * opts["output_tokens"] + 160 > s.context:
                raise ValueError(
                    "Warm conversation needs room for the first prompt, two output budgets and 160 continuation/margin tokens."
                )
            opts["workloads"] = ["long-code"]
            s = replace(
                s,
                cache_ram_mib=2048 if s.cache_ram_mib is None else s.cache_ram_mib,
                context_checkpoints=4
                if s.context_checkpoints is None
                else s.context_checkpoints,
            )
        if opts["search_context"] and opts["max_context"] < opts["output_tokens"] + 160:
            raise ValueError(
                "Context ceiling must fit the output budget plus a real prompt."
            )
        effective_output = max(
            (output_budget(w, opts["output_tokens"]) for w in opts["workloads"]),
            default=opts["output_tokens"],
        )
        if (
            opts["workloads"]
            and (128 if opts["sweep_prompts"] else opts["prompt_tokens"])
            + effective_output
            + 32
            > s.context // s.slots
        ):
            raise ValueError(
                "Shared prompt and output budgets must fit context per slot, with 32 tokens of margin."
            )
        variants, skipped = (
            suite(s)
            if mode == "suite"
            else ([("baseline" if mode == "baseline" else "custom", replace(s))], [])
        )
        if mode not in [
            "baseline",
            "suite",
            "custom",
            "combinations",
            "warm-conversation",
        ]:
            raise ValueError("Unknown benchmark mode")
        if mode == "warm-conversation":
            opts["measurement_mode"] = mode
            variants = [("warm-conversation", s)]
        if mode == "combinations":
            choices = data.get("combinations", [])
            if not choices or len(choices) > 20:
                raise ValueError("Choose 1..20 combinations.")
            variants = [
                (f"combination-{i + 1}", Settings.parse(v))
                for i, v in enumerate(choices)
            ]
            if any(
                (v.model, v.engine, v.backend, v.device, v.context, v.slots)
                != (s.model, s.engine, s.backend, s.device, s.context, s.slots)
                for _, v in variants
            ):
                raise ValueError(
                    "Combinations must share model, engine, backend, device, context and slots for comparison."
                )
        for _, v in variants:
            launch_args(v, config.ENGINE_PORT)
        with self.lock:
            if self.active:
                raise ValueError("An operation is already running.")
            running = self.engine.state()
            if running["running"] and (
                data.get("replace_running") is not True
                or data.get("expected_pid") != running["pid"]
            ):
                raise ValueError(
                    "A model is serving or has changed. Refresh status and choose Stop model and run experiment to replace it."
                )
            self.active = True
            self.cancel.clear()
            self.progress = {
                "kind": "experiment",
                "status": "queued",
                "total": len(variants),
                "completed": 0,
                "skipped": skipped,
            }
        threading.Thread(target=self._run, args=(variants, opts), daemon=True).start()
        return self.snapshot()

    def _run(self, variants, opts):
        group = str(uuid.uuid4())
        cancelled = False
        fatal = None
        try:
            for i, (label, s) in enumerate(variants):
                if self.cancel.is_set():
                    raise Cancelled()
                r = {
                    "id": str(uuid.uuid4()),
                    "group": group,
                    "label": label,
                    "started": stamp(),
                    "status": "running",
                    "settings": s.dict(),
                    "options": opts,
                    "model": identity(s.model),
                    "engine": {
                        k: v
                        for k, v in probe(s.engine).items()
                        if k not in ["help", "flags"]
                    },
                    "batch_settings": batch_settings(s),
                    "cache_settings": cache_settings(s),
                    "execution_settings": execution_settings(s),
                    "hardware": hardware(),
                    "samples": [],
                    "probes": [],
                    "largest_observed_context": None,
                    "recommended_context": None,
                    "note": "Cold, uncached single-request coding probes; no quality or long-term stability claim. Context numbers are per slot.",
                }
                if opts.get("measurement_mode") == "warm-conversation":
                    r.update(
                        measurement_mode="warm-conversation",
                        note="Controlled token-prefix conversation, six turns plus identical uncached replays. First-token event timing is client-observed. Exact generated IDs are validated; engines that omit byte tokens while buffering UTF-8 fail with partial evidence. Reuse is measured, not assumed; larger-context reuse is unverified. No quality claim or cold-baseline promotion.",
                    )
                if s.drafter and s.speculation == "draft-dflash":
                    r["drafter"] = identity(s.drafter)
                    r["pair_evidence"] = (
                        "User-declared target-specific ordinary DFlash pair; successful runs verify execution only."
                    )
                self.store.put("result", r["id"], r)
                self.update(
                    status="running",
                    current=label,
                    completed=i,
                    result_id=r["id"],
                    phase="loading",
                    context_probe_started=None,
                )
                try:
                    if opts.get("measurement_mode") == "warm-conversation":
                        from .warm import run_conversation

                        run_conversation(self, s, opts, r)
                        r["status"] = "complete"
                        continue
                    # Context discovery first: its load checks are quick and the
                    # usable window is usually what people came for.
                    if opts["search_context"]:
                        self.context_search(s, opts, r)
                    sizes = prompt_sizes(s, opts) if opts["workloads"] else []
                    r["prompt_sizes"] = sizes
                    # Restart every sample to keep cold prefill unambiguous.
                    for workload in opts["workloads"]:
                        for repetition in range(opts["repeats"]):
                            for point, tokens in enumerate(sizes, 1):
                                self.update(
                                    phase=f"{workload}: prompt {point}/{len(sizes)}, {tokens:,} tokens; repeat {repetition + 1}/{opts['repeats']}"
                                )
                                self.engine.start(s, self.cancel, opts["timeout"])
                                r["argv"] = self.engine.argv
                                sample = self.measure(
                                    s,
                                    workload,
                                    tokens,
                                    opts["output_tokens"],
                                    opts["timeout"],
                                )
                                sample["repetition"] = repetition + 1
                                r["samples"].append(sample)
                                if (sample.get("adherence") or {}).get(
                                    "status"
                                ) == "failed":
                                    r["quality_status"] = "failed"
                                r["execution_settings"] = sample["execution_settings"]
                                self.store.put("result", r["id"], r)
                                self.engine.stop()
                    r["status"] = "complete"
                except Cancelled:
                    r["status"] = "cancelled"
                    cancelled = True
                except Exception as e:
                    r.update(
                        status="failed", error=str(e), logs=self.engine.state()["logs"]
                    )
                    attempt_env = self.engine.attempt_environment
                    r["execution_settings"] = execution_settings(
                        s,
                        attempt_env,
                        self.engine.state()["logs"] if attempt_env is not None else (),
                    )
                    if isinstance(e, GPUUnavailable) or not hardware()["gpus"]:
                        fatal = str(e)
                finally:
                    r["finished"] = stamp()
                    self.store.put("result", r["id"], r)
                    self.engine.stop()
                if cancelled or fatal:
                    break
            self.update(
                status="cancelled" if cancelled else "halted" if fatal else "complete",
                completed=i + 1,
                error=fatal,
            )
        except Cancelled:
            self.update(status="cancelled")
        except Exception as e:
            self.update(status="halted", error=str(e))
        finally:
            try:
                self.engine.stop()
            finally:
                with self.lock:
                    self.active = False

    def req(self, path, body, timeout):
        return self.engine.guarded_request(path, body, self.cancel, timeout)

    def prompt(self, s, workload, tokens, timeout):
        if workload in TASKS:
            return source_prompt(self, s, workload, tokens, timeout)

        # Tokenize the actual checkpoint's formatted chat prompt, preserving both
        # the complete instruction and template suffix while filling with code.
        def format_prompt(n):
            text = (
                "".join(source_text(i) for i in range(n)) + "\n" + WORKLOADS[workload]
            )
            body = {
                "messages": [{"role": "user", "content": text}],
                "add_generation_prompt": True,
            }
            if s.effort != "default":
                body["reasoning_effort"] = s.effort
            formatted = self.req("/apply-template", body, timeout)["prompt"]
            tok = self.req(
                "/tokenize",
                {"content": formatted, "add_special": True, "parse_special": True},
                timeout,
            )["tokens"]
            return text, formatted, tok

        # First estimate is deliberately oversized; truncate only source code tokens.
        text, formatted, tok = format_prompt(max(2, math.ceil(tokens / 90)))
        if len(tok) < tokens:
            text, formatted, tok = format_prompt(
                math.ceil(tokens / max(len(tok), 1) * max(2, math.ceil(tokens / 90)))
                + 10
            )
        if len(tok) < tokens:
            raise RuntimeError("Could not construct requested prompt length.")
        # Keep template start and the final instruction/template ending. Synthetic
        # modules may end partway through a function; recorded prompt is exact.
        keep_tail = min(256, tokens // 2)
        actual = tok[: tokens - keep_tail] + tok[-keep_tail:]
        return actual, {
            "generator": "ledger-v1",
            "workload": workload,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "formatted_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
            "token_ids": actual,
        }

    def measure(self, s, workload, tokens, output, timeout):
        with self.engine.log_lock:
            batches = batch_settings(s, list(self.engine.lines))
        prompt, provenance = self.prompt(s, workload, tokens, timeout)
        from .warm import host_memory

        with self.engine.guard:
            process = self.engine.process
        host_before = host_memory(process)
        requested_output = output
        output = output_budget(workload, output)
        source_task = workload in TASKS
        if tokens + output + 32 > s.context // s.slots:
            raise ValueError(
                "Source output cap and complete input must fit context with 32 tokens of margin."
            )
        memory = []
        finished = threading.Event()

        def sample_memory():
            while not finished.is_set():
                h = hardware()
                memory.append(
                    {
                        "time": stamp(),
                        "gpus": h["gpus"],
                        "error": h["error"],
                        "host": host_memory(process),
                    }
                )
                finished.wait(0.5)

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        started = time.monotonic()
        try:
            response = self.req(
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": output,
                    "temperature": 0,
                    "seed": 42,
                    "cache_prompt": False,
                    "stream": False,
                    "ignore_eos": not source_task,
                },
                timeout,
            )
            elapsed = time.monotonic() - started
        finally:
            finished.set()
            sampler.join(timeout=5)
        timings = response.get("timings", {})
        predicted = response.get("tokens_predicted", timings.get("predicted_n", 0))
        evaluated = response.get("tokens_evaluated", timings.get("prompt_n", 0))
        host_after = host_memory(process)
        host_points = [host_before, host_after] + [m["host"] for m in memory]
        if (
            not source_task and (response.get("truncated") or predicted < output)
        ) or evaluated < tokens:
            raise RuntimeError(
                f"Incomplete workload: requested {tokens}+{output} tokens, evaluated {evaluated}, generated {predicted}; truncated={response.get('truncated')}."
            )
        if timings.get("prompt_n", 0) < tokens - 1:
            raise RuntimeError(
                "Prompt cache reuse or incomplete timing detected; cannot report cold prefill."
            )
        with self.engine.log_lock:
            execution = execution_settings(
                s, self.engine.execution_environment, list(self.engine.lines), response
            )
        return {
            "workload": workload,
            "input_tokens": evaluated,
            "output_tokens": predicted,
            "output_budget": output,
            "requested_output_budget": requested_output,
            "adherence": adherence(
                response
                | {
                    "stopped_limit": response.get("stopped_limit", False)
                    or predicted >= output
                },
                provenance["expected_output"],
            )
            if source_task
            else None,
            "speculative_settings": speculative_settings(s, timings),
            "host_before": host_before,
            "host_after": host_after,
            "peak_engine_rss_mib": max(
                (h["rss_mib"] for h in host_points if h.get("rss_mib") is not None),
                default=None,
            ),
            "batch_settings": batches,
            "cache_settings": cache_settings(s),
            "execution_settings": execution,
            "context_per_slot": s.context // s.slots,
            "slots": s.slots,
            "wall_seconds": elapsed,
            "prefill_tok_s": timings.get("prompt_per_second"),
            "decode_tok_s": timings.get("predicted_per_second"),
            "timings": timings,
            "memory": memory,
            "peak_total_gpu_used_mib": max(
                (sum(g["used_mib"] for g in m["gpus"]) for m in memory), default=None
            ),
            "prompt": provenance,
            "output": response.get("content", ""),
            "sampling": {
                "temperature": 0,
                "seed": 42,
                "ignore_eos": not source_task,
                "cache_prompt": False,
            },
        }

    def context_search(self, s, opts, r):
        ceiling = min(
            opts["max_context"],
            metadata(s.model)["context"] or opts["max_context"],
            1048576 // s.slots,
        )
        floor = max(512, opts["output_tokens"] + 160)
        floor = math.ceil(floor / 256) * 256
        ceiling = ceiling // 256 * 256
        # `started` is the largest context the engine loaded, `good` the largest
        # verified with a full workload, `bad` the smallest failed probe and
        # `slow` the smallest timed-out one. A timeout bounds the search but is
        # not evidence of a memory limit. Workload confirmation is opt-in; without
        # it the loaded size is reported as usable but unconfirmed.
        good, started, bad, slow = 0, 0, ceiling + 256, ceiling + 256
        confirming = bool(opts.get("full_window"))
        # The search runs before any speed sample, so it begins by checking the
        # experiment's own window loads and doubles from there.
        selected = s.context // s.slots

        def best():
            # A successful workload probe is the confirmed size. Without
            # confirmation the loaded size is reported; with confirmation
            # requested but no workload success, nothing is reported as usable.
            if good or confirming:
                return good
            return started

        def resolution(base=None):
            base = best() if base is None else base
            return max(1024, math.ceil(base * 0.1 / 256) * 256)

        def bounds(base):
            lower = max(base // 256 * 256, floor - 256)
            upper = math.ceil(min(bad, slow) / 256) * 256
            return lower, upper

        def next_attempt(base):
            # Double from the last success while nothing above it has failed or
            # timed out, so the slowest probes come last; then bisect the range.
            if min(bad, slow) > ceiling:
                if not base:
                    return min(max(floor, selected), ceiling)
                return min(base * 2 // 256 * 256, ceiling)
            lower, upper = bounds(base)
            return ((lower + upper) // 2) // 256 * 256

        def resolved(base):
            lower, upper = bounds(base)
            if base >= ceiling:
                return True
            return upper - lower <= resolution(base)

        def publish():
            r["largest_observed_context"] = best() or None
            r["largest_started_context"] = started or None
            r["context_confirmed"] = bool(good)
            # Suggested headroom is an estimate, not another measured limit.
            recommended = math.floor(best() * 0.9 / 256) * 256
            r["recommended_context"] = recommended if recommended >= floor else None
            r["recommended_context_is_estimate"] = True
            r["context_ceiling"] = ceiling
            self.store.put("result", r["id"], r)

        publish()
        context_timeout = opts.get("context_timeout", 900)
        r["context_search_status"] = "running"

        def probe(kind, attempt, phase):
            nonlocal good, started, bad, slow
            if self.cancel.is_set():
                raise Cancelled()
            self.update(phase=phase, context_probe_started=time.time())
            entry = {
                "kind": kind,
                "context_per_slot": attempt,
                "status": "running",
                "started": stamp(),
                "timeout_seconds": context_timeout,
            }
            r["probes"].append(entry)
            self.store.put("result", r["id"], r)
            candidate = replace(s, context=attempt * s.slots)
            try:
                self.engine.start(candidate, self.cancel, context_timeout)
                started = max(started, attempt)
                if kind == "workload":
                    entry["sample"] = self.measure(
                        candidate,
                        "long-code",
                        attempt - opts["output_tokens"] - 32,
                        opts["output_tokens"],
                        context_timeout,
                    )
                    good = attempt
                entry["status"] = "success"
            except Cancelled:
                entry["status"] = "cancelled"
                raise
            except TimeoutError as e:
                entry.update(
                    status="timed_out", error=str(e), logs=self.engine.state()["logs"]
                )
                slow = min(slow, attempt)
            except Exception as e:
                logs = self.engine.state()["logs"]
                entry.update(status="failed", error=str(e), logs=logs)
                if isinstance(e, GPUUnavailable) or not hardware()["gpus"]:
                    raise GPUUnavailable(
                        "GPU unavailable after context probe; no further restarts."
                    ) from e
                # A load that fails for a reason other than memory would fail
                # at every size; stop rather than report a false memory limit.
                if kind == "startup" and not memory_exhausted(logs, str(e)):
                    entry["memory_limit"] = False
                    raise RuntimeError(
                        f"Engine failed to load {attempt:,} tokens per slot for a reason other than memory: {e}"
                    ) from e
                entry["memory_limit"] = True
                bad = min(bad, attempt)
            finally:
                self.engine.stop()
                entry["finished"] = stamp()
                publish()

        # Phase 1: load-only checks. Engine start allocates the cache and compute
        # buffers, so a refused load is a memory limit found in seconds. Loads
        # are cheap, so this ladder runs to resolution before any long prefill.
        checks = 0
        attempt = next_attempt(started)
        while not resolved(started) and floor <= attempt <= ceiling and checks < 12:
            checks += 1
            probe(
                "startup",
                attempt,
                f"Context load check {checks} (up to 12): {attempt:,} tokens per slot; {context_timeout}s timeout",
            )
            attempt = next_attempt(started)
        # Phase 2 (opt-in): confirm the largest loaded size with a full workload,
        # then narrow with workload probes only if that confirmation fails.
        attempts = 0
        attempt = started if started > good else next_attempt(good)
        while (
            confirming
            and not resolved(good)
            and floor <= attempt <= ceiling
            and attempts < 8
        ):
            attempts += 1
            probe(
                "workload",
                attempt,
                f"Context probe {attempts} (up to 8): {attempt:,} tokens per slot; {context_timeout}s timeout per operation",
            )
            attempt = next_attempt(good)
        if floor > ceiling:
            r["context_search_status"] = "skipped"
            r["context_search_stop_reason"] = (
                "Context ceiling cannot fit the minimum prompt and output budget."
            )
        elif best() >= ceiling or (resolved(best()) and bad <= slow):
            r["context_search_status"] = "complete"
        elif resolved(best()):
            r["context_search_status"] = "inconclusive_timeout"
            r["context_search_stop_reason"] = (
                f"Context probe timed out at {slow:,} tokens per slot; the usable-context limit above the largest success is unknown. Increase the context-probe timeout to continue testing. Successful probes remain valid."
            )
        else:
            r["context_search_status"] = "inconclusive_probe_limit"
            r["context_search_stop_reason"] = (
                "Stopped at the probe limit; earlier successes remain valid, but the usable-context limit is unresolved."
                + (
                    f" A probe at {slow:,} tokens per slot timed out; raising the context-probe timeout may help."
                    if slow <= ceiling
                    else ""
                )
            )
        r["context_search_resolution"] = resolution()
        r["context_failed_upper_bound"] = bad if bad <= ceiling else None
        r["context_timeout_upper_bound"] = slow if slow <= ceiling else None
        r["context_ceiling_reached"] = best() == ceiling
