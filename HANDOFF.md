# Local Codex handoff — 2026-09-06

Read this alongside PLAN.md. PLAN.md records the original scope; its statement
that implementation has not started is historical. Giles has approved merging
iteration one (PR #1) and will continue locally in Codex. This handoff is an
explicit exception to the deferred-project-documentation rule.

## Scope and working preferences

- Keep the local web panel and rapid uv/Python iteration. Priorities are coding
  prefill/decode speed and empirically usable context, not answer-quality scoring.
- Target NVIDIA hardware, with both CUDA and Vulkan backends. Do not remove
  NVIDIA checks merely to support AMD/Intel Vulkan: that is outside agreed scope.
- No test suite, CI, installers, deployment or documentation infrastructure yet.
  Lightweight temporary smoke checks are appropriate; do not claim GPU validation
  without running on the workstation.
- Preserve LICENSE/NOTICE and the existing lllm3090 installation, models and state.
  Stop only app-owned engine processes. Do not kill desktop applications.
- Continue with reasonable implementation decisions; avoid repeated permission
  questions for already requested work. Inspect local changes before editing.

## Local startup

- Run `uv run python -m lllm2`; the panel defaults to loopback port 8082 and
  its engine to port 1920. Keep ports separate from other running applications.
- Discover actual GPU hardware, model locations and installed engine binaries
  locally. Do not assume a particular workstation layout or upgrade engines.
- Config overrides: LLLM2_MODELS_DIR, LLLM2_ENGINE_ROOTS (path-separated),
  LLLM2_STATE_DIR and LLLM2_ENGINE_PORT. See config.py for defaults.
- After pulling code, restart the panel and refresh the page. Cancel any old
  benchmark first; it does not adopt new code mid-run.

## Implementation map

- app.py: stdlib HTTP server, loopback/Host/Origin/token guards, APIs and local
  directory browser for selecting a drafter GGUF (no upload required).
- settings.py, discovery.py: launch settings, runtime capability/device probes,
  GGUF metadata, model/engine discovery. engine.py owns process lifecycle.
- defaults.py: tuned starting settings and context/slot planner distilled from
  lllm3090 revision e56c8d0fb19f76a254b5453bf7f206b229705d6a.
- bench.py: sequential baseline, individual feature variations, selected
  combinations, coding workloads, measurements and context search.
- store.py: SQLite results and saved defaults. static/index.html: inline UI/JS.
- downloads.py, gguf.py, models.json and templates/: catalogue/download support
  and inherited Qwen3.8 chat template.

## Defaults and fixes already made

- Giles explicitly wants the previously benchmarked lllm3090 defaults as starting
  points, rather than generic f16/4096. Main/draft KV cache q8_0, flash attention
  on, MTP where actual checkpoint and binary support it, draft length 3, effort
  default; calibrated hardware-aware context/slot estimates. These are inherited
  estimates, not new measurements. Saved lllm2 defaults take precedence.
- Launch context and context-search ceiling are separate. Ceiling comes from
  model metadata/catalogue, with generic 131072 fallback and safety caps; the
  Qwen3.8 model used here advertises 262144. Launch context comes from selected
  settings, normally the tuned planner or saved defaults.
- CUDA probes and engine launches share an environment which prepends the
  binary directory to LD_LIBRARY_PATH and removes inherited LLAMA_ARG_* options.
  Rescan clears stale probe results; the UI exposes raw device-probe diagnostics.
- GPU conflict checking allows recognized desktop apps (Chrome, Nautilus,
  desktop portals, Slack, VS Code, etc.). Unknown processes and other inference
  servers still block. Desktop VRAM/activity remains part of measurements.

## Context search: exact semantics and recent discussion

- Baseline is ONE configuration and gets ONE context search. A selected suite
  runs a separate search for each settings variant because memory usage changes.
  Workload repetitions do not each trigger a context search.
- First probe uses selected total launch context divided by slots, bounded by
  the permitted probe range. That starting value is NOT the search ceiling.
- Latest code immediately bisects the remaining bounds after success/failure.
  Older code doubled upward until failure/ceiling, then bisected. Neither used
  linear increments. Resolution is 256 tokens, with at most 24 probes.
- Each candidate restarts the server, fills a fresh long coding prompt and
  generates output. `prompt processing, n_tokens = 4096, 6144, ...` is progress
  THROUGH ONE prompt, not a context-size sweep. Restarting resets that counter.
- Speed-test timeout defaults to 180 seconds per operation. Separate context
  timeout defaults to 900 seconds per operation. Timeout stops that configuration's
  context search as inconclusive, preserving earlier successes; it is not a
  measured memory failure. A suite can proceed to the next configuration.
- Earlier bug treated 180-second timeouts as capacity failures and repeatedly
  narrowed around the time limit. Fixed before merge. Probe number/elapsed time
  and stop reasons are displayed.
- Largest observed success and a 90% headroom recommendation are separate;
  recommendation is an estimate. Full context validation can still be slow.
## Validation and follow-up

The cloud session performed Python/JS syntax checks, local HTTP/mock-engine smoke
checks, planner comparisons with the source, GPU process classification checks,
and simulated context searches (up/down/ceiling/no fit/off-grid seed/timeout).
There was no cloud GPU validation or completed browser automation. Local end-to-end
validation remains necessary for CUDA and DFlash. Do not equate capabilities with measured performance.

PR: https://github.com/gilesknap/lllm2/pull/1
CodeRabbit was explicitly triggered; reviews were COMMENTED, not blocking
changes-requested. Outstanding review findings were carried into this handoff,
not silently fixed or marked resolved during merge preparation:

1. downloads.py resumes a .part from a moving Hugging Face resolve/main URL.
   Offset/header validation cannot prove artifact identity. Prioritize pinned
   revision/validator handling and final digest verification before relying on
   interrupted downloads. Existing local model use is a separate path.
2. Qwen3.8 template accepts low/medium/xhigh and maps high to xhigh, but shared
   settings also allow minimal/max. Those currently fail template validation.
   Decide and consistently expose supported values/mappings.
3. capabilities() does not catch unreadable/deleted custom template files.
4. Empty LLLM2_ENGINE_ROOTS entries become Path('.') and can scan the checkout.
5. Context range with ceiling below the minimum prompt/output budget lacks a
   useful explicit skipped-range explanation.
6. Large token-id provenance is repeatedly serialized into results; polling can
   overlap and repeatedly fetch all summaries. These are performance follow-ups.
7. The review request to support AMD/Intel Vulkan contradicts NVIDIA-only scope;
   do not apply it without a scope change.

Review details: https://github.com/gilesknap/lllm2/pull/1/files

Issue #2 tracks a future concurrent coding-subagent benchmark and possible vLLM
comparison: https://github.com/gilesknap/lllm2/issues/2 . Current sequential tests
cannot establish vLLM's benefit for concurrency. Benchmark 1/2/4/8 requests,
shared-prefix and independent prompts, per-request latency, aggregate throughput
and usable context before adding another backend. It is not implemented yet.

## Suggested next-session starting prompt

Read PLAN.md and HANDOFF.md, inspect this checkout and my actual GPU/engine setup,
then continue iteration two with me. Preserve the agreed scope and existing
installations. Start by reviewing outstanding CodeRabbit findings and the current
workstation benchmark results; do not assume the cloud session validated CUDA,
DFlash or the final usable context limit.
