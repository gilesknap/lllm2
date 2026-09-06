# Qwen performance and understandable defaults: implementation slices

Status: plan written; implementation slices not started.
Created 6 September 2026 from the user's request to save the review, plan small
slices, keep the UI understandable to novices, and ship useful tested RTX 3090
defaults. Research is in [RTX_PERFORMANCE_REVIEW.md](RTX_PERFORMANCE_REVIEW.md).
Read [HANDOFF.md](HANDOFF.md) for existing behavior and unresolved issues.

These root Markdown files are explicitly requested planning/research artifacts.
They do not start a documentation site, CI, release machinery or a test framework.
The original PLAN.md remains historical; this file defines the new work order.

## Intended outcome

A novice can choose a model, understand its main settings, launch with a useful
starting configuration, and learn what an experiment improved or sacrificed.
Prioritize Qwen3.8-27B dense and Qwen3.6-35B-A3B MoE on the actual 24 GB RTX 3090.
Performance gains must be measured separately for those two architectures.

The initial dense checkpoint is Qwen3.8-27B UD-Q4_K_S. The catalogue contains
several Qwen3.6 variants, including IQ4_XS with and without embedded MTP and
Q4_K_S. Inspect the actual selected file and saved results before establishing
the MoE baseline; prefer an already installed MTP-bearing variant for MTP trials.
Do not silently swap the user's selection or treat identical filenames in
different directories as the same checkpoint. Keep non-MTP variants usable.

Stay with llama.cpp CUDA first; retain Vulkan as a working choice with separately
qualified defaults. No CUDA result is evidence of a Vulkan gain.

## UI rules for every slice

- Keep the normal path short: model, recommended settings, context, Start.
  Put tuning controls in an Advanced section as they are added, with a concise
  visible summary of settings that differ from the recommendation.
- Use familiar names with technical names nearby: “Read the prompt (prefill)”,
  “Generate the answer (decode)”, “Conversation memory (KV cache)”, and
  “Guess ahead (speculative decoding)”. Avoid replacing every term with opaque
  marketing labels; the user should be able to connect the UI with engine docs.
- Every launch setting gets a short explanation: what it changes, likely benefit,
  main tradeoff, and when to leave the recommended value alone.
- Tooltips supplement visible essential information. Reasons a feature cannot
  run, the meaning of a context number, and unsaved/pending launch changes must
  remain understandable without hovering.
- Reuse the existing `.tip` implementation from experiment controls. Support
  keyboard focus, touch/click, Escape and viewport positioning. Dynamic feature
  rows need working help after every re-render; avoid duplicate handlers and IDs.
- Feature rows distinguish **supported**, **selected**, and **measured benefit**.
  An advertised engine flag is not proof of a speedup. Map raw statuses into
  plain language while preserving technical diagnostics in expandable detail.
- Do not add controls for every researched technique. Expose only the small set
  in the active slice, and hide irrelevant advanced fields without silently
  changing stored settings.
- Loading recommendations changes the form; it does not overwrite saved defaults
  or change a running model. Preserve that behavior and explain when restart is
  needed. Manual saved settings are not labelled “measured”.

Example copy, to refine during implementation:

| Setting/feature | Explanation |
|---|---|
| Context | How much text a request can hold, including the reply. More context needs more memory. The tested recommendation leaves some headroom. |
| Slots | How many requests can have their own state. Today the allocated context is divided between slots; this is not the number of CPU threads. |
| Flash Attention | A more memory-efficient way to calculate attention. Usually leave the tested setting on; it does not shrink the model weights. |
| KV cache | Stores information about text already read. Lower precision can make room for more text, but speed and accuracy can change. |
| MTP | The model guesses several next tokens and checks them together. It needs a compatible prediction head; longer guesses are not always faster. |
| Available, not measured | This model and engine meet the known requirements. We have not yet measured a benefit for this configuration. |

## Slices and dependencies

Each slice should finish as a working, reviewable change with a short handoff.
Start it in a fresh conversation/context if convenient. Do not automatically
execute the whole roadmap or launch an exhaustive GPU sweep in one session.

| Slice | Deliverable | Depends on | Status |
|---|---|---|---|
| 1 | Launch-setting tooltips and clearer feature availability | Existing panel | Not started |
| 2 | Baselines and portable, evidence-backed defaults | 1 | Not started |
| 3 | Prefill batch/microbatch controls and bounded comparisons | 2 | Not started |
| 4 | Target GPU sampling and concurrent-stream experiments | 2; use stable batch choice from 3 | Not started |
| 5 | Warm conversation/prefix-reuse measurements and controls | 2 | Not started |
| 6 | Prompt lookup combined with MTP for copying/editing | 2; stable settings from 3/4 | Not started |
| 7 | Separate K/V precision with context validation | 2; stable runtime settings | Not started |
| 8 | Validate combinations and ship Qwen 3090 recommendations | Completed relevant slices | Not started |

### Slice 1: explain what already exists

Add help for every launch control and every feature availability row. Use one
small source of UI descriptions so controls and feature rows use consistent names.
Explain model versus engine/backend, total/per-slot context, GPU layers, cache
precision, effort, MTP/DFlash/prompt lookup, draft length/cache and chat templates.
Describe effect on speed, capacity or behavior without promising gains.

Keep this slice about clarity: no new performance modes, changed defaults or GPU
benchmark runs. Existing experiment tooltips already provide styling/interaction;
extend their initialization so dynamically rendered feature rows work too.

Likely files: `lllm2/static/index.html`, with minimal capability wording changes
in `settings.py` only if necessary. Avoid changing status meanings/API contracts.

Done when a fresh user can find an explanation for every launch setting, identify
why a feature is unavailable, and distinguish the form from the running model.
Check keyboard/touch behavior, narrow layouts, rescan/model switching, and existing
form actions. A tooltip click must not toggle a checkbox or accidentally submit.

### Slice 2: evidence and defaults foundation

Inspect existing stored results read-only and identify usable measurements for
each exact Qwen checkpoint/backend. Pin the baseline engine identity and actual
launch settings. Fill gaps with a small baseline run rather than rerunning every
historical experiment. Keep inherited recommendations available, labelled as such.

Add portable built-in recommendation records with model/quantization identity,
backend, tested GPU, engine evidence, workload/budgets, date, settings and source
result references. Store compact evidence with recommendations so another user's
installation does not need our local SQLite database or absolute home paths.
Do not duplicate raw logs or build a reporting framework.

Preserve precedence: user-saved settings first, compatible measured built-in
recommendation next, inherited/generic fallback otherwise. Display that source.
New settings must default to existing behavior when loading old results/settings.
Preserve result provenance when “Use as default” is used; current code stores
only the settings and loses the reference. Keep manual saves distinct.

Check model/quantization/backend/hardware compatibility; do not claim an exact
tested result on a different engine. On changed builds retain qualification
information and recheck capabilities rather than discarding every useful value.
Unknown models and non-3090 GPUs still get conservative, clearly labelled fallbacks.

Done when built-in recommendations survive a clean installation, saved settings
still win, old data loads, and no unmeasured context estimate is labelled tested.

### Slice 3: prefill tuning

Add optional logical batch and physical microbatch settings. Omitted means engine
default, preserving existing launch behavior. Probe support and validate positive
bounds and microbatch <= batch. Record both requested and effective values when
available. Put numeric tuning in Advanced; explain “how much prompt is processed
at once” and the extra-memory tradeoff.

Start with a small CUDA microbatch sweep around 512 on each Qwen model. Hold
checkpoint, occupied/allocated context, slots, cache, MTP and output budget fixed.
Measure cold prefill, decode and peak memory. Screen at modest context, then test
only finalists at longer context. Avoid a Cartesian product of all batch values.

Done when results identify whether a change helps each model without silently
trading away context or decode speed. Promote only a measured compatible choice.

### Slice 4: CUDA execution overhead

Trial target backend sampling and concurrent streams separately, then combine
only if each helps. Draft backend sampling is already enabled in installed help;
do not present it as a newly discovered gain. Ordinary CUDA Graphs and the
`GGML_CUDA_GRAPH_OPT` streams switch are different mechanisms.

Expose only supported, useful controls in Advanced. An application-owned
environment override must apply to the child engine, be recorded with results,
and leave unrelated processes untouched. Confirm the exact installed build uses
the switch. Record sampling fallback/compatibility with the actual API requests.

Done when a small repeated comparison shows a benefit or a documented no-gain
decision. Do not retain a prominent novice control solely because a flag exists.

### Slice 5: warm conversations

Keep the existing cold benchmark unchanged. Add a separately labelled multi-turn
mode: first turn cold, append new text, change a suffix, switch away and back.
Use truthful processed/reused-token accounting; do not report cached tokens as
newly processed prefill throughput. Add streaming first-token and completion
timing where the engine supplies it reliably, and label other timings accurately.

Expose a small cache/checkpoint policy only after checking Qwen hybrid/recurrent
state behavior. Measure host memory as well as VRAM; no unlimited cache default
on this 32 GB host. A successful HTTP response is insufficient proof of reuse.

Done when each Qwen has a reproducible multi-turn result and cold/warm rows cannot
be confused. Include edited-history misses and cancellation. If a complete
streaming adapter would make the slice too large, split measurement support from
the cache policy and update this table before proceeding.

### Slice 6: copying and editing with MTP plus lookup

Add engine-qualified lookup combinations, initially `draft-mtp,ngram-cache` or
the best supported minimal equivalent, keeping ordinary MTP as the general
baseline. Represent allowed combinations explicitly; do not concatenate arbitrary
mode names or imply DFlash and MTP should run together.

Add a true source-copy/small-edit workload. Current `long-code` asks for review
and implementation; it is not the old long-copy benchmark. Keep newly generated
code, copying and broader editing results separate. Synthetic repetitive ledger
text alone can flatter lookup; include representative source with mixed repeated
and novel spans and hold prompts constant across comparisons.

Compare MTP alone, MTP+lookup at the same width, and only then width changes.
The historical 115.1 versus 94.0 tok/s dense result does not establish a MoE gain.
Account for lookup host memory. Put detailed method names/widths in Advanced;
offer “Copying existing code” only if a measured profile earns that label.

Done when gains and regressions are recorded for both targets. A copy win must
not replace a general default if generation/editing regresses materially.

### Slice 7: context through independent cache precision

Add independent K/V choices, initially a bounded set such as q8/q8 and q8/q4.
Expand types only when the selected engine/backend/architecture supports them.
An advertised type is a prerequisite, not proof that its attention kernel works.
Legacy shared `cache` settings map to both values; old results and saved defaults
must still load with unchanged meaning.

Keep the normal UI understandable with a linked common-precision choice and an
Advanced option to set K/V separately. Always show the actual pair for a saved
custom configuration; never silently convert it to a preset.

Compare memory and speed at identical occupied context, then run a separate
capacity probe for finalists. Include a small fixed long-context retrieval/edit
check before claiming a useful recommendation; no broad answer-quality
leaderboard. Record failures or unassessed quality explicitly. Avoid passing the
old q8-specific memory planner off as a measured estimate for new pairs.

Done when new pairs have clear evidence, backward-compatible settings, and honest
context labels. Do not switch the default to q4 merely because allocation fits.

### Slice 8: ship useful recommendations

Combine only promising settings from prior slices and rerun a bounded comparison
against each frozen baseline. Improvements are not assumed to add together.
Validate start, multi-turn use if enabled, occupied-context decode, and a context
recommendation with desktop/driver headroom. Timeouts remain inconclusive, not
memory failures. Preserve partial results and explicit cancellation.

Ship one recommended general configuration per tested checkpoint/backend. Add a
copy or larger-context alternative only when evidence supports a meaningful user
choice. Use portable profile identities and descriptions, not local file paths.
Keep saved user settings untouched. Label inherited Vulkan guidance if it has
not received new Vulkan measurements.

Done when a new 3090 installation gets the intended compatible defaults, existing
users keep their choices, and every claimed improvement links to reproducible
evidence. Document regressions and no-gain experiments alongside successful ones.

## Common validation and stopping rules

- For performance comparisons hold checkpoint, engine, prompt, output count,
  effort/sampling, context and slots fixed except for the declared variable.
  Repeat promising comparisons and report variability; do not promote noise.
- Begin with one slot for controlled single-request measurements without silently
  changing a user's normal saved slot setting. Slot count is not concurrency.
- Use short screening runs, then longer-context finalists. Do not run every
  option at every context. Show estimated run work and retain cancellation.
- No performance claims from syntax checks or model-load success. UI-only work
  needs UI checks; feature work needs appropriate runtime/benchmark checks.
- Current research and old benchmark numbers are hypotheses/evidence references,
  not newly completed lllm2 results. No accuracy or stability guarantees from a
  short synthetic probe.
- Preserve existing models, engine builds, old application state and desktop
  processes. Add an isolated engine build only as a deliberate comparison; do
  not overwrite the old installation. Keep software smoke checks lightweight.
- No commit/push is implied by this planning request. Keep each slice reviewable
  and update its status and evidence before handing off.

## Later work, after the Qwen phase

Concurrent 2/4/8-request benchmarking; alternative vLLM/ExLlamaV3/SGLang/ik engines;
Gemma external MTP; EAGLE3/DSpark/DFlash2 qualification; weight-quantization sweeps;
selective CPU offload; sparse attention/Lucebox/TurboQuant; YaRN beyond metadata;
multi-GPU and newer-RTX-specific FP8/FP4 work. None is needed for slice 1.

An isolated newer llama.cpp A/B comparison is a useful bounded experiment after
slice 2. Freeze any adopted build before tuning later settings; do not confound
an engine upgrade with a settings change.

## Starting a fresh context

Copy this prompt, adjusting the slice number:

> Read HANDOFF.md and PERFORMANCE_PLAN.md, then inspect the current code and git
> changes. Implement slice 1 only: tooltips for launch settings and clearer feature
> availability. Reuse the existing tooltip behavior and keep novice explanations
> concise. Preserve launch behavior and all existing defaults. Run appropriate
> lightweight checks, update the slice status and handoff, and stop at the slice
> boundary. RTX_PERFORMANCE_REVIEW.md is dated research, not measured local gains.

At each boundary record changed files, checks, result IDs/evidence where relevant,
remaining uncertainties, and the next slice. Prefer a new context at a completed
boundary; context compaction is not a reason to redo completed work.

The panel is currently managed as a detached process with PID file
`/tmp/lllm2-panel.pid` and log `/tmp/lllm2-panel.log`, bound to `0.0.0.0:8082` with
the user's explicit LAN authorization. Verify the PID still belongs to this app
before a restart; read live job status and do not interrupt a benchmark silently.
Code changes that need a restart should preserve this bind setting and verify
the LAN response at `http://192.168.1.10:8082/`.
