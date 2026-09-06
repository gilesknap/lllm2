# Qwen performance and understandable defaults: implementation slices

Status: all eight slices complete.
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
| 1 | Launch-setting tooltips and clearer feature availability | Existing panel | Complete (6 September 2026) |
| 2 | Baselines and portable, evidence-backed defaults | 1 | Complete (6 September 2026) |
| 3 | Prefill batch/microbatch controls and bounded comparisons | 2 | Complete (6 September 2026) |
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

Slice 1 completion — 6 September 2026:
- Changed `lllm2/static/index.html` only for implementation: shared descriptions
  for all 16 launch controls and six feature rows; reusable tooltip initialization;
  plain support labels, visible reasons, form choices and expandable diagnostics;
  visible pending changes versus the running model; clearer inherited-default copy.
- Preserved launch controls, options, initial values, settings serialization,
  capability status/API meanings, defaults and backend code. No performance modes
  or GPU benchmarks added; no new result IDs or performance claims.
- Checks passed: JavaScript syntax, `git diff --check`, comparison of original
  input/option attributes and serialization, temporary headless Chrome smoke checks
  at narrow and desktop widths (focus/click/Escape/outside dismissal, viewport fit,
  re-render/rescan/model switching, unique IDs/handlers, feature selection, pending
  state, mocked load/save/start/stop actions). Chrome emulated touch also passed.
- Real LAN page rendered all help and feature rows with no displayed error;
  real-page emulated touch/Escape passed without toggling the DFlash checkbox.
  HTTP 200 served the exact updated HTML. Static HTML is read on each request,
  so no panel restart was needed; `0.0.0.0:8082` was preserved. No active benchmark
  was present. The user stopped the model during validation.
- Limits: no physical touch-device or screen-reader check; launch/save mutations
  were tested with mocked APIs. No new GPU validation. Next: slice 2 in a fresh task.

Pre-slice-2 CUDA check: complete. The installed CUDA build detects CUDA0 and
passed a short real Qwen3.8 launch/generation. The UI now offers explicit selection
of a matching installed build when backend/binary choices disagree. Live browser
recovery passed for CUDA and Vulkan. No system changes or benchmark gain claimed;
see HANDOFF.md for configuration and evidence.

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

Slice 2 completion — 6 September 2026:
- Added `recommendations.py` and `recommendations.json`; updated `defaults.py`,
  `app.py` and the UI. Portable records match exact checkpoint SHA256, backend,
  GPU and template identity. Changed engine/driver/template carries qualification;
  incompatible settings fall back explicitly. Saved preferences still win.
- Result promotion retains compact evidence separately from legacy settings;
  manual saves clear that attribution. Old saved defaults remain unchanged with
  unknown origin. Edited forms visibly stop claiming the loaded measurement.
- Four new CUDA result records, 14 cold samples (256 output each), RTX 3090,
  installed b10715-cuda-sm86, q8_0 main/draft, flash on, MTP3, default effort:
  dense short `685605f4-33a5-4560-8a58-0df38ce8dfe1`, occupied
  `a2b4390a-ba5e-4460-ac29-5cf5977dbd6a`; MTP MoE short
  `5687730a-2582-4ad6-ac79-c71472898ee2`, occupied
  `1af11620-f398-4cc1-a132-28c7e37c6659`. Short runs cover generate/edit/long-code
  at 1024 input twice; occupied runs cover long-code at 16096 input once.
- Both portable starting allocations are 16384 total tokens, one slot. These are
  tested baselines, not speedup claims, searched maxima or headroom estimates.
  Prior dense Vulkan evidence remains qualified history; its saved 159744 setting
  is preserved. No CUDA/Vulkan comparison at mismatched allocations is claimed.
- Checks passed: independent code/data review; all portable metrics/settings
  matched saved records; Python/JS syntax; temporary compatibility/provenance
  checks; live exact-profile resolution for both files; actual non-MTP same-name
  rejection; legacy Vulkan preference preservation; narrow live browser measured,
  modified/reset and saved-source labels. Idle panel restart preserved environment
  and `0.0.0.0:8082`; LAN HTTP 200 verified. No model left serving.
- Limits: synthetic execution/capacity screening only, not answer quality or
  stability validation; no searched maximum. Next: slice 3.

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

Slice 3 completion — 6 September 2026:
- Added nullable logical/physical batch controls in Advanced; omitted/blank fields
  preserve engine defaults and older settings. Validate bounds, supported flags
  and microbatch <= logical batch where explicit/default values are known.
  Records distinguish requested, advertised-default and startup-observed values.
- Independent review and temporary compatibility/UI checks passed; live Advanced
  controls, help, narrow layout, null serialization and old-setting reset passed.
  Restarted idle panel, preserved LAN binding/environment, verified HTTP 200.
- CUDA screen: both exact checkpoints, context16384, one slot, logical2048,
  microbatch512/256/1024, long-code4096+256 twice each; MTP3, q8_0 caches,
  flash on and default effort held fixed. Actual argv matched every request.
  Runtime effective batch sizes were not logged at existing verbosity and remain
  explicitly unknown; installed advertised defaults are2048/512.

| Model / microbatch | Result ID | Prefill tok/s (two samples) | Decode tok/s (two samples) | Peak total GPU MiB |
|---|---|---|---|---|
| Dense /512 | `85a890c7-5bc8-4a1e-b331-c83b38a93735` |1057.9 /1059.7|54.7 /54.9|17172|
| Dense /256 | `09344445-bfc7-42f2-afce-70bd72cc083a` |1011.7 /1004.7|55.9 /56.0|17068|
| Dense /1024 | `7f304562-1ee1-42d9-afac-e317cb440d66` |1073.5 /1073.8|55.7 /55.7|17400|
| MTP MoE /512 | `a4063b75-d0b7-4367-bc86-60fd5e2b51c6` |2264.8 /2364.1|169.7 /173.0|19052|
| MTP MoE /256 | `0fb97db6-16f4-49f6-afd9-e74ccebc2293` |1717.7 /1718.6|174.5 /174.1|18992|
| MTP MoE /1024 | `b396a885-e628-4899-8e0d-50933845fc5e` |2914.0 /2903.5|177.7 /178.2|19204|

Promising MoE1024 was checked against512 in reversed order at context65536,
long-code65248+256, twice each:1024 result`9aa2ddf8-b287-4dc1-894a-98372824a005`
(2455.7/2445.6 prefill,110.0/110.0 decode,20148MiB peak);512
result`18d130e7-b748-4e7d-b15e-270dcc640930` (1992.3/1985.0 prefill,
116.8/117.9 decode,19906MiB peak). All16 samples completed.

Decision: preserve general2048/512 and shipped baseline defaults. Dense1024's
~1.4% prefill difference is insufficient to promote. MoE1024 is a prefill-oriented
candidate:~23.2% higher median prefill at the longer window, but~6.3% lower median
decode and242MiB extra peak GPU use. Do not hide that tradeoff or promote it as an
unqualified general win. Both long-window settings executed successfully; neither
is a searched maximum or quality/stability guarantee. Keep512 for controlled
slice4 general comparisons; revisit1024 as a qualified alternative in slice8.

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

Slice 4 completion — 6 September 2026:
- Added experimental target GPU sampling and tri-state concurrent CUDA streams in
  collapsed Advanced controls. Omitted sampling and default streams preserve
  existing launch behavior, draft sampling and inherited environment. Explicit
  streams overrides apply only to the child engine; defaults remain unchanged.
- The adjacent installed CUDA library contains the compiled streams switch.
  Exact build662a0b0 source confirms one visible device and ordinary CUDA Graphs
  are prerequisites. Streams are distinct from ordinary graphs. Sampling response
  settings confirm the request, but normal logs do not prove complete GPU offload;
  evidence retains that uncertainty and any observed fallback diagnostics.
- Records retain actual approved child environment and request evidence. Failed
  preflight cannot inherit the previous launch's evidence. Built-in profiles are
  qualified when inherited CUDA execution variables were not measured.
- Both exact CUDA checkpoints: context16384, one slot, logical2048/micro512,
  MTP3, q8_0 main/draft, flash on, default effort; long-code4096+256, twice each.
  Baseline explicitly disabled streams; each candidate changed only one option.

| Model / option | Result ID | Prefill tok/s | Decode tok/s | Peak GPU MiB |
|---|---|---|---|---|
| Dense / baseline | `795c6c15-fcd3-47eb-81da-197879184434` |1048.3 /1064.7|54.9 /55.3|17169|
| Dense / sampling | `f2f4940d-39ac-4173-9af9-052149409439` |1043.2 /1036.8|54.7 /54.7|17183|
| Dense / streams | `f289aa2b-b172-4065-a770-cd6f2fd8acda` |1046.1 /1046.9|54.4 /54.5|17165|
| MTP MoE / baseline | `9b171462-e73b-4fe6-9026-039a8a958774` |2287.0 /2354.4|171.5 /174.9|19049|
| MTP MoE / sampling | `e858a973-f148-4248-9109-0728367c8566` |2434.0 /2300.4|177.9 /178.7|19067|
| MTP MoE / streams | `9b47d963-b6e0-42d3-9e5d-f70f17b50115` |2350.6 /2334.0|174.0 /173.0|19049|

MoE sampling confirmation reversed order at4096+1024, twice each:
`5ee32831-4bc1-43a5-9fa5-0128bc86982f` sampling (2343.2/2329.8 prefill,
161.5/160.6 decode,19076MiB peak), then
`4a9061e3-235a-4568-b888-51febc80ef36` baseline (2364.5/2326.8 prefill,
156.7/156.5 decode,19049MiB peak). All16 cold samples completed.

Decision: sampling is a small workload-specific MoE candidate (~2.8% median decode,
~0.4% lower prefill in confirmation). Dense showed no useful gain; streams showed
no useful gain for either model. No combination was justified. Preserve profiles
and saved defaults; retain controls as collapsed experimental options, not novice
recommendations. Keep the baseline for slice5 and revisit sampling in slice8.

Independent review, temporary compatibility/environment/provenance/error checks,
Python/JS syntax and live narrow-browser Advanced/help/reset/modified-state checks
passed. Idle panel restarted with inherited environment and0.0.0.0:8082 preserved;
LAN HTTP200 verified. No engine left serving. Research remains distinct from local
measurements; see exact-source [streams implementation](https://github.com/ggml-org/llama.cpp/blob/662a0b0/ggml/src/ggml-cuda/ggml-cuda.cu#L4355)
and [sampling arguments](https://github.com/ggml-org/llama.cpp/blob/662a0b0/common/arg.cpp#L2303).

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

Slice 5 completion — 6 September 2026:
- Separate controlled token-prefix conversation mode runs six turns, then exact
  uncached replays: cold, append, suffix edit, switch away/back, early-history edit.
  One owned engine per repetition; existing cold benchmark/context search remains
  unchanged. Exact generated token IDs form the next prefix; no retokenization.
- Verified pinned build serialization: `timings.cache_n` is reused input,
  `prompt_n` is processed input; `tokens_cached` is final slot occupancy. Samples
  require processed+reused=input and complete output IDs. Upstream UTF-8 buffering
  can omit stream IDs; such runs fail with partial evidence instead of guessing.
- Bounded native SSE records client first-token event, first text and completion
  separately from engine timings. Memory evidence samples owned-process RSS,
  anonymous/swap/available RAM and GPU use. RSS is not cache allocation. Errors,
  cancellation and latest-launch-only diagnostics retain partial evidence.
- Nullable finite cache/checkpoint Advanced controls preserve normal launch
  defaults (installed8192MiB/32). Warm-only blanks resolve to measured2048MiB/4;
  explicit0 and512 remain valid. These are not total RAM caps or a larger-context
  reuse guarantee. Warm results cannot be promoted as general cold baselines.

Both exact CUDA checkpoints used16384 allocation, one slot, logical2048/micro512,
MTP3, q8_0 main/draft, flash on, default effort, target sampling omitted and streams
explicitly off. Initial input4096, each output128; appended/edited input4240.
Six records contain96 requests (48 conversation turns and48 exact cold controls):

| Model | Cache MiB / checkpoints | Repeats | Result ID |
|---|---|---|---|
| Dense |512 /4|1|`4ec9dd89-8ec8-411c-b699-b2ff83699020`|
| MTP MoE |512 /4|1|`e54a0ab2-c091-4e15-acdc-eeda35c718b3`|
| Dense |0 /0|1|`3e56eeb2-8fd4-4344-bfb6-3cda32d3e075`|
| MTP MoE |0 /0|1|`b868ae82-1419-416f-b721-53bdf149704a`|
| Dense |2048 /4|2|`ca55808b-8ad1-4d5d-b878-70c6e1fbe7ba`|
| MTP MoE |2048 /4|2|`4d4dfdad-9cfa-43bb-a175-27f28e0e4e41`|

The512MiB pilot reused append/suffix prefixes but could not restore switch-back.
Dense saved states (~777/614MiB) exceeded the cap; MoE evicted the prior entry.
The0/0 control retained immediate append reuse but replayed edited/switch-back
prompts. With2048/4, both repetitions of both models restored switch-back.
All uncached controls processed their full input; early-history edits still missed.

| Model / turn (2048 /4) | Processed / reused input | Median first-token event seconds, warm / uncached | Median completion seconds, warm / uncached |
|---|---|---|---|
| Dense / append |17 /4223|0.257 /4.074|2.565 /6.396|
| Dense / suffix edit |660 /3580|0.875 /4.105|3.354 /6.346|
| Dense / switch back |4 /4236|0.540 /4.467|3.019 /6.740|
| Dense / early-history edit |4240 /0|4.374 /4.468|6.601 /6.700|
| MTP MoE / append |17 /4223|0.128 /1.741|0.969 /2.598|
| MTP MoE / suffix edit |660 /3580|0.417 /1.794|1.250 /2.535|
| MTP MoE / switch back |4 /4236|0.214 /1.942|1.053 /2.681|
| MTP MoE / early-history edit |4240 /0|1.915 /1.940|2.762 /2.797|

At2048/4, sampled peak process RSS was2735MiB dense /1875MiB MoE; total GPU
peaks17171/19055MiB, minimum post-turn available host RAM24077/25160MiB.
At0/0 RSS peaks were650/745MiB. Checkpoint/cache policy trades RAM and overhead
for edited/switch-back reuse; no general decode-speed claim or normal-default
promotion. Memory includes working buffers/mapped pages, not just saved states.

Real cancellation `cae65eb2-f663-44a7-832c-38f2493b8377` retained140 output IDs
and stopped the owned engine in0.59seconds, with no next turn. Independent review,
mocked bad/missing counts, timeout/error/dribbling SSE/cancellation/late callbacks,
legacy defaults and launch-log isolation checks passed. Full saved token vectors
matched their replay controls; saved preferences remained unchanged. Live browser
and legacy cold smoke are recorded in HANDOFF.md. No model left serving.

Exact-source evidence: [timing serialization](https://github.com/ggml-org/llama.cpp/blob/662a0b0/tools/server/server-common.cpp#L67),
[finite policy flags](https://github.com/ggml-org/llama.cpp/blob/662a0b0/common/arg.cpp#L1695),
[stream token buffering](https://github.com/ggml-org/llama.cpp/blob/662a0b0/tools/server/server-context.cpp#L1828).

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

Slice 6 completion — 6 September 2026:
- Added one explicit `draft-mtp,ngram-simple` combination, paired nullable lookup
  match N/draft M in Advanced, MTP/draft-cache eligibility and provenance. Legacy
  modes omit new flags; the explicit combination resolves blank N/M to3/3. The
  verified build prioritizes lookup, with MTP fallback, rather than concatenating
  drafts. Ordinary HTTP acceptance counters aggregate methods, not MTP alone.
- Chose controllable `ngram-simple`: `ngram-cache` hardcodes8 independently of MTP
  width; simple N must not exceed M. Do not expose its ineffective min-hits flag.
- Added complete fixed-source copy and single-condition edit tasks, preserving
  the entire excerpt/instructions and padding only unrelated reference context.
  Source tasks use at least2048 output tokens as a cap and stop naturally; raw
  reasoning/output is retained. Exact final source is checked after optional
  thinking/fence removal; wrong/truncated results cannot establish a copy win or
  be promoted. Existing generation budgets/EOS and all defaults remain unchanged.
- User-requested **Reset experiment settings** restores experiment HTML defaults
  and the selected model's context ceiling. It makes no API call and preserves
  launch settings, saved defaults, results, queued comparisons and current jobs.

Two initial MTP source-task pilot records passed four adherence checks:
dense`90c673b3-f031-4b71-b4e0-3696bb3d7053` (705/643 output tokens),
MoE`f938a920-1fba-43f5-918c-f04f7a92f8dc` (1896/1671 output tokens).
Then24 equal-width samples compared MTP3 with MTP3+N3/M3, followed by six dense
N3/M6 samples after the small-edit signal. All34 samples completed; all24 source
answers passed exact adherence. Same exact input vectors within each workload
and checkpoint:4096 tokens, context16384, one slot, logical2048/micro512,
q8_0 main/draft, flash on, default effort, target sampling omitted, streams off.
Generate output256 fixed; source cap2048 with natural completion. Each screen/
width point repeated twice with a fresh engine for every sample.

| Model / method | Result ID | Generate median decode (actual outputs) | Copy median decode (actual outputs) | Small-edit median decode (actual outputs) | Peak GPU / sampled process RSS MiB |
|---|---|---|---|---|---|
|Dense / MTP3 + lookup N3/M3|`e0934ddb-20d1-4d00-8b67-7cbf57f4570a`|47.96 (256/256)|70.85 (705/705)|76.33 (644/644)|17239 / 1683|
|Dense / MTP3|`1f5b40fc-91ec-430b-a192-ba512bc4bdfd`|50.33 (256/256)|69.42 (705/705)|69.72 (643/643)|17316 / 1682|
|MTP MoE / MTP3 + lookup N3/M3|`4265ba41-aca4-4b07-aaf1-200571619d35`|147.38 (256/256)|180.58 (1609/1609)|180.46 (1670/1670)|19153 / 1584|
|MTP MoE / MTP3|`b41bc78f-050a-4c59-9667-49f41a9d2383`|154.49 (256/256)|189.82 (1896/1896)|185.87 (1671/1671)|19158 / 1583|
|Dense / MTP3 + lookup N3/M6|`ca37264c-c2f6-4d35-8bab-ac25e194e6f2`|46.43 (256/256)|65.69 (994/994)|63.82 (640/640)|17316 / 1833|

Decision: preserve ordinary MTP3 and all profiles/saved preferences. Dense N3/M3
small-edit median decode improved~9.5% (643 versus644 output tokens), but generate
fell~4.7%; copy's~2.1% median change was within observed variation. Keep only a
narrow small-edit candidate for slice8, not a copying/general recommendation.
Dense M6 widened lookup while holding MTP3: generation and both source decode
rates regressed, and copy emitted substantially more reasoning. Stop that path.
MoE N3/M3 median decode regressed~4.6% generate,~4.9% copy and~2.9% small edit.
MoE copy completed sooner because it emitted1609 instead of1896 tokens; that is
not an equal-work lookup speedup. Final copied source matched in both cases.
Do not promote these shorter-reasoning results as a general performance gain.

Independent review, temporary capability/legacy/pair/prompt/extraction/limit/
promotion checks, full stored-vector/argv/adherence verification and live narrow
browser controls/help/reset/results checks passed. Reset covered every experiment
parameter, source checkboxes, model ceiling, dependent disabled states, no network
calls and preserved simulated running-job/launch/results state. Source tasks no
longer inherit the old fixed-output time estimate. Saved preferences unchanged.
Idle panel restart preserved environment and0.0.0.0:8082; final LAN browser check
passed and no engine remains serving. Source evidence: pinned [speculative selection](https://github.com/ggml-org/llama.cpp/blob/662a0b0/common/speculative.cpp)
and [lookup implementation](https://github.com/ggml-org/llama.cpp/blob/662a0b0/common/ngram-map.cpp).

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

Slice 7 complete (6 September 2026). Independent nullable K/V overrides preserve
legacy shared-cache settings and draft cache. Explicit common selection relinks
both; form, running settings and results display the resolved pair. Quantized
pairs retain the flash-attention guard. Exact adjacent CUDA library identity and
compiled-kernel limitations are recorded separately from unobserved dispatch;
other builds remain unverified. Hybrid recurrent states are not re-quantized.

Four cold CUDA runtime probes used identical 4096-token long-code input and32
output tokens at16K allocation, one slot, MTP3, batch2048/512, sampling off and
streams off. Baseline q8/q8 versus q8/q4:

| Checkpoint | Prefill tok/s | Decode tok/s | Wall seconds | Peak total GPU MiB |
|---|---:|---:|---:|---:|
| Dense q8/q8 |1037.2|47.10|4.61|17199|
| Dense q8/q4 |90.1|18.22|47.16|17155|
| MoE q8/q8 |2278.4|130.94|2.04|19201|
| MoE q8/q4 |236.9|64.32|17.77|19070|

Dense result IDs: `6e720118-fe93-46ff-92c0-4540f52be03f` baseline and
`b1a9e8f8-1aec-47a9-a997-40eb21a1c0d2` mixed. MoE IDs:
`35b8b2dd-a2e5-4b39-9e8c-6d3ff8b18328` baseline and
`13c6271c-417d-4d69-a698-3dcb0fa0394c` mixed. All completed. One screening
sample each establishes a severe regression, not a precise repeatable percentage.
The slowdown is consistent with missing mixed CUDA kernels; runtime dispatch was
not instrumented. No useful finalist, so no larger-capacity or quality trial was
warranted. Mixed-pair quality and maximum capacity remain unassessed. Keep q8/q8.

Temporary compatibility tests covered all nine type pairs, legacy defaults,
invalid overrides, flash guards, draft isolation, suite variants clearing overrides,
provenance and changed-library hash invalidation. Independent review passed. Live
390px LAN browser verified collapsed controls, resolved pair, experimental warning,
common/legacy reset, actual result rows, experiment-reset isolation and no overflow.
The transient Unknown launch setting error came from new static HTML with the old
running parser; an idle restart and user hard refresh resolved it. LAN8082 remains
available with the original environment and bind. No default changes or engine
upgrade. Detailed negative evidence is in the four stored results.

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

Slice 8 complete (6 September 2026). Fresh occupied-context CUDA runs hold65,536 total
allocation, one slot, q8/q8 main and draft cache, MTP3, default effort, flash on,
batch2048, streams explicitly off,65,248 input+256 generated tokens, two fresh
engine starts per configuration. Frozen16K baseline records remain unchanged.

| Configuration | Prefill tok/s (two runs) | Decode tok/s (two runs) | Peak total GPU MiB |
|---|---|---|---:|
| Dense baseline, micro512/sampling off |898.51 /893.42|52.12 /52.39|19459|
| MoE baseline, micro512/sampling off |2197.24 /2199.28|132.03 /131.62|19955|
| MoE candidate, micro1024/sampling on |2297.61 /2276.38|118.34 /111.46|20218|

IDs: dense `22f82ddb-fa99-47d2-9a3f-0ea90eff3084`, MoE baseline
`ba337bef-b73e-4d2f-bb4b-e892898550ec`, MoE combination
`60400368-f8c6-4b41-9e87-89fdbada91a0`. The four MoE samples have identical
stored input vectors and output budgets. Combined median prefill+4.04%, decode
-12.84%, elapsed-2.74%; candidate uses263MiB more peak device memory. Reject as
a general recommendation. Prior isolated gains did not add together. Dense stays
with ordinary MTP3; its lookup small-edit gain did not earn a general/copy profile.
No mixed-cache finalist or new Vulkan measurements.

The new reproducible context-retrieval-edit task disperses three fixed facts early,
middle and late, then asks for a complete precise source edit. It uses natural EOS,
at least2048 output cap and exact adherence, never executes generated code, and
retains token vectors/fact offsets/raw output. Its first wording ambiguously asked
to replace variables with values: dense preliminary `cb396cd0-7a63-412d-afae-0079daaf5300`
passed, MoE `63a072f8-0eae-4473-8298-a8c7817ee2b8` exhausted2048 output tokens
while discussing that ambiguity. This is not a measured memory failure or a pass.
Version2 explicitly changes only assignment right-hand-side zeros, preserving
variable names. Both checkpoints were rerun; preliminary evidence stays recorded.

Version2 passed exact source adherence on both checkpoints with61,440 input tokens
and natural EOS below the2048 output cap. Dense
`4a578132-a306-4ab7-8fa2-5de33c6756fc` generated414 tokens; MoE
`80edb2bb-a987-446c-b8c3-3dd969793c68` generated1099. Fact offsets are recorded
near the beginning,30.7K and61.3K tokens. This validates only three fixed facts and
one small exact edit amid repeated reference text, not general accuracy/stability.

Final short-conversation validation completed at65K allocation, one slot and
normal installed cache policy explicitly requested as8192MiB/32 checkpoints. The
profile leaves those normal launch flags omitted; the warm experiment's separate
blank2048/4 policy is unchanged. Each model ran six turns plus six exact uncached
replays,4096 initial/4240 subsequent input and128 generated tokens. Dense
`b32af64c-f21b-492e-a30a-0576846abb3d`; MoE
`4a61cbcd-940b-465b-989e-141df4c1dd1c`. Both reused4223 tokens on append,
3580 on suffix edit and4236 on switch-back; early-history edits reused zero.
All controls reused zero; stored vectors matched each paired turn exactly.
Dense peak process RSS2338MiB; MoE1517MiB. This checks short multi-turn operation
at the new allocation; warm reuse near a full65K conversation remains unverified.

Across all baseline-setting final/preliminary requests, minimum sampled free GPU
memory was4975MiB dense and4595MiB MoE, including the running desktop. The rejected
MoE combination is excluded from the general-profile reserve. Sampling covers
request execution, not startup peaks; repeated fresh startup succeeded. Recommend
65,536 total tokens with one slot for these exact CUDA checkpoints/build. This is
a useful tested allocation with observed reserve, not a searched maximum or a
promise for other apps/workloads. No general speedup is claimed. Preserve q8/q8,
MTP3,2048/512 and target sampling off; explicit streams off reproduces the trials.

There were34 completed requests in nine slice8 records, including the retained
preliminary MoE adherence failure. The final v2 quality checks and all24 streamed
warm/control requests passed. Portable general-profile identities preserve the
original16K records unchanged under baseline, with compact final measurements,
actual budgets, prompt digests/fact offsets, observed reserve and rejected trials.
Changed engine/driver/template/adjacent CUDA library qualifies the evidence.
Saved preferences and inherited Vulkan guidance remain separate and unchanged.

Final validation: temporary workload/adversarial, compatibility, portable identity,
saved-default precedence and experiment-reset checks passed; Python/JavaScript
syntax and diff checks passed. Independent reviewer reconciled every compact
sample, settings/options, prompt digest, adherence and decision median against
read-only SQLite records. Live390px Chrome checked both65K CUDA profiles, frozen
baseline/observed-reserve evidence, the existing159744 Vulkan saved preference,
all four quality rows including the unpromotable preliminary failure, experiment
reset isolation and no horizontal overflow. Idle restart preserved environment
and0.0.0.0:8082; LAN HTTP200 verified. All eight slices are complete; stop here.

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

Copy this prompt:

> Read HANDOFF.md and PERFORMANCE_PLAN.md, then inspect the current code and git
> changes. Implement the next unfinished slice only, following its scope and
> validation criteria above. Keep novice explanations concise and preserve existing
> behavior except for changes explicitly required by that slice. Run appropriate
> checks and bounded measurements, update the slice status and handoff, and stop
> at the slice boundary. RTX_PERFORMANCE_REVIEW.md is dated research, not measured
> local gains.

At each boundary record changed files, checks, result IDs/evidence where relevant,
remaining uncertainties, and the next slice. Prefer a new context at a completed
boundary; context compaction is not a reason to redo completed work.

The panel is currently managed as a detached process with PID file
`/tmp/lllm2-panel.pid` and log `/tmp/lllm2-panel.log`, bound to `0.0.0.0:8082` with
the user's explicit LAN authorization. Verify the PID still belongs to this app
before a restart; read live job status and do not interrupt a benchmark silently.
Code changes that need a restart should preserve this bind setting and verify
the LAN response at `http://192.168.1.10:8082/`.
