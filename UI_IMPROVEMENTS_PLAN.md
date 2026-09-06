# UI improvements: open the panel, start a recommended model

Status: proposed; implementation not started.
Requested 6 September 2026. This PR contains this plan only.

## Outcome and scope

The panel's primary job is to launch a good local model. On an idle, configured
RTX 3090 installation, opening it selects the highest-ranked compatible installed
recommendation, loads its recommended settings, and presents **Start Model** as
the obvious next action. No settings editing, defaults-loading button, experiment,
or additional confirmation is required for that normal start.

Keep the current visual language: restrained green accent, cards, typography,
light/dark themes, useful help, and responsive layout. Change information hierarchy
and selection behavior. Experiments remain a substantial capability, reached through
secondary navigation. The user is happy with the UI's general appearance.

This request establishes a launch-first direction beyond the historical workbench
emphasis in PLAN.md. It explicitly authorizes this root planning artifact and its
PR. It does not implement the plan or authorize a new performance sweep. Another
agent owns the remaining PERFORMANCE_PLAN work; consume its final recommendations
when implementing this plan and leave its code, measurements, and handoffs alone.

## Review basis and limits

Reviewed worktree revision `54a61a0`, especially `lllm2/static/index.html`,
`discovery.py`, `models.json`, `app.py`, `defaults.py`, `recommendations.py`,
`recommendations.json`, `engine.py`, and the performance/handoff documents.
PERFORMANCE_PLAN's completion notes cover slices 1–7 at this revision; some earlier
status-table entries lag those notes. Slice 8's final settings remain a dependency,
not a reason to duplicate its work here.

Rendered the actual worktree HTML in isolated headless Chrome at 1440×900 and
390×844, with its startup API calls disabled. Inspected both screenshots and
measured element positions. Start begins about 1,460px down on desktop and 2,026px
on the narrow screen; Experiments begins at 112px on desktop. These are static
layout observations, not a live launch test: loaded recommendations, feature rows,
and status text can change the geometry. Neither viewport had horizontal page
overflow. No model was started, downloaded, stopped, or benchmarked for this review.

This is an expert usability/code review, not a user study or a new model-quality
comparison. Model ranking below is an explicit product choice, informed by the
project's tested checkpoints; it is not an assertion of universal model superiority.

## Findings, in priority order

| Priority | Current behavior and evidence | Effect on the desired journey | Decision |
|---|---|---|---|
| P0 | The first card asks for checkpoint, binary path, backend and GPU device. The second exposes context, slots, GPU layers, attention, cache, effort, speculative method, draft settings, template, and DFlash pairing before Start. See HTML markup and `launchTips()`. | Starting appears to require engine expertise. A novice cannot tell which choices are already handled. | One compact launch card; automatically resolve setup and settings. Put configuration behind Customize. |
| P0 | Desktop `.grid` gives Experiments 1.2 fractions against launch's 1 fraction. Run baseline is visible while Start is below the viewport. | The most prominent available action starts a benchmark rather than serving a model. | Launch is the initial view; Experiments is a secondary view with its own primary action. |
| P0 | `discovery.models()` sorts filesystem paths; `scan()` selects the first returned model on first load. Model options show directory/file names, without ranking or reasons. | Installation names determine the default. Reordering `models.json` alone will not fix installed-model selection. | Explicit, shared recommendation rank; concise model names and identity-safe matching. |
| P0 | `engines()` sorts paths and `scan()` takes the first engine. The backend initially says CUDA, with a separate recovery button when the chosen binary cannot use it. | The default path can fail despite a compatible installed engine. | Select a compatible model/profile/engine/device together; preserve intentional custom overrides. |
| P0 | Startup and defaults resolution are asynchronous. `poll()` enables Start whenever no job is active, regardless of discovery, capability/default resolution or a pending start POST. | A fast click can submit placeholder, stale, or partly updated settings. Backend validation helps but does not make this a dependable one-click launch. | Explicit readiness state, immediate pending-action guard, and atomic resolved launch snapshot. |
| P1 | `loadDefaults('auto')` loads saved settings before built-ins. Several similarly prominent load/reset/save buttons and a long paragraph explain their differences. | An old experiment preference may be used when the user expects today's recommendation; “default” has several meanings. | Recommended is the initial idle launch mode. Saved settings remain preserved and explicitly selectable, with their real source shown. |
| P1 | `Engine.start()` stops the owned engine before loading another; the UI says stop/start to apply changes, but Start remains enabled while serving. | A button still labelled Start can actually interrupt and replace a running service. | Running, restart and model-switch actions must say exactly what they do. |
| P1 | Queue & engine status sits below Experiments. `engine.running` means a live process; `/api/start` only reports Ready after the health check. | Loading, ready, failure and benchmark progress are hard to distinguish near the action. | Put lifecycle status next to Start and use health-qualified readiness, not PID existence. |
| P1 | Downloads are under Paths & downloads; successful downloads are rendered but `poll()` does not rescan installed models. | A fresh installation has a hidden next step and may need manual Rescan after downloading. | A visible setup state for recommended models, followed by automatic discovery/resolution on completion. |
| P1 | Experiments use the launch form's `settings()` directly; result promotion fills the same form and saves defaults. | Exploring settings or results can change the next normal launch in a way the new navigation would conceal. | Separate experiment configuration from the launch draft, with explicit apply/save actions. |
| P2 | Feature availability is expanded by default; context/cache/batch notes mix serving and benchmark explanations. Metadata context competes with usable allocation. | Honest technical detail becomes mandatory reading and can imply the wrong usable context. | Compact recommendation/context summary; collapsed diagnostics and evidence; visible actionable blockers. |
| P2 | The endpoint is plain header text, including while idle, and points to loopback on the server workstation. | Users lack a clear “ready to use” outcome, especially when viewing the panel over LAN. | Ready status plus Copy API address with correct workstation scope. |

The existing UI already supplies useful building blocks: accessible tooltip behavior,
source/provenance tracking, pending-versus-running settings, collapsible advanced
controls, cancel/progress, and detailed saved evidence. Preserve these capabilities
while shortening the normal path. Moving Start alone would leave selection and
readiness defects unresolved; adding more explanation would leave the hierarchy
problem unresolved.

## Recommended models and their order

Use these two as the initial curated list for the project's 24 GB RTX 3090 target:

| Rank | Card label and short reason | Exact catalogue choice | Basis and qualification |
|---|---|---|---|
| 1 | **Qwen3.8-27B — Recommended**. “Our default starting model.” | `qwen3.8-27b`; `unsloth/Qwen3.8-27B-GGUF`; `Qwen3.8-27B-UD-Q4_K_S.gguf` | Matches the requested Qwen3.8-first direction and the dense checkpoint with portable local evidence. General first choice is a product decision, not a locally measured quality win over Qwen3.6. |
| 2 | **Qwen3.6-35B-A3B — Faster alternative**. “Faster generation in our tested coding workloads.” | `qwen3.6-35b-a3b-mtp`; `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`; `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | This is the Qwen3.6 variant repeatedly exercised by PERFORMANCE_PLAN, with embedded MTP and a portable profile. Limit the speed claim to those workloads/hardware; no exhaustive quantization or answer-quality ranking was performed. |

Keep “MTP build · IQ4_XS” in the second card's details and variant picker, so users
can identify their actual file without having to understand MTP to choose it.
The catalogue also includes non-MTP IQ4_XS and Q4_K_S Qwen3.6 files. Keep them in
**All models**, usable with correctly qualified settings; do not present three
near-duplicate Qwen3.6 variants as equally recommended or silently replace them.

The MTP and non-MTP IQ4_XS files share a basename. Rank/display identity must
include the catalogue/repository variant and actual checkpoint identity. Reuse
existing profile digest/metadata checks before assigning a measured settings badge;
a filename, parent directory, or `mtp: true` catalogue hint alone is insufficient.
Custom installation directories must work. Show a path/variant disambiguator when
multiple installed files would otherwise have identical labels.

Store rank and short recommendation copy once, preferably as additive catalogue
metadata. Use it for installed selection, recommended cards, and downloads. Keep
settings/evidence in the existing portable recommendation records, not copied into
HTML. Don't rank by benchmark tok/s alone, model-name version, file path, or size.

Publisher references checked during this review: the [Qwen3.8 GGUF model card](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
and the [Qwen3.6 MTP GGUF model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF).
They identify the upstream distributions; local PERFORMANCE_PLAN evidence and
portable records determine the usable quantization/settings here. Do not substitute
a publisher's example quantization or advertised context for the tested profile.

### Deterministic automatic selection

1. On a fresh idle page, select rank 1 if installed with an eligible configuration;
   otherwise select rank 2 if eligible. If rank 1 is absent, leave it first in the
   curated list with its download action and explain why rank 2 is selected.
2. Resolve the model's matching built-in profile, compatible installed engine,
   backend, and device as one selection. Prefer the qualified CUDA configuration
   on the target 3090. Use deterministic compatibility/evidence-based tie-breaking
   between eligible builds, not lexical order or “newest must be best.”
3. If there is no eligible curated model, show why. Offer installing one or choosing
   an installed model under All models. Do not label an arbitrary smaller model as
   measured/recommended merely because it sorts first or might fit.
4. Once the user explicitly selects a model or edits setup, retain that selection
   through polling, navigation, and Rescan while it exists. An incompatible explicit
   selection stays visible with its remedy; do not silently launch a different model.
5. On reopening while a model is loading/running, show that actual operation/model
   first. Never replace it with rank 1 just because the page loaded. A separate next
   launch draft can be chosen deliberately without claiming it is already running.

For other GPUs or Vulkan, preserve working choices and existing qualification
rules. Distinguish **tested recommendation**, **qualified recommendation** and
**estimated starting settings**. A known incompatibility blocks launch; lack of an
exact historical benchmark alone need not block a capability-valid fallback. Do not
promise the 3090 one-click measured result on unsupported hardware or absent setup.

## Proposed information hierarchy

The following is a structural sketch, retaining current styling. Actual context,
engine, and source text must come from the resolved profile.

```text
lllm2                                 GPU / service status
Run a local model. Explore performance when you need to.

Launch model                 Experiments
---------------------------------------------------------
Choose a model
(o) Qwen3.8-27B               Recommended · Installed
( ) Qwen3.6-35B-A3B           Faster alternative · Installed
All models…

Recommended settings         Tested on RTX 3090 / CUDA
Context: <resolved tokens> per conversation · <source>
[ Start Model ]               Ready to start

Customize settings ▸         Recommendation details ▸
Manage models & engines ▸
```

Use a compact native radio group or equivalently accessible selection for two
recommendations, rather than large marketing cards. All models opens the complete
installed/catalogue selection with variants and identity details. Primary action,
selected model, settings source, and blocking reason must be visible together.

Use two views, **Launch model** (initial) and **Experiments**, within the current
single page. Avoid a framework/router rewrite. Support keyboard navigation,
accessible selected-view state, and stable deep links such as `#experiments` for
intentional return visits. Ordinary opening without an explicit deep link lands on
Launch. Switching views must preserve drafts, comparisons, active jobs and results.
A compact global “Experiment running · View progress” indicator remains visible
when the user is on Launch.

### Launch settings disclosure

Normal view shows model selection, recommendation source, concise context summary,
Start/status, and disclosures. No editable engine, cache, MTP, slots or template
fields are required to start. Context summary states usable allocation per
conversation and, when relevant, total allocation divided across slots. A metadata
maximum is never the headline recommendation. Context editing is one disclosure
away; extra context presets appear only if final performance evidence supports them.

One **Customize settings** disclosure contains the existing controls in coherent
sections: conversation/context and reasoning; engine/backend/device; acceleration
and memory; speculative methods; advanced execution/reuse. Preserve exact stored
values, nullable overrides, and the ability to restore the recommendation. Show
DFlash file/pairing fields only when DFlash is selected. Do not erase their draft
values just because the section is hidden. Keep mixed K/V and experimental methods
visibly qualified when active; a compact custom-change summary stays near Start.

Collapse full feature availability and raw recommendation evidence. Surface a
launch-blocking explanation and its corrective action beside Start regardless of
whether diagnostics are expanded. Supported, selected, and measured benefit remain
separate concepts; ordinary users need not read the full feature matrix to launch.

### Recommended, saved, custom, and running are distinct

The desired fresh idle experience intentionally changes the current automatic
saved-first behavior **for the Launch view**:

| State/action | Required behavior |
|---|---|
| Fresh idle opening / new explicit model selection | Load compatible built-in recommended settings automatically. No manual Load step. If falling back, label the actual source. |
| Existing saved defaults | Preserve values and provenance in storage. Offer **Use my saved settings** beside the settings summary; expose that choice only when saved settings exist. Never relabel them as the current recommendation. |
| First visit after this change, with saved settings | Briefly explain: “Recommended settings are selected. Your saved settings are still available.” This is informational, not a blocking migration wizard. |
| Use my saved settings | Load them into the launch draft, validate against the selected setup, and label **My saved settings** with any qualifications. This does not overwrite the built-in profile. |
| Manual edit | Label **Custom settings** with a short list of differences and **Use recommended settings**. Preserve draft across tabs/rescans; no automatic overwrite from a late response. |
| Save my settings | Explicitly persist only this model/backend's chosen settings and provenance. Loading or starting never writes saved defaults implicitly. |
| Use recommended settings | Restore the form from the resolver without deleting saved preferences or changing a running model. |
| Running/loading model | Actual running/loading snapshot is authoritative. Draft edits mean “For next start”; a changed model means a proposed switch. |

Keep the resolver's existing saved-first `auto` API behavior for legacy callers
unless separately migrated. The new launch flow can request `source='built-in'`
explicitly and use `source='saved'` for the saved-settings action. Do not globally
change defaults precedence in PERFORMANCE_PLAN as an incidental UI refactor.

Resolve profiles from the final slice-8 data. At review time the shipped baselines
use 16,384 total tokens, one slot, q8 main/draft cache and MTP length 3. These values
are historical review context, not UI constants or requirements to freeze them.
If slice 8 changes context, reuse, or other settings, display and submit its final
compatible profile. Preserve negative evidence about mixed caches and lookup;
experimental availability must not turn into a general recommendation badge.

## Reliable Start and useful completion

Treat selection resolution and engine lifecycle as explicit states. Derive button
state in one place; the recurring poll must not override local pending operations.

| State | Primary control and visible feedback |
|---|---|
| Discovering / resolving identity / checking defaults | Disabled **Preparing model…** with the current phase. No submission of placeholder HTML defaults. |
| Ready and idle | Enabled **Start Model** with selected model and actual settings source. |
| Download/setup required | Specific **Download model · size**, **Choose engine**, or other repair action. Explain the unmet prerequisite. |
| Start POST pending / engine loading | Immediately disable repeat starts; show **Starting…**, elapsed time and **Cancel start**. Do not invent a percentage when load progress is unknown. |
| Healthy and serving; draft matches | **Running** status with model, applied summary and Copy API address; secondary Stop. No active Start button that reloads the same model. |
| Healthy and serving; draft differs | **Restart with these settings** or **Switch to <model>**, with visible text that current requests will be interrupted. Do not stop merely on selection/edit. |
| Experiment owns the engine | Start disabled with **View experiment** and access to its cancel action. Navigating away does not cancel it. |
| Failure / connection lost | Persistent inline error and recovery action. Preserve draft and last-known state, mark it stale on disconnect, and avoid an enabled Start backed by stale readiness. |

Tie defaults and capability responses to a complete selection generation, including
device and any edited launch inputs. Existing sequence counters are a useful start;
`fill()` currently initiates `inspect()` without awaiting it. Only declare Ready
when the selected model, compatible engine/device, resolved settings and validation
belong to the same current snapshot. Scope hashing/caching to relevant candidates
and invalidate identity caches on changes; don't hash every installed large file
before the first card can render.

Submit that snapshot once. Continue server-side launch validation and job ownership
checks for other tabs/concurrent requests; disabling a button alone is insufficient.
Known setup conflicts should be detected before interrupting an existing owned
server where possible. Never stop external engines or desktop applications as an
automatic recovery. A failed switch must honestly report whether the old server is
still running; do not claim rollback that did not happen.

Report **Ready** only after the engine health check succeeds. Preserve or add an
explicit lifecycle signal rather than conflating `engine.running` with readiness
or trusting stale job text after an engine exits. A network timeout submitting Start
requires checking actual operation state before retry, to avoid launching twice.

After successful start, place Copy API address next to Ready. The current engine
binds to workstation loopback while the panel can be LAN-accessible. Label the API
address “on the model workstation”; do not rewrite it to the browser host, imply
remote reachability, or add an Open chat link without a verified supported target.
An embedded chat interface or changing network bindings is outside this plan.

### Fresh installation and recovery

Show the two curated models even when uninstalled, ranked identically, with exact
variant and download size. A deliberate Download action shows progress, cancellation,
failure/retry and destination in expandable detail. On completion, discover/verify
the installed file and resolve its recommended configuration automatically. Then
present Start Model. Download does not start a GPU process as a hidden side effect.
If both models/engine are missing, explain both prerequisites and guide the next
step; engine setup remains a focused repair path using existing configured roots,
not a new installer project. Incomplete/cancelled downloads must not become launch
candidates merely because a file exists.

If an automatic initial candidate is known incompatible, explain the eligible
alternative. A file disappearing after selection, device mismatch, invalid saved
settings, model load failure or occupied port gets specific recovery text with
technical details expandable. No automatic model downloads, engine upgrades,
process killing, repeated start loop, or fabricated memory-fit guarantee.

## Experiments remain useful and separate

Retain existing workloads, quick sweep, warm conversation mode, source adherence,
combination queue, context search, reset, export, result detail and provenance.
Put Saved comparisons and benchmark-specific queue details inside Experiments;
keep global engine ownership/status reachable from either view.

On first entry, initialize an experiment draft from the current launch snapshot.
Subsequent launch edits must not overwrite an existing experiment draft; offer
**Use launch settings** explicitly. Running a benchmark uses an immutable snapshot
of that draft. Experiment controls, reset, results inspection and navigation never
change the normal launch selection or saved settings implicitly.

Use **Try these settings in Launch** to copy a completed result for review, and a
separate explicit save action to persist preferences. Retain eligibility checks and
measurement provenance; warm-only results or failed source adherence cannot become
a general measured default. Ensure legacy “Use as default” actions have unambiguous
copy and do not silently defeat Recommended mode on the next fresh idle opening.

Because benchmarks own and restart the same engine, show that effect before a run
when a model is serving. Use an explicitly labelled action such as **Stop model and
run experiment** for that case; no interruption occurs just by opening Experiments.
Preserve cancellation, partial results and the existing reset isolation guarantee.

## Implementation slices

Implement one bounded slice at a time; update this file's status/checks at each
boundary. Use the latest integrated PERFORMANCE_PLAN implementation before changing
UI/settings code. Do not derive work order from its stale overview table alone.

| Slice | Deliverable | Likely files | Done when |
|---|---|---|---|
| 1 — Curated automatic selection | Ranked catalogue metadata, variant-safe installed matching, compatible engine/profile selection, and a unified resolution/readiness guard. Integrate final portable profiles without duplicating their settings. | `models.json`, `discovery.py`, `app.py`, `recommendations.py` as needed, `static/index.html` | Both recommended files choose rank 1 regardless of path order; rank-2-only works; duplicate filenames remain distinct; missing/mismatched setup is explained; fast clicks cannot launch unresolved settings. |
| 2 — Launch-first layout and settings source | Compact launch card, initial Launch view, secondary Experiments view, one customization disclosure, recommended-first Launch mode and explicit preserved saved settings. Keep one source of truth for each control. | `static/index.html`, minimal resolver response additions if needed | Idle first-screen Start works without edits; saved/manual/measured sources remain honest; expanded controls retain all values/help; a reloaded running model remains visible. |
| 3 — Lifecycle, setup and recovery | Adjacent startup/readiness/error states, repeat-submit protection, explicit restart/switch, download completion resolution, API copy with workstation scope. | `static/index.html`, `app.py`, `engine.py`, discovery/download response handling as needed | Start/loading/cancel/ready/failure/disconnect are distinguishable; selected model/profile is what starts; missing installs have a clear path; serving is never restarted by a plain duplicate Start. |
| 4 — Experiment separation and usability pass | Separate experiment draft and explicit transfers/promotion; preserve all benchmark/result features; complete accessibility/responsive checks and bounded end-to-end validation. | `static/index.html`, minimal API contract changes only if required | Experiment actions preserve launch draft/defaults; engine interruption is explicit; the acceptance matrix below passes with no performance sweep. |

Slices 1–2 establish the primary journey; slices 3–4 complete its recovery and
regression coverage. Avoid shipping a layout that claims Ready before slice 1's
resolution guard is connected. Don't add a framework, test infrastructure, model
leaderboard, new tuning modes, telemetry, documentation site, or deployment work.

## Acceptance and validation

Use lightweight temporary browser/API checks consistent with HANDOFF.md. Test
state transitions with mocked APIs; reserve actual GPU checks for a short authorized
idle window after the other agent's performance work. Never interrupt its jobs.

| Scenario | Observable acceptance criterion |
|---|---|
| Fresh idle 3090, both recommended checkpoints and compatible engine installed | Qwen3.8 preselected with resolved built-in settings; one Start click; posted/applied settings and exact file match the displayed recommendation. No experiment/save/download POST. |
| Only recommended Qwen3.6 installed | Exact MTP IQ4_XS selected; absent rank 1 remains visible with download option and explanation. One Start click after resolution. |
| MTP/non-MTP same basename, renamed directories, multiple engine paths | Identity and compatibility drive selection and badges; no lexical/path-only substitution; explicit variant/build overrides survive. |
| Existing saved custom or old measured defaults | New idle Recommended mode leaves storage byte-for-byte unchanged; Use my saved settings restores values/provenance and correct qualification; no accidental overwrite on reset/start. |
| Slow/error/reordered discovery, hash, defaults or capabilities responses | Start cannot submit partial/stale configuration; switching model/device or editing a field defeats stale responses; resolver failure has an explicit recovery path. |
| Double click, another browser tab, start timeout | One owned launch operation; authoritative status reconciliation; active-job conflicts shown without losing draft. |
| Running model, refreshed page, then model switch | Current model/status shown immediately when known; draft versus running settings explicit; only a clearly labelled switch/restart interrupts service. |
| Startup failure, cancellation, process exit, panel disconnection | No false Ready state; errors persist near action; logs reachable; stale status marked and recovery works. |
| No model, no engine, non-3090 or Vulkan-only setup | Prerequisites/fallback source stated accurately; no unsupported measured claim; download/cancel/retry/verification and engine repair paths remain usable. |
| Experiment view, reset, queued combinations, result promotion | Drafts/queue/results survive navigation; reset stays experiment-only; applying/saving is explicit; hidden experiment progress/cancel remains accessible. |
| Desktop 1440×900 and narrow 390×844, normal idle state | Model, source and Start visible without page scrolling; Start precedes customization in visual and keyboard order. No horizontal page overflow; long variant/path/error text wraps. |
| Keyboard, touch, zoom and themes | Controls have names and focus indicators; radio/view navigation works; hidden views leave the tab order; help retains focus/touch/Escape behavior and unique IDs after rendering. At 200% zoom content reflows without clipping; scrolling is acceptable. |

After implementation, perform one bounded actual start/readiness/short response/stop
check for each curated checkpoint on the intended setup, verifying submitted versus
applied profile identity. Check behavior with an existing saved preference and a
running-model page reload. These are functional checks, not performance or broad
quality validation; refer performance claims to the final PERFORMANCE_PLAN evidence.

Success means a first-time user on a configured machine can identify the selected
recommendation, start it, and see that it is ready without opening Customize or
Experiments. For a small observed walkthrough, record click count, accidental
experiment entry, need for explanation and whether the user can identify the
running model. Time from **Ready to start** to the click measures UI friction;
model load time is separate. No production telemetry is required.
