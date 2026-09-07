# UI usability and visual polish

Status: all three implementation slices complete; validated 7 September 2026.
Requested 7 September 2026. Execution authorized 7 September 2026. Progress and validation are recorded below.

## Objective

Make the everyday journey obvious: choose/download a model, start it, then connect
Pi, Claude Code or Codex. Make settings easy to save and recover, and expose both
RAM and VRAM now that automatic offloading works on smaller GPUs.

The user explicitly requested:

- Save my settings at the top right of Customize settings, beside load controls.
- Recoverable built-in defaults and settings obtained from experiments.
- Choose a model at the same width as the other panels.
- RAM usage alongside the existing VRAM badge.
- A prettier UI that helps a first-time user get running without hunting for controls.

The original session wrote this plan only. Implementation was subsequently
authorized. Merging and publishing remain separate from this local execution.

## Starting point and evidence

- Current checkout: `codex/local-harnesses-auto-gpu-fit`, `d45dc6e`.
  PR #18 contains harness launchers and automatic fitting. Check its current merge
  state before branching; base the UI work on a revision containing those changes.
- `UI_IMPROVEMENTS_PLAN.md` records the earlier completed launch-first redesign.
  Preserve that history; this plan supersedes its conflicting UI decisions only.
- Review was of current markup, styles, frontend state handling, backend defaults,
  and hardware reporting. No new rendered visual review has been performed yet.
- The real machine used in this conversation is an RTX A1000 with 8 GiB VRAM and
  roughly 62 GiB RAM. Pi works well using CPU/RAM offloading. Do not assume the
  workstation is the RTX 3090 from older performance records.
- Automatic fitting was smoke-tested with Qwen3-8B at 32K. The user subsequently
  ran Qwen3.6-35B-A3B-MTP. Claude's initial request was 38,261 tokens: 32K failed,
  and the user confirmed success after increasing context to 65,536 and relaunching.
  This is one observed harness workload, not a universal minimum-context rule.
- A machine-local Pi shell wrapper previously discarded arguments. That is outside
  the repository UI scope; do not change external wrappers as part of this work.

## Product decisions

### Layout and appearance

Use one shared outer width for Launch, Customize, Experiments and logs. Currently
`#launch-view` is capped at 800px inside the 1100px main container. Remove the
special narrow cap and align card edges; constrain text and field columns inside
cards when needed for readability.

Retain the green accent, light/dark themes and restrained styling. Standardize
card padding, header spacing, button sizes, text hierarchy and state badges.
Start/Restart remains the primary action; Save and Load are secondary. Reduce
always-visible technical prose, keeping concise explanations beside relevant
controls and full diagnostics/evidence in disclosures.

Do not introduce a frontend framework, redesign the experiment methodology,
change model rankings, or undertake a general documentation/deployment project.

### Settings toolbar and meaning of defaults

Place a stable toolbar at the top right of Customize:

```text
Customize settings                 [Load settings ▾] [Save my settings]
Settings: My saved settings · Modified                 Unsaved changes
```

Use a proper header/disclosure control with adjacent buttons; clicking Save or
Load must not accidentally toggle the panel. On narrow screens the actions may
wrap below the title while keeping their order and labels.

The Load menu offers:

1. Recommended defaults: the built-in measured/estimated starting configuration.
2. My saved settings: the user's explicit saved configuration.
3. From an experiment…: an eligible completed-result picker with date, workload,
   backend, context and source. Preserve the existing preview/promotion validation.

Keep menu items discoverable; disable unavailable choices with a short reason
instead of hiding the entire action. Show unsaved changes and a brief Saved
confirmation. Loading changes the draft, saving persists it, and restarting
applies it to the running model. These are distinct operations.

Actual storage today is per resolved checkpoint path and backend, not global.
Manual saves and promoted experiment settings share the `default` slot, with
separate provenance in `default-evidence`. Use the wording above rather than
inventing a global settings profile that applies GPU-specific tuning everywhere.

Keep recommended defaults independently recoverable and expose stored experiment
results through the picker, so a manual save cannot erase the ability to recover
an eligible experiment configuration. No destructive migration is needed just
to offer these three sources. Preserve existing saved defaults and provenance,
including experiment-derived saves. If implementation needs new metadata, add it
compatibly; do not silently split or overwrite old user records.

Launch and Experiment drafts remain separate. In Experiments, label the header
Customize experiment settings and retain its draft-only controls. Saving normal
launch defaults stays an explicit Launch action after trying an experiment there.
Keep existing source/adherence/warm-experiment eligibility restrictions.

### Models and downloads

Replace the misleading installed-only All models & variants disclosure and the
buried download catalogue with a visible browser near Choose a model:

```text
Choose a model                       [Installed] [Browse catalogue]
Model name       Installed / Available / Downloading       [Action]
```

Keep curated recommendations easy to find. Installed includes every discovered
usable checkpoint, including custom paths and duplicate variants. Browse catalogue
includes installed and not-yet-installed catalogue entries. Show counts and useful
empty states. Put long paths, exact quantization and identity details in secondary
content; preserve unambiguous selection when names match.

Active downloads stay visible whichever filter is selected. Show a progress bar,
downloaded/total size where known, speed and Cancel; show useful retry states.
On completion, refresh discovery and offer Use this model. Preserve an explicit
selection and edited draft while polling or rescanning. Do not auto-start or
silently select a newly downloaded model. Reuse the existing download backend.

Performance recommendation badges retain hardware qualification near their claim,
especially 3090-specific measurements shown on smaller GPUs. Automatic placement
does not establish speed, fit or suitability for a given harness.

### Memory visibility

Header badges show GPU name, VRAM used/total and RAM used/total in GiB. Label them
as resources of the model workstation, including when the browser is remote.
Keep the numbers stable in width to avoid layout movement on each poll.

On Linux derive RAM usage as MemTotal minus MemAvailable, avoiding the misleading
interpretation that reclaimable filesystem cache is unavailable memory. Expose
available memory in details. Handle unavailable metrics as unavailable, not zero.
Keep system usage and optional engine RSS distinctly labelled; never equate RSS
with total model memory or sum overlapping memory measures.

`hardware()` currently reports total RAM only. Extend the status data compatibly
with a small, cheap memory read; use existing `/proc/meminfo` handling in `warm.py`
as a reference. Do not instantiate Store or trigger GPU probes/benchmarks merely
to read host memory. Preserve current GPU reporting and failure behaviour.

Give GPU placement a visible Auto/manual mode in Customize. Avoid making an empty
number field the only indication of automatic fitting. Preserve backend values:
`gpu_layers: null` is Auto, 0 is explicit CPU layer placement, 999 requests all.
Do not display a guessed numeric count as observed placement. Logs can provide
details; MoE placement may mix tensors within layers.

### First-run and ready states

Show the next actionable step in the main flow:

| State | Primary next step |
| --- | --- |
| No compatible engine | Set up an engine |
| No installed model | Browse models |
| Download active | Progress and cancel |
| Download complete | Use this model |
| Model and settings ready | Start model |
| Model serving | Connect an agent |
| Edited running configuration | Restart with these settings |

Engine setup should expose the existing CLI installer command and existing-engine
selection clearly. Do not create a browser-driven package manager or silently
install software as part of this UI pass.

Once ready, reveal a compact Connect a coding agent section with Pi, Claude Code
and Codex choices and a copyable command (`lllm2 pi`, etc.). Keep Copy API address
for other clients. Commands run in a terminal on the model workstation; do not
imply the browser can launch a CLI on a remote user's computer. Include a fallback
for clipboard access being unavailable on an HTTP/LAN page.

Show context per conversation prominently. Explain that harness instructions and
tools consume context and that the harness must be relaunched after server context
changes. Use the observed Claude case as supporting guidance, not a hardcoded
universal minimum. Do not silently change the 32K fallback or running context in
this UI project.

Errors lead with a short explanation and concrete remedy, with exact diagnostics
expandable. Distinguish missing engine/model, insufficient memory, context overflow,
and disconnected panel where evidence permits. A browser may not observe a harness
request failure; do not promise diagnostics without an actual data source. Preserve
unrecognized raw errors, lifecycle guards and explicit restart/switch semantics.

## Implementation slices

### 1. Consistent layout and settings management

- Capture the current rendered desktop/mobile layout with representative data.
- Align card widths and apply the spacing/type/button polish.
- Add the Customize toolbar, load-source menu and experiment picker.
- Preserve source/provenance and independent Launch/Experiment drafts.
- Make Auto/manual GPU placement clear.

Acceptance: save/load controls are at the requested location; all three sources
are discoverable; saved values and eligible experiment results survive manual
saves; loading does not restart the model; numeric zero survives round trips;
cards align; keyboard and narrow-screen interactions work.

### 2. Model discovery, downloads and setup

- Introduce Installed/Browse catalogue views, clear labels and empty states.
- Integrate active download progress and completion actions into the chooser.
- Add visible engine setup guidance using existing capabilities.
- Shorten model rows while preserving variant identity and measured qualifiers.

Acceptance: a new user can find and download a model without expanding Customize;
installed/custom variants remain selectable; active downloads stay visible across
filters; completion does not discard edits or switch models; cancellation/retry and
rescan preserve coherent state. Exercise slow responses and selection races.

### 3. Memory and ready-to-use guidance

- Add RAM telemetry and coordinated RAM/VRAM badges.
- Add agent connection commands and context guidance after startup.
- Refine lifecycle/empty/error copy and disclosures.
- Finish the rendered visual/accessibility pass across all new states.

Acceptance: RAM reflects reclaimable cache correctly; unavailable telemetry is
honest; polling does not move focus or shift the layout; commands use running
server context and workstation scope; raw diagnostics remain reachable; no action
claims settings are live until the corresponding restart succeeds.

Keep commits scoped to coherent changes, separating backend telemetry/settings
logic from larger presentation work where practical. Update this plan's progress
and record validation/limitations as each slice completes.

## Validation and execution boundaries

- Read applicable AGENTS.md, inspect git status and check PR #18 ancestry first.
- Preserve current launches, downloads, models, saved preferences and results.
  Avoid restarting a panel while it owns an active download merely to review UI.
- Use an isolated state directory for tests. Store initialization changes the
  status of previously running result records, so never use real state for fixtures.
- Reuse existing tests and add targeted coverage for persistence/source selection,
  memory calculations and asynchronous UI state transitions. No broad test or CI
  infrastructure project. Existing baseline: 22 Python tests plus JS syntax checks.
- Render representative fresh-install, idle, downloading, ready, edited and failed
  states at desktop and narrow mobile widths in both themes. Check overflow,
  contrast, keyboard operation, focus retention, announcements and clipboard fallback.
- Include network delay/disconnection and switching views while requests are pending.
- Prefer fixtures for destructive lifecycle and download tests. Live GPU tests are
  only needed if implementation changes launch behaviour; do not make performance
  claims from visual checks or restart the user's model unnecessarily.
- Before completion, inspect the final diff and document what was tested, what
  remains uncertain and how the user should refresh/restart to adopt changes.
- This plan does not request subagents. Follow the next session's applicable agent
  instructions; do not treat the three slices as authorization for delegation.

## Recommended next session

Use **High reasoning effort**. This is a bounded UI project, but persistence,
experiment provenance, asynchronous polling and running-versus-draft state need
careful reasoning. High is a task-specific recommendation, not a benchmark claim
or a requirement to use the maximum effort available.

Suggested opening prompt:

> Implement UI_USABILITY_PLAN.md in its three slices. Start by checking repository
> state and whether PR #18 is merged. Preserve running operations, existing saved
> settings and experiment provenance. Validate the rendered UI on desktop/mobile
> and keep changes in sensible commits. Update the plan with progress and evidence.

When that implementation request is given, proceed through the slices without
reconfirming routine decisions already recorded here. Resolve small implementation
choices from the plan; ask only when a materially different product decision is
required. Publishing or merging follows the authorization in that new session.

## Execution progress — 7 September 2026

- Base verified: main at `684d197` includes merged PR #18. No applicable
  AGENTS.md was present. The earlier UI plan remains untouched.
- Slice 1 complete: shared outer width, Customize header toolbar, three load
  sources, eligible experiment picker, source/modified/saved status, and explicit
  Auto/manual GPU placement. Browser fixtures passed save/load with null and zero,
  result preview without saving, separate drafts, and narrow-screen overflow.
- Baseline screenshots and isolated Chrome checks are in `/tmp/lllm2-usability`.
  Existing 22 Python tests passed; four targeted memory/settings-recovery checks
  also pass. Tests use temporary storage and mocked APIs, with no live operations.
- Slice 2 complete: Installed/Browse catalogue filters, exact-path variants,
  visible download progress/cancellation/retry/destination, explicit Use this
  model after completion, and engine setup guidance. Polling and rescans preserve
  edited drafts and focus; pending download requests cannot be submitted twice.
- Slice 3 complete: workstation RAM/VRAM in GiB, MemAvailable-based RAM usage,
  honest unavailable/stale readings, Pi/Claude Code/Codex commands with manual
  clipboard fallback, running-context guidance, and expandable exact errors.
- Incorporated the user's review: Feature availability is outside Customize and
  follows the active draft between views. It and Model locations & engine setup
  each have a separate divider and heading.

Final verification:

- `.venv/bin/python -m unittest discover -s tests -q`: 26 tests passed. New
  coverage checks RAM calculations/unavailable counters, experiment recovery
  after manual saves, independent built-in/saved sources, zero GPU layers, and
  rejection of warm-only/failed/incomplete result promotion. All storage is temporary.
- `node tests/test_ui_browser.cjs`: passed with isolated Chrome and mocked APIs.
  Set `CHROME_BIN` if Chrome is installed elsewhere. Screenshots and temporary
  browser state are written to the printed `/tmp/lllm2-ui-*` directory.
- Browser assertions cover the toolbar, saved and experiment sources, null/zero
  GPU placement, separate drafts, late settings responses, navigation during a
  slow rescan, pending download requests across polling, download focus retention,
  completion with both an edited selection and an empty chooser, running-context
  wording, clipboard fallback, disconnect/reconnect, persistent exact errors,
  unique IDs, keyboard activation/menu Escape, emulated touch help, separate
  support dividers, and duplicate Start protection.
- Rendered fresh-install, idle, downloading, ready, edited and failed states in
  light/dark themes at 1440×900, 390×844 and 720 CSS pixels at 2× density: 36
  combinations with no horizontal page overflow. Inspected representative
  screenshots, including the expanded Customize boundary. Final artifacts are
  `/tmp/lllm2-ui-3pjhTB`; original before screenshots are in `/tmp/lllm2-usability`.
- JavaScript syntax and `git diff --check` passed. No dependencies, performance
  profiles, saved-data schema, engine launch arguments or network bindings changed.

Limits and adoption: no physical touch-device or screen-reader user study, large
real download, new GPU launch, or performance measurement was performed. All
model lifecycle/download mutations were fixtures. The running panel was not
restarted. Restart it when its operations are idle and hard-refresh the browser
to load matching HTML, script and RAM telemetry. Local commits only; no push, PR
or merge is part of this execution.

Follow-up review: the initial divider treatment still looked part of Customize.
Feature availability now has its own full-width card outside the settings editor
and follows the active draft through dedicated Launch/Experiments hosts. Model
locations & engine setup also has a sibling card. Neither section is contained
inside Customize or the launch configuration card.

## Additional UI tweaks after PR #19

Authorized after reviewing `pi.ui.ideas.md`; based on merged PR #19 at `8cfacb5`.
These changes are on the separate `codex/ui-results-polish` branch.

- Saved comparisons now have sortable sample rows with text-labelled status badges
  and expandable context, timing, errors and exact evidence. Model/backend/context,
  workload and cold/warm/replay distinctions remain visible. Sorting retains numeric
  zero and puts missing rates last; expansion and keyboard focus survive refreshed
  measurements and view changes. Result promotion keeps the existing eligibility checks.
- Added CSV export, Copy table and Copy row in displayed order. Exports retain
  individual samples and empty failed runs, source/quality status, model/build
  identities where recorded, separate cold/processed-prefill rates, GPU/RSS metrics,
  context estimates and settings. Text cells cannot become spreadsheet formulas.
  Full JSON export still reads the complete results endpoint. Clipboard fallback
  now supports multiline text in a selectable textarea.
- Extracted the inline stylesheet to a formatted `static/panel.css`, with section
  comments and an explicit CSS route. Existing styles retain their cascade order.
  Added a keyboard-visible skip link to the active Launch/Experiments view without
  changing its deep link or draft.

Validation: 26 Python tests passed; the isolated browser suite passed its existing
36 state/theme/viewport cases plus results views on desktop/mobile in both themes.
New assertions cover numeric sorting, zero/missing rates, refreshed metrics without
new samples, preserved open evidence/focus, promotion restrictions, CSV parsing via
Python's csv module (including quotes/newlines), formula-like labels, multiline
clipboard fallback, unchanged full JSON, keyboard skip targets and no launch/save
side effects. A temporary backend check verified exact CSS bytes/content type and
rejection of arbitrary static paths without constructing a real App/Store/engine.
JavaScript syntax and diff checks passed. No GPU jobs or real downloads were run.
Restart the idle panel before refreshing so the new stylesheet route is available.

Follow-up review: “Try in Launch” and the available headroom-context action are
visible on every eligible sample row, alongside Details. Samples share their run's
settings; no first-sample restriction remains. Browser checks cover visibility with
details closed and loading distinct runs' settings from sorted rows without
starting an engine or changing saved preferences.
