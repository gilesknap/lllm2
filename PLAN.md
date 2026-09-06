# lllm2 — implementation handoff

## Task and authority

Build a new public GitHub repository, `gilesknap/lllm2`, distilled from https://github.com/gilesknap/lllm3090. The user has completed the scope discussion and asked to begin implementing when enough detail is known. This handoff is that agreed direction; do not restart the broad design quiz. Ask only for genuine blockers or material new scope decisions.

The user has now created the public repository `gilesknap/lllm2`; its existence and empty state were verified before adding this PLAN.md. No implementation has started. The source repository lllm3090 was only inspected and cloned; no changes or pushes were made to it. Reuse this repository and do not modify lllm3090.

The user explicitly requested committing this handoff as PLAN.md. This is an exception to the no-project-docs constraint below, not permission to build documentation infrastructure.

The user selected Astra with Medium reasoning for the implementation session. Do not assume an attached handoff changes the active model.

## Product goal and firm constraints

This is a learning and experimentation workbench for local LLMs, primarily coding workloads. Make it easy to discover which levers apply to each model, switch them on or off, measure their effects, compare combinations, and save preferred launch settings.

- Rapidly iterable, locally runnable Python project managed with uv.
- A public repo is wanted so the user can pull and run it on their workstation.
- No project documentation, test suite, CI, release machinery, package publishing, or production polish until iteration is finished. PLAN.md is the explicitly requested handoff exception; do not add further documentation during prototyping.
- Retain the web UI as a central feature. Do NOT implement the earlier, superseded proposal to drop the panel and deliver only a planner/CLI.
- Benchmarks are first-class UI functionality, not merely developer scripts.
- Preserve necessary licence/attribution for reused source even though project docs are deferred.
- Keep normal development smoke checks lightweight; do not claim GPU validation without running on the target hardware. Benchmark workloads are product functionality, not the deferred software test suite.

## Target environment

- Ubuntu 26.04.
- RTX 3090 (24 GB VRAM), 32 GB system RAM.
- AMD CPU described as “AMD 7”; exact model unknown and not a blocker.
- NVIDIA GPUs only. Prototype may focus on this workstation, but detect hardware rather than embedding 3090 capacity everywhere.
- Reuse the user's existing lllm3090 model downloads and llama.cpp installations.
- Retain CUDA and Vulkan selection. Actual workstation paths, engine versions and supported flags have NOT been inspected.
- Do not overwrite or upgrade those installations silently; preserve the old app's state and service. Keep lllm2's state separate and handle port/GPU conflicts explicitly.
- This is local workstation software, not a request to cloud-host or deploy the panel.

## Agreed experiment workflow

1. Select a model and engine/backend in the panel.
2. See applicable features and their prerequisites, including MTP, DFlash, and other verified options. Distinguish unsupported, unknown and missing prerequisites. Do not claim support merely from a model name.
3. Choose launch settings, including reasoning effort and inference/acceleration parameters.
4. Benchmark a baseline, then run applicable options individually against that baseline.
5. Select specific combinations for additional benchmark runs. Exhaustive combinatorial search is not required.
6. Compare speed and usable context, retaining the exact settings and workload for each result.
7. Promote a chosen result to normal panel launch defaults. Save defaults per model AND backend (the assistant proposed this distinction; it fits the agreed workflow).

Primary metrics: prompt-processing/prefill speed, generation/decode speed, and empirically achievable context size. Record VRAM to explain the tradeoffs. No answer-quality grading initially.

Coding workloads: generating code, editing supplied code, and longer code context. Keep workload-specific results separate; copying performance must not be labelled general coding speed.

Context discovery is an explicit requirement: actively find how large context can go. Engine failures and restarts during this process are acceptable. Validate real prompt processing plus generation, not just allocation/load success. Report the largest observed successful size separately from a recommended size with headroom. Do not claim a short probe proves arbitrary long-context stability or quality.

## DFlash decision

User accepted optional experimental DFlash from the start, initially validating one compatible target/drafter pair on CUDA. Missing prerequisites should disable it with an explanation, not block the rest of the prototype.

- Requires a compatible companion drafter; this is not a universal switch for all models.
- Record drafter identity and draft length/settings in benchmark results.
- Account for extra memory by measuring context limits with the drafter actually loaded.
- Do not assume MTP and DFlash can be combined. Validate permitted combinations against the selected engine.
- Do not promise a speedup on this 3090 or extrapolate published research multipliers.
- DFlash versus DFlash 2 must be distinguished where their requirements differ. The user agreed to DFlash generally, not an obligation to implement every variant.

Research checked on 2026-09-06: ordinary DFlash was merged upstream in llama.cpp PR #22105 (page showed 2026-06-28). Current speculative decoding docs describe `draft-dflash`, a target-specific draft GGUF, and draft-length control. This does not establish support in the user's installed binaries or all model/backend combinations. Reverify exact flags and compatibility before implementation:

- https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- https://github.com/ggml-org/llama.cpp/pull/22105
- https://github.com/z-lab/dflash
- https://z-lab.ai/projects/dflash/

## Source inspection already performed

Inspected main at commit `e56c8d0fb19f76a254b5453bf7f206b229705d6a`, tagged 0.7.0. A shallow checkout existed at `/workspace/scratch/80505345bfbc/lllm3090`; a new session must not assume this path survives. Fetch the source again if needed, read applicable repository instructions, and record the revision actually used.

The source contains about 5,500 lines of Python and 4,400 lines of tests, a nine-entry model catalogue, FastAPI/uvicorn panel, Typer CLI and plain HTML frontend. Dependencies include huggingface-hub and PyYAML; packaging uses setuptools. These facts describe the inspected revision, not a required lllm2 architecture.

Useful source paths:

- `src/lllm3090/panel.py`, `static/index.html`: panel routes and UI.
- `catalog.py`: model metadata, fit/context planning, catalogue/installed-model state. Currently imports config, engines, gguf and hardware, so it is not yet a pure standalone planner.
- `gguf.py`, `hardware.py`: checkpoint facts and NVIDIA hardware detection.
- `engine.py`: launch arguments, lifecycle, capability probing, MTP activation and effort handling.
- `engines.py`: engine discovery/selection, installation and CUDA build machinery.
- `downloads.py`, `state.py`: downloads and panel snapshot data.
- `speculation.py`: backend-specific speculation profiles.
- `config.py`, `data/models.yaml`, `data/profiles.yaml`: paths, assumptions, calibration and catalogue data.
- `cli.py`: existing benchmarks and many installer/service/CLI operations; selectively extract rather than inherit all of it.

Existing defaults observed in source: models `~/models`; engine `~/.local/share/lllm3090/llama.cpp`; additional builds `~/.local/share/lllm3090/engines`; old state `~/.local/state/lllm3090`; engine port 1919; panel port 8080. These are defaults, not verified workstation locations.

The planner includes model architecture, quantised KV cache, backend and MTP overhead, desktop/driver/workspace reserves and context/slot policy. These include empirical constants, not universally exact physics. Preserve useful knowledge without importing every historical assumption as a guarantee.

No tests or GPU benchmarks were run in the inspection session. Documentation contains some stale claims; prefer source plus runtime evidence over blindly copying text.

## Implementation direction — recommended, not additional user requirements

Keep the existing panel's model list, model downloads, start/stop and logs, then add experiment controls, a benchmark queue/progress/cancel surface, comparison results and “use as default.” The assistant proposed retaining these existing controls during the final quiz; the user then confirmed coding workloads rather than separately enumerating each control.

Use a small Python backend and lightweight frontend, reusing appropriate source. No elaborate plugin framework or unrelated UI redesign is needed. A simple capability/settings registry can drive launch controls, valid combinations and the single-option suite. The exact initial suite beyond MTP, optional DFlash and effort remains an implementation choice to ground in real engine support; cache precision, draft length and prompt-lookup are candidates, not commitments to all possible features.

Use durable local results storage (JSON or SQLite is an implementation choice). Record model/checkpoint identity, engine build/backend, GPU, parameters, workload/input and output token counts, context/slots, timings, memory observations, failures and timestamps. Keep actual measurements distinct from estimates.

For comparable speed measurements, use shared prompt/context sizes across configurations in addition to each configuration's own maximum-context probe. Hold sampling/output budgets consistent, separate cold prefill from cached runs, and avoid interpreting effort-induced output-length changes as throughput gains. Defaulting to one slot for initial coding experiments is reasonable but was not explicitly decided; expose slot/context semantics clearly.

Use bounded context probing with timeouts and cancellation. Restart only processes owned by the application, stop escalating after an unresponsive GPU, preserve partial results, and do not reboot/reset the workstation automatically. Permission for engine restarts is not permission for unrelated destructive actions.

Suggested sequence:

1. Verify source and available GitHub access; use this existing public repo, without changing lllm3090.
2. Implement a uv-run local app with reusable model/engine discovery and the familiar panel.
3. Add capability-driven launch controls and persisted per-model/backend defaults.
4. Implement sequential baseline/single-option/custom-combination benchmarks, context search, progress/cancel and saved comparisons.
5. Integrate optional DFlash with clear prerequisites; do not block the overall prototype on unavailable weights/builds.
6. Perform lightweight local startup/UI checks where feasible, then commit/push the prototype for workstation iteration. Give exact launch instructions in the conversation and state which GPU checks still require the user's workstation.

Proceed with reasonable engineering choices within this scope; do not reintroduce docs, CI, a test suite, installers, systemd management or cloud deployment as “best practice.” Do not assume remote access to the workstation. Ask only when actual access, a prerequisite or a material scope expansion blocks progress.
