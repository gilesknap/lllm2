# Experiment and model settings

The controls in **Experiments** answer two different questions: how should the
model run, and what work should it be measured on? **Customize settings** controls
the engine and model. The **Experiments** card controls workloads, token budgets,
repetitions and context probes. A useful comparison keeps the work fixed while
changing one engine setting, then checks both the timing and the returned output.

This guide covers the controls in both views. For the steps to run and save a
comparison, see [Compare and save settings](../how-to/compare-settings.md).

## Launch settings and experiment settings

Launch and Experiments have separate drafts. **Use launch settings** copies the
launch configuration into the experiment draft. Editing that draft does not
change the serving model, but **running an experiment stops the serving model**
because it uses the same GPU.

**Load settings → Recommended defaults** restores the starting configuration for
the selected model and hardware. **My saved settings** loads your saved choice.
**From an experiment…** loads a completed result into Launch for review; it does
not save it or restart the model. **Save my settings** saves the current draft
for that model and backend.
The panel shows the settings source and any measurement evidence. Defaults can
be estimates rather than measurements on your exact hardware.

## Workloads

Each selected workload runs at every selected prompt size for every repeat and
configuration. Selecting more workloads adds more samples; it does not divide a
fixed time allowance among them.

| Checkbox | What the model is asked to do | What the result tells you |
|---|---|---|
| Generate code | Implement a Python LRU cache with bounded capacity and constant-time operations. | Timing for code generation and explanation. The workbench does not execute or grade the generated implementation. |
| Edit code | Change supplied ledger modules to reject negative amounts and use decimal arithmetic. | Timing for a broader editing task. There is no exact-output correctness check for this workload. |
| Long code context | Review supplied modules and implement a reusable ledger service with validation and pagination. | Prompt-reading and generation timing with code context. The selected token budget determines how long the context actually is. |
| Copy source exactly | Return the complete fixed source excerpt unchanged. | Whether the returned source matches the expected text, alongside its timing. Useful for investigating copying and prompt lookup. |
| Single source edit | Change `if amount < 0:` to `if amount <= 0:` and preserve the rest of the source. | Whether the complete returned source contains only that edit. |
| Long-context retrieval/edit | Retrieve three fixed facts placed early, middle and late in the reference input, then update a small source excerpt and one comparison. | A narrow retrieval and editing check. Passing does not establish general long-context accuracy or stability. |

The first three workloads force the requested output budget, ignoring an early
end-of-sequence token so that speed samples do comparable amounts of generation.
The three source-checking workloads instead stop naturally and use an output cap
of **at least 2048 tokens**, or your larger requested budget up to 4096. Thinking
also consumes that cap. Compare actual output counts and elapsed time as well as
tokens per second; a shorter, incorrect answer is not evidence of a speedup.

Source adherence compares the complete extracted source with the expected text.
The checker permits surrounding whitespace, a code fence and a preceding thinking
section. It does not run the source or assess general answer quality. Inspect the
adherence details and any truncation before accepting a result.

## Experiment budgets

| Control | Meaning and effect |
|---|---|
| Single-size prompt tokens | The input size used when Quick sweep is off; also used by the warm conversation experiment. Larger inputs increase prompt-processing work. Allowed range: 128–131,072. |
| Output tokens | Requested generation budget, normally 256. Range: 16–4096. Source checks have the larger cap described above. Increasing it adds generation work and leaves less room for input. |
| Repeats per workload | Repeats each cold workload/size sample, or the whole warm conversation sequence, 1–5 times. Repeats help expose variation and multiply run time. Context search runs once per configuration. |
| Speed-test timeout per operation | Maximum wait for an individual speed-test operation, including engine requests, rather than a deadline for the whole experiment. Default: 900 seconds; range: 10–86,400. Increasing it only changes how long slow operations are allowed to take. |
| Context-probe timeout per operation | Separate timeout for context search. Default: 900 seconds; range: 10–86,400. A timeout makes the search inconclusive; it does not prove that memory ran out. |
| Quick sweep: 1K / 16K / 64K | Tests 1024, 16,384 and 65,536 input tokens. Sizes are capped to the available space per slot and duplicates are removed. Off means one explicit prompt size. |
| Also test full launch window | With Discover usable context, confirms the largest loaded window with one full long-code prompt; this is the slow step. For speed samples, adds a prompt filling the experiment's configured window per slot, after reserving output space and a margin. |
| Discover usable context | Runs before any speed sample. Quick load-only checks find the largest window the engine accepts. On by default; with no workloads selected it is the whole experiment. The result is reported as loaded but unconfirmed unless Also test full launch window is on. |
| Context search ceiling (per slot) | The largest window the search may try, including input and output. Initially based on model metadata when available; range: 512–1,048,576. This is a search limit, not a promise that the model or GPU can use it. |
| Reset experiment settings | Resets workload selections, budgets and experiment checkboxes, including the ceiling for the selected model. It does not reset the model customization draft or delete results. |

Input and output must fit in `total allocated context ÷ slots`, with 32 tokens
reserved as a margin for cold tests. When source-checking and ordinary workloads
are mixed, their shared input sizes reserve space for the larger source output
cap. For example, 32,768 total tokens with two slots gives 16,384 per slot. A source
workload with a 2048-token cap leaves at most 14,304 input tokens for the sweep.

The displayed sample count is more useful than a universal time estimate:
`configurations × workloads × distinct prompt sizes × repeats`. Context probes
are additional. The panel's rough time estimate comes from an observed run;
models, GPUs and settings can change it substantially.

Context search runs first. It starts the engine at the experiment's own window,
doubles the window after each successful load up to the ceiling, then narrows
between the largest load and the smallest refusal, at most twelve loads. Loads
are quick because no prompt is sent. A load refused for lack of memory counts
as a memory limit; a load that fails for any other reason stops the search and
fails the run with the engine log, since it would fail at every size.
With **Also test full launch window** on, it then confirms the largest loaded
window with one full long-code prompt and only bisects with further prompt
probes, at most eight, if that confirmation fails or times out. Without it, the
largest loaded window is reported as usable but unconfirmed by a prompt. With
it, only a window that completed a prompt is reported as confirmed; if no prompt
probe succeeds, the loaded window is shown separately as unconfirmed. The
search stops when the uncertainty is roughly 10% of the largest success, with a
minimum resolution of 1024 tokens. A timed-out probe bounds the
search without ending it and is reported separately. Failures are retained. Recommended context applies
10% headroom and rounds down to a multiple of 256; it is an estimate for future
workloads, not another measured or guaranteed limit.

## Choosing a run

| Action | Configurations tested |
|---|---|
| Run baseline | The experiment settings currently shown, including any inherited or saved tuning. Baseline does not mean that acceleration is switched off. |
| Baseline + single options | The baseline plus supported alternatives for speculation, common cache precision, flash attention and reasoning effort. Each variant changes one option group; cache variants set both K and V to the selected common precision. Invalid variants are skipped with reasons; runtime failures are recorded. It is not an exhaustive search of every advanced control. |
| Run current combination | The current experiment settings as one custom configuration. |
| Add current settings | Copies the current experiment configuration into the combinations list. Later edits do not change that copy. |
| Clear | Empties the combinations list. Saved results are retained. |
| Run combinations | Tests 1–20 added configurations using the workload and budget controls above. Keep model, engine, backend, device, context and slots the same so that comparisons isolate tuning changes. |
| Cancel operation | Requests cancellation of the active launch or experiment. Review saved partial samples and failures rather than treating cancellation as a completed comparison. |

### Warm conversation

**Run warm conversation** measures reuse across six controlled turns: cold start,
append, suffix edit, switch away, switch back and an early history edit. It then
replays the same prompts with reuse disabled. That is twelve requests per repeat.

It requires **one slot** and uses the fixed prompt size, output budget and repeats.
The panel uses the long-code workload for this mode and ignores the selected
workloads, Quick sweep, full-window option and context search. Leave enough context
for the conversation growth as well as the output.

Blank conversation-reuse controls become **2048 MiB host prompt cache** and **four
checkpoints per slot** for this experiment only. Explicit zero remains zero.
Normal launches keep engine defaults when these fields are blank. Compare warm
results with their uncached replays, and inspect measured reused/processed tokens,
first-token timing and process RAM. Reuse at one size does not establish behavior
at a much larger context.

## Customize settings

The same editor is used for the Launch and experiment drafts. Blank advanced
fields generally preserve the engine default; they do not always mean zero.
Changes take effect on the next launch or experiment.

### Engine, device and memory allocation

| Control | Meaning and tradeoff |
|---|---|
| Installed checkpoint | The GGUF file containing the target model weights. Different quantizations and copies can have different memory requirements and behavior. Use the exact same checkpoint for tuning comparisons. |
| llama-server binary | The engine executable. Its build determines the available controls and kernels. Changing the engine makes a different comparison. |
| Backend | CUDA or Vulkan, as supported by the selected engine. The bundled installer supplies CUDA engines; an existing compatible engine can be selected by path. |
| GPU device | The device reported by that engine for the selected backend. Free memory and other GPU applications affect what fits. |
| Total allocated context | Capacity in tokens for prompts and replies, shared across slots. A larger allocation needs more conversation memory; it does not make every request contain more text. |
| Slots | Concurrent conversation slots, not CPU threads. The total context is divided between them. More slots leave less room per conversation. |
| GPU placement | Auto asks a compatible engine to fit weights and buffers to available VRAM, with a 1 GiB margin. Some weights may remain in system RAM. Manual enables an explicit layer count. |
| GPU layers | Number of model layers requested on the GPU in manual mode. Zero keeps model layers on the CPU; 999 requests all layers. Automatic fitting is disabled for an explicit count. Check the engine log for actual placement. |

Weights, conversation state and runtime buffers all consume memory. Moving layers
to system RAM can allow a larger configuration to start, but can reduce speed.
Total GPU memory shown by the panel includes other applications, such as the
desktop, so it is not the engine's allocation alone.

### Attention, cache and prompt processing

The attention **KV cache** stores information about tokens already processed.
K and V are two parts of that state; their precision is independent of the model
weight quantization. Smaller cache types can leave more room for context, but
speed, compatibility and output behavior still need comparison.

| Control | Meaning and tradeoff |
|---|---|
| Flash attention | `on` requests a memory-efficient attention implementation; `auto` leaves the choice to the engine; `off` disables it. This workbench requires `on` for quantized KV cache and DFlash. |
| KV cache (K and V), also labelled Common conversation memory precision | Common attention-cache precision: `f16`, `q8_0` or `q4_0`. Lower precision saves cache memory but may change performance and accuracy. Selecting a common value explicitly clears independent overrides and links K and V again. |
| K cache override / V cache override | Inherit uses the common choice; an explicit value changes just that part of the attention cache. Mixed pairs are experimental and may lack a working GPU kernel even when both types are advertised. Draft cache and hybrid recurrent state precision are separate. |
| Logical batch (tokens) | Maximum prompt tokens submitted together. Larger batches may improve prefill speed but need more working memory. Blank preserves the engine default. |
| Physical microbatch (tokens) | Physical chunks used to process a logical batch. It must not exceed the logical batch. Larger chunks may increase memory use; blank preserves the engine default. |

The inherited context planner is calibrated for `q8_0` K and V together. It does
not establish the capacity of every other cache pair. Use actual launches and
context measurements rather than assuming that changing precision scales capacity
by a fixed factor.

### Reasoning and speculative decoding

**Reasoning effort** requests `default`, `minimal`, `low`, `medium`, `high`,
`xhigh` or `max` from a compatible model/template. Supported values vary. More
reasoning can consume more of the output budget before the final answer. This
control is separate from cache precision and speculation.

Speculation guesses upcoming tokens and asks the target model to verify them.
Accepted guesses can reduce generation work; rejected guesses and draft-model
execution add overhead. Availability alone does not establish a speedup.

| Speculation option | What it uses |
|---|---|
| Off | No speculative guesses. Useful as a comparison point. |
| MTP | A multi-token prediction head contained in the target checkpoint. Requires actual MTP tensors and engine support. |
| DFlash · experimental | A separate drafter trained and converted for the exact target. Requires CUDA, flash attention on, a compatible drafter and confirmation of that pairing. It uses additional memory. |
| Prompt lookup (ngram) | Repeated text in the prompt supplies candidate continuations without a separate draft model. Copying is a useful workload to investigate; benefit still requires measurement. |
| MTP + lookup · experimental | An explicit combination requiring support for both methods and the lookup controls. Ordinary counters aggregate the methods, so they do not establish which method caused a benefit. |

| Related control | Meaning and tradeoff |
|---|---|
| Draft length | Maximum guessed tokens per draft for supported model-based speculation. Longer drafts may accept more tokens or waste more work. Unused when speculation is off; it is separate from lookup M. |
| Draft KV cache | Precision for the model-based drafter's cache: engine `default`, `f16`, `q8_0` or `q4_0`. It does not change the target K/V cache. |
| Lookup match N | Matching-token count for prompt lookup. Set N and M together, with `1 ≤ N ≤ M`. |
| Lookup draft M | Lookup continuation width, independent of MTP draft length. Blank N/M preserves the engine defaults for ordinary lookup; the combined MTP + lookup mode explicitly selects N=3, M=3 when both are blank. |
| DFlash drafter GGUF | Local path on the model workstation to the separate drafter. Browse locates a file; a filename does not prove compatibility. |
| Pair verification checkbox | Your confirmation that the ordinary DFlash drafter was trained and converted for the exact target. This supplements the workbench's checks. |
| Chat template override | Optional path to a local template file formatting messages and reasoning instructions. Blank uses the checkpoint template. A mismatched template can change answers or cause failures. |

### Experimental CUDA execution

**Target GPU sampling** asks the engine to choose the next target token on the GPU.
Unsupported sampler requests may fall back to CPU sampling. The control does not
change draft sampling. Unchecked preserves the normal engine behavior.

**Concurrent CUDA streams** controls overlap of eligible operations within CUDA
graph execution. It requires ordinary CUDA Graphs and exactly one visible CUDA
device. `Engine default` preserves the inherited environment; `On` and `Off`
request an explicit choice. This does not add conversation slots or concurrent
requests. Compare performance before retaining either experimental setting.

### Conversation reuse

**Host prompt cache (MiB)** gives a finite RAM allowance for saved conversation
state. Zero disables that component; blank preserves normal engine defaults.
It does not cap total process RAM, model weights or working buffers.

**Context checkpoints per slot** controls saved states that can help resume edited
history on recurrent models. More checkpoints consume additional memory. Zero
disables checkpoints; blank preserves normal defaults. The warm conversation
experiment's special blank-field defaults are described above.

## Reading results and deciding what to keep

**Feature availability** reports what engine and checkpoint checks found.
“Available to try” is distinct from a successful launch, and both are distinct
from a measured benefit. Read incompatibility reasons and the exact engine log
when a configuration fails.

Cold speed samples start a fresh engine and disable prompt caching. Benchmark
requests use temperature zero and seed 42; these are fixed measurement settings,
not additional controls in this editor.

In **Experiment history**, compare the same checkpoint, build, workload, token
budgets and measurement mode. **Prefill tok/s** measures prompt reading;
**Decode tok/s** measures generated tokens. Warm processed-prefill rates describe
only newly processed tokens and belong in a separate comparison. Inspect total
elapsed time, actual output tokens, source adherence and failures alongside rates.
Client first-token event timing can include an event without visible answer text.

Expand a row for evidence, context, resolved settings and memory samples. Peak
VRAM is sampled total device use; engine process RAM is a separate measurement.
A brief peak can fall between samples. **Export CSV** and **Copy table** use the
current displayed sample order and retain partial or failed entries. **Export
full JSON** keeps the complete records and diagnostics.

**Try in Launch** loads an eligible result’s settings for review. Expand the
row to reach it. The **Context** choice defaults to **Tested**, the largest
context that passed the search; **90%** selects the rounded-down headroom
estimate instead, and **Original** keeps the experiment's own context.
The choice appears only when a successful context measurement exists.
Warm-only evidence, failed source adherence and incomplete runs cannot be
promoted this way. Loading does not save settings or restart the model:
review the draft, save it if desired, then start or restart to apply it. See also
[Defaults and measurements](defaults-and-measurements.md).
