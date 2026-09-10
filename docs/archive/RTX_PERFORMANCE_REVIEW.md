# RTX local inference performance review

> **Historical record — not current instructions.** Archived 9 September 2026.
> Scope, permissions, paths and next steps below describe the original session.
> Use the [maintained documentation](../index.md) for current behaviour and
> development guidance. See the [archive index](index.md) for context.

Research date: 6 September 2026. Archived from the conversation at the user's
request. This is a dated research snapshot, not a record of new local benchmarks.
Implementation priorities and progress live in [PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md).

## Scope and evidence

Review technologies that could improve model fit, context capacity, prompt
processing (prefill), or token generation (decode) in lllm2. Three research
subagents examined memory/quantization, serving runtimes, and speculation; the
main review checked the application, installed engine help, and primary sources.

Verified workstation: RTX 3090, 24,576 MiB VRAM, compute capability 8.6,
driver 595.84; Ryzen 7 5800X; approximately 32 GB system RAM. Installed CUDA and
Vulkan llama-server builds: b10715, commit `662a0b012`.

Current application supports GGUF selection, GPU layer placement, Flash
Attention, a shared f16/q8_0/q4_0 setting for both K and V, embedded MTP, ordinary
DFlash, simple n-gram drafting, draft length/cache, and slots. Benchmarks use
cold sequential requests, with prompt caching disabled. They establish execution
and memory capacity, not answer quality or concurrent-serving performance.

Distinguish three kinds of evidence throughout:

- **Local historical measurement:** old lllm3090 results, specific to their
  checkpoint, engine, workload and settings.
- **Available implementation:** upstream documentation/code or installed flags;
  this establishes a candidate, not a local gain.
- **Research or author-reported result:** requires reproduction and compatibility
  checks before informing shipped defaults.

## Practical shortlist

| Candidate | Benefit | Missing application work | Initial priority |
|---|---|---|---|
| Prefix reuse and state checkpoints | Lower repeat-request prefill/first-token latency | Cache controls and multi-turn measurements | High |
| Batch/microbatch tuning | Prefill speed versus working-memory use | Settings, bounded sweeps and memory measurement | High |
| Separate K/V precision | More context or a different quality/speed tradeoff | Independent values, additional types, compatibility checks | High |
| MTP plus prompt lookup | Faster copying/editing on suitable workloads | Combined modes and a true copy workload | High |
| External Gemma MTP assistants | Decode on additional model families | Separate assistant pairing and launch validation | High generally; deferred for the Qwen-first phase |
| Weight-quantization comparisons | Larger models or more context | Group checkpoint variants and retain provenance | Medium |
| vLLM or ExLlamaV3 | Different kernels, memory use and scheduling | Engine adapters and fair comparisons | Medium; later phase |
| Selective CPU expert/FFN placement | Fit modestly larger models | Finer controls than whole-layer placement | Medium; capacity-driven |
| Sparse prefill/selective KV access | Potentially large long-context gains | Separate engine and information-retention evaluation | Experimental |

### Prefix reuse

Repeated system prompts, source files and conversation history can reuse cached
state. This saves processing the shared prefix; it does not inherently speed up
generation of new tokens. The current cold benchmark intentionally excludes it.
[vLLM's explanation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

The installed engine exposes host-memory prompt-cache and checkpoint controls.
Add initial-turn, appended-turn, changed-suffix and conversation-switch tests.
Record reused/newly evaluated tokens and actual first-token latency. Hybrid and
recurrent models require working state restoration, not just a cache flag.
Budget host RAM as well as VRAM; preserving more checkpoints has a cost.

### Prefill batches and CUDA runtime controls

Installed help reports a 2,048-token logical batch default and 512-token physical
microbatch default. Candidate microbatches: 256, 512, 1,024 and 2,048, with
compatible logical batches. Larger is not necessarily better: workspace memory,
context occupancy and verification workloads change the optimum.
[Server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

Target backend sampling is experimental and disabled by default in this build;
draft backend sampling is already enabled. Check the active sampler chain and
fallback behavior. `GGML_CUDA_GRAPH_OPT=1` enables concurrent CUDA streams in the
documented implementation; it is distinct from CUDA Graphs. Most other fusion
improvements already run automatically. Treat both as isolated A/B candidates,
not guaranteed speed switches.
[CUDA maintainer explanation](https://github.com/ggml-org/llama.cpp/discussions/17621),
[sampling source](https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp)

A reported backend-sampling throughput gain at 32 slots on an RTX 5090 and older
Xeon did not improve that report's single-slot result. Do not transfer its
percentage to this 3090.
[First-hand upstream measurement](https://github.com/ggml-org/llama.cpp/issues/27050)

### KV precision and context

The installed engine accepts independent K/V types and additional `iq4_nl`/q5
formats. lllm2 currently restricts both arrays to the same f16/q8_0/q4_0 choice.
Try a small set of mixed pairs, initially q8 keys with q4 values; do not assume
keys always need more precision for every architecture.

Illustrative calculation for equal-sized conventional arrays, including basic
block scales: q8_0 uses 8.5 bits/value and q4_0 uses 4.5. q8/q4 therefore saves
approximately 24% of KV storage versus q8/q8. This is not a total-VRAM saving or
a guarantee about usable context; padding, recurrent state and workspaces differ.

Hadamard attention rotation is already upstream, including CUDA support; b10715
should inherit applicable paths. Verify actual use rather than presenting it as
a new feature to implement.
[Rotation implementation](https://github.com/ggml-org/llama.cpp/pull/21038),
[CUDA implementation](https://github.com/ggml-org/llama.cpp/pull/23615)

### Speculative decoding

Installed help advertises ordinary draft models, EAGLE3, MTP, DFlash, DSpark and
multiple n-gram methods. lllm2 permits only a subset. Upstream can combine a
model-based drafter with model-free lookup; draftless proposals have precedence.
[Speculation documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

Historical local evidence is particularly relevant: old lllm3090 recorded CUDA
Qwen3.8-27B long-copy at 115.1 tok/s for MTP plus ngram-cache, width 7, versus
94.0 tok/s for MTP alone (about 22% faster). Other workloads regressed. The
comparison changes both lookup and width, so new experiments should isolate
those factors before testing the combination. It supports an optional copy
profile, not replacing the general default.

Historical source on this workstation:
`/home/giles/code/lllm3090/docs/explanations/going-faster.md`.
This is not a portable evidence artifact; preserve the relevant result provenance
when shipping recommendations.

Candidate extensions: ngram-cache/mod/map, compatible EAGLE3 and ordinary
drafters, and draft-width tuning by workload/context. Acceptance percentage alone
is not the objective; elapsed time includes drafting and verification costs.
Model-free suffix decoding is also interesting for repeated files and agent
loops, and is available through vLLM.
[Suffix implementation](https://github.com/snowflakedb/ArcticInference/blob/main/docs/suffix-decoding.rst),
[vLLM dynamic drafting](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)

**DFlash2 status:** the old project tested it, but current lllm2 explicitly labels
it unintegrated and records ordinary-DFlash pair evidence. A shared engine flag
does not constitute validated integration. The historical Qwen3.8 comparison
approximately matched MTP while adding about 1.1 GB of drafter weights/VRAM cost.
New implementations may warrant retesting, but cannot erase that result or be
assumed faster. [DFlash2 author's report](https://inco.ai/blog/dflash2/)

Gemma 4 has separate MTP assistants for 12B and 26B-A4B, among others. lllm2
checks for MTP tensors inside the target and passes a separate drafter only for
DFlash, so external MTP is a real integration gap. Start with dense 12B when
returning to other models; upstream's implementation report found dense-model
benefits but no MoE benefit on its author's machine. Assistant file sizes are
not total runtime overhead, and matching the target's exact variant matters.
[Google MTP documentation](https://ai.google.dev/gemma/docs/mtp/mtp),
[merged implementation](https://github.com/ggml-org/llama.cpp/pull/23398)

Exact speculative verification preserves the target distribution when correctly
implemented. It does not promise identical seeded stochastic outputs across
different numerical implementations. Approximate acceptance is a separate
experiment. [Original algorithm](https://arxiv.org/abs/2211.17192)

Dedicated speculative file-apply methods are another possibility, but would add
a distinct API/workload rather than a general serving option. They are outside
the first phase. [EfficientEdit research](https://arxiv.org/abs/2506.02780)

### Weight quantization

GGUF/IQ support already exists. The useful addition is comparing variants of the
same checkpoint, including importance-matrix calibration and precision of
sensitive tensors. A nominal four-to-three-bit change for 27B weights would save
about 3.1 GiB before overhead/mixed tensors. Whether that is worthwhile depends
on coding behavior, runtime kernels, and whether it avoids CPU spill.
[Quantizer](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md),
[importance matrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)

Retain checkpoint identity, calibration provenance and actual resident memory.
Smallest file, highest nominal parameter count and best useful model are
different objectives. Extreme quantization needs targeted correctness checks.

## Alternative engines

| Engine | Reason to evaluate | Qualification |
|---|---|---|
| vLLM | Marlin/AWQ/GPTQ kernels, scheduling, paged KV and chunked prefill | Compare single-user latency separately from aggregate throughput; exact model/kernel support matters |
| ExLlamaV3 / TabbyAPI | EXL3 weight precision, 2–8-bit cache, consumer-GPU serving | Separate artifacts; maintainer notes Ampere efficiency needs work |
| SGLang | Radix prefix reuse, scheduling and HiCache | Host caches consume scarce system RAM; caching is not automatically active-context offload |
| ik_llama.cpp | Additional quants and CPU/CUDA execution | Diverged fork, requiring explicit compatibility checks |
| TensorRT-LLM | NVIDIA runtime/quantization paths | Qualify exact SM86/model/release; older engine-building recipes are obsolete on current main |

Marlin is relevant to Ampere/Ada; kernel gains against FP16 are not promised
gains over existing GGUF serving. vLLM's current GGUF support is experimental and
under-optimized; compare a supported native checkpoint path instead.
[Marlin](https://github.com/IST-DASLab/marlin),
[vLLM GGUF status](https://docs.vllm.ai/en/latest/features/quantization/gguf/),
[vLLM scheduling/tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)

ExLlamaV3's extreme low-bit examples establish execution, not acceptable coding
accuracy. HiCache retains reusable state across GPU/host/storage; it should not
be described as automatically enlarging fully attended context.
[ExLlamaV3](https://github.com/turboderp-org/exllamav3),
[HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design),
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

Current TensorRT-LLM documentation removes its old TensorRT engine backend in
favor of PyTorch execution. TensorRT-RTX is a separate product and integration
path, not a llama-server switch. Neither is the first project for this 3090.
[TensorRT-LLM migration](https://nvidia.github.io/TensorRT-LLM/legacy/tensorrt-backend-removal.html),
[TensorRT-RTX architecture](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/architecture/architecture-overview.html)

## Capacity offload and experimental context methods

Selective placement of CPU experts/FFNs/tensors could preserve GPU attention
while fitting a modestly larger model. Whole-layer placement already exists in
lllm2. KTransformers' flagship large-model results require much more host memory:
its single-24-GB-GPU DeepSeek recipe required 382 GB DRAM. This host's memory and
CPU limit its immediate appeal.
[KTransformers recipe](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md)

Lucebox publishes RTX 3090 PFlash/KVFlash recipes and a reported 256K prefill
reduction from 411 to 67.3 seconds for one Laguna configuration. That is an
author-reported result for a specific model and policy, not our Qwen baseline.
It makes a separate engine trial concrete. Selected-block attention requires
information-retention checks: bit-exact storage of evicted blocks does not prove
the output used the same attention information as full attention.
[Lucebox](https://github.com/Luce-Org/lucebox),
[long-context report](https://www.lucebox.com/blog/laguna-xs21)

| Technology | Potential | Readiness/limitation |
|---|---|---|
| TurboQuant | More KV compression | Tracked CPU PR closed unmerged; not a stock CUDA toggle. H100 attention-kernel headlines are not whole-model RTX gains |
| SageAttention | Faster quantized attention, including Ampere kernels | Engine integration and accuracy work; kernel timings omit some overhead |
| MInference / Quest | Sparse prefill or selected KV access | Research integration; verify long-context information remains usable |
| YaRN / RoPE extension | Extend positional context | Does not save KV memory; use exact model recipes and verify behavior beyond metadata limits |
| Pruning/distillation | Smaller/faster models | Mostly model preparation/training, not a generic runtime switch |
| Specialized megakernels | Reduce execution overhead | Narrow architecture/model coverage; small-model results do not generalize to 27B/35B |

Sources:
[TurboQuant explanation](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/),
[unmerged PR](https://github.com/ggml-org/llama.cpp/pull/21089),
[SageAttention](https://github.com/thu-ml/SageAttention),
[MInference](https://github.com/microsoft/MInference),
[Quest](https://github.com/mit-han-lab/Quest),
[Qwen context-extension recipe](https://huggingface.co/Qwen/Qwen3-8B),
[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer),
[megakernel experiment](https://github.com/Luce-Org/lucebox/blob/main/optimizations/megakernel/RESULTS.md).

## RTX generation and upstream version

3090/Ampere lacks native FP8/FP4 arithmetic, while suitable low-bit stored weights
and dequantizing kernels remain useful. Ada adds native FP8 candidates. Consumer
Blackwell adds native FP4 paths, but SM120 support differs from datacenter
Blackwell; do not assume identical cache/kernel support. Stored precision is not
the same as native arithmetic.
[NVIDIA quantization matrix](https://nvidia.github.io/TensorRT-LLM/features/quantization.html)

FlashAttention-3/4 target other hardware/execution paths; installing their Python
packages does not upgrade llama.cpp's 3090 attention implementation.
[FlashAttention support](https://github.com/Dao-AILab/flash-attention)

Multi-GPU placement can increase model/context capacity with additional cards,
but topology and communication affect speed. It has no immediate benefit on
this single-GPU machine.

llama.cpp v0.4.0 was released September 4, after installed b10715, with relevant
CUDA, assistant and cache changes. Compare a separate build before adopting it;
upstream changes can regress particular shapes and models. Keep the existing
control engine intact.
[Release](https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.0),
[example mixed performance results](https://github.com/ggml-org/llama.cpp/pull/25635)

## Measurement requirements

Preserve the existing cold baseline. Add separately labelled warm multi-turn,
copy/edit and eventually concurrent workloads. Measure actual first-token latency,
decode at occupied context, total completion time, peak VRAM and host RAM.
Record exact model/engine/settings and compare equal prompt/output budgets.
Time saved by generating fewer tokens is not an increase in token throughput.

An execution-capacity probe does not demonstrate accurate retrieval or arbitrary
long-context stability. Quantization/sparse-attention changes need a small,
targeted correctness check before promotion; this is distinct from adding a broad
answer-quality leaderboard or a software test infrastructure.

Initial direction: exploit the existing llama.cpp stack for the two Qwen models,
publish evidence-backed 3090 defaults, and keep additional engines and speculative
research as later, individually justified experiments.
