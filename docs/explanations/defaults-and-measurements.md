# Defaults and measurements

Defaults are starting points. A matching measured profile can supply tuning;
otherwise the workbench uses inherited estimates or generic settings. The
panel records the source and qualifications. Changing the GPU, checkpoint
or engine can invalidate a measurement; a speed observed on one card is
not a prediction for another.

**Context** is the total allocation shared across **slots**. More slots allow
concurrent conversations but leave less context for each one. Weights, KV
cache and runtime buffers all compete for VRAM. Automatic placement can put
weights in system RAM when the engine supports it, with a possible speed cost.

**Prefill** measures prompt processing; **decode** measures output generation.
Warm runs can reuse prompt tokens, so compare them with like-for-like runs.
A completed speed test alone does not establish quality or stability. Read
the workload checks and failure evidence too.

Context search tests large prompts and an output budget, retaining failed
probes. A timeout is inconclusive, rather than proof of a memory limit.
Try in Launch defaults to **Tested** context, the largest successful probe, or
**Loaded** when only load checks ran.
**90%** selects a smaller, rounded-down estimate instead, and **Original**
keeps the experiment's own context. Neither measured choice
guarantees the same result with different workloads or memory use.
