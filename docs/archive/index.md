# Historical plans and research

These records describe work completed during 6–7 September 2026. They are kept
for research sources, measurement caveats and implementation history. They are
**not current instructions or a roadmap**. Statements about permissions, absent
tests/CI/documentation, local paths and subsequent work belong to those sessions.
The original contents are retained apart from archive notices and repaired links;
status lines such as “awaiting user review” have not been rewritten as current claims.

Use the [current documentation](../index.md),
[development guide](../how-to/development.md) and
[defaults and measurements](../explanations/defaults-and-measurements.md) for
maintained guidance. Runtime recommendation evidence lives in the packaged
`recommendations.json` and saved experiment records.

```{toctree}
:maxdepth: 1

PERFORMANCE_PLAN
RTX_PERFORMANCE_REVIEW
UI_IMPROVEMENTS_PLAN
UI_USABILITY_PLAN
```

## Retired handoffs

The original [PLAN.md](https://github.com/gilesknap/lllm2/blob/d42f1b6ad45a1decbaa5794d68322c12512d0658/PLAN.md) and
[HANDOFF.md](https://github.com/gilesknap/lllm2/blob/d42f1b6ad45a1decbaa5794d68322c12512d0658/HANDOFF.md) are available at the last revision before retirement.
They have been removed from the working tree because their setup, authority and
next-session instructions are superseded. Their historical validation records
remain accessible through these permanent links.

## Handoff follow-up review, 9 September 2026

The following review preserves unresolved work without treating the old handoff
as current authority. This was a source inspection, not new GPU validation.

| Original follow-up | Finding at retirement |
| --- | --- |
| Download artifact identity | Partially addressed: newly discovered HF variants pin a repository revision. Seed entries still default to `main`, and model downloads validate the GGUF header rather than a final content digest. Resume identity and integrity remain follow-ups. |
| Qwen3.8 reasoning effort | Still relevant: shared settings allow `minimal` and `max`, while the bundled Qwen3.8 template accepts `low`, `medium`, `xhigh`, and maps `high` to `xhigh`. Model-specific choices need qualification. |
| Unreadable custom templates | Still relevant: capability inspection directly reads the template. Launch validation checks file existence, but capability reporting could give a clearer unavailable state for read failures. |
| Empty engine-root entries | Addressed: `config.ENGINE_ROOTS` filters out empty entries. |
| Context ceiling below the workload minimum | Addressed: context search records a skipped result with an explicit minimum-budget explanation. |
| Result size and polling cost | Partially addressed: status polling is guarded against overlap, and result summaries omit prompt payloads. Full records retain provenance; summary refresh cost remains worth profiling as the database grows. |
| AMD/Intel Vulkan support | Remains outside the NVIDIA workbench scope; no hardware scope change is implied by this archive. |

Existing issues retain the larger research threads:
[concurrent coding requests and vLLM comparison](https://github.com/gilesknap/lllm2/issues/2),
[panel downloads for DFlash drafters](https://github.com/gilesknap/lllm2/issues/7), and
[CUDA mixed K/V kernels](https://github.com/gilesknap/lllm2/issues/13).
Other ideas in the performance plan's “Later work” section remain historical
research candidates, rather than accepted implementation tasks.
