# Modal GPU types

Pass the **Type** value to `lllm2 launch --backend modal --gpu`. Memory is the
vendor's nominal figure; the first launch on each type probes the real value.
Prices are Modal's published on-demand prices for one GPU, checked 15 September
2026. They exclude CPU, memory and storage charges. Check
[current Modal pricing](https://modal.com/pricing).

| Type | GPU | Memory (GB) | Estimated $/hour |
| --- | --- | ---: | ---: |
| `T4` | NVIDIA T4 | 16 | 0.590 |
| `L4` | NVIDIA L4 | 24 | 0.799 |
| `A10` | NVIDIA A10 | 24 | 1.102 |
| `L40S` | NVIDIA L40S | 48 | 1.951 |
| `A100-40GB` | NVIDIA A100 40GB | 40 | 2.099 |
| `A100-80GB` | NVIDIA A100 80GB | 80 | 2.498 |
| `RTX-PRO-6000` | NVIDIA RTX PRO 6000 Blackwell | 96 | 3.031 |
| `H100` | NVIDIA H100 | 80 | 3.949 |
| `H200` | NVIDIA H200 | 141 | 4.540 |
| `B200` | NVIDIA B200 | 180 | 6.250 |

Modal may run an `H100` request on an H200. The `B300` is not offered, because
Modal requires CUDA 13.1 or later for it.

## Settings

Saved and panel settings use these fields for a remote backend.

| Field | Values | Meaning |
| --- | --- | --- |
| `backend` | `CUDA`, `Vulkan` or `modal` | `modal` serves the model on Modal. |
| `gpu_type` | A **Type** from the table | Required with `modal`; must be blank for a local backend. |
| `idle_timeout_minutes` | Blank or 0 to 1440 | Minutes without requests before the container stops. Default 30, or `LLLM2_IDLE_TIMEOUT_MINUTES`; 0 or blank disables it. `lllm2 launch` takes `--idle-timeout`, then the variable, then this field. |

A remote backend requires a blank engine path and device `CUDA0`. Saved
defaults for a remote backend are keyed on the model, backend and GPU type.

## Modal resources

| Name | Kind | Contents |
| --- | --- | --- |
| `lllm2` | App | The `probe`, `download` and `serve` functions. |
| `lllm2-models` | Volume | Downloaded models. |
| `lllm2-state` | Dict | Call records, tunnel addresses, heartbeats and download progress. |
| `lllm2-logs` | Queue | llama-server log lines for each call. |

| Limit | Value |
| --- | --- |
| Owner heartbeat grace | 3 minutes |
| Serve call maximum | 12 hours |
| Download call maximum | 2 hours |
