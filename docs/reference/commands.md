# Commands

Run `lllm2 COMMAND --help` for all options.

| Command | Purpose |
| --- | --- |
| `lllm2 --version` | Print the installed package version and exit. |
| `lllm2 service install` | Install, enable and start the panel user service; accepts `--host`, `--port` and `--no-start`. |
| `lllm2` or `lllm2 panel` | Start the panel; `--host` and `--port` control its listener. |
| `lllm2 models [--json]` | List installed GGUF checkpoints. |
| `lllm2 engines list [--json]` | List discovered llama-server builds. |
| `lllm2 engines install cuda` | Download this release’s pinned CUDA engine; accepts `--name` and `--force` (try CUDA 12 when driver checks fail). |
| `lllm2 launch` | Serve in the foreground using the selected model/backend's saved workbench settings when available; accepts `--model`, `--engine`, `--backend CUDA\|Vulkan`, `--device` and `--timeout`. |
| `lllm2 claude`, `lllm2 codex` | Connect an installed coding-agent CLI to the running model. |
| `lllm2 pi` | Launch the Pi container with local rootless Podman. Only `--pat` is consumed by lllm2; all other arguments go to Pi. |

Engine installation shows download progress, transfer speed, and estimated time
remaining, followed by checksum, extraction, and startup status. Progress goes to
stderr; stdout contains the installed engine path. Redirected output uses periodic
text updates instead of an animated bar. If the server does not provide a download
size, the display shows bytes transferred without a percentage or time estimate.
