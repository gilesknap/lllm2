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
| `lllm2 launch` | Serve in the foreground using the selected model/backend's saved workbench settings when available; accepts `--model`, `--engine`, `--backend CUDA\|Vulkan\|modal`, `--device` and `--timeout`. |
| `lllm2 launch --backend modal --gpu TYPE` | Serve on a Modal GPU through the local engine port; `--model` also accepts a catalogue id or name. `--idle-timeout MINUTES` accepts 0 to 1440 or `off`, and `0` or `off` disables it. Without it, lllm2 uses `LLLM2_IDLE_TIMEOUT_MINUTES`, then saved settings, then 30. See [Modal GPU types](modal-gpus.md). |
| `lllm2 modal setup` | Check Modal credentials and deploy the lllm2 app unless this version is already deployed. |
| `lllm2 modal probe --gpu TYPE [--json]` | Run a short, billed probe container, print the GPU name, memory, engine devices, the engine build (llama.cpp release, CUDA track and compiler) and its sha256, and save the result for later launches. |
| `lllm2 modal download ID` | Download a catalogue model and its companion files into the Modal Volume, with progress; a stored model downloads nothing. Ctrl-C cancels, and a rerun resumes. |
| `lllm2 modal list [--json]` | List running lllm2 serve calls with owner, elapsed time and estimated cost. |
| `lllm2 modal stop [CALL_ID\|--all] [--force]` | Stop one call, or every orphaned call with `--all`; a call whose owner still heartbeats is skipped, and `--force` stops one such call by id. |
| `lllm2 modal models [--json]` | List models stored in the Modal Volume, with sizes and catalogue ids. |
| `lllm2 modal remove ID\|NAME` | Delete a stored model, by catalogue id or the name `models` prints, with its companion files, unless a running call serves it. |
| `lllm2 claude`, `lllm2 codex` | Connect an installed coding-agent CLI to the running model. |

Engine installation shows download progress, transfer speed, and estimated time
remaining, followed by checksum, extraction, and startup status. Progress goes to
stderr; stdout contains the installed engine path. Redirected output uses periodic
text updates instead of an animated bar. If the server does not provide a download
size, the display shows bytes transferred without a percentage or time estimate.
