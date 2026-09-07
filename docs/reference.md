# Reference

## Commands

Run `lllm2 COMMAND --help` for all options.

| Command | Purpose |
| --- | --- |
| `lllm2` or `lllm2 panel` | Start the panel; `--host` and `--port` control its listener. |
| `lllm2 models [--json]` | List installed GGUF checkpoints. |
| `lllm2 engines list [--json]` | List discovered llama-server builds. |
| `lllm2 engines install cuda\|vulkan` | Build an engine; accepts `--ref`, `--name`, `--jobs` and `--cuda-architectures`. |
| `lllm2 launch` | Serve in the foreground; accepts `--model`, `--engine`, `--backend CUDA\|Vulkan`, `--device` and `--timeout`. |
| `lllm2 claude`, `lllm2 codex`, `lllm2 pi` | Connect an installed coding-agent CLI to the running model. |

## Paths and ports

Set environment variables before starting the workbench or a client wrapper.

| Variable | Default | Contents |
| --- | --- | --- |
| `LLLM2_MODELS_DIR` | `~/models` | GGUF checkpoints and downloads. |
| `LLLM2_STATE_DIR` | `~/.local/state/lllm2` | `workbench.sqlite3` and the panel lock. |
| `LLLM2_ENGINE_HOME` | `~/.local/share/lllm2/engines` | Newly compiled engines. |
| `LLLM2_ENGINE_ROOTS` | Engine home and `~/.local/share/lllm3090` | Colon-separated engine search roots; setting it replaces the defaults. |
| `LLLM2_ENGINE_PORT` | `1920` | Model server port on loopback. |

The panel defaults to `127.0.0.1:8082`. Its controls use `/api/*`; model
clients connect to the separate llama.cpp server on port 1920. Protocol
support depends on the selected engine build.

## Packaged data

`lllm2/models.json` contains the downloadable catalogue;
`lllm2/recommendations.json` records measured recommendations and their
provenance. The package also includes the panel's HTML, CSS and JavaScript
and model-specific chat templates. User settings and experiment results
live in the state directory.
