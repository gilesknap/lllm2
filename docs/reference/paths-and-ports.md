# Paths and ports

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
