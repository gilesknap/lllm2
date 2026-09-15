# Paths and ports

Set environment variables before starting the workbench or a client wrapper.

| Variable | Default | Contents |
| --- | --- | --- |
| `LLLM2_MODELS_DIR` | `~/models` | GGUF checkpoints and downloads. |
| `LLLM2_STATE_DIR` | `~/.local/state/lllm2` | `workbench.sqlite3`, the panel lock, and the remote files `remote-calls.json` (owned serve calls, mode 0600) and `remote-probes.json` (GPU probe results). |
| `LLLM2_ENGINE_HOME` | `~/.local/share/lllm2/engines` | Newly compiled engines. |
| `LLLM2_ENGINE_ROOTS` | `LLLM2_ENGINE_HOME` | Colon-separated engine search roots; setting it replaces the defaults. |
| `LLLM2_ENGINE_PORT` | `1920` | Model server port on loopback. A remote backend's local proxy uses the same port. |
| `LLLM2_IDLE_TIMEOUT_MINUTES` | `30` | Default idle stop for a remote backend, in whole minutes from 0 to 1440; `0` or `off` disables it. For `lllm2 launch`, `--idle-timeout` overrides it, and it overrides saved settings. The panel fills its idle field from it. lllm2 reads it at startup. |

The panel defaults to `127.0.0.1:8082`. Its controls use `/api/*`; model
clients connect to the separate llama.cpp server on port 1920. Protocol
support depends on the selected engine build.
