# Commands

Run `lllm2 COMMAND --help` for all options.

| Command | Purpose |
| --- | --- |
| `lllm2 --version` | Print the installed package version and exit. |
| `lllm2 service install` | Install, enable and start the panel user service; accepts `--host`, `--port` and `--no-start`. |
| `lllm2` or `lllm2 panel` | Start the panel; `--host` and `--port` control its listener. |
| `lllm2 models [--json]` | List installed GGUF checkpoints. |
| `lllm2 engines list [--json]` | List discovered llama-server builds. |
| `lllm2 engines install cuda\|vulkan` | Build an engine; accepts `--ref`, `--name`, `--jobs` and `--cuda-architectures`. |
| `lllm2 launch` | Serve in the foreground; accepts `--model`, `--engine`, `--backend CUDA\|Vulkan`, `--device` and `--timeout`. |
| `lllm2 claude`, `lllm2 codex`, `lllm2 pi` | Connect an installed coding-agent CLI to the running model. |
