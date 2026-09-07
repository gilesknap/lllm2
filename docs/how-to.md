# Common tasks

## Use your own GGUF or engine

Put GGUF files under `~/models`, or set a different directory before starting:

```bash
LLLM2_MODELS_DIR=/data/models lllm2
```

Choose the checkpoint in **All models**. Under **Customize settings**, select
the llama-server binary, backend and GPU. To discover engines elsewhere, set
`LLLM2_ENGINE_ROOTS` to colon-separated directories.

## Serve from the terminal

```bash
lllm2 models
lllm2 engines list
lllm2 launch --model /data/models/model.gguf --engine /path/to/llama-server
```

With no selection options, `lllm2 launch` looks for a compatible installed
recommendation. It runs in the foreground; Ctrl-C stops the engine.

## Connect a client

Start a model first. OpenAI-compatible clients use
`http://127.0.0.1:1920/v1`; query `/v1/models` for the served model ID.

If their CLIs are installed, these wrappers configure a local session and
forward additional arguments:

```bash
lllm2 claude
lllm2 codex
lllm2 pi
```

The wrappers read the running model's context and slots. Relaunch the client
after changing server context. Use `lllm2 claude -- --help` (or the equivalent
wrapper) to read the client's own help.

## Compare and save settings

In **Experiments**, choose workloads and run a baseline. Change settings and
add combinations, or run **Baseline + single options**. **Discover usable
context** adds context probes; long prompts can take several minutes.

Inspect completed results and their evidence before loading a configuration
into **Launch**. Click **Save my settings** to keep it. Launch and experiment
drafts are separate; loading a result does not save it or restart the model.
Export results as CSV for comparison or JSON for the full record.

## Access over SSH

Forward the panel and API ports from your client machine:

```bash
ssh -N -L 8082:127.0.0.1:8082 -L 1920:127.0.0.1:1920 user@gpu-host
```

Open the usual localhost URL. `lllm2 panel --host 0.0.0.0` also exposes the
panel on a trusted LAN, with no login or TLS; anyone who can reach it can
control the workbench.

## Troubleshoot

- **No engine:** check `lllm2 engines list` and the configured search roots;
  build CUDA or Vulkan if needed.
- **No GPU:** check `nvidia-smi` and the selected engine's device probe.
- **Startup fails:** read the engine log, reduce context or GPU layers, and
  disable unsupported acceleration options. Automatic placement needs an
  engine with memory-fitting support.
- **Client exceeds context:** increase context or reduce slots, then restart
  the model and client. Available memory still limits the allocation.
