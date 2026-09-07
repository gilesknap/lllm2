# Serve from the terminal

```bash
lllm2 models
lllm2 engines list
lllm2 launch --model /data/models/model.gguf --engine /path/to/llama-server
```

With no selection options, `lllm2 launch` looks for a compatible installed
recommendation. It runs in the foreground; Ctrl-C stops the engine.
