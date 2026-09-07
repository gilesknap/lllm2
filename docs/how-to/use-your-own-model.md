# Use your own GGUF or engine

Put GGUF files under `~/models`, or set a different directory before starting:

```bash
LLLM2_MODELS_DIR=/data/models lllm2
```

Choose the checkpoint in **All models**. Under **Customize settings**, select
the llama-server binary, backend and GPU. To discover engines elsewhere, set
`LLLM2_ENGINE_ROOTS` to colon-separated directories.
