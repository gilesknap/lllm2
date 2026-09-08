# Troubleshoot

- **No engine:** check `lllm2 engines list` and the configured search roots;
  run `lllm2 engines install cuda` if needed.
- **No GPU:** check `nvidia-smi` and the selected engine's device probe.
- **Startup fails:** read the engine log, reduce context or GPU layers, and
  disable unsupported acceleration options. Automatic placement needs an
  engine with memory-fitting support.
- **Client exceeds context:** increase context or reduce slots, then restart
  the model and client. Available memory still limits the allocation.
