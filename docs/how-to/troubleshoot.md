# Troubleshoot

- **No engine:** check `lllm2 engines list` and the configured search roots;
  build CUDA or Vulkan if needed.
- **Engine build refuses to start:** the missing dependency is named with the
  commands that install it. A toolkit installed outside `PATH` is reported as
  such, with the exports that reveal it to CMake. Use `--skip-checks` only when
  a dependency is installed somewhere the checks cannot see.
- **Toolkit too old for the card:** the check compares the compute capability
  `nvidia-smi` reports against the architectures `nvcc` can generate, and names
  the CUDA release the GPU needs. Distribution packages lag: Ubuntu 24.04 ships
  CUDA 12.0, which cannot target Blackwell, so the hint also gives NVIDIA's
  repository commands for a new enough toolkit.
- **No GPU:** check `nvidia-smi` and the selected engine's device probe.
- **Startup fails:** read the engine log, reduce context or GPU layers, and
  disable unsupported acceleration options. Automatic placement needs an
  engine with memory-fitting support.
- **Client exceeds context:** increase context or reduce slots, then restart
  the model and client. Available memory still limits the allocation.
