# lllm2

A local NVIDIA LLM workbench: download GGUF models, run llama.cpp, compare
settings and save measured results from a browser panel.

## Install and run

On Linux with Python 3.11+ and a working NVIDIA driver,
[install uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv tool install lllm2
lllm2 engines install cuda
lllm2
```

The engine build needs Git, CMake, a C++ compiler and the CUDA toolkit; install
those separately. Existing builds can also be discovered. Open
<http://127.0.0.1:8082>, download a model and click **Start**. The model API
listens at `http://127.0.0.1:1920/v1`.

[Documentation](https://gilesknap.github.io/lllm2/) covers
[getting started](docs/tutorial.md), [common tasks](docs/how-to.md),
[reference](docs/reference.md), [measurements](docs/explanation.md) and
[development and releases](docs/development.md).
