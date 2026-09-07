# Install and serve a model

You need Linux, Python 3.11+, a working NVIDIA driver and enough RAM and disk
space for your chosen model. Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
then install the workbench:

```bash
uv tool install lllm2
```

## Prepare an engine

Existing llama-server builds under the [engine search paths](reference.md#paths-and-ports)
are discovered automatically. Otherwise install Git, CMake, a C++ compiler,
the NVIDIA CUDA toolkit and a compiler compatible with that toolkit, then run:

```bash
lllm2 engines install cuda
lllm2 engines list
```

This compiles llama.cpp into your user directory. System prerequisites are
installed separately; missing build tools produce installation hints.
For Vulkan, use `lllm2 engines install vulkan` with Vulkan development
libraries and `glslc` installed. Use `--ref TAG` to choose a particular
llama.cpp revision; the default is `master`.

## Start the panel

```bash
lllm2
```

1. Open <http://127.0.0.1:8082> and choose a model from the catalogue.
2. Download it, or select an installed checkpoint in **All models**.
3. Review **Customize settings** if needed, then click **Start**.
4. When ready, connect a client to `http://127.0.0.1:1920/v1`, or use one of
   the [coding-agent launchers](how-to.md#connect-a-client).

Use **Stop** to release the GPU. Keep the panel process running while serving.

## Upgrade

Stop the panel and any foreground launch, run `uv tool upgrade lllm2`, then
start the panel again. Engine builds and model downloads are managed separately.
