# Install and serve a model

You need Linux, Python 3.11+, a working NVIDIA driver and enough RAM and disk
space for your chosen model. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

````{admonition} For DLS users
uv is available through the DLS module system. Load it in your shell:

```bash
module load uv
```
````

Then install the workbench:

```bash
uv tool install lllm2
```

## Prepare an engine

Existing llama-server builds under the [engine search paths](../reference/paths-and-ports.md)
are discovered automatically. Otherwise install Git, CMake, a C++ compiler,
the NVIDIA CUDA toolkit and a compiler compatible with that toolkit.

````{admonition} For DLS users
The CUDA toolkit is available through the DLS module system. Load the default
CUDA module in the same shell before building the engine:

```bash
module load cuda
```

This makes `nvcc` available to CMake and avoids the “CUDA Toolkit not found”
error.

Alternatively, build the Vulkan engine, which works at DLS without loading
any modules:

```bash
lllm2 engines install vulkan
```

Store downloaded models on scratch to keep large model files out of your home
directory. Before downloading models, create a scratch directory and link it
to the default `~/models` location. Replace `<fedid>` with your DLS FedID:

```bash
mkdir -p /scratch/<fedid>/models
ln -s /scratch/<fedid>/models ~/models
```

These commands assume `~/models` does not already exist.
````

For CUDA, then run:

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
   the [coding-agent launchers](../how-to/connect-a-client.md).

Use **Stop** to release the GPU. Keep the panel process running while serving.

## Upgrade

Stop the panel and any foreground launch, run `uv tool upgrade lllm2`, then
start the panel again. Engine builds and model downloads are managed separately.
