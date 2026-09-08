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
uv tool install --upgrade lllm2
```

## Prepare an engine

Existing llama-server builds under the [engine search paths](../reference/paths-and-ports.md)
are discovered automatically. To install the engine pinned to your lllm2 release:

```bash
lllm2 engines install cuda
lllm2 engines list
```

The installer downloads a checksum-verified release artifact into your user
engine directory. You need a working NVIDIA driver; no CUDA toolkit, compiler,
container runtime or module load is needed on the host. Artifacts target Linux
x86_64 with glibc 2.28 or newer and an AVX2-capable CPU.

The CUDA version reported by `nvidia-smi` selects the CUDA 13 track for drivers
reporting 13 or newer, or CUDA 12.9 for drivers reporting 12.x. CUDA 12 supports
older Pascal/Volta cards. NVIDIA's minor-version compatibility has limitations:
GPUs using PTX may need a driver supporting the artifact's full CUDA version.
See [NVIDIA's compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Each lllm2 release pins one llama.cpp revision and CUDA version per track.
Releases with unchanged pins reuse identical engine tarballs. Repeating the
install after a Python-only upgrade is a no-op too; older engines remain
available. `engines list` shows the ref, CUDA version, installing lllm2 version
and whether the engine matches the running package's pins. Use `--json`
for full metadata or `--name` to choose an installation directory name.
Development checkouts need a published lllm2 version to download release engines.

Vulkan installation is no longer offered. Hand-placed Vulkan engines and saved
Vulkan preferences continue to work through normal discovery and launch.

````{admonition} For DLS users
Store downloaded models on scratch to keep large model files out of your home
directory. If `~/models` does not already exist, replace `<fedid>` with your FedID:

```bash
mkdir -p /scratch/<fedid>/models
ln -s /scratch/<fedid>/models ~/models
```
````

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
run `lllm2 engines install cuda`, then start the panel again. This downloads the
engine required by the new release, unless the same pins are already installed.
Existing engines and models are preserved.
