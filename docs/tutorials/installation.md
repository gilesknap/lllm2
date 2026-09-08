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

The CUDA version reported by `nvidia-smi` determines which bundle the host
NVIDIA driver supports:

| Driver reports | Engine bundle |
| --- | --- |
| CUDA 13.3 or newer | CUDA 13.3.1 |
| CUDA 12.9 through 13.2 | CUDA 12.9.1 |
| Below CUDA 12.9 | Update the NVIDIA driver before installing an engine |

Maxwell, Pascal and Volta GPUs select CUDA 12.9.1 even with a newer driver,
using `nvidia-smi`'s compute-capability query. See [NVIDIA's architecture support matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).
The installer conservatively requires support for the bundle's CUDA major and
minor version because GPUs using PTX cannot rely on minor-version compatibility
with older drivers. These requirements follow the dependency pins automatically.
See [NVIDIA's compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Each lllm2 release pins one llama.cpp revision and CUDA version per track.
Releases with unchanged pins leave engine tarballs on their original release.
The installer searches published releases for the newest matching tarball and checksum. Repeating the
install after a Python-only upgrade is a no-op too; older engines remain
available. `engines list` shows the ref, CUDA version, build and installation lllm2 versions
and whether the engine matches the running package's pins. Use `--json`
for full metadata or `--name` to choose an installation directory name.
Development checkouts can also download engines when their pins match published artifacts.

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
