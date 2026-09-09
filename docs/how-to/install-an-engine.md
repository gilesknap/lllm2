# Install a model engine

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

To try an older driver anyway, run:

```bash
lllm2 engines install cuda --force
```

If driver detection or compatibility checks fail, `--force` selects the CUDA 12
bundle and prints a warning. Successful checks keep the normal bundle selection.
Checksum, archive, metadata and startup checks still apply, and existing engines
are preserved. A successful installation does not guarantee GPU inference will
work: the startup check only runs `llama-server --help`.

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
