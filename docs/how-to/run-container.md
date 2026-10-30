# Run in a container

Pre-built containers with lllm2, its Python and its pinned CUDA 12.9.1 engine
already installed are available on
[GitHub Container Registry](https://github.com/gilesknap/lllm2/pkgs/container/lllm2).
Each release has an image tagged with its version. `latest` is the newest
release and `main` is the newest build of the `main` branch.

## Starting the container

To pull the container from GitHub Container Registry and run:

```bash
docker run --rm ghcr.io/gilesknap/lllm2:latest --version
```

To get a released version, use a numbered release instead of `latest`.

## Run the workbench on a GPU

The host needs an NVIDIA driver that reports CUDA 12.9 or newer and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
It needs no CUDA toolkit. Keep models and state in named volumes:

```bash
docker run --rm --gpus all \
  -p 127.0.0.1:8082:8082 -p 127.0.0.1:1920:1920 \
  -v lllm2-models:/models -v lllm2-data:/data \
  ghcr.io/gilesknap/lllm2:latest panel --host 0.0.0.0
```

Open `http://127.0.0.1:8082` and clients use `http://127.0.0.1:1920/v1`, as
with a local install. The panel listens on every address inside the container
so that Docker can forward to it. Publishing both ports on `127.0.0.1` keeps
them on this machine, as neither the panel nor the model server has a login.

With Podman and the toolkit's CDI specification, use
`--device nvidia.com/gpu=all` in place of `--gpus all`.

The image keeps models in `/models`, state in `/data/state`, and engines you
add in `/data/engines`; see [Paths and ports](../reference/paths-and-ports.md).
To run lllm2 on a Kubernetes GPU node, see
[Deploy on Kubernetes](deploy-on-kubernetes.md).
