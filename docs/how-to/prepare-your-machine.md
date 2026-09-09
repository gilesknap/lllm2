# Prepare your machine

You need Linux, Python 3.11+, a working NVIDIA driver and enough RAM and disk
space for your chosen model. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
See [Install a model engine](install-an-engine.md) for CPU and driver requirements.

## DLS machines

Load uv through the module system:

```bash
module load uv
```

Store downloaded models on scratch to keep large model files out of your home
directory. If `~/models` does not already exist, replace `<fedid>` with your FedID:

```bash
mkdir -p /scratch/<fedid>/models
ln -s /scratch/<fedid>/models ~/models
```

Other locations can be selected with
[LLLM2_MODELS_DIR](../reference/paths-and-ports.md).

## Coding client

For the Pi devcontainer workflow, install VS Code with the Dev Containers
extension and a supported container runtime. Set up GitHub SSH access so you
can clone the sandbox repository.
