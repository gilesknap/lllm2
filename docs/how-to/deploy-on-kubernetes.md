# Deploy on Kubernetes

The `lllm2` Helm chart runs the panel and its model server in one pod on a GPU
node. The pod gets one whole GPU, and keeps models and state on persistent
volumes. The model API is a ClusterIP Service on port 1920. Nothing is exposed
outside the cluster.

## Before you start

You need:

- A GPU node whose NVIDIA driver reports CUDA 12.9 or newer in `nvidia-smi`.
  The node needs no CUDA toolkit: the image bundles the CUDA runtime.
- The NVIDIA Container Toolkit and device plugin on that node, or the
  [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html),
  so that pods can request `nvidia.com/gpu`.
- A default StorageClass, or claims you have made yourself.
- Helm 3.8 or newer, and kubectl.

The image `ghcr.io/gilesknap/lllm2` carries lllm2 and the CUDA 12.9.1 build of
its pinned llama.cpp engine. Each release's chart runs the image of the same
release, so the chart version picks the lllm2 version and the engine together.

## Install

Write a values file for your node. This one picks the NVIDIA runtime class, an
L40S node and its taint, and a larger models volume:

```yaml
runtimeClassName: nvidia
nodeSelector:
  nvidia.com/gpu.product: NVIDIA-L40S
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
resources:
  # With no requests the pod is the first one evicted when the node runs short
  # of memory, which would end an experiment part way.
  requests:
    cpu: "4"
    memory: 16Gi
persistence:
  models:
    size: 500Gi
```

Install a release of the chart, for example `0.11.0`:

```bash
helm install lllm2 oci://ghcr.io/gilesknap/charts/lllm2 --version 0.11.0 \
  -n lllm2 --create-namespace -f values.yaml
kubectl -n lllm2 rollout status deploy/lllm2
```

Check that the pod sees its GPU and the engine:

```bash
kubectl -n lllm2 exec deploy/lllm2 -- nvidia-smi
kubectl -n lllm2 exec deploy/lllm2 -- lllm2 engines list
```

`engines list` shows the engine under `/opt/lllm2/engines` and says it matches
the running release. Run `helm show values oci://ghcr.io/gilesknap/charts/lllm2`
to see every value. Each release on GitHub also has `lllm2.schema.json`
attached, which editors can use to check a values file.

## Reach the panel

Forward the panel's port from your workstation:

```bash
kubectl -n lllm2 port-forward deploy/lllm2 8082:8082
```

Then open `http://127.0.0.1:8082`. Keep the local port at 8082, as the panel
accepts only its own port in the browser's address.

The panel has no login or TLS. Anyone who reaches it can download models, start
the GPU, and set the engine to any program in the pod and run it. So the chart
makes it listen only on the pod's loopback address. Only people with
`pods/portforward` rights in the namespace can reach it.

There is no Ingress. The panel also refuses a request whose `Host` header is
not its own address, so an Ingress or Service name would not work anyway. To
put the panel behind an authenticating proxy in the cluster, set
`panel.host: 0.0.0.0`. The chart then adds the panel to the Service on port
8082. The proxy must send `Host: localhost:8082`. Anything else that reaches
that port controls the workbench, so allow only the proxy with a
NetworkPolicy.

## Point clients at the engine

Start a model in the panel first. Clients in the cluster use:

```text
http://lllm2.lllm2.svc.cluster.local:1920/v1
```

`helm install` prints this address for your release name and namespace. From
a workstation, forward the port:

```bash
kubectl -n lllm2 port-forward svc/lllm2 1920:1920
```

Clients then use the usual `http://127.0.0.1:1920/v1`, and `lllm2 claude` and
`lllm2 codex` work as they do with a local model (see
[Connect a client](connect-a-client.md)).

The model server has no API key, so any pod that reaches the Service can use
the GPU. In a shared cluster, allow only your clients with a NetworkPolicy.

## Models and storage

The chart creates two ReadWriteOnce claims:

| Claim | Mounted at | Holds | Default size |
| --- | --- | --- | --- |
| `lllm2-models` | `/models` | Downloaded GGUF files | 200Gi |
| `lllm2-data` | `/data` | `/data/state` (settings and results) and `/data/engines` (engines you add) | 10Gi |

Set `persistence.<name>.size` and `storageClassName` before the first install.
To use a volume you have prepared, such as an NFS share, set
`persistence.<name>.existingClaim`. The pod runs as user and group 1000, so the
volume must be writable by group 1000, or set your own `podSecurityContext`.

`helm uninstall` keeps both claims, as models are slow to download again and
the data claim holds your results. A new install with the same release name
and namespace uses them again. Delete them with `kubectl -n lllm2 delete pvc`
when you no longer need them.

**Find models** downloads from Hugging Face, so the pod needs HTTPS egress. Set
a proxy through `env` if your cluster needs one:

```yaml
env:
  - name: HTTPS_PROXY
    value: http://proxy.example.com:3128
```

Without egress, copy GGUF files into `/models` with `kubectl cp`. For an
air-gapped cluster, mirror the image and set `image.repository`.

If you set a memory limit, keep it above the model file size. llama.cpp maps
the model file into memory, and the page cache counts against the limit.

## Upgrade

Wait for running experiments to finish, then:

```bash
helm upgrade lllm2 oci://ghcr.io/gilesknap/charts/lllm2 --version 0.12.0 \
  -n lllm2 -f values.yaml
```

The old pod stops before the new one starts, as both need the GPU and the
panel's state lock. The engine moves to the new release's pin.

To run the newest build of `main` instead of a release, install the chart from
a checkout of `main`:

```bash
helm install lllm2 ./Charts/lllm2 -n lllm2 --create-namespace -f values.yaml
```

It runs the `main` image, which the pod pulls each time it starts, so
`kubectl -n lllm2 rollout restart deploy/lllm2` picks up a newer build.

## Limits

- One pod uses one whole GPU. Time-sliced or MPS-shared GPUs are not
  supported: lllm2 checks that no other process uses its GPU.
- The image carries only the CUDA 12.9.1 engine, which runs on every supported
  driver and GPU. On a driver that reports CUDA 13.3 or newer, you can add the
  CUDA 13.3.1 build with
  `kubectl -n lllm2 exec deploy/lllm2 -- lllm2 engines install cuda`. It goes
  to `/data/engines`, where lllm2 finds it first and uses it. It stays there
  through upgrades, so delete it before `helm upgrade`, or lllm2 may keep
  using the old engine.
- The Modal backend is not installed in the image.
- `lllm2 service` and the panel's upgrade notice do not apply. Upgrade with
  `helm upgrade`.
