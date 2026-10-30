# Deploy to Kubernetes with an image and a Helm chart

## Status

Accepted, 1 October 2026.

## Context

lllm2 ran only as a user install on a workstation. Teams with a Kubernetes
cluster and GPU nodes need a supported way to run the panel and engine there
(issue #98). The python-copier-template that lllm2 uses has a container option,
and epics-containers/podbench, made from the same template release, adds a
Helm chart and its CI. Release engines are Rocky Linux 8 tarballs that bundle
the CUDA runtime, so a host needs only the NVIDIA driver.

## Decision

**Engine in the image.** The image build installs the CUDA 12 release engine
with lllm2's own installer and its checksum, archive and pin checks. The image
tag, the lllm2 version and the engine pin then always match, a pod downloads
nothing at start, and an air-gapped cluster mirrors one image. The build has no
GPU to choose a track with. The CUDA 12.9.1 build runs on every driver and GPU
that lllm2 supports, and automatic selection would choose it over the CUDA
13.3.1 build anyway, so the image leaves the 13 build out and saves about
770 MB. CI hands the image an engine it has built but not yet published as a
separate build context.

**Runtime base.** The template's `ubuntu:resolute` plus `ca-certificates`. The
engine needs only glibc and the driver, which the NVIDIA runtime mounts from
the node. uv-managed Python trusts the system certificate directory for model
and engine downloads. A build step fails if an engine library other than the
driver's does not resolve on this base.

**Panel exposure.** The panel has no login or TLS. Its Host check stops DNS
rebinding but is not authentication, and anyone who reaches it can point the
engine setting at any program in the pod and run it. So by default it listens
on the pod's loopback address and is reached with `kubectl port-forward`, which
Kubernetes RBAC controls. An optional Ingress, which signs users in first, is
in [ADR 5](0005-ingress-with-a-sign-in-sidecar.md). The model API is a
ClusterIP Service on port 1920, with no API key.

**Storage.** Two ReadWriteOnce claims, models at `/models` and state plus
extra engines at `/data`, each with an `existingClaim` option. Both are kept on
uninstall, as podbench keeps its claim: models are slow to download again and
the data claim holds results.

**lllm2 in a container.**

- `LLLM2_ENGINE_HOST=0.0.0.0` makes llama-server bind every interface; lllm2
  still reaches it through 127.0.0.1.
- While lllm2's own engine runs, a GPU process that the pod's PID namespace
  cannot see counts as that engine, so a model switch works when nvidia-smi
  reports host PIDs.
- Without `systemctl`, the planner asks the driver whether the card drives a
  display before reserving memory for a desktop.
- The image sets `HOME=/data`, the lllm2 paths under `/models` and `/data`,
  `LLLM2_UPDATE_CHECK=0` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.
- The pod runs as UID and GID 1000 with one replica and the Recreate strategy,
  so the old pod releases the GPU, the state lock and the claims first.

**CI.** `container` builds and tests the image on every run and pushes `main`
and release tags. `helm` lints and renders the chart and pushes it to
`oci://ghcr.io/gilesknap/charts` on release tags. A release waits for both.
Between a llama.cpp bump merging and its release tag, no release has the new
engine, so `main` builds no image and CI stays green for the bump release.

## Consequences

- The image is about 1.2 GB compressed, mostly the engine, so a node's first
  pull is slow.
- The engine layer differs for each build, because the installed engine records
  the lllm2 version that installed it. Every `main` push uploads it again.
- The `main` image lags a bump merge until its release tag.
- `container / build` and `helm / package` should become required checks once
  the bump branch has the new jobs.
- The panel is reached through port-forward unless the optional Ingress is
  turned on.
- Copier updates will conflict in `Dockerfile`, `_container.yml` and `ci.yml`;
  the [template adoption decision](0002-switched-to-python-copier-template.md)
  lists what to keep.
