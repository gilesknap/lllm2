# The devcontainer should use the developer target and run as root with podman
# or docker with user namespaces.
FROM ghcr.io/diamondlightsource/ubuntu-devcontainer:resolute AS developer

# Add any system dependencies for the developer/build environment here
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    graphviz \
    curl \
    ca-certificates \
    && apt-get dist-clean

# Match CI Helm and the pre-commit schema plugin; verify downloaded binaries.
ARG HELM_VERSION=v3.17.1
RUN arch="$(dpkg --print-architecture)" \
    && tarball="helm-${HELM_VERSION}-linux-${arch}.tar.gz" \
    && curl -fsSL "https://get.helm.sh/${tarball}" -o "/tmp/${tarball}" \
    && curl -fsSL "https://get.helm.sh/${tarball}.sha256sum" -o "/tmp/${tarball}.sha256sum" \
    && (cd /tmp && sha256sum -c "${tarball}.sha256sum") \
    && tar -xz -C /tmp -f "/tmp/${tarball}" \
    && mv /tmp/linux-*/helm /usr/local/bin/helm \
    && rm -rf /tmp/linux-* "/tmp/${tarball}" "/tmp/${tarball}.sha256sum" \
    && helm plugin install https://github.com/losisin/helm-values-schema-json \
    --version v2.5.0

# CI passes the engine tarballs it built, and has not published yet, as a
# build context of this name. Without one this stage is empty, and the build
# downloads the published engine.
FROM scratch AS engine-dist

# The build stage installs the context into the venv
FROM developer AS build

# Change the working directory to the `app` directory
# and copy in the project
WORKDIR /app
COPY . /app
RUN chmod o+wrX .

# Tell uv sync to install python in a known location so we can copy it out later
ENV UV_PYTHON_INSTALL_DIR=/python

# Compile bytecode now, as the runtime user cannot write to the venv
ENV UV_COMPILE_BYTECODE=1

# Sync the project without its dev dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev --managed-python

# Install this release's CUDA 12 engine with the installer's checksum, archive
# and pin checks. It runs on every driver and GPU that lllm2 supports, and the
# build has no GPU to choose a track with. .github/scripts/image_engines.py
# checks the same track before CI builds.
RUN --mount=type=bind,from=engine-dist,target=/engine-dist \
    /app/.venv/bin/python - <<'PY'
from pathlib import Path

from lllm2.engine_install import install
from lllm2.engine_release import asset_name

track = "12"
dist = Path("/engine-dist")
asset = asset_name(track)
built = all((dist / name).is_file() for name in (asset, asset + ".sha256"))
install(
    "cuda",
    track=track,
    root=Path("/opt/lllm2/engines"),
    check_startup=False,
    source=dist if built else None,
)
PY

# The runtime stage copies the built venv into a runtime container
FROM ubuntu:resolute AS runtime

# Model and engine downloads need the CA certificates. The engine bundles the
# CUDA runtime: it needs only glibc here, and the driver from the node.
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    && apt-get dist-clean

COPY --from=build /opt/lllm2/engines /opt/lllm2/engines

# Every library an engine loads must resolve here, except the driver's.
RUN set -eu; \
    for server in /opt/lllm2/engines/*/llama-server; do \
        test -x "$server"; \
        LD_LIBRARY_PATH="${server%/*}" ldd "$server" > /tmp/ldd.txt; \
        if grep 'not found' /tmp/ldd.txt | grep -v -e 'libcuda\.so' -e 'libnvidia-'; then \
            exit 1; \
        fi; \
    done; \
    rm /tmp/ldd.txt

# Copy the python installation from the build stage
COPY --from=build /python /python

# Copy the environment, but not the source code
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH

# The chart mounts its claims at /models and /data. Any user can write both
# here, for a container run without volumes. HOME is /data because lllm2
# derives its default paths from it.
RUN mkdir -p /models /data && chmod 1777 /models /data
ENV HOME=/data \
    LLLM2_MODELS_DIR=/models \
    LLLM2_STATE_DIR=/data/state \
    LLLM2_ENGINE_HOME=/data/engines \
    LLLM2_ENGINE_ROOTS=/opt/lllm2/engines:/data/engines
# Serve the model API on the pod address. lllm2 still uses 127.0.0.1.
ENV LLLM2_ENGINE_HOST=0.0.0.0
# A new image upgrades lllm2, so the panel's upgrade notice does not apply.
ENV LLLM2_UPDATE_CHECK=0
# The NVIDIA runtime mounts the driver and nvidia-smi. The device plugin sets
# NVIDIA_VISIBLE_DEVICES, so the image leaves it unset.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Check the managed interpreter works on the runtime base.
RUN lllm2 --version && lllm2 engines list

# change this entrypoint if it is not the same as the repo
ENTRYPOINT ["lllm2"]
CMD ["--version"]
