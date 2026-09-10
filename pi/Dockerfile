# Build from the repository root: podman build -f pi/Dockerfile -t lllm2-pi .
FROM node:22-bookworm-slim AS node
FROM ubuntu:24.04

ARG SANDBOX_REV=1a646951225c89c181e8d4732a5131e7b598fd46
ARG PI_VERSION=0.85.1
LABEL org.opencontainers.image.title="lllm2 Pi" \
      org.opencontainers.image.description="Pi with extensions and the claude-sandbox network jail" \
      org.opencontainers.image.source="https://github.com/gilesknap/lllm2" \
      io.lllm2.pi.version="${PI_VERSION}" \
      io.lllm2.pi.sandbox-revision="${SANDBOX_REV}"

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       bash bubblewrap ca-certificates curl git gh jq passt socat iproute2 \
       ripgrep fd-find procps util-linux \
    && rm -rf /var/lib/apt/lists/*

# Reuse the pinned isolation implementation; install only Pi and its guards.
# Retain upstream source, tests and license for inspection and CI.
RUN git init /opt/claude-sandbox \
    && git -C /opt/claude-sandbox remote add origin https://github.com/diamondlightsource/claude-sandbox.git \
    && git -C /opt/claude-sandbox fetch --depth 1 origin "$SANDBOX_REV" \
    && git -C /opt/claude-sandbox checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/claude-sandbox rev-parse HEAD)" = "$SANDBOX_REV"
RUN bash -c 'set -euo pipefail; \
    source /opt/claude-sandbox/.devcontainer/claude-sandbox/install.sh; \
    install_pi_binary; \
    test -x /usr/libexec/claude-sandbox/pi-dist/pi; \
    install_file "$SCRIPT_DIR/claude-shadow" /usr/local/bin/pi; \
    install_file "$SCRIPT_DIR/pi-run" /usr/libexec/claude-sandbox/pi-upstream-run; \
    install_file "$SCRIPT_DIR/claude-sandbox" /usr/local/bin/claude-sandbox; \
    install_guard_scripts'

WORKDIR /opt/pi
COPY pi/package.json pi/package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts --legacy-peer-deps \
    && npm cache clean --force
COPY pi/smoke.sh pi/test-bundle.mjs /opt/pi/
COPY pi/sandbox.conf /etc/claude-sandbox.conf
COPY pi/entrypoint.sh pi/run.sh pi/seed.mjs pi/defaults.json /opt/pi/
RUN chmod 755 /opt/pi/entrypoint.sh /opt/pi/run.sh \
    && ln -s /opt/pi/run.sh /usr/libexec/claude-sandbox/pi-run \
    && mkdir -p /root/.pi/agent /workspaces
WORKDIR /workspaces
ENTRYPOINT ["/opt/pi/entrypoint.sh"]
