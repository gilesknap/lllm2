#!/usr/bin/env bash
# Run inside a disposable rootless container; called only by image CI.
set -euo pipefail
command -v node npm rg fdfind
! command -v claude
! command -v codex
# Upstream battery check 18 still looks up the name "claude" even for Pi.
# Give that unmodified test a temporary alias to the same installed shadow.
# This alias exists only in this disposable test container, never the image.
ln -s /usr/local/bin/pi /usr/local/bin/claude
trap 'rm -f /usr/local/bin/claude' EXIT
# Exercise upstream's mock streaming model, real tools, model switching and
# filesystem/network battery against the installed Pi-only image.
CLAUDE_SANDBOX_WORKSPACE_ROOT=/work bash /opt/claude-sandbox/tests/pi_e2e.sh
rm /usr/local/bin/claude
trap - EXIT
# Now start Pi with bundled extensions enabled. RPC needs no provider login.
# Keep stdin open until startup and get_commands have completed.
node /opt/pi/test-bundle.mjs
