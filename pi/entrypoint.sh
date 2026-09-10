#!/usr/bin/env bash
set -euo pipefail
# Nothing from the mounted project or Pi home executes before entering bwrap.
source /opt/claude-sandbox/.devcontainer/claude-sandbox/install.sh
probe_userns_or_refuse
if [ -f /run/secrets/pi-github-token ]; then
    # Only the trusted GitHub CLI runs here, outside the jail. Avoid project
    # configuration during login. The resulting credential directory is
    # container-scoped, and bwrap masks the source secret from Pi.
    (cd / && gh auth login --hostname github.com --git-protocol https \
        --with-token < /run/secrets/pi-github-token)
    export CLAUDE_SANDBOX_NO_FORGE=0
else
    export CLAUDE_SANDBOX_NO_FORGE=1
fi
exec /usr/local/bin/pi "$@"
