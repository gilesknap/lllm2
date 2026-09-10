#!/usr/bin/env bash
set -euo pipefail
if [ "${IS_SANDBOX:-}" != 1 ] || [ "${IS_SANDBOX_AGENT:-}" != pi ]; then
    echo 'Pi must be launched through the sandbox.' >&2
    exit 2
fi
# All shared-config handling and third-party code run INSIDE the jail.
node /opt/pi/seed.mjs
case "${1:-}" in
    install|remove|update|list|config|--help|-h|--version|-v)
        exec /usr/libexec/claude-sandbox/pi-upstream-run "$@" ;;
esac
bundle=(
    --extension /opt/pi/node_modules/pi-mcp-adapter
    --extension /opt/pi/node_modules/pi-web-access
    --extension /opt/pi/node_modules/pi-powerline
    --append-system-prompt /opt/pi/context.md
)
# Pi loads explicit --extension paths even with --no-extensions. Honour that
# flag as an opt-out from this image's defaults, including the shared context
# appended to the system prompt.
for arg in "$@"; do
    if [ "$arg" = --no-extensions ]; then bundle=(); break; fi
done
exec /usr/libexec/claude-sandbox/pi-upstream-run "${bundle[@]}" "$@"
