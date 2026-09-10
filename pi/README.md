# Pi container

A Pi-only coding-agent image with a small extension bundle and the existing
[claude-sandbox](https://github.com/diamondlightsource/claude-sandbox) isolation.
No Claude Code or Codex binaries are installed. The sandbox implementation is
fetched at a pinned commit during the build, not copied into this repository.

## Run

Requirements: Linux, Python 3.11+, **local rootless Podman**, `/dev/net/tun`, and
unprivileged nested user namespaces. Run on the host as your ordinary user.
Docker, remote engines, and rootful Podman are not supported by this launcher.
The namespace probe refuses to launch when the host cannot provide isolation.

After the image workflow publishes `ghcr.io/gilesknap/lllm2-pi:latest`:

```bash
cd /path/to/project
python3 /path/to/lllm2/pi/launch.py
# Start a model in lllm2 first to use the local provider:
python3 /path/to/lllm2/pi/launch.py -- --provider lllm2
# Prompt for a GitHub PAT without echoing it:
python3 /path/to/lllm2/pi/launch.py --pat
# Cloud-only session: no local model relay
python3 /path/to/lllm2/pi/launch.py --model-port 0
# Inspect the invocation without launching or prompting:
python3 /path/to/lllm2/pi/launch.py --dry-run --pat
```

Use `--help` for launcher options and `-- --help` for Pi's help. Arguments after
`--` go to Pi literally, including spaces and shell punctuation. `lllm2 pi`
remains unchanged: CLI integration is deferred so this feature only changes
`pi/`, workflows, and docs.

Each invocation creates a fresh container and removes it on exit. The current
directory is mounted **read-write at `/workspaces`**, which is also the working
directory. Host `~/.pi` is mounted **read-write at `/root/.pi`**. Symlinks are
resolved, so a `~/.pi` symlink into a shared terminal configuration works too.
Use `--pi-dir /path/to/.pi` to choose another store. Sessions, logins, installed
packages and settings persist there, as in claude-sandbox. Other home directories,
SSH agents, Docker sockets, and forge credentials are not mounted.

The image's Pi binary and bundled extensions are read-only inside the jail.
New homes receive dark-theme/Powerline defaults; existing settings are never
replaced or merged. The upstream local-model discovery updates only its provider
in `models.json`, preserving other providers and preserving the file if discovery
fails. The selected project and shared `.pi` remain writable by Pi and plugins:
that includes credentials and executable extensions. Use a separate `--pi-dir`
when you need separate trust boundaries between projects.

## Extensions and appearance

| Included package | Purpose |
| --- | --- |
| [pi-mcp-adapter](https://pi.dev/packages/pi-mcp-adapter) | Lazy MCP tool discovery and `/mcp` configuration |
| [pi-web-access](https://pi.dev/packages/pi-web-access) | Web search and page extraction |
| [pi-powerline](https://pi.dev/packages/pi-powerline) | Styled footer, breadcrumb, and gradient header |

Exact versions and transitive integrity hashes live in `package.json` and
`package-lock.json`. Install scripts are disabled during the build. Packages run
inside the same jail as Pi. MCP servers are not preconfigured; their commands
must exist in the image or project, and remote servers must obey the network
policy. Node/npm, Git/gh, curl, jq, ripgrep, and fd are available. This is a small
agent image, not a full language toolchain or browser development environment.

Bundled extensions are supplied as explicit local `--extension` arguments so
mounting an existing `.pi` does not hide them or require first-run downloads.
Use `-- --no-extensions` to disable the bundle and auto-discovered extensions
for a session. Additional explicit `--extension` arguments still work. If your
shared configuration already installs these packages, remove the duplicate
user-installed copies or opt out of the image bundle. To add another package:

```bash
python3 /path/to/lllm2/pi/launch.py -- install npm:PACKAGE
```

The initial footer uses the dark theme, a breadcrumb inside the editor, and a
styled header without diagnostic startup information. Powerline uses text
fallbacks when the terminal does not advertise Nerd Font support. Change
`~/.pi/agent/settings.json` to adjust its `powerline`, `breadcrumb`, `footer`,
`header`, and `header-info` settings. User settings take precedence over seeding.

## GitHub PAT

`--pat` reads the token with hidden terminal input. It refuses to fall back to
visible input when no suitable terminal exists. The launcher writes a temporary
0600 file in a private temporary directory, mounts it read-only, and runs
`gh auth login --with-token` before starting the sandbox. The token is never put
in command arguments, container environment metadata, or the shared `.pi` store.
The source file is masked inside the sandbox and deleted when the launcher exits.

The resulting `gh` credential directory is scoped to this disposable container
and is explicitly made accessible to Pi. Git over HTTPS and `gh` can then use it.
The agent can read and use this PAT; choose a repository-limited token with the
permissions you intend to grant. Pi could copy an accessible credential into a
writable directory, so temporary storage is not a secret boundary against Pi.
Without `--pat`, forge credentials are hidden. No host `gh` login is reused.

## Networking

The **outer** container uses host networking so the model can stay bound to
`127.0.0.1:1920`. Before any Pi/plugin code runs, the upstream launcher puts it
in a separate network namespace, with the same pasta/bubblewrap egress jail as
claude-sandbox:

- Public internet access for model APIs, web access, and package downloads.
- RFC1918, CGNAT, link-local, and connected LAN subnets blocked by routing policy.
- IPv4-only forwarding; no IPv6 path around those blocks.
- DNS via the jail's forwarding endpoint.
- A relay for one host loopback TCP port, default **1920**. `--model-port 0`
  disables it; another port can be selected explicitly.

No proxy sidecar, domain allowlist, or site-specific `allow-ip` exception is
needed. This preserves the upstream network policy instead of substituting a
plain container bridge. The relay exposes the entire selected TCP service,
including any administrative API endpoints it offers. Public internet access
also permits uploading readable project data: this is not an anti-exfiltration
policy. Do not point the relay at an unrelated sensitive service.

The launcher permits the outer seccomp/AppArmor operations needed for nested
user namespaces and disables SELinux mount labelling, matching the upstream CI
runtime. It never uses `--privileged`. The security boundary is the rootless
container plus the inner jail, which drops capabilities, isolates processes,
scrubs environment variables, and makes the image filesystem read-only. It is
not a VM boundary. A custom image or overridden entrypoint is outside this
launcher's supported safety contract.

`PI_OFFLINE=1` appears only in CI probes: it suppresses Pi's background catalog
and update activity to keep tests deterministic. It does **not** prohibit
network access. Normal sessions do not set it.

## Build, test, update, publish

```bash
podman build -f pi/Dockerfile -t lllm2-pi .
python3 pi/launch.py --image lllm2-pi -- --version
python3 -m unittest discover -s pi -p 'test_*.py'
node --test pi/test_seed.mjs
```

The `Pi container` workflow builds natively on amd64 and arm64, runs the upstream
Pi streaming/tool-call and sandbox battery tests, checks bundled extension
startup, and exercises this launcher before publishing a multi-architecture
image to `ghcr.io/OWNER/REPOSITORY-pi`. PR builds do not publish. Main builds
publish `latest` and a commit tag; releases also publish the Git tag. A first
GHCR publication may require changing package visibility to public.

Pi and the sandbox commit are pinned in `Dockerfile`. Change those pins or the
extension versions deliberately, regenerate the lock with
`npm install --prefix pi --package-lock-only --ignore-scripts --legacy-peer-deps`,
and run the workflow. The OS/Node base tags receive updates on rebuild; this is
not a fully reproducible base-image pin. Existing local images are reused until
pulled again; use `podman pull ghcr.io/gilesknap/lllm2-pi:latest` to update or
`--image IMAGE@sha256:DIGEST` for a fixed image.
