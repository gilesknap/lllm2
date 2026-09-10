# Run Pi in a container

The standalone Pi feature lives in the repository's
[`pi/` directory](https://github.com/gilesknap/lllm2/tree/main/pi).
It includes a Pi-only container with MCP, web access, and a Powerline footer,
plus a launcher for local rootless Podman on Linux.

From your project directory:

```bash
lllm2 pi
```

The current directory is writable at `/workspaces`; host `~/.pi` is shared at
`/root/.pi`. Start a model in lllm2 and add `--provider lllm2` to select it.
Use `--pat` to paste a GitHub PAT at a hidden prompt for that container only.
Existing Pi settings and cloud logins are retained.

The image reuses a pinned claude-sandbox launcher. Pi runs inside its private
network jail, with public internet access and one loopback model-port relay
(default 1920); LAN ranges are blocked.

Only `--pat` is handled by lllm2. All other arguments, including `--help` and
`-e git:github.com/badlogic/pi-skills`, are forwarded to Pi unchanged. The separate
`pi/launch.py` source launcher has container-development options such as
`--image` and `--model-port 0` (disable the relay).

See the [Pi container guide](https://github.com/gilesknap/lllm2/blob/main/pi/README.md)
for image build/publication, extension configuration, PAT handling, and the
precise isolation limits. The default image becomes available after its new
workflow first passes and publishes it.
