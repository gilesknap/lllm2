# Connect a client

Start a model first. OpenAI-compatible clients use
`http://127.0.0.1:1920/v1`; query `/v1/models` for the served model ID.

For coding agents, the safer starting point is **claude-sandbox**, which runs
the client in a devcontainer and reduces its access to your host environment.
See [Run Pi in claude-sandbox](../tutorials/installation.md#3-run-pi) for setup.
Keep lllm2 running on the host and run the client inside the devcontainer.

If their CLIs are installed, these wrappers configure a local session and
forward additional arguments:

```bash
lllm2 claude
lllm2 codex
```

The Claude and Codex wrappers read the running model's context and slots. Relaunch the client
after changing server context. Use `lllm2 claude -- --help` (or the equivalent
wrapper) to read the client's own help.

`lllm2 pi` launches the [Pi container](pi-container.md), with the current project
writable at `/workspaces` and `~/.pi` shared at `/root/.pi`. It requires local
rootless Podman, not a host Pi installation. It consumes only `--pat`; all
other arguments go to Pi unchanged:

```bash
lllm2 pi --provider lllm2
lllm2 pi --pat -e git:github.com/badlogic/pi-skills
lllm2 pi --help
```
