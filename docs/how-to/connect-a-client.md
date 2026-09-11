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

For Pi, use claude-sandbox's launcher, which runs it in the same sandbox and
discovers the running lllm2 model on each launch:

```bash
claude-container --host-net --agent pi
```

See [Run Pi in a container](pi-container.md).
