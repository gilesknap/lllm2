# Run Pi in a container

lllm2 no longer ships its own Pi container. Use claude-sandbox's launcher,
which runs Pi in the same sandbox it gives Claude Code and Codex, and already
discovers the lllm2 model:

```bash
claude-container --host-net --agent pi
```

`--host-net` shares the model server's network namespace so Pi can reach
`http://127.0.0.1:1920` on the host. Inside Pi, choose the lllm2 model with
`--provider lllm2` or from `/model`; discovery refreshes at each launch, so a
model change in lllm2 only needs a Pi restart.

The launcher persists `~/.pi` between sessions, so extensions installed once
from inside Pi (for example `pi install npm:pi-mcp-adapter`) stay installed.

See claude-sandbox's
[Use Pi](https://diamondlightsource.github.io/claude-sandbox/how-to/use-pi.html)
guide for installation, cloud logins, forge authentication and the network
isolation limits.
