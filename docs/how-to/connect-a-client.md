# Connect a client

Start a model first. OpenAI-compatible clients use
`http://127.0.0.1:1920/v1`; query `/v1/models` for the served model ID.

If their CLIs are installed, these wrappers configure a local session and
forward additional arguments:

```bash
lllm2 claude
lllm2 codex
lllm2 pi
```

The wrappers read the running model's context and slots. Relaunch the client
after changing server context. Use `lllm2 claude -- --help` (or the equivalent
wrapper) to read the client's own help.
