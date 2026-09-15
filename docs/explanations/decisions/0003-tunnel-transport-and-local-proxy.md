# Serve remote engines through a tunnel and a local proxy

## Status

Accepted, 15 September 2026.

## Context

lllm2 can run llama-server on a remote GPU, starting with Modal. Bench, warm,
harness wrappers, the panel and claude-sandbox all reach the engine at a
loopback URL on the engine port, and bench depends on llama-server's raw
completion stream.

Modal offers two ways to reach a server in a container. A web endpoint gives a
stable HTTPS URL but applies Modal's request timeout limits, which a long
prompt can exceed. A tunnel exposes a container port directly for as long as the
function runs.

Another provider must be able to plug in without changes to consumers.

## Decision

The serve function opens a Modal tunnel to llama-server's port and publishes
the tunnel address. llama-server requires an API key that lllm2 generates for
each launch.

lllm2 runs a streaming HTTP proxy on `127.0.0.1` at the usual engine port. It
forwards requests to the tunnel without buffering and adds the API key.
`Engine.base` stays a loopback URL for both local and remote engines.

The proxy and the engine lifecycle live in a provider-neutral `RemoteEngine`.
Each provider implements the `RemoteProvider` interface and registers a factory
and a GPU table.

## Consequences

- Consumers need no changes, and clients keep sending placeholder tokens.
- llama-server's full native API is available, with no provider request
  timeout on long prompts.
- Anyone who learns the tunnel address can reach llama-server's endpoints that
  need no key: `/health`, `/v1/health` and the web UI files. Completions and
  the model list need the key.
- The local lllm2 process must stay running while the model serves. The proxy
  also sees every request, which drives the idle timer.
- The tunnel address changes on every launch, so a client cannot connect to the
  remote server without lllm2.
- A new provider needs only a `RemoteProvider` implementation and a GPU table.
