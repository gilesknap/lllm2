# Remote engines

Bench, warm, harness wrappers and clients only talk HTTP to the engine
endpoint. lllm2 keeps that endpoint the same for a remote GPU, so none of them
needs to know where the model runs.

## Engines

`Engine` is the abstract base. It holds what every engine shares: the HTTP
request helpers, the streaming completion, the guarded operation with its
wall-clock bound, the bounded log buffer and `base`, the loopback URL.
Bench and warm use `alive()` and `logs()` and never branch on the backend.

Two implementations exist:

- `LocalEngine` runs llama-server as a process on this machine, with the local
  GPU checks: ownership, desktop coexistence and port occupancy.
- `RemoteEngine` runs llama-server through a remote provider. It skips the
  local GPU checks and does not sample host memory.

`backends.create_engine` picks one from the settings. `CUDA` and `Vulkan` give
a `LocalEngine`; any other backend names a provider.

## Providers

A `RemoteProvider` rents out a GPU. Its interface covers probing a GPU type,
placing a model in its store, listing and removing stored models, and spawning,
polling, cancelling and listing serve calls. `RemoteEngine` drives any provider
through that interface.

A provider registers two things under the same name:

- a factory, with `remote.register_provider`, which may import the provider's
  client library on first use;
- a GPU table, with `gpu_tables.register_gpu_table`, listing GPU types, memory
  and hourly prices.

The settings then accept the name as a backend, and launch arguments, starting
defaults and cost estimates work from the table. Modal is the only provider
today: `modal_provider` implements the interface and `modal_app` holds the
functions that run in Modal. `modal_app` imports without the `modal` package,
so a local-only install has no Modal dependency.

## Launch

Launch arguments come from the same builder as a local launch. The provider's
probe supplies the engine capabilities and GPU memory, and the downloaded
model's GGUF metadata supplies the model facts. Until a GPU type has been
probed, validation uses the table figures and the pinned engine release's flags,
so it never starts a container.

The remote container installs the same lllm2 CUDA engine release as
`lllm2 engines install cuda`, pinned by version, so a remote and a local
engine for the same lllm2 version come from the same build.

## Proxy

`EngineProxy` listens on `127.0.0.1` at the engine port. It streams request and
response bodies without buffering, so server-sent events reach the client byte
for byte. Reads have no timeout, because prompt processing can take minutes
before the first response byte. Each client request opens its own upstream
connection, so concurrent requests do not wait for each other. Until the remote
server is reachable, the proxy answers 503.

Every proxied request restarts the idle timer. When the timer expires,
`RemoteEngine` cancels the serve call.

## Security

- **Loopback only.** The proxy binds `127.0.0.1`, as a local llama-server does.
- **Per-launch API key.** Each launch generates a random key. llama-server
  receives it as the `LLAMA_API_KEY` environment variable, never on the command
  line. The proxy removes client credentials and sends the key with every
  request, so clients keep their placeholder token.
- **Encrypted tunnel.** The serve container opens a TLS Modal tunnel to
  llama-server. The tunnel address alone grants no access to completions or
  to the model list. llama-server answers `/health`, `/v1/health` and its web
  UI files without the key, so anyone who learns the address can see that a
  server runs.
- **Private records.** lllm2 saves each owned call, including its key, in
  `remote-calls.json` in the state directory with mode 0600. A later session
  uses the record to adopt a call that is still running.

## Lifetime

lllm2 owns the container's lifetime, because a forgotten GPU keeps billing.
Several mechanisms stop a call:

- Stop in the panel, Ctrl-C in `lllm2 launch` and panel shutdown cancel owned
  calls.
- The idle timer cancels a call with no requests.
- The owning process sends heartbeats to serve and download containers while
  it drives them. A container whose owner is silent for about 3 minutes stops
  itself.
- Modal ends a serve call after 12 hours and a download call after 2 hours.

That heartbeat also decides who owns a call. The provider reports how long ago
each call's owner reported, and a call whose heartbeat is fresh belongs to a
live session wherever that session runs. Only a call whose heartbeat has gone
stale is an orphan. This matters because local records are per-workstation: a
panel run by systemd, a CLI in a container and a CLI on the host each have
their own state directory, so a local record proves nothing about calls started
elsewhere. Records still decide what this workstation can do about a call,
namely adopt it when it holds the key, and they report the one thing the
heartbeat cannot: an owner process on this host that has since died, whose call
is an orphan straight away, unless the heartbeat has moved on without that
record because a later session took the call over.

`lllm2 modal list` shows orphans, and a new session can adopt or cancel them.
A live call belongs to its own session: bulk stops skip it, no session may
adopt it, and only `lllm2 modal stop CALL_ID --force` takes it away.

See [ADR 3](decisions/0003-tunnel-transport-and-local-proxy.md) for why lllm2
uses a tunnel and a local proxy.
