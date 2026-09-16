# Serve a model from Modal

lllm2 can run a catalogue model on a [Modal](https://modal.com) GPU instead of
a local card. The remote server appears on the usual engine port,
`http://127.0.0.1:1920/v1`, so clients, harness wrappers and claude-sandbox
need no changes.

Modal bills for every second a GPU container runs. Read
[Estimate the cost](#estimate-the-cost) and
[Clean up](#clean-up) before your first launch.

## Install the Modal extra

The Modal client is an optional dependency:

```bash
uv tool install --upgrade 'lllm2[modal]'
```

With pip, use `pip install 'lllm2[modal]'`. In a checkout, use
`uv sync --extra modal`. Without the extra, a Modal launch stops with a message
that names it.

## Set up Modal

1. Create a Modal account and sign in from the terminal:

   ```bash
   modal token new
   ```

   The Modal client also reads `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

2. Add a payment method in the Modal dashboard. Modal refuses to run GPU
   functions for a workspace without one.

3. Check the credentials and deploy the lllm2 app:

   ```bash
   lllm2 modal setup
   ```

   This deploys an app named `lllm2` to your workspace. lllm2 deploys it again
   automatically when its version changes, so you run `setup` only once. The
   first image build downloads the lllm2 CUDA engine release and takes a few
   minutes.

4. Optionally check a GPU type:

   ```bash
   lllm2 modal probe --gpu L4
   ```

   A short, billed container prints the GPU name, its memory, the engine's
   device list, the engine build it ran and the engine sha256. The build line
   names three different versions: the llama.cpp release, the CUDA track and
   the compiler. lllm2 saves the result, and later launches and validation use
   it. Add `--json` to print the record as JSON.
   The container installs the same engine release as
   `lllm2 engines install cuda` for the same lllm2 version.

## Download a model

Launch downloads a missing model automatically, but you can store it ahead of
time:

```bash
lllm2 modal download qwen3-8b
```

Pass a catalogue id, such as `qwen3-8b`. The download runs inside Modal
and fetches the model and its companion files from Hugging Face into the
`lllm2-models` Volume, so no weights pass through your connection. The command
prints progress and downloads nothing if the model is already stored. Press
Ctrl-C to cancel; the partial download stays in the Volume and a rerun resumes
it.

## Launch a model

```bash
lllm2 launch --backend modal --gpu L40S --model qwen3.8-27b
```

`--model` accepts a catalogue id or name. Without `--model`, lllm2 serves the
top-ranked catalogue recommendation. Only catalogue models can be downloaded;
lllm2 does not upload local files. `--engine` and `--device` do not apply,
because every Modal launch runs the lllm2 CUDA engine release for the installed
lllm2 version.

See [Modal GPU types](../reference/modal-gpus.md) for the `--gpu` values and
their memory. lllm2 sizes context and slots for the chosen GPU, as it does for
a local card. Saved workbench settings for the same model, backend and GPU type
take precedence.

lllm2 ships no measured recommendation profiles for Modal GPUs yet, so the
starting settings are inherited estimates. Measured profiles arrive after a
later paid measurement run. Until then, run an experiment on your GPU type and
save the settings you choose.

The launch goes through these steps:

1. **Probe.** On the first launch with a GPU type, a short container reports the
   GPU's real memory and the engine's capabilities. lllm2 saves the result in
   `remote-probes.json` in the state directory and does not probe again until
   lllm2 or its engine release changes.
2. **Download.** If the model is not in the `lllm2-models` Volume, Modal
   downloads it from Hugging Face into the Volume. The download runs in Modal,
   not over your connection, and later launches reuse the file.
3. **Cold start.** A GPU container starts, runs llama-server and opens an
   encrypted Modal tunnel to it. Loading a large model from the Volume takes
   tens of seconds to a few minutes. `--timeout` (default 180 seconds) bounds
   this step; raise it for large models.
4. **Serve.** lllm2 listens on `127.0.0.1:1920` and forwards each request to the
   tunnel. It prints the API URL, the Modal call ID, the estimated hourly cost
   and the idle timeout.

Press Ctrl-C to stop the container. Billing stops when Modal ends the call.

## In the panel

Start the panel with `lllm2` as usual. You can switch between a local and a
Modal backend without restarting it.

1. Open **Launch → Customize settings** and choose the Modal backend in
   **Backend**. The engine and device fields are hidden, because every Modal
   launch uses the lllm2 CUDA engine release.
2. Choose a **Remote GPU type**. Each option shows the GPU's memory and its
   estimated hourly price. Starting settings are sized for the chosen type.
3. Optionally set **Idle stop (minutes)**. Blank or 0 disables the idle stop.
4. Choose a model. With the Modal backend, the model list shows catalogue
   entries with their presence in the Volume: **In Modal storage** starts
   without a download, and **Not in Modal storage** downloads on the first
   start. **Download to Modal** fetches a model ahead of time, and
   **Remove from Modal…** deletes it. Download progress also appears under
   **Find models → Downloads**. No weights pass through your workstation.
5. Start the model. The status line names the cold-start phase: checking the
   GPU type, downloading the model into remote storage, starting the GPU
   container, loading the model into GPU memory, then ready.

While the container runs, the launch status shows the elapsed time, the
estimated cost so far at the hourly price, and the time left until the idle
stop, with the pricing caveat underneath. To change the idle stop for the
running model, edit **Idle stop after (minutes)** and click
**Apply to running model**; blank disables it. When the idle stop fires, the
status says that the model stopped after that many minutes without requests
and no longer bills.

**Engine logs & exact command** shows the llama-server log lines that the
container sends back. **Stop model** cancels the call. Closing the panel with
Ctrl-C, or a stopped service, cancels every call the panel owns.

The **Modal storage & running calls** card, near the bottom of Launch, lists
the stored models with their sizes and the running lllm2 calls with elapsed
time and estimated cost. **Remove…** is disabled for a model that a running
call serves. **Refresh storage and calls** reads the account again.

The panel checks for calls from earlier sessions when it loads, after
**Refresh storage and calls**, and after you adopt or stop a call. It asks a
provider only when that provider is the selected backend, the panel already
runs its engine, or the call records in the state directory name it. The banner
lists only calls with no live owner: a call another session still heartbeats to
never appears there, however this workstation's records look. If a call from an
earlier session is still running, a banner at the top of the panel shows its
GPU type, model, elapsed time and estimated cost:

- **Adopt and serve** takes over the call and serves it on the usual engine
  port. If a model is already running, the button reads
  **Stop current model and adopt**. Adopting needs the call's saved key, so
  only calls started from this workstation can be adopted.
- **Stop call** cancels it.

## Connect claude-sandbox

A Modal model serves on the same local port as a local one, so
[Connect a client](connect-a-client.md) applies unchanged. Keep lllm2 running
on the host while the model serves, and start the client as usual, for example:

```bash
claude-container --host-net --agent pi
```

The client never sees the tunnel or the API key. If the idle stop ends the
call, the client's next request fails until you start the model again.

## Estimate the cost

lllm2 estimates cost from a bundled table of Modal's per-GPU hourly prices. The
estimate excludes CPU, memory and Volume storage charges, and Modal can change
its prices, so check [current Modal pricing](https://modal.com/pricing).

Some examples from the bundled table:

| Session | Estimate |
| --- | --- |
| One hour on a T4 | $0.59 |
| One hour on an L40S | $1.95 |
| An L40S left running until the default 30-minute idle timeout | $0.98 after the last request |

The probe and download steps also run containers, but for much less time than a
serving session.

## Set the idle timeout

A running container stops after 30 minutes without requests. The timer does
not run while a request is in progress, and it restarts when a request passes
through the local proxy.

```bash
lllm2 launch --backend modal --gpu T4 --model qwen3-8b --idle-timeout 10
```

`--idle-timeout` accepts whole minutes from 0 to 1440, or `off`. `0` and `off`
disable the timeout. `lllm2 launch` takes the first of these that is set:

1. `--idle-timeout`.
2. `LLLM2_IDLE_TIMEOUT_MINUTES`.
3. The saved launch settings' `idle_timeout_minutes`.
4. 30 minutes.

The panel fills its **Idle stop (minutes)** field from
`LLLM2_IDLE_TIMEOUT_MINUTES`, or 30, and sends the field with each start. lllm2
reads the variable once at startup, so set it before starting lllm2:

```bash
LLLM2_IDLE_TIMEOUT_MINUTES=off lllm2
```

With the timeout disabled, the GPU bills until you stop the launch. A serve
call ends after 12 hours regardless.

Disable the timeout only for unattended runs that leave long gaps between
requests.

## Understand the safety net

While the lllm2 process that starts a container drives it, it sends the
container heartbeats, at most every 10 seconds. This applies to serve and
download containers. About 3 minutes after the owning process goes silent, the
container stops its work and ends the call. A crash, lost network or
powered-off machine therefore stops billing within a few minutes. The same
rule applies when the owner exits cleanly without cancelling. As a hard limit,
Modal ends a serve call after 12 hours and a download call after 2 hours.

Those heartbeats also say who owns a call. A container belongs to the session
that keeps heartbeating to it, on whichever machine or container that session
runs. Another lllm2 session sees the call as in use until the heartbeat goes
silent, and only then as an orphan it may adopt or stop.

## Clean up

List running lllm2 serve calls with their owner, elapsed time and estimated
cost:

```bash
lllm2 modal list
```

Each call is **owned by this process**, **in use** by another live lllm2
session, or an **orphan** that no live session owns. A call counts as in use
while its owner still heartbeats, even when this workstation has no record of
it, so a CLI in a container never mistakes the panel's call for an orphan.
Stop calls:

```bash
lllm2 modal stop CALL_ID
lllm2 modal stop --all
```

`--all` stops every orphan and never touches a call in use. `stop CALL_ID`
skips a call in use and tells you where it runs; adding `--force` stops it
anyway, which leaves that session without its model, so stop it from that
session where you can. `--force` works on one call id, not with `--all`.

Stored models incur Modal storage charges. List and remove them:

```bash
lllm2 modal models
lllm2 modal remove qwen3-8b
```

Pass a catalogue id, or a name exactly as `models` prints it. `remove` deletes
the model with its companion files, and refuses while a running call serves
that model.

To remove everything lllm2 created, stop all calls first, then delete the `lllm2` app, the `lllm2-models`
Volume, the `lllm2-state` Dict and the `lllm2-logs` Queue from the Modal
dashboard or with the `modal` CLI.

See [Remote engines](../explanations/remote-engines.md) for how the proxy,
tunnel and API key fit together.
