# Run a model on a Modal GPU

The short path to a rented GPU, driven entirely from the panel. Modal bills for
every second a GPU container runs, so read step 5 before you start.
[Serve a model from Modal](../how-to/serve-from-modal.md) is the long version,
with cost, cleanup and the command line.

## 1. Install the extra and sign in

In a terminal on the machine that runs the panel:

```bash
uv tool install --upgrade 'lllm2[modal]'
uv tool install modal
modal token new && lllm2 modal setup
```

The second line installs the Modal command line. The extra puts the Modal
client inside lllm2, but it does not put the `modal` command on your path.

`modal token new` opens the Modal sign-in; the workspace needs a payment method
before it will run GPU functions. `lllm2 modal setup` deploys the lllm2 app once.
See [Set up Modal](../how-to/serve-from-modal.md#set-up-modal) if either step
complains. Then start the panel with `lllm2` and open
<http://127.0.0.1:8082>.

## 2. Choose the Modal backend

In **Launch model**, open **Customize settings** and set **Backend** to
**Modal**. The engine and device fields disappear, because every Modal launch
runs the lllm2 CUDA engine. Pick a **Remote GPU type**: each option shows its
memory and hourly price, and the starting settings are sized for it. See
[Modal GPU types](../reference/modal-gpus.md).

## 3. Find a model the rented GPU can hold

A new catalogue holds little, so open **Find models**. lllm2 rates every result
against the GPU you selected, not against your own card. A rented 96 GB card
therefore rates models as a good fit that no local card of yours could load.
Change the GPU type and the list rates them again.

Pick one and add it to your catalogue, then return to **Launch model**.

## 4. Download a model into Modal storage

**Choose a model** now lists catalogue models with their presence in Modal
storage. Click **Use this model** on one, then **Download to Modal**. Modal
fetches the weights from Hugging Face straight into its Volume, so nothing
passes through your machine. Wait for the line to read **In Modal storage ·
starts without a download**. A large model takes several minutes, and progress
appears in the **Downloads** card on this page.

## 5. Start the model

Click **Start Model**. The status line names each phase: checking the GPU type
(a first launch on a GPU type runs a short probe container), downloading the
model into remote storage, starting the GPU container, loading the model into
GPU memory, then ready. In our test a 29 GB model took about 169 seconds from
container start to ready on a T4, on top of the one-off download.

## 6. Connect a coding agent

When the model is ready, **Connect a coding agent** appears with the command to
run. Copy it and run it in a terminal on the same machine, in your project
directory:

```bash
uv tool install claude-sandbox && claude-sandbox pi
```

**Claude Code** and **Codex** give you their commands instead. The agent talks
to `http://127.0.0.1:1920/v1` as it would for a local model and never sees the
tunnel.

## 7. Stop paying

The launch status shows the elapsed time, the cost so far at the hourly price
and the time left until the idle stop. Watch that line.

- **Stop model** ends the container, and billing, immediately.
- Left alone, the model stops after 30 idle minutes. Change that in **Idle stop
  after (minutes)** and click **Apply to running model**; blank means it runs,
  and bills, until you stop it.
- Closing the panel with Ctrl-C stops every call it owns.

Stored models still cost a little while they sit in the Volume. Use
**Remove from Modal…** in the model list, and see
[Clean up](../how-to/serve-from-modal.md#clean-up) for the account-wide sweep.
