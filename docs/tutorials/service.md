# Run the panel as a service

After [installing lllm2 and trying a model](installation.md), you can run the
panel as a systemd user service so it does not need an open terminal. Use a
login session on the Linux machine where you installed lllm2. No sudo is needed.
Run this tutorial's commands in a terminal on that machine, outside the Pi
devcontainer. Service controls are terminal commands, rather than panel buttons.

## 1. Stop the foreground panel

Wait for experiments and downloads to finish. In the panel's **Launch model**
tab, click **Stop model** beneath the model summary,
then press Ctrl-C in the terminal running `lllm2`.

## 2. Install and start the service

Run this from the same environment you use to start lllm2, with any custom
model paths or GPU settings already set:

```bash
lllm2 service install
systemctl --user status lllm2-panel
```

Look for **active (running)** in the status output. Press **q** to return to
your shell if the output opens in a pager. If startup failed, use the log
command in step 3 to see why.

The installer enables and starts `lllm2-panel.service` for your user. Open
<http://127.0.0.1:8082>, choose your model and click **Start**. The service starts
the panel; you still start the model yourself.

On DLS machines, the `~/models` symlink to scratch from the installation
tutorial continues to work. If you use `LLLM2_MODELS_DIR` instead, set it before
installing the service so the installer saves it in the service environment.

You can now close the terminal. The service normally runs with your login
session. To start it at boot and keep it running after logout, run:

```bash
loginctl enable-linger "$USER"
```

Your system may require administrator permission for lingering.

## 3. Manage the service

To view its logs:

```bash
journalctl --user -u lllm2-panel -f
```

Press Ctrl-C to leave the log viewer; the service keeps running. To stop or
start the panel:

```bash
systemctl --user stop lllm2-panel
systemctl --user start lllm2-panel
```

To stop it and disable automatic startup:

```bash
systemctl --user disable --now lllm2-panel
```

Follow [Upgrade lllm2](upgrade.md) when updating the package. For custom ports,
environment changes, checkout installations and removal, see the
[service options](../how-to/serve-from-terminal.md#run-the-panel-as-a-user-service).
