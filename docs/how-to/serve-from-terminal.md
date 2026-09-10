# Serve from the terminal

```bash
lllm2 models
lllm2 engines list
lllm2 launch --model /data/models/model.gguf --engine /path/to/llama-server
```

With no selection options, `lllm2 launch` looks for a compatible installed
recommendation. After selecting the model, backend, engine and device, it loads
the saved workbench settings for that model and backend when they exist. The
currently discovered engine and device take precedence over their saved paths.
An empty or separate `LLLM2_STATE_DIR` has no saved settings, so launch falls
back to the built-in measured or inherited starting settings. It runs in the
foreground; Ctrl-C stops the engine.

## Run the panel as a user service

On Linux with systemd, stop any foreground panel and run:

```bash
lllm2 service install
```

This installs `lllm2-panel.service` under
`${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user`, enables it for your user,
and starts it at `http://127.0.0.1:8082`. No sudo is used. It starts the panel;
choose and start a model in the panel as usual.

Use `--host` and `--port` to change the listener, or `--no-start` to install
and enable without starting or restarting it. Reinstalling updates the managed
unit and restarts it by default. Existing services not created by this command
are left alone.

The service uses the Python installation running the installer. Keep that
installation in place. It saves the current `LLLM2_*` path and port settings,
`PATH`, `LD_LIBRARY_PATH`, and GPU visibility settings. Run the installer again
after changing those settings or moving the installation. For a checkout, use
`uv run python -m lllm2 service install` from a persistent, executable filesystem.

```bash
systemctl --user status lllm2-panel
journalctl --user -u lllm2-panel -f
systemctl --user restart lllm2-panel
systemctl --user disable --now lllm2-panel
```

To remove the disabled service, delete its unit file and run
`systemctl --user daemon-reload`. User services normally run with your login
session. To run at boot and remain running after logout, enable lingering with
`loginctl enable-linger "$USER"` (your system may require administrator permission).

Check the installed version with `lllm2 --version`. The panel header shows the
version of its running server; restart the service after upgrading the package.
