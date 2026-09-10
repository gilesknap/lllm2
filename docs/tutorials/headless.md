# Run a model without GNOME

Running without the graphical desktop leaves more GPU memory available for the
model. Save your desktop work before continuing: switching targets immediately
ends the GNOME session.

## Switch to terminal-only mode

To switch only for the current boot:

```bash
sudo systemctl isolate multi-user.target
```

To make terminal-only mode the default for future boots as well:

```bash
sudo systemctl set-default multi-user.target
```

Log in on the text console, or connect over SSH. `lllm2` detects that
`graphical.target` is inactive and does not reserve model memory for the
desktop.

## Launch the model

List the available models if needed, then launch:

```bash
lllm2 models
lllm2 launch
```

The launch command selects a compatible installed model and engine. If the
workbench database in `LLLM2_STATE_DIR` contains saved settings for that model
and backend, it uses them; a missing or empty state directory falls back to the
built-in settings. The API is served at <http://127.0.0.1:1920/v1> until you
press Ctrl-C.

To choose a particular installed model:

```bash
lllm2 launch --model /path/to/model.gguf
```

## Return to GNOME

Stop the model with Ctrl-C before restoring the desktop, because GNOME needs
some of the GPU memory used by a headless launch. Then run:

```bash
sudo systemctl set-default graphical.target
sudo systemctl isolate graphical.target
```

If you changed only the current session and left the default target graphical,
the `set-default` command is unnecessary.
