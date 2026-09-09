# Upgrade lllm2

Wait for experiments and downloads to finish, then stop the model in the panel.
On DLS machines, load uv in your shell with `module load uv`.
Run the commands below in a terminal on the machine hosting lllm2, outside
the Pi devcontainer. The model's **Stop model** button is in the panel's
**Launch model** tab, beneath the model summary.

## If you installed the service

Stop the panel service, upgrade the package and engine, then start the service:

```bash
systemctl --user stop lllm2-panel
uv tool upgrade lllm2
lllm2 engines install cuda
systemctl --user start lllm2-panel
systemctl --user status lllm2-panel
```

Check for **active (running)**; press **q** if the status display opens in a pager.

If you moved the Python installation or changed its environment settings,
rerun `lllm2 service install` before starting the service, preserving any custom
`--host` and `--port` options. The installer updates the unit and starts it.
See [Run the panel as a service](service.md).

## If you run the panel in a terminal

Stop the panel with Ctrl-C in its terminal, then run:

```bash
uv tool upgrade lllm2
lllm2 engines install cuda
lllm2
```

Open <http://127.0.0.1:8082> (or your configured address), check the version in
the panel header, and start your model again. Models, saved settings
and existing engines are preserved. Engine installation is a no-op when the
required version is already installed.

See [Install a model engine](../how-to/install-an-engine.md) for driver
compatibility, engine discovery and installation options.
