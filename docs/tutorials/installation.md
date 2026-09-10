# From a model to Pi

You need a Linux NVIDIA machine and [uv](../how-to/prepare-your-machine.md).
Run the shell commands below in a terminal on that machine, outside the Pi
devcontainer. Use the browser for the panel steps.

````{admonition} For DLS users
Load uv through the DLS module system in your shell:

```bash
module load uv
```

Store downloaded models on scratch to keep large model files out of your home
directory. If `~/models` does not already exist, replace `<fedid>` with your FedID
and run:

```bash
mkdir -p /scratch/<fedid>/models
ln -s /scratch/<fedid>/models ~/models
```
````

Install and open the panel:

```bash
uv tool install --upgrade lllm2
lllm2 engines install cuda
lllm2
```

Open <http://127.0.0.1:8082>. Keep the panel running.

## 1. Download a model

Click **Find models** in the navigation bar near the top of the panel.
Scroll to **My catalogue** and click **Queue download** beside a model.
To find another model, enter its name in **Search Hugging Face**, click
**Find models**, then **Add to catalogue** on the variant you want.

```{figure} ../images/tutorial-find-models.png
:alt: My catalogue card in Find models, with Queue download beside a saved model.
:width: 760px

Queue a download from My catalogue.
```

Once the download finishes, click **Launch model** in the top navigation and
select the model under **Choose a model** (**Find more models →** takes you back
to the catalogue). Already downloaded one? Start here.

## 2. Find its context window

With your model selected, click **Experiments** in the top navigation.
Check the model name in **Experiment configuration**; **Use launch settings**
copies your current Launch settings if needed. Scroll down to the **Experiments**
card. Leave **Discover usable context** checked and click **Run baseline**
below the workload options. Progress appears in **Queue & engine** below it.
Wait for the run to finish; each load check takes a few seconds.

```{figure} ../images/tutorial-experiments.png
:alt: Experiments card with Discover usable context checked and the Run baseline button below the workload choices.
:width: 760px

Keep the defaults for your first baseline.
```

Scroll further down to **Experiment history**. Click the **▸** at the start of
the completed run's row to expand it. Leave **Tested** selected under **Context**
and click **Try in Launch** to use the full successful context. **90%** leaves
some headroom; **Original** keeps the experiment's original context.

```{figure} ../images/tutorial-history.png
:alt: Completed baseline row expanded, showing Try in Launch and the three context choices.
:width: 760px

Try in Launch brings the selected settings back to the Launch view.
```

In **Launch model**, find **Save my settings** beside **Load settings**, below
the model controls. Save, then click **Start Model** above that settings toolbar.
Wait until the panel reports the model is ready.

## 3. Run Pi

In another terminal on the same machine, clone the sandbox and open it in VS Code:

```bash
git clone git@github.com:DiamondLightSource/claude-sandbox.git
code claude-sandbox
```

In VS Code, press **Ctrl+Shift+P** (or choose **View → Command Palette…**),
type **Dev Containers: Reopen in Container**, and select that command.
Once the container opens, choose **Terminal → New Terminal**. This terminal
runs inside the devcontainer; run Pi here:

```bash
pi
```

Pi connects to the lllm2 model port by default. Start coding!

For later: [run the panel as a service](service.md), [upgrade lllm2](upgrade.md),
[compare settings](../how-to/compare-settings.md),
or [connect another client](../how-to/connect-a-client.md).
Use **Stop** in the panel when you want to release the GPU.
