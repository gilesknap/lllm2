# Compare and save settings

In **Experiments**, choose workloads and run a baseline. Change settings and
add combinations, or run **Baseline + single options**. **Discover usable
context** runs first: quick load-only checks find the largest window the engine
accepts, then one long prompt confirms it. Context discovery is the only option
enabled by default; select workloads to add cold speed samples after it, and
turn on Quick sweep or Full launch window to vary their prompt sizes.

Inspect completed results and their evidence before loading a configuration
into **Launch**. Click **Save my settings** to keep it. Launch and experiment
drafts are separate; loading a result does not save it or restart the model.
Export results as CSV for comparison or JSON for the full record.

See [Experiment and model settings](../explanations/experiment-settings.md) for
each workload, budget and customization control, including cold versus warm
measurements and how to interpret the results.

**Try in Launch** copies the experiment settings into Launch. Expand a row with
the **▸** control to reach it, together with **Copy row** and **Delete run…**.
The **Context** choice defaults to **Tested**, which applies the largest
successful context, keeping the slot count. **90%** selects the smaller,
rounded-down estimate instead; **Original** keeps the experiment's own context.
The choice appears only when a successful context measurement exists.
Loading does not
start the model or save preferences; review Launch, then save or restart explicitly.

## Manage experiment history

**Experiment history** contains individual runs, with one row per sample.
Expand a row and choose **Delete run…** to remove that entire run, including all its
samples, logs and context probes. **Delete failed / cancelled…** selects every
run whose overall status is failed or cancelled. It keeps completed runs even
when an individual sample or quality check failed, and keeps interrupted runs
unless you delete them individually.

Both actions show a confirmation with the number of runs and samples. Deletion
is permanent; export full JSON first if you want to keep the records. Saved
settings, their saved provenance, model files and the running model are kept.
Deletion is unavailable while an operation is active.
