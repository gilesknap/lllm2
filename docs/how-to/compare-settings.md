# Compare and save settings

In **Experiments**, choose workloads and run a baseline. Change settings and
add combinations, or run **Baseline + single options**. **Discover usable
context** adds context probes; long prompts can take several minutes. Context
discovery is enabled by default, while Quick sweep and Full launch window are
off. The Long code workload remains selected for one short speed sample before
the context probes.

Inspect completed results and their evidence before loading a configuration
into **Launch**. Click **Save my settings** to keep it. Launch and experiment
drafts are separate; loading a result does not save it or restart the model.
Export results as CSV for comparison or JSON for the full record.

See [Experiment and model settings](../explanations/experiment-settings.md) for
each workload, budget and customization control, including cold versus warm
measurements and how to interpret the results.

**Try in Launch** copies the experiment settings into Launch. **Use tested
context** is checked by default and applies the largest successful context,
keeping the slot count. **Use 90% of tested context** selects the smaller,
rounded-down estimate instead. Selecting either option clears the other;
leave both unchecked to keep the experiment's original context. These options
appear only when a successful context measurement exists. Loading does not
start the model or save preferences; review Launch, then save or restart explicitly.

## Manage experiment history

**Experiment history** contains individual runs, with one row per sample.
Choose **Delete run…** on any row to remove that entire run, including all its
samples, logs and context probes. **Delete failed / cancelled…** selects every
run whose overall status is failed or cancelled. It keeps completed runs even
when an individual sample or quality check failed, and keeps interrupted runs
unless you delete them individually.

Both actions show a confirmation with the number of runs and samples. Deletion
is permanent; export full JSON first if you want to keep the records. Saved
settings, their saved provenance, model files and the running model are kept.
Deletion is unavailable while an operation is active.
