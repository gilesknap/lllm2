# Let a coding agent find your best settings

A sandboxed coding agent can drive **Experiments** through the panel's API,
compare the results and tell you which configuration to keep. This guide sets
up Claude Code or Codex in a [claude-sandbox](https://github.com/DiamondLightSource/claude-sandbox)
container that can reach the panel, then gives you a prompt to paste in.

Each experiment stops the model that is currently serving, so run this when
nobody needs the model. Expect an hour or more of GPU time for a 27B model.

## 1. Install lllm2 and start a model

If you do not have lllm2 yet, follow the
[installation tutorial](../tutorials/installation.md). Before you continue,
the panel should be open at <http://127.0.0.1:8082> and the model you want to
tune should start from **Launch** at least once. The agent needs the panel
running; it does not matter whether the model is running.

## 2. Get a sandboxed agent that can reach the panel

Install rootless Podman and VS Code with the Dev Containers extension, then
clone claude-sandbox next to an empty directory for the agent's notes and open
the clone in VS Code:

```bash
mkdir -p ~/src/lllm2-tuning
git clone https://github.com/DiamondLightSource/claude-sandbox ~/src/claude-sandbox
code ~/src/claude-sandbox
```

Choose **Reopen in Container** when prompted. The container build installs
the sandboxed `claude` and `codex` commands and mounts the parent directory
at `/workspaces`, so the notes directory is reachable inside. The
[getting-started tutorial](https://diamondlightsource.github.io/claude-sandbox/tutorials/getting-started.html)
covers the Podman setup for VS Code and the first login.

The sandbox's network egress jail gives the agent a private network namespace
with its own loopback, so `127.0.0.1:8082` inside the jail is **not** your
panel, even though the panel listens on localhost. Its single-port relay only
serves Pi and only forwards the model port. The devcontainer itself runs on
the host network, so turning the jail off for this one session is enough.
In the VS Code terminal:

```bash
cd /workspaces/lllm2-tuning
CLAUDE_SANDBOX_EGRESS_JAIL=0 claude
```

Use `codex` in place of `claude` for Codex. With the jail off the agent
shares the machine's network but still has no access to your host
credentials, home directory or shell environment. Do not set the variable
for ordinary coding sessions; a plain `claude` launch is jailed again.

Log in to the agent when prompted, then confirm it can see the panel by asking
it to run:

```bash
curl -fsS http://127.0.0.1:8082/api/status | head -c 300
```

A JSON document with `"version"` and `"engine"` keys means the API is
reachable. `Connection refused` means the panel is not running, or the agent
was started without the variable.

## 3. Give the agent the tuning prompt

Paste the prompt below into the agent. Replace the first line with the model
you want to tune. The agent works entirely through the API, waits for each run
and reports back before anything is saved.

````text
Tune the lllm2 settings for the model whose filename contains "Qwen3.8-27B".

lllm2 is a local llama.cpp workbench. Its panel runs at http://127.0.0.1:8082
and exposes a JSON API under /api/. Always use exactly that base URL, because
the server checks the Host header. Use curl or python3 urllib; there is no
browser here.

Rules
- Read GET /api/status first. Send the value of its "token" field as the
  X-LLLM2-Token header on every POST, with Content-Type: application/json.
- Only use these endpoints: GET /api/status, GET /api/results,
  GET /api/results/export, POST /api/discover, POST /api/default/resolve,
  POST /api/capabilities, POST /api/benchmark, POST /api/cancel. Do not delete
  results, download models, or save settings unless I say so in this session.
- An experiment stops any running model; that is expected. Before submitting,
  read the "engine" object from /api/status. If "running" is true, include
  "replace_running": true and "expected_pid": <engine.pid> in the benchmark
  request.
- Only one operation runs at a time. After submitting, poll GET /api/status
  every 30 seconds and wait until "job"."active" is false. Never submit while
  a job is active, and never cancel a run unless I ask you to.
- Do not change model, engine, backend, device, context or slots between
  configurations in one comparison; the server rejects mixed combinations
  and the comparison would be meaningless anyway.

Step 1: find the model and starting settings
- POST /api/discover with {} lists installed models and engines. Pick the
  model whose path contains the name above and the newest CUDA engine.
- POST /api/default/resolve with {"settings": {"model": "<path>", "engine":
  "<path>"}} returns the recommended or saved settings for it. Use the
  returned "settings" object as the baseline. It has these keys: model,
  engine, backend, device, context, slots, gpu_layers, flash, cache, cache_k,
  cache_v, speculation, drafter, pair_confirmed, draft_length, effort,
  draft_cache, chat_template, batch_size, ubatch_size, backend_sampling,
  cuda_graph_opt, cache_ram_mib, context_checkpoints, lookup_ngram_n,
  lookup_ngram_m. Send the whole object back each time; unknown keys are
  rejected.
- POST /api/capabilities with {"settings": <baseline>} reports which
  features the engine and checkpoint support (speculation modes, cache
  precisions, flash attention, reasoning effort). Only test what it reports
  as available.
- Report the baseline and the available features to me before running
  anything.

Step 2: run the built-in single-option sweep
POST /api/benchmark with:
{"mode": "suite", "settings": <baseline>, "workloads": ["long-code"],
 "sweep_prompts": true, "output_tokens": 256, "repeats": 2,
 "search_context": false, "full_window": false, "timeout": 900,
 "context_timeout": 900}
plus replace_running/expected_pid if needed. This runs the baseline and one
variant per option group (speculation mode, KV cache precision, flash
attention, reasoning effort). The response lists skipped variants with
reasons. Wait for it to finish.

Step 3: read the results
GET /api/results returns one row per run, newest first. Rows from the same
submission share a "group" value; each row has "label" ("baseline",
"speculation-none", "cache-q4_0" and so on), "status" and "settings". Each
row's "samples" list has one entry per workload, prompt size and repeat with
"input_tokens", "output_tokens", "prefill_tok_s", "decode_tok_s",
"wall_seconds", "peak_total_gpu_used_mib", "peak_engine_rss_mib" and
"adherence". Compare rows only at the same input size. Average the repeats.
Print a table: label, and for each prompt size the mean decode tok/s,
prefill tok/s and peak GPU MiB. Note any failed or skipped variants and
their reasons.

Step 4: combine the winners
Build up to 6 configurations from the baseline that combine the options that
helped, plus one or two follow-ups worth checking (for example a different
draft_length if speculation helped, or batch_size 4096 with ubatch_size 1024
if prefill matters). Show me the list and wait for my go-ahead. Then POST
/api/benchmark with {"mode": "combinations", "settings": <baseline>,
"combinations": [<settings>, ...]} and the same workload and budget fields as
step 2. Rows come back labelled "combination-1", "combination-2" and so on in
the order sent. Wait, then extend the table.

Step 5: confirm the finalist on more workloads
Take the best configuration by decode tok/s at 16K input that is within 3%
of the best prefill, and rerun it alone with {"mode": "custom", "settings":
<finalist>, "workloads": ["long-code", "source-small-edit",
"context-retrieval-edit"], "sweep_prompts": true, "repeats": 2}. The two
source workloads report "adherence"; treat a failed adherence as
disqualifying and fall back to the next candidate. Optionally, with my
go-ahead, run the finalist once more with "search_context": true,
"workloads": [] and "max_context": 131072 to measure the largest context that
loads.

Step 6: report
Give me a final table with baseline, finalist and the difference in decode
and prefill tok/s at each prompt size, peak GPU memory, and the list of
settings that changed. State what you could not measure. Do not save
anything. I will open Experiment history in the panel, expand the finalist
row and click "Try in Launch" then "Save my settings" myself, or I will ask
you to POST /api/default/save with {"result_id": "<row id>"}.
````

## 4. Keep the result

When the agent reports, open **Experiments** in the panel, expand the finalist
in **Experiment history**, click **Try in Launch**, review the draft and click
**Save my settings**. See [Compare and save settings](compare-settings.md)
for the context choices on that row and
[Experiment and model settings](../explanations/experiment-settings.md) for
what each measured number means.
