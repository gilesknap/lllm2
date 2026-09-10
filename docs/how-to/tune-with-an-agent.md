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
panel, even though the panel listens on localhost. The jail relays chosen
loopback ports into the agent; by default only the model API on 1920. Add the
panel port for this session. In the VS Code terminal:

```bash
cd /workspaces/lllm2-tuning
CLAUDE_SANDBOX_LOCAL_PORTS=8082 claude
```

Use `codex` in place of `claude` for Codex. The jail stays on: the agent sees
the panel and the model API and nothing else on your machine's network, and it
still has no access to your host credentials, home directory or shell
environment. A plain `claude` launch relays only the model port again. This
needs claude-sandbox with the port relay available to every agent (see its
[egress jail how-to](https://diamondlightsource.github.io/claude-sandbox/how-to/network-egress-jail.html#reach-services-on-the-hosts-loopback)).

Log in to the agent when prompted, then confirm it can see the panel by asking
it to run:

```bash
curl -fsS http://127.0.0.1:8082/api/status | head -c 300
```

A JSON document with `"version"` and `"engine"` keys means the API is
reachable. `Connection refused` means the panel is not running, the agent was
started without the variable, or the installed claude-sandbox predates the
relay for Claude and Codex.

## 3. Give the agent the tuning prompt

Paste the prompt below into the agent. Replace the first line with the model
you want to tune. The agent works through the API, adapts later rounds to the
measurements, and leaves saved settings unchanged.

````text
Tune the lllm2 settings for the model whose filename contains "Qwen3.8-27B".

Drive the local panel API at exactly http://127.0.0.1:8082; there is no browser.
Read GET /api/status first and send its "token" as X-LLLM2-Token on every POST.
Use GET /api/status, /api/results and /api/results/export, and POST
/api/discover, /api/default/resolve, /api/capabilities, /api/benchmark and
/api/cancel only. Do not download or delete anything, and do not save settings.

Find the matching installed checkpoint and CUDA engine, resolve its saved or
recommended settings, and use that complete settings object as the baseline.
Check capabilities and test only supported settings. Experiments may stop the
served model. If one is running, include "replace_running": true and
"expected_pid": <engine.pid>. Run one operation at a time, poll status about
every 30 seconds, and do not cancel a healthy run.

Find useful settings with small adaptive rounds rather than one Cartesian
search:

1. Establish a repeated 1K baseline, then screen speculation, common K/V cache
   precision, reasoning effort, Flash Attention and other available single
   options. Combine only promising changes.
2. If MTP helps, tune draft length. Test CUDA execution controls separately.
   Compare batch and microbatch settings at 16K and 64K before accepting a
   prefill improvement.
3. Compare matching prompts and output budgets. Use medians from at least three
   repeats for short screens. Judge wall time alongside prefill/decode rates,
   actual output count, draft acceptance and peak GPU memory.
4. Quality-check finalists with source-small-edit or context-retrieval-edit.
   Exact adherence must pass; a shorter or truncated answer is not a speed win.
5. Find the largest one-slot context up to the checkpoint metadata limit. A
   load-only search is preliminary: confirm the candidate with a nearly full
   context-retrieval-edit prompt. Use q4 cache or headroom if needed. Raise the
   speed-test timeout to 1800 seconds for prompts near 256K; keep load-probe
   timeout at 900 seconds.
6. Test whether two and four slots load and serve. Report total context and
   context per slot. The current harness sends requests sequentially, so do not
   claim concurrent throughput unless you measure simultaneous requests by
   another method.
7. Re-run the original baseline and finalists at 1K, 16K and 64K. Give separate
   recommendations when short decode speed, long-context work, capacity and
   multiple clients have different winners.

Keep the exact checkpoint, engine, backend and device fixed within a comparison.
The combinations endpoint also requires context and slots to match. Preserve
all results and failures. Report result IDs, settings changed, medians and
percentage differences, quality outcomes, memory headroom, timeouts and anything
the harness could not establish. Leave the queue idle and tell me which result
could be saved with POST /api/default/save if I later authorize it.
````

## Example: maximize Qwen3.8-27B context on an RTX 3090

On 10 September 2026 this process tested
`Qwen3.8-27B-UD-Q4_K_S.gguf` with llama.cpp commit `662a0b0` on an NVIDIA
GeForce RTX 3090 (24,576 MiB VRAM, compute capability 8.6, driver 595.91.07)
with a Ryzen 7 5800X and 30.3 GiB system RAM. The chosen one-slot coding
profile was:

| Setting | Value |
|---|---|
| Total context / slots | 262,144 / 1 |
| GPU layers / Flash Attention | 999 / on |
| Common K/V cache | q4_0 |
| Reasoning effort | default |
| Speculation / draft length / draft cache | MTP / 3 / q8_0 |
| Logical batch / physical microbatch | 2048 / 512 |
| Target GPU sampling / concurrent CUDA streams | off / off |

A 260,064-token retrieval/edit prompt reserved 2,048 reply tokens plus a
32-token margin. It passed exact adherence, processed the prompt at 443.5 tok/s,
decoded at 29.8 tok/s and finished in 601 seconds. Sampled total GPU use peaked
at 24,093 MiB, so startup was sensitive to other GPU applications; 235,776 is
the 10%-headroom alternative.

For repeated 1K/256-token long-code requests, q4/default improved median decode
from 71.34 to 72.99 tok/s (2.3%), reduced wall time from 4.54 to 4.48 seconds
(1.4%), and saved about 1 GiB of sampled GPU memory. A separate q8/low-effort
profile reached 78.76 tok/s (10.4% faster decode and 7.8% lower wall time) and
passed the exact small-edit check, but regressed at 16K and 64K. It was therefore
a short-request option, not the saved long-context default.

Two slots at 131,072 total context also loaded and served, providing 65,536
tokens per slot at a sampled 22,876 MiB. Four slots at 65,536 total provided
16,384 per slot and also served, but simultaneous-request throughput was not
measured.

## 4. Keep the result

When the agent reports, open **Experiments** in the panel, expand the finalist
in **Experiment history**, click **Try in Launch**, review the draft and click
**Save my settings**. See [Compare and save settings](compare-settings.md)
for the context choices on that row and
[Experiment and model settings](../explanations/experiment-settings.md) for
what each measured number means.
