# Find and download models

Open **Find models** to search public, ungated GGUF repositories on Hugging Face.
A blank search inspects up to 30 popular repositories; enter a model or publisher
to narrow the search. The pane reads metadata only until you queue a download.

Click a column heading to cycle through ascending, descending and no sort
(the original suggestion order). Results scroll within a short window, keeping
the horizontal scrollbar and catalogue close by. Titles and filters stay visible
while scrolling vertically. Use the filters below each
heading to match text or set numeric minimums and maximums. Text filters require
all space-separated terms and ignore case. Prefix a term with `!` to exclude it:
`!Qwen 29` excludes names containing “Qwen” and requires “29”. Double quotes
group a phrase: `!"Qwen 29"` excludes that phrase while allowing other Qwen models. These filters apply
to the fetched batch. **Clear column filters** resets them. The initial view
shows likely suitable chat, instruct and coding models with 4–8 bit quantisations;
uncheck those options to see other variants. Sort by **Model** to group variants
by repository. Popularity and task labels are discovery signals, not quality tests.

Suitability uses total download size, including a required vision projector:

- **Likely GPU fit:** weights fit the largest GPU's total VRAM minus 3 GiB.
- **Likely needs CPU offload:** weights fit system RAM minus 4 GiB, with an NVIDIA
  GPU detected. CPU offload can be slow.
- **Unknown / Too large:** metadata or hardware is missing, or the simple memory
  budget is exceeded.

GPU memory is not summed across cards. These estimates do not predict engine
architecture support, speed or usable context. Use **Experiments** to measure them.

Choose **Add to catalogue** on a specific variant, then **Queue download** in
**My catalogue**. New entries pin the HF repository revision. Split GGUFs are
queued as a complete set. A unique vision projector is included when present;
ambiguous projectors and incomplete shard sets cannot be added. Downloads run
one at a time, show progress, and can be cancelled and retried. Pending jobs
survive panel restarts; partial files are retained for resume. Once downloaded,
open **Launch model** or **Experiments** to use the model.

**Remove…** opens a dialog with the managed files that exist on this workstation.
By default it removes only the catalogue entry. Check **Also permanently delete…**
to delete those weights and partial downloads too. Other copies, experiment
results and settings are kept. Stop active downloads and model operations before
deleting their files. Files shared by another catalogue entry are protected.

The catalogue is stored in the workbench SQLite database and seeded once from
the packaged catalogue. Removing every entry leaves it empty across restarts and
upgrades. HF search metadata is cached for an hour; **Refresh HF metadata** bypasses
the cache. If HF is unavailable, cached search results remain available with a
notice, and the saved catalogue can still be viewed and edited offline.
