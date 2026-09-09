# Packaged data

`lllm2/models.json` seeds the persistent SQLite catalogue once and supplies
inherited tuning for recognised checkpoints;
`lllm2/recommendations.json` records measured recommendations and their
provenance. The package also includes the panel's HTML, CSS and JavaScript
and model-specific chat templates. The editable catalogue, cached HF metadata, download queue, user settings and
experiment results live in the state directory. See [Find models](../how-to/find-models.md).
