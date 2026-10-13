# Adopt the DLS Python Copier template

## Status

Accepted, 7 September 2026.

## Context

lllm2 needs a consistent development workflow and documentation structure.
The [DLS Python Copier template](https://github.com/DiamondLightSource/python-copier-template)
provides both, along with a way to merge future template updates.

## Decision

Adopt release `5.4.0`, recording the rendered answers in `.copier-answers.yml`.
Use its `src/` layout, setuptools and setuptools-scm packaging, pytest, tox,
Ruff, pre-commit, mypy, editor settings and development container.

Organize documentation into tutorials, how-to guides, explanations and
reference, with the template's Sphinx Design cards and PyData theme.
Keep the authored workbench documentation and DLS setup instructions.

Retain these project adaptations when updating the template:

- Keep the Typer CLI entry point and explicitly package the model catalogues,
  web assets and chat templates. Include both `LICENSE` and `NOTICE`.
- Keep release tests that build tagged source distributions and wheels, and
  validate the installed CLI and assets outside the source tree.
- Keep the Chrome browser checks alongside the Python tests.
- Keep GitHub Actions Pages deployment and the `_pypi.yml` trusted publisher
  in the `release` environment. The documentation remains at the existing
  URL; a version switcher and `gh-pages` publishing are not configured.
- Keep Sphinx builds independent of network requests. Reference pages
  describe the user-facing commands and settings; internal Python modules
  are not presented as a supported public API.
- Keep a separate `docs` dependency group, included in `dev`.
- Let Ruff format code without splitting embedded prompts and shell commands.
  Preserve existing exception names, CLI enums and download cancellation access
  through narrow lint exceptions.
- Keep the container image's adaptations (see
  [the Kubernetes decision](0004-kubernetes-image-and-chart.md)): Helm in the
  developer stage, the `engine-dist` stage and the engine install in the build
  stage, the runtime CA certificates, ldd check and environment, and the
  `.dockerignore` entries for engine builds.
- Keep the `container` job's engine wiring: `needs: [test, engines]`, the
  `engine-dist` build context and the check that skips the image until the
  pinned engine exists, pushes from `main` and tags, and the `container` and
  `helm` jobs in the release job's `needs`.

## Consequences

Contributors install the project with `uv sync --locked` before running it.
`uv run --locked tox -p` provides the shared Python and documentation checks.
Future `copier update` changes must be reviewed against these adaptations.
