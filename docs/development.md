# Development and releases

## Local checks

From a checkout, run:

```bash
uv sync --locked --group docs
uv run --locked python -m unittest discover -s tests -v
node tests/test_ui_browser.cjs
uv run --locked sphinx-build -W --keep-going -b html docs docs/_build/html
uv run --locked python -m build
uv run --locked twine check --strict dist/*
```

Browser checks need Node.js 22 and Chrome; set `CHROME_BIN` if Chrome is not
at `/opt/google/chrome/chrome`. Tests use mocked engines and browser data;
no GPU or model download is required. Preview the built documentation with
`uv run python -m http.server --directory docs/_build/html`.

## CI and publishing setup

Pull requests, pushes to `main` and all tags run Python tests, browser tests,
a strict docs build and distribution checks. CI installs the built wheel in
isolation and checks its CLI, catalogue, recommendations and UI/template assets.
Successful `main` builds deploy documentation; version tags publish the same
checked distribution artifacts to PyPI.

Configure these once in the hosting accounts:

1. In GitHub **Settings → Pages**, select **GitHub Actions** as the build source
   ([Pages instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)).
2. Create the repository environment `release`.
3. Add a [PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
   for project `lllm2` (or a pending publisher for the first release):

| Field | Value |
| --- | --- |
| Owner | `gilesknap` |
| Repository | `lllm2` |
| Workflow | `_pypi.yml` |
| Environment | `release` |

The reusable publishing workflow is `_pypi.yml`; use that filename when
registering the publisher. [Trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
uses the workflow identity, with no PyPI API token in repository secrets.

## Release

Run the checks above and merge the changes to `main`. Tag the merged commit
with the new version and push it, for example:

```bash
git switch main
git pull --ff-only
git tag 0.1.2
git push origin refs/tags/0.1.2
```

Hatch derives the package version from Git using `hatch-vcs`: `0.1.2` and
`v0.1.2` both build version `0.1.2`. There is no version string to update in
`pyproject.toml` or `uv.lock` for a release. Builds between tags get a development
version. CI checks that the built wheel matches the release tag. Each release
needs a new version. Once published, users install with `uv tool install lllm2`
or upgrade with `uv tool upgrade lllm2`.

The tagged commit must contain the workflow changes: fixing `main` does not
rerun an existing tag.
