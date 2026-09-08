# Development and releases

## Local checks

The package lives in `src/lllm2`. Install the editable project and development
tools from a checkout before running commands:

```bash
uv sync --locked
uv run --locked tox -p
node tests/test_ui_browser.cjs
uv run --locked python -m build
uv run --locked twine check --strict dist/*
```

Browser checks need Node.js 22 and Chrome; set `CHROME_BIN` if Chrome is not
at `/opt/google/chrome/chrome`. Tests use mocked engines and browser data;
no GPU or model download is required. `tox -p` runs pre-commit, mypy, pytest with
coverage, and the strict documentation build. Run one check with
`uv run --locked tox -e tests` or `uv run --locked tox -e docs`.

Preview the documentation with `uv run --locked tox -e docs-autobuild`, or serve
the built files with `uv run python -m http.server --directory build/html`.
Install the Git hooks with `uv run pre-commit install`. A VS Code devcontainer
and editor settings are also provided.

## Update the template

The project adopts DLS `python-copier-template` release `5.4.0`.
`.copier-answers.yml` records the template URL, version and project choices.
From a clean checkout with your changes committed, run:

```bash
uvx copier update --trust
git diff
```

Review the merged changes, resolve any conflicts and run the checks above.
Keep the project adaptations described in the
[template adoption decision](../explanations/decisions/0002-switched-to-python-copier-template.md),
particularly package assets, browser tests and the existing publishing setup.

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

`setuptools-scm` derives the package version from Git: `0.1.2` and
`v0.1.2` both build version `0.1.2`. There is no version string to update in
`pyproject.toml` or `uv.lock` for a release. Builds between tags get a development
version. It generates `src/lllm2/_version.py`, which is ignored by Git.
CI checks that the built wheel matches the release tag. Each release
needs a new version. Once published, users install with `uv tool install --upgrade lllm2`
or upgrade with `uv tool upgrade lllm2`.

The tagged commit must contain the workflow changes: fixing `main` does not
rerun an existing tag.

## CUDA release engines

`src/lllm2/engine_release.py` defines the llama.cpp pin and both CUDA image
versions. Change the pin in a normal PR when panel features need a newer engine.
Version-tag CI first looks for matching llama.cpp/CUDA asset names in earlier
GitHub releases. Each track skips building, downloading and uploading when its exact tarball
and checksum already exist on a published release; only a missing combination is built in NVIDIA's Rocky Linux 8
development image. For new or draft-seeded artifacts, CI verifies checksums and engine metadata, checks each
packaged binary with the panel probe, and attaches the tarballs, SHA-256 files
and Python distributions to the same GitHub release. Runtime libraries retain
their symlinks, so each library payload is stored only once.

A Python-only release does not compile or attach engines. The installer finds
matching pins on earlier published releases, independently of its package version. Changing a CUDA pin rebuilds
that track; changing the llama.cpp pin rebuilds both. If changing the engine
build configuration, change the engine pins too: artifact names are immutable
identities, not a cache keyed by Python changes. Archive metadata records the
original build release; installation separately records the lllm2 version that
installed it. The installer and engine list match by pins, not by build release.

A maintainer can seed a draft version release with locally built and validated
tarballs and checksums before pushing its version tag. CI reuses those assets
as well, then adds the checked Python distributions and publishes the draft. PyPI publishing
waits for that release. The engine workflow also supports manual dispatch from a
selected branch for testing pin changes without publishing a release.

To iterate locally with Podman (no GPU required for compilation):

```bash
mkdir -p engine-dist engine-checks
CUDA=$(uv run python -c 'from lllm2.engine_release import CUDA_TRACKS; print(CUDA_TRACKS["13"])')
VERSION=$(uv run python -c 'from lllm2 import __version__; print(__version__)')
podman run --rm -e BUILD_JOBS=2 \
  -v "$PWD:/repo:ro" -v "$PWD/engine-dist:/out" \
  -v "$PWD/engine-checks:/checks" \
  "docker.io/nvidia/cuda:${CUDA}-devel-rockylinux8" \
  bash /repo/scripts/build-engine.sh 13 "$VERSION"
mkdir -p engine-smoke
tar -xzf engine-dist/*.tar.gz -C engine-smoke
LD_LIBRARY_PATH="$PWD/engine-checks" uv run python scripts/check-engine.py engine-smoke/llama-server
```

Repeat with track `12` and its image version. Use an empty output directory for
each smoke test. Allow several GB for images, build files and bundled CUDA
libraries. The builder keeps upstream's non-native CUDA architecture defaults
and uses an AVX2 CPU baseline. Runtime inference and performance measurements
still require an NVIDIA GPU; a successful `--help` probe does not test GPU kernels.

On a GPU-less build host, `engine-checks` contains the CUDA driver stub used only
for linking and smoke tests. It is never included in the release tarball. GPU
evaluation must use the real NVIDIA driver and omit this stub from the environment.
