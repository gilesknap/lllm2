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

The pre-commit hook `helm-schema` regenerates `Charts/lllm2/values.schema.json`
from the annotations in `values.yaml`, and `tests/test_helm_chart.py` renders
the chart. Both need Helm 3.17.1 and the
[helm-values-schema-json](https://github.com/losisin/helm-values-schema-json)
plugin v2.5.0, which the devcontainer installs. Without them, the chart tests
skip and `SKIP=helm-schema` skips the hook; CI runs both.

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

Every run also builds the container image and smoke-tests its CLI and engine
(`container / build`), and lints, renders and packages the Helm chart
(`helm / package`). Pushes to `main` publish the image as
`ghcr.io/gilesknap/lllm2:main`. Version tags publish the image with the tag and
`latest`, push the chart to `oci://ghcr.io/gilesknap/charts/lllm2`, and attach
the chart's values schema, `lllm2.schema.json`, to the GitHub release. The
release, and so PyPI, waits for both jobs. See
[Deploy on Kubernetes](deploy-on-kubernetes.md).

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

4. After the first image and chart are published, make both GitHub packages,
   `lllm2` and `charts/lllm2`, public (**Package settings > Change
   visibility**). New packages start private, and then anonymous pulls and
   `helm install` fail.

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
CI checks that the built wheel and the image's `lllm2 --version` match the
release tag. Each release needs a new version. Once published, users install with `uv tool install --upgrade lllm2`
or upgrade with `uv tool upgrade lllm2`.

The tagged commit must contain the workflow changes: fixing `main` does not
rerun an existing tag.

The chart's `appVersion` is the tag, which is also the image tag. Its chart
version is the tag as SemVer: `0.1.2` and `v0.1.2` publish chart `0.1.2`, and a
pre-release such as `0.1.2rc1` publishes `0.1.2-rc.1`. Any other tag form
publishes no chart, with a warning.

## CUDA release engines

`src/lllm2/engine_release.py` defines the llama.cpp pin and both CUDA image
versions. Change the pin in a normal PR when panel features need a newer engine.

The `Propose llama.cpp bump` workflow (`.github/workflows/llama-cpp-bump.yml`)
does this weekly, or on manual dispatch. When upstream llama.cpp has a newer
per-commit `bNNNNN` build than `LLAMA_CPP_REF` (the stable `vX.Y.Z` releases
lag new model support, so they are not tracked), `.github/scripts/bump_llama_cpp.py` moves the pin
forward and the workflow force-pushes `bot/llama-cpp-bump` and opens or updates a
single "Bump llama.cpp to bNNNNN" PR with the upstream compare link. It also
dispatches the engine workflow on that branch, so both CUDA tracks are built and
smoke-tested before review. Pushes made with the default `GITHUB_TOKEN` do not
trigger CI: set a `LLAMA_CPP_BUMP_TOKEN` repository secret (a fine-grained token
with contents and pull request write access) to run CI on the PR automatically,
or close and reopen the PR to start it. The repository setting "Allow GitHub
Actions to create and approve pull requests" must be enabled. Merging the PR
changes nothing for users until a new lllm2 version is tagged; they then get the
new engine with `uv tool install --upgrade lllm2`.

Bump PRs can also merge and release themselves. When enabled, the bump workflow
turns on GitHub auto-merge for the `bot/llama-cpp-bump` PR, and GitHub merges it
only once every required check passes. `Release llama.cpp bump`
(`.github/workflows/llama-cpp-release.yml`) then waits for CI to pass on `main`
for that merge commit and pushes the next patch tag after the highest `X.Y.Z`
tag (`0.9.0` becomes `0.9.1`), which starts the normal tag pipeline: engines,
GitHub release and PyPI. `.github/scripts/tag_bump_release.py` tags only the
merge commit of a merged `bot/llama-cpp-bump` PR from this repository, and never
a commit that already has a version tag, so each bump releases at most once.
Nothing else is auto-merged or auto-tagged. A bump PR merged by hand is released
the same way.

To enable it, a maintainer:

- Adds the `LLAMA_CPP_BUMP_TOKEN` secret described above. Its pushes start CI
  on the PR, the auto-merge it enables is attributed to it so the merge starts
  CI on `main`, and the release tag must be pushed with it: a tag pushed with
  `GITHUB_TOKEN` does not start the tag pipeline. Grant the fine-grained token
  contents and pull request read and write access to this repository.
- Adds the `MODAL_TOKEN_ID` and `MODAL_SECRET` secrets to the GitHub
  environment `modal` for the GPU smoke test (see
  [GPU smoke test on Modal](#gpu-smoke-test-on-modal)). Without them the GPU
  test is skipped and auto-merge stays off.
- Enables **Settings > General > Allow auto-merge**.
- Protects `main` with a branch protection rule or ruleset that requires the
  CI checks to pass before merging: `lint / run`, `test (3.11)` to
  `test (3.14)`, `browser`, `docs-build`, `dist / build`, `container / build`,
  `helm / package` and the GPU smoke test, `gpu-smoke`. A skipped check counts
  as passed, so `gpu-smoke` gates only the PRs it runs on: bump PRs and PRs that
  change Modal code. Without required checks, GitHub refuses to enable
  auto-merge on a PR that is already mergeable. Add `container / build` and
  `helm / package` only once the open bump PR's branch has those jobs, or that
  PR waits for checks that never run.
- Sets the repository variable `LLAMA_CPP_AUTO_RELEASE` to `true` (**Settings >
  Secrets and variables > Actions > Variables**). Delete it to return to
  merging and tagging by hand.

Users who hit a problem with a new engine can return to an earlier release with
`uv tool install lllm2==X.Y.Z` (see [Upgrade lllm2](../tutorials/upgrade.md)).
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

The container image installs the CUDA 12 engine the same way. When the run's
`engines` job built it, `_container.yml` downloads the `engine-cuda12`
artifact and passes it to the Docker build as the `engine-dist` build context;
otherwise the build installs the published tarball.
`.github/scripts/image_engines.py` checks first. After a bump PR merges, no
release has its engine until the bump's release tag, so `main` builds no image
until then: the job passes with a notice, and CI stays green for the bump
release. On a tag the same case fails the job, so a release never ships
without its image.

The Modal image normally installs the same published tarballs. To test an
engine that is not released yet, such as a bump PR's, on a Modal GPU, point
`LLLM2_MODAL_ENGINE_DIR` at a directory holding both tracks' tarballs and
`.sha256` files (the `engine-cuda*` workflow artifacts) before the app deploys.
The image then copies them in and installs them with the same checksum and
metadata checks, and the app redeploys whenever the tarballs change. Use it only
in a Modal environment kept for testing: the deployment replaces the `lllm2`
app in the selected environment.

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

## GPU smoke test on Modal

CI otherwise proves only that an engine compiles and that `llama-server --help`
runs against a stub driver. The `gpu-smoke` job runs
`tests/test_modal_integration.py` on a Modal T4: it deploys the app, downloads
`qwen3-8b` into the Volume (only on the first run), serves it, streams a
completion and stops the call. It needs no approval. It runs:

- on `bot/llama-cpp-bump` PRs;
- on other PRs from branches of this repository that change Modal code;
- when CI is run by hand (**Actions > CI > Run workflow**), on any branch.

On bump PRs and manual runs, the `engines` job first builds the engines when no
release has them yet, and the app installs those (`LLLM2_MODAL_ENGINE_DIR`).
Otherwise the app installs the released engine for the current pins, as a user's
deployment does.

The `modal-changes` job decides which PRs change Modal code. It lists the files
the PR changes and looks for those in `PATHS` in
`.github/scripts/modal_changes.py`: the remote engine (`remote.py`), the engine
proxy (`proxy.py`), the Modal provider and app, the integration test, the sweep
and change scripts, and `ci.yml`, which defines the job. Other PRs skip the job
and spend nothing. So do PRs from forks, which get no secrets; `modal-changes`
adds a notice when such a PR changes Modal code. Pushes to `main` do not run
the job: their PR already did, and a Modal failure on `main` would stop the
bump release, which tags only a commit whose CI passed.

A skipped job counts as a passed check. So when `engines` or `modal-changes`
fails, `gpu-smoke` fails at once rather than skipping, and a bump PR whose
engines failed is never merged automatically.

The concurrency group `modal-lllm2-ci` serialises the runs, so two PRs never
deploy the app or spend at the same time. GitHub keeps only one run waiting:
when another arrives, it cancels the older waiting run. Re-run that run's
`gpu-smoke` job.

The job runs in its own Modal environment, `lllm2-ci`, so it never touches the
apps you run yourself. `lllm2-ci` is reserved for CI: every run stops every app
and container in it, whoever started them. Modal tokens cover the whole
workspace; the sweep below names the environment on every command and refuses
`main`.

Every run ends with `.github/scripts/modal_sweep.py`, which stops every app
and container in `lllm2-ci` and then lists the environment again until nothing
runs, failing the job if anything still does after two minutes. The step uses
`if: always()`, so it also runs after a failure, a job timeout or a normal
cancellation; only a force-cancel or a lost runner skips it. Before the test,
the same sweep stops anything an earlier run left. After a passing test,
`modal_sweep.py --check` first fails the job if lllm2 left any container
running, which the final sweep would otherwise hide. The `lllm2` app is stopped
too; the next run redeploys it. The job also has a 60-minute timeout.

Set it up once. Create the Modal environment:

```bash
uv run --extra modal modal environment create lllm2-ci
```

Create a Modal API token (**Settings > API Tokens** in the Modal dashboard, or
`modal token new`, which writes it to `~/.modal.toml`). In GitHub, create the
environment `modal` (**Settings > Environments > New environment**) with no
protection rules, so runs need no approval. Store the token ID and the token
secret as its secrets:

```bash
gh secret set MODAL_TOKEN_ID --env modal --repo gilesknap/lllm2
gh secret set MODAL_SECRET --env modal --repo gilesknap/lllm2
```

The job maps `MODAL_SECRET` to `MODAL_TOKEN_SECRET`, the variable the Modal
client reads, and sets `MODAL_ENVIRONMENT=lllm2-ci`. It uses the environment
only for its secrets, so GitHub records no deployment. Without both secrets the
job reports a notice and passes without touching Modal. The bump workflow reads
the environment too, only to check that both secrets exist before it turns on
auto-merge. Anyone who can push a branch to this repository can use the token
from a PR.

To test a branch by hand, run CI on it with **Run workflow**. To run the test
from a checkout instead, use a Modal environment of your own, never `lllm2-ci`,
where a CI run would stop it part way.

Expected cost: about US$0.05 to US$0.10 per run. A run holds a T4 (US$0.59 per
hour) for about five minutes, for the GPU probe, model load and completion,
plus CPU time to build the image and, on the first run only, to download the
5 GB model. Each push to a PR that changes Modal code runs it again, and a
weekly bump costs well under US$1 a month. Each new bump also uploads the two
engine tarballs (about 1.8 GB) from the runner into the Modal image.
