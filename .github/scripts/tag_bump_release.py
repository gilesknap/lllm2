"""Choose the patch release tag for a merged llama.cpp bump commit.

Usage: tag_bump_release.py SHA

``.github/workflows/llama-cpp-release.yml`` runs this after CI passes on a
``main`` commit. The commit gets a tag only when it is the merge commit of a
merged ``bot/llama-cpp-bump`` PR from this repository and carries no version
tag yet, so nothing else is tagged and each bump tags at most once. The tag is
the next patch after the highest ``X.Y.Z`` tag. Writes ``tag=`` (empty when
nothing should be tagged) to ``GITHUB_OUTPUT``. Uses the standard library and
the ``gh`` and ``git`` commands only.
"""

import json
import os
import re
import subprocess
import sys

BRANCH = "bot/llama-cpp-bump"
VERSION = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")


def next_patch(tags: list[str]) -> str:
    """Return the patch version after the highest ``X.Y.Z`` tag."""
    versions = [
        tuple(int(part) for part in match.groups())
        for tag in tags
        if (match := VERSION.fullmatch(tag.strip()))
    ]
    if not versions:
        raise ValueError("No X.Y.Z release tag to follow")
    major, minor, patch = max(versions)
    return f"{major}.{minor}.{patch + 1}"


def bump_pull(pulls: list[dict], sha: str, repository: str) -> dict | None:
    """Return the merged bump PR whose merge commit is ``sha``, if any."""
    for pull in pulls:
        head = pull.get("head") or {}
        if (
            pull.get("merged_at")
            and pull.get("merge_commit_sha") == sha
            and head.get("ref") == BRANCH
            and (head.get("repo") or {}).get("full_name") == repository
        ):
            return pull
    return None


def release_tag(sha: str, repository: str) -> str:
    """Return the tag for ``sha``, or an empty string when it gets none."""
    pulls = json.loads(
        subprocess.run(
            ["gh", "api", f"repos/{repository}/commits/{sha}/pulls"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    pull = bump_pull(pulls, sha, repository)
    if pull is None:
        print(f"{sha} is not a merged {BRANCH} commit; nothing to tag")
        return ""

    def tags(*options: str) -> list[str]:
        return subprocess.run(
            ["git", "tag", "--list", *options],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()

    if existing := [tag for tag in tags("--points-at", sha) if VERSION.fullmatch(tag)]:
        print(f"{sha} is already released as {', '.join(existing)}")
        return ""
    tag = next_patch(tags())
    print(f"Tagging {sha} from PR #{pull['number']} as {tag}")
    return tag


def main(argv: list[str]) -> None:
    tag = release_tag(argv[1], os.environ["GH_REPO"])
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as handle:
            handle.write(f"tag={tag}\n")


if __name__ == "__main__":
    main(sys.argv)
