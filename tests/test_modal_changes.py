"""A pull request runs the paid GPU smoke test only when it changes Modal code."""

import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/modal_changes.py"
script = runpy.run_path(str(SCRIPT))
OTHER = "src/lllm2/cli.py"


def git(*args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Change test",
            "-c",
            "user.email=change-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def commit(message: str) -> str:
    git("add", "--all")
    git("commit", "-qm", message)
    return git("rev-parse", "HEAD")


@pytest.fixture
def base(tmp_path, monkeypatch):
    """A repository whose first commit holds every covered file and another."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.chdir(repository)
    git("init", "-q")
    for path in (*script["PATHS"], OTHER):
        write(path, "first\n")
    return commit("Base")


def run(base, capsys, tmp_path, monkeypatch):
    """Run the script on the change from ``base`` to HEAD."""
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert script["main"](["modal_changes.py", base, git("rev-parse", "HEAD")]) == 0
    return output.read_text(), capsys.readouterr().out


def test_a_change_to_modal_code_runs_the_smoke_test(
    base, capsys, tmp_path, monkeypatch
):
    write("src/lllm2/modal_app.py", "second\n")
    write(OTHER, "second\n")
    commit("Change the Modal app and the CLI")
    output, printed = run(base, capsys, tmp_path, monkeypatch)
    assert output == "modal=true\n"
    assert "src/lllm2/modal_app.py" in printed
    assert OTHER not in printed


def test_other_changes_spend_nothing(base, capsys, tmp_path, monkeypatch):
    write(OTHER, "second\n")
    write("docs/index.md", "new\n")
    commit("Change the CLI and the docs")
    output, printed = run(base, capsys, tmp_path, monkeypatch)
    assert output == "modal=false\n"
    assert "No change that gpu-smoke covers" in printed


def test_moving_modal_code_away_counts(base, capsys, tmp_path, monkeypatch):
    # Git would otherwise report a rename under its new name only.
    git("mv", "src/lllm2/proxy.py", "src/lllm2/tunnel.py")
    commit("Rename the proxy")
    output, printed = run(base, capsys, tmp_path, monkeypatch)
    assert output == "modal=true\n"
    assert "src/lllm2/proxy.py" in printed


def test_every_covered_path_exists():
    """A renamed module must not silently stop running the test."""
    missing = [path for path in script["PATHS"] if not (ROOT / path).is_file()]
    assert not missing, f"Update PATHS in {SCRIPT.name}: {missing}"
