"""The CI sweep stops everything in its Modal environment and proves it."""

import contextlib
import io
import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/modal_sweep.py"
script = runpy.run_path(str(SCRIPT))
ENV = "lllm2-ci"


class FakeModalCli:
    """The Modal CLI commands the sweep runs, with modal 1.5's JSON shapes.

    A stopped app reports ``stopping...`` and its tasks for one more listing,
    as a real container takes a few seconds to wind down.
    """

    def __init__(self, apps=(), containers=()):
        self.apps = {app["app_id"]: dict(app) for app in apps}
        self.containers = {item["container_id"]: dict(item) for item in containers}
        self.commands = []
        self.winding = set()
        self.stubborn = set()

    def __call__(self, command, **_kwargs):
        assert command[:3] == [sys.executable, "-m", "modal"]
        args = command[3:]
        self.commands.append(args)
        if args[:2] in (["app", "list"], ["container", "list"]):
            assert args[2:] == ["--env", ENV, "--json"]
            if args[0] == "app":
                return self.result(json.dumps(self.app_listing()))
            return self.result(json.dumps(list(self.containers.values())))
        if args[:2] == ["app", "stop"]:
            assert args[2:5] == ["--env", ENV, "--yes"]
            app_id = args[5]
            if app_id in self.stubborn:
                raise subprocess.CalledProcessError(1, command, stderr="refused")
            self.apps[app_id]["state"] = "stopping..."
            self.winding.add(app_id)
            return self.result("")
        if args[:2] == ["container", "stop"]:
            assert args[2] == "--yes"
            self.containers.pop(args[3])
            return self.result("")
        raise AssertionError(args)

    def app_listing(self):
        listing = [dict(app) for app in self.apps.values()]
        for app_id in list(self.winding):
            self.winding.discard(app_id)
            self.apps[app_id].update(state="stopped", tasks="0")
            for key, item in list(self.containers.items()):
                if item["app_id"] == app_id:
                    del self.containers[key]
        return listing

    @staticmethod
    def result(stdout):
        return subprocess.CompletedProcess([], 0, stdout=stdout)


def app(app_id, state="deployed", tasks="0", description="lllm2"):
    return {
        "app_id": app_id,
        "description": description,
        "state": state,
        "tasks": tasks,
        "created_at": "2026-10-01 10:00 UTC",
        "stopped_at": None,
    }


def container(container_id, app_id, app_name="lllm2"):
    return {
        "container_id": container_id,
        "app_id": app_id,
        "app_name": app_name,
        "start_time": "2026-10-01 10:01 UTC",
    }


def run(cli, *options):
    output = io.StringIO()
    with (
        patch("subprocess.run", side_effect=cli),
        patch("time.sleep"),
        contextlib.redirect_stdout(output),
    ):
        code = script["main"](["modal_sweep.py", ENV, *options])
    return code, output.getvalue()


def test_sweep_stops_every_app_and_container_then_verifies():
    cli = FakeModalCli(
        apps=[
            app("ap-1", tasks="1"),
            app("ap-2", state="ephemeral", description="other"),
            app("ap-old", state="stopped"),
        ],
        containers=[container("ta-1", "ap-1"), container("ta-x", "ap-gone", "x")],
    )
    code, output = run(cli)
    assert code == 0, output
    stopped = [c[-1] for c in cli.commands if c[1] == "stop"]
    assert set(stopped) == {"ap-1", "ap-2", "ta-x"}
    assert "ap-old" not in stopped
    assert all(a["state"] == "stopped" for a in cli.apps.values())
    assert not cli.containers
    assert "every app stopped, no containers" in output


def test_sweep_fails_when_anything_keeps_running():
    cli = FakeModalCli(apps=[app("ap-1", tasks="1")], containers=[])
    cli.stubborn.add("ap-1")
    code, output = run(cli, "--timeout", "0")
    assert code == 1
    assert "ap-1" in output and "refused" in output


def test_check_waits_for_containers_and_stops_nothing():
    cli = FakeModalCli(apps=[app("ap-1")])
    code, output = run(cli, "--check")
    # A deployed app with no containers costs nothing.
    assert code == 0, output
    assert not [c for c in cli.commands if c[1] == "stop"]

    # A cancelled call's container winds down within the timeout.
    cli = FakeModalCli(apps=[app("ap-1", tasks="1")])
    cli.winding.add("ap-1")
    code, output = run(cli, "--check")
    assert code == 0, output
    assert len([c for c in cli.commands if c[:2] == ["app", "list"]]) == 2

    cli = FakeModalCli(
        apps=[app("ap-1", tasks="1")], containers=[container("ta-1", "ap-1")]
    )
    code, output = run(cli, "--check", "--timeout", "0")
    assert code == 1
    assert "ta-1" in output and "did not stop" in output
    assert not [c for c in cli.commands if c[1] == "stop"]


@pytest.mark.parametrize("environment", ["main", ""])
def test_the_default_environment_is_never_swept(environment):
    with (
        patch("subprocess.run") as run_command,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        assert script["main"](["modal_sweep.py", environment]) == 2
    run_command.assert_not_called()
