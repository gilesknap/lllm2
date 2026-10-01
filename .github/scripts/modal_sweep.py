"""Stop every Modal app and container in a CI environment, then prove it.

Usage: modal_sweep.py ENVIRONMENT [--check] [--timeout SECONDS]

The GPU smoke test runs in a Modal environment kept for CI, so this may stop
everything there without touching anyone's interactive apps. It stops each app
that is not stopped yet and each running container, then lists both again until
nothing runs. It exits 1 if anything still runs after ``--timeout`` seconds.

With ``--check`` it stops nothing and only waits for every container to end. CI
runs that after a passing test: lllm2 must have stopped its own GPU containers,
and the sweep would otherwise hide a leak. A deployed app with no containers
costs nothing and passes the check.

Run it with the ``modal`` package installed, for example through
``uv run --extra modal``. A container takes a few seconds to wind down after
its call is cancelled, so both modes poll rather than list once.
"""

import argparse
import json
import subprocess
import sys
import time

#: Interactive apps live in the default environment; never sweep it.
PROTECTED = {"", "main"}


def modal(*args: str) -> str:
    """Run a Modal CLI command and return its output."""
    return subprocess.run(
        [sys.executable, "-m", "modal", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def listing(environment: str) -> tuple[list[dict], list[dict]]:
    """Return the environment's apps and running containers."""
    apps = json.loads(modal("app", "list", "--env", environment, "--json"))
    containers = json.loads(modal("container", "list", "--env", environment, "--json"))
    return apps, containers


def busy(apps: list[dict], containers: list[dict]) -> list[str]:
    """Describe every app with running tasks and every running container."""
    found = [
        f"app {app['app_id']} ({app['description']}) runs {app['tasks']} tasks"
        for app in apps
        if int(app.get("tasks") or 0) > 0
    ]
    found += [
        f"container {item['container_id']} of app {item['app_id']} ({item['app_name']})"
        for item in containers
    ]
    return found


def live(apps: list[dict]) -> list[dict]:
    """Return the apps that are not stopped yet: deployed, ephemeral, stopping."""
    return [app for app in apps if app.get("state") != "stopped"]


def sweep(environment: str) -> None:
    """Ask Modal to stop every live app and running container once."""
    apps, containers = listing(environment)
    stopping = live(apps)
    for app in stopping:
        print(f"Stopping app {app['app_id']} ({app['description']}, {app['state']})")
        try:
            modal("app", "stop", "--env", environment, "--yes", app["app_id"])
        except subprocess.CalledProcessError as error:
            # Keep going: the final listing decides whether the sweep worked.
            print(f"  failed: {error.stderr.strip()}")
    # Stopping an app terminates its containers. Stop any that remain, and any
    # that belong to apps the listing did not show.
    if stopping:
        containers = listing(environment)[1]
    for item in containers:
        print(f"Stopping container {item['container_id']} ({item['app_name']})")
        try:
            modal("container", "stop", "--yes", item["container_id"])
        except subprocess.CalledProcessError as error:
            print(f"  failed: {error.stderr.strip()}")


def wait(environment: str, timeout: float, check: bool, poll: float = 5.0) -> list[str]:
    """List until nothing runs. Returns what still runs at the timeout.

    Without ``check``, every round stops what is still live first, so a
    container that Modal starts again while an app stops is stopped too.
    """
    deadline = time.monotonic() + timeout
    while True:
        if not check:
            sweep(environment)
        apps, containers = listing(environment)
        left = busy(apps, containers)
        if not check:
            left += [
                f"app {app['app_id']} ({app['description']}) is {app['state']}"
                for app in live(apps)
            ]
        if not left or time.monotonic() >= deadline:
            return left
        time.sleep(poll)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("environment")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    options = parser.parse_args(argv[1:])
    if options.environment in PROTECTED:
        print(f"Refusing to sweep the {options.environment!r} Modal environment.")
        return 2
    left = wait(options.environment, options.timeout, options.check)
    if left:
        print(f"Still running in Modal environment {options.environment}:")
        print("\n".join(f"  {line}" for line in left))
        if options.check:
            print("lllm2 did not stop these; the always-run sweep stops them next.")
        return 1
    state = "no containers" if options.check else "every app stopped, no containers"
    print(f"Modal environment {options.environment}: {state}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
