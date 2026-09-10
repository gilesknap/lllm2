#!/usr/bin/env python3
"""Launch the Pi image with local rootless Podman."""

import argparse
import getpass
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

IMAGE = "ghcr.io/gilesknap/lllm2-pi:latest"


def mount(source: Path, target: str) -> list[str]:
    # --mount uses CSV syntax. Refuse ambiguous paths instead of accidentally
    # turning part of a filename into another mount option.
    if any(c in str(source) for c in ',\n\r"'):
        raise ValueError(f"Unsupported characters in mount path: {source}")
    return ["--mount", f"type=bind,src={source},dst={target},rw"]


def command(
    options, args, project: Path, home: Path, tty: bool, token_file: Path | None = None
) -> list[str]:
    if project == Path("/") or project == Path.home().resolve():
        raise ValueError("Run from a project directory, not / or your home directory.")
    argv = [
        "podman",
        "run",
        "--rm",
        "--init",
        "-i",
        "--pull=missing",
        "--user=0:0",
        "--userns=host",
        # Host here is the rootless engine's user namespace (not host UID 0).
        # Upstream's nested bwrap/pasta needs userns-related syscalls permitted.
        "--security-opt=seccomp=unconfined",
        "--security-opt=apparmor=unconfined",
        "--security-opt=label=disable",
        "--device=/dev/net/tun",
        "--network=host",
        "--workdir=/workspaces",
        "--env=HOME=/root",
        "--env=CLAUDE_SANDBOX_EGRESS_JAIL=1",
        f"--env=CLAUDE_SANDBOX_NO_FORGE={0 if token_file else 1}",
        "--env=CLAUDE_SANDBOX_WORKSPACE_ROOT=/workspaces",
        f"--env=CLAUDE_SANDBOX_LOCAL_MODEL_PORT={options.model_port}",
        f"--env=TERM={os.environ.get('TERM', 'xterm-256color')}",
    ]
    if tty:
        argv.append("-t")
    argv += mount(project, "/workspaces") + mount(home, "/root/.pi")
    if token_file:
        argv += [
            "--mount",
            f"type=bind,src={token_file},dst=/run/secrets/pi-github-token,ro",
        ]
    argv += [options.image, *args]
    return argv


def check_runtime() -> None:
    if sys.platform != "linux":
        raise RuntimeError("This launcher requires Linux and local rootless Podman.")
    if not shutil.which("podman"):
        raise RuntimeError(
            "Install rootless Podman first; see the Pi container guide in the lllm2 docs."
        )
    if any(os.environ.get(key) for key in ("CONTAINER_HOST", "CONTAINER_CONNECTION")):
        raise RuntimeError(
            "Remote Podman is unsupported: mounts must refer to this host."
        )
    result = subprocess.run(
        ["podman", "info", "--format=json"], capture_output=True, text=True, check=True
    )
    info = json.loads(result.stdout)
    host = info.get("host", {})
    if host.get("security", {}).get("rootless") is not True:
        raise RuntimeError("Refusing rootful Podman. Run as your ordinary host user.")
    if host.get("serviceIsRemote", False):
        raise RuntimeError("Remote Podman is unsupported.")
    if not Path("/dev/net/tun").exists():
        raise RuntimeError("/dev/net/tun is required by the network jail.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=IMAGE, help="Container image tag or digest")
    parser.add_argument(
        "--pi-dir",
        type=Path,
        default=Path.home() / ".pi",
        help="Shared Pi directory (default: ~/.pi, symlinks resolved)",
    )
    parser.add_argument(
        "--model-port",
        type=int,
        default=1920,
        help="Host loopback model port; 0 disables the relay",
    )
    parser.add_argument(
        "--pat",
        action="store_true",
        help="Prompt without echo for a GitHub PAT, used only by this container",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without creating directories or running Podman",
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments for Pi after --"
    )
    options = parser.parse_args(argv)
    if not 0 <= options.model_port <= 65535:
        parser.error("--model-port must be between 0 and 65535")
    if not options.image or options.image.startswith("-"):
        parser.error("--image must be an image reference")
    args = options.args
    if args[:1] == ["--"]:
        args = args[1:]
    return launch(
        args,
        image=options.image,
        pi_dir=options.pi_dir,
        model_port=options.model_port,
        pat=options.pat,
        dry_run=options.dry_run,
    )


def launch(
    args: list[str],
    *,
    image: str = IMAGE,
    pi_dir: Path | None = None,
    model_port: int = 1920,
    pat: bool = False,
    dry_run: bool = False,
) -> int:
    """Run a disposable Pi container, forwarding agent arguments literally."""
    options = argparse.Namespace(
        image=image,
        pi_dir=pi_dir if pi_dir is not None else Path.home() / ".pi",
        model_port=model_port,
        pat=pat,
        dry_run=dry_run,
    )
    try:
        home = options.pi_dir.expanduser().resolve()
        project = Path.cwd().resolve()
        if home == Path("/") or home == Path.home().resolve():
            raise ValueError(
                "--pi-dir must name a dedicated Pi configuration directory."
            )
        run = command(
            options, args, project, home, sys.stdin.isatty() and sys.stdout.isatty()
        )
        if options.dry_run:
            if options.pat:
                run = command(
                    options, args, project, home, False, Path("/temporary-secret/token")
                )
            print(shlex.join(run))
            return 0
        check_runtime()
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        if options.pat:
            # Refuse getpass's echoing fallback if no secure terminal is present.
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                try:
                    token = getpass.getpass("GitHub PAT (hidden): ")
                except (getpass.GetPassWarning, EOFError) as error:
                    raise RuntimeError(
                        "--pat requires a terminal with hidden input."
                    ) from error
            if not token.strip():
                raise ValueError("The GitHub PAT is empty.")
            # Never put the PAT in argv, Podman metadata/env, or shared Pi state.
            # /run/secrets is masked by upstream bwrap after gh imports the login.
            with tempfile.TemporaryDirectory(prefix="lllm2-pi-secret-") as directory:
                secret = Path(directory) / "token"
                secret.touch(mode=0o600)
                secret.write_text(token)
                run = command(
                    options,
                    args,
                    project,
                    home,
                    sys.stdin.isatty() and sys.stdout.isatty(),
                    secret,
                )
                result = subprocess.call(run)
        else:
            result = subprocess.call(run)
        return result if result >= 0 else 128 - result
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Pi container: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
