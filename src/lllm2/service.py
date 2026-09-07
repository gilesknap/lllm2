"""Install the panel into the current user's systemd manager."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

UNIT = "lllm2-panel.service"
MARKER = "# Managed by lllm2 service install"
ENVIRONMENT = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "LLLM2_MODELS_DIR",
    "LLLM2_STATE_DIR",
    "LLLM2_ENGINE_HOME",
    "LLLM2_ENGINE_ROOTS",
    "LLLM2_ENGINE_PORT",
)


def _quote(value: str, *, command: bool = False) -> str:
    """Quote one systemd word, including specifier and ExecStart expansion."""
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Service settings cannot contain control characters.")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def render_unit(host: str, port: int) -> str:
    """Use the current interpreter without resolving its virtualenv symlink."""
    if not host or not 1 <= port <= 65535:
        raise ValueError("Provide a host and a port between 1 and 65535.")
    command = " ".join(
        _quote(arg, command=True)
        for arg in (
            sys.executable,
            "-m",
            "lllm2",
            "panel",
            "--host",
            host,
            "--port",
            str(port),
        )
    )
    environment = "\n".join(
        "Environment=" + _quote(f"{name}={os.environ[name]}")
        for name in ENVIRONMENT
        if name in os.environ
    )
    return f"""{MARKER}
[Unit]
Description=lllm2 web panel

[Service]
Type=simple
WorkingDirectory=%h
ExecStart={command}
{environment}
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=default.target
"""


def install_service(
    *, host: str = "127.0.0.1", port: int = 8082, start: bool = True
) -> Path:
    """Write, enable and optionally restart a managed systemd user unit."""
    if sys.platform != "linux" or shutil.which("systemctl") is None:
        raise RuntimeError("Panel service installation requires Linux with systemd.")
    content = render_unit(host, port)
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    if not config_home.is_absolute():
        raise ValueError("XDG_CONFIG_HOME must be an absolute path.")
    unit = config_home / "systemd/user" / UNIT
    if unit.is_symlink() or (
        unit.exists() and not unit.read_text().startswith(MARKER + "\n")
    ):
        raise FileExistsError(f"Refusing to replace an unmanaged service: {unit}")
    unit.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically; keep environment settings private to this user.
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", dir=unit.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(content)
            file.flush()
            temporary.replace(unit)
        finally:
            temporary.unlink(missing_ok=True)
    commands = [["daemon-reload"], ["enable", UNIT]]
    if start:
        commands.append(["restart", UNIT])
    for args in commands:
        result = subprocess.run(
            ["systemctl", "--user", *args], capture_output=True, text=True
        )
        if result.returncode:
            raise RuntimeError(
                f"Service file saved at {unit}, but systemctl --user {' '.join(args)} failed: "
                f"{result.stderr.strip() or result.stdout.strip()}. "
                "Run from a login session with a systemd user manager, then retry. "
                f"Check journalctl --user -u {UNIT} for startup errors."
            )
    return unit
