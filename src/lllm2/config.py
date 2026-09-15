import os
from pathlib import Path

MODELS_DIR = Path(
    os.environ.get("LLLM2_MODELS_DIR", Path.home() / "models")
).expanduser()
STATE_DIR = Path(
    os.environ.get("LLLM2_STATE_DIR", Path.home() / ".local/state/lllm2")
).expanduser()
ENGINE_HOME = Path(
    os.environ.get("LLLM2_ENGINE_HOME", Path.home() / ".local/share/lllm2/engines")
).expanduser()
ENGINE_ROOTS = [
    Path(p).expanduser()
    for p in os.environ.get(
        "LLLM2_ENGINE_ROOTS",
        str(ENGINE_HOME),
    ).split(os.pathsep)
    if p
]
ENGINE_PORT = int(os.environ.get("LLLM2_ENGINE_PORT", "1920"))


def idle_timeout_minutes(environ=os.environ):
    """Read the default remote idle timeout from ``LLLM2_IDLE_TIMEOUT_MINUTES``.

    The value becomes ``Settings.idle_timeout_minutes`` for settings that do
    not set the field. For ``lllm2 launch`` the order, highest first, is:

    1. The ``--idle-timeout`` flag.
    2. This variable, when it is set.
    3. The saved launch settings value.
    4. 30 minutes.

    The panel sends the idle field of its launch settings with each start.
    It fills that field from this variable, or 30 minutes, and its
    running-engine idle control changes the serving call. The variable is
    read once at import, so set it before starting the panel or the CLI.

    Args:
        environ: The environment mapping to read.

    Returns:
        Whole minutes in 0..1440. 0 disables the idle timeout. The default is 30.

    Raises:
        ValueError: The variable is neither whole minutes in range nor ``off``.
    """
    return parse_idle_timeout_minutes(
        environ.get("LLLM2_IDLE_TIMEOUT_MINUTES", "30"), "LLLM2_IDLE_TIMEOUT_MINUTES"
    )


def parse_idle_timeout_minutes(value, name):
    """Parse an idle timeout given as text.

    Args:
        value: Whole minutes, or blank, ``off``, ``never`` or ``none``.
        name: The variable or option name for the error message.

    Returns:
        Whole minutes in 0..1440. 0 disables the idle timeout.

    Raises:
        ValueError: The value is neither whole minutes in range nor ``off``.
    """
    value = value.strip().lower()
    if value in ("", "off", "never", "none"):
        return 0
    try:
        minutes = int(value)
    except ValueError:
        minutes = -1
    if not 0 <= minutes <= 1440:
        raise ValueError(
            f"{name} must be whole minutes in 0..1440, or 0 or off to disable the idle timeout."
        )
    return minutes


IDLE_TIMEOUT_MINUTES = idle_timeout_minutes()
# True when LLLM2_IDLE_TIMEOUT_MINUTES is set, so it overrides saved settings.
IDLE_TIMEOUT_FROM_ENVIRONMENT = "LLLM2_IDLE_TIMEOUT_MINUTES" in os.environ
