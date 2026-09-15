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

    Args:
        environ: The environment mapping to read.

    Returns:
        Whole minutes in 0..1440. 0 disables the idle timeout. The default is 30.

    Raises:
        ValueError: The variable is neither whole minutes in range nor ``off``.
    """
    value = environ.get("LLLM2_IDLE_TIMEOUT_MINUTES", "30").strip().lower()
    if value in ("", "off", "never", "none"):
        return 0
    try:
        minutes = int(value)
    except ValueError:
        minutes = -1
    if not 0 <= minutes <= 1440:
        raise ValueError(
            "LLLM2_IDLE_TIMEOUT_MINUTES must be whole minutes in 0..1440, or 0 or off to disable the idle timeout."
        )
    return minutes


IDLE_TIMEOUT_MINUTES = idle_timeout_minutes()
