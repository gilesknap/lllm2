"""Choose the engine that serves a set of launch settings.

A local backend (``CUDA`` or ``Vulkan``) runs ``LocalEngine`` on this machine.
Any other backend names a remote provider registered with
``remote.register_provider``, and runs ``RemoteEngine`` with that provider.
Callers use ``create_engine`` and ``engine_serves`` so that they need no
provider names of their own.
"""

from collections.abc import Callable, Iterable

from .engine import Engine, LocalEngine
from .remote import RemoteEngine, remote_provider
from .settings import Settings


def engine_kind(s: Settings) -> str:
    """Return the key that groups settings served by the same engine.

    Args:
        s: Launch settings.

    Returns:
        ``"local"`` for a local backend, otherwise the provider name.
    """
    return s.backend if s.remote else "local"


def idle_timeout_seconds(s: Settings) -> int | None:
    """Convert the settings' idle timeout to seconds.

    Args:
        s: Launch settings.

    Returns:
        Seconds, or None when the idle timeout is disabled.
    """
    return s.idle_timeout_minutes * 60 if s.idle_timeout_minutes else None


def create_engine(
    s: Settings | None = None,
    *,
    catalogue: Iterable[dict] | Callable[[], Iterable[dict]] | None = None,
    **remote_options,
) -> Engine:
    """Create the engine that serves settings.

    Args:
        s: Launch settings, or None for the local engine.
        catalogue: Catalogue entries, or a callable that returns them, so a
            remote engine downloads catalogue models from their repositories.
            None uses the bundled catalogue.
        **remote_options: Extra ``RemoteEngine`` keyword arguments, such as
            ``port`` or ``records``. A local engine ignores them.

    Returns:
        A ``LocalEngine`` for a local backend, otherwise a ``RemoteEngine`` for
        the backend's provider and GPU type.

    Raises:
        ValueError: The backend names no registered provider.
        RuntimeError: The provider's client library is missing.
    """
    if s is None or not s.remote:
        return LocalEngine()
    options = {"idle_timeout": idle_timeout_seconds(s)} | remote_options
    return RemoteEngine(
        remote_provider(s.backend), s.gpu_type, catalogue=catalogue, **options
    )


def engine_serves(engine: Engine, s: Settings) -> bool:
    """Return whether an existing engine can serve settings.

    Args:
        engine: An engine from ``create_engine``.
        s: Launch settings.

    Returns:
        True when the engine runs the settings' backend. A remote engine serves
        every GPU type of its provider.
    """
    if isinstance(engine, RemoteEngine):
        return s.remote and engine.provider.name == s.backend
    return isinstance(engine, LocalEngine) and not s.remote
