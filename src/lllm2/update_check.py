"""Tell the panel when a newer lllm2 release is published."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request

from . import __version__, config
from .engine_release import LLAMA_CPP_REF, RELEASE_REPOSITORY
from .tls import download_context

CHECK_INTERVAL = 24 * 3600
# A failed check retries sooner, but never on every page load.
RETRY_INTERVAL = 3600
UPGRADE_COMMAND = "uv tool upgrade lllm2"
PIP_UPGRADE_COMMAND = "pip install --upgrade lllm2"
ENGINE_COMMAND = "lllm2 engines install cuda"


def version_key(text: str) -> tuple[tuple[int, ...], bool] | None:
    """Order release and development versions, or None when unreadable.

    setuptools-scm names a build after a tag ``0.3.1.dev4+g1234abc``: the
    next release in progress. It sorts after ``0.3.0`` and before ``0.3.1``.
    Pre-releases sort the same way, and a local ``+`` suffix alone is ignored.
    """
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)(.*)", text.strip())
    if not match:
        return None
    release = [int(part) for part in match[1].split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    return tuple(release), match[2] == "" or match[2].startswith("+")


def is_newer(latest: str, current: str) -> bool:
    """True only when ``latest`` is a final release after ``current``."""
    latest_key, current_key = version_key(latest), version_key(current)
    if latest_key is None or current_key is None or not latest_key[1]:
        return False
    return latest_key > current_key


def engine_ref(assets: list[dict]) -> str | None:
    """Read the llama.cpp pin from a release's engine asset names."""
    for item in assets:
        match = re.match(r"lllm2-engine-(.+?)-cuda", item.get("name", ""))
        if match:
            return match[1]
    return None


def fetch_latest() -> dict:
    """Return the latest published release's version, page and llama.cpp pin."""
    url = f"https://api.github.com/repos/{RELEASE_REPOSITORY}/releases/latest"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "lllm2", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=10, context=download_context()
        ) as response:
            release = json.load(response)
    except urllib.error.HTTPError as error:
        error.close()
        raise
    if release.get("draft") or release.get("prerelease"):
        raise ValueError("The latest release is not published.")
    return {
        "version": str(release["tag_name"]).removeprefix("v"),
        "url": str(release["html_url"]),
        "llama_cpp": engine_ref(release.get("assets") or []),
    }


class UpdateCheck:
    """Cache the latest release in the store and refresh it in the background.

    ``notice`` never waits for the network: it answers from the cache and
    starts a daily refresh thread when the cache is stale. Failures are silent.
    """

    def __init__(
        self,
        store,
        *,
        enabled=None,
        fetch=fetch_latest,
        clock=time.time,
        version=__version__,
    ) -> None:
        self.store = store
        self.enabled = config.UPDATE_CHECK if enabled is None else enabled
        self.fetch = fetch
        self.clock = clock
        self.version = version
        self.lock = threading.Lock()
        self.running = False
        self.retry_at = 0.0
        self.thread: threading.Thread | None = None

    def notice(self) -> dict | None:
        """Describe a newer release for the panel, or None."""
        if not self.enabled:
            return None
        cached = self.store.get("update", "latest")
        now = self.clock()
        with self.lock:
            stale = not cached or now - cached.get("checked_at", 0) >= CHECK_INTERVAL
            start = stale and not self.running and now >= self.retry_at
            if start:
                self.running = True
        if start:
            self.thread = threading.Thread(target=self.refresh, daemon=True)
            self.thread.start()
        return self.describe(cached)

    def refresh(self) -> None:
        try:
            release = self.fetch()
            self.store.put("update", "latest", {**release, "checked_at": self.clock()})
        except Exception:
            # Offline workstations and API rate limits must not reach the panel.
            with self.lock:
                self.retry_at = self.clock() + RETRY_INTERVAL
        finally:
            with self.lock:
                self.running = False

    def describe(self, cached: dict | None) -> dict | None:
        if not cached or not is_newer(str(cached.get("version", "")), self.version):
            return None
        llama_cpp = cached.get("llama_cpp")
        return {
            "version": cached["version"],
            "url": cached.get("url", ""),
            "llama_cpp": llama_cpp,
            "upgrade_command": UPGRADE_COMMAND,
            "pip_upgrade_command": PIP_UPGRADE_COMMAND,
            # Engines match their pin exactly, so a new pin needs a new download.
            "engine_command": ENGINE_COMMAND if llama_cpp != LLAMA_CPP_REF else None,
        }
