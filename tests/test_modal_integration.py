"""Opt-in end-to-end test against a real Modal account. It costs money.

Run it with ``LLLM2_MODAL_INTEGRATION=1`` and Modal credentials configured, for
example through ``MODAL_CONFIG_PATH``. CI never sets the variable, so the test
is skipped there. ``LLLM2_MODAL_GPU`` (default ``T4``) and
``LLLM2_MODAL_MODEL`` (a catalogue id, default ``qwen3-8b``) choose what runs.
The model stays in the Volume afterwards, so a repeat run skips the download.
"""

import json
import os
import threading
import time
from pathlib import Path

import pytest

import lllm2
from fake_remote import free_port
from lllm2 import config
from lllm2.remote import RemoteEngine, catalogue_source, remote_provider
from lllm2.settings import Settings

pytestmark = pytest.mark.skipif(
    os.environ.get("LLLM2_MODAL_INTEGRATION") != "1",
    reason="Set LLLM2_MODAL_INTEGRATION=1 to run against a real Modal account.",
)


def test_download_serve_stream_and_stop(tmp_path, monkeypatch):
    pytest.importorskip("modal")
    gpu = os.environ.get("LLLM2_MODAL_GPU", "T4")
    model_id = os.environ.get("LLLM2_MODAL_MODEL", "qwen3-8b")
    catalogue = json.loads(Path(lllm2.__file__).with_name("models.json").read_text())
    entry = next(e for e in catalogue if e["id"] == model_id)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    source = catalogue_source(entry)
    provider = remote_provider("modal")
    engine = RemoteEngine(
        provider,
        gpu,
        idle_timeout=0,
        port=free_port(),
        poll_interval=1.0,
        records=tmp_path / "remote-calls.json",
        sources=lambda _path: source,
    )
    settings = Settings(model=str(config.MODELS_DIR / source.name), context=4096)
    started = time.monotonic()
    arrivals = []
    try:
        engine.start(settings, threading.Event(), timeout=900)
        print(f"Ready after {time.monotonic() - started:.0f} s on {engine.hardware()}")
        final = engine.stream_completion(
            {"prompt": "Count from one to thirty:", "n_predict": 64, "stream": True},
            threading.Event(),
            180,
            lambda event: arrivals.append(time.monotonic()),
        )
        spread = arrivals[-1] - arrivals[0]
        print(f"{len(arrivals)} events over {spread:.2f} s: {final.get('timings')}")
        assert final["stop"] is True
        # Token events arrive one by one rather than in one buffered block.
        assert len(arrivals) > 10 and spread > 0.05
    finally:
        print("\n".join(engine.logs(40)))
        engine.stop()
    deadline = time.monotonic() + 60
    while provider.calls() and time.monotonic() < deadline:
        time.sleep(2)
    assert provider.calls() == []
