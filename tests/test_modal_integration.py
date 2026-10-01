"""Opt-in end-to-end test against a real Modal account. It costs money.

Run it with ``LLLM2_MODAL_INTEGRATION=1`` and Modal credentials configured, for
example through ``MODAL_CONFIG_PATH``, in a Modal environment of your own.
``LLLM2_MODAL_GPU`` (default ``T4``) and ``LLLM2_MODAL_MODEL`` (a catalogue id,
default ``qwen3-8b``) choose what runs. The model stays in the Volume
afterwards, so a repeat run skips the download.

The ``test`` CI jobs skip it. The ``gpu-smoke`` CI job runs it in the
``lllm2-ci`` Modal environment (``MODAL_ENVIRONMENT``) on llama.cpp bump PRs,
PRs that change Modal code and manual runs, with any engine CI built through
``LLLM2_MODAL_ENGINE_DIR``, then checks that no container is left running.
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

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("LLLM2_MODAL_INTEGRATION") != "1",
        reason="Set LLLM2_MODAL_INTEGRATION=1 to run against a real Modal account.",
    ),
    # The Modal client's multipart upload leaves its file handles to the
    # engine tarballs from LLLM2_MODAL_ENGINE_DIR for the garbage collector.
    # The pytest setting that turns warnings into errors would then fail a
    # passing test, so ignore exactly those handles.
    pytest.mark.filterwarnings(
        r"ignore:Exception ignored in. <_io\.FileIO name='[^']*/"
        r"lllm2-engine-[^']*\.tar\.gz'"
        ":pytest.PytestUnraisableExceptionWarning"
    ),
]


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
    call_id = ""
    try:
        engine.start(settings, threading.Event(), timeout=900)
        call_id = engine.call_id or ""
        assert call_id
        print(f"Ready after {time.monotonic() - started:.0f} s on {engine.hardware()}")
        # The serve container names its call, so the listing matches it.
        view = engine.remote_containers()
        assert view["complete"], view["error"]
        (row,) = [r for r in view["containers"] if r["id"] == call_id]
        assert row["container_id"] and row["function"] == "serve"
        assert row["status"] == "owned"
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
    # ``calls`` lists recorded calls, and stopping drops the record at once, so
    # also ask Modal whether the serve call itself has ended.
    deadline = time.monotonic() + 60
    while (provider.calls() or provider.poll(call_id).running) and (
        time.monotonic() < deadline
    ):
        time.sleep(2)
    assert provider.calls() == []
    assert not provider.poll(call_id).running
    # Modal lists a stopped container for a few seconds after it is told to stop.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and any(
        c.call_id == call_id for c in provider.containers() or []
    ):
        time.sleep(2)
    assert all(c.call_id != call_id for c in provider.containers() or [])
