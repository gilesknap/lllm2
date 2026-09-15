"""The panel drives a launch and a measurement on a fake engine, with no GPU.

The fake engine takes the local engine's place in the panel's engine table,
so the panel selects it for local settings through ``App.engine_for``. Local
hardware, binary and checkpoint probes fail the test if anything calls them.
"""

import contextlib
import threading
import time
from unittest.mock import patch

import pytest

from fake_engine import BATCH_LOG, LOAD_FAILURE_LOG, SAMPLER_LOG, FakeEngine
from lllm2 import config
from lllm2.app import App
from lllm2.bench import Bench
from lllm2.catalogue import Catalogue
from lllm2.remote import StoreDownloads
from lllm2.settings import Settings
from lllm2.store import Store


def eventually(check, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


@pytest.fixture
def no_local_gpu():
    def refuse(*args, **kwargs):
        raise AssertionError("A fake engine run must not probe this machine.")

    with contextlib.ExitStack() as stack:
        for name in (
            "lllm2.engine.hardware",
            "lllm2.engine.probe",
            "lllm2.engine.metadata",
            "lllm2.settings.probe",
            "lllm2.settings.metadata",
            "lllm2.settings.cache_kernel_support",
        ):
            stack.enter_context(patch(name, side_effect=refuse))
        yield


@pytest.fixture
def app(tmp_path, monkeypatch, no_local_gpu):
    """A panel App with temporary state whose local engine is a fake engine."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    panel = App.__new__(App)
    panel.store = Store()
    panel.catalogue = Catalogue(panel.store)
    panel.finder = None
    panel.engines = {"local": FakeEngine()}
    panel.engine_lock = threading.Lock()
    panel.bench = Bench(panel.engines["local"], panel.store)
    panel.token = "test"
    panel.start_requests = {}
    panel.store_downloads = StoreDownloads()
    yield panel
    panel.bench.cancel.set()
    for engine in panel.engines.values():
        engine.stop()
    panel.store.db.close()


def settings():
    model = str(config.MODELS_DIR / "example" / "model.gguf")
    return Settings.parse({"model": model, "engine": "/opt/fake/llama-server"}).dict()


def benchmark(app, **values):
    request = {
        "settings": settings(),
        "mode": "baseline",
        "sweep_prompts": False,
        "search_context": False,
        "prompt_tokens": 128,
        "output_tokens": 16,
        "timeout": 60,
    }
    app.action("/api/benchmark", request | values)
    assert eventually(lambda: not app.bench.snapshot()["active"])
    snapshot = app.bench.snapshot()
    assert snapshot["status"] == "complete", snapshot
    return app.store.get("result", snapshot["result_id"])


def test_fake_engine_drives_a_panel_launch_and_measurement(app):
    engine = app.engines["local"]
    s = settings()
    assert app.engine_for(Settings.parse(s)) is engine

    app.action("/api/start", {"settings": s, "request_id": "launch"})
    assert eventually(lambda: app.bench.snapshot()["status"] == "serving")
    state = app.engine.state()
    assert app.engine is engine
    assert state["ready"] and state["pid"] is None and state["settings"] == s
    assert state["argv"][:3] == ["/opt/fake/llama-server", "--model", s["model"]]
    assert engine.request("/health") == {"status": "ok"}

    result = benchmark(app, replace_running=True, expected_pid=state["pid"])
    assert result["status"] == "complete", result.get("error")
    assert result["hardware"]["source"] == "fake-engine"
    (sample,) = result["samples"]
    assert (sample["input_tokens"], sample["output_tokens"]) == (128, 16)
    assert sample["decode_tok_s"] == 100.0
    assert sample["peak_total_gpu_used_mib"] == 4096
    assert "/completion" in engine.requests
    # Argv, the launch environment and parsed engine logs reach the result.
    port = str(engine.port)
    assert result["argv"] == engine.launch_args(Settings.parse(s))
    assert result["argv"][result["argv"].index("--port") + 1] == port
    execution = result["execution_settings"]
    assert execution["child_environment"] == {
        "GGML_CUDA_GRAPH_OPT": None,
        "GGML_CUDA_DISABLE_GRAPHS": None,
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert execution["diagnostics"] == [SAMPLER_LOG]
    assert BATCH_LOG.endswith(str(sample["batch_settings"]["batch_size"]["effective"]))
    # The measurement restarts the engine per sample and stops it afterwards.
    assert not engine.alive()
    assert not app.engine.state()["running"]


def test_failed_fake_launch_records_logs_and_environment(app):
    engine = app.engines["local"]
    engine.fail_load = True
    result = benchmark(app)
    assert result["status"] == "failed"
    assert "exited during load" in result["error"]
    assert result["logs"][0].startswith("Launching: /opt/fake/llama-server ")
    assert LOAD_FAILURE_LOG in result["logs"]
    assert (
        result["execution_settings"]["child_environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    )
    assert not engine.alive()
