"""Regression checks for issue triage; no workstation state or GPU required."""

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lllm2.engine import Engine
from lllm2.launch import choose_launch


def test_engine_output_reaches_terminal_and_panel():
    engine = Engine()
    process = SimpleNamespace(stdout=io.StringIO("loading model\nready\n"))
    output = io.StringIO()
    with contextlib.redirect_stderr(output):
        engine._logs(process)
    assert list(engine.lines) == ["loading model", "ready"]
    assert output.getvalue() == "[llama-server] loading model\n[llama-server] ready\n"
    assert process.stdout.closed


def test_closed_terminal_does_not_stop_engine_pipe_drain():
    engine = Engine()
    process = SimpleNamespace(stdout=io.StringIO("first\nsecond\n"))
    with patch("builtins.print", side_effect=BrokenPipeError):
        engine._logs(process)
    assert list(engine.lines) == ["first", "second"]
    assert process.stdout.closed


def test_engine_search_roots_are_independent_and_overridable():
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLLM2_ENGINE_")}
    command = [
        sys.executable,
        "-c",
        "import json; from lllm2.config import ENGINE_ROOTS; print(json.dumps([str(p) for p in ENGINE_ROOTS]))",
    ]
    assert json.loads(subprocess.check_output(command, env=env)) == [
        str(Path.home() / ".local/share/lllm2/engines")
    ]
    env["LLLM2_ENGINE_ROOTS"] = "/custom/one:/custom/two"
    assert json.loads(subprocess.check_output(command, env=env)) == [
        "/custom/one",
        "/custom/two",
    ]


def test_recommendation_prefers_measured_engine_and_selects_cuda_device():
    model = {
        "path": "/model.gguf",
        "name": "Test",
        "catalog_id": "test",
        "identity_verified": True,
    }
    builds = [
        {"path": "/vulkan", "devices": ["Vulkan0"], "sha256": "other"},
        {"path": "/a-new-cuda", "devices": ["CUDA0"], "sha256": "new"},
        {"path": "/z-measured", "devices": ["CUDA1"], "sha256": "measured"},
    ]
    profile = {"id": "profile", "backend": "CUDA", "engine": {"sha256": "measured"}}
    with (
        patch("lllm2.launch.installed_models", return_value=[model]),
        patch("lllm2.launch.engines", return_value=builds),
        patch("lllm2.launch.hardware", return_value={"gpus": [{}]}),
        patch("lllm2.launch.PROFILES", [profile]),
        patch(
            "lllm2.launch.starting_defaults",
            side_effect=lambda s: {"settings": s.dict(), "source": "test"},
        ),
        patch("lllm2.launch.launch_args"),
    ):
        catalogue = [
            {"id": "test", "recommendation": {"profile": "profile", "rank": 1}}
        ]
        result = choose_launch(model_path="/model.gguf", catalogue=catalogue)
        assert result["settings"]["engine"] == "/z-measured"
        assert result["settings"]["device"] == "CUDA1"
        builds.pop()
        result = choose_launch(model_path="/model.gguf", catalogue=catalogue)
        assert result["settings"]["engine"] == "/a-new-cuda"
        assert result["settings"]["backend"] == "CUDA"
