"""Session-local configuration for coding harnesses using the running engine."""

import json
import os
import shutil
import subprocess

from .engine import Engine


def served_model():
    engine = Engine()
    try:
        models = engine.request("/v1/models", timeout=3)
        props = engine.request("/props", timeout=3)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Cannot query the model at {engine.base}. Start a model in the panel "
            "or run lllm2 launch in another terminal first."
        ) from error
    try:
        model = models["data"][0]["id"]
        # llama.cpp reports the context per slot here, not the shared pool.
        window = props["default_generation_settings"]["n_ctx"]
        slots = props["total_slots"]
        if not isinstance(model, str) or not model:
            raise ValueError("Missing model ID")
        if type(window) is not int or window < 1 or type(slots) is not int or slots < 1:
            raise ValueError("Invalid context or slot count")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            "The engine did not report a model, context window and slot count."
        ) from error
    return engine.base, model, window, slots


def claude_env(base, model, window, slots):
    """Adapted from gilesknap/lllm3090's Claude launcher."""
    return {
        "ANTHROPIC_BASE_URL": base,
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(window),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(min(32768, max(1, window // 4))),
        "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": str(max(1, slots - 1)),
    }


def codex_args(base, model, window):
    settings = {
        "model": model,
        "model_provider": "lllm2",
        "model_providers.lllm2.name": "Local llama.cpp",
        "model_providers.lllm2.base_url": base + "/v1",
        "model_providers.lllm2.wire_api": "responses",
        "model_providers.lllm2.requires_openai_auth": False,
        "model_context_window": window,
        "model_auto_compact_token_limit": max(1, window * 3 // 4),
    }
    return [
        arg
        for key, value in settings.items()
        for arg in ("-c", f"{key}={json.dumps(value)}")
    ]


def run_harness(name: str, args: list[str]) -> int:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"{name} is not on PATH; install its CLI first.")
    base, model, window, slots = served_model()
    env = dict(os.environ)
    options = []
    if name == "claude":
        env.update(claude_env(base, model, window, slots))
        env.pop("ANTHROPIC_API_KEY", None)
    elif name == "codex":
        options = codex_args(base, model, window)
    else:
        raise ValueError(f"Unknown harness: {name}")
    try:
        result = subprocess.call([executable, *options, *args], env=env)
    except KeyboardInterrupt:
        return 130
    return result if result >= 0 else 128 - result
