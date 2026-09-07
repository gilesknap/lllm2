"""Terminal entry points over the same discovery, installer and lifecycle code as the panel."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading

from .discovery import engines
from .engine import Cancelled, Engine
from .engine_install import install, prerequisite_hint
from .launch import choose_launch, installed_models
from .settings import Settings


def _print_rows(rows, json_output: bool) -> None:
    if json_output:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print("None found.")
        return
    for row in rows:
        print(row["path"])


def _serve(args) -> int:
    from .app import serve
    serve(args.host, args.port)
    return 0


def _launch(args) -> int:
    resolved = choose_launch(args.model, args.engine, args.backend, args.device)
    if not resolved.get("settings"):
        raise RuntimeError(resolved["reason"])
    settings = Settings.parse(resolved["settings"])
    engine, cancel = Engine(), threading.Event()

    def stop(*_):
        cancel.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        engine.start(settings, cancel, timeout=args.timeout)
        print(f"Ready: {engine.base}/v1 (pid {engine.process.pid})", flush=True)
        if resolved.get("reason"):
            print(resolved["reason"], flush=True)
        while not cancel.wait(1):
            if not engine.state()["running"]:
                raise RuntimeError(engine.state()["error"] or "Engine stopped.")
    except Cancelled:
        return 130
    finally:
        engine.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lllm2", description="Local LLM workbench")
    commands = parser.add_subparsers(dest="command")
    panel = commands.add_parser("panel", help="run the web panel (default)")
    panel.add_argument("--port", type=int, default=8082)
    panel.add_argument("--host", default="127.0.0.1")
    panel.set_defaults(run=_serve)

    models = commands.add_parser("models", help="list installed model checkpoints")
    models.add_argument("--json", action="store_true")
    models.set_defaults(run=lambda a: (_print_rows(installed_models(), a.json), 0)[1])

    engine = commands.add_parser("engines", help="discover or install llama.cpp engines")
    engine_commands = engine.add_subparsers(dest="engine_command", required=True)
    engine_list = engine_commands.add_parser("list", help="list installed engines")
    engine_list.add_argument("--json", action="store_true")
    engine_list.set_defaults(run=lambda a: (_print_rows(engines(), a.json), 0)[1])
    engine_install = engine_commands.add_parser("install", help="build an isolated llama.cpp engine")
    engine_install.add_argument("backend", choices=("cuda", "vulkan"))
    engine_install.add_argument("--ref", default="master", help="llama.cpp branch or tag")
    engine_install.add_argument("--name")
    engine_install.add_argument("--jobs", type=int)
    engine_install.add_argument("--cuda-architectures", default="native",
                                help="CMake CUDA architectures (default: native)")
    engine_install.add_argument("--show-prerequisites", action="store_true")

    def run_install(a):
        if a.show_prerequisites:
            print(prerequisite_hint(a.backend))
            return 0
        print(install(a.backend, name=a.name or "", ref=a.ref, jobs=a.jobs,
                      cuda_architectures=a.cuda_architectures))
        return 0
    engine_install.set_defaults(run=run_install)

    launch = commands.add_parser("launch", help="start a model server in the foreground")
    launch.add_argument("--model", default="", help="exact installed GGUF path")
    launch.add_argument("--engine", default="", help="exact llama-server path")
    launch.add_argument("--backend", choices=("CUDA", "Vulkan"), default="")
    launch.add_argument("--device", default="")
    launch.add_argument("--timeout", type=int, default=180)
    launch.set_defaults(run=_launch)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original ``python -m lllm2 --host/--port`` spelling.
    if not raw or raw[0].startswith("-"):
        raw.insert(0, "panel")
    args = parser.parse_args(raw)
    try:
        return args.run(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
