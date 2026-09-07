"""Terminal entry points over the same discovery, installer and lifecycle code as the panel."""

from __future__ import annotations

import json
import signal
import threading
from enum import Enum
from typing import Annotated

import typer

from .discovery import engines
from .engine import Cancelled, Engine
from .engine_install import install
from .harness import run_harness
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


def _serve(host: str, port: int) -> int:
    from .app import serve

    serve(host, port)
    return 0


def _launch(
    model: str, engine_path: str, backend: str, device: str, timeout: int
) -> int:
    resolved = choose_launch(model, engine_path, backend, device)
    if not resolved.get("settings"):
        raise RuntimeError(resolved["reason"])
    settings = Settings.parse(resolved["settings"])
    engine, cancel = Engine(), threading.Event()

    def stop(*_):
        cancel.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        engine.start(settings, cancel, timeout=timeout)
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


class BuildBackend(str, Enum):
    cuda = "cuda"
    vulkan = "vulkan"


class LaunchBackend(str, Enum):
    cuda = "CUDA"
    vulkan = "Vulkan"


app = typer.Typer(
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
    epilog="Examples: lllm2 panel --port 8082; lllm2 models --json; "
    "lllm2 engines install cuda; lllm2 launch. "
    "Use COMMAND --help for command options.",
)
engine_app = typer.Typer(
    help="Discover existing engines or build an isolated llama.cpp engine.",
    no_args_is_help=True,
)
app.add_typer(engine_app, name="engines")

Host = Annotated[
    str,
    typer.Option(
        help="Address to bind the web panel to; use 0.0.0.0 for remote access."
    ),
]
Port = Annotated[int, typer.Option(min=1, max=65535, help="Port for the web panel.")]
JsonOutput = Annotated[
    bool, typer.Option("--json", help="Print full records as JSON instead of paths.")
]


@app.callback(invoke_without_command=True)
def workbench(ctx: typer.Context, host: Host = "127.0.0.1", port: Port = 8082) -> None:
    """Local LLM workbench: browse models, build engines and serve a model.

    With no command, start the web panel. The top-level --host and --port
    options configure that default panel; explicit commands have their own options.

    Model and engine locations can be set with LLLM2_MODELS_DIR,
    LLLM2_ENGINE_HOME and LLLM2_ENGINE_ROOTS (colon-separated search paths).
    """
    if ctx.invoked_subcommand is None:
        _serve(host, port)


@app.command()
def panel(host: Host = "127.0.0.1", port: Port = 8082) -> None:
    """Start the web panel for model discovery, experiments and settings.

    Example: lllm2 panel --host 0.0.0.0 --port 8082
    """
    _serve(host, port)


@app.command()
def models(json_output: JsonOutput = False) -> None:
    """List installed GGUF checkpoints under LLLM2_MODELS_DIR (default: ~/models)."""
    _print_rows(installed_models(), json_output)


@engine_app.command("list")
def list_engines(json_output: JsonOutput = False) -> None:
    """List llama-server builds found in the configured engine search paths."""
    _print_rows(engines(), json_output)


@engine_app.command("install")
def install_engine(
    backend: Annotated[
        BuildBackend, typer.Argument(help="GPU backend to compile: cuda or vulkan.")
    ],
    ref: Annotated[
        str, typer.Option(help="llama.cpp branch or tag to clone.")
    ] = "master",
    name: Annotated[
        str, typer.Option(help="Build directory name; defaults to llama-REF-BACKEND.")
    ] = "",
    jobs: Annotated[
        int | None,
        typer.Option(min=1, help="Parallel build jobs; defaults to CPU count."),
    ] = None,
    cuda_architectures: Annotated[
        str,
        typer.Option(
            help="CUDA targets: native, all, all-major, or a quoted list such as '86;89'.",
        ),
    ] = "native",
) -> None:
    """Build llama-server from source into LLLM2_ENGINE_HOME.

    Defaults to ~/.local/share/lllm2/engines. Existing builds are never
    overwritten. System packages must be installed separately; CUDA builds
    also require an installed NVIDIA CUDA toolkit and compatible compiler.
    If build tools are missing, print prerequisite instructions and exit.

    Example: lllm2 engines install cuda --name my-cuda --jobs 8
    """
    print(
        install(
            backend.value,
            name=name,
            ref=ref,
            jobs=jobs,
            cuda_architectures=cuda_architectures,
        )
    )


@app.command()
def launch(
    model: Annotated[
        str,
        typer.Option(help="Exact installed GGUF path; omit for automatic selection."),
    ] = "",
    engine: Annotated[
        str, typer.Option(help="Exact llama-server path; omit for automatic selection.")
    ] = "",
    backend: Annotated[
        LaunchBackend | None,
        typer.Option(help="GPU backend; omit for automatic selection."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Engine device identifier; omit for automatic selection."),
    ] = "",
    timeout: Annotated[
        int, typer.Option(min=1, help="Seconds to wait for the model server to start.")
    ] = 180,
) -> None:
    """Start a model server in the foreground and print its API URL.

    Use saved settings and available model/engine combinations when options
    are omitted. Press Ctrl-C to stop the server and release the GPU.

    Example: lllm2 launch --backend CUDA --timeout 300
    """
    raise typer.Exit(
        _launch(model, engine, backend.value if backend else "", device, timeout)
    )


HARNESS_CONTEXT = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "allow_interspersed_args": False,
}


@app.command(context_settings=HARNESS_CONTEXT)
def claude(ctx: typer.Context) -> None:
    """Launch Claude Code against the running local model.

    Start a model in the panel or with lllm2 launch first. Configuration applies
    only to this session. Additional arguments go to Claude unchanged.
    Use -- --help for Claude's help. Example: lllm2 claude -p "Explain this repo"
    """
    raise typer.Exit(run_harness("claude", ctx.args))


@app.command(context_settings=HARNESS_CONTEXT)
def codex(ctx: typer.Context) -> None:
    """Launch Codex against the running local model using the Responses API.

    Start a model first; the engine must support /v1/responses. Configuration
    applies only to this session. Additional arguments go to Codex unchanged.
    Use -- --help for Codex's help. Example: lllm2 codex exec "Explain this repo"
    """
    raise typer.Exit(run_harness("codex", ctx.args))


@app.command(context_settings=HARNESS_CONTEXT)
def pi(ctx: typer.Context) -> None:
    """Launch Pi (pi.dev) against the running local model.

    Start a model first. A temporary extension configures the local provider
    for this session. Additional arguments go to Pi unchanged.
    Use -- --help for Pi's help. Example: lllm2 pi -p "Explain this repo"
    """
    raise typer.Exit(run_harness("pi", ctx.args))


def main(argv: list[str] | None = None) -> int:
    """Shared entry point for the console script and python -m lllm2."""
    try:
        app(args=argv, prog_name="lllm2")
    except SystemExit as error:
        return int(error.code or 0)
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        return 2
    return 0
