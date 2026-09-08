"""Terminal entry points over the same discovery, installer and lifecycle code as the panel."""

from __future__ import annotations

import json
import signal
import threading
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .discovery import engines
from .engine import Cancelled, Engine
from .engine_install import install, provenance
from .harness import run_harness
from .launch import choose_launch, installed_models
from .service import install_service
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


class InstallBackend(str, Enum):
    cuda = "cuda"


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
    help="Discover existing engines or download a release CUDA engine.",
    no_args_is_help=True,
)
app.add_typer(engine_app, name="engines")
service_app = typer.Typer(
    help="Manage the panel systemd user service.", no_args_is_help=True
)
app.add_typer(service_app, name="service")

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


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"lllm2 {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def workbench(
    ctx: typer.Context,
    host: Host = "127.0.0.1",
    port: Port = 8082,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Local LLM workbench: browse models, install engines and serve a model.

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


@service_app.command("install")
def install_panel_service(
    host: Host = "127.0.0.1",
    port: Port = 8082,
    start: Annotated[
        bool,
        typer.Option(
            "--start/--no-start",
            help="Start or restart the panel after enabling the service.",
        ),
    ] = True,
) -> None:
    """Install and enable the panel as a systemd user service (no sudo).

    Uses this Python installation and saves panel paths and runtime environment.
    Stop a foreground panel before starting the service. Rerun after moving
    your installation or changing environment settings.
    """
    path = install_service(host=host, port=port, start=start)
    typer.echo(f"Installed and enabled {path}")
    if start:
        typer.echo("Panel service started. Check: systemctl --user status lllm2-panel")
    typer.echo("Logs: journalctl --user -u lllm2-panel -f")


@app.command()
def models(json_output: JsonOutput = False) -> None:
    """List installed GGUF checkpoints under LLLM2_MODELS_DIR (default: ~/models)."""
    _print_rows(installed_models(), json_output)


@engine_app.command("list")
def list_engines(json_output: JsonOutput = False) -> None:
    """List llama-server builds found in the configured engine search paths."""
    rows = [dict(row, provenance=provenance(Path(row["path"]))) for row in engines()]
    if json_output:
        _print_rows(rows, True)
    elif not rows:
        print("None found.")
    else:
        for row in rows:
            record = row["provenance"]
            print(row["path"])
            if record:
                marker = (
                    " (matches running release)"
                    if record.get("matches_release")
                    else ""
                )
                print(
                    f"  ref={record.get('requested_ref', '?')} CUDA={record.get('cuda_track', '?')}; built for lllm2 {record.get('built_for_lllm2_version', record.get('lllm2_version', '?'))}; installed by lllm2 {record.get('lllm2_version', '?')}{marker}"
                )


@engine_app.command("install")
def install_engine(
    backend: Annotated[
        InstallBackend, typer.Argument(help="Release engine backend: cuda.")
    ],
    name: Annotated[str, typer.Option(help="Optional engine directory name.")] = "",
) -> None:
    """Download this lllm2 release's CUDA engine into LLLM2_ENGINE_HOME.

    Requires an NVIDIA driver, with no host compiler or CUDA toolkit.
    Existing engines are preserved; reinstalling this release is a no-op.

    Example: lllm2 engines install cuda
    """
    print(install(backend.value, name=name))


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
