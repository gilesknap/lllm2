"""Terminal entry points over the same discovery, installer and lifecycle code as the panel."""

from __future__ import annotations

import contextlib
import difflib
import json
import re
import signal
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from . import __version__, config
from .backends import create_engine
from .catalogue import local_paths
from .defaults import starting_defaults
from .discovery import CATALOG, engines
from .engine import Cancelled
from .engine_install import install, provenance
from .gpu_tables import GPU_TABLES, gpu_type, gpu_types, pricing_caveat
from .harness import run_harness
from .launch import choose_launch, installed_models
from .remote import (
    PROVIDERS,
    CallRecords,
    ProbeCache,
    catalogue_source,
    companion_names,
    describe_calls,
    model_users,
    remote_provider,
    stored_entries,
)
from .service import install_service
from .settings import LOCAL_BACKENDS, Settings, default_key
from .store import Store


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


def _saved_launch_settings(settings: Settings) -> tuple[Settings, bool]:
    """Load the saved default for the selected model/backend, when present."""
    # Read-only: the panel may be mid-experiment, so leave running results alone.
    store = Store(recover_running=False)
    try:
        saved = store.get("default", default_key(settings))
    finally:
        store.db.close()
    if not saved:
        return settings, False
    resolved = Settings.parse(saved)
    # Discovery owns these volatile paths; saved tuning owns the remaining fields.
    resolved.engine, resolved.device = settings.engine, settings.device
    return resolved, True


def _catalogue() -> list[dict]:
    """Return the panel's catalogue, or the bundled one before first panel use."""
    store = Store(recover_running=False)
    try:
        return store.list("catalogue") or CATALOG
    finally:
        store.db.close()


def _remote_model_path(model: str, catalogue: list[dict]) -> str:
    """Resolve a remote launch's model option to a managed model path.

    Args:
        model: A catalogue id or name, a model path, or empty for the
            top-ranked catalogue recommendation.
        catalogue: The catalogue entries.

    Returns:
        The model path under the managed model directory.

    Raises:
        ValueError: No model is given and the catalogue has no recommendation.
    """
    if not model:
        ranked = sorted(
            (e for e in catalogue if e.get("recommendation")),
            key=lambda e: e["recommendation"].get("rank", 999),
        )
        if not ranked:
            raise ValueError("Choose a model with --model.")
        return str(local_paths(ranked[0])[0])
    entry = next((e for e in catalogue if model in (e.get("id"), e["name"])), None)
    if entry is not None:
        return str(local_paths(entry)[0])
    return str(Path(model).expanduser().resolve())


def _launch(
    model: str,
    engine_path: str,
    backend: str,
    device: str,
    timeout: int,
    gpu: str = "",
    idle_timeout: int | None = None,
) -> int:
    notes = []
    if backend and backend not in LOCAL_BACKENDS:
        if engine_path or device:
            raise ValueError(
                "--engine and --device apply only to local backends; a remote backend runs the provider's engine release."
            )
        if not gpu:
            names = ", ".join(g.name for g in gpu_types(backend))
            raise ValueError(f"Choose a {backend} GPU type with --gpu: {names}.")
        catalogue = _catalogue()
        values: dict[str, object] = {
            "model": _remote_model_path(model, catalogue),
            "backend": backend,
            "gpu_type": gpu,
        }
        if idle_timeout is not None:
            values["idle_timeout_minutes"] = idle_timeout
        selection = Settings.parse(values)
        engine = create_engine(selection, catalogue=catalogue)
        settings, saved = _saved_launch_settings(selection)
        if not saved:
            resolved = starting_defaults(selection, **engine.defaults_inputs(selection))
            settings = Settings.parse(resolved["settings"])
            notes.extend(resolved.get("notes", []))
        # Idle timeout order: the flag, then LLLM2_IDLE_TIMEOUT_MINUTES, then
        # the saved launch settings, then the default (see config).
        if idle_timeout is not None:
            settings.idle_timeout_minutes = idle_timeout
        elif config.IDLE_TIMEOUT_FROM_ENVIRONMENT:
            settings.idle_timeout_minutes = config.IDLE_TIMEOUT_MINUTES
    else:
        if gpu or idle_timeout is not None:
            raise ValueError(
                "--gpu and --idle-timeout apply only to a remote backend such as modal."
            )
        resolved = choose_launch(model, engine_path, backend, device)
        if not resolved.get("settings"):
            raise RuntimeError(resolved["reason"])
        settings = Settings.parse(resolved["settings"])
        settings, saved = _saved_launch_settings(settings)
        if resolved.get("reason"):
            notes.append(resolved["reason"])
        engine = create_engine(settings)
    cancel = threading.Event()

    def stop(*_):
        cancel.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        engine.start(settings, cancel, timeout=timeout)
        state = engine.state()
        pid = state["pid"]
        print(f"Ready: {engine.base}/v1" + (f" (pid {pid})" if pid else ""), flush=True)
        if settings.remote:
            _print_remote_launch(settings, state)
        if saved:
            print("Using saved settings from the workbench database.", flush=True)
        for note in notes:
            print(note, flush=True)
        while not cancel.wait(1):
            state = engine.state()
            if not state["running"]:
                if state.get("phase") == "idle stopped":
                    print("Stopped the remote engine after the idle timeout.")
                    return 0
                raise RuntimeError(state["error"] or "Engine stopped.")
    except Cancelled:
        return 130
    finally:
        engine.stop()
    return 0


def _print_remote_launch(settings: Settings, state: dict) -> None:
    price = state.get("usd_per_hour")
    print(
        f"Serving on {settings.backend} {settings.gpu_type}, call {state.get('call_id')}."
        + (f" Estimated cost: ${price:.2f} per hour." if price is not None else ""),
        flush=True,
    )
    print(pricing_caveat(settings.backend), flush=True)
    if settings.idle_timeout_minutes:
        print(
            f"The remote engine stops after {settings.idle_timeout_minutes} minutes without requests.",
            flush=True,
        )
    else:
        print(
            "Idle timeout disabled: the GPU bills until you press Ctrl-C.", flush=True
        )


@contextlib.contextmanager
def _transfer_progress():
    """Show phase labels and a byte progress bar on stderr.

    A terminal gets a live bar. Other output gets a line every 10 seconds and
    at completion, so stdout stays free for results.

    Yields:
        A callable ``report(label, completed, total, transfer)``. A new label
        prints once. ``transfer`` True shows ``completed`` of ``total`` bytes
        (``total`` None when unknown); False hides the bar.
    """
    console = Console(stderr=True)
    with Progress(
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not console.is_terminal,
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as display:
        task = display.add_task("download", total=None, visible=False)
        phase = ""
        last_update = time.monotonic()

        def report(label: str, completed: int, total: int | None, transfer: bool):
            nonlocal phase, last_update
            if label != phase:
                phase = label
                console.print(label, markup=False)
                display.update(task, visible=transfer)
                display.refresh()
            if not transfer:
                return
            display.update(
                task,
                completed=completed,
                total=total,
                refresh=total is not None and completed == total,
            )
            now = time.monotonic()
            if not console.is_terminal and (
                now - last_update >= 10 or (total is not None and completed == total)
            ):
                amount = f"Downloaded {completed / 1_000_000:.1f} MB"
                if total:
                    amount += f" / {total / 1_000_000:.1f} MB ({completed / total:.0%})"
                console.print(amount, markup=False)
                last_update = now

        yield report


def _catalogue_choice(model: str, catalogue: list[dict]) -> dict | None:
    """Find a catalogue entry by id or directory name.

    Args:
        model: The catalogue id or name.
        catalogue: The catalogue entries.

    Returns:
        The entry, or None when no entry matches.
    """
    return next((e for e in catalogue if model in (e.get("id"), e.get("name"))), None)


def _close_matches(model: str, choices) -> str:
    """Return a sentence naming the choices that resemble a mistyped name."""
    matches = difflib.get_close_matches(model, sorted(set(choices)), n=5, cutoff=0.5)
    return f" Close matches: {', '.join(matches)}." if matches else ""


def _gpu_choice(provider: str, gpu: str):
    """Return a provider's GPU table entry, naming the valid types on a miss.

    Args:
        provider: The provider name.
        gpu: The GPU type string.

    Returns:
        The ``GpuType`` entry.

    Raises:
        ValueError: The provider's table has no such GPU type.
    """
    try:
        return gpu_type(provider, gpu)
    except ValueError:
        names = ", ".join(g.name for g in gpu_types(provider))
        raise ValueError(
            f"Unknown {provider} GPU type: {gpu}. Choose one of: {names}."
        ) from None


def _launch_backend(value: str) -> str:
    choices = [*LOCAL_BACKENDS, *PROVIDERS]
    if value and value not in choices:
        raise typer.BadParameter(f"Choose one of: {', '.join(choices)}.")
    return value


class InstallBackend(str, Enum):
    cuda = "cuda"


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
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Try the CUDA 12 bundle if driver checks fail. GPU execution may fail.",
        ),
    ] = False,
) -> None:
    """Download this lllm2 release's CUDA engine into LLLM2_ENGINE_HOME.

    Requires an NVIDIA driver, with no host compiler or CUDA toolkit.
    Existing engines are preserved; reinstalling this release is a no-op.

    Example: lllm2 engines install cuda
    """
    with _transfer_progress() as report:
        binary = install(
            backend.value,
            name=name,
            force=force,
            progress=lambda label, completed, total: report(
                label, completed, total, label == "Downloading engine"
            ),
        )
    print(binary)


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
        str,
        typer.Option(
            callback=_launch_backend,
            help="GPU backend (CUDA or Vulkan), or a remote provider such as modal; omit for automatic local selection.",
        ),
    ] = "",
    device: Annotated[
        str,
        typer.Option(help="Engine device identifier; omit for automatic selection."),
    ] = "",
    timeout: Annotated[
        int, typer.Option(min=1, help="Seconds to wait for the model server to start.")
    ] = 180,
    gpu: Annotated[
        str,
        typer.Option(
            help="Remote GPU type, such as T4 or L40S; required with a remote backend. "
            + " ".join(pricing_caveat(p) for p in GPU_TABLES)
        ),
    ] = "",
    idle_timeout: Annotated[
        str | None,
        typer.Option(
            help="Minutes without requests before a remote engine stops; 0 or off disables it. Default: LLLM2_IDLE_TIMEOUT_MINUTES, else saved settings, else 30.",
        ),
    ] = None,
) -> None:
    """Start a model server in the foreground and print its API URL.

    Use saved settings and available model/engine combinations when options
    are omitted. Press Ctrl-C to stop the server and release the GPU. With a
    remote backend, --model also accepts a catalogue id or name, and the model
    is served on the same local port.

    Example: lllm2 launch --backend CUDA --timeout 300;
    lllm2 launch --backend modal --gpu L40S --model qwen3.8-27b
    """
    minutes = None
    if idle_timeout is not None:
        try:
            minutes = config.parse_idle_timeout_minutes(idle_timeout, "--idle-timeout")
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--idle-timeout") from e
    raise typer.Exit(_launch(model, engine, backend, device, timeout, gpu, minutes))


def _elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "unknown time"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _engine_build(engine: dict) -> str:
    """Describe which engine build a probe found, in one line.

    The compiler version alone reads like a CUDA version, so name every part.
    The llama.cpp release and CUDA track come from the engine record the
    provider probed, not from parsing help or version text.

    Args:
        engine: A probe's engine record.

    Returns:
        A comma-separated description of the build.
    """
    path = engine.get("path") or ""
    ref = engine.get("requested_ref") or _match(r"llama-(.+?)-cuda", path)
    track = engine.get("cuda_track") or _match(r"-cuda([\d.]+)", path)
    lines = (engine.get("version") or "").strip().splitlines()
    last = lines[-1].strip() if lines else ""
    compiler = _match(r"(?i)^built with (.+)", last)
    parts = [
        f"llama.cpp {ref}" if ref else "",
        f"CUDA track {track}" if track else "",
        f"compiler {compiler}" if compiler else f"version {last}" if last else "",
    ]
    return ", ".join(p for p in parts if p) or "unknown"


def _match(pattern: str, text: str) -> str | None:
    found = re.search(pattern, text)
    return found[1] if found else None


def _ownership(row: dict, command: str) -> str:
    owner = row["owner"] or {}
    if row["status"] == "owned":
        return "owned by this process"
    if row["status"] == "active":
        where = (
            f" pid {owner.get('pid')} on {owner.get('host')}"
            if owner
            else " on another machine or in another container"
        )
        return f"in use by lllm2{where}; its heartbeat is fresh"
    return f"orphan: no live lllm2 session owns it; stop it with `{command} stop {row['id']}`"


def provider_app(name: str, label: str) -> typer.Typer:
    """Build the command group that manages one remote provider.

    Args:
        name: The provider registry name, which is also the command name.
        label: The provider name for messages.

    Returns:
        A Typer group with setup, probe, list, stop, models and remove commands.
    """
    command = f"lllm2 {name}"
    caveat = pricing_caveat(name)
    group = typer.Typer(
        help=f"Set up {label}, probe a GPU type, and list or stop lllm2 serve calls and stored models there.",
        no_args_is_help=True,
    )

    @group.command("setup")
    def setup() -> None:
        """Verify credentials and deploy the lllm2 app unless it is current.

        Rerunning changes nothing when this lllm2 version is already deployed.
        """
        deployment = remote_provider(name).setup()
        message = f"{label} credentials work."
        if deployment.version and deployment.deployed:
            message += f" Deployed lllm2 app version {deployment.version}."
        elif deployment.version:
            message += f" lllm2 app version {deployment.version} is already deployed; nothing changed."
        typer.echo(message)

    @group.command(
        "probe",
        help=f"Start a short, billed {label} GPU container on a GPU type and print "
        "the GPU name, VRAM, engine devices and engine sha256. Later launches and "
        f"validation use the saved result. {caveat}",
    )
    def probe(
        # A default value, not Annotated: postponed annotations cannot see
        # this function's enclosing variables.
        gpu: str = typer.Option(
            ...,
            help=f"The {label} GPU type: "
            + ", ".join(g.name for g in gpu_types(name))
            + ".",
        ),
        json_output: Annotated[
            bool, typer.Option("--json", help="Print the probe record as JSON.")
        ] = False,
    ) -> None:
        """Run the probe function on one GPU type."""
        entry = _gpu_choice(name, gpu)
        provider = remote_provider(name)
        result = provider.probe(entry.name)
        engine = result.engine
        saved = config.STATE_DIR / "remote-probes.json"
        try:
            ProbeCache(saved).put(provider.name, entry.name, result)
        except OSError as error:
            typer.echo(f"Could not save the probe to {saved}: {error}", err=True)
        if json_output:
            record = {
                "gpu_type": entry.name,
                "name": result.name,
                "total_mib": result.total_mib,
                "engine": {k: v for k, v in engine.items() if k != "help"},
            }
            print(json.dumps(record, indent=2))
            return
        typer.echo(f"GPU type: {entry.name}")
        typer.echo(f"GPU: {result.name}")
        typer.echo(
            f"VRAM: {result.total_mib} MiB ({result.total_mib / 1024:.1f} GiB); "
            f"table figure {entry.vram_gb} GB"
        )
        typer.echo(
            f"Engine devices: {', '.join(engine.get('devices') or []) or 'none'}"
        )
        typer.echo(f"Engine build: {_engine_build(engine)}")
        typer.echo(f"Engine sha256: {engine.get('sha256') or 'unknown'}")
        if engine.get("error"):
            typer.echo(f"Engine error: {engine['error']}", err=True)
        typer.echo(f"Estimated cost: ${entry.usd_per_hour:.2f} per hour. {caveat}")

    @group.command(
        "list",
        help="List running lllm2 serve calls with owner, elapsed time and cost.\n\n"
        "A call whose owning session still heartbeats is in use, wherever that "
        "session runs. A call with no live owner is an orphan that bills until "
        f"you stop it. {caveat}",
    )
    def list_calls(json_output: JsonOutput = False) -> None:
        """List running lllm2 serve calls."""
        provider = remote_provider(name)
        rows = describe_calls(provider)
        if json_output:
            print(json.dumps(rows, indent=2))
            return
        if not rows:
            typer.echo(f"No lllm2 serve calls are running on {label}.")
            return
        for row in rows:
            cost = row["estimated_cost_usd"]
            typer.echo(
                f"{row['id']}  {row['gpu'] or 'unknown GPU'}  "
                f"{_elapsed(row['elapsed_seconds'])}  "
                + (f"~${cost:.2f}" if cost is not None else "cost unknown")
                + f"  {_ownership(row, command)}"
                + (f"  {row['model']}" if row["model"] else "")
            )
        typer.echo(pricing_caveat(provider.name))

    @group.command("stop")
    def stop_calls(
        call_id: Annotated[
            str, typer.Argument(help="The call ID that `list` prints.")
        ] = "",
        all_calls: Annotated[
            bool,
            typer.Option(
                "--all",
                help="Stop every orphaned call. Calls a live session still "
                "serves are left alone, even with --force.",
            ),
        ] = False,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                help="Stop one call by ID even though a live lllm2 session "
                "serves it. That session loses its model and its work.",
            ),
        ] = False,
    ) -> None:
        """Stop a serve call, or every orphaned call with --all."""
        if bool(call_id) == all_calls:
            raise typer.BadParameter("Give a call ID or --all.")
        if force and all_calls:
            raise typer.BadParameter(
                "Use --force with a call ID, so a live session's call is never "
                "stopped in bulk."
            )
        provider = remote_provider(name)
        records = CallRecords(config.STATE_DIR / "remote-calls.json")
        rows = describe_calls(provider, records)
        targets = [r for r in rows if r["id"] == call_id] if call_id else rows
        if call_id and not targets:
            raise ValueError(f"No running lllm2 call {call_id} on {label}.")
        if not targets:
            typer.echo(f"No lllm2 serve calls are running on {label}.")
        skipped = 0
        for row in targets:
            if row["status"] != "orphan" and not force:
                skipped += 1
                typer.echo(
                    f"Skipped {row['id']}: {_ownership(row, command)}. "
                    "Stop it from that session, or add --force.",
                    err=True,
                )
                continue
            if row["status"] != "orphan":
                typer.echo(
                    f"Warning: {row['id']} is {_ownership(row, command)}. "
                    "Forcing it to stop leaves that session without a model.",
                    err=True,
                )
            provider.cancel(row["id"])
            records.remove(provider.name, row["id"])
            typer.echo(f"Stopped {row['id']}")
        if skipped and call_id:
            raise typer.Exit(1)

    @group.command("models")
    def list_models(json_output: JsonOutput = False) -> None:
        """List the model files stored for serving, with sizes and catalogue ids."""
        stored = remote_provider(name).models()
        ids: dict[str, str | None] = {}
        for main, entry in stored_entries(_catalogue()).items():
            for file in (main, *companion_names(entry)):
                ids.setdefault(file, entry.get("id"))
        if json_output:
            rows = [
                {
                    "name": m.name,
                    "size_bytes": m.size_bytes,
                    "catalogue_id": ids.get(m.name),
                }
                for m in stored
            ]
            print(json.dumps(rows, indent=2))
        elif not stored:
            typer.echo(f"No models are stored on {label}.")
        else:
            for m in stored:
                catalogue_id = ids.get(m.name)
                typer.echo(
                    f"{m.name}  {m.size_bytes / 1e9:.1f} GB"
                    + (f"  {catalogue_id}" if catalogue_id else "")
                )

    @group.command("download")
    def download_model(
        model: Annotated[
            str, typer.Argument(help="The catalogue id, as in `lllm2 launch --model`.")
        ],
    ) -> None:
        """Download a catalogue model and its companion files into the store.

        The download runs inside the provider, so no weights pass through this
        machine. A stored model downloads nothing. Press Ctrl-C to cancel; the
        partial download stays in the store and a rerun resumes it.

        Example: lllm2 modal download qwen3-8b
        """
        catalogue = _catalogue()
        entry = _catalogue_choice(model, catalogue)
        if entry is None:
            raise ValueError(
                f"Unknown catalogue id: {model}."
                + _close_matches(model, (e["id"] for e in catalogue if e.get("id")))
            )
        source = catalogue_source(entry)
        provider = remote_provider(name)
        cancel = threading.Event()
        reported = False

        def progress(update) -> None:
            nonlocal reported
            reported = True
            report(
                f"Downloading {update.file} inside {label}",
                update.done_bytes,
                update.total_bytes,
                True,
            )

        def stop(*_) -> None:
            cancel.set()

        handlers = {
            sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            with _transfer_progress() as report:
                provider.ensure_model(source, progress, cancel)
        except Cancelled:
            typer.echo(
                f"Cancelled. The partial download stays on {label}; "
                f"rerun `{command} download {entry['id']}` to resume.",
                err=True,
            )
            raise typer.Exit(130) from None
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if reported:
            typer.echo(f"Stored {source.name} on {label}.")
        else:
            typer.echo(
                f"{source.name} is already stored on {label}; nothing downloaded."
            )

    @group.command("remove")
    def remove_model(
        model: Annotated[
            str,
            typer.Argument(help="A catalogue id, or the name that `models` prints."),
        ],
    ) -> None:
        """Delete a stored model with its companion files, unless a call serves it.

        Example: lllm2 modal remove qwen3-8b
        """
        catalogue = _catalogue()
        entries = stored_entries(catalogue)
        provider = remote_provider(name)
        entry = _catalogue_choice(model, catalogue)
        if entry is not None:
            stored = catalogue_source(entry).name
        else:
            stored = model
            entry = entries.get(model)
            names = [] if entry else [m.name for m in provider.models()]
            if entry is None and model not in names:
                raise ValueError(
                    f"No catalogue id or stored model named {model} on {label}."
                    + _close_matches(
                        model, [e["id"] for e in catalogue if e.get("id")] + names
                    )
                )
        users = model_users(describe_calls(provider), stored)
        if users:
            raise ValueError(
                f"Serve call {', '.join(users)} uses {stored}. Stop it with `{command} stop` first."
            )
        provider.remove_model(stored, companion_names(entry) if entry else ())
        typer.echo(f"Removed {stored} from {label}.")

    return group


app.add_typer(provider_app("modal", "Modal"), name="modal")

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
