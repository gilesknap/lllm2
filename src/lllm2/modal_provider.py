"""The Modal remote provider.

``ModalProvider`` drives the app in ``modal_app``. It deploys the app on first
use and again when the lllm2 code changes. Serve calls are tracked in a Modal
Dict: the provider records each call it spawns, and the serve container
publishes its tunnel address there. ``calls`` lists only those recorded serve
calls. ``containers`` lists every running container in the Modal environment
through the ``modal`` command line, whatever app started it, and matches
lllm2's own containers to their calls through the records each function
publishes under ``modal_app.container_key``. Log lines travel through a Modal
Queue partition per call.

While it drives a call through ``poll`` or ``ensure_model``, the provider writes
heartbeats to the Dict. A container whose owner stays silent for
``modal_app.OWNER_GRACE_SECONDS`` stops itself, so a crashed or powered-off
machine does not leave a GPU billing. Listing calls sends no heartbeat; it
reports how old each call's heartbeat is instead, which tells a call a live
session still drives from an abandoned one, whichever machine owns it.
"""

import functools
import itertools
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import PurePosixPath

from . import modal_app
from .engine import Cancelled
from .proxy import Upstream
from .remote import (
    Deployment,
    DownloadProgress,
    GpuProbe,
    ProviderError,
    RemoteCall,
    RemoteContainer,
    RemoteProvider,
    ServeStatus,
    StoredModel,
)

CREDENTIALS_MESSAGE = (
    "Modal credentials are missing or invalid. Sign in with `modal token new` "
    "(or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET), then run `lllm2 modal setup`."
)
# Seconds before a ``modal`` command line call counts as failed.
CLI_TIMEOUT = 60


def create_provider():
    """Create the Modal provider, the factory registered under ``"modal"``.

    Returns:
        A new ``ModalProvider``.

    Raises:
        ProviderError: The ``modal`` package is not installed.
    """
    try:
        import modal
    except ImportError as error:
        raise ProviderError(modal_app.INSTALL_MESSAGE) from error
    return ModalProvider(modal)


def output_pending(error, errors):
    """Return whether a timeout only means a call has not finished yet.

    Modal raises a bare timeout with no message while a call's output is not
    ready. Every other timeout ends the call: an expired output, the
    function's own timeout, and a timeout raised inside the container, which
    Modal re-raises here with its original message. Treating those as "not
    ready" would leave a caller polling a call that is already over.

    Args:
        error: The timeout the Modal client raised.
        errors: The ``modal.exception`` module.

    Returns:
        True when the caller should poll again.
    """
    return type(error) in (TimeoutError, errors.TimeoutError) and not error.args


def cli_message(text):
    """Return the ``modal`` command line's error text without its box drawing.

    Args:
        text: The command's stderr.

    Returns:
        One line, or ``CREDENTIALS_MESSAGE`` when the command could not
        authenticate.
    """
    line = " ".join(re.sub(r"[╭╮╰╯│─]+", " ", text or "").split())
    line = line.removeprefix("Error ").strip()
    lowered = line.lower()
    if "token" in lowered and "authenticate" in lowered:
        return CREDENTIALS_MESSAGE
    return line or "the modal command failed"


def start_time(value):
    """Parse a ``modal container list --json`` start time.

    Args:
        value: The ``start_time`` field, such as ``"2026-10-01 19:19:11+00:00"``.

    Returns:
        Unix seconds, or None for ``"Pending"`` or a value that does not parse.
    """
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _translated(method):
    """Turn Modal client errors into ``ProviderError`` with a user-facing message."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        errors = self.modal.exception
        try:
            return method(self, *args, **kwargs)
        except errors.AuthError as error:
            raise ProviderError(CREDENTIALS_MESSAGE) from error
        except errors.Error as error:
            raise ProviderError(f"Modal request failed: {error}") from error

    return wrapper


class ModalProvider(RemoteProvider):
    """Run llama-server on Modal GPUs.

    Attributes:
        modal: The ``modal`` module, or a replacement in tests.
        poll_interval: Seconds between download progress checks.
    """

    name = "modal"
    server_host = "0.0.0.0"
    server_port = modal_app.SERVER_PORT
    heartbeat_grace = modal_app.OWNER_GRACE_SECONDS

    def __init__(
        self,
        modal,
        *,
        poll_interval=1.0,
        deploy=None,
        heartbeat=modal_app.HEARTBEAT_SECONDS,
    ):
        """Create a provider. No network request happens until first use.

        Args:
            modal: The ``modal`` module.
            poll_interval: Seconds between download progress checks.
            deploy: A callable that deploys the app. None uses
                ``modal_app.deploy``.
            heartbeat: The minimum seconds between heartbeat writes for a call.
        """
        self.modal = modal
        self.poll_interval = poll_interval
        self.heartbeat = heartbeat
        self._beats = {}
        self._beat_count = itertools.count()
        self._deploy = deploy or modal_app.deploy
        self._deployed = False
        self._lock = threading.Lock()
        self._state = modal.Dict.from_name(modal_app.STATE_NAME, create_if_missing=True)
        self._logs = modal.Queue.from_name(
            modal_app.LOG_QUEUE_NAME, create_if_missing=True
        )
        self._volume = modal.Volume.from_name(
            modal_app.VOLUME_NAME, create_if_missing=True
        )

    @_translated
    def setup(self):
        """Check the credentials and deploy the app if it is out of date.

        Returns:
            A ``Deployment`` with the app version, and ``deployed`` False when
            the workspace already runs this version.
        """
        deployed = self._ensure_deployed(verify=True)
        return Deployment(self._deployed, deployed)

    @_translated
    def probe(self, gpu):
        result = self._call("probe", gpu, lambda function: function.remote())
        return GpuProbe(
            name=result["name"],
            total_mib=int(result["total_mib"]),
            engine=result["engine"],
        )

    @_translated
    def ensure_model(self, source, progress, cancel):
        """Make a model present in the Volume, one download per model at a time.

        A download holds a lease in the state Dict, keyed on the model's store
        name and claimed with ``put(..., skip_if_exists=True)``, so one caller
        in the workspace downloads a model at a time. Another caller follows
        that download's progress, with ``attached`` set, and returns once the
        model is stored. The owner heartbeats the lease while it drives the
        download. A follower takes over a lease whose owner stays silent for
        ``heartbeat_grace``, and downloads what is missing itself.
        """
        key = modal_app.lease_key(source.name)
        while True:
            if cancel.is_set():
                raise Cancelled()
            meta = self._stored(source)
            if meta is not None:
                return meta
            if source.repo is None or not source.files:
                raise ValueError(
                    "This model is not in the Modal Volume and has no download source. "
                    "Choose a catalogue model."
                )
            # A first deployment builds the image and can outlast the grace, so
            # it runs before this caller holds a lease it could not heartbeat.
            self._ensure_deployed()
            owner = "lease-" + secrets.token_hex(8)
            claim = {"owner": owner, "claimed": time.time()}
            if self._state.put(key, claim, skip_if_exists=True):
                try:
                    self._beat(owner)
                    # Another session may have stored the model since the
                    # check, for example during a long first deployment.
                    meta = self._stored(source)
                    if meta is not None:
                        return meta
                    return self._download(source, progress, cancel, owner)
                finally:
                    self._release(key, owner)
            lease = self._state.get(key)
            if lease is not None:
                self._follow(source, key, lease, progress, cancel)

    def _stored(self, source):
        """Return the cached metadata of a completely stored model, or None."""
        sizes = self._sizes()
        cached = self._state.get(modal_app.meta_key(source.name))
        if (
            cached is not None
            and cached.get("size") == sizes.get(source.name)
            and all(self._stored_name(source, f) in sizes for f in source.files)
        ):
            return cached["meta"]
        return None

    def _download(self, source, progress, cancel, owner):
        """Download a model under the lease ``owner`` holds.

        Returns:
            The model's metadata, cached for the next caller.
        """
        call = self._call(
            "download",
            None,
            lambda function: function.spawn(
                source.name, source.repo, list(source.files), source.revision
            ),
        )
        key = modal_app.download_key(call.object_id)
        errors = self.modal.exception
        finished = False
        try:
            # Followers find this call's progress through the lease owner.
            self._state.put(modal_app.lease_call_key(owner), call.object_id)
            while True:
                if cancel.is_set():
                    raise Cancelled()
                self._beat(call.object_id)
                self._beat(owner)
                try:
                    meta = call.get(timeout=self.poll_interval)
                    finished = True
                    break
                except (TimeoutError, errors.TimeoutError) as error:
                    if not output_pending(error, errors):
                        raise
                update = self._state.get(key)
                if update:
                    progress(
                        DownloadProgress(
                            update["file"],
                            update["done_bytes"],
                            update["total_bytes"],
                            update.get("retry"),
                        )
                    )
        finally:
            if not finished:
                # Cancellation or a failed check must not leave a download running.
                try:
                    call.cancel(terminate_containers=True)
                except Exception:
                    pass  # The container stops itself once heartbeats cease.
            self._state.pop(key, None)
            self._forget(call.object_id)
        size = self._sizes().get(source.name)
        self._state.put(modal_app.meta_key(source.name), {"size": size, "meta": meta})
        return meta

    def _follow(self, source, key, lease, progress, cancel):
        """Report another session's download of a model until its lease ends.

        Returns when the lease is released or changes hands, or after removing
        a lease whose owner has been silent for the grace. An unreadable lease
        goes at once, and so does one whose heartbeat, or claim when it has no
        heartbeat, is already older than the grace.

        Args:
            source: The model being downloaded.
            key: The model's lease key.
            lease: The lease record as last read.
            progress: A callable that receives the download's progress.
            cancel: An event that stops the following when set.

        Raises:
            Cancelled: The cancel event was set. The other download goes on.
        """
        owner = self._lease_owner(lease)
        if owner is None or (self._lease_age(owner, lease) or 0) > self.heartbeat_grace:
            self._reclaim(key, owner)
            return
        # The watch compares heartbeat values, not clocks, from here on.
        watch = modal_app.OwnerWatch(self._state, owner, self.heartbeat_grace)
        call_id = None
        reported = False
        while True:
            if call_id is None:
                found = self._state.get(modal_app.lease_call_key(owner))
                call_id = found if isinstance(found, str) else None
            update = (
                self._state.get(modal_app.download_key(call_id)) if call_id else None
            )
            if update:
                progress(
                    DownloadProgress(
                        update["file"],
                        update["done_bytes"],
                        update["total_bytes"],
                        update.get("retry"),
                        attached=True,
                    )
                )
                reported = True
            elif not reported:
                # Say at once that a download is running, before it reports.
                progress(DownloadProgress(source.files[0], 0, None, attached=True))
                reported = True
            if cancel.wait(self.poll_interval):
                raise Cancelled()
            lease = self._state.get(key)
            if lease is None or self._lease_owner(lease) != owner:
                return
            if watch.lost():
                self._reclaim(key, owner)
                return

    def _lease_age(self, owner, lease):
        """Return seconds since a lease's owner last showed it was alive, or None.

        The owner's heartbeat says so, or its claim while it has not beaten
        yet. Both carry the owner's wall clock, as ``_heartbeat_age`` explains.
        """
        age = self._heartbeat_age(owner)
        claimed = lease.get("claimed")
        if age is None and isinstance(claimed, int | float):
            age = max(0.0, time.time() - claimed)
        return age

    def _drop_lease(self, key, owner):
        """Remove a lease if ``owner`` holds it, and put back anyone else's.

        The Dict has no compare-and-delete, so a pop that takes another
        session's lease restores it with ``skip_if_exists``.

        Args:
            key: The lease key.
            owner: The owner token, or None to remove an unreadable lease.

        Returns:
            True when this call removed ``owner``'s lease.
        """
        lease = self._state.pop(key, None)
        if lease is None:
            return False
        if self._lease_owner(lease) == owner:
            return True
        self._state.put(key, lease, skip_if_exists=True)
        return False

    def _reclaim(self, key, owner):
        """Remove a lease whose owner went silent, with what the owner left.

        The owner's download call is left alone, because its container stops
        itself on the same grace. An owner judged dead by mistake therefore
        keeps its download, and the cost is a repeated download.
        """
        if self._drop_lease(key, owner) and owner is not None:
            self._forget_lease(owner)

    def _release(self, key, owner):
        """Give up the lease this caller holds, and its heartbeat."""
        self._beats.pop(owner, None)
        try:
            self._drop_lease(key, owner)
            self._forget_lease(owner)
        except Exception:
            pass  # A lease left behind goes silent, and another session reclaims it.

    def _forget_lease(self, owner):
        """Remove a lease owner's heartbeat and the name of its download call."""
        self._state.pop(modal_app.heartbeat_key(owner), None)
        self._state.pop(modal_app.lease_call_key(owner), None)

    @staticmethod
    def _lease_owner(lease):
        """Return a lease record's owner token, or None when it is unreadable."""
        owner = lease.get("owner") if isinstance(lease, dict) else None
        return owner if isinstance(owner, str) and owner else None

    @_translated
    def stored_metadata(self, name):
        cached = self._state.get(modal_app.meta_key(name))
        if cached is None or cached.get("size") != self._sizes().get(name):
            return None
        return cached["meta"]

    @_translated
    def models(self):
        return [
            StoredModel(name, size)
            for name, size in sorted(self._sizes().items())
            if name.lower().endswith(".gguf")
        ]

    @_translated
    def remove_model(self, name, companions=()):
        files = (name, *companions)
        for file in files:
            self._check_name(file)
        errors = self.modal.exception
        removed = False
        for file in files:
            for path in (file, file + ".part"):
                try:
                    self._volume.remove_file(path)
                    removed = True
                except errors.NotFoundError:
                    pass
        self._state.pop(modal_app.meta_key(name), None)
        if not removed:
            raise ValueError(f"No model named {name} in the Modal Volume.")

    def model_path(self, name):
        self._check_name(name)
        return f"{modal_app.MODEL_ROOT}/{name}"

    def file_path(self, name):
        return f"{modal_app.FILE_ROOT}/{PurePosixPath(name).name}"

    @_translated
    def spawn(self, gpu, argv, api_key, env, files):
        call = self._call(
            "serve",
            gpu,
            lambda function: function.spawn(
                list(argv), api_key, dict(env), dict(files)
            ),
        )
        try:
            self._state.put(
                modal_app.call_key(call.object_id),
                {"gpu": gpu, "started": time.time()},
            )
            self._beat(call.object_id)
        except BaseException:
            # Nothing would know of this call: no caller gets its id and
            # ``calls`` would not list it. Stop its GPU now.
            try:
                call.cancel(terminate_containers=True)
            except Exception:
                pass  # The container stops itself once heartbeats cease.
            raise
        return call.object_id

    @_translated
    def poll(self, call_id):
        running, error = self._status(call_id)
        if running:
            self._beat(call_id)
        lines = tuple(
            self._logs.get_many(modal_app.LOG_BATCH, False, partition=call_id)
        )
        upstream = None
        if running:
            tunnel = self._state.get(modal_app.tunnel_key(call_id))
            if tunnel:
                upstream = Upstream(
                    tunnel["host"], int(tunnel["port"]), bool(tunnel["tls"])
                )
        else:
            self._forget(call_id)
        return ServeStatus(running=running, upstream=upstream, logs=lines, error=error)

    @_translated
    def cancel(self, call_id):
        try:
            self.modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        except self.modal.exception.NotFoundError:
            pass
        self._forget(call_id)

    @_translated
    def calls(self):
        found = []
        for key, record in list(self._state.items()):
            if not isinstance(key, str) or not key.startswith("call:"):
                continue
            call_id = key.removeprefix("call:")
            running, _ = self._status(call_id)
            if running:
                found.append(
                    RemoteCall(
                        call_id,
                        record.get("gpu"),
                        record.get("started"),
                        self._heartbeat_age(call_id),
                    )
                )
            else:
                self._forget(call_id)
        return sorted(found, key=lambda call: call.started or 0)

    @_translated
    def containers(self):
        # Read the records before listing. A container that publishes its
        # record after this read is not in it, so the clean-up below cannot
        # drop the record of a container too new to appear in the listing.
        records = {
            key.removeprefix("container:"): value
            for key, value in list(self._state.items())
            if isinstance(key, str) and key.startswith("container:")
        }
        result = self._cli("container", "list", "--json")
        if result.returncode != 0:
            raise ProviderError(
                f"Modal could not list containers: {cli_message(result.stderr)}"
            )
        try:
            listed = json.loads(result.stdout)
            found = [
                (item["container_id"], item.get("app_name"), item.get("start_time"))
                for item in listed
            ]
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise ProviderError(
                "Unexpected output from `modal container list --json`. "
                "Update lllm2, or report the modal version in an lllm2 issue."
            ) from error
        containers = []
        for task_id, app, started in found:
            record = records.pop(task_id, None)
            record = record if isinstance(record, dict) else {}
            containers.append(
                RemoteContainer(
                    id=task_id,
                    app=app or None,
                    function=record.get("function"),
                    call_id=record.get("call_id"),
                    started=start_time(started),
                )
            )
        for task_id, record in records.items():
            self._prune(task_id, record)
        return sorted(containers, key=lambda c: (c.started is None, c.started or 0))

    def _prune(self, task_id, record):
        """Drop a container record whose container the listing no longer shows.

        A container removes its own record when its function returns, but not
        when Modal kills it. A listing can also lag, so a record that names a
        call stays until that call has ended: a live container that one
        listing missed keeps its match. The clean-up is best effort; a failed
        request leaves the record for a later listing.

        Args:
            task_id: The container id.
            record: The record the container published.
        """
        call_id = record.get("call_id") if isinstance(record, dict) else None
        try:
            if call_id and self._status(call_id)[0]:
                return
            self._state.pop(modal_app.container_key(task_id), None)
        except Exception:
            return

    @_translated
    def stop_container(self, container_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", container_id or ""):
            raise ValueError(f"Unsafe container id: {container_id}")
        record = self._state.get(modal_app.container_key(container_id))
        call_id = record.get("call_id") if isinstance(record, dict) else None
        if call_id:
            # Cancelling the call stops its container, and Modal does not retry it.
            self.cancel(call_id)
        else:
            result = self._cli("container", "stop", "--yes", "--", container_id)
            if result.returncode != 0 and "already stopped" not in result.stderr:
                raise ProviderError(
                    f"Modal could not stop container {container_id}: "
                    f"{cli_message(result.stderr)}"
                )
        self._state.pop(modal_app.container_key(container_id), None)

    def _cli(self, *args):
        """Run the ``modal`` command line of this interpreter.

        The Python client has no supported call that lists or stops
        containers, so lllm2 runs ``modal container list`` and ``modal
        container stop``. They read the same credentials, profile and
        environment as the client.

        Args:
            args: The command line arguments after ``modal``.

        Returns:
            The ``subprocess.CompletedProcess`` with text stdout and stderr.

        Raises:
            ProviderError: The command could not run or timed out.
        """
        try:
            return subprocess.run(
                [sys.executable, "-m", "modal", *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=CLI_TIMEOUT,
                env={**os.environ, "NO_COLOR": "1", "COLUMNS": "200"},
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ProviderError(f"The modal command line failed: {error}") from error

    def _heartbeat_age(self, call_id):
        """Return seconds since the owning session last heartbeated, or None.

        The owner stamps its own wall clock, so machines whose clocks differ
        widely read the age imprecisely. The grace is minutes, which is far
        more than the skew between clocks that any lllm2 machine keeps.

        Args:
            call_id: The call identifier.

        Returns:
            The age in seconds, or None when no heartbeat is recorded.
        """
        beat = self._state.get(modal_app.heartbeat_key(call_id))
        stamp = beat[0] if isinstance(beat, list | tuple) and beat else None
        return max(0.0, time.time() - stamp) if isinstance(stamp, int | float) else None

    def _ensure_deployed(self, verify=False):
        """Deploy the app unless the workspace already runs this version.

        Args:
            verify: Look the app up in the workspace as well as reading the
                recorded version, so an app deleted outside lllm2 redeploys.
                A failed lookup other than "not found" raises.

        Returns:
            True when this call deployed the app.
        """
        with self._lock:
            if self._deployed is not False and not verify:
                return False
            version = modal_app.deployment_version()
            deployed = self._state.get(modal_app.DEPLOYMENT_KEY) != version
            if not deployed and verify:
                try:
                    self.modal.Function.from_name(modal_app.APP_NAME, "probe").hydrate()
                except self.modal.exception.NotFoundError:
                    deployed = True
            if deployed:
                self._deploy()
                self._state.put(modal_app.DEPLOYMENT_KEY, version)
            self._deployed = version
            return deployed

    def _call(self, name, gpu, action):
        """Look up an app function and act on it, redeploying a missing app once."""
        for attempt in (1, 2):
            self._ensure_deployed()
            function = self.modal.Function.from_name(modal_app.APP_NAME, name)
            if gpu is not None:
                function = function.with_options(gpu=gpu)
            try:
                return action(function)
            except self.modal.exception.NotFoundError:
                if attempt == 2:
                    raise
                # The app was stopped or deleted outside lllm2.
                with self._lock:
                    self._deployed = False
                    self._state.pop(modal_app.DEPLOYMENT_KEY, None)
        raise AssertionError("unreachable")

    def _status(self, call_id):
        """Return whether a call is running and why it ended."""
        errors = self.modal.exception
        try:
            result = self.modal.FunctionCall.from_id(call_id).get(timeout=0)
        except errors.OutputExpiredError:
            return False, "the serve call ended"
        except (TimeoutError, errors.TimeoutError) as error:
            # Only a bare timeout means the output is not ready yet. The
            # function's own timeout arrives as one of these too, and calling
            # that "running" would keep the panel polling a finished call.
            if output_pending(error, errors):
                return True, None
            return False, str(error) or "the serve call ended"
        except errors.NotFoundError:
            return False, "unknown call"
        except errors.InputCancellation:
            return False, "the serve call was cancelled"
        except (
            errors.AuthError,
            errors.ConnectionError,
            errors.ServiceError,
            errors.InternalError,
            errors.ClientClosed,
        ):
            # A client or transport failure says nothing about the call, and
            # reporting it as ended would drop a call that is still billing.
            raise
        except Exception as error:
            return False, str(error) or type(error).__name__
        if isinstance(result, dict) and result.get("error"):
            return False, result["error"]
        code = result.get("exit_code") if isinstance(result, dict) else None
        return False, f"llama-server exited with status {code}"

    def _beat(self, call_id):
        """Tell the call's container that its owner is still alive."""
        now = time.monotonic()
        last = self._beats.get(call_id)
        if last is not None and now - last < self.heartbeat:
            return
        self._state.put(
            modal_app.heartbeat_key(call_id), (time.time(), next(self._beat_count))
        )
        self._beats[call_id] = now

    def _forget(self, call_id):
        self._beats.pop(call_id, None)
        for key in (
            modal_app.call_key(call_id),
            modal_app.tunnel_key(call_id),
            modal_app.heartbeat_key(call_id),
        ):
            self._state.pop(key, None)

    def _sizes(self):
        """Map every stored file's store-relative path to its size."""
        file_type = self.modal.volume.FileEntryType.FILE
        try:
            entries = self._volume.listdir("/", recursive=True)
        except self.modal.exception.NotFoundError:
            return {}
        return {
            entry.path.lstrip("/"): entry.size
            for entry in entries
            if entry.type == file_type
        }

    @staticmethod
    def _stored_name(source, file):
        return str(modal_app.store_directory(source.name, source.files) / file)

    @staticmethod
    def _check_name(name):
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe model store path: {name}")
