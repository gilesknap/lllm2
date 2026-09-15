"""The Modal remote provider.

``ModalProvider`` drives the app in ``modal_app``. It deploys the app on first
use and again when the lllm2 code changes. Serve calls are tracked in a Modal
Dict: the provider records each call it spawns, and the serve container
publishes its tunnel address there. ``calls`` lists only recorded calls, so it
never reports other apps in the workspace. Log lines travel through a Modal
Queue partition per call.

While it drives a call through ``poll`` or ``ensure_model``, the provider writes
heartbeats to the Dict. A container whose owner stays silent for
``modal_app.OWNER_GRACE_SECONDS`` stops itself, so a crashed or powered-off
machine does not leave a GPU billing. Listing calls sends no heartbeat.
"""

import functools
import itertools
import threading
import time
from pathlib import PurePosixPath

from . import modal_app
from .engine import Cancelled
from .proxy import Upstream
from .remote import (
    DownloadProgress,
    GpuProbe,
    RemoteCall,
    RemoteProvider,
    ServeStatus,
    StoredModel,
)

CREDENTIALS_MESSAGE = (
    "Modal credentials are missing or invalid. Run `lllm2 modal setup` to sign in "
    "and deploy the lllm2 app."
)


def create_provider():
    """Create the Modal provider, the factory registered under ``"modal"``.

    Returns:
        A new ``ModalProvider``.

    Raises:
        RuntimeError: The ``modal`` package is not installed.
    """
    try:
        import modal
    except ImportError as error:
        raise RuntimeError(modal_app.INSTALL_MESSAGE) from error
    return ModalProvider(modal)


def _translated(method):
    """Turn Modal client errors into ``RuntimeError`` with a user-facing message."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        errors = self.modal.exception
        try:
            return method(self, *args, **kwargs)
        except errors.AuthError as error:
            raise RuntimeError(CREDENTIALS_MESSAGE) from error
        except errors.Error as error:
            raise RuntimeError(f"Modal request failed: {error}") from error

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
            The deployed app version.
        """
        return self._ensure_deployed()

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
        if cancel.is_set():
            raise Cancelled()
        sizes = self._sizes()
        cached = self._state.get(modal_app.meta_key(source.name))
        if (
            cached is not None
            and cached.get("size") == sizes.get(source.name)
            and all(self._stored_name(source, f) in sizes for f in source.files)
        ):
            return cached["meta"]
        if source.repo is None or not source.files:
            raise ValueError(
                "This model is not in the Modal Volume and has no download source. "
                "Choose a catalogue model."
            )
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
            while True:
                if cancel.is_set():
                    raise Cancelled()
                self._beat(call.object_id)
                try:
                    meta = call.get(timeout=self.poll_interval)
                    finished = True
                    break
                except errors.OutputExpiredError:
                    raise
                except (TimeoutError, errors.TimeoutError):
                    pass
                update = self._state.get(key)
                if update:
                    progress(
                        DownloadProgress(
                            update["file"], update["done_bytes"], update["total_bytes"]
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

    @_translated
    def models(self):
        return [
            StoredModel(name, size)
            for name, size in sorted(self._sizes().items())
            if name.lower().endswith(".gguf")
        ]

    @_translated
    def remove_model(self, name):
        self._check_name(name)
        errors = self.modal.exception
        for path in (name, name + ".part"):
            try:
                self._volume.remove_file(path)
            except errors.NotFoundError:
                if path == name:
                    raise ValueError(
                        f"No model named {name} in the Modal Volume."
                    ) from None
        self._state.pop(modal_app.meta_key(name), None)

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
        self._state.put(
            modal_app.call_key(call.object_id), {"gpu": gpu, "started": time.time()}
        )
        self._beat(call.object_id)
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
                    RemoteCall(call_id, record.get("gpu"), record.get("started"))
                )
            else:
                self._forget(call_id)
        return sorted(found, key=lambda call: call.started or 0)

    def _ensure_deployed(self):
        with self._lock:
            if self._deployed is False:
                version = modal_app.deployment_version()
                if self._state.get(modal_app.DEPLOYMENT_KEY) != version:
                    self._deploy()
                    self._state.put(modal_app.DEPLOYMENT_KEY, version)
                self._deployed = version
            return self._deployed

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
        except (TimeoutError, errors.TimeoutError):
            return True, None
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
