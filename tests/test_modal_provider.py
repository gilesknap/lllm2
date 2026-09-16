"""The Modal provider and app functions, with a fake ``modal`` module."""

import dis
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from fake_remote import ENGINE, FAKE_LLAMA_SERVER, free_port
from fake_remote import META as FAKE_META
from lllm2 import config, discovery, engine_install, modal_app
from lllm2.engine import Cancelled
from lllm2.engine_release import CUDA_TRACKS, LLAMA_CPP_REF, asset_name
from lllm2.gpu_tables import pricing_caveat
from lllm2.modal_provider import ModalProvider
from lllm2.proxy import Upstream
from lllm2.remote import (
    Deployment,
    DownloadProgress,
    GpuProbe,
    ModelSource,
    RemoteCall,
    RemoteEngine,
    StoredModel,
    catalogue_source,
    heartbeat_fresh,
    remote_provider,
)
from lllm2.settings import Settings

GGUF = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
META = {"architecture": "qwen3", "context": 40960, "mtp": False, "template": ""}


class Error(Exception):
    pass


class AuthError(Error):
    pass


class NotFoundError(Error):
    pass


class ModalTimeoutError(Error):
    pass


class OutputExpiredError(ModalTimeoutError):
    pass


class ModalConnectionError(Error):
    pass


class InputCancellation(BaseException):
    pass


class FakeDict(dict):
    def put(self, key, value, *, skip_if_exists=False):
        self[key] = value
        return True


class FakeQueue:
    def __init__(self):
        self.partitions = {}

    def put_many(self, values, block=True, timeout=None, *, partition=None):
        self.partitions.setdefault(partition, []).extend(values)

    def get_many(self, n_values, block=True, timeout=None, *, partition=None):
        values = self.partitions.get(partition, [])
        taken, self.partitions[partition] = values[:n_values], values[n_values:]
        return taken


class FakeVolume:
    def __init__(self):
        self.files = {}

    def listdir(self, path, *, recursive=False):
        return [
            SimpleNamespace(path=name, type=1, size=size)
            for name, size in self.files.items()
        ]

    def remove_file(self, path, recursive=False):
        if path not in self.files:
            raise NotFoundError(path)
        del self.files[path]


class FakeCall:
    """A function call that reports running for ``pending`` polls, then ``result``."""

    def __init__(
        self, modal, function, gpu, args, pending=0, result=None, on_poll=None
    ):
        self.object_id = f"fc-{len(modal.calls) + 1}"
        self.function, self.gpu, self.args = function, gpu, args
        self.pending, self.result, self.on_poll = pending, result, on_poll
        self.cancelled = False
        modal.calls[self.object_id] = self

    def get(self, timeout=None):
        if self.cancelled:
            raise InputCancellation()
        if self.pending:
            self.pending -= 1
            if self.on_poll:
                self.on_poll(self)
            raise ModalTimeoutError()
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def cancel(self, terminate_containers=False):
        self.cancelled = True


class FakeFunction:
    def __init__(self, modal, name, gpu=None):
        self.modal, self.name, self.gpu = modal, name, gpu

    def with_options(self, *, gpu=None):
        return FakeFunction(self.modal, self.name, gpu)

    def hydrate(self):
        if self.modal.lookup_error is not None:
            raise self.modal.lookup_error
        return self

    def remote(self, *args):
        self.modal.remote_calls.append((self.name, self.gpu, args))
        return self.modal.behaviour[self.name](*args)

    def spawn(self, *args):
        options = self.modal.behaviour[self.name](*args)
        return FakeCall(self.modal, self.name, self.gpu, args, **options)


class FakeModal:
    """The parts of the ``modal`` module that the provider uses."""

    def __init__(self):
        self.exception = SimpleNamespace(
            Error=Error,
            AuthError=AuthError,
            NotFoundError=NotFoundError,
            TimeoutError=ModalTimeoutError,
            OutputExpiredError=OutputExpiredError,
            InputCancellation=InputCancellation,
            ConnectionError=ModalConnectionError,
            ServiceError=ModalConnectionError,
            InternalError=ModalConnectionError,
            ClientClosed=ModalConnectionError,
        )
        self.volume = SimpleNamespace(FileEntryType=SimpleNamespace(FILE=1))
        self.state, self.queue, self.store = FakeDict(), FakeQueue(), FakeVolume()
        self.calls, self.remote_calls = {}, []
        self.behaviour = {}
        self.lookup_error = None
        self.Dict = SimpleNamespace(from_name=lambda name, **_: self.state)
        self.Queue = SimpleNamespace(from_name=lambda name, **_: self.queue)
        self.Volume = SimpleNamespace(from_name=lambda name, **_: self.store)
        self.Function = SimpleNamespace(
            from_name=lambda app, name: FakeFunction(self, name)
        )
        self.FunctionCall = SimpleNamespace(from_id=self._from_id)

    def _from_id(self, call_id):
        try:
            return self.calls[call_id]
        except KeyError:
            raise NotFoundError(call_id) from None


@pytest.fixture
def fake():
    return FakeModal()


@pytest.fixture
def deploys():
    return []


@pytest.fixture
def provider(fake, deploys):
    return ModalProvider(fake, poll_interval=0, deploy=lambda: deploys.append(1))


def test_factory_names_the_extra_when_modal_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(RuntimeError, match=r"pip install 'lllm2\[modal\]'"):
        remote_provider("modal")


def test_modules_import_without_modal():
    code = (
        "import sys; sys.modules['modal'] = None\n"
        "import lllm2.modal_app as app, lllm2.modal_provider, lllm2.remote\n"
        "assert not hasattr(app, 'app')\n"
        "try:\n    app.deploy()\nexcept RuntimeError as e:\n    print(e)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "lllm2[modal]" in result.stdout


def test_missing_credentials_point_to_setup(fake, provider):
    def refuse(*_args):
        raise AuthError("Token missing")

    fake.state.items = refuse
    with pytest.raises(RuntimeError, match="lllm2 modal setup"):
        provider.calls()


def test_app_deploys_on_first_use_and_after_a_code_change(fake, deploys, monkeypatch):
    fake.behaviour["probe"] = lambda: {
        "name": "Tesla T4",
        "total_mib": 15360,
        "engine": {},
    }
    monkeypatch.setattr(modal_app, "deployment_version", lambda: "1.0+abc")
    ModalProvider(fake, deploy=lambda: deploys.append(1)).probe("T4")
    first = ModalProvider(fake, deploy=lambda: deploys.append(1))
    first.probe("T4")
    first.probe("L4")
    assert deploys == [1]
    monkeypatch.setattr(modal_app, "deployment_version", lambda: "1.1+def")
    assert ModalProvider(fake, deploy=lambda: deploys.append(1)).setup() == Deployment(
        "1.1+def", True
    )
    assert deploys == [1, 1]


def test_setup_rerun_does_not_redeploy(fake, provider, deploys):
    first = provider.setup()
    assert first == Deployment(fake.state[modal_app.DEPLOYMENT_KEY], True)
    assert provider.setup() == Deployment(first.version, False)
    rerun = ModalProvider(fake, deploy=lambda: deploys.append(1)).setup()
    assert rerun == Deployment(first.version, False)
    assert deploys == [1]


def test_setup_redeploys_an_app_deleted_outside_lllm2(fake, provider, deploys):
    first = provider.setup()
    fake.lookup_error = NotFoundError("app not found")
    rerun = ModalProvider(fake, deploy=lambda: deploys.append(1)).setup()
    assert rerun == Deployment(first.version, True)
    assert deploys == [1, 1]


def test_setup_reports_a_failed_app_lookup(fake, provider, deploys):
    provider.setup()
    fake.lookup_error = Error("service unavailable")
    with pytest.raises(RuntimeError, match="service unavailable"):
        ModalProvider(fake, deploy=lambda: deploys.append(1)).setup()
    assert deploys == [1]


def test_probe_hashes_the_binary_that_a_local_install_hashes(tmp_path, monkeypatch):
    """The image and a local install unpack the same tarball to the same sha256."""
    server = b"#!/bin/sh\nexit 0\n"
    metadata = {
        "requested_ref": LLAMA_CPP_REF,
        "cuda_track": CUDA_TRACKS["13"],
        "lllm2_version": "0.3.0",
        "backend": "cuda",
        "architecture": "x86_64",
        "glibc": "2.28",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, data in {
            "./llama-server": server,
            "./lllm2-engine.json": json.dumps(metadata).encode(),
            "./libggml-cuda.so": b"library",
        }.items():
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755
            bundle.addfile(member, io.BytesIO(data))
    archive = buffer.getvalue()
    checksum = f"{hashlib.sha256(archive).hexdigest()}  {asset_name('13')}\n"

    def download(url, destination, **_kwargs):
        destination.write_bytes(checksum.encode() if url == "checksum" else archive)

    monkeypatch.setattr(
        engine_install, "_release_asset_urls", lambda _asset: ("archive", "checksum")
    )
    monkeypatch.setattr(engine_install, "_download", download)
    monkeypatch.setattr(engine_install, "cuda_track", lambda: "13")
    monkeypatch.setattr(engine_install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(engine_install.platform, "machine", lambda: "x86_64")
    local = engine_install.install("cuda", root=tmp_path / "local")
    image = tmp_path / "image"
    # The image build installs each track without a driver, as install_engines does.
    engine_install.install("cuda", track="13", root=image, check_startup=False)
    tools = tmp_path / "bin"
    tools.mkdir()
    smi = tools / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'NVIDIA L4, 23034'\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")

    binary = modal_app.engine_binary(str(image))
    # The probe reads the build the container installed, whatever track a
    # driver would pick now, so the reported build follows the binary.
    monkeypatch.setattr(engine_install, "cuda_track", lambda: "12")
    probed = modal_app.probe_container(binary)

    expected = hashlib.sha256(server).hexdigest()
    assert discovery.probe(local)["sha256"] == expected
    assert probed["engine"]["sha256"] == expected
    assert (probed["name"], probed["total_mib"]) == ("NVIDIA L4", 23034)
    assert probed["engine"]["cuda_track"] == CUDA_TRACKS["13"]
    assert probed["engine"]["requested_ref"] == LLAMA_CPP_REF


def test_a_deleted_app_is_redeployed(fake, provider, deploys):
    missing = [True]

    def probe():
        if missing.pop() if missing else False:
            raise NotFoundError("app not found")
        return {"name": "NVIDIA L4", "total_mib": 23034, "engine": {}}

    fake.behaviour["probe"] = probe
    assert provider.probe("L4").name == "NVIDIA L4"
    assert deploys == [1, 1]


def test_probe_runs_on_the_requested_gpu(fake, provider):
    engine = {"path": "/opt/lllm2/engines/x/llama-server", "flags": ["--model"]}
    fake.behaviour["probe"] = lambda: {
        "name": "NVIDIA L4",
        "total_mib": "23034",
        "engine": engine,
    }
    assert provider.probe("L4") == GpuProbe("NVIDIA L4", 23034, engine)
    assert fake.remote_calls == [("probe", "L4", ())]


def test_download_reports_progress_then_reuses_the_stored_model(fake, provider):
    source = ModelSource(
        "Qwen3-8B/Qwen3-8B-Q4_K_M.gguf",
        "unsloth/Qwen3-8B-GGUF",
        ("Qwen3-8B-Q4_K_M.gguf",),
    )

    def download(name, repo, files, revision):
        assert (name, repo, files, revision) == (
            source.name,
            source.repo,
            list(source.files),
            None,
        )

        def poll(call):
            # The provider sends heartbeats so the download container keeps going.
            assert fake.state.get(modal_app.heartbeat_key(call.object_id))
            key = modal_app.download_key(call.object_id)
            fake.state.put(
                key, {"file": files[0], "done_bytes": 500, "total_bytes": 1000}
            )
            if not call.pending:
                fake.store.files[source.name] = 1000

        return {"pending": 2, "result": dict(META), "on_poll": poll}

    fake.behaviour["download"] = download
    updates = []
    assert provider.ensure_model(source, updates.append, threading.Event()) == META
    assert updates == [DownloadProgress("Qwen3-8B-Q4_K_M.gguf", 500, 1000)] * 2
    assert provider.models() == [StoredModel(source.name, 1000)]
    assert provider.model_path(source.name) == "/models/" + source.name
    # The second launch reads cached metadata without starting a container.
    assert provider.ensure_model(source, updates.append, threading.Event()) == META
    assert len(fake.calls) == 1


def test_cancelling_a_download_cancels_the_call(fake, provider):
    cancel = threading.Event()
    fake.behaviour["download"] = lambda *args: {
        "pending": 100,
        "on_poll": lambda call: cancel.set(),
    }
    source = ModelSource("M/m.gguf", "org/repo", ("m.gguf",))
    with pytest.raises(Cancelled):
        provider.ensure_model(source, lambda update: None, cancel)
    assert [call.cancelled for call in fake.calls.values()] == [True]
    assert not any(str(key).startswith("heartbeat:") for key in fake.state)


def test_polling_keeps_a_call_alive_and_silence_ends_it(fake):
    provider = ModalProvider(fake, poll_interval=0, deploy=lambda: None, heartbeat=0)
    fake.behaviour["serve"] = lambda *args: {"pending": 10**6}
    call_id = provider.spawn("T4", ["llama-server"], "k", {}, {})
    now = [0.0]
    watch = modal_app.OwnerWatch(fake.state, call_id, grace=60, clock=lambda: now[0])
    for _ in range(5):
        now[0] += 50
        provider.poll(call_id)
        assert not watch.lost()
    # Listing calls, as `lllm2 modal list` does, is not a heartbeat. It
    # reports the silence the container's own watch measures, on the same
    # grace, so both sides end a call at the same moment.
    assert provider.heartbeat_grace == modal_app.OWNER_GRACE_SECONDS
    (listed,) = provider.calls()
    assert listed.heartbeat_age is not None and listed.heartbeat_age < 60
    assert heartbeat_fresh(listed, provider.heartbeat_grace)
    now[0] += 61
    assert watch.lost()

    silent = time.time() - provider.heartbeat_grace - 60
    fake.state.put(modal_app.heartbeat_key(call_id), (silent, 7))
    (listed,) = provider.calls()
    assert listed.heartbeat_age > provider.heartbeat_grace
    assert not heartbeat_fresh(listed, provider.heartbeat_grace)

    # A heartbeat this provider cannot read is not a fresh one.
    fake.state.put(modal_app.heartbeat_key(call_id), "unreadable")
    assert provider.calls()[0].heartbeat_age is None


def test_a_connection_failure_does_not_forget_a_running_call(
    fake, provider, monkeypatch
):
    fake.behaviour["serve"] = lambda *args: {"pending": 10**6}
    call_id = provider.spawn("T4", ["llama-server"], "k", {}, {})

    def unreachable(timeout=None):
        raise ModalConnectionError("network down")

    monkeypatch.setattr(fake.calls[call_id], "get", unreachable)
    with pytest.raises(RuntimeError, match="network down"):
        provider.poll(call_id)
    monkeypatch.undo()
    assert [call.id for call in provider.calls()] == [call_id]


def test_a_model_without_a_source_must_be_stored(provider):
    with pytest.raises(ValueError, match="catalogue model"):
        provider.ensure_model(
            ModelSource("M/m.gguf"), lambda u: None, threading.Event()
        )


def test_remove_model_deletes_the_file_and_its_partial_download(fake, provider):
    fake.store.files.update({"A/a.gguf": 10, "A/a.gguf.part": 5, "B/b.gguf": 20})
    provider.remove_model("A/a.gguf")
    assert provider.models() == [StoredModel("B/b.gguf", 20)]
    with pytest.raises(ValueError, match="No model"):
        provider.remove_model("A/a.gguf")
    with pytest.raises(ValueError, match="Unsafe"):
        provider.remove_model("../escape.gguf")


def test_remove_model_deletes_companions_and_a_partial_only_model(fake, provider):
    fake.store.files.update(
        {
            "G/g.gguf": 10,
            "G/mmproj.gguf": 2,
            "G/mmproj.gguf.part": 1,
            "P/p.gguf.part": 4,
            "K/k.gguf": 9,
        }
    )
    fake.state.put(modal_app.meta_key("G/g.gguf"), {"size": 10, "meta": META})
    provider.remove_model("G/g.gguf", ("G/mmproj.gguf",))
    # A cancelled download leaves only a partial file, which remove also deletes.
    provider.remove_model("P/p.gguf")
    assert fake.store.files == {"K/k.gguf": 9}
    assert provider.stored_metadata("G/g.gguf") is None
    for name, companions in (
        ("K/k.gguf", ("../escape.gguf",)),
        ("/K/k.gguf", ()),
        ("K/../K/k.gguf", ()),
        ("K/k.gguf", ("/etc/passwd",)),
    ):
        with pytest.raises(ValueError, match="Unsafe"):
            provider.remove_model(name, companions)
    assert fake.store.files == {"K/k.gguf": 9}


def _values(value):
    """Yield every leaf value, key included, of nested call arguments."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _values(key)
            yield from _values(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _values(item)
    else:
        yield value


def test_the_provider_never_uploads_model_files(fake, provider):
    source = ModelSource("M/m.gguf", "org/repo", ("m.gguf", "mmproj.gguf"))

    def download(*_args):
        def poll(call):
            if not call.pending:
                fake.store.files.update({"M/m.gguf": 10**9, "M/mmproj.gguf": 10**8})

        return {"pending": 1, "result": dict(META), "on_poll": poll}

    fake.behaviour["download"] = download
    fake.behaviour["serve"] = lambda *args: {"pending": 10**6}
    provider.ensure_model(source, lambda update: None, threading.Event())
    provider.spawn(
        "T4",
        ["llama-server", "--model", provider.model_path(source.name)],
        "key",
        {},
        {"chat.jinja": "{{ messages }}"},
    )
    download_call, serve_call = fake.calls.values()
    # The download names the Hugging Face files; the container fetches them.
    assert download_call.args == ("M/m.gguf", "org/repo", list(source.files), None)
    assert serve_call.args[3] == {"chat.jinja": "{{ messages }}"}
    for call in (download_call, serve_call):
        for value in _values(call.args):
            assert not isinstance(value, bytes | bytearray | memoryview)
            assert not isinstance(value, str) or len(value) < 4096


def test_fetch_model_requests_nothing_when_every_file_is_complete(
    tmp_path, monkeypatch
):
    directory = tmp_path / "M"
    directory.mkdir()
    for file in ("m.gguf", "mmproj.gguf"):
        (directory / file).write_bytes(GGUF)

    def opener(*_args, **_kwargs):
        raise AssertionError("a complete file must not be downloaded again")

    monkeypatch.setattr(modal_app.discovery, "metadata", lambda path: dict(META))
    updates, commits = [], []
    meta = modal_app.fetch_model(
        "M/m.gguf",
        "org/repo",
        ["m.gguf", "mmproj.gguf"],
        None,
        lambda *update: updates.append(update),
        lambda: commits.append(1),
        root=str(tmp_path),
        opener=opener,
    )
    assert meta == META
    size = len(GGUF)
    assert updates == [("m.gguf", size, size), ("mmproj.gguf", size, size)]


def test_serve_call_lifecycle(fake, provider):
    fake.behaviour["serve"] = lambda argv, key, env, files: {"pending": 10**6}
    fake.state.put(modal_app.download_key("fc-other"), {"file": "x"})
    call_id = provider.spawn(
        "L4", ["/bin/llama-server"], "secret", {"A": "1"}, {"t.jinja": "x"}
    )
    call = fake.calls[call_id]
    assert (call.function, call.gpu, call.args) == (
        "serve",
        "L4",
        (["/bin/llama-server"], "secret", {"A": "1"}, {"t.jinja": "x"}),
    )
    listed = provider.calls()
    assert [(c.id, c.gpu) for c in listed] == [(call_id, "L4")]
    assert isinstance(listed[0], RemoteCall) and listed[0].started
    assert provider.poll(call_id).upstream is None

    fake.queue.put_many(["loading model"], partition=call_id)
    fake.state.put(
        modal_app.tunnel_key(call_id),
        {"host": "abc.modal.host", "port": 443, "tls": True},
    )
    status = provider.poll(call_id)
    assert status.running and status.logs == ("loading model",)
    assert status.upstream == Upstream("abc.modal.host", 443, True)

    provider.cancel(call_id)
    assert call.cancelled
    assert provider.calls() == []
    assert not provider.poll(call_id).running
    provider.cancel("fc-unknown")


def test_remote_engine_serves_and_stops_a_modal_call(fake, provider, tmp_path):
    """A serve call's tunnel record points the engine proxy at a local server."""
    script = tmp_path / "fake_llama_server.py"
    script.write_text(FAKE_LLAMA_SERVER)
    servers = []

    def serve(argv, key, env, files):
        argv = list(argv)
        argv[argv.index("--port") + 1] = str(port := free_port())
        servers.append(
            subprocess.Popen(
                [sys.executable, str(script), *argv[1:]],
                env={**os.environ, "LLAMA_API_KEY": key},
            )
        )
        call_id = f"fc-{len(fake.calls) + 1}"
        fake.state.put(
            modal_app.tunnel_key(call_id),
            {"host": "127.0.0.1", "port": port, "tls": False},
        )
        return {"pending": 10**6}

    fake.behaviour["probe"] = lambda: {
        "name": "Tesla T4",
        "total_mib": 15360,
        "engine": ENGINE,
    }
    fake.behaviour["serve"] = serve
    fake.store.files["example/model.gguf"] = 3000
    fake.state.put(
        modal_app.meta_key("example/model.gguf"), {"size": 3000, "meta": FAKE_META}
    )
    engine = RemoteEngine(
        provider,
        "T4",
        port=free_port(),
        poll_interval=0.05,
        records=tmp_path / "calls.json",
        probes=tmp_path / "probes.json",
        sources=lambda path: ModelSource("example/model.gguf"),
    )
    try:
        engine.start(
            Settings(model="/models/example/model.gguf"), threading.Event(), 30
        )
        call_id = engine.call_id
        assert [call.id for call in provider.calls()] == [call_id]
        # The proxy adds the call's key; the client sends only a placeholder.
        request = urllib.request.Request(
            engine.base + "/tokenize",
            data=json.dumps({"content": "ab"}).encode(),
            headers={
                "Authorization": "Bearer placeholder",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            assert json.load(response) == {"tokens": [98, 99]}
        engine.stop()
        assert fake.calls[call_id].cancelled
        assert provider.calls() == []
        assert not engine.alive()
    finally:
        engine.shutdown()
        for server in servers:
            server.kill()
            server.wait()


def test_an_ended_call_reports_its_exit_status(fake, provider):
    fake.behaviour["serve"] = lambda *args: {"pending": 1, "result": {"exit_code": 1}}
    call_id = provider.spawn("T4", ["llama-server"], "k", {}, {})
    assert provider.poll(call_id).running
    status = provider.poll(call_id)
    assert (status.running, status.error) == (
        False,
        "llama-server exited with status 1",
    )
    assert provider.calls() == []


def test_paths_inside_the_container(provider):
    assert provider.file_path("dir/template.jinja") == "/tmp/lllm2-files/template.jinja"
    with pytest.raises(ValueError):
        provider.model_path("/etc/passwd")


def test_pricing_caveat_asks_to_check_current_prices():
    assert "check current modal pricing" in pricing_caveat("modal").lower()


def test_catalogue_source_uses_the_local_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    entry = {
        "name": "Gemma",
        "repo": "unsloth/gemma-GGUF",
        "file": "gemma.gguf",
        "mmproj": "mmproj-F16.gguf",
    }
    assert catalogue_source(entry) == ModelSource(
        "Gemma/gemma.gguf", "unsloth/gemma-GGUF", ("gemma.gguf", "mmproj-F16.gguf")
    )
    with pytest.raises(ValueError):
        catalogue_source({**entry, "file": "../x.gguf"})


class Response(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status, self.headers = status, headers


def test_fetch_model_resumes_a_partial_file(tmp_path):
    directory = tmp_path / "M"
    directory.mkdir()
    (directory / "m.gguf.part").write_bytes(GGUF[:10])
    (directory / "extra.gguf").write_bytes(GGUF)
    requests = []

    def opener(request, timeout, context):
        requests.append((request.full_url, request.get_header("Range")))
        return Response(
            GGUF[10:],
            206,
            {
                "Content-Range": f"bytes 10-{len(GGUF) - 1}/{len(GGUF)}",
                "Content-Length": str(len(GGUF) - 10),
            },
        )

    updates, commits = [], []
    meta = modal_app.fetch_model(
        "M/m.gguf",
        "org/repo",
        ["m.gguf", "extra.gguf"],
        None,
        lambda *update: updates.append(update),
        lambda: commits.append(1),
        root=str(tmp_path),
        opener=opener,
    )
    assert requests == [
        ("https://huggingface.co/org/repo/resolve/main/m.gguf", "bytes=10-")
    ]
    assert (directory / "m.gguf").read_bytes() == GGUF
    assert not (directory / "m.gguf.part").exists()
    assert updates[-1] == ("extra.gguf", len(GGUF), len(GGUF))
    assert commits and meta["error"] is None


def test_fetch_model_keeps_a_short_download_for_resume(tmp_path):
    def opener(request, timeout, context):
        return Response(GGUF[:5], 200, {"Content-Length": str(len(GGUF))})

    with pytest.raises(RuntimeError, match="ended early"):
        modal_app.fetch_model(
            "M/m.gguf",
            "o/r",
            ["m.gguf"],
            "v1",
            lambda *u: None,
            lambda: None,
            root=str(tmp_path),
            opener=opener,
        )
    assert (tmp_path / "M" / "m.gguf.part").read_bytes() == GGUF[:5]


def test_download_stops_when_the_owner_goes_silent(tmp_path):
    def opener(request, timeout, context):
        return Response(GGUF, 200, {"Content-Length": str(len(GGUF))})

    state, commits = FakeDict(), []
    arguments = ("M/m.gguf", "o/r", ["m.gguf"], None)
    options = {"call_id": "fc-1", "state": state, "root": str(tmp_path)}
    with pytest.raises(RuntimeError, match="heartbeats"):
        modal_app.run_download(
            *arguments,
            commit=lambda: commits.append(1),
            grace=-1,
            opener=opener,
            **options,
        )
    assert commits and not (tmp_path / "M" / "m.gguf").exists()
    state.put(modal_app.heartbeat_key("fc-1"), 1)
    meta = modal_app.run_download(
        *arguments, commit=lambda: None, opener=opener, **options
    )
    assert meta["error"] is None and (tmp_path / "M" / "m.gguf").read_bytes() == GGUF
    assert modal_app.download_key("fc-1") not in state


def test_run_server_stops_llama_server_when_the_owner_goes_silent(tmp_path):
    script = tmp_path / "server.py"
    script.write_text("import time\ntime.sleep(60)\n")

    @contextmanager
    def forward(port):
        yield SimpleNamespace(tls_socket=("abc.modal.host", 443))

    started = time.monotonic()
    result = modal_app.run_server(
        [sys.executable, str(script)],
        "k",
        {},
        {},
        call_id="fc-1",
        state=FakeDict(),
        logs=FakeQueue(),
        forward=forward,
        file_root=str(tmp_path / "files"),
        heartbeat=0.05,
        grace=0.3,
    )
    assert result["exit_code"] is not None and "heartbeats" in result["error"]
    assert time.monotonic() - started < 30


def test_store_paths_must_stay_inside_the_store():
    assert str(modal_app.store_directory("A/b/c.gguf", ["b/c.gguf"])) == "A"
    for name, files in (
        ("A/c.gguf", ["d.gguf"]),
        ("A/../c.gguf", ["c.gguf"]),
        ("/c.gguf", ["c.gguf"]),
    ):
        with pytest.raises(ValueError):
            modal_app.store_directory(name, files)


def test_run_server_publishes_the_tunnel_and_logs(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(
        "import os, sys, time\n"
        "print('key', os.environ['LLAMA_API_KEY'], os.environ['EXTRA'], flush=True)\n"
        "print('template', open(sys.argv[1]).read(), flush=True)\n"
        "time.sleep(0.3)\n"
        "sys.exit(3)\n"
    )
    published = []

    class State(FakeDict):
        def put(self, key, value, **kwargs):
            published.append((key, value))
            return super().put(key, value)

    ports = []

    @contextmanager
    def forward(port):
        ports.append(port)
        yield SimpleNamespace(tls_socket=("abc.modal.host", 443))

    state, logs = State(), FakeQueue()
    files = tmp_path / "files"
    result = modal_app.run_server(
        [sys.executable, str(script), str(files / "t.jinja")],
        "secret",
        {"EXTRA": "yes"},
        {"t.jinja": "hello"},
        call_id="fc-1",
        state=state,
        logs=logs,
        forward=forward,
        port=9000,
        file_root=str(files),
        heartbeat=0.05,
    )
    assert result == {"exit_code": 3, "error": None}
    assert ports == [9000]
    key, record = published[0]
    assert key == "tunnel:fc-1"
    assert (record["host"], record["port"], record["tls"]) == (
        "abc.modal.host",
        443,
        True,
    )
    assert "tunnel:fc-1" not in state
    assert logs.partitions["fc-1"] == ["key secret yes", "template hello"]


def test_engine_layer_installs_each_pinned_track_without_a_gpu(monkeypatch, tmp_path):
    from lllm2 import engine_install
    from lllm2.engine_release import CUDA_TRACKS, LLAMA_CPP_REF

    installed = []
    monkeypatch.setattr(
        engine_install,
        "install",
        lambda backend, **options: installed.append((backend, options)),
    )
    modal_app.install_engines(LLAMA_CPP_REF, dict(CUDA_TRACKS), str(tmp_path))
    assert [(b, o["track"], o["check_startup"]) for b, o in installed] == [
        ("cuda", track, False) for track in CUDA_TRACKS
    ]
    # Stale layer arguments fail the build instead of installing other engines.
    with pytest.raises(RuntimeError, match="pins"):
        modal_app.install_engines("b1", dict(CUDA_TRACKS), str(tmp_path))
    # The layer key must not depend on module globals, which change with code.
    global_reads = {
        op.argval
        for op in dis.get_instructions(modal_app.install_engines)
        if op.opname in ("LOAD_GLOBAL", "LOAD_NAME")
    }
    assert not global_reads & set(vars(modal_app))


def test_opener_default_is_urlopen():
    assert modal_app.fetch_model.__defaults__[-1] is urllib.request.urlopen
