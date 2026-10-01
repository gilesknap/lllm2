"""Every running container of a provider, joined to lllm2's serve calls.

The fake provider lists a container per serve call and the containers a test
records with ``run_foreign``, so ownership, cost and the unknown class are
checked without Modal.
"""

import threading
import time

import pytest

from fake_remote import FakeProvider, free_port
from lllm2 import config
from lllm2.remote import (
    CallRecords,
    ModelSource,
    ProviderError,
    RemoteEngine,
    RemoteProvider,
    describe_calls,
    describe_containers,
)
from lllm2.settings import Settings

ROW_KEYS = {
    "id",
    "gpu",
    "started",
    "elapsed_seconds",
    "usd_per_hour",
    "estimated_cost_usd",
    "model",
    "heartbeat_age",
    "adoptable",
    "owner",
    "status",
}


class CallsOnly(FakeProvider):
    """A provider that can list its serve calls but no other containers."""

    def containers(self):
        return None


def eventually(check, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    made = FakeProvider(tmp_path / "remote")
    yield made
    made.close()


def records():
    return CallRecords(config.STATE_DIR / "remote-calls.json")


def spawn(provider):
    call_id = provider.spawn(
        "FAKE-24", ["llama-server", "--port", str(free_port())], "key", {}, {}
    )
    assert eventually(lambda: any(c.id == call_id for c in provider.calls()))
    return call_id


def test_a_container_runs_the_status_of_its_serve_call(provider):
    owned, active, orphan = spawn(provider), spawn(provider), spawn(provider)
    provider.silence(orphan)
    calls = {row["id"]: row for row in describe_calls(provider, records(), {owned})}
    view = describe_containers(provider, records(), {owned})
    assert (view["complete"], view["error"]) == (True, None)
    rows = {row["id"]: row for row in view["containers"]}
    assert {call: row["status"] for call, row in rows.items()} == {
        owned: "owned",
        active: "active",
        orphan: "orphan",
    }
    # Times move between the two listings; everything else is the call's.
    same = ROW_KEYS - {"elapsed_seconds", "estimated_cost_usd", "heartbeat_age"}
    for call, row in rows.items():
        # The container takes its call's row and adds where it runs.
        assert row["container_id"] == "ct-" + call
        assert (row["app"], row["function"]) == ("lllm2", "serve")
        assert {k: row[k] for k in same} == {k: calls[call][k] for k in same}
        assert (row["gpu"], row["usd_per_hour"]) == ("FAKE-24", 3.6)
        assert row["estimated_cost_usd"] is not None


def test_a_container_without_a_tracked_call_is_unknown(provider):
    foreign = provider.run_foreign("my-batch-job", started=provider.clock() - 3600)
    priced = provider.run_foreign("gpu-job", gpu="FAKE-24")
    pending = provider.run_foreign("queued-job", pending=True)
    probe = provider.run_foreign("lllm2", function="probe", call_id="fc-probe")
    rows = {
        row["container_id"]: row
        for row in describe_containers(provider, records())["containers"]
    }
    assert {row["status"] for row in rows.values()} == {"unknown"}
    assert all(row["id"] is None and not row["adoptable"] for row in rows.values())
    # Modal's listing reports no GPU, and lllm2 never guesses one or its price.
    row = rows[foreign]
    assert (row["app"], row["function"], row["gpu"]) == ("my-batch-job", None, None)
    assert row["elapsed_seconds"] >= 3600
    assert (row["usd_per_hour"], row["estimated_cost_usd"]) == (None, None)
    # A GPU type the listing does name is priced from the provider's table.
    assert rows[priced]["usd_per_hour"] == 3.6
    assert rows[priced]["estimated_cost_usd"] is not None
    assert rows[pending]["elapsed_seconds"] is None
    # lllm2's own probe is named, though it is not a serve call this panel tracks.
    assert (rows[probe]["app"], rows[probe]["function"]) == ("lllm2", "probe")
    # Oldest first, and a container that has not started yet last.
    order = [row["container_id"] for row in describe_containers(provider)["containers"]]
    assert order[0] == foreign and order[-1] == pending


def test_a_provider_that_cannot_enumerate_still_reports_its_calls(provider, tmp_path):
    calls_only = CallsOnly(tmp_path / "remote")
    try:
        call_id = spawn(calls_only)
        calls_only.run_foreign("my-batch-job")
        view = describe_containers(calls_only, records())
    finally:
        calls_only.close()
    assert (view["complete"], view["error"]) == (False, None)
    (row,) = view["containers"]
    assert (row["id"], row["container_id"], row["app"]) == (call_id, None, None)
    assert (row["function"], row["status"]) == ("serve", "active")


def test_a_failed_listing_keeps_the_serve_calls(provider, monkeypatch):
    call_id = spawn(provider)

    def boom():
        raise ProviderError("boom")

    monkeypatch.setattr(provider, "containers", boom)
    view = describe_containers(provider, records())
    assert (view["complete"], view["error"]) == (False, "boom")
    assert [row["id"] for row in view["containers"]] == [call_id]
    assert view["containers"][0]["container_id"] is None


def test_describe_calls_keeps_its_row_shape(provider):
    spawn(provider)
    (row,) = describe_calls(provider, records())
    assert set(row) == ROW_KEYS
    assert row["estimated_cost_usd"] == pytest.approx(
        row["elapsed_seconds"] * 3.6 / 3600
    )


def test_the_default_provider_contract_lists_nothing_and_stops_nothing(provider):
    class Minimal(FakeProvider):
        containers = RemoteProvider.containers
        stop_container = RemoteProvider.stop_container

    minimal = Minimal(provider.root)
    assert minimal.containers() is None
    with pytest.raises(ProviderError, match="cannot stop containers"):
        minimal.stop_container("ct-1")


def test_stop_listed_cancels_calls_and_stops_other_containers(provider, tmp_path):
    engine = RemoteEngine(
        provider,
        "FAKE-24",
        port=free_port(),
        poll_interval=0.05,
        sources=lambda path: ModelSource(
            "example/model.gguf", "fake/example", ("model.gguf",)
        ),
    )
    try:
        orphan = spawn(provider)
        provider.silence(orphan)
        records().put("fake", orphan, {"gpu": "FAKE-24", "saved": 0})
        foreign = provider.run_foreign("my-batch-job")
        rows = {
            row["container_id"]: row for row in engine.remote_containers()["containers"]
        }
        engine.stop_listed(rows["ct-" + orphan])
        assert all(c.id != orphan for c in provider.calls())
        assert records().get("fake", orphan) is None
        engine.stop_listed(rows[foreign])
        assert foreign not in {c.id for c in provider.containers()}
        assert any(foreign in line for line in engine.logs())
        # The engine's own call stops through the engine, not as a listed row.
        engine.start(
            Settings.parse(
                {
                    "model": str(config.MODELS_DIR / "example" / "model.gguf"),
                    "backend": "fake",
                    "gpu_type": "FAKE-24",
                }
            ),
            threading.Event(),
            timeout=30,
        )
        (owned,) = engine.remote_containers()["containers"]
        assert owned["status"] == "owned"
        with pytest.raises(ValueError, match="Stop the engine instead"):
            engine.stop_listed(owned)
    finally:
        engine.shutdown()
