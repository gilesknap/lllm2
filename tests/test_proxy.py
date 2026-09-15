"""The engine proxy through its HTTP interface, against an in-process upstream."""

import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from lllm2 import proxy as proxy_module
from lllm2.proxy import EngineProxy, Upstream, body_chunks

EVENTS = b'data: {"n": 1}\n\ndata: {"n": 2}\n\ndata: [DONE]\n\n'


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            first, rest = EVENTS[:16], EVENTS[16:]
            self.wfile.write(first)
            self.wfile.flush()
            self.server.gate.wait(10)  # type: ignore[attr-defined]
            self.wfile.write(rest)
            self.wfile.flush()
            self.close_connection = True
            return
        if self.path == "/silent":
            # Prompt processing: no response bytes until the proxy drops the request.
            self.server.entered.set()  # type: ignore[attr-defined]
            self.connection.settimeout(10)
            try:
                self.connection.recv(1)
            except OSError:
                return
            self.server.dropped.set()  # type: ignore[attr-defined]
            return
        if self.path == "/chunked-events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for n in range(3):
                event = b"data: %d\n\n" % n
                self.wfile.write(b"%x\r\n%s\r\n" % (len(event), event))
                self.wfile.flush()
                self.server.gate.wait(10)  # type: ignore[attr-defined]
                self.server.gate.clear()  # type: ignore[attr-defined]
            self.wfile.write(b"0\r\n\r\n")
            return
        if self.path.startswith("/slow"):
            time.sleep(float(self.path.partition("?")[2]))
        self.reply({"path": self.path})

    def do_POST(self):  # noqa: N802
        chunked = "chunked" in self.headers.get("Transfer-Encoding", "")
        body = b""
        for part in body_chunks(
            self.rfile, self.headers.get("Content-Length"), chunked
        ):
            body += part
            self.server.first_part.set()  # type: ignore[attr-defined]
        self.reply(
            {
                "body": body.decode(),
                "authorization": self.headers.get("Authorization"),
                "x_api_key": self.headers.get("X-Api-Key"),
                "chunked": chunked,
            }
        )

    def log_message(self, *args):
        pass


@pytest.fixture
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    server.daemon_threads = True
    server.gate = threading.Event()  # type: ignore[attr-defined]
    server.first_part = threading.Event()  # type: ignore[attr-defined]
    server.entered = threading.Event()  # type: ignore[attr-defined]
    server.dropped = threading.Event()  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.gate.set()  # type: ignore[attr-defined]
        server.shutdown()
        server.server_close()


@pytest.fixture
def proxy(upstream):
    engine_proxy = EngineProxy(0)
    engine_proxy.start()
    engine_proxy.connect(
        Upstream("127.0.0.1", upstream.server_address[1]), "remote-secret"
    )
    try:
        yield engine_proxy
    finally:
        engine_proxy.close()


def url(proxy, path):
    return f"http://127.0.0.1:{proxy.port}{path}"


def get(proxy, path, timeout=10):
    with urllib.request.urlopen(url(proxy, path), timeout=timeout) as response:
        return json.load(response)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_proxy_injects_the_key_and_drops_client_credentials(proxy):
    request = urllib.request.Request(
        url(proxy, "/v1/chat/completions"),
        data=b'{"messages": []}',
        headers={
            "Authorization": "Bearer placeholder",
            "X-Api-Key": "client",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        seen = json.load(response)
    assert seen["authorization"] == "Bearer remote-secret"
    assert seen["x_api_key"] is None
    assert seen["body"] == '{"messages": []}'


def test_proxy_streams_server_sent_events_without_buffering(proxy, upstream):
    connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=3)
    connection.request("GET", "/events", headers={"Accept": "text/event-stream"})
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/event-stream"
    # The upstream holds the rest of the stream until the client has the first event.
    first = response.readline() + response.readline()
    assert first == b'data: {"n": 1}\n\n'
    upstream.gate.set()
    assert first + response.read() == EVENTS
    connection.close()


def test_proxy_relays_each_chunked_event_as_it_arrives(proxy, upstream):
    connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=3)
    connection.request("GET", "/chunked-events")
    response = connection.getresponse()
    for n in range(3):
        # The upstream sends the next event only after the client has this one.
        assert response.readline() + response.readline() == b"data: %d\n\n" % n
        upstream.gate.set()
    assert response.read() == b""
    connection.close()


def wait_for(check, timeout=5):
    deadline = time.monotonic() + timeout
    while not check() and time.monotonic() < deadline:
        time.sleep(0.01)
    return check()


def test_client_leaving_during_prompt_processing_ends_the_upstream_request(
    proxy, upstream
):
    client = socket.create_connection(("127.0.0.1", proxy.port))
    client.sendall(b"GET /silent HTTP/1.1\r\nHost: localhost\r\n\r\n")
    assert upstream.entered.wait(5)
    assert proxy.activity()[0] == 1
    client.close()
    assert upstream.dropped.wait(3)
    # The abandoned request no longer holds off the idle timer.
    assert wait_for(lambda: proxy.activity()[0] == 0)


def test_client_leaving_mid_stream_ends_the_upstream_request(proxy, upstream):
    connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=3)
    connection.request("GET", "/events")
    response = connection.getresponse()
    assert response.readline() == b'data: {"n": 1}\n'
    assert proxy.activity()[0] == 1
    connection.sock.shutdown(socket.SHUT_RDWR)
    connection.close()
    upstream.gate.set()
    assert wait_for(lambda: proxy.activity()[0] == 0)


def test_proxy_streams_request_bodies_without_buffering(proxy, upstream):
    def body():
        yield b"first-part;"
        # The upstream sees the first part before the client sends the second.
        assert upstream.first_part.wait(3)
        yield b"second-part"

    connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    connection.request("POST", "/upload", body=body(), encode_chunked=True)
    seen = json.load(connection.getresponse())
    assert seen["body"] == "first-part;second-part"
    assert seen["chunked"] is True
    connection.close()


def test_proxy_keeps_client_connections_alive_between_requests(proxy):
    connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    for n in range(3):
        connection.request("POST", "/echo", body=str(n).encode())
        assert json.load(connection.getresponse())["body"] == str(n)
    connection.close()


def test_proxy_serves_concurrent_requests_in_parallel(proxy):
    started = time.monotonic()
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: get(proxy, "/slow?0.5"), range(8)))
    assert all(r["path"] == "/slow?0.5" for r in results)
    assert time.monotonic() - started < 3


def test_slow_first_byte_is_not_a_proxy_timeout(proxy):
    with patch.object(proxy_module, "CONNECT_TIMEOUT", 0.2):
        assert get(proxy, "/slow?1")["path"] == "/slow?1"


def test_activity_counts_requests_in_flight_and_resets_on_each_request(proxy):
    active, before = proxy.activity()
    assert active == 0
    worker = threading.Thread(target=get, args=(proxy, "/slow?0.5"))
    worker.start()
    deadline = time.monotonic() + 5
    while proxy.activity()[0] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert proxy.activity()[0] == 1
    worker.join()
    # The handler finishes its bookkeeping just after the client has the response.
    deadline = time.monotonic() + 5
    while proxy.activity()[0] != 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    active, after = proxy.activity()
    assert active == 0 and after > before


def test_unconnected_proxy_answers_service_unavailable():
    engine_proxy = EngineProxy(0)
    engine_proxy.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(engine_proxy, "/health")
        assert error.value.code == 503
        error.value.close()
    finally:
        engine_proxy.close()


def test_unreachable_remote_answers_bad_gateway():
    messages = []
    engine_proxy = EngineProxy(0, messages.append)
    engine_proxy.start()
    engine_proxy.connect(Upstream("127.0.0.1", free_port()), "remote-secret")
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(engine_proxy, "/health")
        assert error.value.code == 502
        error.value.close()
        assert any("/health failed" in m for m in messages)
    finally:
        engine_proxy.close()


def test_close_ends_streams_in_flight_and_releases_the_port(proxy):
    connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
    connection.request("GET", "/events")
    response = connection.getresponse()
    assert response.readline() == b'data: {"n": 1}\n'
    started = time.monotonic()
    proxy.close()
    with pytest.raises(http.client.IncompleteRead):
        response.read()
    assert time.monotonic() - started < 2
    connection.close()
    # A new proxy for the next launch can listen on the same port.
    replacement = EngineProxy(proxy.port)
    replacement.start()
    replacement.close()
