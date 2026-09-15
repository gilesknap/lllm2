"""A loopback HTTP proxy that forwards the engine port to a remote llama-server.

The proxy keeps ``Engine.base`` a loopback URL for remote engines, so bench,
warm, harness wrappers and other clients need no changes. It streams request
and response bodies in both directions without buffering, so server-sent event
framing reaches the client byte for byte. It removes client credentials and adds
the remote server's API key to every forwarded request.

The key does not protect every path. llama-server answers ``/health``,
``/v1/health`` and its web UI files, such as ``/``, without it. Anyone who
learns a provider tunnel's address can check that the server is up and load the
web UI page, but the UI's requests need the key. Every other path, including
``/v1/models``, ``/props``, completion, tokenization and template application,
needs the key. The proxy itself listens only on 127.0.0.1.
"""

import http.client
import json
import select
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Headers that describe one connection, not the message (RFC 9110 section 7.6.1),
# plus Expect, which the local server already answers.
HOP_BY_HOP = {
    "connection",
    "expect",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
# Client credentials. Clients send a placeholder token; the proxy sends the real key.
CREDENTIALS = {"authorization", "x-api-key"}
CHUNK = 64 * 1024
# Bounds only the connection to the remote server. Reads have no timeout because
# prompt processing can take minutes before the first response byte.
CONNECT_TIMEOUT = 30
# Seconds between checks for a client that left while the remote processes a prompt.
WATCH_INTERVAL = 0.25
# TCP keepalive: probe an idle upstream connection after a minute, give up after
# about two more, so a dead tunnel does not hold a request open forever.
KEEPALIVE_IDLE = 60
KEEPALIVE_INTERVAL = 20
KEEPALIVE_COUNT = 6


@dataclass(frozen=True)
class Upstream:
    """The address of a remote llama-server.

    Attributes:
        host: The host name or address.
        port: The TCP port.
        tls: True to connect with TLS, as an encrypted provider tunnel requires.
    """

    host: str
    port: int
    tls: bool = False


class EngineProxy:
    """Forward HTTP requests from a loopback port to a remote llama-server.

    Bind the port with ``start``, then set the remote address with ``connect``
    once the provider reports it. Until then the proxy answers 503. Each client
    request opens its own upstream connection, so concurrent requests do not
    wait for each other.
    """

    def __init__(self, port, on_log=None, clock=time.monotonic):
        """Create a proxy for a loopback port.

        Args:
            port: The loopback port to listen on. Zero picks a free port.
            on_log: A callable that receives one log line, or None.
            clock: A monotonic clock in seconds that stamps request activity.
                Tests replace it to move time without sleeping.
        """
        self.port = port
        self._on_log = on_log
        self._clock = clock
        self._lock = threading.Lock()
        self._upstream = None
        self._api_key = None
        self._server = None
        self._closed = False
        # Upstream connections in flight, mapped to their socket once connected.
        # http.client drops the socket from a connection whose response closes
        # it, so ``close`` needs its own reference.
        self._connections = {}
        self._active = 0
        self._last_activity = self._clock()

    def start(self):
        """Bind the loopback port and serve requests on a background thread.

        Raises:
            OSError: The port cannot be bound.
        """
        server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
        server.daemon_threads = True
        self.port = server.server_address[1]
        self._server = server
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        ).start()

    def connect(self, upstream, api_key):
        """Set the remote server address and the key to send with each request.

        Args:
            upstream: The remote ``Upstream`` address.
            api_key: The remote server's API key.
        """
        with self._lock:
            self._upstream, self._api_key = upstream, api_key
            self._last_activity = self._clock()

    def close(self):
        """Release the port and end every forwarded request in flight.

        A client with a response in flight sees its connection close before the
        response completes.
        """
        with self._lock:
            server, self._server = self._server, None
            self._upstream = self._api_key = None
            self._closed = True
            sockets = [s for s in self._connections.values() if s is not None]
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if server is not None:
            server.shutdown()
            server.server_close()

    def activity(self):
        """Return the proxy's request activity for an idle timer.

        Returns:
            A tuple of the number of requests in flight and the
            clock value when a request last started or ended.
        """
        with self._lock:
            return self._active, self._last_activity

    def _log(self, message):
        if self._on_log is not None:
            self._on_log(message)

    def _begin(self):
        with self._lock:
            upstream, key = self._upstream, self._api_key
            if upstream is None:
                return None, None
            if upstream.tls:
                connection = http.client.HTTPSConnection(
                    upstream.host,
                    upstream.port,
                    timeout=CONNECT_TIMEOUT,
                    context=ssl.create_default_context(),
                )
            else:
                connection = http.client.HTTPConnection(
                    upstream.host, upstream.port, timeout=CONNECT_TIMEOUT
                )
            self._connections[connection] = None
            self._active += 1
            self._last_activity = self._clock()
            return connection, key

    def _end(self, connection):
        connection.close()
        with self._lock:
            self._connections.pop(connection, None)
            self._active -= 1
            self._last_activity = self._clock()

    def _track(self, connection):
        with self._lock:
            if self._closed:
                raise ConnectionAbortedError("The engine proxy is closed.")
            self._connections[connection] = connection.sock

    def _is_closed(self):
        with self._lock:
            return self._closed

    def _handler(self):
        begin, end, track = self._begin, self._end, self._track
        log, closed = self._log, self._is_closed

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # noqa: N802
                self.forward()

            do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = do_GET  # noqa: N815

            def log_message(self, *args):
                pass

            def error(self, status, message):
                data = json.dumps(
                    {"error": {"code": status, "message": message, "type": "proxy"}}
                ).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def forward(self):
                length = self.headers.get("Content-Length")
                chunked = "chunked" in self.headers.get("Transfer-Encoding", "").lower()
                if length is not None and (chunked or not length.isdigit()):
                    self.close_connection = True
                    self.error(400, "Invalid request body framing.")
                    return
                connection, key = begin()
                if connection is None:
                    try:
                        for _ in body_chunks(self.rfile, length, chunked):
                            pass
                    except (OSError, ValueError):
                        self.close_connection = True
                        return
                    self.error(503, "Remote engine is not ready.")
                    return
                self.replied = self.abandoned = False
                try:
                    self.relay(connection, key, length, chunked)
                except (OSError, ValueError, http.client.HTTPException) as e:
                    self.close_connection = True
                    if self.abandoned:
                        log(f"Proxy: client left {self.command} {self.path} early.")
                    elif not self.replied and not closed():
                        log(f"Proxy: {self.command} {self.path} failed: {e}")
                        try:
                            self.error(502, f"Remote engine unreachable: {e}")
                        except OSError:
                            pass
                finally:
                    end(connection)

            def relay(self, connection, key, length, chunked):
                connection.putrequest(
                    self.command, self.path, skip_accept_encoding=True
                )
                for name, value in self.headers.items():
                    lower = name.lower()
                    if lower in HOP_BY_HOP or lower in CREDENTIALS:
                        continue
                    if lower not in ("host", "content-length"):
                        connection.putheader(name, value)
                connection.putheader("Authorization", f"Bearer {key}")
                if chunked:
                    connection.putheader("Transfer-Encoding", "chunked")
                elif length is not None:
                    connection.putheader("Content-Length", length)
                connection.endheaders()
                track(connection)
                for part in body_chunks(self.rfile, length, chunked):
                    connection.send(
                        b"%x\r\n%s\r\n" % (len(part), part) if chunked else part
                    )
                if chunked:
                    connection.send(b"0\r\n\r\n")
                if connection.sock is not None:
                    connection.sock.settimeout(None)
                    keep_alive(connection.sock)
                with ClientWatch(self.connection, connection.sock, self):
                    response = connection.getresponse()
                self.send_response_only(response.status, response.reason)
                self.replied = True
                has_body = self.command != "HEAD" and not (
                    100 <= response.status < 200 or response.status in (204, 304)
                )
                framed = has_body and response.getheader("Content-Length") is None
                for name, value in response.getheaders():
                    if name.lower() not in HOP_BY_HOP:
                        self.send_header(name, value)
                if framed:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                if not has_body:
                    return
                while part := response.read1(CHUNK):
                    self.wfile.write(
                        b"%x\r\n%s\r\n" % (len(part), part) if framed else part
                    )
                    self.wfile.flush()
                if closed():
                    # A stopped engine must not look like a complete response.
                    self.close_connection = True
                    return
                if framed:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()

        return Handler


def keep_alive(sock):
    """Enable TCP keepalive so a silently dead remote ends a waiting request.

    Args:
        sock: The upstream socket.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (
            ("TCP_KEEPIDLE", KEEPALIVE_IDLE),
            ("TCP_KEEPINTVL", KEEPALIVE_INTERVAL),
            ("TCP_KEEPCNT", KEEPALIVE_COUNT),
        ):
            if hasattr(socket, name):
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
    except OSError:
        pass


class ClientWatch:
    """End an upstream request if the client disconnects before the response starts.

    While the remote server processes a prompt, the proxy sends the client
    nothing, so it cannot see a disconnect by writing. A background thread
    watches the client socket instead. When the client closes it, the thread
    shuts the upstream socket down. The remote server then stops the request,
    and the request no longer holds off the idle timer.
    """

    def __init__(self, client, upstream, handler):
        """Prepare a watch for one request.

        Args:
            client: The client socket.
            upstream: The upstream socket, or None.
            handler: The request handler. Its ``abandoned`` attribute becomes
                True when the client disconnects.
        """
        self._client = client
        self._upstream = upstream
        self._handler = handler
        self._lock = threading.Lock()
        self._done = False

    def __enter__(self):
        if self._upstream is not None:
            threading.Thread(target=self._watch, daemon=True).start()
        return self

    def __exit__(self, *exc):
        with self._lock:
            self._done = True

    def _watch(self):
        while True:
            with self._lock:
                if self._done:
                    return
            try:
                readable, _, _ = select.select([self._client], [], [], WATCH_INTERVAL)
                if not readable:
                    continue
                if self._client.recv(1, socket.MSG_PEEK):
                    # A pipelined request; the client is still there.
                    return
            except (OSError, ValueError):
                pass
            with self._lock:
                if self._done:
                    return
                self._handler.abandoned = True
                try:
                    self._upstream.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            return


def body_chunks(stream, length, chunked):
    """Yield a request body in pieces as the client sends them.

    Args:
        stream: The client connection's buffered reader.
        length: The ``Content-Length`` header value, or None.
        chunked: True if the body uses chunked transfer coding.

    Yields:
        Byte strings of body data, without transfer coding.

    Raises:
        ConnectionError: The client closed the connection mid-body.
        ValueError: A chunk size line is malformed.
    """
    if not chunked:
        remaining = int(length or 0)
        while remaining:
            part = stream.read1(min(remaining, CHUNK))
            if not part:
                raise ConnectionError("Client closed during the request body.")
            remaining -= len(part)
            yield part
        return
    while True:
        line = stream.readline(1024)
        if not line:
            raise ConnectionError("Client closed during a chunked body.")
        size = int(line.split(b";", 1)[0].strip(), 16)
        if size == 0:
            # Discard trailers up to the blank line.
            while stream.readline(1024) not in (b"\r\n", b"\n", b""):
                pass
            return
        remaining = size
        while remaining:
            part = stream.read1(min(remaining, CHUNK))
            if not part:
                raise ConnectionError("Client closed during a chunked body.")
            remaining -= len(part)
            yield part
        stream.readline(1024)
