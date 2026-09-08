"""Native CA discovery must preserve certificate and hostname verification."""

import os
import shutil
import ssl
import struct
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from lllm2 import downloads, engine_install, tls


class ContextTests(unittest.TestCase):
    def test_explicit_overrides_bypass_automatic_bundle(self):
        for variable in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            for value in ("", "/custom/trust"):
                with (
                    self.subTest(variable=variable, value=value),
                    patch.dict(os.environ, {variable: value}, clear=True),
                    patch.object(tls.ssl, "create_default_context") as create,
                    patch.object(tls.Path, "is_file") as exists,
                ):
                    tls.download_context()
                    create.assert_called_once_with()
                    exists.assert_not_called()

    def test_missing_bundle_and_other_platforms_keep_python_defaults(self):
        for platform, exists in (("linux", False), ("darwin", True), ("win32", True)):
            with (
                self.subTest(platform=platform),
                patch.dict(os.environ, {}, clear=True),
                patch.object(tls.sys, "platform", platform),
                patch.object(tls.Path, "is_file", return_value=exists),
                patch.object(tls.ssl, "create_default_context") as create,
            ):
                tls.download_context()
                create.assert_called_once_with()


class Handler(BaseHTTPRequestHandler):
    def body(self):
        return (
            struct.pack("<4sIQQ", b"GGUF", 3, 0, 0) if self.path == "/model" else b"[]"
        )

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body())))
        self.end_headers()

    def do_GET(self):
        self.do_HEAD()
        self.wfile.write(self.body())

    def log_message(self, *_args):
        pass


@unittest.skipUnless(shutil.which("openssl"), "openssl needed for test certificates")
class DownloadTrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        # Two independent roots let us prove that an explicit override excludes
        # the automatically discovered root. Generate fresh, short-lived keys.
        for name in ("server", "other"):
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(cls.root / f"{name}.key"),
                    "-out",
                    str(cls.root / f"{name}.pem"),
                    "-days",
                    "2",
                    "-subj",
                    "/CN=localhost",
                    "-addext",
                    "subjectAltName=DNS:localhost",
                ],
                check=True,
                capture_output=True,
            )
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(cls.root / "server.pem", cls.root / "server.key")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.addClassCleanup(server.server_close)
        server.socket = server_context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cls.addClassCleanup(thread.join)
        cls.addClassCleanup(server.shutdown)
        cls.url = f"https://localhost:{server.server_port}/"

    def setUp(self):
        for patcher in (
            patch.dict(os.environ, {}, clear=True),
            patch.object(tls.sys, "platform", "linux"),
            patch.object(tls, "SYSTEM_CA_BUNDLE", self.root / "server.pem"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_engine_download_and_model_head_use_native_bundle(self):
        target = self.root / "download"
        engine_install._download(self.url, target)
        self.assertEqual(target.read_bytes(), b"[]")
        with patch.object(downloads, "url_for", return_value=self.url):
            self.assertEqual(downloads._size_of("repo", "model"), 2)

    def test_model_fetch_uses_native_bundle(self):
        target = self.root / "model.gguf"
        download = downloads.Download("id", "name", "repo", "model", target)
        with patch.object(downloads, "url_for", return_value=self.url + "model"):
            self.assertTrue(downloads._fetch(download, "model", target, 0))
        self.assertEqual(target.read_bytes(), struct.pack("<4sIQQ", b"GGUF", 3, 0, 0))

    def test_release_lookup_uses_native_bundle(self):
        request = urllib.request.Request
        with (
            patch.object(
                engine_install.urllib.request,
                "Request",
                side_effect=lambda _url, **kwargs: request(self.url, **kwargs),
            ),
            self.assertRaisesRegex(RuntimeError, "No published release"),
        ):
            engine_install._release_asset_urls("missing")

    def test_untrusted_server_is_rejected(self):
        with (
            patch.object(tls, "SYSTEM_CA_BUNDLE", self.root / "other.pem"),
            self.assertRaisesRegex(RuntimeError, "CERTIFICATE_VERIFY_FAILED"),
        ):
            engine_install._download(self.url, self.root / "untrusted")

    def test_wrong_hostname_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CERTIFICATE_VERIFY_FAILED"):
            engine_install._download(
                self.url.replace("localhost", "127.0.0.1"), self.root / "wrong-host"
            )

    def test_explicit_bundle_does_not_add_native_roots(self):
        with (
            patch.dict(os.environ, {"SSL_CERT_FILE": str(self.root / "other.pem")}),
            self.assertRaises(urllib.error.URLError),
        ):
            with urllib.request.urlopen(self.url, context=tls.download_context()):
                self.fail("The explicit CA bundle should reject this server")

    def test_explicit_trusted_bundle_works(self):
        with (
            patch.dict(os.environ, {"SSL_CERT_FILE": str(self.root / "server.pem")}),
            patch.object(tls, "SYSTEM_CA_BUNDLE", self.root / "other.pem"),
        ):
            engine_install._download(self.url, self.root / "explicit")

    def test_invalid_native_bundle_is_reported(self):
        invalid = self.root / "invalid.pem"
        invalid.write_text("not a certificate")
        with (
            patch.object(tls, "SYSTEM_CA_BUNDLE", invalid),
            self.assertRaises(ssl.SSLError),
        ):
            tls.download_context()
