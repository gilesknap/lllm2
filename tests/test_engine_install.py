"""Release downloads are verified before any engine becomes discoverable."""

import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from lllm2 import engine_install as installer
from lllm2.engine_release import CUDA_TRACKS, LLAMA_CPP_REF, asset_name


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name, value in (("__version__", "0.3.0"),):
            patcher = patch.object(installer, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in (("system", "Linux"), ("machine", "x86_64")):
            patcher = patch.object(installer.platform, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.metadata = {
            "requested_ref": LLAMA_CPP_REF,
            "cuda_track": CUDA_TRACKS["13"],
            "lllm2_version": "0.3.0",
            "backend": "cuda",
            "architecture": "x86_64",
            "glibc": "2.28",
        }

    def archive(self, extra=None):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
            for name, data in {
                "./llama-server": b"#!/bin/sh\nexit 0\n",
                "./lllm2-engine.json": json.dumps(self.metadata).encode(),
                "./libggml-cuda.so": b"library",
            }.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mode = 0o755
                bundle.addfile(member, io.BytesIO(data))
            if extra:
                bundle.addfile(extra)
        return buffer.getvalue()

    def install(self, archive=None, checksum=None, probe_code=0):
        archive = self.archive() if archive is None else archive
        digest = checksum or hashlib.sha256(archive).hexdigest()

        def download(url, destination):
            destination.write_bytes(
                f"{digest}  {asset_name('13')}\n".encode()
                if url == "checksum"
                else archive
            )

        with (
            patch.object(installer, "cuda_track", return_value="13"),
            patch.object(
                installer, "_release_asset_urls", return_value=("archive", "checksum")
            ),
            patch.object(installer, "_download", side_effect=download),
            patch.object(
                installer.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], probe_code, stderr="probe failed"
                ),
            ) as run,
        ):
            binary = installer.install("cuda", root=self.root)
            if run.called:
                self.assertEqual(run.call_args.args[0][-1], "--help")
                self.assertEqual(
                    run.call_args.kwargs["env"]["LD_LIBRARY_PATH"].split(os.pathsep)[0],
                    str(Path(run.call_args.args[0][0]).parent),
                )
            return binary

    def test_install_and_repeat_preserve_existing_engine(self):
        old = self.root / "old-vulkan"
        old.mkdir()
        (old / "llama-server").write_text("old")
        binary = self.install()
        self.assertTrue(binary.is_file())
        self.assertTrue(installer.provenance(binary)["matches_release"])
        with (
            patch.object(installer, "cuda_track", return_value="13"),
            patch.object(installer, "_release_asset_urls") as lookup,
            patch.object(installer, "_download") as download,
        ):
            self.assertEqual(installer.install("cuda", root=self.root), binary)
            lookup.assert_not_called()
            download.assert_not_called()
        self.assertEqual((old / "llama-server").read_text(), "old")
        self.assertFalse(list(self.root.glob(".*")))

    def test_failed_checksum_and_probe_leave_no_engine(self):
        for kwargs, message in (
            ({"checksum": "0" * 64}, "checksum verification"),
            ({"probe_code": 1}, "could not start"),
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.install(**kwargs)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_interrupted_download_cleans_staging(self):
        def download(url, destination):
            destination.write_bytes(b"partial")
            raise RuntimeError("connection lost")

        with (
            patch.object(installer, "cuda_track", return_value="13"),
            patch.object(
                installer, "_release_asset_urls", return_value=("archive", "checksum")
            ),
            patch.object(installer, "_download", side_effect=download),
            self.assertRaisesRegex(RuntimeError, "connection lost"),
        ):
            installer.install("cuda", root=self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_download_streams_bytes_and_reports_network_failure(self):
        destination = self.root / "download"
        with patch.object(
            installer.urllib.request, "urlopen", return_value=io.BytesIO(b"downloaded")
        ):
            installer._download("https://example.invalid/engine", destination)
        self.assertEqual(destination.read_bytes(), b"downloaded")
        with (
            patch.object(
                installer.urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("offline"),
            ),
            self.assertRaisesRegex(RuntimeError, "Could not download"),
        ):
            installer._download("https://example.invalid/engine", destination)

    def test_same_engine_is_reused_across_python_releases(self):
        self.metadata["lllm2_version"] = "0.2.0"
        binary = self.install()
        record = installer.provenance(binary)
        self.assertEqual(record["built_for_lllm2_version"], "0.2.0")
        self.assertEqual(record["lllm2_version"], "0.3.0")
        with patch.object(installer, "__version__", "0.4.0"):
            self.assertTrue(installer.provenance(binary)["matches_release"])
            self.assertEqual(self.install(), binary)
        self.assertEqual(installer.provenance(binary)["lllm2_version"], "0.3.0")

    def test_wrong_engine_pins_are_rejected(self):
        for key, value in (
            ("requested_ref", "b1"),
            ("cuda_track", "12.0.0"),
            ("architecture", "aarch64"),
            ("glibc", "2.35"),
        ):
            original = self.metadata[key]
            self.metadata[key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RuntimeError, "metadata"),
            ):
                self.install()
            self.assertEqual(list(self.root.iterdir()), [])
            self.metadata[key] = original

    def test_unsafe_archive_members_rejected(self):
        for name, kind in (
            ("../escape", tarfile.REGTYPE),
            ("/escape", tarfile.REGTYPE),
            ("link", tarfile.SYMTYPE),
            ("hard", tarfile.LNKTYPE),
            ("fifo", tarfile.FIFOTYPE),
            ("./llama-server", tarfile.REGTYPE),
        ):
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "../escape"
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(RuntimeError, "Unsafe"),
            ):
                self.install(archive=self.archive(member))
            self.assertEqual(list(self.root.iterdir()), [])

    def test_internal_library_symlink_is_preserved(self):
        member = tarfile.TarInfo("./libggml-cuda.so.0")
        member.type = tarfile.SYMTYPE
        member.linkname = "libggml-cuda.so"
        binary = self.install(archive=self.archive(member))
        link = binary.with_name("libggml-cuda.so.0")
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.read_bytes(), b"library")
        self.assertEqual(os.readlink(link), "libggml-cuda.so")

    def test_dangling_and_cyclic_links_are_rejected(self):
        for target in ("missing", "cycle"):
            member = tarfile.TarInfo("cycle")
            member.type = tarfile.SYMTYPE
            member.linkname = target
            with (
                self.subTest(target=target),
                self.assertRaisesRegex(RuntimeError, "Unsafe"),
            ):
                self.install(archive=self.archive(member))
            self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_different_engine_is_not_overwritten(self):
        binary = self.install()
        binary.with_name("lllm2-engine.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            self.install()
        self.assertTrue(binary.is_file())

    def test_invalid_names_and_backend(self):
        for name in ("..", ".", "../outside", "/absolute", "has space"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                installer.install("cuda", name=name, root=self.root)
        with self.assertRaises(ValueError):
            installer.install("vulkan", root=self.root)

    def test_driver_track_selection_and_errors(self):
        for version, track in (
            ("12.9", "12"),
            ("13.0", "12"),
            ("13.2", "12"),
            ("13.3", "13"),
            ("14.0", "13"),
        ):
            with patch.object(
                installer.subprocess,
                "run",
                side_effect=lambda command, version=version, **kwargs: (
                    subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=f"CUDA Version: {version}"
                        if len(command) == 1
                        else "8.6\n",
                    )
                ),
            ):
                self.assertEqual(installer.cuda_track(), track)
        for output in (
            "CUDA Version: 11.8",
            "CUDA Version: 12.0",
            "CUDA Version: 12.8",
            "CUDA Version: N/A",
            "",
        ):
            with (
                patch.object(
                    installer.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, stdout=output),
                ),
                self.assertRaisesRegex(RuntimeError, "NVIDIA driver"),
            ):
                installer.cuda_track()
        with (
            patch.object(installer.subprocess, "run", side_effect=FileNotFoundError),
            self.assertRaisesRegex(RuntimeError, "NVIDIA driver"),
        ):
            installer.cuda_track()

    def test_driver_floors_follow_changed_cuda_pins(self):
        with (
            patch.dict(installer.CUDA_TRACKS, {"13": "13.4.1", "12": "12.10.1"}),
            patch.object(
                installer.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 0, stdout="CUDA Version: 13.3"
                ),
            ) as run,
        ):
            self.assertEqual(installer.cuda_track(), "12")
            self.assertEqual(run.call_count, 1)
            run.return_value.stdout = "CUDA Version: 12.9"
            with self.assertRaisesRegex(RuntimeError, "CUDA 12.10"):
                installer.cuda_track()

    def test_old_driver_rejected_before_release_lookup_or_download(self):
        with (
            patch.object(
                installer.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 0, stdout="CUDA Version: 12.8"
                ),
            ),
            patch.object(installer, "_release_asset_urls") as lookup,
            patch.object(installer, "_download") as download,
            self.assertRaisesRegex(RuntimeError, "Update or install the NVIDIA driver"),
        ):
            installer.install("cuda", root=self.root)
        lookup.assert_not_called()
        download.assert_not_called()

    def test_old_gpu_on_cuda13_driver_uses_cuda12(self):
        for capability, expected in (
            ("5.2", "12"),
            ("6.1", "12"),
            ("7.0", "12"),
            ("7.5", "13"),
            ("8.6", "13"),
            ("12.0", "13"),
            ("8.6\n7.0", "12"),
        ):
            with patch.object(
                installer.subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="CUDA Version: 13.3"),
                    subprocess.CompletedProcess([], 0, stdout=capability),
                ],
            ):
                self.assertEqual(installer.cuda_track(), expected)
        with (
            patch.object(
                installer.subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="CUDA Version: 13.3"),
                    subprocess.CompletedProcess([], 0, stdout="N/A"),
                ],
            ),
            self.assertRaisesRegex(RuntimeError, "compute capability"),
        ):
            installer.cuda_track()

    def test_release_search_crosses_pages_and_ignores_python_version(self):
        asset = asset_name("13")
        release = {
            "assets": [
                {
                    "name": name,
                    "state": "uploaded",
                    "browser_download_url": "https://example/" + name,
                }
                for name in (asset, asset + ".sha256")
            ]
        }
        first_page = [
            {"assets": []},
            dict(release, draft=True),
            dict(release, prerelease=True),
            {"assets": release["assets"][:1]},
        ]
        with (
            patch.object(installer, "__version__", "0.9.0.dev1"),
            patch.object(
                installer.urllib.request,
                "urlopen",
                side_effect=[
                    io.BytesIO(json.dumps(page).encode())
                    for page in (first_page, [release])
                ],
            ) as request,
        ):
            self.assertEqual(
                installer._release_asset_urls(asset)[0], "https://example/" + asset
            )
            self.assertTrue(request.call_args.args[0].full_url.endswith("page=2"))

    def test_missing_release_assets(self):
        with (
            patch.object(
                installer.urllib.request, "urlopen", return_value=io.BytesIO(b"[]")
            ),
            self.assertRaisesRegex(RuntimeError, "No published release"),
        ):
            installer._release_asset_urls(asset_name("13"))

    def test_release_api_error_is_not_treated_as_missing_pins(self):
        with (
            patch.object(
                installer.urllib.request,
                "urlopen",
                side_effect=urllib.error.HTTPError(
                    "url", 403, "rate limited", None, None
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "Could not find published"),
        ):
            installer._release_asset_urls(asset_name("13"))

    def test_malformed_or_old_provenance_is_unmarked(self):
        binary = self.root / "llama-server"
        for data in ("[]", "not json", '{"requested_ref": "master"}'):
            binary.with_name("lllm2-engine.json").write_text(data)
            self.assertFalse(installer.provenance(binary).get("matches_release"))
