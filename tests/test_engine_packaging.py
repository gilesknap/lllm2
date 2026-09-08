"""Packaging stores a library once even when aliases have been dereferenced."""

import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path

from lllm2.engine_install import _unpack

package_engine = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/package-engine.py")
)["package_engine"]


class EnginePackagingTests(unittest.TestCase):
    def test_symlinks_and_copied_aliases_share_one_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, installed = root / "source", root / "installed"
            source.mkdir()
            installed.mkdir()
            real = source / "libggml-cuda.so.0.23.0"
            real.write_bytes(b"unique library payload")
            real.chmod(0o755)
            (source / "libggml-cuda.so.0").symlink_to(real.name)
            (source / "libggml-cuda.so").symlink_to("libggml-cuda.so.0")
            (source / "libcopied.so").write_bytes(real.read_bytes())
            (source / "llama-server").write_bytes(b"server")
            archive = root / "engine.tar.gz"
            package_engine(source, archive)
            with tarfile.open(archive) as bundle:
                members = bundle.getmembers()
                payloads = [m for m in members if ".so" in m.name and m.isfile()]
                self.assertEqual([m.name for m in payloads], [real.name])
                self.assertEqual(sum(m.issym() for m in members), 3)
            _unpack(archive, installed)
            for name in ("libggml-cuda.so", "libggml-cuda.so.0", "libcopied.so"):
                self.assertTrue((installed / name).is_symlink())
                self.assertEqual((installed / name).read_bytes(), real.read_bytes())
            self.assertEqual((installed / real.name).stat().st_mode & 0o777, 0o755)

    def test_external_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (root / "external.so").write_bytes(b"outside")
            (source / "libexternal.so").symlink_to("../external.so")
            with self.assertRaisesRegex(ValueError, "flat local files"):
                package_engine(source, root / "engine.tar.gz")
