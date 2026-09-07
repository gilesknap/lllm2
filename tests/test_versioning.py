"""Build real releases from tagged repositories, including the sdist round trip."""

from email.parser import BytesParser
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class TaggedBuildTests(unittest.TestCase):
    def test_tag_sets_wheel_version_via_sdist(self):
        for tag, expected in [('0.1.1', '0.1.1'), ('v0.2.0', '0.2.0')]:
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'source'
                source.mkdir()
                for name in ('pyproject.toml', 'README.md', 'LICENSE', 'NOTICE'):
                    shutil.copy2(ROOT / name, source / name)
                shutil.copytree(ROOT / 'lllm2', source / 'lllm2',
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

                def run(*args):
                    result = subprocess.run(args, cwd=source, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

                run('git', 'init', '-q')
                run('git', 'add', '.')
                run('git', '-c', 'user.name=Release test', '-c',
                    'user.email=release-test@example.invalid', '-c',
                    'commit.gpgsign=false', 'commit', '-qm', 'Release fixture')
                run('git', '-c', 'tag.gpgsign=false', 'tag', tag)
                # The default build makes an sdist, then builds the wheel from it.
                # This also verifies version preservation without a .git directory.
                run(sys.executable, '-m', 'build', '--no-isolation')
                dist = source / 'dist'
                self.assertTrue((dist / f'lllm2-{expected}.tar.gz').is_file())
                wheel = dist / f'lllm2-{expected}-py3-none-any.whl'
                self.assertTrue(wheel.is_file(), list(dist.iterdir()))
                with zipfile.ZipFile(wheel) as archive:
                    metadata = BytesParser().parsebytes(
                        archive.read(f'lllm2-{expected}.dist-info/METADATA'))
                self.assertEqual(metadata['Version'], expected)
