"""Check for newer lllm2 releases without slowing or breaking the panel."""

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from lllm2 import config
from lllm2 import update_check as updates
from lllm2.engine_release import LLAMA_CPP_REF
from lllm2.store import Store


class VersionTests(unittest.TestCase):
    def test_newer_final_release(self):
        for latest, current, expected in [
            ("0.4.0", "0.3.0", True),
            ("v0.4.0", "0.3.9", True),
            ("0.3.0", "0.3.0", False),
            ("0.3", "0.3.0", False),
            ("0.2.9", "0.3.0", False),
            ("0.10.0", "0.9.0", True),
            # A development build is ahead of its last tag and behind the next.
            ("0.3.0", "0.3.1.dev4+g1234abc", False),
            ("0.3.1", "0.3.1.dev4+g1234abc", True),
            ("0.4.0", "0.3.1.dev4+g1234abc.d20261001", True),
            # Pre-release candidates sort before their final release.
            ("0.4.0", "0.4.0rc1", True),
            ("0.3.9", "0.4.0rc1", False),
            # Pre-releases and unreadable versions never claim an update.
            ("0.5.0rc1", "0.4.0", False),
            ("latest", "0.4.0", False),
            ("0.5.0", "unknown", False),
            ("0.5.0", "0.4.0+local", True),
        ]:
            with self.subTest(latest=latest, current=current):
                self.assertEqual(updates.is_newer(latest, current), expected)

    def test_engine_ref_from_asset_names(self):
        assets = [
            {"name": "lllm2-0.4.0-py3-none-any.whl"},
            {"name": "lllm2-engine-b12000-cuda13.3.1-el8-x64.tar.gz"},
        ]
        self.assertEqual(updates.engine_ref(assets), "b12000")
        self.assertIsNone(updates.engine_ref([{"name": "notes.txt"}]))


class FetchTests(unittest.TestCase):
    def test_reads_latest_release(self):
        body = {
            "tag_name": "v0.4.0",
            "html_url": "https://github.com/gilesknap/lllm2/releases/tag/v0.4.0",
            "draft": False,
            "prerelease": False,
            "assets": [{"name": "lllm2-engine-b12000-cuda12.9.1-el8-x64.tar.gz"}],
        }
        with patch.object(
            updates.urllib.request,
            "urlopen",
            return_value=io.BytesIO(json.dumps(body).encode()),
        ) as urlopen:
            release = updates.fetch_latest()
        self.assertEqual(
            urlopen.call_args.args[0].full_url,
            "https://api.github.com/repos/gilesknap/lllm2/releases/latest",
        )
        self.assertEqual(
            release,
            {"version": "0.4.0", "url": body["html_url"], "llama_cpp": "b12000"},
        )


class UpdateCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patcher = patch.object(config, "STATE_DIR", Path(temporary.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = Store()
        self.addCleanup(self.store.db.close)
        self.now = 1_000_000.0
        self.calls = 0
        self.release = {"version": "0.4.0", "url": "https://example", "llama_cpp": "b1"}

    def fetch(self):
        self.calls += 1
        if isinstance(self.release, Exception):
            raise self.release
        return self.release

    def check(self, **kwargs):
        return updates.UpdateCheck(
            self.store,
            enabled=kwargs.get("enabled", True),
            fetch=self.fetch,
            clock=lambda: self.now,
            version=kwargs.get("version", "0.3.0"),
        )

    def settle(self, check):
        """Run the background refresh to completion before asserting."""
        if check.thread:
            check.thread.join(timeout=5)
        self.assertFalse(check.running)

    def test_first_load_answers_without_waiting_then_shows_the_release(self):
        check = self.check()
        self.assertIsNone(check.notice())
        self.settle(check)
        self.assertEqual(
            check.notice(),
            {
                "version": "0.4.0",
                "url": "https://example",
                "llama_cpp": "b1",
                "upgrade_command": "uv tool upgrade lllm2",
                "pip_upgrade_command": "pip install --upgrade lllm2",
                "engine_command": "lllm2 engines install cuda",
            },
        )
        self.assertEqual(self.calls, 1)

    def test_cache_is_reused_for_a_day_and_survives_restart(self):
        first = self.check()
        first.notice()
        self.settle(first)
        self.now += updates.CHECK_INTERVAL - 1
        restarted = self.check()
        self.assertEqual(restarted.notice()["version"], "0.4.0")
        self.settle(restarted)
        self.assertEqual(self.calls, 1)
        self.now += 1
        self.release = {**self.release, "version": "0.5.0"}
        restarted.notice()
        self.settle(restarted)
        self.assertEqual(restarted.notice()["version"], "0.5.0")
        self.assertEqual(self.calls, 2)

    def test_failures_are_silent_and_retry_later(self):
        self.release = urllib.error.URLError("offline")
        check = self.check()
        self.assertIsNone(check.notice())
        self.settle(check)
        self.assertIsNone(check.notice())
        self.settle(check)
        self.assertEqual(self.calls, 1)
        self.now += updates.RETRY_INTERVAL
        check.notice()
        self.settle(check)
        self.assertEqual(self.calls, 2)

    def test_no_notice_when_current_or_disabled(self):
        check = self.check(version="0.4.1.dev2+gabc")
        check.notice()
        self.settle(check)
        self.assertIsNone(check.notice())
        disabled = self.check(enabled=False)
        self.assertIsNone(disabled.notice())
        self.assertEqual(self.calls, 1)

    def test_same_engine_pin_needs_no_engine_download(self):
        self.release = {**self.release, "llama_cpp": LLAMA_CPP_REF}
        check = self.check()
        check.notice()
        self.settle(check)
        self.assertIsNone(check.notice()["engine_command"])


class OptOutTests(unittest.TestCase):
    def test_environment_variable_disables_the_check(self):
        for value, expected in [("0", False), ("off", False), ("1", True)]:
            with self.subTest(value=value):
                environ = {"LLLM2_UPDATE_CHECK": value}
                self.assertEqual(config.update_check_enabled(environ), expected)
        self.assertTrue(config.update_check_enabled({}))
