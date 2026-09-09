"""Deleting experiment history is scoped, atomic, and preserves saved settings."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lllm2.app import App
from lllm2.store import Store


class ResultDeletionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        with patch("lllm2.config.STATE_DIR", Path(temp.name)):
            self.store = Store()
        self.addCleanup(self.store.db.close)
        self.app = App.__new__(App)
        self.app.store = self.store
        self.app.bench = Mock(lock=threading.Lock(), active=False)
        for key, status in [
            ("ok", "complete"),
            ("failed", "failed"),
            ("cancelled", "cancelled"),
            ("interrupted", "interrupted"),
            ("running", "running"),
        ]:
            self.store.put(
                "result",
                key,
                {
                    "id": key,
                    "status": status,
                    "samples": [{"prompt": "private source", "status": "failed"}],
                    "probes": [{"sample": {"large": "payload"}}],
                    "logs": ["log"],
                },
            )
        self.store.put("default", "saved", {"context": 4096})
        self.store.put(
            "default-evidence",
            "saved",
            {"result_id": "ok", "settings": {"context": 4096}},
        )
        self.store.put("catalogue", "model", {"id": "model"})

    def delete(self, ids, **extra):
        return self.app.action("/api/results/delete", {"result_ids": ids, **extra})

    def test_individual_removes_full_result_and_summary_only(self):
        self.assertEqual(self.delete(["ok"]), {"deleted": ["ok"]})
        for kind in ("result", "summary"):
            self.assertIsNone(self.store.get(kind, "ok"))
            self.assertIsNotNone(self.store.get(kind, "failed"))
        self.assertEqual(self.store.get("default", "saved"), {"context": 4096})
        self.assertIsNotNone(self.store.get("default-evidence", "saved"))
        self.assertIsNotNone(self.store.get("catalogue", "model"))
        self.assertEqual(self.delete(["ok"]), {"deleted": []})

    def test_bulk_only_deletes_confirmed_failed_cancelled_runs(self):
        self.assertEqual(
            self.delete(["failed", "cancelled"], failed_only=True),
            {"deleted": ["failed", "cancelled"]},
        )
        self.assertEqual(
            {r["id"] for r in self.store.list("result")},
            {"ok", "running", "interrupted"},
        )
        # Failed samples within a completed run do not make the run eligible.
        with self.assertRaises(ValueError):
            self.delete(["ok"], failed_only=True)

    def test_status_change_rejects_entire_batch_without_partial_deletion(self):
        with self.assertRaises(ValueError):
            self.delete(["failed", "ok"], failed_only=True)
        for key in ("failed", "ok"):
            self.assertIsNotNone(self.store.get("result", key))
            self.assertIsNotNone(self.store.get("summary", key))

    def test_running_runs_and_active_operations_are_protected(self):
        with self.assertRaises(ValueError):
            self.delete(["failed", "running"])
        self.assertIsNotNone(self.store.get("result", "failed"))
        self.app.bench.active = True
        with self.assertRaisesRegex(ValueError, "current operation"):
            self.delete(["ok"])
        self.assertIsNotNone(self.store.get("result", "ok"))

    def test_interrupted_individual_and_duplicate_ids(self):
        self.assertEqual(
            self.delete(["interrupted", "interrupted"]), {"deleted": ["interrupted"]}
        )

    def test_invalid_selection_is_rejected(self):
        for ids in (None, "ok", [], [None], [""], [3]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.delete(ids)
