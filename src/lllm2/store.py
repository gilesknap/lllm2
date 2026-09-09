import json
import sqlite3
import threading

from . import config


class Store:
    def __init__(self):
        config.STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(
            config.STATE_DIR / "workbench.sqlite3", check_same_thread=False
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS objects (kind TEXT, key TEXT, value TEXT, PRIMARY KEY(kind,key))"
        )
        self.db.commit()
        for r in self.list("result"):
            if r["status"] == "running":
                r.update(
                    status="interrupted",
                    error="Panel stopped before this experiment completed.",
                )
                self.put("result", r["id"], r)

    def put(self, kind, key, value):
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO objects VALUES (?,?,?)",
                (kind, key, json.dumps(value)),
            )
            if kind == "result":
                summary = {k: v for k, v in value.items() if k != "logs"}
                summary["samples"] = [
                    {
                        k: v
                        for k, v in s.items()
                        if k not in ["prompt", "output", "memory"]
                    }
                    for s in value["samples"]
                ]
                summary["probes"] = [
                    {k: v for k, v in p.items() if k not in ["sample", "logs"]}
                    for p in value["probes"]
                ]
                self.db.execute(
                    "INSERT OR REPLACE INTO objects VALUES (?,?,?)",
                    ("summary", key, json.dumps(summary)),
                )
            self.db.commit()

    def get(self, kind, key):
        with self.lock:
            row = self.db.execute(
                "SELECT value FROM objects WHERE kind=? AND key=?", (kind, key)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def list(self, kind):
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT value FROM objects WHERE kind=? ORDER BY rowid DESC",
                    (kind,),
                )
            ]

    def delete(self, kind, key):
        with self.lock, self.db:
            self.db.execute("DELETE FROM objects WHERE kind=? AND key=?", (kind, key))

    def delete_results(self, result_ids, failed_only=False):
        """Delete full records and summaries together, after validating every run."""
        if (
            not isinstance(result_ids, list)
            or not 1 <= len(result_ids) <= 10000
            or any(not isinstance(key, str) or not key for key in result_ids)
        ):
            raise ValueError("Select between 1 and 10,000 experiment runs.")
        allowed = (
            {"failed", "cancelled"}
            if failed_only
            else {"complete", "failed", "cancelled", "interrupted"}
        )
        with self.lock, self.db:
            found = []
            for key in dict.fromkeys(result_ids):
                row = self.db.execute(
                    "SELECT value FROM objects WHERE kind='result' AND key=?", (key,)
                ).fetchone()
                if row is None:
                    continue
                if json.loads(row[0]).get("status") not in allowed:
                    raise ValueError(
                        "An experiment is running or its status has changed. Refresh history and try again."
                    )
                found.append(key)
            self.db.executemany(
                "DELETE FROM objects WHERE kind IN ('result','summary') AND key=?",
                [(key,) for key in found],
            )
        return found
