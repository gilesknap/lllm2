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
