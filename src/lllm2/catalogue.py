"""Persistent catalogue and conservative metadata-only hardware suggestions."""

import hashlib
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from . import config
from .tls import download_context


def files(entry):
    return list(
        dict.fromkeys(
            entry.get("files")
            or [entry["file"]] + ([entry["mmproj"]] if entry.get("mmproj") else [])
        )
    )


def local_paths(entry):
    """Only explicitly owned GGUF files beneath the managed model directory."""
    root = config.MODELS_DIR.resolve()
    name = entry["name"]
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Invalid model directory.")
    paths = []
    for file in files(entry):
        p = PurePosixPath(file)
        if (
            p.is_absolute()
            or ".." in p.parts
            or "\\" in file
            or not file.lower().endswith(".gguf")
        ):
            raise ValueError("Invalid model file path.")
        target = root / name / file
        for candidate in (target, target.with_suffix(target.suffix + ".part")):
            if not candidate.resolve().is_relative_to(root):
                raise ValueError("Model path escapes the model directory.")
        paths.append(target)
    return paths


class Catalogue:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        # Seed and marker share a transaction: an empty catalogue stays empty.
        with store.lock, store.db:
            if not store.db.execute(
                "SELECT 1 FROM objects WHERE kind='migration' AND key='catalogue-v1'"
            ).fetchone():
                seeds = json.loads(Path(__file__).with_name("models.json").read_text())
                store.db.executemany(
                    "INSERT OR IGNORE INTO objects VALUES ('catalogue',?,?)",
                    [(e["id"], json.dumps(e)) for e in seeds],
                )
                store.db.execute(
                    "INSERT INTO objects VALUES ('migration','catalogue-v1','true')"
                )

    def list(self):
        return self.store.list("catalogue")

    def add(self, entry):
        with self.lock:
            existing = next(
                (
                    e
                    for e in self.list()
                    if e["repo"] == entry["repo"] and e["file"] == entry["file"]
                ),
                None,
            )
            if existing:
                return existing
            local_paths(entry)
            self.store.put("catalogue", entry["id"], entry)
            return entry

    def get(self, key):
        entry = self.store.get("catalogue", key)
        if not entry:
            raise ValueError("Unknown catalogue model.")
        return entry

    def remove(self, key):
        self.store.delete("catalogue", key)


def suitability(size, host):
    gpu = max((g["total_mib"] / 1024 for g in host.get("gpus", [])), default=0)
    ram = host.get("ram", {}).get("total_gib") or host.get("ram_gib") or 0
    if not size:
        return {"fit": "Unknown", "fit_rank": 2, "reason": "HF file size is missing."}
    gib = size / 2**30
    if gib <= max(0, gpu - 3) and ram and gib > max(0, ram - 4):
        return {
            "fit": "Unknown",
            "fit_rank": 2,
            "reason": "Weights fit the GPU budget but system RAM headroom is insufficient.",
        }
    if gib <= max(0, gpu - 3):
        return {
            "fit": "Likely GPU fit",
            "fit_rank": 0,
            "reason": f"{gib:.1f} GiB weights; {gpu:g} GiB on the largest GPU, reserving 3 GiB. Runtime support and context are untested.",
        }
    if gpu and gib <= max(0, ram - 4):
        return {
            "fit": "Likely needs CPU offload",
            "fit_rank": 1,
            "reason": f"{gib:.1f} GiB weights; {ram:g} GiB RAM, reserving 4 GiB. CPU offload may be slow; context is untested.",
        }
    return {
        "fit": "Too large" if ram and gpu else "Unknown",
        "fit_rank": 3 if ram and gpu else 2,
        "reason": "Exceeds the simple memory budget or no supported NVIDIA GPU was detected.",
    }


def metadata_get(path, params=None):
    url = "https://huggingface.co/api/models" + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "lllm2/find-models"})
    with urllib.request.urlopen(
        req, timeout=15, context=download_context()
    ) as response:
        raw = response.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ValueError("HF metadata response is too large.")
    return json.loads(raw)


def variants(info):
    """A repository can contain independent quants, shards and projectors."""
    if info.get("private") or info.get("gated") or info.get("disabled"):
        return []
    repo = info.get("id", "")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        return []
    revision = info.get("sha") or ""
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        return []
    siblings = {
        s["rfilename"]: s.get("size") or (s.get("lfs") or {}).get("size")
        for s in info.get("siblings", [])
        if s.get("rfilename", "").lower().endswith(".gguf")
    }
    projectors = [f for f in siblings if "mmproj" in f.lower()]
    tags = info.get("tags") or []
    task = info.get("pipeline_tag") or "Unknown"
    if task not in {"text-generation", "image-text-to-text", "Unknown"}:
        return []
    instruct = any(
        t in (repo + " " + " ".join(tags)).lower()
        for t in ("instruct", "chat", "coder", "coding", "conversational")
    ) or bool((info.get("config") or {}).get("chat_template"))
    card = info.get("cardData") or {}
    license_name = card.get("license") or next(
        (t[8:] for t in tags if t.startswith("license:")), "Unknown"
    )
    gguf = info.get("gguf") or {}
    groups = {}
    for file in siblings:
        if file in projectors or any(
            t in file.lower() for t in ("imatrix", "draft", "dflash")
        ):
            continue
        match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", file)
        key = file[: match.start()] if match else file
        groups.setdefault(key, []).append(file)
    out = []
    for key, group in groups.items():
        group.sort()
        issue = ""
        match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", group[0])
        if match:
            count = int(match.group(2))
            expected = [
                f"{key}-{i:05d}-of-{count:05d}.gguf"
                for i in range(1, min(count, 1000) + 1)
            ]
            if count > 1000 or group != expected:
                issue = "Incomplete split GGUF metadata."
        extras = projectors if len(projectors) == 1 else []
        if len(projectors) > 1 or (task == "image-text-to-text" and not projectors):
            issue = "Cannot identify a unique vision projector from HF metadata."
        wanted = group + extras
        size = (
            sum(siblings[f] for f in wanted)
            if all(isinstance(siblings[f], int) and siblings[f] > 0 for f in wanted)
            else None
        )
        quant = re.search(
            r"(?:IQ\d[^./-]*|Q\d[^./-]*|BF16|FP16|F16|FP32|F32)",
            PurePosixPath(key).name,
            re.I,
        )
        quant = quant.group().upper() if quant else "Unknown"
        identity = hashlib.sha256(
            (repo + "@" + revision + ":" + group[0]).encode()
        ).hexdigest()[:20]
        entry = {
            "id": "hf-" + identity,
            "name": repo.split("/")[1] + "-" + identity[:8],
            "display_name": repo.split("/")[1],
            "repo": repo,
            "revision": revision,
            "file": group[0],
            "files": wanted,
            "size_bytes": size,
            "size_gb": round(size / 1e9, 3) if size else None,
            "quant": quant,
            "task": "Coding"
            if any(t in repo.lower() for t in ("coder", "coding"))
            else "Chat / instruct"
            if instruct
            else task,
            "instruct": instruct,
            "downloads": info.get("downloads") or 0,
            "likes": info.get("likes") or 0,
            "updated": info.get("lastModified") or info.get("last_modified") or "",
            "license": str(license_name),
            "max_ctx": gguf.get("context_length"),
            "issue": issue,
            "source": "huggingface",
        }
        if extras:
            entry["mmproj"] = extras[0]
        try:
            local_paths(entry)
        except ValueError:
            continue
        out.append(entry)
    return out


class Finder:
    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()

    def search(self, query, host, refresh=False):
        query = str(query).strip()[:160]
        key = hashlib.sha256(query.encode()).hexdigest()
        with self.lock:
            cached = self.store.get("hf-search", key)
            if cached and not refresh and time.time() - cached["fetched_at"] < 3600:
                return self.present(cached, host)
            try:
                repos = metadata_get(
                    "",
                    {
                        "filter": "gguf",
                        "search": query,
                        "sort": "downloads",
                        "direction": -1,
                        "limit": 30,
                        "full": "true",
                    },
                )
                errors = []

                def fetch(repo):
                    try:
                        return variants(
                            metadata_get(
                                "/" + urllib.parse.quote(repo["id"], safe="/"),
                                {"blobs": "true"},
                            )
                        )
                    except (OSError, ValueError, KeyError) as error:
                        errors.append(str(error))
                        return []

                with ThreadPoolExecutor(max_workers=6) as pool:
                    entries = [
                        entry for group in pool.map(fetch, repos) for entry in group
                    ]
                if repos and len(errors) == len(repos):
                    raise ValueError(
                        "Could not retrieve HF file metadata. Try again later."
                    )
                result = {
                    "entries": entries,
                    "fetched_at": time.time(),
                    "warning": f"{len(errors)} repositories could not be inspected."
                    if errors
                    else "",
                    "repositories": len(repos),
                }
                self.store.put("hf-search", key, result)
                return self.present(result, host)
            except (OSError, ValueError) as error:
                if cached:
                    return self.present(
                        {
                            **cached,
                            "warning": f"HF unavailable; showing cached results. {error}",
                        },
                        host,
                    )
                raise ValueError(f"HF discovery unavailable: {error}") from error

    def present(self, result, host):
        return {
            **result,
            "entries": [
                {**e, **suitability(e.get("size_bytes"), host)}
                for e in result["entries"]
            ],
        }

    def candidate(self, key):
        for result in self.store.list("hf-search"):
            entry = next((e for e in result["entries"] if e["id"] == key), None)
            if entry:
                if entry.get("issue"):
                    raise ValueError(entry["issue"])
                return entry
        raise ValueError("Search again to select this model.")
