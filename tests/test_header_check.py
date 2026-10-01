"""Find models reads a variant's GGUF header before adding it.

The fixtures are synthetic headers built from recordings of real Hugging Face
files, each read with range requests and pinned by revision. Every entry keeps
the keys that decide its verdict and the true tensor layout: the top-level
names verbatim, one tensor per recorded block, and the ``nextn`` tensors on an
MTP block. No model bytes are stored in the repository.
"""

from __future__ import annotations

import functools
import hashlib
import io
import re
import socket
import struct
import tempfile
import threading
import time
import unittest
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

from lllm2 import config, downloads, gguf
from lllm2.app import App
from lllm2.catalogue import Catalogue, Finder
from lllm2.downloads import HeaderReadError
from lllm2.store import Store

NEXTN = ("nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight")


def blocks(indices: Iterable[int], *suffixes: str) -> list[str]:
    """Block tensor names: one per suffix for each layer index."""
    return [f"blk.{i}.{suffix}" for i in indices for suffix in suffixes]


def gguf_bytes(meta: dict[str, Any], tensors: list[str]) -> bytes:
    """A GGUF v3 header holding this metadata and these one-element tensors."""
    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(meta))]

    def text(value: str) -> bytes:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    for key, value in meta.items():
        out.append(text(key))
        if isinstance(value, bool):
            out.append(struct.pack("<I?", gguf.BOOL, value))
        elif isinstance(value, str):
            out.append(struct.pack("<I", gguf.STRING) + text(value))
        elif isinstance(value, float):
            out.append(struct.pack("<If", gguf.F32, value))
        elif isinstance(value, list):
            out.append(struct.pack("<IIQ", gguf.ARRAY, gguf.STRING, len(value)))
            out.extend(text(item) for item in value)
        elif value < 0:
            out.append(struct.pack("<Ii", gguf.I32, value))
        else:
            out.append(struct.pack("<II", gguf.U32, value))
    for name in tensors:
        # One dimension of one element, ggml type 0, offset 0.
        out.append(text(name) + struct.pack("<IQIQ", 1, 1, 0, 0))
    return b"".join(out)


#: The verdict each recorded kind must get.
EXPECT = {
    "mtp": gguf.MTP_REASON,
    "projector": gguf.PROJECTOR_REASON,
    "adapter": gguf.ADAPTER_REASON,
    "embedding": gguf.EMBEDDING_REASON,
    "reranker": gguf.RERANKER_REASON,
    "audio": gguf.SPEECH_REASON,
    "drafter": gguf.DRAFTER_REASON,
    "no-architecture": gguf.NO_ARCHITECTURE_REASON,
    "model": None,
}

#: Recorded headers: the file at its revision, its verdict, metadata and tensors.
RECORDED: list[tuple[str, str, dict[str, Any], list[str]]] = [
    # 15,767,670-byte header, 49 tensors
    (
        "Mia-AiLab/Gemmable-4-12B-MTP-GGUF/gemmable-4-12b-Q4_K_M-mtp.gguf@de886717f754",
        "mtp",
        {
            "general.architecture": "gemma4-assistant",
            "general.type": "model",
            "gemma4-assistant.block_count": 4,
            "gemma4-assistant.nextn_predict_layers": 4,
        },
        [
            "nextn.post_projection.weight",
            "nextn.pre_projection.weight",
            "output_norm.weight",
            "rope_freqs.weight",
            "token_embd.weight",
            *blocks(range(0, 4), "attn_norm.weight"),
        ],
    ),
    # 10,946,670-byte header, 34 tensors
    (
        "unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf@38bb39ee9782",
        "mtp",
        {
            "general.architecture": "qwen4exp",
            "general.type": "model",
            "qwen4exp.block_count": 49,
            "qwen4exp.nextn_predict_layers": 1,
        },
        [
            "output.weight",
            "token_embd.weight",
            *blocks([48], "attn_norm.weight", *NEXTN),
        ],
    ),
    # 10,946,608-byte header, 32 tensors
    (
        "apetersson/Qwen3.8-Flash-Next-GGUF/shared/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf@9af845f821fe",
        "mtp",
        {
            "general.architecture": "qwen4exp",
            "general.type": "model",
            "qwen4exp.block_count": 49,
            "qwen4exp.nextn_predict_layers": 1,
        },
        [*blocks([48], "attn_norm.weight", *NEXTN)],
    ),
    # 10,945,773-byte header, 19 tensors
    (
        "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf@993a5971fda8",
        "mtp",
        {
            "general.architecture": "qwen35",
            "qwen35.nextn_predict_layers": 1,
            "qwen35.block_count": 65,
            "general.type": "model",
        },
        [
            "output.weight",
            "output_norm.weight",
            "token_embd.weight",
            "d2t",
            *blocks([64], "attn_norm.weight", *NEXTN),
        ],
    ),
    # 20,200-byte header, 334 tensors
    (
        "esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF/mmproj-BF16.gguf@bcd7a7d3e251",
        "projector",
        {
            "general.architecture": "clip",
            "general.type": "mmproj",
            "clip.projector_type": "qwen3vl_merger",
        },
        [
            "v.blk.0.attn_out.bias",
            "v.blk.0.attn_out.weight",
            "v.blk.0.attn_qkv.bias",
            "v.blk.0.attn_qkv.weight",
            "v.blk.0.ffn_up.bias",
            "v.blk.0.ffn_up.weight",
        ],
    ),
    # 16,716-byte header, 240 tensors
    (
        "ibm-granite/granitelib-rag-r1.0/answerability/granite4_micro/lora/Lora-bf16.gguf@2f0b2c79c673",
        "adapter",
        {
            "general.architecture": "granite",
            "general.type": "adapter",
            "adapter.type": "lora",
        },
        [*blocks(range(0, 40), "attn_norm.weight")],
    ),
    # 6,826,483-byte header, 101 tensors
    (
        "bartowski/granite-embedding-107m-multilingual-GGUF/granite-embedding-107m-multilingual-Q4_K_M.gguf@52fed1c81863",
        "embedding",
        {
            "general.architecture": "bert",
            "general.type": "model",
            "bert.block_count": 6,
            "bert.pooling_type": 2,
        },
        [
            "position_embd.weight",
            "token_embd.weight",
            "token_embd_norm.bias",
            "token_embd_norm.weight",
            "token_types.weight",
            *blocks(range(0, 6), "attn_norm.weight"),
        ],
    ),
    # 5,945,533-byte header, 310 tensors
    (
        "Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf@370f27d7550e",
        "embedding",
        {
            "general.architecture": "qwen3",
            "general.type": "model",
            "qwen3.block_count": 28,
            "qwen3.pooling_type": 3,
        },
        [
            "output_norm.weight",
            "token_embd.weight",
            *blocks(range(0, 28), "attn_norm.weight"),
        ],
    ),
    # 6,530,436-byte header, 316 tensors
    (
        "ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf@0f741b5a6585",
        "embedding",
        {
            "general.architecture": "gemma-embedding",
            "general.type": "model",
            "gemma-embedding.block_count": 24,
            "gemma-embedding.pooling_type": 1,
        },
        [
            "dense_2.weight",
            "dense_3.weight",
            "output_norm.weight",
            "token_embd.weight",
            *blocks(range(0, 24), "attn_norm.weight"),
        ],
    ),
    # 1,033,386-byte header, 219 tensors
    (
        "city96/t5-v1_1-xxl-encoder-gguf/t5-v1_1-xxl-encoder-Q3_K_S.gguf@005a6ea51a7d",
        "embedding",
        {
            "general.architecture": "t5encoder",
            "general.type": "model",
            "t5encoder.block_count": 24,
        },
        [
            "enc.blk.0.attn_k.weight",
            "enc.blk.0.attn_o.weight",
            "enc.blk.0.attn_q.weight",
            "enc.blk.0.attn_rel_b.weight",
            "enc.blk.0.attn_v.weight",
            "enc.blk.0.attn_norm.weight",
        ],
    ),
    # 5,945,944-byte header, 311 tensors
    (
        "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf@a02f48bb4f05",
        "reranker",
        {
            "general.architecture": "qwen3",
            "general.type": "model",
            "qwen3.block_count": 28,
            "qwen3.pooling_type": 4,
        },
        [
            "cls.output.weight",
            "output_norm.weight",
            "token_embd.weight",
            *blocks(range(0, 28), "attn_norm.weight"),
        ],
    ),
    # 6,030,164-byte header, 311 tensors
    (
        "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf@ca27d74bc954",
        "audio",
        {
            "general.architecture": "qwen3tts",
            "general.type": "model",
            "qwen3tts.block_count": 28,
        },
        [
            "output.weight",
            "output_norm.weight",
            "token_embd.weight",
            *blocks(range(0, 28), "attn_norm.weight"),
        ],
    ),
    # 10,653-byte header, 161 tensors
    (
        "ggml-org/WavTokenizer/WavTokenizer-Large-75-Q5_1.gguf@0c97fdc09815",
        "audio",
        {
            "general.architecture": "wavtokenizer-dec",
            "general.type": "model",
            "wavtokenizer-dec.block_count": 12,
        },
        [
            "conv1d.bias",
            "conv1d.weight",
            "convnext.0.dw.bias",
            "convnext.0.dw.weight",
            "convnext.0.gamma.weight",
            "convnext.0.norm.bias",
        ],
    ),
    # 6,530,581-byte header, 15 tensors
    (
        "williamliao/gemma-4-26B-A4B-it-speculator.eagle3-F16-GGUF/gemma-4-26B-A4B-it-speculator.eagle3-Q4_K_M.gguf@4b61f204b293",
        "drafter",
        {
            "general.architecture": "eagle3",
            "general.type": "model",
            "eagle3.block_count": 1,
        },
        [
            "d2t",
            "fc.weight",
            "output.weight",
            "output_norm.weight",
            "token_embd.weight",
            *blocks(range(0, 1), "attn_norm.weight"),
        ],
    ),
    # 173-byte header, 1 tensors
    (
        "apetersson/Qwen3.8-Flash-Next-GGUF/shared/ngrams-Qwen3.8-Flash-Next-BF16.gguf@9af845f821fe",
        "no-architecture",
        {"split.no": 1, "split.count": 2, "split.tensors.count": 1224},
        ["per_layer_token_embd.weight"],
    ),
    # 11,015,096-byte header, 1202 tensors
    (
        "esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-MTP-HIGH.gguf@bcd7a7d3e251",
        "model",
        {
            "general.architecture": "qwen35",
            "general.type": "model",
            "qwen35.block_count": 65,
            "qwen35.nextn_predict_layers": 1,
        },
        [
            "output.weight",
            "output_norm.weight",
            "token_embd.weight",
            *blocks(range(0, 64), "attn_norm.weight"),
            *blocks([64], "attn_norm.weight", *NEXTN),
        ],
    ),
    # 10,995,837-byte header, 866 tensors
    (
        "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf@993a5971fda8",
        "model",
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 65,
            "qwen35.nextn_predict_layers": 1,
            "general.type": "model",
        },
        [
            "output.weight",
            "output_norm.weight",
            "token_embd.weight",
            *blocks(range(0, 64), "attn_norm.weight"),
            *blocks([64], "attn_norm.weight", *NEXTN),
        ],
    ),
    # 15,804,352-byte header, 667 tensors
    (
        "Mia-AiLab/Gemmable-4-12B-MTP-GGUF/gemmable-4-12b-Q4_K_M.gguf@de886717f754",
        "model",
        {
            "general.architecture": "gemma4",
            "general.type": "model",
            "gemma4.block_count": 48,
        },
        [
            "output_norm.weight",
            "rope_freqs.weight",
            "token_embd.weight",
            *blocks(range(0, 48), "attn_norm.weight"),
        ],
    ),
    # 10,946,618-byte header, 0 tensors
    (
        "unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf@38bb39ee9782",
        "model",
        {
            "general.architecture": "qwen4exp",
            "general.type": "model",
            "qwen4exp.block_count": 48,
            "split.no": 0,
            "split.tensors.count": 1224,
            "split.count": 3,
        },
        [],
    ),
    # 7,831,525-byte header, 147 tensors
    (
        "bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf@067b946cf014",
        "model",
        {
            "general.architecture": "llama",
            "general.type": "model",
            "llama.block_count": 16,
        },
        [
            "rope_freqs.weight",
            "token_embd.weight",
            "output_norm.weight",
            *blocks(range(0, 16), "attn_norm.weight"),
        ],
    ),
    # 741,039-byte header, 291 tensors
    (
        "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q2_K.gguf@191239b3e26b",
        "model",
        {"general.architecture": "llama", "llama.block_count": 32},
        [
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
            *blocks(range(0, 32), "attn_norm.weight"),
        ],
    ),
    # 3,573,191-byte header, 340 tensors
    (
        "mradermacher/openPangu-Embedded-1B-V1.1-i1-GGUF/openPangu-Embedded-1B-V1.1.i1-IQ1_S.gguf@488fee0d68a9",
        "model",
        {
            "general.architecture": "pangu-embedded",
            "general.type": "model",
            "pangu-embedded.block_count": 26,
        },
        [
            "output_norm.weight",
            "token_embd.weight",
            *blocks(range(0, 26), "attn_norm.weight"),
        ],
    ),
]


def recorded(source: str) -> tuple[dict[str, Any], list[str]]:
    """The metadata and tensors of the recording whose source starts so."""
    return next((m, t) for s, _, m, t in RECORDED if s.startswith(source))


#: An unsloth ``MTP/mtp-*.gguf`` head and the esatapedico full model that
#: carries its own head: same architecture, same keys, different blocks.
MTP_HEAD = recorded("unsloth/Qwen3.8-Flash-Next-GGUF/MTP/")
FULL_WITH_MTP = recorded("esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8")
PROJECTOR = recorded("esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF/mmproj")

VOCABULARY_TENSORS = ["token_embd.weight", *blocks(range(2), "attn_norm.weight")]


@functools.cache
def long_header() -> bytes:
    """A header whose 300,000-token vocabulary is longer than one range."""
    meta = {
        "general.architecture": "llama",
        "general.type": "model",
        "llama.block_count": 2,
        "tokenizer.ggml.tokens": [f"token-{i:010d}" for i in range(300_000)],
    }
    return gguf_bytes(meta, VOCABULARY_TENSORS)


class RecordedHeaderTests(unittest.TestCase):
    """The rules give every recorded header the verdict it was recorded with."""

    def test_recorded_headers_get_their_verdict(self) -> None:
        for source, expect, meta, tensors in RECORDED:
            with self.subTest(source):
                found = gguf.parse(io.BytesIO(gguf_bytes(meta, tensors)))
                self.assertEqual(found, (meta, tensors))
                self.assertEqual(gguf.not_a_model(*found), EXPECT[expect])
        # A recording for every rejected kind, and for full models.
        self.assertEqual({expect for _, expect, _, _ in RECORDED}, set(EXPECT))

    def test_rules_without_a_recording(self) -> None:
        def model(arch: str, **keys: Any) -> dict[str, Any]:
            return {"general.architecture": arch, "general.type": "model", **keys}

        weights = ["token_embd.weight", *blocks(range(4), "attn_norm.weight")]
        cases: list[tuple[str, dict[str, Any], list[str], str | None]] = [
            ("imatrix data", {"general.type": "imatrix"}, [], gguf.IMATRIX_REASON),
            (
                "an older projector",
                {"general.architecture": "clip"},
                weights,
                gguf.PROJECTOR_REASON,
            ),
            ("llama-embed", model("llama-embed"), weights, gguf.EMBEDDING_REASON),
            ("nomic-bert", model("nomic-bert"), weights, gguf.EMBEDDING_REASON),
            ("pockettts", model("pockettts"), weights, gguf.SPEECH_REASON),
            ("dflash", model("dflash"), weights, gguf.DRAFTER_REASON),
            ("dream", model("dream"), weights, gguf.DIFFUSION_REASON),
            ("llada", model("llada"), weights, gguf.DIFFUSION_REASON),
            ("llada-moe", model("llada-moe"), weights, gguf.DIFFUSION_REASON),
            ("rnd1", model("rnd1"), weights, gguf.DIFFUSION_REASON),
            (
                "a chat model, pooling none",
                model("qwen3", **{"qwen3.pooling_type": 0}),
                weights,
                None,
            ),
            ("pangu-embedded is a chat model", model("pangu-embedded"), weights, None),
            (
                "an empty type is a model",
                {**model("llama"), "general.type": ""},
                weights,
                None,
            ),
            ("a vocabulary-only file", model("llama"), [], gguf.NO_WEIGHTS_REASON),
            (
                "a split's first shard, which holds no tensors",
                model(
                    "qwen35",
                    **{
                        "qwen35.block_count": 65,
                        "qwen35.nextn_predict_layers": 1,
                        "split.count": 3,
                    },
                ),
                [],
                None,
            ),
            (
                "a split MTP head with no trunk",
                model(
                    "qwen35",
                    **{
                        "qwen35.block_count": 1,
                        "qwen35.nextn_predict_layers": 1,
                        "split.count": 2,
                    },
                ),
                blocks([0], "attn_norm.weight", *NEXTN),
                gguf.MTP_REASON,
            ),
            (
                "an MTP count without a block count",
                model("qwen35", **{"qwen35.nextn_predict_layers": 1}),
                blocks([64], "attn_norm.weight"),
                None,
            ),
            (
                "a block index too long to be a layer",
                model(
                    "llama", **{"llama.block_count": 2, "llama.nextn_predict_layers": 1}
                ),
                [f"blk.{'9' * 5000}.attn_norm.weight", "blk.1.attn_norm.weight"],
                gguf.MTP_REASON,
            ),
        ]
        for label, meta, tensors, expect in cases:
            with self.subTest(label):
                found = gguf.parse(io.BytesIO(gguf_bytes(meta, tensors)))
                self.assertEqual(gguf.not_a_model(*found), expect)
        unknown = gguf.not_a_model({"general.type": "vocabulary"}, [])
        self.assertEqual(
            unknown, "This file is a GGUF of type “vocabulary”, not a model."
        )

    def test_parse_matches_header(self) -> None:
        data = gguf_bytes(*FULL_WITH_MTP)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            path.write_bytes(data)
            self.assertEqual(gguf.header(path), gguf.parse(io.BytesIO(data)))


class Request(NamedTuple):
    """One request the stand-in hub answered."""

    path: str
    range: str | None
    agent: str | None
    status: int
    served: int


class Hub(ThreadingHTTPServer):
    """Hugging Face, standing in: ``resolve`` redirects to a CDN serving ranges.

    Each file is a header followed by zeros up to a virtual size, so a 1 GiB
    model costs nothing to serve unless it is actually read.
    """

    files: dict[str, tuple[bytes, int]]
    log: list[Request]
    status: int | None
    ignore_range: bool
    #: Answer every range from byte 0, as a broken cache might.
    from_zero: bool


class HubHandler(BaseHTTPRequestHandler):
    server: Hub

    def do_GET(self) -> None:
        hub = self.server
        if "/resolve/" in self.path:
            self.answer(302, {"Location": "/cdn/" + self.path.rsplit("/", 1)[1]})
            return
        if hub.status:
            self.answer(hub.status)
            return
        head, size = hub.files[self.path.removeprefix("/cdn/")]
        ranged = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range") or "")
        if hub.ignore_range or not ranged:
            start, end, status = 0, size - 1, 200
            self.answer(status, {"Content-Length": str(size)}, body=False)
        else:
            start, end, status = int(ranged[1]), min(int(ranged[2]), size - 1), 206
            if hub.from_zero:
                start, end = 0, end - start
            self.answer(
                status,
                {
                    "Content-Range": f"bytes {start}-{end}/{size}",
                    "Content-Length": str(end - start + 1),
                },
                body=False,
            )
        served = 0
        try:
            while start + served <= end:
                at = start + served
                n = min(1 << 20, end + 1 - at)
                self.wfile.write(head[at : at + n].ljust(n, b"\0"))
                served += n
        except (BrokenPipeError, ConnectionResetError):
            pass  # the reader stopped early, as it should on a bad answer
        self.log(self.headers.get("Range"), status, served)

    def answer(
        self, status: int, headers: dict[str, str] | None = None, body: bool = True
    ) -> None:
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if body:
            self.send_header("Content-Length", "0")
            self.log(self.headers.get("Range"), status, 0)
        self.end_headers()

    def log(self, ranged: str | None, status: int, served: int) -> None:
        agent = self.headers.get("User-Agent")
        self.server.log.append(Request(self.path, ranged, agent, status, served))

    def log_message(self, *args: Any) -> None:
        pass


def serve(test: unittest.TestCase) -> Hub:
    """Start a hub for this test and point the reader at it."""
    hub = Hub(("127.0.0.1", 0), HubHandler)
    hub.files, hub.log, hub.status = {}, [], None
    hub.ignore_range = hub.from_zero = False
    # A short poll, so shutting the hub down does not hold each test up.
    thread = threading.Thread(
        target=hub.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    test.addCleanup(hub.server_close)
    test.addCleanup(thread.join)
    test.addCleanup(hub.shutdown)
    base = f"http://127.0.0.1:{hub.server_port}"
    patcher = patch.object(
        downloads,
        "url_for",
        lambda repo, file, revision="main": f"{base}/{repo}/resolve/{revision}/{file}",
    )
    patcher.start()
    test.addCleanup(patcher.stop)
    return hub


def closed_port() -> int:
    """A local port with nothing listening on it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def spans(requests: list[Request]) -> list[tuple[int, int]]:
    """The byte ranges the CDN was asked for, in order."""
    found = [re.fullmatch(r"bytes=(\d+)-(\d+)", r.range or "") for r in requests]
    return [(int(m[1]), int(m[2])) for m in found if m]


class RemoteHeaderTests(unittest.TestCase):
    """The header is read with range requests and never past its end."""

    REVISION = "c" * 40

    def setUp(self) -> None:
        self.hub = serve(self)

    def cdn(self) -> list[Request]:
        return [r for r in self.hub.log if r.path.startswith("/cdn/")]

    def read(self, file: str = "model.gguf") -> tuple[dict[str, Any], list[str]]:
        return downloads.remote_header("owner/Repo-GGUF", file, self.REVISION)

    def test_reads_ranges_through_the_redirect_and_stops_after_the_header(self) -> None:
        head = gguf_bytes(*MTP_HEAD)
        self.hub.files["mtp-model-Q4_K_M.gguf"] = (head, 1 << 30)
        found = self.read("MTP/mtp-model-Q4_K_M.gguf")
        self.assertEqual(found, gguf.parse(io.BytesIO(head)))
        self.assertEqual(gguf.not_a_model(*found), gguf.MTP_REASON)
        redirects = [r for r in self.hub.log if r.status == 302]
        self.assertEqual(
            [r.path for r in redirects],
            [f"/owner/Repo-GGUF/resolve/{self.REVISION}/MTP/mtp-model-Q4_K_M.gguf"],
        )
        # The range survived the redirect, and nothing past it was served.
        self.assertEqual(spans(self.cdn()), [(0, downloads.HEADER_RANGE - 1)])
        self.assertEqual(sum(r.served for r in self.cdn()), downloads.HEADER_RANGE)
        self.assertEqual({r.agent for r in self.hub.log}, {downloads.USER_AGENT})

    def test_a_header_longer_than_one_range_is_read_in_order(self) -> None:
        head = long_header()
        self.assertGreater(len(head), downloads.HEADER_RANGE)
        self.hub.files["model.gguf"] = (head, 1 << 30)
        meta, names = self.read()
        self.assertEqual(meta["tokenizer.ggml.tokens"], "<array of 300000>")
        self.assertEqual(names, VOCABULARY_TENSORS)
        ranges = spans(self.cdn())
        self.assertGreaterEqual(len(ranges), 2)
        self.assertEqual(ranges[0][0], 0)
        for (_, last), (first, _) in zip(ranges, ranges[1:], strict=False):
            self.assertEqual(first, last + 1)
        self.assertLess(
            sum(r.served for r in self.cdn()), len(head) + downloads.HEADER_RANGE
        )

    def test_a_file_shorter_than_one_range_takes_one_request(self) -> None:
        # A LoRA or projector header can be the whole of a small file.
        self.hub.files["mmproj-F16.gguf"] = (gguf_bytes(*PROJECTOR), 20_000)
        found = self.read("mmproj-F16.gguf")
        self.assertEqual(gguf.not_a_model(*found), gguf.PROJECTOR_REASON)
        self.assertEqual(len(self.cdn()), 1)
        self.assertEqual(self.cdn()[0].served, 20_000)

    def test_failures_raise_one_error(self) -> None:
        vocabulary = long_header()
        # A first key that claims to be 100 MiB long, inside a 1 GiB file.
        hostile = b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 100 << 20)
        # One value that is an array of an array of ... deeper than Python
        # recurses.
        nested = (
            b"GGUF"
            + struct.pack("<IQQQ", 3, 0, 1, 1)
            + b"x"
            + struct.pack("<I", gguf.ARRAY)
            + struct.pack("<IQ", gguf.ARRAY, 1) * 5000
            + struct.pack("<IQ", gguf.U8, 0)
        )
        mtp = gguf_bytes(*MTP_HEAD)
        cases: list[tuple[str, dict[str, Any], bytes, int, str]] = [
            ("not found", {"status": 404}, mtp, 1 << 30, "HTTP 404"),
            ("server error", {"status": 500}, mtp, 1 << 30, "HTTP 500"),
            (
                "range ignored",
                {"ignore_range": True},
                mtp,
                64 << 10,
                "did not answer the range request",
            ),
            (
                "a range from the wrong place",
                {"from_zero": True},
                vocabulary,
                1 << 30,
                "did not answer the range request",
            ),
            ("not a GGUF", {}, b"PK\x03\x04" + bytes(100), 1 << 20, "no GGUF magic"),
            (
                "header past the limit",
                {"HEADER_LIMIT": 1 << 20},
                vocabulary,
                1 << 30,
                "larger than 1 MiB",
            ),
            (
                "a hostile length",
                {"HEADER_LIMIT": 12 << 20},
                hostile,
                1 << 30,
                "the file gave fewer",
            ),
            ("arrays nested too deep", {}, nested, 1 << 20, "recursion depth"),
            ("out of time", {"HEADER_SECONDS": 0.0}, mtp, 1 << 30, "took too long"),
        ]
        for label, setup, head, size, message in cases:
            with self.subTest(label):
                self.hub.files["model.gguf"] = (head, size)
                self.hub.log.clear()
                self.hub.status = setup.get("status")
                self.hub.ignore_range = setup.get("ignore_range", False)
                self.hub.from_zero = setup.get("from_zero", False)
                limit = setup.get("HEADER_LIMIT", downloads.HEADER_LIMIT)
                seconds = setup.get("HEADER_SECONDS", downloads.HEADER_SECONDS)
                with (
                    patch.multiple(
                        downloads, HEADER_LIMIT=limit, HEADER_SECONDS=seconds
                    ),
                    self.assertRaisesRegex(HeaderReadError, message),
                ):
                    self.read()
                # However the read failed, it stayed inside its byte budget.
                served = [r.served for r in self.cdn()]
                self.assertLessEqual(sum(served), limit)
                if label == "out of time":
                    self.assertEqual(self.hub.log, [])
        self.hub.status, self.hub.ignore_range, self.hub.from_zero = None, False, False
        with (
            patch.object(
                downloads,
                "url_for",
                return_value=f"http://127.0.0.1:{closed_port()}/x.gguf",
            ),
            self.assertRaisesRegex(HeaderReadError, "refused"),
        ):
            self.read()


class AddVariantTests(unittest.TestCase):
    """Adding a Find models variant refuses one whose header is not a model."""

    HOST = {"gpus": [{"total_mib": 24576}], "ram": {"total_gib": 64}}

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for key, value in (
            ("STATE_DIR", root / "state"),
            ("MODELS_DIR", root / "models"),
        ):
            patcher = patch.object(config, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.store = Store()
        self.addCleanup(self.store.db.close)
        self.app = App.__new__(App)
        self.app.store = self.store
        self.app.catalogue = Catalogue(self.store)
        self.app.finder = self.finder = Finder(self.store)
        self.head = self.variant("mtp-Repo-Q4_K_M.gguf")
        self.full = self.variant("Repo-NVFP4-MTP-HIGH.gguf")
        self.split = {
            **self.variant("Repo-00001-of-00002.gguf"),
            "issue": "Incomplete split GGUF metadata.",
        }
        self.query = "repo"
        self.store.put(
            "hf-search",
            hashlib.sha256(self.query.encode()).hexdigest(),
            {
                "entries": [self.head, self.full, self.split],
                "fetched_at": time.time(),
                "warning": "",
                "repositories": 1,
            },
        )

    @staticmethod
    def variant(file: str) -> dict[str, Any]:
        stem = file.removesuffix(".gguf").lower()
        return {
            "id": f"hf-{stem}",
            "name": f"Repo-GGUF-{stem}",
            "display_name": "Repo-GGUF",
            "repo": "owner/Repo-GGUF",
            "revision": "c" * 40,
            "file": file,
            "files": [file],
            "size_bytes": 1 << 30,
            "issue": "",
            "source": "huggingface",
        }

    def add(self, entry: dict[str, Any]) -> Any:
        return self.app.action("/api/catalogue/add", {"id": entry["id"]})

    def test_adding_reads_the_header_from_hugging_face(self) -> None:
        hub = serve(self)
        hub.files[self.head["file"]] = (gguf_bytes(*MTP_HEAD), 1 << 30)
        with self.assertRaisesRegex(
            ValueError,
            r"^Cannot add mtp-Repo-Q4_K_M\.gguf\. This file is a multi-token prediction \(MTP\) head",
        ):
            self.add(self.head)
        self.assertIsNone(self.store.get("catalogue", self.head["id"]))

    def test_a_full_model_with_an_mtp_head_is_added(self) -> None:
        hub = serve(self)
        hub.files[self.full["file"]] = (gguf_bytes(*FULL_WITH_MTP), 1 << 30)
        self.assertEqual(self.add(self.full), self.full)
        self.assertEqual(self.app.catalogue.get(self.full["id"]), self.full)
        self.assertTrue(any(r.path.startswith("/cdn/") for r in hub.log))

    def test_an_unreachable_hub_adds_the_variant(self) -> None:
        url = f"http://127.0.0.1:{closed_port()}/x.gguf"
        with patch.object(downloads, "url_for", return_value=url):
            self.assertEqual(self.add(self.head), self.head)
        self.assertEqual(self.app.catalogue.get(self.head["id"]), self.head)

    def test_an_unreadable_header_adds_the_variant_and_is_read_again(self) -> None:
        with patch(
            "lllm2.downloads.remote_header", side_effect=HeaderReadError("HTTP 500")
        ) as read:
            self.assertEqual(self.add(self.head), self.head)
            self.assertEqual(self.finder.candidate(self.head["id"]), self.head)
        self.assertEqual(read.call_count, 2)
        self.assertEqual(self.app.catalogue.get(self.head["id"]), self.head)

    def test_a_refused_variant_is_remembered_and_listed(self) -> None:
        with patch("lllm2.downloads.remote_header", return_value=MTP_HEAD) as read:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "multi-token prediction"):
                    self.add(self.head)
            listed = {
                e["id"]: e for e in self.finder.search(self.query, self.HOST)["entries"]
            }
        self.assertEqual(read.call_count, 1)
        self.assertEqual(listed[self.head["id"]]["issue"], gguf.MTP_REASON)
        self.assertEqual(listed[self.full["id"]]["issue"], "")
        self.assertEqual(listed[self.split["id"]]["issue"], self.split["issue"])

    def test_a_variant_with_an_issue_is_not_read(self) -> None:
        with patch("lllm2.downloads.remote_header") as read:
            with self.assertRaisesRegex(ValueError, "Incomplete split"):
                self.add(self.split)
        read.assert_not_called()
