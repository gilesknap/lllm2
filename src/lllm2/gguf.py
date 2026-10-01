"""Read what a GGUF says about itself, without loading it.

A GGUF carries its metadata and its full tensor list in a header, before any
weights. That is enough to answer questions the catalogue cannot answer from a
file size or a file name: does this checkpoint carry a multi-token prediction
head, and is it a model llama-server can serve at all? The second is asked of a
Hugging Face file before it joins the catalogue -- see :func:`not_a_model`.

Everything here reads the header and stops. An 18 GB checkpoint is inspected in
tens of milliseconds, because the tokenizer vocabulary is walked as offsets
rather than materialised as strings.

Why this is derived rather than declared: the catalogue can *say* a repo ships
MTP, but only the file on disk decides whether the engine can use it. Two
builds of the same model differ, quantisers strip the head, and a partially
downloaded file has no tensors at all. A flag passed to llama.cpp on the
strength of a YAML field would produce an engine that fails at load; read from
the file it cannot be wrong.

**The header is not free, and the panel asks every two seconds.** Reading all
nine catalogued checkpoints takes about 0.65 s, which is why :func:`facts`
memoises on the file's identity rather than its name -- see its docstring for
what that costs and what invalidates it.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, NamedTuple, Protocol

#: GGUF metadata value types, by their on-disk tag.
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

#: The fixed-width types, and the struct format that reads each.
_FIXED = {
    U8: "B",
    I8: "b",
    U16: "H",
    I16: "h",
    U32: "I",
    I32: "i",
    F32: "f",
    BOOL: "?",
    U64: "Q",
    I64: "q",
    F64: "d",
}

#: How much of the file to pull in at a time.
#:
#: A real header is a few megabytes -- almost all of it vocabulary -- so this
#: reads most of them in one call and the rest in two, while never reading a
#: whole checkpoint to answer a question about its first pages.
_CHUNK = 4 << 20

#: Substrings that identify a multi-token prediction head in a tensor name.
#:
#: Qwen calls it ``nextn`` and puts it in one extra block past the last real
#: layer -- ``blk.64.nextn.*`` on the 27B, ``blk.40.nextn.*`` on the 35B-A3B.
#: DeepSeek uses the same word. ``mtp`` is accepted for anything that spells it
#: the other way.
MTP_MARKERS = ("nextn", "mtp")


class Malformed(Exception):
    """This file is not a GGUF, or its header does not parse."""


class Readable(Protocol):
    """What the header parse needs of a file: a seek and a read.

    A local file has both. So does a file on a web server read with range
    requests, which is how a Hugging Face file is checked before it is added
    to the catalogue.
    """

    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def read(self, size: int, /) -> bytes: ...


class _Head:
    """The front of the file in memory, grown forward as the parse walks it.

    The parse is a walk over lengths that are themselves read *from the file*,
    so a corrupt or hostile header can ask to step 2**64 bytes forward. Every
    such step goes through :meth:`upto`, which refuses anything the file cannot
    back -- the caller is ``engine.start``, which must not raise ``OverflowError``
    or exhaust memory on a bad checkpoint.

    In memory rather than seeked over, because the cost here is per *element*
    and a modern vocabulary has 150k of them: three method calls and a buffered
    read apiece was 4.5x slower than walking offsets in a ``bytes`` already
    held. Not ``mmap``, which would be faster still and can take the whole
    process down with ``SIGBUS`` if the mapped file is truncated underneath it
    -- which is exactly what a re-download does.
    """

    def __init__(self, fh: Readable) -> None:
        self.fh = fh
        self.size = fh.seek(0, 2)
        self.buf = b""

    def upto(self, end: int) -> bytes:
        """The file's first ``end`` bytes, reading more if they are not held."""
        if not 0 <= end <= self.size:
            raise Malformed(f"offset {end} is not inside a {self.size}-byte file")
        if end > len(self.buf):
            want = min(self.size, max(end, len(self.buf) + _CHUNK))
            self.fh.seek(len(self.buf))
            more = self.fh.read(want - len(self.buf))
            if len(self.buf) + len(more) < end:
                raise Malformed(f"wanted {end} bytes, the file gave fewer")
            self.buf += more
        return self.buf


def _u64_at(buf: bytes, pos: int) -> int:
    return int.from_bytes(buf[pos : pos + 8], "little")


def _string_at(head: _Head, pos: int) -> tuple[int, str]:
    """One length-prefixed UTF-8 string, and where it ends."""
    buf = head.upto(pos + 8)
    length = _u64_at(buf, pos)
    pos += 8
    buf = head.upto(pos + length)
    return pos + length, buf[pos : pos + length].decode("utf-8", "replace")


def _skip_strings(head: _Head, pos: int, count: int) -> int:
    """Step over ``count`` length-prefixed strings, decoding none of them.

    This is the tokenizer vocabulary, which is most of the header and none of
    the answer. It is written out flat rather than as a loop over
    :func:`_string_at` because it is the only part of the parse whose cost
    scales with the model rather than with the format: 150k iterations, where
    each avoided function call is worth more than it looks.
    """
    buf = head.upto(min(head.size, pos + _CHUNK))
    limit = len(buf) - 8
    # Bound as a local and the decode written out: `_u64_at` here rather than
    # inline costs a fifth of the whole read across the catalogue, because the
    # loop body runs 150k times per model and does nothing else.
    from_bytes = int.from_bytes
    for _ in range(count):
        if pos > limit:
            buf = head.upto(min(head.size, pos + _CHUNK))
            limit = len(buf) - 8
            if pos > limit:
                raise Malformed("a string array runs past the end of the file")
        pos += 8 + from_bytes(buf[pos : pos + 8], "little")
    # The final string's *bytes* still have to be inside the file; the loop
    # only proved that each length prefix was.
    head.upto(pos)
    return pos


def _value(head: _Head, pos: int, kind: int) -> tuple[int, Any]:
    """One metadata value, returning a placeholder for the bulky ones.

    Arrays are where the tokenizer vocabulary lives -- a hundred thousand
    strings on a modern model. They are stepped over rather than decoded,
    because nothing here needs them and building the list is most of the cost
    of reading the header at all.
    """
    if kind in _FIXED:
        fmt = _FIXED[kind]
        size = struct.calcsize(fmt)
        buf = head.upto(pos + size)
        return pos + size, struct.unpack_from("<" + fmt, buf, pos)[0]
    if kind == STRING:
        return _string_at(head, pos)
    if kind == ARRAY:
        buf = head.upto(pos + 12)
        element, count = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        if element in _FIXED:
            # `count` is attacker-controlled and multiplied, so the product is
            # checked rather than the count: 2**62 fixed-width elements is a
            # step no file can back.
            end = pos + struct.calcsize(_FIXED[element]) * count
            head.upto(end)
            return end, f"<array of {count}>"
        if element == STRING:
            return _skip_strings(head, pos, count), f"<array of {count}>"
        for _ in range(count):
            pos, _unused = _value(head, pos, element)
        return pos, f"<array of {count}>"
    raise Malformed(f"unknown metadata type {kind}")


def header(path: Path | str) -> tuple[dict[str, Any], list[str]]:
    """A checkpoint's metadata and tensor names, read from its header alone."""
    with open(path, "rb") as raw:
        return parse(raw)


def parse(fh: Readable) -> tuple[dict[str, Any], list[str]]:
    """The same, from anything that seeks and reads, such as a remote file."""
    head = _Head(fh)
    if head.size < 24 or head.upto(4)[:4] != b"GGUF":
        raise Malformed("no GGUF magic; this is not a checkpoint")
    # Byte 4 is the format version, unused: the layout below is stable
    # across 2-3.
    buf = head.upto(24)
    tensors, pairs = struct.unpack_from("<QQ", buf, 8)
    pos = 24
    meta: dict[str, Any] = {}
    for _ in range(pairs):
        pos, key = _string_at(head, pos)
        buf = head.upto(pos + 4)
        kind = struct.unpack_from("<I", buf, pos)[0]
        pos, meta[key] = _value(head, pos + 4, kind)
    names = []
    for _ in range(tensors):
        pos, name = _string_at(head, pos)
        names.append(name)
        buf = head.upto(pos + 4)
        dims = struct.unpack_from("<I", buf, pos)[0]
        # The shape, then the ggml type and the offset into the tensor
        # blob. None of the three is read; stepping over them is checked
        # because `dims` came out of the file like everything else.
        pos += 4 + 8 * dims + 12
        head.upto(pos)
    return meta, names


class Facts(NamedTuple):
    """What the file itself says, as opposed to what ``models.yaml`` declares.

    Both fields are exactly what :func:`has_mtp` and
    :func:`full_attention_layers` return, and both are derived from one read.
    They used to be two, which meant a caller wanting both parsed an 18 GB
    checkpoint's header twice.
    """

    #: A usable multi-token prediction head is present.
    mtp: bool
    #: How many layers keep a KV cache, or None where the question does not
    #: apply to this architecture.
    full_attention_layers: int | None


#: Parsed headers, keyed on ``(path, size, mtime_ns)``.
#:
#: The stat is the point. Keyed on the path alone, a re-download or a truncated
#: part file would keep answering with the old file's head, and the flag that
#: follows from it decides whether llama.cpp starts at all. Keyed on identity,
#: a file that changes mints a new entry and the stale one falls out.
_FACTS: dict[tuple[str, int, int], Facts] = {}

#: How many to keep. Nine checkpoints is a full disk here, and a download in
#: progress mints a fresh entry on every mtime change -- so this is bounded to
#: stop a long-lived panel accumulating one entry per second of a download.
_FACTS_MAX = 64


def facts(path: Path | str) -> Facts:
    """Everything derived from this checkpoint's header, read at most once.

    A failed read is not cached: the overwhelmingly likely reason is a file
    that is still arriving, and remembering "not a GGUF" about a checkpoint
    that is about to become one would need a second invalidation rule to undo.
    Re-reading a header that fails to parse is cheap -- it fails in the first
    few bytes.
    """
    try:
        stat = Path(path).stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return Facts(False, None)
    hit = _FACTS.get(key)
    if hit is not None:
        return hit
    try:
        meta, names = header(path)
    except (OSError, Malformed, struct.error, ValueError, MemoryError):
        # Every failure means "no MTP". This decides whether to append a flag
        # to a working command line: guessing yes on a file that could not be
        # parsed produces an engine that exits at load, guessing no produces
        # one that runs a little slower. ValueError covers OverflowError.
        return Facts(False, None)
    found = Facts(
        mtp=any(m in n.lower() for n in names for m in MTP_MARKERS),
        full_attention_layers=_full_attention_layers(meta),
    )
    if len(_FACTS) >= _FACTS_MAX:
        del _FACTS[next(iter(_FACTS))]
    _FACTS[key] = found
    return found


def _full_attention_layers(meta: dict[str, Any]) -> int | None:
    blocks = next((v for k, v in meta.items() if k.endswith(".block_count")), None)
    interval = next(
        (v for k, v in meta.items() if k.endswith(".full_attention_interval")),
        None,
    )
    if not isinstance(blocks, int) or not isinstance(interval, int):
        return None
    if interval <= 0 or blocks <= 1:
        return None
    return (blocks - 1) // interval


def _integer(meta: dict[str, Any], suffix: str) -> int | None:
    """One integer metadata value, found by key suffix, or None.

    Architectures name their keys after themselves -- ``qwen3next.block_count``
    -- so the suffix is the only stable part. A per-layer value is written as an
    array, which the parse steps over rather than decodes, so it arrives here as
    a placeholder string and is rejected like any other non-integer.
    """
    value = next((v for k, v in meta.items() if k.endswith(suffix)), None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def cache_layers(meta: dict[str, Any], mtp: bool = False) -> int | None:
    """How many of a header's layers keep a KV cache, or None.

    A hybrid keeps a cache on one layer in every ``full_attention_interval``;
    the rest are SSM, whose state costs nothing as context grows. Anything else
    caches every block. Either way the multi-token prediction head is left out
    when the tensors show one, because ``block_count`` includes it and the
    planner prices that head separately as a draft cache.

    :func:`full_attention_layers` divides by the interval after subtracting the
    head unconditionally, which is right for the hybrids that carry one and
    undercounts by a layer for one that does not -- an undercount that would
    price the cache too cheaply and plan a context that does not fit. Here the
    subtraction follows the tensors instead.

    Args:
        meta: A parsed GGUF metadata mapping.
        mtp: Whether the tensors carry a multi-token prediction head.

    Returns:
        The number of caching layers, or None where the header does not say.
    """
    blocks = _integer(meta, ".block_count")
    if blocks is None or blocks <= 1:
        return None
    layers = blocks - 1 if mtp else blocks
    interval = _integer(meta, ".full_attention_interval")
    if interval is None:
        return layers
    if interval <= 0:
        return None
    return (layers // interval) or None


def kv_kib_per_token(meta: dict[str, Any], mtp: bool = False) -> float | None:
    """KiB of f16 KV cache one token of context costs, or None.

    This is the catalogue's hand-entered ``kv_kib_per_token`` derived from the
    file instead: ``kv_heads x (key_length + value_length) x 2 bytes`` for every
    caching layer. It is what lets the context planner budget a checkpoint the
    catalogue has never seen -- any quantisation of any model, local or remote.

    Args:
        meta: A parsed GGUF metadata mapping.
        mtp: Whether the tensors carry a multi-token prediction head.

    Returns:
        The per-token cost in KiB, or None where the header lacks a shape to
        compute it from. None is not a failure: the caller falls back to the
        catalogue, and says so when neither can answer.
    """
    layers = cache_layers(meta, mtp)
    heads = _integer(meta, ".attention.head_count_kv")
    if heads is None:
        heads = _integer(meta, ".attention.head_count")
    key = _integer(meta, ".attention.key_length")
    value = _integer(meta, ".attention.value_length")
    if key is None or value is None:
        # Older conversions leave both out; the heads then split the embedding.
        embedding = _integer(meta, ".embedding_length")
        attention_heads = _integer(meta, ".attention.head_count")
        if embedding and attention_heads and embedding % attention_heads == 0:
            key = embedding // attention_heads if key is None else key
            value = embedding // attention_heads if value is None else value
    if not layers or not heads or not key or not value:
        return None
    return heads * (key + value) * 2 * layers / 1024


def full_attention_layers(path: Path | str) -> int | None:
    """How many of this checkpoint's layers keep a KV cache, or None.

    The catalogue's ``kv_kib_per_token`` is
    ``kv_heads x (key_length + value_length) x 2 bytes x`` this number, and on
    the hybrid models that carry an MTP head it reproduces the hand-entered
    figures exactly. So this is what lets a declared field be *checked* rather
    than trusted.

    Two things make it less obvious than counting blocks:

    * ``full_attention_interval`` -- these are hybrids, and only one layer in
      four keeps a cache. The rest are SSM, whose state is per-sequence rather
      than per-token and therefore costs nothing as context grows.
    * ``block_count`` **includes the MTP head**, which is why it reads 65 for a
      64-layer model and 41 for a 40-layer one. Subtracting it is the whole
      reason the head can be priced as "one more of these".

    ``None`` for an architecture that is not shaped this way -- Gemma-4 with
    its sliding-window pattern and per-layer head counts, gpt-oss with no
    interval at all. That is not a failure: none of them carries an MTP head,
    so nothing needs the number.
    """
    return facts(path).full_attention_layers


def has_mtp(path: Path | str) -> bool:
    """Whether this checkpoint carries a usable multi-token prediction head.

    Requires the *tensors*, not merely the metadata key. A conversion can
    announce ``nextn_predict_layers`` and still ship no head, and llama.cpp
    refuses to start with ``--spec-type draft-mtp`` against a checkpoint that
    has none -- so the tensors are the only honest test.

    Any failure to read the file is a ``False``: this decides whether to add a
    flag to a working command line, and the cost of being wrong in that
    direction is an engine that will not start.
    """
    return facts(path).mtp


# Architecture names are llama.cpp's own, from ``LLM_ARCH_NAMES`` in
# src/llama-arch.cpp at the engine pin (b10850). They are matched exactly, never
# as substrings: ``pangu-embedded`` is a chat model despite its name.

#: Architectures that turn text into vectors: the BERT family, the embedding
#: variants of Gemma and Llama, and T5's encoder half on its own.
ENCODERS = frozenset(
    {
        "bert",
        "modern-bert",
        "neo-bert",
        "eurobert",
        "nomic-bert",
        "nomic-bert-moe",
        "jina-bert-v2",
        "jina-bert-v3",
        "gemma-embedding",
        "llama-embed",
        "t5encoder",
    }
)
#: Speech and audio architectures: a vocoder and two text-to-speech backbones.
SPEECH = frozenset({"wavtokenizer-dec", "qwen3tts", "pockettts"})
#: Speculative-decoding drafters, which run only beside the model they draft for.
DRAFTERS = frozenset({"eagle3", "dflash", "gemma4-assistant"})
#: Diffusion language models. llama.cpp builds them without a KV cache, and
#: llama-server answers every completion on such a context with an error
#: (``tools/server/server-context.cpp`` at b10850: "the current context does
#: not logits computation").
DIFFUSION = frozenset({"dream", "llada", "llada-moe", "rnd1"})

#: A block tensor's layer index, as in ``blk.48.attn_k.weight``. Nine digits is
#: far beyond any real model and keeps ``int()`` cheap on a hostile name.
_BLOCK = re.compile(r"blk\.([0-9]{1,9})\.")

# Why :func:`not_a_model` refuses a file, in the words the panel shows.
ADAPTER_REASON = (
    "This file is an adapter (such as a LoRA), not a model. "
    "It only changes the base model it is loaded with."
)
PROJECTOR_REASON = (
    "This file is a vision projector, not a model. "
    "It is loaded beside the model it belongs to."
)
IMATRIX_REASON = "This file is importance-matrix data for quantising, not a model."
NO_ARCHITECTURE_REASON = (
    "This file does not name a model architecture, "
    "so the engine cannot load it as a model."
)
NO_WEIGHTS_REASON = "This file holds no weights, so it is not a model."
MTP_REASON = (
    "This file is a multi-token prediction (MTP) head for speculative decoding, "
    "not a model on its own."
)
DRAFTER_REASON = "This file is a speculative-decoding drafter, not a model on its own."
RERANKER_REASON = (
    "This file is a reranking model. "
    "It scores text for search rather than writing replies."
)
EMBEDDING_REASON = (
    "This file is an embedding or text-encoder model. "
    "It turns text into vectors rather than writing replies."
)
SPEECH_REASON = "This file is a speech or audio model, not a text model."
DIFFUSION_REASON = (
    "This file is a diffusion language model, which llama-server cannot serve."
)

#: ``general.type`` values other than ``model``, from gguf-py's ``GGUFType``.
_KIND_REASONS = {
    "adapter": ADAPTER_REASON,
    "mmproj": PROJECTOR_REASON,
    "imatrix": IMATRIX_REASON,
}

#: ``pooling_type`` 4 is RANK in llama.cpp's ``PoolingType``; 1-3 pool an
#: embedding (mean, CLS, last token).
_RANK_POOLING = 4


def _integer_at(meta: dict[str, Any], key: str) -> int | None:
    """The integer stored under exactly ``key``, or None.

    Exact rather than by suffix, as :func:`_integer` matches: the key is built
    from the file's own architecture name, so there is nothing to search for,
    and a suffix could match another architecture's key.
    """
    value = meta.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def not_a_model(meta: dict[str, Any], names: list[str]) -> str | None:
    """Why this header is not a model llama-server can serve on its own, or None.

    The first rule that matches gives the reason:

    1. ``general.type`` is not ``model``: an adapter, a projector or imatrix
       data. A file without the key is a model; GGUF v2 files predate it.
    2. No ``general.architecture``: a later shard of a split, or a table such
       as an n-gram file, which nothing can load as a model.
    3. Architecture ``clip``: an older projector written before
       ``general.type``.
    4. No tensors in a file that is not split. A split's first shard holds
       only metadata, so it is not judged on its tensors.
    5. A multi-token prediction head. llama.cpp's ``--mtp`` export keeps the
       full model's architecture, adds the head's layers to ``block_count``,
       writes ``nextn_predict_layers`` and keeps only those last blocks, so a
       head has no block below ``block_count - nextn_predict_layers``. A full
       model that carries its own head has every block and passes. The
       Gemma 4 assistant writes ``nextn_predict_layers = block_count``.
    6. A speculative-decoding drafter's architecture.
    7. A reranker: ``pooling_type`` RANK.
    8. An encoder architecture, or any other ``pooling_type``. llama.cpp's
       converter writes the pooling type only for sentence-transformers
       repositories and rerankers, so on a generative architecture such as
       ``qwen3`` it is the only sign of an embedding model.
    9. A speech or audio architecture.
    10. A diffusion language model.

    Args:
        meta: A parsed GGUF metadata mapping, as :func:`parse` returns.
        names: The file's tensor names.

    Returns:
        A plain-English reason for the panel, or None for a model.
    """
    kind = meta.get("general.type")
    if isinstance(kind, str) and kind not in ("", "model"):
        # The value came from the file, so it is shortened before it is shown.
        return _KIND_REASONS.get(
            kind, f"This file is a GGUF of type “{kind[:40]}”, not a model."
        )
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return NO_ARCHITECTURE_REASON
    if arch == "clip":
        return PROJECTOR_REASON
    split = (_integer_at(meta, "split.count") or 0) > 1
    if not split and not names:
        return NO_WEIGHTS_REASON
    blocks = _integer_at(meta, f"{arch}.block_count")
    heads = _integer_at(meta, f"{arch}.nextn_predict_layers")
    if blocks is not None and heads is not None and blocks > 0 and heads > 0:
        trunk = blocks - heads
        if trunk <= 0:
            return MTP_REASON
        layers = (int(m[1]) for m in map(_BLOCK.match, names) if m)
        # A split's shards each hold some of the blocks, so only a whole file
        # can show that the trunk is missing.
        if not split and not any(layer < trunk for layer in layers):
            return MTP_REASON
    if arch in DRAFTERS:
        return DRAFTER_REASON
    pooling = _integer_at(meta, f"{arch}.pooling_type")
    if pooling == _RANK_POOLING:
        return RERANKER_REASON
    if arch in ENCODERS or (pooling is not None and pooling > 0):
        return EMBEDDING_REASON
    if arch in SPEECH:
        return SPEECH_REASON
    if arch in DIFFUSION:
        return DIFFUSION_REASON
    return None
