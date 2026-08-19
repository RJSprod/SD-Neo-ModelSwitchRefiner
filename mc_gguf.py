"""Just enough GGUF to answer "how much context fits in this much VRAM".

Section 12 asks for a context estimate that is *model-specific* rather than a
tokens-per-gigabyte constant, and the numbers that make it model-specific are
all in the file's own metadata header: how many blocks it has, how wide they
are, and how many key/value heads each one keeps. So this reads that header.

Scope, deliberately small: the metadata block only. GGUF puts every key/value
pair before the first tensor, so the answer is a few kilobytes into a file that
is tens of gigabytes, and nothing below ever reads past it. No tensor data is
touched, no weights are loaded, and a 27 GB model is described in a few
milliseconds without disturbing whatever is currently on the card.

Format reference: ggml-org/ggml docs/gguf.md. Versions 2 and 3 are read (both
use 64-bit counts and lengths); version 1 used 32-bit ones and is refused with
a sentence rather than silently misparsed -- nothing has shipped v1 since 2023.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"GGUF"

# gguf_metadata_value_type
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

_SCALARS = {
    UINT8: ("<B", 1), INT8: ("<b", 1),
    UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4),
    FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8),
    FLOAT64: ("<d", 8),
}

MAX_METADATA_BYTES = 64 * 1024 * 1024
"""Ceiling on how much header this will read before giving up.

A tokenizer vocabulary is a metadata array and a large one runs to a few
megabytes, so the limit has to be generous. It exists because the alternative
to a limit is trusting a length prefix out of a file the user picked, and a
corrupt one would otherwise ask for an allocation the size of the disk.
"""


class GgufError(ValueError):
    """The file is not a GGUF this module can read."""


@dataclass(frozen=True)
class Gguf:
    """A model's shape, as its own header describes it."""

    path: Path
    architecture: str
    metadata: dict
    file_bytes: int
    tensor_count: int

    # -- the five numbers the estimator actually needs -------------------- #

    @property
    def block_count(self) -> int:
        """Transformer blocks. Each one keeps its own K and V cache."""
        return int(self._arch("block_count", 0))

    @property
    def context_length(self) -> int:
        """The context the model was trained for -- its own ceiling (section 12)."""
        return int(self._arch("context_length", 0))

    @property
    def embedding_length(self) -> int:
        return int(self._arch("embedding_length", 0))

    @property
    def head_count(self) -> int:
        return int(self._arch("attention.head_count", 0))

    @property
    def head_count_kv(self) -> int:
        """Key/value heads. Fewer than ``head_count`` under GQA, which is most
        modern models and is exactly why a per-model figure matters: a
        grouped-query model's cache can be a quarter the size of what a
        multi-head estimate would predict for the same width."""
        return int(self._arch("attention.head_count_kv", self.head_count))

    @property
    def key_length(self) -> int:
        """Per-head key width, from the file when it says so.

        Only some architectures record it. When they do it is authoritative --
        a model with a different key and value width, or with a head dimension
        that is not ``embedding_length / head_count``, is described correctly
        only by these keys.
        """
        declared = int(self._arch("attention.key_length", 0))
        return declared or self._head_dim

    @property
    def value_length(self) -> int:
        declared = int(self._arch("attention.value_length", 0))
        return declared or self._head_dim

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name") or self.path.stem)

    @property
    def usable(self) -> bool:
        """Whether the header carried enough to size a cache from."""
        return bool(self.block_count and self.head_count_kv and (self.key_length or self.value_length))

    @property
    def _head_dim(self) -> int:
        if self.head_count <= 0 or self.embedding_length <= 0:
            return 0
        return self.embedding_length // self.head_count

    def _arch(self, suffix: str, default):
        return self.metadata.get(f"{self.architecture}.{suffix}", default)


def read(path: str | Path) -> Gguf:
    """Read ``path``'s metadata header. Raises :class:`GgufError` on anything else."""
    file_path = Path(path).expanduser()
    try:
        size = file_path.stat().st_size
    except OSError as exc:
        raise GgufError(f"{file_path} cannot be read ({exc})") from exc

    with file_path.open("rb") as handle:
        magic = handle.read(4)
        if magic != MAGIC:
            raise GgufError(f"{file_path.name} is not a GGUF file")
        version, = struct.unpack("<I", _exactly(handle, 4))
        if version < 2:
            raise GgufError(
                f"{file_path.name} is GGUF version {version}; this reads version 2 and later"
            )
        tensor_count, kv_count = struct.unpack("<QQ", _exactly(handle, 16))
        reader = _Reader(handle)
        metadata: dict = {}
        for _ in range(min(int(kv_count), 1_000_000)):
            key = reader.string()
            value_type, = struct.unpack("<I", reader.take(4))
            metadata[key] = reader.value(value_type)

    architecture = str(metadata.get("general.architecture") or "")
    return Gguf(path=file_path, architecture=architecture, metadata=metadata,
                file_bytes=size, tensor_count=int(tensor_count))


def describe(path: str | Path) -> Gguf | None:
    """:func:`read`, or ``None`` when the file cannot be described.

    Used by the UI, where a model whose header will not parse is a reason to
    fall back to a coarser estimate and say so -- never a reason to refuse to
    draw the panel.
    """
    try:
        return read(path)
    except (GgufError, OSError, struct.error):
        return None


class _Reader:
    """Sequential metadata reader with a budget. See ``MAX_METADATA_BYTES``."""

    def __init__(self, handle):
        self._handle = handle
        self._spent = 0

    def take(self, count: int) -> bytes:
        count = int(count)
        if count < 0 or self._spent + count > MAX_METADATA_BYTES:
            raise GgufError("GGUF metadata is implausibly large; refusing to read further")
        self._spent += count
        return _exactly(self._handle, count)

    def string(self) -> str:
        length, = struct.unpack("<Q", self.take(8))
        return self.take(length).decode("utf-8", "replace")

    def value(self, value_type: int):
        if value_type == STRING:
            return self.string()
        if value_type == ARRAY:
            element_type, count = struct.unpack("<IQ", self.take(12))
            # Vocabularies run to hundreds of thousands of entries and nothing
            # here reads one, so an array is walked to keep the stream aligned
            # and only kept when it is small enough to be a real setting.
            values = [self.value(int(element_type)) for _ in range(int(count))]
            return values if len(values) <= 1024 else len(values)
        layout = _SCALARS.get(value_type)
        if layout is None:
            raise GgufError(f"unknown GGUF metadata value type {value_type}")
        fmt, width = layout
        return struct.unpack(fmt, self.take(width))[0]


def _exactly(handle, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise GgufError("GGUF file ended inside its metadata header")
    return data
