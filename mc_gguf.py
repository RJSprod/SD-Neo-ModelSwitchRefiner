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


def _whole(value, default: int = 0) -> int:
    """One metadata value as an integer, and ``default`` for anything else.

    Total on purpose. Everything below is read out of a file somebody else
    wrote, and a header that says something unexpected is a reason to estimate
    coarsely and say so -- never a reason for a property access to raise into a
    panel that was only drawing itself.
    """
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


DEFAULT_EXPERT_SHARE = 0.85
"""What fraction of a mixture-of-experts model's weights are experts, when its
header does not say enough to work it out. Every published MoE is above this."""


@dataclass(frozen=True)
class Gguf:
    """A model's shape, as its own header describes it."""

    path: Path
    architecture: str
    metadata: dict
    file_bytes: int
    tensor_count: int

    # -- the numbers the estimator actually needs ------------------------- #

    @property
    def block_count(self) -> int:
        """Transformer blocks. Each one keeps its own K and V cache."""
        return _whole(self._arch("block_count", 0))

    @property
    def context_length(self) -> int:
        """The context the model was trained for -- its own ceiling (section 12)."""
        return _whole(self._arch("context_length", 0))

    @property
    def embedding_length(self) -> int:
        return _whole(self._arch("embedding_length", 0))

    # -- mixture of experts, which changes what "offload less" should mean -- #

    @property
    def expert_count(self) -> int:
        """Experts per block, or 0 for a dense model."""
        return _whole(self._arch("expert_count", 0))

    @property
    def expert_used_count(self) -> int:
        """Experts actually consulted per token. The number speed follows.

        A 26B model that uses two experts of 4B does roughly 4B of arithmetic
        per token, which is why it can be three times faster than a dense 12B
        on the same machine while being twice the file. Nothing else in this
        header explains that, and without it a chooser sorted by size is sorted
        by the wrong thing.
        """
        return _whole(self._arch("expert_used_count", 0))

    @property
    def expert_feed_forward_length(self) -> int:
        return _whole(self._arch("expert_feed_forward_length", 0))

    @property
    def feed_forward_length(self) -> int:
        return _whole(self._arch("feed_forward_length", 0))

    @property
    def mixture_of_experts(self) -> bool:
        """Whether this model has experts at all."""
        return self.expert_count > 1

    @property
    def expert_share(self) -> float:
        """Roughly what fraction of the weights are expert tensors.

        Estimated from the header rather than measured from the tensors,
        because the tensor list is past the metadata block this module has
        promised not to read past -- and because the number is only ever used
        to *choose between two placements*, where being right to a few per cent
        is as good as being exact.

        Three matrices per expert (gate, up, down), each ``embedding_length ×
        expert_feed_forward_length``, against everything else in a block:
        attention projections and the shared feed-forward, if any. A model whose
        header does not answer gets the conservative default -- experts dominate
        every published mixture-of-experts, so under-estimating their share is
        the direction that keeps a placement fitting.
        """
        if not self.mixture_of_experts:
            return 0.0
        width = self.embedding_length
        expert_ff = self.expert_feed_forward_length or self.feed_forward_length
        if width <= 0 or expert_ff <= 0:
            return DEFAULT_EXPERT_SHARE

        experts = 3 * self.expert_count * width * expert_ff
        heads = max(self.head_counts[0] if self.head_counts else 0, 1)
        kv_heads = max(self.head_counts_kv[0] if self.head_counts_kv else heads, 1)
        head_width = width // heads if heads else width
        attention = 2 * width * width + 2 * width * (kv_heads * head_width)
        shared = 3 * width * self.feed_forward_length if self.expert_feed_forward_length else 0
        rest = attention + shared
        if experts + rest <= 0:
            return DEFAULT_EXPERT_SHARE
        return min(max(experts / (experts + rest), 0.0), 0.98)

    # -- per block, because the attention shape is not always one shape ---- #
    #
    # GGUF allows the four attention keys below to be *arrays with one entry
    # per block*, and llama.cpp writes them that way for every architecture
    # whose blocks differ from one another: hybrid attention/state-space models
    # where only some blocks attend at all, and interleaved local/global
    # designs where the sliding-window blocks are shaped differently from the
    # full-attention ones. Reading such a value as a scalar is not a rounding
    # error -- ``int([8, 8, 0, 8])`` raises, and it raised *inside a property*,
    # which put "int() argument must be ... not 'list'" in front of a user who
    # had done nothing but choose a model.
    #
    # So the per-block tuple is the real answer here and the scalars below are
    # derived from it, rather than the other way round.

    @property
    def head_counts(self) -> tuple[int, ...]:
        """Attention heads, one entry per block."""
        return self._spread(self._numbers("attention.head_count"))

    @property
    def head_counts_kv(self) -> tuple[int, ...]:
        """Key/value heads per block. Fewer than ``head_counts`` under GQA,
        which is most modern models and is exactly why a per-model figure
        matters: a grouped-query model's cache can be a quarter the size of
        what a multi-head estimate would predict for the same width. Zero for
        a block that keeps no cache at all, which is what makes a hybrid
        model's cache a fraction of what its block count suggests."""
        found = self._numbers("attention.head_count_kv")
        return self._spread(found) if found else self.head_counts

    @property
    def key_lengths(self) -> tuple[int, ...]:
        """Per-head key width per block, from the file when it says so.

        Only some architectures record it. When they do it is authoritative --
        a model with a different key and value width, or with a head dimension
        that is not ``embedding_length / head_count``, is described correctly
        only by these keys.
        """
        declared = self._numbers("attention.key_length")
        return self._spread(declared) if declared else self._head_dims

    @property
    def value_lengths(self) -> tuple[int, ...]:
        declared = self._numbers("attention.value_length")
        return self._spread(declared) if declared else self._head_dims

    # -- one number each, for a status line ------------------------------- #

    @property
    def head_count(self) -> int:
        return max(self.head_counts, default=0)

    @property
    def head_count_kv(self) -> int:
        return max(self.head_counts_kv, default=0)

    @property
    def key_length(self) -> int:
        return max(self.key_lengths, default=0)

    @property
    def value_length(self) -> int:
        return max(self.value_lengths, default=0)

    @property
    def attending_blocks(self) -> int:
        """Blocks that keep a cache at all. Equal to ``block_count`` for an
        ordinary transformer and smaller for a hybrid one."""
        return sum(1 for heads in self.head_counts_kv if heads > 0)

    @property
    def uniform_attention(self) -> bool:
        """Whether every block is shaped the same, which decides only how the
        panel words itself -- the arithmetic is per block either way."""
        shapes = set(zip(self.head_counts_kv, self.key_lengths, self.value_lengths))
        return len(shapes) <= 1

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name") or self.path.stem)

    @property
    def usable(self) -> bool:
        """Whether the header carried enough to size a cache from."""
        return bool(self.block_count and any(self.head_counts_kv)
                    and (self.key_length or self.value_length))

    @property
    def _head_dims(self) -> tuple[int, ...]:
        width = self.embedding_length
        return tuple(width // heads if heads > 0 and width > 0 else 0
                     for heads in self.head_counts)

    def _numbers(self, suffix: str) -> tuple[int, ...]:
        """One architecture key as whole numbers, scalar or array alike."""
        value = self._arch(suffix, None)
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(_whole(entry) for entry in value)
        return (_whole(value),)

    def _spread(self, values: tuple[int, ...]) -> tuple[int, ...]:
        """``values`` as exactly one entry per block.

        A scalar applies to every block. A short array -- which a header should
        not contain, and a truncated one does -- is extended with its last
        entry rather than dropped, because a cache sized from four fifths of a
        model is a worse answer than one sized from an assumption stated here.
        """
        blocks = self.block_count
        if blocks <= 0:
            return tuple(values)
        if not values:
            return (0,) * blocks
        if len(values) == 1:
            return (values[0],) * blocks
        if len(values) >= blocks:
            return tuple(values[:blocks])
        return tuple(values) + (values[-1],) * (blocks - len(values))

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
