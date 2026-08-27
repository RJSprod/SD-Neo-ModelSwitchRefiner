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

KEPT_ARRAY_LENGTH = 1024
"""Longest metadata array kept as its values rather than as its length.

A settings array -- a rope scaling table, a list of head counts -- is a handful
of numbers and is worth having. A tokenizer vocabulary is a quarter of a million
strings, is read by nothing here (see :class:`Gguf`, which asks only for
``general.*`` and ``{architecture}.*`` keys), and is the reason this file used
to take the better part of a second to parse.
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

    # -- the other half of a hybrid model's memory ------------------------ #
    #
    # A block that keeps no key/value cache is not a block that keeps nothing.
    # Qwen 3.8 interleaves Gated DeltaNet blocks with periodic full attention:
    # the attending blocks grow a cache with the context, and the rest hold a
    # fixed recurrent state per sequence that does not grow at all. Counting
    # only the first half reports a hybrid 27B as cheaper than it is, and the
    # error goes the wrong way -- it is an out-of-memory error at load rather
    # than an avoidable eviction.
    #
    # The keys below are llama.cpp's own, written for every state-space and
    # linear-attention architecture it supports. Where a build writes none of
    # them the state is unknown rather than zero, and
    # ``mc_llm_context.recurrent_bytes`` charges a conservative allowance
    # instead -- see :attr:`recurrent_blocks`.

    @property
    def state_size(self) -> int:
        """Width of one recurrent state slot, ``ssm.state_size`` in the header."""
        return _whole(self._arch("ssm.state_size", 0))

    @property
    def convolution_kernel(self) -> int:
        """``ssm.conv_kernel``: how many steps the short convolution remembers."""
        return _whole(self._arch("ssm.conv_kernel", 0))

    @property
    def inner_size(self) -> int:
        """``ssm.inner_size``: the width the recurrent path runs at."""
        return _whole(self._arch("ssm.inner_size", 0))

    @property
    def group_count(self) -> int:
        """``ssm.group_count``, or one group when the header does not say."""
        return _whole(self._arch("ssm.group_count", 0)) or 1

    @property
    def recurrent_blocks(self) -> int:
        """Blocks holding a recurrent state rather than a key/value cache.

        Derived from the per-block attention array rather than from a count of
        its own, because that array is where llama.cpp already records the
        difference: a Gated DeltaNet block is written with zero key/value
        heads, which is the same fact :attr:`attending_blocks` reads from the
        other side.
        """
        return max(self.block_count - self.attending_blocks, 0)

    @property
    def recurrent(self) -> bool:
        """Whether any block keeps a recurrent state at all."""
        return self.recurrent_blocks > 0

    @property
    def recurrent_state_described(self) -> bool:
        """Whether the header carried enough to size that state exactly."""
        return bool(self.recurrent and self.state_size and self.inner_size
                    and self.convolution_kernel)

    @property
    def recurrent_state_elements(self) -> int:
        """Elements of state one recurrent block holds, per sequence.

        llama.cpp's own arithmetic, in its own terms: a convolution window of
        ``conv_kernel - 1`` steps over the inner width and the two group
        projections beside it, plus the recurrent state itself at
        ``state_size x inner_size``. Zero when the header does not describe it,
        which is a question for the caller rather than a licence to assume the
        state is free.
        """
        if not self.recurrent_state_described:
            return 0
        convolution = (self.convolution_kernel - 1) * (
            self.inner_size + 2 * self.group_count * self.state_size)
        return max(convolution, 0) + self.state_size * self.inner_size

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
        reader = _Reader(handle, size)
        metadata: dict = {}
        for _ in range(min(int(kv_count), 1_000_000)):
            key = reader.string()
            value_type, = struct.unpack("<I", reader.take(4))
            metadata[key] = reader.value(value_type)

    architecture = str(metadata.get("general.architecture") or "")
    return Gguf(path=file_path, architecture=architecture, metadata=metadata,
                file_bytes=size, tensor_count=int(tensor_count))


_described: dict[Path, tuple[tuple, "Gguf"]] = {}
"""Headers already parsed, against the file stamp they were parsed from.

Small and deliberate. :func:`describe` is called on the path that leads to
*reusing* a running llama-server -- before every request, once the active plan
has moved -- and the header it reads cannot have changed unless the file has.
Keeping the answer turns that call from a parse of the whole header into a
stat, which is the difference between a second of silence before a warm request
and none.
"""

_DESCRIBED_LIMIT = 8
"""How many models are remembered before the lot is dropped.

A ceiling rather than a policy. Nobody has eight backbones in play at once, and
an eviction that gets it wrong costs one re-parse -- so the simplest rule that
cannot grow without bound is the right one here.
"""


def _stamp(path: Path) -> tuple:
    """What has to be unchanged for a parsed header to still be true."""
    status = path.stat()
    return (status.st_mtime_ns, status.st_size)


def forget(path: str | Path | None = None) -> None:
    """Drop remembered headers, for ``path`` or for everything.

    Nothing in the extension needs this -- the stamp already notices a file that
    changed -- but a test that writes two different headers to one path inside a
    single mtime tick does, and so does anybody debugging a header by hand.
    """
    if path is None:
        _described.clear()
        return
    _described.pop(Path(path).expanduser(), None)


def describe(path: str | Path) -> Gguf | None:
    """:func:`read`, or ``None`` when the file cannot be described.

    Used by the UI, where a model whose header will not parse is a reason to
    fall back to a coarser estimate and say so -- never a reason to refuse to
    draw the panel.

    The answer is remembered against the file's size and modification time, so a
    second call on an unchanged file costs a ``stat`` instead of a parse. A file
    that has been touched at all is parsed again: the stamp is the guarantee,
    not the path.

    A failure is not remembered. It is cheap -- a header that will not parse
    fails in its first few bytes -- and a model being copied into place is
    exactly the case that must not be answered from a cache of "no".
    """
    file_path = Path(path).expanduser()
    try:
        stamp = _stamp(file_path)
    except OSError:
        return None

    remembered = _described.get(file_path)
    if remembered is not None and remembered[0] == stamp:
        return remembered[1]

    try:
        described = read(file_path)
    except (GgufError, OSError, struct.error):
        return None

    if len(_described) >= _DESCRIBED_LIMIT:
        _described.clear()
    _described[file_path] = (stamp, described)
    return described


class _Reader:
    """Sequential metadata reader with a budget. See ``MAX_METADATA_BYTES``."""

    def __init__(self, handle, file_bytes: int = 0):
        self._handle = handle
        self._spent = 0
        self._file_bytes = int(file_bytes)

    def take(self, count: int) -> bytes:
        count = int(count)
        if count < 0 or self._spent + count > MAX_METADATA_BYTES:
            raise GgufError("GGUF metadata is implausibly large; refusing to read further")
        self._spent += count
        return _exactly(self._handle, count)

    def skip(self, count: int) -> None:
        """Step over ``count`` bytes, spending the budget and keeping nothing.

        A seek rather than a read, which is the whole point: the bytes being
        stepped over are a tokenizer vocabulary, and reading them costs the same
        whether or not anything is done with them afterwards.

        The end-of-file check that :func:`_exactly` performs for free has to be
        done by hand here, because seeking past the end of a file succeeds
        silently. Without it a truncated GGUF would be reported as one with a
        very short vocabulary rather than as the damaged file it is.
        """
        count = int(count)
        if count < 0 or self._spent + count > MAX_METADATA_BYTES:
            raise GgufError("GGUF metadata is implausibly large; refusing to read further")
        self._spent += count
        landing = self._handle.seek(count, 1)
        if self._file_bytes and landing > self._file_bytes:
            raise GgufError("GGUF file ended inside its metadata header")

    def string(self) -> str:
        length, = struct.unpack("<Q", self.take(8))
        return self.take(length).decode("utf-8", "replace")

    def value(self, value_type: int):
        if value_type == STRING:
            return self.string()
        if value_type == ARRAY:
            element_type, count = struct.unpack("<IQ", self.take(12))
            element_type, count = int(element_type), int(count)
            # Vocabularies run to hundreds of thousands of entries and nothing
            # here reads one, so a long array is walked to keep the stream
            # aligned and reported as its length. Only an array short enough to
            # be a real setting is built.
            if count > KEPT_ARRAY_LENGTH:
                self._walk(element_type, count)
                return count
            return [self.value(element_type) for _ in range(count)]
        layout = _SCALARS.get(value_type)
        if layout is None:
            raise GgufError(f"unknown GGUF metadata value type {value_type}")
        fmt, width = layout
        return struct.unpack(fmt, self.take(width))[0]

    def _walk(self, element_type: int, count: int) -> None:
        """Advance past an array's payload without materialising any of it.

        The stream is sequential, so an array that is not wanted still has to be
        stepped over exactly -- but stepping over it and building it are two
        very different amounts of work. A 262,144-entry vocabulary built as a
        Python list of ``str`` and thrown away immediately afterwards was
        costing the better part of a second per call, on a call this extension
        makes before every language-model request.

        Fixed-width elements are one seek for the lot. Strings still cost one
        length prefix each, because that is the only way to know where the next
        one starts, but not the decode and not the object.
        """
        layout = _SCALARS.get(element_type)
        if layout is not None:
            self.skip(layout[1] * count)
            return
        if element_type == STRING:
            for _ in range(count):
                length, = struct.unpack("<Q", self.take(8))
                self.skip(length)
            return
        if element_type == ARRAY:
            for _ in range(count):
                nested_type, nested_count = struct.unpack("<IQ", self.take(12))
                self._walk(int(nested_type), int(nested_count))
            return
        raise GgufError(f"unknown GGUF metadata value type {element_type}")


def _exactly(handle, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise GgufError("GGUF file ended inside its metadata header")
    return data
