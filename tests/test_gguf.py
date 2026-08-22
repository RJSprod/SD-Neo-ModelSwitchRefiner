"""The GGUF metadata reader.

Built against synthetic files rather than a real model, for the obvious reason:
the smallest useful GGUF is a couple of gigabytes and the header is a few
hundred bytes. Every file below is written by hand from the format
specification, which also means a test failing here points at the reader rather
than at whatever model happened to be on the machine.
"""

from __future__ import annotations

import struct

import pytest

import mc_gguf


def _string(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pair(key: str, value_type: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", value_type) + payload


def _u32(key: str, value: int) -> bytes:
    return _pair(key, mc_gguf.UINT32, struct.pack("<I", value))


def _text(key: str, value: str) -> bytes:
    return _pair(key, mc_gguf.STRING, _string(value))


def _u32s(key: str, values) -> bytes:
    """One key whose value is an array of numbers, one per block."""
    payload = struct.pack("<IQ", mc_gguf.UINT32, len(values))
    payload += b"".join(struct.pack("<I", value) for value in values)
    return _pair(key, mc_gguf.ARRAY, payload)


def write_gguf(path, metadata: bytes, count: int, version: int = 3,
               tensors: int = 291, padding: int = 4096):
    header = mc_gguf.MAGIC + struct.pack("<I", version) + struct.pack("<QQ", tensors, count)
    path.write_bytes(header + metadata + b"\x00" * padding)
    return path


@pytest.fixture(autouse=True)
def _forget_remembered_headers():
    """``describe`` remembers what it parsed. Tests must not inherit that."""
    mc_gguf.forget()
    yield
    mc_gguf.forget()


@pytest.fixture
def model(tmp_path):
    """A grouped-query model: 32 blocks, 32 heads, 8 KV heads, 4096 wide."""
    metadata = b"".join([
        _text("general.architecture", "llama"),
        _text("general.name", "Test Model"),
        _u32("llama.block_count", 32),
        _u32("llama.context_length", 8192),
        _u32("llama.embedding_length", 4096),
        _u32("llama.attention.head_count", 32),
        _u32("llama.attention.head_count_kv", 8),
    ])
    return write_gguf(tmp_path / "model.gguf", metadata, 7)


class TestReading:
    def test_it_reads_the_shape_the_estimator_needs(self, model):
        found = mc_gguf.read(model)

        assert found.architecture == "llama"
        assert found.name == "Test Model"
        assert found.block_count == 32
        assert found.context_length == 8192
        assert found.head_count_kv == 8
        assert found.usable

    def test_head_width_is_derived_when_the_file_does_not_state_it(self, model):
        # 4096 / 32 heads. Most architectures leave key_length and value_length
        # out and expect exactly this arithmetic.
        found = mc_gguf.read(model)

        assert found.key_length == 128
        assert found.value_length == 128

    def test_a_declared_head_width_wins_over_the_derived_one(self, tmp_path):
        # A model whose head dimension is not embedding / heads is described
        # correctly only by its own keys, so they have to be preferred.
        metadata = b"".join([
            _text("general.architecture", "deepseek2"),
            _u32("deepseek2.block_count", 8),
            _u32("deepseek2.embedding_length", 4096),
            _u32("deepseek2.attention.head_count", 32),
            _u32("deepseek2.attention.head_count_kv", 32),
            _u32("deepseek2.attention.key_length", 192),
            _u32("deepseek2.attention.value_length", 128),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "d.gguf", metadata, 6))

        assert (found.key_length, found.value_length) == (192, 128)

    def test_kv_heads_default_to_the_head_count(self, tmp_path):
        """A multi-head model states head_count and nothing else."""
        metadata = b"".join([
            _text("general.architecture", "llama"),
            _u32("llama.block_count", 4),
            _u32("llama.embedding_length", 512),
            _u32("llama.attention.head_count", 8),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "mha.gguf", metadata, 4))

        assert found.head_count_kv == 8

    def test_arrays_are_walked_so_later_keys_still_parse(self, tmp_path):
        """A vocabulary sits between the keys that matter; the stream has to
        stay aligned across it or everything after it is garbage."""
        vocabulary = (struct.pack("<IQ", mc_gguf.STRING, 3)
                      + _string("a") + _string("b") + _string("c"))
        metadata = b"".join([
            _text("general.architecture", "llama"),
            _pair("tokenizer.ggml.tokens", mc_gguf.ARRAY, vocabulary),
            _u32("llama.block_count", 16),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "v.gguf", metadata, 3))

        assert found.block_count == 16

    def test_a_huge_array_is_counted_rather_than_kept(self, tmp_path):
        """Keeping a 150,000-entry vocabulary in memory to answer a question
        about block count would be absurd; the length is enough."""
        entries = 2000
        payload = struct.pack("<IQ", mc_gguf.UINT32, entries)
        payload += b"".join(struct.pack("<I", index) for index in range(entries))
        metadata = b"".join([
            _text("general.architecture", "llama"),
            _pair("tokenizer.ggml.token_type", mc_gguf.ARRAY, payload),
            _u32("llama.block_count", 16),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "big.gguf", metadata, 3))

        assert found.metadata["tokenizer.ggml.token_type"] == entries
        assert found.block_count == 16


class TestRefusals:
    def test_a_file_that_is_not_gguf_is_named_as_such(self, tmp_path):
        path = tmp_path / "not.gguf"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 64)

        with pytest.raises(mc_gguf.GgufError, match="not a GGUF"):
            mc_gguf.read(path)

    def test_version_one_is_refused_rather_than_misread(self, tmp_path):
        """v1 used 32-bit counts. Reading it with 64-bit ones would not fail --
        it would produce confident nonsense, which is worse."""
        path = write_gguf(tmp_path / "old.gguf", b"", 0, version=1)

        with pytest.raises(mc_gguf.GgufError, match="version 1"):
            mc_gguf.read(path)

    def test_a_truncated_header_is_an_error_not_a_partial_answer(self, tmp_path):
        path = tmp_path / "cut.gguf"
        path.write_bytes(mc_gguf.MAGIC + struct.pack("<I", 3) + struct.pack("<QQ", 1, 4))

        with pytest.raises(mc_gguf.GgufError, match="ended inside"):
            mc_gguf.read(path)

    def test_an_implausible_length_is_refused_before_it_is_allocated(self, tmp_path):
        """The length prefix comes out of a file the user chose. Trusting it is
        how a corrupt header becomes an allocation the size of the disk."""
        path = tmp_path / "huge.gguf"
        path.write_bytes(mc_gguf.MAGIC + struct.pack("<I", 3) + struct.pack("<QQ", 1, 1)
                         + struct.pack("<Q", 2 ** 40))

        with pytest.raises(mc_gguf.GgufError, match="implausibly large"):
            mc_gguf.read(path)

    def test_describe_answers_none_instead_of_raising(self, tmp_path):
        """The UI calls this. A model whose header will not parse is a reason to
        fall back to a coarser estimate, never to refuse to draw the panel."""
        path = tmp_path / "bad.gguf"
        path.write_bytes(b"nope")

        assert mc_gguf.describe(path) is None
        assert mc_gguf.describe(tmp_path / "absent.gguf") is None


class TestPerBlockAttention:
    """GGUF lets the attention keys be arrays with one entry per block, and
    llama.cpp writes them that way for every architecture whose blocks differ:
    hybrid attention/state-space models where only some blocks attend, and
    interleaved local/global designs where the two kinds are shaped
    differently.

    Reading one as a scalar is not a rounding error. ``int([8, 8, 0, 8])``
    raises, and it raised inside a property — so choosing such a model put
    "int() argument must be ... not 'list'" in front of a user who had done
    nothing but pick a file, and did it again on every reply.
    """

    @pytest.fixture
    def hybrid(self, tmp_path):
        """Six blocks; two of them keep no cache at all."""
        metadata = b"".join([
            _text("general.architecture", "gemma3"),
            _u32("gemma3.block_count", 6),
            _u32("gemma3.context_length", 131072),
            _u32("gemma3.embedding_length", 4096),
            _u32("gemma3.attention.head_count", 32),
            _u32s("gemma3.attention.head_count_kv", [8, 8, 0, 8, 8, 0]),
        ])
        return write_gguf(tmp_path / "hybrid.gguf", metadata, 6)

    def test_an_array_of_kv_heads_reads_as_one_entry_per_block(self, hybrid):
        found = mc_gguf.read(hybrid)

        assert found.head_counts_kv == (8, 8, 0, 8, 8, 0)

    def test_it_is_usable_even_though_some_blocks_keep_no_cache(self, hybrid):
        assert mc_gguf.read(hybrid).usable

    def test_the_scalar_view_answers_without_raising(self, hybrid):
        """Everything that only wants one number still gets one."""
        found = mc_gguf.read(hybrid)

        assert found.head_count_kv == 8
        assert found.key_length == 128
        assert found.attending_blocks == 4
        assert not found.uniform_attention

    def test_a_scalar_still_applies_to_every_block(self, model):
        found = mc_gguf.read(model)

        assert found.head_counts_kv == (8,) * 32
        assert found.uniform_attention
        assert found.attending_blocks == 32

    def test_per_block_key_widths_are_read_and_not_derived(self, tmp_path):
        metadata = b"".join([
            _text("general.architecture", "jamba"),
            _u32("jamba.block_count", 4),
            _u32("jamba.embedding_length", 4096),
            _u32("jamba.attention.head_count", 32),
            _u32s("jamba.attention.head_count_kv", [8, 0, 8, 0]),
            _u32s("jamba.attention.key_length", [64, 0, 64, 0]),
            _u32s("jamba.attention.value_length", [64, 0, 64, 0]),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "j.gguf", metadata, 7))

        assert found.key_lengths == (64, 0, 64, 0)
        assert found.value_lengths == (64, 0, 64, 0)

    def test_a_short_array_is_extended_rather_than_dropped(self, tmp_path):
        """A header should not contain one; a truncated file does, and a cache
        sized from four fifths of a model is the worse answer."""
        metadata = b"".join([
            _text("general.architecture", "llama"),
            _u32("llama.block_count", 6),
            _u32("llama.embedding_length", 4096),
            _u32("llama.attention.head_count", 32),
            _u32s("llama.attention.head_count_kv", [8, 8]),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "short.gguf", metadata, 5))

        assert found.head_counts_kv == (8, 8, 8, 8, 8, 8)

    def test_a_value_that_is_not_a_number_does_not_raise_out_of_a_property(self, tmp_path):
        metadata = b"".join([
            _text("general.architecture", "llama"),
            _u32("llama.block_count", 4),
            _u32("llama.embedding_length", 4096),
            _text("llama.attention.head_count", "thirty-two"),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "odd.gguf", metadata, 4))

        assert found.head_count == 0
        assert not found.usable


class TestSkippingWhatIsNotWanted:
    """A vocabulary has to be stepped over exactly and not built.

    Nothing in this module reads ``tokenizer.*``; :class:`mc_gguf.Gguf` asks only
    for ``general.*`` and ``{architecture}.*``. Building a quarter of a million
    strings to reach the key after them was costing the better part of a second
    per call, on a call the LLM runtime makes before a warm request.
    """

    def test_a_long_string_array_is_stepped_over_and_later_keys_survive(self, tmp_path):
        entries = 3000
        payload = struct.pack("<IQ", mc_gguf.STRING, entries)
        payload += b"".join(_string(f"token{index}") for index in range(entries))
        metadata = b"".join([
            _text("general.architecture", "llama"),
            _pair("tokenizer.ggml.tokens", mc_gguf.ARRAY, payload),
            _u32("llama.block_count", 48),
            _u32("llama.embedding_length", 4096),
            _u32("llama.attention.head_count", 32),
            _u32("llama.attention.head_count_kv", 8),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "vocab.gguf", metadata, 6))

        assert found.metadata["tokenizer.ggml.tokens"] == entries
        assert found.block_count == 48
        assert found.head_count_kv == 8
        assert found.usable

    def test_a_long_array_of_arrays_is_stepped_over_too(self, tmp_path):
        """Rarer, and the format allows it. A walk that cannot nest would leave
        the stream one length prefix out and misread everything after it."""
        inner = struct.pack("<IQ", mc_gguf.UINT32, 2) + struct.pack("<II", 1, 2)
        payload = struct.pack("<IQ", mc_gguf.ARRAY, 1100) + inner * 1100
        metadata = b"".join([
            _text("general.architecture", "llama"),
            _pair("tokenizer.ggml.merges", mc_gguf.ARRAY, payload),
            _u32("llama.block_count", 12),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "nested.gguf", metadata, 3))

        assert found.metadata["tokenizer.ggml.merges"] == 1100
        assert found.block_count == 12

    def test_a_short_array_is_still_kept_as_its_values(self, tmp_path):
        """The threshold is what separates a setting from a vocabulary. A rope
        table or a per-block head count is a handful of numbers and is read."""
        metadata = b"".join([
            _text("general.architecture", "jamba"),
            _u32("jamba.block_count", 4),
            _u32("jamba.embedding_length", 4096),
            _u32("jamba.attention.head_count", 32),
            _u32s("jamba.attention.head_count_kv", [8, 0, 8, 0]),
        ])
        found = mc_gguf.read(write_gguf(tmp_path / "kept.gguf", metadata, 5))

        assert found.metadata["jamba.attention.head_count_kv"] == [8, 0, 8, 0]

    def test_a_file_that_ends_inside_a_stepped_array_is_an_error(self, tmp_path):
        """Seeking past the end of a file succeeds silently, which is exactly
        how a truncated model would come back as one with a short vocabulary."""
        payload = struct.pack("<IQ", mc_gguf.UINT32, 4000)
        payload += b"".join(struct.pack("<I", index) for index in range(10))
        metadata = _pair("tokenizer.ggml.token_type", mc_gguf.ARRAY, payload)
        path = write_gguf(tmp_path / "cut.gguf", metadata, 1, padding=0)

        with pytest.raises(mc_gguf.GgufError, match="ended inside"):
            mc_gguf.read(path)

    def test_an_implausible_length_inside_a_stepped_array_is_still_refused(self, tmp_path):
        payload = struct.pack("<IQ", mc_gguf.STRING, 2000) + struct.pack("<Q", 2 ** 40)
        metadata = _pair("tokenizer.ggml.tokens", mc_gguf.ARRAY, payload)
        path = write_gguf(tmp_path / "greedy.gguf", metadata, 1)

        with pytest.raises(mc_gguf.GgufError, match="implausibly large"):
            mc_gguf.read(path)


class TestRemembering:
    """``describe`` is called before every LLM request once the plan has moved.

    Re-parsing the header to answer the same question every time is what put a
    second of silence in front of a warm generation, so the answer is kept
    against the file's own stamp.
    """

    def test_an_unchanged_file_is_parsed_once(self, model, monkeypatch):
        calls: list = []
        original = mc_gguf.read
        monkeypatch.setattr(mc_gguf, "read",
                            lambda path: (calls.append(path), original(path))[1])

        first = mc_gguf.describe(model)
        second = mc_gguf.describe(model)

        assert len(calls) == 1
        assert first is second
        assert second.block_count == 32

    def test_a_file_that_changed_is_parsed_again(self, model):
        before = mc_gguf.describe(model)
        write_gguf(model, b"".join([
            _text("general.architecture", "llama"),
            _u32("llama.block_count", 80),
            _u32("llama.embedding_length", 4096),
            _u32("llama.attention.head_count", 32),
            _u32("llama.attention.head_count_kv", 8),
        ]), 5)
        after = mc_gguf.describe(model)

        assert before.block_count == 32
        assert after.block_count == 80

    def test_a_file_rewritten_to_the_same_size_is_noticed_by_its_clock(self, model):
        """Size alone would miss it. The modification time is the other half."""
        import os

        before = mc_gguf.describe(model)
        replacement = b"".join([
            _text("general.architecture", "llama"),
            _text("general.name", "Test Modol"),
            _u32("llama.block_count", 32),
            _u32("llama.context_length", 8192),
            _u32("llama.embedding_length", 4096),
            _u32("llama.attention.head_count", 32),
            _u32("llama.attention.head_count_kv", 8),
        ])
        write_gguf(model, replacement, 7)
        stamp = model.stat()
        os.utime(model, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))

        assert before.name == "Test Model"
        assert mc_gguf.describe(model).name == "Test Modol"

    def test_forget_drops_what_was_remembered(self, model, monkeypatch):
        mc_gguf.describe(model)
        mc_gguf.forget(model)

        calls: list = []
        original = mc_gguf.read
        monkeypatch.setattr(mc_gguf, "read",
                            lambda path: (calls.append(path), original(path))[1])
        mc_gguf.describe(model)

        assert len(calls) == 1

    def test_a_header_that_will_not_parse_is_not_remembered(self, tmp_path):
        """A model being copied into place is exactly the file that fails once
        and succeeds a moment later, so "no" is never the remembered answer."""
        path = tmp_path / "arriving.gguf"
        path.write_bytes(b"nope")

        assert mc_gguf.describe(path) is None

        write_gguf(path, b"".join([
            _text("general.architecture", "llama"),
            _u32("llama.block_count", 24),
            _u32("llama.embedding_length", 4096),
            _u32("llama.attention.head_count", 32),
            _u32("llama.attention.head_count_kv", 8),
        ]), 5)

        assert mc_gguf.describe(path).block_count == 24

    def test_it_does_not_grow_without_bound(self, tmp_path):
        metadata = b"".join([
            _text("general.architecture", "llama"),
            _u32("llama.block_count", 8),
        ])
        for index in range(mc_gguf._DESCRIBED_LIMIT * 3):
            assert mc_gguf.describe(
                write_gguf(tmp_path / f"m{index}.gguf", metadata, 2)) is not None

        assert len(mc_gguf._described) <= mc_gguf._DESCRIBED_LIMIT
