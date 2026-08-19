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


def write_gguf(path, metadata: bytes, count: int, version: int = 3,
               tensors: int = 291, padding: int = 4096):
    header = mc_gguf.MAGIC + struct.pack("<I", version) + struct.pack("<QQ", tensors, count)
    path.write_bytes(header + metadata + b"\x00" * padding)
    return path


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
