"""Architecture detection and Stage 2 geometry (sections 6.2, 6.4, 6.5)."""

from __future__ import annotations

import json
import struct

import pytest

import mc_arch


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


class TestSnapping:
    @pytest.mark.parametrize(
        "value, alignment, expected",
        [
            (1024, 8, 1024),
            (1024, 16, 1024),
            (1020, 8, 1024),
            (1000, 64, 1024),
            (1, 16, 16),  # never collapses below one alignment step
            (0, 16, 16),
        ],
    )
    def test_snaps_to_nearest_multiple(self, value, alignment, expected):
        assert mc_arch.snap_dimension(value, alignment) == expected

    def test_result_is_always_on_the_grid(self):
        for alignment in (8, 16, 32, 64):
            for value in range(1, 3000, 37):
                assert mc_arch.snap_dimension(value, alignment) % alignment == 0


class TestScaledSize:
    def test_identity_multiplier_keeps_aligned_dimensions(self):
        assert mc_arch.scaled_size(1024, 1024, 1.0, 16) == (1024, 1024)

    def test_multiplier_scales_both_axes(self):
        assert mc_arch.scaled_size(1024, 512, 2.0, 16) == (2048, 1024)

    @pytest.mark.parametrize("multiplier", [1.0, 1.15, 1.25, 1.5, 1.75, 2.0])
    @pytest.mark.parametrize("size", [(1024, 1024), (1216, 832), (832, 1216), (1344, 768)])
    def test_aspect_ratio_is_preserved_within_the_grid(self, size, multiplier):
        """Section 6.5: aspect ratio preserved at all multiplier values.

        Snapping to the alignment grid can shift the ratio by at most half a
        step per axis, so the check is that the error stays within that bound
        rather than being exactly zero -- an off-grid size the model cannot
        accept would be the worse failure.
        """
        width, height = size
        alignment = 16
        out_w, out_h = mc_arch.scaled_size(width, height, multiplier, alignment)

        assert out_w % alignment == 0
        assert out_h % alignment == 0

        bound = (alignment / 2) * (1 / out_w + 1 / out_h) * 1.001
        assert mc_arch.aspect_ratio_delta((width, height), (out_w, out_h)) <= bound

    def test_alignment_is_not_hardcoded_to_eight(self):
        """Section 6.5 explicitly forbids assuming a multiple of 8."""
        assert mc_arch.scaled_size(1000, 1000, 1.0, 8) == (1000, 1000)
        assert mc_arch.scaled_size(1000, 1000, 1.0, 16) == (1008, 1008)
        assert mc_arch.scaled_size(1000, 1000, 1.0, 64) == (1024, 1024)


class TestArchitectureTable:
    def test_sdxl_aligns_to_eight(self):
        assert mc_arch.by_key("sdxl").alignment == 8

    def test_flux_families_align_to_sixteen(self):
        assert mc_arch.by_key("flux").alignment == 16
        assert mc_arch.by_key("flux2_9b").alignment == 16

    def test_unknown_falls_back_to_a_universally_safe_alignment(self):
        # 16 is a multiple of 8, so it is valid for an 8-aligned model too.
        assert mc_arch.UNKNOWN.alignment == 16
        assert mc_arch.UNKNOWN.alignment % 8 == 0

    def test_distilled_models_default_to_low_cfg(self):
        """Section 6.4: Flux-family models want CFG at or near 1.0."""
        assert mc_arch.by_key("flux2_9b").cfg == pytest.approx(1.0)
        assert mc_arch.by_key("flux").cfg == pytest.approx(1.0)
        assert mc_arch.by_key("sdxl").cfg > 1.0

    def test_by_key_never_raises_on_an_unknown_key(self):
        assert mc_arch.by_key("no-such-arch") is mc_arch.UNKNOWN


# --------------------------------------------------------------------------- #
# Header reading
# --------------------------------------------------------------------------- #


def write_safetensors(path, tensors: dict, metadata: dict | None = None):
    """Write a safetensors file with real headers but zeroed tensor data."""
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = 2
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if metadata:
        header["__metadata__"] = metadata

    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\0" * offset)
    return path


class TestHeaderReading:
    def test_reads_tensor_shapes_without_loading_data(self, tmp_path):
        path = write_safetensors(
            tmp_path / "model.safetensors",
            {"model.diffusion_model.foo.weight": ("F16", (320, 4, 3, 3))},
        )
        header = mc_arch.read_safetensors_header(str(path))
        assert header["model.diffusion_model.foo.weight"]["shape"] == [320, 4, 3, 3]

    def test_returns_none_for_a_non_safetensors_file(self, tmp_path):
        path = tmp_path / "not-a-model.txt"
        path.write_text("hello")
        assert mc_arch.read_safetensors_header(str(path)) is None

    def test_returns_none_for_a_truncated_file(self, tmp_path):
        path = tmp_path / "truncated.safetensors"
        path.write_bytes(b"\x10\x00")
        assert mc_arch.read_safetensors_header(str(path)) is None

    def test_rejects_an_absurd_header_length(self, tmp_path):
        path = tmp_path / "hostile.safetensors"
        path.write_bytes(struct.pack("<Q", 2**60) + b'{"a":1}')
        assert mc_arch.read_safetensors_header(str(path)) is None

    def test_detection_degrades_to_unknown_rather_than_raising(self, tmp_path):
        path = tmp_path / "garbage.safetensors"
        path.write_bytes(b"\x00" * 64)
        assert mc_arch.detect_from_file(str(path)) is mc_arch.UNKNOWN

    def test_missing_file_is_unknown(self):
        assert mc_arch.detect_from_file("/nonexistent/model.safetensors") is mc_arch.UNKNOWN

    def test_gguf_is_reported_unknown_rather_than_guessed(self, tmp_path):
        path = tmp_path / "model.gguf"
        path.write_bytes(b"GGUF" + b"\0" * 64)
        assert mc_arch.detect_from_file(str(path)) is mc_arch.UNKNOWN

    def test_results_are_cached_per_file_identity(self, tmp_path):
        path = write_safetensors(tmp_path / "cached.safetensors", {"a.weight": ("F16", (2, 2))})
        first = mc_arch.detect_from_file(str(path))
        second = mc_arch.detect_from_file(str(path))
        assert first is second


class TestTensorStub:
    def test_exposes_the_attributes_the_detector_reads(self):
        stub = mc_arch._TensorStub([320, 4, 3, 3], "F16")
        assert stub.shape == (320, 4, 3, 3)
        assert stub.ndim == 4
        assert len(stub) == 320
