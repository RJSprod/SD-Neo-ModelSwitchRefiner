"""Architecture detection and per-architecture geometry for the Model Chain extension.

Nothing here imports the WebUI at module scope beyond what is safe at script-load
time, and every host interaction is defensive: a failure to detect an
architecture must degrade to "unknown" rather than break a generation.

Detection reuses the host's own ``huggingface_guess`` detector instead of
hand-rolled key heuristics, so the architecture reported in the UI is the same
one the loader will decide on. To keep it cheap enough for a dropdown we read
only the safetensors *header* (tensor names + shapes, no tensor data) and feed
the detector a dict of lightweight stubs.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Architecture table
# --------------------------------------------------------------------------- #
#
# ``alignment`` is the pixel multiple that both output dimensions must land on
# for the architecture. It is the VAE spatial compression ratio multiplied by
# any patchify factor the transformer applies on top of the latent:
#
#   SD 1.5 / SDXL   VAE 8x,  no patchify        ->  8
#   Flux.1 family   VAE 8x,  2x2 patchify       -> 16
#   Flux.2 family   VAE 16x, patch_size 1       -> 16
#   Wan / Qwen      VAE 8x,  2x2 patchify       -> 16
#
# ``cfg`` is the architecture-appropriate default for the Stage 2 CFG control
# (section 6.4): distilled and flow-matching models want CFG at or near 1.0,
# and defaulting those to an SDXL-typical 7.0 produces burnt output.

_UNKNOWN_ALIGNMENT = 16
"""Fallback alignment. 16 is a multiple of 8, so it is valid for every
architecture in the table; guessing 8 for an unknown model would not be."""

_UNKNOWN_CFG = 7.0


@dataclass(frozen=True)
class Architecture:
    key: str
    """Stable identifier used in comparisons and tests."""
    label: str
    """Human-readable name shown in the UI."""
    alignment: int
    """Required pixel multiple for width and height."""
    cfg: float
    """Sensible default CFG scale for this architecture."""


_ARCHITECTURES: tuple[Architecture, ...] = (
    Architecture("sd15", "SD 1.5", 8, 7.0),
    Architecture("sdxl", "SDXL", 8, 7.0),
    Architecture("sdxl_refiner", "SDXL Refiner", 8, 7.0),
    Architecture("mugen", "Mugen", 8, 7.0),
    Architecture("flux", "Flux.1", 16, 1.0),
    Architecture("flux_schnell", "Flux.1 Schnell", 16, 1.0),
    Architecture("flux2_4b", "Flux.2 Klein 4B", 16, 1.0),
    Architecture("flux2_9b", "Flux.2 Klein 9B", 16, 1.0),
    Architecture("chroma", "Chroma", 16, 1.0),
    Architecture("lumina2", "Lumina 2", 16, 4.0),
    Architecture("zimage", "Z-Image", 16, 1.0),
    Architecture("anima", "Anima", 16, 1.0),
    Architecture("wan", "Wan", 16, 1.0),
    Architecture("qwen", "Qwen-Image", 16, 1.0),
    Architecture("krea2", "Krea 2", 16, 1.0),
    Architecture("ernie", "ERNIE-Image", 16, 1.0),
    Architecture("pid", "PiD", 8, 7.0),
)

UNKNOWN = Architecture("unknown", "unknown", _UNKNOWN_ALIGNMENT, _UNKNOWN_CFG)

_BY_KEY = {a.key: a for a in _ARCHITECTURES}

# Maps the class name reported by huggingface_guess onto our table.
_GUESS_CLASS_TO_KEY = {
    "SD15": "sd15",
    "SDXL": "sdxl",
    "SDXLRefiner": "sdxl_refiner",
    "Mugen": "mugen",
    "Flux": "flux",
    "FluxSchnell": "flux_schnell",
    "Flux2K4B": "flux2_4b",
    "Flux2K9B": "flux2_9b",
    "Chroma": "chroma",
    "Lumina2": "lumina2",
    "ZImage": "zimage",
    "Anima": "anima",
    "WAN21_T2V": "wan",
    "WAN21_I2V": "wan",
    "QwenImage": "qwen",
    "Krea2": "krea2",
    "ErnieImage": "ernie",
    "PiD": "pid",
}


def by_key(key: str) -> Architecture:
    """Look an architecture up by its stable key, falling back to UNKNOWN."""
    return _BY_KEY.get(key, UNKNOWN)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def snap_dimension(value: float, alignment: int) -> int:
    """Round ``value`` to the nearest multiple of ``alignment`` (never below it).

    Ties round up, matching the host's own ``modules.ui.sRound``. Python's
    built-in ``round`` uses banker's rounding, which would send an exact
    half-step such as 1000/16 downwards and disagree with the resolution the
    main width/height sliders would have produced.
    """
    alignment = max(int(alignment), 1)
    snapped = math.floor(float(value) / alignment + 0.5) * alignment
    return max(snapped, alignment)


def scaled_size(width: int, height: int, multiplier: float, alignment: int) -> tuple[int, int]:
    """Apply the Stage 2 size multiplier while preserving aspect ratio.

    Both dimensions are scaled by the same factor -- which is what preserves the
    ratio -- and each is then snapped to the architecture's required grid. The
    snap can shift the ratio by at most half an alignment step per axis, which
    is the smallest deviation possible for a model that cannot accept
    off-grid dimensions.
    """
    return (
        snap_dimension(width * multiplier, alignment),
        snap_dimension(height * multiplier, alignment),
    )


def aspect_ratio_delta(src: tuple[int, int], dst: tuple[int, int]) -> float:
    """Relative aspect-ratio error introduced by snapping, as a fraction."""
    src_ratio = src[0] / src[1]
    dst_ratio = dst[0] / dst[1]
    return abs(dst_ratio - src_ratio) / src_ratio


# --------------------------------------------------------------------------- #
# Header-only safetensors reading
# --------------------------------------------------------------------------- #


class _TensorStub:
    """Stands in for a torch tensor during detection.

    ``huggingface_guess.detection`` only ever reads ``.shape`` (and occasionally
    checks ``len(shape)``) off the state dict values, so a stub is enough to run
    the real detector without touching tensor data.
    """

    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0] if self.shape else 0


_MAX_HEADER_BYTES = 512 * 1024 * 1024
"""Refuse absurd header lengths rather than allocating on a corrupt file."""


def read_safetensors_header(path: str) -> dict | None:
    """Read a safetensors header without loading any tensor data.

    Returns the raw header dict (tensor name -> {dtype, shape, data_offsets}),
    or None if the file is not a readable safetensors file.
    """
    try:
        with open(path, "rb") as file:
            length_bytes = file.read(8)
            if len(length_bytes) < 8:
                return None
            header_len = int.from_bytes(length_bytes, "little")
            if not 2 < header_len < _MAX_HEADER_BYTES:
                return None
            header = json.loads(file.read(header_len))
    except Exception:
        return None

    if not isinstance(header, dict):
        return None
    return header


def _state_dict_stub(header: dict) -> dict[str, _TensorStub]:
    stub = {}
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape")
        if shape is None:
            continue
        stub[key] = _TensorStub(shape, info.get("dtype"))
    return stub


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

_cache: dict[tuple, Architecture] = {}


def _cache_key(path: str) -> tuple | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), stat.st_size, int(stat.st_mtime))


def _detect_from_stub(stub: dict[str, _TensorStub]) -> Architecture:
    """Run the host's detector over a stubbed state dict."""
    try:
        from huggingface_guess import detection
    except Exception:
        return UNKNOWN

    try:
        prefix = detection.unet_prefix_from_state_dict(stub)
        config = detection.model_config_from_unet(stub, prefix)
    except Exception:
        return UNKNOWN

    if config is None:
        return UNKNOWN

    key = _GUESS_CLASS_TO_KEY.get(type(config).__name__)
    if key is None:
        return UNKNOWN
    return _BY_KEY.get(key, UNKNOWN)


def detect_from_file(path: str) -> Architecture:
    """Detect the architecture of a checkpoint file on disk.

    Only safetensors files can be inspected cheaply; GGUF and pickle
    checkpoints report UNKNOWN, which the UI renders as "architecture
    unknown" rather than guessing wrong.
    """
    if not path:
        return UNKNOWN

    key = _cache_key(path)
    if key is not None and key in _cache:
        return _cache[key]

    result = UNKNOWN
    if os.path.splitext(path)[1].lower() in (".safetensors", ".sft"):
        header = read_safetensors_header(path)
        if header:
            result = _detect_from_stub(_state_dict_stub(header))

    if key is not None:
        _cache[key] = result
    return result


def detect_from_checkpoint_name(name: str) -> Architecture:
    """Detect the architecture for a checkpoint named in the WebUI's list."""
    if not name or name in ("None", "none", ""):
        return UNKNOWN
    try:
        from modules import sd_models

        info = sd_models.get_closet_checkpoint_match(name)
    except Exception:
        return UNKNOWN

    if info is None:
        return UNKNOWN
    return detect_from_file(info.filename)


def detect_loaded() -> Architecture:
    """Detect the architecture of the currently loaded checkpoint."""
    try:
        from modules import shared

        info = getattr(shared.sd_model, "sd_checkpoint_info", None)
        if info is None:
            return UNKNOWN
        return detect_from_file(info.filename)
    except Exception:
        return UNKNOWN


def describe(name: str) -> str:
    """Render ``name`` with its architecture appended, for dropdown labels."""
    arch = detect_from_checkpoint_name(name)
    if arch is UNKNOWN:
        return name
    return f"{name}  —  {arch.label}"
