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

EDIT_AUTO = "Auto"
EDIT_ENABLE = "Enable"
EDIT_DISABLE = "Disable"
EDIT_MODES = (EDIT_AUTO, EDIT_ENABLE, EDIT_DISABLE)

REF_VALIDATED = "validated"
"""Supplemental Stage 2 references are tested on this architecture."""

REF_EXPERIMENTAL = "experimental"
"""Forge exposes a reference path here, but Model Chain has not validated it."""


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

    # -- edit / reference conditioning ------------------------------------ #
    #
    # Some architectures can take the input image as an *edit reference* --
    # vision-conditioning the text encoder and concatenating reference latents
    # in the DiT -- instead of using it as an img2img init latent. Forge Neo
    # gates this per architecture behind a global Settings toggle, and the
    # polarity is not consistent: Krea 2 and Anima opt *in*, while Flux.2 Klein
    # uses references by default and opts *out*.
    edit_option: str | None = None
    """``shared.opts`` key gating reference conditioning, if the architecture has one."""
    edit_on_value: bool = True
    """Value of ``edit_option`` that turns reference conditioning ON."""
    edit_needs_lora: bool = False
    """Whether the architecture needs a separate Edit LoRA for this to work."""
    edit_denoise: float = 1.0
    """Denoise strength edit conditioning expects; the reference carries the content."""
    edit_is_default: bool = False
    """True when reference conditioning is simply how the architecture works.

    Klein always sees the input image as a reference *in addition to* using it
    as an img2img init latent, so an ordinary low-denoise refine is perfectly
    valid there. Krea 2 and Anima are the opposite: edit mode is a deliberate
    mode switch away from img2img. Only the deliberate case earns a warning
    about denoise strength.
    """

    # -- supplemental reference images (section 12) ----------------------- #
    #
    # A reference-capable engine overrides ``encode_first_stage`` to divert
    # anything encoded under ``dynamic_args.is_referencing`` into its own
    # ``ref_latents``. That is what Model Chain fills for Stage 2, and it is a
    # property of the engine rather than of the checkpoint's name.
    reference_support: str | None = None
    """``REF_VALIDATED``, ``REF_EXPERIMENTAL``, or None for no reference path."""
    reference_flag: str | None = None
    """``dynamic_args`` field confirming the *loaded* checkpoint has that path.

    For Flux.1 and Qwen-Image the architecture alone does not settle it: only
    the Kontext and Edit variants reference, and the loader decides which is
    which by reading the state dict. The flag is how that decision is read back,
    which is why detection here does not stop at the checkpoint name.
    """

    @property
    def supports_edit(self) -> bool:
        return self.edit_option is not None

    @property
    def supports_references(self) -> bool:
        """Whether this architecture has a reference path at all."""
        return self.reference_support is not None

    @property
    def references_are_validated(self) -> bool:
        return self.reference_support == REF_VALIDATED

    def edit_is_deliberate(self, mode: str) -> bool:
        """Whether ``mode`` represents the user actively choosing edit semantics."""
        if not self.supports_edit:
            return False
        if mode == EDIT_ENABLE:
            return True
        return not self.edit_is_default


_ARCHITECTURES: tuple[Architecture, ...] = (
    Architecture("sd15", "SD 1.5", 8, 7.0),
    Architecture("sdxl", "SDXL", 8, 7.0),
    Architecture("sdxl_refiner", "SDXL Refiner", 8, 7.0),
    Architecture("mugen", "Mugen", 8, 7.0),
    # Only the Kontext variant references, hence the runtime flag.
    Architecture(
        "flux", "Flux.1", 16, 1.0,
        reference_support=REF_EXPERIMENTAL, reference_flag="kontext",
    ),
    Architecture("flux_schnell", "Flux.1 Schnell", 16, 1.0),
    # Klein is reference-conditioned by default; the option disables it.
    Architecture(
        "flux2_4b", "Flux.2 Klein 4B", 16, 1.0,
        edit_option="klein_no_reference", edit_on_value=False, edit_needs_lora=False,
        edit_is_default=True,
        reference_support=REF_VALIDATED, reference_flag="klein",
    ),
    Architecture(
        "flux2_9b", "Flux.2 Klein 9B", 16, 1.0,
        edit_option="klein_no_reference", edit_on_value=False, edit_needs_lora=False,
        edit_is_default=True,
        reference_support=REF_VALIDATED, reference_flag="klein",
    ),
    Architecture("chroma", "Chroma", 16, 1.0),
    Architecture("lumina2", "Lumina 2", 16, 4.0),
    Architecture("zimage", "Z-Image", 16, 1.0),
    Architecture(
        "anima", "Anima", 16, 1.0,
        edit_option="anima_do_reference", edit_on_value=True, edit_needs_lora=True,
        reference_support=REF_EXPERIMENTAL, reference_flag="anima",
    ),
    # Wan is left out on purpose: ImageStitch feeds it a *last frame* for
    # video, not a reference set, and it discards everything past the first
    # image. Routing Stage 2 references there would mean something else
    # entirely.
    Architecture("wan", "Wan", 16, 1.0),
    # Only the Edit variant references, hence the runtime flag.
    Architecture(
        "qwen", "Qwen-Image", 16, 1.0,
        reference_support=REF_EXPERIMENTAL, reference_flag="edit",
    ),
    # Krea 2: VAE downscales 8x and the transformer patchifies 2x2 -> 16.
    Architecture(
        "krea2", "Krea 2", 16, 1.0,
        edit_option="krea2_do_reference", edit_on_value=True, edit_needs_lora=True,
        reference_support=REF_VALIDATED, reference_flag="krea2",
    ),
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
# Edit / reference mode
# --------------------------------------------------------------------------- #

def edit_override(arch: Architecture, mode: str) -> dict:
    """Settings override that forces reference conditioning on or off for Stage 2.

    Returned as a ``p.override_settings`` fragment so the host applies it for
    the Stage 2 generation and restores the user's value afterwards -- the
    underlying toggle is global, and silently leaving it flipped would change
    the behaviour of every later generation.

    ``Auto`` returns an empty dict, leaving the user's global setting alone.
    """
    if mode == EDIT_AUTO or not arch.supports_edit:
        return {}

    want_on = mode == EDIT_ENABLE
    # edit_on_value encodes the polarity, so this handles both the opt-in
    # (krea2/anima) and opt-out (klein) options without special-casing.
    return {arch.edit_option: arch.edit_on_value if want_on else not arch.edit_on_value}


def edit_is_active(arch: Architecture, mode: str) -> bool:
    """Whether reference conditioning will be on for Stage 2 under ``mode``.

    In ``Auto`` this reads the user's current global setting, so warnings and
    defaults match what will actually happen.
    """
    if not arch.supports_edit:
        return False
    if mode == EDIT_ENABLE:
        return True
    if mode == EDIT_DISABLE:
        return False

    try:
        from modules import shared

        return bool(getattr(shared.opts, arch.edit_option, None)) == arch.edit_on_value
    except Exception:
        return False


def references_available(arch: Architecture) -> bool:
    """Whether the *loaded* checkpoint exposes Forge's reference path.

    Checked once Stage 2's model is resident, because for Flux.1 and Qwen-Image
    the architecture is not the whole answer: a plain Flux.1 dev and Flux.1
    Kontext detect identically from their tensor shapes, and only the loader --
    which reads the state dict -- knows which one it just loaded. The flag it
    sets is that answer, and Model Chain's cache carries it across warm swaps
    along with the rest of the model flags.

    This is the pre-flight check. The encode itself is verified afterwards by
    counting what the model actually took, which is the signal nothing can fake.
    """
    if not arch.supports_references:
        return False
    if arch.reference_flag is None:
        return True

    try:
        from backend.args import dynamic_args

        return bool(getattr(dynamic_args, arch.reference_flag, False))
    except Exception:
        return False


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


def resolution_step(default: int = 64) -> int:
    """The host's resolution snapping step (``Settings -> Resolution Step``)."""
    try:
        from modules import shared

        return max(int(shared.opts.res_step), 1)
    except Exception:
        return default


def hires_target_size(
    width: int,
    height: int,
    scale: float = 2.0,
    resize_x: int = 0,
    resize_y: int = 0,
    step: int | None = None,
) -> tuple[int, int]:
    """Size hires fix will upscale a first pass to.

    Mirrors ``StableDiffusionProcessingTxt2Img.calculate_target_resolution``:
    ``hr_resize_x``/``hr_resize_y`` of 0 mean "use the scale factor", a single
    non-zero value drives the other axis by aspect ratio, and both results are
    snapped with the host's rounding.

    This matters because ``p.width``/``p.height`` keep describing the *first
    pass* when hires fix is on -- the image Stage 2 actually receives is this
    size.
    """
    step = resolution_step() if step is None else step

    if not resize_x and not resize_y:
        return snap_dimension(width * scale, step), snap_dimension(height * scale, step)

    if not resize_y:
        return snap_dimension(resize_x, step), snap_dimension(resize_x * (height / width), step)

    if not resize_x:
        return snap_dimension(resize_y * (width / height), step), snap_dimension(resize_y, step)

    return snap_dimension(resize_x, step), snap_dimension(resize_y, step)


def stage1_size(p) -> tuple[int, int]:
    """Pixel size of the image Stage 2 will receive.

    With hires fix enabled that is the upscaled size, not ``p.width``/
    ``p.height`` -- those still describe the first pass.
    """
    width, height = int(getattr(p, "width", 0) or 0), int(getattr(p, "height", 0) or 0)

    if not getattr(p, "enable_hr", False):
        return width, height

    # The host computes these in init(); prefer them when present, since they
    # already account for the old-hires-fix compatibility option.
    target_x = getattr(p, "hr_upscale_to_x", 0) or 0
    target_y = getattr(p, "hr_upscale_to_y", 0) or 0
    if target_x and target_y:
        return int(target_x), int(target_y)

    return hires_target_size(
        width,
        height,
        scale=float(getattr(p, "hr_scale", 2.0) or 2.0),
        resize_x=int(getattr(p, "hr_resize_x", 0) or 0),
        resize_y=int(getattr(p, "hr_resize_y", 0) or 0),
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
