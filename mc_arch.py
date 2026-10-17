"""Architecture detection and per-architecture geometry for the Model Chain extension.

Nothing here imports the WebUI at module scope beyond what is safe at script-load
time, and every host interaction is defensive: a failure to detect an
architecture must degrade to "unknown" rather than break a generation.

Detection reuses the host's own ``huggingface_guess`` detector instead of
hand-rolled key heuristics, so the architecture reported in the UI is the same
one the loader will decide on. To keep it cheap enough for a dropdown we read
only the safetensors *header* (tensor names + shapes, no tensor data) and feed
the detector a dict of lightweight stubs.

An extension that teaches the host a model of its own registers a guess class
with that same detector and an engine with the same loader, so it is known here
by those two class names and by nothing else: nothing of it is imported, and no
file of its own is read. MOSS-Image is the first (``moss``); the row says where
the names come from.
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
    # gates this per architecture behind a global Settings toggle. Since Forge
    # Neo renamed Klein's on 2026-09-01 all three opt *in* through a
    # ``*_do_reference`` option; before that Klein opted *out* through
    # ``klein_no_reference``, and :func:`edit_toggle` still answers that name
    # and polarity on a host that registers only it.
    edit_option: str | None = None
    """``shared.opts`` key gating reference conditioning, if the architecture has one.

    The *current* host's name. Read it through :func:`edit_toggle`, which knows
    the names a host used before a rename.
    """
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
    # Klein is reference-conditioned by default: its option is on until the
    # user turns it off. Forge Neo's name for it since 2026-09-01; the opt-out
    # name it had before is in _FORMER_EDIT_OPTIONS.
    Architecture(
        "flux2_4b", "Flux.2 Klein 4B", 16, 1.0,
        edit_option="klein_do_reference", edit_on_value=True, edit_needs_lora=False,
        edit_is_default=True,
        reference_support=REF_VALIDATED, reference_flag="klein",
    ),
    Architecture(
        "flux2_9b", "Flux.2 Klein 9B", 16, 1.0,
        edit_option="klein_do_reference", edit_on_value=True, edit_needs_lora=False,
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
    # MOSS-Image, from the moss-image-forge-neo extension rather than from
    # Forge Neo itself: the FLUX.2 autoencoder (16x, patch size 1) under a
    # single-stream DiT -> 16, and upstream samples the SFT and the Base model
    # alike at CFG 3. The extension registers its guess class and its engine
    # with the host, which is how the two names in the maps below reach this
    # table without anything here importing it; without the extension a MOSS
    # file is unknown here exactly as it is to the loader. No edit toggle and
    # no reference path: a picture goes in as an img2img init latent, and that
    # is all it does with one. Creative Mode, Spatial Layout and the literal
    # prompts are Krea 2's and stand down for it (mc_krea_pipeline.objection).
    Architecture("moss", "MOSS-Image", 16, 3.0),
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
    # The MOSS-Image extension's huggingface_guess class (forge_moss.native.model).
    "MossImage": "moss",
}


_ENGINE_CLASS_TO_KEY = {
    "Flux": "flux",
    "Krea2": "krea2",
    "Anima": "anima",
    "QwenImage": "qwen",
    "Wan": "wan",
    # The MOSS-Image extension's engine (forge_moss.native.engine).
    "MossImageEngine": "moss",
}
"""Forge Neo diffusion-engine class -> our table.

A last resort, for a loaded model that does not carry a recognisable
``model_config``. ``Flux2`` serves both Klein sizes and is resolved separately.
"""


def by_key(key: str) -> Architecture:
    """Look an architecture up by its stable key, falling back to UNKNOWN."""
    return _BY_KEY.get(key, UNKNOWN)


# --------------------------------------------------------------------------- #
# Edit / reference mode
# --------------------------------------------------------------------------- #

_FORMER_EDIT_OPTIONS: dict[str, tuple[str, bool]] = {
    # Forge Neo's "streamline" commit of 2026-09-01 renamed Klein's toggle and
    # turned it round: klein_no_reference (True disabled references) became
    # klein_do_reference (True enables them, and it is on by default).
    "klein_do_reference": ("klein_no_reference", False),
}
"""An architecture's current option -> the name and on-value an older host used."""


def _host_has_option(opts, name: str) -> bool:
    """Whether the host registers ``name`` as a setting.

    The host's ``Options`` keeps its registry in ``data_labels``, and that is
    the authority: ``Options.set`` -- which is what applies a Stage 2 override
    -- looks the key up there and raises ``KeyError`` on one it does not know,
    whatever attribute may have been set on the object meanwhile. A host
    without the registry (the test host) is read by attribute instead.
    """
    labels = getattr(opts, "data_labels", None)
    if isinstance(labels, dict):
        return name in labels
    try:
        getattr(opts, name)
    except AttributeError:
        return False
    return True


def edit_toggle(arch: Architecture) -> tuple[str | None, bool]:
    """The option gating ``arch``'s reference conditioning *on this host*, and its on-value.

    The table carries the current Forge Neo name. When the host does not
    register it but does register the name it used before a rename, that name
    and its polarity are answered instead, so an installation that has not
    updated keeps working; a host with neither gets the table's name, which is
    the one its error will mention. A stale name is not harmless: the host
    applies a Stage 2 override through ``Options.set``, which raises
    ``KeyError`` on a key it does not register, and the pass never starts.
    """
    option, on_value = arch.edit_option, arch.edit_on_value
    former = _FORMER_EDIT_OPTIONS.get(option or "")
    if option is None or former is None:
        return option, on_value
    try:
        from modules import shared

        opts = shared.opts
    except Exception:
        return option, on_value
    if _host_has_option(opts, option) or not _host_has_option(opts, former[0]):
        return option, on_value
    return former


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
    # The toggle as this host names it, with the value that means "on" for
    # that name: the polarity travels with the name, so an opt-out name on an
    # older host needs no special-casing here.
    option, on_value = edit_toggle(arch)
    return {option: on_value if want_on else not on_value}


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

        option, on_value = edit_toggle(arch)
        return bool(getattr(shared.opts, option, None)) == on_value
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
    return _prefix_bare_keys(stub)


_UNET_PREFIXES = ("model.diffusion_model.", "net.")


def _prefix_bare_keys(stub: dict) -> dict:
    """Mirror ``forge_loader``'s own key preparation before it guesses.

    ``preprocess_state_dict`` prefixes a bare state dict with
    ``model.diffusion_model.``, and only then does the host run the detector.
    That step is not cosmetic: ``unet_prefix_from_state_dict`` falls back to
    ``model.`` when it recognises nothing, so a checkpoint whose UNet keys are
    stored bare -- which plenty of single-file Flux and Klein builds are --
    matches none of the keys the detector looks for and comes back as an
    unknown architecture.

    Detecting differently from the loader is the bug: the point of running the
    host's own detector is to reach the host's own answer.
    """
    if any(key.startswith(_UNET_PREFIXES) for key in stub):
        return stub
    return {f"model.diffusion_model.{key}": value for key, value in stub.items()}


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


def detect_loaded_engine() -> Architecture:
    """The architecture of the loaded model, as the *model itself* reports it.

    The definitive answer, and the one to prefer wherever it can be asked.
    ``detect_from_file`` has to guess from a checkpoint header before anything
    is loaded, and that guess can lose: it cannot open GGUF or pickle
    checkpoints at all, and a quantised or repacked safetensors build can rename
    the very keys the detector looks for -- which is how a genuine Flux.2 Klein
    comes back as "unknown".

    Once the model is loaded there is nothing left to guess. Forge hands the
    engine the config class its own detector settled on and the engine keeps it,
    so reading it back is not a second opinion -- it is the loader's own answer.
    """
    try:
        from modules import shared

        model = shared.sd_model
        if model is None:
            return UNKNOWN

        key = _GUESS_CLASS_TO_KEY.get(type(getattr(model, "model_config", None)).__name__)
        if key is not None:
            return _BY_KEY.get(key, UNKNOWN)

        return _from_engine_class(model)
    except Exception:
        return UNKNOWN


def _from_engine_class(model) -> Architecture:
    """Fall back to the engine class when the model config is unrecognisable."""
    name = type(model).__name__

    if name == "Flux2":
        # One engine, two Klein sizes. They are identical in everything this
        # detection decides, so the hidden size only picks the right label.
        config = getattr(model, "model_config", None)
        hidden = (getattr(config, "unet_config", None) or {}).get("hidden_size")
        return by_key("flux2_4b" if hidden == 3072 else "flux2_9b")

    return by_key(_ENGINE_CLASS_TO_KEY.get(name, ""))


def detect_loaded() -> Architecture:
    """Detect the architecture of the currently loaded checkpoint."""
    engine = detect_loaded_engine()
    if engine is not UNKNOWN:
        return engine

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
