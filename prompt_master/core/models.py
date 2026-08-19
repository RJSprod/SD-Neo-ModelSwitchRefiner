from __future__ import annotations

import random
from dataclasses import dataclass

# A seed the user has not chosen. Upstream has no notion of one — it seeds the
# casting and the sampler with whatever integer it is given — so a request
# carrying this must have it resolved to a real number before the engine sees
# it, which is what ``draw_seed`` is for.
RANDOM_SEED = -1


def draw_seed() -> int:
    """A fresh seed, in the range llama.cpp and upstream's casting both accept."""
    return random.randrange(0, 2 ** 31 - 1)


@dataclass(slots=True)
class PromptRequest:
    """One generation request, in the vocabulary the upstream engine speaks.

    Every field below carries an upstream key, not a display label, and the
    defaults are upstream's own node defaults (see ``upstream/node.py``
    ``INPUT_TYPES``). The UI maps labels to these values; the adapter passes
    them through untranslated wherever upstream already accepts the value.
    """

    intent: str
    image_data_url: str | None = None
    image_name: str = ""
    # upstream keys: "i2v" | "t2v"
    video_mode: str = "i2v"
    # "off" | "male" | "female"
    pov: str = "off"
    # a key from upstream accents.ACCENT_KEYS
    accent: str = "off"
    # "natural" | "strong" | "thick" — upstream accents.STRENGTHS
    accent_strength: str = "natural"
    # 0-100 dial; upstream brain.talk_pct also accepts legacy strings
    dialogue: int = 20
    # "auto" | "off" | "her" | "him"
    wardrobe: str = "auto"
    undress: bool = False
    # keys from upstream cinematics.CAMERA_KEYS / TRANSITION_KEYS
    camera: str = "off"
    transition: str = "off"
    # "auto" or a key from upstream music.MUSIC_KEYS
    music: str = "off"
    music_bg: bool = False
    # free text: "Name = description" lines, filtered against the intent
    lexicon: str = ""
    # a key from upstream shotscript.FORMATS
    fmt: str = "flowing"
    fps: int = 24
    seconds: float = 12.0
    # a key from upstream styles.STYLE_KEYS
    style: str = "off"
    # a key from prompt_engine.motion.PRESETS; "default" is upstream unchanged
    motion: str = "default"
    # prompt_engine.speech multiplier: 1 leaves the intent exactly as typed
    speech: int = 1
    # RANDOM_SEED asks for a new one per generation; the UI resolves it
    seed: int = 7
    negative_extra: str = ""
    smart_negative: bool = False
    output_width: int = 704
    output_height: int = 1216


# The device index that means "no card: run the model on the processor and
# system RAM". Every index nvidia-smi reports is zero or greater, so a negative
# one cannot collide with a real GPU.
CPU_INDEX = -1

# What a chosen device does with the model. Recorded in the setup state, so an
# installed app can say which of the three it is running.
GPU_MODE = "gpu"        # the weights are in VRAM and the card does everything
MIXED_MODE = "mixed"    # the weights are in system RAM; the card is given the
                        # work llama.cpp can hand it, and holds little else
CPU_MODE = "cpu"        # no card is involved at all


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """One device the model can be installed for, and what it will do with it.

    Setup describes hardware with a single type so the processor travels the
    same path a card does — the same manifest lookup, the same download, the
    same state file. Three choices are expressible: a card in ``GPU_MODE``, the
    same card in ``MIXED_MODE``, and the processor. For the processor
    ``memory_total_mb``/``memory_free_mb`` are system RAM rather than VRAM; for
    a card they are always its VRAM, in either mode.
    """

    physical_index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_free_mb: int
    driver_version: str
    # nvidia-smi --query-gpu=compute_cap, e.g. 8.6 for Ampere, 12.0 for
    # Blackwell. None when the installed driver is too old to report it; the
    # runtime choice then falls back to the model number. Always None for the
    # processor, which has no CUDA compute capability at all.
    compute_capability: float | None = None
    # Set when this card was chosen for mixed mode. It is a choice about the
    # same hardware rather than different hardware, which is why it lives here
    # beside it: a card is offered twice, once each way. Never set for the
    # processor, which has no card to hand anything to.
    mixed: bool = False

    @property
    def is_cpu(self) -> bool:
        """True for the processor rather than a CUDA card."""
        return self.physical_index == CPU_INDEX

    @property
    def is_mixed(self) -> bool:
        """True for a card that will leave the weights in system RAM."""
        return self.mixed and not self.is_cpu

    @property
    def weights_in_system_ram(self) -> bool:
        """True when the model is loaded into system RAM rather than VRAM.

        The axis mixed mode and CPU mode share, and what every "how much memory
        does this need" decision turns on: neither is sized against VRAM, so
        neither has a VRAM shortfall to report, and both take longer to load
        than filling a card does.
        """
        return self.is_cpu or self.is_mixed

    @property
    def mode(self) -> str:
        """``GPU_MODE``, ``MIXED_MODE`` or ``CPU_MODE``."""
        if self.is_cpu:
            return CPU_MODE
        return MIXED_MODE if self.mixed else GPU_MODE

    @property
    def supported(self) -> bool:
        """Every CUDA GPU nvidia-smi reports is provisionable, as is the CPU.

        The upstream build accepted only an RTX 3090 or 5090 and refused
        everything else outright. Those two cards keep their pinned runtime and
        quantization (see ``device_detection.PINNED``); any other NVIDIA card is
        now sized from its own VRAM and compute capability instead of being
        rejected, and a machine with no NVIDIA card at all can install the
        CPU runtime instead. A card too small for a full offload is warned about
        at setup, not blocked — and mixed mode is the answer to that warning
        rather than a smaller quantization.
        """
        return True
