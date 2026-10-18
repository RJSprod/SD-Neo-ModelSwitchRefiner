"""What a pass really allocates, measured, beside what the host asked for.

Forge's sampler asks ``load_models_gpu`` for a fixed allowance of free VRAM
beside the weights before every pass -- so many bytes per latent cell, the
batch counted twice, a per-architecture factor -- and frees other models to get
it (mc_memory, "What the host's sampler is about to ask for"). The allowance
is a rule, not a reading, and on the user's card it asked for 2.8 GB an image
where the pass allocated about 0.75. The user's question was whether the host
is making a bad bet with every image added to the batch. This answers it with
the only thing that can: the pass's own peak, in the log beside the request,
every pass.

The window is the sampling itself. It opens at the first sampling step -- the
denoiser callback, which runs after ``sampling_prepare`` has made its room
and loaded what it loads -- by resetting PyTorch's peak counter and reading
what is allocated then: the weights on the card, the latents, the conditioning.
It closes when the host's sampler returns (``post_sample``), or when the next
pass is armed (the hires pass follows the first without a ``post_sample``
between them), reading the peak since. Peak less baseline is the working VRAM
the pass needed beyond what was already there: what the host's allowance is
meant to cover.

PyTorch's counter sees its own allocator and nothing else -- not cuBLAS
workspaces, not the CUDA context, not fragmentation -- so the figure is a
floor on what the driver had to find, and it is said as such.

Said in GB for a reader, and with the increment: when the same size has been
measured at another batch size this session, the line says what each extra
image cost against what the host charged for it.
"""

from __future__ import annotations

import dataclasses
import logging

logger = logging.getLogger("model_chain")

_GB = 1024**3


@dataclasses.dataclass
class Pass:
    stage: str
    width: int
    height: int
    batch: int
    latent: str
    asked: int
    """What the host's sampler asked ``free_memory`` for, in bytes; 0 unknown."""
    asked_activations: int
    """The activations term of that request, in bytes; 0 unknown."""
    counted_once: bool = False
    baseline: int = 0
    """Bytes allocated at the first step, before the pass's own allocations."""
    measured: int = -1
    """Peak working VRAM beyond the baseline; -1 until closed."""

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}" if self.width and self.height else self.latent


_armed: Pass | None = None
_open: Pass | None = None
_seen: dict = {}
"""``(stage, width, height) -> {batch: measured bytes}`` this session."""


def arm(demand, width: int, height: int, batch: int, stage: str, latent: str = "") -> Pass:
    """A pass is about to run: remember what the host asked, wait for its first step.

    A pass still open when the next is armed is closed first -- the hires pass
    follows the first without a ``post_sample`` between them.
    """
    global _armed
    close()
    _armed = Pass(
        stage=stage, width=int(width or 0), height=int(height or 0),
        batch=max(int(batch or 1), 1),
        latent=latent or (getattr(demand, "latent", "") or ""),
        asked=int(getattr(demand, "request", 0) or 0),
        asked_activations=int(getattr(demand, "activations", 0) or 0),
        counted_once=bool(getattr(demand, "counted_once", False)),
    )
    return _armed


def first_step() -> bool:
    """The first sampling step of an armed pass: the window opens here."""
    global _armed, _open
    if _armed is None:
        return False
    found, _armed = _armed, None
    try:
        import torch

        from backend import memory_management

        device = memory_management.get_torch_device()
        torch.cuda.reset_peak_memory_stats(device)
        found.baseline = int(torch.cuda.memory_allocated(device))
    except Exception:
        logger.debug("Model Chain: could not open the pass's VRAM window", exc_info=True)
        return False
    _open = found
    return True


def close() -> Pass | None:
    """The pass has ended: read its peak, say it, and remember it for the increments."""
    global _armed, _open
    _armed = None
    if _open is None:
        return None
    found, _open = _open, None
    try:
        import torch

        from backend import memory_management

        device = memory_management.get_torch_device()
        peak = int(torch.cuda.max_memory_allocated(device))
    except Exception:
        logger.debug("Model Chain: could not read the pass's VRAM peak", exc_info=True)
        return None
    found.measured = max(peak - found.baseline, 0)
    key = (found.stage, found.width, found.height)
    others = dict(_seen.get(key, {}))
    _seen.setdefault(key, {})[found.batch] = found.measured
    logger.info("Model Chain: %s", describe(found, others))
    return found


def describe(found: Pass, others: dict | None = None) -> str:
    """One line: the measured peak, the host's request, and the increment."""
    per_image = found.measured / found.batch
    line = (f"the {found.stage} pass used {found.measured / _GB:.1f} GB of working VRAM at its "
            f"peak — a batch of {found.batch} at {found.size} ({per_image / _GB:.2f} GB per image)")
    if found.asked > 0:
        charged = found.asked_activations / found.batch if found.asked_activations else 0
        line += (f"; Forge's sampler had asked for {found.asked / _GB:.1f} GB free beside the "
                 f"weights")
        if found.asked_activations > 0:
            line += (f", {found.asked_activations / _GB:.1f} GB of it for activations "
                     f"({charged / _GB:.2f} GB per image"
                     f"{', each counted once' if found.counted_once else ''})")
        if found.measured > 0 and found.asked_activations > 0:
            line += f" — {found.asked_activations / found.measured:.1f}x what the pass used"
    other = _nearest(others or {}, found.batch)
    if other is not None:
        batch, measured = other
        extra = found.batch - batch
        used = (found.measured - measured) / extra
        line += (f". Against the batch of {batch} at this size: {used / _GB:+.2f} GB used per "
                 f"extra image")
        if found.asked_activations > 0:
            line += f" against {found.asked_activations / found.batch / _GB:.2f} GB charged for each"
    return line + ". PyTorch's own allocations only, so a floor on what the driver had to find"


def _nearest(others: dict, batch: int):
    """The measured batch size nearest below this one at the same size, else nearest above."""
    below = [b for b in others if b < batch]
    above = [b for b in others if b > batch]
    if below:
        chosen = max(below)
    elif above:
        chosen = min(above)
    else:
        return None
    return chosen, others[chosen]


def forget() -> None:
    """Session state cleared, for tests."""
    global _armed, _open
    _armed = _open = None
    _seen.clear()
