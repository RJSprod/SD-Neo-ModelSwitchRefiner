"""Model residency and cache management for the Model Chain extension.

Design principle (section 4.1): VRAM-FIRST, DEMOTE ONLY UNDER PRESSURE.
The residency cascade is VRAM -> system RAM -> disk, and it is demand driven
rather than a fixed evict-on-switch rule.


How this maps onto Forge Neo
----------------------------
Forge Neo keeps exactly one checkpoint in ``sd_models.model_data.sd_model``, and
``forge_model_reload()`` unconditionally calls ``memory_management.unload_all_models()``
and drops its reference before loading a new checkpoint from disk. Section 4.4
forbids reimplementing offload logic or moving raw state dicts between devices,
so we do not fight that path -- we cooperate with it:

* ``unload_all_models()`` calls ``ModelPatcher.detach()``, which moves weights to
  the patcher's *offload device* (system RAM). It does not free them. The only
  reason a checkpoint normally disappears from RAM is that ``forge_model_reload()``
  drops the last Python reference to it and the garbage collector reclaims it.

* So keeping a checkpoint resident in system RAM is exactly a matter of holding
  a reference to its ``sd_model`` across the switch. That is what this cache is.

* Restoring a cached checkpoint is then a pointer swap: put the object back into
  ``model_data``, set ``forge_hash`` to match ``forge_loading_parameters``, and the
  next ``forge_model_reload()`` returns early instead of touching the disk. No
  ``unload_all_models()`` runs on that path, so the *incoming* model is brought
  back to VRAM lazily by ``memory_management.load_models_gpu()``, which evicts the
  outgoing one only if the VRAM is actually needed. That is the demand-driven
  cascade section 4.2 asks for, implemented entirely through host entry points.

One consequence is worth stating plainly: the very first load of a checkpoint has
to go through ``forge_model_reload()``, and that host function unloads everything
before it loads. A cold first switch therefore always demotes the outgoing model
to RAM even when both models would have fit in VRAM together. Every subsequent
switch between two cached checkpoints is a warm pointer swap with demand-driven
VRAM eviction, which is where the "both models stay hot in VRAM" behaviour and
the measurable speedup over a disk reload come from.


Forward compatibility (section 4.3)
-----------------------------------
The two-slot limit lives behind ``ensure_resident()`` / ``get_model()`` and is a
private detail of this module. ``model_chain.py`` never assumes a slot count, so
swapping ``_Cache`` for an LRU pool of 3+ models needs no change to the
orchestration code -- only ``_MAX_SLOTS`` and ``_Cache.admit()``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

import mc_lora


LOG_TIME_FORMAT = "%H:%M:%S"
"""Wall clock, to the millisecond -- see :class:`_Timestamped`."""


class _Timestamped(logging.Filter):
    """Stamps the time of day into every line this extension writes.

    The host's console handler prints the message, the module and the level and
    no clock at all, which is fine for a line that says what happened and
    useless for a line that says how long something took. Half of what this
    extension logs is the second kind: a plan and the budget it implies, a
    placement and the reason for it, a phase peak against the reserve that was
    meant to cover it. Reading those in order tells you what the extension
    decided; reading them against a clock tells you what it cost, and lets a
    console log be lined up with llama-server's own log, with a profiler, or
    with what Task Manager was showing at the time.

    Done here rather than at the two hundred and eighty-odd call sites, and
    done on the *formatted* message rather than the format string, because one
    of those call sites passes its whole sentence through ``%s`` -- so a filter
    that only rewrote ``record.msg`` would leave exactly one line unstamped,
    which is the sort of gap somebody eventually spends an afternoon on.

    Formatting the message here costs the laziness ``logging`` normally offers,
    and costs it only on records that are actually being emitted: a filter runs
    after the level check, so a suppressed ``debug`` call still never builds its
    string.

    The prefix is kept inside the extension's own name rather than in front of
    it -- ``Model Chain [10:14:07.912]: ...`` -- so that a log somebody greps
    for ``Model Chain:`` still finds every line it used to.
    """

    PREFIX = "Model Chain"

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            when = time.strftime(LOG_TIME_FORMAT, time.localtime(record.created))
            when = f"{when}.{int(record.msecs):03d}"
            message = record.getMessage()
            if message.startswith(f"{self.PREFIX}: "):
                message = (f"{self.PREFIX} [{when}]: "
                           f"{message[len(self.PREFIX) + 2:]}")
            else:
                # A line from somewhere that does not use the house prefix. It
                # still gets a clock, because a log with a hole in it is worse
                # than one with an odd-looking line in it.
                message = f"[{when}] {message}"
            record.msg = message
            record.args = ()
        except Exception:
            # A logger that raises while logging takes the caller with it, and
            # the caller is usually in the middle of a generation. An unstamped
            # line is a complete answer to that.
            pass
        return True


def _make_logger() -> logging.Logger:
    """Console logger using the host's Rich formatting, with a clock on it.

    ``logging.getLogger`` alone yields a logger with no handler, so every
    diagnostic this extension emits would be swallowed -- including the cache
    decisions that explain why a switch was slow. ``setup_logger`` attaches the
    same handler the host's own modules use.

    The filter goes on the *logger*, not on a handler, which is what makes one
    call here cover every module: they all reach this same object through
    ``logging.getLogger("model_chain")``, which is why each of them carries the
    note that the handler is attached once, here.
    """
    log = logging.getLogger("model_chain")
    try:
        from backend.logging import setup_logger

        setup_logger(log)
    except Exception:
        log.setLevel(logging.INFO)
    if not any(isinstance(existing, _Timestamped) for existing in log.filters):
        log.addFilter(_Timestamped())
    return log


logger = _make_logger()

_model_lock = threading.RLock()
"""Serialises this module's own model movement.

Everything here normally runs on the generation thread, one call at a time. The
background Stage 1 preload is the exception, so every entry point that moves or
reassigns weights takes this lock and the preload holds it for its whole run.
Re-entrant because the preload calls several of those entry points itself.
"""

OPT_PIN_ENCODERS = "model_chain_pin_stage1_encoders"
OPT_PRELOAD = "model_chain_preload_stage1"
OPT_PRESERVE_LORA = "model_chain_preserve_lora"
OPT_VRAM_RESERVE = "model_chain_vram_reserve_gb"
OPT_WARM_STAGE_2 = "model_chain_warm_stage_2"

PRELOAD_DEFAULT = False
"""Off by default: the preload is the one mechanism here that leaves the
generation thread, and moving weights off it has proved able to break a
generation outright rather than merely slow one down.

It is also the least valuable of the three. Pinning and the VRAM sizing remove
work; the preload only moves the same work earlier, so the cost of having it
wrong is out of all proportion to the benefit of having it right. Opt in per
machine, once it is known to be safe there.

What changed since that judgement was made is the *failure* behaviour, not the
success behaviour: a preload that goes wrong now falls back to the host's
synchronous load path, and two consecutive failures take the feature out for the
rest of the session rather than letting it fail on every generation. The default
stays off regardless -- a machine where it misbehaves still loses more than a
machine where it is off gains."""


def option(name: str, default):
    """Read one of this extension's settings, falling back before the UI exists.

    Settings registered through ``on_ui_settings`` are absent from ``opts``
    during early imports and in tests, and a missing setting must not change
    behaviour -- so the default is the documented behaviour, not a disabled one.
    """
    try:
        from modules import shared

        value = getattr(shared.opts, name, None)
    except Exception:
        return default
    return default if value is None else value

_MAX_SLOTS = 2
"""v1 holds Model A and Model B. See the forward-compatibility note above."""

_GB = 1024**3

VRAM_HEADROOM_BYTES = 1 * _GB
"""Baseline spare VRAM kept free for activations when judging whether two models fit."""

VRAM_HEADROOM_PER_MEGAPIXEL = 0.75 * _GB
"""Additional activation allowance per megapixel of output.

A flat allowance badly under-estimates a large pass: attention activations grow
with pixel count, so a 2048x2048 refine needs several times what 1024x1024
does. Getting this wrong is not a graceful failure -- on Windows the driver
silently spills the overflow to system RAM over PCIe, and sampling drops from
sub-second to tens of seconds per step with no error to explain it.
"""

VRAM_MODEL_OVERHEAD_FRACTION = 0.15
"""How much larger a model is in VRAM than its file is on disk.

``file_size_bytes`` measures the checkpoint on disk, but that is a floor, not
the resident footprint: weights stored below the compute dtype are widened on
load, and LoRA patches and the parameter bookkeeping the patcher keeps alongside
them are not in the file at all.

Under-estimating here is what produces the ``Unloaded partially`` lines in the
log immediately before a UNet load. ``free_memory`` hits our target and stops,
the target turns out to be short, and the host then makes up the difference the
expensive way -- unloading in chunks *while* it loads, which is the slow path
this module exists to avoid. Two measured chains were short by 11% and 14% of
model size, so the correction is proportional to the model rather than flat.

Over-estimating is much cheaper than under-estimating: it evicts a model that
might have fitted, and that model is still warm in the RAM cache.
"""


PEAK_MARGIN = 1.15
"""Safety factor on an observed activation peak before it is trusted as a reserve.

A peak is the largest thing that happened, not the largest thing that can
happen: a different sampler, a ControlNet unit or one more image in the batch
all push past it. The margin is the difference between "what we saw" and "what
we are prepared to be surprised by".
"""

MAX_LEARNED_BYTES_PER_MEGAPIXEL = 2.0 * VRAM_HEADROOM_PER_MEGAPIXEL
"""Ceiling on the learned activation rate.

An observation is ``peak allocation - weights resident at the end of the pass``,
and during a Model Chain generation weights move constantly, so anything that
left the card during the window is counted as activations. The reading is
therefore biased high by construction, and only bounded by how much moved.

The bound has to be *scale-free*, which an earlier version of this was not. It
capped the learned rate at a multiple of the static estimate for the pass that
produced it -- but the static estimate is mostly a flat 1 GB, so dividing it by
a small pass's megapixels let a 512x512 observation authorise 15 GB per
megapixel, which then applied to every larger pass for the rest of the session.
One real log reached 7.4 GB/megapixel against an a-priori 0.75, and the reserve
that produced (10.4 GB for a 1280x960 pass) was larger than the model it was
protecting. Expressing the ceiling in the same units as the a-priori rate is
what makes it independent of where it was learned.
"""

MAX_RESERVE_FRACTION = 0.25
"""Most of the card a *learned* reserve may claim.

The second half of the same guard. A manual reserve is the user's decision and
Forge's own reservation is the host's, so neither is clamped -- but nothing
inferred from a measurement should be able to quietly annex half the card.
"""


def vram_headroom_bytes(width: int = 0, height: int = 0, batch: int = 1) -> int:
    """VRAM to keep free for a pass at the given size and batch.

    ``batch`` multiplies the pixel count, because that is what it multiplies in
    the sampler: every image in a batch carries its own latents, its own
    attention buffers and its own residual stream, and they are all live at the
    same moment. A reserve sized for one image and spent on five is short by
    four images' worth.

    Not hypothetical. On a 24 GB card holding a 17.8 GB checkpoint, a batch of
    five needed 2.8 GB of activations where the reserve had allowed for one
    image's 0.6 GB -- and the difference was very nearly exactly the 1.4 GB a
    llama-server was holding under a promise the plan had made it. The
    generation died before its first step.

    Four floors, and the largest of them wins:

    * the static estimate below, which scales with pixel count,
    * the largest activation peak actually observed this session, plus a margin,
    * the user's manual reserve, if they set one,
    * whatever Forge has already reserved for its own inference.

    Taking the maximum rather than a sum is the point. Each is an answer to the
    same question -- how much has to stay free for this pass to run without the
    driver spilling into system memory -- so the strongest answer is the useful
    one, and none of them may be undercut by the others.
    """
    return int(max(_static_headroom_bytes(width, height, batch),
                   _observed_headroom_bytes(width, height, batch),
                   manual_reserve_bytes(),
                   host_reserved_bytes()))


def _batched_megapixels(width: int, height: int, batch: int = 1) -> float:
    """Pixels in flight at once, in megapixels. Zero when the size is unknown."""
    if width <= 0 or height <= 0:
        return 0.0
    return (width * height) / 1_000_000 * max(int(batch or 1), 1)


def _static_headroom_bytes(width: int, height: int, batch: int = 1) -> int:
    """The a-priori estimate, from pass size alone."""
    megapixels = _batched_megapixels(width, height, batch)
    if megapixels <= 0:
        return int(VRAM_HEADROOM_BYTES)

    return int(VRAM_HEADROOM_BYTES + VRAM_HEADROOM_PER_MEGAPIXEL * max(megapixels - 1.0, 0.0))


def manual_reserve_bytes() -> int:
    """The user's minimum-VRAM-reserve setting, if they set one.

    Speculative warming may never eat into this, which is the whole reason the
    setting exists: automatic sizing is an estimate, and a user who knows their
    workload needs a way to say so that an estimate cannot argue with.
    """
    try:
        configured = float(option(OPT_VRAM_RESERVE, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0
    return int(max(configured, 0.0) * _GB)


def host_reserved_bytes() -> int:
    """VRAM Forge has already set aside for its own inference.

    Forge Neo exposes this under more than one name depending on build -- the
    "GPU Weights" slider becomes ``current_inference_memory``, and there are
    ``minimum_inference_memory()`` and ``extra_reserved_memory()`` helpers. The
    largest figure any of them reports is taken, because the failure mode is
    asymmetric: reserving more than the host asked for costs an eviction, and
    reserving less silently cancels a reservation the user made in the host's
    own settings.
    """
    try:
        from backend import memory_management
    except Exception:
        return 0

    reserved = 0
    for name in ("minimum_inference_memory", "extra_reserved_memory", "current_inference_memory"):
        source = getattr(memory_management, name, None)
        try:
            value = int(source() if callable(source) else source)
        except Exception:
            continue
        reserved = max(reserved, value)

    return max(reserved, 0)


# -- observed activation peaks --------------------------------------------- #
#
# The static estimate is a starting heuristic and was always documented as one.
# It cannot know the sampler, the batch size, or what else is hooked into the
# UNet on this machine. Watching what passes actually cost is how automatic mode
# stops being a guess.
#
# Only ever upward: the running figure is a maximum, not an average. A pass that
# happened to be cheap is not evidence that the next one will be, and the cost
# of an under-reserve -- the driver spilling into system RAM, sampling dropping
# from sub-second to tens of seconds per step, with no error to explain it -- is
# far worse than the cost of one unnecessary eviction.

_peak_bytes_per_megapixel = 0.0
_peak_observations = 0


def begin_pass_observation() -> None:
    """Start a measurement window for the pass about to run."""
    try:
        import torch

        from backend import memory_management

        torch.cuda.reset_peak_memory_stats(memory_management.get_torch_device())
    except Exception:
        pass


def observe_activation_peak(width: int, height: int, stage: str = "", batch: int = 1) -> int:
    """Fold the pass that just ran into the automatic reserve estimate.

    What is measured is the peak allocation minus the weights resident at the
    end of the pass, which is an estimate of the activations rather than a
    reading of them -- there is no host API that separates the two. It is used
    only as a *floor* on the reserve and only when it exceeds the static
    estimate, so an under-reading changes nothing and an over-reading is capped.

    Returns the bytes-per-megapixel figure now in force, or 0 if nothing could
    be measured.
    """
    global _peak_bytes_per_megapixel, _peak_observations

    if width <= 0 or height <= 0:
        return 0

    try:
        import torch

        from backend import memory_management

        device = memory_management.get_torch_device()
        peak = int(torch.cuda.max_memory_allocated(device))
    except Exception:
        return 0

    if peak <= 0:
        return 0

    weights = _all_resident_bytes()
    # Divided by the batch as well as the size, so what is learned is the cost
    # of *one* image at this resolution. Without that, a batch of five would
    # teach the estimator a per-megapixel figure five times too large, and every
    # single-image generation afterwards would reserve for a batch nobody asked
    # for -- and squeeze the language model off the card to do it.
    megapixels = max(_batched_megapixels(width, height, batch), 0.05)
    per_megapixel = max(peak - weights, 0) / megapixels
    per_megapixel = min(per_megapixel, MAX_LEARNED_BYTES_PER_MEGAPIXEL)

    _peak_observations += 1
    if per_megapixel > _peak_bytes_per_megapixel:
        _peak_bytes_per_megapixel = per_megapixel
        logger.debug(
            "Model Chain: observed %.2f GB of activations for a %dx%d %s pass at batch %d; "
            "the automatic VRAM reserve now allows for it",
            (peak - weights) / _GB,
            width,
            height,
            stage or "generation",
            max(int(batch or 1), 1),
        )

    _report_pass_peak(stage, peak)
    return int(_peak_bytes_per_megapixel)


def _report_pass_peak(stage: str, peak: int) -> None:
    """Tell the plan what the pass really cost, so a wrong reserve is visible.

    The one thing the extension could not see before. When an estimate is short
    the failure does not come back through any of this module's own paths --
    the host's allocator spills, or a VRAM guard in another extension stops the
    generation at ``unet-forward`` and reports it in its own words. Either way
    nothing here noticed, and the panel went on claiming a budget that had just
    been exceeded.

    Comparing the measured peak with what the plan protected is how that
    becomes a recorded miss rather than somebody else's warning.
    """
    try:
        import mc_plan

        mc_plan.check_observed(stage or "the pass", int(peak))
    except Exception:
        logger.debug("Model Chain: could not report the pass peak", exc_info=True)


def _observed_headroom_bytes(width: int, height: int, batch: int = 1) -> int:
    """What the observed peaks say this pass needs free, or 0 before any."""
    megapixels = _batched_megapixels(width, height, batch)
    if _peak_bytes_per_megapixel <= 0 or megapixels <= 0:
        return 0

    estimate = _peak_bytes_per_megapixel * max(megapixels, 0.05) * PEAK_MARGIN

    total = total_vram_bytes()
    if total > 0:
        estimate = min(estimate, total * MAX_RESERVE_FRACTION)

    return int(estimate)


def observed_peaks() -> tuple[int, int]:
    """Bytes-per-megapixel currently in force, and how many passes fed it."""
    return int(_peak_bytes_per_megapixel), _peak_observations


def vram_required_bytes(name: str, modules=None, width: int = 0, height: int = 0,
                        batch: int = 1) -> int:
    """VRAM a pass on ``name`` needs: the model, resident, plus its activations."""
    model = file_size_bytes(name, modules) * (1.0 + VRAM_MODEL_OVERHEAD_FRACTION)
    return int(model + vram_headroom_bytes(width, height, batch))

RAM_RESERVE_BYTES = 2 * _GB
"""System RAM never handed to the cache, so the host process cannot be OOM-killed."""


class ModelChainError(RuntimeError):
    """Raised when a requested checkpoint cannot be resolved or loaded."""


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


MODEL_FLAGS = ("kontext", "edit", "nunchaku", "klein", "wan", "pid", "anima", "krea2")
"""``backend.args.dynamic_args`` fields that describe *which model is loaded*.

``forge_loader`` sets these every time it loads a checkpoint, but a warm swap
never runs the loader, so they have to be carried with the cached model.
Leaving them describing the previously loaded checkpoint is not cosmetic:
``nunchaku`` selects a different LoRA application path, ``kontext`` and ``edit``
decide whether Flux.1 and Qwen-Image use reference conditioning, ``klein``
changes sampling, and ``pid`` changes the latent shape.
"""


def snapshot_model_flags() -> dict:
    """Capture the loader-set flags describing the currently loaded model."""
    try:
        from backend.args import dynamic_args

        return {name: getattr(dynamic_args, name, False) for name in MODEL_FLAGS}
    except Exception:
        return {}


DEFAULT_LATENT_SCALE = 8
"""``modules.processing.opt_f``'s value for every VAE that does not override it.

The host's own fallback, repeated here because the warm path has to reproduce
the loader's arithmetic exactly rather than approximately.
"""


def _model_vae(model):
    """The VAE the loader would read a model's latent scale from.

    ``forge_objects`` is what ``forge_model_reload`` inspects; a generation
    reassigns it from ``forge_objects_original`` on every batch, and the two
    share the same VAE object, so either answers the question.
    """
    for attribute in ("forge_objects", "forge_objects_original"):
        vae = getattr(getattr(model, attribute, None), "vae", None)
        if vae is not None:
            return vae
    return None


def latent_scale_of(model) -> int | None:
    """Pixels per latent unit for ``model``, or None if it cannot be determined.

    The one line of ``forge_model_reload`` that survives it::

        processing.opt_f = vae.upscale_ratio if isinstance(vae.upscale_ratio, int) else 8

    ``opt_f`` is a module-level global in ``modules.processing``, and it is what
    ``process_images_inner`` divides the requested pixel size by to shape the
    noise. Flux.2's VAE sets it to 16; SD/SDXL/Flux.1/Qwen set it to 8; a Wan
    VAE (Krea 2) carries a tuple rather than an int, so it takes the fallback.

    Returning None means "no opinion" -- a model whose VAE is not reachable is
    left alone rather than guessed at, because writing the wrong divisor is the
    exact failure this exists to prevent.
    """
    vae = _model_vae(model)
    if vae is None:
        return None

    ratio = getattr(vae, "upscale_ratio", None)
    if ratio is None:
        return None

    # bool is an int subclass and would sail through isinstance().
    if isinstance(ratio, int) and not isinstance(ratio, bool):
        return int(ratio)
    return DEFAULT_LATENT_SCALE


def current_latent_scale() -> int | None:
    """``modules.processing.opt_f`` as it stands, or None if unreadable."""
    try:
        from modules import processing

        return int(processing.opt_f)
    except Exception:
        return None


def align_latent_scale() -> bool:
    """Make the host's latent divisor agree with the model that is loaded.

    The warm swap sets ``opt_f`` where the loader would have (see
    ``_apply_latent_scale``), so in the ordinary run of things this finds
    nothing to do. It exists because the value is a global that any code path
    can leave stale, and the symptom -- a pass sampled at the wrong size -- is
    both silent and expensive: it is only visible in the finished image.

    Checked at the top of a generation, where correcting it is still free.
    Returns True when it had to correct something, and says so in the log:
    reaching here means a swap happened that nothing accounted for, which is
    worth seeing even though it has just been made harmless.
    """
    try:
        model = model_data_sd_model()
        if model is None:
            return False

        wanted = latent_scale_of(model)
        if wanted is None or wanted == current_latent_scale():
            return False

        from modules import processing

        stale = processing.opt_f
        processing.opt_f = wanted
    except Exception:
        # Called from a generation hook: a diagnostic that cannot read the
        # state it checks must not be the thing that stops the generation.
        logger.warning("Model Chain: failed to align the latent scale", exc_info=True)
        return False

    logger.warning(
        "Model Chain: the host's latent scale said %s but the loaded model's VAE says %s — "
        "corrected before sampling. Left alone, this pass would have been sampled at %s of "
        "the requested size in each dimension.",
        stale,
        wanted,
        f"{wanted}/{stale}",
    )
    return True


def describe_latent_geometry() -> str:
    """The loaded model's state that decides what size a pass comes out at.

    ``process_images_inner`` builds its noise as ``(latent_channels, [frames,]
    height // opt_f, width // opt_f)``, branching on the engine's ``is_wan``,
    and the VAE's own ratio turns that back into pixels. Those values are the
    whole geometry contract, and a warm swap restores them by pointer rather
    than by running the loader -- so when an output arrives at the wrong size,
    this is the state that says which of them stopped describing the loaded
    model. ``opt_f`` is listed both as it stands and as the loaded model's VAE
    would set it, because those disagreeing *is* the wrong size.

    Best effort by construction: it is only ever used to annotate a report.
    """
    try:
        model = model_data_sd_model()
        if model is None:
            return "no model loaded"

        vae = _model_vae(model)
        flags = snapshot_model_flags()
        active = ", ".join(name for name, value in flags.items() if value) or "none"

        return (
            f"engine={type(model).__name__} is_wan={getattr(model, 'is_wan', '?')} "
            f"vae={type(getattr(vae, 'first_stage_model', vae)).__name__} "
            f"latent_channels={getattr(vae, 'latent_channels', '?')} "
            f"downscale={getattr(vae, 'downscale_ratio', '?')} "
            f"opt_f={current_latent_scale()} (model wants {latent_scale_of(model)}) "
            f"flags={active}"
        )
    except Exception:
        return "unavailable"


def _apply_model_flags(flags: dict) -> None:
    """Re-apply captured flags and clear per-generation latent state.

    Mirrors what ``forge_loader`` does after a real load: ``dynamic_args.reset()``
    followed by assigning each model flag.
    """
    if not flags:
        return
    try:
        from backend.args import dynamic_args

        dynamic_args.reset()
        for name, value in flags.items():
            setattr(dynamic_args, name, value)
    except Exception:
        logger.warning("Model Chain: failed to restore model flags on warm swap", exc_info=True)


def _apply_latent_scale(model) -> None:
    """Point ``modules.processing.opt_f`` at the model being restored.

    The other thing ``forge_model_reload`` leaves behind that is not carried on
    the model object. ``opt_f`` is the divisor turning the requested pixel size
    into a latent shape, and it is global: the loader rewrites it on every load,
    so it always describes whichever checkpoint was loaded last.

    A warm swap skips the loader, so without this the divisor keeps describing
    the *other* stage's model -- and a chain between two models with different
    latent scales then samples at the wrong size from the second generation
    onwards. Krea 2 (8) into Flux.2 Klein (16) is the case that showed it: the
    first run loads both from disk and is correct, the second run warm-swaps
    Krea 2 back in under Flux.2's 16, and Stage 1 samples a 60x40 latent for a
    640x960 request and hands Stage 2 a 320x480 image. Stage 2 then faithfully
    refines the half-size image it was given, which is where it surfaces.

    Silent on the way through: this runs on every warm swap, and a line per
    swap saying the divisor is still 8 would be noise. A model whose VAE cannot
    be read leaves the value alone -- see ``latent_scale_of``.
    """
    scale = latent_scale_of(model)
    if scale is None:
        logger.debug("Model Chain: could not read the latent scale of the restored model")
        return

    try:
        from modules import processing

        processing.opt_f = scale
    except Exception:
        logger.warning("Model Chain: failed to restore the latent scale on warm swap", exc_info=True)


@dataclass
class _Entry:
    key: str
    """``str(model_data.forge_loading_parameters)`` -- what ``forge_hash`` compares against."""
    checkpoint_name: str
    sd_model: object
    size_bytes: int
    model_flags: dict = field(default_factory=dict)
    """``dynamic_args`` state as of this model's load; see ``MODEL_FLAGS``."""
    last_used: float = field(default_factory=time.monotonic)

    lora_state: str | None = None
    """``current_lora_hash`` as of the moment this model was put away.

    Recorded so restoring it can tell "the prepared state is the one we left"
    from "something changed it while it was in the cache". Only the first is
    safe to hand back to the host as-is.
    """
    lora_preservable: bool = True
    """Whether this backend's LoRA path survives being cached; see mc_lora."""
    stage: str = ""
    """Which stage prepared this state.

    The prepared state is only ever reused for the stage that built it. Model
    Chain's two stages are separate cache entries whenever they use different
    checkpoints, so this normally decides nothing -- but when a model is used by
    Stage 1 in one job and Stage 2 in another, its prepared LoRA state belongs
    to whichever stage last ran, and reusing it for the other one is precisely
    the leak the extension is supposed to prevent.
    """


class _Cache:
    """Fixed-capacity residency cache keyed by loading parameters."""

    def __init__(self, capacity: int = _MAX_SLOTS):
        self._capacity = capacity
        self._entries: dict[str, _Entry] = {}

    # -- queries ---------------------------------------------------------- #

    def get(self, key: str) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is not None:
            entry.last_used = time.monotonic()
        return entry

    def has(self, key: str) -> bool:
        return key in self._entries

    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    def names(self) -> list[str]:
        return [e.checkpoint_name for e in self._entries.values()]

    # -- mutation --------------------------------------------------------- #

    def admit(self, entry: _Entry, budget_bytes: int, protect: str | None = None) -> bool:
        """Take ``entry`` into the cache, evicting as needed to honour the budget.

        Returns False when the entry cannot be cached at all, in which case the
        caller must let the model be released (a disk reload next time). Host
        RAM exhaustion is far more disruptive than a VRAM OOM -- it can hang or
        kill the whole process -- so this check is fail-safe, never best-effort
        (section 4.5).

        ``protect`` names an entry that must not be evicted to make this room.
        Every switch stashes the outgoing model moments before restoring the
        incoming one, and plain recency picks the wrong victim at exactly that
        moment: the incoming model has not been touched since the last
        generation, so it is the least recently used, so it is the one thrown
        out -- immediately before it is needed. When the budget only fits one
        model, refusing to cache the model being *put away* costs a disk load
        on some later switch; evicting the one being *fetched* costs one now.
        """
        if entry.key in self._entries:
            self._entries[entry.key] = entry
            return True

        if entry.size_bytes > budget_bytes:
            logger.info(
                "Model Chain: %s (%.1f GB) exceeds the whole cache budget (%.1f GB); not caching",
                entry.checkpoint_name,
                entry.size_bytes / _GB,
                budget_bytes / _GB,
            )
            return False

        while self._entries and (
            len(self._entries) >= self._capacity
            or self.total_bytes() + entry.size_bytes > budget_bytes
        ):
            if not self._evict_oldest(protect):
                # Only the protected entry is left. Refusing to cache the
                # incoming one is the better trade: see the note on ``protect``.
                break

        if self.total_bytes() + entry.size_bytes > budget_bytes:
            return False

        self._entries[entry.key] = entry
        return True

    def _evict_oldest(self, protect: str | None = None) -> bool:
        """Drop the least recently used entry. Returns False if none may go."""
        candidates = [e for e in self._entries.values() if e.key != protect]
        if not candidates:
            return False
        self.drop(min(candidates, key=lambda e: e.last_used).key)
        return True

    def drop(self, key: str) -> None:
        """Stop holding ``key``'s model. The entry object is left intact.

        Popping is the whole of the release: it drops the only reference the
        cache had, and the garbage collector reclaims the weights as soon as
        nothing else is using them.

        Emphatically *not* ``entry.sd_model = None``. An entry handed out by
        ``get()`` is a live object a caller may still be holding, and blanking
        it reaches into their hands rather than releasing ours. That is not
        theoretical: ``reinstate_pending`` looks its entry up, stashes the
        outgoing model -- which can evict the very entry it is holding -- and
        then installs it. Blanking made it install None, leaving no model
        loaded and ``forge_hash`` still claiming there was one, which
        ``forge_model_reload`` believes; every generation afterwards failed on
        a null model until the WebUI was restarted.
        """
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        logger.info("Model Chain: releasing cached model %s", entry.checkpoint_name)

    def clear(self) -> None:
        for key in list(self._entries):
            self.drop(key)


_cache = _Cache()

_pending_restore: str | None = None
"""Checkpoint the extension owes a switch back to, set when Stage 2 leaves
Model B loaded while the UI selection has been put back to Model A."""

_last_refusal: str | None = None
"""Most recent model the cache could not take, if any.

Lets a cold load say whether it was a first load (expected, once) or the
consequence of a refused cache entry (actionable), instead of pointing at the
RAM setting every time.
"""


# --------------------------------------------------------------------------- #
# Host helpers
# --------------------------------------------------------------------------- #


def _loading_parameters_key() -> str:
    """Cache key for whatever the loading parameters currently *select*."""
    from modules.sd_models import model_data

    return str(model_data.forge_loading_parameters)


def _loaded_model_key() -> str:
    """Cache key for the model that is actually loaded right now.

    Deliberately not ``_loading_parameters_key()``. ``forge_loading_parameters``
    describes the *selection*, which routinely runs ahead of the loaded model:
    ``restore_selection`` points it back at Stage 1 while Stage 2's model is
    still the one in memory. Keying a stash off the selection there would file
    the outgoing model under the incoming model's key -- and since that key is
    already in the cache, the stash would be skipped entirely and the model
    dropped.

    ``forge_hash`` is by definition the loading-parameter string that produced
    the resident model, which is exactly what a later lookup will compare
    against.
    """
    from modules.sd_models import model_data

    return str(getattr(model_data, "forge_hash", "") or "")


def _is_real_model(model) -> bool:
    """False for ``FakeInitialModel``, the placeholder used before any load."""
    if model is None:
        return False
    return type(model).__name__ != "FakeInitialModel"


def checkpoint_info(name: str):
    from modules import sd_models

    if not name or name in ("None", "none"):
        return None
    return sd_models.get_closet_checkpoint_match(name)


INHERIT_MODULES = "Use same choices"
"""Sentinel meaning "keep Stage 1's VAE / text encoder selection".

Same spelling the host's own Hires VAE/TE dropdown uses, so the control reads
familiarly and infotext written by either is interchangeable.
"""


def resolve_modules(values) -> list[str] | None:
    """Turn a VAE/TE selection into the list of module paths to load.

    Returns None when Stage 1's selection should be inherited. An empty list is
    meaningful and distinct from None: it means "no additional modules", i.e.
    use whatever VAE and text encoder are built into the checkpoint.

    Resolution mirrors ``main_entry.modules_change``: names are looked up in the
    host's module list and the result is sorted, so a selection compares equal
    to what the host will actually store.
    """
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    if INHERIT_MODULES in values:
        return None

    from modules_forge.main_entry import module_list

    resolved = []
    for value in values:
        name = os.path.basename(str(value))
        if name in module_list:
            resolved.append(module_list[name])
        else:
            logger.warning("Model Chain: VAE/text encoder module %r was not found", value)

    return sorted(resolved)


def current_modules() -> list[str]:
    from modules import shared

    return list(getattr(shared.opts, "forge_additional_modules", []) or [])


def file_size_bytes(name: str, modules=None) -> int:
    """Disk footprint of a checkpoint plus the additional modules loaded with it.

    Used to size a model that is not currently loaded. The VAE and text encoders
    of Flux-family models live in separate files selected as additional modules,
    so a checkpoint file size alone would badly under-count them -- a Flux.2
    text encoder is several GB on its own.

    ``modules`` sizes a specific selection; None falls back to whatever is
    currently selected.
    """
    total = 0
    info = checkpoint_info(name)
    if info is not None:
        try:
            total += os.path.getsize(info.filename)
        except OSError:
            pass

    resolved = resolve_modules(modules)
    if resolved is None:
        resolved = current_modules()

    for module in resolved:
        try:
            total += os.path.getsize(module)
        except OSError:
            pass

    return total


def model_patchers(sd_model, attrs=("unet", "clip", "vae")) -> list:
    """The ``ModelPatcher`` objects behind a loaded checkpoint, in load order.

    ``forge_objects.clip`` wraps its patcher rather than being one, so each
    entry is unwrapped before use. Duplicates are dropped: some architectures
    share one patcher between two slots, and both ``free_memory`` and
    ``load_models_gpu`` would otherwise see it twice.
    """
    if not _is_real_model(sd_model):
        return []

    objects = getattr(sd_model, "forge_objects", None)
    if objects is None:
        return []

    found: list = []
    seen: set[int] = set()
    for attr in attrs:
        patcher = getattr(objects, attr, None)
        if patcher is None:
            continue
        patcher = getattr(patcher, "patcher", patcher)
        if id(patcher) in seen:
            continue
        seen.add(id(patcher))
        found.append(patcher)
    return found


def loaded_size_bytes(sd_model) -> int:
    """Actual footprint of a loaded model, summed across its patchers."""
    total = 0
    for patcher in model_patchers(sd_model):
        try:
            total += int(patcher.model_size())
        except Exception:
            continue
    return total


def free_vram_bytes() -> int:
    try:
        from backend import memory_management

        return int(memory_management.get_free_memory(memory_management.get_torch_device()))
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Two namespaces for one set of cards
# --------------------------------------------------------------------------- #
#
# A machine with two GPUs has two ways of numbering them and they do not have
# to agree. ``nvidia-smi`` orders by PCI bus; the CUDA runtime orders by
# ``CUDA_DEVICE_ORDER``, which defaults to fastest-first; and a process started
# with ``CUDA_VISIBLE_DEVICES`` renumbers from zero over the subset it can see.
# The language-model side records the first kind at setup time and Forge speaks
# the second, so the only honest way to ask "are these the same card" is to
# compare something that survives renumbering.
#
# That something is the UUID. Both sides can produce one -- nvidia-smi reports
# it per card and ``torch.cuda.get_device_properties`` carries it -- and the
# maps below are built once, because a machine does not gain a GPU while it is
# running.


def _uuid_key(value) -> str:
    """A GPU UUID reduced to the part both sources agree on.

    nvidia-smi writes ``GPU-6a1f...``; torch hands back a ``uuid.UUID``. The
    hex digits are the same and everything around them is not, so the hex is
    what is compared.
    """
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    return "".join(character for character in text if character in "0123456789abcdef")


_topology: dict | None = None
_topology_lock = threading.Lock()


def _cards() -> dict:
    """The machine's cards, keyed both ways, built once per session.

    Returns ``{"by_uuid": {uuid_key: physical_index},
               "ordinals": {physical_index: torch_ordinal},
               "names": {physical_index: name},
               "count": physical card count}``.

    Empty on any failure, and empty is a state every caller handles: it means
    the two namespaces cannot be reconciled, which is a reason to be careful
    rather than a reason to guess.
    """
    global _topology

    with _topology_lock:
        if _topology is not None:
            return _topology
        _topology = _read_topology()
        return _topology


def _read_topology() -> dict:
    found = {"by_uuid": {}, "ordinals": {}, "names": {}, "count": 0}
    try:
        from prompt_master.inference.device_detection import detect_gpus

        physical = detect_gpus()
    except Exception:
        logger.debug("Model Chain: could not enumerate the machine's GPUs", exc_info=True)
        physical = []
    for card in physical:
        key = _uuid_key(getattr(card, "uuid", ""))
        if key:
            found["by_uuid"][key] = int(card.physical_index)
        found["names"][int(card.physical_index)] = str(getattr(card, "name", "") or "")
    found["count"] = len(physical)

    try:
        import torch

        for ordinal in range(int(torch.cuda.device_count())):
            key = _uuid_key(getattr(torch.cuda.get_device_properties(ordinal), "uuid", ""))
            index = found["by_uuid"].get(key)
            if index is not None:
                found["ordinals"][index] = ordinal
    except Exception:
        logger.debug("Model Chain: could not read the CUDA devices' identities",
                     exc_info=True)

    if not found["ordinals"] and found["count"] == 1:
        # One card. There is nothing to confuse it with, and refusing to say so
        # would give every ordinary single-GPU installation the cautious
        # multi-card behaviour for no reason at all.
        only = next(iter(found["names"]), 0)
        found["ordinals"][only] = 0
    if found["ordinals"]:
        logger.info("Model Chain: GPU topology — %s",
                    "; ".join(f"physical {index} ({found['names'].get(index, '?')}) "
                              f"= CUDA ordinal {ordinal}"
                              for index, ordinal in sorted(found["ordinals"].items())))
    return found


def forget_topology() -> None:
    """Re-read the card map. For tests, and for a driver that came back."""
    global _topology, _smi_readings

    with _topology_lock:
        _topology = None
    _smi_readings = (0.0, {})


def _physical_index_of(ordinal: int) -> int:
    """The nvidia-smi index of the card this process calls ``ordinal``, or -1."""
    for index, mapped in _cards()["ordinals"].items():
        if mapped == ordinal:
            return int(index)
    return -1


def torch_ordinal_of(physical_index: int) -> int:
    """The ordinal this process uses for physical card ``physical_index``, or -1.

    -1 also means "visible to nvidia-smi and not to us", which is a real state
    rather than a failure: Forge started with ``CUDA_VISIBLE_DEVICES`` pinned to
    one card genuinely cannot address the other one through torch, and the
    honest answer is to say so and let the caller ask the driver instead.

    One card and *nothing known about the machine* is answered without any of
    that: there is exactly one ordinal it could be, so zero is the answer, and
    every ordinary single-GPU installation therefore behaves precisely as it
    did before any of this existed.

    That shortcut is deliberately not taken when the map does know something.
    A machine whose two cards nvidia-smi can see and this process can address
    only one of is the ``CUDA_VISIBLE_DEVICES`` case, and there the one visible
    ordinal is emphatically *not* the answer for the card that was hidden --
    saying it was is the same mistaken equality this whole function exists to
    stop making.
    """
    found = _cards()
    try:
        mapped = found["ordinals"].get(int(physical_index))
    except (TypeError, ValueError):
        return -1
    if mapped is not None:
        return int(mapped)
    if found["ordinals"] or found["count"] > 1:
        return -1
    try:
        import torch

        if int(torch.cuda.device_count()) == 1:
            return 0
    except Exception:
        logger.debug("Model Chain: could not count the visible CUDA devices", exc_info=True)
    return -1


def physical_card_name(physical_index: int) -> str:
    """What ``nvidia-smi`` calls physical card ``physical_index``, or ``""``."""
    try:
        return str(_cards()["names"].get(int(physical_index), ""))
    except (TypeError, ValueError):
        return ""


_SMI_READING_SECONDS = 0.5
"""How long a driver reading may be reused.

Short enough that it is still a live figure -- a placement decision that spans
a second is deciding against something that was true within it -- and long
enough that one negotiation asking the same question four times costs one
subprocess rather than four. nvidia-smi is tens of milliseconds each time, and
the negotiation ladder is a loop.
"""

_smi_readings: tuple[float, dict] = (0.0, {})


def physical_free_vram_bytes(physical_index: int) -> int:
    """Free VRAM on a physical card this process cannot address, via nvidia-smi.

    The slow path, and the only one there is for a card outside
    ``CUDA_VISIBLE_DEVICES``: torch cannot report on a device it was not given,
    and the driver can.
    """
    global _smi_readings

    now = time.monotonic()
    when, readings = _smi_readings
    if now - when > _SMI_READING_SECONDS or not readings:
        try:
            from prompt_master.inference.device_detection import detect_gpus

            readings = {int(card.physical_index): max(int(card.memory_free_mb), 0) * 1024 * 1024
                        for card in detect_gpus()}
            _smi_readings = (now, readings)
        except Exception:
            logger.debug("Model Chain: could not ask nvidia-smi for free VRAM on GPU %s",
                         physical_index, exc_info=True)
            return 0
    try:
        return int(readings.get(int(physical_index), 0))
    except (TypeError, ValueError):
        return 0


def image_device_uuid() -> str:
    """The UUID of the card the image side is on, or ``""``.

    The identity that survives both renumberings, and the one the
    language-model side already records at setup. When both halves can produce
    it, "are these the same card" stops being a question about indices at all.
    """
    ordinal = image_torch_ordinal()
    if ordinal < 0:
        return ""
    try:
        import torch

        return _uuid_key(getattr(torch.cuda.get_device_properties(ordinal), "uuid", ""))
    except Exception:
        logger.debug("Model Chain: could not read the image card's UUID", exc_info=True)
        return ""


def image_device_name() -> str:
    """What the driver calls the card the image side is on, or ``""``.

    The weaker identity, and worth having because it is available when the
    other two are not: an older torch exposes no UUID and a machine without
    nvidia-smi has no physical index to translate to, but a device name is
    there in both cases. Two *different* model names cannot be one card, which
    is enough to establish independence even when nothing can establish
    equality.
    """
    ordinal = image_torch_ordinal()
    if ordinal < 0:
        return ""
    try:
        import torch

        return str(torch.cuda.get_device_name(ordinal) or "").strip()
    except Exception:
        logger.debug("Model Chain: could not read the image card's name", exc_info=True)
        return ""


def image_torch_ordinal() -> int:
    """Forge's CUDA device as *this process* numbers it, or -1.

    A torch ordinal, and that is the whole of what it is. It indexes
    ``torch.cuda`` and nothing else: it is not comparable with a card index
    from ``nvidia-smi``, it is not stable across processes, and on a machine
    where Forge was started with ``CUDA_VISIBLE_DEVICES`` it is not even a
    number the rest of the system has heard of. Everything that needs to
    *compare* cards asks :func:`image_device_index` instead.
    """
    try:
        import torch
        from backend import memory_management

        device = memory_management.get_torch_device()
        if getattr(device, "type", "") != "cuda":
            return -1
        index = getattr(device, "index", None)
        if index is not None:
            return int(index)
        # ``torch.device("cuda")`` carries no index and does not mean card
        # zero -- it means whichever card is current, which on a two-card
        # machine is very often not card zero.
        return int(torch.cuda.current_device())
    except Exception:
        logger.debug("Model Chain: could not ask which card the image side is on",
                     exc_info=True)
        return -1


def image_device_index() -> int:
    """Which *physical* card the image side is on, or -1 when it cannot be said.

    Physical means the index ``nvidia-smi`` gives, because that is the number
    the language-model side records at setup time (``GpuInfo.physical_index``,
    written into the state file beside the card's UUID and name). This function
    exists to be compared with that number, and for a while it returned
    something else entirely.

    What it returned was the *torch ordinal* -- the index this process uses --
    and the two namespaces are genuinely different:

    * ``nvidia-smi`` numbers cards by PCI bus order;
    * the CUDA runtime numbers them by ``CUDA_DEVICE_ORDER``, which defaults to
      fastest-first, so a 5090 beside a 3090 is ordinal 0 whatever the bus says;
    * and a process started with ``CUDA_VISIBLE_DEVICES`` renumbers from zero
      over whatever it can see, so its ordinal 0 is any card at all.

    From a user's log: the language model was configured for physical card 0,
    named "NVIDIA GeForce RTX 5090"; Forge reported device 0 with 24.0 GB total,
    which is a 3090. Both were "card 0", they were not the same card, and every
    decision built on that equality was wrong in the same direction -- the
    3090's image plan resized the 5090's server, a 3090 shortfall evicted the
    checkpoint to make room on a card the model was not going to, and an image
    generation waited for a language model it shared nothing with.

    So the ordinal is translated, by UUID, into the physical namespace. -1 when
    that cannot be done, which every caller already treats as "unknown" and
    handles conservatively.
    """
    ordinal = image_torch_ordinal()
    if ordinal < 0:
        return -1
    return _physical_index_of(ordinal)


def device_free_vram_bytes(index: int | None = None) -> int:
    """VRAM the *driver* has free, which is a smaller number than :func:`free_vram_bytes`.

    The host's own figure is free device memory **plus** what its allocator is
    holding cached and not currently using, and for the host that is exactly
    right: torch will reuse its own cache before it asks the driver for
    anything. For another process it is a fiction. llama-server cannot be
    handed a block PyTorch is sitting on, and a placement sized against the
    host's number therefore asks for VRAM that only exists inside this process:
    the card reports twenty-two gigabytes free, the driver refuses an
    allocation of ten, and llama-server exits before it ever answers.

    So the LLM half of this extension asks the driver instead. Falls back to
    the host's figure when the question cannot be put -- a wrong number is
    still better than no placement at all, and it is the number this extension
    used for its whole first year.

    ``index`` is a **physical** card index -- the one ``nvidia-smi`` gives and
    the one the language-model side records at setup. It is translated to this
    process's ordinal before torch is asked, and that translation is the fix
    for a real failure: an index of 0 meaning "the 5090" was being handed
    straight to ``torch.device("cuda", 0)``, which in Forge's process is
    whichever card the CUDA runtime calls zero. On the machine that reported
    this, that was the 3090 -- so a placement destined for one card was sized
    against the free VRAM of another, and shrank to five of sixty-five layers
    because the *image* model had just filled the card it was reading.

    A card this process cannot address at all -- Forge pinned to one GPU with
    ``CUDA_VISIBLE_DEVICES`` -- is asked of the driver through nvidia-smi
    instead. Slower, and the only honest answer available: torch cannot report
    on a device it was not given.

    A machine with two cards has two answers to "how much is free", and
    answering the second card's question with the first card's number is how a
    role pinned to an otherwise idle 5090 gets placed against a 3090 that the
    image model has nearly filled.
    """
    ordinal = None
    if index is not None:
        if int(index) < 0:
            return free_vram_bytes()
        ordinal = torch_ordinal_of(int(index))
        if ordinal < 0:
            # Not addressable from here. nvidia-smi still sees it, and a
            # reading from the driver is better than this process's opinion
            # about a different card.
            reading = physical_free_vram_bytes(int(index))
            if reading > 0:
                return reading
            logger.debug("Model Chain: GPU %s could not be read from this process or "
                         "from nvidia-smi", index)
            return 0
    try:
        import torch
        from backend import memory_management

        if ordinal is None:
            device = memory_management.get_torch_device()
            if getattr(device, "type", "") != "cuda":
                return free_vram_bytes()
        else:
            device = torch.device("cuda", ordinal)
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        logger.debug("Model Chain: could not ask the driver for free VRAM", exc_info=True)
        return free_vram_bytes() if index is None else 0


def release_cached_vram() -> int:
    """Hand the allocator's cached blocks back to the driver. Returns bytes recovered.

    Free for the host and invisible to everybody else: an allocator that has
    finished with a block keeps it, because keeping it is how the next
    allocation is fast. Nothing is unloaded here and no model moves -- what is
    given up is the *empty* space between them, which is worth doing exactly
    once, immediately before another process is asked to fit in it.
    """
    before = device_free_vram_bytes()
    released = False
    try:
        from backend import memory_management

        try:
            memory_management.soft_empty_cache(True)
        except TypeError:
            memory_management.soft_empty_cache()
        released = True
    except Exception:
        logger.debug("Model Chain: the host has no cache-emptying entry point", exc_info=True)
    if not released:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            logger.debug("Model Chain: could not empty the allocator cache", exc_info=True)
            return 0
    return max(device_free_vram_bytes() - before, 0)


def total_vram_bytes() -> int:
    try:
        from backend import memory_management

        return int(memory_management.get_total_memory(memory_management.get_torch_device()))
    except Exception:
        return 0


def free_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def total_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


DEFAULT_BUDGET_FRACTION = 0.6
"""Share of system RAM the cache may hold by default.

This is a *ceiling*, not a reservation. The real guard is the live free-RAM
check in ``_stash_current``, which refuses an entry that would push available
memory below ``RAM_RESERVE_BYTES``.

An earlier default of one third was too tight to be useful: a Flux.2 Klein
checkpoint with a Qwen3 text encoder occupies roughly 14 GB resident, so on a
32 GB machine the budget came to 10.6 GB and the model was refused outright --
every switch then cold-loaded from disk, which is exactly the cost this cache
exists to avoid. Because weights are *moved* between devices rather than
copied, the cache mostly holds whichever model is not currently in VRAM, so a
ceiling near a large model's full size is the useful setting.
"""


def cache_budget_bytes() -> int:
    """Ceiling for RAM held by the cache (section 4.5)."""
    from modules import shared

    configured = getattr(shared.opts, "model_chain_ram_budget_gb", 0) or 0
    if configured > 0:
        return int(configured * _GB)

    total = total_ram_bytes()
    if total <= 0:
        # System RAM could not be queried. Refusing to cache is the fail-safe
        # answer: models reload from disk, which is slow but cannot OOM the
        # host. Set the budget explicitly in Settings to override this.
        logger.warning(
            "Model Chain: unable to detect system RAM; the model cache is disabled. "
            'Set "Model Chain: max system RAM for model cache (GB)" in Settings to enable it.'
        )
        return 0

    return int(total * DEFAULT_BUDGET_FRACTION)


def default_ram_budget_gb() -> float:
    total = total_ram_bytes()
    if total <= 0:
        return 16.0
    return round(total / _GB / 3, 1)


# --------------------------------------------------------------------------- #
# Residency planning (section 6.6)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResidencyPlan:
    kind: str
    """One of ``warm``, ``dual``, ``offload``, ``disk`` or ``unavailable``."""
    message: str

    @property
    def is_warning(self) -> bool:
        return self.kind in ("disk", "unavailable")


def plan(target_name: str, modules=None) -> ResidencyPlan:
    """Predict what the switch to ``target_name`` will cost, for the UI status line."""
    if not target_name or target_name in ("None", "none"):
        return ResidencyPlan("unavailable", "No Stage 2 model selected.")

    info = checkpoint_info(target_name)
    if info is None:
        return ResidencyPlan("unavailable", f'Checkpoint "{target_name}" was not found.')

    from modules import shared

    current = shared.sd_model
    current_size = loaded_size_bytes(current)
    target_size = file_size_bytes(target_name, modules)

    if _cache.has(_target_key_for(target_name, modules)):
        return ResidencyPlan(
            "warm",
            f"{info.name_for_extra} is cached in RAM — switch is a warm swap, no disk read.",
        )

    free_vram = free_vram_bytes()
    if free_vram <= 0:
        return ResidencyPlan(
            "unavailable",
            "Unable to query free VRAM — residency cannot be predicted.",
        )

    # Predict against the same figure make_vram_room() will act on, so the
    # pre-flight label does not promise dual residency the switch then breaks.
    target_required = vram_required_bytes(target_name, modules)
    if free_vram >= target_required:
        return ResidencyPlan(
            "dual",
            f"Both models fit in VRAM — no offload expected "
            f"({free_vram / _GB:.1f} GB free, {target_required / _GB:.1f} GB needed).",
        )

    budget = cache_budget_bytes()
    ram_headroom = min(free_ram_bytes() - RAM_RESERVE_BYTES, budget)
    if current_size and ram_headroom >= current_size:
        return ResidencyPlan(
            "offload",
            f"Insufficient VRAM for both — Model A will offload to system RAM "
            f"({free_vram / _GB:.1f} GB VRAM free, {target_size / _GB:.1f} GB needed).",
        )

    return ResidencyPlan(
        "disk",
        f"Insufficient RAM for cache — Model A will reload from disk on switch "
        f"(needs {current_size / _GB:.1f} GB, {max(ram_headroom, 0) / _GB:.1f} GB available to the cache).",
    )


def _target_key_for(name: str, modules=None) -> str:
    """Best-effort cache key for a checkpoint that is not currently selected.

    Mirrors how ``main_entry.refresh_model_loading_parameters`` builds
    ``forge_loading_parameters`` -- same keys, same order, same sorted module
    list -- so the prediction matches the real key without mutating any global
    state.
    """
    try:
        from modules import shared
        from modules_forge.main_entry import forge_unet_storage_dtype_options

        info = checkpoint_info(name)
        if info is None:
            return ""
        unet_storage_dtype, _ = forge_unet_storage_dtype_options.get(
            shared.opts.forge_unet_storage_dtype, (None, False)
        )
        resolved = resolve_modules(modules)
        if resolved is None:
            resolved = current_modules()
        return str(
            dict(
                checkpoint_info=info,
                additional_modules=resolved,
                unet_storage_dtype=unet_storage_dtype,
            )
        )
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Residency control
# --------------------------------------------------------------------------- #


def _stash_current(stage: str = "", protect: str | None = None) -> None:
    """Move the currently loaded model into the cache before it is displaced.

    ``stage`` is which stage this model was serving, and is carried on the entry
    so its prepared LoRA state is only ever offered back to the same stage.

    ``protect`` is the cache key of the model about to be swapped *in*, which
    must survive making room for this one. Callers know it; the cache's recency
    order does not, and would evict it first.
    """
    from modules.sd_models import model_data

    model = model_data.sd_model
    if not _is_real_model(model):
        return

    key = _loaded_model_key()
    if not key:
        logger.warning(
            "Model Chain: cannot identify the loaded checkpoint, so it will not be cached"
        )
        return

    flags = snapshot_model_flags()
    preservable, unpreservable_because = mc_lora.is_preservable(flags)
    lora_state = mc_lora.state_of(model)

    if _cache.has(key):
        existing = _cache.get(key)  # refresh recency
        # The same object can come back through here having been used by the
        # other stage, or with a different LoRA applied. The entry describes
        # what is being put away *now*, not what was put away last time.
        existing.lora_state = lora_state
        existing.lora_preservable = preservable
        existing.stage = stage or existing.stage
        return

    info = getattr(model, "sd_checkpoint_info", None)
    name = info.name_for_extra if info is not None else "(unknown)"
    size = loaded_size_bytes(model) or file_size_bytes(name)

    if lora_state and not preservable:
        logger.info(
            "Model Chain: %s keeps no reusable LoRA state — %s", name, unpreservable_because
        )

    # What the cache may grow to right now: whatever it already holds, plus the
    # RAM actually free beyond the reserve. Expressing it this way lets admit()
    # evict older entries to make room instead of refusing outright, and keeps
    # the ceiling honest as free memory changes.
    headroom = max(free_ram_bytes() - RAM_RESERVE_BYTES, 0)
    budget = min(cache_budget_bytes(), _cache.total_bytes() + headroom)

    entry = _Entry(
        key=key,
        checkpoint_name=name,
        sd_model=model,
        size_bytes=size,
        model_flags=flags,
        lora_state=lora_state,
        lora_preservable=preservable,
        stage=stage,
    )

    global _last_refusal

    if _cache.admit(entry, budget, protect=protect):
        _last_refusal = None
        logger.info(
            "Model Chain: holding %s in the RAM cache (%.1f GB; cache now %.1f GB of %.1f GB budget)",
            name,
            size / _GB,
            _cache.total_bytes() / _GB,
            budget / _GB,
        )
    else:
        _last_refusal = name
        # Say which of the two it is. "Not enough RAM" alone reads as a fault
        # when the cache has in fact just made the right choice on purpose.
        kept = protect is not None and _cache.has(protect)
        logger.warning(
            "Model Chain: not enough system RAM to cache %s (%.1f GB needed, %.1f GB available to "
            "the cache)%s%s — it will reload from disk on every switch. Raise "
            '"Model Chain: max system RAM for model cache (GB)" in Settings if you have the RAM.',
            name,
            size / _GB,
            budget / _GB,
            ", and the model about to be swapped in was kept in preference to it" if kept else "",
            _llm_ram_note(),
        )


def _llm_ram_note() -> str:
    """Whether a RAM-backed language model is why this cache cannot grow.

    Section 10.9's addition, and it is *only* visibility. The admission
    arithmetic above is unchanged and correct as it stands: it reads live
    available memory, so a processor-resident llama-server has already reduced
    what the cache may take without anybody having to model it. What was
    missing was the sentence saying so -- otherwise the message points at the
    RAM-budget setting, and raising that setting cannot conjure memory another
    process is holding.

    Never adds LLM memory back as though the cache could have it. Deciding to
    stop a running language model is the broker's, and it is not made here.
    """
    try:
        import mc_broker

        held = int(mc_broker.llm_host_ram_bytes())
    except Exception:
        return ""
    if held <= 0:
        return ""
    return (f", while a language model configured for system RAM is using about "
            f"{held / _GB:.1f} GB of it")


def _restore_prepared_state(entry: _Entry, stage: str) -> str:
    """Decide whether a restored model may keep its prepared LoRA state.

    Preserving is the default and costs nothing to do -- the host's own early
    return handles it, and this function's whole job is to spot the cases where
    that early return would be wrong and take it away.

    Returns ``"preserved"``, ``"rebuilt"`` or ``"none"`` (there was no LoRA
    state either way), which is what the caller logs.
    """
    model = entry.sd_model
    live = mc_lora.state_of(model)

    if live is None and entry.lora_state is None:
        return "none"

    def rebuild(reason: str) -> str:
        mc_lora.invalidate(model, f"{entry.checkpoint_name}: {reason}")
        return "rebuilt"

    if not option(OPT_PRESERVE_LORA, True):
        return rebuild("preserving prepared LoRA state is disabled in Settings")

    if not entry.lora_preservable:
        return rebuild("this backend rebuilds its LoRA state rather than moving it")

    if entry.stage and stage and entry.stage != stage:
        # Same checkpoint on both stages. The state on the object belongs to
        # whichever stage last ran; handing it to the other one is exactly the
        # cross-stage leak this extension must not have.
        return rebuild(f"the prepared state belongs to {entry.stage}, not {stage}")

    if live != entry.lora_state:
        # Something moved the hash on while the model sat in the cache. We do
        # not know what, so we do not trust it.
        return rebuild("its LoRA state changed while it was cached")

    return "preserved"


def _log_restored(entry: _Entry, prepared: str) -> None:
    """One line saying what came back and whether its LoRA state came with it."""
    if prepared == "preserved":
        detail = " with its prepared LoRA state intact"
    elif prepared == "rebuilt":
        detail = "; its LoRA state will be rebuilt"
    else:
        detail = ""

    logger.info("Model Chain: restored %s from the RAM cache%s", entry.checkpoint_name, detail)


def invalidate_prepared_state(reason: str = "") -> bool:
    """Force the loaded model's LoRA state to be rebuilt before it is next used.

    The failure valve for section 4.5's "never poison a cached model" rule. A
    LoRA that raised half way through application leaves the host believing a
    state is applied that partly is not, and because Model Chain keeps the model
    object rather than reloading it, that belief would outlive the job that
    caused it. Throwing the belief away costs one re-application.
    """
    try:
        from modules.sd_models import model_data

        return mc_lora.invalidate(model_data.sd_model, reason)
    except Exception:
        logger.debug("Model Chain: could not invalidate the prepared LoRA state", exc_info=True)
        return False


def ensure_resident(name: str, modules=None) -> str:
    """Make ``name`` the loaded checkpoint, and report how it got there.

    ``modules`` is the VAE / text encoder selection for this checkpoint; None
    inherits whatever is currently selected. Because the host folds the module
    list into ``forge_loading_parameters``, a checkpoint paired with its own
    VAE and text encoder is a distinct cache entry -- which is what lets Stage 1
    and Stage 2 hold different encoders in memory at the same time.

    Returns ``"unchanged"``, ``"warm"`` (restored from the RAM cache) or
    ``"cold"`` (read from disk). This is the only entry point the orchestration
    code uses to change models; it knows nothing about slots.
    """
    from modules import processing, sd_models, shared
    from modules.sd_models import model_data
    from modules_forge import main_entry

    # A preload may still be moving Stage 1's weights. Never move models
    # underneath it -- wait, then proceed against a settled state. The lock is
    # belt and braces on top of the join: joining a thread that has already
    # finished starting a *second* one would otherwise be a race, and this way
    # serialisation does not depend on getting that ordering right.
    join_preload()

    info = checkpoint_info(name)
    if info is None:
        raise ModelChainError(f'Stage 2 checkpoint "{name}" was not found.')

    with _model_lock:
        resolved_modules = resolve_modules(modules)

        if _is_real_model(model_data.sd_model):
            current_info = getattr(model_data.sd_model, "sd_checkpoint_info", None)
            same_checkpoint = current_info is not None and current_info.filename == info.filename
            same_modules = resolved_modules is None or resolved_modules == current_modules()
            if same_checkpoint and same_modules:
                # Both stages on one checkpoint. There is no swap and therefore
                # no cache entry involved, so the host's own hash comparison is
                # the only thing deciding what happens to the LoRA state -- which
                # is correct, and is why nothing here interferes with it.
                return "unchanged"

        # Predicted without touching any global state, so it can be known
        # *before* the stash below -- which is the only place it is useful.
        # Evicting Stage 2 here to make room for Stage 1 is the expensive
        # mistake of the two: the swap that follows is the one that would have
        # been warm, and instead it reads the whole checkpoint back off disk.
        incoming_key = _target_key_for(name, resolved_modules) or None

        _stash_current(stage=STAGE_1, protect=incoming_key)

        # Let the host recompute loading parameters, dynamic LoRA flags and the
        # checkpoint selection. save=False keeps this transient switch out of the
        # user's config.json. Both changes are made with refresh=False and followed
        # by a single refresh, the same way the host's own hires-fix pass swaps
        # checkpoint and modules together.
        main_entry.checkpoint_change(name, None, save=False, refresh=False)
        if resolved_modules is not None:
            main_entry.modules_change(resolved_modules, None, save=False, refresh=False)
        main_entry.refresh_model_loading_parameters()

        target_key = _loading_parameters_key()
        entry = _cache.get(target_key)

        if entry is not None and _is_real_model(entry.sd_model):
            # Warm path: pointer swap. forge_model_reload() will return early, so no
            # unload_all_models() runs and VRAM is reclaimed only on demand.
            model_data.set_sd_model(entry.sd_model)
            model_data.forge_hash = target_key
            shared.opts.data["sd_checkpoint_hash"] = entry.sd_model.sd_checkpoint_info.sha256
            # The loader did not run, so re-apply the globals it would have set:
            # the model flags, and the latent scale that sizes the next pass.
            _apply_model_flags(entry.model_flags)
            _apply_latent_scale(entry.sd_model)
            prepared = _restore_prepared_state(entry, STAGE_2)
            processing.need_global_unload = False
            _log_restored(entry, prepared)
            return "warm"

        sd_models.forge_model_reload()
        return "cold"


def restore_selection(name: str, modules=None) -> None:
    """Point the UI selection and loading parameters back at Stage 1's pair.

    Both halves have to go back: restoring the checkpoint but leaving Stage 2's
    VAE and text encoder selected would silently change what Stage 1 loads next
    time, and would leave the user's VAE/TE dropdown showing Stage 2's files.

    Deliberately does *not* reload. Section 3.1 allows exactly one checkpoint
    switch per Generate click, so the swap back to Model A is deferred to the
    start of the next generation, where ``reinstate_pending()`` turns it into a
    warm pointer swap.
    """
    global _pending_restore

    info = checkpoint_info(name)
    if info is None:
        return

    from modules_forge import main_entry

    main_entry.checkpoint_change(name, None, save=False, refresh=False)
    if modules is not None:
        main_entry.modules_change(modules, None, save=False, refresh=False)
    main_entry.refresh_model_loading_parameters()
    _pending_restore = name


def reinstate_pending() -> bool:
    """Warm-swap the selected checkpoint back in, if we left a different one loaded.

    Called at the top of every generation. Returns True when a swap happened.
    """
    global _pending_restore

    from modules import processing
    from modules.sd_models import model_data

    with _model_lock:
        if _pending_restore is None:
            return False

        key = _loading_parameters_key()
        entry = _cache.get(key)
        if entry is None or not _is_real_model(entry.sd_model):
            # Nothing cached: leave it to forge_model_reload() to load from disk.
            _pending_restore = None
            return False

        if model_data.sd_model is entry.sd_model:
            _pending_restore = None
            return False

        # Held in a local across the stash below. Stashing Stage 2's model can
        # evict this very entry -- the two models are the two largest things in
        # the cache, and Stage 2's arrival is exactly when the budget runs out
        # -- so the entry must not be re-read afterwards.
        restored = entry.sd_model

        # The outgoing model is Stage 2's, and this is the last moment it is
        # reachable. Both the preload and an ordinary Generate come through here.
        # Stage 1 is protected: it is the least recently used entry precisely
        # because it has been waiting for this moment, and it is about to be
        # installed on the very next line.
        capture_stage_2_components()
        _stash_current(stage=STAGE_2, protect=key)

        if not _is_real_model(restored):
            # Belt and braces. Installing a null model here would leave the host
            # with nothing loaded and forge_hash still asserting otherwise,
            # which wedges every following generation rather than costing this
            # one a reload. A cold load is the safe answer.
            logger.warning(
                "Model Chain: the cached %s went away during the swap back; "
                "Stage 1 will be loaded from disk instead",
                _pending_restore,
            )
            _pending_restore = None
            return False

        model_data.set_sd_model(restored)
        model_data.forge_hash = key
        _apply_model_flags(entry.model_flags)
        _apply_latent_scale(restored)
        prepared = _restore_prepared_state(entry, STAGE_1)
        processing.need_global_unload = False
        _log_restored(entry, prepared)
        _pending_restore = None
        return True


def ensure_model_loadable() -> bool:
    """Clear a stale loading hash when nothing is actually loaded.

    ``forge_model_reload`` returns early whenever ``forge_hash`` still matches
    the current loading parameters, handing back whatever ``model_data.sd_model``
    holds. If that is empty while the hash goes on asserting a load, every
    generation dies on a null model and retrying cannot help -- the only way out
    is restarting the WebUI.

    Model Chain writes ``forge_hash`` by hand to make its warm swaps work, so
    Model Chain is what should check the invariant those swaps depend on.
    Clearing the hash costs one disk load and gives the user back a working UI.
    Returns True when something was actually wrong.
    """
    try:
        from modules.sd_models import model_data

        if _is_real_model(model_data.sd_model) or not model_data.forge_hash:
            return False

        logger.warning(
            "Model Chain: no model is loaded, but the host still holds a loading hash — "
            "clearing it so this generation loads from disk instead of failing on a null "
            "model. If you saw this after a chained generation, please report it."
        )
        model_data.forge_hash = ""
        return True
    except Exception:
        logger.warning("Model Chain: failed to check the loaded-model state", exc_info=True)
        return False


def last_refusal() -> str | None:
    """Name of the last model the cache refused, or None."""
    return _last_refusal


# -- Stage 1 preload ------------------------------------------------------- #
#
# Restoring Stage 1 is two separate costs. reinstate_pending() does the cheap
# half -- a pointer swap out of the RAM cache -- but the weights themselves are
# still in system RAM at that point, and the expensive half happens lazily when
# the sampler first asks for them. Deferring both to the next Generate click
# means the user waits through the whole move before a single step runs.
#
# Doing it eagerly at the end of the generation only helps if it overlaps the
# time the user spends looking at the result. Running it inline in postprocess()
# would not overlap anything: the gallery is not populated until the generation
# call returns, so an inline preload would simply move the same wait to before
# the images appear and gain nothing. Hence a background thread.
#
# The thread is joined by every entry point of this module that touches models,
# so the extension can never race itself: the worst case is that the user clicks
# Generate immediately and waits exactly as long as they would have anyway.
#
# "Preloaded" has to mean generation-ready, not "a thread ran". The worker
# therefore does three things and checks the third: it swaps Stage 1 back in, it
# makes room for it, and it moves its weights -- and then measures how much of
# the model the host actually reports resident, so the next generation can say
# warm, partially warm or cold and mean it.
#
# And a preload that goes wrong must cost one generation, not every generation.
# Two consecutive failures retire the feature for the session; a single failure
# leaves the model exactly where the host's own synchronous path expects to find
# it, which is the path the next Generate then takes.

_preload_thread: threading.Thread | None = None
_preload_reinstated = False

PRELOAD_FAILURE_LIMIT = 2
"""Consecutive failures before the preload takes itself out of service.

One failure is a bad moment -- a driver hiccup, a model that had just been
evicted underneath it. Two in a row is a machine where this does not work, and
continuing to try on every generation would be a failure loop: the user would
pay the exception, the log noise and the fallback on every click, forever.
"""

READY_FRACTION = 0.98
"""Share of the model that must be on the GPU to call a preload "ready".

Not 1.0. ``model_size()`` and what the host reports loaded are computed
differently enough that an exactly-full model can read a hair under, and a
preload that did its whole job should not be reported as partial because of a
rounding difference.
"""

_preload_failures = 0
_preload_disabled_reason: str | None = None
_preload_option_seen: bool | None = None


@dataclass(frozen=True)
class PreloadResult:
    """What the last preload achieved, in the terms the next generation needs."""

    state: str
    """``ready``, ``partial``, ``nothing`` or ``failed``."""
    key: str = ""
    """Loading-parameter key the preload warmed, for detecting a stale result."""
    checkpoint: str = ""
    moved_bytes: int = 0
    resident_bytes: int = 0
    model_bytes: int = 0
    seconds: float = 0.0
    detail: str = ""


_preload_result: PreloadResult | None = None


def preload_enabled() -> bool:
    """Whether a preload may start, honouring the setting and the failure limit.

    Flipping the setting off and on again clears the failure count. That is the
    only in-session way back from the circuit breaker, and it is the obvious
    one: a user who has just changed something and wants to retry reaches for
    the switch, not for a restart.
    """
    global _preload_failures, _preload_disabled_reason, _preload_option_seen

    enabled = bool(option(OPT_PRELOAD, PRELOAD_DEFAULT))
    if enabled and _preload_option_seen is False:
        _preload_failures = 0
        _preload_disabled_reason = None
    _preload_option_seen = enabled

    if not enabled:
        return False
    return _preload_disabled_reason is None


RESTORE = "restore"
"""Swap a checkpoint this extension swapped out back in from the RAM cache."""

RESIDENT = "resident"
"""Move the loaded checkpoint's own weights the rest of the way onto the card."""

FROM_DISK = "from disk"
"""Have the host load the selected checkpoint, because nothing usable is loaded."""


def _preload_task(allow_disk_load: bool = False) -> str | None:
    """What warming Stage 1 would actually have to do, or None for nothing.

    Three kinds of work, and telling them apart is the difference between a
    warm-up that warms something and one that reports success having moved
    nothing:

    :data:`RESTORE`
        A checkpoint this extension swapped out is waiting to come back. Until
        this function existed it was the *only* reason a preload ever ran --
        and a plan with no Stage 2 never swaps anything out, so on a
        single-stage plan the preload had nothing to trigger it, ever. The
        model then sat in system RAM between generations while the warm-up
        reported, accurately and uselessly, that it had finished in 0.0s with
        the image model cold.

    :data:`RESIDENT`
        The selected checkpoint *is* the loaded one and its weights are not all
        on the card. This is the state a generation ends in: the host moves out
        what it needs to and leaves the rest wherever it landed. Every
        following generation then pays to move it back, and every LoRA is
        applied against weights that have to be walked across the bus first --
        which is why a slow LoRA and a cold model are one symptom, not two.

    :data:`FROM_DISK`
        Nothing usable is loaded at all, or the selection has moved somewhere
        the cache cannot answer for. Only offered when the caller asks for it,
        because reading a checkpoint off disk is the one job here that is
        expensive whether or not anybody is waiting on it -- which makes it
        right for an explicit warm-up and wrong for the background pass that
        follows every generation.
    """
    if _pending_restore is not None:
        return RESTORE

    if not _is_real_model(model_data_sd_model()):
        return FROM_DISK if allow_disk_load else None

    try:
        selection, loaded = _loading_parameters_key(), _loaded_model_key()
    except Exception:
        selection = loaded = ""
    if selection and loaded and selection != loaded:
        # The UI points at something else. Installing a cache entry from here
        # is reinstate_pending()'s job and it has no pending restore to work
        # from, so the host's own loader is the honest answer.
        return FROM_DISK if allow_disk_load else None

    resident, total = _loaded_residency()
    if total <= 0:
        # Residency cannot be measured. Moving weights on a guess is how a
        # warm-up becomes the thing it was added to prevent.
        return None
    if resident >= total * READY_FRACTION:
        return None  # already where the next generation needs it

    return RESIDENT


def preload_async(width: int = 0, height: int = 0, *, allow_disk_load: bool = False,
                  force: bool = False) -> bool:
    """Start warming Stage 1's weights into VRAM in the background.

    ``width``/``height`` size the VRAM budget, and are the *current*
    generation's Stage 1 size because the next one's is not knowable yet. That
    only affects how much room is freed, and ``before_process`` re-checks
    against the real size before Stage 1 runs.

    ``allow_disk_load`` permits the expensive case -- see :func:`_preload_task`.
    ``force`` runs even with the preload setting off, and is for :mod:`mc_arm`:
    that setting is consent to a background thread after every generation,
    which is not the same question as "load what this generation needs before
    it starts", and a user who turned the warm-up on and got a language model
    and no image model had answered the second question, not the first. The
    circuit breaker still applies to both -- a machine where this does not work
    is a machine where it does not work however it was asked for.

    Returns True if a preload was started.
    """
    global _preload_thread, _preload_result

    if not preload_enabled() and not (force and _preload_disabled_reason is None):
        return False

    # Before the task is decided, not after: an in-flight preload is moving the
    # very state that decision reads.
    join_preload()

    task = _preload_task(allow_disk_load)
    if task is None:
        return False

    _preload_result = None
    _preload_thread = threading.Thread(
        target=_preload_worker,
        args=(width, height, task),
        name="model-chain-preload",
        daemon=True,
    )
    _preload_thread.start()
    return True


def join_preload(timeout: float | None = None) -> None:
    """Wait for any in-flight preload to finish.

    Waiting is always the right answer, even when it is the user's Generate
    click doing the waiting: the work being waited on is work that click needs
    doing anyway, so the wait is never wasted. It is logged when it is long
    enough to notice, so a slow first generation has a visible explanation.
    """
    global _preload_thread

    thread = _preload_thread
    if thread is None or thread is threading.current_thread():
        return

    started = time.perf_counter()
    thread.join(timeout)
    waited = time.perf_counter() - started

    if not thread.is_alive():
        _preload_thread = None
        if waited > 0.5:
            logger.info(
                "Model Chain: waited %.1fs for the Stage 1 preload to finish before continuing",
                waited,
            )
    elif timeout is not None:
        logger.warning(
            "Model Chain: the Stage 1 preload is still running after %.1fs; "
            "this generation will wait for the model lock rather than race it",
            waited,
        )


def consume_preload() -> bool:
    """True once if a background preload already moved Stage 1's weights.

    Swapped in from the cache, loaded from disk or topped up where they sat --
    all three leave VRAM looking different from however the last generation
    left it, and all three therefore have the same consequence here.

    Lets ``before_process`` tell "nothing to do" apart from "already done", so
    the VRAM budget still gets checked against this generation's real size.

    A preload whose result no longer describes the selected checkpoint does not
    count. Changing checkpoint, VAE, text encoder or storage dtype in the UI
    after the last job supersedes the preloaded state, and claiming the swap
    happened would skip the VRAM budgeting for a model the host is about to load
    from scratch.
    """
    global _preload_reinstated, _preload_result

    was_reinstated, _preload_reinstated = _preload_reinstated, False

    if was_reinstated and _stale_preload():
        logger.info(
            "Model Chain: the Stage 1 preload was superseded by a checkpoint or "
            "module change; loading the current selection instead"
        )
        _preload_result = None
        return False

    return was_reinstated


def _stale_preload() -> bool:
    """Whether the last preload warmed something other than what is now selected."""
    result = _preload_result
    if result is None or not result.key:
        return False

    try:
        return result.key != _loading_parameters_key()
    except Exception:
        return False


def preload_result() -> PreloadResult | None:
    """The last preload's outcome, or None if none has run since the last job."""
    return _preload_result


def preload_disabled_reason() -> str | None:
    """Why the preload took itself out of service this session, if it did."""
    return _preload_disabled_reason


def _host_torch_context():
    """The grad-mode context the host loads models under, for use off its thread.

    ``torch.inference_mode`` is thread-local, and the host holds it for the
    whole of a generation -- so every model load it performs produces *inference
    tensors*, and the weights of a checkpoint carry that property with them.

    A background thread starts with grad enabled instead. Loading or patching
    weights there produces tensors of the other kind, and mixing the two is not
    a slow path but a hard failure: the first sampling step raises
    ``RuntimeError: Inference tensors do not track version counter``. On-the-fly
    LoRA patching makes this far more likely, because then every load rewrites
    the weights rather than just moving them.

    Entering the same context here means the preload leaves the model in the
    state the host's own loader would have left it in.
    """
    try:
        import torch

        return torch.inference_mode()
    except Exception:
        import contextlib

        return contextlib.nullcontext()


def _preload_worker(width: int, height: int, task: str = RESTORE) -> None:
    global _preload_reinstated, _preload_result, _preload_failures

    started = time.perf_counter()
    try:
        with _model_lock, _host_torch_context():
            if task == RESTORE:
                # reinstate_pending() re-reads the live selection, so a checkpoint
                # changed in the UI since the generation ended is handled correctly
                # here: either it is cached and gets swapped in, or this returns
                # False and the next generation loads it from disk as usual.
                if not reinstate_pending():
                    _preload_result = PreloadResult(
                        "nothing", detail="Stage 1 was already the loaded model"
                    )
                    return
            elif task == FROM_DISK:
                if not _load_selected_from_disk():
                    _preload_result = PreloadResult(
                        "nothing", detail="no checkpoint is selected to load"
                    )
                    return

            # Set before anything can fail. It means "weights moved since the
            # last generation budgeted for them", which is true after all three
            # tasks, and it is what tells the next generation to work its VRAM
            # budget out again rather than inherit one -- especially if the rest
            # of this went wrong.
            _preload_reinstated = True

            from modules import shared

            name = shared.opts.sd_model_checkpoint
            make_vram_room(name, current_modules(), width, height, stage=STAGE_1)
            moved = _load_current_to_gpu(width, height)

            resident, total = _loaded_residency()
            ready = total > 0 and resident >= total * READY_FRACTION
            _preload_result = PreloadResult(
                state="ready" if ready else "partial",
                key=_loading_parameters_key(),
                checkpoint=name,
                moved_bytes=moved,
                resident_bytes=resident,
                model_bytes=total,
                seconds=time.perf_counter() - started,
                detail=mc_lora.describe(mc_lora.state_of(model_data_sd_model())),
            )

            # Whatever is left over after Stage 1 is safely warm belongs to
            # Stage 2, not to the driver.
            warm_secondary(width, height)
    except Exception as exc:
        _record_preload_failure(exc)
        return

    _preload_failures = 0
    _log_preload_result(_preload_result)


def _load_selected_from_disk() -> bool:
    """Have the host load the selected checkpoint, the way a generation would.

    ``forge_model_reload`` is the host's own loader, and the only thing that
    knows which of its several load paths a given checkpoint and module set
    needs. Nothing is reimplemented here and nothing is assembled here; what
    this adds is the *timing*, which is the whole of the warm-up's value. The
    same twenty seconds are spent either way -- the question this answers is
    whether they are spent while somebody watches a progress bar sit still.

    Returns False when there is nothing to load. An installation with no
    checkpoint selected is a legitimate state, not a failure, and the warm-up
    simply has no work in it.
    """
    from modules import sd_models, shared
    from modules.sd_models import model_data

    name = getattr(shared.opts, "sd_model_checkpoint", "")
    if not name or checkpoint_info(name) is None:
        return False

    # A stale loading hash makes forge_model_reload() return early and hand
    # back whatever is loaded -- which here is nothing. Clearing it first is
    # the difference between a warm-up that loads the model and one that
    # cheerfully reports having loaded a null.
    ensure_model_loadable()
    sd_models.forge_model_reload()
    return _is_real_model(model_data.sd_model)


def model_data_sd_model():
    """The loaded ``sd_model``, or None if the host cannot be reached."""
    try:
        from modules.sd_models import model_data

        return model_data.sd_model
    except Exception:
        return None


def _record_preload_failure(exc: BaseException) -> None:
    """Leave a failed preload in a state the next generation can simply ignore.

    Three things have to be true afterwards, and none of them involve retrying:

    * the model is somewhere the host's normal synchronous path can find it.
      It is: either the swap never happened, in which case nothing moved, or it
      did, in which case Stage 1 is the loaded model with its weights in RAM --
      exactly the state a generation starts from without any of this.
    * nothing carries a claim that is no longer true. The pinned encoders are
      dropped, and the prepared LoRA state is invalidated rather than trusted,
      because a load that raised part way through is precisely the case where
      the host's belief about what is applied may have outrun what is.
    * a machine where this keeps failing stops doing it.
    """
    global _preload_failures, _preload_disabled_reason, _preload_result

    logger.warning("Model Chain: Stage 1 preload failed", exc_info=True)

    clear_pinned_encoders()
    clear_stage_2_components()
    invalidate_prepared_state("a Stage 1 preload failed part-way through")

    _preload_failures += 1
    _preload_result = PreloadResult(
        "failed", detail=str(exc) or type(exc).__name__
    )

    if _preload_failures >= PRELOAD_FAILURE_LIMIT:
        _preload_disabled_reason = (
            f"it failed {_preload_failures} times in a row"
        )
        logger.warning(
            "Model Chain: the Stage 1 preload has failed %d times in a row and is now off "
            "for the rest of this session. Generations continue normally on the host's own "
            'load path. Toggle "%s" in Settings off and on to try again.',
            _preload_failures,
            OPT_PRELOAD,
        )


def _log_preload_result(result: PreloadResult | None) -> None:
    if result is None:
        return

    if result.state == "nothing":
        logger.debug("Model Chain: nothing to preload; %s", result.detail)
        return

    if result.state == "ready":
        logger.info(
            "Model Chain: %s is in VRAM — %.1f GB resident after %.1fs (%.1f GB moved, %s). "
            "It stays there until something needs the room, so the next generation starts "
            "sampling immediately and its LoRA is applied to weights already on the card",
            result.checkpoint,
            result.resident_bytes / _GB,
            result.seconds,
            result.moved_bytes / _GB,
            result.detail,
        )
        return

    logger.info(
        "Model Chain: warmed %.1f GB of %.1f GB of %s in %.1fs — the next generation "
        "will move the rest on demand",
        result.resident_bytes / _GB,
        result.model_bytes / _GB,
        result.checkpoint,
        result.seconds,
    )


def _load_current_to_gpu(width: int = 0, height: int = 0) -> int:
    """Move the loaded checkpoint's weights onto the GPU now.

    This is exactly the work the sampler would otherwise trigger on the next
    Generate click, done through the host's own entry point so the model ends
    up in the state normal sampling produces rather than one this extension
    invented. Returns the bytes moved, as best as free VRAM can report it.

    The reserve is handed to the host rather than assumed: a load that filled
    the card would be undone moments later by the host partially unloading to
    make room for its own activations, which is the thrash this module exists
    to avoid.
    """
    from backend import memory_management
    from modules.sd_models import model_data

    patchers = model_patchers(model_data.sd_model)
    if not patchers:
        return 0

    before = free_vram_bytes()
    try:
        memory_management.load_models_gpu(patchers, memory_required=vram_headroom_bytes(width, height))
    except TypeError:
        memory_management.load_models_gpu(patchers)
    return max(before - free_vram_bytes(), 0)


STAGE_1 = "Stage 1"
STAGE_2 = "Stage 2"

_pinned_patchers: list = []
"""Stage 1's text encoder and VAE, held resident across the Stage 2 switch."""


def capture_stage_1_encoders() -> int:
    """Remember Stage 1's text encoder and VAE so the Stage 2 switch can spare them.

    Called while Model A is still the loaded model, immediately before the
    switch. Only the encoders: the UNet is far too large to keep resident
    alongside Stage 2's, and it is also the one part of Model A that Stage 2
    genuinely has no use for.

    The encoders are the opposite case. They are a small fraction of a
    checkpoint but a large fraction of the *time*, because every switch pays to
    move them: a measured Krea 2 -> Flux.2 cycle spent 3.7s on Stage 1's text
    encoder against 5.6s on its UNet. Keeping them put removes that from every
    subsequent generation.

    Re-captured each generation, so pointing Stage 1 at a different checkpoint
    leaves at most one generation's worth of stale entries -- and those are
    dropped anyway by the ``current_loaded_models`` lookup in ``_pinned_keep``.

    Returns the number of patchers captured.
    """
    global _pinned_patchers

    _pinned_patchers = []
    if not option(OPT_PIN_ENCODERS, True):
        return 0

    try:
        from modules.sd_models import model_data

        _pinned_patchers = model_patchers(model_data.sd_model, attrs=("clip", "vae"))
    except Exception:
        logger.debug("Model Chain: could not capture Stage 1's encoders", exc_info=True)
        _pinned_patchers = []

    return len(_pinned_patchers)


def clear_pinned_encoders() -> None:
    """Stop sparing Stage 1's encoders."""
    global _pinned_patchers

    _pinned_patchers = []


def _loaded_entries_for(patchers: list) -> list:
    """The host's ``LoadedModel`` records for ``patchers`` that are on the GPU now.

    ``free_memory`` compares ``keep_loaded`` against its own registry entries,
    not against patchers, so the patchers have to be looked up. Matching on
    identity avoids depending on how ``LoadedModel`` defines equality.
    """
    if not patchers:
        return []

    try:
        from backend import memory_management

        registry = list(getattr(memory_management, "current_loaded_models", []))
    except Exception:
        return []

    wanted = {id(p) for p in patchers}
    return [entry for entry in registry if id(getattr(entry, "model", None)) in wanted]


def _entry_vram_bytes(entry) -> int:
    """VRAM held by one ``LoadedModel``, across host versions."""
    for attribute in ("model_loaded_memory", "model_memory"):
        method = getattr(entry, attribute, None)
        if not callable(method):
            continue
        try:
            size = int(method())
        except Exception:
            continue
        if size > 0:
            return size

    try:
        return int(entry.model.model_size())
    except Exception:
        return 0


def _all_resident_bytes() -> int:
    """VRAM held by every model the host currently has loaded."""
    try:
        from backend import memory_management

        registry = list(getattr(memory_management, "current_loaded_models", []))
    except Exception:
        return 0

    return sum(_entry_vram_bytes(entry) for entry in registry)


def _patcher_bytes(patchers: list) -> int:
    """Total size of those patchers, resident or not."""
    total = 0
    for patcher in patchers:
        try:
            total += int(patcher.model_size())
        except Exception:
            continue
    return total


def _residency_of(patchers: list) -> tuple[int, int]:
    """How much of those patchers is on the GPU, and how much there is in total."""
    return _resident_bytes(patchers), _patcher_bytes(patchers)


def _loaded_residency() -> tuple[int, int]:
    """The same, for whatever checkpoint is loaded right now.

    This is the measurement that lets "preloaded" mean generation-ready. The
    host reports partially loaded models honestly -- ``model_loaded_memory()``
    is the share actually on the card -- so comparing it against the patchers'
    full size distinguishes a model that is ready to sample from one that will
    still be moving weights when the first step runs.
    """
    return _residency_of(model_patchers(model_data_sd_model()))


def _loaded_target_patchers(target_name: str) -> list:
    """The target's own patchers, but only if it really is the loaded model.

    Both call sites reach ``make_vram_room`` *after* the switch, so the target
    is normally ``model_data.sd_model`` already. Verifying that rather than
    assuming it means a caller that gets the order wrong falls back to the
    conservative estimate instead of sizing the budget from the wrong model.
    """
    try:
        from modules.sd_models import model_data

        loaded = model_data.sd_model
        info = checkpoint_info(target_name)
        loaded_info = getattr(loaded, "sd_checkpoint_info", None)
        if info is None or loaded_info is None or loaded_info.filename != info.filename:
            return []
        return model_patchers(loaded)
    except Exception:
        return []


def _pass_requirement(target_name: str, modules, width: int, height: int, patchers: list,
                      batch: int = 1) -> int:
    """VRAM the pass needs in total: the model, resident, plus its activations.

    When the target is the loaded model its patchers report their real size, so
    use that in preference to the disk estimate. The file size is a proxy that
    can be wrong in either direction -- quantised formats read larger than they
    land, mixed-precision builds land larger than they read -- and being wrong
    here either wastes an eviction or leaves the pass short.
    """
    size = 0
    for patcher in patchers:
        try:
            size += int(patcher.model_size())
        except Exception:
            size = 0
            break

    if size > 0:
        # The one moment a real figure for this checkpoint exists. Both stages
        # pass through here as they are switched to, so recording it is how the
        # *plan* -- built long before either is loaded -- stops having to guess
        # from a file size. See :func:`mc_plan.remember_weights`.
        _remember_measured_weights(target_name, modules, size)
    else:
        size = int(file_size_bytes(target_name, modules) * (1.0 + VRAM_MODEL_OVERHEAD_FRACTION))

    return size + _attainable_headroom(size, width, height, batch)


def _remember_measured_weights(name: str, modules, size: int) -> None:
    """Tell the plan what this checkpoint really weighs. Never raises."""
    try:
        import mc_plan

        mc_plan.remember_weights(name, modules, size)
    except Exception:
        logger.debug("Model Chain: could not record the measured weights", exc_info=True)


def measured_weight_bytes(name: str, modules=None) -> int:
    """What ``name``'s weights really occupy on the card, or 0 if it is not loaded.

    A checkpoint file's size is a proxy and it is wrong in both directions:
    quantised formats read larger than they land, and a mixed-precision build
    lands larger than it reads. The extension's own a-priori figure is the file
    plus 15%, and on one user's Krea 2 setup that came to 21.4 GB against the
    17.8 GB Forge actually loaded -- 3.6 GB of a 24 GB card reserved for
    nothing.

    That is not a cosmetic error. The language model gets what the image plan
    does not need, so 3.6 GB of phantom reserve was the whole difference
    between a placement with layers on the GPU and one with the entire model in
    system RAM, running its prompts at 67 tokens a second instead of several
    hundred.

    So when the model is loaded, it is asked rather than estimated. Zero when
    it is not the loaded model, which the caller reads as "no measurement,
    use the estimate".
    """
    patchers = _loaded_target_patchers(name)
    if not patchers:
        return 0
    size = 0
    for patcher in patchers:
        try:
            size += int(patcher.model_size())
        except Exception:
            return 0
    if size > 0:
        _remember_measured_weights(name, modules, size)
    return size


def pass_bytes_from_weights(weights: int, width: int = 0, height: int = 0,
                            batch: int = 1) -> int:
    """A pass requirement built from a known weight figure rather than a file size.

    The same arithmetic :func:`_pass_requirement` performs, exposed for the
    plan, which has a measurement and no patchers to hand. The headroom is the
    attainable one -- trimmed to what the card could actually have given -- for
    the reason given there: a reserve larger than the card is not a demanding
    target but an impossible one.
    """
    weights = max(int(weights), 0)
    if weights <= 0:
        return 0
    return weights + _attainable_headroom(weights, width, height, batch)


def _attainable_headroom(model_bytes: int, width: int, height: int, batch: int = 1) -> int:
    """The reserve, less anything the card could not have given anyway.

    A requirement larger than the whole card is not a demanding target, it is an
    impossible one, and it fails in the worst available way: ``free_memory`` is
    asked for more than exists, so it evicts *everything* it is allowed to and
    still reports a shortfall, and the pass then runs having thrown away models
    it could have kept. A real log showed a 13.9 GB model asking for 24.3 GB on
    a 24 GB card, on every single generation.

    Trimming the reserve rather than the model is the right way round: the model
    has to be resident to sample at all, whereas the reserve is a margin, and a
    margin that cannot be honoured is better spent than pretended.
    """
    headroom = vram_headroom_bytes(width, height, batch)

    total = total_vram_bytes()
    if total <= 0:
        return headroom

    return max(min(headroom, total - model_bytes), 0)


def _resident_bytes(patchers: list) -> int:
    """VRAM those patchers are already holding."""
    return sum(_entry_vram_bytes(entry) for entry in _loaded_entries_for(patchers))


def _pinned_keep(required: int, stage: str) -> tuple[list, int]:
    """Entries to spare in this eviction, and the VRAM they hold.

    Pinning is only ever applied on the way *into* Stage 2, and only when the
    incoming pass still fits with the pinned encoders in place. That condition
    is not a nicety: pinning more than the card can spare is worse than not
    pinning at all, because ``free_memory`` would fall short of its target and
    ``load_models_gpu`` would then make up the difference by partially
    unloading while it loads -- the slow path this whole function exists to
    avoid. Failing to fit therefore drops the pin rather than the target.
    """
    if stage != STAGE_2 or not _pinned_patchers:
        return [], 0

    entries = _loaded_entries_for(_pinned_patchers)
    if not entries:
        return [], 0

    pinned = sum(_entry_vram_bytes(entry) for entry in entries)
    total = total_vram_bytes()

    if total <= 0 or total - pinned < required:
        logger.info(
            "Model Chain: not pinning Stage 1's encoders — %.1f GB of VRAM less "
            "%.1f GB pinned leaves too little for the %.1f GB %s pass",
            total / _GB,
            pinned / _GB,
            required / _GB,
            stage,
        )
        return [], 0

    return entries, pinned


def make_vram_room(target_name: str, modules=None, width: int = 0, height: int = 0,
                   stage: str = STAGE_2, batch: int = 1) -> int:
    """Evict other models from VRAM if the Stage 2 pass will not otherwise fit.

    This is the "demote only under pressure" half of the residency policy, and
    the warm-swap path is where it has to be applied explicitly. A warm swap
    deliberately skips ``unload_all_models()`` so both checkpoints can stay hot
    -- but when the incoming model plus its activations do *not* fit alongside
    the outgoing one, leaving the outgoing one resident is actively harmful:
    the host only frees what each individual ``load_models_gpu`` call asks for,
    which is far less than the whole pass needs.

    Freeing here goes through ``memory_management.free_memory``, which moves
    weights to their offload device rather than discarding them, so the evicted
    model stays in this cache and switching back to it is still warm.

    What the pass needs and what has to be *freed* are different numbers
    whenever the target is already partly on the GPU -- after a preload it may
    be entirely on the GPU. Asking to free the whole requirement in that state
    is self-defeating: the target's own weights are the biggest evictable thing
    present, so the eviction throws out exactly what the next pass is about to
    load and the load happens all over again. So the target's resident bytes are
    subtracted from the requirement, and its patchers are spared from eviction.

    Returns the number of bytes freed (0 when nothing needed doing).
    """
    own = _loaded_target_patchers(target_name)
    required = _pass_requirement(target_name, modules, width, height, own, batch)
    resident = _resident_bytes(own)
    needed = max(required - resident, 0)
    free = free_vram_bytes()

    if free <= 0:
        return 0  # cannot query VRAM; leave the host's own management alone

    if free >= needed:
        logger.info(
            "Model Chain: %s needs %.1f GB, has %.1f GB free%s — no eviction needed",
            stage,
            required / _GB,
            free / _GB,
            f" and {resident / _GB:.1f} GB already resident" if resident else "",
        )
        return 0

    pinned_keep, pinned = _pinned_keep(required, stage)
    keep = _loaded_entries_for(own) + pinned_keep

    try:
        from backend import memory_management

        device = memory_management.get_torch_device()
        if keep:
            try:
                evicted = memory_management.free_memory(needed, device, keep_loaded=keep)
            except TypeError:
                # A host without keep_loaded is a reason to lose the exemptions,
                # never a reason to skip freeing -- falling through to no
                # eviction at all would be far worse than an unpinned encoder.
                logger.debug("Model Chain: free_memory does not take keep_loaded")
                keep, pinned = [], 0
                evicted = memory_management.free_memory(needed, device)
        else:
            evicted = memory_management.free_memory(needed, device)
    except Exception:
        logger.warning("Model Chain: failed to free VRAM for %s", stage, exc_info=True)
        return 0

    after = free_vram_bytes()

    if after < needed:
        # Forge's own eviction has done what it can and the pass still does not
        # fit. Only now is another workload's residency worth taking: moving an
        # image model to RAM is cheap and keeps it warm, and ending a
        # llama-server process is neither, so it is the second answer and not
        # the first (sections 8 and 9).
        #
        # Reaching here at all is a reserve miss. The plan said this phase would
        # fit inside the protected image budget and it did not, so it is filed
        # as such before anything is taken -- emergency eviction is recovery,
        # not scheduling, and a recovery nobody is told about is indistinguishable
        # from the policy working.
        shortfall = needed - after
        held = _llm_residency_bytes()
        foreign = _reclaim_foreign(shortfall, f"the {stage} pass")
        if foreign:
            after = free_vram_bytes()
        _record_reserve_miss(stage, shortfall, held, evicted=bool(foreign))

    freed = max(after - free, 0)
    names = [type(getattr(m, "model", m)).__name__ for m in (evicted or [])]

    logger.info(
        "Model Chain: freed %.1f GB VRAM for %s (%.1f GB -> %.1f GB free, %.1f GB of %.1f GB "
        "still to load)%s%s",
        freed / _GB,
        stage,
        free / _GB,
        after / _GB,
        needed / _GB,
        required / _GB,
        f"; offloaded {', '.join(names)}" if names else "",
        f"; kept {pinned / _GB:.1f} GB of Stage 1 encoders resident" if pinned_keep else "",
    )

    if after < needed:
        total = total_vram_bytes()
        if 0 < total < required:
            # Not a shortfall to act on -- the pass is larger than the card. The
            # advice below would be misleading here, because no amount of
            # evicting reaches a target that does not fit in the first place.
            logger.warning(
                "Model Chain: the %s pass wants %.1f GB but the card holds %.1f GB, so part "
                "of it will run from system memory however much is evicted. A more heavily "
                "quantised model for this stage is the only thing that changes that.",
                stage,
                required / _GB,
                total / _GB,
            )
        else:
            logger.warning(
                "Model Chain: still only %.1f GB VRAM free against %.1f GB left to load for a "
                "%dx%d pass. Expect the driver to spill into system memory, which is "
                "very slow. A smaller Stage 2 size multiplier, a lower Hires. fix "
                "upscale, or a more heavily quantised Stage 2 model would each help.",
                after / _GB,
                needed / _GB,
                width,
                height,
            )

    return freed


# --------------------------------------------------------------------------- #
# Speed-first warming of what is left (section 4.2)
# --------------------------------------------------------------------------- #
#
# Stage 1 comes first and always: it is what the next Generate click needs, and
# nothing else may be warmed at its expense. But a 24 GB card running a pair of
# quantised models is routinely left with several GB doing nothing once Stage 1
# is back, and that spare capacity has an obvious use -- Stage 2's components,
# which the *following* generation will want and which are otherwise moved back
# across PCIe from scratch.
#
# The rule that makes this safe rather than greedy is that speculative warming
# never touches the reserve. Free VRAM is not the budget; free VRAM minus the
# reserve is, and the reserve already accounts for activations, the user's
# manual floor and anything Forge set aside for itself. Under pressure the warm
# components are simply the first thing make_vram_room() evicts -- they are not
# in any keep list -- so the next large pass reclaims them without ceremony.

_stage_2_patchers: list = []
"""Stage 2's patchers, captured while it is still the loaded model."""

WARM_MARGIN_BYTES = int(0.5 * _GB)
"""Slack a component must fit inside *on top of* the reserve to be warmed.

Warming right up to the boundary would make the reserve a number that is
technically respected and practically gone, and the next thing to allocate would
push the driver into system memory. Speculative work should only happen when
there is room to be wrong about it.
"""


def _llm_residency_bytes() -> int:
    """What the language model is holding on the card right now, or 0.

    Asked before the eviction rather than after it, because after it the answer
    is zero for every miss and the panel could not tell a model that gave up
    five gigabytes from one that was never on the card.
    """
    try:
        import mc_broker

        # On the image card, because that is the card the pass was short on.
        # A reserve miss filed with a second card's llama-server in it would
        # tell the panel -- and Auto's learned cap -- that nineteen gigabytes
        # of language model were in the way of a pass that could never have
        # reached them (design intent T19).
        index = mc_broker.image_device_index()
        return max(int(mc_broker.reported_bytes(
            mc_broker.FAMILY_LLM,
            card=index if index >= 0 else mc_broker.ANY_CARD)), 0)
    except Exception:
        return 0


def llm_vram_on_the_image_card() -> int:
    """Bytes of language model held on the card an image pass would run on.

    Public because the orchestration has to ask something this module had only
    ever answered privately: whether there is anything on the far side of the
    reclaim hook worth reaching for.

    That hook is reached from :func:`make_vram_room` and from nowhere else, and
    ``make_vram_room`` was reached for Stage 1 only after a swap -- so a plan
    with no Stage 2 had no path to it at all. llama-server could sit in the
    VRAM Stage 1 was about to fail to fit into, and nothing in the extension
    was in a position to ask for it back. Forge's own eviction cannot: those
    bytes belong to another process, and ``free_memory`` has never been able to
    see them.

    Zero is the ordinary answer and the cheap one -- no language model on this
    card, nothing to reclaim, and the host's own management is the whole of the
    right behaviour.
    """
    return _llm_residency_bytes()


def _record_reserve_miss(stage: str, shortfall: int, llm_bytes: int,
                         evicted: bool = False) -> None:
    """File the miss with the plan, so the panel can show it and Auto can learn.

    Never raises. A generation that is already short of VRAM must not also fail
    because the bookkeeping did.
    """
    try:
        import mc_plan

        mc_plan.record_miss(stage, shortfall, llm_bytes=llm_bytes, evicted=evicted)
    except Exception:
        logger.debug("Model Chain: could not record the reserve miss", exc_info=True)


def _warming_wanted() -> bool:
    """Whether anything is going to start a preload, and so consume a capture.

    Two settings can, and they ask different questions: the preload setting
    permits a background thread after a generation, and the warm-up setting
    asks for the pipeline to be loaded before somebody waits on it. Either one
    produces a preload thread, so either one is a reason to hold on to what a
    preload would warm.

    Read through :mod:`mc_arm` rather than the option directly, because that
    module owns the mapping from the stored label to a mode, and never at
    import time, because this module is the one :mod:`mc_arm` reaches into.
    """
    if option(OPT_PRELOAD, PRELOAD_DEFAULT):
        return True
    try:
        import mc_arm

        return mc_arm.mode() != mc_arm.WARM_OFF
    except Exception:
        return False


def capture_stage_2_components() -> int:
    """Remember Stage 2's patchers before the swap back to Stage 1 hides them.

    Called as the swap happens, for the same reason the encoders are captured
    before the switch into Stage 2: once the pointer moves,
    ``model_data.sd_model`` describes the other stage and these are unreachable.

    Nothing is captured when nothing can consume it. These are references to
    real weights, so holding them after the cache has dropped the model would
    keep gigabytes alive for a warm-up that is never going to run.
    """
    global _stage_2_patchers

    _stage_2_patchers = []
    if not option(OPT_WARM_STAGE_2, True):
        return 0
    if not _warming_wanted():
        # Warming only ever happens on the preload's thread; see the note above
        # warm_secondary. With nothing that would start one there is no consumer.
        return 0

    try:
        _stage_2_patchers = model_patchers(model_data_sd_model())
    except Exception:
        logger.debug("Model Chain: could not capture Stage 2's components", exc_info=True)
        _stage_2_patchers = []

    return len(_stage_2_patchers)


def clear_stage_2_components() -> None:
    """Stop trying to keep Stage 2 warm."""
    global _stage_2_patchers

    _stage_2_patchers = []


def warm_secondary(width: int = 0, height: int = 0) -> int:
    """Spend VRAM left over after Stage 1 on keeping Stage 2 warm.

    Returns the bytes moved, which is 0 whenever there was nothing to spare --
    the common case on a card that is tight for the pair, and not a failure.

    Components are dropped largest-first until the rest fit, so a card with a
    few GB spare keeps Stage 2's text encoder and VAE (small, and a
    disproportionate share of the switch's cost) rather than nothing at all
    because its UNet did not fit.
    """
    patchers = [p for p in _stage_2_patchers if p]
    if not patchers:
        return 0

    if not option(OPT_WARM_STAGE_2, True):
        return 0

    reserve = vram_headroom_bytes(width, height)
    free = free_vram_bytes()
    if free <= 0:
        return 0

    spare = free - reserve - WARM_MARGIN_BYTES
    if spare <= 0:
        logger.info(
            "Model Chain: no VRAM to spare for Stage 2 after Stage 1 (%.1f GB free, "
            "%.1f GB reserved) — it stays in system RAM",
            free / _GB,
            reserve / _GB,
        )
        return 0

    chosen = list(patchers)
    while chosen and _pending_bytes(chosen) > spare:
        largest = max(chosen, key=lambda p: _patcher_bytes([p]))
        chosen = [p for p in chosen if p is not largest]

    if not chosen:
        logger.info(
            "Model Chain: Stage 2 does not fit in the %.1f GB spare beyond the reserve — "
            "it stays in system RAM",
            spare / _GB,
        )
        return 0

    before = free
    try:
        from backend import memory_management

        try:
            memory_management.load_models_gpu(chosen, memory_required=reserve)
        except TypeError:
            # A host without memory_required still honours its own reserve; ours
            # is then advisory rather than enforced, which is why the components
            # were sized to fit before the call rather than during it.
            memory_management.load_models_gpu(chosen)
    except Exception:
        logger.debug("Model Chain: could not warm Stage 2's components", exc_info=True)
        return 0

    moved = max(before - free_vram_bytes(), 0)
    logger.info(
        "Model Chain: kept %d of Stage 2's %d components warm in VRAM (%.1f GB moved, "
        "%.1f GB free, %.1f GB reserved) — a Stage 2 retry starts sooner",
        len(chosen),
        len(patchers),
        moved / _GB,
        free_vram_bytes() / _GB,
        reserve / _GB,
    )
    return moved


def _pending_bytes(patchers: list) -> int:
    """What warming those patchers would still have to move."""
    resident, total = _residency_of(patchers)
    return max(total - resident, 0)


# Warming happens on the preload's thread and nowhere else, and the reason is
# not tidiness. Loading weights *in* rewrites and re-patches them, and doing
# that outside the torch context the host loads under leaves the model in a
# state the next sampling step rejects with "Inference tensors do not track
# version counter" -- the same failure this module's preload notes describe.
#
# The preload has a deliberate answer to that (_host_torch_context), an opt-in
# switch, and a circuit breaker. A hook on the generation thread has none of
# the three, so an attempt to warm from before_process was a way of running the
# riskiest operation here on the path that had opted out of the risk. Freeing
# is different and stays where it is: free_memory moves weights *out*, which is
# what the host does on every eviction anyway.


# --------------------------------------------------------------------------- #
# Readiness reporting
# --------------------------------------------------------------------------- #

WARM = "warm"
PARTIAL = "partially warm"
COLD = "cold"


def stage_1_readiness() -> tuple[str, str]:
    """How ready Stage 1 is for the generation about to start.

    Returns the state -- ``warm``, ``partially warm`` or ``cold`` -- and a line
    saying so in a form worth putting in the console. Deliberately measured
    against the host's live view rather than reported from what the preload
    believed it achieved: the preload's own record is used only to explain the
    measurement, never to stand in for it.
    """
    result = _preload_result
    because = _readiness_explanation(result)

    model = model_data_sd_model()
    if not _is_real_model(model):
        return COLD, f"Model Chain: Stage 1 will load from disk{because}."

    try:
        selection, loaded = _loading_parameters_key(), _loaded_model_key()
    except Exception:
        selection = loaded = ""

    if selection and loaded and selection != loaded:
        return COLD, (
            f"Model Chain: Stage 1 is a cold load — the selected checkpoint is not "
            f"the loaded one{because}."
        )

    resident, total = _loaded_residency()
    if total <= 0:
        return COLD, f"Model Chain: Stage 1 residency is unknown{because}."

    if resident >= total * READY_FRACTION:
        return WARM, (
            f"Model Chain: Stage 1 is warm — {total / _GB:.1f} GB already in VRAM"
            f"{because}."
        )

    if resident > 0:
        return PARTIAL, (
            f"Model Chain: Stage 1 is partially warm — {resident / _GB:.1f} GB of "
            f"{total / _GB:.1f} GB in VRAM, the rest moves on demand{because}."
        )

    return COLD, (
        f"Model Chain: Stage 1 is cold — {total / _GB:.1f} GB still to move from "
        f"system RAM{because}."
    )


def _readiness_explanation(result: PreloadResult | None) -> str:
    """The ", because ..." half of the readiness line."""
    retired = preload_disabled_reason()
    if retired is not None:
        return f" (the preload is off for this session: {retired})"

    if result is None:
        return "" if _warming_wanted() else " (no warm-up is enabled)"

    if result.state == "failed":
        return f" (the preload failed: {result.detail})"
    if result.state == "nothing":
        return ""
    if _stale_preload():
        return " (the preload was superseded by a checkpoint or module change)"
    return f" (preloaded in {result.seconds:.1f}s)"


def clear_references() -> None:
    """Drop reference latents held by the loaded model.

    Edit-capable engines build their reference list from the engine's own
    ``ref_latents`` (populated by ImageStitch) plus the img2img input. Stage 2
    runs with scripts disabled, so ImageStitch never gets to refresh or clear
    that list -- and because a cached model keeps its engine object across
    generations, entries from an earlier run would otherwise be silently mixed
    into every refined image.
    """
    try:
        from backend.args import dynamic_args
        from modules import shared

        model = shared.sd_model
        if hasattr(model, "clear_references"):
            model.clear_references()
        if getattr(model, "ini_latent", None) is not None:
            model.ini_latent = None
        dynamic_args.ref_latents.clear()
    except Exception:
        logger.warning("Model Chain: failed to clear stale references", exc_info=True)


_foreign_reclaim = None
"""Optional callback that frees VRAM held by *another* workload family.

Installed by ``mc_broker`` at import; ``None`` on an installation that never
loads it, and on every test of this module in isolation. That direction of
dependency is the point: this module is about image residency and stays
importable, testable and correct without knowing an LLM exists. What it gains
from the hook is one extra place to look when its own eviction has fallen
short -- and what it must never gain is an opinion about when the *other*
family should give ground, which is the broker's to hold.

Signature: ``reclaim(needed_bytes: int, reason: str) -> int`` (bytes freed).
"""


def set_foreign_reclaim(callback) -> None:
    """Register (or clear, with ``None``) the cross-workload reclaim hook."""
    global _foreign_reclaim

    _foreign_reclaim = callback


def _reclaim_foreign(needed: int, reason: str) -> int:
    """Ask another workload family for ``needed`` bytes. Never raises.

    A hook that fails is a reason to carry on with less VRAM than hoped -- the
    driver spills into system memory and the pass is slow -- and never a reason
    to fail the generation that called it.
    """
    if _foreign_reclaim is None or needed <= 0:
        return 0
    try:
        return int(_foreign_reclaim(needed, reason) or 0)
    except Exception:
        logger.warning("Model Chain: cross-workload reclaim failed", exc_info=True)
        return 0


# --------------------------------------------------------------------------- #
# The warm host-RAM tier, as the broker sees it (design intent section 10.2)
# --------------------------------------------------------------------------- #
#
# Three functions, deliberately narrow. The cross-workload broker needs to know
# how much system RAM this cache is holding, how much of it is safe to drop,
# and how to ask for some back -- and nothing else. Cache identity, module
# pairing, LRU order, prepared-state preservation and which incoming entry is
# protected all stay here, where the Forge entry points and the bookkeeping
# are; the broker is not given a way to reach a model object, because a broker
# that could reach one would eventually be written as though it understood one.


def warm_ram_bytes() -> int:
    """System RAM this cache is explicitly holding.

    Explicitly: cache entries, not "everything Forge's process has touched".
    Weights the current pass is executing against, arena fragmentation and the
    page cache are all real host memory and none of them is this number, which
    is the only one this module can honestly offer to give back.
    """
    return int(_cache.total_bytes())


def reclaimable_warm_ram_bytes() -> int:
    """Of :func:`warm_ram_bytes`, what could be released right now.

    Everything except the entry describing the checkpoint that is loaded. That
    one is the model the host is generating with; its weights may be partly on
    the card and partly offloaded here, and dropping this module's reference to
    it while a pass is running would be releasing memory the active workload
    is using -- section 10.10, which is explicit that active image host memory
    is not a generic victim for an LLM's admission arithmetic.
    """
    loaded = _loaded_model_key()
    return int(sum(entry.size_bytes for key, entry in _cache._entries.items()
                   if key != loaded))


def release_warm_ram(needed_bytes: int, reason: str = "") -> int:
    """Drop least-recently-used cache entries until ``needed_bytes`` is covered.

    Returns bytes given up, measured as the cache's own reduction. The caller
    re-reads available system RAM afterwards and believes that instead
    (invariant I-15): what is dropped here is a *reference*, and the memory
    comes back when the collector agrees, which is usually immediately and is
    not this function's to promise.

    The loaded checkpoint is never a candidate, for the reason
    :func:`reclaimable_warm_ram_bytes` gives. Nothing else is protected: the
    whole purpose of these entries is to make a future switch faster, and a
    future switch is exactly what should be given up before an active workload
    is refused the memory it needs to run at all.
    """
    wanted = max(int(needed_bytes), 0)
    if wanted <= 0:
        return 0
    loaded = _loaded_model_key()
    candidates = sorted(
        ((key, entry) for key, entry in _cache._entries.items() if key != loaded),
        key=lambda pair: pair[1].last_used,
    )
    freed = 0
    dropped: list[str] = []
    for key, entry in candidates:
        if freed >= wanted:
            break
        _cache.drop(key)
        freed += entry.size_bytes
        dropped.append(entry.checkpoint_name)
    if not dropped:
        return 0
    logger.info(
        "Model Chain: released %.1f GB of warm image RAM cache for %s — %s; %.1f GB of "
        "cache remains and those checkpoints will reload from disk on the next switch",
        freed / _GB, reason or "another workload", ", ".join(dropped),
        _cache.total_bytes() / _GB,
    )
    return int(freed)


def resident_vram_bytes() -> int:
    """VRAM the host's image models are holding right now.

    Deliberately not ``loaded_size_bytes(shared.sd_model)``, which sums each
    patcher's ``model_size()`` and is therefore the model's *total* footprint
    wherever it happens to live. A checkpoint that has been offloaded to system
    RAM still answers that question with gigabytes, so a caller asking "is
    there anything of yours on the card" would be told yes forever.

    This asks the host's own loaded-model registry instead, which is where the
    per-device figure lives.
    """
    return _all_resident_bytes()


def release_vram(needed_bytes: int, reason: str = "") -> int:
    """Move image weights out of VRAM until ``needed_bytes`` is free.

    The cross-workload half of ``make_vram_room``. That function knows which
    Stage 2 pass is coming and sizes the requirement from it; this one is
    called by the residency broker on behalf of a workload this module knows
    nothing about -- an LLM about to load -- and is handed the number instead.

    Everything else is deliberately identical, because the properties that make
    the image side safe are all in the mechanism rather than the caller:

    * ``free_memory`` moves weights to their offload device rather than
      discarding them, so an image model demoted for an LLM stays in this
      module's RAM cache and switching back to it is still a warm swap. That is
      section 7.3's "prefer RAM demotion over destruction", and it costs nothing
      extra to honour because it is what the host's own path already does.
    * the currently loaded model's patchers are *not* spared. A caller asking
      for room on another workload's behalf is asking for exactly that, and the
      loaded checkpoint is usually the only thing on the card large enough to
      answer with.
    * pinned Stage 1 encoders are spared while they fit, through the same
      ``_pinned_keep`` the image path uses, so a cross-workload reclaim cannot
      quietly undo a pin the user asked for.

    Returns the bytes actually freed, which the broker reports and logs.
    """
    needed = max(int(needed_bytes), 0)
    if needed <= 0:
        return 0

    before = free_vram_bytes()
    if before <= 0:
        return 0  # cannot query VRAM; leave the host's own management alone
    if before >= needed:
        return 0

    keep, pinned = _pinned_keep(needed, STAGE_2)

    with _model_lock:
        try:
            from backend import memory_management

            device = memory_management.get_torch_device()
            if keep:
                try:
                    memory_management.free_memory(needed, device, keep_loaded=keep)
                except TypeError:
                    keep, pinned = [], 0
                    memory_management.free_memory(needed, device)
            else:
                memory_management.free_memory(needed, device)
        except Exception:
            logger.warning("Model Chain: failed to free VRAM for %s", reason or "another workload",
                           exc_info=True)
            return 0

    freed = max(free_vram_bytes() - before, 0)
    logger.info(
        "Model Chain: released %.1f GB of image VRAM for %s (%.1f GB -> %.1f GB free)%s",
        freed / _GB,
        reason or "another workload",
        before / _GB,
        free_vram_bytes() / _GB,
        f"; kept {pinned / _GB:.1f} GB of pinned encoders resident" if keep else "",
    )
    return freed


def get_model(name: str):
    """Return the cached ``sd_model`` for ``name``, or None if it is not resident."""
    entry = _cache.get(_target_key_for(name))
    return entry.sd_model if entry is not None else None


def cached_names() -> list[str]:
    return _cache.names()


def release_all() -> None:
    """Drop every cached model. Next use of any of them reloads from disk."""
    global _pending_restore, _preload_reinstated, _preload_result

    join_preload()

    _pending_restore = None
    _preload_reinstated = False
    _preload_result = None
    clear_pinned_encoders()
    clear_stage_2_components()
    _cache.clear()

    try:
        import gc

        from backend import memory_management

        gc.collect()
        memory_management.soft_empty_cache()
    except Exception:
        pass
