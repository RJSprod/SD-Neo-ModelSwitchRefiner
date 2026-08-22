"""The llama.cpp runtime, placed by the residency broker.

Section 6 says keep llama.cpp in its own process, and that is what the vendored
``prompt_master.inference`` package already does. This module is the layer
between that process and ModelSwitch's memory policy. Its whole job is to
answer one question before every load:

    given what is on the card right now, and what the user asked for, where
    does this model actually go?

and then to say out loud what it decided, because section 13 forbids quietly
reducing context or quality-critical settings without reporting the change.

Why a separate process is the right shape here
----------------------------------------------
It is an ownership boundary, not an implementation detail. llama.cpp allocates
CUDA memory that PyTorch does not know about and cannot move, so the only
reliable way to hand VRAM back to Forge is to end the process that holds it.
That is a crude mechanism, and it is also an *honest* one: when this module
tells the broker it freed 14 GB, the 14 GB is genuinely gone rather than
cached somewhere by an allocator.

The warm layer, for a model in another process
----------------------------------------------
Section 7.2 asks that the intent of a warm tier survive even where the
mechanism differs, and for llama.cpp it differs completely: there is no
"move these weights to system RAM" call to make. What there is instead is the
operating system's page cache. llama.cpp reads a GGUF through ``mmap``, so
after one load the file's pages are resident in system RAM, and stopping the
server does not evict them. A restart then reads from RAM at memory bandwidth
rather than from disk -- which is the warm tier, arrived at by a different
road. Nothing here has to do anything to get it except *not* do the one thing
that would lose it, which is to thrash the machine's RAM so hard the kernel
drops the pages. That is why :func:`release` prefers stopping the server to
reloading it somewhere else.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mc_broker
import mc_gguf
import mc_llm_context
import mc_llm_paths
import mc_llm_state

if TYPE_CHECKING:
    from prompt_master.models.managed_profiles import ManagedProfile

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3
_MB = 1024**2

OPT_RELEASE = "model_chain_llm_release"

RELEASE_STOP = "stop"
RELEASE_SYSTEM_RAM = "system_ram"

RELEASE_MODES = (
    (RELEASE_STOP, "Stop llama-server — releases all VRAM; weights stay warm in the page cache"),
    (RELEASE_SYSTEM_RAM, "Keep it running in system RAM — slower inference, no reload"),
)

RELEASE_LOCK_TIMEOUT = 30.0
"""Seconds :meth:`Runtime.release` waits for the runtime lock before giving up.

Long enough to outlast anything but a model load, short enough that a wedged
one delays a generation rather than joining it. See the note in ``release``.
"""

MINIMUM_CONTEXT = 2048
"""The floor placement negotiation will shrink context to.

Below this the LLM is not usefully a chat model any more -- a system prompt and
one image already spend a good part of it -- so a placement that would need to
go lower is reported as not fitting rather than silently made useless.
"""


class NotConfigured(RuntimeError):
    """No model has been chosen yet. The panel offers setup rather than an error."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """What the installation is set up to run, from the state file and prefs."""

    runtime: Path | None
    model: Path | None
    mmproj: Path | None
    gpu_index: int
    device: str
    gpu_layers: str
    context_size: int
    context_mode: str
    context_buffer_gb: float
    kv_type_k: str
    kv_type_v: str
    quantization: str = ""
    device_name: str = ""
    mode: str = "gpu"
    source: str = "manual"
    """``"managed"`` when the model came from the catalogue, ``"manual"`` when
    the user pointed at a GGUF themselves. See :mod:`mc_llm_managed_models`."""
    managed_id: str = ""
    profile: "ManagedProfile | None" = None
    """The hidden quality profile in force, or ``None`` on a manual install.

    Not a settings object: everything in it was decided when the catalogue
    entry was written, and it is here rather than in the preferences file
    because it belongs to *the model* rather than to the installation. A
    manual install has none, which is what leaves the Settings page in charge
    of context and cache types exactly as it was before the catalogue existed.
    """

    @property
    def profile_id(self) -> str:
        return getattr(self.profile, "profile_id", "") if self.profile is not None else ""

    def __post_init__(self) -> None:
        """The mode has the last word on where the weights go.

        Mixed mode *is* "no resident layers": the card is named on ``--device``
        and given the work llama.cpp can hand it, and the weights stay in
        system RAM. A state file that says ``mixed`` and also carries a layer
        count is not describing two settings to be reconciled -- it is
        describing one setting written twice, once wrongly, and the mode is the
        half the user chose in the menu. Settling it here, once, is what stops
        a mixed install from being started as a full offload by whichever
        reader did not think to look at the mode: the layer count every one of
        them reads is already the right one.
        """
        from prompt_master.core.models import CPU_MODE, MIXED_MODE
        from prompt_master.inference.device_detection import NO_OFFLOAD

        if (str(self.mode).strip().casefold() in (MIXED_MODE, CPU_MODE)
                and str(self.gpu_layers) != NO_OFFLOAD):
            object.__setattr__(self, "gpu_layers", NO_OFFLOAD)

    @property
    def on_gpu(self) -> bool:
        from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD

        return self.device.casefold() != CPU_DEVICE and str(self.gpu_layers) != NO_OFFLOAD

    @property
    def configured(self) -> bool:
        return self.model is not None and self.runtime is not None

    @property
    def sees(self) -> bool:
        return self.mmproj is not None


def config() -> Config:
    """Current runtime configuration. Never raises -- an unconfigured install
    is a state the panel renders, not an exception it handles.

    Read in the usual three layers -- the state file, the preferences file, the
    Settings page -- and then, for a managed backbone only, a fourth that wins:
    the hidden profile that came with the catalogue entry. That precedence is
    the feature. Somebody who chose "Gemma 4 12B QAT Balanced" chose a context
    size, two cache types and a chat-template flag along with it, whether or
    not they know that; leaving the Settings page authoritative over those
    would mean a curated backbone ran at whatever the *previous* model happened
    to need, which is the failure this catalogue exists to prevent.

    It reaches exactly as far as the profile's own fields. The device, the
    offload, the residency policy and everything else about this machine are
    untouched -- a profile that pinned a GPU index would be a checked-in file
    making a claim about somebody else's hardware.
    """
    from prompt_master.core.config import read_json

    paths = mc_llm_paths.app_paths()
    try:
        state = read_json(paths.state_file)
    except (OSError, ValueError):
        state = {}
    prefs = mc_llm_state.preferences()
    source, managed_id, profile = _managed(state)

    def located(key):
        recorded = str(state.get(key) or "")
        if not recorded:
            return None
        try:
            return paths.locate(recorded)
        except ValueError:
            return None

    runtime = None
    recorded_runtime = str(state.get("runtime") or "")
    if recorded_runtime:
        try:
            runtime = paths.contained(recorded_runtime)
        except ValueError:
            runtime = None

    context_size = int(prefs.get("context_size") or state.get("context_size", 8192) or 8192)
    context_mode = str(prefs.get("context_mode", "auto"))
    kv_type_k = str(prefs.get("kv_type_k", "f16"))
    kv_type_v = str(prefs.get("kv_type_v", "f16"))
    if profile is not None:
        # "fixed" and not "auto": automatic sizing spends whatever VRAM is free
        # on context, and a managed profile's context size is a decision about
        # the workload rather than a number to grow into. Negotiation can still
        # shrink it to make the model fit -- that is a report, not a setting.
        context_size, context_mode = int(profile.context), "fixed"
        kv_type_k, kv_type_v = str(profile.kv_type_k), str(profile.kv_type_v)

    return Config(
        runtime=runtime,
        model=located("model"),
        mmproj=located("mmproj"),
        gpu_index=int(state.get("gpu_index", 0) or 0),
        device=str(state.get("gpu_device", "CUDA0")),
        gpu_layers=str(state.get("gpu_layers", "all")),
        context_size=context_size,
        context_mode=context_mode,
        context_buffer_gb=float(prefs.get("context_buffer_gb", 4.0) or 0.0),
        kv_type_k=kv_type_k,
        kv_type_v=kv_type_v,
        quantization=str(state.get("quantization", "")),
        device_name=str(state.get("gpu_device_name", state.get("gpu_name", ""))),
        mode=str(state.get("mode", "gpu")),
        source=source,
        managed_id=managed_id,
        profile=profile,
    )


def _managed(state: dict) -> tuple:
    """``(source, id, profile)`` for ``state``, and manual defaults on any failure.

    Total, like everything else :func:`config` calls. A registry that will not
    load, a profile this build has never heard of, or a catalogue module that
    cannot be imported at all are every one of them a reason to run the
    recorded GGUF with the installation's own settings and say so in the log --
    never a reason for the panel to fail to draw.
    """
    try:
        import mc_llm_managed_models

        chosen = mc_llm_managed_models.selection(state)
        return chosen.source, chosen.identifier, mc_llm_managed_models.active_profile(chosen)
    except Exception:
        logger.debug("Model Chain: could not read the managed backbone selection",
                     exc_info=True)
        return "manual", "", None


def _release_mode() -> str:
    return mc_broker.resolve(mc_broker.option(OPT_RELEASE, RELEASE_STOP),
                             RELEASE_MODES, RELEASE_STOP)


# --------------------------------------------------------------------------- #
# Placement negotiation (sections 11 and 13)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Negotiation:
    """A placement, and every change made to reach it.

    ``notes`` is not decoration. Section 13: "The app must not quietly reduce
    context or quality-critical settings without reporting what it changed."
    Every branch below that lowers something appends a sentence here, and the
    UI prints them.
    """

    placement: mc_llm_context.Placement
    estimate: mc_llm_context.Estimate
    notes: tuple[str, ...] = ()
    fits: bool = True

    @property
    def degraded(self) -> bool:
        return bool(self.notes)


def projector_bytes(configuration: Config, vision: bool) -> int:
    """What the vision projector will cost the card, or 0 when it is not loaded.

    Read off the file rather than guessed at, plus a quarter for the encoder's
    own working memory -- llama.cpp announces its own worst case for the same
    thing, and for a 26B model's f16 projector that came to 1.3 GB. A
    gigabyte and a third is not a rounding error on a card that is already
    within two of its limit: unaccounted for, it is a gigabyte and a third that
    llama.cpp has to find somewhere, and where it finds it is by leaving part
    of the model in system RAM.
    """
    if not vision or configuration.mmproj is None:
        return 0
    try:
        return int(Path(configuration.mmproj).stat().st_size * 1.25)
    except OSError:
        logger.debug("Model Chain: could not size the vision projector", exc_info=True)
        return 0


def negotiate(configuration: Config | None = None,
              gguf: mc_gguf.Gguf | None = None, *, reclaim: bool = True,
              already_ours: int = 0, extra_reserve: int = 0,
              vision: bool = False,
              expert_floor: int = mc_llm_context.NO_EXPERTS) -> Negotiation:
    """Decide where the LLM goes, given what is on the card right now.

    ``reclaim=False`` is what the estimator panel passes, and it now makes no
    difference to the answer, because no path through this function moves
    anything: a negotiation that cannot fit shrinks the language model rather
    than evicting a checkpoint. The parameter is kept because callers pass it
    and because "this preview will not touch the card" is worth being able to
    say at the call site -- and it is now true of every call, which is a
    stronger guarantee than the one it used to name.

    ``extra_reserve`` is headroom on top of the global safety margin, and the
    only thing that ever passes it is a start that has already failed: the card
    said it had room, the driver disagreed, and the arithmetic here has no way
    to know that except by being told. See :meth:`Runtime.client`.

    ``already_ours`` is VRAM a llama-server this module started is holding at
    this moment. It is *free* for the purposes of every decision below, because
    a re-placement stops that server before it starts the next one -- and a
    negotiation that leaves it out reads its own footprint as somebody else's
    and places the next server in the gap it left. That is not a rounding
    error: a model resident in 17 GB negotiated against the 5 GB left beside it
    demotes itself to two layers on the card and runs the rest from system RAM,
    on a card that was holding all of it a second earlier. See
    :func:`_free_vram`.

    The ladder, in order, and every rung of it is the LLM giving something up.
    Nothing here asks the image side for anything, in either residency mode or
    for any placement -- see :func:`mc_broker._victim_order`. The VRAM this
    function is spending is the VRAM the image family is not using, and when
    there is none of it the answer is system RAM, not an eviction:

    1. Ask for what the user configured. If it fits in what is already free,
       stop -- nothing moves, which is section 8's rule and the common case
       once the checkpoint and the model are both small enough.
    2. Shrink the context. A cache nobody has filled yet is the cheapest thing
       on the card to give up.
    3. For a mixture-of-experts model, move the experts of as few blocks as
       will cover the shortfall into system RAM. They are most of the weights
       and are consulted two at a time, so this buys back the most VRAM for the
       least speed -- and moving six blocks' worth costs a sixth of what moving
       every block's worth costs.
    4. When that reaches every block anyway, move every expert.
    5. Drop blocks, four at a time, until what is left fits.
    6. Land on zero, which is the whole model in system RAM. Slow, and still an
       answer rather than a failure -- and still faster than the alternative it
       replaced, which was evicting a checkpoint and paying to move it back.

    ``expert_floor`` is a lower bound on rung 3, carried in by a caller whose
    last real start ran out of memory at a smaller one. See
    :data:`EXPERT_RETRY_STEP`.
    """
    configuration = configuration or config()
    if not configuration.configured:
        raise NotConfigured(
            "No local model is configured yet. Choose a GGUF in LLM Studio’s Setup mode."
        )

    described = gguf if gguf is not None else mc_gguf.describe(configuration.model)
    wanted = _requested_placement(configuration, described, already_ours)
    notes: list[str] = []

    if not wanted.on_gpu:
        # System RAM or CPU execution: there is no VRAM decision to make, and
        # pretending to make one would report movements that never happened.
        return Negotiation(wanted, mc_llm_context.estimate(configuration.model, wanted, described),
                           (), True)

    reserve = (mc_broker.safety_margin_bytes() + max(int(extra_reserve), 0)
               + projector_bytes(configuration, vision))
    estimate = mc_llm_context.estimate(configuration.model, wanted, described)

    if _fits(estimate, reserve, already_ours):
        return Negotiation(wanted, estimate, (), True)

    placement = wanted
    placement, estimate, shrunk = _shrink_context(configuration, placement, described, reserve,
                                                  already_ours)
    if shrunk:
        notes.append(shrunk)
    if _fits(estimate, reserve, already_ours):
        return Negotiation(placement, estimate, tuple(notes), True)

    placement, estimate, offloaded = _shrink_offload(configuration, placement, described, reserve,
                                                     already_ours, expert_floor)
    if offloaded:
        notes.append(offloaded)

    return Negotiation(placement, estimate, tuple(notes), _fits(estimate, reserve, already_ours))


def _requested_placement(configuration: Config, gguf: mc_gguf.Gguf | None,
                         already_ours: int = 0) -> mc_llm_context.Placement:
    """The placement the user's settings ask for, before any negotiation.

    Mixed asks for the whole model on the card and lets the ladder below take
    it apart. That is a change: Mixed used to be recorded as zero layers and
    pinned there, so a machine with a 3090 in it ran every matrix multiply on
    the processor while the card sat idle -- which is what the mode's own
    description promised it would not do. What it now means is "use the room
    that is genuinely spare, and never ask for any that is not" -- which is
    what *every* placement means now: :func:`negotiate` shrinks the context,
    then moves the experts out, then drops blocks, and lands on zero -- exactly
    Mixed's old behaviour -- when nothing is free.
    """
    from prompt_master.inference.device_detection import NO_OFFLOAD

    layers = mc_llm_context.ALL_LAYERS
    recorded = str(configuration.gpu_layers).strip().casefold()
    if recorded == NO_OFFLOAD:
        layers = mc_llm_context.NO_LAYERS
    elif recorded not in ("all", "-1", ""):
        try:
            layers = int(recorded)
        except ValueError:
            layers = mc_llm_context.ALL_LAYERS

    on_gpu = configuration.on_gpu
    if is_mixed(configuration):
        layers, on_gpu = mc_llm_context.ALL_LAYERS, True

    placement = mc_llm_context.Placement(
        gpu_layers=layers,
        context=max(int(configuration.context_size), MINIMUM_CONTEXT),
        kv_type_k=configuration.kv_type_k,
        kv_type_v=configuration.kv_type_v,
        on_gpu=on_gpu,
    )

    if configuration.context_mode == "auto" and placement.on_gpu:
        # Automatic sizing: spend what is free after weights and reserves on
        # context, rather than whatever number the state file happens to hold.
        weights = mc_llm_context.weights_bytes(gguf, placement) if gguf is not None else 0
        budget = mc_llm_context.automatic_buffer_bytes(
            _spendable(already_ours), weights, mc_broker.safety_margin_bytes())
        sized = mc_llm_context.context_for_budget(configuration.model, placement, budget)
        if sized >= MINIMUM_CONTEXT:
            placement = placement.with_context(sized)

    return _capped(placement, gguf)


def is_mixed(configuration: Config | None = None) -> bool:
    """Whether this installation is on Mixed placement with a card to use."""
    from prompt_master.core.models import MIXED_MODE
    from prompt_master.inference.device_detection import CPU_DEVICE

    try:
        configuration = configuration or config()
    except Exception:
        return False
    return (str(configuration.mode).strip().casefold() == MIXED_MODE
            and str(configuration.device).strip().casefold() != CPU_DEVICE)


def _capped(placement: mc_llm_context.Placement,
            gguf: mc_gguf.Gguf | None) -> mc_llm_context.Placement:
    """Never ask for more context than the model itself declares (section 12)."""
    if gguf is None or not gguf.context_length:
        return placement
    if placement.context <= gguf.context_length:
        return placement
    return placement.with_context(gguf.context_length)


def _free_vram(already_ours: int = 0) -> int:
    """Free VRAM, plus whatever a server of ours is holding while it is asked.

    The addition is the whole of it, and it is only ever correct because of
    what the caller does next: every path that acts on this number stops the
    running server before it starts another one, so those bytes are free by the
    time anything is placed in them. Nothing here may be used to decide that a
    *running* placement fits, which is a different question with a different
    answer -- see :meth:`Runtime.client`.

    The base figure is the *driver's*, not the host's. The host counts the
    blocks its allocator is holding cached as free, because for the host they
    are; llama.cpp is a different process and cannot have them. Sized against
    the host's number, a placement asks for VRAM that exists only inside this
    process -- and what comes back is ``cudaMalloc failed: out of memory`` on a
    card reporting twenty-two gigabytes free.
    """
    return mc_broker.device_free_vram_bytes() + max(int(already_ours), 0)


def _spendable(already_ours: int = 0) -> int:
    """What this placement may actually spend, which is not the same as what is free.

    Free VRAM is a reading of one instant. A generation is not one instant: the
    Creative Writer runs on a card the checkpoint has not been loaded onto yet,
    so what is free when the language model is placed is very nearly the whole
    card -- and what is free three hundred milliseconds later, when Stage 1
    loads and Stage 2 follows it, is not.

    Sizing against the reading is how a user's log comes to contain 71 server
    starts in one session, the negotiated context stepping 7168 -> 8192 -> 7168
    as consecutive generations found the card in different states, and five
    consecutive generations whose every start attempt died with ``cudaMalloc
    failed: out of memory`` on a card that had reported 22.7 GB free moments
    earlier.

    So when a plan is in force, the reading is capped by what that plan leaves
    over -- :func:`mc_plan.persistent_llm_budget` -- and the same answer comes
    back whether the checkpoint happens to be resident at this instant or not.
    That stability is the point: a placement that does not change is a server
    that does not restart, and a server that does not restart keeps llama.cpp's
    prompt cache, which is the difference between a warm second call and
    thirteen seconds of prompt evaluation.

    A learned cap from a previous reserve miss applies on top, and only ever
    downwards (rule 16: no silent promotion back to a placement that failed).

    With no plan published -- LLM Studio writing a prompt with no generation
    behind it -- this is exactly :func:`_free_vram`, which is the behaviour
    every path had before plans existed.
    """
    free = _free_vram(already_ours)
    try:
        import mc_plan

        # Only an *active plan* caps anything. With none published there is no
        # image workload to protect, and the budget arithmetic would then return
        # the whole card -- measured a second way, from a second source, and
        # therefore very slightly different from the reading above. Capping one
        # measurement of the card by another measurement of the same card is not
        # a policy, it is a rounding error with the power to move experts into
        # system RAM.
        budget = (mc_plan.persistent_llm_budget(already_ours)
                  if mc_plan.current() is not None else -1)
        learned = mc_plan.learned_cap_bytes()
    except Exception:
        logger.debug("Model Chain: could not read the active plan's budget", exc_info=True)
        return free

    # -1 is "there is nothing to divide up", not "there is no room".
    if budget >= 0:
        free = min(free, budget)
    if learned > 0:
        free = min(free, learned)
    return max(free, 0)


def _fits(estimate: mc_llm_context.Estimate, reserve: int, already_ours: int = 0) -> bool:
    return _spendable(already_ours) >= estimate.total_bytes + reserve


def _shrink_context(configuration: Config, placement, gguf, reserve: int,
                    already_ours: int = 0):
    """Lower the context until the cache fits what is free, or hit the floor."""
    free = _spendable(already_ours)
    estimate = mc_llm_context.estimate(configuration.model, placement, gguf)
    per_token = estimate.kv_bytes_per_token
    if per_token <= 0:
        return placement, estimate, ""

    budget = free - reserve - estimate.weights_bytes - estimate.compute_bytes
    affordable = int(max(budget, 0) / per_token)
    affordable = affordable // mc_llm_context.CONTEXT_GRANULARITY * mc_llm_context.CONTEXT_GRANULARITY

    if affordable >= placement.context:
        return placement, estimate, ""
    if affordable < MINIMUM_CONTEXT:
        return placement, estimate, ""

    was = placement.context
    placement = placement.with_context(affordable)
    return (placement, mc_llm_context.estimate(configuration.model, placement, gguf),
            f"context reduced from {was:,} to {affordable:,} tokens to fit the free VRAM")


EXPERT_RETRY_STEP = 2
"""How many more blocks' experts move after a load that ran out of memory anyway.

The arithmetic in :func:`_minimum_expert_layers` is planning, and planning can
be wrong in one direction that matters: llama.cpp allocates its own buffers on
top of what this module predicts, and a driver refuses an allocation for reasons
-- fragmentation, another process, a Windows limit -- that no header can see. A
start that failed is evidence that the plan was short, so the next attempt moves
two more blocks' experts rather than re-deriving the same number.

Two rather than one because a retry is not free (a failed start costs the load
time twice over), and rather than "all of them" because that is the cliff this
whole ladder exists to avoid.
"""


def _shrink_offload(configuration: Config, placement, gguf, reserve: int,
                    already_ours: int = 0,
                    expert_floor: int = mc_llm_context.NO_EXPERTS):
    """Move the model off the card in the order that costs the least speed.

    Reached only when a full-precision placement will not fit even at the
    minimum context, and the order is the whole of it:

    1. the experts of as few blocks as will cover the shortfall
       (``--n-cpu-moe N``);
    2. every expert (``--cpu-moe``), once N has reached every block anyway;
    3. whole blocks, four at a time;
    4. system RAM.

    Every block left in system RAM costs speed and gives back both its weights
    and its share of the cache, so the last two converge quickly -- and if they
    converge all the way to zero the model runs from system RAM, which is slow
    but is still an answer rather than a failure.
    """
    if gguf is None or not gguf.usable or gguf.block_count <= 0:
        return placement, mc_llm_context.estimate(configuration.model, placement, gguf), ""

    total = gguf.block_count
    free = _spendable(already_ours)
    notes: list[str] = []

    placement, estimate, spilled = _spill_experts(configuration, placement, gguf, reserve,
                                                  free, expert_floor)
    if spilled:
        notes.append(spilled)
    if free >= estimate.total_bytes + reserve:
        return placement, estimate, "; ".join(notes)

    for layers in range(total - 4, -1, -4):
        candidate = placement.with_layers(max(layers, 0))
        estimate = mc_llm_context.estimate(configuration.model, candidate, gguf)
        if free >= estimate.total_bytes + reserve:
            note = (f"offload reduced to {max(layers, 0)} of {total} layers on the GPU; "
                    "the rest run from system RAM and will be slower")
            if layers <= 0:
                note = ("the whole model was placed in system RAM -- there is not enough free "
                        "VRAM for any of it alongside what is resident; generation will be slow")
            notes.append(note)
            return candidate, estimate, "; ".join(notes)

    candidate = placement.with_layers(mc_llm_context.NO_LAYERS)
    notes.append("the whole model was placed in system RAM; generation will be slow")
    return (candidate, mc_llm_context.estimate(configuration.model, candidate, gguf),
            "; ".join(notes))


def _spill_experts(configuration: Config, placement, gguf, reserve: int, free: int,
                   floor: int = mc_llm_context.NO_EXPERTS):
    """As few blocks' experts in system RAM as will make the model fit.

    The first rung of the ladder, and the one that used to be all-or-nothing.
    For a mixture-of-experts model the first thing to move is the experts, not
    whole blocks: they are the great majority of the weights and are consulted a
    couple at a time, while attention is small and every token touches it.
    Dropping blocks gives up both; this gives up only the idle half.

    What is new is *how much* of the idle half. ``--cpu-moe`` moves all of it,
    which for a model two gigabytes over is thirty-four blocks of experts read
    from system RAM to save a shortfall six would have covered. So the count is
    computed, tried, and stepped up only if the arithmetic was optimistic --
    and it becomes ``--cpu-moe`` exactly when it reaches every block, because at
    that point the older flag says the same thing in a spelling every build
    understands.

    Returns ``(placement, estimate, note)`` whether or not the result fits: a
    placement that still does not fit has given the caller the largest saving
    this rung has, and the caller goes on to drop blocks from it.
    """
    estimate = mc_llm_context.estimate(configuration.model, placement, gguf)
    if not gguf.mixture_of_experts or placement.all_cpu_experts:
        return placement, estimate, ""

    progressive = runtime_supports(N_CPU_MOE_FLAG, configuration)
    everything = runtime_supports(CPU_MOE_FLAG, configuration)
    if not progressive and not everything:
        # An older build that has never heard of either flag. Passing one is not
        # a slower server, it is a server that exits at startup, so this rung is
        # skipped entirely and the blocks below it do the work.
        return placement, estimate, ""

    total = gguf.block_count
    wanted = max(int(floor), int(placement.cpu_expert_layers), mc_llm_context.NO_EXPERTS)

    # A floor of ``ALL_EXPERTS`` is a start that ran out of memory with every
    # expert already elsewhere. Recomputing a smaller N from arithmetic that has
    # just been shown to be optimistic would place the next server somewhere
    # worse than the one that failed.
    if progressive and floor != mc_llm_context.ALL_EXPERTS:
        needed = _minimum_expert_layers(configuration, placement, gguf,
                                        estimate, free - reserve)
        if needed == mc_llm_context.NO_EXPERTS and wanted == mc_llm_context.NO_EXPERTS:
            # There is no shortfall to close. Only a caller asking about a
            # placement that already fits gets here, and the honest answer is
            # the placement it asked about.
            return placement, estimate, ""
        wanted = max(wanted, needed)
        while 0 < wanted < total:
            candidate = placement.with_cpu_expert_layers(wanted)
            trial = mc_llm_context.estimate(configuration.model, candidate, gguf)
            if free >= trial.total_bytes + reserve:
                return candidate, trial, _expert_note(gguf, wanted, total)
            wanted += EXPERT_RETRY_STEP

    if everything:
        candidate = placement.with_cpu_experts()
    elif total > 0:
        # ``--n-cpu-moe`` alone, asked for every block: the same placement in
        # the only spelling this build has.
        candidate = placement.with_cpu_expert_layers(total)
    else:
        return placement, estimate, ""
    trial = mc_llm_context.estimate(configuration.model, candidate, gguf)
    return candidate, trial, _expert_note(gguf, mc_llm_context.ALL_EXPERTS, total)


def _expert_note(gguf: mc_gguf.Gguf, layers: int, total: int) -> str:
    """One clause saying which experts moved and why it is the cheap thing to do."""
    why = (f"{gguf.expert_count} experts a block, {gguf.expert_used_count} consulted per "
           "token, so this costs far less speed than moving blocks")
    if layers == mc_llm_context.ALL_EXPERTS or layers >= total > 0:
        return (f"the experts stay in system RAM and the rest of the model is on the GPU — {why}")
    return (f"the experts of {layers} of {total} layers stay in system RAM and everything "
            f"else is on the GPU — {why}")


def _minimum_expert_layers(configuration: Config, placement, gguf,
                           estimate: mc_llm_context.Estimate, budget: int) -> int:
    """The fewest blocks whose experts have to leave for ``placement`` to fit.

    Solved rather than searched. The saving is linear in the number of blocks
    moved -- that is what :func:`mc_llm_context.expert_fraction_moved` says --
    so the whole-model saving is measured once and the shortfall divided by the
    per-block share of it. One estimate, one division, and the caller verifies
    the answer against the real arithmetic anyway before it uses it.

    Measured from the estimator rather than computed from ``file_bytes`` here so
    that a partial block offload is accounted for: with half the blocks already
    off the card, moving a block's experts saves half as much, and a formula in
    this function would have to know that twice.
    """
    deficit = int(estimate.total_bytes) - int(budget)
    if deficit <= 0:
        return mc_llm_context.NO_EXPERTS

    total = gguf.block_count
    if total <= 0:
        return mc_llm_context.ALL_EXPERTS
    moved = max(int(placement.cpu_expert_layers), mc_llm_context.NO_EXPERTS)
    movable = total - moved
    if movable <= 0:
        return total
    whole = mc_llm_context.estimate(configuration.model,
                                    placement.with_cpu_expert_layers(total), gguf)
    saving = int(estimate.total_bytes) - int(whole.total_bytes)
    if saving <= 0:
        # No expert weights to speak of, or a header that will not say. Moving
        # them buys nothing, and claiming a number here would spend a restart
        # finding that out.
        return mc_llm_context.ALL_EXPERTS

    # ``saving`` is what the blocks that have *not* moved yet are worth, so it
    # is divided by those and the answer is an absolute count again. The two are
    # the same number on the common path, where nothing has moved yet, and this
    # is not the place to depend on that.
    return min(moved + int(math.ceil(deficit / (saving / movable))), total)


# --------------------------------------------------------------------------- #
# What llama.cpp says it did (section 13, read back rather than assumed)
# --------------------------------------------------------------------------- #
#
# Everything above this line is a *decision*: where the model should go, given
# what is on the card. Nothing above it is evidence that llama.cpp did what it
# was told. The two came apart in a way that cost a user an afternoon -- a
# placement reported as "all layers on the GPU" generating at a fifth of the
# speed a fully-resident model on that card generates at -- and the only place
# the truth was written down was llama-server's own log, which nobody reads
# because it is a file in a folder rather than a line in the console.
#
# So it is read. Two lines of somebody else's log format, parsed leniently,
# reported once per start, and never allowed to fail a load: the placement is
# what it is whether or not this can describe it.

_LAYERS = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
_WEIGHTS = re.compile(r"load_tensors:\s*(.+?)\s+(?:model\s+)?buffer size\s*=\s*([0-9.]+)\s*MiB")
_CACHE = re.compile(r"(\S+)\s+KV buffer size\s*=\s*([0-9.]+)\s*MiB")

# The second format, and the reason there is a second one. A 2025 build logs
# none of the three above: it fits the model to the device itself, says so, and
# leaves the per-buffer accounting out of the server log entirely. What it does
# say is what the context ended up as, what it saw free on the card, why a load
# failed, and -- after every request -- how fast that request actually ran,
# which is the number the whole placement argument is really about.
_GRANTED = re.compile(r"n_ctx_seq\s*\((\d+)\)")
_DEVICE = re.compile(r"-\s*(CUDA\d+|GPU\d+)\s*:.*?\(\s*(\d+)\s*MiB,\s*(\d+)\s*MiB free\s*\)")
_ALLOC_FAILED = re.compile(
    r"allocating\s+([0-9.]+)\s*MiB on device\s*(\d+):\s*(?:cudaMalloc|.*?)\s*failed:?\s*(.*)")
_LOAD_FAILED = re.compile(r"(?:error loading model|failed to load model|exiting due to)[^\n]*")
_NO_DEVICE = re.compile(
    r"invalid value for main_gpu:\s*(-?\d+)\s*\(available devices:\s*0\s*\)")
"""A start told to use a GPU by a process that can see none. See :func:`without_gpu_selection`."""
_DEVICE_GONE = re.compile(
    r'error while handling argument "--device":\s*invalid device:\s*(\S+)')
"""The card was named on the command line and llama.cpp could not find it.

The sibling of :data:`_NO_DEVICE`, and the more alarming one, because it fires
on installations where the same token worked seconds earlier. From one user's
log, 31 starts in a single session died here while 86 others enumerated
``CUDA0 : NVIDIA GeForce RTX 3090`` perfectly well.

It is an argument-parsing failure, so it happens before the model is opened and
leaves nothing else in the log to go on. What makes it a *memory* symptom
rather than a configuration one is where it appears: interleaved with
``cudaMalloc failed: out of memory`` on the same card, during the stretch of
the session where the image side was holding nearly all of it. A CUDA context
needs VRAM of its own before any device can be enumerated, and a process that
cannot create one registers no CUDA devices at all -- at which point the
perfectly correct ``--device CUDA0`` on the command line names a device that,
for this process, does not exist.

So it is treated as an out-of-memory failure: the start is retried with more
headroom, exactly as an allocation failure would be, rather than being reported
as a permanent misconfiguration the user is asked to go and fix.
"""
_SPEED = re.compile(
    r"(prompt eval|eval) time\s*=.*?\(\s*[0-9.]+ ms per token,\s*([0-9.]+) tokens per second\)")

SYSTEM_BUFFERS = ("cpu", "host")
"""Buffer names that mean system RAM rather than the card.

llama.cpp names a buffer for the backend that owns it -- ``CUDA0``, ``CPU``,
``CPU_Mapped``, ``CUDA_Host``. The first word is the part that says where the
memory is, except for ``CUDA_Host``, which is pinned *system* memory and is
matched here by its second word rather than its first.
"""

SYSTEM_SHARE_WARNING = 0.10
"""Share of the weights in system RAM that is worth a warning.

Not zero. A full offload still leaves a token-embedding buffer on the host for
many models, and a warning that fires on every load is a warning nobody reads.
A tenth of the model in system RAM is not that: it is the difference between a
reply that streams and one that crawls.
"""


@dataclass(frozen=True)
class Offload:
    """What llama.cpp reported about the load it just did."""

    layers: int = 0
    total_layers: int = 0
    weights: tuple[tuple[str, float], ...] = ()
    cache: tuple[tuple[str, float], ...] = ()
    granted_context: int = 0
    """The context llama.cpp settled on, which is not always the one asked for:
    a build with its own fitter adjusts it to what it thinks will fit."""
    device_free: tuple[tuple[str, int, int], ...] = ()
    """``(device, total MiB, free MiB)`` as llama.cpp saw it at start."""

    @property
    def known(self) -> bool:
        return bool(self.total_layers or self.weights or self.granted_context
                    or self.device_free)

    @staticmethod
    def _bytes(buffers, system: bool) -> int:
        return int(sum(size for name, size in buffers
                       if _is_system(name) is system) * _MB)

    @property
    def system_bytes(self) -> int:
        return self._bytes(self.weights, True)

    @property
    def device_bytes(self) -> int:
        return self._bytes(self.weights, False)

    @property
    def system_share(self) -> float:
        total = self.system_bytes + self.device_bytes
        return self.system_bytes / total if total else 0.0

    @property
    def spilled(self) -> bool:
        """Whether enough of the weights are in system RAM to explain a slow reply."""
        return self.system_share >= SYSTEM_SHARE_WARNING

    def describe(self) -> str:
        parts = []
        if self.total_layers:
            parts.append(f"{self.layers}/{self.total_layers} layers on the GPU")
        for name, size in self.weights + self.cache:
            parts.append(f"{name} {size * _MB / _GB:.1f} GB")
        if self.granted_context:
            parts.append(f"{self.granted_context:,} token context")
        for device, total, free in self.device_free:
            parts.append(f"{device} {free * _MB / _GB:.1f} GB free of "
                         f"{total * _MB / _GB:.1f} GB at start")
        return ", ".join(parts)


def _is_system(name: str) -> bool:
    return any(word in name.casefold() for word in SYSTEM_BUFFERS)


def read_offload(text: str) -> Offload:
    """Parse llama.cpp's load report out of ``text``.

    Lenient by construction: this is somebody else's log format, it has changed
    before, and every field is optional. What comes back from a version that
    writes it differently is an empty report, which reads as "could not tell"
    rather than as "nothing was offloaded" -- the distinction matters, because
    only one of those is worth warning somebody about.
    """
    found = _LAYERS.search(text)
    layers, total = (int(found.group(1)), int(found.group(2))) if found else (0, 0)
    weights = tuple((name.strip(), float(size)) for name, size in _WEIGHTS.findall(text))
    cache = tuple((f"{name.strip()} KV", float(size)) for name, size in _CACHE.findall(text))
    granted = _GRANTED.findall(text)
    devices = tuple((name, int(total_mib), int(free_mib))
                    for name, total_mib, free_mib in _DEVICE.findall(text))
    return Offload(layers=layers, total_layers=total, weights=weights, cache=cache,
                   granted_context=int(granted[-1]) if granted else 0,
                   device_free=devices)


@dataclass(frozen=True)
class Failure:
    """Why a start failed, and whether a smaller placement would help."""

    text: str = ""
    out_of_memory: bool = False

    def __bool__(self) -> bool:
        return bool(self.text)


# --------------------------------------------------------------------------- #
# What this build of llama-server can be asked for
# --------------------------------------------------------------------------- #

CPU_MOE_FLAG = "--cpu-moe"
"""Keep every expert tensor in system RAM, everything else on the card."""

N_CPU_MOE_FLAG = "--n-cpu-moe"
"""Keep the expert tensors of the first N blocks in system RAM, and no more.

The finer-grained half of :data:`CPU_MOE_FLAG`, and the one worth reaching for
first. ``--cpu-moe`` is a cliff: a model two gigabytes too large for a card
moves *all* of its experts into system RAM and generates at the speed of the
slowest thing in the path, when the first six blocks' worth would have covered
the shortfall and left the other thirty-four on the GPU.

Newer than ``--cpu-moe`` in llama.cpp, so the two are asked about separately and
neither is assumed from the other -- see :func:`runtime_capabilities`. A build
that has one, the other, or neither is a build this module still places.
"""

FLASH_ATTENTION_FLAG = "--flash-attn"
"""Fused attention kernels. A CUDA thing: worth nothing with no layers offloaded."""

FULL_ATTENTION_WINDOW_FLAG = "--swa-full"
"""Keep the whole key/value cache, rather than a sliding window of it.

A sliding-window model -- Gemma is one, at ``n_swa = 1024`` -- keeps only a
window of the cache for most of its blocks, and llama.cpp can therefore only
*resume* a cached prompt at one of the context checkpoints it took on the way
past. That turns a nearly-free warm turn into a mostly-cold one. From a user's
log, on three consecutive Krea rolls that shared a 668-token instruction:

    slot get_availabl: selected slot by LCP similarity, sim_best = 0.630
    slot update_slots: restored context checkpoint (pos_max = 460, n_past = 460)
    slot update_slots: erased invalidated context checkpoint (n_swa = 1024)
    slot print_timing: prompt eval time = 21335 ms / 601 tokens

668 tokens matched; 460 were resumed; the other 208 were processed again
because the only checkpoint below the divergence was at 460. At the 28 tokens
a second that placement was managing, those 208 tokens are seven seconds of a
user watching a progress bar before the first character appears.

With the full cache there are no checkpoints to land on and no window to fall
out of: llama.cpp resumes at the exact length of the common prefix. What it
costs is the memory the window was saving -- which this extension has always
budgeted for anyway, because :func:`mc_llm_context.estimate` sizes the cache at
the full context for every block. So the reserve does not move; reality merely
stops being cheaper than the arithmetic that placed it.
"""

_HELP_TIMEOUT = 20
_capabilities: dict[tuple, frozenset] = {}
_capabilities_lock = threading.Lock()


def runtime_capabilities(configuration: Config | None = None) -> frozenset:
    """Every long option this llama-server build advertises, or an empty set.

    Asked of the binary rather than assumed from a version string, because
    there is no version string: the runtime is whatever build the user copied
    in, and the two flags this module wants to add were added to llama.cpp at
    different times and spelled differently before that. A flag that is passed
    to a build which does not know it is not a slower server, it is a server
    that exits at startup -- which is exactly the failure this extension spent a
    week on already.

    Cached per executable and modification time, so adopting a new build asks
    again and an ordinary start asks nothing.
    """
    try:
        configuration = configuration or config()
        executable = configuration.runtime
        if executable is None or not Path(executable).is_file():
            return frozenset()
        stamp = (str(executable), Path(executable).stat().st_mtime)
    except Exception:
        return frozenset()

    with _capabilities_lock:
        found = _capabilities.get(stamp)
    if found is not None:
        return found

    found = _read_capabilities(executable)
    with _capabilities_lock:
        _capabilities[stamp] = found
    return found


def _read_capabilities(executable) -> frozenset:
    """``llama-server --help``, reduced to the set of long options it lists."""
    try:
        finished = subprocess.run(
            [str(executable), "--help"], capture_output=True, text=True,
            timeout=_HELP_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        logger.debug("Model Chain: could not ask llama-server what it supports",
                     exc_info=True)
        return frozenset()
    text = f"{finished.stdout or ''}\n{finished.stderr or ''}"
    flags = frozenset(re.findall(r"(--[a-z0-9][a-z0-9-]+)", text))
    if flags:
        logger.debug("Model Chain: llama-server advertises %d options", len(flags))
    return flags


def runtime_supports(flag: str, configuration: Config | None = None) -> bool:
    """Whether this build advertises ``flag``. False when it cannot be asked."""
    return flag in runtime_capabilities(configuration)


def accelerator_flags(configuration: Config, placement) -> list[str]:
    """Extra command-line flags for this placement, gated on the build having them.

    ``--swa-full`` is the only one that is not about the card, and it is added
    for every placement including a processor-only one, because what it buys is
    prompt *reuse* rather than throughput -- see
    :data:`FULL_ATTENTION_WINDOW_FLAG`. It is also the only one that costs
    memory, and the estimator has always charged for it.

    ``--flash-attn`` is fused attention. It is a CUDA kernel, so it is added
    only when something is actually offloaded -- on a placement with no resident
    layers it would be a flag that changes nothing, and a flag that changes
    nothing is a flag somebody will later believe changed something.

    ``--n-cpu-moe N`` and ``--cpu-moe`` are how the placement above says
    "experts in system RAM", partly and entirely. They reach the command line
    here rather than through the vendored launcher's fixed argument list, which
    has no room for either. Only one of the two is ever passed, and a partial
    split on a build that has only the all-or-nothing flag is not silently
    promoted to it: the placement was chosen against an estimate of *N* layers,
    and moving every expert instead would be a different footprint and a
    different speed than the one negotiated. :func:`_shrink_offload` does not
    produce such a placement, and this is the second place that is true.

    There is no fourth. Sage attention is a quantised attention kernel for
    diffusion models in PyTorch and has no llama.cpp counterpart, and the
    remaining llama.cpp knobs -- thread counts, batch sizes -- are hardware
    guesses this module has no way to verify from here.
    """
    flags: list[str] = []
    if runtime_supports(FULL_ATTENTION_WINDOW_FLAG, configuration):
        flags.append(FULL_ATTENTION_WINDOW_FLAG)
    if not getattr(placement, "on_gpu", False):
        return flags
    flags.extend(expert_flags(configuration, placement))
    if placement.gpu_layers == mc_llm_context.NO_LAYERS:
        return flags
    if runtime_supports(FLASH_ATTENTION_FLAG, configuration):
        flags.append(FLASH_ATTENTION_FLAG)
        if _flash_attention_takes_a_value(configuration):
            flags.append("on")
    return flags


def expert_flags(configuration: Config, placement) -> list[str]:
    """The expert-offload part of a command line, in the spelling this build has.

    Empty for a placement with every expert on the card, for a build that
    advertises neither flag, and for a dense model -- which never reaches a
    placement with experts moved, because :func:`_shrink_offload` reads the
    header before it makes one.
    """
    layers = int(getattr(placement, "cpu_expert_layers", mc_llm_context.NO_EXPERTS))
    if layers == mc_llm_context.NO_EXPERTS:
        return []
    if layers == mc_llm_context.ALL_EXPERTS:
        return [CPU_MOE_FLAG] if runtime_supports(CPU_MOE_FLAG, configuration) else []
    if runtime_supports(N_CPU_MOE_FLAG, configuration):
        return [N_CPU_MOE_FLAG, str(layers)]
    return []


def _flash_attention_takes_a_value(configuration: Config | None = None) -> bool:
    """Whether this build spells it ``--flash-attn on`` rather than as a switch.

    llama.cpp changed it from a boolean switch to a three-state option
    (``on``/``off``/``auto``) and both spellings are in the wild. Passing the
    wrong one is a server that will not start, so the help text decides.
    """
    try:
        configuration = configuration or config()
        executable = configuration.runtime
        if executable is None:
            return False
        finished = subprocess.run(
            [str(executable), "--help"], capture_output=True, text=True,
            timeout=_HELP_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return False
    text = f"{finished.stdout or ''}\n{finished.stderr or ''}"
    match = re.search(r"--flash-attn[^\n]*", text)
    return bool(match and re.search(r"\bon\b.*\boff\b|\{on", match.group(0)))


# --------------------------------------------------------------------------- #
# What this machine measured, per backbone
# --------------------------------------------------------------------------- #

WRITE_RATE = "llm:write"
READ_RATE = "llm:read"
"""Measured tokens per second, in the progress store, keyed by backbone.

Not a proxy and not an estimate: llama.cpp reports both figures for every
request it serves, and this is where they are kept so that something other than
a log line can use them.

Keyed per backbone because that is the axis they differ most on, and the
difference is not the one the catalogue's sizes suggest. Measured on one
machine, in system RAM: a dense 12B wrote at 4.9 tokens a second and a 26B
mixture-of-experts wrote at 12.8, because generation from RAM is bandwidth-bound
and an MoE activates a fraction of its weights per token. The bigger file was two
and a half times faster, and nothing on screen said so.

Keyed per *placement* as well, because that is the axis they differ most on
after the backbone, and by more: the same 26B writes at forty tokens a second
resident on a card and at five from system RAM. One number covering both is a
number that is wrong for each of them, and the ladder now has rungs between
those two -- eight blocks' experts in system RAM is a real speed and not an
interpolation of the other two. So the store learns
``llm:write:<backbone>:ncmoe-8`` and never averages it into
``llm:write:<backbone>``.
"""


def speed_key(kind: str, identity: str = "", placement=None) -> str:
    """The store key for one rate, e.g. ``llm:write:gemma4-12b:ncmoe-8``.

    The identity is normalised here rather than at each caller, because the two
    that matter reach it by different routes -- the running configuration and a
    catalogue entry's id -- and a key written by one and read by the other has
    to be the same key.

    ``placement`` may be a :class:`mc_llm_context.Placement`, a token from
    :attr:`~mc_llm_context.Placement.speed_token`, or ``None`` for the placement
    running right now. Pass ``""`` for the backbone-wide key with no placement
    in it at all, which is what the fallback chain below ends on.
    """
    identity = _key_safe(identity) if identity else writer_identity()
    token = placement if isinstance(placement, str) else placement_token(placement)
    return ":".join(part for part in (kind, identity, _key_safe(token)) if part)


def placement_token(placement=None) -> str:
    """``placement``'s speed token, defaulting to the placement running now.

    Empty when nothing is running and nothing was passed, which is the honest
    answer: a rate recorded then is a rate about a placement nobody can name,
    and it goes under the backbone-wide key rather than under a guess.
    """
    if placement is None:
        try:
            placement = runtime.placement()
        except Exception:
            return ""
    if placement is None:
        return ""
    try:
        return str(placement.speed_token)
    except Exception:
        return ""


PLACEMENT_WORDS = {
    "gpu": "all layers on the GPU",
    "cpu": "in system RAM",
    "cpu-moe": "experts in system RAM",
}


def describe_placement_token(token: str) -> str:
    """One short clause for a placement token, for a line a person reads.

    ``ncmoe-8`` is "8 expert layers in RAM", which is the sentence the design
    intent asks the catalogue to be able to print beside a measured rate. A
    token this does not recognise is returned as it is rather than dropped: an
    unfamiliar word beside a number is a smaller failure than a number whose
    placement is not stated at all.
    """
    token = str(token or "").strip()
    if not token:
        return ""
    head, _, layers = token.partition("-l")
    said = PLACEMENT_WORDS.get(head)
    if said is None and head.startswith("ncmoe-"):
        count = head[len("ncmoe-"):]
        said = f"{count} expert layer{'' if count == '1' else 's'} in RAM"
    if said is None:
        said = head
    if layers:
        said = f"{said}, {layers} layers on the GPU"
    return said


def _key_safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "")).strip("-").casefold()[:64]


def speed_keys(kind: str, identity: str = "", placement=None) -> tuple[str, ...]:
    """That key, then the broader ones, for :func:`mc_progress.rate_for`.

    Most specific first, which is the convention the store was built around: a
    machine that has measured this backbone *at this placement* answers about
    that; one that has measured the backbone somewhere else answers about the
    backbone; one that has measured neither falls back to whatever it has
    learned in general rather than to a built-in guess.

    Reading through the backbone-wide key is not the same as writing to it.
    Nothing writes there any more -- see :data:`WRITE_RATE` -- so what it can
    still hold is a measurement recorded before placements were keyed, which is
    a stale approximation and a better first guess than none.
    """
    keys = (speed_key(kind, identity, placement), speed_key(kind, identity, ""), kind)
    ordered: list[str] = []
    for key in keys:
        if key not in ordered:
            ordered.append(key)
    return tuple(ordered)


def writer_identity(configuration: Config | None = None) -> str:
    """A short, stable name for the configured backbone, for keying rates by.

    The catalogue id when there is one, because that is what a user chose and
    what the catalogue can look a rate up by; the file's stem otherwise. Both
    are reduced to characters a settings key can hold.
    """
    try:
        configuration = configuration or config()
    except Exception:
        return ""
    name = str(getattr(configuration, "managed_id", "") or "")
    if not name:
        model = getattr(configuration, "model", None)
        name = model.stem if model is not None else ""
    return _key_safe(name)


def remember_speed(prompt: float, reply: float, identity: str = "", placement=None) -> None:
    """Fold one request's measured rates into the store. Never fatal.

    Written to the placement-specific key alone. Recording it in the
    backbone-wide key as well would put the average of a resident placement and
    a system-RAM one somewhere a caller can read it, and the whole reason this
    is keyed by placement is that such an average describes neither.
    """
    try:
        import mc_progress

        identity = identity or writer_identity()
        token = placement if isinstance(placement, str) else placement_token(placement)
        if reply > 0:
            mc_progress.learn(speed_key(WRITE_RATE, identity, token), float(reply))
        if prompt > 0:
            mc_progress.learn(speed_key(READ_RATE, identity, token), float(prompt))
    except Exception:
        logger.debug("Model Chain: could not record llama.cpp's measured speed",
                     exc_info=True)


def measured_speed(identity: str = "", placement=None) -> tuple[float, float]:
    """``(prompt, reply)`` tokens per second for one backbone at one placement.

    ``placement`` defaults to the one running now, and zeros come back when
    nothing has been measured there. For "whatever this machine knows about this
    backbone", which is the catalogue's question rather than the status line's,
    see :func:`best_measured`.
    """
    try:
        import mc_progress

        identity = identity or writer_identity()
        if not identity:
            return 0.0, 0.0
        token = placement if isinstance(placement, str) else placement_token(placement)
        return (float(mc_progress.measured(speed_key(READ_RATE, identity, token), 0.0)),
                float(mc_progress.measured(speed_key(WRITE_RATE, identity, token), 0.0)))
    except Exception:
        return 0.0, 0.0


def best_measured(identity: str = "") -> tuple[float, float, str]:
    """``(prompt, reply, placement token)``: the best this backbone has done here.

    The catalogue's question. It is asked about backbones that are not loaded
    and mostly never have been at the placement they would get next, so keying
    the answer to the running placement would blank the line for every entry but
    one. What it reports instead is the placement running now when that is this
    backbone, and otherwise the fastest placement on record -- with the token, so
    the line can say *where* rather than implying the number is unconditional.
    """
    identity = _key_safe(identity) if identity else writer_identity()
    if not identity:
        return 0.0, 0.0, ""
    try:
        import mc_progress

        store = mc_progress.rates()
    except Exception:
        return 0.0, 0.0, ""

    prefix = f"{WRITE_RATE}:{identity}:"
    running = placement_token() if writer_identity() == identity else ""
    found, rate = "", 0.0
    for key, value in store.items():
        if not key.startswith(prefix) or not value:
            continue
        token = key[len(prefix):]
        if token == running:
            found, rate = token, float(value)
            break
        if float(value) > rate:
            found, rate = token, float(value)
    if rate <= 0:
        # Nothing keyed by placement. A measurement recorded before placements
        # were keyed still answers the question, and says nothing about where.
        rate = float(store.get(f"{WRITE_RATE}:{identity}") or 0.0)
        if rate <= 0:
            return 0.0, 0.0, ""
        return float(store.get(f"{READ_RATE}:{identity}") or 0.0), rate, ""
    return float(store.get(f"{READ_RATE}:{identity}:{found}") or 0.0), rate, found


# --------------------------------------------------------------------------- #
# One impossible flag pair, taken back out of the vendored command
# --------------------------------------------------------------------------- #

CPU_DEVICE_TOKEN = "none"
"""llama.cpp's own word for "offload to nothing"; ``device_detection.CPU_DEVICE``.

Restated here rather than imported at module scope for the reason every other
``prompt_master`` import in this file is done inside a function: this module is
imported while the WebUI is still building its UI, and the vendored tree is not
required to be importable for that to work.
"""


def without_gpu_selection(command):
    """``command``, minus the single-GPU selection, when there is no GPU to select.

    The vendored launcher writes the same line every time, and two halves of it
    contradict each other in CPU placement::

        --device none --split-mode none --main-gpu 0

    ``--device none`` says "no devices"; ``--split-mode none --main-gpu 0`` says
    "use device number 0". llama.cpp used to ignore the second half when the
    first had emptied the device list. Builds since validate it, and what a user
    gets is every start of the language model dying at load::

        llama_prepare_model_devices: invalid value for main_gpu: 0 (available devices: 0)
        llama_model_load_from_file_impl: failed to load model
        srv  llama_server: exiting due to model loading error

    -- which is not a degraded LLM, it is no LLM: LLM Studio cannot answer and
    Creative Mode falls back to the prompt as typed, on every generation, for as
    long as the installation stays on CPU placement.

    So the selection comes off when the device is ``none``, which is the only
    case it can be wrong in. A GPU or Mixed placement names a card and keeps the
    line it has always had.

    ``prompt_master/`` is a byte-identical vendored tree whose own
    ``VENDORED_FROM.txt`` says "do not hand-edit these files -- changes belong in
    the mc_llm_* modules that sit on top of them". This is that change, and
    :class:`_Launcher` is how it reaches the command without one being edited.
    """
    argv = [str(part) for part in command or ()]
    try:
        device = argv[argv.index("--device") + 1].strip().casefold()
    except (ValueError, IndexError):
        return list(argv)
    if device != CPU_DEVICE_TOKEN:
        return list(argv)

    kept, dropped, skip = [], [], False
    for position, part in enumerate(argv):
        if skip:
            skip = False
            dropped.append(part)
            continue
        if part in ("--split-mode", "--main-gpu"):
            skip = position + 1 < len(argv)
            dropped.append(part)
            continue
        kept.append(part)

    if dropped:
        logger.info("Model Chain: this llama-server is starting with no GPU visible, so "
                    "%s was left off its command line — llama.cpp refuses to select a "
                    "device it does not have", " ".join(dropped))
    return kept


_pending_flags: list[str] = []
"""Flags to append to the very next llama-server command. See :class:`_Launcher`.

Module state, and deliberately the smallest kind: it is written and read inside
the runtime's own lock, microseconds apart, by the one method that starts a
server. The alternative is a vendored launcher that takes arbitrary extra
arguments, and it does not -- ``LlamaProcess.start`` has a fixed keyword list,
which is the whole reason the command is corrected at the boundary instead.
"""


def _arm_flags(flags) -> None:
    del _pending_flags[:]
    _pending_flags.extend(str(flag) for flag in flags or ())


def with_extra_flags(command) -> list[str]:
    """``command`` plus whatever the start that is happening now armed.

    Appended at the end, where llama.cpp takes its options in any order, and
    consumed exactly once: a flag left armed would attach itself to the next
    server started for any reason, which on this path is a server started for a
    different placement.
    """
    argv = [str(part) for part in command or ()]
    if not _pending_flags:
        return argv
    extra, wanted = list(_pending_flags), False
    del _pending_flags[:]
    # Only a llama-server command gets them. Anything else spawned while a start
    # is in flight -- a device probe, a help query -- passes through untouched.
    wanted = "--model" in argv and "--ctx-size" in argv
    if not wanted:
        return argv
    logger.info("Model Chain: llama-server is starting with %s", " ".join(extra))
    return argv + extra


class _Launcher:
    """The vendored launcher's ``subprocess``, with one command rewritten.

    It forwards every attribute untouched and intercepts exactly one call, on
    exactly one command shape, on its way to the operating system. Installed in
    place of the module-level ``subprocess`` name inside one vendored file, which
    leaves that file byte-identical to its upstream and ``diff -r`` clean.

    A subclass overriding ``start()`` was the obvious alternative and is worse:
    the command is assembled inside that method, so overriding it means copying
    the whole assembly into this repository, where it would silently stop
    matching the next version of the vendored tree.
    """

    def __getattr__(self, name):
        return getattr(subprocess, name)

    @staticmethod
    def Popen(command, *args, **kwargs):  # noqa: N802 - subprocess's own spelling
        return subprocess.Popen(
            with_extra_flags(without_gpu_selection(command)), *args, **kwargs)


def _repair_launcher() -> None:
    """Put :class:`_Launcher` in front of the vendored launcher's ``subprocess``.

    Idempotent, and done at the moment a process is about to be started rather
    than at import: nothing here should run on an installation that never starts
    a language model.
    """
    try:
        from prompt_master.inference import llama_process
    except Exception:
        logger.debug("Model Chain: the vendored launcher could not be imported",
                     exc_info=True)
        return
    if not isinstance(getattr(llama_process, "subprocess", None), _Launcher):
        llama_process.subprocess = _Launcher()


def read_failure(text: str) -> Failure:
    """Why a start failed, in llama.cpp's own words.

    "llama-server exited before becoming ready" is true of every failed start
    and useful for none of them. The server always says why, in the log, one
    line before it goes -- and the sentence it uses is the difference between
    "buy a bigger card", "close the other thing using this one", and "ask for
    less of it", which is the one this module can act on by itself.
    """
    alloc = _ALLOC_FAILED.search(text)
    if alloc:
        asked = float(alloc.group(1)) * _MB
        why = (alloc.group(3) or "").strip().rstrip(".") or "the allocation was refused"
        seen = ""
        devices = _DEVICE.findall(text)
        if devices:
            _name, _total, free_mib = devices[0]
            seen = f", with {int(free_mib) * _MB / _GB:.1f} GB reported free"
        return Failure(
            f"llama-server could not fit on the card: it asked the driver for "
            f"{asked / _GB:.1f} GB in one piece and was refused ({why}){seen}",
            out_of_memory=True)
    gone = _DEVICE_GONE.search(text)
    if gone:
        return Failure(
            f"llama-server could not find {gone.group(1)} when it started, though the "
            "card is configured and has worked before. That normally means no CUDA "
            "context could be created — another process was holding the card when this "
            "one started, leaving too little for a CUDA context. If it persists, check "
            "that the runtime build has its CUDA backend beside it and that "
            "CUDA_VISIBLE_DEVICES is not set in the environment.",
            out_of_memory=True)
    no_device = _NO_DEVICE.search(text)
    if no_device:
        # Said in full because the log's own sentence -- "invalid value for
        # main_gpu: 0 (available devices: 0)" -- names the symptom and none of
        # the cause, and the two causes have completely different remedies.
        return Failure(
            "llama-server started with no GPU visible and was still told to use GPU "
            f"{no_device.group(1)}. If this installation is on CPU placement, update "
            "the extension: the flag pair is removed there now. Otherwise the card is "
            "not reaching llama.cpp — CUDA_VISIBLE_DEVICES set in the environment, or a "
            "runtime build without the CUDA backend beside it.")
    failure = _LOAD_FAILED.search(text)
    if failure:
        return Failure(failure.group(0).strip())
    return Failure()


def _text_since(log_path, offset: int, tail: int = 0) -> str:
    """The log this start wrote, or "" when it cannot be read.

    ``tail`` reads only the last that many bytes of it, never crossing back
    over ``offset`` into an earlier run's. A load report is at the beginning of
    a start and a request's timings are at the end of one, and a server that
    has been answering for an hour has a great deal in between.
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            start = max(int(offset), 0)
            if tail > 0:
                handle.seek(0, 2)
                start = max(start, handle.tell() - int(tail))
            handle.seek(start)
            return handle.read(_LOG_READ_LIMIT)
    except OSError:
        logger.debug("Model Chain: could not read llama-server's log", exc_info=True)
        return ""


_TIMING_TAIL = 64 * 1024
"""How far back to look for the last request's timings."""


def read_speed(text: str) -> tuple[float, float]:
    """The last request's ``(prompt, reply)`` tokens per second, or ``(0, 0)``.

    The one measurement that settles the argument this module spends its time
    having. A placement is a plan; this is the speed that plan produced, from
    the process that produced it, and on a card that has quietly stopped
    holding what it was given the two disagree by a factor of forty.
    """
    prompt = reply = 0.0
    for kind, rate in _SPEED.findall(text):
        if kind == "prompt eval":
            prompt = float(rate)
        else:
            reply = float(rate)
    return prompt, reply


def _offload_since(log_path, offset: int) -> Offload:
    """Read back what the server just wrote to ``log_path`` about itself.

    ``offset`` is where the file ended before this start, because the log is
    appended to across runs and the report wanted is this run's.
    """
    return read_offload(_text_since(log_path, offset))


_LOG_READ_LIMIT = 4 * 1024 * 1024
"""How much of one start's log to read. The load report is in the first lines
of it; the rest is request logging, and on a long-lived server there is a lot
of that."""

OFFLOAD_WAIT_SECONDS = 5.0
"""How long to wait for llama.cpp's load report to reach the file.

The server answers ``/health`` from the moment the model is loaded, and what it
wrote while loading is on the other side of somebody else's output buffer --
which on Windows is a block buffer rather than a line one when the output is a
file. Reading once, immediately, therefore found nothing at all on the platform
that most needed the answer. Waiting is safe because it is only ever reached
after a start, which already took long enough that five seconds is noise, and
it stops the moment there is something to read.
"""

OFFLOAD_POLL_SECONDS = 0.25


def _await_offload(log_path, offset: int) -> Offload:
    """:func:`_offload_since`, given a few seconds to appear."""
    deadline = time.monotonic() + OFFLOAD_WAIT_SECONDS
    while True:
        found = _offload_since(log_path, offset)
        if found.known or time.monotonic() >= deadline:
            return found
        time.sleep(OFFLOAD_POLL_SECONDS)


# --------------------------------------------------------------------------- #
# The runtime itself
# --------------------------------------------------------------------------- #

RESIDENCY_KEY = "llm:llama.cpp"

LAYER_UPGRADE_FRACTION = 0.25
"""How much more of the model has to reach the card before a restart is worth it.

The same quarter, and the same reasoning, as :data:`CONTEXT_UPGRADE_FRACTION` --
which had it and the layer comparison beside it did not, so *any* gain at all
was acted on.

What that cost, from a user's console with the clock on it: a second generation
of an identical request stopped a server holding no layers and started one
holding two of thirty, which took 9.9 seconds. llama.cpp's own timings either
side of it were 61.5 tokens a second on prompts before and 62.1 after, 12.71 a
second generating before and 12.46 after -- inside the noise, and slightly
worse on the half that matters. The restart also emptied the prompt cache, so
the writer's 523-token prompt was read from scratch for the second time in a
minute at a further 8.4 seconds. Eighteen of the fifty-three seconds that
"warm" generation spent on the language model went on undoing its own warmth.

A floor as well as a fraction, because a quarter of a small model is a rounding
error: on an eight-block model two blocks really are a quarter, and two blocks
are not worth ten seconds of anybody's time.
"""

MINIMUM_LAYER_GAIN = 4
"""Blocks a restart must move regardless of how few the model has in total."""

CONTEXT_UPGRADE_FRACTION = 0.25
"""How much more context has to be available before a running server is replaced.

A restart is not free -- it re-reads the weights, and it throws away llama.cpp's
prompt cache, so the next reply pays for the whole conversation to be processed
again from the first token. A few hundred tokens of extra window is not worth
that; a quarter more of it is. The asymmetry is deliberate: this is only ever
consulted about placing *more* on the card, never less.
"""


PRIME_LABEL = "priming the prompt cache"

_priming: threading.Thread | None = None


def _prime_prompt_cache(client) -> None:
    """Prefill the writer's standing instruction, on a thread nobody is waiting on.

    llama.cpp caches a prompt and resumes the next one at their common prefix,
    so the *second* Krea roll on a given server is much cheaper than the first:
    only the creative brief has changed, and the instruction above it is
    already in the cache. The first roll pays for the whole thing. From a
    user's log, on a processor-only placement at 30 tokens a second:

        task 5   | prompt eval time = 35337 ms / 1065 tokens   <- first roll
        task 199 | prompt eval time = 21335 ms /  601 tokens   <- second
        task 405 | prompt eval time = 20258 ms /  596 tokens   <- third

    Thirty-five seconds against twenty. The difference is the instruction, and
    there is no reason for a person watching a progress bar to be the one who
    pays for it: it is the same 460-odd tokens every time, it is known before
    anybody presses anything, and llama-server is started at WebUI boot with
    nothing else to do.

    So it is prefilled then, with a one-token completion whose answer is thrown
    away. What makes this safe to fire from a start that happened *inside* a
    run -- a re-placement mid-roll -- is the workload lock: this asks for it as
    background work and does not wait, so a start that some job is holding the
    GPU for finds it taken and does nothing at all. The roll it would have
    queued in front of is exactly the roll it was meant to help.
    """
    global _priming

    if _priming is not None and _priming.is_alive():
        return
    thread = threading.Thread(target=_prime, args=(client,), name="mc-llm-prime",
                              daemon=True)
    _priming = thread
    thread.start()


def _prime(client) -> None:
    try:
        from prompt_master.krea import enhancer

        if mc_broker.host_busy():
            return
        with mc_broker.workload(mc_broker.FAMILY_LLM, PRIME_LABEL, timeout=0,
                                background=True, required=False) as held:
            if not held:
                return
            started = time.monotonic()
            client.stream_chat(
                [{"role": "system", "content": enhancer.system_prompt(False)},
                 {"role": "user", "content": ""}],
                PRIME_TOKENS, PRIME_SEED, lambda _chunk: None,
            )
            logger.info("Model Chain: the writer's instruction is in llama.cpp's prompt "
                        "cache (%.1fs) — the first Krea roll now costs only what the "
                        "creative brief adds", time.monotonic() - started)
    except Exception:
        # Never a failure. Nothing depends on this having run; the only thing
        # that changes when it does not is that somebody waits for the tokens
        # this would have paid for.
        logger.debug("Model Chain: could not prime the prompt cache", exc_info=True)


PRIME_TOKENS = 1
"""Enough to make llama-server prefill and keep the prompt; no more."""

PRIME_SEED = 1


def _warn_about_an_idle_card(configuration: Config, layers: str) -> None:
    """Say once, per start, when a card is present and doing nothing.

    Mixed placement records ``--n-gpu-layers 0``, and llama.cpp with no
    offloaded layers runs every matrix multiply on the processor -- so a machine
    with a 3090 in it can spend twenty seconds a press at four tokens a second
    while the card sits idle, with nothing on screen or in the log saying that
    was the arrangement. This is that line.
    """
    from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD

    if str(layers) != NO_OFFLOAD:
        return
    if str(configuration.device).casefold() == CPU_DEVICE:
        return
    logger.info("Model Chain: %s will do no work in this placement — nothing was free "
                "to offload into, so the processor writes the whole prompt. Free VRAM "
                "before pressing Generate, choose GPU placement in LLM Studio → Setup, "
                "or pick a backbone that is faster in system RAM (a mixture-of-experts "
                "one is several times faster there than a dense one of the same size).",
                configuration.device_name or configuration.device)


def _layers_argument(placement: mc_llm_context.Placement,
                     gguf: mc_gguf.Gguf | None) -> str:
    """``--n-gpu-layers``, as a number whenever one can be worked out.

    llama.cpp's own argument is an integer, and every invocation of it in the
    wild passes one -- "all" is this project's word, understood here and
    nowhere else with any guarantee. A build that does not parse it does not
    quietly offload fewer layers, it refuses to start, so this is not the cause
    of a slow reply; it is one unknown removed from the list of things a slow
    reply could be. A count above the model's own is clamped by llama.cpp, so
    the fallback when the header could not be read is deliberately far too big
    rather than a guess that might be too small.
    """
    from prompt_master.inference.device_detection import NO_OFFLOAD

    if placement.gpu_layers == mc_llm_context.NO_LAYERS:
        return NO_OFFLOAD
    if placement.gpu_layers != mc_llm_context.ALL_LAYERS:
        return str(placement.gpu_layers)
    blocks = gguf.block_count if gguf is not None else 0
    # +1 for the output layer, which llama.cpp counts separately from the
    # repeating blocks and which is the one whose absence costs the most.
    return str(blocks + 1) if blocks > 0 else str(EVERY_LAYER)


EVERY_LAYER = 999
"""What to ask for when the model's own layer count could not be read."""


def _profile_arguments(configuration: Config,
                       placement: mc_llm_context.Placement) -> dict:
    """The llama-server flags a managed profile adds, and none at all without one.

    Two things arrive here that a running server cannot be told about later:
    the key/value cache types, and whether to use the chat template baked into
    the GGUF. Both are start-time arguments, which is why a profile change
    restarts the server (see :func:`_identity`).

    The cache types come off the *placement* rather than off the profile
    directly, so that whatever negotiation settled on is what llama.cpp is
    actually told -- the estimate and the command line cannot then describe two
    different caches.

    A manual install returns ``{}`` and its command line is byte-for-byte the
    one it has always had. That is deliberate rather than tidy: the Settings
    page's cache types have never been passed to llama-server, they feed the
    VRAM estimate alone, and quietly starting to honour them here would change
    both the quality and the footprint of every existing install as a side
    effect of adding a catalogue. It is worth fixing; it is not worth fixing
    without the user asking.
    """
    if configuration.profile is None:
        return {}
    return {"cache_type_k": placement.kv_type_k, "cache_type_v": placement.kv_type_v,
            "jinja": bool(configuration.profile.jinja)}


def _device_label(configuration: Config) -> str:
    """The card's name, without the numbers recorded beside it during setup.

    ``gpu_device_name`` is stored once, when the device was detected, and it
    carries a snapshot of that moment's free VRAM inside it: "NVIDIA GeForce
    RTX 3090 (24575 MiB, 23304 MiB free)". Printed months later beside a
    placement decision that was made from a live reading, it is not merely
    stale -- it is the number a person debugging a placement will believe,
    sitting in the same sentence as the placement it appears to explain. So the
    name is kept and the parenthetical is dropped, and the line that used to
    carry it now says what was actually free when the decision was made.
    """
    name = configuration.device_name or configuration.device
    head, _, _ = name.partition(" (")
    return head.strip() or configuration.device


def _identity(configuration: Config, projector=None) -> tuple:
    """Everything the user chose, as opposed to everything the card decided.

    The split this returns is the point. A server has to be restarted when the
    *settings* behind it change -- a different model, another device, a context
    the user typed -- and must not be restarted merely because the arithmetic
    that reads free VRAM came back with a slightly different answer than it did
    a minute ago. Only the first list is here; the second is the negotiated
    placement, and :func:`_worth_restarting` decides how much of a difference
    there has to be in that before a running server is given up.
    """
    return (str(configuration.runtime), str(configuration.model), str(projector),
            int(configuration.gpu_index), str(configuration.device),
            str(configuration.gpu_layers), int(configuration.context_size),
            str(configuration.context_mode), str(configuration.kv_type_k),
            str(configuration.kv_type_v),
            # The profile is a *choice* even though nobody typed it: switching
            # backbones changes the template flag and the samplers, and two of
            # those are command-line arguments a running server cannot be told
            # about. Without this, a switch between two models that happened to
            # want the same context and cache types would have reused the
            # server that was already up -- still holding the previous weights.
            str(configuration.profile_id))


def _offloaded_layers(placement: mc_llm_context.Placement, total: int) -> int:
    """``placement``'s layer count as a number two placements can be compared by."""
    if not placement.on_gpu:
        return 0
    if placement.gpu_layers == mc_llm_context.ALL_LAYERS:
        return total if total > 0 else 1 << 30
    return max(int(placement.gpu_layers), 0)


def _spilled_experts(placement: mc_llm_context.Placement, total: int) -> int:
    """``placement``'s expert split as a number two placements can be compared by."""
    layers = int(placement.cpu_expert_layers)
    if layers == mc_llm_context.ALL_EXPERTS:
        return total if total > 0 else 1 << 30
    return max(layers, 0)


def _worthwhile_layer_gain(total_layers: int) -> int:
    """The fewest extra blocks on the card that justify stopping a working server.

    A quarter of the model, never fewer than :data:`MINIMUM_LAYER_GAIN`. With no
    block count to go on the floor is the whole answer, which is the
    conservative direction: an unknown model is not a reason to restart for one
    block.
    """
    if total_layers <= 0:
        return MINIMUM_LAYER_GAIN
    return max(int(total_layers * LAYER_UPGRADE_FRACTION), MINIMUM_LAYER_GAIN)


def _worth_restarting(current: mc_llm_context.Placement, wanted: mc_llm_context.Placement,
                      total_layers: int = 0) -> bool:
    """Whether ``wanted`` is enough of an improvement to stop a server for.

    Only ever an improvement. A running llama-server holds the VRAM it was
    given, and moving it somewhere smaller frees nothing anybody asked for --
    the image side has its own way of asking, which is :meth:`Runtime.release`.
    So a placement that would be *worse* than the one already running is not a
    reason to restart, and answering that question the other way round is
    exactly how a card with room on it ends up running two layers of a model in
    VRAM and twenty-eight in system RAM.
    """
    if wanted.on_gpu != current.on_gpu:
        return bool(wanted.on_gpu)

    here = _offloaded_layers(current, total_layers)
    there = _offloaded_layers(wanted, total_layers)
    if there != here:
        return there - here >= _worthwhile_layer_gain(total_layers)

    # Same blocks on the card, fewer of their experts in system RAM: the same
    # placement with more of the weights resident, which is the improvement this
    # ladder's finer steps exist to be able to offer back.
    spilled_here = _spilled_experts(current, total_layers)
    spilled_there = _spilled_experts(wanted, total_layers)
    if spilled_there != spilled_here:
        return spilled_there < spilled_here
    return wanted.context >= current.context * (1.0 + CONTEXT_UPGRADE_FRACTION)


@dataclass
class Report:
    """What the last load did, for the status line and the residency panel."""

    placement: mc_llm_context.Placement | None = None
    estimate: mc_llm_context.Estimate | None = None
    notes: tuple[str, ...] = ()
    observed_bytes: int = 0
    started_at: float = 0.0
    model: str = ""
    fits: bool = True
    offload: Offload = field(default_factory=Offload)
    """What llama.cpp said it did, as opposed to what it was asked to do."""


SYSTEM_RAM_MARGIN = 1.15
"""How much free system RAM a load wants, as a multiple of what it will read.

llama.cpp reads a GGUF through ``mmap``, so the file's pages go through system
RAM whether the weights end up on the card or not, and anything left on the
processor stays there. Fifteen per cent over the top of it covers the runtime,
the projector and the page cache having something else to do -- and below that
the operating system starts paging the model against itself, which is how a
load that normally takes twenty seconds takes four minutes and a reply that
should stream arrives a word at a time.
"""


def _warn_about_system_ram(configuration: Config, placement: mc_llm_context.Placement) -> None:
    """Say so when there is not enough system RAM to read the model comfortably.

    A warning and not a refusal. It is the machine's memory, the user may know
    exactly what else is using it, and llama.cpp will make an honest attempt
    either way. What is not acceptable is for it to be slow for a reason
    nothing on screen mentions -- this is the single most common cause of a
    load that appears to hang, and the one thing this extension cannot do
    anything about from inside the WebUI's own process.
    """
    try:
        import mc_memory

        free = int(mc_memory.free_ram_bytes())
        needed = int(Path(configuration.model).stat().st_size)
    except (OSError, ValueError, AttributeError):
        return
    if free <= 0 or needed <= 0 or free >= needed * SYSTEM_RAM_MARGIN:
        return
    logger.warning(
        "Model Chain: only %.1f GB of system RAM is free and the model is %.1f GB. llama.cpp "
        "reads it through the page cache whatever ends up on the card, so expect a slow load "
        "and slow replies until something else on this machine gives that memory back%s",
        free / _GB, needed / _GB,
        "" if placement.gpu_layers == mc_llm_context.ALL_LAYERS
        else " — and this placement already leaves part of the model there, which is worse")


class _StartFailed(RuntimeError):
    """A start that did not come up, with llama.cpp's own reason attached."""

    def __init__(self, message: str, out_of_memory: bool = False):
        super().__init__(message)
        self.out_of_memory = out_of_memory


START_ATTEMPTS = 3
"""How many placements to try before giving up on a start.

Only a placement that ran out of VRAM is retried, and each attempt asks for
meaningfully less than the last, so three is enough to walk a large model down
onto a card that will not take it whole -- and small enough that a start which
is failing for some other reason still fails quickly.
"""

RESIDENCY_SETTLE_ATTEMPTS = 4
RESIDENCY_SETTLE_SECONDS = 0.25
"""How long to keep asking the driver what a new server took.

A second at the outside, and only on the path where the first answer was
impossible -- an on-GPU placement that appears to be holding nothing. A start
already costs several seconds; a fifth of one to stop printing "0.0 GB VRAM"
about a model holding fourteen is cheap.
"""

OVERSPEND_TOLERANCE = 256 * 1024 * 1024
"""How far above its allowance a running server may sit before it is re-placed.

Both sides of that comparison are measurements. A quarter of a gigabyte is
larger than the noise in either and far smaller than any placement step the
ladder can offer, so a real overshoot is always acted on and a rounding error
never is.
"""

RETRY_HEADROOM = 3 * _GB
"""How much more room each retry leaves. Large enough to change the placement:
a step that only trimmed the context would ask the driver for the same
allocation that had just been refused."""


def _signature_of(configuration: Config, projector, placement) -> tuple:
    """What a running server would have to be restarted to change.

    The expert split is in it because it is a start-time argument like the
    other two: a placement that moved two more blocks' experts is a different
    command line, and a signature that could not tell the two apart would hand
    back the server that had just run out of memory.
    """
    return (configuration.runtime, configuration.model, projector,
            configuration.gpu_index, configuration.device,
            placement.context, placement.gpu_layers, placement.cpu_expert_layers)


def _next_expert_floor(placement) -> int:
    """The expert split to insist on after ``placement`` ran out of memory.

    Two more blocks than the attempt that failed -- see
    :data:`EXPERT_RETRY_STEP` -- and nothing at all when there were no experts
    moved yet, because the added headroom will have the ladder compute a real
    number for itself. A placement that had already moved every expert has
    nothing left to give at this rung and says so by keeping the sentinel.
    """
    layers = int(getattr(placement, "cpu_expert_layers", mc_llm_context.NO_EXPERTS))
    if layers == mc_llm_context.ALL_EXPERTS:
        return mc_llm_context.ALL_EXPERTS
    if layers <= mc_llm_context.NO_EXPERTS:
        return mc_llm_context.NO_EXPERTS
    return layers + EXPERT_RETRY_STEP


class Runtime:
    """One managed llama-server, placed by the broker and reclaimable by it."""

    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._signature: tuple | None = None
        self._identity: tuple | None = None
        self._placement: mc_llm_context.Placement | None = None
        self._log: tuple | None = None
        """``(path, offset)`` of the running server's slice of the log."""
        self.report = Report()

    # -- lifecycle -------------------------------------------------------- #

    def client(self, needs_vision: bool = False, reserve: int = 0):
        """A client for a ready server, started or restarted as placement requires.

        ``reserve`` is VRAM this request promises not to take: room for a
        workload the caller knows is coming. Krea Creative Mode is the caller
        that has one -- it runs the writer and then, a fraction of a second
        later, an image generation on a checkpoint that needs several
        gigabytes. Without the reserve llama.cpp sizes itself to an empty card
        and the checkpoint gets the remainder, which on a 24 GB card is the
        difference between "both fit" and "the image model does not".

        Leaving the room is very much cheaper than reclaiming it afterwards.
        :meth:`release` can only give VRAM back by stopping the server, so a
        reserve that is right costs nothing and a reserve that is missing costs
        a restart per generation.

        Mirrors ``prompt_master.inference.service.InferenceService.client`` --
        the same validation, the same signature comparison, the same readiness
        wait -- and differs in exactly one way, which is the reason it is not
        simply called: the context size and offload this starts the server with
        come from :func:`negotiate` rather than from the state file, because
        the state file does not know what else is on the card.

        A server that is already up and was started from the settings in force
        is handed back without negotiating anything at all. That early return
        is not an optimisation, it is the fix for the thing that made a
        conversation unusable: negotiation reads free VRAM, a running server is
        *why* free VRAM is low, and re-deciding a placement against that number
        before every message produced a different answer before every message.
        Each different answer stopped the server and started another one --
        losing llama.cpp's prompt cache, so the whole conversation was
        processed again from the first token -- and each one was placed in the
        gap left by the model it had just evicted, which is how a card holding
        thirty layers came to be running two of them. What a warm turn costs
        now is a tuple comparison and one read of the model's header.

        The check is against the *settings*, not the placement: change the
        model, the device or the context and the server is replaced, as it has
        to be. A placement that merely came out differently is only acted on
        when it is a real improvement -- see :func:`_worth_restarting` -- and
        never when it is worse, because a running server holds its VRAM either
        way and giving it a smaller share of the card helps nobody.
        """
        from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD
        from prompt_master.inference.service import CPU_READY_TIMEOUT, GPU_READY_TIMEOUT

        with self._lock:
            configuration = config()
            if not configuration.configured:
                raise NotConfigured(
                    "No local model is configured yet. Choose a GGUF and a llama.cpp runtime "
                    "in LLM Studio’s Setup mode."
                )
            for label, path in (("llama-server", configuration.runtime),
                                ("model", configuration.model),
                                ("vision projector", configuration.mmproj)):
                if path is not None and not Path(path).is_file():
                    raise RuntimeError(f"Configured {label} is missing: {path}")
            if needs_vision and configuration.mmproj is None:
                raise RuntimeError(
                    "This request carries an image, and the model running has no vision "
                    "projector. Choose one in LLM Studio’s Setup mode, or send the "
                    "request without the image; text-only fallback is disabled."
                )

            # The projector is loaded for a request that carries an image and
            # for no other. It is a gigabyte and a third of a card this model
            # is already filling, and every text-only turn was paying it. The
            # cost of the rule is one restart when a picture is finally
            # attached, which is a trade worth making the other way round.
            projector = configuration.mmproj if needs_vision else None

            ours = self.resident_bytes()
            if (self._running and self._identity == _identity(configuration, projector)
                    and not self._outgrown(configuration, ours)):
                self._touch(configuration, ours)
                return self._client(configuration)

            # Before anything is measured, and only on a path that is really
            # going to start a server. The image allocator keeps the blocks it
            # has finished with, which is free memory to this process and to no
            # other -- so it is handed back to the driver first, and the
            # placement below is decided against what llama.cpp will actually
            # be able to allocate.
            recovered = mc_broker.release_cached_vram()
            if recovered:
                logger.info("Model Chain: returned %.1f GB of cached VRAM to the driver before "
                            "placing the LLM", recovered / _GB)

            negotiated = negotiate(configuration, already_ours=ours, vision=needs_vision,
                                   extra_reserve=reserve)
            placement = negotiated.placement
            signature = _signature_of(configuration, projector, placement)

            if self._running and signature == self._signature:
                self._identity = _identity(configuration, projector)
                self._touch(configuration, ours)
                return self._client(configuration)

            self._stop_locked("making way for a new placement")

            # The card said it had room and the driver disagreed. That is not a
            # hypothetical: a 24 GB card with 22.8 GB free refuses a single
            # 17.8 GB allocation, because what a driver can give out in one
            # piece is not the same number as what it has left -- and Windows
            # is stricter about it than the arithmetic here can model. Nothing
            # this module knows can predict it, so it is *learned*, once, by
            # asking for less and trying again. Two extra attempts, each with
            # more headroom than the last, and every one of them says so.
            penalty = 0
            expert_floor = mc_llm_context.NO_EXPERTS
            for attempt in range(START_ATTEMPTS):
                if penalty or expert_floor:
                    negotiated = negotiate(configuration, already_ours=ours,
                                           extra_reserve=reserve + penalty,
                                           vision=needs_vision, expert_floor=expert_floor)
                    placement = negotiated.placement
                    signature = _signature_of(configuration, projector, placement)
                try:
                    process, observed, offload = self._launch(configuration, placement,
                                                              projector)
                    break
                except _StartFailed as failure:
                    if not failure.out_of_memory or attempt == START_ATTEMPTS - 1:
                        raise RuntimeError(str(failure)) from None
                    penalty += RETRY_HEADROOM
                    expert_floor = _next_expert_floor(placement)
                    # And hand the allocator's blocks back again. They were
                    # released before the first attempt, but a failed start
                    # takes seconds and the image side is free to re-cache in
                    # them -- which for the ``invalid device`` failure is the
                    # whole difference, because what that start could not find
                    # room for was a CUDA context of a few hundred megabytes,
                    # not the model.
                    reclaimed = mc_broker.release_cached_vram()
                    logger.warning("Model Chain: %s. Trying again with %.1f GB more headroom%s%s",
                                   failure, penalty / _GB,
                                   f" and the experts of at least {expert_floor} layers in "
                                   "system RAM" if expert_floor > 0 else "",
                                   f", having returned {reclaimed / _GB:.1f} GB of cached VRAM "
                                   "to the driver" if reclaimed else "")

            self._process, self._signature, self._placement = process, signature, placement
            self._identity = _identity(configuration, projector)
            self._record(configuration, negotiated, observed, offload)
            # Which plan this placement answers. Everything after this point
            # compares against it rather than against free VRAM, so the server
            # survives every phase of the generation it was started for.
            try:
                import mc_plan

                mc_plan.note_placement(mc_plan.current())
            except Exception:
                logger.debug("Model Chain: could not record the plan this server was "
                             "placed for", exc_info=True)
            prepared = self._client(configuration)
            _prime_prompt_cache(prepared)
            return prepared

    def _launch(self, configuration: Config, placement: mc_llm_context.Placement,
                projector=None):
        """One llama-server, started and waited for. Returns it, its VRAM and its report.

        Raises :class:`_StartFailed` carrying llama.cpp's own reason rather
        than "llama-server exited before becoming ready", which is true of
        every failed start and useful for none of them.
        """
        from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD
        from prompt_master.inference.service import CPU_READY_TIMEOUT, GPU_READY_TIMEOUT

        before = mc_broker.device_free_vram_bytes()
        layers = _layers_argument(placement, mc_gguf.describe(configuration.model))
        paths = mc_llm_paths.app_paths()
        paths.logs.mkdir(parents=True, exist_ok=True)
        log_path = paths.logs / "llama-server.log"
        # Where this start's log begins. The file is appended to across runs,
        # and what is read back afterwards has to be this run's.
        written_before = log_path.stat().st_size if log_path.exists() else 0

        logger.info(
            "Model Chain: starting llama-server — %s on %s, %s, %s token context, "
            "%.1f GB free",
            configuration.quantization or Path(configuration.model).stem,
            _device_label(configuration),
            placement.describe(),
            f"{placement.context:,}",
            before / _GB,
        )
        # Said every time, and said in full. llama-server's own log is where the
        # answer lives when a placement and a reply speed disagree, and a log
        # nobody can find is a log nobody reads. It is one line per start, and
        # starts are rare.
        logger.info("Model Chain: llama-server log — %s", log_path)
        _warn_about_an_idle_card(configuration, layers)
        _warn_about_system_ram(configuration, placement)
        self._log = (log_path, written_before)

        process = self._new_process()
        from_system_ram = (configuration.device.casefold() == CPU_DEVICE
                           or layers == NO_OFFLOAD)
        _arm_flags(accelerator_flags(configuration, placement))
        try:
            process.start(configuration.runtime, configuration.model, projector,
                          configuration.gpu_index, configuration.device, placement.context,
                          log_path, gpu_layers=layers,
                          **_profile_arguments(configuration, placement))
            process.wait_ready(CPU_READY_TIMEOUT if from_system_ram else GPU_READY_TIMEOUT)
        except Exception as exc:
            process.stop()
            said = read_failure(_text_since(log_path, written_before))
            raise _StartFailed(said.text or str(exc), said.out_of_memory) from exc

        observed = self._observed_residency(before, placement)
        return process, observed, _await_offload(log_path, written_before)

    @staticmethod
    def _observed_residency(before: int, placement: mc_llm_context.Placement) -> int:
        """How much VRAM the server that just started actually took.

        A difference of two free-VRAM readings, which is the only measurement
        available from outside another process -- and which came back as *zero*
        for a placement that llama.cpp was demonstrably running at 108 tokens a
        second on prompts, in a user's log with the clock on it. The card had
        not finished settling when the second reading was taken: the health
        endpoint answers as soon as the server will accept a request, and the
        driver's free figure is not obliged to have caught up by then.

        A zero there is not a small error. It is printed on the ready line as
        "0.0 GB VRAM" about a model holding fourteen, and it is the figure every
        later question about whether that server is overspending its allowance
        starts from.

        So a non-positive reading on a placement that is supposed to be on the
        card is retried, briefly, before it is believed.
        """
        if before <= 0:
            return 0
        observed = max(before - mc_broker.device_free_vram_bytes(), 0)
        if observed > 0 or not getattr(placement, "on_gpu", False):
            return observed
        if placement.gpu_layers == mc_llm_context.NO_LAYERS:
            return observed  # nothing was asked for; zero is the right answer

        for _attempt in range(RESIDENCY_SETTLE_ATTEMPTS):
            time.sleep(RESIDENCY_SETTLE_SECONDS)
            observed = max(before - mc_broker.device_free_vram_bytes(), 0)
            if observed > 0:
                return observed

        logger.debug("Model Chain: the card reported no change in free VRAM after "
                     "llama-server started; the residency figure falls back to the "
                     "estimate")
        return 0

    def _client(self, configuration: Config):
        """A client for the server that is up, carrying the profile's samplers.

        One helper rather than three constructions, because the three places
        this is reached from -- a warm server, a server whose placement did not
        change, and one that has just been started -- must not be able to
        disagree about which sampler fields a request goes out with. They did
        not disagree before because there were none.

        The fields are the managed profile's fixed ones and nothing else; a
        manual install builds exactly the client it always built.
        """
        from prompt_master.inference.llama_client import LlamaClient
        from prompt_master.models import managed_profiles

        return LlamaClient(f"http://127.0.0.1:{self._process.port}", self._process.api_key,
                           managed_profiles.sampler_arguments(configuration.profile))

    def _outgrown(self, configuration: Config, ours: int) -> bool:
        """Whether the card could now hold more of the model than this server does.

        Asked with ``reclaim=False``: this runs before every request, and a
        question that evicted a checkpoint to answer itself would be a worse
        bug than the degraded placement it was checking for. So it can only
        ever see room that is already free -- which is the right way round.
        A guess that goes wrong keeps the server that is running, because the
        placement it has is one that worked.

        Two things now stop the question being asked at all, and both are the
        same rule from different angles (sections 10 and 11): a placement is
        reconsidered at plan boundaries, and a normal phase transition inside
        one generation is not a plan boundary.

        The first is the image job. Halfway through a generation there is
        always memory that has just been released and is about to be taken
        again -- Stage 1's weights after sampling, a VAE between decodes, the
        gap before Stage 2 loads. Every one of those instants looks like room
        to grow into, and growing into them means stopping the server the next
        LLM call was going to reuse. Free VRAM during a generation is not an
        offer.

        The second is the plan. When the plan the running server was placed for
        is still the plan in force, its placement is by definition the one this
        module would choose again, and re-deriving it can only produce noise --
        which is exactly what a user's log shows it producing: a context of
        7168 and one of 8192 alternating across consecutive generations, each
        change a restart, each restart a lost prompt cache and a cold thirteen
        seconds of prompt evaluation.
        """
        current = self._placement
        if current is None:
            return True

        # The plan decides when there is one, and it decides both ways. Asking
        # about the image job first would be wrong in the case that matters
        # most: a user who has just switched Stage 2 off has genuinely freed
        # the room, and the first LLM call of the next generation is the one
        # that should be placed in it -- but it runs inside a host job, so a
        # host-busy check reached first would decline for the whole generation.
        overspending = False
        try:
            import mc_plan

            if mc_plan.current() is not None:
                overspending = self._overspending(ours)
                if not mc_plan.boundary_moved() and not overspending:
                    return False
                # A real boundary. Fall through and let the negotiation below
                # say whether the new plan is worth a restart; a boundary alone
                # is permission to ask, not an answer.
            elif (mc_broker.host_busy()
                  or mc_broker.active_family() == mc_broker.FAMILY_IMAGE):
                return False
        except Exception:
            logger.debug("Model Chain: could not check the plan boundary", exc_info=True)
            if mc_broker.host_busy():
                return False
        try:
            described = mc_gguf.describe(configuration.model)
            preview = negotiate(configuration, described, reclaim=False, already_ours=ours)
        except Exception:
            logger.debug("Model Chain: could not re-check the LLM placement", exc_info=True)
            return False
        if overspending:
            # Downwards, and the one direction :func:`_worth_restarting` will
            # never ask for. Its rule -- only ever an improvement -- was written
            # when a running server sat in VRAM nobody else had a claim on, and
            # moving it somewhere smaller really did free nothing anybody had
            # asked for. A plan is exactly that claim: it says how much the
            # image side needs, and a server holding more than what is left over
            # is holding memory the next pass is going to want.
            #
            # A user found this the hard way. A batch of five was planned for
            # while a llama-server placed under a batch-of-one plan held 1.4 GB,
            # and the generation died before its first step with 255 MB free on
            # the card. The placement that had been right five seconds earlier
            # was the thing in the way.
            logger.info("Model Chain: re-placing llama-server — it holds %.1f GB where the "
                        "active plan leaves %.1f GB, so it gives the difference back",
                        ours / _GB, max(self._allowance(ours), 0) / _GB)
            return True

        if not _worth_restarting(current, preview.placement,
                                 described.block_count if described else 0):
            return False
        logger.info("Model Chain: re-placing llama-server — %s now fits, where it is running %s",
                    preview.placement.describe(described.block_count if described else 0),
                    current.describe(described.block_count if described else 0))
        return True

    @staticmethod
    def _allowance(ours: int) -> int:
        """What the active plan leaves for the language model, or -1 if it says nothing."""
        try:
            import mc_plan

            if mc_plan.current() is None:
                return -1
            return mc_plan.persistent_llm_budget(ours)
        except Exception:
            logger.debug("Model Chain: could not read the LLM allowance", exc_info=True)
            return -1

    def _overspending(self, ours: int) -> bool:
        """Whether this server holds more VRAM than the active plan now allows.

        The question ``_worth_restarting`` cannot answer, because it compares
        two placements and this compares a placement with a *promise*. Both
        matter, and they point in opposite directions: one asks whether more of
        the model could be resident, the other whether it may still be.

        A tolerance, because both sides of the comparison are measurements and
        neither is exact -- the residency is a difference of two free-VRAM
        readings taken either side of a process start, and the allowance moves
        with whatever else is on the card. Restarting a server to recover a
        rounding error would be the flapping this whole change set removed.
        """
        allowance = self._allowance(ours)
        if allowance < 0:
            return False
        return ours > allowance + OVERSPEND_TOLERANCE

    def _touch(self, configuration: Config, held: int) -> None:
        """Keep a reused server at the top of the register, and say nothing.

        Reuse is the common case -- every message of a conversation after the
        first -- so it is not narrated at INFO. What it does do is declare the
        residency again, which is what stops a server that is being used every
        minute from ageing into a stale entry the image side evicts first.

        A server with nothing on the card declares nothing. The figure to hand
        is the one measured when it started, and after a demotion to system RAM
        that figure describes VRAM the process gave back -- declaring it would
        put bytes in the register that nothing is holding, and the image side
        would come looking for them.
        """
        placement = self._placement
        if placement is not None and (not placement.on_gpu
                                      or placement.gpu_layers == mc_llm_context.NO_LAYERS):
            return
        estimated = self.report.estimate.resident_bytes if self.report.estimate else 0
        mc_broker.declare(mc_broker.FAMILY_LLM, RESIDENCY_KEY, self._label(configuration),
                          held or self.report.observed_bytes or estimated,
                          rank=mc_broker.RANK_HOT)

    def _new_process(self):
        from prompt_master.inference.llama_process import LlamaProcess

        _repair_launcher()
        return LlamaProcess()

    @property
    def _running(self) -> bool:
        return self._process is not None and self._process.running

    def running(self) -> bool:
        with self._lock:
            return self._running

    def placement(self) -> mc_llm_context.Placement | None:
        """Where the running server was actually placed, or ``None``.

        The plan that was carried out, which is what a measurement taken from
        that server has to be filed under -- see :func:`speed_key`.
        """
        with self._lock:
            return self._placement if self._running else None

    def _record(self, configuration: Config, negotiated: Negotiation, observed: int,
                offload: Offload | None = None) -> None:
        self.report = Report(
            placement=negotiated.placement, estimate=negotiated.estimate,
            notes=negotiated.notes, observed_bytes=observed, started_at=time.time(),
            model=Path(configuration.model).name if configuration.model else "",
            fits=negotiated.fits, offload=offload or Offload(),
        )
        for text in negotiated.notes:
            mc_broker.note(mc_broker.FAMILY_LLM, text)

        if negotiated.placement.on_gpu and observed > 0:
            mc_broker.declare(mc_broker.FAMILY_LLM, RESIDENCY_KEY, self._label(configuration),
                              observed, rank=mc_broker.RANK_HOT)
            mc_llm_context.record_observation(configuration.model, negotiated.placement, observed)
        elif negotiated.placement.on_gpu:
            mc_broker.declare(mc_broker.FAMILY_LLM, RESIDENCY_KEY, self._label(configuration),
                              negotiated.estimate.resident_bytes, rank=mc_broker.RANK_HOT)
        else:
            mc_broker.retire(RESIDENCY_KEY)

        logger.info(
            "Model Chain: llama-server ready — %s, %s token context, %.1f GB VRAM%s",
            negotiated.placement.describe(),
            f"{negotiated.placement.context:,}",
            observed / _GB,
            "" if not negotiated.notes else f" ({'; '.join(negotiated.notes)})",
        )
        self._report_offload(negotiated)

    def _report_offload(self, negotiated: Negotiation) -> None:
        """Say what llama.cpp reported, and say plainly when it is not the plan.

        The line above this one is what was *asked for*. This is what happened,
        and the two being printed together is the point: a placement that says
        "all layers on the GPU" while llama.cpp put a third of the weights in
        system RAM is a five-fold slowdown with nothing on screen to explain
        it, and no amount of reading the placement would ever have found it.
        The likeliest cause is not llama.cpp -- it is the card being fuller
        than this extension can see, or a driver quietly spilling an
        allocation it could not fit -- so what is said is what was observed
        rather than a diagnosis.
        """
        self._report_residency(negotiated)
        offload = self.report.offload
        granted = offload.granted_context
        if granted and self.report.placement is not None \
                and granted != self.report.placement.context:
            # A build with its own fitter adjusts the context to what it thinks
            # will fit, and then the number this extension is reasoning about is
            # not the number the server is running.
            logger.info("Model Chain: llama.cpp settled on a %s token context, not the %s "
                        "asked for", f"{granted:,}", f"{self.report.placement.context:,}")
        if not offload.known:
            # Not silence. The line above this one said where the model was
            # sent; with nothing after it there is no way to tell a report that
            # said everything was fine from a report that was never read.
            logger.info("Model Chain: llama.cpp wrote no load report this run — see %s",
                        mc_llm_paths.app_paths().logs / "llama-server.log")
            return
        logger.info("Model Chain: llama.cpp reports %s", offload.describe())
        if offload.spilled and negotiated.placement.on_gpu:
            logger.warning(
                "Model Chain: %.1f GB of the weights (%.0f%%) are in system RAM, not on the "
                "card — generation will run at a fraction of the speed it would with all of "
                "them resident. The card has less room than this extension can see: check "
                "nvidia-smi for another process holding VRAM, and on Windows check that the "
                "driver's CUDA sysmem-fallback policy is not spilling the allocation",
                offload.system_bytes / _GB, offload.system_share * 100)

    RESIDENT_SHORTFALL = 0.75
    """How much of the weights have to reach the card before it counts as placed.

    Below three quarters something else decided where this model went. Not a
    tight bound on purpose: the measurement is a difference of two free-VRAM
    readings taken either side of a process start, and anything else on the
    machine moving in between is noise in it.
    """

    def _report_residency(self, negotiated: Negotiation) -> None:
        """Say when the card did not take what it was asked to take.

        The one check that works on every build of llama.cpp, because it reads
        nothing llama.cpp wrote: the placement says all thirty layers, the
        weights are seventeen gigabytes, and the card's free memory fell by
        four. Whatever the log format, whatever the fitter decided and did not
        mention, the rest of that model is being read over PCIe on every single
        token -- which is the difference between a reply that streams and one
        that arrives at five tokens a second.
        """
        estimate, placement = negotiated.estimate, negotiated.placement
        observed = self.report.observed_bytes
        if not placement.on_gpu or estimate is None or observed <= 0:
            return
        expected = estimate.weights_bytes
        if expected <= 0 or observed >= expected * self.RESIDENT_SHORTFALL:
            return
        logger.warning(
            "Model Chain: the card took %.1f GB where this placement needs %.1f GB of weights — "
            "llama.cpp has left the rest in system RAM and will read it over PCIe for every "
            "token. Something outside this extension is holding VRAM, or llama.cpp's own fitter "
            "made room for its context and compute buffers by moving weights off the card",
            observed / _GB, expected / _GB)

    def _label(self, configuration: Config) -> str:
        name = configuration.quantization or (
            Path(configuration.model).stem if configuration.model else "the LLM")
        return f"the LLM ({name})"

    # -- the broker's reclaimer ------------------------------------------- #

    def release(self, needed_bytes: int, reason: str = "") -> int:
        """Give VRAM back. Registered with the broker as the LLM family's reclaimer.

        ``needed_bytes`` is honoured in spirit rather than to the byte: a
        process either holds its allocation or it does not, and there is no
        partial surrender short of restarting somewhere else. So a request for
        any amount releases the whole of it, which is the only granularity the
        mechanism has -- and is why :func:`negotiate` works hard not to have to
        ask for image VRAM in the first place.
        """
        # Timed rather than blocking, and that is a deliberate asymmetry. The
        # two reclaim paths take their locks in opposite orders -- an LLM load
        # holds this lock and asks mc_memory for room, an image pass holds
        # nothing and asks this for room, but mc_memory's own lock sits inside
        # the first path -- so a blocking acquire here is a cycle waiting for
        # the one moment the workload lock fails to keep the two apart. Giving
        # up costs an eviction that did not happen, which the caller already
        # handles: it is reported as a shortfall and the driver spills. A
        # deadlock would cost the whole WebUI.
        if not self._lock.acquire(timeout=RELEASE_LOCK_TIMEOUT):
            logger.warning("Model Chain: the LLM runtime was busy and could not release VRAM "
                           "for %s", reason or "another workload")
            return 0
        try:
            if not self._running:
                mc_broker.retire(RESIDENCY_KEY)
                return 0
            if self._placement is not None and not self._placement.on_gpu:
                return 0  # already in system RAM; there is nothing on the card to give

            before = mc_broker.free_vram_bytes()
            if _release_mode() == RELEASE_SYSTEM_RAM:
                freed = self._restart_in_system_ram(before, reason)
                if freed:
                    return freed
            self._stop_locked(reason or "another workload needed the VRAM")
            freed = max(mc_broker.free_vram_bytes() - before, 0)
            mc_broker.note(
                mc_broker.FAMILY_LLM,
                f"stopped llama-server for {reason or 'another workload'}; the weights stay warm "
                f"in the system page cache, so restarting it re-reads from RAM, not disk"
            )
            return freed or self.report.observed_bytes
        finally:
            self._lock.release()

    def _restart_in_system_ram(self, before: int, reason: str) -> int:
        """Move the model off the card without losing the loaded server."""
        from prompt_master.inference.device_detection import NO_OFFLOAD
        from prompt_master.inference.service import CPU_READY_TIMEOUT

        configuration = config()
        if not configuration.configured:
            return 0
        try:
            self._stop_locked("moving the LLM to system RAM")
            process = self._new_process()
            paths = mc_llm_paths.app_paths()
            paths.logs.mkdir(parents=True, exist_ok=True)
            placement = (self._placement or mc_llm_context.Placement()).with_layers(
                mc_llm_context.NO_LAYERS)
            process.start(configuration.runtime, configuration.model, configuration.mmproj,
                          configuration.gpu_index, configuration.device, placement.context,
                          paths.logs / "llama-server.log", gpu_layers=NO_OFFLOAD)
            process.wait_ready(CPU_READY_TIMEOUT)
        except Exception:
            # The process is stopped before the handle to it is dropped, which
            # is the whole point of this branch and is what it was missing.
            #
            # ``start`` can succeed and ``wait_ready`` still fail -- a server
            # that came up and then died, or one that took longer than the
            # timeout. What was left behind then was a live llama-server that
            # nothing had a handle to any more: started in its own process
            # group and with no window, so it outlived the WebUI, held its
            # CUDA context and its weights, and was invisible outside Task
            # Manager. The panel said "Unloaded" -- truthfully, about the
            # runtime it knew about -- while the card stayed full and neither
            # Unload here nor Forge's own unload could reach the thing holding
            # it. The main start path has always stopped its process on this
            # failure; this one said "stopping it instead" and stopped nothing.
            _discard(process, "the LLM could not be moved to system RAM")
            logger.warning("Model Chain: could not move the LLM to system RAM; stopping it instead",
                           exc_info=True)
            return 0

        self._process = process
        self._placement = placement
        self._signature = (configuration.runtime, configuration.model, configuration.mmproj,
                           configuration.gpu_index, configuration.device,
                           placement.context, placement.gpu_layers)
        self._identity = _identity(configuration)
        mc_broker.retire(RESIDENCY_KEY)
        freed = max(mc_broker.free_vram_bytes() - before, 0)
        mc_broker.note(mc_broker.FAMILY_LLM,
                       f"moved the LLM to system RAM for {reason or 'another workload'}; it stays "
                       f"loaded and answering, more slowly")
        return freed or self.report.observed_bytes

    def resident_bytes(self) -> int:
        with self._lock:
            if not self._running or (self._placement is not None and not self._placement.on_gpu):
                return 0
            return self.report.observed_bytes or (
                self.report.estimate.resident_bytes if self.report.estimate else 0)

    def speed(self) -> tuple[float, float]:
        """``(prompt, reply)`` tokens per second for the most recent request.

        The number every other number in this module is a proxy for. A
        placement is a plan; this is what the plan produced, measured by the
        process that produced it -- and on one card, one model and one week,
        those have differed by a factor of forty while every line this
        extension wrote said "all layers on the GPU".
        """
        with self._lock:
            if self._log is None or not self._running:
                return 0.0, 0.0
            path, offset = self._log
        return read_speed(_text_since(path, offset, tail=_TIMING_TAIL))

    def speed_note(self) -> str:
        """:meth:`speed` as one clause, and kept for whoever asks for it.

        Recording it here rather than only printing it: the same measurement
        answers "how long will this take" for the progress bar and "which
        backbone is faster on this machine" for the catalogue, and both of those
        were being estimated from character counts while llama.cpp was measuring
        the real thing once per request and writing it to a log.
        """
        prompt, reply = self.speed()
        if reply <= 0:
            return ""
        remember_speed(prompt, reply, placement=self._placement)
        measured = f"llama.cpp measured {reply:.1f} tokens/s"
        return f"{measured}, prompt at {prompt:.0f} tokens/s" if prompt > 0 else measured

    def describe(self) -> str:
        return self._label(config())

    def stop(self) -> None:
        with self._lock:
            self._stop_locked("stop requested")

    def _forget_placement_plan(self) -> None:
        """Let the next start re-decide, because this one is no longer running.

        Called wherever the server stops for a reason that was not a change of
        plan -- an emergency eviction, a demotion to system RAM, a shutdown.
        Without it the plan the dead server was placed for stays recorded, and
        :func:`mc_plan.boundary_moved` goes on answering "no" about a placement
        that no longer exists.
        """
        try:
            import mc_plan

            mc_plan.note_placement(None)
        except Exception:
            logger.debug("Model Chain: could not clear the recorded placement plan",
                         exc_info=True)

    def _stop_locked(self, reason: str) -> None:
        process, self._process = self._process, None
        held = self.report.observed_bytes if self._placement is not None else 0
        self._signature, self._identity, self._placement = None, None, None
        self._log = None
        mc_broker.retire(RESIDENCY_KEY)
        self._forget_placement_plan()
        if process is None:
            return
        try:
            process.stop()
        except Exception:
            logger.warning("Model Chain: failed to stop llama-server (%s)", reason, exc_info=True)
            return
        logger.info("Model Chain: llama-server stopped — %s%s", reason,
                    f", {held / _GB:.1f} GB of VRAM released" if held else "")

    # -- status ----------------------------------------------------------- #

    def status(self) -> dict:
        """Everything the panel needs about the runtime, and nothing that costs a load."""
        with self._lock:
            configuration = config()
            return {
                "configured": configuration.configured,
                "has_runtime": configuration.runtime is not None,
                "has_model": configuration.model is not None,
                "running": self._running,
                "model": Path(configuration.model).name if configuration.model else "",
                "quantization": configuration.quantization,
                "device": configuration.device_name or configuration.device,
                "mode": configuration.mode,
                "sees": configuration.sees,
                "placement": self._placement,
                "report": self.report,
                "resident_bytes": self.resident_bytes(),
            }


runtime = Runtime()
"""The single managed server. One process per WebUI, as the standalone app had
one per window: llama.cpp serialises requests within a process anyway, and two
would double the VRAM for no concurrency the broker would allow to be used."""

mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, runtime)


def shutdown() -> None:
    """Stop the server. Called from ``on_script_unloaded``."""
    runtime.stop()


# --------------------------------------------------------------------------- #
# Servers this extension started and no longer owns
# --------------------------------------------------------------------------- #

SERVER_ALIAS = "prompt-master"
"""``--alias`` on every llama-server this extension starts.

It is set in ``prompt_master/inference/llama_process.py``, which is vendored
and not ours to edit, so the value is restated here and a test asserts the two
still agree. It is what makes one of these processes recognisable as ours in a
process list, which is the whole of how a stray is found.
"""


def _discard(process, reason: str) -> None:
    """Stop a server this module is about to lose its handle to.

    Never raises: every caller is already on a failure path and has something
    more useful to report than the failure of the cleanup. What it must not do
    is return without trying, because the handle is gone after this and nothing
    will ever be able to try again.
    """
    try:
        process.stop()
    except Exception:
        logger.warning("Model Chain: could not stop llama-server after %s; it may still be "
                       "running and holding VRAM", reason, exc_info=True)


def _own_pid() -> int:
    process = runtime._process
    inner = getattr(process, "process", None) if process is not None else None
    return int(getattr(inner, "pid", 0) or 0)


def strays() -> list[int]:
    """llama-server processes carrying our alias that this WebUI does not own.

    A stray is what is left when a server outlives the thing that started it:
    the WebUI was killed rather than closed, a start failed in a way that lost
    the handle, or a previous version of this extension dropped one. It is
    started in its own process group and with no console window, so it survives
    its parent and is invisible outside Task Manager -- and it is still holding
    a CUDA context and a model's worth of weights. What that looks like from
    the tab is a card with no free VRAM, a chip reading "Unloaded", and an
    Unload button that truthfully reports it has nothing to stop.

    Never raises, and returns nothing rather than guessing when the process
    list cannot be read: psutil may be absent, and a process belonging to
    another user answers ``AccessDenied`` rather than a command line.
    """
    try:
        import psutil
    except Exception:
        logger.debug("Model Chain: psutil is not available, so strays cannot be looked for",
                     exc_info=True)
        return []

    ours = _own_pid()
    found: list[int] = []
    try:
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if process.info["pid"] in (ours, os.getpid()):
                    continue
                name = (process.info.get("name") or "").casefold()
                if "llama-server" not in name:
                    continue
                command = process.info.get("cmdline") or []
                # The alias has to be the *value* of --alias rather than a word
                # somewhere on the line, so that a path which happens to
                # contain it cannot make somebody else's server look like ours.
                if SERVER_ALIAS not in _alias_of(command):
                    continue
                found.append(int(process.info["pid"]))
            except Exception:
                continue
    except Exception:
        logger.debug("Model Chain: could not read the process list", exc_info=True)
        return []
    return found


def _alias_of(command) -> str:
    """The ``--alias`` argument on a command line, or ""."""
    arguments = [str(part) for part in (command or [])]
    for position, part in enumerate(arguments[:-1]):
        if part == "--alias":
            return arguments[position + 1]
    return ""


def release_strays() -> tuple[int, int]:
    """Stop every stray, and say how many and how much VRAM came back.

    Deliberately not automatic. Two WebUIs sharing one card would each see the
    other's server as a stray, and a startup that quietly killed it would be a
    worse bug than the one this fixes. So it runs from Unload, where somebody
    has asked for their VRAM back and this is the only remaining thing holding
    it.
    """
    pids = strays()
    if not pids:
        return 0, 0
    try:
        import psutil
    except Exception:
        return 0, 0

    before = mc_broker.free_vram_bytes()
    stopped = 0
    for pid in pids:
        try:
            process = psutil.Process(pid)
            process.terminate()
            try:
                process.wait(10)
            except Exception:
                process.kill()
                process.wait(5)
            stopped += 1
        except Exception:
            logger.warning("Model Chain: could not stop the stray llama-server at pid %s", pid,
                           exc_info=True)
    freed = max(mc_broker.free_vram_bytes() - before, 0)
    if stopped:
        logger.info("Model Chain: stopped %d stray llama-server process(es)%s", stopped,
                    f", {freed / _GB:.1f} GB of VRAM released" if freed else "")
    return stopped, freed
