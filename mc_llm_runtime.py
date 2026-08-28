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
import mc_llm_accel
import mc_llm_context
import mc_llm_paths
import mc_llm_state
import mc_llm_vision

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
    gpu_uuid: str = ""
    """The card's UUID, recorded at setup beside ``gpu_index``.

    The identity that survives renumbering, and the reason this field exists:
    ``gpu_index`` is ``nvidia-smi``'s number for the card, Forge's device index
    is the CUDA runtime's, and the two namespaces are not the same. Comparing
    them as though they were is how a 5090's language model came to be judged
    against a 3090's image plan. See :func:`shares_the_image_card`.
    """
    gpu_name: str = ""
    """``nvidia-smi``'s name for the card, e.g. "NVIDIA GeForce RTX 5090".

    Beside :attr:`device_name` rather than replacing it, because they come from
    different places and only one of them is comparable. ``device_name`` is
    llama.cpp's own device string and carries a snapshot of that moment's free
    VRAM inside it; this is the driver's name for the hardware, in the same
    form ``torch.cuda.get_device_name`` returns -- which is what makes the two
    halves of the machine able to recognise the same card.
    """

    @property
    def card_name(self) -> str:
        """The card's model name for comparison, from whichever field has one."""
        name = self.gpu_name or self.device_name
        head, _, _ = str(name or "").partition(" (")
        return head.strip()
    mode: str = "gpu"
    source: str = "manual"
    """``"managed"`` when the model came from the catalogue, ``"manual"`` when
    the user pointed at a GGUF themselves. See :mod:`mc_llm_managed_models`."""
    managed_id: str = ""
    accelerator: str = "auto"
    memory_priority: str = "cooperative"
    """The two performance axes, resolved for this role. See :mod:`mc_llm_accel`.

    On the configuration rather than passed per call because they are settings
    like every other field here -- and because a change to either has to
    invalidate a warm server, which is a question :func:`_identity` answers by
    reading this object and nothing else.
    """
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
        from prompt_master.core.models import CPU_MODE, MIXED_MODES, normalise_mode
        from prompt_master.inference.device_detection import NO_OFFLOAD

        named = normalise_mode(self.mode)
        if named:
            object.__setattr__(self, "mode", named)
        if named in (*MIXED_MODES, CPU_MODE) and str(self.gpu_layers) != NO_OFFLOAD:
            object.__setattr__(self, "gpu_layers", NO_OFFLOAD)

    @property
    def on_gpu(self) -> bool:
        """Whether model layers are *resident* in VRAM.

        Deliberately not "uses the card" -- see :attr:`uses_cuda_compute`. Both
        mixed modes answer False here and one of them ends up holding layers
        anyway, because what this decides is how the placement is *asked for*:
        a mixed placement asks for nothing and is then given what the ladder
        finds. Every residency figure the plan and the broker reason about is
        measured afterwards rather than read from here.
        """
        from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD

        return self.device.casefold() != CPU_DEVICE and str(self.gpu_layers) != NO_OFFLOAD

    @property
    def uses_cuda_compute(self) -> bool:
        """Whether a card is named at all, resident layers or not.

        The other half of what ``on_gpu`` used to be asked to mean. Mixed
        Conservative has no model layers in VRAM and is still a CUDA
        installation: the card is on ``--device``, it takes the operations
        llama.cpp can hand it, and it holds a context and compute buffers. A
        question about the *card* has to be asked here; a question about
        *residency* is ``on_gpu``.
        """
        from prompt_master.core.models import CPU_MODE
        from prompt_master.inference.device_detection import CPU_DEVICE

        return (self.device.casefold() != CPU_DEVICE
                and str(self.mode).strip().casefold() != CPU_MODE)

    @property
    def configured(self) -> bool:
        return self.model is not None and self.runtime is not None

    @property
    def sees(self) -> bool:
        return self.mmproj is not None


def config(role: str = "") -> Config:
    """Configuration for ``role``. Never raises -- an unconfigured install
    is a state the panel renders, not an exception it handles.

    ``role`` is ``mc_llm_roles.CREATIVE``, ``mc_llm_roles.SPATIAL``, or the
    empty string for the installation's own configuration -- which is what
    Prompt Studio, Conversation, MiniMax and LLM Studio's model loading pass,
    and what a role that has not been split resolves to anyway. See
    :mod:`mc_llm_roles` for why that is inheritance rather than a copy.

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

    import mc_llm_roles

    paths = mc_llm_paths.app_paths()
    try:
        state = read_json(paths.state_file)
    except (OSError, ValueError):
        state = {}
    prefs = mc_llm_state.preferences()
    # Layered before anything is read out of either, so every line below is the
    # line it has always been and none of them has to know a role was asked for.
    if mc_llm_roles.named(role):
        overridden = (state, prefs)
        state = mc_llm_roles.layered(role, state, *overridden,
                                     keys=mc_llm_roles.STATE_FIELDS)
        prefs = mc_llm_roles.layered(role, prefs, *overridden,
                                     keys=mc_llm_roles.PREFS_FIELDS)
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

    performance = mc_llm_accel.settings(role)

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
        gpu_uuid=str(state.get("gpu_uuid", "")),
        gpu_name=str(state.get("gpu_name", "")),
        mode=str(state.get("mode", "gpu")),
        source=source,
        managed_id=managed_id,
        accelerator=performance.accelerator,
        memory_priority=performance.memory_priority,
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
    # A count the user typed is part of what was *asked for*, so it goes on the
    # placement before anything is measured -- which is what puts it on the
    # ladder, where the first rung can sell it back one cache at a time. Only
    # Automatic waits until after the fit, because Automatic is defined as
    # spending the leftovers and there are no leftovers until something fits.
    explicit = _wanted_slots()
    if explicit > 1:
        wanted = wanted.with_slots(_slots_for(configuration, wanted.with_slots(explicit)))
    estimate = mc_llm_context.estimate(configuration.model, wanted, described)

    if _fits(estimate, reserve, already_ours, card_of(configuration), configuration):
        # It fits as asked. Only now are extra caches considered, and only out
        # of what is left over -- so the answer to "how many caches" can never
        # be the reason the answer to "does it fit" changed.
        wanted, estimate = _add_slots(configuration, wanted, described, reserve,
                                      already_ours)
        return Negotiation(wanted, estimate, (), True)

    placement = wanted
    placement, estimate, given = _drop_slots(configuration, placement, described, reserve,
                                             already_ours)
    if given:
        notes.append(given)
    if _fits(estimate, reserve, already_ours, card_of(configuration), configuration):
        return Negotiation(placement, estimate, tuple(notes), True)

    placement, estimate, shrunk = _shrink_context(configuration, placement, described, reserve,
                                                  already_ours)
    if shrunk:
        notes.append(shrunk)
    if _fits(estimate, reserve, already_ours, card_of(configuration), configuration):
        return Negotiation(placement, estimate, tuple(notes), True)

    placement, estimate, offloaded = _shrink_offload(configuration, placement, described, reserve,
                                                     already_ours, expert_floor)
    if offloaded:
        notes.append(offloaded)

    fits = _fits(estimate, reserve, already_ours, card_of(configuration), configuration)
    notes.extend(_unsatisfied(configuration, placement, described))
    return Negotiation(placement, estimate, tuple(notes), fits)


PARALLEL_FLAG = "--parallel"


def _wanted_slots() -> int:
    """The slot count the user asked for, or 0 for Automatic."""
    chosen = mc_broker.resolve(mc_broker.option(OPT_LLM_SLOTS, SLOTS_AUTOMATIC),
                               SLOT_MODES, SLOTS_AUTOMATIC)
    if chosen == SLOTS_AUTOMATIC:
        return 0
    try:
        return max(min(int(chosen), SLOT_CEILING), 1)
    except (TypeError, ValueError):
        return 0


def _slots_that_fit(configuration: Config, placement: mc_llm_context.Placement,
                    gguf, reserve: int, already_ours: int = 0) -> int:
    """How many warm caches Automatic may take, which is only ever the spare.

    The rule that makes Automatic safe to leave on: **a slot is bought with
    VRAM that is left over after the placement already fits, and with nothing
    else.** It is worked out from the *un-degraded* placement -- the one the
    ladder has not touched -- so an extra cache can never be the reason a
    context shrank, an expert moved, or a layer left the card. If the model
    only just fits, the answer is one.

    That ordering is the whole trade. A cache costs one prompt re-read when it
    is missing; every other rung of the ladder costs speed on every token from
    then on. So caches are the last thing bought and, in :func:`negotiate`, the
    first thing sold.

    Capped by the number of modes that actually have a system prompt of their
    own, because a slot nothing routes to is a key/value cache doing nothing.
    """
    if not placement.on_gpu or placement.slots > 1:
        return 1
    estimate = mc_llm_context.estimate(configuration.model, placement, gguf)
    per_slot = int(estimate.kv_bytes + estimate.state_bytes)
    if per_slot <= 0:
        return 1
    spare = (_spendable(already_ours, card_of(configuration), configuration=configuration)
             - estimate.total_bytes - reserve)
    if spare <= 0:
        return 1
    return max(min(1 + int(spare // per_slot), DISTINCT_PROMPTS, SLOT_CEILING), 1)


DISTINCT_PROMPTS = 6
"""Modes that open with a system prompt of their own.

Conversation, Prompt Studio, MiniMax, the Krea writer, the Spatial Composer,
and the prompt-cache prime. Automatic never asks for more caches than there are
prompts to put in them.
"""


def _add_slots(configuration: Config, placement: mc_llm_context.Placement, gguf,
               reserve: int, already_ours: int = 0):
    """Spend leftover VRAM on warm caches. Returns ``(placement, estimate)``.

    Automatic only. A count the user typed is already on the placement by the
    time this is reached -- :func:`negotiate` puts it there before the fit, so
    that the ladder can sell it back -- and honouring it again here would grow
    past what was asked for.

    No note either way. A note is what :attr:`Negotiation.degraded` is made of,
    and gaining a warm cache is not a degradation -- how many there are is said
    on the ready line, through :meth:`mc_llm_context.Placement.describe`, where
    every other fact about the placement is already reported.
    """
    if _wanted_slots():
        return placement, mc_llm_context.estimate(configuration.model, placement, gguf)
    slots = _slots_for(configuration, placement.with_slots(
        _slots_that_fit(configuration, placement, gguf, reserve, already_ours)))
    if slots <= 1:
        return placement, mc_llm_context.estimate(configuration.model, placement, gguf)
    grown = placement.with_slots(slots)
    estimate = mc_llm_context.estimate(configuration.model, grown, gguf)
    if not _fits(estimate, reserve, already_ours, card_of(configuration), configuration):
        # Only reachable if the arithmetic and the fit disagree at the margin.
        # The placement that was known to fit wins.
        return placement, mc_llm_context.estimate(configuration.model, placement, gguf)
    return grown, estimate


def _drop_slots(configuration: Config, placement: mc_llm_context.Placement, gguf,
                reserve: int, already_ours: int = 0):
    """Sell warm caches back until the placement fits, cheapest thing first.

    The first rung, because it is the cheapest thing on the ladder to give up.
    A cache that is not there costs one prompt re-read the next time that mode
    runs; a context that is not there costs conversation length, an expert in
    system RAM costs speed on every token that consults it, and a layer that is
    not on the card costs speed on every token full stop.
    """
    if placement.slots <= 1:
        return placement, mc_llm_context.estimate(configuration.model, placement, gguf), ""
    started = placement.slots
    while placement.slots > 1:
        smaller = placement.with_slots(placement.slots - 1)
        estimate = mc_llm_context.estimate(configuration.model, smaller, gguf)
        placement = smaller
        if _fits(estimate, reserve, already_ours, card_of(configuration), configuration):
            break
    estimate = mc_llm_context.estimate(configuration.model, placement, gguf)
    return placement, estimate, (f"warm prompt caches reduced from {started} to "
                                 f"{placement.slots} to make the model fit; the modes that "
                                 f"share a cache will re-read each other's prompts")


def _unsatisfied(configuration: Config, placement, gguf) -> list[str]:
    """Say so when the mode the user chose could not be carried out.

    Section 21: "GPU-only placement does not fit: report the inability to
    satisfy the selected mode rather than silently converting it." The ladder
    below is the same ladder for every placement, and for the mixed modes its
    degradations *are* the mode -- Aggressive is defined as "take what fits".
    GPU / VRAM Only is not. It is a statement that the model belongs on the
    card, and a run that quietly put it in system RAM answered a question the
    user had already answered.

    A sentence rather than a refusal, because the fallback contract everywhere
    else in this module is to degrade and report, and a generation that stops
    dead because a card filled up is worse than a slow one that says why. What
    it must not do is stay quiet.
    """
    from prompt_master.core.models import GPU_MODE, normalise_mode

    if normalise_mode(configuration.mode) != GPU_MODE:
        return []
    if placement.gpu_layers == mc_llm_context.ALL_LAYERS:
        return []
    total = gguf.block_count if gguf is not None else 0
    where = ("none of it is on the card" if placement.gpu_layers == mc_llm_context.NO_LAYERS
             else f"only {placement.gpu_layers} of {total or '?'} layers are")
    return [f"GPU / VRAM Only was chosen and could not be satisfied — {where}. "
            f"{_who_filled_the_card(configuration)}Free VRAM on this card, pick a smaller "
            f"quantization, or choose Mixed Aggressive, which is this fallback as a "
            f"deliberate setting"]


def _who_filled_the_card(configuration: Config) -> str:
    """Name our own other servers when *they* are what left no room.

    "Free VRAM on this card" is advice for a card somebody else filled. It is
    unactionable, and slightly insulting, when what filled it is two more copies
    of the same model this extension started a minute ago -- which is what a
    user's log showed: a conversation holding twenty gigabytes while the two
    roles took turns in the eleven that were left, at four tokens a second.

    So the sentence says which, and names the setting that collapses them into
    one. It costs a registry walk on a path that has already read a GGUF header
    and started a process, and it is only ever reached when a placement has
    already been degraded.
    """
    card = card_of(configuration)
    if card is None:
        return ""
    try:
        held = 0
        servers = 0
        for found in registry.running(card=card):
            bytes_held = _held_by(found, card)
            if bytes_held <= 0:
                continue
            held += bytes_held
            servers += 1
        if servers <= 0 or held <= 0:
            return ""
    except Exception:
        logger.debug("Model Chain: could not ask what else of ours is on the card",
                     exc_info=True)
        return ""
    return (f"{held / _GB:.1f} GB of it is held by {servers} other llama-server"
            f"{'' if servers == 1 else 's'} this extension started — under "
            f"“When Creative and Spatial are configured identically”, "
            f"“{mc_broker.label_for(PROCESS_MODES, PROCESSES_SHARED)}” gives identically "
            f"configured roles one server and one copy of the weights. ")


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
            _spendable(already_ours, card_of(configuration), configuration=configuration),
            weights,
            mc_broker.safety_margin_bytes())
        sized = mc_llm_context.context_for_budget(configuration.model, placement, budget)
        if sized >= MINIMUM_CONTEXT:
            placement = placement.with_context(sized)

    return _capped(placement, gguf)


def _mode_is(configuration: Config | None, wanted: str) -> bool:
    """Whether ``configuration`` is in ``wanted``, with a card to use."""
    from prompt_master.core.models import normalise_mode
    from prompt_master.inference.device_detection import CPU_DEVICE

    try:
        configuration = configuration or config()
    except Exception:
        return False
    return (normalise_mode(configuration.mode) == wanted
            and str(configuration.device).strip().casefold() != CPU_DEVICE)


def is_mixed(configuration: Config | None = None) -> bool:
    """Whether this installation is on Mixed **Aggressive** with a card to use.

    The mode that asks the ladder for everything and takes what it is given.
    Conservative is a card too and answers False here on purpose: it asks for
    nothing, so the lift to ``ALL_LAYERS`` in :func:`_requested_placement` --
    the whole of what this function gates -- must not happen for it.
    """
    from prompt_master.core.models import MIXED_AGGRESSIVE_MODE

    return _mode_is(configuration, MIXED_AGGRESSIVE_MODE)


def is_conservative(configuration: Config | None = None) -> bool:
    """Whether this installation keeps every model layer out of VRAM.

    Zero resident layers, by the user's choice rather than by a shortfall, and
    the card still named so llama.cpp can use it. Nothing negotiates this away:
    a placement ladder that promoted layers into VRAM because VRAM happened to
    be free would be answering a question the user has already answered.
    """
    from prompt_master.core.models import MIXED_CONSERVATIVE_MODE

    return _mode_is(configuration, MIXED_CONSERVATIVE_MODE)


def _capped(placement: mc_llm_context.Placement,
            gguf: mc_gguf.Gguf | None) -> mc_llm_context.Placement:
    """Never ask for more context than the model itself declares (section 12)."""
    if gguf is None or not gguf.context_length:
        return placement
    if placement.context <= gguf.context_length:
        return placement
    return placement.with_context(gguf.context_length)


def _free_vram(already_ours: int = 0, card: int | None = None) -> int:
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
    return mc_broker.device_free_vram_bytes(card) + max(int(already_ours), 0)


def card_of(configuration: Config) -> int | None:
    """Which CUDA index ``configuration`` will place against, or None.

    None for a processor-only installation, where there is no card to ask
    about and every VRAM figure is beside the point.
    """
    if not configuration.uses_cuda_compute:
        return None
    try:
        return int(configuration.gpu_index)
    except (TypeError, ValueError):
        return None


def execution_domain(configuration: Config | None) -> mc_broker.ExecutionDomain:
    """Which processor ``configuration`` will materially execute on (section 4.1).

    Built from :attr:`Config.uses_cuda_compute`, never from :attr:`Config.on_gpu`,
    and the difference is the whole reason both properties exist. Mixed
    Conservative keeps every weight in system RAM and still names a card on
    ``--device``: it holds a CUDA context there, it issues CUDA work there, and
    an image generation on that same card is competing with it for the
    processor even though its VRAM residency is nil. Asking ``on_gpu`` would
    have called that runtime independent and let the two fight.

    Three answers, not two:

    ``CPU_EXECUTION``
        no CUDA device is named. Positive evidence of independence from every
        image generation, which is what lets a processor-resident conversation
        run while Forge samples (invariant I-1).
    ``cuda_execution(n)``
        a resolved physical card. Independent of an image job on any other.
    ``UNKNOWN_CUDA_EXECUTION``
        CUDA, card unresolvable. Conservative against every CUDA domain --
        false serialisation costs throughput, and being wrong the other way
        means two jobs on one card each sized as though alone (section 6.4).
    """
    if configuration is None:
        return mc_broker.UNKNOWN_CUDA_EXECUTION
    try:
        if not configuration.uses_cuda_compute:
            return mc_broker.CPU_EXECUTION
    except Exception:
        logger.debug("Model Chain: could not read the runtime's compute device",
                     exc_info=True)
        return mc_broker.UNKNOWN_CUDA_EXECUTION
    # ``getattr`` because this is also reached with stand-ins for a
    # configuration -- a role's settings object in a test, a partially built
    # one during setup. A stand-in that cannot name its card should fall back
    # to the index it does have, not to "unknown", which would put every such
    # caller behind the conservative wait.
    return mc_broker.cuda_execution(card_of(configuration),
                                    uuid=getattr(configuration, "gpu_uuid", ""),
                                    name=getattr(configuration, "card_name", ""))


def host_ram_demand(configuration: Config | None,
                    placement: mc_llm_context.Placement | None = None) -> int:
    """Host RAM this placement materially needs, as an estimate (section 10.7).

    Coarse on purpose, and it says so in the log rather than pretending to
    measure physical residency. What it must get *right* is the shape:

    * a processor placement, or Mixed Conservative, needs the model in system
      RAM. That is a real, model-size-scale claim on the host pool and is
      counted in full;
    * a partial offload needs the share that is not on the card. Counted in
      proportion to the layers left behind, which is coarse and is far closer
      than either extreme;
    * a full-GPU placement needs *load headroom*, not a permanent reservation.
      Its GGUF is read through mmap and the pages the OS keeps afterwards are
      the OS's to reclaim, so counting the file size as hard host residency
      would block image work that is perfectly safe -- section 10.7 and test
      T41 are both explicit about this, and it is why this returns zero there.

    Zero when the model cannot be sized. An unknown demand admits nothing and
    reclaims nothing, which is the only honest answer.
    """
    if configuration is None or configuration.model is None:
        return 0
    try:
        size = int(Path(configuration.model).stat().st_size)
    except (OSError, ValueError, TypeError):
        return 0
    if size <= 0:
        return 0

    if not configuration.uses_cuda_compute or is_conservative(configuration):
        return size

    if placement is None:
        # No placement decided yet on a CUDA configuration whose mode is not
        # Conservative: the weights are going to the card, so the load-headroom
        # case applies rather than a permanent host claim.
        return 0
    if not placement.on_gpu or placement.gpu_layers == mc_llm_context.NO_LAYERS:
        # Named a card and placed nothing on it. Every weight is in system RAM
        # and every byte of it is material host demand.
        return size
    if placement.gpu_layers == mc_llm_context.ALL_LAYERS:
        return 0
    try:
        described = mc_gguf.describe(configuration.model)
        total = int(getattr(described, "block_count", 0) or 0)
    except Exception:
        return 0
    placed = int(placement.gpu_layers)
    if total <= 0 or placed <= 0 or placed >= total:
        return 0
    return int(size * (total - placed) / total)


def shares_the_image_card(card: int | None, configuration: Config | None = None) -> bool:
    """Whether a placement on ``card`` is competing with the image model.

    The question the plan's protection is really about. Every figure in
    :mod:`mc_plan` is about the card Forge is generating on, and a role pinned
    to a different one neither takes from that budget nor is limited by it --
    which is the whole reason somebody puts a second card in the machine.

    ``configuration`` carries the card's UUID and name, and passing it is what
    makes the answer trustworthy. Without it this compares indices, and an
    index is only meaningful inside the namespace that issued it: ``gpu_index``
    is ``nvidia-smi``'s, Forge's is the CUDA runtime's, and on the machine that
    reported this both said "0" about different cards. Every caller that has a
    configuration passes it.

    Unanswerable questions are answered *yes*, and deliberately: treating an
    unknown card as the image card keeps the placement conservative, and the
    cost of being wrong that way is a smaller language model rather than an
    image generation that runs out of memory. What has changed is how rarely
    the question is unanswerable -- a UUID settles it outright, and two
    different card names settle it in the negative even when no index can be
    translated at all.
    """
    if card is None and configuration is None:
        return True
    try:
        mine = mc_broker.cuda_execution(
            card,
            uuid=getattr(configuration, "gpu_uuid", "") if configuration else "",
            name=getattr(configuration, "card_name", "") if configuration else "")
        image = mc_broker.image_execution_domain()
    except Exception:
        logger.debug("Model Chain: could not ask which card the image side is on",
                     exc_info=True)
        return True
    if mine.uuid and image.uuid:
        return mine.uuid == image.uuid
    if mine.name and image.name and mine.name != image.name:
        return False
    if not mine.known or not image.known:
        return True
    return mine.card == image.card


def _spendable(already_ours: int = 0, card: int | None = None, *,
               image_budget: bool = True, configuration: "Config | None" = None) -> int:
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
    free = _free_vram(already_ours, card)
    if not image_budget:
        # LLM priority, on the card the user gave it priority on. The plan's
        # budget is the *image* side's reservation, and overriding that
        # reservation on one card is the entire content of what was asked for
        # -- capping by it here would make the setting unable to do the one
        # thing it says it does. The learned cap below is not lifted with it:
        # that one records an allocation the driver actually refused, and no
        # amount of permission makes a refused allocation succeed.
        try:
            import mc_plan

            learned = (mc_plan.learned_cap_bytes()
                       if shares_the_image_card(card, configuration) else 0)
        except Exception:
            return free
        return max(min(free, learned) if learned > 0 else free, 0)
    try:
        import mc_plan

        # Only an *active plan* caps anything. With none published there is no
        # image workload to protect, and the budget arithmetic would then return
        # the whole card -- measured a second way, from a second source, and
        # therefore very slightly different from the reading above. Capping one
        # measurement of the card by another measurement of the same card is not
        # a policy, it is a rounding error with the power to move experts into
        # system RAM.
        # The plan protects one card. A placement on another one is outside
        # what it is protecting, so its budget is not this placement's ceiling
        # -- capping here is what made a role on an idle second card negotiate
        # itself down to the leftovers of the card it was never going to touch.
        here = shares_the_image_card(card, configuration)
        budget = (mc_plan.persistent_llm_budget(already_ours)
                  if mc_plan.current() is not None and here else -1)
        learned = mc_plan.learned_cap_bytes() if here else 0
    except Exception:
        logger.debug("Model Chain: could not read the active plan's budget", exc_info=True)
        return free

    # -1 is "there is nothing to divide up", not "there is no room".
    if budget >= 0:
        free = min(free, budget)
    if learned > 0:
        free = min(free, learned)
    return max(free, 0)


def _fits(estimate: mc_llm_context.Estimate, reserve: int, already_ours: int = 0,
          card: int | None = None, configuration: "Config | None" = None) -> bool:
    return (_spendable(already_ours, card, configuration=configuration)
            >= estimate.total_bytes + reserve)


def _shrink_context(configuration: Config, placement, gguf, reserve: int,
                    already_ours: int = 0):
    """Lower the context until the cache fits what is free, or hit the floor."""
    free = _spendable(already_ours, card_of(configuration), configuration=configuration)
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
    free = _spendable(already_ours, card_of(configuration), configuration=configuration)
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
                _supersede(notes)
                note = ("the whole model was placed in system RAM -- there is not enough free "
                        "VRAM for any of it alongside what is resident; generation will be slow")
            notes.append(note)
            return candidate, estimate, "; ".join(notes)

    candidate = placement.with_layers(mc_llm_context.NO_LAYERS)
    _supersede(notes)
    notes.append("the whole model was placed in system RAM; generation will be slow")
    return (candidate, mc_llm_context.estimate(configuration.model, candidate, gguf),
            "; ".join(notes))


def _supersede(notes: list[str]) -> None:
    """Drop the rungs that a landing in system RAM has just contradicted.

    Which is all of them. Every rung above the last describes a saving made by
    leaving *part* of the model behind, and the last rung leaves all of it
    behind: reporting both produced the sentence a user sent in, which said in
    one breath that "the experts stay in system RAM and the rest of the model is
    on the GPU" and that "the whole model was placed in system RAM". Both halves
    were written by this function, the second one is the true one, and somebody
    reading the first half would go looking for a partial offload that was never
    there -- as, in that report, they did.
    """
    notes.clear()


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
class Speculation:
    """How much of the draft was accepted, when llama.cpp said.

    The number that decides whether speculative decoding was worth having.
    Drafted tokens cost a forward pass through the draft heads whether or not
    they are kept; accepted ones are full forward passes that did not have to
    happen. A low acceptance rate is an accelerated run that is slower than an
    ordinary one, and it is invisible without this.

    Absent rather than zero when the log said nothing, which is a distinction
    the panel needs: "no drafted tokens were accepted" and "this build does not
    report acceptance" are not the same news.
    """

    drafted: int = 0
    accepted: int = 0
    known: bool = False

    @property
    def acceptance(self) -> float:
        """Accepted over drafted, or zero when nothing was drafted."""
        return self.accepted / self.drafted if self.drafted > 0 else 0.0

    def describe(self) -> str:
        if not self.known or self.drafted <= 0:
            return ""
        return (f"{self.accepted:,} of {self.drafted:,} drafted tokens accepted "
                f"({self.acceptance * 100:.0f}%)")


_SPECULATION = re.compile(
    r"n_draft(?:ed)?\s*[=:]\s*(\d+)[^\n]*?n_accept(?:ed)?\s*[=:]\s*(\d+)"
    r"|draft\s+acceptance[^\n]*?(\d+)\s*/\s*(\d+)",
    re.IGNORECASE)
"""Both shapes llama.cpp has printed speculative counters in.

Two alternatives rather than one, and neither is load-bearing: a build that
prints a third shape reports :attr:`Speculation.known` as False, the panel says
the runtime did not report acceptance, and nothing else changes. Parsing a log
is the only way to get this figure from another process, and a parser that
*failed loudly* when a log format moved would break generation for a statistic.
"""


def read_speculation(text: str) -> Speculation:
    """Drafted and accepted token counts from a slice of llama-server's log."""
    found = None
    for match in _SPECULATION.finditer(text or ""):
        found = match
    if found is None:
        return Speculation()
    if found.group(1) is not None:
        return Speculation(int(found.group(1)), int(found.group(2)), True)
    return Speculation(int(found.group(4)), int(found.group(3)), True)


@dataclass(frozen=True)
class Failure:
    """Why a start failed, and whether a smaller placement would help."""

    text: str = ""
    out_of_memory: bool = False
    bad_argument: str = ""
    """The option llama.cpp refused, when it died at argument parsing.

    A different kind of failure from every other one here, and the difference
    is who is at fault. Out of memory is a fact about the machine and the
    answer is to ask for less of it; an argument llama.cpp will not take is a
    fact about *this extension* -- a flag added on the strength of a help text
    that said less than it seemed to -- and the answer is to stop adding it.
    Nothing about the model is wrong, so nothing about the model should be
    given up over it."""
    bad_value: str = ""
    """The value it refused, where the message named one."""

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

NO_MMAP_FLAG = "--no-mmap"
"""Read the weights into memory instead of mapping the file.

Added for one placement and no other: the one that overrides some tensors to
the processor while the rest stay on the card. llama.cpp warns about that
combination itself -- ``tensor overrides to CPU are used with mmap enabled --
consider using --no-mmap for better performance`` -- because an overridden
tensor is reached through the page cache on every token rather than out of a
buffer of its own.

It is not added to a placement that is entirely in system RAM. That is
llama.cpp's ordinary processor path, mapping is its default there for good
reasons, and the flag would trade a slower start for nothing.
"""

NO_KV_OFFLOAD_FLAG = "--no-kv-offload"
"""Keep the key/value cache in system RAM rather than on the card.

Mixed Conservative's, and only its. The mode's promise is that nothing of this
model persists in VRAM, and a KV cache is the one thing that would grow there
all through a generation while the layer count sat honestly at zero -- an 8k
context of f16 cache on a 26B model is over a gigabyte of exactly the VRAM the
image plan was told it could have.
"""

OP_OFFLOAD_FLAG = "--op-offload"
"""Let the card execute host-tensor operations it supports.

The half of Mixed Conservative that makes it faster than the processor alone:
no weights move, but the multiplications can still happen on the card. Passed
only when the build advertises this exact spelling. Several builds enable it by
default and expose only ``--no-op-offload`` to turn it off, and on those
``runtime_supports`` answers False and nothing is passed -- which is the right
answer, because the behaviour is already what this mode wants.
"""

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

OPTIONAL_FLAGS = frozenset({
    CPU_MOE_FLAG, N_CPU_MOE_FLAG, FLASH_ATTENTION_FLAG, NO_MMAP_FLAG,
    NO_KV_OFFLOAD_FLAG, OP_OFFLOAD_FLAG, FULL_ATTENTION_WINDOW_FLAG,
    mc_llm_accel.SPEC_TYPE_FLAG, mc_llm_accel.SPEC_MAX_FLAG,
})
"""Every flag this extension *chooses* to append, and none that it must.

The list matters because it decides who is blamed for a start that died at
argument parsing. A flag on this list being refused means the help text
promised more than the build delivers, the flag can simply be left off, and
nothing about the model is wrong. ``--device``, ``--model`` and the rest of the
vendored launcher's fixed line are deliberately absent: those are not optional,
a refusal of one is a real misconfiguration, and each already has a diagnosis
of its own that says far more than this branch could.
"""


_BAD_ARGUMENT = re.compile(
    r'error while handling argument "(--[a-z0-9-]+)"\s*:\s*([^\n]+)', re.IGNORECASE)
_BAD_VALUE = re.compile(r"unknown [a-z ]*type:\s*(\S+)", re.IGNORECASE)
"""llama.cpp's own words when it will not take an option, and the value it named.

Both are best-effort and neither is load-bearing: a message these do not match
is an ordinary start failure, reported as one. What they buy is the difference
between "this backbone would not start" -- which is what the user saw, and is
false -- and "this flag was wrong", which is actionable and true.
"""

_HELP_TIMEOUT = 20
_capabilities: dict[tuple, str] = {}
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
    text = _capability_text(configuration)
    if not text:
        return frozenset()
    return frozenset(re.findall(r"(--[a-z0-9][a-z0-9-]+)", text))


def _runtime_stamp(configuration: Config | None = None):
    """``(path, mtime)`` for the executable in force, or ``None``.

    The cache key everything in this section is filed under. Adopting a new
    build changes the modification time, so a replaced runtime asks again
    rather than inheriting what the previous one said.
    """
    try:
        configuration = configuration or config()
        executable = configuration.runtime
        if executable is None or not Path(executable).is_file():
            return None
        return (str(executable), Path(executable).stat().st_mtime)
    except Exception:
        return None


def _capability_text(configuration: Config | None = None) -> str:
    """``llama-server --help``, whole, cached per executable.

    The text rather than the parsed flag set, because two questions are asked
    of it now: which long options exist, and which *values* an enumerated
    option lists. Parsing it twice from one subprocess run is free; running the
    subprocess twice is twenty seconds on a cold start.
    """
    stamp = _runtime_stamp(configuration)
    if stamp is None:
        return ""
    with _capabilities_lock:
        found = _capabilities.get(stamp)
    if found is not None:
        return found

    found = _read_capabilities(stamp[0])
    with _capabilities_lock:
        _capabilities[stamp] = found
    return found


def _read_capabilities(executable) -> str:
    """``llama-server --help``, as one string. Empty when it cannot be asked."""
    try:
        finished = subprocess.run(
            [str(executable), "--help"], capture_output=True, text=True,
            timeout=_HELP_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        logger.debug("Model Chain: could not ask llama-server what it supports",
                     exc_info=True)
        return ""
    text = f"{finished.stdout or ''}\n{finished.stderr or ''}"
    logger.debug("Model Chain: llama-server's help is %d characters", len(text))
    return text


def runtime_supports(flag: str, configuration: Config | None = None) -> bool:
    """Whether this build advertises ``flag``. False when it cannot be asked."""
    return flag in runtime_capabilities(configuration)


_rejected: dict[tuple, set] = {}
"""Enumerated values a build printed in its help and then refused at startup.

Session-scoped, keyed the way :data:`_capabilities` is. It exists because the
help text can be *wrong by omission*: an option's usage line lists the values
that build takes, and a build whose list this module misread -- or which takes
the value only in combination with something else -- rejects it at startup and
exits. Without this, every subsequent start would make the same doomed attempt
and pay the same three seconds for it.

Not persisted. Relearning it once per session costs one failed start, and a
file on disk claiming a runtime cannot do something would outlive the runtime
being replaced.
"""


def runtime_accepts(flag: str, value: str, configuration: Config | None = None) -> bool:
    """Whether this build takes ``value`` for ``flag``, as far as it will say.

    A second question, and the one whose absence cost a user their model
    switch. ``--spec-type`` has been in llama.cpp since the speculative
    framework landed, so :func:`runtime_supports` answers yes for it on every
    build in the wild; *which types* it accepts is an enumeration that has
    grown release by release, and a build handed one it does not have prints
    ``error while handling argument "--spec-type": unknown speculative type``
    and exits before it loads a single tensor.

    Matched as a whole word anywhere in the help output rather than by parsing
    the option's usage block, which is a deliberate looseness: llama.cpp has
    formatted that block three ways, and the values this is asked about --
    ``draft-mtp`` -- is distinctive enough that finding it anywhere in the text
    means the build knows it. A false negative costs an
    accelerator that would have worked and is reported; a false positive costs
    a failed start, which is why the negative cache below exists too.
    """
    if not runtime_supports(flag, configuration):
        return False
    stamp = _runtime_stamp(configuration)
    if stamp is not None and (flag, value) in _rejected.get(stamp, ()):
        return False
    text = _capability_text(configuration)
    if not text:
        return False
    return bool(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(value)}(?![A-Za-z0-9_-])", text))


def runtime_refused(flag: str, value: str, configuration: Config | None = None) -> bool:
    """Whether this build has already refused ``value`` for ``flag`` at startup.

    The negative cache alone, and nothing about the help text. It is the right
    question for a flag whose argument is a *number*, where
    :func:`runtime_accepts` is the wrong one: that function decides by finding
    the value as a whole word in ``--help``, which is exactly right for an
    enumeration like ``--spec-type draft-mtp`` and nonsense for ``--parallel
    6`` -- llama.cpp's help has no reason to contain a bare "6", so every count
    reads as unsupported.

    That is not hypothetical. It shipped: a user's log said "this llama.cpp
    build refused --parallel 6 on an earlier start" while their llama-server
    log showed no such start had ever been attempted, and the feature was
    disabled by a sentence about something that never happened.
    """
    stamp = _runtime_stamp(configuration)
    return stamp is not None and (flag, str(value)) in _rejected.get(stamp, ())


def note_rejected_value(flag: str, value: str, configuration: Config | None = None) -> None:
    """Record that this build refused ``value`` for ``flag`` when it started.

    Called from the one place that can know it -- a start that died on an
    argument error -- so the next start does not repeat it.
    """
    stamp = _runtime_stamp(configuration)
    if stamp is None:
        return
    with _capabilities_lock:
        _rejected.setdefault(stamp, set()).add((flag, value))
    logger.warning("Model Chain: this llama-server does not accept %s %s, whatever its help "
                   "text lists; it will not be asked for it again this session", flag, value)


def rejected_values(configuration: Config | None = None) -> tuple:
    """What this build has refused, for a status line and for the tests."""
    stamp = _runtime_stamp(configuration)
    return tuple(sorted(_rejected.get(stamp, ()))) if stamp is not None else ()


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
    flags.extend(conservative_flags(configuration))
    if not getattr(placement, "on_gpu", False):
        return flags
    experts = expert_flags(configuration, placement)
    flags.extend(experts)
    if experts and runtime_supports(NO_MMAP_FLAG, configuration):
        flags.append(NO_MMAP_FLAG)
    if placement.gpu_layers == mc_llm_context.NO_LAYERS:
        return flags
    if runtime_supports(FLASH_ATTENTION_FLAG, configuration):
        flags.append(FLASH_ATTENTION_FLAG)
        if _flash_attention_takes_a_value(configuration):
            flags.append("on")
    return flags


# --------------------------------------------------------------------------- #
# Acceleration: what is available, then whether it fits
# --------------------------------------------------------------------------- #
#
# Two stages, and they are separated because they cost different things and are
# asked at different moments.
#
# The *choice* is a question about files and records -- does this backbone
# advertise an accelerator at all. Nothing in it reads free VRAM, so its answer
# does not change between two requests a second apart, and that stability is
# what lets it go into the warm server's identity: a warm turn compares it and
# reuses the server, exactly as it compares the model and the device.
#
# The *plan* is the fit, and it reads the card. Section 9 asks for it only on a
# path that is really going to start something, for the same reason
# ``Runtime.client`` stopped re-negotiating placements on warm turns: a number
# that is different every time it is read will restart a server every time it
# is read.


def accelerator_choice(configuration: Config,
                       needs_vision: bool = False) -> mc_llm_accel.Plan:
    """Which mechanism this configuration can use, before any VRAM is measured.

    A question about *files and records* -- does the catalogue entry advertise
    anything at all -- and nothing in it reads free VRAM, so its answer does not
    change between two requests a second apart. That stability is what lets it
    into the warm server's identity.

    ``needs_vision`` is accepted and unused. Multi-token heads are in the GGUF
    and are as available to an image request as to a text one; the parameter
    stays because the accelerator this module used to have was gated on it, and
    because a caller asking "what can this request use" should not have to know
    which of the two it is asking about.
    """
    requested = mc_llm_accel._named(configuration.accelerator, mc_llm_accel.ACCELERATORS,
                                    mc_llm_accel.ACCEL_AUTO)
    priority = mc_llm_accel._named(configuration.memory_priority, mc_llm_accel.PRIORITIES,
                                   mc_llm_accel.PRIORITY_COOPERATIVE)
    ordinary = mc_llm_accel.Plan(requested=requested, accelerator=mc_llm_accel.ACCEL_NONE,
                                 memory_priority=priority)
    if requested == mc_llm_accel.ACCEL_NONE:
        return ordinary

    multitoken = getattr(_advertised_accelerators(configuration), "mtp", None)
    if multitoken is not None and multitoken.embedded:
        return mc_llm_accel.Plan(requested=requested, accelerator=mc_llm_accel.ACCEL_MTP,
                                 memory_priority=priority)
    if requested == mc_llm_accel.ACCEL_MTP:
        # Not a refusal. Fast LLM is defined as "MTP when supported, otherwise
        # ordinary decoding" -- so a backbone without the heads decodes as it
        # always has, keeps the memory priority the preset also carries, and
        # says which of the two it got.
        return _replaced(ordinary,
                         notes=(mc_llm_accel.no_heads(_backbone_label(configuration)),))
    return ordinary


def _advertised_accelerators(configuration: Config):
    """What the catalogue says this backbone can be accelerated with, if anything.

    ``None`` for a manual GGUF, and that is not a gap to be filled by probing:
    an accelerator claim is a statement about a specific file, and this
    extension knows which file a *managed* backbone is because its hash is
    pinned. Somebody's own GGUF gets ordinary decoding, which is what it has
    always got.
    """
    if not configuration.managed_id:
        return None
    try:
        import mc_llm_managed_models

        return mc_llm_managed_models.entry(configuration.managed_id).accelerators
    except Exception:
        logger.debug("Model Chain: could not read %s's accelerators",
                     configuration.managed_id, exc_info=True)
        return None


def _backbone_label(configuration: Config) -> str:
    """What to call the model in a sentence a user reads."""
    if configuration.managed_id:
        try:
            import mc_llm_managed_models

            return mc_llm_managed_models.entry(configuration.managed_id).label
        except Exception:
            pass
    if configuration.model is not None:
        return Path(configuration.model).stem
    return "this backbone"


def _replaced(value, **changes):
    """``dataclasses.replace``, imported here rather than at module scope.

    Every ``prompt_master`` and stdlib-adjacent import in this file is done
    inside a function for the reason the module docstring gives; this one is
    called often enough from the accelerator paths to be worth naming.
    """
    from dataclasses import replace

    return replace(value, **changes)


def accelerator_plan(configuration: Config, gguf: mc_gguf.Gguf | None = None, *,
                     needs_vision: bool = False, already_ours: int = 0,
                     extra_reserve: int = 0, chosen: mc_llm_accel.Plan | None = None,
                     reclaim: bool = True) -> mc_llm_accel.Plan:
    """The complete accelerator decision for one start, flags included.

    Two stages rather than one because they cost different things.
    :func:`accelerator_choice` asks about files and is stable between requests,
    which is what puts it in the warm identity; this one asks the *binary* what
    it accepts, which costs a ``--help`` on a cold runtime and belongs on a path
    that is really going to start something.

    Everything after ``chosen`` is accepted and unused. They were the terms of
    a residency plan for an accelerator that needed a second model on the card;
    multi-token heads need nothing beyond the weights that are already there.
    The parameters stay because every caller passes them and because a fit
    question is a reasonable thing to ask an accelerator, even when this one's
    answer is always yes.
    """
    plan = chosen if chosen is not None else accelerator_choice(configuration, needs_vision)
    if plan.refused or plan.accelerator != mc_llm_accel.ACCEL_MTP:
        return plan

    multitoken = getattr(_advertised_accelerators(configuration), "mtp", None)
    flags = mc_llm_accel.mtp_flags(
        multitoken,
        supports=lambda flag: runtime_supports(flag, configuration),
        accepts=lambda value: runtime_accepts(mc_llm_accel.SPEC_TYPE_FLAG, value,
                                              configuration))
    if not flags:
        return _replaced(plan, accelerator=mc_llm_accel.ACCEL_NONE, flags=(),
                         notes=(*plan.notes,
                                mc_llm_accel.no_option(_backbone_label(configuration))))
    return _replaced(plan, flags=flags)


def _with_runtime(configuration: Config, runtime: Path | None) -> Config:
    """``configuration`` as it would be with ``runtime`` as its executable.

    Capabilities are cached per executable, so asking what a *different* binary
    advertises means asking about that one rather than about the one recorded
    in the state file. Nothing produces a plan with its own runtime today; the
    seam stays because a second runtime family is a thing this extension has
    had once and the alternative is a launch path that silently probes the
    wrong program the next time it has one.
    """
    if runtime is None or runtime == configuration.runtime:
        return configuration
    return _replaced(configuration, runtime=runtime)


def _without_accelerator(plan: mc_llm_accel.Plan, because: str) -> mc_llm_accel.Plan:
    """``plan`` with the accelerator taken off and the reason kept.

    Ordinary decoding rather than the next mechanism down, deliberately: what
    has just been established is that this build rejects an option, and trying
    a second option built out of the same help text would be guessing twice
    with one piece of evidence. The next start re-plans from scratch against a
    capability record that now knows better.
    """
    return mc_llm_accel.Plan(
        requested=plan.requested, accelerator=mc_llm_accel.ACCEL_NONE,
        memory_priority=plan.memory_priority,
        notes=(*plan.notes,
               f"{mc_llm_accel.short_label(plan.accelerator)} was not used: {because}"))


def _make_room_for_the_llm(configuration: Config, already_ours: int = 0,
                           needs_vision: bool = False, extra_reserve: int = 0) -> int:
    """Ask the image side for this card's deficit, when LLM priority is set.

    Nothing at all under cooperative memory, which is the default and is what
    every version of this extension has done: the language model lives in the
    VRAM the image side is not using and shrinks itself when there is little.

    Under LLM priority it is the *whole* of what that setting does, so it is
    worth being plain about how narrow it is. It asks for the placement the
    user configured, on the card that placement is going to, for the shortfall
    and never for a sweep -- and :func:`mc_broker.release_for_llm` will not
    cross to another card, because releasing a card this model is not being
    placed on cannot free a byte of the one it is.

    Returns what was actually released, which is a measurement rather than an
    expectation: the negotiation that follows reads the card again, so a
    request that freed nothing simply places against what was already there.
    """
    if configuration.memory_priority != mc_llm_accel.PRIORITY_LLM:
        return 0
    card = card_of(configuration)
    if card is None:
        return 0
    described = mc_gguf.describe(configuration.model)
    placement = _requested_placement(configuration, described, already_ours)
    if not placement.on_gpu:
        return 0
    wanted = mc_llm_context.estimate(configuration.model, placement, described)
    needed = (wanted.total_bytes + projector_bytes(configuration, needs_vision)
              + max(int(extra_reserve), 0))
    released = mc_broker.release_for_llm(
        needed, card=card, uuid=configuration.gpu_uuid, name=configuration.card_name,
        reason=f"{_backbone_label(configuration)}, which has been given "
               f"priority on this card")
    if released.freed:
        logger.info("Model Chain: released %.1f GB of image VRAM on GPU %d — this "
                    "configuration is set to LLM priority", released.freed / _GB, card)
    return int(released.freed)


def _launch_flags(configuration: Config, placement: mc_llm_context.Placement,
                  plan: mc_llm_accel.Plan | None) -> list[str]:
    """Every extra flag one start needs: the placement's, then the accelerator's.

    Both sets are gated against the executable that is *actually* going to run,
    which is the configured one today -- capabilities are cached per binary, and
    a launch that ever grows a runtime of its own must ask that one rather than
    the one recorded in the state file.

    ``--flash-attn`` can be produced by both halves, and llama.cpp takes the
    last spelling it is given -- but a switch-style build passed the flag twice
    is a build passed an argument it will try to parse as a value. So the
    accelerator's flags are filtered against what the placement's already
    carry, rather than the two being concatenated and hoped about.
    """
    special = _with_runtime(configuration, plan.runtime if plan is not None else None)
    flags = accelerator_flags(special, placement)
    if plan is None or not plan.flags:
        return flags
    extra = list(plan.flags)
    if FLASH_ATTENTION_FLAG in flags:
        extra = _without_flash_attention(extra)
    return flags + extra


def _without_flash_attention(flags: list[str]) -> list[str]:
    """``flags`` with ``--flash-attn`` and any value beside it removed."""
    kept, skip = [], False
    for position, flag in enumerate(flags):
        if skip:
            skip = False
            continue
        if flag == FLASH_ATTENTION_FLAG:
            following = flags[position + 1] if position + 1 < len(flags) else ""
            skip = following in ("on", "off", "auto")
            continue
        kept.append(flag)
    return kept


def conservative_flags(configuration: Config) -> list[str]:
    """What Mixed Conservative asks for beyond zero layers, where the build has it.

    Empty for every other mode. Conservative is the only placement that names a
    card and wants nothing resident on it, so it is the only one with anything
    to say here: keep the cache off the card, and use the card for arithmetic.

    Both are gated on the build advertising them. Section 21's rule for this
    mode is that an unsupported optional flag is left off and the reduced
    capability reported -- never a reason to abandon the zero-layer promise,
    which is the part of the mode that does not depend on any flag at all.
    """
    if not is_conservative(configuration):
        return []
    flags = []
    if runtime_supports(NO_KV_OFFLOAD_FLAG, configuration):
        flags.append(NO_KV_OFFLOAD_FLAG)
    if runtime_supports(OP_OFFLOAD_FLAG, configuration):
        flags.append(OP_OFFLOAD_FLAG)
    return flags


def expert_flags(configuration: Config, placement) -> list[str]:
    """The expert-offload part of a command line, in the spelling this build has.

    Empty for a placement with every expert on the card, for a build that
    advertises neither flag, and for a dense model -- which never reaches a
    placement with experts moved, because :func:`_shrink_offload` reads the
    header before it makes one.

    Empty, too, for a placement with no layers on the card at all. "The experts
    are in system RAM" is not a thing that can be asked for once *everything* is
    in system RAM: the flag selects nothing that ``--n-gpu-layers 0`` has not
    already selected, and asking for it anyway puts llama.cpp on its tensor-
    override path, which costs a warning about mmap and the performance that
    warning is about. It also produced a console line that contradicted itself,
    announcing that the experts had been left behind and that the whole model
    had been -- see :func:`_shrink_offload`.
    """
    layers = int(getattr(placement, "cpu_expert_layers", mc_llm_context.NO_EXPERTS))
    if layers == mc_llm_context.NO_EXPERTS:
        return []
    resident = int(getattr(placement, "gpu_layers", mc_llm_context.ALL_LAYERS))
    if resident == mc_llm_context.NO_LAYERS:
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


def measurement_token(placement=None, accelerator: str = "", card: int | None = None) -> str:
    """One key-safe word naming everything a decode rate actually depends on.

    Section 17: key learned speed by backbone, quantisation, physical GPU,
    placement and accelerator -- and never average an accelerated rate together
    with an ordinary one. The backbone and its quantisation are the identity half of
    the key; this is the other three.

    Both suffixes are omitted when they would say nothing, so every rate this
    machine has already measured keeps the key it was written under: ordinary
    decoding on an unknown card is ``gpu``, exactly as it always was, and only
    a run that really used an accelerator or really knows its card writes
    somewhere new.
    """
    token = placement if isinstance(placement, str) else placement_token(placement)
    if not token:
        return ""
    if accelerator and accelerator not in (mc_llm_accel.ACCEL_NONE, mc_llm_accel.ACCEL_AUTO):
        token = f"{token}-{accelerator}"
    if card is not None and int(card) >= 0:
        token = f"{token}-cuda{int(card)}"
    return token


PLACEMENT_WORDS = {
    "gpu": "all layers on the GPU",
    "cpu": "in system RAM",
    "cpu-moe": "experts in system RAM",
}

ACCELERATOR_WORDS = {
    "mtp": "MTP",
}
"""How :func:`describe_placement_token` reads the suffix back out."""


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
    token, card = _split_suffix(token, ("cuda",))
    token, accelerator = _split_suffix(token, ACCELERATOR_WORDS)
    head, _, layers = token.partition("-l")
    said = PLACEMENT_WORDS.get(head)
    if said is None and head.startswith("ncmoe-"):
        count = head[len("ncmoe-"):]
        said = f"{count} expert layer{'' if count == '1' else 's'} in RAM"
    if said is None:
        said = head
    if layers:
        said = f"{said}, {layers} layers on the GPU"
    if accelerator:
        said = f"{said}, {ACCELERATOR_WORDS.get(accelerator, accelerator)}"
    if card:
        said = f"{said}, GPU {card[len('cuda'):]}"
    return said


def _split_suffix(token: str, known) -> tuple[str, str]:
    """``token`` with a trailing ``-<suffix>`` removed, and that suffix.

    ``known`` is either the exact suffixes to recognise or a prefix tuple for
    ones carrying a number. Anything unrecognised is left on the token and
    printed as it is -- an unfamiliar word beside a rate is a smaller failure
    than a rate whose placement is not stated at all.
    """
    head, separator, tail = token.rpartition("-")
    if not separator:
        return token, ""
    if isinstance(known, tuple):
        if any(tail.startswith(prefix) and tail[len(prefix):].isdigit() for prefix in known):
            return head, tail
        return token, ""
    return (head, tail) if tail in known else (token, "")


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


_job_handle = None
"""The Windows job object every llama-server this extension starts is put in.

Held for the life of the process on purpose: what makes it work is the handle
being *closed*, which happens when this process ends however it ends -- exit,
Ctrl+C, or a kill nothing in here can run a handler for. See
:func:`_die_with_us`.
"""


def _die_with_us(process) -> None:
    """Ask the operating system to end ``process`` when this one ends.

    The third exit, and the one no handler can cover: a hard kill runs nothing
    inside the process being killed, so a llama-server holding twenty gigabytes
    outlives the WebUI and there is nobody left to stop it. What answers it has
    to be arranged *before* the kill and enforced by the kernel.

    On Windows that is a job object with ``KILL_ON_JOB_CLOSE``. Every server
    goes into one job; the last handle to it closes when this process dies, and
    the kernel ends everything in it. This is also why the vendored launcher's
    ``CREATE_NEW_PROCESS_GROUP`` is not the answer -- that flag is *why* Ctrl+C
    never reached the child in the first place.

    On Linux it is ``PR_SET_PDEATHSIG``, which cannot be set from here because
    it has to run in the child between fork and exec; the ordinary exits are
    covered by :func:`stop_on_exit` there and a ``kill -9`` of the parent is
    not, which is stated rather than pretended about.

    Best-effort throughout. Every failure is logged at debug and the server
    starts anyway: a language model that runs and might outlive its parent is
    better than one that does not run.
    """
    global _job_handle

    if os.name != "nt":
        return
    handle = getattr(process, "_handle", None)
    if handle is None:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        if _job_handle is None:
            job = kernel.CreateJobObjectW(None, None)
            if not job:
                raise ctypes.WinError(ctypes.get_last_error())

            class _Limits(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                            ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class _Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", _Limits),
                            ("IoInfo", ctypes.c_byte * 48),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            information = _Extended()
            information.BasicLimitInformation.LimitFlags = JOB_KILL_ON_CLOSE
            if not kernel.SetInformationJobObject(
                    job, JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(information),
                    ctypes.sizeof(information)):
                raise ctypes.WinError(ctypes.get_last_error())
            _job_handle = job
            logger.info("Model Chain: llama-server processes will be ended by Windows if "
                        "this process is killed")
        if not kernel.AssignProcessToJobObject(_job_handle, int(handle)):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception:
        logger.debug("Model Chain: could not tie llama-server's lifetime to this process; "
                     "a hard kill of the WebUI may leave it running", exc_info=True)


JOB_KILL_ON_CLOSE = 0x00002000
"""``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``."""

JOB_EXTENDED_LIMIT_INFORMATION = 9
"""``JobObjectExtendedLimitInformation``."""


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
        started = subprocess.Popen(
            with_extra_flags(without_gpu_selection(command)), *args, **kwargs)
        # Every server, however it was started -- including the smoke tests and
        # anything a future path spawns through the vendored launcher. This is
        # the one place they all pass through, which is exactly why the command
        # rewrite lives here too.
        _die_with_us(started)
        return started


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
    # The first thing that is really about to start a server is the right
    # moment to arrange for it to be stopped again.
    stop_on_exit()


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
    # Last, and narrowly. "error while handling argument" is llama.cpp's line
    # for every option it will not take, including ``--device``, which is a
    # symptom of a card that could not be enumerated and is diagnosed above in
    # far more useful terms. What is left for this branch is the case that
    # diagnosis cannot cover: an *optional* flag this extension chose to add,
    # which the model knows nothing about and should not be blamed for.
    argument = _BAD_ARGUMENT.search(text)
    if argument and argument.group(1) in OPTIONAL_FLAGS:
        flag, complaint = argument.group(1), argument.group(2).strip().rstrip(".")
        value = _BAD_VALUE.search(complaint)
        return Failure(
            f"llama-server would not take {flag}: {complaint}. That is an optional flag this "
            f"extension adds, not anything about the model itself",
            bad_argument=flag, bad_value=value.group(1) if value else "")
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


def _stands_down_for_the_image_job(configuration: Config | None) -> bool:
    """Whether background priming should give way to a running image job.

    Section 6.6's rule, which is the execution rule and not a new one: a prime
    on GPU 1 while Forge samples on GPU 0 competes for nothing and may proceed
    if the workload lock is free; a prime on GPU 0 stands down as it always
    has; a processor prime is independent of GPU activity altogether.
    """
    try:
        if not mc_broker.host_busy():
            return False
        return execution_domain(configuration).conflicts_with(
            mc_broker.image_execution_domain())
    except Exception:
        logger.debug("Model Chain: could not compare the priming device with the image job",
                     exc_info=True)
        return True


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
    thread = threading.Thread(target=_prime, args=(client, config()), name="mc-llm-prime",
                              daemon=True)
    _priming = thread
    thread.start()


def _prime(client, configuration: Config | None = None) -> None:
    try:
        from prompt_master.krea import enhancer

        if _stands_down_for_the_image_job(configuration):
            return
        with mc_broker.workload(mc_broker.FAMILY_LLM, PRIME_LABEL, timeout=0,
                                background=True, required=False,
                                domain=execution_domain(configuration)) as held:
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
    """Say once, per start, when a card is present and holding nothing it wanted to.

    Mixed Aggressive and GPU / VRAM Only both ask for layers, and llama.cpp with
    none offloaded runs every matrix multiply on the processor -- so a machine
    with a 3090 in it can spend twenty seconds a press at four tokens a second
    while the card sits idle, with nothing on screen or in the log saying that
    was the arrangement. This is that line.

    Mixed Conservative is not that arrangement and is excluded. Zero layers is
    what the user asked that mode for, so there is no shortfall to report; and
    the card is not idle either -- it takes the operations ``--op-offload``
    hands it, which in a user's log was the difference between 50 tokens a
    second on the processor and 82 with the card named. Warning about a setting
    working correctly is how somebody comes to change it back.
    """
    from prompt_master.core.models import CPU_MODE, MIXED_CONSERVATIVE_MODE, normalise_mode
    from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD

    if str(layers) != NO_OFFLOAD:
        return
    if str(configuration.device).casefold() == CPU_DEVICE:
        return
    if normalise_mode(configuration.mode) in (MIXED_CONSERVATIVE_MODE, CPU_MODE):
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


def _slots_for(configuration: Config, placement: mc_llm_context.Placement) -> int:
    """The slot count this start may actually ask for.

    Asked in two places on purpose -- once by :func:`_add_slots`, so the
    estimate is of the placement that will really run, and once by the launch
    itself as the last word before the command line is built. A footprint
    priced for three caches and a server started with one would be two
    descriptions of one placement, which is the mistake
    :func:`_profile_arguments` already refuses to make about cache types.
    """
    slots = max(int(getattr(placement, "slots", 1)), 1)
    if slots <= 1:
        return 1
    if not runtime_supports(PARALLEL_FLAG, configuration):
        logger.info("Model Chain: this llama.cpp build does not advertise %s, so it keeps "
                    "one prompt cache; the modes that use different system prompts will "
                    "re-read each other's", PARALLEL_FLAG)
        return 1
    # Refused *at startup*, which is the only evidence worth having about a
    # numeric argument -- and deliberately not `runtime_accepts`, whose
    # help-text search answers an enumeration's question and would call every
    # slot count unsupported. See :func:`runtime_refused`.
    while slots > 1 and runtime_refused(PARALLEL_FLAG, str(slots), configuration):
        # Halved rather than abandoned. A build that would not take six caches
        # may take three, and the two extra attempts this can cost are argument
        # parsing rather than model loads. Keeping the accelerator is the
        # user's stated choice when neither can be had: it is a setting they
        # picked and it pays back on every token, where a cache is worth one
        # prompt re-read.
        slots //= 2
    if slots <= 1:
        logger.info("Model Chain: this llama.cpp build refused %s at every count it was "
                    "offered, so it keeps one prompt cache and its accelerator",
                    PARALLEL_FLAG)
        return 1
    return slots


_SLOT_REPORT = re.compile(r"n_slots\s*=\s*(\d+).*?n_ctx_slot\s*=\s*(\d+)", re.S)


def _check_slots(log_path, offset: int, placement: mc_llm_context.Placement) -> None:
    """Read back what llama.cpp actually built, and say so when it differs.

    ``--ctx-size`` is a *total* that llama.cpp divides among its slots, and the
    one way to get this wrong is silent: a per-slot number passed as the total
    gives every slot a fraction of the context asked for and truncates prompts
    to it without an error. llama.cpp prints what it decided --
    ``n_slots = 3, n_ctx_slot = 8192`` -- so the arithmetic is checked against
    the server rather than trusted.

    Never raises and never fails a start. A log line that could not be read is
    a diagnostic that did not happen, not a placement that is wrong.
    """
    try:
        found = _SLOT_REPORT.search(_text_since(log_path, offset))
        if found is None:
            return
        slots, per_slot = int(found.group(1)), int(found.group(2))
    except Exception:
        logger.debug("Model Chain: could not read the slot report", exc_info=True)
        return
    if slots == placement.slots and per_slot >= placement.context:
        return
    logger.warning(
        "Model Chain: llama.cpp built %d slot(s) of %s tokens where this placement asked "
        "for %d of %s. Prompts longer than %s tokens will be truncated — the context on "
        "the command line is a total divided among the slots, and this start's arithmetic "
        "did not survive it.",
        slots, f"{per_slot:,}", placement.slots, f"{placement.context:,}", f"{per_slot:,}")


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
            # The mode, because two of them are otherwise indistinguishable
            # here. Mixed Aggressive and Mixed Conservative name the same card
            # and both record zero layers -- the difference is what happens
            # next, and it is a different command line and a different amount
            # of the card. Without this, switching between them reuses the
            # server that is already up, and two roles configured one each way
            # would share one process against the user's explicit choice.
            str(configuration.mode),
            # The profile is a *choice* even though nobody typed it: switching
            # backbones changes the template flag and the samplers, and two of
            # those are command-line arguments a running server cannot be told
            # about. Without this, a switch between two models that happened to
            # want the same context and cache types would have reused the
            # server that was already up -- still holding the previous weights.
            str(configuration.profile_id),
            # The two performance axes, because both are start-time facts. A
            # server started with ``--spec-type`` cannot be given it later,
            # and one started with it cannot be told to stop. A change to the
            # *setting* restarts the server here; what is actually running is
            # compared beside it -- see :meth:`Runtime.client` and
            # :attr:`mc_llm_accel.Plan.identity` -- because ``auto`` resolves
            # against the catalogue and the binary rather than against this.
            str(configuration.accelerator), str(configuration.memory_priority))


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
    plan: mc_llm_accel.Plan = field(default_factory=mc_llm_accel.Plan)
    """Which accelerator actually ran, and what it was measured to need.

    Section 14 asks the status line to name the mechanism rather than the
    preset, and section 9 says the same thing about ``auto``: "status must name
    the mechanism actually used". This is that, kept beside the placement it
    was decided with so the two cannot be reported out of step."""
    speculation: "Speculation" = field(default_factory=lambda: Speculation())
    """Drafted and accepted token counts, where llama.cpp reported them."""


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

    def __init__(self, message: str, out_of_memory: bool = False,
                 bad_argument: str = "", bad_value: str = ""):
        super().__init__(message)
        self.out_of_memory = out_of_memory
        self.bad_argument = bad_argument
        self.bad_value = bad_value


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


def _signature_of(configuration: Config, projector, placement,
                  plan: mc_llm_accel.Plan | None = None) -> tuple:
    """What a running server would have to be restarted to change.

    The expert split is in it because it is a start-time argument like the
    other two: a placement that moved two more blocks' experts is a different
    command line, and a signature that could not tell the two apart would hand
    back the server that had just run out of memory.

    The accelerator is in it for the same reason and one more. ``--spec-draft-
    model`` is a start-time argument, so a server started without one cannot be
    given one later, and a plan carrying a runtime of its own would be a
    different program besides. A signature blind to either would hand back an
    ordinary server for a request that had just been promised an accelerator.
    """
    return (plan.runtime if plan is not None and plan.runtime is not None
            else configuration.runtime,
            configuration.model, projector,
            configuration.gpu_index, configuration.device,
            placement.context, placement.gpu_layers, placement.cpu_expert_layers,
            # A start-time argument like the other three: --parallel and
            # --ctx-size are fixed when the process starts, so a server holding
            # one cache cannot be given three without being replaced.
            placement.slots,
            plan.identity if plan is not None else ())


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

    def __init__(self, residency_key: str = "", roles: tuple = ()):
        self._lock = threading.RLock()
        self._process = None
        self._signature: tuple | None = None
        self._identity: tuple | None = None
        self._accelerator: tuple = ()
        """The accelerator identity the running server was started with.

        Beside ``_identity`` rather than inside it, because the two answer
        different questions. ``_identity`` is what the *user chose*, and a
        change to it is a setting change. This is what was *resolved* from that
        choice against the files on disk -- so installing a draft model turns
        an ``auto`` configuration from ordinary decoding into MTP without any
        setting having changed -- a different backbone, or a runtime that has
        learned what it accepts -- and the warm server has to be replaced."""
        self._projector: Path | None = None
        """The vision projector the *running* process was started with.

        The third of the three facts :mod:`mc_llm_vision` keeps apart, and the
        only one that is process state. ``configuration.mmproj`` says which
        projector is compatible with the selected model; this says whether the
        server that is up was actually launched with ``--mmproj``, which is the
        difference between a capability that has to be paid for and one that has
        already been paid for.

        It is what makes vision *sticky*. A text-only request reads this rather
        than deriving a wanted projector of ``None`` from its own needs, so a
        warm vision-capable server satisfies it as it stands -- see
        :meth:`_wanted_projector`, and design intent sections 5.6 and 6."""
        self._placement: mc_llm_context.Placement | None = None
        self._log: tuple | None = None
        """``(path, offset)`` of the running server's slice of the log."""
        self.report = Report()
        self.residency_key = residency_key or RESIDENCY_KEY
        """This server's line in the broker's register.

        Per runtime rather than per module, because two of these may be up at
        once and one key between them would have each declaration overwrite the
        other's -- so the broker would believe there was one server holding
        whatever the last one to start happened to hold. Section 17: "residency
        keys must be runtime-specific rather than one global LLM key."
        """
        self.roles: tuple = tuple(roles)
        """Which roles resolved to this runtime. Both when they are sharing it."""
        self._role: str = ""
        """Whose configuration to start from. See :meth:`configuration`."""
        self._key: tuple | None = None
        """The identity the registry filed this runtime under."""
        self._placed_for: tuple | None = None
        """Which image-plan boundary *this* server has been reconciled against.

        Per runtime, and that is section 8.3. It used to be one module-level
        value in :mod:`mc_plan`, which was right for as long as there could
        only be one llama-server; with two, whichever started last overwrote
        the other's baseline, so a Creative role on a second card inherited a
        boundary it had never been evaluated against and was re-placed for a
        plan that does not describe its card at all.

        None means "never placed under a plan". Cleared whenever the process
        stops, is replaced, or is demoted, because a baseline describes a
        placement and there is no longer one to describe.
        """
        self._card: int | None = None
        """The physical card the running process was launched against.

        Read from the configuration at launch rather than re-derived later, so
        that a reclaim aimed at this runtime is judged against the card it is
        actually on and not the card its settings currently say -- those can
        differ for exactly as long as it takes a reconfigured role to restart.
        """

    def _said_for(self) -> str:
        """``"[Creative] "`` or ``""``, for the front of this runtime's own lines.

        The placement lines used to say which model, which device and which
        context, and not which *configuration* they came from -- which is
        exactly the fact somebody needs when two servers are up and one of them
        is on the wrong card. A user reading "on Intel(R) Core(TM) Ultra 9" had
        no way to tell whether that was the Creative role they had pointed at a
        5090, or the Spatial one they had pointed at the processor.
        """
        import mc_llm_roles

        return mc_llm_roles.prefix(self._role)

    def adopt(self, role: str, key: tuple) -> None:
        """Record whose configuration this server runs on. Called by the registry."""
        if role:
            self._role, self._key = role, key
        elif self._key is None:
            self._key = key

    def configuration(self) -> Config:
        """The configuration this server is, or would be, started from.

        The bug this exists to close: the registry resolved *which* runtime
        serves a role, and then the runtime asked ``config()`` -- with no role
        -- for what to start. So a Creative role pinned to a 5090 got a server
        of its own, correctly, and that server was launched from the
        installation's settings. Two roles on different cards both came up on
        whatever the installation said, which in a user's log was the processor:
        two llama-servers, two prompt caches, and one wrong placement each.

        The identity is re-checked rather than trusted, because the shared
        runtime is also every non-role mode's server. A role that was mapped
        here and has since been reconfigured no longer belongs to this runtime,
        and starting Prompt Studio from the settings of a Creative role that has
        moved to another card would be this same bug pointing the other way.
        """
        if not self._role:
            return config()
        resolved = config(self._role)
        if self._key is not None and _identity(resolved, resolved.mmproj) != self._key:
            return config()
        return resolved

    # -- lifecycle -------------------------------------------------------- #

    def client(self, needs_vision: bool = False, reserve: int = 0, cancel=None):
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

        ``needs_vision`` is a *requirement*, not a description. True means this
        request cannot be served without the projector; False means it does not
        care, which is not the same as "start without one". So the flag can only
        ever move capability upward -- OFF to TEXT_ONLY to VISION_LOADED -- and
        the server that comes back for a text turn after a picture is the same
        process, with the same prompt cache. See :meth:`_wanted_projector` for
        the rule and :meth:`_prepare_vision` for where a missing projector is
        put back before any of this is reached.
        """
        from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD
        from prompt_master.inference.service import CPU_READY_TIMEOUT, GPU_READY_TIMEOUT

        # Before the lock, and that is the whole reason it is a separate step.
        # Resolving a projector can mean a gigabyte over somebody's connection,
        # and section 13 is explicit that a slow transfer must not hold the
        # process lock of a server that is answering other requests perfectly
        # well meanwhile. Nothing is stopped here either, so a repair that
        # cannot be completed leaves a running text server running (24.1).
        if needs_vision:
            self._prepare_vision(cancel)

        with self._lock:
            configuration = self.configuration()
            if not configuration.configured:
                raise NotConfigured(
                    "No local model is configured yet. Choose a GGUF and a llama.cpp runtime "
                    "in LLM Studio’s Setup mode."
                )
            # Capability, not request: what this server should be running with
            # once this request has been served. A picture asks for the
            # compatible projector; anything else asks for whatever is already
            # loaded, which is how vision stops being a mode that every turn
            # switches back off.
            projector = self._wanted_projector(configuration, needs_vision)
            vision = projector is not None
            for label, path in (("llama-server", configuration.runtime),
                                ("model", configuration.model),
                                ("vision projector", projector)):
                if path is not None and not Path(path).is_file():
                    raise RuntimeError(f"Configured {label} is missing: {path}")
            if needs_vision and projector is None:
                raise RuntimeError(mc_llm_vision.NO_PROJECTOR)

            # Which mechanism this request can use, before anything is measured
            # and before anything is stopped. A forced accelerator that is
            # simply not installed refuses here, in a sentence, rather than
            # after a twenty-second model load -- and an automatic one has
            # already stepped down to whatever it can prove is available.
            chosen = accelerator_choice(configuration, vision)
            if chosen.refused:
                raise RuntimeError(chosen.refusal)

            ours = self.resident_bytes()
            if (self._running and self._identity == _identity(configuration, projector)
                    and self._accelerator == chosen.identity
                    and not self._outgrown(configuration, ours, vision)):
                if self._projector is not None and not needs_vision:
                    logger.debug("Model Chain: %sreusing the vision-loaded server for a "
                                 "text-only request", self._said_for())
                self._touch(configuration, ours)
                return self._client(configuration)

            if self._running and vision and self._projector is None:
                logger.info("Model Chain: %svision is required by this request; restarting the "
                            "warm server with %s", self._said_for(), Path(projector).name)

            # Before anything is measured, and only on a path that is really
            # going to start a server. The image allocator keeps the blocks it
            # has finished with, which is free memory to this process and to no
            # other -- so it is handed back to the driver first, and the
            # placement below is decided against what llama.cpp will actually
            # be able to allocate.
            #
            # Only when this server can actually consume those blocks, which
            # means the same physical card (section 13). A 5090 llama-server
            # cannot be handed a cached 3090 block, so emptying the image
            # allocator for it gives up useful image state -- every one of
            # those blocks has to be re-obtained from the driver on the next
            # pass -- and frees nothing at all where the model is going.
            recovered = self._release_the_image_cache_if_it_helps(configuration)
            if recovered:
                logger.info("Model Chain: returned %.1f GB of cached VRAM to the driver before "
                            "placing the LLM", recovered / _GB)

            plan = accelerator_plan(configuration, already_ours=ours,
                                    needs_vision=vision, extra_reserve=reserve,
                                    chosen=chosen)
            if plan.refused:
                raise RuntimeError(plan.refusal)

            # The one place image residency can be released for the language
            # model, and only because somebody set LLM priority: on the card it
            # is being placed on, for the deficit and no more. Before the
            # negotiation rather than inside it, because :func:`negotiate`
            # promises to move nothing and is relied on for that by the
            # estimator -- what it does here is find more free VRAM than it
            # would have found a moment ago, and place against that.
            _make_room_for_the_llm(configuration, ours, vision, reserve)

            negotiated = negotiate(configuration, already_ours=ours, vision=vision,
                                   extra_reserve=reserve)
            placement = negotiated.placement
            signature = _signature_of(configuration, projector, placement, plan)

            if self._running and signature == self._signature:
                self._identity = _identity(configuration, projector)
                self._accelerator = plan.identity
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
            dropped_accelerator = False
            for attempt in range(START_ATTEMPTS):
                if penalty or expert_floor:
                    negotiated = negotiate(configuration, already_ours=ours,
                                           extra_reserve=reserve + penalty,
                                           vision=vision, expert_floor=expert_floor)
                    placement = negotiated.placement
                    signature = _signature_of(configuration, projector, placement, plan)
                try:
                    process, observed, offload = self._launch(configuration, placement,
                                                              projector, plan)
                    break
                except _StartFailed as failure:
                    # An option llama.cpp will not take is this extension's
                    # mistake, not the model's, and the model must not be what
                    # pays for it. Without this the start died at argument
                    # parsing, the switch that asked for it rolled back, and
                    # what the user read was "this backbone was downloaded but
                    # would not start" about a backbone that was perfectly fine.
                    if failure.bad_argument == PARALLEL_FLAG and placement.slots > 1:
                        # The caches, not the accelerator. A build that will not
                        # take --parallel beside a draft model is refusing the
                        # optimisation, and the accelerator is the setting: it
                        # was chosen, and it pays back on every token rather
                        # than once per prompt.
                        note_rejected_value(PARALLEL_FLAG, str(placement.slots),
                                            _with_runtime(configuration, plan.runtime))
                        logger.warning("Model Chain: %s. The accelerator was kept and the "
                                       "warm prompt caches were given up instead.", failure)
                        negotiated = negotiate(configuration, already_ours=ours,
                                               extra_reserve=reserve + penalty,
                                               vision=vision, expert_floor=expert_floor)
                        placement = negotiated.placement
                        signature = _signature_of(configuration, projector, placement, plan)
                        continue
                    if failure.bad_argument and plan.flags and not dropped_accelerator:
                        dropped_accelerator = True
                        if failure.bad_value:
                            note_rejected_value(failure.bad_argument, failure.bad_value,
                                                _with_runtime(configuration, plan.runtime))
                        logger.warning("Model Chain: %s. It has been left off and the "
                                       "start tried again.", failure)
                        plan = _without_accelerator(plan, str(failure))
                        negotiated = negotiate(configuration, already_ours=ours,
                                               extra_reserve=reserve + penalty,
                                               vision=vision,
                                               expert_floor=expert_floor)
                        placement = negotiated.placement
                        signature = _signature_of(configuration, projector, placement, plan)
                        continue
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
                    reclaimed = self._release_the_image_cache_if_it_helps(configuration)
                    logger.warning("Model Chain: %s. Trying again with %.1f GB more headroom%s%s",
                                   failure, penalty / _GB,
                                   f" and the experts of at least {expert_floor} layers in "
                                   "system RAM" if expert_floor > 0 else "",
                                   f", having returned {reclaimed / _GB:.1f} GB of cached VRAM "
                                   "to the driver" if reclaimed else "")

            self._process, self._signature, self._placement = process, signature, placement
            self._identity = _identity(configuration, projector)
            # The card this process is actually on, fixed at launch. Read later
            # from the configuration it would move with a setting change, and a
            # reclaim aimed at a running server has to be judged against where
            # the server is rather than where it is next going to be.
            self._card = card_of(configuration)
            self._projector = Path(projector) if projector is not None else None
            self._accelerator = plan.identity
            if self._projector is not None:
                # Section 25's transition line. Said once per capability change
                # and never per request, because what somebody debugging a slow
                # first token needs to know is which starts loaded a projector
                # -- not that every turn since has correctly reused it.
                logger.info("Model Chain: %sllama-server is now vision-loaded — %s",
                            self._said_for(), self._projector.name)
            self._record(configuration, negotiated, observed, offload, plan)
            # Which plan this placement answers. Everything after this point
            # compares against it rather than against free VRAM, so the server
            # survives every phase of the generation it was started for -- and
            # it is recorded on *this runtime*, so a second server on a second
            # card cannot inherit or overwrite the boundary (section 8.3).
            self._note_placement()
            prepared = self._client(configuration)
            _prime_prompt_cache(prepared)
            return prepared

    def _release_the_image_cache_if_it_helps(self, configuration: Config) -> int:
        """Empty Forge's allocator cache, but only for a start that can use it.

        Section 13, and the whole of the rule is "same physical card". The
        blocks the image allocator is sitting on belong to one GPU's driver:
        a llama-server being placed on that GPU can have them once they are
        handed back, and a llama-server anywhere else -- another card, or the
        processor -- cannot. Emptying it anyway costs the image side every
        cached block it had ready and buys the language model nothing.

        An unresolvable relation keeps the old behaviour, because the release
        is harmless when it turns out to be the same card and the reported
        figure is the image card's either way. What must never happen is
        reporting image-card bytes as though they had been freed on another
        target GPU, which is why the release is skipped rather than reported
        for a card known to be different.
        """
        try:
            if not configuration.uses_cuda_compute:
                logger.debug("Model Chain: this placement is on the processor; the image "
                             "allocator's cached VRAM was left where it is")
                return 0
            card = card_of(configuration)
            if card is not None and not shares_the_image_card(card, configuration):
                logger.info("Model Chain: llama-server is being placed on GPU %d and the image "
                            "allocator's cache is on %s — it was left alone, because a block "
                            "cached on one card cannot be handed to a process on another",
                            card, mc_broker.image_execution_domain().describe())
                return 0
        except Exception:
            logger.debug("Model Chain: could not compare the placement card with the image "
                         "card", exc_info=True)
        return mc_broker.release_cached_vram()

    def _admit_host_ram(self, configuration: Config,
                        placement: mc_llm_context.Placement | None = None) -> None:
        """Make host RAM safe for a materially RAM-backed placement (section 10.8).

        Never a refusal. llama.cpp will make an honest attempt whatever this
        finds, it is the user's machine, and the existing behaviour has always
        been to warn -- so what this adds is the step *before* the warning:
        when the demand does not fit above the host floor, explicitly managed
        warm image cache is asked to give ground first, because a checkpoint
        kept warm to shorten a future switch is exactly the thing that should
        yield to memory an active workload needs to run at all (invariant I-6).

        Nothing happens at all when it already fits, which is the common case
        and the one worth protecting: two RAM-backed servers and a warm Stage 2
        checkpoint that all fit above the reserve should all stay put
        (invariant I-5).
        """
        wanted = host_ram_demand(configuration, placement)
        if wanted <= 0 or mc_broker.host_ram_fits(wanted):
            return
        admission = mc_broker.admit_host_ram(
            wanted, reason=f"{_backbone_label(configuration)} in system RAM")
        if admission.fits:
            if admission.moved_anything:
                logger.info("Model Chain: %s%s before starting llama-server",
                            self._said_for(), admission.describe())
            return
        logger.warning(
            "Model Chain: %sthis placement needs about %.1f GB of system RAM and only "
            "%.1f GB is available above the %.1f GB reserve%s. llama.cpp will still try, "
            "and reads through the page cache, so expect a slow load and slow replies "
            "until something on this machine gives that memory back.",
            self._said_for(), wanted / _GB, admission.available / _GB,
            admission.reserve / _GB,
            f" (after {admission.describe()})" if admission.moved_anything else "")

    def _prepare_vision(self, cancel=None) -> None:
        """Make sure a compatible projector exists on disk. Outside the lock.

        The one place invariant I-4 is met -- "the backend must ensure the
        correct projector is available and loaded before sending image content"
        -- and the one place section 11's repair happens. What it can change is
        the *state file*: a managed bundle whose projector was missing has it
        downloaded and recorded here, so the ``configuration`` read a few lines
        later inside the lock already names it.

        A configuration that cannot be read at all is left to the locked path,
        which has the sentence for it. A selection that has no projector to
        find is left alone too: refusing here would mean this function decided
        what a text-only backbone does with a picture, and that answer belongs
        beside the one about a missing runtime.
        """
        try:
            configuration = self.configuration()
        except Exception:
            logger.debug("Model Chain: could not read the configuration before resolving a "
                         "vision projector", exc_info=True)
            return
        if not configuration.configured:
            return
        # The repair's own progress goes to the console rather than nowhere.
        # A projector is over a gigabyte, and a request that appears to have
        # hung for two minutes is the one thing a silent download guarantees.
        mc_llm_vision.ensure_projector(
            configuration, role=self._role, cancel=cancel,
            say=lambda text: logger.info("Model Chain: %s%s", self._said_for(), text))

    def _wanted_projector(self, configuration: Config, needs_vision: bool):
        """Which projector the server should be running with after this request.

        The asymmetry in section 6's table, as one function. A request that
        carries an image requires vision and names the compatible projector; a
        request that does not require vision requires *nothing about* vision,
        and so it inherits whatever the running process already has.

        That inheritance is the fix. The old rule read the request's own
        ``needs_vision`` as a complete description of the server that should be
        running, so a text turn after an image turn asked for a server with no
        projector, did not get one, and stopped a perfectly good process to
        build it -- which the next picture then paid to undo. Two restarts, a
        lost prompt cache each time, and a conversation that alternated between
        fast and thirteen seconds of prompt evaluation for no gain at all:
        vision capability is a superset of text capability, and a superset
        satisfies the subset.

        The projector is inherited only while it is still *this* configuration's
        compatible one and still on disk. A model change makes the running
        projector the wrong projector, and a file that has been deleted
        underneath a live server cannot be passed to the next start; both fall
        back to a text-only identity, which the lines below then treat as the
        ordinary replacement it is rather than as a downgrade this request asked
        for.
        """
        if needs_vision:
            return configuration.mmproj
        loaded = self._projector if self._running else None
        if loaded is None:
            return None
        known = configuration.mmproj
        if known is not None and Path(known) == Path(loaded) and Path(loaded).is_file():
            return loaded
        return None

    def vision_loaded(self) -> bool:
        """Whether the process that is up was started with a projector.

        Not ``configuration.sees``, which answers the different and equally
        real question of whether a compatible projector is *known*. Section 4's
        three states are OFF, TEXT_ONLY and VISION_LOADED, and this is the only
        thing that can tell the second from the third.
        """
        with self._lock:
            return self._running and self._projector is not None

    def loaded_projector(self) -> Path | None:
        """The projector the running process holds, or ``None``."""
        with self._lock:
            return self._projector if self._running else None

    def _launch(self, configuration: Config, placement: mc_llm_context.Placement,
                projector=None, plan: mc_llm_accel.Plan | None = None):
        """One llama-server, started and waited for. Returns it, its VRAM and its report.

        Raises :class:`_StartFailed` carrying llama.cpp's own reason rather
        than "llama-server exited before becoming ready", which is true of
        every failed start and useful for none of them.
        """
        from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD
        from prompt_master.inference.service import CPU_READY_TIMEOUT, GPU_READY_TIMEOUT

        # This card's, not the image side's: it is both what the start line
        # reports and what the residency is measured against afterwards, and a
        # reading of another card makes the second of those a subtraction of two
        # unrelated numbers.
        before = mc_broker.device_free_vram_bytes(card_of(configuration))
        layers = _layers_argument(placement, mc_gguf.describe(configuration.model))
        # Which program is started. The plan carries its own executable only
        # if it ever grows one; today this is the configured runtime, and the
        # line exists so that a plan which does grow one cannot be launched
        # against the wrong binary by omission.
        executable = plan.runtime if plan is not None and plan.runtime is not None \
            else configuration.runtime
        paths = mc_llm_paths.app_paths()
        paths.logs.mkdir(parents=True, exist_ok=True)
        log_path = paths.logs / "llama-server.log"
        # Where this start's log begins. The file is appended to across runs,
        # and what is read back afterwards has to be this run's.
        written_before = log_path.stat().st_size if log_path.exists() else 0

        logger.info(
            "Model Chain: %sstarting llama-server — %s on %s, %s, %s token context, "
            "%.1f GB free%s",
            self._said_for(),
            configuration.quantization or Path(configuration.model).stem,
            _device_label(configuration),
            placement.describe(),
            f"{placement.context:,}",
            before / _GB,
            "" if plan is None or plan.accelerator == mc_llm_accel.ACCEL_NONE
            else f", {mc_llm_accel.short_label(plan.accelerator)}",
        )
        # Said every time, and said in full. llama-server's own log is where the
        # answer lives when a placement and a reply speed disagree, and a log
        # nobody can find is a log nobody reads. It is one line per start, and
        # starts are rare.
        logger.info("Model Chain: llama-server log — %s", log_path)
        # Asked for only when this build advertises the flag. A runtime too old
        # for --parallel is not a reason to refuse to start; it is a reason to
        # run the one cache it has always run, and to say so once rather than
        # to fail a start on an argument it will not parse.
        asked = _slots_for(configuration, placement)
        if asked != placement.slots:
            placement = placement.with_slots(asked)
        _warn_about_an_idle_card(configuration, layers)
        # Ahead of the warning, and it is the same subject seen from a step
        # earlier: the warning says the machine is short of RAM for this model,
        # and this is the chance to stop being short of it by dropping warm
        # image cache nobody is executing against (section 10.8). When it
        # already fits, neither says anything and nothing moves.
        self._admit_host_ram(configuration, placement)
        _warn_about_system_ram(configuration, placement)
        self._log = (log_path, written_before)

        process = self._new_process()
        from_system_ram = (configuration.device.casefold() == CPU_DEVICE
                           or layers == NO_OFFLOAD)
        # The accelerator's flags go on last, so a build that spells
        # ``--flash-attn`` as a switch does not get it twice: the ordinary set
        # adds it for a resident placement, and an accelerator's flags are
        # filtered against what it already carries. See ``_launch_flags``.
        _arm_flags(_launch_flags(configuration, placement, plan))
        try:
            process.start(executable, configuration.model, projector,
                          configuration.gpu_index, configuration.device, placement.context,
                          log_path, gpu_layers=layers, slots=asked,
                          **_profile_arguments(configuration, placement))
            process.wait_ready(CPU_READY_TIMEOUT if from_system_ram else GPU_READY_TIMEOUT)
        except Exception as exc:
            process.stop()
            said = read_failure(_text_since(log_path, written_before))
            raise _StartFailed(said.text or str(exc), said.out_of_memory,
                               said.bad_argument, said.bad_value) from exc

        observed = self._observed_residency(before, placement, card_of(configuration))
        _check_slots(log_path, written_before, placement)
        return process, observed, _await_offload(log_path, written_before)

    @staticmethod
    def _observed_residency(before: int, placement: mc_llm_context.Placement,
                            card: int | None) -> int:
        """How much VRAM the server that just started actually took, on ``card``.

        ``card`` has no default, deliberately: the default *was* the bug. The
        measurement is a difference of
        two free-VRAM readings, and a difference is only meaningful when both
        readings are of the same card -- which they were not: ``before`` came
        from the card the server is being placed on and ``after`` came from
        whichever card the image side is using, because it was read with no
        argument at all.

        On one card that subtracts a number from itself an instant later and is
        very nearly right. On two it subtracts one card from another and is
        nonsense: from a user's log, a 5090 with 31.4 GB free and a 3090 with
        22.7 GB produced "8.7 GB VRAM" for a model that had taken about twenty,
        a warning that llama.cpp had "left the rest in system RAM" about a
        server running at 92 tokens a second, and 8.7 GB of phantom residency
        declared to the broker.

        ``None`` still means the image card, and is right for a placement that
        is on it. What it must not be is what a caller gets for saying nothing.

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

        def reading() -> int:
            return mc_broker.device_free_vram_bytes(card)

        observed = max(before - reading(), 0)
        if observed > 0 or not getattr(placement, "on_gpu", False):
            return observed
        if placement.gpu_layers == mc_llm_context.NO_LAYERS:
            return observed  # nothing was asked for; zero is the right answer

        for _attempt in range(RESIDENCY_SETTLE_ATTEMPTS):
            time.sleep(RESIDENCY_SETTLE_SECONDS)
            observed = max(before - reading(), 0)
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

    def _outgrown(self, configuration: Config, ours: int, vision: bool = False) -> bool:
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

            if mc_plan.current() is not None and self._plan_applies(configuration):
                overspending = self._overspending(ours)
                if not self._boundary_moved() and not overspending:
                    return False
                # A real boundary. Fall through and let the negotiation below
                # say whether the new plan is worth a restart; a boundary alone
                # is permission to ask, not an answer.
            elif self._image_job_conflicts(configuration):
                # "The host is busy" is not a fact about this server unless the
                # host's job is on the processor this server executes on
                # (section 8.4). A generation on GPU 0 is no reason to decline
                # to reconsider a placement on GPU 1 -- and it was: the
                # host-busy branch below is what a second card's role hit on
                # every request for the whole length of every generation.
                return False
        except Exception:
            logger.debug("Model Chain: could not check the plan boundary", exc_info=True)
            if self._image_job_conflicts(configuration):
                return False
        try:
            described = mc_gguf.describe(configuration.model)
            # ``vision`` and not ``False``: a preview that forgot the projector
            # this server is holding would price a placement nobody can have,
            # find it roomier than the one that is running, and restart a warm
            # vision-capable server to reach it. Section 21 -- once
            # VISION_LOADED, a request must not re-negotiate as though the
            # projector were absent.
            preview = negotiate(configuration, described, reclaim=False, already_ours=ours,
                                vision=vision)
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
            self._reconciled()
            return False
        logger.info("Model Chain: re-placing llama-server — %s now fits, where it is running %s",
                    preview.placement.describe(described.block_count if described else 0),
                    current.describe(described.block_count if described else 0))
        return True

    def _image_job_conflicts(self, configuration: Config | None = None) -> bool:
        """Whether a running image job is competing with this server's processor.

        Replaces a bare ``host_busy()`` in the two places where "busy" was
        being read as "busy with something that concerns you". It does not
        concern a runtime on another card or on the processor, and treating it
        as though it did is what froze a second card's placement decisions for
        the duration of every generation on the first (section 8.4).
        """
        settings = configuration if configuration is not None else self.configuration()
        try:
            mine = execution_domain(settings)
            if not (mc_broker.host_busy()
                    or mc_broker.active_family() == mc_broker.FAMILY_IMAGE):
                return False
            return mine.conflicts_with(mc_broker.image_execution_domain())
        except Exception:
            logger.debug("Model Chain: could not compare the image job's device",
                         exc_info=True)
            return True

    def _plan_applies(self, configuration: Config | None = None) -> bool:
        """Whether the active image plan says anything about *this* server.

        Section 8.2, and it is the difference between a language model that
        survives a generation on a second card and one that is stopped by a
        budget describing a card it has never touched. The plan is a statement
        about Forge's image card: how much of *that* card the image side needs
        and therefore how much of *that* card is left over. A runtime on
        another card is not competing for either number, and a processor
        runtime is not competing for VRAM at all.

        Unresolvable relationships answer yes, matching
        :func:`shares_the_image_card`: the cost of treating an unknown card as
        the image card is a language model sized more conservatively than it
        needed to be, and the cost of the opposite is an image generation that
        runs out of memory.
        """
        settings = configuration if configuration is not None else self.configuration()
        try:
            if not settings.uses_cuda_compute:
                return False
        except Exception:
            logger.debug("Model Chain: could not read the runtime's compute device",
                         exc_info=True)
            return True
        card = self._card if self._card is not None else card_of(settings)
        return shares_the_image_card(card, settings)

    def _boundary_moved(self) -> bool:
        """Whether the plan has changed since *this* server was placed under one.

        The per-runtime half of :func:`mc_plan.boundary_moved`. Same rule --
        a server placed with no plan in force is not re-placed merely because
        one has since appeared -- read from this runtime's own baseline, so
        two servers cannot answer each other's question (section 8.3, T13).
        """
        if self._placed_for is None:
            return False
        try:
            import mc_plan

            plan = mc_plan.current()
        except Exception:
            logger.debug("Model Chain: could not read the active plan", exc_info=True)
            return False
        if plan is None:
            return False
        return plan.identity() != self._placed_for

    def _note_placement(self) -> None:
        """Record the plan this runtime's placement answers.

        Only for a runtime the plan applies to. A different-card server that
        recorded boundaries would accumulate exactly the state section 8.3 says
        it must not: a baseline that later "moves", forcing a re-negotiation
        and a GGUF read for a plan about a card it is not on (T16).
        """
        try:
            import mc_plan

            plan = mc_plan.current() if self._plan_applies() else None
            self._placed_for = plan.identity() if plan is not None else None
            if plan is not None:
                # Kept in step for the panel and for anything still reading the
                # module-level value; the decision above is this runtime's.
                mc_plan.note_placement(plan)
        except Exception:
            logger.debug("Model Chain: could not record the placement plan", exc_info=True)

    def _reconciled(self) -> None:
        """Record that the running placement has been checked against this plan.

        The missing half of :func:`mc_plan.boundary_moved`, and the reason a
        warm request could pause for a second before every generation.

        ``mc_plan.note_placement`` was called from exactly one place -- the path
        that starts a server -- so a plan boundary that was examined and
        *declined* left the old plan recorded. ``boundary_moved`` then went on
        answering "yes" to every later request, and every one of them paid for
        the answer: a GGUF header re-read and a full re-negotiation, to reach
        the same conclusion as the request before it. It never healed, because
        the only thing that could record the new plan was the restart this
        function has just decided against.

        A boundary that has been considered has been considered whichever way it
        came out, so the plan is recorded here too. What that costs is nothing
        the overspend guard was doing: :meth:`_overspending` is measured on
        every call, before the boundary is consulted at all, and still forces
        the check through when a running server holds more than the plan leaves
        it.
        """
        self._note_placement()

    def _allowance(self, ours: int) -> int:
        """What the active plan leaves *this* server, or -1 when it says nothing.

        -1 is "not applicable", and it now covers two cases rather than one:
        no plan is published, or there is one and it does not describe this
        runtime's card (section 8.2). Both mean the same thing downstream --
        there is no plan-derived ceiling here -- which is why one value carries
        them: the alternative would be a second sentinel that every caller had
        to treat identically.
        """
        try:
            import mc_plan

            if mc_plan.current() is None or not self._plan_applies():
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
        mc_broker.declare(mc_broker.FAMILY_LLM, self.residency_key, self._label(configuration),
                          held or self.report.observed_bytes or estimated,
                          rank=mc_broker.RANK_HOT,
                          card=self._card if self._card is not None
                          else card_of(configuration))

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
                offload: Offload | None = None,
                plan: mc_llm_accel.Plan | None = None) -> None:
        # The placement's notes and the accelerator's, in one list. Both are
        # section 13's "say what you changed": a context that had to shrink and
        # an accelerator that stepped down are the same kind of news, and the
        # console and the panel print this list.
        notes = (*negotiated.notes, *(plan.notes if plan is not None else ()))
        self.report = Report(
            placement=negotiated.placement, estimate=negotiated.estimate,
            notes=notes, observed_bytes=observed, started_at=time.time(),
            model=Path(configuration.model).name if configuration.model else "",
            fits=negotiated.fits, offload=offload or Offload(),
            plan=plan or mc_llm_accel.Plan(),
        )
        for text in notes:
            mc_broker.note(mc_broker.FAMILY_LLM, text)

        # Which card these bytes are on, recorded with them. A residency the
        # broker cannot place on a card is a residency it must not target
        # (invariant I-11), and one it can is the whole reason a shortfall on
        # GPU 0 can leave a server on GPU 1 alone (invariant I-3).
        card = self._card if self._card is not None else card_of(configuration)
        if negotiated.placement.on_gpu and observed > 0:
            mc_broker.declare(mc_broker.FAMILY_LLM, self.residency_key, self._label(configuration),
                              observed, rank=mc_broker.RANK_HOT, card=card)
            mc_llm_context.record_observation(configuration.model, negotiated.placement, observed)
        elif negotiated.placement.on_gpu:
            mc_broker.declare(mc_broker.FAMILY_LLM, self.residency_key, self._label(configuration),
                              negotiated.estimate.resident_bytes, rank=mc_broker.RANK_HOT,
                              card=card)
        else:
            mc_broker.retire(self.residency_key)

        logger.info(
            "Model Chain: %sllama-server ready — %s, %s token context, %.1f GB VRAM%s",
            self._said_for(),
            negotiated.placement.describe(),
            f"{negotiated.placement.context:,}",
            observed / _GB,
            "" if not notes else f" ({'; '.join(notes)})",
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
        import mc_llm_roles

        name = configuration.quantization or (
            Path(configuration.model).stem if configuration.model else "the LLM")
        if not _roles_of(self):
            return f"the LLM ({name})"
        # Named roles rather than "the LLM", because with two servers up a
        # register that calls both of them the same thing is a register nobody
        # can read -- and because which one the broker is about to stop is
        # exactly what somebody reading that line needs to know.
        who = " and ".join(mc_llm_roles.label(role) for role in _roles_of(self))
        return f"the {who} LLM ({name})"

    # -- the broker's reclaimer ------------------------------------------- #

    def on_card(self, card) -> bool:
        """Whether releasing this runtime could add VRAM to ``card``.

        Defence in depth (section 7.4). The registry already filters victims by
        card, and this is the second lock on the same door: a direct caller, or
        a filter that regresses, would otherwise be able to stop a language
        model on GPU 1 to answer a shortage on GPU 0 -- which frees exactly
        zero bytes where they were wanted and costs a prompt cache, a process
        start and a model load to do it.

        Three answers rather than two:

        * a known different card is a refusal;
        * a processor runtime holds no VRAM anywhere, so it is not a VRAM
          victim for any card;
        * an unknown card is *also* not a targeted victim -- section 7.3 --
          because a runtime whose card cannot be named cannot be shown to help.
        """
        if isinstance(card, mc_broker._AnyCard):
            return True
        mine = self._card
        if mine is None:
            settings = self.configuration()
            try:
                if not settings.uses_cuda_compute:
                    return False
            except Exception:
                return False
            mine = card_of(settings)
        if mine is None or card is None:
            return False
        return int(mine) == int(card)

    def host_ram_bytes(self) -> int:
        """Host RAM this server materially needs, or 0. See :func:`host_ram_demand`."""
        with self._lock:
            if not self._running:
                return 0
            return host_ram_demand(self.configuration(), self._placement)

    def release(self, needed_bytes: int, reason: str = "", *,
                card=mc_broker.ANY_CARD) -> int:
        """Give VRAM back. Registered with the broker as the LLM family's reclaimer.

        ``needed_bytes`` is honoured in spirit rather than to the byte: a
        process either holds its allocation or it does not, and there is no
        partial surrender short of restarting somewhere else. So a request for
        any amount releases the whole of it, which is the only granularity the
        mechanism has -- and is why :func:`negotiate` works hard not to have to
        ask for image VRAM in the first place.

        ``card`` names the physical GPU the caller needs room on. A request for
        a card this server is not on is refused outright -- see
        :meth:`on_card` -- and the measurement below is taken on *that* card
        rather than on the image side's, because a release aimed at GPU 1 whose
        result was read from GPU 0 would report whatever the image job happened
        to be doing at the time (section 7.4).
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
        if not self.on_card(card):
            logger.info("Model Chain: %sllama-server was not asked to release for %s — it is "
                        "on %s and the memory is wanted on %s, so stopping it would free "
                        "nothing where it is needed", self._said_for(),
                        reason or "another workload",
                        mc_broker.cuda_execution(self._card).describe(),
                        mc_broker.cuda_execution(card).describe()
                        if card is not None else "an unidentified card")
            return 0
        if not self._lock.acquire(timeout=RELEASE_LOCK_TIMEOUT):
            logger.warning("Model Chain: the LLM runtime was busy and could not release VRAM "
                           "for %s", reason or "another workload")
            return 0
        try:
            if not self._running:
                mc_broker.retire(self.residency_key)
                return 0
            if self._placement is not None and not self._placement.on_gpu:
                return 0  # already in system RAM; there is nothing on the card to give

            where = self._card if self._card is not None else mc_broker._card_index(card)
            measure = (lambda: mc_broker.device_free_vram_bytes(where)) if where is not None \
                else mc_broker.free_vram_bytes
            before = measure()
            if _release_mode() == RELEASE_SYSTEM_RAM:
                freed = self._restart_in_system_ram(before, reason, measure)
                if freed:
                    return freed
            self._stop_locked(reason or "another workload needed the VRAM")
            freed = max(measure() - before, 0)
            mc_broker.note(
                mc_broker.FAMILY_LLM,
                f"stopped llama-server for {reason or 'another workload'}; the weights stay warm "
                f"in the system page cache, so restarting it re-reads from RAM, not disk"
            )
            return freed or self.report.observed_bytes
        finally:
            self._lock.release()

    def _restart_in_system_ram(self, before: int, reason: str, measure=None) -> int:
        """Move the model off the card without losing the loaded server.

        Checks the *destination* first, which is section 17.9's whole point and
        the clearest thing cross-domain awareness buys. This move solves a VRAM
        shortage by creating a host-RAM demand of roughly the model's size: on
        a machine with room that is exactly the right trade -- the server keeps
        answering, more slowly, and its prompt cache survives. On a machine
        already near its host floor it is not a trade at all, it is swapping a
        problem the driver can spill around for one that pages the whole
        desktop, so stopping the server is the safer reclaim and the file pages
        stay soft-warm in the OS cache for the next start (section 18.11).
        """
        from prompt_master.inference.device_detection import NO_OFFLOAD
        from prompt_master.inference.service import CPU_READY_TIMEOUT

        configuration = self.configuration()
        if not configuration.configured:
            return 0

        wanted = host_ram_demand(configuration,
                                 (self._placement or mc_llm_context.Placement()).with_layers(
                                     mc_llm_context.NO_LAYERS))
        if wanted > 0 and not mc_broker.host_ram_fits(wanted):
            admission = mc_broker.admit_host_ram(
                wanted,
                reason=f"moving {_backbone_label(configuration)} out of VRAM")
            if not admission.fits:
                mc_broker.note(
                    mc_broker.FAMILY_LLM,
                    f"llama-server was stopped rather than moved to system RAM for "
                    f"{reason or 'another workload'}: the move needs about "
                    f"{wanted / _GB:.1f} GB of host memory and only "
                    f"{admission.available / _GB:.1f} GB is available above the "
                    f"{admission.reserve / _GB:.1f} GB reserve. A VRAM shortage is not "
                    f"worth solving by exhausting system RAM")
                return 0
        # Read before the stop clears it. This move is a *relocation* of the
        # server that is running, so the capability it comes back with has to
        # be the one it went away with: starting from ``configuration.mmproj``
        # would give a text-only server eyes it never had -- a projector loaded
        # into system RAM to satisfy nothing -- and starting from ``None`` would
        # take vision away from a conversation that is mid-picture, which
        # section 7 lists nowhere among the ways stickiness ends.
        projector = self._projector
        try:
            self._stop_locked("moving the LLM to system RAM")
            process = self._new_process()
            paths = mc_llm_paths.app_paths()
            paths.logs.mkdir(parents=True, exist_ok=True)
            placement = (self._placement or mc_llm_context.Placement()).with_layers(
                mc_llm_context.NO_LAYERS)
            process.start(configuration.runtime, configuration.model, projector,
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
        self._projector = projector
        self._signature = (configuration.runtime, configuration.model, projector,
                           configuration.gpu_index, configuration.device,
                           placement.context, placement.gpu_layers)
        self._identity = _identity(configuration, projector)
        # A relocated placement is a new placement, so the boundary it was
        # reconciled against no longer describes anything (section 8.3).
        self._placed_for = None
        mc_broker.retire(self.residency_key)
        freed = max((measure() if measure is not None else mc_broker.free_vram_bytes())
                    - before, 0)
        mc_broker.note(mc_broker.FAMILY_LLM,
                       f"moved the LLM to system RAM for {reason or 'another workload'}; it stays "
                       f"loaded and answering, more slowly")
        return freed or self.report.observed_bytes

    def resident_bytes(self, *, card=mc_broker.ANY_CARD) -> int:
        """VRAM this server holds, on ``card`` when one is named.

        Zero for another card, and that is not a rounding-down -- it is the
        honest answer to the question asked. Reporting this runtime's bytes
        under another card's heading is how an image-card budget on a 3090
        comes to subtract a 5090's nineteen gigabytes (invariant I-2).
        """
        if not self.on_card(card):
            return 0
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
        remember_speed(prompt, reply, placement=self.measurement_token())
        measured = f"llama.cpp measured {reply:.1f} tokens/s"
        said = f"{measured}, prompt at {prompt:.0f} tokens/s" if prompt > 0 else measured
        accepted = self.speculation().describe()
        return f"{said}, {accepted}" if accepted else said

    def measurement_token(self) -> str:
        """The key this server's measured rates belong under.

        Composed from the running placement, the accelerator that is actually
        running and the physical card -- so an MTP rate on a 5090 and an
        ordinary rate on a 3090 are two measurements of two things rather than
        one average of neither.
        """
        with self._lock:
            configuration = self.configuration()
            return measurement_token(self._placement, self.report.plan.accelerator,
                                     card_of(configuration))

    def speculation(self) -> Speculation:
        """What llama.cpp reported about drafted and accepted tokens, if anything.

        Empty for an ordinary run, which is not a missing measurement -- there
        was nothing to draft. Empty too for an accelerated run on a build that
        does not print the counters, and the two are told apart by
        :attr:`Speculation.known` rather than by a zero.
        """
        with self._lock:
            accelerated = self.report.plan.accelerator not in (
                mc_llm_accel.ACCEL_NONE, mc_llm_accel.ACCEL_AUTO)
            if self._log is None or not self._running or not accelerated:
                return Speculation()
            path, offset = self._log
        return read_speculation(_text_since(path, offset, tail=_TIMING_TAIL))

    def describe(self) -> str:
        return self._label(self.configuration())

    def stop(self) -> None:
        with self._lock:
            self._stop_locked("stop requested")

    def _forget_placement_plan(self) -> None:
        """Let the next start re-decide, because this one is no longer running.

        Called wherever the server stops for a reason that was not a change of
        plan -- an emergency eviction, a demotion to system RAM, a shutdown.
        Without it the plan the dead server was placed for stays recorded, and
        :meth:`_boundary_moved` goes on answering "no" about a placement that
        no longer exists.

        This runtime's own baseline always clears (section 8.3: stopping,
        replacing or demoting clears the applicable baseline). The module-level
        one clears only when this runtime is the one it describes -- with two
        servers up, a Creative role stopping is no reason to tell the panel
        that a Spatial role placed under a plan was not.
        """
        mine, self._placed_for = self._placed_for, None
        try:
            import mc_plan

            if mine is None or mc_plan.placed_for() == mine:
                mc_plan.note_placement(None)
        except Exception:
            logger.debug("Model Chain: could not clear the recorded placement plan",
                         exc_info=True)

    def _stop_locked(self, reason: str) -> None:
        process, self._process = self._process, None
        held = self.report.observed_bytes if self._placement is not None else 0
        said_vision = self._projector is not None
        self._signature, self._identity, self._placement = None, None, None
        # Section 7: vision residency ends when the server does, and only then.
        # Cleared with the rest of the running process's state rather than by
        # any request, so the next start is asked what it needs rather than
        # inheriting what a dead process happened to hold.
        self._projector = None
        # Cleared with the rest of the running server's identity. A stale
        # accelerator tuple surviving a stop is how a speculative flag comes
        # to be believed about a server that is not running, and the guarantee
        # wanted is the opposite one: no such flag reaches an ordinary start.
        self._accelerator = ()
        self._log = None
        mc_broker.retire(self.residency_key)
        self._forget_placement_plan()
        if process is None:
            return
        try:
            process.stop()
        except Exception:
            logger.warning("Model Chain: failed to stop llama-server (%s)", reason, exc_info=True)
            return
        logger.info("Model Chain: llama-server stopped — %s%s%s", reason,
                    f", {held / _GB:.1f} GB of VRAM released" if held else "",
                    "; its vision projector is no longer resident" if said_vision else "")

    # -- status ----------------------------------------------------------- #

    def status(self) -> dict:
        """Everything the panel needs about the runtime, and nothing that costs a load."""
        with self._lock:
            configuration = self.configuration()
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
                # Two answers because they are two facts (section 10): "sees"
                # is whether a compatible projector is known, and this is
                # whether the process that is up has actually loaded it.
                "vision_loaded": self._running and self._projector is not None,
                "placement": self._placement,
                "report": self.report,
                "resident_bytes": self.resident_bytes(),
                "accelerator": self.report.plan.accelerator,
                "memory_priority": configuration.memory_priority,
            }


# --------------------------------------------------------------------------- #
# Role-specific runtimes (design intent sections 9, 10 and 17)
# --------------------------------------------------------------------------- #

OPT_LLM_SLOTS = "model_chain_llm_prompt_caches"

SLOTS_AUTOMATIC = "auto"

SLOT_MODES = (
    (SLOTS_AUTOMATIC, "Automatic — as many as the card has room for once the model fits"),
    ("1", "One — every mode shares a single cache, as before"),
    ("2", "Two"),
    ("3", "Three — a cache each for Conversation, the Krea writer and the Composer"),
    ("4", "Four"),
    ("6", "Six — a cache for every mode that has its own system prompt"),
)
"""How many warm prompt caches llama.cpp keeps (``--parallel``).

Each mode here opens with a different system prompt -- Conversation, Prompt
Studio, MiniMax, the Krea writer, the Spatial Composer -- and with one cache
between them every switch re-reads a prefix the previous one had just cached.
With a cache each, llama.cpp routes an incoming prompt to the slot whose cached
prefix matches best and none of them evicts another. One process, one copy of
the weights, several caches.

The cost is a key/value cache per slot, which is why Automatic is defined the
way it is: see :func:`_slots_that_fit`.
"""

SLOT_CEILING = 8
"""Most caches anybody may ask for. Past this the cache is not the bottleneck."""

OPT_ROLE_PROCESSES = "model_chain_llm_role_processes"

PROCESSES_SHARED = "shared"
PROCESSES_SEPARATE = "separate"

PROCESS_MODES = (
    (PROCESSES_SHARED, "One server — identical roles share it, and its prompt cache"),
    (PROCESSES_SEPARATE, "One each — a server per role even when they are identical"),
)
"""What to do when Creative and Spatial are configured *identically*.

The other half of the memory question, and the half the design intent left
optional (section 10.3). Sharing is right when the memory is tight: one process,
one copy of the weights, one cache. It is wrong when it is not, and a user with
a 32 GB card said so plainly -- two servers on that card both stay warm, and
each keeps its own system prompt, so neither pass ever re-reads the other's.

Section 10.2's handoff cost is the thing being bought off here. Two roles on one
server switch system prompts, and switching system prompts re-reads a prefix
llama.cpp had cached; two servers never do. That is the entire trade, and it is
a memory decision, so it is the user's.
"""

OPT_ROLE_SHARING = "model_chain_llm_role_sharing"

SHARE_AUTO = "auto"
SHARE_TAKE_TURNS = "take_turns"
SHARE_COEXIST = "coexist"

SHARING_MODES = (
    (SHARE_AUTO, "Automatic — take turns on one card, coexist in system RAM"),
    (SHARE_TAKE_TURNS, "Take turns — stop one role's server before starting the other's"),
    (SHARE_COEXIST, "Coexist — leave both running and let them compete"),
)
"""What to do when Creative and Spatial land in the same memory.

Only reached when the two roles are configured *differently* and still point at
the same pool: identical configurations share one server outright and have
nothing to decide. Two servers cannot share a process, so "one" here means one
at a time -- the second start stops the first through the release path that
already exists, which is the same mechanism the image side uses and obeys the
same rules.

Automatic is not a fourth policy, it is the two above chosen per pool. Two
servers in system RAM coexist happily on a machine with the RAM for them, and
taking turns there would buy a model reload per role switch for nothing. Two
servers on one card are the case the user described as "fighting over what is
left", so that one takes turns. Either can be forced.
"""

POOL_SYSTEM_RAM = "ram"
"""Where the weights live for CPU and Mixed Conservative alike.

Conservative belongs here and not with the card it names: its whole promise is
that no model layer is resident in VRAM, so what two Conservative roles contend
for is system RAM, exactly as two CPU roles do. The card they are pointed at is
holding a context and compute buffers, which is real but is not the thing that
runs a machine out of memory.
"""


def pool(configuration: Config) -> str:
    """Which memory ``configuration`` will actually fill.

    The question the sharing option is asked about, and it is deliberately
    about *weights* rather than about devices. A role on CUDA0 in Conservative
    and a role on the processor are both spending system RAM and neither is
    spending the card, so they are in one pool; a role on CUDA0 in Aggressive is
    in the card's.
    """
    from prompt_master.core.models import (
        CPU_MODE, MIXED_CONSERVATIVE_MODE, normalise_mode)

    mode = normalise_mode(configuration.mode)
    if mode in (CPU_MODE, MIXED_CONSERVATIVE_MODE) or not configuration.uses_cuda_compute:
        return POOL_SYSTEM_RAM
    return f"cuda:{int(configuration.gpu_index)}"


def _process_mode() -> str:
    return mc_broker.resolve(mc_broker.option(OPT_ROLE_PROCESSES, PROCESSES_SHARED),
                             PROCESS_MODES, PROCESSES_SHARED)


def separate_processes() -> bool:
    """Whether identical roles should still get a server each."""
    return _process_mode() == PROCESSES_SEPARATE


def _sharing_mode() -> str:
    return mc_broker.resolve(mc_broker.option(OPT_ROLE_SHARING, SHARE_AUTO),
                             SHARING_MODES, SHARE_AUTO)


def resolved_sharing(where: str, chosen: str = "") -> str:
    """:data:`SHARE_TAKE_TURNS` or :data:`SHARE_COEXIST` for a pool.

    Automatic resolved to one of the two, so that every caller downstream is
    looking at a decision rather than at a policy that still has to be applied.
    """
    picked = chosen or _sharing_mode()
    if picked in (SHARE_TAKE_TURNS, SHARE_COEXIST):
        return picked
    return SHARE_COEXIST if where == POOL_SYSTEM_RAM else SHARE_TAKE_TURNS


class RuntimeRegistry:
    """The llama-servers this installation is running, one per distinct identity.

    Sections 9 and 10. A role asks for its runtime and gets one back; two roles
    whose complete resolved identity matches get the *same* one back, which is
    the whole of "shared-runtime coalescing" -- there is no branch anywhere that
    decides to share, only an identity that turns out to be equal.

    That is why the identity comes from :func:`_identity`, the function that
    already decides when a *single* runtime has to be restarted. The two
    questions are the same question asked from different directions: settings
    that would force a restart are exactly the settings two roles cannot share
    a process across. Keeping one answer for both is what stops the registry
    handing back a server that the next request would immediately replace.

    The registry is also the broker's reclaimer for the LLM family, in place of
    any one runtime, because with two servers up the broker asking one of them
    to give VRAM back would free half of what it asked for and be told it had
    freed all of it.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._runtimes: dict[tuple, Runtime] = {}

    # -- resolution ------------------------------------------------------- #

    def for_role(self, role: str = "", configuration: Config | None = None) -> Runtime:
        """The runtime that serves ``role``, started or not.

        Never starts anything: what comes back is the object that owns that
        role's server, and asking it for a client is what starts one. So this
        is safe to call from a status panel, which is where half the callers
        are.
        """
        import mc_llm_roles

        chosen = mc_llm_roles.named(role)
        configuration = configuration or config(chosen)
        return self._runtime_for(chosen, configuration)

    def _runtime_for(self, role: str, configuration: Config) -> Runtime:
        import mc_llm_roles

        key = self.key_for(role, configuration)
        with self._lock:
            found = self._runtimes.get(key)
            if found is None:
                found = self._instance(key)
                self._runtimes[key] = found
            try:
                found.adopt(role, key)
            except AttributeError:
                logger.debug("Model Chain: %s cannot be told which role it serves",
                             type(found).__name__)
            if role and role not in _roles_of(found):
                try:
                    found.roles = tuple(sorted({*_roles_of(found), role},
                                               key=mc_llm_roles.ROLES.index))
                except AttributeError:
                    # A stand-in for the module singleton -- a test double, or
                    # anything else somebody has substituted. It can still serve
                    # requests; it just cannot be labelled, and labelling is the
                    # least important thing this registry does.
                    logger.debug("Model Chain: %s cannot record which role it serves",
                                 type(found).__name__)
            self._forget_stale(key, role)
            return found

    def _instance(self, key: tuple) -> Runtime:
        """A new runtime, or the module singleton when this is its identity.

        The singleton is not a legacy wart to route around: it is the server
        every mode that is not a role already uses, and an installation with no
        role split resolves both roles to exactly its identity. Adopting it
        there is what makes "nothing is configured differently" mean *one*
        llama-server rather than three -- the shared one, plus one per role
        pointing at the same model with the same settings.
        """
        try:
            shared = config()
            if key == _identity(shared, shared.mmproj) and not self._holds(runtime):
                runtime.residency_key = _residency_key(key)
                return runtime
        except Exception:
            logger.debug("Model Chain: could not compare the shared runtime identity",
                         exc_info=True)
        return Runtime(residency_key=_residency_key(key))

    def _holds(self, wanted: Runtime) -> bool:
        return any(found is wanted for found in self._runtimes.values())

    def _forget_stale(self, keeping: tuple, role: str) -> None:
        """Drop ``role`` from any other runtime, and forget one nobody wants.

        A role that has just been reconfigured leaves its old identity behind.
        Keeping the entry would keep a stopped server's residency key in the
        register and would make ``all()`` report a runtime nothing can reach.
        A runtime still claimed by the other role is left exactly as it is --
        that is the un-sharing case, and the role that stayed put must not have
        its server taken away because the other one moved.
        """
        if not role:
            return
        for key, existing in list(self._runtimes.items()):
            if key == keeping or role not in _roles_of(existing):
                continue
            existing.roles = tuple(other for other in _roles_of(existing) if other != role)
            if not existing.roles and not _is_running(existing):
                self._runtimes.pop(key, None)

    @staticmethod
    def key_for(role: str, configuration: Config) -> tuple:
        """What this role's runtime is filed under.

        The resolved identity, and the role beside it when the user has asked
        for a server each. Two roles cannot be told apart by an identity that is
        equal by definition, so the only way to give them separate processes is
        to file them separately -- everything downstream then follows, because
        the registry has never decided anything except by this key.
        """
        key = _identity(configuration, configuration.mmproj)
        if role and separate_processes():
            return (*key, role)
        return key

    def shared(self) -> bool:
        """Whether both roles currently resolve to one server."""
        import mc_llm_roles

        try:
            keys = {self.key_for(role, config(role)) for role in mc_llm_roles.ROLES}
        except Exception:
            logger.debug("Model Chain: could not compare the role runtimes", exc_info=True)
            return True
        return len(keys) == 1

    def contending(self) -> str:
        """The pool both roles are spending, or ``""`` when they are not sharing one.

        Empty when the roles coalesce -- one server is not two servers competing
        -- and empty when they are in different pools, which is the arrangement
        the scenarios companion recommends and which needs no policy at all.
        """
        import mc_llm_roles

        if self.shared():
            return ""
        try:
            pools = {pool(config(role)) for role in mc_llm_roles.ROLES}
        except Exception:
            logger.debug("Model Chain: could not compare the role pools", exc_info=True)
            return ""
        return pools.pop() if len(pools) == 1 else ""

    def all(self) -> tuple:
        """Every runtime this registry knows about, the shared one included.

        The singleton is unioned in even before any role has asked for it,
        because the broker registers this object as the LLM family's reclaimer
        at import and may ask what is resident long before Creative Mode runs.
        A registry that answered "nothing" then would be telling the image side
        there was VRAM to take that a running llama-server was holding.
        """
        with self._lock:
            found = list(self._runtimes.values())
            if not self._holds(runtime):
                found.append(runtime)
            return tuple(found)

    def running(self, *, card=mc_broker.ANY_CARD) -> tuple:
        """Every running server, or only those executing on ``card``.

        Executing rather than holding: the broker asks this to decide whether
        a CUDA context of ours explains VRAM on a card, and a server with its
        weights in system RAM still has one on the card it names.
        """
        return tuple(found for found in self.all()
                     if _is_running(found) and _executes_on(found, card))

    def forget(self) -> None:
        """Drop every entry. Stops nothing -- for tests and for a settings reset."""
        with self._lock:
            self._runtimes.clear()

    # -- taking turns ----------------------------------------------------- #

    def make_room_for(self, role: str, configuration: Config) -> int:
        """Stop the other role's server when the two may not coexist.

        Called on the way into a start rather than after one, because the point
        is that the memory is free *before* the next placement is negotiated
        against it -- a negotiation that reads the card while the other role is
        still holding it places this one in the gap, which is the flapping the
        placement rules already exist to prevent.

        Returns bytes released, and does nothing at all in the two cases that
        are not a contention: roles that share a runtime, and roles in different
        pools.
        """
        import mc_llm_roles

        chosen = mc_llm_roles.named(role)
        if not chosen:
            return 0
        where = pool(configuration)
        sharing = resolved_sharing(where)
        if sharing != SHARE_TAKE_TURNS:
            if where == POOL_SYSTEM_RAM and not self._can_coexist_in_ram(chosen, configuration):
                # Section 11.4. "Coexist" is a preference about *warmth*, not a
                # licence to ignore the host's safety floor -- a second
                # RAM-backed server admitted below it does not run faster, it
                # pages, and takes the desktop with it. So the warm image cache
                # is asked first (inside :meth:`_can_coexist_in_ram`), and only
                # when that is not enough does this become a contention at all.
                if _sharing_mode() == SHARE_COEXIST:
                    logger.warning(
                        "Model Chain: %sboth roles are configured for system RAM and asked to "
                        "coexist, but there is not enough of it left above the safety reserve "
                        "after releasing what warm image cache could be released. The other "
                        "server is being left up; this one may load slowly or not at all.",
                        mc_llm_roles.prefix(chosen))
                    return 0
                logger.info("Model Chain: %ssystem RAM cannot safely hold both roles' models, "
                            "so they take turns", mc_llm_roles.prefix(chosen))
            else:
                return 0
        mine = self.key_for(chosen, configuration)
        freed = 0
        stood_down = 0
        for other in self.all():
            if other is self._runtimes.get(mine) or not other.running():
                continue
            serves = _roles_of(other)
            if serves and not [name for name in serves if name != chosen]:
                continue  # this role's own server, under another key
            try:
                # A runtime nobody has claimed is the *shared* one -- the server
                # Conversation, Prompt Studio, MiniMax and LLM Studio all use --
                # and it was being skipped, because "serves no role other than
                # mine" and "serves no role at all" read the same way to the
                # test that used to be here.
                #
                # It is also, routinely, the largest thing on the card. From a
                # user's log: a conversation left a 20 GB server up, and the two
                # roles then took turns in the 11 GB that was left, each getting
                # 25 of 65 layers and four tokens a second, stopping and
                # reloading a model on every switch. Take turns has to mean all
                # of our servers or it does not mean anything.
                elsewhere = next((name for name in serves if name != chosen), "")
                if pool(config(elsewhere)) != where:
                    continue
                freed += int(other.release(0, f"the {mc_llm_roles.label(chosen)} runtime "
                                              f"needs the same memory") or 0)
                stood_down += 1
            except Exception:
                logger.debug("Model Chain: could not stand down the other runtime",
                             exc_info=True)
        if stood_down:
            logger.info("Model Chain: %sstood %d other llama-server(s) down — %.1f GB, "
                        "they are configured for the same memory",
                        mc_llm_roles.prefix(chosen), stood_down, freed / _GB)
        return freed

    def _can_coexist_in_ram(self, role: str, configuration: Config) -> bool:
        """Whether this role's model fits in host RAM beside what is already there.

        Asked only when the sharing policy would otherwise have said "coexist",
        and answered by the same admission path everything else uses: read live
        available memory, and if the demand does not fit above the reserve, ask
        the explicitly managed warm image cache to give ground before deciding
        that two language models are in conflict at all (section 10.8, steps
        1-6).

        True when the demand cannot be sized, which preserves today's
        behaviour. A model whose size cannot be read is not evidence of
        pressure, and inventing pressure would stand a working server down for
        a number nobody has.
        """
        wanted = host_ram_demand(configuration, _requested_placement(configuration, None))
        if wanted <= 0:
            return True
        if mc_broker.host_ram_fits(wanted):
            return True
        admission = mc_broker.admit_host_ram(
            wanted, reason=f"the {role} role's model in system RAM")
        return admission.fits

    # -- the broker's reclaimer, fanned out ------------------------------- #

    def release(self, needed_bytes: int, reason: str = "", *,
                card=mc_broker.ANY_CARD) -> int:
        """Give VRAM back from every eligible runtime until ``needed_bytes`` is covered.

        Ordered by what each is holding, largest first, so the fewest servers
        are stopped for the memory asked for. A request for zero -- which is how
        the broker spells "everything" -- goes to all of them.

        ``card`` narrows "eligible" to runtimes whose reclaimable VRAM is on
        that card (section 7.3). Runtimes on another known card are *absent*
        from the victim set rather than ranked last; processor runtimes are
        absent because they hold no VRAM anywhere; and a runtime whose card
        cannot be resolved is absent too, because a targeted release has to be
        able to show that it helps. Every one of those exclusions is
        invariant I-3 -- a nineteen-gigabyte model on GPU 1 is not a smaller
        answer to a four-gigabyte shortage on GPU 0, it is not an answer at all.
        """
        eligible = [found for found in self.running() if self._may_release(found, card)]
        held = sorted(eligible, key=lambda found: -_held_by(found, card))
        freed = 0
        skipped = [found for found in self.running() if found not in eligible]
        if skipped and not isinstance(card, mc_broker._AnyCard):
            logger.info("Model Chain: %d llama-server(s) were not considered for %s — their "
                        "VRAM is not on %s", len(skipped), reason or "this reclaim",
                        mc_broker.cuda_execution(card).describe() if card is not None
                        else "the requested card")
        for found in held:
            if needed_bytes and freed >= needed_bytes:
                break
            try:
                freed += int(self._ask_release(found, max(needed_bytes - freed, 0),
                                               reason, card) or 0)
            except Exception:
                logger.debug("Model Chain: a runtime could not release", exc_info=True)
        return freed

    @staticmethod
    def _may_release(found, card) -> bool:
        if isinstance(card, mc_broker._AnyCard):
            return True
        asking = getattr(found, "on_card", None)
        if callable(asking):
            try:
                return bool(asking(card))
            except Exception:
                logger.debug("Model Chain: could not ask a runtime which card it is on",
                             exc_info=True)
                return False
        return False

    @staticmethod
    def _ask_release(found, needed: int, reason: str, card):
        if isinstance(card, mc_broker._AnyCard):
            return found.release(needed, reason)
        try:
            return found.release(needed, reason, card=card)
        except TypeError:
            logger.debug("Model Chain: %s cannot take a card-scoped release",
                         type(found).__name__)
            return 0

    def resident_bytes(self, *, card=mc_broker.ANY_CARD) -> int:
        return sum(_held_by(found, card) for found in self.all())

    def host_ram_bytes(self) -> int:
        """System RAM the running servers materially need. See :func:`host_ram_demand`."""
        total = 0
        for found in self.running():
            asking = getattr(found, "host_ram_bytes", None)
            if not callable(asking):
                continue
            try:
                total += max(int(asking() or 0), 0)
            except Exception:
                logger.debug("Model Chain: could not read a runtime's host-RAM demand",
                             exc_info=True)
        return total

    def describe(self) -> str:
        import mc_llm_roles

        live = self.running()
        if not live:
            return "the LLM"
        if len(live) == 1 and not live[0].roles:
            return live[0].describe()
        return " and ".join(
            f"{mc_llm_roles.label(_roles_of(found)[0]) if _roles_of(found) else 'the'} LLM"
            for found in live)


def _roles_of(found) -> tuple:
    """Which roles ``found`` serves, for an object that may not be a Runtime.

    ``mc_llm_runtime.runtime`` is a module-level name, and a module-level name
    is something callers substitute -- a test double stands in for it in several
    files here, and anything that does so is under no obligation to grow the
    attributes this registry added. Every question the registry asks of the
    objects it holds goes through one of these three helpers, so a stand-in
    degrades to "serves nobody in particular, holds nothing, is not running"
    rather than raising into a generation.
    """
    return tuple(getattr(found, "roles", ()) or ())


def _is_running(found) -> bool:
    try:
        return bool(found.running())
    except Exception:
        logger.debug("Model Chain: could not ask a runtime whether it is up", exc_info=True)
        return False


def _executes_on(found, card) -> bool:
    """Whether ``found`` issues CUDA work on ``card``.

    Execution, not residency -- so a Mixed Conservative server with every
    weight in system RAM counts here for the card it names. That is what makes
    the CUDA-context allowance in :func:`mc_broker.unaccounted_bytes` land on
    the right card, and what stops GPU 0's status hiding a gigabyte as "our own
    LLM context" for a process that is entirely on GPU 1 (section 7.6, T21).
    """
    if isinstance(card, mc_broker._AnyCard):
        return True
    try:
        settings = found.configuration()
        if not settings.uses_cuda_compute:
            return False
        mine = getattr(found, "_card", None)
        if mine is None:
            mine = card_of(settings)
    except Exception:
        logger.debug("Model Chain: could not ask a runtime which card it uses", exc_info=True)
        return False
    if mine is None or card is None:
        return False
    return int(mine) == int(card)


def _held_by(found, card=mc_broker.ANY_CARD) -> int:
    try:
        if isinstance(card, mc_broker._AnyCard):
            return max(int(found.resident_bytes() or 0), 0)
        return max(int(found.resident_bytes(card=card) or 0), 0)
    except TypeError:
        # A stand-in that predates card filtering. Answering with its
        # machine-wide total under one card's heading is exactly the mistake
        # this parameter exists to prevent, so it answers nothing instead.
        logger.debug("Model Chain: %s cannot report residency for one card",
                     type(found).__name__)
        return 0
    except Exception:
        logger.debug("Model Chain: could not ask a runtime what it holds", exc_info=True)
        return 0


def _residency_key(identity: tuple) -> str:
    """One register key per distinct runtime.

    Hashed rather than spelled out because the identity carries absolute paths,
    and a register the panel prints is not the place for somebody's model
    directory. Stable within a session, which is all the register needs.
    """
    import hashlib

    digest = hashlib.blake2b(repr(identity).encode("utf-8"), digest_size=6).hexdigest()
    return f"{RESIDENCY_KEY}:{digest}"


runtime = Runtime()
"""The single managed server. One process per WebUI, as the standalone app had
one per window: llama.cpp serialises requests within a process anyway, and two
would double the VRAM for no concurrency the broker would allow to be used."""

registry = RuntimeRegistry()
"""Every managed llama-server, by identity. See :class:`RuntimeRegistry`."""

mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, registry)


def shutdown() -> None:
    """Stop every llama-server this extension started. Never raises.

    *Every* one, not the shared singleton: an installation with the Creative
    and Spatial roles split has two servers up, and a shutdown that stopped one
    of them left the other holding twenty gigabytes with nothing left in the
    process that knew about it.

    The stray sweep after it is the belt to that pair of braces. A server whose
    Python handle was lost -- a reload, an exception during a start, an
    extension update in place -- is not in the registry to be stopped and is
    still very much running, and it is recognisable in the process list by the
    ``--alias`` every one of ours carries. See :func:`strays`.
    """
    for found in registry.all():
        try:
            found.stop()
        except Exception:
            logger.warning("Model Chain: failed to stop %s", found.describe(), exc_info=True)
    try:
        stopped, freed = release_strays()
    except Exception:
        logger.debug("Model Chain: the stray sweep failed during shutdown", exc_info=True)
        return
    if stopped:
        logger.info("Model Chain: stopped %d llama-server process%s on shutdown, releasing "
                    "%.1f GB", stopped, "" if stopped == 1 else "es", freed / _GB)


_shutdown_registered = False
_shutdown_lock = threading.Lock()


def stop_on_exit() -> None:
    """Arrange for the servers to be stopped when this process ends.

    The reported symptom: *"if I kill the webui process, there tends to be
    llama-server.exe running on my system."* Three different exits, and the
    extension was covered for none of them.

    ``on_script_unloaded`` is Forge asking an extension to tidy up, and it does
    not fire when somebody closes the window or kills the process -- so it was
    the one exit already handled and the one exit nobody performs.

    ``atexit`` covers the ordinary ones: the interpreter finishing, and Ctrl+C
    unwinding out of the top of it. It is registered once, from the first thing
    that touches the runtime, rather than at import -- an installation that
    never starts a language model should not register a hook to stop one.

    The hard kill is the third, and no handler inside a process that is being
    killed can run at all. That one is answered where the child is started
    instead -- see :func:`_die_with_us`.
    """
    global _shutdown_registered

    with _shutdown_lock:
        if _shutdown_registered:
            return
        _shutdown_registered = True
    import atexit

    atexit.register(_at_exit)
    for name in ("SIGTERM", "SIGINT"):
        _relay_signal(name)


def _at_exit() -> None:
    """``shutdown``, with every failure swallowed. Nothing may raise here."""
    try:
        shutdown()
    except Exception:
        logger.debug("Model Chain: the shutdown hook failed", exc_info=True)


def _relay_signal(name: str) -> None:
    """Stop the servers on ``name``, then do whatever was going to happen.

    Chained rather than replaced: the WebUI installs its own handlers and a
    handler of ours that swallowed the signal would leave a window nobody can
    close. What this adds is a stop in front of whatever was already there.

    Only from the main thread, because that is the only one Python will let
    install a handler, and only where the signal exists.
    """
    import signal

    number = getattr(signal, name, None)
    if number is None:
        return
    try:
        previous = signal.getsignal(number)
    except (OSError, ValueError):
        return

    def handler(received, frame):
        _at_exit()
        if callable(previous) and previous not in (signal.SIG_IGN, signal.SIG_DFL):
            previous(received, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(received, signal.SIG_DFL)
            signal.raise_signal(received)

    try:
        signal.signal(number, handler)
    except (OSError, ValueError):
        # Not the main thread, or a platform without it. The atexit hook and
        # the job object below both still apply.
        logger.debug("Model Chain: could not chain the %s handler", name, exc_info=True)


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
    """The shared runtime's server pid, or 0. See :func:`_own_pids`."""
    return _pid_of(runtime)


def _pid_of(found) -> int:
    process = getattr(found, "_process", None)
    inner = getattr(process, "process", None) if process is not None else None
    return int(getattr(inner, "pid", 0) or 0)


def _own_pids() -> set:
    """Every llama-server pid this WebUI started, across every runtime.

    A stray is a server nothing here has a handle to. Asking only the shared
    runtime made that definition wrong the moment a role could have a server of
    its own: two perfectly live role servers answered to nobody's handle as far
    as this function was concerned, so Unload reported them as strays and killed
    them -- and any future sweep would have done the same mid-generation.
    """
    pids = {_pid_of(runtime)}
    try:
        pids.update(_pid_of(found) for found in registry.all())
    except Exception:
        logger.debug("Model Chain: could not enumerate the runtimes' server pids",
                     exc_info=True)
    return {pid for pid in pids if pid}


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

    ours = _own_pids()
    found: list[int] = []
    try:
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if process.info["pid"] in ours or process.info["pid"] == os.getpid():
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
