"""Cross-workload residency policy for image models and local LLMs.

Sections 7-10, 13, 14 and 15 of the design intent. ModelSwitch already owns
image residency; this module is what makes it own *cross-workload* residency,
so that a llama.cpp process and a Forge checkpoint stop making unilateral
decisions about the same card.

What this module is and is not
------------------------------
It is policy and coordination. It decides *whether* something has to move,
*which* thing, and *how much*, and it records why so the panel can say so.

It is emphatically not a tensor mover (section 8, section 17). Image weights
are moved by ``mc_memory``, which moves them through Forge's own memory
manager; LLM weights are moved by restarting llama.cpp with a different
placement, which is the only honest way to move them given the runtime lives
in another process. Both mechanisms are registered here as callables and
called back into. Nothing below touches a tensor.

The invariant the whole thing exists to protect
-----------------------------------------------
    Never unload merely because another workload started. Demote only because
    the incoming workload actually needs the memory.

Which is why :func:`request_vram` begins by asking whether the incoming
workload already fits. In Hybrid mode a small checkpoint and a small LLM
simply coexist, and alternating between them costs nothing at all, because the
answer to "does it fit" is yes and no code below the first branch runs.

The other invariant, which is newer and narrower
------------------------------------------------
    An image residency is never demoted for the LLM.

The image model is the workload. The language model writes a prompt for it, and
a helper that evicts what it is helping has made the job slower, not faster --
the checkpoint is wanted again seconds later, so every byte the LLM borrows is
paid for twice over, moving weights out and then back in. What the LLM gets is
the VRAM the image side is not using, and when that is not enough it makes
itself smaller: a shorter context, its experts in system RAM, fewer blocks on
the card, and finally the whole model in system RAM. It never asks the image
side for anything. See :func:`_victim_order`.

Serialization
-------------
Co-residency is not co-execution (section 8). Everything that actually uses
the GPU takes :func:`workload`, so an image pass and an LLM completion take
turns even while both models stay resident. That lock is also what makes the
"an active workload is never evicted" rule (sections 9 and 15) true by
construction rather than by vigilance: a residency cannot be demoted out from
under a running job, because the demotion cannot start until the job that owns
the lock has finished or been cancelled.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

FAMILY_IMAGE = "image"
FAMILY_LLM = "llm"

MODE_EXCLUSIVE = "exclusive"
MODE_HYBRID = "hybrid"

MODES = (
    (MODE_HYBRID, "Keep the LLM loaded — llama-server stays in the spare VRAM, "
                  "so the next prompt it writes starts warm"),
    (MODE_EXCLUSIVE, "Free the LLM for every image — stops llama-server when a generation "
                     "starts; the most headroom for one pass, and a model load per image"),
)
"""The two modes, named for the only thing that still differs between them.

They used to be "Hybrid" and "Exclusive" -- a question about who *owns* VRAM.
That question no longer has two answers: the image model always keeps its VRAM
and the LLM is placed in what is left over, whichever mode is chosen. What is
left is a question about what happens to a *warm* llama-server that is already
confined to the spare room when a generation starts, and that one is a real
trade: keeping it costs the pass nothing measurable and saves a model load,
freeing it hands the pass every last byte and costs one load per image.

The stored constants keep their old spellings so a config holding ``hybrid`` or
``exclusive`` still resolves. A config holding the old *label* does not, and
falls back to the default -- which is deliberate: "one family owns VRAM at a
time" was a choice about a question that is gone, and the nearest thing to it
is not the setting a user with that answer stored would want today.
"""

OPT_MODE = "model_chain_memory_mode"

# Residency ranks, lowest value evicted first (section 9). Recency breaks ties
# within a rank; rank itself is never overridden by recency, which is the whole
# reason it exists -- an actively executing model is the least recently *used*
# thing on the card at the moment it is asked for, and plain LRU throws it out.
RANK_STALE = 0
"""Inactive for long enough that nothing is expected to want it back soon."""
RANK_SPECULATIVE = 1
"""Warmed on a guess about the next job. First to go, always."""
RANK_HOT = 2
"""Used recently by a real request."""
RANK_NEXT = 3
"""Selected as the next workload's model, but not yet running."""
RANK_PINNED = 4
"""The user said keep this. Only an explicit override moves it."""
RANK_ACTIVE = 5
"""Executing right now. Never evicted, under any policy."""

RANK_LABELS = {
    RANK_STALE: "stale",
    RANK_SPECULATIVE: "speculative",
    RANK_HOT: "hot",
    RANK_NEXT: "selected next",
    RANK_PINNED: "pinned",
    RANK_ACTIVE: "active",
}

# --------------------------------------------------------------------------- #
# Resource domains (section 4)
# --------------------------------------------------------------------------- #
#
# The whole of this change set is one correction, and it is worth stating
# before any of the machinery below: a conflict is between two workloads that
# want the *same physical thing*. Two workloads are not in conflict because
# both are "GPU work", or "AI work", or "models". They are in conflict when
# they want the same processor to execute on, or the same pool of memory that
# is short of room.
#
# So there are two vocabularies here rather than one, because there are two
# questions:
#
#   ExecutionDomain -- "which processor is this request materially using?"
#   MemoryDomain    -- "if this residency goes away, which pool gains room?"
#
# They are deliberately not the same object. A Mixed Conservative llama-server
# executes on CUDA 1 and keeps its weights in system RAM: it conflicts with an
# image job on CUDA 1 for execution and with a warm image checkpoint for host
# RAM, and it is not a source of CUDA 1 VRAM for anybody. One field could not
# have said that.

EXEC_CPU = "cpu"
"""Executes on the processor. Positive evidence of no CUDA conflict."""
EXEC_CUDA = "cuda"
"""Executes on a *known* physical CUDA card."""
EXEC_CUDA_UNKNOWN = "cuda?"
"""Executes on CUDA, on a card that could not be resolved.

Distinct from :data:`EXEC_CPU` and that distinction is the point (invariant
I-10). Both used to be spelled "no card index", which meant one nullable
integer was being asked to carry two states needing opposite decisions: a
processor runtime must *not* wait for an image generation, and a CUDA runtime
whose card nobody can name must.
"""


@dataclass(frozen=True)
class ExecutionDomain:
    """Which processor a workload is materially executing on (section 4.1).

    Conflict is symmetric, and it is conservative in exactly one direction:
    an unresolved CUDA card conflicts with every CUDA domain, because the cost
    of being wrong is two jobs sharing a card for a while, whereas the cost of
    guessing "different card" wrongly is an image generation and a language
    model fighting over one processor with neither aware of the other.

    A CUDA workload's incidental host-thread activity is not modelled as a
    second domain. Forge uses CPU threads for every generation; if that made a
    processor-resident llama-server "conflicting", the CPU case this design
    exists to unblock would never run.
    """

    kind: str = EXEC_CUDA_UNKNOWN
    card: int | None = None

    @property
    def is_cpu(self) -> bool:
        return self.kind == EXEC_CPU

    @property
    def is_cuda(self) -> bool:
        return self.kind in (EXEC_CUDA, EXEC_CUDA_UNKNOWN)

    @property
    def known(self) -> bool:
        """Whether the physical processor is identified."""
        return self.kind == EXEC_CPU or (self.kind == EXEC_CUDA and self.card is not None)

    def conflicts_with(self, other: "ExecutionDomain | None") -> bool:
        """Whether two active workloads are competing for one processor."""
        if other is None:
            return False
        if self.is_cpu or other.is_cpu:
            # Two processor workloads really do share the processor; a
            # processor workload and a CUDA one do not, whatever else they
            # share (invariant I-1).
            return self.is_cpu and other.is_cpu
        if self.kind == EXEC_CUDA and other.kind == EXEC_CUDA:
            return self.card == other.card
        return True  # at least one unresolved CUDA card

    def describe(self) -> str:
        if self.is_cpu:
            return "the processor"
        if self.kind == EXEC_CUDA and self.card is not None:
            return f"GPU {self.card}"
        return "an unidentified GPU"


CPU_EXECUTION = ExecutionDomain(EXEC_CPU)
"""The processor. A module-level value because there is only ever one of it."""

UNKNOWN_CUDA_EXECUTION = ExecutionDomain(EXEC_CUDA_UNKNOWN)
"""CUDA, card unresolved. Conservative against every CUDA domain."""


def cuda_execution(card: int | None) -> ExecutionDomain:
    """The execution domain of a CUDA workload on ``card``.

    ``None`` or a negative index is :data:`UNKNOWN_CUDA_EXECUTION` and is
    emphatically *not* the processor -- that is the collapse invariant I-10
    forbids.
    """
    if card is None:
        return UNKNOWN_CUDA_EXECUTION
    try:
        index = int(card)
    except (TypeError, ValueError):
        return UNKNOWN_CUDA_EXECUTION
    return ExecutionDomain(EXEC_CUDA, index) if index >= 0 else UNKNOWN_CUDA_EXECUTION


def image_execution_domain() -> ExecutionDomain:
    """Which processor the image side executes on (section 4.1).

    Always CUDA: Forge is fixed to the card it was started on, and a card that
    cannot be resolved is unresolved rather than absent. Answering "processor"
    for an unreadable index would tell every CUDA language model it was
    independent of a generation it may well be sitting on top of.
    """
    return cuda_execution(image_device_index())


# --------------------------------------------------------------------------- #
# Memory domains (section 4.2)
# --------------------------------------------------------------------------- #

MEMORY_VRAM = "vram"
MEMORY_HOST_RAM = "ram"


class _AnyCard:
    """"Every card", which is not the same question as "the unknown card".

    A card filter has three states and one nullable integer can only carry
    two, so section 7.1 asks for them to be kept apart: ``card=ANY_CARD`` means
    *do not filter*, ``card=3`` means card three, and ``card=None`` means the
    residency whose card nobody could name. Overloading ``None`` for the first
    and the third is how a query for "LLM bytes on the image card" comes back
    with a 5090's nineteen gigabytes.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return "ANY_CARD"

    def __bool__(self) -> bool:
        return False


ANY_CARD = _AnyCard()
"""Sentinel for "no card filter requested". See :class:`_AnyCard`."""


STALE_AFTER_SECONDS = 15 * 60
"""How long a hot residency stays hot without being touched.

A number rather than a policy because the only thing it decides is eviction
*order*, never whether an eviction happens at all -- and something unused for
a quarter of an hour is a better victim than something unused for one minute
whatever the exact boundary is.
"""


def _named(family: str) -> str:
    """A family as it is written in a sentence rather than looked up in a dict.

    ``FAMILY_LLM`` is ``"llm"`` because it is a key, and "llm workload needed
    VRAM" is not a sentence anybody wrote on purpose.
    """
    return "LLM" if family == FAMILY_LLM else "image"


def option(name: str, default):
    """Read one of this extension's settings, falling back before the UI exists."""
    try:
        from modules import shared

        value = getattr(shared.opts, name, None)
    except Exception:
        return default
    return default if value in (None, "") else value


def resolve(raw, table, default: str) -> str:
    """A stored setting as one of ``table``'s values.

    Accepts either half of the pair. The Settings page shows a radio of the
    *labels*, because "Hybrid — image and LLM may share VRAM when they fit" is
    a setting somebody can choose without reading the manual and "hybrid" is
    not, and what a Gradio radio stores is whatever string it displayed. So the
    stored value is usually a label, sometimes a value -- from an older
    config, from a hand-edited one, or from a caller passing the constant --
    and both have to mean the same thing.
    """
    text = str(raw or "").strip()
    if not text:
        return default
    for value, label in table:
        if text == value or text == label:
            return value
    folded = text.casefold()
    for value, label in table:
        if folded in (value.casefold(), label.casefold()):
            return value
    # A label whose explanation has been reworded since it was stored. What a
    # radio saves is the whole string, so rewriting the half after the dash
    # would otherwise silently reset everybody's choice to the default -- and a
    # residency mode that changes itself because the sentence describing it was
    # improved is a far worse bug than the wording it fixed.
    name = _label_name(folded)
    if name:
        for value, label in table:
            if name == _label_name(label.casefold()):
                return value
    return default


def _label_name(text: str) -> str:
    """The naming half of a label -- everything before the explanatory dash."""
    for dash in ("—", "–", " - "):
        head, _, tail = text.partition(dash)
        if tail:
            return head.strip()
    return ""


def label_for(table, value: str) -> str:
    """``table``'s display text for ``value``, for putting in a control."""
    for candidate, label in table:
        if candidate == value:
            return label
    return value


def mode() -> str:
    """The configured residency mode. Hybrid is the default (section 8)."""
    return resolve(option(OPT_MODE, MODE_HYBRID), MODES, MODE_HYBRID)


# --------------------------------------------------------------------------- #
# The residency register
# --------------------------------------------------------------------------- #


@dataclass
class Residency:
    """One thing occupying VRAM, as the broker sees it.

    Deliberately thin. The broker does not want to know what a checkpoint or a
    GGUF *is* -- only how big it is, how much anybody would miss it, and who to
    call to make it go away.
    """

    family: str
    key: str
    label: str
    bytes: int = 0
    rank: int = RANK_HOT
    pinned: bool = False
    last_used: float = field(default_factory=time.monotonic)
    card: int | None = None
    """Which physical card gains room if this residency goes away (section 7.1).

    ``None`` is "not known", never "all of them" and never "the image card".
    A residency whose card is unknown is a residency nothing may be targeted
    at: invariant I-11 -- an unidentifiable memory domain never authorises
    destructive reclaim, because stopping the wrong runtime can free zero bytes
    where they were needed and destroy a warm prompt cache on the way.
    """
    domain: str = MEMORY_VRAM
    """Which pool this occupies. Everything declared here is VRAM today."""

    def on_card(self, card) -> bool:
        """Whether releasing this could add room on ``card``.

        ``ANY_CARD`` matches everything, an integer matches only that card,
        and an unknown card matches only an explicit request for the unknown.
        """
        if isinstance(card, _AnyCard):
            return True
        if card is None:
            return self.card is None
        return self.card is not None and int(self.card) == int(card)

    @property
    def effective_rank(self) -> int:
        """``rank``, with staleness and pinning applied.

        Pinning wins over staleness: a pinned model that has not been touched
        all session is still pinned. That is the point of pinning it.
        """
        if self.pinned and self.rank < RANK_PINNED:
            return RANK_PINNED
        if self.rank == RANK_HOT and time.monotonic() - self.last_used > STALE_AFTER_SECONDS:
            return RANK_STALE
        return self.rank

    @property
    def evictable(self) -> bool:
        return self.effective_rank < RANK_PINNED

    def touch(self) -> None:
        self.last_used = time.monotonic()


_register: dict[str, Residency] = {}
_register_lock = threading.RLock()


def declare(family: str, key: str, label: str, size_bytes: int,
            rank: int = RANK_HOT, pinned: bool = False,
            card: int | None = None) -> Residency:
    """Record that ``key`` now occupies VRAM, or update what is known about it.

    ``card`` is the physical GPU whose free memory grows if this goes away.
    Passing it is what makes a reclaim request for one card able to leave the
    other card's runtime alone (invariant I-3).
    """
    with _register_lock:
        entry = _register.get(key)
        if entry is None:
            entry = Residency(family=family, key=key, label=label, bytes=int(size_bytes),
                              rank=rank, pinned=pinned, card=_card_index(card))
            _register[key] = entry
        else:
            entry.family, entry.label = family, label
            entry.bytes, entry.rank, entry.pinned = int(size_bytes), rank, pinned
            entry.card = _card_index(card)
            entry.touch()
        return entry


def _card_index(card) -> int | None:
    """``card`` as a physical index, or None when it is not one."""
    if card is None or isinstance(card, _AnyCard):
        return None
    try:
        index = int(card)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def retire(key: str) -> None:
    """Record that ``key`` no longer occupies VRAM."""
    with _register_lock:
        _register.pop(key, None)


def retire_family(family: str) -> None:
    with _register_lock:
        for key in [k for k, e in _register.items() if e.family == family]:
            _register.pop(key, None)


def rank(key: str, value: int) -> None:
    with _register_lock:
        entry = _register.get(key)
        if entry is not None:
            entry.rank = value
            entry.touch()


def pin(key: str, pinned: bool = True) -> None:
    with _register_lock:
        entry = _register.get(key)
        if entry is not None:
            entry.pinned = bool(pinned)


def residencies(family: str | None = None, *, card=ANY_CARD) -> list[Residency]:
    """Everything ``family`` holds, optionally only on one physical card.

    ``card`` defaults to :data:`ANY_CARD` -- no filter -- which is why the
    sentinel exists rather than a ``None`` default. See :class:`_AnyCard`.
    """
    with _register_lock:
        return [e for e in _register.values()
                if (family is None or e.family == family) and e.on_card(card)]


def resident_bytes(family: str | None = None, *, card=ANY_CARD) -> int:
    return sum(e.bytes for e in residencies(family, card=card))


def clear() -> None:
    """Forget everything. For tests and for ``on_script_unloaded``."""
    with _register_lock:
        _register.clear()
    with _decision_lock:
        _decisions.clear()
    _said_independence.clear()


# --------------------------------------------------------------------------- #
# Mechanism registration
# --------------------------------------------------------------------------- #
#
# The broker decides; these do. Both are optional: an installation that never
# opens LLM Studio registers no LLM controller and the LLM half of every
# decision below is skipped, which is section 18's regression requirement --
# ordinary txt2img must be unaffected by a feature that was never used.

_reclaimers: dict[str, object] = {}


def register_reclaimer(family: str, reclaimer) -> None:
    """Register who frees VRAM for ``family``.

    ``reclaimer`` needs one method::

        release(needed_bytes: int, reason: str) -> int   # bytes actually freed

    and may offer ``resident_bytes() -> int`` and ``describe() -> str``.
    """
    _reclaimers[family] = reclaimer


def unregister_reclaimer(family: str) -> None:
    _reclaimers.pop(family, None)


def _reclaimer(family: str):
    return _reclaimers.get(family)


# --------------------------------------------------------------------------- #
# Workload serialization (section 15)
# --------------------------------------------------------------------------- #


class Busy(RuntimeError):
    """Raised when a workload could not take the GPU within its timeout."""


@dataclass(frozen=True)
class Active:
    family: str
    label: str
    since: float
    domain: ExecutionDomain = UNKNOWN_CUDA_EXECUTION
    """Which processor this workload is materially using (section 12.4).

    Defaulted rather than required so that every existing caller keeps working,
    and defaulted to *unknown CUDA* rather than to anything convenient: a
    workload that did not say where it executes is one nothing may be told it
    is independent of.
    """

    def describe(self) -> str:
        return f"{self.label} on {self.domain.describe()}"


class _JobLock:
    """The workload lock: reentrant, and owned by a *job* rather than a thread.

    ``threading.RLock`` was the obvious thing and was the wrong thing, for one
    reason: a workload here is held across a generator, and a generator is not
    guaranteed to be finished on the thread that started it. Gradio hands a
    handler's generator to a worker thread, and an abandoned one -- a cancelled
    run, a closed tab, a run whose frame ended up in a reference cycle because
    it raised -- is finalized by the garbage collector, on whichever thread
    happened to trigger the collection. ``RLock.release`` from that thread
    raises ``RuntimeError: cannot release un-acquired lock``, the ``finally``
    that was giving the GPU back does not, and the lock stays held for the rest
    of the session. Every later run on another thread then waits for a job that
    finished minutes ago, which is exactly the "Waiting for the GPU…" that
    never ends.

    So ownership is recorded for reentrancy only. ``acquire`` still lets the
    owning thread nest freely -- a chained generation is one workload with two
    stages (section 15) -- while ``release`` accepts the handle back from
    anywhere, because :class:`workload` releases exactly once per acquisition
    and it is that pairing, not the thread it happens on, that makes the count
    honest.
    """

    def __init__(self):
        self._condition = threading.Condition(threading.Lock())
        self._owner: int | None = None
        self._depth = 0

    def acquire(self, timeout: float = -1) -> bool:
        me = threading.get_ident()
        deadline = None if timeout is None or timeout < 0 else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._depth == 0 or self._owner == me:
                    self._owner = me
                    self._depth += 1
                    return True
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                self._condition.wait(remaining)

    def release(self) -> bool:
        """Give one acquisition back. False if there was nothing to give back."""
        with self._condition:
            if self._depth == 0:
                return False
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                self._condition.notify_all()
            return True


_gpu = _JobLock()
_active: list[Active] = []
_active_lock = threading.Lock()
_background_wait = threading.Event()
_background_wait.set()
"""Cleared while a foreground workload wants the GPU.

Background work -- speculative warming, preloading -- waits on this rather than
queueing on the lock itself, so a foreground request never sits behind a guess
about the future (section 15). It is an event and not a counter because the
only question a background job has is "should I be running at all", and the
answer while anything foreground is pending is no.
"""

_foreground_waiting = 0


class _Held:
    """The handle :func:`workload` yields. Truthy when the GPU was acquired."""

    def __init__(self, family: str, label: str, acquired: bool):
        self.family, self.label, self.acquired = family, label, acquired

    def __bool__(self) -> bool:
        return self.acquired


def active() -> Active | None:
    """The workload currently holding the GPU, if any."""
    with _active_lock:
        return _active[-1] if _active else None


def active_workloads() -> tuple[Active, ...]:
    """Every workload holding the lock right now, outermost first.

    A list rather than one value because section 14.1 requires the UI to be
    able to say "image generating on GPU 0" *and* "conversation generating on
    GPU 1" at the same time. It does not mean the lock has stopped serialising
    LLM turns -- it has not -- only that "what is running" is no longer a
    question with at most one answer.
    """
    with _active_lock:
        return tuple(_active)


def active_family() -> str | None:
    running = active()
    return running.family if running is not None else None


def image_activity() -> Active | None:
    """The host's own image job as a workload, when one is running.

    Derived rather than registered: Forge starts its generations itself and
    takes no lock of ours (see the note above :func:`host_busy`), so the only
    honest source for "is an image job running" is ``shared.state``. Presented
    in the same shape as an LLM workload so that :func:`activities` can hand
    the panel one list.
    """
    if not host_busy():
        return None
    return Active(FAMILY_IMAGE, "Image generation", 0.0, image_execution_domain())


def activities() -> tuple[Active, ...]:
    """Everything active right now, image job included (section 14.1).

    The one function a status display should ask. ``active()`` answers "which
    workload holds the broker's lock", which was the same question only for as
    long as an image generation and an LLM turn could not both be running.
    """
    found = [entry for entry in active_workloads() if entry.family != FAMILY_IMAGE]
    image = image_activity()
    return (image, *found) if image is not None else tuple(found)


class workload:
    """Context manager taking the shared workload lock for one job.

    Nested acquisition by the same thread is free: a Model Chain generation is
    one workload with two stages, not two workloads, and Stage 2 must not have
    to reason about a lock Stage 1 is already holding.

    ``background=True`` marks work nobody is waiting for. It never queues ahead
    of a foreground request and gives up rather than blocking one.
    """

    def __init__(self, family: str, label: str = "", *, timeout: float | None = None,
                 background: bool = False, required: bool = True,
                 domain: ExecutionDomain | None = None):
        self.family, self.label = family, label or family
        self.timeout, self.background, self.required = timeout, background, required
        self.domain = domain or UNKNOWN_CUDA_EXECUTION
        """Where this workload executes, for the *other* family's conflict check.

        The lock this class takes is still global across LLM turns (section
        6.5, non-goal N2). What the domain changes is who has to wait for the
        holder from outside that lock -- an image generation on GPU 0 does not
        wait for a completion on GPU 1, and did, for as long as "an LLM is
        active" was the whole of the question.
        """
        self._held: _Held | None = None
        self._counted = False
        self._entry: Active | None = None

    def __enter__(self) -> _Held:
        global _foreground_waiting

        if self.background and not _background_wait.is_set():
            self._held = _Held(self.family, self.label, False)
            return self._held

        if not self.background:
            with _active_lock:
                _foreground_waiting += 1
                _background_wait.clear()
            self._counted = True

        timeout = -1 if self.timeout is None else self.timeout
        acquired = _gpu.acquire(timeout=timeout)
        if not acquired:
            self._release_count()
            if self.required:
                running = active()
                raise Busy(
                    f"{running.label} is using the GPU" if running else "The GPU is busy"
                )
            self._held = _Held(self.family, self.label, False)
            return self._held

        self._entry = Active(self.family, self.label, time.monotonic(), self.domain)
        with _active_lock:
            _active.append(self._entry)
        self._held = _Held(self.family, self.label, True)
        return self._held

    def __exit__(self, *exc) -> bool:
        """Give the GPU back, once, from wherever the job happens to end.

        Idempotent because the exit can be reached twice -- a caller that
        releases explicitly and then unwinds, a generator closed after its
        ``finally`` already ran -- and a second release would hand the card to
        somebody while the job that took it is still on it. The entry is
        removed by identity rather than popped, because a release that arrives
        late (a generator finalized after a later workload started) would
        otherwise take the running job's name off the list and leave its own.
        """
        with _active_lock:
            held, self._held = self._held, None
            entry, self._entry = self._entry, None
            if held is None or not held.acquired:
                held = None
            else:
                for position, running in enumerate(_active):
                    if running is entry:
                        del _active[position]
                        break
        if held is not None:
            _gpu.release()
        self._release_count()
        return False

    def _release_count(self) -> None:
        global _foreground_waiting

        with _active_lock:
            if not self._counted:
                return
            self._counted = False
            _foreground_waiting = max(_foreground_waiting - 1, 0)
            if _foreground_waiting == 0:
                _background_wait.set()


def foreground_pending() -> bool:
    """Whether a foreground workload is waiting for or holding the GPU."""
    return not _background_wait.is_set()


# --------------------------------------------------------------------------- #
# Mutual exclusion with the host's own generations
# --------------------------------------------------------------------------- #
#
# The workload lock above serialises everything *this extension* runs. An
# ordinary txt2img or img2img generation is not one of those things: the host
# starts it, and there is no pairing of hooks an extension can hold a lock
# across safely. `postprocess` is not called from a `finally`, so a generation
# that raises would leave the lock held and every later LLM request blocked
# until the WebUI restarted. That failure mode is far worse than the race it
# would be preventing.
#
# What the host does maintain, in its own try/finally, is `shared.state`. It is
# begun before a job and ended after one however the job turned out, so reading
# it cannot leak. So exclusion is arranged from the two sides differently:
#
#   * the LLM refuses to *start* a turn while the host has a job running, and
#     re-checks after taking the lock;
#   * an image generation *waits*, once, at the top of before_process, for any
#     LLM turn already in flight -- and then proceeds without holding anything.
#
# The residual window is the few microseconds between an LLM's last check and
# its first token, during which a generation can begin. What happens then is
# that the two overlap for one completion: both are on the card, both are
# slower, and neither is corrupted, because co-residency was always allowed and
# it is only the *timing* that was meant to be tidy. That is a materially
# better trade than a lock that can strand the feature.


def host_busy() -> bool:
    """Whether the WebUI is running a job of its own right now."""
    try:
        from modules import shared

        state = shared.state
    except Exception:
        return False
    if getattr(state, "job", ""):
        return True
    return bool(getattr(state, "job_count", 0))


# --------------------------------------------------------------------------- #
# The one caller that must not wait for the host
# --------------------------------------------------------------------------- #
#
# `host_busy()` above is the rule that keeps an LLM turn from starting on top of
# an image generation. It has exactly one legitimate exception, and it is not a
# loosening of the rule so much as the rule read carefully: the wait exists so
# that an LLM turn does not *compete* with a running image job, and a turn the
# image job is itself blocked waiting for is not competing with it. It is the
# job, in the part of the job that happens before any sampling.
#
# That is what Krea Creative Mode does. The generation cannot proceed until the
# prompt has been written, so `before_process` writes it -- and if that request
# waited for `host_busy()` to go false, it would be waiting for the generation
# that is waiting for it. Forever.
#
# The permission is scoped, thread-local and declared rather than inferred:
# nothing here tries to work out whether the calling thread "looks like" a job
# thread. A caller that knows it holds up the host job says so, for exactly as
# long as that is true, and `mc_llm_sessions._Gpu.acquire` is the only reader.
#
# What is deliberately *not* relaxed: the workload lock itself. Two LLM turns
# still serialise, so an in-job roll queues behind a turn LLM Studio started and
# that wait terminates, because the turn ahead of it is running rather than
# waiting on anything this generation holds. The image side takes no broker lock
# at all (see above), so there is no cycle left to close.


_host_job = threading.local()


class host_job:
    """Declare that this thread is the host's own job, for as long as it is.

    Re-entrant and nestable, so a caller does not have to know whether it is
    already inside one. Everything about it is scoped to the thread that entered
    it: a worker thread started from inside the block is *not* the host job, and
    should not be, because nothing is blocked waiting for it.
    """

    def __enter__(self) -> "host_job":
        _host_job.depth = getattr(_host_job, "depth", 0) + 1
        return self

    def __exit__(self, *exc) -> bool:
        _host_job.depth = max(getattr(_host_job, "depth", 0) - 1, 0)
        return False


def inside_host_job() -> bool:
    """Whether this thread is running as part of the host's own image job.

    True only inside :class:`host_job`. The one thing it licenses is skipping
    the ``host_busy()`` wait -- it grants no lock, no VRAM and no priority.
    """
    return getattr(_host_job, "depth", 0) > 0


_said_independence: set[tuple[str, str]] = set()


def _say_independence(domain: ExecutionDomain) -> None:
    """Say once, per pair of processors, that two workloads are not competing.

    Section 21.3 asks for positive non-interference to be observable, and asks
    for it not to be printed per token or per chunk. So it is said at the one
    boundary where it is decided -- an image generation about to start beside a
    language model already running -- and said once for each pair of
    processors, because the interesting fact is "these two are independent",
    which does not become more true the twentieth time.
    """
    others = {entry.domain for entry in active_workloads() if entry.family == FAMILY_LLM}
    for other in others:
        pair = (domain.describe(), other.describe())
        if pair in _said_independence:
            continue
        _said_independence.add(pair)
        logger.info("Model Chain: image generation is on %s and the LLM is on %s — they use "
                    "different processors and run at the same time", *pair)


def conflicting_llm(domain: ExecutionDomain | None) -> Active | None:
    """An active LLM workload competing with ``domain`` for a processor.

    The predicate the image side's wait is built on, and the whole of what
    section 12 narrows. It used to be "is an LLM running"; it is now "is an LLM
    running *on the thing I am about to use*", which on a two-card machine is a
    different question with a different answer several times a minute.

    ``None`` keeps the old meaning -- any LLM at all conflicts -- so a caller
    that cannot say where it executes is no worse off than before.
    """
    for entry in active_workloads():
        if entry.family != FAMILY_LLM:
            continue
        if domain is None or domain.conflicts_with(entry.domain):
            return entry
    return None


def await_idle(timeout: float = 120.0, *, domain: ExecutionDomain | None = None) -> bool:
    """Block until no *conflicting* LLM workload holds a processor this one needs.

    Called from the image side before a generation, with the image card's
    execution domain. Returns immediately -- and this is the point of the
    parameter -- for an LLM executing on another known card or on the
    processor, because neither of those is competing with a generation on this
    one (invariant I-1). It still waits for a same-card LLM and for one whose
    card could not be resolved (section 12.3).

    Bounded because an image generation must never be blocked indefinitely by
    this extension: on timeout it returns False, the caller proceeds, and the
    two overlap -- which is slow, and is still better than a generation that
    never starts.
    """
    deadline = time.monotonic() + max(float(timeout), 0.0)
    first = True
    while True:
        running = conflicting_llm(domain)
        if running is None:
            if not first:
                logger.info("Model Chain: the LLM on %s has finished; generating",
                            domain.describe() if domain is not None else "the card")
            elif domain is not None:
                _say_independence(domain)
            return True
        if time.monotonic() >= deadline:
            note(FAMILY_IMAGE,
                 f"an LLM request on {running.domain.describe()} was still running after the "
                 f"wait expired; the generation starts anyway and the two will share the "
                 f"card briefly")
            return False
        if first:
            first = False
            logger.info("Model Chain: waiting for the LLM on %s to finish before generating "
                        "on %s", running.domain.describe(),
                        domain.describe() if domain is not None else "the image card")
        time.sleep(0.05)


# --------------------------------------------------------------------------- #
# Decisions, for the status panel (section 14)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Decision:
    when: float
    family: str
    text: str


_decisions: list[Decision] = []
_decision_lock = threading.Lock()
_DECISION_LIMIT = 40


def note(family: str, text: str) -> None:
    """Record why something moved, so the panel can answer "why" (section 14)."""
    with _decision_lock:
        _decisions.append(Decision(time.time(), family, text))
        del _decisions[:-_DECISION_LIMIT]
    logger.info("Model Chain: %s", text)


def decisions(limit: int = 10) -> list[Decision]:
    with _decision_lock:
        return list(_decisions[-limit:])


# --------------------------------------------------------------------------- #
# Fit and reclaim (sections 8, 10, 11)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Reclaim:
    """What :func:`request_vram` did, and whether it was enough."""

    needed: int
    free_before: int
    freed: int
    deficit: int
    actions: tuple[str, ...] = ()

    @property
    def satisfied(self) -> bool:
        return self.deficit <= 0

    @property
    def moved_anything(self) -> bool:
        return bool(self.actions)

    def describe(self) -> str:
        if not self.actions:
            return "no residency had to move"
        return "; ".join(self.actions)


def free_vram_bytes() -> int:
    """Device-free VRAM, which already accounts for other processes.

    Worth stating because it is what makes co-residency with llama.cpp
    measurable at all: llama-server's allocations belong to another process and
    would be invisible to a torch-only figure, but ``mem_get_info`` is a driver
    query about the whole card, and Forge's helper is built on it.
    """
    try:
        import mc_memory

        return int(mc_memory.free_vram_bytes())
    except Exception:
        return 0


def image_device_index() -> int:
    """Which card the image side is on, or -1. See :func:`mc_memory.image_device_index`."""
    try:
        import mc_memory

        return int(mc_memory.image_device_index())
    except Exception:
        return -1


def device_free_vram_bytes(index: int | None = None) -> int:
    """Free VRAM as *another process* would find it (section 6's boundary).

    :func:`free_vram_bytes` is the host's own accounting and includes what its
    allocator is holding cached, which the host may reuse and llama-server may
    not. Every figure the LLM side places against is this one; every figure the
    image side places against is the other. Two numbers because there are two
    questions, and answering the second with the first is how a card with
    twenty-two gigabytes free refuses an allocation of ten.
    """
    try:
        import mc_memory

        return int(mc_memory.device_free_vram_bytes(index))
    except Exception:
        return free_vram_bytes()


def cuda_device_count() -> int:
    """How many CUDA cards this machine has, or 0 when the question cannot be put.

    Asked in exactly one place -- :func:`_reclaim_scope` -- and only to tell two
    kinds of "the image card is unknown" apart. On a single-card machine an
    unreadable index changes nothing: everything is on the one card, so an
    unfiltered reclaim is a card-local reclaim by construction. On a machine
    with two, the same unreadable index means residency cannot be attributed at
    all, and stopping a runtime on the strength of that is the guess section
    16.1 forbids.
    """
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def release_cached_vram() -> int:
    """Give the image allocator's cached blocks back to the driver, for the LLM."""
    try:
        import mc_memory

        return int(mc_memory.release_cached_vram())
    except Exception:
        logger.debug("Model Chain: could not release cached VRAM", exc_info=True)
        return 0


def total_vram_bytes() -> int:
    try:
        import mc_memory

        return int(mc_memory.total_vram_bytes())
    except Exception:
        return 0


def safety_margin_bytes() -> int:
    """Global spare VRAM no fit calculation may spend (section 11).

    Reuses the image side's own reserve arithmetic rather than inventing a
    second number, so a user who has already told ModelSwitch how much to keep
    free does not have to tell LLM Studio separately.
    """
    try:
        import mc_memory

        return int(mc_memory.vram_headroom_bytes())
    except Exception:
        return int(0.5 * _GB)


def fits(needed_bytes: int, *, margin: int | None = None) -> bool:
    """Whether ``needed_bytes`` is available right now without moving anything."""
    reserve = safety_margin_bytes() if margin is None else margin
    return free_vram_bytes() >= int(needed_bytes) + int(reserve)


# --------------------------------------------------------------------------- #
# Host RAM (section 10)
# --------------------------------------------------------------------------- #
#
# System RAM is in this module for one reason, and it is not tidiness: this
# repository already uses it as a residency tier. ``mc_memory`` keeps whole
# checkpoints, their text encoders and their VAEs warm there so a Stage 1 /
# Stage 2 switch is a pointer swap rather than a disk load, and llama.cpp puts
# its weights there for every CPU and Mixed Conservative placement. Those are
# not two budgets. They are one physical pool, and planning them separately
# fails in exactly the way mixing a 3090's free VRAM with a 5090's residency
# fails -- arithmetic about different owners of one thing.
#
# What is emphatically *not* here: a second image cache, a page-cache manager,
# or any notion that the broker can pin or account for mmap'd GGUF pages.
# ``mc_memory`` owns the image cache and is asked; the operating system owns
# file-backed warmth and is left alone (invariant I-8, non-goal N5). All this
# module adds is the admission boundary where the two families meet.


def free_ram_bytes() -> int:
    """Physical memory the OS can hand out right now, or 0 if it cannot be asked.

    Deliberately the operating system's *available* figure rather than a sum of
    process working sets (section 10.5). A working set is a momentary thing
    full of shared and file-backed pages the OS will trim under pressure;
    available memory is the answer to the question admission actually asks,
    which is "how much can this machine give to new demand immediately".
    """
    try:
        import mc_memory

        return max(int(mc_memory.free_ram_bytes()), 0)
    except Exception:
        logger.debug("Model Chain: could not read available system RAM", exc_info=True)
        return 0


def total_ram_bytes() -> int:
    try:
        import mc_memory

        return max(int(mc_memory.total_ram_bytes()), 0)
    except Exception:
        return 0


def ram_reserve_bytes() -> int:
    """Host memory nothing here may intentionally consume (section 10.6).

    ``mc_memory``'s own floor, reused rather than re-invented, so that the
    image cache and an incoming language model are protecting the same number.
    Two reserves would mean the second workload could spend the first's.
    """
    try:
        import mc_memory

        return max(int(mc_memory.RAM_RESERVE_BYTES), 0)
    except Exception:
        return 2 * _GB


def image_warm_ram_bytes() -> int:
    """Host RAM the image cache is explicitly holding, or 0."""
    try:
        import mc_memory

        return max(int(mc_memory.warm_ram_bytes()), 0)
    except Exception:
        return 0


def reclaimable_image_ram_bytes() -> int:
    """Of that, what ``mc_memory`` says is safe to drop right now.

    Not the same number, and the difference is invariant I-10 of section 10:
    a checkpoint the current pass is executing against is host memory the image
    job *needs*, and is not a victim for anybody's admission arithmetic. Only
    ``mc_memory`` knows which entries are which, so only ``mc_memory`` is asked.
    """
    try:
        import mc_memory

        return max(int(mc_memory.reclaimable_warm_ram_bytes()), 0)
    except Exception:
        return 0


def release_image_warm_ram(needed_bytes: int, reason: str = "") -> int:
    """Ask ``mc_memory`` to give back ``needed_bytes`` of warm cache.

    A request, through ``mc_memory``'s own cache operations. Nothing here
    touches a model object, knows what a patcher is, or decides which
    checkpoint is least useful -- invariant I-7, and the same division of
    labour the VRAM half of this module has always kept.
    """
    try:
        import mc_memory

        return max(int(mc_memory.release_warm_ram(int(needed_bytes), reason=reason)), 0)
    except Exception:
        logger.debug("Model Chain: could not release warm image RAM", exc_info=True)
        return 0


def llm_host_ram_bytes() -> int:
    """Host RAM the running language models materially need, or 0.

    Asked of the LLM family's reclaimer, which is the only thing that knows how
    each of its servers was placed. A full-GPU llama-server answers zero here
    even though its mapped GGUF may be warm in the page cache -- that warmth is
    the operating system's and is reclaimable by it (invariant I-8), and
    counting the file's whole size as a hard reservation would block image
    generations that are perfectly safe (section 10.7, test T41).
    """
    reclaimer = _reclaimer(FAMILY_LLM)
    if reclaimer is None:
        return 0
    asking = getattr(reclaimer, "host_ram_bytes", None)
    if not callable(asking):
        return 0
    try:
        return max(int(asking() or 0), 0)
    except Exception:
        logger.debug("Model Chain: could not read the LLM's host-RAM demand", exc_info=True)
        return 0


@dataclass(frozen=True)
class Admission:
    """Whether a new host-RAM demand can be met, and what it cost to say yes."""

    needed: int
    reserve: int
    available_before: int
    available: int
    freed: int = 0
    actions: tuple[str, ...] = ()
    known: bool = True
    """False when available RAM could not be read at all (section 16.3)."""

    @property
    def fits(self) -> bool:
        """Whether the demand can be admitted above the safety floor.

        True when the question could not be asked, and that is not optimism --
        it is the existing behaviour preserved. Refusing to start a language
        model because ``psutil`` is missing would break installations that work
        today, so an unanswerable question warns and proceeds; what it must not
        do is *reclaim* on the strength of numbers nobody has.
        """
        if not self.known:
            return True
        return self.available >= self.needed + self.reserve

    @property
    def moved_anything(self) -> bool:
        return bool(self.actions)

    @property
    def shortfall(self) -> int:
        if not self.known:
            return 0
        return max(self.needed + self.reserve - self.available, 0)

    def describe(self) -> str:
        if not self.known:
            return "available system RAM could not be read"
        head = (f"{self.needed / _GB:.1f} GB wanted, {self.available / _GB:.1f} GB available, "
                f"{self.reserve / _GB:.1f} GB reserved")
        if self.actions:
            return f"{head}; {'; '.join(self.actions)}"
        return head


def host_ram_fits(needed_bytes: int, *, reserve: int | None = None) -> bool:
    """Whether ``needed_bytes`` fits above the host reserve without moving anything."""
    available = free_ram_bytes()
    if available <= 0:
        return True
    floor = ram_reserve_bytes() if reserve is None else max(int(reserve), 0)
    return available >= max(int(needed_bytes), 0) + floor


def admit_host_ram(needed_bytes: int, *, reason: str = "",
                   reserve: int | None = None) -> Admission:
    """Make room in system RAM for ``needed_bytes``, moving as little as possible.

    Section 10.8's order, and the first step is the one that matters most:
    **if it already fits, nothing moves.** Sharing a memory domain is not a
    conflict (invariant I-5). Two processor-resident language models and a warm
    Stage 2 checkpoint that all fit above the reserve should all stay exactly
    where they are; a scheduler that emptied the cache merely because somebody
    else also uses RAM would have turned a capacity-aware design back into
    evict-on-switch.

    When it does not fit, the *lowest-priority* residency yields first, and
    that is explicitly managed warm image cache -- state whose only purpose is
    to make a future switch faster (section 11.3). What comes back is measured
    afterwards rather than assumed: invariant I-15, because the number that
    decides whether a llama-server may start is what the OS says is available,
    not the sum of what a cache believed it was holding.

    Stopping another language model is *not* done here. It is a stronger action
    than dropping a cache entry, it depends on the user's role-sharing setting,
    and the runtime layer owns both of those; this returns a shortfall and lets
    it decide (section 10.8 step 7).
    """
    needed = max(int(needed_bytes), 0)
    floor = ram_reserve_bytes() if reserve is None else max(int(reserve), 0)
    before = free_ram_bytes()
    if before <= 0:
        logger.warning(
            "Model Chain: available system RAM could not be read, so safe coexistence "
            "could not be established for %s. Nothing was released — reclaiming on "
            "invented numbers is worse than not knowing.", reason or "this workload")
        return Admission(needed, floor, 0, 0, known=False)

    if before >= needed + floor:
        return Admission(needed, floor, before, before)

    deficit = needed + floor - before
    reclaimable = reclaimable_image_ram_bytes()
    actions: list[str] = []
    if reclaimable <= 0:
        note(FAMILY_LLM,
             f"host RAM is {deficit / _GB:.1f} GB short for {reason or 'a new runtime'} "
             f"({before / _GB:.1f} GB available, {floor / _GB:.1f} GB reserved) and no image "
             f"warm cache is safe to drop")
        return Admission(needed, floor, before, before)

    freed = release_image_warm_ram(deficit, reason or "a RAM-backed language model")
    # The measurement wins over the arithmetic, always. A cache that reports
    # fourteen gigabytes released has said what it stopped referencing, not
    # what the operating system has actually made available again.
    after = free_ram_bytes()
    if after <= 0:
        after = before
    if freed > 0:
        actions.append(f"released {freed / _GB:.1f} GB of inactive image RAM cache")
        note(FAMILY_IMAGE,
             f"released {freed / _GB:.1f} GB of inactive image RAM cache for "
             f"{reason or 'a RAM-backed language model'}; host RAM is now "
             f"{after / _GB:.1f} GB available")
    return Admission(needed, floor, before, after, freed, tuple(actions))


def _reason_for(family: str) -> str:
    """What to call a request that did not say what it was for.

    Every message built from a ``reason`` reads it as a noun phrase -- "X is
    short 2 GB", "freed 2 GB for X" -- so the fallback has to be one too.
    """
    return f"the {_named(family)} workload"


def _reclaim_scope(family: str, card) -> tuple[object, bool]:
    """Which card a VRAM request is about, and whether reclaim may act on it.

    Returns ``(filter, allowed)``. The filter is what :func:`residencies` and
    the family's reclaimer are asked to restrict themselves to; ``allowed`` is
    whether cross-family reclaim may happen at all.

    Three cases, and the third is the one worth the function:

    * A card was named. That card, and reclaim proceeds -- invariant I-3, and
      the reclaimer is told the number so residency on any other card is
      absent from the victim set rather than merely ranked below it.
    * No card was named and the image side's card can be read. The image card,
      because a request from the image family is by definition about the card
      Forge generates on.
    * No card can be read at all. Then it depends on how many cards there are.
      One card means the unfiltered answer *is* the card-local answer. More
      than one means residency cannot be attributed, and section 16.1 is
      explicit that a guessed GPU is not a reclaim target -- so nothing is
      taken and the shortfall is reported honestly instead.
    """
    named = _card_index(card)
    if named is not None:
        return named, True
    if not isinstance(card, _AnyCard):
        # An explicit ``None``: the caller knows it cannot name its card.
        return ANY_CARD, cuda_device_count() <= 1
    index = image_device_index() if family == FAMILY_IMAGE else -1
    if index >= 0:
        return index, True
    return ANY_CARD, cuda_device_count() <= 1


def request_vram(family: str, needed_bytes: int, *, reason: str = "",
                 margin: int | None = None, exclusive_sweep: bool = True,
                 card=ANY_CARD) -> Reclaim:
    """Make room for ``needed_bytes`` of ``family`` work, moving as little as possible.

    The order of the branches below is the policy, so it is worth reading as
    prose:

    1. In Exclusive mode an *image* workload takes the whole card first,
       whether or not the arithmetic would have demanded it, because Exclusive
       mode is a promise about ownership rather than an optimisation
       (section 10). The sweep runs in that direction and no other: an LLM
       never takes VRAM from the image family in either mode, so there is
       nothing for it to sweep. See :func:`_victim_order`.
    2. Otherwise, if the incoming workload already fits, nothing moves at all.
       This is the co-residency case and it is what Hybrid mode is for
       (section 8).
    3. Only then is the deficit -- and only the deficit -- freed, from the
       family that gives ground for this one (section 9).

    ``card`` is the physical GPU the request is about, and it runs through
    every one of those branches. It is what makes the sweep in (1) a sweep of
    Forge's card rather than of every GPU in the machine, the reading in (2) a
    reading of that card, and the release in (3) unable to reach residency
    that could not add a byte to it -- invariants I-2 and I-3. Left
    unspecified, an image request resolves it to the image card and everything
    else falls back to the machine-wide behaviour this function has always had.
    """
    needed = max(int(needed_bytes), 0)
    reserve = safety_margin_bytes() if margin is None else int(margin)
    target = needed + reserve
    scope, may_reclaim = _reclaim_scope(family, card)
    where = scope if isinstance(scope, int) else None
    # Which "free" this is depends on who is asking. An image pass can spend
    # the host allocator's cache; llama.cpp is another process and cannot, so
    # asking on its behalf against the host's figure frees nothing and reports
    # a shortfall that is not there -- or, worse, no shortfall when there is.
    # Either way it is *this card's* figure: mixing a reading from one card
    # with residency bytes from another is invariant I-2's whole subject.
    if family == FAMILY_LLM:
        def reading():
            return device_free_vram_bytes(where)
    elif where is not None and where != image_device_index():
        def reading():
            return device_free_vram_bytes(where)
    else:
        reading = free_vram_bytes
    free = reading()
    before = free
    actions: list[str] = []
    freed = 0
    # Read once. A setting that changed between the two branches below would
    # otherwise be able to produce a call that swept a family *and* went on to
    # demote a second one for the same request.
    #
    # Only an image workload sweeps. Exclusive mode used to be symmetrical, and
    # the symmetry was the bug: a Krea roll swept the checkpoint off the card,
    # reserved room for it again, placed llama-server in what was left, and
    # then the generation two seconds later spent thirteen seconds moving the
    # very same weights back -- every press, on a card that had been holding
    # both of them comfortably. See :func:`_victim_order`.
    sweeping = (exclusive_sweep and family == FAMILY_IMAGE and mode() == MODE_EXCLUSIVE
                and may_reclaim)
    swept = False

    def say_which_setting_did_this() -> None:
        """Name the setting, once, after the sweep has been reported.

        A model load per image is a real cost and it is easy to pay without
        knowing why: the LLM is confined to spare VRAM either way, so the pass
        gains almost nothing here and the next prompt starts cold. Said only
        when a server was really stopped, so a machine that never runs one, or
        one whose server was already down, hears nothing.
        """
        if not swept:
            return
        note(FAMILY_LLM,
             "llama-server was stopped because VRAM residency is set to free the LLM for "
             "every image. It was holding only VRAM the checkpoint is not using, so "
             "“Keep the LLM loaded” would have left it warm and the next prompt would "
             "have started without a model load")

    # Exclusive mode is checked before the fit, and that ordering is the whole
    # difference between the two modes. Hybrid asks "does this fit"; Exclusive
    # asks "does the image family own the card", and the answer has to be the
    # same whether or not the generation happened to fit alongside the LLM --
    # otherwise ownership would depend on the size of the last request, which
    # is precisely the unpredictability Exclusive mode exists to remove
    # (sections 10 and 18).
    if sweeping:
        # Scoped to the image card, and only to it. "The image family owns the
        # card" is a promise about Forge's physical card; it carries no
        # authority whatever over a second GPU (section 9.2), and a sweep that
        # crossed to one would stop a language model to free memory the
        # generation cannot use.
        released = _release(FAMILY_LLM, target,
                            reason or f"the {_named(family)} workload taking VRAM ownership",
                            sweep=True, card=scope)
        freed += released.freed
        actions.extend(released.actions)
        free = reading()
        swept = released.moved_anything

    if free <= 0:
        # VRAM could not be queried. Guessing at a deficit here would evict on
        # no evidence, which is exactly what this module exists not to do.
        say_which_setting_did_this()
        return Reclaim(needed, free, freed, 0, tuple(actions))

    if free >= target:
        result = Reclaim(needed, free, freed, 0, tuple(actions))
        if actions:
            note(family, f"{reason or _reason_for(family)}: {result.describe()}")
        say_which_setting_did_this()
        return result

    deficit = target - free

    if not sweeping and may_reclaim:
        for victim_family in _victim_order(family):
            if freed >= deficit:
                break
            released = _release(victim_family, deficit - freed,
                                reason or f"the {_named(family)} workload", card=scope)
            freed += released.freed
            actions.extend(released.actions)
    elif not may_reclaim:
        note(family,
             f"{reason or _reason_for(family)} is short {deficit / _GB:.1f} GB, and which "
             f"card it is short on could not be established on a machine with more than one "
             f"— nothing was released, because a runtime chosen by guesswork can free every "
             f"byte it holds and still leave this shortfall exactly where it was")

    after = reading()
    remaining = max(target - max(after, free + freed), 0)
    result = Reclaim(needed, free, freed, remaining, tuple(actions))

    if actions:
        # Reported from the reading taken *before* anything moved. The local
        # ``free`` has been re-read since the sweep, and quoting it on both
        # sides of the arrow produced "22.5 GB -> 22.5 GB free" on a call that
        # had just recovered fourteen gigabytes.
        note(family,
             f"freed {freed / _GB:.1f} GB for {reason or _reason_for(family)} "
             f"({before / _GB:.1f} GB -> {max(after, free) / _GB:.1f} GB free): "
             f"{result.describe()}")
    elif remaining > 0:
        note(family,
             f"{reason or _reason_for(family)} is short {remaining / _GB:.1f} GB on "
             f"{cuda_execution(where).describe() if where is not None else 'the card'} and "
             f"nothing evictable was found there; expect the driver to spill into system "
             f"memory{_unaccounted_note(scope)}")

    say_which_setting_did_this()
    return result


def release_for_llm(needed_bytes: int, *, card: int | None = None, reason: str = "",
                    margin: int | None = None) -> Reclaim:
    """The one door out of :func:`_victim_order`'s rule, and the user opens it.

    Every other path in this module keeps the asymmetry that module's docstring
    argues for: an image residency is never demoted for the language model,
    because the image model is the workload and the LLM is the helper writing a
    prompt for it. That rule is right as a *default* and wrong as an absolute,
    and the case it is wrong in is the one somebody states explicitly -- "on
    this card, right now, I would rather have the fast language model".

    So this exists, it is reached only from an accelerator plan whose memory
    priority the user set to LLM priority, and it is narrower than
    :func:`request_vram` in three ways that are the whole point:

    **It is scoped to one physical card.** ``card`` is the GPU the language
    model is being placed on, and image residency anywhere else is not touched
    and not even asked about. A Creative Writer on a 3090 gains nothing by
    emptying a 5090 -- the bytes it needs are on the 3090 -- so a reclaim that
    crossed cards would cost an image generation its checkpoint and buy the
    language model precisely nothing. Section 10 of the design intent gives
    that case three times, in both directions.

    **It asks for the deficit and no more.** Never a sweep. Exclusive mode's
    take-the-whole-card behaviour is a promise about *image* ownership and has
    no counterpart here: this is a request for enough room, and enough is
    exactly what it asks for.

    **It re-measures.** What comes back is a reading taken after the release,
    not the arithmetic that was hoped for -- so a caller that goes on to launch
    is deciding against what the driver says is there, which is the only number
    llama.cpp will agree with.

    Returns a :class:`Reclaim` whose ``deficit`` is what is *still* missing, so
    the caller can refuse rather than launch something that will not fit. It
    never launches, never places, and never decides that a plan is acceptable.
    """
    needed = max(int(needed_bytes), 0)
    reserve = safety_margin_bytes() if margin is None else int(margin)
    target = needed + reserve
    free = device_free_vram_bytes(card)
    if free <= 0:
        # The card could not be queried. Evicting on no evidence is the one
        # thing this module exists not to do, in this direction most of all.
        return Reclaim(needed, free, 0, max(target - max(free, 0), 0), ())
    if free >= target:
        return Reclaim(needed, free, 0, 0, ())

    deficit = target - free
    if not _same_card_as_the_image_side(card):
        note(FAMILY_LLM,
             f"{reason or 'the LLM'} is short {deficit / _GB:.1f} GB on GPU {card}, and the "
             f"image residency is on another card — nothing was released, because releasing "
             f"a card this model is not being placed on would not free a byte of the one it "
             f"is")
        return Reclaim(needed, free, 0, deficit, ())

    released = _release(FAMILY_IMAGE, deficit,
                        reason or "the LLM, which the user has given priority on this card",
                        card=card)
    after = device_free_vram_bytes(card)
    # The measurement wins over the arithmetic. ``_release`` reports what the
    # image side believes it handed back, and what matters to the launch that
    # follows is what the driver now says is free.
    remaining = max(target - max(after, free), 0)
    result = Reclaim(needed, free, released.freed, remaining, released.actions)
    if released.actions:
        note(FAMILY_LLM,
             f"released {released.freed / _GB:.1f} GB of image VRAM on GPU {card} for "
             f"{reason or 'the LLM'} ({free / _GB:.1f} GB -> {after / _GB:.1f} GB free): "
             f"{result.describe()}")
    elif remaining > 0:
        note(FAMILY_LLM,
             f"{reason or 'the LLM'} is short {remaining / _GB:.1f} GB on GPU {card} and the "
             f"image side had nothing evictable to give{_unaccounted_note(card)}")
    return result


def _same_card_as_the_image_side(card: int | None) -> bool:
    """Whether releasing image residency could help a placement on ``card``.

    Unanswerable questions are answered **no** here, which is the opposite of
    :func:`mc_llm_runtime.shares_the_image_card` and is the same caution
    pointing the other way. There, an unknown card is treated as the image card
    so the language model sizes itself conservatively and the cost of being
    wrong is a smaller model. Here, the cost of being wrong is an evicted
    checkpoint -- so an unknown card releases nothing.
    """
    if card is None:
        return False
    try:
        image = image_device_index()
    except Exception:
        logger.debug("Model Chain: could not ask which card the image side is on",
                     exc_info=True)
        return False
    if image < 0 or int(card) < 0:
        return False
    return int(card) == int(image)


def unaccounted_bytes(*, card=ANY_CARD) -> int:
    """VRAM in use that neither family admits to holding.

    The subtraction is the point. The card reports what is free; the image side
    reports what its models are holding; the register holds what this extension
    started. When those do not add up to the card, the difference is somebody
    else's -- another program on the same GPU, or a llama-server left running
    by a WebUI that was killed rather than closed, which holds its allocation
    for as long as it lives and is invisible to every check here.
    """
    total = total_vram_bytes()
    if total <= 0:
        return 0
    # Which card this is about: the one asked for, else the image side's, else
    # -- when neither can be established -- the machine, which is the answer
    # this function always gave and is right on the single-card installation it
    # was written for.
    named = _card_index(card if not isinstance(card, _AnyCard) else image_device_index())
    scope = named if named is not None else ANY_CARD
    accounted = sum(held_bytes(family, card=scope)
                    for family in (FAMILY_IMAGE, FAMILY_LLM))
    return max(total - free_vram_bytes() - accounted - _DRIVER_OVERHEAD
               - _own_llm_context_bytes(scope), 0)


_DRIVER_OVERHEAD = 1 * _GB
"""Allowed for before any of this is called unexplained.

A card in a desktop is never entirely free: the driver, the compositor and
whatever is drawing the browser this is being read in all hold some of it, and
a CUDA context is hundreds of megabytes before a single weight is loaded. A
gigabyte is generous for that and small enough that the case worth reporting --
a whole model somebody cannot see -- still clears it easily.
"""


def _own_llm_running(card=ANY_CARD) -> bool:
    """Whether a llama-server this WebUI started is executing on ``card``.

    Executing, not resident. A Mixed Conservative server holds its weights in
    system RAM and still keeps a CUDA context on the card it names, which is
    exactly the allowance :func:`_own_llm_context_bytes` exists to make -- and
    exactly the allowance that must not be made on a *different* card, where
    the server has nothing at all (section 7.6).
    """
    reclaimer = _reclaimer(FAMILY_LLM)
    if reclaimer is None:
        return False
    try:
        answer = _ask(reclaimer, "running", card, spans_cards=True)
    except Exception:
        logger.debug("Model Chain: could not ask whether llama-server is running",
                     exc_info=True)
        return False
    return bool(answer)


def _own_llm_context_bytes(card=ANY_CARD) -> int:
    """The card our own llama-server holds without holding any weights there.

    A server placed in system RAM reports nothing resident and declares
    nothing, which is correct: its weights are not on the card and the image
    side must not be told to come looking for them. Its *process* is on the
    card all the same -- a CUDA context is hundreds of megabytes before a
    single weight is loaded, which is exactly what :data:`_DRIVER_OVERHEAD`
    allows for -- so a second CUDA process gets a second allowance, and the
    figure below stops describing our own server as somebody else's stray.

    Only when it holds nothing here. A placement on the card is measured as a
    change in free VRAM at startup, so the context is already inside the
    declared figure, and subtracting it again would hide a gigabyte of real
    residency.
    """
    if not _own_llm_running(card):
        return 0
    if held_bytes(FAMILY_LLM, card=card) > 0:
        return 0
    return _DRIVER_OVERHEAD


def stray_explanation() -> str:
    """What VRAM neither family is holding is, as a sentence about the card.

    Returned without the amount so that each caller can format its own -- the
    console writes a figure into a log line, LLM Studio bolds one in a list --
    and shared so that the two cannot drift into two accounts of one card.

    Which sentence it is depends on whether a llama-server of ours is up. "A
    llama-server left running by a previous session" is a good guess when there
    is none of ours to blame, and is simply wrong when there is: the server
    this WebUI started keeps a CUDA context on the card even with its weights
    in system RAM. Telling somebody to go hunting for an orphaned process sends
    them looking for something that is not there, and the thing that would
    actually give those bytes back -- Unload -- is a button they already have.
    """
    if _own_llm_running():
        return ("of the card is in use by something outside both families. The "
                "llama-server this WebUI started keeps a CUDA context there even with "
                "its weights in system RAM, and any other program on the same GPU keeps "
                "its own; Unload in LLM Studio releases ours. Check nvidia-smi for the rest")
    return ("of the card is in use by something this WebUI is not managing — another "
            "program on the same GPU, or a llama-server left running by a previous "
            "session. Nothing here can reclaim that; check nvidia-smi")


def _unaccounted_note(card=ANY_CARD) -> str:
    """The other half of "nothing evictable was found", when there is one.

    "Nothing evictable" is true and, on its own, misleading: it reads as though
    the card were full of things this extension chose not to move, when what it
    usually means is that the card is full of something this extension cannot
    see at all. Two very different problems, one message -- so the message says
    which.
    """
    stray = unaccounted_bytes(card=card)
    if stray <= 0:
        return ""
    return f". {stray / _GB:.1f} GB {stray_explanation()}"


def _victim_order(family: str) -> tuple[str, ...]:
    """Which *other* family gives ground for an incoming ``family`` workload.

    Never the asking family. Making room for an LLM by demoting the LLM is not
    a policy, and image-on-image eviction is ``mc_memory``'s own business --
    it has the cache bookkeeping and the Forge entry points to do it properly,
    and this module would only be guessing at both.

    The asymmetry is the policy, and there is only one of it now: **an image
    residency is never demoted for the LLM.** The image model is the workload;
    the language model is a helper that writes a prompt for it, and a helper
    that throws thirteen gigabytes of checkpoint off the card so it can think
    faster has made the thing it was helping with slower. It is also not a
    trade that can be won: the checkpoint is needed again seconds later, so
    what the LLM borrows it pays back twice, once moving the weights out and
    once moving them back in.

    What is left for the LLM is the VRAM the image side is not using -- which
    :func:`mc_llm_runtime.negotiate` sizes itself against, shrinking its
    context, moving its experts to system RAM, dropping blocks, and running
    entirely from system RAM when nothing at all is spare.

    An image pass, conversely, always outranks an idle LLM. That is not a
    preference, it is section 18's regression requirement: ordinary txt2img has
    to keep working, and a background llama-server that could starve a
    generation the user is watching would break it.
    """
    return (FAMILY_LLM,) if family == FAMILY_IMAGE else ()


def _ask(reclaimer, method: str, card, *args, spans_cards: bool = False):
    """Call ``reclaimer.method`` with a card filter when it accepts one.

    The registered contract is deliberately small -- ``release(needed, reason)``
    and optionally ``resident_bytes()`` -- so a reclaimer is free to be
    card-blind, and one that is must keep working. The keyword is therefore
    offered and its refusal handled; what happens next depends on whether that
    reclaimer's family can span cards at all.

    ``spans_cards=False`` (the image family): a card-blind answer *is* the
    card-scoped answer. Forge generates on one card, everything the image side
    holds is on it, and the broker only ever targets that card for this family
    -- so the unfiltered call is used and is exact.

    ``spans_cards=True`` (the LLM family, which can be holding two cards at
    once): a card-blind answer is a machine-wide answer wearing one card's
    label, and using it is precisely the mistake invariant I-3 exists to
    prevent. So it comes back as "nothing eligible" instead, which costs a
    reclaim that did not happen and never costs the wrong process.
    """
    call = getattr(reclaimer, method, None)
    if not callable(call):
        return None
    if isinstance(card, _AnyCard):
        return call(*args)
    try:
        return call(*args, card=card)
    except TypeError:
        if spans_cards:
            logger.debug("Model Chain: %s cannot answer %s about one card, and its family "
                         "can hold more than one", type(reclaimer).__name__, method)
            return None
        return call(*args)


def _release(family: str, needed: int, reason: str, sweep: bool = False,
             card=ANY_CARD) -> Reclaim:
    """Ask ``family``'s reclaimer to free ``needed`` bytes on ``card``.

    ``card`` is a physical GPU index or :data:`ANY_CARD`. When it is an index,
    everything below -- the protection check, the sweep's "what does it hold",
    and the reclaimer call itself -- is about that card and nothing else, which
    is what makes a shortfall on GPU 0 unable to reach a runtime on GPU 1
    (invariant I-3).
    """
    reclaimer = _reclaimer(family)
    if reclaimer is None:
        return Reclaim(needed, 0, 0, needed, ())

    protected = [e for e in residencies(family, card=card) if not e.evictable]
    if protected and not sweep:
        held = sum(e.bytes for e in protected)
        available = max(resident_bytes(family, card=card) - held, 0)
        if available <= 0:
            note(family,
                 f"kept {', '.join(e.label for e in protected)} resident: "
                 f"{'pinned' if any(e.pinned for e in protected) else 'active'}")
            return Reclaim(needed, 0, 0, needed, ())
        needed = min(needed, available)

    if sweep:
        # Everything the family holds goes, not merely the arithmetic deficit.
        # What it holds is read from the register first and from the reclaimer
        # second: the register is the broker's own picture and is authoritative
        # when it has one, and a reclaimer that has never declared a residency
        # can still know it is holding something.
        held = resident_bytes(family, card=card)
        if held <= 0:
            try:
                held = max(int(_ask(reclaimer, "resident_bytes", card,
                                    spans_cards=family == FAMILY_LLM) or 0), 0)
            except Exception:
                held = 0
        if held <= 0:
            return Reclaim(needed, 0, 0, 0, ())
        needed = max(needed, held)

    try:
        freed = int(_ask(reclaimer, "release", card, needed, reason,
                         spans_cards=family == FAMILY_LLM) or 0)
    except Exception:
        logger.warning("Model Chain: %s reclaimer failed", family, exc_info=True)
        return Reclaim(needed, 0, 0, needed, ())

    if freed <= 0:
        return Reclaim(needed, 0, 0, needed, ())

    label = _describe(family, reclaimer)
    return Reclaim(needed, 0, freed, max(needed - freed, 0),
                   (f"demoted {label} ({freed / _GB:.1f} GB)",))


def _describe(family: str, reclaimer) -> str:
    describe = getattr(reclaimer, "describe", None)
    if callable(describe):
        try:
            described = str(describe() or "").strip()
            if described:
                return described
        except Exception:
            pass
    entries = residencies(family)
    if entries:
        return ", ".join(e.label for e in entries)
    return _reason_for(family)


# --------------------------------------------------------------------------- #
# Status (section 14)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Status:
    """One card's picture, plus the machine's, kept apart on purpose.

    ``free_vram``, ``total_vram`` and ``llm_bytes`` are all about the *image*
    card. That was already true of the first two and was not of the third, and
    the mismatch is section 7.5's complaint: nineteen gigabytes of 5090
    language model displayed beside a 3090's four gigabytes free reads as a
    card that is over-subscribed by fifteen, when in fact neither card is
    short of anything. Machine-wide figures still exist -- ``llm_bytes_total``
    -- they are simply no longer the number printed next to one card's total.
    """

    mode: str
    free_vram: int
    total_vram: int
    reserve: int
    image_bytes: int
    llm_bytes: int
    active: Active | None
    residencies: tuple[Residency, ...]
    card: int | None = None
    """The image card these VRAM figures describe, or None if unresolvable."""
    llm_bytes_total: int = 0
    """LLM VRAM everywhere in the machine, which is a different question."""
    activities: tuple[Active, ...] = ()
    """Everything running, image job included. See :func:`activities`."""
    free_ram: int = 0
    ram_reserve: int = 0
    image_warm_ram: int = 0

    @property
    def owners(self) -> tuple[str, ...]:
        families = []
        if self.image_bytes > 0:
            families.append(_named(FAMILY_IMAGE))
        if self.llm_bytes > 0:
            families.append(_named(FAMILY_LLM))
        return tuple(families)

    @property
    def llm_bytes_elsewhere(self) -> int:
        """LLM VRAM on cards other than the image card."""
        return max(self.llm_bytes_total - self.llm_bytes, 0)


def reported_bytes(family: str, *, card=ANY_CARD) -> int:
    """What ``family``'s reclaimer says it is holding on ``card``, or 0.

    The register is this module's own picture and only has entries for things
    that were declared. Image checkpoints are not: they are loaded by Forge,
    moved by Forge, and ``mc_memory`` cooperates with that rather than
    announcing every load here. So the honest answer for the image family comes
    from asking it, and :func:`status` asks.

    A ``card`` that the reclaimer cannot answer about comes back as 0 rather
    than as its machine-wide total: see :func:`_ask`.
    """
    reclaimer = _reclaimer(family)
    if reclaimer is None:
        return 0
    try:
        return max(int(_ask(reclaimer, "resident_bytes", card,
                            spans_cards=family == FAMILY_LLM) or 0), 0)
    except Exception:
        return 0


def held_bytes(family: str, *, card=ANY_CARD) -> int:
    """What ``family`` is holding on the card, declared or merely reported.

    The one function to ask when the question is "is it already there". The
    register alone answers 0 for a loaded checkpoint -- image models are loaded
    and moved by Forge and are never declared here (see :func:`reported_bytes`)
    -- and a caller that reserved room on the strength of that 0 reserved it
    for a model that was already resident. That is not hypothetical: it is why
    a Krea roll on a 24 GB card holding a 13.9 GB checkpoint sized its language
    model against 8.7 GB *minus another 13.9 GB*, decided almost nothing fit,
    and ran sixteen of forty-eight blocks from system RAM.
    """
    return max(resident_bytes(family, card=card), reported_bytes(family, card=card))


def status() -> Status:
    index = image_device_index()
    scope = index if index >= 0 else ANY_CARD
    entries = list(residencies())
    families = {}
    for family in (FAMILY_IMAGE, FAMILY_LLM):
        declared = resident_bytes(family, card=scope)
        families[family] = held_bytes(family, card=scope)
        if declared <= 0 and families[family] > 0:
            # Nothing declared, but the family says it is holding VRAM. Shown
            # as a synthetic row rather than left out: a residency panel that
            # omitted the loaded checkpoint would be answering a different
            # question from the one it appears to answer (section 14).
            entries.append(Residency(family=family, key=f"{family}:reported",
                                     label=_describe(family, _reclaimer(family)),
                                     bytes=families[family], rank=RANK_HOT,
                                     card=index if index >= 0 else None))

    return Status(
        mode=mode(),
        free_vram=free_vram_bytes(),
        total_vram=total_vram_bytes(),
        reserve=safety_margin_bytes(),
        image_bytes=families[FAMILY_IMAGE],
        llm_bytes=families[FAMILY_LLM],
        active=active(),
        residencies=tuple(sorted(entries, key=lambda e: (-e.effective_rank, -e.bytes))),
        card=index if index >= 0 else None,
        llm_bytes_total=held_bytes(FAMILY_LLM),
        activities=activities(),
        free_ram=free_ram_bytes(),
        ram_reserve=ram_reserve_bytes(),
        image_warm_ram=image_warm_ram_bytes(),
    )


# --------------------------------------------------------------------------- #
# The image family's reclaimer
# --------------------------------------------------------------------------- #
#
# Registered here rather than inside mc_memory so that mc_memory stays a module
# about image residency that knows nothing about LLMs -- it is imported and
# tested on its own, and a cross-workload broker is not a dependency it should
# grow. The mechanism is still entirely mc_memory's: everything below delegates.


class _ImageReclaimer:
    """Frees image-model VRAM by asking mc_memory to, and reports what it holds.

    Card-aware in the only way it can be. Forge generates on one card, so every
    byte this reclaimer can free is on that card and a filter naming it is
    satisfied by definition -- while a filter naming any *other* card is
    answered with nothing, honestly, rather than with the image card's figure
    under another card's heading (invariant I-2).
    """

    @staticmethod
    def _ours(card) -> bool:
        if isinstance(card, _AnyCard) or card is None:
            return True
        index = image_device_index()
        return index < 0 or int(card) == index

    def release(self, needed_bytes: int, reason: str = "", *, card=ANY_CARD) -> int:
        import mc_memory

        if not self._ours(card):
            return 0
        return int(mc_memory.release_vram(needed_bytes, reason=reason))

    def resident_bytes(self, *, card=ANY_CARD) -> int:
        import mc_memory

        if not self._ours(card):
            return 0
        try:
            return int(mc_memory.resident_vram_bytes())
        except Exception:
            return 0

    def describe(self) -> str:
        try:
            from modules import shared

            info = getattr(shared.sd_model, "sd_checkpoint_info", None)
            name = getattr(info, "name_for_extra", "") or getattr(info, "title", "")
            return f"the image checkpoint{f' ({name})' if name else ''}"
        except Exception:
            return "the image checkpoint"


register_reclaimer(FAMILY_IMAGE, _ImageReclaimer())


def _reclaim_for_image(shortfall_bytes: int, reason: str = "") -> int:
    """What ``mc_memory`` calls when its own eviction has fallen short.

    The two functions speak in different units and the conversion is the whole
    of this one. ``mc_memory`` has already subtracted what is free and already
    included its reserve, so what it passes is a *deficit*: bytes still missing.
    :func:`request_vram` takes a *requirement* and subtracts free VRAM itself.
    Handing the deficit straight over would therefore subtract free VRAM twice
    and ask for far less than the pass is short by -- which is the quiet kind
    of wrong, because something would be evicted and the pass would still not
    fit.

    So free VRAM is added back on, and the margin is zero because the reserve
    is already inside the number.
    """
    shortfall = max(int(shortfall_bytes), 0)
    if shortfall <= 0:
        return 0
    # Carrying the card is the whole of section 9.1. A pass that is four
    # gigabytes short on GPU 0 can only be helped by LLM residency on GPU 0,
    # and a nineteen-gigabyte language model on GPU 1 must survive the request
    # untouched -- it is not a candidate, not a last resort, and not a
    # smaller-than-ideal answer. It is simply on the wrong card.
    index = image_device_index()
    return request_vram(FAMILY_IMAGE, free_vram_bytes() + shortfall, reason=reason,
                        margin=0, exclusive_sweep=False,
                        card=index if index >= 0 else ANY_CARD).freed


try:
    import mc_memory as _mc_memory

    _mc_memory.set_foreign_reclaim(_reclaim_for_image)
except Exception:  # pragma: no cover - mc_memory is always importable in practice
    logger.debug("Model Chain: could not install the cross-workload reclaim hook",
                 exc_info=True)
