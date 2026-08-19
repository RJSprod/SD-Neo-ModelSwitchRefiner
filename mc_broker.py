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
    (MODE_HYBRID, "Hybrid — image and LLM may share VRAM when they fit"),
    (MODE_EXCLUSIVE, "Exclusive — one family owns VRAM at a time"),
)

POLICY_ADAPTIVE = "adaptive"
POLICY_PRESERVE_IMAGE = "preserve_image"
POLICY_LLM_PRIORITY = "llm_priority"

POLICIES = (
    (POLICY_ADAPTIVE, "Adaptive — demote whichever costs least"),
    (POLICY_PRESERVE_IMAGE, "Preserve image — shrink the LLM before the checkpoint moves"),
    (POLICY_LLM_PRIORITY, "LLM priority — give the LLM what it asked for"),
)

OPT_MODE = "model_chain_memory_mode"
OPT_POLICY = "model_chain_hybrid_policy"

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

STALE_AFTER_SECONDS = 15 * 60
"""How long a hot residency stays hot without being touched.

A number rather than a policy because the only thing it decides is eviction
*order*, never whether an eviction happens at all -- and something unused for
a quarter of an hour is a better victim than something unused for one minute
whatever the exact boundary is.
"""


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
    return default


def label_for(table, value: str) -> str:
    """``table``'s display text for ``value``, for putting in a control."""
    for candidate, label in table:
        if candidate == value:
            return label
    return value


def mode() -> str:
    """The configured residency mode. Hybrid is the default (section 8)."""
    return resolve(option(OPT_MODE, MODE_HYBRID), MODES, MODE_HYBRID)


def policy() -> str:
    """The configured hybrid placement policy (section 13)."""
    return resolve(option(OPT_POLICY, POLICY_ADAPTIVE), POLICIES, POLICY_ADAPTIVE)


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
            rank: int = RANK_HOT, pinned: bool = False) -> Residency:
    """Record that ``key`` now occupies VRAM, or update what is known about it."""
    with _register_lock:
        entry = _register.get(key)
        if entry is None:
            entry = Residency(family=family, key=key, label=label, bytes=int(size_bytes),
                              rank=rank, pinned=pinned)
            _register[key] = entry
        else:
            entry.family, entry.label = family, label
            entry.bytes, entry.rank, entry.pinned = int(size_bytes), rank, pinned
            entry.touch()
        return entry


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


def residencies(family: str | None = None) -> list[Residency]:
    with _register_lock:
        return [e for e in _register.values() if family is None or e.family == family]


def resident_bytes(family: str | None = None) -> int:
    return sum(e.bytes for e in residencies(family))


def clear() -> None:
    """Forget everything. For tests and for ``on_script_unloaded``."""
    with _register_lock:
        _register.clear()
    with _decision_lock:
        _decisions.clear()


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


_gpu = threading.RLock()
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


def active_family() -> str | None:
    running = active()
    return running.family if running is not None else None


class workload:
    """Context manager taking the shared workload lock for one job.

    Nested acquisition by the same thread is free: a Model Chain generation is
    one workload with two stages, not two workloads, and Stage 2 must not have
    to reason about a lock Stage 1 is already holding.

    ``background=True`` marks work nobody is waiting for. It never queues ahead
    of a foreground request and gives up rather than blocking one.
    """

    def __init__(self, family: str, label: str = "", *, timeout: float | None = None,
                 background: bool = False, required: bool = True):
        self.family, self.label = family, label or family
        self.timeout, self.background, self.required = timeout, background, required
        self._held: _Held | None = None
        self._counted = False

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

        with _active_lock:
            _active.append(Active(self.family, self.label, time.monotonic()))
        self._held = _Held(self.family, self.label, True)
        return self._held

    def __exit__(self, *exc) -> bool:
        if self._held is not None and self._held.acquired:
            with _active_lock:
                if _active:
                    _active.pop()
            _gpu.release()
        self._release_count()
        return False

    def _release_count(self) -> None:
        global _foreground_waiting

        if not self._counted:
            return
        self._counted = False
        with _active_lock:
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


def await_idle(timeout: float = 120.0) -> bool:
    """Block until no LLM workload holds the GPU. Returns whether it went idle.

    Called from the image side before a generation. Bounded because an image
    generation must never be blocked indefinitely by this extension: on
    timeout it returns False, the caller proceeds, and the two overlap -- which
    is slow, and is still better than a generation that never starts.
    """
    deadline = time.monotonic() + max(float(timeout), 0.0)
    first = True
    while active_family() == FAMILY_LLM:
        if time.monotonic() >= deadline:
            note(FAMILY_IMAGE,
                 "an LLM request was still running after the wait expired; the generation "
                 "starts anyway and the two will share the card briefly")
            return False
        if first:
            first = False
            logger.info("Model Chain: waiting for the LLM to finish before generating")
        time.sleep(0.05)
    return True


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


def device_free_vram_bytes() -> int:
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

        return int(mc_memory.device_free_vram_bytes())
    except Exception:
        return free_vram_bytes()


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


def request_vram(family: str, needed_bytes: int, *, reason: str = "",
                 margin: int | None = None, exclusive_sweep: bool = True) -> Reclaim:
    """Make room for ``needed_bytes`` of ``family`` work, moving as little as possible.

    The order of the branches below is the policy, so it is worth reading as
    prose:

    1. In Exclusive mode the other family leaves VRAM first, whether or not the
       arithmetic would have demanded it, because Exclusive mode is a promise
       about ownership rather than an optimisation (section 10).
    2. Otherwise, if the incoming workload already fits, nothing moves at all.
       This is the co-residency case and it is what Hybrid mode is for
       (section 8).
    3. Only then is the deficit -- and only the deficit -- freed, from the
       family the configured policy prefers to take it from (sections 9 and 13).
    """
    needed = max(int(needed_bytes), 0)
    reserve = safety_margin_bytes() if margin is None else int(margin)
    target = needed + reserve
    # Which "free" this is depends on who is asking. An image pass can spend
    # the host allocator's cache; llama.cpp is another process and cannot, so
    # asking on its behalf against the host's figure frees nothing and reports
    # a shortfall that is not there -- or, worse, no shortfall when there is.
    reading = device_free_vram_bytes if family == FAMILY_LLM else free_vram_bytes
    free = reading()
    actions: list[str] = []
    freed = 0
    # Read once. A setting that changed between the two branches below would
    # otherwise be able to produce a call that swept a family *and* went on to
    # demote a second one for the same request.
    sweeping = exclusive_sweep and mode() == MODE_EXCLUSIVE

    # Exclusive mode is checked before the fit, and that ordering is the whole
    # difference between the two modes. Hybrid asks "does this fit"; Exclusive
    # asks "who owns the card", and the answer has to be the same whether or
    # not the incoming workload happened to fit alongside the outgoing one --
    # otherwise ownership would depend on the size of the last request, which
    # is precisely the unpredictability Exclusive mode exists to remove
    # (sections 10 and 18).
    if sweeping:
        other = FAMILY_LLM if family == FAMILY_IMAGE else FAMILY_IMAGE
        released = _release(other, target, reason or f"{family} took VRAM ownership",
                            sweep=True)
        freed += released.freed
        actions.extend(released.actions)
        free = reading()

    if free <= 0:
        # VRAM could not be queried. Guessing at a deficit here would evict on
        # no evidence, which is exactly what this module exists not to do.
        return Reclaim(needed, free, freed, 0, tuple(actions))

    if free >= target:
        result = Reclaim(needed, free, freed, 0, tuple(actions))
        if actions:
            note(family, f"{reason or family}: {result.describe()}")
        return result

    deficit = target - free

    if not sweeping:
        for victim_family in _victim_order(family):
            if freed >= deficit:
                break
            released = _release(victim_family, deficit - freed,
                                reason or f"{family} workload needed VRAM")
            freed += released.freed
            actions.extend(released.actions)

    after = reading()
    remaining = max(target - max(after, free + freed), 0)
    result = Reclaim(needed, free, freed, remaining, tuple(actions))

    if actions:
        note(family,
             f"freed {freed / _GB:.1f} GB for {reason or family} "
             f"({free / _GB:.1f} GB -> {max(after, free) / _GB:.1f} GB free): {result.describe()}")
    elif remaining > 0:
        note(family,
             f"{reason or family} is short {remaining / _GB:.1f} GB and nothing evictable "
             f"was found; expect the driver to spill into system memory{_unaccounted_note()}")

    return result


def unaccounted_bytes() -> int:
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
    accounted = sum(max(resident_bytes(family), reported_bytes(family))
                    for family in (FAMILY_IMAGE, FAMILY_LLM))
    return max(total - free_vram_bytes() - accounted - _DRIVER_OVERHEAD, 0)


_DRIVER_OVERHEAD = 1 * _GB
"""Allowed for before any of this is called unexplained.

A card in a desktop is never entirely free: the driver, the compositor and
whatever is drawing the browser this is being read in all hold some of it, and
a CUDA context is hundreds of megabytes before a single weight is loaded. A
gigabyte is generous for that and small enough that the case worth reporting --
a whole model somebody cannot see -- still clears it easily.
"""


def _unaccounted_note() -> str:
    """The other half of "nothing evictable was found", when there is one.

    "Nothing evictable" is true and, on its own, misleading: it reads as though
    the card were full of things this extension chose not to move, when what it
    usually means is that the card is full of something this extension cannot
    see at all. Two very different problems, one message -- so the message says
    which.
    """
    stray = unaccounted_bytes()
    if stray <= 0:
        return ""
    return (f". {stray / _GB:.1f} GB of the card is in use by something this WebUI is not "
            f"managing — another program on the same GPU, or a llama-server left running by "
            f"a previous session. Nothing here can reclaim that; check nvidia-smi")


def _victim_order(family: str) -> tuple[str, ...]:
    """Which *other* family gives ground for an incoming ``family`` workload.

    Never the asking family. Making room for an LLM by demoting the LLM is not
    a policy, and image-on-image eviction is ``mc_memory``'s own business --
    it has the cache bookkeeping and the Forge entry points to do it properly,
    and this module would only be guessing at both.

    An image pass always outranks an idle LLM, under every policy. That is not
    a preference, it is section 18's regression requirement: ordinary txt2img
    has to keep working, and a policy that let a background llama-server starve
    a generation the user is watching would break it. What the policy governs
    is the LLM's ambitions, not whether images can run.
    """
    if family == FAMILY_IMAGE:
        return (FAMILY_LLM,)

    if policy() == POLICY_PRESERVE_IMAGE:
        # Nothing gives ground. The answer for an LLM that does not fit is to
        # place it smaller, which mc_llm_runtime does before asking at all.
        return ()
    return (FAMILY_IMAGE,)


def _release(family: str, needed: int, reason: str, sweep: bool = False) -> Reclaim:
    """Ask ``family``'s reclaimer to free ``needed`` bytes (or everything, if sweeping)."""
    reclaimer = _reclaimer(family)
    if reclaimer is None:
        return Reclaim(needed, 0, 0, needed, ())

    protected = [e for e in residencies(family) if not e.evictable]
    if protected and not sweep:
        held = sum(e.bytes for e in protected)
        available = max(resident_bytes(family) - held, 0)
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
        held = resident_bytes(family)
        if held <= 0:
            reported = getattr(reclaimer, "resident_bytes", None)
            try:
                held = int(reported() or 0) if callable(reported) else 0
            except Exception:
                held = 0
        if held <= 0:
            return Reclaim(needed, 0, 0, 0, ())
        needed = max(needed, held)

    try:
        freed = int(reclaimer.release(needed, reason) or 0)
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
    return f"the {family} workload"


# --------------------------------------------------------------------------- #
# Status (section 14)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Status:
    mode: str
    policy: str
    free_vram: int
    total_vram: int
    reserve: int
    image_bytes: int
    llm_bytes: int
    active: Active | None
    residencies: tuple[Residency, ...]

    @property
    def owners(self) -> tuple[str, ...]:
        families = []
        if self.image_bytes > 0:
            families.append("image")
        if self.llm_bytes > 0:
            families.append("LLM")
        return tuple(families)


def reported_bytes(family: str) -> int:
    """What ``family``'s reclaimer says it is holding, or 0.

    The register is this module's own picture and only has entries for things
    that were declared. Image checkpoints are not: they are loaded by Forge,
    moved by Forge, and ``mc_memory`` cooperates with that rather than
    announcing every load here. So the honest answer for the image family comes
    from asking it, and :func:`status` asks.
    """
    reclaimer = _reclaimer(family)
    reported = getattr(reclaimer, "resident_bytes", None)
    if not callable(reported):
        return 0
    try:
        return max(int(reported() or 0), 0)
    except Exception:
        return 0


def status() -> Status:
    entries = list(residencies())
    families = {}
    for family in (FAMILY_IMAGE, FAMILY_LLM):
        declared = resident_bytes(family)
        families[family] = max(declared, reported_bytes(family))
        if declared <= 0 and families[family] > 0:
            # Nothing declared, but the family says it is holding VRAM. Shown
            # as a synthetic row rather than left out: a residency panel that
            # omitted the loaded checkpoint would be answering a different
            # question from the one it appears to answer (section 14).
            entries.append(Residency(family=family, key=f"{family}:reported",
                                     label=_describe(family, _reclaimer(family)),
                                     bytes=families[family], rank=RANK_HOT))

    return Status(
        mode=mode(),
        policy=policy(),
        free_vram=free_vram_bytes(),
        total_vram=total_vram_bytes(),
        reserve=safety_margin_bytes(),
        image_bytes=families[FAMILY_IMAGE],
        llm_bytes=families[FAMILY_LLM],
        active=active(),
        residencies=tuple(sorted(entries, key=lambda e: (-e.effective_rank, -e.bytes))),
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
    """Frees image-model VRAM by asking mc_memory to, and reports what it holds."""

    def release(self, needed_bytes: int, reason: str = "") -> int:
        import mc_memory

        return int(mc_memory.release_vram(needed_bytes, reason=reason))

    def resident_bytes(self) -> int:
        import mc_memory

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
    return request_vram(FAMILY_IMAGE, free_vram_bytes() + shortfall, reason=reason,
                        margin=0, exclusive_sweep=False).freed


try:
    import mc_memory as _mc_memory

    _mc_memory.set_foreign_reclaim(_reclaim_for_image)
except Exception:  # pragma: no cover - mc_memory is always importable in practice
    logger.debug("Model Chain: could not install the cross-workload reclaim hook",
                 exc_info=True)
