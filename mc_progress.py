"""Whole-job progress and ETA for a Model Chain generation.

The host computes progress from completed sampler jobs and steps
(``modules/progress.py``), which describes a single sampling loop well and a
Model Chain job badly. A chained job is a sequence of *unlike* phases -- waiting
for a preload, freeing VRAM, sampling Stage 1, decoding, moving several
gigabytes of weights, sampling Stage 2 once per image -- and counting steps
weights them all as though they cost the same. The visible result is a bar that
reaches 100% when Stage 1 ends and then jumps backwards, and an ETA that is
wrong by however long the model switch takes.

This module models one job as an ordered list of phases, each with a predicted
duration, and reports *the fraction of predicted wall time spent* rather than
the fraction of steps taken. Two things follow from that choice:

* the bar moves at a roughly constant rate through work of different kinds, and
* the remaining time falls out of the same arithmetic, so there is no second,
  separately-wrong ETA calculation to maintain.

Predictions start from a small built-in table scaled by detectable hardware, and
are replaced by measurement. Every phase is timed as it runs, and the resulting
rates -- seconds per gigabyte moved, seconds per step per megapixel sampled --
are folded into a store that persists across restarts. The table exists only to
make the first job of a fresh install approximately right; after that the user's
own machine is the authority.

Nothing here is allowed to affect what comes out of the pipeline. Every entry
point swallows its own exceptions and every consumer treats a missing estimate
as "fall back to the host's own number".
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

FILENAME = "model_chain_timing.json"

SCHEMA_VERSION = 1

OPT_PROGRESS = "model_chain_progress"
"""Settings key for the whole-job progress calculation."""

PROGRESS_DEFAULT = True

_GB = 1024**3


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

BASELINES = {
    # Residency kinds, as mc_memory.plan() names them. "warm" is the only one
    # that avoids a disk read: the model is already in the RAM cache and the
    # switch is a copy back across PCIe. "dual" means it fits alongside Model A
    # without an eviction, which says nothing about where it is coming *from* --
    # it is still a first load. "offload" adds moving Model A out to RAM on top.
    "move:warm": 0.45,
    "move:dual": 2.0,
    "move:disk": 2.2,
    "move:offload": 2.4,
    "move:cold": 2.0,
    "move:unchanged": 0.02,
    "free": 0.35,
    "join": 0.0,
    "finalize": 0.4,
    "sample": 0.55,
}
"""First-run guesses, in seconds per unit of whatever the phase measures.

Movement and freeing are per gigabyte, sampling is per step per megapixel per
image, and the rest are flat per-job costs. These are deliberately coarse. They
are what the very first chained generation on a fresh install has to work from,
and they are overwritten by measurement from the second one onwards -- so the
cost of a wrong entry here is one job's worth of a poor ETA, not a permanently
wrong prediction.
"""

_REFERENCE_VRAM_GB = 24.0
"""The card the sampling baseline above was written for.

Used only to tilt the first-run sampling guess by card class. VRAM is a poor
proxy for compute, but it is the one number that is reliably readable offline on
every platform the host runs on, and being roughly right on a 8 GB card beats
being confidently wrong. The result is clamped hard because the correlation
breaks down at both ends.
"""

_MIN_SAMPLE_BASELINE = 0.15
_MAX_SAMPLE_BASELINE = 4.0

_SMOOTHING = 0.4
"""Weight given to the newest measurement when folding it into a stored rate.

Recent comparable jobs should outweigh generic guesses, which is why this is
high enough to move quickly. It is not 1.0 because a single pass that contended
with something else on the machine should not become the new truth.
"""

_TAIL_FRACTION = 0.1
"""How much of a phase's estimate is held back once its estimate is exhausted.

A phase that runs longer than predicted must still report time remaining, or the
bar would stall at exactly the moment the user most wants to know it has not
hung. Reporting a shrinking remainder rather than zero keeps it moving.
"""

_CEILING = 0.999
"""Progress never reaches 1.0 from the model.

The host owns completion: the bar is removed when the task finishes, not when
the estimate says it should have. Reporting 1.0 early would show a full bar with
work still running, which is the failure this module exists to remove.
"""

SAMPLE_TAIL = 0.12
"""Share of a sampling phase the step counter cannot see.

A pass is not over when its last step is: the latents still have to be decoded,
and the result assembled. The host exposes no counter for that, so the step
fraction is scaled to leave this much of the phase unaccounted for, and the bar
keeps creeping through the decode instead of stalling on a full step count. The
measured rate for the phase includes the decode too -- consistently, every time
-- so this only has to be roughly right.
"""


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

PHASE_JOIN = "join"
PHASE_STAGE1_PREPARE = "stage1_prepare"
PHASE_STAGE1 = "stage1"
PHASE_SWITCH = "switch"
PHASE_STAGE2_PREPARE = "stage2_prepare"
PHASE_STAGE2 = "stage2"
PHASE_FINALIZE = "finalize"

ORDER = (
    PHASE_JOIN,
    PHASE_STAGE1_PREPARE,
    PHASE_STAGE1,
    PHASE_SWITCH,
    PHASE_STAGE2_PREPARE,
    PHASE_STAGE2,
    PHASE_FINALIZE,
)
"""Every phase of a chained job, in the order it runs.

The list is what changed most when the VRAM work landed. Waiting on a preload
that is still running, and freeing VRAM so a model loads in one piece instead of
being squeezed in against a full card, are both real spans of a real job now --
the first at the very front, where the bar has nothing else to show, and the
second twice. Neither existed when this was first specified.

What is *not* here is the Stage 1 preload that runs after the images are
delivered. It is deliberately off the list: it happens on a background thread
after postprocess returns, by which point the host has finished the task and
torn the progress bar out of the DOM. There is nothing left to report it on, so
"job complete" means the images are ready, and the panel's readiness line is
where the preload's state is visible.
"""


class Phase:
    """One timed span of a job, with a prediction and somewhere to record fact.

    ``units`` is what the phase's rate is expressed per: gigabytes for a move or
    a free, megapixels for a decode, step-megapixel-images for sampling, and
    zero for a flat cost. A phase with no units cannot teach the store a rate,
    only a duration.
    """

    __slots__ = (
        "key", "rate_keys", "label", "units", "estimate",
        "weights", "tail", "started", "actual", "passes_started",
    )

    def __init__(self, key, rate_keys, label, units, estimate, weights=(), tail=0.0):
        self.key = key
        self.rate_keys = tuple(rate_keys)
        self.label = label
        self.units = float(units)
        self.estimate = max(float(estimate), 0.0)
        # One entry per sampler pass, holding that pass's share of the phase.
        # Hires fix makes these uneven -- a 20-step first pass and a 10-step
        # second one alternate through the batch -- so a flat "passes done over
        # passes total" would make the bar speed up and slow down for no reason
        # the user can see.
        self.weights = tuple(float(w) for w in weights)
        self.tail = float(tail)
        self.started: float | None = None
        self.actual: float | None = None
        self.passes_started = 0

    @property
    def finished(self) -> bool:
        return self.actual is not None

    def elapsed(self, now: float) -> float:
        if self.actual is not None:
            return self.actual
        if self.started is None:
            return 0.0
        return max(now - self.started, 0.0)


# --------------------------------------------------------------------------- #
# Calibration store
# --------------------------------------------------------------------------- #

_lock = threading.RLock()
_rates: dict[str, float] | None = None
_dirty = False


def path() -> str:
    """Where measured timings are stored.

    Beside the presets, in the WebUI data directory rather than the extension
    folder, so reinstalling the extension does not throw away a calibration that
    took real generations to earn.
    """
    try:
        from modules import paths

        base = paths.data_path
    except Exception:
        base = os.getcwd()
    return os.path.join(base, FILENAME)


def _load() -> dict[str, float]:
    global _rates
    if _rates is not None:
        return _rates

    _rates = {}
    file = path()
    if not os.path.exists(file):
        return _rates

    try:
        with open(file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        logger.warning(
            "Model Chain: could not read measured timings from %s; starting from the "
            "built-in estimates instead",
            file,
            exc_info=True,
        )
        return _rates

    rates = data.get("rates") if isinstance(data, dict) else None
    if isinstance(rates, dict):
        for key, value in rates.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            # A stored zero or a negative would make a phase predict instantly,
            # which is worse than having no measurement at all.
            if value > 0:
                _rates[str(key)] = value

    return _rates


def save() -> None:
    """Persist measured rates, best-effort.

    A failure here costs the next session its calibration and nothing else, so
    it is logged once at debug level rather than surfaced.
    """
    global _dirty

    with _lock:
        if not _dirty or _rates is None:
            return
        payload = {"version": SCHEMA_VERSION, "rates": dict(_rates)}
        _dirty = False

    file = path()
    directory = os.path.dirname(file) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        # Written beside the target so the replace stays on one filesystem.
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".model_chain_timing", suffix=".tmp", delete=False
        )
        try:
            with handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(handle.name, file)
        except Exception:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise
    except Exception:
        logger.debug("Model Chain: could not save measured timings to %s", file, exc_info=True)


def forget() -> None:
    """Drop every measurement. Exposed for tests and for a manual reset."""
    global _rates, _dirty
    with _lock:
        _rates = {}
        _dirty = True


def rates() -> dict[str, float]:
    """A copy of the measured rates currently in force."""
    with _lock:
        return dict(_load())


def _baseline(rate_key: str) -> float:
    """The built-in guess for a rate key, tilted by hardware where that helps."""
    if rate_key.startswith("sample"):
        return _sample_baseline()

    if rate_key in BASELINES:
        return BASELINES[rate_key]

    # An unseen transition kind is treated as the most expensive one rather than
    # as free: an under-estimate shows a bar that stalls, which reads as a hang.
    if rate_key.startswith("move:"):
        return BASELINES["move:disk"]
    return 0.0


def _sample_baseline() -> float:
    """Seconds per step per megapixel per image, before any measurement."""
    baseline = BASELINES["sample"]
    try:
        import mc_memory

        total = mc_memory.total_vram_bytes() / _GB
    except Exception:
        total = 0.0

    if total > 0:
        baseline *= _REFERENCE_VRAM_GB / total

    return min(max(baseline, _MIN_SAMPLE_BASELINE), _MAX_SAMPLE_BASELINE)


def rate_for(rate_keys) -> float:
    """The best rate available for a phase, most specific key first.

    Phases carry more than one key so a measurement can be recorded against both
    an exact match and a broader one. Sampling is the case that matters: a rate
    learned at batch size 4 is the right answer for the next batch of 4 and only
    an approximation for a batch of 1, so both are kept and the exact key wins.
    """
    store = _load()
    for key in rate_keys:
        value = store.get(key)
        if value:
            return value
    return _baseline(rate_keys[-1] if rate_keys else "")


def _record(rate_keys, seconds: float, units: float) -> None:
    """Fold one measurement into every key the phase answers to."""
    global _dirty

    if units <= 0 or seconds <= 0 or not rate_keys:
        return

    measured = seconds / units
    store = _load()
    for key in rate_keys:
        previous = store.get(key)
        if previous:
            store[key] = previous + (measured - previous) * _SMOOTHING
        else:
            # Nothing measured yet: take the observation whole rather than
            # averaging it with a guess that was never about this machine.
            store[key] = measured
    _dirty = True


# --------------------------------------------------------------------------- #
# Job model
# --------------------------------------------------------------------------- #


class Job:
    """One chained generation, as an ordered list of phases.

    Phases are declared up front, in the order they will run, and entered by key
    as the job reaches them. Entering a phase closes every phase before it, so a
    phase that turns out not to be needed -- a VRAM free that was unnecessary, a
    switch to a model already loaded -- costs nothing and simply shortens the
    prediction.
    """

    def __init__(self):
        self.phases: list[Phase] = []
        self._index: dict[str, int] = {}
        self.current = -1
        self.started: float | None = None
        # Wall-clock counterpart of ``started``, kept only so a job can be
        # recognised as belonging to an earlier task than the one being polled.
        self.wall_started = 0.0
        self.reported = 0.0
        self.finished = False

    # -- construction ---------------------------------------------------- #

    def add(self, key, label, *, rate_keys=(), units=0.0, flat=0.0, weights=(), tail=0.0) -> Phase:
        """Declare a phase.

        ``flat`` is a per-job cost in seconds used when the phase has no units
        to scale by; ``units`` and ``rate_keys`` describe one that does.
        """
        if units > 0 and rate_keys:
            estimate = rate_for(rate_keys) * units
        else:
            estimate = rate_for(rate_keys) if rate_keys else flat

        phase = Phase(key, rate_keys, label, units, estimate, weights, tail)
        self._index[key] = len(self.phases)
        self.phases.append(phase)
        return phase

    def add_move(self, key, label, kind, gigabytes) -> Phase:
        return self.add(key, label, rate_keys=(f"move:{kind or 'disk'}",), units=max(gigabytes, 0.0))

    def add_free(self, key, label, gigabytes) -> Phase:
        return self.add(key, label, rate_keys=("free",), units=max(gigabytes, 0.0))

    def add_sampling(self, key, label, *, arch, batch, passes, tail=SAMPLE_TAIL) -> Phase:
        """Declare a run of sampler passes.

        ``passes`` is a sequence of ``(steps, megapixels)``, one entry per
        sampler pass in the order they will run -- so a hires job lists its
        first and second passes separately, and a batch count of two lists all
        four.

        The unit is one step over one megapixel of one image, so a rate learned
        at one resolution predicts another. Batch size is folded into the units
        rather than modelled separately, and also into the most specific rate
        key: batching is sublinear on real hardware, so a rate measured at batch
        4 is kept apart from one measured at batch 1 and only falls back to it
        when there is nothing better. Stage 2 is where that matters most --
        every refine runs as its own batch-of-one pass, so a chain gets none of
        Stage 1's batching gain and must not be predicted as though it did.
        """
        batch = max(int(batch), 1)
        weights = [max(int(steps), 0) * max(float(megapixels), 0.0) for steps, megapixels in passes]
        units = sum(weights) * batch
        arch = (arch or "unknown").replace(":", "_")
        return self.add(
            key,
            label,
            rate_keys=(f"sample:{arch}:b{batch}", f"sample:{arch}", "sample"),
            units=units,
            weights=weights,
            tail=tail,
        )

    # -- running ---------------------------------------------------------- #

    def start(self, at: float | None = None) -> None:
        """Begin the job's clock, optionally backdated.

        Backdating matters: a job's first phases run before the extension knows
        the chain is armed, so the plan is built after they are already over.
        Starting the clock when they started is what stops that time being
        invisible to the bar.
        """
        now = time.perf_counter()
        self.started = at if at is not None else now
        self.wall_started = time.time() - (now - self.started)

    def record(self, key: str, seconds: float, units: float | None = None) -> None:
        """Mark a phase complete with a duration measured outside the job.

        Used for the phases that finish before the plan exists. The measurement
        still teaches the store, so a preload wait or a Stage 1 VRAM free counts
        towards the next job's prediction exactly as any other phase would.
        """
        index = self._index.get(key)
        if index is None:
            return
        phase = self.phases[index]
        if phase.finished:
            return
        if units is not None:
            phase.units = max(float(units), 0.0)
        phase.actual = max(float(seconds), 0.0)
        _record(phase.rate_keys, phase.actual, phase.units)

    def enter(self, key: str) -> None:
        """Advance to a phase, closing everything declared before it."""
        target = self._index.get(key)
        if target is None or target < self.current:
            return

        now = time.perf_counter()
        for index in range(max(self.current, 0), target):
            self._close(self.phases[index], now)

        self.current = target
        phase = self.phases[target]
        if phase.started is None:
            phase.started = now

    def note_pass(self) -> None:
        """A sampler pass is about to begin inside the current phase."""
        if 0 <= self.current < len(self.phases):
            self.phases[self.current].passes_started += 1

    def relabel(self, text: str) -> None:
        """Rename the current phase, for a label that counts through a batch."""
        if 0 <= self.current < len(self.phases) and text:
            self.phases[self.current].label = text

    def finish(self) -> None:
        now = time.perf_counter()
        for phase in self.phases:
            self._close(phase, now)
        self.finished = True

    def _close(self, phase: Phase, now: float) -> None:
        if phase.finished:
            return
        if phase.started is None:
            # Never reached. It contributed an estimate to the prediction and
            # nothing to the measurement; dropping it shortens what remains.
            phase.actual = 0.0
            return
        phase.actual = max(now - phase.started, 0.0)
        _record(phase.rate_keys, phase.actual, phase.units)

    # -- reporting -------------------------------------------------------- #

    def remaining(self, now: float, sampling: tuple[int, int] | None) -> float:
        """Predicted seconds left, from the current phase onwards."""
        total = 0.0
        for index, phase in enumerate(self.phases):
            if phase.finished:
                continue
            if index > self.current:
                total += phase.estimate
                continue
            total += self._phase_remaining(phase, now, sampling)
        return total

    def _phase_remaining(self, phase: Phase, now: float, sampling) -> float:
        elapsed = phase.elapsed(now)
        fraction = self._fraction(phase, sampling)

        if fraction is not None and fraction > 0.02:
            # Real progress inside the phase: extrapolate from what it has
            # actually cost so far rather than from what it was predicted to.
            # This is what makes a bad initial estimate self-correct mid-phase.
            projected = elapsed / fraction
            return max(projected - elapsed, 0.0)

        # Opaque phase -- weights moving, VRAM being freed. Nothing reports
        # partial completion, so the estimate is all there is; hold a shrinking
        # tail back so an overrun keeps counting down instead of stalling at 0.
        return max(phase.estimate - elapsed, phase.estimate * _TAIL_FRACTION)

    @staticmethod
    def _fraction(phase: Phase, sampling) -> float | None:
        """How far through a phase is, where the phase can say."""
        weights = phase.weights
        if not weights or phase.passes_started <= 0:
            return None

        total = sum(weights)
        if total <= 0:
            return None

        index = min(phase.passes_started, len(weights)) - 1
        step, steps = sampling if sampling else (0, 0)
        within = (step / steps) if steps > 0 else 0.0

        done = sum(weights[:index]) + weights[index] * min(within, 1.0)
        # Scaled down by the tail so a phase whose steps are all counted still
        # has somewhere to go while its output is decoded.
        return min(done / total, 1.0) * (1.0 - phase.tail)

    def snapshot(self, sampling) -> tuple[float, float, str]:
        """Progress in 0..1, seconds remaining, and the current phase label."""
        now = time.perf_counter()
        spent = max(now - (self.started or now), 0.0)
        remaining = self.remaining(now, sampling)

        total = spent + remaining
        progress = (spent / total) if total > 0 else 0.0

        # Monotonic by construction. Remaining time is free to grow -- a phase
        # running long should push the ETA out -- but the bar may not go
        # backwards, which is the whole complaint this module answers.
        progress = min(max(progress, self.reported), _CEILING)
        self.reported = progress

        label = ""
        if 0 <= self.current < len(self.phases):
            label = self.phases[self.current].label

        return progress, remaining, label

    def describe(self) -> str:
        """One line naming the predicted cost of each phase, for the console."""
        parts = [f"{phase.label} {phase.estimate:.1f}s" for phase in self.phases if phase.estimate > 0.05]
        return ", ".join(parts)

    @property
    def estimate(self) -> float:
        return sum(phase.estimate for phase in self.phases)


# --------------------------------------------------------------------------- #
# Module-level job control
# --------------------------------------------------------------------------- #

_job: Job | None = None


def enabled() -> bool:
    """Whether the whole-job calculation is switched on."""
    try:
        from modules import shared

        return bool(getattr(shared.opts, OPT_PROGRESS, PROGRESS_DEFAULT))
    except Exception:
        return PROGRESS_DEFAULT


def begin(job: Job, since: float | None = None) -> None:
    """Adopt a freshly built plan and start its clock.

    ``since`` is a ``time.perf_counter()`` reading from before the plan existed,
    for the phases already recorded against it.
    """
    global _job
    with _lock:
        job.start(since)
        _job = job
    logger.info(
        "Model Chain: predicting %.0fs for this job (%s)", job.estimate, job.describe() or "no phases"
    )


def new_job() -> Job:
    """An empty plan. Populate it, then hand it to :func:`begin`."""
    return Job()


def build(
    *,
    stage1_arch: str,
    stage1_passes,
    batch_size: int,
    stage2_arch: str,
    stage2_passes,
    transition: str,
    move_gigabytes: float,
    free_gigabytes: float,
    target_label: str = "",
) -> Job:
    """The full phase list for one chained generation.

    ``stage1_passes`` and ``stage2_passes`` are sequences of ``(steps,
    megapixels)`` -- see :meth:`Job.add_sampling`. ``transition`` is the
    residency kind ``mc_memory.plan()`` predicted for the switch, which is what
    decides whether the move is a pointer swap, a copy back from system RAM, or
    a read from disk. Getting that from the same call the console log uses means
    the estimate and the explanation the user is shown cannot disagree.
    """
    job = Job()

    job.add(PHASE_JOIN, "Waiting for Stage 1 preload", rate_keys=("join",))
    job.add_free(PHASE_STAGE1_PREPARE, "Preparing Stage 1", free_gigabytes)
    job.add_sampling(
        PHASE_STAGE1, "Stage 1", arch=stage1_arch, batch=batch_size, passes=stage1_passes
    )
    job.add_move(
        PHASE_SWITCH,
        f"Loading {target_label}" if target_label else "Stage 2 model",
        transition,
        move_gigabytes,
    )
    job.add_free(PHASE_STAGE2_PREPARE, "Freeing VRAM for Stage 2", free_gigabytes)
    job.add_sampling(
        PHASE_STAGE2, "Stage 2", arch=stage2_arch, batch=1, passes=stage2_passes
    )
    job.add(PHASE_FINALIZE, "Finishing", rate_keys=("finalize",))

    return job


def enter(key: str) -> None:
    """Advance the live job to a phase. Safe to call when there is no job."""
    with _lock:
        if _job is not None and not _job.finished:
            _job.enter(key)


def note_pass() -> None:
    """Tell the live job a sampler pass is starting."""
    with _lock:
        if _job is not None and not _job.finished:
            _job.note_pass()


def relabel(text: str) -> None:
    """Rename the phase currently running, for example "Stage 2 1/2"."""
    with _lock:
        if _job is not None and not _job.finished:
            _job.relabel(text)


def end() -> None:
    """Close the live job, fold its measurements in, and persist them."""
    global _job
    with _lock:
        job = _job
        _job = None
        if job is None:
            return
        job.finish()
    save()


def abandon() -> None:
    """Drop the live job without learning from it.

    Used when a generation is interrupted or fails: a phase cut short mid-way
    measures the interruption, not the work, and folding that in would teach the
    store that everything is faster than it is.
    """
    global _job
    with _lock:
        _job = None


def active() -> bool:
    with _lock:
        return _job is not None and not _job.finished


def snapshot() -> tuple[float, float, str] | None:
    """Progress, seconds remaining and phase label for the live job, or None."""
    with _lock:
        job = _job
        if job is None or job.finished or job.started is None:
            return None
        sampling = _sampling_position()
        try:
            return job.snapshot(sampling)
        except Exception:
            return None


def _stale() -> bool:
    """Whether the live job belongs to a task the host has already moved past.

    ``state.time_start`` is stamped when a task begins, so a plan that started
    before it cannot be describing the task now being polled.
    """
    with _lock:
        job = _job
        if job is None:
            return False
        began = job.wall_started

    try:
        from modules import shared

        task_started = float(getattr(shared.state, "time_start", 0) or 0)
    except Exception:
        return False

    return bool(task_started) and bool(began) and began < task_started


def _sampling_position() -> tuple[int, int] | None:
    """The host's step counter for the pass running right now."""
    try:
        from modules import shared

        steps = int(getattr(shared.state, "sampling_steps", 0) or 0)
        step = int(getattr(shared.state, "sampling_step", 0) or 0)
    except Exception:
        return None
    if steps <= 0:
        return None
    return min(step, steps), steps


# --------------------------------------------------------------------------- #
# Host integration
# --------------------------------------------------------------------------- #

_installed = False


def install() -> bool:
    """Wrap the host's progress endpoint so a chained job can report its own.

    ``modules.progress.setup_progress_api`` resolves ``progressapi`` from the
    module globals when it registers the route, and that happens long after
    extensions are imported -- so rebinding the global here is picked up. The
    original is called first and only the three fields that describe progress
    are overwritten, which leaves live previews, the queue and task bookkeeping
    entirely the host's.

    Returns True if the wrapper is in place. A False here is not fatal: the
    job_count fallback in the script still removes the backwards jump, it simply
    cannot weight the phases by time.
    """
    global _installed

    if _installed:
        return True

    try:
        from modules import progress as host_progress
    except Exception:
        return False

    original = getattr(host_progress, "progressapi", None)
    if original is None:
        return False
    if getattr(original, "_model_chain_wrapped", False):
        _installed = True
        return True

    request_model = getattr(host_progress, "ProgressRequest", None)
    if request_model is None:
        return False

    def progressapi(req):
        response = original(req)
        try:
            _apply(response)
        except Exception:
            pass
        return response

    # FastAPI reads the annotation off the function to build the request body
    # model. This module uses postponed annotations, so a written-out annotation
    # would arrive as the string "ProgressRequest" and be resolved against this
    # module's globals, where it does not exist. Assigning the class itself
    # sidesteps the lookup entirely.
    progressapi.__annotations__ = {"req": request_model}
    progressapi.__name__ = "progressapi"
    progressapi._model_chain_wrapped = True

    host_progress.progressapi = progressapi
    _installed = True
    logger.debug("Model Chain: whole-job progress reporting is installed")
    return True


def installed() -> bool:
    return _installed


def _apply(response) -> None:
    """Overwrite progress, ETA and label on a response for a live chained job."""
    if not getattr(response, "active", False):
        return
    if not enabled():
        return
    if _stale():
        # A generation that raised between process() and postprocess() leaves
        # its plan behind. Describing the *next* task with the last one's phases
        # is worse than not describing it at all, so the leftover is dropped and
        # the host's own numbers stand.
        abandon()
        return

    snap = snapshot()
    if snap is None:
        return

    progress, eta, label = snap
    response.progress = progress
    response.eta = max(eta, 0.0)

    if label:
        # The host's JS prefixes textinfo to the percentage, but only when it
        # contains no newline -- a multi-line value is treated as a log message
        # and dropped. Keeping the label on one line is what makes it visible.
        response.textinfo = label.replace("\n", " ")
