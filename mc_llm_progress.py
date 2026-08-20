"""The Krea roll, on Forge's own progress bar.

A Creative Mode roll takes twenty seconds on a mixed-placement 26B model, and
until this module existed it took them behind a UI that showed nothing
whatsoever. That is the worst possible way to spend twenty seconds: a button
that has visibly done nothing is a button people press again.

What this module does is small and deliberately unoriginal -- it makes the roll
look like every other long job the host runs, by using the host's own machinery
rather than drawing something of its own:

    a task the bar is drawn and polled for
        -> shared.state carries the label and the token counters
        -> mc_progress supplies the fraction and the ETA, as it does for a chain
        -> the counters are handed back

Owned bars and borrowed ones
----------------------------
There are two kinds of caller and the difference is one line of behaviour.

Creative Mode's txt2img roll runs *inside* the image generation, from
``before_process``, so the host has already started a task and the browser is
already polling its bar. That roll **borrows** it: ``begin(claim=False)``, no
``start_task``, no ``finish_task``, and Interrupt means stop the generation
rather than stop the roll. Nothing about the bar is this module's to end,
because the generation it describes has not finished.

A caller with no generation around it -- LLM Studio, or anything that wants a
bar of its own -- **owns** its task: ``start_task`` on the way in,
``finish_task`` on the way out, and an Interrupt that stops the roll and goes no
further.

Nothing here is a second progress implementation. The bar, the polling, the
Interrupt button and the arithmetic are all the host's and this extension's
existing ones; what is added is the three lines that say a language model is
what is currently running.

Why the phases are the ones they are
------------------------------------
Because they are the ones llama.cpp's own log distinguishes, and the ones the
user feels as different kinds of waiting:

*Waiting* is the GPU handover, and on a cold start twenty seconds of
llama-server reading weights off a disk. It reports nothing.

*Reading* is prompt evaluation. It also reports nothing -- not one byte comes
back while it runs -- and it is the phase that got dramatically longer when
Creative Mode arrived, because the creative brief is several hundred tokens that
change every roll and therefore cannot be cached. It is proportional to the
length of what was sent, which is a number this module knows before the request
goes out.

*Writing* is generation. It streams, so it is the only phase that can honestly
say how far through it is, and it says so by driving the same two counters a
sampler drives: ``sampling_step`` and ``sampling_steps``.

Honesty
-------
The prediction is time-proportional and self-correcting, because it is
``mc_progress``'s and that is what that module is for. The first roll on a fresh
install runs on a coarse built-in guess; every roll after that runs on what this
machine actually measured, smoothed. Characters stand in for tokens throughout,
which is a constant-factor error the store folds in on the first measurement and
never sees again.
"""

from __future__ import annotations

import logging
import threading
import time

import mc_progress

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

WAITING = "Waiting for the language model"
READING = "Reading the prompt"
WRITING = "Writing the Krea prompt"

# What one roll's phases are called on the bar. Deliberately plain: the bar is
# read at a glance by somebody wondering whether anything is happening, and
# "Reading the prompt" answers that where "Krea expansion phase 2" does not.
_LABELS = {
    mc_progress.PHASE_KREA_WAIT: WAITING,
    mc_progress.PHASE_KREA_READ: READING,
    mc_progress.PHASE_KREA_WRITE: WRITING,
}

ADOPTED = "(the host's own bar)"
"""Stands in for a task id when the roll is borrowing a bar rather than owning one.

A borrowed bar has no task of this module's to name -- the id belongs to the
image generation the roll is running inside, and this module must not touch it.
Something non-empty is still needed, because every method below is a no-op while
no roll is in progress and "no roll in progress" is spelled ``_task is None``.
"""

MINIMUM_REPLY = 200.0
"""Shortest expected reply, in characters, however short the last one was.

The learned average is what sizes the writing phase, and a run of unusually
terse prompts would otherwise teach the bar to expect eighty characters and
then sit at 99% for the whole of a normal one. A floor costs an early roll a
slightly pessimistic bar and buys every later one a bar that still has
somewhere to go.
"""

HEADROOM = 1.35
"""How far past the expected reply the counter is allowed to stretch.

The denominator grows when a reply outruns the estimate, rather than the bar
pinning at full and stalling there. ``mc_progress`` already refuses to let the
fraction go backwards, so growing the denominator slows the bar down instead of
rewinding it -- which is what a long answer should look like.
"""


class Reporter:
    """One roll's progress, told to the host as it happens.

    Every method is a no-op when nothing was claimed, so the Creative Mode gate
    can drive one of these unconditionally and a host without a progress module
    -- or a user who turned the whole thing off -- costs nothing but a few
    attribute lookups.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._task: str | None = None
        self._claimed = False
        self._written = 0
        self._expected = 0.0
        self._phase = ""

    # -- lifecycle --------------------------------------------------------- #

    @property
    def task(self) -> str | None:
        with self._lock:
            return self._task

    def begin(self, task_id, prompt_characters: int, warm: bool = None,
              claim: bool = True) -> bool:
        """Plan the roll and put it on a bar. Returns whether the bar will show.

        ``prompt_characters`` is everything the model is about to read -- Krea's
        instruction, the user's line and the creative brief -- because that is
        what prompt evaluation is proportional to, and it is the whole reason a
        Creativity-10 roll takes three times as long to start as a Creativity-2
        one.

        ``warm`` says whether llama-server is already up. It picks which of two
        waiting rates to predict from, because the wait is bimodal: a warm
        server is a lock acquisition and a cold one is twenty seconds of reading
        weights off a disk. The caller knows; this module would have to import
        the runtime to find out.

        ``claim`` says whether this roll owns the host task or is borrowing one.
        A roll that runs *inside* an image generation -- which is where Creative
        Mode's roll runs -- is described by the bar the host already started for
        that generation, so it must neither start a task nor finish one:
        ``finish_task`` on somebody else's task tells the browser the image job
        ended, and the browser believes it. Borrowed or owned, everything else
        below is identical; only the two lines that talk to
        ``modules.progress`` are skipped.
        """
        task_id = str(task_id or "").strip() or (ADOPTED if not claim else "")
        if not task_id:
            return False

        expected = max(mc_progress.measured("krea:reply", 700.0), MINIMUM_REPLY)
        waiting = ("krea:wait",) if warm is None else (
            ("krea:wait:warm",) if warm else ("krea:wait:cold",)) + ("krea:wait",)

        job = mc_progress.new_job()
        job.add(mc_progress.PHASE_KREA_WAIT, WAITING, rate_keys=waiting, units=1.0)
        job.add(mc_progress.PHASE_KREA_READ, READING, rate_keys=("krea:read",),
                units=max(int(prompt_characters), 1))
        job.add(mc_progress.PHASE_KREA_WRITE, WRITING, rate_keys=("krea:write",),
                units=expected, weights=(expected,))

        if claim and not self._claim(task_id):
            return False

        with self._lock:
            self._task = task_id
            self._claimed = bool(claim)
            self._written = 0
            self._expected = expected
            self._phase = ""

        mc_progress.begin(job)
        self.enter(mc_progress.PHASE_KREA_WAIT)
        logger.info("Model Chain: Krea roll — %s characters to read, expecting about "
                    "%.0f back, predicted %.1fs", f"{int(prompt_characters):,}",
                    expected, job.estimate)
        return True

    def enter(self, phase: str) -> None:
        """Move to a phase, and say so on the bar."""
        with self._lock:
            if self._task is None or phase == self._phase:
                return
            self._phase = phase
        mc_progress.enter(phase)
        self._say(_LABELS.get(phase, ""))
        if phase == mc_progress.PHASE_KREA_WRITE:
            # The writing phase is the one that can report itself, so this is
            # where the counters start meaning something. Set before the first
            # chunk is counted so a fast reply cannot arrive ahead of them.
            self._counters(0, self._expected)
            mc_progress.note_pass()

    def wrote(self, text: str) -> None:
        """Count what has streamed back so far.

        Characters, not tokens, and compared against a learned expectation of
        the same kind. The denominator stretches when a reply outruns it so the
        bar slows rather than stalling at full.
        """
        with self._lock:
            if self._task is None:
                return
            self._written += len(text or "")
            written = self._written
            expected = max(self._expected, written * HEADROOM)
            self._expected = expected
        self._counters(written, expected)

    def end(self, reply: str = "") -> None:
        """Finish the roll, release the bar, and record what it cost.

        The reply's length is learned here rather than in the phase's own
        close, because it is a property of the answer rather than of the time
        it took -- and it is what sizes the writing phase of the *next* roll.
        """
        with self._lock:
            task, self._task = self._task, None
            claimed, self._claimed = self._claimed, False
            self._phase = ""
        if task is None:
            return

        length = len(reply or "")
        if length > 0:
            mc_progress.learn("krea:reply", float(length))
        mc_progress.end()
        self._release(task, claimed)

    def abandon(self) -> None:
        """Give the bar back without recording anything.

        For a roll that failed or was interrupted. A cancelled run's timings
        describe how long it took to give up, and folding that into the store
        would teach the bar that rolls are quick.
        """
        with self._lock:
            task, self._task = self._task, None
            claimed, self._claimed = self._claimed, False
            self._phase = ""
        if task is None:
            return
        mc_progress.abandon()
        self._release(task, claimed)

    # -- what the user can do to it ---------------------------------------- #

    def interrupted(self) -> bool:
        """Whether the Interrupt button has been pressed since the roll began.

        Asked once per streamed chunk by the caller. The bar the host draws
        carries Interrupt and Skip whether or not anything is listening, and a
        button that does nothing is worse than no button -- so the roll listens.

        What happens to the flag afterwards depends on whose bar this is, and
        the two answers are opposite for the same reason. On an owned bar the
        roll is the whole of what is running, and the image generation that
        follows it is a separate press: the flag is cleared, so that generation
        does not inherit a stop nobody meant for it. On a borrowed bar the roll
        is the first part of a generation that is already running, and Interrupt
        during it means *stop this generation* -- so the flag is left exactly as
        the user set it, and the host's own processing loop reads it a moment
        later and stops.
        """
        with self._lock:
            if self._task is None:
                return False
            claimed = self._claimed
        try:
            from modules import shared

            if not (getattr(shared.state, "interrupted", False)
                    or getattr(shared.state, "stopping_generation", False)
                    or getattr(shared.state, "skipped", False)):
                return False
            if claimed:
                shared.state.interrupted = False
                shared.state.stopping_generation = False
                shared.state.skipped = False
        except Exception:
            return False
        return True

    # -- the host --------------------------------------------------------- #

    def _claim(self, task_id: str) -> bool:
        """Tell the host this task is running, so its progress endpoint reports it.

        ``add_task_to_queue`` then ``start_task`` is the pair
        ``modules.call_queue`` uses around every ordinary Gradio GPU call. Using
        the same two functions is what makes the roll indistinguishable from a
        native job as far as the bar is concerned, and means nothing here has to
        know how progress is computed or drawn.

        **``shared.state.job`` and ``shared.state.job_count`` are deliberately
        not set, and must never be.** ``mc_broker.host_busy()`` is exactly
        "either of those is truthy", and the LLM run this reporter describes
        waits for ``host_busy()`` to go false before it will start. Setting them
        makes the roll wait for an image generation that is itself -- the
        deadlock this whole architecture is arranged to avoid, reintroduced
        through the back door by a progress indicator. It is intermittent rather
        than reliable, because any Gradio call that finishes nearby clears both
        fields in its own ``finally``, which is what makes it so unpleasant to
        diagnose: it hangs on a fresh restart and not on the next attempt.

        The cost is one line of the host's own arithmetic, which needs
        ``job_count`` to compute a fraction. Nothing is lost while this
        extension's whole-job reporting is on -- ``mc_progress`` overwrites that
        fraction anyway -- and with it switched off the bar shows the phase name
        without moving, which is a smaller loss than a hang by any measure.
        """
        try:
            from modules import progress as host_progress
            from modules import shared

            host_progress.add_task_to_queue(task_id)
            host_progress.start_task(task_id)
            shared.state.sampling_step = 0
            shared.state.sampling_steps = 0
            shared.state.interrupted = False
            shared.state.skipped = False
            # The host's progress endpoint divides by this. It is None until
            # the first generation of the session, so a roll made straight after
            # a restart would otherwise take the endpoint down with a
            # TypeError -- and the browser polls that endpoint four times a
            # second.
            if not getattr(shared.state, "time_start", None):
                shared.state.time_start = time.time()
        except Exception:
            logger.debug("Model Chain: the host progress endpoint could not be claimed",
                         exc_info=True)
            return False
        return True

    def _release(self, task_id: str, claimed: bool = True) -> None:
        """Hand the bar back: always the counters, only sometimes the task.

        The counters are this reporter's own writing whichever kind of bar it
        was drawing on, and leaving them at a finished roll's values would show
        the image generation that follows as already complete until its sampler
        overwrites them. So they are always cleared.

        ``finish_task`` is different, and is exactly the line that must not run
        on a borrowed bar. On an owned task it makes the roll's bar disappear;
        on the host's own task it tells the browser -- and anything else polling
        the progress endpoint -- that the image generation has finished, in the
        middle of ``before_process``.
        """
        try:
            from modules import shared

            shared.state.sampling_step = 0
            shared.state.sampling_steps = 0
            shared.state.textinfo = None
        except Exception:
            logger.debug("Model Chain: the roll's progress counters could not be cleared",
                         exc_info=True)
        if not claimed:
            return
        try:
            from modules import progress as host_progress

            # job and job_count are not cleared here because they are never set
            # -- see _claim. Clearing a field this module does not own would be
            # this reporter tidying away somebody else's running generation.
            host_progress.finish_task(task_id)
        except Exception:
            logger.debug("Model Chain: the host progress endpoint could not be released",
                         exc_info=True)

    def _say(self, text: str) -> None:
        if not text:
            return
        try:
            from modules import shared

            shared.state.textinfo = text
        except Exception:
            pass

    def _counters(self, done, total) -> None:
        """Drive the two counters a sampler drives, with characters in them.

        ``mc_progress`` reads exactly these to interpolate inside a phase, and
        so does the host's own arithmetic when this extension's whole-job
        reporting is switched off. Reusing them means the writing phase is
        described by one mechanism rather than by a special case.
        """
        try:
            from modules import shared

            shared.state.sampling_steps = max(int(total), 1)
            shared.state.sampling_step = max(min(int(done), int(total)), 0)
        except Exception:
            pass


reporter = Reporter()
"""The one reporter.

A module singleton for the same reason the Creative session is one: there is a
single Generate button and a single host progress endpoint, and two rolls cannot
be in flight at once -- the browser gate refuses the second press.
"""
