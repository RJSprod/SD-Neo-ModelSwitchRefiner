"""The Krea roll, as reported on Forge's progress bar.

The complaint this answers is not subtle: a Creative Mode roll takes twenty
seconds on a mixed-placement model, all of it before the image job starts, and
none of it used to appear anywhere. So the assertions below are mostly about
*visibility* -- that a task is claimed, that the label changes as the phases do,
that the counters move while tokens stream, and above all that the bar is given
back afterwards, on every path out including the ones nobody plans for.

The last one is the reason this file exists rather than being three lines in
another. A claimed task that is never released leaves the host believing a job
is running: the next Generate draws a bar that never moves, and the only cure a
user can find is restarting the WebUI. Every failure mode below -- a refused
checkpoint, an empty prompt, a library that will not load, a model that throws
mid-stream, an Interrupt -- is checked for the release rather than only for its
own error.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import mc_broker
import mc_creative_krea
import mc_llm_paths
import mc_llm_progress
import mc_llm_sessions as sessions
import mc_progress


class FakeClient:
    """A writer that streams in pieces, so the counters have something to count."""

    def __init__(self, pieces=("A tall ", "white ", "lighthouse."), fail=None):
        self.pieces = list(pieces)
        self.fail = fail
        self.calls: list[dict] = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.85, top_p=0.95):
        self.calls.append({"messages": messages, "seed": seed})
        if self.fail is not None:
            raise self.fail
        produced = []
        for piece in self.pieces:
            if cancel is not None and cancel.is_set():
                break
            produced.append(piece)
            on_text(piece)
        return "".join(produced)


def _position():
    """The two counters a sampler drives, as the host would read them."""
    from modules import shared

    return (shared.state.sampling_step, shared.state.sampling_steps)


def watch_counters():
    """Sample the counters after every chunk the reporter is told about.

    Observed on the reporter rather than inside the fake client, because the
    client streams on a worker thread and the reporter is driven on the
    consumer's -- sampling from inside ``on_text`` would read the counters a
    moment before the chunk that moves them has arrived.
    """
    positions: list = []
    original = mc_llm_progress.reporter.wrote

    def wrote(text):
        original(text)
        positions.append(_position())

    mc_llm_progress.reporter.wrote = wrote
    return positions


def _label():
    from modules import shared

    return shared.state.textinfo


def _current_task():
    from modules import progress

    return progress.current_task


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(mc_progress, "path", lambda: str(tmp_path / mc_progress.FILENAME))
    mc_progress.forget()
    mc_progress.abandon()
    yield tmp_path
    mc_progress.abandon()


@pytest.fixture
def client(monkeypatch, host, store):
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
    fake = FakeClient()
    monkeypatch.setattr(sessions, "_client", lambda needs_vision=False, reserve=0: fake)
    monkeypatch.setattr(sessions, "_placement_notes", list)
    monkeypatch.setattr(mc_creative_krea, "checkpoint_objection", lambda: "")
    monkeypatch.setattr(mc_creative_krea, "_warm", lambda: True)
    mc_creative_krea.creative = mc_creative_krea.Creative()
    mc_llm_progress.reporter = mc_llm_progress.Reporter()
    yield fake
    mc_llm_progress.reporter.abandon()
    mc_creative_krea.creative = mc_creative_krea.Creative()
    mc_broker.clear()


def roll(task_id="task(krea)", source="car", creativity=10):
    stored = mc_creative_krea.settings()
    stored["creativity"] = creativity
    return list(mc_creative_krea.creative.roll(source, stored, task_id=task_id))


# --------------------------------------------------------------------------- #
# The bar appears
# --------------------------------------------------------------------------- #


class TestTheTaskIsClaimed:
    def test_a_roll_claims_the_task_the_browser_minted(self, client):
        seen = []

        original = mc_llm_progress.reporter.enter

        def watch(phase):
            seen.append(_current_task())
            original(phase)

        mc_llm_progress.reporter.enter = watch
        roll("task(abc)")

        assert "task(abc)" in seen

    def test_the_host_reports_it_as_the_running_task_while_it_runs(self, client):
        during = []
        client.pieces = ("one", "two")
        original = mc_llm_progress.reporter.wrote

        def watch(text):
            during.append(_current_task())
            original(text)

        mc_llm_progress.reporter.wrote = watch
        roll("task(abc)")

        assert during and all(task == "task(abc)" for task in during)

    def test_no_task_id_means_no_claim_and_no_complaint(self, client):
        """LLM Studio rolls with no bar arranged. The roll still runs; the
        reporter simply has nothing to report on."""
        events = roll(task_id="")

        assert events[-1].kind == sessions.DONE
        assert _current_task() is None

    def test_a_host_with_no_progress_module_is_not_fatal(self, client, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "modules.progress" or name.endswith("progress"):
                if name == "modules.progress":
                    raise ImportError("no progress module here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        events = roll("task(abc)")

        assert events[-1].kind == sessions.DONE


class TestTheBarNeverBlocksTheRunItDescribes:
    """The regression this class exists for, in one sentence: an indicator that
    tells the host a job is running makes the language model wait for a job that
    is itself.

    ``mc_broker.host_busy()`` is "``state.job`` or ``state.job_count`` is
    truthy", and ``mc_llm_sessions._Gpu.acquire`` refuses to start while it is
    true. A progress claim that set either field therefore hung the roll it was
    drawing a bar for -- intermittently, because any Gradio call finishing
    nearby clears both in its own ``finally``.
    """

    def test_the_host_is_not_busy_at_any_point_during_a_roll(self, client):
        seen = []
        original = mc_llm_progress.reporter.enter

        def watch(phase):
            original(phase)
            seen.append(mc_broker.host_busy())

        mc_llm_progress.reporter.enter = watch
        roll()

        assert seen and not any(seen)

    def test_the_reporter_never_touches_the_two_fields_that_mean_busy(self, client):
        from modules import shared

        shared.state.job = ""
        shared.state.job_count = 0
        reporter = mc_llm_progress.Reporter()
        reporter.begin("task(x)", 2000, warm=True)
        try:
            reporter.enter(mc_progress.PHASE_KREA_READ)
            reporter.enter(mc_progress.PHASE_KREA_WRITE)
            reporter.wrote("some text")

            assert shared.state.job == ""
            assert shared.state.job_count == 0
            assert mc_broker.host_busy() is False
        finally:
            reporter.end("some text")

        assert mc_broker.host_busy() is False

    def test_the_gpu_wait_is_not_entered_because_of_us(self, client):
        """The observable symptom was a console line saying the LLM was waiting
        for an image generation, on a machine generating nothing."""
        waited = [event.text for event in roll()
                  if event.kind == sessions.STATUS and "image generation" in event.text]

        assert waited == []

    def test_an_image_job_that_really_is_running_still_holds_the_llm_back(self, host):
        """The guard the deadlock fix must not have removed. host_busy() is
        still consulted and still says yes for a real generation; it simply no
        longer says yes because of a progress bar.

        Deliberately without the ``client`` fixture, which stubs host_busy out
        -- a test of host_busy that stubbed host_busy would assert nothing.
        """
        from modules import shared

        assert mc_broker.host_busy() is False
        shared.state.job_count = 1
        try:
            assert mc_broker.host_busy() is True
        finally:
            shared.state.job_count = 0
        shared.state.job = "txt2img"
        try:
            assert mc_broker.host_busy() is True
        finally:
            shared.state.job = ""

    def test_a_roll_with_no_generation_around_it_never_makes_the_host_look_busy(
            self, monkeypatch, host, store):
        """End to end, with ``host_busy`` *not* stubbed.

        Every other test in this file replaces it, which is right for what they
        are each about and useless for this one: the deadlock was a real call to
        a real ``host_busy`` returning a real True. So this drives a real roll
        through the actual ``_Gpu.acquire`` and asserts the answer was False at
        every frame -- because nothing was generating, and a progress bar is not
        a generation.
        """
        mc_broker.clear()
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False, reserve=0: FakeClient())
        monkeypatch.setattr(sessions, "_placement_notes", list)
        monkeypatch.setattr(mc_creative_krea, "checkpoint_objection", lambda: "")
        mc_creative_krea.creative = mc_creative_krea.Creative()
        mc_llm_progress.reporter = mc_llm_progress.Reporter()

        stored = mc_creative_krea.settings()
        busy = []
        try:
            events = []
            for event in mc_creative_krea.creative.roll("car", stored,
                                                        task_id="task(abc)"):
                busy.append(mc_broker.host_busy())
                events.append(event)
        finally:
            mc_broker.clear()

        assert not any(busy)
        assert events[-1].kind == sessions.DONE

    def test_the_source_says_why(self):
        """A comment, asserted, because the two lines removed here are exactly
        the two lines somebody adds back for completeness."""
        source = Path(mc_llm_progress.__file__).read_text(encoding="utf-8")

        assert "host_busy()" in source
        assert "state.job" in source


# --------------------------------------------------------------------------- #
# The bar says what is happening
# --------------------------------------------------------------------------- #


class TestThePhases:
    def test_it_names_the_three_spans_in_order(self, client):
        seen = []
        original = mc_llm_progress.reporter.enter

        def watch(phase):
            original(phase)
            if mc_llm_progress.reporter.task and phase not in seen:
                seen.append(phase)

        mc_llm_progress.reporter.enter = watch
        roll()

        assert seen == [mc_progress.PHASE_KREA_WAIT,
                        mc_progress.PHASE_KREA_READ,
                        mc_progress.PHASE_KREA_WRITE]

    def test_the_label_the_user_reads_changes_with_them(self, client):
        labels = []
        original = mc_llm_progress.reporter.enter

        def watch(phase):
            original(phase)
            if _label() and _label() not in labels:
                labels.append(_label())

        mc_llm_progress.reporter.enter = watch
        roll()

        assert labels == [mc_llm_progress.WAITING, mc_llm_progress.READING,
                          mc_llm_progress.WRITING]

    def test_reading_begins_when_the_request_goes_out(self, client):
        """The session says "Writing the Krea prompt…" immediately before it
        calls the client, and that is exactly when prompt evaluation starts --
        the phase that reports nothing and takes ten seconds."""
        from prompt_master.krea import enhancer

        assert mc_creative_krea._phase_for(
            f"{enhancer.label(0)}…") == mc_progress.PHASE_KREA_READ
        assert mc_creative_krea._phase_for(
            "Starting llama-server…") == mc_progress.PHASE_KREA_WAIT
        assert mc_creative_krea._phase_for(
            "Waiting for image generation…") == mc_progress.PHASE_KREA_WAIT

    def test_writing_begins_on_the_first_streamed_chunk(self, client):
        """The one boundary that is observable from this side of the wire."""
        positions = watch_counters()
        roll()

        # The counters are live from the first chunk onwards: a real total, and
        # a count that is no longer zero.
        assert positions
        assert all(total > 0 for _done, total in positions)
        assert positions[-1][0] > 0


class TestTheCounters:
    def test_they_climb_as_the_reply_streams(self, client):
        client.pieces = ("a" * 50, "b" * 50, "c" * 50)
        positions = watch_counters()
        roll()

        counted = [done for done, _total in positions]
        assert counted == sorted(counted)
        assert counted[-1] > counted[0]

    def test_the_total_stretches_rather_than_the_bar_pinning_at_full(self, client):
        """A reply that outruns the estimate should slow the bar down, not park
        it at 100% for the rest of the run."""
        client.pieces = tuple("x" * 400 for _ in range(6))
        positions = watch_counters()
        roll()

        assert all(done <= total for done, total in positions)
        assert positions[-1][1] > positions[0][1]

    def test_they_are_cleared_when_the_roll_ends(self, client):
        roll()

        assert _position() == (0, 0)
        assert _label() is None


class TestTheEstimate:
    def test_a_longer_prompt_is_predicted_to_take_longer_to_read(self, client):
        """The whole explanation for Creative Mode's delay, as arithmetic: the
        brief is what the model has to read before it can say anything."""
        short = mc_creative_krea._prompt_size("car", (), "")
        long = mc_creative_krea._prompt_size("car", (), "x" * 3000)

        assert long > short + 2900

    def test_the_reply_length_is_learned_from_what_came_back(self, client):
        client.pieces = ("y" * 900,)
        roll()

        assert mc_progress.measured("krea:reply") == pytest.approx(900.0)

    def test_an_unusually_short_reply_cannot_shrink_the_estimate_to_nothing(
            self, client):
        """Otherwise one terse answer teaches the bar to expect eighty
        characters and every normal roll sits at 99%."""
        mc_progress.learn("krea:reply", 10.0)
        reporter = mc_llm_progress.Reporter()
        reporter.begin("task(x)", 1000, warm=True)
        try:
            reporter.enter(mc_progress.PHASE_KREA_WRITE)
            _done, total = _position()
        finally:
            reporter.abandon()

        assert total >= mc_llm_progress.MINIMUM_REPLY

    def test_a_cold_start_is_predicted_differently_from_a_warm_one(self):
        assert mc_progress.BASELINES["krea:wait:cold"] > (
            mc_progress.BASELINES["krea:wait:warm"] * 10)

    def test_the_prediction_self_corrects_across_rolls(self, client, store):
        client.pieces = ("z" * 800,)
        roll()
        first = mc_progress.measured("krea:read")
        roll()

        assert first > 0
        assert mc_progress.measured("krea:read") > 0


# --------------------------------------------------------------------------- #
# The bar is always given back
# --------------------------------------------------------------------------- #


class TestTheTaskIsAlwaysReleased:
    """A claimed task that is never released leaves the host believing a job is
    running, and the next Generate draws a bar that never moves."""

    def test_after_a_successful_roll(self, client):
        roll()

        assert _current_task() is None
        assert mc_llm_progress.reporter.task is None

    def test_after_the_model_throws(self, client):
        client.fail = RuntimeError("llama-server went away")
        events = roll()

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_after_an_empty_reply(self, client):
        client.pieces = ("",)
        events = roll()

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_after_a_refused_checkpoint(self, client, monkeypatch):
        monkeypatch.setattr(mc_creative_krea, "checkpoint_objection",
                            lambda: "Select a Krea 2 checkpoint.")
        stored = mc_creative_krea.settings()
        events = list(mc_creative_krea.creative.roll(
            "car", stored, guard_checkpoint=True, task_id="task(abc)"))

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_after_an_empty_source_prompt(self, client):
        events = roll(source="   ")

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_after_the_director_fails(self, client, monkeypatch):
        from prompt_master.krea import director

        def explode(*args, **kwargs):
            raise RuntimeError("no package")

        monkeypatch.setattr(director, "roll", explode)
        events = roll()

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_a_failed_roll_teaches_the_store_nothing_about_the_answer(self, client):
        """A run that gave up says nothing about how long an answer takes, so
        neither the writing rate nor the expected reply length may learn from
        it. What did complete -- the wait -- is real and is allowed to."""
        client.fail = RuntimeError("nope")
        roll()

        learned = mc_progress.rates()
        assert "krea:write" not in learned
        assert "krea:reply" not in learned


class TestInterrupt:
    def test_pressing_interrupt_stops_the_roll(self, client):
        """The bar the host draws carries Interrupt whether or not anything is
        listening, and a button that does nothing is worse than no button."""
        from modules import shared

        def press(text):
            shared.state.interrupted = True

        client.watching = None
        original = mc_llm_progress.reporter.wrote
        mc_llm_progress.reporter.wrote = lambda text: (original(text), press(text))[0]
        events = roll()

        assert events[-1].kind == sessions.CANCELLED
        assert _current_task() is None

    def test_the_flag_is_cleared_so_the_image_job_does_not_inherit_it(self, client):
        from modules import shared

        original = mc_llm_progress.reporter.wrote
        mc_llm_progress.reporter.wrote = lambda text: (
            original(text), setattr(shared.state, "interrupted", True))[0]
        roll()

        assert shared.state.interrupted is False
        assert shared.state.skipped is False

    def test_an_interrupted_roll_records_no_result(self, client):
        """``last`` is what the diagnostics drawer shows and what any later
        reader would take for this roll's output. A run that was stopped has
        none, and must not leave the previous one looking like it."""
        from modules import shared

        original = mc_llm_progress.reporter.wrote
        mc_llm_progress.reporter.wrote = lambda text: (
            original(text), setattr(shared.state, "interrupted", True))[0]
        roll()

        assert mc_creative_krea.creative.last is None

    def test_an_untouched_run_is_never_treated_as_interrupted(self, client):
        events = roll()

        assert events[-1].kind == sessions.DONE


# --------------------------------------------------------------------------- #
# Nothing is left behind for the image job
# --------------------------------------------------------------------------- #


class TestItDoesNotDisturbTheGenerationItPrecedes:
    def test_the_progress_plan_is_finished_before_generate_is_clicked(self, client):
        roll()

        assert mc_progress.active() is False

    def test_a_failed_roll_leaves_no_plan_describing_the_next_job(self, client):
        client.fail = RuntimeError("nope")
        roll()

        assert mc_progress.active() is False

    def test_the_job_counters_are_left_clean(self, client):
        from modules import shared

        roll()

        assert shared.state.sampling_step == 0
        assert shared.state.sampling_steps == 0

    def test_two_rolls_in_a_row_each_claim_and_release(self, client):
        roll("task(one)")
        assert _current_task() is None
        roll("task(two)")
        assert _current_task() is None

    def test_a_reused_task_id_still_reports_as_active(self, client):
        """The host keeps a list of finished tasks and reports membership even
        on an active response, so a browser that reused an id would have its bar
        torn down on the first poll. Two rolls, same id, is the shape that
        catches it."""
        from modules import progress

        roll("task(same)")
        roll("task(same)")

        assert "task(same)" in progress.finished_tasks
        assert progress.current_task is None


# --------------------------------------------------------------------------- #
# The bar the roll borrows
# --------------------------------------------------------------------------- #


class TestABorrowedBar:
    """Creative Mode's txt2img roll runs inside the generation, so the bar it
    reports on belongs to that generation and is still needed after the roll.

    Every assertion here is a variation on one line: ``finish_task`` must not be
    called. The host serves ``finished_tasks`` membership even on an otherwise
    active response, so finishing the image job's task from inside
    ``before_process`` tells every poller that the generation is over -- the bar
    is torn down, the gallery stops waiting, and the image arrives into a page
    that has stopped listening for it.
    """

    def borrowed(self, source="car", task_id=""):
        stored = mc_creative_krea.settings()
        return list(mc_creative_krea.creative.roll(source, stored, task_id=task_id,
                                                   own_bar=False))

    def test_it_reports_the_phases_without_a_task_id_of_its_own(self, client):
        seen = []
        original = mc_llm_progress.reporter.enter

        def watch(phase):
            original(phase)
            seen.append(_label())

        mc_llm_progress.reporter.enter = watch
        events = self.borrowed()

        assert events[-1].kind == sessions.DONE
        assert mc_llm_progress.WRITING in seen

    def test_it_never_starts_a_task(self, client):
        from modules import progress

        during = []
        original = mc_llm_progress.reporter.wrote
        mc_llm_progress.reporter.wrote = lambda text: (
            original(text), during.append(progress.current_task))[0]
        self.borrowed()

        assert during and all(task is None for task in during)

    def test_it_never_finishes_the_generation_task(self, client):
        """The one line that must not run. The host's own task is set here the
        way ``modules.call_queue`` sets it around every GPU call."""
        from modules import progress

        progress.add_task_to_queue("task(the-image-job)")
        progress.start_task("task(the-image-job)")
        try:
            self.borrowed()

            assert progress.current_task == "task(the-image-job)"
            assert "task(the-image-job)" not in progress.finished_tasks
        finally:
            progress.finish_task("task(the-image-job)")

    def test_it_hands_the_counters_back_for_the_sampler(self, client):
        """Not the task, but very much the counters: leaving them at a finished
        roll's values shows the generation as complete until the sampler
        overwrites them."""
        from modules import shared

        self.borrowed()

        assert (shared.state.sampling_step, shared.state.sampling_steps) == (0, 0)
        assert mc_llm_progress.reporter.task is None

    def test_a_failed_roll_hands_them_back_too(self, client):
        from modules import shared

        client.fail = RuntimeError("llama-server went away")
        events = self.borrowed()

        assert events[-1].kind == sessions.FAILED
        assert (shared.state.sampling_step, shared.state.sampling_steps) == (0, 0)
        assert mc_progress.active() is False

    def test_interrupt_is_left_set_so_the_generation_stops_too(self, client):
        """Opposite to the owned-bar case, and for the same reason. On an owned
        bar the roll is all that is running and the flag would leak into a later
        press; on a borrowed one the roll is the first part of a generation that
        is already running, and Interrupt means stop that generation."""
        from modules import shared

        original = mc_llm_progress.reporter.wrote
        mc_llm_progress.reporter.wrote = lambda text: (
            original(text), setattr(shared.state, "interrupted", True))[0]
        try:
            events = self.borrowed()
        finally:
            interrupted = shared.state.interrupted
            shared.state.interrupted = False

        assert events[-1].kind == sessions.CANCELLED
        assert interrupted is True


class TestTheRollRunsInsideTheGeneration:
    """The deadlock, driven rather than reasoned about.

    ``mc_llm_sessions._Gpu.acquire`` refuses to start an LLM turn while
    ``host_busy()`` is true, and inside ``before_process`` it is true -- the host
    set ``state.job`` before calling the hook. A roll that waited for it would
    wait for the generation that is waiting for the roll, for ever. Nothing here
    stubs ``host_busy``: the point is that it really does return True.
    """

    @pytest.fixture
    def running(self, monkeypatch, host, store):
        """A host in the middle of a job, the way call_queue leaves it.

        Everything the ``client`` fixture arranges except the one thing this
        class is about: ``host_busy`` is the real function here, and it is
        answering about a job that really is running.
        """
        from modules import shared

        mc_broker.clear()
        fake = FakeClient()
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False, reserve=0: fake)
        monkeypatch.setattr(sessions, "_placement_notes", list)
        monkeypatch.setattr(mc_creative_krea, "checkpoint_objection", lambda: "")
        monkeypatch.setattr(mc_creative_krea, "_warm", lambda: True)
        mc_creative_krea.creative = mc_creative_krea.Creative()
        mc_llm_progress.reporter = mc_llm_progress.Reporter()

        shared.state.job = "txt2img"
        shared.state.job_count = 1
        yield fake
        shared.state.job = ""
        shared.state.job_count = 0
        mc_llm_progress.reporter.abandon()
        mc_creative_krea.creative = mc_creative_krea.Creative()
        mc_broker.clear()

    def press(self, timeout=20.0, prompt="car"):
        """One press of Generate, on a thread with a deadline.

        A deadlocked hook does not fail a test, it hangs the run -- so the
        deadline is the assertion.
        """
        import threading

        import model_chain_krea_creative as creative_script
        from prompt_master.krea import director
        from prompt_master.krea import library as library_module

        script = creative_script.ScriptKreaCreative()
        values = [10, director.RANDOM_SEED, True, ""]
        for _key in library_module.library().axis_keys:
            values.extend([director.VARY, None, []])

        class Processing:
            def __init__(self):
                self.prompt = prompt
                self.extra_generation_params = {}

        p = Processing()
        error: list[BaseException] = []

        def run():
            try:
                script.before_process(p, True, *values)
            except BaseException as exc:  # surfaced on the calling thread below
                error.append(exc)

        worker = threading.Thread(target=run, name="press-generate", daemon=True)
        worker.start()
        worker.join(timeout)

        assert not worker.is_alive(), "before_process never returned: the roll deadlocked"
        if error:
            raise error[0]
        return p

    def test_the_host_really_is_busy_while_the_hook_runs(self, running):
        assert mc_broker.host_busy() is True

    def test_the_roll_completes_anyway(self, running):
        p = self.press()

        assert p.prompt != "car"
        assert running.calls

    def test_it_never_says_it_is_waiting_for_the_image_generation(self, running):
        """The symptom the old arrangement produced if the roll was ever moved
        here: a console line saying the LLM is waiting for a generation, on the
        thread that generation is blocked on."""
        said = []
        original = sessions._Gpu._announce

        def watch(self, started, announced, what):
            said.append(what)
            return (yield from original(self, started, announced, what))

        sessions._Gpu._announce = watch
        try:
            self.press()
        finally:
            sessions._Gpu._announce = original

        assert "image generation" not in said

    def test_the_permission_is_scoped_to_the_hook(self, running):
        """It is granted for the roll and not for the session. An LLM turn
        started from anywhere else while a generation runs still waits, which is
        the guard the whole broker exists to provide."""
        self.press()

        assert mc_broker.inside_host_job() is False

    def test_it_is_scoped_to_the_thread_as_well(self, running):
        """Declared per thread, so a worker started inside the block does not
        inherit a permission nothing is blocked waiting for."""
        import threading

        seen = []
        with mc_broker.host_job():
            worker = threading.Thread(target=lambda: seen.append(
                mc_broker.inside_host_job()))
            worker.start()
            worker.join()

            assert mc_broker.inside_host_job() is True

        assert seen == [False]
        assert mc_broker.inside_host_job() is False

    def test_the_workload_lock_is_still_given_back(self, running):
        self.press()

        assert mc_broker.active() is None
