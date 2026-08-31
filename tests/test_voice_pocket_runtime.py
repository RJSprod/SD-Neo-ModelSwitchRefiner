"""The PocketTTS runtime: its handshake, its turns, and the drain it owns.

The parent side of the third engine, driven against a real subprocess that
speaks the real protocol and holds no tensors. What is asserted here is the
half of the interruption contract the worker cannot assert on its own:

    a Stop returns at once, because the browser is already silent and a Stop
        that blocked for the length of a generation would feel broken;
    no PCM produced after it reaches playback;
    the drain record survives until the *worker* says its lane is free, and is
        cleared by that frame and by nothing else;
    a new inference does not overlap the old one;
    and only the newest pending turn survives a busy engine, rather than a
        queue of stale replies building up behind it.

Those are I-PKT-11, I-PKT-12, I-PKT-13 and section 21.5, and none of them can
be checked by reading the code -- a drain that deadlocks looks exactly like a
drain that works until something waits on it.

Nothing here imports Torch or PocketTTS. The fake worker in ``conftest`` speaks
the framing and arranges its own parent-death containment, which is what makes
these tests about the boundary rather than about having a PyTorch closure.
"""

from __future__ import annotations

import time

import pytest

import mc_voice_engines as engines
import mc_voice_pocket_runtime as runtime


class Turn:
    """The half of :class:`mc_voice_turn.VoiceTurn` this runtime touches.

    A double rather than the real thing, because these tests are about what the
    *runtime* does with a turn -- and a real turn would bring a segmenter, a
    pump thread and a bounded queue that would all have to be driven to reach
    the two lines being checked.
    """

    def __init__(self, identifier="T1"):
        import threading

        self.id = identifier
        self.sample_rate = 0
        self.streaming = ""
        self.cancelled = threading.Event()
        self.audio = []
        self.drained = 0
        self.segments = []
        self.finished = False
        self.error = ""
        self.draining = False
        self.ready_at = 0.0
        self.unit = {"chars": None, "audio_ms": None}

    def offer_audio(self, pcm, rate=0):
        if self.cancelled.is_set():
            self.drained += 1
            return False
        self.sample_rate = self.sample_rate or int(rate or 0)
        self.audio.append(pcm)
        return True

    def note_segment(self, **values):
        self.segments.append(values)

    def audio_finished(self):
        self.finished = True

    def audio_failed(self, reason):
        self.error = str(reason)

    def cancel(self, reason="user"):
        self.cancelled.set()
        return True

    def drain_audio(self):
        pass

    def interrupting(self, chars=None, audio_ms=None):
        self.draining = True
        if chars is not None:
            self.unit["chars"] = int(chars)
        if audio_ms is not None:
            self.unit["audio_ms"] = int(audio_ms)

    def interrupted(self):
        self.draining = False
        self.ready_at = time.monotonic()


def wait_until(condition, timeout=5.0, what="the condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    raise AssertionError(f"{what} never became true")


# --------------------------------------------------------------------------- #
# Starting
# --------------------------------------------------------------------------- #


class TestTheHandshakeRefusesWhatThisBuildCannotAccept:
    """T-PKT-RT-1, T-PKT-RT-9, and section 18's refusal list."""

    def test_a_good_worker_starts_and_reports_what_it_is(self, host, fake_pocket_worker):
        runtime.ensure_started()
        found = runtime.engine()
        assert found["loaded"] is True
        assert found["device"] == "cpu"
        assert found["interrupt_mode"] == "drain_unit"
        assert runtime.declared_interrupt_mode() == "drain_unit"

    @pytest.mark.pocket(protocol=99)
    def test_a_protocol_this_build_does_not_speak_is_refused(self, host,
                                                             fake_pocket_worker):
        with pytest.raises(runtime.PocketRuntimeError) as raised:
            runtime.ensure_started()
        assert "protocol" in str(raised.value)

    @pytest.mark.pocket(device="cuda", provider="cuda")
    def test_a_worker_that_found_a_graphics_device_is_refused(self, host,
                                                              fake_pocket_worker):
        """I-PKT-7. This release supports PocketTTS on the CPU only, and a
        worker that claimed VRAM an image generation was counting on is worse
        than no speech."""
        with pytest.raises(runtime.PocketRuntimeError) as raised:
            runtime.ensure_started()
        assert "CPU only" in str(raised.value)

    @pytest.mark.pocket(engine="something-else")
    def test_a_worker_that_is_not_pocket_is_refused(self, host, fake_pocket_worker):
        with pytest.raises(runtime.PocketRuntimeError) as raised:
            runtime.ensure_started()
        assert "identify itself" in str(raised.value)

    @pytest.mark.pocket(handshake={"model_fingerprint": "somethingelse"})
    def test_a_different_model_from_the_installed_one_is_refused(self, host,
                                                                 fake_pocket_worker):
        """I-PKT-18. A saved voice state loaded into a model it was not prepared
        for is a voice that sounds wrong for a reason nobody can find."""
        with pytest.raises(runtime.PocketRuntimeError) as raised:
            runtime.ensure_started()
        assert "different model" in str(raised.value)

    @pytest.mark.pocket(interrupt_mode="magic")
    def test_an_interrupt_mode_this_parent_cannot_implement_is_refused(
            self, host, fake_pocket_worker):
        """I-PKT-13. A mode this side has no report for would be a waiting state
        nothing ever clears."""
        with pytest.raises(runtime.PocketRuntimeError) as raised:
            runtime.ensure_started()
        assert "interrupt mode" in str(raised.value)

    @pytest.mark.pocket(sample_rate=0)
    def test_a_sample_rate_the_browser_cannot_play_is_refused(self, host,
                                                              fake_pocket_worker):
        with pytest.raises(runtime.PocketRuntimeError) as raised:
            runtime.ensure_started()
        assert "sample rate" in str(raised.value)

    def test_it_refuses_to_start_when_pocket_is_not_the_selected_engine(
            self, host, fake_pocket_worker):
        """One active TTS worker at a time. A stale request from a page drawn
        before somebody switched must not load a Torch runtime behind them."""
        runtime.stop("test")
        engines.select("kokoro")
        with pytest.raises(runtime.PocketRuntimeError) as raised:
            runtime.ensure_started()
        assert "not the selected" in str(raised.value)

    def test_a_status_read_starts_nothing(self, host, voice_root):
        """Section 17. Reading status must never begin a load."""
        found = runtime.status()
        assert found["loaded"] is False
        assert runtime.engine()["loaded"] is False


class TestNothingPrivateReachesAStatusPayload:
    """Section 18's "do not report" list, checked rather than trusted."""

    def test_the_engine_state_carries_no_path_pid_or_interpreter(self, host,
                                                                 fake_pocket_worker):
        import json

        runtime.ensure_started()
        text = json.dumps(runtime.engine()) + json.dumps(runtime.status())
        for forbidden in ("/tmp", "python", "pid", "\\\\", "token"):
            assert forbidden not in text.casefold(), forbidden


# --------------------------------------------------------------------------- #
# A turn
# --------------------------------------------------------------------------- #


class TestOneSpeakingTurn:
    def test_begin_text_end_produces_audio_and_a_segment_row(self, host,
                                                             fake_pocket_worker):
        """T-PKT-RT-2. The ordinary path, unchanged by anything the drain adds."""
        turn = Turn()
        rate = runtime.begin_turn(turn, "pocket:official:alba", None)
        assert rate == 24000
        runtime.send_segment(turn, "Hello there.")
        wait_until(lambda: turn.audio, what="audio")
        runtime.finish_turn(turn)
        wait_until(lambda: turn.finished, what="tts_done")
        assert turn.segments and turn.segments[0]["blocks"] == 1
        assert turn.streaming == "chunk"

    def test_a_frame_for_a_forgotten_turn_is_dropped_at_one_point(self, host,
                                                                  fake_pocket_worker):
        """T-PKT-RT-5. Late audio for a turn nobody remembers reaches nothing."""
        turn = Turn("gone")
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime._release_turn(turn)
        runtime.send_segment(turn, "Nobody is listening.")
        time.sleep(0.2)
        assert turn.audio == []


# --------------------------------------------------------------------------- #
# The drain
# --------------------------------------------------------------------------- #


class TestInterruptionIsADrainWithABoundedWait:
    """GATE P-3, on the parent side."""

    @pytest.mark.pocket(drain_seconds=0.3)
    def test_stop_returns_at_once_rather_than_waiting_for_the_unit(
            self, host, fake_pocket_worker):
        """I-PKT-11. The browser is already silent; a Stop that blocked for the
        length of a generation would be a Stop that felt broken."""
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "A long sentence.")
        wait_until(lambda: turn.audio, what="audio")
        turn.cancel("user")
        began = time.monotonic()
        runtime.interrupt_turn(turn)
        assert time.monotonic() - began < 0.2, "interrupt_turn waited for the drain"
        assert turn.draining is True
        wait_until(lambda: turn.draining is False, what="the lane to be reported free")

    @pytest.mark.pocket(drain_seconds=0.2)
    def test_no_audio_produced_after_the_stop_reaches_playback(self, host,
                                                               fake_pocket_worker):
        """I-PKT-12 and T-PKT-E2E-5. The worker keeps producing during the
        drain; every block of it is consumed here and thrown away."""
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "Hello.")
        wait_until(lambda: turn.audio, what="audio")
        before = len(turn.audio)
        turn.cancel("user")
        runtime.interrupt_turn(turn)
        wait_until(lambda: turn.draining is False, what="the drain to complete")
        time.sleep(0.1)
        assert len(turn.audio) == before, "audio produced during the drain was played"

    @pytest.mark.pocket(drain_seconds=0.2)
    def test_the_drain_record_lives_until_the_worker_says_the_lane_is_free(
            self, host, fake_pocket_worker):
        """T-PKT-RT-6. Cleared by an authoritative report and by nothing else --
        never a timer, because a waiting state that clears itself is a waiting
        state that lets a second generation start."""
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "Hello.")
        wait_until(lambda: turn.audio, what="audio")
        turn.cancel("user")
        runtime.interrupt_turn(turn)
        assert runtime.status()["draining"] is True
        assert runtime.status()["busy"] is True
        wait_until(lambda: runtime.status()["draining"] is False,
                   what="the drain to complete")
        assert runtime.status()["busy"] is False

    @pytest.mark.pocket(drain_seconds=0.4)
    def test_a_new_turn_waits_for_the_lane_rather_than_overlapping(
            self, host, fake_pocket_worker):
        """I-PKT-13. Upstream documents the model as not thread-safe, so a new
        generation while an abandoned one is alive would be incorrect."""
        first = Turn("A")
        runtime.begin_turn(first, "pocket:official:alba", None)
        runtime.send_segment(first, "The first reply.")
        wait_until(lambda: first.audio, what="audio")
        first.cancel("superseded")
        runtime.interrupt_turn(first)

        second = Turn("B")
        began = time.monotonic()
        runtime.begin_turn(second, "pocket:official:alba", None)
        waited = time.monotonic() - began
        assert waited >= 0.2, "the new turn started while the old one was draining"
        assert runtime.status()["draining"] is False

    @pytest.mark.pocket(drain_seconds=0.6)
    def test_a_waiting_turn_that_is_itself_stopped_does_not_wait_any_longer(
            self, host, fake_pocket_worker):
        """A Stop pressed while a new turn is queued behind a drain should not
        have to wait for the drain too."""
        first = Turn("A")
        runtime.begin_turn(first, "pocket:official:alba", None)
        runtime.send_segment(first, "The first reply.")
        wait_until(lambda: first.audio, what="audio")
        first.cancel("superseded")
        runtime.interrupt_turn(first)

        second = Turn("B")
        second.cancel("user")
        began = time.monotonic()
        runtime.begin_turn(second, "pocket:official:alba", None)
        assert time.monotonic() - began < 0.3

    def test_interrupting_a_turn_this_runtime_never_opened_is_a_no_op(
            self, host, fake_pocket_worker):
        """Stop is idempotent and the browser sends what it has."""
        runtime.ensure_started()
        runtime.interrupt_turn(Turn("never-opened"))
        assert runtime.status()["draining"] is False

    @pytest.mark.pocket(drain_seconds=0.2)
    def test_a_second_stop_on_a_draining_turn_changes_nothing(self, host,
                                                              fake_pocket_worker):
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "Hello.")
        wait_until(lambda: turn.audio, what="audio")
        turn.cancel("user")
        runtime.interrupt_turn(turn)
        runtime.interrupt_turn(turn)
        wait_until(lambda: runtime.status()["draining"] is False, what="the drain")


class TestTheWaitingStateIsAlwaysTakenBackDown:
    """The two halves of "Voice finishing..." that a frame-level test misses.

    Both are about ordering rather than about frames, and both are invisible
    until something waits on the state: an indicator that never clears looks
    exactly like a drain that is still running, and a release envelope that was
    never measured looks exactly like one that measured zero.
    """

    @pytest.mark.pocket(drain_seconds=0.2)
    def test_a_report_that_arrives_before_the_write_returns_still_clears(
            self, host, fake_pocket_worker, monkeypatch):
        """The worker answers ``state="complete"`` synchronously when nothing
        was inside the model -- a Stop between units, or on a turn the lane had
        not picked up. The reader thread can therefore reach
        :func:`_finish_drain` before the line after ``_write`` runs, and
        ``interrupted()`` returns early on a turn that is not yet marked
        draining. Marking it *after* that would leave a "Voice finishing..."
        with nothing left to take it down.

        Driven synchronously rather than by racing the reader, because a race
        left to the scheduler is a race that passes on the machine that runs it.
        """
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "Hello.")
        wait_until(lambda: turn.audio, what="audio")

        write = runtime._write

        def early(header, payload=b""):
            found = write(header, payload)
            if str(header.get("op") or "") == "tts_interrupt":
                runtime._finish_drain(header.get("turn"),
                                      {"state": "complete", "chars": 0, "audio_ms": 0})
            return found

        monkeypatch.setattr(runtime, "_write", early)
        turn.cancel("user")
        runtime.interrupt_turn(turn)

        assert turn.draining is False, "the turn was left finishing forever"
        assert runtime.status()["draining"] is False
        assert runtime.status()["busy"] is False
        assert turn.ready_at > 0.0, "the stop-to-ready measurement was never taken"

    @pytest.mark.pocket(drain_seconds=0.15)
    def test_the_abandoned_unit_is_measured_rather_than_merely_reported(
            self, host, fake_pocket_worker):
        """GATE P-3 is a cost against a size: "a Stop took 4.2 seconds" is only
        a finding if something also recorded that the unit was seventy-eight
        characters. The completing frame carries both, and this is the one place
        they are read off it."""
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "A sentence being spoken.")
        wait_until(lambda: turn.audio, what="audio")
        turn.cancel("user")
        runtime.interrupt_turn(turn)
        wait_until(lambda: turn.draining is False, what="the drain to complete")

        assert turn.unit["chars"] == 42, "the abandoned unit's size was dropped"
        assert turn.unit["audio_ms"] == 300, "the discarded audio was not measured"


class TestLifecycleIsNotStop:
    """Section 21.6. A switch, an unload or a settings restart ends the process."""

    @pytest.mark.pocket(drain_seconds=5.0)
    def test_unloading_during_a_drain_does_not_wait_for_it(self, host,
                                                           fake_pocket_worker):
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "Hello.")
        wait_until(lambda: turn.audio, what="audio")
        turn.cancel("user")
        runtime.interrupt_turn(turn)
        began = time.monotonic()
        runtime.unload("test")
        assert time.monotonic() - began < 4.0, "unload waited for the drain"
        assert runtime.status()["draining"] is False
        assert runtime.status()["busy"] is False

    @pytest.mark.pocket(drain_seconds=5.0)
    def test_the_pipe_ending_releases_the_lane(self, host, fake_pocket_worker):
        """A lane held by a process that no longer exists is a lane nothing
        would ever free."""
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "Hello.")
        wait_until(lambda: turn.audio, what="audio")
        turn.cancel("user")
        runtime.interrupt_turn(turn)
        runtime.stop("the worker went away")
        assert runtime._lane_free.is_set()
        assert turn.draining is False

    def test_switching_engines_stops_the_worker(self, host, fake_pocket_worker):
        runtime.ensure_started()
        assert runtime.engine()["loaded"] is True
        engines.select("kokoro")
        assert runtime.engine()["loaded"] is False


class TestTheDrainFailsafeIsBoundedRatherThanForever:
    @pytest.mark.pocket(never_finishes=True)
    def test_a_unit_that_never_returns_restarts_the_worker(self, host,
                                                           fake_pocket_worker,
                                                           monkeypatch):
        """Section 21.8. A failsafe for abnormal non-quiescence, and not the
        normal implementation of Stop -- so the bound is shortened here rather
        than the test waiting a minute for it."""
        monkeypatch.setattr(runtime, "DRAIN_GRACE_HARD", 0.4)
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "Hello.")
        wait_until(lambda: turn.audio, what="audio")
        turn.cancel("user")
        runtime.interrupt_turn(turn)
        assert runtime.status()["draining"] is True

        second = Turn("B")
        began = time.monotonic()
        runtime.begin_turn(second, "pocket:official:alba", None)
        assert time.monotonic() - began >= 0.3
        # The old process was replaced rather than a second generation started
        # inside it.
        assert runtime.status()["draining"] is False


class TestRequestsReachAWorkerThatIsInsideAGeneration:
    def test_a_status_read_answers_while_a_turn_is_speaking(self, host,
                                                            fake_pocket_worker):
        """T-PKT-RT-4. Nothing that waits holds the state lock, which is what
        lets Stop, a poll and a switch all reach a busy runtime."""
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "Hello.")
        wait_until(lambda: turn.audio, what="audio")
        assert runtime.status()["busy"] is True
        assert runtime.engine()["state"] in ("speaking", "idle")

    def test_a_crash_loop_is_bounded(self, host, voice_root, pocket_installed,
                                     monkeypatch):
        """T-PKT-RT-8. A worker that cannot load its model will not load it on
        the tenth attempt either, and a respawn loop during an image generation
        is a machine that gets slower for no reason."""
        import mc_voice_paths

        engines.select("pocket")
        monkeypatch.setattr(mc_voice_paths, "pocket_worker_script",
                            lambda: mc_voice_paths.extension_root() / "nothing-here.py")
        for _attempt in range(runtime.CRASH_LIMIT):
            with pytest.raises(runtime.PocketRuntimeError):
                runtime.ensure_started()
        with pytest.raises(runtime.PocketRuntimeError) as raised:
            runtime.ensure_started()
        assert "several times in a row" in str(raised.value)
        runtime._failures.clear()


class TestTheLaneIsNeverHeldByATurnThatIsAlreadyOver:
    def test_a_stop_on_the_last_word_does_not_hold_the_lane(self, host,
                                                            fake_pocket_worker):
        """The reply can finish between the turn thread deciding to interrupt
        and the interrupt reaching this module. A drain record made for a turn
        the worker has already closed is a record nothing will ever close, so
        the lane would stay held until the failsafe rather than until the next
        sentence."""
        turn = Turn()
        runtime.begin_turn(turn, "pocket:official:alba", None)
        runtime.send_segment(turn, "Hello.")
        wait_until(lambda: turn.audio, what="audio")
        runtime.finish_turn(turn)
        wait_until(lambda: turn.finished, what="tts_done")
        turn.synthesis_done = True
        turn.cancel("user")
        runtime.interrupt_turn(turn)
        assert runtime.status()["draining"] is False
        assert runtime.status()["busy"] is False
        assert turn.draining is False
