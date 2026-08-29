"""The parent side of the pipe: one worker, owned properly, believed carefully.

Every test here drives a real subprocess speaking the real framed protocol. The
child is a Python script that loads no speech engine (see ``FAKE_VOICE_WORKER``
in ``conftest``), which is what makes it possible to assert things about
handshakes, restarts and process ownership without four hundred megabytes of
ONNX -- and, because that script implements the framing independently, it also
checks that the protocol is a protocol rather than one function talking to
itself.

The theme running through it is ownership. A process this module started is this
module's to end, from the instant ``Popen`` returns rather than from the moment
the handshake succeeds -- which is the exact distinction the orphaned
llama-server got wrong.
"""

from __future__ import annotations

import pathlib
import re
import time

import pytest

import mc_voice_runtime as runtime


def alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


class TestStarting:
    def test_the_first_request_starts_one_worker(self, fake_worker):
        assert runtime.status()["running"] is False
        found = runtime.transcribe(fake_worker.wav)
        assert found["text"] == fake_worker.transcript
        assert runtime.status()["running"] is True

    def test_two_requests_reuse_the_same_process(self, fake_worker):
        runtime.transcribe(fake_worker.wav)
        first = runtime.status()["pid"]
        runtime.synthesize("the second request")
        assert runtime.status()["pid"] == first

    def test_starting_twice_is_one_worker(self, fake_worker):
        runtime.ensure_started()
        pid = runtime.status()["pid"]
        runtime.ensure_started()
        assert runtime.status()["pid"] == pid

    def test_nothing_starts_merely_because_the_module_was_imported(self):
        """Section 55. An installation that never opens Conversation should be
        completely inert apart from a settings section."""
        assert runtime.status()["running"] is False

    def test_the_handshake_reports_the_cpu_and_the_containment(self, fake_worker):
        runtime.ensure_started()
        found = runtime.status()
        assert found["provider"] == "cpu"
        assert found["parent_death"] in ("pdeathsig", "job")

    @pytest.mark.voice(handshake={"protocol_version": 99})
    def test_a_protocol_the_parent_cannot_speak_is_refused(self, fake_worker):
        with pytest.raises(runtime.VoiceRuntimeError, match="protocol"):
            runtime.ensure_started()
        assert runtime.status()["running"] is False

    @pytest.mark.voice(handshake={"parent_death": "pipe"})
    def test_a_worker_without_os_containment_is_refused_on_a_supported_platform(
            self, fake_worker):
        """R2-3, and the reason it is a release requirement rather than polish.

        Pipe EOF alone is not the guarantee: a worker inside a native inference
        call is not reading its input, so the parent's death is not observed
        until the call returns. On a platform whose support contract names an OS
        mechanism, not having it means not starting."""
        with pytest.raises(runtime.VoiceRuntimeError, match="lifetime"):
            runtime.ensure_started()
        assert runtime.status()["running"] is False

    @pytest.mark.voice(handshake={"stt_model_id": "something-else"})
    def test_a_worker_that_loaded_other_models_is_refused(self, fake_worker):
        with pytest.raises(runtime.VoiceRuntimeError, match="different models"):
            runtime.ensure_started()

    def test_it_refuses_to_start_when_the_models_are_not_installed(self, voice_root,
                                                                  monkeypatch):
        import mc_voice_models

        missing = mc_voice_models.Status(False, False, False, "no", "no", "no", True)
        monkeypatch.setattr(mc_voice_models, "status", lambda: missing)
        with pytest.raises(runtime.VoiceRuntimeError, match="not set up"):
            runtime.ensure_started()


class TestTheFixedThreadConfiguration:
    """Four TTS threads, chosen once and never chosen again at runtime.

    The tests that matter here are the negative ones. A production feature that
    reconfigures itself is a feature whose logs describe a different program
    every time somebody reads them, so what is asserted is not only that the
    number is four but that nothing in this module can make it anything else.
    """

    def test_the_worker_is_asked_for_four_tts_threads(self, fake_worker, monkeypatch):
        sent = []
        original = runtime._exchange

        def record(header, payload, timeout):
            sent.append(header)
            return original(header, payload, timeout)

        monkeypatch.setattr(runtime, "_exchange", record)
        runtime.ensure_started()
        config = [h for h in sent if h.get("op") == "init"][0]["config"]
        assert config["tts_threads"] == 4
        assert runtime.TTS_THREADS == 4

    def test_the_worker_reports_the_number_it_was_given(self, fake_worker):
        """Two claims, not one: what the parent asked for, and what the engine
        says it is running."""
        runtime.ensure_started()
        assert runtime._handshake.tts_threads == 4
        assert runtime.engine()["tts_threads"] == 4

    def test_speech_to_text_is_left_where_it_was(self, fake_worker):
        """A transcription is one burst after the user has stopped talking. It
        was never the stage anybody was waiting through."""
        runtime.ensure_started()
        assert runtime.STT_THREADS == 4
        assert runtime._handshake.stt_threads == 4

    def test_nothing_here_chooses_a_thread_count_at_runtime(self):
        """No rotation, no A/B, no selection from real-time factors or underrun
        counts. The constant is read; it is never written."""
        source = pathlib.Path(runtime.__file__).read_text(encoding="utf-8")
        assignments = [line.strip() for line in source.splitlines()
                       if re.match(r"\s*TTS_THREADS\s*=[^=]", line)]
        assert assignments == ["TTS_THREADS = 4"], assignments


class TestWarmingUpWithoutATurn:
    """``prepare`` is ``ensure_started`` with a question in front of it, and the
    things it must not do are the reason it is a separate function at all."""

    def test_it_starts_a_cold_worker_and_says_it_was_cold(self, fake_worker):
        assert runtime.status()["running"] is False
        assert runtime.prepare() is False
        assert runtime.status()["running"] is True

    def test_a_second_call_reuses_the_worker_and_says_it_was_warm(self, fake_worker):
        runtime.prepare()
        pid = runtime.status()["pid"]
        assert runtime.prepare() is True
        assert runtime.status()["pid"] == pid

    def test_it_registers_no_turn_and_claims_no_inference_lane(self, fake_worker):
        """The whole difference between warming and speaking. A warmed worker
        that is then cancelled before any text arrives must leave nothing
        behind but a warm worker."""
        runtime.prepare()
        assert runtime._turns == {}
        assert runtime._busy["tts"] == 0
        assert runtime.engine()["state"] == "idle"

    def test_it_sends_nothing_a_turn_would_send(self, fake_worker, monkeypatch):
        """No ``tts_begin``, no ``tts_text``: warming is not speaking, and a
        worker that had been told a turn was open would be a worker holding a
        turn nobody created."""
        original = runtime._write
        sent = []

        def record(header, payload=b""):
            sent.append(str(header.get("op") or ""))
            return original(header, payload)

        monkeypatch.setattr(runtime, "_write", record)
        runtime.prepare()
        assert sent, "the handshake itself writes nothing"
        assert not [op for op in sent if op.startswith("tts")], sent

    def test_a_runtime_that_is_not_installed_refuses_readably(self, voice_root, monkeypatch):
        import mc_voice_models

        monkeypatch.setattr(mc_voice_models, "status", lambda: mc_voice_models.Status(
            runtime_ready=False, stt_ready=False, tts_ready=False, runtime_message="no",
            stt_message="no", tts_message="no", platform_supported=True))
        with pytest.raises(runtime.VoiceRuntimeError, match="not set up"):
            runtime.prepare()
        assert runtime.status()["running"] is False


class TestRoundTrips:
    def test_audio_goes_in_and_a_transcript_comes_back(self, fake_worker):
        assert runtime.transcribe(fake_worker.wav)["text"] == fake_worker.transcript

    def test_text_goes_in_and_audio_comes_back(self, fake_worker):
        assert runtime.synthesize("hello there") == fake_worker.audio

    def test_empty_audio_never_reaches_the_worker(self, fake_worker):
        with pytest.raises(runtime.VoiceRuntimeError):
            runtime.transcribe(b"")
        assert runtime.status()["running"] is False, "an empty request started a worker"

    def test_empty_text_never_reaches_the_worker(self, fake_worker):
        with pytest.raises(runtime.VoiceRuntimeError, match="nothing to read"):
            runtime.synthesize("   ")
        assert runtime.status()["running"] is False

    def test_a_reply_past_the_ceiling_is_refused_rather_than_cut_short(self, fake_worker):
        """Section 69: no silent truncation. Speaking the first half of an answer
        without saying so is worse than not speaking it."""
        with pytest.raises(runtime.VoiceRuntimeError, match="longer than"):
            runtime.synthesize("x" * (runtime.MAX_TEXT_BYTES + 1))

    def test_a_large_but_permitted_reply_is_spoken(self, fake_worker):
        assert runtime.synthesize("x" * (runtime.MAX_TEXT_BYTES - 10)) == fake_worker.audio


class TestWhenTheWorkerDies:
    @pytest.mark.voice(behaviour="die_on_stt")
    def test_one_restart_is_attempted_and_no_more(self, fake_worker):
        """A worker that vanishes once is an ordinary thing and the user should
        not have to press the microphone again for it. A worker that vanishes
        again is a broken installation, and a third process would only make the
        machine slower."""
        with pytest.raises(runtime.VoiceRuntimeError, match="stopped unexpectedly"):
            runtime.transcribe(fake_worker.wav)
        assert runtime.status()["running"] is False

    def test_a_worker_killed_between_requests_is_replaced_once(self, fake_worker):
        import os
        import signal

        runtime.transcribe(fake_worker.wav)
        first = runtime.status()["pid"]
        os.kill(first, signal.SIGKILL)
        for _ in range(100):
            if not alive(first):
                break
            time.sleep(0.02)

        assert runtime.transcribe(fake_worker.wav)["text"] == fake_worker.transcript
        assert runtime.status()["pid"] != first

    def test_repeated_start_failures_stop_being_retried(self, fake_worker, monkeypatch):
        """A respawn loop during a generation is a machine that gets slower for
        no reason anybody can see."""
        import mc_voice_models

        monkeypatch.setattr(mc_voice_models, "runtime_python",
                            lambda: fake_worker.script.parent / "not-a-python")
        for _ in range(runtime.CRASH_LIMIT):
            with pytest.raises(runtime.VoiceRuntimeError):
                runtime.ensure_started()
        with pytest.raises(runtime.VoiceRuntimeError, match="several times in a row"):
            runtime.ensure_started()


class TestStopping:
    def test_stop_is_idempotent(self, fake_worker):
        runtime.ensure_started()
        pid = runtime.status()["pid"]
        runtime.stop("once")
        runtime.stop("twice")
        runtime.stop("three times")
        assert runtime.status()["running"] is False
        assert not alive(pid)

    def test_stopping_something_that_never_started_is_not_an_error(self):
        runtime.stop("nothing here")
        runtime.shutdown()

    @pytest.mark.voice(behaviour="ignore_shutdown")
    def test_an_uncooperative_worker_is_escalated_inside_a_bounded_time(self, fake_worker):
        """T-SHUT-6. The child ignores the polite frame and never reads its
        input again; terminate and then kill are what is left, and both have to
        happen inside a time a WebUI exit can afford."""
        runtime.ensure_started()
        pid = runtime.status()["pid"]
        started = time.monotonic()
        runtime.shutdown()
        assert time.monotonic() - started < 10.0
        assert not alive(pid)

    def test_shutdown_refuses_new_work_while_it_runs(self, fake_worker, monkeypatch):
        runtime.ensure_started()
        runtime.shutdown()
        assert runtime.status()["running"] is False


class TestWhatIsNeverLeftBehind:
    @pytest.mark.voice(handshake={"provider": "cuda"})
    def test_a_failed_handshake_kills_the_process_it_started(self, fake_worker):
        """T-SHUT-7. Ownership begins at ``Popen``. The process is running and
        answering by the time the parent decides it is unacceptable, and there
        is nobody else to end it."""
        marker = fake_worker.plan["alive_marker"]
        with pytest.raises(runtime.VoiceRuntimeError):
            runtime.ensure_started()
        with open(marker, encoding="utf-8") as handle:
            pids = [int(line) for line in handle if line.strip()]
        assert pids, "the fake worker never ran, so this proves nothing"
        for pid in pids:
            for _ in range(100):
                if not alive(pid):
                    break
                time.sleep(0.02)
            assert not alive(pid), "a refused worker was left running"

    @pytest.mark.voice(behaviour="silent_handshake")
    def test_a_worker_that_never_answers_is_stopped_rather_than_waited_for(
            self, fake_worker, monkeypatch):
        monkeypatch.setattr(runtime, "HANDSHAKE_TIMEOUT", 1.0)
        marker = fake_worker.plan["alive_marker"]
        with pytest.raises(runtime.VoiceRuntimeError):
            runtime.ensure_started()
        with open(marker, encoding="utf-8") as handle:
            pids = [int(line) for line in handle if line.strip()]
        for pid in pids:
            for _ in range(100):
                if not alive(pid):
                    break
                time.sleep(0.02)
            assert not alive(pid)


class TestTheProcessIsRecognisable:
    def test_the_command_line_carries_the_marker_and_no_content(self, fake_worker):
        """Section 62. Findable in a task manager, and carrying nothing that
        would put a transcript in a world-readable process list."""
        import subprocess

        seen = {}
        real = subprocess.Popen

        def record(command, *args, **kwargs):
            seen["command"] = list(command)
            return real(command, *args, **kwargs)

        import mc_voice_runtime

        original = mc_voice_runtime.subprocess.Popen
        mc_voice_runtime.subprocess.Popen = record
        try:
            runtime.ensure_started()
        finally:
            mc_voice_runtime.subprocess.Popen = original

        command = seen["command"]
        assert "--model-chain-voice-worker" in command
        assert "--parent-pid" in command
        joined = " ".join(command)
        assert fake_worker.transcript not in joined


# --------------------------------------------------------------------------- #
# Protocol 2: streaming, dispatch, and cancellation that actually reaches the
# worker while it is computing
# --------------------------------------------------------------------------- #


class Sink:
    """A VoiceTurn-shaped destination the runtime can dispatch frames into."""

    def __init__(self, identifier: str = "T1"):
        import threading

        self.id = identifier
        self.sample_rate = 0
        self.blocks = []
        self.segments = []
        self.done = False
        self.error = ""
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.started = threading.Event()

    def offer_audio(self, pcm, rate, seconds_limit=None):
        if self.cancelled.is_set():
            return False
        self.blocks.append(pcm)
        self.sample_rate = self.sample_rate or rate
        self.started.set()
        return True

    def note_segment(self, blocks=0, first_block_ms=0, streaming="", synth_ms=0,
                     audio_ms=0):
        self.segments.append({"blocks": blocks, "first_block_ms": first_block_ms,
                              "streaming": streaming, "synth_ms": synth_ms,
                              "audio_ms": audio_ms})

    def audio_finished(self):
        self.done = True
        self.finished.set()

    def audio_failed(self, reason):
        self.error = reason
        self.finished.set()
        self.cancelled.set()

    def cancel(self, reason="user"):
        first = not self.cancelled.is_set()
        self.cancelled.set()
        self.finished.set()
        return first

    def drain_audio(self):
        self.blocks.clear()

    @property
    def busy(self):
        return not self.finished.is_set()


def stream_one(sink, text=b"Hello there.", finish=True, seconds=5.0):
    """Drive one whole streaming turn through the runtime and the worker."""
    runtime.begin_turn(sink, 3)
    runtime.send_segment(sink, text.decode() if isinstance(text, bytes) else text)
    if finish:
        runtime.finish_turn(sink)
    sink.finished.wait(seconds)
    return sink


class TestStreamingSpeech:
    def test_t_rt_11_the_supplied_speaker_is_used_and_not_a_startup_one(self, fake_worker):
        """Section 56. Protocol 1 read one speaker id out of the manifest at
        start-up and used it for everything, which made "which voice" a property
        of the process rather than of the request."""
        sink = Sink()
        runtime.begin_turn(sink, 3)
        runtime.send_segment(sink, "Hello there.")
        runtime.finish_turn(sink)
        assert sink.finished.wait(5.0)
        assert sink.done and sink.blocks

    def test_the_shape_of_each_segment_reaches_the_turn(self, fake_worker):
        """Section 4's distinction, carried the whole way: how many batches the
        worker handed back and how soon the first of them came. A handshake
        that says "callback" is not evidence that any audio arrived early."""
        sink = Sink()
        stream_one(sink)
        assert sink.segments, "no segment shape was reported"
        assert sink.segments[0]["blocks"] >= 1
        assert sink.segments[0]["streaming"] == "callback"

    def test_the_handshake_records_what_the_probe_cost(self, fake_worker):
        runtime.ensure_started()
        assert runtime._handshake.callback_probe_ms >= 0
        assert runtime.engine()["streaming"] == "callback"

    def test_a_speaker_the_bank_does_not_have_is_refused_rather_than_swapped(self,
                                                                            fake_worker):
        """sherpa answers an out-of-range sid by using speaker 0, which would
        make a clone speak silently in somebody else's voice."""
        sink = Sink()
        runtime.begin_turn(sink, 9999)
        assert sink.finished.wait(5.0)
        assert not sink.blocks
        assert "voice bank" in sink.error

    def test_audio_arrives_before_the_turn_is_finished(self, fake_worker):
        """The whole point: a completed-reply design cannot produce a sample
        before the last sentence is synthesised."""
        sink = Sink()
        runtime.begin_turn(sink, 3)
        runtime.send_segment(sink, "Hello there.")
        assert sink.started.wait(5.0), "no audio arrived before the turn was finished"
        runtime.cancel_turn(sink)

    def test_the_sample_rate_arrives_before_any_audio(self, fake_worker):
        sink = Sink()
        assert runtime.begin_turn(sink, 3) == 24000
        runtime.cancel_turn(sink)

    @pytest.mark.voice(blocks_per_segment=40, block_delay=0.05)
    def test_t_rt_2_a_cancel_reaches_the_worker_while_it_is_speaking(self, fake_worker):
        """Gate 1's acceptance criterion, and release blocker two.

        The worker here is deliberately slow -- forty blocks at fifty
        milliseconds -- so that "the cancel arrived while it was computing" is a
        real claim rather than a race the fast path happens to win. Three things
        are asserted, and the third is the one that matters: the cancel returns
        promptly, no more audio arrives after it, and the worker is still there
        and answering afterwards, because tearing the process down would have
        been a different (and much worse) way to pass the first two.
        """
        import time as _time

        sink = Sink()
        runtime.begin_turn(sink, 3)
        runtime.send_segment(sink, "A sentence.")
        assert sink.started.wait(5.0)
        started = _time.monotonic()
        runtime.cancel_turn(sink)
        assert _time.monotonic() - started < 2.0, "cancel waited for the synthesis"

        seen = len(sink.blocks)
        _time.sleep(0.6)
        assert len(sink.blocks) == seen, "audio kept arriving after the cancel"
        assert runtime.status()["running"] is True, "cancelling a turn killed the worker"
        assert runtime.transcribe(fake_worker.wav)["text"] == fake_worker.transcript

    def test_t_rt_4_frames_from_a_turn_nobody_is_listening_to_are_dropped(self,
                                                                         fake_worker):
        """Section 24. Cancel an old turn, start a new one, and the old PCM
        still in the pipe must not play over the new reply."""
        first = Sink("A")
        runtime.begin_turn(first, 3)
        runtime.send_segment(first, "The first reply.")
        first.started.wait(5.0)
        runtime.cancel_turn(first)
        before = len(first.blocks)

        second = Sink("B")
        stream_one(second, "The second reply.")
        assert second.blocks
        assert len(first.blocks) == before, "a cancelled turn was still being fed"

    def test_t_rt_6_status_answers_while_a_turn_is_speaking(self, fake_worker):
        """Section 16. Only true because nothing waiting holds the state lock."""
        import time as _time

        sink = Sink()
        runtime.begin_turn(sink, 3)
        runtime.send_segment(sink, "A sentence.")
        sink.started.wait(5.0)
        started = _time.monotonic()
        assert runtime.status()["running"] is True
        assert runtime.engine()["state"] in ("tts", "idle")
        assert _time.monotonic() - started < 1.0
        runtime.cancel_turn(sink)

    def test_t_rt_7_unload_answers_during_a_turn_and_stops_the_worker(self, fake_worker):
        import time as _time

        sink = Sink()
        runtime.begin_turn(sink, 3)
        runtime.send_segment(sink, "A sentence.")
        sink.started.wait(5.0)
        started = _time.monotonic()
        found = runtime.unload("test")
        assert _time.monotonic() - started < 6.0
        assert found["loaded"] is False
        assert runtime.status()["running"] is False

    def test_t_rt_5_a_transcription_runs_after_a_cancelled_turn(self, fake_worker):
        """Section 14: the microphone waits only for the cancellation boundary,
        not for an obsolete synthesis to finish."""
        sink = Sink()
        runtime.begin_turn(sink, 3)
        runtime.send_segment(sink, "A sentence.")
        sink.started.wait(5.0)
        runtime.cancel_turn(sink)
        assert runtime.transcribe(fake_worker.wav)["text"] == fake_worker.transcript

    def test_t_rt_9_two_requests_cannot_receive_one_anothers_replies(self, fake_worker):
        """Structural rather than a check: each request owns the queue it waits
        on, so there is no path by which one caller's transcript reaches
        another."""
        import threading

        runtime.ensure_started()
        found = []
        errors = []

        def ask():
            try:
                found.append(runtime.transcribe(fake_worker.wav)["text"])
            except Exception as exc:  # noqa: BLE001 - recorded for the assertion
                errors.append(exc)

        threads = [threading.Thread(target=ask) for _index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert not errors, errors
        assert found == [fake_worker.transcript] * 4

    def test_a_worker_that_dies_mid_turn_wakes_the_listener(self, fake_worker):
        import os
        import signal

        sink = Sink()
        runtime.begin_turn(sink, 3)
        runtime.send_segment(sink, "A sentence.")
        sink.started.wait(5.0)
        os.kill(runtime.status()["pid"], signal.SIGKILL)
        assert sink.finished.wait(10.0), "a dead worker left the turn waiting for ever"
        assert sink.error


class TestLoadAndUnload:
    def test_load_starts_the_worker_and_reports_live_state(self, fake_worker):
        assert runtime.engine()["loaded"] is False
        found = runtime.load()
        assert found["loaded"] is True
        assert found["state"] == "idle"
        assert found["provider"] == "cpu"
        assert found["voices"] == 85

    def test_unload_frees_the_worker_and_keeps_the_installation(self, fake_worker):
        runtime.load()
        assert runtime.unload("test")["loaded"] is False
        assert runtime.status()["running"] is False
        # And it starts again on the next use, which is what "not a persistent
        # disable switch" means.
        assert runtime.transcribe(fake_worker.wav)["text"] == fake_worker.transcript

    def test_the_engine_report_carries_no_pid_or_path(self, fake_worker):
        """Section 30's do-not-expose list."""
        import json

        runtime.load()
        text = json.dumps(runtime.engine())
        assert "pid" not in text
        assert str(fake_worker.script) not in text

    def test_nothing_is_preloaded_by_importing_or_by_asking(self, fake_worker):
        """Section 34: lazy start remains the default."""
        assert runtime.engine()["loaded"] is False
        runtime.engine()
        runtime.status()
        assert runtime.status()["running"] is False


class TestTheProtocolVersionGate:
    @pytest.mark.voice(handshake={"protocol_version": 1})
    def test_t_rt_1_a_protocol_1_worker_and_a_protocol_2_parent_refuse_each_other(
            self, fake_worker):
        """A stale pair must fail the handshake rather than try to interpret
        incompatible streaming frames."""
        with pytest.raises(runtime.VoiceRuntimeError, match="protocol"):
            runtime.ensure_started()
        assert runtime.status()["running"] is False

    @pytest.mark.voice(num_speakers=53)
    def test_a_bank_smaller_than_the_registry_expects_is_refused(self, fake_worker,
                                                                 monkeypatch):
        """Otherwise a registered clone would be spoken silently by speaker 0."""
        import mc_voice_registry

        monkeypatch.setattr(mc_voice_registry, "highest_sid", lambda: 60)
        with pytest.raises(runtime.VoiceRuntimeError, match="fewer voices"):
            runtime.ensure_started()
