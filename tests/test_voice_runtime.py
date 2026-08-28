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
