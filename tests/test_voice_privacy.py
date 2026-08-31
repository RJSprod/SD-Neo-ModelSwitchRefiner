"""Nothing anybody says ends up in a log, in a file, or on a wire.

The product promise is stronger than "privacy-focused", so it is worth stating
exactly what is being defended:

    I-4  after installation, speech needs no Internet connection at all;
    I-5  microphone audio and generated speech are never written to disk;
    I-6  transcripts, messages and replies never appear in a log.

Each is tested the same way: a sentinel phrase goes in one end, everything the
extension emitted comes back, and the sentinel must not be in it. A sentinel
rather than a real sentence because a real sentence would eventually appear by
coincidence, and because the assertion should fail loudly on the *day* somebody
adds ``logger.info("transcribed %s", text)`` rather than during a code review
six months later.

The network test is the odd one out and is the strongest of the three: the
worker's request loop is run with socket creation made impossible, and both
speech operations still complete. A feature that can transcribe with no
sockets available is a feature that is not sending anything anywhere.
"""

from __future__ import annotations

import io
import json
import logging
import socket
import sys
from pathlib import Path

import pytest

import mc_voice_api as api
import mc_voice_runtime as runtime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "voice_worker"))
import worker  # noqa: E402

DICTATED = "VOICE_PRIVATE_SENTINEL_7C91"
SPOKEN = "VOICE_PRIVATE_SENTINEL_B4E2"


@pytest.fixture
def installed(monkeypatch):
    import mc_voice_models

    ready = mc_voice_models.Status(True, True, True, "i", "i", "i", True,
                                  "whisper-small-int8", "kokoro-multi-lang-v1-cpu",
                                  "af_heart")
    monkeypatch.setattr(mc_voice_models, "status", lambda: ready)
    return ready


@pytest.fixture
def captured(caplog):
    """Everything the extension's logger emitted, at every level."""
    caplog.set_level(logging.DEBUG, logger="model_chain")
    return caplog


class TestNothingSaidIsLogged:
    def test_a_transcript_never_reaches_the_log(self, captured, installed, monkeypatch,
                                                spoken_wav):
        """T-PRIV-1."""
        monkeypatch.setattr(runtime, "transcribe",
                            lambda data: {"text": DICTATED, "audio_seconds": 1.0,
                                          "elapsed": 0.2})
        found = api.transcribe(spoken_wav(1.0))
        assert found["text"] == DICTATED, "the test transcribed nothing, so it proves nothing"

        assert any("Voice STT finished" in record.getMessage()
                   for record in captured.records), "no diagnostic was written at all"
        for record in captured.records:
            assert DICTATED not in record.getMessage()
            assert DICTATED not in str(record.args or "")

    def test_a_reply_never_reaches_the_log(self, captured, installed, monkeypatch):
        """T-PRIV-2. The length is logged, which is the useful part; the words
        are not, which is the whole of the promise."""
        monkeypatch.setattr(runtime, "synthesize",
                            lambda text, sid=0, profile=None: b"RIFFfake")
        token = api.remember_reply(f"Certainly. {SPOKEN} is the answer.")
        api.speak(token)

        assert any("Voice TTS finished" in record.getMessage()
                   for record in captured.records)
        for record in captured.records:
            assert SPOKEN not in record.getMessage()

    def test_a_failing_request_does_not_report_what_was_in_it(self, captured, installed,
                                                              monkeypatch, spoken_wav):
        """The path where content most easily escapes: an exception message
        built by a library that was handed the input."""
        def explode(data):
            raise runtime.VoiceRuntimeError("the runtime is gone")

        monkeypatch.setattr(runtime, "transcribe", explode)
        with pytest.raises(api.Refused):
            api.transcribe(spoken_wav(1.0))
        for record in captured.records:
            assert DICTATED not in record.getMessage()

    def test_the_worker_reports_sizes_and_durations_and_not_words(self, capsys):
        """The worker's own diagnostics, on its own stderr."""
        class Engines:
            provider = "cpu"
            stt_model_id = "whisper-small-int8"
            tts_model_id = "kokoro-multi-lang-v1-cpu"
            stt_threads = 4
            tts_threads = 4

            def __init__(self, config):
                pass

            def load(self):
                pass

            def transcribe(self, samples, rate):
                return DICTATED

            num_speakers = 85
            sample_rate = 24000
            bank_version = ""
            streaming = "segment"
            callback_probe_ms = 0

            def speaker(self, sid):
                return int(sid or 0)

            def synthesize(self, text, sid=0, profile=None):
                assert SPOKEN in text
                return [0.0] * 2400, 24000

            def stream(self, text, sid, speed, on_audio):
                on_audio(worker.pcm16([0.0] * 2400), 24000)
                return 2400

        stdin = io.BytesIO()
        worker.write_frame(stdin, {"id": 1, "op": "init", "parent_pid": 0, "config": {}})
        worker.write_frame(stdin, {"id": 2, "op": "stt"},
                           worker.encode_wav([0.0] * 16000, 16000))
        worker.write_frame(stdin, {"id": 3, "op": "tts"}, SPOKEN.encode("utf-8"))
        stdin.seek(0)
        stdout = io.BytesIO()

        assert worker.serve(stdin, stdout, engines_factory=Engines) == 0

        noise = capsys.readouterr().err
        assert "ready" in noise, "the worker said nothing, so this proves nothing"
        assert DICTATED not in noise
        assert SPOKEN not in noise

    def test_the_worker_never_hands_a_library_message_back_unread(self):
        """A third-party library is entitled to put the input it was given into
        an exception message. What the parent is told is chosen here instead."""
        class Boom(Exception):
            def __str__(self):
                return f"failed while processing: {DICTATED}"

        assert DICTATED not in worker._safe(Boom())
        assert worker._safe(Boom()) == "Boom"

    def test_a_refusal_the_parent_turns_into_a_sentence_carries_no_content(self):
        for reason in ("audio is not mono", "audio is too long", "something unexpected"):
            assert DICTATED not in runtime._readable(reason)


class TestThePerformanceLoggingIsContentFree:
    """The new diagnostics are the most likely place for content to escape.

    They exist to answer "why was there a four-second pause", which means they
    are written on the hot path, they carry per-unit detail, and one of them is
    an HTTP route a page posts into. Each of those is a way for a sentence to
    end up in ``model_chain.log`` if nobody is checking.
    """

    def test_a_synthesis_unit_is_logged_as_numbers_only(self, captured):
        import mc_voice_turn as turns

        turn = turns.VoiceTurn()
        turn._open_unit(f"Certainly. {SPOKEN} is the answer.", 0.1)
        turn.note_segment(blocks=2, first_block_ms=40, synth_ms=900, audio_ms=1200,
                          streaming="callback")
        lines = [record.getMessage() for record in captured.records
                 if "Voice TTS segment" in record.getMessage()]
        assert lines, "no unit diagnostic was written at all"
        for line in lines:
            assert SPOKEN not in line
        assert SPOKEN not in repr(turn.metrics())

    def test_the_turn_summary_carries_no_words(self, captured, installed, monkeypatch):
        import mc_voice_turn as turns

        turn = turns.VoiceTurn()
        turn.add_text(f"Certainly. {SPOKEN} is the answer, and here is some more of it. ")
        turn.complete()
        api._log_turn(turn)
        for record in captured.records:
            assert SPOKEN not in record.getMessage()

    def test_the_telemetry_route_has_no_field_a_sentence_fits_in(self):
        """Every field is a duration, a count, or one of a fixed set of words.
        Checked against the declaration rather than against one payload, so a
        field added later has to argue with this."""
        for names in api.TELEMETRY.values():
            for name in names:
                assert name.endswith("_ms") or name.endswith("_count"), name
        for allowed in api.TELEMETRY_ENUMS.values():
            for value in allowed:
                assert value.isalpha() or "_" in value, value

    def test_the_telemetry_route_writes_nothing_it_was_not_asked_for(self, captured):
        api.telemetry({"kind": "playback", "underrun_count": 1,
                       "note": f"the user said {SPOKEN}",
                       "playback_end_reason": SPOKEN})
        for record in captured.records:
            assert SPOKEN not in record.getMessage()

    def test_the_configuration_line_names_settings_and_nothing_else(self, captured):
        line = runtime.tts_config_line()
        assert "threads=" in line and "targets=" in line
        for word in ("prompt", "reply", "transcript", "message"):
            assert word not in line


class TestNoMeasurementChangesTheConfiguration:
    """P4, as a property of the source rather than a promise in a docstring.

    The whole point of collecting these numbers is that a person reads them and
    then makes a deliberate change. A repository that reconfigured itself from
    them would be a repository whose logs describe a different program each time
    somebody opens them -- and every one of the knobs below is one somebody
    would reach for first.
    """

    def sources(self):
        root = Path(__file__).resolve().parent.parent
        return {name: (root / name).read_text(encoding="utf-8")
                for name in ("mc_voice_runtime.py", "mc_voice_turn.py", "mc_voice_api.py",
                             "mc_voice_segment.py", "voice_worker/worker.py")}

    def test_no_module_assigns_a_thread_count_or_a_priority_at_runtime(self):
        import re

        for name, source in self.sources().items():
            for line in source.splitlines():
                assert not re.match(r"\s*(TTS_THREADS|STT_THREADS)\s*=[^=]", line) or (
                    name == "mc_voice_runtime.py"
                    and line.strip() in ("TTS_THREADS = 4", "STT_THREADS = 4")), (name, line)
                assert "SetPriorityClass" not in line or name == "voice_worker/worker.py", (
                    name, line)
                assert "os.nice(" not in line or name == "voice_worker/worker.py", (name, line)

    def test_reading_a_priority_does_not_set_one(self):
        """Observation only. ``_priority`` is allowed to look; the one place
        that writes is the start-up call that has always been there."""
        source = self.sources()["voice_worker/worker.py"]
        body = source.split("def _priority()")[1].split("\ndef ")[0]
        assert "GetPriorityClass" in body
        assert "SetPriorityClass" not in body
        assert "os.nice" not in body

    def test_the_segmenter_thresholds_are_constants(self):
        import mc_voice_segment as segment

        machine = segment.Segmenter()
        for name in ("first_target", "second_target", "target", "second_soft_max",
                     "second_hard_max", "soft_max", "hard_max"):
            assert isinstance(getattr(machine, name), int)
        source = self.sources()["mc_voice_segment.py"]
        assert "underrun" not in source, "the segmenter learned about playback"
        assert "rtf" not in source.casefold(), "the segmenter learned about timing"

    def test_no_experiment_or_ab_machinery_was_introduced(self):
        """T-9. Revision 4 removes callback coalescing from scope, and a dormant
        switch for it is the thing most likely to be added "just in case"."""
        for name, source in self.sources().items():
            lowered = source.casefold()
            for banned in ("a/b test", "experiment_", "coalesc", "autotune", "auto_tune"):
                assert banned not in lowered, (name, banned)


class TestNothingIsWrittenToDisk:
    def test_speech_leaves_no_file_behind_anywhere(self, tmp_path, installed, monkeypatch,
                                                  spoken_wav, voice_root):
        """T-PRIV-3. Snapshot, run a dictation and a synthesis, snapshot again."""
        monkeypatch.setattr(runtime, "transcribe", lambda data: {"text": DICTATED})
        monkeypatch.setattr(runtime, "synthesize",
                            lambda text, sid=0, profile=None: b"RIFFfake audio")

        before = {p for p in tmp_path.rglob("*")}
        api.transcribe(spoken_wav(2.0))
        api.speak(api.remember_reply(f"a reply saying {SPOKEN}"))
        after = {p for p in tmp_path.rglob("*")}

        assert before == after
        assert not list(tmp_path.rglob("*.wav"))
        assert not list(tmp_path.rglob("*.webm"))
        assert not list(tmp_path.rglob("*.opus"))

    def test_the_worker_encodes_audio_in_memory(self):
        """``encode_wav`` takes no path and there is nowhere for one to go."""
        import inspect

        assert "path" not in inspect.signature(worker.encode_wav).parameters
        audio = worker.encode_wav([0.0, 0.1, -0.1], 24000)
        assert isinstance(audio, bytes)
        assert audio[:4] == b"RIFF"

    def test_no_voice_module_imports_tempfile(self):
        import ast

        root = Path(__file__).resolve().parent.parent
        for path in (list(root.glob("mc_voice_*.py"))
                     + [root / "voice_worker" / "worker.py",
                        root / "sopro_worker" / "worker.py",
                        root / "pocket_worker" / "worker.py"]):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    assert all(a.name != "tempfile" for a in node.names), path.name
                elif isinstance(node, ast.ImportFrom):
                    assert node.module != "tempfile", path.name

    def test_the_speech_target_is_never_persisted(self, tmp_path, monkeypatch, voice_root):
        """It is a string in this process's RAM with a two-minute life. A target
        written down would be a transcript of somebody's conversation in a file
        nobody asked for."""
        before = {p for p in tmp_path.rglob("*")}
        token = api.remember_reply(SPOKEN)
        assert {p for p in tmp_path.rglob("*")} == before
        api.take_reply(token)


class TestNoNetwork:
    def test_the_worker_serves_both_operations_with_no_sockets_available(self,
                                                                        monkeypatch):
        """T-PRIV-4, and the strongest statement this file makes.

        Socket creation is made impossible for the duration, and a full
        transcribe-and-synthesize round trip still completes. Whatever else the
        speech engine is doing, it is not talking to anybody.
        """
        def refuse(*args, **kwargs):
            raise AssertionError("the voice worker tried to open a socket")

        monkeypatch.setattr(socket, "socket", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)

        class Engines:
            provider = "cpu"
            stt_model_id = "whisper-small-int8"
            tts_model_id = "kokoro-multi-lang-v1-cpu"
            stt_threads = 4
            tts_threads = 4

            def __init__(self, config):
                pass

            def load(self):
                pass

            def transcribe(self, samples, rate):
                return "offline and working"

            num_speakers = 85
            sample_rate = 24000
            bank_version = ""
            streaming = "segment"
            callback_probe_ms = 0

            def speaker(self, sid):
                return int(sid or 0)

            def synthesize(self, text, sid=0, profile=None):
                return [0.0] * 480, 24000

            def stream(self, text, sid, speed, on_audio):
                on_audio(worker.pcm16([0.0] * 480), 24000)
                return 480

        stdin = io.BytesIO()
        worker.write_frame(stdin, {"id": 1, "op": "init", "parent_pid": 0, "config": {}})
        worker.write_frame(stdin, {"id": 2, "op": "stt"},
                           worker.encode_wav([0.0] * 16000, 16000))
        worker.write_frame(stdin, {"id": 3, "op": "tts"}, b"say this")
        stdin.seek(0)
        stdout = io.BytesIO()

        assert worker.serve(stdin, stdout, engines_factory=Engines) == 0

        stdout.seek(0)
        replies = []
        while True:
            frame = worker.read_frame(stdout)
            if frame is None:
                break
            replies.append(frame)
        assert [header["ok"] for header, _ in replies] == [True, True, True]
        assert replies[1][0]["text"] == "offline and working"
        assert replies[2][1][:4] == b"RIFF"

    def test_the_worker_has_no_network_client_at_all(self):
        """Section 53: the worker should not contain a network API. Not "should
        not use one" -- should not have one."""
        import ast

        source = (Path(__file__).resolve().parent.parent
                  / "voice_worker" / "worker.py").read_text(encoding="utf-8")
        banned = {"socket", "urllib", "http", "requests", "httpx", "ftplib", "smtplib",
                  "asyncio"}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, node.module

    def test_the_worker_environment_says_offline_to_the_libraries_that_read_it(self):
        import mc_voice_models

        environ = mc_voice_models.worker_environment()
        assert environ["HF_HUB_OFFLINE"] == "1"
        assert environ["NO_PROXY"] == "*"


class TestWhatTheBrowserIsTold:
    def test_the_status_route_carries_no_paths_and_no_process_details(self, installed):
        payload = api.status_payload()
        text = json.dumps(payload)
        assert "/" not in payload["stt_message"]
        assert "pid" not in text
        assert "session" not in text

    def test_the_transcript_goes_only_to_the_browser_that_asked(self, installed,
                                                                monkeypatch, spoken_wav):
        """It has to come back -- it is going into the composer. What matters is
        that it goes nowhere else, which is what the absence of any other
        transport in this module means."""
        monkeypatch.setattr(runtime, "transcribe", lambda data: {"text": DICTATED})
        found = api.transcribe(spoken_wav(1.0))
        assert found["text"] == DICTATED
        assert "request_id" in found
        assert DICTATED not in found["request_id"]


# --------------------------------------------------------------------------- #
# The third engine's diagnostics
# --------------------------------------------------------------------------- #


class TestPocketSaysNothingItShouldNot:
    """I-PKT-27, and the fields section 36 allows.

    Pocket adds two things nothing else in this feature had: a credential the
    installer may hold, and a *drain* whose measurements are logged after a
    Stop. Both are places content could escape that nobody had had to think
    about before, so both are checked here rather than assumed.
    """

    def test_the_worker_hands_back_a_class_name_rather_than_a_library_message(self):
        """A third-party library is entitled to put the input it was given into
        an exception message -- a path, a tensor shape, or the text that was
        being spoken. What the parent is told is chosen here instead."""
        from pocket_worker import worker as pocket

        class Boom(Exception):
            def __str__(self):
                return f"failed on /home/someone/clones/abc/reference.wav: {SPOKEN}"

        assert SPOKEN not in pocket._safe(Boom())
        assert pocket._safe(Boom()) == "Boom"

    def test_the_workers_own_refusals_survive_because_they_are_already_safe(self):
        from pocket_worker import worker as pocket

        assert pocket._safe(
            pocket.Refusal("That voice's prepared data is missing.")) == \
            "That voice's prepared data is missing."

    def test_a_library_value_error_does_not_pass_for_one_of_the_workers(self):
        """``ValueError`` is the base of much of the numeric stack, so it is not
        evidence that this worker wrote the message. Torch and NumPy raise it
        for a shape; ``json`` raises a subclass of it and names the document."""
        from pocket_worker import worker as pocket

        leaky = ValueError(f"could not parse /home/someone/clones/abc.wav: {SPOKEN}")
        assert pocket._safe(leaky) == "ValueError"
        assert SPOKEN not in pocket._safe(leaky)

    def test_the_runtime_turns_a_class_name_into_a_sentence_with_no_content(self):
        import mc_voice_pocket_runtime as pocket_runtime

        for reason in ("RuntimeError", "OSError", "MemoryError", "something unexpected"):
            found = pocket_runtime._readable(reason)
            assert SPOKEN not in found
            assert DICTATED not in found
            assert found

    def test_the_drain_metrics_are_numbers_and_enumerations_only(self, host):
        """Section 36's allowed list gained six fields for this engine. Every
        one of them is a duration, a count or a mode name."""
        import mc_voice_turn as turns

        turn = turns.create(voice_id="pocket:clone:abc", engine="pocket",
                            interrupt_mode="drain_unit")
        turn.add_text(f"Hello there, {SPOKEN}, and that is the end of the sentence. ")
        turn.cancel("user")
        turn.interrupting(chars=len(SPOKEN), audio_ms=300)
        turn.interrupted()
        found = turn.metrics()
        assert SPOKEN not in repr(found)
        assert found["interrupt_mode"] == "drain_unit"
        for name in ("stop_to_silence_ms", "stop_to_ready_ms", "interrupted_unit_chars",
                     "interrupted_unit_audio_ms", "discarded_chunks"):
            value = found[name]
            assert value is None or isinstance(value, int), (name, value)

    def test_the_configuration_line_names_settings_and_nothing_else(self):
        import mc_voice_pocket_runtime as pocket_runtime

        found = pocket_runtime.config_line()
        assert SPOKEN not in found
        assert "/" not in found or "torch.set_num_threads" in found

    def test_no_credential_can_be_written_anywhere(self):
        """I-PKT-21. The token is read from the environment by the installer and
        never persisted -- so the check is that nothing *writes* one, which is a
        property of the source rather than of a run."""
        import re

        root = Path(__file__).resolve().parent.parent
        names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
        for path in (root / "mc_voice_pocket.py", root / "mc_voice_pocket_runtime.py",
                     root / "pocket_worker" / "worker.py"):
            source = path.read_text(encoding="utf-8")
            for name in names:
                for line in source.splitlines():
                    if name not in line:
                        continue
                    # Reading, naming in prose, and *removing* are all fine. A
                    # write, a log line or a payload is not.
                    assert not re.search(r"(_remember|_write_json|logger\.|json\.dumps)",
                                         line), (path.name, line)

    def test_the_worker_environment_carries_no_credential(self):
        import mc_voice_pocket as pocket

        environ = pocket.worker_environment()
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            assert name not in environ

    def test_a_status_payload_carries_no_path_and_no_token(self, host, voice_root):
        import json

        import mc_voice_pocket as pocket
        import mc_voice_paths as paths

        text = json.dumps(pocket.public_status())
        assert str(paths.data_root()) not in text
        assert "hf_" not in text
        assert ".safetensors" not in text
