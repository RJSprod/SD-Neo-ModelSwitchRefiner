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
            tts_threads = 2

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
        for path in list(root.glob("mc_voice_*.py")) + [root / "voice_worker" / "worker.py"]:
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
            tts_threads = 2

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
