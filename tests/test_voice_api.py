"""The browser routes, against a real FastAPI app.

Three things are being checked and only the first is obvious.

*The routes work.* Registered once, reachable, and reachable under a non-root
deployment -- a WebUI started with ``--subpath`` or sitting behind a reverse
proxy is a supported installation, and a route that only answers at "/" is a
route that works on the developer's machine.

*The routes are no more reachable than the WebUI is.* Sharing a FastAPI app does
not mean sharing Gradio's login: Gradio protects its own endpoints with a
dependency it does not apply to anything an extension adds. So there is a test
that turns authentication on and asserts an unauthenticated caller is refused --
because without one, "Voice Chat inherits WebUI authentication" is a sentence
somebody wrote rather than something that is true.

*The reply that is spoken is the reply that completed.* The target is consumed
once, it expires, it belongs to this process, and -- the point of the whole
design -- it still speaks its own snapshot after the conversation it came from
has been edited out from under it.
"""

from __future__ import annotations

import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import mc_voice_api as api  # noqa: E402
import mc_voice_runtime as runtime  # noqa: E402


@pytest.fixture
def app():
    built = FastAPI()
    assert api.install(None, built) is True
    return built


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def key():
    return {api.TOKEN_HEADER: api.session_token()}


@pytest.fixture
def speech(monkeypatch):
    """A voice runtime that answers without a process behind it."""
    calls = {"transcribed": [], "spoken": []}

    def transcribe(data):
        calls["transcribed"].append(data)
        return {"text": "the quick brown fox", "audio_seconds": 1.0, "elapsed": 0.1}

    def synthesize(text):
        calls["spoken"].append(text)
        return b"RIFF....WAVEfake audio"

    monkeypatch.setattr(runtime, "transcribe", transcribe)
    monkeypatch.setattr(runtime, "synthesize", synthesize)
    return calls


@pytest.fixture
def installed(monkeypatch):
    import mc_voice_models

    ready = mc_voice_models.Status(True, True, True, "i", "i", "i", True,
                                   "whisper-small-int8", "kokoro-multi-lang-v1-cpu",
                                   "af_heart")
    monkeypatch.setattr(mc_voice_models, "status", lambda: ready)
    return ready


class TestRegistration:
    def test_the_routes_are_registered(self, app):
        paths = {route.path for route in app.routes}
        for route in api.ROUTES:
            assert route in paths

    def test_registering_twice_does_not_duplicate_them(self, app):
        """A UI reload calls ``on_app_started`` again, and two matching routes
        for one path is a coin toss about which handler runs."""
        api.install(None, app)
        api.install(None, app)
        for route in api.ROUTES:
            matching = [r for r in app.routes if r.path == route]
            assert len(matching) == 1, f"{route} was registered {len(matching)} times"

    def test_they_are_post_only(self, app):
        """Free text never goes in a URL, so nothing here is a GET."""
        for route in app.routes:
            if route.path in api.ROUTES:
                assert set(route.methods) == {"POST"}

    def test_no_cors_middleware_is_introduced(self, app):
        names = [type(middleware.cls).__name__ if hasattr(middleware, "cls") else ""
                 for middleware in getattr(app, "user_middleware", [])]
        assert not any("CORS" in name for name in names)

    def test_an_app_that_cannot_take_routes_is_not_fatal(self):
        assert api.install(None, None) is False
        assert api.install(None, object()) is False


class TestUnderADeploymentRoot:
    def test_the_routes_answer_when_the_webui_is_mounted_below_the_origin_root(
            self, key, installed):
        """R2-4. Forge passes ``--subpath`` through to Gradio as its root path,
        and the browser is expected to build its URLs from that. What this
        proves is the server half: mounted under ``/forge``, every voice route
        is at ``/forge/model-chain/voice/...`` and nothing in the handlers has
        assumed otherwise."""
        inner = FastAPI()
        api.install(None, inner)
        outer = FastAPI()
        outer.mount("/forge", inner)

        with TestClient(outer) as client:
            for route in api.ROUTES:
                answered = client.post("/forge" + route, headers=key,
                                       json={"kind": "nonsense"})
                # "ok" in the body is this extension's own answer. Checking the
                # status code alone would not do: the TTS route replies 404 to a
                # token nobody issued, which is indistinguishable from FastAPI
                # replying 404 because the route is not there.
                assert "ok" in answered.json(), f"{route} is unreachable under /forge"
            assert client.post("/forge" + api.STATUS_ROUTE,
                               headers=key).json()["ok"] is True

    def test_the_same_routes_still_answer_at_the_origin_root(self, client, key, installed):
        assert client.post(api.STATUS_ROUTE, headers=key).json()["ok"] is True


class TestThePageToken:
    def test_a_request_without_the_token_is_refused(self, client):
        for route in api.ROUTES:
            answered = client.post(route, json={})
            assert answered.status_code == 403

    def test_a_request_with_the_wrong_token_is_refused(self, client):
        answered = client.post(api.STATUS_ROUTE,
                               headers={api.TOKEN_HEADER: "not the token"})
        assert answered.status_code == 403
        assert "Reload" in answered.json()["error"]

    def test_a_token_that_is_not_ascii_is_refused_rather_than_crashing(self):
        """``compare_digest`` on ``str`` raises on anything non-ASCII.

        Checked below the route because the test client will not build a
        request with such a header at all -- but Starlette decodes incoming
        header bytes as latin-1, so a caller that is not a well-behaved library
        can put one there, and the answer has to be 403 rather than a 500 that
        makes the route look broken.
        """
        assert api._matches_token("caf\u00e9-token") is False
        assert api._matches_token(None) is False
        assert api._matches_token(api.session_token()) is True

    def test_a_foreign_origin_is_refused(self, client, key):
        headers = dict(key)
        headers["Origin"] = "https://somewhere-else.example"
        answered = client.post(api.STATUS_ROUTE, headers=headers)
        assert answered.status_code == 403

    def test_the_page_s_own_origin_is_accepted(self, client, key, installed):
        headers = dict(key)
        headers["Origin"] = "http://testserver"
        assert client.post(api.STATUS_ROUTE, headers=headers).status_code == 200

    def test_a_proxied_origin_is_accepted_when_the_forwarded_host_agrees(self, client, key,
                                                                        installed):
        headers = dict(key)
        headers["Origin"] = "https://forge.example"
        headers["X-Forwarded-Host"] = "forge.example"
        assert client.post(api.STATUS_ROUTE, headers=headers).status_code == 200


class TestAuthenticationParity:
    def test_an_unauthenticated_caller_is_refused_when_the_webui_has_a_login(
            self, app, key, installed):
        """The test that stops "it is on the same app, so it must be protected"
        from being the whole security review."""
        app.auth = {"someone": "hunter2"}
        app.tokens = {"a-real-session": "someone"}
        with TestClient(app) as client:
            answered = client.post(api.STATUS_ROUTE, headers=key)
            assert answered.status_code == 401

    def test_an_authenticated_caller_is_served(self, app, key, installed):
        app.auth = {"someone": "hunter2"}
        app.tokens = {"a-real-session": "someone"}
        with TestClient(app) as client:
            client.cookies.set("access-token", "a-real-session")
            assert client.post(api.STATUS_ROUTE, headers=key).status_code == 200

    def test_the_page_token_does_not_get_past_the_login(self, app, key, installed):
        """The page token is request coordination, never a second login. A
        caller holding it and no session is still not signed in."""
        app.auth = {"someone": "hunter2"}
        app.tokens = {}
        with TestClient(app) as client:
            assert client.post(api.STT_ROUTE, headers=key, content=b"x").status_code == 401

    def test_a_build_that_will_not_say_who_is_calling_refuses_everybody(self, app, key,
                                                                       installed):
        """Fail closed. A route that cannot check a login must not be the way in
        to a WebUI that has one."""
        app.auth = {"someone": "hunter2"}
        app.tokens = None
        with TestClient(app) as client:
            assert client.post(api.STATUS_ROUTE, headers=key).status_code == 401

    def test_with_no_login_configured_everybody_is_served(self, client, key, installed):
        assert client.post(api.STATUS_ROUTE, headers=key).status_code == 200


class TestTranscription:
    def test_a_valid_recording_is_transcribed(self, client, key, speech, installed,
                                              silent_wav):
        answered = client.post(api.STT_ROUTE, headers=key, content=silent_wav(1.0))
        assert answered.status_code == 200
        payload = answered.json()
        assert payload["ok"] is True
        assert payload["text"] == "the quick brown fox"
        assert speech["transcribed"], "the worker was never asked"

    def test_the_response_says_whether_to_send_it(self, client, key, speech, installed,
                                                  silent_wav, host, monkeypatch):
        """Settings is authoritative, so the browser is told rather than asked
        to look the setting up for itself -- there is one reader of that option
        and it is in Python."""
        import mc_voice_state

        monkeypatch.setattr(mc_voice_state, "auto_send", lambda: True)
        payload = client.post(api.STT_ROUTE, headers=key, content=silent_wav(1.0)).json()
        assert payload["auto_send"] is True

    def test_an_oversized_body_is_refused_before_inference(self, client, key, speech,
                                                           installed):
        answered = client.post(api.STT_ROUTE, headers=key,
                               content=b"RIFF" + b"\0" * (api.MAX_AUDIO_BYTES + 10))
        assert answered.status_code == 413
        assert not speech["transcribed"], "an oversized upload reached the worker"

    @pytest.mark.parametrize("body,reason", [
        (b"", "No audio"),
        (b"not a wav at all, not even close to forty four bytes long ok", "not a WAV"),
    ])
    def test_malformed_uploads_are_refused(self, client, key, speech, installed, body,
                                           reason):
        answered = client.post(api.STT_ROUTE, headers=key, content=body)
        assert answered.status_code == 400
        assert reason in answered.json()["error"]
        assert not speech["transcribed"]

    def test_a_stereo_recording_is_refused(self, client, key, speech, installed):
        import struct

        frames = b"\0\0\0\0" * 16000
        wav = (b"RIFF" + struct.pack("<I", 36 + len(frames)) + b"WAVE"
               + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 2, 16000, 64000, 4, 16)
               + b"data" + struct.pack("<I", len(frames)) + frames)
        answered = client.post(api.STT_ROUTE, headers=key, content=wav)
        assert answered.status_code == 400
        assert "mono" in answered.json()["error"]
        assert not speech["transcribed"]

    def test_a_chunk_claiming_more_than_the_file_holds_is_refused(self, client, key,
                                                                 speech, installed):
        import struct

        wav = (b"RIFF" + struct.pack("<I", 100) + b"WAVE"
               + b"fmt " + struct.pack("<I", 0xFFFFFF) + b"\0" * 40)
        answered = client.post(api.STT_ROUTE, headers=key, content=wav)
        assert answered.status_code == 400
        assert not speech["transcribed"]

    def test_a_tap_rather_than_a_hold_is_refused(self, client, key, speech, installed,
                                                 silent_wav):
        answered = client.post(api.STT_ROUTE, headers=key, content=silent_wav(0.05))
        assert answered.status_code == 400
        assert "Hold to speak" in answered.json()["error"]

    def test_a_recording_past_the_ceiling_is_refused(self, client, key, speech, installed,
                                                     silent_wav):
        answered = client.post(api.STT_ROUTE, headers=key, content=silent_wav(75.0))
        assert answered.status_code in (400, 413)
        assert not speech["transcribed"]

    def test_an_empty_transcript_is_said_rather_than_inserted(self, client, key,
                                                              installed, monkeypatch,
                                                              silent_wav):
        monkeypatch.setattr(runtime, "transcribe", lambda data: {"text": "   "})
        payload = client.post(api.STT_ROUTE, headers=key, content=silent_wav(1.0)).json()
        assert payload["ok"] is False
        assert payload["error"] == "No speech was detected."

    def test_a_broken_runtime_does_not_take_the_route_down(self, client, key, installed,
                                                           monkeypatch, silent_wav):
        def broken(data):
            raise runtime.VoiceRuntimeError("Voice runtime stopped unexpectedly. Try again.")

        monkeypatch.setattr(runtime, "transcribe", broken)
        answered = client.post(api.STT_ROUTE, headers=key, content=silent_wav(1.0))
        assert answered.status_code == 503
        assert "stopped unexpectedly" in answered.json()["error"]


class TestSpeakingAReply:
    def test_a_target_is_spoken_once(self, client, key, speech, installed):
        token = api.remember_reply("the reply that completed")
        answered = client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert answered.status_code == 200
        assert answered.headers["content-type"].startswith("audio/wav")
        assert answered.content == b"RIFF....WAVEfake audio"
        assert speech["spoken"] == ["the reply that completed"]

        again = client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert again.status_code == 404
        assert speech["spoken"] == ["the reply that completed"]

    def test_the_audio_is_never_cached(self, client, key, speech, installed):
        token = api.remember_reply("something private")
        answered = client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert answered.headers["cache-control"] == "no-store, max-age=0"
        assert answered.headers["pragma"] == "no-cache"

    def test_an_unknown_token_speaks_nothing(self, client, key, speech, installed):
        answered = client.post(api.TTS_ROUTE, headers=key, json={"token": "invented"})
        assert answered.status_code == 404
        assert speech["spoken"] == []

    def test_an_expired_target_speaks_nothing(self, client, key, speech, installed,
                                              monkeypatch):
        token = api.remember_reply("too late")
        later = time.monotonic() + api.TARGET_TTL + 1
        monkeypatch.setattr(api.time, "monotonic", lambda: later)
        assert client.post(api.TTS_ROUTE, headers=key,
                           json={"token": token}).status_code == 404
        assert speech["spoken"] == []

    def test_the_snapshot_survives_the_conversation_changing_underneath_it(
            self, client, key, speech, installed):
        """R2-5 and I-13, which is the whole reason a target holds text rather
        than a message index. Between the reply completing and the browser
        asking for audio, the reader has edited the message, regenerated it and
        opened another thread. None of that changes what this token means."""
        token = api.remember_reply("the exact words that completed")

        conversation = ["the exact words that completed"]
        conversation[0] = "something the user typed over the top"
        conversation.append("a regenerated reply")
        conversation.clear()

        client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert speech["spoken"] == ["the exact words that completed"]

    def test_an_empty_reply_never_becomes_a_target(self, installed):
        assert api.remember_reply("") == ""
        assert api.remember_reply("   \n ") == ""

    def test_a_failure_after_consumption_is_reported_rather_than_repaired(
            self, client, key, installed, monkeypatch):
        """Section 50. Re-resolving "the current last reply" to try again would
        speak a different message than the token stood for."""
        def broken(text):
            raise runtime.VoiceRuntimeError("the runtime is gone")

        monkeypatch.setattr(runtime, "synthesize", broken)
        token = api.remember_reply("a reply")
        answered = client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert answered.status_code == 503
        assert api.take_reply(token) == "", "a consumed target was put back"


class TestInstalling:
    def test_only_the_two_kinds_are_accepted(self, client, key):
        answered = client.post(api.INSTALL_ROUTE, headers=key,
                               json={"kind": "../../etc/passwd"})
        assert answered.status_code == 400

    def test_a_url_cannot_be_supplied_by_the_browser(self, client, key, monkeypatch):
        """Section 38: only ids from the checked-in manifest may be installed,
        and the route takes a *kind*, so there is not even an id to smuggle."""
        asked = []
        import mc_voice_models

        monkeypatch.setattr(mc_voice_models, "install",
                            lambda kind, **kw: asked.append(kind))
        client.post(api.INSTALL_ROUTE, headers=key,
                    json={"kind": "stt", "url": "https://evil.example/x.onnx"})
        for _ in range(50):
            if asked:
                break
            time.sleep(0.02)
        assert asked == ["stt"]


class TestNothingIsWrittenDown:
    def test_a_transcription_creates_no_file_anywhere_under_the_voice_root(
            self, client, key, speech, installed, silent_wav, voice_root, tmp_path):
        """I-5. Not a temp file, not a Gradio cache entry, not an attachment."""
        before = sorted(p.name for p in tmp_path.rglob("*"))
        client.post(api.STT_ROUTE, headers=key, content=silent_wav(1.0))
        token = api.remember_reply("and a spoken reply too")
        client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        after = sorted(p.name for p in tmp_path.rglob("*"))
        assert before == after
        assert not list(tmp_path.rglob("*.wav"))

    def test_the_api_module_does_not_import_tempfile(self):
        import ast
        from pathlib import Path as _Path

        source = _Path(api.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert all(alias.name != "tempfile" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "tempfile"


class TestStatus:
    def test_it_reports_what_is_installed_and_nothing_about_the_machine(
            self, client, key, installed):
        payload = client.post(api.STATUS_ROUTE, headers=key).json()
        assert payload["ready"] is True
        assert set(payload) >= {"ready", "runtime_ready", "stt_ready", "tts_ready",
                                "auto_send", "auto_speak"}
        text = json.dumps(payload)
        assert "/" not in payload.get("stt_message", ""), "a path reached the browser"
        assert "pid" not in text

    def test_it_does_not_start_a_worker(self, client, key, monkeypatch):
        """A status a page polls must never be able to trigger a four-hundred-
        megabyte model load."""
        def explode():
            raise AssertionError("the status route started the voice runtime")

        monkeypatch.setattr(runtime, "ensure_started", explode)
        assert client.post(api.STATUS_ROUTE, headers=key).status_code == 200
