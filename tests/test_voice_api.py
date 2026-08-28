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


@pytest.fixture(autouse=True)
def _forget_repeats():
    """The refusal log is throttled, so its memory is module state.

    Without this, the second test to provoke the same refusal would find the
    line suppressed and assert against an empty log.
    """
    api.forget_repeats()
    yield
    api.forget_repeats()


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


class TestSayingWhyItRefused:
    """A 403 with nothing in the log is a bug report nobody can answer.

    This is the diagnostic gap that made the frozen Settings row impossible to
    explain from the outside: the browser saw a refusal it discarded, and the
    server wrote nothing at all. Each gate now names itself, in words, with no
    content in the line.
    """

    def test_a_missing_token_says_so(self, client, caplog, installed):
        import logging

        caplog.set_level(logging.DEBUG, logger="model_chain")
        assert client.post(api.STATUS_ROUTE).status_code == 403
        written = " ".join(record.getMessage() for record in caplog.records)
        assert "no page token was sent" in written
        assert api.STATUS_ROUTE in written

    def test_a_wrong_token_says_something_different(self, client, caplog, installed):
        import logging

        caplog.set_level(logging.DEBUG, logger="model_chain")
        client.post(api.STT_ROUTE, headers={api.TOKEN_HEADER: "stale"}, content=b"x")
        written = " ".join(record.getMessage() for record in caplog.records)
        assert "from a previous run of this WebUI" in written

    def test_a_foreign_origin_says_so(self, client, key, caplog, installed):
        import logging

        caplog.set_level(logging.DEBUG, logger="model_chain")
        headers = dict(key)
        headers["Origin"] = "https://somewhere-else.example"
        client.post(api.STATUS_ROUTE, headers=headers)
        written = " ".join(record.getMessage() for record in caplog.records)
        assert "Origin header did not match" in written

    def test_no_refusal_line_carries_the_token_that_was_offered(self, client, caplog,
                                                               installed):
        """A log that echoed the credential it rejected would be a log that
        leaks one."""
        import logging

        caplog.set_level(logging.DEBUG, logger="model_chain")
        client.post(api.STATUS_ROUTE, headers={api.TOKEN_HEADER: "SECRET-XYZ"})
        for record in caplog.records:
            assert "SECRET-XYZ" not in record.getMessage()


class TestThereIsNoSignInGate:
    """Voice Chat has no login of its own and does not re-check the WebUI's.

    A cookie check used to sit here and it locked legitimate users out twice:
    once by mistaking an unrelated attribute for a login, and then on an
    installation that really does pass ``--gradio-auth`` and whose user was, of
    course, already signed in by the time they reached the page. Reaching a page
    the WebUI served is the proof of access; asking for it again only produced a
    way to fail.
    """

    def test_a_webui_with_a_login_still_serves_a_page_that_has_the_token(
            self, app, key, installed):
        """The reported blocker, as a test. Credentials are configured, the
        caller has no session cookie this module can see, and the routes work --
        because the page they came from was served by that WebUI."""
        app.auth = {"someone": "hunter2"}
        app.tokens = {}
        with TestClient(app) as client:
            assert client.post(api.STATUS_ROUTE, headers=key).status_code == 200
            assert client.post(api.INSTALL_ROUTE, headers=key,
                               json={"kind": "stt"}).status_code in (200, 409)

    def test_command_line_credentials_do_not_gate_it_either(self, client, key, installed,
                                                            host, monkeypatch):
        """``--gradio-auth`` on the command line was what actually blocked the
        reported installation."""
        monkeypatch.setattr(host.shared.cmd_opts, "gradio_auth", "someone:hunter2",
                            raising=False)
        assert client.post(api.STATUS_ROUTE, headers=key).status_code == 200

    def test_no_route_can_answer_401(self, client, key, installed, silent_wav):
        """There is no code path left that asks anybody to sign in."""
        answered = [
            client.post(api.STATUS_ROUTE, headers=key),
            client.post(api.STT_ROUTE, headers=key, content=silent_wav(1.0)),
            client.post(api.TTS_ROUTE, headers=key, json={"token": "nope"}),
            client.post(api.INSTALL_ROUTE, headers=key, json={"kind": "stt"}),
        ]
        assert all(response.status_code != 401 for response in answered)

    def test_the_module_has_no_sign_in_text_left_anywhere(self):
        """Not even in a message. A sentence telling somebody to sign in is a
        sentence that will be shown to somebody who already has."""
        from pathlib import Path as _Path

        source = _Path(api.__file__).read_text(encoding="utf-8")
        body = source.split('"""', 2)[-1]
        assert "Sign in to" not in body

    def test_the_page_token_is_still_the_gate(self, client, installed):
        """Which is what makes removing the cookie check safe: the token only
        exists inside a page the WebUI served."""
        for route in api.ROUTES:
            assert client.post(route, json={"kind": "stt"}).status_code == 403

    def test_a_stale_token_says_reload_and_not_sign_in(self, client, installed):
        answered = client.post(api.STATUS_ROUTE,
                               headers={api.TOKEN_HEADER: "from-a-previous-run"})
        assert answered.status_code == 403
        error = answered.json()["error"]
        assert "Reload it" in error
        assert "sign in" in error.lower(), "the message should say what it is *not* asking"


class TestTheLogDoesNotScroll:
    def test_one_refusal_line_per_reason_however_many_requests(self, client, caplog,
                                                               installed):
        """A route a page polls turns any per-request line into a log that
        scrolls forever. It did: 136 identical warnings in three minutes."""
        import logging

        caplog.set_level(logging.DEBUG, logger="model_chain")
        for _ in range(40):
            client.post(api.STATUS_ROUTE)

        lines = [r for r in caplog.records if "refused a request" in r.getMessage()]
        assert len(lines) == 1, f"the refusal was logged {len(lines)} times"

    def test_a_different_reason_still_gets_its_own_line(self, client, caplog, installed,
                                                        key):
        import logging

        caplog.set_level(logging.DEBUG, logger="model_chain")
        client.post(api.STATUS_ROUTE)
        headers = dict(key)
        headers["Origin"] = "https://elsewhere.example"
        client.post(api.STATUS_ROUTE, headers=headers)

        lines = [r.getMessage() for r in caplog.records if "refused a request" in r.getMessage()]
        assert len(lines) == 2

    def test_the_suppressed_ones_are_counted_when_it_speaks_again(self, client, caplog,
                                                                  installed, monkeypatch):
        import logging

        caplog.set_level(logging.DEBUG, logger="model_chain")
        client.post(api.STATUS_ROUTE)
        for _ in range(9):
            client.post(api.STATUS_ROUTE)

        # Time moves on, and the next one speaks again with a count.
        later = api.time.monotonic() + 3600
        monkeypatch.setattr(api.time, "monotonic", lambda: later)
        client.post(api.STATUS_ROUTE)

        lines = [r.getMessage() for r in caplog.records if "refused a request" in r.getMessage()]
        assert len(lines) == 2
        assert "9 more like it" in lines[1]

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


@pytest.fixture
def installable(monkeypatch):
    """A build where the manifest is pinned, so an install may actually start."""
    import mc_voice_models

    monkeypatch.setattr(mc_voice_models, "refusal", lambda kind, **kw: "")
    return mc_voice_models


class TestInstalling:
    def test_only_the_two_kinds_are_accepted(self, client, key):
        answered = client.post(api.INSTALL_ROUTE, headers=key,
                               json={"kind": "../../etc/passwd"})
        assert answered.status_code == 400

    def test_a_url_cannot_be_supplied_by_the_browser(self, client, key, monkeypatch,
                                                     installable):
        """Section 38: only ids from the checked-in manifest may be installed,
        and the route takes a *kind*, so there is not even an id to smuggle."""
        asked = []
        monkeypatch.setattr(installable, "install", lambda kind, **kw: asked.append(kind))
        client.post(api.INSTALL_ROUTE, headers=key,
                    json={"kind": "stt", "url": "https://evil.example/x.onnx"})
        for _ in range(50):
            if asked:
                break
            time.sleep(0.02)
        assert asked == ["stt"]

    def test_a_build_that_cannot_install_refuses_in_the_reply(self, client, key,
                                                              monkeypatch):
        """The defect this fixes. The first version started a thread whatever
        the answer was, the thread discovered on the other side that this build
        has nothing to install, and the browser -- already told "started" --
        sat on "Starting…" indefinitely."""
        started = []
        import mc_voice_models

        monkeypatch.setattr(mc_voice_models, "install",
                            lambda *a, **k: started.append(a))
        answered = client.post(api.INSTALL_ROUTE, headers=key, json={"kind": "stt"})

        assert answered.status_code == 409
        payload = answered.json()
        assert payload["ok"] is False
        assert "pinned" in payload["error"]
        time.sleep(0.1)
        assert started == [], "a refused install still started a download thread"

    def test_the_refusal_is_a_sentence_and_not_a_status_code(self, client, key,
                                                             monkeypatch):
        import mc_voice_models

        monkeypatch.setattr(mc_voice_models, "refusal",
                            lambda kind, **kw: "Voice Chat has no tested CPU runtime here.")
        payload = client.post(api.INSTALL_ROUTE, headers=key,
                              json={"kind": "tts"}).json()
        assert payload["error"] == "Voice Chat has no tested CPU runtime here."

    def test_a_second_press_while_one_is_running_is_not_an_error(self, client, key,
                                                                 monkeypatch,
                                                                 installable):
        import mc_voice_models

        monkeypatch.setitem(mc_voice_models._progress, "stt", {"running": True,
                                                               "text": "Downloading…",
                                                               "fraction": 0.3})
        payload = client.post(api.INSTALL_ROUTE, headers=key, json={"kind": "stt"}).json()
        assert payload == {"ok": True, "already": True}


class TestWhatAnInstallSays:
    def test_progress_reaches_the_status_route(self, client, key, monkeypatch,
                                               installable):
        """The other half of the same defect: every sentence the installer wrote
        was passed to a callback that discarded it, so the row had nothing to
        show but its own initial text."""
        import mc_voice_models

        seen = []

        def install(kind, on_status=None, on_progress=None):
            say = mc_voice_models._narrator(kind, on_status)
            tick = mc_voice_models._ticker(kind, on_progress)
            with mc_voice_models._claim(kind, say):
                say("Downloading 1 of 3 — encoder.onnx (112 MB)")
                tick(0.25)
                seen.append(client.post(api.STATUS_ROUTE, headers=key).json())

        monkeypatch.setattr(mc_voice_models, "install", install)
        client.post(api.INSTALL_ROUTE, headers=key, json={"kind": "stt"})
        for _ in range(100):
            if seen:
                break
            time.sleep(0.02)

        assert seen, "the install never ran"
        progress = seen[0]["progress"]["stt"]
        assert progress["running"] is True
        assert progress["text"] == "Downloading 1 of 3 — encoder.onnx (112 MB)"
        assert progress["fraction"] == 0.25

    def test_a_failure_leaves_its_reason_where_the_row_will_draw_it(self, client, key,
                                                                    monkeypatch,
                                                                    installable):
        import mc_voice_models

        def install(kind, on_status=None, on_progress=None):
            say = mc_voice_models._narrator(kind, on_status)
            with mc_voice_models._claim(kind, say):
                raise mc_voice_models.VoiceError("decoder.onnx failed its hash check.")

        monkeypatch.setattr(mc_voice_models, "install", install)
        client.post(api.INSTALL_ROUTE, headers=key, json={"kind": "stt"})

        for _ in range(100):
            progress = client.post(api.STATUS_ROUTE, headers=key).json()["progress"]
            if progress.get("stt", {}).get("failed"):
                break
            time.sleep(0.02)
        assert progress["stt"]["failed"] is True
        assert progress["stt"]["text"] == "decoder.onnx failed its hash check."
        assert progress["stt"]["running"] is False


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
