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
def _no_real_installs(tmp_path, monkeypatch):
    """No test in this file may download anything or write outside tmp_path.

    Both halves of that are load-bearing and both were learned the hard way: a
    test that posted to the install route without stubbing the installer started
    a real thread, fetched eighty-five megabytes of runtime, and left it in the
    working tree. A test that wants an install to run stubs it and gets its own
    behaviour; every other test gets an explosion rather than somebody's
    bandwidth.
    """
    import mc_voice_models
    import mc_voice_paths

    monkeypatch.setattr(mc_voice_paths, "data_root", lambda: tmp_path / "voice")

    def refuse(*args, **kwargs):
        raise AssertionError("a test started a real Voice Chat install")

    monkeypatch.setattr(mc_voice_models, "install", refuse)
    monkeypatch.setattr(mc_voice_models, "install_from", refuse)
    monkeypatch.setattr(mc_voice_models, "install_runtime", refuse)


@pytest.fixture(autouse=True)
def _forget_progress():
    """Install progress is module state, and a test that starts one leaves it.

    Without this, a later test asking to install the same bundle is told it is
    already installing -- which is correct behaviour reported against the wrong
    test.
    """
    import mc_voice_models

    mc_voice_models._progress.clear()
    yield
    mc_voice_models._progress.clear()


@pytest.fixture(autouse=True)
def _forget_settings(host):
    """The chosen speech model and the delivery profile are host options, and a
    test that changes one leaves it changed for every test after it.

    Same hazard as the two fixtures above, one layer out: this is state in the
    host's settings store rather than in a module, and it is reset here rather
    than in each test because forgetting is silent -- the next test gets correct
    behaviour reported against the wrong starting point.
    """
    import mc_voice_models
    import mc_voice_profile

    host.shared.opts.set(mc_voice_models.OPTIONS["stt"], "")
    for option in mc_voice_profile.OPTIONS.values():
        host.shared.opts.set(option, None)
    yield


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

    def synthesize(text, sid=0, profile=None):
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
        started = []
        import mc_voice_models

        monkeypatch_install(mc_voice_models, started)
        app.auth = {"someone": "hunter2"}
        app.tokens = {}
        with TestClient(app) as client:
            assert client.post(api.STATUS_ROUTE, headers=key).status_code == 200
            assert client.post(api.INSTALL_ROUTE, headers=key,
                               json={"kind": "stt"}).status_code == 200

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
                                              spoken_wav):
        answered = client.post(api.STT_ROUTE, headers=key, content=spoken_wav(1.0))
        assert answered.status_code == 200
        payload = answered.json()
        assert payload["ok"] is True
        assert payload["text"] == "the quick brown fox"
        assert speech["transcribed"], "the worker was never asked"

    def test_the_response_says_whether_to_send_it(self, client, key, speech, installed,
                                                  spoken_wav, host, monkeypatch):
        """Settings is authoritative, so the browser is told rather than asked
        to look the setting up for itself -- there is one reader of that option
        and it is in Python."""
        import mc_voice_state

        monkeypatch.setattr(mc_voice_state, "auto_send", lambda: True)
        payload = client.post(api.STT_ROUTE, headers=key, content=spoken_wav(1.0)).json()
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
        assert "Hold at the right-hand end" in answered.json()["error"]

    def test_a_recording_past_the_ceiling_is_refused(self, client, key, speech, installed,
                                                     silent_wav):
        answered = client.post(api.STT_ROUTE, headers=key, content=silent_wav(75.0))
        assert answered.status_code in (400, 413)
        assert not speech["transcribed"]

    def test_an_empty_transcript_is_said_rather_than_inserted(self, client, key,
                                                              installed, monkeypatch,
                                                              spoken_wav):
        monkeypatch.setattr(runtime, "transcribe", lambda data: {"text": "   "})
        payload = client.post(api.STT_ROUTE, headers=key, content=spoken_wav(1.0)).json()
        assert payload["ok"] is False
        assert payload["error"] == "No speech was detected."

    def test_a_broken_runtime_does_not_take_the_route_down(self, client, key, installed,
                                                           monkeypatch, spoken_wav):
        def broken(data):
            raise runtime.VoiceRuntimeError("Voice runtime stopped unexpectedly. Try again.")

        monkeypatch.setattr(runtime, "transcribe", broken)
        answered = client.post(api.STT_ROUTE, headers=key, content=spoken_wav(1.0))
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
        def broken(text, sid=0, profile=None):
            raise runtime.VoiceRuntimeError("the runtime is gone")

        monkeypatch.setattr(runtime, "synthesize", broken)
        token = api.remember_reply("a reply")
        answered = client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert answered.status_code == 503
        assert api.take_reply(token) is None, "a consumed target was put back"


def monkeypatch_install(models_module, recorder):
    """Replace the installer with something that records and returns.

    Written as a plain function rather than a fixture because the tests that
    need it are scattered and each wants its own recorder.
    """
    models_module.install = lambda *a, **k: recorder.append(a)
    return recorder


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

    def test_this_build_not_having_pinned_a_bundle_is_not_a_refusal(self, client, key,
                                                                    monkeypatch):
        """The blocker that shipped. Whether this repository's release machine
        could reach the publishers is a fact about that machine, and it has no
        business appearing in front of somebody who has an Internet connection
        and pressed a button."""
        started = []
        import mc_voice_models

        monkeypatch.setattr(mc_voice_models, "install",
                            lambda *a, **k: started.append(a))
        answered = client.post(api.INSTALL_ROUTE, headers=key, json={"kind": "stt"})

        assert answered.status_code == 200
        assert answered.json()["ok"] is True
        for _ in range(50):
            if started:
                break
            time.sleep(0.02)
        assert started, "the install was accepted and then never started"

    def test_a_refusal_that_does_exist_is_still_in_the_reply(self, client, key,
                                                             monkeypatch):
        """Not every refusal is gone -- an unsupported platform is still one --
        and the ones that remain are answered to the caller rather than
        discovered on a background thread it cannot see."""
        started = []
        import mc_voice_models

        monkeypatch.setattr(mc_voice_models, "install", lambda *a, **k: started.append(a))
        monkeypatch.setattr(mc_voice_models, "current_platform",
                            lambda: ("haiku", "vax", "3.11"))
        answered = client.post(api.INSTALL_ROUTE, headers=key, json={"kind": "stt"})

        assert answered.status_code == 409
        assert "no tested CPU runtime" in answered.json()["error"]
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

        def install(kind, on_status=None, on_progress=None,
                    identifier=""):
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

        def install(kind, on_status=None, on_progress=None,
                    identifier=""):
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
            self, client, key, speech, installed, spoken_wav, voice_root, tmp_path):
        """I-5. Not a temp file, not a Gradio cache entry, not an attachment."""
        before = sorted(p.name for p in tmp_path.rglob("*"))
        client.post(api.STT_ROUTE, headers=key, content=spoken_wav(1.0))
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


# --------------------------------------------------------------------------- #
# Streaming speech, cancelling it, and the engine
# --------------------------------------------------------------------------- #


class Speaker:
    """A worker stand-in that produces a fixed amount of PCM per segment."""

    def __init__(self, blocks: int = 2, samples: int = 1200, rate: int = 24000):
        self.blocks, self.samples, self.rate = blocks, samples, rate
        self.segments = []
        self.cancelled = []

    def begin_turn(self, turn, sid, speed=1.0):
        turn.sample_rate = self.rate
        return self.rate

    def send_segment(self, turn, text):
        self.segments.append(text)
        for _block in range(self.blocks):
            turn.offer_audio(b"\x01\x00" * self.samples, self.rate)

    def finish_turn(self, turn):
        turn.audio_finished()

    def cancel_turn(self, turn):
        self.cancelled.append(turn.id)


@pytest.fixture
def spoken_turn():
    """One turn, already fed with two sentences and completed."""
    import mc_voice_turn as turns

    speaker = Speaker()
    turn = turns.create(voice_id="official:af_heart", sid=3, speaker=speaker)
    turn.start()
    turn.add_text("Hello there, this is the first sentence of the reply. ")
    turn.complete("Hello there, this is the first sentence of the reply. And a second one.")
    yield turn, speaker
    turns.forget_all("test finished")


class TestTheStream:
    def test_t_api_4_it_streams_without_a_content_length(self, client, key, spoken_turn):
        """Section 19. A live body has no length, and an intermediary given one
        would wait for it -- which is how streaming quietly becomes buffering."""
        turn, _speaker = spoken_turn
        with client.stream("POST", api.STREAM_ROUTE, headers=key,
                           json={"turn": turn.id}) as response:
            assert response.status_code == 200
            assert "content-length" not in {name.lower() for name in response.headers}
            total = sum(len(chunk) for chunk in response.iter_bytes())
        assert total > 0
        assert total % 2 == 0, "a stream ended mid-sample"

    def test_t_api_5_the_headers_say_no_store_no_transform_and_no_buffering(
            self, client, key, spoken_turn):
        turn, _speaker = spoken_turn
        with client.stream("POST", api.STREAM_ROUTE, headers=key,
                           json={"turn": turn.id}) as response:
            headers = response.headers
            assert "no-store" in headers["cache-control"]
            assert "no-transform" in headers["cache-control"]
            assert headers["x-accel-buffering"] == "no"
            assert headers["x-model-chain-voice-rate"] == "24000"
            assert headers["x-model-chain-voice-turn"] == turn.id
            response.read()

    def test_the_body_is_raw_pcm_and_not_a_container(self, client, key, spoken_turn):
        turn, _speaker = spoken_turn
        with client.stream("POST", api.STREAM_ROUTE, headers=key,
                           json={"turn": turn.id}) as response:
            body = b"".join(response.iter_bytes())
        assert body[:4] != b"RIFF"
        assert response.headers["content-type"] == "application/octet-stream"

    def test_an_unknown_turn_is_refused_rather_than_synthesised(self, client, key):
        assert client.post(api.STREAM_ROUTE, headers=key,
                           json={"turn": "not-a-turn"}).status_code == 404

    def test_t_api_6_a_client_that_goes_away_cancels_the_turn(self, client, key,
                                                              spoken_turn):
        """The one condition in which the worker's bounded queue never drains."""
        turn, _speaker = spoken_turn
        with client.stream("POST", api.STREAM_ROUTE, headers=key,
                           json={"turn": turn.id}) as response:
            next(response.iter_bytes())
        for _attempt in range(100):
            if turn.cancelled.is_set():
                break
            time.sleep(0.02)
        assert turn.cancelled.is_set()

    def test_no_assistant_text_is_ever_in_a_request(self, client, key, spoken_turn):
        """Section 19: never in the URL, never in a header, never in the body."""
        turn, _speaker = spoken_turn
        with client.stream("POST", api.STREAM_ROUTE, headers=key,
                           json={"turn": turn.id}) as response:
            response.read()
        assert "Hello there" not in turn.id
        assert api.STREAM_ROUTE.count("{") == 0


class TestCancelling:
    def test_it_cancels_only_the_turn_it_names(self, client, key):
        import mc_voice_turn as turns

        turns.create(sid=0, speaker=Speaker())
        second = turns.create(sid=0, speaker=Speaker())
        found = client.post(api.CANCEL_ROUTE, headers=key, json={"turn": second.id}).json()
        assert found == {"ok": True, "cancelled": True, "speaking": False}
        assert second.cancelled.is_set()
        turns.forget_all("test")

    def test_cancelling_twice_is_defined_and_harmless(self, client, key):
        """Section 26: two paths deliberately press Stop."""
        import mc_voice_turn as turns

        turn = turns.create(sid=0, speaker=Speaker())
        assert client.post(api.CANCEL_ROUTE, headers=key,
                           json={"turn": turn.id}).json()["cancelled"] is True
        assert client.post(api.CANCEL_ROUTE, headers=key,
                           json={"turn": turn.id}).json()["cancelled"] is False
        turns.forget_all("test")

    def test_a_stale_token_is_not_an_error(self, client, key):
        found = client.post(api.CANCEL_ROUTE, headers=key, json={"turn": "gone"}).json()
        assert found["ok"] is True and found["cancelled"] is False

    def test_it_does_not_stop_the_whole_runtime(self, client, key, monkeypatch):
        """Section 27. Cancelling a reply happens several times in a
        conversation and must not cost a model reload."""
        def explode(*args, **kwargs):
            raise AssertionError("cancelling a turn stopped the voice runtime")

        monkeypatch.setattr(runtime, "stop", explode)
        client.post(api.CANCEL_ROUTE, headers=key, json={"turn": "anything"})


class TestTheTelemetryRoute:
    """The least powerful route in the module, and deliberately so.

    It exists because the browser is the only party that can say whether the
    speaker actually ran dry, and it writes what it is told into a log file --
    which is precisely why what it will read is a fixed list of numbers rather
    than whatever arrives.
    """

    def test_a_playback_report_is_recorded(self, client, key, caplog):
        with caplog.at_level("INFO", logger="model_chain"):
            found = client.post(api.TELEMETRY_ROUTE, headers=key, json={
                "kind": "playback", "turn": "abcd1234",
                "turn_seen_to_headers_ms": 40, "underrun_count": 2,
                "max_underrun_gap_ms": 410, "total_underrun_gap_ms": 700,
                "startup_buffer_ms": 700, "playback_end_reason": "finished",
            }).json()
        assert found["ok"] is True and found["recorded"] == 6
        line = " ".join(record.getMessage() for record in caplog.records)
        assert "max_underrun_gap_ms=410" in line
        assert "playback_end_reason=finished" in line

    def test_a_capture_report_is_recorded(self, client, key, caplog):
        with caplog.at_level("INFO", logger="model_chain"):
            found = client.post(api.TELEMETRY_ROUTE, headers=key, json={
                "kind": "capture", "first_pcm_ms": 285, "engaged_ms": 620,
                "preroll_ms": 335, "graph": "worklet", "result": "sent",
            }).json()
        assert found["ok"] is True
        assert "preroll_ms=335" in " ".join(r.getMessage() for r in caplog.records)

    def test_a_field_this_build_has_never_heard_of_is_ignored(self, client, key, caplog):
        """The documented policy is ignore rather than reject: a page from a
        newer build should still get its other numbers recorded."""
        with caplog.at_level("INFO", logger="model_chain"):
            found = client.post(api.TELEMETRY_ROUTE, headers=key, json={
                "kind": "playback", "underrun_count": 1,
                "something_from_the_future_ms": 12,
            }).json()
        assert found["ok"] is True and found["recorded"] == 1
        assert "something_from_the_future" not in " ".join(
            r.getMessage() for r in caplog.records)

    def test_nothing_that_could_carry_a_word_gets_through(self, client, key, caplog):
        """Every field is a duration, a count, or one of a fixed set of words.
        There is no field on this route that a sentence could be put in."""
        with caplog.at_level("INFO", logger="model_chain"):
            client.post(api.TELEMETRY_ROUTE, headers=key, json={
                "kind": "playback", "underrun_count": 1,
                "playback_end_reason": "the launch code is four zero four",
                "startup_buffer_ms": "seven hundred",
                "note": "the user said hello",
            })
        line = " ".join(record.getMessage() for record in caplog.records)
        assert "launch code" not in line
        assert "said hello" not in line
        assert "seven hundred" not in line

    def test_an_unknown_kind_is_refused(self, client, key):
        assert client.post(api.TELEMETRY_ROUTE, headers=key,
                           json={"kind": "whatever"}).status_code == 400

    def test_a_report_with_nothing_in_it_is_refused(self, client, key):
        assert client.post(api.TELEMETRY_ROUTE, headers=key,
                           json={"kind": "playback"}).status_code == 400

    def test_a_turn_identifier_that_is_not_one_is_refused(self, client, key):
        assert client.post(api.TELEMETRY_ROUTE, headers=key,
                           json={"kind": "playback", "turn": "../../etc/passwd",
                                 "underrun_count": 1}).status_code == 400

    def test_an_absurd_duration_is_clamped_rather_than_written_out(self, client, key,
                                                                  caplog):
        with caplog.at_level("INFO", logger="model_chain"):
            client.post(api.TELEMETRY_ROUTE, headers=key, json={
                "kind": "playback", "max_underrun_gap_ms": 10 ** 30})
        line = " ".join(record.getMessage() for record in caplog.records)
        assert str(api.TELEMETRY_MAX_MS) in line

    def test_a_body_bigger_than_a_report_is_refused_before_it_is_parsed(self, client, key):
        assert client.post(api.TELEMETRY_ROUTE, headers=key,
                           content=b"x" * (api.MAX_JSON_BYTES + 1)).status_code == 413


class TestTheEngineRoute:
    def test_t_api_9_load_and_unload_go_through_the_route(self, client, key, monkeypatch):
        loaded = []
        monkeypatch.setattr(runtime, "load", lambda: loaded.append("load") or {"loaded": True})
        monkeypatch.setattr(runtime, "unload",
                            lambda reason="": loaded.append("unload") or {"loaded": False})
        # ``engine_state``, not ``engine``. ``engine`` has been the selected
        # engine's *id* in every payload since the scoping filter was written;
        # this assertion was left behind by that change and had been red ever
        # since, invisibly, because it only runs where FastAPI is installed.
        assert client.post(api.RUNTIME_ROUTE, headers=key,
                           json={"action": "load"}).json()["engine_state"]["loaded"] is True
        assert client.post(api.RUNTIME_ROUTE, headers=key,
                           json={"action": "unload"}).json()["engine_state"]["loaded"] \
            is False
        assert loaded == ["load", "unload"]

    def test_an_unknown_action_is_refused(self, client, key):
        assert client.post(api.RUNTIME_ROUTE, headers=key,
                           json={"action": "explode"}).status_code == 400

    def test_a_load_that_fails_says_why(self, client, key, monkeypatch):
        def refuse():
            raise runtime.VoiceRuntimeError("Voice Chat is not set up.")

        monkeypatch.setattr(runtime, "load", refuse)
        response = client.post(api.RUNTIME_ROUTE, headers=key, json={"action": "load"})
        assert response.status_code == 503
        assert "not set up" in response.json()["error"]

    def test_the_status_route_carries_live_engine_state(self, client, key, installed):
        found = client.post(api.STATUS_ROUTE, headers=key).json()
        # ``engine`` is the id; ``engine_state`` is the residency object the
        # Voice flyout draws. One key, one meaning -- and this assertion still
        # named the old one.
        assert found["engine"] == "kokoro"
        assert set(found["engine_state"]) >= {"loaded", "state", "provider"}
        assert found["engine_state"]["loaded"] is False
        assert "pid" not in json.dumps(found["engine"])


class TestVoiceManagement:
    def test_t_api_10_the_browser_is_never_told_a_speaker_number(self, client, key,
                                                                 voice_registry,
                                                                 kokoro_bundle):
        """Section 56. A number the browser knew could address a reserved slot,
        and the answer to that is not to validate it harder."""
        found = client.post(api.VOICES_ROUTE, headers=key, json={}).json()
        assert found["voices"]
        assert all("sid" not in entry and "slot" not in entry for entry in found["voices"])

    def test_the_list_groups_official_voices_by_accent(self, client, key, voice_registry,
                                                       kokoro_bundle):
        found = client.post(api.VOICES_ROUTE, headers=key, json={}).json()
        accents = {entry["accent"] for entry in found["voices"]}
        assert accents == {"American English", "British English"}
        # Backend-qualified, as every id in a payload has been since there was
        # more than one engine to own one.
        assert found["default"] == "kokoro:official:af_heart"

    def test_setting_a_default_takes_and_is_reported_back(self, client, key,
                                                          voice_registry, kokoro_bundle):
        found = client.post(api.VOICE_DEFAULT_ROUTE, headers=key,
                            json={"voice": "official:bf_emma"}).json()
        assert found["default"] == "kokoro:official:bf_emma"

    def test_renaming_or_deleting_an_official_voice_is_refused(self, client, key,
                                                               voice_registry,
                                                               kokoro_bundle):
        assert client.post(api.VOICE_RENAME_ROUTE, headers=key,
                           json={"voice": "official:af_heart",
                                 "display_name": "Mine"}).status_code == 400
        assert client.post(api.VOICE_DELETE_ROUTE, headers=key,
                           json={"voice": "official:af_heart"}).status_code == 400

    def test_a_test_goes_through_the_ordinary_runtime_at_the_resolved_speaker(
            self, client, key, voice_registry, kokoro_bundle):
        """Section 45. A Test down a different path could pass for a voice that
        cannot actually be spoken in a reply."""
        response = client.post(api.VOICE_TEST_ROUTE, headers=key,
                               json={"voice": "official:bf_emma", "text": "Hello."})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert "no-store" in response.headers["cache-control"]
        assert voice_registry.spoken[-1] == 21

    def test_t_api_11_the_test_text_is_bounded_and_persisted(self, client, key,
                                                             voice_registry,
                                                             kokoro_bundle):
        client.post(api.VOICE_TEST_ROUTE, headers=key,
                    json={"voice": "official:af_heart", "text": "x" * 5000})
        found = client.post(api.VOICES_ROUTE, headers=key, json={}).json()
        assert len(found["test_text"]) == voice_registry.MAX_TEST_CHARS

    def test_a_test_for_a_voice_that_is_not_installed_is_refused(self, client, key,
                                                                 voice_registry,
                                                                 kokoro_bundle,
                                                                 monkeypatch):
        monkeypatch.setattr(voice_registry, "resolve",
                            lambda voice_id="": (_ for _ in ()).throw(
                                voice_registry.RegistryError("No voice is installed.")))
        assert client.post(api.VOICE_TEST_ROUTE, headers=key,
                           json={"voice": "clone:gone"}).status_code == 404


class TestTheNewRoutesAreGuarded:
    NEW = ("STREAM_ROUTE", "CANCEL_ROUTE", "TELEMETRY_ROUTE", "RUNTIME_ROUTE", "VOICES_ROUTE",
           "VOICE_DEFAULT_ROUTE", "VOICE_TEST_ROUTE", "VOICE_RENAME_ROUTE",
           "VOICE_DELETE_ROUTE", "CLONING_INSTALL_ROUTE", "CLONING_STATUS_ROUTE",
           "CLONING_START_ROUTE", "CLONING_ABORT_ROUTE")

    def test_t_api_2_every_new_route_needs_the_page_token(self, client):
        for name in self.NEW:
            route = getattr(api, name)
            assert client.post(route, json={}).status_code == 403, route

    def test_t_api_3_every_new_route_refuses_a_foreign_origin(self, client, key):
        headers = dict(key)
        headers["Origin"] = "https://somewhere-else.example"
        for name in self.NEW:
            route = getattr(api, name)
            assert client.post(route, headers=headers, json={}).status_code == 403, route

    def test_t_api_1_the_routes_register_once(self, installed):
        app = fastapi.FastAPI()
        assert api.install(app=app) is True
        assert api.install(app=app) is True
        paths = [route.path for route in app.routes if route.path.startswith(api.PREFIX)]
        assert len(paths) == len(set(paths)) == len(api.ROUTES)

    def test_a_body_bigger_than_a_name_is_refused_before_it_is_parsed(self, client, key):
        assert client.post(api.VOICE_RENAME_ROUTE, headers=key,
                           content=b"x" * (api.MAX_JSON_BYTES + 1)).status_code == 413


class TestNothingBlocksTheEventLoop:
    def test_t_api_7_a_long_synthesis_does_not_block_the_status_route(self, client, key,
                                                                     installed,
                                                                     monkeypatch):
        """Section 17. The V1 handlers were ``async def`` and called straight
        into a runtime that waits for Whisper, so one dictation froze every
        other request in the WebUI -- including the poll drawing the microphone
        that was doing it."""
        import threading

        release = threading.Event()

        def slow(text, sid=0, speed=1.0):
            release.wait(10)
            return b"RIFF" + b"\x00" * 100

        monkeypatch.setattr(runtime, "synthesize", slow)
        token = api.remember_reply("something long")
        speaking = threading.Thread(
            target=lambda: client.post(api.TTS_ROUTE, headers=key, json={"token": token}),
            daemon=True)
        speaking.start()
        time.sleep(0.2)
        try:
            started = time.monotonic()
            assert client.post(api.STATUS_ROUTE, headers=key).status_code == 200
            assert time.monotonic() - started < 3.0, "status waited for the synthesis"
        finally:
            release.set()
            speaking.join(timeout=10)

    def test_t_api_8_a_long_transcription_does_not_block_cancel_or_runtime(
            self, client, key, installed, monkeypatch, spoken_wav):
        import threading

        release = threading.Event()

        def slow(data):
            release.wait(10)
            return {"text": "eventually"}

        monkeypatch.setattr(runtime, "transcribe", slow)
        listening = threading.Thread(
            target=lambda: client.post(api.STT_ROUTE, headers=key, content=spoken_wav(1.0)),
            daemon=True)
        listening.start()
        time.sleep(0.2)
        try:
            started = time.monotonic()
            assert client.post(api.CANCEL_ROUTE, headers=key, json={}).status_code == 200
            assert time.monotonic() - started < 3.0, "cancel waited for the transcription"
        finally:
            release.set()
            listening.join(timeout=10)

    def test_the_handlers_hand_their_blocking_half_to_a_thread(self):
        """Structural, so that a handler added later without ``_offload`` is
        visible rather than merely slow."""
        import inspect

        source = inspect.getsource(api.install)
        for call in ("status_payload", "transcribe", "voices_payload", "cloning_payload",
                     "set_runtime", "test_voice", "open_stream"):
            assert f"_offload({call}" in source, call


class TestCloningRoutes:
    def test_the_status_route_answers_without_cloning_installed(self, client, key,
                                                                voice_root):
        found = client.post(api.CLONING_STATUS_ROUTE, headers=key, json={}).json()
        assert found["ok"] is True
        assert found["state"] in ("not_installed", "unsupported", "invalid")
        assert found["job"]["status"] == "idle"

    def test_a_one_click_install_with_nothing_pinned_says_so(self, client, key,
                                                             voice_root):
        response = client.post(api.CLONING_INSTALL_ROUTE, headers=key, json={})
        assert response.status_code == 409
        assert "pinned" in response.json()["error"] or "not offered" in response.json()["error"]

    def test_starting_a_clone_without_a_recording_is_refused(self, client, key,
                                                             voice_root):
        """Through the base64 fallback, which is the path a host without
        ``python-multipart`` takes -- and the one a test can drive without
        depending on a parser that may not be installed."""
        response = client.post(api.CLONING_START_ROUTE, headers=key,
                               json={"name": "Alice", "language": "en-US", "reference": ""})
        assert response.status_code in (400, 409)
        assert response.json()["error"]

    def test_a_reference_bigger_than_the_ceiling_never_reaches_a_file(self, client, key,
                                                                     voice_root):
        response = client.post(api.CLONING_START_ROUTE, headers=key,
                               content=b"x" * (api.MAX_REFERENCE_BYTES * 2 + 1))
        assert response.status_code == 413

    def test_aborting_when_nothing_runs_is_harmless(self, client, key, voice_root):
        assert client.post(api.CLONING_ABORT_ROUTE, headers=key, json={}).status_code == 200


class TestTheModelRoute:
    """Three speech-to-text qualities, listed and chosen through one route.

    Listing and choosing are the same call so the row that draws the three
    cards redraws from the truth in one round trip rather than from what it
    hoped it had set.
    """

    def test_it_lists_the_tiers_with_what_is_on_disk_beside_them(self, client, key,
                                                                 voice_root):
        payload = client.post(api.MODELS_ROUTE, headers=key, json={"kind": "stt"}).json()
        assert payload["ok"] is True
        assert [entry["tier"] for entry in payload["models"]] == ["low", "medium", "high"]
        assert payload["chosen"] == "whisper-small-int8"
        assert all("installed" in entry and "about_label" in entry
                   for entry in payload["models"])

    def test_choosing_one_answers_with_the_new_state(self, client, key, voice_root):
        payload = client.post(api.MODELS_ROUTE, headers=key,
                              json={"kind": "stt",
                                    "select": "whisper-medium-int8"}).json()
        assert payload["chosen"] == "whisper-medium-int8"
        assert [entry["chosen"] for entry in payload["models"]] == [False, False, True]

    def test_choosing_one_stops_a_worker_holding_the_other(self, client, key, voice_root,
                                                           monkeypatch):
        """The handshake refuses a worker whose models are not the ones this
        installation verified, so a loaded worker holding the previous tier has
        to go -- or the next dictation fails that check instead of starting a
        worker on the new tier."""
        import mc_voice_runtime

        stopped = []
        monkeypatch.setattr(mc_voice_runtime, "unload",
                            lambda reason="": stopped.append(reason) or {})
        client.post(api.MODELS_ROUTE, headers=key,
                    json={"kind": "stt", "select": "whisper-base-int8"})
        assert stopped, "the worker was left holding the previous model"

    def test_choosing_the_one_already_in_use_does_not_stop_anything(self, client, key,
                                                                    voice_root,
                                                                    monkeypatch):
        import mc_voice_runtime

        stopped = []
        monkeypatch.setattr(mc_voice_runtime, "unload",
                            lambda reason="": stopped.append(reason) or {})
        client.post(api.MODELS_ROUTE, headers=key,
                    json={"kind": "stt", "select": "whisper-small-int8"})
        assert not stopped

    def test_an_unknown_id_is_refused_with_a_sentence(self, client, key, voice_root):
        answered = client.post(api.MODELS_ROUTE, headers=key,
                               json={"kind": "stt", "select": "whisper-enormous"})
        assert answered.status_code == 400
        assert answered.json()["ok"] is False

    def test_text_to_speech_is_not_a_choice(self, client, key, voice_root):
        """One Kokoro bundle. What a user varies about it is which voice and how
        it is delivered, which is the registry and the profile -- not a second
        model."""
        answered = client.post(api.MODELS_ROUTE, headers=key, json={"kind": "tts"})
        assert answered.status_code == 400

    def test_an_install_can_name_a_tier(self, client, key, installed, monkeypatch):
        import mc_voice_models

        asked = []
        monkeypatch.setattr(mc_voice_models, "install",
                            lambda kind, on_status=None, on_progress=None, identifier="":
                            asked.append((kind, identifier)))
        client.post(api.INSTALL_ROUTE, headers=key,
                    json={"kind": "stt", "model": "whisper-medium-int8"})
        for _ in range(100):
            if asked:
                break
            time.sleep(0.02)
        assert asked == [("stt", "whisper-medium-int8")]


class TestTheProfileRoute:
    """How the default voice is delivered. One route for both directions: a body
    with a profile in it writes and then answers, a body without one reads."""

    def test_reading_answers_with_the_controls_and_the_values(self, client, key):
        payload = client.post(api.PROFILE_ROUTE, headers=key, json={}).json()
        assert payload["ok"] is True
        assert payload["profile"] == {"speed": 1.0, "pitch": 0.0, "gain": 0.0, "pause": 0.0}
        assert set(payload["fields"]) == {"speed", "pitch", "gain", "pause"}
        assert payload["controls"]["pitch"]["unit"].strip() == "semitones"

    def test_writing_answers_from_the_store_rather_than_the_request(self, client, key):
        """So a value the host refused shows the slider snapping back instead of
        lying about it."""
        payload = client.post(api.PROFILE_ROUTE, headers=key,
                              json={"profile": {"speed": 99, "pitch": -3}}).json()
        assert payload["profile"]["speed"] == 2.0
        assert payload["profile"]["pitch"] == -3.0
        assert "pitch -3 semitones" in payload["summary"]

    def test_a_body_that_is_not_a_profile_is_refused(self, client, key):
        answered = client.post(api.PROFILE_ROUTE, headers=key, json={"profile": "loud"})
        assert answered.status_code == 400

    def test_an_audition_can_carry_a_delivery_that_is_not_saved_anywhere(
            self, client, key, voice_registry, kokoro_bundle):
        """The character screen's play button. Adjusting sliders against a sound
        they do not produce is not a preview."""
        answered = client.post(api.VOICE_TEST_ROUTE, headers=key,
                               json={"voice": "official:af_heart", "text": "Hello.",
                                     "profile": {"pitch": 4}})
        assert answered.status_code == 200
        stored = client.post(api.PROFILE_ROUTE, headers=key, json={}).json()
        assert stored["profile"]["pitch"] == 0.0, "an audition wrote the stored profile"


class TestARecordingWithNothingInIt:
    """The Bluetooth report, at the route. See mc_voice_hearing for the whole
    account: a headset microphone is narrowband and quiet, and what Whisper does
    with a quiet band-limited stream is describe it rather than transcribe it."""

    def test_a_silent_recording_never_reaches_the_model(self, client, key, speech,
                                                        installed, silent_wav):
        answered = client.post(api.STT_ROUTE, headers=key, content=silent_wav(1.0))
        payload = answered.json()
        assert payload["ok"] is False
        assert "Bluetooth" in payload["error"]
        assert not speech["transcribed"], "a large model was run over silence"

    def test_an_annotation_is_not_put_in_the_composer(self, client, key, installed,
                                                      spoken_wav, monkeypatch):
        """`(music)` in the composer is worse than nothing in the composer."""
        import mc_voice_runtime

        monkeypatch.setattr(mc_voice_runtime, "transcribe",
                            lambda body: {"text": "(music)", "elapsed": 0.1})
        payload = client.post(api.STT_ROUTE, headers=key,
                              content=spoken_wav(1.0)).json()
        assert payload["ok"] is False
        assert payload["text"] == ""
        assert "described the sound" in payload["error"]

    def test_words_are_still_words(self, client, key, installed, spoken_wav, monkeypatch):
        import mc_voice_runtime

        monkeypatch.setattr(mc_voice_runtime, "transcribe",
                            lambda body: {"text": "play some music", "elapsed": 0.1})
        payload = client.post(api.STT_ROUTE, headers=key,
                              content=spoken_wav(1.0)).json()
        assert payload["ok"] is True
        assert payload["text"] == "play some music"

    def test_the_level_comes_back_so_the_page_can_say_it_was_quiet(self, client, key,
                                                                   speech, installed,
                                                                   spoken_wav):
        payload = client.post(api.STT_ROUTE, headers=key,
                              content=spoken_wav(1.0)).json()
        assert 0 < payload["level"]["peak"] <= 1.0
        assert payload["level"]["seconds"] == 1.0

    def test_nothing_about_the_recording_but_numbers_is_reported(self, client, key,
                                                                 speech, installed,
                                                                 spoken_wav):
        """I-6 still holds: the level is three numbers, and no audio."""
        payload = client.post(api.STT_ROUTE, headers=key,
                              content=spoken_wav(1.0)).json()
        assert set(payload["level"]) == {"peak", "rms", "seconds"}
        assert all(isinstance(value, (int, float)) for value in payload["level"].values())


class TestTheCompletedReplyPathSpeaksAsTheRightVoice:
    """The non-streaming path used to synthesize with no speaker at all.

    sherpa answers an absent speaker with speaker 0, which in the upstream
    Kokoro map is ``af_alloy`` -- section 113's bug, still live in the one path
    that had not been looked at when the rest of it was fixed. It matters more
    now than it did: a character with a voice of its own has to keep that voice
    here too, or the fallback is the one place a character sounds like somebody
    else.
    """

    def test_the_voice_is_snapshotted_with_the_words(self, client, key, voice_registry,
                                                     kokoro_bundle):
        token = api.remember_reply("Hello there.", voice_id="official:bf_emma")
        answered = client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert answered.status_code == 200
        assert voice_registry.spoken[-1] == 21, "the reply was spoken by the wrong speaker"

    def test_a_reply_with_no_voice_uses_the_default_rather_than_speaker_zero(
            self, client, key, voice_registry, kokoro_bundle):
        token = api.remember_reply("Hello there.")
        client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert voice_registry.spoken[-1] == 3, (
            "the completed-reply path fell back to speaker 0 instead of the default voice")

    def test_a_voice_that_has_since_been_deleted_is_still_spoken(self, client, key,
                                                                 voice_registry,
                                                                 kokoro_bundle):
        """A deleted clone is not a reason for a reply that has already been
        written to go unspoken."""
        token = api.remember_reply("Hello there.", voice_id="clone:gone")
        answered = client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert answered.status_code == 200

    def test_the_delivery_is_snapshotted_too(self, client, key, voice_registry,
                                             kokoro_bundle, monkeypatch):
        import mc_voice_runtime

        seen = {}
        real = mc_voice_runtime.synthesize
        monkeypatch.setattr(mc_voice_runtime, "synthesize",
                            lambda text, sid=0, profile=None:
                            seen.update({"profile": profile}) or real(text, sid=sid))
        token = api.remember_reply("Hello there.", profile={"speed": 1.4, "pitch": -2.0,
                                                            "gain": 0.0, "pause": 0.0})
        client.post(api.TTS_ROUTE, headers=key, json={"token": token})
        assert seen["profile"]["speed"] == 1.4
        assert seen["profile"]["pitch"] == -2.0


# --------------------------------------------------------------------------- #
# The active-engine routes
# --------------------------------------------------------------------------- #


class TestTheGenericCloneAndSettingsRoutes:
    """Section 30. One set of routes for every engine that has a clone
    transaction, rather than a second parallel forest per engine."""

    def test_they_are_registered(self, app):
        paths = {route.path for route in app.routes}
        for route in api.POCKET_ROUTES:
            assert route in paths, route

    def test_sopros_own_routes_are_still_registered(self, app):
        """A transition nobody can take incrementally is not a transition. The
        browser calls these today and an external caller may still call them."""
        paths = {route.path for route in app.routes}
        for route in api.SOPRO_ROUTES:
            assert route in paths, route

    def test_a_request_naming_an_engine_that_is_not_selected_is_a_mismatch(
            self, client, key, host):
        """The one refusal a page reacts to structurally rather than by showing
        a message: its whole document belongs to an engine that is no longer
        selected (section 5)."""
        import mc_voice_engines as engines

        engines.select("kokoro")
        found = client.post(api.ENGINE_SETTINGS_ROUTE,
                            json={"engine": "pocket", "values": {"steps": 3}},
                            headers=key)
        assert found.status_code == 409
        assert found.json()["engine_mismatch"] is True

    def test_an_engine_without_the_capability_is_refused_rather_than_failing(
            self, client, key, host):
        """Kokoro has no engine settings and its cloning has a window of its
        own. Asking anyway gets a sentence and a 409, not a 500."""
        import mc_voice_engines as engines

        engines.select("kokoro")
        found = client.post(api.ENGINE_SETTINGS_ROUTE, json={"values": {"steps": 3}},
                            headers=key)
        assert found.status_code == 409
        assert "Kokoro" in found.json()["error"]
        found = client.post(api.CLONE_SAVE_ROUTE, json={"token": "x"}, headers=key)
        assert found.status_code == 409

    def test_a_setting_this_engine_does_not_have_is_refused(self, client, key, host,
                                                            voice_root):
        import mc_voice_engines as engines

        engines.select("pocket")
        found = client.post(api.ENGINE_SETTINGS_ROUTE,
                            json={"values": {"threads": 8}}, headers=key)
        assert found.status_code == 400
        assert "threads" in found.json()["error"]

    def test_a_pocket_setting_reaches_pockets_own_file(self, client, key, host,
                                                       voice_root, monkeypatch):
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        engines.select("pocket")
        monkeypatch.setattr(pocket, "_retire", lambda reason: None)
        found = client.post(api.ENGINE_SETTINGS_ROUTE,
                            json={"values": {"steps": 3, "precision": "int8"}},
                            headers=key)
        assert found.status_code == 200
        assert found.json()["settings"]["steps"] == 3
        assert pocket.steps() == 3
        assert pocket.precision() == "int8"


class TestOneAdapterSignatureIsReadRatherThanGuessedAt:
    """Section 30's shim, and the one way it could go wrong.

    Sopro's ``prepare_preview`` takes a language hint and Pocket's does not,
    because Pocket's model *is* the language. Which of the two an adapter has is
    read off its signature -- not found out by calling it and catching
    ``TypeError``, because a ``TypeError`` raised inside a preparation is
    indistinguishable from one raised by the call.
    """

    def test_the_fuller_call_is_made_when_the_adapter_takes_it(self):
        seen = []

        class Adapter:
            def prepare_preview(self, name, wav, language=""):
                seen.append((name, wav, language))
                return {"token": "t"}

        api._prepare_preview(Adapter(), "A Voice", b"RIFF", "en")
        assert seen == [("A Voice", b"RIFF", "en")]

    def test_the_shorter_call_is_made_when_it_does_not(self):
        seen = []

        class Adapter:
            def prepare_preview(self, name, wav):
                seen.append((name, wav))
                return {"token": "t"}

        api._prepare_preview(Adapter(), "A Voice", b"RIFF", "en")
        assert seen == [("A Voice", b"RIFF")]

    def test_a_type_error_from_inside_does_not_run_the_preparation_twice(self):
        """The bug this replaced. Sopro's third parameter is defaulted, so the
        fuller call always binds and the fallback could only ever be reached by
        a ``TypeError`` thrown from *inside* -- a decoder handed a shape it
        could not take. Retrying on that is a second worker round trip, a second
        staging directory, and a refusal from the retry rather than from the
        thing that actually went wrong."""
        calls = []

        class Adapter:
            def prepare_preview(self, name, wav, language=""):
                calls.append(language)
                raise TypeError("cannot multiply a bytes-like by a float")

        with pytest.raises(TypeError):
            api._prepare_preview(Adapter(), "A Voice", b"RIFF", "en")
        assert calls == ["en"], calls

    def test_an_adapter_with_no_readable_signature_gets_the_fuller_call(self):
        """A C builtin or something exotic is taken at its word rather than
        refused: the shim's job is to pick a call, not to vet a callable."""
        seen = []

        class Adapter:
            prepare_preview = staticmethod(
                lambda *arguments: seen.append(arguments) or {"token": "t"})

        api._prepare_preview(Adapter(), "A Voice", b"RIFF", "en")
        assert seen == [("A Voice", b"RIFF", "en")]


class TestThePocketRoutes:
    def test_the_status_route_is_refused_while_another_engine_is_selected(
            self, client, key, host):
        import mc_voice_engines as engines

        engines.select("sopro")
        found = client.post(api.POCKET_ROUTE, json={}, headers=key)
        assert found.status_code == 409
        assert found.json()["engine_mismatch"] is True

    def test_the_status_route_reports_the_five_states_and_what_stop_means(
            self, client, key, host, voice_root):
        import mc_voice_engines as engines

        engines.select("pocket")
        found = client.post(api.POCKET_ROUTE, json={}, headers=key)
        assert found.status_code == 200
        payload = found.json()
        for name in ("platform_supported", "runtime_ready", "speech_model_ready",
                     "official_voices_ready", "cloning_ready", "interrupt_mode",
                     "draining", "engine_busy", "pinned"):
            assert name in payload, name
        assert payload["interrupt_mode"] == "drain_unit"
        assert payload["draining"] is False

    def test_the_status_route_carries_no_path(self, client, key, host, voice_root):
        import mc_voice_engines as engines
        import mc_voice_paths as paths

        engines.select("pocket")
        found = client.post(api.POCKET_ROUTE, json={}, headers=key)
        assert str(paths.data_root()) not in found.text

    def test_an_install_that_this_build_will_not_start_is_refused_with_its_reason(
            self, client, key, host, voice_root, monkeypatch):
        """A managed install this build cannot vouch for is refused before a
        thread starts, so the sentence reaches the browser waiting for it."""
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        engines.select("pocket")
        monkeypatch.setattr(pocket, "refusal",
                            lambda manual=False: "PocketTTS says no, for a reason.")
        found = client.post(api.POCKET_INSTALL_ROUTE, json={}, headers=key)
        assert found.status_code == 409
        assert found.json()["error"] == "PocketTTS says no, for a reason."

    def test_no_status_route_starts_an_install_or_a_worker(self, client, key, host,
                                                           voice_root, monkeypatch):
        """Section 17, at the route a browser polls."""
        import mc_voice_engines as engines
        import mc_voice_pocket_runtime as pocket_runtime

        engines.select("pocket")
        monkeypatch.setattr(pocket_runtime, "ensure_started",
                            lambda: pytest.fail("a status read started a worker"))
        assert client.post(api.POCKET_ROUTE, json={}, headers=key).status_code == 200
        assert client.post(api.STATUS_ROUTE, json={}, headers=key).status_code == 200


class TestTheStatusPayloadTellsTheBrowserWhatStopMeans:
    """I-PKT-28. The browser draws its playback state from what the engine
    declares, never from a version string it recognised."""

    def test_every_engine_reports_an_interrupt_mode_and_a_drain_flag(self, client, key,
                                                                     host, voice_root):
        import mc_voice_engines as engines

        for engine, mode in (("kokoro", "cancel"), ("sopro", "cancel"),
                             ("pocket", "drain_unit")):
            engines.select(engine)
            found = client.post(api.STATUS_ROUTE, json={}, headers=key).json()
            assert found["interrupt_mode"] == mode, engine
            assert found["draining"] is False, engine
            assert "engine_busy" in found, engine

    def test_the_inactive_engines_block_is_still_absent(self, client, key, host,
                                                        voice_root):
        import mc_voice_engines as engines

        engines.select("pocket")
        found = client.post(api.STATUS_ROUTE, json={}, headers=key).json()
        assert "pocket" in found
        assert "sopro" not in found
        assert "kokoro" not in found
