"""The browser routes, on the WebUI's own app, under the WebUI's own auth.

Voice Chat adds no second web server. The browser already has a connection to
Forge -- often an HTTPS one across a LAN, because that is what an Android
microphone requires -- and these routes ride on it:

    POST <root>/model-chain/voice/status        what is installed and loaded
    POST <root>/model-chain/voice/stt           a PCM16 WAV in, a transcript out
    POST <root>/model-chain/voice/tts           a target token in, a WAV out
    POST <root>/model-chain/voice/tts-stream    a turn id in, raw PCM out, live
    POST <root>/model-chain/voice/cancel        stop one turn
    POST <root>/model-chain/voice/runtime       load or unload the worker
    POST <root>/model-chain/voice/install       provision one of the bundles
    POST <root>/model-chain/voice/models        the speech-model tiers; choose one
    POST <root>/model-chain/voice/profile       how the default voice is delivered
    POST <root>/model-chain/voice/voices        the voice registry
    POST <root>/model-chain/voice/voice/*       default, test, rename, delete
    POST <root>/model-chain/voice/cloning/*     install, start, status, abort

Blocking work never runs on the event loop
------------------------------------------
Section 17, and it is a correction rather than a new rule: the V1 handlers were
``async def`` and called straight into a runtime that waits for Whisper. One
dictation therefore froze every other request in the WebUI for its duration --
including the status poll drawing the microphone that was doing it. Every
handler below hands its blocking half to :func:`_offload`, which runs it in
Starlette's threadpool, and the streaming response awaits its queue reads the
same way. Nothing here waits for Kokoro, Whisper, a worker handshake, a bank
build or a queue on the loop thread.

Streaming speech
----------------
``/tts-stream`` is the one route that answers with a body that does not exist
yet. It takes an opaque turn id -- never text, in the URL, in a header or in
the body -- and yields raw mono PCM16 as the worker produces it, with the
sample rate in a header because the body has no container. It sets
``X-Accel-Buffering: no`` and no ``Content-Length``, because an intermediary
that buffers the whole response would turn streaming speech back into
completed-reply speech while still calling itself streaming (section 40).

A client that goes away cancels the turn it was listening to. That is not
tidiness: the worker is producing into a bounded queue, and a listener that
vanished is the one condition in which that queue never drains.

``<root>`` is the point of R2-4 and is not this module's problem: FastAPI mounts
these where the app is mounted, and ``javascript/voice_chat.js`` builds its URLs
from the deployment root Gradio publishes rather than from ``location.origin``.
What *is* this module's problem is proving they are no more reachable than the
rest of the WebUI. There is exactly one mechanism for that, and it is not a
login:

The page token
    A per-process random value Python puts into the two pages that need it --
    the Settings row and the Conversation panel -- and which the browser sends
    back in ``X-Model-Chain-Voice``. Both of those pages are served by the
    WebUI, behind whatever login it has, so a caller who cannot get past that
    login never receives a token; it is 24 random bytes, so they cannot guess
    one. It also stops a cross-site page from driving these routes with the
    user's cookies, and makes a tab left open across a restart fail with "reload
    the page" rather than confusingly.

    There is deliberately no second check. A cookie check used to sit behind
    this one and it locked legitimate users out twice -- reaching a page the
    WebUI served is the proof of access, and asking for it again only produced a
    way to fail. Nor does anything here ask for an account on any other service:
    the model artifacts are public files and this extension holds no API key.

Speech targets
    The TTS route takes a token, never text. What it speaks is an immutable
    snapshot of the assistant reply that completed the run which created the
    target (R2-5) -- so editing, regenerating, branching or switching threads
    between the reply and the request cannot change what comes out of the
    speaker. Targets are one-shot, expire in two minutes, live in this process's
    RAM, and are never written down or logged.

Nothing here touches the disk. The uploaded WAV is a ``bytes`` that goes to the
worker and is dropped; the generated WAV is a response body. There is no
``/file=`` URL, no Gradio audio cache, and no ``tempfile`` import in this module
-- which is a thing ``tests/test_voice_privacy.py`` checks rather than a thing
this docstring merely asserts.
"""

from __future__ import annotations

import base64
import logging
import secrets
import struct
import threading
import time

import mc_voice_models as models
import mc_voice_paths as paths
import mc_voice_runtime as runtime
import mc_voice_state as state
import mc_voice_turn as turns

try:
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
except Exception:  # pragma: no cover - a host without FastAPI has no routes to add
    Request = Response = JSONResponse = StreamingResponse = None

"""Imported at module level, and that is not a style preference.

This module carries ``from __future__ import annotations``, so every annotation
is a string that FastAPI resolves later against *module* globals. Imported
inside :func:`install`, ``Request`` would not be there when it looked -- and
FastAPI's answer to an annotation it cannot resolve is not an error, it is to
treat the parameter as a request body. Every route then answers 422 to a
perfectly good call, which is a failure mode that looks like a client bug.

Guarded because this file is imported when the extension is imported, and an
environment without FastAPI should reach :func:`install` and answer False rather
than take the WebUI down on the way past.
"""

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

PREFIX = "/model-chain/voice"

STATUS_ROUTE = f"{PREFIX}/status"
STT_ROUTE = f"{PREFIX}/stt"
TTS_ROUTE = f"{PREFIX}/tts"
INSTALL_ROUTE = f"{PREFIX}/install"
MODELS_ROUTE = f"{PREFIX}/models"
PROFILE_ROUTE = f"{PREFIX}/profile"
STREAM_ROUTE = f"{PREFIX}/tts-stream"
CANCEL_ROUTE = f"{PREFIX}/cancel"
TELEMETRY_ROUTE = f"{PREFIX}/telemetry"
RUNTIME_ROUTE = f"{PREFIX}/runtime"
VOICES_ROUTE = f"{PREFIX}/voices"
VOICE_DEFAULT_ROUTE = f"{PREFIX}/voice/default"
VOICE_TEST_ROUTE = f"{PREFIX}/voice/test"
VOICE_RENAME_ROUTE = f"{PREFIX}/voice/rename"
VOICE_DELETE_ROUTE = f"{PREFIX}/voice/delete"
CLONING_INSTALL_ROUTE = f"{PREFIX}/cloning/install"
CLONING_STATUS_ROUTE = f"{PREFIX}/cloning/status"
CLONING_START_ROUTE = f"{PREFIX}/cloning/start"
CLONING_ABORT_ROUTE = f"{PREFIX}/cloning/abort"

ENGINES_ROUTE = f"{PREFIX}/engines"
ENGINE_SELECT_ROUTE = f"{PREFIX}/engine/select"
SURFACE_ROUTE = f"{PREFIX}/surface"
SOPRO_ROUTE = f"{PREFIX}/sopro"
SOPRO_INSTALL_ROUTE = f"{PREFIX}/sopro/install"
SOPRO_SETTINGS_ROUTE = f"{PREFIX}/sopro/settings"
SOPRO_CLONE_ROUTE = f"{PREFIX}/sopro/clone"
SOPRO_REBUILD_ROUTE = f"{PREFIX}/sopro/rebuild"
SOPRO_STARTER_ROUTE = f"{PREFIX}/sopro/starter"
CLEANUP_ROUTE = f"{PREFIX}/cleanup"
CLEANUP_INSTALL_ROUTE = f"{PREFIX}/cleanup/install"
CLEANUP_RUN_ROUTE = f"{PREFIX}/cleanup/run"
LAB_ROUTE = f"{PREFIX}/lab"
LAB_UPDATE_ROUTE = f"{PREFIX}/lab/update"
LAB_RESET_ROUTE = f"{PREFIX}/lab/reset"
LAB_PLAY_ROUTE = f"{PREFIX}/lab/play"
"""Sopro's own routes, under their own prefix.

Separate paths rather than an ``engine`` parameter on the existing ones,
wherever the operation only exists on one engine. Cloning a Sopro voice and
running a Lab audition have no Kokoro meaning at all, and a shared route that
branched on a parameter would be a route whose refusals had to explain which
half of itself was unavailable.

Where an operation *is* shared -- listing voices, setting a default, renaming,
deleting, auditioning -- the path stays the same and the payload is scoped to
the active engine, which is what keeps the browser's own code engine-neutral.
"""

SOPRO_ROUTES = (ENGINES_ROUTE, ENGINE_SELECT_ROUTE, SURFACE_ROUTE,
                SOPRO_ROUTE, SOPRO_INSTALL_ROUTE,
                SOPRO_SETTINGS_ROUTE, SOPRO_CLONE_ROUTE, SOPRO_REBUILD_ROUTE,
                SOPRO_STARTER_ROUTE,
                CLEANUP_ROUTE, CLEANUP_INSTALL_ROUTE, CLEANUP_RUN_ROUTE,
                LAB_ROUTE, LAB_UPDATE_ROUTE, LAB_RESET_ROUTE, LAB_PLAY_ROUTE)

ROUTES = (STATUS_ROUTE, STT_ROUTE, TTS_ROUTE, INSTALL_ROUTE, MODELS_ROUTE, PROFILE_ROUTE,
          STREAM_ROUTE, CANCEL_ROUTE, TELEMETRY_ROUTE,
          RUNTIME_ROUTE, VOICES_ROUTE, VOICE_DEFAULT_ROUTE, VOICE_TEST_ROUTE,
          VOICE_RENAME_ROUTE, VOICE_DELETE_ROUTE, CLONING_INSTALL_ROUTE,
          CLONING_STATUS_ROUTE, CLONING_START_ROUTE, CLONING_ABORT_ROUTE) + SOPRO_ROUTES

TOKEN_HEADER = "x-model-chain-voice"

MAX_AUDIO_BYTES = 4 * 1024 * 1024
"""Sixty seconds of 16 kHz mono PCM16 is about 1.9 MB, so this is generous and
is still a refusal that happens before a single byte reaches an inference
engine."""

MAX_SECONDS = 60.0
ACCEPTED_RATES = (16000,)
TARGET_TTL = 120.0

MAX_JSON_BYTES = 64 * 1024
"""A JSON body big enough for a name and a voice id, and no bigger. Every route
below takes identifiers; none of them takes prose."""

MAX_REFERENCE_BYTES = 32 * 1024 * 1024
"""One clone's reference recording, refused here before it reaches a file."""

STREAM_IDLE = 0.1
"""How long one queue read waits before the stream checks whether its client is
still there. Short enough to notice a disconnect, long enough not to spin."""

_lock = threading.Lock()
_targets: dict[str, dict] = {}
_token = secrets.token_urlsafe(24)
"""This process's page token. Regenerated by a restart, which is what makes a
tab left open across one fail with "reload the page" rather than half-work."""


def session_token() -> str:
    return _token


# --------------------------------------------------------------------------- #
# Speech targets
# --------------------------------------------------------------------------- #


def remember_reply(text: str, voice_id: str = "", profile=None, engine: str = "") -> str:
    """Take an immutable snapshot of a completed reply. Returns its token.

    The snapshot is the design decision (R2-5, I-13). A message index would have
    been smaller and is wrong: the thread can be edited, regenerated, branched
    or switched between the run finishing and the browser asking for audio, and
    every one of those changes what an index means. A copy of the string cannot
    change.

    *How* it is to be spoken is part of that snapshot for exactly the same
    reason, and was not before: this route used to synthesize with no speaker at
    all, which sherpa answers with speaker 0 -- section 113's bug, still live in
    the one path that had not been looked at. A reply from a character with a
    voice of its own has to keep that voice on the non-streaming path too, and
    the moment the reply finished is the only moment that is certain to know it.
    """
    snapshot = str(text or "")
    if not snapshot.strip():
        return ""
    token = secrets.token_urlsafe(18)
    now = time.monotonic()
    with _lock:
        _expire(now)
        _targets[token] = {"text": snapshot, "created": now, "session": _token,
                           "voice": str(voice_id or ""),
                           # Which engine is part of the snapshot too, and for
                           # the same reason the voice is: the engine can be
                           # switched between the reply finishing and the
                           # browser asking for audio, and a reply spoken by
                           # whichever engine happens to be selected *then* is
                           # a reply in a voice nobody chose.
                           "engine": str(engine or ""),
                           "profile": dict(profile) if profile else None}
    return token


def take_reply(token: str):
    """Consume one target, atomically. ``None`` if there is not one.

    One-shot because a token that could be replayed is a token a page left open
    could make speak the same reply again on every refresh, and because the
    whole life of a target is one POST from one page.
    """
    key = str(token or "")
    now = time.monotonic()
    with _lock:
        _expire(now)
        found = _targets.pop(key, None)
    if found is None or found.get("session") != _token:
        return None
    return found


def _expire(now: float) -> None:
    for key in [key for key, item in _targets.items() if now - item["created"] > TARGET_TTL]:
        _targets.pop(key, None)


def forget_targets() -> None:
    """Drop every pending target. Used by shutdown and by the tests."""
    with _lock:
        _targets.clear()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class Refused(Exception):
    """A request that is not going to reach an inference engine."""

    def __init__(self, status: int, reason: str, mismatch: bool = False):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.mismatch = bool(mismatch)
        """Whether this refusal is specifically "the engine changed under you".

        Flagged rather than inferred from the status code, because 409 already
        means several other things on these routes -- a turn that was over
        before anything listened to it, an install that is already running, a
        Lab session that expired -- and a browser that reloaded the page for
        every one of them would be a browser that reloads the page when a
        slider is pressed twice.
        """


def _active(engine: str = "") -> str:
    """The active engine, or a refusal a browser can act on. Section 5.

    :func:`mc_voice_engines.refuse_mismatch` raises its own exception, and left
    to itself it would reach the route handler's catch-all and come back as a
    500. That is the wrong answer to the wrong question: a mismatch is not "that
    failed", it is "the page you are looking at is out of date", and it is the
    one refusal the browser responds to by reloading.

    Every engine-scoped entry point goes through here rather than calling the
    facade directly, so there is one place the answer is decided.
    """
    import mc_voice_engines as engines

    try:
        return engines.refuse_mismatch(engine)
    except engines.ActiveEngineMismatch as exc:
        raise Refused(409, str(exc), mismatch=True) from None
    except engines.EngineError as exc:
        raise Refused(400, str(exc)) from None


def validate_wav(data: bytes) -> dict:
    """Parse enough of a RIFF/WAVE header to refuse everything unwanted.

    Written out rather than handed to ``wave`` because this is the boundary: the
    module that raises on a malformed chunk table should be the one in front of
    the worker, not the worker. What is accepted is deliberately narrow -- the
    browser is ours and produces exactly one shape -- and everything else is a
    sentence rather than an exception two processes away.
    """
    if not data:
        raise Refused(400, "No audio was received.")
    if len(data) > MAX_AUDIO_BYTES:
        raise Refused(413, "That recording is too large.")
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise Refused(400, "That upload is not a WAV recording.")

    offset, found = 12, {}
    while offset + 8 <= len(data):
        name = data[offset:offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        body = offset + 8
        if body + size > len(data):
            # A chunk claiming more than the file holds is the malformed length
            # table the design intent asks to be refused by name.
            raise Refused(400, "That recording's header is malformed.")
        if name == b"fmt " and size >= 16:
            audio_format, channels, rate, _bps, _align, bits = struct.unpack_from(
                "<HHIIHH", data, body)
            found["fmt"] = (audio_format, channels, rate, bits)
        elif name == b"data":
            found["data"] = size
        offset = body + size + (size % 2)

    if "fmt" not in found or "data" not in found:
        raise Refused(400, "That recording is not a complete WAV.")
    audio_format, channels, rate, bits = found["fmt"]
    if audio_format != 1:
        raise Refused(400, "Voice Chat accepts uncompressed PCM audio only.")
    if channels != 1:
        raise Refused(400, "Voice Chat accepts mono audio only.")
    if bits != 16:
        raise Refused(400, "Voice Chat accepts 16-bit audio only.")
    if rate not in ACCEPTED_RATES:
        raise Refused(400, "Voice Chat accepts 16 kHz audio only.")
    frames = found["data"] // 2
    seconds = frames / float(rate)
    if frames <= 0:
        raise Refused(400, "That recording is empty.")
    if seconds > MAX_SECONDS + 0.5:
        raise Refused(400, "That recording is longer than Voice Chat accepts.")
    if seconds < 0.2:
        raise Refused(400, "Hold at the right-hand end of the track to talk.")
    return {"seconds": seconds, "rate": rate, "frames": frames}


# --------------------------------------------------------------------------- #
# Who is asking
# --------------------------------------------------------------------------- #


_said: dict[str, float] = {}
_said_lock = threading.Lock()


def _once(key: str, message: str, every: float = 600.0) -> None:
    """Say something at most once per ``every`` seconds.

    Because a route a page polls turns any per-request line into a log that
    scrolls forever -- which is exactly what the first version of the refusal
    logging did, at better than one line a second, for as long as the Settings
    page was open.
    """
    now = time.monotonic()
    with _said_lock:
        last = _said.get(key)
        if last is not None and now - last < every:
            _said[key + "#count"] = _said.get(key + "#count", 0) + 1
            return
        suppressed = _said.pop(key + "#count", 0)
        _said[key] = now
    if suppressed:
        message = f"{message} (and {int(suppressed)} more like it since the last line)"
    logger.warning(message)


def forget_repeats() -> None:
    """Drop the throttle's memory. For the tests, and for a UI reload."""
    with _said_lock:
        _said.clear()


def _same_origin(request) -> bool:
    """Reject a page on another origin driving these routes with our cookies.

    Only when an ``Origin`` is present: a same-origin ``fetch`` from some
    browsers does not send one, and treating its absence as hostile would break
    the ordinary case to defend against a case the token already covers.
    ``X-Forwarded-Host`` is honoured because a reverse proxy is a documented
    deployment -- and it is compared rather than trusted blindly, which is the
    difference between supporting a proxy and having no check at all.
    """
    headers = getattr(request, "headers", {}) or {}
    origin = headers.get("origin")
    if not origin:
        return True
    wanted = {headers.get("host"), headers.get("x-forwarded-host")}
    wanted.discard(None)
    try:
        from urllib.parse import urlsplit

        netloc = urlsplit(origin).netloc
    except Exception:
        return False
    return bool(netloc) and netloc in wanted


def _matches_token(offered: str) -> bool:
    """Constant-time comparison that cannot be made to raise.

    ``compare_digest`` on ``str`` requires both sides to be ASCII and raises
    ``TypeError`` otherwise -- so a header with one accented character would
    leave this function through the route's generic handler as a 500 rather
    than as the 403 it is. Encoded first, which makes every input comparable.
    """
    try:
        return secrets.compare_digest(str(offered or "").encode("utf-8"),
                                      _token.encode("utf-8"))
    except Exception:
        return False


def _checked(request, route: str = "") -> None:
    """Every gate one voice route passes. There is no sign-in among them.

    **Voice Chat has no login of its own and does not re-check the WebUI's.**
    That is a deliberate reversal. A cookie check used to live here, reasoning
    that a route an extension adds is not automatically covered by Gradio's own
    authentication dependency -- which is true -- and it locked people out
    twice: once by mistaking an unrelated attribute for a login, and then on an
    installation that really does pass ``--gradio-auth`` and whose user was, of
    course, already signed in by the time they reached the page.

    The page token provides the parity, and more simply. It is minted per WebUI
    process and delivered in exactly two places, both inside pages the WebUI
    itself served: the Settings row and the Conversation panel. Somebody who
    cannot get past the WebUI's login never receives one, and it is 24 random
    bytes, so they cannot guess one. Reaching the page *is* the proof of access,
    and asking for it a second time only ever produced a way to fail.

    The gap that leaves, stated rather than hidden: a token already handed to a
    browser stays usable until the WebUI restarts, so revoking an account
    mid-session does not close a page that is already open. For a local speech
    feature on a single-user WebUI that is the right trade.

    Nothing here asks for an account anywhere else either. The model artifacts
    are public files, and Voice Chat holds no API key and has nowhere to put one.
    """
    headers = getattr(request, "headers", {}) or {}
    offered = headers.get(TOKEN_HEADER)
    if not _matches_token(offered):
        _refused(route, "no page token was sent" if not offered
                 else "the page token is from a previous run of this WebUI")
        raise Refused(403, "This page was loaded before the WebUI restarted. Reload it — "
                           "you are not being asked to sign in to anything.")
    if not _same_origin(request):
        _refused(route, "the Origin header did not match this WebUI's host")
        raise Refused(403, "That request did not come from this WebUI.")


def _refused(route: str, reason: str) -> None:
    """One line per reason, not one line per request.

    The Settings row polls, so a refusal that logged every time turned a
    configuration problem into a hundred and thirty-six identical warnings in
    three minutes -- which buries the line that would have explained it.
    """
    _once(f"refused:{route}:{reason}",
          f"Model Chain: Voice Chat refused a request to {route or 'a voice route'} — {reason}")


# --------------------------------------------------------------------------- #
# The payloads, built without a framework
# --------------------------------------------------------------------------- #


def _offload(call, *args, **kwargs):
    """Run blocking work off the event loop, wherever this is hosted.

    Starlette's own threadpool first, because that is what the host is already
    using and its limits are the host's to tune; AnyIO next, for a FastAPI
    mounted without Starlette's helper in reach; and a direct call last, which
    is what the tests take and is correct there because there is no loop to
    block. Awaited by every handler that can wait for anything at all.
    """
    try:
        from starlette.concurrency import run_in_threadpool

        return run_in_threadpool(call, *args, **kwargs)
    except ImportError:
        pass
    try:
        import functools

        import anyio.to_thread

        return anyio.to_thread.run_sync(functools.partial(call, *args, **kwargs))
    except ImportError:
        pass

    async def here():
        return call(*args, **kwargs)

    return here()


async def _json(request) -> dict:
    """One JSON body, bounded before it is parsed.

    Bounded because every route here takes identifiers: a caller sending a
    megabyte of JSON to ``/voice/rename`` is not a caller whose request should
    be parsed and then rejected.
    """
    body = await request.body()
    if len(body) > MAX_JSON_BYTES:
        raise Refused(413, "That request is too large.")
    if not body:
        return {}
    import json

    try:
        found = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise Refused(400, "That request was not valid JSON.") from None
    return found if isinstance(found, dict) else {}


def status_payload() -> dict:
    """What the browser needs to draw the microphone, and nothing else.

    No paths, no pids, no model filenames: this is read by a page and everything
    in it is a fact that page needs. The messages are the same sentences
    Settings shows, so a user reading one and a user reading the other are being
    told the same thing.
    """
    import mc_voice_engines as engines

    found = models.status()
    active = engines.active()
    payload = {
        "ok": True,
        # STT is outside the engine selector and stays that way. Every field
        # below about speech-to-text is reported whichever TTS engine is
        # selected, because switching Kokoro to Sopro must not reload Whisper,
        # change its tier or reset the microphone (I-7).
        "runtime_ready": found.runtime_ready,
        "stt_ready": found.stt_ready,
        "platform_supported": found.platform_supported,
        "runtime_message": found.runtime_message,
        "stt_message": found.stt_message,
        "auto_send": state.auto_send(),
        "auto_speak": state.auto_speak(),
        "busy": found.busy,
        "progress": models.progress(),
        "sources": _sources(),
        "stt_model": {"id": found.stt_id, "label": found.stt_label, "tier": found.stt_tier},
        "speaking": turns.busy(),
        "engines": engines.state(),
        "not_ready_message": ("Voice Chat is not set up. Install both models in "
                              "Settings → Voice Chat."),
    }
    # Only the selected engine's operational block is built, so a stale DOM, a
    # theme script or a partial re-render has nothing to expose (section 5).
    # The inactive engine appears in ``engines`` as a name and a blurb and
    # nowhere else.
    payload.update(_engine_block(active))
    return engines.scope(payload, active)


def _engine_block(active: str) -> dict:
    """The active TTS engine's operational state, and only that engine's.

    Both branches answer the same four questions -- is it installed, what does
    it say about that, is a worker resident, and which voice speaks next -- so
    every surface that draws them is engine-neutral above this line and engine-
    specific below it.
    """
    import mc_voice_engines as engines

    if active == engines.SOPRO:
        import mc_voice_sopro as sopro
        import mc_voice_sopro_runtime as sopro_runtime

        found = sopro.status()
        return {
            "ready": found.ready,
            "tts_ready": found.ready,
            "tts_message": found.message,
            # ``engine_state``, not ``engine``. ``engine`` is the selected
            # engine's *id* in every payload this feature sends, and it used to
            # be this residency object in the status payload alone -- so the
            # scoping filter, which sets ``engine`` to the id, silently replaced
            # the Voice flyout's Loaded/Unloaded line with the string "sopro".
            # One key, one meaning.
            "engine_state": sopro_runtime.engine(),
            "voice": _default_voice(),
            "delivery": _delivery_summary(),
            "sopro": {
                "installed": found.ready,
                "runtime_ready": found.runtime_ready,
                "model_ready": found.model_ready,
                "runtime_message": found.runtime_message,
                "model_message": found.model_message,
                "platform_supported": found.platform_supported,
                "fingerprint": found.fingerprint,
                "settings": sopro.engine_settings(),
                "defaults": sopro_runtime.defaults(),
                "warnings": sopro.warnings(),
            },
        }
    found = models.status()
    return {
        "ready": found.ready,
        "tts_ready": found.tts_ready,
        "tts_message": found.tts_message,
        "engine_state": runtime.engine(),
        "voice": _default_voice(),
        "delivery": _delivery_summary(),
        "kokoro": {"installed": found.tts_ready, "message": found.tts_message},
    }


def _delivery_summary() -> str:
    """One line naming what has been changed about the default voice's delivery.

    In the status because the Voice flyout draws it beside the voice's name, and
    "why does it sound like that" is a question a settings page two clicks away
    cannot answer while somebody is listening.
    """
    try:
        import mc_voice_engines as engines

        profiles = engines.profiles()
        return profiles.describe(profiles.stored())
    except Exception:
        logger.debug("Model Chain: could not read the voice delivery profile", exc_info=True)
        return ""


def _default_voice() -> dict:
    """Which voice the next reply will use, by name and by stable id.

    Never a raw SID: the number is an implementation detail of a bank the
    browser has no business addressing, and section 56 says a browser-supplied
    one is never to be trusted anyway.
    """
    try:
        import mc_voice_engines as engines

        entry = engines.adapter().default_entry()
        if entry is None:
            return {"id": "", "name": ""}
        return {"id": entry["id"], "name": entry["display_name"], "label": entry["label"],
                "custom": not entry["official"]}
    except Exception:
        logger.debug("Model Chain: could not read the default voice", exc_info=True)
        return {"id": "", "name": ""}


def _tts_configuration() -> dict:
    """The process-level facts a turn's durations have to be read against.

    Separate from the turn's own metrics because they are facts about the
    worker rather than about the reply, and because reading them must never be
    able to stop a turn being logged: an unreachable runtime answers "unknown"
    rather than raising into the summary.
    """
    try:
        import mc_voice_runtime as runtime

        found = runtime.engine()
        return {"tts_threads": found.get("tts_threads") or runtime.TTS_THREADS,
                "priority": found.get("priority") or "unknown"}
    except Exception:
        logger.debug("Model Chain: could not read the Voice runtime configuration",
                     exc_info=True)
        return {"tts_threads": "unknown", "priority": "unknown"}


def _log_turn(turn) -> None:
    """One line per spoken turn, of numbers only. Section 36.

    Every field comes from :meth:`mc_voice_turn.VoiceTurn.metrics`, which is
    built by naming what may be reported rather than by removing what may not.
    No text of any kind reaches this function.
    """
    try:
        found = turn.metrics()
        if not found.get("chunks"):
            # The interesting case, and the one an earlier version of this
            # function skipped: a turn a browser listened to and that produced
            # no audio at all leaves no other trace anywhere.
            logger.warning("Model Chain: Voice turn %s produced no audio — %d characters, "
                           "%d segments handed over, %s", turn.id[:8], found["source_chars"],
                           found["segments"], found["cancelled"] or "no reason recorded")
            if turn.error:
                logger.warning("Model Chain: Voice turn %s — %s", turn.id[:8], turn.error)
            return
        logger.info("Model Chain: Voice turn finished — %d characters, %d segments, "
                    "%.1f s audio in %.1f s compute (RTF %s), first audio %s ms, %s",
                    found["source_chars"], found["segments"], found["audio_seconds"],
                    found["compute_seconds"], found["rtf"], found["first_audio_ms"],
                    found["cancelled"] or "completed")
        # A second line, and it is the one a latency question is answered from.
        # Everything on it is a duration or a count. The thread count and the
        # priority come from the runtime rather than the turn because they
        # belong to the process rather than the reply -- and they are on this
        # line because a duration means nothing without the configuration that
        # produced it. Section 36: no text, and no field here that is not in
        # `metrics` or in the runtime's own configuration.
        engine = _tts_configuration()
        logger.info("Model Chain: Voice turn latency — worker %s, prepare %s ms, "
                    "threads %s, priority %s, %s streaming; "
                    "unit 1 %s chars after %s ms ready, %s ms synth, first block %s ms, "
                    "%s blocks, %s ms audio; "
                    "unit 2 %s chars after %s ms ready, %s ms synth, first block %s ms, "
                    "%s blocks, %s ms audio; "
                    "slowest unit %s was %s ms; RTF %s, first audio %s ms",
                    "warm" if found["worker_warm_at_turn_start"] else "cold",
                    found["runtime_prepare_ms"], engine["tts_threads"], engine["priority"],
                    found["streaming"] or "unknown",
                    found["segment_1_chars"], found["ready_wait_1_ms"],
                    found["segment_1_ms"], found["segment_1_first_block_ms"],
                    found["segment_1_callback_blocks"], found["segment_1_audio_ms"],
                    found["segment_2_chars"], found["ready_wait_2_ms"],
                    found["segment_2_ms"], found["segment_2_first_block_ms"],
                    found["segment_2_callback_blocks"], found["segment_2_audio_ms"],
                    found["max_segment_index"], found["max_segment_ms"],
                    found["rtf"], found["first_audio_ms"])
        _log_shortfall(turn, found)
    except Exception:
        logger.debug("Model Chain: could not record voice turn metrics", exc_info=True)


def _log_shortfall(turn, found: dict) -> None:
    """Say it out loud when speech is being made slower than it is heard.

    An RTF above 1 is the whole of "the audio is choppy on long replies" and
    the previous two lines already carry the number -- but only as a ratio
    among a dozen other numbers, with nothing to say what it means or what
    moves it. Somebody reading a log to find out why their speech stutters
    should not have to know that 1.16 is the bad side of 1.

    Speed is named because on this engine Speed is usually the answer. Sopro V2
    has no model-native speaking rate (``sopro_worker.worker`` says so at
    length), so Speed is a time-compression applied *after* the model has
    produced full-length audio: the work is unchanged and the result is
    shorter, which multiplies the real-time factor by exactly the speed. A
    1.35x setting turns an engine running comfortably at 0.86 into one running
    at 1.16, and no buffer can hide a producer that never catches up. The line
    below reports what the model actually produced next to what came out, so
    the arithmetic is on the page rather than left as an exercise.

    Never fatal, and deliberately not a warning: an RTF above 1 is a machine
    being slow, not the extension being broken.
    """
    rtf = found.get("rtf")
    audio = found.get("audio_seconds") or 0.0
    if not rtf or rtf <= 1.0 or audio <= 0.0:
        return
    speed = 1.0
    try:
        speed = float((turn.profile or {}).get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    shortfall = round((rtf - 1.0) * audio, 1)
    if speed > 1.0:
        logger.info(
            "Model Chain: Voice produced speech slower than it plays — RTF %s, about "
            "%.1f s of silence owed across %.1f s of audio. Speed is %.2fx and is "
            "applied after synthesis, so the model generated %.1f s of audio to yield "
            "%.1f s; at Speed 1.00x the same turn would have measured about RTF %s. "
            "Lower Speed if replies stutter.",
            rtf, shortfall, audio, speed, audio * speed, audio,
            round(rtf / speed, 3))
        return
    logger.info(
        "Model Chain: Voice produced speech slower than it plays — RTF %s, about "
        "%.1f s of silence owed across %.1f s of audio. Speed is already 1.00x, so "
        "this machine is simply not synthesising in real time.", rtf, shortfall, audio)


def _sources() -> dict:
    """Where each bundle's files can be fetched by hand, for the Settings row.

    Read from the same manifest the automatic download uses, so the addresses
    on screen are the addresses this extension would have used -- a page that
    told somebody to go and find "a Whisper small ONNX export" would be a page
    that gets the wrong file installed.
    """
    found = {}
    for kind in ("stt", "tts"):
        try:
            found[kind] = models.sources(kind)
        except Exception:
            found[kind] = []
    try:
        found["runtime"] = models.runtime_sources()
    except Exception:
        found["runtime"] = []
    return found


def transcribe(body: bytes) -> dict:
    """The STT route's body: validate, transcribe, and let the bytes go.

    Two gates sit around the transcription and :mod:`mc_voice_hearing` explains
    both at length. In front of it, a recording whose level never left the noise
    floor is refused without waking Whisper -- which is the Bluetooth case, and
    the answer to it is a sentence naming the microphone rather than a large
    model confirming that silence is silent. Behind it, a result that is
    entirely one of Whisper's non-speech annotations is discarded, because
    ``(music)`` in the composer is worse than nothing in the composer.
    """
    import mc_voice_hearing as hearing

    shape = validate_wav(body)
    level = hearing.measure(body)
    if hearing.silent(level):
        logger.info("Model Chain: Voice STT refused a silent recording — %.1f s, peak %.3f, "
                    "rms %.4f", shape["seconds"], level.get("peak", 0.0),
                    level.get("rms", 0.0))
        return {"ok": False, "error": hearing.quiet_reason(level), "text": "",
                "level": _level(level)}
    try:
        found = runtime.transcribe(body)
    except runtime.VoiceRuntimeError as exc:
        raise Refused(503, str(exc)) from None
    raw = found.get("text") or ""
    text = hearing.speech(raw)
    logger.info("Model Chain: Voice STT finished — %.1f s audio in %.1f s, peak %.3f",
                shape["seconds"], float(found.get("elapsed") or 0.0),
                level.get("peak", 0.0))
    if not text.strip():
        # The two empty answers are different and are reported differently: a
        # model that returned nothing heard nothing, and a model that returned
        # an annotation heard something that was not speech. Only the second
        # one has a microphone to blame, and only it says so.
        if raw.strip():
            logger.info("Model Chain: Voice STT discarded a non-speech result — %.1f s "
                        "audio, peak %.3f", shape["seconds"], level.get("peak", 0.0))
            return {"ok": False, "error": hearing.hallucinated_reason(), "text": "",
                    "level": _level(level)}
        return {"ok": False, "error": "No speech was detected.", "text": "",
                "level": _level(level)}
    return {"ok": True, "text": text, "auto_send": state.auto_send(),
            "level": _level(level), "request_id": secrets.token_hex(6)}


def _level(measurement: dict) -> dict:
    """What the browser is told about the recording it just sent.

    Three numbers, rounded, and no audio. It is what lets the composer say "that
    was very quiet" beside a transcript that came back wrong, which is the one
    piece of evidence a user has that their microphone changed under them.
    """
    if not measurement:
        return {}
    return {"peak": round(float(measurement.get("peak") or 0.0), 4),
            "rms": round(float(measurement.get("rms") or 0.0), 5),
            "seconds": round(float(measurement.get("seconds") or 0.0), 2)}


def speak(token: str) -> bytes:
    """The TTS route's body: consume one target and synthesize its snapshot.

    A failure after consumption is reported, never repaired. Re-resolving "the
    current last reply" to try again would quietly speak a different message
    than the one the token stood for, which is the whole class of bug R2-5
    exists to prevent.
    """
    found = take_reply(token)
    if not found:
        raise Refused(404, "There is nothing waiting to be read aloud.")
    text = found["text"]
    import mc_voice_engines as engines

    wanted = str(found.get("engine") or "") or engines.active()
    if wanted != engines.active():
        # The engine was switched while this reply was waiting to be spoken.
        # Refused rather than re-resolved: there is no cross-engine fallback
        # (I-2), and speaking a reply that was written for one engine through
        # the other is exactly the silent substitution that rule forbids.
        raise Refused(409, f"The text-to-speech engine changed to "
                           f"{engines.label()} while that reply was waiting, so it was not "
                           f"read aloud.")
    voice_id, entry = _resolve_target(found, wanted)
    try:
        if wanted == engines.SOPRO:
            import mc_voice_sopro_runtime as sopro_runtime

            audio = sopro_runtime.synthesize(text, voice_id, profile=found.get("profile"))
        else:
            audio = runtime.synthesize(text, sid=int(entry.get("_sid") or 0),
                                       profile=found.get("profile"))
    except Exception as exc:
        if isinstance(exc, Refused):
            raise
        raise Refused(503, str(exc) or "That reply could not be read aloud.") from None
    logger.info("Model Chain: Voice TTS finished — %d characters, %d bytes of audio on %s",
                len(text), len(audio), engines.label(wanted))
    return audio


def _resolve_target(found: dict, engine: str):
    """``(qualified id, entry)`` for a remembered reply, on its own engine.

    Through the active engine's adapter, always -- the only path from a stable
    id to anything an engine can address, and never a number that came from
    anywhere else (section 56). The adapter already answers a voice that has
    since been deleted with that engine's default, so the one failure left here
    is that the engine has no usable voice at all, which is a sentence rather
    than speaker 0.
    """
    import mc_voice_engines as engines

    try:
        return engines.resolve(found.get("voice") or "", engine)
    except Exception as exc:
        raise Refused(503, str(exc) or "No voice is installed.") from None


# --------------------------------------------------------------------------- #
# Streaming speech
# --------------------------------------------------------------------------- #


def open_stream(token: str):
    """Find the turn a browser is asking to listen to, or refuse.

    Refuses rather than starts anything: a turn exists because Conversation
    created one for a reply it is generating. A ``/tts-stream`` for a turn this
    process does not know about is a stale page, and answering it by
    synthesising something would be answering a question nobody asked.
    """
    turn = turns.lookup(str(token or ""))
    if turn is None:
        logger.warning("Model Chain: a browser asked to hear a reply this WebUI has no "
                       "record of — the page is probably older than this run")
        raise Refused(404, "There is nothing waiting to be read aloud.")
    if turn.cancelled.is_set() and not turn.started.is_set():
        logger.warning("Model Chain: Voice turn %s was over before anything listened to it "
                       "— %s", turn.id[:8], turn.reason or "no reason recorded")
        raise Refused(409, turn.error or "That reply is no longer being read aloud.")
    turn.attached.set()
    logger.info("Model Chain: Voice is streaming turn %s to a browser", turn.id[:8])
    return turn


def stream_headers(turn) -> dict:
    """What the browser needs to decode a body with no container.

    Raw PCM has no header, so the sample rate travels beside it -- and the turn
    id travels with it too, so a response that arrives late cannot be mistaken
    for the current one (section 24). Deliberately no ``Content-Length``:
    the length is not known, and an intermediary given one would wait for it.
    """
    return {
        "Cache-Control": "no-store, no-transform, max-age=0",
        "Pragma": "no-cache",
        "X-Model-Chain-Voice-Rate": str(int(turn.sample_rate or 24000)),
        "X-Model-Chain-Voice-Turn": str(turn.id),
        # nginx and several reverse proxies buffer a response whole unless told
        # not to, which would deliver every sample at once at the end -- the
        # exact failure section 40 says must not be silently called streaming.
        "X-Accel-Buffering": "no",
        "Content-Disposition": "inline",
    }


def cancel_turn(token: str, reason: str = "") -> dict:
    """The Stop button's server half. Idempotent by construction.

    Takes a turn id and never any text, cancels only the turn it names, and is
    deliberately *not* ``runtime.stop()``: cancelling a reply is an ordinary
    thing that happens several times in a conversation, and section 27 is
    explicit that it must not cost a four-hundred-megabyte model reload.
    """
    wanted = str(token or "")
    why = str(reason or "")[:80]
    if why and why != "user":
        # The browser's own account of why it stopped listening. Never text --
        # a fixed word from a fixed list -- and worth writing down, because a
        # page that cannot play a reply is otherwise invisible from the server.
        logger.warning("Model Chain: a browser stopped listening to a reply — %s", why)
    cancelled = turns.cancel(wanted, why or "user") if wanted else turns.cancel_active()
    return {"ok": True, "cancelled": bool(cancelled), "speaking": turns.busy()}


TELEMETRY = {
    "playback": ("turn_seen_to_headers_ms", "headers_to_first_pcm_ms",
                 "first_pcm_to_playback_ms", "startup_buffer_ms", "underrun_count",
                 "first_underrun_after_play_ms", "max_underrun_gap_ms",
                 "total_underrun_gap_ms", "rebuffer_count", "rebuffer_target_ms"),
    "capture": ("mic_request_ms", "stream_ready_ms", "first_pcm_ms", "engaged_ms",
                "preroll_ms", "recorded_ms"),
}
"""The only fields this route will read, by report kind.

A fixed schema rather than a bag, and named here rather than inferred from
whatever arrives, because this is the one route whose whole purpose is to write
into a log file. "Reject or ignore unknown fields according to one documented
policy" is the requirement and the policy is *ignore*: a page from a newer build
reporting a field this one has never heard of should still get its other numbers
recorded rather than have the whole report refused. Nothing outside these tuples
is read, so nothing outside them can reach a log.

Every one of them is a duration in milliseconds or a count. There is no field
here that could carry a word.
"""

TELEMETRY_ENUMS = {"playback_end_reason": ("finished", "cancelled", "error", "replaced"),
                   "graph": ("worklet", "script", "none"),
                   "result": ("sent", "discarded", "error")}
"""The three non-numeric fields, and the complete set of values each may hold.

Enumerated rather than length-limited. A free-text field on a route that writes
to a log is a way to write text to a log, however short it is trimmed.
"""

TELEMETRY_MAX_MS = 24 * 60 * 60 * 1000
"""A day. Not a real duration -- it is the ceiling that stops a page, or
something pretending to be one, filling a log line with digits."""


def telemetry(payload: dict) -> dict:
    """One content-free playback or capture report from a browser, logged.

    The browser is the only party that can answer some of these questions. The
    server knows how long a synthesis took; only the page knows whether the
    speaker actually ran dry, and for how long. So this exists, and it is
    deliberately the least powerful route in this module: it reads a fixed set
    of numbers, writes one line, and returns.

    Best-effort in both directions. Nothing waits for it, its failure changes no
    playback, no capture, no cancellation and no composer state, and a report
    that cannot be understood is dropped rather than argued with.
    """
    kind = str(payload.get("kind") or "")
    if kind not in TELEMETRY:
        raise Refused(400, "That is not a kind of report this WebUI records.")
    turn = str(payload.get("turn") or "")[:24]
    if turn and not turn.replace("-", "").replace("_", "").isalnum():
        raise Refused(400, "That is not a turn identifier.")
    found = {}
    for name in TELEMETRY[kind]:
        value = payload.get(name)
        if value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        found[name] = max(0, min(number, TELEMETRY_MAX_MS))
    for name, allowed in TELEMETRY_ENUMS.items():
        value = payload.get(name)
        if isinstance(value, str) and value in allowed:
            found[name] = value
    if not found:
        raise Refused(400, "That report carried nothing this WebUI records.")
    logger.info("Model Chain: Voice %s timing — turn %s, %s", kind, turn[:8] or "none",
                ", ".join(f"{name}={found[name]}" for name in sorted(found)))
    return {"ok": True, "recorded": len(found)}


def set_runtime(action: str) -> dict:
    """Load or unload the *active* engine's worker. Section 31.

    The active one, resolved here rather than named, because the button lives in
    a flyout that says which engine it belongs to and pressing it must not load
    the other one. On an installation that has never selected Sopro this reaches
    exactly the code it always did.
    """
    import mc_voice_engines as engines

    active = engines.active()
    lifecycle = engines.runtime(active)
    wanted = str(action or "").strip().casefold()
    if wanted not in ("load", "unload"):
        raise Refused(400, "Voice Chat can load or unload its speech engine.")
    try:
        if wanted == "load":
            return {"ok": True, "engine": active, "engine_state": lifecycle.load()}
        return {"ok": True, "engine": active,
                "engine_state": lifecycle.unload("unloaded from the Voice panel")}
    except Exception as exc:
        if isinstance(exc, Refused):
            raise
        raise Refused(503, str(exc) or "The speech engine could not be changed.") from None


# --------------------------------------------------------------------------- #
# Voice management
# --------------------------------------------------------------------------- #


def voices_payload(test_text=None, engine: str = "") -> dict:
    """Everything the Settings voice list draws, for the active engine only.

    Scoped rather than filtered on the page (section 5): a browser asking this
    while Kokoro is selected is never told what Sopro voices exist, so a stale
    DOM has nothing to reveal and no request built from one can name a voice
    from the other engine.

    ``test_text`` is saved when it is given, which is how the editable Test Text
    field persists without a second route -- and it is the one genuinely
    engine-neutral value here, because "what should a test voice say" is a
    property of the person rather than of the engine.
    """
    import mc_voice_engines as engines

    active = _active(engine)
    adapter = engines.adapter(active)
    if test_text is not None:
        try:
            adapter.set_test_text(str(test_text))
        except Exception:
            logger.debug("Model Chain: could not save the voice test text", exc_info=True)
    try:
        found = adapter.entries()
    except Exception:
        logger.warning("Model Chain: the voice registry could not be read", exc_info=True)
        return {"ok": False, "error": "The voice list could not be read.", "voices": [],
                "engine": active}
    return {
        "ok": True,
        "engine": active,
        "engine_label": engines.label(active),
        "voices": [_public(entry) for entry in found],
        "default": adapter.default_id(),
        "test_text": adapter.test_text(),
        "capacity": adapter.capacity(),
        "warnings": adapter.warnings(),
    }


def _public(entry: dict) -> dict:
    """One registry entry as the browser sees it.

    The engine's own address is not in it -- Kokoro's SID under ``_sid``,
    anything else an adapter carries privately. A browser that knew the number
    could ask for a reserved slot or an unregistered speaker, and the answer to
    that is not to validate the number harder, it is not to publish it
    (section 56).

    Built by *naming* what may be published rather than by removing what may
    not, so a field an adapter adds later is absent until somebody adds it here
    on purpose.
    """
    found = {key: entry.get(key) for key in
             ("id", "display_name", "label", "type", "official", "language", "accent",
              "editable", "deletable", "engine")}
    for key in ("created_at", "source_seconds", "compatible", "has_source", "has_lab",
                "fingerprint"):
        if key in entry:
            found[key] = entry[key]
    return found


def set_default_voice(voice_id: str, engine: str = "") -> dict:
    import mc_voice_engines as engines

    active = _active(engine)
    if not engines.belongs(voice_id, active):
        raise Refused(400, f"That is not a {engines.label(active)} voice.")
    try:
        engines.adapter(active).set_default(str(voice_id or ""))
    except Exception as exc:
        raise Refused(400, str(exc) or "That voice could not be set as the default.") from None
    return voices_payload(engine=active)


def rename_voice(voice_id: str, display_name: str, engine: str = "") -> dict:
    import mc_voice_engines as engines

    active = _active(engine)
    if not engines.belongs(voice_id, active):
        raise Refused(400, f"That is not a {engines.label(active)} voice.")
    try:
        engines.adapter(active).rename(str(voice_id or ""), str(display_name or ""))
    except Exception as exc:
        raise Refused(400, str(exc) or "That voice could not be renamed.") from None
    return voices_payload(engine=active)


def delete_voice(voice_id: str, engine: str = "") -> dict:
    """Delete one voice, from the active engine's library and nowhere else.

    Explicit and engine-local (section 8): deleting a Sopro voice cannot touch a
    Kokoro bank and the reverse is equally true, which the ownership check below
    makes structural rather than a rule somebody remembers.
    """
    import mc_voice_engines as engines

    active = _active(engine)
    if not engines.belongs(voice_id, active):
        raise Refused(400, f"That is not a {engines.label(active)} voice.")
    try:
        engines.adapter(active).delete(str(voice_id or ""))
    except Exception as exc:
        raise Refused(400, str(exc) or "That voice could not be deleted.") from None
    return voices_payload(engine=active)


def test_voice(voice_id: str, text: str = "", profile=None, engine: str = "") -> bytes:
    """Audition one voice through the ordinary production runtime.

    "Ordinary" is the requirement (section 36, section 45): a Test that went
    down a different path would be a Test that could pass for a voice which
    cannot actually be spoken in a reply. So this is the same worker, the same
    reconstruction and the same delivery the next assistant turn would use --
    on both engines, through the same function.
    """
    import mc_voice_engines as engines

    active = _active(engine)
    adapter = engines.adapter(active)
    profiles = engines.profiles(active)

    wanted = str(text or "").strip()[:MAX_TEST_CHARS] or adapter.test_text()
    if str(text or "").strip():
        adapter.set_test_text(wanted)
    try:
        resolved, entry = engines.resolve(str(voice_id or ""), active)
    except Exception as exc:
        raise Refused(404, str(exc) or "No voice is installed.") from None
    # An audition of a voice whose delivery is being edited has to *be* that
    # delivery, or the sliders are being adjusted against a sound they do not
    # produce. A body with no profile in it means the default voice's own,
    # which is what the Settings list has always auditioned.
    delivery = None
    if profile is not None:
        if not isinstance(profile, dict):
            raise Refused(400, "That is not a delivery profile.")
        delivery = profiles.resolve(profile)
    try:
        if active == engines.SOPRO:
            import mc_voice_sopro_runtime as sopro_runtime

            audio = sopro_runtime.synthesize(wanted, resolved, profile=delivery)
        else:
            audio = runtime.synthesize(wanted, sid=int(entry.get("_sid") or 0),
                                       profile=delivery)
    except Exception as exc:
        raise Refused(503, str(exc) or "That voice could not be auditioned.") from None
    logger.info("Model Chain: a voice was auditioned on %s — %d characters, %d bytes of "
                "audio", engines.label(active), len(wanted), len(audio))
    return audio


MAX_TEST_CHARS = 400
"""The ceiling on an audition, shared by both engines because it is a property
of the control rather than of a model."""


# --------------------------------------------------------------------------- #
# The engine selector
# --------------------------------------------------------------------------- #


def engines_payload() -> dict:
    """Which TTS engines exist, and which one is selected. Section 4.

    The one payload where both engine names appear at the same time, and it
    carries nothing operational about either: an id, a label, a sentence, and
    whether it is installed. Everything a panel needs beyond that comes from
    :func:`status_payload`, which only ever describes the selected one.
    """
    import mc_voice_engines as engines

    return {"ok": True, **engines.state()}


def select_engine(engine: str) -> dict:
    """Change the active TTS engine. The whole runtime boundary. Section 4.

    Cancels speech, stops whichever TTS worker was running, persists the choice,
    and answers with the new state so every surface redraws from the truth. It
    never starts a download, never loads a model, and never switches back
    because the newly selected engine is not installed -- that state exists so
    that engine's own page can explain and install itself.

    STT is untouched (I-7). Nothing in this call path reaches Whisper's process,
    its tier or the microphone.
    """
    import mc_voice_engines as engines

    try:
        found = engines.select(str(engine or ""))
    except engines.EngineError as exc:
        raise Refused(400, str(exc)) from None
    try:
        import mc_voice_lab as lab

        lab.forget_all("the text-to-speech engine changed")
    except Exception:
        logger.debug("Model Chain: could not discard Voice Lab sessions", exc_info=True)
    return {"ok": True, **found, **status_payload()}


# --------------------------------------------------------------------------- #
# Sopro
# --------------------------------------------------------------------------- #


def _log_engine() -> None:
    """Say, at start-up, what the *selected* engine's state is.

    The line beneath this one describes the sherpa runtime, Whisper and Kokoro,
    because that is what :func:`mc_voice_models.status` knows about. When Sopro
    is the selected engine that line is true and about something else, and a log
    somebody sends after "Voice Chat does not work" cannot answer the first
    question anybody would ask of it: is the engine that is meant to be
    speaking installed, and does it have a voice?
    """
    import mc_voice_engines as engines

    try:
        active = engines.active()
        if active != engines.SOPRO:
            logger.info("Model Chain: Voice Chat text-to-speech engine — %s",
                        engines.label(active))
            return
        import mc_voice_sopro as sopro

        found = sopro.status()
        voices = len(sopro.entries())
        logger.info("Model Chain: Voice Chat text-to-speech engine — %s, runtime %s, "
                    "model %s, %d voice(s)%s", engines.label(active),
                    "installed" if found.runtime_ready else "not installed",
                    "installed" if found.model_ready else "not installed", voices,
                    "" if voices else " — replies stay silent until one is created")
    except Exception:
        logger.debug("Model Chain: could not describe the selected engine", exc_info=True)


def surface_payload() -> dict:
    """The settings markup for the engine that is selected *now*.

    Forge builds a settings row's HTML once, when the extension is imported, and
    hands that same string to every page load for the life of the process. So a
    document served after an engine switch still carries the engine that was
    selected at startup, and reloading cannot change that -- which is how the
    first version of this shipped a browser that reloaded, found the same stale
    markup, decided it was stale, and reloaded again for as long as the tab
    stayed open.

    This is the way out that keeps section 5 intact rather than trading it away.
    The browser asks for the surface, gets markup built for the engine selected
    now, and *replaces* the stale nodes with it. The inactive engine's controls
    are absent from the document afterwards because they were removed from it,
    which is the same guarantee a fresh document would have given and the only
    one available in a settings page that is built once per process.
    """
    import mc_voice_engines as engines
    import mc_voice_ui

    active = engines.active()
    return {
        "ok": True,
        "engine": active,
        "engine_label": engines.label(active),
        "settings": mc_voice_ui.settings_html(),
        "voices": mc_voice_ui.voices_html(),
    }


def sopro_payload() -> dict:
    """Everything the Sopro settings surface draws. Refused when Kokoro is active.

    Refused rather than returned empty, because the two are different bugs and
    only one of them is this route's: an empty payload reads as "Sopro has
    nothing", and a mismatch reads as "your page is out of date", which is what
    it is.
    """
    import mc_voice_engines as engines
    import mc_voice_sopro as sopro
    import mc_voice_sopro_runtime as sopro_runtime

    _active(engines.SOPRO)
    found = sopro.status()
    return {
        "ok": True,
        "engine": engines.SOPRO,
        "installed": found.ready,
        "runtime_ready": found.runtime_ready,
        "model_ready": found.model_ready,
        "runtime_message": found.runtime_message,
        "model_message": found.model_message,
        "platform_supported": found.platform_supported,
        "label": found.label,
        "fingerprint": found.fingerprint,
        "download_bytes": found.download_bytes,
        "ram_bytes": found.ram_bytes,
        "settings": sopro.engine_settings(),
        "defaults": sopro_runtime.defaults(),
        "state": sopro_runtime.engine(),
        "languages": [{"id": code, "label": label} for code, label in sopro.LANGUAGES],
        "clone": {"min_seconds": sopro.MIN_REFERENCE_SECONDS,
                  "max_seconds": sopro.MAX_REFERENCE_SECONDS,
                  "max_bytes": sopro.MAX_REFERENCE_BYTES},
        "sources": {"runtime": sopro.sources("runtime"), "model": sopro.sources("model")},
        "warnings": sopro.warnings(),
        "progress": models.progress(),
    }


def sopro_install(part: str = "", folder: str = "") -> dict:
    """Install Sopro. One button for both halves, or one half from a folder.

    Offloaded by the route in front, because this downloads a hundred and forty
    megabytes and building an isolated interpreter is not something to do on an
    event loop.
    """
    import mc_voice_sopro as sopro

    refused = sopro.refusal(manual=bool(folder))
    if refused:
        raise Refused(409, refused)
    try:
        if folder:
            sopro.install_from(str(part or "runtime"), str(folder))
        else:
            sopro.install()
    except sopro.SoproError as exc:
        raise Refused(409, str(exc)) from None
    return sopro_payload()


def sopro_settings(precision: str = "", steps=None, chunk_frames=None) -> dict:
    import mc_voice_engines as engines
    import mc_voice_sopro as sopro

    _active(engines.SOPRO)
    try:
        sopro.set_engine_settings(precision, steps, chunk_frames)
    except sopro.SoproError as exc:
        raise Refused(400, str(exc)) from None
    return sopro_payload()


def sopro_clone(name: str, language: str, wav: bytes) -> dict:
    """Create a Sopro voice from a reference recording. Returns the audition.

    The audio comes back with the payload rather than through a second route, so
    the user hears the voice that was just validated rather than one synthesized
    a moment later from a cache that might have changed (section 27).
    """
    import mc_voice_engines as engines
    import mc_voice_sopro as sopro

    _active(engines.SOPRO)
    logger.info("Model Chain: a Sopro voice is being created from a %d byte recording",
                len(wav or b""))
    if len(wav or b"") > sopro.MAX_REFERENCE_BYTES:
        logger.warning("Model Chain: a Sopro voice was not created — the recording is "
                       "larger than %d bytes", sopro.MAX_REFERENCE_BYTES)
        raise Refused(413, "That recording is too large.")
    try:
        made = sopro.create(str(name or ""), wav or b"", str(language or ""))
    except sopro.SoproError as exc:
        # Written down as well as answered. A refusal that only ever reached the
        # clone form's status line is a refusal nobody can diagnose from a log
        # afterwards -- which is how "that WAV is not 16-bit PCM" reached its
        # first user as "nothing worked", with a log that said nothing at all.
        logger.warning("Model Chain: a Sopro voice was not created — %s", exc)
        raise Refused(400, str(exc)) from None
    except Exception as exc:
        logger.warning("Model Chain: a Sopro voice could not be created", exc_info=True)
        raise Refused(500, str(exc) or "That voice could not be created.") from None
    return {"ok": True, "voice": _public(made["voice"]),
            "audio": base64.b64encode(made.get("audio") or b"").decode("ascii"),
            **voices_payload(engine=engines.SOPRO)}


def cleanup_payload() -> dict:
    """What the recording-cleanup row draws, and what the clone form asks.

    Deliberately not scoped to the selected engine: cleaning a recording is not
    a text-to-speech operation and the engine selector has no opinion about it
    (I-1). Kokoro's Storytime clone form can use it just as Sopro's can.
    """
    import mc_voice_cleanup as cleanup
    import mc_voice_cleanup_runtime as runtime

    found = cleanup.status()
    return {
        "ok": True,
        "installed": found.ready,
        "runtime_ready": found.runtime_ready,
        "model_ready": found.model_ready,
        "platform_supported": found.platform_supported,
        "runtime_message": found.runtime_message,
        "model_message": found.model_message,
        "message": found.message,
        "label": cleanup.LABEL,
        "download_bytes": found.download_bytes,
        "state": runtime.status(),
        "progress": {cleanup.KIND: models.progress().get(cleanup.KIND) or {}},
    }


def cleanup_install() -> dict:
    """Install the cleanup engine. Offloaded and threaded by the route."""
    import mc_voice_cleanup as cleanup

    try:
        cleanup.install()
    except cleanup.CleanupError as exc:
        raise Refused(409, str(exc)) from None
    return cleanup_payload()


def cleanup_run(wav: bytes) -> bytes:
    """One recording through DeepFilterNet, as a WAV in and a WAV out.

    The parsing and the re-wrapping happen here rather than in the worker,
    because the worker's job is a model and the boundary's job is refusing
    things: a worker that had to understand RIFF would be a worker with a parser
    in front of its model.
    """
    import mc_voice_cleanup_runtime as runtime

    rate, body = _mono_pcm16(wav)
    logger.info("Model Chain: cleaning %.1f s of audio with %s",
                len(body) / 2.0 / max(1, rate), "DeepFilterNet")
    try:
        cleaned = runtime.clean(body, rate)
    except runtime.CleanupRuntimeError as exc:
        logger.warning("Model Chain: a recording could not be cleaned — %s", exc)
        raise Refused(409, str(exc)) from None
    return _wav(cleaned, rate)


def _mono_pcm16(data: bytes) -> tuple:
    """A mono PCM16 WAV, taken apart. Its own parser, and narrow on purpose.

    Not :func:`validate_wav`, which is dictation's and accepts 16 kHz only, and
    not Sopro's ``normalize_reference``, which resamples and enforces a clone
    window. This one accepts exactly what the page sends -- mono PCM16 at
    whatever rate the recording had -- and refuses everything else by name.
    """
    import struct

    if len(data) > MAX_REFERENCE_BYTES:
        raise Refused(413, "That recording is too large.")
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise Refused(400, "That upload is not a WAV recording.")
    offset, fmt, body = 12, None, b""
    while offset + 8 <= len(data):
        name = data[offset:offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        start = offset + 8
        if start + size > len(data):
            raise Refused(400, "That recording's header is malformed.")
        if name == b"fmt " and size >= 16:
            fmt = struct.unpack_from("<HHIIHH", data, start)
        elif name == b"data":
            body = data[start:start + size]
        offset = start + size + (size % 2)
    if fmt is None or not body:
        raise Refused(400, "That recording is not a complete WAV.")
    encoding, channels, rate, _bps, _align, bits = fmt
    if encoding != 1 or bits != 16:
        raise Refused(400, "Recording cleanup takes uncompressed 16-bit PCM.")
    if channels != 1:
        raise Refused(400, "Recording cleanup takes a mono recording.")
    if rate < 8000 or rate > 192000:
        raise Refused(400, "That recording's sample rate is not one this build can use.")
    body = body[:len(body) - (len(body) % 2)]
    if not body:
        raise Refused(400, "That recording had no audio in it.")
    return int(rate), body


def _wav(body: bytes, rate: int) -> bytes:
    """Mono PCM16 samples wrapped as a RIFF/WAVE file."""
    import struct

    return (b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(body)) + body)


def sopro_starter() -> dict:
    """Make the next starter voice, so a fresh Sopro is not an empty one.

    One per request, and the answer says how many are left, because four
    preparations in one call is a request that takes minutes and a browser that
    gives up in the middle of it.
    """
    import mc_voice_engines as engines
    import mc_voice_sopro as sopro

    _active(engines.SOPRO)
    try:
        made = sopro.create_starter_voice()
    except sopro.SoproError as exc:
        logger.warning("Model Chain: a Sopro starter voice was not made — %s", exc)
        raise Refused(400, str(exc)) from None
    except Exception as exc:
        logger.warning("Model Chain: a Sopro starter voice could not be made", exc_info=True)
        raise Refused(500, str(exc) or "That starter voice could not be made.") from None
    return {"ok": True, "created": made.get("created") or "",
            "remaining": int(made.get("remaining") or 0),
            **voices_payload(engine=engines.SOPRO)}


def sopro_rebuild(voice_id: str) -> dict:
    """Prepare a stale voice again from its retained recording. Section 55."""
    import mc_voice_engines as engines
    import mc_voice_sopro as sopro

    _active(engines.SOPRO)
    try:
        made = sopro.rebuild(str(voice_id or ""))
    except sopro.SoproError as exc:
        raise Refused(400, str(exc)) from None
    return {"ok": True, "voice": _public(made["voice"]),
            "audio": base64.b64encode(made.get("audio") or b"").decode("ascii"),
            **voices_payload(engine=engines.SOPRO)}


# --------------------------------------------------------------------------- #
# The Voice Lab
# --------------------------------------------------------------------------- #


def lab_payload(token: str = "", voice_id: str = "") -> dict:
    """Open or read a Voice Lab session. Experimental state, never saved.

    A route of its own rather than a mode of the voice routes, which is section
    39's separation made visible at the API boundary: nothing here writes an
    option, a character or a voice asset, and there is no parameter by which it
    could be asked to.
    """
    import mc_voice_lab as lab

    try:
        found = lab.session(token).public() if token else lab.open_session(voice_id)
        return {"ok": True, "panel": lab.panel(), "session": found}
    except lab.LabError as exc:
        raise Refused(409, str(exc)) from None


def lab_update(token: str, values: dict) -> dict:
    import mc_voice_lab as lab

    try:
        return {"ok": True, "session": lab.update(token, **dict(values or {}))}
    except lab.LabError as exc:
        raise Refused(409, str(exc)) from None


def lab_reset(token: str) -> dict:
    import mc_voice_lab as lab

    try:
        return {"ok": True, "session": lab.reset(token)}
    except lab.LabError as exc:
        raise Refused(409, str(exc)) from None


def lab_audition(token: str, side: str = "b") -> dict:
    """Play A or B. The audio rides in the payload beside the run's numbers.

    Together rather than in two calls, because the whole point of the surface is
    the comparison: a first-audio time that arrived separately from the audio it
    describes would be a number nobody could attribute to a take.
    """
    import mc_voice_lab as lab

    try:
        made = lab.audition(token, side)
    except lab.LabError as exc:
        raise Refused(409, str(exc)) from None
    except Exception as exc:
        raise Refused(503, str(exc) or "That audition could not be played.") from None
    return {"ok": True, "session": made["state"],
            "audio": base64.b64encode(made.get("audio") or b"").decode("ascii")}


# --------------------------------------------------------------------------- #
# Cloning
# --------------------------------------------------------------------------- #


def cloning_payload() -> dict:
    import mc_voice_clone as cloning

    found = cloning.installation()
    found["job"] = cloning.state()
    found["ok"] = True
    return found


def cloning_install(folder: str = "") -> dict:
    import mc_voice_clone as cloning

    try:
        cloning.install(str(folder or ""))
    except cloning.CloneError as exc:
        raise Refused(409, str(exc)) from None
    return cloning_payload()


def cloning_start(name: str, language: str, wav: bytes) -> dict:
    import mc_voice_clone as cloning

    if len(wav or b"") > MAX_REFERENCE_BYTES:
        raise Refused(413, "That recording is too large.")
    try:
        cloning.start(str(name or ""), str(language or ""), wav or b"")
    except cloning.CloneError as exc:
        raise Refused(400, str(exc)) from None
    except Exception:
        logger.warning("Model Chain: a clone could not be started", exc_info=True)
        raise Refused(500, "The clone could not be started.") from None
    return cloning_payload()


def cloning_abort() -> dict:
    import mc_voice_clone as cloning

    cloning.abort()
    return cloning_payload()


def models_payload(kind: str = "stt", select: str = "") -> dict:
    """The offered speech-model tiers, and which one this installation uses.

    ``select`` changes the choice before answering, so the row that draws the
    three tiers redraws from the truth in one round trip rather than from what
    it hoped it had set.

    Choosing stops the worker. It is not a download and it is not an uninstall
    -- the bundle it names may not even be on disk yet -- but a worker that is
    already loaded is holding the *previous* tier's Whisper in memory, and the
    handshake refuses a worker whose models are not the ones this installation
    verified. Stopping it here means the next dictation starts a worker on the
    new tier instead of failing that check.
    """
    wanted = str(kind or "stt").strip() or "stt"
    if wanted not in models.OPTIONS:
        raise Refused(400, "Voice Chat only offers a choice of speech-to-text model.")
    chosen = str(select or "").strip()
    if chosen:
        try:
            before = models.default_id(wanted)
            models.select(wanted, chosen)
        except models.VoiceError as exc:
            raise Refused(400, str(exc)) from None
        if models.default_id(wanted) != before:
            try:
                runtime.unload("the speech-to-text model was changed")
            except Exception:
                logger.debug("Model Chain: could not stop the voice worker after the model "
                             "was changed", exc_info=True)
    try:
        found = models.catalogue(wanted)
    except models.VoiceError as exc:
        raise Refused(500, str(exc)) from None
    return {"ok": True, "kind": wanted, "chosen": models.default_id(wanted),
            "models": found, "progress": models.progress().get(wanted) or {}}


def profile_payload(profile=None, engine: str = "") -> dict:
    """How the default voice is delivered, and a way to change it.

    One route for both directions, like the voices list: a body with a
    ``profile`` in it writes and then answers, and a body without one only
    reads. The answer always comes from the store rather than from what was
    sent, so a value the host refused shows the slider snapping back instead of
    lying about it.
    """
    import mc_voice_engines as engines

    active = _active(engine)
    voice_profile = engines.profiles(active)
    if profile is not None:
        if not isinstance(profile, dict):
            raise Refused(400, "That is not a delivery profile.")
        voice_profile.remember(profile)
    found = voice_profile.stored()
    return {"ok": True, "engine": active, "profile": found,
            "controls": voice_profile.CONTROLS,
            "fields": list(voice_profile.FIELDS),
            "summary": voice_profile.describe(found)}


def provision(kind: str, folder: str = "", identifier: str = "") -> dict:
    """Start installing one bundle in the background, and say so immediately.

    ``folder`` installs from files already on this machine instead of
    downloading -- the escape hatch for a build whose artifacts are not pinned,
    a proxy that refuses large binaries, or an air-gapped install.

    In the background because a download is minutes and an HTTP request that
    takes minutes is a request a phone's browser will give up on. The row polls
    :func:`status_payload` for the rest, which is also what makes the Settings
    page and the Voice flyout agree without either of them owning the download.

    Everything that can be decided *before* the thread starts is decided here
    and answered in the response. That is the correction to the first version,
    which started a thread unconditionally and let it discover on the other side
    that this build cannot install anything -- so the browser was told "started"
    and the row sat on "Starting…" for as long as somebody was willing to watch
    it. A refusal a caller is waiting for belongs in the reply to that caller.
    """
    if kind not in ("stt", "tts", "runtime"):
        raise Refused(400, "Voice Chat installs the voice engine, a speech-to-text model or "
                           "a text-to-speech model.")

    already = models.progress().get(kind) or {}
    if already.get("running"):
        logger.info("Model Chain: Voice Chat was asked to install the %s model again while "
                    "it is already installing", kind.upper())
        return {"ok": True, "already": True}

    manual = str(folder or "").strip()
    wanted = str(identifier or "").strip()
    refusal = models.refusal(kind, manual=bool(manual), identifier=wanted)
    if refusal:
        logger.warning("Model Chain: Voice Chat refused to install the %s model — %s",
                       kind.upper(), refusal)
        raise Refused(409, refusal)

    logger.info("Model Chain: Voice Chat install requested for the %s model%s%s",
                kind.upper(), f" {wanted}" if wanted else "",
                " from a folder on this machine" if manual else "")

    def run():
        try:
            if kind == "runtime":
                models.install_engine(folder=manual or None)
            elif manual:
                models.install_from(kind, manual, identifier=wanted)
            else:
                models.install(kind, identifier=wanted)
        except Exception:
            # Already logged with its reason, and already recorded where the
            # Settings row will draw it -- see mc_voice_models._claim.
            logger.debug("Model Chain: the Voice Chat install thread ended on an error",
                         exc_info=True)

    threading.Thread(target=run, name=f"mc-voice-install-{kind}", daemon=True).start()
    return {"ok": True, "already": False}


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


_installed = False


def install(_demo=None, app=None) -> bool:
    """Register the routes on the WebUI's FastAPI app. Idempotent, never fatal.

    Signature is ``script_callbacks.on_app_started``'s. A UI reload calls it
    again and a second registration would give FastAPI two matching routes for
    one path, so the existing paths are looked for first -- ``mc_literal_report``
    established the pattern and this follows it.
    """
    global _installed

    if app is None or not hasattr(app, "add_api_route"):
        return False
    existing = {getattr(route, "path", None) for route in getattr(app, "routes", [])}
    if all(path in existing for path in ROUTES):
        _installed = True
        return True

    if Request is None:
        logger.debug("Model Chain: Voice Chat has no FastAPI to register routes on")
        return False

    def _refusal(exc):
        found = {"ok": False, "error": exc.reason}
        if getattr(exc, "mismatch", False):
            # The one refusal the page reacts to structurally rather than by
            # showing a message: its whole document belongs to an engine that
            # is no longer selected.
            found["engine_mismatch"] = True
        return JSONResponse(found, status_code=exc.status)

    def _failed(what: str, message: str, status: int = 500):
        logger.warning("Model Chain: %s", what, exc_info=True)
        return JSONResponse({"ok": False, "error": message}, status_code=status)

    async def voice_status(request: Request):
        try:
            _checked(request, STATUS_ROUTE)
        except Refused as exc:
            return _refusal(exc)
        # Off the loop even though it is cheap: ``models.status()`` reads the
        # disk, and a poll that stats a slow network drive should not be able
        # to hold up a reply that is being spoken.
        return JSONResponse(await _offload(status_payload))

    async def voice_stt(request: Request):
        try:
            _checked(request, STT_ROUTE)
            body = await request.body()
            return JSONResponse(await _offload(transcribe, body))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Voice Chat transcription failed",
                           "Voice transcription failed. Your message was not sent.")

    async def voice_tts(request: Request):
        try:
            _checked(request, TTS_ROUTE)
            payload = await _json(request)
            audio = await _offload(speak, str(payload.get("token") or ""))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Voice Chat synthesis failed",
                           "The reply was generated, but Voice could not read it aloud.")
        return Response(content=audio, media_type="audio/wav", headers={
            # Never cached, anywhere: a spoken reply is content, and a proxy or
            # a browser holding a copy of it is a copy nobody asked for.
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Content-Disposition": "inline",
        })

    async def voice_stream(request: Request):
        """Raw PCM for one turn, as it is produced.

        The generator below is the only place in this feature that yields a
        response body over time, and everything about it is arranged so that
        neither end can wedge the other: every queue read is offloaded, the
        client is checked for having gone away on every idle pass, and the turn
        is cancelled when the response ends for *any* reason -- normal
        completion, a disconnect, or the server going down under it.
        """
        try:
            _checked(request, STREAM_ROUTE)
            payload = await _json(request)
            turn = await _offload(open_stream, str(payload.get("turn") or ""))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Voice Chat stream could not be opened",
                           "Voice could not read that reply aloud.")

        async def pcm():
            try:
                while True:
                    kind, block = await _offload(turn.read_audio, STREAM_IDLE)
                    if kind == "end":
                        return
                    if block:
                        yield block
                        continue
                    if turn.cancelled.is_set() and not turn.busy:
                        return
                    try:
                        if await request.is_disconnected():
                            return
                    except Exception:
                        # Some hosts do not implement it. The finally below is
                        # the guarantee; this is only the early notice.
                        pass
            finally:
                # Whatever ended this response, the turn behind it is over. A
                # listener that vanished is the one condition in which the
                # worker's bounded audio queue never drains again.
                turn.cancel("stream closed")
                turn.drain_audio()
                _log_turn(turn)

        return StreamingResponse(pcm(), media_type="application/octet-stream",
                                 headers=stream_headers(turn))

    async def voice_cancel(request: Request):
        try:
            _checked(request, CANCEL_ROUTE)
            payload = await _json(request)
            return JSONResponse(cancel_turn(str(payload.get("turn") or ""),
                                            str(payload.get("reason") or "")))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Voice Chat cancel failed", "Voice could not be stopped.")

    async def voice_telemetry(request: Request):
        try:
            _checked(request, TELEMETRY_ROUTE)
            return JSONResponse(telemetry(await _json(request)))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            # Logged at debug and answered with an ordinary refusal. A page that
            # cannot report its own timings has nothing wrong with its audio,
            # and this route must never be a reason anything else stops.
            logger.debug("Model Chain: a Voice telemetry report was not recorded",
                         exc_info=True)
            return _failed("a Voice Chat timing report failed",
                           "That timing report was not recorded.")

    async def voice_runtime(request: Request):
        try:
            _checked(request, RUNTIME_ROUTE)
            payload = await _json(request)
            return JSONResponse(await _offload(set_runtime, str(payload.get("action") or "")))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Voice Chat runtime request failed",
                           "The speech engine could not be changed.")

    async def voice_install(request: Request):
        try:
            _checked(request, INSTALL_ROUTE)
            payload = await _json(request)
            return JSONResponse(provision(str(payload.get("kind") or ""),
                                          str(payload.get("folder") or ""),
                                          str(payload.get("model") or "")))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Voice Chat install request failed",
                           "Voice Chat could not start that download.")

    async def voice_models(request: Request):
        try:
            _checked(request, MODELS_ROUTE)
            payload = await _json(request)
            return JSONResponse(await _offload(models_payload,
                                               str(payload.get("kind") or "stt"),
                                               str(payload.get("select") or "")))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Voice Chat model request failed",
                           "Voice Chat could not read its own model catalogue.")

    async def voice_profile(request: Request):
        try:
            _checked(request, PROFILE_ROUTE)
            payload = await _json(request)
            # Offloaded like every other handler here. Writing a delivery is a
            # settings-file save, and section 17's rule is about *any* blocking
            # half rather than only about inference: a slider released while a
            # reply is being spoken must not put a disk write on the loop the
            # audio stream is being read from.
            return JSONResponse(await _offload(profile_payload,
                                               payload.get("profile"),
                                               str(payload.get("engine") or "")))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Voice Chat delivery request failed",
                           "Voice Chat could not read how the default voice is delivered.")

    async def voice_voices(request: Request):
        try:
            _checked(request, VOICES_ROUTE)
            payload = await _json(request)
            wanted = payload.get("test_text")
            return JSONResponse(await _offload(voices_payload,
                                               None if wanted is None else str(wanted),
                                               str(payload.get("engine") or "")))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("the voice list could not be read",
                           "The voice list could not be read.")

    def _voice_action(route: str, call):
        async def handler(request: Request):
            try:
                _checked(request, route)
                payload = await _json(request)
                return JSONResponse(await _offload(call, payload))
            except Refused as exc:
                return _refusal(exc)
            except Exception:
                return _failed(f"a voice request to {route} failed",
                               "That voice could not be changed.")

        return handler

    # ``engine`` is carried by every mutation so a page drawn before somebody
    # switched is refused rather than applied to whichever engine is selected
    # now (section 5). An absent one means "whatever is active", which is what
    # an ordinary in-sync page sends.
    voice_default = _voice_action(
        VOICE_DEFAULT_ROUTE,
        lambda payload: set_default_voice(payload.get("voice"),
                                          str(payload.get("engine") or "")))
    voice_rename = _voice_action(
        VOICE_RENAME_ROUTE,
        lambda payload: rename_voice(payload.get("voice"), payload.get("display_name"),
                                     str(payload.get("engine") or "")))
    voice_delete = _voice_action(
        VOICE_DELETE_ROUTE,
        lambda payload: delete_voice(payload.get("voice"), str(payload.get("engine") or "")))

    async def voice_test(request: Request):
        try:
            _checked(request, VOICE_TEST_ROUTE)
            payload = await _json(request)
            audio = await _offload(test_voice, str(payload.get("voice") or ""),
                                   str(payload.get("text") or ""),
                                   payload.get("profile"),
                                   str(payload.get("engine") or ""))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a voice audition failed", "That voice could not be played.")
        return Response(content=audio, media_type="audio/wav", headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Content-Disposition": "inline",
        })

    async def cloning_status(request: Request):
        try:
            _checked(request, CLONING_STATUS_ROUTE)
            return JSONResponse(await _offload(cloning_payload))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("the cloning status could not be read",
                           "The cloning status could not be read.")

    async def cloning_install_route(request: Request):
        try:
            _checked(request, CLONING_INSTALL_ROUTE)
            payload = await _json(request)
            return JSONResponse(await _offload(cloning_install,
                                               str(payload.get("folder") or "")))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a cloning install failed", "Voice cloning could not be installed.")

    async def cloning_start_route(request: Request):
        """The one route that takes a recording. Multipart, because it is a file.

        The name and the language are form fields beside it rather than a
        second request, so that a clone is one atomic thing to accept or refuse
        -- and so that a reference recording is never written for a job whose
        name was going to be rejected anyway.
        """
        try:
            _checked(request, CLONING_START_ROUTE)
            name, language, wav = await _reference(request)
            return JSONResponse(await _offload(cloning_start, name, language, wav))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a clone could not be started", "The clone could not be started.")

    async def _reference(request):
        """The clone's name, language and recording, however the host can parse it.

        Multipart is the ordinary path and is what the browser sends -- it is
        the shape a file input already produces and it does not inflate the
        bytes. It needs ``python-multipart``, which Gradio depends on and every
        Forge install therefore has; a host that somehow does not is answered by
        the base64 fallback rather than by a 500 from inside a parser.
        """
        kind = str((getattr(request, "headers", {}) or {}).get("content-type") or "")
        if "multipart/form-data" in kind:
            try:
                form = await request.form()
            except Exception:
                logger.debug("Model Chain: multipart parsing is unavailable", exc_info=True)
                raise Refused(415, "This WebUI cannot accept a file upload. Reinstall "
                                   "python-multipart, or update Voice Chat.") from None
            upload = form.get("reference")
            wav = await upload.read() if hasattr(upload, "read") else b""
            return (str(form.get("name") or ""), str(form.get("language") or ""), wav)

        body = await request.body()
        if len(body) > MAX_REFERENCE_BYTES * 2:
            raise Refused(413, "That recording is too large.")
        import base64
        import json

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
            wav = base64.b64decode(str(payload.get("reference") or ""), validate=True)
        except Exception:
            raise Refused(400, "That request did not carry a recording.") from None
        return (str(payload.get("name") or ""), str(payload.get("language") or ""), wav)

    def _json_route(route: str, call, failure: str):
        """One POST that reads a JSON body, runs off the loop, and answers JSON.

        Six of the routes below differ only in which function they call and
        what they say when it fails, so they are built rather than written out.
        A refusal keeps its own status -- an active-engine mismatch is a 409 a
        page reacts to by reloading its panel, not a 500 it shows as an error.
        """

        async def handler(request: Request):
            try:
                _checked(request, route)
                payload = await _json(request)
                return JSONResponse(await _offload(call, payload))
            except Refused as exc:
                return _refusal(exc)
            except Exception:
                return _failed(f"a Voice Chat request to {route} failed", failure)

        return handler

    voice_engines = _json_route(
        ENGINES_ROUTE, lambda _payload: engines_payload(),
        "The text-to-speech engines could not be read.")
    voice_engine_select = _json_route(
        ENGINE_SELECT_ROUTE, lambda payload: select_engine(payload.get("engine")),
        "The text-to-speech engine could not be changed.")
    voice_surface = _json_route(
        SURFACE_ROUTE, lambda _payload: surface_payload(),
        "The Voice Chat settings could not be redrawn.")
    sopro_status_route = _json_route(
        SOPRO_ROUTE, lambda _payload: sopro_payload(),
        "Sopro's status could not be read.")
    sopro_settings_route = _json_route(
        SOPRO_SETTINGS_ROUTE,
        lambda payload: sopro_settings(str(payload.get("precision") or ""),
                                       payload.get("steps"), payload.get("chunk_frames")),
        "That Sopro setting could not be changed.")
    sopro_rebuild_route = _json_route(
        SOPRO_REBUILD_ROUTE, lambda payload: sopro_rebuild(str(payload.get("voice") or "")),
        "That voice could not be rebuilt.")
    sopro_starter_route = _json_route(
        SOPRO_STARTER_ROUTE, lambda _payload: sopro_starter(),
        "That starter voice could not be made.")
    cleanup_status_route = _json_route(
        CLEANUP_ROUTE, lambda _payload: cleanup_payload(),
        "The recording cleanup status could not be read.")
    lab_route = _json_route(
        LAB_ROUTE, lambda payload: lab_payload(str(payload.get("token") or ""),
                                               str(payload.get("voice") or "")),
        "The Voice Lab could not be opened.")
    lab_update_route = _json_route(
        LAB_UPDATE_ROUTE, lambda payload: lab_update(str(payload.get("token") or ""),
                                                     payload.get("values") or {}),
        "That Voice Lab control could not be changed.")
    lab_reset_route = _json_route(
        LAB_RESET_ROUTE, lambda payload: lab_reset(str(payload.get("token") or "")),
        "The Voice Lab could not be reset.")
    lab_play_route = _json_route(
        LAB_PLAY_ROUTE, lambda payload: lab_audition(str(payload.get("token") or ""),
                                                     str(payload.get("side") or "b")),
        "That Voice Lab audition could not be played.")

    async def cleanup_install_route(request: Request):
        """Start installing the cleanup engine, and say so immediately.

        In the background for the reason Sopro's is: this is a quarter of a
        gigabyte, an interpreter and a self-test, and an HTTP request that takes
        minutes is one a browser gives up on.
        """
        try:
            _checked(request, CLEANUP_INSTALL_ROUTE)
            import mc_voice_cleanup as cleanup

            already = models.progress().get(cleanup.KIND) or {}
            if already.get("running"):
                return JSONResponse({"ok": True, "already": True})
            if not cleanup.supported_platform():
                raise Refused(409, cleanup.status().message)
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a cleanup install could not be started",
                           "Recording cleanup could not be installed.")

        def run():
            try:
                cleanup_install()
            except Exception:
                logger.debug("Model Chain: the cleanup install thread ended on an error",
                             exc_info=True)

        threading.Thread(target=run, name="mc-cleanup-install", daemon=True).start()
        return JSONResponse({"ok": True, "already": False})

    async def cleanup_run_route(request: Request):
        """A WAV in, a cleaner WAV out. Multipart, because it carries audio."""
        try:
            _checked(request, CLEANUP_RUN_ROUTE)
            _name, _language, wav = await _reference(request)
            cleaned = await _offload(cleanup_run, wav)
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a recording could not be cleaned",
                           "That recording could not be cleaned.")
        return Response(content=cleaned, media_type="audio/wav", headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Content-Disposition": "inline",
        })

    async def sopro_install_route(request: Request):
        """Start installing Sopro in the background, and say so immediately.

        In the background because this is a hundred and forty megabytes plus a
        virtual environment plus a self-test, and an HTTP request that takes
        minutes is a request a phone's browser will give up on. Everything that
        can be decided *before* the thread starts is decided here, so a refusal
        the caller is waiting for reaches that caller instead of a log.
        """
        try:
            _checked(request, SOPRO_INSTALL_ROUTE)
            payload = await _json(request)
            import mc_voice_sopro as sopro

            folder = str(payload.get("folder") or "").strip()
            part = str(payload.get("part") or "").strip() or "runtime"
            already = models.progress().get("sopro") or {}
            if already.get("running"):
                return JSONResponse({"ok": True, "already": True})
            refused = sopro.refusal(manual=bool(folder))
            if refused:
                logger.warning("Model Chain: Sopro refused to install — %s", refused)
                raise Refused(409, refused)
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Sopro install could not be started",
                           "Sopro could not be installed.")

        def run():
            try:
                sopro_install(part, folder)
            except Exception:
                # Already logged with its reason and already recorded where the
                # Settings row will draw it -- see mc_voice_models._claim.
                logger.debug("Model Chain: the Sopro install thread ended on an error",
                             exc_info=True)

        threading.Thread(target=run, name="mc-sopro-install", daemon=True).start()
        return JSONResponse({"ok": True, "already": False})

    async def sopro_clone_route(request: Request):
        """The Sopro clone route. Multipart, because it carries a recording.

        The same shape as Kokoro's cloning route and for the same reasons: the
        name and the language ride beside the file so a clone is one atomic
        thing to accept or refuse, and a reference recording is never written
        for a job whose name was going to be rejected anyway.
        """
        try:
            _checked(request, SOPRO_CLONE_ROUTE)
            name, language, wav = await _reference(request)
            return JSONResponse(await _offload(sopro_clone, name, language, wav))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Sopro voice could not be created",
                           "That voice could not be created.")

    async def cloning_abort_route(request: Request):
        try:
            _checked(request, CLONING_ABORT_ROUTE)
            return JSONResponse(await _offload(cloning_abort))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a clone could not be aborted", "The clone could not be stopped.")

    try:
        for path, handler in ((STATUS_ROUTE, voice_status), (STT_ROUTE, voice_stt),
                              (TTS_ROUTE, voice_tts), (INSTALL_ROUTE, voice_install),
                              (MODELS_ROUTE, voice_models), (PROFILE_ROUTE, voice_profile),
                              (STREAM_ROUTE, voice_stream), (CANCEL_ROUTE, voice_cancel),
                              (TELEMETRY_ROUTE, voice_telemetry),
                              (RUNTIME_ROUTE, voice_runtime), (VOICES_ROUTE, voice_voices),
                              (VOICE_DEFAULT_ROUTE, voice_default),
                              (VOICE_TEST_ROUTE, voice_test),
                              (VOICE_RENAME_ROUTE, voice_rename),
                              (VOICE_DELETE_ROUTE, voice_delete),
                              (CLONING_INSTALL_ROUTE, cloning_install_route),
                              (CLONING_STATUS_ROUTE, cloning_status),
                              (CLONING_START_ROUTE, cloning_start_route),
                              (CLONING_ABORT_ROUTE, cloning_abort_route),
                              (ENGINES_ROUTE, voice_engines),
                              (ENGINE_SELECT_ROUTE, voice_engine_select),
                              (SURFACE_ROUTE, voice_surface),
                              (SOPRO_ROUTE, sopro_status_route),
                              (SOPRO_INSTALL_ROUTE, sopro_install_route),
                              (SOPRO_SETTINGS_ROUTE, sopro_settings_route),
                              (SOPRO_CLONE_ROUTE, sopro_clone_route),
                              (SOPRO_REBUILD_ROUTE, sopro_rebuild_route),
                              (SOPRO_STARTER_ROUTE, sopro_starter_route),
                              (CLEANUP_ROUTE, cleanup_status_route),
                              (CLEANUP_INSTALL_ROUTE, cleanup_install_route),
                              (CLEANUP_RUN_ROUTE, cleanup_run_route),
                              (LAB_ROUTE, lab_route),
                              (LAB_UPDATE_ROUTE, lab_update_route),
                              (LAB_RESET_ROUTE, lab_reset_route),
                              (LAB_PLAY_ROUTE, lab_play_route)):
            if path not in existing:
                app.add_api_route(path, handler, methods=["POST"])
    except Exception:
        logger.debug("Model Chain: could not register the Voice Chat routes", exc_info=True)
        return False
    _installed = True
    found = models.status()
    # Said at start-up rather than left to be inferred from a refusal. There is
    # one gate, it is not a login, and the honest line says so.
    logger.info("Model Chain: Voice Chat routes are reachable from any page this WebUI "
                "served — they carry no login of their own and need no account anywhere")
    logger.info("Model Chain: Voice Chat routes registered at %s", ", ".join(ROUTES))
    logger.info("Model Chain: Voice Chat data directory is %s", paths.data_root())
    logger.info("Model Chain: Voice Chat host — %s", models.describe_host())
    _log_engine()
    logger.info("Model Chain: Voice Chat status — runtime %s, speech-to-text %s, "
                "text-to-speech %s", found.runtime_message, found.stt_message,
                found.tts_message)
    return True
