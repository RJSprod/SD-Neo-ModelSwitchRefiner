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
STREAM_ROUTE = f"{PREFIX}/tts-stream"
CANCEL_ROUTE = f"{PREFIX}/cancel"
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

ROUTES = (STATUS_ROUTE, STT_ROUTE, TTS_ROUTE, INSTALL_ROUTE, STREAM_ROUTE, CANCEL_ROUTE,
          RUNTIME_ROUTE, VOICES_ROUTE, VOICE_DEFAULT_ROUTE, VOICE_TEST_ROUTE,
          VOICE_RENAME_ROUTE, VOICE_DELETE_ROUTE, CLONING_INSTALL_ROUTE,
          CLONING_STATUS_ROUTE, CLONING_START_ROUTE, CLONING_ABORT_ROUTE)

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


def remember_reply(text: str) -> str:
    """Take an immutable snapshot of a completed reply. Returns its token.

    The snapshot is the design decision (R2-5, I-13). A message index would have
    been smaller and is wrong: the thread can be edited, regenerated, branched
    or switched between the run finishing and the browser asking for audio, and
    every one of those changes what an index means. A copy of the string cannot
    change.
    """
    snapshot = str(text or "")
    if not snapshot.strip():
        return ""
    token = secrets.token_urlsafe(18)
    now = time.monotonic()
    with _lock:
        _expire(now)
        _targets[token] = {"text": snapshot, "created": now, "session": _token}
    return token


def take_reply(token: str) -> str:
    """Consume one target, atomically. Empty string if there is not one.

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
        return ""
    return found["text"]


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

    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


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
        raise Refused(400, "Hold to speak.")
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
    found = models.status()
    return {
        "ok": True,
        "ready": found.ready,
        "runtime_ready": found.runtime_ready,
        "stt_ready": found.stt_ready,
        "tts_ready": found.tts_ready,
        "platform_supported": found.platform_supported,
        "runtime_message": found.runtime_message,
        "stt_message": found.stt_message,
        "tts_message": found.tts_message,
        "auto_send": state.auto_send(),
        "auto_speak": state.auto_speak(),
        "busy": found.busy,
        "progress": models.progress(),
        "sources": _sources(),
        # Live residency, which "installed" never answered. Section 30, and it
        # is cheap on purpose: reading it starts nothing.
        "engine": runtime.engine(),
        "voice": _default_voice(),
        "speaking": turns.busy(),
        "not_ready_message": ("Voice Chat is not set up. Install both models in "
                              "Settings → Voice Chat."),
    }


def _default_voice() -> dict:
    """Which voice the next reply will use, by name and by stable id.

    Never a raw SID: the number is an implementation detail of a bank the
    browser has no business addressing, and section 56 says a browser-supplied
    one is never to be trusted anyway.
    """
    try:
        import mc_voice_registry as registry

        entry = registry.default_entry()
        if entry is None:
            return {"id": "", "name": ""}
        return {"id": entry["id"], "name": entry["display_name"], "label": entry["label"],
                "custom": not entry["official"]}
    except Exception:
        logger.debug("Model Chain: could not read the default voice", exc_info=True)
        return {"id": "", "name": ""}


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
    except Exception:
        logger.debug("Model Chain: could not record voice turn metrics", exc_info=True)


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
    """The STT route's body: validate, transcribe, and let the bytes go."""
    shape = validate_wav(body)
    try:
        found = runtime.transcribe(body)
    except runtime.VoiceRuntimeError as exc:
        raise Refused(503, str(exc)) from None
    text = found.get("text") or ""
    logger.info("Model Chain: Voice STT finished — %.1f s audio in %.1f s",
                shape["seconds"], float(found.get("elapsed") or 0.0))
    if not text.strip():
        return {"ok": False, "error": "No speech was detected.", "text": ""}
    return {"ok": True, "text": text, "auto_send": state.auto_send(),
            "request_id": secrets.token_hex(6)}


def speak(token: str) -> bytes:
    """The TTS route's body: consume one target and synthesize its snapshot.

    A failure after consumption is reported, never repaired. Re-resolving "the
    current last reply" to try again would quietly speak a different message
    than the one the token stood for, which is the whole class of bug R2-5
    exists to prevent.
    """
    text = take_reply(token)
    if not text:
        raise Refused(404, "There is nothing waiting to be read aloud.")
    try:
        audio = runtime.synthesize(text)
    except runtime.VoiceRuntimeError as exc:
        raise Refused(503, str(exc)) from None
    logger.info("Model Chain: Voice TTS finished — %d characters, %d bytes of audio",
                len(text), len(audio))
    return audio


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


def set_runtime(action: str) -> dict:
    """Load or unload the Voice Worker. Section 31."""
    wanted = str(action or "").strip().casefold()
    if wanted == "load":
        try:
            return {"ok": True, "engine": runtime.load()}
        except runtime.VoiceRuntimeError as exc:
            raise Refused(503, str(exc)) from None
    if wanted == "unload":
        return {"ok": True, "engine": runtime.unload("unloaded from the Voice panel")}
    raise Refused(400, "Voice Chat can load or unload its speech engine.")


# --------------------------------------------------------------------------- #
# Voice management
# --------------------------------------------------------------------------- #


def voices_payload(test_text=None) -> dict:
    """Everything the Settings voice list draws. Section 71.

    ``test_text`` is saved when it is given, which is how the editable Test Text
    field persists without a second route: a user who types a phrase and never
    presses Test still has it next time.
    """
    import mc_voice_registry as registry

    if test_text is not None:
        try:
            registry.set_test_text(str(test_text))
        except Exception:
            logger.debug("Model Chain: could not save the voice test text", exc_info=True)
    try:
        found = registry.entries()
    except Exception:
        logger.warning("Model Chain: the voice registry could not be read", exc_info=True)
        return {"ok": False, "error": "The voice list could not be read.", "voices": []}
    return {
        "ok": True,
        "voices": [_public(entry) for entry in found],
        "default": registry.default_id(),
        "test_text": registry.test_text(),
        "capacity": registry.capacity(),
        "warnings": registry.warnings(),
    }


def _public(entry: dict) -> dict:
    """One registry entry as the browser sees it.

    The SID is not in it. A browser that knew the number could ask for a
    reserved slot or an unregistered speaker, and the answer to that is not to
    validate the number harder -- it is not to publish it.
    """
    return {key: entry[key] for key in
            ("id", "display_name", "label", "type", "official", "language", "accent",
             "editable", "deletable")}


def set_default_voice(voice_id: str) -> dict:
    import mc_voice_registry as registry

    try:
        registry.set_default(str(voice_id or ""))
    except registry.RegistryError as exc:
        raise Refused(400, str(exc)) from None
    return voices_payload()


def rename_voice(voice_id: str, display_name: str) -> dict:
    import mc_voice_registry as registry

    try:
        registry.rename(str(voice_id or ""), str(display_name or ""))
    except registry.RegistryError as exc:
        raise Refused(400, str(exc)) from None
    return voices_payload()


def delete_voice(voice_id: str) -> dict:
    import mc_voice_registry as registry

    try:
        registry.delete(str(voice_id or ""))
    except registry.RegistryError as exc:
        raise Refused(400, str(exc)) from None
    return voices_payload()


def test_voice(voice_id: str, text: str = "") -> bytes:
    """Audition one voice through the ordinary production runtime.

    "Ordinary" is the requirement (section 45): a Test that went down a
    different path would be a Test that could pass for a voice which cannot
    actually be spoken in a reply. So this is the same sherpa worker, the same
    bank and the same numeric speaker the next assistant turn would use.
    """
    import mc_voice_registry as registry

    wanted = str(text or "").strip()[:registry.MAX_TEST_CHARS] or registry.test_text()
    if str(text or "").strip():
        registry.set_test_text(wanted)
    try:
        sid, _entry = registry.resolve(str(voice_id or ""))
    except registry.RegistryError as exc:
        raise Refused(404, str(exc)) from None
    try:
        audio = runtime.synthesize(wanted, sid=sid)
    except runtime.VoiceRuntimeError as exc:
        raise Refused(503, str(exc)) from None
    logger.info("Model Chain: a voice was auditioned — %d characters, %d bytes of audio",
                len(wanted), len(audio))
    return audio


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


def provision(kind: str, folder: str = "") -> dict:
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
    refusal = models.refusal(kind, manual=bool(manual))
    if refusal:
        logger.warning("Model Chain: Voice Chat refused to install the %s model — %s",
                       kind.upper(), refusal)
        raise Refused(409, refusal)

    logger.info("Model Chain: Voice Chat install requested for the %s model%s",
                kind.upper(), " from a folder on this machine" if manual else "")

    def run():
        try:
            if kind == "runtime":
                models.install_engine(folder=manual or None)
            elif manual:
                models.install_from(kind, manual)
            else:
                models.install(kind)
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
        return JSONResponse({"ok": False, "error": exc.reason}, status_code=exc.status)

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
                                          str(payload.get("folder") or "")))
        except Refused as exc:
            return _refusal(exc)
        except Exception:
            return _failed("a Voice Chat install request failed",
                           "Voice Chat could not start that download.")

    async def voice_voices(request: Request):
        try:
            _checked(request, VOICES_ROUTE)
            payload = await _json(request)
            wanted = payload.get("test_text")
            return JSONResponse(await _offload(voices_payload,
                                               None if wanted is None else str(wanted)))
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

    voice_default = _voice_action(
        VOICE_DEFAULT_ROUTE, lambda payload: set_default_voice(payload.get("voice")))
    voice_rename = _voice_action(
        VOICE_RENAME_ROUTE,
        lambda payload: rename_voice(payload.get("voice"), payload.get("display_name")))
    voice_delete = _voice_action(
        VOICE_DELETE_ROUTE, lambda payload: delete_voice(payload.get("voice")))

    async def voice_test(request: Request):
        try:
            _checked(request, VOICE_TEST_ROUTE)
            payload = await _json(request)
            audio = await _offload(test_voice, str(payload.get("voice") or ""),
                                   str(payload.get("text") or ""))
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
                              (STREAM_ROUTE, voice_stream), (CANCEL_ROUTE, voice_cancel),
                              (RUNTIME_ROUTE, voice_runtime), (VOICES_ROUTE, voice_voices),
                              (VOICE_DEFAULT_ROUTE, voice_default),
                              (VOICE_TEST_ROUTE, voice_test),
                              (VOICE_RENAME_ROUTE, voice_rename),
                              (VOICE_DELETE_ROUTE, voice_delete),
                              (CLONING_INSTALL_ROUTE, cloning_install_route),
                              (CLONING_STATUS_ROUTE, cloning_status),
                              (CLONING_START_ROUTE, cloning_start_route),
                              (CLONING_ABORT_ROUTE, cloning_abort_route)):
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
    logger.info("Model Chain: Voice Chat status — runtime %s, speech-to-text %s, "
                "text-to-speech %s", found.runtime_message, found.stt_message,
                found.tts_message)
    return True
