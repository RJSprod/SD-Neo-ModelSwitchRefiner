"""VibeVoice speaking for Voice Chat: a reply through a card turn, whole or as it is written.

The runtime :mod:`mc_voice_engines` hands Voice Chat when VibeVoice is the
selected engine -- the same small contract Kokoro's, Sopro's and Pocket's
runtimes answer (``synthesize``, ``prepare``, ``begin_turn``, ``send_segment``,
``finish_turn``, ``interrupt_turn``, ``engine``, ``status``, ``load``,
``unload``, ``stop``, ``shutdown``), answered by an engine that is nothing like
theirs underneath.

It owns no process. The worker is the Voice Box's, one per graphics card
(:mod:`mc_voice_vibevoice_runtime`), shared with the Voice Box's own renders,
and it is on a card only while the turn system (``mc_voice_box.turns()``) says
it may be. So every reply here is three things in order: ask for the card, load
the model identity this reply needs (the runtime replaces a different one), and
render. The card is handed back in every ending -- spoken, failed, stopped --
and whether VibeVoice then stays warm on it is the turn system's decision, which
follows Settings. This module never evicts the worker and never imports the
memory side (invariant I-3): the card comes through the client the Voice Box
was given, and nothing else.

A completed reply, and Test
---------------------------
:func:`synthesize`: one card turn, bounded by :data:`SYNTH_CARD_WAIT`, the whole
text rendered as Speaker 1, a WAV back. The clone audition uses it too, with a
recording in place of a sample.

A reply spoken while it is being written
----------------------------------------
Upstream VibeVoice takes no streaming *text* input, so "speaking while the LLM
writes" is each committed sentence rendered as it arrives, with its audio
streamed out while it renders. :func:`begin_turn` answers the rate at once --
VibeVoice speaks at 24 kHz, and the stream's headers need not wait for a card --
asks for the card, and hands the rest to a thread of the turn's own, so nothing
the reply's writer calls ever waits:

    the card     The wait for the grant is on that thread. While the reply is
                 still being written it has no bound: on a card the language
                 model shares, the turn it waits behind *is* the one writing the
                 reply, and that turn is running and will end. Once the reply's
                 text is complete (:func:`finish_turn`) the grant has
                 :data:`SPEECH_CARD_WAIT` seconds to come, or the reply is not
                 read aloud and the audio fails with the turn's own reason.
                 Stop ends the wait at once, either way.
    the units    Everything written meanwhile, up to :data:`MAX_UNIT_CHARS`, is
                 one render, so a slow render gathers the sentences that arrived
                 during it rather than falling further behind one sentence at a
                 time. A unit of :data:`SHORT_UNIT_WORDS` words or fewer is held
                 back until more text or the end arrives: the Realtime model is
                 unstable on very short input.
    the audio    Every frame the worker streams goes to ``turn.offer_audio`` as
                 it arrives; a False answer there -- the turn was cancelled --
                 cancels the render.
    Stop         cancels the render in flight, drops what is queued, and the
                 turn's thread hands the card back. It never waits for the
                 model: the cancel is sent from a thread of its own.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass

import mc_voice_box as box

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

ENGINE = "vibevoice"

MODEL_7B = "vibevoice-7b"
MODEL_REALTIME = "vibevoice-realtime-0.5b"

SAMPLE_RATE = 24000
"""What VibeVoice speaks at, both models, every frame."""

LABEL = "VibeVoice — Voice Chat"
"""What a turn from here is called in the turn system's status lines."""

INTERRUPT_MODE = "cancel"
"""Stop abandons the render: the worker checks a stop flag at every step."""

SYNTH_CARD_WAIT = 120.0
"""How long a completed reply, a Test or a Load waits for its card.

A turn not granted by then is withdrawn and refused with its own reason -- an
HTTP request that waited for ever would be a Test button that never came back.
"""

SPEECH_CARD_WAIT = 90.0
"""How long a streamed reply waits for its card *after its text is complete*.

Counted from the later of :func:`begin_turn` and :func:`finish_turn`, because
until the reply is written the wait may well be for the turn that is writing
it. Not granted by then, the reply is not read aloud, with the reason.
"""

GRANT_POLL = 0.25
TAKE_POLL = 0.25
"""How often the turn's thread looks at its stop and end flags between waits."""

MAX_UNIT_CHARS = 600
"""The most text one streamed render takes: what was written meanwhile, in one."""

SHORT_UNIT_WORDS = 3
"""A unit this short waits for more text or the end before it is rendered."""

MODEL_DEFAULTS = {MODEL_REALTIME: {"steps": 5, "cfg_scale": 1.5},
                  MODEL_7B: {"steps": 10, "cfg_scale": 1.3}}
"""Each model's own guidance and steps, when the installer does not say. The
installer's record of the model (``model_info()["defaults"]``) wins."""

CHAT_DEFAULTS = {"card_uuid": "", "precision": "bf16", "lora_id": "", "lora_scale": 1.0}
"""The installer's ``settings()["chat"]``, key for key, when it has nothing."""

GRANTED = box.TURN_GRANTED
BLOCKED = box.TURN_BLOCKED
CANCELLED = box.TURN_CANCELLED
DONE = "done"
"""The turn system's words, spelled by the Voice Box so this module need not
import the turn system (``mc_turns_guests.PHASES`` holds them equal)."""

NO_CARD = ("Choose the card VibeVoice speaks on — in VibeVoice's engine settings under "
           "Settings → Voice Chat, or in the Voice Box tab's settings.")

_STEM = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
"""What a preset voice's name may be. Never a path."""


class VibeVoiceSpeechError(RuntimeError):
    """A reply VibeVoice will not or could not speak, as a sentence. Never fatal to
    Conversation: the reply is written either way."""


_lock = threading.RLock()
_speaking: dict = {}
"""The streamed replies being spoken, by their Voice Chat turn id."""
_halts: set = set()
"""One event per :func:`synthesize` or :func:`load` waiting for its card; Stop sets them."""
_renders: dict = {}
"""Whole renders in flight from :func:`synthesize`, render id to card, for Stop."""
_phases: dict = {}
"""What each piece of work here is doing -- waiting, loading, speaking -- for the
flyout's line. Content-free, keyed by an opaque id."""
_last_error = ""


# --------------------------------------------------------------------------- #
# The seams
# --------------------------------------------------------------------------- #


def _vibevoice():
    import mc_voice_vibevoice

    return mc_voice_vibevoice


def _runtime():
    import mc_voice_vibevoice_runtime

    return mc_voice_vibevoice_runtime


def _profiles():
    import mc_voice_vibevoice_profile

    return mc_voice_vibevoice_profile


# --------------------------------------------------------------------------- #
# Which card, which model, which voice
# --------------------------------------------------------------------------- #


def chat_settings() -> dict:
    """The installer's ``settings()["chat"]`` with every key present. Never raises."""
    found = dict(CHAT_DEFAULTS)
    try:
        stored = _vibevoice().settings().get("chat")
    except Exception:
        logger.debug("Model Chain: VibeVoice's Voice Chat settings could not be read",
                     exc_info=True)
        stored = None
    if isinstance(stored, dict):
        for key in CHAT_DEFAULTS:
            if stored.get(key) is not None:
                found[key] = stored[key]
    found["card_uuid"] = str(found["card_uuid"] or "").strip()
    found["precision"] = str(found["precision"] or "bf16")
    found["lora_id"] = str(found["lora_id"] or "")
    try:
        found["lora_scale"] = float(found["lora_scale"])
    except (TypeError, ValueError):
        found["lora_scale"] = 1.0
    return found


def effective_card(chat=None) -> tuple:
    """``(card, source)``: Voice Chat's own choice, else the Voice Box's, else nothing."""
    chosen = str((chat if chat is not None else chat_settings()).get("card_uuid") or "")
    if chosen.strip():
        return chosen.strip(), "chat"
    try:
        shared = str(box.settings().get("card_uuid") or "").strip()
    except Exception:
        shared = ""
    if shared:
        return shared, "voice-box"
    return "", ""


def _card() -> str:
    card, _source = effective_card()
    if not card:
        raise VibeVoiceSpeechError(NO_CARD)
    return card


def _client():
    """The turn client the Voice Box was handed, or a refusal on a WebUI with no cards."""
    client = box.turns()
    if client is None:
        raise VibeVoiceSpeechError("VibeVoice is not connected to the cards on this WebUI, "
                                   "so it cannot speak yet.")
    return client


@dataclass
class _Plan:
    """Everything one reply needs, decided before its card is asked for."""

    model: str
    card: str
    voices: dict
    precision: str = "bf16"
    lora_id: str = ""
    lora_scale: float = 1.0
    cfg_scale: float = 1.3
    steps: int = 10
    need_vram: int = 0
    need_ram: int = 0


def _delivery(model: str, profile) -> dict:
    """Guidance and steps: the profile's where it set them, the model's own otherwise."""
    found = dict(MODEL_DEFAULTS.get(model) or MODEL_DEFAULTS[MODEL_7B])
    try:
        own = (_vibevoice().model_info(model) or {}).get("defaults") or {}
        for key in ("steps", "cfg_scale"):
            if own.get(key) is not None:
                found[key] = own[key]
    except Exception:
        logger.debug("Model Chain: VibeVoice's model defaults could not be read",
                     exc_info=True)
    if isinstance(profile, dict):
        try:
            found.update(_profiles().request(profile))
        except Exception:
            logger.debug("Model Chain: could not read a VibeVoice delivery", exc_info=True)
    return {"cfg_scale": float(found["cfg_scale"]), "steps": int(found["steps"])}


def _plan(handle, profile=None) -> _Plan:
    """The model, card, voice, precision, LoRA and delivery for one reply, or a refusal.

    The model is the voice's -- a preset is the Realtime model's, a sample or a
    recording is the 7B's -- and precision and LoRA are Voice Chat's settings
    for the 7B; the Realtime model always runs at full precision with none.
    """
    wanted = dict(handle) if isinstance(handle, dict) else {}
    model = str(wanted.get("model") or "")
    stem = ""
    if model == MODEL_REALTIME:
        stem = str(wanted.get("preset") or "")
        if not _STEM.match(stem):
            raise VibeVoiceSpeechError("That is not one of the Realtime model's preset voices.")
    elif model == MODEL_7B:
        if wanted.get("reference") is None and not wanted.get("sample"):
            raise VibeVoiceSpeechError("That VibeVoice voice names no sample to speak from.")
    else:
        raise VibeVoiceSpeechError("That is not a VibeVoice voice.")
    vibevoice = _vibevoice()
    refused = str(vibevoice.refusal(model_id=model) or "")
    if refused:
        raise VibeVoiceSpeechError(refused)
    card = _card()
    if model == MODEL_REALTIME:
        voices = {1: {"preset": stem}}
        precision, lora_id, lora_scale = "bf16", "", 1.0
    else:
        if wanted.get("reference") is not None:
            pcm16, rate = bytes(wanted["reference"]), SAMPLE_RATE
        else:
            try:
                pcm16, rate = box.sample_pcm(str(wanted["sample"]))
            except box.VoiceBoxError:
                raise VibeVoiceSpeechError("That voice's sample is no longer in the Voice Box "
                                           "library.") from None
        if len(pcm16) < 2:
            raise VibeVoiceSpeechError("That voice's recording is empty.")
        voices = {1: (box.float32_of(pcm16), int(rate or SAMPLE_RATE))}
        chat = chat_settings()
        precision, lora_id, lora_scale = chat["precision"], chat["lora_id"], chat["lora_scale"]
    delivery = _delivery(model, profile)
    return _Plan(model=model, card=card, voices=voices, precision=precision,
                 lora_id=lora_id, lora_scale=lora_scale,
                 cfg_scale=delivery["cfg_scale"], steps=delivery["steps"],
                 need_vram=int(vibevoice.need_vram_bytes(model, precision) or 0),
                 need_ram=int(vibevoice.need_ram_bytes(model, precision) or 0))


# --------------------------------------------------------------------------- #
# Small shared things
# --------------------------------------------------------------------------- #


def _sentence(text: str) -> str:
    """``text`` as a sentence: a capital at the front, a stop at the end."""
    found = " ".join(str(text or "").split())
    if not found:
        return ""
    found = found[0].upper() + found[1:]
    return found if found[-1] in ".!?" else found + "."


def _not_granted(card_turn, phase: str, waited: float, written: bool = False) -> str:
    """Why a turn was not granted, with the turn's own words in it."""
    if phase == BLOCKED:
        warning = str(getattr(card_turn, "warning", "") or "").strip()
        return _sentence(f"VibeVoice could not have its card — "
                         f"{warning or 'the card could not be made ready'}")
    if phase in (CANCELLED, DONE):
        return "VibeVoice's turn on its card was withdrawn before it began."
    reason = str(getattr(card_turn, "reason", "") or "").strip() or "the card was still busy"
    if written:
        return (f"VibeVoice waited {int(waited)} seconds after the reply was written and did "
                f"not get its card ({reason}), so the reply was not read aloud.")
    return f"VibeVoice waited {int(waited)} seconds for its card and did not get it ({reason})."


def _runtime_sentence(exc: BaseException, fallback: str) -> str:
    """A refusal the runtime or the installer wrote, as a sentence; anything else, logged."""
    known = []
    for owner, name in ((_runtime, "VibeVoiceRuntimeError"), (_vibevoice, "VibeVoiceError")):
        try:
            found = getattr(owner(), name, None)
        except Exception:
            found = None
        if isinstance(found, type):
            known.append(found)
    if known and isinstance(exc, tuple(known)) and str(exc).strip():
        return _sentence(str(exc))
    logger.warning("Model Chain: VibeVoice failed in a way it did not describe", exc_info=exc)
    return fallback


def _hand_back(card_turn, granted: bool, keep_warm=None) -> None:
    """Give the card back: finished when it was granted, withdrawn when it was not.

    Withdrawn rather than finished for a turn that never began, because a turn
    that is only *finished* still clears the card and makes room before it
    notices it has nothing to do.
    """
    try:
        if granted:
            card_turn.finish(keep_warm=keep_warm)
        else:
            card_turn.cancel()
    except Exception:
        logger.warning("Model Chain: VibeVoice could not hand its card back", exc_info=True)


def _set_phase(key: str, phase: str) -> None:
    with _lock:
        _phases[key] = phase


def _clear_phase(key: str) -> None:
    with _lock:
        _phases.pop(key, None)


def _note_error(text: str) -> None:
    global _last_error

    with _lock:
        _last_error = str(text or "")


def _clear_error() -> None:
    _note_error("")


def _cancel_render(card: str, job_id: str) -> None:
    """Ask the worker to stop a render, from a thread of its own. Returns at once."""

    def run():
        try:
            _runtime().cancel(card, job_id)
        except Exception:
            logger.debug("Model Chain: a VibeVoice render could not be cancelled",
                         exc_info=True)

    threading.Thread(target=run, name="mc-vibevoice-cancel", daemon=True).start()


def _load(plan: _Plan) -> None:
    """Load this reply's model identity on its card. The runtime replaces another one."""
    try:
        _runtime().load(plan.card, model_id=plan.model, precision=plan.precision,
                        lora_id=plan.lora_id, lora_scale=plan.lora_scale)
    except VibeVoiceSpeechError:
        raise
    except Exception as exc:
        raise VibeVoiceSpeechError(
            _runtime_sentence(exc, "VibeVoice could not load its model.")) from None


def _job_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# A whole reply, and Test
# --------------------------------------------------------------------------- #


def synthesize(text: str, handle=None, profile=None) -> bytes:
    """One utterance as a WAV, through a card turn. What Test and ``/tts`` use.

    The card is asked for, waited for at most :data:`SYNTH_CARD_WAIT` seconds,
    the reply's model identity loaded, the whole text rendered as Speaker 1, and
    the card handed back -- in a ``finally``, whatever happened.
    """
    wanted = " ".join(str(text or "").split())
    if not wanted:
        raise VibeVoiceSpeechError("There is nothing to read aloud.")
    plan = _plan(handle, profile)
    client = _client()
    key = _job_id("vc-synth")
    halt = threading.Event()
    card_turn = client.request(plan.card, need_vram=plan.need_vram, need_ram=plan.need_ram,
                               label=LABEL)
    granted = False
    with _lock:
        _halts.add(halt)
        _phases[key] = "waiting"
    try:
        phase = card_turn.wait(timeout=SYNTH_CARD_WAIT, cancelled=halt)
        if phase != GRANTED:
            if halt.is_set():
                raise VibeVoiceSpeechError("VibeVoice was stopped before its card was free.")
            raise VibeVoiceSpeechError(_not_granted(card_turn, phase, SYNTH_CARD_WAIT))
        granted = True
        _set_phase(key, "loading")
        _load(plan)
        if halt.is_set():
            raise VibeVoiceSpeechError("VibeVoice was stopped before it began reading.")
        _set_phase(key, "speaking")
        runtime = _runtime()
        job = runtime.RenderJob(id=key, script=[(1, wanted)], voices=plan.voices,
                                cfg_scale=plan.cfg_scale, steps=plan.steps)
        with _lock:
            _renders[key] = plan.card
        try:
            result = runtime.render(plan.card, job, on_progress=None)
        except Exception as exc:
            raise VibeVoiceSpeechError(
                _runtime_sentence(exc, "VibeVoice could not read that aloud.")) from None
        finally:
            with _lock:
                _renders.pop(key, None)
        if getattr(result, "cancelled", False) or halt.is_set():
            raise VibeVoiceSpeechError("VibeVoice was stopped before it finished reading "
                                       "that aloud.")
        audio = bytes(getattr(result, "wav", b"") or b"")
        if not audio:
            raise VibeVoiceSpeechError("VibeVoice produced no audio for that.")
        _clear_error()
        return audio
    except VibeVoiceSpeechError as exc:
        _note_error(str(exc))
        raise
    finally:
        _hand_back(card_turn, granted)
        with _lock:
            _halts.discard(halt)
            _phases.pop(key, None)


# --------------------------------------------------------------------------- #
# A reply spoken while it is being written
# --------------------------------------------------------------------------- #


class _Speaking:
    """One Voice Chat reply on its way through a card turn."""

    def __init__(self, turn, plan: _Plan, card_turn):
        self.turn = turn
        self.plan = plan
        self.card_turn = card_turn
        self.changed = threading.Condition(threading.Lock())
        self.pending: list = []
        self.began_at = time.monotonic()
        self.ended_at = None
        """When the reply's text was complete, or ``None`` while it is still being written."""
        self.stopped = False
        self.render_id = ""
        self.renders = 0
        self.thread = None

    def halted(self) -> bool:
        return bool(self.stopped or self.turn.cancelled.is_set())

    def stop(self) -> str:
        """Stop, once: drop what is queued, and answer the render in flight to cancel."""
        with self.changed:
            if self.stopped:
                return ""
            self.stopped = True
            self.pending.clear()
            self.changed.notify_all()
            return self.render_id


def _state(turn):
    with _lock:
        return _speaking.get(getattr(turn, "id", ""))


def prepare() -> bool:
    """Whether the default voice's model is already loaded on the chat card.

    Asked while a reply is still being written, and deliberately asks for
    nothing: a card turn is not something to request for a guess. The answer
    is what a turn's metrics call a warm start.
    """
    try:
        card = _card()
        found = _runtime().loaded(card)
    except Exception:
        return False
    if not found:
        return False
    try:
        import mc_voice_vibevoice_chat as chat

        entry = chat.default_entry()
        wanted = str(((entry or {}).get("_handle") or {}).get("model") or "")
    except Exception:
        wanted = ""
    return not wanted or str(found.get("model_id") or "") == wanted


def begin_turn(turn, handle=None, profile=None) -> int:
    """Start speaking one reply. Returns the rate at once and never waits for the card.

    The rate first -- 24 kHz is known before anything is asked, and the stream's
    headers are waiting on it. Then everything that can be refused without a
    card is refused (the voice, its model, the card to use, the turn client),
    with the sentence in ``turn.audio_failed``; then the card is asked for and
    the turn's own thread takes it from there.
    """
    turn.sample_rate = SAMPLE_RATE
    turn.rate_known.set()
    try:
        plan = _plan(handle, profile)
        card_turn = _client().request(plan.card, need_vram=plan.need_vram,
                                      need_ram=plan.need_ram, label=LABEL)
    except VibeVoiceSpeechError as exc:
        _note_error(str(exc))
        turn.audio_failed(str(exc))
        raise
    except Exception:
        logger.warning("Model Chain: VibeVoice could not begin reading a reply aloud",
                       exc_info=True)
        sentence = "VibeVoice could not start reading that reply aloud."
        _note_error(sentence)
        turn.audio_failed(sentence)
        raise VibeVoiceSpeechError(sentence) from None
    state = _Speaking(turn, plan, card_turn)
    with _lock:
        _speaking[turn.id] = state
        _phases[turn.id] = "waiting"
    state.thread = threading.Thread(target=_speak, args=(state,), name="mc-vibevoice-turn",
                                    daemon=True)
    state.thread.start()
    logger.info("Model Chain: VibeVoice asked for its card to read turn %s aloud",
                str(turn.id)[:8])
    return SAMPLE_RATE


def send_segment(turn, text: str) -> None:
    """Queue one committed sentence. Only queues: never waits for anything."""
    state = _state(turn)
    wanted = str(text or "")
    if state is None or not wanted.strip():
        return
    with state.changed:
        if state.stopped or state.ended_at is not None:
            return
        state.pending.append(wanted)
        state.changed.notify_all()


def finish_turn(turn) -> None:
    """The reply's text is complete. From now, the card has a bound to come in."""
    state = _state(turn)
    if state is None:
        return
    with state.changed:
        if state.ended_at is None:
            state.ended_at = time.monotonic()
        state.changed.notify_all()


def interrupt_turn(turn) -> None:
    """Stop this reply. Idempotent, and returns at once.

    Drops what is queued and asks the worker to cancel the render in flight --
    from a thread of its own, so Stop never waits for the model. The turn's own
    thread notices within :data:`TAKE_POLL`, and hands the card back when the
    render in flight has returned: the card is VibeVoice's until the worker has
    actually stopped using it.
    """
    state = _state(turn)
    if state is None:
        return
    render_id = state.stop()
    if render_id:
        _cancel_render(state.plan.card, render_id)


def cancel_turn(turn) -> None:
    """The shared name. On this engine it is exactly :func:`interrupt_turn`."""
    interrupt_turn(turn)


def _speak(state: _Speaking) -> None:
    """The turn's own thread: the card, the model, the units, and the card back."""
    turn = state.turn
    granted = False
    completed = False
    try:
        granted = _await_card(state)
        if not granted or state.halted():
            return
        _set_phase(turn.id, "loading")
        _load(state.plan)
        if state.halted():
            return
        _set_phase(turn.id, "speaking")
        completed = _render_units(state)
    except VibeVoiceSpeechError as exc:
        if not state.halted():
            _note_error(str(exc))
            turn.audio_failed(str(exc))
    except Exception:
        logger.warning("Model Chain: VibeVoice stopped reading a reply aloud on an error",
                       exc_info=True)
        if not state.halted():
            turn.audio_failed("VibeVoice could not read that reply aloud.")
    finally:
        # The card first, then the end of the audio: by the time anything reads
        # this turn as finished, the card is already the turn system's again.
        _hand_back(state.card_turn, granted)
        with _lock:
            if _speaking.get(turn.id) is state:
                _speaking.pop(turn.id, None)
            _phases.pop(turn.id, None)
        if completed and not state.halted():
            _clear_error()
            turn.audio_finished()


def _await_card(state: _Speaking) -> bool:
    """Wait for the grant. True when granted; False when stopped or refused.

    Unbounded while the reply is still being written; :data:`SPEECH_CARD_WAIT`
    seconds from the moment it was complete. A refusal or a timeout fails the
    turn's audio with the turn's own words and returns False, and the caller
    withdraws the request.
    """
    turn = state.turn
    card_turn = state.card_turn
    while True:
        if state.halted():
            return False
        with state.changed:
            ended_at = state.ended_at
        remaining = None
        if ended_at is not None:
            remaining = (max(ended_at, state.began_at) + SPEECH_CARD_WAIT
                         - time.monotonic())
            if remaining <= 0:
                sentence = _not_granted(card_turn, str(getattr(card_turn, "phase", "")),
                                        SPEECH_CARD_WAIT, written=True)
                _note_error(sentence)
                turn.audio_failed(sentence)
                return False
        step = GRANT_POLL if remaining is None else max(0.01, min(GRANT_POLL, remaining))
        phase = card_turn.wait(timeout=step, cancelled=turn.cancelled)
        if phase == GRANTED:
            return True
        if phase in (BLOCKED, CANCELLED, DONE):
            if state.halted():
                return False
            sentence = _not_granted(card_turn, phase, 0.0)
            _note_error(sentence)
            turn.audio_failed(sentence)
            return False


def _words(text: str) -> int:
    return len(str(text or "").split())


def _unit_of(pending: list) -> tuple:
    """``(text, used)``: the next unit from the front of the queue.

    Everything queued, up to :data:`MAX_UNIT_CHARS` -- except that a unit of
    :data:`SHORT_UNIT_WORDS` words or fewer keeps taking the next sentence even
    past that, because a short unit alone is the one thing not to render.
    """
    parts = []
    size = 0
    words = 0
    used = 0
    for item in pending:
        piece = " ".join(str(item or "").split())
        if (piece and parts and size + 1 + len(piece) > MAX_UNIT_CHARS
                and words > SHORT_UNIT_WORDS):
            break
        used += 1
        if piece:
            size += len(piece) + (1 if parts else 0)
            words += _words(piece)
            parts.append(piece)
    return " ".join(parts), used


def _take(state: _Speaking):
    """The next unit to render, or ``None`` when the reply is over or stopped."""
    with state.changed:
        while True:
            if state.halted():
                return None
            text, used = _unit_of(state.pending)
            ended = state.ended_at is not None
            if used and (ended or _words(text) > SHORT_UNIT_WORDS):
                del state.pending[:used]
                if text:
                    return text
                continue
            if ended and not state.pending:
                return None
            state.changed.wait(TAKE_POLL)


def _render_units(state: _Speaking) -> bool:
    """Render unit after unit until the reply is over. True when it ended by itself."""
    while True:
        text = _take(state)
        if text is None:
            return not state.halted()
        if not _render_unit(state, text):
            return False


def _render_unit(state: _Speaking, text: str) -> bool:
    """One unit, streamed into the turn as it renders. False when it was stopped."""
    runtime = _runtime()
    turn = state.turn
    plan = state.plan
    state.renders += 1
    job_id = f"vc-{str(turn.id)[:10]}-{state.renders}"
    seen = {"frames": 0, "samples": 0, "first": 0.0}
    started = time.monotonic()

    def on_audio(pcm16: bytes, rate: int) -> bool:
        # On the runtime's reader thread. ``offer_audio`` applies the browser's
        # backpressure and answers False once the turn is cancelled, which is
        # what cancels the render.
        if state.halted():
            return False
        if not seen["frames"]:
            seen["first"] = time.monotonic()
        seen["frames"] += 1
        seen["samples"] += len(pcm16 or b"") // 2
        return bool(turn.offer_audio(pcm16, int(rate or SAMPLE_RATE)))

    job = runtime.RenderJob(id=job_id, script=[(1, text)], voices=plan.voices,
                            cfg_scale=plan.cfg_scale, steps=plan.steps, stream=True)
    with state.changed:
        if state.halted():
            return False
        state.render_id = job_id
    try:
        result = runtime.render(plan.card, job, on_progress=None, on_audio=on_audio)
    except Exception as exc:
        if state.halted():
            return False
        raise VibeVoiceSpeechError(
            _runtime_sentence(exc, "VibeVoice could not read that reply aloud.")) from None
    finally:
        with state.changed:
            state.render_id = ""
    first_ms = int(getattr(result, "first_audio_ms", 0) or 0)
    if not first_ms and seen["frames"]:
        first_ms = int((seen["first"] - started) * 1000)
    try:
        turn.note_segment(blocks=seen["frames"], first_block_ms=first_ms,
                          streaming="callback",
                          synth_ms=int((time.monotonic() - started) * 1000),
                          audio_ms=int(seen["samples"] * 1000 / SAMPLE_RATE))
    except Exception:
        logger.debug("Model Chain: a VibeVoice unit's timing could not be recorded",
                     exc_info=True)
    if getattr(result, "cancelled", False) or state.halted():
        return False
    return True


# --------------------------------------------------------------------------- #
# What the surfaces read, and the lifecycle
# --------------------------------------------------------------------------- #


def declared_interrupt_mode() -> str:
    return INTERRUPT_MODE


def _loaded(card: str):
    if not card:
        return None
    try:
        found = _runtime().loaded(card)
    except Exception:
        logger.debug("Model Chain: could not read what VibeVoice holds", exc_info=True)
        return None
    return found if isinstance(found, dict) and found else None


def _where(card: str) -> str:
    """The card's name as the worker's handshake gave it, when there is one."""
    try:
        rows = (_runtime().status(card) or {}).get("cards") or {}
    except Exception:
        return ""
    for row in rows.values():
        name = str((row or {}).get("device_name") or "").strip()
        if name:
            return name
    return ""


def engine() -> dict:
    """The flyout's Loaded/Unloaded line. Reads; never asks for a card or starts anything.

    ``loaded`` is whether the chat card's worker holds a model; ``state`` is one
    of the flyout's words -- ``loading`` while a reply waits for its card or its
    model, ``tts`` while one is rendering, ``idle`` when a model is resident and
    nothing is asked of it. No path, no pid, no text.
    """
    card, _source = effective_card()
    loaded = _loaded(card)
    with _lock:
        phases = set(_phases.values())
        error = _last_error
    if phases & {"waiting", "loading"}:
        state = "loading"
    elif "speaking" in phases:
        state = "tts"
    elif loaded:
        state = "idle"
    elif error:
        state = "error"
    else:
        state = "unloaded"
    found = {
        "loaded": bool(loaded),
        "state": state,
        "backend": ENGINE,
        "card": card,
        "where": (_where(card) if card else "") or "its card",
        "model": str((loaded or {}).get("model_id") or ""),
        "precision": str((loaded or {}).get("precision") or ""),
        "lora": bool((loaded or {}).get("lora_id")),
        "error": error if state == "error" else "",
    }
    return found


def status() -> dict:
    """What the playback control reads: loaded, busy, and what Stop means."""
    card, _source = effective_card()
    with _lock:
        busy = bool(_phases)
        error = _last_error
    return {"loaded": bool(_loaded(card)), "busy": busy, "draining": False,
            "interrupt_mode": INTERRUPT_MODE, "error": error}


def load() -> dict:
    """Load the default voice's model on the chat card, through a turn, and stay warm.

    The flyout's Load. Bounded like :func:`synthesize`; the card is handed back
    asking to stay warm, which the turn system honours as far as Settings allow.
    """
    import mc_voice_vibevoice_chat as chat

    entry = chat.default_entry()
    if entry is None:
        raise VibeVoiceSpeechError(chat.NO_VOICE)
    plan = _plan(entry.get("_handle"), None)
    client = _client()
    key = _job_id("vc-load")
    halt = threading.Event()
    card_turn = client.request(plan.card, need_vram=plan.need_vram, need_ram=plan.need_ram,
                               label=LABEL)
    granted = False
    with _lock:
        _halts.add(halt)
        _phases[key] = "waiting"
    try:
        phase = card_turn.wait(timeout=SYNTH_CARD_WAIT, cancelled=halt)
        if phase != GRANTED:
            if halt.is_set():
                raise VibeVoiceSpeechError("VibeVoice was stopped before its card was free.")
            raise VibeVoiceSpeechError(_not_granted(card_turn, phase, SYNTH_CARD_WAIT))
        granted = True
        _set_phase(key, "loading")
        _load(plan)
        _clear_error()
    except VibeVoiceSpeechError as exc:
        _note_error(str(exc))
        raise
    finally:
        _hand_back(card_turn, granted, keep_warm=True)
        with _lock:
            _halts.discard(halt)
            _phases.pop(key, None)
    return engine()


def unload(reason: str = "unloaded") -> dict:
    """Let go of the chat card through the turn client, so what made room comes back.

    Voice Chat's own speech is stopped first. The worker is the Voice Box's too:
    the client decides whether it is a warm stay to end or a worker to evict.
    """
    stop(reason or "unloaded")
    card, _source = effective_card()
    client = box.turns()
    if card and client is not None:
        try:
            client.unload(card, reason or "unloaded from the Voice panel")
        except Exception:
            logger.warning("Model Chain: VibeVoice could not be unloaded", exc_info=True)
    return engine()


def stop(reason: str = "") -> None:
    """Stop Voice Chat's own VibeVoice speech. Never raises, never evicts.

    Every reply being spoken is cancelled, every render this module started is
    asked to stop, and every request still waiting for a card is withdrawn. The
    worker itself stays: it is shared with the Voice Box, which may be
    rendering, and whether VibeVoice stays warm on a card is the turn system's
    decision rather than this engine's.
    """
    why = str(reason or "stopped")
    with _lock:
        states = list(_speaking.values())
        halts = list(_halts)
        renders = dict(_renders)
    for state in states:
        try:
            state.turn.cancel(why)
            state.turn.drain_audio()
        except Exception:
            logger.debug("Model Chain: a VibeVoice turn could not be cancelled", exc_info=True)
        render_id = state.stop()
        if render_id:
            _cancel_render(state.plan.card, render_id)
    for halt in halts:
        halt.set()
    for job_id, card in renders.items():
        _cancel_render(card, job_id)


def shutdown() -> None:
    """The WebUI is closing. Stop speaking and forget the pending preview. Never raises.

    The worker process is :mod:`mc_voice_vibevoice_runtime`'s to end, from its
    own shutdown door; this one is Voice Chat's half.
    """
    try:
        stop("WebUI shutdown")
    except Exception:
        logger.debug("Model Chain: VibeVoice's Voice Chat speech did not stop cleanly",
                     exc_info=True)
    try:
        import mc_voice_vibevoice_chat as chat

        chat.discard_preview()
    except Exception:
        logger.debug("Model Chain: the VibeVoice preview could not be forgotten",
                     exc_info=True)
