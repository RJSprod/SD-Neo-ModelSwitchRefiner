"""VibeVoice speaking for Voice Chat, against a fake turn client and a fake runtime.

``mc_voice_vibevoice_speech`` is the one Voice Chat runtime that owns no process:
it asks the turn system for a card, loads a model identity there, and renders.
So what is tested here is the order of those three and every way out of them --
the card granted, refused, never granted, withdrawn by Stop -- and the streamed
reply end to end through a real :class:`mc_voice_turn.VoiceTurn`: the rate known
at once, the sentences written during a slow render gathered into one, a unit of
three words held back, a cancel in the middle of a render.

The installer and the runtime are stood in through ``sys.modules`` (the module
imports both lazily), and the turn client through ``mc_voice_box.use_turns``,
which is the only way a voice module is allowed to reach a card. The fakes live
here and ``tests/test_voice_vibevoice_chat.py`` imports them.
"""

from __future__ import annotations

import array
import io
import math
import sys
import threading
import time
import types
import wave
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import mc_voice_box as box
import mc_voice_turn as turns
import mc_voice_vibevoice_speech as speech

MODEL_7B = "vibevoice-7b"
MODEL_REALTIME = "vibevoice-realtime-0.5b"
CARD = "GPU-11111111-2222-3333-4444-555555555555"
BOX_CARD = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
PRESET = {"model": MODEL_REALTIME, "preset": "en-Carter_man"}
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake"
WAITING = "waiting for the language model to finish"


def tone_wav(seconds: float = 5.0, rate: int = 24000, amplitude: float = 0.3) -> bytes:
    """A mono 16-bit WAV of a tone: a recording the Voice Box's envelope accepts."""
    samples = array.array("h", (int(amplitude * 32767 * math.sin(2 * math.pi * 220 * i / rate))
                                for i in range(int(seconds * rate))))
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return out.getvalue()


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


# --------------------------------------------------------------------------- #
# The fakes
# --------------------------------------------------------------------------- #


class FakeCardTurn:
    """``mc_turns.Turn`` as a guest sees it: phase, reason, warning, wait, finish, cancel."""

    def __init__(self, card, need_vram, need_ram, label):
        self.card = card
        self.need_vram = need_vram
        self.need_ram = need_ram
        self.label = label
        self.phase = "queued"
        self.reason = WAITING
        self.warning = ""
        self.finishes = []
        self.cancels = 0
        self._changed = threading.Condition()

    def grant(self):
        with self._changed:
            self.phase = "granted"
            self._changed.notify_all()

    def block(self, warning):
        with self._changed:
            self.phase = "blocked"
            self.warning = warning
            self._changed.notify_all()

    def wait(self, timeout=None, cancelled=None):
        deadline = None if timeout is None else time.monotonic() + max(float(timeout), 0.0)
        with self._changed:
            while self.phase not in ("granted", "blocked", "cancelled", "done"):
                if cancelled is not None and cancelled.is_set():
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return self.phase
                self._changed.wait(0.01 if remaining is None else min(remaining, 0.01))
            else:
                return self.phase
        self.cancel()
        return "cancelled"

    def finish(self, keep_warm=None):
        self.finishes.append(keep_warm)

    def cancel(self):
        with self._changed:
            self.cancels += 1
            if self.phase != "granted":
                self.phase = "cancelled"
            self._changed.notify_all()

    @property
    def handed_back(self) -> bool:
        return bool(self.finishes) or self.cancels > 0


class FakeClient:
    """``mc_turns_guests.TurnClient``, without a card behind it."""

    def __init__(self):
        self.requests = []
        self.grant_at_once = True
        self.unloads = []
        self.card_rows = [
            {"uuid": CARD, "key": "1111", "index": 0, "name": "NVIDIA GeForce RTX 3090",
             "image_card": True, "wangp_card": False},
            {"uuid": BOX_CARD, "key": "aaaa", "index": 1, "name": "NVIDIA GeForce RTX 5090",
             "image_card": False, "wangp_card": True},
        ]

    def request(self, card_uuid, *, need_vram, need_ram=0, label=""):
        found = FakeCardTurn(card_uuid, need_vram, need_ram, label)
        if self.grant_at_once:
            found.grant()
        self.requests.append(found)
        return found

    def cards(self):
        return [dict(row) for row in self.card_rows]

    def unload(self, card_uuid, reason):
        self.unloads.append((card_uuid, reason))
        return 0

    def snapshot(self):
        return []


class FakeInstaller(types.ModuleType):
    """``mc_voice_vibevoice`` by the contract's names, with nothing installed for real."""

    MODEL_7B = MODEL_7B
    MODEL_DEFAULT = MODEL_7B
    MODEL_REALTIME = MODEL_REALTIME
    KIND_LONGFORM = "longform"
    KIND_REALTIME = "realtime"
    PRECISIONS = ("bf16", "int8", "nf4")
    PRECISION_LABELS = {"bf16": "Full (bf16)", "int8": "8-bit", "nf4": "4-bit (NF4)"}
    LABEL = "VibeVoice"

    class VibeVoiceError(RuntimeError):
        pass

    def __init__(self):
        super().__init__("mc_voice_vibevoice")
        self.supported = True
        self.runtime_installed = True
        self.installed = {MODEL_7B: True, MODEL_REALTIME: True}
        self.preset_rows = [
            {"id": "de-Spk0_man", "name": "Spk0", "language": "de", "language_label": "German",
             "gender": "man", "experimental": True, "installed": True},
            {"id": "en-Carter_man", "name": "Carter", "language": "en",
             "language_label": "English", "gender": "man", "experimental": False,
             "installed": True},
            {"id": "en-Emma_woman", "name": "Emma", "language": "en",
             "language_label": "English", "gender": "woman", "experimental": False,
             "installed": True},
            {"id": "fr-Spk1_woman", "name": "Spk1", "language": "fr", "language_label": "French",
             "gender": "woman", "experimental": True, "installed": False},
        ]
        self.chat = {"card_uuid": CARD, "precision": "bf16", "lora_id": "", "lora_scale": 1.0}
        self.refused = {}
        self.defaults = {MODEL_7B: {"steps": 10, "cfg_scale": 1.3},
                         MODEL_REALTIME: {"steps": 5, "cfg_scale": 1.5}}
        self.lora_rows = [{"id": "warm01", "name": "Warm", "base": MODEL_7B, "bytes": 1024,
                           "parts": ["llm"], "created": 1.0, "adapter": "/private/path"}]
        self.writes = []

    def status(self, identifier=""):
        return SimpleNamespace(
            supported=self.supported, runtime_installed=self.runtime_installed,
            runtime_message=("Installed — VibeVoice 0.0.1." if self.runtime_installed
                             else "Not installed — about 3.1 GB of PyTorch."))

    def presets(self, identifier=MODEL_REALTIME):
        return [dict(row) for row in self.preset_rows]

    def model_info(self, identifier=""):
        identifier = identifier or MODEL_7B
        if identifier not in (MODEL_7B, MODEL_REALTIME):
            raise self.VibeVoiceError(f"{identifier!r} is not a VibeVoice model.")
        realtime = identifier == MODEL_REALTIME
        installed = bool(self.installed[identifier])
        return {
            "id": identifier,
            "label": "VibeVoice Realtime 0.5B" if realtime else "VibeVoice 7B",
            "kind": "realtime" if realtime else "longform",
            "installed": installed,
            "runtime_installed": self.runtime_installed,
            "precisions": ["bf16"] if realtime else list(self.PRECISIONS),
            "lora": not realtime,
            "max_speakers": 1 if realtime else 4,
            "voices": "presets" if realtime else "samples",
            "defaults": dict(self.defaults[identifier]),
            "need_vram_bytes": ({"bf16": 3_000_000_000} if realtime else
                                {"bf16": 20_000_000_000, "int8": 12_000_000_000,
                                 "nf4": 8_000_000_000}),
            "need_ram_bytes": 4_000_000_000,
            "presets": self.presets() if realtime else [],
            "message": "" if installed else "Not installed — about 2 GB.",
            "download_bytes": 0 if installed else 2_000_000_000,
            # Something the adapter must not publish.
            "model_dir": "/private/models/" + identifier,
        }

    def models_info(self):
        return [self.model_info(MODEL_7B), self.model_info(MODEL_REALTIME)]

    def refusal(self, manual=False, model_id=""):
        wanted = model_id or MODEL_7B
        if wanted in self.refused:
            return self.refused[wanted]
        if not self.runtime_installed:
            return "VibeVoice's runtime is not installed — install it below."
        if not self.installed.get(wanted):
            return "The model is not installed — install it below."
        return ""

    def need_vram_bytes(self, identifier="", precision=""):
        return self.model_info(identifier)["need_vram_bytes"][precision or "bf16"]

    def need_ram_bytes(self, identifier="", precision=""):
        return 4_000_000_000

    def settings(self):
        return {"model_id": MODEL_7B, "card_uuid": "", "steps": 10, "chat": dict(self.chat)}

    def set_settings(self, values):
        offered = dict(values)
        for key in offered:
            if key != "chat":
                raise self.VibeVoiceError(f"{key!r} is not a VibeVoice setting.")
        chat = dict(offered.get("chat") or {})
        for key in chat:
            if key not in ("card_uuid", "precision", "lora_id", "lora_scale"):
                raise self.VibeVoiceError(f"{key!r} is not a VibeVoice chat setting.")
        if "precision" in chat and chat["precision"] not in self.PRECISIONS:
            raise self.VibeVoiceError("VibeVoice 7B cannot run at that precision.")
        if chat.get("lora_id") and chat["lora_id"] not in {row["id"] for row in self.lora_rows}:
            raise self.VibeVoiceError("That LoRA is no longer in the library.")
        if "lora_scale" in chat and not 0.0 <= float(chat["lora_scale"]) <= 2.0:
            raise self.VibeVoiceError("The LoRA's strength is from 0 to 2.")
        self.chat.update(chat)
        self.writes.append(chat)
        return self.settings()

    def loras(self):
        return [dict(row) for row in self.lora_rows]

    def progress(self):
        return {}


class FakeRuntime(types.ModuleType):
    """``mc_voice_vibevoice_runtime`` by the contract's names: load, loaded, render, cancel."""

    class VibeVoiceRuntimeError(RuntimeError):
        pass

    @dataclass
    class RenderJob:
        id: str
        script: list = field(default_factory=list)
        voices: dict = field(default_factory=dict)
        cfg_scale: float = 1.3
        steps: int = 10
        seed: "int | None" = None
        max_new_tokens: "int | None" = None
        stream: bool = False

    @dataclass
    class RenderResult:
        wav: bytes
        seconds: float
        sample_rate: int
        render_seconds: float
        peak_bytes: int
        cancelled: bool
        tokens: int
        capped: bool = False
        first_audio_ms: int = 0
        streamed: bool = False

    def __init__(self):
        super().__init__("mc_voice_vibevoice_runtime")
        self.loads = []
        self.renders = []
        self.cancels = []
        self.evicts = []
        self.identity = {}
        self.frames = 2
        self.hold_after_first = False
        self.release = threading.Event()
        self.first_frame = threading.Event()
        self.cancel_delay = 0.0
        self.fail_render = ""
        self._cancelled = set()
        self._lock = threading.Lock()

    def load(self, card_uuid, *, model_id="", precision="", lora_id="", lora_scale=1.0):
        self.loads.append({"card": card_uuid, "model_id": model_id, "precision": precision,
                           "lora_id": lora_id, "lora_scale": lora_scale})
        self.identity[card_uuid] = {"model_id": model_id, "precision": precision,
                                    "lora_id": lora_id, "lora_scale": lora_scale,
                                    "kind": "realtime" if model_id == MODEL_REALTIME
                                    else "longform"}
        return {"loaded": True}

    def loaded(self, card_uuid):
        return self.identity.get(card_uuid)

    def render(self, card_uuid, job, on_progress=None, on_audio=None):
        with self._lock:
            self.renders.append((card_uuid, job))
        if self.fail_render:
            raise self.VibeVoiceRuntimeError(self.fail_render)
        if on_audio is None:
            return self.RenderResult(wav=WAV, seconds=1.0, sample_rate=24000, render_seconds=0.1,
                                     peak_bytes=1, cancelled=job.id in self._cancelled,
                                     tokens=3)
        cancelled = False
        for index in range(self.frames):
            if job.id in self._cancelled:
                cancelled = True
                break
            if on_audio(b"\x01\x00" * 240, 24000) is False:
                cancelled = True
                break
            self.first_frame.set()
            if index == 0 and self.hold_after_first:
                while not self.release.wait(0.01):
                    if job.id in self._cancelled:
                        cancelled = True
                        break
                if cancelled:
                    break
        return self.RenderResult(wav=b"", seconds=0.5, sample_rate=24000, render_seconds=0.1,
                                 peak_bytes=1, cancelled=cancelled, tokens=3,
                                 first_audio_ms=7, streamed=True)

    def cancel(self, card_uuid, job_id):
        if self.cancel_delay:
            time.sleep(self.cancel_delay)
        self.cancels.append((card_uuid, job_id))
        self._cancelled.add(job_id)
        return True

    def status(self, card_uuid=""):
        return {"cards": {"1111": {"device_name": "NVIDIA GeForce RTX 3090"}},
                "last_error": ""}

    def evict(self, card_uuid, reason=""):
        self.evicts.append(card_uuid)
        return 0

    def texts(self):
        return [job.script[0][1] for _card, job in self.renders]


@pytest.fixture
def vibevoice(voice_root, monkeypatch):
    """The installer, the runtime and the turn client, all stood in."""
    installer = FakeInstaller()
    runtime = FakeRuntime()
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mc_voice_vibevoice", installer)
    monkeypatch.setitem(sys.modules, "mc_voice_vibevoice_runtime", runtime)
    previous = box.turns()
    box.use_turns(client)
    try:
        yield SimpleNamespace(installer=installer, runtime=runtime, client=client)
    finally:
        runtime.release.set()
        speech.shutdown()
        with speech._lock:
            threads = [state.thread for state in speech._speaking.values() if state.thread]
        for thread in threads:
            thread.join(2.0)
        box.use_turns(previous)
        with speech._lock:
            speech._speaking.clear()
            speech._halts.clear()
            speech._renders.clear()
            speech._phases.clear()
        speech._note_error("")
        turns.forget_all("test finished")


def voice_turn(handle=None) -> turns.VoiceTurn:
    """A real Voice Chat turn, with a browser already listening to it."""
    found = turns.VoiceTurn(voice_id="vibevoice:preset:en-Carter_man", engine="vibevoice",
                            handle=PRESET if handle is None else handle,
                            interrupt_mode="cancel")
    found.attached.set()
    return found


def audio_of(turn, timeout: float = 3.0) -> list:
    """Every block the turn hands its stream, until the stream ends."""
    blocks = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        kind, payload = turn.read_audio(0.05)
        if kind == "audio":
            blocks.append(payload)
        elif kind == "end":
            break
    return blocks


def state_gone(turn) -> bool:
    with speech._lock:
        return turn.id not in speech._speaking


# --------------------------------------------------------------------------- #
# A streamed reply
# --------------------------------------------------------------------------- #


class TestTheRateIsKnownAtOnce:
    def test_the_rate_is_answered_before_the_card_is_granted(self, vibevoice):
        """The stream's headers wait on ``rate_known``, and a card can take minutes."""
        vibevoice.client.grant_at_once = False
        turn = voice_turn()
        started = time.monotonic()
        rate = speech.begin_turn(turn, PRESET, None)
        assert rate == 24000
        assert time.monotonic() - started < 0.5, "begin_turn waited for the card"
        assert turn.rate_known.is_set() and turn.sample_rate == 24000
        assert vibevoice.client.requests[0].phase == "queued"
        assert vibevoice.runtime.loads == []
        turn.cancel("user")
        speech.interrupt_turn(turn)
        assert wait_for(lambda: state_gone(turn))


class TestAReplySpokenAsItIsWritten:
    def test_end_to_end_through_a_real_voice_turn(self, vibevoice):
        """The turn's own pump: prepare, begin, segments, finish, audio, the card back."""
        turn = turns.create(voice_id="vibevoice:preset:en-Carter_man", engine="vibevoice",
                            handle=PRESET, interrupt_mode="cancel", speaker=speech)
        turn.attached.set()
        turn.start()
        turn.add_text("Hello there, this is the first sentence of the reply. ")
        turn.complete("Hello there, this is the first sentence of the reply. And here is "
                      "a second sentence for the voice to read.")
        blocks = audio_of(turn)
        assert blocks, "nothing was streamed"
        assert turn.synthesis_done and not turn.error
        (card_turn,) = vibevoice.client.requests
        assert card_turn.card == CARD
        assert card_turn.label == "VibeVoice — Voice Chat"
        assert card_turn.need_vram == 3_000_000_000
        assert card_turn.finishes == [None], "the card was not handed back to the setting"
        assert vibevoice.runtime.loads == [{"card": CARD, "model_id": MODEL_REALTIME,
                                            "precision": "bf16", "lora_id": "",
                                            "lora_scale": 1.0}]
        spoken = " ".join(vibevoice.runtime.texts())
        assert "first sentence of the reply" in spoken and "second sentence" in spoken
        for _card, job in vibevoice.runtime.renders:
            assert job.stream is True
            assert job.voices == {1: {"preset": "en-Carter_man"}}
            assert (job.cfg_scale, job.steps) == (1.5, 5), "not the Realtime model's own"
        assert turn.metrics()["streaming"] == "callback"
        assert wait_for(lambda: state_gone(turn))

    def test_the_audio_reaches_the_turn_as_it_renders(self, vibevoice):
        vibevoice.runtime.hold_after_first = True
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        assert vibevoice.runtime.first_frame.wait(3.0)
        # One frame is out and the render is still going: it is already the
        # turn's, not held back for the end of the render.
        assert turn.started.is_set()
        kind, payload = turn.read_audio(1.0)
        assert kind == "audio" and payload == b"\x01\x00" * 240
        vibevoice.runtime.release.set()
        speech.finish_turn(turn)
        assert turn.finished.wait(3.0) and turn.synthesis_done


class TestSentencesWrittenDuringARenderAreOneRender:
    def test_a_slow_render_gathers_what_was_written_meanwhile(self, vibevoice):
        vibevoice.runtime.hold_after_first = True
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        assert vibevoice.runtime.first_frame.wait(3.0)
        speech.send_segment(turn, "A second one arrives while that renders.")
        speech.send_segment(turn, "And a third one arrives too.")
        vibevoice.runtime.release.set()
        speech.finish_turn(turn)
        assert turn.finished.wait(3.0) and turn.synthesis_done
        assert vibevoice.runtime.texts() == [
            "The first sentence is long enough to be spoken.",
            "A second one arrives while that renders. And a third one arrives too."]

    def test_one_render_takes_at_most_six_hundred_characters(self, vibevoice):
        vibevoice.runtime.hold_after_first = True
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        assert vibevoice.runtime.first_frame.wait(3.0)
        long = [("Sentence number %d " % n) + "word " * 45 + "end." for n in range(3)]
        for text in long:
            speech.send_segment(turn, text)
        vibevoice.runtime.release.set()
        speech.finish_turn(turn)
        assert turn.finished.wait(3.0) and turn.synthesis_done
        texts = vibevoice.runtime.texts()
        assert len(texts) == 3
        assert all(len(text) <= speech.MAX_UNIT_CHARS for text in texts[1:])
        assert texts[1] == " ".join(" ".join(text.split()) for text in long[:2])
        assert texts[2] == " ".join(long[2].split())


class TestTheThreeWordHold:
    def test_a_unit_of_three_words_waits_for_more_text(self, vibevoice):
        """The Realtime model is unstable on very short input."""
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "Yes, of course.")
        time.sleep(0.4)
        assert vibevoice.runtime.renders == [], "three words were rendered on their own"
        speech.send_segment(turn, "Here is the whole answer to your question.")
        speech.finish_turn(turn)
        assert turn.finished.wait(3.0) and turn.synthesis_done
        assert vibevoice.runtime.texts() == [
            "Yes, of course. Here is the whole answer to your question."]

    def test_four_words_are_not_held(self, vibevoice):
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "That is quite right.")
        assert wait_for(lambda: vibevoice.runtime.renders), "four words were held back"
        speech.finish_turn(turn)
        assert turn.finished.wait(3.0)

    def test_a_short_unit_is_spoken_when_the_reply_ends(self, vibevoice):
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "Here is a sentence of several words.")
        assert wait_for(lambda: len(vibevoice.runtime.renders) == 1)
        speech.send_segment(turn, "Thanks!")
        time.sleep(0.3)
        assert len(vibevoice.runtime.renders) == 1
        speech.finish_turn(turn)
        assert turn.finished.wait(3.0) and turn.synthesis_done
        assert vibevoice.runtime.texts()[-1] == "Thanks!"


class TestStop:
    def test_a_cancel_mid_render_cancels_it_and_hands_the_card_back(self, vibevoice):
        vibevoice.runtime.hold_after_first = True
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        speech.send_segment(turn, "A queued sentence nobody will hear.")
        assert vibevoice.runtime.first_frame.wait(3.0)
        turn.cancel("user")
        speech.interrupt_turn(turn)
        assert wait_for(lambda: vibevoice.runtime.cancels)
        (_card, job) = vibevoice.runtime.renders[0]
        assert vibevoice.runtime.cancels == [(CARD, job.id)]
        assert wait_for(lambda: state_gone(turn))
        (card_turn,) = vibevoice.client.requests
        assert card_turn.finishes == [None]
        assert len(vibevoice.runtime.renders) == 1, "the queued sentence was still rendered"
        assert not turn.synthesis_done

    def test_interrupt_returns_at_once_and_is_idempotent(self, vibevoice):
        vibevoice.runtime.hold_after_first = True
        vibevoice.runtime.cancel_delay = 1.0
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        assert vibevoice.runtime.first_frame.wait(3.0)
        started = time.monotonic()
        speech.interrupt_turn(turn)
        speech.cancel_turn(turn)
        speech.interrupt_turn(turn)
        assert time.monotonic() - started < 0.3, "Stop waited for the model"
        assert wait_for(lambda: state_gone(turn))
        assert len(vibevoice.runtime.cancels) == 1

    def test_stop_ends_voice_chats_speech_and_never_evicts(self, vibevoice):
        vibevoice.runtime.hold_after_first = True
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        assert vibevoice.runtime.first_frame.wait(3.0)
        speech.stop("the text-to-speech engine changed")
        assert turn.cancelled.is_set()
        assert wait_for(lambda: vibevoice.runtime.cancels)
        assert wait_for(lambda: state_gone(turn))
        assert vibevoice.runtime.evicts == []
        assert vibevoice.client.unloads == []
        assert vibevoice.client.requests[0].finishes == [None]

    def test_a_failed_render_fails_the_audio_and_hands_the_card_back(self, vibevoice):
        vibevoice.runtime.fail_render = "the VibeVoice worker exited during the render"
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        assert turn.finished.wait(3.0)
        assert turn.error == "The VibeVoice worker exited during the render."
        assert wait_for(lambda: vibevoice.client.requests[0].finishes == [None])


class TestTheCardWait:
    def test_a_refused_card_fails_the_turn_with_its_warning(self, vibevoice):
        vibevoice.client.grant_at_once = False
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        vibevoice.client.requests[0].block("VibeVoice needs 20.0 GB of VRAM and the card "
                                           "has 12.0 GB free.")
        assert turn.finished.wait(3.0)
        assert "needs 20.0 GB of VRAM" in turn.error
        assert vibevoice.runtime.loads == []
        assert wait_for(lambda: state_gone(turn))
        card_turn = vibevoice.client.requests[0]
        assert card_turn.cancels >= 1 and card_turn.finishes == []

    def test_while_the_reply_is_being_written_the_wait_has_no_bound(self, vibevoice,
                                                                      monkeypatch):
        """The turn it waits behind can be the one writing the reply."""
        monkeypatch.setattr(speech, "SPEECH_CARD_WAIT", 0.3)
        vibevoice.client.grant_at_once = False
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        time.sleep(0.8)
        assert not turn.finished.is_set() and not turn.error, "the bound ran while writing"
        speech.send_segment(turn, "The reply went on for longer than the bound.")
        speech.finish_turn(turn)
        time.sleep(0.1)
        vibevoice.client.requests[0].grant()
        assert turn.finished.wait(3.0)
        assert turn.synthesis_done and not turn.error
        assert vibevoice.runtime.renders

    def test_a_grant_that_misses_the_bound_after_the_reply_fails_it(self, vibevoice,
                                                                     monkeypatch):
        monkeypatch.setattr(speech, "SPEECH_CARD_WAIT", 0.3)
        vibevoice.client.grant_at_once = False
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        time.sleep(0.4)
        ended = time.monotonic()
        speech.finish_turn(turn)
        assert turn.finished.wait(3.0)
        assert time.monotonic() - ended >= 0.25
        assert WAITING in turn.error and "was not read aloud" in turn.error
        assert wait_for(lambda: state_gone(turn))
        assert vibevoice.client.requests[0].cancels >= 1, "the request was not withdrawn"
        assert vibevoice.runtime.loads == []

    def test_stop_during_the_unbounded_wait_ends_it_at_once(self, vibevoice):
        vibevoice.client.grant_at_once = False
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        speech.send_segment(turn, "The first sentence is long enough to be spoken.")
        time.sleep(0.2)
        started = time.monotonic()
        turn.cancel("user")
        assert wait_for(lambda: state_gone(turn), timeout=1.0)
        assert time.monotonic() - started < 0.5
        assert vibevoice.client.requests[0].cancels >= 1
        assert turn.error == "" and vibevoice.runtime.loads == []

    def test_what_cannot_be_asked_for_is_refused_before_the_card(self, vibevoice):
        vibevoice.installer.installed[MODEL_REALTIME] = False
        turn = voice_turn()
        with pytest.raises(speech.VibeVoiceSpeechError):
            speech.begin_turn(turn, PRESET, None)
        assert turn.error == "The model is not installed — install it below."
        assert turn.rate_known.is_set(), "a refused turn still lets its listener go"
        assert vibevoice.client.requests == []


# --------------------------------------------------------------------------- #
# A whole reply, and Test
# --------------------------------------------------------------------------- #


class TestSynthesize:
    def test_a_whole_reply_is_one_card_turn_and_one_render(self, vibevoice):
        audio = speech.synthesize("Hello there, friend.", PRESET, None)
        assert audio == WAV
        (card_turn,) = vibevoice.client.requests
        assert (card_turn.card, card_turn.label) == (CARD, "VibeVoice — Voice Chat")
        assert card_turn.finishes == [None]
        (_card, job) = vibevoice.runtime.renders[0]
        assert job.script == [(1, "Hello there, friend.")]
        assert job.stream is False

    def test_a_card_not_granted_in_time_is_refused_with_its_reason(self, vibevoice,
                                                                   monkeypatch):
        monkeypatch.setattr(speech, "SYNTH_CARD_WAIT", 0.2)
        vibevoice.client.grant_at_once = False
        started = time.monotonic()
        with pytest.raises(speech.VibeVoiceSpeechError) as refused:
            speech.synthesize("Hello there, friend.", PRESET, None)
        assert time.monotonic() - started < 2.0, "a Test waited past its bound"
        assert WAITING in str(refused.value)
        card_turn = vibevoice.client.requests[0]
        assert card_turn.cancels >= 1 and card_turn.finishes == []
        assert vibevoice.runtime.loads == []

    def test_a_blocked_card_is_refused_with_its_warning(self, vibevoice):
        vibevoice.client.grant_at_once = False

        def block():
            assert wait_for(lambda: vibevoice.client.requests)
            vibevoice.client.requests[0].block("There is not enough system RAM.")

        threading.Thread(target=block, daemon=True).start()
        with pytest.raises(speech.VibeVoiceSpeechError) as refused:
            speech.synthesize("Hello there, friend.", PRESET, None)
        assert "not enough system RAM" in str(refused.value)

    def test_a_sample_speaks_at_the_chat_precision_and_lora(self, vibevoice):
        sample = box.add_sample(tone_wav(4.0), title="Ada")
        vibevoice.installer.chat.update({"precision": "int8", "lora_id": "warm01",
                                         "lora_scale": 0.8})
        speech.synthesize("Hello there, friend.", {"model": MODEL_7B, "sample": sample["id"]})
        assert vibevoice.runtime.loads[-1] == {"card": CARD, "model_id": MODEL_7B,
                                               "precision": "int8", "lora_id": "warm01",
                                               "lora_scale": 0.8}
        assert vibevoice.client.requests[0].need_vram == 12_000_000_000
        (_card, job) = vibevoice.runtime.renders[0]
        pcm16, rate = box.sample_pcm(sample["id"])
        assert job.voices == {1: (box.float32_of(pcm16), rate)}
        assert (job.cfg_scale, job.steps) == (1.3, 10)

    def test_the_realtime_model_is_always_full_precision_without_a_lora(self, vibevoice):
        vibevoice.installer.chat.update({"precision": "nf4", "lora_id": "warm01"})
        speech.synthesize("Hello there, friend.", PRESET)
        assert vibevoice.runtime.loads[-1]["precision"] == "bf16"
        assert vibevoice.runtime.loads[-1]["lora_id"] == ""

    def test_a_recording_is_speaker_one(self, vibevoice):
        """The clone audition: a recording in place of a sample, on the 7B."""
        pcm16 = b"\x10\x00\xf0\xff" * 12000
        speech.synthesize("Hello there, friend.", {"model": MODEL_7B, "reference": pcm16})
        (_card, job) = vibevoice.runtime.renders[0]
        assert job.voices == {1: (box.float32_of(pcm16), 24000)}
        assert vibevoice.runtime.loads[-1]["model_id"] == MODEL_7B

    def test_guidance_and_steps_follow_the_profile_else_the_model(self, vibevoice):
        speech.synthesize("Hello there, friend.", PRESET, {"cfg_scale": 2.0, "steps": None})
        speech.synthesize("Hello there, friend.", PRESET, None)
        speech.synthesize("Hello there, friend.", PRESET, {"cfg_scale": None, "steps": 12})
        jobs = [job for _card, job in vibevoice.runtime.renders]
        assert [(job.cfg_scale, job.steps) for job in jobs] == [(2.0, 5), (1.5, 5), (1.5, 12)]

    def test_the_card_is_voice_chats_else_the_voice_boxs_else_refused(self, vibevoice):
        vibevoice.installer.chat["card_uuid"] = ""
        box.set_settings({"card_uuid": BOX_CARD})
        speech.synthesize("Hello there, friend.", PRESET)
        assert vibevoice.client.requests[-1].card == BOX_CARD
        box.set_settings({"card_uuid": ""})
        with pytest.raises(speech.VibeVoiceSpeechError) as refused:
            speech.synthesize("Hello there, friend.", PRESET)
        assert "Choose the card VibeVoice speaks on" in str(refused.value)

    def test_without_a_turn_client_it_is_refused(self, vibevoice):
        box.use_turns(None)
        with pytest.raises(speech.VibeVoiceSpeechError) as refused:
            speech.synthesize("Hello there, friend.", PRESET)
        assert "not connected to the cards" in str(refused.value)

    def test_a_preset_that_is_not_a_name_is_refused(self, vibevoice):
        with pytest.raises(speech.VibeVoiceSpeechError):
            speech.synthesize("Hello there.", {"model": MODEL_REALTIME, "preset": "../x"})
        assert vibevoice.client.requests == []


# --------------------------------------------------------------------------- #
# The flyout and the lifecycle
# --------------------------------------------------------------------------- #


class TestTheFlyoutAndTheLifecycle:
    def test_prepare_asks_for_nothing(self, vibevoice, host):
        assert speech.prepare() is False
        vibevoice.runtime.identity[CARD] = {"model_id": MODEL_REALTIME, "precision": "bf16"}
        assert speech.prepare() is True, "the default voice's model is loaded"
        vibevoice.runtime.identity[CARD] = {"model_id": MODEL_7B, "precision": "bf16"}
        assert speech.prepare() is False
        assert vibevoice.client.requests == []

    def test_load_takes_a_turn_loads_the_default_voice_and_stays_warm(self, vibevoice, host):
        found = speech.load()
        assert vibevoice.runtime.loads[-1]["model_id"] == MODEL_REALTIME
        assert vibevoice.client.requests[0].finishes == [True]
        assert found["loaded"] is True and found["state"] == "idle"
        assert found["backend"] == "vibevoice"
        assert found["where"] == "NVIDIA GeForce RTX 3090"

    def test_unload_goes_through_the_turn_client(self, vibevoice):
        speech.unload("unloaded from the Voice panel")
        assert vibevoice.client.unloads == [(CARD, "unloaded from the Voice panel")]
        assert vibevoice.runtime.evicts == []

    def test_the_state_line_follows_a_reply(self, vibevoice):
        assert speech.engine()["state"] == "unloaded"
        vibevoice.client.grant_at_once = False
        turn = voice_turn()
        speech.begin_turn(turn, PRESET, None)
        assert speech.engine()["state"] == "loading"
        assert speech.status()["busy"] is True
        turn.cancel("user")
        assert wait_for(lambda: state_gone(turn))
        assert speech.status() == {"loaded": False, "busy": False, "draining": False,
                                   "interrupt_mode": "cancel", "error": ""}

    def test_stop_means_cancel(self):
        assert speech.declared_interrupt_mode() == "cancel"

    def test_the_phase_words_are_the_turn_systems(self):
        import mc_turns_guests

        assert speech.GRANTED == mc_turns_guests.PHASES["granted"]
        assert speech.BLOCKED == mc_turns_guests.PHASES["blocked"]
        assert speech.CANCELLED == mc_turns_guests.PHASES["cancelled"]
        assert speech.DONE == mc_turns_guests.PHASES["done"]
