"""A reply, from the Conversation generator to the browser's audio stream.

The unit tests either side of this one are thorough and between them they
missed a whole class of bug: every part working while the *wiring* between them
was wrong. A handler whose declared outputs do not match what its function
returns is refused by Gradio at run time and by nothing at all in a test that
calls the function directly; a turn id that never reaches the hidden field the
browser polls is a feature that is silently switched off.

So these tests run the real generator, count what it yields, and take the token
it produces through the real HTTP route -- including in the order that actually
happens on a slow machine, where the browser opens the audio stream twenty
seconds before the model writes its first character.
"""

from __future__ import annotations

import threading
import time

import pytest

import mc_llm_chat_panel as chat
import mc_llm_paths
import mc_voice_models as models
import mc_voice_turn as turns


class Speaker:
    """A Voice Worker that answers instantly and records what it was asked."""

    def __init__(self):
        self.segments = []

    def begin_turn(self, turn, sid, speed=1.0):
        turn.sample_rate = 24000
        return 24000

    def send_segment(self, turn, text):
        self.segments.append(text)
        turn.offer_audio(b"\x01\x00" * 1200, 24000)

    def finish_turn(self, turn):
        turn.audio_finished()

    def cancel_turn(self, turn):
        pass


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture
def speaking(host, monkeypatch, kokoro_bundle, voice_registry):
    """Voice installed, Auto Speak on, and a worker that answers."""
    host.shared.opts.model_chain_voice_auto_speak = True
    ready = models.Status(runtime_ready=True, stt_ready=True, tts_ready=True,
                          runtime_message="Installed", stt_message="Installed",
                          tts_message="Installed", platform_supported=True,
                          stt_id="whisper-small-int8", tts_id="kokoro-multi-lang-v1-cpu")
    monkeypatch.setattr(models, "status", lambda: ready)
    speaker = Speaker()
    original = turns.create
    monkeypatch.setattr(turns, "create",
                        lambda **kwargs: original(**dict(kwargs, speaker=speaker)))
    return speaker


def thread(store, monkeypatch, pieces=("Hello there, this is a long first sentence. ",
                                       "And a second one that runs on for a while too. ")):
    from prompt_master.chat.characters import Character, Persona, save_persona
    from prompt_master.chat.history import ChatStore
    import mc_llm_sessions as sessions

    save_persona(mc_llm_paths.app_paths(), Persona(name="Me", description="a reader"))
    chats = ChatStore(store / "chats")
    monkeypatch.setattr(chat, "_chats", lambda: chats)

    class Characters:
        def load(self, who):
            return Character(name="Ada", context="a reader of maps")

    monkeypatch.setattr(chat, "_characters", lambda: Characters())
    events = [sessions.Event(sessions.CHUNK, piece) for piece in pieces]
    events.append(sessions.Event(sessions.DONE, "".join(pieces)))
    monkeypatch.setattr(chat.sessions, "conversation", lambda request, cancel: iter(events))
    conversation = chats.new("Ada")
    chats.save(conversation)
    return chats, conversation


def run_reply(store, monkeypatch, **kwargs):
    _chats, conversation = thread(store, monkeypatch, **kwargs)
    return list(chat._send("Ada", conversation.identifier, "hello love",
                           None, None, None, None, None))


class TestTheTurnReachesTheBrowser:
    def test_every_yield_carries_the_turn_and_the_run_state(self, store, monkeypatch,
                                                            speaking):
        """The two hidden values the browser polls. A yield that is missing them
        is a Gradio error, and a yield that carries an empty turn is Voice
        silently switched off."""
        frames = run_reply(store, monkeypatch)
        assert {len(frame) for frame in frames} == {10}, "the stream outputs changed shape"
        tokens = {frame[-2] for frame in frames}
        assert len(tokens) == 1 and tokens != {""}, tokens
        assert [frame[-1] for frame in frames][:1] == [chat.LLM_RUNNING]
        assert [frame[-1] for frame in frames][-1] == chat.LLM_IDLE

    def test_the_token_names_a_turn_the_stream_route_can_find(self, store, monkeypatch,
                                                             speaking):
        """A token the browser cannot exchange for audio is a 404 nothing logs
        and nobody hears."""
        frames = run_reply(store, monkeypatch)
        token = frames[0][-2]
        assert turns.lookup(token) is not None

    def test_the_reply_reaches_the_worker_as_segments(self, store, monkeypatch, speaking):
        frames = run_reply(store, monkeypatch)
        turn = turns.lookup(frames[0][-2])
        turn.attached.set()
        turn.finished.wait(3.0)
        assert speaking.segments, "the reply was never handed to the worker"
        assert "Hello there" in speaking.segments[0]

    def test_auto_speak_off_produces_no_turn_and_leaves_the_fallback(self, store,
                                                                    monkeypatch, speaking,
                                                                    host):
        host.shared.opts.model_chain_voice_auto_speak = False
        frames = run_reply(store, monkeypatch)
        assert {frame[-2] for frame in frames} == {""}
        assert not speaking.segments


class TestTheStopButton:
    def test_stop_answers_with_exactly_what_its_click_declares(self, host):
        """The bug this exists for: ``_cancel`` grew two return values when the
        composer gained its hidden run state, the click's outputs did not, and
        Gradio refuses a handler whose arity disagrees -- so Stop raised instead
        of stopping. Counting the function against the wiring is the only test
        that sees it, because calling either one alone is fine."""
        import gradio

        made = []
        original = gradio.Button

        class Recorded(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                made.append(self)

        gradio.Button = Recorded
        try:
            chat.build()
        finally:
            gradio.Button = original

        stop = [button for button in made
                if getattr(button, "elem_id", "") == "mc-llm-chat-stop"]
        assert len(stop) == 1, "the composer's Stop was not built exactly once"
        clicks = [kwargs for kind, kwargs in stop[0]._callbacks if kind == "click"]
        assert len(clicks) == 1
        assert len(clicks[0]["outputs"]) == len(chat._cancel(None)), (
            "Stop returns a different number of values than its click declares, so "
            "Gradio refuses the handler and pressing Stop raises instead of stopping")

    def test_stop_clears_the_turn_and_the_run_state(self):
        found = chat._cancel(None)
        assert found[-2] == ""
        assert found[-1] == chat.LLM_IDLE


class TestTheLogSaysWhyItIsQuiet:
    """V1 wrote ``Voice TTS finished`` for every spoken reply. V1.1 replaced it
    with a turn metric written only when audio actually flowed -- so a reply
    that was not spoken, for any reason at all, left no trace in the log at the
    default level. That is how a silent feature becomes an undiagnosable one,
    and these tests keep every branch audible."""

    def test_auto_speak_off_says_so(self, store, monkeypatch, speaking, host, caplog):
        """Also the throttle's own bug: a zero default reads as "said just now"
        on a machine whose monotonic clock is under ten minutes -- a WebUI
        started shortly after boot -- and throws away the first and most useful
        line."""
        host.shared.opts.model_chain_voice_auto_speak = False
        import mc_voice_ui

        mc_voice_ui._quiet.clear()
        with caplog.at_level("INFO", logger="model_chain"):
            run_reply(store, monkeypatch)
        said = [record.getMessage() for record in caplog.records]
        assert any("not reading replies aloud" in line for line in said), said

    def test_the_reason_is_said_once_and_then_throttled(self, store, monkeypatch,
                                                        speaking, host, caplog):
        """It is on the path of every assistant turn and the reason does not
        change between them."""
        host.shared.opts.model_chain_voice_auto_speak = False
        import mc_voice_ui

        mc_voice_ui._quiet.clear()
        with caplog.at_level("INFO", logger="model_chain"):
            run_reply(store, monkeypatch)
            run_reply(store, monkeypatch)
            run_reply(store, monkeypatch)
        said = [line for line in (record.getMessage() for record in caplog.records)
                if "not reading replies aloud" in line]
        assert len(said) == 1, said

    def test_a_reply_that_will_be_spoken_says_so_with_its_voice(self, store, monkeypatch,
                                                               speaking, caplog):
        with caplog.at_level("INFO", logger="model_chain"):
            run_reply(store, monkeypatch)
        said = [record.getMessage() for record in caplog.records]
        assert any("will read this reply aloud" in line for line in said), said
        assert not any("hello love" in line for line in said), "the log carried content"

    def test_a_voice_that_cannot_be_resolved_is_a_warning_not_a_shrug(self, store,
                                                                      monkeypatch,
                                                                      speaking, caplog):
        import mc_voice_registry

        def refuse(voice_id=""):
            raise RuntimeError("no bank")

        monkeypatch.setattr(mc_voice_registry, "resolve", refuse)
        with caplog.at_level("WARNING", logger="model_chain"):
            frames = run_reply(store, monkeypatch)
        assert any("could not start speaking" in record.message for record in caplog.records)
        assert {frame[-2] for frame in frames} == {""}, (
            "a turn that could not be created must leave the field empty so the "
            "completed-reply fallback is still allowed to fire")


# --------------------------------------------------------------------------- #
# The same reply, on the engine whose Stop is not a cancellation
# --------------------------------------------------------------------------- #


class DrainingSpeaker:
    """A worker whose Stop leaves it computing, like PocketTTS 3.0.2's.

    ``interrupt_turn`` rather than ``cancel_turn``, and it does what the real
    Pocket runtime does: it returns at once, marks the turn draining, and only
    releases it when a *later* call says the lane is free. Which is the whole
    point -- a double that freed the lane inside ``interrupt_turn`` would be a
    double that could not fail the test.
    """

    def __init__(self):
        self.segments = []
        self.interrupted = []
        self.cancelled = []
        self.turn = None

    def begin_turn(self, turn, handle=None, profile=None):
        self.turn = turn
        turn.sample_rate = 24000
        return 24000

    def send_segment(self, turn, text):
        self.segments.append(text)
        turn.offer_audio(b"\x01\x00" * 1200, 24000)

    def finish_turn(self, turn):
        turn.audio_finished()

    def cancel_turn(self, turn):
        self.cancelled.append(turn.id)

    def interrupt_turn(self, turn):
        self.interrupted.append(turn.id)
        turn.interrupting(chars=42, audio_ms=300)

    def release(self, turn):
        """What ``tts_interrupted state=complete`` does on the real runtime."""
        turn.interrupted()


class TestStopIsEngineSpecificAllTheWayThrough:
    """I-PKT-10 and section 49.2, at the layer both engines share."""

    def test_a_cancelling_engine_is_asked_to_cancel(self, host):
        speaker = Speaker()
        asked = []
        speaker.cancel_turn = lambda turn: asked.append(turn.id)
        turn = turns.create(voice_id="kokoro:official:af_heart", sid=3, speaker=speaker,
                            engine="kokoro", interrupt_mode="cancel")
        turn.attached.set()
        turn.start()
        turn.add_text("Hello there, this is a whole sentence. ")
        turn.complete()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not turn.finished.is_set():
            time.sleep(0.01)
        assert turn.interrupt_mode == "cancel"
        assert turn.draining is False

    def test_a_draining_engine_is_asked_to_interrupt_and_says_it_is_finishing(self, host):
        """The turn is silent at once and *busy* afterwards, which are two
        different facts and are reported as two."""
        speaker = DrainingSpeaker()
        turn = turns.create(voice_id="pocket:official:alba", speaker=speaker,
                            engine="pocket", handle="pocket:official:alba",
                            interrupt_mode="drain_unit")
        turn.attached.set()
        turn.start()
        turn.add_text("Hello there, this is a whole sentence. ")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not speaker.segments:
            time.sleep(0.01)
        assert speaker.segments

        turn.cancel("user")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not speaker.interrupted:
            time.sleep(0.01)
        assert speaker.interrupted == [turn.id]
        assert speaker.cancelled == [], "a draining engine was asked to cancel"
        assert turn.draining is True
        assert turn.finishing is True
        assert turns.finishing() is True
        # Silent already: the browser stops hearing this before any of the above.
        assert turn.cancelled.is_set()

        speaker.release(turn)
        assert turn.draining is False
        assert turns.finishing() is False

    def test_pcm_offered_during_a_drain_is_discarded_and_counted(self, host):
        """I-PKT-12. A muted turn that stopped *reading* would make the drain
        take longer rather than shorter, so the count is how a log can tell the
        two apart afterwards."""
        speaker = DrainingSpeaker()
        turn = turns.create(voice_id="pocket:official:alba", speaker=speaker,
                            engine="pocket", interrupt_mode="drain_unit")
        turn.cancel("user")
        for _block in range(4):
            assert turn.offer_audio(b"\x01\x00" * 1200, 24000) is False
        assert turn.metrics()["discarded_chunks"] == 4

    def test_the_metrics_say_what_the_stop_cost(self, host):
        """Section 36's allowed list, and section 43's tables. Numbers only, and
        absent rather than zero where nothing was measured."""
        speaker = DrainingSpeaker()
        turn = turns.create(voice_id="pocket:official:alba", speaker=speaker,
                            engine="pocket", interrupt_mode="drain_unit")
        turn.attached.set()
        turn.start()
        turn.add_text("Hello there, this is a whole sentence. ")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not speaker.segments:
            time.sleep(0.01)
        turn.cancel("user")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not speaker.interrupted:
            time.sleep(0.01)
        speaker.release(turn)
        found = turn.metrics()
        assert found["interrupt_mode"] == "drain_unit"
        assert found["interrupted"] is True
        assert found["interrupted_unit_chars"] == 42
        assert found["interrupted_unit_audio_ms"] == 300
        assert isinstance(found["stop_to_ready_ms"], int)
        assert "the launch code" not in repr(found)

    def test_a_turn_on_a_cancelling_engine_never_reports_a_drain(self, host):
        """Kokoro and Sopro show no waiting state, and the metrics say so too."""
        turn = turns.create(voice_id="kokoro:official:af_heart", sid=3,
                            speaker=Speaker(), engine="kokoro", interrupt_mode="cancel")
        turn.cancel("user")
        found = turn.metrics()
        assert found["interrupt_mode"] == "cancel"
        assert found["stop_to_ready_ms"] is None
        assert turn.finishing is False
