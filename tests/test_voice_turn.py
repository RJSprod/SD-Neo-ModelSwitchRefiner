"""One reply on its way to a speaker: identity, bounds, and stopping.

:mod:`mc_voice_turn` is where the three producers and two consumers of a spoken
reply meet, and almost everything that can go wrong with streaming speech goes
wrong here rather than in the segmenter or the worker:

    the LLM generator blocks because speech was slow;
    a cancelled reply keeps talking;
    a new reply is spoken over by the last one;
    a queue grows until the WebUI runs out of memory;
    a reply past the speech ceiling is quietly half-spoken.

Each of those is a test below. The speaker is a stub, because what is being
checked is the coordination rather than the synthesis -- ``tests/
test_voice_runtime.py`` drives a real subprocess for the other half.
"""

from __future__ import annotations

import threading
import time

import pytest

import mc_voice_turn as turns


class Speaker:
    """A worker that answers instantly and records what it was asked."""

    def __init__(self, rate: int = 24000, blocks: int = 1, samples: int = 1200,
                 warm: bool = True, prepare_error=None, synth_ms=None):
        self.synth_ms = synth_ms
        self.rate = rate
        self.blocks = blocks
        self.samples = samples
        self.segments = []
        self.began = []
        self.cancelled = []
        self.finished = []
        self.prepared = 0
        self.warm = warm
        self.prepare_error = prepare_error
        # Order matters more than counts for the warmup: "the worker was asked
        # to be ready before any text was handed over" is the whole claim.
        self.calls = []

    def prepare(self):
        self.prepared += 1
        self.calls.append("prepare")
        if self.prepare_error is not None:
            raise self.prepare_error
        return self.warm

    def begin_turn(self, turn, sid, speed=1.0):
        self.began.append(int(sid))
        self.calls.append("begin")
        turn.sample_rate = self.rate
        return self.rate

    def send_segment(self, turn, text):
        self.segments.append(text)
        self.calls.append("segment")
        for _block in range(self.blocks):
            turn.offer_audio(b"\x01\x00" * self.samples, self.rate)
        if self.synth_ms is not None:
            # What the worker answers with once the unit is done: how long it
            # took, how many batches sherpa handed back, and how much speech
            # came out. The real one arrives on the runtime's reader thread.
            index = len(self.segments)
            turn.note_segment(blocks=2, first_block_ms=11 * index,
                              synth_ms=self.synth_ms * index,
                              audio_ms=500 * index, streaming="callback")

    def finish_turn(self, turn):
        self.finished.append(turn.id)
        turn.audio_finished()

    def cancel_turn(self, turn):
        self.cancelled.append(turn.id)


class OldSpeaker(Speaker):
    """A speaker from before ``prepare`` existed.

    The turn asks for the method rather than requiring it, so a stub, a test
    double or a future alternative speaker that has never heard of warmup still
    speaks. Worth a class of its own because "optional" is easy to write and
    easy to stop being true.
    """

    prepare = None


def drain(turn, seconds: float = 3.0) -> int:
    """Read the whole stream the way the HTTP route does. Returns bytes."""
    total = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        kind, block = turn.read_audio(0.05)
        if kind == "end":
            break
        total += len(block)
    return total


def spoken(turn, chunks, final=None, wait: float = 1.0):
    for chunk in chunks:
        turn.add_text(chunk)
    turn.complete(final)
    turn.finished.wait(wait)


class TestOneReply:
    def test_text_becomes_segments_and_segments_become_audio(self):
        speaker = Speaker()
        turn = turns.create(voice_id="official:af_heart", sid=3, speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, ["Hello there, this is a first sentence for the test. ",
                      "And here is a second one that is also long enough to commit."])
        assert speaker.began == [3], "the resolved speaker id did not reach the worker"
        assert len(speaker.segments) >= 2
        assert drain(turn) > 0

    def test_the_label_never_reaches_the_worker(self):
        speaker = Speaker()
        turn = turns.create(sid=0, labels=("Alice",), speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, ["Alice: Hello there, this is what the character actually said."])
        assert speaker.segments
        assert not speaker.segments[0].lower().startswith("alice:")

    def test_adding_text_never_blocks_the_generator(self):
        """Section 6. This runs inside the Conversation generator, between one
        visual update and the next."""
        speaker = Speaker(blocks=8, samples=24000)
        turn = turns.create(sid=0, speaker=speaker)
        turn.start()
        started = time.monotonic()
        for _index in range(200):
            turn.add_text("Some more text arriving from the model right about now. ")
        assert time.monotonic() - started < 2.0, "add_text blocked the generator"
        turn.cancel("test")

    def test_the_sample_rate_comes_from_the_worker(self):
        speaker = Speaker(rate=22050)
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, ["Hello there, this is a first sentence for the rate test."])
        assert turn.sample_rate == 22050


class TestStopping:
    def test_cancel_is_idempotent_and_reports_which_call_did_it(self):
        turn = turns.create(sid=0, speaker=Speaker())
        assert turn.cancel("user") is True
        assert turn.cancel("user") is False

    def test_a_cancelled_turn_produces_no_further_audio(self):
        speaker = Speaker()
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        turn.add_text("A first sentence that is long enough to be committed here. ")
        time.sleep(0.2)
        turn.cancel("user")
        turn.add_text("A second sentence that must never be spoken at all, ever. ")
        turn.complete()
        time.sleep(0.2)
        assert len(speaker.segments) <= 1, speaker.segments

    def test_cancelling_wakes_a_reader_that_is_waiting(self):
        turn = turns.create(sid=0, speaker=Speaker())
        turn.start()

        def stop():
            time.sleep(0.1)
            turn.cancel("user")

        threading.Thread(target=stop, daemon=True).start()
        started = time.monotonic()
        drain(turn, seconds=3.0)
        assert time.monotonic() - started < 2.0

    def test_a_new_turn_cancels_the_one_before_it(self):
        """Section 24's race, from the side that starts it: a reply that is still
        producing audio when its successor begins is how an old answer speaks
        over a new one."""
        first = turns.create(sid=0, speaker=Speaker())
        first.attached.set()
        first.start()
        first.add_text("The first reply, with a sentence long enough to commit. ")
        time.sleep(0.15)
        second = turns.create(sid=0, speaker=Speaker())
        assert first.cancelled.is_set()
        assert turns.active() is second

    def test_cancel_active_only_touches_a_busy_turn(self):
        turn = turns.create(sid=0, speaker=Speaker())
        turn.attached.set()
        turn.start()
        turn.complete("")
        turn.finished.wait(1.0)
        assert turns.cancel_active("user") is False

    def test_the_worker_is_told_when_a_turn_is_cancelled_mid_flight(self):
        speaker = Speaker()
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        turn.add_text("A sentence long enough to have started synthesising by now. ")
        time.sleep(0.2)
        turn.cancel("user")
        time.sleep(0.3)
        assert speaker.cancelled == [turn.id]


class TestBounds:
    def test_a_reply_past_the_ceiling_is_refused_rather_than_half_spoken(self):
        """Section 11. Speaking the first part of an answer and stopping without
        a word is worse than not speaking it."""
        turn = turns.create(sid=0, speaker=Speaker())
        turn._max_source = 200
        turn.start()
        turn.add_text("x " * 300)
        assert turn.cancelled.is_set()
        assert turn.reason == "limit"
        assert "longer than" in turn.error

    def test_the_audio_queue_applies_backpressure_rather_than_growing(self):
        """Section 23: never drop middle speech, never grow without bound. The
        producer waits, which is what reaches the worker through the pipe."""
        turn = turns.create(sid=0, speaker=Speaker())
        turn.sample_rate = 24000
        accepted = []

        def produce():
            for _index in range(200):
                accepted.append(turn.offer_audio(b"\x00\x00" * 24000, 24000, 2.0))

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        time.sleep(0.4)
        assert len(accepted) < 10, "the queue accepted everything it was offered"
        turn.cancel("user")
        worker.join(timeout=2.0)
        assert not worker.is_alive(), "a cancel did not free the blocked producer"

    def test_draining_frees_a_producer_nobody_is_listening_to(self):
        """Section 16: a full queue must never make cancellation wait behind its
        own producer."""
        turn = turns.create(sid=0, speaker=Speaker())
        turn.sample_rate = 24000
        turn.offer_audio(b"\x00\x00" * 24000, 24000, 0.1)
        turn.drain_audio()
        assert turn.read_audio(0.05)[0] in ("wait", "end")


class TestIdentity:
    def test_the_turn_id_is_opaque_and_carries_no_text(self):
        turn = turns.create(voice_id="clone:abc", sid=53, speaker=Speaker())
        turn.add_text("Something secret that nobody should be able to read from an id.")
        assert "secret" not in turn.id
        assert len(turn.id) >= 16

    def test_a_turn_can_be_found_by_its_id_and_a_stale_one_cannot(self):
        turn = turns.create(sid=0, speaker=Speaker())
        assert turns.lookup(turn.id) is turn
        assert turns.lookup("not-a-real-turn") is None
        assert turns.cancel("not-a-real-turn") is False

    def test_forget_all_cancels_everything(self):
        first = turns.create(sid=0, speaker=Speaker())
        first.start()
        turns.forget_all("test")
        assert first.cancelled.is_set()
        assert turns.active() is None


class TestMetrics:
    def test_metrics_carry_numbers_and_never_content(self):
        """Section 36's list, and the reason it is a list: the fields are named
        rather than filtered, so a field added later is absent from a log until
        somebody puts it there deliberately."""
        speaker = Speaker()
        turn = turns.create(voice_id="official:af_heart", sid=3, speaker=speaker)
        turn.attached.set()
        turn.start()
        secret = "the launch code is four zero four"
        spoken(turn, [f"Hello there, {secret}, and that is the end of the sentence. "])
        drain(turn)
        found = turn.metrics()
        assert secret not in repr(found)
        assert found["source_chars"] > 0
        assert found["segments"] >= 1
        assert found["voice_type"] == "official"
        assert set(found) == {"source_chars", "base_chars", "segments", "chunks",
                              "audio_seconds", "compute_seconds", "rtf", "first_segment_ms",
                              "first_audio_ms", "cancelled", "voice_type", "sid", "backend",
                              "worker_warm_at_turn_start", "runtime_prepare_ms",
                              "streaming", "callback_blocks",
                              "max_segment_ms", "max_segment_index",
                              "segment_1_chars", "segment_1_ms", "segment_1_first_block_ms",
                              "segment_1_callback_blocks", "segment_1_audio_ms",
                              "ready_wait_1_ms",
                              "segment_2_chars", "segment_2_ms", "segment_2_first_block_ms",
                              "segment_2_callback_blocks", "segment_2_audio_ms",
                              "ready_wait_2_ms",
                              # What Stop cost, on an engine where Stop is not
                              # free. Numbers only, and here for the same reason
                              # every other field is: named deliberately, so a
                              # drain measurement cannot arrive in a log by
                              # accident (I-PKT-11, section 36).
                              "interrupt_mode", "interrupted", "stop_to_silence_ms",
                              "stop_to_ready_ms", "interrupted_unit_chars",
                              "interrupted_unit_audio_ms", "discarded_chunks"}

    def test_a_clone_is_categorised_without_naming_itself(self):
        turn = turns.create(voice_id="clone:1234", sid=53, speaker=Speaker())
        assert turn.metrics()["voice_type"] == "clone"


class TestWarmingTheWorkerEarly:
    """The cold-start overlap, and the four ways it must not misbehave.

    On a cold run the worker has four hundred megabytes of ONNX to read, and it
    used to read them after the model had finished writing its first sentence
    rather than while it was writing it. Both are the machine waiting; only one
    of them is the user waiting.
    """

    def test_the_worker_is_asked_to_be_ready_before_any_text_arrives(self):
        speaker = Speaker()
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        # Nothing has been fed yet. The pump is already warming.
        deadline = time.monotonic() + 2.0
        while not speaker.prepared and time.monotonic() < deadline:
            time.sleep(0.01)
        assert speaker.prepared == 1
        assert speaker.calls[0] == "prepare"
        spoken(turn, ["Hello there, and this is a whole sentence to say. "])
        drain(turn)
        assert speaker.calls.index("prepare") < speaker.calls.index("segment")

    def test_it_happens_once_and_the_turn_still_opens_once(self):
        speaker = Speaker()
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, ["One sentence, and then a second one to be sure. ",
                      "And a third that is long enough to be its own segment as well. "])
        drain(turn)
        assert speaker.prepared == 1
        assert len(speaker.began) == 1

    def test_add_text_does_not_wait_for_a_slow_cold_start(self):
        """The invariant the whole overlap rests on: warming happens on the
        turn's own thread, and the generator's thread never touches it."""
        class Slow(Speaker):
            def prepare(self):
                time.sleep(0.4)
                return super().prepare()

        speaker = Slow()
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        started = time.monotonic()
        for _ in range(50):
            turn.add_text("Some more of the reply arrives here. ")
        assert time.monotonic() - started < 0.2, "add_text waited for the worker"
        turn.cancel("test")

    def test_a_warm_worker_is_recorded_as_warm(self):
        speaker = Speaker(warm=True)
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, ["A whole sentence, said once and recorded once. "])
        drain(turn)
        assert turn.metrics()["worker_warm_at_turn_start"] is True

    def test_a_cold_worker_is_recorded_as_cold(self):
        """Section 6: a first-audio time that includes a model load is not the
        same measurement as one that does not."""
        speaker = Speaker(warm=False)
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, ["A whole sentence, said once and recorded once. "])
        drain(turn)
        found = turn.metrics()
        assert found["worker_warm_at_turn_start"] is False
        assert found["runtime_prepare_ms"] is not None

    def test_a_failed_warmup_leaves_the_turn_to_report_it_properly(self):
        """A cold-start failure is a Voice error, and it is `begin_turn`'s to
        raise: cancelling the turn here would report a worker problem as though
        the reply itself had gone wrong."""
        import mc_voice_runtime

        class Broken(Speaker):
            def prepare(self):
                super().prepare()

            def begin_turn(self, turn, sid, speed=1.0):
                raise mc_voice_runtime.VoiceRuntimeError("Voice Chat is not set up.")

        speaker = Broken(prepare_error=RuntimeError("no runtime"))
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, ["A whole sentence that will not be spoken today. "])
        turn.finished.wait(2.0)
        assert speaker.prepared == 1
        assert turn.error == "Voice Chat is not set up."
        assert turn.reason == "error"

    def test_a_turn_with_no_text_never_synthesizes(self):
        speaker = Speaker()
        turn = turns.create(sid=0, speaker=speaker)
        turn._pump = None
        turn.start()
        time.sleep(0.1)
        turn.cancel("test")
        turn.finished.wait(2.0)
        assert speaker.began == []
        assert speaker.segments == []

    def test_cancelling_before_the_first_segment_ends_cleanly(self):
        speaker = Speaker()
        turn = turns.create(sid=0, speaker=speaker)
        turn.start()
        turn.cancel("user")
        assert turn.finished.wait(2.0)
        assert speaker.began == []

    def test_a_speaker_without_warmup_still_speaks(self):
        speaker = OldSpeaker()
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, ["A whole sentence, spoken by a speaker that cannot warm up. "])
        drain(turn)
        assert speaker.segments
        assert turn.metrics()["worker_warm_at_turn_start"] is None


class TestWhatEachSynthesisUnitCost:
    """One turn-level total cannot tell "the model was slow to write it" from
    "Kokoro was slow to say it", and those want completely different fixes.

    So each unit carries four durations: how long the lane waited for the text,
    how long the synthesis took, when the first block came back, and how much
    speech came out. The worker was already measuring the middle two and the
    parent was throwing them away.
    """

    def spoken_turn(self, chunks, synth_ms=120):
        speaker = Speaker(synth_ms=synth_ms)
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, chunks)
        drain(turn)
        return turn, speaker

    def test_synthesis_time_crosses_from_the_worker_into_the_metrics(self):
        """T-5. The worker's ``segment_ms`` used to stop at the runtime."""
        turn, _speaker = self.spoken_turn(
            ["Yes, that's possible. And here is a second sentence, long enough to go. "])
        found = turn.metrics()
        assert found["segment_1_ms"] == 120
        assert found["segment_2_ms"] == 240

    def test_the_first_two_units_expose_their_whole_shape(self):
        """T-6, and the reason all four numbers are wanted rather than one."""
        turn, _speaker = self.spoken_turn(
            ["Yes, that's possible. And here is a second sentence, long enough to go. "])
        found = turn.metrics()
        for index in (1, 2):
            assert found[f"segment_{index}_chars"] > 0
            assert found[f"ready_wait_{index}_ms"] is not None
            assert found[f"segment_{index}_ms"] > 0
            assert found[f"segment_{index}_first_block_ms"] > 0
            assert found[f"segment_{index}_callback_blocks"] == 2
            assert found[f"segment_{index}_audio_ms"] > 0

    def test_the_slowest_unit_is_named(self):
        turn, _speaker = self.spoken_turn(
            ["Yes, that's possible. And here is a second sentence, long enough to go. ",
             "A third sentence follows it, and it is long enough to be its own unit. "])
        found = turn.metrics()
        assert found["max_segment_index"] == found["segments"]
        assert found["max_segment_ms"] == 120 * found["segments"]

    def test_a_unit_the_worker_never_answered_for_still_reports_its_text(self):
        """A turn cancelled halfway through its second unit still knows how many
        characters that unit was and how long it waited for them. Reporting
        those as absent because the answer never came back would hide the case
        the numbers are most wanted for."""
        speaker = Speaker(synth_ms=None)
        turn = turns.create(sid=0, speaker=speaker)
        turn.attached.set()
        turn.start()
        spoken(turn, ["Yes, that's possible. "])
        found = turn.metrics()
        assert found["segment_1_chars"] > 0
        assert found["ready_wait_1_ms"] is not None
        assert found["segment_1_ms"] is None, "a synthesis that never answered was invented"

    def test_no_unit_record_carries_a_word_of_the_reply(self):
        secret = "the launch code is four zero four"
        turn, _speaker = self.spoken_turn(
            [f"Hello there, {secret}, and that is the end of the sentence. "])
        assert secret not in repr(turn.metrics())
        assert secret not in repr(turn._units)

    def test_the_unit_log_line_is_numbers_only(self, caplog):
        import logging

        with caplog.at_level(logging.DEBUG, logger="model_chain"):
            self.spoken_turn(["Yes, that's possible. And a second sentence, long enough. "])
        lines = [record.getMessage() for record in caplog.records
                 if "Voice TTS segment" in record.getMessage()]
        assert len(lines) >= 2, lines
        for line in lines:
            assert "possible" not in line
            assert "sentence" not in line
            for wanted in ("chars=", "ready_wait_ms=", "synth_ms=", "first_block_ms=",
                           "callback_blocks=", "audio_ms=", "streaming="):
                assert wanted in line, (wanted, line)


class TestTheClientWatchdog:
    def test_a_turn_nobody_listens_to_gives_up(self):
        """Otherwise it would hold the worker's one inference lane against a
        bounded queue for as long as the WebUI ran."""
        speaker = Speaker()
        turn = turns.create(sid=0, speaker=speaker)
        original = turns.CLIENT_WAIT
        turns.CLIENT_WAIT = 0.2
        try:
            turn.start()
            turn.add_text("A sentence long enough to start synthesising something. ")
            deadline = time.monotonic() + 3.0
            while turn.busy and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            turns.CLIENT_WAIT = original
        assert turn.cancelled.is_set()
        assert turn.reason == "no client"

    def test_a_turn_with_a_listener_is_left_alone(self):
        speaker = Speaker()
        turn = turns.create(sid=0, speaker=speaker)
        original = turns.CLIENT_WAIT
        turns.CLIENT_WAIT = 0.2
        try:
            turn.attached.set()
            turn.start()
            spoken(turn, ["A sentence long enough to start synthesising something. "])
            time.sleep(0.4)
        finally:
            turns.CLIENT_WAIT = original
        assert turn.reason != "no client"


class TestSayingSpeechIsSlowerThanRealTime:
    """An RTF above 1 is the whole of "the audio is choppy on long replies".

    The two summary lines already carry the number, but only as a ratio among a
    dozen others, with nothing to say which side of 1 is the bad side or what
    moves it. A user reading a log to find out why their speech stutters should
    not have to work out that 1.16 means the machine is falling further behind
    every second it speaks.

    Speed is named because on this engine Speed usually *is* the answer. Sopro
    V2 has no model-native speaking rate, so Speed is a time-compression
    applied after the model has produced full-length audio: the work does not
    change and the result is shorter, which multiplies the real-time factor by
    exactly the speed. 1.35x turns an engine running comfortably at 0.86 into
    one running at 1.16.
    """

    @pytest.fixture
    def captured(self, caplog):
        """Everything the extension's logger emitted, at every level."""
        import logging

        caplog.set_level(logging.DEBUG, logger="model_chain")
        return caplog

    @pytest.fixture
    def _api(self):
        import mc_voice_api

        return mc_voice_api

    class FakeTurn:
        def __init__(self, speed=None):
            self.profile = {"speed": speed} if speed is not None else None

    @staticmethod
    def _lines(captured):
        return [record.getMessage() for record in captured.records
                if "slower than it plays" in record.getMessage()]

    def test_a_turn_that_keeps_up_says_nothing(self, captured, _api):
        _api._log_shortfall(self.FakeTurn(1.0), {"rtf": 0.86, "audio_seconds": 38.0})
        assert not self._lines(captured), "a machine that kept up was told off"

    def test_a_turn_exactly_at_real_time_says_nothing(self, captured, _api):
        """1.0 is keeping up. The boundary is worth pinning because the whole
        line hangs off which way it goes."""
        _api._log_shortfall(self.FakeTurn(1.0), {"rtf": 1.0, "audio_seconds": 38.0})
        assert not self._lines(captured)

    def test_the_shortfall_is_reported_as_seconds_of_silence(self, captured, _api):
        """A ratio is not a quantity anybody can act on. Seconds owed is."""
        _api._log_shortfall(self.FakeTurn(1.0), {"rtf": 1.161, "audio_seconds": 38.0})
        line = self._lines(captured)[0]
        # (1.161 - 1) * 38.0, to one decimal.
        assert "6.1 s of silence" in line, line

    def test_speed_above_one_is_named_with_the_arithmetic(self, captured, _api):
        """Both halves of it: what the model actually produced, and what the
        same turn would have measured without the compression."""
        _api._log_shortfall(self.FakeTurn(1.35), {"rtf": 1.161, "audio_seconds": 38.0})
        line = self._lines(captured)[0]
        assert "1.35x" in line, line
        assert "51.3 s of audio to yield 38.0 s" in line, line
        assert "RTF 0.86" in line, line
        assert "Lower Speed" in line, line

    def test_at_neutral_speed_the_machine_is_named_instead(self, captured, _api):
        """Blaming a slider the user has not touched would send them to fix
        the wrong thing."""
        _api._log_shortfall(self.FakeTurn(1.0), {"rtf": 1.161, "audio_seconds": 38.0})
        line = self._lines(captured)[0]
        assert "already 1.00x" in line, line
        assert "not synthesising in real time" in line, line

    def test_a_turn_with_no_profile_at_all_is_still_reported(self, captured, _api):
        """A turn from an engine that carries no delivery, or one that failed
        before the profile was resolved. The shortfall is still true."""
        _api._log_shortfall(self.FakeTurn(), {"rtf": 1.2, "audio_seconds": 10.0})
        assert self._lines(captured)

    def test_a_turn_with_no_audio_is_not_divided_by(self, captured, _api):
        _api._log_shortfall(self.FakeTurn(1.35), {"rtf": None, "audio_seconds": 0.0})
        assert not self._lines(captured)

    def test_the_line_carries_no_text_from_the_reply(self, captured, _api):
        """Section 36 applies here as much as to the two lines above it: this
        one is written on the hot path and reads from the turn."""
        turn = self.FakeTurn(1.35)
        turn.profile = dict(turn.profile, note="the user said pineapple")
        _api._log_shortfall(turn, {"rtf": 1.2, "audio_seconds": 10.0})
        assert "pineapple" not in " ".join(self._lines(captured))
