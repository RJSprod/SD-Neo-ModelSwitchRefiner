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

import mc_voice_turn as turns


class Speaker:
    """A worker that answers instantly and records what it was asked."""

    def __init__(self, rate: int = 24000, blocks: int = 1, samples: int = 1200):
        self.rate = rate
        self.blocks = blocks
        self.samples = samples
        self.segments = []
        self.began = []
        self.cancelled = []
        self.finished = []

    def begin_turn(self, turn, sid, speed=1.0):
        self.began.append(int(sid))
        turn.sample_rate = self.rate
        return self.rate

    def send_segment(self, turn, text):
        self.segments.append(text)
        for _block in range(self.blocks):
            turn.offer_audio(b"\x01\x00" * self.samples, self.rate)

    def finish_turn(self, turn):
        self.finished.append(turn.id)
        turn.audio_finished()

    def cancel_turn(self, turn):
        self.cancelled.append(turn.id)


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
                              "first_audio_ms", "cancelled", "voice_type", "sid"}

    def test_a_clone_is_categorised_without_naming_itself(self):
        turn = turns.create(voice_id="clone:1234", sid=53, speaker=Speaker())
        assert turn.metrics()["voice_type"] == "clone"


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
