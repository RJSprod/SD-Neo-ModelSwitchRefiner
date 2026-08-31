"""The PocketTTS sidecar: its framing, its DSP, and the drain that is its Stop.

Three things are asserted here that nothing else can assert, and all three are
release gates rather than niceties.

The first is that Pocket Speed is pitch-preserving (section 15). The reviewed
PocketTTS API has no speaking-rate parameter, so Speed is Voice Chat's own
time-scaling around the model output -- and a naive resample that transposes the
voice is not parity and does not ship. The only way to know which of those was
built is to synthesize a tone, run it through the shaper at several speeds, and
measure the fundamental.

The second is that the three workers agree on the wire (section 19). They are
separate files in separate dependency closures on purpose, so their framing is
shared by *agreement* rather than by import -- and an agreement nobody checks is
a protocol that drifts.

The third is the one this engine exists to get right. ``tts_interrupt`` on
released PocketTTS 3.0.2 means: stop offering audio now, drop the units that
have not started, keep consuming the one already inside the model, and say when
the lane is free. Every clause of that is tested below against the *real*
``serve`` loop with a fake engine standing in for Torch -- because a drain that
was only reasoned about is a drain that deadlocks in production (I-PKT-11,
I-PKT-12, I-PKT-13, GATE P-3).

Nothing here imports Torch or PocketTTS. The DSP is NumPy and the framing is
the standard library, which is exactly the part of the worker that can be
tested on a machine where PocketTTS was never installed.
"""

from __future__ import annotations

import io
import math
import struct
import threading
import time

import pytest

from pocket_worker import worker as pocket_worker
from sopro_worker import worker as sopro_worker
from voice_worker import worker as kokoro_worker

numpy = pytest.importorskip("numpy", reason="the Pocket delivery DSP is NumPy")

RATE = 24000


def tone(freq: float, seconds: float, rate: int = RATE):
    steps = numpy.arange(int(seconds * rate), dtype=numpy.float32) / rate
    return (0.5 * numpy.sin(2.0 * numpy.pi * freq * steps)).astype(numpy.float32)


def dominant(samples, rate: int = RATE) -> float:
    """The strongest frequency in a block, by FFT. Windowed, because a
    rectangular window on a tone smears the peak across neighbouring bins."""
    if len(samples) < 4096:
        return 0.0
    window = numpy.hanning(len(samples)).astype(numpy.float32)
    spectrum = numpy.abs(numpy.fft.rfft(samples * window))
    return float(numpy.fft.rfftfreq(len(samples), 1.0 / rate)[int(numpy.argmax(spectrum))])


def through(shaper, source, chunk: int = 431):
    """Push a signal through a shaper in awkward-sized chunks and collect the PCM.

    Awkward on purpose. PocketTTS's decoded chunks are not a multiple of the
    shaper's frame, and a DSP that only worked on tidy boundaries would be a DSP
    that clicked in production and passed its tests.
    """
    out = bytearray()
    for start in range(0, len(source), chunk):
        out += shaper.block(source[start:start + chunk])
    out += shaper.flush()
    return numpy.frombuffer(bytes(out), dtype="<i2").astype(numpy.float32) / 32768.0


def delivery(**values):
    return pocket_worker.Delivery.from_header(values)


# --------------------------------------------------------------------------- #
# The wire
# --------------------------------------------------------------------------- #


class TestTheThreeWorkersAgreeOnTheWire:
    """T-PKT-W-1. Three files, three closures, one framing.

    Shared by agreement rather than by import, because a single module would
    have to be importable from inside three isolated interpreters -- which is
    exactly the coupling the separate runtimes exist to prevent. So the
    agreement is checked here instead.
    """

    def test_a_frame_written_by_one_is_read_by_the_others(self):
        header = {"op": "tts_audio", "turn": "abc", "sample_rate": RATE}
        payload = b"\x01\x02\x03\x04"
        for writer in (pocket_worker, sopro_worker, kokoro_worker):
            buffer = io.BytesIO()
            writer.write_frame(buffer, header, payload)
            raw = buffer.getvalue()
            for reader in (pocket_worker, sopro_worker, kokoro_worker):
                found = reader.read_frame(io.BytesIO(raw))
                assert found == (header, payload)

    def test_the_bytes_are_identical_and_not_merely_compatible(self):
        header = {"op": "init", "id": 7}
        first, second = io.BytesIO(), io.BytesIO()
        pocket_worker.write_frame(first, header, b"body")
        sopro_worker.write_frame(second, header, b"body")
        assert first.getvalue() == second.getvalue()

    def test_the_protocol_version_is_pockets_own(self):
        """T-PKT-W-2. Counted from one, and not shared with either other worker.

        They overlap in framing and in several operation names and they are
        still three protocols: Pocket carries ``voice_id`` where Kokoro carries
        ``sid``, and ``tts_interrupt`` means something neither of the others can
        mean. One number covering all three would have to change when any of
        them changed.
        """
        assert pocket_worker.PROTOCOL_VERSION == 1
        assert pocket_worker.MARKER == "--model-chain-pocket-worker"
        assert pocket_worker.MARKER != sopro_worker.MARKER

    def test_an_oversized_header_is_refused_rather_than_allocated(self):
        raw = pocket_worker._LENGTH.pack(pocket_worker.MAX_HEADER + 1)
        with pytest.raises(ValueError):
            pocket_worker.read_frame(io.BytesIO(raw))


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


class TestSpeedIsPitchPreserving:
    """T-PKT-W-16, and the reason section 15 exists.

    A resample changes duration and pitch together. Everything below measures
    the two separately, because "it sounds faster" is true of both the correct
    implementation and the wrong one.
    """

    def test_speed_one_is_a_bypass(self):
        shaper = pocket_worker.Shaper(delivery(), numpy)
        assert shaper.active is False

    @pytest.mark.parametrize("speed", (0.8, 1.0, 1.25, 1.5))
    def test_the_fundamental_does_not_move_with_speed(self, speed):
        source = tone(220.0, 1.5)
        found = through(pocket_worker.Shaper(delivery(speed=speed), numpy), source)
        assert abs(dominant(found) - 220.0) < 4.0

    @pytest.mark.parametrize("speed", (0.8, 1.25, 1.5))
    def test_the_duration_moves_with_speed(self, speed):
        source = tone(220.0, 1.5)
        found = through(pocket_worker.Shaper(delivery(speed=speed), numpy), source)
        assert abs(len(found) / (len(source) / speed) - 1.0) < 0.08

    @pytest.mark.parametrize("semitones", (-5, 4))
    def test_pitch_moves_the_fundamental_by_the_semitones_asked_for(self, semitones):
        ratio = 2.0 ** (semitones / 12.0)
        source = tone(220.0, 1.5)
        found = through(pocket_worker.Shaper(delivery(pitch=ratio), numpy), source)
        assert abs(dominant(found) - 220.0 * ratio) < 6.0

    @pytest.mark.parametrize("semitones", (-5, 4))
    def test_pitch_does_not_materially_change_the_duration(self, semitones):
        ratio = 2.0 ** (semitones / 12.0)
        source = tone(220.0, 1.5)
        found = through(pocket_worker.Shaper(delivery(pitch=ratio), numpy), source)
        assert abs(len(found) / len(source) - 1.0) < 0.08

    def test_speed_and_pitch_compose(self):
        ratio = 2.0 ** (3 / 12.0)
        source = tone(220.0, 1.5)
        found = through(pocket_worker.Shaper(delivery(speed=1.25, pitch=ratio), numpy),
                        source)
        assert abs(dominant(found) - 220.0 * ratio) < 6.0
        assert abs(len(found) / (len(source) / 1.25) - 1.0) < 0.08

    def test_a_chunk_boundary_contains_no_click(self):
        """The state that survives between chunks, measured rather than asserted.

        A stretcher restarted at each chunk puts a step discontinuity at every
        boundary. It is inaudible in a spectrum and obvious in the first
        difference, so that is what is looked at.
        """
        source = tone(220.0, 1.0)
        found = through(pocket_worker.Shaper(delivery(speed=1.3), numpy), source, chunk=337)
        steps = numpy.abs(numpy.diff(found))
        assert float(numpy.max(steps)) < 0.2


class TestVolumeAndPause:
    def test_gain_below_unity_is_exact_scaling(self):
        source = numpy.full(2048, 0.5, dtype=numpy.float32)
        found = numpy.frombuffer(pocket_worker.pcm16(source, 0.5), dtype="<i2")
        assert abs(int(found[0]) - int(0.25 * 32767)) <= 1

    def test_gain_above_unity_is_limited_rather_than_wrapped(self):
        source = numpy.full(2048, 0.95, dtype=numpy.float32)
        found = numpy.frombuffer(pocket_worker.pcm16(source, 4.0), dtype="<i2")
        assert int(found.max()) <= 32767
        assert int(found.min()) >= -32768

    def test_pause_is_exactly_the_requested_zero_samples(self):
        found = pocket_worker.silence(RATE, 250)
        assert len(found) == 2 * int(RATE * 0.25)
        assert set(found) == {0}


class TestTheDeliveryHeader:
    def test_an_absent_temperature_stays_absent(self):
        """I-PKT-25. ``None`` reaches the model as "your own default"."""
        found = delivery(speed=1.0)
        assert found.temperature is None
        assert "temperature" not in found.generation({})

    def test_a_supplied_temperature_is_passed_through(self):
        assert delivery(temperature=0.45).generation({})["temperature"] == 0.45

    def test_nonsense_falls_back_rather_than_raising(self):
        found = delivery(speed="fast", pitch=None, gain=float("nan"), pause_ms=-5)
        assert found.speed == 1.0
        assert found.pitch == 1.0
        assert found.gain == 1.0
        assert found.pause_ms == 0

    def test_pocket_has_no_language_or_sampling_knobs_of_sopros(self):
        """Section 16.5. Absent because nobody has given them a purpose here."""
        assert not hasattr(delivery(), "language")
        assert not hasattr(delivery(), "top_p")
        assert not hasattr(delivery(), "top_k")


class TestTheThreadPolicyIsASentenceRatherThanASlider:
    def test_there_is_no_thread_count_to_set(self):
        """Section 16.4, section 35. PocketTTS 3.0.2 calls
        ``torch.set_num_threads(1)`` and takes its parallelism from its own
        threads, so a number here would be a number that means nothing."""
        assert "torch.set_num_threads(1)" in pocket_worker.thread_policy()
        assert not hasattr(pocket_worker, "INTRAOP_THREADS")
        assert not hasattr(pocket_worker, "OVERRIDE_INTRAOP")


class TestErrorsCarryNothingPrivate:
    def test_a_library_exception_crosses_as_its_class_only(self):
        """I-PKT-27. A library message can carry a path or the spoken text."""
        exc = RuntimeError("failed on /home/someone/voice/clones/abc/reference.wav")
        assert pocket_worker._safe(exc) == "RuntimeError"

    def test_this_workers_own_refusals_are_sentences(self):
        assert pocket_worker._safe(
            pocket_worker.Refusal("That voice is not one PocketTTS has.")) \
            == "That voice is not one PocketTTS has."

    def test_a_library_value_error_is_not_mistaken_for_one_of_this_files(self):
        """``ValueError`` is the base of much of the numeric stack, so it cannot
        be the test for "this file wrote it". ``json`` raises a subclass of it
        for a malformed document and names the document; NumPy and Torch raise
        it for a shape and name the shape. Only :class:`Refusal` crosses whole.
        """
        import json

        try:
            json.loads("{" + '"reference": "/home/someone/clones/abc.wav"')
        except ValueError as exc:
            found = pocket_worker._safe(exc)
        assert found == "JSONDecodeError", found

        numeric = ValueError("could not broadcast input array from shape (2,) into (3,)")
        assert pocket_worker._safe(numeric) == "ValueError"

    def test_every_refusal_this_file_raises_is_one_of_its_own(self):
        """Belt for the rule above: a new ``raise ValueError`` added here would
        be a message forwarded as a class name and a refusal the user could not
        act on, which is the quiet half of the same bug."""
        import pathlib

        source = pathlib.Path(pocket_worker.__file__).read_text(encoding="utf-8")
        assert "raise ValueError(" not in source


# --------------------------------------------------------------------------- #
# The drain, against the real command loop
# --------------------------------------------------------------------------- #


class FakeStream:
    """A pipe the test can feed frames into and read frames out of.

    Reading blocks until something is written, which is what makes the real
    ``serve`` loop behave the way it does in production: it sits in
    :func:`read_frame` waiting, rather than spinning to the end of a canned
    script and exiting before the lane has done anything.
    """

    def __init__(self):
        self._buffer = bytearray()
        self._closed = False
        self._ready = threading.Condition()

    def feed(self, header, payload=b""):
        out = io.BytesIO()
        pocket_worker.write_frame(out, header, payload)
        with self._ready:
            self._buffer += out.getvalue()
            self._ready.notify_all()

    def close(self):
        with self._ready:
            self._closed = True
            self._ready.notify_all()

    def read(self, count):
        with self._ready:
            while len(self._buffer) < count and not self._closed:
                self._ready.wait(0.05)
            found = bytes(self._buffer[:count])
            del self._buffer[:len(found)]
            return found


class Collector:
    """Everything the worker wrote, decoded as it arrives."""

    def __init__(self):
        self._buffer = bytearray()
        self.frames = []
        self._lock = threading.Lock()

    def write(self, block):
        with self._lock:
            self._buffer += block
            self._drain()
        return len(block)

    def flush(self):
        pass

    def _drain(self):
        while True:
            raw = bytes(self._buffer)
            try:
                found = pocket_worker.read_frame(io.BytesIO(raw))
            except Exception:
                return
            if found is None:
                return
            header, payload = found
            used = 8 + len(pocket_worker.json.dumps(header).encode("utf-8")) + len(payload)
            del self._buffer[:used]
            self.frames.append((header, payload))

    def ops(self):
        with self._lock:
            return [header.get("op") for header, _payload in self.frames]

    def of(self, operation):
        with self._lock:
            return [(header, payload) for header, payload in self.frames
                    if header.get("op") == operation]

    def wait_for(self, operation, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.of(operation)
            if found:
                return found
            time.sleep(0.01)
        raise AssertionError(f"the worker never sent {operation}; it sent {self.ops()}")


class FakeEngine:
    """PocketTTS's shape without PocketTTS, and with a generator that can be watched.

    Its whole job is to be *slow in a controllable way*: ``stream`` yields
    ``chunks`` blocks with a gate between them, so a test can interrupt a turn
    while a unit is genuinely in flight and then assert that this object was
    consumed to the end anyway. That last part is what makes it a test of
    I-PKT-12 rather than a test of a flag.
    """

    def __init__(self, config=None, chunks=6, gate=None):
        found = dict(config or {})
        self.model_root = ""
        self.config_path = ""
        self.official_root = ""
        self.clones_root = ""
        self.model_id = str(found.get("model_id") or "english")
        self.precision = str(found.get("precision") or "full")
        self.sampler_steps = int(found.get("sampler_steps") or 1)
        self.fingerprint = str(found.get("fingerprint") or "")
        self.state_schema = 1
        self.sample_rate = RATE
        self.cloning_ready = True
        self.voices = dict(found.get("voices") or {})
        self.defaults = {"temperature": 0.3}
        self.upstream_build_id = "fake"
        self.chunks = int(chunks)
        self.gate = gate
        self.started = []
        self.consumed = []
        self.offered = []
        self.loaded = False

    def load(self):
        self.loaded = True

    def device(self):
        return "cpu"

    def refresh_catalog(self, voices, forget=()):
        self.voices = dict(voices or {})
        return len(self.voices)

    def forget(self, voice_id):
        self.voices.pop(str(voice_id), None)

    def state_for(self, voice_id):
        return {"voice": voice_id}

    def stream(self, text, voice_id, delivery, on_audio, listening):
        self.started.append(text)
        produced = 0
        for index in range(self.chunks):
            if self.gate is not None:
                self.gate(index)
            produced += 1
            # Consumed *whatever* the listener says. That is the invariant:
            # abandoning this generator would leave upstream's generation
            # thread running against an unbounded internal queue.
            block = b"\x00\x01" * 240
            if listening():
                self.offered.append(index)
                on_audio(block)
        self.consumed.append(produced)
        return {"blocks": produced, "first_block_ms": 1, "synth_ms": 10,
                "audio_ms": produced * 10, "streaming": "chunk"}

    def synthesize(self, text, voice_id, delivery):
        return b"\x00\x01" * 240


def run_worker(engine, feed):
    """Drive the real ``serve`` loop on a thread and return what it wrote."""
    stdin = FakeStream()
    stdout = Collector()
    thread = threading.Thread(
        target=pocket_worker.serve,
        args=(stdin, stdout),
        kwargs={"engine_factory": lambda config: engine},
        daemon=True)
    thread.start()
    stdin.feed({"op": "init", "id": 1, "parent_pid": 0, "config": {}})
    stdout.wait_for("ready")
    try:
        feed(stdin, stdout)
    finally:
        stdin.feed({"op": "shutdown", "id": 99})
        thread.join(timeout=5.0)
        stdin.close()
    return stdout


class TestTheHandshakeSaysWhatSectionEighteenAsksFor:
    def test_every_declared_field_is_present_and_nothing_private_is(self):
        engine = FakeEngine()
        found = run_worker(engine, lambda stdin, out: None)
        header, _payload = found.of("ready")[0]
        for name in ("protocol", "engine", "backend", "pocket_version",
                     "upstream_build_id", "torch_version", "sample_rate", "provider",
                     "device", "containment", "model_id", "model_fingerprint",
                     "quantization", "sampler_steps", "streaming", "interrupt_mode",
                     "thread_policy", "voice_state_schema"):
            assert name in header, name
        assert header["engine"] == "pocket"
        assert header["backend"] == "pocket-tts-native"
        assert header["provider"] == "cpu"
        assert header["device"] == "cpu"
        assert header["interrupt_mode"] == "drain_unit"
        # T-PKT-W-17 and section 18's "do not report" list.
        text = pocket_worker.json.dumps(header)
        for forbidden in ("/home", "\\\\Users", "python.exe", "pid", "token"):
            assert forbidden not in text


class TestInterruptionIsADrainAndNotACancellation:
    """GATE P-3, at the level a test can reach without PocketTTS installed."""

    def test_the_in_flight_unit_is_consumed_to_the_end(self):
        """I-PKT-12. Muted is not "stop reading".

        The fake generator counts how many blocks it produced. If the worker
        abandoned it on interruption that count would be short, and upstream's
        real generation thread would be the thing left running.
        """
        seen = threading.Event()
        release = threading.Event()

        def gate(index):
            if index == 1:
                seen.set()
                release.wait(3.0)

        engine = FakeEngine(chunks=6, gate=gate)

        def feed(stdin, out):
            stdin.feed({"op": "tts_begin", "turn": "T1", "voice_id": "pocket:official:alba"})
            stdin.feed({"op": "tts_text", "turn": "T1"}, b"Hello there.")
            stdin.feed({"op": "tts_end", "turn": "T1"})
            assert seen.wait(3.0)
            stdin.feed({"op": "tts_interrupt", "turn": "T1"})
            out.wait_for("tts_interrupted")
            release.set()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not engine.consumed:
                time.sleep(0.01)

        found = run_worker(engine, feed)
        assert engine.consumed == [6], "the abandoned unit was not consumed to the end"
        complete = [header for header, _p in found.of("tts_interrupted")
                    if header.get("state") == "complete"]
        assert complete, "the worker never said its lane was free"

    def test_no_audio_is_offered_after_the_interrupt(self):
        """T-PKT-E2E-5, at the worker. Blocks after the interrupt are dropped
        here rather than written and dropped at the far end."""
        seen = threading.Event()
        release = threading.Event()

        def gate(index):
            if index == 1:
                seen.set()
                release.wait(3.0)

        engine = FakeEngine(chunks=8, gate=gate)

        def feed(stdin, out):
            stdin.feed({"op": "tts_begin", "turn": "T2", "voice_id": "v"})
            stdin.feed({"op": "tts_text", "turn": "T2"}, b"Hello.")
            stdin.feed({"op": "tts_end", "turn": "T2"})
            assert seen.wait(3.0)
            stdin.feed({"op": "tts_interrupt", "turn": "T2"})
            out.wait_for("tts_interrupted")
            release.set()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not engine.consumed:
                time.sleep(0.01)

        found = run_worker(engine, feed)
        # One block was offered before the gate opened; nothing after it.
        assert engine.offered == [0]
        assert len(found.of("tts_audio")) == 1

    def test_a_queued_later_unit_never_enters_the_model(self):
        """Section 21.3. Unit 2 drains, unit 3 is discarded, unit 4 never sent."""
        seen = threading.Event()
        release = threading.Event()

        def gate(index):
            if index == 1:
                seen.set()
                release.wait(3.0)

        engine = FakeEngine(chunks=4, gate=gate)

        def feed(stdin, out):
            stdin.feed({"op": "tts_begin", "turn": "T3", "voice_id": "v"})
            stdin.feed({"op": "tts_text", "turn": "T3"}, b"First unit.")
            assert seen.wait(3.0)
            stdin.feed({"op": "tts_text", "turn": "T3"}, b"Second unit.")
            stdin.feed({"op": "tts_interrupt", "turn": "T3"})
            out.wait_for("tts_interrupted")
            stdin.feed({"op": "tts_text", "turn": "T3"}, b"Third unit.")
            release.set()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not engine.consumed:
                time.sleep(0.01)

        found = run_worker(engine, feed)
        assert engine.started == ["First unit."], engine.started
        complete = [header for header, _p in found.of("tts_interrupted")
                    if header.get("state") == "complete"]
        assert complete and complete[-1]["dropped_units"] >= 1

    def test_interrupting_a_turn_that_never_started_frees_the_lane_at_once(self):
        """A Play control that waited for a drain that never happened would be a
        control that stayed disabled for no reason."""
        engine = FakeEngine(chunks=1)

        def feed(stdin, out):
            stdin.feed({"op": "tts_interrupt", "turn": "never-existed"})
            out.wait_for("tts_interrupted")

        found = run_worker(engine, feed)
        header, _payload = found.of("tts_interrupted")[0]
        assert header["state"] == "complete"

    def test_an_uninterrupted_turn_still_ends_with_done(self):
        """The ordinary path is unchanged by any of the above."""
        engine = FakeEngine(chunks=3)

        def feed(stdin, out):
            stdin.feed({"op": "tts_begin", "turn": "T4", "voice_id": "v"})
            stdin.feed({"op": "tts_text", "turn": "T4"}, b"All of it.")
            stdin.feed({"op": "tts_end", "turn": "T4"})
            out.wait_for("tts_done")

        found = run_worker(engine, feed)
        assert found.of("tts_done")
        assert not found.of("tts_interrupted")
        assert len(found.of("tts_audio")) == 3
        assert found.of("tts_segment")


class TestOneInferenceLane:
    def test_an_audition_waits_for_the_turn_rather_than_overlapping_it(self):
        """I-PKT-8. Upstream documents the model as not thread-safe, so a second
        concurrent generation would be incorrect rather than merely slow."""
        order = []
        release = threading.Event()

        def gate(index):
            if index == 0:
                order.append("turn-started")
                release.wait(2.0)

        engine = FakeEngine(chunks=2, gate=gate)
        original = engine.synthesize

        def watched(text, voice_id, delivery):
            order.append("audition")
            return original(text, voice_id, delivery)

        engine.synthesize = watched

        def feed(stdin, out):
            stdin.feed({"op": "tts_begin", "turn": "T5", "voice_id": "v"})
            stdin.feed({"op": "tts_text", "turn": "T5"}, b"Speaking.")
            stdin.feed({"op": "tts_end", "turn": "T5"})
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and "turn-started" not in order:
                time.sleep(0.01)
            stdin.feed({"op": "tts", "id": 5, "voice_id": "v"}, b"An audition.")
            time.sleep(0.1)
            assert order == ["turn-started"], "the audition overlapped the turn"
            release.set()
            out.wait_for("tts_done")
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and "audition" not in order:
                time.sleep(0.01)

        run_worker(engine, feed)
        assert order == ["turn-started", "audition"]


class TestAQueuedRequestIsAlwaysAnswered:
    def test_a_failing_request_answers_rather_than_vanishing(self):
        engine = FakeEngine()

        def explode(text, voice_id, delivery):
            raise RuntimeError("/home/someone/model/weights.safetensors is unreadable")

        engine.synthesize = explode

        def feed(stdin, out):
            stdin.feed({"op": "tts", "id": 11, "voice_id": "v"}, b"anything")
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                found = [h for h, _p in out.frames if h.get("id") == 11]
                if found:
                    return
                time.sleep(0.01)
            raise AssertionError("the request was never answered")

        found = run_worker(engine, feed)
        answers = [h for h, _p in found.frames if h.get("id") == 11]
        assert answers and answers[0]["ok"] is False
        # The class name and nothing else: a library message can carry a path.
        assert answers[0]["error"] == "RuntimeError"


class FakeModel:
    """PocketTTS's generation surface, and a generator that says if it finished.

    ``exhausted`` is the whole assertion of I-PKT-12: upstream's generation
    thread runs until the generator is drained, so a consumer that stopped
    pulling would leave it running against an unbounded internal queue with
    nobody emptying it. Nothing in the real ``Engine`` may return before this
    goes true.
    """

    def __init__(self, chunks=5):
        self.chunks = int(chunks)
        self.exhausted = False
        self.calls = []
        self.state_copies = []
        self.sample_rate = RATE
        self.device = "cpu"

    def generate_audio_stream(self, state, text, copy_state=True, **arguments):
        self.calls.append({"text": text, "copy_state": copy_state, **arguments})
        # What upstream's ``copy_state`` guarantees, made observable: the base
        # state handed in must not be the object that gets mutated.
        self.state_copies.append(dict(state) if copy_state else state)

        def produce():
            for _index in range(self.chunks):
                yield numpy.zeros(240, dtype=numpy.float32)
            self.exhausted = True

        return produce()

    def get_state_for_audio_prompt(self, data):
        return {"prepared": True}


def pocket_engine(model, **config):
    engine = pocket_worker.Engine({"sample_rate": RATE, **config})
    engine.model = model
    engine._numpy = numpy
    engine.voices = {"v": {"kind": "official", "state": "irrelevant"}}
    engine._states["v"] = {"base": 1}
    return engine


class TestTheEngineKeepsConsumingWhileMuted:
    """I-PKT-12, at the only place it can actually be observed."""

    def test_a_muted_unit_is_still_drained_to_the_end(self):
        model = FakeModel(chunks=5)
        engine = pocket_engine(model)
        offered = []
        found = engine.stream("Hello.", "v", pocket_worker.NEUTRAL, offered.append,
                              lambda: False)
        assert model.exhausted, "the generator was abandoned rather than drained"
        assert offered == [], "audio was offered while muted"
        assert found["blocks"] == 5

    def test_an_unmuted_unit_offers_every_block(self):
        model = FakeModel(chunks=4)
        engine = pocket_engine(model)
        offered = []
        engine.stream("Hello.", "v", pocket_worker.NEUTRAL, offered.append, lambda: True)
        assert model.exhausted
        assert len(offered) == 4

    def test_generation_always_asks_for_a_copy_of_the_base_state(self):
        """I-PKT-16 / GATE P-11. A cached base state must stay a base state.

        Without ``copy_state``, a generation may leave the state it was handed
        carrying this sentence's continuation -- and the next reply in that
        voice would start from the middle of the last one.
        """
        model = FakeModel(chunks=2)
        engine = pocket_engine(model)
        for _turn in range(3):
            engine.stream("Anything.", "v", pocket_worker.NEUTRAL, lambda _b: None,
                          lambda: True)
        assert [call["copy_state"] for call in model.calls] == [True, True, True]
        # And the cached base is the same object every time, unmutated.
        assert engine._states["v"] == {"base": 1}

    def test_the_engine_step_count_rather_than_the_turns_reaches_the_model(self):
        """I-PKT-24. The sampler policy is engine-global; a turn cannot change it."""
        model = FakeModel(chunks=1)
        engine = pocket_engine(model, sampler_steps=3)
        engine.stream("Anything.", "v", pocket_worker.Delivery(temperature=0.4),
                      lambda _b: None, lambda: True)
        assert model.calls[0]["sampler_decode_steps"] == 3
        assert model.calls[0]["temperature"] == 0.4


class TestTheVoiceStateCacheIsBounded:
    def test_it_never_grows_past_its_capacity(self):
        """Section 22. An unbounded dictionary in a process that runs for a
        session is a leak with a slow fuse."""
        model = FakeModel()
        engine = pocket_worker.Engine({"sample_rate": RATE})
        engine.model = model
        engine._numpy = numpy
        engine.voices = {f"v{index}": {"kind": "official", "state": "x"}
                         for index in range(pocket_worker.VOICE_CACHE + 5)}
        for index in range(pocket_worker.VOICE_CACHE + 5):
            engine._states[f"v{index}"] = {"index": index}
            while len(engine._states) > pocket_worker.VOICE_CACHE:
                engine._states.popitem(last=False)
        assert len(engine._states) == pocket_worker.VOICE_CACHE

    def test_a_refresh_drops_the_states_the_catalogue_no_longer_names(self):
        engine = pocket_worker.Engine({})
        engine._states["gone"] = object()
        engine._states["kept"] = object()
        engine.refresh_catalog({"kept": {"kind": "clone", "state": "x"}})
        assert "gone" not in engine._states
        assert "kept" in engine._states


class TestTheLocalConfigMayNotNameANetworkLocation:
    """I-PKT-20, enforced in the process that would do the fetching."""

    def test_an_hf_or_https_path_is_refused(self, tmp_path):
        for location in ("hf://kyutai/pocket-tts/model.safetensors",
                         "https://example.invalid/model.safetensors"):
            path = tmp_path / "model.local.yaml"
            path.write_text(pocket_worker.json.dumps({"weights_path": location}),
                            encoding="utf-8")
            engine = pocket_worker.Engine({"config_path": str(path)})
            with pytest.raises(ValueError) as raised:
                engine._read_config()
            assert "network location" in str(raised.value)

    def test_a_local_path_is_accepted(self, tmp_path):
        path = tmp_path / "model.local.yaml"
        path.write_text(pocket_worker.json.dumps({"weights_path": str(tmp_path / "w.st")}),
                        encoding="utf-8")
        engine = pocket_worker.Engine({"config_path": str(path)})
        assert engine._read_config()["weights_path"].endswith("w.st")

    def test_a_missing_config_is_a_sentence_rather_than_a_traceback(self):
        engine = pocket_worker.Engine({"config_path": "/nowhere/at/all.json"})
        with pytest.raises(ValueError) as raised:
            engine._read_config()
        assert "Reinstall" in str(raised.value)


class TestTheLaneNeverBlocksOnADeadPipe:
    def test_audio_stops_being_queued_once_the_writer_has_gone(self):
        """I-PKT-12, at the one place the lane could wedge.

        The lane is the thread that has to reach the end of an abandoned unit.
        If the pipe breaks and nothing is emptying the outbox, a lane spinning
        on ``put`` is a unit that never finishes and a worker that never becomes
        ready again.
        """
        worker = pocket_worker.Worker(io.BytesIO())
        worker._outbox = pocket_worker.queue.Queue(maxsize=1)
        worker._outbox.put(("filler", b""))

        class Dead:
            def is_alive(self):
                return False

        worker._writer = Dead()
        began = time.monotonic()
        worker.send({"op": "tts_audio", "turn": "T"}, b"\x00\x00", audio=True)
        assert time.monotonic() - began < 2.0, "the lane blocked on a dead writer"
        assert worker._stopping is True


class TestTheGeneratorIsDrainedEvenWhenTheConsumerFails:
    def test_an_exception_in_on_audio_does_not_abandon_the_model(self):
        """I-PKT-12, on the path nobody plans for.

        Abandoning the generator is the one thing that must not happen: upstream
        keeps a generation thread running until it is emptied, so a consumer
        that raised halfway would leave one running against an unbounded queue
        with nothing to drain it.
        """
        model = FakeModel(chunks=6)
        engine = pocket_engine(model)

        def explode(_block):
            raise RuntimeError("the pipe went away")

        with pytest.raises(RuntimeError):
            engine.stream("Hello.", "v", pocket_worker.NEUTRAL, explode, lambda: True)
        assert model.exhausted, "the generator was abandoned when the consumer failed"


class TestNothingStartsAfterTheLaneHasBeenReportedFree:
    """The window between a unit leaving the queue and entering the model.

    ``speaking`` is set by the lane and read by the command loop, and for a few
    instructions the honest answer changes under the reader. A Stop landing
    there used to be told "the lane is free" -- after which the lane started the
    unit anyway, the parent released its drain and began the next reply, and two
    generations ran at once in a model upstream documents as not thread-safe
    (I-PKT-13).
    """

    def test_a_stop_between_units_is_not_followed_by_a_generation(self, monkeypatch):
        pulled = threading.Event()
        go = threading.Event()
        real = pocket_worker.Turn.next_segment

        def held(self):
            found = real(self)
            if found is not None:
                # Between the queue and the model, held open so the command loop
                # below lands exactly in the window rather than nearly in it.
                pulled.set()
                go.wait(5.0)
            return found

        monkeypatch.setattr(pocket_worker.Turn, "next_segment", held)
        engine = FakeEngine(chunks=2)

        def feed(stdin, stdout):
            stdin.feed({"op": "tts_begin", "turn": "T1", "voice_id": "pocket:official:alba"})
            stdin.feed({"op": "tts_text", "turn": "T1"}, b"Hello.")
            stdin.feed({"op": "tts_end", "turn": "T1"})
            assert pulled.wait(5.0), "the lane never took a unit off the queue"
            stdin.feed({"op": "tts_interrupt", "turn": "T1"})
            stdout.wait_for("tts_interrupted")
            go.set()
            time.sleep(0.2)

        found = run_worker(engine, feed)
        assert engine.started == [], \
            "a unit was generated after the lane had been reported free"
        reports = [header for header, _payload in found.of("tts_interrupted")]
        assert [item.get("state") for item in reports] == ["complete"], reports


class TestTheDrainReportsTheSizeOfWhatItThrewAway:
    def test_the_completing_frame_carries_the_unit_and_the_units_dropped(self):
        """GATE P-3 is a cost against a size. "A Stop took 4.2 seconds" is only
        a finding if something also recorded that the unit was seventy-eight
        characters and that two more were discarded behind it."""
        started = threading.Event()
        release = threading.Event()

        def gate(index):
            if index == 0:
                started.set()
                release.wait(5.0)

        engine = FakeEngine(chunks=3, gate=gate)

        def feed(stdin, stdout):
            stdin.feed({"op": "tts_begin", "turn": "T1", "voice_id": "pocket:official:alba"})
            stdin.feed({"op": "tts_text", "turn": "T1"}, b"A sentence of some length.")
            assert started.wait(5.0), "the engine never entered a unit"
            stdin.feed({"op": "tts_text", "turn": "T1"}, b"And another one behind it.")
            stdin.feed({"op": "tts_interrupt", "turn": "T1"})
            release.set()
            stdout.wait_for("tts_interrupted")
            time.sleep(0.1)

        found = run_worker(engine, feed)
        complete = [header for header, _payload in found.of("tts_interrupted")
                    if header.get("state") == "complete"]
        assert complete, [header for header, _ in found.of("tts_interrupted")]
        assert complete[0]["chars"] == len("A sentence of some length.")
        assert complete[0]["audio_ms"] > 0
        assert complete[0]["dropped_units"] == 1

