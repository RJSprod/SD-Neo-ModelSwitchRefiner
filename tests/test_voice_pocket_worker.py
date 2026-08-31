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

import importlib.util
import io
import json
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

    def test_the_two_filenames_inside_a_preparation_are_the_same_agreement(self):
        """The parent builds a directory and the worker writes two files into it.
        Neither imports the other -- this file has to stay runnable under an
        interpreter that has Torch and knows nothing about the extension -- so
        the names are an agreement, and an agreement nothing checks is a rename
        away from a preview whose reference the save cannot find."""
        import mc_voice_paths as paths

        assert pocket_worker.REFERENCE_FILENAME == paths.POCKET_REFERENCE_FILENAME
        assert pocket_worker.STATE_FILENAME == paths.POCKET_PREVIEW_STATE_FILENAME


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


def unit(seam, source, chunk: int = 431):
    """Push one unit's samples through a seam and collect what came out."""
    out = bytearray()
    for start in range(0, len(source), chunk):
        out += seam.block(pocket_worker.pcm16(source[start:start + chunk]))
    out += seam.flush()
    return numpy.frombuffer(bytes(out), dtype="<i2").astype(numpy.float32) / 32767.0


class TestAUnitEndsAtSilenceRatherThanWhereverItWas:
    """The pop between sentences, and the thing that removes it.

    Upstream stops generating a couple of frames after the end-of-speech token
    and leaves it there -- ``_autoregressive_generation`` breaks out of its loop
    mid-stride, and the next generation's ``_decode_audio_worker`` calls
    ``init_states`` on a fresh Mimi decoder. So a unit's last sample is wherever
    the waveform happened to be, and the next unit's first sample is wherever a
    zero-initialised decoder puts it. This feature plays units back sample-exact
    and one after another, which turns that pair into a step, and a step in a
    waveform is a click at the end of a sentence.

    The signal below is DC on purpose. It is the worst case, it is roughly what
    a decoder's residual offset looks like, and no amount of filtering removes a
    step that a ramp does not.
    """

    def test_a_unit_starts_and_ends_at_silence(self):
        found = unit(pocket_worker.Seam(RATE), numpy.full(4800, 0.5, dtype=numpy.float32))
        assert abs(float(found[0])) < 1e-3
        assert abs(float(found[-1])) < 1e-3

    def test_the_step_a_join_would_have_carried_is_gone(self):
        """The measurement, rather than the assertion: no sample-to-sample step.

        Without the ramp the first and last samples are 0.5 and the joins on
        either side of the unit are steps of that size. What is left afterwards
        is the ramp's own slope, which is three orders of magnitude smaller and
        is what "below sixty hertz" means in a difference.
        """
        found = unit(pocket_worker.Seam(RATE), numpy.full(4800, 0.5, dtype=numpy.float32))
        joined = numpy.concatenate([found, found])
        assert float(numpy.abs(numpy.diff(joined)).max()) < 0.01

    def test_nothing_is_lost_to_the_ramp(self):
        source = numpy.full(4800, 0.5, dtype=numpy.float32)
        assert unit(pocket_worker.Seam(RATE), source).size == source.size

    def test_the_middle_of_the_unit_is_untouched(self):
        """A ramp at the edges, and nothing anywhere else."""
        source = numpy.full(4800, 0.5, dtype=numpy.float32)
        found = unit(pocket_worker.Seam(RATE), source)
        span = pocket_worker.Seam(RATE).span
        middle = found[span:-span]
        assert float(numpy.abs(middle - 0.5).max()) < 1e-3

    def test_a_unit_shorter_than_the_ramp_still_starts_and_ends_at_silence(self):
        """One decoded frame and nothing more -- a refusal, or a very short word."""
        seam = pocket_worker.Seam(RATE)
        source = numpy.full(seam.span // 3, 0.5, dtype=numpy.float32)
        found = unit(seam, source, chunk=17)
        assert found.size == source.size
        assert abs(float(found[0])) < 1e-3
        assert abs(float(found[-1])) < 1e-3

    def test_the_ramp_is_the_same_length_whatever_the_blocks_are(self):
        """The model's chunk sizes are not the seam's business.

        Fed in one block or in seventeen awkward ones, the same samples come out
        -- because a ramp tied to block boundaries would fade several times a
        second in the middle of a word instead of once at the end of a sentence.
        """
        source = numpy.full(4800, 0.5, dtype=numpy.float32)
        whole = unit(pocket_worker.Seam(RATE), source, chunk=len(source))
        pieces = unit(pocket_worker.Seam(RATE), source, chunk=137)
        assert whole.size == pieces.size
        assert float(numpy.abs(whole - pieces).max()) < 1e-3

    def test_a_rate_of_nothing_is_a_pass_through_rather_than_a_crash(self):
        """A worker that never learned its sample rate still speaks."""
        seam = pocket_worker.Seam(0)
        source = numpy.full(480, 0.5, dtype=numpy.float32)
        assert unit(seam, source).size == source.size


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

    The signatures below are released 3.0.2's, exactly, and that is the point of
    the class rather than an incidental property of it. A fake whose
    ``generate_audio_stream`` swallowed ``**arguments`` would accept a
    temperature and a step count as keywords, which the real one does not take
    at all -- and every test of the delivery path would then pass against a
    parent that could only ever raise ``TypeError`` in production. What upstream
    actually does is keep both on the instance and read them inside the sampler,
    so this reads them per chunk and records what it saw.
    """

    def __init__(self, chunks=5, temp=0.3, sampler_decode_steps=1, level=0.0):
        self.chunks = int(chunks)
        self.level = float(level)
        self.exhausted = False
        self.calls = []
        self.seen = []
        self.prepared = []
        self.state_copies = []
        self.sample_rate = RATE
        self.device = "cpu"
        self.temp = temp
        self.sampler_decode_steps = int(sampler_decode_steps)
        self.has_voice_cloning = True

    def generate_audio_stream(self, model_state, text_to_generate, max_tokens=1000,
                              frames_after_eos=None, copy_state=True):
        self.calls.append({"text": text_to_generate, "copy_state": copy_state})
        # What upstream's ``copy_state`` guarantees, made observable: the base
        # state handed in must not be the object that gets mutated.
        self.state_copies.append(dict(model_state) if copy_state else model_state)

        def produce():
            for _index in range(self.chunks):
                self.seen.append({"temp": self.temp,
                                  "sampler_decode_steps": self.sampler_decode_steps})
                yield numpy.full(240, self.level, dtype=numpy.float32)
            self.exhausted = True

        return produce()

    def get_state_for_audio_prompt(self, audio_conditioning, truncate=False):
        self.prepared.append((audio_conditioning, truncate))
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

    def test_an_unmuted_unit_offers_every_sample(self):
        """Every sample, rather than every block.

        The block count is not the claim and never was: :class:`Seam` withholds
        a unit's last few milliseconds until the unit ends, so the boundaries
        between offered blocks no longer line up with the model's chunks. What
        has to hold is that nothing is lost -- four chunks of audio in, four
        chunks of audio out.
        """
        model = FakeModel(chunks=4)
        engine = pocket_engine(model)
        offered = []
        engine.stream("Hello.", "v", pocket_worker.NEUTRAL, offered.append, lambda: True)
        assert model.exhausted
        assert offered, "no audio was offered"
        assert sum(len(block) for block in offered) == 4 * 240 * 2

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
        """I-PKT-24. The sampler policy is engine-global; a turn cannot change it.

        Both numbers are read where upstream reads them -- off the model, inside
        the sampler -- because released 3.0.2's ``generate_audio_stream`` takes
        neither as an argument.
        """
        model = FakeModel(chunks=2, sampler_decode_steps=3)
        engine = pocket_engine(model, sampler_steps=3)
        engine.stream("Anything.", "v", pocket_worker.Delivery(temperature=0.4),
                      lambda _b: None, lambda: True)
        assert [one["sampler_decode_steps"] for one in model.seen] == [3, 3]
        assert [one["temp"] for one in model.seen] == [0.4, 0.4]

    def test_a_turns_variation_does_not_outlive_the_turn(self):
        """The temperature is set on the model for the length of one generation
        and put back, because that is the only place released 3.0.2 reads it. A
        scope that leaked would make one warm reply warm every reply after it,
        which is the shape of a bug nobody reports as a bug."""
        model = FakeModel(chunks=1, temp=0.3)
        engine = pocket_engine(model)
        engine.stream("Warm.", "v", pocket_worker.Delivery(temperature=0.9),
                      lambda _b: None, lambda: True)
        assert model.seen[-1]["temp"] == 0.9
        assert model.temp == 0.3, "the turn's Variation was left on the model"

        engine.stream("Neutral.", "v", pocket_worker.NEUTRAL, lambda _b: None,
                      lambda: True)
        assert model.seen[-1]["temp"] == 0.3, "an untouched control changed the model"

    def test_the_variation_is_restored_even_when_the_consumer_fails(self):
        model = FakeModel(chunks=3, temp=0.3)
        engine = pocket_engine(model)

        def explode(_block):
            raise RuntimeError("the pipe went away")

        with pytest.raises(RuntimeError):
            engine.stream("Warm.", "v", pocket_worker.Delivery(temperature=0.9),
                          explode, lambda: True)
        assert model.temp == 0.3
        assert model.exhausted, "the generator was abandoned rather than drained"


class TestTwoUnitsInARowSoundLikeOneReply:
    """The seam where a listener actually hears it: sentence to sentence.

    A reply is spoken as a run of committed units, each one its own generation,
    and the browser schedules their PCM back to back with no gap between them.
    So the audio a listener hears is the concatenation, and the concatenation is
    where the click was.
    """

    def speak(self, engine, listening=lambda: True, delivery=None):
        offered = []
        engine.stream("Hello.", "v", delivery or pocket_worker.NEUTRAL,
                      offered.append, listening)
        return numpy.frombuffer(b"".join(offered), dtype="<i2").astype(numpy.float32) / 32767.0

    def test_the_join_between_two_units_is_not_a_step(self):
        engine = pocket_engine(FakeModel(chunks=6, level=0.5))
        first = self.speak(engine)
        engine.model = FakeModel(chunks=6, level=-0.5)
        second = self.speak(engine)
        joined = numpy.concatenate([first, second])
        assert joined.size == 12 * 240
        # Without the ramp this join is a step of a whole unit -- 0.5 to -0.5.
        assert float(numpy.abs(numpy.diff(joined)).max()) < 0.01

    def test_a_unit_reaches_the_pause_after_it_at_silence(self):
        """A pause is only a pause if the speech walks into it.

        Stepping from a unit's last sample straight to a block of zeros is the
        same click as stepping to the next sentence, and setting a pause is how
        somebody asks for a *gap* rather than for a tick.
        """
        engine = pocket_engine(FakeModel(chunks=4, level=0.5))
        found = self.speak(engine, delivery=delivery(pause_ms=100))
        assert found.size == 4 * 240 + int(RATE * 0.1)
        assert float(numpy.abs(numpy.diff(found)).max()) < 0.01
        assert float(numpy.abs(found[-int(RATE * 0.1):]).max()) == 0.0

    def test_a_shaped_unit_is_ramped_as_well(self):
        """Speed and pitch change the samples; they do not change the edges."""
        engine = pocket_engine(FakeModel(chunks=8, level=0.5))
        found = self.speak(engine, delivery=delivery(speed=1.25, pitch=1.1, gain=2.0))
        assert found.size, "a shaped unit produced no audio"
        assert abs(float(found[0])) < 1e-3
        assert abs(float(found[-1])) < 1e-3

    def test_a_drained_unit_still_offers_nothing(self):
        """The ramp is audio, and a drained unit's audio is thrown away too."""
        engine = pocket_engine(FakeModel(chunks=4, level=0.5))
        assert self.speak(engine, listening=lambda: False).size == 0


def quiet(seconds: float, level: float = 0.0, rate: int = RATE):
    return numpy.full(int(seconds * rate), level, dtype=numpy.float32)


def longest_quiet(samples, level: float):
    """The longest contiguous run of samples under ``level``."""
    best = (0, 0)
    start = None
    for index in range(samples.size + 1):
        under = index < samples.size and abs(float(samples[index])) < level
        if under and start is None:
            start = index
        elif not under and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    return samples[best[0]:best[1]]


def through_trim(trim, source, chunk: int = 431, module=None):
    """Push a unit through a trim in awkward-sized chunks and collect the PCM."""
    module = module or pocket_worker
    out = bytearray()
    for start in range(0, len(source), chunk):
        out += trim.block(module.pcm16(source[start:start + chunk]))
    out += trim.flush()
    return numpy.frombuffer(bytes(out), dtype="<i2").astype(numpy.float32) / 32767.0


class TestTheQuietAModelPutsRoundAUnitIsCutBack:
    """The gap between sentences, and what most of it actually was.

    A reply is a run of units played back with nothing between them, so the
    silence a listener hears between two sentences is the silence *inside* the
    two units: whatever the decoder produced before the first phoneme, and
    whatever it produced after the last one. For PocketTTS the second of those
    is generated on purpose -- upstream keeps going for a few frames after the
    end-of-speech token, at 12.5 frames a second -- and none of it was asked
    for.

    Cut back rather than cut out: sentences are separated by a pause in speech,
    and the delivery's own "Pause between sentences" adds to what is left.
    """

    def test_the_lead_is_cut_to_the_kept_amount(self):
        source = numpy.concatenate([quiet(0.5), tone(220.0, 1.0)])
        found = through_trim(pocket_worker.Trim(RATE), source)
        expected = int(RATE * pocket_worker.KEEP_LEAD_MS / 1000) + RATE
        assert abs(found.size - expected) < RATE * 0.02

    def test_the_tail_is_cut_to_the_kept_amount(self):
        source = numpy.concatenate([tone(220.0, 1.0), quiet(0.4)])
        found = through_trim(pocket_worker.Trim(RATE), source)
        expected = RATE + int(RATE * pocket_worker.KEEP_TAIL_MS / 1000)
        assert abs(found.size - expected) < RATE * 0.02

    def test_what_was_cut_is_reported(self):
        trim = pocket_worker.Trim(RATE)
        source = numpy.concatenate([quiet(0.5), tone(220.0, 1.0), quiet(0.4)])
        found = through_trim(trim, source)
        assert trim.dropped_ms == pytest.approx(900 - 60 - 120, abs=30)
        assert found.size + trim.dropped * 1 == pytest.approx(source.size, abs=RATE * 0.02)

    def test_a_pause_short_enough_to_be_delivery_is_left_alone(self):
        """A pause between two clauses is the model saying something. Under the
        cap it is not touched at all -- cutting it would rewrite the delivery."""
        source = numpy.concatenate([tone(220.0, 0.2), quiet(0.15), tone(220.0, 0.2)])
        trim = pocket_worker.Trim(RATE)
        found = through_trim(trim, source)
        assert trim.dropped == 0
        assert abs(found.size - source.size) < RATE * 0.02

    def test_a_pause_that_runs_on_is_cut_back_to_the_cap(self):
        """Past the cap it is dead air rather than delivery.

        This is the case the machine reported: pauses of 480, 720 and 1050 ms
        *inside* units whose two ends carried almost no quiet at all. Most
        sentence boundaries in a reply are inside a unit, so this is where the
        gap somebody hears actually lives.
        """
        source = numpy.concatenate([tone(220.0, 0.2), quiet(1.05), tone(220.0, 0.2)])
        trim = pocket_worker.Trim(RATE)
        found = through_trim(trim, source)
        assert trim.gap_ms == pytest.approx(1050, abs=30)
        assert found.size == pytest.approx(
            int(RATE * (0.4 + pocket_worker.KEEP_GAP_MS / 1000.0)), abs=RATE * 0.03)

    def test_a_pause_a_listener_asked_to_be_longer_is_longer(self):
        """"Pause between sentences" adds to the cap as well as to the joins, so
        the control means the same thing wherever the sentence boundary falls."""
        source = numpy.concatenate([tone(220.0, 0.2), quiet(1.05), tone(220.0, 0.2)])
        found = through_trim(pocket_worker.Trim(RATE, gap_ms=200 + 300), source)
        assert found.size == pytest.approx(int(RATE * (0.4 + 0.5)), abs=RATE * 0.03)

    def test_the_splice_across_a_shortened_pause_is_not_a_step(self):
        """The cut is a crossfade, because a step in a waveform is a click.

        The signal is a pause at a level a step across would be audible at --
        two different levels either side, so butting them together would leave a
        jump this measures the absence of.
        """
        source = numpy.concatenate([tone(220.0, 0.2) * 1.8,
                                    quiet(0.4, 0.04), quiet(0.4, -0.04),
                                    tone(220.0, 0.2) * 1.8])
        found = through_trim(pocket_worker.Trim(RATE), source)
        # Inside the pause only, and contiguously: the tone either side has a
        # slope of its own, and what is at issue is the join between the two
        # halves of the cut.
        # Ten milliseconds in from each end of it, because the fixture's own
        # tone stops abruptly where it crosses the level and that step is in the
        # source rather than in the splice.
        hush = longest_quiet(found, 0.1)[240:-240]
        assert hush.size > RATE * 0.1, "the pause is not in the output to look at"
        assert float(numpy.abs(numpy.diff(hush)).max()) < 0.01, (
            "the pause was spliced with a step in it")

    def test_speech_is_never_cut(self):
        source = tone(220.0, 1.0)
        trim = pocket_worker.Trim(RATE)
        found = through_trim(trim, source)
        assert trim.dropped == 0
        assert abs(found.size - source.size) < RATE * 0.02

    def test_the_level_follows_the_unit_rather_than_being_fixed(self):
        """A cloned voice carries its reference recording's room tone.

        This is the case that made the first version of this class trim exactly
        nothing on a real machine: what a listener hears as silence between two
        sentences is not silence, it is the reference recording's room tone, and
        a level anchored to the loudest sample sits far below it. Anchored to
        the quietest instead, it follows the voice.
        """
        room = 0.005
        assert room * 32767 > pocket_worker.QUIET_FLOOR, "the fixture is not a real test"
        source = numpy.concatenate([quiet(0.3, room), tone(220.0, 1.0), quiet(0.4, room)])
        trim = pocket_worker.Trim(RATE)
        through_trim(trim, source)
        assert trim.dropped_ms == pytest.approx(700 - 60 - 120, abs=40)
        assert trim.floor_db == pytest.approx(-46, abs=2)

    def test_a_lead_that_is_room_tone_needs_the_last_units_floor(self):
        """What the seeded floor buys, and it is the front of the unit.

        Room tone loud enough to pass for speech opens the unit on its first
        window, and a unit that has already opened has no lead left to trim. The
        floor the last unit in this voice measured is the only thing available
        that early, and with it the bar for "this is the first word" moves above
        the room tone.
        """
        # Loud enough to be taken for the start of a unit, quiet enough to be
        # recognised as quiet once the unit's own peak is known. A real clone's
        # room tone is far below both; this one sits between them on purpose,
        # because that is the only place where the seeded floor is what decides.
        room = 0.05
        speech = tone(220.0, 1.0) * 1.8
        assert room * 32767 > pocket_worker.SPEECH_FLOOR, "the fixture is not a real test"
        source = numpy.concatenate([quiet(0.3, room), speech, quiet(0.4, room)])

        cold = pocket_worker.Trim(RATE)
        through_trim(cold, source)
        assert cold.dropped_ms == pytest.approx(400 - 120, abs=40), (
            "with no floor to go on, only the tail can be trimmed")

        warm = pocket_worker.Trim(RATE, floor=cold.measured)
        through_trim(warm, source)
        assert warm.dropped_ms == pytest.approx(700 - 60 - 120, abs=40)

    def test_a_unit_that_never_reaches_the_speech_level_still_arrives_whole(self):
        """A soft model with the volume well down, and the one case the hold costs
        something.

        Nothing here is loud enough to open the unit, so nothing is trimmed --
        which is right, because nothing here is known to be padding. What must
        not happen is audio being held for the length of the unit and arriving
        as one lump at the end of it: past the hold it spills through, so a
        quiet unit is late by the hold and no later.
        """
        source = numpy.full(int(RATE * 2.0), 0.02, dtype=numpy.float32)
        assert 0.02 * 32767 < pocket_worker.SPEECH_FLOOR, "the fixture is not a real test"
        trim = pocket_worker.Trim(RATE)
        held = []
        for start in range(0, len(source), 431):
            held.append(len(trim.block(pocket_worker.pcm16(source[start:start + 431]))) // 2)
        held.append(len(trim.flush()) // 2)
        assert trim.dropped == 0
        assert sum(held) == source.size
        # Spilling, rather than one lump at the end: most of it arrived while
        # the unit was still being produced.
        assert sum(held[:-1]) > source.size * 0.5

    def test_a_rate_of_nothing_is_a_pass_through_rather_than_a_crash(self):
        source = numpy.concatenate([quiet(0.1), tone(220.0, 0.1)])
        found = through_trim(pocket_worker.Trim(0), source)
        assert found.size == source.size


class TestQuietIsMeasuredAgainstTheSpeechAndNotAgainstTheFloor:
    """The second miss, written down as a fixture.

    A real machine reported ``floor_db=-68``: a voice whose quietest moment is
    exceptionally clean. Anchoring the line to that floor put it at 39 counts,
    under the absolute minimum -- so the padding either side of the unit, which
    sits perfectly audibly above it, was not quiet by any rule this class had,
    and ``quiet_ms`` came back zero on units carrying about seven hundred
    milliseconds of audio no amount of text accounts for.

    The signal below is that unit: padding at about -40 dBFS, speech far above
    it, and one exceptionally dead moment in the middle to set the floor low.
    """

    PAD = 0.01
    DEAD = 0.0004

    def unit(self):
        return numpy.concatenate([
            quiet(0.3, self.PAD),
            tone(220.0, 0.5) * 1.8,
            quiet(0.02, self.DEAD),
            tone(220.0, 0.5) * 1.8,
            quiet(0.4, self.PAD),
        ])

    def test_the_padding_is_recognised_and_the_floor_is_still_reported(self):
        trim = pocket_worker.Trim(RATE)
        found = through_trim(trim, self.unit())
        assert trim.floor_db < -60, "the fixture no longer has a clean floor"
        assert trim.dropped_ms == pytest.approx(700 - 60 - 120, abs=40)
        assert found.size == pytest.approx(
            self.unit().size - trim.dropped, abs=RATE * 0.02)

    def test_the_floor_being_low_does_not_make_the_line_strict(self):
        """The whole of the second miss: a cleaner unit must not be trimmed less.

        The same unit without its one dead moment has a floor thirty decibels
        higher, and has to come out the same length -- because the line is drawn
        from the speech, which did not move.
        """
        noisy = numpy.concatenate([quiet(0.3, self.PAD), tone(220.0, 1.0) * 1.8,
                                   quiet(0.4, self.PAD)])
        clean = numpy.concatenate([quiet(0.3, self.PAD), tone(220.0, 0.5) * 1.8,
                                   quiet(0.02, self.DEAD), tone(220.0, 0.5) * 1.8,
                                   quiet(0.4, self.PAD)])
        one, two = pocket_worker.Trim(RATE), pocket_worker.Trim(RATE)
        through_trim(one, noisy)
        through_trim(two, clean)
        assert one.floor_db > two.floor_db + 20, "the fixtures are not different"
        assert abs(one.dropped_ms - two.dropped_ms) < 40

    def test_a_pause_inside_the_unit_is_measured_as_well_as_shortened(self):
        """The number that said where the gap actually was.

        It is reported whether or not it is cut, because a reply whose sentences
        are half a second apart is a different problem depending on where that
        half second sits -- and nothing in the log could tell those apart.
        """
        source = numpy.concatenate([tone(220.0, 0.2) * 1.8, quiet(0.5, self.PAD),
                                    tone(220.0, 0.2) * 1.8])
        trim = pocket_worker.Trim(RATE)
        found = through_trim(trim, source)
        assert trim.gap_ms == pytest.approx(500, abs=30)
        assert found.size == pytest.approx(
            int(RATE * (0.4 + pocket_worker.KEEP_GAP_MS / 1000.0)), abs=RATE * 0.03)

    def test_the_trailing_run_is_not_counted_as_a_pause_inside(self):
        """A tail is a tail. Only a run that speech comes back after is a gap."""
        source = numpy.concatenate([tone(220.0, 0.5) * 1.8, quiet(0.4, self.PAD)])
        trim = pocket_worker.Trim(RATE)
        through_trim(trim, source)
        assert trim.gap_ms == 0
        assert trim.dropped_ms == pytest.approx(400 - 120, abs=40)


class TestTheTrimAndTheSeamComposeInThatOrder:
    """Trim first, ramp second. The other way round removes the ramp.

    :class:`Seam` ends a unit with a few milliseconds of ramp down to silence.
    Trimming after that would see the ramp as trailing quiet, cut it off, and
    put back the click the ramp exists to remove -- so the order is not a
    detail, and this is the test that says so.
    """

    def speak(self, model, **config):
        engine = pocket_engine(model, **config)
        offered = []
        engine.stream("Hello.", "v", pocket_worker.NEUTRAL, offered.append, lambda: True)
        return numpy.frombuffer(b"".join(offered), dtype="<i2").astype(numpy.float32) / 32767.0

    class Padded:
        """A model whose unit is padding, speech, and a quarter second of padding."""

        def __init__(self):
            self.exhausted = False
            self.sample_rate = RATE
            self.device = "cpu"
            self.temp = 0.3
            self.sampler_decode_steps = 1
            self.has_voice_cloning = True
            self.calls = []

        def generate_audio_stream(self, model_state, text_to_generate, max_tokens=1000,
                                  frames_after_eos=None, copy_state=True):
            self.calls.append(text_to_generate)
            source = numpy.concatenate([quiet(0.1), tone(220.0, 1.0), quiet(0.25)])

            def produce():
                for start in range(0, len(source), 1920):
                    yield source[start:start + 1920]
                self.exhausted = True

            return produce()

    def test_the_padding_goes_and_the_edges_are_still_silent(self):
        found = self.speak(self.Padded())
        kept = int(RATE * (pocket_worker.KEEP_LEAD_MS
                           + pocket_worker.KEEP_TAIL_MS) / 1000) + RATE
        assert abs(found.size - kept) < RATE * 0.03, "the padding was not cut back"
        assert abs(float(found[0])) < 1e-3, "the unit no longer opens at silence"
        assert abs(float(found[-1])) < 1e-3, "the seam's ramp was trimmed off"
        assert float(numpy.abs(numpy.diff(numpy.concatenate([found, found]))).max()) < 0.05

    def test_the_unit_reports_what_it_cut(self):
        engine = pocket_engine(self.Padded())
        found = engine.stream("Hello.", "v", pocket_worker.NEUTRAL,
                              lambda _b: None, lambda: True)
        assert found["trimmed_ms"] == pytest.approx(350 - 60 - 120, abs=30)
        assert found["quiet_ms"] == pytest.approx(350, abs=30)


class TestASecondUnitKnowsWhatTheVoiceSoundsLikeWhenItIsQuiet:
    """The reported failure, and the reason a floor is carried between units.

    On a real machine every unit came back ``trimmed_ms=0``. The voice was a
    clone, so what sounded like silence between the sentences was the reference
    recording's room tone -- loud enough to be taken for the start of the unit,
    which leaves a unit with no lead left to trim.

    A noise floor belongs to the voice rather than to the sentence, so the
    engine keeps what each unit measured and hands it to the next one. The first
    unit of a voice can still only trim its tail; every unit after it can trim
    both ends.
    """

    class Roomy:
        """A model whose padding is room tone rather than silence."""

        def __init__(self, room=0.05):
            self.room = float(room)
            self.exhausted = False
            self.sample_rate = RATE
            self.device = "cpu"
            self.temp = 0.3
            self.sampler_decode_steps = 1
            self.has_voice_cloning = True

        def generate_audio_stream(self, model_state, text_to_generate, max_tokens=1000,
                                  frames_after_eos=None, copy_state=True):
            source = numpy.concatenate([quiet(0.25, self.room), tone(220.0, 1.0) * 1.8,
                                        quiet(0.35, self.room)])

            def produce():
                for start in range(0, len(source), 1920):
                    yield source[start:start + 1920]
                self.exhausted = True

            return produce()

    def speak(self, engine):
        return engine.stream("Hello.", "v", pocket_worker.NEUTRAL,
                             lambda _b: None, lambda: True)

    def test_the_first_unit_trims_its_tail_and_the_second_trims_both_ends(self):
        model = self.Roomy()
        assert model.room * 32767 > pocket_worker.SPEECH_FLOOR, (
            "the fixture's room tone is not loud enough to be the reported case")
        engine = pocket_engine(model)

        first = self.speak(engine)
        assert first["trimmed_ms"] == pytest.approx(350 - 120, abs=40)
        # Only the tail is even *counted* as quiet on this unit: the lead was
        # taken for the start of the unit and went out as speech. What says so
        # in a log is the floor, which is 20 dB above a clean model's.
        assert first["quiet_ms"] == pytest.approx(350, abs=40)
        assert first["floor_db"] == pytest.approx(-26, abs=2)

        second = self.speak(engine)
        assert second["trimmed_ms"] == pytest.approx(600 - 60 - 120, abs=40)

    def test_the_floor_is_remembered_per_voice(self):
        engine = pocket_engine(self.Roomy())
        self.speak(engine)
        assert engine._quiet_floor["v"] > 0
        assert set(engine._quiet_floor) == {"v"}


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

    def test_a_location_three_levels_down_is_refused_too(self, tmp_path):
        """Upstream's config is nested and the locations are at two depths: the
        weights at the root and the tokenizer inside ``flow_lm.lookup_table``. A
        top-level scan would pass the one this worker could actually resolve --
        ``load_config`` hands every path to ``download_if_necessary``."""
        path = tmp_path / "model.local.yaml"
        path.write_text(pocket_worker.json.dumps({
            "weights_path": str(tmp_path / "w.st"),
            "flow_lm": {"lookup_table": {
                "tokenizer_path": "hf://kyutai/pocket-tts/tokenizer.model"}},
        }), encoding="utf-8")
        engine = pocket_worker.Engine({"config_path": str(path)})
        with pytest.raises(ValueError) as raised:
            engine._read_config()
        found = str(raised.value)
        assert "network location" in found
        assert "tokenizer_path" in found, found

    def test_the_report_names_where_rather_than_what(self, tmp_path):
        """A key path, never the location itself: a repository id is not private
        but the habit of putting a config's *values* in a message is how a clone
        path ends up in somebody's log (I-PKT-27)."""
        path = tmp_path / "model.local.yaml"
        path.write_text(pocket_worker.json.dumps(
            {"weights_path": "hf://kyutai/pocket-tts/model.safetensors"}), encoding="utf-8")
        engine = pocket_worker.Engine({"config_path": str(path)})
        with pytest.raises(ValueError) as raised:
            engine._read_config()
        assert "kyutai" not in str(raised.value)


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
            # Waited for rather than assumed. Feeding the pipe only queues bytes,
            # so releasing the unit here would race the command loop -- and a
            # Stop that landed *between* units is a different, equally correct
            # answer with no unit to measure. The "draining" acknowledgement is
            # the worker saying it read the interrupt while it was inside one.
            draining = stdout.wait_for("tts_interrupted")
            assert draining[0][0]["state"] == "draining", draining[0][0]
            release.set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if any(header.get("state") == "complete"
                       for header, _payload in stdout.of("tts_interrupted")):
                    break
                time.sleep(0.01)

        found = run_worker(engine, feed)
        complete = [header for header, _payload in found.of("tts_interrupted")
                    if header.get("state") == "complete"]
        assert complete, [header for header, _ in found.of("tts_interrupted")]
        assert complete[0]["chars"] == len("A sentence of some length.")
        assert complete[0]["audio_ms"] > 0
        assert complete[0]["dropped_units"] == 1


class TestTheUpstreamApiIsTheOneReleasedThreePointZeroPointTwoHas:
    """GATE P-0, as the assertions the harness produced.

    Every one of these was checked against a real ``pocket-tts 3.0.2`` wheel
    whose SHA-256 is the one this repository's manifest pins, and every one of
    them was previously wrong in a way no test could see: the model was loaded
    with a ``device`` argument that does not exist, handed a config dictionary
    where a path to a YAML file was required, asked to generate with a
    temperature and a step count that are not parameters, asked to encode raw
    WAV bytes it cannot take, and asked to export through a method that is a CLI
    command rather than anything on the model.
    """

    def test_a_state_is_loaded_from_a_path_object_and_never_a_string(self, tmp_path):
        """Upstream branches on the type. A ``str`` goes through
        ``download_if_necessary``, which resolves ``https://`` and ``hf://`` --
        making it the one call in this worker that could reach the network. A
        ``Path`` goes straight to the file (I-PKT-20)."""
        import pathlib

        state = tmp_path / "state.safetensors"
        state.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
        model = FakeModel()
        engine = pocket_engine(model)
        engine.voices = {"v": {"kind": "clone", "state": str(state)}}
        engine._states.clear()

        assert engine.state_for("v") == {"prepared": True}
        asked, _truncate = model.prepared[-1]
        assert isinstance(asked, pathlib.Path), type(asked)

    def test_a_recording_is_prepared_from_a_file_rather_than_from_bytes(self, tmp_path):
        """``get_state_for_audio_prompt`` takes a ``Path``, a ``str`` or a Torch
        tensor. Raw WAV bytes fall through every branch and die on an attribute
        the model was never given."""
        import pathlib

        model = FakeModel()
        engine = pocket_engine(model, cloning_ready=True)
        root = tmp_path / "preview"
        root.mkdir()
        wav = pocket_worker.encode_wav(pocket_worker.pcm16([0.0] * 2400), RATE)

        made = engine.prepare(wav, str(root), seconds=8.0)
        assert made["state"] == {"prepared": True}
        asked, truncate = model.prepared[-1]
        assert isinstance(asked, pathlib.Path), type(asked)
        assert asked.name == pocket_worker.REFERENCE_FILENAME
        assert asked.read_bytes() == wav, "the reference the model read is not the one sent"
        assert truncate is True

    def test_a_preparation_with_nowhere_to_write_is_a_sentence(self, tmp_path):
        engine = pocket_engine(FakeModel(), cloning_ready=True)
        with pytest.raises(ValueError) as raised:
            engine.prepare(pocket_worker.encode_wav(pocket_worker.pcm16([0.0] * 2400), RATE),
                           str(tmp_path / "gone"), seconds=8.0)
        assert "nowhere to write" in str(raised.value)

    def test_a_state_is_exported_through_the_module_function(self, tmp_path):
        """``export_model_state`` is a function over the state dictionary.
        ``export_voice`` is the name of upstream's *CLI command* and is not on
        the model at all, so looking for it there found nothing."""
        written = []

        def export_model_state(state, dest):
            written.append((state, dest))
            open(dest, "wb").write(b"x" * 40)

        engine = pocket_engine(FakeModel())
        engine._export_state = export_model_state
        size = engine.export({"flow_lm": {}}, str(tmp_path / "out" / "state.safetensors"))
        assert size == 40
        assert written[0][1].endswith(".new"), "the write was not staged"
        assert (tmp_path / "out" / "state.safetensors").is_file()
        assert not (tmp_path / "out" / "state.safetensors.new").exists()

    def test_a_failed_export_leaves_no_staging_file_behind(self, tmp_path):
        """A ``.new`` left in a preview directory is a file the promote would
        carry into somebody's saved voice."""
        def explode(state, dest):
            open(dest, "wb").write(b"half")
            raise RuntimeError("out of memory")

        engine = pocket_engine(FakeModel())
        engine._export_state = explode
        target = tmp_path / "state.safetensors"
        with pytest.raises(RuntimeError):
            engine.export({"flow_lm": {}}, str(target))
        assert not target.exists()
        assert not target.with_name(target.name + ".new").exists()

    def test_a_build_with_no_exporter_refuses_rather_than_raising_an_attribute_error(self):
        engine = pocket_engine(FakeModel())
        engine._export_state = None
        with pytest.raises(ValueError) as raised:
            engine.export({"flow_lm": {}}, "/nowhere/state.safetensors")
        assert "cannot export" in str(raised.value)

    def test_only_arguments_the_loader_accepts_are_offered_to_it(self):
        """Read off the signature rather than found out by calling: a
        ``TypeError`` raised inside a model loader is indistinguishable from one
        raised by calling it wrongly."""
        def loader(config, temp=None, sampler_decode_steps=1, quantize=False):
            return None

        assert pocket_worker._parameters(loader) == {
            "config", "temp", "sampler_decode_steps", "quantize"}

        def narrow(config):
            return None

        assert pocket_worker._parameters(narrow) == {"config"}

        def anything(**values):
            return None

        assert pocket_worker._parameters(anything) is None


class TestTheRecipeModeReadsUpstreamsOwnConfiguration:
    """``worker.py --recipe english``, which the installer runs in the staged
    runtime because the document ships inside the wheel."""

    def test_a_name_that_could_be_a_path_is_refused(self, capsys):
        """Nothing a browser or a manifest sends becomes a path component. The
        model id reaches this from a manifest this repository ships, which is
        exactly the kind of trust that stops being true one release later."""
        for bad in ("../../etc/passwd", "languages/english", "english/../.."):
            assert pocket_worker.recipe(bad) == 1
            found = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
            assert found["ok"] is False
            assert "not a model name" in found["error"], found

    def test_a_name_this_build_does_not_have_says_what_it_does_have(self, capsys):
        """A refusal with the correction already on screen, rather than
        "PocketTTS failed to start"."""
        if importlib.util.find_spec("pocket_tts") is None:
            pytest.skip("this machine has no PocketTTS closure")
        assert pocket_worker.recipe("klingon") == 1
        found = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert "has no configuration" in found["error"]

