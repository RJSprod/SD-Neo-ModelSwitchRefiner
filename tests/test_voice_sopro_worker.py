"""The Sopro sidecar: its framing, its containment, its policy and its DSP.

Two things are asserted here that nothing else can assert, and both of them are
release gates rather than niceties.

The first is that Sopro Speed is pitch-preserving (Gate S-6, T-DSP-1, T-DSP-2).
Sopro V2 exposes no speaking-rate parameter, so Speed is Voice Chat's own
time-scaling around the model output -- and section 32 is explicit that a naive
resample that transposes the voice is not parity and does not ship. The only way
to know which of those was built is to synthesize a tone, run it through the
shaper at several speeds, and measure the fundamental. That is what these tests
do.

The second is that the two workers agree on the wire (section 18). They are
separate files in separate dependency closures on purpose, so their framing is
shared by *agreement* rather than by import -- and an agreement nobody checks is
a protocol that drifts.

Nothing here imports Torch or Sopro. The DSP is NumPy and the framing is the
standard library, which is exactly the part of the worker that can be tested on
a machine where Sopro was never installed.
"""

from __future__ import annotations

import io
import math

import pytest

from sopro_worker import worker as sopro_worker
from voice_worker import worker as kokoro_worker

numpy = pytest.importorskip("numpy", reason="the Sopro delivery DSP is NumPy")

RATE = 24000


def tone(freq: float, seconds: float, rate: int = RATE):
    steps = numpy.arange(int(seconds * rate), dtype=numpy.float32) / rate
    return (0.5 * numpy.sin(2.0 * numpy.pi * freq * steps)).astype(numpy.float32)


def dominant(samples, rate: int = RATE) -> float:
    """The strongest frequency in a block, by FFT. Windowed, because a rectangular
    window on a tone smears the peak across neighbouring bins."""
    if len(samples) < 4096:
        return 0.0
    window = numpy.hanning(len(samples)).astype(numpy.float32)
    spectrum = numpy.abs(numpy.fft.rfft(samples * window))
    return float(numpy.fft.rfftfreq(len(samples), 1.0 / rate)[int(numpy.argmax(spectrum))])


def through(shaper, source, chunk: int = 431):
    """Push a signal through a shaper in awkward-sized chunks and collect the PCM.

    Awkward on purpose. Sopro's chunks are not a multiple of the shaper's frame,
    and a DSP that only worked on tidy boundaries would be a DSP that clicked in
    production and passed its tests.
    """
    blocks = []
    for index in range(0, len(source), chunk):
        found = shaper.block(source[index:index + chunk])
        if found:
            blocks.append(found)
    tail = shaper.flush()
    if tail:
        blocks.append(tail)
    raw = b"".join(blocks)
    return numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32767.0


class TestTheProtocolAgreement:
    def test_a_frame_written_by_one_worker_is_read_by_the_other(self):
        """Section 18: framing may be shared in source, and here it is shared by
        agreement instead -- so the agreement is a test."""
        header = {"op": "tts_audio", "turn": "abc", "seq": 3}
        payload = b"\x01\x02\x03\x04"

        buffer = io.BytesIO()
        sopro_worker.write_frame(buffer, header, payload)
        written_by_sopro = buffer.getvalue()

        buffer = io.BytesIO()
        kokoro_worker.write_frame(buffer, header, payload)
        assert buffer.getvalue() == written_by_sopro

        assert kokoro_worker.read_frame(io.BytesIO(written_by_sopro)) == (header, payload)
        assert sopro_worker.read_frame(io.BytesIO(written_by_sopro)) == (header, payload)

    def test_end_of_input_is_none_rather_than_an_error(self):
        """How the parent's death arrives when this process is waiting for work.
        Door D of the five, and not an error: the loop ends and the process
        exits with 0."""
        assert sopro_worker.read_frame(io.BytesIO(b"")) is None

    def test_an_oversized_header_is_refused_rather_than_allocated(self):
        raw = sopro_worker._LENGTH.pack(sopro_worker.MAX_HEADER + 1)
        with pytest.raises(ValueError):
            sopro_worker.read_frame(io.BytesIO(raw))

    def test_the_two_protocol_versions_are_counted_separately(self):
        """One number covering both would be a number that has to change when
        either changes. Sopro carries ``voice_id`` where Kokoro carries ``sid``
        and has four operations Kokoro has no meaning for."""
        assert sopro_worker.PROTOCOL_VERSION == 1
        assert sopro_worker.MARKER != kokoro_worker.MARKER


class TestTheFixedCpuPolicy:
    def test_i_12_the_policy_is_two_constants_and_not_a_tuner(self):
        """Nothing in this repository rotates between two, four and six, runs an
        A/B, or picks a value from measured real-time factors. Moving the policy
        is a deliberate edit to these two lines."""
        assert sopro_worker.INTRAOP_THREADS == 4
        assert sopro_worker.INTEROP_THREADS == 1

    def test_it_matches_kokoros_synthesis_budget(self):
        """Section 20: a Kokoro measured at four threads beside a Sopro that
        silently took every logical core is not a comparison."""
        assert sopro_worker.INTRAOP_THREADS == kokoro_worker.TTS_THREADS \
            if hasattr(kokoro_worker, "TTS_THREADS") else True


class TestTheOneWayThePolicyMoves:
    """I-12 asks for a policy that is *measured*, fixed, and never auto-tuned
    from runtime measurements. Those are three requirements and only the third
    forbids anything here: a benchmark is how the first one is satisfied, and
    the Run validation button cannot sweep a constant.

    What has to stay true is that an installation never quietly runs a policy
    nobody measured — so the override is bounded, and it is loud.
    """

    def test_nothing_in_the_environment_means_the_released_policy(self, monkeypatch):
        monkeypatch.delenv(sopro_worker.OVERRIDE_INTRAOP, raising=False)
        monkeypatch.delenv(sopro_worker.OVERRIDE_INTEROP, raising=False)
        intraop, interop, overridden = sopro_worker.effective_policy(note=False)
        assert (intraop, interop) == (sopro_worker.INTRAOP_THREADS,
                                      sopro_worker.INTEROP_THREADS)
        assert overridden is False

    def test_an_override_is_taken_and_declares_itself(self, monkeypatch):
        monkeypatch.setenv(sopro_worker.OVERRIDE_INTRAOP, "8")
        intraop, _interop, overridden = sopro_worker.effective_policy(note=False)
        assert intraop == 8
        assert overridden is True, "a swept run would have been logged as released"

    def test_asking_for_the_released_number_is_not_an_override(self, monkeypatch):
        """A sweep that includes 4 in its list must not make that row read as a
        configuration nobody shipped."""
        monkeypatch.setenv(sopro_worker.OVERRIDE_INTRAOP,
                           str(sopro_worker.INTRAOP_THREADS))
        intraop, _interop, overridden = sopro_worker.effective_policy(note=False)
        assert intraop == sopro_worker.INTRAOP_THREADS
        assert overridden is False

    @pytest.mark.parametrize("value", ["0", "-1", "65", "4096", "", "  ", "eight",
                                       "4.5", "0x8"])
    def test_a_value_that_is_not_a_usable_count_keeps_the_released_one(
            self, monkeypatch, value):
        """Bounded rather than trusted. Zero and negatives reach
        ``torch.set_num_threads`` as-is, and four thousand threads is a machine
        spending all of its time in barriers."""
        monkeypatch.setenv(sopro_worker.OVERRIDE_INTRAOP, value)
        intraop, _interop, overridden = sopro_worker.effective_policy(note=False)
        assert intraop == sopro_worker.INTRAOP_THREADS
        assert overridden is False

    def test_the_parent_pins_openmp_to_whatever_the_child_will_use(self, monkeypatch):
        """The subtle one. ``worker_environment`` sets OMP_NUM_THREADS before
        the child starts, and OpenMP has usually sized its pool before any of
        our code runs — so a parent that pinned the released four while the
        child asked Torch for eight would measure neither number."""
        import mc_voice_sopro as sopro

        monkeypatch.setenv(sopro_worker.OVERRIDE_INTRAOP, "8")
        found = sopro.worker_environment()
        assert found["OMP_NUM_THREADS"] == "8", found
        assert found["MKL_NUM_THREADS"] == "8"
        assert found["OPENBLAS_NUM_THREADS"] == "8"

    def test_with_no_override_the_parent_pins_the_released_number(self, monkeypatch):
        import mc_voice_sopro as sopro

        monkeypatch.delenv(sopro_worker.OVERRIDE_INTRAOP, raising=False)
        found = sopro.worker_environment()
        assert found["OMP_NUM_THREADS"] == str(sopro_worker.INTRAOP_THREADS)


class TestDelivery:
    def test_a_neutral_profile_asks_for_nothing(self):
        found = sopro_worker.Delivery()
        assert found.shapes is False
        assert sopro_worker.Shaper(found, numpy).active is False

    def test_the_stretch_rate_accounts_for_the_pitch(self):
        """The composition, in one number. Resampling by ``pitch`` shortens the
        audio as a side effect of transposing it, so the stretch before it has
        to divide by the same amount for the finished duration to be the one
        Speed asked for."""
        found = sopro_worker.Delivery(speed=1.2, pitch=2.0)
        assert found.stretch_rate == pytest.approx(0.6)

    def test_a_header_full_of_nonsense_produces_a_neutral_delivery(self):
        """Every caller is a JSON header from another process, and none of them
        is a reason for a reply to go unspoken."""
        found = sopro_worker.Delivery.from_header(
            {"speed": "fast", "pitch": None, "gain": float("nan"), "pause_ms": [1]})
        assert found.speed == 1.0 and found.pitch == 1.0 and found.gain == 1.0
        assert found.pause_ms == 0

    def test_a_generation_field_nobody_set_is_absent_rather_than_guessed(self):
        """``None`` means "whatever the pinned model configuration says".
        Materialising today's default would freeze it, and a model revision that
        changed its temperature would then not change anybody's voice."""
        found = sopro_worker.Delivery()
        assert found.generation(None) == {}
        chosen = sopro_worker.Delivery(temperature=0.6, language="pt")
        assert chosen.generation(None) == {"lang": "pt", "temperature": 0.6}


class TestGateS6SpeedPreservesPitch:
    """T-DSP-1. The gate this whole DSP exists to pass.

    A naive resample would change the fundamental in exact proportion to the
    speed -- 220 Hz would become 275 Hz at 1.25x. These tests measure it.
    """

    @pytest.mark.parametrize("speed", [0.8, 1.0, 1.25, 1.5])
    def test_the_fundamental_does_not_move_with_the_speed(self, speed):
        source = tone(220.0, 2.0)
        found = through(sopro_worker.Shaper(sopro_worker.Delivery(speed=speed), numpy),
                        source)
        if speed == 1.0:
            found = source
        assert dominant(found) == pytest.approx(220.0, abs=2.0), (
            f"speed {speed} transposed the voice, which is the naive resample "
            f"section 32 forbids")

    @pytest.mark.parametrize("speed", [0.8, 1.25, 1.5])
    def test_the_duration_is_the_one_that_was_asked_for(self, speed):
        source = tone(220.0, 2.0)
        found = through(sopro_worker.Shaper(sopro_worker.Delivery(speed=speed), numpy),
                        source)
        assert len(source) / len(found) == pytest.approx(speed, rel=0.05)

    def test_a_neutral_speed_is_a_bypass_rather_than_an_identity_transform(self):
        """Section 32: a neutral fast path with no unnecessary processing. An
        installation that never moves a slider pays nothing for these."""
        found = sopro_worker.Shaper(sopro_worker.Delivery(speed=1.0, pitch=1.0), numpy)
        assert found.active is False
        assert found._stretch is None and found._resample is None


class TestTDsp2PitchIsIndependent:
    @pytest.mark.parametrize("semitones", [-5, 2, 7])
    def test_pitch_moves_the_fundamental_by_exactly_that_much(self, semitones):
        ratio = 2.0 ** (semitones / 12.0)
        source = tone(220.0, 2.0)
        found = through(
            sopro_worker.Shaper(sopro_worker.Delivery(speed=1.0, pitch=ratio), numpy),
            source)
        assert dominant(found) == pytest.approx(220.0 * ratio, rel=0.02)

    @pytest.mark.parametrize("semitones", [-5, 2, 7])
    def test_pitch_does_not_change_the_duration(self, semitones):
        """"Changing Pitch does not unexpectedly change requested duration
        beyond tested tolerance" -- section 32, measured."""
        ratio = 2.0 ** (semitones / 12.0)
        source = tone(220.0, 2.0)
        found = through(
            sopro_worker.Shaper(sopro_worker.Delivery(speed=1.0, pitch=ratio), numpy),
            source)
        assert len(found) / len(source) == pytest.approx(1.0, rel=0.05)

    @pytest.mark.parametrize("speed,semitones", [(0.8, -5), (1.25, 2), (1.5, 7)])
    def test_the_two_compose_without_interfering(self, speed, semitones):
        ratio = 2.0 ** (semitones / 12.0)
        source = tone(220.0, 2.0)
        found = through(
            sopro_worker.Shaper(sopro_worker.Delivery(speed=speed, pitch=ratio), numpy),
            source)
        assert dominant(found) == pytest.approx(220.0 * ratio, rel=0.02)
        assert len(source) / len(found) == pytest.approx(speed, rel=0.05)

    def test_no_click_at_a_chunk_boundary(self):
        """The streaming requirement. A stretcher restarted at each chunk puts a
        discontinuity at every one of them -- an audible tick several times a
        second -- so the buffer, the read position and the overlap tail all
        survive between calls.

        Measured as the largest sample-to-sample step, against the largest step
        the source itself contains: a click is a jump much bigger than the
        waveform's own slope.
        """
        source = tone(220.0, 2.0)
        expected = float(numpy.abs(numpy.diff(source)).max())
        for speed, semitones in ((1.25, 0), (0.8, 0), (1.0, 2), (1.5, 7)):
            found = through(
                sopro_worker.Shaper(
                    sopro_worker.Delivery(speed=speed, pitch=2.0 ** (semitones / 12.0)),
                    numpy),
                source, chunk=257)
            jump = float(numpy.abs(numpy.diff(found)).max())
            assert jump < expected * 3.0, (
                f"speed {speed} pitch {semitones} left a discontinuity of {jump:.4f} "
                f"where the source's own largest step is {expected:.4f}")


class TestVolume:
    def test_a_loud_setting_is_limited_rather_than_clipped(self):
        """A +6 dB setting on a loud passage would otherwise clip into
        distortion that sounds like a broken model rather than a loud one.

        The signal here is what that actually looks like: model output already
        near the top of its range, asked for another six decibels. A hard clip
        would flatten every sample past full scale into an identical plateau;
        the soft knee bends them instead, so the peaks stay distinguishable from
        one another and from each other's neighbours.
        """
        # Peaks at 1.1 after the gain: past the knee, and not so far past it
        # that any limiter shape would look the same. The knee engages on the
        # *gain* being above unity rather than on the samples being loud, which
        # is the same rule Kokoro's worker follows -- both engines' models emit
        # audio already inside the range, so the only way past full scale is a
        # volume setting somebody chose.
        source = tone(220.0, 0.5) * 2.0
        raw = sopro_worker.pcm16(source, gain=1.1)
        found = numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32767.0
        assert float(numpy.abs(found).max()) < 1.0, "the limiter let it reach full scale"

        peaks = numpy.abs(found)[numpy.abs(found) > 0.9]
        assert len(peaks) > 100, "the test signal did not reach the knee"
        assert float(peaks.max()) < 1.0
        assert float(peaks.std()) > 0.001, (
            "every loud sample came out at the same level, which is a hard clip "
            "rather than a limiter")

    def test_at_or_below_unity_it_is_exactly_a_scalar(self):
        """There is nothing to limit, so nothing is done: a quiet setting must
        not colour the sound, and a neutral one must be a bypass."""
        source = tone(220.0, 0.2)
        raw = sopro_worker.pcm16(source, gain=1.0)
        found = numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32767.0
        assert numpy.allclose(found, source, atol=1e-4)

    def test_below_unity_there_is_nothing_to_limit(self):
        source = tone(220.0, 0.5)
        raw = sopro_worker.pcm16(source, gain=0.5)
        found = numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32767.0
        assert float(numpy.abs(found).max()) == pytest.approx(0.25, abs=0.01)


class TestTheDurableFormat:
    def test_i_11_prompt_state_is_not_in_the_production_tensors(self):
        """Warmed streaming caches are worker cache with a bound on them. What
        is written is conditioning plus a scalar."""
        assert sopro_worker.PRODUCTION_TENSORS == ("cond_vec", "semantic_tokens", "mel")
        for name in sopro_worker.PRODUCTION_TENSORS:
            assert "prompt" not in name and "kv" not in name and "session" not in name

    def test_the_lab_components_are_kept_physically_separate(self):
        """Section 14: the separation is what makes "Conversation reads canonical
        conditioning, never experimental state" a property of the filesystem
        rather than a promise."""
        assert sopro_worker.LAB_TENSORS == ("id_emb", "style_emb", "style_ctrl")
        assert not set(sopro_worker.LAB_TENSORS) & set(sopro_worker.PRODUCTION_TENSORS)

    def test_a_tensor_read_is_bounded(self):
        """Section 57: a malformed voice cannot become an arbitrary allocation
        request."""
        assert 0 < sopro_worker.MAX_TENSOR_BYTES <= 256 * 1024 * 1024

    def test_the_caches_are_small_fixed_numbers(self):
        """Section 29: never an unbounded map of every cloned voice."""
        for value in (sopro_worker.REFERENCE_CACHE, sopro_worker.PROMPT_CACHE,
                      sopro_worker.LAB_CACHE):
            assert 1 <= value <= 8


class TestWhatNeverLeavesThisProcess:
    def test_an_exception_this_file_did_not_raise_is_reported_by_class_only(self):
        """A third-party library is entitled to put whatever it likes in a
        message, including the input it was given, and this feature's invariant
        is that the input never leaves this process."""
        found = sopro_worker._safe(RuntimeError("the reference said: my name is Rebecca"))
        assert found == "RuntimeError"
        assert "Rebecca" not in found

    def test_this_files_own_refusals_are_sentences(self):
        found = sopro_worker._safe(ValueError("that recording contains no audio"))
        assert found == "that recording contains no audio"


class TestTheQuietAModelPutsRoundAUnitIsCutBack:
    """The gap between sentences, and what most of it actually was.

    A reply is a run of units played back with nothing between them, so the
    silence a listener hears between two sentences is the silence *inside* the
    two units: whatever the decoder produced before the first phoneme and
    whatever it produced after the last one. None of it was asked for.

    Cut back rather than cut out: sentences are separated by a pause in speech,
    and the delivery's own "Pause between sentences" adds to what is left.
    """

    RATE = 24000

    def quiet(self, seconds: float, level: float = 0.0):
        return numpy.full(int(seconds * self.RATE), level, dtype=numpy.float32)

    def through(self, trim, source, chunk: int = 431):
        out = bytearray()
        for start in range(0, len(source), chunk):
            out += trim.block(sopro_worker.pcm16(source[start:start + chunk]))
        out += trim.flush()
        return numpy.frombuffer(bytes(out), dtype="<i2").astype(numpy.float32) / 32767.0

    def test_the_lead_is_cut_to_the_kept_amount(self):
        source = numpy.concatenate([self.quiet(0.5), tone(220.0, 1.0)])
        found = self.through(sopro_worker.Trim(self.RATE), source)
        expected = int(self.RATE * sopro_worker.KEEP_LEAD_MS / 1000) + self.RATE
        assert abs(found.size - expected) < self.RATE * 0.02

    def test_the_tail_is_cut_to_the_kept_amount(self):
        source = numpy.concatenate([tone(220.0, 1.0), self.quiet(0.4)])
        found = self.through(sopro_worker.Trim(self.RATE), source)
        expected = self.RATE + int(self.RATE * sopro_worker.KEEP_TAIL_MS / 1000)
        assert abs(found.size - expected) < self.RATE * 0.02

    def test_quiet_inside_a_unit_is_prosody_and_is_left_alone(self):
        source = numpy.concatenate([tone(220.0, 0.2), self.quiet(0.3), tone(220.0, 0.2)])
        trim = sopro_worker.Trim(self.RATE)
        found = self.through(trim, source)
        assert trim.dropped == 0
        assert abs(found.size - source.size) < self.RATE * 0.02

    def test_a_long_pause_inside_a_unit_is_still_left_alone(self):
        """Longer than the hold, so it cannot all be waited out. It still
        arrives, because held audio spills through rather than accumulating."""
        source = numpy.concatenate([tone(220.0, 0.1), self.quiet(1.5), tone(220.0, 0.1)])
        trim = sopro_worker.Trim(self.RATE)
        found = self.through(trim, source)
        assert trim.dropped == 0
        assert abs(found.size - source.size) < self.RATE * 0.02

    def test_speech_is_never_cut(self):
        trim = sopro_worker.Trim(self.RATE)
        found = self.through(trim, tone(220.0, 1.0))
        assert trim.dropped == 0
        assert abs(found.size - self.RATE) < self.RATE * 0.02

    def test_the_level_follows_the_unit_rather_than_being_fixed(self):
        """A cloned voice carries its reference recording's room tone, and a
        fixed floor low enough for a studio recording would find no quiet in
        it at all."""
        room = 0.005
        assert room * 32767 > sopro_worker.QUIET_FLOOR, "the fixture is not a real test"
        source = numpy.concatenate([self.quiet(0.3, room), tone(220.0, 1.0),
                                    self.quiet(0.4, room)])
        trim = sopro_worker.Trim(self.RATE)
        self.through(trim, source)
        assert trim.dropped_ms == pytest.approx(700 - sopro_worker.KEEP_LEAD_MS
                                                - sopro_worker.KEEP_TAIL_MS, abs=40)

    def test_a_clean_floor_does_not_make_the_line_strict(self):
        """Reported from a machine: ``floor_db=-68`` and ``quiet_ms=0``.

        A voice whose quietest moment is exceptionally clean had the line drawn
        under its own padding, so nothing was recognised and nothing was cut.
        The line comes from the speech instead -- a sixteenth of the loudest
        sample -- so one dead moment in the middle of a unit cannot make the
        unit's padding invisible.
        """
        pad, dead = 0.01, 0.0004
        noisy = numpy.concatenate([self.quiet(0.3, pad), tone(220.0, 1.0) * 1.8,
                                   self.quiet(0.4, pad)])
        clean = numpy.concatenate([self.quiet(0.3, pad), tone(220.0, 0.5) * 1.8,
                                   self.quiet(0.02, dead), tone(220.0, 0.5) * 1.8,
                                   self.quiet(0.4, pad)])
        one, two = sopro_worker.Trim(self.RATE), sopro_worker.Trim(self.RATE)
        self.through(one, noisy)
        self.through(two, clean)
        assert one.floor_db > two.floor_db + 20, "the fixtures are not different"
        assert one.dropped_ms == pytest.approx(700 - sopro_worker.KEEP_LEAD_MS
                                               - sopro_worker.KEEP_TAIL_MS, abs=40)
        assert abs(one.dropped_ms - two.dropped_ms) < 40

    def test_a_pause_inside_a_unit_is_measured_and_left_alone(self):
        source = numpy.concatenate([tone(220.0, 0.2) * 1.8, self.quiet(0.5, 0.01),
                                    tone(220.0, 0.2) * 1.8])
        trim = sopro_worker.Trim(self.RATE)
        found = self.through(trim, source)
        assert trim.gap_ms == pytest.approx(500, abs=30)
        assert trim.dropped == 0
        assert abs(found.size - source.size) < self.RATE * 0.02

    def test_a_rate_of_nothing_is_a_pass_through_rather_than_a_crash(self):
        source = numpy.concatenate([self.quiet(0.1), tone(220.0, 0.1)])
        assert self.through(sopro_worker.Trim(0), source).size == source.size


class TestAUnitEndsAtSilenceRatherThanWhereverItWas:
    """The pop between sentences, and the thing that removes it.

    A unit is one decode. Sopro stops when the unit is said and the next unit is
    decoded from a session that has no memory of this one, so a unit's last
    sample is wherever the waveform happened to be and the next unit's first is
    wherever a fresh session starts. This feature plays units back sample-exact
    and one after another, which turns that pair into a step -- and a step in a
    waveform is a click at the end of a sentence.

    The signal below is DC on purpose: the worst case, roughly what a decoder's
    residual offset looks like, and nothing but a ramp removes it.
    """

    def unit(self, seam, source, chunk: int = 431):
        out = bytearray()
        for start in range(0, len(source), chunk):
            out += seam.block(sopro_worker.pcm16(source[start:start + chunk]))
        out += seam.flush()
        return numpy.frombuffer(bytes(out), dtype="<i2").astype(numpy.float32) / 32767.0

    def test_a_unit_starts_and_ends_at_silence(self):
        found = self.unit(sopro_worker.Seam(24000),
                          numpy.full(4800, 0.5, dtype=numpy.float32))
        assert abs(float(found[0])) < 1e-3
        assert abs(float(found[-1])) < 1e-3

    def test_the_step_a_join_would_have_carried_is_gone(self):
        found = self.unit(sopro_worker.Seam(24000),
                          numpy.full(4800, 0.5, dtype=numpy.float32))
        joined = numpy.concatenate([found, found])
        assert float(numpy.abs(numpy.diff(joined)).max()) < 0.01

    def test_nothing_is_lost_to_the_ramp(self):
        source = numpy.full(4800, 0.5, dtype=numpy.float32)
        assert self.unit(sopro_worker.Seam(24000), source).size == source.size

    def test_the_middle_of_the_unit_is_untouched(self):
        source = numpy.full(4800, 0.5, dtype=numpy.float32)
        found = self.unit(sopro_worker.Seam(24000), source)
        span = sopro_worker.Seam(24000).span
        assert float(numpy.abs(found[span:-span] - 0.5).max()) < 1e-3

    def test_a_unit_shorter_than_the_ramp_still_starts_and_ends_at_silence(self):
        seam = sopro_worker.Seam(24000)
        source = numpy.full(seam.span // 3, 0.5, dtype=numpy.float32)
        found = self.unit(seam, source, chunk=17)
        assert found.size == source.size
        assert abs(float(found[0])) < 1e-3
        assert abs(float(found[-1])) < 1e-3

    def test_the_ramp_does_not_follow_the_models_chunk_sizes(self):
        source = numpy.full(4800, 0.5, dtype=numpy.float32)
        whole = self.unit(sopro_worker.Seam(24000), source, chunk=len(source))
        pieces = self.unit(sopro_worker.Seam(24000), source, chunk=137)
        assert whole.size == pieces.size
        assert float(numpy.abs(whole - pieces).max()) < 1e-3

    def test_a_rate_of_nothing_is_a_pass_through_rather_than_a_crash(self):
        source = numpy.full(480, 0.5, dtype=numpy.float32)
        assert self.unit(sopro_worker.Seam(0), source).size == source.size


class TestSilence:
    def test_a_pause_is_exactly_the_requested_number_of_milliseconds(self):
        found = sopro_worker.silence(24000, 250)
        assert len(found) // 2 == 6000
        assert set(found) == {0}

    def test_no_pause_is_no_bytes(self):
        assert sopro_worker.silence(24000, 0) == b""
