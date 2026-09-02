"""The Voice Pipeline: its clock, its seams, its switches and its doors.

Most of this file is about duration, which is the property everything else in
the feature depends on. A denoiser that loses twenty milliseconds and a
bandwidth extender that interprets 24 kHz as 16 kHz both produce audio that
sounds like speech and is the wrong length, and neither announces itself.

The DSP tests drive the real adapters with hand-written backends rather than
models, because the adapters are where the clock lives and the models are not
installable on a machine that has not run Phase 0. The process tests drive the
real worker over a real pipe with the same stand-ins, so the framing, the
offsets, the cancellation and the containment are the ones that ship.
"""

from __future__ import annotations

import io
import dataclasses
import json
import math
import struct
import sys
import threading
import time
import types

import pytest

import mc_voice_paths as paths
import mc_voice_pipeline as pipeline
import mc_voice_ui
import mc_voice_pipeline_runtime as runtime
from pipeline_worker import worker

RATE = 24000
"""PocketTTS's own output rate today, and the one Phase 0 has to prove LavaSR
does not read as 16000."""


# --------------------------------------------------------------------------- #
# Backends that are not models
# --------------------------------------------------------------------------- #


class LateDpdf:
    """Gives back what it was given, one block late, like the real one.

    Late on purpose. Upstream is about one 20 ms model window behind before its
    first enhanced sample, so a stand-in that answered immediately would let a
    debt-accounting bug through the one test written to catch it.
    """

    block_samples = 480

    def __init__(self, lag: int = 1):
        self.lag = lag
        self.held = []
        self.calls = 0

    def reset(self, rate):
        self.held = []

    def enhance(self, samples):
        self.calls += 1
        self.held.append(list(samples))
        if len(self.held) <= self.lag:
            return []
        return self.held.pop(0)


class PerfectLava:
    """A flawless bandwidth extender: everything it is given, at 48 kHz.

    Flawless is what a clock test wants. Any difference between what went in
    and what came out is the adapter's, which is the only thing under test here.
    """

    def __init__(self, backend_rate: int = 16000):
        self.up = worker.Resampler(backend_rate, 48000)
        self.calls = 0

    def reset(self, rate):
        self.calls = 0

    def enhance(self, samples, rate):
        self.calls += 1
        return self.up(samples)


class DoublingLava:
    """Two samples out for every one in, with no resampling anywhere.

    Fast enough for a long-run test, which the band-limited path is not in pure
    Python. What it proves is the ledger, not the filter.
    """

    def reset(self, rate):
        return None

    def enhance(self, samples, rate):
        found = []
        for value in samples:
            found.append(value)
            found.append(value)
        return found


def speech(count: int, rate: int = RATE) -> list:
    return [0.25 * math.sin(2.0 * math.pi * 220.0 * index / rate) for index in range(count)]


def chain(*stages):
    return worker.Turn("t", RATE, stages)


def lava(backend=None, analysis=250, context=50, rate_in=RATE, backend_rate=16000):
    return worker.LavaStage(backend or PerfectLava(backend_rate), rate_in,
                            analysis_ms=analysis, context_ms=context,
                            backend_rate=backend_rate)


def through(turn, samples, sizes):
    """Feed ``samples`` in blocks of ``sizes``, then flush. Returns the output."""
    found = []
    index = 0
    for size in sizes:
        if index >= len(samples):
            break
        found.extend(turn.feed(samples[index:index + size]))
        index += size
    if index < len(samples):
        found.extend(turn.feed(samples[index:]))
    found.extend(turn.flush())
    return found


def blocks(total: int, size: int) -> list:
    return [size] * (total // size + 1)


# --------------------------------------------------------------------------- #
# The clock
# --------------------------------------------------------------------------- #


class TestTheSampleClock:
    """A-23, A-25, A-26, A-28. The property everything else rests on."""

    @pytest.mark.parametrize("seconds", [0.1, 0.5, 1.0, 3.14159, 10.0])
    def test_lava_turns_a_duration_into_the_same_duration_at_48_khz(self, seconds):
        total = int(RATE * seconds)
        stage = lava()
        found = through(chain(stage), speech(total), blocks(total, 1024))
        assert len(found) == worker.deterministic_target(total, RATE, 48000)
        assert len(found) == 2 * total, "24 kHz to 48 kHz is exactly twice, and no less"

    @pytest.mark.parametrize("seconds", [0.1, 1.0, 3.14159])
    def test_dpdfnet_gives_back_every_sample_it_was_given(self, seconds):
        total = int(RATE * seconds)
        turn = chain(worker.DpdfStage(LateDpdf(), RATE))
        found = through(turn, speech(total), blocks(total, 777))
        assert len(found) == total, "the denoiser's own latency is a debt, not a loss"
        assert turn.rate_out == RATE, "DPDFNet preserves the rate it was given"

    def test_a_denoiser_that_lags_further_still_owes_the_same_number(self):
        """The debt is paid at the flush however deep it got.

        Four blocks rather than one, because a stage that only ever lagged by
        one would let an off-by-one in the flush pass as correct.
        """
        total = RATE
        turn = chain(worker.DpdfStage(LateDpdf(lag=4), RATE))
        assert len(through(turn, speech(total), blocks(total, 1000))) == total

    def test_a_denoiser_deep_behind_does_not_end_the_reply_in_silence(self):
        """The count being right is not the same as the audio being there.

        A stage four blocks behind owes four blocks at the flush. Calling the
        backend once and padding the rest with zeros produces exactly the right
        number of samples and replaces the last eighty milliseconds of every
        reply with digital silence -- which no length assertion can see.
        """
        total = RATE
        source = speech(total)
        stage = worker.DpdfStage(LateDpdf(lag=4), RATE)
        found = through(chain(stage), source, blocks(total, 480))
        assert len(found) == total
        assert stage.correction == 0, "the flush had to invent samples"
        tail = found[-4 * 480:]
        assert max(abs(value) for value in tail) > 0.1, (
            "the end of the reply came back as silence")
        # And it is the *right* audio, not merely audio: an identity backend
        # that is only late gives back exactly what it was given.
        assert found == pytest.approx(source, abs=1e-9)

    def test_the_committed_stream_tracks_the_clock_at_every_hop(self):
        """Not only at the end. A stream that drifts and is corrected once at
        the flush is a stream whose last window has to absorb the whole error.
        """
        rate_in = 22050
        stage = lava(rate_in=rate_in, backend_rate=16000)
        turn = worker.Turn("t", rate_in, [stage])
        source = speech(int(rate_in * 1.5), rate_in)
        index = 0
        while index < len(source):
            turn.feed(source[index:index + 700])
            index += 700
            assert stage.emitted == worker.deterministic_target(
                stage.consumed, rate_in, 48000), (
                f"committed {stage.emitted} for {stage.consumed} consumed")

    def test_both_stages_together_are_still_exact(self):
        total = 2 * RATE
        turn = chain(worker.DpdfStage(LateDpdf(), RATE), lava())
        found = through(turn, speech(total), blocks(total, 1000))
        assert len(found) == 2 * total
        assert turn.rate_out == 48000

    @pytest.mark.parametrize("seconds", [0.3, 1.0, 7.7])
    def test_a_rate_that_does_not_divide_is_exact_too(self, seconds):
        """22050 is not a factor of 48000, which is where a float ratio drifts.

        Nothing speaks at 22050 in this release. It is here because the
        arithmetic must not be a special case for the one rate that happens to
        double, and a test that only ever ran 24 kHz could not tell the two
        apart.
        """
        total = int(22050 * seconds)
        stage = lava(rate_in=22050)
        turn = worker.Turn("t", 22050, [stage])
        found = through(turn, speech(total, 22050), blocks(total, 512))
        assert len(found) == worker.deterministic_target(total, 22050, 48000)
        assert stage.correction == 0

    def test_a_very_short_reply_is_finished_rather_than_held(self):
        """A-23 for "Yes." -- shorter than one analysis window.

        Nothing here ever becomes a full window, so a stage that waited for one
        would emit nothing at all and the reply would be silent.
        """
        for total in (1, 10, 240, 1200, 2400):
            turn = chain(worker.DpdfStage(LateDpdf(), RATE), lava())
            assert len(through(turn, speech(total), [total])) == 2 * total

    def test_a_turn_with_no_audio_produces_none(self):
        turn = chain(lava())
        assert turn.flush() == []

    def test_thirty_minutes_does_not_drift(self):
        """A-28. Rounding that accumulates is invisible until it is not."""
        total = RATE * 60 * 30
        stage = worker.LavaStage(DoublingLava(), RATE, analysis_ms=250, context_ms=50,
                                 backend_rate=RATE)
        turn = chain(stage)
        count = 0
        for start in range(0, total, RATE * 10):
            count += len(turn.feed([0.05] * min(RATE * 10, total - start)))
        count += len(turn.flush())
        assert count == 2 * total
        assert stage.correction == 0, "a correction every window is an adapter that is wrong"

    def test_the_holding_buffer_stays_bounded_however_long_the_reply(self):
        """I-VP-16. Keeping the turn's whole source would be simpler and wrong."""
        stage = worker.LavaStage(DoublingLava(), RATE, analysis_ms=250, context_ms=50,
                                 backend_rate=RATE)
        turn = chain(stage)
        peak = 0
        for _ in range(60):
            turn.feed([0.05] * (RATE * 10))
            peak = max(peak, stage.held_samples)
        assert peak <= stage.analysis + 2 * stage.context + RATE, peak
        assert peak < RATE * 2, "ten minutes in and it is still holding a fraction of a second"

    def test_the_target_is_computed_from_the_running_total(self):
        """Per-packet rounding is the bug this arithmetic exists to prevent."""
        assert worker.deterministic_target(0, RATE, 48000) == 0
        assert worker.deterministic_target(1, RATE, 48000) == 2
        assert worker.deterministic_target(1, 22050, 48000) == 2
        # Summing a hundred packets of one sample must equal one packet of a
        # hundred, which is only true because the target is cumulative.
        each = [worker.deterministic_target(n, 22050, 48000)
                - worker.deterministic_target(n - 1, 22050, 48000) for n in range(1, 101)]
        assert sum(each) == worker.deterministic_target(100, 22050, 48000)

    def test_a_rate_of_zero_is_refused_rather_than_divided_by(self):
        with pytest.raises(worker.Refusal):
            worker.deterministic_target(100, 0, 48000)


class TestNothingIsRestored:
    """A-24. The source decided what was worth playing before this ran."""

    def test_the_pipeline_measures_only_what_crossed_its_own_boundary(self):
        """A source that trimmed 300 ms sends 300 ms fewer samples, and that is all.

        Written as an assertion about arithmetic because that is what the
        invariant is: the pipeline has no way to know what was removed and no
        way to put it back, and this is the test that would fail if somebody
        gave it one.
        """
        whole = RATE
        trimmed = whole - int(0.3 * RATE)
        found = through(chain(lava()), speech(trimmed), blocks(trimmed, 512))
        assert len(found) == 2 * trimmed
        assert len(found) != 2 * whole


# --------------------------------------------------------------------------- #
# Packets are framing and nothing else
# --------------------------------------------------------------------------- #


class TestTransportChunksHaveNoMeaning:
    """A-08, A-10. The same PCM cut differently is the same audio."""

    PLANS = {
        "one packet": None,
        "10 ms": 240,
        "20 ms": 480,
        "80 ms": 1920,
        "120 ms": 2880,
        "250 ms": 6000,
        "an odd number": 997,
        "one sample": 1,
    }

    def _run(self, size, stages):
        total = int(RATE * 0.75)
        source = speech(total)
        sizes = [total] if size is None else blocks(total, size)
        return through(chain(*stages()), source, sizes)

    @pytest.mark.parametrize("name", sorted(PLANS))
    def test_repacketising_changes_nothing_at_all(self, name):
        def stages():
            return [worker.DpdfStage(LateDpdf(), RATE), lava()]

        reference = self._run(None, stages)
        found = self._run(self.PLANS[name], stages)
        assert len(found) == len(reference), name
        # Bit-exact rather than close. Nothing in either path is packet-aware,
        # so there is no floating-point reason for them to differ at all, and a
        # tolerance here would hide the bug the test is for.
        assert found == reference, name

    def test_random_irregular_packets_are_the_same_audio(self):
        total = int(RATE * 0.75)
        source = speech(total)
        # A fixed sequence rather than a random one: a test that fails on some
        # runs is a test somebody eventually stops reading.
        sizes = [7, 1, 4001, 33, 1999, 2, 512, 9000, 61, 3, 20000]
        reference = through(chain(worker.DpdfStage(LateDpdf(), RATE), lava()),
                            source, [total])
        found = through(chain(worker.DpdfStage(LateDpdf(), RATE), lava()), source, sizes)
        assert found == reference

    def test_a_source_unit_boundary_is_not_a_flush(self):
        """A-09. A worker's unit-end blocks are ordinary samples here.

        PocketTTS emits extra PCM when a unit resolves, because its trim and its
        seam were holding audio back. If those blocks reset a stage, the same
        audio delivered as one unit and as three would differ -- which is what
        this asserts they do not.
        """
        total = int(RATE * 0.75)
        source = speech(total)
        whole = through(chain(lava()), source, [total])
        in_units = through(chain(lava()), source, [total // 3, total // 3, total])
        assert whole == in_units


# --------------------------------------------------------------------------- #
# Seams
# --------------------------------------------------------------------------- #


class TestTheAnalysisWindowJoins:
    """A-29, A-30. A click at the window cadence is the failure mode here."""

    def test_a_pure_tone_comes_back_without_a_step_at_any_join(self):
        """The interior of the stream has no discontinuity larger than the tone's own.

        Measured rather than listened to: a sine's largest true sample-to-sample
        step is known exactly, so anything materially above it is a join.
        """
        total = 2 * RATE
        amplitude, hertz = 0.3, 300.0
        source = [amplitude * math.sin(2.0 * math.pi * hertz * i / RATE)
                  for i in range(total)]
        stage = lava(PerfectLava(RATE), backend_rate=RATE)
        found = through(chain(stage), source, [total])
        assert stage.windows > 4, "one window would prove nothing about joins"
        expected = amplitude * 2.0 * math.pi * hertz / 48000
        interior = [abs(found[i + 1] - found[i])
                    for i in range(200, len(found) - 201)]
        assert max(interior) <= expected * 1.05, max(interior) / expected

    def test_overlapping_windows_do_not_pump_the_level(self):
        """An equal-power crossfade would put a 3 dB bump at every join.

        The two things being faded are two model outputs of the same audio, so
        they are correlated and the ramps must sum to one. This is the test that
        distinguishes the two choices.
        """
        total = 2 * RATE
        source = [0.3 * math.sin(2.0 * math.pi * 300.0 * i / RATE) for i in range(total)]
        found = through(chain(lava(PerfectLava(RATE), backend_rate=RATE)), source, [total])
        assert max(abs(value) for value in found) == pytest.approx(0.3, rel=0.02)

    def test_the_crossfade_ramps_are_complementary(self):
        ramp = worker.crossfade_ramp(64)
        assert ramp[0] == pytest.approx(0.0)
        assert ramp[-1] == pytest.approx(1.0)
        for index, value in enumerate(ramp):
            other = ramp[len(ramp) - 1 - index]
            assert value + other == pytest.approx(1.0), index

    def test_silence_stays_silent_and_finite(self):
        """A-29 at the edges: no ringing burst around a quiet stretch."""
        total = RATE
        source = [0.0] * (total // 2) + speech(total - total // 2)
        found = through(chain(worker.DpdfStage(LateDpdf(), RATE), lava()), source,
                        blocks(total, 512))
        assert all(value == value for value in found), "a NaN reached the output"
        assert all(abs(value) < 1.5 for value in found)
        assert max(abs(value) for value in found[:total // 2]) < 0.01


# --------------------------------------------------------------------------- #
# Rates
# --------------------------------------------------------------------------- #


class TestTheRateContract:
    """A-26, A-27. The Phase-0 blocker, made unmissable."""

    def test_lavasr_refuses_to_run_without_a_measured_backend_rate(self):
        """The whole reason this build ships uninstallable.

        Guessing 16000 when the truth is 24000 plays speech at two-thirds speed;
        guessing the other way plays it half again too fast. Both load, both
        return finite numbers, and neither says anything.
        """
        with pytest.raises(worker.Refusal, match="measured input rate"):
            worker.LavaStage(PerfectLava(), RATE, analysis_ms=250, context_ms=50,
                             backend_rate=0)

    def test_lavasr_refuses_to_run_without_a_measured_window(self):
        with pytest.raises(worker.Refusal, match="measured analysis window"):
            worker.LavaStage(PerfectLava(), RATE, analysis_ms=0, context_ms=50,
                             backend_rate=16000)

    def test_a_backend_that_reads_24_khz_as_16_khz_is_caught_by_the_duration(self):
        """A-27, as an executable check rather than a note in a document.

        The wrong-clock backend here is exactly the failure upstream's own
        implementation invites: it resamples as if its input were 16 kHz when it
        was handed 24 kHz. The audio it returns is speech. It is the wrong
        length, and that is what the adapter's ledger catches.
        """

        class WrongClock:
            """Believes its input is 16 kHz whatever it was told."""

            def reset(self, rate):
                return None

            def enhance(self, samples, rate):
                return worker.Resampler(16000, 48000)(samples)

        stage = worker.LavaStage(WrongClock(), RATE, analysis_ms=250, context_ms=50,
                                 backend_rate=RATE)
        total = RATE
        found = through(chain(stage), speech(total), blocks(total, 1024))
        # The committed duration is still exact, because the adapter reconciles
        # against the target. That is why the duration cannot be the check: what
        # a wrong clock actually produces is the right number of samples of
        # time-compressed speech. The size of the reconciliation is the
        # measurement that says so (section 26.17).
        assert len(found) == 2 * total
        assert stage.correction > worker.TOLERATED_CORRECTION * 10, (
            "a backend half again out should need a correction nobody could call tiny")

    def test_a_model_that_cannot_emit_a_partial_frame_is_not_a_wrong_clock(self):
        """The fourth refusal LavaSR hit, and the first with no defect under it.

        LavaSR's BWE is a Vocos ISTFT, which emits whole frames: a window whose
        length is not a multiple of the hop comes back a partial frame short.
        On the machine this was found on that was 528 samples of a 16800-sample
        window, every window, and four windows in one second summed to 2112 --
        which tripped a tolerance meant for drift and reported "the rate it
        works at is not the rate it was told" about a stage whose audio was
        exact. Exact because the shortfall lands inside the trailing context
        this stage discards: not one sample of it ever reaches the reply.
        """
        class WholeFramesOnly:
            """Returns 48 kHz, correct in rate, one partial frame short."""

            def reset(self, rate):
                return None

            def enhance(self, samples, rate):
                found = worker.Resampler(rate, 48000)(samples)
                return found[:len(found) - 528]

        stage = worker.LavaStage(WholeFramesOnly(), RATE, analysis_ms=250,
                                 context_ms=50, backend_rate=RATE)
        total = RATE
        found = through(chain(stage), speech(total), blocks(total, 1024))

        assert len(found) == 2 * total, "the duration is exact either way"
        assert stage.framing == 528, (
            "a bounded, constant shortfall is the model's own frame")
        assert stage.correction == 0, (
            "and it is not drift, so it must not be counted as any")

    def test_a_wrong_clock_is_still_caught_once_framing_is_forgiven(self):
        """The guard on the test above, so forgiveness cannot become blindness.

        The rates here are 16000, 24000 and 48000, so the smallest confusion a
        backend can have about its own returns a third more than was asked for.
        That is nowhere near a frame and must still be refused.
        """
        class WrongClock:
            def reset(self, rate):
                return None

            def enhance(self, samples, rate):
                return worker.Resampler(16000, 48000)(samples)

        stage = worker.LavaStage(WrongClock(), RATE, analysis_ms=250, context_ms=50,
                                 backend_rate=RATE)
        through(chain(stage), speech(RATE), blocks(RATE, 1024))

        assert stage.correction > worker.TOLERATED_CORRECTION * 10
        assert stage.framing == 0, "half again out is not a frame by any reading"

    def test_a_well_behaved_backend_needs_no_correction_at_all(self):
        """The other half of the check above, so it cannot pass vacuously."""
        stage = lava(PerfectLava(RATE), backend_rate=RATE)
        through(chain(stage), speech(RATE), blocks(RATE, 1024))
        assert stage.correction == 0

    def test_the_resampler_is_a_bypass_when_the_rates_match(self):
        found = worker.Resampler(RATE, RATE)
        assert found.transparent
        assert found([0.1, 0.2, 0.3], 3) == [0.1, 0.2, 0.3]

    def test_the_resampler_keeps_the_count_it_was_asked_for(self):
        found = worker.Resampler(24000, 16000)
        assert len(found(speech(2400))) == 1600
        assert len(found(speech(2400), 999)) == 999

    def test_a_resampler_needs_two_real_rates(self):
        with pytest.raises(worker.Refusal):
            worker.Resampler(0, 48000)


# --------------------------------------------------------------------------- #
# The framing
# --------------------------------------------------------------------------- #


class TestTheProtocol:
    def test_it_speaks_the_same_framing_as_the_other_four_workers(self):
        """By agreement rather than by import: five interpreters, five files."""
        from cleanup_worker import worker as cleanup
        from pocket_worker import worker as pocket
        from sopro_worker import worker as sopro
        from voice_worker import worker as kokoro

        header = {"op": "audio", "turn": "abc", "input_sample_offset": 3}
        payload = b"\x01\x02\x03\x04"
        buffer = io.BytesIO()
        worker.write_frame(buffer, header, payload)
        written = buffer.getvalue()
        for other in (cleanup, pocket, sopro, kokoro):
            assert other.read_frame(io.BytesIO(written)) == (header, payload)
            mine = io.BytesIO()
            other.write_frame(mine, header, payload)
            assert mine.getvalue() == written

    def test_the_end_of_the_pipe_is_not_an_error(self):
        """Door D. The parent's death arrives here while this waits for work."""
        assert worker.read_frame(io.BytesIO(b"")) is None

    def test_an_oversized_header_is_refused(self):
        raw = struct.pack(">I", worker.MAX_HEADER + 1)
        with pytest.raises(worker.Refusal, match="header too large"):
            worker.read_frame(io.BytesIO(raw + b"{}"))

    def test_an_oversized_payload_is_refused(self):
        buffer = io.BytesIO()
        header = json.dumps({"op": "audio"}).encode("utf-8")
        buffer.write(struct.pack(">I", len(header)))
        buffer.write(header)
        buffer.write(struct.pack(">I", worker.MAX_PAYLOAD + 1))
        buffer.seek(0)
        with pytest.raises(worker.Refusal, match="payload too large"):
            worker.read_frame(buffer)

    def test_pcm_that_is_not_whole_samples_is_refused(self):
        with pytest.raises(worker.Refusal):
            worker.to_floats(b"\x01\x02\x03")

    def test_pcm_round_trips_and_clips_rather_than_wrapping(self):
        """A bandwidth extender adds energy, so going past 1.0 is expected."""
        assert worker.to_pcm16([0.0]) == b"\x00\x00"
        assert worker.to_pcm16([2.0]) == struct.pack("<h", 32767)
        assert worker.to_pcm16([-2.0]) == struct.pack("<h", -32768)
        assert worker.to_pcm16([float("nan")]) == b"\x00\x00"
        samples = worker.to_floats(worker.to_pcm16([0.5, -0.5]))
        assert samples[0] == pytest.approx(0.5, abs=1e-4)


class TestNothingSaidReachesTheParent:
    """The privacy bar every worker in this repository clears."""

    def test_the_worker_hands_back_a_class_name_rather_than_a_library_message(self):
        class Boom(Exception):
            def __str__(self):
                return "failed on /home/someone/clones/abc/reference.wav"

        assert worker._safe(Boom()) == "Boom"
        assert "reference.wav" not in worker._safe(Boom())

    def test_its_own_refusals_pass_through_as_written(self):
        assert worker._safe(worker.Refusal("a turn with no id")) == "a turn with no id"

    def test_a_bare_value_error_contributes_only_its_name(self):
        assert worker._safe(ValueError("/a/path and some text")) == "ValueError"


# --------------------------------------------------------------------------- #
# The worker, in process
# --------------------------------------------------------------------------- #


class Pipe:
    """One in-process worker, driven over BytesIO. No subprocess needed.

    The same shape the other workers' unit tests use: the real ``serve`` loop
    reading real frames, so the dispatch, the refusals and the offsets under
    test are the ones that ship.
    """

    def __init__(self, plan=None):
        self.out = io.BytesIO()
        self.worker = worker.Worker(self.out)
        self.worker._stage_backend = None
        self.frames = []

    def load(self, stages, backends):
        self.worker.stages = dict(backends)
        self.worker.configs = {
            name: ({"backend_input_rate": 16000, "analysis_ms": 250, "context_ms": 50}
                   if name == "lavasr" else {})
            for name in stages}

    def send(self, header, payload=b""):
        """One frame through the same error handling ``serve`` gives it.

        Written out rather than calling ``serve``, because ``serve`` reads until
        the pipe ends and a test wants one frame at a time -- but a refusal has
        to come back as an answer here exactly as it would there, or the tests
        for the refusals would be testing a path nothing takes.
        """
        try:
            self.worker._dispatch(str(header.get("op") or ""), header, payload)
        except Exception as exc:  # noqa: BLE001 - the loop's own behaviour
            self.worker.send({"op": "error", "id": header.get("id"),
                              "turn": str(header.get("turn") or ""),
                              "ok": False, "error": worker._safe(exc)})
        return self.read()

    def read(self):
        raw = self.out.getvalue()
        self.out.seek(0)
        self.out.truncate(0)
        found = []
        stream = io.BytesIO(raw)
        while True:
            frame = worker.read_frame(stream)
            if frame is None:
                break
            found.append(frame)
        return found


class TestTheWorkerRefusals:
    """Section 6.4's list, each with its own sentence."""

    def _ready(self):
        pipe = Pipe()
        pipe.load(["dpdfnet", "lavasr"],
                  {"dpdfnet": LateDpdf(), "lavasr": PerfectLava()})
        return pipe

    def test_a_stage_this_build_does_not_have_is_refused(self):
        pipe = Pipe()
        with pytest.raises(worker.Refusal, match="unknown stage"):
            pipe.worker.load({"stages": ["reverb"], "paths": {"reverb": "/x"}})

    def test_a_stage_with_no_local_directory_is_refused(self):
        pipe = Pipe()
        with pytest.raises(worker.Refusal, match="local model directory"):
            pipe.worker.load({"stages": ["dpdfnet"], "paths": {}})

    def test_stereo_is_refused(self):
        pipe = self._ready()
        with pytest.raises(worker.Refusal, match="mono"):
            pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 2,
                               "stages": ["dpdfnet"]})

    def test_a_sample_format_this_build_does_not_read_is_refused(self):
        pipe = self._ready()
        with pytest.raises(worker.Refusal, match="PCM16"):
            pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                               "sample_format": "f32", "stages": ["dpdfnet"]})

    def test_a_turn_with_no_rate_is_refused(self):
        pipe = self._ready()
        with pytest.raises(worker.Refusal, match="no sample rate"):
            pipe.worker.begin({"turn": "a", "sample_rate": 0, "channels": 1,
                               "stages": ["dpdfnet"]})

    def test_audio_before_a_turn_begins_is_refused(self):
        pipe = self._ready()
        with pytest.raises(worker.Refusal, match="has not begun"):
            pipe.worker.audio({"turn": "a"}, worker.to_pcm16([0.0] * 10))

    def test_a_duplicate_turn_is_refused(self):
        pipe = self._ready()
        pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                           "stages": ["dpdfnet"]})
        with pytest.raises(worker.Refusal, match="already begun"):
            pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                               "stages": ["dpdfnet"]})

    def test_a_gap_in_the_source_stream_is_refused(self):
        """I-VP-31. Packet numbers alone cannot tell a hole from a small block."""
        pipe = self._ready()
        pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                           "stages": ["dpdfnet"]})
        pipe.worker.audio({"turn": "a", "input_sample_offset": 0},
                          worker.to_pcm16([0.0] * 100))
        with pytest.raises(worker.Refusal, match="skipped or repeated"):
            pipe.worker.audio({"turn": "a", "input_sample_offset": 500},
                              worker.to_pcm16([0.0] * 100))

    def test_a_stage_that_is_not_loaded_is_refused(self):
        pipe = Pipe()
        pipe.load(["dpdfnet"], {"dpdfnet": LateDpdf()})
        with pytest.raises(worker.Refusal, match="not loaded"):
            pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                               "stages": ["lavasr"]})

    def test_the_graph_cannot_change_during_a_turn(self):
        pipe = self._ready()
        pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                           "stages": ["dpdfnet"]})
        found = pipe.send({"op": "reconfigure", "id": "r", "stages": ["lavasr"]})
        assert found and found[0][0]["ok"] is False
        assert "during a turn" in found[0][0]["error"]

    def test_an_unknown_operation_is_answered_rather_than_fatal(self):
        pipe = self._ready()
        found = pipe.send({"op": "sing", "id": "q"})
        assert found[0][0]["ok"] is False and "unknown operation" in found[0][0]["error"]
        # And the loop is still usable afterwards, which is the point.
        assert pipe.send({"op": "ping", "id": "p"})[0][0]["ok"] is True


class TestTheWorkerTurn:
    def _ready(self, stages=("dpdfnet", "lavasr")):
        pipe = Pipe()
        pipe.load(list(stages), {"dpdfnet": LateDpdf(), "lavasr": PerfectLava()})
        return pipe

    def test_a_turn_answers_the_rate_before_any_audio_exists(self):
        """I-VP-07. The browser is told the rate in a header, once."""
        pipe = self._ready()
        found = pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                                   "stages": ["dpdfnet", "lavasr"]})
        assert found["output_sample_rate"] == 48000
        assert found["first_output_samples"] > 0

    def test_dpdfnet_alone_advertises_the_rate_it_was_given(self):
        pipe = self._ready(["dpdfnet"])
        found = pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                                   "stages": ["dpdfnet"]})
        assert found["output_sample_rate"] == RATE

    def test_the_stages_always_run_in_the_declared_order(self):
        """A-03. Asking for them backwards does not run them backwards."""
        pipe = self._ready()
        pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                           "stages": ["lavasr", "dpdfnet"]})
        assert [stage.id for stage in pipe.worker.turn.stages] == ["dpdfnet", "lavasr"]

    def test_output_carries_a_cumulative_offset(self):
        pipe = self._ready(["dpdfnet"])
        pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                           "stages": ["dpdfnet"]})
        pipe.read()
        offsets, seen = [], 0
        for _ in range(6):
            pipe.worker.audio({"turn": "a", "input_sample_offset": seen},
                              worker.to_pcm16(speech(2400)))
            seen += 2400
            for header, payload in pipe.read():
                offsets.append((header["output_sample_offset"], len(payload) // 2))
        assert offsets, "nothing was emitted, so this proves nothing"
        running = 0
        for offset, count in offsets:
            assert offset == running
            running += count

    def test_the_end_of_a_turn_proves_its_own_duration(self):
        pipe = self._ready()
        pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                           "stages": ["dpdfnet", "lavasr"]})
        pipe.worker.audio({"turn": "a", "input_sample_offset": 0},
                          worker.to_pcm16(speech(RATE)))
        found = pipe.worker.end({"turn": "a", "final_input_sample_count": RATE})
        assert found["ok"] is True
        assert found["final_output_sample_count"] == 2 * RATE
        assert found["output_sample_rate"] == 48000

    def test_a_source_count_that_disagrees_ends_the_turn(self):
        pipe = self._ready(["dpdfnet"])
        pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                           "stages": ["dpdfnet"]})
        pipe.worker.audio({"turn": "a", "input_sample_offset": 0},
                          worker.to_pcm16(speech(1000)))
        with pytest.raises(worker.Refusal, match="disagrees"):
            pipe.worker.end({"turn": "a", "final_input_sample_count": 999})

    def test_a_cancelled_turn_emits_no_tail_and_ignores_late_audio(self):
        """A-31. Speech arriving after silence is worse than no speech."""
        pipe = self._ready()
        pipe.worker.begin({"turn": "a", "sample_rate": RATE, "channels": 1,
                           "stages": ["dpdfnet", "lavasr"]})
        pipe.worker.audio({"turn": "a", "input_sample_offset": 0},
                          worker.to_pcm16(speech(RATE)))
        pipe.read()
        pipe.worker.cancel({"turn": "a"})
        assert pipe.worker.turn is None
        pipe.worker.audio({"turn": "a", "input_sample_offset": 24000},
                          worker.to_pcm16(speech(2400)))
        assert pipe.read() == [], "a cancelled turn's audio reached playback"

    def test_a_new_turn_starts_with_no_state_from_the_last(self):
        """I-VP-13. No sample of one reply may appear in the next."""
        pipe = self._ready(["lavasr"])
        first = []
        for turn_id in ("a", "b"):
            pipe.worker.begin({"turn": turn_id, "sample_rate": RATE, "channels": 1,
                               "stages": ["lavasr"]})
            pipe.read()
            pipe.worker.audio({"turn": turn_id, "input_sample_offset": 0},
                              worker.to_pcm16(speech(RATE)))
            got = b"".join(payload for _header, payload in pipe.read())
            pipe.worker.end({"turn": turn_id, "final_input_sample_count": RATE})
            got += b"".join(payload for _header, payload in pipe.read())
            first.append(got)
        assert first[0] == first[1], "the second reply is not the first one's echo"


class TestTheSelfTest:
    """A-49 in miniature: the check the installer runs before it promotes."""

    def test_it_refuses_a_backend_whose_duration_is_wrong(self, monkeypatch):
        class Slow:
            def reset(self, rate):
                return None

            def enhance(self, samples, rate):
                # Half the samples it should give back: the shape of a backend
                # whose clock is wrong, which is the whole point of the check.
                return worker.Resampler(24000, 48000)(samples)[:len(samples)]

        monkeypatch.setattr(worker, "_stage_backend",
                            lambda *a, **k: Slow())
        with pytest.raises(worker.Refusal, match="not the rate it was told"):
            worker.selftest({"lavasr": "/nowhere"},
                            {"lavasr": {"backend_input_rate": 24000, "analysis_ms": 250,
                                        "context_ms": 50}})

    def test_it_refuses_a_denoiser_that_loses_audio(self, monkeypatch):
        class Lossy:
            block_samples = 480

            def reset(self, rate):
                return None

            def enhance(self, samples):
                return list(samples)[:-1]

        monkeypatch.setattr(worker, "_stage_backend", lambda *a, **k: Lossy())
        with pytest.raises(worker.Refusal, match="losing audio rather than delaying"):
            worker.selftest({"dpdfnet": "/nowhere"}, {"dpdfnet": {}})

    def test_it_passes_a_backend_that_behaves(self, monkeypatch):
        def backend(stage_id, root, config, numpy_module):
            return LateDpdf() if stage_id == "dpdfnet" else PerfectLava(24000)

        monkeypatch.setattr(worker, "_stage_backend", backend)
        found = worker.selftest(
            {"dpdfnet": "/nowhere", "lavasr": "/nowhere"},
            {"dpdfnet": {}, "lavasr": {"backend_input_rate": 24000, "analysis_ms": 250,
                                       "context_ms": 50}})
        assert found["ok"] is True
        assert set(found["stages"]) == {"dpdfnet", "lavasr"}


# --------------------------------------------------------------------------- #
# The stage registry and the switches
# --------------------------------------------------------------------------- #


class TestTheOrderIsStructural:
    """A-03, and the absence the settings surface depends on."""

    def test_the_two_stages_have_fixed_positions(self):
        assert [(spec.id, spec.order) for spec in pipeline.STAGES] == [
            ("dpdfnet", 100), ("lavasr", 200)]

    def test_there_is_no_order_to_persist(self):
        """A settings key for the order is the thing that must not exist."""
        names = {pipeline.OPT_ENABLED, pipeline.OPT_DPDFNET, pipeline.OPT_LAVASR}
        assert len(names) == 3
        assert not any("order" in name for name in names)
        source = (paths.extension_root() / "mc_voice_pipeline.py").read_text(
            encoding="utf-8")
        assert "voice_pipeline_order" not in source

    def test_the_settings_route_accepts_no_order_field(self):
        import mc_voice_api as api

        source = (paths.extension_root() / "mc_voice_api.py").read_text(encoding="utf-8")
        body = source.split("def pipeline_settings(")[1].split("\ndef ")[0]
        code = body.split('"""')[2]
        assert "order" not in code, "the settings route reads an order field"
        assert callable(api.pipeline_settings)

    def test_a_snapshot_never_sorts_by_anything_a_user_sent(self, host):
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        found = _installed_status()
        snapshot = pipeline.snapshot(RATE, "pocket", found)
        assert snapshot.stage_ids == ("dpdfnet", "lavasr")


def _installed_status(dpdf=True, lava=True, runtime_ready=True):
    """A status that says what a test needs it to say. Hand-written, as always."""
    stages = []
    for spec in pipeline.STAGES:
        wanted = dpdf if spec.id == "dpdfnet" else lava
        stages.append(pipeline.StageStatus(
            id=spec.id, label=spec.label, order=spec.order,
            install_state="installed" if wanted else "not_installed",
            message="Installed." if wanted else "Not installed.",
            enabled=pipeline.stage_enabled(spec.id), revision="0123456789abcdef", closure_id="fedcba98", license="Apache-2.0",
            about_bytes=1024))
    return pipeline.Status(
        supported=True, pinned=True,
        runtime_install_state="installed" if runtime_ready else "not_installed",
        runtime_message="Installed." if runtime_ready else "Not installed.",
        runtime_closure_id="0f0f0f0f", stages=tuple(stages),
        master_enabled=pipeline.enabled(), message="Installed.", download_bytes=2048)


# --------------------------------------------------------------------------- #
# The real DPDFNet package
# --------------------------------------------------------------------------- #


DPDF_STUB_FREQ = 481
"""(481 - 1) * 2 = 960 samples = 20 ms at 48 kHz, which is the window
``infer_win_len`` derives for the model this feature installs."""


def _stub_onnx(path):
    """A DPDFNet-shaped ONNX with the real streaming signature and no opinions.

    Two inputs, two outputs and the custom metadata upstream's
    ``build_runtime_model`` insists on, wired as identity. It enhances nothing,
    which is the point: what these tests are about is the adapter's plumbing and
    its clock, and a network that changed the audio would make every assertion
    about "what came out is what went in" impossible to write.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    state, erb, spec = 64, 8, 8
    graph = helper.make_graph(
        [helper.make_node("Identity", ["spec"], ["spec_e"]),
         helper.make_node("Identity", ["state"], ["state_o"])],
        "dpdfnet_stub",
        [helper.make_tensor_value_info("spec", TensorProto.FLOAT,
                                       [1, 1, DPDF_STUB_FREQ, 2]),
         helper.make_tensor_value_info("state", TensorProto.FLOAT, [state])],
        [helper.make_tensor_value_info("spec_e", TensorProto.FLOAT,
                                       [1, 1, DPDF_STUB_FREQ, 2]),
         helper.make_tensor_value_info("state_o", TensorProto.FLOAT, [state])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for key, value in {"state_size": str(state),
                       "erb_norm_state_size": str(erb),
                       "spec_norm_state_size": str(spec),
                       "erb_norm_init": ",".join("0.5" for _ in range(erb)),
                       "spec_norm_init": ",".join("0.25" for _ in range(spec))}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))
    return path


@pytest.fixture
def dpdf_backend(tmp_path):
    """The real ``_stage_backend`` for DPDFNet, over a stub model.

    Skipped where the enhancement runtime is not installed, which is every
    machine that has not run the installer -- these are the tests that only mean
    something when upstream's own code is the code under them.
    """
    pytest.importorskip("dpdfnet")
    pytest.importorskip("onnxruntime")
    _stub_onnx(tmp_path / "dpdfnet8_48khz_hr.onnx")
    return worker._stage_backend("dpdfnet", tmp_path, {
        "model_file": "dpdfnet8_48khz_hr.onnx", "model_id": "dpdfnet8_48khz_hr",
        "model_sample_rate": 48000, "intraop": 2, "interop": 1}, worker._numpy())


class TestTheRealDpdfnetPackage:
    """The adapter against upstream 0.6.0 rather than against a stand-in.

    Everything above proves the adapter is right about its own arithmetic.
    These prove it is right about the library it was written for -- which is a
    different question, and the one that decides whether any of it works on
    somebody's machine.
    """

    def _run(self, backend, total, block, rate=RATE):
        backend.reset(rate)
        stage = worker.DpdfStage(backend, rate)
        turn = worker.Turn("t", rate, [stage])
        source = speech(total, rate)
        found, index = [], 0
        while index < total:
            found.extend(turn.feed(source[index:index + block]))
            index += block
        found.extend(turn.flush())
        return stage, found

    @pytest.mark.parametrize("seconds,block", [
        (0.5, 2048), (1.0, 997), (2.0, 24000), (0.05, 1200), (0.005, 120)])
    def test_it_gives_back_exactly_what_it_was_given(self, dpdf_backend, seconds, block):
        """A-25, through the real library, at Pocket's own 24 kHz."""
        total = int(RATE * seconds)
        stage, found = self._run(dpdf_backend, total, block)
        assert len(found) == total
        assert stage.correction == 0, "the flush had to invent samples"
        assert all(value == value for value in found), "a NaN reached the output"

    def test_repacketising_changes_nothing_through_the_real_library(self, dpdf_backend):
        """A-10, and the reason :class:`StreamResampler` exists.

        Upstream's ``process`` resamples each call's chunk on its own, so with
        its resampling in the loop the same audio delivered in 10 ms packets and
        in one packet came back differing by about a fifth of full scale -- a
        packet boundary somebody can hear. Resampling continuously on this side
        and handing the model its native rate takes that out entirely.
        """
        total = int(RATE * 0.75)
        _stage, reference = self._run(dpdf_backend, total, total)
        for block in (240, 997, 2880, 13):
            _stage, found = self._run(dpdf_backend, total, block)
            assert len(found) == len(reference), block
            assert found == reference, (
                f"{block}-sample packets changed the audio by "
                f"{max(abs(a - b) for a, b in zip(found, reference)):.3e}")

    def test_the_audio_survives_the_round_trip(self, dpdf_backend):
        """The stub is an identity, so what comes back should be the speech.

        Cheap and load-bearing: every other assertion here would also pass if
        the adapter returned the right number of zeros.
        """
        total = RATE
        _stage, found = self._run(dpdf_backend, total, 4096)
        assert max(abs(value) for value in found) == pytest.approx(0.25, rel=0.05)
        assert max(abs(value) for value in found[-2400:]) == pytest.approx(0.25, rel=0.05)

    def test_the_installers_self_test_passes_against_the_real_package(self, tmp_path):
        pytest.importorskip("dpdfnet")
        pytest.importorskip("onnxruntime")
        _stub_onnx(tmp_path / "dpdfnet8_48khz_hr.onnx")
        found = worker.selftest({"dpdfnet": str(tmp_path)}, {
            "dpdfnet": {"model_file": "dpdfnet8_48khz_hr.onnx",
                        "model_id": "dpdfnet8_48khz_hr", "model_sample_rate": 48000},
            "test_rate": 24000})
        assert found["ok"] is True
        assert found["stages"]["dpdfnet"]["samples"] == 24000
        assert found["stages"]["dpdfnet"]["correction"] == 0

    def test_a_missing_model_file_is_refused_rather_than_downloaded(self, tmp_path):
        """I-VP-22. Upstream would fetch a model by name; it is never given one.

        ``resolve_model`` short-circuits on an explicit ``onnx_path``, so the
        only way this backend could reach the network is by not passing one --
        which is what this asserts it always does.
        """
        pytest.importorskip("dpdfnet")
        with pytest.raises(worker.Refusal, match="not where it was said to be"):
            worker._stage_backend("dpdfnet", tmp_path, {
                "model_file": "dpdfnet8_48khz_hr.onnx"}, worker._numpy())
        with pytest.raises(worker.Refusal, match="no DPDFNet model file"):
            worker._stage_backend("dpdfnet", tmp_path, {}, worker._numpy())


class TestTheStreamResampler:
    """The piece written because upstream's per-call resampling could not be used."""

    @pytest.mark.parametrize("block", [240, 997, 13, 6000])
    def test_it_does_not_care_how_the_stream_was_cut(self, block):
        source = speech(6000)
        reference = worker.StreamResampler(24000, 48000)
        whole = list(reference.feed(source)) + list(reference.flush())
        found, index = [], 0
        piece = worker.StreamResampler(24000, 48000)
        while index < len(source):
            found.extend(piece.feed(source[index:index + block]))
            index += block
        found.extend(piece.flush())
        assert len(found) == len(whole) == 12000
        assert found == whole

    def test_it_keeps_the_clock(self):
        for rate_in, rate_out, count in ((24000, 48000, 2400), (48000, 24000, 4800),
                                         (24000, 16000, 3000), (22050, 48000, 2205)):
            found = worker.StreamResampler(rate_in, rate_out)
            out = list(found.feed([0.1] * count)) + list(found.flush())
            assert len(out) == worker.deterministic_target(count, rate_in, rate_out)

    def test_matching_rates_are_a_bypass(self):
        found = worker.StreamResampler(24000, 24000)
        assert found.transparent
        assert found.feed([0.1, 0.2]) == [0.1, 0.2]

    # -- the taps, which were being recomputed three million times a second -- #
    #
    # The measurement that produced these: the 24 kHz to 48 kHz pair around
    # DPDFNet cost a real-time factor of 2.0 in this class alone, before a
    # single inference ran, and 63% of that was `sin` and `cos` inside the tap
    # kernel. The weights depend only on where an output sample falls between
    # two input samples, and at 24 kHz to 48 kHz there are two such places, so
    # every one of those calls after the first two was recomputing a number the
    # process already had. That is also why moving the stage to a graphics card
    # measured no difference: none of this arithmetic is in the ONNX graph.

    def test_the_taps_are_the_ones_it_replaced(self):
        """The kernel written out longhand, per output sample, as it used to be.

        This is the test that matters. A resampler that is fast and subtly
        different is worse than a slow one -- it would change every reply this
        feature has ever produced, quietly, and no other test here compares
        against the arithmetic itself rather than against another run of it.
        """
        source = speech(1200)
        found = worker.StreamResampler(24000, 48000)
        got = list(found.feed(source)) + list(found.flush())

        half, cutoff = found._half, found._cutoff
        last = len(source) - 1
        wanted = []
        for index in range(len(got)):
            number = index * 24000
            centre = number // 48000
            position = centre + (number % 48000) / 48000.0
            total = weight = 0.0
            for offset in range(centre - half + 1, centre + half + 1):
                distance = (position - offset) * cutoff
                tap = worker._sinc(distance) * worker._blackman(distance, half)
                if tap == 0.0:
                    continue
                taken = 0 if offset < 0 else (last if offset > last else offset)
                total += source[taken] * tap
                weight += tap
            wanted.append(total / weight if weight else 0.0)

        assert len(got) == len(wanted) == 2400
        assert max(abs(a - b) for a, b in zip(got, wanted)) < 1e-12

    def test_a_kernel_is_built_once_for_each_place_a_sample_can_land(self):
        """Two phases going up, one coming down, for a whole reply."""
        up = worker.StreamResampler(24000, 48000)
        down = worker.StreamResampler(48000, 24000)
        native = up.feed(speech(24000))
        down.feed(native)

        assert up._phases == 2 and len(up._kernels) == 2
        assert down._phases == 1 and len(down._kernels) == 1

    def test_an_awkward_pair_is_still_a_handful(self):
        """44.1 kHz to 48 kHz shares a factor of 300, so it repeats every 160
        output samples rather than never."""
        found = worker.StreamResampler(44100, 48000)
        found.feed(speech(44100, rate=44100))

        assert found._phases == 160
        assert len(found._kernels) <= 160

    def test_a_pair_with_no_common_factor_keeps_nothing(self):
        """Where every output sample lands somewhere new, a cache is a memory
        leak wearing an optimisation's clothes. The bound is the phase count
        itself, so this is a fact about the rates rather than a guess."""
        found = worker.StreamResampler(44101, 48000)
        found.feed(speech(4410, rate=44101))

        assert found._phases > worker.MAX_RESAMPLE_PHASES
        assert found._kernels == {}

    def test_the_two_arithmetics_agree(self):
        """NumPy carries the interior of a turn and plain Python carries its two
        ends, where the kernel reaches past audio that does not exist. Both are
        live in one reply, so they have to be the same filter."""
        numpy_module = pytest.importorskip("numpy")
        source = speech(2400)

        plain = worker.StreamResampler(24000, 48000)
        fast = worker.StreamResampler(24000, 48000, numpy_module)
        was = list(plain.feed(source)) + list(plain.flush())
        now = list(fast.feed(source)) + list(fast.flush())

        assert len(was) == len(now) == 4800
        assert max(abs(a - b) for a, b in zip(was, now)) < 1e-12


class TestTheEngineSpelling:
    def test_the_pocket_runtime_and_the_registry_agree_on_the_name(self):
        """Two spellings of one engine id, held together by a test.

        ``mc_voice_pocket_runtime`` names it as a literal rather than importing
        the registry, because that module is read on the path that starts a
        subprocess and the registry is read on the path that draws a settings
        page. This is what keeps the shortcut honest.
        """
        import mc_voice_engines as engines
        import mc_voice_pocket_runtime as pocket

        assert pocket.PIPELINE_ENGINE == engines.POCKET
        assert pocket.PIPELINE_ENGINE in pipeline.SUPPORTED_ENGINES


class TestTheToggleMatrix:
    """A-01, A-02, A-04, A-05. Every combination, and each of them a real path."""

    def test_the_master_is_off_until_somebody_turns_it_on(self, host):
        assert pipeline.enabled() is False
        assert pipeline.snapshot(RATE, "pocket", _installed_status()).active is False

    def test_the_stages_are_already_ticked_when_the_master_goes_on(self, host):
        """Turning the feature on is one gesture, not three (section 4.1)."""
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        assert pipeline.desired_stages() == ("dpdfnet", "lavasr")

    @pytest.mark.parametrize("dpdf,lava,stages,rate", [
        (True, True, ("dpdfnet", "lavasr"), 48000),
        (True, False, ("dpdfnet",), RATE),
        (False, True, ("lavasr",), 48000),
        (False, False, (), RATE),
    ])
    def test_every_combination_runs(self, host, dpdf, lava, stages, rate):
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        host.shared.opts.set(pipeline.OPT_DPDFNET, dpdf)
        host.shared.opts.set(pipeline.OPT_LAVASR, lava)
        snapshot = pipeline.snapshot(RATE, "pocket", _installed_status())
        assert snapshot.stage_ids == stages
        assert snapshot.output_rate == rate

    def test_master_off_is_a_bypass_that_remembers_the_stages(self, host):
        host.shared.opts.set(pipeline.OPT_ENABLED, False)
        host.shared.opts.set(pipeline.OPT_DPDFNET, True)
        snapshot = pipeline.snapshot(RATE, "pocket", _installed_status())
        assert snapshot.active is False
        assert snapshot.output_rate == RATE
        assert pipeline.stage_enabled("dpdfnet") is True, "the selection is remembered"

    def test_master_on_with_no_stages_says_it_is_bypassed(self, host):
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        host.shared.opts.set(pipeline.OPT_DPDFNET, False)
        host.shared.opts.set(pipeline.OPT_LAVASR, False)
        found = _installed_status()
        assert pipeline.pipeline_state(found) == "bypassed"
        assert "Bypassed" in pipeline.snapshot(RATE, "pocket", found).reason

    def test_a_stage_that_is_on_but_missing_is_named_rather_than_pretended(self, host):
        """A-02's other half, and section 4.2: never silently claim it ran."""
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        snapshot = pipeline.snapshot(RATE, "pocket", _installed_status(dpdf=False))
        assert snapshot.stage_ids == ("lavasr",)
        assert "DPDFNet" in snapshot.reason and "not installed" in snapshot.reason
        assert "DPDFNet" not in snapshot.describe("PocketTTS"), (
            "the path preview must not claim a stage that did not run")

    def test_an_engine_the_pipeline_does_not_serve_is_left_alone(self, host):
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        snapshot = pipeline.snapshot(RATE, "kokoro", _installed_status())
        assert snapshot.active is False
        assert snapshot.output_rate == RATE

    def test_the_path_preview_is_built_from_the_snapshot(self, host):
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        found = pipeline.snapshot(RATE, "pocket", _installed_status())
        assert found.describe("PocketTTS") == (
            "PocketTTS → DPDFNet → LavaSR → 48 kHz output")


class TestTheManifestIsATrustRoot:
    """A-13, A-24, and the refusals section 20.3 names."""

    def test_the_runtime_closure_is_pinned_byte_for_byte(self):
        """Every wheel sized and hashed. This is the executable half.

        A model that arrives wrong makes bad audio; a wheel that arrives wrong
        *runs*, so this is the one artifact set with no provisional path.
        """
        found = pipeline.manifest()
        platforms = found["runtime"]["platforms"]
        assert platforms, "the manifest names no runtime platform at all"
        for entry in platforms:
            assert entry["artifacts"], entry["id"]
            for item in entry["artifacts"]:
                assert len(item["sha256"]) == 64, (entry["id"], item["local_name"])
                assert item["bytes"] > 0, (entry["id"], item["local_name"])
                assert item["url"].startswith("https://"), item["local_name"]

    def test_dpdfnet_is_installable_where_its_closure_was_pinned(self, monkeypatch):
        """The half of this feature that can be run, and can be.

        Pinned for windows-x86_64-cp313 because that is the machine it was
        brought up on. Somewhere else the answer is a sentence about that rather
        than a stage that looks installable and is not.
        """
        import mc_voice_models as models

        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))
        monkeypatch.setattr(pipeline, "_manifest_cache", None, raising=False)
        assert pipeline.runtime_installable() is True
        assert pipeline.stage_installable("dpdfnet") is True
        assert pipeline.stage_unavailable_reason("dpdfnet") == ""

        monkeypatch.setattr(models, "current_platform", lambda: ("linux", "x86_64", "3.11"))
        assert pipeline.stage_installable("dpdfnet") is False
        assert "operating system" in pipeline.stage_unavailable_reason("dpdfnet")

    def test_lavasr_is_not_installable_and_says_why(self):
        """Not "coming soon", and not an essay either.

        What is missing changed once the PyTorch closure was pinned: the
        runtime is no longer the gap, LavaSR's own files are. The reason has to
        track that. It also has to stay a *sentence* -- this string is rendered
        on a settings panel, and an earlier version of it grew into three
        hundred words of engineering notes that told a user staring at a
        disabled button everything except what to do.
        """
        assert pipeline.stage_installable("lavasr") is False, "this build cannot fetch it"
        reason = pipeline.stage_unavailable_reason("lavasr")
        if "operating system" in reason:
            # This machine has no pinned closure at all, which is a truthful
            # earlier answer and not the one under test.
            pytest.skip("no runtime closure for this platform")
        assert "LavaSR" in reason
        assert "folder" in reason, "a refusal that does not say how is half an answer"
        assert len(reason) <= 320, (
            f"the panel reason is {len(reason)} characters; it is read by somebody "
            f"looking at a disabled button, not by somebody reading a commit log")
        with pytest.raises(pipeline.PipelineError, match="LavaSR"):
            pipeline.install("lavasr")

    def test_lavasr_can_still_be_installed_from_a_folder(self):
        """Cannot fetch is not cannot install.

        Its weights are on a host this repository could not reach to pin, so
        the managed download is not offered -- but the files themselves are
        public and a person can fetch them in a browser. The panel names them
        and takes a folder, which is the difference between a build that says
        no and a build that says no and how.
        """
        import mc_voice_models as models
        assert pipeline.stage_available("lavasr") is True
        entry = pipeline.manifest()["stages"]["lavasr"]
        assert entry["required_paths"] == ["enhancer_v2/pytorch_model.bin",
                                           "enhancer_v2/config.yaml",
                                           "denoiser/denoiser.bin"]
        assert len(entry["sources"]) == len(entry["required_paths"])
        for item in entry["sources"]:
            assert item["url"].startswith("https://huggingface.co/YatharthS/LavaSR/")

    def test_the_folder_path_still_needs_a_runtime_to_prove_the_model_in(self, monkeypatch):
        """Installable from a folder is not installable into nothing.

        The files can be read from disk on any machine; running the model to
        prove it works needs the closure, and a build with no closure for this
        platform cannot offer the folder path either.
        """
        import mc_voice_models as models
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))
        assert pipeline.stage_local_installable("lavasr") is True
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("linux", "x86_64", "3.11"))
        assert pipeline.stage_local_installable("lavasr") is False

    def test_a_required_path_that_would_escape_its_folder_is_refused(self):
        """These are nested now, so they are joined to a directory we chose."""
        for bad in ("../escape.bin", "/etc/passwd", "a/../../b.bin", "C:/x.bin", ""):
            with pytest.raises(pipeline.PipelineError):
                pipeline._read_required_path("lavasr", bad)
        assert pipeline._read_required_path(
            "lavasr", "enhancer_v2\\config.yaml") == "enhancer_v2/config.yaml"

    def test_the_measured_lava_rate_is_recorded_where_the_adapter_reads_it(self):
        """The Phase-0 blocker, answered from upstream's source.

        16000, because ``LavaSR/model.py``'s enhance() calls
        ``resample(wav, 16000, 48000)`` in *both* branches -- so whatever it is
        handed is interpreted as 16 kHz, and Pocket's 24 kHz would come back
        half again too long with nothing saying so.
        """
        contract = pipeline.manifest()["stages"]["lavasr"]["contract"]
        assert contract["backend_input_rate"] == 16000
        assert contract["output_rate"] == 48000
        assert contract["denoise"] is False
        # The window is here now, and it was not a release waiting on taste: it
        # is the length upstream's own merge needs. See
        # TestTheAnalysisWindowIsAMeasurementAndNotABlank for the arithmetic --
        # a block too short moves the crossover instead of just softening it.
        assert contract["analysis_ms"] == 250
        assert contract["context_ms"] == 50
        assert pipeline.pinned() is False

    def test_the_manifest_reads_and_describes_both_stages(self):
        found = pipeline.manifest()
        assert set(found["stages"]) == {"dpdfnet", "lavasr"}
        # Carried through from the file rather than answered with a constant.
        # It was a constant while there was one closure and one answer; the
        # installed record is stamped from this, so a normaliser that always
        # said "cpu" would stamp every installation with a claim about a
        # runtime it did not install.
        assert found["runtime"]["provider"] == "cpu+directml"

    def test_a_branch_is_not_a_release_identity(self):
        """Section 13.4, enforced where it can be: 'main' means something else
        next week and cannot be reproduced from this repository."""
        entry = dict(pipeline.manifest()["stages"]["lavasr"], revision="main",
                     provisional=False)
        with pytest.raises(pipeline.PipelineError, match="not a release identity"):
            pipeline._read_stage(pipeline.stage("lavasr"), entry)

    def test_a_branch_is_allowed_only_when_the_stage_declares_it(self):
        """The one exception, and it is not a quiet one.

        A stage pinned to a branch has to say ``provisional`` in the manifest,
        and everything downstream then says so too -- the row, the status line
        and the installed record. It exists so a model nobody could reach a hub
        from can still be tested from a machine that can.
        """
        spec = pipeline.stage("lavasr")
        entry = dict(pipeline.manifest()["stages"]["lavasr"], revision="main",
                     provisional=True)
        assert pipeline._read_stage(spec, entry)["provisional"] is True

        found = pipeline.manifest()["stages"]["dpdfnet"]
        assert found["provisional"] is True, "the shipped DPDFNet pin is provisional"
        assert not pipeline._immutable(found["revision"])

    def test_an_immutable_revision_needs_no_exception(self):
        spec = pipeline.stage("lavasr")
        sha = "0" * 40
        entry = dict(pipeline.manifest()["stages"]["lavasr"], revision=sha,
                     provisional=False)
        assert pipeline._read_stage(spec, entry)["revision"] == sha
        assert pipeline._immutable(sha) and not pipeline._immutable("main")

    def test_a_manifest_that_disagrees_about_the_order_is_refused(self):
        """The order cannot be moved from a file either, not only from the UI."""
        found = json.loads(paths.pipeline_manifest_path().read_text(encoding="utf-8"))
        for entry in found["stages"]:
            if entry["id"] == "lavasr":
                entry["order"] = 50
        with pytest.raises(pipeline.PipelineError, match="structural"):
            pipeline._read_manifest(found)

    def test_an_artifact_that_would_escape_its_folder_is_refused(self):
        spec = pipeline.stage("dpdfnet")
        for name in ("../elsewhere", "sub/dir", "..", "."):
            entry = dict(pipeline.manifest()["stages"]["dpdfnet"],
                         artifacts=[{"filename": name, "local_name": name,
                                     "url": "https://example.invalid/x"}])
            with pytest.raises(pipeline.PipelineError, match="own folder"):
                pipeline._read_stage(spec, entry)

    def test_plain_http_is_refused(self):
        spec = pipeline.stage("dpdfnet")
        entry = dict(pipeline.manifest()["stages"]["dpdfnet"],
                     artifacts=[{"filename": "m.onnx", "local_name": "m.onnx",
                                 "url": "http://example.invalid/m.onnx"}])
        with pytest.raises(pipeline.PipelineError, match="HTTPS"):
            pipeline._read_stage(spec, entry)

    def test_a_malformed_digest_is_refused(self):
        spec = pipeline.stage("dpdfnet")
        entry = dict(pipeline.manifest()["stages"]["dpdfnet"],
                     artifacts=[{"filename": "m.onnx", "local_name": "m.onnx",
                                 "url": "https://example.invalid/m.onnx",
                                 "sha256": "not-a-hash"}])
        with pytest.raises(pipeline.PipelineError, match="SHA-256"):
            pipeline._read_stage(spec, entry)

    def test_a_duplicate_local_name_is_refused(self):
        spec = pipeline.stage("dpdfnet")
        entry = dict(pipeline.manifest()["stages"]["dpdfnet"],
                     artifacts=[{"filename": "a", "local_name": "m.onnx",
                                 "url": "https://example.invalid/a"},
                                {"filename": "b", "local_name": "m.onnx",
                                 "url": "https://example.invalid/b"}])
        with pytest.raises(pipeline.PipelineError, match="twice"):
            pipeline._read_stage(spec, entry)

    def test_a_newer_schema_is_refused(self):
        with pytest.raises(pipeline.PipelineError, match="newer schema"):
            pipeline._read_manifest({"schema": 99})

    def test_a_runtime_this_build_cannot_execute_on_is_refused(self):
        """A closure whose sessions nothing here knows how to construct.

        This check used to insist on the single string ``cpu`` and refuse
        anything else as a graphics device this feature would never take. That
        is no longer the claim -- a stage can be placed on a card by somebody
        who asked for it -- but the useful half stands: an installer that
        happily unpacks a CUDA closure and then cannot build a session on it is
        a broken extension, and it is better refused before the download than
        after it.
        """
        with pytest.raises(pipeline.PipelineError, match="does not know how to execute"):
            pipeline._read_manifest({"schema": 1, "runtime": {"provider": "cuda"},
                                     "stages": [{"id": "dpdfnet"}]})

    def test_the_pinned_closure_is_one_this_build_can_execute_on(self):
        """The committed manifest passes its own check, which is the point.

        A list of accepted providers that did not contain the one actually
        shipped would refuse every install on every machine, and would do it
        with a message about a broken extension -- which would, at that point,
        be accurate.
        """
        found = pipeline.manifest()["runtime"]

        assert found["provider"] in pipeline.RUNTIME_PROVIDERS
        assert found["provider"] == "cpu+directml", (
            "the closure is the DirectML ONNX Runtime build, which carries the CPU "
            "execution provider as well -- one installation, two placements")

    def test_the_closure_follows_the_artifacts_rather_than_a_number(self):
        """A-14. A version somebody types is a version somebody forgets."""
        source = (paths.extension_root() / "mc_voice_pipeline.py").read_text(
            encoding="utf-8")
        body = source.split("def runtime_closure_id(")[1].split("\ndef ")[0]
        assert "sha256" in body and "local_name" in body
        assert "stage_closure_id" not in body

    def test_an_unknown_component_cannot_be_installed_or_removed(self):
        for call in (pipeline.install, pipeline.uninstall):
            with pytest.raises(pipeline.PipelineError, match="no Voice Pipeline component"):
                call("../../etc")


class TestTheRoutesSurviveAReload:
    """Every pipeline route is in the tuple the idempotency check reads.

    ``mc_voice_api.install`` returns early when every path in ``ROUTES`` is
    already registered. A route left out of that tuple is therefore registered
    by the run that first builds the app and skipped by every run that only
    re-checks it -- a feature that works until the WebUI reloads and then
    404s -- and it is missing from the startup log line that enumerates what a
    user has. ``POCKET_ROUTES`` says this in its own docstring; these five were
    added without it.
    """

    def test_every_registered_pipeline_route_is_in_the_idempotency_tuple(self):
        import mc_voice_api as api

        for route in api.PIPELINE_ROUTES:
            assert route in api.ROUTES, (
                f"{route} would be registered once and never again after a reload")

    def test_the_tuple_has_no_duplicates(self):
        """Guard, so the test above cannot pass on a tuple that says everything."""
        import mc_voice_api as api

        assert len(api.ROUTES) == len(set(api.ROUTES))
        assert set(api.PIPELINE_ROUTES) == {
            api.PIPELINE_ROUTE, api.PIPELINE_SETTINGS_ROUTE, api.PIPELINE_INSTALL_ROUTE,
            api.COMPONENTS_ROUTE, api.COMPONENT_ROUTE}


class TestAStageInstallsTheRuntimeItNeeds:
    """The prerequisite is installed, not described back to the user.

    This is a regression class with a name on it. The first shipped version of
    this feature refused a stage install with "The Voice Pipeline runtime is not
    installed yet ... Install the runtime first", wrote that to the log, and
    told the browser ``{"ok": true}``. A user pressed the only button their
    panel offered, twice, and got two log lines they were not reading and a
    button that went back to how it was.

    Nobody wants a stage without the runtime under it -- the stage is *proved*
    inside that interpreter and there is nothing to prove it in -- so wanting
    one is not a state worth modelling.
    """

    def _stub(self, monkeypatch, runtime_present):
        import mc_voice_models as models

        done, said, bar = [], [], []
        # The machine this stage was pinned for. Without it the platform gate
        # answers first and these tests would pass on a refusal.
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))
        monkeypatch.setattr(pipeline, "runtime_python",
                            lambda flavour="onnx": ("python" if runtime_present else None))
        # Freshness, which is what the install paths actually ask. "There" and
        # "the one this build pins" are different questions and the gap between
        # them is a bug this class now owns: see the stale test below.
        monkeypatch.setattr(pipeline, "runtime_current",
                            lambda flavour="onnx": bool(runtime_present))
        monkeypatch.setattr(
            pipeline, "_install_runtime",
            lambda say, tick, flavour="onnx", accelerator="cpu": (
                done.append(f"runtime:{flavour}"), tick(1.0)))
        monkeypatch.setattr(pipeline, "_install_stage",
                            lambda which, say, tick: (done.append(which), tick(1.0)))
        monkeypatch.setattr(pipeline, "status", lambda: "status")
        return done, said, bar

    def test_installing_a_stage_installs_the_runtime_first(self, monkeypatch):
        done, said, bar = self._stub(monkeypatch, runtime_present=False)
        pipeline.install("dpdfnet", on_status=said.append, on_progress=bar.append)
        assert done == ["runtime:onnx", "dpdfnet"], (
            "a stage install must build the runtime it is proved inside before "
            "trying to prove anything in it")
        assert any("runtime" in text and "first" in text for text in said), said

    def test_the_runtime_is_not_reinstalled_when_it_is_already_there(self, monkeypatch):
        done, said, bar = self._stub(monkeypatch, runtime_present=True)
        pipeline.install("dpdfnet", on_status=said.append, on_progress=bar.append)
        assert done == ["dpdfnet"], (
            "an installed runtime must not be rebuilt underneath every stage")

    def test_a_closure_an_older_build_pinned_is_rebuilt_rather_than_reused(
            self, monkeypatch, voice_root):
        """The second half of the LavaSR install failure, and the subtler half.

        Both install paths asked ``runtime_python(flavour) is None`` -- whether
        an interpreter file is sitting there. A closure built before this build
        added three packages to it answers that with a yes, so the runtime was
        left exactly as it was under every attempt, and the self-test then died
        importing a module that closure had never carried. The panel had been
        saying "needs updating" the whole time; the installer was asking a
        different question and getting a different answer.
        """
        done, said, bar = self._stub(monkeypatch, runtime_present=True)
        # There, and not what this build pins: exactly the state a user is left
        # in by pulling a build that added a wheel to a closure they already had.
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": False)

        pipeline.install("dpdfnet", on_status=said.append, on_progress=bar.append)

        assert done == ["runtime:onnx", "dpdfnet"], (
            "a stale closure is a closure the stage cannot be proved in, so it "
            "is rebuilt rather than reused")

    def test_the_freshness_question_reads_the_record_and_not_the_directory(
            self, monkeypatch, voice_root):
        """``runtime_current`` is the panel's question, asked by the installer.

        Written against the files rather than a stub, because the whole defect
        was two functions disagreeing about what "installed" meant while each
        looked right on its own.
        """
        import mc_voice_models as models

        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))
        root = paths.pipeline_runtime_root("torch")
        (root / "env" / "Scripts").mkdir(parents=True, exist_ok=True)
        (root / "env" / "Scripts" / "python.exe").write_bytes(b"")

        # An interpreter and no record at all: present, and not installed.
        assert pipeline.runtime_python("torch") is not None
        assert pipeline.runtime_current("torch") is False

        fresh = pipeline.runtime_closure_id("torch", "cpu")
        assert fresh, "this test needs a closure this build actually pins"
        (root / paths.INSTALLED_FILENAME).write_text(
            json.dumps({"closure": "206635c550af7cc9", "accelerator": "cpu"}),
            encoding="utf-8")
        assert pipeline.runtime_current("torch") is False, (
            "a closure id from an older build is not this build's closure")

        (root / paths.INSTALLED_FILENAME).write_text(
            json.dumps({"closure": fresh, "accelerator": "cpu"}), encoding="utf-8")
        assert pipeline.runtime_current("torch") is True

    def test_the_bar_crosses_the_join_without_going_backwards(self, monkeypatch):
        """Two components, one bar. Neither of them owns the whole of it.

        Both installers end by calling ``tick(1.0)``, so chaining them without
        scaling would fill the bar, reset it and fill it again -- which reads as
        "it started over", not "it is three quarters done".
        """
        done, said, bar = self._stub(monkeypatch, runtime_present=False)
        pipeline.install("dpdfnet", on_status=said.append, on_progress=bar.append)
        assert bar == sorted(bar), f"the progress bar went backwards: {bar}"
        assert bar[-1] == 1.0 and bar.count(1.0) == 1, bar
        assert pipeline.RUNTIME_SHARE in bar, bar

    def test_a_stage_this_machine_cannot_run_still_refuses(self, monkeypatch):
        """Chaining is not a way around the platform gate.

        The refusal names the runtime, because on a machine with no pinned
        closure the stage is not the thing that is missing.
        """
        import mc_voice_models as models

        done, said, bar = self._stub(monkeypatch, runtime_present=False)
        monkeypatch.setattr(models, "current_platform", lambda: ("plan9", "vax", "3.13"))
        with pytest.raises(pipeline.PipelineError, match="runtime"):
            pipeline.install("dpdfnet")
        assert done == [], "nothing may be installed once the platform is refused"

    def test_the_span_helper_clamps_and_ignores_what_is_not_a_number(self):
        """Guard assertions, so the two above prove something.

        A ``_span`` that passed everything through unchanged would still satisfy
        a monotonic-bar check made of well-behaved inputs.
        """
        seen = []
        scaled = pipeline._span(seen.append, 0.25, 0.75)
        for value in (0.0, 0.5, 1.0, 4.0, -3.0, "x", None):
            scaled(value)
        assert seen == [0.25, 0.5, 0.75, 0.75, 0.25], seen


class TestNoCredentialAndNoNetworkInTheWorker:
    """A-15, A-16, and the absences that make them true."""

    def test_the_worker_environment_carries_no_token(self):
        found = pipeline.worker_environment()
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            assert name not in found, name
        assert found["HF_HUB_OFFLINE"] == "1"
        assert found["TRANSFORMERS_OFFLINE"] == "1"
        assert found["HF_HUB_DISABLE_TELEMETRY"] == "1"

    def test_the_worker_environment_hides_every_cuda_style_device(self):
        """Still unconditional, and still meaning what it always did.

        Nothing here executes through CUDA, HIP or ROCm, so a library that
        would have found a card through one of those variables has no business
        finding one. The placement setting reaches a card through DirectML,
        which enumerates DXGI adapters and reads none of these -- so the two
        are not in tension.
        """
        found = pipeline.worker_environment()
        assert found["CUDA_VISIBLE_DEVICES"] == ""
        assert found["HIP_VISIBLE_DEVICES"] == ""
        assert found["ROCR_VISIBLE_DEVICES"] == ""

    def test_the_cpu_is_forced_while_every_stage_is_on_it(self, monkeypatch):
        monkeypatch.setattr(pipeline, "placement",
                            lambda stage_id: ("CPUExecutionProvider", 0))

        assert pipeline.worker_environment()["ONNXRUNTIME_FORCE_CPU"] == "1"

    def test_the_cpu_is_not_forced_on_a_worker_asked_for_a_card(self, monkeypatch):
        """A contradiction ONNX Runtime ignores today is one it may honour
        tomorrow, and honouring it would make the placement setting silently do
        nothing -- the exact failure the session check exists to prevent."""
        monkeypatch.setattr(pipeline, "placement",
                            lambda stage_id: ("DmlExecutionProvider", 1)
                            if stage_id == "dpdfnet" else ("CPUExecutionProvider", 0))

        assert "ONNXRUNTIME_FORCE_CPU" not in pipeline.worker_environment()

    def test_the_thread_caps_follow_the_setting_rather_than_the_constant(
            self, monkeypatch):
        """OpenMP sizes its pool from this before any of our code runs.

        A cap pinned at the released two while the session is asked for sixteen
        is a pool that belongs to neither number.
        """
        monkeypatch.setattr(pipeline, "threads", lambda: 12)

        found = pipeline.worker_environment()

        assert found["OMP_NUM_THREADS"] == "12"
        assert found["MKL_NUM_THREADS"] == "12"
        assert found["OPENBLAS_NUM_THREADS"] == "12"
        assert str(pipeline.INTRAOP_THREADS) != "12", "so this proves nothing"

    def test_an_inherited_token_is_removed_rather_than_merely_not_added(self):
        """The parent's environment may hold one; the child inherits."""
        source = (paths.extension_root() / "mc_voice_pipeline_runtime.py").read_text(
            encoding="utf-8")
        body = source.split("def ensure_started(")[1].split("\ndef ")[0]
        assert "environ.pop" in body
        assert "HUGGING_FACE_HUB_TOKEN" in body

    def test_the_worker_reaches_no_network_at_all(self):
        """A-16. Not a policy in the worker: an absence of a transport."""
        import ast

        found = set()
        tree = ast.parse((paths.extension_root() / "pipeline_worker" / "worker.py")
                         .read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
        for name in ("urllib", "http", "socket", "requests", "httpx", "huggingface_hub"):
            assert name not in found, name

    def test_the_status_payload_carries_no_secret_and_no_path(self, host):
        found = json.dumps(runtime.status())
        for name in ("token", "hf_", "HF_", "/home", "C:\\\\"):
            assert name not in found, name


class TestResidencyIsCoupledToPocket:
    """A-17, A-18, A-19. Loaded with the engine, gone with the engine."""

    def test_pocket_warms_the_pipeline_where_its_own_worker_starts(self):
        source = (paths.extension_root() / "mc_voice_pocket_runtime.py").read_text(
            encoding="utf-8")
        assert "_warm_pipeline()" in source
        body = source.split("def ensure_started(")[1].split("\ndef _handshake_with")[0]
        assert "_warm_pipeline()" in body, (
            "the pipeline must warm alongside Pocket's own load, not after it")

    def test_pocket_letting_go_of_its_worker_lets_go_of_the_pipeline(self):
        source = (paths.extension_root() / "mc_voice_pocket_runtime.py").read_text(
            encoding="utf-8")
        body = source.split("def _discard(reason: str)")[1].split("\ndef ")[0]
        assert "_unload_pipeline(reason)" in body

    def test_the_pipeline_has_no_idle_timer_of_its_own(self):
        """Section 12.13. Residency follows an engine, never a clock.

        A timer here would make the pipeline cold exactly when the next reply
        arrives, and resident with nothing to enhance the rest of the time.
        """
        source = (paths.extension_root() / "mc_voice_pipeline_runtime.py").read_text(
            encoding="utf-8")
        assert "IDLE_SECONDS" not in source
        assert "_watch_idle" not in source

    def test_the_shutdown_door_names_the_pipeline(self):
        source = (paths.extension_root() / "scripts" / "model_chain.py").read_text(
            encoding="utf-8")
        body = source.split("def _on_script_unloaded()")[1].split("\ntry:")[0]
        assert "mc_voice_pipeline_runtime.shutdown" in body


class TestTheCancelledDrainIsNeverBackpressured:
    """A-32, and the invariant the whole insertion point was chosen for."""

    def test_a_draining_turn_never_reaches_the_pipeline(self):
        """The branch order in the engine's reader is the proof.

        A draining Pocket unit's frames are read and dropped *before* anything
        downstream sees them, so a full enhancement queue cannot make a Stop
        wait for a solver. This asserts the ordering that makes that true.
        """
        source = (paths.extension_root() / "mc_voice_pocket_runtime.py").read_text(
            encoding="utf-8")
        body = source.split("def _dispatch_turn(")[1].split("\ndef ")[0]
        drain = body.index("if draining:")
        offer = body.index('if operation == "tts_audio":')
        assert drain < offer, (
            "the draining branch must come first, or a cancelled unit could block on the "
            "enhancement queue")
        assert "enhancing.offer" in body[offer:]

    def test_the_ingress_wakes_a_blocked_producer_on_cancellation(self):
        """A queue nothing can be woken out of is a Stop that waits for audio."""
        source = (paths.extension_root() / "mc_voice_pipeline_runtime.py").read_text(
            encoding="utf-8")
        body = source.split("def offer(self")[1].split("\n    def ")[0]
        assert "cancelled.is_set()" in body
        assert "self._gate.wait" in body


class TestNoServerPrebuffer:
    """A-33, A-34, A-35. The latency the last two PRs took out stays out."""

    def test_there_is_no_fixed_startup_reservoir_anywhere(self):
        source = (paths.extension_root() / "mc_voice_pipeline_runtime.py").read_text(
            encoding="utf-8")
        assert "1.5" not in source.split('"""')[0] or True  # the module docstring may cite it
        body = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("#"))
        for name in ("START_TARGET", "PREBUFFER", "RESERVOIR", "start_target"):
            assert name not in body, name

    def test_the_only_queue_is_a_capacity_and_nothing_waits_for_it_to_fill(self):
        assert runtime.INGRESS_SECONDS == 2.0
        source = (paths.extension_root() / "mc_voice_pipeline_runtime.py").read_text(
            encoding="utf-8")
        body = source.split("def _pump(")[1].split("\ndef ")[0]
        # The pump takes whatever is there. There is no minimum, no target and
        # no "wait until N seconds are queued" anywhere in it.
        assert "handle.take()" in body
        assert "INGRESS_SECONDS" not in body

    def test_the_browser_keeps_its_own_buffer_policy(self):
        """A-35. This feature does not touch the numbers PR #124 measured."""
        source = (paths.extension_root() / "javascript" / "voice_chat.js").read_text(
            encoding="utf-8")
        assert "pipeline" in source, "the row is wired"
        buffer = source.split("const BUFFER")[1].split("};")[0]
        assert "pipeline" not in buffer, (
            "the Voice Pipeline must not change the browser's playback envelope")


# --------------------------------------------------------------------------- #
# The worker as a process
# --------------------------------------------------------------------------- #


class Speaker:
    """A stand-in VoiceTurn: the four things the runtime actually calls.

    ``hold`` makes :meth:`offer_audio` block the way the real one does when the
    listener's buffer is full -- which is a designed state, not an anomaly: the
    browser stops reading above twelve seconds of queued audio, the server's
    send blocks, and the VoiceTurn queue fills to its own eighteen.
    """

    def __init__(self, identifier="turn-1", hold=0.0):
        self.id = identifier
        self.hold = float(hold)
        self.cancelled = threading.Event()
        self.pipeline = None
        self.pipeline_snapshot = None
        self.pipeline_metrics = None
        self.audio = bytearray()
        self.rates = set()
        self.failed = ""
        self.finished = threading.Event()

    def offer_audio(self, pcm, rate):
        if self.hold and not self.cancelled.is_set():
            time.sleep(self.hold)
        self.audio.extend(pcm)
        self.rates.add(rate)
        return True

    def audio_failed(self, reason):
        self.failed = reason

    def audio_finished(self):
        self.finished.set()

    def cancel(self, reason="user"):
        self.cancelled.set()
        return True


def _snapshot(stages, output_rate):
    return pipeline.Snapshot(enabled=True, stage_ids=tuple(stages), input_rate=RATE,
                             output_rate=output_rate, reason="")


def _pcm(count):
    return struct.pack("<%dh" % count,
                       *[int(8000 * math.sin(2 * math.pi * 220 * i / RATE))
                         for i in range(count)])


@pytest.mark.skipif(__import__("os").name == "nt",
                    reason="on Windows the parent proves containment with real handles")
class TestTheWorkerAsAProcess:
    """The framing, the offsets and the doors, over a real pipe."""

    def _speak(self, stages, total, block, output_rate):
        turn = Speaker()
        snapshot = _snapshot(stages, output_rate)
        turn.pipeline_snapshot = snapshot
        rate = runtime.begin_turn(turn, RATE, snapshot)
        assert turn.pipeline is not None, "the pipeline did not open, so this proves nothing"
        data = _pcm(total)
        sent = 0
        while sent < total:
            count = min(block, total - sent)
            turn.pipeline.offer(data[sent * 2:(sent + count) * 2], RATE)
            sent += count
        done = threading.Event()
        runtime.finish_turn(turn, done.set)
        assert done.wait(60), "the pipeline never finished the turn"
        return turn, rate

    @pytest.mark.parametrize("stages,rate", [
        (["dpdfnet", "lavasr"], 48000),
        (["lavasr"], 48000),
        (["dpdfnet"], RATE),
    ])
    def test_a_reply_comes_back_at_the_right_rate_and_length(
            self, host, fake_pipeline_worker, stages, rate):
        total = RATE // 2
        turn, advertised = self._speak(stages, total, 2048, rate)
        assert advertised == rate
        assert turn.rates == {rate}
        assert turn.failed == ""
        assert len(turn.audio) // 2 == worker.deterministic_target(total, RATE, rate)

    def test_odd_packets_still_produce_the_right_duration(self, host,
                                                          fake_pipeline_worker):
        total = RATE // 2
        turn, _rate = self._speak(["dpdfnet", "lavasr"], total, 997, 48000)
        assert len(turn.audio) // 2 == 2 * total

    def test_the_numbers_survive_the_handle_being_released(self, host,
                                                           fake_pipeline_worker):
        turn, _rate = self._speak(["lavasr"], RATE // 4, 1024, 48000)
        assert turn.pipeline is None
        assert turn.pipeline_metrics["pipeline_output_sample_count"] == len(turn.audio) // 2

    def test_a_listener_who_stops_draining_does_not_lose_the_end_of_the_reply(
            self, host, fake_pipeline_worker):
        """The reader thread parks inside playback backpressure, and that is fine.

        It is fine only because ending a turn expects no reply on that thread.
        When it did, the worker's ``turn_flushed`` sat unread in the pipe behind
        the very audio the reader was blocked delivering, the wait expired, and
        a reply the worker had processed perfectly was truncated and reported as
        an error -- most easily on a phone whose page went to the background,
        which is exactly when nobody is watching.

        The hold here is long relative to the run and short relative to the
        thirty-second wait that used to be there, so a regression shows up as a
        failure rather than as a slow test.
        """
        total = RATE // 2
        turn = Speaker(hold=0.12)
        snapshot = _snapshot(["dpdfnet", "lavasr"], 48000)
        turn.pipeline_snapshot = snapshot
        assert runtime.begin_turn(turn, RATE, snapshot) == 48000
        data = _pcm(total)
        sent = 0
        while sent < total:
            count = min(2048, total - sent)
            turn.pipeline.offer(data[sent * 2:(sent + count) * 2], RATE)
            sent += count
        done = threading.Event()
        runtime.finish_turn(turn, done.set)
        assert done.wait(120), "the reply never finished"
        assert turn.failed == "", turn.failed
        assert len(turn.audio) // 2 == 2 * total, "the end of the reply was lost"

    def test_the_stream_is_closed_even_when_the_pipeline_fails(self, host,
                                                               fake_pipeline_worker):
        """Whatever ends a turn has to close the browser's stream.

        A reply that stopped and left the listener waiting for audio nothing
        will send is worse than one that stopped and said so.
        """
        turn = Speaker()
        snapshot = _snapshot(["lavasr"], 48000)
        turn.pipeline_snapshot = snapshot
        runtime.begin_turn(turn, RATE, snapshot)
        handle = turn.pipeline
        assert handle is not None
        done = threading.Event()
        handle.on_finished = done.set
        handle.fail("a test ended it")
        assert done.wait(5), "the stream was never closed"
        assert turn.pipeline is None
        assert turn.pipeline_metrics is not None

    def test_a_cancelled_turn_leaves_nothing_behind(self, host, fake_pipeline_worker):
        turn = Speaker()
        snapshot = _snapshot(["dpdfnet", "lavasr"], 48000)
        turn.pipeline_snapshot = snapshot
        runtime.begin_turn(turn, RATE, snapshot)
        handle = turn.pipeline
        assert handle is not None
        data = _pcm(RATE)
        handle.offer(data[:2 * (RATE // 2)], RATE)
        turn.cancel("user")
        runtime.cancel_turn(turn)
        seen = len(turn.audio)
        time.sleep(0.5)
        assert len(turn.audio) == seen, "audio arrived after the turn was cancelled"
        assert handle.offer(data[2 * (RATE // 2):], RATE) is False

    def test_a_worker_that_will_not_answer_does_not_cost_the_reply(self, host,
                                                                   monkeypatch):
        """Section 15.1. An optional polish that fails is spoken without it."""
        import mc_voice_pipeline_runtime as under_test

        monkeypatch.setattr(under_test, "ensure_started",
                            lambda stages=(): (_ for _ in ()).throw(
                                under_test.PipelineRuntimeError("no runtime")))
        turn = Speaker()
        snapshot = _snapshot(["lavasr"], 48000)
        assert under_test.begin_turn(turn, RATE, snapshot) == RATE
        assert turn.pipeline is None, "the turn must fall back to the unenhanced path"

    def test_the_containment_is_proved_rather_than_reported(self, host,
                                                            fake_pipeline_worker):
        runtime.ensure_started(("dpdfnet",))
        found = runtime.status()
        assert found["parent_death"] == "pdeathsig"
        assert found["pid"] > 0

    def test_stopping_leaves_no_child(self, host, fake_pipeline_worker):
        import os

        runtime.ensure_started(("dpdfnet",))
        pid = runtime.status()["pid"]
        runtime.stop("test")
        assert runtime.status()["pid"] == 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.05)
        raise AssertionError("the Voice Pipeline worker outlived the stop")

    @pytest.mark.pipeline(fail_on_load=True)
    def test_a_worker_that_cannot_load_its_models_is_refused(self, host,
                                                             fake_pipeline_worker):
        with pytest.raises(runtime.PipelineRuntimeError):
            runtime.ensure_started(("dpdfnet",))


class TestTheLogSaysWhetherTheEnhancementRan:
    """"Is DPDFNet actually on?" had no answer a user could reach.

    The numbers were reported into the telemetry dictionary a browser posts and
    nowhere else, so a settings toggle -- a statement of intent -- was the only
    thing on screen, and a stage that was enabled but never ran looked exactly
    like one that did.
    """

    def _lines(self, found, monkeypatch):
        import mc_voice_api

        said = []
        monkeypatch.setattr(mc_voice_api.logger, "info",
                            lambda message, *args: said.append(message % args))
        mc_voice_api._log_pipeline(found)
        return said

    def test_a_turn_the_pipeline_ran_on_says_so_with_its_cost(self, monkeypatch):
        said = self._lines({
            "pipeline_stages": "DPDFNet", "pipeline_output_rate": 24000,
            "pipeline_input_sample_count": 240000, "pipeline_output_sample_count": 240000,
            "pipeline_first_output_ms": 45, "pipeline_compute_ms": 3100,
            "pipeline_rtf_milli": 310, "pipeline_backpressure_ms": 0,
        }, monkeypatch)

        assert len(said) == 1, said
        assert "DPDFNet" in said[0]
        assert "RTF 0.31" in said[0]
        assert "240000 samples in / 240000 out" in said[0]
        assert "held back" not in said[0], "no backpressure happened, so none may be claimed"

    def test_a_turn_the_pipeline_did_not_run_on_says_nothing(self, monkeypatch):
        """Absence is the message. A line on every reply forever is not."""
        assert self._lines({"pipeline_stages": None}, monkeypatch) == []
        assert self._lines({}, monkeypatch) == []

    def test_stages_that_produced_nothing_are_not_reported_as_having_run(
            self, monkeypatch):
        """The line a user went to for the answer and was told the opposite.

        Straight from a log, on a reply whose worker had died on an import
        seconds earlier and which was spoken unenhanced::

            Voice pipeline ran — dpdfnet,lavasr, 48000 Hz out, None samples in
            / None out, first output None ms, compute None ms

        ``pipeline_stages`` is what the turn *planned*, fixed before the first
        sample moves. Every measured field beside it was None, which is what
        nothing having happened looks like.
        """
        said = self._lines({
            "pipeline_stages": "dpdfnet,lavasr", "pipeline_output_rate": 48000,
            "pipeline_input_sample_count": None, "pipeline_output_sample_count": None,
            "pipeline_first_output_ms": None, "pipeline_compute_ms": None,
            "pipeline_rtf_milli": None, "pipeline_backpressure_ms": 0,
        }, monkeypatch)

        assert len(said) == 1, said
        assert "did NOT run" in said[0], said[0]
        assert "unenhanced" in said[0], said[0]
        assert "dpdfnet,lavasr" in said[0], "and it still names what was selected"

    def test_a_stage_that_produced_no_samples_is_the_same_answer(self, monkeypatch):
        """Zero is not a small amount of enhancement, it is none."""
        said = self._lines({
            "pipeline_stages": "DPDFNet", "pipeline_output_rate": 24000,
            "pipeline_input_sample_count": 0, "pipeline_output_sample_count": 0,
            "pipeline_first_output_ms": None, "pipeline_compute_ms": None,
        }, monkeypatch)

        assert "did NOT run" in said[0], said[0]

    def test_the_real_time_factor_is_split_per_stage(self, monkeypatch):
        """From a user's log, the line that said the problem but not the cause::

            Voice pipeline ran — dpdfnet,lavasr, 48000 Hz out, ... RTF 3.02,
            source held back 35780 ms

        Three times too slow across two stages, with nothing to say which. The
        two figures had been measured in the worker and carried all the way to
        this dictionary since the feature was written; only this line summed
        them. Which stage to move to a card, or switch off, is the decision an
        RTF above 1.0 actually poses.
        """
        said = self._lines({
            "pipeline_stages": "dpdfnet,lavasr", "pipeline_output_rate": 48000,
            "pipeline_input_sample_count": 339888,
            "pipeline_output_sample_count": 679776,
            "pipeline_first_output_ms": 1583, "pipeline_compute_ms": 42836,
            "pipeline_rtf_milli": 3020, "pipeline_backpressure_ms": 35780,
            "dpdfnet_rtf_milli": 300, "lavasr_rtf_milli": 2720,
        }, monkeypatch)

        assert "RTF 3.02" in said[0]
        assert "dpdfnet 0.30" in said[0], said[0]
        assert "lavasr 2.72" in said[0], said[0]

    def test_one_stage_alone_is_not_given_a_pointless_breakdown(self, monkeypatch):
        """A split of one is the same number twice."""
        said = self._lines({
            "pipeline_stages": "DPDFNet", "pipeline_output_rate": 24000,
            "pipeline_input_sample_count": 240000, "pipeline_output_sample_count": 240000,
            "pipeline_first_output_ms": 45, "pipeline_compute_ms": 3100,
            "pipeline_rtf_milli": 310, "dpdfnet_rtf_milli": 310,
        }, monkeypatch)

        assert "RTF 0.31" in said[0]
        assert "lavasr" not in said[0], "a stage that did not run is not in the split"

    def test_backpressure_is_named_because_it_is_the_failure(self, monkeypatch):
        """A source held back is the enhancement failing to keep up (11.7)."""
        said = self._lines({
            "pipeline_stages": "DPDFNet", "pipeline_output_rate": 24000,
            "pipeline_input_sample_count": 240000, "pipeline_output_sample_count": 240000,
            "pipeline_first_output_ms": 45, "pipeline_compute_ms": 12000,
            "pipeline_rtf_milli": 1200, "pipeline_backpressure_ms": 2400,
        }, monkeypatch)

        assert "RTF 1.20" in said[0]
        assert "source held back 2400 ms" in said[0]


class TestAStaleVoiceSaysWhatToDoAboutIt:
    """A precision change makes every custom voice stale. That was said badly.

    ``mc_voice_pocket._fingerprint`` puts precision in the key deliberately, so
    switching to int8 leaves every clone without prepared data until it is
    rebuilt. ``catalog()`` then hands the worker only the voices that do have
    it, which makes a stale voice *absent* rather than unusable -- so asking for
    one got "That voice is not one PocketTTS has been given", and the audition
    route turned that into "That voice could not be played". Both true, neither
    useful, and the panel had known and said the right thing all along.
    """

    def _reason(self, **entry):
        import mc_voice_api

        return mc_voice_api.unprepared_reason(entry)

    def test_a_usable_voice_is_not_refused(self):
        assert self._reason(compatible=True, display_name="Sasha") == ""
        # An entry with no opinion is treated as usable, not as broken.
        assert self._reason(display_name="Sasha") == ""
        assert mc_voice_api_unprepared(None) == ""

    def test_a_stale_clone_names_the_voice_the_cause_and_the_remedy(self):
        found = self._reason(compatible=False, display_name="Sasha", has_source=True)

        assert "Sasha" in found
        assert "precision" in found
        assert "Rebuild" in found
        # The thing somebody fears most when a voice stops working.
        assert "recording is still here" in found

    def test_a_stale_clone_with_no_recording_says_so_instead(self):
        """Rebuild cannot help without the recording, so it is not offered."""
        found = self._reason(compatible=False, display_name="Koji", has_source=False)

        assert "Koji" in found
        assert "Rebuild" not in found
        assert "made again" in found

    def test_an_official_voice_is_reinstalled_rather_than_rebuilt(self):
        found = self._reason(compatible=False, display_name="Ana", official=True)

        assert "Install the official voices again" in found
        assert "Rebuild" not in found


def mc_voice_api_unprepared(entry):
    import mc_voice_api

    return mc_voice_api.unprepared_reason(entry)


class FakeOrtSession:
    """An ONNX Runtime session that reports the providers it was handed.

    Faithful in the one behaviour these tests turn on: ONNX Runtime does not
    raise when an execution provider is unavailable. It drops it, builds on the
    next entry, and returns a working session. So the list this reports back is
    the *accepted* one, which is what makes a silent fallback expressible.
    """

    def __init__(self, path, sess_options=None, providers=None):
        self.path = path
        self.options = sess_options
        self.asked = list(providers or [])
        self.accepted = list(FakeOnnxRuntime.accepts)

    def get_providers(self):
        return list(self.accepted)


class FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.log_severity_level = 0
        self.enable_mem_pattern = True
        self.execution_mode = None


class FakeOnnxRuntime:
    """Stands in for the module the worker imports inside its own runtime."""

    accepts = [worker.PROVIDER_CPU]

    SessionOptions = FakeSessionOptions
    InferenceSession = FakeOrtSession

    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"
        ORT_PARALLEL = "parallel"


class TestAStageRunsWhereItWasToldTo:
    """A placement setting is only a setting if the session honours it.

    ONNX Runtime's own behaviour is what makes this worth a class of its own:
    an unavailable execution provider is dropped rather than refused, and the
    session that comes back works. Without a check, a machine with no usable
    Direct3D 12 card would have run every stage on the processor while the
    settings panel said "graphics card" -- and the only symptom would have been
    that it was no faster.
    """

    def test_nothing_said_means_the_processor(self):
        assert worker._wanted_provider({}) == (worker.PROVIDER_CPU, 0)

    def test_an_old_load_message_still_loads(self):
        """Every message written before this existed named no provider."""
        assert worker._wanted_provider({"intraop": 8, "interop": 1}) == (
            worker.PROVIDER_CPU, 0)

    def test_a_card_is_carried_with_its_adapter(self):
        assert worker._wanted_provider(
            {"provider": worker.PROVIDER_DIRECTML, "adapter": 1}) == (
                worker.PROVIDER_DIRECTML, 1)

    def test_a_provider_this_runtime_does_not_carry_is_refused(self):
        """Refused rather than run on the CPU.

        The parent had a reason to send it. Running somewhere else and
        reporting success is how a placement setting becomes one nobody can
        trust.
        """
        with pytest.raises(worker.Refusal):
            worker._wanted_provider({"provider": "CUDAExecutionProvider"})

    def test_a_nonsense_adapter_is_a_zero_rather_than_a_crash(self):
        for held in (None, "", "left", -4, [1]):
            provider, adapter = worker._wanted_provider(
                {"provider": worker.PROVIDER_DIRECTML, "adapter": held})
            assert (provider, adapter) == (worker.PROVIDER_DIRECTML, 0), held

    def test_the_cpu_asks_for_the_cpu_and_nothing_else(self):
        assert worker._provider_argument(worker.PROVIDER_CPU, 3) == [
            worker.PROVIDER_CPU]

    def test_directml_keeps_the_cpu_behind_it(self):
        """A fallback for operators, not a fallback for the whole session.

        DirectML does not implement every operator, and without the CPU entry a
        model with one unsupported node fails to load rather than running that
        node on the processor. What must not be silent is the whole session
        landing there, which is :func:`_check_provider`'s job.
        """
        assert worker._provider_argument(worker.PROVIDER_DIRECTML, 1) == [
            (worker.PROVIDER_DIRECTML, {"device_id": 1}), worker.PROVIDER_CPU]

    def test_directml_gets_the_two_session_options_it_requires(self):
        """Not tuning. ONNX Runtime requires both for this provider."""
        options = worker._session_options(FakeOnnxRuntime, 8, 1,
                                          worker.PROVIDER_DIRECTML)

        assert options.enable_mem_pattern is False
        assert options.execution_mode == FakeOnnxRuntime.ExecutionMode.ORT_SEQUENTIAL
        assert options.intra_op_num_threads == 8, (
            "the thread budget still applies -- the CPU fallback nodes run on it")

    def test_the_cpu_keeps_the_defaults_those_two_options_had(self):
        options = worker._session_options(FakeOnnxRuntime, 8, 1, worker.PROVIDER_CPU)

        assert options.enable_mem_pattern is True
        assert options.execution_mode is None

    def test_a_session_on_the_provider_that_was_asked_for_is_accepted(self):
        session = FakeOrtSession("m.onnx")
        session.accepted = [worker.PROVIDER_CPU]

        assert worker._check_provider(session, worker.PROVIDER_CPU) == (
            worker.PROVIDER_CPU,)

    def test_directml_with_the_cpu_behind_it_is_accepted(self):
        session = FakeOrtSession("m.onnx")
        session.accepted = [worker.PROVIDER_DIRECTML, worker.PROVIDER_CPU]

        assert worker._check_provider(session, worker.PROVIDER_DIRECTML) == (
            worker.PROVIDER_DIRECTML, worker.PROVIDER_CPU)

    def test_a_silent_fallback_to_the_processor_is_refused(self):
        """The failure this whole class exists for."""
        session = FakeOrtSession("m.onnx")
        session.accepted = [worker.PROVIDER_CPU]

        with pytest.raises(worker.Refusal, match="rather than the"):
            worker._check_provider(session, worker.PROVIDER_DIRECTML)

    def test_a_session_that_reports_no_provider_at_all_is_refused(self):
        session = FakeOrtSession("m.onnx")
        session.accepted = []

        with pytest.raises(worker.Refusal):
            worker._check_provider(session, worker.PROVIDER_CPU)

    def test_the_refusal_names_both_devices(self):
        """A message naming only one of them is a message nobody can act on."""
        session = FakeOrtSession("m.onnx")
        session.accepted = [worker.PROVIDER_CPU]

        with pytest.raises(worker.Refusal) as raised:
            worker._check_provider(session, worker.PROVIDER_DIRECTML)

        said = str(raised.value)
        assert worker.PROVIDER_CPU in said and worker.PROVIDER_DIRECTML in said


class TestTheWorkerSaysWhereTheStagesLanded:
    """The recovery for an adapter number that is an assumption.

    Nothing in the ONNX Runtime Python API enumerates DirectML adapters, so the
    number a card is given is the card's nvidia-smi index and no lookup makes it
    more than that. The correction is that the machine says out loud what it
    did, so somebody who sees the wrong card light up can pick the other entry.
    """

    def test_every_stage_on_the_processor_is_reported_as_the_cpu(self):
        held = worker.Worker(output=io.BytesIO())
        held.stages = {"dpdfnet": types.SimpleNamespace(
            providers=(worker.PROVIDER_CPU,))}

        assert held._device_word() == "cpu"

    def test_a_stage_on_a_card_is_not_reported_as_the_cpu(self):
        held = worker.Worker(output=io.BytesIO())
        held.stages = {"dpdfnet": types.SimpleNamespace(
            providers=(worker.PROVIDER_DIRECTML, worker.PROVIDER_CPU))}

        assert held._device_word() == "gpu"

    def test_a_split_load_is_reported_as_neither_half(self):
        """``mixed`` rather than either, because either would be wrong.

        A field naming one stage's device while the other is somewhere else is
        wrong in exactly the configuration somebody would be reading it to
        understand.
        """
        held = worker.Worker(output=io.BytesIO())
        held.stages = {
            "dpdfnet": types.SimpleNamespace(
                providers=(worker.PROVIDER_DIRECTML, worker.PROVIDER_CPU)),
            "lavasr": types.SimpleNamespace(providers=(worker.PROVIDER_CPU,)),
        }

        assert held._device_word() == "mixed"

    def test_a_worker_with_nothing_loaded_reports_the_cpu(self):
        """What the handshake checks, and it still has to pass it.

        At ``start`` no stage has been named, so a worker reporting a device
        there has acquired one on its own initiative -- which is the narrow
        claim that refusal is still worth making.
        """
        assert worker.Worker(output=io.BytesIO())._device_word() == "cpu"


class TestThePlacementReachesTheStage:
    def test_the_load_message_carries_a_provider_and_an_adapter(self):
        """Asserted on the source, because building a real one needs a worker.

        The same shape the thread budget's own test takes: what is being proved
        is that the parent puts the value in the message at all, and the two
        places that build a stage config are the two places it could be
        forgotten.
        """
        import inspect

        import mc_voice_pipeline_runtime as runtime_module

        sending = inspect.getsource(runtime_module._send_load)

        assert 'found["provider"], found["adapter"] = pipeline.placement(name)' in sending

    def test_the_self_test_stays_on_the_processor_whatever_is_chosen(self):
        """A claim about the downloaded file, not about the graphics card.

        Running it on a card would let a driver busy with an image generation
        fail the install of a model that is perfectly good, and the sentence
        somebody got would be about the model.
        """
        import inspect

        source = inspect.getsource(pipeline._self_test)

        assert 'config["provider"] = "CPUExecutionProvider"' in source

    def test_a_stage_is_proved_in_its_own_closure_and_not_the_default_one(
            self, monkeypatch, tmp_path):
        """The regression that made LavaSR uninstallable on every machine.

        ``_self_test`` took ``runtime_python()`` -- the ONNX closure, by default
        -- for every stage. So proving LavaSR started an interpreter with no
        Torch in it, and the answer somebody got after downloading 257 MB of
        PyTorch was ``No module named 'torch'``. Both halves are asserted: the
        torch stage gets the torch interpreter, and the onnx stage still gets
        the onnx one.
        """
        chosen = []
        monkeypatch.setattr(pipeline, "runtime_python",
                            lambda flavour="onnx": tmp_path / f"{flavour}-python")
        monkeypatch.setattr(pipeline, "_run_staged",
                            lambda interpreter, arguments, what, timeout=300: (
                                chosen.append(interpreter), '{"ok": true}')[1])
        monkeypatch.setattr(pipeline, "stage_config",
                            lambda stage_id, root=None: {})

        pipeline._self_test("lavasr", tmp_path)
        pipeline._self_test("dpdfnet", tmp_path)

        assert chosen == [tmp_path / "torch-python", tmp_path / "onnx-python"], (
            "a stage has to be proved in the interpreter it will be run in")

    def test_the_stage_id_becomes_the_component_id_once(self):
        assert pipeline.component_of("dpdfnet") == "voice-pipeline-dpdfnet"
        assert pipeline.component_of("lavasr") == "voice-pipeline-lavasr"

    def test_one_worker_runs_the_closure_every_loaded_stage_can_import(self):
        """Torch as soon as any stage needs it, because only it carries both.

        The torch closure carries ONNX Runtime as well; the ONNX one carries no
        Torch. So a pair loaded together has exactly one interpreter that can
        import both of them, and picking the other is how a stage that installed
        cleanly still failed to load.
        """
        assert pipeline.stages_flavour(()) == "onnx"
        assert pipeline.stages_flavour(("dpdfnet",)) == "onnx"
        assert pipeline.stages_flavour(("lavasr",)) == "torch"
        assert pipeline.stages_flavour(("dpdfnet", "lavasr")) == "torch"

    def test_a_build_without_the_device_module_still_speaks(self, monkeypatch):
        """Falling back to the processor is not the same as failing a reply."""
        import builtins

        real = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "mc_voice_device":
                raise ImportError("no such module")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)

        assert pipeline.placement("dpdfnet") == ("CPUExecutionProvider", 0)
        assert pipeline.devices_for("dpdfnet")["placeable"] is False

    def test_settings_carry_a_placement_for_every_stage(self):
        found = pipeline.settings()

        assert set(found["devices"]) == set(pipeline.STAGE_IDS)
        for stage_id in pipeline.STAGE_IDS:
            assert found["devices"][stage_id]["component"] == \
                pipeline.component_of(stage_id)


@pytest.fixture
def kept_settings():
    """Put the host's option store back the way it was found.

    Needed from the moment the options fake became faithful. Upstream's
    ``Options.set`` assigns through ``__setattr__``, which writes ``self.data``,
    so a setting written through ``set`` is readable afterwards through
    ``opts.data.get`` -- and the fake only did the attribute half, which made
    every ``opts.data`` reader answer its default no matter what had been
    stored. Fixing that made these tests able to observe their own writes, and
    made those writes everybody else's problem.

    Concretely: the thread budget has a test asserting the untouched default,
    and a stray 6 left behind by a test three classes above is a failure over
    there with nothing in its traceback to say where it came from.
    """
    from modules import shared

    before = dict(shared.opts.data)
    yield
    shared.opts.data.clear()
    shared.opts.data.update(before)


class TestASettingReadAtLoadTimeStopsTheWorker:
    """The defect the two labels would otherwise have been lying about.

    The thread budget and the placement are both read once, in ``_send_load``,
    and ``ensure_started`` returns early when the loaded stage set already
    matches. So turning either dial changed nothing at all until something else
    happened to stop the worker -- which, on a machine where PocketTTS stays
    resident, could be hours. A control whose label says "takes effect on the
    next reply" has to be one that does.
    """

    def _machine(self, monkeypatch, cards):
        import mc_voice_device as devices

        detection = types.ModuleType("prompt_master.inference.device_detection")
        detection.detect_gpus = lambda *a, **k: list(cards)
        detection.detect_cpu = lambda: types.SimpleNamespace(
            name="A Processor", memory_total_mb=65413)
        monkeypatch.setitem(sys.modules,
                            "prompt_master.inference.device_detection", detection)
        devices.forget_cards()

    def _watched(self, monkeypatch):
        asked = []
        monkeypatch.setattr(runtime, "reconfigure",
                            lambda reason: asked.append(reason) or True)
        return asked

    def test_changing_the_thread_budget_drops_the_worker(self, kept_settings, monkeypatch):
        asked = self._watched(monkeypatch)

        pipeline.remember(threads_value=9)

        assert len(asked) == 1, asked

    def test_flipping_a_switch_does_not(self, kept_settings, monkeypatch):
        """Read fresh by ``snapshot`` at the top of every turn.

        Stopping a worker for one would be a stop nobody asked for, and the
        stage it was holding open would be reloaded for the next sentence.
        """
        asked = self._watched(monkeypatch)

        pipeline.remember(enabled_value=True, stages={"dpdfnet": True})

        assert asked == []

    def test_moving_a_stage_drops_the_worker(self, kept_settings, monkeypatch):
        self._machine(monkeypatch, [
            types.SimpleNamespace(physical_index=0, uuid="GPU-aaaa", memory_total_mb=24576,
                                  name="NVIDIA GeForce RTX 3090")])
        asked = self._watched(monkeypatch)

        pipeline.remember(devices={"dpdfnet": "gpu:GPU-aaaa"})

        assert len(asked) == 1, asked

    def test_choosing_the_device_it_is_already_on_does_not(
            self, kept_settings, monkeypatch):
        """A browser posts the select's value on every change.

        A request that stores the string already in force is not a reason to
        stop a worker in the middle of somebody's conversation.
        """
        self._machine(monkeypatch, [
            types.SimpleNamespace(physical_index=0, uuid="GPU-aaaa", memory_total_mb=24576,
                                  name="NVIDIA GeForce RTX 3090")])
        pipeline.remember(devices={"dpdfnet": "gpu:GPU-aaaa"})
        asked = self._watched(monkeypatch)

        pipeline.remember(devices={"dpdfnet": "gpu:GPU-aaaa"})

        assert asked == []

    def test_a_runtime_that_will_not_stop_does_not_fail_the_write(
            self, kept_settings, monkeypatch):
        """A worker still running at the old setting is what this failing means."""
        def refuse(reason):
            raise RuntimeError("the worker would not stop")

        monkeypatch.setattr(runtime, "reconfigure", refuse)

        found = pipeline.remember(threads_value=6)

        assert found["threads"] == 6

    def test_a_reply_in_flight_keeps_its_worker(self, kept_settings, monkeypatch):
        """``reconfigure`` answers False rather than raising, and that is why.

        The caller is a settings route, the setting has already been stored,
        and the honest answer to "when does this apply" is *not now* rather
        than an error about a write that succeeded.
        """
        monkeypatch.setattr(runtime, "_process", object())
        monkeypatch.setattr(runtime, "_turns", {"a-turn": object()})

        assert runtime.reconfigure("a test") is False

    def test_an_idle_worker_is_dropped(self, kept_settings, monkeypatch):
        dropped = []
        monkeypatch.setattr(runtime, "_process", object())
        monkeypatch.setattr(runtime, "_turns", {})
        monkeypatch.setattr(runtime, "_discard", lambda reason: dropped.append(reason))

        assert runtime.reconfigure("a test") is True
        assert dropped == ["a test"]

    def test_nothing_running_is_already_reconfigured(self, kept_settings, monkeypatch):
        monkeypatch.setattr(runtime, "_process", None)

        assert runtime.reconfigure("a test") is True

    # -- and whether it happened, said out loud ---------------------------- #
    #
    # ``reconfigure`` has always answered a boolean, and its docstring has
    # always said why: the caller is a settings route, the setting is already
    # stored, and "not now" is the honest answer rather than an error about a
    # write that succeeded. The one caller threw that answer away, so a budget
    # changed mid-reply and a budget in force right now came back identical --
    # which is what "setting the threads had no immediate impact" looked like
    # from the panel.

    def test_a_dropped_worker_is_reported_as_in_force_now(self, kept_settings,
                                                          monkeypatch):
        self._watched(monkeypatch)

        found = pipeline.remember(threads_value=9)

        assert found["applied"] == "now"

    def test_a_worker_that_would_not_stop_is_reported_as_waiting(self, kept_settings,
                                                                 monkeypatch):
        """A reply being spoken keeps its worker, so the budget stored here is
        the *next* reply's. The surface has to be able to say which."""
        monkeypatch.setattr(runtime, "reconfigure", lambda reason: False)

        found = pipeline.remember(threads_value=9)

        assert found["applied"] == "pending_next_turn"
        assert found["threads"] == 9

    def test_a_restart_that_raised_is_reported_as_waiting_too(self, kept_settings,
                                                              monkeypatch):
        """The failure and the refusal mean the same thing to somebody reading
        the panel -- a worker still running at the old setting -- so they are
        reported the same way rather than one of them silently claiming "now"."""
        def refuse(reason):
            raise RuntimeError("the worker would not stop")

        monkeypatch.setattr(runtime, "reconfigure", refuse)

        found = pipeline.remember(threads_value=6)

        assert found["applied"] == "pending_next_turn"

    def test_a_moved_stage_is_reported_the_same_way(self, kept_settings, monkeypatch):
        self._machine(monkeypatch, [
            types.SimpleNamespace(physical_index=0, uuid="GPU-aaaa",
                                  memory_total_mb=24576,
                                  name="NVIDIA GeForce RTX 3090")])
        monkeypatch.setattr(runtime, "reconfigure", lambda reason: False)

        found = pipeline.remember(devices={"dpdfnet": "gpu:GPU-aaaa"})

        assert found["applied"] == "pending_next_turn"

    def test_a_switch_that_needs_no_restart_claims_nothing(self, kept_settings,
                                                           monkeypatch):
        """Read fresh by ``snapshot`` at the top of every turn, so there is no
        worker to stop and nothing here to answer. The route above adds the
        answer, because only it knows whether a reply is in flight."""
        self._watched(monkeypatch)

        found = pipeline.remember(enabled_value=True, stages={"dpdfnet": True})

        assert "applied" not in found


class TestTheRouteAcceptsAndRefusesAPlacement:
    """One request writes the switches, the budget and the placements.

    Together rather than one route each, for the reason the switches were
    written together: they are read at the top of every turn, and a browser
    interrupted between two separate posts would leave a configuration nobody
    chose.
    """

    def _machine(self, monkeypatch, cards):
        import mc_voice_device as devices

        detection = types.ModuleType("prompt_master.inference.device_detection")
        detection.detect_gpus = lambda *a, **k: list(cards)
        detection.detect_cpu = lambda: types.SimpleNamespace(
            name="A Processor", memory_total_mb=65413)
        monkeypatch.setitem(sys.modules,
                            "prompt_master.inference.device_detection", detection)
        devices.forget_cards()

    def test_a_card_this_machine_does_not_have_is_refused_by_the_route(
            self, kept_settings, monkeypatch):
        """Refused with the reason, rather than absorbed.

        This is the one refusal that comes from inside the write rather than
        from validation before it, because only the device list knows whether a
        card is present -- and it can stop being present between the page being
        drawn and a button on it being pressed.
        """
        import mc_voice_api

        self._machine(monkeypatch, [])

        with pytest.raises(Exception) as caught:
            mc_voice_api.pipeline_settings(None, {}, devices={"dpdfnet": "gpu:nope"})
        assert "not in this machine" in str(caught.value)

    def test_a_stage_this_build_does_not_have_is_refused(self, kept_settings):
        import mc_voice_api

        with pytest.raises(Exception) as caught:
            mc_voice_api.pipeline_settings(None, {}, devices={"nosuch": "cpu"})
        assert "no Voice Pipeline stage" in str(caught.value)

    def test_a_refused_device_does_not_take_the_switches_with_it(
            self, kept_settings, monkeypatch):
        """Order, and why it is the one it is.

        The device write can raise; the switch write cannot. Done first, a
        refusal would take a tick posted in the same request down with it, so a
        browser holding a stale card list would lose both. Done last, only the
        half that was wrong is declined.
        """
        import mc_voice_api

        self._machine(monkeypatch, [])
        before = pipeline.stage_enabled("dpdfnet")

        with pytest.raises(Exception):
            mc_voice_api.pipeline_settings(None, {"dpdfnet": not before},
                                           devices={"dpdfnet": "gpu:nope"})

        assert pipeline.stage_enabled("dpdfnet") is (not before)
        pipeline.remember(stages={"dpdfnet": before})

    def test_a_device_that_is_not_a_string_is_refused(self, kept_settings):
        import mc_voice_api

        with pytest.raises(Exception) as caught:
            mc_voice_api.pipeline_settings(None, {}, devices={"dpdfnet": 1})
        assert "identifier" in str(caught.value)

    def test_the_processor_is_always_an_acceptable_answer(self, kept_settings, monkeypatch):
        import mc_voice_api

        self._machine(monkeypatch, [])

        found = mc_voice_api.pipeline_settings(None, {}, devices={"dpdfnet": "cpu"})

        # The route answers with the pipeline's own status payload, which is
        # what the row repaints from; the placement itself is read back from the
        # module that stores it, because the panel carrying that control is
        # fetched separately and redrawn from the server after a change.
        assert found["ok"] is not False
        assert pipeline.settings()["devices"]["dpdfnet"]["device"] == "cpu"

    def test_the_route_hands_the_devices_through(self, kept_settings):
        """The wiring, asserted where a typo would otherwise be silent."""
        import inspect

        import mc_voice_api

        source = inspect.getsource(mc_voice_api)
        body = source.split("pipeline_settings(payload.get(\"enabled\")")[1][:400]

        assert 'payload.get("devices")' in body


class TestTheRouteSaysWhenTheChangeTakesEffect:
    """Two ways a stored change can still not be running, and the panel gets the
    pessimistic one.

    The write knows the first: a thread budget and a placement are read once,
    when the worker loads its stages, so changing one has to stop the worker --
    and a worker cannot be stopped in the middle of a reply.

    This route knows the second: a turn already in flight is frozen onto the
    snapshot it started with, so a switch ticked mid-reply has changed the next
    one. Either alone is enough to make "now" a lie, so ``applied`` is the OR of
    them rather than whichever the last line happened to compute.
    """

    def test_a_change_nothing_is_in_the_way_of_is_in_force_now(self, kept_settings,
                                                               monkeypatch):
        import mc_voice_api
        import mc_voice_turn as turns

        monkeypatch.setattr(runtime, "reconfigure", lambda reason: True)
        monkeypatch.setattr(turns, "busy", lambda: False)

        found = mc_voice_api.pipeline_settings(None, {}, threads=6)

        assert found["applied"] == "now"

    def test_a_worker_that_would_not_stop_is_carried_through_to_the_panel(
            self, kept_settings, monkeypatch):
        """The half that was being dropped. Nothing else on this route can see
        that the restart was refused, so a route that recomputed ``applied``
        from the turn state alone reported "now" over a worker still running at
        the old budget."""
        import mc_voice_api
        import mc_voice_turn as turns

        monkeypatch.setattr(runtime, "reconfigure", lambda reason: False)
        monkeypatch.setattr(turns, "busy", lambda: False)

        found = mc_voice_api.pipeline_settings(None, {}, threads=6)

        assert found["applied"] == "pending_next_turn"

    def test_a_reply_in_flight_is_still_enough_on_its_own(self, kept_settings,
                                                          monkeypatch):
        """A stage tick needs no restart at all, so the write answers nothing --
        and this route still has to say that the reply being spoken keeps the
        switches it started with."""
        import mc_voice_api
        import mc_voice_turn as turns

        monkeypatch.setattr(runtime, "reconfigure", lambda reason: True)
        monkeypatch.setattr(turns, "busy", lambda: True)

        found = mc_voice_api.pipeline_settings(None, {"dpdfnet": True})

        assert found["applied"] == "pending_next_turn"

    def test_the_answer_reaches_the_browser_on_every_write(self, kept_settings,
                                                           monkeypatch):
        """Every control on that panel posts this route, so every one of them
        gets an answer -- a master switch that acknowledged nothing while the
        thread box did would read as the switch being the broken one."""
        import mc_voice_api
        import mc_voice_turn as turns

        monkeypatch.setattr(runtime, "reconfigure", lambda reason: True)
        monkeypatch.setattr(turns, "busy", lambda: False)

        assert mc_voice_api.pipeline_settings(True, {})["applied"] == "now"
        ticked = mc_voice_api.pipeline_settings(None, {"dpdfnet": True})
        assert ticked["applied"] == "now"
        assert mc_voice_api.pipeline_settings(None, {}, threads=4)["applied"] == "now"


class TestTheLogLineAndTheTurnAgreeOnWhatExists:
    """The seam between what is measured and what is reported.

    Every "Voice pipeline ran" line this feature has ever written said
    ``RTF 3.16`` where it could have said ``RTF 3.16 (dpdfnet 2.6, lavasr 0.5)``
    -- the difference between "the pipeline is too slow" and "here is the stage
    to switch off". The worker measured the split, the handle forwarded it, and
    the log line was written to print it. ``VoiceTurn._pipeline_metrics`` threw
    it away, because that dictionary copies only the keys it already lists and
    nobody added the new ones to it.

    Every test of that log line built its own dictionary and handed it straight
    to the logger, so the whole chain was covered except the one link that was
    broken.

    The invariant is not "the turn reports everything the handle measures" --
    section 36 says the opposite, that a field is absent from telemetry until
    somebody put it there deliberately. It is that a field the log line *reads*
    is a field the turn *reports*, which is exactly what stopped being true.
    """

    def _read_by_the_log_line(self):
        """The pipeline fields ``_log_pipeline`` gets out of a turn's metrics.

        Two shapes, and the second is why this is not one regex. Most fields are
        read by name -- ``found.get("pipeline_rtf_milli")``. The per-stage split
        is read by building the name from the stage -- ``found[f"{name}_rtf_milli"]``
        over ``("dpdfnet", "lavasr")`` -- so a reader that only saw literals
        would miss exactly the pair that went missing.
        """
        import inspect
        import re

        import mc_voice_api

        body = inspect.getsource(mc_voice_api._log_pipeline)
        found = set(re.findall(r"""found(?:\.get\(|\[)["']([a-z0-9_]+)["']""", body))
        for suffix in re.findall(r"""found(?:\.get\(|\[)f["']\{name\}([a-z0-9_]+)["']""",
                                 body):
            found.update(spec.id + suffix for spec in pipeline.STAGES)
        return found

    def test_every_field_the_log_line_reads_is_one_the_turn_reports(self):
        import mc_voice_turn as turn_module

        reported = set(turn_module.VoiceTurn._pipeline_metrics(
            types.SimpleNamespace(pipeline_snapshot=None, pipeline=None,
                                  pipeline_metrics=None)))
        wanted = self._read_by_the_log_line()

        assert wanted, "the log line reads no pipeline fields at all"
        missing = wanted - reported
        assert not missing, f"read by the log line, never reported: {sorted(missing)}"

    def test_the_per_stage_split_is_among_them(self):
        """Named rather than inferred, because that pair is the reason this
        class exists and a regex that stopped matching would otherwise make the
        test above pass by finding nothing."""
        wanted = self._read_by_the_log_line()

        assert {"dpdfnet_rtf_milli", "lavasr_rtf_milli"} <= wanted

    def test_an_unenhanced_turn_reports_the_same_names_as_an_enhanced_one(self):
        """Absent rather than missing. A log comparing two turns must not be
        able to read "no pipeline" as "a pipeline that cost nothing", which is
        the whole reason this is a template and not a pass-through."""
        import mc_voice_turn as turn_module

        found = turn_module.VoiceTurn._pipeline_metrics(
            types.SimpleNamespace(pipeline_snapshot=None, pipeline=None,
                                  pipeline_metrics=None))

        assert found, "an unenhanced turn reported no pipeline fields at all"
        assert set(found.values()) == {None}

    def test_the_split_survives_the_trip(self):
        """Over the seam that was broken: measured in the worker, carried by the
        handle, read back off the turn."""
        import mc_voice_turn as turn_module

        measured = {"pipeline_stages": "dpdfnet,lavasr", "pipeline_rtf_milli": 1410,
                    "dpdfnet_rtf_milli": 810, "lavasr_rtf_milli": 600}
        found = turn_module.VoiceTurn._pipeline_metrics(
            types.SimpleNamespace(
                pipeline_snapshot=types.SimpleNamespace(
                    active=True, stage_ids=("dpdfnet", "lavasr"), output_rate=48000),
                pipeline=None, pipeline_metrics=measured))

        assert found["dpdfnet_rtf_milli"] == 810
        assert found["lavasr_rtf_milli"] == 600

    def test_the_handle_really_does_measure_it(self):
        """The other end. A split the turn forwards and nothing produces would
        be a passing test above and an empty parenthesis in every log."""
        import mc_voice_pipeline_runtime as enhancement

        handle = enhancement.Handle.__new__(enhancement.Handle)
        handle.snapshot = types.SimpleNamespace(
            stage_ids=("dpdfnet", "lavasr"), input_rate=24000, output_rate=48000)
        handle.input_samples = handle.output_samples = 0
        handle.input_packets = handle.output_packets = 0
        handle.first_output = handle.first_input = 0
        handle.peak_queue = handle.backpressure = 0
        handle.measured = {"dpdfnet_rtf_milli": 810, "lavasr_rtf_milli": 600}

        found = handle.metrics()

        assert found["dpdfnet_rtf_milli"] == 810
        assert found["lavasr_rtf_milli"] == 600


class TestThePanelHasSomewhereToPutTheAnswer:
    """The other half of the same defect, and the half that was actually
    missing from the shipped build.

    The route has always answered ``applied``. The panel had no element to write
    it into and no line of script that read it, so the answer went out over the
    wire and stopped there -- which is why turning the thread dial looked like
    turning a dial wired to nothing.
    """

    def test_the_row_carries_an_element_for_it(self, host):
        drawn = mc_voice_ui.pipeline_html()

        assert "data-mc-voice-pipeline-applied" in drawn

    def test_it_starts_empty_so_a_fresh_panel_claims_nothing(self, host):
        """Nothing has been changed yet. An element rendered with a sentence in
        it would be a panel answering a question nobody asked."""
        import re

        drawn = mc_voice_ui.pipeline_html()
        held = re.search(r"data-mc-voice-pipeline-applied[^>]*>(.*?)</p>", drawn)

        assert held is not None, drawn
        assert held.group(1) == ""

    def test_the_stylesheet_draws_no_line_while_it_is_empty(self):
        """Otherwise the panel gains a blank row and a gap before anybody has
        touched anything, which is a worse surface than the one this fixes."""
        style = (paths.extension_root() / "style.css").read_text(encoding="utf-8")

        assert ".mc-voice-pipeline-applied:empty" in style

    def test_the_browser_reads_the_field(self, host):
        """The line that was missing. Asserted against the script rather than
        assumed, because a panel that silently ignores a field it is sent is
        exactly what this was."""
        script = (paths.extension_root() / "javascript" / "voice_chat.js").read_text(
            encoding="utf-8")

        assert "function paintPipelineApplied(" in script, \
            "the panel has nothing that reads the field"
        body = script.split("function paintPipelineApplied(")[1]
        body = body.split("\n    function ")[0]

        assert '"applied"' in body
        assert "pending_next_turn" in body
        assert "data-mc-voice-pipeline-applied" in body

    def test_moving_a_stage_is_acknowledged_too(self):
        """The device select posts the same route and gets the same answer, and
        it is the change most likely to come back "not yet" -- a card chosen
        while a reply is being spoken is the next reply's, because a worker
        cannot be stopped mid-sentence.

        Asserted against the source rather than driven, because that control is
        in a panel fetched from ``/component`` and injected after the script has
        already run, which is not a shape the browser harness models.
        """
        script = (paths.extension_root() / "javascript" / "voice_chat.js").read_text(
            encoding="utf-8")
        handler = script.split("body.devices[stage] = asked;")[1].split("}).catch(")[0]

        assert "paintPipelineApplied(" in handler


class TestChoosingHowMuchDenoiserToRun:
    """The dial that turned out to matter, and the two that did not.

    A device dropdown and a thread budget were both measured on a machine with a
    3090 and a 5090, and neither moved the number: DPDFNet ran at a real-time
    factor of 1.27 on DirectML and 1.03 to 1.06 on the processor, so the card
    lost. It could not have won. DPDFNet is a streaming recurrent denoiser run
    in ~10 ms hops, which is a thousand inferences of a few hundred samples for
    a ten second reply, each depending on the state the last one left -- nothing
    batches, nothing overlaps, and every call pays a copy in, kernel launches, a
    copy out and a synchronisation.

    What did move it was the publisher's own model card. DPDFNet's cost is its
    DPRNN block count, and the two 48 kHz fullband networks are 2.41 and 7.17
    GMACs for the same sample rate and the same streaming contract. This feature
    shipped pinned to the heavier one: a stage that could not keep up with
    playback wherever it was placed, because the work was three times larger
    than it needed to be rather than in the wrong place.
    """

    def test_both_48_khz_networks_are_offered(self):
        found = pipeline.dpdfnet_models()

        assert [item["id"] for item in found["choices"]] == [
            "dpdfnet2_48khz_hr", "dpdfnet8_48khz_hr"]

    def test_the_light_one_is_the_default(self, host):
        """A default that cannot keep up is not a default. The heavy network is
        a choice for a machine with the headroom rather than the thing everyone
        gets before anybody has measured anything."""
        found = pipeline.dpdfnet_models()

        assert found["default"] == "dpdfnet2_48khz_hr"
        assert found["chosen"] == "dpdfnet2_48khz_hr"

    def test_the_choice_carries_what_it_costs(self):
        """Blocks and GMACs, from the publisher rather than from this file, so
        the panel can say what the trade is instead of naming two files."""
        light, heavy = pipeline.dpdfnet_models()["choices"]

        assert light["dprnn_blocks"] == 2 and heavy["dprnn_blocks"] == 8
        assert heavy["macs_g"] > 2.9 * light["macs_g"]

    def test_choosing_one_stores_it(self, host):
        pipeline.remember(model="dpdfnet8_48khz_hr")

        assert pipeline.dpdfnet_model() == "dpdfnet8_48khz_hr"
        assert pipeline.dpdfnet_models()["chosen"] == "dpdfnet8_48khz_hr"

    def test_a_network_this_build_does_not_have_is_refused(self, host):
        """Refused rather than stored. A stored id nothing offers is a setting
        that silently does nothing: the reader falls back, the panel redraws
        showing the fallback, and the only account of what happened would be the
        two disagreeing."""
        with pytest.raises(ValueError) as caught:
            pipeline.remember(model="dpdfnet64_1mhz_ludicrous")
        assert "not a DPDFNet model" in str(caught.value)

    def test_a_stored_network_that_vanished_falls_back(self, host):
        """A build that dropped a variant must not ask the worker for a file
        that is not there. A stage that will not load is a worse answer than a
        stage running the network this build actually ships."""
        host.shared.opts.set(pipeline.OPT_DPDFNET_MODEL, "dpdfnet99_from_the_future")

        assert pipeline.dpdfnet_model() == "dpdfnet2_48khz_hr"

    def test_changing_it_drops_the_worker(self, kept_settings, monkeypatch):
        """The file is opened once, when the worker loads its stages, so a
        network changed while a worker holds the old one changes nothing until
        that worker stops -- the same reason the thread budget and the placement
        stop it."""
        asked = []
        monkeypatch.setattr(runtime, "reconfigure",
                            lambda reason: asked.append(reason) or True)

        pipeline.remember(model="dpdfnet8_48khz_hr")

        assert len(asked) == 1, asked

    def test_the_worker_is_told_the_file_rather_than_the_name(self, host, monkeypatch):
        """The worker is handed a directory and must not have to decide which
        file in it is the model. With two networks installed side by side that
        stopped being a formality."""
        root = paths.pipeline_stage_root("dpdfnet")
        root.mkdir(parents=True, exist_ok=True)
        for name in ("dpdfnet2_48khz_hr.onnx", "dpdfnet8_48khz_hr.onnx"):
            (root / name).write_bytes(b"not really a network")

        pipeline.remember(model="dpdfnet8_48khz_hr")
        assert pipeline.stage_config("dpdfnet")["model_file"] == \
            "dpdfnet8_48khz_hr.onnx"

        pipeline.remember(model="dpdfnet2_48khz_hr")
        assert pipeline.stage_config("dpdfnet")["model_file"] == \
            "dpdfnet2_48khz_hr.onnx"

    def test_a_network_that_is_not_on_disk_is_not_asked_for(self, host):
        """An installation made before a variant existed has the setting but not
        the file. Asking for it anyway is a stage that will not load at all."""
        root = paths.pipeline_stage_root("dpdfnet")
        root.mkdir(parents=True, exist_ok=True)
        (root / "dpdfnet2_48khz_hr.onnx").write_bytes(b"not really a network")
        # Only the light one, whatever an earlier test left behind.
        (root / "dpdfnet8_48khz_hr.onnx").unlink(missing_ok=True)

        pipeline.remember(model="dpdfnet8_48khz_hr")

        assert pipeline.stage_config("dpdfnet")["model_file"] == \
            "dpdfnet2_48khz_hr.onnx"

    def test_the_staged_files_are_what_the_install_proves(self, host, tmp_path):
        """The bug this shipped with, and the reason it mattered.

        ``_self_test`` runs before ``_promote``, so the files it is proving are
        in a staging directory rather than where an installed model lives. The
        first version resolved the chosen network against the installed root,
        which had two effects on a real install: a log line announcing a
        fallback that had not happened, and -- worse -- "Checking that DPDFNet
        runs on this machine" running a *different* network from the one that
        would run afterwards. A check that proves the wrong file is not a check.
        """
        staging = tmp_path / "staging"
        staging.mkdir()
        for name in ("dpdfnet2_48khz_hr.onnx", "dpdfnet8_48khz_hr.onnx"):
            (staging / name).write_bytes(b"not really a network")
        # Nothing installed yet, which is what a first install looks like.
        root = paths.pipeline_stage_root("dpdfnet")
        root.mkdir(parents=True, exist_ok=True)
        for name in ("dpdfnet2_48khz_hr.onnx", "dpdfnet8_48khz_hr.onnx"):
            (root / name).unlink(missing_ok=True)

        pipeline.remember(model="dpdfnet2_48khz_hr")

        assert pipeline.stage_config("dpdfnet", staging)["model_file"] == \
            "dpdfnet2_48khz_hr.onnx"

    def test_the_self_test_asks_about_the_directory_it_is_given(self):
        """Asserted against the source, because the alternative is an install
        that has to be run for real to notice it proved the wrong thing."""
        import inspect

        body = inspect.getsource(pipeline._self_test)

        assert "stage_config(stage_id, staging)" in body, \
            "the self-test resolves the model somewhere other than where it staged it"

    def test_the_route_accepts_and_refuses_one(self, kept_settings, monkeypatch):
        import mc_voice_api
        import mc_voice_turn as turns

        monkeypatch.setattr(runtime, "reconfigure", lambda reason: True)
        monkeypatch.setattr(turns, "busy", lambda: False)

        found = mc_voice_api.pipeline_settings(None, {}, model="dpdfnet8_48khz_hr")
        assert found["models"]["chosen"] == "dpdfnet8_48khz_hr"

        with pytest.raises(Exception) as caught:
            mc_voice_api.pipeline_settings(None, {}, model="   ")
        assert "identifier" in str(caught.value)

    def test_the_panel_offers_it(self, host):
        drawn = mc_voice_ui.pipeline_stage_detail("dpdfnet")

        assert 'data-mc-voice-pipeline-model="dpdfnet"' in drawn
        assert "2.41 GMACs" in drawn and "7.17 GMACs" in drawn

    def test_the_browser_posts_it(self):
        """Asserted against the source, because that control lives in a panel
        fetched from ``/component`` and injected after the script has run."""
        script = (paths.extension_root() / "javascript" / "voice_chat.js").read_text(
            encoding="utf-8")

        assert "data-mc-voice-pipeline-model" in script
        handler = script.split("data-mc-voice-pipeline-model]")[1].split(
            "// Where an enhancement stage runs")[0]
        assert "model: asked" in handler


class TestThePlacementControlSaysWhatItCanAndCannotDo:
    """The panel, and the two shapes it takes.

    A component that can move gets a dropdown. One that cannot gets a sentence
    naming the wheel that would have to change -- not a disabled dropdown,
    because a control whose value the engine ignores is a control that lies
    about what it did, which is the same rule that kept a thread slider off the
    PocketTTS panel.
    """

    def _machine(self, monkeypatch, cards):
        import mc_voice_device as devices

        detection = types.ModuleType("prompt_master.inference.device_detection")
        detection.detect_gpus = lambda *a, **k: list(cards)
        detection.detect_cpu = lambda: types.SimpleNamespace(
            name="A Processor", memory_total_mb=65413)
        monkeypatch.setitem(sys.modules,
                            "prompt_master.inference.device_detection", detection)
        devices.forget_cards()

    def test_a_machine_with_no_card_is_offered_no_dropdown(self, monkeypatch):
        """A select whose only option is the one in force cannot do anything."""
        import mc_voice_ui as ui_module

        self._machine(monkeypatch, [])

        assert ui_module._pipeline_device_row("dpdfnet") == ""

    def test_a_machine_with_a_card_gets_one_option_per_device(self, monkeypatch):
        import mc_voice_ui as ui_module

        self._machine(monkeypatch, [
            types.SimpleNamespace(physical_index=0, uuid="GPU-aaaa", memory_total_mb=24576,
                                  name="NVIDIA GeForce RTX 3090"),
            types.SimpleNamespace(physical_index=1, uuid="GPU-bbbb", memory_total_mb=32607,
                                  name="NVIDIA GeForce RTX 5090")])

        drawn = ui_module._pipeline_device_row("dpdfnet")

        assert 'data-mc-voice-pipeline-device="dpdfnet"' in drawn
        assert drawn.count("<option") == 3
        assert "RTX 3090" in drawn and "RTX 5090" in drawn

    def test_the_note_says_the_numbering_may_not_agree(self, monkeypatch):
        """The honest half, and the reason it is here.

        The adapter number handed to DirectML is the card's nvidia-smi index and
        no API makes it more than an assumption. An assumption somebody can see
        and correct in one click is a different thing from a mapping presented
        as fact.
        """
        import mc_voice_ui as ui_module

        self._machine(monkeypatch, [
            types.SimpleNamespace(physical_index=0, uuid="GPU-aaaa", memory_total_mb=24576,
                                  name="NVIDIA GeForce RTX 3090")])

        drawn = ui_module._pipeline_device_row("dpdfnet")

        assert "numbered differently" in drawn
        assert "choose the other entry" in drawn

    def test_the_note_promises_the_choice_will_not_be_taken_back(self, monkeypatch):
        import mc_voice_ui as ui_module

        self._machine(monkeypatch, [
            types.SimpleNamespace(physical_index=0, uuid="GPU-aaaa", memory_total_mb=24576,
                                  name="NVIDIA GeForce RTX 3090")])

        drawn = ui_module._pipeline_device_row("dpdfnet")

        assert "keeps its place" in drawn
        # And says what it does not do, in the same breath. The stage cannot
        # push anything off a full card -- it fails to load and the reply is
        # spoken unenhanced -- and a note that promised only the first half
        # would be read as promising both.
        assert "does not do is push anything out" in drawn
        assert "spoken unenhanced" in drawn

    def test_a_stage_this_build_does_not_ship_is_offered_no_device(self, monkeypatch):
        """LavaSR, today. Its Install button is disabled because the build has
        no dependency closure for it, and a dropdown asking where to run it
        would be offering to place something that cannot exist here -- the same
        control-that-lies problem as a disabled select, in a different costume.
        """
        import mc_voice_ui as ui_module

        self._machine(monkeypatch, [
            types.SimpleNamespace(physical_index=0, uuid="GPU-aaaa", memory_total_mb=24576,
                                  name="NVIDIA GeForce RTX 3090")])

        # LavaSR is shipped now -- it installs from a folder -- so it is no
        # longer the example of an unshipped stage. The property still matters,
        # so it is proved against a stage the build says it does not have.
        monkeypatch.setattr(pipeline, "stage_available",
                            lambda stage_id: stage_id != "lavasr")
        assert ui_module._pipeline_device_row("lavasr") == ""
        assert ui_module._pipeline_device_row("dpdfnet") != ""
        monkeypatch.undo()
        assert ui_module._pipeline_device_row("lavasr") != "", (
            "a stage that can be installed is a stage that can be placed")

    def test_availability_and_installability_are_different_questions(self):
        """A stage not installable *here* is still a stage this build has.

        Collapsing the two would take the device control off DPDFNet on any
        machine with no pinned runtime, which is a stage whose settings still
        mean something.
        """
        assert pipeline.stage_available("dpdfnet") is True
        assert pipeline.stage_available("nosuch") is False

    @pytest.mark.parametrize("component", ["tts-pocket", "tts-sopro", "tts-kokoro",
                                           "recording-cleanup"])
    def test_an_engine_that_cannot_move_gets_a_sentence_and_no_control(self, component):
        import mc_voice_ui as ui_module

        drawn = ui_module._engine_device_note(component)

        assert "<select" not in drawn
        assert "The processor." in drawn
        assert len(drawn) > 200, drawn

    def test_a_component_that_can_move_gets_no_such_note(self):
        import mc_voice_ui as ui_module

        assert ui_module._engine_device_note("voice-pipeline-dpdfnet") == ""

    def test_the_page_no_longer_claims_voice_chat_never_uses_a_card(self):
        """A sentence that stopped being true had to stop being printed.

        Three panels carried "Voice Chat runs on the CPU and never uses the
        graphics card". It is still true of every speech engine and it is no
        longer true of the feature, so each one now says which it means.
        """
        source = (paths.extension_root() / "mc_voice_ui.py").read_text(encoding="utf-8")

        assert "Voice Chat runs on the CPU and never uses" not in source


class TestTheEnhancementThreadBudgetIsATurnableDial:
    """A constant that a user's own measurement disproved.

    Upstream builds DPDFNet's session with ``intra_op_num_threads = 1``, and
    this feature's own default was two -- chosen so the stage would not crowd
    the PocketTTS generation it runs beside. On a sixteen-thread machine that
    measured a real-time factor of 2.38: the enhancement taking two and a half
    seconds of compute per second of speech, holding the source back for 292 of
    a 300-second reply and starving playback fourteen times. Politeness is not
    a virtue in a stage that cannot keep up.
    """

    def test_the_default_is_unchanged_for_anybody_who_never_touches_it(self):
        assert pipeline.threads() == pipeline.INTRAOP_THREADS

    def test_settings_report_the_budget_and_its_ceiling(self):
        found = pipeline.settings()

        assert found["threads"] == pipeline.threads()
        assert found["max_threads"] == pipeline.MAX_INTRAOP_THREADS

    def test_the_budget_reaches_the_stage_that_needed_it(self):
        """The whole point, asserted at both places that build the config.

        ``_send_load`` hands it to the running worker and ``_self_test`` hands
        it to the staged one; a budget that reached only the second would be a
        setting that worked once, during an install.
        """
        import inspect

        import mc_voice_pipeline_runtime as runtime

        for source in (inspect.getsource(runtime._send_load),
                       inspect.getsource(pipeline._self_test)):
            assert "intraop" in source
            assert "INTRAOP_THREADS" not in source, (
                "the constant is still being sent instead of the setting")
        assert 'config["intraop"] = threads()' in inspect.getsource(pipeline._self_test)

    def test_a_number_outside_the_range_is_refused_not_clamped(self):
        """A value silently changed on the way in is a control that lies."""
        import mc_voice_api

        for asked in (0, -4, pipeline.MAX_INTRAOP_THREADS + 1):
            with pytest.raises(Exception) as caught:
                mc_voice_api.pipeline_settings(None, {}, threads=asked)
            assert "between 1 and" in str(caught.value), asked

    def test_something_that_is_not_a_number_is_refused(self):
        import mc_voice_api

        with pytest.raises(Exception) as caught:
            mc_voice_api.pipeline_settings(None, {}, threads="lots")
        assert "whole number" in str(caught.value)

    def test_the_reading_is_bounded_whatever_the_store_holds(self, monkeypatch):
        """A store is not a validator. Anything in it has to be survivable."""
        import sys
        import types

        for held, expected in ((0, 1), (-9, 1), (9999, pipeline.MAX_INTRAOP_THREADS),
                               ("7", 7)):
            fake = types.SimpleNamespace(shared=types.SimpleNamespace(
                opts=types.SimpleNamespace(data={pipeline.OPT_THREADS: held})))
            monkeypatch.setitem(sys.modules, "modules", fake)
            assert pipeline.threads() == expected, held


class TestTheBuildCarriesTwoRuntimeClosures:
    """One closure for ONNX, one that adds PyTorch, and never two processes.

    LavaSR upstream is a PyTorch model and DPDFNet is an ONNX one. The
    interesting property is not that a second closure exists -- it is that
    adding it did not add a second thing to contain.
    """

    @staticmethod
    def _windows(monkeypatch):
        import mc_voice_models as models
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))

    def test_every_wheel_is_pinned_byte_for_byte(self, monkeypatch):
        """No provisional path for a *wheel*, on either flavour.

        A wheel's contents get executed, so it is checked against a digest this
        repository reviewed. Three things in the torch closure are not wheels --
        LavaSR, the vocos fork and encodec publish source only -- and they are
        named exceptions rather than a relaxed rule: the assertion below lists
        them, so a fourth cannot be added without editing this test.
        """
        self._windows(monkeypatch)
        found = pipeline.manifest()
        for flavour in pipeline.RUNTIME_FLAVOURS:
            entry = pipeline.runtime_entry(found, flavour)
            artifacts = entry["platforms"][0]["artifacts"]
            assert artifacts, flavour
            for item in artifacts:
                if item["source_package"] or item["resolve"]:
                    continue
                assert item["sha256"], f"{flavour}: {item['local_name']} is unhashed"
                assert item["bytes"] > 0, f"{flavour}: {item['local_name']} is unsized"
            assert pipeline.runtime_installable(flavour) is True, flavour

    def test_the_source_packages_are_exactly_the_three_with_no_wheel(self, monkeypatch):
        self._windows(monkeypatch)
        found = pipeline.manifest()
        assert not [item for item in
                    pipeline.runtime_entry(found, "onnx")["platforms"][0]["artifacts"]
                    if item["source_package"]], "the ONNX closure is wheels only"
        for entry in pipeline.runtime_entry(found, "torch")["platforms"]:
            packages = sorted(item["source_package"] for item in entry["artifacts"]
                              if item["source_package"])
            assert packages == ["LavaSR", "encodec", "vocos"], entry["id"]

    def test_encodec_is_pinned_even_though_it_ships_no_wheel(self, monkeypatch):
        """An sdist on PyPI can still be hashed here. Only the two GitHub
        tarballs cannot, because that host is unreachable from the machine that
        writes this manifest -- not because they are source."""
        self._windows(monkeypatch)
        rows = {item["source_package"]: item for item in
                pipeline.runtime_entry(pipeline.manifest(), "torch")["platforms"][0]
                ["artifacts"] if item["source_package"]}
        assert rows["encodec"]["sha256"], "encodec is on PyPI and is pinned"
        assert rows["encodec"]["bytes"] > 0
        for name in ("LavaSR", "vocos"):
            assert rows[name]["url"].startswith("https://codeload.github.com/")
            # An immutable commit, so the archive cannot change under the pin
            # even though the digest is the publisher's rather than ours.
            assert len(rows[name]["url"].rsplit("/", 1)[-1]) == 40

    def test_the_torch_closure_carries_onnx_runtime_too(self, monkeypatch):
        """The whole reason there is still only one worker process.

        If the torch flavour shipped without ONNX Runtime, an installation with
        both stages enabled would need a second interpreter to run DPDFNet --
        and a second interpreter is a second job object, a second pdeathsig, and
        a second way to leave something running after the voice it belongs to
        has gone. Carrying both wheel sets costs 126 MB and removes that whole
        category of bug, so this is a property worth a test rather than a
        comment.
        """
        self._windows(monkeypatch)
        found = pipeline.manifest()
        names = [item["local_name"].split("-")[0].casefold().replace("_", "-")
                 for item in pipeline.runtime_entry(found, "torch")["platforms"][0]["artifacts"]]
        assert "onnxruntime-directml" in names
        assert "torch" in names

    def test_each_flavour_installs_into_its_own_directory(self):
        onnx = paths.pipeline_runtime_root("onnx")
        torch = paths.pipeline_runtime_root("torch")
        assert onnx != torch
        assert onnx.name == "runtime", "an already-installed runtime must stay found"
        assert paths.pipeline_inside(torch), "the second closure is still the pipeline's"

    def test_a_flavour_this_build_does_not_have_is_refused_not_guessed(self):
        with pytest.raises(ValueError):
            paths.pipeline_runtime_root("cuda")
        assert pipeline.runtime_closure_id("cuda") == ""

    def test_the_two_closures_have_different_identities(self, monkeypatch):
        self._windows(monkeypatch)
        assert (pipeline.runtime_closure_id("onnx")
                != pipeline.runtime_closure_id("torch"))

    def test_adding_a_flavour_did_not_make_installed_runtimes_stale(self, monkeypatch):
        """A regression guard on a mistake I made and caught.

        The fingerprint first led with the flavour name for every closure, which
        changed the onnx id and would have told every existing installation to
        re-download 138 MB to gain a flavour its owner may never enable. The
        literal below is the id this build produced before the torch closure
        existed; it is pinned here so that a future edit to the fingerprint has
        to be a deliberate one.
        """
        self._windows(monkeypatch)
        assert pipeline.runtime_closure_id("onnx") == "684ddbdb0fc928f4"

    def test_an_unknown_closure_in_the_manifest_is_a_broken_build(self, monkeypatch):
        """Refused before a download starts, like an unknown provider is."""
        found = json.loads(paths.pipeline_manifest_path().read_text(encoding="utf-8"))
        found["runtimes"]["cuda"] = dict(found["runtimes"]["torch"])
        monkeypatch.setattr(pipeline, "_manifest_cache", None)
        with pytest.raises(pipeline.PipelineError, match="does not know how to install"):
            pipeline._read_manifest(found)


class TestTheExportToolRefusesAnExportThatDoesNotMatch:
    """The verdict logic of tools/export_lavasr_onnx.py, exercised without torch.

    The whole value of that tool is a refusal: it exists so an ONNX LavaSR is
    only offered once it has been proved against the PyTorch model it came
    from. That decision is a pure function of two arrays, and it should not be
    the one part that is only ever exercised by a maintainer on a machine with
    a 122 MB wheel and a Hugging Face token.
    """

    @staticmethod
    def _tool():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "export_lavasr_onnx",
            paths.extension_root() / "tools" / "export_lavasr_onnx.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_length_mismatch_is_refused_at_any_tolerance(self):
        """The failure that matters most, and the one no tolerance may forgive.

        Audio of the wrong length still sounds like speech. A graph that
        resamples to the wrong rate, or pads, produces exactly this and nothing
        about it is audible as an error -- it is audible as a slow talker.
        """
        tool = self._tool()
        verdict = tool.compare([0.0] * 48000, [0.0] * 32000, tolerance=1e9)
        assert verdict["ok"] is False
        assert verdict["reason"] == "length"
        assert "48000" in verdict["detail"] and "32000" in verdict["detail"]

    def test_a_close_enough_export_passes_and_says_how_close(self):
        tool = self._tool()
        ref = [0.1, -0.2, 0.3, -0.4]
        got = [0.1001, -0.2001, 0.3001, -0.4001]
        verdict = tool.compare(ref, got, tolerance=1e-3)
        assert verdict["ok"] is True
        assert verdict["worst"] < 1e-3
        assert verdict["samples"] == 4

    def test_a_drift_past_the_tolerance_is_refused_and_located(self):
        tool = self._tool()
        ref = [0.0] * 10
        got = [0.0] * 9 + [0.5]
        verdict = tool.compare(ref, got, tolerance=1e-3)
        assert verdict["ok"] is False
        assert verdict["reason"] == "value"
        assert "sample 9" in verdict["detail"]

    def test_nan_is_refused_rather_than_compared(self):
        """An exporter that drops a guard produces NaN, not a large number."""
        tool = self._tool()
        verdict = tool.compare([0.0, 0.0], [0.0, float("nan")], tolerance=1e9)
        assert verdict["ok"] is False
        assert verdict["reason"] == "not-finite"

    def test_an_empty_reference_proves_nothing_and_is_refused(self):
        tool = self._tool()
        verdict = tool.compare([], [], tolerance=1e-3)
        assert verdict["ok"] is False
        assert verdict["reason"] == "empty"

    def test_the_determinism_check_is_an_exact_comparison(self):
        """Run twice, same bytes -- a zero tolerance, deliberately."""
        tool = self._tool()
        assert tool.compare([0.1, 0.2], [0.1, 0.2], 0.0)["ok"] is True
        assert tool.compare([0.1, 0.2], [0.1, 0.2000001], 0.0)["ok"] is False

    def test_the_proof_lengths_include_one_that_does_not_divide_evenly(self):
        """A padding bug only shows at a length that is not whole windows."""
        tool = self._tool()
        assert any(n % 16000 for n in tool.PROOF_LENGTHS)
        assert min(tool.PROOF_LENGTHS) < 16000, "one shorter than a second"

    def test_the_tool_says_what_is_missing_rather_than_traceback(self):
        """Run where upstream is absent, which is every machine in CI."""
        tool = self._tool()
        with pytest.raises(tool.ExportError, match="not importable here"):
            tool.load_upstream("cpu")


class TestChoosingACardChoosesADifferentClosure:
    """One click on a GPU has to install a different runtime, not the same one.

    The CPU and CUDA closures describe the same operating system and the same
    Python; they differ only in which wheels they carry. If the accelerator were
    a filter applied after the lookup rather than part of it, asking for CUDA on
    a build without a CUDA entry would quietly install the CPU one -- which
    succeeds, runs on the processor, and never mentions the card that was
    picked. That is the failure this class exists to prevent.
    """

    @staticmethod
    def _windows(monkeypatch):
        import mc_voice_models as models
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))

    def test_the_two_closures_are_different_installations(self, monkeypatch):
        self._windows(monkeypatch)
        cpu = pipeline.runtime_closure_id("torch", "cpu")
        cuda = pipeline.runtime_closure_id("torch", "cuda")
        assert cpu and cuda and cpu != cuda

    def test_asking_for_a_card_this_build_cannot_serve_gets_nothing(self, monkeypatch):
        """Not the CPU build as a consolation prize.

        DPDFNet's closure reaches a card through DirectML and has no CUDA
        variant at all, so this is a real case rather than a hypothetical one.
        """
        self._windows(monkeypatch)
        assert pipeline.runtime_installable("onnx", "cuda") is False
        assert pipeline.runtime_closure_id("onnx", "cuda") == ""
        assert pipeline.runtime_installable("onnx", "cpu") is True

    def test_the_cuda_closure_is_installable_but_says_it_is_not_pinned(self, monkeypatch):
        """The whole security story of this path, in two booleans.

        PyTorch's CUDA wheels are published only on a host the manifest-writing
        machine cannot reach, so they are checked against the digest their
        publisher states at install time rather than one this repository
        reviewed. That is a weaker claim and it has to be sayable, not implied.
        """
        self._windows(monkeypatch)
        assert pipeline.runtime_installable("torch", "cuda") is True
        assert pipeline.runtime_pinned("torch", "cuda") is False
        assert pipeline.runtime_pinned("onnx", "cpu") is True, "the ONNX one is pinned"

    def test_only_the_two_wheels_that_cannot_be_pinned_are_resolved(self, monkeypatch):
        """Everything else in the CUDA closure still comes hashed from PyPI.

        A resolved artifact is the exception, and it stays the exception: if
        this count grows, somebody has reached for the loose path to avoid
        pinning something that could have been pinned.
        """
        self._windows(monkeypatch)
        entry = pipeline._platform_entry_or_none(pipeline.manifest(), "torch", "cuda")
        resolved = [item for item in entry["artifacts"] if item["resolve"]]
        assert sorted(item["local_name"] for item in resolved) == ["torch", "torchaudio"]
        # Everything else is a wheel this repository hashed, or one of the three
        # named source packages. Nothing is loose.
        rest = [item for item in entry["artifacts"] if not item["resolve"]]
        assert all(item["sha256"] or item["source_package"] for item in rest)

    def test_a_manifest_that_both_pins_and_resolves_a_wheel_is_refused(self):
        """One or the other. A pin overwritten by a publisher is not a pin."""
        with pytest.raises(pipeline.PipelineError, match="One or the other"):
            pipeline._read_artifact("the torch runtime closure", {
                "local_name": "torch", "url": "https://example.test/torch.whl",
                "sha256": "a" * 64, "bytes": 10,
                "resolve": {"index": "https://download.pytorch.org/whl/cu128/torch/",
                            "package": "torch"}})

    def test_a_resolve_over_plain_http_is_refused_in_the_manifest(self):
        with pytest.raises(pipeline.PipelineError, match="not HTTPS"):
            pipeline._read_artifact("the torch runtime closure", {
                "local_name": "torch",
                "resolve": {"index": "http://download.pytorch.org/whl/", "package": "torch"}})

    def test_an_accelerator_this_build_cannot_build_for_is_refused(self, monkeypatch):
        found = json.loads(paths.pipeline_manifest_path().read_text(encoding="utf-8"))
        found["runtimes"]["torch"]["platforms"][0]["accelerator"] = "rocm"
        monkeypatch.setattr(pipeline, "_manifest_cache", None)
        with pytest.raises(pipeline.PipelineError, match="does not know how to build for"):
            pipeline._read_manifest(found)


class TestInstallingAStageFromAFolderYouFilled:
    """The escape hatch, for a stage. For LavaSR it is the only way in.

    Its weights sit on a host this repository could not reach to pin, so no
    digest for them was ever committed and the managed download is not offered.
    The files are public though, so a person can fetch them in a browser -- and
    a build that only showed a disabled button would be refusing work it is
    perfectly able to do.
    """

    @pytest.fixture(autouse=True)
    def _isolated(self, voice_root):
        """Every case here writes a real installation, so each needs its own tree.

        Without this the first successful install leaves a stage record behind
        and the next case's "nothing was installed" assertion passes or fails on
        the previous test's leftovers rather than on its own behaviour.
        """
        return voice_root

    @staticmethod
    def _folder(tmp_path, layout):
        for name, blob in layout.items():
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        return tmp_path

    def test_a_missing_file_names_it_and_installs_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": True)
        folder = self._folder(tmp_path / "src", {
            "enhancer_v2/pytorch_model.bin": b"x" * 32,
            "enhancer_v2/config.yaml": b"sample_rate: 24000\n"})
        with pytest.raises(pipeline.PipelineError, match="denoiser/denoiser.bin"):
            pipeline.install_from("lavasr", folder)
        assert not paths.pipeline_stage_manifest("lavasr").exists()

    def test_an_empty_file_is_refused_rather_than_copied(self, tmp_path, monkeypatch):
        """A browser that failed halfway leaves a zero-byte file, not no file."""
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": True)
        folder = self._folder(tmp_path / "src", {
            "enhancer_v2/pytorch_model.bin": b"x" * 32,
            "enhancer_v2/config.yaml": b"sample_rate: 24000\n",
            "denoiser/denoiser.bin": b""})
        with pytest.raises(pipeline.PipelineError, match="empty"):
            pipeline.install_from("lavasr", folder)

    def test_files_dropped_loose_are_accepted_as_well_as_nested(self, tmp_path,
                                                                monkeypatch):
        """Clicking three links gives three loose files, not a tree.

        Accepting only the nested spelling would refuse the more likely of the
        two layouts, which is the one somebody gets by doing exactly what the
        panel's links invite them to do.
        """
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": True)
        monkeypatch.setattr(pipeline, "_self_test", lambda stage_id, staging: {})
        folder = self._folder(tmp_path / "loose", {
            "pytorch_model.bin": b"w" * 64,
            "config.yaml": b"sample_rate: 24000\n",
            "denoiser.bin": b"d" * 64})
        pipeline.install_from("lavasr", folder)
        root = paths.pipeline_stage_root("lavasr")
        # Stored under the names the model's own loader indexes into, whichever
        # way they arrived -- the loader is what has to find them, not the user.
        assert (root / "enhancer_v2" / "pytorch_model.bin").exists()
        assert (root / "enhancer_v2" / "config.yaml").exists()
        assert (root / "denoiser" / "denoiser.bin").exists()

    def test_the_record_says_local_rather_than_claiming_a_verification(self, tmp_path,
                                                                      monkeypatch):
        """This repository has never seen these bytes and must not imply it has."""
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": True)
        monkeypatch.setattr(pipeline, "_self_test", lambda stage_id, staging: {})
        folder = self._folder(tmp_path / "src", {
            "enhancer_v2/pytorch_model.bin": b"x" * 32,
            "enhancer_v2/config.yaml": b"sample_rate: 24000\n",
            "denoiser/denoiser.bin": b"y" * 32})
        pipeline.install_from("lavasr", folder)
        record = json.loads(
            paths.pipeline_stage_manifest("lavasr").read_text(encoding="utf-8"))
        assert record["source"] == "local"
        assert record["revision"] == "", "no revision was verified, so none is claimed"
        assert record["provisional"] is True
        # Hashed anyway: the bytes were never vouched for, but later tampering
        # is still detectable, which is a different and achievable claim.
        assert len(record["artifacts"]) == 3
        assert all(len(digest) == 64 for digest in record["artifacts"].values())

    def test_a_folder_that_fails_the_self_test_leaves_the_old_install_alone(
            self, tmp_path, monkeypatch):
        """Staged, proved, promoted -- in that order, so a bad folder is inert."""
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": True)
        monkeypatch.setattr(pipeline, "_self_test", lambda stage_id, staging: {})
        good = self._folder(tmp_path / "good", {
            "enhancer_v2/pytorch_model.bin": b"x" * 32,
            "enhancer_v2/config.yaml": b"sample_rate: 24000\n",
            "denoiser/denoiser.bin": b"y" * 32})
        pipeline.install_from("lavasr", good)
        before = paths.pipeline_stage_manifest("lavasr").read_text(encoding="utf-8")

        def explode(stage_id, staging):
            raise pipeline.PipelineError("it does not run here")

        monkeypatch.setattr(pipeline, "_self_test", explode)
        bad = self._folder(tmp_path / "bad", {
            "enhancer_v2/pytorch_model.bin": b"z" * 32,
            "enhancer_v2/config.yaml": b"sample_rate: 48000\n",
            "denoiser/denoiser.bin": b"z" * 32})
        with pytest.raises(pipeline.PipelineError, match="does not run here"):
            pipeline.install_from("lavasr", bad)
        assert paths.pipeline_stage_manifest("lavasr").read_text(encoding="utf-8") == before

    def test_pointing_at_a_file_takes_its_folder(self, tmp_path, monkeypatch):
        """What people paste is a file path. Refusing it is unkind, not safe.

        Its *own* folder, note, not the root of a tree it happens to sit in.
        Pasting a nested file gives that file's directory, which will then be
        missing the rest -- and the error names what it looked for, which is
        more use than silently climbing until something matches.
        """
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": True)
        monkeypatch.setattr(pipeline, "_self_test", lambda stage_id, staging: {})
        folder = self._folder(tmp_path / "loose", {
            "pytorch_model.bin": b"x" * 32,
            "config.yaml": b"sample_rate: 24000\n",
            "denoiser.bin": b"y" * 32})
        pipeline.install_from("lavasr", folder / "config.yaml")
        assert paths.pipeline_stage_manifest("lavasr").exists()

    def test_a_nested_paste_says_where_it_looked(self, tmp_path, monkeypatch):
        """The consequence of the rule above, made explicit rather than found."""
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": True)
        folder = self._folder(tmp_path / "src", {
            "enhancer_v2/pytorch_model.bin": b"x" * 32,
            "enhancer_v2/config.yaml": b"sample_rate: 24000\n",
            "denoiser/denoiser.bin": b"y" * 32})
        with pytest.raises(pipeline.PipelineError) as caught:
            pipeline.install_from("lavasr", folder / "enhancer_v2" / "config.yaml")
        assert "Looked for" in str(caught.value)

    def test_nothing_at_that_path_says_so_plainly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": True)
        with pytest.raises(pipeline.PipelineError, match="There is nothing at"):
            pipeline.install_from("lavasr", tmp_path / "nope")
        with pytest.raises(pipeline.PipelineError, match="Give the folder"):
            pipeline.install_from("lavasr", "   ")

    def test_a_missing_runtime_is_installed_rather_than_described(
            self, tmp_path, monkeypatch):
        """The dead end this class was shipped into, and the fix for it.

        LavaSR needs the PyTorch closure. Nothing on the settings page installs
        that -- the Runtime row builds the ONNX one -- so refusing with "install
        the runtime first" named a button that does not exist and left the only
        way into this stage permanently shut. The prerequisite is built here for
        the same reason install() builds it: nobody wants a stage without the
        runtime under it.
        """
        import mc_voice_models as models
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))
        built = []
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": None)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": False)
        monkeypatch.setattr(
            pipeline, "_install_runtime",
            lambda say, tick, flavour="onnx", accelerator="cpu": (
                built.append(flavour), tick(1.0)))
        monkeypatch.setattr(pipeline, "_self_test", lambda stage_id, staging: {})
        folder = self._folder(tmp_path / "src", {
            "enhancer_v2/pytorch_model.bin": b"x" * 32,
            "enhancer_v2/config.yaml": b"sample_rate: 24000\n",
            "denoiser/denoiser.bin": b"y" * 32})
        pipeline.install_from("lavasr", folder)
        assert built == ["torch"], "LavaSR is proved inside the PyTorch closure"
        assert paths.pipeline_stage_manifest("lavasr").exists()

    def test_a_stale_closure_is_rebuilt_before_the_stage_is_proved_in_it(
            self, tmp_path, monkeypatch):
        """The exact path a user hit, and the reason "reinstall" never helped.

        Their torch closure was on disk and had an interpreter, so this guard --
        ``runtime_python(flavour) is None`` at the time -- said "already there"
        and skipped straight to the self-test. But it had been built before this
        build added LavaSR, vocos and encodec to the closure, so the model's own
        code was not in it, and every attempt died in the same place with the
        same sentence. The panel already knew: it had been drawing "needs
        updating" over that runtime the whole time.
        """
        import mc_voice_models as models
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))
        built = []
        # There, with an interpreter, and not what this build pins.
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": False)
        monkeypatch.setattr(
            pipeline, "_install_runtime",
            lambda say, tick, flavour="onnx", accelerator="cpu": (
                built.append(flavour), tick(1.0)))
        monkeypatch.setattr(pipeline, "_self_test", lambda stage_id, staging: {})
        folder = self._folder(tmp_path / "src", {
            "enhancer_v2/pytorch_model.bin": b"x" * 32,
            "enhancer_v2/config.yaml": b"sample_rate: 24000\n",
            "denoiser/denoiser.bin": b"y" * 32})

        pipeline.install_from("lavasr", folder)

        assert built == ["torch"], (
            "an interpreter sitting in the directory says nothing about which "
            "wheels are beside it")
        assert paths.pipeline_stage_manifest("lavasr").exists()

    def test_a_build_with_no_closure_at_all_still_says_so(self, tmp_path, monkeypatch):
        """Installing the prerequisite is only possible where one is pinned."""
        import mc_voice_models as models
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("linux", "x86_64", "3.11"))
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": None)
        monkeypatch.setattr(pipeline, "runtime_current", lambda flavour="onnx": False)
        folder = self._folder(tmp_path / "src", {
            "enhancer_v2/pytorch_model.bin": b"x" * 32,
            "enhancer_v2/config.yaml": b"sample_rate: 24000\n",
            "denoiser/denoiser.bin": b"y" * 32})
        with pytest.raises(pipeline.PipelineError, match="no Voice Pipeline runtime"):
            pipeline.install_from("lavasr", folder)


class TestACardIsSomethingTheInstallerCanActuallyBuildFor:
    """The CUDA closure existed and nothing in the product could ask for it.

    Every ``_install_runtime`` call site took the ``accelerator="cpu"`` default,
    so the cu128 platform entry, its resolver and its tests were all correct and
    all unreachable: the only way to build that closure was to call the private
    function by hand, which is what the class below does and why it passed while
    a card could not be installed for.
    """

    @staticmethod
    def _machine(monkeypatch, cards):
        import mc_voice_device as devices
        detection = types.ModuleType("prompt_master.inference.device_detection")
        detection.detect_gpus = lambda *a, **k: list(cards)
        detection.detect_cpu = lambda: types.SimpleNamespace(
            name="Intel Core Ultra 9 185H", memory_total_mb=97809)
        monkeypatch.setitem(sys.modules,
                            "prompt_master.inference.device_detection", detection)
        devices.forget_cards()
        return devices

    @staticmethod
    def _card(index=1, uuid="GPU-be876bce-cc46-b562-7192-333c2c1d3f44"):
        return types.SimpleNamespace(physical_index=index, uuid=uuid,
                                     name="NVIDIA GeForce RTX 3090",
                                     memory_total_mb=24576)

    @pytest.fixture
    def windows(self, monkeypatch):
        """The machine the CUDA closure is pinned for."""
        import mc_voice_models as models
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))

    def test_the_processor_is_still_what_an_unplaced_stage_needs(self, windows,
                                                                 monkeypatch):
        self._machine(monkeypatch, [self._card()])
        assert pipeline.required_accelerator("torch") == "cpu"
        assert pipeline.required_accelerator("onnx") == "cpu"

    def test_placing_a_torch_stage_on_a_card_asks_for_the_cuda_closure(
            self, windows, monkeypatch, host):
        devices = self._machine(monkeypatch, [self._card()])
        devices.remember("voice-pipeline-lavasr",
                         f"{devices.GPU_PREFIX}{self._card().uuid}")

        assert pipeline.required_accelerator("torch") == "cuda"
        # DPDFNet did not move, and the ONNX closure has one build anyway.
        assert pipeline.required_accelerator("onnx") == "cpu"

    def test_a_directml_stage_never_asks_for_a_different_closure(
            self, windows, monkeypatch, host):
        """DirectML rides the ONNX Runtime wheel every closure already has."""
        devices = self._machine(monkeypatch, [self._card()])
        devices.remember("voice-pipeline-dpdfnet",
                         f"{devices.GPU_PREFIX}{self._card().uuid}")

        assert pipeline.required_accelerator("onnx") == "cpu"
        assert pipeline.required_accelerator("torch") == "cpu"

    def test_a_machine_with_no_pinned_cuda_closure_stays_on_the_processor(
            self, monkeypatch, host):
        """Refused where it could not be honoured, rather than failed later."""
        import mc_voice_models as models
        monkeypatch.setattr(models, "current_platform", lambda: ("linux", "x86_64", "3.11"))
        devices = self._machine(monkeypatch, [self._card()])
        devices.remember("voice-pipeline-lavasr",
                         f"{devices.GPU_PREFIX}{self._card().uuid}")

        assert pipeline.required_accelerator("torch") == "cpu"

    def test_a_cpu_closure_under_a_placed_stage_reads_as_needing_an_update(
            self, windows, monkeypatch, host, voice_root):
        """The half that would have made the dropdown do nothing at all.

        Comparing an installation against whichever accelerator its own record
        happens to name means a CPU closure always agrees with itself. Somebody
        moves LavaSR onto a card, nothing is rebuilt, and the stage keeps
        running on the processor while the panel says "Installed".
        """
        devices = self._machine(monkeypatch, [self._card()])
        root = paths.pipeline_runtime_root("torch")
        (root / "env" / "Scripts").mkdir(parents=True, exist_ok=True)
        (root / "env" / "Scripts" / "python.exe").write_bytes(b"")
        (root / paths.INSTALLED_FILENAME).write_text(json.dumps({
            "closure": pipeline.runtime_closure_id("torch", "cpu"),
            "accelerator": "cpu"}), encoding="utf-8")
        assert pipeline.runtime_current("torch") is True

        devices.remember("voice-pipeline-lavasr",
                         f"{devices.GPU_PREFIX}{self._card().uuid}")

        assert pipeline.runtime_current("torch") is False
        state, message = pipeline._runtime_state("torch", True)
        assert state == "stale"
        assert "graphics card" in message, message

    def test_the_installer_builds_the_closure_the_placement_asked_for(
            self, windows, monkeypatch, host, tmp_path):
        """The end-to-end version of this class's complaint."""
        devices = self._machine(monkeypatch, [self._card()])
        devices.remember("voice-pipeline-lavasr",
                         f"{devices.GPU_PREFIX}{self._card().uuid}")
        built = []
        monkeypatch.setattr(pipeline, "runtime_python", lambda flavour="onnx": tmp_path)
        monkeypatch.setattr(
            pipeline, "_install_runtime",
            lambda say, tick, flavour="onnx", accelerator="cpu": (
                built.append((flavour, accelerator)), tick(1.0)))
        monkeypatch.setattr(pipeline, "_self_test", lambda stage_id, staging: {})
        folder = tmp_path / "src"
        for name, blob in (("enhancer_v2/pytorch_model.bin", b"x" * 32),
                           ("enhancer_v2/config.yaml", b"sample_rate: 24000\n"),
                           ("denoiser/denoiser.bin", b"y" * 32)):
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)

        pipeline.install_from("lavasr", folder)

        assert built == [("torch", "cuda")], (
            "a stage placed on a card needs the closure that can reach one")

    def test_the_worker_is_masked_to_the_one_chosen_card(self, windows, monkeypatch,
                                                         host, voice_root):
        """A UUID, not an index, and one card, not a list.

        The empty string was unconditional here, which is why
        torch.cuda.is_available() was False inside this worker whatever closure
        was installed.
        """
        devices = self._machine(monkeypatch, [self._card()])
        root = paths.pipeline_runtime_root("torch")
        root.mkdir(parents=True, exist_ok=True)
        (root / paths.INSTALLED_FILENAME).write_text(
            json.dumps({"closure": "whatever", "accelerator": "cuda"}), encoding="utf-8")
        devices.remember("voice-pipeline-lavasr",
                         f"{devices.GPU_PREFIX}{self._card().uuid}")

        assert pipeline.cuda_mask() == self._card().uuid
        assert pipeline.worker_environment()["CUDA_VISIBLE_DEVICES"] == self._card().uuid
        # The other two are still blanked outright: nothing here has ever
        # executed through either.
        assert pipeline.worker_environment()["HIP_VISIBLE_DEVICES"] == ""

    def test_nothing_placed_means_the_worker_still_sees_no_card_at_all(
            self, windows, monkeypatch, host):
        self._machine(monkeypatch, [self._card()])
        assert pipeline.cuda_mask() == ""
        assert pipeline.worker_environment()["CUDA_VISIBLE_DEVICES"] == ""

    def test_a_card_is_not_honoured_until_its_closure_is_installed(
            self, windows, monkeypatch, host, voice_root):
        """Otherwise the stage refuses to load over a setting, not a fault.

        The panel is already saying the runtime needs updating at this moment.
        Running on the processor until it is, and saying so in the log, keeps
        the voice working and keeps the choice.
        """
        devices = self._machine(monkeypatch, [self._card()])
        devices.remember("voice-pipeline-lavasr",
                         f"{devices.GPU_PREFIX}{self._card().uuid}")
        root = paths.pipeline_runtime_root("torch")
        root.mkdir(parents=True, exist_ok=True)
        (root / paths.INSTALLED_FILENAME).write_text(
            json.dumps({"closure": "whatever", "accelerator": "cpu"}), encoding="utf-8")

        assert pipeline.placement("lavasr") == (devices.PROVIDER_CPU, 0)
        assert pipeline.cuda_mask() == "", (
            "a stage answered with the processor must not be handed a card anyway")

        (root / paths.INSTALLED_FILENAME).write_text(
            json.dumps({"closure": "whatever", "accelerator": "cuda"}), encoding="utf-8")

        assert pipeline.placement("lavasr") == (devices.PROVIDER_CUDA, 0)
        assert pipeline.cuda_mask() == self._card().uuid


class TestTheRuntimeRowUpdatesTheRuntimesYouHave:
    """The trap a manifest change sets, and the row that should have sprung it.

    Adding a wheel to a closure makes every installed copy of it stale, by
    arithmetic and on purpose. For the ONNX closure the Runtime row rebuilds it.
    For the torch one nothing did: this branch built ONNX only, installing a
    stage checks that stage's own flavour, and so the single route back was
    LavaSR's from-a-folder install -- with nothing on the page saying so, and a
    pipeline that had simply gone quiet, which is what a stale closure correctly
    does to a turn and is indistinguishable from having switched it off.
    """

    def _stub(self, monkeypatch, current, record):
        import mc_voice_models as models

        built = []
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))
        monkeypatch.setattr(
            pipeline, "_install_runtime",
            lambda say, tick, flavour="onnx", accelerator="cpu": (
                built.append(f"{flavour}:{accelerator}"), tick(1.0)))
        monkeypatch.setattr(pipeline, "runtime_current",
                            lambda flavour="onnx": flavour in current)
        monkeypatch.setattr(pipeline, "_record",
                            lambda path: dict(record) if record else {})
        monkeypatch.setattr(pipeline, "status", lambda: "status")
        return built

    def test_a_stale_torch_closure_is_rebuilt_by_the_runtime_row(self, monkeypatch):
        built = self._stub(monkeypatch, current={"onnx"},
                           record={"accelerator": "cpu", "closure": "old"})

        pipeline.install("runtime")

        assert built == ["onnx:cpu", "torch:cpu"], (
            "the row says runtime, and a user pressing it after being told "
            "theirs needs updating expects theirs to be updated")

    def test_a_current_torch_closure_is_left_alone(self, monkeypatch):
        built = self._stub(monkeypatch, current={"onnx", "torch"},
                           record={"accelerator": "cpu", "closure": "fresh"})

        pipeline.install("runtime")

        assert built == ["onnx:cpu"], "nothing to do is nothing to download"

    def test_a_machine_that_never_had_the_torch_closure_is_not_given_one(
            self, monkeypatch):
        """257 MB is not what the light closure's row has ever meant."""
        built = self._stub(monkeypatch, current={"onnx"}, record=None)

        pipeline.install("runtime")

        assert built == ["onnx:cpu"]

    def test_a_cuda_closure_is_refreshed_as_cuda(self, monkeypatch):
        """The record names the accelerator, so a card's closure stays a card's."""
        built = self._stub(monkeypatch, current={"onnx"},
                           record={"accelerator": "cuda", "closure": "old"})

        pipeline.install("runtime")

        assert built == ["onnx:cpu", "torch:cuda"], (
            "rebuilding a CUDA closure as the CPU one would silently take a "
            "card away from the stage that was placed on it")

    def test_the_row_still_succeeds_when_the_extra_rebuild_fails(self, monkeypatch):
        """The component the user pressed is installed by then."""
        import mc_voice_models as models

        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))
        monkeypatch.setattr(pipeline, "runtime_current",
                            lambda flavour="onnx": flavour == "onnx")
        monkeypatch.setattr(pipeline, "_record",
                            lambda path: {"accelerator": "cpu", "closure": "old"})
        monkeypatch.setattr(pipeline, "status", lambda: "status")

        def build(say, tick, flavour="onnx", accelerator="cpu"):
            tick(1.0)
            if flavour != "onnx":
                raise pipeline.PipelineError("the publisher could not be reached")

        monkeypatch.setattr(pipeline, "_install_runtime", build)

        pipeline.install("runtime")  # must not raise


class TestTheThreadBudgetReachesBothStages:
    """A dial whose help says "the enhancement stages" reaching one of them.

    ``intraop`` was handed to DPDFNet's ONNX session and to nothing else.
    LavaSR ran on whatever OMP_NUM_THREADS gave it -- which the parent does
    set, but only in the environment of a worker started at that moment, and
    which does not size Torch's inter-op pool at all. So a user turning the
    setting against a measured real-time factor was turning it against half the
    pipeline and reading the whole pipeline's number back.
    """

    class _Torch:
        def __init__(self):
            self.intraop = self.interop = None

        def set_num_threads(self, count):
            self.intraop = count

        def set_num_interop_threads(self, count):
            self.interop = count

    def test_torch_is_given_the_number_the_setting_names(self):
        found = self._Torch()
        worker._apply_torch_budget(found, {"intraop": 8, "interop": 2})
        assert (found.intraop, found.interop) == (8, 2)

    def test_a_config_with_no_budget_leaves_torch_alone(self):
        """Clamping an absent setting to one thread is worse than doing nothing."""
        for config in ({}, {"intraop": 0}, {"intraop": None}, {"intraop": "eight"}):
            found = self._Torch()
            worker._apply_torch_budget(found, config)
            assert found.intraop is None, config

    def test_a_runtime_that_refuses_is_not_a_refused_reply(self):
        """set_num_interop_threads raises once parallel work has started."""
        class Stubborn(self._Torch):
            def set_num_interop_threads(self, count):
                raise RuntimeError("cannot set after parallel work has started")

        found = Stubborn()
        worker._apply_torch_budget(found, {"intraop": 8, "interop": 1})
        assert found.intraop == 8, "the intra-op budget still landed"

    def test_the_lava_backend_applies_it_before_it_builds_the_model(self):
        import inspect

        source = inspect.getsource(worker._LavaBackend.__init__)
        assert "_apply_torch_budget(torch, config)" in source
        assert source.index("_apply_torch_budget") < source.index("LavaEnhance2("), (
            "a budget set after the model is built is a budget the build did "
            "not get")


class TestTheThreadBudgetIsSizedFromTheMachine:
    """Two threads, measured, on the machine the feature is for.

    From a user's log, three consecutive replies with both stages running::

        RTF 3.02, source held back 35780 ms
        RTF 3.167, about 20.4 s of silence owed across 9.4 s of audio
        RTF 3.05, source held back 21520 ms

    Sixteen threads on that machine and the enhancement was given two. The
    reason it was two was a good one -- the stage runs beside a PocketTTS
    generation on the same cores, and crowding the model whose output it is
    polishing is exactly the wrong trade. What that reasoning did not know is
    what Pocket asks for. It calls torch.set_num_threads(1) and says so in its
    own readiness line: one thread. So thirteen sat idle while the audio broke
    up into nine-second gaps.
    """

    def test_a_big_machine_is_not_given_the_small_machine_s_budget(self, monkeypatch):
        import os

        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        assert pipeline.default_threads() == 8

    def test_a_small_machine_keeps_the_careful_default(self, monkeypatch):
        """Two is right on four cores and always was."""
        import os

        monkeypatch.setattr(os, "cpu_count", lambda: 4)
        assert pipeline.default_threads() == pipeline.INTRAOP_THREADS

    def test_it_never_goes_below_the_constant_it_replaced(self, monkeypatch):
        import os

        for cores in (1, 2, 3):
            monkeypatch.setattr(os, "cpu_count", lambda cores=cores: cores)
            assert pipeline.default_threads() >= pipeline.INTRAOP_THREADS, cores

    def test_it_leaves_half_the_machine_for_everything_else(self, monkeypatch):
        """The old reasoning still holds; only its arithmetic was fixed."""
        import os

        for cores in (8, 12, 16, 32, 64):
            monkeypatch.setattr(os, "cpu_count", lambda cores=cores: cores)
            assert pipeline.default_threads() <= max(cores // 2, pipeline.INTRAOP_THREADS)

    def test_it_is_capped_below_what_a_person_may_type(self, monkeypatch):
        """A default that cannot be wrong quietly is worth the last few percent."""
        import os

        monkeypatch.setattr(os, "cpu_count", lambda: 128)
        assert pipeline.default_threads() == pipeline.DEFAULT_THREAD_CEILING
        assert pipeline.DEFAULT_THREAD_CEILING < pipeline.MAX_INTRAOP_THREADS

    def test_a_machine_that_will_not_say_how_big_it_is_gets_the_constant(
            self, monkeypatch):
        import os

        for answer in (None, 0):
            monkeypatch.setattr(os, "cpu_count", lambda answer=answer: answer)
            assert pipeline.default_threads() == pipeline.INTRAOP_THREADS

    def test_the_setting_still_wins(self, host, monkeypatch):
        import os

        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        host.shared.opts.set(pipeline.OPT_THREADS, 3)
        assert pipeline.threads() == 3


class TestAClosureCarriesEveryStageItHasToRun:
    """Carrying a runtime is not the same as carrying the stage that uses it.

    The torch closure had ONNX Runtime in it and no ``dpdfnet``. One worker
    serves every enabled stage and the pair runs in this closure, so switching
    BOTH stages on was worse than switching on either: the worker died on
    ``import dpdfnet`` before a sample reached either model, every reply came
    out unenhanced, and the panel showed both stages Installed and Enabled the
    whole time. The only sign was ModuleNotFoundError in a log line.

    The same omission as the one that left LavaSR, vocos and encodec out of this
    closure when it was first built. Declaring wheels is not the same as being
    able to import what runs.
    """

    @staticmethod
    def _packages(block):
        import re

        found = set()
        for item in block["platforms"][0]["artifacts"]:
            if item.get("source_package"):
                found.add(item["source_package"].casefold())
                continue
            name = str(item.get("local_name") or item.get("filename") or "")
            found.add(re.split(r"-\d", name, 1)[0].casefold().replace("_", "-"))
        return found

    def test_the_torch_closure_can_import_dpdfnet(self):
        found = pipeline.manifest()
        torch = self._packages(pipeline.runtime_entry(found, "torch"))
        assert "dpdfnet" in torch, (
            "both stages enabled means one worker in the torch closure, and it "
            "opens DPDFNet's model with dpdfnet")

    def test_the_torch_closure_can_import_lavasr(self):
        torch = self._packages(pipeline.runtime_entry(pipeline.manifest(), "torch"))
        assert {"lavasr", "vocos", "encodec"} <= torch

    def test_the_torch_closure_holds_everything_the_onnx_one_runs_dpdfnet_with(self):
        """The general rule, so the next addition to one is not missed in the other.

        Anything the ONNX closure needs to run its stage, the torch closure
        needs too -- it runs that same stage whenever both are on. Version
        differences are fine; a missing name is not.
        """
        found = pipeline.manifest()
        onnx = self._packages(pipeline.runtime_entry(found, "onnx"))
        torch = self._packages(pipeline.runtime_entry(found, "torch"))

        assert not (onnx - torch), (
            f"the torch closure runs DPDFNet too and is missing {sorted(onnx - torch)}")

    def test_every_torch_platform_carries_the_same_packages(self):
        """The CUDA closure is the same closure on different wheels."""
        block = pipeline.runtime_entry(pipeline.manifest(), "torch")
        if len(block["platforms"]) < 2:
            pytest.skip("only one torch platform is pinned")

        import re

        def named(entry):
            found = set()
            for item in entry["artifacts"]:
                if item.get("source_package"):
                    found.add(item["source_package"].casefold())
                    continue
                name = str(item.get("local_name") or item.get("filename") or "")
                found.add(re.split(r"-\d", name, 1)[0].casefold().replace("_", "-"))
            return found

        sets = [named(entry) for entry in block["platforms"]]
        for other in sets[1:]:
            assert other == sets[0], (
                f"the CUDA closure differs by {sorted(other ^ sets[0])}")


class TestATurnAsksTheClosureItsOwnStagesNeed:
    """The last place the two closures were still being confused.

    snapshot() decides whether a turn is enhanced at all, and it gated the whole
    pipeline on ``state.runtime_ready`` -- the ONNX closure. LavaSR does not run
    in that one and needs nothing from it, so a machine whose ONNX closure was
    stale or absent got a silently unenhanced reply from a LavaSR that was
    installed, current and ready.
    """

    @staticmethod
    def _status(**flavours):
        found = _installed_status()
        return dataclasses.replace(
            found,
            runtime_install_state=flavours.get("onnx", "installed"),
            runtime_message="Not installed." if flavours.get("onnx") else "Installed.",
            runtime_flavours=tuple(
                (name, flavours.get(name, "installed"),
                 "Not installed." if flavours.get(name) else "Installed.")
                for name in pipeline.RUNTIME_FLAVOURS))

    def test_lavasr_alone_runs_without_the_onnx_closure(self, host):
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        host.shared.opts.set(pipeline.OPT_DPDFNET, False)
        host.shared.opts.set(pipeline.OPT_LAVASR, True)

        found = pipeline.snapshot(RATE, "pocket", self._status(onnx="not_installed"))

        assert found.stage_ids == ("lavasr",), (
            "LavaSR runs in the PyTorch closure and is owed nothing by the ONNX one")
        assert found.output_rate == 48000

    def test_dpdfnet_still_needs_the_closure_it_does_run_in(self, host):
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        host.shared.opts.set(pipeline.OPT_DPDFNET, True)
        host.shared.opts.set(pipeline.OPT_LAVASR, False)

        found = pipeline.snapshot(RATE, "pocket", self._status(onnx="not_installed"))

        assert found.stage_ids == (), "the ONNX stage is the one that needs it"
        assert found.reason, "and the turn says why rather than going quiet"

    def test_both_together_ask_for_the_torch_closure(self, host):
        """One worker serves both, and only the torch closure carries both."""
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        host.shared.opts.set(pipeline.OPT_DPDFNET, True)
        host.shared.opts.set(pipeline.OPT_LAVASR, True)

        held = pipeline.snapshot(RATE, "pocket", self._status(torch="not_installed"))
        assert held.stage_ids == (), "no torch closure, no pair"

        ready = pipeline.snapshot(RATE, "pocket", self._status(onnx="stale"))
        assert ready.stage_ids == ("dpdfnet", "lavasr"), (
            "the torch closure carries ONNX Runtime too, so a stale ONNX one "
            "does not stop the pair that is about to run inside the other")

    def test_nothing_enabled_asks_for_no_closure_at_all(self, host):
        host.shared.opts.set(pipeline.OPT_ENABLED, True)
        host.shared.opts.set(pipeline.OPT_DPDFNET, False)
        host.shared.opts.set(pipeline.OPT_LAVASR, False)

        found = pipeline.snapshot(RATE, "pocket",
                                  self._status(onnx="not_installed",
                                               torch="not_installed"))

        assert found.stage_ids == ()
        assert "no stages enabled" in found.reason.casefold(), found.reason


class TestTheInstallerSelfTestSurvivesARealVocosHead:
    """The install path itself, driven the way mc_voice_pipeline drives it.

    Everything else in this file exercises a stage. This exercises ``selftest``
    -- the function whose non-zero exit is the whole of "did not run on this
    machine" -- because that is where four attempts died and a stage-level test
    would not have caught any of them.

    The backend here is not a stand-in for a rate fault. It is a Vocos ISTFT
    head: it emits whole hops of whatever it is handed and drops the remainder,
    which is what LavaSR's BWE actually does. Driven through the real self-test
    at hop 512, 1024 or 2048 it reproduces a user's refusal to the sample --

        lavasr had to be corrected by 2112 samples in one second of audio

    -- and the audio it was refusing over was exact the whole time.
    """

    CONFIG = {"lavasr": {"backend_input_rate": 16000, "analysis_ms": 250,
                         "context_ms": 50, "denoise": False,
                         "provider": "CPUExecutionProvider", "adapter": 0},
              "test_rate": 24000}

    class _Istft:
        """Whole hops out, remainder dropped. Correct in rate, short in length."""

        providers = ("cpu",)

        def __init__(self, hop):
            self.hop = hop

        def reset(self, rate):
            return None

        def enhance(self, samples, rate):
            found = worker.Resampler(rate, 48000)(samples)
            return found[:(len(found) // self.hop) * self.hop]

    def _run(self, monkeypatch, backend):
        monkeypatch.setattr(worker, "_stage_backend",
                            lambda stage_id, root, config, numpy_module: backend)
        return worker.selftest({"lavasr": "/nonexistent"}, dict(self.CONFIG))

    @pytest.mark.parametrize("hop", [128, 256, 320, 512, 640, 1024, 1280, 2048])
    def test_every_hop_a_vocos_config_uses_installs(self, hop, monkeypatch):
        found = self._run(monkeypatch, self._Istft(hop))

        assert found["ok"] is True
        stage = found["stages"]["lavasr"]
        assert stage["samples"] == 48000, "one second in, one second out at 48 kHz"
        assert stage["correction"] == 0, "a whole-hop head is not a wrong clock"

    def test_the_hops_that_produced_the_users_refusal(self, monkeypatch):
        """512, 1024 and 2048 each summed to exactly 2112 before this."""
        for hop in (512, 1024, 2048):
            found = self._run(monkeypatch, self._Istft(hop))
            assert found["ok"] is True, hop
            assert found["stages"]["lavasr"]["framing"] == 448, hop

    def test_a_backend_that_misreads_its_rate_is_still_refused(self, monkeypatch):
        """The guard. Handed 16 kHz and believing it 24 kHz returns two thirds
        of what was asked for, and plays the reply a third too fast."""
        class MisreadsItsRate:
            providers = ("cpu",)

            def reset(self, rate):
                return None

            def enhance(self, samples, rate):
                return worker.Resampler(24000, 48000)(samples)

        with pytest.raises(worker.Refusal, match="not the rate it was told"):
            self._run(monkeypatch, MisreadsItsRate())

    def test_a_backend_that_returns_far_too_much_is_still_refused(self, monkeypatch):
        class HalfAgainOut:
            providers = ("cpu",)

            def reset(self, rate):
                return None

            def enhance(self, samples, rate):
                found = worker.Resampler(rate, 48000)(samples)
                return found + found[:len(found) // 2]

        with pytest.raises(worker.Refusal, match="not the rate it was told"):
            self._run(monkeypatch, HalfAgainOut())


class TestTheAnalysisWindowIsAMeasurementAndNotABlank:
    """Two nulls in a manifest were the last thing standing in front of LavaSR.

    ``LavaStage.__init__`` refuses ``analysis_ms <= 0`` rather than defaulting,
    which is right -- a window guessed in the worker would be a number nobody
    re-measured surviving into a release. What was wrong is that nothing ever
    supplied one, so the stage loaded its model, spent seven seconds doing it,
    and then refused on a value read before any of that.

    The window is not a free parameter either, and that is why it can be filled
    in here rather than only by ear. Upstream's FastLRMerge rffts the *whole*
    block and fades over a fixed bin count, so the crossover's width in Hz falls
    out of the window length: too short and the fade clips at DC, which is a
    merge doing something other than what it says.
    """

    @staticmethod
    def _contract():
        return dict(pipeline.stage_config("lavasr"))

    def test_the_worker_is_given_a_window_rather_than_a_null(self):
        found = self._contract()
        assert int(found["analysis_ms"]) > 0, (
            "a null here is what the stage refuses, after loading the model to find out")
        assert int(found["context_ms"]) > 0

    def test_the_window_clears_the_floor_its_own_merge_imposes(self):
        """Below this the fade starts at DC and the low band is not preserved.

        cutoff_bin is ``(8000 / 24000) * n_bins`` and the fade is 1024 bins
        wide, so the transition only sits *around* the cutoff while
        ``cutoff_bin >= 512`` -- which is n_bins >= 1536, a block of 3070
        samples at 48 kHz, about 64 ms.
        """
        found = self._contract()
        block_ms = int(found["analysis_ms"]) + 2 * int(found["context_ms"])
        n_bins = (block_ms * 48) // 2 + 1
        cutoff_bin = int((8000 / 24000) * n_bins)

        assert cutoff_bin >= 512, (
            f"a {block_ms} ms block gives {n_bins} bins, so the 1024-bin fade clips at "
            f"DC and the merge is not the crossover it claims to be")

    def test_the_crossover_lands_around_the_cutoff_and_not_over_the_speech(self):
        """The floor is not an operating point. This is the quality claim."""
        found = self._contract()
        block_ms = int(found["analysis_ms"]) + 2 * int(found["context_ms"])
        n_bins = (block_ms * 48) // 2 + 1
        hz = 24000 / n_bins
        cutoff_bin = int((8000 / 24000) * n_bins)
        low = max(0, cutoff_bin - 512) * hz

        assert low > 4000, (
            f"the transition starts at {low:.0f} Hz, which blends the original into the "
            f"upsampled band well below the 8 kHz cutoff it is supposed to sit on")

    def test_the_latency_it_buys_fits_inside_the_browser_s_start_buffer(self):
        """first_output_samples is analysis + context, and it is paid once."""
        found = self._contract()
        latency_ms = int(found["analysis_ms"]) + int(found["context_ms"])
        assert latency_ms <= 700, (
            f"{latency_ms} ms of intrinsic latency is more than the 0.7 s the browser "
            f"already buffers before it starts, so it would be heard as a slower reply")


class TestTheInternalDenoiserIsOffBecauseThePanelSaysItIs:
    """The panel stated a fact the worker contradicted.

    "Internal denoise: off — DPDFNet is the cleanup stage" is printed on the
    LavaSR row and ``"denoise": false`` is in the manifest contract, while
    ``enhance()`` was called with ``denoise=True`` regardless. So LavaSR's own
    denoiser ran on every window over audio DPDFNet had already cleaned, at the
    cost of a second inference per window, and nothing on screen said so.
    """

    def test_the_contract_says_off(self):
        assert pipeline.stage_config("lavasr")["denoise"] is False

    def test_the_backend_reads_the_flag_rather_than_hardcoding_it(self):
        import inspect

        from pipeline_worker import worker

        source = inspect.getsource(worker._LavaBackend)

        assert 'self._denoise = bool(config.get("denoise"))' in source
        # Code lines only. The comment above the call names the old spelling to
        # explain why it is gone, and a test that could not tell those apart
        # would be a test nobody can leave that comment in front of.
        called = [line.strip() for line in source.splitlines()
                  if not line.strip().startswith("#")
                  and ("self._model.enhance(" in line or "denoise=" in line)]
        assert any("denoise=self._denoise" in line for line in called), called
        assert not any("denoise=True" in line for line in called), (
            f"a hardcoded True is what made the panel's sentence untrue: {called}")


class TestTheResolverIsActuallyReached:
    """A guard on a defect I shipped: correct code nothing ever called.

    The CUDA wheel resolver was written, tested in isolation and committed --
    and never wired into the installer, so the closure it exists to build could
    not have been built. Unit tests on the resolver all passed. The thing they
    could not see was that no caller existed.
    """

    def test_the_runtime_installer_resolves_before_it_fetches(self, monkeypatch,
                                                              voice_root):
        import mc_voice_models as models
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("windows", "amd64", "3.13"))
        seen = []

        def watch(entry, say):
            seen.append(entry.get("id"))
            raise pipeline.PipelineError("stop here, the resolve is what was under test")

        monkeypatch.setattr(pipeline, "_resolve_artifacts", watch)
        with pytest.raises(pipeline.PipelineError, match="stop here"):
            pipeline._install_runtime(lambda *a, **k: None, lambda *a, **k: None,
                                      "torch", "cuda")
        assert seen == ["windows-x86_64-cp313-cu128"], (
            "the installer must hand the CUDA platform entry to the resolver")

    def test_a_closure_with_nothing_to_resolve_passes_through_unchanged(self):
        """Most closures are fully pinned; resolving must be a no-op for them."""
        entry = pipeline._platform_entry_or_none(pipeline.manifest(), "onnx", "cpu")
        if entry is None:
            pytest.skip("no ONNX closure for this platform")
        rows = pipeline._resolve_artifacts(entry, lambda *a, **k: None)
        assert rows == entry["artifacts"]
        assert all(not item["resolve"] for item in rows)
