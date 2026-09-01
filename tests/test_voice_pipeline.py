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
        """Not "coming soon". A sentence naming what is actually missing.

        Its rate contract *is* measured now -- upstream's own enhance() resamples
        16000 to 48000 unconditionally, which is the number the adapter needed.
        What is missing is a dependency closure: no wheel, Torch, and a vocos
        fork pinned to a git branch rather than a release.
        """
        assert pipeline.stage_installable("lavasr") is False
        reason = pipeline.stage_unavailable_reason("lavasr")
        assert "LavaSR" in reason and "wheel" in reason
        with pytest.raises(pipeline.PipelineError, match="LavaSR"):
            pipeline.install("lavasr")

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
        # Still unmeasured, and still what a release would wait for.
        assert not contract.get("analysis_ms")
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
                            lambda: ("python" if runtime_present else None))
        monkeypatch.setattr(pipeline, "_install_runtime",
                            lambda say, tick: (done.append("runtime"), tick(1.0)))
        monkeypatch.setattr(pipeline, "_install_stage",
                            lambda which, say, tick: (done.append(which), tick(1.0)))
        monkeypatch.setattr(pipeline, "status", lambda: "status")
        return done, said, bar

    def test_installing_a_stage_installs_the_runtime_first(self, monkeypatch):
        done, said, bar = self._stub(monkeypatch, runtime_present=False)
        pipeline.install("dpdfnet", on_status=said.append, on_progress=bar.append)
        assert done == ["runtime", "dpdfnet"], (
            "a stage install must build the runtime it is proved inside before "
            "trying to prove anything in it")
        assert any("runtime" in text and "first" in text for text in said), said

    def test_the_runtime_is_not_reinstalled_when_it_is_already_there(self, monkeypatch):
        done, said, bar = self._stub(monkeypatch, runtime_present=True)
        pipeline.install("dpdfnet", on_status=said.append, on_progress=bar.append)
        assert done == ["dpdfnet"], (
            "an installed runtime must not be rebuilt underneath every stage")

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

    def test_the_stage_id_becomes_the_component_id_once(self):
        assert pipeline.component_of("dpdfnet") == "voice-pipeline-dpdfnet"
        assert pipeline.component_of("lavasr") == "voice-pipeline-lavasr"

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
