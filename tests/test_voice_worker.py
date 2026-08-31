"""The worker's engine adapter, against the runtime it actually ships into.

The isolated Voice runtime is two unpacked wheels -- ``sherpa_onnx`` and its
core -- and nothing else. In particular it has no NumPy, and that turns out to
decide which of sherpa's two synthesis paths this feature can use:

    generate(text, sid, speed)             samples come back as a plain list
    generate(..., callback=fn)             each batch is delivered to Python as
                                           ``py::array_t<float>``, which pybind11
                                           builds with NumPy

So the completed-audio path works and the streaming callback raises. The first
version of protocol 2 decided between them by looking for the word "callback"
in ``generate``'s docstring -- a question about documentation rather than about
what happens when you call it -- reported "callback streaming" in its handshake,
and then failed every single spoken reply while auditions kept working
perfectly. These tests are that bug, written down.
"""

from __future__ import annotations

import pathlib

import pytest

from voice_worker import worker


BATCH = 480
"""How many samples a fake batch carries -- twenty milliseconds at 24 kHz.

Longer than :data:`voice_worker.worker.DECLICK_MS` on purpose. A segment's last
few milliseconds are withheld by ``Seam`` until the segment ends, so a fixture
whose whole synthesis was shorter than that ramp would be a fixture in which no
audio ever reaches the parent -- and tests about what happens *after* audio has
been sent would then be testing the opposite case.
"""


class Audio:
    def __init__(self, samples, rate=24000):
        self.samples = samples
        self.sample_rate = rate


class Tts:
    """A sherpa OfflineTts whose callback path fails the way the real one does.

    ``callback_error`` stands in for pybind11 being unable to build the NumPy
    array it needs to deliver a batch. ``callback_works`` is the same class on a
    runtime that has NumPy.
    """

    num_speakers = 53
    sample_rate = 24000

    def __init__(self, callback_error=None, batches=2, level=0.5):
        self.callback_error = callback_error
        self.batches = batches
        self.level = float(level)
        self.calls = []

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls.append({"text": text, "sid": sid, "speed": speed,
                           "callback": callback is not None})
        if callback is not None:
            if self.callback_error is not None:
                raise self.callback_error
            for _batch in range(self.batches):
                if callback([self.level] * BATCH, 0.5) == 0:
                    break
            return Audio([])
        return Audio([0.25] * (2 * BATCH))


def engines(tts):
    found = worker.Engines({})
    found.tts = tts
    found.sample_rate = tts.sample_rate
    found.num_speakers = tts.num_speakers
    return found


class TestTheStreamingCapability:
    def test_a_runtime_without_numpy_reports_segment_streaming(self):
        """The real failure: pybind11 cannot build ``py::array_t<float>`` with
        no NumPy installed, so the callback raises before the first batch."""
        found = engines(Tts(callback_error=ImportError("No module named 'numpy'")))
        assert found._callback_supported() is False

    def test_a_runtime_with_numpy_reports_callback_streaming(self):
        found = engines(Tts())
        assert found._callback_supported() is True

    def test_a_build_with_no_callback_parameter_reports_segment_streaming(self):
        class Old:
            num_speakers = 53
            sample_rate = 24000

            def generate(self, text, sid=0, speed=1.0):
                return Audio([0.25] * (2 * BATCH))

        assert engines(Old())._callback_supported() is False

    def test_a_callback_that_is_accepted_and_never_called_is_not_streaming(self):
        """Believed only when it actually delivers a batch: a build that takes
        the argument and ignores it would otherwise be reported as streaming and
        then produce audio only at the very end."""
        found = engines(Tts(batches=0))
        assert found._callback_supported() is False

    def test_the_probe_stops_at_the_first_batch(self):
        tts = Tts(batches=50)
        engines(tts)._callback_supported()
        assert len(tts.calls) == 1


class TestSpeakingEitherWay:
    def collect(self, found, text="Hello there."):
        blocks = []
        metrics = found.stream(text, 3, worker.NEUTRAL,
                               lambda block, rate: blocks.append((block, rate)) or True)
        return blocks, metrics

    def test_a_callback_runtime_streams_each_batch(self):
        """Two batches handed back, and both are passed on as they arrive.

        Three blocks leave rather than two, and the third is ``Seam``'s: a
        segment's last few milliseconds are withheld until the segment ends so
        that it can be ramped down into whatever follows it. What must not
        change is the audio -- every sample sherpa produced still goes out.
        """
        found = engines(Tts())
        found.streaming = "callback"
        blocks, metrics = self.collect(found)
        assert metrics["blocks"] == 2
        assert len(blocks) == 3
        assert metrics["samples"] == 2 * BATCH

    def test_a_runtime_without_numpy_still_speaks_the_segment(self):
        """The behaviour that matters. Segment-at-a-time is coarser than
        batch-at-a-time and it is still streaming: the parent commits one
        sentence at a time, so speech still starts before the reply is over."""
        found = engines(Tts(callback_error=ImportError("No module named 'numpy'")))
        found.streaming = "callback"
        blocks, metrics = self.collect(found)
        assert blocks, "a reply was lost instead of being spoken the coarser way"
        assert metrics["samples"] == 2 * BATCH
        assert found.streaming == "segment", "the fallback was not remembered"

    def test_the_fallback_is_remembered_rather_than_retried_every_segment(self):
        tts = Tts(callback_error=ImportError("numpy"))
        found = engines(tts)
        found.streaming = "callback"
        self.collect(found, "First sentence.")
        self.collect(found, "Second sentence.")
        assert [call["callback"] for call in tts.calls] == [True, False, False]

    def test_a_failure_after_audio_has_been_sent_is_not_retried(self):
        """Re-running a segment that has already been half spoken would say
        those words twice."""
        class Broken(Tts):
            def generate(self, text, sid=0, speed=1.0, callback=None):
                if callback is not None:
                    callback([0.5] * BATCH, 0.5)
                    raise RuntimeError("the pipe went away")
                return Audio([0.25] * (2 * BATCH))

        found = engines(Broken())
        found.streaming = "callback"
        with pytest.raises(RuntimeError):
            self.collect(found)

    def test_the_speaker_the_parent_resolved_is_the_one_used(self):
        tts = Tts()
        found = engines(tts)
        found.streaming = "callback"
        self.collect(found)
        assert tts.calls[0]["sid"] == 3

    def test_a_speaker_the_bank_does_not_have_is_refused(self):
        found = engines(Tts())
        with pytest.raises(ValueError, match="voice bank"):
            found.stream("Hello.", 999, worker.NEUTRAL, lambda block, rate: True)


class TestReadingTheProcessPriority:
    """Observation only, and a value that is not a lie.

    The first shipped version reported ``priority=class_0x0`` in real logs.
    Zero is not a Windows priority class: it is ``GetPriorityClass`` returning
    failure, most likely because an undeclared pointer-sized handle reached a
    64-bit API as an int. A diagnostic that reports a failed read as a value is
    worse than no diagnostic at all, because it is the thing somebody will
    reason from.
    """

    def test_it_reports_this_process_nice_value_on_linux(self, monkeypatch):
        monkeypatch.setattr(worker.os, "name", "posix")
        found = worker._priority()
        assert found.startswith("nice"), found
        assert int(found[4:]) == worker.os.getpriority(worker.os.PRIO_PROCESS, 0)

    def test_reading_it_does_not_change_it(self, monkeypatch):
        monkeypatch.setattr(worker.os, "name", "posix")
        before = worker.os.getpriority(worker.os.PRIO_PROCESS, 0)
        worker._priority()
        worker._priority()
        assert worker.os.getpriority(worker.os.PRIO_PROCESS, 0) == before

    def windows(self, monkeypatch, answer):
        """The Windows branch, with a kernel32 that answers ``answer``."""
        import types

        calls = []

        class Function:
            def __init__(self, name, value):
                self.name = name
                self.value = value
                self.restype = None
                self.argtypes = None

            def __call__(self, *args):
                calls.append((self.name, self.restype, self.argtypes))
                return self.value

        class Kernel32:
            GetCurrentProcess = Function("GetCurrentProcess", -1)
            GetPriorityClass = Function("GetPriorityClass", answer)

        fake = types.ModuleType("ctypes")
        fake.windll = types.SimpleNamespace(kernel32=Kernel32())
        fake.c_void_p = "c_void_p"
        fake.c_uint = "c_uint"
        monkeypatch.setitem(__import__("sys").modules, "ctypes", fake)
        monkeypatch.setattr(worker.os, "name", "nt")
        return worker._priority(), calls

    def test_a_windows_class_is_named(self, monkeypatch):
        found, _calls = self.windows(monkeypatch, 0x00004000)
        assert found == "below_normal"

    def test_a_zero_is_reported_as_a_failed_read_and_not_as_a_class(self, monkeypatch):
        """The exact bug the logs showed."""
        found, _calls = self.windows(monkeypatch, 0)
        assert found == "unreadable"
        assert "class_0x0" not in found

    def test_the_handle_is_declared_pointer_sized(self, monkeypatch):
        """Which is the reason zero was being returned in the first place."""
        _found, calls = self.windows(monkeypatch, 0x00000020)
        declared = {name: (restype, argtypes) for name, restype, argtypes in calls}
        assert declared["GetCurrentProcess"][0] == "c_void_p"
        assert declared["GetPriorityClass"][1] == ["c_void_p"]

    def test_an_unknown_class_still_says_what_it_saw(self, monkeypatch):
        found, _calls = self.windows(monkeypatch, 0x1234)
        assert found == "class_0x1234"

    def test_lowering_the_priority_is_left_exactly_as_it_was(self):
        """Declaring types there would not be a diagnostic change -- it would
        make the call start working, and a worker that began yielding CPU for
        the first time is a change to how fast speech is synthesised. That is a
        decision to take from the corrected reading, not a side effect of
        tidying a neighbouring function."""
        source = pathlib.Path(worker.__file__).read_text(encoding="utf-8")
        body = source.split("def _lower_priority()")[1].split("\ndef ")[0]
        assert "SetPriorityClass" in body
        assert "argtypes" not in body
        assert "restype" not in body


class TestCallbackGranularity:
    """What "callback streaming" is actually worth, measured rather than assumed.

    sherpa calls the generation callback once every ``max_num_sentences``
    sentences, and this worker configures that to one. So callback mode overlaps
    production *across sentences* and does nothing at all for a segment that
    contains a single sentence -- and a handshake that says "callback" is not
    evidence to the contrary. The worker therefore reports how many batches it
    received and how soon the first arrived, so the difference is visible in a
    log instead of being argued about.
    """

    class Sentences:
        """A synthesiser that batches by sentence, the way the real one does."""

        num_speakers = 53
        sample_rate = 24000

        def generate(self, text, sid=0, speed=1.0, callback=None):
            batches = max(1, text.count(".") + text.count("?") + text.count("!"))
            if callback is None:
                return Audio([0.25] * (BATCH * batches))
            for _batch in range(batches):
                if callback([0.5] * BATCH, 0.5) == 0:
                    return Audio([])
            return Audio([])

    def blocks_for(self, text):
        """How many times sherpa handed audio back for ``text``.

        The count comes from ``stream`` rather than from the number of blocks
        that left it, because those are no longer the same number: ``Seam``
        withholds a segment's last few milliseconds and releases them in a block
        of its own. Sherpa's hand-backs are what this whole class is about.
        """
        found = engines(self.Sentences())
        found.streaming = "callback"
        seen = []
        metrics = found.stream(text, 3, worker.NEUTRAL,
                               lambda block, rate: seen.append(block) or True)
        assert seen, "nothing was spoken at all"
        return metrics["blocks"]

    def test_one_sentence_arrives_in_one_batch(self):
        """The measurement that stops callback mode being oversold: for a
        one-sentence segment the first batch *is* the whole synthesis."""
        assert self.blocks_for("Yes, that is possible.") == 1

    def test_a_multi_sentence_segment_arrives_in_several(self):
        assert self.blocks_for("One. Two. Three. Four.") == 4

    def test_the_probe_records_what_it_cost(self):
        found = worker.Engines({})
        found.tts = Tts()
        found.sample_rate = 24000
        found.num_speakers = 53
        started = worker.time.monotonic()
        found.streaming = "callback" if found._callback_supported() else "segment"
        found.callback_probe_ms = int((worker.time.monotonic() - started) * 1000)
        assert found.streaming == "callback"
        assert found.callback_probe_ms >= 0


class TestWhatTheParentIsToldAboutASegment:
    """``tts_segment_done`` carries counts and milliseconds, and nothing else."""

    def speak_one(self, tts, text="One. Two. Three.", pause_ms=0):
        found = engines(tts)
        found.streaming = "callback"
        machine = worker.Worker(stdout=None)
        machine.engines = found
        sent = []
        machine.send = lambda header, payload=b"", audio=False: sent.append(header)
        turn = worker.Turn("T1", 3, worker.Delivery(pause_ms=pause_ms))
        turn.segments.put(text)
        turn.finish()
        machine.speak(turn)
        return sent

    def test_the_block_count_is_the_sentence_batch_count(self):
        sent = self.speak_one(TestCallbackGranularity.Sentences())
        done = [frame for frame in sent if frame["op"] == "tts_segment_done"]
        assert len(done) == 1
        assert done[0]["blocks"] == 3
        assert done[0]["streaming"] == "callback"
        assert done[0]["first_block_ms"] >= 0

    def test_a_configured_pause_is_not_counted_as_this_segments_audio(self):
        """A pause the user asked for is prosody. Counting the silence as the
        segment's first block would make an intentional gap read as a fast
        synthesis, which is the opposite of the truth."""
        first = self.speak_one(TestCallbackGranularity.Sentences(), pause_ms=0)
        second = self.speak_one(TestCallbackGranularity.Sentences(), pause_ms=250)
        done = [frame for frame in second if frame["op"] == "tts_segment_done"]
        assert done[0]["blocks"] == [f for f in first
                                     if f["op"] == "tts_segment_done"][0]["blocks"]

    def test_nothing_about_the_text_is_reported(self):
        sent = self.speak_one(TestCallbackGranularity.Sentences(),
                              text="The launch code is four zero four.")
        assert "four zero four" not in repr(sent)


class TestPcm16:
    def test_it_converts_a_plain_list_without_numpy(self, monkeypatch):
        """The list is what ``GeneratedAudio.samples`` is on this runtime, and
        the path that handles it must not need the package that is missing."""
        monkeypatch.setattr(worker, "_NUMPY", None)
        assert worker.pcm16([0.0, 1.0, -1.0]).hex() == "0000ff7f0180"

    def test_it_converts_a_numpy_array_the_same_way(self):
        numpy = pytest.importorskip("numpy")
        worker._NUMPY = "unasked"
        assert worker.pcm16(numpy.array([0.0, 1.0, -1.0], dtype="float32")).hex() == \
            "0000ff7f0180"

    def test_it_does_not_modify_what_it_was_given(self):
        numpy = pytest.importorskip("numpy")
        worker._NUMPY = "unasked"
        samples = numpy.array([2.0, -2.0], dtype="float32")
        worker.pcm16(samples)
        assert list(samples) == [2.0, -2.0], "the caller's samples were clipped in place"

    def test_it_asks_for_numpy_once_rather_than_per_block(self, monkeypatch):
        """On this runtime the answer is "no", and asking per audio block turned
        that into an ImportError raised and swallowed thousands of times a
        minute while somebody was being spoken to."""
        asked = []
        real = __import__

        def counting(name, *args, **kwargs):
            if name == "numpy":
                asked.append(name)
                raise ImportError("No module named 'numpy'")
            return real(name, *args, **kwargs)

        worker._NUMPY = "unasked"
        monkeypatch.setattr("builtins.__import__", counting)
        worker.pcm16([0.0])
        worker.pcm16([0.0])
        worker.pcm16([0.0])
        assert len(asked) == 1, asked
        assert worker._NUMPY is None, "the answer was not remembered"

    def test_empty_input_is_empty_output(self):
        assert worker.pcm16([]) == b""
        assert worker.pcm16(None) == b""


class TestTheQuietAModelPutsRoundASegmentIsCutBack:
    """The gap between sentences, and what most of it actually was.

    A reply is a run of segments played back with nothing between them, so the
    silence a listener hears between two sentences is the silence *inside* the
    two segments: the model's own quiet before the first word and after the
    last. None of it was asked for.

    Cut back rather than cut out: sentences are separated by a pause in speech,
    and the delivery's own "Pause between sentences" adds to what is left.
    """

    class Padded:
        """A synthesiser whose one batch is quiet, speech, and quiet."""

        num_speakers = 53
        sample_rate = 24000

        def __init__(self, lead=0.5, tail=0.4, level=0.0):
            self.lead = lead
            self.tail = tail
            self.level = level

        def samples(self):
            rate = self.sample_rate
            return ([self.level] * int(self.lead * rate)
                    + [0.5] * rate
                    + [self.level] * int(self.tail * rate))

        def generate(self, text, sid=0, speed=1.0, callback=None):
            if callback is not None:
                callback(self.samples(), 0.5)
                return Audio([])
            return Audio(self.samples())

    def spoken(self, tts, delivery=None):
        import array
        import sys

        found = engines(tts)
        found.streaming = "callback"
        blocks = []
        metrics = found.stream("Hello there.", 3, delivery or worker.NEUTRAL,
                               lambda block, rate: blocks.append(block) or True)
        numbers = array.array("h")
        numbers.frombytes(b"".join(blocks))
        if sys.byteorder == "big":
            numbers.byteswap()
        return numbers, metrics

    def kept(self, tts):
        rate = tts.sample_rate
        return rate + int(rate * (worker.KEEP_LEAD_MS + worker.KEEP_TAIL_MS) / 1000)

    def test_the_padding_either_side_is_cut_back(self):
        tts = self.Padded()
        found, metrics = self.spoken(tts)
        assert abs(len(found) - self.kept(tts)) < tts.sample_rate * 0.03
        assert metrics["trimmed_ms"] == pytest.approx(
            900 - worker.KEEP_LEAD_MS - worker.KEEP_TAIL_MS, abs=40)

    def test_the_edges_are_still_ramped_to_silence(self):
        """Trim first, ramp second. The other way round cuts the ramp off and
        puts back the click it exists to remove."""
        found, _metrics = self.spoken(self.Padded())
        assert abs(found[0]) < 300
        assert abs(found[-1]) < 300

    def test_a_segment_with_no_padding_loses_nothing(self):
        tts = self.Padded(lead=0.0, tail=0.0)
        found, metrics = self.spoken(tts)
        assert metrics["trimmed_ms"] == 0
        assert abs(len(found) - tts.sample_rate) < tts.sample_rate * 0.02

    def test_the_level_follows_the_segment_rather_than_being_fixed(self):
        """A voice recorded with room tone still has its padding recognised."""
        tts = self.Padded(lead=0.3, level=0.005)
        assert 0.005 * 32767 > worker.QUIET_FLOOR, "the fixture is not a real test"
        _found, metrics = self.spoken(tts)
        assert metrics["trimmed_ms"] == pytest.approx(
            700 - worker.KEEP_LEAD_MS - worker.KEEP_TAIL_MS, abs=40)
        assert metrics["floor_db"] == pytest.approx(-46, abs=2)


class TestASegmentEndsAtSilenceRatherThanWhereverItWas:
    """The pop between sentences, and the thing that removes it.

    A segment is one sentence's forward pass. Where the waveform happens to be
    when the sentence runs out is where the samples stop, and the next sentence
    is a fresh pass that starts wherever it starts. This feature plays segments
    back sample-exact and one after another, with no gap between them unless
    somebody asked for one -- so the join between two of them is a step, and a
    step in a waveform is a click at the end of a sentence.

    The batches below are DC on purpose. It is the worst case and it is what a
    residual offset actually looks like; nothing but a ramp removes it.
    """

    def spoken(self, tts, delivery=None):
        """One segment's PCM as signed samples, decoded without NumPy.

        Without NumPy because this runtime has none -- see the module docstring
        -- and a test that reached for it here would be testing a Python this
        worker never runs on.
        """
        import array
        import sys

        found = engines(tts)
        found.streaming = "callback"
        blocks = []
        found.stream("Hello there.", 3, delivery or worker.NEUTRAL,
                     lambda block, rate: blocks.append(block) or True)
        numbers = array.array("h")
        numbers.frombytes(b"".join(blocks))
        if sys.byteorder == "big":
            numbers.byteswap()
        return numbers

    def test_a_segment_starts_and_ends_at_silence(self):
        found = self.spoken(Tts())
        assert len(found) == 2 * BATCH, "audio was lost to the ramp"
        assert abs(found[0]) < 300, f"a segment opened at {found[0]}"
        assert abs(found[-1]) < 300, f"a segment closed at {found[-1]}"

    def test_the_step_a_join_would_have_carried_is_gone(self):
        """The measurement rather than the assertion: no sample-to-sample step.

        Without the ramp both edges sit at half full scale, so the join on
        either side of the segment is a step of that size. What is left is the
        ramp's own slope, which is two orders of magnitude smaller.
        """
        joined = list(self.spoken(Tts(level=0.5))) + list(self.spoken(Tts(level=-0.5)))
        steps = [abs(joined[index + 1] - joined[index]) for index in range(len(joined) - 1)]
        assert max(steps) < 32767 * 0.01

    def test_the_middle_of_the_segment_is_untouched(self):
        """A ramp at the edges, and nothing anywhere else."""
        found = self.spoken(Tts())
        span = worker.Seam(24000).span
        middle = list(found)[span:-span]
        assert middle, "the fixture is shorter than two ramps"
        assert min(middle) > 32767 * 0.49

    def test_a_shaped_segment_is_ramped_as_well(self):
        """Pitch and volume change the samples; they do not change the edges."""
        found = self.spoken(Tts(), delivery=worker.Delivery(pitch=1.1, gain=2.0))
        assert len(found), "a shaped segment produced no audio"
        assert abs(found[0]) < 300
        assert abs(found[-1]) < 300


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


class TestWhatKokoroIsAskedFor:
    """One of the four delivery controls is the model's and three are this
    file's arithmetic on what it produced. The split is the point."""

    def test_a_neutral_delivery_asks_for_speed_one_and_shapes_nothing(self):
        assert worker.NEUTRAL.generation_speed == 1.0
        assert worker.NEUTRAL.shapes is False

    def test_speed_goes_straight_into_generate(self):
        tts = Tts()
        found = engines(tts)
        found.streaming = "segment"
        found.stream("Hello.", 3, worker.Delivery(speed=1.4), lambda block, rate: True)
        assert tts.calls[-1].get("speed") == pytest.approx(1.4)

    def test_pitch_divides_out_of_the_synthesis_speed(self):
        """The resampler reads the result back at ``pitch`` samples per output
        sample, which shortens it again -- so the model has to be asked for
        ``speed x pitch`` or the pitch would come with a duration change nobody
        asked for."""
        tts = Tts()
        found = engines(tts)
        found.streaming = "segment"
        found.stream("Hello.", 3, worker.Delivery(speed=1.0, pitch=2.0),
                     lambda block, rate: True)
        assert tts.calls[-1].get("speed") == pytest.approx(2.0)

    def test_a_header_that_is_nonsense_is_kokoros_own_delivery(self):
        """The boundary between a JSON header and a number that multiplies a
        synthesis rate. Every way it can be wrong means the same thing."""
        found = worker.Delivery.from_header(
            {"speed": "fast", "pitch": None, "gain": float("nan"), "pause_ms": "soon"})
        assert (found.speed, found.pitch, found.gain, found.pause_ms) == (1.0, 1.0, 1.0, 0)

    def test_a_runaway_multiplier_is_bounded_here_as_well_as_in_the_parent(self):
        found = worker.Delivery.from_header({"speed": 500, "pitch": 500, "gain": 500,
                                             "pause_ms": 500000})
        assert found.speed == worker.Delivery.SPEED[1]
        assert found.pitch == worker.Delivery.PITCH[1]
        assert found.gain == worker.Delivery.GAIN[1]
        assert found.pause_ms == worker.Delivery.PAUSE[1]


class TestTheResampler:
    def test_a_ratio_of_one_returns_what_it_was_given(self):
        found = worker.Resampler(1.0)
        assert found.feed([0.0, 0.25, 0.5, 0.75]) == pytest.approx([0.0, 0.25, 0.5, 0.75])

    def test_a_higher_ratio_shortens_the_audio(self):
        """Which is what raises the pitch: the same waveform read faster."""
        found = worker.Resampler(2.0)
        out = found.feed([float(i) for i in range(100)])
        assert 45 <= len(out) <= 51, len(out)

    def test_a_lower_ratio_lengthens_it(self):
        found = worker.Resampler(0.5)
        out = found.feed([float(i) for i in range(100)])
        assert 190 <= len(out) <= 201, len(out)

    def test_it_reads_across_a_block_boundary_rather_than_restarting(self):
        """An output sample routinely needs two input samples that arrived in
        different callback batches. Restarting at each block is a click at every
        boundary; carrying the read position is not."""
        whole = worker.Resampler(1.5)
        streamed = worker.Resampler(1.5)
        source = [float(i) for i in range(60)]
        one = whole.feed(list(source))
        two = streamed.feed(source[:17]) + streamed.feed(source[17:40]) \
            + streamed.feed(source[40:])
        assert one == pytest.approx(two)

    def test_it_does_not_hold_the_whole_reply_in_memory(self):
        """Everything before the next read position can never be read again, and
        a long reply that kept its first sentence would be a memory leak with a
        sample rate."""
        found = worker.Resampler(1.0)
        for _block in range(50):
            found.feed([0.1] * 1000)
        assert len(found._tail) < 100


class TestTheShaper:
    def test_a_neutral_delivery_is_exactly_pcm16(self):
        samples = [0.0, 0.5, -0.5, 1.0]
        assert worker.Shaper(worker.NEUTRAL).block(samples) == worker.pcm16(samples)

    def test_volume_scales_the_samples(self):
        found = worker.Shaper(worker.Delivery(gain=2.0)).block([0.25])
        assert found == worker.pcm16([0.5])

    def test_volume_cannot_distort(self):
        """The clamp that was already protecting against a model that overshot
        is what limits this: a loud setting stops getting louder rather than
        clipping into distortion."""
        found = worker.Shaper(worker.Delivery(gain=8.0)).block([0.9])
        assert found == worker.pcm16([1.0])

    def test_pitch_shortens_the_block(self):
        samples = [float(i) / 100 for i in range(100)]
        found = worker.Shaper(worker.Delivery(pitch=2.0)).block(samples)
        assert len(found) // 2 < len(samples)

    def test_the_completed_path_and_the_streamed_path_shape_alike(self):
        """An audition and a spoken reply have to come out the same, or the
        sliders are being adjusted against a sound they do not produce."""
        delivery = worker.Delivery(pitch=1.5, gain=1.5)
        samples = [0.4] * 64
        streamed = worker.Shaper(delivery).block(samples)
        completed = worker.pcm16(worker.Shaper(delivery).levelled(samples))
        assert streamed == completed


class TestPacing:
    def test_silence_is_the_length_it_was_asked_for(self):
        found = worker.silence(24000, 250)
        assert len(found) == 2 * 6000
        assert set(found) == {0}

    def test_no_pause_is_no_bytes(self):
        assert worker.silence(24000, 0) == b""
        assert worker.silence(0, 250) == b""
