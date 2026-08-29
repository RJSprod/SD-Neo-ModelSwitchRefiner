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

import pytest

from voice_worker import worker


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

    def __init__(self, callback_error=None, batches=2):
        self.callback_error = callback_error
        self.batches = batches
        self.calls = []

    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls.append({"text": text, "sid": sid, "callback": callback is not None})
        if callback is not None:
            if self.callback_error is not None:
                raise self.callback_error
            for _batch in range(self.batches):
                if callback([0.5] * 4, 0.5) == 0:
                    break
            return Audio([])
        return Audio([0.25] * 8)


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
                return Audio([0.25] * 8)

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
        produced = found.stream(text, 3, 1.0,
                                lambda block, rate: blocks.append((block, rate)) or True)
        return blocks, produced

    def test_a_callback_runtime_streams_each_batch(self):
        found = engines(Tts())
        found.streaming = "callback"
        blocks, produced = self.collect(found)
        assert len(blocks) == 2
        assert produced == 8

    def test_a_runtime_without_numpy_still_speaks_the_segment(self):
        """The behaviour that matters. Segment-at-a-time is coarser than
        batch-at-a-time and it is still streaming: the parent commits one
        sentence at a time, so speech still starts before the reply is over."""
        found = engines(Tts(callback_error=ImportError("No module named 'numpy'")))
        found.streaming = "callback"
        blocks, produced = self.collect(found)
        assert blocks, "a reply was lost instead of being spoken the coarser way"
        assert produced == 8
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
                    callback([0.5] * 4, 0.5)
                    raise RuntimeError("the pipe went away")
                return Audio([0.25] * 8)

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
            found.stream("Hello.", 999, 1.0, lambda block, rate: True)


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
