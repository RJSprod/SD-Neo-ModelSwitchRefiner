"""The VibeVoice sidecar: its framing, its script, its upstream calls, its loop.

Four things are asserted here that nothing else can assert.

The first is that the worker speaks the same wire as the other sidecars -- by
agreement rather than by import, so the agreement is checked to byte equality
against the Pocket worker.

The second is that the script and the voice samples are handed to the
processor the way the processor actually reads them: one ``Speaker N:`` line
per entry, numbers dense from one, and the samples ordered by speaker
*number*, because the processor binds its i-th sample to the line numbered i.
A script that named Speakers 1 and 3 with samples in order of first appearance
would have given the third speaker no voice at all.

The third is the upstream binding itself: the exact arguments of the four
calls the design fixes, checked against stand-ins for Torch, the model and the
processor. A wrong keyword there is a ``TypeError`` inside somebody's first
forty-minute render.

The fourth is the command loop against a fake engine: the handshake fields,
one render at a time, a cancel that lands while a render is in flight, an
error that crosses as a class name only, and a shutdown that answers.

Since the Realtime model, quantisation, LoRAs and streamed audio, a fifth:
what a load *is* (its identity, and that anything else unloads first, but only
after every refusal it could earn), the exact arguments of the calls those
features add -- ``BitsAndBytesConfig``, ``PeftModel.from_pretrained``, the
0.5B's scheduler swap, its preset read under ``safe_globals`` and its
``generate`` -- and a streamed render whose ``generate`` really runs on
another thread while its frames are delivered. ``tools/smoke_vibevoice_worker.py``
drives the same code against real tiny models; these tests pin the calls.

Nothing here imports Torch, transformers, vibevoice, PEFT or bitsandbytes.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import queue
import re
import struct
import sys
import threading
import time
import types

import pytest

from pocket_worker import worker as pocket_worker
from vibevoice_worker import worker

numpy = pytest.importorskip("numpy", reason="the worker cuts its voice samples with NumPy")

RATE = 24000
UUID = "GPU-0123abcd-4567-89ef-0123-456789abcdef"

PROCESSOR_LINE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
"""The regex ``VibeVoiceProcessor._parse_script`` reads each line with, copied
so a text this worker builds is checked against the reader it is for."""


def pcm(seconds: float, value: float = 0.25) -> bytes:
    return (numpy.full(int(seconds * RATE), value, dtype="<f4")).tobytes()


# --------------------------------------------------------------------------- #
# The wire
# --------------------------------------------------------------------------- #


class TestTheWireIsTheOneTheOtherWorkersSpeak:
    def test_a_frame_written_by_one_is_read_by_the_other(self):
        header = {"op": "render", "id": "a1", "job": "j"}
        payload = b"\x01\x02\x03\x04"
        for writer, reader in ((worker, pocket_worker), (pocket_worker, worker)):
            buffer = io.BytesIO()
            writer.write_frame(buffer, header, payload)
            assert reader.read_frame(io.BytesIO(buffer.getvalue())) == (header, payload)

    def test_the_bytes_are_identical_and_not_merely_compatible(self):
        first, second = io.BytesIO(), io.BytesIO()
        worker.write_frame(first, {"op": "init", "id": "7"}, b"body")
        pocket_worker.write_frame(second, {"op": "init", "id": "7"}, b"body")
        assert first.getvalue() == second.getvalue()
        assert worker._LENGTH.format == ">I"

    def test_the_protocol_is_its_own(self):
        assert worker.PROTOCOL_VERSION == 1
        assert worker.WORKER_NAME == "vibevoice"
        assert worker.MARKER == "--model-chain-vibevoice-worker"
        assert worker.MARKER != pocket_worker.MARKER
        assert worker.SAMPLE_RATE == 24000

    def test_the_payload_ceiling_holds_a_forty_five_minute_render(self):
        assert worker.MAX_PAYLOAD == 256 << 20
        assert worker.MAX_HEADER == 1 << 20
        assert worker.MAX_PAYLOAD > 45 * 60 * RATE * 2

    def test_an_oversized_header_is_refused_rather_than_allocated(self):
        raw = worker._LENGTH.pack(worker.MAX_HEADER + 1)
        with pytest.raises(worker.Refusal):
            worker.read_frame(io.BytesIO(raw))

    def test_an_oversized_payload_is_refused_rather_than_allocated(self):
        head = json.dumps({"op": "render"}).encode("utf-8")
        raw = (worker._LENGTH.pack(len(head)) + head
               + worker._LENGTH.pack(worker.MAX_PAYLOAD + 1))
        with pytest.raises(worker.Refusal):
            worker.read_frame(io.BytesIO(raw))

    def test_end_of_input_is_none_and_not_an_error(self):
        assert worker.read_frame(io.BytesIO(b"")) is None


class TestErrorsCarryNothingPrivate:
    def test_a_library_exception_crosses_as_its_class_only(self):
        exc = RuntimeError("failed on /home/someone/voice_box/samples/abc/sample.wav")
        assert worker._safe(exc) == "RuntimeError"

    def test_this_workers_own_refusals_are_sentences(self):
        assert worker._safe(worker.Refusal("the script has no words")) == "the script has no words"

    def test_a_library_value_error_is_not_mistaken_for_one_of_this_files(self):
        try:
            json.loads("{" + '"sample": "/home/someone/samples/abc.wav"')
        except ValueError as exc:
            found = worker._safe(exc)
        assert found == "JSONDecodeError", found
        assert worker._safe(ValueError("could not broadcast (2,) into (3,)")) == "ValueError"

    def test_every_refusal_this_file_raises_is_one_of_its_own(self):
        source = pathlib.Path(worker.__file__).read_text(encoding="utf-8")
        assert "raise ValueError(" not in source

    def test_module_level_imports_are_the_standard_library_only(self):
        """The parent imports this file under Forge's interpreter to read the
        frame format; the day it needs Torch is the day the WebUI stops
        starting."""
        import ast

        found = set()
        for node in ast.parse(pathlib.Path(worker.__file__).read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
        for name in found:
            assert name in sys.stdlib_module_names, name
        for engine in ("torch", "numpy", "transformers", "vibevoice", "peft", "bitsandbytes",
                       "safetensors"):
            assert engine not in found


# --------------------------------------------------------------------------- #
# The script and the voices
# --------------------------------------------------------------------------- #


class TestTheScriptIsWrittenTheWayTheProcessorReadsIt:
    def test_one_line_per_entry_numbered_as_the_caller_numbered(self):
        text, speakers = worker.script_text([{"speaker": 1, "text": "Hello there."},
                                             {"speaker": 2, "text": "Hi."},
                                             {"speaker": 1, "text": "Bye."}])
        assert text == "Speaker 1: Hello there.\nSpeaker 2: Hi.\nSpeaker 1: Bye."
        assert speakers == [1, 2]

    def test_every_line_parses_with_the_processors_own_regex(self):
        text, _speakers = worker.script_text([{"speaker": 2, "text": "Two"},
                                              {"speaker": 1, "text": "One: with a colon"}])
        for line in text.split("\n"):
            found = PROCESSOR_LINE.match(line)
            assert found is not None, line
        assert [PROCESSOR_LINE.match(line).group(2) for line in text.split("\n")] \
            == ["Two", "One: with a colon"]

    def test_the_voice_order_is_by_speaker_number_not_first_appearance(self):
        """The processor binds ``voice_samples[i]`` to the line numbered i
        (after taking one off every number), so Speaker 1's sample has to come
        first even when Speaker 2 speaks first."""
        text, speakers = worker.script_text([{"speaker": 2, "text": "I go first."},
                                             {"speaker": 1, "text": "I go second."}])
        assert text == "Speaker 2: I go first.\nSpeaker 1: I go second."
        assert speakers == [1, 2]

    def test_sparse_numbers_are_made_dense_and_the_order_says_which_is_which(self):
        """Left as 1 and 3, the processor would take the two samples as
        Speakers 0 and 1 and give the third speaker no voice at all."""
        text, speakers = worker.script_text([{"speaker": 1, "text": "A"},
                                             {"speaker": 3, "text": "C"},
                                             {"speaker": 1, "text": "A again"}])
        assert text == "Speaker 1: A\nSpeaker 2: C\nSpeaker 1: A again"
        assert speakers == [1, 3]

    def test_newlines_inside_an_entry_collapse_and_the_apostrophe_is_straightened(self):
        """The processor drops a line that does not begin with ``Speaker N:``,
        so a paragraph break inside one entry would lose its second half. The
        curly apostrophe is what upstream's own demo straightens."""
        text, _speakers = worker.script_text([{"speaker": 1,
                                               "text": "First\nsecond\r\n  third’s"}])
        assert text == "Speaker 1: First second third's"

    def test_entries_without_words_are_skipped_and_a_wordless_script_is_refused(self):
        text, speakers = worker.script_text([{"speaker": 2, "text": "  \n "},
                                             {"speaker": 1, "text": "Words"}])
        assert text == "Speaker 1: Words"
        assert speakers == [1]
        with pytest.raises(worker.Refusal) as raised:
            worker.script_text([{"speaker": 1, "text": ""}])
        assert "no words" in str(raised.value)
        with pytest.raises(worker.Refusal):
            worker.script_text([])

    @pytest.mark.parametrize("speaker", [0, 5, -1, "two", None, True])
    def test_a_speaker_outside_one_to_four_is_refused(self, speaker):
        with pytest.raises(worker.Refusal):
            worker.script_text([{"speaker": speaker, "text": "Hello"}])

    def test_an_entry_that_is_not_an_object_is_refused(self):
        with pytest.raises(worker.Refusal):
            worker.script_text(["Speaker 1: hello"])


class TestVoiceSamplesFollowTheSpeakerOrder:
    def test_each_speaker_gets_its_own_samples_in_speaker_order(self):
        one, two = pcm(0.5, 0.25), pcm(0.25, -0.5)
        payload = two + one
        voices = [{"speaker": 2, "offset": 0, "count": len(two) // 4, "rate": RATE},
                  {"speaker": 1, "offset": len(two) // 4, "count": len(one) // 4, "rate": RATE}]
        arrays = worker.voice_arrays(voices, payload, [1, 2])
        assert len(arrays) == 2
        assert arrays[0].dtype == numpy.float32 and arrays[1].dtype == numpy.float32
        assert len(arrays[0]) == len(one) // 4 and float(arrays[0][0]) == 0.25
        assert len(arrays[1]) == len(two) // 4 and float(arrays[1][0]) == -0.5
        assert arrays[0].flags.writeable

    def test_a_speaker_without_a_sample_is_refused_by_name(self):
        one = pcm(0.5)
        with pytest.raises(worker.Refusal) as raised:
            worker.voice_arrays([{"speaker": 1, "offset": 0, "count": len(one) // 4,
                                  "rate": RATE}], one, [1, 2])
        assert str(raised.value) == "Speaker 2 has no voice sample"

    def test_a_sample_outside_the_payload_is_refused(self):
        one = pcm(0.5)
        for offset, count in ((0, len(one) // 4 + 1), (-1, 10), (0, 0)):
            with pytest.raises(worker.Refusal) as raised:
                worker.voice_arrays([{"speaker": 1, "offset": offset, "count": count,
                                      "rate": RATE}], one, [1])
            assert "outside" in str(raised.value)

    def test_a_sample_at_another_rate_is_refused(self):
        one = pcm(0.5)
        with pytest.raises(worker.Refusal) as raised:
            worker.voice_arrays([{"speaker": 1, "offset": 0, "count": len(one) // 4,
                                  "rate": 48000}], one, [1])
        assert "48000" in str(raised.value)

    def test_two_samples_for_one_speaker_and_non_numbers_are_refused(self):
        one = pcm(0.5)
        entry = {"speaker": 1, "offset": 0, "count": len(one) // 4, "rate": RATE}
        with pytest.raises(worker.Refusal):
            worker.voice_arrays([entry, dict(entry)], one, [1])
        bad = numpy.array([0.1, float("nan"), 0.2], dtype="<f4").tobytes()
        with pytest.raises(worker.Refusal):
            worker.voice_arrays([{"speaker": 1, "offset": 0, "count": 3, "rate": RATE}],
                                bad, [1])

    def test_extra_voices_for_speakers_the_script_does_not_use_are_left_out(self):
        one = pcm(0.5)
        arrays = worker.voice_arrays(
            [{"speaker": 1, "offset": 0, "count": len(one) // 4, "rate": RATE},
             {"speaker": 3, "offset": 0, "count": 10, "rate": RATE}], one, [1])
        assert len(arrays) == 1


class TestAudioLeavesAsPcm16Wav:
    def test_samples_are_clipped_and_quantised(self):
        found = numpy.frombuffer(worker.pcm16([0.0, 1.0, -1.0, 2.0, -2.0, 0.5]), dtype="<i2")
        assert list(found) == [0, 32767, -32767, 32767, -32767, 16383]

    def test_the_wav_is_mono_sixteen_bit_at_the_models_rate(self):
        import wave

        body = worker.pcm16(numpy.zeros(2400, dtype=numpy.float32))
        with wave.open(io.BytesIO(worker.encode_wav(body, RATE)), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == RATE
            assert handle.getnframes() == 2400


def parsed(script=None, voices=None, payload=b"", **values):
    """A render header with preset voices, parsed the way the loop parses it."""
    header = {"op": "render", "job": "j1",
              "script": script if script is not None else [{"speaker": 1, "text": "Hello there."}],
              "voices": voices if voices is not None else [{"speaker": 1,
                                                            "preset": "en-Carter_man"}]}
    header.update(values)
    return worker.RenderRequest.parse(header, payload)


class TestPresetVoicesAndStreamedRequestsParse:
    def test_a_preset_voice_is_a_stem_per_speaker_and_carries_no_audio(self):
        request = parsed()
        assert request.presets == {1: "en-Carter_man"}
        assert request.voices == []
        assert request.stream is False

    @pytest.mark.parametrize("stem", ["../en-Carter_man", "en.Carter", "en Carter", "",
                                      "a" * 65, "C:evil", "voices/en", 7, None])
    def test_a_preset_name_that_could_leave_the_voices_folder_is_refused(self, stem):
        with pytest.raises(worker.Refusal) as raised:
            parsed(voices=[{"speaker": 1, "preset": stem}])
        assert "letters, digits" in str(raised.value)

    def test_presets_and_samples_are_never_mixed(self):
        one = pcm(0.5)
        with pytest.raises(worker.Refusal) as raised:
            parsed(script=[{"speaker": 1, "text": "A"}, {"speaker": 2, "text": "B"}],
                   voices=[{"speaker": 1, "preset": "en-Carter_man"},
                           {"speaker": 2, "offset": 0, "count": len(one) // 4, "rate": RATE}],
                   payload=one)
        assert "both preset voices and voice samples" in str(raised.value)

    def test_a_speaker_without_a_preset_is_refused_by_name(self):
        with pytest.raises(worker.Refusal) as raised:
            parsed(script=[{"speaker": 1, "text": "A"}, {"speaker": 2, "text": "B"}])
        assert str(raised.value) == "Speaker 2 has no voice"

    def test_two_presets_for_one_speaker_are_refused(self):
        with pytest.raises(worker.Refusal):
            parsed(voices=[{"speaker": 1, "preset": "en-Carter_man"},
                           {"speaker": 1, "preset": "en-Emma_woman"}])

    def test_presets_for_speakers_the_script_does_not_use_are_left_out(self):
        request = parsed(voices=[{"speaker": 1, "preset": "en-Carter_man"},
                                 {"speaker": 3, "preset": "en-Emma_woman"}])
        assert request.presets == {1: "en-Carter_man"}

    def test_only_a_literal_true_streams(self):
        assert parsed(stream=True).stream is True
        for value in (False, None, 1, "true", "yes"):
            assert parsed(stream=value).stream is False, value

    def test_an_absent_guidance_is_left_to_the_model(self):
        assert parsed().cfg_scale is None
        assert parsed(cfg_scale=1.7).cfg_scale == 1.7
        with pytest.raises(worker.Refusal):
            parsed(cfg_scale="loud")

    def test_the_realtime_text_is_prose_with_upstreams_quotes_straightened(self):
        request = parsed(script=[{"speaker": 1, "text": "“Hello,” she said.\n It’s  late."},
                                 {"speaker": 1, "text": "Go  on."}])
        assert request.plain == "\"Hello,\" she said. It's late. Go on."
        assert request.text == ("Speaker 1: “Hello,” she said. It's late.\n"
                                "Speaker 1: Go on."), "the 7B's text keeps upstream's own"


# --------------------------------------------------------------------------- #
# The upstream binding, against stand-ins
# --------------------------------------------------------------------------- #


class FakeTensor:
    def __init__(self, data=(), shape=None):
        self.data = list(data)
        self.shape = tuple(shape) if shape is not None else (1, len(self.data))
        self.device = "cpu"

    def to(self, target):
        found = FakeTensor(self.data, self.shape)
        found.device = target if isinstance(target, str) else self.device
        return found

    def detach(self):
        return self

    def reshape(self, *_shape):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return numpy.asarray(self.data, dtype=numpy.float32)

    def any(self):
        return types.SimpleNamespace(item=lambda: any(self.data))

    def __iter__(self):
        """A one-dimensional tensor iterates its elements, each with ``item()``."""
        return (types.SimpleNamespace(item=lambda value=value: value) for value in self.data)

    def __getitem__(self, index):
        """Row ``index`` of a batch: what upstream's streamer queues per sample;
        ``[0, -1]`` is one element, the last token of the first sequence."""
        if isinstance(index, tuple):
            value = self.data[index[-1]]
            return types.SimpleNamespace(item=lambda: value)
        return FakeTensor(self.data, self.shape[1:] or (len(self.data),))


class FakeCuda:
    def __init__(self, available=True, uuid=UUID):
        self.available = available
        self.uuid = uuid
        self.calls = []
        self.reserved = 18_000_000_000
        self.peak = 20_000_000_000

    def is_available(self):
        return self.available

    def get_device_properties(self, index):
        self.calls.append(("properties", index))
        found = types.SimpleNamespace(name="Fake GPU 24GB")
        if self.uuid is not None:
            found.uuid = self.uuid
        return found

    def mem_get_info(self, index):
        return 23_000_000_000, 24_000_000_000

    def memory_reserved(self, index):
        return self.reserved

    def max_memory_reserved(self, index):
        self.calls.append(("max_memory_reserved", index))
        return self.peak

    def reset_peak_memory_stats(self, index):
        self.calls.append(("reset_peak_memory_stats", index))

    def manual_seed_all(self, seed):
        self.calls.append(("manual_seed_all", seed))

    def empty_cache(self):
        self.calls.append(("empty_cache",))


class FakeSerialization:
    """``torch.serialization.safe_globals``: records what was allowed, and when."""

    def __init__(self):
        self.allowed = []
        self.active = False

    def safe_globals(self, classes):
        serialization = self

        class Allowed:
            def __enter__(self):
                serialization.allowed.append(list(classes))
                serialization.active = True

            def __exit__(self, *exc):
                serialization.active = False
                return False

        return Allowed()


class FakeHidden:
    """A tensor that knows its dtype, for the preset's hidden states and caches."""

    def __init__(self, dtype="bfloat16", length=4):
        self.dtype = dtype
        self.length = length

    def to(self, dtype):
        return FakeHidden(dtype, self.length)

    def size(self, dim):
        return self.length


def preset_file(dtype="bfloat16", keys=worker.PRESET_KEYS):
    """What ``torch.load`` gives for a preset: four prefilled passes."""
    return {key: types.SimpleNamespace(
        last_hidden_state=FakeHidden(dtype),
        past_key_values=types.SimpleNamespace(key_cache=[FakeHidden(dtype), FakeHidden(dtype)],
                                              value_cache=[FakeHidden(dtype)]))
        for key in keys}


class FakeTorch:
    __version__ = "2.8.0+cu128"
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self, cuda=None):
        self.cuda = cuda or FakeCuda()
        self.seeds = []
        self.serialization = FakeSerialization()
        self.loads = []
        self.files = {}

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def load(self, path, map_location=None, weights_only=None):
        self.loads.append({"path": path, "map_location": map_location,
                           "weights_only": weights_only,
                           "inside_safe_globals": self.serialization.active})
        found = self.files.get(path)
        if isinstance(found, Exception):
            raise found
        return found if found is not None else preset_file()


class FakeProcessor:
    made = []

    def __init__(self, path, prompt_length=40):
        self.path = path
        self.prompt_length = prompt_length
        self.calls = []
        self.tokenizer = types.SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [1] * len(str(text).split()))

    @classmethod
    def from_pretrained(cls, path):
        found = cls(path)
        cls.made.append(found)
        return found

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": FakeTensor([0] * self.prompt_length, (1, self.prompt_length)),
                "attention_mask": FakeTensor([1] * self.prompt_length, (1, self.prompt_length)),
                "speech_tensors": FakeTensor([0.0] * 10, (1, 10)),
                "speech_masks": FakeTensor([True], (1, 1)),
                "speech_input_mask": FakeTensor([False] * self.prompt_length,
                                                (1, self.prompt_length)),
                "parsed_scripts": [[(0, " hello")]],
                "all_speakers_list": [[0]]}


class FakeModule:
    """One of the 7B's submodules: a state dict's keys and a strict load."""

    def __init__(self, name, keys=("weight",)):
        self.name = name
        self.keys = tuple(keys)
        self.loaded = None

    def state_dict(self):
        return {key: 0 for key in self.keys}

    def load_state_dict(self, state, strict=True):
        if strict and set(state) != set(self.keys):
            raise RuntimeError(f"size mismatch for {self.name} in /home/someone/lora/x.bin")
        self.loaded = dict(state)

    def modules(self):
        yield self


class FakeModel:
    made = []

    def __init__(self, path, kwargs, steps=6, gate=None, capped=False):
        self.path = path
        self.load_kwargs = kwargs
        self.steps = steps
        self.gate = gate
        self.capped = capped
        self.evaluated = 0
        self.step_settings = []
        self.generate_calls = []
        self.generate_threads = []
        self.config = types.SimpleNamespace(
            decoder_config=types.SimpleNamespace(max_position_embeddings=32768))
        self.model = types.SimpleNamespace(
            language_model=FakeModule("language_model", ("layers.0.q_proj.weight",)),
            prediction_head=FakeModule("prediction_head", ("proj.weight", "proj.bias")),
            acoustic_connector=FakeModule("acoustic_connector", ("fc1.weight",)),
            semantic_connector=FakeModule("semantic_connector", ("fc1.weight",)))

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        found = cls(path, kwargs)
        cls.made.append(found)
        return found

    def eval(self):
        self.evaluated += 1

    def set_ddpm_inference_steps(self, num_steps=None):
        self.step_settings.append(num_steps)

    def parameters(self):
        return [types.SimpleNamespace(numel=lambda: 1000, element_size=lambda: 2),
                types.SimpleNamespace(numel=lambda: 500, element_size=lambda: 2)]

    def modules(self):
        yield self
        for part in (self.model.language_model, self.model.prediction_head,
                     self.model.acoustic_connector, self.model.semantic_connector):
            yield from part.modules()

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        self.generate_threads.append(threading.current_thread().name)
        stop = kwargs.get("stop_check_fn")
        streamer = kwargs.get("audio_streamer")
        made = []
        for step in range(self.steps):
            if self.gate is not None:
                self.gate(step)
            if stop is not None and stop():
                if streamer is not None:
                    streamer.end()
                break
            chunk = [0.5 - step / 20.0] * 3200
            made.extend(chunk)
            if streamer is not None:
                streamer.put(FakeTensor(chunk, (1, 1, 3200)), FakeTensor([0], (1,)))
        if streamer is not None:
            streamer.end()
        prompt = kwargs["input_ids"].shape[-1]
        return types.SimpleNamespace(
            speech_outputs=[FakeTensor(made, (1, len(made)))] if made else [None],
            sequences=FakeTensor([0] * (prompt + len(made) // 3200),
                                 (1, prompt + len(made) // 3200)),
            reach_max_step_sample=FakeTensor([self.capped], (1,)))


class FakeAudioStreamer:
    """Upstream's ``AudioStreamer``, line for line in the parts the worker leans
    on: one queue per sample, a chunk dropped once its sample has ended, the end
    marked by ``stop_signal`` exactly once, and ``get_stream`` iterating until it."""

    made = []

    def __init__(self, batch_size, stop_signal=None, timeout=None):
        self.batch_size = batch_size
        self.stop_signal = stop_signal
        self.timeout = timeout
        self.audio_queues = [queue.Queue() for _ in range(batch_size)]
        self.finished_flags = [False for _ in range(batch_size)]
        self.put_threads = []
        FakeAudioStreamer.made.append(self)

    def put(self, audio_chunks, sample_indices):
        self.put_threads.append(threading.current_thread().name)
        for i, sample_idx in enumerate(sample_indices):
            idx = sample_idx.item()
            if idx < self.batch_size and not self.finished_flags[idx]:
                self.audio_queues[idx].put(audio_chunks[i], timeout=self.timeout)

    def end(self, sample_indices=None):
        indices = range(self.batch_size) if sample_indices is None else \
            [one.item() if hasattr(one, "item") else one for one in sample_indices]
        for idx in indices:
            if idx < self.batch_size and not self.finished_flags[idx]:
                self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                self.finished_flags[idx] = True

    def get_stream(self, sample_idx):
        while True:
            value = self.audio_queues[sample_idx].get(timeout=self.timeout)
            if value is self.stop_signal:
                return
            yield value


class FakeScheduler:
    """The noise scheduler's ``config`` and ``from_config``, as diffusers has them."""

    def __init__(self, config=None, overrides=None):
        self.config = config if config is not None else {"beta_schedule": "cosine",
                                                         "prediction_type": "v_prediction"}
        self.overrides = dict(overrides or {})

    def from_config(self, config, **kwargs):
        return FakeScheduler(config, kwargs)


class FakeRealtimeProcessor:
    made = []

    def __init__(self, path):
        self.path = path
        self.calls = []
        self.tokenizer = types.SimpleNamespace(name="the realtime tokenizer")

    @classmethod
    def from_pretrained(cls, path):
        found = cls(path)
        cls.made.append(found)
        return found

    def process_input_with_cached_prompt(self, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": FakeTensor([0] * 8, (1, 8)),
                "attention_mask": FakeTensor([1] * 8, (1, 8)),
                "tts_lm_input_ids": FakeTensor([0] * 12, (1, 12)),
                "tts_lm_attention_mask": FakeTensor([1] * 12, (1, 12)),
                "tts_text_ids": FakeTensor([5] * 6, (1, 6)),
                "speech_input_mask": FakeTensor([False] * 12, (1, 12)),
                "speech_tensors": None, "speech_masks": None}


class FakeRealtimeModel:
    """The 0.5B's shape: a scheduler to replace, and a ``generate`` that keeps
    speaking for the rest of its window after deciding to stop -- audio the
    streamer, and the worker, must not keep."""

    made = []

    def __init__(self, path, kwargs):
        self.path = path
        self.load_kwargs = kwargs
        self.evaluated = 0
        self.step_settings = []
        self.generate_calls = []
        self.chunks = 5
        self.stop_after = None
        self.gate = None
        self.window = 1
        self.model = types.SimpleNamespace(noise_scheduler=FakeScheduler())

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        found = cls(path, kwargs)
        cls.made.append(found)
        return found

    def eval(self):
        self.evaluated += 1

    def set_ddpm_inference_steps(self, num_steps=None):
        self.step_settings.append(num_steps)

    def parameters(self):
        return [types.SimpleNamespace(numel=lambda: 100, element_size=lambda: 2)]

    def modules(self):
        yield self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        streamer = kwargs["audio_streamer"]
        stop = kwargs["stop_check_fn"]
        made = []
        for step in range(self.chunks):
            if self.gate is not None:
                self.gate(step)
            if step % self.window == 0 and stop():
                streamer.end()
                break
            chunk = [0.25] * 3200
            if self.stop_after is None or step < self.stop_after:
                made.extend(chunk)
            streamer.put(FakeTensor(chunk, (1, 1, 3200)), FakeTensor([0], (1,)))
            if self.stop_after is not None and step + 1 == self.stop_after:
                streamer.end(FakeTensor([0], (1,)))
        streamer.end()
        return types.SimpleNamespace(
            speech_outputs=[FakeTensor(made, (1, len(made)))] if made else [None],
            sequences=FakeTensor([0] * 40, (1, 40)),
            reach_max_step_sample=FakeTensor([False], (1,)))


class FakeBnbConfig:
    """``transformers.BitsAndBytesConfig``, recording what it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLoraConfig:
    made = []

    def __init__(self, folder):
        self.folder = folder
        self.task_type = "CAUSAL_LM"
        self.inference_mode = False

    @classmethod
    def from_pretrained(cls, folder):
        found = cls(folder)
        cls.made.append(found)
        return found


class FakeLoraLayer:
    def __init__(self):
        self.scaling = {"default": 2.0}

    def modules(self):
        yield self


class FakePeftModel:
    """``PeftModel``: wraps a module and brings LoRA layers into its tree."""

    calls = []
    refuse = set()

    def __init__(self, base, folder, config, is_trainable):
        self.base = base
        self.folder = folder
        self.config = config
        self.is_trainable = is_trainable
        self.layers = [FakeLoraLayer(), FakeLoraLayer()]

    @classmethod
    def from_pretrained(cls, module, folder, config=None, is_trainable=True, **kwargs):
        cls.calls.append({"module": module, "folder": folder, "config": config,
                          "task_type": getattr(config, "task_type", "unset"),
                          "is_trainable": is_trainable, "kwargs": kwargs})
        if os.path.basename(folder) in cls.refuse:
            raise ValueError(f"Target modules not found in {folder}")
        return cls(module, folder, config, is_trainable)

    def modules(self):
        yield self
        for layer in self.layers:
            yield from layer.modules()
        yield from self.base.modules()


class PresetOutput:
    """Stands in for ``BaseModelOutputWithPast`` in the allowed classes."""


class PresetCache:
    """Stands in for ``DynamicCache`` in the allowed classes."""


class BoundEngine(worker.Engine):
    """The real engine with every import point answered by stand-ins.

    ``bnb`` and ``peft`` may be an exception instead, which is then raised the
    way the real import points raise it when the runtime lacks the package.
    """

    def __init__(self, torch=None, model_class=FakeModel, processor_class=FakeProcessor,
                 realtime=None, streamer=None, bnb=None, peft=None):
        super().__init__()
        self._stand_ins = (torch or FakeTorch(), model_class, processor_class)
        self._realtime = realtime or (FakeRealtimeModel, FakeRealtimeProcessor)
        self._streamer = streamer or FakeAudioStreamer
        self._bnb = bnb or FakeBnbConfig
        self._peft_stand_ins = peft or (FakeLoraConfig, FakePeftModel, FakeLoraLayer)
        self.states_read = []

    def _frameworks(self):
        self.torch, self.numpy = self._stand_ins[0], numpy
        return self.torch, self.numpy

    def _upstream(self):
        return self._stand_ins[1], self._stand_ins[2]

    def _upstream_realtime(self):
        return self._realtime

    def _streamer_class(self):
        return self._streamer

    def _quantisation_class(self):
        if isinstance(self._bnb, Exception):
            raise self._bnb
        return self._bnb

    def _peft(self):
        if isinstance(self._peft_stand_ins, Exception):
            raise self._peft_stand_ins
        return self._peft_stand_ins

    def _preset_classes(self):
        return PresetOutput, PresetCache

    def _read_state(self, path):
        self.states_read.append(os.path.basename(path))
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _forget_stand_ins():
    stand_ins = (FakeModel.made, FakeProcessor.made, FakeRealtimeModel.made,
                 FakeRealtimeProcessor.made, FakeAudioStreamer.made, FakeLoraConfig.made,
                 FakePeftModel.calls)
    for made in stand_ins:
        made.clear()
    FakePeftModel.refuse.clear()
    yield
    for made in stand_ins:
        made.clear()
    FakePeftModel.refuse.clear()


def request_for(script=None, voices=None, **values):
    script = script if script is not None else [{"speaker": 1, "text": "Hello there."}]
    if voices is None:
        one = pcm(0.5)
        voices = ([{"speaker": 1, "offset": 0, "count": len(one) // 4, "rate": RATE}], one)
    entries, payload = voices
    header = {"op": "render", "job": "j1", "script": script, "voices": entries,
              "cfg_scale": 1.3, "seed": None, "max_new_tokens": None, "steps": 10}
    header.update(values)
    return worker.RenderRequest.parse(header, payload)


class TestTheCardIsProvedAtInit:
    def test_the_probe_reports_torchs_uuid_as_a_string(self):
        found = BoundEngine().probe()
        assert found["cuda"] is True
        assert found["device_uuid"] == UUID
        assert found["device_name"] == "Fake GPU 24GB"
        assert found["device_index"] == 0
        assert found["total_vram_bytes"] == 24_000_000_000
        assert found["free_vram_bytes"] == 23_000_000_000
        assert found["torch"] == "2.8.0+cu128"

    def test_a_torch_without_a_device_uuid_reports_an_empty_string(self):
        found = BoundEngine(FakeTorch(FakeCuda(uuid=None))).probe()
        assert found["device_uuid"] == ""

    def test_no_cuda_device_is_the_sentence_the_parent_expects(self):
        with pytest.raises(worker.Refusal) as raised:
            BoundEngine(FakeTorch(FakeCuda(available=False))).probe()
        assert str(raised.value) == "no CUDA device is visible to the worker"


class TestLoadingCallsUpstreamExactly:
    def test_the_processor_and_model_are_loaded_from_the_directory_in_bf16_on_cuda(self, tmp_path):
        engine = BoundEngine()
        found = engine.load(str(tmp_path), 10, "bf16", "sdpa")
        assert FakeProcessor.made[0].path == str(tmp_path)
        model = FakeModel.made[0]
        assert model.path == str(tmp_path)
        assert model.load_kwargs == {"torch_dtype": "bfloat16", "device_map": "cuda",
                                     "attn_implementation": "sdpa"}
        assert model.evaluated == 1
        assert model.step_settings == [10]
        assert found["weights_bytes"] == 3000
        assert found["resident_bytes"] == 18_000_000_000
        assert found["load_seconds"] >= 0.0
        assert engine.model is model and engine.model_dir == str(tmp_path)

    def test_loading_twice_is_a_no_op_reply(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        found = engine.load(str(tmp_path), 10)
        assert found.get("already_loaded") is True
        assert len(FakeModel.made) == 1 and len(FakeProcessor.made) == 1

    def test_a_missing_directory_is_refused_without_naming_it(self, tmp_path):
        missing = tmp_path / "nothing-here"
        with pytest.raises(worker.Refusal) as raised:
            BoundEngine().load(str(missing), 10)
        assert str(missing) not in str(raised.value)
        assert "does not exist" in str(raised.value)

    def test_unloading_lets_go_and_empties_the_cache(self, tmp_path):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(str(tmp_path), 10)
        found = engine.unload()
        assert engine.model is None and engine.processor is None and engine.model_dir == ""
        assert ("empty_cache",) in torch.cuda.calls
        assert "resident_bytes" in found

    def test_rendering_without_a_model_is_refused(self):
        with pytest.raises(worker.Refusal) as raised:
            BoundEngine().render(request_for(), threading.Event())
        assert "no model is loaded" in str(raised.value)


class TestRenderingCallsUpstreamExactly:
    def test_the_processor_call_the_seed_and_the_generate_call(self, tmp_path):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(str(tmp_path), 10)
        one, two = pcm(0.5, 0.25), pcm(0.25, -0.5)
        request = request_for(
            script=[{"speaker": 2, "text": "Second speaks first."},
                    {"speaker": 1, "text": "First speaks second."}],
            voices=([{"speaker": 1, "offset": 0, "count": len(one) // 4, "rate": RATE},
                     {"speaker": 2, "offset": len(one) // 4, "count": len(two) // 4,
                      "rate": RATE}], one + two),
            seed=7, cfg_scale=1.7, max_new_tokens=None)
        progress = []
        reply, audio = engine.render(request, threading.Event(), progress.append)

        call = FakeProcessor.made[0].calls[0]
        assert call["text"] == ["Speaker 2: Second speaks first.\nSpeaker 1: First speaks second."]
        assert call["padding"] is True
        assert call["return_tensors"] == "pt"
        assert call["return_attention_mask"] is True
        assert len(call["voice_samples"]) == 1 and len(call["voice_samples"][0]) == 2
        assert float(call["voice_samples"][0][0][0]) == 0.25, "Speaker 1's sample first"
        assert float(call["voice_samples"][0][1][0]) == -0.5

        assert torch.seeds == [7]
        assert ("manual_seed_all", 7) in torch.cuda.calls

        generated = FakeModel.made[0].generate_calls[0]
        assert generated["input_ids"].device == "cuda"
        assert generated["attention_mask"].device == "cuda"
        assert generated["speech_tensors"].device == "cuda"
        assert generated["cfg_scale"] == 1.7
        assert generated["tokenizer"] is FakeProcessor.made[0].tokenizer
        assert generated["generation_config"] == {"do_sample": False}
        assert generated["verbose"] is False
        assert generated["show_progress_bar"] is False
        assert callable(generated["stop_check_fn"])
        assert generated["max_new_tokens"] == max(worker.MIN_NEW_TOKENS,
                                                  worker.TOKENS_PER_TEXT_TOKEN * 6)
        assert generated["max_length_times"] * 40 >= generated["max_new_tokens"]
        assert generated["audio_streamer"] is not None

        assert reply["seconds"] == pytest.approx(6 * 3200 / RATE)
        assert reply["sample_rate"] == RATE
        assert reply["cancelled"] is False
        assert reply["tokens"] == 6
        assert reply["peak_bytes"] == 20_000_000_000
        assert reply["capped"] is False
        assert reply["render_seconds"] >= 0.0
        assert audio[:4] == b"RIFF" and len(audio) == 44 + 6 * 3200 * 2
        assert progress and all("seconds" in one for one in progress)

    def test_the_peak_is_reset_before_the_render_and_read_after(self, tmp_path):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(str(tmp_path), 10)
        torch.cuda.calls.clear()
        engine.render(request_for(), threading.Event())
        names = [call[0] for call in torch.cuda.calls]
        assert names.index("reset_peak_memory_stats") < names.index("max_memory_reserved")
        assert ("reset_peak_memory_stats", 0) in torch.cuda.calls
        assert "empty_cache" in names, "the transient tensors are given back after a render"

    def test_a_caller_cap_is_passed_as_written_and_no_seed_means_no_seeding(self, tmp_path):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(str(tmp_path), 10)
        engine.render(request_for(max_new_tokens=333, seed=None), threading.Event())
        generated = FakeModel.made[0].generate_calls[0]
        assert generated["max_new_tokens"] == 333
        assert torch.seeds == []

    def test_the_budget_never_passes_the_context_the_prompt_leaves(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        FakeModel.made[0].config.decoder_config.max_position_embeddings = 300
        engine.render(request_for(), threading.Event())
        assert FakeModel.made[0].generate_calls[0]["max_new_tokens"] == 300 - 40

    def test_a_render_with_other_steps_resets_the_solver(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        engine.render(request_for(steps=20), threading.Event())
        assert FakeModel.made[0].step_settings == [10, 20]

    def test_cancellation_is_read_between_steps_and_the_audio_so_far_is_kept(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        cancelled = threading.Event()

        def gate(step):
            if step == 2:
                cancelled.set()

        FakeModel.made[0].gate = gate
        reply, audio = engine.render(request_for(), cancelled)
        assert reply["cancelled"] is True
        assert reply["tokens"] == 2
        assert len(audio) == 44 + 2 * 3200 * 2

    def test_a_render_that_hit_its_budget_says_so(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        FakeModel.made[0].capped = True
        reply, _audio = engine.render(request_for(), threading.Event())
        assert reply["capped"] is True

    def test_no_audio_at_all_is_an_empty_wav_rather_than_an_error(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        FakeModel.made[0].steps = 0
        reply, audio = engine.render(request_for(), threading.Event())
        assert reply["seconds"] == 0.0
        assert len(audio) == 44


class TestProgressCostsTheRenderNothing:
    def test_samples_are_counted_from_the_shape_and_reported_at_intervals(self):
        seen = []
        progress = worker.Progress(RATE, seen.append, interval=0.0)
        for _step in range(3):
            progress.put(FakeTensor([0.0] * 3200, (1, 1, 3200)), FakeTensor([0]))
        assert progress.samples == 9600
        assert seen[-1] == {"seconds": pytest.approx(9600 / RATE)}
        assert progress.finished_flags == [False]
        progress.end(FakeTensor([0]))
        assert progress.finished_flags == [True]

    def test_the_interval_holds_frames_back(self):
        seen = []
        progress = worker.Progress(RATE, seen.append, interval=60.0)
        for _step in range(5):
            progress.put(FakeTensor([0.0] * 3200, (1, 1, 3200)), FakeTensor([0]))
        assert len(seen) == 1

    def test_a_chunk_after_the_end_is_not_counted_as_the_real_streamer_drops_it(self):
        """The Realtime model finishes its speech window after it decides to
        stop, and neither ``generate`` nor upstream's streamer keeps that
        audio; progress must not count it either."""
        progress = worker.Progress(RATE, None, interval=0.0)
        progress.put(FakeTensor([0.0] * 3200, (1, 1, 3200)), FakeTensor([0]))
        progress.end(FakeTensor([0]))
        progress.put(FakeTensor([0.0] * 3200, (1, 1, 3200)), FakeTensor([0]))
        assert progress.samples == 3200 and progress.chunks == 1


# --------------------------------------------------------------------------- #
# What a load is: its identity, and every refusal before anything is unloaded
# --------------------------------------------------------------------------- #


def lora_folder(root, *, adapter="", parts=None, peft_type="LORA"):
    """A LoRA folder on disk. The adapter pair's weights are PEFT's to read and
    are placeholders here; a part is either ``"adapter"`` or ``(file, state)``
    with the state written as JSON for ``BoundEngine._read_state``."""
    root = pathlib.Path(root)
    target = root / adapter if adapter else root
    target.mkdir(parents=True, exist_ok=True)
    (target / "adapter_config.json").write_text(
        json.dumps({"peft_type": peft_type, "r": 4, "lora_alpha": 8,
                    "task_type": "CAUSAL_LM"}), encoding="utf-8")
    (target / "adapter_model.safetensors").write_bytes(b"placeholder")
    for name, spec in (parts or {}).items():
        folder = root / name
        folder.mkdir()
        if spec == "adapter":
            (folder / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}),
                                                        encoding="utf-8")
            (folder / "adapter_model.safetensors").write_bytes(b"placeholder")
        else:
            for filename, state in (spec if isinstance(spec, list) else [spec]):
                (folder / filename).write_text(json.dumps(state), encoding="utf-8")
    return str(root)


@pytest.fixture
def model_dir(tmp_path):
    found = tmp_path / "model"
    found.mkdir()
    return str(found)


@pytest.fixture
def voices(tmp_path):
    found = tmp_path / "voices"
    found.mkdir()
    for stem in ("en-Carter_man", "en-Emma_woman"):
        (found / f"{stem}.pt").write_bytes(b"placeholder")
    return str(found)


class TestALoadIsAnIdentity:
    def test_the_same_identity_is_a_no_op_and_says_what_is_loaded(self, model_dir):
        engine = BoundEngine()
        first = engine.load(model_dir, 10)
        again = engine.load(model_dir, 10, kind="longform", precision="bf16", lora_dir=None,
                            lora_scale=1.0, device="cuda")
        assert "already_loaded" not in first
        assert again["already_loaded"] is True
        assert len(FakeModel.made) == 1
        assert engine.identity() == (model_dir, "longform", "bf16", "", 1.0, "cuda")
        for reply in (first, again):
            assert reply["kind"] == "longform" and reply["precision"] == "bf16"
            assert reply["lora"] is False and reply["device"] == "cuda"

    @pytest.mark.parametrize("change", [
        {"precision": "int8"}, {"precision": "nf4"}, {"lora": True},
        {"lora": True, "lora_scale": 0.5}, {"device": "cpu"}, {"kind": "realtime"},
        {"other_dir": True}])
    def test_any_other_identity_unloads_first(self, tmp_path, model_dir, change):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        lora = lora_folder(tmp_path / "lora")
        base = {"lora_dir": lora} if change.get("lora") and "lora_scale" in change else {}
        engine.load(model_dir, 10, **base)
        loaded = engine.model
        torch.cuda.calls.clear()
        wanted = dict(base)
        for key in ("precision", "device", "kind", "lora_scale"):
            if key in change:
                wanted[key] = change[key]
        if change.get("lora"):
            wanted["lora_dir"] = lora
        directory = model_dir
        if change.get("other_dir"):
            directory = str(tmp_path / "other")
            os.mkdir(directory)
        reply = engine.load(directory, 10, **wanted)
        assert "already_loaded" not in reply
        assert engine.model is not loaded, "the old model was replaced"
        assert ("empty_cache",) in torch.cuda.calls, "and let go of before the new one loaded"

    def test_a_no_op_load_applies_new_steps_and_a_new_voices_folder(self, model_dir, voices,
                                                                     tmp_path):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        engine.preset("en-Carter_man")
        assert len(torch.loads) == 1
        other = tmp_path / "other-voices"
        other.mkdir()
        (other / "en-Carter_man.pt").write_bytes(b"placeholder")
        reply = engine.load(model_dir, 7, kind="realtime", voices_dir=str(other))
        assert reply["already_loaded"] is True
        assert FakeRealtimeModel.made[0].step_settings == [5, 7]
        engine.preset("en-Carter_man")
        assert len(torch.loads) == 2, "a new voices folder is a new preset cache"
        assert torch.loads[-1]["path"] == str(other / "en-Carter_man.pt")

    def test_a_strength_without_a_lora_is_not_part_of_what_is_loaded(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        again = engine.load(model_dir, 10, lora_dir=None, lora_scale=0.5)
        assert again["already_loaded"] is True and len(FakeModel.made) == 1
        assert engine.identity()[4] == 1.0

    def test_the_phase_one_dtype_names_the_precision_when_none_is_given(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10, "bf16", "sdpa")
        assert engine.precision == "bf16"
        with pytest.raises(worker.Refusal) as raised:
            BoundEngine().load(model_dir, 10, "fp16", "sdpa")
        assert str(raised.value) == "that precision is not one this worker loads"


class TestRefusalsComeBeforeAnythingIsUnloaded:
    @pytest.mark.parametrize("request_, sentence", [
        ({"kind": "podcast"}, "that kind of model is not one this worker loads"),
        ({"precision": "fp8"}, "that precision is not one this worker loads"),
        ({"device": "tpu"}, "that device is not one this worker uses"),
        ({"lora_scale": 2.5}, "a LoRA's strength is from 0 to 2"),
        ({"lora_scale": -0.1}, "a LoRA's strength is from 0 to 2"),
        ({"lora_scale": float("nan")}, "a number in the request is not a number"),
        ({"kind": "realtime", "precision": "int8"},
         "the Realtime model loads at full precision only"),
        ({"kind": "realtime", "lora_dir": "somewhere"}, "the Realtime model does not take a LoRA"),
        ({"precision": "nf4", "device": "cpu"}, "8-bit and 4-bit loading need a CUDA device"),
        ({"lora_dir": "missing"}, "the LoRA folder does not exist"),
    ])
    def test_a_bad_request_keeps_the_model_that_was_loaded(self, tmp_path, model_dir,
                                                           request_, sentence):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        loaded = engine.model
        values = dict(request_)
        if values.get("lora_dir") == "missing":
            values["lora_dir"] = str(tmp_path / "no-such-lora")
        with pytest.raises(worker.Refusal) as raised:
            engine.load(model_dir, 10, **values)
        assert str(raised.value) == sentence
        assert engine.model is loaded and len(FakeModel.made) == 1

    def test_a_runtime_without_bitsandbytes_keeps_its_model_and_says_reinstall(self, model_dir):
        engine = BoundEngine(bnb=worker.Refusal("the runtime was installed before quantisation "
                                                "was added — reinstall it"))
        engine.load(model_dir, 10)
        loaded = engine.model
        with pytest.raises(worker.Refusal) as raised:
            engine.load(model_dir, 10, precision="int8")
        assert "reinstall it" in str(raised.value)
        assert engine.model is loaded

    def test_a_card_that_went_away_keeps_the_model_it_had(self, model_dir):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(model_dir, 10)
        loaded = engine.model
        torch.cuda.available = False
        with pytest.raises(worker.Refusal) as raised:
            engine.load(model_dir, 10, precision="int8")
        assert str(raised.value) == "no CUDA device is visible to the worker"
        assert engine.model is loaded


class TestQuantisationLeavesEverythingButTheLanguageModelAlone:
    SKIP = ["acoustic_tokenizer", "semantic_tokenizer", "prediction_head",
            "acoustic_connector", "semantic_connector", "lm_head"]

    def test_the_skip_list_is_the_designs(self):
        assert list(worker.QUANTISE_SKIP) == self.SKIP

    def test_eight_bit(self, model_dir):
        BoundEngine().load(model_dir, 10, precision="int8")
        found = FakeModel.made[0].load_kwargs
        assert isinstance(found["quantization_config"], FakeBnbConfig)
        assert found["quantization_config"].kwargs == {"load_in_8bit": True,
                                                        "llm_int8_skip_modules": self.SKIP}
        assert found["torch_dtype"] == "bfloat16" and found["device_map"] == "cuda"

    def test_four_bit_nf4_computes_in_bf16_with_double_quantisation(self, model_dir):
        BoundEngine().load(model_dir, 10, precision="nf4")
        assert FakeModel.made[0].load_kwargs["quantization_config"].kwargs == {
            "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True,
            "llm_int8_skip_modules": self.SKIP}

    def test_full_precision_passes_no_quantisation(self, model_dir):
        BoundEngine().load(model_dir, 10, precision="bf16")
        assert "quantization_config" not in FakeModel.made[0].load_kwargs

    def test_the_real_import_point_turns_a_missing_bitsandbytes_into_a_sentence(self,
                                                                                monkeypatch):
        import builtins

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "bitsandbytes":
                raise ImportError("No module named 'bitsandbytes' at /home/someone/env")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        with pytest.raises(worker.Refusal) as raised:
            worker.Engine()._quantisation_class()
        assert str(raised.value) == ("the runtime was installed before quantisation was added "
                                     "— reinstall it")


class TestALoraIsJoinedWithPeftAndScaled:
    def test_the_language_model_adapter_uses_the_plain_wrapper_and_no_training(self, tmp_path,
                                                                                model_dir):
        lora = lora_folder(tmp_path / "lora")
        engine = BoundEngine()
        reply = engine.load(model_dir, 10, lora_dir=lora, lora_scale=0.5)
        model = FakeModel.made[0]
        call = FakePeftModel.calls[0]
        assert call["folder"] == lora
        assert call["module"].name == "language_model"
        assert call["is_trainable"] is False
        assert call["task_type"] is None, "a causal-LM task would wrap the bare decoder wrongly"
        assert call["config"] is FakeLoraConfig.made[0] and call["config"].folder == lora
        assert call["config"].inference_mode is True
        assert isinstance(model.model.language_model, FakePeftModel)
        assert model.model.language_model.base is call["module"]
        assert reply["lora"] is True and engine.lora_parts == ("llm",)

    def test_the_strength_multiplies_every_lora_layers_scaling(self, tmp_path, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10, lora_dir=lora_folder(tmp_path / "lora", parts={
            "semantic_connector": "adapter"}), lora_scale=0.25)
        model = FakeModel.made[0]
        layers = [module for module in model.modules() if isinstance(module, FakeLoraLayer)]
        assert len(layers) == 4 and engine.lora_layers == 4
        assert all(layer.scaling == {"default": 0.5} for layer in layers)

    def test_the_optional_parts_are_whole_weights_or_adapter_pairs(self, tmp_path, model_dir):
        lora = lora_folder(tmp_path / "lora", parts={
            "diffusion_head": ("diffusion_head_full.bin", {"proj.weight": 1, "proj.bias": 2}),
            "acoustic_connector": ("pytorch_model.bin", {"fc1.weight": 3}),
            "semantic_connector": "adapter"})
        engine = BoundEngine()
        engine.load(model_dir, 10, lora_dir=lora)
        model = FakeModel.made[0]
        assert model.model.prediction_head.loaded == {"proj.weight": 1, "proj.bias": 2}
        assert model.model.acoustic_connector.loaded == {"fc1.weight": 3}
        assert isinstance(model.model.semantic_connector, FakePeftModel)
        assert model.model.semantic_connector.base.name == "semantic_connector"
        assert engine.lora_parts == ("llm", "diffusion_head", "acoustic_connector",
                                     "semantic_connector")
        assert engine.states_read == ["diffusion_head_full.bin", "pytorch_model.bin"]

    def test_safetensors_is_read_before_a_pickle(self, tmp_path, model_dir):
        lora = lora_folder(tmp_path / "lora", parts={"diffusion_head": [
            ("pytorch_model.bin", {"wrong": 0}),
            ("model.safetensors", {"proj.weight": 1, "proj.bias": 2})]})
        engine = BoundEngine()
        engine.load(model_dir, 10, lora_dir=lora)
        assert engine.states_read == ["model.safetensors"]

    def test_weights_saved_with_their_module_path_are_read(self, tmp_path, model_dir):
        lora = lora_folder(tmp_path / "lora", parts={"diffusion_head": (
            "model.safetensors", {"model.prediction_head.proj.weight": 1,
                                  "model.prediction_head.proj.bias": 2})})
        BoundEngine().load(model_dir, 10, lora_dir=lora)
        assert FakeModel.made[0].model.prediction_head.loaded == {"proj.weight": 1,
                                                                   "proj.bias": 2}

    def test_the_adapter_may_sit_in_a_training_scripts_subfolder(self, tmp_path, model_dir):
        lora = lora_folder(tmp_path / "lora", adapter="lora")
        BoundEngine().load(model_dir, 10, lora_dir=lora)
        assert FakePeftModel.calls[0]["folder"] == os.path.join(lora, "lora")

    @pytest.mark.parametrize("setup, sentence", [
        ("no-adapter", "the LoRA folder holds no language-model adapter"),
        ("not-lora", "the LoRA's language-model adapter is not a LoRA"),
        ("peft-refuses", "the LoRA's language-model adapter does not fit this model"),
        ("head-mismatch", "the LoRA's diffusion head does not fit this model"),
        ("connector-empty", "the LoRA's acoustic connector folder holds no weights this worker "
                            "reads"),
        ("semantic-refuses", "the LoRA's semantic connector adapter does not fit this model"),
    ])
    def test_a_lora_the_model_will_not_take_is_refused_naming_the_part(
            self, tmp_path, model_dir, setup, sentence):
        root = tmp_path / "lora"
        if setup == "no-adapter":
            root.mkdir()
            (root / "readme.txt").write_text("nothing here", encoding="utf-8")
            lora = str(root)
        elif setup == "not-lora":
            lora = lora_folder(root, peft_type="IA3")
        elif setup == "peft-refuses":
            lora = lora_folder(root)
            FakePeftModel.refuse.add("lora")
        elif setup == "head-mismatch":
            lora = lora_folder(root, parts={"diffusion_head": ("model.safetensors",
                                                               {"proj.weight": 1})})
        elif setup == "connector-empty":
            lora = lora_folder(root, parts={"acoustic_connector": ("notes.txt", {})})
        else:
            lora = lora_folder(root, parts={"semantic_connector": "adapter"})
            FakePeftModel.refuse.add("semantic_connector")
        torch = FakeTorch()
        engine = BoundEngine(torch)
        with pytest.raises(worker.Refusal) as raised:
            engine.load(model_dir, 10, lora_dir=lora)
        assert str(raised.value) == sentence
        assert "/" not in str(raised.value) and "\\" not in str(raised.value)
        assert engine.model is None, "a model half-joined to a LoRA is not what was asked for"
        assert ("empty_cache",) in torch.cuda.calls

    def test_the_real_import_point_turns_a_missing_peft_into_a_sentence(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name.startswith("peft"):
                raise ImportError("No module named 'peft'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        with pytest.raises(worker.Refusal) as raised:
            worker.Engine()._peft()
        assert str(raised.value) == ("the runtime was installed before LoRA support was added "
                                     "— reinstall it")

    def test_a_prefix_is_only_stripped_when_nothing_else_fits(self):
        expected = {"proj.weight"}
        assert worker._without_prefix({"model.head.proj.weight": 1}, expected,
                                      ("model.head.",)) == {"proj.weight": 1}
        assert worker._without_prefix({"proj.weight": 1, "model.head.x": 2}, expected,
                                      ("model.head.",)) == {"proj.weight": 1, "model.head.x": 2}
        assert worker._without_prefix({"other.proj.weight": 1}, expected,
                                      ("model.head.",)) == {"other.proj.weight": 1}


class TestTheRealtimeModelLoadsAsMicrosoftsDemoLoadsIt:
    def test_its_classes_bf16_sdpa_and_the_scheduler_swap(self, model_dir, voices):
        engine = BoundEngine()
        reply = engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        processor = FakeRealtimeProcessor.made[0]
        model = FakeRealtimeModel.made[0]
        assert processor.path == model_dir and model.path == model_dir
        assert model.load_kwargs == {"torch_dtype": "bfloat16", "device_map": "cuda",
                                     "attn_implementation": "sdpa"}
        assert model.evaluated == 1
        scheduler = model.model.noise_scheduler
        assert scheduler.overrides == {"algorithm_type": "sde-dpmsolver++",
                                       "beta_schedule": "squaredcos_cap_v2"}
        assert scheduler.config == {"beta_schedule": "cosine",
                                    "prediction_type": "v_prediction"}, "its own configuration"
        assert model.step_settings == [5] and engine.steps == 5
        assert reply["kind"] == "realtime" and engine.voices_dir == voices
        assert not FakeModel.made, "the 7B's classes were not touched"

    def test_a_step_count_given_is_the_step_count_used(self, model_dir, voices):
        BoundEngine().load(model_dir, 7, kind="realtime", voices_dir=voices)
        assert FakeRealtimeModel.made[0].step_settings == [7]

    def test_a_runtime_without_the_overlay_is_told_to_reinstall(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name.startswith("vibevoice"):
                raise ImportError("No module named 'vibevoice.modular.streaming'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        with pytest.raises(worker.Refusal) as raised:
            worker.Engine()._upstream_realtime()
        assert "before the Realtime model was added — reinstall it" in str(raised.value)

    def test_on_the_cpu_the_model_is_float32_and_the_card_is_never_touched(self, model_dir):
        torch = FakeTorch(FakeCuda(available=False))
        engine = BoundEngine(torch)
        engine.load(model_dir, 10, device="cpu")
        assert FakeModel.made[0].load_kwargs == {"torch_dtype": "float32", "device_map": "cpu",
                                                 "attn_implementation": "sdpa"}
        reply, _audio = engine.render(request_for(seed=None), threading.Event())
        generated = FakeModel.made[0].generate_calls[0]
        assert generated["input_ids"].device == "cpu"
        touched = {call[0] for call in torch.cuda.calls}
        assert not touched & {"reset_peak_memory_stats", "max_memory_reserved", "empty_cache"}
        assert reply["peak_bytes"] == 0 and reply["resident_bytes"] == 0
        engine.unload()
        assert not {call[0] for call in torch.cuda.calls} & {"empty_cache"}


class TestPresetVoicesAreReadSafelyAndOnce:
    def test_read_under_safe_globals_with_weights_only_onto_the_models_device(self, model_dir,
                                                                             voices):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        engine.render(parsed(), threading.Event())
        assert torch.loads == [{"path": os.path.join(voices, "en-Carter_man.pt"),
                                "map_location": "cuda", "weights_only": True,
                                "inside_safe_globals": True}]
        assert torch.serialization.allowed == [[PresetOutput, PresetCache]]

    def test_each_stem_is_read_once_while_the_model_stays(self, model_dir, voices):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        for _turn in range(3):
            engine.render(parsed(), threading.Event())
        engine.render(parsed(voices=[{"speaker": 1, "preset": "en-Emma_woman"}]),
                      threading.Event())
        assert [os.path.basename(one["path"]) for one in torch.loads] == [
            "en-Carter_man.pt", "en-Emma_woman.pt"]
        engine.unload()
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        engine.render(parsed(), threading.Event())
        assert len(torch.loads) == 3, "the cache ends with the model"

    def test_a_voice_that_is_not_installed_is_refused(self, model_dir, voices):
        engine = BoundEngine()
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        with pytest.raises(worker.Refusal) as raised:
            engine.render(parsed(voices=[{"speaker": 1, "preset": "de-Spk0_man"}]),
                          threading.Event())
        assert str(raised.value) == "that preset voice is not installed"

    def test_a_stem_is_checked_again_where_it_becomes_a_file_name(self, model_dir, voices):
        engine = BoundEngine()
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        with pytest.raises(worker.Refusal):
            engine.preset("../en-Carter_man")

    @pytest.mark.parametrize("content", ["missing-keys", "refused-unpickle"])
    def test_a_file_that_is_not_a_preset_is_refused_without_its_reason(self, model_dir, voices,
                                                                       content):
        torch = FakeTorch()
        path = os.path.join(voices, "en-Carter_man.pt")
        torch.files[path] = preset_file(keys=("lm", "tts_lm")) if content == "missing-keys" \
            else RuntimeError(f"Weights only load failed for {path}: GLOBAL os.system")
        engine = BoundEngine(torch)
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        with pytest.raises(worker.Refusal) as raised:
            engine.render(parsed(), threading.Event())
        assert str(raised.value) == "that preset voice file is not one the Realtime model reads"

    def test_a_preset_in_another_type_is_cast_to_the_models(self, model_dir, voices):
        torch = FakeTorch()
        torch.files[os.path.join(voices, "en-Carter_man.pt")] = preset_file("float32")
        engine = BoundEngine(torch)
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        found = engine.preset("en-Carter_man")
        for key in worker.PRESET_KEYS:
            assert found[key].last_hidden_state.dtype == "bfloat16"
            assert {one.dtype for one in found[key].past_key_values.key_cache} == {"bfloat16"}
            assert {one.dtype for one in found[key].past_key_values.value_cache} == {"bfloat16"}


class TestTheRealtimeModelRendersAsTheWebDemoCallsIt:
    def test_the_processor_call_and_the_generate_call(self, model_dir, voices):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        request = parsed(script=[{"speaker": 1, "text": "“Hello,” she said."}], seed=3)
        reply, audio = engine.render(request, threading.Event())
        processor = FakeRealtimeProcessor.made[0]
        cached = engine.preset("en-Carter_man")
        assert processor.calls == [{"text": "\"Hello,\" she said.", "cached_prompt": cached,
                                    "padding": True, "return_tensors": "pt",
                                    "return_attention_mask": True}]
        generated = FakeRealtimeModel.made[0].generate_calls[0]
        assert generated["max_new_tokens"] is None, "the model's own budget unless given"
        assert generated["cfg_scale"] == 1.5, "the Realtime model's published guidance"
        assert generated["tokenizer"] is processor.tokenizer
        assert generated["generation_config"] == {"do_sample": False}
        assert generated["verbose"] is False and generated["show_progress_bar"] is False
        assert generated["refresh_negative"] is True
        assert callable(generated["stop_check_fn"])
        assert generated["all_prefilled_outputs"] is not cached, "a copy: generate grows it"
        assert set(generated["all_prefilled_outputs"]) == set(worker.PRESET_KEYS)
        assert generated["tts_text_ids"].device == "cuda"
        assert generated["tts_lm_input_ids"].device == "cuda"
        assert isinstance(generated["audio_streamer"], worker.Progress)
        assert torch.seeds == [3]
        assert reply["seconds"] == pytest.approx(5 * 3200 / RATE) and reply["tokens"] == 5
        assert audio[:4] == b"RIFF" and len(audio) == 44 + 5 * 3200 * 2

    def test_a_budget_and_a_guidance_given_are_passed_as_given(self, model_dir, voices):
        engine = BoundEngine()
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        engine.render(parsed(max_new_tokens=77, cfg_scale=2.0), threading.Event())
        generated = FakeRealtimeModel.made[0].generate_calls[0]
        assert generated["max_new_tokens"] == 77 and generated["cfg_scale"] == 2.0

    def test_the_audio_after_the_end_of_speech_is_not_kept(self, model_dir, voices):
        engine = BoundEngine()
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        FakeRealtimeModel.made[0].stop_after = 3
        reply, audio = engine.render(parsed(), threading.Event())
        assert reply["tokens"] == 3 and reply["seconds"] == pytest.approx(3 * 3200 / RATE)
        assert len(audio) == 44 + 3 * 3200 * 2

    def test_the_realtime_model_refuses_a_recording_and_a_second_voice(self, model_dir,
                                                                       voices):
        engine = BoundEngine()
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        with pytest.raises(worker.Refusal) as raised:
            engine.render(request_for(), threading.Event())
        assert "cannot clone a recording" in str(raised.value)
        two = parsed(script=[{"speaker": 1, "text": "One."}, {"speaker": 2, "text": "Two."}],
                     voices=[{"speaker": 1, "preset": "en-Carter_man"},
                             {"speaker": 2, "preset": "en-Emma_woman"}])
        with pytest.raises(worker.Refusal) as raised:
            engine.render(two, threading.Event())
        assert str(raised.value) == "the Realtime model speaks with one voice at a time"
        assert not FakeRealtimeModel.made[0].generate_calls

    def test_the_7b_refuses_a_preset(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        with pytest.raises(worker.Refusal) as raised:
            engine.render(parsed(), threading.Event())
        assert "preset voices belong to the Realtime model" in str(raised.value)

    def test_the_7b_keeps_its_own_guidance_when_none_is_given(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        engine.render(request_for(cfg_scale=None), threading.Event())
        assert FakeModel.made[0].generate_calls[0]["cfg_scale"] == 1.3


def stream_request(**values):
    values.setdefault("stream", True)
    return request_for(**values)


class TestAStreamedRenderStreamsWhileItGenerates:
    def test_frames_arrive_while_generate_is_still_running_on_its_own_thread(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        second_frame = threading.Event()
        waited = []

        def gate(step):
            if step == 3:
                waited.append(second_frame.wait(5.0))

        FakeModel.made[0].gate = gate
        frames = []

        def on_audio(seq, pcm_bytes, samples):
            frames.append((seq, pcm_bytes, samples, threading.current_thread().name))
            if seq == 2:
                second_frame.set()

        reply, body = engine.render(stream_request(), threading.Event(), None, on_audio)
        assert waited == [True], "frames 1 and 2 were delivered while generate was at step 3"
        assert FakeModel.made[0].generate_threads == ["vibevoice-generate"]
        assert {name for _seq, _pcm, _samples, name in frames} == \
            {threading.current_thread().name}, "consumed on the calling (lane) thread"
        assert [seq for seq, _pcm, _samples, _name in frames] == [1, 2, 3, 4, 5, 6]
        assert all(len(one) == samples * 2 for _seq, one, samples, _name in frames)
        assert body == b""
        assert reply["streamed"] is True and reply["frames"] == 6
        assert reply["first_audio_ms"] >= 1
        assert reply["seconds"] == pytest.approx(6 * 3200 / RATE)
        streamer = FakeAudioStreamer.made[0]
        assert (streamer.batch_size, streamer.stop_signal, streamer.timeout) == (1, None, None)
        assert FakeModel.made[0].generate_calls[0]["audio_streamer"] is streamer

    def test_a_streamed_render_is_the_same_samples_as_a_whole_one(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        _reply, whole = engine.render(request_for(), threading.Event())
        frames = []
        engine.render(stream_request(), threading.Event(), None,
                      lambda seq, pcm_bytes, samples: frames.append(pcm_bytes))
        assert b"".join(frames) == whole[44:]

    def test_a_send_that_fails_stops_the_generation_and_is_raised(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        FakeModel.made[0].steps = 50
        cancelled = threading.Event()

        def gate(step):
            # Step 2 waits for whatever the failed send did about it; without
            # the cancel it would carry on to step 50 after five seconds.
            if step == 2:
                cancelled.wait(5.0)

        FakeModel.made[0].gate = gate

        def on_audio(seq, pcm_bytes, samples):
            if seq == 2:
                raise BrokenPipeError("the parent went away")

        began = time.monotonic()
        with pytest.raises(BrokenPipeError):
            engine.render(stream_request(), cancelled, None, on_audio)
        assert cancelled.is_set()
        assert time.monotonic() - began < 4.0
        assert not [thread for thread in threading.enumerate()
                    if thread.name == "vibevoice-generate"], "the generation thread ended"
        assert len(FakeAudioStreamer.made[0].put_threads) == 2, "it stopped at its next step"

    def test_an_error_in_generate_is_raised_once_the_stream_has_ended(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)

        def gate(step):
            if step == 2:
                raise RuntimeError("CUDA error at /home/someone/model")

        FakeModel.made[0].gate = gate
        frames = []
        with pytest.raises(RuntimeError):
            engine.render(stream_request(), threading.Event(), None,
                          lambda seq, pcm_bytes, samples: frames.append(seq))
        assert frames == [1, 2]

    def test_a_cancel_mid_stream_ends_it_with_the_audio_so_far(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        cancelled = threading.Event()
        second = threading.Event()

        def gate(step):
            # The cancel arrives once the listener has had two frames.
            if step == 2:
                second.wait(5.0)
                cancelled.set()

        FakeModel.made[0].gate = gate
        frames = []

        def on_audio(seq, pcm_bytes, samples):
            frames.append(seq)
            if seq == 2:
                second.set()

        reply, body = engine.render(stream_request(), cancelled, None, on_audio)
        assert reply["cancelled"] is True and reply["frames"] == 2 and frames == [1, 2]
        assert body == b""

    def test_a_streamed_realtime_render_keeps_only_what_was_spoken(self, model_dir, voices):
        engine = BoundEngine()
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        FakeRealtimeModel.made[0].stop_after = 3
        frames = []
        reply, body = engine.render(parsed(stream=True), threading.Event(), None,
                                    lambda seq, pcm_bytes, samples: frames.append(samples))
        assert frames == [3200, 3200, 3200]
        assert reply["tokens"] == 3 and reply["streamed"] is True and body == b""

    def test_a_streamed_render_with_nowhere_to_send_is_refused(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        with pytest.raises(worker.Refusal):
            engine.render(stream_request(), threading.Event())
        assert not FakeModel.made[0].generate_calls

    def test_nothing_is_sent_after_a_cancel_though_the_model_runs_on(self, model_dir, voices):
        """The Realtime model reads its stop flag once per six-frame window, as
        upstream's does; what it makes after the cancel is never sent."""
        engine = BoundEngine()
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        model = FakeRealtimeModel.made[0]
        model.chunks, model.window = 12, 6
        cancelled = threading.Event()
        # Three frames are made before step 3 waits for the cancel; the model
        # then runs on to its window's end at step 6 before it looks.
        model.gate = lambda step: cancelled.wait(5.0) if step == 3 else None
        frames = []

        def on_audio(seq, pcm_bytes, samples):
            frames.append(seq)
            if seq == 2:
                cancelled.set()

        reply, _body = engine.render(parsed(stream=True), cancelled, None, on_audio)
        assert frames == [1, 2] and reply["frames"] == 2 and reply["cancelled"] is True
        assert reply["seconds"] == pytest.approx(2 * 3200 / RATE), "only what was sent"
        assert len(FakeAudioStreamer.made[0].put_threads) == 6, "the model made six first"


class TestTheSevenBsBudgetIsNoticedWhenUpstreamDoesNot:
    """Upstream's 7B ``generate`` never raises ``reach_max_step_sample`` when the
    budget runs out -- its loop ends one step before the check -- which the
    smoke tool found against a real model. Using the whole budget without
    ending on end-of-speech is what reaching it means."""

    def test_a_render_that_used_its_whole_budget_is_capped(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        reply, _audio = engine.render(request_for(max_new_tokens=6), threading.Event())
        assert reply["tokens"] == 6 and reply["capped"] is True

    def test_one_that_ended_on_end_of_speech_at_the_boundary_is_not(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        FakeProcessor.made[0].tokenizer.eos_token_id = 0
        reply, _audio = engine.render(request_for(max_new_tokens=6), threading.Event())
        assert reply["capped"] is False

    def test_one_with_budget_to_spare_or_cancelled_is_not(self, model_dir):
        engine = BoundEngine()
        engine.load(model_dir, 10)
        reply, _audio = engine.render(request_for(max_new_tokens=7), threading.Event())
        assert reply["capped"] is False
        cancelled = threading.Event()

        def gate(step):
            if step == 5:
                cancelled.set()

        FakeModel.made[0].gate = gate
        reply, _audio = engine.render(request_for(max_new_tokens=5), cancelled)
        assert reply["capped"] is False and reply["cancelled"] is True

    def test_the_realtime_model_keeps_upstreams_own_flag(self, model_dir, voices):
        engine = BoundEngine()
        engine.load(model_dir, 0, kind="realtime", voices_dir=voices)
        reply, _audio = engine.render(parsed(max_new_tokens=5), threading.Event())
        assert reply["tokens"] == 5 and reply["capped"] is False

    def test_the_last_token_is_read_from_the_first_sequence(self):
        assert worker._last_token(FakeTensor([4, 5, 9], (1, 3))) == 9
        assert worker._last_token(None) is None


# --------------------------------------------------------------------------- #
# The command loop, against a fake engine
# --------------------------------------------------------------------------- #


class FakeStream:
    """A pipe the test can feed frames into and read frames out of; reading blocks."""

    def __init__(self):
        self._buffer = bytearray()
        self._closed = False
        self._ready = threading.Condition()

    def feed(self, header, payload=b""):
        out = io.BytesIO()
        worker.write_frame(out, header, payload)
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
                found = worker.read_frame(io.BytesIO(raw))
            except Exception:
                return
            if found is None:
                return
            header, payload = found
            used = 8 + len(json.dumps(header).encode("utf-8")) + len(payload)
            del self._buffer[:used]
            self.frames.append((header, payload))

    def of(self, request_id):
        with self._lock:
            return [(header, payload) for header, payload in self.frames
                    if header.get("id") == request_id]

    def reply(self, request_id, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for header, payload in self.of(request_id):
                if "ok" in header:
                    return header, payload
            time.sleep(0.01)
        raise AssertionError(f"no reply to {request_id}; frames: "
                             f"{[header for header, _payload in self.frames]}")

    def progress(self, request_id):
        return [header["progress"] for header, _payload in self.of(request_id)
                if "progress" in header]

    def audio(self, request_id):
        return [(header["audio"], payload) for header, payload in self.of(request_id)
                if "audio" in header]


class FakeEngine:
    """The engine's shape without Torch, with a render that can be gated.

    ``load`` takes the phase-one four positionally and the identity as
    keywords, exactly as ``_do_load`` hands them over; ``loads`` keeps the
    phase-one view and ``identities`` the rest.
    """

    def __init__(self, uuid=UUID, cuda=True, chunks=5, gate=None, fail=None):
        self.uuid = uuid
        self.cuda = cuda
        self.chunks = chunks
        self.gate = gate
        self.fail = fail
        self.model = None
        self.model_dir = ""
        self.kind = ""
        self.precision = ""
        self.lora_dir = ""
        self.loads = []
        self.identities = []
        self.unloads = 0
        self.renders = []

    def probe(self):
        if not self.cuda:
            raise worker.Refusal("no CUDA device is visible to the worker")
        return {"torch": "2.8.0+cu128", "cuda": True, "device_name": "Fake GPU",
                "device_uuid": self.uuid, "device_index": 0,
                "total_vram_bytes": 24_000_000_000, "free_vram_bytes": 23_000_000_000}

    def resident(self):
        return 18_000_000_000 if self.model is not None else 0

    def peak(self):
        return 20_000_000_000

    def load(self, model_dir, steps, dtype, attention, **identity):
        self.loads.append((model_dir, steps, dtype, attention))
        self.identities.append(dict(identity))
        self.model = object()
        self.model_dir = model_dir
        self.kind = identity.get("kind", "longform")
        self.precision = identity.get("precision") or dtype
        self.lora_dir = identity.get("lora_dir") or ""
        return {"resident_bytes": 18_000_000_000, "weights_bytes": 17_000_000_000,
                "load_seconds": 0.01}

    def unload(self):
        self.unloads += 1
        self.model = None
        self.model_dir = ""
        return {"resident_bytes": 0}

    def render(self, request, cancelled, on_progress, on_audio=None):
        if self.fail is not None:
            raise self.fail
        self.renders.append(request)
        made = 0
        frames = 0
        for index in range(self.chunks):
            if self.gate is not None:
                self.gate(index)
            if cancelled.is_set():
                break
            made += 3200
            if request.stream:
                frames += 1
                on_audio(frames, struct.pack("<h", index + 1) * 3200, 3200)
            else:
                on_progress({"seconds": made / RATE})
        reply = {"seconds": made / RATE, "sample_rate": RATE, "render_seconds": 0.01,
                 "peak_bytes": 20_000_000_000, "resident_bytes": 18_000_000_000,
                 "cancelled": cancelled.is_set(), "tokens": made // 3200, "capped": False}
        if request.stream:
            reply.update({"streamed": True, "first_audio_ms": 7, "frames": frames})
            return reply, b""
        return reply, worker.encode_wav(b"\x00\x00" * made, RATE)


def run_worker(engine, feed, init=True, stdout=None):
    """Drive the real ``serve`` loop on a thread and return what it wrote."""
    stdin = FakeStream()
    stdout = stdout if stdout is not None else Collector()
    thread = threading.Thread(target=worker.serve, args=(stdin, stdout),
                              kwargs={"engine_factory": lambda: engine}, daemon=True)
    thread.start()
    if init:
        stdin.feed({"op": "init", "id": "i1", "expect_uuid": "0123abcd", "parent_pid": 0})
        stdout.reply("i1")
    try:
        feed(stdin, stdout)
    finally:
        stdin.feed({"op": "shutdown", "id": "z"})
        thread.join(timeout=5.0)
        stdin.close()
    stdout.exit_thread = thread
    return stdout


def render_header(request_id, job="j1", script=None):
    one = pcm(0.5)
    return ({"op": "render", "id": request_id, "job": job,
             "script": script or [{"speaker": 1, "text": "Hello there."}],
             "voices": [{"speaker": 1, "offset": 0, "count": len(one) // 4, "rate": RATE}],
             "cfg_scale": 1.3, "seed": 1, "max_new_tokens": None, "steps": 10}, one)


class TestTheHandshakeSaysWhatTheParentAsksFor:
    def test_every_declared_field_is_present_and_nothing_private_is(self):
        found = run_worker(FakeEngine(), lambda stdin, out: None)
        header, _payload = found.reply("i1")
        for name in ("protocol", "worker", "python", "torch", "cuda", "device_name",
                     "device_uuid", "device_index", "total_vram_bytes", "free_vram_bytes",
                     "containment", "vibevoice", "transformers", "sample_rate"):
            assert name in header, name
        assert header["ok"] is True
        assert header["protocol"] == 1
        assert header["worker"] == "vibevoice"
        assert header["cuda"] is True
        assert header["device_uuid"] == UUID
        assert header["sample_rate"] == 24000
        assert header["containment"] in ("pdeathsig", "job", "none", "unknown")
        text = json.dumps(header)
        for forbidden in ("/home", "\\\\Users", "python.exe", "pid", "token", "model_dir"):
            assert forbidden not in text

    def test_no_cuda_device_is_a_refusal_at_init(self):
        def feed(stdin, out):
            header, _payload = out.reply("i1")
            assert header["ok"] is False
            assert header["error"] == "no CUDA device is visible to the worker"
            assert header["refusal"] is True
            stdin.feed({"op": "status", "id": "s1"})
            header, _payload = out.reply("s1")
            assert header["ok"] is False and "not initialised" in header["error"]

        stdin = FakeStream()
        stdout = Collector()
        thread = threading.Thread(target=worker.serve, args=(stdin, stdout),
                                  kwargs={"engine_factory": lambda: FakeEngine(cuda=False)},
                                  daemon=True)
        thread.start()
        stdin.feed({"op": "init", "id": "i1", "parent_pid": 0})
        try:
            feed(stdin, stdout)
        finally:
            stdin.feed({"op": "shutdown", "id": "z"})
            thread.join(timeout=5.0)

    def test_a_request_before_init_is_refused(self):
        stdin = FakeStream()
        stdout = Collector()
        thread = threading.Thread(target=worker.serve, args=(stdin, stdout),
                                  kwargs={"engine_factory": lambda: FakeEngine()}, daemon=True)
        thread.start()
        stdin.feed({"op": "load", "id": "l1", "model_dir": "x"})
        header, _payload = stdout.reply("l1")
        assert header["ok"] is False and header["error"] == "runtime is not initialised"
        stdin.feed({"op": "shutdown", "id": "z"})
        thread.join(timeout=5.0)


class TestTheLoopAnswersEveryOperation:
    def test_status_load_render_unload_in_order(self, tmp_path):
        engine = FakeEngine()

        def feed(stdin, out):
            stdin.feed({"op": "status", "id": "s1"})
            header, _payload = out.reply("s1")
            assert header["ok"] is True and header["loaded"] is False
            assert header["rendering"] is False and header["resident_bytes"] == 0
            assert set(header) >= {"loaded", "rendering", "job", "resident_bytes",
                                   "peak_bytes", "model_dir"}

            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10,
                        "dtype": "bf16", "attention": "sdpa"})
            header, _payload = out.reply("l1")
            assert header["ok"] is True and header["resident_bytes"] == 18_000_000_000
            assert header["weights_bytes"] == 17_000_000_000
            assert engine.loads == [(str(tmp_path), 10, "bf16", "sdpa")]

            stdin.feed({"op": "status", "id": "s2"})
            header, _payload = out.reply("s2")
            assert header["loaded"] is True and header["resident_bytes"] == 18_000_000_000
            assert header["model_dir"] == str(tmp_path)

            head, payload = render_header("r1")
            stdin.feed(head, payload)
            header, audio = out.reply("r1")
            assert header["ok"] is True
            assert header["seconds"] == pytest.approx(5 * 3200 / RATE)
            assert header["cancelled"] is False and header["tokens"] == 5
            assert audio[:4] == b"RIFF"
            assert out.progress("r1"), "progress frames carry the render's id"
            assert all("seconds" in one for one in out.progress("r1"))
            first = out.of("r1")[0][0]
            assert "progress" in first and "ok" not in first, "progress comes before the reply"

            stdin.feed({"op": "unload", "id": "u1"})
            header, _payload = out.reply("u1")
            assert header["ok"] is True and header["resident_bytes"] == 0
            assert engine.unloads == 1

        run_worker(engine, feed)

    def test_an_unknown_operation_is_answered(self):
        def feed(stdin, out):
            stdin.feed({"op": "dance", "id": "d1"})
            header, _payload = out.reply("d1")
            assert header["ok"] is False and header["error"] == "unknown operation"

        run_worker(FakeEngine(), feed)

    def test_shutdown_is_answered_and_the_loop_returns_zero(self):
        stdin = FakeStream()
        stdout = Collector()
        result = []
        thread = threading.Thread(
            target=lambda: result.append(worker.serve(stdin, stdout,
                                                      engine_factory=lambda: FakeEngine())),
            daemon=True)
        thread.start()
        stdin.feed({"op": "init", "id": "i1", "parent_pid": 0})
        stdout.reply("i1")
        stdin.feed({"op": "shutdown", "id": "z"})
        header, _payload = stdout.reply("z")
        assert header["ok"] is True
        thread.join(timeout=5.0)
        assert result == [0]

    def test_end_of_input_ends_the_loop_with_zero(self):
        stdin = FakeStream()
        stdout = Collector()
        result = []
        thread = threading.Thread(
            target=lambda: result.append(worker.serve(stdin, stdout,
                                                      engine_factory=lambda: FakeEngine())),
            daemon=True)
        thread.start()
        stdin.close()
        thread.join(timeout=5.0)
        assert result == [0]


class TestOneRenderAtATime:
    def test_a_second_render_is_refused_while_the_first_runs(self, tmp_path):
        entered = threading.Event()
        release = threading.Event()

        def gate(index):
            if index == 1:
                entered.set()
                release.wait(5.0)

        engine = FakeEngine(gate=gate)

        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10})
            out.reply("l1")
            head, payload = render_header("r1", job="one")
            stdin.feed(head, payload)
            assert entered.wait(5.0)
            head, payload = render_header("r2", job="two")
            stdin.feed(head, payload)
            header, _payload = out.reply("r2")
            assert header["ok"] is False and header["error"] == "one render at a time"
            assert header["refusal"] is True
            stdin.feed({"op": "status", "id": "s1"})
            header, _payload = out.reply("s1")
            assert header["rendering"] is True and header["job"] == "one"
            release.set()
            header, _payload = out.reply("r1")
            assert header["ok"] is True
            head, payload = render_header("r3", job="three")
            stdin.feed(head, payload)
            header, _payload = out.reply("r3")
            assert header["ok"] is True, "the slot is free once the first has answered"

        run_worker(engine, feed)

    def test_cancel_lands_while_a_render_is_in_flight(self, tmp_path):
        entered = threading.Event()
        release = threading.Event()

        def gate(index):
            if index == 2:
                entered.set()
                release.wait(5.0)

        engine = FakeEngine(gate=gate)

        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10})
            out.reply("l1")
            head, payload = render_header("r1", job="one")
            stdin.feed(head, payload)
            assert entered.wait(5.0)
            stdin.feed({"op": "cancel", "id": "c0", "job": "other"})
            header, _payload = out.reply("c0")
            assert header["cancelled"] is False, "a cancel names its render"
            stdin.feed({"op": "cancel", "id": "c1", "job": "one"})
            header, _payload = out.reply("c1")
            assert header["ok"] is True and header["cancelled"] is True
            release.set()
            header, audio = out.reply("r1")
            assert header["ok"] is True and header["cancelled"] is True
            assert header["tokens"] == 2 and len(audio) == 44 + 2 * 3200 * 2
            stdin.feed({"op": "cancel", "id": "c2", "job": "one"})
            header, _payload = out.reply("c2")
            assert header["cancelled"] is False, "nothing is rendering any more"

        run_worker(engine, feed)

    def test_a_failing_render_is_answered_with_its_class_only_and_frees_the_slot(self, tmp_path):
        engine = FakeEngine(fail=RuntimeError("CUDA error at /home/someone/model/shard-1"))

        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10})
            out.reply("l1")
            head, payload = render_header("r1")
            stdin.feed(head, payload)
            header, _payload = out.reply("r1")
            assert header["ok"] is False
            assert header["error"] == "RuntimeError" and header["refusal"] is False
            engine.fail = None
            head, payload = render_header("r2")
            stdin.feed(head, payload)
            header, _payload = out.reply("r2")
            assert header["ok"] is True

        run_worker(engine, feed)

    def test_a_malformed_render_is_refused_at_once_with_a_sentence(self, tmp_path):
        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10})
            out.reply("l1")
            head, payload = render_header("r1", script=[{"speaker": 1, "text": " "}])
            stdin.feed(head, payload)
            header, _payload = out.reply("r1")
            assert header["ok"] is False and header["error"] == "the script has no words"
            assert header["refusal"] is True
            stdin.feed({"op": "status", "id": "s1"})
            header, _payload = out.reply("s1")
            assert header["rendering"] is False

        run_worker(FakeEngine(), feed)

    def test_shutdown_during_a_render_sets_the_cancel_flag(self, tmp_path):
        entered = threading.Event()
        seen = []

        def gate(index):
            if index == 1:
                entered.set()
                time.sleep(0.2)
            seen.append(index)

        engine = FakeEngine(gate=gate, chunks=50)
        stdin = FakeStream()
        stdout = Collector()
        thread = threading.Thread(target=worker.serve, args=(stdin, stdout),
                                  kwargs={"engine_factory": lambda: engine}, daemon=True)
        thread.start()
        stdin.feed({"op": "init", "id": "i1", "parent_pid": 0})
        stdout.reply("i1")
        stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10})
        stdout.reply("l1")
        head, payload = render_header("r1")
        stdin.feed(head, payload)
        assert entered.wait(5.0)
        stdin.feed({"op": "shutdown", "id": "z"})
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert len(seen) < 50, "the render stopped at the step after the shutdown"


class TestTheRenderSlotIsFreeBeforeTheReplyIsWritten:
    def test_a_parent_that_sends_its_next_render_on_the_reply_is_never_refused(
            self, tmp_path, monkeypatch):
        """Looked at from inside the write of the reply itself: by then the
        slot must be free, or a parent that answers the reply with its next
        section's render -- the Voice Box does -- races the lane and loses."""
        made = []

        class Recorded(worker.Worker):
            def __init__(self, stdout):
                super().__init__(stdout)
                made.append(self)

        monkeypatch.setattr(worker, "Worker", Recorded)
        seen = {}

        class Watching(Collector):
            def _drain(self):
                before = len(self.frames)
                super()._drain()
                for header, _payload in self.frames[before:]:
                    if "ok" in header and header.get("id") in ("r1", "r2"):
                        seen[header["id"]] = made[0].rendering()

        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10})
            out.reply("l1")
            head, payload = render_header("r1", job="first")
            stdin.feed(head, payload)
            out.reply("r1")
            head, payload = render_header("r2", job="second")
            head["stream"] = True
            stdin.feed(head, payload)
            out.reply("r2")

        run_worker(FakeEngine(), feed, stdout=Watching())
        assert seen == {"r1": "", "r2": ""}, "whole and streamed alike"


class TestALoadCarriesItsIdentityAndStatusSaysWhatIsLoaded:
    def test_a_phase_one_load_is_the_long_form_model_on_the_card(self, tmp_path):
        engine = FakeEngine()

        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10,
                        "dtype": "bf16", "attention": "sdpa"})
            assert out.reply("l1")[0]["ok"] is True

        run_worker(engine, feed)
        assert engine.identities == [{"kind": "longform", "precision": "", "lora_dir": None,
                                      "lora_scale": 1.0, "voices_dir": "", "device": "cuda"}]

    def test_every_identity_field_reaches_the_engine(self, tmp_path):
        engine = FakeEngine()

        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 0,
                        "kind": "realtime", "precision": "bf16", "lora_dir": None,
                        "lora_scale": 0.7, "voices_dir": str(tmp_path / "voices"),
                        "device": "cpu"})
            assert out.reply("l1")[0]["ok"] is True
            stdin.feed({"op": "load", "id": "l2", "model_dir": str(tmp_path), "steps": 10,
                        "precision": "nf4", "lora_dir": str(tmp_path / "lora"),
                        "lora_scale": 1.5})
            assert out.reply("l2")[0]["ok"] is True

        run_worker(engine, feed)
        assert engine.identities == [
            {"kind": "realtime", "precision": "bf16", "lora_dir": None, "lora_scale": 0.7,
             "voices_dir": str(tmp_path / "voices"), "device": "cpu"},
            {"kind": "longform", "precision": "nf4", "lora_dir": str(tmp_path / "lora"),
             "lora_scale": 1.5, "voices_dir": "", "device": "cuda"}]

    @pytest.mark.parametrize("field, value, sentence", [
        ("lora_dir", 7, "the LoRA folder is not a path"),
        ("voices_dir", ["a"], "the voices folder is not a path"),
        ("lora_scale", "strong", "a number in the request is not a number"),
    ])
    def test_a_malformed_identity_is_refused_with_a_sentence(self, tmp_path, field, value,
                                                             sentence):
        engine = FakeEngine()

        def feed(stdin, out):
            header = {"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10}
            header[field] = value
            stdin.feed(header)
            found, _payload = out.reply("l1")
            assert found["ok"] is False and found["refusal"] is True
            assert found["error"] == sentence

        run_worker(engine, feed)
        assert engine.loads == []

    def test_status_says_the_kind_the_precision_and_whether_a_lora_is_joined(self, tmp_path):
        def feed(stdin, out):
            stdin.feed({"op": "status", "id": "s1"})
            header, _payload = out.reply("s1")
            assert (header["kind"], header["precision"], header["lora"]) == ("", "", False)
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10,
                        "precision": "int8", "lora_dir": str(tmp_path)})
            out.reply("l1")
            stdin.feed({"op": "status", "id": "s2"})
            header, _payload = out.reply("s2")
            assert (header["kind"], header["precision"], header["lora"]) == \
                ("longform", "int8", True)
            stdin.feed({"op": "unload", "id": "u1"})
            out.reply("u1")
            stdin.feed({"op": "status", "id": "s3"})
            header, _payload = out.reply("s3")
            assert (header["kind"], header["precision"], header["lora"]) == ("", "", False)

        run_worker(FakeEngine(), feed)


class TestAStreamedRenderOnTheWire:
    def test_audio_frames_carry_the_request_id_then_the_reply_carries_no_wav(self, tmp_path):
        engine = FakeEngine(chunks=4)

        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10})
            out.reply("l1")
            head, payload = render_header("r1")
            head["stream"] = True
            stdin.feed(head, payload)
            header, body = out.reply("r1")
            frames = out.audio("r1")
            assert [info["seq"] for info, _pcm in frames] == [1, 2, 3, 4]
            assert all(info == {"seq": info["seq"], "samples": 3200, "rate": RATE}
                       for info, _pcm in frames)
            assert [pcm[:2] for _info, pcm in frames] == [struct.pack("<h", n)
                                                          for n in (1, 2, 3, 4)]
            assert all(len(pcm) == 3200 * 2 for _info, pcm in frames)
            ordered = [head_ for head_, _payload in out.of("r1")]
            assert "ok" in ordered[-1] and all("audio" in one for one in ordered[:-1]), \
                "every frame before the reply that ends the render"
            assert header["ok"] is True and header["streamed"] is True
            assert header["first_audio_ms"] == 7 and header["frames"] == 4
            assert body == b""
            assert not out.progress("r1"), "a streamed render sends audio, not progress"
            assert engine.renders[0].stream is True

        run_worker(engine, feed)

    def test_a_render_that_does_not_ask_to_stream_still_answers_with_a_wav(self, tmp_path):
        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10})
            out.reply("l1")
            head, payload = render_header("r1")
            stdin.feed(head, payload)
            header, body = out.reply("r1")
            assert body[:4] == b"RIFF" and "streamed" not in header
            assert out.audio("r1") == []

        run_worker(FakeEngine(), feed)

    def test_a_streamed_render_can_be_cancelled_between_frames(self, tmp_path):
        entered = threading.Event()
        release = threading.Event()

        def gate(index):
            if index == 2:
                entered.set()
                release.wait(5.0)

        engine = FakeEngine(gate=gate, chunks=10)

        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 10})
            out.reply("l1")
            head, payload = render_header("r1", job="spoken")
            head["stream"] = True
            stdin.feed(head, payload)
            assert entered.wait(5.0)
            stdin.feed({"op": "cancel", "id": "c1", "job": "spoken"})
            assert out.reply("c1")[0]["cancelled"] is True
            release.set()
            header, body = out.reply("r1")
            assert header["cancelled"] is True and header["frames"] == 2
            assert len(out.audio("r1")) == 2 and body == b""

        run_worker(engine, feed)

    def test_a_preset_render_reaches_the_engine_as_a_preset(self, tmp_path):
        engine = FakeEngine()

        def feed(stdin, out):
            stdin.feed({"op": "load", "id": "l1", "model_dir": str(tmp_path), "steps": 0,
                        "kind": "realtime"})
            out.reply("l1")
            stdin.feed({"op": "render", "id": "r1", "job": "j",
                        "script": [{"speaker": 1, "text": "Hello there, how are you?"}],
                        "voices": [{"speaker": 1, "preset": "en-Carter_man"}],
                        "cfg_scale": None, "seed": None, "max_new_tokens": None, "steps": 0})
            assert out.reply("r1")[0]["ok"] is True

        run_worker(engine, feed)
        request = engine.renders[0]
        assert request.presets == {1: "en-Carter_man"} and request.voices == []
        assert request.cfg_scale is None and request.plain == "Hello there, how are you?"


class TestRunningTheFileDirectly:
    def test_selftest_dispatches_and_serve_is_the_default(self, monkeypatch):
        calls = []
        claimed = io.BytesIO()
        monkeypatch.setattr(worker, "selftest", lambda: calls.append("selftest") or 3)
        monkeypatch.setattr(worker, "_claim_stdout", lambda: claimed)
        monkeypatch.setattr(worker, "serve",
                            lambda stdin, stdout, engine_factory=None:
                            calls.append(("serve", stdin, stdout, engine_factory)) or 0)
        assert worker.main(["--selftest"]) == 3
        assert worker.main([worker.MARKER, "--parent-pid", "12"]) == 0
        assert calls[0] == "selftest"
        assert calls[1][0] == "serve"
        assert calls[1][1] is sys.stdin.buffer
        assert calls[1][2] is claimed, "the protocol runs on the claimed copy of stdout"
        assert calls[1][3] is None, "the real engine unless one is handed in"
        factory = object()
        assert worker.main([worker.MARKER], engine_factory=factory) == 0
        assert calls[2][3] is factory

    def test_a_stray_print_never_reaches_the_protocol_pipe(self, tmp_path):
        """Upstream prints to stdout at the end of a capped Realtime render. Run
        as a real process, the worker's frames stay readable and every stray
        line -- Python's or a raw write to descriptor 1 -- lands on stderr."""
        import subprocess

        program = tmp_path / "claim.py"
        program.write_text(
            "import os, sys\n"
            f"sys.path.insert(0, {str(pathlib.Path(worker.__file__).parent.parent)!r})\n"
            "from vibevoice_worker import worker\n"
            "pipe = worker._claim_stdout()\n"
            "print('Reached maximum generation length 99, stopped it.')\n"
            "os.write(1, b'a native library wrote this\\n')\n"
            "worker.write_frame(pipe, {'id': 'r1', 'ok': True}, b'audio')\n"
            "print('and once more after the frame')\n"
            "worker.write_frame(pipe, {'id': 'r2', 'ok': True})\n", encoding="utf-8")
        done = subprocess.run([sys.executable, str(program)], capture_output=True, timeout=60)
        assert done.returncode == 0, done.stderr
        stream = io.BytesIO(done.stdout)
        assert worker.read_frame(stream) == ({"id": "r1", "ok": True}, b"audio")
        assert worker.read_frame(stream) == ({"id": "r2", "ok": True}, b"")
        assert worker.read_frame(stream) is None, "nothing else on the protocol pipe"
        for line in (b"Reached maximum generation length", b"a native library wrote this",
                     b"and once more after the frame"):
            assert line in done.stderr

    def test_selftest_reports_a_missing_closure_rather_than_raising(self, monkeypatch):
        """Without Torch on this interpreter the report says why, on one line,
        with a non-zero status -- which is what the installer reads."""
        import builtins

        real_import = builtins.__import__

        def refuse_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no module named torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_torch)
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        status = worker.selftest()
        report = json.loads(out.getvalue().strip().splitlines()[-1])
        assert status == 1
        assert report["ok"] is False and "ImportError" in report["error"]


def fake_torch_module(monkeypatch, cuda=True):
    found = types.ModuleType("torch")
    found.__version__ = "2.8.0+cu128"
    found.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "torch", found)
    return found


class TestTheSelftestProvesBothModelsImport:
    """What the installer's smoke test reads from ``--selftest``: the 7B's
    classes, the Realtime model's from the overlay and upstream's streamer are
    required; PEFT and bitsandbytes are reported, each one feature."""

    def run(self, monkeypatch):
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        status = worker.selftest()
        return status, json.loads(out.getvalue().strip().splitlines()[-1])

    def test_every_class_is_resolved_and_the_features_are_reported(self, monkeypatch):
        fake_torch_module(monkeypatch)
        resolved = []
        monkeypatch.setattr(worker.Engine, "_upstream",
                            lambda self: resolved.append("longform") or (object, object))
        monkeypatch.setattr(worker.Engine, "_upstream_realtime",
                            lambda self: resolved.append("realtime") or (object, object))
        monkeypatch.setattr(worker.Engine, "_streamer_class",
                            lambda self: resolved.append("streamer") or object)
        monkeypatch.setattr(worker.Engine, "_peft", lambda self: (object, object, object))

        def no_bitsandbytes(self):
            raise worker.Refusal("the runtime was installed before quantisation was added — "
                                 "reinstall it")

        monkeypatch.setattr(worker.Engine, "_quantisation_class", no_bitsandbytes)
        status, report = self.run(monkeypatch)
        assert status == 0 and report["ok"] is True
        assert resolved == ["longform", "realtime", "streamer"]
        assert report["realtime"] is True and report["cuda"] is True
        assert report["lora"] is True and report["quantisation"] is False
        assert report["bitsandbytes"] == ""

    def test_a_runtime_without_the_overlay_fails_the_selftest(self, monkeypatch):
        fake_torch_module(monkeypatch)
        monkeypatch.setattr(worker.Engine, "_upstream", lambda self: (object, object))

        def no_overlay(self):
            raise worker.Refusal("the runtime was installed before the Realtime model was "
                                 "added — reinstall it")

        monkeypatch.setattr(worker.Engine, "_upstream_realtime", no_overlay)
        status, report = self.run(monkeypatch)
        assert status == 1 and report["ok"] is False
        assert "Realtime model" in report["error"]
