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

Nothing here imports Torch, transformers or vibevoice.
"""

from __future__ import annotations

import io
import json
import pathlib
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
        assert worker.PROTOCOL_VERSION == 2
        assert worker.WORKER_NAME == "vibevoice"
        assert worker.MARKER == "--model-chain-vibevoice-worker"
        assert worker.MARKER != pocket_worker.MARKER
        assert worker.SAMPLE_RATE == 24000

    def test_the_payload_ceilings_hold_a_request_and_four_forty_five_minute_takes(self):
        assert worker.MAX_PAYLOAD == 256 << 20
        assert worker.MAX_HEADER == 1 << 20
        assert worker.MAX_REPLY_PAYLOAD > worker.MAX_TAKES * 45 * 60 * RATE * 4
        assert worker.MAX_REPLY_PAYLOAD < 1 << 32, "the length prefix is four bytes"

    def test_a_reply_larger_than_a_request_is_read_with_the_reply_ceiling(self):
        head = json.dumps({"id": "r"}).encode("utf-8")
        body = b"\x00" * 64
        raw = worker._LENGTH.pack(len(head)) + head + worker._LENGTH.pack(len(body)) + body
        with pytest.raises(worker.Refusal):
            worker.read_frame(io.BytesIO(raw), max_payload=32)
        assert worker.read_frame(io.BytesIO(raw), max_payload=64) == ({"id": "r"}, body)

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
        for engine in ("torch", "numpy", "transformers", "vibevoice"):
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


class TestAudioLeavesAsTheModelsOwnFloats:
    def test_samples_cross_unclipped_and_unquantised(self):
        """The parent makes the file straight from these, so nothing is lost here."""
        values = [0.0, 1.0, -1.0, 1.25, -1.5, 0.123456789]
        found = numpy.frombuffer(worker.float32_bytes(values), dtype="<f4")
        assert list(found) == [numpy.float32(value) for value in values]

    def test_a_value_that_is_not_a_number_is_silence_and_infinity_is_unity(self):
        found = numpy.frombuffer(worker.float32_bytes([float("nan"), float("inf"),
                                                       -float("inf")]), dtype="<f4")
        assert list(found) == [0.0, 1.0, -1.0]


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
        """``[take, -1]`` or ``[take]``: one element; ``[take, a:]``: a row's slice."""
        found = self.data[index[-1] if isinstance(index, tuple) else index]
        if isinstance(found, list):
            return types.SimpleNamespace(tolist=lambda: list(found))
        return types.SimpleNamespace(item=lambda: found)


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


class FakeGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeTorch:
    __version__ = "2.8.0+cu128"
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"
    Generator = FakeGenerator

    def __init__(self, cuda=None):
        self.cuda = cuda or FakeCuda()
        self.seeds = []
        self.multinomial = self._multinomial

    @staticmethod
    def _multinomial(probabilities, num_samples, replacement=False, *, generator=None):
        return ("drawn", generator)

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    def manual_seed(self, seed):
        self.seeds.append(seed)


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
        rows = len(kwargs.get("text") or [None])
        return {"input_ids": FakeTensor([0] * self.prompt_length, (rows, self.prompt_length)),
                "attention_mask": FakeTensor([1] * self.prompt_length,
                                             (rows, self.prompt_length)),
                "speech_tensors": FakeTensor([0.0] * 10, (rows, 10)),
                "speech_masks": FakeTensor([True], (rows, 1)),
                "speech_input_mask": FakeTensor([False] * self.prompt_length,
                                                (rows, self.prompt_length)),
                "parsed_scripts": [[(0, " hello")]] * rows,
                "all_speakers_list": [[0]] * rows}


class FakeScheduler:
    """The model's own scheduler: a configuration, and ``from_config`` to make another."""

    made = []

    def __init__(self, **config):
        base = {"num_train_timesteps": 1000, "beta_schedule": "cosine",
                "prediction_type": "v_prediction", "algorithm_type": "dpmsolver++",
                "solver_order": 2, "use_karras_sigmas": False, "use_lu_lambdas": False}
        base.update(config)
        self.config = types.SimpleNamespace(**base)

    @classmethod
    def from_config(cls, config, **overrides):
        found = dict(vars(config))
        found.update(overrides)
        made = cls(**found)
        cls.made.append((made, dict(overrides)))
        return made


class FakeAcousticTokenizer:
    def encode(self, audio):
        return types.SimpleNamespace(mean=audio, std=0.5)


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
        self.config = types.SimpleNamespace(
            decoder_config=types.SimpleNamespace(max_position_embeddings=32768))
        self.model = types.SimpleNamespace(
            noise_scheduler=FakeScheduler(), acoustic_tokenizer=FakeAcousticTokenizer(),
            language_model=types.SimpleNamespace(
                config=types.SimpleNamespace(_attn_implementation="sdpa")))
        self.during = []

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

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        self.during.append({
            "hooked": "sample_speech_tokens" in vars(self),
            "encode_hooked": "encode" in vars(self.model.acoustic_tokenizer),
            "solver": self.model.noise_scheduler,
            "attention": self.model.language_model.config._attn_implementation})
        rows = int(kwargs["input_ids"].shape[0])
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
            chunk = [0.5] * 3200
            made.extend(chunk)
            if streamer is not None:
                streamer.put(FakeTensor(chunk, (1, 1, 3200)), FakeTensor([0], (1,)))
        if streamer is not None:
            streamer.end()
        prompt = kwargs["input_ids"].shape[-1]
        return types.SimpleNamespace(
            speech_outputs=[FakeTensor(made, (1, len(made))) if made else None] * rows,
            sequences=FakeTensor([0] * (prompt + len(made) // 3200),
                                 (rows, prompt + len(made) // 3200)),
            reach_max_step_sample=FakeTensor([self.capped] * rows, (rows,)))


class BoundEngine(worker.Engine):
    """The real engine with its two import points answered by stand-ins."""

    def __init__(self, torch=None, model_class=FakeModel, processor_class=FakeProcessor):
        super().__init__()
        self._stand_ins = (torch or FakeTorch(), model_class, processor_class)

    def _frameworks(self):
        self.torch, self.numpy = self._stand_ins[0], numpy
        return self.torch, self.numpy

    def _upstream(self):
        return self._stand_ins[1], self._stand_ins[2]


@pytest.fixture(autouse=True)
def _forget_stand_ins():
    FakeModel.made.clear()
    FakeProcessor.made.clear()
    FakeScheduler.made.clear()
    yield
    FakeModel.made.clear()
    FakeProcessor.made.clear()
    FakeScheduler.made.clear()


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
        assert reply["format"] == "f32le"
        assert reply["takes"] == [{"seed": 7, "samples": 6 * 3200,
                                   "seconds": pytest.approx(6 * 3200 / RATE),
                                   "tokens": 6, "capped": False}]
        assert len(audio) == 6 * 3200 * 4
        assert set(numpy.frombuffer(audio, dtype="<f4")) == {numpy.float32(0.5)}
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
        assert len(audio) == 2 * 3200 * 4

    def test_a_render_that_hit_its_budget_says_so(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        FakeModel.made[0].capped = True
        reply, _audio = engine.render(request_for(), threading.Event())
        assert reply["capped"] is True

    def test_a_sampling_request_samples_with_its_temperature_and_top_p(self, tmp_path):
        """Greedy unless asked: the model's language part only picks control
        tokens, so sampling varies the pacing, and the seed still fixes it."""
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(str(tmp_path), 10)
        engine.render(request_for(sampling=True, temperature=0.8, top_p=0.9, seed=5),
                      threading.Event())
        engine.render(request_for(sampling=True), threading.Event())
        first, second = FakeModel.made[0].generate_calls
        assert first["generation_config"] == {"do_sample": True, "temperature": 0.8,
                                              "top_p": 0.9}
        assert second["generation_config"] == {
            "do_sample": True, "temperature": worker.SAMPLING_TEMPERATURE,
            "top_p": worker.SAMPLING_TOP_P}, "sampling without values takes the defaults"
        assert torch.seeds == [5], "a sampled render is seeded like any other"

    @pytest.mark.parametrize("asked", [None, False, "true", 1, "yes"])
    def test_anything_but_true_is_the_models_own_greedy_choice(self, tmp_path, asked):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        engine.render(request_for(sampling=asked, temperature=0.8, top_p=0.9),
                      threading.Event())
        assert FakeModel.made[0].generate_calls[0]["generation_config"] == {"do_sample": False}

    @pytest.mark.parametrize("values, sentence", [
        ({"temperature": 0}, "the temperature must be above 0 and at most 2"),
        ({"temperature": 2.5}, "the temperature must be above 0 and at most 2"),
        ({"top_p": 0}, "top-p must be above 0 and at most 1"),
        ({"top_p": 1.5}, "top-p must be above 0 and at most 1"),
        ({"temperature": "warm"}, "a number in the request is not a number"),
    ])
    def test_sampling_values_out_of_range_are_refused_in_a_sentence(self, values, sentence):
        with pytest.raises(worker.Refusal) as raised:
            request_for(sampling=True, **values)
        assert str(raised.value) == sentence

    def test_a_render_that_used_its_whole_budget_is_capped(self, tmp_path):
        """Upstream's ``generate`` never raises ``reach_max_step_sample`` when
        the budget runs out -- its loop's range ends one step before the check
        -- so using the whole budget without ending on end-of-speech is what
        reaching it means."""
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        reply, _audio = engine.render(request_for(max_new_tokens=6), threading.Event())
        assert reply["tokens"] == 6 and reply["capped"] is True

    def test_one_that_ended_on_end_of_speech_at_the_boundary_is_not(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        FakeProcessor.made[0].tokenizer.eos_token_id = 0
        reply, _audio = engine.render(request_for(max_new_tokens=6), threading.Event())
        assert reply["tokens"] == 6 and reply["capped"] is False

    def test_one_with_budget_to_spare_or_cancelled_is_not(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        reply, _audio = engine.render(request_for(max_new_tokens=7), threading.Event())
        assert reply["capped"] is False
        cancelled = threading.Event()

        def gate(step):
            if step == 5:
                cancelled.set()

        FakeModel.made[0].gate = gate
        reply, _audio = engine.render(request_for(max_new_tokens=5), cancelled)
        assert reply["capped"] is False and reply["cancelled"] is True

    def test_the_last_token_is_read_from_the_first_sequence(self):
        assert worker._last_token(FakeTensor([4, 5, 9], (1, 3))) == 9
        assert worker._last_token(None) is None

    def test_no_audio_at_all_is_an_empty_take_rather_than_an_error(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        FakeModel.made[0].steps = 0
        reply, audio = engine.render(request_for(), threading.Event())
        assert reply["seconds"] == 0.0
        assert reply["takes"][0]["samples"] == 0
        assert audio == b""


class TestTheSolverTheAttentionAndTheTakes:
    def test_the_solvers_are_vibevoices_own_scheduler_configured(self):
        """Every solver is the model's DPMSolverMultistepScheduler with its
        algorithm and order changed: no deprecated algorithm, and no third-order
        SDE, whose update takes no noise."""
        assert list(worker.SOLVERS) == ["dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m", "dpmpp_1m",
                                        "dpmpp_1m_sde"]
        for entry in worker.SOLVERS.values():
            assert entry["algorithm_type"] in ("dpmsolver++", "sde-dpmsolver++")
            assert entry["solver_order"] in (1, 2, 3)
            assert not (entry["algorithm_type"].startswith("sde")
                        and entry["solver_order"] == 3)
        assert worker.SOLVERS[worker.SOLVER_DEFAULT] == {
            "name": "DPM++ 2M", "algorithm_type": "dpmsolver++", "solver_order": 2}, \
            "the default is the scheduler the model is built with"
        assert worker.SOLVERS["dpmpp_2m_sde"]["algorithm_type"] == "sde-dpmsolver++", \
            "upstream's Gradio demo"

    def test_the_attentions_are_the_ones_vibevoice_declares(self):
        assert list(worker.ATTENTION) == ["sdpa", "eager", "flash_attention_2"]
        assert worker.ATTENTION_DEFAULT == "sdpa"

    def test_a_request_names_its_solver_attention_and_takes(self):
        request = request_for(solver="dpmpp_2m_sde", attention="eager", takes=4, seed=9990)
        assert (request.solver, request.attention, request.takes) == \
            ("dpmpp_2m_sde", "eager", 4)
        assert request.seeds == [9990, 9991, 9992, 9993]
        plain = request_for()
        assert (plain.solver, plain.attention, plain.takes) == ("dpmpp_2m", "sdpa", 1)
        assert plain.seeds is None

    @pytest.mark.parametrize("values, sentence", [
        ({"solver": "euler"}, "that solver is not one VibeVoice's scheduler has"),
        ({"attention": "flex_attention"}, "that attention is not one VibeVoice supports"),
        ({"takes": 5, "seed": 1}, "a render makes 1 to 4 takes"),
        ({"takes": 2}, "a render of several takes needs a seed"),
        ({"takes": -1, "seed": 1}, "the number of takes is negative"),
    ])
    def test_what_the_worker_cannot_do_is_refused_in_a_sentence(self, values, sentence):
        with pytest.raises(worker.Refusal) as raised:
            request_for(**values)
        assert str(raised.value) == sentence

    def test_the_default_solver_is_the_models_own_scheduler_object(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        own = FakeModel.made[0].model.noise_scheduler
        engine.render(request_for(), threading.Event())
        assert FakeModel.made[0].during[-1]["solver"] is own
        assert FakeScheduler.made == [], "nothing is rebuilt for the solver it already is"

    def test_another_solver_is_the_models_scheduler_with_its_algorithm_and_order(self,
                                                                                   tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        own = FakeModel.made[0].model.noise_scheduler
        engine.render(request_for(solver="dpmpp_2m_sde"), threading.Event())
        chosen, overrides = FakeScheduler.made[0]
        assert overrides == {"algorithm_type": "sde-dpmsolver++", "solver_order": 2}
        assert FakeModel.made[0].during[-1]["solver"] is chosen
        assert chosen.config.beta_schedule == "cosine" and \
            chosen.config.prediction_type == "v_prediction", "the rest is the model's own"
        engine.render(request_for(solver="dpmpp_3m"), threading.Event())
        assert FakeScheduler.made[1][1] == {"algorithm_type": "dpmsolver++", "solver_order": 3}
        engine.render(request_for(), threading.Event())
        assert FakeModel.made[0].during[-1]["solver"] is own, "and back to the model's own"

    def test_attention_is_switched_on_the_language_model_without_a_second_load(self,
                                                                                 tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10, "bf16", "sdpa")
        engine.render(request_for(attention="eager"), threading.Event())
        assert FakeModel.made[0].during[-1]["attention"] == "eager"
        engine.render(request_for(), threading.Event())
        assert FakeModel.made[0].during[-1]["attention"] == "sdpa"
        assert len(FakeModel.made) == 1, "one load"

    def test_flash_attention_without_its_package_is_refused_and_nothing_changes(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(worker, "flash_attention_available", lambda: False)
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        with pytest.raises(worker.Refusal) as raised:
            engine.render(request_for(attention="flash_attention_2"), threading.Event())
        assert "flash-attn" in str(raised.value)
        assert FakeModel.made[0].model.language_model.config._attn_implementation == "sdpa"
        assert FakeModel.made[0].generate_calls == []
        monkeypatch.setattr(worker, "flash_attention_available", lambda: True)
        engine.render(request_for(attention="flash_attention_2"), threading.Event())
        assert FakeModel.made[0].during[-1]["attention"] == "flash_attention_2"

    def test_a_batch_is_one_generate_over_the_same_prompt_with_each_takes_seed(self,
                                                                                  tmp_path):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(str(tmp_path), 10)
        reply, audio = engine.render(request_for(takes=4, seed=9990), threading.Event())
        call = FakeProcessor.made[0].calls[0]
        assert call["text"] == ["Speaker 1: Hello there."] * 4
        assert len(call["voice_samples"]) == 4
        assert len(FakeModel.made[0].generate_calls) == 1
        assert [take["seed"] for take in reply["takes"]] == [9990, 9991, 9992, 9993]
        assert len(audio) == sum(take["samples"] for take in reply["takes"]) * 4
        assert torch.seeds == [9990], "the global generators hold the first take's seed"

    def test_a_batch_draws_its_randomness_take_by_take_and_puts_everything_back(self,
                                                                                   tmp_path):
        torch = FakeTorch()
        engine = BoundEngine(torch)
        engine.load(str(tmp_path), 10)
        original = torch.multinomial
        engine.render(request_for(takes=2, seed=5), threading.Event())
        during = FakeModel.made[0].during[-1]
        assert during["hooked"] and during["encode_hooked"]
        model = FakeModel.made[0]
        assert "sample_speech_tokens" not in vars(model)
        assert "encode" not in vars(model.model.acoustic_tokenizer)
        assert torch.multinomial == original

    def test_a_single_take_keeps_the_global_generators_as_it_always_has(self, tmp_path):
        engine = BoundEngine()
        engine.load(str(tmp_path), 10)
        engine.render(request_for(seed=5), threading.Event())
        during = FakeModel.made[0].during[-1]
        assert not during["hooked"] and not during["encode_hooked"]

    def test_each_take_has_generators_of_its_own_on_the_host_and_the_card(self):
        torch = FakeTorch()
        found = worker.TakeRandomness(torch, None, [3, 4], clips=1, device="cuda")
        assert [one.seed for one in found.host] == [3, 4]
        assert [one.seed for one in found.card] == [3, 4]
        assert all(one.device == "cuda" for one in found.card)
        on_cpu = worker.TakeRandomness(torch, None, [3, 4], clips=1, device="cpu")
        assert on_cpu.card is on_cpu.host, "a machine without a card has one generator"

    def test_the_token_choice_is_drawn_a_take_at_a_time(self):
        torch = FakeTorch()
        found = worker.TakeRandomness(torch, None, [3, 4], clips=1)
        found._multinomial = torch.multinomial

        class Rows:
            shape = (2, 7)

            def dim(self):
                return 2

            def __getitem__(self, index):
                return ("row", index.start)

        drawn = []
        torch.cat = lambda parts, dim=0: drawn.extend(parts) or parts
        found.choose(Rows(), num_samples=1)
        assert [generator.seed for _tag, generator in drawn] == [3, 4]

    def test_each_takes_tokens_stop_at_its_end_of_speech(self):
        """A batch runs until its last take ends and pads the others with end of
        speech, so each take's count stops at its first one."""
        rows = {0: [7, 7, 9, 9, 9], 1: [7, 7, 7, 7, 7]}

        class Sequences:
            def __getitem__(self, index):
                take, span = index
                return types.SimpleNamespace(tolist=lambda: rows[take][span.start - 3:])

        assert worker._tokens_of(Sequences(), 0, 3, 9, 5) == 3
        assert worker._tokens_of(Sequences(), 1, 3, 9, 5) == 5
        assert worker._tokens_of(None, 0, 3, 9, 5) == 0


class TestProgressCostsTheRenderNothing:
    def test_samples_are_counted_from_the_shape_and_reported_at_intervals(self):
        seen = []
        progress = worker.Progress(RATE, seen.append, interval=0.0)
        for _step in range(3):
            progress.put(FakeTensor([0.0] * 3200, (1, 1, 3200)), FakeTensor([0]))
        assert progress.samples == 9600
        assert seen[-1] == {"seconds": pytest.approx(9600 / RATE)}
        assert progress.finished_flags == [False]
        progress.end()
        assert progress.finished_flags == [True]

    def test_a_take_that_ends_raises_no_flag_that_would_stop_the_others(self):
        """``generate`` leaves its loop the moment any flag is up, for every
        take at once; a take ending early must not cut the rest off."""
        progress = worker.Progress(RATE, None, takes=4)
        progress.end(FakeTensor([2], (1,)))
        assert progress.finished_flags == [False] * 4
        progress.end()
        assert progress.finished_flags == [True] * 4

    def test_each_take_is_counted_and_the_furthest_is_reported(self):
        seen = []
        progress = worker.Progress(RATE, seen.append, takes=3, interval=0.0)
        progress.put(FakeTensor([0.0] * 3200, (2, 1, 3200)), FakeTensor([0, 2], (2,)))
        progress.put(FakeTensor([0.0] * 3200, (1, 1, 3200)), FakeTensor([2], (1,)))
        assert progress.counts == [3200, 0, 6400]
        assert seen[-1] == {"seconds": pytest.approx(6400 / RATE)}

    def test_the_interval_holds_frames_back(self):
        seen = []
        progress = worker.Progress(RATE, seen.append, interval=60.0)
        for _step in range(5):
            progress.put(FakeTensor([0.0] * 3200, (1, 1, 3200)), FakeTensor([0]))
        assert len(seen) == 1


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


class FakeEngine:
    """The engine's shape without Torch, with a render that can be gated."""

    def __init__(self, uuid=UUID, cuda=True, chunks=5, gate=None, fail=None):
        self.uuid = uuid
        self.cuda = cuda
        self.chunks = chunks
        self.gate = gate
        self.fail = fail
        self.model = None
        self.model_dir = ""
        self.loads = []
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

    def load(self, model_dir, steps, dtype, attention):
        self.loads.append((model_dir, steps, dtype, attention))
        self.model = object()
        self.model_dir = model_dir
        return {"resident_bytes": 18_000_000_000, "weights_bytes": 17_000_000_000,
                "load_seconds": 0.01}

    def unload(self):
        self.unloads += 1
        self.model = None
        self.model_dir = ""
        return {"resident_bytes": 0}

    def render(self, request, cancelled, on_progress):
        if self.fail is not None:
            raise self.fail
        self.renders.append(request)
        made = 0
        for index in range(self.chunks):
            if self.gate is not None:
                self.gate(index)
            if cancelled.is_set():
                break
            made += 3200
            on_progress({"seconds": made / RATE})
        return {"format": "f32le", "seconds": made / RATE, "sample_rate": RATE,
                "render_seconds": 0.01, "peak_bytes": 20_000_000_000,
                "resident_bytes": 18_000_000_000, "cancelled": cancelled.is_set(),
                "tokens": made // 3200, "capped": False,
                "takes": [{"seed": request.seed, "samples": made, "seconds": made / RATE,
                           "tokens": made // 3200, "capped": False}]}, \
            worker.float32_bytes(numpy.zeros(made, dtype=numpy.float32))


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
        assert header["protocol"] == 2
        assert header["worker"] == "vibevoice"
        assert header["attention"][:2] == ["sdpa", "eager"]
        assert header["max_takes"] == 4
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
            assert len(audio) == 5 * 3200 * 4 and header["format"] == "f32le"
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
            assert header["tokens"] == 2 and len(audio) == 2 * 3200 * 4
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
            for request_id, job in (("r1", "first"), ("r2", "second")):
                head, payload = render_header(request_id, job=job)
                stdin.feed(head, payload)
                header, _payload = out.reply(request_id)
                assert header["ok"] is True

        run_worker(FakeEngine(), feed, stdout=Watching())
        assert seen == {"r1": "", "r2": ""}


class TestRunningTheFileDirectly:
    def test_selftest_dispatches_and_serve_is_the_default(self, monkeypatch):
        calls = []
        claimed = io.BytesIO()
        monkeypatch.setattr(worker, "selftest", lambda: calls.append("selftest") or 3)
        monkeypatch.setattr(worker, "_claim_stdout", lambda: claimed)
        monkeypatch.setattr(worker, "serve",
                            lambda stdin, stdout: calls.append(("serve", stdin, stdout)) or 0)
        assert worker.main(["--selftest"]) == 3
        assert worker.main([worker.MARKER, "--parent-pid", "12"]) == 0
        assert calls[0] == "selftest"
        assert calls[1][0] == "serve"
        assert calls[1][1] is sys.stdin.buffer
        assert calls[1][2] is claimed, "the protocol runs on the claimed copy of stdout"

    def test_a_stray_print_never_reaches_the_protocol_pipe(self, tmp_path):
        """Run as a real process: the worker's frames stay readable and every
        stray line -- Python's or a raw write to descriptor 1 -- lands on stderr."""
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
