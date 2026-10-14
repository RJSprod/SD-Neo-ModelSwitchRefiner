"""Breeze's sidecar: its wire, its request, its text, its upstream calls, its loop.

Four things are asserted here that nothing else can assert.

The first is that the worker speaks the same wire as the other sidecars, by
agreement rather than by import, checked to byte equality against VibeVoice's.

The second is the request as the Voice Box's runtime sends it: lines with their
directions, voices that are a sample *with its transcript* or a description,
and every bound the worker holds -- refused in a sentence rather than handed to
the model.

The third is the upstream binding: how a section becomes Breeze's calls. A line
at a time in its speaker's voice; a plain clone at CFG 1, because upstream's
cloning template has no negative prompt and refuses any other; a directed line
and a designed voice at the configuration's CFG; the gap between two speakers
and never between one speaker's lines; take k at seed + k; each speaker's
sample encoded once per render however many lines and takes use it; and the
cancel flag read at every chunk. Checked against stand-ins for Torch, the model,
the codec and upstream's templates and streaming runtime, with the exact
keywords upstream's own functions take.

The fourth is the command loop against a fake engine: the handshake's fields,
one render at a time, progress frames, and a shutdown that answers.

Nothing here imports Torch, transformers, qwen_tts or Breeze's code.
"""

from __future__ import annotations

import dataclasses
import io
import json
import threading
import types

import pytest

from breeze_worker import worker
from vibevoice_worker import worker as vibevoice_worker

numpy = pytest.importorskip("numpy", reason="the worker cuts its voice samples with NumPy")

RATE = 24000


def pcm(seconds: float, value: float = 0.25) -> bytes:
    return numpy.full(int(seconds * RATE), value, dtype="<f4").tobytes()


def header(lines, voices, **values) -> dict:
    found = {"op": "render", "job": "job-1", "lines": lines, "voices": voices}
    found.update(values)
    return found


def sample_voice(speaker, offset, seconds, transcript="Hello there.", **values) -> dict:
    found = {"speaker": speaker, "offset": offset, "count": int(seconds * RATE), "rate": RATE,
             "transcript": transcript}
    found.update(values)
    return found


# --------------------------------------------------------------------------- #
# The wire
# --------------------------------------------------------------------------- #


class TestTheWire:
    def test_a_frame_is_byte_for_byte_vibevoices(self):
        """The runtime reads every GPU worker with one framing; agreed, not imported."""
        mine, theirs = io.BytesIO(), io.BytesIO()
        worker.write_frame(mine, {"id": 3, "op": "render", "x": [1, 2]}, b"\x00\x01\x02")
        vibevoice_worker.write_frame(theirs, {"id": 3, "op": "render", "x": [1, 2]},
                                     b"\x00\x01\x02")
        assert mine.getvalue() == theirs.getvalue()
        mine.seek(0)
        assert worker.read_frame(mine) == ({"id": 3, "op": "render", "x": [1, 2]},
                                           b"\x00\x01\x02")

    def test_an_oversized_frame_is_refused_before_it_is_read(self):
        stream = io.BytesIO(worker._LENGTH.pack(worker.MAX_HEADER + 1) + b"{}")
        with pytest.raises(ValueError):
            worker.read_frame(stream)

    def test_the_end_of_input_is_none(self):
        assert worker.read_frame(io.BytesIO(b"")) is None


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


class TestTheRequest:
    def test_lines_voices_and_options_are_read_as_sent(self):
        payload = pcm(1.0) + pcm(0.5, 0.5)
        found = worker.RenderRequest.parse(header(
            [{"speaker": 1, "text": "  Hello   there. ", "instruction": "warmly"},
             {"speaker": 2, "text": "Hi.", "instruction": ""}],
            [sample_voice(1, 0, 1.0, instruction="calm"),
             sample_voice(2, RATE, 0.5, transcript="Hi back.")],
            seed=7, takes=2, cfg_scale=3.5, attention="sdpa", sampling=True,
            temperature=0.7, top_p=0.95, max_new_tokens=900, line_gap_ms=250), payload)
        assert found.lines == [(1, "Hello there.", "warmly"), (2, "Hi.", "")]
        assert found.seeds == [7, 8] and found.takes == 2
        assert (found.cfg_scale, found.attention, found.sampling, found.temperature,
                found.top_p, found.max_new_tokens, found.gap_ms) == \
            (3.5, "sdpa", True, 0.7, 0.95, 900, 250)
        assert found.voices[1].instruction == "calm" and found.voices[2].transcript == "Hi back."
        assert found.voices[2].samples.size == RATE // 2
        assert float(found.voices[2].samples[0]) == 0.5

    def test_the_defaults_are_upstreams_sampling_and_the_owners_cfg(self):
        found = worker.RenderRequest.parse(header([{"speaker": 1, "text": "Hi."}],
                                                  [sample_voice(1, 0, 1.0)]), pcm(1.0))
        assert (found.cfg_scale, found.attention, found.sampling, found.temperature,
                found.top_p, found.max_new_tokens, found.gap_ms) == \
            (4.0, "flash_attention_2", True, 0.9, 1.0, None, 150)

    def test_greedy_sends_no_temperature_or_top_p(self):
        found = worker.RenderRequest.parse(header([{"speaker": 1, "text": "Hi."}],
                                                  [sample_voice(1, 0, 1.0)], sampling=False,
                                                  temperature=5, top_p=7), pcm(1.0))
        assert (found.sampling, found.temperature, found.top_p) == (False, None, None)

    def test_a_voice_may_be_a_description_with_no_sample(self):
        found = worker.RenderRequest.parse(header(
            [{"speaker": 3, "text": "Hi."}],
            [{"speaker": 3, "offset": 0, "count": 0, "rate": RATE,
              "description": "An old sailor, gravelly."}]), b"")
        assert found.voices[3].samples is None
        assert found.voices[3].description == "An old sailor, gravelly."

    @pytest.mark.parametrize("change, words", [
        ({"lines": []}, "no words"),
        ({"lines": [{"speaker": 5, "text": "Hi."}]}, "outside 1 to 4"),
        ({"lines": [{"speaker": True, "text": "Hi."}]}, "no speaker number"),
        ({"voices": [sample_voice(1, 0, 1.0, transcript="")]}, "has no transcript"),
        ({"voices": [sample_voice(1, 0, 1.0, rate=16000)]}, "16000 Hz"),
        ({"voices": [sample_voice(1, 0, 2.0)]}, "outside the audio"),
        ({"voices": [{"speaker": 1, "offset": 0, "count": 0, "rate": RATE}]},
         "neither a sample nor a description"),
        ({"voices": []}, "Speaker 1 has no voice"),
        ({"voices": [sample_voice(1, 0, 0.5), sample_voice(1, 0, 0.5)]}, "two voices"),
        ({"takes": 5, "seed": 1}, "1 to 4 takes"),
        ({"takes": 2}, "needs a seed"),
        ({"cfg_scale": 0.5}, "CFG"),
        ({"cfg_scale": 9}, "CFG"),
        ({"attention": "flash_attention_3"}, "attention"),
        ({"temperature": 0}, "temperature"),
        ({"top_p": 1.5}, "top-p"),
        ({"max_new_tokens": 1501}, "at most 1500"),
        ({"line_gap_ms": 2001}, "at most 2000 ms"),
        ({"seed": -1}, "negative"),
    ])
    def test_what_the_model_cannot_take_is_refused_in_a_sentence(self, change, words):
        values = header([{"speaker": 1, "text": "Hi."}], [sample_voice(1, 0, 1.0)])
        values.update(change)
        with pytest.raises(worker.Refusal, match=words):
            worker.RenderRequest.parse(values, pcm(1.0))

    def test_a_sample_that_is_not_numbers_is_refused(self):
        bad = numpy.array([0.1, numpy.nan] * (RATE // 2), dtype="<f4").tobytes()
        with pytest.raises(worker.Refusal, match="not numbers"):
            worker.RenderRequest.parse(header([{"speaker": 1, "text": "Hi."}],
                                              [sample_voice(1, 0, 1.0)]), bad)


# --------------------------------------------------------------------------- #
# The text
# --------------------------------------------------------------------------- #


class TestTheText:
    def test_a_line_that_fits_is_one_piece(self):
        assert worker.split_text("  Hello,   there.  ", 1000) == ["Hello, there."]
        assert worker.split_text("", 10) == []

    def test_a_long_line_is_split_at_its_sentences_and_loses_nothing(self):
        line = " ".join(f"Sentence number {n} says a few more words here." for n in range(12))
        pieces = worker.split_text(line, 100)
        assert len(pieces) > 1
        assert " ".join(pieces) == line
        assert all(worker.estimate_frames(piece) <= 100 for piece in pieces)
        assert all(piece.endswith(".") for piece in pieces), "cut at sentence ends"

    def test_a_sentence_too_long_on_its_own_is_cut_at_clauses_then_words(self):
        line = "one, " * 40 + "end."
        pieces = worker.split_text(line, 30)
        assert " ".join(pieces) == " ".join(line.split())
        assert all(worker.estimate_frames(piece) <= 30 for piece in pieces)

    def test_chinese_is_split_and_joined_without_spaces(self):
        line = "今天天氣很好。" * 30
        pieces = worker.split_text(line, 60)
        assert len(pieces) > 1 and "".join(pieces) == line

    def test_frames_are_budgeted_slowly(self):
        assert worker.estimate_frames("one two three") == int(3 * worker.FRAMES_PER_WORD)
        assert worker.estimate_frames("你好") == int(2 * worker.FRAMES_PER_CJK)


# --------------------------------------------------------------------------- #
# The upstream binding, against stand-ins
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Config:
    """Upstream's ``FastStreamingConfig``, the fields this worker reads and writes."""

    max_new_tokens: int = 750
    max_seq_len: int = 1024
    collect_timing: bool = False
    fast_all: "bool | None" = None
    fast_text_encoder: bool = False
    fast_backbone_prefill: bool = False
    fast_backbone_decode: bool = False
    fast_depth_decoder: bool = False
    fast_codec: bool = False
    temperature: "float | None" = None
    top_k: "int | None" = None
    top_p: "float | None" = None
    do_sample: "bool | None" = None
    repetition_penalty: float = 1.1


class Codes:
    """A tensor of codec frames, as far as the worker touches one."""

    def __init__(self, frames: int):
        self.shape = (frames, 16)
        self.ndim = 2

    def cpu(self):
        return self

    def contiguous(self):
        return self


class Torch:
    int16 = "int16"
    __version__ = "2.9.1+cu128"

    @staticmethod
    def as_tensor(value, dtype=None):
        return value


class Chunk:
    def __init__(self, samples):
        self.audio = samples


class Inputs(dict):
    pass


class Upstream:
    """``breeze_infer`` and ``models.fast_streaming``, recording every call made."""

    def __init__(self, frames_per_call=4, samples_per_frame=1920):
        self.frames_per_call = frames_per_call
        self.samples_per_frame = samples_per_frame
        self.calls = []
        self.seeds = []
        self.loads = []
        self.encoded = {}
        self.cancel_after = None
        upstream = self

        def encode_prompt_audio(_audio_tokenizer, audio_path):
            return ("from a file", audio_path)

        def select_template_name(entry):
            if entry.get("ref_audio_path"):
                return "ref_edit_tata" if entry.get("instruction") else "ref_clone_tata"
            return "tts_instruction" if entry.get("instruction") else "tts_plain"

        def prepare_inputs(tokenizer, audio_tokenizer, model, entries, template, *,
                           guidance_scale, guidance_scale_ref, guidance_scale_ins):
            entry = dict(entries[0])
            reference = entry.get("ref_audio_path")
            frames = templates._encode_prompt_audio(audio_tokenizer, reference) \
                if reference else None
            upstream.calls.append({"entry": entry, "template": template, "cfg": guidance_scale,
                                   "ref": guidance_scale_ref, "ins": guidance_scale_ins,
                                   "frames": frames})
            found = Inputs()
            found["input_ids"] = types.SimpleNamespace(shape=(1, 40))
            return found

        templates = types.SimpleNamespace(
            _encode_prompt_audio=encode_prompt_audio,
            select_template_name=select_template_name,
            get_template=lambda name: name,
            prepare_inputs=prepare_inputs)
        self.templates = templates
        self.original_encode = encode_prompt_audio
        self.Path = str
        self.FastStreamingConfig = Config

        def load_runtime(path, *, device, attn_implementation):
            upstream.loads.append({"path": path, "device": device,
                                   "attn": attn_implementation})
            return Tokenizer(), Model(), AudioTokenizer(upstream)

        self.load_runtime = load_runtime
        self.update_generation_config_for_breeze = lambda model: setattr(model, "tuned", True)
        self.set_all_seeds = lambda seed: upstream.seeds.append(("all", seed))

        class Runtime:
            def __init__(self, model, audio_tokenizer, config, *, tokenizer=None):
                self.config = config
                self.configs = []

            def iter_audio_chunks(self, inputs, *, request_id=None, seed=None,
                                  token_observer=None):
                upstream.seeds.append(("stream", seed))
                self.configs.append(self.config)
                upstream.calls[-1]["request_id"] = request_id
                for frame in range(upstream.frames_per_call):
                    if token_observer is not None:
                        token_observer(frame)
                    if upstream.cancel_after is not None \
                            and len(upstream.calls) >= upstream.cancel_after:
                        upstream.cancel_event.set()
                    yield Chunk(numpy.full(upstream.samples_per_frame, 0.1 * (frame + 1),
                                           dtype=numpy.float32))

        self.FastBreezeStreamingRuntime = Runtime


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(str(text).split())))}


class Encoder:
    def __init__(self):
        self.config = types.SimpleNamespace(_attn_implementation="eager")
        self.layer = types.SimpleNamespace(config=self.config)

    def modules(self):
        return [self, self.layer]


class Model:
    def __init__(self):
        self.text_encoder = Encoder()

    def parameters(self):
        return [types.SimpleNamespace(numel=lambda: 1000, element_size=lambda: 2)]


class AudioTokenizer:
    def __init__(self, upstream):
        self.upstream = upstream
        self.model = None

    def encode(self, samples, sr):
        key = float(samples[0])
        self.upstream.encoded[key] = self.upstream.encoded.get(key, 0) + 1
        return {"audio_codes": [Codes(25)]}


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "audio_tokenizer").mkdir()
    return str(tmp_path)


def loaded(model_dir, upstream=None, attention="sdpa", fast_path=False, monkeypatch=None):
    upstream = upstream or Upstream()
    engine = worker.Engine(device="cpu", upstream=upstream)
    engine.torch, engine.numpy = Torch, numpy
    engine.load(model_dir, attention, fast_path)
    return engine, upstream


def parse(lines, voices, payload, **values):
    """A render as the runtime sends one: with the attention it loaded with."""
    values.setdefault("attention", "sdpa")
    return worker.RenderRequest.parse(header(lines, voices, **values), payload)


def run(engine, request, cancelled=None):
    progress = []
    found, audio = engine.render(request, cancelled or threading.Event(), progress.append)
    return found, numpy.frombuffer(audio, dtype="<f4"), progress


class TestLoading:
    def test_a_load_binds_upstreams_runtime_with_the_decoders_attention(self, model_dir):
        engine, upstream = loaded(model_dir, attention="sdpa")
        assert upstream.loads == [{"path": model_dir, "device": "cpu", "attn": "sdpa"}]
        assert engine.model.tuned is True, "upstream's own generation settings"
        config = engine.runtime.config
        assert (config.max_new_tokens, config.max_seq_len) == (1500, 2048)
        assert not any((config.fast_text_encoder, config.fast_backbone_decode,
                        config.fast_backbone_prefill, config.fast_depth_decoder,
                        config.fast_codec)), "eager unless the fast path is on"

    def test_flash_attention_runs_the_text_encoder_and_the_decoders_eager(self, model_dir,
                                                                            monkeypatch):
        """Breeze's decoders hand transformers a 4-D mask, which its Flash path reads
        as padding; so the choice means the text encoder, set on its configurations."""
        monkeypatch.setattr(worker, "flash_attention_available", lambda: True)
        engine, upstream = loaded(model_dir, attention="flash_attention_2")
        assert upstream.loads[0]["attn"] == "eager"
        assert engine.model.text_encoder.config._attn_implementation == "flash_attention_2"
        assert engine.attention == "flash_attention_2"

    def test_flash_attention_without_the_package_is_refused(self, model_dir, monkeypatch):
        monkeypatch.setattr(worker, "flash_attention_available", lambda: False)
        engine = worker.Engine(device="cpu", upstream=Upstream())
        engine.torch, engine.numpy = Torch, numpy
        with pytest.raises(worker.Refusal, match="flash-attn"):
            engine.load(model_dir, "flash_attention_2", False)

    def test_the_fast_path_is_the_cuda_graph_stages_without_triton(self, model_dir,
                                                                     monkeypatch):
        monkeypatch.setattr(worker, "compile_available", lambda: False)
        engine, _upstream = loaded(model_dir, fast_path=True)
        config = engine.runtime.config
        assert (config.fast_text_encoder, config.fast_backbone_decode) == (True, True)
        assert (config.fast_backbone_prefill, config.fast_depth_decoder, config.fast_codec) \
            == (False, False, False)
        monkeypatch.setattr(worker, "compile_available", lambda: True)
        engine, _upstream = loaded(model_dir, fast_path=True)
        assert (engine.runtime.config.fast_depth_decoder, engine.runtime.config.fast_codec) \
            == (True, True)

    def test_loading_what_is_loaded_is_a_no_op(self, model_dir):
        engine, upstream = loaded(model_dir)
        found = engine.load(model_dir, "sdpa", False)
        assert found["already_loaded"] is True and len(upstream.loads) == 1

    def test_a_folder_without_the_codec_is_refused(self, tmp_path):
        engine = worker.Engine(device="cpu", upstream=Upstream())
        engine.torch, engine.numpy = Torch, numpy
        with pytest.raises(worker.Refusal, match="audio_tokenizer"):
            engine.load(str(tmp_path), "sdpa", False)
        with pytest.raises(worker.Refusal, match="does not exist"):
            engine.load(str(tmp_path / "missing"), "sdpa", False)


class TestRenderingLineByLine:
    def test_each_line_is_a_call_in_its_speakers_voice_with_the_gap_between_speakers(
            self, model_dir):
        engine, upstream = loaded(model_dir)
        request = parse([{"speaker": 1, "text": "First."}, {"speaker": 1, "text": "Second."},
                         {"speaker": 2, "text": "Third."}],
                        [sample_voice(1, 0, 1.0, transcript="One sample."),
                         sample_voice(2, RATE, 1.0, transcript="Two sample.")],
                        pcm(1.0, 0.25) + pcm(1.0, 0.5), seed=11, line_gap_ms=100)
        found, audio, _progress = run(engine, request)

        assert [call["entry"]["text"] for call in upstream.calls] == \
            ["First.", "Second.", "Third."]
        assert [call["entry"]["ref_text"] for call in upstream.calls] == \
            ["One sample.", "One sample.", "Two sample."]
        assert [call["template"] for call in upstream.calls] == ["ref_clone_tata"] * 3
        per_line = upstream.frames_per_call * upstream.samples_per_frame
        gap = int(RATE * 0.1)
        assert audio.size == 3 * per_line + gap, "a gap between speakers, none within one"
        assert numpy.all(audio[2 * per_line:2 * per_line + gap] == 0.0)
        assert found["takes"] == [{"seed": 11, "samples": audio.size,
                                   "seconds": audio.size / RATE,
                                   "tokens": 3 * upstream.frames_per_call, "capped": False}]

    def test_a_plain_clone_is_at_cfg_one_and_a_directed_line_at_the_configurations(
            self, model_dir):
        engine, upstream = loaded(model_dir)
        request = parse([{"speaker": 1, "text": "Plain."},
                         {"speaker": 1, "text": "Directed.", "instruction": "whispering"},
                         {"speaker": 2, "text": "Standing."}],
                        [sample_voice(1, 0, 1.0),
                         sample_voice(2, RATE, 1.0, instruction="calm and slow")],
                        pcm(1.0) + pcm(1.0, 0.5), cfg_scale=4.0)
        run(engine, request)
        assert [(call["entry"].get("instruction"), call["cfg"]) for call in upstream.calls] == \
            [(None, 1.0), ("whispering", 4.0), ("calm and slow", 4.0)]
        assert [call["template"] for call in upstream.calls] == \
            ["ref_clone_tata", "ref_edit_tata", "ref_edit_tata"]
        assert all(call["ref"] is None and call["ins"] is None for call in upstream.calls)

    def test_a_lines_direction_overrides_the_slots(self, model_dir):
        engine, upstream = loaded(model_dir)
        request = parse([{"speaker": 1, "text": "Hi.", "instruction": "shouting"}],
                        [sample_voice(1, 0, 1.0, instruction="calm")], pcm(1.0))
        run(engine, request)
        assert upstream.calls[0]["entry"]["instruction"] == "shouting"

    def test_a_designed_voice_speaks_from_its_description_with_the_direction_after(
            self, model_dir):
        engine, upstream = loaded(model_dir)
        request = parse([{"speaker": 2, "text": "Ahoy.", "instruction": "laughing"},
                         {"speaker": 2, "text": "Again."}],
                        [{"speaker": 2, "offset": 0, "count": 0, "rate": RATE,
                          "description": "An old sailor."}], b"", cfg_scale=3.0)
        run(engine, request)
        assert [(call["entry"].get("instruction"), call["template"], call["cfg"])
                for call in upstream.calls] == \
            [("An old sailor. laughing", "tts_instruction", 3.0),
             ("An old sailor.", "tts_instruction", 3.0)]
        assert all("ref_audio_path" not in call["entry"] for call in upstream.calls)

    def test_each_sample_is_encoded_once_and_upstreams_encoder_put_back(self, model_dir):
        engine, upstream = loaded(model_dir)
        request = parse([{"speaker": 1, "text": "A."}, {"speaker": 2, "text": "B."},
                         {"speaker": 1, "text": "C."}],
                        [sample_voice(1, 0, 1.0), sample_voice(2, RATE, 1.0)],
                        pcm(1.0, 0.25) + pcm(1.0, 0.5), seed=3, takes=3)
        run(engine, request)
        assert upstream.encoded == {0.25: 1, 0.5: 1}, "once a render, whatever reads it"
        assert all(isinstance(call["frames"], Codes) for call in upstream.calls), \
            "the templates found the frames encoded here, never a file"
        assert upstream.templates._encode_prompt_audio is upstream.original_encode

    def test_take_k_is_rendered_whole_at_seed_plus_k(self, model_dir):
        engine, upstream = loaded(model_dir)
        request = parse([{"speaker": 1, "text": "A."}, {"speaker": 1, "text": "B."}],
                        [sample_voice(1, 0, 1.0)], pcm(1.0), seed=100, takes=2)
        found, _audio, progress = run(engine, request)
        assert upstream.seeds == [("all", 100), ("stream", 100), ("all", 100),
                                  ("stream", 100), ("all", 101), ("stream", 101),
                                  ("all", 101), ("stream", 101)]
        assert [take["seed"] for take in found["takes"]] == [100, 101]
        assert [call["request_id"] for call in upstream.calls] == \
            ["job-1-1-1", "job-1-1-2", "job-1-2-1", "job-1-2-2"]
        assert {(entry["line"], entry["take"]) for entry in progress} >= \
            {(1, 1), (2, 1), (1, 2), (2, 2)}
        assert all(entry["lines"] == 2 and entry["takes"] == 2 for entry in progress)

    def test_sampling_and_its_values_reach_both_decoders_through_the_runtime(self, model_dir):
        engine, _upstream = loaded(model_dir)
        run(engine, parse([{"speaker": 1, "text": "A."}], [sample_voice(1, 0, 1.0)],
                          pcm(1.0), temperature=0.7, top_p=0.8, max_new_tokens=600))
        config = engine.runtime.configs[-1]
        assert (config.do_sample, config.temperature, config.top_p, config.max_new_tokens) == \
            (True, 0.7, 0.8, 600)
        run(engine, parse([{"speaker": 1, "text": "A."}], [sample_voice(1, 0, 1.0)],
                          pcm(1.0), sampling=False))
        config = engine.runtime.configs[-1]
        assert (config.do_sample, config.temperature, config.top_p) == (False, None, None)

    def test_a_long_line_is_several_calls_cloned_from_one_reference(self, model_dir):
        engine, upstream = loaded(model_dir)
        line = " ".join(f"This is sentence {n} of a long speech." for n in range(40))
        run(engine, parse([{"speaker": 1, "text": line}], [sample_voice(1, 0, 1.0)], pcm(1.0),
                          max_new_tokens=120))
        assert len(upstream.calls) > 1
        assert " ".join(call["entry"]["text"] for call in upstream.calls) == line
        assert {call["entry"]["ref_audio_path"] for call in upstream.calls} == \
            {"model-chain-voice-1"}

    def test_a_call_that_fills_its_cap_is_reported_capped(self, model_dir):
        engine, upstream = loaded(model_dir, upstream=Upstream(frames_per_call=8))
        found, _audio, _progress = run(engine, parse(
            [{"speaker": 1, "text": "A."}], [sample_voice(1, 0, 1.0)], pcm(1.0),
            max_new_tokens=8))
        assert found["capped"] is True and found["takes"][0]["capped"] is True

    def test_a_cancel_stops_at_the_next_chunk_and_says_so(self, model_dir):
        upstream = Upstream()
        engine, upstream = loaded(model_dir, upstream=upstream)
        cancelled = threading.Event()
        upstream.cancel_event = cancelled
        upstream.cancel_after = 1
        found, audio, _progress = run(engine, parse(
            [{"speaker": 1, "text": "A."}, {"speaker": 1, "text": "B."}],
            [sample_voice(1, 0, 1.0)], pcm(1.0)), cancelled)
        assert found["cancelled"] is True
        assert len(upstream.calls) == 1, "the second line never started"
        assert audio.size == upstream.samples_per_frame, "stopped at the chunk it was read"

    def test_the_text_encoders_attention_changes_without_a_reload(self, model_dir,
                                                                   monkeypatch):
        """Flash attention 2 and Eager build the same eager decoders; only the
        text encoder's attribute differs, and that is set, not loaded."""
        monkeypatch.setattr(worker, "flash_attention_available", lambda: True)
        engine, upstream = loaded(model_dir, attention="eager")
        run(engine, parse([{"speaker": 1, "text": "A."}], [sample_voice(1, 0, 1.0)], pcm(1.0),
                          attention="flash_attention_2"))
        assert len(upstream.loads) == 1
        assert engine.model.text_encoder.layer.config._attn_implementation == \
            "flash_attention_2"
        with pytest.raises(worker.Refusal, match="load it again"):
            run(engine, parse([{"speaker": 1, "text": "A."}], [sample_voice(1, 0, 1.0)],
                              pcm(1.0), attention="sdpa"))

    def test_a_render_without_a_model_is_refused(self):
        engine = worker.Engine(device="cpu", upstream=Upstream())
        engine.torch, engine.numpy = Torch, numpy
        with pytest.raises(worker.Refusal, match="no model"):
            engine.render(parse([{"speaker": 1, "text": "A."}], [sample_voice(1, 0, 1.0)],
                                pcm(1.0)), threading.Event())


# --------------------------------------------------------------------------- #
# The command loop
# --------------------------------------------------------------------------- #


class FakeEngine:
    def __init__(self):
        self.model = None
        self.model_dir = ""
        self.attention = ""
        self.fast_path = False
        self.loads = []
        self.release = threading.Event()

    def probe(self):
        return {"torch": "2.9.1+cu128", "cuda": True, "device_name": "Card",
                "device_uuid": "0123abcd-4567-89ef-0123-456789abcdef", "device_index": 0,
                "total_vram_bytes": 32 << 30, "free_vram_bytes": 30 << 30}

    def resident(self):
        return 9 << 30

    def peak(self):
        return 10 << 30

    def load(self, model_dir, attention, fast_path):
        self.loads.append((model_dir, attention, fast_path))
        self.model = object()
        return {"resident_bytes": 9 << 30, "weights_bytes": 8 << 30, "load_seconds": 1.0,
                "attention": attention, "fast_path": fast_path}

    def unload(self):
        self.model = None
        return {"resident_bytes": 0}

    def render(self, request, cancelled, report):
        report({"seconds": 0.5, "line": 1, "lines": 1, "take": 1, "takes": 1})
        self.release.wait(5.0)
        audio = numpy.full(RATE, 0.1, dtype="<f4").tobytes()
        return {"format": "f32le", "sample_rate": RATE, "cancelled": cancelled.is_set(),
                "takes": [{"seed": request.seed, "samples": RATE, "seconds": 1.0,
                           "tokens": 12, "capped": False}]}, audio


class Pipe:
    """The worker's stdout, read back as frames by the test."""

    def __init__(self):
        self.buffer = io.BytesIO()
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.buffer.write(data)

    def flush(self):
        pass

    def frames(self):
        with self.lock:
            stream = io.BytesIO(self.buffer.getvalue())
        found = []
        while True:
            frame = worker.read_frame(stream)
            if frame is None:
                return found
            found.append(frame)


def frames_in(*requests):
    stream = io.BytesIO()
    for found, payload in requests:
        worker.write_frame(stream, found, payload)
    stream.seek(0)
    return stream


class TestTheLoop:
    def test_the_handshake_names_the_protocol_the_card_and_the_runtime(self, monkeypatch):
        monkeypatch.setattr(worker, "_containment", lambda _pid: "job")
        engine = FakeEngine()
        out = Pipe()
        status = worker.serve(frames_in(({"id": 1, "op": "init", "parent_pid": 0}, b""),
                                        ({"id": 2, "op": "shutdown"}, b"")), out,
                              engine_factory=lambda: engine)
        assert status == 0
        (init, _), (stop, _) = out.frames()
        assert (init["protocol"], init["worker"], init["sample_rate"], init["containment"],
                init["max_takes"]) == (1, "breeze", RATE, "job", 4)
        assert init["device_uuid"] == "0123abcd-4567-89ef-0123-456789abcdef"
        assert "attention" in init and "compile" in init and "breeze" in init
        assert stop == {"ok": True, "id": 2}

    def test_load_render_and_one_render_at_a_time(self, monkeypatch):
        monkeypatch.setattr(worker, "_containment", lambda _pid: "job")
        engine = FakeEngine()
        out = Pipe()
        voices = [sample_voice(1, 0, 1.0)]
        stdin = frames_in(
            ({"id": 1, "op": "init", "parent_pid": 0}, b""),
            ({"id": 2, "op": "load", "model_dir": "/m", "attention": "sdpa",
              "fast_path": True}, b""),
            (header([{"speaker": 1, "text": "Hi."}], voices, id=3, seed=5), pcm(1.0)),
            (header([{"speaker": 1, "text": "Hi."}], voices, id=4, job="job-2"), pcm(1.0)),
            ({"id": 5, "op": "cancel", "job": "job-1"}, b""))

        def release_later():
            import time

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if any(found.get("id") == 5 for found, _ in out.frames()):
                    break
                time.sleep(0.01)
            engine.release.set()

        helper = threading.Thread(target=release_later)
        helper.start()
        reader = io.BufferedReader(io.BytesIO(stdin.getvalue()))
        status = worker.serve(_Held(reader, engine.release), out,
                              engine_factory=lambda: engine)
        helper.join()
        assert status == 0
        replies = {found["id"]: (found, payload) for found, payload in out.frames()
                   if "progress" not in found}
        assert engine.loads == [("/m", "sdpa", True)]
        assert replies[2][0]["ok"] is True
        assert replies[4][0] == {"ok": False, "error": "one render at a time", "refusal": True,
                                 "id": 4}
        assert replies[5][0]["cancelled"] is True
        render, audio = replies[3]
        assert render["ok"] is True and render["cancelled"] is True and len(audio) == 4 * RATE
        progress = [found for found, _ in out.frames() if "progress" in found]
        assert progress and progress[0]["id"] == 3 and progress[0]["progress"]["lines"] == 1

    def test_a_request_before_init_is_refused(self):
        out = Pipe()
        worker.serve(frames_in(({"id": 1, "op": "status"}, b"")), out,
                     engine_factory=FakeEngine)
        assert out.frames()[0][0] == {"ok": False, "error": "runtime is not initialised",
                                      "refusal": True, "id": 1}


class _Held:
    """stdin that ends only once the render has been let go: the loop's end
    joins the lane, and a test that closed its input at once would race it."""

    def __init__(self, reader, release):
        self.reader = reader
        self.release = release

    def read(self, count):
        found = self.reader.read(count)
        if not found:
            self.release.wait(5.0)
        return found


class TestTheSelfTest:
    def test_a_runtime_without_torch_reports_why_and_fails(self, capsys, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        assert worker.selftest() == 1
        report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert report["ok"] is False and "torch" in report["error"]
