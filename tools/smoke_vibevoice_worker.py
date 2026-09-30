"""Drive the VibeVoice worker's own engine against tiny real models, on the CPU.

Never imported by the extension. A maintainer tool, like the pin tools: it
answers "does ``vibevoice_worker/worker.py`` actually drive upstream's code?"
without a graphics card and without downloading a model.

Why it exists
-------------
The worker's unit tests (``tests/test_vibevoice_worker.py``) run it against
stand-ins, because the extension's own test suite never imports Torch. That
pins the calls the worker makes -- and cannot tell whether upstream answers
them the way the stand-ins do. This tool closes that gap: it builds, in a
temporary folder, *real* models of both kinds with random weights and a few
layers each (hidden size 64), saves them with ``save_pretrained``, and has the
worker's :class:`Engine` load them, render with them, stream from them and
cancel them, exactly as it would the published checkpoints on a card.

What it builds
--------------
* A tiny Qwen2-style byte-level BPE tokenizer (the ``tokenizers`` library) that
  carries every special token VibeVoice's text tokenizer looks up:
  ``<|endoftext|>``, ``<|vision_start|>``, ``<|vision_end|>``, ``<|vision_pad|>``
  and ``<|image_pad|>``, in a folder whose name contains ``qwen`` because the
  processors choose their tokenizer class by that substring.
* A tiny long-form model (the 7B's architecture: two decoder layers, both speech
  tokenizers, the diffusion head) and a tiny Realtime model (the 0.5B's: one
  plain decoder layer under two speech layers). Two things are set rather than
  left random, and both only make the renders *finish predictably*: the
  long-form model's ``lm_head`` is zero and the tokenizer numbers the diffusion
  token first among the four the model may choose, so the first of equal
  maxima is always "keep speaking" and a render ends at its token budget; and
  the Realtime model's end-of-speech classifier always says "not yet", so it
  too ends at its budget. The speech scaling buffers are set to 1 and 0 (they
  are NaN until a model is trained), and the diffusion head's zero-initialised
  layers are given random weights so that what the language model says
  reaches the audio -- without which no LoRA could change a sample.
* A Realtime *preset voice* made by running the tiny model's own forward
  passes the way ``generate`` reads ``all_prefilled_outputs``: ``lm`` and
  ``tts_lm`` over a prompt, ``neg_lm`` and ``neg_tts_lm`` over the negative
  token, each kept as ``BaseModelOutputWithPast`` with its ``DynamicCache`` and
  saved with ``torch.save`` -- the format of Microsoft's own preset files.
* PEFT LoRAs made with ``peft`` on the tiny long-form model: a language-model
  adapter (saved as a causal-LM task, as community training scripts save it),
  a folder with every optional part (a whole diffusion head as safetensors, an
  acoustic connector as a pickle, a semantic connector as an adapter pair), and
  two that must be refused.
* A voice "recording": 1.23 seconds of synthetic sound at 24 kHz, deliberately
  not a whole number of the tokenizer's 3200-sample frames, used the way the
  Voice Box and Voice Chat's clone preview both use a recording.

What it proves, and what it cannot
----------------------------------
One line per check, ``ok``, ``FAIL`` or ``skip``, and a non-zero exit status on
any failure. The checks load both kinds; render the 7B from one recording
(Voice Chat's clone preview) and from two; stream the 7B and the 0.5B and
show the frames arrive while ``generate`` is still running on its own thread,
and add up to exactly the samples a whole render of the same seed makes;
cancel each mid-render; join LoRAs at several strengths (strength 0 is the base
model sample for sample, strength 1 is not); load a preset under
``weights_only`` and refuse one that carries code before the code runs; and
drive the whole process over real pipes -- sections sent back to back the
instant each reply is read, a cancel between frames, the stdout claim --
while upstream prints to stdout.

Three things in the worker were found by this tool and fixed: the 7B's
``generate`` never flags a render that ran out of budget (its loop ends one
step before the check that would), so ``capped`` was never true; the render
slot was freed only after the reply was written, so a parent sending its next
section on that reply could be refused "one render at a time" (a race: it
shows in some runs and not others, and the unit test pins it); and upstream
prints to stdout -- the protocol pipe -- at the end of every capped Realtime
render. A fourth observation needs no fix: PEFT's causal-LM wrapper happens to
work on the bare decoder with this closure, so the worker's plain wrapper is
insurance rather than a repair.

bitsandbytes' 8-bit and 4-bit loading need a CUDA device and are *not*
exercised: the tool checks the worker refuses them on the CPU with its
sentence and that ``transformers`` accepts the exact quantisation
configuration the worker builds, and says ``skip`` for the load itself.

Usage
-----
    <python> tools/smoke_vibevoice_worker.py [--keep DIR] [--real-preset FILE]

The interpreter must have torch, transformers, vibevoice (with the five
overlay files the installer copies in), peft and the tokenizers library -- the
installed runtime's, which on Windows is

    model_chain_voice\\vibevoice\\runtime\\env\\Scripts\\python.exe

or any scratch environment with the same closure. ``--real-preset`` also reads
one of Microsoft's own preset files through the worker's loader.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibevoice_worker import worker  # noqa: E402 - the path above is what makes it importable

RATE = worker.SAMPLE_RATE
HOP = 3200
"""Samples per speech frame: 24 kHz at 7.5 frames a second."""

TOKENIZER_DIRNAME = "tokenizer-qwen-tiny"
SPECIALS = ("<|vision_pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>")
"""Added after the 256 byte symbols, in this order: the diffusion token first,
so a zero ``lm_head`` always chooses it (see the module docstring)."""

QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
              r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
"""Qwen2's own pre-tokenizer split."""

HIDDEN = 64
ACOUSTIC = {"vae_dim": 8, "fix_std": 0.5, "std_dist_type": "gaussian", "channels": 1,
            "causal": True, "encoder_n_filters": 4, "decoder_n_filters": 4,
            "encoder_ratios": [8, 5, 5, 4, 2, 2], "encoder_depths": "1-1-1-1-1-1-1",
            "conv_norm": "none", "pad_mode": "constant", "mixer_layer": "depthwise_conv",
            "layernorm": "RMSNorm", "disable_last_norm": True}
SEMANTIC = {"vae_dim": 8, "fix_std": 0, "std_dist_type": "none", "channels": 1,
            "causal": True, "encoder_n_filters": 4, "encoder_ratios": [8, 5, 5, 4, 2, 2],
            "encoder_depths": "1-1-1-1-1-1-1", "conv_norm": "none", "pad_mode": "constant",
            "mixer_layer": "depthwise_conv", "layernorm": "RMSNorm", "disable_last_norm": True}
DIFFUSION = {"hidden_size": HIDDEN, "head_layers": 1, "head_ffn_ratio": 2.0, "latent_size": 8,
             "speech_vae_dim": 8, "prediction_type": "v_prediction", "ddpm_num_steps": 1000,
             "ddpm_num_inference_steps": 4, "ddpm_beta_schedule": "cosine",
             "rms_norm_eps": 1e-5}
"""The encoder ratios multiply to 3200, the processors' compression ratio, so a
recording of any length turns into as many latent frames as the prompt has
speech positions -- the one size in these models that is not free."""

PRESET = "en-Smoke_man"
LONG_BUDGET = 16
REALTIME_BUDGET = 40
"""Token budgets that keep every render to a second or two of CPU time."""


class Skip(Exception):
    """A check that cannot run here, and says why."""


def expect(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------- #
# Building the tiny world
# --------------------------------------------------------------------------- #


def build_tokenizer(folder: Path) -> int:
    """The tiny tokenizer, as the four files the installer ships. Returns its size."""
    from tokenizers import (AddedToken, Regex, Tokenizer, decoders, models, normalizers,
                            pre_tokenizers)

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {symbol: index for index, symbol in enumerate(alphabet)}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(token, special=True, normalized=False)
                                  for token in SPECIALS])
    folder.mkdir(parents=True)
    tokenizer.save(str(folder / "tokenizer.json"))
    (folder / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (folder / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    added = {str(len(vocab) + index): {"content": token, "lstrip": False, "normalized": False,
                                       "rstrip": False, "single_word": False, "special": True}
             for index, token in enumerate(SPECIALS)}
    config = {"add_bos_token": False, "add_prefix_space": False, "added_tokens_decoder": added,
              "additional_special_tokens": [token for token in SPECIALS
                                            if token != "<|endoftext|>"],
              "bos_token": None, "clean_up_tokenization_spaces": False,
              "eos_token": "<|endoftext|>", "errors": "replace", "model_max_length": 32768,
              "pad_token": "<|endoftext|>", "split_special_tokens": False,
              "tokenizer_class": "Qwen2Tokenizer", "unk_token": None}
    (folder / "tokenizer_config.json").write_text(json.dumps(config, indent=2),
                                                   encoding="utf-8")
    return len(vocab) + len(SPECIALS)


def decoder_config(vocab_size: int, layers: int, context: int) -> dict:
    return {"model_type": "qwen2", "vocab_size": vocab_size, "hidden_size": HIDDEN,
            "intermediate_size": 2 * HIDDEN, "num_hidden_layers": layers,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": context, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": False, "use_sliding_window": False, "sliding_window": None,
            "hidden_act": "silu"}


def preprocessor(folder: Path, tokenizer: Path, processor_class: str) -> None:
    """The local ``preprocessor_config.json`` the installer writes, pointing at
    the local tokenizer so nothing is fetched from the hub."""
    (folder / "preprocessor_config.json").write_text(json.dumps({
        "processor_class": processor_class, "speech_tok_compress_ratio": HOP,
        "db_normalize": True, "language_model_pretrained_name": str(tokenizer),
        "audio_processor": {"feature_extractor_type": "VibeVoiceTokenizerProcessor",
                            "sampling_rate": RATE, "normalize_audio": True,
                            "target_dB_FS": -25, "eps": 1e-6}}, indent=2), encoding="utf-8")


def settle(model) -> None:
    """The few values a random model cannot be left with (module docstring)."""
    import torch

    with torch.no_grad():
        model.model.speech_scaling_factor.fill_(1.0)
        model.model.speech_bias_factor.fill_(0.0)
        for parameter in model.model.prediction_head.parameters():
            if not bool(parameter.abs().sum()):
                parameter.normal_(0.0, 0.05)
        head = getattr(model, "lm_head", None)
        if head is not None:
            head.weight.zero_()


def build_longform(folder: Path, tokenizer: Path, vocab_size: int) -> None:
    import torch
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )

    torch.manual_seed(1234)
    config = VibeVoiceConfig(acoustic_tokenizer_config=dict(ACOUSTIC),
                             semantic_tokenizer_config=dict(SEMANTIC),
                             decoder_config=decoder_config(vocab_size, 2, 1024),
                             diffusion_head_config=dict(DIFFUSION))
    model = VibeVoiceForConditionalGenerationInference(config)
    settle(model)
    model.save_pretrained(str(folder))
    preprocessor(folder, tokenizer, "VibeVoiceProcessor")


def build_realtime(folder: Path, tokenizer: Path, vocab_size: int) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast

    torch.manual_seed(4321)
    config = VibeVoiceStreamingConfig(acoustic_tokenizer_config=dict(ACOUSTIC),
                                      decoder_config=decoder_config(vocab_size, 3, 512),
                                      diffusion_head_config=dict(DIFFUSION),
                                      tts_backbone_num_hidden_layers=2)
    model = VibeVoiceStreamingForConditionalGenerationInference(config)
    settle(model)
    with torch.no_grad():
        model.tts_eos_classifier.fc2.weight.zero_()
        model.tts_eos_classifier.fc2.bias.fill_(-10.0)
    model.save_pretrained(str(folder))
    preprocessor(folder, tokenizer, "VibeVoiceStreamingProcessor")

    # The preset voice, made by the model's own prefill passes.
    text = VibeVoiceTextTokenizerFast.from_pretrained(str(tokenizer))
    prompt = torch.tensor([text.encode(" A voice prompt for the smoke test.\n",
                                       add_special_tokens=False)], dtype=torch.long)
    negative = torch.tensor([[text.convert_tokens_to_ids("<|image_pad|>")]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        lm = model.forward_lm(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                              use_cache=True, return_dict=True)
        tts = model.forward_tts_lm(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                   lm_last_hidden_state=lm.last_hidden_state,
                                   tts_text_masks=torch.ones_like(prompt), use_cache=True,
                                   return_dict=True)
        neg_lm = model.forward_lm(input_ids=negative, attention_mask=torch.ones_like(negative),
                                  use_cache=True, return_dict=True)
        neg_tts = model.forward_tts_lm(input_ids=negative,
                                       attention_mask=torch.ones_like(negative),
                                       lm_last_hidden_state=neg_lm.last_hidden_state,
                                       tts_text_masks=torch.ones_like(negative),
                                       use_cache=True, return_dict=True)

    def kept(output):
        return BaseModelOutputWithPast(last_hidden_state=output.last_hidden_state,
                                       past_key_values=output.past_key_values)

    voices = folder / "voices"
    voices.mkdir()
    torch.save({"lm": kept(lm), "tts_lm": kept(tts), "neg_lm": kept(neg_lm),
                "neg_tts_lm": kept(neg_tts)}, str(voices / f"{PRESET}.pt"))

    marker = folder / "a-preset-file-ran-code"

    class CarriesCode:
        """Unpickled without ``weights_only`` this makes a folder; with it, it
        is refused before anything runs. The check looks for the folder."""

        def __reduce__(self):
            return (os.mkdir, (str(marker),))

    torch.save({"lm": CarriesCode()}, str(voices / "en-Mallory_man.pt"))


def build_loras(root: Path, longform: Path) -> dict:
    """The LoRA folders the checks load. Returns their paths by name."""
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )

    torch.manual_seed(99)
    base = VibeVoiceForConditionalGenerationInference.from_pretrained(
        str(longform), torch_dtype=torch.float32, device_map="cpu")

    def randomise(module) -> None:
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0.0, 0.5)

    found = {name: root / name for name in ("lora-llm", "lora-full", "lora-bad-target",
                                            "lora-bad-head")}
    llm = get_peft_model(base.model.language_model,
                         LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    randomise(llm)
    llm.save_pretrained(str(found["lora-llm"]))

    shutil.copytree(found["lora-llm"], found["lora-full"])
    head = {key: value + 0.01 * torch.randn_like(value)
            for key, value in base.model.prediction_head.state_dict().items()}
    (found["lora-full"] / "diffusion_head").mkdir()
    save_file({key: value.contiguous() for key, value in head.items()},
              str(found["lora-full"] / "diffusion_head" / "model.safetensors"))
    acoustic = {key: value * 0.5
                for key, value in base.model.acoustic_connector.state_dict().items()}
    (found["lora-full"] / "acoustic_connector").mkdir()
    torch.save(acoustic, str(found["lora-full"] / "acoustic_connector" / "pytorch_model.bin"))
    semantic = get_peft_model(base.model.semantic_connector,
                              LoraConfig(r=2, lora_alpha=4, target_modules=["fc1", "fc2"],
                                         lora_dropout=0.0, bias="none"))
    randomise(semantic)
    semantic.save_pretrained(str(found["lora-full"] / "semantic_connector"))

    shutil.copytree(found["lora-llm"], found["lora-bad-target"])
    config_path = found["lora-bad-target"] / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target_modules"] = ["no_such_proj"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    shutil.copytree(found["lora-llm"], found["lora-bad-head"])
    (found["lora-bad-head"] / "diffusion_head").mkdir()
    wrong = {key: torch.zeros(3, 3) for key in base.model.prediction_head.state_dict()}
    save_file(wrong, str(found["lora-bad-head"] / "diffusion_head" / "model.safetensors"))
    return {"paths": {name: str(path) for name, path in found.items()},
            "head": {key: value.clone() for key, value in head.items()},
            "acoustic": {key: value.clone() for key, value in acoustic.items()},
            "alpha_over_r": 8 / 4, "semantic_alpha_over_r": 4 / 2}


def recording(seconds: float, pitch: float) -> "tuple[bytes, int]":
    """A voice sample: float32 little-endian PCM at 24 kHz, and its length."""
    import numpy

    count = int(round(seconds * RATE))
    time_axis = numpy.arange(count, dtype=numpy.float64) / RATE
    envelope = 0.5 - 0.5 * numpy.cos(2 * math.pi * numpy.minimum(time_axis / seconds, 1.0))
    sound = (0.3 * numpy.sin(2 * math.pi * pitch * time_axis)
             + 0.1 * numpy.sin(2 * math.pi * 2.5 * pitch * time_axis)) * envelope
    sound += 0.01 * numpy.random.default_rng(7).standard_normal(count)
    return sound.astype("<f4").tobytes(), count


# --------------------------------------------------------------------------- #
# Requests, as the parent sends them
# --------------------------------------------------------------------------- #


def longform_request(world, speakers=1, **values):
    script = [{"speaker": 1, "text": "Hello there, this is a smoke test of the long model."}]
    voices = [{"speaker": 1, "offset": 0, "count": world["voice_one"][1], "rate": RATE}]
    payload = world["voice_one"][0]
    if speakers == 2:
        script.append({"speaker": 2, "text": "And this is the second voice answering."})
        voices.append({"speaker": 2, "offset": world["voice_one"][1],
                       "count": world["voice_two"][1], "rate": RATE})
        payload += world["voice_two"][0]
    header = {"op": "render", "job": "smoke", "script": script, "voices": voices,
              "cfg_scale": 1.3, "seed": 11, "max_new_tokens": LONG_BUDGET, "steps": 3}
    header.update(values)
    return header, payload


def realtime_request(**values):
    header = {"op": "render", "job": "smoke",
              "script": [{"speaker": 1, "text": "“Realtime,” it said, “speaks as it goes.”"}],
              "voices": [{"speaker": 1, "preset": PRESET}], "cfg_scale": None, "seed": 5,
              "max_new_tokens": REALTIME_BUDGET, "steps": 0}
    header.update(values)
    return header, b""


def parse(header_payload):
    header, payload = header_payload
    return worker.RenderRequest.parse(header, payload)


def wav_samples(body: bytes) -> "tuple[int, int, int, bytes]":
    with wave.open(io.BytesIO(body), "rb") as handle:
        return (handle.getnchannels(), handle.getframerate(), handle.getnframes(),
                handle.readframes(handle.getnframes()))


def streamed(engine, request, cancel_at=0, cancelled=None):
    """Render with ``stream`` on, collecting frames and whether ``generate`` was
    still running when each arrived."""
    cancelled = cancelled or threading.Event()
    frames = []

    def on_audio(seq, pcm, samples):
        running = any(thread.name == "vibevoice-generate" and thread.is_alive()
                      for thread in threading.enumerate())
        frames.append({"seq": seq, "pcm": pcm, "samples": samples, "generating": running,
                       "thread": threading.current_thread().name})
        if cancel_at and seq == cancel_at:
            cancelled.set()

    reply, body = engine.render(request, cancelled, None, on_audio)
    return reply, body, frames


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #


class Smoke:
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.passed = self.failed = self.skipped = 0

    def check(self, name: str, function) -> None:
        began = time.monotonic()
        try:
            detail = function() or ""
        except Skip as exc:
            self.skipped += 1
            print(f"skip  {name} — {exc}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - a failed check is reported, not raised
            self.failed += 1
            print(f"FAIL  {name} — {exc.__class__.__name__}: {exc}", flush=True)
            if self.verbose:
                traceback.print_exc()
            return
        self.passed += 1
        print(f"ok    {name} — {detail} ({time.monotonic() - began:.1f} s)", flush=True)


def run_checks(smoke: Smoke, world: dict, real_preset: str) -> None:
    import numpy
    import torch

    state = {}
    engine = worker.Engine()

    def environment():
        from vibevoice.modular import streamer
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (  # noqa: F401
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        import peft
        import transformers

        expect(streamer.AudioStreamer is not None, "no AudioStreamer")
        return (f"torch {torch.__version__}, transformers {transformers.__version__}, "
                f"vibevoice {worker._package_version('vibevoice')} with the overlay, "
                f"peft {peft.__version__}, CUDA {'present' if torch.cuda.is_available() else 'absent'}")

    smoke.check("environment", environment)

    def tokenizer():
        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast

        found = VibeVoiceTextTokenizerFast.from_pretrained(world["tokenizer"])
        ids = {"start": found.speech_start_id, "end": found.speech_end_id,
               "diffusion": found.speech_diffusion_id, "eos": found.eos_id, "pad": found.pad_id}
        expect(len(set(ids.values())) == 5, f"the special tokens collide: {ids}")
        expect(ids["diffusion"] < min(ids["start"], ids["end"], ids["eos"]),
               "the diffusion token is not the first of the four the 7B may choose")
        expect(found.convert_tokens_to_ids("<|image_pad|>") == ids["pad"], "pad is not image_pad")
        return f"{len(found)} tokens, speech {ids['start']}/{ids['end']}/{ids['diffusion']}, eos {ids['eos']}"

    smoke.check("the tiny tokenizer carries every special token", tokenizer)

    # -- the long-form model --------------------------------------------- #

    def longform_load():
        reply = engine.load(world["longform"], 3, kind="longform", precision="bf16",
                            device="cpu")
        expect(engine.kind == "longform" and engine.device == "cpu", "wrong identity")
        dtypes = {parameter.dtype for parameter in engine.model.parameters()}
        expect(dtypes == {torch.float32}, f"parameters are {dtypes}, not float32 on the CPU")
        again = engine.load(world["longform"], 3, kind="longform", precision="bf16",
                            device="cpu")
        expect(again.get("already_loaded") is True, "the same identity reloaded")
        expect(engine.model.ddpm_inference_steps == 3, "the step count was not applied")
        return (f"{reply['weights_bytes'] / 1e6:.1f} MB of float32 weights in "
                f"{reply['load_seconds']:.2f} s; the same identity again is a no-op")

    smoke.check("long-form load on the CPU", longform_load)

    def longform_clone():
        progress = []
        request = parse(longform_request(world))
        reply, body = engine.render(request, threading.Event(), progress.append)
        channels, rate, frames, data = wav_samples(body)
        samples = numpy.frombuffer(data, dtype="<i2")
        expect(channels == 1 and rate == RATE, f"{channels} channel(s) at {rate} Hz")
        expect(frames == LONG_BUDGET * HOP, f"{frames} samples, not {LONG_BUDGET} frames")
        expect(bool(numpy.any(samples)), "the audio is silent")
        expect(reply["tokens"] == LONG_BUDGET and reply["capped"] is True,
               f"tokens {reply['tokens']}, capped {reply['capped']}")
        expect(reply["cancelled"] is False and progress, "no progress frames")
        state["long_whole"] = data
        return (f"{reply['seconds']:.2f} s of audio from a {world['voice_one'][1] / RATE:.2f} s "
                f"recording, {reply['tokens']} tokens, capped at its budget, "
                f"{len(progress)} progress frame(s)")

    smoke.check("long-form render: one speaker cloned from a recording (Voice Chat's "
                "clone preview)", longform_clone)

    def longform_two():
        reply, body = engine.render(parse(longform_request(world, speakers=2)),
                                    threading.Event())
        expect(wav_samples(body)[2] == LONG_BUDGET * HOP, "wrong length")
        return f"{reply['seconds']:.2f} s from two recordings"

    smoke.check("long-form render: two speakers", longform_two)

    def longform_same_seed():
        _reply, body = engine.render(parse(longform_request(world)), threading.Event())
        expect(wav_samples(body)[3] == state["long_whole"], "one seed, two different renders")
        return "the same seed renders the same samples"

    smoke.check("long-form render is repeatable for one seed", longform_same_seed)

    def longform_stream():
        request = parse(longform_request(world, stream=True))
        reply, body, frames = streamed(engine, request)
        expect(body == b"" and reply["streamed"] is True, "a streamed render carried a WAV")
        expect([frame["seq"] for frame in frames] == list(range(1, len(frames) + 1)),
               "frames out of order")
        expect(len(frames) == LONG_BUDGET, f"{len(frames)} frames, not {LONG_BUDGET}")
        expect(frames[0]["generating"], "the first frame arrived after generate had finished")
        expect(frames[0]["thread"] != "vibevoice-generate", "consumed on the generation thread")
        joined = b"".join(frame["pcm"] for frame in frames)
        expect(joined == state["long_whole"],
               "the streamed samples differ from a whole render of the same seed")
        expect(0 < reply["first_audio_ms"] <= reply["render_seconds"] * 1000 + 1000,
               f"first audio after {reply['first_audio_ms']} ms")
        return (f"{len(frames)} frames, the first after {reply['first_audio_ms']} ms while "
                f"generate was still running, the same samples as a whole render")

    smoke.check("long-form streamed render: frames before the end", longform_stream)

    def longform_budget():
        """The default budget, against the real tokenizer and configuration --
        read off the call and cancelled early, so no 512-step render is needed."""
        seen = {}
        real = engine.model.generate

        def watching(**kwargs):
            seen.update(max_new_tokens=kwargs.get("max_new_tokens"),
                        times=kwargs.get("max_length_times"),
                        prompt=int(kwargs["input_ids"].shape[-1]))
            return real(**kwargs)

        engine.model.generate = watching
        try:
            request = parse(longform_request(world, max_new_tokens=None, stream=True))
            reply, _body, frames = streamed(engine, request, cancel_at=2)
        finally:
            del engine.model.generate
        text_tokens = len(engine.processor.tokenizer.encode(request.text,
                                                            add_special_tokens=False))
        ceiling = 1024 - seen["prompt"]
        wanted = min(ceiling, max(worker.MIN_NEW_TOKENS,
                                  worker.TOKENS_PER_TEXT_TOKEN * text_tokens))
        expect(seen["max_new_tokens"] == wanted, f"budget {seen['max_new_tokens']} != {wanted}")
        expect(seen["times"] * seen["prompt"] >= wanted, "max_length_times would bind first")
        expect(reply["cancelled"] is True and len(frames) == 2, "the cancel did not land")
        return (f"{wanted} tokens for a {seen['prompt']}-token prompt, upstream's multiplier "
                f"{seen['times']} kept out of the way")

    smoke.check("long-form default token budget", longform_budget)

    def longform_cancel():
        cancelled = threading.Event()
        began = time.monotonic()
        request = parse(longform_request(world, max_new_tokens=400, stream=True))
        reply, _body, frames = streamed(engine, request, cancel_at=3, cancelled=cancelled)
        expect(reply["cancelled"] is True, "not marked cancelled")
        expect(len(frames) == 3 and reply["frames"] == 3,
               f"{len(frames)} frames sent after a cancel at the third")
        expect(reply["capped"] is False, "a cancelled render claimed its budget")
        return (f"stopped at the third of 400 frames in {time.monotonic() - began:.2f} s: "
                f"the 7B reads the flag every frame")

    smoke.check("long-form cancel mid-render", longform_cancel)

    # -- LoRAs ------------------------------------------------------------ #

    loras = world["loras"]["paths"]

    def lora_scaled(scale):
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer

        engine.load(world["longform"], 3, kind="longform", precision="bf16",
                    lora_dir=loras["lora-llm"], lora_scale=scale, device="cpu")
        expect(isinstance(engine.model.model.language_model, PeftModel),
               "the language model is not wrapped")
        layers = [module for module in engine.model.modules() if isinstance(module, LoraLayer)]
        expect(layers and len(layers) == engine.lora_layers, "LoRA layers miscounted")
        wanted = world["loras"]["alpha_over_r"] * scale
        expect(all(abs(layer.scaling["default"] - wanted) < 1e-9 for layer in layers),
               f"scaling is not {wanted}")
        _reply, body = engine.render(parse(longform_request(world)), threading.Event())
        return wav_samples(body)[3], len(layers)

    def lora_full_strength():
        data, layers = lora_scaled(1.0)
        expect(data != state["long_whole"], "a LoRA at full strength changed nothing")
        return f"{layers} LoRA layers joined, a causal-LM adapter on the bare decoder, the audio differs from the base model's"

    smoke.check("LoRA on the language model at strength 1", lora_full_strength)

    def lora_zero_strength():
        data, _layers = lora_scaled(0.0)
        expect(data == state["long_whole"], "a LoRA at strength 0 is not the base model")
        return "strength 0 renders the base model's samples exactly: the scale reaches every layer"

    smoke.check("LoRA at strength 0", lora_zero_strength)

    def lora_every_part():
        from peft import PeftModel

        engine.load(world["longform"], 3, kind="longform", precision="bf16",
                    lora_dir=loras["lora-full"], lora_scale=0.5, device="cpu")
        head = engine.model.model.prediction_head.state_dict()
        for key, value in world["loras"]["head"].items():
            expect(torch.equal(head[key], value), f"the diffusion head's {key} was not loaded")
        acoustic = engine.model.model.acoustic_connector.state_dict()
        for key, value in world["loras"]["acoustic"].items():
            expect(torch.equal(acoustic[key], value),
                   f"the acoustic connector's {key} was not loaded")
        expect(isinstance(engine.model.model.semantic_connector, PeftModel),
               "the semantic connector's adapter was not joined")
        expect(engine.lora_parts == ("llm", "diffusion_head", "acoustic_connector",
                                     "semantic_connector"), f"parts {engine.lora_parts}")
        reply, body = engine.render(parse(longform_request(world)), threading.Event())
        samples = numpy.frombuffer(wav_samples(body)[3], dtype="<i2")
        expect(samples.size == LONG_BUDGET * HOP and numpy.any(samples), "no audio")
        return (f"diffusion head (safetensors), acoustic connector (pickle, weights only), "
                f"semantic connector (adapter pair), {engine.lora_layers} layers at 0.5")

    smoke.check("LoRA with every optional part", lora_every_part)

    def lora_refusals():
        found = []
        for name, sentence in (("lora-bad-target", "language-model adapter does not fit"),
                               ("lora-bad-head", "diffusion head does not fit")):
            try:
                engine.load(world["longform"], 3, kind="longform", lora_dir=loras[name],
                            device="cpu")
            except worker.Refusal as exc:
                expect(sentence in str(exc), f"{name}: {exc}")
                found.append(str(exc))
            else:
                raise AssertionError(f"{name} was accepted")
            expect(engine.model is None, "a model half-joined to a refused LoRA stayed loaded")
        return "; ".join(found)

    smoke.check("a LoRA the model will not take is refused naming the part", lora_refusals)

    def quantised():
        engine.load(world["longform"], 3, kind="longform", device="cpu")
        for precision in ("int8", "nf4"):
            try:
                engine.load(world["longform"], 3, kind="longform", precision=precision,
                            device="cpu")
            except worker.Refusal as exc:
                expect(str(exc) == "8-bit and 4-bit loading need a CUDA device", str(exc))
            else:
                raise AssertionError(f"{precision} loaded on the CPU")
            expect(engine.kind == "longform" and engine.precision == "bf16",
                   "the refusal cost the loaded model")
        eight = engine._quantisation("int8")
        four = engine._quantisation("nf4")
        expect(eight.load_in_8bit and eight.llm_int8_skip_modules == list(worker.QUANTISE_SKIP),
               "the 8-bit configuration")
        expect(four.load_in_4bit and four.bnb_4bit_quant_type == "nf4"
               and four.bnb_4bit_compute_dtype == torch.bfloat16
               and four.bnb_4bit_use_double_quant
               and four.llm_int8_skip_modules == list(worker.QUANTISE_SKIP),
               "the NF4 configuration")
        raise Skip("refused on the CPU with its sentence, and transformers accepts both "
                   "BitsAndBytesConfigs the worker builds; loading them needs a CUDA device")

    smoke.check("8-bit and NF4", quantised)

    # -- the Realtime model ---------------------------------------------- #

    realtime = worker.Engine()

    def realtime_load():
        realtime.load(world["realtime"], 0, kind="realtime", voices_dir=world["voices"],
                      device="cpu")
        scheduler = realtime.model.model.noise_scheduler.config
        expect(scheduler.algorithm_type == "sde-dpmsolver++", scheduler.algorithm_type)
        expect(scheduler.beta_schedule == "squaredcos_cap_v2", scheduler.beta_schedule)
        expect(realtime.model.ddpm_inference_steps == 5, "not five steps")
        return "Microsoft's scheduler swap (SDE DPM-Solver++, cosine) and five steps"

    smoke.check("Realtime load on the CPU", realtime_load)

    def realtime_preset():
        prefilled = realtime.preset(PRESET)
        expect(set(prefilled) == set(worker.PRESET_KEYS), f"keys {sorted(prefilled)}")
        for key in worker.PRESET_KEYS:
            expect(prefilled[key].past_key_values is not None, f"{key} has no cache")
        refused = []
        for stem, sentence in (("en-Mallory_man", "not one the Realtime model reads"),
                               ("de-Nobody_man", "is not installed"),
                               ("../en-Smoke_man", "letters, digits")):
            try:
                realtime.preset(stem)
            except worker.Refusal as exc:
                expect(sentence in str(exc), f"{stem}: {exc}")
                refused.append(stem)
            else:
                raise AssertionError(f"{stem} was read")
        expect(not (Path(world["realtime"]) / "a-preset-file-ran-code").exists(),
               "reading a preset file ran the code it carried")
        return ("read under weights_only with the two allowed classes; a file carrying code "
                "was refused before it ran, a missing voice and a path were refused")

    smoke.check("Realtime preset voices load safely", realtime_preset)

    def realtime_whole():
        reply, body = realtime.render(parse(realtime_request()), threading.Event())
        channels, rate, frames, data = wav_samples(body)
        expect(channels == 1 and rate == RATE and frames and frames % HOP == 0,
               f"{frames} samples")
        expect(reply["capped"] is True, "a budget-bound render not marked capped")
        state["realtime_whole"] = data
        return f"{reply['seconds']:.2f} s in {reply['tokens']} speech frames"

    smoke.check("Realtime render with the fabricated preset", realtime_whole)

    def realtime_stream():
        reply, body, frames = streamed(realtime, parse(realtime_request(stream=True)))
        expect(body == b"" and reply["streamed"] is True, "a streamed render carried a WAV")
        expect(len(frames) > 1 and frames[0]["generating"],
               "the first frame did not arrive while generating")
        expect(b"".join(frame["pcm"] for frame in frames) == state["realtime_whole"],
               "the streamed samples differ from a whole render of the same seed")
        return (f"{len(frames)} frames, the first after {reply['first_audio_ms']} ms while "
                f"generate was still running, the same samples as a whole render")

    smoke.check("Realtime streamed render: frames before the end", realtime_stream)

    def realtime_refusals():
        refused = []
        for header, sentence in (
                (realtime_request(script=[{"speaker": 1, "text": "One voice."},
                                          {"speaker": 2, "text": "Another voice."}],
                                  voices=[{"speaker": 1, "preset": PRESET},
                                          {"speaker": 2, "preset": PRESET}]),
                 "one voice at a time"),
                (longform_request(world), "cannot clone a recording")):
            try:
                realtime.render(parse(header), threading.Event())
            except worker.Refusal as exc:
                expect(sentence in str(exc), str(exc))
                refused.append(sentence)
            else:
                raise AssertionError(f"{sentence!r} was not refused")
        for values, sentence in (({"lora_dir": loras["lora-llm"]}, "does not take a LoRA"),
                                 ({"precision": "int8"}, "full precision only")):
            try:
                realtime.load(world["realtime"], 0, kind="realtime", device="cpu", **values)
            except worker.Refusal as exc:
                expect(sentence in str(exc), str(exc))
                refused.append(sentence)
            else:
                raise AssertionError(f"{sentence!r} was not refused")
        expect(realtime.kind == "realtime", "a refusal cost the loaded model")
        return "; ".join(refused)

    smoke.check("Realtime refuses two voices, a recording, a LoRA and 8-bit", realtime_refusals)

    def realtime_cancel():
        request = parse(realtime_request(stream=True, max_new_tokens=400))
        began = time.monotonic()
        reply, _body, frames = streamed(realtime, request, cancel_at=2)
        expect(reply["cancelled"] is True and len(frames) == 2 and reply["frames"] == 2,
               f"{len(frames)} frames sent, cancelled {reply['cancelled']}")
        return (f"nothing sent after the cancel at the second frame; generation stopped at "
                f"its window's end, {time.monotonic() - began:.2f} s in")

    smoke.check("Realtime cancel mid-render", realtime_cancel)

    def realtime_own_budget():
        """``max_new_tokens`` left to the model: its context is its budget."""
        reply, _body = realtime.render(parse(realtime_request(max_new_tokens=None)),
                                       threading.Event())
        expect(reply["capped"] is True and reply["tokens"] > REALTIME_BUDGET,
               f"{reply['tokens']} tokens")
        return f"{reply['tokens']} speech frames before the 512-token context ran out"

    smoke.check("Realtime render with the model's own budget", realtime_own_budget)

    if real_preset:
        def microsoft_preset():
            reader = worker.Engine()
            reader._frameworks()
            reader.device = "cpu"
            reader.voices_dir = str(Path(real_preset).parent)
            found = reader.preset(Path(real_preset).stem)
            hidden = found["lm"].last_hidden_state
            return (f"{Path(real_preset).name}: four passes, lm {tuple(hidden.shape)}, "
                    f"tts_lm {tuple(found['tts_lm'].last_hidden_state.shape)}, cast to "
                    f"{hidden.dtype}")

        smoke.check("one of Microsoft's own preset files through the worker's loader",
                    microsoft_preset)

    engine.unload()
    realtime.unload()
    smoke.check("the whole process over real pipes, while upstream prints",
                lambda: over_pipes(world))


# --------------------------------------------------------------------------- #
# The process, over real pipes
# --------------------------------------------------------------------------- #


WRAPPER = """
import sys
sys.path.insert(0, {root!r})
from vibevoice_worker import worker


class CpuEngine(worker.Engine):
    \"\"\"The worker's engine, with a card the smoke test does not have.\"\"\"

    def probe(self):
        torch, _numpy = self._frameworks()
        return {{"torch": str(torch.__version__), "cuda": True,
                 "device_name": "the smoke test's processor",
                 "device_uuid": "GPU-00000000-0000-0000-0000-000000000000",
                 "device_index": 0, "total_vram_bytes": 0, "free_vram_bytes": 0}}


raise SystemExit(worker.main(sys.argv[1:], engine_factory=CpuEngine))
"""


class Pipe:
    """The parent's end: frames out, frames in on a reader thread."""

    def __init__(self, process):
        self.process = process
        self.frames = []
        self.lock = threading.Condition()
        self.closed = False
        self.stderr = []
        self.chained = {}
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        self.next_id = 0

    def _read(self):
        try:
            while True:
                found = worker.read_frame(self.process.stdout)
                if found is None:
                    break
                with self.lock:
                    self.frames.append(found)
                    self.lock.notify_all()
                    then = self.chained.pop(found[0].get("id"), None) \
                        if "ok" in found[0] else None
                if then is not None:
                    then(found[0])
        except Exception as exc:  # noqa: BLE001 - a corrupted stream is the failure to show
            with self.lock:
                self.frames.append(({"id": "?", "ok": False, "error": repr(exc)}, b""))
        finally:
            with self.lock:
                self.closed = True
                self.lock.notify_all()

    def _drain(self):
        for line in iter(self.process.stderr.readline, b""):
            self.stderr.append(line.decode("utf-8", "replace").rstrip())

    def send(self, header, payload=b"") -> str:
        self.next_id += 1
        found = dict(header)
        found["id"] = f"s{self.next_id}"
        worker.write_frame(self.process.stdin, found, payload)
        return found["id"]

    def chain(self, header, count, timeout=120.0):
        """``count`` renders, each written by the reader thread the instant it
        has read the previous one's reply. Returns the replies."""
        replies = []
        done = threading.Event()
        state = {"left": count}

        def next_one(reply=None):
            if reply is not None:
                replies.append(reply)
                state["left"] -= 1
            if state["left"] <= 0:
                done.set()
                return
            request_id = self.send(header)
            with self.lock:
                self.chained[request_id] = next_one

        next_one()
        expect(done.wait(timeout), f"{len(replies)} of {count} chained renders answered")
        return replies

    def reply(self, request_id, timeout=120.0, on_frame=None):
        deadline = time.monotonic() + timeout
        seen = 0
        with self.lock:
            while True:
                mine = [(header, payload) for header, payload in self.frames
                        if header.get("id") in (request_id, "?")]
                for header, payload in mine[seen:]:
                    if on_frame is not None and "audio" in header:
                        on_frame(header, payload)
                seen = len(mine)
                for header, payload in mine:
                    if "ok" in header:
                        if header.get("id") == "?":
                            raise AssertionError(f"the protocol stream broke: {header['error']}")
                        return header, payload, [one for one in mine if "audio" in one[0]]
                if self.closed:
                    raise AssertionError("the worker's stdout closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"no reply to {request_id}")
                self.lock.wait(remaining)


def over_pipes(world) -> str:
    """The worker as a real process: init, loads, streamed and whole renders,
    status, cancel and shutdown, over the pipes the parent would hold -- and
    upstream's own ``print`` at the end of every capped Realtime render, which
    must land on stderr rather than in the frame stream."""
    process = subprocess.Popen(
        [sys.executable, "-c", WRAPPER.format(root=str(ROOT)), worker.MARKER,
         "--parent-pid", str(os.getpid())],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1"))
    pipe = Pipe(process)
    try:
        init, _payload, _audio = pipe.reply(pipe.send({"op": "init", "parent_pid": os.getpid()}))
        expect(init["ok"] and init["protocol"] == 1 and init["worker"] == "vibevoice",
               f"handshake {init}")
        loaded, _payload, _audio = pipe.reply(pipe.send({
            "op": "load", "model_dir": world["realtime"], "steps": 0, "kind": "realtime",
            "precision": "bf16", "voices_dir": world["voices"], "device": "cpu"}))
        expect(loaded["ok"] and loaded["kind"] == "realtime", f"load {loaded}")
        status, _payload, _audio = pipe.reply(pipe.send({"op": "status"}))
        expect((status["kind"], status["precision"], status["lora"]) ==
               ("realtime", "bf16", False), f"status {status}")

        header, _unused = realtime_request(stream=True)
        reply, body, audio = pipe.reply(pipe.send(header))
        expect(reply["ok"] and reply["streamed"] and body == b"", f"streamed reply {reply}")
        expect(audio and [one[0]["audio"]["seq"] for one in audio] ==
               list(range(1, len(audio) + 1)), "audio frames out of order")
        expect(all(one[0]["audio"]["rate"] == RATE and
                   len(one[1]) == 2 * one[0]["audio"]["samples"] for one in audio),
               "an audio frame's header and payload disagree")
        header, _unused = realtime_request()
        whole, body, _audio = pipe.reply(pipe.send(header))
        expect(whole["ok"] and body[:4] == b"RIFF", f"whole reply {whole}")
        expect(wav_samples(body)[3] == b"".join(one[1] for one in audio),
               "over the pipe too, streamed and whole are the same samples")

        # Sections back to back, as the Voice Box renders a script with pauses:
        # each next render leaves the moment the last reply is read.
        sections = 16
        header, _unused = realtime_request(max_new_tokens=12)
        replies = pipe.chain(header, sections)
        refused = [one.get("error") for one in replies if not one.get("ok")]
        expect(not refused, f"a section sent on its predecessor's reply was refused: {refused}")

        loaded, _payload, _audio = pipe.reply(pipe.send({
            "op": "load", "model_dir": world["longform"], "steps": 3, "kind": "longform",
            "lora_dir": world["loras"]["paths"]["lora-llm"], "lora_scale": 0.5,
            "device": "cpu"}))
        expect(loaded["ok"] and loaded["kind"] == "longform" and loaded["lora"] is True,
               f"load {loaded}")
        header, payload = longform_request(world, max_new_tokens=400, stream=True,
                                           job="spoken")
        request_id = pipe.send(header, payload)
        cancelled = {}

        def cancel_on_second(frame_header, _payload):
            if frame_header["audio"]["seq"] == 2 and not cancelled:
                cancelled["id"] = pipe.send({"op": "cancel", "job": "spoken"})

        reply, body, audio = pipe.reply(request_id, on_frame=cancel_on_second)
        expect(reply["ok"] and reply["cancelled"] is True and body == b"",
               f"cancelled reply {reply}")
        expect(len(audio) < 400, "the cancel did not land")
        answer, _payload, _audio = pipe.reply(cancelled["id"])
        expect(answer["ok"] and answer["cancelled"] is True, f"cancel answer {answer}")

        done, _payload, _audio = pipe.reply(pipe.send({"op": "shutdown"}))
        expect(done["ok"], "shutdown")
        expect(process.wait(timeout=60) == 0, f"exit status {process.returncode}")
        diverted = [line for line in pipe.stderr if "Reached maximum generation length" in line]
        expect(diverted, "upstream never printed; the claim on stdout went unexercised")
        return (f"handshake, two loads, a Realtime render streamed and whole with the same "
                f"samples, {sections} sections back to back, {len(audio)} long-form frames "
                f"before a cancel, a clean exit; upstream printed {len(diverted)} line(s) and "
                f"every one reached stderr")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)


# --------------------------------------------------------------------------- #
# Running this file
# --------------------------------------------------------------------------- #


def build(root: Path) -> dict:
    tokenizer = root / TOKENIZER_DIRNAME
    vocab_size = build_tokenizer(tokenizer)
    size = 64 * math.ceil((vocab_size + 1) / 64)
    longform = root / "longform"
    realtime = root / "realtime"
    build_longform(longform, tokenizer, size)
    build_realtime(realtime, tokenizer, size)
    return {"tokenizer": str(tokenizer), "longform": str(longform), "realtime": str(realtime),
            "voices": str(realtime / "voices"), "loras": build_loras(root, longform),
            "voice_one": recording(1.23, 180.0), "voice_two": recording(0.8, 260.0)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", default="", help="build in this folder and keep it")
    parser.add_argument("--real-preset", default="",
                        help="also read one of Microsoft's preset .pt files")
    parser.add_argument("--verbose", action="store_true", help="tracebacks for failures")
    options = parser.parse_args(argv)
    # The worker is always offline; so is this.
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        import torch
    except ImportError:
        print("FAIL  environment — this interpreter has no torch", flush=True)
        return 2
    # One thread: the models are tiny, a pool of them only adds overhead, and
    # in a container with a CPU quota an OpenMP team on the main thread was
    # measured ten to thirty times slower per operation than one thread.
    torch.set_num_threads(1)
    smoke = Smoke(options.verbose)
    began = time.monotonic()
    if options.keep:
        root = Path(options.keep).resolve()
        root.mkdir(parents=True, exist_ok=True)
        holder = None
    else:
        holder = tempfile.TemporaryDirectory(prefix="vibevoice-smoke-")
        root = Path(holder.name)
    try:
        world = {}

        def building():
            world.update(build(root / "world"))
            return f"tokenizer, two models, a preset voice and four LoRAs under {root.name}"

        smoke.check("build the tiny models", building)
        if world:
            run_checks(smoke, world, options.real_preset)
    finally:
        if holder is not None:
            holder.cleanup()
    print(f"{smoke.passed} passed, {smoke.failed} failed, {smoke.skipped} skipped in "
          f"{time.monotonic() - began:.0f} s", flush=True)
    return 1 if smoke.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
