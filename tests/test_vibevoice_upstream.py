"""The worker against upstream's own code: a tiny VibeVoice, on the CPU, when one can be built.

Everything else about the VibeVoice worker is tested against stand-ins, which
prove the calls it makes and never what upstream does with them. The three
things this round reaches into -- the scheduler, the language model's attention,
and the random numbers a batch of takes draws -- are only proven by running
them, so this file builds a VibeVoice from upstream's own classes, shrunk to a
few thousand parameters and randomly initialised, with a toy byte-level
tokenizer, and drives the worker's real ``Engine`` through it on the CPU.

It needs ``torch``, ``transformers`` 4.51, ``diffusers`` and the ``vibevoice``
0.0.1 package, which neither Forge nor this suite installs (they live in
VibeVoice's own runtime), so it skips everywhere they are missing. To run it,
install them and pytest in a venv and run ``<venv>/bin/python -m pytest
tests/test_vibevoice_upstream.py`` from the extension's folder (about a minute
on a CPU).

The language model's head is replaced by one that speaks for a few frames and
then ends a take when that take's own hidden state says so; takes therefore end
at different steps, which is the case that matters for a batch.
"""

from __future__ import annotations

import importlib.util
import json
import threading

import pytest

if not all(importlib.util.find_spec(name) for name in ("torch", "vibevoice", "tokenizers")):
    pytest.skip("torch, tokenizers and the vibevoice package are not installed here",
                allow_module_level=True)

import numpy  # noqa: E402
import torch  # noqa: E402

from vibevoice_worker import worker  # noqa: E402

RATE = 24000


def _tokenizer(root):
    from tokenizers import Tokenizer, decoders, models
    from tokenizers.pre_tokenizers import ByteLevel

    folder = root / "tokenizer-qwen-tiny"
    folder.mkdir()
    vocab = {ch: index for index, ch in enumerate(sorted(ByteLevel.alphabet()))}
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
                "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>"]
    for name in specials:
        vocab[name] = len(vocab)
    found = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    found.pre_tokenizer = ByteLevel(add_prefix_space=False)
    found.decoder = decoders.ByteLevel()
    found.add_special_tokens(specials)
    found.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
        "model_max_length": 8192, "add_prefix_space": False}), encoding="utf-8")
    return folder, len(vocab)


class _Head(torch.nn.Module):
    """Diffusion frames, then a take ends when its own hidden state leans far enough."""

    def __init__(self, vocab_size, diffusion, eos, hidden, quiet=4, offset=1.0):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.register_buffer("direction", torch.randn(hidden, generator=generator))
        self.vocab_size, self.diffusion, self.eos = vocab_size, diffusion, eos
        self.quiet, self.offset = quiet, offset
        self.steps = 0
        self.positive = False

    def forward(self, hidden):
        logits = torch.full(hidden.shape[:-1] + (self.vocab_size,), -30.0, dtype=hidden.dtype)
        logits[..., self.diffusion] = 0.0
        if self.positive and self.steps > self.quiet:
            score = (hidden @ self.direction.to(hidden.dtype)) / (
                hidden.norm(dim=-1) * self.direction.norm() + 1e-6)
            logits[..., self.eos] = (score * 4.0 - self.offset).to(hidden.dtype)
        return logits


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )

    root = tmp_path_factory.mktemp("vibevoice-tiny")
    tokenizer_dir, vocab = _tokenizer(root)
    small = dict(encoder_n_filters=4, encoder_ratios=[8, 5, 5, 4, 2, 2],
                 encoder_depths="1-1-1-1-1-1-1", vae_dim=8)
    config = VibeVoiceConfig(
        acoustic_tokenizer_config=dict(small, decoder_n_filters=4, fix_std=0.5,
                                       std_dist_type="gaussian"),
        semantic_tokenizer_config=dict(small),
        decoder_config=dict(model_type="qwen2", vocab_size=vocab + 9, hidden_size=32,
                            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, max_position_embeddings=8192,
                            tie_word_embeddings=False, rms_norm_eps=1e-6),
        diffusion_head_config=dict(hidden_size=32, head_layers=2, head_ffn_ratio=2.0,
                                   latent_size=8, speech_vae_dim=8, ddpm_num_steps=1000,
                                   ddpm_num_inference_steps=20, ddpm_beta_schedule="cosine",
                                   prediction_type="v_prediction"))
    torch.manual_seed(1234)
    model = VibeVoiceForConditionalGenerationInference(config)
    with torch.no_grad():
        model.model.speech_scaling_factor.fill_(1.0)
        model.model.speech_bias_factor.fill_(0.0)
    model_dir = root / "model"
    model.save_pretrained(str(model_dir))
    (model_dir / "preprocessor_config.json").write_text(json.dumps({
        "processor_class": "VibeVoiceProcessor", "speech_tok_compress_ratio": 3200,
        "db_normalize": True,
        "audio_processor": {"feature_extractor_type": "VibeVoiceTokenizerProcessor",
                            "sampling_rate": RATE, "normalize_audio": True,
                            "target_dB_FS": -25, "eps": 1e-6},
        "language_model_pretrained_name": str(tokenizer_dir)}), encoding="utf-8")

    found = worker.Engine(device="cpu")
    found.load(str(model_dir), 12, "fp32", "sdpa")
    tokenizer = found.processor.tokenizer
    head = _Head(found.model.config.decoder_config.vocab_size, tokenizer.speech_diffusion_id,
                 tokenizer.eos_token_id, found.model.config.decoder_config.hidden_size)
    found.model.lm_head = head

    def count(_module, _args, kwargs):
        head.positive = kwargs.get("logits_to_keep") == 1
        if head.positive:
            head.steps += 1

    found.model.register_forward_pre_hook(count, with_kwargs=True)
    found.head = head
    return found


def _render(engine, *, seed, takes=1, solver="dpmpp_2m", attention="sdpa", sampling=False,
            speakers=1, steps=12, max_new_tokens=60, hooks_for_one=False):
    voices = [numpy.random.default_rng(10 + index).standard_normal(RATE * 3).astype("<f4") * 0.1
              for index in range(speakers)]
    entries, offset = [], 0
    for index, voice in enumerate(voices):
        entries.append({"speaker": index + 1, "offset": offset, "count": len(voice),
                        "rate": RATE})
        offset += len(voice)
    request = worker.RenderRequest.parse({
        "op": "render", "job": "t", "voices": entries, "cfg_scale": 1.3, "seed": seed,
        "script": [{"speaker": 1 + index % speakers, "text": f"Line {index} of the script."}
                   for index in range(max(2, speakers))],
        "steps": steps, "max_new_tokens": max_new_tokens, "sampling": sampling,
        "temperature": 0.9 if sampling else None, "top_p": 0.95 if sampling else None,
        "solver": solver, "attention": attention, "takes": takes},
        b"".join(voice.tobytes() for voice in voices))
    engine.head.steps = 0
    original = worker._NoHooks
    if hooks_for_one:
        worker._NoHooks = lambda: worker.TakeRandomness(torch, engine.model, [seed], speakers,
                                                        engine.device)
    try:
        header, payload = engine.render(request, threading.Event())
    finally:
        worker._NoHooks = original
    audio, at = [], 0
    for take in header["takes"]:
        audio.append(numpy.frombuffer(payload[at * 4:(at + take["samples"]) * 4], dtype="<f4"))
        at += take["samples"]
    return header, audio


class TestTheSolversRunOnVibeVoicesOwnSchedule:
    def test_every_solver_finishes_every_step_count_and_karras_and_lu_would_not(self):
        """Karras and Lu spacings land several noise levels on timestep 999 of
        the cosine schedule, and the scheduler runs off its end: that is why
        the worker keeps the model's own spacing for every solver."""
        from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler

        base = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="cosine",
                                           prediction_type="v_prediction")

        def finishes(steps, **choice):
            scheduler = type(base).from_config(base.config, **choice)
            scheduler.set_timesteps(steps)
            sample = torch.randn(2, 8)
            try:
                for step in scheduler.timesteps:
                    sample = scheduler.step(torch.randn(2, 8), step, sample).prev_sample
            except IndexError:
                return False
            return len(scheduler.timesteps) == steps and bool(torch.isfinite(sample).all())

        for entry in worker.SOLVERS.values():
            choice = {"algorithm_type": entry["algorithm_type"],
                      "solver_order": entry["solver_order"]}
            assert all(finishes(steps, **choice) for steps in range(1, 51)), entry["name"]
        assert not finishes(12, use_karras_sigmas=True)
        assert not finishes(12, use_lu_lambdas=True)

    @pytest.mark.parametrize("solver", list(worker.SOLVERS))
    def test_a_batch_of_every_solver_renders(self, engine, solver):
        header, audio = _render(engine, seed=77, takes=2, solver=solver, max_new_tokens=8)
        assert len(audio) == 2 and all(len(take) and numpy.isfinite(take).all()
                                       for take in audio)

    def test_the_default_is_the_models_own_scheduler_and_the_sde_one_is_the_demos(self,
                                                                                  engine):
        _render(engine, seed=1, solver="dpmpp_2m_sde", max_new_tokens=4)
        chosen = engine.model.model.noise_scheduler
        assert (chosen.config.algorithm_type, chosen.config.solver_order) == \
            ("sde-dpmsolver++", 2)
        assert chosen.config.beta_schedule == "cosine" and \
            chosen.config.prediction_type == "v_prediction"
        _render(engine, seed=1, max_new_tokens=4)
        assert engine.model.model.noise_scheduler is engine.scheduler


class TestEachTakeIsItsOwnSeed:
    @pytest.mark.parametrize("solver, sampling", [("dpmpp_2m", False), ("dpmpp_2m_sde", True)])
    @pytest.mark.parametrize("speakers", [1, 2])
    def test_the_per_take_draws_are_exactly_the_global_ones_for_a_single_take(
            self, engine, solver, sampling, speakers):
        plain_header, plain = _render(engine, seed=4242, solver=solver, sampling=sampling,
                                      speakers=speakers)
        hooked_header, hooked = _render(engine, seed=4242, solver=solver, sampling=sampling,
                                        speakers=speakers, hooks_for_one=True)
        assert len(plain[0]) > 0
        assert numpy.array_equal(plain[0], hooked[0])
        assert plain_header["takes"][0]["tokens"] == hooked_header["takes"][0]["tokens"]

    @pytest.mark.parametrize("solver, sampling", [("dpmpp_2m", False), ("dpmpp_2m_sde", True)])
    def test_take_k_of_a_batch_is_the_render_of_seed_plus_k(self, engine, solver, sampling):
        header, batch = _render(engine, seed=9990, takes=4, solver=solver, sampling=sampling)
        assert [take["seed"] for take in header["takes"]] == [9990, 9991, 9992, 9993]
        for index in range(4):
            alone_header, alone = _render(engine, seed=9990 + index, solver=solver,
                                          sampling=sampling)
            assert len(batch[index]) == len(alone[0]) > 0
            assert header["takes"][index]["tokens"] == alone_header["takes"][0]["tokens"]
            assert numpy.abs(batch[index] - alone[0]).max() < 1e-3

    def test_a_take_that_ends_early_does_not_end_the_others(self, engine):
        header, batch = _render(engine, seed=9990, takes=4)
        lengths = [len(take) for take in batch]
        assert len(set(lengths)) > 1, "the takes end at different steps"
        assert max(lengths) > min(lengths)

    def test_nothing_is_left_replaced_after_a_batch(self, engine):
        original = torch.multinomial
        _render(engine, seed=5, takes=2, sampling=True, max_new_tokens=6)
        assert torch.multinomial is original
        assert "sample_speech_tokens" not in vars(engine.model)
        assert "encode" not in vars(engine.model.model.acoustic_tokenizer)


class TestAttentionSwitchesWithoutALoad:
    def test_eager_runs_the_eager_kernel_and_agrees_with_sdpa(self, engine, monkeypatch):
        import transformers.models.qwen2.modeling_qwen2 as qwen

        calls = []
        original = qwen.eager_attention_forward

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(qwen, "eager_attention_forward", counting)
        _header, sdpa = _render(engine, seed=31)
        assert calls == []
        _header, eager = _render(engine, seed=31, attention="eager")
        assert calls, "the eager kernel ran"
        assert len(sdpa[0]) == len(eager[0])
        assert numpy.abs(sdpa[0] - eager[0]).max() < 1e-3
        del calls[:]
        _render(engine, seed=31, max_new_tokens=4)
        assert calls == [], "and SDPA again after it"

    def test_flash_attention_without_its_package_is_refused_and_nothing_changes(self, engine):
        if worker.flash_attention_available():
            pytest.skip("flash-attn is installed here")
        with pytest.raises(worker.Refusal, match="flash-attn"):
            _render(engine, seed=31, attention="flash_attention_2", max_new_tokens=4)
        assert engine.model.model.language_model.config._attn_implementation == "sdpa"
