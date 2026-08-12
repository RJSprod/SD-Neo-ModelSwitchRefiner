"""Two-stage orchestration: batch sequencing, seeds, prompts, interruption.

These map onto the acceptance criteria in section 9 of the design document.
"""

from __future__ import annotations

import types

import pytest

import mc_arch
import mc_infotext
import mc_memory

UI_ORDER = (
    "enabled", "target", "modules", "prompt_mode", "prompt", "negative", "styles",
    "seed_mode", "seed_offset", "fixed_seed", "cfg", "steps", "sampler", "scheduler",
    "denoise", "size_multiplier", "edit_mode",
)

DEFAULTS = dict(
    enabled=True,
    target="fluxKlein9B.safetensors",
    modules=[mc_memory.INHERIT_MODULES],
    prompt_mode="Inherit",
    prompt="",
    negative="",
    styles=[],
    seed_mode="Inherit",
    seed_offset=0,
    fixed_seed=-1,
    cfg=1.0,
    steps=20,
    sampler=mc_infotext.INHERIT,
    scheduler=mc_infotext.INHERIT,
    denoise=0.35,
    size_multiplier=1.0,
    edit_mode=mc_arch.EDIT_AUTO,
)


@pytest.fixture
def chain(host, style_store, monkeypatch, image_factory):
    """A ScriptModelChain wired to fakes, with a recording Stage 2."""
    import model_chain

    monkeypatch.setattr(
        host.sd_models,
        "get_closet_checkpoint_match",
        lambda name: None
        if not name or name == "None"
        else types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name.split(".")[0], title=name, sha256="abc123"
        ),
    )
    monkeypatch.setattr(mc_memory, "plan", lambda name, mods=None: mc_memory.ResidencyPlan("dual", "both fit"))

    switches: list[str] = []
    monkeypatch.setattr(
        mc_memory, "ensure_resident",
        lambda name, mods=None: (switches.append((name, mods)), "cold")[1],
    )

    restores: list[str] = []
    monkeypatch.setattr(
        mc_memory, "restore_selection",
        lambda name, mods=None: restores.append((name, mods)),
    )
    monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)

    refine_calls: list = []

    def fake_process_images(p2):
        refine_calls.append(p2)
        image = image_factory(p2.width, p2.height)
        return types.SimpleNamespace(images=[image], index_of_first_image=0)

    monkeypatch.setattr(model_chain, "process_images", fake_process_images)
    monkeypatch.setattr(
        model_chain, "create_infotext",
        lambda p, prompts, seeds, subseeds, index=None, **kw: f"infotext#{index} seed={seeds[index]}",
    )

    script = model_chain.ScriptModelChain()
    return types.SimpleNamespace(
        script=script,
        switches=switches,
        restores=restores,
        refine_calls=refine_calls,
        module=model_chain,
    )


def make_p(host, batch_size=1, n_iter=1, prompt="a castle", negative="lowres", width=1024, height=1024):
    total = batch_size * n_iter
    return types.SimpleNamespace(
        prompt=prompt,
        negative_prompt=negative,
        all_prompts=[prompt] * total,
        all_negative_prompts=[negative] * total,
        batch_size=batch_size,
        n_iter=n_iter,
        width=width,
        height=height,
        steps=30,
        cfg_scale=7.0,
        sampler_name="Euler",
        scheduler="Simple",
        outpath_samples="/out/samples",
        outpath_grids="/out/grids",
        extra_generation_params={},
        comments={},
        do_not_save_samples=False,
        do_not_save_grid=False,
        sd_model_name="modelA",
        save_samples=lambda: host.shared.opts.samples_save,
        comment=lambda text: None,
    )


def make_processed(host, p, image_factory, seed_start=1000):
    total = p.batch_size * p.n_iter
    return host.processing and _Processed(
        images=[image_factory(p.width, p.height) for _ in range(total)],
        all_seeds=[seed_start + i for i in range(total)],
        all_prompts=list(p.all_prompts),
        all_negative_prompts=list(p.all_negative_prompts),
        infotexts=[f"stage1#{i}" for i in range(total)],
    )


class _Processed:
    def __init__(self, images, all_seeds, all_prompts, all_negative_prompts, infotexts, index_of_first_image=0):
        self.images = list(images)
        self.all_seeds = list(all_seeds)
        self.all_subseeds = [-1] * len(all_seeds)
        self.all_prompts = list(all_prompts)
        self.all_negative_prompts = list(all_negative_prompts)
        self.infotexts = list(infotexts)
        self.index_of_first_image = index_of_first_image
        self.comments = ""
        self.info = infotexts[0] if infotexts else ""


def run_chain(chain, host, p, processed, **overrides):
    """Drive before_process -> process -> postprocess with UI-ordered args."""
    settings = {**DEFAULTS, **overrides}
    args = [settings[name] for name in UI_ORDER]

    chain.script.before_process(p, *args)
    chain.script.process(p, *args)
    chain.script.postprocess(p, processed, *args)
    return processed


# --------------------------------------------------------------------------- #
# Batch sequencing (section 3, acceptance criteria 2-5)
# --------------------------------------------------------------------------- #


class TestBatchSequencing:
    @pytest.mark.parametrize(
        "batch_size, n_iter",
        [(1, 1), (4, 2), (2, 3), (8, 1), (1, 8)],
    )
    def test_n_in_n_out(self, chain, host, image_factory, batch_size, n_iter):
        """Acceptance: batch size 4 / batch count 2 produces exactly 8 images."""
        p = make_p(host, batch_size=batch_size, n_iter=n_iter)
        processed = make_processed(host, p, image_factory)
        expected = batch_size * n_iter

        run_chain(chain, host, p, processed)

        assert len(processed.images) == expected
        assert len(processed.infotexts) == expected
        assert len(chain.refine_calls) == expected

    @pytest.mark.parametrize("batch_size, n_iter", [(1, 1), (4, 2), (2, 3), (8, 1)])
    def test_exactly_one_checkpoint_switch_per_generate(self, chain, host, image_factory, batch_size, n_iter):
        """Acceptance: exactly ONE checkpoint switch, regardless of batch dims.

        Naive per-image switching would show up here as one entry per image.
        """
        p = make_p(host, batch_size=batch_size, n_iter=n_iter)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert chain.switches == [("fluxKlein9B.safetensors", [mc_memory.INHERIT_MODULES])]

    def test_model_b_stays_resident_for_the_whole_loop(self, chain, host, image_factory):
        """The switch must happen before the loop, not inside it."""
        p = make_p(host, batch_size=4, n_iter=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert len(chain.switches) == 1
        assert len(chain.refine_calls) == 8

    def test_selection_is_restored_for_the_next_generation(self, chain, host, image_factory):
        """Switching back is deferred, so it costs no second switch now."""
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert chain.restores == [(host.shared.opts.sd_model_checkpoint, [])]

    def test_gallery_holds_only_refined_output(self, chain, host, image_factory):
        """Acceptance: Stage 1 images absent from the gallery."""
        p = make_p(host, batch_size=3)
        processed = make_processed(host, p, image_factory)
        stage1 = list(processed.images)

        run_chain(chain, host, p, processed)

        assert len(processed.images) == 3
        for image in processed.images:
            assert not any(image is original for original in stage1)

    def test_stage_1_images_are_not_written_to_the_normal_output_folder(self, chain, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert p.do_not_save_samples is True
        assert p.do_not_save_grid is True
        for record in host.images.saved:
            assert "model-chain-stage1" not in record["path"]


class TestStageOneIntermediates:
    def test_not_saved_by_default(self, chain, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert not any("model-chain-stage1" in r["path"] for r in host.images.saved)

    def test_saved_to_a_subfolder_when_enabled(self, chain, host, image_factory):
        """Section 3.4: intermediates go to disk only, never the gallery."""
        host.shared.opts.model_chain_save_stage1 = True
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        stage1_saves = [r for r in host.images.saved if "model-chain-stage1" in r["path"]]
        assert len(stage1_saves) == 2
        assert len(processed.images) == 2  # still only refined images in the gallery


# --------------------------------------------------------------------------- #
# Seeds (section 6.3, acceptance criterion 4)
# --------------------------------------------------------------------------- #


class TestSeedHandling:
    def test_each_refine_inherits_its_own_stage_1_seed(self, chain, host, image_factory):
        """Acceptance: refined_image[i].seed == stage1_image[i].seed."""
        p = make_p(host, batch_size=4)
        processed = make_processed(host, p, image_factory, seed_start=5000)

        run_chain(chain, host, p, processed)

        used = [call.seed for call in chain.refine_calls]
        assert used == processed.all_seeds == [5000, 5001, 5002, 5003]

    def test_inherit_is_the_default_without_touching_anything(self, chain, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory, seed_start=77)

        run_chain(chain, host, p, processed)

        assert [c.seed for c in chain.refine_calls] == [77, 78]

    def test_offset_is_added_to_each_inherited_seed(self, chain, host, image_factory):
        p = make_p(host, batch_size=3)
        processed = make_processed(host, p, image_factory, seed_start=100)

        run_chain(chain, host, p, processed, seed_mode="Offset", seed_offset=7)

        assert [c.seed for c in chain.refine_calls] == [107, 108, 109]

    def test_fixed_overrides_the_whole_batch(self, chain, host, image_factory):
        p = make_p(host, batch_size=3)
        processed = make_processed(host, p, image_factory, seed_start=100)

        run_chain(chain, host, p, processed, seed_mode="Fixed", fixed_seed=42)

        assert [c.seed for c in chain.refine_calls] == [42, 42, 42]


# --------------------------------------------------------------------------- #
# Prompt handling (section 5)
# --------------------------------------------------------------------------- #


class TestPromptModes:
    def test_inherit_uses_the_stage_1_prompt_unchanged(self, chain, host, image_factory):
        p = make_p(host, prompt="a castle", negative="lowres")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Inherit")

        call = chain.refine_calls[0]
        assert call.prompt == "a castle"
        assert call.negative_prompt == "lowres"

    def test_append_extends_the_stage_1_prompt(self, chain, host, image_factory):
        p = make_p(host, prompt="a castle", negative="lowres")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Append", prompt="golden hour", negative="haze")

        call = chain.refine_calls[0]
        assert call.prompt == "a castle, golden hour"
        assert call.negative_prompt == "lowres, haze"

    def test_replace_discards_the_stage_1_prompt(self, chain, host, image_factory):
        p = make_p(host, prompt="a castle", negative="lowres")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Replace", prompt="a lake", negative="grain")

        call = chain.refine_calls[0]
        assert call.prompt == "a lake"
        assert call.negative_prompt == "grain"

    def test_styles_apply_in_append_mode(self, chain, host, image_factory, style_store):
        """Acceptance: styles apply correctly in both Append and Replace."""
        p = make_p(host, prompt="a castle")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Append", prompt="dusk", styles=["Cinematic"])

        assert chain.refine_calls[0].prompt == "cinematic still of a castle, dusk, film grain"

    def test_styles_apply_in_replace_mode(self, chain, host, image_factory, style_store):
        p = make_p(host, prompt="a castle")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Replace", prompt="a lake", styles=["Cinematic"])

        assert chain.refine_calls[0].prompt == "cinematic still of a lake, film grain"

    def test_styles_are_ignored_in_inherit_mode(self, chain, host, image_factory, style_store):
        """Section 5.2: the dropdown is ignored (and disabled) in Inherit mode."""
        p = make_p(host, prompt="a castle")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Inherit", styles=["Cinematic"])

        assert chain.refine_calls[0].prompt == "a castle"
        assert mc_infotext.STYLES not in p.extra_generation_params

    def test_styles_are_expanded_once_not_twice(self, chain, host, image_factory, style_store):
        """The resolved prompt is passed through, so p2 must carry no styles."""
        p = make_p(host, prompt="a castle")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Replace", prompt="a lake", styles=["Detailed"])

        call = chain.refine_calls[0]
        assert call.prompt == "a lake, highly detailed"
        assert call.styles == []


class TestExtraNetworks:
    def test_lora_tag_reaches_stage_2_untouched(self, chain, host, image_factory):
        """Section 5.4: no stripping, sanitising or pre-parsing in our code."""
        p = make_p(host, prompt="a castle")
        processed = make_processed(host, p, image_factory)

        run_chain(
            chain, host, p, processed,
            prompt_mode="Replace", prompt="a lake <lora:flux_detail:0.8>",
        )

        assert chain.refine_calls[0].prompt == "a lake <lora:flux_detail:0.8>"

    def test_lora_tag_from_a_style_reaches_stage_2(self, chain, host, image_factory, style_store):
        """Acceptance: a LoRA tag inside a saved style expands and applies."""
        p = make_p(host, prompt="a castle")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Replace", prompt="a lake", styles=["WithLora"])

        assert chain.refine_calls[0].prompt == "a lake <lora:filmgrain:0.6>"

    def test_stage_1_lora_does_not_leak_into_stage_2(self, chain, host, image_factory):
        """Acceptance: Stage 1 LoRAs are not applied to Stage 2.

        In Replace mode the Stage 1 prompt -- LoRA tag and all -- is discarded.
        """
        p = make_p(host, prompt="a castle <lora:sdxl_style:1.0>")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Replace", prompt="a lake")

        assert "sdxl_style" not in chain.refine_calls[0].prompt

    def test_stage_2_runs_without_stage_1_scripts(self, chain, host, image_factory):
        """Stage 2 is an independent generation, not a nested script run."""
        p = make_p(host)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert chain.refine_calls[0].scripts is None


# --------------------------------------------------------------------------- #
# Size (section 6.5)
# --------------------------------------------------------------------------- #


class TestOutputSize:
    def test_default_multiplier_keeps_the_stage_1_size(self, chain, host, image_factory):
        p = make_p(host, width=1024, height=1024)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, size_multiplier=1.0)

        call = chain.refine_calls[0]
        assert (call.width, call.height) == (1024, 1024)

    def test_multiplier_preserves_aspect_ratio(self, chain, host, image_factory):
        p = make_p(host, width=1216, height=832)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, size_multiplier=1.5)

        call = chain.refine_calls[0]
        delta = mc_arch.aspect_ratio_delta((1216, 832), (call.width, call.height))
        assert delta < 0.02

    def test_dimensions_land_on_the_architecture_grid(self, chain, host, image_factory, monkeypatch):
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key("flux2_9b"))
        p = make_p(host, width=1000, height=1000)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, size_multiplier=1.3)

        call = chain.refine_calls[0]
        assert call.width % 16 == 0
        assert call.height % 16 == 0

    def test_size_is_uniform_across_the_batch(self, chain, host, image_factory):
        """Section 3.3: Stage 2 settings apply uniformly."""
        p = make_p(host, batch_size=4, width=1024, height=768)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, size_multiplier=1.25)

        sizes = {(c.width, c.height) for c in chain.refine_calls}
        assert len(sizes) == 1


# --------------------------------------------------------------------------- #
# Interruption (section 3.5)
# --------------------------------------------------------------------------- #


class TestInterruption:
    def test_interrupt_during_stage_1_performs_no_switch(self, chain, host, image_factory):
        p = make_p(host, batch_size=4)
        processed = make_processed(host, p, image_factory)
        host.shared.state.interrupted = True

        run_chain(chain, host, p, processed)

        assert chain.switches == []
        assert chain.refine_calls == []

    def test_interrupt_during_stage_1_returns_labelled_unrefined_images(self, chain, host, image_factory):
        p = make_p(host, batch_size=4)
        processed = make_processed(host, p, image_factory)
        stage1 = list(processed.images)
        host.shared.state.interrupted = True

        run_chain(chain, host, p, processed)

        assert processed.images == stage1
        assert "unrefined" in processed.comments.lower()

    def test_interrupt_during_stage_1_still_writes_the_images_to_disk(self, chain, host, image_factory):
        """Stage 1 saving was suppressed, so the abort path must save instead."""
        p = make_p(host, batch_size=3)
        processed = make_processed(host, p, image_factory)
        host.shared.state.interrupted = True

        run_chain(chain, host, p, processed)

        assert len(host.images.saved) == 3

    def test_interrupt_mid_stage_2_returns_a_full_length_mixed_batch(self, chain, host, image_factory):
        """Section 3.5: never silently return a short batch."""
        p = make_p(host, batch_size=4)
        processed = make_processed(host, p, image_factory)
        stage1 = list(processed.images)

        original = chain.module.process_images
        calls = {"n": 0}

        def interrupt_after_two(p2):
            calls["n"] += 1
            if calls["n"] == 2:
                host.shared.state.interrupted = True
            return original(p2)

        chain.module.process_images = interrupt_after_two
        try:
            run_chain(chain, host, p, processed)
        finally:
            chain.module.process_images = original

        assert len(processed.images) == 4
        # First two refined, last two are the untouched Stage 1 images.
        assert processed.images[2] is stage1[2]
        assert processed.images[3] is stage1[3]

    def test_mixed_batch_surfaces_a_clear_warning(self, chain, host, image_factory):
        p = make_p(host, batch_size=4)
        processed = make_processed(host, p, image_factory)

        original = chain.module.process_images
        calls = {"n": 0}

        def interrupt_after_one(p2):
            calls["n"] += 1
            if calls["n"] == 1:
                host.shared.state.interrupted = True
            return original(p2)

        chain.module.process_images = interrupt_after_one
        try:
            run_chain(chain, host, p, processed)
        finally:
            chain.module.process_images = original

        assert "unrefined" in processed.comments.lower()
        assert "1 of 4" in processed.comments


class TestFailureHandling:
    def test_a_failing_refine_falls_back_to_the_stage_1_image(self, chain, host, image_factory):
        p = make_p(host, batch_size=3)
        processed = make_processed(host, p, image_factory)
        stage1 = list(processed.images)

        original = chain.module.process_images
        calls = {"n": 0}

        def fail_on_second(p2):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("CUDA out of memory")
            return original(p2)

        chain.module.process_images = fail_on_second
        try:
            run_chain(chain, host, p, processed)
        finally:
            chain.module.process_images = original

        assert len(processed.images) == 3
        assert processed.images[1] is stage1[1]

    def test_a_failing_switch_returns_unrefined_stage_1_output(self, chain, host, image_factory, monkeypatch):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)
        stage1 = list(processed.images)

        def boom(name, mods=None):
            raise mc_memory.ModelChainError("no such checkpoint")

        monkeypatch.setattr(mc_memory, "ensure_resident", boom)

        run_chain(chain, host, p, processed)

        assert processed.images == stage1
        assert "unrefined" in processed.comments.lower()


# --------------------------------------------------------------------------- #
# Inertness (acceptance criterion 12)
# --------------------------------------------------------------------------- #


class TestDisabled:
    def test_disabled_adds_nothing_to_infotext(self, chain, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, enabled=False)

        assert p.extra_generation_params == {}

    def test_disabled_performs_no_switch_and_no_refine(self, chain, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)
        stage1 = list(processed.images)

        run_chain(chain, host, p, processed, enabled=False)

        assert chain.switches == []
        assert chain.refine_calls == []
        assert processed.images == stage1

    def test_disabled_leaves_stage_1_saving_alone(self, chain, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, enabled=False)

        assert p.do_not_save_samples is False
        assert p.do_not_save_grid is False

    def test_no_target_selected_is_a_no_op(self, chain, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, target="None")

        assert chain.switches == []
        assert p.extra_generation_params == {}

    def test_builtin_refiner_conflict_is_refused_not_corrupted(self, chain, host, image_factory):
        """The Refiner restores Model A's UNet in a later postprocess hook.

        Letting both run would write those weights into Model B.
        """
        p = make_p(host, batch_size=2)
        p.refiner_checkpoint = "sdxl_refiner.safetensors"
        processed = make_processed(host, p, image_factory)
        stage1 = list(processed.images)

        comments = []
        p.comment = comments.append

        run_chain(chain, host, p, processed)

        assert chain.switches == []
        assert processed.images == stage1
        assert any("Refiner" in c for c in comments)

    def test_missing_checkpoint_is_a_non_fatal_skip(self, chain, host, image_factory, monkeypatch):
        monkeypatch.setattr(host.sd_models, "get_closet_checkpoint_match", lambda name: None)
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)
        stage1 = list(processed.images)

        run_chain(chain, host, p, processed, target="deleted-model.safetensors")

        assert chain.switches == []
        assert processed.images == stage1


# --------------------------------------------------------------------------- #
# Infotext integration (section 7)
# --------------------------------------------------------------------------- #


class TestInfotextIntegration:
    def test_enabled_chain_writes_namespaced_keys(self, chain, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Replace", prompt="a lake")

        assert p.extra_generation_params[mc_infotext.ENABLED] == "True"
        assert p.extra_generation_params[mc_infotext.TARGET] == "fluxKlein9B.safetensors"
        assert all(k.startswith("Model Chain") for k in p.extra_generation_params)

    def test_records_the_resolved_prompt_not_the_template(self, chain, host, image_factory, style_store):
        """Section 5.2: reproduction must not depend on the style library."""
        p = make_p(host, prompt="a castle")
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, prompt_mode="Replace", prompt="a lake", styles=["Cinematic"])

        recorded = p.extra_generation_params[mc_infotext.PROMPT]
        assert recorded == ["cinematic still of a lake, film grain"]
        assert p.extra_generation_params[mc_infotext.STYLES] == "Cinematic"

    def test_records_the_stage_1_size(self, chain, host, image_factory):
        p = make_p(host, width=1216, height=832)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, size_multiplier=1.5)

        assert p.extra_generation_params[mc_infotext.STAGE1_SIZE] == "1216x832"

    def test_refined_images_carry_stage_1_derived_infotext(self, chain, host, image_factory):
        """Pasting a refined image reproduces the whole pipeline, not just Stage 2."""
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory, seed_start=900)

        run_chain(chain, host, p, processed)

        assert processed.infotexts == ["infotext#0 seed=900", "infotext#1 seed=901"]


# --------------------------------------------------------------------------- #
# Edit / reference conditioning
# --------------------------------------------------------------------------- #


@pytest.fixture
def krea2_target(chain, monkeypatch):
    """Point the Stage 2 target at a Krea 2 checkpoint."""
    monkeypatch.setattr(
        mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key("krea2")
    )
    return chain


class TestEditModeOrchestration:
    def test_enable_scopes_the_toggle_to_the_stage_2_pass(self, krea2_target, host, image_factory):
        """The toggle is global, so it must ride on override_settings.

        That way the host applies it for this generation and restores the
        user's value afterwards instead of leaving it flipped.
        """
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Enable", denoise=1.0)

        for call in krea2_target.refine_calls:
            assert call.override_settings == {"krea2_do_reference": True}

    def test_the_global_setting_is_not_mutated_directly(self, krea2_target, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Enable", denoise=1.0)

        assert host.shared.opts.krea2_do_reference is False

    def test_disable_forces_plain_img2img_for_stage_2(self, krea2_target, host, image_factory):
        host.shared.opts.krea2_do_reference = True
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Disable")

        assert krea2_target.refine_calls[0].override_settings == {"krea2_do_reference": False}

    def test_auto_leaves_the_users_setting_alone(self, krea2_target, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Auto")

        assert krea2_target.refine_calls[0].override_settings == {}

    def test_an_architecture_without_edit_mode_gets_no_override(self, chain, host, image_factory, monkeypatch):
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key("sdxl"))
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, edit_mode="Enable")

        assert chain.refine_calls[0].override_settings == {}

    def test_klein_polarity_is_inverted_end_to_end(self, chain, host, image_factory, monkeypatch):
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key("flux2_9b"))
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, edit_mode="Enable", denoise=1.0)

        assert chain.refine_calls[0].override_settings == {"klein_no_reference": False}

    def test_stale_references_are_cleared_before_refining(self, krea2_target, host, image_factory, monkeypatch):
        """A cached model keeps its engine -- and its reference list -- around."""
        cleared = []
        monkeypatch.setattr(mc_memory, "clear_references", lambda: cleared.append(True))

        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Enable", denoise=1.0)

        assert cleared, "references must be cleared before Stage 2 in edit mode"

    def test_each_image_is_refined_singly_so_it_is_its_own_reference(self, krea2_target, host, image_factory):
        """Krea 2 captures only x[0] as the reference, so batches must be 1."""
        p = make_p(host, batch_size=4)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Enable", denoise=1.0)

        assert len(krea2_target.refine_calls) == 4
        for call in krea2_target.refine_calls:
            assert call.batch_size == 1
            assert len(call.init_images) == 1

    def test_reference_is_the_matching_stage_1_image(self, krea2_target, host, image_factory):
        p = make_p(host, batch_size=3)
        processed = make_processed(host, p, image_factory)
        stage1 = list(processed.images)

        run_chain(krea2_target, host, p, processed, edit_mode="Enable", denoise=1.0)

        for index, call in enumerate(krea2_target.refine_calls):
            assert call.init_images[0] is stage1[index]


class TestEditModeWarnings:
    def test_low_denoise_is_flagged(self, krea2_target, host, image_factory):
        """The reference carries the content, so a low denoise is a mistake."""
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Enable", denoise=0.35)

        assert "denoise 0.35 is" in processed.comments

    def test_expected_denoise_is_not_flagged(self, krea2_target, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Enable", denoise=1.0)

        assert "denoise" not in processed.comments

    def test_missing_edit_lora_is_flagged(self, krea2_target, host, image_factory):
        """Krea 2 edit mode is documented as requiring a specific Edit LoRA."""
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(
            krea2_target, host, p, processed,
            edit_mode="Enable", denoise=1.0, prompt_mode="Replace", prompt="a lake",
        )

        assert "Edit LoRA" in processed.comments

    def test_a_lora_tag_satisfies_the_check(self, krea2_target, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(
            krea2_target, host, p, processed,
            edit_mode="Enable", denoise=1.0, prompt_mode="Replace",
            prompt="a lake <lora:krea2_edit:1.0>",
        )

        assert "Edit LoRA" not in processed.comments

    def test_a_lora_from_a_style_satisfies_the_check(self, krea2_target, host, image_factory, style_store):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(
            krea2_target, host, p, processed,
            edit_mode="Enable", denoise=1.0, prompt_mode="Replace",
            prompt="a lake", styles=["WithLora"],
        )

        assert "Edit LoRA" not in processed.comments

    def test_inherit_mode_without_a_lora_is_flagged(self, krea2_target, host, image_factory):
        """Inherit offers no way to add the tag, which is worth saying out loud."""
        p = make_p(host, prompt="a castle", batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Enable", denoise=1.0)

        assert "Edit LoRA" in processed.comments

    def test_klein_does_not_demand_a_lora(self, chain, host, image_factory, monkeypatch):
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key("flux2_9b"))
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, edit_mode="Enable", denoise=1.0)

        assert "Edit LoRA" not in processed.comments

    def test_no_warnings_when_edit_mode_is_off(self, krea2_target, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Disable", denoise=0.35)

        assert processed.comments == ""


class TestEditModeInfotext:
    def test_explicit_mode_is_recorded(self, krea2_target, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(krea2_target, host, p, processed, edit_mode="Enable", denoise=1.0)

        assert p.extra_generation_params[mc_infotext.EDIT_MODE] == "Enable"

    def test_auto_is_absent_so_existing_infotext_is_unchanged(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, edit_mode="Auto")

        assert mc_infotext.EDIT_MODE not in p.extra_generation_params


class TestKleinDefaultReferences:
    """Klein sees the input image as a reference *and* as an img2img init, so
    an ordinary low-denoise refine is valid and must not be second-guessed."""

    @pytest.fixture
    def klein(self, chain, monkeypatch):
        monkeypatch.setattr(
            mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key("flux2_9b")
        )
        return chain

    def test_default_klein_chain_is_not_warned_about_denoise(self, klein, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(klein, host, p, processed, edit_mode="Auto", denoise=0.35)

        assert processed.comments == ""

    def test_explicitly_enabling_klein_edit_does_warn(self, klein, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(klein, host, p, processed, edit_mode="Enable", denoise=0.35)

        assert "denoise 0.35 is" in processed.comments

    def test_stale_references_are_still_cleared_on_the_default_path(self, klein, host, image_factory, monkeypatch):
        """Not warning is a judgement call; not clearing would be a bug."""
        cleared = []
        monkeypatch.setattr(mc_memory, "clear_references", lambda: cleared.append(True))

        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(klein, host, p, processed, edit_mode="Auto", denoise=0.35)

        assert cleared

    def test_krea2_left_on_the_global_toggle_still_warns(self, chain, host, image_factory, monkeypatch):
        """Krea 2 edit mode is opt-in, so having it on is always deliberate."""
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key("krea2"))
        host.shared.opts.krea2_do_reference = True

        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, edit_mode="Auto", denoise=0.35)

        assert "denoise 0.35 is" in processed.comments


class TestStageTwoModules:
    """The Stage 2 VAE / text encoder selection reaches the model transition."""

    def test_selection_is_passed_to_the_switch(self, chain, host, image_factory):
        p = make_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, modules=["flux2_vae.safetensors", "qwen3_8b.safetensors"])

        assert chain.switches == [
            ("fluxKlein9B.safetensors", ["flux2_vae.safetensors", "qwen3_8b.safetensors"])
        ]

    def test_switch_still_happens_exactly_once(self, chain, host, image_factory):
        p = make_p(host, batch_size=4, n_iter=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, modules=["flux2_vae.safetensors"])

        assert len(chain.switches) == 1
        assert len(chain.refine_calls) == 8

    def test_stage_1_modules_are_captured_and_restored(self, chain, host, image_factory):
        """Restoring the checkpoint without its encoders would change what
        Stage 1 loads on the next generation."""
        host.shared.opts.forge_additional_modules = ["/models/VAE/sdxl_vae.safetensors"]
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, modules=["flux2_vae.safetensors"])

        assert chain.restores == [
            (host.shared.opts.sd_model_checkpoint, ["/models/VAE/sdxl_vae.safetensors"])
        ]

    def test_selection_is_recorded_in_infotext(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, modules=["flux2_vae.safetensors"])

        assert p.extra_generation_params["Model Chain Module 1"] == "flux2_vae"

    def test_inheriting_records_nothing(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, modules=[mc_memory.INHERIT_MODULES])

        assert not any(k.startswith("Model Chain Module") for k in p.extra_generation_params)

    def test_built_in_selection_is_recorded(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, modules=[])

        assert p.extra_generation_params["Model Chain Module 1"] == "Built-in"


class TestSamplingSelection:
    """Sampling method and schedule type are chosen independently for Stage 2."""

    def test_both_inherit_stage_1_by_default(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        call = chain.refine_calls[0]
        assert call.sampler_name == p.sampler_name
        assert call.scheduler == p.scheduler

    def test_sampler_override_reaches_stage_2(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, sampler="DPM++ 2M")

        assert chain.refine_calls[0].sampler_name == "DPM++ 2M"

    def test_scheduler_override_reaches_stage_2(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, scheduler="Karras")

        assert chain.refine_calls[0].scheduler == "Karras"

    def test_either_can_be_overridden_without_the_other(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, sampler="Euler")

        call = chain.refine_calls[0]
        assert call.sampler_name == "Euler"
        assert call.scheduler == p.scheduler  # still inherited

    def test_both_are_uniform_across_the_batch(self, chain, host, image_factory):
        """Section 3.3: Stage 2 settings apply uniformly."""
        p = make_p(host, batch_size=4)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, sampler="Euler", scheduler="Beta")

        assert {(c.sampler_name, c.scheduler) for c in chain.refine_calls} == {("Euler", "Beta")}

    def test_overrides_are_recorded_in_infotext(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, sampler="Euler", scheduler="Karras")

        assert p.extra_generation_params[mc_infotext.SAMPLER] == "Euler"
        assert p.extra_generation_params[mc_infotext.SCHEDULER] == "Karras"

    def test_inherited_values_are_not_recorded(self, chain, host, image_factory):
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert mc_infotext.SAMPLER not in p.extra_generation_params
        assert mc_infotext.SCHEDULER not in p.extra_generation_params


class TestUiContract:
    """The UI's return order *is* the hook signature order.

    Gradio passes these positionally, so a control inserted in ui() without a
    matching change to process()/postprocess() silently shifts every argument
    after it -- the prompt would arrive as the seed mode. UI_ORDER is what the
    rest of this file drives the hooks with, so pinning it here keeps the
    tests honest too.
    """

    # elem_id per position; "enabled" is the accordion, whose id is "enable".
    EXPECTED_IDS = (
        "enable", "target", "modules", "prompt_mode", "prompt", "negative",
        "styles", "seed_mode", "seed_offset", "fixed_seed", "cfg", "steps",
        "sampler", "scheduler", "denoise", "size_multiplier", "edit_mode",
    )

    def test_returned_controls_match_the_documented_order(self, chain):
        returned = chain.script.ui(is_img2img=False)

        assert len(returned) == len(UI_ORDER)
        actual = [c.elem_id.removeprefix("script_modelchain_") for c in returned]
        assert tuple(actual) == self.EXPECTED_IDS

    def test_presets_cover_every_stage_2_control(self, chain):
        """A control the preset field list misses would silently not be saved."""
        import mc_presets

        chain.script.ui(is_img2img=False)
        assert set(mc_presets.FIELDS) == set(UI_ORDER)

    def test_preset_field_order_matches_the_ui(self, chain):
        """Preset values map onto controls positionally, so order matters."""
        import mc_presets

        assert tuple(mc_presets.FIELDS) == UI_ORDER

    def test_every_control_is_registered_for_pasting(self, chain):
        chain.script.ui(is_img2img=False)
        registered = {field.api for field in chain.script.infotext_fields}
        for name in ("model_chain_sampler", "model_chain_scheduler", "model_chain_denoise"):
            assert name in registered


class TestDenoiseDefault:
    def test_default_is_a_full_pass(self, chain):
        import model_chain

        assert model_chain.DEFAULT_DENOISE == 1.0

    def test_the_slider_ships_at_the_default(self, chain):
        """The control's own value, not just the constant."""
        import model_chain

        returned = chain.script.ui(is_img2img=False)
        denoise = returned[UI_ORDER.index("denoise")]
        assert denoise.value == model_chain.DEFAULT_DENOISE

    def test_default_needs_no_adjustment_for_edit_mode(self, chain, host, image_factory, monkeypatch):
        """Krea 2 edit expects 1.0, which is now what the slider already holds."""
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key("krea2"))
        import model_chain

        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(
            chain, host, p, processed,
            edit_mode="Enable", denoise=model_chain.DEFAULT_DENOISE,
            prompt_mode="Replace", prompt="a lake <lora:krea2_edit:1.0>",
        )

        assert processed.comments == ""

    def test_a_lowered_denoise_is_still_warned_about_in_edit_mode(self, chain, host, image_factory, monkeypatch):
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key("krea2"))
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(
            chain, host, p, processed,
            edit_mode="Enable", denoise=0.3,
            prompt_mode="Replace", prompt="a lake <lora:krea2_edit:1.0>",
        )

        assert "denoise 0.3 is" in processed.comments
