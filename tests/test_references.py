"""Stage 2 supplemental reference images (section 12).

Three modes: Disabled, Pass Through ImageStitch, and Decoupled. The first is the
default and must leave Stage 2 exactly as it was before the feature existed; the
other two hand Stage 2 an ordered set of references *in addition to* each
image's own Stage 1 handoff.

These map onto the acceptance criteria in the Item 12 specification.
"""

from __future__ import annotations

import types

import pytest

import mc_arch
import mc_infotext
import mc_memory
import mc_presets
import mc_references
from test_orchestration import DEFAULTS, UI_ORDER, make_p, make_processed, run_chain


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


class FakeStitch:
    """Stands in for Forge Neo's "ImageStitch Integrated".

    Only the surface Pass Through is allowed to touch: the title it is found
    by, the argument span its controls occupy, and the class-level memo Model
    Chain has to invalidate. ``process`` records calls so the tests can assert
    it is never used as the Stage 2 implementation.
    """

    cached_parameters = None
    args_from = 1
    args_to = 4

    def __init__(self):
        self.process_calls = 0

    def title(self):
        return mc_references.STITCH_TITLE

    def process(self, *args, **kwargs):
        self.process_calls += 1


class FakeEngine:
    """A reference-capable diffusion engine, as far as Model Chain can see it."""

    def __init__(self):
        self.ref_latents = []
        self.ini_latent = None
        self.cleared = 0

    def clear_references(self):
        self.ref_latents.clear()
        self.cleared += 1


class PlainEngine:
    """An engine with no reference path at all -- no ``ref_latents`` to fill."""

    def __init__(self):
        self.ini_latent = None


def attach_stitch(p, images, *, enabled=True, max_dim=1024):
    """Give ``p`` an ImageStitch script with ``images`` in its input gallery."""
    stitch = FakeStitch()
    FakeStitch.cached_parameters = ["from an earlier generation"]
    gallery = [(image, None) for image in images]
    p.scripts = types.SimpleNamespace(alwayson_scripts=[stitch])
    p.script_args = [None, enabled, gallery, max_dim]
    return stitch, gallery


def wire_engine(chain, host, monkeypatch, arch_key, *, flag=None, engine=None):
    """Give the chain a Stage 2 engine and watch what each refine pass sees.

    ``seen`` is the important half. References are cleared at both ends of a job
    by design, so the finished state says nothing about what Stage 2 was given
    -- the only place to observe that is at each pass, which is also exactly
    what the acceptance criteria are written in terms of.

    ``_to_tensor`` is the identity here: the conversion it performs needs numpy
    and torch, and every assertion below is about *which* images arrive and in
    what order, which the images themselves answer better than tensors would.
    """
    engine = FakeEngine() if engine is None else engine
    host.shared.sd_model = engine
    monkeypatch.setattr(mc_references, "_to_tensor", lambda image: image)
    set_architecture(monkeypatch, arch_key)
    if flag:
        set_model_flag(host, flag)

    seen = []
    refine = chain.module.process_images

    def recording(p2):
        seen.append(list(getattr(engine, "ref_latents", [])))
        return refine(p2)

    monkeypatch.setattr(chain.module, "process_images", recording)

    chain.engine = engine
    chain.seen = seen
    return chain


@pytest.fixture
def refs(chain, host, monkeypatch):
    """A chain whose Stage 2 model is a reference-capable Klein engine."""
    return wire_engine(chain, host, monkeypatch, "flux2_9b", flag="klein")


def set_architecture(monkeypatch, key):
    monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.by_key(key))
    return mc_arch.by_key(key)


def set_model_flag(host, name, value=True):
    from backend.args import dynamic_args

    setattr(dynamic_args, name, value)


def run(chain, host, image_factory, *, batch_size=1, stitch=None, **overrides):
    """Drive one chained generation, optionally with ImageStitch attached."""
    p = make_p(host, batch_size=batch_size)
    if stitch is not None:
        attach_stitch(p, stitch.get("images", []), **{k: v for k, v in stitch.items() if k != "images"})
    processed = make_processed(host, p, image_factory)
    run_chain(chain, host, p, processed, **overrides)
    return p, processed


# --------------------------------------------------------------------------- #
# Disabled (acceptance: Disabled)
# --------------------------------------------------------------------------- #


class TestDisabled:
    def test_is_the_default(self):
        assert DEFAULTS["reference_mode"] == mc_references.DISABLED
        assert mc_references.MODES[0] == mc_references.DISABLED

    def test_no_supplemental_references_reach_stage_2(self, refs, host, image_factory):
        stitch_images = [image_factory(512, 512) for _ in range(3)]

        run(refs, host, image_factory, stitch={"images": stitch_images})

        assert refs.seen == [[]]

    def test_stage_1_imagestitch_references_do_not_survive_into_stage_2(
        self, refs, host, image_factory
    ):
        """The same engine object serves both stages when they share a checkpoint.

        ImageStitch fills it for Stage 1 and cannot clear it for Stage 2,
        because Stage 2 runs with scripts disabled.
        """
        refs.engine.ref_latents.extend(["stage 1 reference"])

        run(refs, host, image_factory)

        assert refs.seen == [[]]

    def test_the_native_stage_1_gallery_is_left_alone(self, refs, host, image_factory):
        stitch_images = [image_factory(512, 512) for _ in range(2)]
        p = make_p(host)
        _, gallery = attach_stitch(p, stitch_images)
        before = list(gallery)

        run_chain(refs, host, p, make_processed(host, p, image_factory))

        assert p.script_args[2] is gallery
        assert gallery == before

    def test_nothing_is_recorded_in_infotext(self, refs, host, image_factory):
        p, _ = run(refs, host, image_factory)

        assert mc_infotext.REFERENCE_MODE not in p.extra_generation_params
        assert mc_infotext.REFERENCE_COUNT not in p.extra_generation_params


# --------------------------------------------------------------------------- #
# Pass Through (acceptance: Pass Through)
# --------------------------------------------------------------------------- #


class TestPassThrough:
    def test_every_stage_2_image_receives_the_references_in_order(
        self, refs, host, image_factory
    ):
        """Acceptance: references [A, B, C] arrive as A, B, C for every image."""
        a, b, c = (image_factory(512, 512, color) for color in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

        run(
            refs,
            host,
            image_factory,
            batch_size=3,
            stitch={"images": [a, b, c]},
            reference_mode=mc_references.PASS_THROUGH,
        )

        assert refs.seen == [[a, b, c]] * 3

    def test_the_same_set_is_reused_for_every_stage_1_result(self, refs, host, image_factory):
        """One encode serves the batch: conditioning reads ref_latents without
        consuming it, so the set is still whole after the last pass."""
        images = [image_factory(512, 512) for _ in range(2)]

        run(
            refs,
            host,
            image_factory,
            batch_size=4,
            stitch={"images": images},
            reference_mode=mc_references.PASS_THROUGH,
        )

        assert len(refs.refine_calls) == 4
        assert refs.seen == [images] * 4

    def test_imagestitch_is_never_executed_as_a_stage_2_script(
        self, refs, host, image_factory
    ):
        p = make_p(host, batch_size=2)
        stitch, _ = attach_stitch(p, [image_factory(512, 512)])

        run_chain(
            refs,
            host,
            p,
            make_processed(host, p, image_factory),
            reference_mode=mc_references.PASS_THROUGH,
        )

        assert stitch.process_calls == 0

    def test_changing_the_gallery_changes_the_stage_2_references(
        self, refs, host, image_factory
    ):
        first = [image_factory(512, 512, (1, 0, 0))]
        second = [image_factory(512, 512, (0, 1, 0)), image_factory(512, 512, (0, 0, 1))]

        run(refs, host, image_factory, stitch={"images": first},
            reference_mode=mc_references.PASS_THROUGH)
        run(refs, host, image_factory, stitch={"images": second},
            reference_mode=mc_references.PASS_THROUGH)

        assert refs.seen == [first, second]

    def test_no_references_from_the_previous_job_survive(self, refs, host, image_factory):
        images = [image_factory(512, 512) for _ in range(2)]

        run(refs, host, image_factory, stitch={"images": images},
            reference_mode=mc_references.PASS_THROUGH)

        assert refs.engine.ref_latents == []
        assert refs.engine.cleared >= 2  # once before the pass, once after

    def test_imagestitch_switched_off_is_a_notice_not_a_failure(
        self, refs, host, image_factory
    ):
        images = [image_factory(512, 512)]

        _, processed = run(
            refs,
            host,
            image_factory,
            stitch={"images": images, "enabled": False},
            reference_mode=mc_references.PASS_THROUGH,
        )

        assert refs.seen == [[]]
        assert "switched off" in processed.comments
        assert len(refs.refine_calls) == 1

    def test_an_empty_gallery_is_a_notice_not_a_failure(self, refs, host, image_factory):
        _, processed = run(
            refs,
            host,
            image_factory,
            stitch={"images": []},
            reference_mode=mc_references.PASS_THROUGH,
        )

        assert "no reference images selected" in processed.comments

    def test_a_host_without_imagestitch_is_a_notice_not_a_failure(
        self, refs, host, image_factory
    ):
        _, processed = run(refs, host, image_factory, reference_mode=mc_references.PASS_THROUGH)

        assert "ImageStitch was not found" in processed.comments

    def test_imagestitchs_own_maximum_side_length_is_honoured(
        self, refs, host, image_factory, monkeypatch
    ):
        """The sizing belongs to ImageStitch in this mode, not to Model Chain."""
        seen = []
        monkeypatch.setattr(
            mc_references, "preprocess", lambda image, limit: seen.append(limit) or image
        )

        run(
            refs,
            host,
            image_factory,
            stitch={"images": [image_factory(512, 512)], "max_dim": 768},
            reference_mode=mc_references.PASS_THROUGH,
            reference_max_dim=256,
        )

        assert seen == [768]


# --------------------------------------------------------------------------- #
# Decoupled (acceptance: Decoupled)
# --------------------------------------------------------------------------- #


class TestDecoupled:
    def test_ordered_references_are_used_for_every_image(self, refs, host, image_factory):
        a, b = image_factory(512, 512, (1, 0, 0)), image_factory(512, 512, (0, 1, 0))

        run(
            refs,
            host,
            image_factory,
            batch_size=3,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(a, None), (b, None)],
        )

        assert refs.seen == [[a, b]] * 3
        assert len(refs.refine_calls) == 3

    def test_stage_1_imagestitch_references_are_not_included(self, refs, host, image_factory):
        """Acceptance: native Stage 1 references do not leak into Stage 2."""
        stitched = image_factory(512, 512, (1, 0, 0))
        own = image_factory(512, 512, (0, 1, 0))

        run(
            refs,
            host,
            image_factory,
            stitch={"images": [stitched]},
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(own, None)],
        )

        assert refs.seen == [[own]]

    def test_works_when_imagestitch_is_disabled(self, refs, host, image_factory):
        own = image_factory(512, 512)

        run(
            refs,
            host,
            image_factory,
            stitch={"images": [image_factory(512, 512)], "enabled": False},
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(own, None)],
        )

        assert refs.seen == [[own]]

    def test_works_when_imagestitch_is_absent_entirely(self, refs, host, image_factory):
        own = image_factory(512, 512)

        run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(own, None)],
        )

        assert refs.seen == [[own]]

    def test_an_empty_gallery_is_a_notice_not_a_failure(self, refs, host, image_factory):
        _, processed = run(
            refs, host, image_factory, reference_mode=mc_references.DECOUPLED, reference_images=[]
        )

        assert "no Stage 2 reference images have been added" in processed.comments

    def test_model_chains_own_maximum_side_length_is_used(
        self, refs, host, image_factory, monkeypatch
    ):
        seen = []
        monkeypatch.setattr(
            mc_references, "preprocess", lambda image, limit: seen.append(limit) or image
        )

        run(
            refs,
            host,
            image_factory,
            stitch={"images": [], "max_dim": 768},
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
            reference_max_dim=512,
        )

        assert seen == [512]


# --------------------------------------------------------------------------- #
# Edit-mode interaction
# --------------------------------------------------------------------------- #


class TestEditModeInteraction:
    def test_supplying_references_implies_reference_conditioning(
        self, refs, host, image_factory
    ):
        """Klein opts *out*, so implying references on means clearing its flag."""
        run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
            edit_mode=mc_arch.EDIT_AUTO,
        )

        assert refs.refine_calls[0].override_settings == {"klein_no_reference": False}

    def test_the_opt_in_polarity_is_implied_the_other_way(
        self, chain, host, image_factory, monkeypatch
    ):
        krea2 = wire_engine(chain, host, monkeypatch, "krea2", flag="krea2")

        run(
            krea2,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
            edit_mode=mc_arch.EDIT_AUTO,
            denoise=1.0,
        )

        assert krea2.refine_calls[0].override_settings == {"krea2_do_reference": True}

    def test_an_explicit_disable_wins_and_the_references_are_dropped(
        self, refs, host, image_factory
    ):
        _, processed = run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
            edit_mode=mc_arch.EDIT_DISABLE,
        )

        assert refs.seen == [[]]
        assert "the explicit setting wins" in processed.comments
        assert refs.refine_calls[0].override_settings == {"klein_no_reference": True}

    def test_disable_still_vetoes_an_architecture_with_no_edit_toggle(
        self, chain, host, image_factory, monkeypatch
    ):
        """Flux.1 Kontext references, but no setting turns that on or off."""
        kontext = wire_engine(chain, host, monkeypatch, "flux", flag="kontext")

        _, processed = run(
            kontext,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
            edit_mode=mc_arch.EDIT_DISABLE,
        )

        assert kontext.seen == [[]]
        assert "the explicit setting wins" in processed.comments

    def test_that_veto_is_explained_in_the_panel(self, chain, monkeypatch):
        from model_chain import _edit_notice

        set_architecture(monkeypatch, "flux")

        note = _edit_notice("kontext.safetensors", mc_arch.EDIT_DISABLE)

        assert "no edit toggle" in note
        assert "Stage 2 reference images" in note

    def test_a_plain_architecture_keeps_its_old_notice(self, chain, monkeypatch):
        from model_chain import _edit_notice

        set_architecture(monkeypatch, "sdxl")

        assert "will be ignored" in _edit_notice("sdxl.safetensors", mc_arch.EDIT_DISABLE)

    def test_the_global_toggle_is_restored_after_encoding(self, refs, host, image_factory):
        """The encode borrows the toggle; it must not keep it."""
        host.shared.opts.klein_no_reference = True

        run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
        )

        assert host.shared.opts.klein_no_reference is True

    def test_the_toggle_is_open_while_the_references_encode(
        self, refs, host, image_factory, monkeypatch
    ):
        host.shared.opts.klein_no_reference = True
        observed = []
        monkeypatch.setattr(
            mc_references,
            "_to_tensor",
            lambda image: observed.append(host.shared.opts.klein_no_reference) or image,
        )

        run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
        )

        assert observed == [False]


# --------------------------------------------------------------------------- #
# Compatibility (acceptance: Compatibility)
# --------------------------------------------------------------------------- #


class TestCompatibility:
    def test_klein_and_krea2_are_the_validated_targets(self):
        for key in ("flux2_4b", "flux2_9b", "krea2"):
            arch = mc_arch.by_key(key)
            assert arch.supports_references
            assert arch.references_are_validated

    def test_the_other_reference_paths_forge_exposes_are_experimental(self):
        for key in ("anima", "qwen", "flux"):
            arch = mc_arch.by_key(key)
            assert arch.supports_references
            assert not arch.references_are_validated

    def test_wan_is_not_a_reference_target(self):
        """ImageStitch feeds Wan a last frame for video, not a reference set."""
        assert not mc_arch.by_key("wan").supports_references

    def test_plain_architectures_have_no_reference_path(self):
        for key in ("sdxl", "sd15", "chroma", "lumina2", "flux_schnell"):
            assert not mc_arch.by_key(key).supports_references

    def test_an_unsupported_architecture_cannot_consume_references(
        self, chain, host, image_factory, monkeypatch
    ):
        sdxl = wire_engine(chain, host, monkeypatch, "sdxl")

        _, processed = run(
            sdxl,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
        )

        assert sdxl.seen == [[]]
        assert "no reference-conditioning path" in processed.comments

    def test_a_variant_without_the_runtime_flag_is_refused(
        self, chain, host, image_factory, monkeypatch
    ):
        """A plain Flux.1 detects as Flux.1; only Kontext has the path."""
        plain_flux = wire_engine(chain, host, monkeypatch, "flux")
        set_model_flag(host, "kontext", False)

        _, processed = run(
            plain_flux,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
        )

        assert plain_flux.seen == [[]]
        assert "does not expose" in processed.comments

    def test_the_same_variant_with_the_flag_set_is_allowed(
        self, chain, host, image_factory, monkeypatch
    ):
        kontext = wire_engine(chain, host, monkeypatch, "flux", flag="kontext")
        own = image_factory(512, 512)

        run(
            kontext,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(own, None)],
        )

        assert kontext.seen == [[own]]

    def test_an_experimental_target_says_so_but_still_runs(
        self, chain, host, image_factory, monkeypatch
    ):
        anima = wire_engine(chain, host, monkeypatch, "anima", flag="anima")
        own = image_factory(512, 512)

        _, processed = run(
            anima,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(own, None)],
            denoise=1.0,
        )

        assert anima.seen == [[own]]
        assert "experimental" in processed.comments

    def test_a_model_that_takes_nothing_is_noticed_by_observation(
        self, chain, host, image_factory, monkeypatch
    ):
        """The count the model actually took is the signal nothing can fake."""
        plain = wire_engine(
            chain, host, monkeypatch, "flux2_9b", flag="klein", engine=PlainEngine()
        )

        _, processed = run(
            plain,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
        )

        assert "took 0 of 1" in processed.comments

    def test_a_partial_set_is_dropped_rather_than_used_out_of_order(
        self, refs, host, image_factory, monkeypatch
    ):
        monkeypatch.setattr(mc_references, "encode", lambda images, max_dim: 2)

        _, processed = run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None) for _ in range(3)],
        )

        assert refs.seen == [[]]
        assert "took 2 of 3" in processed.comments

    def test_an_encode_failure_leaves_the_chain_running(
        self, refs, host, image_factory, monkeypatch
    ):
        def boom(images, max_dim):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(mc_references, "encode", boom)

        _, processed = run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
        )

        assert "could not be encoded" in processed.comments
        assert len(refs.refine_calls) == 1


# --------------------------------------------------------------------------- #
# Cleanup (acceptance: cached models do not retain references)
# --------------------------------------------------------------------------- #


class TestCleanup:
    def test_references_are_cleared_at_both_ends_of_a_job(
        self, refs, host, image_factory, monkeypatch
    ):
        cleared = []
        monkeypatch.setattr(mc_memory, "clear_references", lambda: cleared.append(True))

        run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
        )

        assert len(cleared) >= 2

    def test_a_disabled_chain_still_clears_stale_references(
        self, refs, host, image_factory, monkeypatch
    ):
        """Section 12 extends the clearing discipline to all three modes."""
        cleared = []
        monkeypatch.setattr(mc_memory, "clear_references", lambda: cleared.append(True))

        run(refs, host, image_factory)

        assert len(cleared) >= 2

    def test_a_failed_refine_still_clears_references(
        self, refs, host, image_factory, monkeypatch
    ):
        def boom(p2):
            raise RuntimeError("the pass failed")

        monkeypatch.setattr(refs.module, "process_images", boom)

        run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None)],
        )

        assert refs.engine.ref_latents == []

    def test_the_imagestitch_memo_is_invalidated(self, refs, host, image_factory):
        """Otherwise ImageStitch skips an encode Stage 1 needed next time."""
        p = make_p(host)
        attach_stitch(p, [image_factory(512, 512)])
        assert FakeStitch.cached_parameters is not None

        run_chain(refs, host, p, make_processed(host, p, image_factory))

        assert FakeStitch.cached_parameters is None

    def test_the_tab_runners_are_reached_when_the_generation_has_no_script(self, host):
        stitch = FakeStitch()
        FakeStitch.cached_parameters = ["stale"]
        host.scripts.scripts_txt2img.alwayson_scripts.append(stitch)

        mc_references.invalidate_stitch_cache(None)

        assert FakeStitch.cached_parameters is None


# --------------------------------------------------------------------------- #
# Infotext and restoration (acceptance: Restoration)
# --------------------------------------------------------------------------- #


class TestInfotext:
    def base(self, **overrides):
        params = dict(
            target="fluxKlein9B.safetensors", prompt_mode="Replace", prompt="a lake",
            negative="", styles=[], seed_mode="Inherit", seed_offset=0, fixed_seed=-1,
            cfg=1.0, steps=8, sampler=mc_infotext.INHERIT, scheduler=mc_infotext.INHERIT,
            denoise=1.0, size_multiplier=1.0, stage1_size="1024x1024",
        )
        params.update(overrides)
        return mc_infotext.build_params(**params)

    def test_the_default_mode_is_omitted(self):
        assert mc_infotext.REFERENCE_MODE not in self.base()
        assert mc_infotext.REFERENCE_MODE not in self.base(reference_mode=mc_references.DISABLED)

    def test_an_active_mode_is_recorded(self):
        params = self.base(reference_mode=mc_references.PASS_THROUGH, reference_count=3)
        assert params[mc_infotext.REFERENCE_MODE] == mc_references.PASS_THROUGH
        assert params[mc_infotext.REFERENCE_COUNT] == 3

    def test_a_zero_count_is_omitted(self):
        params = self.base(reference_mode=mc_references.PASS_THROUGH, reference_count=0)
        assert mc_infotext.REFERENCE_COUNT not in params

    def test_the_maximum_side_is_recorded_for_decoupled_only(self):
        decoupled = self.base(
            reference_mode=mc_references.DECOUPLED, reference_count=1, reference_max_dim=768
        )
        assert decoupled[mc_infotext.REFERENCE_MAX_DIM] == 768

        passed = self.base(
            reference_mode=mc_references.PASS_THROUGH, reference_count=1, reference_max_dim=768
        )
        assert mc_infotext.REFERENCE_MAX_DIM not in passed

    def test_no_reference_pixels_are_recorded(self):
        """Native ImageStitch stores none either; the count is diagnostic only."""
        params = self.base(reference_mode=mc_references.DECOUPLED, reference_count=2)
        for value in params.values():
            assert not hasattr(value, "size")

    def test_the_keys_are_declared_for_pasting(self):
        names = mc_infotext.paste_field_names()
        assert mc_infotext.REFERENCE_MODE in names
        assert mc_infotext.REFERENCE_COUNT in names
        assert mc_infotext.REFERENCE_MAX_DIM in names

    def test_the_recorded_count_is_what_stage_2_was_given(self, refs, host, image_factory):
        p, _ = run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.DECOUPLED,
            reference_images=[(image_factory(512, 512), None) for _ in range(2)],
        )

        assert p.extra_generation_params[mc_infotext.REFERENCE_MODE] == mc_references.DECOUPLED
        assert p.extra_generation_params[mc_infotext.REFERENCE_COUNT] == 2

    def test_a_dropped_set_is_recorded_as_zero(self, refs, host, image_factory):
        p, _ = run(
            refs,
            host,
            image_factory,
            reference_mode=mc_references.PASS_THROUGH,
            reference_images=[],
        )

        assert p.extra_generation_params[mc_infotext.REFERENCE_COUNT] == 0


class TestRestoration:
    def fields(self, chain):
        chain.script.ui(is_img2img=False)
        return {field.api: field for field in chain.script.infotext_fields}

    def test_the_mode_is_registered_for_pasting(self, chain):
        assert "model_chain_reference_mode" in self.fields(chain)

    def test_an_absent_key_restores_the_default(self, chain):
        field = self.fields(chain)["model_chain_reference_mode"]
        assert field.function({}) == mc_references.DISABLED

    def test_a_recorded_mode_is_restored(self, chain):
        field = self.fields(chain)["model_chain_reference_mode"]
        params = {mc_infotext.REFERENCE_MODE: mc_references.DECOUPLED}
        assert field.function(params) == mc_references.DECOUPLED

    def test_a_restored_mode_without_images_is_a_notice_not_a_crash(
        self, refs, host, image_factory
    ):
        """Acceptance: the mode restores, the images do not, and Stage 2 says so."""
        _, processed = run(
            refs, host, image_factory, reference_mode=mc_references.DECOUPLED, reference_images=None
        )

        assert refs.seen == [[]]
        assert "will run with only its own Stage 1 image" in processed.comments
        assert len(refs.refine_calls) == 1


# --------------------------------------------------------------------------- #
# Reading the ImageStitch gallery
# --------------------------------------------------------------------------- #


class TestStitchArguments:
    def test_the_selection_is_read_from_the_script_arguments(self, host, image_factory):
        p = make_p(host)
        images = [image_factory(512, 512) for _ in range(2)]
        attach_stitch(p, images, enabled=True, max_dim=768)

        enabled, gallery, max_dim = mc_references.stitch_arguments(p)

        assert enabled is True
        assert max_dim == 768
        assert mc_references.extract_images(gallery) == images

    def test_a_missing_script_reads_as_nothing(self, host):
        assert mc_references.stitch_arguments(make_p(host)) is None

    def test_the_hosts_empty_dict_default_reads_as_nothing(self, host, image_factory):
        """StableDiffusionProcessing defaults script_args to {}, not []."""
        p = make_p(host)
        attach_stitch(p, [image_factory(512, 512)])
        p.script_args = {}

        assert mc_references.stitch_arguments(p) is None

    def test_an_unmapped_argument_span_reads_as_nothing(self, host, image_factory):
        p = make_p(host)
        stitch, _ = attach_stitch(p, [image_factory(512, 512)])
        stitch.args_from = None

        assert mc_references.stitch_arguments(p) is None

    def test_a_non_numeric_maximum_side_falls_back_to_the_default(self, host, image_factory):
        p = make_p(host)
        attach_stitch(p, [image_factory(512, 512)], max_dim="wide")

        _, _, max_dim = mc_references.stitch_arguments(p)

        assert max_dim == mc_references.DEFAULT_MAX_DIM


class TestExtractImages:
    def test_order_is_preserved(self, image_factory):
        images = [image_factory(64, 64, (i, 0, 0)) for i in range(4)]
        assert mc_references.extract_images([(i, None) for i in images]) == images

    def test_bare_images_are_accepted(self, image_factory):
        images = [image_factory(64, 64)]
        assert mc_references.extract_images(images) == images

    def test_dictionary_entries_are_accepted(self, image_factory):
        image = image_factory(64, 64)
        assert mc_references.extract_images([{"image": image}]) == [image]

    def test_an_empty_gallery_is_empty(self):
        assert mc_references.extract_images(None) == []
        assert mc_references.extract_images([]) == []

    def test_unreadable_entries_are_skipped_rather_than_fatal(self, image_factory):
        image = image_factory(64, 64)
        assert mc_references.extract_images([None, image, 42]) == [image]

    def test_a_malformed_gallery_is_empty_rather_than_fatal(self):
        for value in ("not a gallery", {"a": 1}, 42, object()):
            assert mc_references.extract_images(value) == []

    def test_a_file_path_entry_is_loaded(self, tmp_path, image_factory):
        path = tmp_path / "reference.png"
        image_factory(64, 64).save(path)

        loaded = mc_references.extract_images([str(path)])

        assert len(loaded) == 1
        assert loaded[0].size == (64, 64)


class TestPreprocess:
    def test_the_longest_side_is_capped(self, image_factory):
        result = mc_references.preprocess(image_factory(2048, 1024), 1024)
        assert result.size == (1024, 512)

    def test_dimensions_are_snapped_to_the_forge_grid(self, image_factory):
        result = mc_references.preprocess(image_factory(500, 300), 0)
        assert result.size == (512, 320)

    def test_a_zero_limit_means_no_cap(self, image_factory):
        result = mc_references.preprocess(image_factory(2048, 1024), 0)
        assert result.size == (2048, 1024)

    def test_an_already_conforming_image_is_returned_untouched(self, image_factory):
        image = image_factory(1024, 512)
        assert mc_references.preprocess(image, 1024) is image

    def test_a_tiny_image_never_snaps_to_nothing(self, image_factory):
        result = mc_references.preprocess(image_factory(20, 20), 0)
        assert result.size == (64, 64)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


class TestUI:
    def test_the_mode_control_offers_all_three_modes(self, chain):
        chain.script.ui(is_img2img=False)
        assert mc_references.MODES == (
            mc_references.DISABLED,
            mc_references.PASS_THROUGH,
            mc_references.DECOUPLED,
        )

    def test_the_gallery_is_returned_as_a_control(self, chain):
        returned = chain.script.ui(is_img2img=False)
        ids = [c.elem_id.removeprefix("script_modelchain_") for c in returned]
        assert ids[-3:] == ["reference_mode", "reference_images", "reference_max_dim"]

    def test_presets_carry_the_mode_but_not_the_images(self):
        assert "reference_mode" in mc_presets.FIELDS
        assert "reference_max_dim" in mc_presets.FIELDS
        assert "reference_images" not in mc_presets.FIELDS

    def test_a_disabled_mode_says_nothing(self, chain):
        from model_chain import _reference_notice

        assert _reference_notice(mc_references.DISABLED, "klein.safetensors") == ""

    def test_pass_through_counts_the_imagestitch_gallery(self, chain, image_factory, monkeypatch):
        from model_chain import _reference_notice

        set_architecture(monkeypatch, "flux2_9b")
        gallery = [(image_factory(64, 64), None) for _ in range(3)]

        note = _reference_notice(mc_references.PASS_THROUGH, "klein.safetensors", gallery)

        assert "**3** ImageStitch reference image(s)" in note

    def test_pass_through_says_so_when_imagestitch_is_not_installed(self, chain, monkeypatch):
        from model_chain import _reference_notice

        set_architecture(monkeypatch, "flux2_9b")

        note = _reference_notice(mc_references.PASS_THROUGH, "klein.safetensors")

        assert "not installed" in note
        assert "Decoupled" in note

    def test_pass_through_defers_to_the_generation_when_the_gallery_is_unreadable(
        self, chain, host, monkeypatch
    ):
        """ImageStitch is there, but its gallery never reached after_component."""
        host.scripts.scripts_txt2img.alwayson_scripts.append(FakeStitch())
        set_architecture(monkeypatch, "flux2_9b")

        from model_chain import _reference_notice

        note = _reference_notice(mc_references.PASS_THROUGH, "klein.safetensors")

        assert "whatever reference images ImageStitch holds" in note

    def test_pass_through_warns_when_the_gallery_is_empty(self, chain, monkeypatch):
        from model_chain import _reference_notice

        set_architecture(monkeypatch, "flux2_9b")

        note = _reference_notice(mc_references.PASS_THROUGH, "klein.safetensors", [])

        assert "no reference images" in note

    def test_an_unsupported_target_shows_a_compatibility_notice(
        self, chain, image_factory, monkeypatch
    ):
        from model_chain import _reference_notice

        set_architecture(monkeypatch, "sdxl")

        note = _reference_notice(
            mc_references.DECOUPLED, "sdxl.safetensors", None, [(image_factory(64, 64), None)]
        )

        assert "no reference-conditioning path" in note

    def test_an_experimental_target_is_marked_as_such(self, chain, image_factory, monkeypatch):
        from model_chain import _reference_notice

        set_architecture(monkeypatch, "anima")

        note = _reference_notice(
            mc_references.DECOUPLED, "anima.safetensors", None, [(image_factory(64, 64), None)]
        )

        assert "experimental" in note

    def test_an_undetected_target_is_checked_again_at_load(self, chain, image_factory, monkeypatch):
        from model_chain import _reference_notice

        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.UNKNOWN)

        note = _reference_notice(
            mc_references.DECOUPLED, "mystery.gguf", None, [(image_factory(64, 64), None)]
        )

        assert "could not be detected" in note

    def test_the_ui_argument_list_matches_the_documented_order(self):
        assert UI_ORDER[-3:] == ("reference_mode", "reference_images", "reference_max_dim")


class TestStatusWiring:
    """ImageStitch's gallery does not exist yet when this panel is built.

    It sorts below Model Chain, and a Gradio event captures its input list at
    registration -- so a status wired too early could never see the gallery it
    is supposed to count.
    """

    STITCH_GALLERY_ID = "script_txt2img_imagestitch_integrated_ref_latent"

    def changes(self, component):
        return [kwargs for name, kwargs in component._callbacks if name == "change"]

    def mode_control(self, returned):
        return returned[UI_ORDER.index("reference_mode")]

    def test_it_wires_immediately_when_imagestitch_is_absent(self, chain, host):
        returned = chain.script.ui(is_img2img=False)

        wired = self.changes(self.mode_control(returned))

        assert len(wired) == 1
        assert len(wired[0]["inputs"]) == 3

    def test_it_waits_for_the_gallery_when_imagestitch_is_installed(self, chain, host):
        host.scripts.scripts_txt2img.alwayson_scripts.append(FakeStitch())

        returned = chain.script.ui(is_img2img=False)

        assert self.changes(self.mode_control(returned)) == []

    def test_the_gallery_completes_the_wiring_when_it_arrives(self, chain, host):
        host.scripts.scripts_txt2img.alwayson_scripts.append(FakeStitch())
        returned = chain.script.ui(is_img2img=False)

        gallery = types.SimpleNamespace(_callbacks=[], change=lambda **kw: None)
        gallery.change = lambda **kw: gallery._callbacks.append(("change", kw))
        chain.script.after_component(gallery, elem_id=self.STITCH_GALLERY_ID)

        wired = self.changes(self.mode_control(returned))
        assert len(wired) == 1
        assert wired[0]["inputs"][-1] is gallery

    def test_a_second_gallery_cannot_wire_the_same_handlers_twice(self, chain, host):
        host.scripts.scripts_txt2img.alwayson_scripts.append(FakeStitch())
        returned = chain.script.ui(is_img2img=False)

        for _ in range(2):
            gallery = types.SimpleNamespace()
            gallery._callbacks = []
            gallery.change = lambda **kw: None
            chain.script.after_component(gallery, elem_id=self.STITCH_GALLERY_ID)

        assert len(self.changes(self.mode_control(returned))) == 1

    def test_an_unrelated_component_does_not_wire_anything(self, chain, host):
        host.scripts.scripts_txt2img.alwayson_scripts.append(FakeStitch())
        returned = chain.script.ui(is_img2img=False)

        chain.script.after_component(object(), elem_id="txt2img_prompt")

        assert self.changes(self.mode_control(returned)) == []
