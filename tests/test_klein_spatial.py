"""Spatial Layout on FLUX.2 Klein: the matrix, the resolver, and the unwinding.

The design intent puts a spike in front of the sampling mechanism and nothing in
front of anything else, and that division is the one this file is written
around. What a card decides -- whether a regional condition actually moves an
object -- is measured by ``tools/klein_regional_spike.py`` and cannot be
asserted here. What *code* decides is all here, and it is the larger half:

* which modes are offered, and when (§4, §26);
* what Auto resolves to, and what it is never allowed to resolve to (§5);
* what happens when an explicitly chosen mode loses its source (§10);
* which images become references, in which order, and which one is deliberately
  not registered twice (§17, §45);
* that a normalized box lands on the grid the tensor actually has (§29);
* that nothing survives the job that installed it (§18, §41, §46).

Every one of those is a silent failure in production. A mode that resolves
wrongly, a reference registered twice, a box compiled against the wrong grid and
a reference set left resident all produce a perfectly good image that is not the
one that was asked for, and none of them raises anything.

The seventh thing this file checks is that Krea 2 did not move. Two backends
share one canvas, one editor and one serialized document, and the way that goes
wrong is not with an exception -- it is with a Krea generation quietly acquiring
a Klein behaviour because a shared function grew a branch.
"""

from __future__ import annotations

import json
import types

import pytest

import mc_arch
import mc_infotext
import mc_references
import mc_spatial
import mc_spatial_klein
from prompt_master import spatial as generic
from prompt_master.krea import spatial as document


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


class FakeStitch:
    """ImageStitch Integrated, as far as this feature is allowed to see it."""

    cached_parameters = None
    args_from = 1
    args_to = 4

    def title(self):
        return mc_references.STITCH_TITLE


class FakeEngine:
    """A reference-capable engine, with an encoder that answers predictably."""

    def __init__(self, config=None):
        self.ref_latents = []
        self.ini_latent = None
        self.encoded = []
        if config is not None:
            self.model_config = config

    def get_learned_conditioning(self, prompts):
        self.encoded.extend(prompts)
        return [[f"cond({prompts[0]})", {}]]


def make_image(colour):
    """A real PIL image, because ``extract_images`` deliberately only takes those.

    A stub with the right attributes would pass through the extractor's liberal
    entry handling and prove nothing: the extractor's whole job is to be strict
    about what counts as an image and forgiving about how it was wrapped.
    """
    from PIL import Image

    return Image.new("RGB", (64, 64), colour)


def make_p(images=(), *, enabled=True, max_dim=1024, width=1024, height=1024,
           installed=True, stitch=None):
    """A processing object carrying an ImageStitch gallery, or carrying none."""
    p = types.SimpleNamespace(width=width, height=height,
                              extra_generation_params={}, override_settings={})
    if not installed:
        p.scripts = types.SimpleNamespace(alwayson_scripts=[])
        p.script_args = []
        return p
    p.scripts = types.SimpleNamespace(alwayson_scripts=[stitch or FakeStitch()])
    p.script_args = [None, enabled, [(image, None) for image in images], max_dim]
    return p


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Preferences in a temporary directory, so one test cannot steer the next."""
    import mc_llm_paths

    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def installed_stitch(host):
    """ImageStitch present in the tab's script runner, as the panel would see it."""
    stitch = FakeStitch()
    host.scripts.scripts_txt2img.alwayson_scripts.append(stitch)
    yield stitch
    FakeStitch.cached_parameters = None


def klein(monkeypatch, key="flux2_9b"):
    """Make every architecture question answer ``key``."""
    monkeypatch.setattr(mc_arch, "detect_loaded_engine", lambda: mc_arch.by_key(key))
    monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name",
                        lambda name: mc_arch.by_key(key))


REGION = {"id": "lamp", "name": "Lamp", "type": "obj", "bbox": [680, 150, 910, 820],
          "prompt": "tall brass floor lamp", "framing": "", "angle": "", "z": 0}

SECOND = {"id": "sofa", "name": "Sofa", "type": "obj", "bbox": [40, 500, 460, 900],
          "prompt": "navy blue sofa", "z": 1}


def layout(regions=(REGION,), width=1024, height=1024) -> str:
    return json.dumps({"version": 1,
                       "canvas": {"width": width, "height": height, "grid": "thirds"},
                       "compose_mode": "smart", "auto_position_hint": True,
                       "regions": list(regions)})


IMAGES = (make_image((255, 0, 0)), make_image((0, 255, 0)), make_image((0, 0, 255)))


# --------------------------------------------------------------------------- #
# §43 -- the mode-resolution table
# --------------------------------------------------------------------------- #


class TestWhichModesAreOffered:
    """§4 and §26. The matrix drives five separate things, so it is one tuple."""

    def test_without_a_source_only_two_modes_are_selectable(self):
        available = generic.available_modes(has_source=False)

        assert available == (generic.AUTO, generic.REGIONAL_GENERATE)
        for mode in generic.IMAGE_REQUIRED_MODES:
            assert mode not in available

    def test_with_a_source_every_mode_is_selectable(self):
        assert generic.available_modes(has_source=True) == generic.MODES

    def test_the_unavailable_ones_stay_visible_with_a_reason(self):
        """§4's last line. A mode that vanished would look like it was removed."""
        import model_chain_krea_creative as creative_script

        choices = creative_script._klein_choices(has_source=False)
        offered = [value for _label, value in choices]

        assert offered == list(generic.MODES)
        for label, value in choices:
            if value in generic.IMAGE_REQUIRED_MODES:
                assert "ImageStitch" in label
            else:
                assert "ImageStitch" not in label


class TestSourceA_NoImageStitchScriptAtAll:
    """§43 A. Nothing to read: Auto generates, and only two modes are offered."""

    def test_the_source_is_not_usable(self, host, monkeypatch):
        klein(monkeypatch)
        source = mc_spatial_klein.source_for(make_p(installed=False))

        assert source.usable is False
        assert source.image_count == 0

    def test_auto_resolves_to_regional_generate(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(make_p(installed=False), layout(),
                                               enabled=True)

        assert request.resolved_mode == generic.REGIONAL_GENERATE
        assert request.has_source is False


class TestSourceB_ImageStitchDisabledWithAnImageInIt:
    """§43 B. A full gallery behind a cleared checkbox is not a source."""

    def test_the_source_is_not_usable(self, host, monkeypatch):
        klein(monkeypatch)
        source = mc_spatial_klein.source_for(make_p(IMAGES[:1], enabled=False))

        assert source.usable is False

    def test_auto_still_resolves_to_regional_generate(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(
            make_p(IMAGES[:1], enabled=False), layout(), enabled=True)

        assert request.resolved_mode == generic.REGIONAL_GENERATE

    def test_the_image_required_modes_are_unavailable(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(
            make_p(IMAGES[:1], enabled=False), layout(), enabled=True)

        for mode in generic.IMAGE_REQUIRED_MODES:
            assert not generic.is_available(mode, request.has_source)


class TestSourceC_ImageStitchEnabledAndEmpty:
    """§43 C. The same answer as B, reached from the other direction."""

    def test_it_is_the_same_as_a_disabled_one(self, host, monkeypatch):
        klein(monkeypatch)
        empty = mc_spatial_klein.source_for(make_p((), enabled=True))
        disabled = mc_spatial_klein.source_for(make_p(IMAGES[:1], enabled=False))

        assert empty.usable is disabled.usable is False

    def test_auto_resolves_to_regional_generate(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(make_p((), enabled=True), layout(),
                                               enabled=True)

        assert request.resolved_mode == generic.REGIONAL_GENERATE


class TestSourceD_OneValidImage:
    """§43 D. Everything opens, and Auto takes the reference path."""

    def test_every_mode_becomes_available(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(make_p(IMAGES[:1]), layout(),
                                               enabled=True)

        assert request.has_source is True
        for mode in generic.MODES:
            assert generic.is_available(mode, request.has_source)

    def test_auto_resolves_to_reference_plus_regions(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(make_p(IMAGES[:1]), layout(),
                                               enabled=True)

        assert request.resolved_mode == generic.REFERENCE_REGIONS

    def test_auto_never_picks_a_source_preserving_mode(self):
        """§5's last line, asserted for every reachable combination.

        Regional Img2Img and Strict Regional Edit change how strongly the source
        survives, which is a decision about the picture rather than about what is
        available. A resolver that made it would be choosing how much of
        somebody's image to keep on their behalf.
        """
        for has_source in (True, False):
            assert generic.resolve(generic.AUTO, has_source) \
                not in generic.SOURCE_PRESERVING_MODES


class TestSourceE_MalformedGalleryEntriesOnly:
    """§43 E. Entries nothing can resolve to an image are not images."""

    def test_a_gallery_of_junk_is_not_a_source(self, host, monkeypatch):
        klein(monkeypatch)
        p = make_p()
        p.script_args = [None, True, [None, {}, ("", None)], 1024]

        assert mc_spatial_klein.source_for(p).usable is False

    def test_a_bare_string_gallery_is_not_a_one_image_gallery(self, host, monkeypatch):
        klein(monkeypatch)
        p = make_p()
        p.script_args = [None, True, "not-a-gallery", 1024]

        assert mc_spatial_klein.source_for(p).usable is False


class TestSourceF_AnExplicitModeLosesItsSource:
    """§43 F, and the sharpest line in the whole design.

    Auto is allowed to adapt. An explicit choice is not. Somebody who picked
    Strict Regional Edit picked it because they want their source preserved, and
    turning that into a fresh txt2img because a gallery emptied between the
    click and the press is not a fallback -- it is a different picture from the
    same button.
    """

    @pytest.mark.parametrize("mode", generic.IMAGE_REQUIRED_MODES)
    def test_it_fails_rather_than_falling_back(self, host, monkeypatch, mode):
        klein(monkeypatch)

        with pytest.raises(generic.ModeUnavailable):
            mc_spatial_klein.request_for(make_p(()), layout(), enabled=True,
                                         requested_mode=mode)

    def test_the_message_names_the_mode_that_was_chosen(self, host, monkeypatch):
        klein(monkeypatch)

        with pytest.raises(generic.ModeUnavailable) as raised:
            mc_spatial_klein.request_for(make_p(()), layout(), enabled=True,
                                         requested_mode=generic.STRICT_REGIONAL_EDIT)

        assert "Strict Regional Edit" in str(raised.value)
        assert "ImageStitch" in str(raised.value)

    def test_the_hook_refuses_the_generation_before_anything_samples(self, host,
                                                                    monkeypatch,
                                                                    store):
        import model_chain_krea_creative as creative_script

        klein(monkeypatch)
        mc_spatial.remember(**{mc_spatial.ENABLED: True,
                               mc_spatial.LAYOUT: layout(),
                               mc_spatial.KLEIN_MODE: generic.REGIONAL_IMG2IMG})
        script = creative_script.ScriptKreaCreative()
        p = make_p(())
        p.prompt = "a living room"
        p.negative_prompt = ""

        with pytest.raises(RuntimeError, match="Regional Img2Img"):
            script.before_process(p, False)

        # And the prompt was not touched on the way out.
        assert p.prompt == "a living room"


class TestSourceG_RegionalGenerateIgnoresAFullGallery:
    """§6 and §43 G. An unused image must not change what a mode means."""

    def test_no_references_are_taken_from_it(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(
            make_p(IMAGES), layout(), enabled=True,
            requested_mode=generic.REGIONAL_GENERATE)

        assert request.resolved_mode == generic.REGIONAL_GENERATE
        assert request.references == ()
        assert request.uses_source is False

    def test_the_source_is_still_reported_as_present(self, host, monkeypatch):
        """Present and unused are different facts, and the file records both."""
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(
            make_p(IMAGES), layout(), enabled=True,
            requested_mode=generic.REGIONAL_GENERATE)
        recorded = mc_spatial_klein.metadata(request, layout_serialized=layout())

        assert request.has_source is True
        assert recorded[mc_infotext.KLEIN_SPATIAL_SOURCE] == "None"

    def test_the_reference_toggle_is_turned_off_for_the_generation(self, host,
                                                                    monkeypatch):
        """§6, enforced through Forge's own switch rather than through an encode.

        ImageStitch registers its own references for Stage 1, so "ignore
        ImageStitch" cannot mean "do not encode them" -- that decision is not
        this extension's to make. What it can mean, exactly, is that the engine
        does not keep them, which is what ``klein_no_reference`` says.
        """
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(
            make_p(IMAGES), layout(), enabled=True,
            requested_mode=generic.REGIONAL_GENERATE)
        override = mc_spatial_klein.reference_override(request,
                                                       mc_arch.by_key("flux2_9b"))

        assert override == {"klein_no_reference": True}

    def test_this_module_never_encodes_a_reference_itself(self):
        """The double-registration §17 warns about, refused structurally.

        ``ref_latents`` is appended to rather than replaced, so an encode here
        beside ImageStitch's own would register every image twice -- and which of
        the two ran first would depend on the order Forge happens to run two
        always-on scripts in.
        """
        import inspect

        source = inspect.getsource(mc_spatial_klein)

        assert "mc_references.encode(" not in source


# --------------------------------------------------------------------------- #
# §44 -- the layout document
# --------------------------------------------------------------------------- #


class TestTheLayoutDocumentIsShared:
    """One editor, one document. §44, and the regression half of §51."""

    def test_the_normalized_contract_is_unchanged(self):
        assert generic.SCALE == document.SCALE == 1000

    def test_reversed_corners_are_ordered(self):
        request = generic.request_from(
            layout([dict(REGION, bbox=[910, 820, 680, 150])]), enabled=True)

        assert request.regions[0].bbox_norm == (680, 150, 910, 820)

    def test_out_of_range_values_clamp(self):
        request = generic.request_from(
            layout([dict(REGION, bbox=[-40, -10, 5000, 3000])]), enabled=True)

        assert request.regions[0].bbox_norm == (0, 0, 1000, 1000)

    def test_zero_area_boxes_are_discarded(self):
        request = generic.request_from(
            layout([dict(REGION, bbox=[500, 500, 500, 900])]), enabled=True)

        assert request.regions == ()

    def test_a_version_1_layout_still_loads(self):
        request = generic.request_from(layout(), enabled=True)

        assert request.unreadable is False
        assert [region.identifier for region in request.regions] == ["lamp"]

    def test_a_missing_strength_defaults_to_one(self):
        """§12. Every layout ever saved is a layout with no strength in it."""
        request = generic.request_from(layout(), enabled=True)

        assert request.regions[0].strength == 1.0

    def test_a_recorded_strength_is_read_and_clamped(self):
        request = generic.request_from(
            layout([dict(REGION, strength=7.5)]), enabled=True)

        assert request.regions[0].strength == generic.MAX_STRENGTH

    def test_a_default_strength_is_not_written_back(self):
        """A layout drawn before this build round-trips byte-identical.

        §51: existing version-1 editor documents unchanged. The field exists,
        defaults correctly and stays out of the serialization until something
        actually sets it, so the browser's normalizer has nothing new to drop.
        """
        source = json.dumps(json.loads(layout()), separators=(",", ":"),
                            ensure_ascii=False)
        parsed = document.parse(source)

        assert parsed.serialize() == source
        assert "strength" not in parsed.serialize()

    def test_region_order_and_z_survive_the_round_trip(self):
        request = generic.request_from(layout([SECOND, REGION]), enabled=True)

        assert [region.identifier for region in request.regions] == ["lamp", "sofa"]
        assert [region.z for region in request.regions] == [0, 1]

    def test_changing_the_output_resolution_does_not_move_a_box(self):
        """§15. Normalized coordinates are the source of truth."""
        drawn = generic.request_from(layout(), enabled=True, width=1024, height=1024)
        again = generic.request_from(layout(), enabled=True, width=1536, height=640)

        assert drawn.regions[0].bbox_norm == again.regions[0].bbox_norm

    def test_a_layout_from_a_later_build_is_refused_rather_than_half_read(self):
        request = generic.request_from(
            json.dumps({"version": 99, "regions": [REGION]}), enabled=True)

        assert request.unreadable is True
        assert request.regions == ()


class TestFramingAndAngleStayOutOfTheGeometry:
    """§12's last line, which is easy to get wrong in the helpful direction."""

    def test_they_join_the_text_and_not_the_box(self):
        request = generic.request_from(
            layout([dict(REGION, framing="Close-up", angle="3/4 left")]),
            enabled=True)
        region = request.regions[0]

        assert region.bbox_norm == (680, 150, 910, 820)
        assert region.qualified_prompt() == (
            "tall brass floor lamp, shown in a close-up view, "
            "in a three-quarter view from the left")

    def test_the_position_hint_is_deliberately_not_appended(self):
        """Telling a region in words where it already is conditions the rest of
        the frame on it too, which is the leak §31 is about."""
        request = generic.request_from(layout(), enabled=True)

        assert "positioned in" not in request.regions[0].qualified_prompt()
        assert "occupying" not in request.regions[0].qualified_prompt()


# --------------------------------------------------------------------------- #
# §29 -- normalized boxes on a real grid
# --------------------------------------------------------------------------- #


class TestTheCoordinateCompiler:
    """The arithmetic between a rectangle on a canvas and a rectangle of tokens."""

    def test_the_worked_example_from_the_design_intent(self):
        assert generic.to_grid((0.680, 0.150, 0.910, 0.820), 100, 100) == \
            (68, 15, 91, 82)

    def test_the_near_corner_floors_and_the_far_corner_ceils(self):
        """A box covers at least the cells it visually touches.

        Rounding inwards on both sides loses a row at small latent sizes, and it
        loses the largest share from the smallest boxes -- the ones with the
        least to spare.
        """
        assert generic.to_grid((0.11, 0.11, 0.19, 0.19), 10, 10) == (1, 1, 2, 2)

    def test_a_box_always_keeps_at_least_one_cell(self):
        assert generic.to_grid((0.5, 0.5, 0.5001, 0.5001), 8, 8) is not None

    def test_the_grid_comes_from_the_tensor_and_not_from_the_canvas(self):
        """§15 and §29 both refuse a hard-coded VAE ratio. The measurement is
        the last two dimensions of whatever is about to be sampled."""
        tensor = types.SimpleNamespace(shape=(1, 16, 40, 64))

        assert mc_spatial_klein.latent_grid(tensor) == (64, 40)

    def test_two_pass_sizes_compile_the_same_box_differently(self):
        request = generic.request_from(layout(), enabled=True)
        first = mc_spatial_klein.compile_regions(request, grid=(64, 64))
        second = mc_spatial_klein.compile_regions(request, grid=(128, 128))

        assert first.pairs[0][1] != second.pairs[0][1]
        assert first.grid == (64, 64) and second.grid == (128, 128)

    def test_a_region_too_small_for_the_grid_is_named_rather_than_widened(self):
        request = generic.request_from(
            layout([dict(REGION, bbox=[500, 500, 501, 900])]), enabled=True)
        compiled = mc_spatial_klein.compile_regions(request, grid=(4, 4))

        assert len(compiled) == 1 or compiled.notes

    def test_compiling_without_a_shape_is_refused(self):
        request = generic.request_from(layout(), enabled=True)

        with pytest.raises(ValueError, match="shape"):
            mc_spatial_klein.compile_regions(request)


# --------------------------------------------------------------------------- #
# §17 / §45 -- ordered images
# --------------------------------------------------------------------------- #


class TestWhichImagesBecomeReferences:
    """Order is meaning here, and image 1 means something different per mode."""

    def source(self):
        return generic.SpatialSource(images=IMAGES, enabled=True)

    def test_reference_plus_regions_takes_every_image_in_order(self):
        assert self.source().references_for(generic.REFERENCE_REGIONS) == IMAGES

    def test_regional_generate_takes_none_of_them(self):
        assert self.source().references_for(generic.REGIONAL_GENERATE) == ()

    @pytest.mark.parametrize("mode", generic.SOURCE_PRESERVING_MODES)
    def test_image_one_is_not_registered_twice(self, mode):
        """§17's IMPORTANT, and §45's third acceptance line.

        In the source-preserving modes image 1 becomes the init latent, and Klein
        inserts ``ini_latent`` ahead of ``ref_latents`` itself. Registering it
        here as well would hand the model the same picture as reference 1 and
        reference 2.
        """
        references = self.source().references_for(mode)

        assert references == IMAGES[1:]
        assert IMAGES[0] not in references

    def test_the_primary_and_the_supplemental_set_do_not_overlap(self):
        source = self.source()

        assert source.primary is IMAGES[0]
        assert source.supplemental == IMAGES[1:]

    def test_an_unusable_source_offers_nothing_to_any_mode(self):
        source = generic.SpatialSource(images=IMAGES, enabled=False)

        for mode in generic.MODES:
            assert source.references_for(mode) == ()


# --------------------------------------------------------------------------- #
# §18 / §41 / §46 -- lifecycle
# --------------------------------------------------------------------------- #


class TestNothingSurvivesTheJob:
    """A failed generation must leave the next ordinary one ordinary."""

    @pytest.fixture
    def engine(self, host, monkeypatch):
        klein(monkeypatch)
        model = FakeEngine()
        host.shared.sd_model = model
        monkeypatch.setattr(mc_references, "_to_tensor", lambda image: image)
        return model

    def test_pre_existing_references_are_cleared_before_the_job(self, engine):
        """§18, and the half of it that is easy to leave out.

        Clearing only afterwards would leave the previous job's references
        resident for the whole of this one -- which is exactly the state that
        makes an empty gallery look source-capable.
        """
        engine.ref_latents.extend(["from an earlier generation"])

        with mc_spatial_klein.reference_scope():
            assert engine.ref_latents == []

    def test_references_are_cleared_after_success(self, engine):
        with mc_spatial_klein.reference_scope():
            engine.ref_latents.extend(["encoded during the job"])

        assert engine.ref_latents == []

    def test_references_are_cleared_after_an_exception(self, engine):
        with pytest.raises(RuntimeError):
            with mc_spatial_klein.reference_scope():
                engine.ref_latents.extend(["encoded during the job"])
                raise RuntimeError("sampling failed")

        assert engine.ref_latents == []

    def test_the_referencing_flag_is_lowered_after_an_exception(self, engine):
        from backend.args import dynamic_args

        with pytest.raises(RuntimeError):
            with mc_spatial_klein.reference_scope():
                dynamic_args.is_referencing = True
                raise RuntimeError("sampling failed")

        assert dynamic_args.is_referencing is False

    def test_the_klein_reference_option_is_scoped_and_never_assigned(self, host,
                                                                     monkeypatch):
        """§22's last line. The toggle is global; leaving it flipped changes
        every later generation.

        So it is never assigned. It travels as a ``p.override_settings``
        fragment, which the host applies for one generation and restores itself
        -- the same mechanism Stage 2's edit mode already uses, and the reason
        this feature has no global setting of its own to leak.
        """
        import inspect

        klein(monkeypatch)
        host.shared.opts.klein_no_reference = True
        request = mc_spatial_klein.request_for(make_p(IMAGES[:1]), layout(),
                                               enabled=True)
        override = mc_spatial_klein.reference_override(request)

        assert override == {"klein_no_reference": False}
        assert host.shared.opts.klein_no_reference is True
        assert "opts.klein_no_reference" not in inspect.getsource(mc_spatial_klein)

    def test_the_hook_scopes_the_toggle_rather_than_setting_it(self, engine,
                                                               monkeypatch, store):
        import model_chain_krea_creative as creative_script

        script = creative_script.ScriptKreaCreative()
        mc_spatial.remember(**{mc_spatial.ENABLED: True, mc_spatial.LAYOUT: layout()})
        p = make_p(IMAGES[:1])
        p.prompt = "a living room"
        p.negative_prompt = ""
        script.before_process(p, False)

        assert p.override_settings == {"klein_no_reference": False}

    def test_the_imagestitch_memo_is_dropped_so_it_re_encodes(self, engine):
        """§18, and the same rule Model Chain's own clear already follows.

        ImageStitch memoises on its own inputs and returns early when they have
        not changed, because the references it encoded last time are still on the
        model. Clearing those without dropping the memo means it skips an encode
        it needed to do, and the next generation runs with no references at all.
        """
        stitch = FakeStitch()
        FakeStitch.cached_parameters = ["from an earlier generation"]
        p = make_p(IMAGES[:1], stitch=stitch)

        with mc_spatial_klein.reference_scope(p):
            pass

        assert FakeStitch.cached_parameters is None

    def test_the_regional_conditioning_is_removed_after_an_exception(self, engine):
        """§41. Whatever was appended is taken off again, however the block ends."""
        request = generic.request_from(layout([REGION, SECOND]), enabled=True)
        conditioning = [["the global prompt", {}]]

        with pytest.raises(RuntimeError):
            with mc_spatial_klein.regional_conditioning(
                    request, conditioning, grid=(64, 64),
                    backend=mc_spatial_klein.AreaConditioningBackend(),
                    model=engine):
                assert len(conditioning) == 3
                raise RuntimeError("sampling failed")

        assert conditioning == [["the global prompt", {}]]

    def test_an_ordinary_generation_after_a_failed_one_starts_clean(self, engine,
                                                                    monkeypatch,
                                                                    store):
        """§46's last line, asserted through the hook rather than the module.

        An exception during sampling never reaches ``postprocess``, so the
        guarantee cannot rest on it -- the release at the top of the next
        ``before_process`` is the one that actually holds.
        """
        import model_chain_krea_creative as creative_script

        script = creative_script.ScriptKreaCreative()
        mc_spatial.remember(**{mc_spatial.ENABLED: True, mc_spatial.LAYOUT: layout()})
        p = make_p(IMAGES[:1])
        p.prompt = "a living room"
        p.negative_prompt = ""
        script.before_process(p, False)
        assert script._klein is not None

        # ...and the job dies without postprocess ever running.
        second = make_p(())
        second.prompt = "something else"
        second.negative_prompt = ""
        monkeypatch.setattr(mc_arch, "detect_loaded_engine",
                            lambda: mc_arch.by_key("sdxl"))
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name",
                            lambda name: mc_arch.by_key("sdxl"))
        script.before_process(second, False)

        assert script._klein is None
        assert script._klein_job is None
        assert script._klein_pass is None


# --------------------------------------------------------------------------- #
# §34 -- an engine that cannot do this
# --------------------------------------------------------------------------- #


class TestAnEngineWithNoRegionalPath:
    """§28's forbidden shortcut, and §34's required refusal."""

    def test_no_backend_is_selected_when_the_host_offers_nothing(self, host,
                                                                 monkeypatch):
        monkeypatch.setattr(mc_spatial_klein, "_host_sampling_module", lambda: None)

        assert mc_spatial_klein.select_backend(FakeEngine()) is None
        assert mc_spatial_klein.supports_klein_regional_conditioning(FakeEngine()) \
            is False

    def test_the_compatibility_error_names_what_was_detected(self, host, monkeypatch):
        """§34. The interesting failure is a repacked build whose header said one
        thing and whose engine implements another, and a message that said only
        "not available" would send somebody to look at their layout."""
        klein(monkeypatch)
        message = mc_spatial_klein.compatibility_error(model=FakeEngine())

        assert "Flux.2 Klein 9B" in message
        assert "FakeEngine" in message

    def test_applying_regions_without_a_backend_raises(self, host, monkeypatch):
        monkeypatch.setattr(mc_spatial_klein, "_host_sampling_module", lambda: None)
        request = generic.request_from(layout(), enabled=True)

        with pytest.raises(mc_spatial_klein.RegionalConditioningUnavailable):
            with mc_spatial_klein.regional_conditioning(request, [["global", {}]],
                                                        grid=(64, 64)):
                pass

    def test_there_is_no_prompt_hint_fallback_anywhere_in_the_module(self):
        """§28, asserted against the source because that is where it would go.

        "A fallback natural-language position hint may be useful for debugging,
        but it must be labeled as degraded mode and not called regional
        conditioning." The honest version of that in a shipped build is not to
        have one: a failure that looks like a success is worse than a failure.
        """
        import inspect

        source = inspect.getsource(mc_spatial_klein)

        assert "position_hint" not in source
        assert "size_hint" not in source


# --------------------------------------------------------------------------- #
# §33 -- an enabled feature with nothing drawn
# --------------------------------------------------------------------------- #


class TestZeroRegions:
    """Spatial Layout on and no boxes is valid, and is an ordinary generation."""

    def test_regional_generate_degrades_to_ordinary_txt2img(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(make_p(()), layout(regions=()),
                                               enabled=True)
        conditioning = [["the global prompt", {}]]

        with mc_spatial_klein.regional_conditioning(
                request, conditioning, grid=(64, 64),
                backend=mc_spatial_klein.AreaConditioningBackend()) as compiled:
            assert len(compiled) == 0

        assert conditioning == [["the global prompt", {}]]

    def test_reference_plus_regions_still_registers_its_references(self, host,
                                                                   monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(make_p(IMAGES[:2]), layout(regions=()),
                                               enabled=True)

        assert request.resolved_mode == generic.REFERENCE_REGIONS
        assert request.references == IMAGES[:2]


# --------------------------------------------------------------------------- #
# §38 -- what a Klein spatial image records
# --------------------------------------------------------------------------- #


class TestTheInfotext:
    """Its own namespace, and no source pixels in it."""

    def test_it_records_the_requested_and_the_resolved_mode(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(make_p(IMAGES[:2]), layout(),
                                               enabled=True)
        recorded = mc_spatial_klein.metadata(request, layout_serialized=layout())

        assert recorded[mc_infotext.KLEIN_SPATIAL_MODE] == "Auto"
        assert recorded[mc_infotext.KLEIN_SPATIAL_RESOLVED] == "Reference + Regions"
        assert recorded[mc_infotext.KLEIN_SPATIAL_SOURCE] == "ImageStitch"
        assert recorded[mc_infotext.KLEIN_SPATIAL_SOURCE_COUNT] == 2

    def test_it_records_the_backend_that_produced_the_image(self, host, monkeypatch):
        """§38. Two mechanisms over one layout are two experiments, and a file
        that recorded only "regional conditioning happened" could not tell them
        apart afterwards."""
        klein(monkeypatch)
        backend = mc_spatial_klein.AreaConditioningBackend()
        request = mc_spatial_klein.request_for(make_p(()), layout(), enabled=True)
        recorded = mc_spatial_klein.metadata(request, backend,
                                             layout_serialized=layout())

        assert recorded[mc_infotext.KLEIN_SPATIAL_BACKEND] == backend.name
        assert recorded[mc_infotext.KLEIN_SPATIAL_BACKEND_VERSION] == backend.version

    def test_the_layout_is_recorded_because_the_prompt_cannot_carry_it(self, host,
                                                                       monkeypatch):
        """§39. Krea's structured prompt has its boxes in it; Klein's regional
        conditioning leaves no trace in the prompt at all."""
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(make_p(()), layout(), enabled=True)
        recorded = mc_spatial_klein.metadata(request, layout_serialized=layout())

        assert json.loads(recorded[mc_infotext.KLEIN_SPATIAL_LAYOUT])["regions"]

    def test_no_source_pixels_reach_the_file(self, host, monkeypatch):
        klein(monkeypatch)
        request = mc_spatial_klein.request_for(make_p(IMAGES), layout(), enabled=True)
        recorded = mc_spatial_klein.metadata(request, layout_serialized=layout())

        rendered = " ".join(str(value) for value in recorded.values())
        for image in IMAGES:
            assert repr(image) not in rendered
        # The count is an expectation. The images are not in the file.
        assert recorded[mc_infotext.KLEIN_SPATIAL_SOURCE_COUNT] == len(IMAGES)

    def test_the_two_namespaces_do_not_collide(self):
        """§37. An old Krea record must never be read as a Klein job."""
        assert not set(mc_infotext.SPATIAL_KEYS) & set(mc_infotext.KLEIN_SPATIAL_KEYS)
        for key in mc_infotext.KLEIN_SPATIAL_KEYS:
            assert key.startswith("Klein Spatial")

    def test_the_keys_are_forwarded_by_send_to_txt2img(self):
        forwarded = mc_infotext.creative_paste_field_names()

        for key in mc_infotext.KLEIN_SPATIAL_KEYS:
            assert key in forwarded


# --------------------------------------------------------------------------- #
# §51 -- Krea 2 did not move
# --------------------------------------------------------------------------- #


class TestKreaIsUnaffected:
    """Two backends, one canvas. The way this goes wrong is not an exception."""

    def test_a_krea_checkpoint_still_takes_the_krea_path(self, host, monkeypatch,
                                                        store):
        import model_chain_krea_creative as creative_script

        monkeypatch.setattr(mc_arch, "detect_loaded_engine",
                            lambda: mc_arch.by_key("krea2"))
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name",
                            lambda name: mc_arch.by_key("krea2"))
        script = creative_script.ScriptKreaCreative()
        mc_spatial.remember(**{mc_spatial.ENABLED: True, mc_spatial.LAYOUT: layout()})
        p = make_p(())
        p.prompt = "a living room"
        p.negative_prompt = ""
        script.before_process(p, False)

        assert script._klein is None
        # The Krea compositor ran: the prompt is now a structured document.
        assert p.prompt != "a living room"
        assert mc_infotext.SPATIAL_LAYOUT in p.extra_generation_params

    def test_a_klein_checkpoint_leaves_the_prompt_exactly_as_typed(self, host,
                                                                   monkeypatch,
                                                                   store):
        """The difference that matters. Klein's regional conditioning never
        rewrites the prompt, which is why §39 makes the recorded layout essential
        metadata rather than a convenience."""
        import model_chain_krea_creative as creative_script

        klein(monkeypatch)
        script = creative_script.ScriptKreaCreative()
        mc_spatial.remember(**{mc_spatial.ENABLED: True, mc_spatial.LAYOUT: layout()})
        p = make_p(())
        p.prompt = "a living room"
        p.negative_prompt = ""
        script.before_process(p, False)

        assert p.prompt == "a living room"
        assert mc_infotext.SPATIAL_LAYOUT not in p.extra_generation_params
        assert script._klein.resolved_mode == generic.REGIONAL_GENERATE

    def test_the_klein_modes_are_not_offered_for_other_architectures(self, host,
                                                                     monkeypatch):
        """§51's last section. SDXL, Flux.1 and Krea keep their own panel."""
        import model_chain_krea_creative as creative_script

        for key in ("sdxl", "flux", "krea2", "anima", "zimage"):
            monkeypatch.setattr(mc_arch, "detect_loaded_engine",
                                lambda key=key: mc_arch.by_key(key))
            monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name",
                                lambda name, key=key: mc_arch.by_key(key))
            assert creative_script._klein_visible() is False

    def test_klein_4b_is_deliberately_not_served_yet(self, host, monkeypatch):
        """§7 makes 4B a non-goal until it has test coverage of its own, and the
        gate is one tuple so that adding it stays one line and one decision."""
        assert mc_spatial_klein.ARCHITECTURES == ("flux2_9b",)
        assert mc_spatial_klein.is_klein(mc_arch.by_key("flux2_4b")) is False

    def test_spatial_layout_off_reaches_none_of_this(self, host, monkeypatch, store):
        import model_chain_krea_creative as creative_script

        klein(monkeypatch)
        script = creative_script.ScriptKreaCreative()
        mc_spatial.remember(**{mc_spatial.ENABLED: False, mc_spatial.LAYOUT: layout()})
        p = make_p(IMAGES)
        p.prompt = "a living room"
        p.negative_prompt = ""
        script.before_process(p, False)

        assert script._klein is None
        assert p.prompt == "a living room"
        assert p.extra_generation_params == {}


# --------------------------------------------------------------------------- #
# The live panel, which is advisory and says so
# --------------------------------------------------------------------------- #


class TestTheLiveSourceIsAdvisory:
    """§9 and R4. The panel offers; the runtime decides."""

    def test_an_unwired_panel_does_not_claim_the_gallery_is_empty(self, host,
                                                                  installed_stitch):
        """No gallery to read is not the same fact as a gallery with nothing in
        it, and announcing an emptiness it cannot see would be a guess."""
        live = mc_references.live_source()

        assert live.installed is True
        assert live.known is False
        # Installed and unread, not installed and empty. The panel says nothing
        # about contents it has not been shown -- and offers the image-required
        # modes rather than withholding them on a guess, because a mode refused
        # at the next press explains itself and a missing one does not.
        assert "read when you generate" in live.describe()
        assert live.usable is True

    def test_it_counts_what_the_gallery_actually_holds(self, host, installed_stitch):
        gallery = [(image, None) for image in IMAGES]
        live = mc_references.live_source(enabled=True, gallery=gallery)

        assert live.count == 3
        assert "3 usable images" in live.describe()

    def test_a_disabled_script_is_not_a_source_however_full_it_is(self, host,
                                                                  installed_stitch):
        gallery = [(image, None) for image in IMAGES]
        live = mc_references.live_source(enabled=False, gallery=gallery)

        assert live.usable is False
        assert "switched off" in live.describe()

    def test_the_runtime_answer_does_not_come_from_the_engine(self, host, monkeypatch):
        """§16's last paragraph, and the Definition of Done's tenth line.

        ``reference_count`` reports what the engine is holding, which survives a
        job. Reading it here would make an emptied gallery look source-capable
        for exactly as long as it takes somebody to be surprised by it.
        """
        klein(monkeypatch)
        engine = FakeEngine()
        engine.ref_latents.extend(["left over from a previous job"])
        host.shared.sd_model = engine

        assert mc_references.reference_count(engine) == 1
        assert mc_spatial_klein.source_for(make_p(())).usable is False
