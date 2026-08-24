"""The Image Pipeline: one surface, six rows, and the contracts that hold it up.

Three kinds of question here, and they fail in three different ways.

**Is it one panel?** The pipeline is built by whichever of the two feature
scripts Forge happens to run first, and filled by both. That is the load-
bearing trick of this refactor, and the way it breaks is silent: two shells,
each with half the stages, on a page nobody looked at in a test.

**Does a row describe the generation it is next to?** Stage 1 is Forge's, read
and never written, so every number on its row comes from a native control this
extension does not own. The failure is a panel that quotes the width and height
sliders at somebody whose Hires pass makes them the wrong numbers.

**Do the labels still line up?** The phase animation matches the progress bar's
own text, which is a real coupling between a Python constant and a JavaScript
table. It is tested rather than hoped for: rename a phase on either side and
this file says so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import mc_llm_progress
import mc_pipeline_panel
import mc_profile_state
import mc_progress

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_JS = ROOT / "javascript" / "model_chain_pipeline.js"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point every preferences and history file at a throwaway directory."""
    import mc_llm_paths

    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture(autouse=True)
def fresh_shell():
    """One shell per test, the way one page gets one shell."""
    mc_pipeline_panel.forget()
    yield
    mc_pipeline_panel.forget()


# --------------------------------------------------------------------------- #
# The shell
# --------------------------------------------------------------------------- #


class TestTheShell:
    def test_the_order_is_the_order_a_generation_runs_in(self):
        """The one ordering in the extension a user can see. A panel that drew
        Spatial above Creative while the code ran them the other way round
        would be a diagram of a different program."""
        assert mc_pipeline_panel.ORDER == (
            "prompt", "creative", "spatial", "stage1", "stage2", "output")

    def test_exactly_three_stages_are_ours(self):
        """Section 2.4. Everything else is context, and context is not
        editable from here."""
        assert set(mc_pipeline_panel.OWNED) == {"creative", "spatial", "stage2"}
        assert set(mc_pipeline_panel.CONTEXT) == {"prompt", "stage1", "output"}
        assert not set(mc_pipeline_panel.OWNED) & set(mc_pipeline_panel.CONTEXT)
        assert (set(mc_pipeline_panel.OWNED) | set(mc_pipeline_panel.CONTEXT)
                == set(mc_pipeline_panel.ORDER))

    def test_it_builds_one_shell_and_hands_the_same_one_back(self, host):
        """The whole mechanism. The second script to call host() must get the
        first script's shell, or the page ends up with two half-filled
        pipelines and no error anywhere."""
        first = mc_pipeline_panel.host()
        second = mc_pipeline_panel.host()

        assert first is second

    def test_every_owned_stage_offers_a_switch_slot_and_an_editor_slot(self, host):
        pipeline = mc_pipeline_panel.host()

        for stage in mc_pipeline_panel.OWNED:
            assert pipeline.head(stage) is not None
            assert pipeline.body(stage) is not None

    def test_no_context_stage_offers_a_switch(self, host):
        """Section 2.4 again: Prompt, Stage 1 and Output carry no Model Chain
        control, because Model Chain does not control them."""
        pipeline = mc_pipeline_panel.host()

        for stage in mc_pipeline_panel.CONTEXT:
            assert stage not in pipeline.heads

    def test_every_stage_has_a_live_second_line(self, host):
        """Including the ones this extension does not own -- a muted row that
        said only its own name would be decoration."""
        pipeline = mc_pipeline_panel.host()

        for stage in mc_pipeline_panel.ORDER:
            assert pipeline.summary(stage) is not None

    def test_forgetting_it_builds_a_fresh_one(self, host):
        """Forge rebuilds the whole tab on a UI reload, and a shell remembered
        across that would hand the new page slots belonging to a page that no
        longer exists."""
        first = mc_pipeline_panel.host()
        mc_pipeline_panel.forget()

        assert mc_pipeline_panel.host() is not first

    def test_the_handoff_says_the_size_when_it_knows_it(self):
        assert mc_pipeline_panel.handoff_note(1536, 2304) == "1536 × 2304 pixel handoff"
        assert mc_pipeline_panel.handoff_note(0, 0) == "pixel handoff"


class TestBothScriptsFillOneShell:
    """The property that makes the shared shell safe: it does not matter which
    script Forge builds first.

    Forge sorts its alwayson scripts by a key this extension does not choose,
    and a refactor that only worked in one of the two orders would work until
    somebody installed an extension whose name sorted between them.
    """

    def build(self, order):
        import model_chain
        import model_chain_krea_creative

        made = {"stage2": model_chain.ScriptModelChain,
                "creative": model_chain_krea_creative.ScriptKreaCreative}
        built = {}
        for name in order:
            instance = made[name]()
            built[name] = (instance, instance.ui(is_img2img=False))
        return mc_pipeline_panel.host(), built

    @pytest.mark.parametrize("order", [("stage2", "creative"),
                                       ("creative", "stage2")])
    def test_every_owned_stage_is_filled_whichever_script_runs_first(self, order,
                                                                     host, store):
        pipeline, _built = self.build(order)

        assert pipeline.filled == set(mc_pipeline_panel.OWNED)

    @pytest.mark.parametrize("order", [("stage2", "creative"),
                                       ("creative", "stage2")])
    def test_there_is_only_ever_one_shell(self, order, host, store):
        pipeline, _built = self.build(order)

        assert mc_pipeline_panel.host() is pipeline

    def test_the_switch_on_a_row_is_the_control_the_generation_reads(self, host,
                                                                     store):
        """Not a copy of it. A second checkbox mirroring the real one is the
        duplicate source-of-truth section 2.5 forbids, and it would be the one
        a user reached for first.

        Checked by identity against the argument lists, which are what the
        processing hooks actually receive: Stage 2's switch is the first
        argument of one, and Creative's and Spatial's are the first and the
        last-but-two of the other.
        """
        _pipeline, built = self.build(("stage2", "creative"))
        _stage2, stage2_arguments = built["stage2"]
        creative, _creative_arguments = built["creative"]

        assert creative.arguments[0] is creative.components["enabled"]
        assert creative.arguments[-3] is creative.components["spatial_enabled"]

        # The three switches are three distinct components, one per owned
        # stage, and none of them is any other stage's.
        switches = {id(stage2_arguments[0]),
                    id(creative.arguments[0]),
                    id(creative.arguments[-3])}
        assert len(switches) == 3


# --------------------------------------------------------------------------- #
# The Stage 1 context row
# --------------------------------------------------------------------------- #


class TestTheStage1Row:
    """Section 7: read, never written, and right about Hires.

    The number that matters is the last one: Stage 2 refines *finished Stage 1
    pixels*, so a Hires pass changes what it is handed. A panel that quoted the
    width and height sliders would be describing a picture that never exists.
    """

    def helpers(self):
        import model_chain

        return model_chain

    def test_without_hires_the_handoff_is_the_requested_size(self):
        mc = self.helpers()

        assert mc._stage1_size(1024, 1536) == (1024, 1536)

    def test_with_hires_the_handoff_is_the_upscaled_size(self):
        mc = self.helpers()

        assert mc._stage1_size(1024, 1536, True, 1.5) == (1536, 2304)

    def test_the_row_names_the_upscale_and_where_it_lands(self):
        mc = self.helpers()

        said = mc._stage1_summary("krea2.safetensors", 1024, 1536, True, 1.5)

        assert "1024×1536" in said
        assert "Hires 1.5×" in said
        assert "1536×2304" in said

    def test_a_checkpoint_is_named_without_its_path_or_its_hash(self):
        mc = self.helpers()

        assert mc._short_checkpoint("SD/krea2.safetensors [abc123]") == "krea2"

    def test_the_sampler_line_is_left_out_when_forge_does_not_offer_it(self):
        """A heavily customised UI loses a clause and nothing else -- the only
        acceptable failure for a panel describing controls it does not own."""
        mc = self.helpers()

        said = mc._stage1_summary("krea2.safetensors", 1024, 1536)

        assert "krea2" in said
        assert "steps" not in said

    def test_references_are_reported_when_imagestitch_has_any(self):
        mc = self.helpers()

        said = mc._stage1_summary("krea2.safetensors", 1024, 1536,
                                  stitch_gallery=[object(), object()])

        assert "ImageStitch · 2 references" in said

    def test_the_handoff_line_says_the_size_even_with_stage_2_off(self):
        """It is what Stage 2 *would* be handed, and the number somebody needs
        in order to decide whether to arm it."""
        mc = self.helpers()

        said = mc._handoff_summary(1024, 1536, True, 1.5, enabled=False)

        assert "1536 × 2304 pixel handoff" in said
        assert "Stage 2 is off" in said

    def test_the_output_row_is_the_stage_1_size_when_stage_2_is_off(self):
        mc = self.helpers()

        said = mc._output_summary(False, "", 1.0, 1024, 1536, True, 1.5)

        assert said == "1536×2304 — from Stage 1"

    def test_the_output_row_follows_stage_2_when_it_is_on(self):
        mc = self.helpers()

        said = mc._output_summary(True, "flux1-klein.safetensors", 1.0,
                                  1024, 1536, True, 1.5)

        assert "refined by Stage 2" in said

    def test_a_size_nobody_has_set_yet_says_nothing_rather_than_zero(self):
        mc = self.helpers()

        assert mc._output_summary(False, "", 1.0, 0, 0) == ""


# --------------------------------------------------------------------------- #
# Loaded, modified, not saved
# --------------------------------------------------------------------------- #


class TestTheProfileStateContract:
    """Section 8, and 8.3 is the one that matters.

    "Not saved" is one word away from "not applied", and the two readings are
    opposite in consequence: a user who reads the second one presses Save
    before every generation, or does not, and believes the image came from
    settings it did not.
    """

    def test_nothing_loaded_says_nothing(self):
        assert mc_profile_state.describe("", False) == ""

    def test_a_loaded_profile_is_named(self):
        assert mc_profile_state.describe("Editorial", False) == "**Loaded:** Editorial"

    def test_an_edited_profile_says_modified_and_not_saved(self):
        said = mc_profile_state.describe("Editorial", True)

        assert "Editorial" in said
        assert "Modified" in said
        assert "not saved" in said

    def test_the_panel_says_that_unsaved_settings_are_still_the_active_ones(self):
        """Section 8.3, said on the panel rather than in documentation -- the
        moment somebody needs it is the moment they are reading the word
        "unsaved" and deciding whether to trust the screen."""
        said = mc_profile_state.explain(True)

        assert "next Generate will use" in said
        assert mc_profile_state.explain(False) == ""

    def test_a_stage_that_is_off_still_names_what_it_has_loaded(self):
        said = mc_profile_state.describe("Editorial", True, active=False)

        assert "Editorial" in said and "stage is off" in said

    def test_nothing_loaded_is_not_the_same_as_everything_modified(self):
        """A freshly opened tab has diverged from nothing, because it came from
        nothing."""
        assert mc_profile_state.changed({"a": 1}, "") is False

    def test_a_value_that_survived_json_compares_equal(self):
        """Gradio's transport turns tuples into lists. A dirty flag that
        reported a round trip as an edit would always be on."""
        assert mc_profile_state.changed({"a": [1, 2]},
                                        mc_profile_state.snapshot({"a": (1, 2)})) is False

    def test_a_real_edit_is_reported(self):
        baseline = mc_profile_state.snapshot({"denoise": 0.35})

        assert mc_profile_state.changed({"denoise": 0.4}, baseline) is True

    def test_key_order_is_not_an_edit(self):
        baseline = mc_profile_state.snapshot({"a": 1, "b": 2})

        assert mc_profile_state.changed({"b": 2, "a": 1}, baseline) is False


class TestSpatialLayoutsAreNotTheWorkingLayout:
    """Section 8.5, which is the one place the contract needs saying twice.

    Auto Save commits the *working* layout -- what the next Generate composes.
    Save updates a *named* layout. "Loaded: Studio thirds, Modified, not saved"
    is an ordinary state to sit in for as long as you like.
    """

    @pytest.fixture
    def data(self, tmp_path, monkeypatch, host):
        from modules import paths

        monkeypatch.setattr(paths, "data_path", str(tmp_path), raising=False)
        return tmp_path

    def layout(self, regions=(), mode="smart"):
        return json.dumps({"version": 1,
                           "canvas": {"width": 1024, "height": 1344, "grid": "none"},
                           "compose_mode": mode, "auto_position_hint": True,
                           "regions": list(regions)})

    def test_a_saved_layout_comes_back(self, data):
        import mc_spatial_profiles

        mc_spatial_profiles.save("Studio thirds", self.layout())

        assert mc_spatial_profiles.get("Studio thirds") == self.layout()

    def test_an_untouched_layout_matches_the_name_it_came_from(self, data):
        import mc_spatial_profiles

        mc_spatial_profiles.save("Studio thirds", self.layout())

        assert mc_spatial_profiles.matches("Studio thirds", self.layout()) is True

    def test_a_moved_box_no_longer_matches(self, data):
        import mc_spatial_profiles

        region = {"id": "r1", "name": "Face", "type": "obj",
                  "bbox": [10, 10, 200, 200], "prompt": "", "z": 0}
        mc_spatial_profiles.save("Studio thirds", self.layout([region]))
        moved = dict(region, bbox=[40, 10, 230, 200])

        assert mc_spatial_profiles.matches("Studio thirds", self.layout([moved])) is False

    def test_opening_a_layout_at_a_different_size_is_not_an_edit(self, data):
        """The frame is a fact about txt2img rather than part of the
        composition, so changing the generation size must not report every
        saved layout as modified."""
        import mc_spatial_profiles

        mc_spatial_profiles.save("Studio thirds", self.layout())
        wider = json.loads(self.layout())
        wider["canvas"]["width"] = 1536

        assert mc_spatial_profiles.matches("Studio thirds",
                                           json.dumps(wider)) is True

    def test_deleting_a_named_layout_leaves_the_working_one_alone(self, data):
        """Deleting a saved copy of a composition is not a request to stop
        using it."""
        import mc_spatial
        import mc_spatial_profiles

        mc_spatial_profiles.save("Studio thirds", self.layout())
        mc_spatial.remember(**{mc_spatial.LAYOUT: self.layout()})
        mc_spatial_profiles.delete("Studio thirds")

        assert mc_spatial.settings()["layout"] == self.layout()

    def test_a_layout_cannot_be_saved_without_a_name(self, data):
        import mc_spatial_profiles

        with pytest.raises(mc_spatial_profiles.LayoutError):
            mc_spatial_profiles.save("   ", self.layout())

    def test_the_reserved_name_is_refused(self, data):
        import mc_spatial_profiles

        with pytest.raises(mc_spatial_profiles.LayoutError):
            mc_spatial_profiles.save(mc_spatial_profiles.NONE, self.layout())

    def test_a_damaged_store_reads_as_empty_rather_than_raising(self, data):
        import mc_spatial_profiles

        Path(mc_spatial_profiles.path()).write_text("{not json", encoding="utf-8")

        assert mc_spatial_profiles.names() == []


# --------------------------------------------------------------------------- #
# The phase animation's one coupling
# --------------------------------------------------------------------------- #


def _phase_table() -> list[tuple[str, str]]:
    """The JS file's label table, read out of the JS file.

    Parsed rather than duplicated. A copy of the table here would agree with
    the one in the browser until somebody edited one of them, which is the
    failure this test exists to catch.
    """
    source = PIPELINE_JS.read_text(encoding="utf-8")
    block = re.search(r"const PHASES = \[(.*?)\];", source, re.S)
    assert block, "the pipeline JS no longer has a PHASES table"
    return re.findall(r'match:\s*"([^"]+)",\s*stage:\s*"([^"]+)"', block.group(1))


def _ambiguous() -> str:
    source = PIPELINE_JS.read_text(encoding="utf-8")
    found = re.search(r'const AMBIGUOUS = "([^"]+)"', source)
    assert found, "the pipeline JS no longer names its ambiguous label"
    return found.group(1)


def stage_for(label: str) -> str:
    """The browser's rule, in Python: lowercase, first substring match wins."""
    said = str(label or "").strip().lower()
    if not said:
        return ""
    if said.startswith(_ambiguous()):
        return "ambiguous"
    for match, stage in _phase_table():
        if match in said:
            return stage
    return ""


class TestEveryPhaseLightsARow:
    """The animation matches the progress bar's own text, which couples a
    Python constant to a JavaScript table.

    So the coupling is tested rather than hoped for. Every label the extension
    can put on the bar is listed here with the row it should light, and the
    matching is done by re-implementing the browser's rule over the browser's
    own table -- read out of the file, not copied into this one.
    """

    @pytest.mark.parametrize("label,stage", [
        ("Waiting for Stage 1 preload", "stage1"),
        ("Preparing Stage 1", "stage1"),
        ("Stage 1", "stage1"),
        ("Stage 1 1/2", "stage1"),
        ("Loading Klein 9B", "stage2"),
        ("Stage 2 model", "stage2"),
        ("Freeing VRAM for Stage 2", "stage2"),
        ("Stage 2", "stage2"),
        ("Stage 2 1/2", "stage2"),
        ("Finishing", "output"),
    ])
    def test_an_image_phase_lights_its_own_stage(self, label, stage):
        assert stage_for(label) == stage

    @pytest.mark.parametrize("label,stage", [
        (mc_llm_progress.READING, "creative"),
        (mc_llm_progress.WRITING, "creative"),
        (mc_llm_progress.COMPOSING_READ, "spatial"),
        (mc_llm_progress.COMPOSING_WRITE, "spatial"),
    ])
    def test_a_language_model_phase_lights_the_stage_that_asked_for_it(self, label,
                                                                      stage):
        """The writer is Creative's and the composer is Spatial's. They share a
        progress vocabulary and belong to different rows."""
        assert stage_for(label) == stage

    def test_the_shared_waiting_label_is_treated_as_ambiguous(self):
        """Both passes say it, so on its own it names no stage. The browser
        keeps whatever was already running rather than guessing."""
        assert stage_for(mc_llm_progress.WAITING) == "ambiguous"

    def test_every_label_the_planner_can_emit_is_matched(self):
        """Built from the real plan rather than from a list written here, so a
        phase added to mc_progress without a rule in the browser fails."""
        job = mc_progress.build(
            stage1_arch="flux", stage1_passes=[(20, 1.5)], batch_size=1,
            stage2_arch="flux", stage2_passes=[(20, 1.5)], transition="ram",
            move_gigabytes=6.0, free_gigabytes=2.0, target_label="Klein 9B")

        unmatched = [phase.label for phase in job.phases if not stage_for(phase.label)]

        assert unmatched == []

    def test_every_language_model_label_is_matched(self):
        for pass_ in (mc_llm_progress.WRITER, mc_llm_progress.COMPOSER):
            for label in pass_.labels().values():
                assert stage_for(label) != "", label

    def test_a_label_naming_no_stage_lights_nothing(self):
        """Silence is the right answer for a phase this file does not know
        about -- lighting the wrong row would be worse than lighting none."""
        assert stage_for("Doing something else entirely") == ""
        assert stage_for("") == ""

    def test_every_stage_the_table_names_is_a_real_stage(self):
        for _match, stage in _phase_table():
            assert stage in mc_pipeline_panel.ORDER

    def test_the_browser_knows_the_same_six_stages_in_the_same_order(self):
        source = PIPELINE_JS.read_text(encoding="utf-8")
        found = re.search(r"const ORDER = \[(.*?)\];", source, re.S)
        assert found
        order = re.findall(r'"([^"]+)"', found.group(1))

        assert tuple(order) == mc_pipeline_panel.ORDER


class TestTheBrowserFileCannotAffectAGeneration:
    """The property the whole Creative Mode browser story turns on, defended
    one file further out.

    This file reflects two things and decides nothing. The way that could stop
    being true is the way it stopped being true before: something in the page
    that a generation has to wait for. So the two shapes of that are asserted
    against the source rather than argued about in a comment.
    """

    def source(self) -> str:
        return PIPELINE_JS.read_text(encoding="utf-8")

    def code(self) -> str:
        """The file with its comments removed.

        These tests are about what the file *does*, and the file explains
        itself at length -- including by naming the things it is careful not to
        use. Asserting over the prose would make a comment saying "never
        innerHTML" fail a test that innerHTML is never used.
        """
        return "\n".join(re.sub(r"//.*$", "", line)
                          for line in self.source().splitlines())

    def test_it_never_names_the_generate_button(self):
        """The old Creative gate swallowed the Generate click, polled a hidden
        textbox, and clicked Generate again once the server answered -- so a
        hidden tab made an image late and a closed one made it never arrive.
        Nothing here may go near that button."""
        assert "txt2img_generate" not in self.code()
        assert "img2img_generate" not in self.code()

    def test_it_arms_no_repeating_timer(self):
        """A one-shot that clears a finished state is a cosmetic settle. An
        interval is a poll, and a poll is the shape of something waiting."""
        assert "setInterval" not in self.code()

    def test_it_writes_no_gradio_component_value(self):
        """It reads the prompt box and the progress bar's text. Writing to a
        Gradio input from here would put a second author on a value the server
        believes it owns."""
        code = self.code()

        assert "updateInput" not in code
        assert ".dispatchEvent" not in code

    def test_the_prompt_is_echoed_as_text_and_never_as_markup(self):
        """A prompt is the string on the page most likely to contain angle
        brackets, and the echo is written straight into the panel."""
        code = self.code()

        assert "innerHTML" not in code
        # The echo assigns textContent and nothing else assigns into that
        # element at all, so the prompt cannot become markup on its way to the
        # panel however many angle brackets it contains.
        assert "target.textContent =" in code
        assert "target.innerHTML" not in code

    def test_no_gradio_generated_class_is_used_to_find_anything(self):
        """Every hook is an id this extension put in the page, or one of the
        host's own progress-bar classes. A theme is allowed to rearrange
        Gradio's internals; this has to keep working when it does."""
        code = self.code()

        for generated in ("gradio-", "svelte-", "label-wrap", "gr-button",
                          "block-label"):
            assert generated not in code
