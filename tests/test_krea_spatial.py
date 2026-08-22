"""Spatial BBOX: the coordinates, the compositor, and the two things it refuses.

The feature's claim is that a language model can improve the *words* around a
composition without ever becoming the source of truth for the composition
itself. Everything below is one of the ways that claim can quietly stop being
true.

The first is arithmetic. A box drawn bottom-right to top-left, a box dragged
past the edge, a click mistaken for a region, a resolution change that reframes
the picture: each of those has one right answer and several plausible wrong
ones, and every wrong one shows up as a subject in the wrong place rather than
as an error.

The second is authority. Pass 1 must never see a region prompt, pass 2 must
never be able to move a box, and the compositor must put the user's own words in
the elements array unchanged. Those are checked adversarially -- by handing pass
2 a reply that tries to do all three and asserting the finished prompt is
unaffected -- because the failure is silent and the output still looks fine.

The third is that "off" stops meaning off. A generation with Spatial Layout
switched off has to be byte-identical to one made before this feature existed,
down to the user turn the writer is sent, because that is what makes the feature
free for everybody who does not use it.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

import mc_broker
import mc_creative_krea
import mc_infotext
import mc_llm_paths
import mc_llm_sessions as sessions
import mc_spatial
from prompt_master.krea import composer, director, spatial
from prompt_master.krea import library as library_module


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


class FakeClient:
    """A llama.cpp client that answers instantly and remembers the asking."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls: list[dict] = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.85, top_p=0.95):
        self.calls.append({"messages": messages, "seed": seed,
                           "temperature": temperature, "top_p": top_p})
        answer = self.answers.pop(0) if self.answers else "An expanded Krea prompt."
        on_text(answer)
        return answer

    def system(self, index=-1) -> str:
        return self.calls[index]["messages"][0]["content"]

    def turn(self, index=-1) -> str:
        return self.calls[index]["messages"][-1]["content"]


class Processing:
    """The half of a StableDiffusionProcessing this feature touches."""

    def __init__(self, prompt="a quiet street", width=1024, height=1344):
        self.prompt = prompt
        self.width = width
        self.height = height
        self.extra_generation_params = {}


class Result:
    def __init__(self):
        self.comments = ""


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def client(monkeypatch, host, store):
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
    fake = FakeClient()
    monkeypatch.setattr(sessions, "_client", lambda needs_vision=False, reserve=0, role='': fake)
    monkeypatch.setattr(sessions, "_placement_notes", lambda role="": [])
    monkeypatch.setattr(mc_creative_krea, "checkpoint_objection", lambda: "")
    mc_creative_krea.creative = mc_creative_krea.Creative()
    yield fake
    mc_creative_krea.creative = mc_creative_krea.Creative()
    mc_broker.clear()


@pytest.fixture
def script():
    import model_chain_krea_creative as creative_script

    return creative_script.ScriptKreaCreative()


FACE = {"id": "r1", "name": "Face", "type": "obj", "bbox": [35, 55, 315, 360],
        "prompt": "elderly Japanese woman, silver hair, gentle expression",
        "framing": "Close-up", "angle": "3/4 left", "z": 0}

SIGN = {"id": "r2", "name": "Sign", "type": "text", "bbox": [100, 40, 500, 180],
        "text": "MIDNIGHT CAFE", "prompt": "large readable title lettering", "z": 1}


def document(regions=(FACE,), mode="smart", width=1024, height=1344, auto=True) -> str:
    return json.dumps({"version": 1,
                       "canvas": {"width": width, "height": height, "grid": "thirds"},
                       "compose_mode": mode,
                       "auto_position_hint": auto,
                       "regions": list(regions)})


def panel_values(creativity=10, seed=director.RANDOM_SEED, anti=True, loras="",
                 mode=director.NATURAL, spatial_on=False, compose="smart", layout=""):
    """What Forge hands ``before_process`` after the enabled flag.

    The four scalars, three controls per axis, then the three Spatial controls,
    exactly as ``ui()`` returns them. Built from the library rather than written
    out, because the middle block's length is the library's and a test that
    hard-coded it would pass against the wrong shape.
    """
    values = [creativity, seed, anti, loras]
    for _key in library_module.library().axis_keys:
        values.extend([mode, None, []])
    values.extend([spatial_on, compose, layout])
    return values


def generate(script, prompt="a quiet street", enabled=True, timeout=20.0,
             width=1024, height=1344, **panel):
    """One press of Generate, on a thread with a deadline.

    The deadline is the assertion: a hook that waits for the image job it is
    part of does not fail a test run, it hangs one.
    """
    p = Processing(prompt, width=width, height=height)
    error: list[BaseException] = []

    def press():
        try:
            script.before_process(p, enabled, *panel_values(**panel))
        except BaseException as exc:  # surfaced on the calling thread below
            error.append(exc)

    worker = threading.Thread(target=press, name="press-generate", daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "before_process did not return; the roll deadlocked"
    if error:
        raise error[0]
    return p


def composed(p) -> dict:
    """The structured prompt this generation produced, parsed back."""
    return json.loads(p.prompt)


def shared_prefix(first: str, second: str) -> str:
    """What llama.cpp's prompt cache keeps between two requests: the common
    head, up to the first thing that differs.

    The cache works in tokens rather than characters, so the real boundary can
    sit a token either side of this one. Nothing below turns on that: the
    property being measured is which *blocks* survive the comparison, and a
    block is hundreds of characters wide.
    """
    return os.path.commonprefix([first, second])


# --------------------------------------------------------------------------- #
# The line between what a model writes and what code builds
# --------------------------------------------------------------------------- #


class TestTheCompositorCannotAskAnything:
    def test_it_reaches_no_model_and_no_network(self):
        """Stated against the import graph rather than trusted, the same way
        the Director's is. "Ask the model for the finished JSON" is the single
        most tempting refactor available here, and a compositor that could reach
        a client is one commit away from having done it."""
        import ast
        from pathlib import Path as _Path

        tree = ast.parse(_Path(spatial.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")

        assert not any(name.startswith("mc_") for name in imported)
        assert not any("inference" in name or "client" in name for name in imported)
        assert not any(name in {"httpx", "requests", "socket", "urllib"}
                       for name in imported)

    def test_the_composer_is_asked_for_two_strings_and_reads_two_strings(self):
        """The isolation property as a property of the code rather than of the
        instruction: there is no key but these two that ``parse`` can return."""
        scene, background = composer.parse(json.dumps({
            "scene": "a", "background": "b", "elements": [1], "bbox": [0, 0, 1, 1]}))

        assert (scene, background) == ("a", "b")

    def test_a_reply_with_no_scene_is_refused_rather_than_patched(self):
        with pytest.raises(composer.Refused):
            composer.parse('{"background": "neon"}')
        with pytest.raises(composer.Refused):
            composer.parse("")

    def test_a_reply_longer_than_this_pass_may_write_is_refused(self):
        """The pass is defined as one that shortens. A reply several times the
        length of its input has done something else, and truncating it would
        produce a sentence cut in half."""
        with pytest.raises(composer.Refused):
            composer.parse(json.dumps({"scene": "x" * (composer.MAX_SCENE + 1)}))

    def test_it_finds_the_object_a_local_model_wrapped_or_prefaced(self):
        """Lenient about the wrapping, strict about the contents. A model that
        fenced its answer is not a model failing to do the job."""
        assert composer.parse('Sure:\n```json\n{"scene": "a"}\n```\nhope that helps') \
            == ("a", "")
        assert composer.parse('<think>hmm</think>{"scene": "b"}') == ("b", "")


# --------------------------------------------------------------------------- #
# Coordinates (design intent §11, "Coordinate tests")
# --------------------------------------------------------------------------- #


class TestCoordinates:
    def test_a_box_survives_the_round_trip_unchanged(self):
        layout = spatial.parse(document())

        assert layout.regions[0].bbox == (35, 55, 315, 360)

    def test_a_reversed_drag_is_the_same_box(self):
        """Dragging bottom-right to top-left is not an error and is not the
        browser's to fix silently -- the canonical form is defined once, here, so
        a hand-edited document behaves like a mouse."""
        assert spatial.normalise_bbox([315, 360, 35, 55]) == [35, 55, 315, 360]

    def test_coordinates_are_clamped_rather_than_refused(self):
        """A box dragged past the edge is a box somebody meant to touch it."""
        assert spatial.normalise_bbox([-40, -10, 1400, 1200]) == [0, 0, 1000, 1000]

    def test_a_box_with_no_area_is_refused(self):
        """A click is not a region, and there is no coordinate this module is
        entitled to invent for one."""
        assert spatial.normalise_bbox([200, 200, 200, 400]) is None
        assert spatial.normalise_bbox([200, 200, 400, 200]) is None

    def test_a_region_with_no_usable_box_is_skipped_with_a_reason(self):
        layout = spatial.parse(document([dict(FACE, bbox=[10, 10, 10, 10])]))

        assert layout.regions == ()
        assert "no usable box" in " ".join(layout.notes)

    def test_only_the_invalid_region_is_skipped(self):
        """§10: skip the invalid region where possible, and log the reason. The
        two good boxes beside it are still the user's."""
        layout = spatial.parse(document([FACE, dict(SIGN, bbox=[0, 0, 0, 0]), SIGN]))

        assert [region.identifier for region in layout.ordered] == ["r1", "r2"]
        assert len(layout.notes) == 1

    def test_the_same_layout_at_a_different_resolution_is_the_same_composition(self):
        """Normalized coordinates are fractions of the frame, so a resolution
        change moves nothing. This is the property that makes 0..1000 worth the
        conversion in the first place."""
        small = spatial.compose(spatial.parse(document(width=1024, height=1344)),
                                scene="x", ratio=spatial.aspect_ratio(1024, 1344))
        large = spatial.compose(spatial.parse(document(width=1536, height=2016)),
                                scene="x", ratio=spatial.aspect_ratio(1536, 2016))

        assert small == large

    @pytest.mark.parametrize("size, expected", [
        ((1024, 1024), "1:1"),
        ((1152, 896), "9:7"),
        ((1024, 1344), "3:4"),
        ((832, 1216), "2:3"),
        ((1216, 832), "3:2"),
        ((1920, 1080), "16:9"),
    ])
    def test_the_aspect_ratio_is_a_form_a_prompt_names(self, size, expected):
        """832x1216 reduces exactly to 13:19, which is true and useless. It is
        2:3 to within two and a half per cent, and 2:3 is a form the model has
        seen a great many times."""
        assert spatial.aspect_ratio(*size) == expected

    def test_an_unknown_frame_says_nothing_about_the_frame(self):
        """Absent rather than empty: an empty ratio is a claim, and this module
        would be making it up."""
        prompt = json.loads(spatial.compose(spatial.parse(document()), scene="x"))

        assert "aspect_ratio" not in prompt


# --------------------------------------------------------------------------- #
# The compositor (§11, "Compositor tests")
# --------------------------------------------------------------------------- #


class TestTheCompositor:
    def test_the_user_s_own_words_are_present_verbatim(self):
        """The promise the whole feature makes about region prompts, kept by
        concatenation rather than by asking anything to preserve it."""
        layout = spatial.parse(document())
        element = json.loads(spatial.compose(layout, scene="x"))[
            "compositional_deconstruction"]["elements"][0]

        assert element["desc"].startswith(FACE["prompt"])

    def test_the_hints_are_the_design_intent_s_own_example(self):
        """§4.3 works this exact box through, and the wording is a contract with
        the reader of that document as much as with the model."""
        region = spatial.parse(document()).regions[0]

        assert region.describe() == (
            "elderly Japanese woman, silver hair, gentle expression, "
            "shown in a close-up view, in a three-quarter view from the left, "
            "positioned in the upper-left area, occupying a medium-sized area")

    def test_the_position_hint_follows_the_thirds_grid_the_canvas_draws(self):
        """The sentence the model reads and the guides the user drew against
        describe the same division of the frame."""
        assert spatial.position_hint([0, 0, 100, 100]) == \
            "positioned in the upper-left area"
        assert spatial.position_hint([900, 900, 1000, 1000]) == \
            "positioned in the lower-right area"
        assert spatial.position_hint([450, 450, 550, 550]) == spatial.CENTRE_HINT

    def test_the_size_hint_has_five_bands_and_no_percentages(self):
        """A text-to-image model has no calibration for "8.54% of the frame"."""
        assert spatial.size_hint([0, 0, 100, 100]) == "occupying a very small area"
        assert spatial.size_hint([0, 0, 250, 250]) == "occupying a small area"
        assert spatial.size_hint([0, 0, 400, 400]) == "occupying a medium-sized area"
        assert spatial.size_hint([0, 0, 600, 600]) == "occupying a large area"
        assert spatial.size_hint([0, 0, 1000, 1000]) == spatial.DOMINANT

    def test_hints_can_be_turned_off_and_the_words_stay(self):
        layout = spatial.parse(document(auto=False))

        assert layout.regions[0].describe(False) == (
            "elderly Japanese woman, silver hair, gentle expression, "
            "shown in a close-up view, in a three-quarter view from the left")

    def test_a_text_region_keeps_its_words_apart_from_its_description(self):
        """Only ``text`` is intended to become visible writing. Merging the two
        would ask the model to render the adjectives."""
        layout = spatial.parse(document([SIGN]))
        element = json.loads(spatial.compose(layout, scene="x"))[
            "compositional_deconstruction"]["elements"][0]

        assert element["type"] == "text"
        assert element["text"] == "MIDNIGHT CAFE"
        assert "MIDNIGHT CAFE" not in element["desc"]
        assert element["desc"].startswith("large readable title lettering")

    def test_a_text_region_with_no_text_is_skipped(self):
        layout = spatial.parse(document([dict(SIGN, text="")]))

        assert layout.regions == ()
        assert "no text to render" in " ".join(layout.notes)

    def test_an_object_region_with_no_prompt_is_skipped(self):
        layout = spatial.parse(document([dict(FACE, prompt="")]))

        assert layout.regions == ()
        assert "no prompt" in " ".join(layout.notes)

    def test_the_element_order_is_the_canvas_z_order(self):
        top = dict(FACE, id="top", z=5)
        bottom = dict(SIGN, id="bottom", z=1)
        layout = spatial.parse(document([top, bottom]))

        assert [region.identifier for region in layout.ordered] == ["bottom", "top"]

    def test_regions_at_the_same_depth_keep_the_order_they_were_drawn(self):
        """An elements array that reordered itself between two generations of
        one layout would make an A/B comparison meaningless."""
        first = dict(FACE, id="first", z=0)
        second = dict(FACE, id="second", z=0)
        layout = spatial.parse(document([first, second]))

        assert [region.identifier for region in layout.ordered] == ["first", "second"]

    def test_the_same_layout_composes_to_the_same_bytes(self):
        layout = spatial.parse(document([FACE, SIGN]))
        once = spatial.compose(layout, scene="a scene", background="a place", ratio="3:4")
        again = spatial.compose(spatial.parse(document([FACE, SIGN])),
                                scene="a scene", background="a place", ratio="3:4")

        assert once == again

    def test_the_prompt_is_compact_and_in_a_fixed_key_order(self):
        """Every space in here is a token the image model's text encoder pays
        for and none of them is read by a person."""
        prompt = spatial.compose(spatial.parse(document()), scene="a scene", ratio="3:4")

        assert prompt.startswith('{"aspect_ratio":"3:4","high_level_description":')
        assert '","' in prompt and ", " not in prompt.split('"desc"')[0]

    def test_a_framing_this_build_does_not_know_is_dropped_with_a_note(self):
        """Never passed through to the prompt, and never silent: a dropped
        selection the user made is exactly what §10 is about."""
        layout = spatial.parse(document([dict(FACE, framing="Dutch tilt")]))

        assert layout.regions[0].framing == ""
        assert "does not know" in " ".join(layout.notes)

    def test_more_regions_than_one_layout_carries_are_truncated_with_a_note(self):
        layout = spatial.parse(document([FACE] * (spatial.MAX_REGIONS + 4)))

        assert len(layout.regions) == spatial.MAX_REGIONS
        assert "the first 24 were used" in " ".join(layout.notes)


# --------------------------------------------------------------------------- #
# Versioning (§7, §8.4)
# --------------------------------------------------------------------------- #


class TestVersioning:
    def test_a_later_version_is_refused_visibly_rather_than_read(self):
        """A v2 that moved one field would otherwise present as a v1 with
        several regions missing, and boxes vanishing with no explanation is the
        worst thing this feature could do to somebody's afternoon."""
        layout = spatial.parse(json.dumps({"version": 2, "regions": [FACE]}))

        assert layout.unreadable
        assert layout.regions == ()
        assert "cannot read" in " ".join(layout.notes)

    def test_unreadable_json_is_refused_the_same_way(self):
        layout = spatial.parse("{not json")

        assert layout.unreadable
        assert "could not be read" in " ".join(layout.notes)

    def test_an_empty_box_is_not_an_error(self):
        """A user who has never opened the editor is not a failure to report."""
        layout = spatial.parse("")

        assert not layout.unreadable
        assert layout.notes == ()

    def test_unknown_fields_are_ignored_where_it_is_safe_to(self):
        layout = spatial.parse(json.dumps({
            "version": 1, "something_later": True,
            "regions": [dict(FACE, opacity=0.5)]}))

        assert layout.regions[0].bbox == (35, 55, 315, 360)
        assert "opacity" not in layout.serialize()


# --------------------------------------------------------------------------- #
# Pass 1 (§11, "Creative + Direct integration")
# --------------------------------------------------------------------------- #


class TestTheWriterNeverSeesTheRegions:
    def test_no_region_prompt_reaches_the_first_pass(self, script, client, store):
        """§2.2. Handing pass 1 the region text would put the user's own words
        through a rewriter, and the compositor would then place the original
        beside the rewrite -- which is a request for two of the same subject."""
        generate(script, spatial_on=True, compose="direct", layout=document())

        assert FACE["prompt"] not in client.turn(0)
        assert "35" not in client.turn(0)

    def test_the_writer_is_told_one_thing_and_it_is_that_placement_is_handled(
            self, script, client, store):
        generate(script, spatial_on=True, compose="direct", layout=document())

        assert spatial.PLACEMENT_NOTE in client.turn(0)

    def test_the_note_is_labelled_like_every_other_block_in_the_turn(self):
        """An unlabelled sentence dropped into a labelled turn is the failure
        ``enhancer`` spends a page avoiding."""
        assert spatial.directed("").startswith(director.BRIEF_HEADING)
        assert spatial.directed("creative_direction:\nrule").count(
            director.BRIEF_HEADING) == 1

    def test_with_no_layout_the_turn_is_the_turn_it_always_was(self, script, client,
                                                               store):
        """The compatibility guarantee, checked as bytes. Spatial Layout off has
        to mean the writer is asked exactly what it was asked before this
        feature existed -- same user turn, same prompt-cache prefix, same
        seconds."""
        generate(script, "a quiet street", spatial_on=False, layout=document())
        without = client.turn(0)
        client.calls.clear()
        generate(script, "a quiet street", spatial_on=True, compose="direct",
                 layout="")

        assert client.turn(0) == without
        assert spatial.PLACEMENT_NOTE not in without

    def test_direct_mode_makes_exactly_one_request(self, script, client, store):
        """The A/B control, and the reason it is worth having: Direct is
        Creative Mode's own cost, unchanged."""
        generate(script, spatial_on=True, compose="direct", layout=document())

        assert len(client.calls) == 1


# --------------------------------------------------------------------------- #
# Pass 2 (§11, "Creative + Smart integration", "Spatial isolation test")
# --------------------------------------------------------------------------- #


class TestTheSpatialComposer:
    def test_smart_mode_makes_a_second_request_and_only_a_second(self, script, client,
                                                                 store):
        client.answers = ["A woman stands in the centre of a rainy street.",
                          '{"scene": "A rainy street at night.", '
                          '"background": "wet asphalt and neon"}']
        generate(script, spatial_on=True, compose="smart", layout=document())

        assert len(client.calls) == 2

    def test_the_second_pass_is_shown_the_regions(self, script, client, store):
        """It has to be: the whole job is stopping the scene from arguing with
        boxes it cannot see."""
        client.answers = ["A woman in the centre.", '{"scene": "A rainy street."}']
        generate(script, spatial_on=True, compose="smart", layout=document())

        assert FACE["prompt"] in client.turn(1)
        assert "upper-left" in client.turn(1)

    def test_the_second_pass_is_not_shown_the_coordinates(self, script, client, store):
        """What it needs to know is that a face is in the upper-left so it can
        stop the scene saying "centred". The numbers are the compositor's, and
        telling the model about them only gives it something to adjust."""
        client.answers = ["A woman in the centre.", '{"scene": "A rainy street."}']
        generate(script, spatial_on=True, compose="smart", layout=document())

        assert "315" not in client.turn(1)

    def test_its_reply_supplies_the_scene_and_the_background_and_nothing_else(
            self, script, client, store):
        client.answers = ["A woman stands in the centre of a rainy street.",
                          '{"scene": "A rainy street at night.", '
                          '"background": "wet asphalt and neon"}']
        p = generate(script, spatial_on=True, compose="smart", layout=document())
        prompt = composed(p)

        assert prompt["high_level_description"] == "A rainy street at night."
        assert prompt["compositional_deconstruction"]["background"] == \
            "wet asphalt and neon"

    def test_an_adversarial_reply_cannot_move_a_box(self, script, client, store):
        """§11's isolation test, and the property stated as code: only two names
        are ever read out of the reply, so anything else it carries cannot reach
        the prompt whatever it contains."""
        client.answers = [
            "A woman in the centre.",
            json.dumps({"scene": "A street.",
                        "background": "neon",
                        "elements": [{"type": "obj", "bbox": [0, 0, 1000, 1000],
                                      "desc": "a completely different subject"}],
                        "regions": [], "bbox": [1, 2, 3, 4]})]
        p = generate(script, spatial_on=True, compose="smart", layout=document())
        elements = composed(p)["compositional_deconstruction"]["elements"]

        assert len(elements) == 1
        assert elements[0]["bbox"] == [35, 55, 315, 360]
        assert elements[0]["desc"].startswith(FACE["prompt"])
        assert "completely different subject" not in p.prompt

    def test_a_reply_that_rewrites_a_region_prompt_is_ignored(self, script, client,
                                                              store):
        client.answers = [
            "A woman in the centre.",
            json.dumps({"scene": "A street.",
                        "regions": [dict(FACE, prompt="a young man")]})]
        p = generate(script, spatial_on=True, compose="smart", layout=document())

        assert "a young man" not in p.prompt
        assert FACE["prompt"] in p.prompt

    def test_the_overreach_is_counted_even_though_it_is_ignored(self):
        """A Composer that keeps trying to write the elements array is a
        Composer whose instruction needs work, and nobody finds that out unless
        the attempt is visible somewhere."""
        assert composer.overreached('{"scene":"a","elements":[]}') == ("elements",)
        assert composer.overreached('{"scene":"a"}') == ()

    def test_its_instruction_says_what_it_may_not_do(self):
        """The belt beside the braces. Isolation is enforced by reading two
        keys; the instruction is what stops the model wasting a request trying
        anyway."""
        said = composer.SYSTEM_PROMPT

        assert "coordinates" in said
        assert "centered" in said
        assert '"scene"' in said and '"background"' in said

    def test_it_does_not_carry_krea_s_expansion_instruction(self):
        """It cannot: that file's rule 6 forbids JSON and its rule 1 asks for
        expansion, and both are the opposite of this pass's job. Appending an
        addendum telling the model to disregard two of the rules above it would
        be this extension arguing with a vendored file inside the context
        window."""
        from prompt_master.krea import enhancer

        assert enhancer.base_instruction() not in composer.SYSTEM_PROMPT

    def test_its_seed_is_derived_from_the_creative_seed(self, store):
        """One recorded number reproduces the whole roll, both passes."""
        assert mc_spatial.composer_seed(4242) == mc_spatial.composer_seed(4242)
        assert mc_spatial.composer_seed(4242) != mc_spatial.composer_seed(4243)

    def test_it_samples_cool_and_off_the_creativity_curve(self, script, client, store):
        """Creativity is about how much art direction pass 1 is given. This pass
        is an edit, and a sampler exploring alternatives here would make the
        Smart-against-Direct comparison a comparison of two draws."""
        client.answers = ["A woman in the centre.", '{"scene": "A street."}']
        generate(script, creativity=10, spatial_on=True, compose="smart",
                 layout=document())

        assert client.calls[1]["temperature"] == composer.TEMPERATURE
        assert client.calls[1]["temperature"] < client.calls[0]["temperature"]


# --------------------------------------------------------------------------- #
# The order of the Composer's turn is a prompt-cache decision
# --------------------------------------------------------------------------- #


class TestTheComposerReadsTheBoxesBeforeTheScene:
    """llama.cpp reuses a common prefix and stops at the first difference.

    The scene is written by the pass immediately before this one, so it is new
    text on every generation and the prefix can never survive it. Whatever sits
    after it is re-read every time. The layout is the boxes the user drew once
    and has not touched since, and it used to sit there -- about 130 tokens of a
    230-token re-read, on a server that was already warm.

    So the turn is ordered stable-first: source, layout, scene. The exchange is
    that a run which redraws the boxes *and* keeps the scene identical -- a
    locked creative seed and a dragged box -- re-reads the scene it would
    previously have kept. That is the one case this costs anything, and it is
    priced below, not hidden.
    """

    def turn(self, scene, regions=(FACE,), source="a quiet street", ratio="3:4"):
        return composer.user_content(source, scene, spatial.parse(document(regions)),
                                     ratio)

    def previous_order(self, scene, regions=(FACE,), source="a quiet street",
                       ratio="3:4"):
        """The turn as it was built before this change: source, scene, layout.

        Written out here rather than imported, because the whole point of the
        comparisons below is that this shape no longer exists in the code.
        """
        layout = spatial.parse(document(regions))
        lines = [composer.LAYOUT_HEADING, f"frame aspect ratio: {ratio}"]
        lines.extend(composer.region_line(position, region)
                     for position, region in enumerate(layout.ordered, start=1))
        return "\n\n".join([f"{composer.SOURCE_HEADING}\n{source}",
                             f"{composer.USER_HEADING}\n{scene}",
                             "\n".join(lines)])

    def test_the_stable_blocks_come_first_and_the_new_one_comes_last(self):
        turn = self.turn("Rain over an empty street.", regions=(FACE, SIGN))

        assert turn.index(composer.SOURCE_HEADING) \
            < turn.index(composer.LAYOUT_HEADING) \
            < turn.index(composer.USER_HEADING)

    def test_a_new_scene_over_the_same_boxes_reaches_the_model_as_a_new_tail(self):
        """The warm path, and the one that runs on nearly every generation."""
        kept = shared_prefix(self.turn("Rain over an empty street."),
                             self.turn("Noon at a dry harbour."))

        assert kept.endswith(f"{composer.USER_HEADING}\n")
        assert composer.LAYOUT_HEADING in kept
        assert FACE["prompt"] in kept
        assert "upper-left" in kept

    def test_which_is_the_layout_no_longer_being_read_a_second_time(self):
        """The same two runs, stated as what stopped being re-read."""
        first = self.turn("Rain over an empty street.", regions=(FACE, SIGN))
        second = self.turn("Noon at a dry harbour.", regions=(FACE, SIGN))
        reread = first[len(shared_prefix(first, second)):]
        before = self.previous_order("Rain over an empty street.",
                                     regions=(FACE, SIGN))
        reread_before = before[len(shared_prefix(
            before, self.previous_order("Noon at a dry harbour.",
                                        regions=(FACE, SIGN)))):]

        assert composer.LAYOUT_HEADING not in reread
        assert SIGN["text"] not in reread
        assert len(reread) < len(reread_before) / 2

    def test_redrawing_a_box_is_no_worse_than_the_order_this_replaced(self):
        """The other ordinary case: the user moves a box, and pass 1 writes a
        fresh scene as it always does. Both orders re-read from the change
        onwards; this one starts re-reading later, because the part of the
        layout above the change is still shared."""
        moved = dict(FACE, bbox=[400, 55, 700, 360])
        now = shared_prefix(self.turn("Rain over an empty street.",
                                      regions=(FACE, SIGN)),
                            self.turn("Noon at a dry harbour.",
                                      regions=(moved, SIGN)))
        was = shared_prefix(self.previous_order("Rain over an empty street.",
                                                regions=(FACE, SIGN)),
                            self.previous_order("Noon at a dry harbour.",
                                                regions=(moved, SIGN)))

        assert len(now) >= len(was)

    def test_the_one_run_it_costs_something_is_a_locked_seed_and_a_moved_box(self):
        """Priced rather than papered over. With the creative seed pinned the
        scene repeats, so the old order kept it and re-read only the layout
        below it; this one re-reads the scene as well. It buys the saving on
        every run where the scene is new, which is every run where the seed is
        not held still."""
        same = "Rain over an empty street."
        moved = dict(FACE, bbox=[400, 55, 700, 360])
        now = shared_prefix(self.turn(same, regions=(FACE, SIGN)),
                            self.turn(same, regions=(moved, SIGN)))
        was = shared_prefix(self.previous_order(same, regions=(FACE, SIGN)),
                            self.previous_order(same, regions=(moved, SIGN)))

        assert len(now) < len(was)
        assert composer.USER_HEADING not in now

    def test_the_recorded_instruction_version_moved_with_the_order(self):
        """What the model is shown changed, so an image made before this and an
        image made after it are not the same recipe. §7's whole purpose."""
        assert composer.INSTRUCTION_VERSION == 2


# --------------------------------------------------------------------------- #
# Failure (§10)
# --------------------------------------------------------------------------- #


class TestFailureFallsBackAndSaysSo:
    def test_a_composer_that_fails_falls_back_to_direct_merge(self, script, client,
                                                              store):
        """§5.5, and the boxes are unaffected: what is lost is the de-conflicting
        of the global scene, not the composition."""
        client.answers = ["A woman stands in the centre of a rainy street.",
                          "I'm afraid I can't help with that."]
        p = generate(script, spatial_on=True, compose="smart", layout=document())
        prompt = composed(p)

        assert prompt["high_level_description"] == \
            "A woman stands in the centre of a rainy street."
        assert prompt["compositional_deconstruction"]["elements"][0]["bbox"] == \
            [35, 55, 315, 360]

    def test_the_image_is_never_cancelled_because_the_composer_failed(self, script,
                                                                      client, store):
        client.answers = ["An expanded prompt.", ""]
        p = generate(script, spatial_on=True, compose="smart", layout=document())

        assert p.prompt
        assert composed(p)["compositional_deconstruction"]["elements"]

    def test_it_says_on_the_result_that_the_scene_was_merged_directly(self, script,
                                                                      client, store):
        client.answers = ["An expanded prompt.", "not a schema"]
        generate(script, spatial_on=True, compose="smart", layout=document())
        result = Result()
        script.postprocess(Processing(), result)

        assert "Spatial Layout" in result.comments
        assert "merged directly" in result.comments
        assert "have not been changed" in result.comments

    def test_no_valid_regions_generates_with_the_creative_output_only(self, script,
                                                                      client, store):
        """§10. Not an error, not a refusal -- a Creative Mode generation."""
        client.answers = ["A rainy street."]
        p = generate(script, spatial_on=True, compose="smart",
                     layout=document([dict(FACE, prompt="")]))

        assert p.prompt == "A rainy street."
        assert mc_infotext.SPATIAL_MODE not in p.extra_generation_params

    def test_an_unreadable_layout_says_so_rather_than_composing_nothing(self, script,
                                                                        client, store):
        client.answers = ["A rainy street."]
        generate(script, spatial_on=True, layout='{"version": 99}')
        result = Result()
        script.postprocess(Processing(), result)

        assert p_said(result, "Spatial Layout")
        assert "version" in result.comments

    def test_a_compositor_that_raises_falls_back_to_the_creative_output(
            self, script, client, store, monkeypatch):
        """§10's last line: fall back to the Creative output and *visibly*
        report that the layout was not applied."""
        client.answers = ["A rainy street."]

        def explode(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(spatial, "compose", explode)
        p = generate(script, spatial_on=True, compose="direct", layout=document())
        result = Result()
        script.postprocess(p, result)

        assert p.prompt == "A rainy street."
        assert "was not applied" in result.comments

    def test_a_writer_failure_is_still_the_generation_the_user_typed(self, script,
                                                                     client, store,
                                                                     monkeypatch):
        """Unchanged from before this feature: the existing Creative Mode
        fallback, with the layout reported as not applied rather than composed
        around a prompt nobody expanded."""
        def refuse(*args, **kwargs):
            raise RuntimeError("llama-server is not running")

        monkeypatch.setattr(sessions, "_client", refuse)
        p = generate(script, "a quiet street", spatial_on=True, layout=document())

        assert p.prompt == "a quiet street"
        assert p.extra_generation_params == {}


def p_said(result, text) -> bool:
    return text in result.comments


# --------------------------------------------------------------------------- #
# Infotext (§8, §11 "Exact PNG replay", "Workflow restore")
# --------------------------------------------------------------------------- #


class TestWhatTheImageRecords:
    def test_the_prompt_line_is_the_finished_structured_prompt(self, script, client,
                                                               store):
        """The whole of "how do I make this picture again". It is assigned to
        ``p.prompt`` before Forge writes the infotext, so the recorded Prompt is
        exactly what Krea was given."""
        client.answers = ["A rainy street.", '{"scene": "A rainy street."}']
        p = generate(script, spatial_on=True, compose="smart", layout=document())

        assert composed(p)["compositional_deconstruction"]["elements"]

    def test_the_canvas_is_recorded_and_it_is_the_normalised_one(self, script, client,
                                                                 store):
        """What an image records is what this build actually used -- clamped
        coordinates, canonical ordering, unknown fields gone."""
        p = generate(script, spatial_on=True, compose="direct",
                     layout=document([dict(FACE, bbox=[-5, 55, 315, 360], junk=1)]))
        recorded = json.loads(p.extra_generation_params[mc_infotext.SPATIAL_LAYOUT])

        assert recorded["regions"][0]["bbox"] == [0, 55, 315, 360]
        assert "junk" not in recorded["regions"][0]

    def test_it_records_which_merge_made_it(self, script, client, store):
        p = generate(script, spatial_on=True, compose="direct", layout=document())
        recorded = p.extra_generation_params

        assert recorded[mc_infotext.SPATIAL_MODE] == "True"
        assert recorded[mc_infotext.SPATIAL_COMPOSE_MODE] == "direct"
        assert recorded[mc_infotext.SPATIAL_VERSION] == spatial.VERSION
        assert recorded[mc_infotext.SPATIAL_PROMPT_VERSION] == spatial.PROMPT_VERSION

    def test_smart_records_both_scenes_because_one_of_them_is_nowhere_else(
            self, script, client, store):
        client.answers = ["A woman in the centre of a rainy street.",
                          '{"scene": "A rainy street."}']
        p = generate(script, spatial_on=True, compose="smart", layout=document())
        recorded = p.extra_generation_params

        assert recorded[mc_infotext.SPATIAL_ENHANCED_SCENE] == \
            "A woman in the centre of a rainy street."
        assert recorded[mc_infotext.SPATIAL_SCENE] == "A rainy street."
        assert recorded[mc_infotext.SPATIAL_COMPOSER_SEED]

    def test_direct_records_neither_because_the_prompt_line_already_says_it(
            self, script, client, store):
        """§5.2's reasoning, applied consistently: in Direct mode the enhanced
        scene *is* the high_level_description."""
        p = generate(script, spatial_on=True, compose="direct", layout=document())
        recorded = p.extra_generation_params

        assert mc_infotext.SPATIAL_ENHANCED_SCENE not in recorded
        assert mc_infotext.SPATIAL_SCENE not in recorded

    def test_an_ordinary_generation_records_no_spatial_keys_at_all(self, script,
                                                                   client, store):
        p = generate(script, spatial_on=False)

        assert not any(key in p.extra_generation_params
                       for key in mc_infotext.SPATIAL_KEYS)

    def test_the_recorded_layout_survives_the_host_s_own_quoting(self, script, client,
                                                                 store, host):
        """The layout is JSON inside a comma-separated parameter line, which is
        the one thing about it that could break silently. The host quotes it and
        parses it back; this is that round trip, through the host's own code."""
        from modules import infotext_utils

        p = generate(script, spatial_on=True, compose="direct",
                     layout=document([FACE, SIGN]))
        line = ", ".join(f"{key}: {infotext_utils.quote(value)}"
                         for key, value in p.extra_generation_params.items())
        parsed = infotext_utils.parse_generation_parameters(f"prompt\n{line}")

        assert json.loads(parsed[mc_infotext.SPATIAL_LAYOUT])["regions"][1]["text"] \
            == "MIDNIGHT CAFE"


class TestPastingOneBack:
    """§8.1: an ordinary paste reproduces the picture and re-runs nothing."""

    @pytest.fixture
    def built(self, store, host):
        import model_chain_krea_creative as creative_script

        instance = creative_script.ScriptKreaCreative()
        instance.ui(False)
        return instance

    def fields(self, built):
        return {field.api: field for field in built.infotext_fields}

    def test_a_paste_switches_spatial_layout_off(self, built):
        field = self.fields(built)["krea_spatial_enabled"]

        assert field.function({mc_infotext.SPATIAL_MODE: "True"}) is False

    def test_a_paste_switches_creative_mode_off_as_well(self, built):
        """Two switches because they are two features. Creative Mode off with
        Spatial on would compose boxes around a prompt nobody expanded."""
        fields = self.fields(built)
        params = {mc_infotext.CREATIVE_MODE: "True", mc_infotext.SPATIAL_MODE: "True"}

        assert fields["krea_creative_enabled"].function(params) is False
        assert fields["krea_spatial_enabled"].function(params) is False

    def test_an_image_with_no_spatial_keys_leaves_the_control_alone(self, built):
        """Returning None is how the host is told not to touch a control. An
        ordinary image must not be able to switch a feature off any more than
        on."""
        field = self.fields(built)["krea_spatial_enabled"]

        assert field.function({"Steps": 20}) is None

    def test_every_key_a_spatial_image_writes_is_forwarded(self):
        """The "send to txt2img" buttons forward by exact name, so a key that is
        not listed simply does not arrive and a restore finds half a record."""
        forwarded = mc_infotext.creative_paste_field_names()

        for key in mc_infotext.SPATIAL_KEYS:
            assert key in forwarded


class TestRestoringTheWorkflow:
    """§8.3: the other half, and the one that is allowed to overwrite things."""

    @pytest.fixture
    def built(self, store, host):
        import model_chain_krea_creative as creative_script

        instance = creative_script.ScriptKreaCreative()
        instance.ui(False)
        return instance

    @pytest.fixture
    def pasted(self, built):
        mc_creative_krea.pasted.remember(mc_infotext.creative_setup({
            mc_infotext.CREATIVE_MODE: "True",
            mc_infotext.CREATIVE_SOURCE: "a quiet street",
            mc_infotext.CREATIVE_CREATIVITY: "7",
            mc_infotext.SPATIAL_MODE: "True",
            mc_infotext.SPATIAL_VERSION: str(spatial.VERSION),
            mc_infotext.SPATIAL_COMPOSE_MODE: "direct",
            mc_infotext.SPATIAL_LAYOUT: document([FACE, SIGN]),
        }))
        yield
        mc_creative_krea.pasted.clear()

    def restore(self):
        import model_chain_krea_creative as creative_script

        return creative_script._restore_setup(False)

    def test_it_puts_the_canvas_back(self, pasted, store):
        *_head, layout, spatial_on, mode, _status = self.restore()
        restored = json.loads(layout["value"])

        assert [region["id"] for region in restored["regions"]] == ["r1", "r2"]
        assert restored["regions"][0]["bbox"] == [35, 55, 315, 360]
        assert spatial_on["value"] is True
        assert mode["value"] == "direct"

    def test_it_puts_the_compose_mode_back_too(self, pasted, store):
        self.restore()

        assert mc_spatial.settings()["compose_mode"] == "direct"
        assert mc_spatial.settings()["enabled"] is True

    def test_it_says_it_did(self, pasted, store):
        _prompt, _enabled, status, *_rest = self.restore()

        assert "spatial canvas is back" in status

    def test_a_layout_from_a_later_build_is_left_alone_and_reported(self, built, store):
        """§8.4: exact replay never depended on this record, so the honest thing
        to do with a layout this build cannot read is nothing, out loud."""
        import model_chain_krea_creative as creative_script

        mc_creative_krea.pasted.remember(mc_infotext.creative_setup({
            mc_infotext.CREATIVE_MODE: "True",
            mc_infotext.CREATIVE_SOURCE: "a quiet street",
            mc_infotext.SPATIAL_MODE: "True",
            mc_infotext.SPATIAL_VERSION: "99",
            mc_infotext.SPATIAL_LAYOUT: document(),
        }))
        try:
            _prompt, _enabled, status, _view, layout, *_rest = \
                creative_script._restore_setup(False)
        finally:
            mc_creative_krea.pasted.clear()

        assert "later version" in status
        assert layout == {}  # gr.update() with nothing in it: leave it as it is

    def test_the_record_is_readable_before_anything_is_restored(self, pasted, store):
        import model_chain_krea_creative as creative_script

        said = creative_script._pasted_view()

        assert "2 regions" in said or "1 region, 1 text region" in said
        assert "MIDNIGHT CAFE" in said


# --------------------------------------------------------------------------- #
# Off is off
# --------------------------------------------------------------------------- #


class TestOffChangesNothing:
    def test_the_layout_is_not_parsed_when_the_feature_is_off(self, script, client,
                                                              store):
        client.answers = ["A rainy street."]
        p = generate(script, spatial_on=False, layout=document())

        assert p.prompt == "A rainy street."
        assert len(client.calls) == 1

    def test_smart_with_an_empty_canvas_makes_no_second_request(self, script, client,
                                                                store):
        """Smart is the default compose mode, and it has to cost nothing on a
        fresh install: there is nothing for pass 2 to reconcile the scene
        with."""
        client.answers = ["A rainy street."]
        generate(script, spatial_on=True, compose="smart", layout="")

        assert len(client.calls) == 1

    def test_the_two_passes_measure_themselves_separately(self):
        """Sharing one set of keys would teach the bar that a reply is the
        average of a four-hundred-character prompt and a hundred-character
        edit, and then predict neither."""
        import mc_llm_progress

        assert mc_llm_progress.WRITER.reply_key != mc_llm_progress.COMPOSER.reply_key
        assert mc_llm_progress.WRITER.read_key != mc_llm_progress.COMPOSER.read_key
