"""Literal commands: the boundary between text a model may rewrite and text it may not.

The feature's whole claim is one sentence -- ``[[...]]`` reaches the image model
unchanged and reaches nothing else at all -- and every test here is one of the
ways that claim can quietly stop being true.

The first is leakage *forwards*: a payload that reaches the Creative Writer, the
Spatial Composer, or a Stage 2 model. Each of those is silent. A writer handed
``<lora:krea2_edit:1>`` produces a perfectly good paragraph with the tag reworded
out of existence, and nobody finds out until the reference edit does not happen.

The second is leakage *sideways*: a command written inside Region 1 that turns up
in the global scene, in Region 2, or in the background. The picture that comes
back is plausible and wrong, and the elements array is where you would have to
look to see it.

The third is arithmetic: inserted twice, dropped on a fallback path, or reordered.
Every failure path in the pipeline is exercised here for exactly that -- a
Composer that will not answer, a compositor that raises, a writer that fails --
because "restored exactly once" is a property of the assembly step and not of any
one route through it.

The fourth is the payload stopping being opaque. ModelSwitch must not know what a
LoRA tag is, and a test that asserts it can find one is a test that has quietly
approved the opposite.
"""

from __future__ import annotations

import json
import threading

import pytest

import mc_broker
import mc_creative_krea
import mc_infotext
import mc_llm_paths
import mc_llm_sessions as sessions
import mc_lora
from prompt_master.krea import director, literals, spatial
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
        self.calls.append({"messages": messages, "seed": seed})
        answer = self.answers.pop(0) if self.answers else "An expanded Krea prompt."
        on_text(answer)
        return answer

    @property
    def everything(self) -> str:
        """Every byte any pass was sent, for the assertion that matters most."""
        return "\n".join(message["content"] for call in self.calls
                         for message in call["messages"])

    def turn(self, index=-1) -> str:
        return self.calls[index]["messages"][-1]["content"]


class Processing:
    """The half of a StableDiffusionProcessing this feature touches."""

    def __init__(self, prompt="a quiet street", negative="", width=1024, height=1344):
        self.prompt = prompt
        self.negative_prompt = negative
        self.width = width
        self.height = height
        self.extra_generation_params = {}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def client(monkeypatch, host, store):
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
    fake = FakeClient()
    monkeypatch.setattr(sessions, "_client",
                        lambda needs_vision=False, reserve=0, role='': fake)
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


def document(regions=(), mode="direct", auto=True) -> str:
    return json.dumps({"version": 1,
                       "canvas": {"width": 1024, "height": 1344, "grid": "thirds"},
                       "compose_mode": mode,
                       "auto_position_hint": auto,
                       "regions": list(regions)})


def panel_values(creativity=10, seed=director.RANDOM_SEED, anti=True,
                 mode=director.NATURAL, spatial_on=False, compose="direct", layout="",
                 literal_positive=None, literal_negative=None):
    """What Forge hands ``before_process`` after the enabled flag.

    The two Literal Prompt boxes sit in the variable middle, immediately before
    the Spatial tail -- mc_plan reads that tail off the end, so the ends stay
    the ends. Leaving them at ``None`` sends the shape a caller built before
    the boxes existed, which is a thing worth being able to do from here: it is
    what an older API request looks like, and :func:`_split` has to keep cutting
    it in exactly the places it always did.
    """
    values = [creativity, seed, anti]
    for _key in library_module.library().axis_keys:
        values.extend([mode, None, []])
    if literal_positive is not None or literal_negative is not None:
        values.extend([literal_positive or "", literal_negative or ""])
    values.extend([spatial_on, compose, layout])
    return values


def generate(script, prompt="a quiet street", enabled=True, negative="", timeout=20.0,
             **panel):
    """One press of Generate, on a thread with a deadline.

    The deadline is the assertion a hang would otherwise fail to make: a hook
    that waits for the image job it is part of hangs the run rather than
    failing it.
    """
    p = Processing(prompt, negative=negative)
    error: list[BaseException] = []

    def press():
        try:
            script.before_process(p, enabled, *panel_values(**panel))
        except BaseException as exc:  # surfaced on the calling thread below
            error.append(exc)

    worker = threading.Thread(target=press, name="press-generate", daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "before_process did not return"
    if error:
        raise error[0]
    return p


SHIRT = {"id": "r1", "name": "Shirt", "type": "obj", "bbox": [120, 250, 470, 720],
         "prompt": "[[Her shirt from image 1]]", "z": 0}

HAT = {"id": "r2", "name": "Hat", "type": "obj", "bbox": [600, 100, 900, 380],
       "prompt": "[[hat from image 1]]", "z": 1}


def elements(p) -> list[dict]:
    return json.loads(p.prompt)["compositional_deconstruction"]["elements"]


def descriptions(p) -> str:
    """Every element's ``desc``, joined, for "did this land in the wrong box"."""
    return "\n".join(element["desc"] for element in elements(p))


# --------------------------------------------------------------------------- #
# The grammar
# --------------------------------------------------------------------------- #


class TestTheGrammar:
    def test_a_bare_command_is_a_prefix(self):
        parsed = literals.parse("[[<lora:krea2_edit:1>]] a portrait")

        assert parsed.clean_text == "a portrait"
        assert parsed.prefixes == ("<lora:krea2_edit:1>",)
        assert parsed.suffixes == ()

    def test_the_signs_choose_a_side(self):
        parsed = literals.parse("+[[before]] scene -[[after]]")

        assert parsed.clean_text == "scene"
        assert parsed.prefixes == ("before",)
        assert parsed.suffixes == ("after",)

    def test_source_order_survives_the_classification(self):
        """Acceptance test B, and the only ordering rule the grammar has."""
        parsed = literals.parse("-[[D]] +[[A]] [[B]] -[[E]] +[[C]] scene")

        assert parsed.clean_text == "scene"
        assert parsed.prefixes == ("A", "B", "C")
        assert parsed.suffixes == ("D", "E")
        assert literals.restore("a written scene", parsed) == "A B C a written scene D E"

    def test_a_payload_is_one_opaque_run_of_characters(self):
        """Acceptance test C. Three syntaxes, three owners, one command.

        The moment this splits into three, ModelSwitch has an opinion about
        which extension owns what -- and is wrong about the next one somebody
        installs.
        """
        parsed = literals.parse(
            "[[the woman in image 1 is smiling, <lora:test:1>, __face__]]")

        assert len(parsed.commands) == 1
        assert parsed.commands[0].payload == (
            "the woman in image 1 is smiling, <lora:test:1>, __face__")

    def test_the_outer_whitespace_goes_and_the_inner_whitespace_stays(self):
        assert literals.parse("[[  foo,   bar  ]]").prefixes == ("foo,   bar",)

    def test_the_first_close_ends_the_command(self):
        """No nesting in version 1, and no escape syntax invented to allow it."""
        parsed = literals.parse("[[a [[b]] c]]")

        assert parsed.prefixes == ("a [[b",)
        assert parsed.clean_text == "c]]"

    def test_an_unterminated_command_is_left_exactly_as_typed(self):
        """Acceptance test Q. A warning, and not one character removed."""
        parsed = literals.parse("[[unclosed command")

        assert parsed.clean_text == "[[unclosed command"
        assert parsed.commands == ()
        assert parsed.warnings and "never closed" in parsed.warnings[0]

    def test_an_empty_command_is_dropped_and_reported(self):
        parsed = literals.parse("[[]] a portrait")

        assert parsed.clean_text == "a portrait"
        assert parsed.commands == ()
        assert parsed.warnings

    def test_the_hole_a_lifted_command_leaves_is_closed(self):
        """A prompt that became ", , b" is not a prompt that lost nothing."""
        assert literals.parse("a, [[x]], b").clean_text == "a, b"
        assert literals.parse("[[x]]").clean_text == ""
        assert literals.parse("[[x]],").clean_text == ""

    def test_a_prompt_with_no_command_is_returned_byte_for_byte(self):
        """Off is off. No strip, no tidy, no normalisation, nothing.

        The writer's user turn has to be the same bytes it was before this
        feature existed, or llama.cpp's prompt cache misses on every generation
        by everybody who never types a bracket.
        """
        source = "  a quiet street,, with  two spaces  \n\n\n and blank lines  "

        assert literals.parse(source).clean_text == source
        assert literals.present(source) is False

    def test_a_sign_needs_to_be_touching_the_brackets(self):
        """``2 + 2 [[x]]`` is arithmetic, not a signed command."""
        parsed = literals.parse("2 + 2 [[x]]")

        assert parsed.clean_text == "2 + 2"
        assert parsed.prefixes == ("x",)


# --------------------------------------------------------------------------- #
# Fields nobody typed brackets for
# --------------------------------------------------------------------------- #


class TestTheConvenienceFields:
    """Two ordinary text boxes that behave exactly like a bracket.

    The Literal Prompts UX exists so somebody can protect a LoRA tag without
    learning a syntax for it. The way that could go wrong is by becoming a
    second implementation -- a parser that reads the field, an assembly step
    that concatenates it, a rule about where it lands that is *nearly* the rule
    the brackets follow.

    So the field becomes a LiteralCommand and joins the sidecar the parser
    already produced, and every test below is a way of asking whether anything
    downstream can still tell the two apart.
    """

    def test_a_field_becomes_one_opaque_command(self):
        """One field, one payload. Not split on commas, newlines or anything
        else: deciding where to cut it would be interpreting it."""
        made = literals.command("<lora:test:1>, __face__\nsecond line", literals.PREFIX)

        assert made.payload == "<lora:test:1>, __face__\nsecond line"
        assert made.placement == literals.PREFIX
        assert made.scope == literals.GLOBAL

    def test_an_empty_field_is_no_command_at_all(self):
        assert literals.command("", literals.PREFIX) is None
        assert literals.command("   \n  ", literals.SUFFIX) is None
        assert literals.command(None, literals.PREFIX) is None

    def test_the_edges_are_trimmed_and_the_middle_is_not(self):
        made = literals.command("  foo,   bar  ", literals.PREFIX)

        assert made.payload == "foo,   bar"

    def test_literal_positive_lands_before_the_body(self):
        """Section 3.1, in the form the acceptance checklist states it."""
        merged = literals.merge(literals.parse("portrait of a woman"),
                                before="<lora:realfilter:1>")

        assert literals.restore(merged.clean_text, merged) == (
            "<lora:realfilter:1> portrait of a woman")

    def test_literal_negative_lands_after_the_body(self):
        """Section 3.2. It is the suffix side of the protected positive prompt
        and has nothing to do with Forge's own Negative Prompt."""
        merged = literals.merge(literals.parse("portrait of a woman"),
                                after="blue hat")

        assert literals.restore(merged.clean_text, merged) == (
            "portrait of a woman blue hat")

    def test_typed_syntax_outranks_a_field_on_both_sides(self):
        """Section 4's worked example, which is the whole priority rule in one
        line: explicit commands sit further from the body than the fields do,
        so adding a field cannot move something already placed by hand."""
        parsed = literals.parse("+[[A]] scene description -[[D]]")
        merged = literals.merge(parsed, before="B", after="C")

        assert merged.clean_text == "scene description"
        assert literals.restore("<final body>", merged) == "A B <final body> C D"

    def test_explicit_source_order_survives_the_merge(self):
        parsed = literals.parse("-[[D]] +[[A]] [[B]] -[[E]] +[[C]] scene")
        merged = literals.merge(parsed, before="P", after="S")

        assert merged.prefixes == ("A", "B", "C", "P")
        assert merged.suffixes == ("S", "D", "E")

    def test_two_empty_fields_change_nothing_at_all(self):
        """Identity, not equality. Almost every generation takes this path, and
        the parse it was given has to come back out of it untouched."""
        parsed = literals.parse("a portrait")

        assert literals.merge(parsed, "", "") is parsed

    def test_a_field_works_on_a_prompt_that_has_no_brackets_in_it(self):
        """The ordinary case the feature was built for: somebody who has never
        typed a bracket and never will."""
        merged = literals.merge(literals.parse("just words"), before="X", after="Y")

        assert merged.clean_text == "just words"
        assert literals.restore(merged.clean_text, merged) == "X just words Y"

    def test_a_field_adds_nothing_to_the_clean_text(self):
        """The boundary, stated as arithmetic. A field was never in the prompt
        body, so there is nothing about it for the writer to see."""
        parsed = literals.parse("+[[A]] scene -[[D]]")
        merged = literals.merge(parsed, before="secret", after="also secret")

        assert merged.clean_text == parsed.clean_text == "scene"
        assert "secret" not in merged.clean_text

    def test_a_field_carries_no_warning_of_its_own(self):
        parsed = literals.parse("[[unclosed")
        merged = literals.merge(parsed, before="B")

        assert merged.warnings == parsed.warnings

    def test_a_region_field_is_scoped_to_its_region(self):
        """Section 6. A region's literals reach that element or nothing."""
        merged = literals.merge(literals.EMPTY, before="prefix", after="suffix",
                                scope=literals.REGION, region_id="r3")

        assert all(entry.scope == literals.REGION for entry in merged.commands)
        assert {entry.region_id for entry in merged.commands} == {"r3"}

    def test_merging_onto_nothing_is_allowed(self):
        """A caller with no parse at hand -- a region whose prompt box is empty
        and whose literal fields are not."""
        merged = literals.merge(None, before="X")

        assert merged.prefixes == ("X",)
        assert merged.clean_text == ""

    def test_a_field_payload_is_as_opaque_as_a_bracketed_one(self):
        """The fourth kind of leakage this file watches for: the moment a field
        is treated as a special sort of payload, the two paths have diverged
        and one of them will grow an opinion about LoRA tags."""
        source = "text-inversion, <lora:x:1>, __wild__, $style"
        typed = literals.parse(f"[[{source}]]").commands[0]
        field = literals.command(source, literals.PREFIX)

        assert typed.payload == field.payload
        assert typed.placement == field.placement


class TestTheFieldsReachTheGeneration:
    """The two boxes, driven the way Forge drives them.

    :class:`TestTheConvenienceFields` proves the merge is right about text.
    This proves the hook actually performs it -- that the value in the box on
    screen is the value that reaches the image model, on every one of the paths
    a generation can take out of ``before_process``.
    """

    def test_a_field_reaches_stage_one_with_neither_feature_on(self, script, store,
                                                               host):
        """Acceptance: *both fields still affect generation when Creative and
        Spatial are OFF*. No language model runs on this path at all, which is
        exactly why it has to work -- protection is about delivery, not about
        anything having been protected from."""
        p = generate(script, "portrait of a woman", enabled=False,
                     literal_positive="<lora:realfilter:1>",
                     literal_negative="blue hat")

        assert p.prompt == "<lora:realfilter:1> portrait of a woman blue hat"

    def test_the_writer_is_never_shown_a_field(self, script, client, store, host):
        """The boundary the whole feature rests on, asked of the new fields.

        ``client.everything`` is every byte every pass was sent, which is the
        assertion worth making here rather than the last user turn alone.
        """
        generate(script, "a quiet street",
                 literal_positive="<lora:secret_prefix:1>",
                 literal_negative="__secret_suffix__")

        assert "secret_prefix" not in client.everything
        assert "secret_suffix" not in client.everything

    def test_the_field_wraps_what_the_writer_wrote(self, script, client, store, host):
        p = generate(script, "a quiet street",
                     literal_positive="<lora:realfilter:1>",
                     literal_negative="__grain__")

        assert p.prompt.startswith("<lora:realfilter:1> ")
        assert p.prompt.endswith(" __grain__")
        assert "a quiet street" not in p.prompt or p.prompt.count("realfilter") == 1

    def test_typed_syntax_still_outranks_the_fields_end_to_end(self, script, store,
                                                               host):
        """Section 4's example, all the way through the hook rather than through
        the merge alone."""
        p = generate(script, "+[[A]] scene description -[[D]]", enabled=False,
                     literal_positive="B", literal_negative="C")

        assert p.prompt == "A B scene description C D"

    def test_a_field_is_restored_exactly_once(self, script, client, store, host):
        p = generate(script, "a quiet street", literal_positive="ONCE")

        assert p.prompt.count("ONCE") == 1

    def test_stage_two_never_inherits_a_field(self, script, client, store, host):
        """A Stage 1 filter LoRA is meaningless to a Stage 2 model, and the
        field is no more inheritable for having been typed without brackets."""
        p = generate(script, "a quiet street",
                     literal_positive="<lora:stage_one_only:1>")

        inheritable = (p.extra_generation_params or {}).get("Model Chain Inheritable Prompt", "")
        assert "stage_one_only" not in str(inheritable)

    def test_an_empty_field_changes_nothing(self, script, store, host):
        """Off is off. A prompt with no brackets and two empty boxes has to
        reach the model as the bytes it always did."""
        p = generate(script, "a quiet street", enabled=False,
                     literal_positive="", literal_negative="")

        assert p.prompt == "a quiet street"

    def test_a_caller_that_predates_the_fields_is_cut_where_it_always_was(
            self, script, store, host):
        """The older argument shape, sent verbatim. Its Spatial block has to be
        read from the same place, and its absent fields must not be filled in
        from the end of the tuple."""
        p = generate(script, "[[<lora:x:1>]] a quiet street", enabled=False)

        assert p.prompt == "<lora:x:1> a quiet street"

    def test_the_saved_values_answer_for_a_caller_that_sends_none(self, script,
                                                                  store, host):
        """Section 3.3, in the shape it actually reaches an API request: the
        fields keep working when nothing on screen sent them."""
        import mc_literal_prompts

        mc_literal_prompts.remember(**{mc_literal_prompts.POSITIVE: "<lora:kept:1>"})
        p = generate(script, "a quiet street", enabled=False)

        assert p.prompt == "<lora:kept:1> a quiet street"

    def test_a_field_is_not_split_on_its_commas(self, script, store, host):
        """One field is one payload. Splitting it would be interpreting it, and
        the order of the pieces would then be this extension's opinion."""
        p = generate(script, "a quiet street", enabled=False,
                     literal_positive="<lora:a:1>, __b__, $c")

        assert p.prompt == "<lora:a:1>, __b__, $c a quiet street"


# --------------------------------------------------------------------------- #
# The global scope
# --------------------------------------------------------------------------- #


class TestTheWriterNeverSeesAPayload:
    def test_the_writer_is_given_the_clean_text_only(self, script, client):
        """Acceptance test A."""
        client.answers = ["A cinematic portrait in a modern restaurant."]
        p = generate(script, "[[the woman in image 1 is smiling]]\ncinematic portrait")

        assert "image 1" not in client.everything
        assert "cinematic portrait" in client.turn()
        assert p.prompt == ("the woman in image 1 is smiling "
                            "A cinematic portrait in a modern restaurant.")

    def test_the_director_is_not_locked_by_a_lora_filename(self, client):
        """§12. "anime" in a filename is not the user asking for anime.

        The Director reads the source to notice when the user has already made a
        decision -- type "oil painting" and the Medium axis stays out of the
        brief. A payload it could read would let a *filename* make that
        decision, silently, on every roll.
        """
        recipe = director.roll(
            source=literals.parse("[[<lora:anime_style:1>]] portrait of a woman"
                                  ).clean_text,
            creativity=10, creative_seed=7)

        assert "medium" not in recipe.locked

    def test_a_prompt_that_is_all_command_writes_nothing_and_keeps_it(self, script,
                                                                      client):
        """§14. No transformable text is not an error; it is an empty brief."""
        p = generate(script, "[[<lora:krea2_edit:1>]]")

        assert client.calls == []
        assert p.prompt == "<lora:krea2_edit:1>"

    def test_the_source_recorded_is_the_one_with_the_brackets_on(self, script, client):
        """Acceptance test R, the global half. A restore hands back what was typed."""
        generate(script, "[[<lora:krea2_edit:1>]] a quiet street")
        setup = mc_creative_krea.pasted
        recorded = mc_creative_krea.prepare(
            mc_creative_krea.creative.last).metadata

        assert recorded[mc_infotext.CREATIVE_SOURCE] == (
            "[[<lora:krea2_edit:1>]] a quiet street")
        assert setup is not None  # the module-level slot exists; nothing pasted here

    def test_the_count_is_recorded_and_the_payloads_are_not(self, script, client):
        p = generate(script, "[[<lora:krea2_edit:1>]] -[[__grain__]] a quiet street")
        recorded = p.extra_generation_params

        assert recorded[mc_infotext.LITERAL_COUNT] == 2
        assert recorded[mc_infotext.LITERAL_VERSION] == literals.SYNTAX_VERSION
        assert not any("lora" in str(value) for key, value in recorded.items()
                       if key != mc_infotext.CREATIVE_SOURCE)

    def test_an_ordinary_generation_records_no_literal_keys(self, script, client):
        p = generate(script, "a quiet street")

        assert mc_infotext.LITERAL_COUNT not in p.extra_generation_params
        assert mc_infotext.LITERAL_VERSION not in p.extra_generation_params


class TestWithNeitherFeatureOn:
    """The syntax means the same thing whether or not a language model runs.

    This is the property that makes it teachable. A user who switches Creative
    Mode off for one image must not discover that their reference instruction
    has started reaching the text encoder with its brackets on.
    """

    def test_the_brackets_come_off_anyway(self, script, client):
        p = generate(script, "[[<lora:krea2_edit:1>]] a quiet street", enabled=False)

        assert p.prompt == "<lora:krea2_edit:1> a quiet street"
        assert client.calls == []

    def test_the_order_is_the_same_order(self, script, client):
        p = generate(script, "-[[D]] +[[A]] [[B]] scene", enabled=False)

        assert p.prompt == "A B scene D"

    def test_a_prompt_with_no_command_is_not_touched_at_all(self, script, client):
        p = generate(script, "a quiet street,  with  odd   spacing", enabled=False)

        assert p.prompt == "a quiet street,  with  odd   spacing"
        assert p.extra_generation_params == {}

    def test_the_negative_prompt_is_unwrapped_too(self, script, client):
        """No language model has ever seen it; what it needs is the same syntax."""
        p = generate(script, "a quiet street", enabled=False,
                     negative="[[__bad_hands__]] blurry")

        assert p.negative_prompt == "__bad_hands__ blurry"

    def test_a_negative_prompt_with_no_command_is_not_touched(self, script, client):
        p = generate(script, "a quiet street", enabled=False, negative="blurry,  ugly")

        assert p.negative_prompt == "blurry,  ugly"


# --------------------------------------------------------------------------- #
# The BBOX scope
# --------------------------------------------------------------------------- #


class TestARegionKeepsItsOwn:
    def test_a_literal_only_region_is_a_valid_region(self):
        """Acceptance test F. It is not an empty region; it is a region whose
        content was deliberately removed from LLM-visible text."""
        layout = spatial.parse(document([SHIRT]))

        assert len(layout.regions) == 1
        assert layout.regions[0].prompt == ""
        assert layout.regions[0].has_content
        assert layout.regions[0].prefix_literals == ("Her shirt from image 1",)

    def test_a_region_with_no_content_at_all_is_still_skipped(self):
        layout = spatial.parse(document([
            {"id": "r1", "bbox": [10, 10, 200, 200], "prompt": ""}]))

        assert layout.regions == ()
        assert layout.notes

    def test_the_desc_wraps_the_users_words_and_keeps_the_hint_order(self):
        """Acceptance test E: prefix, prompt, existing hints, suffix."""
        layout = spatial.parse(document([
            {"id": "r1", "bbox": [120, 250, 470, 720], "framing": "Medium shot",
             "prompt": "+[[Her shirt from image 1]]\nred satin blouse\n"
                       "-[[__fabric_detail__]]"}]))

        assert layout.regions[0].describe() == (
            "Her shirt from image 1, red satin blouse, shown as a medium shot, "
            "positioned in the center-left area, occupying a medium-sized area, "
            "__fabric_detail__")

    def test_the_composer_is_never_shown_a_regions_payload(self, script, client):
        """Acceptance test D. The Composer is a copy-editor; this is not copy."""
        client.answers = ["A written scene.",
                          '{"scene": "A reconciled scene.", "background": "a room"}']
        p = generate(script, "cinematic editorial portrait", spatial_on=True,
                     compose="smart", layout=document([SHIRT], mode="smart"))

        assert len(client.calls) == 2
        assert "Her shirt" not in client.everything
        assert "Her shirt from image 1" in elements(p)[0]["desc"]

    def test_two_regions_keep_their_commands_apart(self, script, client):
        """Acceptance test N."""
        p = generate(script, "a quiet street", spatial_on=True,
                     layout=document([SHIRT, HAT]))
        first, second = elements(p)

        assert "shirt from image 1" in first["desc"]
        assert "hat" not in first["desc"]
        assert "hat from image 1" in second["desc"]
        assert "shirt" not in second["desc"]

    def test_a_regions_command_never_reaches_the_global_scene(self, script, client):
        client.answers = ["A written scene."]
        p = generate(script, "a quiet street", spatial_on=True,
                     layout=document([SHIRT]))
        written = json.loads(p.prompt)

        assert "shirt" not in written["high_level_description"]
        assert "shirt" not in written["compositional_deconstruction"]["background"]

    def test_the_canvas_records_what_the_user_typed(self):
        """Acceptance test R, the layout half. Restore returns the syntax."""
        layout = spatial.parse(document([SHIRT]))
        again = json.loads(layout.serialize())["regions"][0]["prompt"]

        assert again == "[[Her shirt from image 1]]"

    def test_a_regions_malformed_command_is_a_note_and_not_a_lost_box(self):
        layout = spatial.parse(document([
            {"id": "r1", "bbox": [10, 10, 200, 200], "prompt": "[[unclosed"}]))

        assert len(layout.regions) == 1
        assert layout.regions[0].prompt == "[[unclosed"
        assert any("never closed" in note for note in layout.notes)


class TestARegionsOwnLiteralFields:
    """Section 6: the same two boxes, per region, in the full editor only.

    A region's literals have always been the sideways-leakage risk -- a command
    written inside Region 1 that turns up in the global scene or in Region 2
    produces a plausible picture that is wrong, and the elements array is the
    only place you could see it. Adding a second way to author them adds a
    second way for that to happen, so these ask the same questions the
    bracketed ones are asked.
    """

    def region(self, prompt="astronaut holding a flower", prefix="", suffix=""):
        found = {"id": "r1", "name": "Sub", "type": "obj",
                 "bbox": [120, 250, 470, 720], "prompt": prompt, "z": 0}
        if prefix:
            found["literal_prefix"] = prefix
        if suffix:
            found["literal_suffix"] = suffix
        return found

    def test_a_region_field_wraps_that_region_s_description(self, script, store,
                                                            host):
        p = generate(script, "a quiet street", enabled=False, spatial_on=True,
                     layout=document(regions=[
                         self.region(prefix="<lora:a:1>", suffix="__grain__")],
                         auto=False))
        said = descriptions(p)

        assert said == "<lora:a:1>, astronaut holding a flower, __grain__"

    def test_typed_syntax_outranks_a_region_field_on_both_sides(self, script,
                                                                store, host):
        """The ordering rule is the global one, applied to a region."""
        p = generate(script, "a quiet street", enabled=False, spatial_on=True,
                     layout=document(regions=[
                         self.region(prompt="+[[A]] astronaut -[[D]]",
                                     prefix="B", suffix="C")],
                         auto=False))

        assert descriptions(p) == "A, B, astronaut, C, D"

    def test_a_region_field_stays_out_of_the_global_scene(self, script, store,
                                                          host):
        """Sideways leakage, asked of the new authoring path."""
        p = generate(script, "a quiet street", enabled=False, spatial_on=True,
                     layout=document(regions=[
                         self.region(prefix="<lora:region_only:1>")], auto=False))
        payload = json.loads(p.prompt)

        assert "region_only" not in str(payload.get("high_level_description", ""))
        assert "region_only" in descriptions(p)

    def test_a_region_field_stays_out_of_the_other_region(self, script, store,
                                                          host):
        other = {"id": "r2", "name": "Lamp", "type": "obj",
                 "bbox": [600, 100, 900, 380], "prompt": "a lamp", "z": 1}
        p = generate(script, "a quiet street", enabled=False, spatial_on=True,
                     layout=document(regions=[
                         self.region(prefix="<lora:mine:1>"), other], auto=False))
        found = {entry["desc"] for entry in elements(p)}

        assert any("mine" in entry for entry in found)
        assert any("a lamp" in entry and "mine" not in entry for entry in found)

    def test_a_region_field_never_reaches_the_composer(self, script, client,
                                                       store, host):
        """The Spatial Composer is a language model like any other, and a
        region literal is no more visible to it for having been typed in a box
        without brackets."""
        generate(script, "a quiet street", spatial_on=True, compose="smart",
                 layout=document(regions=[
                     self.region(prefix="<lora:unseen:1>")], mode="smart"))

        assert "unseen" not in client.everything

    def test_a_region_that_is_only_a_literal_field_survives(self, script, store,
                                                            host):
        """A box holding nothing but protected text is a box whose content was
        deliberately kept away from the language models, not an empty one.
        Skipping it would delete the region for saying exactly what the feature
        was built to carry."""
        p = generate(script, "a quiet street", enabled=False, spatial_on=True,
                     layout=document(regions=[
                         self.region(prompt="", prefix="[[Her shirt from image 1]]")],
                         auto=False))

        assert len(elements(p)) == 1
        assert "Her shirt from image 1" in descriptions(p)

    def test_stage_two_inherits_no_region_field(self, script, store, host):
        """A Stage 1 reference instruction is meaningless to a Stage 2 model
        with no reference behind it, and the only way to be sure none travels
        is a representation none was ever put into."""
        from prompt_master.krea import spatial as spatial_module

        layout = spatial_module.parse(document(regions=[
            self.region(prefix="<lora:stage_one:1>", suffix="__grain__")],
            auto=False))

        assert "stage_one" not in layout.regions[0].describe(False, literals=False)
        assert "__grain__" not in layout.regions[0].describe(False, literals=False)

    def test_the_fields_round_trip_back_into_the_editor(self, script, store, host):
        """Separately from the prompt box, so reopening the editor cannot move
        a command out of one and into the other."""
        from prompt_master.krea import spatial as spatial_module

        layout = spatial_module.parse(document(regions=[
            self.region(prompt="+[[A]] astronaut", prefix="B", suffix="C")]))
        state = layout.regions[0].state()

        assert state["prompt"] == "+[[A]] astronaut"
        assert state["literal_prefix"] == "B"
        assert state["literal_suffix"] == "C"

    def test_a_region_without_them_serializes_as_it_always_did(self, script, store,
                                                               host):
        from prompt_master.krea import spatial as spatial_module

        layout = spatial_module.parse(document(regions=[self.region()]))
        state = layout.regions[0].state()

        assert "literal_prefix" not in state
        assert "literal_suffix" not in state


class TestGlobalLiteralsInASpatialPrompt:
    """§7.5. The document stays one valid structured prompt.

    A tag loose beside the JSON object would be two semantic texts in one
    prompt, and which of them the structured parser reads is a question about
    Krea's parser that this extension cannot answer for every build. Inside
    ``high_level_description`` the object is still an object, and Forge's own
    extra-network pass strips a tag out of the middle of a JSON *string* without
    the surrounding document ceasing to parse.
    """

    def test_a_global_command_goes_inside_the_scene_field(self, script, client):
        client.answers = ["A written scene."]
        p = generate(script, "[[<lora:krea2_edit:1>]] a quiet street",
                     spatial_on=True, layout=document([SHIRT]))
        written = json.loads(p.prompt)

        assert written["high_level_description"] == (
            "<lora:krea2_edit:1> A written scene.")

    def test_the_prompt_is_still_one_parseable_object(self, script, client):
        p = generate(script, "[[<lora:a:1>]] street -[[__grain__]]", spatial_on=True,
                     layout=document([SHIRT]))

        assert p.prompt.startswith("{") and p.prompt.endswith("}")
        json.loads(p.prompt)  # raises if the object did not survive

    def test_a_global_command_does_not_leak_into_a_region(self, script, client):
        p = generate(script, "[[<lora:a:1>]] street", spatial_on=True,
                     layout=document([SHIRT, HAT]))

        assert "lora" not in descriptions(p)


# --------------------------------------------------------------------------- #
# Exactly once, on every path
# --------------------------------------------------------------------------- #


class TestEveryFallbackRestoresExactlyOnce:
    def counted(self, p, payload="<lora:krea2_edit:1>") -> int:
        return p.prompt.count(payload)

    def test_direct_and_smart_agree(self, script, client):
        """Acceptance tests G and H, stated as the equality they are about."""
        client.answers = ["A written scene."]
        direct = generate(script, "[[<lora:krea2_edit:1>]] street", spatial_on=True,
                          compose="direct", layout=document([SHIRT], mode="direct"))
        client.answers = ["A written scene.",
                          '{"scene": "A written scene.", "background": ""}']
        smart = generate(script, "[[<lora:krea2_edit:1>]] street", spatial_on=True,
                         compose="smart", layout=document([SHIRT], mode="smart"))

        assert self.counted(direct) == 1
        assert self.counted(smart) == 1
        assert descriptions(direct) == descriptions(smart)

    def test_a_composer_that_fails_still_restores_once(self, script, client):
        """Acceptance test G. Direct merge answers, with everything in it."""
        client.answers = ["A written scene.", "not json at all"]
        p = generate(script, "[[<lora:krea2_edit:1>]] street", spatial_on=True,
                     compose="smart", layout=document([SHIRT], mode="smart"))

        assert self.counted(p) == 1
        assert descriptions(p).count("Her shirt from image 1") == 1

    def test_a_writer_that_fails_still_restores_once(self, script, client, monkeypatch):
        def refuse(*args, **kwargs):
            raise RuntimeError("llama-server would not start")

        monkeypatch.setattr(sessions, "_client", refuse)
        p = generate(script, "[[<lora:krea2_edit:1>]] street")

        assert p.prompt == "<lora:krea2_edit:1> street"

    def test_a_writer_that_fails_with_boxes_behind_it_restores_once(self, script,
                                                                    client, monkeypatch):
        def refuse(*args, **kwargs):
            raise RuntimeError("llama-server would not start")

        monkeypatch.setattr(sessions, "_client", refuse)
        p = generate(script, "[[<lora:krea2_edit:1>]] street", spatial_on=True,
                     layout=document([SHIRT]))

        assert self.counted(p) == 1
        assert descriptions(p).count("Her shirt from image 1") == 1

    def test_a_compositor_that_raises_still_restores_the_global_ones(
            self, script, client, monkeypatch):
        from prompt_master.krea import spatial as spatial_module

        client.answers = ["A written scene."]
        monkeypatch.setattr(spatial_module, "compose",
                            lambda *args, **kwargs: 1 / 0)
        p = generate(script, "[[<lora:krea2_edit:1>]] street", spatial_on=True,
                     layout=document([SHIRT]))

        assert p.prompt == "<lora:krea2_edit:1> A written scene."

    def test_a_checkpoint_that_refuses_the_layout_still_restores_them(
            self, script, client, monkeypatch):
        """The layout is refused; the prompt syntax the user typed is not."""
        monkeypatch.setattr(mc_creative_krea, "checkpoint_objection",
                            lambda: "that is not a Krea 2 checkpoint")
        p = generate(script, "[[<lora:a:1>]] street", enabled=False, spatial_on=True,
                     layout=document([SHIRT]))

        assert p.prompt == "<lora:a:1> street"


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #


class TestStageTwoNeverInheritsALiteral:
    """§9. A Stage 1 command is a statement about Stage 1's pipeline.

    Stage 2 may be a different architecture with a different text encoder, no
    ImageStitch references at all, and none of Stage 1's LoRAs. Inheriting
    ``[[<lora:krea2_edit:1>]]`` there is applying an edit LoRA to a model that
    has never seen the thing it edits against.
    """

    def test_the_inheritable_prompt_is_the_one_without_them(self, script, client):
        """Acceptance test I."""
        client.answers = ["A written scene."]
        p = generate(script, "[[<lora:krea2_edit:1>]]\n[[woman from image 1 smiling]]\n"
                             "scene")
        positive, _ = mc_lora.stage1_inheritable(p)

        assert "lora" not in positive
        assert "image 1" not in positive
        assert positive == "A written scene."
        assert p.prompt != positive

    def test_it_is_built_and_not_searched_for(self, script, client):
        """Acceptance test J. The writer's own words are never collateral.

        ``[[red hat]]`` over a scene the writer independently gave a red hat is
        the case a string-based cleanup cannot survive: it cannot tell which
        occurrence came from the sidecar, and removing both loses the writer's
        sentence.
        """
        client.answers = ["A portrait of a man in a red hat."]
        p = generate(script, "[[red hat]] portrait with a red hat")
        positive, _ = mc_lora.stage1_inheritable(p)

        assert positive == "A portrait of a man in a red hat."
        assert p.prompt == "red hat A portrait of a man in a red hat."

    def test_a_regions_command_does_not_travel_either(self, script, client):
        client.answers = ["A written scene."]
        p = generate(script, "street", spatial_on=True, layout=document([SHIRT]))
        positive, _ = mc_lora.stage1_inheritable(p)

        assert "image 1" not in positive
        assert "Her shirt" not in positive
        # Still the structured document, still every box in it.
        assert len(json.loads(positive)["compositional_deconstruction"]
                   ["elements"]) == 1

    def test_the_negative_half_is_isolated_the_same_way(self, script, client):
        p = generate(script, "street", enabled=False, negative="[[__bad_hands__]] blurry")
        _, negative = mc_lora.stage1_inheritable(p)

        assert negative == "blurry"
        assert p.negative_prompt == "__bad_hands__ blurry"

    def test_a_generation_nobody_rewrote_says_nothing(self, script, client):
        """Two empty strings, so Stage 2 reads ``all_prompts`` as it always did."""
        p = generate(script, "a quiet street", enabled=False)

        assert mc_lora.stage1_inheritable(p) == ("", "")


class TestTheChainReadsIt:
    """The other end of the same wire, in Model Chain's own prompt assembly."""

    def test_inherit_uses_the_isolated_prompt(self):
        import model_chain

        p = Processing("<lora:krea2_edit:1> A written scene.")
        mc_lora.remember_inheritable(p, "A written scene.", "")
        script = model_chain.ScriptModelChain()

        positive, _ = script._resolve_prompts(
            "<lora:krea2_edit:1> A written scene.", "", "Inherit", "", "", [])
        chosen, _ = mc_lora.stage1_inheritable(p)

        assert chosen == "A written scene."
        # And what the ordinary path would have produced from the model prompt:
        # stripped of the tag by pattern, which is the defence-in-depth half.
        assert positive == "A written scene."

    def test_the_pattern_stripper_is_still_there_underneath(self):
        """A bare tag typed outside a command is still a Stage 1 tag."""
        cleaned, dropped = mc_lora.strip_networks("a street <lora:film:0.8>")

        assert cleaned == "a street"
        assert dropped == ["<lora:film:0.8>"]


# --------------------------------------------------------------------------- #
# What ModelSwitch does not do
# --------------------------------------------------------------------------- #


class TestTheFeatureCostsNothing:
    def test_it_never_invalidates_a_prepared_lora_state(self, script, client,
                                                        monkeypatch):
        """Acceptance tests K and L. Forge owns the LoRA; this owns the delivery.

        A literal command existing is not news to the LoRA loader. If the
        effective configuration is unchanged, Forge's own early return keeps the
        prepared state; if it changed, Forge notices for itself. Either way
        nothing here is entitled to force a rebuild.
        """
        invalidated = []
        monkeypatch.setattr(mc_lora, "invalidate",
                            lambda *args, **kwargs: invalidated.append(args))
        generate(script, "[[<lora:test:1>]] street")
        generate(script, "[[<lora:test:0.8>]] street")

        assert invalidated == []

    def test_it_does_not_parse_the_payload(self, script, client, monkeypatch):
        """Acceptance test C, as a statement about the code rather than the text.

        The extra-network pattern is Forge's business and Model Chain's Stage 2
        isolation. Nothing on the Stage 1 literal path may consult it: the day
        it does, ``[[<custom-extension:foo>]]`` starts being treated as
        something this extension has an opinion about.
        """
        looked = []

        class Watched:
            def __init__(self, pattern):
                self._pattern = pattern

            def __getattr__(self, name):
                looked.append(name)
                return getattr(self._pattern, name)

        monkeypatch.setattr(mc_lora, "RE_EXTRA_NET",
                            Watched(mc_lora.RE_EXTRA_NET))
        generate(script, "[[<lora:test:1>]] street")

        assert looked == []

    def test_a_region_command_is_not_regional_and_the_docs_say_so(self):
        """§10.1. Scope here is about surviving the LLM, not about diffusion.

        A LoRA tag inside a box reaches the prompt inside that element's desc,
        and Forge then applies it globally, because Forge's extra-network system
        has its own scope and this syntax does not change it. The assertion is
        that nothing pretends otherwise -- there is no per-region LoRA path.
        """
        layout = spatial.parse(document([
            {"id": "r1", "bbox": [10, 10, 500, 500],
             "prompt": "[[<lora:shirt_style:1>]]"}]))

        assert layout.regions[0].prefix_literals == ("<lora:shirt_style:1>",)
        assert not hasattr(layout.regions[0], "loras")


# --------------------------------------------------------------------------- #
# What an image records about its Literal Prompt boxes
# --------------------------------------------------------------------------- #


@pytest.fixture
def built(store, host):
    """A script with its UI assembled, so its paste fields are registered."""
    import model_chain_krea_creative as creative_script

    instance = creative_script.ScriptKreaCreative()
    instance.ui(False)
    return instance


def pasted_value(script, name, params):
    """What the paste field for ``name`` answers for this infotext."""
    component = script.components[name]
    for entry in script.infotext_fields:
        if entry.component is not component:
            continue
        if entry.function is not None:
            return entry.function(params)
        return params.get(entry.label)
    raise AssertionError(f"no paste field for {name}")


class TestTheFieldsAreRecorded:
    """Section 10. The one place a literal payload gets a key of its own.

    A bracketed command is already in the image twice -- restored into the
    ``Prompt:`` line, and with its brackets still on in the recorded source --
    so a third copy would repeat the file. A field's text is in neither: the
    prompt line has it unbracketed and indistinguishable from the words around
    it, and the source line never had it. Without these keys the authoring
    setup could not be reconstructed, which is exactly what section 10 asks
    for.
    """

    def test_both_fields_reach_the_metadata(self, script, store, host):
        p = generate(script, "a quiet street", enabled=False,
                     literal_positive="<lora:realfilter:1>",
                     literal_negative="blue hat")

        assert p.extra_generation_params[mc_infotext.LITERAL_POSITIVE] == \
            "<lora:realfilter:1>"
        assert p.extra_generation_params[mc_infotext.LITERAL_NEGATIVE] == "blue hat"

    def test_an_empty_box_records_nothing(self, script, store, host):
        """An ordinary image should say nothing about a feature it did not
        use."""
        p = generate(script, "a quiet street", enabled=False,
                     literal_positive="", literal_negative="")

        assert mc_infotext.LITERAL_POSITIVE not in p.extra_generation_params
        assert mc_infotext.LITERAL_NEGATIVE not in p.extra_generation_params

    def test_only_the_box_that_was_used_is_recorded(self, script, store, host):
        p = generate(script, "a quiet street", enabled=False,
                     literal_positive="<lora:x:1>")

        assert mc_infotext.LITERAL_POSITIVE in p.extra_generation_params
        assert mc_infotext.LITERAL_NEGATIVE not in p.extra_generation_params

    def test_the_prompt_line_still_describes_what_stage_one_was_given(
            self, script, store, host):
        """Section 10 again: the recorded Prompt keeps meaning the prompt that
        was actually delivered, keys or no keys."""
        p = generate(script, "a quiet street", enabled=False,
                     literal_positive="<lora:x:1>")

        assert p.prompt == "<lora:x:1> a quiet street"

    def test_a_field_that_ran_with_no_feature_on_still_records(self, script, store,
                                                               host):
        """The path with no language model in it records the fields the same
        way, because the fields did the same thing on it."""
        p = generate(script, "a quiet street", enabled=False,
                     literal_positive="<lora:x:1>")

        assert p.extra_generation_params[mc_infotext.LITERAL_POSITIVE] == "<lora:x:1>"


class TestPastingOneBack:
    """The rule that outranks convenience: exact reproduction.

    A pasted image's ``Prompt:`` already has these payloads restored into it.
    Refilling the boxes as well would insert them a second time and the picture
    would not reproduce -- so a paste *empties* them, for the same reason and by
    the same mechanism that switches Creative Mode off.
    """

    def test_a_paste_of_one_of_our_images_empties_both_boxes(self, built):
        params = {mc_infotext.LITERAL_POSITIVE: "<lora:x:1>"}

        assert pasted_value(built, "literal_positive", params) == ""
        assert pasted_value(built, "literal_negative", params) == ""

    def test_an_image_of_ours_with_empty_boxes_still_empties_them(self, built):
        """It records no literal key at all, and leaving somebody's current
        boxes in place would add text that image never had."""
        params = {mc_infotext.CREATIVE_MODE: "on"}

        assert pasted_value(built, "literal_positive", params) == ""

    def test_a_legacy_image_leaves_them_exactly_as_they_are(self, built):
        """``None`` is how the host is told to leave a control alone. An
        ordinary image should not be able to empty a control any more than it
        can switch a feature off."""
        params = {"Steps": "20", "CFG scale": "7"}

        assert pasted_value(built, "literal_positive", params) is None
        assert pasted_value(built, "literal_negative", params) is None

    def test_the_recorded_values_are_read_back_off_the_infotext(self):
        setup = mc_infotext.creative_setup({
            mc_infotext.LITERAL_POSITIVE: "<lora:x:1>",
            mc_infotext.LITERAL_NEGATIVE: "blue hat"})

        assert setup.literal_positive == "<lora:x:1>"
        assert setup.literal_negative == "blue hat"
        assert setup.literals is True

    def test_literal_fields_alone_are_not_a_creative_record(self):
        """Saying "Creative image restored" over an image that never ran the
        writer would be describing a feature that did not happen."""
        setup = mc_infotext.creative_setup({mc_infotext.LITERAL_POSITIVE: "x"})

        assert setup.literals is True
        assert setup.present is False

    def test_the_keys_are_forwarded_by_send_to_txt2img(self):
        """The buttons forward by exact name, so a key that is not listed
        simply does not arrive and the restore finds half a record."""
        declared = set(mc_infotext.creative_paste_field_names())

        assert mc_infotext.LITERAL_POSITIVE in declared
        assert mc_infotext.LITERAL_NEGATIVE in declared

    def test_restoring_the_setup_puts_the_boxes_back(self, built, store):
        """The explicit action, which is allowed to do what the paste refused
        to: these controls still exist, unlike the Pinned LoRAs field an older
        image records, so they are restored rather than merely shown."""
        import model_chain_krea_creative as creative_script

        mc_creative_krea.pasted.remember(mc_infotext.creative_setup({
            mc_infotext.LITERAL_POSITIVE: "<lora:x:1>",
            mc_infotext.LITERAL_NEGATIVE: "blue hat"}))
        returned = creative_script._restore_setup(False)

        assert returned[-2]["value"] == "<lora:x:1>"
        assert returned[-1]["value"] == "blue hat"

    def test_a_literal_only_restore_changes_nothing_else(self, built, store):
        """No Creative record means no prompt overwrite and no switching
        Creative Mode on -- an image that never used it is not a reason to."""
        import model_chain_krea_creative as creative_script

        mc_creative_krea.pasted.remember(mc_infotext.creative_setup({
            mc_infotext.LITERAL_POSITIVE: "<lora:x:1>"}))
        prompt, enabled, status, _view, positive, _negative = \
            creative_script._restore_setup(False)

        assert positive["value"] == "<lora:x:1>"
        assert prompt == {} or "value" not in prompt
        assert enabled == {} or "value" not in enabled
        assert "no Creative Mode setup" in status

    def test_the_pasted_view_says_what_the_boxes_held(self, built, store):
        import model_chain_krea_creative as creative_script

        mc_creative_krea.pasted.remember(mc_infotext.creative_setup({
            mc_infotext.LITERAL_POSITIVE: "<lora:x:1>"}))
        said = creative_script._pasted_view()

        assert "<lora:x:1>" in said
        assert "reproduces exactly" in said

    def test_a_pasted_image_reproduces_with_the_boxes_emptied(self, script, built,
                                                              store, host):
        """The property all of the above is for, end to end: generate with
        fields, take the recorded prompt, paste it back, and press Generate
        again with the boxes as the paste left them."""
        first = generate(script, "a quiet street", enabled=False,
                         literal_positive="<lora:x:1>", literal_negative="grain")

        again = generate(script, first.prompt, enabled=False,
                         literal_positive="", literal_negative="")

        assert again.prompt == first.prompt


class TestWhenTheRowIsOnScreen:
    """Section 5. Visibility is presentation and never execution.

    The row appears when either prompt-transforming feature is on, because that
    is when protecting text from one is a thing somebody is thinking about. It
    is emphatically not a switch, and the tests that matter are the ones showing
    the values still work while it is hidden.
    """

    def visible(self, creative, spatial) -> bool:
        import model_chain_krea_creative as creative_script

        return creative_script._literal_row(creative, spatial)["visible"]

    def test_creative_alone_shows_it(self):
        assert self.visible(True, False) is True

    def test_spatial_alone_shows_it(self):
        """Either feature, not both. Spatial composes a structured prompt around
        the user's words with no writer involved at all, and a LoRA tag needs
        protecting from the compositor just the same."""
        assert self.visible(False, True) is True

    def test_both_off_hides_it(self):
        assert self.visible(False, False) is False

    def test_both_on_shows_it(self):
        assert self.visible(True, True) is True

    def test_hiding_it_does_not_stop_it_working(self, script, store, host):
        """The whole of section 3.3 in one assertion: the row is hidden exactly
        when both features are off, which is exactly the generation this proves
        still applies the fields."""
        assert self.visible(False, False) is False

        p = generate(script, "a quiet street", enabled=False,
                     literal_positive="<lora:still:1>")

        assert p.prompt == "<lora:still:1> a quiet street"

    def test_the_note_counts_fields_rather_than_words(self):
        """One field is one command however much text is in it, so the number
        somebody reads has to be the number of things that will be inserted."""
        import mc_literal_prompts

        assert mc_literal_prompts.active_note("a, b, c", "") == "1 literal active"
        assert mc_literal_prompts.active_note("a", "b") == "2 literals active"
        assert mc_literal_prompts.active_note("", "") == ""
        assert mc_literal_prompts.active_note("   ", "") == ""
