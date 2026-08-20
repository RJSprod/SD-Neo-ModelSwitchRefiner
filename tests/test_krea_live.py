"""Creativity, and the one rule Krea Live is built on.

Two features, one file, because they are two halves of the same promise. The
slider says a value means one thing everywhere; Live says a prompt is written
once and then reused. Both are claims about *how many times a model is asked*
and about *what it is asked with*, and both are the kind of claim that decays
silently: an extra sampler field creeps into value 1, a cache key grows a
field that has nothing to do with the language model, and nothing visibly
breaks -- prompts just stop being what they were and rerolls start costing eight
seconds each.

So almost everything below counts requests. ``FakeClient.calls`` is the whole
instrument: one entry per completion llama.cpp was asked for, with the sampling
that was asked for in it. A test that asserts a number of calls is asserting
the product.

The compatibility anchor is the other half. Creativity 1 is defined as "exactly
what the Krea writer did before this control existed", which is a promise about
a payload rather than about a prompt, and the way to keep it is to check the
payload: temperature 0.6, top_p 0.9, and no new field smuggled in beside them.
"""

from __future__ import annotations

import pytest

import mc_broker
import mc_infotext
import mc_live_krea
import mc_llm_krea_panel as panel
import mc_llm_paths
import mc_llm_sessions as sessions
from prompt_master.krea import variation


class FakeClient:
    """A llama.cpp client that answers instantly and remembers the asking.

    Deliberately the same shape as the one in ``test_llm_krea``: what a test
    wants to know here is what was in the payload, and the payload is what this
    records.
    """

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls: list[dict] = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.85, top_p=0.95, extra_sampling=None):
        self.calls.append({"messages": messages, "seed": seed, "temperature": temperature,
                           "top_p": top_p, "extra_sampling": dict(extra_sampling or {})})
        answer = self.answers.pop(0) if self.answers else "An expanded Krea prompt."
        on_text(answer)
        return answer


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point every preferences and history file at a throwaway directory."""
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture
def client(monkeypatch, host):
    """A fake writer, a free GPU, and a Live session that starts empty."""
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
    fake = FakeClient()
    monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)
    monkeypatch.setattr(sessions, "_placement_notes", list)
    # The cache key asks which LLM would write, and a test machine has none.
    monkeypatch.setattr(mc_live_krea, "writer_identity", lambda: "test-model.gguf")
    # No checkpoint is loaded under the harness, so the guard would be reading
    # an absent model. Its own behaviour is tested separately.
    monkeypatch.setattr(mc_live_krea, "checkpoint_objection", lambda: "")
    mc_live_krea.live = mc_live_krea.Live()
    yield fake
    mc_live_krea.live = mc_live_krea.Live()
    mc_broker.clear()


def drain(generator) -> list:
    return list(generator)


def expand(source="a lighthouse in a storm", creativity=1, loras="", prompt_seed=7):
    """One trip through the Live gate, as the strip's hidden button makes it."""
    return drain(mc_live_krea.live.prepare(source, creativity, loras, prompt_seed))


# --------------------------------------------------------------------------- #
# The scale
# --------------------------------------------------------------------------- #


class TestTheScaleItself:
    def test_it_runs_from_zero_to_ten_in_whole_steps(self):
        assert (variation.MINIMUM, variation.MAXIMUM) == (0, 10)
        for value in range(0, 11):
            assert variation.creativity_profile(value).creativity == value

    def test_the_default_is_todays_behaviour_and_not_the_middle(self):
        """Installing a feature is not asking for it. Somebody who updates the
        extension and presses Generate must get the prompt they always got."""
        assert variation.DEFAULT == 1
        assert variation.creativity_profile(variation.DEFAULT) == variation.legacy_profile()

    def test_out_of_range_and_nonsense_land_on_the_default_or_the_ends(self):
        assert variation.clamp(-4) == 0
        assert variation.clamp(99) == 10
        assert variation.clamp("banana") == variation.DEFAULT
        assert variation.clamp(None) == variation.DEFAULT
        assert variation.clamp(6.0) == 6

    def test_every_position_says_what_it_is_for(self):
        for value in range(0, 11):
            assert variation.creativity_profile(value).meaning


class TestTheLegacyAnchor:
    """Creativity 1 is a compatibility guarantee, not a tuning opinion."""

    def test_value_one_is_exactly_the_sampling_the_writer_always_used(self):
        from prompt_master.krea import enhancer

        profile = variation.creativity_profile(1)
        assert (profile.temperature, profile.top_p) == (enhancer.TEMPERATURE, enhancer.TOP_P)

    def test_value_one_sends_no_field_that_did_not_exist_before(self):
        """Omitted rather than sent with a guessed neutral value. A neutral
        value somebody invented is still a field llama.cpp reads."""
        assert variation.creativity_profile(1).optional_request_fields == {}

    def test_value_zero_adds_nothing_either(self):
        profile = variation.creativity_profile(0)
        assert (profile.temperature, profile.top_p) == (0.0, 1.0)
        assert profile.optional_request_fields == {}

    def test_a_request_at_one_carries_the_old_payload_and_nothing_else(self, client):
        drain(sessions.krea("a rainy street", [], 7, sessions.Cancellation(), 1))

        assert len(client.calls) == 1
        assert client.calls[0]["temperature"] == 0.6
        assert client.calls[0]["top_p"] == 0.9
        assert client.calls[0]["extra_sampling"] == {}

    def test_a_krea_run_with_no_creativity_given_is_a_run_at_one(self, client):
        """Every caller that has not been told about the slider keeps its
        behaviour, which is what makes this an addition rather than a change."""
        drain(sessions.krea("a rainy street", [], 7, sessions.Cancellation()))

        assert (client.calls[0]["temperature"], client.calls[0]["top_p"]) == (0.6, 0.9)
        assert client.calls[0]["extra_sampling"] == {}


class TestEveryValueAboveOneIsMoreCreative:
    def test_temperature_climbs_and_never_dips(self):
        temperatures = [variation.creativity_profile(v).temperature for v in range(0, 11)]
        assert temperatures == sorted(temperatures)

    def test_top_p_climbs_and_never_dips_above_the_anchor(self):
        tops = [variation.creativity_profile(v).top_p for v in range(1, 11)]
        assert tops == sorted(tops)

    def test_every_value_two_to_ten_is_looser_than_the_anchor(self):
        anchor = variation.legacy_profile()
        for value in range(2, 11):
            profile = variation.creativity_profile(value)
            assert profile.temperature > anchor.temperature
            assert profile.top_p >= anchor.top_p
            assert profile.optional_request_fields

    def test_the_extra_fields_only_ever_widen_the_pool(self):
        """top_k up and min_p down are both "allow more tokens". A row edited
        into a dip would make 6 flatter than 5, which nobody would notice until
        the prompts came back dull."""
        top_ks = [variation.creativity_profile(v).optional_request_fields["top_k"]
                  for v in range(2, 11)]
        min_ps = [variation.creativity_profile(v).optional_request_fields["min_p"]
                  for v in range(2, 11)]
        assert top_ks == sorted(top_ks)
        assert min_ps == sorted(min_ps, reverse=True)

    def test_a_high_value_reaches_the_request(self, client):
        drain(sessions.krea("a rainy street", [], 7, sessions.Cancellation(), 9))

        sent = client.calls[0]
        assert sent["temperature"] == variation.creativity_profile(9).temperature
        assert sent["extra_sampling"] == variation.creativity_profile(9).optional_request_fields


class TestThereIsNoSecondRequest:
    """No candidates, no judge, no hidden follow-up. One state, one completion."""

    def test_the_profile_cannot_ask_for_more_than_one_completion(self):
        fields = set(variation.SamplingProfile.__dataclass_fields__)
        assert not fields & {"candidates", "candidate_count", "judge", "novelty_score", "n"}

    @pytest.mark.parametrize("value", [0, 1, 5, 10])
    def test_one_press_is_one_request_at_every_position(self, client, value):
        drain(sessions.krea("a rainy street", [], 7, sessions.Cancellation(), value))

        assert len(client.calls) == 1


class TestTheInstructionIsNotTouched:
    def test_creativity_changes_the_sampler_and_not_a_word_of_the_prompt(self, client):
        """Krea's expansion instruction is authoritative at every position. A
        creativity control implemented by appending "be adventurous" to it would
        be this repository rewriting upstream's file at run time."""
        drain(sessions.krea("a lighthouse", [], 7, sessions.Cancellation(), 1))
        drain(sessions.krea("a lighthouse", [], 7, sessions.Cancellation(), 10))

        assert client.calls[0]["messages"] == client.calls[1]["messages"]

    def test_the_variation_module_cannot_reach_the_instruction_at_all(self):
        """It has no way to read, write or assemble prompt text: no file access,
        no message building, and no import of the module that owns those. A
        creativity control that could touch the instruction is one edit away
        from being a second prompt writer nobody can see."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path(variation.__file__).read_text(encoding="utf-8"))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert not called & {"open", "read_text", "Path"}

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert not any("krea.enhancer" in name or name.endswith("enhancer")
                       for name in imported)


class TestTheClientWillNotForwardArbitraryFields:
    """The client is imported through a fixture rather than at the top of this
    file: it pulls in httpx, which the extension needs at run time and a bare
    checkout may not have, and one absent HTTP library should skip four tests
    rather than collapse the whole module."""

    @pytest.fixture
    def llama_client(self):
        pytest.importorskip("httpx")
        from prompt_master.inference import llama_client

        return llama_client

    def test_a_field_llama_cpp_has_no_name_for_is_dropped(self, llama_client):
        assert llama_client.sampling_fields({"top_k": 40, "make_it_good": True}) == {"top_k": 40}

    def test_values_are_coerced_to_what_the_server_expects(self, llama_client):
        """A slider hands back a float and a preferences file hands back
        whatever was written into it. Neither may reach llama.cpp as a string
        where a number was meant."""
        assert llama_client.sampling_fields({"top_k": "50", "min_p": "0.05"}) == {
            "top_k": 50, "min_p": 0.05}

    def test_nothing_at_all_is_the_empty_payload(self, llama_client):
        assert llama_client.sampling_fields(None) == {}
        assert llama_client.sampling_fields({}) == {}

    def test_every_field_the_scale_uses_is_one_the_client_accepts(self, llama_client):
        for value in range(0, 11):
            for name in variation.creativity_profile(value).optional_request_fields:
                assert name in llama_client.SAMPLING_FIELDS


# --------------------------------------------------------------------------- #
# Both surfaces, one meaning
# --------------------------------------------------------------------------- #


class TestOneMappingForTwoSurfaces:
    def test_the_manual_panel_offers_the_whole_scale_at_the_default(self, store):
        slider = panel.build()["creativity"]

        assert (slider.minimum, slider.maximum, slider.step) == (0, 10, 1)
        assert slider.value == variation.DEFAULT

    def test_the_live_strip_offers_the_whole_scale_at_the_default(self, store):
        import model_chain_krea_live as live_script

        script = live_script.ScriptKreaLive()
        script.ui(False)
        slider = script.components["creativity"]

        assert (slider.minimum, slider.maximum, slider.step) == (0, 10, 1)
        assert slider.value == variation.DEFAULT

    def test_both_surfaces_resolve_a_value_through_the_same_function(self):
        """Separate saved positions, one shared meaning. If either surface grew
        a table of its own, this is where the two would start to drift."""
        from pathlib import Path

        for module in (panel.__file__,
                       Path(__file__).resolve().parent.parent / "mc_live_krea.py"):
            source = Path(module).read_text(encoding="utf-8")
            assert "krea.variation" in source
            assert "temperature" not in source

    def test_the_two_positions_are_remembered_separately(self, store):
        import mc_llm_state

        mc_llm_state.remember(krea_manual_creativity=2, krea_live_creativity=8)
        assert panel._remembered_creativity() == 2
        assert mc_live_krea.remembered()["creativity"] == 8


# --------------------------------------------------------------------------- #
# One call per new prompt-authoring state
# --------------------------------------------------------------------------- #


class TestTheOneCallRule:
    def test_a_new_source_prompt_costs_exactly_one_request(self, client):
        expand(source="a lighthouse in a storm")

        assert len(client.calls) == 1

    def test_the_same_state_again_costs_nothing(self, client):
        """This is the whole feature. Pressing Generate again with nothing
        changed must not ping the model."""
        expand()
        expand()
        expand()

        assert len(client.calls) == 1

    def test_a_reroll_reuses_the_expansion_and_never_writes_a_new_one(self, client):
        expand()
        for _ in range(10):
            expand()

        assert len(client.calls) == 1

    @pytest.mark.parametrize("changed", [
        {"loras": "<lora:film:0.9>"},
        {"loras": "<lora:film:0.4>"},
    ])
    def test_image_only_settings_never_reach_the_language_model(self, client, changed):
        """Steps, image seed, size, sampler and CFG are not arguments here at
        all -- they cannot reach the key because they are not in the call. The
        pinned LoRAs are, which is why they are the ones worth asserting."""
        expand()
        expand(**changed)

        assert len(client.calls) == 1

    def test_moving_creativity_is_a_new_prompt_authoring_state(self, client):
        expand(creativity=1)
        expand(creativity=6)

        assert len(client.calls) == 2
        assert client.calls[1]["temperature"] == variation.creativity_profile(6).temperature

    def test_changing_the_source_text_is_a_new_state(self, client):
        expand(source="a lighthouse")
        expand(source="a lighthouse at dawn")

        assert len(client.calls) == 2

    def test_changing_the_prompt_seed_is_a_new_state(self, client):
        expand(prompt_seed=7)
        expand(prompt_seed=8)

        assert len(client.calls) == 2

    def test_swapping_the_writing_model_is_a_new_state(self, client, monkeypatch):
        expand()
        monkeypatch.setattr(mc_live_krea, "writer_identity", lambda: "another-model.gguf")
        expand()

        assert len(client.calls) == 2

    def test_the_key_excludes_everything_that_is_not_sent_to_the_writer(self):
        """Stated once, against the function, so the exclusion list cannot be
        eroded one plausible-looking field at a time."""
        first = mc_live_krea.cache_key("a street", 3, 7, identity="m")
        assert first == mc_live_krea.cache_key("a street", 3, 7, identity="m")
        assert first != mc_live_krea.cache_key("a street", 4, 7, identity="m")
        assert first != mc_live_krea.cache_key("a road", 3, 7, identity="m")
        assert first != mc_live_krea.cache_key("a street", 3, 8, identity="m")
        assert first != mc_live_krea.cache_key("a street", 3, 7, identity="n")

    def test_whitespace_around_the_source_is_not_a_new_state(self, client):
        expand(source="a lighthouse")
        expand(source="  a lighthouse  ")

        assert len(client.calls) == 1


class TestARandomPromptSeedIsDrawnOnceAndKept:
    def test_the_drawn_seed_is_stored_with_the_expansion(self, client):
        expand(prompt_seed=-1)

        cached = mc_live_krea.live.cached
        assert cached.prompt_seed != -1
        assert client.calls[0]["seed"] == cached.prompt_seed

    def test_a_second_generation_does_not_redraw_it(self, client):
        """Re-drawing would be a second LLM call for a prompt nobody asked to
        have rewritten -- and would make every reroll cost one."""
        expand(prompt_seed=-1)
        drawn = mc_live_krea.live.cached.prompt_seed
        expand(prompt_seed=-1)

        assert len(client.calls) == 1
        assert mc_live_krea.live.cached.prompt_seed == drawn


# --------------------------------------------------------------------------- #
# The prompt that is generated from
# --------------------------------------------------------------------------- #


class TestTheThreePrompts:
    def test_the_expansion_is_kept_apart_from_the_source(self, client):
        client.answers = ["A tall white lighthouse, storm light, long lens."]
        expand(source="lighthouse in a storm")

        cached = mc_live_krea.live.cached
        assert cached.source == "lighthouse in a storm"
        assert cached.expanded == "A tall white lighthouse, storm light, long lens."

    def test_pinned_loras_are_appended_to_the_generation_prompt_only(self, client):
        client.answers = ["A tall white lighthouse."]
        expand(loras="<lora:film:0.8>")

        armed = mc_live_krea.live.armed
        assert armed.generation == "A tall white lighthouse. <lora:film:0.8>"
        assert armed.expansion.expanded == "A tall white lighthouse."

    def test_the_language_model_is_never_shown_a_lora_tag(self, client):
        expand(loras="<lora:film:0.8>")

        asked = client.calls[0]["messages"][-1]["content"]
        assert "lora" not in asked

    def test_prose_typed_into_the_pinned_field_is_not_smuggled_into_the_prompt(self):
        """The field contributes networks and never prompt text. Anything else
        typed there would be a second, invisible prompt input reaching the image
        model without passing the writer."""
        assert mc_live_krea.lora_suffix(
            "a beautiful sunset <lora:film:0.8> masterpiece") == "<lora:film:0.8>"
        assert mc_live_krea.pinned_tags("nothing but words") == []

    def test_a_changed_lora_weight_changes_the_prompt_without_a_new_expansion(self, client):
        client.answers = ["A tall white lighthouse."]
        expand(loras="<lora:film:0.7>")
        expand(loras="<lora:film:0.9>")

        assert len(client.calls) == 1
        assert mc_live_krea.live.armed.generation.endswith("<lora:film:0.9>")


# --------------------------------------------------------------------------- #
# Arming, and the hook that spends it
# --------------------------------------------------------------------------- #


class TestTheArmingToken:
    def test_an_expansion_arms_exactly_one_generation(self, client):
        expand()

        assert mc_live_krea.live.consume() is not None
        assert mc_live_krea.live.consume() is None

    def test_a_wrong_token_is_refused(self, client):
        expand()

        assert mc_live_krea.live.consume("not-the-token") is None

    def test_nothing_is_armed_before_an_expansion_exists(self):
        assert mc_live_krea.Live().consume() is None

    def test_re_arming_replaces_rather_than_stacks(self, client):
        expand()
        first = mc_live_krea.live.armed.token
        expand()
        second = mc_live_krea.live.armed.token

        assert first != second
        assert mc_live_krea.live.consume().token == second
        assert mc_live_krea.live.consume() is None


class Processing:
    """The half of a StableDiffusionProcessing the Live hook touches."""

    def __init__(self, prompt="a lighthouse"):
        self.prompt = prompt
        self.extra_generation_params = {}


class TestTheProcessingHook:
    @pytest.fixture
    def script(self):
        import model_chain_krea_live as live_script

        return live_script.ScriptKreaLive()

    def test_an_unarmed_generation_is_left_exactly_as_it_was(self, script, client):
        p = Processing("a lighthouse")
        script.before_process(p, True)

        assert p.prompt == "a lighthouse"
        assert p.extra_generation_params == {}

    def test_live_off_leaves_the_prompt_alone_even_if_something_was_armed(self, script, client):
        expand()
        p = Processing("a lighthouse")
        script.before_process(p, False)

        assert p.prompt == "a lighthouse"

    def test_an_armed_generation_is_given_the_expansion(self, script, client):
        client.answers = ["A tall white lighthouse under storm light."]
        expand(source="lighthouse in a storm")
        p = Processing("lighthouse in a storm")
        script.before_process(p, True)

        assert p.prompt == "A tall white lighthouse under storm light."

    def test_the_pinned_loras_arrive_with_it(self, script, client):
        client.answers = ["A tall white lighthouse."]
        expand(loras="<lora:film:0.8>")
        p = Processing()
        script.before_process(p, True)

        assert p.prompt == "A tall white lighthouse. <lora:film:0.8>"

    def test_the_source_and_the_settings_are_recorded(self, script, client):
        expand(source="lighthouse in a storm", creativity=6, prompt_seed=11,
               loras="<lora:film:0.8>")
        p = Processing()
        script.before_process(p, True)

        assert p.extra_generation_params[mc_infotext.LIVE_SOURCE] == "lighthouse in a storm"
        assert p.extra_generation_params[mc_infotext.LIVE_CREATIVITY] == 6
        assert p.extra_generation_params[mc_infotext.LIVE_PROMPT_SEED] == 11
        assert p.extra_generation_params[mc_infotext.LIVE_LORAS] == "<lora:film:0.8>"

    def test_the_expanded_paragraph_is_not_written_into_metadata_twice(self, script, client):
        """It is already the generation's own prompt line. A second copy under a
        key of ours would be a few hundred bytes per PNG repeating what the file
        says, and a copy a later paste could disagree with."""
        client.answers = ["A tall white lighthouse under storm light."]
        expand()
        p = Processing()
        script.before_process(p, True)

        assert "A tall white lighthouse" not in "".join(
            str(value) for value in p.extra_generation_params.values())

    def test_one_expansion_cannot_make_two_images(self, script, client):
        """A nested or queued generation must not inherit the permission. The
        token is consumed on first use and the second call sees nothing."""
        expand()
        first, second = Processing("source"), Processing("source")
        script.before_process(first, True)
        script.before_process(second, True)

        assert first.prompt != "source"
        assert second.prompt == "source"

    def test_the_hook_never_asks_for_an_expansion_of_its_own(self, script, client):
        """It applies; it does not request. An LLM run waits for the host to
        stop generating, so a hook that asked for one would deadlock against the
        job it was running inside."""
        p = Processing("something nothing was armed for")
        script.before_process(p, True)

        assert client.calls == []


# --------------------------------------------------------------------------- #
# Latest input wins
# --------------------------------------------------------------------------- #


class TestStaleText:
    def test_a_revision_drops_the_cached_expansion(self, client):
        expand()
        mc_live_krea.live.revise()

        assert mc_live_krea.live.cached is None

    def test_a_revision_disarms_whatever_was_armed(self, client):
        expand()
        mc_live_krea.live.revise()

        assert mc_live_krea.live.consume() is None

    def test_the_next_generation_after_a_revision_writes_one_new_prompt(self, client):
        expand()
        mc_live_krea.live.revise()
        expand(source="a different lighthouse")

        assert len(client.calls) == 2

    def test_an_answer_that_arrives_after_the_text_moved_on_never_reaches_forge(self, client):
        """The whole point of the revision counter. A late expansion is about
        text nobody is looking at, and drawing an image from it would be worse
        than dropping it."""
        live = mc_live_krea.live

        def answer_late():
            live.revise()
            return "A prompt about the text you deleted."

        client.answers = [answer_late]

        class Answering(FakeClient):
            def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                            temperature=0.85, top_p=0.95, extra_sampling=None):
                self.calls.append({"messages": messages, "seed": seed,
                                   "temperature": temperature, "top_p": top_p,
                                   "extra_sampling": dict(extra_sampling or {})})
                live.revise()
                on_text("A prompt about the text you deleted.")
                return "A prompt about the text you deleted."

        answering = Answering()
        import mc_llm_sessions

        original = mc_llm_sessions._client
        mc_llm_sessions._client = lambda needs_vision=False: answering
        try:
            events = expand()
        finally:
            mc_llm_sessions._client = original

        assert [event.kind for event in events][-1] == sessions.CANCELLED
        assert live.cached is None
        assert live.consume() is None

    def test_stopping_cancels_the_writer_and_disarms(self, client):
        expand()
        mc_live_krea.live.stop()

        assert mc_live_krea.live.consume() is None
        assert mc_live_krea.live.status == mc_live_krea.OFF


class TestTheGpuIsGivenBack:
    def test_an_expansion_leaves_no_workload_holding_the_card(self, client):
        """The Live gate stops early the moment it has the finished prompt, and
        the statement that releases the workload lock is in the run it stopped.
        A lock released whenever the frame is next collected is a lock the image
        generation two milliseconds later waits on for no reason."""
        expand()

        assert mc_broker.active() is None

    def test_a_cache_hit_never_takes_the_card_at_all(self, client):
        expand()
        expand()

        assert mc_broker.active() is None

    def test_a_failed_expansion_gives_it_back_too(self, client, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("llama-server is not running")
            yield  # pragma: no cover - generator shape only

        monkeypatch.setattr(sessions, "krea", explode)
        expand()

        assert mc_broker.active() is None


class TestFailures:
    def test_an_empty_source_is_refused_before_the_model_is_touched(self, client):
        events = expand(source="   ")

        assert events[-1].kind == sessions.FAILED
        assert client.calls == []

    def test_a_failed_expansion_arms_nothing(self, client, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("llama-server is not running")
            yield  # pragma: no cover - generator shape only

        monkeypatch.setattr(sessions, "krea", explode)
        events = expand()

        assert events[-1].kind == sessions.FAILED
        assert mc_live_krea.live.consume() is None

    def test_repeated_failures_stop_automatic_repetition(self, client, monkeypatch):
        """A reroll that fails and rerolls again fails forever, and the first
        anybody knows of it is the log file."""
        def explode(*args, **kwargs):
            raise RuntimeError("nope")
            yield  # pragma: no cover - generator shape only

        monkeypatch.setattr(sessions, "krea", explode)
        for _ in range(mc_live_krea.FAILURE_LIMIT):
            expand()

        assert mc_live_krea.live.exhausted

    def test_one_success_clears_the_count(self, client):
        mc_live_krea.live.note_failure()
        mc_live_krea.live.note_failure()
        expand()

        assert mc_live_krea.live.failures == 0

    def test_a_non_krea_checkpoint_refuses_to_arm(self, client, monkeypatch):
        monkeypatch.setattr(mc_live_krea, "checkpoint_objection",
                            lambda: "Select a Krea 2 checkpoint.")
        events = expand()

        assert events[-1].kind == sessions.FAILED
        assert client.calls == []
        assert mc_live_krea.live.consume() is None


class TestTheCheckpointGuard:
    def test_krea_2_is_allowed(self, monkeypatch, host):
        import mc_arch

        monkeypatch.setattr(mc_arch, "detect_loaded_engine",
                            lambda: mc_arch.by_key("krea2"))
        assert mc_live_krea.checkpoint_objection() == ""

    def test_another_architecture_is_named_in_the_refusal(self, monkeypatch, host):
        import mc_arch

        monkeypatch.setattr(mc_arch, "detect_loaded_engine",
                            lambda: mc_arch.by_key("sdxl"))
        objection = mc_live_krea.checkpoint_objection()
        assert "SDXL" in objection

    def test_a_checkpoint_nobody_can_identify_is_not_refused(self, monkeypatch, host):
        """Detection reads a header and cannot see inside every GGUF or repacked
        build. A guard that blocked real Krea 2 checkpoints would be worse than
        no guard."""
        import mc_arch

        monkeypatch.setattr(mc_arch, "detect_loaded_engine", lambda: mc_arch.UNKNOWN)
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name",
                            lambda name: mc_arch.UNKNOWN)
        assert mc_live_krea.checkpoint_objection() == ""


# --------------------------------------------------------------------------- #
# The strip, and the browser controller
# --------------------------------------------------------------------------- #


class TestTheStrip:
    @pytest.fixture
    def built(self, store):
        import model_chain_krea_live as live_script

        script = live_script.ScriptKreaLive()
        return script, script.ui(False)

    def test_it_is_txt2img_only(self, built):
        import modules.scripts as scripts

        script, _ = built
        assert script.show(False) is scripts.AlwaysVisible
        assert script.show(True) is None

    def test_the_toggle_is_off_until_somebody_turns_it_on(self, built):
        script, _ = built

        assert script.components["enabled"].value is False

    def test_the_strip_is_hidden_until_live_is_on(self, built):
        """Live off is one checkbox and nothing else. The positive prompt must
        not change meaning without something visible saying so, and one checkbox
        is the least that can honestly say it."""
        script, _ = built

        assert script.components["strip"].visible is False
        assert script.components["config"].visible is False
        assert script.components["status"].visible is False

    def test_turning_it_on_reveals_the_strip(self, built, store):
        import model_chain_krea_live as live_script

        strip, config, status = live_script._toggled(True, 4, 5.0, "", -1)
        assert strip["visible"] is True
        assert config["visible"] is True
        assert status["visible"] is True

    def test_turning_it_off_hides_it_again_and_stops_live(self, built, store):
        import model_chain_krea_live as live_script

        strip, _, _ = live_script._toggled(False, 4, 5.0, "", -1)
        assert strip["visible"] is False
        assert mc_live_krea.live.status == mc_live_krea.OFF

    def test_the_gate_never_writes_to_the_positive_prompt(self, built):
        """The visible box keeps the short phrase the user is iterating on. A
        gate that could write to it would be a gate that eats your source text
        five seconds after you stop typing."""
        script, _ = built
        script._prompt_component = _Native("txt2img_prompt")
        script.ui(False)

        for handler in _handlers(script):
            assert script._prompt_component not in list(handler.get("outputs") or [])

    def test_the_gate_reads_the_positive_prompt_as_its_source(self, built):
        script, _ = built
        script._prompt_component = _Native("txt2img_prompt")
        script.ui(False)

        reading = [handler for handler in _handlers(script)
                   if script._prompt_component in list(handler.get("inputs") or [])]
        assert reading

    def test_quick_steps_is_a_handle_on_the_native_control(self, built):
        """Not a second value. A hidden override that made the native slider say
        20 while the generation ran 8 would put a number in the PNG metadata
        that never happened -- so both directions are wired, and both to
        user-input events rather than to change, which would make the two
        controls answer each other forever."""
        script, _ = built
        native = _Native("txt2img_steps", value=20)
        script._steps_component = native
        script.ui(False)

        assert script.components["steps"].value == 20
        outward = [kind for kind, kwargs in script.components["steps"]._callbacks
                   if native in list(kwargs.get("outputs") or [])]
        inward = [kind for kind, kwargs in native._callbacks
                  if script.components["steps"] in list(kwargs.get("outputs") or [])]
        assert outward == ["input"]
        assert inward == ["release"]

    def test_the_positions_are_remembered(self, built, store):
        import model_chain_krea_live as live_script

        live_script._toggled(True, 8, 2.5, "<lora:film:0.8>", 42)
        stored = mc_live_krea.remembered()
        assert stored["creativity"] == 8
        assert stored["delay"] == 2.5
        assert stored["loras"] == "<lora:film:0.8>"
        assert stored["seed"] == 42

    def test_a_remembered_delay_outside_the_range_is_brought_back_into_it(self, store):
        import mc_llm_state

        mc_llm_state.remember(krea_live_delay=900.0)
        assert mc_live_krea.remembered()["delay"] == mc_live_krea.MAX_DELAY


class _Native:
    """A native txt2img control, as ``after_component`` would have handed it over."""

    def __init__(self, elem_id, value=None):
        self.elem_id = elem_id
        self.value = value
        self._callbacks: list = []

    def _record(self, kind, kwargs):
        self._callbacks.append((kind, kwargs))
        return self

    def then(self, **kwargs):
        return self

    def change(self, **kwargs):
        return self._record("change", kwargs)

    def input(self, **kwargs):
        return self._record("input", kwargs)

    def release(self, **kwargs):
        return self._record("release", kwargs)

    def click(self, **kwargs):
        return self._record("click", kwargs)


def _handlers(script) -> list[dict]:
    """Every event handler the Live strip registered on its own components."""
    found: list[dict] = []
    for component in script.components.values():
        for _kind, kwargs in getattr(component, "_callbacks", []) or []:
            found.append(kwargs)
    return found


class TestTheBrowserController:
    """Read rather than executed. What matters here is what the file does and
    does not contain -- these are structural promises about a script whose
    behaviour needs a page to exercise."""

    @pytest.fixture
    def source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / "javascript"
                / "model_chain_live_krea.js").read_text(encoding="utf-8")

    def test_it_adds_to_forges_generate_menu_rather_than_replacing_it(self, source):
        assert "appendContextMenuOption" in source
        assert "Generate with Krea Live once" in source
        assert "Start Krea Live" in source
        assert "Stop Krea Live" in source

    def test_it_never_removes_an_entry_from_that_menu(self, source):
        """Forge's own "Generate forever" and "Cancel generate forever" have to
        still be there afterwards."""
        assert "removeContextMenuOption" not in source

    def test_it_intercepts_in_the_capture_phase(self, source):
        """Before any handler the host or another extension attached to the
        button itself, which is the only place the gate can run and still be
        ahead of the native submission."""
        assert 'addEventListener("click", onGenerateClick, true)' in source

    def test_the_bypass_is_one_shot(self, source):
        """Cleared before the click proceeds, so a second programmatic click
        cannot ride through on the same permission."""
        assert "state.bypass = false;" in source

    def test_it_reads_the_delay_from_the_strip_rather_than_hard_coding_it(self, source):
        assert "function delayMs()" in source

    def test_it_refuses_the_native_repeat_loop(self, source):
        assert "generateOnRepeatInterval" in source

    def test_every_element_is_found_by_an_extension_owned_or_documented_id(self, source):
        for identifier in ("mc-krea-live-toggle", "mc-krea-live-run", "mc-krea-live-token",
                           "txt2img_prompt", "txt2img_generate"):
            assert identifier in source
