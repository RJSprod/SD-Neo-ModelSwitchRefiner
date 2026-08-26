"""Creative Mode: the library, the Director, the scale and the one-call rule.

The feature's whole claim is that art direction can be chosen well *without*
asking a model, and everything below is one of the two ways that claim can be
false.

The first is that the Director does not really choose. A single-word prompt at
Creativity 10 has to come back materially different on successive seeds, a lock
in the source text has to survive every roll, and a Natural axis has to be
genuinely absent rather than mentioned-and-hedged. None of those are visible
from reading the code, because they are properties of a distribution and of a
vocabulary rather than of a function; they are measured here over hundreds of
rolls.

The second is that "one model call" quietly stops being true. So the writer is a
counter: one entry per completion llama.cpp was asked for. A test that asserts a
number of calls is asserting the product, and the number is always one.

The package's own ``tests/acceptance_cases.json`` is read and checked off rather
than paraphrased -- :class:`TestTheAcceptanceCases` fails if the package grows a
case nothing here covers, which is what stops the data and the tests drifting
apart when the library is next updated.
"""

from __future__ import annotations

import collections
import json
import types
from pathlib import Path

import pytest

import mc_broker
import mc_creative_krea
import mc_creative_panel
import mc_creative_profiles as profiles
import mc_infotext
import mc_llm_krea_panel as panel
import mc_llm_paths
import mc_llm_sessions as sessions
from prompt_master.krea import director, variation
from prompt_master.krea import library as library_module

LIBRARY = Path(__file__).resolve().parent.parent / "prompt_master" / "krea" / "creativity"


class FakeClient:
    """A llama.cpp client that answers instantly and remembers the asking."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls: list[dict] = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.85, top_p=0.95):
        self.calls.append({"messages": messages, "seed": seed, "temperature": temperature,
                           "top_p": top_p})
        answer = self.answers.pop(0) if self.answers else "An expanded Krea prompt."
        on_text(answer)
        return answer

    @property
    def turn(self) -> str:
        """The user half of the last request: source, direction and references."""
        return self.calls[-1]["messages"][-1]["content"]


@pytest.fixture
def lib():
    return library_module.library()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point every preferences and history file at a throwaway directory."""
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture
def client(monkeypatch, host, store):
    """A fake writer, a free GPU, and a Creative session that starts empty."""
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
    fake = FakeClient()
    monkeypatch.setattr(sessions, "_client", lambda needs_vision=False, reserve=0, role='': fake)
    monkeypatch.setattr(sessions, "_placement_notes", lambda role="": [])
    # No checkpoint is loaded under the harness; the guard has its own tests.
    monkeypatch.setattr(mc_creative_krea, "checkpoint_objection", lambda: "")
    mc_creative_krea.creative = mc_creative_krea.Creative()
    yield fake
    mc_creative_krea.creative = mc_creative_krea.Creative()
    mc_broker.clear()


def axes(**overrides) -> dict:
    """Axis settings with every axis on Vary, minus whatever is overridden."""
    settings = {key: director.AxisSetting(mode=director.VARY)
                for key in library_module.library().axis_keys}
    settings.update(overrides)
    return settings


def roll(source="car", creativity=10, seed=director.RANDOM_SEED, settings=None, history=()):
    return director.roll(source, creativity, seed, settings or axes(), history)


def varying(creativity=10, **overrides) -> dict:
    """The stored settings, with every axis varying.

    The factory configuration is now every axis Natural -- a fresh install
    directs nothing until somebody asks it to -- so a test about *how a directed
    roll is made* has to say that it wants direction. Saying it here once is what
    keeps that from being restated in every such test.
    """
    stored = mc_creative_krea.settings()
    stored["creativity"] = creativity
    stored["axis_modes"] = {key: director.VARY
                            for key in library_module.library().axis_keys}
    stored.update(overrides)
    return stored


# --------------------------------------------------------------------------- #
# The data package
# --------------------------------------------------------------------------- #


class TestTheLibrary:
    def test_it_loads_and_says_what_it_is(self, lib):
        assert lib.schema_version == 1
        assert lib.version
        assert lib.variant_count == 164

    def test_every_axis_the_manifest_names_has_a_file(self, lib):
        manifest = json.loads((LIBRARY / "library_manifest.json").read_text(encoding="utf-8"))
        assert list(lib.axis_keys) == list(manifest["axes"])
        for key in lib.axis_keys:
            assert lib.axis(key).variants

    def test_every_variant_can_say_itself_at_every_strength(self, lib):
        """A missing tier is a Creativity slider that stops scaling on one axis
        and says nothing about it, so the loader refuses one rather than falling
        back to a neighbour."""
        for key in lib.axis_keys:
            for variant in lib.axis(key).variants:
                for tier in library_module.TIERS:
                    assert variant.expression(tier).strip()

    def test_the_four_tiers_are_in_ascending_order(self):
        assert library_module.TIERS == ("light", "moderate", "strong", "extreme")

    def test_ids_are_unique_within_an_axis(self, lib):
        """Saved Fixed selections are stored as ids. Two entries sharing one id
        means a saved configuration whose meaning depends on file order."""
        for key in lib.axis_keys:
            identifiers = [v.identifier for v in lib.axis(key).variants]
            assert len(identifiers) == len(set(identifiers))

    def test_a_package_with_a_missing_tier_is_refused(self, tmp_path):
        _copy_library(tmp_path)
        path = tmp_path / "axes" / "texture.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        del document["entries"][0]["expressions"]["extreme"]
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(library_module.LibraryError) as raised:
            library_module.load(tmp_path)
        assert "extreme" in str(raised.value)

    def test_a_package_with_a_duplicate_id_is_refused(self, tmp_path):
        _copy_library(tmp_path)
        path = tmp_path / "axes" / "mood.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["entries"][1]["id"] = document["entries"][0]["id"]
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(library_module.LibraryError):
            library_module.load(tmp_path)

    def test_a_missing_package_is_a_library_error_and_not_a_crash(self, tmp_path):
        with pytest.raises(library_module.LibraryError):
            library_module.load(tmp_path / "nowhere")

    def test_the_sampling_table_matches_the_package(self, lib):
        """Duplicated on purpose -- Creativity 0 and 1 are compatibility
        guarantees and a data package must not be able to move them -- so the
        duplication is held to agreeing."""
        for position in range(0, 11):
            row = lib.policy.sampling[position]
            profile = variation.creativity_profile(position)
            assert profile.temperature == pytest.approx(row["temperature"])
            assert profile.top_p == pytest.approx(row["top_p"])

    def test_the_package_agrees_that_one_is_the_legacy_row(self, lib):
        assert lib.policy.sampling[1].get("legacy_exact") is True
        assert variation.legacy_profile().temperature == 0.6
        assert variation.legacy_profile().top_p == 0.9


def _copy_library(destination: Path) -> None:
    import shutil

    shutil.copytree(LIBRARY, destination, dirs_exist_ok=True)


# --------------------------------------------------------------------------- #
# The Director makes no calls
# --------------------------------------------------------------------------- #


class TestTheDirectorNeverAsksAModel:
    def test_a_roll_touches_no_client(self, client):
        for creativity in range(0, 11):
            roll(creativity=creativity)

        assert client.calls == []

    def test_the_module_imports_nothing_that_could_reach_a_model(self):
        """Stated against the import graph rather than trusted. A Director that
        could reach the runtime is one refactor away from a planner call."""
        import ast

        tree = ast.parse(Path(director.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert not any(name.startswith("mc_") for name in imported)
        assert not any("inference" in name or "client" in name for name in imported)
        assert not any(name in {"httpx", "requests", "socket", "urllib"} for name in imported)

    def test_no_recipe_type_can_ask_for_a_second_completion(self):
        fields = set(director.CreativeRecipe.__dataclass_fields__)
        assert not fields & {"candidates", "candidate_count", "judge", "novelty_score"}
        assert not set(variation.SamplingProfile.__dataclass_fields__) & {
            "candidates", "candidate_count", "judge", "novelty_score", "n"}


# --------------------------------------------------------------------------- #
# Natural / Vary / Fixed
# --------------------------------------------------------------------------- #


class TestNatural:
    def test_a_natural_axis_produces_no_line_at_any_creativity(self):
        """Absent, not hedged. A brief saying "Texture: your choice" has put
        texture in the model's foreground, which is the opposite of leaving it
        alone."""
        settings = axes(texture=director.AxisSetting(mode=director.NATURAL))
        for creativity in range(0, 11):
            for _ in range(20):
                recipe = roll(creativity=creativity, settings=settings)
                assert all(item.axis != "texture" for item in recipe.items)
                assert "Texture:" not in recipe.brief

    def test_every_axis_natural_means_no_brief_at_all(self):
        settings = {key: director.AxisSetting(mode=director.NATURAL)
                    for key in library_module.library().axis_keys}
        recipe = roll(creativity=10, settings=settings)

        assert recipe.items == ()
        assert recipe.brief == ""
        assert not recipe


class TestVary:
    def test_a_varied_axis_is_chosen_locally_and_differs_across_seeds(self):
        seen = {roll(creativity=10).compact for _ in range(50)}

        assert len(seen) > 40, "Creativity 10 is producing near-identical recipes"

    def test_one_vary_axis_still_scales_from_light_to_extreme(self):
        """The case the design calls out: a user with only Medium on Vary must
        still see the difference between 2 and 10."""
        settings = {key: director.AxisSetting(mode=director.NATURAL)
                    for key in library_module.library().axis_keys}
        settings["medium"] = director.AxisSetting(mode=director.VARY)

        strengths = {}
        for creativity in (2, 5, 7, 10):
            recipe = roll(creativity=creativity, settings=settings)
            assert len(recipe.items) == 1
            strengths[creativity] = recipe.items[0]

        assert strengths[2].strength == "light"
        assert strengths[5].strength == "moderate"
        assert strengths[7].strength == "strong"
        assert strengths[10].strength == "extreme"
        assert len(strengths[10].expression) > len(strengths[2].expression) * 3


class TestFixed:
    def test_a_fixed_axis_repeats_across_random_seeds(self):
        settings = axes(texture=director.AxisSetting(mode=director.FIXED,
                                                     fixed_id="paper_grain"))
        for _ in range(25):
            recipe = roll(creativity=10, settings=settings)
            texture = [item for item in recipe.items if item.axis == "texture"]
            assert [item.variant_id for item in texture] == ["paper_grain"]
            assert texture[0].source == director.FIXED

    def test_fixed_still_applies_at_creativity_zero_and_one(self):
        """Explicit user configuration, not variation -- and the scale governs
        variation. A pin that vanished at 1 would be a pin that silently stopped
        meaning anything."""
        settings = axes(texture=director.AxisSetting(mode=director.FIXED,
                                                     fixed_id="paper_grain"))
        for creativity in (0, 1):
            recipe = roll(creativity=creativity, settings=settings)
            assert recipe.compact == "texture=paper_grain"
            assert recipe.items[0].strength == "light"

    def test_a_pinned_id_the_package_no_longer_has_falls_silent(self):
        """A library update that renamed an id must not stop the panel building
        or invent a substitute the user never chose."""
        settings = axes(texture=director.AxisSetting(mode=director.FIXED,
                                                     fixed_id="no_such_variant"))
        recipe = roll(creativity=10, settings=settings)

        assert all(item.axis != "texture" for item in recipe.items)

    def test_a_fixed_axis_is_never_also_varied(self):
        settings = axes(medium=director.AxisSetting(mode=director.FIXED,
                                                    fixed_id="watercolor"))
        for _ in range(20):
            recipe = roll(creativity=10, settings=settings)
            mediums = [item for item in recipe.items if item.axis == "medium"]
            assert len(mediums) == 1
            assert mediums[0].variant_id == "watercolor"


# --------------------------------------------------------------------------- #
# The source prompt wins
# --------------------------------------------------------------------------- #


class TestTheSourcePromptWins:
    def test_a_stated_medium_is_never_replaced(self):
        """The lock test. "oil painting of a car" stays an oil painting at
        Creativity 10 however Medium is configured."""
        for _ in range(40):
            recipe = roll("oil painting of a car", creativity=10)
            assert "medium" in recipe.locked
            assert all(item.axis != "medium" for item in recipe.items)

    def test_a_stated_viewpoint_locks_the_viewpoint(self):
        recipe = roll("top-down photo of a car", creativity=10)

        assert {"viewpoint", "medium"} <= set(recipe.locked)

    def test_an_alias_that_names_two_axes_locks_both(self):
        """"direct flash photo of a car" has settled the medium *and* the light,
        and directing golden hour over it would be the Director overruling the
        person it works for."""
        recipe = roll("direct flash photo of a car", creativity=10)

        assert {"medium", "lighting"} <= set(recipe.locked)

    def test_a_bare_noun_locks_nothing(self):
        recipe = roll("car", creativity=10)

        assert recipe.locked == ()

    def test_a_lock_beats_a_fixed_pin(self):
        """Precedence: explicit source, then Fixed, then Vary. A pin the user's
        own words contradict loses to the words."""
        settings = axes(medium=director.AxisSetting(mode=director.FIXED,
                                                    fixed_id="anime_keyframe"))
        recipe = roll("oil painting of a car", creativity=10, settings=settings)

        assert all(item.axis != "medium" for item in recipe.items)

    def test_the_rule_is_in_every_brief_that_exists(self):
        """Alias matching will never understand every phrase, so the brief does
        not rely on having found everything -- it tells the model which of the
        two to drop when they disagree."""
        for creativity in range(2, 11):
            recipe = roll(creativity=creativity)
            assert director.SOURCE_PRIORITY_RULE in recipe.brief

    def test_a_short_alias_does_not_match_inside_a_longer_word(self):
        assert "medium" not in roll("a bowl of risotto", creativity=10).locked


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #


class TestScaling:
    def test_nothing_is_directed_at_zero_or_one(self):
        for creativity in (0, 1):
            for _ in range(20):
                recipe = roll(creativity=creativity)
                assert recipe.items == ()
                assert recipe.brief == ""

    @pytest.mark.parametrize("creativity,tier", [
        (2, "light"), (3, "light"), (4, "moderate"), (5, "moderate"), (6, "moderate"),
        (7, "strong"), (8, "strong"), (9, "extreme"), (10, "extreme")])
    def test_each_position_uses_its_own_expression_tier(self, creativity, tier):
        for _ in range(10):
            recipe = roll(creativity=creativity)
            assert {item.strength for item in recipe.items} == {tier}

    def test_the_active_axis_count_follows_the_packages_policy(self, lib):
        for creativity in range(2, 11):
            low, high = lib.policy.activation_range(creativity, len(lib.axis_keys))
            for _ in range(30):
                assert low <= len(roll(creativity=creativity).items) <= high

    def test_breadth_climbs_with_the_slider(self, lib):
        """Two claims, because the policy and the draw can fail separately.

        The policy's own ranges must never dip -- that is deterministic and is
        checked exactly. The measured averages are then checked at four spaced
        positions rather than at all nine, because neighbouring rows share a
        range by design (2 and 3 are both "1 to 2") and demanding a strict climb
        between two identical rows is demanding that a coin land the same way
        twice."""
        widest = len(lib.axis_keys)
        ranges = [lib.policy.activation_range(creativity, widest)
                  for creativity in range(2, 11)]
        assert [low for low, _ in ranges] == sorted(low for low, _ in ranges)
        assert [high for _, high in ranges] == sorted(high for _, high in ranges)

        average = {creativity: sum(len(roll(creativity=creativity).items)
                                   for _ in range(40)) / 40
                   for creativity in (2, 5, 8, 10)}
        counts = [average[creativity] for creativity in (2, 5, 8, 10)]
        assert counts == sorted(counts)

    def test_ten_activates_every_eligible_axis(self, lib):
        recipe = roll(creativity=10)

        assert len(recipe.items) == len(lib.axis_keys)

    def test_ten_activates_every_eligible_axis_when_some_are_natural(self):
        settings = axes(texture=director.AxisSetting(mode=director.NATURAL),
                        mood=director.AxisSetting(mode=director.NATURAL))
        recipe = roll(creativity=10, settings=settings)

        assert len(recipe.items) == 8

    def test_a_variant_above_the_position_is_never_drawn(self, lib):
        for creativity in range(2, 11):
            for _ in range(20):
                for item in roll(creativity=creativity).items:
                    variant = lib.axis(item.axis).variant(item.variant_id)
                    assert variant.min_creativity <= creativity


class TestTheSingleWordTest:
    """The design's own headline case: "car" at Creativity 10."""

    def test_successive_rolls_produce_materially_different_art_direction(self):
        mediums = collections.Counter()
        for _ in range(200):
            recipe = roll("car", creativity=10)
            mediums[dict((item.axis, item.variant_id) for item in recipe.items)["medium"]] += 1

        assert len(mediums) >= 12, f"only {len(mediums)} mediums reachable: {mediums}"

    def test_the_reachable_treatments_span_the_families_the_design_names(self, lib):
        seen = set()
        for _ in range(300):
            for item in roll("car", creativity=10).items:
                if item.axis == "medium":
                    seen.add(lib.axis("medium").variant(item.variant_id).family)

        assert {"painting", "illustration", "photography", "print"} <= seen


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


class TestSeeds:
    def test_random_is_the_default(self, lib):
        assert lib.defaults["creative_seed"] == director.RANDOM_SEED
        assert mc_creative_krea.settings()["seed"] == director.RANDOM_SEED

    def test_each_roll_resolves_a_concrete_seed(self):
        resolved = {roll(creativity=10).creative_seed for _ in range(20)}

        assert director.RANDOM_SEED not in resolved
        assert len(resolved) > 15

    def test_a_fixed_seed_reproduces_the_recipe_and_the_writer_seed(self):
        first = roll(creativity=8, seed=20260820)
        second = roll(creativity=8, seed=20260820)

        assert first.items == second.items
        assert first.llm_seed == second.llm_seed

    def test_the_derivation_does_not_move_between_processes(self):
        """SHA-256 and not hash(), which Python salts per process -- otherwise
        "a fixed seed reproduces" would be true only within a session."""
        assert director.stable_hash(20260820, "director") == 2250260909
        assert director.stable_hash(20260820, "llm") == 2152571984

    def test_the_two_sub_seeds_are_not_the_same_number(self):
        for seed in (0, 1, 42, 20260820):
            head, tail = director.derive(seed)
            assert head != tail

    def test_a_different_seed_gives_different_direction(self):
        assert roll(creativity=8, seed=1).items != roll(creativity=8, seed=2).items

    def test_the_seed_is_carried_on_the_recipe_for_the_metadata_to_record(self):
        recipe = roll(creativity=6, seed=777)

        assert recipe.creative_seed == 777
        assert recipe.library_version


# --------------------------------------------------------------------------- #
# Compatibility and anti-repetition
# --------------------------------------------------------------------------- #


class TestCompatibility:
    def test_impasto_texture_never_lands_on_a_photographic_medium(self, lib):
        settings = axes(texture=director.AxisSetting(mode=director.FIXED,
                                                     fixed_id="impasto"))
        for _ in range(120):
            recipe = roll(creativity=10, settings=settings)
            chosen = {item.axis: item.variant_id for item in recipe.items}
            family = lib.axis("medium").variant(chosen["medium"]).family
            assert family != "photography"

    def test_a_fisheye_lens_is_never_paired_with_a_telephoto_partner(self, lib):
        settings = axes(lens_zoom=director.AxisSetting(mode=director.FIXED,
                                                       fixed_id="fisheye"))
        for _ in range(60):
            recipe = roll(creativity=10, settings=settings)
            for item in recipe.items:
                variant = lib.axis(item.axis).variant(item.variant_id)
                if item.axis != "lens_zoom":
                    assert "telephoto" not in variant.tags

    def test_a_preference_is_a_boost_and_never_a_deadlock(self):
        """A hard preference would empty the pool whenever the preferred partner
        had not been drawn yet, and the axis would silently fall out."""
        settings = axes(texture=director.AxisSetting(mode=director.FIXED,
                                                     fixed_id="impasto"))
        for _ in range(40):
            assert len(roll(creativity=10, settings=settings).items) == 10


class TestAntiRepetition:
    def test_recent_variants_are_avoided_at_ten_when_alternatives_exist(self, lib):
        recent = [v.identifier for v in lib.axis("medium").variants[:8]]
        reused = 0
        for _ in range(200):
            recipe = roll(creativity=10, history=recent)
            chosen = {item.axis: item.variant_id for item in recipe.items}
            if chosen["medium"] in recent:
                reused += 1

        assert reused == 0

    def test_it_does_not_refuse_to_direct_when_everything_is_recent(self, lib):
        """"Avoid what you used last time" must never become "produce nothing"."""
        every = [v.identifier for v in lib.axis("medium").variants]
        for _ in range(30):
            recipe = roll(creativity=10, history=every)
            assert any(item.axis == "medium" for item in recipe.items)

    def test_low_creativity_does_not_penalise_at_all(self, lib):
        assert lib.anti_repetition.strength_at(2) == 0
        assert lib.anti_repetition.strength_at(10) == 1.0

    def test_generic_cliches_are_discouraged_only_high_up(self):
        assert roll(creativity=4).avoid == ()
        assert roll(creativity=10).avoid

    def test_a_cliche_the_user_asked_for_is_never_discouraged(self):
        """"ultra detailed" is a cliché until somebody types it, at which point
        it is a request."""
        recipe = roll("an ultra detailed car", creativity=10)

        assert not any("ultra detailed" in phrase for phrase in recipe.avoid)
        assert "ultra detailed" not in recipe.brief

    def test_the_history_the_controller_keeps_is_ids_and_not_prompts(self, client):
        recipe = roll(creativity=10)
        mc_creative_krea.note_roll(recipe)

        assert list(mc_creative_krea.recent_ids()) == list(recipe.variant_ids)
        assert all(" " not in identifier for identifier in mc_creative_krea.recent_ids())

    def test_the_history_is_trimmed_to_the_packages_length(self, client, lib):
        for _ in range(20):
            mc_creative_krea.note_roll(roll(creativity=10))

        assert len(mc_creative_krea.history()) == lib.anti_repetition.history_length


# --------------------------------------------------------------------------- #
# One model call
# --------------------------------------------------------------------------- #


class TestTheOneCallRule:
    def test_one_roll_is_one_writer_request(self, client):
        list(mc_creative_krea.creative.roll("car"))

        assert len(client.calls) == 1

    @pytest.mark.parametrize("creativity", [0, 1, 2, 5, 10])
    def test_that_holds_at_every_position(self, client, creativity):
        stored = mc_creative_krea.settings()
        stored["creativity"] = creativity
        list(mc_creative_krea.creative.roll("car", stored))

        assert len(client.calls) == 1

    def test_ten_presses_are_ten_requests_and_never_more(self, client):
        for _ in range(10):
            list(mc_creative_krea.creative.roll("car"))

        assert len(client.calls) == 10

    def test_the_brief_travels_in_the_user_turn_and_not_the_instruction(self, client):
        list(mc_creative_krea.creative.roll("car", varying()))

        system, user = client.calls[0]["messages"]
        assert director.BRIEF_HEADING not in system["content"]
        assert director.BRIEF_HEADING in user["content"]

    def test_krea_s_own_instruction_is_the_same_at_every_position(self, client):
        for creativity in (1, 5, 10):
            stored = mc_creative_krea.settings()
            stored["creativity"] = creativity
            list(mc_creative_krea.creative.roll("car", stored))

        systems = {call["messages"][0]["content"] for call in client.calls}
        assert len(systems) == 1

    def test_the_writer_runs_at_the_seed_the_director_derived(self, client):
        stored = mc_creative_krea.settings()
        stored["creativity"] = 7
        stored["seed"] = 4242
        list(mc_creative_krea.creative.roll("car", stored))

        assert client.calls[0]["seed"] == director.derive(4242)[1]

    def test_creativity_one_asks_exactly_what_it_always_did(self, client):
        """The compatibility guarantee, checked as a payload *and* as a message:
        legacy sampling, and a user turn with nothing added to it."""
        stored = mc_creative_krea.settings()
        stored["creativity"] = 1
        list(mc_creative_krea.creative.roll("car", stored))

        assert client.calls[0]["temperature"] == 0.6
        assert client.calls[0]["top_p"] == 0.9
        assert client.turn == "user_prompt:\ncar"

    def test_a_library_failure_refuses_the_roll_rather_than_writing_undirected(
            self, client, monkeypatch):
        """Somebody who turned Creative Mode on asked for art direction. A plain
        expansion pretending to be one answers a question they did not ask."""
        def explode(*args, **kwargs):
            raise library_module.LibraryError("no package")

        monkeypatch.setattr(director, "roll", explode)
        events = list(mc_creative_krea.creative.roll("car"))

        assert events[-1].kind == sessions.FAILED
        assert client.calls == []


# --------------------------------------------------------------------------- #
# Room for the picture that follows
# --------------------------------------------------------------------------- #


class TestTheImageModelKeepsItsRoom:
    """Creative Mode inverted an order that used to be safe.

    Before it, the image checkpoint was loaded first and the language model
    negotiated its placement against whatever was left. A Creative roll loads
    the language model *first*, onto a card with nothing on it -- so llama.cpp
    sizes itself to the whole card and the checkpoint that has to run three
    hundred milliseconds later gets the remainder.

    The fix is to say up front how much to leave. These check that the number
    is right, that it reaches the placement, and that it is asked for only on
    the path where a picture actually follows.
    """

    @pytest.fixture
    def card(self, monkeypatch, host):
        """A 24 GB card, an 8 GB checkpoint, no margin, and nothing resident."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "vram_required_bytes",
                            lambda name, *a, **k: 8 * 1024 ** 3)
        monkeypatch.setattr(mc_broker, "resident_bytes", lambda family=None: 0)
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        from modules import shared

        shared.opts.sd_model_checkpoint = "krea2.safetensors"
        yield

    def test_it_reserves_what_the_pass_needs(self, card):
        assert mc_creative_krea.image_reserve_bytes() == 8 * 1024 ** 3

    def test_what_the_image_side_already_holds_is_not_reserved_twice(
            self, card, monkeypatch):
        """Those bytes are the loaded checkpoint. Reserving them again would
        shrink the language model to make room for a model already there."""
        monkeypatch.setattr(mc_broker, "resident_bytes", lambda family=None: 6 * 1024 ** 3)

        assert mc_creative_krea.image_reserve_bytes() == 2 * 1024 ** 3

    def test_the_safety_margin_is_not_reserved_twice_either(self, card, monkeypatch):
        """``negotiate`` adds the global margin on top of whatever this returns,
        and it is the same number -- both are ``vram_headroom_bytes()``, which
        ``vram_required_bytes`` already includes. Counted twice it is one
        activation peak's worth of card held back for nothing, which on a 24 GB
        machine is several blocks of the language model."""
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 1024 ** 3)

        assert mc_creative_krea.image_reserve_bytes() == 7 * 1024 ** 3

    def test_a_fully_resident_checkpoint_needs_nothing_reserved(self, card, monkeypatch):
        monkeypatch.setattr(mc_broker, "resident_bytes", lambda family=None: 20 * 1024 ** 3)

        assert mc_creative_krea.image_reserve_bytes() == 0

    def test_a_checkpoint_nobody_declared_still_counts(self, card, monkeypatch):
        """The register only holds what was declared to it, and image
        checkpoints never are -- Forge loads and moves them. Sizing the reserve
        from the register therefore answered 0 for a checkpoint sitting in
        fourteen gigabytes of VRAM, and the roll reserved room for it all over
        again: on a 24 GB card that put the language model in 8.7 GB *minus*
        another 13.9 GB, which is sixteen of forty-eight blocks on the card and
        the rest crawling from system RAM.
        """
        monkeypatch.setattr(mc_broker, "resident_bytes", lambda family=None: 0)
        monkeypatch.setattr(mc_broker, "reported_bytes",
                            lambda family: 8 * 1024 ** 3 if family == mc_broker.FAMILY_IMAGE
                            else 0)

        assert mc_creative_krea.image_reserve_bytes() == 0

    def test_an_unknown_checkpoint_reserves_nothing_rather_than_refusing(self, card):
        from modules import shared

        shared.opts.sd_model_checkpoint = ""

        assert mc_creative_krea.image_reserve_bytes() == 0

    def test_it_never_raises_however_the_host_answers(self, card, monkeypatch):
        import mc_memory

        def explode(*args, **kwargs):
            raise RuntimeError("no such checkpoint")

        monkeypatch.setattr(mc_memory, "vram_required_bytes", explode)

        assert mc_creative_krea.image_reserve_bytes() == 0

    def test_the_reserve_reaches_the_placement(self, client, card, monkeypatch):
        """All the way to negotiate's extra_reserve, which is the parameter that
        was there for this and had never been used for it."""
        seen = []
        original = sessions._client
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False, reserve=0, role='': (
                                seen.append(reserve), original(needs_vision))[1])

        stored = mc_creative_krea.settings()
        list(mc_creative_krea.creative.roll("car", stored, guard_checkpoint=True))

        assert seen == [8 * 1024 ** 3]

    def test_llm_studio_reserves_nothing_because_no_picture_follows(
            self, client, card, monkeypatch):
        """Reserving image VRAM there would shrink the writer for a picture
        nobody asked for."""
        seen = []
        original = sessions._client
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False, reserve=0, role='': (
                                seen.append(reserve), original(needs_vision))[1])

        stored = mc_creative_krea.settings()
        list(mc_creative_krea.creative.roll("car", stored))

        assert seen == [0]

    def test_the_room_is_asked_for_before_the_prompt_is_handed_over(
            self, client, card, monkeypatch):
        """The reserve prevents the problem; this recovers a card that was
        already full when the roll started."""
        asked = []
        monkeypatch.setattr(mc_creative_krea, "hand_back_vram",
                            lambda *a, **k: asked.append(True) or 0)

        stored = mc_creative_krea.settings()
        events = list(mc_creative_krea.creative.roll("car", stored, guard_checkpoint=True))

        assert asked == [True]
        assert events[-1].kind == sessions.DONE

    def test_it_is_not_asked_for_on_the_studio_path(self, client, card, monkeypatch):
        asked = []
        monkeypatch.setattr(mc_creative_krea, "hand_back_vram",
                            lambda *a, **k: asked.append(True) or 0)

        list(mc_creative_krea.creative.roll("car", mc_creative_krea.settings()))

        assert asked == []

    def test_asking_costs_nothing_when_the_card_already_has_room(self, card, monkeypatch):
        """request_vram returns immediately when what is free covers the
        requirement, so a correctly sized card never pays for the call."""
        monkeypatch.setattr(mc_broker, "resident_bytes", lambda family=None: 20 * 1024 ** 3)
        called = []
        monkeypatch.setattr(mc_broker, "request_vram",
                            lambda *a, **k: called.append(a) or None)

        assert mc_creative_krea.hand_back_vram() == 0
        assert called == []


# --------------------------------------------------------------------------- #
# There is no Live mode
# --------------------------------------------------------------------------- #


class TestThereIsNoLiveMode:
    def test_no_module_schedules_anything(self):
        """Retired in full: no timer, no typing watcher, no repeat loop.

        Asserted against the *code* rather than the file, because both modules
        discuss at length what was removed and a plain text search would trip
        over their own explanations. What is checked is that nothing calls a
        scheduler and nothing imports one -- the failure mode being a leftover
        timer nobody notices until it generates on its own.
        """
        import ast

        root = Path(__file__).resolve().parent.parent
        for name in ("mc_creative_krea.py", "scripts/model_chain_krea_creative.py"):
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            called = set()
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    called.add(node.attr)
                elif isinstance(node, ast.Name):
                    called.add(node.id)
                elif isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.add(node.module or "")
            assert not called & {"Timer", "sleep", "monotonic", "perf_counter"}
            assert "sched" not in imported
            assert "asyncio" not in imported

    def test_the_browser_file_watches_nothing(self):
        """The scheduler assertions live in ``test_krea_creative_js.py``, which
        runs the file; this is the one that belongs with the Python, because it
        is about the *input* side. Nothing in the page observes the prompt box,
        so there is no path from "text changed" to anything at all."""
        source = (Path(__file__).resolve().parent.parent / "javascript"
                  / "model_chain_creative_krea.js").read_text(encoding="utf-8")

        assert '"input"' not in source
        assert "reroll" not in source.casefold()
        assert "MutationObserver" not in source

    def test_the_live_modules_are_gone(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("mc_live_krea.py", "scripts/model_chain_krea_live.py",
                     "javascript/model_chain_live_krea.js"):
            assert not (root / name).exists()

    def test_a_roll_only_happens_when_something_calls_for_one(self, client):
        """Nothing observes anything, so there is no path from "text changed" to
        "a model was asked". Merely having a session is not a roll."""
        session = mc_creative_krea.Creative()

        assert session.last is None
        assert client.calls == []


# --------------------------------------------------------------------------- #
# The hook that writes the prompt
# --------------------------------------------------------------------------- #


class Processing:
    """The half of a StableDiffusionProcessing the Creative hook touches."""

    def __init__(self, prompt="car"):
        self.prompt = prompt
        self.extra_generation_params = {}


class _Result:
    """The half of a Processed the notice is written on."""

    def __init__(self):
        self.comments = ""


@pytest.fixture
def script():
    import model_chain_krea_creative as creative_script

    return creative_script.ScriptKreaCreative()


def rolled(client, source="car", creativity=10):
    """One roll, packaged the way the processing hook packages it."""
    stored = mc_creative_krea.settings()
    stored["creativity"] = creativity
    list(mc_creative_krea.creative.roll(source, stored))
    return mc_creative_krea.prepare(mc_creative_krea.creative.last)


def panel_values(creativity=10, seed=director.RANDOM_SEED, anti=True,
                 mode=director.VARY, modes=None, fixed=None, excluded=None):
    """What Forge hands ``before_process`` after the enabled flag.

    The panel's own controls, in the order ``ui()`` returns them: the three
    scalars and then the axis controls, three per axis -- mode, pinned value,
    excluded ids. Built from the library rather than written out, for the same
    reason the panel is.
    """
    values = [creativity, seed, anti]
    for key in library_module.library().axis_keys:
        values.extend([(modes or {}).get(key, mode),
                       (fixed or {}).get(key),
                       list((excluded or {}).get(key) or ())])
    return values


def generate(script, prompt="car", enabled=True, timeout=20.0, **panel):
    """One press of Generate, driven the way the host drives it.

    On a thread with a deadline, because the failure this whole change is about
    is a hook that never returns: an LLM request that waits for the image job to
    finish, inside the image job. A deadlocked hook would otherwise hang the
    test run rather than fail it.
    """
    import threading

    p = Processing(prompt)
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


class TestTheProcessingHook:
    def test_creative_mode_off_leaves_the_prompt_alone(self, script, client):
        p = generate(script, "car", enabled=False)

        assert p.prompt == "car"
        assert p.extra_generation_params == {}
        assert client.calls == []

    def test_the_generation_is_given_the_expansion(self, script, client):
        client.answers = ["A low-angle impasto painting of a car."]
        p = generate(script, "car")

        assert p.prompt == "A low-angle impasto painting of a car."

    def test_the_hook_asks_for_the_expansion_itself(self, script, client):
        """The whole of the change this file was rewritten for.

        The hook used to apply an expansion somebody else had requested, because
        an LLM run waited for the host to stop generating and one asked for from
        inside a running image job would have waited for the job that was
        waiting for it. That ordering is now declared rather than choreographed
        in a browser, so the request happens here -- which is what makes a press
        of Generate sufficient on its own.
        """
        generate(script, "a car in the rain")

        assert len(client.calls) == 1
        assert "a car in the rain" in client.turn

    def test_a_literal_command_arrives_with_it(self, script, client):
        """What the Pinned LoRAs field used to do, said in the prompt instead."""
        client.answers = ["A painted car."]
        p = generate(script, "[[<lora:film:0.8>]] car")

        assert p.prompt == "<lora:film:0.8> A painted car."

    def test_everything_needed_to_roll_it_again_is_recorded(self, script, client):
        p = generate(script, "car")
        recorded = p.extra_generation_params

        assert recorded[mc_infotext.CREATIVE_MODE] == "True"
        assert recorded[mc_infotext.CREATIVE_CREATIVITY] == 10
        assert recorded[mc_infotext.CREATIVE_SOURCE] == "car"
        assert recorded[mc_infotext.CREATIVE_SEED] != director.RANDOM_SEED
        assert recorded[mc_infotext.CREATIVE_LLM_SEED] == director.derive(
            recorded[mc_infotext.CREATIVE_SEED])[1]
        assert "medium=" in recorded[mc_infotext.CREATIVE_RECIPE]

    def test_the_expanded_paragraph_is_not_recorded_twice(self, script, client):
        """It is already the generation's own prompt line."""
        client.answers = ["A tall white lighthouse under storm light."]
        p = generate(script)

        assert "A tall white lighthouse" not in "".join(
            str(value) for value in p.extra_generation_params.values())

    def test_each_press_gets_its_own_roll(self, script, client):
        client.answers = ["First.", "Second."]
        first = generate(script, "car")
        second = generate(script, "car")

        assert (first.prompt, second.prompt) == ("First.", "Second.")
        assert len(client.calls) == 2

    def test_the_panel_values_are_read_from_the_press_and_not_the_file(self, script,
                                                                      client, store):
        """The slider somebody just moved wins over the last value saved."""
        mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: 1})
        generate(script, "car", creativity=10)

        assert mc_creative_krea.creative.last.creativity == 10

    def test_an_api_request_with_no_panel_behind_it_uses_the_saved_settings(
            self, script, client, store):
        """``before_process`` is reachable without a page: the API passes the
        script's arguments through, and a caller that sends only the flag must
        get the installation's own configuration rather than an invented one."""
        mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: 4})
        p = Processing("car")
        script.before_process(p, True)

        assert mc_creative_krea.creative.last.creativity == 4
        assert p.prompt != "car"

    def test_nothing_from_an_earlier_roll_reaches_this_generation(self, script,
                                                                  client):
        """There is no state between a roll and the generation that uses it, so
        an earlier roll cannot be picked up by a later press. This used to be a
        token with a lifetime; now it is the absence of anywhere to put one."""
        rolled(client, source="something else")
        client.answers = ["A fresh roll for this press."]
        p = generate(script, "car")

        assert p.prompt == "A fresh roll for this press."

    def test_an_empty_prompt_is_left_alone(self, script, client):
        p = generate(script, "   ")

        assert p.prompt == "   "
        assert client.calls == []

    def test_a_generation_nested_inside_the_roll_does_not_roll_again(self, script,
                                                                     client):
        """Stage 2 runs with ``p.scripts`` unset, so this is belt and braces --
        and the cost of getting it wrong is not a duplicate image but a second
        language-model request begun while the first is still streaming."""
        nested = Processing("car")

        def reenter(*args, **kwargs):
            script.before_process(nested, True, *panel_values())
            return "An expanded Krea prompt."

        client.answers = []
        original = client.stream_chat

        def stream_chat(*args, **kwargs):
            reenter()
            return original(*args, **kwargs)

        client.stream_chat = stream_chat
        outer = generate(script, "car")

        assert outer.prompt != "car"
        assert nested.prompt == "car"
        assert len(client.calls) == 1

    def test_a_roll_that_fails_still_generates_what_the_user_typed(self, script,
                                                                   client, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("llama-server is not running")
            yield  # pragma: no cover - generator shape only

        monkeypatch.setattr(sessions, "krea", explode)
        p = generate(script, "car")

        assert p.prompt == "car"
        assert p.extra_generation_params == {}

    def test_a_roll_that_fails_says_so_on_the_result(self, script, client,
                                                     monkeypatch):
        """Reported from a user's log: llama-server would not start, so every
        Creative generation quietly used the prompt as typed. Falling back is
        right; falling back in silence is what makes it read as "Creative Mode
        does nothing" rather than "the writer is down"."""
        def explode(*args, **kwargs):
            raise RuntimeError("llama-server is not running")
            yield  # pragma: no cover - generator shape only

        monkeypatch.setattr(sessions, "krea", explode)
        p = generate(script, "car")
        processed = _Result()
        script.postprocess(p, processed)

        assert "Creative Mode did not write this prompt" in processed.comments
        assert "llama-server is not running" in processed.comments

    def test_a_roll_that_worked_says_nothing_at_all(self, script, client):
        p = generate(script, "car")
        processed = _Result()
        script.postprocess(p, processed)

        assert processed.comments == ""

    def test_the_reason_is_not_carried_into_the_next_generation(self, script, client,
                                                                monkeypatch):
        """One generation's failure explains that generation and no other."""
        real = sessions.krea
        attempts: list = []

        def first_one_fails(*args, **kwargs):
            attempts.append(1)
            if len(attempts) > 1:
                return real(*args, **kwargs)

            def failing():
                raise RuntimeError("llama-server is not running")
                yield  # pragma: no cover - generator shape only

            return failing()

        monkeypatch.setattr(sessions, "krea", first_one_fails)
        generate(script, "car")
        script.postprocess(Processing("car"), _Result())

        second = generate(script, "car")
        processed = _Result()
        script.postprocess(second, processed)

        assert second.prompt != "car"
        assert processed.comments == ""

    def test_creative_mode_off_says_nothing(self, script, client):
        """The hook is not reached, so there is nothing to explain."""
        p = generate(script, "car", enabled=False)
        processed = _Result()
        script.postprocess(p, processed)

        assert processed.comments == ""

    def test_a_checkpoint_that_is_not_krea_generates_the_typed_prompt(
            self, script, client, monkeypatch):
        """Refusing the expansion is right; refusing the generation is not. The
        user pressed Generate, and a short prompt is exactly what an SD 1.5
        checkpoint wanted anyway."""
        monkeypatch.setattr(mc_creative_krea, "checkpoint_objection",
                            lambda: "The loaded checkpoint is not Krea 2.")
        p = generate(script, "car")

        assert p.prompt == "car"
        assert client.calls == []


class TestThereIsNothingToHandOver:
    """The arming token is gone, and this is what replaced it.

    It was permission for exactly one native generation to spend a roll made
    ahead of the click, and it existed because the roll and the generation were
    two separate server calls with a browser in between. They are one call now.
    The properties the token bought are still properties -- they are just facts
    about the shape of the code rather than a mechanism to be checked at
    runtime -- so what is asserted here is that no way of holding a roll for a
    later generation came back.
    """

    def test_the_session_holds_nothing_but_the_last_roll(self, client):
        rolled(client)
        session = mc_creative_krea.creative

        assert session.last is not None
        assert not [name for name in vars(session) if "arm" in name]

    def test_the_session_has_no_way_to_authorise_a_later_generation(self, client):
        for gone in ("arm", "consume", "disarm", "armed"):
            assert not hasattr(mc_creative_krea.creative, gone)

    def test_the_last_roll_is_a_record_and_not_a_permission(self, script, client):
        """It is what the diagnostics drawer reads. A generation must never
        substitute it, or closing and reopening the tab would silently reuse the
        art direction of whatever was rolled before."""
        rolled(client, source="a lighthouse")
        client.answers = ["Something written for this press."]
        p = generate(script, "car")

        assert p.prompt == "Something written for this press."


class TestTheGpuIsGivenBack:
    def test_a_roll_leaves_no_workload_holding_the_card(self, client):
        list(mc_creative_krea.creative.roll("car"))

        assert mc_broker.active() is None

    def test_a_failed_roll_gives_it_back_too(self, client, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("llama-server is not running")
            yield  # pragma: no cover - generator shape only

        monkeypatch.setattr(sessions, "krea", explode)
        list(mc_creative_krea.creative.roll("car"))

        assert mc_broker.active() is None


# --------------------------------------------------------------------------- #
# What replaced the pinned LoRAs
# --------------------------------------------------------------------------- #


class TestThePinnedLorasFieldIsGone:
    """The control, its setting and its profile field, all of them.

    A text box beside a prompt box that accepted one kind of syntax was a second
    prompt input with a narrower grammar. ``[[<lora:name:weight>]]`` in the
    prompt does the same job for every kind of syntax, in the place a person
    would have typed the tag anyway -- so what is asserted here is absence, and
    the presence of the thing that replaced it is asserted in
    ``test_krea_literals.py``.
    """

    def test_the_setting_is_gone(self):
        assert "loras" not in mc_creative_krea.settings()
        assert not hasattr(mc_creative_krea, "LORAS")

    def test_the_helpers_that_parsed_it_are_gone(self):
        assert not hasattr(mc_creative_krea, "lora_suffix")
        assert not hasattr(mc_creative_krea, "pinned_tags")

    def test_a_profile_no_longer_carries_one(self):
        import mc_creative_profiles

        assert "loras" not in mc_creative_profiles.FIELDS
        assert "loras" not in mc_creative_profiles.factory()

    def test_a_profile_written_by_an_older_build_still_loads(self):
        """One key nobody reads, and every other field restored."""
        import mc_creative_profiles

        values = mc_creative_profiles.normalise(
            {"creativity": 7, "loras": "<lora:film:0.8>"})

        assert values["creativity"] == 7
        assert "loras" not in values

    def test_an_older_image_still_says_which_tags_it_used(self):
        """Read and shown, never applied. There is nothing left to apply it to."""
        setup = mc_infotext.creative_setup(
            {mc_infotext.CREATIVE_MODE: "True",
             mc_infotext.CREATIVE_LORAS: "<lora:film:0.8>"})

        assert setup.loras == "<lora:film:0.8>"


# --------------------------------------------------------------------------- #
# The surfaces
# --------------------------------------------------------------------------- #


class TestTheTxt2imgSurface:
    @pytest.fixture
    def built(self, store, host):
        import model_chain_krea_creative as creative_script

        instance = creative_script.ScriptKreaCreative()
        instance.ui(False)
        return instance

    def test_it_is_txt2img_only(self, built):
        import modules.scripts as scripts

        assert built.show(False) is scripts.AlwaysVisible
        assert built.show(True) is None

    def test_the_stage_is_off_by_default_and_says_so_on_its_pipeline_row(self,
                                                                           built):
        """The switch is on the collapsed pipeline row, and the row says what
        the stage will do -- section 3.3. The drawer behind it is no longer
        hidden by the toggle: configuring a stage before arming it is an
        ordinary thing to want, and a drawer that emptied itself when the stage
        was off made turning it on the only way to set it up.

        "Bypassed" and not "Off", because a card that is visible, expandable
        and full of settings needs its summary to be the thing that says
        whether any of them will run.
        """
        assert built.components["enabled"].value is False
        assert built.components["creative_line"].value.startswith("Bypassed")

    def test_the_two_stages_this_script_owns_have_their_own_switches(self, built):
        """Creative and Spatial are peers. Two rows, two switches, and neither
        one shows or hides the other."""
        assert built.components["enabled"].value is False
        assert built.components["spatial_enabled"].value is False
        assert built.components["enabled"] is not built.components["spatial_enabled"]

    def test_the_slider_offers_the_whole_scale(self, built):
        slider = built.components["creativity"]

        assert (slider.minimum, slider.maximum, slider.step) == (0, 10, 1)

    def test_there_are_three_controls_for_every_axis_in_the_library(self, built, lib):
        """Mode, pinned value, exclusions -- in that order, per axis. The order is
        the contract ``before_process`` unpacks by."""
        assert len(built.components["axes"]) == len(lib.axis_keys) * 3

    def test_every_axis_offers_the_three_modes(self, built):
        for control in built.components["axes"][::3]:
            assert [value for _label, value in control.choices] == list(director.MODES)

    def test_an_axis_can_exclude_its_own_treatments(self, built, lib):
        """Exclusion is a modifier of Vary, so it is a multiselect over the same
        vocabulary the axis varies within -- not a fourth mode."""
        excluded = built.components["axes"][2]
        expected = [variant.identifier for variant in lib.axis(lib.axis_keys[0]).variants]

        assert excluded.multiselect is True
        assert [value for _label, value in excluded.choices] == expected

    def test_a_fixed_dropdown_lists_that_axis_own_variants(self, built, lib):
        values = built.components["axes"][1]
        expected = [variant.identifier for variant in lib.axis(lib.axis_keys[0]).variants]

        assert [value for _label, value in values.choices] == expected

    def test_no_handler_touches_anything_outside_the_panel(self, built):
        """The visible prompt box keeps the short phrase the user is iterating
        on, and the panel has no business with any other native control. Asserted
        as "every component every handler names is one of ours" rather than as a
        list of forbidden ids, so a handler wired to something new fails here."""
        ours = set()
        for component in built.components.values():
            for entry in (component if isinstance(component, list) else [component]):
                ours.add(id(entry))

        for kwargs in _handlers(built):
            for component in (list(kwargs.get("inputs") or [])
                              + list(kwargs.get("outputs") or [])):
                assert id(component) in ours

    def test_the_source_is_the_generation_own_prompt(self, built, client):
        """Not a component read out of the page. ``p.prompt`` is what the host is
        about to generate from, it is present whether the press came from the
        tab or the API, and it is still there when the tab is not."""
        p = generate(built, "a lighthouse in a storm")

        assert "a lighthouse in a storm" in client.turn
        assert mc_creative_krea.creative.last.source == "a lighthouse in a storm"

    def test_the_panel_has_no_hidden_plumbing_for_the_browser(self, built):
        """The gate's half is gone: no hidden button for JavaScript to press and
        no hidden box for it to poll. Both existed only to sequence a roll and a
        click from the page, and both are how a closed tab used to strand a
        generation.

        What is hidden now is progressive disclosure, which is the opposite kind
        of thing: a row for a decision nobody has made yet, revealed by a press on
        a control that is right there. Those are allowed by identity -- they are
        the panel's own rows, editors and axis controls -- so a *new* hidden
        component with no visible way to reach it still fails here.
        """
        disclosure = {id(entry)
                      for name in ("rows", "editors", "axes")
                      for entry in built.components.get(name) or []}

        for name, component in built.components.items():
            if isinstance(component, list):
                continue
            if getattr(component, "visible", True) is False:
                # The controls that hide with the toggle are a different thing:
                # they are visible the moment Creative Mode is on. "name_row" is
                # the Save As name box, revealed by the Save As button.
                #
                # "spatial_state" is the one component here that is hidden and
                # stays hidden, and it is allowed by name because it is an
                # *input* rather than a channel: the layout editor writes the
                # serialized canvas into it when somebody saves, and Forge reads
                # it with every other control when Generate is pressed. Nothing
                # polls it and no generation waits for it, which is the whole of
                # the difference between it and the token box the old gate had.
                # tests/test_krea_spatial_js.py is where that is checked rather
                # than asserted.
                #
                # "literal_row" is the third kind again, and the one worth
                # naming carefully: it is hidden when neither Creative nor
                # Spatial is on, and its two boxes still reach the next
                # generation while it is. That is deliberate -- section 3.3 of
                # the Literal Prompts intent, where hidden explicitly does not
                # mean inactive -- so what makes it acceptable is not that it is
                # reachable but that it is *reported*: the Image Pipeline's
                # Prompt row says how many literals are active whenever this row
                # is off screen and its boxes are not empty.
                assert (name in ("creativity", "status", "controls", "name_row",
                                 "spatial_group", "spatial_state", "literal_row")
                        or id(component) in disclosure), name

    def test_what_ui_returns_is_what_before_process_reads(self, store, host, client):
        """The contract that is easiest to break and hardest to notice.

        ``ui()``'s return list becomes this script's arguments, positionally, and
        ``before_process`` unpacks them by position. Get the order wrong and the
        seed arrives as the Creativity, which produces a valid roll at a wrong
        setting rather than an error. So the panel is built, its own values are
        taken off its own components, and they are passed through in exactly the
        order the panel returned them.
        """
        import model_chain_krea_creative as creative_script

        instance = creative_script.ScriptKreaCreative()
        returned = instance.ui(False)

        # Distinct values, placed by identity rather than by position, so the
        # test knows what each control was set to without assuming where it
        # sits. Two are enough: a Creativity and a seed that arrived in each
        # other's places would both still be plausible integers.
        marks = {id(instance.components["enabled"]): True,
                 id(instance.components["creativity"]): 7,
                 id(instance.components["seed"]): 12345}
        values = [marks.get(id(component), getattr(component, "value", None))
                  for component in returned]

        p = Processing("car")
        instance.before_process(p, *values)
        last = mc_creative_krea.creative.last

        assert (last.creativity, last.creative_seed) == (7, 12345)
        assert p.prompt != "car"

    def test_no_handler_on_the_panel_queues(self, built):
        """Nothing here does work worth queueing, and nothing here starts,
        stops or waits for a generation any more."""
        for kwargs in _handlers(built):
            assert kwargs.get("queue") is False

    def test_it_adds_no_image_controls(self, built):
        """Forge owns the image. Steps, size, sampler, CFG and the image seed
        stay where they already are, and duplicating one would put a number in
        the metadata that never happened."""
        made = " ".join(str(getattr(component, "label", "") or "")
                        for component in built.components.values()
                        if not isinstance(component, list)).casefold()
        for native in ("steps", "sampler", "scheduler", "cfg", "width", "height",
                       "batch"):
            assert native not in made


class TestTheLlmStudioSurface:
    @pytest.fixture
    def built(self, store, host):
        return panel.build()

    def test_it_offers_the_same_controls(self, built):
        assert built["creative"].value is False
        assert built["controls"].visible is False
        assert (built["creativity"].minimum, built["creativity"].maximum) == (0, 10)

    def test_it_has_the_same_three_controls_per_axis(self, built, lib):
        assert len(built["axes"]) == len(lib.axis_keys) * 3

    def test_it_adds_no_image_backend(self, built):
        """This mode writes prompts. It settles nothing about how they would be
        drawn, and a sampler or a gallery here would be a second image pipeline
        living in the prompt-authoring tab."""
        import gradio as gr

        assert not any(isinstance(component, gr.Gallery)
                       for component in built.values()
                       if not isinstance(component, list))

    def test_both_surfaces_share_one_settings_file(self, built, store):
        mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: 8})

        assert mc_creative_krea.settings()["creativity"] == 8

    def test_both_surfaces_go_through_one_director(self):
        """One engine, or the two would drift within a release."""
        root = Path(__file__).resolve().parent.parent
        for name in ("mc_llm_krea_panel.py", "scripts/model_chain_krea_creative.py"):
            source = (root / name).read_text(encoding="utf-8")
            assert "krea import director" in source or "krea import" in source
            assert "expressions" not in source


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

    def blur(self, **kwargs):
        return self._record("blur", kwargs)


def _handlers(script) -> list[dict]:
    """Every event handler the panel registered on its own components."""
    found: list[dict] = []
    for component in script.components.values():
        for entry in (component if isinstance(component, list) else [component]):
            for _kind, kwargs in getattr(entry, "_callbacks", []) or []:
                found.append(kwargs)
    return found


# --------------------------------------------------------------------------- #
# The checkpoint guard
# --------------------------------------------------------------------------- #


class TestTheCheckpointGuard:
    def test_krea_2_is_allowed(self, monkeypatch, host):
        import mc_arch

        monkeypatch.setattr(mc_arch, "detect_loaded_engine",
                            lambda: mc_arch.by_key("krea2"))
        assert mc_creative_krea.checkpoint_objection() == ""

    def test_another_architecture_is_named_in_the_refusal(self, monkeypatch, host):
        import mc_arch

        monkeypatch.setattr(mc_arch, "detect_loaded_engine", lambda: mc_arch.by_key("sdxl"))
        assert "SDXL" in mc_creative_krea.checkpoint_objection()

    def test_a_checkpoint_nobody_can_identify_is_not_refused(self, monkeypatch, host):
        """Detection reads a header and cannot see inside every GGUF or repacked
        build. A guard that blocked real Krea 2 checkpoints would be worse than
        no guard."""
        import mc_arch

        monkeypatch.setattr(mc_arch, "detect_loaded_engine", lambda: mc_arch.UNKNOWN)
        monkeypatch.setattr(mc_arch, "detect_from_checkpoint_name", lambda name: mc_arch.UNKNOWN)
        assert mc_creative_krea.checkpoint_objection() == ""

    def test_it_refuses_the_roll_before_the_model_is_touched(self, client, monkeypatch):
        monkeypatch.setattr(mc_creative_krea, "checkpoint_objection",
                            lambda: "Select a Krea 2 checkpoint.")
        events = list(mc_creative_krea.creative.roll("car", guard_checkpoint=True))

        assert events[-1].kind == sessions.FAILED
        assert client.calls == []

    def test_llm_studio_does_not_consult_it(self, client, monkeypatch):
        """Writing a prompt settles nothing about which checkpoint draws it."""
        monkeypatch.setattr(mc_creative_krea, "checkpoint_objection",
                            lambda: "Select a Krea 2 checkpoint.")
        list(mc_creative_krea.creative.roll("car"))

        assert len(client.calls) == 1


# --------------------------------------------------------------------------- #
# Exclusions: a modifier of Vary, and an absolute one
# --------------------------------------------------------------------------- #


class TestExclusions:
    """"Vary this axis, but never that treatment."

    The gap the old panel left: a user could allow everything or pin one thing,
    and had no way to say which two of a hundred treatments they never want. It
    is a modifier of Vary rather than a fourth mode, because "never harsh noon"
    is a statement about *how* to vary and making it a mode would force somebody
    who wants two treatments gone to stop varying altogether.
    """

    def test_an_excluded_variant_is_never_chosen(self, lib):
        """Over three hundred rolls at the position that activates everything."""
        banned = {variant.identifier for variant in lib.axis("medium").variants[:4]}
        settings = axes(medium=director.AxisSetting(mode=director.VARY,
                                                    excluded_ids=frozenset(banned)))
        for seed in range(300):
            recipe = roll(creativity=10, seed=seed, settings=settings)
            for item in recipe.items:
                assert item.variant_id not in banned

    def test_the_rest_of_the_axis_is_still_reachable(self, lib):
        """Excluding four treatments is not excluding the axis."""
        banned = {variant.identifier for variant in lib.axis("medium").variants[:4]}
        settings = axes(medium=director.AxisSetting(mode=director.VARY,
                                                    excluded_ids=frozenset(banned)))
        chosen = {item.variant_id for seed in range(200)
                  for item in roll(creativity=10, seed=seed, settings=settings).items
                  if item.axis == "medium"}

        assert len(chosen) > 3

    def test_excluding_everything_skips_the_axis_and_says_so(self, lib):
        """Never the excluded value anyway, and never silence about it.

        The two wrong answers are both quiet: choosing an excluded treatment
        because nothing else was left, or dropping the axis with nothing said and
        letting somebody conclude that exclusions break Vary."""
        everything = {variant.identifier for variant in lib.axis("texture").variants}
        settings = axes(texture=director.AxisSetting(mode=director.VARY,
                                                     excluded_ids=frozenset(everything)))
        recipe = roll(creativity=10, settings=settings)

        assert all(item.axis != "texture" for item in recipe.items)
        assert any("Texture" in note for note in recipe.notes)

    def test_a_skipped_axis_does_not_cost_another_axis_its_line(self, lib):
        """An axis that cannot be directed must not consume an activation slot.

        Left in the pool it would be drawn, produce nothing, and leave the roll
        one line shorter -- so a user who excluded one small axis would find the
        others directed less often with nothing to explain it."""
        everything = {variant.identifier for variant in lib.axis("texture").variants}
        settings = axes(texture=director.AxisSetting(mode=director.VARY,
                                                     excluded_ids=frozenset(everything)))
        for seed in range(40):
            with_exclusion = roll(creativity=6, seed=seed, settings=settings)
            assert with_exclusion.items, "the roll directed nothing at all"

    def test_exclusion_does_not_apply_to_a_fixed_pin(self, lib):
        """A pin is the decision. An exclusion that could cancel one would leave
        the axis meaning two things at once."""
        pinned = lib.axis("texture").variants[0].identifier
        settings = axes(texture=director.AxisSetting(
            mode=director.FIXED, fixed_id=pinned, excluded_ids=frozenset({pinned})))
        recipe = roll(creativity=5, settings=settings)

        assert any(item.variant_id == pinned for item in recipe.items)

    def test_the_setting_takes_a_list_as_readily_as_a_set(self):
        """It arrives from a JSON file, a Gradio multiselect and an infotext
        line, and one of the three hands over a list every time."""
        setting = director.AxisSetting(mode=director.VARY, excluded_ids=["a", "b", "a"])

        assert setting.excluded_ids == frozenset({"a", "b"})
        assert setting.excludes("a") and not setting.excludes("c")

    def test_they_survive_the_settings_file(self, store, lib):
        banned = [lib.axis("lighting").variants[0].identifier]
        mc_creative_krea.remember(**{
            mc_creative_krea.AXIS_MODES: {"lighting": director.VARY},
            mc_creative_krea.EXCLUDED_VALUES: {"lighting": banned}})

        assert mc_creative_krea.settings()["excluded_values"] == {"lighting": banned}
        assert mc_creative_krea.axis_settings()["lighting"].excluded_ids == frozenset(banned)

    def test_an_id_the_library_no_longer_has_is_dropped(self, store):
        """A saved configuration outlives the package version it was written
        against, and an exclusion naming a treatment that has gone excludes
        nothing -- so keeping it would leave somebody reading a list of
        protections they no longer have."""
        mc_creative_krea.remember(**{
            mc_creative_krea.EXCLUDED_VALUES: {"lighting": ["no_such_treatment"]},
            mc_creative_krea.FIXED_VALUES: {"medium": "no_such_medium"}})
        stored = mc_creative_krea.settings()

        assert stored["excluded_values"] == {}
        assert stored["fixed_values"] == {}


# --------------------------------------------------------------------------- #
# The fresh install directs nothing
# --------------------------------------------------------------------------- #


class TestTheFactoryConfiguration:
    """A new configuration contains no art direction the user did not ask for.

    The package as delivered set nine of the ten axes to Vary, which meant a
    fresh install arrived with nine decisions already taken -- none of them made
    by the user, and none of them visible until they opened the drawer and read
    ten rows of controls.
    """

    def test_every_axis_is_natural(self, lib, store):
        stored = mc_creative_krea.settings()

        assert set(stored["axis_modes"]) == set(lib.axis_keys)
        assert set(stored["axis_modes"].values()) == {director.NATURAL}

    def test_nothing_is_pinned_and_nothing_is_excluded(self, store):
        stored = mc_creative_krea.settings()

        assert stored["fixed_values"] == {}
        assert stored["excluded_values"] == {}

    def test_a_fresh_configuration_directs_nothing_at_any_position(self, store):
        settings = mc_creative_krea.axis_settings()
        for creativity in range(0, 11):
            recipe = director.roll("car", creativity, 7, settings)
            assert recipe.items == ()
            assert recipe.brief == ""

    def test_an_axis_setting_defaults_to_natural(self):
        """A missing axis -- one a later package adds, one an older profile does
        not mention -- fails neutral rather than silently varying."""
        assert director.AxisSetting().mode == director.NATURAL

    def test_the_packages_defaults_agree(self, lib):
        assert set((lib.defaults.get("axis_modes") or {}).values()) == {director.NATURAL}
        assert not lib.defaults.get("fixed_values")
        assert not lib.defaults.get("excluded_values")

    def test_the_creativity_position_is_still_five(self, store):
        """Neutral means no *decisions*, not a feature turned down: somebody who
        adds one direction should get it at the middle of the scale."""
        assert mc_creative_krea.settings()["creativity"] == 5


# --------------------------------------------------------------------------- #
# Named profiles
# --------------------------------------------------------------------------- #


class TestProfiles:
    @pytest.fixture
    def data(self, tmp_path, monkeypatch, host):
        """Point the profile store at a throwaway WebUI data directory."""
        from modules import paths

        monkeypatch.setattr(paths, "data_path", str(tmp_path), raising=False)
        return tmp_path

    def test_the_factory_profile_is_always_there_and_is_neutral(self, data, lib):
        assert profiles.FACTORY in profiles.choices()
        values = profiles.get(profiles.FACTORY)

        assert set(values["axis_modes"].values()) == {director.NATURAL}
        assert values["fixed_values"] == {} and values["excluded_values"] == {}

    def test_a_saved_profile_round_trips_through_the_file(self, data, store, lib):
        banned = [lib.axis("lighting").variants[0].identifier]
        mc_creative_krea.remember(**{
            mc_creative_krea.CREATIVITY: 8,
            mc_creative_krea.AXIS_MODES: {"lighting": director.VARY},
            mc_creative_krea.EXCLUDED_VALUES: {"lighting": banned}})
        profiles.save("Editorial", profiles.from_settings())

        # A fresh read of the file, as a restarted WebUI would do.
        loaded = profiles.get("Editorial")
        assert loaded["creativity"] == 8
        assert loaded["axis_modes"]["lighting"] == director.VARY
        assert loaded["excluded_values"]["lighting"] == banned

    def test_a_profile_does_not_carry_whether_creative_mode_is_on(self, data, store):
        """A profile says how the feature behaves. Whether it runs at all is a
        decision made at the moment somebody presses Generate."""
        mc_creative_krea.remember(**{mc_creative_krea.ENABLED: True})
        saved = profiles.from_settings()

        assert "enabled" not in saved
        assert "enabled" not in profiles.FIELDS
        assert set(profiles.FIELDS) & set(profiles.EXCLUDED_FIELDS) == set()

    def test_applying_a_profile_leaves_the_toggle_alone(self, data, store):
        mc_creative_krea.remember(**{mc_creative_krea.ENABLED: True})
        profiles.save("Quiet", profiles.factory())
        stored, complaint = profiles.apply("Quiet")

        assert complaint == ""
        assert stored["enabled"] is True

    def test_the_panel_opens_on_the_profile_the_settings_came_from(self, data, store):
        profiles.save("Editorial", profiles.factory())
        profiles.apply("Editorial")

        assert profiles.selected() == "Editorial"

    def test_a_selection_naming_a_deleted_profile_falls_back(self, data, store):
        profiles.save("Editorial", profiles.factory())
        profiles.apply("Editorial")
        profiles.delete("Editorial")

        assert profiles.selected() == profiles.FACTORY

    def test_nothing_is_applied_merely_by_opening_the_panel(self, data, store):
        """A panel that reapplied its default every time a tab opened would
        silently discard whatever the last tab adjusted."""
        profiles.save("Editorial", profiles.factory())
        profiles.set_default("Editorial")
        mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: 9,
                                     mc_creative_krea.AXIS_MODES:
                                         {"medium": director.VARY}})

        assert profiles.selected() == "Editorial"
        assert mc_creative_krea.settings()["creativity"] == 9
        assert mc_creative_krea.settings()["axis_modes"]["medium"] == director.VARY

    def test_the_chosen_default_survives_a_restart(self, data, store):
        profiles.save("Editorial", profiles.factory())
        profiles.set_default("Editorial")

        # Nothing cached: every reader opens the file.
        assert profiles.default_name() == "Editorial"
        name, _values, complaint = profiles.default_profile()
        assert (name, complaint) == ("Editorial", "")

    def test_a_default_that_has_gone_falls_back_to_factory_and_says_so(self, data,
                                                                      store):
        """A panel that refused to build over a missing preset would be a
        Creative Mode nobody can turn on to fix it.

        The profile is removed from the file rather than through :func:`delete`,
        because delete tidies the default up after itself. What is being tested
        is the case nothing tidied: a hand-edited store, or one written by a
        version that is no longer installed."""
        import json

        profiles.save("Editorial", profiles.factory())
        profiles.set_default("Editorial")
        with open(profiles.path(), "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "profiles": {}, "default": "Editorial"}, handle)

        name, values, complaint = profiles.default_profile()
        assert name == profiles.FACTORY
        assert set(values["axis_modes"].values()) == {director.NATURAL}
        assert "Editorial" in complaint

    def test_deleting_the_default_leaves_no_dangling_nomination(self, data, store):
        profiles.save("Editorial", profiles.factory())
        profiles.set_default("Editorial")
        profiles.delete("Editorial")

        assert profiles.default_name() == profiles.FACTORY
        assert profiles.default_profile()[2] == ""

    def test_the_factory_profile_cannot_be_deleted_or_overwritten(self, data, store):
        with pytest.raises(profiles.ProfileError):
            profiles.delete(profiles.FACTORY)
        with pytest.raises(profiles.ProfileError):
            profiles.save(profiles.FACTORY, profiles.factory())

    def test_there_is_a_built_in_that_varies_everything(self, data, store, lib):
        """The neutral default took something away that people had, and one
        click is the honest way to give it back: the package shipped nine of its
        ten axes on Vary, so anybody using Creative Mode before the rebuild had
        been running something close to this without choosing it."""
        values = profiles.get(profiles.SPREAD)

        assert profiles.SPREAD in profiles.choices()
        assert set(values["axis_modes"]) == set(lib.axis_keys)
        assert set(values["axis_modes"].values()) == {director.VARY}
        assert values["fixed_values"] == {} and values["excluded_values"] == {}

    def test_neither_built_in_can_be_deleted_or_overwritten(self, data, store):
        for name in profiles.BUILT_IN:
            with pytest.raises(profiles.ProfileError):
                profiles.delete(name)
            with pytest.raises(profiles.ProfileError):
                profiles.save(name, profiles.factory())

    def test_a_built_in_can_be_the_chosen_default(self, data, store):
        profiles.set_default(profiles.SPREAD)

        assert profiles.default_profile()[0] == profiles.SPREAD
        assert profiles.default_profile()[2] == ""

    def test_applying_it_directs_every_axis(self, data, store, lib):
        stored, complaint = profiles.apply(profiles.SPREAD)

        assert complaint == ""
        assert set(stored["axis_modes"].values()) == {director.VARY}
        assert len(mc_creative_krea.active_axes(stored)) == len(lib.axis_keys)

    def test_deleting_one_profile_keeps_the_others(self, data, store):
        profiles.save("One", profiles.factory())
        profiles.save("Two", profiles.factory())
        profiles.delete("One")

        assert profiles.names() == ["Two"]

    def test_an_unnamed_save_is_refused_rather_than_written(self, data, store):
        with pytest.raises(profiles.ProfileError):
            profiles.save("   ", profiles.factory())

        assert profiles.names() == []

    def test_the_store_is_outside_the_extension_directory(self, data):
        """Reinstalling or updating the extension must not throw profiles away,
        which is why this file lives beside Model Chain's presets in the WebUI's
        data directory rather than in the extension folder."""
        import os

        location = profiles.path()
        assert os.path.basename(location) == "krea_creative_profiles.json"
        assert str(Path(__file__).resolve().parent.parent) not in location

    def test_a_damaged_store_reads_as_empty_rather_than_raising(self, data, store):
        with open(profiles.path(), "w", encoding="utf-8") as handle:
            handle.write("{not json at all")

        assert profiles.names() == []
        assert profiles.default_profile()[0] == profiles.FACTORY

    def test_a_profile_written_by_an_older_version_still_loads(self, data, store):
        """Fields added since are filled from the current defaults rather than
        blanking the control -- which is what makes adding exclusions to a schema
        that had none cost nobody their saved work."""
        profiles.save("Old", {"creativity": 7, "axis_modes": {"medium": director.VARY}})
        loaded = profiles.get("Old")

        assert loaded["creativity"] == 7
        assert loaded["excluded_values"] == {}
        assert loaded["anti_repetition"] is True

    def test_a_mode_that_is_not_a_mode_reads_as_natural(self, data, store):
        profiles.save("Odd", {"axis_modes": {"medium": "enthusiastic"}})

        assert profiles.get("Odd")["axis_modes"]["medium"] == director.NATURAL


# --------------------------------------------------------------------------- #
# The compact panel
# --------------------------------------------------------------------------- #


class TestTheCompactPanel:
    """What the drawer shows is what the user has decided, and nothing else."""

    @pytest.fixture
    def data(self, tmp_path, monkeypatch, host):
        from modules import paths

        monkeypatch.setattr(paths, "data_path", str(tmp_path), raising=False)
        return tmp_path

    @pytest.fixture
    def built(self, store, data, host):
        import model_chain_krea_creative as creative_script

        instance = creative_script.ScriptKreaCreative()
        instance.ui(False)
        return instance

    def panel(self, built):
        return built.panel

    def rendered(self, built, updates) -> dict:
        """One render's updates, keyed by the component they are for."""
        return {id(component): update
                for component, update in zip(built.panel.outputs(), updates)}

    def test_a_fresh_drawer_has_no_axis_rows_at_all(self, built):
        assert all(row.visible is False for row in built.components["rows"])
        assert all(editor.visible is False for editor in built.components["editors"])

    def test_it_says_so_in_words_rather_than_showing_an_empty_table(self, built):
        assert "None" in built.components["summary"].value

    def test_every_natural_axis_is_offered_as_something_to_add(self, built, lib):
        offered = [value for _label, value in built.components["add"].choices]

        assert offered == list(lib.axis_keys)

    def test_adding_a_direction_shows_that_row_and_nothing_else(self, built, lib):
        panel = self.panel(built)
        updates = self.rendered(built, _fire(built, panel.add, "medium"))

        assert updates[id(panel.rows["medium"])]["visible"] is True
        assert updates[id(panel.rows["lighting"])]["visible"] is False

    def test_a_new_direction_starts_with_nothing_chosen(self, built):
        """Section 5.1. Adding an axis opens a question rather than answering
        one: the row appears, the picker is empty, and the brief is exactly
        what it would have been until a treatment is chosen."""
        panel = self.panel(built)
        updates = self.rendered(built, _fire(built, panel.add, "medium"))

        assert updates[id(panel.treatments["medium"])]["value"] == []
        assert "no treatments chosen" in updates[id(panel.labels["medium"])]["value"]

    def test_an_added_axis_is_no_longer_offered_to_be_added(self, built):
        panel = self.panel(built)
        updates = self.rendered(built, _fire(built, panel.add, "medium"))
        offered = [value for _label, value in updates[id(panel.add)]["choices"]]

        assert "medium" not in offered

    def test_an_empty_direction_leaves_the_axis_natural(self, built):
        """The row is a fact about the panel; the axis underneath is untouched.
        A half-made decision the Director acted on would be a direction nobody
        chose."""
        _fire(built, self.panel(built).add, "medium")

        assert mc_creative_krea.settings()["axis_modes"]["medium"] == director.NATURAL
        assert "medium" in mc_creative_krea.settings()["directions"]

    def test_two_treatments_become_a_seeded_pool(self, built, lib):
        """Section 5.3: 2+ selected is a pool the Creative seed chooses from,
        which underneath is Vary with everything else excluded."""
        panel = self.panel(built)
        chosen = [variant.identifier for variant in lib.axis("medium").variants[:2]]
        _fire(built, panel.add, "medium")
        _fire(built, panel.treatments["medium"], chosen)

        stored = mc_creative_krea.settings()
        assert stored["axis_modes"]["medium"] == director.VARY
        assert stored["fixed_values"].get("medium") is None
        allowed = set(chosen)
        assert not allowed & set(stored["excluded_values"]["medium"])

    def test_one_treatment_is_a_fixed_treatment(self, built, lib):
        """Section 5.3: 1 selected repeats every roll."""
        panel = self.panel(built)
        pinned = lib.axis("medium").variants[0].identifier
        _fire(built, panel.add, "medium")
        _fire(built, panel.treatments["medium"], [pinned])

        stored = mc_creative_krea.settings()
        assert stored["axis_modes"]["medium"] == director.FIXED
        assert stored["fixed_values"]["medium"] == pinned
        assert not stored["excluded_values"].get("medium")

    def test_clearing_the_picker_stops_the_axis_directing(self, built, lib):
        """Section 5.3 again, in the direction that matters: emptying a picker
        must leave the axis out of the brief entirely, not leave the last
        treatment pinned where nobody can see it."""
        panel = self.panel(built)
        pinned = lib.axis("medium").variants[0].identifier
        _fire(built, panel.add, "medium")
        _fire(built, panel.treatments["medium"], [pinned])
        _fire(built, panel.treatments["medium"], [])

        stored = mc_creative_krea.settings()
        assert stored["axis_modes"]["medium"] == director.NATURAL
        assert stored["fixed_values"].get("medium") is None
        # The row stays, because the user has not taken it away.
        assert "medium" in stored["directions"]

    def test_a_selection_round_trips_through_the_settings(self, built, lib):
        """The picker reads what it wrote. Selection and storage are one
        mapping in two directions, and halves that disagree are a panel that
        forgets a choice the moment the tab is reloaded."""
        panel = self.panel(built)
        chosen = [variant.identifier for variant in lib.axis("style").variants[:3]]
        _fire(built, panel.add, "style")
        updates = self.rendered(built, _fire(built, panel.treatments["style"], chosen))

        assert sorted(updates[id(panel.treatments["style"])]["value"]) == sorted(chosen)

    def test_the_machine_facing_controls_are_never_shown(self, built, lib):
        """Mode, pinned value and exclusions still travel to the generation --
        that contract is unchanged -- but the picker is the only thing a user
        touches. Two controls for one decision is how a panel ends up showing
        an exclusion list for an axis that is no longer varying."""
        panel = self.panel(built)
        _fire(built, panel.add, "medium")
        _fire(built, panel.treatments["medium"],
              [variant.identifier for variant in lib.axis("medium").variants[:2]])

        assert all(editor.visible is False for editor in built.components["editors"])

    def test_removing_a_row_takes_its_treatments_with_it(self, built, lib):
        """One action, because on screen they are one thing. A row removed but
        still pinned would keep directing the brief from somewhere the user can
        no longer see it."""
        panel = self.panel(built)
        _fire(built, panel.add, "medium")
        _fire(built, panel.treatments["medium"],
              [lib.axis("medium").variants[0].identifier])
        updates = self.rendered(built, _click(built, "row", "medium", "remove"))

        stored = mc_creative_krea.settings()
        assert updates[id(panel.rows["medium"])]["visible"] is False
        assert stored["axis_modes"]["medium"] == director.NATURAL
        assert stored["fixed_values"].get("medium") is None
        assert "medium" not in stored["directions"]

    def test_the_row_summarises_the_decision_in_one_line(self, built, lib):
        panel = self.panel(built)
        pinned = lib.axis("medium").variants[0]
        _fire(built, panel.add, "medium")
        updates = self.rendered(built, _fire(built, panel.treatments["medium"],
                                             [pinned.identifier]))

        assert updates[id(panel.labels["medium"])]["value"] == (
            f"**{lib.axis('medium').label}** · {pinned.label}")

    def test_a_pool_row_says_how_many_it_may_choose_between(self, built, lib):
        panel = self.panel(built)
        chosen = [variant.identifier for variant in lib.axis("lighting").variants[:3]]
        _fire(built, panel.add, "lighting")
        updates = self.rendered(built, _fire(built, panel.treatments["lighting"],
                                             chosen))
        line = updates[id(panel.labels["lighting"])]["value"]

        assert "3 treatments" in line and "Creative seed" in line

    def test_what_was_not_chosen_reaches_the_settings_file_as_exclusions(self, built,
                                                                          lib):
        """The picker asks which treatments to use; the Director has always
        been told which ones not to. This is that translation, written down."""
        panel = self.panel(built)
        variants = [variant.identifier for variant in lib.axis("lighting").variants]
        _fire(built, panel.add, "lighting")
        _fire(built, panel.treatments["lighting"], variants[:2])

        assert (mc_creative_krea.settings()["excluded_values"]["lighting"]
                == variants[2:])

    def test_every_treatment_picker_answers_on_input_rather_than_on_change(self,
                                                                            built):
        """Every handler here rewrites the whole panel, including the control
        that fired it. ``change`` fires when the server sets a value, so wiring
        these to it would be a feedback loop."""
        panel = self.panel(built)
        for key in panel.keys:
            kinds = {kind for kind, _kwargs in panel.treatments[key]._callbacks}
            assert kinds == {"input"}

    def test_the_machine_facing_controls_are_written_only_by_a_render(self, built):
        """They are outputs, never inputs. A handler on one of them would be a
        second writer for a value the picker already owns."""
        panel = self.panel(built)
        for key in panel.keys:
            for control in (panel.modes[key], panel.fixed[key], panel.excluded[key]):
                assert control._callbacks == []

    def test_saving_a_profile_from_the_panel_stores_what_is_on_screen(self, built,
                                                                      data, lib):
        panel = self.panel(built)
        pinned = lib.axis("medium").variants[0].identifier
        _fire(built, panel.add, "medium")
        _fire(built, panel.treatments["medium"], [pinned])
        panel.profile_name.value = "Editorial"
        _click(built, "profile", "create")

        assert profiles.get("Editorial")["axis_modes"]["medium"] == director.FIXED
        assert profiles.get("Editorial")["fixed_values"]["medium"] == pinned

    def test_a_profile_remembers_a_row_that_has_no_treatments_yet(self, built, data):
        """An unfinished direction is still work somebody did. A profile that
        carried the settings but not the rows would load as a panel that had
        silently forgotten it."""
        panel = self.panel(built)
        _fire(built, panel.add, "medium")
        panel.profile_name.value = "Half done"
        _click(built, "profile", "create")

        assert profiles.get("Half done")["directions"] == ["medium"]

    def test_loading_a_profile_redraws_every_control(self, built, data, lib):
        panel = self.panel(built)
        pinned = lib.axis("medium").variants[0].identifier
        _fire(built, panel.add, "medium")
        _fire(built, panel.treatments["medium"], [pinned])
        panel.profile_name.value = "Editorial"
        _click(built, "profile", "create")
        _click(built, "row", "medium", "remove")

        updates = self.rendered(built, _fire(built, panel.profile, "Editorial"))
        assert updates[id(panel.rows["medium"])]["visible"] is True
        assert updates[id(panel.treatments["medium"])]["value"] == [pinned]
        assert updates[id(panel.modes["medium"])]["value"] == director.FIXED

    def test_a_choice_made_in_the_panel_reaches_the_writer(self, built, client, lib):
        """The whole point of the panel, end to end, driven the way a browser
        drives it: add a direction, pin a treatment, press Generate, and the
        pinned treatment is in the brief the writer was handed.

        Reported as "I fixed Style to anime and the result did not work" — which
        it did not, because llama-server would not start (see
        ``TestTheProcessingHook``). This is what proves the other half."""
        panel = self.panel(built)
        pinned = lib.axis("style").variants[0]
        _fire(built, panel.add, "style")
        _apply(built, _fire(built, panel.treatments["style"], [pinned.identifier]))

        # The values Forge sends: whatever the panel's own controls hold.
        values = [getattr(control, "value", None) for control in built.arguments]
        values[0] = True
        p = Processing("a car")
        built.before_process(p, *values)

        recipe = mc_creative_krea.creative.last.recipe
        assert [item.variant_id for item in recipe.items] == [pinned.identifier]
        assert pinned.label.casefold() in client.turn.casefold()
        assert p.prompt != "a car"

    def test_what_the_panel_holds_is_what_the_generation_reads(self, built):
        """The contract that is easiest to break and hardest to notice: the
        argument list is positional both ways."""
        panel = self.panel(built)

        spatial = [built.components["spatial_enabled"],
                   built.components["spatial_compose"],
                   built.components["spatial_state"]]
        literal = [built.components["literal_positive"],
                   built.components["literal_negative"]]

        assert built.arguments[:2] == [built.components["enabled"],
                                       built.components["creativity"]]
        assert built.arguments[2:4] == list(panel.settings_controls)
        assert built.arguments[4:-5] == list(panel.axis_controls)
        # The Spatial block goes last and stays last, and that is load-bearing
        # rather than tidy: mc_plan reads it off the *end* of this tuple,
        # because the axis block in the middle is a variable length and the two
        # ends are the two that can be found without counting. The Literal
        # Prompt boxes therefore go in the middle, immediately before it.
        assert built.arguments[-5:-3] == literal
        assert built.arguments[-3:] == spatial

    def test_it_says_what_the_directions_cost_before_the_image_starts(self, built,
                                                                       lib):
        """Reported as "why does the reading step take ten seconds". The brief
        is different every roll, so no cache can hold it, and it is the whole of
        what a press pays before the writer says anything. That is a decision
        somebody makes on this panel, so the panel is where the number belongs
        -- the progress bar is the first place they see it and the last place
        they can act on it."""
        panel = self.panel(built)
        _fire(built, panel.add, "medium")
        _fire(built, panel.treatments["medium"],
              [variant.identifier for variant in lib.axis("medium").variants[:2]])
        _fire(built, panel.add, "lighting")
        updates = self.rendered(built, _fire(
            built, panel.treatments["lighting"],
            [variant.identifier for variant in lib.axis("lighting").variants[:2]]))
        said = updates[id(panel.cost)]["value"]

        assert "characters of brief" in said
        assert "of reading" in said

    def test_a_row_with_no_treatments_costs_nothing(self, built):
        """It directs nothing, so it adds nothing to the brief -- and the cost
        line has to agree with the axis mode, or one of them is lying."""
        panel = self.panel(built)
        updates = self.rendered(built, _fire(built, panel.add, "medium"))

        assert "No directions" in updates[id(panel.cost)]["value"]

    def test_a_configuration_that_directs_nothing_costs_nothing(self, built):
        assert "No directions" in built.components["cost"].value
        assert mc_creative_panel.brief_cost()[0] == 0

    def test_more_directions_cost_more(self, built, store, lib):
        """Linear, and it is what makes the trade-off real: the brief is the one
        part of the request that cannot come out of the model's cache."""
        keys = list(lib.axis_keys)
        costs = []
        for count in (1, 3, 6):
            mc_creative_krea.remember(**{
                mc_creative_krea.CREATIVITY: 10,
                mc_creative_krea.AXIS_MODES: {
                    key: (director.VARY if position < count else director.NATURAL)
                    for position, key in enumerate(keys)}})
            costs.append(mc_creative_panel.brief_cost()[0])

        assert costs[0] < costs[1] < costs[2]

    def test_a_lower_creativity_is_a_shorter_brief(self, built, store, lib):
        """The expressions themselves get shorter down the scale, so the same
        directions cost less to read."""
        modes = {lib.axis_keys[0]: director.VARY}
        sizes = []
        for creativity in (3, 10):
            mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: creativity,
                                         mc_creative_krea.AXIS_MODES: modes})
            sizes.append(mc_creative_panel.brief_cost()[0])

        assert sizes[0] < sizes[1]

    def test_the_seconds_come_from_this_machine_s_own_measurement(self, built, store,
                                                                  lib, monkeypatch):
        """The same store the progress bar predicts from, so the panel and the
        bar cannot disagree about what a press costs."""
        import mc_progress

        mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: 10,
                                     mc_creative_krea.AXIS_MODES:
                                         {lib.axis_keys[0]: director.VARY}})
        monkeypatch.setattr(mc_progress, "rate_for",
                            lambda keys: 1.0 if "krea:read" in keys[-1] else 0.0)
        characters, seconds = mc_creative_panel.brief_cost()

        assert seconds == pytest.approx(float(characters))

    def test_the_rate_it_reads_is_this_backbone_s_own(self, built, store, lib):
        """Backbones differ at this by more than anything else the estimate
        models, and not in the direction their sizes suggest -- so a switch must
        not leave the panel quoting the previous model's speed."""
        import mc_llm_progress

        keys = mc_llm_progress.writer_rates("krea:write")

        assert keys[-1] == "krea:write"
        assert len(keys) in (1, 2)

    def test_the_slider_updates_the_cost_line_with_it(self, built):
        """The number beside the directions must not be one action behind the
        control that changes it -- moving the slider is exactly when somebody
        looks at what it costs."""
        panel = self.panel(built)
        moved = [kwargs for _kind, kwargs
                 in built.components["creativity"]._callbacks]

        assert moved, "the slider registers no handler"
        assert panel.cost in (moved[0].get("outputs") or [])

    def test_the_slider_says_when_it_has_nothing_to_scale(self, built, store):
        """Reported as "the creativity slider is not working". It was not: at
        Creativity 10 with every axis Natural the panel promised "extreme
        direction on every eligible axis" over a brief of zero characters."""
        said = mc_creative_panel.describe_creativity(10)

        assert "nothing to scale" in said
        assert "Add a direction" in said

    def test_it_says_what_the_scale_means_once_there_is_something_to_scale(
            self, built, store, lib):
        mc_creative_krea.remember(**{mc_creative_krea.AXIS_MODES:
                                     {lib.axis_keys[0]: director.VARY}})
        said = mc_creative_panel.describe_creativity(10)

        assert "nothing to scale" not in said
        assert "1 direction" in said

    def test_it_names_the_other_way_a_direction_produces_nothing(self, built, store,
                                                                 lib):
        """0 and 1 add no direction by design, so a user who has just added one
        and set the slider to 0 sees exactly what "nothing to scale" looks
        like, for a completely different reason."""
        mc_creative_krea.remember(**{mc_creative_krea.AXIS_MODES:
                                     {lib.axis_keys[0]: director.VARY}})
        said = mc_creative_panel.describe_creativity(1)

        assert "no direction by design" in said
        assert "start at 2" in said

    def test_the_status_line_says_whether_anything_is_directed(self, built, store,
                                                               lib):
        """The only Creative text on screen while the drawer is shut. "Creative
        Mode is on" is true and, on a fresh configuration, deeply misleading on
        its own."""
        assert "No directions are set" in mc_creative_panel.active_note()

        mc_creative_krea.remember(**{mc_creative_krea.AXIS_MODES:
                                     {lib.axis_keys[0]: director.VARY}})
        assert mc_creative_panel.active_note() == "1 direction set."

    def test_the_cost_line_names_the_writing_as_well_as_the_reading(self, built,
                                                                    store, lib,
                                                                    monkeypatch):
        """A user told "4s of reading" who waits twenty seconds is owed the
        other sixteen: the expansion itself is usually the larger half, and no
        control on this panel changes it."""
        import mc_progress

        monkeypatch.setattr(mc_progress, "rate_for", lambda keys: 0.05)
        monkeypatch.setattr(mc_progress, "measured",
                            lambda key, default=0.0: 400.0 if key == "krea:reply" else default)

        assert "20s of writing" in mc_creative_panel.describe_cost()

    def test_one_render_answers_for_every_control_it_owns(self, built):
        """The contract every handler relies on: a render is positional, so an
        outputs list and a render that disagree by one put every update after
        the mismatch on the wrong control."""
        panel = self.panel(built)

        assert len(panel.render()) == len(panel.outputs())
        assert len(panel.render(told="hello")) == len(panel.outputs())

    def test_a_library_that_will_not_load_leaves_a_sentence_and_no_panel(self, store,
                                                                        monkeypatch):
        """Creative Mode then has no vocabulary to direct with, and a drawer of
        empty dropdowns would invite somebody to configure a feature that cannot
        run."""
        import model_chain_krea_creative as creative_script
        from prompt_master.krea import library as library_module

        def broken():
            raise library_module.LibraryError("the package is not there")

        monkeypatch.setattr(library_module, "library", broken)
        instance = creative_script.ScriptKreaCreative()
        returned = instance.ui(False)

        assert instance.panel is None
        # The Creative toggle and slider, and nothing of Creative's after them --
        # but the Literal Prompt boxes and the three Spatial controls are still
        # there. Neither needs a vocabulary: Spatial Layout places a box without
        # one, and a literal payload is protected from language models that were
        # never going to run. A library that will not load takes Creative Mode
        # down and leaves both standing.
        assert len(returned) == (2 + creative_script.LITERAL_CONTROLS
                                 + creative_script.SPATIAL_CONTROLS)

    def test_both_surfaces_build_the_same_panel(self, built):
        """One implementation. Two would disagree within a release, and the
        first thing they would disagree about is what a fresh install does."""
        root = Path(__file__).resolve().parent.parent
        for name in ("mc_llm_krea_panel.py", "scripts/model_chain_krea_creative.py"):
            source = (root / name).read_text(encoding="utf-8")
            assert "mc_creative_panel" in source
            assert "_axis_table" not in source

    def test_the_panel_never_offers_a_fourth_mode(self, built):
        """Exclusion is a modifier of Vary. A mode for it would make "vary this
        but not that" impossible to say."""
        for control in built.components["axes"][::3]:
            assert [value for _label, value in control.choices] == list(director.MODES)


def _fire(script, component, value):
    """Drive one component's ``input`` handler the way the browser would."""
    for kind, kwargs in component._callbacks:
        if kind != "input":
            continue
        component.value = value
        inputs = list(kwargs.get("inputs") or [])
        return kwargs["fn"](*[value if entry is component
                              else getattr(entry, "value", None) for entry in inputs])
    raise AssertionError("that component has no input handler")


def _apply(script, updates):
    """Write one render's updates back onto the components, as Gradio does.

    The panel's mode, pinned-value and exclusion controls are outputs now: the
    treatment picker writes the settings, and a render is what carries the
    result back to the three controls that travel to the generation. A test
    that read those controls without applying the render would be reading the
    page as it was before the click.
    """
    for component, update in zip(script.panel.outputs(), updates):
        if isinstance(update, dict) and "value" in update:
            component.value = update["value"]
    return updates


def _click(script, *parts):
    """Press the button with this extension-owned id."""
    import model_chain_krea_creative as creative_script

    wanted = creative_script.ident(*parts)
    for button in script.components.get("buttons") or []:
        if getattr(button, "elem_id", "") != wanted:
            continue
        for kind, kwargs in button._callbacks:
            if kind != "click":
                continue
            inputs = list(kwargs.get("inputs") or [])
            return kwargs["fn"](*[getattr(entry, "value", None) for entry in inputs])
    raise AssertionError(f"no button {wanted} with a click handler")


# --------------------------------------------------------------------------- #
# Pasting an image back
# --------------------------------------------------------------------------- #


class TestOrdinaryPasteReproducesTheImage:
    """The bug this half of the work exists for.

    A Creative generation's recorded ``Prompt:`` line is the *expanded* prompt,
    because Creative Mode assigns it before Forge writes infotext. So a paste
    that left Creative Mode on would hand a finished Krea paragraph back to the
    writer as though it were a short idea and expand it again, and what came out
    would be a picture of the prompt of the picture.
    """

    @pytest.fixture
    def built(self, store, host):
        import model_chain_krea_creative as creative_script

        instance = creative_script.ScriptKreaCreative()
        instance.ui(False)
        return instance

    @pytest.fixture
    def made(self, built, client):
        """One Creative image, and its infotext as a paste would parse it."""
        client.answers = ["A tall white lighthouse under storm light, "
                          "photographed on large-format film."]
        p = generate(built, "a lighthouse")
        return p, _parsed(p)

    def test_the_prompt_line_is_the_prompt_the_image_model_saw(self, made):
        p, params = made

        assert params["Prompt"] == p.prompt

    def test_pasting_switches_creative_mode_off(self, built, made):
        _p, params = made

        assert _pasted_value(built, "enabled", params) is False

    def test_the_writer_is_not_called_when_the_paste_is_regenerated(self, built, made,
                                                                    client):
        """The acceptance test, end to end: paste, press Generate, and the
        prompt handed to the image model is byte-for-byte the recorded one with
        no writer call at all."""
        p, params = made
        before = len(client.calls)
        enabled = _pasted_value(built, "enabled", params)

        again = generate(built, params["Prompt"], enabled=bool(enabled))

        assert again.prompt == p.prompt
        assert len(client.calls) == before
        assert again.extra_generation_params == {}

    def test_an_ordinary_image_leaves_the_toggle_where_it_is(self, built):
        """An image made without Creative Mode should not be able to switch a
        feature off any more than on."""
        assert _pasted_value(built, "enabled", {"Steps": "20"}) is None

    def test_the_paste_says_what_it_did(self, built, made):
        _p, params = made
        said = _pasted_value(built, "status", params)

        assert "Creative Mode was disabled" in said

    def test_every_recorded_key_is_forwarded_by_the_send_to_buttons(self, built, made):
        """The buttons forward by exact name, so a key that is not declared
        simply does not arrive -- and "restore the setup" would find half a
        record."""
        _p, params = made
        declared = set(mc_infotext.creative_paste_field_names())
        written = {key for key in params if key.startswith("Krea ")}

        assert written <= declared


class TestRestoringTheCreativeSetup:
    """The other half: the workflow, restored on purpose and never by surprise."""

    @pytest.fixture
    def built(self, store, host):
        import model_chain_krea_creative as creative_script

        instance = creative_script.ScriptKreaCreative()
        instance.ui(False)
        return instance

    @pytest.fixture
    def made(self, built, client, lib):
        """One image made with a real configuration: one axis varying, one
        treatment excluded, and the other nine axes Natural."""
        client.answers = ["A painted lighthouse."]
        p = generate(built, "a lighthouse", mode=director.NATURAL,
                     modes={"medium": director.VARY},
                     excluded={"medium": [lib.axis("medium").variants[0].identifier]})
        params = _parsed(p)
        _pasted_value(built, "status", params)  # the capture a real paste performs
        return p, params

    def test_the_record_carries_the_configuration_the_image_was_made_under(self, made):
        _p, params = made

        assert params[mc_infotext.CREATIVE_AXES]
        assert params[mc_infotext.CREATIVE_EXCLUDED]
        assert mc_infotext.CREATIVE_ANTI in params

    def test_the_axes_round_trip_through_the_infotext(self, made, lib):
        _p, params = made
        modes, fixed = mc_infotext.parse_creative_axes(params[mc_infotext.CREATIVE_AXES])
        excluded = mc_infotext.parse_creative_exclusions(
            params[mc_infotext.CREATIVE_EXCLUDED])

        assert modes["medium"] == director.VARY
        assert excluded["medium"] == [lib.axis("medium").variants[0].identifier]
        assert fixed == {}

    def test_natural_axes_are_not_written_down_at_all(self, made):
        """Absence is what Natural means, so a configuration that directs two
        axes records two axes rather than ten."""
        _p, params = made
        recorded = params[mc_infotext.CREATIVE_AXES]

        assert "natural" not in recorded

    def test_restoring_puts_the_source_phrase_back_in_the_prompt_box(self, built, made):
        import model_chain_krea_creative as creative_script

        prompt_update, _enabled, _status, _view, *_spatial = \
            creative_script._restore_setup(False)

        assert prompt_update["value"] == "a lighthouse"

    def test_restoring_is_the_only_thing_that_writes_to_the_prompt_box(self, built,
                                                                       made):
        """Every other handler on this panel stays inside it."""
        ours = {id(entry) for component in built.components.values()
                for entry in (component if isinstance(component, list) else [component])}
        outside = []
        for kwargs in _handlers(built):
            for component in list(kwargs.get("outputs") or []):
                if id(component) not in ours:
                    outside.append(kwargs["fn"].__name__)

        assert outside in ([], ["_restore_setup"])

    def test_restoring_turns_creative_mode_back_on(self, built, made, store):
        """The paste turned it off so the picture would reproduce. Continuing
        from the *source* is the opposite request, and a short idea generated
        with the writer switched off is a bare phrase handed to Krea 2."""
        import model_chain_krea_creative as creative_script

        mc_creative_krea.remember(**{mc_creative_krea.ENABLED: False})
        _prompt, enabled, status, _view, *_spatial = \
            creative_script._restore_setup(False)

        assert mc_creative_krea.settings()["enabled"] is True
        assert enabled["value"] is True
        assert "Creative Mode is on again" in status

    def test_only_that_button_ever_turns_it_on(self, built):
        """Loading a profile, adding a direction, excluding a treatment, drawing
        a box: none of them may switch a feature on. A profile says how Creative
        Mode behaves; whether it runs is a decision made at the Generate button.

        Two features can be switched on now -- Creative Mode and Spatial Layout
        -- and the rule is the same for both and for the same reason. The paste
        turned them off so the picture would reproduce; a restore button is the
        request to do the opposite, and they are the only ones. Two buttons
        because they are two features: each one turns on its own and leaves the
        other exactly as it found it.
        """
        import inspect

        import model_chain_krea_creative as creative_script

        source = Path(creative_script.__file__).read_text(encoding="utf-8")
        buttons = (inspect.getsource(creative_script._restore_setup)
                   + inspect.getsource(creative_script._restore_spatial))
        turning_on = [line.strip() for line in source.splitlines()
                      if "ENABLED: True" in line]

        assert len(turning_on) == 2
        for line in turning_on:
            assert line in buttons, line
        # And neither reaches into the other's switch.
        assert "mc_spatial.ENABLED: True" not in \
            inspect.getsource(creative_script._restore_setup)
        assert "mc_creative_krea.ENABLED: True" not in \
            inspect.getsource(creative_script._restore_spatial)

    def test_it_says_which_of_the_two_things_it_did(self, built, made):
        import model_chain_krea_creative as creative_script

        _prompt, _enabled, armed, _view, *_spatial = creative_script._restore_setup(True)
        _prompt, _enabled, fresh, _view, *_spatial = creative_script._restore_setup(False)

        assert "replays it exactly" in armed
        assert "new roll from the same idea" in fresh


class TestReplayingARecordedRecipe:
    """A replay reproduces the art direction. It never pretends to be a roll."""

    def test_the_recorded_ids_come_back_verbatim(self, client, lib):
        recipe = roll(creativity=10, seed=11)
        replayed = director.replay("car", recipe.creativity, recipe.creative_seed,
                                   recipe.llm_seed, recipe.compact)

        assert replayed.compact == recipe.compact
        assert replayed.llm_seed == recipe.llm_seed
        assert replayed.replayed is True

    def test_nothing_in_it_is_drawn(self, lib):
        """Same answer at any history, which a re-roll cannot promise: the draw
        is weighted by recent choices, and a machine six months later has other
        recent choices."""
        recipe = roll(creativity=10, seed=11)
        crowded = director.replay("car", 10, recipe.creative_seed, recipe.llm_seed,
                                  recipe.compact)

        assert crowded.compact == recipe.compact

    def test_a_line_the_library_no_longer_has_is_dropped_and_named(self):
        replayed = director.replay("car", 6, 3, 4, "medium=no_such_thing")

        assert replayed.items == ()
        assert any("no_such_thing" in note for note in replayed.notes)

    def test_an_armed_replay_directs_the_next_roll_and_only_that_one(self, client):
        recipe = roll(creativity=10, seed=11)
        mc_creative_krea.replay.arm(mc_creative_krea.ReplayPlan(
            creativity=10, creative_seed=recipe.creative_seed,
            llm_seed=recipe.llm_seed, recipe=recipe.compact))

        list(mc_creative_krea.creative.roll("car", varying()))
        first = mc_creative_krea.creative.last.recipe
        list(mc_creative_krea.creative.roll("car", varying()))
        second = mc_creative_krea.creative.last.recipe

        assert first.compact == recipe.compact and first.replayed is True
        assert second.replayed is False

    def test_it_is_still_exactly_one_writer_call(self, client):
        recipe = roll(creativity=10, seed=11)
        mc_creative_krea.replay.arm(mc_creative_krea.ReplayPlan(
            creativity=10, creative_seed=recipe.creative_seed,
            llm_seed=recipe.llm_seed, recipe=recipe.compact))
        list(mc_creative_krea.creative.roll("car", varying()))

        assert len(client.calls) == 1

    def test_it_holds_no_prompt_and_no_model_output(self, client):
        """The arming token that used to sit between a handler and a generation
        held a *finished prompt* -- made before the click, waiting to be spent.
        This holds variant ids the user chose, which is why it is allowed to
        exist at all."""
        plan = mc_creative_krea.ReplayPlan(creativity=5, creative_seed=1, llm_seed=2,
                                           recipe="medium=oil_impasto")
        values = " ".join(str(value) for value in vars(plan).values())

        assert "expanded" not in vars(plan)
        assert len(values) < 200

    def test_a_failed_roll_does_not_leave_it_armed(self, client, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("llama-server is not running")
            yield  # pragma: no cover - generator shape only

        monkeypatch.setattr(sessions, "krea", explode)
        mc_creative_krea.replay.arm(mc_creative_krea.ReplayPlan(
            creativity=5, creative_seed=1, llm_seed=2, recipe="medium=oil_impasto"))
        list(mc_creative_krea.creative.roll("car"))

        assert mc_creative_krea.replay.pending is None

    def test_a_replay_is_not_written_into_the_recent_memory(self, client, store):
        """Its ids were recorded when they were first drawn. Writing them again
        would push a user's own reproduction away from what they asked to
        reproduce."""
        recipe = roll(creativity=10, seed=11)
        mc_creative_krea.forget_history()
        mc_creative_krea.replay.arm(mc_creative_krea.ReplayPlan(
            creativity=10, creative_seed=recipe.creative_seed,
            llm_seed=recipe.llm_seed, recipe=recipe.compact))
        list(mc_creative_krea.creative.roll("car", varying()))

        assert mc_creative_krea.history() == []


def _parsed(p) -> dict:
    """One generation's infotext, as the host would write and re-parse it."""
    from conftest import parse_generation_parameters, quote

    line = ", ".join(f"{key}: {quote(value)}"
                     for key, value in p.extra_generation_params.items())
    parsed = parse_generation_parameters(f"{p.prompt}\n{line}" if line else p.prompt)
    parsed["Prompt"] = p.prompt
    return parsed


def _pasted_value(script, name, params):
    """What the paste field for ``name`` answers for this infotext."""
    component = script.components[name]
    for field in script.infotext_fields:
        if field.component is not component:
            continue
        if field.function is not None:
            return field.function(params)
        return params.get(field.label)
    raise AssertionError(f"no paste field for {name}")


# --------------------------------------------------------------------------- #
# The panel on a narrow screen, and under somebody else's theme
# --------------------------------------------------------------------------- #


class TestTheStylesheet:
    @pytest.fixture
    def css(self):
        return (Path(__file__).resolve().parent.parent
                / "style.css").read_text(encoding="utf-8")

    @pytest.fixture
    def section(self, css):
        return css.split("Krea Creative Mode", 1)[1]

    def test_it_selects_nothing_gradio_generated(self, section):
        """A ``.svelte-`` hash is regenerated on every Gradio build, and a
        layout that depends on one is a layout that breaks on an upgrade."""
        assert ".svelte" not in section

    def test_it_states_no_colour_of_its_own(self, section):
        """Every colour is one of the host's custom properties, which is what
        lets a theme -- Lobe included -- decide what all of this looks like."""
        import re

        for line in section.splitlines():
            stripped = line.strip()
            if ":" not in stripped or stripped.startswith(("*", "/", "#", "@", ".")):
                continue
            assert not re.search(r":\s*#[0-9a-fA-F]{3,8}\b", stripped), stripped
            assert not re.search(r":\s*rgba?\(", stripped), stripped

    def test_a_long_summary_wraps_rather_than_scrolling(self, section):
        """A flex child's minimum is its content width by default, which is how
        "excludes a, b and c" becomes a horizontal scroll bar on a phone."""
        rule = section.split(".mc-creative-direction-summary {", 1)[1].split("}", 1)[0]

        assert "min-width: 0" in rule
        assert "flex-wrap: wrap" in section.split(".mc-creative-direction {", 1)[1] \
                                            .split("}", 1)[0]

    def test_everything_stacks_into_one_column_on_a_phone(self, section):
        """320-375px: the rows become columns, and nothing states a width a
        narrow viewport cannot honour."""
        assert "@media (max-width: 480px)" in section
        narrow = _braced(section, "@media (max-width: 480px) {")

        assert "flex-direction: column" in narrow
        for line in narrow.splitlines():
            # A width in pixels is a width a 320px viewport may not have.
            assert "px" not in line.split(":", 1)[-1], line

    def test_the_panel_classes_are_this_extensions_own(self):
        """One prefix, shared by both surfaces, because it is one layout."""
        import mc_creative_panel

        assert mc_creative_panel.classes("editor") == ["mc-creative-editor"]


def _braced(text: str, opening: str) -> str:
    """The block ``opening`` starts, up to its own closing brace."""
    body = text.split(opening, 1)[1]
    depth = 1
    for position, character in enumerate(body):
        depth += (character == "{") - (character == "}")
        if depth == 0:
            return body[:position]
    raise AssertionError(f"{opening!r} is never closed")


# --------------------------------------------------------------------------- #
# The package's own acceptance cases
# --------------------------------------------------------------------------- #


class TestTheAcceptanceCases:
    """Every case the data package ships, checked off by name.

    The package is independently versioned and is meant to grow. This is what
    stops it growing a promise that nothing here tests: the last assertion fails
    if a case appears with no test claiming it.
    """

    COVERED = {
        "one_call": "TestTheOneCallRule",
        "no_planner_llm": "TestTheDirectorNeverAsksAModel",
        "no_live": "TestThereIsNoLiveMode",
        "natural_omit": "TestNatural",
        "fixed_repeat": "TestFixed",
        "source_wins": "TestTheSourcePromptWins",
        "c2_tier": "TestScaling",
        "c10_tier": "TestScaling",
        "fixed_seed": "TestSeeds",
        "anti_repeat": "TestAntiRepetition",
    }

    @pytest.fixture
    def cases(self):
        document = json.loads((LIBRARY / "tests" / "acceptance_cases.json")
                              .read_text(encoding="utf-8"))
        return {case["id"]: case for case in document["acceptance_cases"]}

    def test_every_shipped_case_has_a_test_class_claiming_it(self, cases):
        assert set(cases) == set(self.COVERED), (
            "the creativity package's acceptance cases and this file have drifted")

    def test_the_classes_named_above_exist(self, cases):
        for name in set(self.COVERED.values()):
            assert name in globals(), f"{name} is claimed but not defined"

    def test_the_worked_examples_still_describe_what_happens(self):
        """The package ships five worked examples. Two of them make claims a
        test can check directly against the Director."""
        locked = json.loads((LIBRARY / "examples" / "locked_medium.json")
                            .read_text(encoding="utf-8"))
        recipe = roll(locked["source"], creativity=locked["creativity"])
        assert "medium" in recipe.locked

        fixed = json.loads((LIBRARY / "examples" / "fixed_texture.json")
                           .read_text(encoding="utf-8"))
        settings = axes(texture=director.AxisSetting(
            mode=director.FIXED, fixed_id=fixed["setting"]["texture"]["fixed_id"]))
        recipe = roll(fixed["source"], creativity=fixed["creativity"], settings=settings)
        assert any(item.variant_id == fixed["setting"]["texture"]["fixed_id"]
                   for item in recipe.items)


class TestOneWarmServerForTheWholePlan:
    """Section 13: consecutive LLM calls share one process.

    The Creative Writer and the Spatial Composer are two requests a fraction of
    a second apart. Between them the roll used to hand the card back to the
    image side -- and the only way the broker can hand image VRAM back is to
    stop llama-server, which is the very process the Composer was about to send
    its one request to. The cost is a second GGUF load and a cold prompt cache
    in the middle of one generation, to free room nothing uses until the
    Composer has finished with the card anyway.
    """

    def layout(self, mode, regions=("a",)):
        from prompt_master.krea import spatial as spatial_module

        return types.SimpleNamespace(regions=tuple(regions), compose_mode=mode,
                                     notes=(), unreadable=False)

    def test_a_smart_layout_means_a_composer_is_still_to_run(self):
        from prompt_master.krea import spatial as spatial_module

        assert mc_creative_krea._composer_follows(self.layout(spatial_module.SMART))

    def test_a_direct_merge_asks_no_model_anything(self):
        from prompt_master.krea import spatial as spatial_module

        assert not mc_creative_krea._composer_follows(self.layout(spatial_module.DIRECT))

    def test_an_empty_canvas_composes_nothing(self):
        from prompt_master.krea import spatial as spatial_module

        assert not mc_creative_krea._composer_follows(
            self.layout(spatial_module.SMART, regions=()))

    def test_no_layout_at_all_is_the_end_of_the_llm_work(self):
        assert not mc_creative_krea._composer_follows(None)

    def test_the_roll_does_not_reclaim_before_a_composer(self, client, monkeypatch):
        from prompt_master.krea import spatial as spatial_module

        asked = []
        monkeypatch.setattr(mc_creative_krea, "hand_back_vram",
                            lambda *a, **k: asked.append(True) or 0)
        stored = mc_creative_krea.settings()

        list(mc_creative_krea.creative.roll(
            "car", stored, guard_checkpoint=True,
            spatial_layout=self.layout(spatial_module.SMART)))

        assert not asked

    def test_the_roll_does_reclaim_when_it_is_the_last_llm_phase(
            self, client, monkeypatch):
        asked = []
        monkeypatch.setattr(mc_creative_krea, "hand_back_vram",
                            lambda *a, **k: asked.append(True) or 0)
        stored = mc_creative_krea.settings()

        list(mc_creative_krea.creative.roll("car", stored, guard_checkpoint=True))

        assert asked


class TestTheReserveComesFromTheWholePlan:
    """The long-chain bug, at the point it was actually paid.

    ``image_reserve_bytes`` answered a Stage-1-shaped question, and a long
    chain is not Stage-1-shaped: Krea 2 into Klein 9B has its largest moment in
    the handoff, where Stage 2's weights and Stage 1's spared encoders are on
    the card together. A writer placed against Stage 1 alone is holding VRAM
    that phase needs, and pays for it with an eviction seconds later.
    """

    @pytest.fixture
    def chained(self, monkeypatch, host):
        import mc_plan

        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        monkeypatch.setattr(mc_broker, "held_bytes", lambda family: 0)
        monkeypatch.setattr(mc_plan, "usable_vram_bytes", lambda ours=0: 24 * 1024 ** 3)
        mc_plan.publish(mc_plan.Plan((
            mc_plan.Phase(mc_plan.STAGE_1, mc_plan.KIND_IMAGE, "Stage 1",
                          8 * 1024 ** 3),
            mc_plan.Phase(mc_plan.HANDOFF, mc_plan.KIND_TRANSITION, "Handoff",
                          17 * 1024 ** 3),
            mc_plan.Phase(mc_plan.STAGE_2, mc_plan.KIND_IMAGE, "Stage 2",
                          15 * 1024 ** 3),
        ), 1024, 1024))
        yield

    def test_the_reserve_is_the_largest_phase_not_the_first(self, chained):
        assert mc_creative_krea.image_reserve_bytes() == 17 * 1024 ** 3

    def test_it_falls_back_to_stage_1_when_no_plan_was_built(self, chained, monkeypatch):
        """An API request that never went through either script's hook still
        gets the behaviour it had before plans existed."""
        import mc_memory
        import mc_plan
        from modules import shared

        mc_plan.clear()
        monkeypatch.setattr(mc_memory, "vram_required_bytes",
                            lambda name, *a, **k: 8 * 1024 ** 3)
        monkeypatch.setattr(mc_broker, "resident_bytes", lambda family=None: 0)
        shared.opts.sd_model_checkpoint = "krea2.safetensors"

        assert mc_creative_krea.image_reserve_bytes() == 8 * 1024 ** 3
