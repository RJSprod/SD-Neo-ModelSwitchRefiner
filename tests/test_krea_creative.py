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
from pathlib import Path

import pytest

import mc_broker
import mc_creative_krea
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
    monkeypatch.setattr(sessions, "_client", lambda needs_vision=False, reserve=0: fake)
    monkeypatch.setattr(sessions, "_placement_notes", list)
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
        stored = mc_creative_krea.settings()
        stored["creativity"] = 10
        list(mc_creative_krea.creative.roll("car", stored))

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
        """A 24 GB card, an 8 GB checkpoint, and nothing resident."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "vram_required_bytes",
                            lambda name, *a, **k: 8 * 1024 ** 3)
        monkeypatch.setattr(mc_broker, "resident_bytes", lambda family=None: 0)
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

    def test_a_fully_resident_checkpoint_needs_nothing_reserved(self, card, monkeypatch):
        monkeypatch.setattr(mc_broker, "resident_bytes", lambda family=None: 20 * 1024 ** 3)

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
                            lambda needs_vision=False, reserve=0: (
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
                            lambda needs_vision=False, reserve=0: (
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

    def test_the_browser_gate_has_no_timer_and_no_repeat(self):
        source = (Path(__file__).resolve().parent.parent / "javascript"
                  / "model_chain_creative_krea.js").read_text(encoding="utf-8")

        assert "setTimeout" not in source
        assert '"input"' not in source
        assert "reroll" not in source.casefold()
        # One interval only: the poll that waits for the server's answer.
        assert source.count("setInterval") == 1

    def test_the_live_modules_are_gone(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("mc_live_krea.py", "scripts/model_chain_krea_live.py",
                     "javascript/model_chain_live_krea.js"):
            assert not (root / name).exists()

    def test_a_roll_only_happens_when_something_calls_for_one(self, client):
        """Nothing observes anything, so there is no path from "text changed" to
        "a model was asked". Merely having a session is not a roll."""
        session = mc_creative_krea.Creative()

        assert session.armed is None
        assert session.last is None
        assert client.calls == []


# --------------------------------------------------------------------------- #
# Arming, and the hook that spends it
# --------------------------------------------------------------------------- #


class Processing:
    """The half of a StableDiffusionProcessing the Creative hook touches."""

    def __init__(self, prompt="car"):
        self.prompt = prompt
        self.extra_generation_params = {}


@pytest.fixture
def script():
    import model_chain_krea_creative as creative_script

    return creative_script.ScriptKreaCreative()


def armed(client, source="car", loras="", creativity=10):
    stored = mc_creative_krea.settings()
    stored["creativity"] = creativity
    stored["loras"] = loras
    list(mc_creative_krea.creative.roll(source, stored))
    return mc_creative_krea.creative.arm(mc_creative_krea.creative.last, loras)


class TestTheProcessingHook:
    def test_an_unarmed_generation_is_left_exactly_as_it_was(self, script, client):
        p = Processing("car")
        script.before_process(p, True)

        assert p.prompt == "car"
        assert p.extra_generation_params == {}

    def test_creative_mode_off_leaves_the_prompt_alone(self, script, client):
        armed(client)
        p = Processing("car")
        script.before_process(p, False)

        assert p.prompt == "car"

    def test_an_armed_generation_is_given_the_expansion(self, script, client):
        client.answers = ["A low-angle impasto painting of a car."]
        armed(client)
        p = Processing("car")
        script.before_process(p, True)

        assert p.prompt == "A low-angle impasto painting of a car."

    def test_the_pinned_loras_arrive_with_it(self, script, client):
        client.answers = ["A painted car."]
        armed(client, loras="<lora:film:0.8>")
        p = Processing()
        script.before_process(p, True)

        assert p.prompt == "A painted car. <lora:film:0.8>"

    def test_everything_needed_to_roll_it_again_is_recorded(self, script, client):
        armed(client, source="car", loras="<lora:film:0.8>")
        p = Processing()
        script.before_process(p, True)
        recorded = p.extra_generation_params

        assert recorded[mc_infotext.CREATIVE_MODE] == "True"
        assert recorded[mc_infotext.CREATIVE_CREATIVITY] == 10
        assert recorded[mc_infotext.CREATIVE_SOURCE] == "car"
        assert recorded[mc_infotext.CREATIVE_SEED] != director.RANDOM_SEED
        assert recorded[mc_infotext.CREATIVE_LLM_SEED] == director.derive(
            recorded[mc_infotext.CREATIVE_SEED])[1]
        assert "medium=" in recorded[mc_infotext.CREATIVE_RECIPE]
        assert recorded[mc_infotext.CREATIVE_LORAS] == "<lora:film:0.8>"

    def test_the_expanded_paragraph_is_not_recorded_twice(self, script, client):
        """It is already the generation's own prompt line."""
        client.answers = ["A tall white lighthouse under storm light."]
        armed(client)
        p = Processing()
        script.before_process(p, True)

        assert "A tall white lighthouse" not in "".join(
            str(value) for value in p.extra_generation_params.values())

    def test_one_roll_cannot_make_two_images(self, script, client):
        armed(client)
        first, second = Processing("car"), Processing("car")
        script.before_process(first, True)
        script.before_process(second, True)

        assert first.prompt != "car"
        assert second.prompt == "car"

    def test_the_hook_never_asks_for_an_expansion_of_its_own(self, script, client):
        """It applies; it does not request. An LLM run waits for the host to stop
        generating, so a hook that asked for one would deadlock against the job
        it was running inside."""
        script.before_process(Processing("nothing was armed for this"), True)

        assert client.calls == []

    def test_turning_creative_mode_off_throws_the_permission_away(self, script, client):
        armed(client)
        mc_creative_krea.creative.disarm()
        p = Processing("car")
        script.before_process(p, True)

        assert p.prompt == "car"


class TestTheArmingToken:
    def test_a_roll_arms_exactly_one_generation(self, client):
        armed(client)

        assert mc_creative_krea.creative.consume() is not None
        assert mc_creative_krea.creative.consume() is None

    def test_a_wrong_token_is_refused(self, client):
        armed(client)

        assert mc_creative_krea.creative.consume("not-the-token") is None

    def test_re_arming_replaces_rather_than_stacks(self, client):
        first = armed(client).token
        second = armed(client).token

        assert first != second
        assert mc_creative_krea.creative.consume().token == second
        assert mc_creative_krea.creative.consume() is None


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
# The pinned LoRAs
# --------------------------------------------------------------------------- #


class TestPinnedLoras:
    def test_the_language_model_is_never_shown_a_tag(self, client):
        armed(client, loras="<lora:film:0.8>")

        assert "lora" not in client.turn

    def test_prose_typed_into_the_field_is_not_smuggled_into_the_prompt(self):
        assert mc_creative_krea.lora_suffix(
            "a beautiful sunset <lora:film:0.8> masterpiece") == "<lora:film:0.8>"
        assert mc_creative_krea.pinned_tags("nothing but words") == []

    def test_they_are_appended_to_the_generation_prompt_only(self, client):
        client.answers = ["A painted car."]
        record = armed(client, loras="<lora:film:0.8>")

        assert record.generation == "A painted car. <lora:film:0.8>"
        assert record.roll.expanded == "A painted car."


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

    def test_the_default_surface_is_a_toggle_and_a_slider(self, built):
        """Simple by default: everything else is behind the accordion, and the
        accordion is closed."""
        assert built.components["enabled"].value is False
        assert built.components["controls"].visible is False
        assert built.components["controls"].open is False
        assert built.components["creativity"].visible is False

    def test_the_slider_offers_the_whole_scale(self, built):
        slider = built.components["creativity"]

        assert (slider.minimum, slider.maximum, slider.step) == (0, 10, 1)

    def test_there_is_a_row_for_every_axis_in_the_library(self, built, lib):
        assert len(built.components["axes"]) == len(lib.axis_keys) * 2

    def test_every_axis_offers_the_three_modes(self, built):
        for control in built.components["axes"][::2]:
            assert [value for _label, value in control.choices] == list(director.MODES)

    def test_a_fixed_dropdown_lists_that_axis_own_variants(self, built, lib):
        values = built.components["axes"][1]
        expected = [variant.identifier for variant in lib.axis(lib.axis_keys[0]).variants]

        assert [value for _label, value in values.choices] == expected

    def test_no_handler_can_write_to_the_positive_prompt(self, built):
        """The visible box keeps the short phrase the user is iterating on."""
        built._prompt_component = _Native("txt2img_prompt")
        built.ui(False)

        for kwargs in _handlers(built):
            assert built._prompt_component not in list(kwargs.get("outputs") or [])

    def test_the_gate_reads_the_positive_prompt_as_its_source(self, built):
        built._prompt_component = _Native("txt2img_prompt")
        built.ui(False)

        reading = [kwargs for kwargs in _handlers(built)
                   if built._prompt_component in list(kwargs.get("inputs") or [])]
        assert reading

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

    def test_it_has_a_row_for_every_axis_too(self, built, lib):
        assert len(built["axes"]) == len(lib.axis_keys) * 2

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
