"""The Voice Lab, and the one thing it must never be able to do.

T-LAB-1, T-LAB-2, T-LAB-3 and Gate S-7. The Lab exists because Sopro's
conditioning is more interesting than Kokoro's speaker bank and nobody knows
what its eight ``style_ctrl`` dimensions mean -- and the whole design of it is
built around making that investigation unable to become a product claim by
accident.

So most of these tests are absence tests. They move every control the Lab has,
run auditions, and then assert that a character, a default, a saved voice asset
and a Conversation turn are all exactly what they were. An experiment that
cannot be saved is the feature; the sliders are the interface to it.
"""

from __future__ import annotations

import hashlib

import pytest

import mc_voice_engines as engines
import mc_voice_lab as lab
import mc_voice_paths as paths
import mc_voice_sopro as sopro


@pytest.fixture
def a_voice(host, voice_root, fake_sopro_worker, spoken_wav):
    """One registered Sopro voice, made through the ordinary path."""
    made = sopro.create("Rebecca", spoken_wav(9.0, 24000), "")
    return made["voice"]


@pytest.fixture(autouse=True)
def _forget_lab_sessions():
    yield
    lab.forget_all("test finished")


def _hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestWhereTheLabExists:
    def test_it_is_part_of_sopro_and_refuses_while_kokoro_is_selected(self, host,
                                                                     voice_root):
        """Section 38. It does not appear in the Conversation overlay, the
        character editor, the ordinary voice picker or any Kokoro surface, and
        the server says so rather than relying on the page not to draw it."""
        engines.select("kokoro")
        with pytest.raises(lab.LabError):
            lab.open_session()
        with pytest.raises(lab.LabError):
            lab.panel()

    def test_it_needs_a_voice_before_it_can_open(self, host, voice_root,
                                                fake_sopro_worker):
        with pytest.raises(lab.LabError):
            lab.open_session()

    def test_the_notice_and_the_caution_are_part_of_the_surface(self, host, voice_root,
                                                               fake_sopro_worker, a_voice):
        """Section 44 requires the Lab to make it difficult to mistake a result
        for a character's saved voice, and section 41 forbids naming the eight
        controls after emotions. Both sentences live in one place so that every
        surface drawing the Lab shows the same one."""
        found = lab.panel()
        assert "not used in Conversation" in found["notice"]
        assert "not saved to characters" in found["notice"]
        assert "not emotions" in found["caution"]
        assert "not proven identity or style transfer" in found["caution"]

    def test_the_sliders_are_numbered_rather_than_named(self, host, voice_root,
                                                        fake_sopro_worker, a_voice):
        import mc_voice_ui

        markup = mc_voice_ui._lab_html()
        assert "Style control 1" in markup and "Style control 8" in markup
        for invented in ("Emotion", "Warmth", "Energy", "Breathiness", "Excitement"):
            assert invented not in markup


class TestSessions:
    def test_a_session_starts_neutral_on_the_default_voice(self, host, voice_root,
                                                           fake_sopro_worker, a_voice):
        found = lab.open_session()
        assert found["voice_id"] == a_voice["id"]
        assert found["neutral"] is True
        assert found["deltas"] == [0.0] * lab.STYLE_CONTROLS

    def test_the_base_style_comes_from_the_saved_voice(self, host, voice_root,
                                                       fake_sopro_worker, a_voice):
        """The sliders sit at zero against whatever the voice was prepared with,
        so Reset is exactly "no offset" rather than "some remembered vector"."""
        found = lab.open_session()
        assert len(found["base_style"]) == lab.STYLE_CONTROLS

    def test_switching_engines_destroys_the_session(self, host, voice_root,
                                                    fake_sopro_worker, a_voice):
        """Section 39. Not because the state is dangerous where it is, but
        because a session that survived would be one pointing at a voice library
        that is no longer the active engine's."""
        found = lab.open_session()
        engines.select("kokoro")
        with pytest.raises(lab.LabError):
            lab.session(found["token"])

    def test_an_expired_token_says_so_rather_than_opening_a_new_one(self, host, voice_root,
                                                                    fake_sopro_worker,
                                                                    a_voice):
        with pytest.raises(lab.LabError):
            lab.session("nothing-like-a-token")

    def test_only_a_few_sessions_are_kept(self, host, voice_root, fake_sopro_worker,
                                          a_voice):
        """Bounded, because a Lab session is a token and a handful of floats and
        an unbounded map of them is a settings page that leaks a little every
        time somebody opens it."""
        tokens = [lab.open_session()["token"] for _ in range(lab.MAX_SESSIONS + 3)]
        alive = 0
        for token in tokens:
            try:
                lab.session(token)
                alive += 1
            except lab.LabError:
                pass
        assert alive <= lab.MAX_SESSIONS


class TestTheControls:
    def test_a_delta_is_clamped_to_the_tested_range(self, host, voice_root,
                                                    fake_sopro_worker, a_voice):
        token = lab.open_session()["token"]
        found = lab.update(token, deltas=[99.0, -99.0] + [0.0] * 6)
        assert found["deltas"][0] == lab.DELTA_LIMIT
        assert found["deltas"][1] == -lab.DELTA_LIMIT

    def test_reset_all_returns_every_experimental_value_to_nothing(self, host, voice_root,
                                                                   fake_sopro_worker,
                                                                   a_voice):
        """Mandatory (section 44), and it is one call rather than eight."""
        token = lab.open_session()["token"]
        lab.update(token, deltas=[1.0] * 8, seed=99, temperature=1.2, top_p=0.5,
                   top_k=10, steps=4)
        assert lab.session(token).neutral is False
        found = lab.reset(token)
        assert found["neutral"] is True
        assert found["deltas"] == [0.0] * lab.STYLE_CONTROLS
        assert found["seed"] is None and found["temperature"] is None

    def test_reset_keeps_the_voice_and_the_text_it_is_being_tested_with(self, host,
                                                                       voice_root,
                                                                       fake_sopro_worker,
                                                                       a_voice):
        token = lab.open_session()["token"]
        lab.update(token, text="the quick brown fox", deltas=[1.0] * 8)
        found = lab.reset(token)
        assert found["text"] == "the quick brown fox"
        assert found["voice_id"] == a_voice["id"]

    def test_changing_the_voice_starts_a_new_experiment(self, host, voice_root,
                                                        fake_sopro_worker, spoken_wav,
                                                        a_voice):
        """One voice's offsets applied to another's base vector is the one
        comparison in this surface that means nothing at all."""
        other = sopro.create("Grace", spoken_wav(9.0, 24000), "")["voice"]
        token = lab.open_session()["token"]
        lab.update(token, deltas=[1.0] * 8)
        found = lab.update(token, voice_id=other["id"])
        assert found["voice_id"] == other["id"]
        assert found["deltas"] == [0.0] * lab.STYLE_CONTROLS

    def test_an_unknown_field_is_ignored_rather_than_forwarded(self, host, voice_root,
                                                              fake_sopro_worker, a_voice):
        """A page from a later build cannot set a field this one would then pass
        to the worker unvalidated."""
        token = lab.open_session()["token"]
        found = lab.update(token, deltas=[0.0] * 8, weights=[9, 9], model="something")
        assert "weights" not in found and "model" not in found


class TestConditioningBlend:
    def test_a_blend_names_the_components_it_substitutes(self, host, voice_root,
                                                         fake_sopro_worker, spoken_wav,
                                                         a_voice):
        other = sopro.create("Grace", spoken_wav(9.0, 24000), "")["voice"]
        token = lab.open_session()["token"]
        found = lab.update(token, blend={"voice_id": other["id"], "weight": 0.5,
                                         "style_emb": True})
        assert found["blend"]["voice_id"] == other["id"]
        assert found["blend"]["weight"] == 0.5
        assert found["blend"]["style_emb"] is True

    def test_it_can_only_substitute_pre_projection_speaker_components(self):
        """Section 42: the semantic reference tokens and the reference mel are
        never touched, which is exactly why this is called Conditioning Blend
        and not identity or style transfer."""
        assert lab.BLEND_FIELDS == ("id_emb", "style_emb", "style_ctrl")
        assert "semantic_tokens" not in lab.BLEND_FIELDS
        assert "mel" not in lab.BLEND_FIELDS

    def test_a_blend_with_itself_is_dropped(self, host, voice_root, fake_sopro_worker,
                                            a_voice):
        token = lab.open_session()["token"]
        found = lab.update(token, blend={"voice_id": a_voice["id"], "weight": 1.0,
                                         "id_emb": True})
        assert found["blend"] == {}

    def test_a_blend_with_no_components_or_no_weight_is_dropped(self, host, voice_root,
                                                                fake_sopro_worker,
                                                                spoken_wav, a_voice):
        other = sopro.create("Grace", spoken_wav(9.0, 24000), "")["voice"]
        token = lab.open_session()["token"]
        assert lab.update(token, blend={"voice_id": other["id"],
                                        "weight": 0.5})["blend"] == {}
        assert lab.update(token, blend={"voice_id": other["id"], "weight": 0.0,
                                        "id_emb": True})["blend"] == {}

    def test_a_blend_naming_a_voice_that_does_not_exist_is_refused(self, host, voice_root,
                                                                   fake_sopro_worker,
                                                                   a_voice):
        token = lab.open_session()["token"]
        with pytest.raises(lab.LabError):
            lab.update(token, blend={"voice_id": "sopro:clone:nothing", "weight": 0.5,
                                     "id_emb": True})


class TestGateS7Isolation:
    def test_t_lab_1_a_non_neutral_lab_does_not_reach_conversation(self, host, voice_root,
                                                                    fake_sopro_worker,
                                                                    a_voice):
        """Move everything, run an audition, then resolve a Conversation turn's
        voice and profile: they are the saved production ones and nothing else."""
        import mc_voice_sopro_profile as profiles

        token = lab.open_session()["token"]
        lab.update(token, deltas=[1.2] * 8, seed=4242, temperature=1.4, top_p=0.4,
                   top_k=3, steps=4)
        lab.audition(token, "b")

        voice_id, entry = engines.resolve("", "sopro")
        assert voice_id == a_voice["id"]
        found = profiles.resolve(None)
        assert found["temperature"] is None, "a Lab temperature reached the default profile"
        assert found["top_p"] is None and found["top_k"] is None
        assert found["speed"] == 1.0 and found["pitch"] == 0.0

    def test_t_lab_2_saving_the_default_while_the_lab_is_loud_saves_nothing_of_it(
            self, host, voice_root, fake_sopro_worker, a_voice):
        """Section 39: no Lab delta, blend or seed may be serialized into
        production state by an ordinary save."""
        import mc_voice_sopro_profile as profiles

        token = lab.open_session()["token"]
        lab.update(token, deltas=[1.5] * 8, seed=7, temperature=1.5)
        profiles.remember({"speed": 1.1})
        found = profiles.stored()
        assert found["speed"] == 1.1
        assert found["temperature"] is None
        stored = host.shared.opts.__dict__
        assert not any("delta" in str(key) or "blend" in str(key) or "seed" in str(key)
                       for key in stored if str(key).startswith("model_chain_voice_sopro"))

    def test_t_lab_3_the_production_asset_is_byte_identical_after_lab_use(
            self, host, voice_root, fake_sopro_worker, a_voice):
        """The physical half of the separation. Hash it, use the Lab hard, hash
        it again."""
        identifier = a_voice["id"].split(":")[-1]
        production = paths.sopro_voice_file(identifier, paths.SOPRO_PRODUCTION_FILENAME)
        before = _hash(production)

        token = lab.open_session()["token"]
        lab.update(token, deltas=[1.5, -1.5, 1.0, -1.0, 0.5, -0.5, 1.2, -1.2], seed=11)
        lab.audition(token, "b")
        lab.audition(token, "a")

        assert _hash(production) == before

    def test_the_lab_asset_can_go_without_touching_production(self, host, voice_root,
                                                              fake_sopro_worker, a_voice):
        """Section 14 and section 55: a Lab conditioning asset may be rebuilt or
        deleted independently, and failing to rebuild one must not invalidate a
        still-compatible production voice."""
        identifier = a_voice["id"].split(":")[-1]
        production = paths.sopro_voice_file(identifier, paths.SOPRO_PRODUCTION_FILENAME)
        before = _hash(production)
        paths.sopro_voice_file(identifier, paths.SOPRO_LAB_FILENAME).unlink()

        found = sopro.lookup(a_voice["id"])
        assert found["compatible"] is True
        assert found["has_lab"] is False
        assert _hash(production) == before
        assert a_voice["id"] in sopro.catalog()

    def test_there_is_no_promotion_path_at_all(self):
        """There is no "Apply to Conversation" and no "Promote" in this release.
        If a latent control proves meaningful, promoting it becomes a later
        design change with named semantics, bounds, migration and tests -- not a
        button somebody adds."""
        for forbidden in ("save", "apply", "promote", "commit", "to_profile",
                          "write_default", "assign"):
            assert not hasattr(lab, forbidden), (
                f"mc_voice_lab grew a {forbidden!r}, which is the promotion path "
                f"section 42 says has to be a new design")

    def test_a_lab_session_is_not_a_type_production_accepts(self, host, voice_root,
                                                           fake_sopro_worker, a_voice):
        """Section 39 asks for production code to be unable to read Lab state
        *because the types are separate*, rather than because a caller promised
        not to."""
        import mc_voice_sopro_profile as profiles

        found = lab.session(lab.open_session()["token"])
        assert isinstance(found, lab.Session)
        assert not isinstance(found, dict)
        # ``clamp`` takes a mapping; a Session is not one, and the profile module
        # has no branch that would make it into one.
        with pytest.raises((TypeError, ValueError, AttributeError)):
            profiles.clamp(found)


class TestAuditions:
    def test_playing_b_returns_audio_and_the_runs_own_numbers(self, host, voice_root,
                                                              fake_sopro_worker, a_voice):
        token = lab.open_session()["token"]
        found = lab.audition(token, "b")
        assert found["audio"][:4] == b"RIFF"
        assert found["state"]["last"]["side"] == "b"
        assert found["state"]["last"]["first_audio_ms"] >= 0

    def test_playing_a_goes_through_the_ordinary_production_path(self, host, voice_root,
                                                                 fake_sopro_worker,
                                                                 a_voice, monkeypatch):
        """If A went through the Lab with zero deltas it would prove that the
        Lab at neutral sounds like the Lab at neutral, which is not the
        comparison anybody wants."""
        import mc_voice_sopro_runtime as runtime

        used = []
        original_synthesize = runtime.synthesize
        original_lab = runtime.lab_audition
        monkeypatch.setattr(runtime, "synthesize",
                            lambda *a, **k: (used.append("production"),
                                             original_synthesize(*a, **k))[1])
        monkeypatch.setattr(runtime, "lab_audition",
                            lambda *a, **k: (used.append("lab"),
                                             original_lab(*a, **k))[1])
        token = lab.open_session()["token"]
        lab.audition(token, "a")
        assert used == ["production"]
        lab.audition(token, "b")
        assert used == ["production", "lab"]

    def test_an_audition_without_a_voice_is_refused_rather_than_silent(self, host,
                                                                      voice_root,
                                                                      fake_sopro_worker,
                                                                      a_voice):
        found = lab.session(lab.open_session()["token"])
        found.voice_id = ""
        with pytest.raises(lab.LabError):
            lab.audition(found.token, "b")
