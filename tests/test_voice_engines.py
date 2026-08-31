"""One TTS engine at a time, and what that has to mean everywhere else.

The design intent's first invariant is a product rule with a lot of machinery
under it: engine exclusivity, no cross-engine fallback, per-engine state that
cannot overwrite the other's, inheritance by absence, and a speech-to-text stack
that is not part of any of it. These are the tests for the module that owns that
rule -- T-ENG-1 through T-ENG-3, T-STT-1, T-PROTO-1 and T-MIG-1.

What is deliberately not here is anything about Sopro's runtime, its voices or
its DSP. Those are ``tests/test_voice_sopro.py`` and
``tests/test_voice_sopro_worker.py``. This file is about the boundary, and it
passes on an installation where Sopro was never installed at all.
"""

from __future__ import annotations

import pytest

import mc_voice_engines as engines


class Character:
    """Whatever the chat panel happens to hand over. Read defensively."""

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)


class TestTheSelector:
    def test_a_fresh_installation_is_kokoro(self, host):
        """Section 12: no stored engine means Kokoro, so an upgrade does not
        change anybody's voice."""
        assert engines.active() == engines.KOKORO

    def test_a_value_nobody_recognises_is_kokoro(self, host):
        """Read on the path that decides whether a reply is spoken. A
        hand-edited config.json is not a reason for silence."""
        host.shared.opts.set(engines.OPT_ENGINE, "festival")
        assert engines.active() == engines.KOKORO

    def test_selecting_persists_and_reports_the_new_state(self, host):
        found = engines.select("sopro")
        assert found["active"] == "sopro"
        assert engines.active() == "sopro"
        assert host.shared.opts.__dict__[engines.OPT_ENGINE] == "sopro"

    def test_selecting_what_is_already_selected_changes_nothing(self, host, monkeypatch):
        stopped = []
        monkeypatch.setattr(engines, "_stop_all", lambda: stopped.append(1))
        engines.select("kokoro")
        assert stopped == []

    def test_an_unknown_engine_is_refused(self, host):
        with pytest.raises(engines.EngineError):
            engines.select("festival")

    @pytest.mark.parametrize("wanted", ("sopro", "pocket"))
    def test_t_eng_3_switching_stops_every_tts_worker(self, host, monkeypatch, wanted):
        """T-ENG-P4. Exactly one TTS runtime is active, and the way that is
        guaranteed is that *every* one is stopped before the new id is
        persisted -- rather than "the one that was running", because asking
        which one it was is one more thing that can be wrong.

        Driven off the registry rather than a written-down list, so a fourth
        engine is stopped by having been registered (I-PKT-30).
        """
        stopped = []
        for spec in engines.SPECS:
            module = engines._module(spec.runtime)
            monkeypatch.setattr(module, "stop",
                                lambda reason="", name=spec.id: stopped.append(name))
        engines.select(wanted)
        assert sorted(stopped) == sorted(engines.ENGINES)

    def test_switching_cancels_the_speaking_turn(self, host, monkeypatch):
        """A reply half spoken in Kokoro must not finish in Sopro."""
        import mc_voice_turn as turns

        cancelled = []
        monkeypatch.setattr(turns, "forget_all", lambda reason: cancelled.append(reason))
        engines.select("sopro")
        assert cancelled and "Sopro" in cancelled[0]

    def test_switching_starts_no_download_and_loads_no_model(self, host, monkeypatch):
        """Section 4, steps 5 and 6. Selecting an uninstalled engine is allowed
        and is what makes its install surface reachable at all."""
        import mc_voice_models as models
        import mc_voice_sopro as sopro

        monkeypatch.setattr(models, "install", lambda *a, **k: pytest.fail("downloaded"))
        monkeypatch.setattr(sopro, "install", lambda *a, **k: pytest.fail("downloaded"))
        engines.select("sopro")
        assert engines.active() == "sopro"

    def test_selecting_an_uninstalled_engine_does_not_switch_back(self, host, monkeypatch):
        import mc_voice_sopro as sopro

        monkeypatch.setattr(sopro, "status",
                            lambda: sopro.Status(runtime_message="Not installed"))
        engines.select("sopro")
        assert engines.active() == "sopro"
        assert engines.installed("sopro") is False


class TestSpeechToTextIsNotInThisSelector:
    def test_t_stt_1_switching_does_not_touch_whisper(self, host, monkeypatch):
        """I-7, and the reason it is a hard architecture requirement rather than
        an implementation option: dictation has its own model, its own process
        and its own quality tier, and none of them is a property of which voice
        reads replies aloud."""
        import mc_voice_models as models

        touched = []
        monkeypatch.setattr(models, "select",
                            lambda kind, identifier: touched.append((kind, identifier)))
        monkeypatch.setattr(models, "install", lambda *a, **k: touched.append("install"))
        before = models.default_id("stt")
        for _ in range(3):
            engines.select("sopro")
            engines.select("kokoro")
        assert touched == []
        assert models.default_id("stt") == before

    def test_the_stt_option_is_not_one_of_this_modules(self, host):
        import mc_voice_models as models

        assert engines.OPT_ENGINE not in set(models.OPTIONS.values())


class TestVoiceIdentity:
    def test_t_proto_1_an_id_says_which_engine_owns_it(self):
        assert engines.qualify("official:af_heart", "kokoro") == "kokoro:official:af_heart"
        assert engines.qualify("clone:abc", "sopro") == "sopro:clone:abc"
        assert engines.engine_of("sopro:clone:abc") == "sopro"
        assert engines.native("kokoro:official:af_heart") == "official:af_heart"

    def test_an_already_qualified_id_is_not_re_badged(self):
        """A spelling function, not a policy one. Silently re-badging a Kokoro
        id as Sopro because Sopro happens to be selected is exactly the
        cross-engine confusion I-2 forbids."""
        assert engines.qualify("kokoro:official:af_heart", "sopro") \
            == "kokoro:official:af_heart"

    def test_a_legacy_id_is_kokoros(self):
        """Section 12: migration is by reading. A character file written before
        Sopro existed resolves today without being rewritten."""
        assert engines.engine_of("official:af_heart") == "kokoro"
        assert engines.engine_of("clone:deadbeef") == "kokoro"
        assert engines.engine_of("") == "kokoro"

    def test_an_empty_id_stays_empty(self):
        assert engines.qualify("") == ""


class TestCharacterState:
    def test_t_mig_1_a_legacy_character_is_a_kokoro_character(self):
        character = Character(voice="official:af_nicole",
                              voice_profile={"speed": 1.2, "pitch": None,
                                             "gain": None, "pause": None})
        assert engines.character_voice(character, "kokoro") == "kokoro:official:af_nicole"
        assert engines.character_profile(character, "kokoro")["speed"] == 1.2

    def test_i_4_a_legacy_character_has_no_sopro_state_and_does_not_borrow_one(self):
        """Inheritance by absence, and no translation. A Kokoro speaker number
        means nothing to Sopro, and answering with it would be worse than
        answering with nothing."""
        character = Character(voice="official:af_nicole", voice_profile={"speed": 1.2})
        assert engines.character_voice(character, "sopro") == ""
        assert engines.character_profile(character, "sopro") == {}

    def test_the_engine_aware_shape_is_read_when_it_is_there(self):
        character = Character(voices={"kokoro": "official:af_heart",
                                      "sopro": "sopro:clone:abc"},
                              voice_profiles={"sopro": {"speed": 0.95}})
        assert engines.character_voice(character, "sopro") == "sopro:clone:abc"
        assert engines.character_voice(character, "kokoro") == "kokoro:official:af_heart"
        assert engines.character_profile(character, "sopro") == {"speed": 0.95}
        assert engines.character_profile(character, "kokoro") == {}

    def test_the_two_engines_read_from_separate_fields(self):
        """I-3 from the reading side. The writing side is
        ``Character.voice_fields`` and ``tests/test_llm_panels.py`` drives it
        through the editor, which is the path a user actually takes -- there is
        deliberately only one function in this repository that knows how to
        write a character's voice fields."""
        from prompt_master.chat.characters import Character

        found = Character(name="Ada", voice="kokoro:official:af_heart", voice_speed=1.1,
                          sopro_voice="sopro:clone:abc", sopro_speed=0.95)
        assert engines.character_voice(found, "kokoro") == "kokoro:official:af_heart"
        assert engines.character_voice(found, "sopro") == "sopro:clone:abc"
        assert engines.character_profile(found, "kokoro")["speed"] == 1.1
        assert engines.character_profile(found, "sopro")["speed"] == 0.95

    def test_an_engine_with_nothing_set_is_absent_rather_than_empty(self):
        """I-4's named failure: today's defaults must never be materialised in
        order to represent inheritance."""
        from prompt_master.chat.characters import Character

        found = Character(name="Ada", voice="kokoro:official:af_heart")
        assert "sopro" not in found.voices
        assert "sopro" not in found.voice_profiles
        assert engines.character_profile(found, "sopro") == {}


class TestScoping:
    def test_t_eng_1_the_inactive_engine_is_absent_from_a_payload(self, host):
        """Section 5: absent, not collapsed. A stale DOM cannot expose what was
        never sent."""
        payload = {"ready": True, "kokoro": {"installed": True},
                   "sopro": {"installed": True, "voices": ["secret"]}}
        found = engines.scope(payload, "kokoro")
        assert "sopro" not in found
        assert found["kokoro"] == {"installed": True}
        assert found["engine"] == "kokoro"

    def test_a_request_naming_the_other_engine_is_refused(self, host):
        engines.select("sopro")
        with pytest.raises(engines.ActiveEngineMismatch):
            engines.refuse_mismatch("kokoro")

    def test_a_request_naming_no_engine_means_the_active_one(self, host):
        engines.select("sopro")
        assert engines.refuse_mismatch("") == "sopro"

    def test_a_request_naming_an_unknown_engine_is_refused(self, host):
        with pytest.raises(engines.EngineError):
            engines.refuse_mismatch("festival")


class TestNoCrossEngineFallback:
    def test_i_2_a_voice_from_the_other_engine_is_not_resolved(self, host, voice_registry,
                                                              kokoro_bundle):
        """T-ENG-3. A Sopro id while Kokoro is selected resolves to the *Kokoro*
        default, not to Sopro -- and the caller is told the character's voice is
        missing rather than being handed the other engine's."""
        found, entry = engines.resolve("sopro:clone:whatever", "kokoro")
        assert found.startswith("kokoro:")
        assert entry["engine"] == "kokoro"

    def test_belongs_is_the_check_every_mutation_uses(self):
        assert engines.belongs("sopro:clone:abc", "sopro")
        assert not engines.belongs("sopro:clone:abc", "kokoro")
        assert engines.belongs("official:af_heart", "kokoro")


class TestTheFacade:
    def test_t_pkt_eng_10_each_engine_gets_its_own_modules(self, host):
        import mc_voice_kokoro
        import mc_voice_pocket
        import mc_voice_pocket_profile
        import mc_voice_pocket_runtime
        import mc_voice_profile
        import mc_voice_runtime
        import mc_voice_sopro
        import mc_voice_sopro_profile
        import mc_voice_sopro_runtime

        assert engines.adapter("kokoro") is mc_voice_kokoro
        assert engines.adapter("sopro") is mc_voice_sopro
        assert engines.adapter("pocket") is mc_voice_pocket
        assert engines.runtime("kokoro") is mc_voice_runtime
        assert engines.runtime("sopro") is mc_voice_sopro_runtime
        assert engines.runtime("pocket") is mc_voice_pocket_runtime
        assert engines.profiles("kokoro") is mc_voice_profile
        assert engines.profiles("sopro") is mc_voice_sopro_profile
        assert engines.profiles("pocket") is mc_voice_pocket_profile

    def test_every_engine_is_a_row_rather_than_a_branch(self):
        """I-PKT-30. A fourth engine registers a spec and an adapter; it does
        not require finding every shared fallthrough first."""
        assert engines.ENGINES == tuple(spec.id for spec in engines.SPECS)
        assert set(engines.LABELS) == set(engines.ENGINES)
        assert set(engines.BLURBS) == set(engines.ENGINES)
        for spec in engines.SPECS:
            assert spec.adapter and spec.runtime and spec.profiles

    def test_no_two_profile_modules_share_a_single_option(self):
        """Section 35, I-PKT-3: common labels, separate storage. Speed means a
        Kokoro generate argument on one side and a time-scale on the other two,
        and one option holding two of them would be a setting that changed when
        it was edited somewhere else."""
        seen = {}
        for engine in engines.ENGINES:
            for name in engines.profiles(engine).OPTIONS.values():
                assert name not in seen, f"{name} is shared by {seen.get(name)} and {engine}"
                seen[name] = engine

    def test_t_pkt_eng_11_every_engine_declares_what_stop_means(self, host):
        """The one capability shared code reads before it promises anything."""
        found = {engine: engines.interrupt_mode(engine) for engine in engines.ENGINES}
        assert found == {"kokoro": "cancel", "sopro": "cancel", "pocket": "drain_unit"}
        for engine in engines.ENGINES:
            assert set(engines.capabilities(engine)) == {
                "clone_preview", "rebuild", "engine_settings", "starter_voices",
                "voice_lab", "interrupt_mode"}

    def test_an_engine_that_will_not_import_answers_conservatively(self, host,
                                                                   monkeypatch):
        """An uninstalled engine is the ordinary case, and a caller that
        believed it drains would draw a waiting state nothing ever clears."""

        def missing(name):
            raise ImportError(name)

        monkeypatch.setattr(engines, "_module", missing)
        assert engines.installed("pocket") is False
        assert engines.capabilities("pocket")["interrupt_mode"] == "cancel"
        assert engines.refusals("pocket") == (engines.EngineError,)

    def test_a_refusal_type_comes_from_the_engine_rather_than_a_fallthrough(self, host):
        """"Not Sopro means the Kokoro registry" was true while there were two
        engines and became wrong the moment there were three."""
        import mc_voice_pocket as pocket
        import mc_voice_registry as registry
        import mc_voice_sopro as sopro

        assert registry.RegistryError in engines.refusals("kokoro")
        assert sopro.SoproError in engines.refusals("sopro")
        assert pocket.PocketError in engines.refusals("pocket")
        assert sopro.SoproError not in engines.refusals("pocket")


class TestTheKokoroAdapter:
    def test_it_speaks_qualified_ids_and_publishes_no_speaker_number(
            self, voice_registry, kokoro_bundle):
        import mc_voice_kokoro as kokoro

        found = kokoro.entries()
        assert all(entry["id"].startswith("kokoro:") for entry in found)
        assert all("sid" not in entry for entry in found)

    def test_resolve_carries_the_sid_privately(self, voice_registry, kokoro_bundle):
        """Section 18: the sherpa speaker number exists inside the adapter and
        nowhere else. It rides under a name the public payload builder does not
        copy."""
        import mc_voice_kokoro as kokoro

        found, entry = kokoro.resolve("kokoro:official:af_heart")
        assert found == "kokoro:official:af_heart"
        assert entry["_sid"] == 3

    def test_an_id_from_the_other_engine_is_not_this_adapters(self, voice_registry,
                                                             kokoro_bundle):
        import mc_voice_kokoro as kokoro

        assert kokoro.lookup("sopro:clone:abc") is None
        with pytest.raises(kokoro.KokoroError):
            kokoro.set_default("sopro:clone:abc")
