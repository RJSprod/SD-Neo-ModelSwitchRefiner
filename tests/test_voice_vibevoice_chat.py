"""VibeVoice as a Voice Chat engine: its voices, its default, its settings, and cloning.

The adapter is the product side of the fourth engine, and most of what is worth
checking about it is a relationship with something it does not own: its presets
are the installer's, its samples are the Voice Box's, its card is the turn
system's. So the tests read both sides -- a voice renamed here is the sample
renamed in the Voice Box, a voice cloned here is a Voice Box sample with its
source, a default kept here is in VibeVoice's own file and not in a Forge
option -- and go through ``mc_voice_api``'s payload functions as well as the
adapter, because those are what a browser actually reaches.

The installer, the runtime and the turn client are the fakes from
``tests/test_voice_vibevoice_speech.py``.
"""

from __future__ import annotations

import base64
import json

import pytest

import mc_voice_box as box
import mc_voice_engines as engines
import mc_voice_paths as paths
import mc_voice_vibevoice_chat as chat
import mc_voice_vibevoice_speech as speech
from test_voice_vibevoice_speech import (  # noqa: F401  (the fixture is used by name)
    BOX_CARD, CARD, MODEL_7B, MODEL_REALTIME, WAV, tone_wav, vibevoice, wait_for)

CARTER = "vibevoice:preset:en-Carter_man"


@pytest.fixture
def selected(host, vibevoice):
    """VibeVoice selected, with the fakes in place."""
    engines.select("vibevoice")
    yield vibevoice
    chat.discard_preview()


def sample(title="Ada", seconds=4.0, source="file"):
    return box.add_sample(tone_wav(seconds), title=title, source=source)


def files_under(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


# --------------------------------------------------------------------------- #
# The registry row and what the facade asks
# --------------------------------------------------------------------------- #


class TestTheFourthEngine:
    def test_it_is_a_row_in_the_registry(self):
        assert engines.VIBEVOICE == "vibevoice" == chat.ENGINE
        assert engines.ENGINES[-1] == "vibevoice"
        found = engines.spec("vibevoice")
        assert (found.adapter, found.runtime, found.profiles) == (
            "mc_voice_vibevoice_chat", "mc_voice_vibevoice_speech",
            "mc_voice_vibevoice_profile")
        assert "graphics card" in found.blurb and "Voice Box sample" in found.blurb

    def test_the_facade_reaches_its_own_modules(self, host):
        import mc_voice_vibevoice_profile

        assert engines.adapter("vibevoice") is chat
        assert engines.runtime("vibevoice") is speech
        assert engines.profiles("vibevoice") is mc_voice_vibevoice_profile

    def test_the_selector_lists_four_engines(self, host, voice_root):
        import mc_voice_api as api
        import mc_voice_ui

        listed = [entry["id"] for entry in api.engines_payload()["engines"]]
        assert listed == ["kokoro", "sopro", "pocket", "vibevoice"]
        drawn = mc_voice_ui.engine_selector_html()
        for name in listed:
            assert f'data-mc-voice-engine-pick="{name}"' in drawn

    def test_the_model_ids_are_the_installers(self):
        import mc_voice_vibevoice

        assert chat.MODEL_7B == speech.MODEL_7B == mc_voice_vibevoice.MODEL_7B
        assert chat.MODEL_REALTIME == speech.MODEL_REALTIME == \
            mc_voice_vibevoice.MODEL_REALTIME

    def test_capabilities_clone_and_cancel_and_nothing_else(self, vibevoice):
        assert chat.capabilities() == {"clone_preview": True, "rebuild": False,
                                       "engine_settings": True, "starter_voices": False,
                                       "voice_lab": False, "interrupt_mode": "cancel"}

    def test_its_refusals_are_states(self, vibevoice):
        found = chat.refusals()
        assert chat.VibeVoiceChatError in found
        assert speech.VibeVoiceSpeechError in found
        assert vibevoice.installer.VibeVoiceError in found
        assert box.VoiceBoxError in found

    def test_the_clone_window_is_the_voice_boxs(self):
        assert chat.clone_hints() == {"min_seconds": box.SAMPLE_MIN_SECONDS,
                                      "ideal_seconds": 10.0,
                                      "max_seconds": box.SAMPLE_MAX_SECONDS}


class TestWhatIsInstalled:
    def test_ready_with_the_runtime_and_a_model_with_a_voice(self, vibevoice):
        assert chat.status().ready, "the Realtime model with presets is a voice"
        vibevoice.installer.installed[MODEL_REALTIME] = False
        assert not chat.status().ready, "the 7B with no sample has nothing to say"
        sample()
        assert chat.status().ready
        vibevoice.installer.runtime_installed = False
        found = chat.status()
        assert not found.ready
        assert "runtime is not installed" in found.message and "Voice Box tab" in found.message

    def test_cloning_needs_the_7b_and_says_the_realtime_model_cannot(self, vibevoice):
        assert chat.status().cloning_ready
        assert chat.status().cloning_message == ""
        vibevoice.installer.installed[MODEL_7B] = False
        found = chat.status()
        assert not found.cloning_ready
        assert "VibeVoice 7B" in found.cloning_message
        assert "Voice Box tab" in found.cloning_message
        assert "Realtime 0.5B" in found.cloning_message and "cannot" in found.cloning_message

    def test_reading_it_asks_for_no_card_and_starts_nothing(self, vibevoice):
        chat.status()
        chat.entries()
        chat.public_status()
        assert vibevoice.client.requests == []
        assert vibevoice.runtime.loads == [] and vibevoice.runtime.renders == []

    def test_the_public_block_names_what_a_page_may_see(self, vibevoice):
        found = chat.public_status()
        assert found["ready"] is True and found["interrupt_mode"] == "cancel"
        assert found["draining"] is False
        block = found["block"]
        for name in ("models", "settings", "cards", "loras", "warnings", "cloning_ready",
                     "cloning_message", "runtime_message", "state", "tab"):
            assert name in block, name
        assert "/private" not in json.dumps(block), "a path reached the page"
        assert {model["id"] for model in block["models"]} == {MODEL_7B, MODEL_REALTIME}
        assert [card["role"] for card in block["cards"]] == ["the image model's card",
                                                             "WanGP's card"]
        assert block["loras"] == [{"id": "warm01", "name": "Warm", "base": MODEL_7B,
                                   "bytes": 1024, "parts": ["llm"], "created": 1.0}]


# --------------------------------------------------------------------------- #
# The voices
# --------------------------------------------------------------------------- #


class TestTheVoices:
    def test_installed_presets_english_first_then_the_samples(self, vibevoice):
        made = sample("Ada")
        ids = [entry["id"] for entry in chat.entries()]
        assert ids == ["vibevoice:preset:en-Carter_man", "vibevoice:preset:en-Emma_woman",
                       "vibevoice:preset:de-Spk0_man", f"vibevoice:sample:{made['id']}"]

    def test_each_voice_carries_its_model_as_a_private_handle(self, vibevoice):
        made = sample("Ada")
        found = {entry["id"]: entry for entry in chat.entries()}
        assert found[CARTER]["_handle"] == {"model": MODEL_REALTIME, "preset": "en-Carter_man"}
        assert found[f"vibevoice:sample:{made['id']}"]["_handle"] == {
            "model": MODEL_7B, "sample": made["id"]}

    def test_the_page_sees_no_handle(self, vibevoice):
        import mc_voice_api as api

        sample("Ada")
        for entry in chat.entries():
            public = api._public(entry)
            assert "_handle" not in public and "_unprepared" not in public
            assert public["engine"] == "vibevoice"

    def test_presets_are_official_and_samples_are_editable(self, vibevoice):
        made = sample("Ada")
        found = {entry["id"]: entry for entry in chat.entries()}
        preset = found[CARTER]
        assert preset["official"] and not preset["editable"] and not preset["deletable"]
        assert preset["label"] == "Carter (English, man)"
        own = found[f"vibevoice:sample:{made['id']}"]
        assert not own["official"] and own["editable"] and own["deletable"]
        assert own["display_name"] == "Ada"
        # Nothing to rebuild: the page offers Rebuild for an incompatible voice
        # that has a retained recording, and this engine has no Rebuild.
        assert own["has_source"] is False and preset["has_source"] is False

    def test_compatible_says_whether_the_voices_model_is_installed(self, vibevoice):
        made = sample("Ada")
        vibevoice.installer.installed[MODEL_7B] = False
        found = {entry["id"]: entry for entry in chat.entries()}
        assert found[CARTER]["compatible"] is True
        own = found[f"vibevoice:sample:{made['id']}"]
        assert own["compatible"] is False
        assert "VibeVoice 7B" in own["_unprepared"] and "Voice Box tab" in own["_unprepared"]

    def test_an_uninstalled_preset_is_not_a_voice(self, vibevoice):
        assert "vibevoice:preset:fr-Spk1_woman" not in {entry["id"]
                                                        for entry in chat.entries()}
        assert chat.lookup("vibevoice:preset:fr-Spk1_woman") is None

    def test_another_engines_id_is_nothing_here(self, vibevoice):
        assert chat.lookup("pocket:official:alba") is None
        assert chat.lookup("kokoro:official:af_heart") is None


class TestTheDefaultVoice:
    def test_carter_first_then_the_first_preset_then_the_first_sample(self, vibevoice):
        assert chat.default_id() == CARTER
        vibevoice.installer.preset_rows = [row for row in vibevoice.installer.preset_rows
                                           if row["id"] != "en-Carter_man"]
        assert chat.default_id() == "vibevoice:preset:en-Emma_woman"
        vibevoice.installer.preset_rows = []
        assert chat.default_id() == ""
        made = sample("Ada")
        assert chat.default_id() == f"vibevoice:sample:{made['id']}"

    def test_set_default_is_kept_in_vibevoices_own_file_at_once(self, host, vibevoice):
        chat.set_default("vibevoice:preset:en-Emma_woman")
        stored = json.loads(paths.vibevoice_chat_path().read_text(encoding="utf-8"))
        assert stored["default"] == "vibevoice:preset:en-Emma_woman"
        assert chat.default_id() == "vibevoice:preset:en-Emma_woman"

    def test_an_apply_style_option_cannot_override_it(self, host, vibevoice):
        """I-PKT-19: no Forge option holds it, so *Apply settings* has nothing to
        write back over it."""
        chat.set_default("vibevoice:preset:en-Emma_woman")
        for name in ("model_chain_voice_vibevoice_voice_id", "model_chain_voice_vibevoice"):
            host.shared.opts.set(name, CARTER)
        assert chat.default_id() == "vibevoice:preset:en-Emma_woman"

    def test_a_default_that_is_gone_falls_back(self, host, vibevoice):
        made = sample("Ada")
        chat.set_default(f"vibevoice:sample:{made['id']}")
        box.delete_sample(made["id"])
        assert chat.default_id() == CARTER

    def test_it_cannot_be_given_another_engines_voice(self, vibevoice):
        with pytest.raises(chat.VibeVoiceChatError):
            chat.set_default("pocket:official:alba")

    def test_resolve_falls_back_to_the_default_and_refuses_when_there_is_none(
            self, vibevoice):
        assert chat.resolve("vibevoice:sample:0123456789abcdef")[0] == CARTER
        assert chat.resolve("")[0] == CARTER
        vibevoice.installer.preset_rows = []
        with pytest.raises(chat.VibeVoiceChatError) as refused:
            chat.resolve("")
        assert "no voice to speak with" in str(refused.value)

    def test_resolve_hands_back_a_voice_that_cannot_speak_as_itself(self, vibevoice):
        """Not quietly another voice: the caller refuses with the entry's sentence."""
        made = sample("Ada")
        vibevoice.installer.installed[MODEL_7B] = False
        voice_id, entry = chat.resolve(f"vibevoice:sample:{made['id']}")
        assert voice_id == f"vibevoice:sample:{made['id']}"
        assert entry["compatible"] is False and entry["_unprepared"]


class TestRenameAndDeleteAreTheVoiceBoxs:
    def test_renaming_a_sample_voice_renames_the_voice_box_sample(self, vibevoice):
        made = sample("Ada")
        found = chat.rename(f"vibevoice:sample:{made['id']}", "  Ada   Lovelace ")
        assert found["display_name"] == "Ada Lovelace"
        assert box.sample(made["id"])["title"] == "Ada Lovelace"

    def test_a_preset_keeps_its_name(self, vibevoice):
        with pytest.raises(chat.VibeVoiceChatError) as refused:
            chat.rename(CARTER, "Bob")
        assert "keeps its own name" in str(refused.value)

    def test_a_name_is_checked(self, vibevoice):
        made = sample("Ada")
        with pytest.raises(chat.VibeVoiceChatError):
            chat.rename(f"vibevoice:sample:{made['id']}", "   ")
        with pytest.raises(chat.VibeVoiceChatError):
            chat.rename(f"vibevoice:sample:{made['id']}", "x" * 81)

    def test_deleting_a_sample_voice_deletes_the_voice_box_sample(self, host, vibevoice):
        made = sample("Ada")
        kept = sample("Grace")
        configuration = box.save_configuration({"name": "Duet",
                                                "speakers": {"1": made["id"],
                                                             "2": kept["id"]}})
        chat.set_default(f"vibevoice:sample:{made['id']}")
        chat.delete(f"vibevoice:sample:{made['id']}")
        assert [entry["id"] for entry in box.samples()] == [kept["id"]]
        # Exactly the Voice Box's own Delete: the configuration loses the speaker.
        assert box.configuration(configuration["id"])["speakers"] == {"2": kept["id"]}
        assert chat.default_id() == CARTER, "the deleted default was not forgotten"

    def test_a_preset_cannot_be_deleted(self, vibevoice):
        with pytest.raises(chat.VibeVoiceChatError) as refused:
            chat.delete(CARTER)
        assert "ships with the Realtime model" in str(refused.value)

    def test_capacity_counts_samples_and_has_no_limit(self, vibevoice):
        sample("Ada")
        assert chat.capacity() == {"used": 1, "total": None, "free": None}


class TestWarnings:
    def test_the_7b_with_no_sample(self, vibevoice):
        assert any("no samples yet" in line for line in chat.warnings())
        sample("Ada")
        assert not any("no samples yet" in line for line in chat.warnings())

    def test_the_realtime_model_without_its_presets(self, vibevoice):
        for row in vibevoice.installer.preset_rows:
            row["installed"] = False
        assert any("preset voices are not" in line for line in chat.warnings())

    def test_samples_without_the_7b(self, vibevoice):
        sample("Ada")
        vibevoice.installer.installed[MODEL_7B] = False
        assert any("needs VibeVoice 7B" in line for line in chat.warnings())


class TestEngineSettings:
    def test_they_are_the_installers_chat_settings(self, vibevoice):
        found = chat.engine_settings()
        assert (found["card_uuid"], found["precision"], found["lora_id"],
                found["lora_scale"]) == (CARD, "bf16", "", 1.0)
        assert (found["card_effective"], found["card_source"]) == (CARD, "chat")
        assert [item["id"] for item in found["precisions"]] == ["bf16", "int8", "nf4"]
        assert found["precisions"][1] == {"id": "int8", "label": "8-bit",
                                          "need_vram_bytes": 12_000_000_000}

    def test_an_empty_card_is_the_voice_boxs(self, vibevoice):
        vibevoice.installer.chat["card_uuid"] = ""
        box.set_settings({"card_uuid": BOX_CARD})
        found = chat.engine_settings()
        assert (found["card_effective"], found["card_source"]) == (BOX_CARD, "voice-box")

    def test_a_change_is_merged_and_validated_by_the_installer(self, vibevoice):
        chat.apply_engine_settings({"precision": "int8"})
        chat.apply_engine_settings({"lora_id": "warm01", "lora_scale": "0.75"})
        assert vibevoice.installer.chat == {"card_uuid": CARD, "precision": "int8",
                                            "lora_id": "warm01", "lora_scale": 0.75}
        # The whole chat block every time, so a partial write cannot drop a key.
        assert set(vibevoice.installer.writes[-1]) == {"card_uuid", "precision", "lora_id",
                                                       "lora_scale"}

    def test_an_unknown_setting_is_refused_rather_than_dropped(self, vibevoice):
        with pytest.raises(chat.VibeVoiceChatError):
            chat.apply_engine_settings({"threads": 4})
        with pytest.raises(chat.VibeVoiceChatError):
            chat.apply_engine_settings({"lora_scale": "loud"})
        with pytest.raises(vibevoice.installer.VibeVoiceError):
            chat.apply_engine_settings({"precision": "fp4"})
        assert vibevoice.installer.writes == []


# --------------------------------------------------------------------------- #
# Making a voice from a recording
# --------------------------------------------------------------------------- #


class TestTheCloneTransaction:
    def test_it_is_refused_without_the_7b(self, vibevoice):
        vibevoice.installer.installed[MODEL_7B] = False
        with pytest.raises(chat.VibeVoiceChatError) as refused:
            chat.prepare_preview("Ada", tone_wav(5.0))
        assert "VibeVoice 7B" in str(refused.value)
        assert vibevoice.client.requests == []

    def test_the_name_is_checked_first(self, vibevoice):
        with pytest.raises(chat.VibeVoiceChatError):
            chat.prepare_preview("  ", tone_wav(5.0))
        with pytest.raises(chat.VibeVoiceChatError):
            chat.prepare_preview("x" * 81, tone_wav(5.0))
        assert vibevoice.client.requests == []

    def test_a_recording_outside_the_voice_boxs_window_is_refused_in_its_words(
            self, vibevoice):
        with pytest.raises(chat.VibeVoiceChatError) as refused:
            chat.prepare_preview("Ada", tone_wav(1.0))
        assert "3 to 60 seconds" in str(refused.value)
        with pytest.raises(chat.VibeVoiceChatError):
            chat.prepare_preview("Ada", b"not a wav at all")
        assert vibevoice.client.requests == [] and chat.preview_state() == {"pending": False}

    def test_the_audition_is_the_7b_through_a_card_turn_at_the_chat_settings(
            self, vibevoice):
        vibevoice.installer.chat.update({"precision": "nf4", "lora_id": "warm01",
                                         "lora_scale": 1.5})
        made = chat.prepare_preview("Ada", tone_wav(5.0))
        assert made["audio"] == WAV and made["name"] == "Ada"
        assert 4.9 < made["seconds"] < 5.1
        (card_turn,) = vibevoice.client.requests
        assert card_turn.finishes == [None] and card_turn.need_vram == 8_000_000_000
        assert vibevoice.runtime.loads[-1] == {"card": CARD, "model_id": MODEL_7B,
                                               "precision": "nf4", "lora_id": "warm01",
                                               "lora_scale": 1.5}
        (_card, job) = vibevoice.runtime.renders[0]
        assert job.script == [(1, chat.test_text())]
        pcm, rate = job.voices[1]
        assert rate == 24000 and len(pcm) == 5 * 24000 * 4, "not the recording, at 24 kHz"

    def test_nothing_is_written_before_save(self, vibevoice, voice_root):
        before = files_under(voice_root)
        made = chat.prepare_preview("Ada", tone_wav(5.0))
        assert files_under(voice_root) == before
        assert chat.preview_state() == {"pending": True, "name": "Ada",
                                        "seconds": made["seconds"]}
        assert box.samples() == []

    def test_save_makes_a_voice_box_sample_and_answers_the_new_voice(self, vibevoice):
        made = chat.prepare_preview("Ada", tone_wav(5.0))
        kept = chat.save_preview(made["token"])["voice"]
        (entry,) = box.samples()
        assert entry["title"] == "Ada" and entry["source"] == "voice-chat"
        assert kept["id"] == f"vibevoice:sample:{entry['id']}"
        assert kept["_handle"] == {"model": MODEL_7B, "sample": entry["id"]}
        assert chat.preview_state() == {"pending": False}
        assert chat.default_id() == CARTER, "saving a voice made it the default"

    def test_a_stale_token_is_refused_and_leaves_the_preview_pending(self, vibevoice):
        made = chat.prepare_preview("Ada", tone_wav(5.0))
        with pytest.raises(chat.VibeVoiceChatError) as refused:
            chat.save_preview("not-that-one")
        assert "no longer the one waiting" in str(refused.value)
        assert chat.preview_state()["pending"] is True
        chat.save_preview(made["token"])
        with pytest.raises(chat.VibeVoiceChatError) as refused:
            chat.save_preview(made["token"])
        assert "no voice waiting" in str(refused.value)

    def test_a_second_preview_replaces_the_first(self, vibevoice):
        first = chat.prepare_preview("Ada", tone_wav(5.0))
        chat.prepare_preview("Grace", tone_wav(4.0))
        assert chat.preview_state()["name"] == "Grace"
        with pytest.raises(chat.VibeVoiceChatError):
            chat.save_preview(first["token"])

    def test_discard_forgets_it_and_a_stale_discard_does_not(self, vibevoice):
        made = chat.prepare_preview("Ada", tone_wav(5.0))
        assert chat.discard_preview("someone-elses") is False
        assert chat.preview_state()["pending"] is True
        assert chat.discard_preview(made["token"]) is True
        assert chat.preview_state() == {"pending": False}
        assert box.samples() == []

    def test_a_failed_audition_leaves_nothing_pending(self, vibevoice):
        chat.prepare_preview("Ada", tone_wav(5.0))
        vibevoice.runtime.fail_render = "the VibeVoice worker exited during the render"
        with pytest.raises(speech.VibeVoiceSpeechError) as refused:
            chat.prepare_preview("Grace", tone_wav(5.0))
        assert "worker exited" in str(refused.value)
        assert chat.preview_state() == {"pending": False}, "the earlier preview survived"
        assert vibevoice.client.requests[-1].handed_back

    def test_a_cloned_voice_can_be_deleted_like_any_sample(self, vibevoice):
        made = chat.prepare_preview("Ada", tone_wav(5.0))
        kept = chat.save_preview(made["token"])["voice"]
        chat.delete(kept["id"])
        assert box.samples() == []

    def test_shutdown_forgets_a_pending_preview(self, vibevoice):
        chat.prepare_preview("Ada", tone_wav(5.0))
        speech.shutdown()
        assert chat.preview_state() == {"pending": False}


# --------------------------------------------------------------------------- #
# Through mc_voice_api, which is what a browser reaches
# --------------------------------------------------------------------------- #


class TestThroughTheApi:
    def test_the_voices_payload(self, selected):
        import mc_voice_api as api

        made = sample("Ada")
        found = api.voices_payload()
        assert found["engine"] == "vibevoice" and found["default"] == CARTER
        assert [voice["id"] for voice in found["voices"]][-1] == f"vibevoice:sample:{made['id']}"
        assert all("_handle" not in voice for voice in found["voices"])
        assert found["clone"] == chat.clone_hints()
        assert found["capacity"] == {"used": 1, "total": None, "free": None}

    def test_the_status_payloads_engine_block(self, selected):
        import mc_voice_api as api

        found = api.status_payload()
        assert found["engine"] == "vibevoice"
        assert found["tts_ready"] is True and found["interrupt_mode"] == "cancel"
        assert found["engine_state"]["backend"] == "vibevoice"
        assert found["vibevoice"]["cloning_ready"] is True
        assert found["voice"]["id"] == CARTER
        assert "pocket" not in found and "sopro" not in found

    def test_default_rename_and_delete(self, selected):
        import mc_voice_api as api

        made = sample("Ada")
        own = f"vibevoice:sample:{made['id']}"
        assert api.set_default_voice(own)["default"] == own
        api.rename_voice(own, "Ada Lovelace")
        assert box.sample(made["id"])["title"] == "Ada Lovelace"
        with pytest.raises(api.Refused) as refused:
            api.delete_voice(CARTER)
        assert refused.value.status == 400 and "ships with the Realtime model" in \
            refused.value.reason
        api.delete_voice(own)
        assert box.samples() == []
        with pytest.raises(api.Refused):
            api.set_default_voice("pocket:official:alba")

    def test_engine_settings_through_the_generic_route(self, selected):
        import mc_voice_api as api

        found = api.engine_settings({"precision": "int8"}, "vibevoice")
        assert found["ok"] and found["settings"]["precision"] == "int8"
        assert found["vibevoice"]["settings"]["precision"] == "int8", \
            "the panel was not handed its fresh state"
        with pytest.raises(api.Refused) as refused:
            api.engine_settings({"threads": 2}, "vibevoice")
        assert refused.value.status == 400

    def test_the_panel_payload_and_its_mismatch(self, host, vibevoice):
        import mc_voice_api as api

        engines.select("pocket")
        with pytest.raises(api.Refused) as refused:
            api.vibevoice_payload()
        assert refused.value.status == 409 and refused.value.mismatch
        engines.select("vibevoice")
        found = api.vibevoice_payload()
        assert found["ok"] and found["engine"] == "vibevoice"
        for name in ("settings", "cards", "models", "loras", "cloning_ready", "clone",
                     "progress", "message"):
            assert name in found, name
        assert "/private" not in json.dumps(found)

    def test_a_test_goes_through_the_speech_runtime(self, selected):
        import mc_voice_api as api

        audio = api.test_voice("", text="Hello there, friend.")
        assert audio == WAV
        (card_turn,) = selected.client.requests
        assert card_turn.finishes == [None]
        assert selected.runtime.renders[0][1].script == [(1, "Hello there, friend.")]

    def test_a_voice_that_cannot_speak_says_why_in_its_own_words(self, selected):
        import mc_voice_api as api

        made = sample("Ada")
        selected.installer.installed[MODEL_7B] = False
        with pytest.raises(api.Refused) as refused:
            api.test_voice(f"vibevoice:sample:{made['id']}", text="Hello there, friend.")
        assert refused.value.status == 409
        assert "VibeVoice 7B" in refused.value.reason
        assert "Rebuild" not in refused.value.reason
        assert selected.client.requests == []

    def test_the_clone_routes(self, selected, voice_root):
        import mc_voice_api as api

        before = files_under(voice_root)
        made = api.clone_preview("Ada", tone_wav(5.0), "vibevoice")
        assert made["ok"] and base64.b64decode(made["audio"]) == WAV
        assert made["clone"] == chat.clone_hints()
        assert files_under(voice_root) == before, "a preview wrote to disk"
        kept = api.clone_save(made["token"], "vibevoice")
        assert kept["voice"]["display_name"] == "Ada"
        assert "_handle" not in kept["voice"]
        assert kept["voice"]["id"] in [voice["id"] for voice in kept["voices"]]
        assert box.samples()[0]["source"] == "voice-chat"
        with pytest.raises(api.Refused) as refused:
            api.clone_save(made["token"], "vibevoice")
        assert refused.value.status == 400

    def test_clone_discard_and_its_refusals(self, selected):
        import mc_voice_api as api

        made = api.clone_preview("Ada", tone_wav(5.0), "vibevoice")
        found = api.clone_discard(made["token"], "vibevoice")
        assert found["discarded"] is True and found["pending"] is False
        selected.installer.installed[MODEL_7B] = False
        with pytest.raises(api.Refused) as refused:
            api.clone_preview("Ada", tone_wav(5.0), "vibevoice")
        assert refused.value.status == 400 and "VibeVoice 7B" in refused.value.reason
        with pytest.raises(api.Refused) as refused:
            api.clone_rebuild(CARTER, "vibevoice")
        assert refused.value.status == 409

    def test_a_completed_reply_is_spoken_through_a_card_turn(self, selected):
        import mc_voice_api as api

        token = api.remember_reply("Hello there, this is a reply.", voice_id=CARTER,
                                   engine="vibevoice")
        assert api.speak(token) == WAV
        assert selected.client.requests[0].finishes == [None]

    def test_the_load_button_takes_a_turn_and_stays_warm(self, selected):
        import mc_voice_api as api

        found = api.set_runtime("load")
        assert found["engine"] == "vibevoice"
        assert found["engine_state"]["loaded"] is True
        assert selected.client.requests[0].finishes == [True]
        api.set_runtime("unload")
        assert selected.client.unloads == [(CARD, "unloaded from the Voice panel")]

    def test_the_stream_says_which_engine_its_reply_was_frozen_onto(self, selected):
        """What a silent stream means is the engine's: on VibeVoice a reply
        waiting for its card, on the others something buffering. The page reads
        it here, because its last status answer may be minutes old."""
        import mc_voice_api as api
        import mc_voice_turn as turns

        spoken = turns.VoiceTurn(voice_id=CARTER, engine="vibevoice",
                                 interrupt_mode="cancel")
        spoken.sample_rate = 24000
        assert api.stream_headers(spoken)["X-Model-Chain-Voice-Engine"] == "vibevoice"
        older = turns.VoiceTurn(voice_id="official:af_heart")
        assert api.stream_headers(older)["X-Model-Chain-Voice-Engine"] == "kokoro"


class TestTheRoute:
    def test_the_status_route_is_registered_and_answers(self, host, vibevoice, voice_root):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import mc_voice_api as api

        assert api.VIBEVOICE_ROUTE in api.ROUTES
        app = FastAPI()
        assert api.install(None, app) is True
        client = TestClient(app)
        key = {api.TOKEN_HEADER: api.session_token()}
        engines.select("sopro")
        refused = client.post(api.VIBEVOICE_ROUTE, json={}, headers=key)
        assert refused.status_code == 409 and refused.json()["engine_mismatch"] is True
        engines.select("vibevoice")
        found = client.post(api.VIBEVOICE_ROUTE, json={}, headers=key)
        assert found.status_code == 200
        payload = found.json()
        assert payload["engine"] == "vibevoice" and payload["cloning_ready"] is True
        assert str(paths.data_root()) not in found.text
        assert vibevoice.client.requests == [], "a status read asked for a card"


# --------------------------------------------------------------------------- #
# The surfaces
# --------------------------------------------------------------------------- #


class TestTheSurfaces:
    def test_the_panel_says_where_it_is_installed_and_offers_no_install(self, selected):
        import mc_voice_ui

        drawn = mc_voice_ui.settings_html()
        assert 'data-mc-voice-kind="vibevoice"' in drawn
        assert "installed in the Voice Box tab" in drawn
        assert "data-mc-voice-vibevoice-install" not in drawn
        assert f'data-mc-voice-vibevoice-model="{MODEL_7B}"' in drawn
        assert 'data-mc-voice-vibevoice-part="runtime"' in drawn

    def test_the_settings_offer_the_cards_with_their_roles(self, selected):
        import mc_voice_ui

        drawn = mc_voice_ui.settings_html()
        assert 'data-mc-voice-vibevoice-setting="card_uuid"' in drawn
        assert "NVIDIA GeForce RTX 3090 — the image model&#x27;s card" in drawn or \
            "NVIDIA GeForce RTX 3090 — the image model's card" in drawn
        assert f'value="{CARD}" selected' in drawn
        assert 'data-mc-voice-vibevoice-setting="precision"' in drawn
        assert "4-bit (NF4)" in drawn
        assert 'data-mc-voice-vibevoice-setting="lora_id"' in drawn and ">Warm<" in drawn
        assert 'data-mc-voice-vibevoice-setting="lora_scale"' in drawn

    def test_the_clone_form_is_enabled_only_where_it_can_work(self, selected):
        import mc_voice_ui

        drawn = mc_voice_ui.settings_html()
        assert "Make a voice from a recording" in drawn
        assert "data-mc-voice-vibevoice-create>" in drawn, "Create is disabled"
        assert "data-mc-voice-trim" in drawn and "data-mc-voice-vibevoice-preview" in drawn
        assert "Voice Box" in drawn and "spoken by the 7B" in drawn
        selected.installer.installed[MODEL_7B] = False
        drawn = mc_voice_ui.settings_html()
        assert "data-mc-voice-vibevoice-create disabled" in drawn
        assert "Making a voice from a recording needs" in drawn

    def test_the_voice_list_is_the_generic_one_with_its_note(self, selected):
        import mc_voice_ui

        drawn = mc_voice_ui.voices_html()
        assert 'data-mc-voice-engine-id="vibevoice"' in drawn
        assert "data-mc-voice-list" in drawn and "data-mc-voice-delivery" in drawn
        assert "A sample added in the Voice Box tab appears here" in drawn
        assert 'data-mc-voice-slider-input="cfg_scale"' in drawn
        assert 'data-mc-voice-slider-input="steps"' in drawn
        assert 'data-mc-voice-unset="1"' in drawn, "a model's-own slider reads as a number"
        assert 'value="None"' not in drawn
        assert "no Speed, Pitch, Volume or Pause" in drawn
        assert "vibevoice:preset:" not in drawn, "voice data in the markup"

    def test_the_flyout_says_it_is_on_a_card(self, selected):
        import mc_voice_ui

        speech.load()
        drawn = mc_voice_ui.engine_panel()
        assert "Loaded — NVIDIA GeForce RTX 3090, idle" in drawn

    def test_the_components_overview_has_a_row(self, selected):
        import mc_voice_ui

        rows = {row["id"]: row for row in mc_voice_ui.component_rows()}
        assert rows["tts-vibevoice"]["install_state"] == "installed"
        assert rows["tts-vibevoice"]["selected"] is True


# --------------------------------------------------------------------------- #
# The character editor's sliders
# --------------------------------------------------------------------------- #


class TestTheCharacterEditorShowsTheModelsOwnNumbers:
    """A slider cannot show "the model's own". Handed ``None``, a Gradio slider
    sits at its minimum -- one diffusion step, guidance 1.0 -- and a character
    whose *Own delivery* was ticked without moving anything was saved with them.
    The editor shows the numbers the voice's own model would use instead."""

    def state(self, voice):
        import mc_voice_ui
        import mc_voice_vibevoice_profile as profile
        from prompt_master.chat.characters import Character

        found = mc_voice_ui.character_state(Character(name="Ada", vibevoice_voice=voice))
        return dict(zip(profile.FIELDS, found["values"])), found

    def test_a_sample_voice_shows_the_7bs_and_a_preset_the_realtime_models(self, selected):
        made = sample("Ada")
        cloned, found = self.state(f"vibevoice:sample:{made['id']}")
        assert cloned == {"cfg_scale": 1.3, "steps": 10}
        assert found["custom"] is False, "showing a number is not owning one"
        preset, _ = self.state(CARTER)
        assert preset == {"cfg_scale": 1.5, "steps": 5}

    def test_no_voice_of_its_own_shows_the_default_voices_model(self, selected):
        assert self.state("")[0] == {"cfg_scale": 1.5, "steps": 5}

    def test_what_somebody_set_is_shown_as_set_and_nothing_is_stored(self, selected):
        import mc_voice_vibevoice_profile as profile

        profile.remember({"steps": 12})
        assert self.state(CARTER)[0] == {"cfg_scale": 1.5, "steps": 12}
        assert profile.stored() == {"cfg_scale": None, "steps": 12}
        assert profile.shown({"cfg_scale": None, "steps": None}, CARTER) == \
            {"cfg_scale": 1.5, "steps": 5}
