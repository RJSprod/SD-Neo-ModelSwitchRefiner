"""VibeVoice's delivery in Voice Chat: guidance and diffusion steps, the model's own by default.

Two controls, both the model's, and the one property that matters most about
them: ``None`` means "the speaking model's own value" all the way to the render,
because the Realtime 0.5B and the 7B have different ones and a number frozen
here would hand one model's to the other. And where they live: VibeVoice's own
file, never a Forge option that *Apply settings* could write a stale copy over.
"""

from __future__ import annotations

import json

import pytest

import mc_voice_paths as paths
import mc_voice_vibevoice_profile as profile


class TestTheControls:
    def test_two_controls_both_the_models(self):
        assert profile.FIELDS == ("cfg_scale", "steps")
        assert profile.DELIVERY_FIELDS == ()
        assert profile.CONTROLS["cfg_scale"]["label"] == "Guidance"
        assert profile.CONTROLS["steps"]["label"] == "Diffusion steps"
        assert (profile.CONTROLS["cfg_scale"]["minimum"], profile.CONTROLS["cfg_scale"]["maximum"],
                profile.CONTROLS["cfg_scale"]["step"]) == (1.0, 3.0, 0.05)
        assert (profile.CONTROLS["steps"]["minimum"], profile.CONTROLS["steps"]["maximum"],
                profile.CONTROLS["steps"]["step"]) == (1, 30, 1)

    def test_the_default_is_the_models_own(self):
        assert profile.DEFAULTS == {"cfg_scale": None, "steps": None}
        assert profile.neutral(None)
        assert profile.describe(None) == "VibeVoice's own delivery"
        assert profile.value_label("steps", None) == "model default"

    def test_no_speed_pitch_volume_or_pause(self):
        """The worker has no signal processing to apply them with (section 37)."""
        for name in ("speed", "pitch", "gain", "pause", "temperature"):
            assert name not in profile.FIELDS
            assert name not in profile.CONTROLS

    def test_the_storage_names_are_its_own(self):
        assert profile.OPTIONS == {"cfg_scale": "model_chain_voice_vibevoice_cfg",
                                   "steps": "model_chain_voice_vibevoice_steps"}


class TestValues:
    def test_clamp_keeps_none_and_fits_the_rest(self):
        assert profile.clamp({}) == {"cfg_scale": None, "steps": None}
        assert profile.clamp({"cfg_scale": 9, "steps": 0}) == {"cfg_scale": 3.0, "steps": 1}
        assert profile.clamp({"cfg_scale": "1.234", "steps": "12.6"}) == {"cfg_scale": 1.23,
                                                                          "steps": 13}
        assert profile.clamp({"cfg_scale": "loud", "steps": float("nan")}) == {
            "cfg_scale": None, "steps": None}
        assert profile.clamp({"cfg_scale": True}) == {"cfg_scale": None, "steps": None}

    def test_labels_and_the_summary(self):
        assert profile.value_label("cfg_scale", 1.5) == "1.5"
        assert profile.value_label("steps", 12) == "12"
        assert profile.describe({"cfg_scale": 2.0, "steps": 8}) == \
            "guidance 2, diffusion steps 8"
        assert profile.describe({"steps": 8}) == "diffusion steps 8"

    def test_a_request_carries_only_what_was_set(self):
        """An absent key is what the speech runtime reads as the model's own."""
        assert profile.request(None) == {}
        assert profile.request({"cfg_scale": 2.0}) == {"cfg_scale": 2.0}
        assert profile.request({"steps": 7.4}) == {"steps": 7}
        assert isinstance(profile.request({"steps": 7})["steps"], int)


class TestWhereItIsKept:
    def test_remember_writes_voice_chats_vibevoice_file_at_once(self, host, voice_root):
        profile.remember({"cfg_scale": 2.2, "steps": 12})
        stored = json.loads(paths.vibevoice_chat_path().read_text(encoding="utf-8"))
        assert stored[profile.OPT_CFG] == 2.2
        assert stored[profile.OPT_STEPS] == 12
        assert profile.stored() == {"cfg_scale": 2.2, "steps": 12}

    def test_none_survives_a_round_trip(self, host, voice_root):
        profile.remember({"cfg_scale": 2.2, "steps": 12})
        profile.remember({"cfg_scale": None, "steps": 12})
        assert profile.stored() == {"cfg_scale": None, "steps": 12}

    def test_an_apply_style_option_cannot_override_the_file(self, host, voice_root):
        """I-PKT-19. A host option is a component on the settings page too, and
        *Apply settings* writes the page's build-time copy back into the store."""
        profile.remember({"cfg_scale": 2.2, "steps": None})
        host.shared.opts.set(profile.OPT_CFG, 1.0)
        host.shared.opts.set(profile.OPT_STEPS, 30)
        assert profile.stored() == {"cfg_scale": 2.2, "steps": None}

    def test_an_option_is_not_read_even_when_the_file_is_empty(self, host, voice_root):
        """There is no older build whose option this has to honour."""
        host.shared.opts.set(profile.OPT_CFG, 2.9)
        assert profile.stored() == {"cfg_scale": None, "steps": None}

    def test_nothing_is_registered_with_the_host(self, host):
        import model_chain  # noqa: F401  (registers the settings sections on import)

        for name in profile.OPTIONS.values():
            assert name not in host.shared.options_templates, name

    def test_an_unreadable_file_is_the_models_own(self, host, voice_root):
        target = paths.vibevoice_chat_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{not json", encoding="utf-8")
        assert profile.stored() == {"cfg_scale": None, "steps": None}


class TestLayers:
    def test_a_character_overrides_one_field_and_follows_the_other(self, host, voice_root):
        profile.remember({"cfg_scale": 2.2, "steps": 12})
        assert profile.resolve({"steps": 20}) == {"cfg_scale": 2.2, "steps": 20}
        assert profile.resolve({"cfg_scale": None, "steps": None}) == {"cfg_scale": 2.2,
                                                                       "steps": 12}

    def test_none_everywhere_is_the_model_at_the_render(self, host, voice_root):
        assert profile.request(profile.resolve({})) == {}

    def test_overrides_keep_what_a_character_did_not_set(self):
        assert profile.overrides({"steps": 9}) == {"cfg_scale": None, "steps": 9}


@pytest.mark.parametrize("name", ["cfg_scale", "steps"])
def test_every_control_has_help_that_is_not_a_promise(name):
    """Section 37: said what it is, and not an emotion control."""
    text = profile.CONTROLS[name]["help"]
    assert len(text) > 80
    assert "model's own value" in text
