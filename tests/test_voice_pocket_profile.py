"""PocketTTS's delivery profile: its ranges, and the ``None`` that means inherit.

Small, and load-bearing. Three engines now have a control called Speed and a
control called Variation, and the whole of I-PKT-3 rests on those being three
separate values in three separate places -- so the first thing asserted here is
that no storage name overlaps.

The second is the sentinel. ``None`` means something different at each of three
layers and all three meanings are correct: in a character it means "follow the
Pocket default", in the Pocket default it means "follow the model's own
configuration", and at the worker it means "no temperature key at all". A layer
that substituted today's number for it would freeze the model's own
recommendation into an installation, and a model revision that changed it would
then change nobody's voice (I-PKT-25, section 14).
"""

from __future__ import annotations

import math

import mc_voice_pocket_profile as profile
import mc_voice_profile as kokoro_profile
import mc_voice_sopro_profile as sopro_profile


class TestNoValueIsSharedWithAnotherEngine:
    """T-PROFILE-P1 / T-PKT-PRO-6."""

    def test_no_storage_name_overlaps_kokoro_or_sopro(self):
        mine = set(profile.OPTIONS.values())
        theirs = set(sopro_profile.OPTIONS.values()) | {
            kokoro_profile.OPT_SPEED, kokoro_profile.OPT_PITCH,
            kokoro_profile.OPT_GAIN, kokoro_profile.OPT_PAUSE}
        assert mine & theirs == set()

    def test_every_name_is_prefixed_with_this_engine(self):
        assert all(name.startswith("model_chain_voice_pocket_")
                   for name in profile.OPTIONS.values())

    def test_moving_a_pocket_default_leaves_the_other_engines_alone(self, host,
                                                                    voice_root):
        """T-PROFILE-P5 / I-PKT-3. Enforced by the spelling rather than by a
        check somebody could forget."""
        before = sopro_profile.stored(), kokoro_profile.stored()
        profile.remember(speed=1.4, pitch=-2.0)
        assert profile.stored()["speed"] == 1.4
        assert (sopro_profile.stored(), kokoro_profile.stored()) == before


class TestTheControls:
    def test_every_field_has_a_complete_control(self):
        for name in profile.FIELDS:
            spec = profile.CONTROLS[name]
            for key in ("label", "unit", "minimum", "maximum", "step", "default",
                        "decimals", "group", "owner", "help"):
                assert key in spec, f"{name} is missing {key}"

    def test_the_delivery_controls_belong_to_voice_chat_and_say_so(self):
        """Section 15. The reviewed PocketTTS API has no speaking-rate argument,
        so Speed is this feature's own DSP and the help text has to say which."""
        for name in profile.DELIVERY_FIELDS:
            assert profile.CONTROLS[name]["owner"] == "voice-chat"
        assert "no speaking-rate input of its own" in profile.CONTROLS["speed"]["help"]

    def test_variation_belongs_to_pocket_and_is_not_an_emotion_control(self):
        """Section 14, I-PKT-26. The model has no such input and a slider
        claiming to be one would be a promise nobody has tested."""
        spec = profile.CONTROLS["temperature"]
        assert spec["owner"] == "pocket"
        assert spec["label"] == "Variation"
        assert "not an emotion" in spec["help"]

    def test_the_ranges_are_the_ones_the_design_names(self):
        assert (profile.CONTROLS["speed"]["minimum"],
                profile.CONTROLS["speed"]["maximum"]) == (0.5, 2.0)
        assert (profile.CONTROLS["pitch"]["minimum"],
                profile.CONTROLS["pitch"]["maximum"]) == (-12.0, 12.0)
        assert (profile.CONTROLS["gain"]["minimum"],
                profile.CONTROLS["gain"]["maximum"]) == (-12.0, 12.0)
        assert (profile.CONTROLS["pause"]["minimum"],
                profile.CONTROLS["pause"]["maximum"]) == (0.0, 1200.0)
        assert (profile.CONTROLS["temperature"]["minimum"],
                profile.CONTROLS["temperature"]["maximum"]) == (0.1, 1.2)

    def test_there_is_no_compute_setting_in_here(self):
        """I-PKT-23. Precision, sampler steps and the model change how the
        runtime executes rather than how one character sounds, and a character
        carrying one would restart a worker by speaking."""
        for name in ("precision", "steps", "sampler_steps", "threads", "model",
                     "model_id", "seed"):
            assert name not in profile.FIELDS
            assert name not in profile.CONTROLS

    def test_pocket_has_no_language_field(self):
        """Section 13. The model *is* the language and it is engine-global."""
        assert "language" not in profile.FIELDS
        assert "language" not in profile.clamp({})


class TestClamping:
    """T-PKT-PRO-1 and T-PROFILE-P3."""

    def test_values_are_held_inside_their_range(self):
        found = profile.clamp({"speed": 9.0, "pitch": -99.0, "gain": 40.0,
                               "pause": 9999})
        assert found["speed"] == 2.0
        assert found["pitch"] == -12.0
        assert found["gain"] == 12.0
        assert found["pause"] == 1200

    def test_nonsense_becomes_the_default_rather_than_raising(self):
        found = profile.clamp({"speed": "quick", "pitch": None, "gain": float("nan"),
                               "pause": float("inf")})
        assert found["speed"] == 1.0
        assert found["pitch"] == 0.0
        assert found["gain"] == 0.0
        assert found["pause"] == 0

    def test_a_clamped_profile_is_always_complete(self):
        assert set(profile.clamp({})) == set(profile.FIELDS)


class TestTheModelDefaultSentinel:
    """T-PKT-PRO-2, T-PKT-PRO-3, T-PROFILE-P4."""

    def test_an_unset_variation_stays_unset_through_clamp(self):
        assert profile.clamp({})["temperature"] is None

    def test_an_unset_variation_stays_unset_through_overrides(self):
        assert profile.overrides({})["temperature"] is None
        assert profile.overrides({"temperature": ""})["temperature"] is None

    def test_a_delivery_field_takes_its_default_but_a_generation_field_does_not(self):
        found = profile.clamp({})
        assert found["speed"] == 1.0
        assert found["temperature"] is None

    def test_it_survives_the_character_to_resolve_path(self, host, voice_root):
        """Three layers, one meaning, and no layer substitutes a number."""
        assert profile.resolve({})["temperature"] is None
        assert profile.resolve({"speed": 1.2})["temperature"] is None
        assert profile.resolve({"temperature": 0.45})["temperature"] == 0.45

    def test_it_reaches_the_worker_as_an_absent_key(self):
        """I-PKT-25. Sending a number would make the model's own recommendation
        a value this process had to know."""
        assert "temperature" not in profile.request({})
        assert profile.request({"temperature": 0.5})["temperature"] == 0.5


class TestInheritanceIsAbsence:
    """I-PKT-4. A character with nothing set follows the Pocket default."""

    def test_an_unset_character_field_takes_the_default(self, host, voice_root):
        profile.remember(speed=1.3, pitch=2.0)
        found = profile.resolve({"speed": None, "pitch": None})
        assert found["speed"] == 1.3
        assert found["pitch"] == 2.0

    def test_a_set_character_field_wins_field_by_field(self, host, voice_root):
        profile.remember(speed=1.3, pitch=2.0)
        found = profile.resolve({"speed": 0.9})
        assert found["speed"] == 0.9
        assert found["pitch"] == 2.0

    def test_overrides_keeps_none_rather_than_materialising_a_default(self):
        found = profile.overrides({"speed": 1.1})
        assert found["speed"] == 1.1
        assert found["pitch"] is None
        assert found["gain"] is None


class TestWhatTheWorkerIsTold:
    def test_the_request_carries_the_converted_values(self):
        found = profile.request({"speed": 1.25, "pitch": 12.0, "gain": 6.0, "pause": 200})
        assert found["speed"] == 1.25
        assert math.isclose(found["pitch"], 2.0, rel_tol=1e-6)
        assert math.isclose(found["gain"], 10.0 ** (6.0 / 20.0), rel_tol=1e-6)
        assert found["pause_ms"] == 200

    def test_a_neutral_profile_is_recognised_as_neutral(self):
        assert profile.neutral({}) is True
        assert profile.neutral({"speed": 1.1}) is False
        assert profile.neutral({"temperature": 0.4}) is False

    def test_semitones_and_decibels_are_converted_once_here(self):
        assert math.isclose(profile.pitch_ratio(-12), 0.5, rel_tol=1e-6)
        assert math.isclose(profile.gain_scale(-6.0), 10.0 ** (-6.0 / 20.0), rel_tol=1e-6)


class TestWhatAPanelWrites:
    def test_a_value_label_carries_its_unit_and_sign(self):
        assert profile.value_label("speed", 1.25) == "1.25x"
        assert profile.value_label("pitch", 3.0) == "+3 semitones"
        assert profile.value_label("gain", -2.5) == "-2.5 dB"
        assert profile.value_label("pause", 200) == "200 ms"

    def test_an_unset_generation_field_reads_as_the_model_default(self):
        assert profile.value_label("temperature", None) == "model default"

    def test_describe_names_only_what_changed(self):
        assert profile.describe({}) == "PocketTTS's own delivery"
        assert "speed 1.2x" in profile.describe({"speed": 1.2})
        assert "variation" in profile.describe({"temperature": 0.5}).casefold()
