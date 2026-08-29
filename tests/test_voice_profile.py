"""Delivery: what Kokoro does, what Voice Chat does, and where the line is.

The interesting property here is not that a slider stores a number. It is that
one of the four controls is the model's and three are arithmetic on the samples
it produced, and that the split is visible rather than glossed:

    speed   goes into ``generate``
    pitch   is a resample, and the synthesis speed is divided by it so the
            duration comes back to where speed asked for it
    volume  is a scalar folded into the PCM16 conversion
    pacing  is silence between segments, which only a streamed turn has

The other property is inheritance. A character field that is ``None`` follows
the default voice, and that is not the same as a character field set to today's
default -- one tracks Settings and the other freezes it.
"""

from __future__ import annotations

import math

import pytest

import mc_voice_profile as profile


class TestTheValues:
    def test_a_missing_profile_is_kokoro_exactly_as_it_comes(self):
        assert profile.clamp() == {"speed": 1.0, "pitch": 0.0, "gain": 0.0, "pause": 0.0}
        assert profile.neutral() is True

    def test_every_field_is_bounded_rather_than_believed(self):
        found = profile.clamp({"speed": 99, "pitch": -99, "gain": 99, "pause": -5})
        assert found["speed"] == profile.CONTROLS["speed"]["maximum"]
        assert found["pitch"] == profile.CONTROLS["pitch"]["minimum"]
        assert found["gain"] == profile.CONTROLS["gain"]["maximum"]
        assert found["pause"] == profile.CONTROLS["pause"]["minimum"]

    @pytest.mark.parametrize("offered", ["", "fast", None, [], {}, float("nan"),
                                         float("inf"), True])
    def test_a_value_that_is_not_a_number_is_the_default(self, offered):
        """Read from a settings store, a hand-edited YAML file and a JSON body,
        so every way this can be wrong has to mean the same thing."""
        assert profile.clamp({"speed": offered})["speed"] == 1.0

    def test_overrides_keep_none_as_none(self):
        """The whole of how a character follows Settings. ``clamp`` fills a
        blank with the default; ``overrides`` leaves it blank."""
        found = profile.overrides({"speed": 1.2})
        assert found == {"speed": 1.2, "pitch": None, "gain": None, "pause": None}

    def test_describe_names_only_what_has_changed(self):
        assert profile.describe() == "Kokoro's own delivery"
        assert profile.describe({"pitch": 3}) == "pitch +3 semitones"
        assert profile.describe({"speed": 1.2, "gain": -2}) == "speed 1.2x, volume -2 dB"


class TestWhatTheWorkerIsTold:
    def test_pitch_is_a_ratio_and_twelve_semitones_is_an_octave(self):
        assert profile.pitch_ratio(0) == pytest.approx(1.0)
        assert profile.pitch_ratio(12) == pytest.approx(2.0)
        assert profile.pitch_ratio(-12) == pytest.approx(0.5)

    def test_volume_is_a_linear_scalar_and_six_db_is_about_double(self):
        assert profile.gain_scale(0) == pytest.approx(1.0)
        assert profile.gain_scale(6) == pytest.approx(2.0, rel=0.01)
        assert profile.gain_scale(-6) == pytest.approx(0.5, rel=0.01)

    def test_the_request_is_multipliers_rather_than_settings(self):
        """The conversion happens in the process that has the settings, so the
        worker stays a thing that multiplies and resamples what it is told."""
        found = profile.request({"speed": 1.1, "pitch": 12, "gain": 6, "pause": 250})
        assert found["speed"] == pytest.approx(1.1)
        assert found["pitch"] == pytest.approx(2.0)
        assert found["gain"] == pytest.approx(2.0, rel=0.01)
        assert found["pause_ms"] == 250

    def test_a_neutral_profile_asks_for_nothing(self):
        assert profile.request() == {"speed": 1.0, "pitch": 1.0, "gain": 1.0, "pause_ms": 0}


class TestInheritance:
    def test_a_character_with_nothing_set_follows_the_default(self, host):
        profile.remember({"speed": 1.3, "pitch": 2.0})
        found = profile.resolve({"speed": None, "pitch": None, "gain": None, "pause": None})
        assert found["speed"] == 1.3
        assert found["pitch"] == 2.0

    def test_a_character_overrides_only_what_it_names(self, host):
        profile.remember({"speed": 1.3, "pitch": 2.0})
        found = profile.resolve({"speed": 0.8})
        assert found["speed"] == 0.8
        assert found["pitch"] == 2.0, "an unset field stopped following the default"

    def test_changing_the_default_moves_every_character_that_follows_it(self, host):
        """The reason inheritance is by absence rather than by copying: somebody
        who slows the default voice down expects the characters they never
        configured to slow down with it."""
        profile.remember({"speed": 1.0})
        assert profile.resolve({})["speed"] == 1.0
        profile.remember({"speed": 0.9})
        assert profile.resolve({})["speed"] == 0.9

    def test_a_host_that_will_not_answer_is_kokoro_rather_than_an_exception(self,
                                                                           monkeypatch):
        """Read on the path that speaks a reply. A settings object that raises
        is a reply in the model's own delivery, never a reply that goes
        unspoken."""
        import sys

        monkeypatch.setitem(sys.modules, "modules", None)
        assert profile.stored() == profile.DEFAULTS


class TestTheHonestPart:
    def test_there_is_no_emotion_control(self):
        """Kokoro-82M has no emotion input and sherpa's ``generate`` takes
        ``sid`` and ``speed``. A slider for one would do nothing, so there is
        not one -- and this test is here so that adding one has to argue with
        something."""
        assert set(profile.FIELDS) == {"speed", "pitch", "gain", "pause"}
        assert "emotion" not in profile.CONTROLS

    def test_the_module_says_which_control_is_the_models(self):
        assert "Kokoro's own" in profile.__doc__
        assert "no emotion input" in profile.__doc__ or "emotion" in profile.__doc__
