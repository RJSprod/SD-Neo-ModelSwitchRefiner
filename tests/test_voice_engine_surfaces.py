"""T-ENG-1, read from the markup the server actually produces.

Section 5 is the rule this file exists for: the inactive engine's operational
controls are *absent* from the document rather than hidden in it, so that a
stale DOM, a theme script, a partial Gradio re-render or one stray CSS override
cannot expose them. The only way to assert absence is to render every surface
on each engine and look.
"""

from __future__ import annotations

import pytest

import mc_voice_engines as engines
import mc_voice_ui


def test_kokoro_surfaces_render_and_carry_no_sopro_controls(host, voice_root,
                                                            kokoro_bundle, voice_registry):
    engines.select("kokoro")
    settings = mc_voice_ui.settings_html()
    voices = mc_voice_ui.voices_html()
    overlay = mc_voice_ui.engine_panel()
    both = settings + voices + overlay
    assert "Kokoro" in settings
    # The selector is the one place both names appear.
    assert settings.count("Sopro") >= 1
    for forbidden in ("data-mc-voice-sopro-install", "data-mc-voice-lab",
                      "data-mc-voice-sopro-create", "Style control",
                      "Conditioning Blend", "data-mc-voice-sopro-setting"):
        assert forbidden not in voices + overlay, forbidden
        if forbidden != "data-mc-voice-sopro-install":
            assert forbidden not in settings, forbidden


def test_sopro_surfaces_render_and_carry_no_kokoro_controls(host, voice_root,
                                                            kokoro_bundle, voice_registry):
    engines.select("sopro")
    settings = mc_voice_ui.settings_html()
    voices = mc_voice_ui.voices_html()
    overlay = mc_voice_ui.engine_panel()
    assert "Sopro V2 Turbo" in settings
    assert "data-mc-voice-sopro-install" in settings
    assert "Style control 1" in voices and "Style control 8" in voices
    assert "Conditioning Blend" in voices
    # Kokoro's operational controls are absent from the document, not hidden.
    for forbidden in ('data-mc-voice-install="tts"', "Download default TTS",
                      "data-mc-voice-cloning", "Storytime"):
        assert forbidden not in settings + voices + overlay, forbidden
    print("\nsopro settings bytes:", len(settings), "voices bytes:", len(voices))


def test_the_status_payload_carries_only_the_selected_engine(host, voice_root,
                                                             kokoro_bundle,
                                                             voice_registry):
    """The other half of section 5, and the one a stale page actually reads.

    Scoping the *payload* rather than only the markup is what makes the rule
    survive a page that was open when somebody switched: there is nothing in
    the answer for it to paint.
    """
    import mc_voice_api as api

    engines.select("kokoro")
    found = api.status_payload()
    assert found["engine"] == "kokoro"
    assert "sopro" not in found
    assert "kokoro" in found

    engines.select("sopro")
    found = api.status_payload()
    assert found["engine"] == "sopro"
    assert "kokoro" not in found
    assert "sopro" in found


def test_speech_to_text_is_reported_on_both_engines(host, voice_root, kokoro_bundle,
                                                    voice_registry):
    """I-7. Dictation is outside the selector, so every speech-to-text field is
    in the payload whichever engine is selected -- a browser that lost its
    microphone because somebody changed the voice would be the exact coupling
    the invariant forbids."""
    import mc_voice_api as api

    engines.select("kokoro")
    before = {key: api.status_payload()[key]
              for key in ("stt_ready", "stt_message", "stt_model", "runtime_ready")}
    engines.select("sopro")
    after = {key: api.status_payload()[key]
             for key in ("stt_ready", "stt_message", "stt_model", "runtime_ready")}
    assert before == after


def test_a_mutation_naming_the_other_engine_is_refused(host, voice_root, kokoro_bundle,
                                                       voice_registry):
    """A request replayed from a page drawn before somebody switched. Refused
    with a sentence rather than applied to whichever engine is selected now."""
    import mc_voice_api as api

    engines.select("sopro")
    for call in (lambda: api.voices_payload(engine="kokoro"),
                 lambda: api.set_default_voice("kokoro:official:af_heart",
                                               engine="kokoro"),
                 lambda: api.profile_payload(None, "kokoro"),
                 lambda: api.test_voice("kokoro:official:af_heart", engine="kokoro")):
        with pytest.raises(api.Refused) as raised:
            call()
        # 409 rather than 500, and flagged, because this is not "that failed" --
        # it is "the page you are looking at is out of date", and the flag is
        # what the browser reloads on. A status code alone would not do: 409
        # already means several other things on these routes.
        assert raised.value.status == 409
        assert raised.value.mismatch is True


def test_a_refusal_that_is_not_a_mismatch_is_not_flagged(host, voice_root, kokoro_bundle,
                                                         voice_registry):
    """The other half of the flag. A Lab session that expired, a turn that was
    over before anything listened to it and an install already running are all
    409s, and none of them is a reason to reload somebody's settings page."""
    import mc_voice_api as api

    engines.select("kokoro")
    with pytest.raises(api.Refused) as raised:
        api.open_stream("nothing-like-a-turn")
    assert raised.value.mismatch is False


def test_the_engines_payload_is_the_one_place_both_names_appear(host, voice_root):
    import mc_voice_api as api

    found = api.engines_payload()
    names = {entry["id"] for entry in found["engines"]}
    assert names == {"kokoro", "sopro"}
    # And it carries nothing operational about either: an id, a label, a
    # sentence, and whether it is installed.
    for entry in found["engines"]:
        assert set(entry) == {"id", "label", "blurb", "active", "installed"}


def test_the_residency_object_survives_the_scoping_filter(host, voice_root, kokoro_bundle,
                                                          voice_registry):
    """``engine`` is the selected engine's id in every payload this feature
    sends, and the flyout's Loaded/Unloaded state is ``engine_state``.

    One key meaning two things is not a style question here: the scoping filter
    sets ``engine`` to the id, so while the residency object was under that name
    the filter silently replaced the Voice flyout's status line with the string
    "kokoro". Two names, and this test is why.
    """
    import mc_voice_api as api

    for chosen in ("kokoro", "sopro"):
        engines.select(chosen)
        found = api.status_payload()
        assert found["engine"] == chosen
        assert isinstance(found["engine_state"], dict)
        assert "state" in found["engine_state"]


def test_load_and_unload_follow_the_selected_engine(host, voice_root, kokoro_bundle,
                                                    voice_registry, monkeypatch):
    """The button lives in a flyout that says which engine it belongs to, so
    pressing it must not load the other one."""
    import mc_voice_api as api
    import mc_voice_runtime
    import mc_voice_sopro_runtime

    asked = []
    monkeypatch.setattr(mc_voice_runtime, "unload",
                        lambda reason="": asked.append("kokoro") or {"state": "unloaded"})
    monkeypatch.setattr(mc_voice_sopro_runtime, "unload",
                        lambda reason="": asked.append("sopro") or {"state": "unloaded"})

    engines.select("kokoro")
    assert api.set_runtime("unload")["engine"] == "kokoro"
    engines.select("sopro")
    assert api.set_runtime("unload")["engine"] == "sopro"
    assert asked == ["kokoro", "sopro"]
