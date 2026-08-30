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


def test_the_surface_route_answers_for_the_engine_selected_now(host, voice_root,
                                                               kokoro_bundle,
                                                               voice_registry):
    """The fix for a browser that reloaded itself until the tab was closed.

    Forge builds a settings row's HTML once, when the extension is imported, and
    hands that same string to every page load for the life of the process. So
    the document a reload came back to was still built for the engine that was
    selected at startup, the page decided it was stale -- correctly -- and
    reloaded again. What breaks the loop is markup built *now*, which is what
    this route is: both calls below happen in one process, and they differ.
    """
    import mc_voice_api as api

    engines.select("kokoro")
    first = api.surface_payload()
    engines.select("sopro")
    second = api.surface_payload()

    assert first["engine"] == "kokoro" and second["engine"] == "sopro"
    assert first["settings"] != second["settings"]
    assert "data-mc-voice-sopro-install" not in first["settings"]
    assert "data-mc-voice-sopro-install" in second["settings"]
    # And the voices half with it, or the page would swap one and keep the
    # other engine's voice list beside it.
    assert "Conditioning Blend" in second["voices"]
    assert "Conditioning Blend" not in first["voices"]


def test_the_surface_carries_the_engine_id_the_page_compares_against(host, voice_root,
                                                                     kokoro_bundle,
                                                                     voice_registry):
    """A swap that did not update the id would mismatch again on the next poll,
    which is the loop wearing different clothes."""
    import mc_voice_api as api

    for wanted in ("kokoro", "sopro"):
        engines.select(wanted)
        found = api.surface_payload()
        assert f'data-mc-voice-engine-id="{wanted}"' in found["settings"]
        assert f'data-mc-voice-engine-id="{wanted}"' in found["voices"]
        assert found["engine"] == api.voices_payload()["engine"]


def test_sopros_two_manual_sections_say_which_is_which(host, voice_root, kokoro_bundle,
                                                       voice_registry, monkeypatch):
    """Both used to read "Or install from files you download yourself", one
    directly under the other, which says nothing about which installs the
    PyTorch runtime and which installs the model.

    The addresses are stood in for because this row only lists them on a
    platform the Sopro closure is pinned for, and the labels are not a property
    of the platform.
    """
    import mc_voice_sopro as sopro

    monkeypatch.setattr(sopro, "sources", lambda kind: [
        {"url": f"https://example.invalid/{kind}.bin", "filename": f"{kind}.bin",
         "save_as": f"{kind}.bin"}])
    engines.select("sopro")
    settings = mc_voice_ui.sopro_html()

    assert "Or install the PyTorch runtime from files you download yourself" in settings
    assert "Or install the model artifacts from files you download yourself" in settings
    assert "<summary>Or install from files you download yourself</summary>" not in settings


def test_the_startup_line_describes_the_engine_that_is_selected(host, voice_root,
                                                                kokoro_bundle,
                                                                voice_registry, caplog):
    """The log's one-line summary used to describe sherpa, Whisper and Kokoro
    whatever engine was selected.

    So a log sent after "Voice Chat does not work" could not answer the first
    question anybody asks of it -- is the engine that is supposed to be speaking
    installed, and does it have a voice to speak with -- and one real report was
    diagnosed from a traceback further down instead.
    """
    import logging

    import mc_voice_api as api

    engines.select("sopro")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="model_chain"):
        api._log_engine()

    said = " ".join(record.getMessage() for record in caplog.records)
    assert "Sopro" in said
    assert "not installed" in said
    assert "0 voice(s)" in said
    assert "silent until one is created" in said


def test_the_startup_line_names_kokoro_when_kokoro_is_selected(host, voice_root,
                                                               kokoro_bundle,
                                                               voice_registry, caplog):
    import logging

    import mc_voice_api as api

    engines.select("kokoro")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="model_chain"):
        api._log_engine()

    said = " ".join(record.getMessage() for record in caplog.records)
    assert "Kokoro" in said


def test_a_refused_clone_is_written_down_and_not_only_answered(host, voice_root,
                                                               kokoro_bundle,
                                                               voice_registry, caplog):
    """The diagnostic hole behind a real report.

    The clone route answered the browser with an exact reason -- "that WAV is
    not something Sopro can read" -- and logged nothing at all, so the log the
    user sent said only that a reply had not been spoken. A refusal nobody can
    reconstruct afterwards is a refusal that reads as "nothing worked".
    """
    import logging

    import mc_voice_api as api

    engines.select("sopro")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="model_chain"):
        with pytest.raises(api.Refused) as raised:
            api.sopro_clone("Ada", "en", b"not a wav at all")

    assert raised.value.status == 400
    said = [record.getMessage() for record in caplog.records]
    assert any("is being created" in line for line in said), said
    assert any("was not created" in line for line in said), said

