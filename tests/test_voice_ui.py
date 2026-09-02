"""Voice Chat's controls in Conversation, and the switch behind each of them.

Two kinds of test. The first kind builds the Conversation panel and reads what
came out of it: a Voice chip in the header, a microphone in the composer row, a
fourth overlay in the panel's existing one-at-a-time surface machinery, and --
the one that matters most -- a success-only continuation on every single one of
the six ways a reply can be produced.

That last one is worth saying plainly. ``.then()`` runs whether or not the
event before it raised. ``.success()`` runs only after one that did not. If the
speech marker were attached with ``then``, a generation that failed would still
reach it, and Voice Chat would read a failure aloud. So the test walks the built
panel's recorded callbacks and asserts the marker is on ``success`` and on all
six -- not five.

The second kind drives the handlers directly: what the flyout says when nothing
is installed, that a switch writes through to the host's own options store
immediately, and that the marker refuses in each of the four ways it has to.
"""

from __future__ import annotations

import pytest

import mc_llm_chat_panel
import mc_llm_paths
import mc_voice_api
import mc_voice_models
import mc_voice_state
import mc_voice_profile
import mc_voice_paths as paths
import mc_voice_ui


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture
def installed(monkeypatch):
    ready = mc_voice_models.Status(True, True, True, "Installed", "Installed", "Installed",
                                  True, "whisper-small-int8", "kokoro-multi-lang-v1-cpu",
                                  "af_heart")
    monkeypatch.setattr(mc_voice_models, "status", lambda: ready)
    return ready


def callbacks(component):
    return getattr(component, "_callbacks", [])


def build():
    return mc_llm_chat_panel.build()


def find(root, elem_id):
    """The component carrying ``elem_id``, from everything the build created.

    Gradio's own components are the only record of what a panel built, and the
    stub keeps every one of them: walking them is how a test asks "is the
    microphone in the composer" without a browser.
    """
    for component in _COMPONENTS:
        if getattr(component, "elem_id", None) == elem_id:
            return component
    return None


_COMPONENTS = []


@pytest.fixture(autouse=True)
def _collect(monkeypatch):
    """Remember every component Gradio was asked to make during a build."""
    import gradio as gr

    _COMPONENTS.clear()
    original = gr.components.Component.__init__

    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _COMPONENTS.append(self)

    monkeypatch.setattr(gr.components.Component, "__init__", record)
    yield
    _COMPONENTS.clear()


class TestTheControlsAreThere:
    def test_the_header_has_a_voice_chip(self):
        build()
        assert find(None, "mc-llm-chat-to-voice") is not None

    def test_the_composer_has_a_microphone(self):
        build()
        mic = find(None, "mc-llm-chat-voice-mic")
        assert mic is not None
        # The same touch target the paperclip beside it establishes.
        assert mic.__dict__.get("min_width") == 44

    def test_the_microphone_sits_in_a_track_twice_its_own_width(self):
        """The 2 x 1 area the slide gesture needs. The handle rests at the left
        of it and is carried to the right-hand end to start recording, so the
        track's width *is* the travel -- a track the size of the handle would be
        a press-and-hold with extra steps."""
        build()
        track = find(None, "mc-llm-chat-voice-track")
        assert track is not None, "the microphone has no track to slide along"
        assert track.__dict__.get("min_width") == 2 * mc_voice_ui.MIC_PX
        assert "mc-llm-voice-track" in (track.__dict__.get("elem_classes") or [])

    def test_the_character_editor_carries_a_voice_and_a_delivery(self):
        """"I can have nicole for one character and heart for another."

        The list itself is painted by the browser -- it changes when a clone
        finishes -- so what Gradio owns is the hidden field the browser writes
        and the ordinary Save reads. The four sliders are Gradio's own, because
        they are values with no liveness to them.
        """
        build()
        assert find(None, "mc-llm-chat-character-voice-list") is not None
        assert find(None, "mc-llm-chat-character-voice") is not None
        assert find(None, "mc-llm-chat-character-voice-custom") is not None
        for name in ("speed", "pitch", "gain", "pause"):
            slider = find(None, f"mc-llm-chat-character-voice-{name}")
            assert slider is not None, name
            control = mc_voice_profile.CONTROLS[name]
            assert slider.__dict__.get("minimum") == control["minimum"], name
            assert slider.__dict__.get("maximum") == control["maximum"], name

    def test_the_selection_is_a_hidden_field_rather_than_a_second_save(self):
        """No second store and nothing to get out of step with the character
        file: the browser writes the id, Save character reads it."""
        build()
        held = find(None, "mc-llm-chat-character-voice")
        assert held.__dict__.get("visible") is False

    def test_the_microphone_is_never_disabled(self):
        """Section 14: pressing it with nothing installed is how somebody finds
        out what is wrong. A dead control is one that gets pressed three times
        and then reported as broken."""
        build()
        mic = find(None, "mc-llm-chat-voice-mic")
        assert mic.__dict__.get("interactive") is not False
        assert mic.__dict__.get("visible") is not False

    def test_voice_is_one_of_the_panel_s_own_surfaces(self):
        """Not a new mechanism. Everything that follows from being in SCREENS is
        what a flyout needs: no room when closed, closed when another opens, and
        put away by the menu button with the rest."""
        assert mc_voice_ui.SCREEN in mc_llm_chat_panel.SCREENS
        answered = mc_llm_chat_panel._screens("voice")
        shown = [update.get("visible")
                 for update in answered[1:1 + len(mc_llm_chat_panel.SCREENS)]]
        assert shown.count(True) == 1
        assert shown[mc_llm_chat_panel.SCREENS.index("voice")] is True

    def test_the_flyout_has_both_switches(self):
        build()
        assert find(None, "mc-llm-chat-voice-auto-send") is not None
        assert find(None, "mc-llm-chat-voice-auto-speak") is not None

    def test_the_switches_listen_to_the_tap_and_not_to_the_refill(self):
        """``change`` also fires when the server puts a value in, and opening
        the flyout puts both stored values in -- so listening to it would write
        the settings file every time somebody looked at the menu."""
        build()
        for elem_id in ("mc-llm-chat-voice-auto-send", "mc-llm-chat-voice-auto-speak"):
            kinds = {kind for kind, _kwargs in callbacks(find(None, elem_id))}
            assert kinds == {"input"}, f"{elem_id} listens to {sorted(kinds)}"

    def test_the_page_token_reaches_the_page_and_the_reply_token_starts_empty(self):
        build()
        key = find(None, "mc-llm-chat-voice-key")
        token = find(None, "mc-llm-chat-voice-token")
        assert key.value == mc_voice_api.session_token()
        assert token.value == ""
        assert token.__dict__.get("visible") is False


class TestSuccessOnlySpeech:
    """R2-1, checked against the wiring rather than against a docstring."""

    def test_the_marker_is_attached_with_success_and_never_with_then(self):
        """The whole distinction. ``then`` runs after a run that raised."""
        import conftest

        recorded = []
        original = conftest._Dependency._chain

        def watch(self, kind, kwargs):
            recorded.append((kind, kwargs.get("fn")))
            return original(self, kind, kwargs)

        conftest._Dependency._chain = watch
        try:
            build()
        finally:
            conftest._Dependency._chain = original

        markers = [(kind, fn) for kind, fn in recorded
                   if getattr(fn, "__name__", "") == "marker"]
        assert len(markers) == 6, (
            f"expected the speech marker on all six reply paths, found {len(markers)}")
        assert all(kind == "success" for kind, _fn in markers), (
            "the speech marker is attached with .then(), which runs after a run that "
            "failed — a failed generation would be read aloud")

    def test_the_six_paths_share_one_marker_object(self):
        """Section 49: the registration must be structurally shared, so fixing
        one path cannot leave another silently unspoken."""
        import conftest

        recorded = []
        original = conftest._Dependency._chain

        def watch(self, kind, kwargs):
            recorded.append(kwargs.get("fn"))
            return original(self, kind, kwargs)

        conftest._Dependency._chain = watch
        try:
            build()
        finally:
            conftest._Dependency._chain = original

        markers = {id(fn) for fn in recorded if getattr(fn, "__name__", "") == "marker"}
        assert len(markers) == 1, "the six paths were wired with six different handlers"


class TestWhatTheRunLeftBehind:
    def test_a_completed_reply_is_recorded_and_consumed_once(self):
        mc_llm_chat_panel._begin_run()
        mc_llm_chat_panel._completed_reply("the reply that finished")
        assert mc_llm_chat_panel.take_completed_reply() == "the reply that finished"
        assert mc_llm_chat_panel.take_completed_reply() == ""

    def test_a_new_run_clears_what_the_last_one_left(self):
        """The mechanism that makes a failed run unable to inherit the previous
        run's answer, which is half of what makes speech success-only."""
        mc_llm_chat_panel._completed_reply("an older reply")
        mc_llm_chat_panel._begin_run()
        assert mc_llm_chat_panel.take_completed_reply() == ""

    def test_only_the_completed_branch_records_anything(self, store, monkeypatch):
        """Driven through the real streaming handler: a run that is Stopped and
        a run that fails both save the text they had and neither becomes
        something to speak."""
        import mc_llm_sessions as sessions
        from prompt_master.chat.characters import Character, CharacterStore
        from prompt_master.chat.history import ASSISTANT, ChatStore, USER

        CharacterStore(store / "characters").save(Character(name="Ada", context="c"))
        chats = ChatStore(store / "chats")

        def run(events):
            conversation = chats.new("Ada")
            conversation.append(USER, "ask")
            conversation.append(ASSISTANT, "")
            chats.save(conversation)
            monkeypatch.setattr(sessions, "conversation",
                                lambda request, cancel: iter(events))
            list(mc_llm_chat_panel._stream("Ada", conversation,
                                           len(conversation.messages) - 1,
                                           0.7, 0.9, 256, -1))
            return mc_llm_chat_panel.take_completed_reply()

        chunk = sessions.Event(sessions.CHUNK, "half a reply")

        assert run([chunk, sessions.Event(sessions.DONE, "a whole reply")])
        assert run([chunk, sessions.Event(sessions.CANCELLED, "")]) == "", (
            "a run the reader Stopped produced something to read aloud")
        assert run([chunk, sessions.Event(sessions.FAILED, "the server died")]) == "", (
            "a failed run produced something to read aloud")


class TestTheMarker:
    def test_it_creates_a_target_when_everything_is_right(self, installed, host,
                                                          monkeypatch):
        monkeypatch.setattr(mc_voice_state, "auto_speak", lambda: True)
        marker = mc_voice_ui.speech_marker(lambda: "the reply that completed")
        token = marker()
        assert token
        assert mc_voice_api.take_reply(token)["text"] == "the reply that completed"

    def test_it_creates_nothing_when_the_run_left_nothing(self, installed, monkeypatch):
        monkeypatch.setattr(mc_voice_state, "auto_speak", lambda: True)
        assert mc_voice_ui.speech_marker(lambda: "")() == ""
        assert mc_voice_ui.speech_marker(lambda: "   ")() == ""

    def test_it_creates_nothing_when_the_switch_is_off(self, installed, monkeypatch):
        monkeypatch.setattr(mc_voice_state, "auto_speak", lambda: False)
        assert mc_voice_ui.speech_marker(lambda: "a whole reply")() == ""

    def test_it_creates_nothing_when_the_voice_is_not_installed(self, monkeypatch):
        """Section 44: an attempt to speak without TTS fails visibly and does
        not alter the setting, so intent can be chosen before the download
        finishes."""
        monkeypatch.setattr(mc_voice_state, "auto_speak", lambda: True)
        missing = mc_voice_models.Status(True, True, False, "i", "i", "no", True)
        monkeypatch.setattr(mc_voice_models, "status", lambda: missing)
        assert mc_voice_ui.speech_marker(lambda: "a whole reply")() == ""

    def test_a_broken_voice_stack_cannot_break_a_finished_reply(self, monkeypatch):
        """I-8 and section 64: TTS failing must not cancel a reply that has
        already arrived."""
        def explode():
            raise RuntimeError("everything is on fire")

        assert mc_voice_ui.speech_marker(explode)() == ""

    def test_each_completed_reply_gets_its_own_token(self, installed, monkeypatch):
        monkeypatch.setattr(mc_voice_state, "auto_speak", lambda: True)
        marker = mc_voice_ui.speech_marker(lambda: "the same words every time")
        assert marker() != marker()


class TestTheFlyout:
    def test_it_says_what_is_missing(self, monkeypatch):
        missing = mc_voice_models.Status(False, False, False, "n", "n", "n", True)
        monkeypatch.setattr(mc_voice_models, "status", lambda: missing)
        line = mc_voice_ui.readiness_notice()
        assert "Settings" in line
        assert "warn" in line

    def test_it_says_ready_when_it_is(self, installed):
        assert "Ready." in mc_voice_ui.readiness_notice()

    def test_it_names_the_half_that_is_missing(self, monkeypatch):
        half = mc_voice_models.Status(True, True, False, "i", "i", "n", True)
        monkeypatch.setattr(mc_voice_models, "status", lambda: half)
        line = mc_voice_ui.readiness_notice()
        assert "text to speech" in line
        assert "speech to text" not in line

    def test_it_opens_on_what_is_stored_rather_than_on_what_it_last_drew(
            self, host, installed):
        """Settings may have changed either switch since the panel was built,
        and a flyout showing a stale value is a flyout that turns a setting off
        by being tapped."""
        host.shared.opts.set(mc_voice_state.OPT_AUTO_SPEAK, True)
        answered = mc_voice_ui.open_sheet(mc_llm_chat_panel._screens)
        assert answered[-1].get("value") is True
        assert answered[-2].get("value") is False

    def test_a_switch_is_written_through_immediately(self, host):
        """Section 43. Somebody who turns on "speak replies" while talking to a
        character expects the next reply to be spoken, not to be told to visit
        Settings and press Apply."""
        saved = []
        host.shared.opts.save = lambda *args, **kwargs: saved.append(True)

        mc_voice_ui.set_auto_speak(True)
        assert mc_voice_state.auto_speak() is True
        assert saved, "the option was set but never written to the config file"

        mc_voice_ui.set_auto_speak(False)
        assert mc_voice_state.auto_speak() is False

    def test_a_write_the_host_refuses_snaps_the_box_back(self, host, monkeypatch):
        """Answered from the store rather than echoed, so a failed write shows
        rather than lies."""
        def refuse(name, value):
            raise RuntimeError("no")

        monkeypatch.setattr(host.shared.opts, "set", refuse)
        assert mc_voice_ui.set_auto_send(True).get("value") is False


class TestTheSettingsSection:
    def test_the_two_switches_are_registered_and_default_off(self, host):
        import model_chain  # noqa: F401  (registers the section on import)

        for name in mc_voice_state.OPTIONS:
            option = host.shared.options_templates[name]
            assert option.default is False
            assert option.section == ("model_chain_voice", "Voice Chat")

    def test_the_section_is_its_own_and_not_a_corner_of_model_chain(self, host):
        import model_chain  # noqa: F401

        sections = {host.shared.options_templates[name].section
                    for name in mc_voice_state.OPTIONS}
        assert sections == {("model_chain_voice", "Voice Chat")}
        assert model_chain.SETTINGS_SECTION != model_chain.VOICE_SECTION

    def test_every_registered_voice_option_is_one_this_module_reads(self, host):
        import model_chain  # noqa: F401

        registered = {name for name in host.shared.options_templates
                      if name.startswith("model_chain_voice_")
                      or name == "model_chain_tts_engine"}
        import mc_voice_clone
        import mc_voice_engines
        import mc_voice_models
        import mc_voice_paths
        import mc_voice_pipeline
        import mc_voice_profile
        import mc_voice_registry
        import mc_voice_sopro
        import mc_voice_device
        import mc_voice_sopro_profile

        known = set(mc_voice_state.OPTIONS) | set(mc_voice_models.OPTIONS.values()) | {
            mc_voice_paths.OPT_ROOT,
            mc_voice_registry.OPT_VOICE,
            mc_voice_registry.OPT_TEST_TEXT,
            mc_voice_clone.OPT_ROOT,
            mc_voice_profile.OPT_SPEED,
            mc_voice_profile.OPT_PITCH,
            mc_voice_profile.OPT_GAIN,
            mc_voice_profile.OPT_PAUSE,
            # The engine selector, which is deliberately not under this
            # feature's own prefix -- the design intent names the key.
            mc_voice_engines.OPT_ENGINE,
            # Sopro's own eight, named apart from Kokoro's four so that editing
            # one engine's delivery cannot reach the other's storage.
            mc_voice_sopro.OPT_VOICE,
            mc_voice_sopro.OPT_PRECISION,
            mc_voice_sopro.OPT_STEPS,
            mc_voice_sopro.OPT_CHUNK,
            "model_chain_voice_status",
            "model_chain_voice_voices",
            # The Voice Pipeline's three: a master and one per stage, and
            # deliberately no fourth. There is no order key here because there
            # is no order to persist -- the chain's order is structural
            # (I-VP-01), and a settings key for it is exactly the thing whose
            # absence this list can prove.
            mc_voice_pipeline.OPT_ENABLED,
            mc_voice_pipeline.OPT_DPDFNET,
            mc_voice_pipeline.OPT_LAVASR,
            # And its two execution settings, which are not switches: how many
            # cores the enhancement may use, and which device each stage runs
            # on. Both are read on the path that builds a session, and both were
            # registered here so that ``opts.set`` writes them -- an option the
            # host has never been told about is an option a write goes nowhere.
            mc_voice_pipeline.OPT_THREADS,
            mc_voice_device.OPT_DEVICE_DPDFNET,
            mc_voice_device.OPT_DEVICE_LAVASR,
        } | set(mc_voice_sopro_profile.OPTIONS.values()) | {
            mc_voice_sopro_profile.OPT_LANGUAGE,
        }
        assert registered <= known, (
            "a voice option is registered on the settings page and never read")

    def test_sopros_settings_are_not_also_rows_on_the_settings_page(self, host):
        """One control per value, which is what the second one cost.

        A host option is a component on the settings page as well as a stored
        value, and "Apply settings" writes every component on that page back into
        the store using the copy the browser was given when the page was built.
        So each of Sopro's twelve had a twin further down the page that knew
        nothing about the panel above it, and pressing Apply put the default
        voice, the delivery and the engine settings back as they were. They live
        in Sopro's own files now and are set in one place.
        """
        import model_chain  # noqa: F401
        import mc_voice_sopro
        import mc_voice_sopro_profile

        theirs = {mc_voice_sopro.OPT_VOICE, mc_voice_sopro.OPT_PRECISION,
                  mc_voice_sopro.OPT_STEPS, mc_voice_sopro.OPT_CHUNK,
                  mc_voice_sopro_profile.OPT_LANGUAGE}
        theirs |= set(mc_voice_sopro_profile.OPTIONS.values())
        clashing = theirs & set(host.shared.options_templates)
        assert not clashing, sorted(clashing)

    def test_every_button_this_module_draws_is_wired_to_something(self):
        """A button nothing listens to is dead, and looks exactly like a working
        one until somebody presses it.

        This has now happened twice. The recording-cleanup row shipped with its
        markup, its route, its module, its worker and its runtime all present and
        no click handler at all, so pressing Install did nothing whatsoever --
        no request, no error, no log line, nothing to diagnose. The engine
        selector's cards went the same way earlier for a different reason.

        Read off the source rather than off rendered markup on purpose: this has
        to cover every branch of every row, including the ones that only render
        on the engine that is not selected in this test.
        """
        import re

        markup = (paths.extension_root() / "mc_voice_ui.py").read_text(encoding="utf-8")
        script = (paths.extension_root() / "javascript" / "voice_chat.js").read_text(
            encoding="utf-8")

        attributes = set(re.findall(r"<button\b[^>]*?(data-mc-voice-[a-z-]+)", markup))
        assert len(attributes) > 15, "the scan found almost no buttons; it has stopped working"
        # The bracket matters. Looking for the bare name would find it
        # inside a longer one, which is how the first version of this
        # test passed against a build that had been broken on purpose.
        unwired = sorted(name for name in attributes
                         if ("[" + name) not in script)
        assert not unwired, f"drawn but nothing listens: {unwired}"

    def test_the_default_voice_is_a_stable_id_and_not_a_number(self, host):
        """Section 113. The V1 manifest stored a numeric speaker and a name that
        disagreed with it. A number is an address in a voice bank that gets
        rebuilt; a stable id survives that."""
        import model_chain  # noqa: F401
        import mc_voice_registry

        option = host.shared.options_templates[mc_voice_registry.OPT_VOICE]
        assert isinstance(option.default, str)
        assert option.default.startswith("official:")

    def test_download_is_not_a_persisted_boolean(self, host):
        import model_chain  # noqa: F401

        for name in host.shared.options_templates:
            if name.startswith("model_chain_voice"):
                assert "download" not in name

    def test_the_status_row_carries_the_page_token_and_both_buttons(self):
        markup = mc_voice_ui.settings_html()
        assert mc_voice_api.session_token() in markup
        assert 'data-mc-voice-install="tts"' in markup
        # Speech to text is three qualities rather than one bundle, so its
        # Download button is per tier. The row still has to offer every one of
        # them a way in, or a tier is a card nobody can install.
        assert 'data-mc-voice-tiers="stt"' in markup
        for identifier in ("whisper-base-int8", "whisper-small-int8", "whisper-medium-int8"):
            assert f'data-mc-voice-tier="{identifier}"' in markup
            assert f'data-mc-voice-tier-install="{identifier}"' in markup
            assert f'data-mc-voice-tier-use="{identifier}"' in markup

    def test_every_tier_says_what_it_costs_before_it_is_chosen(self):
        """The choice is between a fast one that mishears and a heavy one that
        does not, and nobody can make it from three labels."""
        markup = mc_voice_ui.settings_html()
        for rank in ("Low", "Medium", "High"):
            assert f'>{rank}</span>' in markup
        assert "to download" in markup
        assert "of memory while it is loaded" in markup

    def test_the_delivery_block_offers_the_four_controls_and_no_emotion(self):
        import mc_voice_profile

        markup = mc_voice_ui.voices_html()
        for name in mc_voice_profile.FIELDS:
            assert f'data-mc-voice-slider-input="{name}"' in markup
        # Stated rather than implied. Kokoro has no emotion input, and a
        # slider for one would be a control that does nothing.
        assert "no emotion control" in markup

    def test_the_status_row_is_not_saved_to_the_config_file(self, host):
        import model_chain  # noqa: F401

        option = host.shared.options_templates.get("model_chain_voice_status")
        if option is None:
            pytest.skip("this host build has no HTML settings component")
        assert getattr(option, "do_not_save", False) is True


class TestACharactersVoiceReachesTheTurn:
    """The end of the chain the character screen starts: which voice, and which
    delivery, the reply that is about to be written will be spoken in.

    Resolved once, here, at the moment the turn opens -- for the same reason the
    voice is (section 56): what a reply sounds like is decided at its beginning,
    and a slider moved while it is speaking changes the next one.
    """

    @pytest.fixture
    def speaking(self, monkeypatch, installed):
        import mc_voice_turn

        made = {}

        def create(**values):
            made.update(values)

            class Turn:
                id = "TURN"
                base_chars = 0

                def start(self):
                    return None

            return Turn()

        monkeypatch.setattr(mc_voice_turn, "create", create)
        monkeypatch.setattr(mc_voice_state, "auto_speak", lambda: True)
        return made

    def character(self, **values):
        from prompt_master.chat.characters import Character

        return Character(name="Ada", **values)

    def test_the_characters_own_voice_is_the_one_resolved(self, speaking, monkeypatch):
        import mc_voice_registry

        asked = []

        def resolve(voice_id=""):
            asked.append(voice_id)
            return 6, {"id": voice_id or "official:af_heart"}

        monkeypatch.setattr(mc_voice_registry, "resolve", resolve)
        mc_voice_ui.begin_speech(character=self.character(voice="official:af_nicole"))
        # The registry is still spoken to in its own dialect -- the adapter
        # strips the backend before it gets there -- and what reaches the turn
        # is the backend-qualified id the shared protocol carries (T-PROTO-1).
        assert asked == ["official:af_nicole"]
        assert speaking["voice_id"] == "kokoro:official:af_nicole"
        assert speaking["engine"] == "kokoro"
        assert speaking["handle"] == 6

    def test_a_voice_that_cannot_be_resolved_is_said_once_without_a_traceback(
            self, speaking, monkeypatch, caplog):
        """"No voice has been created yet" is a state, not a fault.

        It is true on every reply until somebody creates one, and it used to
        raise into the catch-all -- so a user whose engine had no voice got a
        full traceback per assistant turn, at WARNING, burying the failures that
        warning exists to surface. One throttled sentence says the same thing.
        """
        import logging

        import mc_voice_registry

        def refuse(voice_id=""):
            raise mc_voice_registry.RegistryError("No voice has been created yet.")

        monkeypatch.setattr(mc_voice_registry, "resolve", refuse)
        mc_voice_ui._quiet.clear()
        with caplog.at_level(logging.INFO, logger="model_chain"):
            assert mc_voice_ui.begin_speech(character=self.character()) is None

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING], \
            "an ordinary state was logged as a warning with a traceback"
        said = [r.getMessage() for r in caplog.records]
        assert any("No voice has been created yet." in line for line in said), said

    def test_a_fault_still_reads_like_a_fault(self, speaking, monkeypatch, caplog):
        """The other half of the distinction, in the branch next door.

        A voice bank that is broken raises something the adapter never declared,
        and that must keep its warning and its traceback -- quieting it would be
        trading one unreadable log for another.
        """
        import logging

        import mc_voice_registry

        def collapse(voice_id=""):
            raise RuntimeError("no bank")

        monkeypatch.setattr(mc_voice_registry, "resolve", collapse)
        mc_voice_ui._quiet.clear()
        with caplog.at_level(logging.INFO, logger="model_chain"):
            assert mc_voice_ui.begin_speech(character=self.character()) is None

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "a genuine fault was quieted"
        assert any(r.exc_info for r in warnings), "the traceback was lost"

    def test_that_sentence_is_not_repeated_on_every_reply(self, speaking, monkeypatch,
                                                          caplog):
        import logging

        import mc_voice_registry

        monkeypatch.setattr(mc_voice_registry, "resolve",
                            lambda voice_id="": (_ for _ in ()).throw(
                                mc_voice_registry.RegistryError(
                                    "No voice has been created yet.")))
        mc_voice_ui._quiet.clear()
        with caplog.at_level(logging.INFO, logger="model_chain"):
            for _ in range(4):
                mc_voice_ui.begin_speech(character=self.character())

        said = [r.getMessage() for r in caplog.records
                if "has been created yet" in r.getMessage()]
        assert len(said) == 1, said

    def test_a_character_with_no_voice_asks_for_the_default(self, speaking, monkeypatch):
        import mc_voice_registry

        asked = []
        monkeypatch.setattr(mc_voice_registry, "resolve",
                            lambda voice_id="": (asked.append(voice_id)
                                                 or (3, {"id": "official:af_heart"})))
        mc_voice_ui.begin_speech(character=self.character())
        assert asked == [""]

    def test_the_characters_delivery_is_frozen_onto_the_turn(self, speaking, monkeypatch):
        import mc_voice_profile
        import mc_voice_registry

        monkeypatch.setattr(mc_voice_registry, "resolve",
                            lambda voice_id="": (3, {"id": "official:af_heart"}))
        mc_voice_profile.remember({"speed": 1.0, "pitch": 0.0})
        mc_voice_ui.begin_speech(character=self.character(voice_pitch=-3.0))
        assert speaking["profile"]["pitch"] == -3.0
        assert speaking["profile"]["speed"] == 1.0

    def test_a_character_with_none_follows_the_default_voices_delivery(self, speaking,
                                                                       monkeypatch):
        import mc_voice_profile
        import mc_voice_registry

        monkeypatch.setattr(mc_voice_registry, "resolve",
                            lambda voice_id="": (3, {"id": "official:af_heart"}))
        mc_voice_profile.remember({"speed": 0.85})
        mc_voice_ui.begin_speech(character=self.character())
        assert speaking["profile"]["speed"] == 0.85

    def test_a_character_object_from_an_older_build_is_not_a_silent_reply(self, speaking,
                                                                          monkeypatch):
        """This runs inside the generator that produces a reply. Anything odd
        about the character is the default voice, never an exception."""
        import mc_voice_registry

        monkeypatch.setattr(mc_voice_registry, "resolve",
                            lambda voice_id="": (3, {"id": "official:af_heart"}))

        class Older:
            name = "Ada"

        assert mc_voice_ui.voice_of(Older()) == ""
        assert mc_voice_ui.profile_of(Older()) == {}
        assert mc_voice_ui.begin_speech(character=Older()) is not None


class TestThePocketSurface:
    """The PocketTTS settings surface, and the fall-through it replaced.

    Selecting PocketTTS used to draw Sopro's entire install row -- its two
    manual sections, its engine settings, its validation button -- and then
    Kokoro's voice list with the Storytime cloning panel under it, from two
    branches that each read "the other engine" and meant it. Every test here is
    about something the registry now has to keep true rather than about
    decoration.
    """

    @staticmethod
    def _ready(**changed):
        """A PocketTTS installation that is complete, for a panel to describe.

        The test machine is not one this build ships a PocketTTS closure for, so
        every readiness is false and the panel correctly says so. That is the
        right first frame and the wrong fixture for asserting what an installed
        panel offers, which is what this stands in for.
        """
        import mc_voice_pocket as pocket

        found = pocket.Status(
            platform_supported=True, runtime_ready=True, speech_model_ready=True,
            official_voices_ready=True, cloning_ready=True, label="PocketTTS English",
            fingerprint="pkt-1234", model_id="english",
            runtime_message="Installed — PocketTTS 3.0.2, Torch 2.6.0, CPU only.",
            model_message="Installed — PocketTTS English.",
            cloning_message="Installed — you can clone a voice from a recording.")
        for name, value in changed.items():
            setattr(found, name, value)
        return found

    def test_the_panel_reports_five_readinesses_and_not_one_boolean(self):
        """Section 24. A machine whose speech works and whose Clone button does
        not is not "half installed", and one line saying "Not installed" is the
        panel hiding the fact that decides what to do next."""
        import mc_voice_engines as engines

        engines.select("pocket")
        markup = mc_voice_ui.settings_html()
        for hook in ("data-mc-voice-pocket-runtime", "data-mc-voice-pocket-model",
                     "data-mc-voice-pocket-voices", "data-mc-voice-pocket-cloning"):
            assert hook in markup, hook
        assert 'data-mc-voice-status="pocket"' in markup

    def test_an_unpinned_build_says_so_and_says_the_folder_is_the_way_in(self, monkeypatch):
        """A closure this repository makes no claim about is one it will not
        fetch, and being told that beats pressing a button that refuses."""
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        monkeypatch.setattr(pocket, "status", lambda: self._ready(runtime_ready=False))
        monkeypatch.setattr(pocket, "pinned", lambda: False)
        engines.select("pocket")
        markup = mc_voice_ui.settings_html()
        assert "has not pinned a PocketTTS runtime closure yet" in markup
        assert "folder you filled yourself" in markup

    def test_a_pinned_build_does_not_repeat_the_warning(self, monkeypatch):
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        monkeypatch.setattr(pocket, "status", lambda: self._ready())
        monkeypatch.setattr(pocket, "pinned", lambda: True)
        engines.select("pocket")
        assert "has not pinned a PocketTTS runtime closure yet" \
            not in mc_voice_ui.settings_html()

    def test_each_installable_part_has_a_folder_of_its_own_that_says_which(self,
                                                                          monkeypatch):
        """Four collapsed sections all reading "Or install from files you
        download yourself" would say nothing about which installs what, and the
        four folder boxes would all be found by one selector."""
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        monkeypatch.setattr(pocket, "sources", lambda part="runtime": [
            {"url": f"https://example.invalid/{part}.bin", "filename": f"{part}.bin",
             "save_as": f"{part}.bin"}])
        engines.select("pocket")
        markup = mc_voice_ui.settings_html()
        for part in ("runtime", "model", "voices", "cloning"):
            assert f'data-mc-voice-local="pocket-{part}"' in markup, part
            assert f'data-mc-voice-folder="pocket-{part}"' in markup, part
        for title in ("Or install the PyTorch runtime from files you download yourself",
                      "Or install the model artifacts from files you download yourself",
                      "Or install the official voices from files you download yourself",
                      "Or install the voice-cloning weights from files you download "
                      "yourself"):
            assert f"<summary>{title}</summary>" in markup, title

    def test_the_engine_settings_are_precision_and_quality_and_nothing_per_character(self):
        """I-PKT-23. These change compute, memory and which prepared voice states
        are still valid for the whole worker."""
        import mc_voice_engines as engines

        engines.select("pocket")
        markup = mc_voice_ui.settings_html()
        assert 'data-mc-voice-pocket-setting="precision"' in markup
        assert 'data-mc-voice-pocket-setting="steps"' in markup
        assert "<option value=\"full\"" in markup and "<option value=\"int8\"" in markup

    def test_neither_precision_claims_a_speed_nobody_has_measured(self):
        """GATE P-5, I-PKT-26. Upstream reports a faster INT8 path and it may
        well hold; the identical-looking setting on the other streaming engine
        measured 40% the other way on the first machine anybody tried."""
        import mc_voice_engines as engines

        engines.select("pocket")
        markup = mc_voice_ui.settings_html()
        assert "Neither of these claims to be the faster one" in markup

    def test_the_step_choices_keep_their_numbers_beside_their_names(self):
        """A label is a shorthand and this setting is a compute trade, so hiding
        the value would hide the thing being traded."""
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        engines.select("pocket")
        markup = mc_voice_ui.settings_html()
        for value in pocket.STEP_CHOICES:
            plural = "" if value == 1 else "s"
            assert f"{pocket.STEP_LABELS[value]} — {value} step{plural}" in markup

    def test_there_is_a_thread_note_and_no_thread_control(self):
        """Section 16.4. Released PocketTTS exposes nothing supported to set, so
        a slider here would be a control that lies about what it did."""
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        engines.select("pocket")
        markup = mc_voice_ui.settings_html()
        assert pocket.thread_policy() in markup
        assert 'data-mc-voice-pocket-setting="threads"' not in markup
        assert "data-mc-voice-sopro-setting" not in markup

    def test_the_model_row_appears_only_when_there_is_more_than_one_to_choose(self,
                                                                             monkeypatch):
        """The setting is persisted from day one even so: "PocketTTS means
        English" must not end up encoded in stable storage (section 16.3)."""
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        engines.select("pocket")
        one = pocket.engine_settings()
        assert len(one["model_choices"]) == 1
        assert 'data-mc-voice-pocket-setting="model_id"' not in \
            mc_voice_ui.settings_html()

        two = dict(one, model_choices=[{"id": "english", "label": "PocketTTS English"},
                                       {"id": "french", "label": "PocketTTS French"}])
        monkeypatch.setattr(pocket, "engine_settings", lambda: two)
        assert 'data-mc-voice-pocket-setting="model_id"' in mc_voice_ui.settings_html()

    def test_there_is_no_validation_sweep_because_there_is_no_lever_to_sweep(self):
        import mc_voice_engines as engines

        engines.select("pocket")
        markup = mc_voice_ui.settings_html()
        assert "data-mc-voice-sopro-validate" not in markup
        assert "Run validation" not in markup

    def test_the_voices_row_offers_pockets_five_delivery_controls(self):
        import mc_voice_engines as engines
        import mc_voice_pocket_profile

        engines.select("pocket")
        markup = mc_voice_ui.voices_html()
        for name in mc_voice_pocket_profile.FIELDS:
            assert f'data-mc-voice-slider-input="{name}"' in markup, name
        assert 'data-mc-voice-slider-input="temperature"' in markup

    def test_the_delivery_note_says_which_four_are_not_the_models(self):
        """Section 14. The reviewed PocketTTS generation call has no rate, pitch,
        gain or emotion argument at all, so four of these five are Voice Chat's
        and the surface has to say so rather than let somebody assume."""
        import mc_voice_engines as engines

        engines.select("pocket")
        markup = mc_voice_ui.voices_html()
        assert "Speed, Pitch, Volume and Pause are Voice Chat" in markup
        assert "sampling temperature" in markup
        assert "not an emotion, warmth, energy or identity control" in markup
        # And not the paragraph written for the engine that does take a speed.
        assert "no emotion control" not in markup
        assert "Kokoro exposes one of these itself" not in markup

    def test_a_gated_cloning_install_explains_itself_rather_than_offering_a_button(self):
        """Section 30. A capability the engine has is not a readiness this
        machine has, and a Create button that cannot work is worse than none."""
        import mc_voice_engines as engines

        engines.select("pocket")
        markup = mc_voice_ui.voices_html()
        assert "data-mc-voice-pocket-clone" in markup
        assert "gated" in markup
        assert "data-mc-voice-pocket-create" not in markup
        assert "data-mc-voice-pocket-form" not in markup

    def test_the_clone_workspace_appears_once_the_gated_weights_are_installed(self,
                                                                              monkeypatch):
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        monkeypatch.setattr(pocket, "status", lambda: self._ready())
        engines.select("pocket")
        markup = mc_voice_ui.voices_html()
        for hook in ("data-mc-voice-pocket-form", "data-mc-voice-pocket-name",
                     "data-mc-voice-pocket-file", "data-mc-voice-pocket-record",
                     "data-mc-voice-pocket-create", "data-mc-voice-pocket-preview",
                     "data-mc-voice-pocket-preview-save",
                     "data-mc-voice-pocket-preview-discard",
                     "data-mc-voice-trim", "data-mc-voice-wave"):
            assert hook in markup, hook
        # The reference envelope comes from the adapter, because the ideal
        # length is a release measurement and a number baked into a page is a
        # number that goes stale (section 26.1).
        #
        # The *ceiling* rather than the ideal, because that is the length this
        # button picks and the length the selection opens at: conditioning is
        # built from whatever it is given, and more of it costs nothing at
        # speaking time. The page rewrites this label from the engine's own
        # answer as soon as it has one; what is rendered here is what somebody
        # sees before the first poll lands, and it should not disagree.
        assert f"Pick {int(pocket.MAX_REFERENCE_SECONDS)} s for me" in markup
        assert f"{int(pocket.MIN_REFERENCE_SECONDS)} to " \
               f"{int(pocket.MAX_REFERENCE_SECONDS)} seconds" in markup
        # No language hint: the PocketTTS model is the language and it is
        # engine-global, so a per-voice selector would be a control that either
        # did nothing or disagreed with Engine settings.
        assert "data-mc-voice-pocket-language" not in markup

    def test_the_pocket_voice_row_carries_no_lab_no_starter_voices_and_no_storytime(self,
                                                                                    monkeypatch):
        import mc_voice_engines as engines
        import mc_voice_pocket as pocket

        monkeypatch.setattr(pocket, "status", lambda: self._ready())
        engines.select("pocket")
        markup = mc_voice_ui.voices_html()
        for forbidden in ("data-mc-voice-lab", "Style control", "Conditioning Blend",
                          "data-mc-voice-starter-make", "Starter voices",
                          "data-mc-voice-cloning", "Storytime",
                          "data-mc-voice-clone-start"):
            assert forbidden not in markup, forbidden

    def test_the_first_frame_carries_no_voice_and_no_rename_or_delete(self):
        """Section 56 and section 10 together. The list is fetched, so no
        engine-native address is in the document; and an official voice offering
        a Rename that always failed would be worse than no Rename at all."""
        import mc_voice_engines as engines

        engines.select("pocket")
        markup = mc_voice_ui.voices_html()
        assert "data-mc-voice-list" in markup and "data-mc-voice-warnings" in markup
        assert "data-mc-voice-action" not in markup
        assert "pocket:official:" not in markup


class TestTheRendererRegistry:
    """I-10, applied to markup: a miss draws nothing rather than another engine."""

    def test_every_renderer_names_an_engine_this_build_actually_has(self):
        import mc_voice_engines as engines

        for registry in (mc_voice_ui._ENGINE_PANELS, mc_voice_ui._ENGINE_VOICES,
                         mc_voice_ui._DELIVERY_NOTES):
            # Every one, and only those. A subset would be an engine the
            # selector offers and the page cannot draw; a superset would be a
            # renderer for something that is not an engine any more. Both are
            # gaps this file is the only place that could have (I-PKT-30).
            assert set(registry) == set(engines.ENGINES)

    def test_an_engine_with_no_renderer_draws_nothing_at_all(self):
        """Nothing is a surface somebody notices and reports. Another engine's
        controls is a surface nobody notices until a request built from them is
        refused."""
        assert mc_voice_ui._panel(mc_voice_ui._ENGINE_PANELS, "not-an-engine",
                                  "settings surface") == ""
        assert mc_voice_ui._panel(mc_voice_ui._ENGINE_VOICES, "", "voice library") == ""
        assert mc_voice_ui._delivery_note("not-an-engine") == ""

    def test_each_engine_gets_its_own_delivery_paragraph(self):
        """Section 37. The same labels mean different things on three engines,
        and the paragraph that told a PocketTTS user Kokoro exposes speed was
        wrong about every sentence in it."""
        notes = {name: mc_voice_ui._delivery_note(name)
                 for name in ("kokoro", "sopro", "pocket")}
        assert len(set(notes.values())) == 3
        assert "Kokoro exposes one of these itself" in notes["kokoro"]
        assert "Sopro has no speaking-rate input of its own" in notes["sopro"]
        assert "are Voice Chat" in notes["pocket"]


class TestTheCharacterEditorFollowsTheEngine:
    """Section 7. The editor is drawn from the active engine's own profile."""

    def test_it_draws_pockets_five_sliders_including_variation(self):
        import mc_voice_engines as engines
        import mc_voice_pocket_profile as profile

        engines.select("pocket")
        build()
        for name in profile.FIELDS:
            slider = find(None, f"mc-llm-chat-character-voice-{name}")
            assert slider is not None, name
            assert slider.minimum == profile.CONTROLS[name]["minimum"]
            assert slider.maximum == profile.CONTROLS[name]["maximum"]
        assert find(None, "mc-llm-chat-character-voice-temperature") is not None

    def test_it_draws_no_engine_setting_on_any_engine(self):
        """I-PKT-23. Precision, decode steps and the model are global to the
        worker and changing one stops it, so a character carrying them would be
        a character whose turn to speak restarted a subprocess."""
        import mc_voice_engines as engines

        engines.select("pocket")
        build()
        for name in ("precision", "steps", "threads", "model", "sampler_steps"):
            assert find(None, f"mc-llm-chat-character-voice-{name}") is None, name

    def test_an_unset_field_is_none_rather_than_todays_default(self):
        """I-4. A character that follows Settings has to keep following it when
        Settings changes, so unchecked is every field ``None``."""
        import mc_voice_engines as engines
        import mc_voice_pocket_profile as profile

        engines.select("pocket")
        assert mc_voice_ui.character_profile(False, [1.5] * len(profile.FIELDS)) == \
            {name: None for name in profile.FIELDS}

    def test_a_character_with_no_pocket_voice_opens_with_none_selected(self):
        """It does not translate a Kokoro pitch into a PocketTTS one, and it
        does not offer a Kokoro voice on a PocketTTS page."""
        import mc_voice_engines as engines
        import mc_voice_pocket_profile as profile
        from prompt_master.chat.characters import Character

        engines.select("pocket")
        found = mc_voice_ui.character_state(Character(name="Ada", voice="official:af_heart"))
        assert found["engine"] == "pocket"
        assert found["voice"] == ""
        assert len(found["values"]) == len(profile.FIELDS)


class TestACharactersDeliveryControlsAreTheEnginesOwn:
    """The contract ``javascript/voice_chat.js`` reads the field list through.

    The script used to carry the list itself — the four Kokoro has — so a
    PocketTTS character's fifth control, Variation, was saved, was used when
    the character actually spoke, and was dropped from its own Test button. It
    now reads every slider inside the delivery group and takes each field name
    off the element id, which only works while these two agree.
    """

    def test_the_panel_holds_one_slider_per_field_of_the_active_engine(self, host):
        import mc_llm_ui as llm_ui
        import mc_voice_engines as engines
        import mc_voice_ui as voice_ui

        for engine in engines.ENGINES:
            engines.select(engine)
            fields = list(engines.profiles(engine).FIELDS)
            assert voice_ui._field_names() == tuple(fields), engine
            drawn = [control["name"] for control in voice_ui.delivery_controls()]
            assert drawn == fields, engine
            # And every one of them is reachable by the prefix the script slices
            # the name back out of.
            # Built the way the script builds it: the base id plus a dash.
            # ``ident`` strips a trailing dash from its own parts, so asking it
            # for the prefix directly gives a different string.
            prefix = llm_ui.ident("chat", "character-voice") + "-"
            for name in fields:
                found = llm_ui.ident("chat", f"character-voice-{name}")
                assert found.startswith(prefix), (engine, name)
                assert found[len(prefix):] == name, (engine, name)

    def test_the_group_the_script_queries_is_the_one_the_sliders_are_in(self):
        """The script asks the group for its sliders, so the group's own id must
        not be one of the answers — a field called "delivery" would be read as a
        slider named after the group."""
        import mc_llm_ui as llm_ui
        import mc_voice_engines as engines

        group = llm_ui.ident("chat", "character-voice-delivery")
        for engine in engines.ENGINES:
            assert "delivery" not in engines.profiles(engine).FIELDS, engine
        assert group.endswith("-delivery")

    def test_the_script_does_not_carry_its_own_copy_of_the_list(self):
        """The regression itself: a list written here goes stale the moment an
        engine has a field the author of the list did not have."""
        from pathlib import Path

        source = Path("javascript/voice_chat.js").read_text(encoding="utf-8")
        assert '["speed", "pitch", "gain", "pause"].forEach' not in source


class TestAFailedInstallSaysSoWhereItHappened:
    """The refusal reaches the panel, not only the log.

    The defect this covers: ``/pipeline/install`` answers ``{"ok": true}`` the
    instant the thread starts, the real outcome lands in the progress map
    minutes later, and the overview repainted two state chips and nothing else.
    ``mc_voice_models._claim`` puts the reason there precisely so a surface can
    draw it -- "the button went back to how it was" is not an answer to "what
    happened".
    """

    def _progress(self, monkeypatch, entry):
        import mc_voice_pipeline as pipeline

        monkeypatch.setattr(mc_voice_models, "progress",
                            lambda: {pipeline.KIND: entry})

    def test_the_reason_is_drawn_on_the_component_it_belongs_to(self, monkeypatch):
        self._progress(monkeypatch, {"running": False, "failed": True, "model": "dpdfnet",
                                     "text": "DPDFNet did not run on this machine.",
                                     "fraction": 0.0})
        drawn = mc_voice_ui._pipeline_status_line("dpdfnet")
        assert "did not run on this machine" in drawn
        assert "hidden" not in drawn
        assert "mc-voice-failed" in drawn

    def test_another_components_failure_is_not_drawn_here(self, monkeypatch):
        """One install runs at a time, so an unscoped line would blame the wrong thing."""
        self._progress(monkeypatch, {"running": False, "failed": True, "model": "runtime",
                                     "text": "The runtime could not be built.",
                                     "fraction": 0.0})
        drawn = mc_voice_ui._pipeline_status_line("dpdfnet")
        assert "could not be built" not in drawn
        assert "hidden" in drawn

    def test_a_clean_install_leaves_no_line_behind(self, monkeypatch):
        self._progress(monkeypatch, {"running": False, "failed": False, "model": "dpdfnet",
                                     "text": "Installed.", "fraction": 1.0})
        assert "hidden" in mc_voice_ui._pipeline_status_line("dpdfnet")

    def test_what_it_is_doing_is_drawn_while_it_runs(self, monkeypatch):
        self._progress(monkeypatch, {"running": True, "failed": False, "model": "dpdfnet",
                                     "text": "Installing the runtime first\u2026",
                                     "fraction": 0.1})
        drawn = mc_voice_ui._pipeline_status_line("dpdfnet")
        assert "Installing the runtime first" in drawn and "hidden" not in drawn

    def test_the_markup_escapes_what_it_is_given(self, monkeypatch):
        """A reason is an exception string, and this one is written into HTML."""
        self._progress(monkeypatch, {"running": False, "failed": True, "model": "dpdfnet",
                                     "text": "<script>alert(1)</script>", "fraction": 0.0})
        drawn = mc_voice_ui._pipeline_status_line("dpdfnet")
        assert "<script>" not in drawn and "&lt;script&gt;" in drawn

    def test_a_running_install_says_so_in_the_row(self, monkeypatch):
        """The state that existed, had a label, and was never once produced.

        ``pipeline.status()`` is a filesystem read and cannot see an install in
        flight, so every row said "Not installed" throughout one. The browser
        stops polling as soon as no row is busy, so that single answer was also
        the last one it asked for.
        """
        self._progress(monkeypatch, {"running": True, "failed": False,
                                     "model": "dpdfnet", "text": "Working\u2026",
                                     "fraction": 0.1})
        rows = {row["id"]: row for row in mc_voice_ui.component_rows()}
        assert rows["voice-pipeline-dpdfnet"]["install_state"] == "installing"
        # Scoped, or the overview would report three installs where there is one.
        assert rows["voice-pipeline-runtime"]["install_state"] != "installing"
        assert rows["voice-pipeline-lavasr"]["install_state"] != "installing"

    def test_an_idle_overview_reports_no_install(self, monkeypatch):
        """Guard, so the test above is not passing on a row that always says it."""
        monkeypatch.setattr(mc_voice_models, "progress", lambda: {})
        rows = {row["id"]: row for row in mc_voice_ui.component_rows()}
        assert rows["voice-pipeline-dpdfnet"]["install_state"] == "not_installed"

    def test_a_stage_panel_names_the_runtime_it_needs(self):
        """Said where the user is looking, before they press anything."""
        drawn = mc_voice_ui.pipeline_stage_detail("dpdfnet")
        assert "Requires" in drawn and "Voice Pipeline runtime" in drawn

    def test_a_stage_panel_names_the_closure_that_stage_actually_needs(self):
        """LavaSR's row is about the PyTorch closure, not the ONNX one."""
        drawn = mc_voice_ui.pipeline_stage_detail("lavasr")
        assert "Voice Pipeline runtime, PyTorch build" in drawn


class TestTheCardRowSaysWhatIsTrueOfTheStageItIsUnder:
    """One note was printed for both stages and was true of only one.

    The card-numbering caveat is DirectML's: its adapter number is a DXGI index
    nothing enumerates, so the honest advice is "if the wrong card lights up,
    choose the other entry". LavaSR is pinned to its card by UUID and has no
    adapter number at all, so printing that under it would send somebody
    switching cards to fix a problem they do not have.
    """

    @staticmethod
    def _describe(monkeypatch, **overrides):
        import mc_voice_pipeline as pipeline
        found = {"component": "voice-pipeline-lavasr", "placeable": True, "reason": "",
                 "device": "gpu:GPU-1234", "provider": "CUDAExecutionProvider",
                 "adapter": 0, "effective_provider": "CUDAExecutionProvider",
                 "accelerator": "cuda", "pinned_by_uuid": True, "honoured": True,
                 "devices": [{"token": "cpu", "label": "Processor"},
                             {"token": "gpu:GPU-1234", "label": "GPU 1 — RTX 3090"}]}
        found.update(overrides)
        monkeypatch.setattr(pipeline, "devices_for", lambda stage_id: dict(found))
        monkeypatch.setattr(pipeline, "stage_available", lambda stage_id: True)
        return found

    def test_a_uuid_pinned_card_is_not_told_it_might_be_the_wrong_one(self, monkeypatch):
        self._describe(monkeypatch)
        drawn = mc_voice_ui.pipeline_stage_detail("lavasr")
        assert "pinned by its own identifier" in drawn
        assert "choose the other entry" not in drawn

    def test_a_directml_card_keeps_the_caveat_that_is_true_of_it(self, monkeypatch):
        self._describe(monkeypatch, provider="DmlExecutionProvider",
                       effective_provider="DmlExecutionProvider",
                       accelerator="cpu", pinned_by_uuid=False)
        drawn = mc_voice_ui.pipeline_stage_detail("lavasr")
        assert "choose the other entry" in drawn
        assert "pinned by its own identifier" not in drawn

    def test_a_choice_that_is_not_being_honoured_says_what_would_honour_it(
            self, monkeypatch):
        """The dropdown must not look like a control that did nothing."""
        self._describe(monkeypatch, effective_provider="CPUExecutionProvider",
                       honoured=False)
        drawn = mc_voice_ui.pipeline_stage_detail("lavasr")
        assert "until the CUDA one is installed" in drawn
        assert "the setting is already saved" in drawn

    def test_a_machine_with_no_cuda_closure_is_told_that_rather_than_to_wait(
            self, monkeypatch):
        self._describe(monkeypatch, effective_provider="CPUExecutionProvider",
                       accelerator="cpu", honoured=False)
        drawn = mc_voice_ui.pipeline_stage_detail("lavasr")
        assert "no CUDA runtime" in drawn
        assert "until the CUDA one is installed" not in drawn

    def test_an_honoured_choice_says_nothing_extra(self, monkeypatch):
        """Guard, so the three above are not passing on a note always printed."""
        self._describe(monkeypatch)
        drawn = mc_voice_ui.pipeline_stage_detail("lavasr")
        assert "no CUDA runtime" not in drawn
        assert "until the CUDA one is installed" not in drawn

