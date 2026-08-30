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
        import mc_voice_profile
        import mc_voice_registry
        import mc_voice_sopro
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
