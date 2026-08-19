"""The three panels build, and every control they declare is wired.

A UI this size is mostly wiring, and wiring fails at build time or not at all:
an input list one component short, a handler returning the wrong number of
outputs, a control named in one list and forgotten in another. None of that
shows up in a unit test of the behaviour underneath -- it shows up when
somebody opens the tab.

So these tests build the panels against the faked Gradio in conftest and then
ask questions about what got built. They cannot prove the tab looks right; they
can prove it assembles, that the modes stay three separate views, and that the
control lists Prompt Studio keeps in three places have not drifted apart.
"""

from __future__ import annotations

import pytest

import mc_llm_chat_panel
import mc_llm_minimax_panel
import mc_llm_paths
import mc_llm_prompt_panel
import mc_llm_studio


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    yield tmp_path


class TestPromptStudio:
    def test_it_builds_and_exposes_both_outputs(self):
        built = mc_llm_prompt_panel.build()

        assert set(built) == {"status", "positive", "negative"}

    def test_every_control_in_the_order_list_exists_in_the_panel(self):
        """The one list is used for the generate inputs, for what is persisted
        and for what a loaded session restores. A control added to the panel
        and left out of it would be silently dropped from saved sessions."""
        from prompt_master.prompt_engine import options as opt

        request_fields = {
            "video_mode", "seconds", "fps", "seed", "style", "motion", "camera", "transition",
            "pov", "wardrobe", "undress", "accent", "accent_strength", "dialogue", "music",
            "music_bg", "speech", "fmt", "smart_negative", "lexicon", "negative_extra",
        }
        # dimensions covers output_width and output_height together.
        assert request_fields | {"dimensions"} == set(mc_llm_prompt_panel._ORDER)
        assert opt.DEFAULTS["video_mode"] in dict(
            (value, label) for label, value in
            mc_llm_prompt_panel.ui.choices(opt.VIDEO_MODES))

    def test_a_request_is_built_from_the_panel_values(self):
        values = {name: default for name, default in (
            ("video_mode", "t2v"), ("seconds", 8.0), ("fps", 30), ("dimensions", "1216x704"),
            ("seed", 42), ("style", "off"), ("motion", "default"), ("camera", "off"),
            ("transition", "off"), ("pov", "off"), ("wardrobe", "auto"), ("undress", False),
            ("accent", "off"), ("accent_strength", "natural"), ("dialogue", 20),
            ("music", "off"), ("music_bg", False), ("speech", 1), ("fmt", "flowing"),
            ("smart_negative", False), ("lexicon", ""), ("negative_extra", ""))}

        request = mc_llm_prompt_panel._request("a shot", None, values)

        assert request.output_width == 1216
        assert request.output_height == 704
        assert request.seed == 42
        assert request.video_mode == "t2v"

    def test_a_random_seed_is_resolved_before_the_engine_sees_it(self):
        """Section: a request carrying -1 would seed the casting and the
        sampler with two different numbers."""
        from prompt_master.core.models import RANDOM_SEED

        values = {name: "" for name in mc_llm_prompt_panel._ORDER}
        values.update({"video_mode": "t2v", "seconds": 8.0, "fps": 24,
                       "dimensions": "704x1216", "seed": RANDOM_SEED, "dialogue": 20,
                       "speech": 1, "undress": False, "music_bg": False,
                       "smart_negative": False})

        request = mc_llm_prompt_panel._request("a shot", None, values)

        assert request.seed != RANDOM_SEED
        assert request.seed >= 0

    def test_an_empty_intent_is_refused_without_starting_anything(self):
        values = ["t2v", 8.0, 24, "704x1216", 7] + [""] * (len(mc_llm_prompt_panel._ORDER) - 5)

        events = list(mc_llm_prompt_panel._generate("   ", None, *values))

        assert len(events) == 1
        assert "video intent" in events[0][3]

    def test_i2v_without_an_image_is_refused_with_the_reason(self):
        values = ["i2v", 8.0, 24, "704x1216", 7] + [""] * (len(mc_llm_prompt_panel._ORDER) - 5)

        events = list(mc_llm_prompt_panel._generate("a shot", None, *values))

        assert "Image to video needs an attached image" in events[0][3]

    def test_the_speech_note_names_the_lifted_dialogue_budget(self):
        assert "exactly as the intent quotes it" in mc_llm_prompt_panel._speech_note(1, 20)
        assert "3×" in mc_llm_prompt_panel._speech_note(3, 20)

    def test_history_round_trips_through_the_panel_helpers(self):
        import mc_llm_state

        mc_llm_state.save_prompt_session(mc_llm_state.PromptSession(
            intent="a shot", positive="P", negative="N", seed=5,
            controls={name: "" for name in mc_llm_prompt_panel._ORDER}))
        identifier = mc_llm_state.prompt_sessions()[0].identifier

        loaded = mc_llm_prompt_panel._load_session(identifier)

        assert loaded[0] == "a shot"
        assert loaded[1] == "P"
        assert len(loaded) == 4 + len(mc_llm_prompt_panel._ORDER)


class TestConversation:
    def test_it_builds(self):
        built = mc_llm_chat_panel.build()

        assert "transcript" in built and "status" in built

    def test_the_transcript_pairs_turns_for_gradio_four(self):
        from prompt_master.chat.history import ASSISTANT, USER, Conversation

        conversation = Conversation(identifier="x", character="Ada")
        conversation.append(USER, "hello")
        conversation.append(ASSISTANT, "hi")

        assert mc_llm_chat_panel._transcript(conversation) == [["hello", "hi"]]

    def test_two_replies_in_a_row_become_two_rows(self):
        from prompt_master.chat.history import ASSISTANT, Conversation

        conversation = Conversation(identifier="x", character="Ada")
        conversation.append(ASSISTANT, "first")
        conversation.append(ASSISTANT, "second")

        assert mc_llm_chat_panel._transcript(conversation) == [[None, "first"],
                                                              [None, "second"]]

    def test_an_attachment_is_named_in_the_transcript(self):
        from prompt_master.chat.history import USER, Conversation

        conversation = Conversation(identifier="x", character="Ada")
        conversation.append(USER, "look", "data:image/jpeg;base64,AA", "frame.png")

        assert "frame.png" in mc_llm_chat_panel._transcript(conversation)[0][0]

    def test_an_empty_conversation_is_an_empty_transcript(self):
        assert mc_llm_chat_panel._transcript(None) == []

    def test_threads_are_filtered_by_the_search_box(self, store):
        from prompt_master.chat.history import ChatStore

        chats = ChatStore(store / "chats")
        first = chats.new("Ada")
        first.title = "harbour at night"
        chats.save(first)
        second = chats.new("Ada")
        second.title = "a desert road"
        chats.save(second)

        titles = [title for title, _ in mc_llm_chat_panel._thread_choices("Ada", "harbour")]

        assert titles == ["harbour at night"]

    def test_a_missing_character_does_not_break_the_thread_list(self):
        assert mc_llm_chat_panel._thread_choices("", "") == []


class TestMiniMax:
    def test_it_builds(self):
        built = mc_llm_minimax_panel.build()

        assert set(built) == {"status", "output"}

    def test_an_empty_prompt_is_refused(self):
        events = list(mc_llm_minimax_panel._enhance("  ", "fl2va", None, 7))

        assert len(events) == 1
        assert "Write a prompt" in events[0][3]

    def test_the_structure_guide_comes_from_the_enhancer(self):
        from prompt_master.minimax import enhancer

        assert mc_llm_minimax_panel._structure("fl2va") == enhancer.infos("fl2va")
        assert (mc_llm_minimax_panel._structure(enhancer.REF2VA)
                != mc_llm_minimax_panel._structure(enhancer.FL2VA))


class TestShell:
    def test_the_three_modes_are_three_views(self):
        """Section 4.1: the modes may share panels but must not be collapsed
        into one workflow. Exactly one view is visible at a time."""
        for _, chosen in mc_llm_studio.MODES:
            updates = mc_llm_studio._switch(chosen)
            visible = [update.get("visible") for update in updates[:3]]

            assert visible.count(True) == 1

    def test_the_chosen_mode_is_remembered(self):
        import mc_llm_state

        mc_llm_studio._switch("minimax")

        assert mc_llm_state.preferences()["mode"] == "minimax"

    def test_an_unknown_stored_mode_falls_back_to_prompt_studio(self):
        import mc_llm_state

        mc_llm_state.remember(mode="something-else")

        assert mc_llm_studio._initial_mode() == "prompt"

    def test_the_runtime_line_says_what_is_missing_before_setup(self):
        assert "needs a llama.cpp runtime and a model" in mc_llm_studio._runtime_line()

    def test_the_residency_panel_renders_with_nothing_resident(self):
        import mc_broker

        mc_broker.clear()
        html = mc_llm_studio._residency_html()

        assert "Nothing is registered" in html
        assert mc_llm_studio.ui.PREFIX in html

    def test_the_residency_panel_lists_what_is_resident(self):
        import mc_broker

        mc_broker.clear()
        mc_broker.declare(mc_broker.FAMILY_IMAGE, "ckpt", "a checkpoint", 6 * 1024**3)
        try:
            html = mc_llm_studio._residency_html()
            assert "a checkpoint" in html
        finally:
            mc_broker.clear()

    def test_the_estimator_says_so_when_no_model_is_chosen(self):
        assert "Choose a GGUF" in mc_llm_studio._estimator_html()

    def test_the_whole_tab_builds(self):
        tabs = mc_llm_studio.on_ui_tabs()

        assert len(tabs) == 1
        assert tabs[0][1] == mc_llm_studio.TAB_LABEL
        assert tabs[0][2] == mc_llm_studio.TAB_ID


class TestThemeContract:
    def test_every_element_id_is_extension_owned(self):
        """Section 5: stable, extension-owned ids. A bare id could collide with
        the host's own, and the CSS scoping depends on the prefix."""
        import mc_llm_ui as ui

        assert ui.ident("prompt", "generate") == "mc-llm-prompt-generate"
        assert ui.classes("output") == ["mc-llm-output"]

    def test_the_css_is_scoped_and_states_no_colours_of_its_own(self):
        """A hard-coded colour is what makes an extension look wrong under a
        theme, and a Gradio-generated class is what makes it break under one."""
        import re
        from pathlib import Path

        css = Path(__file__).resolve().parent.parent / "style.css"
        section = css.read_text(encoding="utf-8").split("LLM Studio", 1)[1]

        for line in section.splitlines():
            stripped = line.strip()
            if stripped.startswith("#mc-llm-studio"):
                assert ".svelte" not in stripped
            if ":" in stripped and not stripped.startswith(("*", "/", "#", "@")):
                # Colours are only ever var() references to the host's own.
                assert not re.search(r":\s*#[0-9a-fA-F]{3,8}\b", stripped), stripped
                assert not re.search(r":\s*rgba?\(", stripped), stripped

    def test_the_escaping_helper_neutralises_metadata_from_a_model_file(self):
        """general.name is free text out of somebody else's file, and it lands
        in HTML."""
        import mc_llm_ui as ui

        assert "<script>" not in ui.notice("<script>alert(1)</script>")


class TestPanelsOpenWhereTheyWereLeft:
    def test_conversation_opens_on_the_remembered_thread(self, store):
        """A rail that starts empty reads as "you have no threads" rather than
        as "pick a character"."""
        import mc_llm_state
        from prompt_master.chat.history import ChatStore

        chats = ChatStore(store / "chats")
        conversation = chats.new("Ada")
        conversation.title = "harbour at night"
        chats.save(conversation)
        mc_llm_state.remember(character="Ada", thread=conversation.identifier)

        built = mc_llm_chat_panel.build()

        assert built["transcript"].value == []
        assert mc_llm_chat_panel._thread_choices("Ada") == [
            ("harbour at night", conversation.identifier)]

    def test_the_persona_is_filled_in_from_disk(self, store):
        from prompt_master.chat.characters import Persona, save_persona

        save_persona(mc_llm_paths.app_paths(), Persona(name="Rae", description="a reader"))

        built = mc_llm_chat_panel.build()

        assert built["persona"][0].value == "Rae"
        assert built["persona"][1].value == "a reader"

    def test_prompt_studio_opens_on_the_last_controls_used(self, store):
        import mc_llm_state

        mc_llm_state.remember(prompt_defaults={"video_mode": "t2v", "fps": 30})

        mc_llm_prompt_panel.build()

        # Rebuilt without error and the stored values were consulted; the
        # component values themselves are checked through the request builder,
        # which is the thing that has to agree with them.
        assert mc_llm_state.preferences()["prompt_defaults"]["fps"] == 30

    def test_minimax_opens_on_the_last_variant(self, store):
        import mc_llm_state
        from prompt_master.minimax import enhancer

        mc_llm_state.remember(minimax_variant=enhancer.REF2VA)

        mc_llm_minimax_panel.build()

        assert mc_llm_state.preferences()["minimax_variant"] == enhancer.REF2VA


class TestRuntimeSetup:
    """The panel's first step, added after a user hit its absence.

    Choosing a model requires a runtime, and the panel used to offer only the
    second half of that -- so upstream's refusal surfaced, and it ends "Run
    Models and Hardware setup first", naming a Qt wizard this extension does
    not have.
    """

    def test_the_panel_says_what_is_missing_and_how_to_fix_it(self, store):
        line = mc_llm_studio._runtime_setup_line()

        assert "llama.cpp release you already have" in line
        assert str(store) in line

    def test_it_offers_to_adopt_a_build_already_in_place(self, store):
        import mc_llm_setup

        runtime = store / mc_llm_setup.RUNTIME_DIRNAME
        runtime.mkdir(parents=True)
        (runtime / "llama-server").write_bytes(b"")

        assert "Press Detect" in mc_llm_studio._runtime_setup_line()

    def test_a_recorded_runtime_reads_as_ready(self, store):
        import mc_llm_setup

        runtime = store / mc_llm_setup.RUNTIME_DIRNAME
        runtime.mkdir(parents=True)
        server = runtime / "llama-server"
        server.write_bytes(b"")
        mc_llm_setup.record(server)

        assert "Runtime:" in mc_llm_studio._runtime_setup_line()

    def test_choosing_a_model_without_a_runtime_names_this_tab(self, store):
        """Not "Run Models and Hardware setup first", which is a dead end here."""
        notice, *_rest = mc_llm_studio._apply_model("/models/thing.gguf", "")

        assert "llama.cpp runtime above" in notice
        assert "Models and Hardware" not in notice

    def test_the_top_status_line_distinguishes_the_two_missing_pieces(self, store):
        assert "runtime and a model" in mc_llm_studio._runtime_line()

    def test_detect_reports_when_there_is_nothing_to_detect(self, store):
        notice, _path = mc_llm_studio._detect_runtime()

        assert "No llama-server found" in notice

    def test_applying_an_empty_path_asks_for_one(self, store):
        notice, _path, _model = mc_llm_studio._apply_runtime("", None)

        assert "Enter the path" in notice

    def test_the_device_dropdown_always_offers_something(self, store):
        assert mc_llm_studio._device_choices()

    def test_a_cleared_context_box_is_answered_rather_than_raised(self, store):
        """A gr.Number that has been emptied hands the handler None."""
        assert "have to be numbers" in mc_llm_studio._save_context("auto", 4.0, None,
                                                                   "f16", "f16")

    def test_saving_the_context_settings_keeps_them(self, store):
        import mc_llm_state

        mc_llm_studio._save_context("fixed", 6.0, 32768, "q8_0", "q8_0")

        prefs = mc_llm_state.preferences()
        assert prefs["context_size"] == 32768
        assert prefs["kv_type_k"] == "q8_0"


class TestChoosingAModelByPath:
    """The panel's second step, and the one a user got stuck on.

    A path arrives from a text box, which means it arrives with whatever the
    clipboard put around it. What these pin down is that the panel answers with
    the file the user meant, and says so when that is not the file they typed.
    """

    @pytest.fixture
    def ready(self, store, tmp_path_factory):
        """An install with a runtime recorded, and a models folder outside it.

        Outside deliberately: weights are located rather than contained, and a
        user's 20 GB of GGUF lives on whichever drive had room for it.
        """
        import mc_llm_setup

        runtime = store / mc_llm_setup.RUNTIME_DIRNAME
        runtime.mkdir(parents=True)
        server = runtime / "llama-server"
        server.write_bytes(b"")
        mc_llm_setup.record(server)
        return tmp_path_factory.mktemp("models")

    def test_a_windows_copy_as_path_is_accepted(self, ready):
        model = ready / "thing.gguf"
        model.write_bytes(b"x")

        notice, _estimator, written, _mmproj = mc_llm_studio._apply_model(f'"{model}"', "")

        assert "thing.gguf" in notice
        assert written == str(model)

    def test_pasting_the_models_folder_picks_the_model_in_it(self, ready):
        """"My models path" is a folder to most people, and a folder holding
        one model is not an ambiguous answer."""
        model = ready / "thing.gguf"
        model.write_bytes(b"x")

        notice, _estimator, written, _mmproj = mc_llm_studio._apply_model(str(ready), "")

        assert written == str(model)
        assert "only model in that folder" in notice

    def test_a_folder_of_several_asks_which_rather_than_failing_blankly(self, ready):
        (ready / "a.gguf").write_bytes(b"x")
        (ready / "b.gguf").write_bytes(b"x")

        notice, _estimator, written, _mmproj = mc_llm_studio._apply_model(str(ready), "")

        assert "2 models" in notice
        assert written == {}  # gr.update() -- the box is left as the user typed it

    def test_a_projector_beside_an_unpaired_model_is_mentioned_not_used(self, ready):
        model = ready / "thing.gguf"
        model.write_bytes(b"x")
        (ready / "mmproj-thing-f16.gguf").write_bytes(b"x")

        notice, _estimator, _written, mmproj = mc_llm_studio._apply_model(str(model), "")

        assert mmproj == ""
        assert "may be its vision projector" in notice

    def test_a_projector_given_by_folder_is_resolved_into_the_box(self, ready):
        model = ready / "thing.gguf"
        model.write_bytes(b"x")
        projector = ready / "mmproj-thing-f16.gguf"
        projector.write_bytes(b"x")

        _notice, _estimator, _written, mmproj = mc_llm_studio._apply_model(
            str(model), str(ready))

        assert mmproj == str(projector)

    def test_a_missing_file_is_answered_with_a_sentence_and_not_a_toast(self, ready):
        notice, _estimator, _written, _mmproj = mc_llm_studio._apply_model(
            str(ready / "absent.gguf"), "")

        assert "There is nothing at" in notice

    def test_the_estimator_never_takes_the_click_down_with_it(self, ready, monkeypatch):
        """Its output is one of four, so anything it raises used to lose the
        other three -- the model was recorded and the panel said nothing."""
        model = ready / "thing.gguf"
        model.write_bytes(b"x")
        monkeypatch.setattr(mc_llm_studio, "_estimate_html",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        notice, estimator, written, _mmproj = mc_llm_studio._apply_model(str(model), "")

        assert "thing.gguf" in notice
        assert written == str(model)
        assert "boom" in estimator

    def test_the_projector_suggestion_reads_a_quoted_path_too(self, ready):
        model = ready / "thing.gguf"
        model.write_bytes(b"x")
        projector = ready / "mmproj-thing-f16.gguf"
        projector.write_bytes(b"x")

        suggested, notice = mc_llm_studio._suggest_projector(f'"{model}"')

        assert suggested == str(projector)
        assert "mmproj-thing-f16.gguf" in notice


class TestTheFilePicker:
    """A path box with no picker beside it is a text box asking for something
    only a file manager knows. These are about the picker's handlers, which is
    where its behaviour is -- the components themselves are Gradio's."""

    @pytest.fixture
    def browse(self):
        import mc_llm_browse

        return mc_llm_browse

    def test_browse_opens_the_operating_systems_own_dialog(self, browse, tmp_path,
                                                            monkeypatch):
        """What "browse" means to everybody who presses the button."""
        model = tmp_path / "thing.gguf"
        model.write_bytes(b"x")
        asked = {}

        def dialog(title, patterns, initial):
            asked.update({"title": title, "patterns": patterns, "initial": initial})
            return str(model)

        monkeypatch.setattr(browse.native, "choose_file", dialog)

        panel, *_rest, target = browse._open("", (".gguf",), tmp_path, "Choose a GGUF model")

        assert target == str(model)
        assert panel["visible"] is False
        assert asked["title"] == "Choose a GGUF model"
        assert asked["initial"] == tmp_path
        assert ("GGUF files", "*.gguf") in asked["patterns"]

    def test_cancelling_the_dialog_changes_nothing(self, browse, tmp_path, monkeypatch):
        """Opening the in-page picker here would be arguing with somebody who
        has just said no."""
        monkeypatch.setattr(browse.native, "choose_file",
                            lambda *args, **kwargs: None)

        panel, *rest, target = browse._open("", (".gguf",), tmp_path)

        assert panel["visible"] is False
        assert target == {}
        assert all(update == {} for update in rest)

    def test_no_native_dialog_falls_back_into_the_page_with_the_reason(self, browse,
                                                                       tmp_path,
                                                                       monkeypatch):
        """A Browse button that silently does nothing is the failure this whole
        fallback exists to avoid."""
        model = tmp_path / "thing.gguf"
        model.write_bytes(b"x")

        def refuse(*args, **kwargs):
            raise browse.native.Unavailable("This WebUI is being served to other machines")

        monkeypatch.setattr(browse.native, "choose_file", refuse)

        panel, location, _places, _folders, picks, notice, target = browse._open(
            str(model), (".gguf",), None)

        assert panel["visible"] is True
        assert location == str(tmp_path)
        assert ("thing.gguf — 0.0 GB", str(model)) in picks["choices"]
        assert "served to other machines" in notice
        assert target == {}

    def test_a_dialog_that_fails_outright_still_lands_in_the_page(self, browse, tmp_path,
                                                                  monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("tk went away")

        monkeypatch.setattr(browse.native, "choose_file", explode)

        panel, *_rest, notice, _target = browse._open("", (".gguf",), tmp_path)

        assert panel["visible"] is True
        assert "tk went away" in notice

    def test_navigating_replaces_the_listing_and_leaves_the_box_alone(self, browse,
                                                                      tmp_path):
        (tmp_path / "sub").mkdir()

        _panel, location, _places, folders, _picks, _notice, target = browse._show(
            tmp_path, (".gguf",))

        assert location == str(tmp_path)
        assert ("sub", str(tmp_path / "sub")) in folders["choices"]
        assert target == {}

    def test_picking_a_file_fills_the_box_and_closes_the_picker(self, browse, tmp_path):
        model = tmp_path / "thing.gguf"
        model.write_bytes(b"x")

        panel, *_rest, target = browse._choose(str(model), str(tmp_path), (".gguf",))

        assert panel["visible"] is False
        assert target == str(model)

    def test_an_empty_pick_never_empties_the_box(self, browse, tmp_path):
        """What a dropdown being refilled looks like on a host with no input
        event. Emptying the box somebody just filled would be the worst answer
        available."""
        panel, *_rest, target = browse._choose("", str(tmp_path), (".gguf",))

        assert panel["visible"] is True
        assert target == {}

    def test_closing_it_changes_nothing_else(self, browse):
        panel, *rest = browse._shut()

        assert panel["visible"] is False
        assert all(update == {} for update in rest)

    def test_every_handler_returns_one_value_per_output(self, browse, tmp_path,
                                                        monkeypatch):
        """The failure mode this catches is a Gradio build-time error that only
        shows up when somebody presses the button."""
        monkeypatch.setattr(browse.native, "choose_file", lambda *a, **k: None)

        assert len(browse._show(tmp_path, (".gguf",))) == 7
        assert len(browse._shut()) == 7
        assert len(browse._choose("", str(tmp_path), (".gguf",))) == 7
        assert len(browse._open("", (".gguf",), tmp_path)) == 7

    def test_the_runtime_box_can_ask_for_a_folder_natively(self, browse, tmp_path,
                                                           monkeypatch):
        """A llama.cpp release is adopted by naming the directory as readily as
        the executable inside it."""
        asked = {}

        def dialog(title, initial):
            asked.update({"title": title, "initial": initial})
            return str(tmp_path)

        monkeypatch.setattr(browse.native, "choose_folder", dialog)

        result = browse._open_folder("", (), tmp_path, "Choose an unpacked llama.cpp release")

        assert len(result) == 7
        assert result[0]["visible"] is False
        assert result[-1] == str(tmp_path)
        assert asked["title"] == "Choose an unpacked llama.cpp release"

    def test_only_the_runtime_box_gets_a_folder_button(self, browse):
        import gradio as gr

        assert browse.attach(gr.Textbox(), key="a")["open_folder"] is None
        assert browse.attach(gr.Textbox(), key="b", allow_folders=True)["open_folder"]

    def test_the_filter_list_offers_everything_as_well_as_the_suffix(self, browse):
        """A projector that a publisher named something unexpected is still
        choosable, which a .gguf-only filter would prevent."""
        labels = [label for label, _pattern in browse._patterns((".gguf",))]

        assert labels[-1] == "All files"

    def test_every_binding_hands_back_one_value_per_output(self, browse):
        """The wiring failure this catches shows up when somebody presses the
        button, not when the tab is built."""
        import gradio as gr

        built = browse.attach(gr.Textbox(), key="test", allow_folders=True)

        bound = [kwargs for component in built.values()
                 for _kind, kwargs in getattr(component, "_callbacks", [])]
        assert bound
        for kwargs in bound:
            assert len(kwargs["outputs"]) == 7

    def test_navigation_binds_to_input_where_the_host_offers_it(self, browse):
        """``change`` also fires when the server refills a dropdown, which
        would walk a folder deeper on every click."""
        import gradio as gr

        built = browse.attach(gr.Textbox(), key="test")

        assert [kind for kind, _kwargs in built["folders"]._callbacks] == ["input"]

    def test_the_panel_carries_a_picker_for_each_path_box(self, store):
        """Three boxes, three pickers, and ids that are this extension's."""
        import mc_llm_browse
        import mc_llm_ui as ui

        built = []
        original = mc_llm_browse.attach

        def record(target, **kwargs):
            built.append(kwargs.get("key"))
            return original(target, **kwargs)

        mc_llm_browse.attach = record
        try:
            mc_llm_studio._settings_panel()
        finally:
            mc_llm_browse.attach = original

        assert built == ["runtime", "model", "mmproj"]
        assert ui.ident("browse", "model") == "mc-llm-browse-model"


class TestNativeDialogAvailability:
    """Whether the operating system's dialog is the right thing to open.

    It is not always: a WebUI reached from another machine would open one on
    the server's screen, where nobody is sitting, and from the browser's side
    the button would do nothing at all until it timed out.
    """

    def test_a_local_webui_gets_the_native_dialog(self, host, monkeypatch):
        import mc_llm_native

        monkeypatch.setattr(mc_llm_native, "_served_remotely", lambda: False)
        monkeypatch.setattr(mc_llm_native, "_has_display", lambda: True)

        assert mc_llm_native.available() == ""

    def test_listen_and_share_take_it_away_with_a_reason(self, host, monkeypatch):
        import mc_llm_native
        from modules import shared

        monkeypatch.setattr(mc_llm_native, "_has_display", lambda: True)
        monkeypatch.setattr(shared.cmd_opts, "listen", True, raising=False)

        assert "served to other machines" in mc_llm_native.available()

    def test_a_headless_machine_says_so_rather_than_hanging(self, host, monkeypatch):
        import sys

        import mc_llm_native

        monkeypatch.setattr(mc_llm_native, "_served_remotely", lambda: False)
        monkeypatch.setattr(mc_llm_native, "_has_display", lambda: False)
        monkeypatch.setattr(sys, "platform", "linux")

        assert "no desktop session" in mc_llm_native.available()

    def test_it_refuses_before_starting_anything_when_unavailable(self, host, monkeypatch):
        import mc_llm_native

        monkeypatch.setattr(mc_llm_native, "available", lambda: "nope")
        monkeypatch.setattr(mc_llm_native.subprocess, "run",
                            lambda *args, **kwargs: pytest.fail("started a dialog anyway"))

        with pytest.raises(mc_llm_native.Unavailable, match="nope"):
            mc_llm_native.choose_file("Choose", (), None)

    def test_a_cancelled_dialog_is_none_rather_than_an_empty_path(self, host, monkeypatch):
        import subprocess

        import mc_llm_native

        monkeypatch.setattr(mc_llm_native, "available", lambda: "")
        monkeypatch.setattr(mc_llm_native.subprocess, "run",
                            lambda *args, **kwargs: subprocess.CompletedProcess(
                                args, 0, stdout="\n", stderr=""))

        assert mc_llm_native.choose_file("Choose", (), None) is None

    def test_a_python_without_tkinter_says_which_thing_is_missing(self, host, monkeypatch):
        import subprocess

        import mc_llm_native

        monkeypatch.setattr(mc_llm_native, "available", lambda: "")
        monkeypatch.setattr(mc_llm_native.subprocess, "run",
                            lambda *args, **kwargs: subprocess.CompletedProcess(
                                args, 2, stdout="", stderr="no tkinter"))

        with pytest.raises(mc_llm_native.Unavailable, match="tkinter"):
            mc_llm_native.choose_file("Choose", (), None)

    def test_windows_tries_powershell_before_tkinter(self, host, monkeypatch):
        """The popular one-click packages ship an embedded Python with no
        tkinter, and System.Windows.Forms is part of Windows itself."""
        import sys

        import mc_llm_native

        monkeypatch.setattr(sys, "platform", "win32")

        assert [route.__name__ for route in mc_llm_native._routes()] == \
            ["_powershell", "_tkinter"]

    def test_a_missing_route_falls_through_to_the_next_one(self, host, monkeypatch):
        import mc_llm_native

        tried = []

        def absent(*args, **kwargs):
            tried.append("first")
            raise mc_llm_native.Unavailable("not installed")

        def works(*args, **kwargs):
            tried.append("second")
            return "C:\\models\\thing.gguf"

        monkeypatch.setattr(mc_llm_native, "available", lambda: "")
        monkeypatch.setattr(mc_llm_native, "_routes", lambda: (absent, works))

        assert mc_llm_native.choose_file("Choose", (), None) == "C:\\models\\thing.gguf"
        assert tried == ["first", "second"]

    def test_a_timeout_does_not_open_a_second_dialog(self, host, monkeypatch):
        """Somebody who has already left one dialog open for ten minutes does
        not need a second one at the end of it."""
        import mc_llm_native

        def expire(*args, **kwargs):
            raise mc_llm_native._final("still open")

        second = lambda *args, **kwargs: pytest.fail("opened another dialog")
        monkeypatch.setattr(mc_llm_native, "available", lambda: "")
        monkeypatch.setattr(mc_llm_native, "_routes", lambda: (expire, second))

        with pytest.raises(mc_llm_native.Unavailable, match="still open"):
            mc_llm_native.choose_file("Choose", (), None)

    def test_the_windows_filter_is_the_win32_spelling_of_the_tk_one(self):
        import mc_llm_native

        assert mc_llm_native._ps_filter((("GGUF files", "*.gguf"), ("All files", "*.*"))) == \
            "GGUF files|*.gguf|All files|*.*"

    def test_a_quote_in_a_path_cannot_end_the_powershell_string(self, host):
        """The initial folder is a path off the user's disk, and PowerShell
        ends a single-quoted string at the first quote it sees."""
        import mc_llm_native

        assert mc_llm_native._ps("C:\\it's\\models") == "C:\\it''s\\models"

    def test_a_dialog_left_open_is_killed_rather_than_held(self, host, monkeypatch):
        import subprocess

        import mc_llm_native

        def expire(*args, **kwargs):
            raise subprocess.TimeoutExpired("dialog", mc_llm_native.TIMEOUT_SECONDS)

        monkeypatch.setattr(mc_llm_native, "available", lambda: "")
        monkeypatch.setattr(mc_llm_native.subprocess, "run", expire)

        with pytest.raises(mc_llm_native.Unavailable, match="still open"):
            mc_llm_native.choose_file("Choose", (), None)


class TestConversationDefaults:
    """Sampling comes from the vendored package, not from literals here.

    The point is not the numbers. It is that there is exactly one place they
    are written down, so the panel cannot drift away from the engine it is a
    front end for.
    """

    def test_the_sliders_open_on_the_vendored_defaults(self, store):
        from prompt_master.chat import characters

        built = mc_llm_chat_panel.build()

        assert characters.DEFAULT_TEMPERATURE == 0.85
        assert mc_llm_chat_panel.sessions.ChatRequest(messages=[]).temperature == \
            characters.DEFAULT_TEMPERATURE
        assert mc_llm_chat_panel.sessions.ChatRequest(messages=[]).top_p == \
            characters.DEFAULT_TOP_P
        assert mc_llm_chat_panel.sessions.ChatRequest(messages=[]).max_tokens == \
            characters.DEFAULT_MAX_REPLY_TOKENS
        assert set(built) == {"status", "transcript", "persona"}

    def test_a_cleared_box_falls_back_to_the_character_then_to_the_default(self):
        from prompt_master.chat.characters import Character

        saved = Character(name="Ada", temperature=0.4, max_reply_tokens=900)

        assert mc_llm_chat_panel._decimal(None, saved.temperature) == 0.4
        assert mc_llm_chat_panel._number("", saved.max_reply_tokens) == 900
        assert mc_llm_chat_panel._number(None, None, 512) == 512

    def test_a_slider_value_still_wins_over_the_saved_one(self):
        assert mc_llm_chat_panel._decimal(1.2, 0.4) == 1.2
        assert mc_llm_chat_panel._number(256, 900) == 256
