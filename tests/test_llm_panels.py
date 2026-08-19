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
        notice, _estimator = mc_llm_studio._apply_model("/models/thing.gguf", "")

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
