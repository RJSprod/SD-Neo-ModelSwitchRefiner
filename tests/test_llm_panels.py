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
import mc_llm_krea_panel
from prompt_master.krea import library
import mc_llm_minimax_panel
import mc_llm_paths
import mc_llm_prompt_panel
import mc_llm_studio


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture
def attached(store):
    """A thread whose last message of yours carries a picture on disk.

    Built through the store rather than by writing a data URL into a message,
    because that is what a conversation now holds and because the round trip is
    the thing under test: bytes in, a record out, and a file somebody could go
    and look at.
    """
    from PIL import Image

    import mc_llm_attachments
    from prompt_master.chat.history import ASSISTANT, ChatStore, USER

    chats = ChatStore(store / "chats")
    conversation = chats.new("Ada")
    for index in range(3):
        conversation.append(USER, f"ask {index}")
        conversation.append(ASSISTANT, f"reply {index}")
    record = mc_llm_attachments.store(Image.new("RGB", (12, 12), (200, 30, 30)), "Ada")
    conversation.append(USER, "look at this", image_name="frame.png", image_path=record)
    chats.save(conversation)
    return chats, conversation, record


@pytest.fixture
def a_card(monkeypatch):
    """One CUDA card in the machine, so the device list offers it both ways.

    Faked at the nvidia-smi boundary rather than above it, so what is under
    test includes the pairing itself: every card is offered once holding the
    weights and once in mixed mode.
    """
    import mc_llm_setup
    from prompt_master.core.models import GpuInfo

    card = GpuInfo(0, "GPU-0000", "NVIDIA GeForce RTX 3090", 24576, 23304, "560.94", 8.6)
    monkeypatch.setattr("prompt_master.inference.device_detection.detect_gpus",
                        lambda *args, **kwargs: [card])
    mc_llm_setup.forget_devices()
    yield card
    mc_llm_setup.forget_devices()


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

    def test_an_attachment_is_shown_in_the_transcript(self):
        """The picture itself, not a line of italic text saying there was one.
        A conversation about a photograph that does not show the photograph is
        a conversation missing half of itself a week later."""
        from prompt_master.chat.history import USER, Conversation

        conversation = Conversation(identifier="x", character="Ada")
        conversation.append(USER, "look", "data:image/jpeg;base64,AA", "frame.png")

        shown = mc_llm_chat_panel._transcript(conversation)[0][0]

        assert "<img" in shown and "data:image/jpeg;base64,AA" in shown
        assert shown.endswith("look")

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


class TestPerMessageActions:
    """The actions the standalone application hangs on every bubble.

    Gradio 4\u2019s Chatbot has nowhere to hang them, which is why they are one
    bar under the transcript applying to whichever message was clicked. The two
    things that can silently go wrong with that shape are both tested here: the
    map from a click to a message, and the arity of the updates the bar is
    redrawn from -- a handler one value short would put a label into a
    visibility and nothing would raise.
    """

    def _thread(self, store, turns=2):
        from prompt_master.chat.history import ASSISTANT, ChatStore, USER

        chats = ChatStore(store / "chats")
        conversation = chats.new("Ada")
        for index in range(turns):
            conversation.append(USER, f"ask {index}")
            conversation.append(ASSISTANT, f"reply {index}")
        chats.save(conversation)
        return conversation

    def test_the_map_names_the_message_the_click_lands_on(self):
        from prompt_master.chat.history import ASSISTANT, Conversation, USER

        conversation = Conversation(identifier="x", character="Ada")
        conversation.append(USER, "hello")
        conversation.append(ASSISTANT, "hi")
        conversation.append(ASSISTANT, "and again")

        rows, positions = mc_llm_chat_panel._view(conversation)

        assert rows == [["hello", "hi"], [None, "and again"]]
        assert mc_llm_chat_panel._message_at(positions, 0, 0) == 0
        assert mc_llm_chat_panel._message_at(positions, 0, 1) == 1
        assert mc_llm_chat_panel._message_at(positions, 1, 1) == 2

    def test_a_click_on_nothing_is_not_a_message(self):
        _, positions = mc_llm_chat_panel._view(None)

        assert mc_llm_chat_panel._message_at(positions, 0, 0) == \
            mc_llm_chat_panel.NO_SELECTION
        assert mc_llm_chat_panel._message_at([[0, 0, 0]], 4, 1) == \
            mc_llm_chat_panel.NO_SELECTION

    def test_tapping_the_same_message_again_puts_the_bar_away(self, store):
        """One gesture opens and closes it: a Chatbot bubble has no second
        affordance to dismiss from."""
        class Click:
            index = [0, 0]

        conversation = self._thread(store)
        _, positions = mc_llm_chat_panel._view(conversation)

        opened = mc_llm_chat_panel._select_message("Ada", conversation.identifier, positions,
                                                   mc_llm_chat_panel.NO_SELECTION, Click())
        closed = mc_llm_chat_panel._select_message("Ada", conversation.identifier, positions,
                                                   opened[2], Click())

        assert opened[2] == 0 and opened[5].get("visible") is True
        assert closed[2] == mc_llm_chat_panel.NO_SELECTION
        assert closed[5].get("visible") is False

    def test_tapping_a_different_message_moves_the_bar_to_it(self, store):
        class Click:
            index = [0, 1]

        conversation = self._thread(store)
        _, positions = mc_llm_chat_panel._view(conversation)

        moved = mc_llm_chat_panel._select_message("Ada", conversation.identifier, positions,
                                                  0, Click())

        assert moved[2] == 1 and moved[5].get("visible") is True

    def test_the_sheet_and_the_updates_that_redraw_it_are_the_same_length(self):
        assert len(mc_llm_chat_panel.SELECTION_ORDER) == \
            len(mc_llm_chat_panel._selection_updates(None, -1))

    def test_which_actions_apply_depends_on_the_message(self, store):
        conversation = self._thread(store)
        names = mc_llm_chat_panel.SELECTION_ORDER

        for index, expected in ((0, "resend"), (1, "regenerate")):
            shown = dict(zip(names, mc_llm_chat_panel._selection_updates(conversation, index)))
            assert shown["sheet"].get("visible") is True
            assert shown[expected].get("visible") is True

        # Continue is offered only on the reply still at the end: anything
        # before it would be carrying on with a turn already answered.
        last = len(conversation.messages) - 1
        assert mc_llm_chat_panel._selection_updates(
            conversation, last)[7].get("visible") is True
        assert mc_llm_chat_panel._selection_updates(
            conversation, 1)[7].get("visible") is False

    def test_the_version_pager_appears_once_there_is_more_than_one(self, store):
        conversation = self._thread(store)
        message = conversation.messages[1]
        message.add_version("a second attempt")

        shown = mc_llm_chat_panel._selection_updates(conversation, 1)

        assert shown[2].get("visible") is True          # back
        assert "2/2" in shown[3].get("value")           # the pager itself
        assert shown[4].get("interactive") is False     # nothing after the last
        assert shown[5].get("visible") is True          # delete this version

    def test_paging_back_shows_the_earlier_attempt(self, store):
        conversation = self._thread(store)
        conversation.messages[1].add_version("second attempt")
        mc_llm_chat_panel._chats().save(conversation)

        mc_llm_chat_panel._page_version(-1)("Ada", conversation.identifier, 1)
        reloaded = mc_llm_chat_panel._load("Ada", conversation.identifier)

        assert reloaded.messages[1].text == "reply 0"

    def test_branching_copies_up_to_the_chosen_message(self, store):
        conversation = self._thread(store)

        result = mc_llm_chat_panel._branch_here("Ada", conversation.identifier, 1, "")
        branched = mc_llm_chat_panel._load("Ada", result[1])

        assert [message.text for message in branched.messages] == ["ask 0", "reply 0"]
        # The thread it came from is untouched, which is what "a branch is a
        # copy" has to mean for it to survive being reloaded.
        assert len(mc_llm_chat_panel._load("Ada", conversation.identifier).messages) == 4

    def test_deleting_from_here_takes_the_message_and_everything_after(self, store):
        conversation = self._thread(store)

        mc_llm_chat_panel._delete_from("Ada", conversation.identifier, 2)
        reloaded = mc_llm_chat_panel._load("Ada", conversation.identifier)

        assert [message.text for message in reloaded.messages] == ["ask 0", "reply 0"]

    def test_editing_writes_the_showing_version_and_nothing_else(self, store):
        conversation = self._thread(store)
        conversation.messages[1].add_version("second attempt")
        mc_llm_chat_panel._chats().save(conversation)

        mc_llm_chat_panel._commit_edit("Ada", conversation.identifier, 1, "  edited  ", None)
        reloaded = mc_llm_chat_panel._load("Ada", conversation.identifier)

        assert reloaded.messages[1].versions == ["reply 0", "edited"]

    def test_every_action_hands_back_one_value_per_output(self, store):
        """The bar is redrawn from the same list by every one of them, so one
        handler returning a short list is one handler putting a value in the
        wrong control."""
        conversation = self._thread(store)
        identifier = conversation.identifier
        # The four leading values, the header, and one per action-sheet control.
        width = 5 + len(mc_llm_chat_panel.SELECTION_ORDER)

        for result in (
            mc_llm_chat_panel._close_selection("Ada", identifier),
            mc_llm_chat_panel._page_version(1)("Ada", identifier, 1),
            mc_llm_chat_panel._drop_version("Ada", identifier, 1),
            mc_llm_chat_panel._commit_edit("Ada", identifier, 0, "changed", None),
            mc_llm_chat_panel._delete_message("Ada", identifier, 3),
            mc_llm_chat_panel._delete_from("Ada", identifier, 2),
            mc_llm_chat_panel._select_message("Ada", identifier, [[0, 0, 0]],
                                             mc_llm_chat_panel.NO_SELECTION),
            mc_llm_chat_panel._open_thread("Ada", identifier)[2:],
            mc_llm_chat_panel._open_editor("Ada", identifier, 1),
        ):
            assert len(result) == width

    def test_regenerate_falls_back_to_the_last_reply(self, store):
        """"Again" is about the end of the thread unless somebody has said
        otherwise, so the button still works with nothing selected."""
        conversation = self._thread(store)

        assert mc_llm_chat_panel._last_reply(conversation, -1) == 3
        assert mc_llm_chat_panel._last_reply(conversation, 1) == 1
        assert mc_llm_chat_panel._last_reply(conversation, None) == 3

    def test_a_reply_that_never_arrived_is_taken_away(self, store):
        from prompt_master.chat.history import ASSISTANT

        conversation = self._thread(store)
        conversation.append(ASSISTANT, "")

        mc_llm_chat_panel._tidy(conversation, len(conversation.messages) - 1)

        assert len(conversation.messages) == 4

    def test_a_failed_regenerate_falls_back_to_the_attempt_it_had(self, store):
        conversation = self._thread(store)
        conversation.messages[1].add_version("")

        mc_llm_chat_panel._tidy(conversation, 1)

        assert conversation.messages[1].text == "reply 0"
        assert len(conversation.messages) == 4


class TestRegeneratingInTheMiddleBranches:
    """Asking for a different reply must not cost the conversation after it.

    Reported as exactly that: "if I go back to the original response, I expect
    the entire thread to load". It did not, because regenerating ran
    ``truncate_after`` -- every message after the one being rewritten was
    deleted, and paging back to the first attempt showed it with the rest of
    the conversation gone for ever.

    A version is one string, so versions cannot hold what followed. A branch
    can, and the store already had one. So the rule is now positional: at the
    end of a thread, where nothing follows, regenerating pages between
    attempts; anywhere else it branches, and the thread it came from keeps
    every word of what came after.
    """

    def _thread(self, store, monkeypatch, turns=3, pieces=("A new", " reply")):
        import mc_llm_chat_panel as chat
        from prompt_master.chat.characters import Character, Persona, save_persona
        from prompt_master.chat.history import ASSISTANT, ChatStore, USER

        save_persona(mc_llm_paths.app_paths(), Persona(name="Me", description="a reader"))
        chats = ChatStore(store / "chats")
        monkeypatch.setattr(chat, "_chats", lambda: chats)

        class Characters:
            def load(self, who):
                return Character(name="Ada", context="a reader of maps")

        monkeypatch.setattr(chat, "_characters", lambda: Characters())
        events = [chat.sessions.Event(chat.sessions.CHUNK, piece) for piece in pieces]
        events.append(chat.sessions.Event(chat.sessions.DONE, "".join(pieces)))
        monkeypatch.setattr(chat.sessions, "conversation",
                            lambda request, cancel: iter(events))

        conversation = chats.new("Ada")
        for index in range(turns):
            conversation.append(USER, f"ask {index}")
            conversation.append(ASSISTANT, f"reply {index}")
        chats.save(conversation)
        return chat, chats, conversation

    def _run(self, chat, identifier, index):
        """Regenerate to completion, and hand back the events it yielded."""
        return list(chat._regenerate("Ada", identifier, index,
                                     None, None, None, None, ""))

    # -- the middle: a branch ------------------------------------------------ #

    def test_the_thread_it_came_from_keeps_everything_after_it(self, store, monkeypatch):
        """The bug, as an assertion: six messages before, six messages after."""
        chat, chats, conversation = self._thread(store, monkeypatch)

        self._run(chat, conversation.identifier, 1)

        kept = chats.load("Ada", conversation.identifier)
        assert [message.text for message in kept.messages] == [
            "ask 0", "reply 0", "ask 1", "reply 1", "ask 2", "reply 2"]

    def test_the_middle_reply_never_grows_a_version(self, store, monkeypatch):
        """Versions only ever exist where nothing follows them, which is what
        makes paging between them safe."""
        chat, chats, conversation = self._thread(store, monkeypatch)

        self._run(chat, conversation.identifier, 1)

        assert chats.load("Ada", conversation.identifier).messages[1].versions == ["reply 0"]

    def test_the_new_reply_is_written_to_a_branch_of_its_own(self, store, monkeypatch):
        chat, chats, conversation = self._thread(store, monkeypatch)

        events = self._run(chat, conversation.identifier, 1)
        identifier = events[-1][1]

        assert identifier != conversation.identifier
        branched = chats.load("Ada", identifier)
        # Up to the message the reply was answering, and then the new reply --
        # so the branch ends on the same turn, answered differently.
        assert [message.text for message in branched.messages] == ["ask 0", "A new reply"]

    def test_the_panel_moves_onto_the_branch_it_just_made(self, store, monkeypatch):
        """Every event carries it, not just the last: a panel still pointing at
        the thread it came from would apply the next action to the wrong
        conversation."""
        chat, chats, conversation = self._thread(store, monkeypatch)

        events = self._run(chat, conversation.identifier, 1)
        identifier = events[-1][1]

        assert {event[1] for event in events} == {identifier}
        assert events[-1][0].get("value") == identifier
        assert identifier in [value for _, value in events[-1][0].get("choices")]

    def test_the_thread_it_made_is_the_one_reopened_later(self, store, monkeypatch):
        import mc_llm_state

        chat, chats, conversation = self._thread(store, monkeypatch)

        identifier = self._run(chat, conversation.identifier, 1)[-1][1]

        assert mc_llm_state.preferences().get("thread") == identifier

    def test_an_opening_reply_branches_rather_than_emptying_the_thread(self, store,
                                                                      monkeypatch):
        """Index nought is the one place ``truncate_after`` would have taken the
        whole conversation. There is no turn before it, so the branch it starts
        is empty -- but it is still a branch, and nothing is lost."""
        from prompt_master.chat.history import ASSISTANT, USER

        chat, chats, conversation = self._thread(store, monkeypatch, turns=0)
        conversation.append(ASSISTANT, "hello, traveller")
        conversation.append(USER, "hello yourself")
        conversation.append(ASSISTANT, "reply 0")
        chats.save(conversation)

        identifier = self._run(chat, conversation.identifier, 0)[-1][1]

        assert [message.text for message in
                chats.load("Ada", conversation.identifier).messages] == [
                    "hello, traveller", "hello yourself", "reply 0"]
        assert [message.text for message in chats.load("Ada", identifier).messages] == \
            ["A new reply"]

    # -- the end: a version -------------------------------------------------- #

    def test_the_last_reply_still_pages_between_attempts(self, store, monkeypatch):
        chat, chats, conversation = self._thread(store, monkeypatch)

        self._run(chat, conversation.identifier, 5)

        kept = chats.load("Ada", conversation.identifier)
        assert len(kept.messages) == 6
        assert kept.messages[5].versions == ["reply 2", "A new reply"]

    def test_regenerating_the_end_stays_in_the_thread_it_is_in(self, store, monkeypatch):
        chat, chats, conversation = self._thread(store, monkeypatch)

        events = self._run(chat, conversation.identifier, 5)

        assert {event[1] for event in events} == {conversation.identifier}
        # A no-op update: there is no new thread for the list to learn about,
        # and nothing was branched to make one.
        assert events[-1][0] == {}
        assert len(chats.listing("Ada")) == 1

    def test_nothing_selected_still_asks_the_last_reply_again(self, store, monkeypatch):
        chat, chats, conversation = self._thread(store, monkeypatch)

        self._run(chat, conversation.identifier, chat.NO_SELECTION)

        assert chats.load("Ada", conversation.identifier).messages[5].versions == \
            ["reply 2", "A new reply"]

    # -- the shape of what it yields ----------------------------------------- #

    def test_every_event_carries_one_value_per_output(self, store, monkeypatch):
        """Two more than :func:`_stream` yields, because this is the one
        streaming handler whose outputs begin with the thread list and the open
        thread. One short and a thread identifier would land in a textbox."""
        chat, chats, conversation = self._thread(store, monkeypatch)
        width = 2 + len(chat._idle(conversation, "", None, ""))

        for index in (1, 5, chat.NO_SELECTION):
            for event in self._run(chat, conversation.identifier, index):
                assert len(event) == width

    def test_a_message_of_your_own_is_refused_in_that_shape_too(self, store, monkeypatch):
        chat, chats, conversation = self._thread(store, monkeypatch)

        events = self._run(chat, conversation.identifier, 0)

        assert len(events) == 1 and events[0][1] == conversation.identifier
        assert [message.text for message in
                chats.load("Ada", conversation.identifier).messages][-1] == "reply 2"


class TestTheRegenerateIconOnAReply:
    """One tap on the bubble, rather than three through the sheet.

    Everything the icon can do is here, because the browser half of it does not
    decide anything: it reports which reply on screen was tapped, and that
    ordinal has to survive the trip. What makes it worth testing rather than
    reading is that ``_view`` pairs turns -- two replies in a row are two rows
    with an empty left side, one exchange is one row holding two messages -- so
    "the third reply" and "the third row" are not the same number, and the map
    between them is the only thing standing between an icon and a rewritten
    message somebody did not point at.
    """

    def _conversation(self, store):
        from prompt_master.chat.history import ASSISTANT, ChatStore, USER

        chats = ChatStore(store / "chats")
        conversation = chats.new("Ada")
        conversation.append(USER, "ask 0")
        conversation.append(ASSISTANT, "reply 0")
        # Two replies in a row: the pairing that makes rows and replies differ.
        conversation.append(ASSISTANT, "reply 1")
        conversation.append(USER, "ask 1")
        conversation.append(ASSISTANT, "reply 2")
        chats.save(conversation)
        return chats, conversation

    def test_the_nth_reply_is_the_nth_reply_and_not_the_nth_row(self, store):
        chats, conversation = self._conversation(store)
        rows, positions = mc_llm_chat_panel._view(conversation)

        # Three replies over three rows, but the second one is alone on its row
        # and the third is on a row of its own after a message of yours.
        assert len(rows) == 3
        assert [mc_llm_chat_panel._reply_at(positions, ordinal) for ordinal in range(3)] == \
            [1, 2, 4]

    def test_a_message_of_your_own_is_never_what_an_ordinal_names(self, store):
        chats, conversation = self._conversation(store)
        _, positions = mc_llm_chat_panel._view(conversation)

        named = [mc_llm_chat_panel._reply_at(positions, ordinal) for ordinal in range(3)]

        assert all(conversation.messages[index].role == "assistant" for index in named)

    def test_an_ordinal_past_the_end_names_nothing(self, store):
        chats, conversation = self._conversation(store)
        _, positions = mc_llm_chat_panel._view(conversation)

        assert mc_llm_chat_panel._reply_at(positions, 3) == mc_llm_chat_panel.NO_SELECTION
        assert mc_llm_chat_panel._reply_at(positions, -1) == mc_llm_chat_panel.NO_SELECTION

    def test_what_the_browser_sends_is_text_and_is_read_as_a_number(self, store):
        """It comes back out of a Textbox, so it is a string, and an empty one
        the first time the tab is opened."""
        chats, conversation = self._conversation(store)
        _, positions = mc_llm_chat_panel._view(conversation)

        assert mc_llm_chat_panel._reply_at(positions, " 2 ") == 4
        for nothing in ("", None, "second", []):
            assert mc_llm_chat_panel._reply_at(positions, nothing) == \
                mc_llm_chat_panel.NO_SELECTION

    def test_no_transcript_at_all_names_nothing(self):
        assert mc_llm_chat_panel._reply_at(None, 0) == mc_llm_chat_panel.NO_SELECTION
        assert mc_llm_chat_panel._reply_at([], 0) == mc_llm_chat_panel.NO_SELECTION

    def test_it_regenerates_the_reply_the_icon_was_on(self, store, monkeypatch):
        branching = TestRegeneratingInTheMiddleBranches()
        chat, chats, conversation = branching._thread(store, monkeypatch)
        _, positions = chat._view(conversation)

        # The second reply of three: mid-thread, so this also proves the icon
        # branches rather than truncating, because it is the same handler.
        events = list(chat._regenerate_reply("Ada", conversation.identifier, positions, "1",
                                             None, None, None, None, ""))

        assert [message.text for message in
                chats.load("Ada", conversation.identifier).messages] == [
                    "ask 0", "reply 0", "ask 1", "reply 1", "ask 2", "reply 2"]
        branched = chats.load("Ada", events[-1][1])
        assert [message.text for message in branched.messages] == \
            ["ask 0", "reply 0", "ask 1", "A new reply"]

    def test_a_stale_icon_is_refused_rather_than_pointed_somewhere_else(self, store,
                                                                       monkeypatch):
        """A transcript the browser is a moment behind on must never cost
        somebody a message: an ordinal that names nothing says so, and does not
        fall back to the last reply the way the sheet's Regenerate does with
        nothing selected."""
        branching = TestRegeneratingInTheMiddleBranches()
        chat, chats, conversation = branching._thread(store, monkeypatch)

        events = list(chat._regenerate_reply("Ada", conversation.identifier, [], "4",
                                             None, None, None, None, ""))

        assert len(events) == 1
        assert [message.text for message in
                chats.load("Ada", conversation.identifier).messages] == [
                    "ask 0", "reply 0", "ask 1", "reply 1", "ask 2", "reply 2"]

    def test_it_answers_in_the_same_shape_the_sheet_does(self, store, monkeypatch):
        branching = TestRegeneratingInTheMiddleBranches()
        chat, chats, conversation = branching._thread(store, monkeypatch)
        _, positions = chat._view(conversation)
        width = 2 + len(chat._idle(conversation, "", None, ""))

        for ordinal in ("0", "2", "nowhere"):
            for event in chat._regenerate_reply("Ada", conversation.identifier, positions,
                                                ordinal, None, None, None, None, ""):
                assert len(event) == width

    def test_the_pair_the_browser_presses_is_in_the_panel(self, monkeypatch):
        """Two invisible components with one job between them. If either loses
        its id the icon stops working silently, because a control nobody can
        see is a control nobody notices has gone."""
        import gradio as gr

        import mc_llm_ui as ui

        seen = []

        def recording(original):
            class Recorded(original):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    seen.append(kwargs.get("elem_id"))
            return Recorded

        for name in ("Textbox", "Button"):
            monkeypatch.setattr(gr, name, recording(getattr(gr, name)))
        mc_llm_chat_panel.build()

        assert ui.ident("chat", "regenerate-at") in seen
        assert ui.ident("chat", "regenerate-now") in seen

    def test_the_script_presses_the_ids_the_panel_declares(self):
        """The two halves are in two languages and nothing links them but these
        strings, so they are compared rather than trusted."""
        from pathlib import Path

        import mc_llm_ui as ui

        script = (Path(mc_llm_chat_panel.__file__).resolve().parent
                  / "javascript" / "llm_studio.js").read_text(encoding="utf-8")

        for name in ("regenerate-at", "regenerate-now", "transcript"):
            assert f'"{ui.ident("chat", name)}"' in script



class TestTheHeaderCarriesTheDestinations:
    """Threads, Character and You are on the bar, not behind a menu.

    Asked for as "I would like to pull out the THREADS, CHARACTER, and YOU so
    they appear top layer like the load state. It should not be in a menu unless
    screen is greatly truncated" -- and the answer to the truncation is that the
    header wraps, which is a smaller loss than three destinations behind a tap.
    """

    def _built_ids(self, monkeypatch):
        import gradio as gr

        seen = []

        def recording(original):
            class Recorded(original):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    seen.append(kwargs.get("elem_id"))
            return Recorded

        for name in ("Button", "Image"):
            monkeypatch.setattr(gr, name, recording(getattr(gr, name)))
        mc_llm_chat_panel.build()
        return seen

    def test_the_three_destinations_are_components_of_their_own(self, store, monkeypatch):
        import mc_llm_ui as ui

        seen = self._built_ids(monkeypatch)

        for name in ("to-threads", "to-character", "to-persona"):
            assert ui.ident("chat", name) in seen

    def test_the_picture_chips_are_where_the_script_looks_for_them(self, store, monkeypatch):
        """Two halves in two languages with nothing between them but these
        strings, so they are compared rather than trusted."""
        from pathlib import Path

        import mc_llm_ui as ui

        seen = self._built_ids(monkeypatch)
        script = (Path(mc_llm_chat_panel.__file__).resolve().parent
                  / "javascript" / "llm_studio.js").read_text(encoding="utf-8")

        for name in ("attach", "image", "edit-attach", "edit-image"):
            assert ui.ident("chat", name) in seen
            assert f'"{ui.ident("chat", name)}"' in script


class TestEditingAMessageInPlace:
    """Edit edits. It does not branch, and it does not re-ask.

    It used to do both, for one of your own messages: the text was lifted into
    the composer and, mid-thread, the whole thread was copied first so that
    sending it again would not destroy the replies that followed.

    Which made Edit a second Branch -- reported as exactly that. Branch is one
    button along and says what it does, so Edit now changes the words where they
    are and leaves the thread alone. The conversation afterwards is one whose
    earlier turn says something else, replies included, and that is the point of
    it: a thread where you asked about the sky, were told "blue", and then made
    the question about the sun is a thread you can ask about.
    """

    def _thread(self, store, turns=3):
        from prompt_master.chat.history import ASSISTANT, ChatStore, USER

        chats = ChatStore(store / "chats")
        conversation = chats.new("Ada")
        for index in range(turns):
            conversation.append(USER, f"ask {index}")
            conversation.append(ASSISTANT, f"reply {index}")
        chats.save(conversation)
        return chats, conversation

    def _waiting(self, store):
        """A thread ending in a message of yours that never got a reply."""
        from prompt_master.chat.history import USER

        chats, conversation = self._thread(store)
        conversation.append(USER, "and one more thing")
        chats.save(conversation)
        return chats, conversation

    def shown(self, answers):
        """The action-sheet half of one ``view`` answer, by name."""
        return dict(zip(mc_llm_chat_panel.SELECTION_ORDER, answers[5:]))

    # -- one behaviour, both roles ------------------------------------------- #

    def test_your_own_message_opens_the_editor_rather_than_the_composer(self, store):
        chats, conversation = self._thread(store)

        shown = self.shown(mc_llm_chat_panel._open_editor("Ada", conversation.identifier, 2))

        assert shown["edit"].get("visible") is True
        assert shown["edit_box"].get("value") == "ask 1"
        assert shown["composer"].get("visible") is False

    def test_a_reply_opens_the_same_editor(self, store):
        chats, conversation = self._thread(store)

        shown = self.shown(mc_llm_chat_panel._open_editor("Ada", conversation.identifier, 1))

        assert shown["edit"].get("visible") is True
        assert shown["edit_box"].get("value") == "reply 0"

    def test_nothing_after_it_is_destroyed(self, store):
        """The whole difference from what this used to do. Editing message two
        of six leaves six messages, five of them untouched."""
        chats, conversation = self._thread(store)

        mc_llm_chat_panel._commit_edit("Ada", conversation.identifier, 2,
                                       "what colour is the sun", None)

        assert [message.text for message in
                chats.load("Ada", conversation.identifier).messages] == [
                    "ask 0", "reply 0", "what colour is the sun", "reply 1",
                    "ask 2", "reply 2"]

    def test_no_branch_is_made(self, store):
        from prompt_master.chat.history import ChatStore

        chats, conversation = self._thread(store)

        mc_llm_chat_panel._commit_edit("Ada", conversation.identifier, 2, "changed", None)

        assert [row.identifier for row in ChatStore(store / "chats").listing("Ada")] == \
            [conversation.identifier]

    def test_saving_goes_home_rather_than_reopening_the_sheet(self, store):
        """The sheet covers the bottom of the transcript, and reopening it over
        the message just saved is the panel looking stuck on a finished thing."""
        chats, conversation = self._thread(store)

        shown = self.shown(mc_llm_chat_panel._commit_edit(
            "Ada", conversation.identifier, 1, "changed", None))

        assert shown["sheet"].get("visible") is False
        assert shown["edit"].get("visible") is False
        assert shown["composer"].get("visible") is True
        assert chats.load("Ada", conversation.identifier).messages[1].text == "changed"

    def test_nothing_selected_is_refused_rather_than_editing_something(self, store):
        chats, conversation = self._thread(store)

        answers = mc_llm_chat_panel._open_editor("Ada", conversation.identifier, -1)

        assert "Choose a message" in answers[3]

    def test_every_answer_is_the_same_shape(self, store):
        """One output list, whichever branch answered it."""
        chats, conversation = self._thread(store)
        wanted = 5 + len(mc_llm_chat_panel.SELECTION_ORDER)

        for index in (-1, 0, 1, 2):
            assert len(mc_llm_chat_panel._open_editor("Ada", conversation.identifier,
                                                      index)) == wanted

    # -- the picture is part of the message ---------------------------------- #

    def test_the_editor_offers_the_picture_the_message_carries(self, store, attached):
        chats, conversation, record = attached

        shown = self.shown(mc_llm_chat_panel._open_editor("Ada", conversation.identifier, 6))

        assert shown["edit_image"].get("visible") is True
        assert record.split("/")[-1] in str(shown["edit_image"].get("value"))

    def test_a_message_with_no_picture_offers_an_empty_chip(self, store):
        chats, conversation = self._thread(store)

        shown = self.shown(mc_llm_chat_panel._open_editor("Ada", conversation.identifier, 2))

        assert shown["edit_image"].get("visible") is False
        assert shown["edit_image"].get("value") is None

    def test_saving_with_an_empty_chip_takes_the_picture_off(self, store, attached):
        """The chip is the message's picture, so emptying it empties the
        message. Asked for as "I should also be able to remove or change an
        attachment"."""
        chats, conversation, _record = attached

        mc_llm_chat_panel._commit_edit("Ada", conversation.identifier, 6, "look at this", None)

        message = chats.load("Ada", conversation.identifier).messages[6]
        assert not message.attached

    def test_saving_a_different_picture_replaces_it(self, store, attached):
        from PIL import Image

        chats, conversation, record = attached

        mc_llm_chat_panel._commit_edit("Ada", conversation.identifier, 6, "look at this",
                                       Image.new("RGB", (9, 9), (0, 200, 0)))

        message = chats.load("Ada", conversation.identifier).messages[6]
        assert message.image_path and message.image_path != record

    def test_saving_the_same_picture_keeps_the_same_file(self, store, attached):
        """The name of a stored picture is the hash of its bytes, so a save that
        did not change the picture writes nothing and points at what is there."""
        import mc_llm_attachments
        from PIL import Image

        chats, conversation, record = attached
        again = Image.open(mc_llm_attachments.locate(record))

        mc_llm_chat_panel._commit_edit("Ada", conversation.identifier, 6, "look", again)

        assert chats.load("Ada", conversation.identifier).messages[6].image_path == record

    # -- a message nobody ever answered -------------------------------------- #

    def test_a_thread_ending_in_your_own_message_opens_with_it_in_the_box(self, store):
        """What a cancelled or failed reply leaves behind: the message is saved
        before the request goes out, so it survives the reply not arriving.
        Unchanged by any of the above -- it is a different feature that happens
        to move text into the same box."""
        chats, conversation = self._waiting(store)

        identifier, composer, *_ = mc_llm_chat_panel._open_thread(
            "Ada", conversation.identifier)

        assert composer == "and one more thing"
        assert [message.text for message in
                chats.load("Ada", conversation.identifier).messages][-1] == "reply 2"

    def test_an_ordinary_thread_leaves_the_box_alone(self, store):
        chats, conversation = self._thread(store)

        identifier, composer, *_ = mc_llm_chat_panel._open_thread(
            "Ada", conversation.identifier)

        assert composer == {}
        assert len(chats.load("Ada", conversation.identifier).messages) == 6

    def test_it_never_writes_over_something_half_written(self, store):
        chats, conversation = self._waiting(store)

        identifier, composer, *_ = mc_llm_chat_panel._open_thread(
            "Ada", conversation.identifier, "something I was in the middle of")

        assert composer == {}
        assert [message.text for message in
                chats.load("Ada", conversation.identifier).messages][-1] == \
            "and one more thing"

    def test_switching_character_lifts_it_too(self, store):
        chats, conversation = self._waiting(store)

        threads, identifier, composer, *_ = mc_llm_chat_panel._select_character("Ada", "")

        assert composer == "and one more thing"

    def test_a_message_carrying_a_picture_is_left_where_it_is(self, store, attached):
        """A saved picture is a file beside the chat and the composer's chip is
        one the browser uploaded. Rather than reconstruct the second from the
        first, the message stays put -- Edit changes it in place, and **Send
        again from here** re-asks it without disturbing it."""
        chats, conversation, _record = attached

        identifier, composer, *_ = mc_llm_chat_panel._open_thread(
            "Ada", conversation.identifier)

        assert composer == {}
        assert len(chats.load("Ada", conversation.identifier).messages) == 7

    def test_an_empty_last_message_is_not_lifted(self, store):
        from prompt_master.chat.history import USER

        chats, conversation = self._thread(store)
        conversation.append(USER, "   ")
        chats.save(conversation)

        identifier, composer, *_ = mc_llm_chat_panel._open_thread(
            "Ada", conversation.identifier)

        assert composer == {}


class TestTheFooterGoesAway:
    """The one rule in this extension's stylesheet that reaches outside its own
    panels, so the three halves of it are checked against each other here.

    The conversation workspace is built to fit the window: the page does not
    scroll, the transcript does. The footer sits below the fold and takes real
    space, so the page scrolls anyway -- by exactly the height of a row of
    links, from an element no measurement inside the workspace can reach.
    """

    def test_the_setting_is_registered_and_defaults_to_hiding_it(self, host):
        import model_chain

        registered = host.shared.options_templates

        assert model_chain.OPT_HIDE_FOOTER in registered
        assert registered[model_chain.OPT_HIDE_FOOTER].default is True

    def test_the_stylesheet_hides_what_the_script_marks(self):
        """Two files, one attribute, and nothing but the spelling linking them."""
        from pathlib import Path

        root = Path(mc_llm_chat_panel.__file__).resolve().parent
        script = (root / "javascript" / "llm_studio.js").read_text(encoding="utf-8")
        css = (root / "style.css").read_text(encoding="utf-8")

        assert '"data-mc-footer"' in script
        assert '[data-mc-footer="hidden"] #footer' in css

    def test_the_script_reads_the_setting_by_the_name_python_registers(self):
        from pathlib import Path

        import model_chain

        script = ((Path(mc_llm_chat_panel.__file__).resolve().parent
                   / "javascript" / "llm_studio.js").read_text(encoding="utf-8"))

        assert f'"{model_chain.OPT_HIDE_FOOTER}"' in script

    def test_it_hides_a_footer_and_never_one_of_ours(self):
        """`footer` is a tag as well as an id, so the rule is scoped to the
        containers a host footer actually sits in -- and nothing this extension
        builds is a `footer` in the first place."""
        from pathlib import Path

        root = Path(mc_llm_chat_panel.__file__).resolve().parent
        panels = "".join((root / name).read_text(encoding="utf-8")
                         for name in ("mc_llm_chat_panel.py", "mc_llm_studio.py"))

        assert "gr.HTML(\"<footer" not in panels
        assert "<footer" not in panels


class TestTheSurfaces:
    """Conversation is the home state, and everything else is temporary.

    The old drawer was a column, so it was in the layout: on a phone Gradio
    wrapped it above the stage and pushed the composer off the bottom of the
    window. Every configuration surface is now an overlay, and the rules that
    used to be spread over a toggle, three section flags and a media query are
    one function -- which is what these tests are about.
    """

    def _thread(self, store):
        from prompt_master.chat.history import ASSISTANT, ChatStore, USER

        chats = ChatStore(store / "chats")
        conversation = chats.new("Ada")
        conversation.append(USER, "ask")
        conversation.append(ASSISTANT, "reply")
        chats.save(conversation)
        return conversation

    def _visible(self, answered):
        """The visibilities out of a ``_screens``-shaped answer, without its
        leading State."""
        return [update.get("visible")
                for update in answered[1:1 + len(mc_llm_chat_panel.SCREENS)]]

    def test_one_surface_is_open_and_the_others_are_not(self):
        answered = mc_llm_chat_panel._screens("character")
        shown = self._visible(answered)

        assert answered[0] == "character"
        assert shown.count(True) == 1
        assert shown[mc_llm_chat_panel.SCREENS.index("character")] is True

    def test_closing_leaves_the_conversation_alone_on_screen(self):
        answered = mc_llm_chat_panel._close_screens()

        assert answered[0] == ""
        assert all(shown is False for shown in self._visible(answered))

    def test_a_surface_it_has_never_heard_of_opens_nothing(self):
        """Better a menu that did not open than a screen drawn over another."""
        answered = mc_llm_chat_panel._screens("elsewhere")

        assert answered[0] == ""
        assert all(shown is False for shown in self._visible(answered))

    def test_there_is_no_menu_among_the_surfaces(self):
        """Threads, Character and You are buttons on the header now. A menu
        whose whole contents are three destinations is a tap in front of each
        of them, and the button it hung on has a job everywhere else in LLM
        Studio -- it opens the workspace chooser."""
        assert "nav" not in mc_llm_chat_panel.SCREENS

    def test_the_menu_button_puts_this_panel_s_own_surfaces_away(self, store):
        """It opens the shell's workspace sheet, which this panel does not own.
        What it does here is get out of the way, so the sheet does not open over
        a thread list."""
        conversation = self._thread(store)

        answered = mc_llm_chat_panel._leave("Ada", conversation.identifier)

        assert answered[0] == ""
        assert all(shown is False for shown in self._visible(answered))

    def test_the_menu_button_puts_the_message_actions_away(self, store):
        """The action sheet applies to a message the reader is about to stop
        looking at, and a sheet left open under another sheet is the second half
        of every "why is this still here?"."""
        conversation = self._thread(store)
        screens = 1 + len(mc_llm_chat_panel.SCREENS)

        answered = mc_llm_chat_panel._leave("Ada", conversation.identifier)

        assert answered[screens + 2] == mc_llm_chat_panel.NO_SELECTION
        assert answered[screens + 5].get("visible") is False

    def test_tapping_a_thread_opens_it_and_comes_home(self, store):
        conversation = self._thread(store)
        screens = 1 + len(mc_llm_chat_panel.SCREENS)

        answered = mc_llm_chat_panel._open_thread_home("Ada", conversation.identifier)

        assert answered[0] == conversation.identifier
        assert answered[-screens] == ""
        assert all(update.get("visible") is False for update in answered[-screens + 1:])

    def test_a_new_thread_comes_home_too(self, store):
        self._thread(store)
        screens = 1 + len(mc_llm_chat_panel.SCREENS)

        answered = mc_llm_chat_panel._new_thread("Ada", "")

        assert answered[-screens] == ""
        assert all(update.get("visible") is False for update in answered[-screens + 1:])

    def test_the_threads_screen_opens_on_the_current_list(self, store):
        self._thread(store)

        answered = mc_llm_chat_panel._open_threads("Ada", "")

        assert answered[0] == "threads"
        assert self._visible(answered)[
            mc_llm_chat_panel.SCREENS.index("threads")] is True
        assert len(answered[-1].get("choices")) == 1

    def test_the_persona_screen_opens_on_what_is_saved(self, store):
        from prompt_master.chat.characters import Persona, save_persona
        import mc_llm_paths

        save_persona(mc_llm_paths.app_paths(), Persona(name="Rin", description="a reader"))

        answered = mc_llm_chat_panel._open_persona()

        assert answered[-2:] == ["Rin", "a reader"]


class TestTheComposer:
    """Send becomes Stop in the same place, and editing borrows that place."""

    def _thread(self, store):
        from prompt_master.chat.history import ASSISTANT, ChatStore, USER

        chats = ChatStore(store / "chats")
        conversation = chats.new("Ada")
        conversation.append(USER, "ask")
        conversation.append(ASSISTANT, "reply")
        chats.save(conversation)
        return conversation

    def test_only_one_of_send_and_stop_is_ever_on_screen(self):
        idle_send, idle_stop = mc_llm_chat_panel.IDLE
        busy_send, busy_stop = mc_llm_chat_panel.BUSY

        assert (idle_send.get("visible"), idle_stop.get("visible")) == (True, False)
        assert (busy_send.get("visible"), busy_stop.get("visible")) == (False, True)
        # And the one that is not on screen is disabled as well, so a keyboard
        # shortcut aimed at a hidden Stop finds a control that refuses rather
        # than one that quietly cancels nothing.
        assert idle_stop.get("interactive") is False
        assert busy_send.get("interactive") is False

    def test_editing_replaces_the_composer_rather_than_growing_the_panel(self, store):
        """The editor borrows the composer's space, so the transcript above it
        does not move -- which is the whole reason an edit does not open a panel
        of its own."""
        conversation = self._thread(store)
        order = mc_llm_chat_panel.SELECTION_ORDER

        opened = mc_llm_chat_panel._open_editor("Ada", conversation.identifier, 1)
        shown = dict(zip(order, opened[5:]))

        assert shown["edit"].get("visible") is True
        assert shown["edit_box"].get("value") == "reply"
        assert shown["composer"].get("visible") is False
        assert shown["sheet"].get("visible") is False

    def test_saving_an_edit_comes_back_to_the_composer(self, store):
        conversation = self._thread(store)
        order = mc_llm_chat_panel.SELECTION_ORDER

        saved = mc_llm_chat_panel._commit_edit("Ada", conversation.identifier, 1,
                                               "changed", None)
        shown = dict(zip(order, saved[5:]))

        assert shown["edit"].get("visible") is False
        assert shown["composer"].get("visible") is True

    def test_the_header_says_who_and_which_thread(self, store):
        conversation = self._thread(store)
        conversation.title = "harbour at night"

        heading = mc_llm_chat_panel._heading(None, conversation)

        assert "Ada" in heading and "harbour at night" in heading

    def test_the_header_survives_having_no_thread_at_all(self):
        assert "No thread" in mc_llm_chat_panel._heading("", None)

    def test_the_picture_chip_is_not_in_the_layout_until_it_is_asked_for(self):
        """It was a full-width drop target above the composer, open whenever the
        paperclip had been pressed: a panel's worth of empty dashed border on
        the one surface that must not grow."""
        chip, _note = mc_llm_chat_panel._offer_attachment()

        assert chip.get("visible") is True

    def test_emptying_the_chip_takes_it_out_of_the_layout(self):
        """The component draws its own ✕. What that leaves behind is a slot with
        nothing in it, which is a slot that should not be there."""
        chip, note = mc_llm_chat_panel._cleared_attachment()

        assert chip.get("visible") is False
        assert "Ready" in note

    def test_the_paperclip_warns_before_a_message_is_written(self, store, monkeypatch):
        """Whether a picture can be sent depends on the model running, and
        finding that out after writing the message is finding it out late."""
        class Blind:
            sees = False

        monkeypatch.setattr(mc_llm_chat_panel.mc_llm_runtime, "config", lambda: Blind())
        _chip, note = mc_llm_chat_panel._offer_attachment()

        assert "no vision projector" in note


class TestMiniMax:
    def test_it_builds(self):
        built = mc_llm_minimax_panel.build()

        assert set(built) == {"status", "output", "stop"}

    def test_an_empty_prompt_is_refused(self):
        events = list(mc_llm_minimax_panel._enhance("  ", "fl2va", None, 7))

        assert len(events) == 1
        assert "Write a prompt" in events[0][3]

    def test_the_structure_guide_comes_from_the_enhancer(self):
        from prompt_master.minimax import enhancer

        assert mc_llm_minimax_panel._structure("fl2va") == enhancer.infos("fl2va")
        assert (mc_llm_minimax_panel._structure(enhancer.REF2VA)
                != mc_llm_minimax_panel._structure(enhancer.FL2VA))


class TestKrea:
    """The wiring, here; the behaviour is in ``test_llm_krea.py``.

    What this file is for is the questions that are only answerable once the
    panel has been assembled against a Gradio -- that it assembles at all, that
    it hands the shell the three handles the shell wires, and that the box a
    generation streams into cannot grow and push Stop off the screen.
    """

    def test_it_builds(self):
        built = mc_llm_krea_panel.build()

        assert {"status", "output", "stop"} <= set(built)

    def test_an_empty_request_is_refused(self):
        axes = []
        for _ in library.library().axis_keys:
            axes.extend(["vary", None])
        frames = list(mc_llm_krea_panel._generate("  ", 7, False, 1, -1, False,
                                                  *axes, None, None, None, None))

        assert len(frames) == 1
        assert "Describe the image you want" in frames[0][4]

    def test_the_reference_slots_are_numbered_by_position_and_nothing_else(self):
        """Section 4 of the Krea design intent: not the filename, not the
        upload time, not the temporary path, not what the picture contains."""
        found, complaint = mc_llm_krea_panel.references(["/tmp/zzz.png", "/tmp/aaa.png",
                                                         None, None])

        assert complaint == ""
        assert [reference.ui_index for reference in found] == [1, 2]
        assert [reference.name for reference in found] == ["zzz.png", "aaa.png"]


class TestShell:
    def test_every_mode_is_its_own_view(self):
        """Section 4.1: the modes may share panels but must not be collapsed
        into one workflow. Exactly one view is visible at a time."""
        modes = len(mc_llm_studio.MODES)
        for _, chosen in mc_llm_studio.MODES:
            updates = mc_llm_studio._switch(chosen)
            visible = [update.get("visible") for update in updates[:modes]]

            assert visible.count(True) == 1

    def test_the_shell_bar_gives_way_to_conversations_own_header(self):
        """Two menus and two state chips above one transcript is one of each
        too many: Conversation draws them beside the character and the thread,
        so the shell\u2019s bar is not drawn at all while it is open."""
        modes = len(mc_llm_studio.MODES)

        opened = mc_llm_studio._switch("chat")
        elsewhere = mc_llm_studio._switch("prompt")

        assert opened[modes + 1].get("visible") is False
        assert elsewhere[modes + 1].get("visible") is True

    def test_switching_workspace_closes_the_sheet_that_chose_it(self):
        """A mode chooser still open over the mode it has just chosen is a
        control asking the question again."""
        modes = len(mc_llm_studio.MODES)
        answered = mc_llm_studio._switch("minimax")

        # The runtime line, the bar and the title, then the sheets: a name and
        # one visibility each.
        assert answered[modes + 3] == ""
        sheets = answered[modes + 4:modes + 4 + len(mc_llm_studio.SHEETS)]
        assert all(update.get("visible") is False for update in sheets)

    def test_one_sheet_is_open_at_a_time(self):
        answered = mc_llm_studio._sheet("model")
        shown = [update.get("visible") for update in answered[1:]]

        assert answered[0] == "model"
        assert shown.count(True) == 1
        assert shown[mc_llm_studio.SHEETS.index("model")] is True
        assert all(update.get("visible") is False for update in mc_llm_studio._sheet("")[1:])

    def test_the_menu_button_closes_the_sheet_it_opened(self):
        """The shell bar's menu is not covered by what it opens on a desktop,
        so a press that appears to do nothing is a press that did nothing."""
        opened = mc_llm_studio._toggle_sheet("mode")("")
        closed = mc_llm_studio._toggle_sheet("mode")("mode")

        assert opened[0] == "mode"
        assert closed[0] == ""
        assert all(update.get("visible") is False for update in closed[1:])

    def test_a_sheet_that_is_not_a_sheet_closes_everything(self):
        assert mc_llm_studio._sheet("elsewhere")[0] == ""

    def test_the_state_control_says_the_state_in_one_word(self, store, monkeypatch):
        """A button\u2019s label is text, and text is the one thing a theme
        cannot restyle into invisibility."""
        import mc_llm_runtime

        def status(configured=True, running=False):
            return {"configured": configured, "has_runtime": configured,
                    "has_model": configured, "running": running, "model": "thinker.gguf",
                    "quantization": "Q4_K_M", "device": "RTX 3090", "mode": "gpu",
                    "sees": True, "placement": None, "report": mc_llm_runtime.Report(),
                    "resident_bytes": 0}

        monkeypatch.setattr(mc_llm_runtime.runtime, "status", lambda: status(running=True))
        assert "Loaded" in mc_llm_studio._chip_label()

        monkeypatch.setattr(mc_llm_runtime.runtime, "status", lambda: status(running=False))
        assert "Unloaded" in mc_llm_studio._chip_label()

        monkeypatch.setattr(mc_llm_runtime.runtime, "status", lambda: status(configured=False))
        assert "Not set up" in mc_llm_studio._chip_label()
        assert mc_llm_studio.LOADING in mc_llm_studio._chip_label(mc_llm_studio.LOADING)

    def test_every_state_control_on_the_tab_says_the_same_thing(self, store):
        """The one in the shell bar and the one in Conversation\u2019s header
        are two controls describing one runtime."""
        said = [update.get("value") for update in mc_llm_studio._chips()]

        assert len(said) == 2 and len(set(said)) == 1

    def test_the_workspace_name_is_the_selector_s_own_label(self):
        assert "MiniMax H3" in mc_llm_studio._mode_title("minimax")
        assert "LLM Studio" in mc_llm_studio._mode_title("nothing-of-the-kind")

    def test_krea_is_a_workspace_of_its_own(self):
        """Structurally MiniMax, not a Conversation persona and not an option
        on LTX Prompt Studio: one task in, one finished Krea prompt out."""
        assert "krea" in [value for _, value in mc_llm_studio.MODES]
        assert "Krea 2" in mc_llm_studio._mode_title("krea")

    def test_krea_gets_exactly_one_workspace_view(self):
        modes = len(mc_llm_studio.MODES)
        visible = [update.get("visible") for update in mc_llm_studio._switch("krea")[:modes]]

        assert visible.count(True) == 1
        assert visible[[value for _, value in mc_llm_studio.MODES].index("krea")] is True

    def test_the_krea_workspace_is_remembered(self, store):
        import mc_llm_state

        mc_llm_studio._switch("krea")

        assert mc_llm_state.preferences()["mode"] == "krea"
        assert mc_llm_studio._initial_mode() == "krea"

    def test_setup_is_a_mode_rather_than_an_accordion(self):
        """The plain values went to the Settings page; what is left needs a
        workspace, not a footnote under whichever chat happened to be open."""
        assert "setup" in [value for _, value in mc_llm_studio.MODES]

    def test_the_chosen_mode_is_remembered(self):
        import mc_llm_state

        mc_llm_studio._switch("minimax")

        assert mc_llm_state.preferences()["mode"] == "minimax"

    def test_the_tab_opens_on_the_mode_it_was_left_on(self, store):
        """The selector was restored from preferences and the views were not:
        a tab left on Conversation opened with the selector reading
        Conversation and Prompt Studio's panel underneath it."""
        import mc_llm_state

        mc_llm_state.remember(mode="chat")
        opening = mc_llm_studio._initial_mode()

        assert opening == "chat"
        shown = [value for _, value in mc_llm_studio.MODES if value == opening]
        assert shown == ["chat"]

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


class TestTheModelChooser:
    """Switching models from the top of the tab, without a path box.

    Load and Unload are the two presses this is meant to reduce a model switch
    to; the chooser is what makes them mean anything, so what it offers is
    worth being exact about.
    """

    def _gguf(self, folder, name, size=32):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(b"GGUF" + b"\0" * size)
        return path

    def test_the_scan_offers_models_and_not_their_projectors(self, store):
        import mc_llm_files

        self._gguf(store / "models", "thinker-Q4_K_M.gguf")
        self._gguf(store / "models", "mmproj-thinker-f16.gguf")

        found = mc_llm_files.library(store / "models")

        assert [path.name for path in found.models] == ["thinker-Q4_K_M.gguf"]

    def test_only_the_first_shard_of_a_split_model_is_offered(self, store):
        import mc_llm_files

        for part in (1, 2, 3):
            self._gguf(store / "models", f"big-{part:05d}-of-00003.gguf")

        found = mc_llm_files.library(store / "models")

        assert [path.name for path in found.models] == ["big-00001-of-00003.gguf"]

    def test_it_looks_into_the_folders_a_downloader_makes(self, store):
        import mc_llm_files

        self._gguf(store / "models" / "publisher" / "repo", "one.gguf")

        found = mc_llm_files.library(store / "models")

        assert [path.name for path in found.models] == ["one.gguf"]

    def test_a_walk_stops_at_the_depth_limit(self, store, monkeypatch):
        import mc_llm_files

        monkeypatch.setattr(mc_llm_files, "MAX_LIBRARY_DEPTH", 1)
        self._gguf(store / "models" / "a" / "b" / "c", "deep.gguf")

        assert mc_llm_files.library(store / "models").models == ()

    def test_a_folder_that_is_not_there_is_an_empty_library(self, store):
        import mc_llm_files

        found = mc_llm_files.library(store / "nowhere")

        assert not found and found.models == ()

    def test_the_running_model_is_offered_even_from_outside_the_folder(self, store,
                                                                      monkeypatch):
        """A model may be recorded from anywhere on the machine. A chooser
        filled only from the scan would show nothing selected on an install
        that is working perfectly, which reads as the model having been lost."""
        elsewhere = self._gguf(store / "elsewhere", "hand-picked.gguf")
        self._gguf(store / "models", "in-the-folder.gguf")
        monkeypatch.setattr(mc_llm_studio, "_current_model", lambda: str(elsewhere))

        values = [value for _, value in mc_llm_studio._model_choices()]

        assert str(elsewhere) in values
        assert str(store / "models" / "in-the-folder.gguf") in values

    def test_a_model_is_named_by_where_it_sits_under_the_folder(self, store):
        root = store / "models"
        self._gguf(root / "publisher", "one.gguf")

        labels = [label for label, _ in mc_llm_studio._model_choices()]

        assert "publisher/one.gguf" in labels

    def test_an_empty_folder_says_which_folder_and_what_to_do(self, store):
        _, note = mc_llm_studio._rescan_models()

        assert "No .gguf files under" in note and "models folder setting" in note

    def test_choosing_a_model_without_a_runtime_names_setup(self, store):
        self._gguf(store / "models", "one.gguf")

        note = mc_llm_studio._choose_model(str(store / "models" / "one.gguf"))[0]

        assert "no llama.cpp runtime" in note and "Setup" in note

    def test_unloading_reports_rather_than_raising(self, store):
        status, residency = mc_llm_studio._unload_model()

        assert mc_llm_studio.ui.PREFIX in status
        assert mc_llm_studio.ui.PREFIX in residency


class TestTheRuntimeStateChip:
    """One word in the top bar, and the sentence it replaced kept elsewhere.

    The sentence was accurate and unreadable at a glance -- "Model: Q4_K_M ·
    Device: NVIDIA GeForce RTX 3090 (24575 MiB, 23304 MiB free) · Server:
    stopped" -- and long enough that the top bar wrapped it into ten lines and
    pushed the conversation off the bottom of the window.
    """

    def test_it_is_one_word_and_one_line(self, store):
        chip = mc_llm_studio._runtime_line()

        assert "\n" not in chip
        assert f'{mc_llm_studio.ui.PREFIX}-state' in chip
        # The visible text, with the tooltip and the markup taken out.
        import re

        visible = re.sub(r"<[^>]+>", "", re.sub(r'title="[^"]*"', "", chip)).strip()
        assert len(visible.split()) <= 3, visible

    def test_it_says_which_state_the_runtime_is_in(self, store, monkeypatch):
        import mc_llm_runtime

        def status(configured=True, running=False):
            return {"configured": configured, "has_runtime": configured, "has_model": configured,
                    "running": running, "model": "thinker.gguf", "quantization": "Q4_K_M",
                    "device": "NVIDIA GeForce RTX 3090", "mode": "gpu", "sees": True,
                    "placement": None, "report": mc_llm_runtime.Report(), "resident_bytes": 0}

        monkeypatch.setattr(mc_llm_runtime.runtime, "status", lambda: status(running=True))
        assert "Loaded" in mc_llm_studio._runtime_line()

        monkeypatch.setattr(mc_llm_runtime.runtime, "status", lambda: status(running=False))
        assert "Unloaded" in mc_llm_studio._runtime_line()

        monkeypatch.setattr(mc_llm_runtime.runtime, "status", lambda: status(configured=False))
        assert "Not set up" in mc_llm_studio._runtime_line()

    def test_the_detail_is_moved_to_the_tooltip_rather_than_dropped(self, store, monkeypatch):
        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime.runtime, "status", lambda: {
            "configured": True, "has_runtime": True, "has_model": True, "running": True,
            "model": "thinker.gguf", "quantization": "Q4_K_M", "device": "RTX 3090",
            "mode": "gpu", "sees": True, "placement": None,
            "report": mc_llm_runtime.Report(), "resident_bytes": 0})

        chip = mc_llm_studio._runtime_line()

        assert "Q4_K_M" in chip and "RTX 3090" in chip
        assert 'title="' in chip

    def test_a_model_files_metadata_cannot_escape_the_tooltip(self):
        """general.name is free text out of somebody else's file, and it lands
        in an HTML attribute now as well as in a body."""
        chip = mc_llm_studio.ui.state("Loaded", "info", '"><script>alert(1)</script>')

        assert "<script>" not in chip

    def test_load_says_loading_before_it_says_anything_else(self, store):
        """Twenty gigabytes off a disk is long enough that a button which looks
        like it did nothing gets pressed again."""
        steps = list(mc_llm_studio._load_model(progress=lambda *a, **k: None))

        assert mc_llm_studio.LOADING in steps[0][0]
        assert len(steps) >= 2

    def test_unload_with_nothing_running_reports_the_state_it_is_in(self, store):
        status, residency = mc_llm_studio._unload_model()

        assert "Unloaded" in status or "Not set up" in status
        assert mc_llm_studio.ui.PREFIX in residency

    def test_the_residency_view_still_carries_the_whole_sentence(self, store):
        """Nothing was dropped; it was moved. Setup is where it moved to."""
        assert "LLM runtime:" in mc_llm_studio._residency_html()


class TestSettingsOwnThePlainValues:
    """The five preferences the WebUI's Settings page is now the front end for.

    The file underneath them is still read -- it is what answers headless -- so
    which of the two wins, and whether a write reaches both, are the two things
    that can quietly go wrong.
    """

    def test_the_settings_page_wins_over_the_file(self, store, monkeypatch):
        import mc_llm_state
        from modules import shared

        mc_llm_state.remember(context_size=4096)
        monkeypatch.setattr(shared.opts, mc_llm_state.OPT_CONTEXT_SIZE, 65536,
                            raising=False)

        assert mc_llm_state.preferences()["context_size"] == 65536

    def test_a_label_from_a_radio_is_read_as_its_value(self, store, monkeypatch):
        import mc_llm_state
        from modules import shared

        monkeypatch.setattr(shared.opts, mc_llm_state.OPT_CONTEXT_MODE,
                            mc_llm_state.label_for_context_mode("fixed"), raising=False)

        assert mc_llm_state.preferences()["context_mode"] == "fixed"

    def test_remembering_writes_through_to_the_settings_page(self, store):
        import mc_llm_state
        from modules import shared

        mc_llm_state.remember(context_size=32768, kv_type_k="q8_0")

        assert getattr(shared.opts, mc_llm_state.OPT_CONTEXT_SIZE) == 32768
        assert mc_llm_state.preferences()["context_size"] == 32768
        assert mc_llm_state.preferences()["kv_type_k"] == "q8_0"

    def test_a_setting_the_host_has_never_heard_of_leaves_the_file_in_charge(self,
                                                                            store,
                                                                            monkeypatch):
        import mc_llm_state

        monkeypatch.setattr(mc_llm_state, "_option", lambda name: None)
        mc_llm_state.remember(context_buffer_gb=9.5)

        assert mc_llm_state.preferences()["context_buffer_gb"] == 9.5


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

    def test_the_transcript_never_takes_a_surface_from_an_accent_colour(self):
        """Reported against the Lobe theme: every message you had sent rendered
        as a solid white box with nothing in it, beside replies that were dark
        and perfectly readable.

        Gradio’s Chatbot paints the user bubble from the accent family and the
        bot bubble from the neutral surface, which is why exactly one of the two
        survived the theme. A theme is entitled to a light accent; it is not
        entitled to make half a conversation unreadable. So the transcript
        redefines those properties on this extension’s own element, and the
        replacements come from the neutral family the bot bubble already proves
        readable.
        """
        import re
        from pathlib import Path

        css = Path(__file__).resolve().parent.parent / "style.css"
        text = css.read_text(encoding="utf-8")

        block = re.search(
            r"#mc-llm-studio \.mc-llm-transcript \{[^}]*--color-accent-soft[^}]*\}", text)
        assert block, "the transcript has to neutralise the accent surface"

        body = block.group(0)
        for name in ("--color-accent-soft", "--border-color-accent-subdued"):
            replacement = re.search(rf"{re.escape(name)}\s*:\s*([^;]+);", body)
            assert replacement, name
            # Replaced with the neutral family, never with another accent and
            # never with a literal this file would have to keep in step with a
            # theme it has never seen.
            assert "accent" not in replacement.group(1)
            assert "var(--" in replacement.group(1)

        assert re.search(r"color:\s*var\(--body-text-color", body), \
            "the text colour has to be stated beside the surface, not assumed"

    def test_the_one_rule_that_names_a_gradio_class_names_no_generated_one(self):
        """The fallback for a theme that paints the bubble directly rather than
        through the variables. Gradio’s ``.message`` and ``.user`` have been
        stable across the 4.x line; a ``.svelte-`` hash is regenerated on every
        build and is the reason the rest of this file depends on none of it."""
        import re
        from pathlib import Path

        css = (Path(__file__).resolve().parent.parent / "style.css").read_text(encoding="utf-8")
        section = css.split("LLM Studio", 1)[1]

        reaching = [line.strip() for line in section.splitlines()
                    if line.strip().startswith("#mc-llm-studio")
                    and re.search(r"\.(message|user|bot|message-row)\b", line)]

        assert reaching, "the fallback rule is supposed to exist"
        for selector in reaching:
            assert ".svelte" not in selector, selector
            assert selector.startswith("#mc-llm-studio .mc-llm-"), selector

    def test_nothing_in_a_sheet_can_be_squashed_out_of_existence(self):
        """A sheet is a flex column of fixed height, so its children shrink by
        default when they do not all fit -- and a thread list as long as your
        history took that space out of the controls underneath it. What that
        looks like is a screen emptying into a blank gap, with no scrollbar,
        because from the sheet's point of view everything fitted."""
        from pathlib import Path

        css = (Path(__file__).resolve().parent.parent / "style.css").read_text(encoding="utf-8")
        rule = css.split("#mc-llm-studio .mc-llm-sheet > *", 1)

        assert len(rule) == 2, "a sheet's contents may still be shrunk"
        assert "flex: 0 0 auto" in rule[1].split("}", 1)[0]

    def test_a_sheet_is_an_overlay_rather_than_a_row_in_the_layout(self):
        """The whole responsive contract in one declaration: a surface that
        opens takes no room, so nothing it opens over can be pushed anywhere --
        least of all the composer, off the bottom of a phone."""
        from pathlib import Path

        css = (Path(__file__).resolve().parent.parent / "style.css").read_text(encoding="utf-8")
        rule = css.split("#mc-llm-studio .mc-llm-sheet {", 1)[1].split("}", 1)[0]

        assert "position: absolute" in rule
        assert "inset: 0" in rule

    def test_the_tab_root_is_the_box_the_shell_s_sheets_are_measured_against(self):
        """Otherwise they are positioned against whatever the host happens to
        have positioned further up the page, which is a different element under
        every theme."""
        from pathlib import Path

        css = (Path(__file__).resolve().parent.parent / "style.css").read_text(encoding="utf-8")
        rule = css.split("#mc-llm-studio {", 1)[1].split("}", 1)[0]

        assert "position: relative" in rule

    def test_the_conversation_workspace_is_the_box_the_sheets_are_measured_against(self):
        """An overlay is only an overlay if something near it is positioned;
        without this the sheets would be laid out against the page."""
        from pathlib import Path

        css = (Path(__file__).resolve().parent.parent / "style.css").read_text(encoding="utf-8")
        rule = css.split("#mc-llm-studio .mc-llm-chat-workspace {", 1)[1].split("}", 1)[0]

        assert "position: relative" in rule
        assert "100dvh" in rule, "the fallback should survive a phone's address bar"

    def test_no_rule_sets_display_on_anything_gradio_hides(self):
        """The bug this whole contract exists to prevent, and the one that got
        through: Gradio hides a container with a class whose rule is
        ``display: none`` at *class* specificity, so any rule of ours that
        names ``#mc-llm-studio`` and sets ``display`` outranks it and the
        hidden thing stays on screen for ever. What that looked like was a
        model sheet floating over the tab that Close would not close, a menu
        that appeared not to toggle, and no way back to Conversation.

        The list of classes at risk is read out of the panels rather than
        written down here, so a surface added tomorrow with ``visible=False``
        is covered by this test the moment it exists.
        """
        import ast
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent

        hidden = set()
        for name in ("mc_llm_chat_panel.py", "mc_llm_studio.py", "mc_llm_prompt_panel.py",
                     "mc_llm_minimax_panel.py", "mc_llm_krea_panel.py",
                     "mc_llm_browse.py"):
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                given = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                classes = given.get("elem_classes")
                # "visible" given at all, not "visible=False": a container the
                # panel decides the visibility of is a container Gradio may
                # hide, whichever way it opens.
                if "visible" not in given or not isinstance(classes, ast.Call):
                    continue
                if getattr(classes.func, "attr", "") != "classes":
                    continue
                for argument in classes.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        hidden.add(f"mc-llm-{argument.value}")

        assert "mc-llm-sheet" in hidden, "the sheets are supposed to be hideable"

        css = re.sub(r"/\*.*?\*/", "", (root / "style.css").read_text(encoding="utf-8"),
                     flags=re.S)
        for rule in re.finditer(r"([^{}]*)\{([^{}]*)\}", css):
            selector, body = rule.group(1).strip(), rule.group(2)
            if not selector.startswith("#mc-llm-studio"):
                continue
            declared = [line.strip() for line in body.splitlines()
                        if re.match(r"\s*display\s*:", line) and "none" not in line]
            if not declared:
                continue
            for name in hidden:
                assert f".{name}" not in selector, f"{selector} → {declared}"

    def test_the_escaping_helper_neutralises_metadata_from_a_model_file(self):
        """general.name is free text out of somebody else's file, and it lands
        in HTML."""
        import mc_llm_ui as ui

        assert "<script>" not in ui.notice("<script>alert(1)</script>")


class TestBoxesThatDoNotMoveTheButtons:
    """A Gradio Textbox grows from ``lines`` towards ``max_lines`` as text
    arrives. An output box that grows while a generation streams into it walks
    everything below it -- including Stop -- off the bottom of the window, at
    the one moment somebody wants to press it. So the boxes a generation writes
    into do not change size, and text longer than the box is scrolled inside
    it.
    """

    def test_the_minimax_output_does_not_grow_while_it_is_written(self):
        import mc_llm_minimax_panel

        written = mc_llm_minimax_panel.build()["output"]

        assert written.max_lines == written.lines

    def test_the_krea_output_does_not_grow_while_it_is_written(self):
        written = mc_llm_krea_panel.build()["output"]

        assert written.max_lines == written.lines

    def test_neither_prompt_studio_output_grows(self):
        import mc_llm_prompt_panel

        built = mc_llm_prompt_panel.build()

        for box in (built["positive"], built["negative"]):
            assert box.max_lines == box.lines


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

    def test_a_role_s_model_is_recorded_against_that_role(self, store, a_card):
        """The design intent's own example: a large backbone for the writer and
        a small instruction follower for the Composer."""
        import mc_llm_roles
        import mc_llm_runtime
        import mc_llm_setup

        runtime = store / mc_llm_setup.RUNTIME_DIRNAME
        runtime.mkdir(parents=True)
        server = runtime / "llama-server"
        server.write_bytes(b"")
        mc_llm_studio._apply_runtime(str(server), "gpu:0")
        big = store / "big.gguf"
        big.write_bytes(b"")
        small = store / "small.gguf"
        small.write_bytes(b"")
        mc_llm_studio._apply_model(str(big), "")

        mc_llm_studio._apply_model(str(small), "", mc_llm_roles.SPATIAL)

        assert mc_llm_runtime.config().model.name == "big.gguf"
        assert mc_llm_runtime.config(mc_llm_roles.SPATIAL).model.name == "small.gguf"
        assert mc_llm_runtime.config(mc_llm_roles.CREATIVE).model.name == "big.gguf"

    def test_applying_an_empty_path_asks_for_one(self, store):
        notice, _path, _model, _role = mc_llm_studio._apply_runtime("", None)

        assert "Enter the path" in notice

    def test_the_device_dropdown_always_offers_something(self, store):
        assert mc_llm_studio._device_choices()

    def test_each_way_of_using_a_card_is_its_own_option(self, store, a_card):
        """A card is offered twice — holding the weights, and in mixed mode —
        and two options sharing one value are one option to a dropdown. When
        they shared the card's index, picking mixed mode recorded a full
        offload of the same card and filled the VRAM it exists to keep free."""
        values = [value for _label, value in mc_llm_studio._device_choices()]

        assert len(values) == len(set(values))

    def test_the_mixed_option_resolves_to_a_mixed_device(self, store, a_card):
        chosen = mc_llm_studio._device_for("mixed:0")

        assert chosen.is_mixed

    def test_the_plain_option_resolves_to_the_card_itself(self, store, a_card):
        chosen = mc_llm_studio._device_for("gpu:0")

        assert not chosen.is_mixed and not chosen.is_cpu

    def test_an_index_written_by_an_older_build_still_means_the_card(self, store, a_card):
        chosen = mc_llm_studio._device_for("0")

        assert not chosen.is_mixed and not chosen.is_cpu

    def test_choosing_mixed_mode_records_a_mixed_install(self, store, a_card):
        import mc_llm_runtime
        import mc_llm_setup

        runtime = store / mc_llm_setup.RUNTIME_DIRNAME
        runtime.mkdir(parents=True)
        server = runtime / "llama-server"
        server.write_bytes(b"")

        mc_llm_studio._apply_runtime(str(server), "mixed:0")

        configuration = mc_llm_runtime.config()
        assert configuration.mode == "mixed_aggressive"
        assert configuration.gpu_layers == "0"

    def test_the_dropdown_comes_back_on_the_option_that_was_chosen(self, store, a_card):
        import mc_llm_setup

        runtime = store / mc_llm_setup.RUNTIME_DIRNAME
        runtime.mkdir(parents=True)
        server = runtime / "llama-server"
        server.write_bytes(b"")
        mc_llm_studio._apply_runtime(str(server), "mixed:0")

        assert mc_llm_studio._current_device() == "mixed_aggressive:0"

    def test_the_detail_line_says_when_the_weights_are_in_system_ram(self, store):
        state = {"configured": True, "has_runtime": True, "has_model": True,
                 "running": False, "model": "model.gguf", "quantization": "Q4_K_M",
                 "device": "NVIDIA GeForce RTX 3090", "mode": "mixed", "sees": False,
                 "placement": None, "report": None, "resident_bytes": 0}

        assert "mixed" in mc_llm_studio._runtime_detail(state)

    def test_the_setup_panel_says_where_the_plain_values_went(self, store):
        """A control that used to be here and is not any more is, from the
        reader\u2019s side, indistinguishable from one that was removed."""
        pointer = mc_llm_studio._settings_pointer()

        assert "Settings" in pointer and "Model Chain" in pointer
        for missing in ("Context sizing", "cache types", "residency mode"):
            assert missing in pointer


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
            mc_llm_studio._setup_panel()
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
        # What the shell wires: the state chip, the menu that opens its
        # workspace chooser, and the handles it refreshes. Setup is no longer
        # among them -- it is one of the workspaces that chooser lists.
        assert set(built) == {"status", "transcript", "header", "persona",
                              "model", "modes", "chip"}

    def test_a_cleared_box_falls_back_to_the_character_then_to_the_default(self):
        from prompt_master.chat.characters import Character

        saved = Character(name="Ada", temperature=0.4, max_reply_tokens=900)

        assert mc_llm_chat_panel._decimal(None, saved.temperature) == 0.4
        assert mc_llm_chat_panel._number("", saved.max_reply_tokens) == 900
        assert mc_llm_chat_panel._number(None, None, 512) == 512

    def test_a_slider_value_still_wins_over_the_saved_one(self):
        assert mc_llm_chat_panel._decimal(1.2, 0.4) == 1.2
        assert mc_llm_chat_panel._number(256, 900) == 256


class TestTheCharacterMenu:
    """Create, load, edit, delete -- and the one that got away.

    Reported from a real installation: "I created a new character, tried to
    switch back to existing and it wasn't an option." Save was wired to the
    *Talking to* drop-down for the name to write over, so New + Save renamed
    whichever character happened to be selected -- moving its file, taking its
    picture with it, and leaving a list that no longer had the character the
    user started from. Every test here is a way of doing that again.
    """

    @pytest.fixture
    def characters(self, store):
        from prompt_master.chat.characters import Character

        (store / "characters").mkdir(parents=True, exist_ok=True)
        held = mc_llm_chat_panel._characters()
        held.save(Character(name="Ada", context="an existing character"))
        return held

    def test_creating_one_keeps_the_one_that_was_selected(self, characters):
        editor = mc_llm_chat_panel._new_character()

        mc_llm_chat_panel._save_character(editor[1], "Grace", "someone else", "", "",
                                          0.85, 0.95, 512, -1)

        assert sorted(mc_llm_chat_panel._character_choices()) == ["Ada", "Grace"]
        assert characters.load("Ada").context == "an existing character"

    def test_new_binds_the_editor_to_nothing(self, characters):
        """The whole of the fix, asserted at the field that carries it."""
        editor = mc_llm_chat_panel._new_character()

        assert editor[1] == mc_llm_chat_panel.NOT_EDITING

    def test_a_second_save_edits_rather_than_creating_again(self, characters):
        editor = mc_llm_chat_panel._new_character()
        _dropdown, editing, _note = mc_llm_chat_panel._save_character(
            editor[1], "Grace", "first", "", "", 0.85, 0.95, 512, -1)

        mc_llm_chat_panel._save_character(editing, "Grace", "second", "", "",
                                          0.85, 0.95, 512, -1)

        assert sorted(mc_llm_chat_panel._character_choices()) == ["Ada", "Grace"]
        assert characters.load("Grace").context == "second"

    def test_creating_over_an_existing_name_is_refused(self, characters):
        """Overwriting somebody's character silently is the same loss by a
        shorter road."""
        _dropdown, _editing, note = mc_llm_chat_panel._save_character(
            "", "Ada", "hijack", "", "", 0.85, 0.95, 512, -1)

        assert "already a character" in note
        assert characters.load("Ada").context == "an existing character"

    def test_editing_binds_to_the_character_opened(self, characters):
        editor = mc_llm_chat_panel._open_character("Ada")

        assert editor[1] == "Ada"
        assert editor[2] == "Ada"

    def test_renaming_moves_the_character_rather_than_copying_it(self, characters):
        editor = mc_llm_chat_panel._open_character("Ada")

        mc_llm_chat_panel._save_character(editor[1], "Ada Lovelace", "renamed", "", "",
                                          0.85, 0.95, 512, -1)

        assert mc_llm_chat_panel._character_choices() == ["Ada Lovelace"]

    def test_deleting_lands_on_whatever_is_left(self, characters):
        from prompt_master.chat.characters import Character

        characters.save(Character(name="Grace"))

        dropdown, _editor, editing, _note = mc_llm_chat_panel._delete_character("Ada")

        assert dropdown["choices"] == ["Grace"]
        assert dropdown["value"] == "Grace"
        assert editing == mc_llm_chat_panel.NOT_EDITING

    def test_deleting_the_last_one_says_what_to_do_next(self, characters):
        dropdown, _editor, _editing, note = mc_llm_chat_panel._delete_character("Ada")

        assert dropdown["choices"] == []
        assert "press New" in note

    def test_refresh_sees_a_file_copied_in_while_the_tab_was_open(self, characters):
        from prompt_master.chat.characters import Character

        characters.save(Character(name="Hopper"))

        dropdown, _note = mc_llm_chat_panel._refresh_characters("Ada")

        assert dropdown["choices"] == ["Ada", "Hopper"]
        assert dropdown["value"] == "Ada", "refreshing must not change who you are talking to"

    def test_refresh_moves_off_a_character_that_is_no_longer_there(self, characters):
        characters.delete("Ada")

        dropdown, note = mc_llm_chat_panel._refresh_characters("Ada")

        assert dropdown["value"] is None
        assert "No characters yet" in note

    def test_cancel_puts_the_sampling_back(self, characters):
        """New resets the boxes, and they are the boxes the *conversation*
        uses -- so thinking better of it has to restore them."""
        from prompt_master.chat.characters import Character

        characters.save(Character(name="Ada", temperature=0.4, seed=99))
        mc_llm_chat_panel._new_character()

        restored = mc_llm_chat_panel._cancel_character("Ada")

        assert restored[0] == {"visible": False}
        assert restored[6] == 0.4
        assert restored[9] == 99


class TestTheSystemPromptIsVisible:
    """Asked for: "expose the current system prompt in the character view".

    It was never on screen anywhere. The override box showed what a character
    had *instead of* the built prompt, which is empty for almost every
    character and says nothing at all about what the model is actually told.
    """

    @pytest.fixture
    def characters(self, store):
        from prompt_master.chat.characters import Character

        (store / "characters").mkdir(parents=True, exist_ok=True)
        held = mc_llm_chat_panel._characters()
        held.save(Character(name="Ada", context="a mathematician"))
        return held

    def test_it_shows_the_prompt_that_would_actually_be_sent(self, characters):
        from prompt_master.chat import prompt as chat_prompt
        from prompt_master.chat.characters import Character, Persona

        shown = mc_llm_chat_panel._system_preview("Ada", "a mathematician", "")

        assert shown == chat_prompt.system_text(
            Character(name="Ada", context="a mathematician"), Persona())
        assert "You are Ada" in shown
        assert "a mathematician" in shown

    def test_it_follows_what_is_being_typed_rather_than_what_is_on_disk(self, characters):
        """The character being written is the one worth previewing."""
        shown = mc_llm_chat_panel._system_preview("Grace", "a rear admiral", "")

        assert "You are Grace" in shown
        assert "a rear admiral" in shown

    def test_an_override_is_what_it_previews_when_there_is_one(self, characters):
        shown = mc_llm_chat_panel._system_preview("Ada", "a mathematician",
                                                  "You are {{char}}. Be brief.")

        assert shown == "You are Ada. Be brief."
        assert "a mathematician" not in shown

    def test_opening_the_editor_fills_it_in(self, characters):
        opened = mc_llm_chat_panel._open_character("Ada")

        assert "You are Ada" in opened[10]

    def test_a_new_character_previews_the_bare_wrapper(self, characters):
        created = mc_llm_chat_panel._new_character()

        assert "the character" in created[10]

    def test_editing_it_copies_it_into_the_override_box(self, characters):
        """One press, because the alternative is selecting eight lines of
        read-only text and pasting them into the box underneath."""
        copied, note = mc_llm_chat_panel._adopt_system_prompt("Ada", "a mathematician", "")

        assert "You are Ada" in copied
        assert "no longer change it" in note

    def test_it_will_not_overwrite_an_override_somebody_wrote(self, characters):
        copied, note = mc_llm_chat_panel._adopt_system_prompt("Ada", "a mathematician",
                                                              "mine")

        assert copied == {}
        assert "already has a system prompt of its own" in note

    def test_what_is_copied_is_what_was_running(self, characters):
        """So saving without touching it changes nothing about how the
        character behaves -- it only stops the wrapper following the persona."""
        before = mc_llm_chat_panel._system_preview("Ada", "a mathematician", "")

        copied, _note = mc_llm_chat_panel._adopt_system_prompt("Ada", "a mathematician", "")

        assert mc_llm_chat_panel._system_preview("Ada", "a mathematician", copied) == before

    def test_it_never_raises_into_the_panel(self, monkeypatch, characters):
        monkeypatch.setattr(mc_llm_chat_panel, "_persona",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        assert mc_llm_chat_panel._system_preview("Ada", "", "") == ''

    def test_every_editor_handler_answers_the_same_shape(self, characters):
        """They share one output list, so one of them returning a different
        number of values is a panel that breaks on a button press."""
        shapes = {
            len(mc_llm_chat_panel._new_character()),
            len(mc_llm_chat_panel._open_character("Ada")),
            len(mc_llm_chat_panel._open_character("")),
            len(mc_llm_chat_panel._cancel_character("Ada")),
            len(mc_llm_chat_panel._cancel_character("")),
        }

        assert shapes == {12}


class TestSeedsAreRandomUntilSomebodyChoosesOne:
    """A seed control that opens on a fixed number is a generator that repeats.

    ``prompt_engine.options.DEFAULTS["seed"]`` is 7 because upstream's node
    needs a fixed number for its own self-tests -- it is chosen to make two
    runs identical, which is the opposite of what the box is for.
    """

    @staticmethod
    def _seed_boxes(module, monkeypatch):
        """Every control this panel builds that is labelled "Seed", as built.

        Recorded at construction rather than read off the returned handles,
        because most of these panels do not hand their controls back -- and the
        value under test is precisely the one the box *opens* on.
        """
        import gradio as gr

        found = []
        original = gr.Number

        def record(*args, **kwargs):
            made = original(*args, **kwargs)
            if str(kwargs.get("label", "")).strip().casefold() == "seed":
                found.append(made)
            return made

        monkeypatch.setattr(gr, "Number", record)
        module.build()
        return found

    def test_prompt_studio_opens_on_a_random_seed(self, store, monkeypatch):
        from prompt_master.core.models import RANDOM_SEED
        from prompt_master.prompt_engine import options as opt

        boxes = self._seed_boxes(mc_llm_prompt_panel, monkeypatch)

        assert opt.DEFAULTS["seed"] == 7, "the engine's own test default"
        assert boxes and all(box.value == RANDOM_SEED for box in boxes)

    def test_a_seed_somebody_chose_still_wins(self, store, monkeypatch):
        import mc_llm_state

        monkeypatch.setattr(mc_llm_state, "preferences",
                            lambda: {"prompt_defaults": {"seed": 4242}})

        boxes = self._seed_boxes(mc_llm_prompt_panel, monkeypatch)

        assert [box.value for box in boxes] == [4242]

    @pytest.mark.parametrize("module", ["minimax", "krea", "chat"])
    def test_every_other_mode_opens_on_one_too(self, store, monkeypatch, module):
        from prompt_master.core.models import RANDOM_SEED

        panel = {"minimax": mc_llm_minimax_panel, "krea": mc_llm_krea_panel,
                 "chat": mc_llm_chat_panel}[module]

        boxes = self._seed_boxes(panel, monkeypatch)

        assert boxes, "this panel offers no seed at all"
        assert all(box.value == RANDOM_SEED for box in boxes)

    def test_a_character_with_no_seed_draws_a_fresh_one(self):
        from prompt_master.core.models import RANDOM_SEED
        from prompt_master.chat.characters import Character

        assert Character(name="").seed == RANDOM_SEED

    def test_a_character_can_hold_a_random_seed(self, store):
        """So a character with a seed can be given one back."""
        from prompt_master.core.models import RANDOM_SEED
        from prompt_master.chat.characters import Character

        (store / "characters").mkdir(parents=True, exist_ok=True)
        held = mc_llm_chat_panel._characters()
        held.save(Character(name="Ada", seed=99))

        mc_llm_chat_panel._save_character("Ada", "Ada", "", "", "", 0.85, 0.95, 512,
                                          RANDOM_SEED)

        assert held.load("Ada").seed == RANDOM_SEED


class TestStoppingGivesTheControlsBack:
    """``cancels=`` closes the running generator where it stands, which is what
    makes Stop immediate — and a closed generator never reaches the yield that
    would have re-enabled the submit button and greyed out Stop. The run
    stopped, the partial output stayed, and the panel was left permanently busy
    with no way to ask for anything else. Whatever puts those controls back has
    to be the stop handler, because it is the only one that still runs.
    """

    def _restored(self, cancelled):
        """(submit, stop) interactivity out of a stop handler's return."""
        _status, submit, stop = cancelled
        return submit.get("interactive"), stop.get("interactive")

    def test_minimax_can_be_asked_for_another_prompt(self):
        import mc_llm_minimax_panel

        assert self._restored(mc_llm_minimax_panel._cancel(None)) == (True, False)

    def test_krea_can_be_asked_for_another_prompt(self):
        assert self._restored(mc_llm_krea_panel._cancel(None)) == (True, False)

    def test_prompt_studio_can_be_asked_for_another_prompt(self):
        import mc_llm_prompt_panel

        assert self._restored(mc_llm_prompt_panel._cancel(None)) == (True, False)

    def test_conversation_can_be_sent_another_message(self):
        assert self._restored(mc_llm_chat_panel._cancel(None)) == (True, False)

    def test_the_stop_button_is_wired_to_put_them_back(self):
        """The handler returning them is not enough on its own: the click has
        to be told where they go. Outputs of one is the bug."""
        import mc_llm_minimax_panel

        stop = mc_llm_minimax_panel.build()["stop"]
        clicks = [kwargs for kind, kwargs in stop._callbacks if kind == "click"]

        assert len(clicks) == 1
        assert len(clicks[0]["outputs"]) == 3
