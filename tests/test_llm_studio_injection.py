"""LLM Studio's Injection section: the thread's words.

"I like all three options, give it a menu for 'Injection'" (2026-11-09); then,
of the third, "No vector support". The section in the Conversation pop-up keeps
the thread's words when Save or a switch is pressed -- a switch keeps the box
with it, so nothing typed is lost to a press -- and says how each word reads.
``tests/live/llm_studio_conversation_live.py`` drives it on a real Gradio 4.40
page; these hold the handlers and the wiring.
"""

from __future__ import annotations

import pytest

import mc_llm_chat_panel as chat
import mc_llm_compact as compact
import mc_llm_conversation_ops as ops
import mc_llm_conversation_service as service
import mc_llm_conversation_store as store_module
import mc_llm_injection as injection
import mc_llm_paths
from test_llm_panels import _built, _by_id


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(ops, "_context_size", lambda: 3000)
    service.reset()
    store_module.registry.reset()
    ops.reset()
    compact.reset()
    from prompt_master.chat.characters import Character, CharacterStore

    CharacterStore(tmp_path / "characters").save(Character(name="Ada",
                                                           context="a reader of maps"))
    yield tmp_path
    compact.reset()
    ops.reset()
    service.reset()
    store_module.registry.reset()


def thread(store):
    from prompt_master.chat.history import ASSISTANT, ChatStore, USER

    chats = ChatStore(store / "chats")
    conversation = chats.new("Ada")
    conversation.append(USER, "where does the river go")
    conversation.append(ASSISTANT, "past the mill")
    chats.save(conversation)
    return conversation.identifier


class TestTheWords:
    def test_save_keeps_the_box_and_both_switches_and_says_what_is_kept(self, store):
        identifier = thread(store)

        drawn = chat._injection_saved("Ada", identifier, "rain +2\ndelve ban", True, False)

        assert injection.for_thread("Ada", identifier) == injection.Record(
            "rain +2\ndelve ban", True, False)
        assert drawn[0]["label"] == "Injection · 2 words · odds"
        assert drawn[1:4] == ["rain +2\ndelve ban", True, False]
        assert "2 words kept, weighing the odds." in drawn[-1]

    def test_a_switch_pressed_keeps_what_was_typed_with_it(self, store):
        identifier = thread(store)

        chat._injection_saved("Ada", identifier, "rain", True, True)

        assert injection.for_thread("Ada", identifier).words == "rain"
        assert injection.for_thread("Ada", identifier).prompt is True

    def test_with_no_thread_it_asks_for_one(self, store):
        drawn = chat._injection_saved("Ada", "", "rain", True, False)

        assert "Choose a character and a thread first." in drawn[-1]

    def test_each_word_is_listed_escaped(self, store):
        identifier = thread(store)
        injection.save("Ada", identifier, words="<b>rain</b> +2")

        shown = chat._injection_read("Ada", identifier)

        assert "&lt;b&gt;rain&lt;/b&gt;" in shown and "<b>rain" not in shown
        assert "+2" in shown
        assert "once the model is running" in shown, "no server here to read it with"

    def test_the_label_says_the_words_and_the_ways_and_nothing_else(self, store):
        identifier = thread(store)
        injection.save("Ada", identifier, words="rain")

        assert chat._injection_label("Ada", identifier) == "Injection · 1 word · odds"


class TestTheWiring:
    def test_save_and_both_switches_keep_the_words(self, store, monkeypatch):
        built = _built(monkeypatch)

        def handlers(name, kind):
            return [kwargs["fn"].__name__ for found, kwargs in _by_id(built, name)._callbacks
                    if found == kind]

        assert handlers("injection-save", "click") == ["_injection_saved"]
        assert handlers("injection-odds", "input") == ["_injection_saved"]
        assert handlers("injection-prompt", "input") == ["_injection_saved"]

    def test_a_switch_listens_to_the_press_not_to_the_section_being_drawn(self, store,
                                                                         monkeypatch):
        built = _built(monkeypatch)

        for name in ("injection-odds", "injection-prompt"):
            kinds = {kind for kind, _ in _by_id(built, name)._callbacks}
            assert "change" not in kinds, name

    def test_a_change_of_the_note_says_the_read_again_on_the_status_line(self, store,
                                                                         monkeypatch):
        built = _built(monkeypatch)

        for name in ("injection-save", "injection-odds", "injection-prompt"):
            chained = [dependency.kwargs["fn"] for found in _by_id(built, name)._dependencies
                       for dependency in found.chained]
            assert chat._cache_status in chained, name

    def test_the_section_is_drawn_again_as_the_pop_up_opens_and_the_thread_changes(
            self, store, monkeypatch):
        built = _built(monkeypatch)
        section = _by_id(built, "injection")
        opened = [kwargs["fn"].__name__ for kind, kwargs
                  in _by_id(built, "to-conversation")._callbacks if kind == "click"]
        assert "_injection_view" in opened

        delete = next(one for one in built if getattr(one, "value", None)
                      == "Delete this thread")
        for changed in (_by_id(built, "who"), delete):
            redrawn = [chained.kwargs["fn"] for dependency in changed._dependencies
                       for chained in dependency.chained
                       if section in (chained.kwargs.get("outputs") or [])]
            assert redrawn == [chat._injection_view], changed

    def test_no_control_of_it_is_restored_from_forges_ui_config(self, store, monkeypatch):
        built = _built(monkeypatch)

        for name in ("injection-words", "injection-odds", "injection-prompt"):
            assert getattr(_by_id(built, name), "do_not_save_to_config", False) is True, name

    def test_the_section_holds_no_control_vector(self, store, monkeypatch):
        """2026-11-09, of the control vector built beside the words: "No vector
        support". Nothing in the section chooses, strengthens or adds one."""
        built = _built(monkeypatch)

        named = [getattr(one, "elem_id", None) or "" for one in built]
        assert not [name for name in named if "vector" in name]
        drawn = chat._injection_view("Ada", "")
        assert len(drawn) == 5, "the label, the box, the two ways and the read-back"
