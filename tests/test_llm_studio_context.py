"""LLM Studio's half of the context window: the meter's markup, the Context
section in the Conversation pop-up, and what each of its actions answers.

The meter is drawn into a hidden box the page shows on the status line
(``tests/test_llm_studio_context_js.py`` holds that half); the section's label
says how full the context is, and Trim, Compact and Clear go through the same
service commands the Forge Assistant's panel sends.
"""

from __future__ import annotations

import time

import pytest

import mc_llm_chat_panel as chat
import mc_llm_compact as compact
import mc_llm_conversation_ops as ops
import mc_llm_conversation_service as service
import mc_llm_conversation_store as store_module
import mc_llm_paths
import mc_llm_window as window_module
from test_llm_panels import _built, _by_id

READING = {"used": 4401, "limit": 8192, "room": 7232, "reserve": 960, "exact": True,
           "front": 12, "sent": 23, "messages": 35, "first": 0, "summary": None,
           "compacting": None, "updating": False}


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


def thread(store, exchanges=6, reply=600):
    from prompt_master.chat.history import ASSISTANT, ChatStore, USER

    chats = ChatStore(store / "chats")
    conversation = chats.new("Ada")
    for index in range(exchanges):
        conversation.append(USER, f"{index}: " + "where does the river go " * 6)
        conversation.append(ASSISTANT, f"{index}: " + ("past the mill " * 60)[:reply])
    chats.save(conversation)
    return conversation.identifier


class TestTheMeter:
    def test_it_says_the_numbers_and_the_whole_reading_in_its_title(self):
        drawn = chat._context_meter("Ada", "t1", reading=READING)

        assert ">4,401 / 8,192</span>" in drawn
        assert 'data-state="ok"' in drawn
        assert "23 of 35 messages sent" in drawn and "older messages leave past 7,232" in drawn

    def test_an_estimate_says_so(self):
        assert ">≈4,401 / 8,192<" in chat._context_meter("Ada", "t1",
                                                        reading=dict(READING, exact=False))

    def test_it_is_coloured_only_as_the_room_runs_out(self):
        assert [chat._context_state(dict(READING, used=used)) for used in (4401, 6600, 7300)] \
            == ["ok", "near", "full"]
        assert chat._context_state(dict(READING, compacting={"phase": "writing"})) == \
            "compacting"
        assert chat._context_state(dict(READING, compacting={"phase": "done"})) == "ok"

    def test_nothing_for_no_thread(self):
        assert chat._context_meter("", "") == ""

    def test_a_real_thread_is_read(self, store):
        identifier = thread(store)

        drawn = chat._context_meter("Ada", identifier)

        assert "/ 3,000</span>" in drawn


class TestTheSection:
    def test_its_label_says_how_full_and_whether_there_is_a_summary(self):
        assert chat._context_label(READING) == "Context · 4,401 / 8,192"
        assert chat._context_label(dict(READING, summary={"tokens": 3, "covers": 4})) == \
            "Context · 4,401 / 8,192 · summary"
        assert chat._context_label(None) == "Context"

    def test_keep_the_latest_offers_the_context_and_starts_at_three_fifths_of_it(self):
        assert chat._keep_slider(READING) == {"minimum": 0, "maximum": 8100, "step": 100,
                                              "value": 4900}

    def test_trim_keeps_the_latest_and_says_how_many(self, store, monkeypatch):
        identifier = thread(store)
        monkeypatch.setattr("mc_llm_cache_update.warm", lambda: False)

        drawn = chat._context_trim("Ada", identifier, 600)

        front = window_module.for_thread("Ada", identifier).front
        assert front > 0
        assert drawn[0]["label"].startswith("Context · ")
        note = drawn[-1]
        assert f"Sending the latest {12 - front} of 12 messages" in note
        assert drawn[2]["value"] == 600, "the slider keeps what was chosen"

    def test_clear_puts_it_back(self, store):
        identifier = thread(store)
        window_module.set_summary("Ada", identifier, "They met.", 4, 3)

        drawn = chat._context_clear("Ada", identifier, 600)

        assert window_module.for_thread("Ada", identifier) == window_module.Window(
            ratio=window_module.for_thread("Ada", identifier).ratio)
        assert "newest messages that fit again" in drawn[-1]

    def test_with_no_thread_it_says_so(self):
        assert "Choose a character and a thread first." in chat._context_trim("", "", 100)[-1]

    def test_compact_follows_the_summary_until_it_is_written(self, store, monkeypatch):
        import mc_broker
        import mc_llm_sessions as sessions

        identifier = thread(store)
        monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.02)
        monkeypatch.setattr(sessions._Gpu, "_domain",
                            lambda self: mc_broker.UNKNOWN_CUDA_EXECUTION)
        monkeypatch.setattr(sessions._Gpu, "_blocked_by_the_image_job",
                            staticmethod(lambda domain: False))
        monkeypatch.setattr(sessions._Gpu, "_blocked_by_a_guest",
                            staticmethod(lambda domain, ours: ""))

        class Writer:
            def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None, **kw):
                time.sleep(0.6)
                on_text("They walked.")
                return "They walked."

        monkeypatch.setattr(sessions, "_client", lambda needs_vision, **kwargs: Writer())
        monkeypatch.setattr(chat.time if hasattr(chat, "time") else time, "sleep",
                            time.sleep)

        steps = list(chat._context_compact("Ada", identifier, 600, 200))

        assert "Compacting" in steps[0][-1] and "mc-llm-busy" in steps[0][-1]
        assert len(steps) >= 3, "it followed the summary while it was written"
        assert "Compacted" in steps[-1][-1]
        assert window_module.for_thread("Ada", identifier).summary == "They walked."
        assert steps[-1][0]["label"].endswith("· summary")


class TestTheWiring:
    def test_the_actions_are_wired_to_their_handlers(self, store, monkeypatch):
        built = _built(monkeypatch)

        wired = {name: [kwargs["fn"].__name__ for kind, kwargs
                        in _by_id(built, name)._callbacks if kind == "click"]
                 for name in ("context-trim", "context-compact", "context-clear",
                              "context-cancel", "context-now")}

        assert wired == {"context-trim": ["_context_trim"],
                         "context-compact": ["_context_compact"],
                         "context-clear": ["_context_clear"],
                         "context-cancel": ["_context_cancel"],
                         "context-now": ["_context_meter"]}

    def test_the_pop_up_draws_it_again_as_it_opens(self, store, monkeypatch):
        built = _built(monkeypatch)

        opened = [kwargs["fn"].__name__ for kind, kwargs
                  in _by_id(built, "to-conversation")._callbacks if kind == "click"]

        assert "_context_view" in opened

    def test_and_when_the_thread_changes_under_the_open_pop_up(self, store, monkeypatch):
        built = _built(monkeypatch)
        section = _by_id(built, "context-section")
        delete = next(one for one in built if getattr(one, "value", None)
                      == "Delete this thread")

        for changed in (_by_id(built, "who"), delete):
            redrawn = [chained.kwargs["fn"] for dependency in changed._dependencies
                       for chained in dependency.chained
                       if section in (chained.kwargs.get("outputs") or [])]
            assert redrawn == [chat._context_view], changed

    def test_the_meters_box_is_hidden_and_its_bridge_is_a_hidden_button(self, store,
                                                                         monkeypatch):
        built = _built(monkeypatch)

        assert _by_id(built, "context").visible is False
        assert _by_id(built, "context-now").visible is False
