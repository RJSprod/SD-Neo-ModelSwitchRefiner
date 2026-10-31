"""Start reply with: the words the character's very next reply begins with.

The ask: Oobabooga's *Start reply with*, for the next reply alone -- "set the
beginning response to be the color is blue ... ask the LLM, what color are my
red shoes? And it would be forced to start off with the color is blue" -- from a
button beside the others. The start is the reply's own first words: shown,
spoken and saved; the model carries on from them as it wrote, and the start is
spent by the reply that begins with it.

``mc_llm_sessions.conversation`` is replaced with a list of events where the
reply's own runtime is not the point, as in ``test_character_emotion``; where it
is, the real runtime streams from a client that sends the prefill back at the
front, as llama-server b10621 does.
"""

from __future__ import annotations

import pytest

import mc_llm_conversation_feed as feed
import mc_llm_conversation_service as service
import mc_llm_emotions as emotions
import mc_llm_sessions as sessions
import mc_llm_start_with as start_with
from test_character_emotion import (HAPPY, Echoing, asking, choose, serving,  # noqa: F401
                                    updates)
from test_conversation_ops import chats, run, store, thread  # noqa: F401 (fixtures)
from test_conversation_service import envelope

BLUE = "The color is blue"


def write(conversation, text, operation_id="start-1"):
    body = envelope(service.SET_START_WITH, conversation, revision=None,
                    payload={"start_with": text}, operation_id=operation_id)
    return service.submit(body)


def details(subscriber, cursor, kind=service.CONVERSATION_CHANGED):
    return [event["payload"].get("kind_detail") for event in subscriber.since(cursor)[0]
            if event["kind"] == kind]


class TestTheStore:
    def test_a_thread_has_none_until_one_is_written(self, store):
        assert start_with.for_thread("Ada", "one") == ""
        assert not start_with.path().exists()

    def test_it_is_kept_for_its_thread_with_its_ends_trimmed_and_its_lines_kept(self, store):
        assert start_with.write("Ada", "one", "  The color is blue.\nAnd  ") == \
            "The color is blue.\nAnd"
        assert start_with.for_thread("Ada", "one") == "The color is blue.\nAnd"
        assert start_with.for_thread("Ada", "two") == ""

    def test_empty_clears_it(self, store):
        start_with.write("Ada", "one", BLUE)
        start_with.write("Ada", "one", "   ")
        assert start_with.for_thread("Ada", "one") == ""

    def test_a_long_one_is_refused_in_a_sentence(self, store):
        with pytest.raises(start_with.Refused, match="characters at most"):
            start_with.write("Ada", "one", "x" * (start_with.MAX_LENGTH + 1))

    def test_spending_clears_only_the_start_it_was_given(self, store):
        start_with.write("Ada", "one", BLUE)
        start_with.write("Ada", "one", "Written since")

        assert start_with.spend("Ada", "one", BLUE) is False
        assert start_with.for_thread("Ada", "one") == "Written since"
        assert start_with.spend("Ada", "one", "Written since") is True
        assert start_with.for_thread("Ada", "one") == ""

    def test_an_unreadable_file_reads_as_none(self, store):
        start_with.path().write_text("{not json", encoding="utf-8")
        assert start_with.for_thread("Ada", "one") == ""


class TestTheCommand:
    def test_it_needs_no_revision_writes_nothing_in_the_thread_and_is_in_the_snapshot(
            self, chats, thread):
        before = chats.load("Ada", thread.identifier).revision
        cursor = feed.cursor()
        subscriber = feed.subscribe("page-1")

        answer = write(thread, BLUE)

        assert answer["ok"] and answer["start_with"] == BLUE
        assert chats.load("Ada", thread.identifier).revision == before
        assert service.snapshot("Ada", thread.identifier)["conversation"]["start_with"] == BLUE
        assert "start_with" in details(subscriber, cursor)

    def test_a_long_one_is_refused_and_nothing_kept(self, thread):
        outcome = write(thread, "x" * (start_with.MAX_LENGTH + 1))

        assert outcome["ok"] is False
        assert outcome["error"]["code"] == service.INVALID_INPUT
        assert start_with.for_thread("Ada", thread.identifier) == ""

    def test_a_deleted_thread_forgets_it(self, chats, thread):
        write(thread, BLUE)
        token = chats.load("Ada", thread.identifier).revision or service.store_module.token(
            chats, service.Key("Ada", thread.identifier))
        body = envelope(service.DELETE_THREAD, thread, revision=token,
                        payload={"confirm_title": thread.title}, operation_id="gone-1")
        service.submit(body)

        assert start_with.for_thread("Ada", thread.identifier) == ""


class TestTheNextReply:
    def test_it_starts_from_the_words_and_they_are_the_reply_s_own(self, chats, thread,
                                                                  monkeypatch):
        asked = asking(monkeypatch, pieces=(",", " like the sky."))
        write(thread, BLUE)

        run("send", thread, payload={"text": "What color are my red shoes?"},
            operation_id="send-1")

        assert asked[-1].messages[-1] == {"role": "assistant", "content": BLUE}
        assert asked[-1].prefilled is True and asked[-1].start_with == BLUE
        saved = chats.load("Ada", thread.identifier)
        assert saved.messages[-1].text == "The color is blue, like the sky."

    def test_what_the_model_wrote_after_them_is_kept_as_written(self, chats, thread,
                                                                monkeypatch):
        asking(monkeypatch, pieces=(" because I said so.",))
        write(thread, BLUE)

        run("send", thread, payload={"text": "Why?"}, operation_id="send-1")

        assert chats.load("Ada", thread.identifier).messages[-1].text == \
            "The color is blue because I said so."

    def test_it_is_spent_by_the_reply_and_both_views_are_told(self, thread, monkeypatch):
        asking(monkeypatch)
        write(thread, BLUE)
        cursor = feed.cursor()
        subscriber = feed.subscribe("page-1")

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        assert start_with.for_thread("Ada", thread.identifier) == ""
        assert "start_with" in details(subscriber, cursor)

    def test_the_reply_after_it_starts_freely(self, thread, monkeypatch):
        asked = asking(monkeypatch)
        write(thread, BLUE)
        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        run("send", thread, payload={"text": "And now?"}, operation_id="send-2")

        assert asked[-1].messages[-1]["role"] == "user"
        assert asked[-1].prefilled is False and asked[-1].start_with == ""

    def test_with_an_emotion_on_both_start_the_turn_and_only_the_words_are_kept(
            self, chats, thread, monkeypatch):
        asked = asking(monkeypatch, pieces=(", see?",))
        choose(thread, HAPPY)
        write(thread, BLUE)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        assert asked[-1].messages[-1] == {"role": "assistant", "content": f"{HAPPY} {BLUE}"}
        assert chats.load("Ada", thread.identifier).messages[-1].text == \
            "The color is blue, see?"

    def test_regenerate_starts_from_it_too(self, chats, thread, monkeypatch):
        asked = asking(monkeypatch, pieces=(".",))
        write(thread, BLUE)

        run("regenerate", thread, index=len(thread.messages) - 1, operation_id="again-1")

        assert asked[-1].messages[-1] == {"role": "assistant", "content": BLUE}
        assert chats.load("Ada", thread.identifier).messages[-1].text == "The color is blue."
        assert start_with.for_thread("Ada", thread.identifier) == ""

    def test_regenerating_an_earlier_reply_uses_the_start_written_where_it_was_pressed(
            self, chats, thread, monkeypatch):
        """That reply is regenerated in a branch, a thread of its own: the start
        is the one written in the thread the press was made in, and is spent there."""
        asked = asking(monkeypatch, pieces=(".",))
        write(thread, BLUE)

        run("regenerate", thread, index=1, operation_id="again-1")

        assert asked[-1].messages[-1] == {"role": "assistant", "content": BLUE}
        branch = [row.identifier for row in chats.listing("Ada")
                  if row.identifier != thread.identifier]
        assert len(branch) == 1
        assert chats.load("Ada", branch[0]).messages[-1].text == "The color is blue."
        assert start_with.for_thread("Ada", thread.identifier) == ""

    def test_continue_neither_uses_it_nor_spends_it(self, thread, monkeypatch):
        asked = asking(monkeypatch)
        write(thread, BLUE)

        run("continue", thread, index=len(thread.messages) - 1, operation_id="more-1")

        assert asked[-1].prefilled is False
        assert all(message.get("content") != BLUE for message in asked[-1].messages)
        assert start_with.for_thread("Ada", thread.identifier) == BLUE

    def test_a_refused_send_keeps_it_for_the_next(self, thread, monkeypatch):
        asking(monkeypatch)
        write(thread, BLUE)
        body = envelope(service.SEND, thread, revision=12345,
                        payload={"text": "Hi."}, operation_id="stale-1")

        outcome = service.submit(body)

        assert outcome["ok"] is False
        assert outcome["error"]["code"] == service.STALE_REVISION
        assert start_with.for_thread("Ada", thread.identifier) == BLUE

    def test_one_refused_once_the_request_is_built_keeps_it_too(self, thread, monkeypatch):
        import mc_llm_conversation_ops as ops
        import prompt_master.chat.prompt as prompt

        asked = asking(monkeypatch)
        write(thread, BLUE)
        monkeypatch.setattr(prompt, "needs_vision", lambda wire: True)
        monkeypatch.setattr(ops, "_sees", lambda: False)

        outcome = run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        assert outcome["ok"] is False
        assert outcome["error"]["code"] == service.VISION_UNAVAILABLE
        assert asked == []
        assert start_with.for_thread("Ada", thread.identifier) == BLUE

    def test_the_cache_is_read_again_after_it(self, thread, monkeypatch, updates):
        asking(monkeypatch, pieces=(".",))
        write(thread, BLUE)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        import test_character_emotion

        assert test_character_emotion.settled()
        assert len(updates) == 1
        assert updates[0].messages[-1] == {"role": "assistant",
                                           "content": "The color is blue."}

    def test_it_is_spoken_as_the_reply_s_first_words(self, thread, monkeypatch):
        import mc_voice_ui

        said = []

        class Turn:
            id = "turn-1"

            def add_text(self, text):
                said.append(("text", text))

            def complete(self, tail):
                said.append(("complete", tail))

        openings = []
        monkeypatch.setattr(mc_voice_ui, "begin_speech",
                            lambda **kwargs: openings.append(kwargs["opening"]) or Turn())
        asking(monkeypatch, pieces=(",", " like the sky."))
        write(thread, BLUE)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        assert openings == [""]
        assert said[0] == ("text", BLUE)
        assert [text for kind, text in said if kind == "text"] == [BLUE, ",", " like the sky."]
        assert said[-1] == ("complete", "The color is blue, like the sky.")


class TestThroughTheRuntime:
    def test_the_words_sent_back_are_put_back_once_and_the_rest_kept_as_written(
            self, chats, thread, serving):
        serving(Echoing([BLUE + " because", " I said so."]))
        write(thread, BLUE)

        run("send", thread, payload={"text": "Why?"}, operation_id="send-1")

        assert chats.load("Ada", thread.identifier).messages[-1].text == \
            "The color is blue because I said so."

    def test_with_an_emotion_only_the_emotion_is_left_out(self, chats, thread, serving):
        serving(Echoing([f"{HAPPY} {BLUE},", " see?"]))
        choose(thread, HAPPY)
        write(thread, BLUE)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        assert chats.load("Ada", thread.identifier).messages[-1].text == \
            "The color is blue, see?"

    def test_the_stream_keeps_the_model_s_first_space(self, serving):
        serving(Echoing([BLUE + " because", " I said so."]))
        request = sessions.ChatRequest(
            messages=[{"role": "user", "content": "Why?"},
                      {"role": "assistant", "content": BLUE}],
            conversation="Ada/one", prefilled=True, start_with=BLUE)

        events = list(sessions._conversation(request, sessions.Cancellation()))

        assert [event.text for event in events if event.kind == sessions.CHUNK] == \
            [" because", " I said so."]
        assert events[-1].text == " because I said so."


class TestTheStripKeepsTheSpaceAfterAStart:
    def test_the_space_after_the_line_is_kept(self):
        reply = emotions.Unechoed(BLUE, trim=False)
        assert [reply.feed(BLUE), reply.feed(" because"), reply.finish()] == \
            ["", " because", ""]
        assert emotions.without(BLUE, BLUE + " because", trim=False) == " because"
        assert emotions.without(BLUE, BLUE + ", see", trim=False) == ", see"


class TestTheTab:
    @staticmethod
    def panel():
        import mc_llm_chat_panel

        return mc_llm_chat_panel

    def test_the_heading_marks_whether_the_chip_s_dot_is_filled(self, thread):
        panel = self.panel()

        assert 'data-start="off"' in panel._heading("Ada", thread)
        write(thread, BLUE)
        assert 'data-start="on"' in panel._heading("Ada", thread)

    def test_opening_the_box_shows_what_is_kept(self, thread):
        write(thread, BLUE)

        title, text, heading = self.panel()._start_view("Ada", thread.identifier)

        assert text == BLUE
        assert "Start the next reply with" in title
        assert 'data-start="on"' in heading

    def test_set_keeps_the_words_through_the_service_and_says_so(self, chats, thread):
        before = chats.load("Ada", thread.identifier).revision

        title, text, heading, status = self.panel()._start_written(
            "Ada", thread.identifier, BLUE)

        assert start_with.for_thread("Ada", thread.identifier) == BLUE
        assert text == BLUE and 'data-start="on"' in heading
        assert "starts with your words" in status
        assert chats.load("Ada", thread.identifier).revision == before

    def test_clear_empties_it(self, thread):
        write(thread, BLUE)

        _, text, heading, status = self.panel()._start_written("Ada", thread.identifier, "")

        assert start_with.for_thread("Ada", thread.identifier) == ""
        assert text == "" and 'data-start="off"' in heading
        assert "starts freely" in status

    def test_a_refused_start_is_said_and_the_box_kept(self, thread):
        _, text, _, status = self.panel()._start_written(
            "Ada", thread.identifier, "x" * (start_with.MAX_LENGTH + 1))

        assert start_with.for_thread("Ada", thread.identifier) == ""
        assert text != "" and "characters at most" in status

    def test_with_no_thread_nothing_is_kept(self, store):
        _, _, _, status = self.panel()._start_written("", "", BLUE)

        assert "Choose a character and a thread first." in status
        assert not start_with.path().exists()

    def test_the_title_is_written_escaped(self, chats, thread):
        conversation = chats.load("Ada", thread.identifier)
        conversation.title = "<b>bold</b>"

        assert "<b>" not in self.panel()._start_title(conversation)
