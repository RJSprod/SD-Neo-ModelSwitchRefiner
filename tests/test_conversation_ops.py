"""Replies the server owns: the two data-loss fixes, and the completion guard.

Two of these tests are about specification S8, and both are defects that are
live at HEAD rather than hypotheses:

    "Send again from here" called ``truncate_after(index)`` and then ``save()``.
    Every message after the selected one, permanently deleted. It is the same
    defect Regenerate was fixed for and the fix was never applied here.

    "Continue" on an earlier reply carried on writing into a message that had
    descendants under it, so the replies below went on answering a paragraph
    that no longer said what they were answering.

The rest are about ownership. A reply lives on a service thread now, so closing
the generator that was drawing it costs a subscription; a reply that finishes
after the conversation has moved is kept for the person rather than written over
whatever arrived; and two replies finishing at once cannot take each other's
completion, which one process-global slot could and did.

``mc_llm_sessions.conversation`` is replaced with a list of events. That is the
only thing replaced: the prompt is built by the real builder, the request is a
real ``ChatRequest``, and the saving is the real guarded transaction.
"""

from __future__ import annotations

import threading

import pytest

import mc_llm_conversation_feed as feed
import mc_llm_conversation_ops as ops
import mc_llm_conversation_service as service
import mc_llm_conversation_store as store_module
import mc_llm_paths
import mc_llm_sessions as sessions


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    service.reset()
    store_module.registry.reset()
    ops.reset()
    feed.reset()
    from prompt_master.chat.characters import Character, CharacterStore

    CharacterStore(tmp_path / "characters").save(Character(name="Ada",
                                                           context="a reader of maps"))
    yield tmp_path
    ops.reset()
    service.reset()
    store_module.registry.reset()
    feed.reset()


@pytest.fixture
def chats(store):
    from prompt_master.chat.history import ChatStore

    return ChatStore(store / "chats")


@pytest.fixture
def thread(chats):
    from prompt_master.chat.history import ASSISTANT, USER

    found = chats.new("Ada")
    for index in range(3):
        found.append(USER, f"ask {index}")
        found.append(ASSISTANT, f"reply {index}")
    chats.save(found)
    return found


def replies(monkeypatch, pieces=("A new", " reply"), terminal=sessions.DONE):
    events = [sessions.Event(sessions.CHUNK, piece) for piece in pieces]
    events.append(sessions.Event(terminal, "".join(pieces)))
    monkeypatch.setattr(sessions, "conversation", lambda request, cancel: iter(events))
    return events


def run(action, conversation, *, index=None, payload=None, operation_id="op-1"):
    from test_conversation_service import envelope

    outcome = service.submit(envelope(action, conversation, index=index,
                                      payload=payload, operation_id=operation_id))
    assert ops.drain(timeout=10), "a reply did not finish on its own thread"
    return outcome


class TestWhichPicturesAReplyIsShown:
    """The reply operation reads the *Show the model every picture* setting
    at the moment of the reply and hands it to both the reader that loads the
    stills and the builder that keeps them, so the two agree."""

    def _pictured(self, chats, store):
        import mc_llm_attachments
        from PIL import Image
        from prompt_master.chat.history import ASSISTANT, USER

        found = chats.new("Ada")
        for index in range(3):
            found.append(USER, f"look {index}")
            found.messages[-1].image_path = mc_llm_attachments.store(
                Image.new("RGB", (32, 24), (index * 40, 90, 200)), "Ada")
            found.append(ASSISTANT, f"reply {index}")
        chats.save(found)
        return found

    @staticmethod
    def _stills(request) -> int:
        return sum(1 for message in request.messages
                   if isinstance(message.get("content"), list)
                   and any(part.get("type") == "image_url" for part in message["content"]))

    def test_the_newest_picture_alone_by_default(self, chats, store, monkeypatch):
        asked = []
        events = [sessions.Event(sessions.CHUNK, "ok"), sessions.Event(sessions.DONE, "ok")]
        monkeypatch.setattr(sessions, "conversation",
                            lambda request, cancel: (asked.append(request), iter(events))[1])
        monkeypatch.setattr(ops, "_sees", lambda: True)
        thread = self._pictured(chats, store)

        outcome = run("send", thread, payload={"text": "and now?"}, operation_id="op-one")

        assert outcome["ok"] is True, outcome
        assert self._stills(asked[-1]) == 1

    def test_every_picture_when_the_setting_asks(self, chats, store, monkeypatch):
        import mc_llm_vision

        asked = []
        events = [sessions.Event(sessions.CHUNK, "ok"), sessions.Event(sessions.DONE, "ok")]
        monkeypatch.setattr(sessions, "conversation",
                            lambda request, cancel: (asked.append(request), iter(events))[1])
        monkeypatch.setattr(mc_llm_vision, "every_picture", lambda: True)
        monkeypatch.setattr(ops, "_sees", lambda: True)
        thread = self._pictured(chats, store)

        outcome = run("send", thread, payload={"text": "and now?"}, operation_id="op-all")

        assert outcome["ok"] is True, outcome
        assert self._stills(asked[-1]) == 3
        assert asked[-1].needs_vision is True


class TestSendAgainFromHereBranches:
    """Specification S8, first half. The original thread keeps every word.

    Translated from the assertions that recorded the old behaviour: they used
    to say the thread was truncated, which is the defect. They say the thread is
    intact and the answer is in a branch, which is the fix.
    """

    def test_the_thread_it_came_from_is_untouched(self, chats, thread, monkeypatch):
        replies(monkeypatch)

        run("resend_from_user", thread, index=2)

        kept = chats.load("Ada", thread.identifier)
        assert [message.text for message in kept.messages] == [
            "ask 0", "reply 0", "ask 1", "reply 1", "ask 2", "reply 2"]

    def test_the_new_answer_is_written_in_a_branch(self, chats, thread, monkeypatch):
        replies(monkeypatch)

        outcome = run("resend_from_user", thread, index=2)

        made = outcome["resulting_conversation"]["thread_id"]
        assert made != thread.identifier
        assert [message.text for message in chats.load("Ada", made).messages] == [
            "ask 0", "reply 0", "ask 1", "A new reply"]

    def test_it_applies_to_your_own_message_only(self, thread, monkeypatch):
        replies(monkeypatch)

        outcome = run("resend_from_user", thread, index=1)

        assert outcome["ok"] is False
        assert outcome["error"]["code"] == service.INVALID_INPUT


@pytest.fixture
def unanswered(chats, thread):
    """A thread whose last reply was deleted: yours is the last word.

    The state the flyout's SEND AGAIN is for, and the one a stopped or failed
    reply leaves behind as well.
    """
    thread.delete(len(thread.messages) - 1)
    chats.save(thread)
    return chats.load("Ada", thread.identifier)


class TestSendingAgainTheLastMessageAnswersInPlace:
    """Nothing follows the last message, so there is nothing for a branch to
    protect -- and a copy of the whole thread to answer it was a second thread
    nobody asked for, holding the conversation somebody was in the middle of.
    "SEND AGAIN will take the latest response ... and send it again to continue
    the conversation."
    """

    def test_the_reply_lands_in_the_same_thread(self, chats, unanswered, monkeypatch):
        replies(monkeypatch)

        outcome = run("resend_from_user", unanswered, index=4)

        assert outcome["ok"] is True
        assert outcome["resulting_conversation"]["thread_id"] == unanswered.identifier
        assert [message.text for message in
                chats.load("Ada", unanswered.identifier).messages] == [
            "ask 0", "reply 0", "ask 1", "reply 1", "ask 2", "A new reply"]

    def test_no_second_thread_is_made(self, chats, unanswered, monkeypatch):
        replies(monkeypatch)
        before = {entry.identifier for entry in chats.listing("Ada")}

        run("resend_from_user", unanswered, index=4)

        assert {entry.identifier for entry in chats.listing("Ada")} == before

    def test_your_message_is_not_written_twice(self, chats, unanswered, monkeypatch):
        """Send appends your words and then the answer. This has your words
        already, so the acceptance writes nothing and the thread gains exactly
        one message: the reply."""
        replies(monkeypatch)

        outcome = run("resend_from_user", unanswered, index=4)

        assert outcome["persisted"] is False
        texts = [message.text for message in
                 chats.load("Ada", unanswered.identifier).messages]
        assert texts.count("ask 2") == 1

    def test_the_message_as_it_reads_now_is_what_is_answered(self, chats, unanswered,
                                                              monkeypatch):
        """Edited first, then sent again: the model is asked the edited words."""
        asked = []
        events = [sessions.Event(sessions.CHUNK, "ok"), sessions.Event(sessions.DONE, "ok")]

        def conversation(request, cancel):
            asked.append(request)
            return iter(events)

        monkeypatch.setattr(sessions, "conversation", conversation)
        run("edit_message", unanswered, index=4, payload={"text": "ask 2, edited"},
            operation_id="op-edit")
        edited = chats.load("Ada", unanswered.identifier)

        run("resend_from_user", edited, index=4, operation_id="op-again")

        last = asked[-1].messages[-1]
        spoken = last["content"] if isinstance(last["content"], str) else str(last["content"])
        assert last["role"] == "user"
        assert "ask 2, edited" in spoken

    def test_a_message_with_replies_after_it_still_branches(self, chats, thread,
                                                             monkeypatch):
        """Only the last message is answered in place. One further up has
        replies under it, and those are what the branch exists to keep."""
        replies(monkeypatch)

        outcome = run("resend_from_user", thread, index=4)

        assert outcome["resulting_conversation"]["thread_id"] != thread.identifier
        assert [message.text for message in
                chats.load("Ada", thread.identifier).messages][-1] == "reply 2"


class TestContinueOnAnEarlierReplyBranches:
    """Specification S8, second half. Continuing in place mutated a message
    with descendants, and nothing recorded that it had changed."""

    def test_the_original_reply_is_left_exactly_as_it_was(self, chats, thread,
                                                          monkeypatch):
        replies(monkeypatch, pieces=(" and then",))

        run("continue", thread, index=1)

        kept = chats.load("Ada", thread.identifier)
        assert kept.messages[1].text == "reply 0"
        assert len(kept.messages) == 6

    def test_the_continuation_happens_in_a_branch(self, chats, thread, monkeypatch):
        replies(monkeypatch, pieces=(" and then",))

        outcome = run("continue", thread, index=1)
        made = outcome["resulting_conversation"]["thread_id"]

        assert made != thread.identifier
        branched = chats.load("Ada", made)
        assert branched.messages[-1].text.startswith("reply 0")
        assert "and then" in branched.messages[-1].text

    def test_continuing_the_final_reply_is_unchanged(self, chats, thread, monkeypatch):
        """Final-reply continue keeps its opening and stays where it is. It was
        never the defect and is not changed by the fix."""
        replies(monkeypatch, pieces=(" and then",))

        outcome = run("continue", thread, index=5)

        assert outcome["resulting_conversation"]["thread_id"] == thread.identifier
        kept = chats.load("Ada", thread.identifier)
        assert kept.messages[5].text.startswith("reply 2")
        assert "and then" in kept.messages[5].text

    def test_there_must_be_something_to_carry_on_from(self, chats, thread, monkeypatch):
        from prompt_master.chat.history import ASSISTANT

        thread.append(ASSISTANT, "")
        chats.save(thread)
        replies(monkeypatch)

        outcome = run("continue", thread, index=6)

        assert outcome["ok"] is False


class TestRegenerate:
    def test_the_final_reply_grows_a_version(self, chats, thread, monkeypatch):
        replies(monkeypatch)

        outcome = run("regenerate", thread, index=5)

        assert outcome["resulting_conversation"]["thread_id"] == thread.identifier
        kept = chats.load("Ada", thread.identifier)
        assert kept.messages[5].versions == ["reply 2", "A new reply"]
        assert kept.messages[5].active == 1

    def test_a_middle_reply_branches_and_keeps_what_followed(self, chats, thread,
                                                             monkeypatch):
        replies(monkeypatch)

        outcome = run("regenerate", thread, index=1)

        assert [message.text for message in
                chats.load("Ada", thread.identifier).messages] == [
                    "ask 0", "reply 0", "ask 1", "reply 1", "ask 2", "reply 2"]
        made = outcome["resulting_conversation"]["thread_id"]
        assert [message.text for message in chats.load("Ada", made).messages] == [
            "ask 0", "A new reply"]

    def test_versions_only_ever_exist_where_nothing_follows(self, chats, thread,
                                                            monkeypatch):
        """What makes paging between attempts safe: a version is one string, so
        a version in the middle of a thread cannot hold what came after it."""
        replies(monkeypatch)

        run("regenerate", thread, index=1)

        assert chats.load("Ada", thread.identifier).messages[1].versions == ["reply 0"]


class TestSending:
    def test_your_turn_is_saved_before_the_model_is_asked(self, chats, thread,
                                                          monkeypatch):
        """Deliberately, so it is not lost when a reply is. ``persisted`` says
        so in the answer."""
        replies(monkeypatch)

        outcome = run("send", thread, payload={"text": "a new question"})

        assert outcome["persisted"] is True
        kept = chats.load("Ada", thread.identifier)
        assert kept.messages[6].text == "a new question"
        assert kept.messages[7].text == "A new reply"

    def test_a_blank_message_with_no_picture_is_refused(self, thread, monkeypatch):
        replies(monkeypatch)

        outcome = run("send", thread, payload={"text": "   "})

        assert outcome["ok"] is False
        assert outcome["error"]["code"] == service.INVALID_INPUT

    def test_a_reply_that_produced_nothing_leaves_no_empty_bubble(self, chats, thread,
                                                                  monkeypatch):
        replies(monkeypatch, pieces=(), terminal=sessions.DONE)

        run("send", thread, payload={"text": "a question"})

        kept = chats.load("Ada", thread.identifier)
        assert [message.role for message in kept.messages][-1] == "user"

    def test_a_failed_reply_keeps_your_turn(self, chats, thread, monkeypatch):
        monkeypatch.setattr(sessions, "conversation", lambda request, cancel: iter(
            [sessions.Event(sessions.FAILED, "the server would not start")]))

        run("send", thread, payload={"text": "a question"})

        kept = chats.load("Ada", thread.identifier)
        assert kept.messages[-1].text == "a question"


class TestBusy:
    def test_a_second_reply_in_one_thread_is_refused(self, thread, monkeypatch):
        """Not queued. A queue whose head is a language model is a UI that has
        silently hung."""
        started = threading.Event()
        release = threading.Event()

        def slow(request, cancel):
            started.set()
            release.wait(5)
            yield sessions.Event(sessions.DONE, "eventually")

        monkeypatch.setattr(sessions, "conversation", slow)
        from test_conversation_service import envelope

        service.submit(envelope("send", thread, payload={"text": "one"},
                                operation_id="op-a"))
        assert started.wait(5)
        try:
            outcome = service.submit(envelope("send", thread, payload={"text": "two"},
                                              operation_id="op-b"))
            assert outcome["ok"] is False
            assert outcome["error"]["code"] == service.THREAD_BUSY
        finally:
            release.set()
            ops.drain(timeout=10)

    def test_an_edit_during_a_reply_is_refused(self, thread, monkeypatch):
        """H08, through the front door rather than by claiming the lock."""
        started = threading.Event()
        release = threading.Event()

        def slow(request, cancel):
            started.set()
            release.wait(5)
            yield sessions.Event(sessions.DONE, "eventually")

        monkeypatch.setattr(sessions, "conversation", slow)
        from test_conversation_service import envelope

        service.submit(envelope("send", thread, payload={"text": "one"},
                                operation_id="op-a"))
        assert started.wait(5)
        try:
            outcome = service.submit(envelope("edit_message", thread, index=0,
                                              payload={"text": "changed"},
                                              operation_id="op-b"))
            assert outcome["error"]["code"] == service.THREAD_BUSY
        finally:
            release.set()
            ops.drain(timeout=10)


class TestStop:
    def test_a_stopped_reply_keeps_what_arrived(self, chats, thread, monkeypatch):
        replies(monkeypatch, pieces=("half a ", "reply"), terminal=sessions.CANCELLED)

        run("send", thread, payload={"text": "a question"})

        kept = chats.load("Ada", thread.identifier)
        assert kept.messages[-1].text == "half a reply"

    def test_stop_is_harmless_twice_and_after_the_end(self, thread, monkeypatch):
        replies(monkeypatch)
        outcome = run("send", thread, payload={"text": "a question"})
        operation = outcome["operation_id"]

        assert ops.stop(operation) is True
        assert ops.stop(operation) is True
        assert ops.stop("an id nothing ever had") is False

    def test_an_accepted_stop_leaves_no_automatic_speech(self, thread, monkeypatch):
        """7.7. Once Stop is accepted for a running operation, no automatic
        playback may begin -- that race had two winners and the audio was one."""
        started = threading.Event()
        release = threading.Event()

        def slow(request, cancel):
            started.set()
            release.wait(5)
            yield sessions.Event(sessions.DONE, "a whole reply")

        monkeypatch.setattr(sessions, "conversation", slow)
        from test_conversation_service import envelope

        outcome = service.submit(envelope("send", thread, payload={"text": "one"}))
        assert started.wait(5)
        ops.stop(outcome["operation_id"])
        release.set()
        assert ops.drain(timeout=10)

        completion = ops.completion(outcome["operation_id"])
        assert completion.status == ops.STOPPED
        assert completion.auto_speech_owner_page == ""


class TestTheCompletionGuard:
    def test_a_reply_is_not_written_into_a_conversation_that_moved(self, chats, thread,
                                                                   monkeypatch):
        """H11. The completion compares against the revision the operation was
        accepted at. A reply saved into a conversation that is no longer the one
        it answers is worse than a reply that needs a click."""
        started = threading.Event()
        release = threading.Event()

        def slow(request, cancel):
            started.set()
            release.wait(5)
            yield sessions.Event(sessions.DONE, "the reply")

        monkeypatch.setattr(sessions, "conversation", slow)
        from test_conversation_service import envelope

        accepted = service.submit(envelope("send", thread, payload={"text": "one"}))
        assert accepted["ok"] is True
        assert started.wait(5)
        # Something else writes to the file while the model is thinking. Not
        # through the service -- that would be refused as busy -- but as an
        # external editor or a second process would.
        moved = chats.load("Ada", thread.identifier)
        moved.append("user", "from outside")
        chats.save(moved)
        release.set()
        assert ops.drain(timeout=10)

        kept = chats.load("Ada", thread.identifier)
        assert [message.text for message in kept.messages][-1] == "from outside"
        assert ops.recoverable(store_module.key_of(thread)), (
            "the reply was neither written nor kept")

    def test_a_change_that_leaves_the_target_where_it_was_is_still_refused(
            self, chats, thread, monkeypatch):
        """The revision guard, on its own.

        The test above is caught by two things at once: the conversation grew,
        so the index the reply was going to occupy is no longer the end of it.
        This one is caught by exactly one. An earlier message is *edited* while
        the model is thinking -- the length does not change, the target index is
        still correct, and the only thing that knows the conversation is not the
        one this reply answers is the revision it was accepted at.

        Without that comparison this reply would be appended to a conversation
        whose question now says something else, and nothing anywhere would
        record that it had happened.
        """
        started = threading.Event()
        release = threading.Event()

        def slow(request, cancel):
            started.set()
            release.wait(5)
            yield sessions.Event(sessions.DONE, "an answer to the old question")

        monkeypatch.setattr(sessions, "conversation", slow)
        from test_conversation_service import envelope

        outcome = service.submit(envelope("send", thread, payload={"text": "one"}))
        assert started.wait(5)
        moved = chats.load("Ada", thread.identifier)
        moved.messages[0].text = "a completely different question"
        chats.save(moved)
        release.set()
        assert ops.drain(timeout=10)

        kept = chats.load("Ada", thread.identifier)
        assert [message.text for message in kept.messages][-1] == "one", (
            "the reply was written into a conversation that had changed")
        assert ops.recoverable(store_module.key_of(thread)), (
            "the reply was neither written nor kept")
        assert ops.get(outcome["operation_id"]).recovery == "result_conflict"

    def test_an_unresolved_reply_can_be_saved_as_a_branch(self, chats, thread,
                                                          monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def slow(request, cancel):
            started.set()
            release.wait(5)
            yield sessions.Event(sessions.DONE, "the reply that lost its race")

        monkeypatch.setattr(sessions, "conversation", slow)
        from test_conversation_service import envelope

        outcome = service.submit(envelope("send", thread, payload={"text": "one"}))
        assert started.wait(5)
        moved = chats.load("Ada", thread.identifier)
        moved.append("user", "from outside")
        chats.save(moved)
        release.set()
        assert ops.drain(timeout=10)

        found = ops.resolve(outcome["operation_id"], "branch")

        made = found["resulting_conversation"]["thread_id"]
        assert chats.load("Ada", made).messages[-1].text == "the reply that lost its race"
        assert ops.recoverable(store_module.key_of(thread)) == []

    def test_an_unresolved_reply_can_be_discarded(self, chats, thread, monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def slow(request, cancel):
            started.set()
            release.wait(5)
            yield sessions.Event(sessions.DONE, "unwanted")

        monkeypatch.setattr(sessions, "conversation", slow)
        from test_conversation_service import envelope

        outcome = service.submit(envelope("send", thread, payload={"text": "one"}))
        assert started.wait(5)
        moved = chats.load("Ada", thread.identifier)
        moved.append("user", "from outside")
        chats.save(moved)
        release.set()
        assert ops.drain(timeout=10)

        ops.resolve(outcome["operation_id"], "discard")

        assert ops.recoverable(store_module.key_of(thread)) == []

    def test_a_save_failure_is_reported_and_the_reply_is_kept(self, chats, thread,
                                                              monkeypatch):
        """H12. No false "Saved", and no lost reply."""
        replies(monkeypatch)
        real = store_module.transaction

        def refuse(*args, **kwargs):
            # The completion is the one transaction that writes while it is
            # itself the active writer, so ``allow_busy`` is what names it --
            # and ``**kwargs`` so this stub does not have to be rewritten every
            # time the real signature gains an argument.
            if kwargs.get("allow_busy"):
                raise store_module.SaveFailed("the disk is full")
            return real(*args, **kwargs)

        monkeypatch.setattr(store_module, "transaction", refuse)

        outcome = run("send", thread, payload={"text": "a question"})

        operation = ops.get(outcome["operation_id"])
        assert operation.phase == ops.SAVE_FAILED
        assert ops.recoverable(store_module.key_of(thread))


class TestCompletionRecords:
    def test_two_replies_cannot_take_each_others_answer(self, chats, monkeypatch):
        """V05. One process-global slot could do exactly this, and the page that
        lost spoke somebody else's reply."""
        from prompt_master.chat.history import USER

        first = chats.new("Ada")
        first.append(USER, "one")
        chats.save(first)
        second = chats.new("Ada")
        second.append(USER, "two")
        chats.save(second)

        monkeypatch.setattr(sessions, "conversation", lambda request, cancel: iter(
            [sessions.Event(sessions.DONE, "answer one")]))
        a = run("send", first, payload={"text": "x"}, operation_id="op-a")
        monkeypatch.setattr(sessions, "conversation", lambda request, cancel: iter(
            [sessions.Event(sessions.DONE, "answer two")]))
        b = run("send", second, payload={"text": "y"}, operation_id="op-b")

        assert ops.completion(a["operation_id"]).final_text == "answer one"
        assert ops.completion(b["operation_id"]).final_text == "answer two"

    def test_a_record_is_read_rather_than_taken(self, thread, monkeypatch):
        replies(monkeypatch)
        outcome = run("send", thread, payload={"text": "x"})

        first = ops.completion(outcome["operation_id"])
        again = ops.completion(outcome["operation_id"])

        assert first.final_text == again.final_text == "A new reply"

    def test_only_a_whole_reply_is_eligible_to_be_spoken(self, thread, monkeypatch):
        replies(monkeypatch, terminal=sessions.CANCELLED)

        outcome = run("send", thread, payload={"text": "x"})

        assert ops.completion(outcome["operation_id"]).status == ops.STOPPED
        assert ops.completion(outcome["operation_id"]).auto_speech_owner_page == ""


class TestCheckpoints:
    def test_a_checkpoint_never_moves_the_revision(self, chats, thread):
        """It is not a conversation write, which is what makes it safe to take
        while another window is reading."""
        key = store_module.key_of(thread)
        operation = ops.Operation(operation_id="op-c", key=key, action="send",
                                  target_index=6)
        before = chats.load("Ada", thread.identifier).revision

        assert ops.checkpoint(operation, "half a reply") is True

        assert chats.load("Ada", thread.identifier).revision == before

    def test_an_interrupted_reply_is_offered_and_never_resumed(self, chats, thread):
        key = store_module.key_of(thread)
        operation = ops.Operation(operation_id="op-d", key=key, action="send",
                                  target_index=6)
        ops.checkpoint(operation, "half a reply")

        found = ops.interrupted()

        assert [item["operation_id"] for item in found] == ["op-d"]
        assert found[0]["text"] == "half a reply"
        assert found[0]["phase"] == ops.INTERRUPTED
        assert chats.load("Ada", thread.identifier).messages[-1].text == "reply 2", (
            "an interrupted reply was resumed into the thread")


class TestFollowing:
    def test_a_follower_sees_the_phases_and_ends_at_a_terminal(self, thread,
                                                               monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def slow(request, cancel):
            started.set()
            yield sessions.Event(sessions.CHUNK, "half ")
            release.wait(5)
            yield sessions.Event(sessions.DONE, "half a reply")

        monkeypatch.setattr(sessions, "conversation", slow)
        from test_conversation_service import envelope

        outcome = service.submit(envelope("send", thread, payload={"text": "x"}))
        assert started.wait(5)
        release.set()

        seen = [state["phase"] for state in ops.listen(outcome["operation_id"])]

        assert seen[-1] in ops.TERMINAL
        assert ops.GENERATING in seen or ops.SAVING in seen

    def test_a_follower_that_goes_away_does_not_stop_the_reply(self, chats, thread,
                                                               monkeypatch):
        """H10. Closing the generator costs a subscription and nothing else."""
        replies(monkeypatch)
        from test_conversation_service import envelope

        outcome = service.submit(envelope("send", thread, payload={"text": "x"}))
        watcher = ops.listen(outcome["operation_id"])
        next(watcher)
        watcher.close()

        assert ops.drain(timeout=10)
        assert chats.load("Ada", thread.identifier).messages[-1].text == "A new reply"
