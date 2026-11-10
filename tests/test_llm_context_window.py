"""A thread's context window, end to end: what each reply's request sends.

"i get a lot of delay still between reply ... why did it take so long between
my submitted message and time to first token?" (2026-11-09). The answer was in
the user's own logs: a long thread's oldest messages left every second message
or so, which changed the beginning of the prompt, so llama-server read the
whole conversation again -- 33 seconds on the Intel Arc -- before the first
word. These hold what replaced it, through the real service, store and builder,
with only the model replaced:

* a thread's window starts where it started last time while it fits, so
  message after message begins the way the cached prompt did;
* when your next message would not fit, the window moves as soon as the reply
  is saved and the thread is read again in the background, so the message
  itself reads only itself;
* Trim keeps the latest whole messages, Clear puts the window back, Compact has
  the model write a summary that is sent after the system prompt in place of
  the messages it covers -- and a message in a thread being compacted is
  refused with a sentence.
"""

from __future__ import annotations

import time

import pytest

import mc_llm_cache_update as cache_update
import mc_llm_compact as compact
import mc_llm_conversation_feed as feed
import mc_llm_conversation_ops as ops
import mc_llm_conversation_service as service
import mc_llm_conversation_store as store_module
import mc_llm_paths
import mc_llm_sessions as sessions
import mc_llm_tokens
import mc_llm_window as window_module

CONTEXT = 3000

WORDS = "a river ran past the old mill and the miller sang to the wheel all day "
"""Seventy characters of ordinary words."""


def words(count: int) -> str:
    return (WORDS * (count // len(WORDS) + 1))[:count].strip()


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(ops, "_context_size", lambda: CONTEXT)
    service.reset()
    store_module.registry.reset()
    ops.reset()
    feed.reset()
    compact.reset()
    cache_update.reset()
    mc_llm_tokens.forget()
    from prompt_master.chat.characters import Character, CharacterStore

    CharacterStore(tmp_path / "characters").save(Character(name="Ada",
                                                           context="a reader of maps"))
    yield tmp_path
    compact.reset()
    cache_update.reset()
    ops.reset()
    service.reset()
    store_module.registry.reset()
    feed.reset()


@pytest.fixture
def chats(store):
    from prompt_master.chat.history import ChatStore

    return ChatStore(store / "chats")


def long_thread(chats, exchanges: int = 12, reply: int = 800, yours: int = 160):
    from prompt_master.chat.history import ASSISTANT, USER

    found = chats.new("Ada")
    for index in range(exchanges):
        found.append(USER, f"{index}: " + words(yours))
        found.append(ASSISTANT, f"{index}: " + words(reply))
    chats.save(found)
    return found


class Model:
    """The language model: every request it was asked, and one reply."""

    def __init__(self, monkeypatch, reply: str = ""):
        self.asked = []
        self.reply = reply or words(800)
        monkeypatch.setattr(sessions, "conversation", self.answer)

    def answer(self, request, cancel):
        self.asked.append(request)
        return iter([sessions.Event(sessions.CHUNK, self.reply),
                     sessions.Event(sessions.DONE, self.reply)])


class Updates:
    """The cache update: a server that is up, and every update scheduled."""

    def __init__(self, monkeypatch):
        self.scheduled = []
        monkeypatch.setattr(cache_update, "warm", lambda: True)
        monkeypatch.setattr(cache_update, "schedule",
                            lambda key, build: self.scheduled.append((key, build)) or True)


def send(conversation, text, operation_id):
    from test_conversation_service import envelope

    current, _ = store_module.read(service.chats(), store_module.Key(
        character=conversation.character, thread_id=conversation.identifier))
    outcome = service.submit(envelope("send", current, payload={"text": text},
                                      operation_id=operation_id))
    assert ops.drain(timeout=10), "a reply did not finish on its own thread"
    return outcome


def command(action, conversation, payload=None, operation_id="w-1"):
    from test_conversation_service import envelope

    current, _ = store_module.read(service.chats(), store_module.Key(
        character=conversation.character, thread_id=conversation.identifier))
    return service.submit(envelope(action, current, payload=payload, revision=None,
                                   operation_id=operation_id))


def first_message(request) -> str:
    """The first message after the system prompt: what decides whether a
    prompt begins the way the cached one did."""
    return request.messages[1]["content"]


def system_of(request) -> str:
    return request.messages[0]["content"]


class TestTheWindowHoldsMessageAfterMessage:
    def test_a_long_thread_begins_its_prompt_the_same_way_until_it_has_to_move(
            self, chats, monkeypatch):
        """The defect, in the user's terms: the beginning of the prompt changed
        on about every second message, and every change was a full read."""
        conversation = long_thread(chats)
        model = Model(monkeypatch, reply=words(300))

        send(conversation, "first", "s-1")
        first = model.asked[-1]
        kept = window_module.for_thread("Ada", conversation.identifier)
        assert kept.front > 0, "a thread this long has to leave its oldest messages"

        send(conversation, "second", "s-2")
        send(conversation, "third", "s-3")

        assert [first_message(request) for request in model.asked] == [first_message(first)] * 3
        assert window_module.for_thread("Ada", conversation.identifier).front == kept.front

    def test_it_moves_far_enough_not_to_move_again_for_several_messages(
            self, chats, monkeypatch):
        conversation = long_thread(chats)
        model = Model(monkeypatch, reply=words(200))

        for turn in range(4):
            send(conversation, f"turn {turn}", f"s-{turn}")

        beginnings = {first_message(request) for request in model.asked}
        assert len(beginnings) == 1

    def test_the_old_messages_it_left_are_not_sent(self, chats, monkeypatch):
        conversation = long_thread(chats)
        model = Model(monkeypatch)

        send(conversation, "first", "s-1")

        sent = [str(message["content"]) for message in model.asked[-1].messages[1:]]
        assert not any(text.startswith("0: ") for text in sent)
        assert any(text.startswith("11: ") for text in sent)


class TestCountedInTheModelsTokens:
    def test_a_window_counted_at_the_threads_own_ratio_is_sent_whole(self, chats, monkeypatch):
        """The builder trims no further than the window chosen for it: a thread
        whose words run five characters a token keeps more than the old
        3.2-a-token estimate would have let through."""
        conversation = long_thread(chats)
        window_module.remember_ratio("Ada", conversation.identifier, 0.2)
        model = Model(monkeypatch, reply=words(200))

        send(conversation, "first", "s-1")

        front = window_module.for_thread("Ada", conversation.identifier).front
        asked = model.asked[-1]
        assert len(asked.messages) - 1 == 25 - front, "every message of the window is sent"


class TestAfterAReply:
    def test_when_your_next_message_would_not_fit_the_window_moves_now(
            self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=6)
        Model(monkeypatch, reply=words(2400))
        updates = Updates(monkeypatch)

        send(conversation, "go on", "s-1")

        kept = window_module.for_thread("Ada", conversation.identifier)
        assert kept.front > 0
        assert len(updates.scheduled) == 1, "the thread is read again under its new window"
        key, build = updates.scheduled[0]
        assert (key.character, key.thread_id) == ("Ada", conversation.identifier)
        read_again = build()
        assert read_again.max_tokens == 1
        assert read_again.messages[-1]["role"] == "assistant"

    def test_and_your_next_message_then_begins_where_that_read_began(
            self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=6)
        model = Model(monkeypatch, reply=words(2400))
        updates = Updates(monkeypatch)

        send(conversation, "go on", "s-1")
        read_again = updates.scheduled[0][1]()
        model.reply = words(200)
        send(conversation, "and then?", "s-2")

        nxt = model.asked[-1]
        assert first_message(nxt) == first_message(read_again)
        assert system_of(nxt) == system_of(read_again)
        # The read ends on the reply; the message's prompt goes on from it.
        assert nxt.messages[:len(read_again.messages)] == read_again.messages

    def test_a_thread_with_room_to_spare_is_not_read_again(self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=2, reply=200)
        Model(monkeypatch, reply=words(200))
        updates = Updates(monkeypatch)

        send(conversation, "hello", "s-1")

        assert updates.scheduled == []
        assert window_module.for_thread("Ada", conversation.identifier).front == 0

    def test_with_no_server_running_nothing_is_read_and_nothing_moves_early(
            self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=6)
        Model(monkeypatch, reply=words(2400))
        scheduled = []
        monkeypatch.setattr(cache_update, "warm", lambda: False)
        monkeypatch.setattr(cache_update, "schedule",
                            lambda key, build: scheduled.append(key) or True)

        send(conversation, "go on", "s-1")

        assert scheduled == []


class TestTrim:
    def test_it_keeps_the_latest_whole_messages_and_reads_the_thread_again(
            self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=6, reply=600)
        updates = Updates(monkeypatch)

        outcome = command("set_window", conversation, {"keep_tokens": 800})

        assert outcome["ok"], outcome
        kept = window_module.for_thread("Ada", conversation.identifier)
        assert kept.front > 0
        assert outcome["kept"]["messages"] == 12 - kept.front
        assert outcome["kept"]["tokens"] <= 800 or outcome["kept"]["messages"] == 2
        assert outcome["context"]["front"] == kept.front
        assert len(updates.scheduled) == 1

    def test_the_next_reply_sends_from_there(self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=6, reply=600)
        Updates(monkeypatch)
        model = Model(monkeypatch, reply=words(100))
        command("set_window", conversation, {"keep_tokens": 800})
        front = window_module.for_thread("Ada", conversation.identifier).front

        send(conversation, "carry on", "s-1")

        history = store_module.read(service.chats(), store_module.Key(
            character="Ada", thread_id=conversation.identifier))[0].messages
        assert first_message(model.asked[-1]) == history[front].text

    def test_it_asks_for_no_revision_and_changes_none(self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=3)
        Updates(monkeypatch)
        before = store_module.read(service.chats(), store_module.Key(
            character="Ada", thread_id=conversation.identifier))[0].revision

        command("set_window", conversation, {"keep_tokens": 400})

        after = store_module.read(service.chats(), store_module.Key(
            character="Ada", thread_id=conversation.identifier))[0].revision
        assert before == after

    def test_a_count_that_is_not_one_is_refused(self, chats):
        conversation = long_thread(chats, exchanges=2)

        outcome = command("set_window", conversation, {"keep_tokens": "lots"})

        assert outcome["ok"] is False
        assert outcome["error"]["code"] == service.INVALID_INPUT


class TestClear:
    def test_it_puts_the_window_back_and_forgets_the_summary(self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=4)
        Updates(monkeypatch)
        window_module.set_summary("Ada", conversation.identifier, "They met.", 4, 3)

        outcome = command("clear_window", conversation)

        assert outcome["ok"], outcome
        kept = window_module.for_thread("Ada", conversation.identifier)
        assert (kept.front, kept.summary) == (0, "")


class TestASummary:
    def test_it_is_sent_after_the_system_prompt_and_its_messages_are_not(
            self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=4, reply=200)
        window_module.set_summary("Ada", conversation.identifier, "They met at the mill.", 4, 6)
        model = Model(monkeypatch, reply=words(100))

        send(conversation, "and now?", "s-1")

        asked = model.asked[-1]
        assert window_module.SUMMARY_HEAD + "\nThey met at the mill." in system_of(asked)
        assert first_message(asked).startswith("2: ")

    def test_the_emotion_comes_after_it(self, chats, monkeypatch):
        import mc_llm_emotions

        conversation = long_thread(chats, exchanges=4, reply=200)
        window_module.set_summary("Ada", conversation.identifier, "They met at the mill.", 4, 6)
        mc_llm_emotions.choose("Ada", conversation.identifier, mc_llm_emotions.listed()[0])
        model = Model(monkeypatch, reply=words(100))

        send(conversation, "and now?", "s-1")

        system = system_of(model.asked[-1])
        assert system.index(window_module.SUMMARY_HEAD) < system.index("[Current emotion]")


@pytest.fixture
def quick_turn(monkeypatch):
    """The language model's own wait made quick: nothing on the card."""
    import mc_broker

    monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.02)
    monkeypatch.setattr(sessions._Gpu, "_domain",
                        lambda self: mc_broker.UNKNOWN_CUDA_EXECUTION)
    monkeypatch.setattr(sessions._Gpu, "_blocked_by_the_image_job",
                        staticmethod(lambda domain: False))
    monkeypatch.setattr(sessions._Gpu, "_blocked_by_a_guest",
                        staticmethod(lambda domain, ours: ""))
    return monkeypatch


class Writer:
    """The server a compaction is written by: what it was asked, and a summary."""

    base_url, api_key = "http://127.0.0.1:1", "k"

    def __init__(self, text="**Summary**\nThey walked to the mill and talked.", hold=None):
        self.asked = []
        self.text = text
        self.hold = hold

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None, **kwargs):
        self.asked.append((messages, max_tokens, kwargs))
        if self.hold is not None:
            self.hold(cancel)
        on_text(self.text)
        return self.text


def compacted(conversation, monkeypatch, writer, keep=700, length=200, wait=True,
              operation_id="k-1"):
    monkeypatch.setattr(sessions, "_client", lambda needs_vision, **kwargs: writer)
    outcome = command("compact", conversation, {"keep_tokens": keep, "summary_tokens": length},
                      operation_id=operation_id)
    if wait:
        key = store_module.Key(character="Ada", thread_id=conversation.identifier)
        deadline = time.monotonic() + 10
        while compact.running(key) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not compact.running(key), "the compaction did not finish"
    return outcome


class TestCompact:
    def test_the_model_is_asked_in_the_threads_own_frame(self, chats, quick_turn):
        conversation = long_thread(chats, exchanges=5, reply=600)
        writer = Writer()

        outcome = compacted(conversation, quick_turn, writer)

        assert outcome["ok"], outcome
        messages, max_tokens, kwargs = writer.asked[0]
        assert messages[0]["role"] == "system" and "You are Ada" in messages[0]["content"]
        assert messages[1]["content"].startswith("0: ")
        assert messages[-1]["role"] == "user"
        assert "Write a summary of the conversation between Ada" in messages[-1]["content"]
        assert "150 words" in messages[-1]["content"]
        assert kwargs["temperature"] == compact.TEMPERATURE
        assert "sampling" not in kwargs, "none of the thread's own sampling"
        assert max_tokens <= 200 * compact.SLACK

    def test_the_summary_is_kept_without_its_heading_and_the_window_starts_after_it(
            self, chats, quick_turn):
        conversation = long_thread(chats, exchanges=5, reply=600)

        outcome = compacted(conversation, quick_turn, Writer())

        kept = window_module.for_thread("Ada", conversation.identifier)
        assert kept.summary == "They walked to the mill and talked."
        assert kept.upto == outcome["compacting"]["cut"] == kept.front
        assert 0 < kept.front < 10

    def test_the_next_reply_sends_the_summary_and_the_kept_messages(self, chats, quick_turn):
        conversation = long_thread(chats, exchanges=5, reply=600)
        compacted(conversation, quick_turn, Writer())
        front = window_module.for_thread("Ada", conversation.identifier).front
        model = Model(quick_turn, reply=words(100))

        send(conversation, "and then?", "s-1")

        asked = model.asked[-1]
        assert "They walked to the mill and talked." in system_of(asked)
        history = store_module.read(service.chats(), store_module.Key(
            character="Ada", thread_id=conversation.identifier))[0].messages
        assert first_message(asked) == history[front].text

    def test_a_second_compaction_folds_the_first_in(self, chats, quick_turn):
        conversation = long_thread(chats, exchanges=8, reply=600)
        first = compacted(conversation, quick_turn, Writer("They met."), keep=1200)
        assert first["ok"], first
        writer = Writer("They met, walked and talked.")

        second = compacted(conversation, quick_turn, writer, keep=500, operation_id="k-2")

        assert second["ok"], second
        assert second["compacting"]["cut"] > first["compacting"]["cut"]
        messages = writer.asked[0][0]
        assert "They met." in messages[0]["content"]
        assert "folding in the summary" in messages[-1]["content"]
        assert window_module.for_thread("Ada", conversation.identifier).summary == \
            "They met, walked and talked."

    def test_a_message_is_refused_while_the_thread_is_compacted(self, chats, quick_turn):
        import threading

        conversation = long_thread(chats, exchanges=5, reply=600)
        let_go = threading.Event()
        writer = Writer(hold=lambda cancel: let_go.wait(5))
        compacted(conversation, quick_turn, writer, wait=False)
        Model(quick_turn)

        outcome = send(conversation, "hello?", "s-1")
        let_go.set()

        assert outcome["ok"] is False
        assert outcome["error"]["code"] == service.THREAD_BUSY
        assert "being compacted" in outcome["error"]["message"]

    def test_reset_waits_for_a_running_compaction_to_end(self, chats, quick_turn):
        """``reset`` stops a compaction and waits for its thread. One left running
        past its test keeps its summary in whatever data folder is current by then
        -- the checkout's, once the test's own has been put back: a full run of
        2026-11-10 found window.json there, from the test above."""
        import threading

        conversation = long_thread(chats, exchanges=5, reply=600)
        started = threading.Event()

        def hold(cancel):
            started.set()
            cancel.wait(5)
            time.sleep(0.3)  # still ending when a reset that does not wait returns

        compacted(conversation, quick_turn, Writer(hold=hold), wait=False)
        assert started.wait(5)
        key = store_module.Key(character="Ada", thread_id=conversation.identifier)
        job = compact._jobs[compact._name(key)]

        compact.reset()

        assert not job.worker.is_alive()
        assert window_module.for_thread("Ada", conversation.identifier).summary == ""

    def test_cancel_keeps_nothing(self, chats, quick_turn):
        import threading

        conversation = long_thread(chats, exchanges=5, reply=600)
        started = threading.Event()

        def hold(cancel):
            started.set()
            cancel.wait(5)

        compacted(conversation, quick_turn, Writer(hold=hold), wait=False)
        assert started.wait(5)

        command("cancel_compact", conversation, operation_id="c-1")
        key = store_module.Key(character="Ada", thread_id=conversation.identifier)
        deadline = time.monotonic() + 5
        while compact.running(key) and time.monotonic() < deadline:
            time.sleep(0.02)

        assert window_module.for_thread("Ada", conversation.identifier).summary == ""
        assert compact.progress(key)["phase"] == "cancelled"

    def test_a_thread_changed_under_it_keeps_nothing(self, chats, quick_turn):
        conversation = long_thread(chats, exchanges=5, reply=600)

        def edit(cancel):
            found = chats.load("Ada", conversation.identifier)
            found.messages[1].text = "something else entirely"
            chats.save(found)

        compacted(conversation, quick_turn, Writer(hold=edit))

        key = store_module.Key(character="Ada", thread_id=conversation.identifier)
        assert window_module.for_thread("Ada", conversation.identifier).summary == ""
        assert compact.progress(key)["phase"] == "failed"
        assert "changed while it was being compacted" in compact.progress(key)["status"]

    def test_nothing_older_than_what_is_kept_is_refused(self, chats, quick_turn):
        conversation = long_thread(chats, exchanges=2, reply=200)

        outcome = compacted(conversation, quick_turn, Writer(), keep=100_000, wait=False)

        assert outcome["ok"] is False
        assert "Nothing older" in outcome["error"]["message"]


class TestTheReading:
    def test_the_snapshot_says_how_full_the_context_is(self, chats):
        conversation = long_thread(chats, exchanges=4, reply=400)

        found = service.snapshot("Ada", conversation.identifier)["context"]

        assert found["limit"] == CONTEXT
        assert 0 < found["used"] <= found["limit"]
        assert found["room"] == CONTEXT - found["reserve"]
        assert (found["sent"], found["messages"], found["front"]) == (8, 8, 0)
        assert found["exact"] is False, "no server is running: an estimate"
        assert found["summary"] is None and found["compacting"] is None

    def test_a_trimmed_thread_reads_less(self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=4, reply=400)
        Updates(monkeypatch)
        before = service.snapshot("Ada", conversation.identifier)["context"]["used"]

        command("set_window", conversation, {"keep_tokens": 300})

        after = service.snapshot("Ada", conversation.identifier)["context"]
        assert after["used"] < before
        assert after["sent"] < after["messages"]


class TestEditingTheThread:
    def test_a_message_deleted_before_the_window_keeps_it_on_the_same_messages(
            self, chats, monkeypatch):
        from test_conversation_service import envelope

        conversation = long_thread(chats, exchanges=4)
        window_module.set_front("Ada", conversation.identifier, 4)

        current, _ = store_module.read(service.chats(), store_module.Key(
            character="Ada", thread_id=conversation.identifier))
        outcome = service.submit(envelope("delete_message", current, index=1,
                                          operation_id="d-1"))

        assert outcome["ok"], outcome
        assert window_module.for_thread("Ada", conversation.identifier).front == 3

    def test_a_branch_keeps_the_window(self, chats):
        from test_conversation_service import envelope

        conversation = long_thread(chats, exchanges=4)
        window_module.set_front("Ada", conversation.identifier, 2)
        current, _ = store_module.read(service.chats(), store_module.Key(
            character="Ada", thread_id=conversation.identifier))

        outcome = service.submit(envelope("branch", current, index=5, operation_id="b-1"))

        made = outcome["resulting_conversation"]["thread_id"]
        assert window_module.for_thread("Ada", made).front == 2

    def test_a_deleted_thread_forgets_its_window(self, chats):
        from test_conversation_service import envelope

        conversation = long_thread(chats, exchanges=2)
        window_module.set_front("Ada", conversation.identifier, 2)
        current, _ = store_module.read(service.chats(), store_module.Key(
            character="Ada", thread_id=conversation.identifier))

        service.submit(envelope("delete_thread", current, operation_id="x-1"))

        assert window_module.for_thread("Ada", conversation.identifier) == \
            window_module.Window()
