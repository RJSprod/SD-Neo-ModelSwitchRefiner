"""Character Emotion: a line that starts the next reply, and the cache update after it.

The ask: Oobabooga's *character bias* for every conversation -- set per thread,
Off in a new one, a list of lines that can be added to and deleted from -- and,
because the line is sent before the reply and never kept, the last reply read
again in the background so the next message is no slower for it, with the wait
said in the progress area when a message is sent before that is done.

``mc_llm_sessions.conversation`` is replaced with a list of events, as in
``test_conversation_ops``: the prompt is built by the real builder and the reply
saved by the real guarded transaction. The cache update's reading is replaced
where it would reach a llama-server.
"""

from __future__ import annotations

import threading
import time

import pytest

import mc_llm_cache_update as cache_update
import mc_llm_conversation_api as api
import mc_llm_conversation_feed as feed
import mc_llm_conversation_ops as ops
import mc_llm_conversation_service as service
import mc_llm_conversation_store as store_module
import mc_llm_emotions as emotions
import mc_llm_sessions as sessions
from test_conversation_ops import chats, replies, run, store, thread  # noqa: F401 (fixtures)
from test_conversation_service import envelope

HAPPY = "*I am so happy*"


@pytest.fixture(autouse=True)
def updates(monkeypatch):
    """The cache update's reading, recorded instead of sent."""
    cache_update.reset()
    asked = []
    monkeypatch.setattr(cache_update, "_update",
                        lambda build, update: asked.append(build()) or "")
    yield asked
    cache_update.reset()


def name_of(conversation) -> str:
    return f"{conversation.character}/{conversation.identifier}"


def asking(monkeypatch, pieces=("Oh,", " hello!")):
    asked = []
    events = [sessions.Event(sessions.CHUNK, piece) for piece in pieces]
    events.append(sessions.Event(sessions.DONE, "".join(pieces)))
    monkeypatch.setattr(sessions, "conversation",
                        lambda request, cancel: (asked.append(request), iter(events))[1])
    return asked


def settled(timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with cache_update._guard:
            if not cache_update._pending:
                return True
        time.sleep(0.01)
    return False


def choose(conversation, line, operation_id="emotion-1"):
    body = envelope(service.SET_EMOTION, conversation, revision=None,
                    payload={"emotion": line}, operation_id=operation_id)
    return service.submit(body)


# --------------------------------------------------------------------------- #
# The list
# --------------------------------------------------------------------------- #


class TestTheList:
    def test_a_new_installation_has_oobaboogas_five(self, store):
        assert emotions.listed() == ["*I am so happy*", "*I am so sad*", "*I am so excited*",
                                     "*I am so bored*", "*I am so angry*"]

    def test_reading_it_writes_nothing(self, store):
        emotions.listed()

        assert not emotions.path().exists()

    def test_a_line_is_added_once_at_the_end(self, store):
        emotions.add("*I am so nervous*")
        found = emotions.add("*I am so nervous*")

        assert found[-1] == "*I am so nervous*"
        assert found.count("*I am so nervous*") == 1
        assert emotions.listed() == found

    def test_a_line_is_kept_as_one_line(self, store):
        found = emotions.add("  *I am so\n   tired*  ")

        assert found[-1] == "*I am so tired*"

    def test_an_empty_or_long_line_is_refused_in_a_sentence(self, store):
        with pytest.raises(emotions.Refused, match="Write the emotion first"):
            emotions.add("   ")
        with pytest.raises(emotions.Refused, match="short line"):
            emotions.add("x" * (emotions.MAX_LENGTH + 1))

    def test_a_full_list_is_refused(self, store, monkeypatch):
        monkeypatch.setattr(emotions, "MAX_COUNT", 5)

        with pytest.raises(emotions.Refused, match="full"):
            emotions.add("*I am so nervous*")

    def test_a_line_is_deleted(self, store):
        found = emotions.remove(HAPPY)

        assert HAPPY not in found
        assert emotions.listed() == found

    def test_an_unreadable_file_reads_as_the_defaults(self, store):
        emotions.path().parent.mkdir(parents=True, exist_ok=True)
        emotions.path().write_text("{ not json", encoding="utf-8")

        assert emotions.listed() == list(emotions.DEFAULTS)


# --------------------------------------------------------------------------- #
# Each thread's choice
# --------------------------------------------------------------------------- #


class TestEachThreadsChoice:
    def test_a_new_thread_is_off(self, thread):
        found = service.snapshot("Ada", thread.identifier)

        assert found["conversation"]["emotion"] == ""

    def test_choosing_one_is_said_by_the_snapshot_with_the_list(self, thread):
        outcome = choose(thread, HAPPY)
        found = service.snapshot("Ada", thread.identifier)

        assert outcome["ok"] is True, outcome
        assert outcome["emotion"] == HAPPY
        assert found["conversation"]["emotion"] == HAPPY
        assert found["emotions"] == emotions.listed()

    def test_choosing_writes_nothing_in_the_thread_and_moves_no_revision(self, chats, thread):
        """A setting, not a change to what was said: the other window's next
        command is against the revision it has, and a reply streaming now still
        saves into the file it was accepted against."""
        before = store_module.token(service.chats(), store_module.key_of(thread))
        written = chats.path_for("Ada", thread.identifier).read_bytes()

        outcome = choose(thread, HAPPY)

        assert outcome["revision"] == before
        assert store_module.token(service.chats(), store_module.key_of(thread)) == before
        assert chats.path_for("Ada", thread.identifier).read_bytes() == written

    def test_it_needs_no_revision_to_compare(self, thread):
        assert choose(thread, HAPPY)["ok"] is True

    def test_a_line_that_is_not_on_the_list_is_refused(self, thread):
        outcome = choose(thread, "*I am so invented*")

        assert outcome["ok"] is False
        assert outcome["error"]["code"] == service.INVALID_INPUT
        assert emotions.for_thread("Ada", thread.identifier) == ""

    def test_off_clears_it(self, thread):
        choose(thread, HAPPY)
        choose(thread, "", operation_id="emotion-2")

        assert emotions.for_thread("Ada", thread.identifier) == ""

    def test_it_belongs_to_the_thread_not_the_character(self, chats, thread):
        other = chats.new("Ada")
        chats.save(other)

        choose(thread, HAPPY)

        assert emotions.for_thread("Ada", other.identifier) == ""

    def test_a_branch_keeps_it(self, thread):
        choose(thread, HAPPY)

        outcome = service.submit(envelope(service.BRANCH, thread, index=1,
                                          operation_id="branch-1"))
        made = outcome["resulting_conversation"]["thread_id"]

        assert made != thread.identifier
        assert emotions.for_thread("Ada", made) == HAPPY

    def test_a_deleted_thread_forgets_it(self, thread):
        choose(thread, HAPPY)

        outcome = service.submit(envelope(service.DELETE_THREAD, thread,
                                          operation_id="delete-1"))

        assert outcome["ok"] is True, outcome
        assert f"Ada/{thread.identifier}" not in emotions._read()["threads"]

    def test_a_choice_of_a_line_that_is_not_on_the_list_reads_as_off(self, thread):
        """A file written by hand, or by a version that kept lines differently:
        the list is what decides which lines exist."""
        import json

        emotions.path().parent.mkdir(parents=True, exist_ok=True)
        emotions.path().write_text(json.dumps({
            "version": 1, "emotions": [HAPPY],
            "threads": {f"Ada/{thread.identifier}": "*I am so invented*"}}), encoding="utf-8")

        assert emotions.for_thread("Ada", thread.identifier) == ""

    def test_deleting_the_line_turns_its_threads_off_for_good(self, thread):
        """Off at once, and not back on when the same line is added again."""
        choose(thread, HAPPY)

        emotions.remove(HAPPY)
        assert emotions.for_thread("Ada", thread.identifier) == ""

        emotions.add(HAPPY)
        assert emotions.for_thread("Ada", thread.identifier) == ""


# --------------------------------------------------------------------------- #
# The wire
# --------------------------------------------------------------------------- #


def system_of(request) -> str:
    assert request.messages[0]["role"] == "system"
    return request.messages[0]["content"]


def note(line=HAPPY):
    return emotions.system_note(line, "Ada", ops._persona().display)


class TestTheWire:
    def test_the_emotion_ends_the_system_prompt_and_nothing_starts_the_reply(
            self, thread, monkeypatch):
        """Not Oobabooga's prefill, which the user found starting every reply "like
        there is hidden response that's been cropped out": the thread's system
        prompt, the character's own and then the emotion."""
        asked = asking(monkeypatch)
        run("send", thread, payload={"text": "Hi."}, operation_id="send-0")
        plain = system_of(asked[-1])
        choose(thread, HAPPY)

        run("send", thread, payload={"text": "Did you finish the painting?"},
            operation_id="send-1")

        assert system_of(asked[-1]) == plain + "\n\n" + note()
        assert asked[-1].messages[-1] == {"role": "user",
                                          "content": "Did you finish the painting?"}
        assert asked[-1].prefilled is False
        assert all(HAPPY not in str(message.get("content"))
                   for message in asked[-1].messages[1:])

    def test_the_note_has_one_shape_for_every_character_and_line(self):
        found = emotions.system_note(HAPPY, "Ada", "Sam")

        assert found.startswith("[Current emotion]\n")
        assert "Ada feels this right now, in Ada's own words: *I am so happy*" in found
        assert "Do not quote this note" in found
        assert emotions.system_note("", "Ada") == ""
        assert "*Ada is so glad Sam came*" in emotions.system_note(
            "*{{char}} is so glad {{user}} came*", "Ada", "Sam")

    def test_the_reply_is_saved_as_written(self, chats, thread, monkeypatch):
        asking(monkeypatch, pieces=(" Oh,", " hello!"))
        choose(thread, HAPPY)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        saved = chats.load("Ada", thread.identifier)
        assert saved.messages[-1].text == "Oh, hello!"
        assert all(HAPPY not in message.text for message in saved.messages)

    def test_every_request_in_the_thread_has_the_same_system_prompt(self, chats, thread,
                                                                     monkeypatch):
        """Send, Regenerate, Continue and Send again from here: one front for the
        whole thread, so llama-server keeps the conversation cached from one
        message to the next, and nothing is read again until the emotion changes."""
        asked = asking(monkeypatch)
        choose(thread, HAPPY)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")
        last = len(chats.load("Ada", thread.identifier).messages) - 1
        run("regenerate", thread, index=last, operation_id="again-1")
        run("continue", thread, index=last, operation_id="more-1")
        run("resend_from_user", thread, index=0, operation_id="resend-1")

        assert len(asked) == 4
        assert {system_of(request) for request in asked} == {system_of(asked[0])}
        assert system_of(asked[0]).endswith(note())
        assert not any(request.prefilled for request in asked)

    def test_off_is_the_character_s_own_prompt_exactly(self, thread, monkeypatch):
        from prompt_master.chat.prompt import system_text

        asked = asking(monkeypatch)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        assert system_of(asked[-1]) == system_text(ops._character("Ada"), ops._persona())
        assert asked[-1].messages[-1] == {"role": "user", "content": "Hi."}
        assert asked[-1].prefilled is False

    def test_a_character_s_own_prompt_stays_as_written_with_the_note_after_it(
            self, thread, monkeypatch):
        """The system prompt the user sees or overrides stays as it is: the
        emotion is added on the wire, never written into the character."""
        service.set_system_prompt("Ada", "You are {{char}}, a harbour pilot.")
        asked = asking(monkeypatch)
        choose(thread, HAPPY)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        assert system_of(asked[-1]) == "You are Ada, a harbour pilot.\n\n" + note()
        assert service.system_prompt("Ada")["text"] == "You are {{char}}, a harbour pilot."


# --------------------------------------------------------------------------- #
# The line llama-server sends back
# --------------------------------------------------------------------------- #


class Echoing:
    """llama-server b10621 continuing a reply from a line: the request's last
    assistant message, if it ends on one, comes back at the front of the first
    piece, then what the model wrote (``pieces``). ``echo`` false is a server that
    sends back only what the model wrote. ``stop`` is pressed before the piece at
    ``after``."""

    def __init__(self, pieces, stop=None, after=None, echo=True):
        self.pieces, self.stop, self.after, self.echo = list(pieces), stop, after, echo
        self.asked = []

    def stream_conversation(self, conversation, messages, max_tokens, seed, on_text,
                            cancel=None, **kwargs):
        self.asked.append(messages)
        pieces = list(self.pieces)
        if self.echo and pieces and messages and messages[-1].get("role") == "assistant":
            pieces[0] = messages[-1]["content"] + pieces[0]
        sent = []
        for index, piece in enumerate(pieces):
            if index == self.after:
                self.stop.cancel()
                break
            sent.append(piece)
            on_text(piece)
        return "".join(sent)


@pytest.fixture
def serving(monkeypatch):
    """The real reply runtime, with its card always free and ``client`` its server."""
    def acquire(self):
        return True
        yield  # pragma: no cover

    monkeypatch.setattr(sessions._Gpu, "acquire", acquire)
    monkeypatch.setattr(sessions, "_preparing", lambda role="": "Preparing…")
    monkeypatch.setattr(sessions, "_placement_notes", lambda role="": [])

    def serve(client):
        monkeypatch.setattr(sessions, "_client", lambda *args, **kwargs: client)

    return serve


def from_the_line(line=HAPPY):
    return sessions.ChatRequest(messages=[{"role": "user", "content": "Hi."},
                                          {"role": "assistant", "content": line}],
                                conversation="Ada/one", prefilled=True)


def streamed(request, cancel=None):
    events = list(sessions._conversation(request, cancel or sessions.Cancellation()))
    return [event.text for event in events if event.kind == sessions.CHUNK], events[-1]


def shown(pieces, line=HAPPY) -> list:
    reply = emotions.Unechoed(line)
    out = [reply.feed(piece) for piece in pieces]
    out.append(reply.finish())
    return out


SAMPLES = (HAPPY + " Oh, hello!", HAPPY + "\n\n*smiles* Oh, hello!", "  " + HAPPY + "Oh,",
           HAPPY + " " + HAPPY, " Oh, hello!", "*smiles* Oh, hello!", "*I am so sad* Oh.",
           "Oh, " + HAPPY, "*I am", "*I am so happy", HAPPY, "  ")


class TestTheLineIsTakenOffTheFront:
    def test_sent_back_with_the_first_words_it_is_dropped_with_the_space_after_it(self):
        assert "".join(shown([HAPPY + " Oh,", " hello!"])) == "Oh, hello!"
        assert emotions.without(HAPPY, HAPPY + " Oh, hello!") == "Oh, hello!"

    def test_pieces_are_held_only_while_they_could_still_be_the_line(self):
        assert shown(["*I am so", " happy*", "\n\n", "Oh,", " hello!"]) == \
            ["", "", "", "Oh,", " hello!", ""]
        assert shown(["*smi", "les* Oh,"]) == ["*smi", "les* Oh,", ""]

    def test_the_models_own_asterisks_are_its_reply(self):
        for reply in ("*smiles* Oh, hello!", "*I am so sad* Oh.", "Oh, " + HAPPY,
                      "Oh, *I am so happy* too."):
            assert "".join(shown([reply])) == reply
            assert emotions.without(HAPPY, reply) == reply
        assert emotions.without(HAPPY, HAPPY + "\n\n*smiles* Oh!") == "*smiles* Oh!"
        assert emotions.without(HAPPY, HAPPY + " " + HAPPY) == HAPPY

    def test_a_reply_without_the_line_passes_exactly_as_it_came(self):
        assert shown([" Oh,", " hello!"]) == [" Oh,", " hello!", ""]
        assert emotions.without(HAPPY, " Oh, hello!") == " Oh, hello!"

    def test_a_reply_that_ends_while_it_could_still_be_the_line_is_kept(self):
        assert "".join(shown(["*I am"])) == "*I am"
        assert emotions.without(HAPPY, "*I am") == "*I am"

    @pytest.mark.parametrize("sample", SAMPLES)
    def test_however_the_pieces_fall_the_stream_says_what_the_whole_does(self, sample):
        whole = emotions.without(HAPPY, sample)
        for cut in range(len(sample) + 1):
            for second in range(cut, len(sample) + 1):
                pieces = [sample[:cut], sample[cut:second], sample[second:]]
                assert "".join(shown(pieces)) == whole, pieces


class TestTheLineIsNotInTheReply:
    def test_it_is_never_streamed_and_the_reply_ends_without_it(self, serving):
        serving(Echoing([" Oh,", " hello!"]))

        chunks, last = streamed(from_the_line())

        assert chunks == ["Oh,", " hello!"]
        assert (last.kind, last.text) == (sessions.DONE, "Oh, hello!")

    def test_a_stopped_reply_keeps_what_was_written_without_it(self, serving):
        stop = sessions.Cancellation()
        serving(Echoing([" Oh,", " hello!"], stop=stop, after=1))

        chunks, last = streamed(from_the_line(), stop)

        assert chunks == ["Oh,"]
        assert (last.kind, last.text) == (sessions.CANCELLED, "Oh,")

    def test_words_still_held_when_the_reply_ends_are_streamed_then(self, serving):
        serving(Echoing(["*I am"], echo=False))

        chunks, last = streamed(from_the_line())

        assert chunks == ["*I am"]
        assert last.text == "*I am"

    def test_any_other_request_streams_exactly_as_it_came(self, serving):
        serving(Echoing([HAPPY + " Oh,", " hello!"], echo=False))
        ending_on_yours = sessions.ChatRequest(messages=[{"role": "user", "content": "Hi."}],
                                               conversation="Ada/one")
        not_from_a_line = sessions.ChatRequest(messages=from_the_line().messages,
                                               conversation="Ada/one")

        for request in (ending_on_yours, not_from_a_line):
            chunks, last = streamed(request)

            assert chunks == [HAPPY + " Oh,", " hello!"]
            assert last.text == HAPPY + " Oh, hello!"

    def test_with_an_emotion_on_the_reply_streams_and_saves_as_the_model_wrote_it(
            self, chats, thread, serving):
        server = Echoing([" Oh,", " *smiles*", " hello!"])
        serving(server)
        choose(thread, HAPPY)
        cursor = feed.cursor()
        subscriber = feed.subscribe("page-1")

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        patches = [event["payload"].get("text", "") for event in subscriber.since(cursor)[0]
                   if event["kind"] == service.REPLY_PATCH]
        assert patches and all(HAPPY not in text for text in patches)
        assert server.asked[-1][-1]["role"] == "user"
        saved = chats.load("Ada", thread.identifier)
        assert saved.messages[-1].text == "Oh, *smiles* hello!"


# --------------------------------------------------------------------------- #
# The cache update is asked for
# --------------------------------------------------------------------------- #


@pytest.fixture
def running(monkeypatch):
    """A server is running: a change of emotion has a cache to keep."""
    monkeypatch.setattr(cache_update, "warm", lambda: True)


class TestTheCacheUpdateIsAskedFor:
    def test_a_change_of_emotion_reads_the_thread_again_under_its_new_system_prompt(
            self, thread, updates, running):
        choose(thread, HAPPY)

        assert settled()
        request = updates[-1]
        assert system_of(request).endswith(note())
        assert request.messages[-1] == {"role": "assistant", "content": "reply 2"}
        assert request.max_tokens == 1
        assert request.prefilled is False
        assert request.conversation == name_of(thread)

    def test_off_reads_it_again_under_the_character_s_own(self, thread, updates, running):
        choose(thread, HAPPY)
        assert settled()

        choose(thread, "", operation_id="emotion-2")

        assert settled()
        assert len(updates) == 2
        assert "[Current emotion]" not in system_of(updates[-1])

    def test_the_same_emotion_again_reads_nothing(self, thread, updates, running):
        choose(thread, HAPPY)
        assert settled()

        choose(thread, HAPPY, operation_id="emotion-2")

        assert settled()
        assert len(updates) == 1

    def test_without_a_server_running_nothing_is_asked_for(self, thread, updates, monkeypatch):
        monkeypatch.setattr(cache_update, "warm", lambda: False)

        choose(thread, HAPPY)

        assert settled()
        assert updates == []
        assert not cache_update.pending(name_of(thread))

    def test_a_reply_written_with_an_emotion_needs_none(self, thread, monkeypatch, updates,
                                                        running):
        """Once the thread's system prompt has the emotion it stays the same for
        every message, so a reply leaves nothing to read again."""
        choose(thread, HAPPY)
        assert settled()
        updates.clear()
        asking(monkeypatch)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")

        assert settled()
        assert updates == []

    def test_a_thread_with_nothing_in_it_has_nothing_to_read(self, chats, updates, running):
        found = chats.new("Ada")
        chats.save(found)

        choose(found, HAPPY)

        assert settled()
        assert updates == [None]


# --------------------------------------------------------------------------- #
# Waiting for it
# --------------------------------------------------------------------------- #


def pending_update(name, reading=True):
    """An update registered as the scheduler registers one, reading or still
    waiting for its turn."""
    update = cache_update._Update()
    update.reading = reading
    with cache_update._guard:
        cache_update._pending[name] = update
    return update


class TestWaitingForIt:
    def test_nothing_to_wait_for_says_nothing(self):
        steps = list(cache_update.wait("Ada/one", sessions.Cancellation()))

        assert steps == []

    def test_a_reply_says_so_at_once_and_goes_ahead_when_it_ends(self):
        update = pending_update("Ada/one")
        heard, answer = [], {}

        def consume():
            answer["went"] = yield from cache_update.wait("Ada/one", sessions.Cancellation())

        walker = consume()
        heard.append(next(walker))
        assert heard[0].kind == sessions.STATUS
        assert heard[0].text == "Waiting for the cache update…"

        threading.Timer(0.05, update.done.set).start()
        with pytest.raises(StopIteration):
            next(walker)
        assert answer["went"] is True
        assert not update.cancel.is_set()

    def test_a_reply_takes_the_place_of_one_still_waiting_for_its_turn(self):
        """It reads the same reply in its own prompt, so nothing is waited for
        twice -- and it says nothing about a cache update it is not waiting for."""
        update = pending_update("Ada/one", reading=False)

        steps = list(cache_update.wait("Ada/one", sessions.Cancellation()))

        assert steps == []
        assert update.cancel.is_set()

    def test_a_stopped_reply_stops_waiting(self):
        pending_update("Ada/one")
        cancel = sessions.Cancellation()
        cancel.cancel()

        def consume():
            return (yield from cache_update.wait("Ada/one", cancel))

        walker = consume()
        next(walker)
        with pytest.raises(StopIteration) as stopped:
            next(walker)
        assert stopped.value.value is False

    def test_another_conversation_neither_waits_nor_stops_it(self):
        update = pending_update("Ada/one", reading=False)

        assert list(cache_update.wait("Ada/two", sessions.Cancellation())) == []
        assert not update.cancel.is_set()

    def test_the_reply_waits_before_it_asks_for_the_card(self, monkeypatch):
        order = []

        def waiting(name, cancel):
            order.append(("wait", name))
            yield sessions.Event(sessions.STATUS, cache_update.WAITING)
            return False

        def acquire(self):
            order.append(("card", None))
            return True
            yield  # pragma: no cover

        monkeypatch.setattr(cache_update, "wait", waiting)
        monkeypatch.setattr(sessions._Gpu, "acquire", acquire)
        request = sessions.ChatRequest(messages=[{"role": "user", "content": "hi"}],
                                       conversation="Ada/one")

        events = list(sessions._conversation(request, sessions.Cancellation()))

        assert order == [("wait", "Ada/one")]
        assert [event.kind for event in events] == [sessions.STATUS, sessions.CANCELLED]
        assert events[0].text == "Waiting for the cache update…"

    def test_a_prefilled_reply_keeps_no_saved_cache_and_others_are_asked_as_before(
            self, monkeypatch):
        calls = []

        class Client:
            def stream_conversation(self, *args, **kwargs):
                calls.append(kwargs)
                return "ok"

        def acquire(self):
            return True
            yield  # pragma: no cover

        monkeypatch.setattr(sessions._Gpu, "acquire", acquire)
        monkeypatch.setattr(sessions, "_client", lambda *args, **kwargs: Client())
        monkeypatch.setattr(sessions, "_preparing", lambda role="": "Preparing…")
        monkeypatch.setattr(sessions, "_placement_notes", lambda role="": [])

        for prefilled in (True, False):
            request = sessions.ChatRequest(messages=[{"role": "user", "content": "hi"}],
                                           conversation="Ada/one", prefilled=prefilled)
            list(sessions._conversation(request, sessions.Cancellation()))

        assert calls[0].get("save") is False
        assert "save" not in calls[1]


# --------------------------------------------------------------------------- #
# The update itself
# --------------------------------------------------------------------------- #


ENDING = [{"role": "system", "content": "s"}, {"role": "user", "content": "Hi."},
          {"role": "assistant", "content": "Oh, hello!"}]


class Rendered:
    """llama-server's /apply-template, as a dictionary of what it renders."""

    def __init__(self, continued: str, followed: str):
        self.continued, self.followed = continued, followed

    def __call__(self, client, messages):
        return self.followed if messages[-1]["role"] == "user" else self.continued


class TestContinues:
    def test_a_continued_reply_that_starts_the_next_prompt(self, monkeypatch):
        monkeypatch.setattr(cache_update, "_render",
                            Rendered("<t>model\nOh, hello!", "<t>model\nOh, hello!<e>\n<t>user"))

        assert cache_update.continues(object(), ENDING) is True

    def test_a_template_that_renders_a_past_reply_otherwise(self, monkeypatch):
        """Gemma 4's own, without the kept thought marker: the past reply has
        no empty thought block where the continued one has."""
        monkeypatch.setattr(cache_update, "_render",
                            Rendered("<t>model\n<c>thought\n<c|>Oh, hello!",
                                     "<t>model\nOh, hello!<e>\n<t>user"))

        assert cache_update.continues(object(), ENDING) is False

    def test_a_render_that_failed(self, monkeypatch):
        monkeypatch.setattr(cache_update, "_render", Rendered("", ""))

        assert cache_update.continues(object(), ENDING) is False

    def test_a_request_that_does_not_end_on_a_reply(self, monkeypatch):
        monkeypatch.setattr(cache_update, "_render", Rendered("a", "ab"))

        assert cache_update.continues(object(), ENDING[:2]) is False


class Warm:
    """The runtime's ``warm_client``: the given clients in turn, the last for ever."""

    def __init__(self, *clients):
        self.clients = list(clients) or [None]
        self.asked = []

    def warm_client(self, needs_vision=False):
        self.asked.append(needs_vision)
        return self.clients.pop(0) if len(self.clients) > 1 else self.clients[0]


class Client:
    base_url, api_key = "http://127.0.0.1:1", "k"

    def __init__(self, update=None):
        self.streamed = []
        self.update = update
        self.reading = []

    def stream_conversation(self, conversation, messages, max_tokens, seed, on_text,
                            cancel=None, **kwargs):
        if self.update is not None:
            self.reading.append(self.update.reading)
        self.streamed.append((conversation, messages, max_tokens, kwargs))
        return "x"


@pytest.fixture
def real_update(monkeypatch):
    """The update as written, with llama-server and the runtime replaced and the
    language model's own wait made quick: nothing on the card, at first."""
    import mc_broker

    monkeypatch.setattr(cache_update, "_update", TestTheUpdate.original)
    monkeypatch.setattr(cache_update, "continues", lambda client, messages: True)
    monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.02)
    monkeypatch.setattr(sessions._Gpu, "_domain",
                        lambda self: mc_broker.UNKNOWN_CUDA_EXECUTION)
    monkeypatch.setattr(sessions._Gpu, "_blocked_by_the_image_job",
                        staticmethod(lambda domain: False))
    monkeypatch.setattr(sessions._Gpu, "_blocked_by_a_guest",
                        staticmethod(lambda domain, ours: ""))
    return monkeypatch


def request_for(conversation="Ada/one"):
    return sessions.ChatRequest(messages=list(ENDING), max_tokens=1, seed=3,
                                conversation=conversation)


def busy_for(answers):
    """A check that answers each of ``answers`` in turn, then the last for ever,
    counting how often it was asked."""
    asked = []

    def check(*_args):
        asked.append(True)
        return answers[min(len(asked), len(answers)) - 1]

    return check, asked


class TestTheUpdate:
    original = staticmethod(cache_update._update)

    def resolve(self, monkeypatch, runtime):
        monkeypatch.setattr(sessions, "_runtime_for",
                            lambda role="": sessions._Resolved(runtime, None, None))

    def test_it_reads_the_conversation_on_the_warm_server_and_saves_it(self, real_update):
        client = Client()
        self.resolve(real_update, Warm(client))

        said = cache_update._update(request_for, cache_update._Update())

        assert said.startswith("the cache update read the conversation again")
        conversation, messages, max_tokens, kwargs = client.streamed[-1]
        assert (conversation, messages, max_tokens) == ("Ada/one", ENDING, 1)
        assert "save" not in kwargs

    def test_it_starts_no_server_and_waits_for_no_turn_without_one(self, real_update):
        """With no server there is no cache to update; the next message starts
        one and reads the conversation as it would without an emotion. And it
        does not queue for the card for nothing."""
        runtime = Warm(None)
        self.resolve(real_update, runtime)
        busy, asked = busy_for([True, True, False])
        real_update.setattr(sessions._Gpu, "_blocked_by_the_image_job", staticmethod(busy))

        assert cache_update._update(request_for, cache_update._Update()) == ""
        assert runtime.asked == [False]
        assert asked == []

    def test_it_waits_for_the_image_job_as_a_reply_does(self, real_update):
        update = cache_update._Update()
        client = Client(update)
        self.resolve(real_update, Warm(client))
        busy, asked = busy_for([True, True, True, False])
        real_update.setattr(sessions._Gpu, "_blocked_by_the_image_job", staticmethod(busy))

        said = cache_update._update(request_for, update)

        assert len(asked) >= 4
        assert said.startswith("the cache update read the conversation again")
        assert len(client.streamed) == 1
        assert client.reading == [True], "reading only once its turn had come"

    def test_it_waits_for_a_voice_box_render_as_a_reply_does(self, real_update):
        client = Client()
        self.resolve(real_update, Warm(client))
        busy, asked = busy_for(["a Voice Box render", "a Voice Box render", ""])
        real_update.setattr(sessions._Gpu, "_blocked_by_a_guest", staticmethod(busy))

        said = cache_update._update(request_for, cache_update._Update())

        assert len(asked) >= 3
        assert said.startswith("the cache update read the conversation again")
        assert len(client.streamed) == 1

    def test_it_waits_behind_the_language_model_rather_than_giving_up(self, real_update):
        import mc_broker

        client = Client()
        self.resolve(real_update, Warm(client))
        held = threading.Event()

        def holder():
            with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply"):
                held.set()
                time.sleep(0.3)

        worker = threading.Thread(target=holder, daemon=True)
        worker.start()
        assert held.wait(5)
        began = time.monotonic()
        said = cache_update._update(request_for, cache_update._Update())
        took = time.monotonic() - began
        worker.join(5)

        assert took >= 0.2
        assert said.startswith("the cache update read the conversation again")
        assert len(client.streamed) == 1

    def test_a_message_that_takes_its_place_stops_it_while_it_waits(self, real_update):
        client = Client()
        self.resolve(real_update, Warm(client))
        busy, _ = busy_for([True])
        real_update.setattr(sessions._Gpu, "_blocked_by_the_image_job", staticmethod(busy))
        update = pending_update("Ada/one", reading=False)
        answer = {}
        worker = threading.Thread(
            target=lambda: answer.setdefault("said", cache_update._update(request_for, update)),
            daemon=True)
        worker.start()
        time.sleep(0.1)

        steps = list(cache_update.wait("Ada/one", sessions.Cancellation()))
        worker.join(5)

        assert steps == []
        assert answer["said"] == ""
        assert client.streamed == []

    def test_a_server_gone_while_it_waited_is_not_started_again(self, real_update):
        client = Client()
        runtime = Warm(client, None)
        self.resolve(real_update, runtime)

        assert cache_update._update(request_for, cache_update._Update()) == ""
        assert client.streamed == []
        assert len(runtime.asked) == 2

    def test_on_any_other_template_the_reply_is_read_as_a_past_one(self, real_update):
        """Gemma 4's own template, or any that writes the newest reply
        differently: the reply finished, a placeholder message after it, and
        no saved slot -- it would hold the placeholder."""
        client = Client()
        self.resolve(real_update, Warm(client))
        real_update.setattr(cache_update, "continues", lambda client, messages: False)

        said = cache_update._update(request_for, cache_update._Update())

        conversation, messages, max_tokens, kwargs = client.streamed[-1]
        assert messages == ENDING + [{"role": "user", "content": cache_update.PROBE}]
        assert (conversation, max_tokens) == ("Ada/one", 1)
        assert kwargs.get("save") is False
        assert said.startswith("the cache update read the conversation again")
        assert "the last reply as a past one" in said

    def test_it_builds_its_request_only_once_its_turn_has_come(self, real_update):
        """So a reply being written while it waited is in what it reads, and the
        emotion is the one the thread has by then."""
        client = Client()
        self.resolve(real_update, Warm(client))
        busy, asked = busy_for([True, True, True, False])
        real_update.setattr(sessions._Gpu, "_blocked_by_the_image_job", staticmethod(busy))
        seen = []

        def build():
            seen.append(len(asked))
            return request_for()

        cache_update._update(build, cache_update._Update())

        assert seen and seen[0] >= 4
        assert len(client.streamed) == 1

    def test_nothing_to_read_reads_nothing(self, real_update):
        client = Client()
        self.resolve(real_update, Warm(client))

        assert cache_update._update(lambda: None, cache_update._Update()) == ""
        assert client.streamed == []

    def test_a_thread_that_ends_on_your_message_is_sent_as_it_is_and_kept_unsaved(
            self, real_update):
        client = Client()
        self.resolve(real_update, Warm(client))

        def asked_anyway(client, messages):
            raise AssertionError("there is no reply to compare")

        real_update.setattr(cache_update, "continues", asked_anyway)
        ending = ENDING[:2]

        said = cache_update._update(
            lambda: sessions.ChatRequest(messages=list(ending), max_tokens=1,
                                         conversation="Ada/one"),
            cache_update._Update())

        conversation, messages, max_tokens, kwargs = client.streamed[-1]
        assert messages == ending
        assert kwargs.get("save") is False
        assert said.startswith("the cache update read the conversation again")
        assert "as a past one" not in said

    def test_it_holds_the_card_under_its_own_name(self, real_update):
        import mc_broker

        seen = []

        class Watching(Client):
            def stream_conversation(self, *args, **kwargs):
                running = mc_broker.active()
                seen.append(running.label if running else None)
                return super().stream_conversation(*args, **kwargs)

        self.resolve(real_update, Warm(Watching()))

        cache_update._update(request_for, cache_update._Update())

        assert seen == ["the cache update"]

    def test_schedule_says_when_it_starts_and_ends(self, thread, monkeypatch):
        release = threading.Event()
        monkeypatch.setattr(cache_update, "_update",
                            lambda build, update: release.wait(5) and "")
        published = []
        monkeypatch.setattr(service, "publish",
                            lambda kind, key=None, **payload: published.append((kind, payload)))
        key = store_module.key_of(thread)

        assert cache_update.schedule(key, lambda: request_for(name_of(thread))) is True
        assert cache_update.pending(name_of(thread))
        release.set()

        assert settled()
        updates = [payload for kind, payload in published if kind == service.CACHE_UPDATE]
        assert updates == [{"updating": True}, {"updating": False}]

    def test_another_change_while_one_waits_for_its_turn_takes_its_place(self, thread,
                                                                         monkeypatch):
        waiting, release, built = threading.Event(), threading.Event(), []

        def update_(build, update):
            if not waiting.is_set():
                waiting.set()
                release.wait(5)
            if update.cancel.is_set():
                return ""
            built.append(build())
            return ""

        monkeypatch.setattr(cache_update, "_update", update_)
        published = []
        monkeypatch.setattr(service, "publish",
                            lambda kind, key=None, **payload: published.append((kind, payload)))
        key = store_module.key_of(thread)
        cache_update.schedule(key, lambda: "first")
        assert waiting.wait(5)
        with cache_update._guard:
            first = cache_update._pending[name_of(thread)]

        assert cache_update.schedule(key, lambda: "second") is True
        release.set()

        assert settled()
        assert first.done.wait(5)
        assert first.cancel.is_set()
        assert built == ["second"]
        updates = [payload for kind, payload in published if kind == service.CACHE_UPDATE]
        assert updates == [{"updating": True}, {"updating": True}, {"updating": False}]

    def test_another_change_while_one_reads_has_it_read_once_more(self, thread, monkeypatch):
        reading, release, built = threading.Event(), threading.Event(), []

        def update_(build, update):
            with cache_update._guard:
                update.reading = True
            if not built and not reading.is_set():
                reading.set()
                release.wait(5)
            built.append(build())
            return ""

        monkeypatch.setattr(cache_update, "_update", update_)
        published = []
        monkeypatch.setattr(service, "publish",
                            lambda kind, key=None, **payload: published.append((kind, payload)))
        key = store_module.key_of(thread)
        cache_update.schedule(key, lambda: "first")
        assert reading.wait(5)

        assert cache_update.schedule(key, lambda: "second") is False
        release.set()

        assert settled()
        assert built == ["first", "second"]
        updates = [payload for kind, payload in published if kind == service.CACHE_UPDATE]
        assert updates == [{"updating": True}, {"updating": False}]

    def test_a_failure_still_ends_it(self, thread, monkeypatch):
        def broken(build, update):
            raise RuntimeError("no")

        monkeypatch.setattr(cache_update, "_update", broken)
        key = store_module.key_of(thread)

        cache_update.schedule(key, lambda: request_for(name_of(thread)))

        assert settled()
        assert not cache_update.pending(name_of(thread))


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The one-time cleanup of the line saved with replies before it was taken off
# --------------------------------------------------------------------------- #


def leaked(chats, replies, asks=None, character="Ada"):
    """A thread whose replies were saved as they arrived before the line was taken off."""
    from prompt_master.chat.history import ASSISTANT, USER

    found = chats.new(character)
    for index, reply in enumerate(replies):
        found.append(USER, (asks or {}).get(index, f"ask {index}"))
        versions = reply if isinstance(reply, list) else [reply]
        found.append(ASSISTANT, versions[0])
        for version in versions[1:]:
            found.messages[-1].add_version(version)
    chats.save(found)
    return found


def last_written(chats, conversation, when):
    """Say the thread was last written at ``when``, as the file would."""
    from prompt_master.core.config import atomic_write_json, read_json

    where = chats.path_for(conversation.character, conversation.identifier)
    data = read_json(where)
    data["updated"] = when
    atomic_write_json(where, data)


class TestTheOneTimeCleanup:
    def test_a_saved_reply_loses_the_line_and_the_space_after_it_in_every_version(
            self, chats, store):
        found = leaked(chats, [[HAPPY + " Oh, hello!", "*I am so sad*\n\nFine, thanks."],
                               "*smiles* No line here."],
                       asks={0: HAPPY + " a message of yours"})
        before = chats.load("Ada", found.identifier).revision
        cursor = feed.cursor()
        subscriber = feed.subscribe("page-1")

        assert emotions.clean_saved_replies(chats) == 2

        saved = chats.load("Ada", found.identifier)
        assert saved.messages[1].versions == ["Oh, hello!", "Fine, thanks."]
        assert saved.messages[3].text == "*smiles* No line here."
        assert saved.messages[0].text == HAPPY + " a message of yours"
        assert saved.revision == before + 1
        kinds = [event["kind"] for event in subscriber.since(cursor)[0]]
        assert service.CONVERSATION_CHANGED in kinds

    def test_it_runs_once_and_never_again(self, chats, store):
        leaked(chats, [HAPPY + " Oh, hello!"])
        assert emotions.clean_saved_replies(chats) == 1
        assert (store / emotions.CLEANUP_FILE).is_file()

        later = leaked(chats, [HAPPY + " written after the fix, on purpose"])

        assert emotions.clean_saved_replies(chats) == 0
        assert chats.load("Ada", later.identifier).messages[1].text.startswith(HAPPY)

    def test_a_line_the_user_added_is_cleaned_too(self, chats, store):
        emotions.add("*I am so grumpy*")
        found = leaked(chats, ["*I am so grumpy* What now?"])

        emotions.clean_saved_replies(chats)

        assert chats.load("Ada", found.identifier).messages[1].text == "What now?"

    def test_a_thread_last_written_before_emotions_shipped_is_left_alone(self, chats, store):
        found = leaked(chats, [HAPPY + " from a time before there were emotions"])
        last_written(chats, found, emotions.LEAK_BEGAN - 1)

        assert emotions.clean_saved_replies(chats) == 0
        assert chats.load("Ada", found.identifier).messages[1].text.startswith(HAPPY)

    def test_with_no_chats_folder_there_is_nothing_to_clean_and_it_is_recorded(
            self, chats, store):
        assert not chats.directory.exists()

        assert emotions.clean_saved_replies(chats) == 0
        assert (store / emotions.CLEANUP_FILE).is_file()

    def test_a_folder_that_cannot_be_read_is_looked_at_next_time(self, chats, store,
                                                                monkeypatch):
        found = leaked(chats, [HAPPY + " Oh, hello!"])

        def unreadable(store, since):
            raise OSError("not mounted")
            yield  # pragma: no cover

        with monkeypatch.context() as patched:
            patched.setattr(emotions, "_threads_since", unreadable)
            assert emotions.clean_saved_replies(chats) == 0
        assert not (store / emotions.CLEANUP_FILE).exists()

        assert emotions.clean_saved_replies(chats) == 1
        assert chats.load("Ada", found.identifier).messages[1].text == "Oh, hello!"

    def test_the_extension_runs_it_when_it_starts(self, chats, store, monkeypatch):
        import mc_llm_conversation_startup as startup

        found = leaked(chats, [HAPPY + " Oh, hello!"])
        monkeypatch.setattr(startup, "_done", False)

        startup.on_app_started()

        assert chats.load("Ada", found.identifier).messages[1].text == "Oh, hello!"


class TestTheRoute:
    def test_it_reads_the_list(self, store):
        payload, status = api.emotions()

        assert status == 200
        assert payload["emotions"] == list(emotions.DEFAULTS)

    def test_it_adds_and_removes(self, store):
        added, _ = api.emotions({"add": "*I am so nervous*"})
        removed, _ = api.emotions({"remove": HAPPY})

        assert added["emotions"][-1] == "*I am so nervous*"
        assert HAPPY not in removed["emotions"]

    def test_one_change_at_a_time_and_only_text(self, store):
        for body in ({}, {"add": "a", "remove": "b"}, {"add": 3}, {"rename": "x"}):
            payload, status = api.emotions(body)
            assert status == 400, body
            assert payload["ok"] is False

    def test_a_refusal_comes_back_in_its_own_words(self, store):
        payload, status = api.emotions({"add": "  "})

        assert status == 400
        assert payload["error"]["message"] == "Write the emotion first."

    def test_it_is_registered(self):
        assert api.EMOTIONS_ROUTE in api.ROUTES


# --------------------------------------------------------------------------- #
# LLM Studio's tab
# --------------------------------------------------------------------------- #


class TestTheTab:
    @staticmethod
    def panel():
        import mc_llm_chat_panel

        return mc_llm_chat_panel

    def test_the_heading_marks_whether_the_chip_s_dot_is_filled(self, thread):
        panel = self.panel()

        assert 'data-emotion="off"' in panel._heading("Ada", thread)
        choose(thread, HAPPY)
        assert 'data-emotion="on"' in panel._heading("Ada", thread)

    def test_the_rows_are_off_and_the_list_with_the_thread_s_own_checked(self, thread):
        panel = self.panel()
        choose(thread, HAPPY)

        drawn = panel._emotion_rows("Ada", thread.identifier)

        assert drawn.count('class="mc-llm-emotion-choice"') == 1 + len(emotions.DEFAULTS)
        assert drawn.count('aria-checked="true"') == 1
        assert f'aria-checked="true" data-emotion="{HAPPY}"' in drawn
        assert drawn.count('class="mc-llm-emotion-delete"') == len(emotions.DEFAULTS)

    def test_with_no_thread_only_the_list_can_be_changed(self, store):
        drawn = self.panel()._emotion_rows("", "")

        assert drawn.count(" disabled>") == 1 + len(emotions.DEFAULTS)
        assert 'class="mc-llm-emotion-delete"' in drawn

    def test_a_line_is_written_into_the_page_escaped(self, store):
        emotions.add('*I am so <b>"bold"</b>*')

        drawn = self.panel()._emotion_rows("Ada", "t")

        assert "<b>" not in drawn
        assert "&lt;b&gt;&quot;bold&quot;&lt;/b&gt;" in drawn

    def test_a_press_chooses_through_the_service(self, thread):
        panel = self.panel()

        drawn = panel._emotion_pressed("Ada", thread.identifier,
                                       '{"do": "choose", "line": "*I am so happy*"}')

        assert emotions.for_thread("Ada", thread.identifier) == HAPPY
        assert 'data-emotion="on"' in drawn[3]

    def test_a_press_deletes_the_line(self, thread):
        panel = self.panel()

        panel._emotion_pressed("Ada", thread.identifier, '{"do": "delete", "line": "*I am so sad*"}')

        assert "*I am so sad*" not in emotions.listed()

    def test_a_press_it_cannot_read_changes_nothing(self, thread):
        panel = self.panel()
        before = emotions.listed()

        for nomination in ("", "not json", '{"do": "rename", "line": "x"}', "[1]"):
            panel._emotion_pressed("Ada", thread.identifier, nomination)

        assert emotions.listed() == before
        assert emotions.for_thread("Ada", thread.identifier) == ""

    def test_adding_a_line_chooses_it_and_empties_the_box(self, thread):
        found = self.panel()._emotion_added("Ada", thread.identifier, "*I am so nervous*")

        assert emotions.listed()[-1] == "*I am so nervous*"
        assert emotions.for_thread("Ada", thread.identifier) == "*I am so nervous*"
        assert found[-1] == ""

    def test_a_refused_line_is_said_and_the_box_kept(self, thread):
        import gradio as gr

        found = self.panel()._emotion_added("Ada", thread.identifier, "   ")

        assert "Write the emotion first" in found[-2]
        assert found[-1] == gr.update()


class TestTheTabSaysTheCacheIsUpdating:
    @staticmethod
    def statuses(name_who, name_thread):
        import mc_llm_chat_panel

        return mc_llm_chat_panel._cache_status(name_who, name_thread)

    def test_nothing_to_say_when_none_is_running(self, thread):
        import gradio as gr

        assert list(self.statuses("Ada", thread.identifier)) == [gr.update()]

    def test_quietly_while_it_runs_and_ready_after(self, thread):
        update = pending_update(name_of(thread))
        threading.Timer(0.3, update.done.set).start()

        said = list(self.statuses("Ada", thread.identifier))

        assert "Updating cache…" in said[0] and "mc-llm-updating" in said[0]
        assert "mc-llm-busy" not in said[0]
        assert "Ready." in said[-1]

    def test_it_stops_writing_once_a_message_is_on_its_way(self, thread, monkeypatch):
        import gradio as gr
        import mc_llm_chat_panel

        update = pending_update(name_of(thread))
        replying = iter([False, True])
        monkeypatch.setattr(mc_llm_chat_panel, "_replying",
                            lambda who, identifier: next(replying, True))
        try:
            said = list(self.statuses("Ada", thread.identifier))
        finally:
            update.done.set()

        assert "Updating cache…" in said[0]
        assert said[-1] == gr.update()
        assert not any("Ready." in str(line) for line in said)
