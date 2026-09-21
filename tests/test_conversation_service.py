"""The command service: what it accepts, what it refuses, and what it does once.

The envelope is where a browser meets a conversation, so most of this file is
about refusals. Three of them carry the whole design:

    an ``expected_revision`` that no longer matches -- the person acted on a
    conversation that has since moved, and the answer is a sentence and an
    intact draft rather than an overwrite;

    an ``operation_id`` that has been used before with the same content -- the
    acknowledgement was lost, not the request, and the answer is the first
    attempt's outcome rather than a second message;

    the same id with *different* content -- a bug in the caller, refused with
    no side effect at all, because guessing which of the two was meant is how
    somebody's message is quietly replaced by another.

The two authorised behaviour changes (specification S8) are here too, and they
are the two live data-loss paths at HEAD: "Send again from here" truncated, and
"Continue" on an earlier reply mutated a message with descendants under it.
"""

from __future__ import annotations

import pytest

import mc_llm_conversation_feed as feed
import mc_llm_conversation_ops as ops
import mc_llm_conversation_service as service
import mc_llm_conversation_store as store_module
import mc_llm_paths


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    service.reset()
    store_module.registry.reset()
    ops.reset()
    feed.reset()
    yield tmp_path
    service.reset()
    store_module.registry.reset()
    ops.reset()
    feed.reset()


@pytest.fixture
def chats(store):
    from prompt_master.chat.history import ChatStore

    return ChatStore(store / "chats")


@pytest.fixture
def conversation(chats):
    from prompt_master.chat.history import ASSISTANT, USER

    found = chats.new("Ada")
    found.append(USER, "ask one")
    found.append(ASSISTANT, "reply one")
    found.append(USER, "ask two")
    found.append(ASSISTANT, "reply two")
    chats.save(found)
    return found


CURRENT = object()
"""Ask the store what the thread compares as right now.

The default, because a chat saved through the plain store has *no* revision --
it compares by the hash of its bytes, exactly as every conversation written
before this existed does. A test that hard-coded revision 0 would be testing
the one case a real page never sends.
"""


def token_for(conversation):
    return store_module.token(service.chats(),
                              store_module.key_of(conversation))


def expected_for(token):
    if token is None:
        return None
    if isinstance(token, int):
        return {"kind": "revision", "value": token}
    return {"kind": "legacy", "fingerprint": token}


def envelope(action, conversation, *, revision=CURRENT, index=None, version=None,
             payload=None, operation_id="op-1", **extra):
    token = token_for(conversation) if revision is CURRENT else revision
    body = {"protocol_version": service.PROTOCOL_VERSION,
            "server_epoch": service.SERVER_EPOCH,
            "operation_id": operation_id,
            "action": action,
            "conversation": {"character": conversation.character,
                             "thread_id": conversation.identifier},
            "expected_revision": expected_for(token),
            "payload": payload or {}}
    if index is not None:
        body["target"] = {"index": index, "version": version}
    body.update(extra)
    return body


class TestTheEnvelope:
    def test_a_body_that_is_not_an_object_is_refused(self):
        outcome = service.submit(["not", "an", "object"])

        assert outcome["ok"] is False
        assert outcome["error"]["code"] == service.INVALID_INPUT

    def test_an_unsupported_protocol_version_is_refused(self, conversation):
        body = envelope("branch", conversation, index=0)
        body["protocol_version"] = 99

        outcome = service.submit(body)

        assert outcome["error"]["code"] == service.UNSUPPORTED_PROTOCOL

    def test_an_epoch_from_another_process_is_refused(self, conversation):
        body = envelope("branch", conversation, index=0)
        body["server_epoch"] = "from-a-process-that-has-gone"

        outcome = service.submit(body)

        assert outcome["error"]["code"] == service.SERVER_RESTARTED

    def test_a_field_the_server_does_not_understand_is_refused(self, conversation):
        """Never ignored. A newer client's "and also delete the rest" dropped
        silently by an older server is the worst possible reading of it."""
        body = envelope("branch", conversation, index=0)
        body["also_delete_everything"] = True

        outcome = service.submit(body)

        assert outcome["error"]["code"] == service.INVALID_INPUT
        assert "also_delete_everything" in outcome["error"]["message"]

    def test_an_unknown_payload_field_is_refused_too(self, conversation):
        body = envelope("rename_thread", conversation,
                        payload={"title": "x", "and_delete": True})

        outcome = service.submit(body)

        assert outcome["error"]["code"] == service.INVALID_INPUT

    def test_a_missing_expected_revision_is_refused(self, conversation):
        body = envelope("delete_message", conversation, index=0, revision=None)

        outcome = service.submit(body)

        assert outcome["error"]["code"] == service.INVALID_INPUT
        assert "expected_revision" in outcome["error"]["message"]

    def test_text_beyond_the_limit_is_refused_and_never_cut(self, conversation):
        body = envelope("rename_thread", conversation,
                        payload={"title": "x" * (service.MAX_TEXT_BYTES + 1)})

        outcome = service.submit(body)

        assert outcome["error"]["code"] == service.INVALID_INPUT

    def test_an_action_that_is_not_an_action_is_refused(self, conversation):
        body = envelope("delete_everything", conversation, index=0)

        outcome = service.submit(body)

        assert outcome["error"]["code"] == service.INVALID_INPUT

    @pytest.mark.parametrize("index", [-1, "2", 1.5, True])
    def test_a_target_index_that_is_not_an_index_is_refused(self, conversation, index):
        body = envelope("delete_message", conversation)
        body["target"] = {"index": index}

        outcome = service.submit(body)

        assert outcome["error"]["code"] == service.INVALID_INPUT


class TestDeduplication:
    def test_the_same_request_twice_is_applied_once(self, chats, conversation):
        body = envelope("rename_thread", conversation, payload={"title": "renamed"})

        first = service.submit(body)
        again = service.submit(dict(body))

        assert first["ok"] is True
        assert again["ok"] is True
        assert again["duplicate"] is True
        assert chats.load("Ada", conversation.identifier).revision == 1, (
            "the second attempt wrote a second time")

    def test_the_same_id_with_different_content_is_refused(self, chats, conversation):
        service.submit(envelope("rename_thread", conversation,
                                payload={"title": "first"}))

        outcome = service.submit(envelope("rename_thread", conversation,
                                          payload={"title": "second"}))

        assert outcome["error"]["code"] == service.OPERATION_ID_REUSED
        assert chats.load("Ada", conversation.identifier).title == "first"

    def test_a_retry_after_a_page_refresh_is_still_the_same_request(self, conversation):
        """The digest deliberately excludes ``page_id`` and
        ``selection_epoch``: a retry from a reloaded page is a different page
        with the same intention, and refusing it as a reuse would make a
        refresh mid-send produce nothing at all."""
        first = envelope("rename_thread", conversation, payload={"title": "x"},
                         page_id="page-a", selection_epoch="e1")
        second = envelope("rename_thread", conversation, payload={"title": "x"},
                          page_id="page-b", selection_epoch="e2")

        service.submit(first)
        outcome = service.submit(second)

        assert outcome["duplicate"] is True
        assert outcome["ok"] is True

    def test_a_refused_request_is_remembered_as_refused(self, conversation):
        body = envelope("delete_message", conversation, index=99)

        first = service.submit(body)
        again = service.submit(dict(body))

        assert first["ok"] is False
        assert again["ok"] is False
        assert again["duplicate"] is True


class TestMessageActions:
    def test_editing_a_message_leaves_what_follows_alone(self, chats, conversation):
        """4.2: in place, at any index. The replies under it are still there and
        still say what they said, which is the feature rather than a side
        effect -- the file is the only record of what was said."""
        outcome = service.submit(envelope("edit_message", conversation, index=0,
                                          payload={"text": "asked differently"}))

        assert outcome["ok"] is True
        kept = chats.load("Ada", conversation.identifier)
        assert [message.text for message in kept.messages] == [
            "asked differently", "reply one", "ask two", "reply two"]

    def test_an_absent_image_action_keeps_the_picture(self, chats, conversation):
        """Absence has meant *remove* in one shape of this code before, which is
        how an edited caption lost the photograph it was a caption for."""
        from prompt_master.chat.history import Message

        conversation.messages[0] = Message(role="user", versions=["ask one"],
                                           image_path="Ada/abc.jpg", image_name="a.jpg")
        chats.save(conversation)

        service.submit(envelope("edit_message", conversation, index=0,
                                payload={"text": "changed"}))

        kept = chats.load("Ada", conversation.identifier)
        assert kept.messages[0].image_path == "Ada/abc.jpg"
        assert kept.messages[0].image_name == "a.jpg"

    def test_removing_a_picture_takes_an_explicit_word_for_it(self, chats, conversation):
        from prompt_master.chat.history import Message

        conversation.messages[0] = Message(role="user", versions=["ask one"],
                                           image_path="Ada/abc.jpg")
        chats.save(conversation)

        service.submit(envelope("edit_message", conversation, index=0,
                                payload={"text": "changed", "image_action": "remove"}))

        assert chats.load("Ada", conversation.identifier).messages[0].image_path == ""

    def test_deleting_from_here_checks_the_count_it_was_shown(self, chats, conversation):
        """"Delete 2 messages" that arrives when there are now four is not a
        delete of four. It is a question the person has to be asked again."""
        outcome = service.submit(envelope("delete_from", conversation, index=2,
                                          payload={"confirm_count": 4}))

        assert outcome["error"]["code"] == service.STALE_REVISION
        assert len(chats.load("Ada", conversation.identifier).messages) == 4

    def test_deleting_from_here_with_the_right_count_works(self, chats, conversation):
        outcome = service.submit(envelope("delete_from", conversation, index=2,
                                          payload={"confirm_count": 2}))

        assert outcome["ok"] is True
        assert len(chats.load("Ada", conversation.identifier).messages) == 2

    def test_the_only_version_of_a_message_cannot_be_dropped(self, conversation):
        outcome = service.submit(envelope("drop_version", conversation, index=1))

        assert outcome["error"]["code"] == service.INVALID_INPUT

    def test_selecting_a_version_persists_and_moves_the_revision(self, chats,
                                                                 conversation):
        conversation.messages[1].add_version("another attempt")
        chats.save(conversation)

        outcome = service.submit(envelope("select_version", conversation, index=1,
                                          version=0))

        assert outcome["ok"] is True
        kept = chats.load("Ada", conversation.identifier)
        assert kept.messages[1].active == 0
        assert kept.revision == 1

    def test_a_stale_index_is_refused_rather_than_reinterpreted(self, chats,
                                                               conversation):
        """C01. Never "the current last message" -- that is how a delete aimed
        at message four deletes a message that arrived after it."""
        outcome = service.submit(envelope("delete_message", conversation, index=9))

        assert outcome["error"]["code"] == service.STALE_REVISION
        assert len(chats.load("Ada", conversation.identifier).messages) == 4

    def test_branching_leaves_the_original_untouched(self, chats, conversation):
        outcome = service.submit(envelope("branch", conversation, index=1))

        assert outcome["ok"] is True
        made = outcome["resulting_conversation"]["thread_id"]
        assert made != conversation.identifier
        assert [message.text for message in
                chats.load("Ada", conversation.identifier).messages] == [
                    "ask one", "reply one", "ask two", "reply two"]
        assert [message.text for message in chats.load("Ada", made).messages] == [
            "ask one", "reply one"]

    def test_a_branch_retried_returns_the_same_thread(self, chats, conversation):
        """A retry must not make a second branch. The key is in the receipt and
        comes back from it, never recomputed."""
        body = envelope("branch", conversation, index=1)

        first = service.submit(body)
        again = service.submit(dict(body))

        assert again["duplicate"] is True
        assert len(list((chats.folder("Ada")).glob("*.json"))) == 2


class TestTheUnansweredMessageLift:
    def test_it_is_a_command_and_it_writes(self, chats):
        """H14. Navigation writes, and always did. What changed is that it is
        a command somebody asked for rather than a side effect of opening a
        thread -- so the read-only refresh cannot reach it."""
        from prompt_master.chat.history import USER

        found = chats.new("Ada")
        found.append(USER, "never answered")
        chats.save(found)

        outcome = service.submit(envelope("restore_to_draft", found, index=0))

        assert outcome["ok"] is True
        assert chats.load("Ada", found.identifier).messages == []

    def test_only_the_last_message_can_be_taken_back(self, conversation):
        outcome = service.submit(envelope("restore_to_draft", conversation, index=0))

        assert outcome["error"]["code"] == service.INVALID_INPUT

    def test_a_message_carrying_a_picture_stays_in_the_thread(self, chats):
        from prompt_master.chat.history import USER

        found = chats.new("Ada")
        found.append(USER, "look at this", image_path="Ada/x.jpg")
        chats.save(found)

        outcome = service.submit(envelope("restore_to_draft", found, index=0))

        assert outcome["error"]["code"] == service.INVALID_INPUT
        assert len(chats.load("Ada", found.identifier).messages) == 1


class TestThreads:
    def test_a_new_thread_starts_at_revision_one(self, chats):
        body = {"protocol_version": service.PROTOCOL_VERSION,
                "server_epoch": service.SERVER_EPOCH, "operation_id": "op-new",
                "action": "create_thread",
                "conversation": {"character": "Ada", "thread_id": ""},
                "expected_revision": None, "payload": {}}

        outcome = service.submit(body)

        assert outcome["ok"] is True
        made = outcome["resulting_conversation"]["thread_id"]
        assert chats.load("Ada", made).revision == 1

    def test_deleting_a_thread_checks_the_title_it_was_shown(self, chats, conversation):
        outcome = service.submit(envelope("delete_thread", conversation,
                                          payload={"confirm_title": "a different name"}))

        assert outcome["error"]["code"] == service.STALE_REVISION
        assert chats.path_for("Ada", conversation.identifier).exists()

    def test_deleting_a_thread_tombstones_its_name(self, chats, conversation):
        outcome = service.submit(envelope("delete_thread", conversation))

        assert outcome["ok"] is True
        assert chats.entombed("Ada", conversation.identifier) is True

    def test_a_thread_being_replied_to_cannot_be_deleted(self, chats, conversation):
        store_module.claim(chats, store_module.key_of(conversation), "op-running")

        outcome = service.submit(envelope("delete_thread", conversation))

        assert outcome["error"]["code"] == service.THREAD_BUSY


class TestSnapshots:
    def test_a_snapshot_carries_no_filesystem_path(self, chats, conversation):
        from prompt_master.chat.history import Message

        conversation.messages[0] = Message(role="user", versions=["ask one"],
                                           image_path="Ada/abc.jpg", image_name="a.jpg")
        chats.save(conversation)

        found = service.snapshot("Ada", conversation.identifier)

        blob = repr(found)
        assert "Ada/abc.jpg" not in blob
        assert str(chats.directory) not in blob

    def test_a_snapshot_says_which_revision_it_is_of(self, conversation):
        found = service.snapshot("Ada", conversation.identifier)

        assert found["conversation"]["thread_id"] == conversation.identifier
        assert "revision" in found["conversation"]

    def test_a_snapshot_of_a_thread_that_is_gone_says_so(self, chats, conversation):
        chats.delete("Ada", conversation.identifier)

        found = service.snapshot("Ada", conversation.identifier)

        assert found["conversation"] is None
        assert found["error"]["code"] == service.NOT_FOUND

    def test_reading_a_snapshot_never_writes(self, chats, conversation):
        path = chats.path_for("Ada", conversation.identifier)
        before = path.read_bytes()

        service.snapshot("Ada", conversation.identifier)

        assert path.read_bytes() == before


class TestPublishingFailsOpen:
    def test_a_publish_that_raises_never_fails_a_save(self, chats, conversation,
                                                      monkeypatch):
        """P04. A successful save is a success whether or not anybody could be
        told about it -- the opposite would make a working write look like a
        failed one, which is the one thing worse than a window that has to be
        refreshed."""
        def explode(*args, **kwargs):
            raise RuntimeError("the feed is on fire")

        monkeypatch.setattr(feed, "publish", explode)

        outcome = service.submit(envelope("rename_thread", conversation,
                                          payload={"title": "renamed"}))

        assert outcome["ok"] is True
        assert chats.load("Ada", conversation.identifier).title == "renamed"
