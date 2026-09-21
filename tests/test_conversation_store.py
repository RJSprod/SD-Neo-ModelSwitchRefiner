"""The guarded store, against the failures it was written for.

Every test here is a thing that happened, or a thing that could not be prevented
from happening, before there was a revision on a conversation:

    two windows, and the second one saving its ten-second-old copy over
    everything that arrived in between;

    two pages opening a thread in the same second and being handed the same
    file name, so the second one's conversation replaced the first's;

    a retry after a lost acknowledgement, and a second copy of somebody's
    message;

    a reply arriving into a thread that had been deleted, and the file coming
    back with one message in it.

None of those needs a GPU, a browser or a model, and none of them shows up in a
single-window test. What they need is two callers and a file.
"""

from __future__ import annotations

import json
import pathlib
import threading

import pytest

import mc_llm_conversation_store as store_module
import mc_llm_paths
from mc_llm_conversation_store import Key


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    store_module.registry.reset()
    yield tmp_path
    store_module.registry.reset()


@pytest.fixture
def chats(store):
    from prompt_master.chat.history import ChatStore

    return ChatStore(store / "chats")


def thread(chats, character="Ada", messages=("ask", "reply")):
    from prompt_master.chat.history import ASSISTANT, USER

    conversation = chats.new(character)
    for index, text in enumerate(messages):
        conversation.append(USER if index % 2 == 0 else ASSISTANT, text)
    chats.save(conversation)
    return conversation


def key_for(conversation):
    return Key(character=conversation.character, thread_id=conversation.identifier)


class TestRevisions:
    def test_a_legacy_file_reads_as_revision_zero_and_is_not_rewritten(self, chats):
        """H01. Drawing the thread list must not rewrite every file in the
        folder, which is what a store that wrote revisions on read would do."""
        conversation = thread(chats)
        path = chats.path_for("Ada", conversation.identifier)
        before = path.read_bytes()

        found, raw = store_module.read(chats, key_for(conversation))

        assert found.revision == 0
        assert path.read_bytes() == before
        assert isinstance(store_module.token_of(found, raw), str), (
            "a revisionless file compares by its bytes, not by a number")

    def test_the_first_guarded_write_makes_it_revision_one(self, chats):
        conversation = thread(chats)
        key = key_for(conversation)
        token = store_module.token(chats, key)

        committed = store_module.transaction(chats, key, token,
                                             lambda copy: copy.messages.pop(0))

        assert committed.revision == 1
        assert chats.load("Ada", conversation.identifier).revision == 1

    def test_every_committed_write_moves_it_by_exactly_one(self, chats):
        conversation = thread(chats)
        key = key_for(conversation)
        token = store_module.token(chats, key)

        seen = []
        for round_number in range(3):
            committed = store_module.transaction(
                chats, key, token, lambda copy: copy.append("user", "again"))
            seen.append(committed.revision)
            token = committed.revision

        assert seen == [1, 2, 3]

    def test_a_change_that_does_nothing_writes_nothing(self, chats):
        """A transaction whose function decides there is nothing to do must not
        move the revision: a counter that ticks for a write that did not happen
        is a counter that makes another window's next write fail for no reason.
        """
        conversation = thread(chats)
        key = key_for(conversation)
        path = chats.path_for("Ada", conversation.identifier)
        before = path.read_bytes()

        committed = store_module.transaction(chats, key, store_module.token(chats, key),
                                             lambda copy: False)

        assert committed.written is False
        assert committed.revision == 0
        assert path.read_bytes() == before

    def test_a_stale_revision_is_refused_and_writes_nothing(self, chats):
        """H04, at its smallest: one window at revision N, one at N-1."""
        conversation = thread(chats)
        key = key_for(conversation)
        first = store_module.token(chats, key)
        store_module.transaction(chats, key, first,
                                 lambda copy: copy.append("user", "from window one"))

        with pytest.raises(store_module.StaleRevision) as refusal:
            store_module.transaction(chats, key, first,
                                     lambda copy: copy.append("user", "from window two"))

        assert refusal.value.current == 1
        kept = chats.load("Ada", conversation.identifier)
        assert [message.text for message in kept.messages][-1] == "from window one"

    def test_a_legacy_fingerprint_is_a_comparison_a_second_window_can_lose(self, chats):
        conversation = thread(chats)
        key = key_for(conversation)
        fingerprint = store_module.token(chats, key)
        store_module.transaction(chats, key, fingerprint,
                                 lambda copy: copy.append("user", "first"))

        with pytest.raises(store_module.StaleRevision):
            store_module.transaction(chats, key, fingerprint,
                                     lambda copy: copy.append("user", "second"))

    def test_a_revision_that_is_not_a_revision_refuses_every_write(self, chats):
        """Never silently reset. A counter a reader repaired to zero is a
        counter two windows can then both win against."""
        conversation = thread(chats)
        path = chats.path_for("Ada", conversation.identifier)
        data = json.loads(path.read_text())
        data["revision"] = -4
        path.write_text(json.dumps(data))

        with pytest.raises(store_module.MetadataDamaged):
            store_module.transaction(chats, key_for(conversation), 0,
                                     lambda copy: copy.append("user", "x"))

    def test_true_is_not_revision_one(self, chats):
        """``True == 1`` in Python, which is the one wrong answer here that
        looks entirely plausible in a file listing."""
        conversation = thread(chats)
        path = chats.path_for("Ada", conversation.identifier)
        data = json.loads(path.read_text())
        data["revision"] = True
        path.write_text(json.dumps(data))

        found, _ = store_module.read(chats, key_for(conversation))

        assert found.damaged is True
        assert found.revision == 0


class TestConcurrency:
    def test_two_threads_writing_at_one_revision_produce_one_winner(self, chats):
        """The defect, reproduced. Both read at N; one commits; the other is
        refused rather than overwriting what the first one wrote."""
        conversation = thread(chats)
        key = key_for(conversation)
        token = store_module.token(chats, key)
        outcomes = []
        barrier = threading.Barrier(2)

        def write(word):
            barrier.wait()
            try:
                store_module.transaction(chats, key, token,
                                         lambda copy: copy.append("user", word))
                outcomes.append(("ok", word))
            except store_module.StoreRefusal as refusal:
                outcomes.append((refusal.code, word))

        threads = [threading.Thread(target=write, args=(word,))
                   for word in ("one", "two")]
        for item in threads:
            item.start()
        for item in threads:
            item.join(10)

        accepted = [word for state, word in outcomes if state == "ok"]
        refused = [state for state, _ in outcomes if state != "ok"]
        assert len(accepted) == 1, outcomes
        assert refused == ["STALE_REVISION"], outcomes
        kept = chats.load("Ada", conversation.identifier)
        assert kept.messages[-1].text == accepted[0]

    def test_one_lock_per_file_however_many_stores_are_made(self, chats, store):
        """This repository makes a new ``ChatStore`` per call. A registry that
        lived on the instance would hand two callers two different locks and
        neither would wait for the other."""
        from prompt_master.chat.history import ChatStore

        conversation = thread(chats)
        second = ChatStore(store / "chats")

        first_lock = store_module.registry.lock_for(
            store_module.canonical(chats, key_for(conversation)))
        second_lock = store_module.registry.lock_for(
            store_module.canonical(second, key_for(conversation)))

        assert first_lock is second_lock

    def test_a_busy_thread_refuses_a_competing_content_change(self, chats):
        """H08. While a reply owns the conversation, an edit is refused rather
        than queued -- a queue whose head is a language model is a UI that has
        silently hung."""
        conversation = thread(chats)
        key = key_for(conversation)
        store_module.claim(chats, key, "op-1")

        with pytest.raises(store_module.ThreadBusy) as refusal:
            store_module.transaction(chats, key, store_module.token(chats, key),
                                     lambda copy: copy.append("user", "x"))

        assert refusal.value.operation_id == "op-1"

    def test_the_owning_operation_may_still_write_its_own_reply(self, chats):
        conversation = thread(chats)
        key = key_for(conversation)
        store_module.claim(chats, key, "op-1")

        committed = store_module.transaction(
            chats, key, store_module.token(chats, key),
            lambda copy: copy.append("assistant", "the reply"), allow_busy=True)

        assert committed.written is True

    def test_reading_is_never_refused_while_a_reply_is_arriving(self, chats):
        """Nothing queues silently. Browsing, selecting and reading stay
        available while a thread is busy."""
        conversation = thread(chats)
        key = key_for(conversation)
        store_module.claim(chats, key, "op-1")

        found, _ = store_module.read(chats, key)

        assert [message.text for message in found.messages] == ["ask", "reply"]


class TestReceipts:
    def test_a_retried_operation_is_answered_rather_than_repeated(self, chats):
        """H06/H07. The acknowledgement was lost; the request was not. What
        comes back is the first attempt's answer, not a second message."""
        conversation = thread(chats)
        key = key_for(conversation)
        receipt = store_module.Receipt(operation_id="op-9", action="send",
                                       payload_digest="d")

        first = store_module.transaction(chats, key, store_module.token(chats, key),
                                         lambda copy: copy.append("user", "hello"),
                                         receipt=receipt)
        again = store_module.transaction(
            chats, key, first.revision, lambda copy: copy.append("user", "hello"),
            receipt=store_module.Receipt(operation_id="op-9", action="send",
                                         payload_digest="d"))

        assert again.duplicate is True
        assert again.written is False
        kept = chats.load("Ada", conversation.identifier)
        assert [message.text for message in kept.messages].count("hello") == 1

    def test_receipts_are_bounded(self, chats):
        conversation = thread(chats)
        key = key_for(conversation)
        token = store_module.token(chats, key)
        for number in range(store_module.MAX_RECEIPTS + 8):
            committed = store_module.transaction(
                chats, key, token,
                lambda copy: copy.append("user", str(number)),
                receipt=store_module.Receipt(operation_id=f"op-{number}", action="send"))
            token = committed.revision

        kept = chats.load("Ada", conversation.identifier)
        assert len(store_module.receipts_of(kept)) == store_module.MAX_RECEIPTS

    def test_a_receipt_survives_a_restart(self, chats):
        """The in-memory registry is gone with the process. The file is not,
        which is what makes a retry after a restart answerable."""
        conversation = thread(chats)
        key = key_for(conversation)
        store_module.transaction(chats, key, store_module.token(chats, key),
                                 lambda copy: copy.append("user", "hello"),
                                 receipt=store_module.Receipt(operation_id="op-3",
                                                              action="send"))
        store_module.registry.reset()

        reloaded = chats.load("Ada", conversation.identifier)

        assert store_module.receipt_for(reloaded, "op-3") is not None
        assert store_module.receipt_for(reloaded, "op-4") is None


class TestIdentifiers:
    def test_two_callers_in_the_same_second_get_different_names(self, chats):
        """H13. The old identifier looked for a free name and then used it,
        which is a check and a create with a gap between them."""
        names = {chats.new("Ada").identifier for _ in range(64)}

        assert len(names) == 64

    def test_the_name_is_reserved_rather_than_merely_chosen(self, chats):
        found = chats.new("Ada").identifier

        assert chats.path_for("Ada", found).exists(), (
            "the file was not created, so two processes could both be given it")

    def test_a_reserved_name_that_is_never_written_is_invisible(self, chats):
        chats.new("Ada")

        assert chats.listing("Ada") == []

    def test_a_deleted_identifier_is_never_handed_out_again(self, chats):
        conversation = thread(chats)
        chats.delete("Ada", conversation.identifier)

        assert chats.entombed("Ada", conversation.identifier) is True
        assert conversation.identifier not in {chats.new("Ada").identifier
                                               for _ in range(32)}

    def test_creating_into_a_tombstoned_thread_is_refused(self, chats):
        """A reply still in flight when its thread was deleted must not
        recreate the file with one message in it."""
        conversation = thread(chats)
        key = key_for(conversation)
        chats.delete("Ada", conversation.identifier)

        with pytest.raises(store_module.ConversationMissing):
            store_module.create(chats, key, lambda: conversation)


class TestContainment:
    """A character and a thread id arrive from a browser.

    Two defences, and the test is written against both because either alone is
    one change away from being the only one. ``safe_stem`` filters the name; the
    resolve-and-compare in ``canonical`` proves the result. A filter is not a
    proof -- it is a list of things somebody thought of -- which is why the
    second test subverts the filter and checks that the proof still refuses.
    """

    @pytest.mark.parametrize("thread_id", ["../escape", "../../etc/passwd",
                                           "..\\escape", "a/../../b",
                                           "....//....//x"])
    def test_a_key_never_resolves_outside_the_chats_folder(self, chats, thread_id):
        root = pathlib.Path(chats.directory).resolve()

        found = store_module.canonical(chats, Key(character="Ada", thread_id=thread_id))

        assert root in found.parents, found

    @pytest.mark.parametrize("character", ["../..", "/etc", "..\\.."])
    def test_a_character_never_resolves_outside_it_either(self, chats, character):
        root = pathlib.Path(chats.directory).resolve()

        found = store_module.canonical(chats, Key(character=character, thread_id="x"))

        assert root in found.parents, found

    def test_a_path_that_escapes_is_refused_even_so(self, chats, monkeypatch, tmp_path):
        """The proof, on its own. If the filter above ever stopped catching
        something -- a new separator, a symlink in the folder, a change to
        ``safe_stem`` -- this is what is left, so it is tested without it."""
        outside = tmp_path / "elsewhere" / "secret.json"
        monkeypatch.setattr(chats, "path_for", lambda character, identifier: outside)

        with pytest.raises(store_module.ConversationMissing):
            store_module.canonical(chats, Key(character="Ada", thread_id="x"))

    def test_an_empty_key_is_not_a_key(self, chats):
        with pytest.raises(store_module.ConversationMissing):
            store_module.canonical(chats, Key(character="", thread_id=""))

    def test_a_key_with_only_whitespace_is_not_a_key(self, chats):
        with pytest.raises(store_module.ConversationMissing):
            store_module.canonical(chats, Key(character="Ada", thread_id="   "))


class TestExternalChange:
    def test_a_file_replaced_under_the_lock_refuses_the_write(self, chats,
                                                              monkeypatch):
        """The transaction compares once more at the moment of writing, so a
        file changed by something outside this application is not overwritten
        by a transaction that read it a moment earlier."""
        conversation = thread(chats)
        key = key_for(conversation)
        token = store_module.token(chats, key)

        from prompt_master.chat.history import RevisionMismatch

        def moved(conversation_to_save, expected=None):
            raise RevisionMismatch(99)

        monkeypatch.setattr(chats, "save", moved)

        with pytest.raises(store_module.StorageChanged):
            store_module.transaction(chats, key, token,
                                     lambda copy: copy.append("user", "x"))

    def test_a_save_that_fails_is_reported_rather_than_claimed(self, chats,
                                                              monkeypatch):
        """H12. No false "Saved"."""
        conversation = thread(chats)
        key = key_for(conversation)
        token = store_module.token(chats, key)

        def broken(conversation_to_save, expected=None):
            raise OSError("the disk is full")

        monkeypatch.setattr(chats, "save", broken)

        with pytest.raises(store_module.SaveFailed):
            store_module.transaction(chats, key, token,
                                     lambda copy: copy.append("user", "x"))


class TestMaintenance:
    def test_adoption_runs_as_a_guarded_transaction(self, chats):
        conversation = thread(chats)
        key = key_for(conversation)

        committed = store_module.maintain(chats, key,
                                          lambda copy: copy.append("user", "moved"))

        assert committed.written is True
        assert committed.revision == 1

    def test_maintenance_does_nothing_while_a_reply_owns_the_thread(self, chats):
        """A background tidy-up must never be the reason a reply cannot be
        saved."""
        conversation = thread(chats)
        key = key_for(conversation)
        store_module.claim(chats, key, "op-1")

        committed = store_module.maintain(chats, key,
                                          lambda copy: copy.append("user", "moved"))

        assert committed.written is False
        assert chats.load("Ada", conversation.identifier).revision == 0

    def test_maintenance_that_finds_nothing_to_do_writes_nothing(self, chats):
        conversation = thread(chats)
        path = chats.path_for("Ada", conversation.identifier)
        before = path.read_bytes()

        store_module.maintain(chats, key_for(conversation), lambda copy: False)

        assert path.read_bytes() == before


class TestOrphans:
    def test_a_reserved_but_unwritten_file_is_reported_not_deleted(self, chats):
        """Tier 1's one accepted imperfection. An orphan is somebody's
        conversation as far as this code can tell, so it is named in the log
        and left where it is."""
        reserved = chats.new("Ada").identifier

        found = store_module.orphans(chats, "Ada")

        assert [path.stem for path in found] == [reserved]
        assert chats.path_for("Ada", reserved).exists()
