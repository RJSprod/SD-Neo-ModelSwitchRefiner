"""A thread's injected words (``mc_llm_injection``): the list, and the two ways it steers.

"Is there a way to make a list of words the the LLM becomes more likely to
include in its response? Like a word injection into the mind of the LLM that
can be used to subconsciously steer it." ... "I like all three options, give it
a menu for 'Injection'" (2026-11-09).

These hold the list as read (a strength after a word; ``Route 66`` a phrase),
the forms each word is sent in, the odds as llama-server's ``logit_bias`` -- the
first piece of a word its tokenizer splits, a ban winning -- the note at the end
of the system prompt, and, through the real service, store and builder with only
the model stood in, what each reply actually carries: the odds and never a
prompt change for them, the note in every prompt the thread builds -- compaction's
too -- read again in the background when it changes, a branch keeping the list
and a deleted thread forgetting it.
"""

from __future__ import annotations

import logging
import math

import pytest

import mc_llm_cache_update as cache_update
import mc_llm_compact as compact
import mc_llm_conversation_feed as feed
import mc_llm_conversation_ops as ops
import mc_llm_conversation_service as service
import mc_llm_conversation_store as store_module
import mc_llm_injection as injection
import mc_llm_paths
import mc_llm_sessions as sessions
import mc_llm_tokens
from test_llm_context_window import Model, command, long_thread, send, system_of, words


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(ops, "_context_size", lambda: 3000)
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


class Tokenizer:
    """The model's tokenizer as ``Counter.pieces`` answers: a table, else one
    token per form numbered by its text."""

    exact = True

    def __init__(self, table=None):
        self.table = dict(table or {})
        self.asked = []

    def pieces(self, text):
        self.asked.append(text)
        if text in self.table:
            return self.table[text]
        return [(1000 + sum(map(ord, text)), text)]


def ids(tokenizer, text):
    return tokenizer.pieces(text)[0][0]


# -- the list ------------------------------------------------------------------ #


class TestReadingTheList:
    def test_a_word_with_no_strength_is_given_the_default(self):
        entries, left = injection.parse("rain")

        assert [(entry.word, entry.strength, entry.banned) for entry in entries] == [
            ("rain", injection.DEFAULT_STRENGTH, False)]
        assert left == []

    @pytest.mark.parametrize("written, word, strength", [
        ("rain +3", "rain", 3.0), ("rain -1.5", "rain", -1.5), ("rain: 2", "rain", 2.0),
        ("rain = -4", "rain", -4.0), ("rain:.5", "rain", 0.5), ("gentle rain +1", "gentle rain", 1.0),
    ])
    def test_a_strength_follows_its_word_with_a_sign_a_colon_or_an_equals(self, written,
                                                                          word, strength):
        entries, _ = injection.parse(written)

        assert [(entry.word, entry.strength) for entry in entries] == [(word, strength)]

    def test_a_number_without_a_sign_is_part_of_the_phrase(self):
        """``Route 66`` is a phrase, not Route at 66."""
        entries, _ = injection.parse("Route 66")

        assert [(entry.word, entry.strength) for entry in entries] == [
            ("Route 66", injection.DEFAULT_STRENGTH)]

    @pytest.mark.parametrize("written", ["delve ban", "delve BAN", "delve: ban"])
    def test_ban_bans(self, written):
        entries, _ = injection.parse(written)

        assert entries[0].word == "delve" and entries[0].banned and entries[0].avoided
        assert entries[0].shown == "ban"

    def test_a_word_that_merely_ends_in_ban_is_a_word(self):
        entries, _ = injection.parse("urban\nban")

        assert [(entry.word, entry.banned) for entry in entries] == [("urban", False),
                                                                     ("ban", False)]

    def test_lines_and_commas_part_the_list_and_quotes_come_off(self):
        entries, _ = injection.parse('rain +2, lantern\n  "quiet" \n\n“ember” -1')

        assert [entry.word for entry in entries] == ["rain", "lantern", "quiet", "ember"]

    def test_a_strength_is_held_within_the_limit(self):
        entries, _ = injection.parse("rain +40\nsleet -99")

        assert [entry.strength for entry in entries] == [injection.STRENGTH_LIMIT,
                                                         -injection.STRENGTH_LIMIT]

    def test_a_word_listed_twice_is_kept_once_where_first_listed_at_its_last_strength(self):
        entries, _ = injection.parse("rain +1\nlantern\nRAIN -2")

        assert [(entry.word, entry.strength) for entry in entries] == [("RAIN", -2.0),
                                                                       ("lantern", 2.0)]

    def test_too_many_words_and_a_word_too_long_are_left_out_and_named(self):
        listed = "\n".join(f"word{index}" for index in range(injection.MAX_WORDS + 2))
        entries, left = injection.parse(listed + "\n" + "x" * (injection.MAX_WORD_CHARS + 1))

        assert len(entries) == injection.MAX_WORDS
        assert len(left) == 3
        assert "past the first" in left[0] and "longer than" in left[2]


class TestTheFormsSent:
    def test_a_boosted_word_is_sent_after_a_space_as_written_and_with_a_capital(self):
        entry = injection.parse("rain")[0][0]

        assert injection.forms(entry) == (" rain", " Rain")

    def test_a_word_made_rarer_is_sent_without_the_space_too(self):
        """A line's first word, or one after an asterisk, has no space before it."""
        rarer = injection.parse("rain -2")[0][0]
        banned = injection.parse("rain ban")[0][0]

        assert injection.forms(rarer) == (" rain", " Rain", "rain", "Rain")
        assert injection.forms(banned) == injection.forms(rarer)

    def test_a_word_written_with_its_capital_is_sent_once(self):
        entry = injection.parse("Elara")[0][0]

        assert injection.forms(entry) == (" Elara",)


# -- the odds ------------------------------------------------------------------- #


class TestTheOdds:
    def test_each_form_is_one_token_at_its_strength(self):
        tokenizer = Tokenizer()
        entries = injection.parse("rain +3")[0]

        bias = injection.logit_bias(entries, tokenizer)

        assert bias == [[ids(tokenizer, " rain"), 3.0], [ids(tokenizer, " Rain"), 3.0]]

    def test_a_split_form_has_its_first_piece_changed_alone(self):
        """llama-server would change every piece of it, and ``estry`` is in
        other words."""
        tokenizer = Tokenizer({" tapestry": [(7, " tap"), (8, "estry")],
                               " Tapestry": [(9, " Tap"), (8, "estry")]})

        bias = injection.logit_bias(injection.parse("tapestry +2")[0], tokenizer)

        assert bias == [[7, 2.0], [9, 2.0]]

    def test_a_ban_is_false_and_wins_over_a_strength_on_the_same_token(self):
        tokenizer = Tokenizer({" tap": [(7, " tap")], " tapestry": [(7, " tap"), (8, "estry")]})

        bias = injection.logit_bias(injection.parse("tap +2\ntapestry ban")[0], tokenizer)

        assert [7, False] in bias
        assert not any(pair[0] == 7 and pair[1] is not False for pair in bias)

    def test_two_words_sharing_a_first_piece_add_up_within_the_limit(self):
        tokenizer = Tokenizer({" lumen": [(5, " lum"), (6, "en")],
                               " luminous": [(5, " lum"), (7, "inous")],
                               " Lumen": [(50, " Lumen")], " Luminous": [(51, " Luminous")]})

        bias = dict((pair[0], pair[1]) for pair in injection.logit_bias(
            injection.parse("lumen +6\nluminous +6")[0], tokenizer))

        assert bias[5] == injection.STRENGTH_LIMIT

    def test_a_word_at_no_strength_sends_nothing(self):
        tokenizer = Tokenizer()

        assert injection.logit_bias(injection.parse("rain +0")[0], tokenizer) == []
        assert tokenizer.asked == []

    def test_with_no_tokenizer_the_forms_go_as_text_for_the_server_to_read(self):
        class Gone:
            def pieces(self, text):
                return None

        bias = injection.logit_bias(injection.parse("rain +2\ndelve ban")[0], Gone())

        assert [" rain", 2.0] in bias and [" Rain", 2.0] in bias
        assert ["delve", False] in bias and [" Delve", False] in bias


# -- the note -------------------------------------------------------------------- #


class TestTheNote:
    def test_it_names_the_words_to_use_and_the_words_to_avoid(self):
        note = injection.note(injection.parse("rain +2\nlantern\ntapestry -4\ndelve ban")[0])

        lines = note.splitlines()
        assert lines[0] == injection.NOTE_HEADING
        assert lines[1].endswith("rain, lantern.")
        assert lines[2] == "Avoid these words: tapestry, delve."
        assert lines[3] == "Do not mention this note."

    def test_nothing_to_say_is_no_note(self):
        assert injection.note([]) == ""
        assert injection.note(injection.parse("rain +0")[0]) == ""

    def test_only_words_to_avoid_has_no_sentence_to_use(self):
        note = injection.note(injection.parse("delve ban")[0])

        assert "use some of these" not in note and "Avoid these words: delve." in note


class TestHowEachWordReads:
    def test_a_split_word_says_so_and_names_the_piece_changed(self):
        tokenizer = Tokenizer({" tapestry": [(7, " tap"), (8, "estry")]})

        found = injection.described(injection.parse("rain\ntapestry -2")[0], tokenizer)

        assert found[0] == {"word": "rain", "strength": "+2", "banned": False,
                            "pieces": ["rain"], "note": ""}
        assert found[1]["pieces"] == ["tap", "estry"]
        assert "only ‘tap’ is changed" in found[1]["note"]

    def test_without_a_tokenizer_nothing_is_said_about_pieces(self):
        found = injection.described(injection.parse("rain")[0], None)

        assert found[0]["pieces"] is None and found[0]["note"] == ""


# -- each thread's list ---------------------------------------------------------- #


class TestTheThreadsList:
    def test_nothing_kept_is_the_defaults_odds_on_and_prompt_off(self):
        assert injection.for_thread("Ada", "t1") == injection.Record(words="", odds=True,
                                                                     prompt=False)

    def test_what_is_saved_is_kept_beside_the_threads(self, store):
        injection.save("Ada", "t1", words="rain +2", prompt=True)

        assert injection.for_thread("Ada", "t1") == injection.Record("rain +2", True, True)
        assert (store / injection.FILE_NAME).is_file()
        assert injection.for_thread("Ada", "t2") == injection.Record()

    def test_what_is_not_given_stays_as_it_was(self):
        injection.save("Ada", "t1", words="rain", odds=False)
        injection.save("Ada", "t1", prompt=True)

        assert injection.for_thread("Ada", "t1") == injection.Record("rain", False, True)

    def test_back_at_the_defaults_the_thread_is_dropped_from_the_file(self, store):
        import json

        injection.save("Ada", "t1", words="rain")
        injection.save("Ada", "t1", words="")

        kept = json.loads((store / injection.FILE_NAME).read_text(encoding="utf-8"))
        assert kept["threads"] == {}

    def test_no_thread_and_a_list_too_long_are_refused(self):
        with pytest.raises(injection.Refused):
            injection.save("Ada", "", words="rain")
        with pytest.raises(injection.Refused):
            injection.save("Ada", "t1", words="x" * (injection.MAX_TEXT_BYTES + 1))

    def test_a_branch_keeps_its_threads_list_and_a_deleted_thread_forgets_its_own(self):
        injection.save("Ada", "t1", words="rain", prompt=True)

        injection.carry("Ada", "t1", "t2")
        injection.forget("Ada", "t1")

        assert injection.for_thread("Ada", "t2") == injection.Record("rain", True, True)
        assert injection.for_thread("Ada", "t1") == injection.Record()

    def test_a_file_that_cannot_be_read_is_no_list_said_without_its_path(self, store, caplog):
        (store / injection.FILE_NAME).write_text("{not json", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="model_chain"):
            assert injection.for_thread("Ada", "t1") == injection.Record()
        assert "injected words could not be read" in caplog.text
        assert str(store) not in caplog.text

    def test_the_odds_and_the_note_follow_their_switches(self):
        injection.save("Ada", "t1", words="rain\ndelve ban", odds=False, prompt=False)
        assert injection.odds_for("Ada", "t1") == ()
        assert injection.note_for("Ada", "t1") == ""

        injection.save("Ada", "t1", odds=True, prompt=True)
        assert [entry.word for entry in injection.odds_for("Ada", "t1")] == ["rain", "delve"]
        assert injection.note_for("Ada", "t1").startswith(injection.NOTE_HEADING)

    @pytest.mark.parametrize("record, said", [
        (injection.Record(), "Injection"),
        (injection.Record("rain"), "Injection · 1 word · odds"),
        (injection.Record("rain\nember", True, True), "Injection · 2 words · odds and prompt"),
        (injection.Record("rain", False, True), "Injection · 1 word · prompt"),
        (injection.Record("rain", False, False), "Injection · 1 word, off"),
    ])
    def test_the_label_says_what_is_in_force(self, record, said):
        assert injection.label(record) == said


# -- through the service, the store and the builder ---------------------------------- #


def thread_of(conversation):
    return store_module.Key(character=conversation.character,
                            thread_id=conversation.identifier)


class TestWhatAReplyCarries:
    def test_a_reply_carries_the_odds_and_its_prompt_says_nothing_of_them(self, chats,
                                                                          monkeypatch):
        conversation = long_thread(chats, exchanges=2)
        injection.save("Ada", conversation.identifier, words="rain +3\ndelve ban")
        model = Model(monkeypatch, reply=words(120))

        send(conversation, "hello", "s-1")

        asked = model.asked[-1]
        assert [(entry.word, entry.strength, entry.banned) for entry in asked.injection] == [
            ("rain", 3.0, False), ("delve", 0.0, True)]
        assert injection.NOTE_HEADING not in system_of(asked)

    def test_with_the_odds_off_a_reply_carries_none(self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=2)
        injection.save("Ada", conversation.identifier, words="rain", odds=False)
        model = Model(monkeypatch, reply=words(120))

        send(conversation, "hello", "s-1")

        assert model.asked[-1].injection == ()

    def test_told_the_model_the_note_ends_the_system_prompt_before_the_emotion(
            self, chats, monkeypatch):
        import mc_llm_emotions

        conversation = long_thread(chats, exchanges=2)
        injection.save("Ada", conversation.identifier, words="rain", prompt=True)
        mc_llm_emotions.choose("Ada", conversation.identifier,
                               mc_llm_emotions.listed()[0])
        model = Model(monkeypatch, reply=words(120))

        send(conversation, "hello", "s-1")

        system = system_of(model.asked[-1])
        assert injection.NOTE_HEADING in system and "[Current emotion]" in system
        assert system.index(injection.NOTE_HEADING) < system.index("[Current emotion]")

    def test_the_note_is_in_the_request_the_cache_update_reads_again(self, chats):
        conversation = long_thread(chats, exchanges=2)
        injection.save("Ada", conversation.identifier, words="rain", prompt=True)

        request = ops._thread_request(thread_of(conversation))

        assert injection.NOTE_HEADING in request.messages[0]["content"]
        assert request.injection == (), "a read of the cache weighs no odds"

    def test_the_meter_counts_the_note_as_soon_as_the_model_is_told(self, chats):
        conversation = long_thread(chats, exchanges=2)
        before = ops.context_reading(thread_of(conversation))["used"]

        injection.save("Ada", conversation.identifier, words="rain\nlantern\nember",
                       prompt=True)

        assert ops.context_reading(thread_of(conversation))["used"] > before

    def test_compaction_writes_the_summary_in_the_frame_that_carries_the_note(
            self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=6)
        injection.save("Ada", conversation.identifier, words="rain", prompt=True)
        seen = {}
        framed = compact._wire

        def wire(*args):
            seen["wire"] = framed(*args)
            return seen["wire"]

        monkeypatch.setattr(compact, "_wire", wire)
        monkeypatch.setattr(compact, "_run", lambda *args: None)

        compact.start(thread_of(conversation), keep_tokens=300, summary_tokens=200)

        assert injection.NOTE_HEADING in seen["wire"][0]["content"]


class TestTheCommand:
    def test_it_keeps_the_list_against_no_revision_and_answers_how_it_reads(self, chats):
        conversation = long_thread(chats, exchanges=1)

        outcome = command(service.SET_INJECTION, conversation,
                          payload={"words": "rain +2", "odds": True, "prompt": False})

        assert outcome["ok"], outcome
        assert outcome["injection"]["entries"][0]["word"] == "rain"
        assert injection.for_thread("Ada", conversation.identifier) == injection.Record(
            "rain +2", True, False)

    def test_a_change_of_the_note_reads_the_conversation_again_and_the_odds_alone_do_not(
            self, chats, monkeypatch):
        conversation = long_thread(chats, exchanges=1)
        read = []
        monkeypatch.setattr(ops, "rewarm", lambda key: read.append(key) or True)

        command(service.SET_INJECTION, conversation, payload={"words": "rain"},
                operation_id="w-1")
        command(service.SET_INJECTION, conversation, payload={"odds": False},
                operation_id="w-2")
        assert read == []

        command(service.SET_INJECTION, conversation, payload={"prompt": True},
                operation_id="w-3")
        command(service.SET_INJECTION, conversation, payload={"words": "rain\nember"},
                operation_id="w-4")
        assert len(read) == 2

    def test_a_switch_that_is_not_true_or_false_is_refused(self, chats):
        conversation = long_thread(chats, exchanges=1)

        outcome = command(service.SET_INJECTION, conversation, payload={"odds": "yes"})

        assert not outcome["ok"]
        assert outcome["error"]["code"] == service.INVALID_INPUT

    def test_the_snapshot_says_how_the_threads_words_read(self, chats):
        conversation = long_thread(chats, exchanges=1)
        injection.save("Ada", conversation.identifier, words="rain", prompt=True)

        found = service.snapshot("Ada", conversation.identifier)

        assert found["injection"]["words"] == "rain"
        assert found["injection"]["prompt"] is True
        assert found["injection"]["label"] == "Injection · 1 word · odds and prompt"
        assert "control_vector" in found

    def test_a_branch_keeps_the_list_and_a_delete_forgets_it(self, chats):
        from test_conversation_service import envelope

        conversation = long_thread(chats, exchanges=2)
        injection.save("Ada", conversation.identifier, words="rain", prompt=True)
        current, _ = store_module.read(service.chats(), thread_of(conversation))

        branched = service.submit(envelope("branch", current, index=1, operation_id="b-1"))
        assert branched["ok"], branched
        made = branched["resulting_conversation"]["thread_id"]
        assert injection.for_thread("Ada", made) == injection.Record("rain", True, True)

        current, _ = store_module.read(service.chats(), thread_of(conversation))
        deleted = service.submit(envelope("delete_thread", current, operation_id="d-1",
                                          payload={"confirm_title": current.title}))
        assert deleted["ok"], deleted
        assert injection.for_thread("Ada", conversation.identifier) == injection.Record()


class TestTheReplyResolvesItsWords:
    """``mc_llm_sessions``: the words are read into the answering server's tokens
    as the reply is sent, and given to the client as ``logit_bias``."""

    def request(self, **kwargs):
        return sessions.ChatRequest(messages=[{"role": "user", "content": "hi"}], **kwargs)

    def test_a_request_with_words_asks_the_client_for_them(self, monkeypatch):
        tokenizer = Tokenizer()
        monkeypatch.setattr(mc_llm_tokens, "for_client", lambda client, ratio=0.0: tokenizer)

        found = sessions._injected(self.request(injection=tuple(injection.parse("rain")[0])),
                                   object())

        assert found == {"logit_bias": [[ids(tokenizer, " rain"), 2.0],
                                        [ids(tokenizer, " Rain"), 2.0]]}

    def test_a_request_without_asks_nothing(self, monkeypatch):
        monkeypatch.setattr(mc_llm_tokens, "for_client",
                            lambda *args: pytest.fail("no words, no tokenizer"))

        assert sessions._injected(self.request(), object()) == {}

    def test_words_that_cannot_be_read_leave_the_reply_without_them(self, monkeypatch, caplog):
        def broken(client, ratio=0.0):
            raise RuntimeError("tokenizer gone")

        monkeypatch.setattr(mc_llm_tokens, "for_client", broken)

        with caplog.at_level(logging.WARNING, logger="model_chain"):
            found = sessions._injected(self.request(injection=tuple(injection.parse("rain")[0])),
                                       object())

        assert found == {}
        assert "written without them" in caplog.text

    def test_the_client_is_given_them_beside_the_threads_sampling(self, monkeypatch):
        tokenizer = Tokenizer()
        monkeypatch.setattr(mc_llm_tokens, "for_client", lambda client, ratio=0.0: tokenizer)
        given = {}

        class Client:
            def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None, **kwargs):
                given.update(kwargs)
                on_text("ok")
                return "ok"

        monkeypatch.setattr(sessions, "_client", lambda *args, **kwargs: Client())
        monkeypatch.setattr(sessions, "_placement_notes", lambda: [])

        class Gpu:
            def __init__(self, *args, **kwargs):
                pass

            def acquire(self):
                if False:
                    yield
                return True

            def release(self):
                pass

        monkeypatch.setattr(sessions, "_Gpu", Gpu)
        request = self.request(injection=tuple(injection.parse("rain +1")[0]),
                               sampling={"min_p": 0.1})

        events = list(sessions._conversation(request, sessions.Cancellation()))

        assert events[-1].kind == sessions.DONE
        assert given["logit_bias"] == [[ids(tokenizer, " rain"), 1.0],
                                       [ids(tokenizer, " Rain"), 1.0]]
        assert given["sampling"] == {"min_p": 0.1}


class TestTheClientSendsThem:
    def test_the_vendored_client_keeps_only_well_formed_pairs(self):
        from prompt_master.inference.llama_client import request_bias

        found = request_bias([[5, 2.0], [" rain", -1], [7, False], [8, True], ["", 1.0],
                              [9, math.inf], [True, 1.0], (10, 3), "junk", [11]])

        assert found == {"logit_bias": [[5, 2.0], [" rain", -1], [7, False], [10, 3]]}
        assert request_bias([]) == {} and request_bias(None) == {}

    def test_the_vendored_client_puts_them_in_the_request_body(self, monkeypatch):
        from prompt_master.inference import llama_client

        sent = {}

        class Response:
            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(())

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class Http:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def stream(self, method, url, json=None):
                sent.update(json)
                return Response()

        monkeypatch.setattr(llama_client.httpx, "Client", Http)
        client = llama_client.LlamaClient.__new__(llama_client.LlamaClient)
        client.base_url, client.api_key, client.sampling = "http://127.0.0.1:1", "k", {}

        client.stream_chat([{"role": "user", "content": "hi"}], 8, 1, lambda text: None,
                           logit_bias=[[5, 2.0]])

        assert sent["logit_bias"] == [[5, 2.0]]

    def test_the_saving_client_builds_the_same_body(self):
        import mc_llm_slot_cache

        made = mc_llm_slot_cache._define_client()
        client = made.__new__(made)
        client.sampling = {}

        body = client.request_body([{"role": "user", "content": "hi"}], 8, 1, 0.8, 0.9,
                                   {"min_p": 0.1}, [[5, 2.0]])

        assert body["logit_bias"] == [[5, 2.0]] and body["min_p"] == 0.1
        assert "logit_bias" not in client.request_body(
            [{"role": "user", "content": "hi"}], 8, 1, 0.8, 0.9)
