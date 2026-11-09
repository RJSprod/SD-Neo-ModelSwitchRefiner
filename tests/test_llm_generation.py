"""A thread's generation settings: kept with the thread, read by every request in it.

The ask (2026-11-09): "It seems for LLM studio that my generation settings
(like temperature) are not being kept. I have to keep setting them again when i
return to the conversation. These should be kept per conversation. I would also
like more controls there ... Im not able to get random enough output, and
infact, i get a lot of repetition."

``mc_llm_sessions.conversation`` is replaced with a list of events, as in
``test_conversation_ops``: the request is built by the real builder from the
real files, and what it would send is read off it.
"""

from __future__ import annotations

import json

import pytest

import mc_llm_conversation_service as service
import mc_llm_generation as generation
import mc_llm_sessions as sessions
from test_conversation_ops import chats, run, store, thread  # noqa: F401 (fixtures)
from test_conversation_service import envelope


def asking(monkeypatch, pieces=("Oh,", " hello!")):
    asked = []
    events = [sessions.Event(sessions.CHUNK, piece) for piece in pieces]
    events.append(sessions.Event(sessions.DONE, "".join(pieces)))
    monkeypatch.setattr(sessions, "conversation",
                        lambda request, cancel: (asked.append(request), iter(events))[1])
    return asked


def created(character="Ada", operation_id="create-1", payload=None):
    body = {"protocol_version": service.PROTOCOL_VERSION,
            "server_epoch": service.SERVER_EPOCH, "operation_id": operation_id,
            "action": service.CREATE_THREAD,
            "conversation": {"character": character, "thread_id": ""},
            "expected_revision": None,
            "payload": payload or {}}
    return service.submit(body)


# --------------------------------------------------------------------------- #
# The controls
# --------------------------------------------------------------------------- #


class TestTheControls:
    def test_the_fields_sent_beyond_the_four_are_the_ones_the_client_lets_through(self):
        """Two lists that must agree, in two packages: a key the client dropped
        would be a control that moves and changes nothing."""
        from prompt_master.inference.llama_client import REQUEST_SAMPLER_FIELDS
        from prompt_master.models.managed_profiles import SAMPLER_FIELDS

        assert set(generation.SENT_IF_CHOSEN) == set(REQUEST_SAMPLER_FIELDS)
        assert set(SAMPLER_FIELDS) <= set(REQUEST_SAMPLER_FIELDS), \
            "a profile's field the thread could not choose over"
        assert set(generation.SENT_ALWAYS) == {"temperature", "top_p", "reply_tokens", "seed"}

    def test_every_default_is_in_its_own_range(self):
        for setting in generation.SETTINGS:
            assert setting.minimum <= setting.default <= setting.maximum, setting.key
            assert setting.step > 0, setting.key

    def test_the_ranges_are_ones_llama_server_accepts(self):
        """b10621 refuses these below zero (``server-schema.cpp``'s hard limits,
        no -1 for "the whole context"), clamps the probabilities into 0..1 and
        disables XTC above a threshold of 0.5."""
        by = generation.BY_KEY
        for key in ("repeat_last_n", "dry_penalty_last_n", "dry_allowed_length", "top_k"):
            assert by[key].minimum >= 0 and by[key].whole, key
        for key in ("top_p", "min_p", "xtc_probability"):
            assert (by[key].minimum, by[key].maximum) == (0.0, 1.0), key
        assert by["xtc_threshold"].maximum == 0.5
        assert by["dry_base"].minimum == 1.0, "llama-server replaces a base under 1"
        assert by["repeat_penalty"].minimum == 1.0, "under 1 rewards repeating"

    def test_the_ranges_reach_past_one_reply(self):
        """The penalties and DRY look back 64 tokens by default -- less than
        one reply -- so a phrase repeated from an earlier reply is penalised
        only once a range is raised; the controls have to reach well past it."""
        assert generation.BY_KEY["repeat_last_n"].maximum >= 2048
        assert generation.BY_KEY["dry_penalty_last_n"].maximum >= 4096
        assert generation.BY_KEY["repeat_last_n"].default == 64
        assert generation.BY_KEY["dry_penalty_last_n"].default == 64

    def test_the_defaults_are_llama_servers_own(self):
        """``common_params_sampling`` in b10621's ``common/common.h``."""
        expected = {"min_p": 0.05, "top_k": 40, "xtc_probability": 0.0, "xtc_threshold": 0.1,
                    "dynatemp_range": 0.0, "repeat_penalty": 1.0, "repeat_last_n": 64,
                    "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
                    "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": 64}
        assert {key: generation.BY_KEY[key].default for key in expected} == expected

    def test_a_value_is_held_in_its_range_and_whole_where_it_is_whole(self):
        top_k = generation.BY_KEY["top_k"]
        assert generation.value_of(top_k, 900) == 200
        assert generation.value_of(top_k, 39.6) == 40
        assert isinstance(generation.value_of(top_k, 39.6), int)
        assert generation.value_of(generation.BY_KEY["min_p"], -1) == 0.0
        for nothing in (None, True, "many", float("nan"), float("inf")):
            assert generation.value_of(top_k, nothing) is None, nothing


# --------------------------------------------------------------------------- #
# What a thread keeps
# --------------------------------------------------------------------------- #


class TestWhatAThreadKeeps:
    def test_a_choice_is_kept_beside_the_threads_by_character_and_thread(self, store):
        generation.choose("Ada", "t1", "temperature", 1.2)
        generation.choose("Ada", "t1", "dry_multiplier", 0.8)

        assert generation.for_thread("Ada", "t1") == {"temperature": 1.2,
                                                      "dry_multiplier": 0.8}
        assert generation.for_thread("Ada", "t2") == {}
        assert generation.path() == store / "generation.json"
        written = json.loads(generation.path().read_text(encoding="utf-8"))
        assert written == {"version": 1, "threads": {"Ada/t1": {"temperature": 1.2,
                                                                 "dry_multiplier": 0.8}}}

    def test_none_forgets_one_choice_and_the_last_forgets_the_thread(self, store):
        generation.choose("Ada", "t1", "top_k", 80)
        generation.choose("Ada", "t1", "min_p", 0.1)

        assert generation.choose("Ada", "t1", "top_k", None) == {"min_p": 0.1}
        generation.choose("Ada", "t1", "min_p", None)
        assert "Ada/t1" not in json.loads(generation.path().read_text())["threads"]

    def test_what_is_not_a_setting_or_not_a_number_is_refused(self, store):
        with pytest.raises(generation.Refused):
            generation.choose("Ada", "t1", "logit_bias", 1)
        with pytest.raises(generation.Refused):
            generation.choose("Ada", "t1", "top_k", "many")
        with pytest.raises(generation.Refused):
            generation.choose("Ada", "", "top_k", 10)
        assert not generation.path().exists()

    def test_a_file_that_cannot_be_read_is_no_choices_and_no_crash(self, store):
        generation.path().write_text("{not json", encoding="utf-8")

        assert generation.for_thread("Ada", "t1") == {}

    def test_a_value_out_of_range_in_the_file_is_held_in_range(self, store):
        generation.path().write_text(json.dumps({"version": 1, "threads": {
            "Ada/t1": {"top_k": 5000, "dry_base": 0.2, "logit_bias": 3, "min_p": "x"}}}))

        assert generation.for_thread("Ada", "t1") == {"top_k": 200, "dry_base": 1.0}

    def test_carry_gives_a_thread_another_threads_choices(self, store):
        generation.choose("Ada", "t1", "xtc_probability", 0.5)

        generation.carry("Ada", "t1", "t2")
        generation.carry("Ada", "nothing-chosen", "t3")

        assert generation.for_thread("Ada", "t2") == {"xtc_probability": 0.5}
        assert generation.for_thread("Ada", "t3") == {}

    def test_reset_and_forget_drop_them(self, store):
        generation.choose("Ada", "t1", "seed", 7)
        generation.choose("Ada", "t2", "seed", 8)

        generation.reset("Ada", "t1")
        generation.forget("Ada", "t2")

        assert generation.for_thread("Ada", "t1") == {} == generation.for_thread("Ada", "t2")


# --------------------------------------------------------------------------- #
# What applies without a choice
# --------------------------------------------------------------------------- #


class Profile:
    def __init__(self, sampling):
        self.sampling = sampling


class Configuration:
    def __init__(self, model=None, profile=None):
        self.model = model
        self.profile = profile


class TestWhatAppliesWithoutAChoice:
    def test_the_packages_four_and_llama_servers_defaults(self):
        """"It should be default until I override for a thread" (2026-11-09):
        the package's own for the four, whatever a character's file holds."""
        from prompt_master.chat import characters

        found = generation.base(configuration=Configuration())

        assert (found["temperature"], found["top_p"], found["reply_tokens"],
                found["seed"]) == (characters.DEFAULT_TEMPERATURE, characters.DEFAULT_TOP_P,
                                   characters.DEFAULT_MAX_REPLY_TOKENS, -1)
        assert found["top_k"] == 40 and found["min_p"] == 0.05
        assert found["dry_multiplier"] == 0.0 and found["repeat_last_n"] == 64

    def test_the_model_files_recommendation_then_the_profiles_field(self, monkeypatch, tmp_path):
        """llama-server's order: its default, the model file's ``general.sampling.*``
        (its default for every request that does not say), then what the
        profile sends with every request."""
        import mc_gguf

        class Described:
            metadata = {"general.sampling.top_k": 64, "general.sampling.min_p": 0.0,
                        "general.sampling.penalty_repeat": "1.05",
                        "general.sampling.penalty_last_n": 256,
                        "general.sampling.temp": 1.0}

        model = tmp_path / "model.gguf"
        monkeypatch.setattr(mc_gguf, "describe", lambda path: Described())

        from_the_file = generation.base(configuration=Configuration(model=model))
        assert (from_the_file["top_k"], from_the_file["min_p"],
                from_the_file["repeat_penalty"], from_the_file["repeat_last_n"]) == (
                    64, 0.0, 1.05, 256)
        assert from_the_file["temperature"] == 0.85, "the four are the package's"

        with_a_profile = generation.base(configuration=Configuration(
            model=model, profile=Profile({"top_k": 20, "repeat_penalty": 1.1})))
        assert (with_a_profile["top_k"], with_a_profile["repeat_penalty"],
                with_a_profile["min_p"]) == (20, 1.1, 0.0)

    def test_a_model_that_cannot_be_read_recommends_nothing(self, monkeypatch, tmp_path):
        import mc_gguf

        def broken(path):
            raise mc_gguf.GgufError("no")

        monkeypatch.setattr(mc_gguf, "describe", broken)

        assert generation.model_recommendations(Configuration(model=tmp_path / "x.gguf")) == {}

    def test_in_force_is_the_choice_over_the_default(self):
        defaults = generation.base(configuration=Configuration())

        found = generation.in_force({"top_k": 0, "temperature": 1.3}, defaults)

        assert found["top_k"] == 0 and found["temperature"] == 1.3
        assert found["min_p"] == defaults["min_p"]

    def test_the_label_says_what_the_thread_chose(self):
        assert generation.label({}) == "Generation"
        assert generation.label({"temperature": 1.2}) == "Generation · Temperature 1.2"
        assert generation.label({"temperature": 1.2, "dry_multiplier": 0.8, "seed": -1,
                                 "top_k": 50}) == (
            "Generation · Temperature 1.2 · Top-k 50 · +2")
        assert generation.shown(generation.BY_KEY["seed"], -1) == "random"


# --------------------------------------------------------------------------- #
# Every request in the thread
# --------------------------------------------------------------------------- #


class TestEveryRequestInTheThread:
    def test_a_reply_is_made_from_what_the_thread_chose(self, thread, monkeypatch):
        asked = asking(monkeypatch)
        for key, value in (("temperature", 1.3), ("top_p", 0.9), ("reply_tokens", 900),
                           ("seed", 4242), ("dry_multiplier", 0.8),
                           ("dry_penalty_last_n", 2048), ("min_p", 0.1)):
            generation.choose("Ada", thread.identifier, key, value)

        run("send", thread, payload={"text": "Again?"})

        request = asked[-1]
        assert (request.temperature, request.top_p, request.max_tokens, request.seed) == (
            1.3, 0.9, 900, 4242)
        assert request.sampling == {"dry_multiplier": 0.8, "dry_penalty_last_n": 2048,
                                    "min_p": 0.1}

    def test_a_thread_that_chose_nothing_is_sampled_at_the_defaults(self, chats, thread,
                                                                    monkeypatch):
        """The package's four -- never the four in the character's file, which
        nothing shows any more -- and nothing beyond them for the profile and
        llama-server to be overruled with."""
        from dataclasses import replace

        import mc_llm_conversation_service as svc
        from prompt_master.chat import characters
        from prompt_master.core import models

        held = svc._characters()
        held.save(replace(held.load("Ada"), temperature=0.4, max_reply_tokens=300, seed=11))
        monkeypatch.setattr(models, "draw_seed", lambda: 1111)
        asked = asking(monkeypatch)

        run("send", thread, payload={"text": "Again?"})

        request = asked[-1]
        assert (request.temperature, request.top_p, request.max_tokens, request.seed) == (
            characters.DEFAULT_TEMPERATURE, characters.DEFAULT_TOP_P,
            characters.DEFAULT_MAX_REPLY_TOKENS, 1111)
        assert request.sampling == {}

    def test_every_view_and_every_way_of_asking_uses_them(self, chats, thread, monkeypatch):
        """Send, Regenerate, Continue and Send again from here; the panel sends
        no settings at all, and neither does the tab any more."""
        asked = asking(monkeypatch)
        generation.choose("Ada", thread.identifier, "xtc_probability", 0.5)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")
        last = len(chats.load("Ada", thread.identifier).messages) - 1
        run("regenerate", thread, index=last, operation_id="again-1")
        run("continue", thread, index=last, operation_id="more-1")

        assert len(asked) == 3
        assert all(request.sampling == {"xtc_probability": 0.5} for request in asked)

    def test_a_request_naming_its_own_settings_wins_for_that_request(self, thread,
                                                                     monkeypatch):
        asked = asking(monkeypatch)
        generation.choose("Ada", thread.identifier, "temperature", 1.3)
        generation.choose("Ada", thread.identifier, "top_k", 80)

        run("send", thread, payload={"text": "Hi.", "settings": {"temperature": 0.2,
                                                                 "top_k": 10}})

        assert asked[-1].temperature == 0.2 and asked[-1].sampling == {"top_k": 10}
        assert generation.for_thread("Ada", thread.identifier) == {"temperature": 1.3,
                                                                   "top_k": 80}

    def test_the_seed_is_new_each_reply_unless_the_thread_chose_one(self, thread,
                                                                    monkeypatch):
        """The 2026-11-09 Regenerate that came back the same: the box showed -1
        while the character's file held a seed, and a request's -1 left that
        seed in force. The file's seed is no thread's; a thread's own is kept,
        and a request's -1 still says nothing."""
        from dataclasses import replace

        import mc_llm_conversation_service as svc
        from prompt_master.core import models

        held = svc._characters()
        held.save(replace(held.load("Ada"), seed=99))
        drawn = iter([1111, 2222])
        monkeypatch.setattr(models, "draw_seed", lambda: next(drawn))
        asked = asking(monkeypatch)

        run("send", thread, payload={"text": "Hi."}, operation_id="send-1")
        assert asked[-1].seed == 1111, "the character's leftover seed is not the default"

        generation.choose("Ada", thread.identifier, "seed", 4242)
        run("send", thread, payload={"text": "Hi again.", "settings": {"seed": -1}},
            operation_id="send-2")
        assert asked[-1].seed == 4242, "a request's -1 says nothing, as it always said"

        generation.choose("Ada", thread.identifier, "seed", -1)
        run("send", thread, payload={"text": "And again."}, operation_id="send-3")
        assert asked[-1].seed == 2222

    def test_the_tabs_send_does_not_overrule_them(self, thread, monkeypatch):
        """The reported bug: the tab sent its sliders with every reply, and they
        were drawn at the package's defaults whenever the page was built -- so
        a choice kept in a file was not the one in force. The tab's Send names
        no settings now; the thread's are read where the request is built."""
        import mc_llm_chat_panel
        import mc_llm_conversation_ops as ops

        asked = asking(monkeypatch)
        generation.choose("Ada", thread.identifier, "temperature", 1.3)
        generation.choose("Ada", thread.identifier, "xtc_probability", 0.5)

        list(mc_llm_chat_panel._send("Ada", thread.identifier, "Again?", None))
        assert ops.drain(timeout=10)

        assert asked[-1].temperature == 1.3
        assert asked[-1].sampling == {"xtc_probability": 0.5}

    def test_the_service_takes_the_new_names_and_refuses_others(self, thread, monkeypatch):
        asking(monkeypatch)
        body = envelope("send", thread, payload={"text": "Hi.",
                                                 "settings": {"logit_bias": 1}})

        refused = service.submit(body)

        assert refused["ok"] is False and refused["error"]["code"] == service.INVALID_INPUT
        for key in generation.KEYS:
            assert service._settings({key: 1}) == {key: 1}, key


class TestTheConversationReadAgain:
    """After a change of emotion the thread is read again in the background, and
    its prompt has to begin as the next reply's will -- the history trimmed to
    the same window, which the reply's budget decides."""

    def _built_with(self, monkeypatch):
        from prompt_master.chat import prompt

        seen = []
        real = prompt.build

        def recording(*args, **kwargs):
            seen.append(kwargs.get("reply_tokens"))
            return real(*args, **kwargs)

        monkeypatch.setattr(prompt, "build", recording)
        return seen

    def test_with_the_threads_own_budget_else_the_default(self, thread, monkeypatch):
        from dataclasses import replace

        import mc_llm_conversation_ops as ops
        import mc_llm_conversation_service as svc
        import mc_llm_conversation_store as store_module
        from prompt_master.chat import characters

        held = svc._characters()
        held.save(replace(held.load("Ada"), max_reply_tokens=300))
        seen = self._built_with(monkeypatch)
        key = store_module.key_of(thread)

        ops._thread_request(key)
        generation.choose("Ada", thread.identifier, "reply_tokens", 900)
        ops._thread_request(key)

        assert seen == [characters.DEFAULT_MAX_REPLY_TOKENS, 900]


class TestTheClientIsAskedOnlyWhenThereIsSomething:
    """A client from before the keyword -- or a test double -- is called
    exactly as it always was for a thread that chose nothing."""

    class Old:
        def __init__(self):
            self.calls = []

        def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                        temperature=0.85, top_p=0.95):
            self.calls.append({"temperature": temperature})
            on_text("ok")
            return "ok"

    class New(Old):
        def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                        temperature=0.85, top_p=0.95, sampling=None):
            self.calls.append({"temperature": temperature, "sampling": sampling})
            on_text("ok")
            return "ok"

    @pytest.fixture
    def installed(self, monkeypatch, host):
        import mc_broker

        mc_broker.clear()
        monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
        holder = {}
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False, reserve=0, role='', cancel=None:
                            holder["client"])
        monkeypatch.setattr(sessions, "_placement_notes", lambda role="": [])
        yield holder
        mc_broker.clear()

    def test_nothing_chosen_asks_nothing_new(self, installed):
        installed["client"] = client = self.Old()
        request = sessions.ChatRequest(messages=[{"role": "user", "content": "hi"}])

        events = list(sessions.conversation(request, sessions.Cancellation()))

        assert events[-1].kind == sessions.DONE and client.calls == [{"temperature": 0.85}]

    def test_what_was_chosen_reaches_the_client(self, installed):
        installed["client"] = client = self.New()
        request = sessions.ChatRequest(messages=[{"role": "user", "content": "hi"}],
                                       sampling={"dry_multiplier": 0.8})

        list(sessions.conversation(request, sessions.Cancellation()))

        assert client.calls[-1]["sampling"] == {"dry_multiplier": 0.8}


# --------------------------------------------------------------------------- #
# New threads, branches and deletes
# --------------------------------------------------------------------------- #


class TestThreadsMadeFromAThread:
    def test_a_new_thread_starts_at_the_defaults_whatever_was_open(self, thread):
        """"It should be default until I override for a thread. And then kept
        for that thread through sessions" (2026-11-09): a thread tuned is not
        where a new one starts, from the tab or from the panel."""
        generation.choose("Ada", thread.identifier, "dry_multiplier", 0.8)

        outcome = created()

        made = outcome["resulting_conversation"]["thread_id"]
        assert outcome["ok"] is True and made != thread.identifier
        assert generation.for_thread("Ada", made) == {}
        assert generation.for_thread("Ada", thread.identifier) == {"dry_multiplier": 0.8}

    def test_a_thread_to_start_from_is_not_something_it_takes(self, thread):
        """The first build of this round named one (``like``) and carried its
        settings; the service no longer understands the field."""
        refused = created(payload={"like": thread.identifier})

        assert refused["ok"] is False and refused["error"]["code"] == service.INVALID_INPUT

    def test_a_branch_carries_them(self, thread):
        generation.choose("Ada", thread.identifier, "top_k", 0)

        outcome = service.submit(envelope("branch", thread, index=1))

        made = outcome["resulting_conversation"]["thread_id"]
        assert generation.for_thread("Ada", made) == {"top_k": 0}

    def test_a_deleted_thread_forgets_them(self, thread):
        generation.choose("Ada", thread.identifier, "top_k", 0)

        outcome = service.submit(envelope("delete_thread", thread))

        assert outcome["ok"] is True, outcome
        assert generation.for_thread("Ada", thread.identifier) == {}

    def test_choosing_writes_nothing_in_the_thread(self, chats, thread):
        """A setting, not a change to what was said: the conversation's own
        file -- and so its revision -- is never touched."""
        written = chats.path_for("Ada", thread.identifier).read_bytes()

        generation.choose("Ada", thread.identifier, "temperature", 1.5)

        assert chats.path_for("Ada", thread.identifier).read_bytes() == written
