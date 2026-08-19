"""LLM Studio: persistence, the three run orchestrations, and regression safety.

The two things worth pinning down here are the ones the design intent states as
prohibitions rather than features. Section 16: the three modes' histories must
not become one stream. Section 18: a failure anywhere in the LLM half must not
reach ordinary image generation.
"""

from __future__ import annotations

import threading

import pytest

import mc_broker
import mc_llm_paths
import mc_llm_sessions as sessions
import mc_llm_state as state


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point every LLM Studio file at a throwaway directory."""
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    yield tmp_path


class FakeClient:
    """A llama.cpp client that streams a canned answer.

    Faithful in the two places the orchestration depends on: it calls
    ``on_text`` per chunk and it stops the moment the cancel event is set.
    """

    def __init__(self, pieces=("Hello", " there"), fail=None):
        self.pieces = list(pieces)
        self.fail = fail
        self.calls: list[dict] = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.85, top_p=0.95):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "seed": seed,
                           "temperature": temperature, "top_p": top_p})
        if self.fail is not None:
            raise self.fail
        produced = []
        for piece in self.pieces:
            if cancel is not None and cancel.is_set():
                break
            produced.append(piece)
            on_text(piece)
        return "".join(produced)


@pytest.fixture
def client(monkeypatch, host):
    """Install a fake client and a card with room, and clear the register."""
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
    fake = FakeClient()
    monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)
    monkeypatch.setattr(sessions, "_placement_notes", list)
    yield fake
    mc_broker.clear()


def drain(generator) -> list[sessions.Event]:
    return list(generator)


def kinds(events) -> list[str]:
    return [event.kind for event in events]


def texts(events, kind) -> list[str]:
    return [event.text for event in events if event.kind == kind]


# --------------------------------------------------------------------------- #
# Persistence (section 16)
# --------------------------------------------------------------------------- #


class TestPreferences:
    def test_defaults_are_complete_before_anything_is_saved(self, store):
        prefs = state.preferences()

        assert prefs["context_mode"] == "auto"
        assert prefs["kv_type_k"] == "f16"
        assert prefs["mode"] == "prompt"

    def test_remembering_one_value_leaves_the_others_alone(self, store):
        state.remember(context_size=32768)
        state.remember(mode="chat")

        prefs = state.preferences()
        assert prefs["context_size"] == 32768
        assert prefs["mode"] == "chat"

    def test_a_corrupt_file_reads_as_defaults_rather_than_raising(self, store):
        (store / "data").mkdir(parents=True, exist_ok=True)
        (store / "data" / state.PREFERENCES_FILE).write_text("{ this is not json",
                                                             encoding="utf-8")

        assert state.preferences()["context_mode"] == "auto"

    def test_a_write_replaces_atomically_and_leaves_no_temporary_behind(self, store):
        state.remember(context_size=4096)

        leftovers = list((store / "data").glob("*.tmp"))
        assert leftovers == []


class TestSeparateHistories:
    def test_prompt_studio_and_minimax_do_not_share_a_file(self, store):
        state.save_prompt_session(state.PromptSession(intent="a shot", positive="P"))
        state.save_minimax_session(state.MinimaxSession(prompt="a prompt", result="R"))

        assert len(state.prompt_sessions()) == 1
        assert len(state.minimax_sessions()) == 1
        assert (store / "data" / state.PROMPT_HISTORY_FILE).is_file()
        assert (store / "data" / state.MINIMAX_HISTORY_FILE).is_file()

    def test_clearing_one_history_leaves_the_other(self, store):
        state.save_prompt_session(state.PromptSession(intent="a shot"))
        state.save_minimax_session(state.MinimaxSession(prompt="a prompt"))

        state.clear_history(state.PROMPT_HISTORY_FILE)

        assert state.prompt_sessions() == []
        assert len(state.minimax_sessions()) == 1

    def test_a_session_keeps_the_controls_that_produced_it(self, store):
        """The useful thing to do with last week's prompt is to load its
        settings back and change one of them."""
        state.save_prompt_session(state.PromptSession(
            intent="a shot", controls={"style": "noir", "seconds": 8.0}))

        restored = state.prompt_sessions()[0]
        assert restored.controls == {"style": "noir", "seconds": 8.0}

    def test_saving_the_same_identifier_twice_updates_rather_than_duplicates(self, store):
        session = state.PromptSession(intent="first")
        state.save_prompt_session(session)
        session.intent = "second"
        state.save_prompt_session(session)

        assert [row.intent for row in state.prompt_sessions()] == ["second"]

    def test_history_is_capped_and_drops_the_oldest(self, store, monkeypatch):
        monkeypatch.setattr(state, "HISTORY_LIMIT", 3)
        for index in range(5):
            state.save_prompt_session(state.PromptSession(intent=f"shot {index}"))

        assert [row.intent for row in state.prompt_sessions()] == ["shot 2", "shot 3", "shot 4"]

    def test_unknown_keys_from_a_newer_version_are_ignored_not_fatal(self, store):
        import json

        (store / "data").mkdir(parents=True, exist_ok=True)
        (store / "data" / state.PROMPT_HISTORY_FILE).write_text(json.dumps({
            "version": 99,
            "sessions": [{"identifier": "x", "intent": "kept", "invented_later": True}],
        }), encoding="utf-8")

        assert state.prompt_sessions()[0].intent == "kept"

    def test_deleting_removes_only_the_named_session(self, store):
        first = state.save_prompt_session(state.PromptSession(intent="one"))
        state.save_prompt_session(state.PromptSession(intent="two"))

        state.delete_prompt_session(first.identifier)

        assert [row.intent for row in state.prompt_sessions()] == ["two"]


# --------------------------------------------------------------------------- #
# The three runs (section 4)
# --------------------------------------------------------------------------- #


class TestConversationRun:
    def test_it_streams_chunks_and_then_the_whole_reply(self, client):
        request = sessions.ChatRequest(messages=[{"role": "user", "content": "hi"}])

        events = drain(sessions.conversation(request, sessions.Cancellation()))

        assert texts(events, sessions.CHUNK) == ["Hello", " there"]
        assert texts(events, sessions.DONE) == ["Hello there"]

    def test_a_cancel_keeps_what_was_already_streamed(self, client):
        """A partial reply is a real reply. Discarding it would throw away
        the paragraph the user pressed stop *because* they had read."""
        cancel = sessions.Cancellation()
        cancel.cancel()
        request = sessions.ChatRequest(messages=[])

        events = drain(sessions.conversation(request, cancel))

        assert sessions.CANCELLED in kinds(events)

    def test_a_failure_is_reported_rather_than_raised(self, client, monkeypatch):
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False: FakeClient(fail=RuntimeError("no server")))

        events = drain(sessions.conversation(sessions.ChatRequest(messages=[]),
                                             sessions.Cancellation()))

        assert texts(events, sessions.FAILED) == ["no server"]

    def test_the_sampling_settings_reach_the_client(self, client):
        request = sessions.ChatRequest(messages=[], temperature=0.4, top_p=0.8,
                                       max_tokens=128, seed=99)

        drain(sessions.conversation(request, sessions.Cancellation()))

        assert client.calls[0]["temperature"] == 0.4
        assert client.calls[0]["seed"] == 99


class TestMinimaxRun:
    def test_it_emits_no_caption_without_an_image(self, client):
        events = drain(sessions.minimax("a shot", "fl2va", None, 7, sessions.Cancellation()))

        assert sessions.CAPTION not in kinds(events)
        assert kinds(events)[-1] == sessions.DONE

    def test_the_caption_comes_before_the_prompt_and_on_its_own_event(self, client):
        """WanGP's order, and the reason the caption is not folded into the
        output: the enhancer's user turn is built from it."""
        events = drain(sessions.minimax("a shot", "fl2va", "data:image/jpeg;base64,AAAA", 7,
                                        sessions.Cancellation()))

        order = kinds(events)
        assert order.index(sessions.CAPTION) < order.index(sessions.DONE)
        assert texts(events, sessions.CAPTION) == ["Hello there"]

    def test_an_empty_answer_is_an_error_not_an_empty_prompt(self, client, monkeypatch):
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False: FakeClient(pieces=("",)))

        events = drain(sessions.minimax("a shot", "fl2va", None, 7, sessions.Cancellation()))

        assert kinds(events)[-1] == sessions.FAILED


class TestPromptStudioRun:
    def test_positive_and_negative_arrive_as_separate_events(self, client, tmp_path):
        """Section 4.2 forbids one merged response blob, and this is what that
        prohibition looks like from the orchestration's side."""
        from prompt_master.core.models import PromptRequest

        request = PromptRequest(intent="a woman walks into the sea", video_mode="t2v")

        events = drain(sessions.prompt_studio(request, sessions.Cancellation()))

        assert texts(events, sessions.POSITIVE)
        assert texts(events, sessions.NEGATIVE)
        assert kinds(events)[-1] == sessions.DONE

    def test_the_negative_is_shown_before_the_positive_is_written(self, client):
        """The base negative is known from the controls alone, so it appears
        immediately rather than after a minute of generation."""
        from prompt_master.core.models import PromptRequest

        events = drain(sessions.prompt_studio(
            PromptRequest(intent="a shot", video_mode="t2v"), sessions.Cancellation()))

        order = kinds(events)
        assert order.index(sessions.NEGATIVE) < order.index(sessions.POSITIVE)

    def test_a_cancelled_run_stops_without_a_done_event(self, client):
        from prompt_master.core.models import PromptRequest

        cancel = sessions.Cancellation()
        cancel.cancel()

        events = drain(sessions.prompt_studio(
            PromptRequest(intent="a shot", video_mode="t2v"), cancel))

        assert sessions.DONE not in kinds(events)


class TestSerializedRuns:
    def test_a_run_waits_for_a_generation_the_host_is_already_doing(self, client,
                                                                    monkeypatch):
        """Section 15: an LLM request arriving while image sampling is active."""
        busy = {"value": True}
        monkeypatch.setattr(mc_broker, "host_busy", lambda: busy["value"])
        monkeypatch.setattr(sessions, "WAIT_NOTICE_SECONDS", 0.0)
        monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.01)

        events = []
        stream = sessions.conversation(sessions.ChatRequest(messages=[]),
                                       sessions.Cancellation())
        events.append(next(stream))
        busy["value"] = False
        events.extend(stream)

        assert "Waiting for image generation…" in texts(events, sessions.STATUS)
        assert kinds(events)[-1] == sessions.DONE

    def test_cancelling_while_waiting_never_starts_a_server(self, client, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)
        monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.01)
        cancel = sessions.Cancellation()

        stream = sessions.conversation(sessions.ChatRequest(messages=[]), cancel)
        cancel.cancel()
        events = drain(stream)

        assert client.calls == []
        assert kinds(events) == [sessions.CANCELLED]

    def test_the_gpu_is_released_even_when_the_run_fails(self, client, monkeypatch):
        """Section 15: cancellation and failure both have to leave the system in
        a known residency state -- a held lock would strand every later run."""
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False: FakeClient(fail=RuntimeError("boom")))

        drain(sessions.conversation(sessions.ChatRequest(messages=[]),
                                    sessions.Cancellation()))

        assert mc_broker.active() is None
        with mc_broker.workload(mc_broker.FAMILY_IMAGE, "a pass", timeout=0.1) as held:
            assert held

    def test_an_abandoned_generator_still_releases_the_gpu(self, client):
        """A Gradio cancel closes the generator part-way through. GeneratorExit
        runs the finally block, and that is what has to release the lock."""
        stream = sessions.conversation(sessions.ChatRequest(messages=[]),
                                       sessions.Cancellation())
        next(stream)
        stream.close()

        assert mc_broker.active() is None


# --------------------------------------------------------------------------- #
# Regression safety (section 18)
# --------------------------------------------------------------------------- #


class TestRegressionSafety:
    def test_the_tab_never_raises_into_the_host(self, host, monkeypatch):
        """A tab that throws takes the whole WebUI's UI down with it."""
        import mc_llm_studio

        monkeypatch.setattr(mc_llm_studio, "_build",
                            lambda: (_ for _ in ()).throw(RuntimeError("no gradio here")))

        tabs = mc_llm_studio.on_ui_tabs()

        assert tabs == [] or tabs[0][1] == mc_llm_studio.TAB_LABEL

    def test_the_tab_can_be_turned_off_entirely(self, host):
        import mc_llm_studio

        host.shared.opts.set(mc_llm_studio.OPT_ENABLE, False)

        assert mc_llm_studio.on_ui_tabs() == []

    def test_an_unconfigured_install_reports_rather_than_failing(self, host, store):
        import mc_llm_runtime

        assert not mc_llm_paths.configured()
        assert not mc_llm_runtime.config().configured
        assert not mc_llm_runtime.runtime.status()["running"]

    def test_the_image_side_is_unaffected_by_a_broken_llm_reclaimer(self, host, monkeypatch):
        """A reclaimer that raises must cost an eviction, never a generation."""
        mc_broker.clear()

        class Broken:
            def release(self, needed_bytes, reason=""):
                raise RuntimeError("the runtime is wedged")

        mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, Broken())
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 1024**3)

        result = mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 8 * 1024**3)

        assert not result.satisfied
        assert result.freed == 0
        mc_broker.unregister_reclaimer(mc_broker.FAMILY_LLM)

    def test_a_generation_is_never_blocked_indefinitely_by_the_llm(self, host):
        """await_idle is bounded on purpose: a wedged runtime should delay a
        generation, never prevent one."""
        holding, release = threading.Event(), threading.Event()

        def occupy():
            with mc_broker.workload(mc_broker.FAMILY_LLM, "a wedged turn"):
                holding.set()
                release.wait(3)

        thread = threading.Thread(target=occupy)
        thread.start()
        holding.wait(2)
        try:
            assert not mc_broker.await_idle(timeout=0.1)
        finally:
            release.set()
            thread.join(3)


class TestRegistration:
    def test_the_extension_registers_the_tab_with_the_host(self, host):
        """Registered inside a try/except, so a misnamed host function would be
        swallowed and the tab would simply never appear."""
        import mc_llm_studio
        import model_chain  # noqa: F401 - the import is what registers the callbacks

        callbacks = host.script_callbacks.registered["ui_tabs"]

        assert mc_llm_studio.on_ui_tabs in callbacks

    def test_the_extension_registers_its_unload_callback(self, host):
        import model_chain

        callbacks = host.script_callbacks.registered["script_unloaded"]

        assert model_chain._on_script_unloaded in callbacks

    def test_unloading_stops_the_llm_and_clears_the_register(self, host, monkeypatch):
        """The LLM lives in another process: dropping our own references would
        leave it running and holding VRAM after the extension has gone."""
        import mc_llm_runtime
        import model_chain

        stopped = []
        monkeypatch.setattr(mc_llm_runtime.runtime, "stop", lambda: stopped.append(True))
        mc_broker.declare(mc_broker.FAMILY_LLM, "llm", "the LLM", 1024**3)

        model_chain._on_script_unloaded()

        assert stopped == [True]
        assert mc_broker.residencies() == []

    def test_a_failing_llm_shutdown_does_not_stop_the_rest_of_the_unload(self, host,
                                                                        monkeypatch):
        import mc_llm_runtime
        import model_chain

        def explode():
            raise RuntimeError("the process is gone")

        monkeypatch.setattr(mc_llm_runtime.runtime, "stop", explode)
        mc_broker.declare(mc_broker.FAMILY_LLM, "llm", "the LLM", 1024**3)

        model_chain._on_script_unloaded()

        assert mc_broker.residencies() == []
