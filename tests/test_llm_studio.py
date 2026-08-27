"""LLM Studio: persistence, the three run orchestrations, and regression safety.

The two things worth pinning down here are the ones the design intent states as
prohibitions rather than features. Section 16: the three modes' histories must
not become one stream. Section 18: a failure anywhere in the LLM half must not
reach ordinary image generation.
"""

from __future__ import annotations

import os
import types

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
                    temperature=0.85, top_p=0.95, extra_sampling=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "seed": seed,
                           "temperature": temperature, "top_p": top_p,
                           "extra_sampling": dict(extra_sampling or {})})
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
    monkeypatch.setattr(sessions, "_client", lambda needs_vision=False, reserve=0, role='', cancel=None: fake)
    monkeypatch.setattr(sessions, "_placement_notes", lambda role="": [])
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
                            lambda needs_vision=False, reserve=0, role='', cancel=None: FakeClient(fail=RuntimeError("no server")))

        events = drain(sessions.conversation(sessions.ChatRequest(messages=[]),
                                             sessions.Cancellation()))

        assert texts(events, sessions.FAILED) == ["no server"]

    def test_the_sampling_settings_reach_the_client(self, client):
        request = sessions.ChatRequest(messages=[], temperature=0.4, top_p=0.8,
                                       max_tokens=128, seed=99)

        drain(sessions.conversation(request, sessions.Cancellation()))

        assert client.calls[0]["temperature"] == 0.4
        assert client.calls[0]["seed"] == 99


class TestWhatTheConsoleIsTold:
    """Every run narrates itself to the WebUI console, and narrates no content.

    The second half is the one worth a test. A status line saying a reply is
    2,300 characters long is a status line; one quoting it is a transcript of a
    private conversation in a file somebody else may read, and the difference
    between the two is one careless format string.
    """

    def test_a_run_says_that_it_started_and_that_it_finished(self, client, caplog):
        with caplog.at_level("INFO", logger="model_chain"):
            drain(sessions.conversation(sessions.ChatRequest(messages=[]),
                                        sessions.Cancellation()))

        said = "\n".join(record.getMessage() for record in caplog.records)
        assert "LLM run started" in said
        assert "LLM run finished" in said
        assert "characters in" in said

    def test_nothing_that_was_said_reaches_the_console(self, client, caplog):
        """The prompt, the reply and the character are content. Sizes and
        stages are not."""
        secret = "the-private-thing-somebody-typed"
        client.pieces = [secret]
        request = sessions.ChatRequest(
            messages=[{"role": "user", "content": secret}])

        with caplog.at_level("DEBUG", logger="model_chain"):
            drain(sessions.conversation(request, sessions.Cancellation()))

        for record in caplog.records:
            assert secret not in record.getMessage()

    def test_a_failure_is_logged_with_the_sentence_that_explains_it(self, client, monkeypatch,
                                                                    caplog):
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False, reserve=0, role='', cancel=None: FakeClient(fail=RuntimeError("no server")))

        with caplog.at_level("INFO", logger="model_chain"):
            drain(sessions.conversation(sessions.ChatRequest(messages=[]),
                                        sessions.Cancellation()))

        assert any("no server" in record.getMessage() for record in caplog.records)

    def test_a_cancelled_run_says_so_rather_than_going_quiet(self, client, caplog):
        cancel = sessions.Cancellation()
        cancel.cancel()

        with caplog.at_level("INFO", logger="model_chain"):
            drain(sessions.conversation(sessions.ChatRequest(messages=[]), cancel))

        assert any("cancelled" in record.getMessage() for record in caplog.records)

    def test_tracing_does_not_change_what_the_panel_receives(self, client):
        """The wrapper is a passthrough. Every panel reads event.kind."""
        request = sessions.ChatRequest(messages=[])

        events = drain(sessions.conversation(request, sessions.Cancellation()))

        assert texts(events, sessions.CHUNK) == ["Hello", " there"]
        assert texts(events, sessions.DONE) == ["Hello there"]

    def test_an_abandoned_run_is_logged_and_still_releases_the_gpu(self, client, caplog):
        """Gradio cancels by closing the generator, and the wrapper must let
        that reach the inner one -- the GPU is given back in its finally."""
        with caplog.at_level("INFO", logger="model_chain"):
            run = sessions.conversation(sessions.ChatRequest(messages=[]),
                                        sessions.Cancellation())
            next(run)
            run.close()

        assert mc_broker.active() is None
        assert any("abandoned" in record.getMessage() for record in caplog.records)

    def test_a_run_that_finished_is_not_also_called_abandoned(self, client, caplog):
        """Gradio closes a handler's generator once it has consumed the last
        event, so every completed reply raised GeneratorExit here a moment
        after saying it was complete -- and a console reporting both read as
        though every reply in the conversation had gone wrong at the end."""
        with caplog.at_level("INFO", logger="model_chain"):
            run = sessions.conversation(sessions.ChatRequest(messages=[]),
                                        sessions.Cancellation())
            drain(run)
            run.close()

        messages = [record.getMessage() for record in caplog.records]
        assert any("run finished" in message for message in messages)
        assert not any("abandoned" in message for message in messages)


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
                            lambda needs_vision=False, reserve=0, role='', cancel=None: FakeClient(pieces=("",)))

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
                            lambda needs_vision=False, reserve=0, role='', cancel=None: FakeClient(fail=RuntimeError("boom")))

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


# --------------------------------------------------------------------------- #
# The activity indicator
# --------------------------------------------------------------------------- #
#
# What is being guarded here is not that a bar is drawn -- style.css draws it,
# and a stylesheet is not a thing a test can watch move. It is *which lines get
# it*, because that is the part that can be wrong: a status line still sweeping
# under a run that finished says the opposite of the sentence printed on it,
# and the whole point of the indicator is to be believed.


class TestTheActivityIndicator:
    def test_a_busy_line_is_a_notice_with_one_more_class(self, store):
        """So a theme that styles notices still styles this, and the bar is the
        only difference between the two."""
        import mc_llm_ui as ui

        assert 'class="mc-llm-notice mc-llm-notice-info mc-llm-busy"' in ui.working("Replying…")
        assert "mc-llm-busy-bar" in ui.working("Replying…")
        assert "mc-llm-busy" not in ui.notice("Reply complete.")

    def test_it_escapes_what_it_is_given_like_every_other_status(self, store):
        import mc_llm_ui as ui

        assert "&lt;b&gt;" in ui.working("<b>")

    def test_a_run_in_flight_says_so_and_a_finished_one_stops_saying_it(self, client, store):
        """Read off one real run: every line up to the last carries the bar,
        and the line that says the prompt is complete does not."""
        import mc_llm_minimax_panel as minimax

        drawn = [frame[3] for frame in minimax._enhance("a car", "video", None, 1234)
                 if isinstance(frame[3], str)]

        assert drawn, "the run drew no status at all"
        assert all("mc-llm-busy" in line for line in drawn[:-1])
        assert "mc-llm-busy" not in drawn[-1]

    def test_a_warm_run_does_not_claim_to_be_starting_a_server(self, client, store,
                                                               monkeypatch):
        """It used to say "Starting llama-server…" before every message, which
        was true when every message restarted it. A warm turn starts nothing."""
        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime.runtime, "running", lambda: True)
        warm = texts(drain(sessions.conversation(sessions.ChatRequest(messages=[]),
                                                 sessions.Cancellation())), sessions.STATUS)

        monkeypatch.setattr(mc_llm_runtime.runtime, "running", lambda: False)
        cold = texts(drain(sessions.conversation(sessions.ChatRequest(messages=[]),
                                                 sessions.Cancellation())), sessions.STATUS)

        assert "Starting llama-server…" not in warm
        assert "Starting llama-server…" in cold

    def test_a_stopped_run_stops_saying_it_is_working(self, store):
        """``cancels=`` closes the generator where it stands, so by the time
        this handler runs there is nothing still in flight to sweep for."""
        import mc_llm_chat_panel as chat

        assert "mc-llm-busy" not in chat._cancel(sessions.Cancellation())[0]

    def test_the_chip_pulses_while_the_model_is_loading(self, host, store):
        """The top bar has room for a dot and not for a bar, and Load is the
        one press long enough that a chip which looks inert gets pressed
        twice."""
        import mc_llm_studio

        assert "mc-llm-state-busy" in mc_llm_studio._runtime_line(mc_llm_studio.LOADING)
        assert "mc-llm-state-busy" not in mc_llm_studio._runtime_line()


# --------------------------------------------------------------------------- #
# Replacing the runtime under a running server
# --------------------------------------------------------------------------- #
#
# Windows will not let a file be renamed or deleted while a process holds it
# open, and llama-server holds every DLL beside it. Adopting a build replaces
# that whole folder, so doing it while the server was still up failed with
# "[WinError 5] Access is denied: ...\\runtime\\cublas64_12.dll" -- which names
# a CUDA library and reads like a driver problem rather than like the file lock
# it is. The stop was always there; it was after the copy.


class TestSwappingTheRuntime:
    def test_the_server_is_stopped_before_the_build_is_replaced(self, host, store, monkeypatch):
        import mc_llm_files
        import mc_llm_runtime
        import mc_llm_setup
        import mc_llm_studio

        elsewhere = store.parent / "elsewhere" / "llama-server"
        order = []
        monkeypatch.setattr(mc_llm_runtime.runtime, "stop", lambda: order.append("stop"))
        monkeypatch.setattr(mc_llm_files, "resolve_runtime",
                            lambda path: types.SimpleNamespace(path=elsewhere))
        monkeypatch.setattr(mc_llm_setup, "adopt",
                            lambda source: (order.append("copy"), (source, "Copied."))[1])
        monkeypatch.setattr(mc_llm_setup, "record",
                            lambda executable, device=None, role="": order.append("record"))

        mc_llm_studio._apply_runtime(str(elsewhere), "gpu:0")

        assert order == ["stop", "copy", "record"]

    def test_a_build_already_in_place_is_not_unloaded_to_record_a_device(
            self, host, store, monkeypatch):
        """Changing the device replaces no files, and a stop here costs a
        reload of the whole model to record one line of state."""
        import mc_llm_files
        import mc_llm_runtime
        import mc_llm_setup
        import mc_llm_studio

        order = []
        monkeypatch.setattr(mc_llm_runtime.runtime, "stop", lambda: order.append("stop"))
        monkeypatch.setattr(mc_llm_files, "resolve_runtime",
                            lambda path: types.SimpleNamespace(path=store / "runtime"
                                                               / "llama-server"))
        monkeypatch.setattr(mc_llm_setup, "adopt",
                            lambda source: (source, "Using the build already in place."))
        monkeypatch.setattr(mc_llm_setup, "record",
                            lambda executable, device=None, role="": order.append("record"))

        mc_llm_studio._apply_runtime(str(store / "runtime" / "llama-server"), "mixed:0")

        assert order == ["record"]

    def test_an_empty_box_records_the_device_against_the_runtime_in_place(
            self, host, store, monkeypatch):
        """The only way to change the device is this button, so an empty box
        with a runtime already recorded means "this one" rather than "you
        forgot something"."""
        import mc_llm_setup
        import mc_llm_studio

        in_place = store / "runtime" / "llama-server"
        in_place.parent.mkdir(parents=True)
        in_place.write_bytes(b"")
        recorded = []
        monkeypatch.setattr(mc_llm_setup, "record",
                            lambda executable, device=None, role="": recorded.append((executable, device)))

        notice, path, _model, _role = mc_llm_studio._apply_runtime("", "mixed:0")

        assert recorded and recorded[0][0] == in_place
        assert "already in place" in notice

    def test_the_download_stops_it_too(self, host, store, monkeypatch):
        """The pinned build is extracted over the same folder."""
        import mc_llm_runtime
        import mc_llm_setup
        import mc_llm_studio

        order = []
        monkeypatch.setattr(mc_llm_runtime.runtime, "stop", lambda: order.append("stop"))
        monkeypatch.setattr(mc_llm_setup, "download",
                            lambda device, on_status=None, on_progress=None:
                            (order.append("extract"), store / "llama-server")[1])
        monkeypatch.setattr(mc_llm_setup, "record",
                            lambda executable, device=None, role="": order.append("record"))

        mc_llm_studio._download_runtime("0")

        assert order == ["stop", "extract", "record"]

    def test_a_folder_still_in_use_is_reported_as_one(self, store):
        """The sentence a user can act on, in place of a Windows error code
        pointing at a CUDA library."""
        import mc_llm_setup

        message = str(mc_llm_setup.in_use_error(store / "runtime",
                                                PermissionError("[WinError 5] Access is denied")))

        assert "in use" in message and "Unload" in message


# --------------------------------------------------------------------------- #
# The reply that was on screen and not on disk
# --------------------------------------------------------------------------- #


class TestAReplySurvivesWhateverEndsTheGenerator:
    """Reported: the first replies of every old thread came back blank.

    Stop is wired as ``cancels=``, and what Gradio does with that is *close*
    the handler's generator. That raises ``GeneratorExit`` -- a
    ``BaseException`` -- straight past the ``except Exception`` that held the
    only ``store.save`` on that path. So the reply stayed on screen, because
    Gradio keeps the rows it was last given, and never reached the thread: the
    next time it was opened there was a message of yours with nothing under it.
    A browser refresh and a dropped queue entry went the same way.
    """

    def _thread(self, store, monkeypatch, pieces=("Hello", " there")):
        import mc_llm_chat_panel as chat
        from prompt_master.chat.characters import Persona, save_persona
        from prompt_master.chat.history import ChatStore

        save_persona(mc_llm_paths.app_paths(), Persona(name="Me", description="a reader"))
        chats = ChatStore(store / "chats")
        monkeypatch.setattr(chat, "_chats", lambda: chats)

        from prompt_master.chat.characters import Character

        class Characters:
            def load(self, who):
                return Character(name="Ada", context="a reader of maps")

        monkeypatch.setattr(chat, "_characters", lambda: Characters())
        # Chunks and then DONE, which is the shape the real generator has: the
        # completed case has to go through the branch that actually runs in
        # production, or the test proves nothing about it.
        events = [sessions.Event(sessions.CHUNK, piece) for piece in pieces]
        events.append(sessions.Event(sessions.DONE, "".join(pieces)))
        monkeypatch.setattr(chat.sessions, "conversation",
                            lambda request, cancel: iter(events))
        conversation = chats.new("Ada")
        chats.save(conversation)
        return chat, chats, conversation

    def test_a_closed_generator_still_writes_the_reply_to_the_thread(self, store, host,
                                                                     monkeypatch):
        chat, chats, conversation = self._thread(store, monkeypatch)

        run = chat._send("Ada", conversation.identifier, "hello love", None, None, None, None, None)
        next(run)          # the composer clears and the run starts
        next(run)          # the first chunk arrives
        run.close()        # what Stop does

        saved = chats.load("Ada", conversation.identifier)
        assert [message.role for message in saved.messages] == ["user", "assistant"]
        assert saved.messages[0].text == "hello love"
        assert saved.messages[1].text == "Hello", "the partial reply is a real reply"

    def test_a_reply_that_ran_to_completion_is_saved_once_and_kept(self, store, host,
                                                                   monkeypatch):
        chat, chats, conversation = self._thread(store, monkeypatch)

        list(chat._send("Ada", conversation.identifier, "hello love", None, None, None, None, None))

        saved = chats.load("Ada", conversation.identifier)
        assert saved.messages[1].text == "Hello there"

    def test_a_reply_that_never_started_leaves_no_blank_message_behind(self, store, host,
                                                                       monkeypatch):
        """``_tidy``'s job, and the reason the save cannot simply be
        unconditional: a reply that produced nothing is cleared up rather than
        written to the thread as an empty bubble."""
        chat, chats, conversation = self._thread(store, monkeypatch, pieces=())

        run = chat._send("Ada", conversation.identifier, "hello love", None, None, None, None, None)
        next(run)
        run.close()

        saved = chats.load("Ada", conversation.identifier)
        assert [message.role for message in saved.messages] == ["user"]

    def test_the_transcript_shows_a_message_with_no_reply_under_it(self, store, host):
        """What the bug looked like, kept as the description of the shape: a
        user message whose reply never reached the file is a row with an empty
        right-hand side, which is the blank bubble in the screenshot."""
        import mc_llm_chat_panel as chat
        from prompt_master.chat.history import Conversation, Message

        conversation = Conversation(identifier="x", character="Ada",
                                    messages=[Message(role="user", versions=["hello love"])])
        rows, _ = chat._view(conversation)

        assert rows == [["hello love", None]]


# --------------------------------------------------------------------------- #
# Servers that outlived the WebUI (the VRAM that would not come back)
# --------------------------------------------------------------------------- #


class FakeProcess:
    def __init__(self, pid, name="llama-server.exe", cmdline=None, denied=False):
        self.info = {"pid": pid, "name": name,
                     "cmdline": cmdline if cmdline is not None else
                     ["llama-server", "--model", "m.gguf", "--alias", "prompt-master"]}
        self.denied = denied
        self.stopped = False
        self.killed = False
        if denied:
            self.info["cmdline"] = None

    def terminate(self):
        self.stopped = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def fake_psutil(monkeypatch, processes):
    import types

    module = types.ModuleType("psutil")
    module.process_iter = lambda attrs=None: list(processes)
    module.Process = lambda pid: next(p for p in processes if p.info["pid"] == pid)
    monkeypatch.setitem(__import__("sys").modules, "psutil", module)
    return module


class TestStrayServers:
    """A llama-server is started in its own process group and with no window,
    so it survives the WebUI that started it and is invisible outside Task
    Manager -- while still holding a CUDA context and a model's worth of
    weights. The tab then reads "Unloaded", truthfully about the runtime it
    knows about, over a card with no free VRAM and an Unload button with
    nothing to stop.
    """

    def test_the_alias_here_is_the_one_the_vendored_starter_actually_passes(self):
        """The two are in different files and only one of them is ours to edit,
        so the agreement is asserted rather than assumed."""
        import mc_llm_runtime
        from pathlib import Path

        import prompt_master.inference.llama_process as starter

        source = Path(starter.__file__).read_text(encoding="utf-8")
        assert f'"--alias","{mc_llm_runtime.SERVER_ALIAS}"' in source.replace(", ", ",")

    def test_a_server_carrying_our_alias_is_a_stray(self, monkeypatch):
        import mc_llm_runtime

        fake_psutil(monkeypatch, [FakeProcess(4321)])

        assert mc_llm_runtime.strays() == [4321]

    def test_somebody_else_s_llama_server_is_left_alone(self, monkeypatch):
        import mc_llm_runtime

        fake_psutil(monkeypatch, [
            FakeProcess(1, cmdline=["llama-server", "--alias", "someone-else"]),
            FakeProcess(2, name="python.exe"),
        ])

        assert mc_llm_runtime.strays() == []

    def test_a_path_that_merely_contains_the_alias_is_not_a_match(self, monkeypatch):
        """The alias has to be the value of --alias, not a word on the line: a
        model kept under a folder of that name would otherwise make every
        server on the machine look like ours."""
        import mc_llm_runtime

        fake_psutil(monkeypatch, [FakeProcess(
            7, cmdline=["llama-server", "--model", "C:/prompt-master/m.gguf"])])

        assert mc_llm_runtime.strays() == []

    def test_the_server_this_webui_owns_is_never_a_stray(self, monkeypatch):
        import mc_llm_runtime

        fake_psutil(monkeypatch, [FakeProcess(4321), FakeProcess(9999)])
        owned = types.SimpleNamespace(process=types.SimpleNamespace(pid=4321))
        monkeypatch.setattr(mc_llm_runtime.runtime, "_process", owned, raising=False)

        assert mc_llm_runtime.strays() == [9999]

    def test_a_process_list_that_cannot_be_read_reports_nothing_rather_than_guessing(
            self, monkeypatch):
        import mc_llm_runtime

        fake_psutil(monkeypatch, [FakeProcess(1, denied=True)])

        assert mc_llm_runtime.strays() == []

    def test_releasing_stops_them_and_says_how_many(self, monkeypatch):
        import mc_llm_runtime

        processes = [FakeProcess(4321), FakeProcess(9999)]
        fake_psutil(monkeypatch, processes)
        monkeypatch.setattr(mc_llm_runtime.runtime, "_process", None, raising=False)

        stopped, _freed = mc_llm_runtime.release_strays()

        assert stopped == 2
        assert all(process.stopped for process in processes)

    def test_releasing_nothing_is_not_an_error(self, monkeypatch):
        import mc_llm_runtime

        fake_psutil(monkeypatch, [])

        assert mc_llm_runtime.release_strays() == (0, 0)

    def test_unload_stops_a_stray_and_says_so(self, store, host, monkeypatch):
        """"Unload" is a request for the card back. Until now it could only
        reach the one server this WebUI had a handle to."""
        import mc_llm_studio

        monkeypatch.setattr(mc_llm_studio, "_release_strays", lambda: (1, 15 * 1024**3))
        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime.runtime, "running", lambda: False)

        line, _residency = mc_llm_studio._unload_model()

        assert "stray" in line and "15.0 GB" in line

    def test_a_failure_looking_for_strays_never_costs_unload_its_answer(self, store, host,
                                                                        monkeypatch):
        import mc_llm_runtime
        import mc_llm_studio

        monkeypatch.setattr(mc_llm_runtime, "release_strays",
                            lambda: (_ for _ in ()).throw(RuntimeError("no process list")))
        monkeypatch.setattr(mc_llm_runtime.runtime, "running", lambda: False)

        line, _residency = mc_llm_studio._unload_model()

        # Whatever the runtime's own state is, the press is answered with it
        # rather than with a traceback or an empty panel.
        assert "mc-llm-state" in line


class TestAServerIsNeverLostOnTheWayToSystemRam:
    def test_a_health_check_that_fails_stops_the_process_it_started(self, store, host,
                                                                    monkeypatch):
        """``start`` can succeed and ``wait_ready`` still fail. What was left
        behind then was a live llama-server nothing had a handle to -- in its
        own process group, with no window, holding the card, while the panel
        said "Unloaded" about the runtime it knew about."""
        import mc_llm_runtime

        started = []

        class Process:
            def __init__(self):
                self.stopped = False

            def start(self, *args, **kwargs):
                started.append(self)

            def wait_ready(self, timeout=None):
                raise TimeoutError("llama-server did not become ready")

            def stop(self):
                self.stopped = True

        monkeypatch.setattr(mc_llm_runtime.runtime, "_new_process", lambda: Process())
        monkeypatch.setattr(mc_llm_runtime.runtime, "_stop_locked", lambda reason: None)
        monkeypatch.setattr(mc_llm_runtime, "config", lambda: types.SimpleNamespace(
            configured=True, runtime="llama-server", model="m.gguf", mmproj=None,
            gpu_index=0, device="CUDA0"))

        freed = mc_llm_runtime.runtime._restart_in_system_ram(0, "an image pass")

        assert freed == 0
        assert started and started[0].stopped, "the process it started has to be stopped"


class TestPreferencesSurviveAWindowsFileLock:
    """``os.replace`` is atomic on POSIX and merely usually-atomic on Windows.

    A scanner, a backup agent or a search indexer holding the destination open
    for a moment answers ``[WinError 5] Access is denied``, and the write was
    given up on the first refusal -- so a user reported a traceback in their
    console and a preference that did not stick. Every one of those holders
    lets go within milliseconds, so the fix is to ask again.
    """

    def test_a_replace_that_is_refused_once_is_retried(self, store, monkeypatch):
        attempts = []
        real = os.replace

        def refuse_once(source, target):
            attempts.append(target)
            if len(attempts) == 1:
                raise PermissionError(5, "Access is denied")
            return real(source, target)

        monkeypatch.setattr(state.os, "replace", refuse_once)
        monkeypatch.setattr(state.time, "sleep", lambda seconds: None)

        state._write("preferences.json", {"kept": True})

        assert len(attempts) == 2
        assert state._read("preferences.json", {}) == {"kept": True}

    def test_a_lock_that_never_lets_go_is_reported_rather_than_raised(self, store,
                                                                     monkeypatch):
        def always_refuse(source, target):
            raise PermissionError(5, "Access is denied")

        monkeypatch.setattr(state.os, "replace", always_refuse)
        monkeypatch.setattr(state.time, "sleep", lambda seconds: None)

        state._write("preferences.json", {"kept": True})

        assert state._read("preferences.json", None) is None

    def test_a_write_that_could_not_land_leaves_no_temporary_file_behind(self, store,
                                                                        monkeypatch):
        """The directory is one the user opens to read their own settings, and
        the log showed a ``preferences.jsonfyz9tar1.tmp`` left in it."""
        monkeypatch.setattr(state.os, "replace",
                            lambda source, target: (_ for _ in ()).throw(
                                PermissionError(5, "Access is denied")))
        monkeypatch.setattr(state.time, "sleep", lambda seconds: None)

        state._write("preferences.json", {"kept": True})

        assert list(store.glob("*.tmp")) == []
