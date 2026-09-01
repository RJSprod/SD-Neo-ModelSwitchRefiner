"""The GPU lease, and the one rule that keeps it from outliving a run.

Written for a bug that made a whole installation unusable in a way nothing
recovered from: a reply stopped or failed *while it was streaming* never gave
the card back, and every later run queued behind it forever -- reporting
"Waiting for a conversation reply…" about a conversation reply that had
finished minutes earlier.
"""

from __future__ import annotations

import ast
import gc
import pathlib

import pytest

import mc_broker
import mc_llm_sessions as sessions


TERMINAL = {"CANCELLED", "DONE", "FAILED"}


class Stoppable:
    """A Cancellation double. The real one carries a threading.Event."""

    def __init__(self, stopped=False):
        self._stopped = stopped

    def set(self):
        self._stopped = True

    def is_set(self):
        return self._stopped

    @property
    def event(self):
        return self


@pytest.fixture(autouse=True)
def _no_lease_left_behind():
    yield
    gc.collect()
    assert mc_broker.active() is None, "a test left the GPU booked"


class TestTheCardComesBackBeforeTheLastWord:
    """The invariant, stated where it can be checked rather than hoped for."""

    def _terminal_yields(self, tree, name):
        """Every terminal ``yield Event(...)`` in one function, with its parent."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                found = []
                for inner in ast.walk(node):
                    for field in ("body", "orelse", "finalbody"):
                        block = getattr(inner, field, None)
                        if not isinstance(block, list):
                            continue
                        for position, statement in enumerate(block):
                            kind = self._terminal_kind(statement)
                            if kind:
                                found.append((position, block, kind))
                return found
        raise AssertionError(f"{name} is not in this module any more")

    @staticmethod
    def _terminal_kind(statement):
        if not isinstance(statement, ast.Expr):
            return ""
        value = statement.value
        if not isinstance(value, ast.Yield) or not isinstance(value.value, ast.Call):
            return ""
        call = value.value
        if not (isinstance(call.func, ast.Name) and call.func.id == "Event"):
            return ""
        first = call.args[0] if call.args else None
        if isinstance(first, ast.Name) and first.id in TERMINAL:
            return first.id
        return ""

    @staticmethod
    def _is_release(statement) -> bool:
        return (isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "release"
                and isinstance(statement.value.func.value, ast.Name)
                and statement.value.func.value.id == "gpu")

    @pytest.mark.parametrize("name", ["_prompt_studio", "_conversation", "_minimax",
                                      "_krea", "_compose"])
    def test_every_terminal_event_is_preceded_by_a_release(self, name):
        """The whole fix, as a property rather than as five careful edits.

        A ``finally`` cannot be relied on here: Gradio's ``cancels=`` leaves an
        abandoned handler generator un-finalised, so the card would stay booked
        to a run that is over. Releasing before the event goes out makes
        finalisation a tidy-up rather than the mechanism.
        """
        source = pathlib.Path(sessions.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        found = self._terminal_yields(tree, name)
        assert found, f"{name} yields no terminal event, so this proves nothing"
        for position, block, kind in found:
            assert position > 0 and self._is_release(block[position - 1]), (
                f"{name} yields {kind} without giving the GPU back first — a run "
                f"stopped there would keep the card until something finalises "
                f"the generator, and nothing reliably does")

    def test_the_check_can_fail(self):
        """Guard: the matcher above must not pass on code that lacks the release."""
        tree = ast.parse(
            "def _conversation():\n"
            "    try:\n"
            "        yield Event(DONE, text)\n"
            "    finally:\n"
            "        gpu.release()\n")
        found = self._terminal_yields(tree, "_conversation")
        assert found, "the matcher found no terminal yield at all"
        position, block, _kind = found[0]
        assert not (position > 0 and self._is_release(block[position - 1]))


class TestAStoppedReplyGivesTheCardBack:
    """The behaviour the property above protects, driven end to end."""

    def _reply(self, stopped=False, fail=False):
        """A reply that streams, then stops, fails, or finishes."""
        cancel = Stoppable()
        gpu = sessions._Gpu("a conversation reply", cancel)
        seen = []
        try:
            acquired = yield from gpu.acquire()
            assert acquired
            seen.append(mc_broker.active())
            yield sessions.Event(sessions.CHUNK, "partial")
            if fail:
                raise RuntimeError("[WinError 10054] connection forcibly closed")
            if stopped:
                cancel.set()
                gpu.release()
                yield sessions.Event(sessions.CANCELLED, "partial")
                return
            gpu.release()
            yield sessions.Event(sessions.DONE, "whole")
        except Exception as exc:
            gpu.release()
            yield sessions.Event(sessions.FAILED, str(exc))
        finally:
            gpu.release()

    @pytest.mark.parametrize("stopped,fail,kind",
                             [(True, False, sessions.CANCELLED),
                              (False, True, sessions.FAILED),
                              (False, False, sessions.DONE)])
    def test_the_card_is_free_the_moment_the_panel_hears_the_news(self, stopped, fail,
                                                                  kind):
        """The panel returns out of its loop here. Nothing finalises anything.

        ``gc`` is off for the assertion, because a fix that only works once the
        collector happens to run is the bug this replaces.
        """
        gc.disable()
        try:
            events = self._reply(stopped=stopped, fail=fail)
            held = None
            for event in events:
                if event.kind == sessions.CHUNK:
                    held = mc_broker.active()
                    continue
                assert event.kind == kind
                assert mc_broker.active() is None, (
                    "the card was still booked when the terminal event arrived")
                break
            assert held is not None, "the run never held the card, so this proves nothing"
            del events
        finally:
            gc.enable()
            gc.collect()

    def test_a_run_stopped_before_it_starts_holds_nothing(self):
        """Stopping while queued was never the broken case, and must stay sound."""
        cancel = Stoppable(stopped=True)
        gpu = sessions._Gpu("a conversation reply", cancel)
        assert list(gpu.acquire()) == []
        assert mc_broker.active() is None


class TestTheReadinessLineDoesNotContradictItself:
    """"All layers on the GPU ... 0.0 GB VRAM" is two halves of one sentence.

    ``_report_offload`` normally catches a placement that did not happen, by
    printing what llama.cpp said it did. It cannot catch this one: a build with
    no backend in it writes no load report to disagree with, so the planned
    placement was the only thing on screen.
    """

    def test_dropping_the_card_is_recorded_for_the_line_that_reports_it(self,
                                                                        monkeypatch):
        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime, "_runtime_enumerates_a_device",
                            lambda executable: False)
        mc_llm_runtime.without_gpu_selection(
            ["llama-server", "--device", "CUDA0", "--main-gpu", "0"])

        assert mc_llm_runtime._dropped_the_card is True

    def test_an_ordinary_gpu_start_records_nothing(self, monkeypatch):
        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime, "_runtime_enumerates_a_device",
                            lambda executable: True)
        mc_llm_runtime._dropped_the_card = True
        mc_llm_runtime.without_gpu_selection(
            ["llama-server", "--device", "CUDA0", "--main-gpu", "0"])

        assert mc_llm_runtime._dropped_the_card is False, (
            "one corrected start would have annotated every later one")

    def test_a_cpu_start_records_nothing(self):
        import mc_llm_runtime

        mc_llm_runtime._dropped_the_card = True
        mc_llm_runtime.without_gpu_selection(
            ["llama-server", "--device", "none", "--main-gpu", "0"])

        assert mc_llm_runtime._dropped_the_card is False
