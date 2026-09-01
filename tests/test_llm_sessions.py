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


class TestTheCardIsPinnedByNameNotBySlot:
    """The two enumerations disagree, and the launcher was trusting the number.

    The vendored launcher pins the server to one card and then always passes
    ``--main-gpu 0``, which is the right shape: with a single visible card,
    llama.cpp's ``CUDA0`` is unambiguously that card. What was wrong is the
    value. ``gpu_index`` is nvidia-smi's number, ``CUDA_VISIBLE_DEVICES`` is
    read in CUDA's order, this extension never sets ``CUDA_DEVICE_ORDER``, and
    ``mc_memory.image_device_index`` documents the two disagreeing on real
    hardware -- "card 0" naming a 5090 at one end and a 3090 at the other.

    Getting it wrong is silent: the server starts and answers, on the other
    card, while every VRAM decision is measured against the one it is not on.
    """

    UUID = "GPU-0f4c1c26-8c1d-4f4a-9a2b-77c3a1d5e9b0"

    def _start(self, device="CUDA0"):
        return ["llama-server", "--model", "m.gguf", "--device", device,
                "--split-mode", "none", "--main-gpu", "0", "--ctx-size", "8192"]

    def test_a_card_start_is_pinned_to_the_uuid(self):
        import mc_llm_runtime

        mc_llm_runtime._arm_visibility(self.UUID)
        found = mc_llm_runtime.with_pinned_card(self._start(),
                                                {"CUDA_VISIBLE_DEVICES": "0"})

        assert found["CUDA_VISIBLE_DEVICES"] == self.UUID

    def test_a_processor_start_keeps_its_empty_string(self):
        """CPU placement sets it empty on purpose, and must stay that way."""
        import mc_llm_runtime

        mc_llm_runtime._arm_visibility(self.UUID)
        found = mc_llm_runtime.with_pinned_card(self._start(device="none"),
                                                {"CUDA_VISIBLE_DEVICES": ""})

        assert found["CUDA_VISIBLE_DEVICES"] == ""

    def test_a_start_whose_card_was_dropped_is_not_pinned(self):
        """``without_gpu_selection`` has already decided this runs on the CPU."""
        import mc_llm_runtime

        mc_llm_runtime._arm_visibility(self.UUID)
        dropped = ["llama-server", "--model", "m.gguf", "--ctx-size", "8192"]
        found = mc_llm_runtime.with_pinned_card(dropped, {"CUDA_VISIBLE_DEVICES": "0"})

        assert found["CUDA_VISIBLE_DEVICES"] == "0"

    def test_a_state_file_with_no_uuid_changes_nothing(self):
        """Older installations recorded an index alone. Left exactly as it was."""
        import mc_llm_runtime

        mc_llm_runtime._arm_visibility("")
        found = mc_llm_runtime.with_pinned_card(self._start(),
                                                {"CUDA_VISIBLE_DEVICES": "1"})

        assert found["CUDA_VISIBLE_DEVICES"] == "1"

    def test_something_that_is_not_a_server_start_is_untouched(self):
        """A device probe spawned while a start is in flight passes through."""
        import mc_llm_runtime

        mc_llm_runtime._arm_visibility(self.UUID)
        found = mc_llm_runtime.with_pinned_card(["llama-server", "--list-devices"],
                                                {"CUDA_VISIBLE_DEVICES": "0"})

        assert found["CUDA_VISIBLE_DEVICES"] == "0"

    def test_the_pin_is_spent_once(self):
        """A pin left armed would follow the next start onto a different card."""
        import mc_llm_runtime

        mc_llm_runtime._arm_visibility(self.UUID)
        first = mc_llm_runtime.with_pinned_card(self._start(),
                                                {"CUDA_VISIBLE_DEVICES": "0"})
        second = mc_llm_runtime.with_pinned_card(self._start(),
                                                 {"CUDA_VISIBLE_DEVICES": "0"})

        assert first["CUDA_VISIBLE_DEVICES"] == self.UUID
        assert second["CUDA_VISIBLE_DEVICES"] == "0", "the pin outlived its start"

    def test_a_value_that_is_not_a_uuid_is_refused_before_it_is_armed(self):
        """Guard: only nvidia-smi's own spelling reaches the environment."""
        import mc_llm_runtime

        for rubbish in ("0", "GPU-1", "; rm -rf /", "GPU-zzzzzzzz-1111-2222-3333-444444444444"):
            mc_llm_runtime._arm_visibility(rubbish)
            found = mc_llm_runtime.with_pinned_card(self._start(),
                                                    {"CUDA_VISIBLE_DEVICES": "0"})
            assert found["CUDA_VISIBLE_DEVICES"] == "0", rubbish

    def test_a_start_that_carries_no_environment_still_spends_the_pin(self):
        """The consumption cannot depend on there being something to change.

        A pin left armed by a start that passed no environment of its own would
        attach itself to the next server, which is the one failure this whole
        mechanism exists to prevent.
        """
        import mc_llm_runtime

        mc_llm_runtime._arm_visibility(self.UUID)
        assert mc_llm_runtime.with_pinned_card(self._start(), None) is None

        after = mc_llm_runtime.with_pinned_card(self._start(),
                                                {"CUDA_VISIBLE_DEVICES": "0"})
        assert after["CUDA_VISIBLE_DEVICES"] == "0", "the pin outlived its start"

