"""The Krea roll, as reported on Forge's progress bar.

The complaint this answers is not subtle: a Creative Mode roll takes twenty
seconds on a mixed-placement model, all of it before the image job starts, and
none of it used to appear anywhere. So the assertions below are mostly about
*visibility* -- that a task is claimed, that the label changes as the phases do,
that the counters move while tokens stream, and above all that the bar is given
back afterwards, on every path out including the ones nobody plans for.

The last one is the reason this file exists rather than being three lines in
another. A claimed task that is never released leaves the host believing a job
is running: the next Generate draws a bar that never moves, and the only cure a
user can find is restarting the WebUI. Every failure mode below -- a refused
checkpoint, an empty prompt, a library that will not load, a model that throws
mid-stream, an Interrupt -- is checked for the release rather than only for its
own error.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import mc_broker
import mc_creative_krea
import mc_llm_paths
import mc_llm_progress
import mc_llm_sessions as sessions
import mc_progress


class FakeClient:
    """A writer that streams in pieces, so the counters have something to count."""

    def __init__(self, pieces=("A tall ", "white ", "lighthouse."), fail=None):
        self.pieces = list(pieces)
        self.fail = fail
        self.calls: list[dict] = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.85, top_p=0.95):
        self.calls.append({"messages": messages, "seed": seed})
        if self.fail is not None:
            raise self.fail
        produced = []
        for piece in self.pieces:
            if cancel is not None and cancel.is_set():
                break
            produced.append(piece)
            on_text(piece)
        return "".join(produced)


def _position():
    """The two counters a sampler drives, as the host would read them."""
    from modules import shared

    return (shared.state.sampling_step, shared.state.sampling_steps)


def watch_counters():
    """Sample the counters after every chunk the reporter is told about.

    Observed on the reporter rather than inside the fake client, because the
    client streams on a worker thread and the reporter is driven on the
    consumer's -- sampling from inside ``on_text`` would read the counters a
    moment before the chunk that moves them has arrived.
    """
    positions: list = []
    original = mc_llm_progress.reporter.wrote

    def wrote(text):
        original(text)
        positions.append(_position())

    mc_llm_progress.reporter.wrote = wrote
    return positions


def _label():
    from modules import shared

    return shared.state.textinfo


def _current_task():
    from modules import progress

    return progress.current_task


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(mc_progress, "path", lambda: str(tmp_path / mc_progress.FILENAME))
    mc_progress.forget()
    mc_progress.abandon()
    yield tmp_path
    mc_progress.abandon()


@pytest.fixture
def client(monkeypatch, host, store):
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
    fake = FakeClient()
    monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)
    monkeypatch.setattr(sessions, "_placement_notes", list)
    monkeypatch.setattr(mc_creative_krea, "checkpoint_objection", lambda: "")
    monkeypatch.setattr(mc_creative_krea, "_warm", lambda: True)
    mc_creative_krea.creative = mc_creative_krea.Creative()
    mc_llm_progress.reporter = mc_llm_progress.Reporter()
    yield fake
    mc_llm_progress.reporter.abandon()
    mc_creative_krea.creative = mc_creative_krea.Creative()
    mc_broker.clear()


def roll(task_id="task(krea)", source="car", creativity=10):
    stored = mc_creative_krea.settings()
    stored["creativity"] = creativity
    return list(mc_creative_krea.creative.roll(source, stored, task_id=task_id))


# --------------------------------------------------------------------------- #
# The bar appears
# --------------------------------------------------------------------------- #


class TestTheTaskIsClaimed:
    def test_a_roll_claims_the_task_the_browser_minted(self, client):
        seen = []

        original = mc_llm_progress.reporter.enter

        def watch(phase):
            seen.append(_current_task())
            original(phase)

        mc_llm_progress.reporter.enter = watch
        roll("task(abc)")

        assert "task(abc)" in seen

    def test_the_host_reports_it_as_the_running_task_while_it_runs(self, client):
        during = []
        client.pieces = ("one", "two")
        original = mc_llm_progress.reporter.wrote

        def watch(text):
            during.append(_current_task())
            original(text)

        mc_llm_progress.reporter.wrote = watch
        roll("task(abc)")

        assert during and all(task == "task(abc)" for task in during)

    def test_no_task_id_means_no_claim_and_no_complaint(self, client):
        """LLM Studio rolls with no bar arranged. The roll still runs; the
        reporter simply has nothing to report on."""
        events = roll(task_id="")

        assert events[-1].kind == sessions.DONE
        assert _current_task() is None

    def test_a_host_with_no_progress_module_is_not_fatal(self, client, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "modules.progress" or name.endswith("progress"):
                if name == "modules.progress":
                    raise ImportError("no progress module here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        events = roll("task(abc)")

        assert events[-1].kind == sessions.DONE


# --------------------------------------------------------------------------- #
# The bar says what is happening
# --------------------------------------------------------------------------- #


class TestThePhases:
    def test_it_names_the_three_spans_in_order(self, client):
        seen = []
        original = mc_llm_progress.reporter.enter

        def watch(phase):
            original(phase)
            if mc_llm_progress.reporter.task and phase not in seen:
                seen.append(phase)

        mc_llm_progress.reporter.enter = watch
        roll()

        assert seen == [mc_progress.PHASE_KREA_WAIT,
                        mc_progress.PHASE_KREA_READ,
                        mc_progress.PHASE_KREA_WRITE]

    def test_the_label_the_user_reads_changes_with_them(self, client):
        labels = []
        original = mc_llm_progress.reporter.enter

        def watch(phase):
            original(phase)
            if _label() and _label() not in labels:
                labels.append(_label())

        mc_llm_progress.reporter.enter = watch
        roll()

        assert labels == [mc_llm_progress.WAITING, mc_llm_progress.READING,
                          mc_llm_progress.WRITING]

    def test_reading_begins_when_the_request_goes_out(self, client):
        """The session says "Writing the Krea prompt…" immediately before it
        calls the client, and that is exactly when prompt evaluation starts --
        the phase that reports nothing and takes ten seconds."""
        from prompt_master.krea import enhancer

        assert mc_creative_krea._phase_for(
            f"{enhancer.label(0)}…") == mc_progress.PHASE_KREA_READ
        assert mc_creative_krea._phase_for(
            "Starting llama-server…") == mc_progress.PHASE_KREA_WAIT
        assert mc_creative_krea._phase_for(
            "Waiting for image generation…") == mc_progress.PHASE_KREA_WAIT

    def test_writing_begins_on_the_first_streamed_chunk(self, client):
        """The one boundary that is observable from this side of the wire."""
        positions = watch_counters()
        roll()

        # The counters are live from the first chunk onwards: a real total, and
        # a count that is no longer zero.
        assert positions
        assert all(total > 0 for _done, total in positions)
        assert positions[-1][0] > 0


class TestTheCounters:
    def test_they_climb_as_the_reply_streams(self, client):
        client.pieces = ("a" * 50, "b" * 50, "c" * 50)
        positions = watch_counters()
        roll()

        counted = [done for done, _total in positions]
        assert counted == sorted(counted)
        assert counted[-1] > counted[0]

    def test_the_total_stretches_rather_than_the_bar_pinning_at_full(self, client):
        """A reply that outruns the estimate should slow the bar down, not park
        it at 100% for the rest of the run."""
        client.pieces = tuple("x" * 400 for _ in range(6))
        positions = watch_counters()
        roll()

        assert all(done <= total for done, total in positions)
        assert positions[-1][1] > positions[0][1]

    def test_they_are_cleared_when_the_roll_ends(self, client):
        roll()

        assert _position() == (0, 0)
        assert _label() is None


class TestTheEstimate:
    def test_a_longer_prompt_is_predicted_to_take_longer_to_read(self, client):
        """The whole explanation for Creative Mode's delay, as arithmetic: the
        brief is what the model has to read before it can say anything."""
        short = mc_creative_krea._prompt_size("car", (), "")
        long = mc_creative_krea._prompt_size("car", (), "x" * 3000)

        assert long > short + 2900

    def test_the_reply_length_is_learned_from_what_came_back(self, client):
        client.pieces = ("y" * 900,)
        roll()

        assert mc_progress.measured("krea:reply") == pytest.approx(900.0)

    def test_an_unusually_short_reply_cannot_shrink_the_estimate_to_nothing(
            self, client):
        """Otherwise one terse answer teaches the bar to expect eighty
        characters and every normal roll sits at 99%."""
        mc_progress.learn("krea:reply", 10.0)
        reporter = mc_llm_progress.Reporter()
        reporter.begin("task(x)", 1000, warm=True)
        try:
            reporter.enter(mc_progress.PHASE_KREA_WRITE)
            _done, total = _position()
        finally:
            reporter.abandon()

        assert total >= mc_llm_progress.MINIMUM_REPLY

    def test_a_cold_start_is_predicted_differently_from_a_warm_one(self):
        assert mc_progress.BASELINES["krea:wait:cold"] > (
            mc_progress.BASELINES["krea:wait:warm"] * 10)

    def test_the_prediction_self_corrects_across_rolls(self, client, store):
        client.pieces = ("z" * 800,)
        roll()
        first = mc_progress.measured("krea:read")
        roll()

        assert first > 0
        assert mc_progress.measured("krea:read") > 0


# --------------------------------------------------------------------------- #
# The bar is always given back
# --------------------------------------------------------------------------- #


class TestTheTaskIsAlwaysReleased:
    """A claimed task that is never released leaves the host believing a job is
    running, and the next Generate draws a bar that never moves."""

    def test_after_a_successful_roll(self, client):
        roll()

        assert _current_task() is None
        assert mc_llm_progress.reporter.task is None

    def test_after_the_model_throws(self, client):
        client.fail = RuntimeError("llama-server went away")
        events = roll()

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_after_an_empty_reply(self, client):
        client.pieces = ("",)
        events = roll()

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_after_a_refused_checkpoint(self, client, monkeypatch):
        monkeypatch.setattr(mc_creative_krea, "checkpoint_objection",
                            lambda: "Select a Krea 2 checkpoint.")
        stored = mc_creative_krea.settings()
        events = list(mc_creative_krea.creative.roll(
            "car", stored, guard_checkpoint=True, task_id="task(abc)"))

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_after_an_empty_source_prompt(self, client):
        events = roll(source="   ")

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_after_the_director_fails(self, client, monkeypatch):
        from prompt_master.krea import director

        def explode(*args, **kwargs):
            raise RuntimeError("no package")

        monkeypatch.setattr(director, "roll", explode)
        events = roll()

        assert events[-1].kind == sessions.FAILED
        assert _current_task() is None

    def test_a_failed_roll_teaches_the_store_nothing_about_the_answer(self, client):
        """A run that gave up says nothing about how long an answer takes, so
        neither the writing rate nor the expected reply length may learn from
        it. What did complete -- the wait -- is real and is allowed to."""
        client.fail = RuntimeError("nope")
        roll()

        learned = mc_progress.rates()
        assert "krea:write" not in learned
        assert "krea:reply" not in learned


class TestInterrupt:
    def test_pressing_interrupt_stops_the_roll(self, client):
        """The bar the host draws carries Interrupt whether or not anything is
        listening, and a button that does nothing is worse than no button."""
        from modules import shared

        def press(text):
            shared.state.interrupted = True

        client.watching = None
        original = mc_llm_progress.reporter.wrote
        mc_llm_progress.reporter.wrote = lambda text: (original(text), press(text))[0]
        events = roll()

        assert events[-1].kind == sessions.CANCELLED
        assert _current_task() is None

    def test_the_flag_is_cleared_so_the_image_job_does_not_inherit_it(self, client):
        from modules import shared

        original = mc_llm_progress.reporter.wrote
        mc_llm_progress.reporter.wrote = lambda text: (
            original(text), setattr(shared.state, "interrupted", True))[0]
        roll()

        assert shared.state.interrupted is False
        assert shared.state.skipped is False

    def test_an_interrupted_roll_arms_nothing(self, client):
        from modules import shared

        original = mc_llm_progress.reporter.wrote
        mc_llm_progress.reporter.wrote = lambda text: (
            original(text), setattr(shared.state, "interrupted", True))[0]
        roll()

        assert mc_creative_krea.creative.consume() is None

    def test_an_untouched_run_is_never_treated_as_interrupted(self, client):
        events = roll()

        assert events[-1].kind == sessions.DONE


# --------------------------------------------------------------------------- #
# Nothing is left behind for the image job
# --------------------------------------------------------------------------- #


class TestItDoesNotDisturbTheGenerationItPrecedes:
    def test_the_progress_plan_is_finished_before_generate_is_clicked(self, client):
        roll()

        assert mc_progress.active() is False

    def test_a_failed_roll_leaves_no_plan_describing_the_next_job(self, client):
        client.fail = RuntimeError("nope")
        roll()

        assert mc_progress.active() is False

    def test_the_job_counters_are_left_clean(self, client):
        from modules import shared

        roll()

        assert shared.state.sampling_step == 0
        assert shared.state.sampling_steps == 0

    def test_two_rolls_in_a_row_each_claim_and_release(self, client):
        roll("task(one)")
        assert _current_task() is None
        roll("task(two)")
        assert _current_task() is None

    def test_a_reused_task_id_still_reports_as_active(self, client):
        """The host keeps a list of finished tasks and reports membership even
        on an active response, so a browser that reused an id would have its bar
        torn down on the first poll. Two rolls, same id, is the shape that
        catches it."""
        from modules import progress

        roll("task(same)")
        roll("task(same)")

        assert "task(same)" in progress.finished_tasks
        assert progress.current_task is None


# --------------------------------------------------------------------------- #
# The browser half
# --------------------------------------------------------------------------- #


SCRIPT = (Path(__file__).resolve().parent.parent / "javascript"
          / "model_chain_creative_krea.js")

HARNESS = """
const record = {progressStarted: [], args: null};

globalThis.setTimeout = (fn, ms) => 0;
globalThis.setInterval = (fn, ms) => 0;
globalThis.clearTimeout = () => {};
globalThis.clearInterval = () => {};

function holder(id, map) {
    return {
        id: id, tagName: "DIV", dataset: {},
        querySelector(selector) {
            const keys = selector.split(",").map((part) => part.trim());
            for (const key of keys) { if (map[key]) return map[key]; }
            return null;
        },
        querySelectorAll() { return []; },
        addEventListener() {},
    };
}

const elements = {
    "mc-krea-creative-toggle": holder("mc-krea-creative-toggle",
        {"input[type=checkbox]": {checked: true}}),
    "txt2img_generate": {
        tagName: "BUTTON", dataset: {},
        classList: {names: {}, toggle() {}, contains() { return false; }},
        contains() { return false; }, querySelector() { return null; },
        addEventListener() {}, click() {},
    },
    GALLERY_ENTRY
};

globalThis.document = {
    querySelector: (selector) => elements[selector.replace("#", "")] || null,
    querySelectorAll: () => [],
    addEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.gradioApp = () => globalThis.document;
globalThis.Event = function (type) { this.type = type; };
globalThis.onUiLoaded = (fn) => fn();
globalThis.onAfterUiUpdate = () => {};

RANDOM_ID
PROGRESS_FN

SOURCE

const args = window.mcKreaCreativeSubmit("placeholder", "car", 10);
console.log(JSON.stringify({
    args: args,
    started: record.progressStarted,
    exposed: typeof window.mcKreaCreativeSubmit === "function",
}));
"""


def run_js(gallery="txt2img_gallery_container", with_random_id=True,
           progress_available=True) -> dict:
    entry = (f'"{gallery}": holder("{gallery}", {{}})' if gallery else "")
    random_id = ('globalThis.randomId = () => "task(hostmade)";'
                 if with_random_id else "")
    progress_fn = ("globalThis.requestProgress = function (id, container, gallery, atEnd) "
                   "{ record.progressStarted.push([id, container ? container.id : null, "
                   "gallery]); };" if progress_available else "")
    harness = (HARNESS.replace("SOURCE", SCRIPT.read_text())
               .replace("GALLERY_ENTRY", entry)
               .replace("RANDOM_ID", random_id)
               .replace("PROGRESS_FN", progress_fn))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
class TestTheSubmitHook:
    def test_it_puts_a_fresh_task_id_in_argument_zero(self):
        """The host's own submit() idiom. A hidden textbox would be racy: the
        value has to be in the request as Gradio builds it, not in a component
        that may not have reached Gradio's state yet."""
        found = run_js()

        assert found["args"][0] == "task(hostmade)"
        assert found["args"][1:] == ["car", 10]

    def test_it_mints_its_own_id_when_the_host_has_no_helper(self):
        found = run_js(with_random_id=False)

        assert found["args"][0].startswith("task(")
        assert found["args"][0] != "placeholder"

    def test_it_asks_the_host_to_draw_the_bar_for_that_id(self):
        found = run_js()

        assert found["started"] == [["task(hostmade)", "txt2img_gallery_container", None]]

    def test_it_draws_where_image_progress_appears(self):
        """In the gallery container, so the roll's bar and the image's bar are
        the same bar in the same place rather than two things in two places."""
        found = run_js(gallery="txt2img_results")

        assert found["started"][0][1] == "txt2img_results"

    def test_it_passes_no_gallery_because_a_roll_makes_no_picture(self):
        found = run_js()

        assert found["started"][0][2] is None

    def test_a_host_that_cannot_draw_a_bar_still_gets_the_id(self):
        """A bar that will not draw must never be a roll that will not run."""
        found = run_js(progress_available=False)

        assert found["args"][0] == "task(hostmade)"
        assert found["started"] == []

    def test_a_page_with_no_gallery_container_still_gets_the_id(self):
        found = run_js(gallery="")

        assert found["args"][0] == "task(hostmade)"
        assert found["started"] == []

    def test_the_hook_is_reachable_by_the_name_gradio_resolves(self):
        assert run_js()["exposed"] is True
