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
    monkeypatch.setattr(sessions, "_client", lambda needs_vision=False, reserve=0: fake)
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


class TestTheBarNeverBlocksTheRunItDescribes:
    """The regression this class exists for, in one sentence: an indicator that
    tells the host a job is running makes the language model wait for a job that
    is itself.

    ``mc_broker.host_busy()`` is "``state.job`` or ``state.job_count`` is
    truthy", and ``mc_llm_sessions._Gpu.acquire`` refuses to start while it is
    true. A progress claim that set either field therefore hung the roll it was
    drawing a bar for -- intermittently, because any Gradio call finishing
    nearby clears both in its own ``finally``.
    """

    def test_the_host_is_not_busy_at_any_point_during_a_roll(self, client):
        seen = []
        original = mc_llm_progress.reporter.enter

        def watch(phase):
            original(phase)
            seen.append(mc_broker.host_busy())

        mc_llm_progress.reporter.enter = watch
        roll()

        assert seen and not any(seen)

    def test_the_reporter_never_touches_the_two_fields_that_mean_busy(self, client):
        from modules import shared

        shared.state.job = ""
        shared.state.job_count = 0
        reporter = mc_llm_progress.Reporter()
        reporter.begin("task(x)", 2000, warm=True)
        try:
            reporter.enter(mc_progress.PHASE_KREA_READ)
            reporter.enter(mc_progress.PHASE_KREA_WRITE)
            reporter.wrote("some text")

            assert shared.state.job == ""
            assert shared.state.job_count == 0
            assert mc_broker.host_busy() is False
        finally:
            reporter.end("some text")

        assert mc_broker.host_busy() is False

    def test_the_gpu_wait_is_not_entered_because_of_us(self, client):
        """The observable symptom was a console line saying the LLM was waiting
        for an image generation, on a machine generating nothing."""
        waited = [event.text for event in roll()
                  if event.kind == sessions.STATUS and "image generation" in event.text]

        assert waited == []

    def test_an_image_job_that_really_is_running_still_holds_the_llm_back(self, host):
        """The guard the deadlock fix must not have removed. host_busy() is
        still consulted and still says yes for a real generation; it simply no
        longer says yes because of a progress bar.

        Deliberately without the ``client`` fixture, which stubs host_busy out
        -- a test of host_busy that stubbed host_busy would assert nothing.
        """
        from modules import shared

        assert mc_broker.host_busy() is False
        shared.state.job_count = 1
        try:
            assert mc_broker.host_busy() is True
        finally:
            shared.state.job_count = 0
        shared.state.job = "txt2img"
        try:
            assert mc_broker.host_busy() is True
        finally:
            shared.state.job = ""

    def test_the_whole_gate_runs_with_the_real_broker_watching(self, monkeypatch,
                                                               host, store):
        """End to end, with ``host_busy`` *not* stubbed.

        Every other test in this file replaces it, which is right for what they
        are each about and useless for this one: the deadlock was a real call to
        a real ``host_busy`` returning a real True. So this drives the actual
        txt2img gate through the actual ``_Gpu.acquire`` and asserts the answer
        was False at every frame.
        """
        import model_chain_krea_creative as creative_script
        from prompt_master.krea import director
        from prompt_master.krea import library as library_module

        mc_broker.clear()
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False, reserve=0: FakeClient())
        monkeypatch.setattr(sessions, "_placement_notes", list)
        monkeypatch.setattr(mc_creative_krea, "checkpoint_objection", lambda: "")
        mc_creative_krea.creative = mc_creative_krea.Creative()
        mc_llm_progress.reporter = mc_llm_progress.Reporter()

        axes: list = []
        for _ in library_module.library().axis_keys:
            axes.extend([director.VARY, None])

        busy = []
        signals = []
        try:
            for frame in creative_script._gate("car", 6, -1, True, "", *axes):
                busy.append(mc_broker.host_busy())
                if isinstance(frame[1], str):
                    signals.append(frame[1].split(":")[0])
        finally:
            mc_broker.clear()

        assert not any(busy)
        assert signals[0] == "task"
        assert signals[-1] == "ready"
        assert mc_creative_krea.creative.armed is not None

    def test_the_source_says_why(self):
        """A comment, asserted, because the two lines removed here are exactly
        the two lines somebody adds back for completeness."""
        source = Path(mc_llm_progress.__file__).read_text(encoding="utf-8")

        assert "host_busy()" in source
        assert "state.job" in source


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
// A txt2img page whose hidden box a test can put server messages into, so the
// handshake can be driven exactly as the server drives it.

const record = {progressStarted: [], generates: 0, rolls: 0};

let queue = [];
let now = 0;
globalThis.setInterval = function (fn) { queue.push(fn); return queue.length; };
globalThis.clearInterval = function (id) { queue[id - 1] = null; };
globalThis.setTimeout = function () { return 0; };
globalThis.clearTimeout = function () {};
globalThis.Date.now = () => now;

// Run every live poll once. The controller polls on an interval; a test needs
// to say "and then a moment passed" without one.
function tick() {
    queue.filter(Boolean).forEach((fn) => fn());
}
const flush = () => new Promise((resolve) => setImmediate(resolve));

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

const toggle = {type: "checkbox", checked: true, dataset: {}, addEventListener() {}};
const tokenField = {value: ""};
const promptField = {value: "car"};
const statusLine = {tagName: "DIV", textContent: ""};

const generateButton = {
    id: "txt2img_generate", tagName: "BUTTON", dataset: {},
    classList: {names: {}, toggle() {}, contains() { return false; }},
    contains() { return false; }, querySelector() { return null; },
    addEventListener() {},
    click() { record.generates += 1; },
};

const elements = {
    "mc-krea-creative-toggle": holder("mc-krea-creative-toggle",
        {"input[type=checkbox]": toggle}),
    "mc-krea-creative-token": holder("mc-krea-creative-token", {"textarea": tokenField}),
    "mc-krea-creative-status": holder("mc-krea-creative-status", {"div": statusLine}),
    "mc-krea-creative-run": {
        id: "mc-krea-creative-run", tagName: "BUTTON", dataset: {},
        contains() { return false; }, querySelector() { return null; },
        addEventListener() {}, click() { record.rolls += 1; },
    },
    "txt2img_prompt": holder("txt2img_prompt", {"textarea": promptField}),
    "txt2img_generate": generateButton,
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
PROGRESS_FN

SOURCE

const kc = globalThis.modelChainKreaCreative;

// What the server sends down the hidden box, in order.
function server(value) { tokenField.value = value; }

BODY
"""


def run_js(body: str, gallery="txt2img_gallery_container",
           progress_available=True) -> dict:
    entry = (f'"{gallery}": holder("{gallery}", {{}})' if gallery else "")
    progress_fn = ("globalThis.requestProgress = function (id, container, gallery, atEnd) "
                   "{ record.progressStarted.push([id, container ? container.id : null, "
                   "gallery]); };" if progress_available else "")
    harness = (HARNESS.replace("SOURCE", SCRIPT.read_text())
               .replace("BODY", body)
               .replace("GALLERY_ENTRY", entry)
               .replace("PROGRESS_FN", progress_fn))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


REPORT = """
console.log(JSON.stringify({
    started: record.progressStarted,
    rolls: record.rolls,
    generates: record.generates,
    rolling: kc.state.rolling,
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
class TestTheHandshake:
    """How the browser learns which task the bar is for.

    It used to mint the id and pass it in through a Gradio ``js=`` hook. Now the
    server mints it and sends it down the hidden box the browser was already
    polling for the arming token -- so nothing has to arrive intact for the roll
    to work, and a bar that cannot be drawn costs a bar rather than a
    generation.
    """

    def test_the_bar_starts_when_the_server_names_the_task(self):
        found = run_js("""
            kc.runRoll();
            server("task:abcd:task(mc-123456)");
            tick();
            await flush();
        """ + REPORT)

        assert found["started"] == [["task(mc-123456)", "txt2img_gallery_container", None]]

    def test_naming_the_task_is_not_an_outcome_and_polling_continues(self):
        """The old shape resolved on the first new value it saw. If it did that
        now, the task message would be read as "not ready" and no image would
        ever be generated."""
        found = run_js("""
            kc.runRoll();
            server("task:abcd:task(mc-123456)");
            tick();
            await flush();
        """ + REPORT)

        assert found["rolling"] is True
        assert found["generates"] == 0

    def test_the_arming_token_still_releases_one_generation(self):
        found = run_js("""
            const done = kc.runRoll();
            server("task:abcd:task(mc-123456)");
            tick();
            await flush();
            server("ready:efgh:armed");
            tick();
            await flush();
            await done;
        """ + REPORT)

        assert found["generates"] == 1
        assert found["rolling"] is False

    def test_a_failure_after_the_task_starts_no_generation(self):
        found = run_js("""
            const done = kc.runRoll();
            server("task:abcd:task(mc-123456)");
            tick();
            await flush();
            server("failed:efgh:");
            tick();
            await flush();
            await done;
        """ + REPORT)

        assert found["generates"] == 0

    def test_a_roll_with_no_bar_available_still_generates(self):
        """The property the handshake was rearranged for."""
        found = run_js("""
            const done = kc.runRoll();
            server("task:abcd:task(mc-123456)");
            tick();
            await flush();
            server("ready:efgh:armed");
            tick();
            await flush();
            await done;
        """ + REPORT, progress_available=False)

        assert found["started"] == []
        assert found["generates"] == 1

    def test_a_page_with_no_gallery_container_still_generates(self):
        found = run_js("""
            const done = kc.runRoll();
            server("task:abcd:task(mc-123456)");
            tick();
            await flush();
            server("ready:efgh:armed");
            tick();
            await flush();
            await done;
        """ + REPORT, gallery="")

        assert found["started"] == []
        assert found["generates"] == 1

    def test_it_draws_where_image_progress_appears(self):
        found = run_js("""
            kc.runRoll();
            server("task:abcd:task(mc-123456)");
            tick();
            await flush();
        """ + REPORT, gallery="txt2img_results")

        assert found["started"][0][1] == "txt2img_results"

    def test_it_passes_no_gallery_because_a_roll_makes_no_picture(self):
        found = run_js("""
            kc.runRoll();
            server("task:abcd:task(mc-123456)");
            tick();
            await flush();
        """ + REPORT)

        assert found["started"][0][2] is None

    def test_an_empty_task_id_draws_nothing(self):
        found = run_js("""
            kc.startBar("");
        """ + REPORT)

        assert found["started"] == []

    def test_the_page_is_never_asked_to_call_a_js_hook_by_name(self):
        """Gradio's js= contract is one this extension no longer depends on."""
        source = SCRIPT.read_text()

        assert "mcKreaCreativeSubmit" not in source


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
class TestTheServerMintsTheId:
    def test_the_gate_names_the_task_before_it_does_any_work(self):
        """First frame out, so the bar covers the Director and the GPU wait as
        well as the model call."""
        import model_chain_krea_creative as creative_script

        assert creative_script._task_id().startswith("task(")
        assert ":" not in creative_script._task_id()

    def test_two_rolls_never_share_an_id(self):
        import model_chain_krea_creative as creative_script

        minted = {creative_script._task_id() for _ in range(50)}

        assert len(minted) == 50
