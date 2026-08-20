"""The Krea Live browser controller, executed rather than read.

Two things in that file are state machines, and a state machine is worth
running.

The first is the debounce. "One request five seconds after the last keystroke"
is four separate claims -- that typing does not fire it, that the timer restarts
on every edit, that the delay is the one showing on the strip, and that it fires
exactly once -- and every one of them is a claim about a clock. Reading the
source proves none of them; a fake clock and a counter prove all four.

The second is the bypass. Live intercepts the Generate click, does its work, and
then clicks the very button it just intercepted, which is a loop unless the
second click is let through and *only* the second. That is one boolean and
exactly the kind of one boolean that is wrong in one direction (nothing ever
generates) or the other (every click generates twice) with no middle ground, so
it is driven here through the same capture-phase listener the page installs.

The clock is synthetic on purpose: these tests are about ordering, not duration,
and a test that waited five real seconds to prove a five-second debounce would
be a test nobody runs.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "javascript" / "model_chain_live_krea.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


HARNESS = """
// A txt2img page with the Krea Live strip in it, and a clock a test can move.

const record = {runs: 0, revises: 0, halts: 0, generates: 0, foreverCleared: false};

let now = 0;
let sequence = 0;
let timers = [];

globalThis.setTimeout = function (fn, ms) {
    const id = ++sequence;
    timers.push({id: id, fn: fn, at: now + (ms || 0), every: 0});
    return id;
};
globalThis.setInterval = function (fn, ms) {
    const id = ++sequence;
    timers.push({id: id, fn: fn, at: now + (ms || 0), every: Math.max(ms || 0, 1)});
    return id;
};
globalThis.clearTimeout = function (id) {
    // Forge's Generate forever keeps its timer id on window. Clearing that
    // exact id is what its own "Cancel generate forever" does, and is what this
    // records so a test can say the native loop was stopped rather than raced.
    if (id === globalThis.generateOnRepeatInterval) record.foreverCleared = true;
    timers = timers.filter((timer) => timer.id !== id);
};
globalThis.clearInterval = globalThis.clearTimeout;

// Move the clock, running whatever falls due in order. Intervals re-arm, which
// is what lets the token poll and the idle poll be driven the same way.
function advance(ms) {
    const target = now + ms;
    for (let guard = 0; guard < 10000; guard += 1) {
        const due = timers.filter((timer) => timer.at <= target)
            .sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        now = due.at;
        if (due.every) {
            due.at = now + due.every;
        } else {
            timers = timers.filter((timer) => timer.id !== due.id);
        }
        due.fn();
    }
    now = target;
}

// Let promise callbacks run. The controller resolves its promises from inside
// timer callbacks, so a test has to give the microtask queue a turn between
// moving the clock and reading the result.
const flush = () => new Promise((resolve) => setImmediate(resolve));

globalThis.Date.now = () => now;

function input(props) {
    return Object.assign({
        tagName: "INPUT",
        dataset: {},
        listeners: {},
        addEventListener(kind, fn) { this.listeners[kind] = fn; },
        dispatchEvent(event) {
            const fn = this.listeners[event.type];
            if (fn) fn(event);
            return true;
        },
    }, props);
}

function button(id, onClick) {
    return {
        id: id,
        tagName: "BUTTON",
        dataset: {},
        classList: {
            names: {},
            toggle(name, force) { this.names[name] = !!force; },
            contains(name) { return !!this.names[name]; },
        },
        contains() { return false; },
        querySelector() { return null; },
        addEventListener() {},
        click: onClick,
    };
}

function holder(id, map) {
    return {
        id: id,
        tagName: "DIV",
        dataset: {},
        querySelector(selector) {
            const keys = selector.split(",").map((part) => part.trim());
            for (const key of keys) {
                if (map[key]) return map[key];
            }
            return null;
        },
        querySelectorAll() { return []; },
        addEventListener() {},
    };
}

const toggle = input({type: "checkbox", checked: LIVE_ON});
const reroll = input({type: "checkbox", checked: REROLL_ON});
const delayField = input({value: DELAY_SECONDS});
const tokenField = input({value: ""});
const promptField = input({value: PROMPT_TEXT});
const statusLine = {tagName: "DIV", textContent: ""};

// Clicking is dispatching, exactly as a browser does it: the capture-phase
// listener runs first, and the native submission happens only if nothing
// prevented it. That is what makes the one-shot bypass observable -- the
// programmatic click the controller makes after an expansion goes through the
// same listener as the user's, and has to spend the flag on the way past.
const generateButton = button("txt2img_generate", function () {
    dispatchClick(true);
});

const progress = {tagName: "DIV"};
let generating = false;

const elements = {
    "mc-krea-live-toggle": holder("mc-krea-live-toggle", {"input[type=checkbox]": toggle}),
    "mc-krea-live-reroll": holder("mc-krea-live-reroll", {"input[type=checkbox]": reroll}),
    "mc-krea-live-delay": holder("mc-krea-live-delay", {"input": delayField}),
    "mc-krea-live-token": holder("mc-krea-live-token", {"textarea": tokenField}),
    "mc-krea-live-status": holder("mc-krea-live-status", {"div": statusLine}),
    "mc-krea-live-run": button("mc-krea-live-run", () => { record.runs += 1; }),
    "mc-krea-live-revise": button("mc-krea-live-revise", () => { record.revises += 1; }),
    "mc-krea-live-halt": button("mc-krea-live-halt", () => { record.halts += 1; }),
    "txt2img_prompt": holder("txt2img_prompt", {"textarea": promptField}),
    "txt2img_generate": generateButton,
    "txt2img_results": {
        tagName: "DIV",
        querySelector(selector) {
            return selector === ".progressDiv" && generating ? progress : null;
        },
        querySelectorAll() { return []; },
        addEventListener() {},
    },
};

const documentListeners = [];

// Declared before the button that uses it, and hoisted, because the page has
// the same shape: the listener is on the document and the button only knows how
// to dispatch at it.
function dispatchClick(trusted) {
    let prevented = false;
    const event = {
        target: generateButton,
        isTrusted: trusted !== false,
        preventDefault() { prevented = true; },
        stopImmediatePropagation() {},
    };
    documentListeners
        .filter((entry) => entry.kind === "click")
        .forEach((entry) => entry.fn(event));
    if (!prevented) record.generates += 1;
    return prevented;
}

globalThis.document = {
    querySelector: (selector) => elements[selector.replace("#", "")] || null,
    querySelectorAll: () => [],
    addEventListener(kind, fn, capture) { documentListeners.push({kind, fn, capture}); },
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.gradioApp = () => globalThis.document;
globalThis.Event = function (type) { this.type = type; this.bubbles = true; };
globalThis.appendContextMenuOption = () => {};
globalThis.generateOnRepeatInterval = 4242;

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());

const kl = globalThis.modelChainKreaLive;

function clickGenerate(trusted) {
    return dispatchClick(trusted !== false);
}

function type(text) {
    promptField.value = text;
    promptField.dispatchEvent({type: "input"});
}

function answer(value) {
    tokenField.value = value;
}

const report = (extra) => console.log(JSON.stringify(Object.assign({
    runs: record.runs,
    revises: record.revises,
    halts: record.halts,
    generates: record.generates,
    foreverCleared: record.foreverCleared,
    armed: generateButton.classList.contains("mc-krea-live-armed"),
    status: statusLine.textContent,
    phase: kl.state.phase,
    bypass: kl.state.bypass,
    pending: kl.state.pending,
}, extra || {})));

BODY
"""


def run(body: str, live_on: bool = True, reroll_on: bool = False,
        delay_seconds: float = 5, source: str = "a lighthouse in a storm") -> dict:
    harness = (
        HARNESS.replace("SOURCE", SCRIPT.read_text())
        .replace("BODY", body)
        .replace("LIVE_ON", json.dumps(bool(live_on)))
        .replace("REROLL_ON", json.dumps(bool(reroll_on)))
        .replace("DELAY_SECONDS", json.dumps(delay_seconds))
        .replace("PROMPT_TEXT", json.dumps(source))
    )
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# The debounce
# --------------------------------------------------------------------------- #


class TestTypingDoesNotGenerate:
    def test_a_burst_of_keystrokes_asks_for_nothing(self):
        """The whole reason the delay exists. A request per keystroke is a
        request per keystroke however fast the model is."""
        found = run("""
            for (const text of ["a", "a l", "a lig", "a light", "a lighthouse"]) {
                type(text);
                advance(200);
            }
            report();
        """)

        assert found["runs"] == 0

    def test_stopping_for_the_delay_asks_exactly_once(self):
        found = run("""
            type("a lighthouse");
            advance(5100);
            report();
        """)

        assert found["runs"] == 1

    def test_a_new_edit_invalidates_the_timer_the_last_one_armed(self):
        """Otherwise a pause halfway through a sentence generates the half."""
        found = run("""
            type("a light");
            advance(4000);
            type("a lighthouse");
            advance(4000);
            report();
        """)

        assert found["runs"] == 0

    def test_and_then_fires_once_for_the_newest_text(self):
        found = run("""
            type("a light");
            advance(4000);
            type("a lighthouse");
            advance(6000);
            report();
        """)

        assert found["runs"] == 1

    def test_the_delay_is_the_one_showing_on_the_strip(self):
        found = run("""
            type("a lighthouse");
            advance(1200);
            report();
        """, delay_seconds=1)

        assert found["runs"] == 1

    def test_a_delay_changed_between_keystrokes_takes_effect_immediately(self):
        found = run("""
            type("a light");
            delayField.value = 1;
            type("a lighthouse");
            advance(1200);
            report();
        """)

        assert found["runs"] == 1

    def test_nothing_happens_while_live_is_off(self):
        found = run("""
            type("a lighthouse");
            advance(30000);
            report();
        """, live_on=False)

        assert found["runs"] == 0

    def test_an_empty_prompt_asks_for_nothing(self):
        found = run("""
            type("   ");
            advance(9000);
            report();
        """)

        assert found["runs"] == 0

    def test_editing_during_an_expansion_tells_the_server_once(self):
        """Once per typing burst, not once per keystroke: the point of doing the
        counting in the browser is that the server never sees the keystrokes."""
        found = run("""
            type("a lighthouse");
            advance(5100);
            type("a lighthouse at dawn");
            type("a lighthouse at dawn in fog");
            advance(100);
            report();
        """)

        assert found["runs"] == 1
        assert found["revises"] == 1

    def test_no_revision_is_sent_when_nothing_is_in_flight(self):
        found = run("""
            type("a lighthouse");
            type("a lighthouse at dawn");
            advance(100);
            report();
        """)

        assert found["revises"] == 0


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


class TestTheGenerateGate:
    def test_live_off_leaves_the_button_completely_alone(self):
        """The first thing this feature must not break. With the toggle off,
        txt2img is txt2img."""
        found = run("""
            const prevented = clickGenerate();
            report({prevented: prevented});
        """, live_on=False)

        assert found["prevented"] is False
        assert found["runs"] == 0
        assert found["generates"] == 1

    def test_live_on_holds_the_click_and_asks_for_an_expansion_first(self):
        """Before the native submission, never after: an LLM run waits for the
        host to stop generating, so an expansion asked for from inside a running
        image job would be waiting on the job that is waiting on it."""
        found = run("""
            const prevented = clickGenerate();
            report({prevented: prevented});
        """)

        assert found["prevented"] is True
        assert found["runs"] == 1
        assert found["generates"] == 0

    def test_a_ready_expansion_releases_exactly_one_native_generation(self):
        found = run("""
            clickGenerate();
            answer("ready:abcd:");
            advance(200);
            await flush();
            report();
        """)

        assert found["runs"] == 1
        assert found["generates"] == 1

    def test_the_bypass_is_spent_by_the_click_it_was_set_for(self):
        """One boolean, wrong in one direction if nothing ever generates and in
        the other if every click generates twice."""
        found = run("""
            clickGenerate();
            answer("ready:abcd:");
            advance(200);
            await flush();
            const second = clickGenerate();
            report({secondPrevented: second});
        """)

        assert found["generates"] == 1
        assert found["bypass"] is False
        assert found["secondPrevented"] is True

    def test_a_failed_expansion_starts_no_generation(self):
        found = run("""
            clickGenerate();
            answer("failed:abcd:");
            advance(200);
            await flush();
            report();
        """)

        assert found["generates"] == 0
        assert found["phase"] == "idle"

    def test_text_that_changed_while_the_model_wrote_is_never_generated_from(self):
        found = run("""
            clickGenerate();
            type("something else entirely");
            answer("ready:abcd:");
            advance(200);
            await flush();
            report();
        """)

        assert found["generates"] == 0
        assert found["pending"] is True

    def test_a_second_click_during_an_expansion_does_not_start_a_second_cycle(self):
        found = run("""
            clickGenerate();
            clickGenerate();
            clickGenerate();
            report();
        """)

        assert found["runs"] == 1

    def test_the_native_repeat_loop_is_refused_rather_than_raced(self):
        """Two repeat schedulers pressing the same button is not something to
        referee. Krea Live's own Reroll knows how to reuse the cached expansion;
        Generate forever does not."""
        found = run("""
            const prevented = clickGenerate(false);
            report({prevented: prevented});
        """)

        assert found["prevented"] is True
        assert found["foreverCleared"] is True
        assert found["runs"] == 0
        assert "Reroll" in found["status"]

    def test_the_native_repeat_loop_is_untouched_while_live_is_off(self):
        found = run("""
            clickGenerate(false);
            report();
        """, live_on=False)

        assert found["foreverCleared"] is False


# --------------------------------------------------------------------------- #
# Repetition and the visible state
# --------------------------------------------------------------------------- #


class TestRerollAndState:
    def test_a_reroll_goes_back_through_the_same_gate(self):
        """Which is what makes it free: the gate finds the cached expansion,
        makes no request, and the only thing that changes is the image seed
        Forge draws."""
        found = run("""
            clickGenerate();
            answer("ready:one:");
            advance(200);
            await flush();
            generating = true;
            advance(500);
            generating = false;
            advance(1000);
            await flush();
            advance(500);
            await flush();
            answer("ready:two:");
            advance(300);
            await flush();
            report();
        """, reroll_on=True)

        assert found["generates"] == 2

    def test_reroll_off_generates_once_and_stops(self):
        found = run("""
            clickGenerate();
            answer("ready:one:");
            advance(200);
            await flush();
            generating = true;
            advance(500);
            generating = false;
            advance(2000);
            await flush();
            advance(2000);
            await flush();
            report();
        """, reroll_on=False)

        assert found["generates"] == 1

    def test_stopping_live_leaves_no_timer_able_to_fire(self):
        """A stop that left a timer armed would be a stop that generated again
        five seconds later, which is the worst possible answer to Stop."""
        found = run("""
            type("a lighthouse");
            kl.stopLive();
            advance(60000);
            await flush();
            report();
        """)

        assert found["runs"] == 0
        assert found["generates"] == 0
        assert found["halts"] == 1

    def test_stopping_live_turns_the_visible_toggle_off_too(self):
        """The strip and the context menu must never disagree about whether the
        positive prompt has changed meaning."""
        found = run("""
            kl.stopLive();
            report({checked: toggle.checked});
        """)

        assert found["checked"] is False
        assert found["armed"] is False

    def test_starting_live_from_the_menu_turns_the_visible_toggle_on(self):
        found = run("""
            kl.startLive();
            report({checked: toggle.checked});
        """, live_on=False)

        assert found["checked"] is True
        assert found["armed"] is True

    def test_the_armed_state_is_painted_on_the_button_whose_meaning_changed(self):
        found = run("""
            report();
        """)

        assert found["armed"] is True
