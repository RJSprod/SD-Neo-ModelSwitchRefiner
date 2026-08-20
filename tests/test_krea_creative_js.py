"""The Creative Mode browser gate, executed rather than read.

One thing in that file is a state machine, and a state machine is worth running.

Creative Mode intercepts the Generate click, does its work, and then clicks the
very button it just intercepted -- which is an infinite loop unless the second
click is let through and *only* the second. That is one boolean, and exactly the
kind of one boolean that is wrong in one direction (nothing ever generates) or
the other (every click generates twice) with no middle ground. So it is driven
here through the same capture-phase listener the page installs, with a fake
button that dispatches the way a real one does.

The rest of the file is about what is *not* there. The controller this replaced
had a debounce, a repeat scheduler and a typing watcher, and the first thing
anybody will want to know about the replacement is that none of them came back.
A synthetic clock is the way to ask: run it forward an hour with nobody touching
anything and assert that nothing happened.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parent.parent / "javascript"
          / "model_chain_creative_krea.js")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


HARNESS = """
// A txt2img page with the Creative Mode controls in it, and a clock a test can
// move. The clock is synthetic because these tests are about ordering, not
// duration, and because "nothing fires on its own" is only checkable if a test
// can run an hour forward in a millisecond.

const record = {rolls: 0, generates: 0};

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
    timers = timers.filter((timer) => timer.id !== id);
};
globalThis.clearInterval = globalThis.clearTimeout;

function advance(ms) {
    const target = now + ms;
    for (let guard = 0; guard < 100000; guard += 1) {
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

// Let promise callbacks run: the controller resolves from inside a timer, so a
// test has to give the microtask queue a turn between moving the clock and
// reading the result.
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

const toggle = input({type: "checkbox", checked: CREATIVE_ON});
const tokenField = input({value: ""});
const promptField = input({value: PROMPT_TEXT});
const statusLine = {tagName: "DIV", textContent: ""};

// Clicking is dispatching, exactly as a browser does it: the capture-phase
// listener runs first, and the native submission happens only if nothing
// prevented it. That is what makes the one-shot bypass observable -- the
// programmatic click the controller makes after a roll goes through the same
// listener as the user's, and has to spend the flag on the way past.
const generateButton = button("txt2img_generate", function () {
    dispatchClick(true);
});

const elements = {
    "mc-krea-creative-toggle": holder("mc-krea-creative-toggle",
        {"input[type=checkbox]": toggle}),
    "mc-krea-creative-token": holder("mc-krea-creative-token", {"textarea": tokenField}),
    "mc-krea-creative-status": holder("mc-krea-creative-status", {"div": statusLine}),
    "mc-krea-creative-run": button("mc-krea-creative-run", () => { record.rolls += 1; }),
    "txt2img_prompt": holder("txt2img_prompt", {"textarea": promptField}),
    "txt2img_generate": generateButton,
};

const documentListeners = [];

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

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());

const kc = globalThis.modelChainKreaCreative;

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
    rolls: record.rolls,
    generates: record.generates,
    armed: generateButton.classList.contains("mc-krea-creative-armed"),
    status: statusLine.textContent,
    rolling: kc.state.rolling,
    bypass: kc.state.bypass,
    timers: timers.length,
}, extra || {})));

BODY
"""


def run(body: str, creative_on: bool = True, prompt: str = "car") -> dict:
    harness = (
        HARNESS.replace("SOURCE", SCRIPT.read_text())
        .replace("BODY", body)
        .replace("CREATIVE_ON", json.dumps(bool(creative_on)))
        .replace("PROMPT_TEXT", json.dumps(prompt))
    )
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# Nothing happens on its own
# --------------------------------------------------------------------------- #


class TestNothingIsScheduled:
    def test_typing_never_starts_anything(self):
        """The headline difference from what this replaced. Editing the prompt
        is editing the prompt."""
        found = run("""
            for (const text of ["a", "a c", "a ca", "a car"]) {
                type(text);
                advance(1000);
            }
            report();
        """)

        assert found["rolls"] == 0
        assert found["generates"] == 0

    def test_stopping_typing_never_starts_anything(self):
        found = run("""
            type("a car");
            advance(60 * 60 * 1000);
            report();
        """)

        assert found["rolls"] == 0
        assert found["generates"] == 0

    def test_an_hour_of_nobody_touching_it_leaves_no_timer_armed(self):
        """A leftover timer is the failure nobody notices until it generates on
        its own, so it is asked about directly."""
        found = run("""
            advance(60 * 60 * 1000);
            report();
        """)

        assert found["timers"] == 0
        assert found["rolls"] == 0

    def test_a_finished_roll_arms_nothing_for_later(self):
        found = run("""
            clickGenerate();
            answer("ready:abcd:");
            advance(200);
            await flush();
            advance(60 * 60 * 1000);
            report();
        """)

        assert found["generates"] == 1
        assert found["timers"] == 0


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


class TestTheGenerateGate:
    def test_creative_mode_off_leaves_the_button_completely_alone(self):
        """The first thing this feature must not break: with the toggle off,
        txt2img is txt2img."""
        found = run("""
            const prevented = clickGenerate();
            report({prevented: prevented});
        """, creative_on=False)

        assert found["prevented"] is False
        assert found["rolls"] == 0
        assert found["generates"] == 1

    def test_creative_mode_on_holds_the_click_and_rolls_first(self):
        """Before the native submission, never after: an LLM run waits for the
        host to stop generating, so a roll asked for from inside a running image
        job would be waiting on the job that is waiting on it."""
        found = run("""
            const prevented = clickGenerate();
            report({prevented: prevented});
        """)

        assert found["prevented"] is True
        assert found["rolls"] == 1
        assert found["generates"] == 0

    def test_a_ready_roll_releases_exactly_one_native_generation(self):
        found = run("""
            clickGenerate();
            answer("ready:abcd:");
            advance(200);
            await flush();
            report();
        """)

        assert found["rolls"] == 1
        assert found["generates"] == 1

    def test_the_bypass_is_spent_by_the_click_it_was_set_for(self):
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

    def test_a_second_press_is_a_second_roll_and_a_second_image(self):
        """Explicit means explicit. Pressing it again is how you get another
        one, and it is the only way."""
        found = run("""
            clickGenerate();
            answer("ready:one:");
            advance(200);
            await flush();
            clickGenerate();
            answer("ready:two:");
            advance(200);
            await flush();
            report();
        """)

        assert found["rolls"] == 2
        assert found["generates"] == 2

    def test_a_failed_roll_starts_no_generation(self):
        found = run("""
            clickGenerate();
            answer("failed:abcd:");
            advance(200);
            await flush();
            report();
        """)

        assert found["generates"] == 0
        assert found["rolling"] is False

    def test_pressing_again_mid_roll_does_not_start_a_second_one(self):
        """Swallowed rather than queued: the roll in flight will click Generate
        itself when it is ready, and a queued second roll would be a second model
        call nobody asked for."""
        found = run("""
            clickGenerate();
            clickGenerate();
            clickGenerate();
            report();
        """)

        assert found["rolls"] == 1
        assert "Still writing" in found["status"]

    def test_an_empty_prompt_rolls_nothing(self):
        found = run("""
            clickGenerate();
            report();
        """, prompt="   ")

        assert found["rolls"] == 0
        assert found["generates"] == 0

    def test_turning_creative_mode_off_mid_roll_abandons_it(self):
        found = run("""
            clickGenerate();
            toggle.checked = false;
            advance(200);
            await flush();
            answer("ready:abcd:");
            advance(200);
            await flush();
            report();
        """)

        assert found["generates"] == 0


# --------------------------------------------------------------------------- #
# The visible state
# --------------------------------------------------------------------------- #


class TestTheArmedIndicator:
    def test_it_is_painted_on_the_button_whose_meaning_changed(self):
        assert run("report();")["armed"] is True

    def test_it_is_absent_while_creative_mode_is_off(self):
        assert run("report();", creative_on=False)["armed"] is False

    def test_it_follows_the_toggle(self):
        found = run("""
            toggle.checked = false;
            toggle.dispatchEvent({type: "change"});
            report();
        """)

        assert found["armed"] is False
