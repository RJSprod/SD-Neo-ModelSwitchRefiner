"""The Creative Mode browser file, executed rather than read.

There is not much left of it, and the tests are mostly about that.

This file used to hold a state machine. Creative Mode intercepted the Generate
click, ran its roll, and then clicked the very button it had just intercepted --
which is an infinite loop unless the second click is let through and *only* the
second. That was one boolean, and exactly the kind of one boolean that is wrong
in one direction (nothing ever generates) or the other (every click generates
twice), so it was driven here through the real capture-phase listener.

The boolean is gone because the interception is. A press of Generate is a press
of Generate again: the roll happens inside the generation, in Python, and
nothing in the browser stands between the button and the image. That is the
property this file exists to defend, because it is the difference between a
generation that survives a closed tab and one that does not -- and it is checked
the only way it can honestly be checked, by dispatching a click at the real
listener list and asserting the native submission happened.

The rest is what is *not* there. The controller two designs ago had a debounce,
a repeat scheduler and a typing watcher; the one before this had a poll loop and
a fifteen-minute timeout. A synthetic clock is the way to ask whether any of
them came back: run it forward an hour with nobody touching anything and assert
that nothing happened, and that nothing is still waiting to.

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

const record = {generates: 0, rolls: 0, prevented: 0};

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
const promptField = input({value: PROMPT_TEXT});

const generateButton = button("txt2img_generate", function () {
    dispatchClick(true);
});

// The gate's half of the page, still present so that a file which went looking
// for it would find it. Nothing may press this button: a roll started from the
// browser is the arrangement that made a generation depend on a live tab.
const runButton = button("mc-krea-creative-run", () => { record.rolls += 1; });

const elements = {
    "mc-krea-creative-toggle": holder("mc-krea-creative-toggle",
        {"input[type=checkbox]": toggle}),
    "mc-krea-creative-token": holder("mc-krea-creative-token", {"textarea": input({value: ""})}),
    "mc-krea-creative-status": holder("mc-krea-creative-status",
        {"div": {tagName: "DIV", textContent: ""}}),
    "mc-krea-creative-run": runButton,
    "txt2img_prompt": holder("txt2img_prompt", {"textarea": promptField}),
    "txt2img_generate": generateButton,
};

const documentListeners = [];

// Clicking is dispatching, exactly as a browser does it: any capture-phase
// listener runs first, and the native submission happens only if nothing
// prevented it. So "the button still generates" is observable rather than
// argued from the absence of a listener.
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
    if (prevented) {
        record.prevented += 1;
    } else {
        record.generates += 1;
    }
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

const report = (extra) => console.log(JSON.stringify(Object.assign({
    rolls: record.rolls,
    generates: record.generates,
    prevented: record.prevented,
    armed: generateButton.classList.contains("mc-krea-creative-armed"),
    timers: timers.length,
    clickListeners: documentListeners.filter((e) => e.kind === "click").length,
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
# The button is the host's again
# --------------------------------------------------------------------------- #


class TestGenerateIsNeverIntercepted:
    """The whole point of the change this file was rewritten for.

    While the browser held the click, a press did not start an image -- it
    started a roll, and a timer in the page started the image afterwards. A
    hidden tab throttles that timer to one tick a second, a frozen one to a tick
    a minute, and a closed one never runs it at all, so a Creative generation was
    late if you changed windows and absent if you closed the tab. Every test
    below is a way of asking whether the press reaches Forge on its own.
    """

    def test_a_press_generates_immediately_with_creative_mode_on(self):
        found = run("""
            const prevented = clickGenerate();
            report({returned: prevented});
        """)

        assert found["returned"] is False
        assert found["generates"] == 1
        assert found["prevented"] == 0

    def test_a_press_generates_immediately_with_creative_mode_off(self):
        found = run("""
            clickGenerate();
            report();
        """, creative_on=False)

        assert found["generates"] == 1
        assert found["prevented"] == 0

    def test_nothing_in_the_page_presses_the_hidden_roll_button(self):
        """It is still in the page for anything that looks for it, and this file
        must never be what looks."""
        found = run("""
            clickGenerate();
            advance(60 * 60 * 1000);
            await flush();
            report();
        """)

        assert found["rolls"] == 0

    def test_the_press_does_not_wait_for_the_page_to_be_told_anything(self):
        """No second click arrives later, because there is no first one being
        held back. One press is one generation, decided synchronously."""
        found = run("""
            clickGenerate();
            advance(60 * 60 * 1000);
            await flush();
            report();
        """)

        assert found["generates"] == 1

    def test_three_presses_are_three_generations(self):
        found = run("""
            clickGenerate();
            clickGenerate();
            clickGenerate();
            report();
        """)

        assert found["generates"] == 3
        assert found["prevented"] == 0

    def test_an_empty_prompt_is_still_the_host_business(self):
        """Whether an empty prompt generates is Forge's decision and was never
        this file's to make."""
        found = run("""
            clickGenerate();
            report();
        """, prompt="   ")

        assert found["prevented"] == 0

    def test_no_click_listener_is_installed_at_all(self):
        found = run("report();")

        assert found["clickListeners"] == 0


# --------------------------------------------------------------------------- #
# Nothing happens on its own
# --------------------------------------------------------------------------- #


class TestNothingIsScheduled:
    def test_typing_never_starts_anything(self):
        found = run("""
            for (const text of ["a", "a c", "a ca", "a car"]) {
                type(text);
                advance(1000);
            }
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
        assert found["generates"] == 0

    def test_a_press_arms_no_timer_either(self):
        """The poll loop is the specific thing that must not come back: it is
        the mechanism a background tab throttles and a closed one kills."""
        found = run("""
            clickGenerate();
            await flush();
            report();
        """)

        assert found["timers"] == 0


# --------------------------------------------------------------------------- #
# What the file may contain at all
# --------------------------------------------------------------------------- #


def code() -> str:
    """The file with its comments taken out.

    The header explains at length what this file used to do, and it names every
    mechanism the assertions below forbid. Checking the comments would make the
    explanation unwritable, which is the wrong way round: the prose is how the
    next person learns why none of it may come back.
    """
    lines = []
    for line in SCRIPT.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        lines.append(line.split("//")[0] if " // " in line else line)
    return "\n".join(lines)


class TestTheFileItself:
    """Read rather than run, for the handful of things whose absence is the
    feature. A file with no timers today grows one the next time somebody wants
    to know when the server has finished something."""

    def test_it_never_defers_anything(self):
        source = code()

        assert "setInterval" not in source
        assert "setTimeout" not in source
        assert "requestAnimationFrame" not in source

    def test_it_never_stops_an_event(self):
        source = code()

        assert "preventDefault" not in source
        assert "stopImmediatePropagation" not in source

    def test_it_never_clicks_anything(self):
        source = code()

        assert ".click(" not in source

    def test_the_page_is_never_asked_to_call_a_js_hook_by_name(self):
        """Gradio's js= contract is one this extension does not depend on."""
        assert "mcKreaCreativeSubmit" not in SCRIPT.read_text()


# --------------------------------------------------------------------------- #
# The visible state
# --------------------------------------------------------------------------- #


class TestTheArmedIndicator:
    """The only thing left that this file does.

    Generate does still mean something different with Creative Mode on -- it
    writes a prompt before it makes a picture -- and saying so on the button is
    worth four lines. It is also the whole blast radius: if every line of this
    file fails, a button goes unpainted and txt2img generates exactly as it
    would have.
    """

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

    def test_a_page_without_the_controls_is_not_an_exception(self):
        found = run("""
            delete elements["mc-krea-creative-toggle"];
            delete elements["txt2img_generate"];
            kc.wire();
            clickGenerate();
            report();
        """)

        assert found["generates"] == 1
