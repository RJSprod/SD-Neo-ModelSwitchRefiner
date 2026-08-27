"""The LLM Studio polish script, executed rather than read.

One thing in that file is arithmetic, and arithmetic is worth running.

It measures how much room the conversation workspace has and publishes it as a
custom property, and the first version of it measured from
``getBoundingClientRect().top`` alone. That falls as the page scrolls, so
``innerHeight - top`` *grows* as the page scrolls -- and since the number is
then used to set the height of an element on that page, a taller element means
more page to scroll, which means a larger measurement next time. What that
looks like from the outside is a panel that grows a little on every click, with
blank space under the messages, and it is not a thing any amount of reading the
file makes obvious.

So the property is asserted to be **scroll-invariant**: the same element, in the
same place in the document, measured at two scroll positions, has to publish the
same height. That is the whole bug, stated as a test.

The transcript's anchoring is here for the same reason and has the same shape of
bug behind it. Whether to follow a reply is a question about where the reader
*was*, and the first version asked it from inside a MutationObserver -- which
runs after the new content is in the DOM, so a reply longer than the slack made
the answer "no" for somebody who had been at the bottom a millisecond earlier.
The observer's own callback is captured and driven here, against a scroller
whose numbers a test can set, which is the only way to ask "and what did it do
about it?" without a browser.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "javascript" / "llm_studio.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


HARNESS = """
// A workspace WORKSPACE_TOP pixels down a document, in a WINDOW_HEIGHT window,
// looked at with the page scrolled by SCROLLED. The tab around it is offered
// too, higher up the page, so that a version of the script which measures the
// wrong element is measured rather than skipped.
const STUDIO_TOP = WORKSPACE_TOP - 60;

function element(id, top) {
    const style = {
        values: {},
        setProperty(name, value) { style.values[name] = value; },
        removeProperty(name) { delete style.values[name]; },
        getPropertyValue(name) { return style.values[name] || ""; },
    };
    return {
        id,
        style,
        top,
        // Not null: null is how this script recognises an element in a tab
        // that is not open, and skips it.
        offsetParent: {},
        dataset: {},
        tagName: "DIV",
        scrollHeight: 2000,
        clientHeight: 400,
        scrollTop: 0,
        querySelector: () => null,
        querySelectorAll: () => [],
        addEventListener() {},
        getBoundingClientRect() {
            // The viewport-relative top falls by exactly the scroll offset,
            // which is the whole of what a scrolled page changes.
            return {top: top - SCROLLED, height: 400, bottom: 0, left: 0, right: 0};
        },
    };
}

const elements = {
    "mc-llm-studio": element("mc-llm-studio", STUDIO_TOP),
    "mc-llm-chat": element("mc-llm-chat", WORKSPACE_TOP),
};

globalThis.document = {
    documentElement: {scrollTop: SCROLLED, getAttribute: () => null,
                      setAttribute() {}, removeAttribute() {}},
    querySelector: (selector) => elements[selector.replace("#", "")] || null,
    addEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.innerHeight = WINDOW_HEIGHT;
globalThis.scrollY = SCROLLED;
globalThis.addEventListener = () => {};
globalThis.setTimeout = (fn) => { fn(); return 0; };
// Swallowed rather than run: a real interval keeps node alive after the
// harness has printed its answer, and neither harness is about the clock.
globalThis.setInterval = () => 0;
globalThis.MutationObserver = function () { this.observe = () => {}; };
globalThis.gradioApp = () => globalThis.document;

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());

const read = (id) => elements[id].style.getPropertyValue("--mc-llm-available");
console.log(JSON.stringify({
    chat: read("mc-llm-chat"),
    studio: read("mc-llm-studio"),
    // Whichever element the script chose to publish on, so a measurement made
    // against the wrong one is still measured rather than skipped.
    available: read("mc-llm-chat") || read("mc-llm-studio"),
}));
"""


def published(top: int = 240, window: int = 900, scrolled: int = 0) -> dict:
    harness = (
        HARNESS.replace("SOURCE", SCRIPT.read_text())
        .replace("WORKSPACE_TOP", json.dumps(top))
        .replace("WINDOW_HEIGHT", json.dumps(window))
        .replace("SCROLLED", json.dumps(scrolled))
    )
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def measure(top: int = 240, window: int = 900, scrolled: int = 0) -> str:
    """What the script published, wherever it published it."""
    return published(top, window, scrolled)["available"]


def pixels(value: str) -> int:
    assert value.endswith("px"), value
    return int(value[:-2])


class TestFittingTheWorkspace:
    def test_it_publishes_the_room_below_where_the_workspace_starts(self):
        """Not below where the *tab* starts. The mode selector, the model
        chooser and the status line sit in between, and measuring from the top
        of the tab handed the workspace their height as well -- which is exactly
        how far below the fold the composer ended up."""
        found = published(top=240, window=900)

        assert found["chat"], "the workspace was not measured"
        assert 860 - 240 <= pixels(found["chat"]) <= 900 - 240

    def test_the_measurement_does_not_move_when_the_page_is_scrolled(self):
        """The regression this file exists for. A measurement that grew with
        the scroll offset fed the height it set, and the panel grew on every
        click until it was twice the window with blank space under it."""
        unscrolled = measure(top=240, window=900, scrolled=0)

        for offset in (120, 400, 900, 2000):
            assert measure(top=240, window=900, scrolled=offset) == unscrolled

    def test_it_is_never_taller_than_the_window(self):
        """A measurement that has somehow gone wrong should cost a workspace
        that is a little short, never one that cannot be scrolled back out of."""
        for window in (700, 900, 1400):
            assert pixels(measure(top=0, window=window)) <= window

    def test_a_window_too_short_to_lay_out_in_publishes_nothing(self):
        """Below that, style.css hands the page its scroll bar back rather than
        squeezing the transcript into nothing."""
        assert measure(top=240, window=420) == ""


# --------------------------------------------------------------------------- #
# The transcript's anchoring
# --------------------------------------------------------------------------- #


ANCHOR = """
// A transcript holder with one scrolling child, and a captured
// MutationObserver so a test can say "and then content arrived".
const mutations = [];
const scrollListeners = [];

const bubbles = {
    tagName: "DIV",
    dataset: {},
    style: {setProperty() {}, removeProperty() {}, getPropertyValue: () => ""},
    scrollHeight: 1000,
    clientHeight: 400,
    scrollTop: START_SCROLL_TOP,
    querySelectorAll: () => [],
    querySelector: () => null,
    addEventListener: (kind, fn) => { if (kind === "scroll") scrollListeners.push(fn); },
};

const holder = {
    tagName: "DIV",
    dataset: {},
    offsetParent: {},
    style: {setProperty() {}, removeProperty() {}, getPropertyValue: () => ""},
    // The holder itself does not overflow; the child does. That is the shape
    // Gradio actually renders, and finding the right child is part of what is
    // being tested.
    scrollHeight: 400,
    clientHeight: 400,
    scrollTop: 0,
    querySelectorAll: () => [bubbles],
    querySelector: () => null,
    addEventListener() {},
    getBoundingClientRect: () => ({top: 240, height: 400, bottom: 0, left: 0, right: 0}),
};

globalThis.document = {
    documentElement: {scrollTop: 0, getAttribute: () => null,
                      setAttribute() {}, removeAttribute() {}},
    querySelector: (selector) =>
        (selector === "#mc-llm-chat-transcript" ? holder : null),
    addEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
globalThis.addEventListener = () => {};
globalThis.setTimeout = (fn) => { fn(); return 0; };
// Swallowed rather than run: a real interval keeps node alive after the
// harness has printed its answer, and neither harness is about the clock.
globalThis.setInterval = () => 0;
globalThis.MutationObserver = function (callback) {
    mutations.push(callback);
    this.observe = () => {};
};
globalThis.gradioApp = () => globalThis.document;

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());

// What the reader did, if anything: a scroll to READER_SCROLL_TOP, reported the
// way a browser reports it.
if (READER_SCROLL_TOP !== null) {
    bubbles.scrollTop = READER_SCROLL_TOP;
    scrollListeners.forEach((fn) => fn());
}

// And then a reply arrives: taller content, and — when COLLAPSE is true — the
// scrollTop a re-render leaves behind when the list is empty for an instant.
bubbles.scrollHeight = GROWN_HEIGHT;
if (COLLAPSE) bubbles.scrollTop = 0;
mutations.forEach((fn) => fn());

console.log(JSON.stringify({scrollTop: bubbles.scrollTop, watched: scrollListeners.length}));
"""


def arrival(start: int = 600, reader=None, grown: int = 1600, collapse: bool = False) -> dict:
    """Open a transcript, optionally scroll it, then let a reply land."""
    harness = (
        ANCHOR.replace("SOURCE", SCRIPT.read_text())
        .replace("START_SCROLL_TOP", json.dumps(start))
        .replace("READER_SCROLL_TOP", json.dumps(reader))
        .replace("GROWN_HEIGHT", json.dumps(grown))
        .replace("COLLAPSE", "true" if collapse else "false")
    )
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestAnchoringTheTranscript:
    """At the end, stay at the end. Away from it, stay where you are."""

    def test_it_watches_the_child_that_scrolls_not_the_holder(self):
        """Gradio's transcript does not overflow; the list inside it does."""
        assert arrival()["watched"] == 1

    def test_a_reader_at_the_end_is_carried_to_the_new_end(self):
        """The regression. The reply is 600px taller than the slack, which is
        exactly the case the old check got wrong: it asked whether we were near
        the bottom *after* the reply landed, and the answer was no."""
        landed = arrival(start=600, reader=600, grown=1600)

        assert landed["scrollTop"] == 1600

    def test_within_the_slack_still_counts_as_the_end(self):
        """Gradio's own threshold is 100px and this has to agree with it, or
        the two fight over every reply that arrives near the bottom."""
        assert arrival(start=600, reader=550, grown=1600)["scrollTop"] == 1600

    def test_a_reader_who_has_scrolled_away_is_left_where_they_are(self):
        landed = arrival(start=600, reader=120, grown=1600)

        assert landed["scrollTop"] == 120

    def test_a_re_render_does_not_throw_them_to_the_top(self):
        """A full re-render empties the list for an instant, and scrollTop is
        clamped to a scrollHeight that was briefly zero. Nobody scrolled."""
        landed = arrival(start=600, reader=120, grown=1600, collapse=True)

        assert landed["scrollTop"] == 120

    def test_a_thread_just_opened_shows_its_newest_message(self):
        """No scroll event has happened yet, so there is nothing recorded to
        hold — and the newest message is what somebody opening a chat wants."""
        landed = arrival(start=0, reader=None, grown=1600)

        assert landed["scrollTop"] == 1600


# --------------------------------------------------------------------------- #
# The elapsed-time readout
# --------------------------------------------------------------------------- #
#
# The rule worth executing is not the formatting, it is *where the clock is
# kept*. A run's status line is replaced wholesale every time the run says
# something new -- "Starting…", "Replying…" are three separate elements, not
# one element with three texts -- so a start time stored on the line itself
# resets at each of them, and a reply that took ninety seconds reads as having
# taken five. It is kept on the component instead, which survives the run, and
# is cleared when the run stops being busy. That is what these drive.

CLOCK = """
// A status component whose notice can be replaced under it, as Gradio replaces
// it: a fresh object with the same class, exactly as a repaint produces.
function notice() {
    const children = [];
    return {
        className: "mc-llm-notice mc-llm-busy",
        querySelector(selector) {
            return children.find((child) => "." + child.className === selector) || null;
        },
        appendChild(child) { children.push(child); },
    };
}

const status = {
    dataset: {},
    current: notice(),
    querySelector(selector) {
        return selector === ".mc-llm-busy" ? status.current : null;
    },
};

const now = {value: 0};
globalThis.Date = {now: () => now.value};
globalThis.document = {
    documentElement: {scrollTop: 0, getAttribute: () => null,
                      setAttribute() {}, removeAttribute() {}},
    querySelector: (selector) => (selector === "#mc-llm-chat-status" ? status : null),
    createElement: () => ({className: "", textContent: ""}),
    addEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
globalThis.addEventListener = () => {};
globalThis.setTimeout = (fn) => { fn(); return 0; };
globalThis.MutationObserver = function () { this.observe = () => {}; };
globalThis.gradioApp = () => globalThis.document;

let tick = () => {};
globalThis.setInterval = (fn) => { tick = fn; return 0; };

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());

function readout() {
    // null is the answer for a status line with no busy notice in it at all,
    // which is what "the run has finished" looks like from here.
    if (!status.current) return null;
    const found = status.current.querySelector(".mc-llm-busy-elapsed");
    return found ? found.textContent : null;
}

const written = [];
STEPS.forEach((step) => {
    now.value = step.at;
    if (step.repaint) status.current = notice();
    if (step.idle) status.current = null;
    if (step.busy) status.current = notice();
    tick();
    written.push(readout());
});
console.log(JSON.stringify(written));
"""


def clock(steps: list[dict]) -> list:
    """What the readout said at each step, driving the script's own timer."""
    harness = (CLOCK.replace("SOURCE", SCRIPT.read_text())
               .replace("STEPS", json.dumps(steps)))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTheElapsedReadout:
    def test_a_reply_that_arrives_at_once_is_never_counted_at_all(self):
        """Every request is "starting" for a moment. A readout that flickers
        0s and vanishes is noise where the point was reassurance."""
        assert clock([{"at": 0}, {"at": 1200}]) == ["", ""]

    def test_it_counts_from_the_first_busy_line(self):
        assert clock([{"at": 0}, {"at": 5000}]) == ["", "5s"]

    def test_a_repainted_status_does_not_restart_the_clock(self):
        """The whole bug this is shaped around: "Starting…", "Replying…" and
        every progress line are separate elements, and the run is one run."""
        written = clock([{"at": 0}, {"at": 40000, "repaint": True}])

        assert written[-1] == "40s"

    def test_minutes_are_read_as_minutes(self):
        assert clock([{"at": 0}, {"at": 94500}])[-1] == "1m 34s"

    def test_a_finished_run_stops_the_clock_and_the_next_one_starts_over(self):
        written = clock([
            {"at": 0},
            {"at": 30000},
            {"at": 31000, "idle": True},
            {"at": 90000, "busy": True},
            {"at": 95000},
        ])

        assert written == ["", "30s", None, "", "5s"]


# --------------------------------------------------------------------------- #
# Keeping an opened section in view
# --------------------------------------------------------------------------- #
#
# Conversation's screens are fixed-height scrolling sheets. Open a disclosure
# near the bottom of one — the character editor, the advanced sampling settings
# — and what you opened is below the fold, which is a thing browsers do not fix
# for you, because the click landed on the heading and the heading was already
# visible. The rule is arithmetic, so it is run rather than read: the smallest
# move that brings the opened section into view, and no move at all when it is
# already there.

SHEET = """
function section(top, height) {
    const node = {offsetTop: top, offsetHeight: height, parentElement: null};
    node.parentElement = sheet;
    return node;
}

const sheet = {
    id: "mc-llm-chat-character",
    dataset: {},
    offsetTop: 0,
    clientHeight: VIEW,
    scrollTop: SCROLLED,
    handler: null,
    addEventListener(kind, fn) { if (kind === "click") sheet.handler = fn; },
    querySelector: () => null,
    querySelectorAll: () => [],
};

globalThis.document = {
    documentElement: {scrollTop: 0, getAttribute: () => null,
                      setAttribute() {}, removeAttribute() {}},
    querySelector: (selector) => (selector === "#mc-llm-chat-character" ? sheet : null),
    createElement: () => ({className: "", textContent: ""}),
    addEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
globalThis.addEventListener = () => {};
globalThis.setTimeout = (fn) => { fn(); return 0; };
globalThis.setInterval = () => 0;
globalThis.MutationObserver = function () { this.observe = () => {}; };
globalThis.gradioApp = () => globalThis.document;

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());

// A control inside a section of the sheet, clicked. The section is what the
// script has to find its way back to — the target itself is a button inside it.
const opened = section(TOP, HEIGHT);
sheet.handler({target: {parentElement: opened}});
console.log(JSON.stringify({scrollTop: sheet.scrollTop, wired: sheet.dataset.mcLlmInView}));
"""


def opened(top: int, height: int, view: int = 400, scrolled: int = 0) -> dict:
    """Where the sheet ended up after a section at ``top`` was opened."""
    harness = (SHEET.replace("SOURCE", SCRIPT.read_text())
               .replace("TOP", json.dumps(top))
               .replace("HEIGHT", json.dumps(height))
               .replace("VIEW", json.dumps(view))
               .replace("SCROLLED", json.dumps(scrolled)))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestKeepingAnOpenedSectionInView:
    def test_a_section_already_in_view_is_left_alone(self):
        """Nothing is worse than a panel that jumps when you touch it."""
        assert opened(top=0, height=200, view=400, scrolled=0)["scrollTop"] == 0

    def test_a_section_that_opened_below_the_fold_is_brought_up(self):
        """It is 320 tall starting at 300, so its end is at 620 and the sheet
        shows 400: the smallest move that shows the end of it is 220."""
        assert opened(top=300, height=320, view=400, scrolled=0)["scrollTop"] == 220

    def test_a_section_taller_than_the_sheet_shows_its_beginning(self):
        """Which is where the control you just pressed is."""
        assert opened(top=300, height=900, view=400, scrolled=0)["scrollTop"] == 300

    def test_a_section_scrolled_off_the_top_is_brought_back_down(self):
        assert opened(top=0, height=200, view=400, scrolled=350)["scrollTop"] == 0

    def test_the_sheet_is_wired_once(self):
        assert opened(top=0, height=100)["wired"] == "1"


# --------------------------------------------------------------------------- #
# The regenerate icon on a reply
# --------------------------------------------------------------------------- #
#
# The one feature in this file that reaches into the host's own DOM, because a
# Gradio 4.40 Chatbot draws its own bubbles and there is nowhere in one to put a
# component. What it must do is narrow and worth running rather than reading:
# draw one icon per reply and no more than one, say *which* reply was tapped at
# the moment of the tap, and put the icons away entirely rather than guess when
# the bubbles are a shape it does not recognise.
#
# The ordinal is the part with a real bug behind it. Captured when the icon is
# drawn, it names the wrong message the moment anything above it is deleted --
# and for this particular button, naming the wrong message means rewriting a
# reply the reader did not point at. So it is read from the live transcript when
# the icon is pressed, and that is asserted by moving the bubbles underneath it.

TRANSCRIPT = """
function bubble(name) {
    return {
        name,
        tagName: "DIV",
        dataset: {},
        children: [],
        appendChild(child) { bubble.last = child; this.children.push(child); return child; },
        querySelector: () => null,
        querySelectorAll: () => [],
        addEventListener() {},
    };
}

const bubbles = __BUBBLES__.map(bubble);
const shown = {list: bubbles};

const transcript = {
    id: "mc-llm-chat-transcript",
    tagName: "DIV",
    dataset: {},
    scrollHeight: 0,
    clientHeight: 0,
    scrollTop: 0,
    querySelector: () => null,
    querySelectorAll(selector) {
        return selector === __SELECTOR__ ? shown.list : [];
    },
    addEventListener() {},
};

const field = {tagName: "TEXTAREA", value: "", events: [],
               dispatchEvent(event) { field.events.push(event.type); return true; },
               addEventListener() {}};
const box = {
    id: "mc-llm-chat-regenerate-at",
    tagName: "DIV",
    dataset: {},
    querySelector: (selector) => (selector.indexOf("textarea") >= 0 ? field : null),
    querySelectorAll: () => [],
    addEventListener() {},
};
const presses = [];
const trigger = {
    id: "mc-llm-chat-regenerate-now",
    tagName: "BUTTON",
    dataset: {},
    disabled: false,
    click() { presses.push(field.value); },
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener() {},
};

const elements = {
    "mc-llm-chat-transcript": transcript,
    "mc-llm-chat-regenerate-at": box,
    "mc-llm-chat-regenerate-now": trigger,
};

const created = [];

globalThis.Event = function (kind) { this.type = kind; };
const html = {scrollTop: 0, getAttribute: () => null,
              setAttribute() {}, removeAttribute() {}};
globalThis.document = {
    documentElement: html,
    querySelector: (selector) => elements[selector.replace("#", "")] || null,
    createElement() {
        const node = {
            tagName: "BUTTON",
            className: "",
            textContent: "",
            type: "",
            title: "",
            attributes: {},
            handlers: {},
            setAttribute(name, value) { node.attributes[name] = value; },
            addEventListener(kind, fn) { node.handlers[kind] = fn; },
        };
        created.push(node);
        return node;
    },
    addEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
globalThis.addEventListener = () => {};
globalThis.setTimeout = (fn) => { fn(); return 0; };
globalThis.setInterval = () => 0;
globalThis.MutationObserver = function () { this.observe = () => {}; };
globalThis.gradioApp = () => globalThis.document;

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

__BREAK__
loaded.forEach((fn) => fn());

const stopped = [];
function tap(which) {
    const node = bubbles[which].children[0];
    if (!node || !node.handlers.click) return false;
    node.handlers.click({
        preventDefault() { stopped.push("default"); },
        stopPropagation() { stopped.push("propagation"); },
    });
    return true;
}

__REHEARSAL__

console.log(JSON.stringify({
    drawn: bubbles.map((one) => one.children.length),
    label: created.length ? created[0].textContent : "",
    title: created.length ? created[0].attributes["aria-label"] : "",
    classes: created.map((one) => one.className),
    presses,
    events: field.events,
    stopped,
}));
"""


def transcript(bubbles: int = 3, rehearsal: str = "", selector: str = '[data-testid="bot"]',
               break_first: bool = False):
    """Run the script against a transcript of ``bubbles`` replies."""
    broken = ("html.getAttribute = () => { throw new Error('no'); };"
              if break_first else "")
    harness = (TRANSCRIPT.replace("SOURCE", SCRIPT.read_text())
               .replace("__BREAK__", broken)
               .replace("__REHEARSAL__", rehearsal)
               .replace("__BUBBLES__", json.dumps([f"reply {index}" for index in range(bubbles)]))
               .replace("__SELECTOR__", json.dumps(selector)))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTheRegenerateIcon:
    def test_every_reply_gets_one(self):
        assert transcript(bubbles=3)["drawn"] == [1, 1, 1]

    def test_it_is_the_reload_glyph_and_it_says_what_it_does(self):
        """A bare icon with no accessible name is a button a screen reader
        reads out as "button"."""
        drawn = transcript(bubbles=1)

        assert drawn["label"] == "↻"
        assert drawn["title"] == "Regenerate this reply"
        assert drawn["classes"] == ["mc-llm-again"]

    def test_wiring_twice_does_not_draw_it_twice(self):
        """This runs after every update the host makes, and a transcript that
        grew a second icon on every streamed chunk would be unreadable within a
        sentence."""
        assert transcript(bubbles=2, rehearsal="loaded.forEach((fn) => fn());"
                                               "loaded.forEach((fn) => fn());")["drawn"] == \
            [1, 1]

    def test_tapping_one_names_which_reply_it_is(self):
        drawn = transcript(bubbles=3, rehearsal="tap(2);")

        assert drawn["presses"] == ["2"]
        # Written *and* announced: Gradio learns a value from the event, and a
        # box written to without one is a box the server still reads as empty.
        assert drawn["events"] == ["input"]

    def test_the_tap_never_reaches_the_bubble_under_it(self):
        """The bubble is wired to the Chatbot's own select event, and a click
        that reached it would open the action sheet over the reply that is
        about to start arriving."""
        assert transcript(bubbles=2, rehearsal="tap(0);")["stopped"] == \
            ["default", "propagation"]

    def test_which_reply_it_is_is_read_when_it_is_tapped(self):
        """The bug this is here for: an ordinal captured when the icon was
        drawn names the wrong message as soon as anything above it goes, and
        for this button that means rewriting a reply nobody pointed at."""
        drawn = transcript(
            bubbles=3,
            # The first reply is deleted out of the thread, the way deleting a
            # message renumbers every bubble below it, and *then* the icon on
            # what is now the second reply is tapped.
            rehearsal="const kept = [bubbles[1], bubbles[2]];"
                      "shown.list = kept;"
                      "tap(2);")

        assert drawn["presses"] == ["1"]

    def test_another_feature_throwing_does_not_stop_it_being_drawn(self):
        """Each concern in this file is wired on its own. A single try around
        all of them meant the first one throwing took the six after it down with
        it, and the failure was silent -- the icons simply were not there."""
        assert transcript(bubbles=2, break_first=True)["drawn"] == [1, 1]

    def test_bubbles_it_cannot_recognise_cost_an_icon_and_nothing_else(self):
        """A theme that replaces Gradio's DOM wholesale. Regenerate is still on
        the sheet a tap on the bubble opens, which is where it was before this
        existed -- so the right behaviour is to draw nothing and carry on, not
        to guess at an element."""
        drawn = transcript(bubbles=3, selector=".something-a-theme-invented")

        assert drawn["drawn"] == [0, 0, 0]
        assert drawn["presses"] == []


# --------------------------------------------------------------------------- #
# The WebUI's footer
# --------------------------------------------------------------------------- #
#
# The conversation workspace is built to fit the window -- the page does not
# scroll, the transcript does -- and the footer defeats that from outside
# anything this extension lays out: it sits below the fold and takes real space,
# so the page scrolls by exactly the height of a row of links. Nothing is
# removed and no style is written on the element; an attribute goes on the root
# and style.css does the hiding, which is what makes turning the setting off put
# the footer straight back rather than at the next reload.

FOOTER = """
const html = {
    attributes: {},
    writes: 0,
    getAttribute: (name) => (name in html.attributes ? html.attributes[name] : null),
    setAttribute(name, value) { html.attributes[name] = value; html.writes += 1; },
    removeAttribute(name) { delete html.attributes[name]; html.writes += 1; },
};

globalThis.document = {
    documentElement: html,
    querySelector: () => null,
    createElement: () => ({className: "", textContent: "", setAttribute() {},
                          addEventListener() {}}),
    addEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
globalThis.addEventListener = () => {};
globalThis.setTimeout = (fn) => { fn(); return 0; };
globalThis.setInterval = () => 0;
globalThis.MutationObserver = function () { this.observe = () => {}; };
globalThis.gradioApp = () => globalThis.document;
__OPTS__

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());
const first = html.getAttribute("data-mc-footer");
const settled = html.writes;
__AFTER__

console.log(JSON.stringify({
    first,
    footer: html.getAttribute("data-mc-footer"),
    writes: html.writes,
    settled,
}));
"""


def footer(options="{}", after=""):
    """What the script put on the root, given a settings object."""
    declared = "globalThis.opts = OPTIONS;" if options is not None else ""
    harness = (FOOTER.replace("SOURCE", SCRIPT.read_text())
               .replace("__OPTS__", declared.replace("OPTIONS", options or "{}"))
               .replace("__AFTER__", after))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestHidingTheFooter:
    def test_it_is_hidden_by_default(self):
        """The setting defaults on, and the fallback here has to match it or the
        footer flickers back for the moment before the options load."""
        assert footer('{"model_chain_hide_footer": true}')["footer"] == "hidden"

    def test_the_setting_turning_it_off_puts_the_footer_back(self):
        assert footer('{"model_chain_hide_footer": false}')["footer"] is None

    def test_no_settings_at_all_still_hides_it(self):
        """``opts`` does not exist until the host has loaded the settings, and
        an exception reading it would take the rest of this file's wiring down
        with it."""
        assert footer(None)["footer"] == "hidden"
        assert footer("{}")["footer"] == "hidden"

    def test_it_writes_the_attribute_once_and_not_on_every_update(self):
        """This runs after every update the host makes, and setting an attribute
        invalidates style whether or not the value moved."""
        settled = footer('{"model_chain_hide_footer": true}',
                         after="loaded.forEach((fn) => fn());"
                               "loaded.forEach((fn) => fn());")

        assert settled["settled"] == 1
        assert settled["writes"] == 1

    def test_turning_it_off_while_the_page_is_open_takes_effect(self):
        """No reload: the rule is CSS keyed on this attribute, and the value is
        read live rather than baked in at startup."""
        changed = footer('{"model_chain_hide_footer": true}',
                         after='opts.model_chain_hide_footer = false;'
                               'loaded.forEach((fn) => fn());')

        assert changed["first"] == "hidden"
        assert changed["footer"] is None
