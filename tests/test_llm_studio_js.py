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

// The page, as tall as whatever the workspace was given plus whatever sits
// under it -- the container's own bottom padding, and anything the host puts
// after the tab. TRAILING is that strip, and it is what makes the page scroll
// when the workspace has been sized only against the window.
const page = {
    scrollTop: SCROLLED,
    clientHeight: WINDOW_HEIGHT,
    get scrollHeight() {
        const published = elements["mc-llm-chat"].style.getPropertyValue("--mc-llm-available");
        const height = published ? parseInt(published, 10) : 400;
        return WORKSPACE_TOP + height + TRAILING;
    },
    getAttribute: () => null,
    setAttribute() {},
    removeAttribute() {},
};

globalThis.document = {
    documentElement: page,
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


def published(top: int = 240, window: int = 900, scrolled: int = 0,
              trailing: int = 0) -> dict:
    harness = (
        HARNESS.replace("SOURCE", SCRIPT.read_text())
        .replace("WORKSPACE_TOP", json.dumps(top))
        .replace("WINDOW_HEIGHT", json.dumps(window))
        .replace("TRAILING", json.dumps(trailing))
        .replace("SCROLLED", json.dumps(scrolled))
    )
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def measure(top: int = 240, window: int = 900, scrolled: int = 0,
            trailing: int = 0) -> str:
    """What the script published, wherever it published it."""
    return published(top, window, scrolled, trailing)["available"]


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

    def test_it_does_not_give_back_room_for_overflow_that_is_not_its_own(self):
        """There was a pass here that read the page's scrollHeight and shrank
        the workspace by whatever still hung below the fold, to find a strip
        left behind by a hidden footer without naming it.

        It found the wrong thing. The page's overflow is not all this
        extension's -- the host's layout has its own -- so what the workspace
        gave back was somebody else's, and what appeared was a band of empty
        space *above* the strip, with the page still scrolling. Reported as
        "your fix added space instead of removed it". A footer is a thing that
        can be named, and style.css names it."""
        for trailing in (0, 48, 5000):
            assert pixels(measure(top=240, window=900, trailing=trailing)) == 900 - 240 - 16

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
const listeners = {};

const bubbles = {
    tagName: "DIV",
    dataset: {},
    style: {setProperty() {}, removeProperty() {}, getPropertyValue: () => ""},
    scrollHeight: 1000,
    clientHeight: 400,
    scrollTop: START_SCROLL_TOP,
    querySelectorAll: () => [],
    querySelector: () => null,
    addEventListener: (kind, fn) => {
        if (kind === "scroll") scrollListeners.push(fn);
        (listeners[kind] = listeners[kind] || []).push(fn);
    },
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

// What the reader did, if anything, in order: a number is a scroll to that
// position, reported the way a browser reports it; {wheel: dy} is a wheel
// turned, which a browser reports before it scrolls anything.
const steps = READER_SCROLL_TOP === null ? []
    : (Array.isArray(READER_SCROLL_TOP) ? READER_SCROLL_TOP : [READER_SCROLL_TOP]);
for (const step of steps) {
    if (typeof step === "number") {
        bubbles.scrollTop = step;
        scrollListeners.forEach((fn) => fn());
    } else if (step && typeof step.wheel === "number") {
        (listeners.wheel || []).forEach((fn) => fn({deltaY: step.wheel, deltaMode: 0}));
    }
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

    def test_a_little_away_from_the_end_is_away_from_it(self):
        """The rule as asked for: "if i scroll away from the bottom even just a
        little bit, i become undocked". There is no slack any more -- the old
        100 px agreed with Gradio's own Chatbot, and both put a reader who had
        scrolled up less than that back at the end with every chunk."""
        assert arrival(start=600, reader=550, grown=1600)["scrollTop"] == 550

    def test_a_wheel_upward_undocks_before_anything_has_scrolled(self):
        """A chunk can land between the wheel and the scroll it causes."""
        assert arrival(start=600, reader=[{"wheel": -3}], grown=1600)["scrollTop"] == 600

    def test_a_wheel_downward_does_not_undock(self):
        assert arrival(start=600, reader=[{"wheel": 100}], grown=1600)["scrollTop"] == 1600

    def test_reaching_the_end_again_docks_again(self):
        landed = arrival(start=600, reader=[{"wheel": -100}, 300, 590, 600], grown=1600)

        assert landed["scrollTop"] == 1600

    def test_most_of_the_way_down_is_not_the_end(self):
        assert arrival(start=600, reader=[300, 590], grown=1600)["scrollTop"] == 590

    def test_a_wheel_downward_at_the_end_docks_without_a_scroll(self):
        """At the very end nothing can scroll, so no scroll event comes."""
        landed = arrival(start=600, reader=[{"wheel": -100}, {"wheel": 100}], grown=1600)

        assert landed["scrollTop"] == 1600

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
# The row of actions in every bubble
# --------------------------------------------------------------------------- #
#
# The one feature in this file that reaches into the host's own DOM, because a
# Gradio 4.40 Chatbot draws its own bubbles and there is nowhere in one to put a
# component. What it must do is worth running rather than reading: draw one row
# per message and no more than one, show it on a tap and put it away on the
# next, say *which* message a button is on at the moment of the press, and put
# the rows away entirely rather than guess when the bubbles are a shape it does
# not recognise. Copy, Send to VibeVoice and Play never reach Python, so they
# are driven here against a clipboard and a Voice Box that record what they
# were asked.
#
# The ordinal is the part with a real bug behind it. Captured when the row is
# drawn, it names the wrong message the moment anything above it is deleted --
# and for a delete button, naming the wrong message means deleting a message
# the reader did not point at. So it is read from the live transcript when the
# button is pressed, and that is asserted by moving the bubbles underneath it.

ROW = """
// A small DOM with the parts the row reads and writes: classes, attributes,
// datasets, a tree, selectors of the shapes the script uses, cloneNode for the
// copy, closest for the tap, and events that bubble.
function matches(node, selector) {
    return selector.split(",").some((part) => {
        const piece = part.trim();
        if (!piece) return false;
        let rest = piece;
        if (rest.startsWith(":scope")) rest = rest.slice(6).trim();
        const tag = (rest.match(/^[a-z]+/i) || [""])[0];
        if (tag && node.tagName.toLowerCase() !== tag.toLowerCase()) return false;
        rest = rest.slice(tag.length);
        const classes = rest.match(/\\.[\\w-]+/g) || [];
        if (!classes.every((c) => node.classList.contains(c.slice(1)))) return false;
        const attrs = rest.match(/\\[[^\\]]+\\]/g) || [];
        return attrs.every((a) => {
            const [, name, value] = a.match(/^\\[([\\w-]+)(?:="([^"]*)")?\\]$/) || [];
            if (!name) return false;
            const held = node.getAttribute(name);
            return value === undefined ? held !== null : held === value;
        });
    });
}

class El {
    constructor(tag) {
        this.tagName = tag.toUpperCase();
        this.attributes = {};
        this.dataset = {};
        this.children = [];
        this.parentNode = null;
        this.handlers = {};
        this.style = {};
        this.hidden = false;
        this.disabled = false;
        this.paused = true;
        this._text = "";
        this.classList = {
            contains: (c) => this.classes().includes(c),
            add: (c) => { if (!this.classes().includes(c)) this.className = this.classes().concat(c).join(" "); },
            remove: (c) => { this.className = this.classes().filter((x) => x !== c).join(" "); },
            toggle: (c, on) => { if (on) this.classList.add(c); else this.classList.remove(c); },
        };
        this.className = "";
    }
    classes() { return this.className.split(/\\s+/).filter(Boolean); }
    get textContent() {
        return this.children.length ? this.children.map((c) => c.textContent).join("") : this._text;
    }
    set textContent(value) { this._text = String(value); this.children = []; }
    get innerText() { return this.textContent; }
    set innerHTML(value) { this._text = ""; this.children = []; this.html = value; }
    get innerHTML() { return this.html || this.textContent; }
    getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    hasAttribute(name) { return name in this.attributes; }
    appendChild(child) {
        if (child.parentNode) child.parentNode.removeChild(child);
        child.parentNode = this;
        this.children.push(child);
        return child;
    }
    removeChild(child) {
        this.children = this.children.filter((c) => c !== child);
        child.parentNode = null;
        return child;
    }
    contains(node) {
        for (let at = node; at; at = at.parentNode) if (at === this) return true;
        return false;
    }
    closest(selector) {
        for (let at = this; at; at = at.parentNode) if (at.tagName && matches(at, selector)) return at;
        return null;
    }
    all() { return this.children.flatMap((c) => [c].concat(c.all())); }
    querySelectorAll(selector) { return this.all().filter((n) => matches(n, selector)); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    cloneNode(deep) {
        const copy = new El(this.tagName);
        copy.className = this.className;
        copy.attributes = Object.assign({}, this.attributes);
        copy.dataset = Object.assign({}, this.dataset);
        copy._text = this._text;
        if (deep) this.children.forEach((c) => copy.appendChild(c.cloneNode(true)));
        return copy;
    }
    addEventListener(kind, fn) { (this.handlers[kind] = this.handlers[kind] || []).push(fn); }
    dispatchEvent(event) {
        event.target = event.target || this;
        if (!event.stopPropagation) event.stopPropagation = () => { event.stopped = true; };
        if (!event.preventDefault) event.preventDefault = () => { event.defaulted = true; };
        for (let at = this; at && !event.stopped; at = at.parentNode) {
            (at.handlers[event.type] || []).forEach((fn) => fn.call(at, event));
            if (!event.bubbles) break;
        }
        return true;
    }
    click() {
        this.clicks = (this.clicks || 0) + 1;
        this.dispatchEvent({type: "click", bubbles: true, preventDefault() {}, stopPropagation() {}});
    }
    focus() {}
    select() {}
    play() { this.paused = false; this.played = (this.played || 0) + 1; this.dispatchEvent({type: "play"}); return Promise.resolve(); }
    pause() { this.paused = true; this.dispatchEvent({type: "pause"}); }
    getBoundingClientRect() { return {top: 0, bottom: 0, height: 0, left: 0, right: 0}; }
}

const byId = {};
function element(tag, id) {
    const node = new El(tag);
    if (id) { node.setAttribute("id", id); byId[id] = node; }
    return node;
}

// The transcript, each message with the marker Python puts first in it. Two
// shapes: "flat" is one div with the testid, the simplest thing a theme could
// draw; "gradio" is what Gradio 4.40's Chatbot really draws (its
// ChatBot.svelte): `div.message-row.bot-row > div.flex-wrap.bot >
// div.message.bot > button[data-testid="bot"]`, the words inside the button,
// which selects the message on click and on Enter. `bubbles[i]` is the bubble
// the script should treat as the message: the div in both shapes.
const SHAPE = __SHAPE__;
const holder = element("div", "mc-llm-chat-transcript");
holder.scrollHeight = 0; holder.clientHeight = 0; holder.scrollTop = 0;
const bubbles = [];
const hostButtons = [];
const selects = [];
function bubble(role, text, meta) {
    const kind = role === "user" ? "user" : "bot";
    const marker = new El("span");
    marker.className = "mc-llm-meta";
    marker.setAttribute("data-mc-role", role);
    marker.setAttribute("data-mc-versions", String((meta && meta.versions) || 1));
    marker.setAttribute("data-mc-active", String((meta && meta.active) || 0));
    marker.setAttribute("data-mc-key", (meta && meta.key) || ("thread-1:" + text.replace(/\\W/g, "")));
    marker.setAttribute("hidden", "");
    const words = new El("p");
    words.textContent = text;
    if (SHAPE === "gradio") {
        const row = new El("div");
        row.className = "message-row bubble " + kind + "-row";
        const flex = new El("div");
        flex.className = "flex-wrap " + kind;
        const node = new El("div");
        node.className = "message " + kind;
        const button = new El("button");
        button.setAttribute("data-testid", kind);
        button.addEventListener("click", () => selects.push(kind));   // Gradio's handle_select
        button.appendChild(marker);
        button.appendChild(words);
        node.appendChild(button);
        flex.appendChild(node);
        row.appendChild(flex);
        holder.appendChild(row);
        bubbles.push(node);
        hostButtons.push(button);
        return node;
    }
    // "button": a theme that kept Gradio's button and dropped its wrapper, so
    // the bubble is the button itself and the row has to live inside it.
    const node = new El(SHAPE === "button" ? "button" : "div");
    node.setAttribute("data-testid", kind);
    node.className = "message " + kind;
    node.appendChild(marker);
    node.appendChild(words);
    holder.appendChild(node);
    bubbles.push(node);
    if (SHAPE === "button") hostButtons.push(node);
    return node;
}
__BUBBLES__

// The hidden box and buttons Python builds.
const box = element("div", "mc-llm-chat-action-at");
const field = new El("textarea");
field.value = "";
field.events = [];
field.dispatchEvent = (event) => { field.events.push(event.type); return true; };
box.appendChild(field);
const presses = [];
["regenerate", "edit", "continue", "resend", "branch", "delete", "delete-from", "back",
 "forward", "drop"].forEach((name) => {
    const button = element("button", "mc-llm-chat-" + name + "-now");
    button.click = () => presses.push([name, field.value]);
});

const clipboard = [];
Object.defineProperty(globalThis, "navigator", {
    value: {clipboard: {writeText(text) { clipboard.push(text); return Promise.resolve(); }}},
    configurable: true,
});

// The Voice Box's bridge, as the scenario sets it.
const voiceBox = {
    asked: [], urls: [], jobs: {}, outputs: [],
    canRender(text) { return voiceBox.refusal || ""; },
    renderText(text, options) {
        voiceBox.asked.push({text, options});
        if (voiceBox.refuseSend) return Promise.reject(new Error(voiceBox.refuseSend));
        const id = "j" + voiceBox.asked.length;
        voiceBox.jobs[id] = {id, phase: "queued", live: true, output_id: ""};
        return Promise.resolve(voiceBox.jobs[id]);
    },
    jobById(id) { return voiceBox.jobs[id] || null; },
    outputsFor(prefix) {
        voiceBox.prefixes = (voiceBox.prefixes || []).concat(prefix);
        return Promise.resolve(voiceBox.outputs.filter((o) => o.render.origin.key.startsWith(prefix)));
    },
    outputAudioUrl(id) { voiceBox.urls.push(id); return Promise.resolve("blob:test-" + id); },
};
globalThis.mcVoiceBox = __VOICE_BOX__ ? voiceBox : undefined;

const focusEvents = [];
const documentListeners = {};
globalThis.Event = function (kind, init) { this.type = kind; this.bubbles = !!(init && init.bubbles); };
globalThis.CustomEvent = function (kind, init) { this.type = kind; this.detail = init && init.detail; };
const html = {scrollTop: 0, getAttribute: () => null, setAttribute() {}, removeAttribute() {}};
globalThis.document = {
    documentElement: html,
    body: new El("body"),
    querySelector: (selector) => byId[selector.replace("#", "")] || null,
    createElement: (tag) => new El(tag),
    addEventListener(kind, fn) { (documentListeners[kind] = documentListeners[kind] || []).push(fn); },
    dispatchEvent(event) {
        if (event.type === "mc:audio-focus") focusEvents.push(event.detail);
        (documentListeners[event.type] || []).forEach((fn) => fn(event));
        return true;
    },
    execCommand: () => false,
    readyState: "complete",
};
globalThis.window = globalThis;
const windowListeners = {};
globalThis.addEventListener = (kind, fn) => { (windowListeners[kind] = windowListeners[kind] || []).push(fn); };
globalThis.getSelection = () => ({isCollapsed: true});
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
// A timer with no delay runs at once (the nomination's next tick); one with
// a delay waits for the scenario to fire it, so a note can be read before it
// goes away.
const timeouts = [];
globalThis.setTimeout = (fn, ms) => {
    if (!ms) { fn(); return 0; }
    timeouts.push({fn, ms});
    return timeouts.length;
};
globalThis.clearTimeout = (id) => { if (timeouts[id - 1]) timeouts[id - 1].cleared = true; };
const intervals = [];
globalThis.setInterval = (fn, ms) => { intervals.push({fn, ms}); return intervals.length; };
globalThis.clearInterval = (id) => { if (intervals[id - 1]) intervals[id - 1].cleared = true; };
globalThis.MutationObserver = function () { this.observe = () => {}; };
globalThis.gradioApp = () => globalThis.document;

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

// What the page holds before the script first runs.
__SETUP__

loaded.forEach((fn) => fn());

// The scenario's hands.
function rowOf(node) { return node.children.filter((c) => c.classList.contains("mc-llm-message-actions"))[0] || null; }
function buttons(node) { return rowOf(node) ? rowOf(node).querySelectorAll("button") : []; }
function actionsOf(node) { return buttons(node).map((b) => b.getAttribute("data-action")); }
function buttonOf(node, action) { return buttons(node).filter((b) => b.getAttribute("data-action") === action)[0]; }
function tap(node) {
    const words = node.querySelector("p") || node;
    words.dispatchEvent({type: "click", bubbles: true, target: words, preventDefault() {}, stopPropagation() {}});
}
function pressDown(node) {
    (windowListeners.pointerdown || []).forEach((fn) => fn({target: node}));
}
function pressAction(node, action) {
    const button = buttonOf(node, action);
    button.dispatchEvent({type: "click", bubbles: true, target: button});
}
function clickOn(target) {
    target.dispatchEvent({type: "click", bubbles: true, target});
}
function noteOf(node) { const n = rowOf(node).querySelector(".mc-llm-message-actions-note"); return n && !n.hidden ? n.textContent : ""; }
async function settle() { await new Promise((r) => process.nextTick(r)); await new Promise((r) => process.nextTick(r)); }
function look() { intervals.filter((i) => !i.cleared).forEach((i) => i.fn()); }
function fire() { timeouts.splice(0).filter((t) => !t.cleared).forEach((t) => t.fn()); }
function rewire() { loaded.forEach((fn) => fn()); }
function state(node) {
    return {open: rowOf(node) ? !rowOf(node).hidden : null, revealed: node.classList.contains("mc-llm-revealed"),
            expanded: node.getAttribute("aria-expanded"), vibe: node.getAttribute("data-mc-vibe"),
            play: buttonOf(node, "play") ? {disabled: buttonOf(node, "play").disabled, glyph: buttonOf(node, "play").textContent} : null,
            send: buttonOf(node, "vibe") ? {disabled: buttonOf(node, "vibe").disabled, title: buttonOf(node, "vibe").title} : null};
}

const out = {};
await (async function () {
__REHEARSAL__
})();

console.log(JSON.stringify(Object.assign({
    rows: bubbles.map((b) => node_actions(b)),
    presses,
    events: field.events,
    clipboard,
    asked: voiceBox.asked,
    urls: voiceBox.urls,
    prefixes: voiceBox.prefixes || [],
    focus: focusEvents,
    selects,
}, out)));
function node_actions(b) { return actionsOf(b); }
"""


def row(bubbles=(("assistant", "reply 0"),), rehearsal: str = "", voice_box: bool = True,
        setup: str = "", shape: str = "flat"):
    """Run the script against a transcript of ``bubbles``: ``(role, text[, meta])``.
    ``setup`` runs before the script first wires the page; ``rehearsal`` after.
    ``shape`` is "flat" (a div with the testid), "gradio" (Gradio 4.40's own
    markup, the words inside a button) or "button" (the testid on a bare button)."""
    made = []
    for one in bubbles:
        role, text = one[0], one[1]
        meta = json.dumps(one[2] if len(one) > 2 else {})
        made.append(f"bubble({json.dumps(role)}, {json.dumps(text)}, {meta});")
    harness = (ROW.replace("SOURCE", SCRIPT.read_text())
               .replace("__BUBBLES__", "\n".join(made))
               .replace("__REHEARSAL__", rehearsal)
               .replace("__SETUP__", setup)
               .replace("__SHAPE__", json.dumps(shape))
               .replace("__VOICE_BOX__", "true" if voice_box else "false"))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


REPLY_ROW = ["edit", "regenerate", "continue", "branch", "copy", "vibe", "play", "delete",
             "delete_from"]
PROMPT_ROW = ["edit", "resend", "branch", "copy", "vibe", "play", "delete", "delete_from"]


class TestTheActionRow:
    def test_every_message_gets_one_row_in_the_order_that_reads(self):
        """Edit and the ways of asking again, then Branch; Copy and the two
        VibeVoice buttons; the two that lose something last. A prompt of yours
        has Send again where a reply has Regenerate and Continue."""
        found = row([("assistant", "reply 0"), ("user", "ask 1"), ("assistant", "reply 1")])

        assert found["rows"] == [REPLY_ROW, PROMPT_ROW, REPLY_ROW]

    def test_wiring_twice_does_not_draw_it_twice(self):
        """This runs after every update the host makes, and a transcript that
        grew a second row on every streamed chunk would be unreadable within a
        sentence."""
        found = row([("assistant", "reply 0"), ("user", "ask 1")], "rewire(); rewire();"
                    "out.rows_each = bubbles.map((b) => b.children.filter((c) => c.classList.contains('mc-llm-message-actions')).length);")

        assert found["rows_each"] == [1, 1]

    def test_the_row_is_hidden_until_the_message_is_tapped_and_the_next_tap_puts_it_away(self):
        found = row([("assistant", "reply 0"), ("user", "ask 1")], """
            out.before = state(bubbles[0]);
            tap(bubbles[0]);
            out.opened = state(bubbles[0]);
            tap(bubbles[0]);
            out.closed = state(bubbles[0]);
        """)

        assert found["before"]["open"] is False and found["before"]["expanded"] == "false"
        assert found["opened"]["open"] is True and found["opened"]["revealed"] is True
        assert found["opened"]["expanded"] == "true"
        assert found["closed"]["open"] is False and found["closed"]["revealed"] is False

    def test_tapping_another_message_moves_the_row_and_a_press_elsewhere_puts_it_away(self):
        found = row([("assistant", "reply 0"), ("user", "ask 1")], """
            tap(bubbles[0]);
            tap(bubbles[1]);
            out.moved = [state(bubbles[0]).open, state(bubbles[1]).open];
            pressDown(bubbles[1].querySelector("p"));
            out.inside = state(bubbles[1]).open;
            pressDown(holder);
            out.outside = state(bubbles[1]).open;
        """)

        assert found["moved"] == [False, True]
        assert found["inside"] is True, "a press on the open message itself is the tap's to handle"
        assert found["outside"] is False

    def test_the_row_survives_a_redraw_while_it_is_open(self):
        """A reply arriving token by token redraws the transcript once a
        frame; a row that closed whenever a word arrived could not be used,
        and a row *rebuilt* on every word would lose what it was saying."""
        found = row([("assistant", "reply 0")], """
            tap(bubbles[0]);
            pressAction(bubbles[0], "vibe");
            const bar = rowOf(bubbles[0]);
            rewire();
            out.after = state(bubbles[0]);
            out.note = noteOf(bubbles[0]);
            out.same = rowOf(bubbles[0]) === bar;
        """)

        assert found["after"]["open"] is True
        assert found["note"] == "Sent to VibeVoice…" and found["same"] is True

    def test_a_row_is_drawn_again_when_the_marker_under_it_changes(self):
        """A version paged, a reply still arriving: the row reads the marker
        when it is built, so a changed marker is a rebuilt row -- once."""
        found = row([("assistant", "reply 0")], """
            const marker = bubbles[0].querySelector(".mc-llm-meta");
            marker.setAttribute("data-mc-versions", "3");
            marker.setAttribute("data-mc-active", "1");
            rewire();
            out.pager = rowOf(bubbles[0]).querySelector(".mc-llm-message-actions-pager").textContent;
            out.rows = bubbles[0].children.filter((c) => c.classList.contains("mc-llm-message-actions")).length;
        """)

        assert found["pager"] == "2/3" and found["rows"] == 1

    def test_escape_puts_it_away_from_anywhere(self):
        found = row([("assistant", "reply 0")], """
            tap(bubbles[0]);
            (windowListeners.keydown || []).forEach((fn) => fn({key: "Escape", target: holder}));
            out.after = state(bubbles[0]).open;
        """)

        assert found["after"] is False

    def test_the_keyboard_opens_it_too(self):
        found = row([("assistant", "reply 0")], """
            out.tabindex = bubbles[0].getAttribute("tabindex");
            bubbles[0].dispatchEvent({type: "keydown", key: "Enter", bubbles: true, target: bubbles[0], preventDefault() {}});
            out.opened = state(bubbles[0]).open;
            bubbles[0].dispatchEvent({type: "keydown", key: " ", bubbles: true, target: bubbles[0], preventDefault() {}});
            out.closed = state(bubbles[0]).open;
        """)

        assert found["tabindex"] == "0"
        assert found["opened"] is True and found["closed"] is False

    def test_a_press_names_the_message_and_presses_the_actions_button(self):
        """Written *and* announced: Gradio learns a value from the event, and a
        box written to without one is a box the server still reads as empty."""
        found = row([("assistant", "reply 0"), ("user", "ask 1"), ("assistant", "reply 1")], """
            pressAction(bubbles[2], "regenerate");
            pressAction(bubbles[1], "resend");
            pressAction(bubbles[2], "delete_from");
            pressAction(bubbles[0], "edit");
        """)

        assert found["presses"] == [["regenerate", "assistant:1"], ["resend", "user:0"],
                                    ["delete-from", "assistant:1"], ["edit", "assistant:0"]]
        assert found["events"] == ["input"] * 4

    def test_a_pressed_row_goes_away(self):
        """Pressed is chosen: the transcript that comes back is drawn afresh."""
        found = row([("assistant", "reply 0")], """
            tap(bubbles[0]);
            pressAction(bubbles[0], "branch");
            out.after = state(bubbles[0]).open;
        """)

        assert found["after"] is False

    def test_which_message_it_is_is_read_when_it_is_pressed(self):
        """The bug this is here for: an ordinal captured when the row was
        drawn names the wrong message as soon as anything above it goes, and
        for a delete button that means deleting a message nobody pointed at."""
        found = row([("assistant", "reply 0"), ("assistant", "reply 1"), ("assistant", "reply 2")], """
            holder.removeChild(bubbles[0]);
            pressAction(bubbles[2], "delete");
        """)

        assert found["presses"] == [["delete", "assistant:1"]]

    def test_the_version_pager_appears_once_there_is_more_than_one(self):
        found = row([("assistant", "reply 0", {"versions": 3, "active": 2}),
                     ("assistant", "reply 1")], """
            const cluster = rowOf(bubbles[0]).querySelector(".mc-llm-message-actions-versions");
            out.pager = cluster.querySelector(".mc-llm-message-actions-pager").textContent;
            out.cluster = cluster.querySelectorAll("button").map((b) => [b.getAttribute("data-action"), b.disabled]);
            out.plain = !!rowOf(bubbles[1]).querySelector(".mc-llm-message-actions-versions");
            pressAction(bubbles[0], "back");
            pressAction(bubbles[0], "drop");
        """)

        assert found["pager"] == "3/3"
        assert found["cluster"] == [["back", False], ["forward", True], ["drop", False]]
        assert found["plain"] is False
        assert found["presses"] == [["back", "assistant:0"], ["drop", "assistant:0"]]

    def test_every_button_has_a_name_a_screen_reader_can_say(self):
        found = row([("assistant", "reply 0", {"versions": 2})], """
            out.names = buttons(bubbles[0]).map((b) => [b.getAttribute("aria-label"), b.title]);
        """)

        for label, title in found["names"]:
            assert label and label == title

    def test_bubbles_it_cannot_recognise_cost_the_row_and_nothing_else(self):
        """A theme that replaces Gradio's DOM wholesale. The right behaviour is
        to draw nothing and carry on, not to guess at an element."""
        found = row([("assistant", "reply 0")], """
            bubbles[0].removeAttribute("data-testid");
            bubbles[0].className = "something-a-theme-invented";
            bubbles[0].children = bubbles[0].children.filter((c) => !c.classList.contains("mc-llm-message-actions"));
            delete bubbles[0].dataset.mcLlmActions;
            rewire();
            out.drawn = bubbles[0].children.filter((c) => c.classList.contains("mc-llm-message-actions")).length;
        """)

        assert found["drawn"] == 0


class TestGradiosOwnBubble:
    """Gradio 4.40 draws a message as `div.message > button[data-testid]`, the
    words inside the button (its ChatBot.svelte). The first build read the
    button as the bubble and refused every tap "inside a button", so on the
    real page a tap did nothing at all, while the flat stand-in passed. These
    run the script against that shape."""

    def test_the_row_sits_beside_the_hosts_button_inside_the_bubble(self):
        found = row([("assistant", "reply 0"), ("user", "ask 1")], """
            out.in_button = hostButtons.map((b) => b.querySelectorAll(".mc-llm-message-actions").length);
            out.in_bubble = bubbles.map((b) => b.children.filter((c) => c.classList.contains("mc-llm-message-actions")).length);
            out.tabindex = bubbles.map((b) => b.getAttribute("tabindex"));
            out.role = bubbles.map((b) => b.getAttribute("data-mc-role"));
        """, shape="gradio")

        assert found["rows"] == [REPLY_ROW, PROMPT_ROW]
        assert found["in_button"] == [0, 0], "a button inside a button presses both"
        assert found["in_bubble"] == [1, 1]
        assert found["tabindex"] == [None, None], "the host's button is the tab stop"
        assert found["role"] == ["assistant", "user"]

    def test_a_tap_on_the_words_inside_the_hosts_button_opens_the_row(self):
        found = row([("assistant", "reply 0"), ("user", "ask 1")], """
            tap(bubbles[0]);
            out.opened = state(bubbles[0]).open;
            clickOn(hostButtons[0]);
            out.closed = state(bubbles[0]).open;
            clickOn(hostButtons[1]);
            out.other = [state(bubbles[0]).open, state(bubbles[1]).open];
        """, shape="gradio")

        assert found["opened"] is True
        assert found["closed"] is False, "the button itself is the message too"
        assert found["other"] == [False, True]
        assert found["selects"] == ["bot", "bot", "user"], "Gradio's own select still fires; nothing is bound to it"

    def test_a_link_in_the_words_is_the_links(self):
        found = row([("assistant", "reply 0")], """
            const link = new El("a");
            link.setAttribute("href", "https://example.test/");
            bubbles[0].querySelector("p").appendChild(link);
            clickOn(link);
            out.open = state(bubbles[0]).open;
        """, shape="gradio")

        assert found["open"] is False

    def test_pressing_a_row_button_does_not_press_the_host_or_toggle(self):
        found = row([("assistant", "Hello there.")], """
            tap(bubbles[0]);
            const before = selects.length;
            pressAction(bubbles[0], "copy");
            await settle();
            out.selected_more = selects.length - before;
            out.open = state(bubbles[0]).open;
        """, shape="gradio")

        assert found["clipboard"] == ["Hello there."]
        assert found["selected_more"] == 0
        assert found["open"] is True, "a local action leaves the row where it is"

    def test_enter_on_the_hosts_button_is_left_to_the_click_it_becomes(self):
        found = row([("assistant", "reply 0")], """
            hostButtons[0].dispatchEvent({type: "keydown", key: "Enter", bubbles: true, target: hostButtons[0]});
            out.after_key = state(bubbles[0]).open;
            clickOn(hostButtons[0]);   // what the browser fires for that Enter
            out.after_click = state(bubbles[0]).open;
        """, shape="gradio")

        assert found["after_key"] is False, "toggling on the key and again on its click would open and close"
        assert found["after_click"] is True

    def test_a_bubble_that_is_itself_a_button_still_opens_once_per_press(self):
        """A theme that kept the button and dropped its wrapper: the row lives
        inside the button, a tap on it opens, Enter is left to the click the
        browser makes of it, and the button needs no tabindex of ours."""
        found = row([("assistant", "reply 0")], """
            out.inside = bubbles[0].children.filter((c) => c.classList.contains("mc-llm-message-actions")).length;
            out.tabindex = bubbles[0].getAttribute("tabindex");
            tap(bubbles[0]);
            out.opened = state(bubbles[0]).open;
            bubbles[0].dispatchEvent({type: "keydown", key: "Enter", bubbles: true, target: bubbles[0]});
            out.after_key = state(bubbles[0]).open;
            clickOn(bubbles[0]);
            out.after_click = state(bubbles[0]).open;
        """, shape="button")

        assert found["inside"] == 1 and found["tabindex"] is None
        assert found["opened"] is True
        assert found["after_key"] is True, "the key alone changes nothing; its click does"
        assert found["after_click"] is False

    def test_which_message_it_is_counts_bubbles_and_moves_with_them(self):
        found = row([("assistant", "reply 0"), ("assistant", "reply 1"), ("assistant", "reply 2")], """
            holder.removeChild(bubbles[0].parentNode.parentNode);   // the whole message-row
            pressAction(bubbles[2], "delete");
        """, shape="gradio")

        assert found["presses"] == [["delete", "assistant:1"]]

    def test_a_pressed_action_names_the_message_from_inside_its_row(self):
        found = row([("assistant", "reply 0"), ("user", "ask 1"), ("assistant", "reply 1")], """
            pressAction(bubbles[2], "regenerate");
            pressAction(bubbles[1], "resend");
        """, shape="gradio")

        assert found["presses"] == [["regenerate", "assistant:1"], ["resend", "user:0"]]


class TestCopyAndVibeVoice:
    def test_copy_puts_the_words_on_the_clipboard_without_the_row_or_the_marker(self):
        found = row([("assistant", "Hello there, friend.")], """
            pressAction(bubbles[0], "copy");
            await settle();
            out.glyph = buttonOf(bubbles[0], "copy").textContent;
            fire();
            out.later = buttonOf(bubbles[0], "copy").textContent;
        """)

        assert found["clipboard"] == ["Hello there, friend."]
        assert found["glyph"] == "✓", "the button says it did it"
        assert found["later"] == "⧉", "and goes back to being Copy"

    def test_send_renders_the_words_with_the_pages_voice_box_under_the_messages_key(self):
        found = row([("assistant", "Hello there.", {"key": "thread-1:abc123"})], """
            pressAction(bubbles[0], "vibe");
            out.sent = state(bubbles[0]);
            await settle();
            out.pending = state(bubbles[0]);
        """)

        assert len(found["asked"]) == 1
        assert found["asked"][0]["text"] == "Hello there."
        assert found["asked"][0]["options"]["origin"] == {"kind": "llm", "key": "thread-1:abc123",
                                                           "label": "LLM Studio"}
        assert found["asked"][0]["options"]["name"].startswith("LLM Studio")
        assert found["sent"]["vibe"] == "rendering"
        assert found["pending"]["play"]["disabled"] is True

    def test_play_comes_alive_when_the_render_is_done_and_the_message_blinks_until_played(self):
        found = row([("assistant", "Hello there.", {"key": "thread-1:abc123"})], """
            pressAction(bubbles[0], "vibe");
            await settle();
            look();
            out.live = state(bubbles[0]);
            voiceBox.jobs.j1 = {id: "j1", phase: "done", live: false, output_id: "o9"};
            look();
            out.done = state(bubbles[0]);
            pressAction(bubbles[0], "play");
            await settle();
            out.playing = state(bubbles[0]);
            out.src = bubbles[0].querySelector("audio") ? "inside" : "shared";
        """)

        assert found["live"]["play"]["disabled"] is True and found["live"]["vibe"] == "rendering"
        assert found["done"]["play"]["disabled"] is False and found["done"]["vibe"] == "ready"
        assert found["urls"] == ["o9"]
        assert found["playing"]["vibe"] == "played"
        assert found["playing"]["play"]["glyph"] == "‖"
        assert found["focus"] == [{"owner": "llm-studio", "kind": "playback"}]

    def test_a_second_send_is_one_to_one_with_the_last_press(self):
        """The first render's completion is not this message's audio any more:
        Play waits for the second and offers only it."""
        found = row([("assistant", "Hello there.", {"key": "thread-1:abc123"})], """
            pressAction(bubbles[0], "vibe");
            await settle();
            pressAction(bubbles[0], "vibe");
            await settle();
            voiceBox.jobs.j1 = {id: "j1", phase: "done", live: false, output_id: "o1"};
            look();
            out.first = state(bubbles[0]);
            voiceBox.jobs.j2 = {id: "j2", phase: "done", live: false, output_id: "o2"};
            look();
            out.second = state(bubbles[0]);
            pressAction(bubbles[0], "play");
            await settle();
        """)

        assert len(found["asked"]) == 2
        assert found["first"]["play"]["disabled"] is True and found["first"]["vibe"] == "rendering"
        assert found["second"]["play"]["disabled"] is False
        assert found["urls"] == ["o2"]

    def test_a_render_that_fails_says_so_and_play_stays_dark(self):
        found = row([("assistant", "Hello there.")], """
            pressAction(bubbles[0], "vibe");
            await settle();
            voiceBox.jobs.j1 = {id: "j1", phase: "failed", live: false, output_id: "", warning: "The card could not be made ready."};
            look();
            out.failed = state(bubbles[0]);
            out.note = noteOf(bubbles[0]);
        """)

        assert found["failed"]["play"]["disabled"] is True and found["failed"]["vibe"] is None
        assert found["note"] == "The card could not be made ready."

    def test_what_the_voice_box_refuses_is_said_in_the_row(self):
        found = row([("assistant", "Hello there.")], """
            voiceBox.refusal = "Speaker 1 has no sample.";
            pressAction(bubbles[0], "vibe");
            out.note = noteOf(bubbles[0]);
            out.state = state(bubbles[0]);
        """)

        assert found["asked"] == []
        assert found["note"] == "Speaker 1 has no sample."
        assert found["state"]["vibe"] is None

    def test_without_a_voice_box_on_the_page_the_send_button_says_so_and_is_dark(self):
        found = row([("assistant", "Hello there.")], """
            out.state = state(bubbles[0]);
            pressAction(bubbles[0], "play");
            out.note = noteOf(bubbles[0]);
        """, voice_box=False)

        assert found["state"]["send"]["disabled"] is True
        assert "Voice Box is not on this page" in found["state"]["send"]["title"]
        assert found["note"] == "Send this message to VibeVoice first."

    def test_a_reload_finds_the_renders_already_made_for_the_thread(self):
        """Read once per thread from the Voice Box's outputs by their origin
        keys, newest first; found again, a render is steady rather than
        blinking, because it is not news."""
        found = row([("assistant", "reply 0", {"key": "thread-1:aaa"}),
                     ("user", "ask 1", {"key": "thread-1:bbb"})], """
            await settle();
            out.first = state(bubbles[0]);
            out.second = state(bubbles[1]);
            pressAction(bubbles[0], "play");
            await settle();
            rewire();
            rewire();
        """, setup="""
            voiceBox.outputs = [
                {id: "o5", render: {origin: {kind: "llm", key: "thread-1:aaa"}}},
                {id: "o4", render: {origin: {kind: "llm", key: "thread-1:aaa"}}},
                {id: "o3", render: {origin: {kind: "llm", key: "thread-2:aaa"}}},
            ];
        """)

        assert found["prefixes"] == ["thread-1:"], "asked once, not on every redraw"
        assert found["first"]["vibe"] == "played" and found["first"]["play"]["disabled"] is False
        assert found["second"]["vibe"] is None and found["second"]["play"]["disabled"] is True
        assert found["urls"] == ["o5"]

    def test_somebody_else_claiming_the_speaker_pauses_the_render(self):
        found = row([("assistant", "Hello there.", {"key": "k"})], """
            pressAction(bubbles[0], "vibe");
            await settle();
            voiceBox.jobs.j1 = {id: "j1", phase: "done", live: false, output_id: "o1"};
            look();
            pressAction(bubbles[0], "play");
            await settle();
            out.playing = state(bubbles[0]).play.glyph;
            document.dispatchEvent(new CustomEvent("mc:audio-focus", {detail: {owner: "voice-chat", kind: "speech"}}));
            out.after = state(bubbles[0]).play.glyph;
            document.dispatchEvent(new CustomEvent("mc:audio-focus", {detail: {owner: "llm-studio", kind: "playback"}}));
        """)

        assert found["playing"] == "‖" and found["after"] == "▶︎"

    def test_nothing_in_the_row_is_what_lobes_icon_swap_replaces(self):
        """Lobe's SVG icon option (its ``replaceIcon``) swaps the whole content
        of any button, span or link whose text *contains* one of its strings
        (CLAUDE.md: the Voice Box lost three actions to it). The row's glyphs
        are compared with that list, not with a guess at what an emoji is: the
        waveform is a picture inside a button with no text, and ↪ is not ↩."""
        found = row([("assistant", "reply 0", {"versions": 2})], """
            const bar = rowOf(bubbles[0]);
            out.texts = buttons(bubbles[0]).map((b) => b.textContent)
                .concat(bar.querySelectorAll("span, a").map((s) => s.textContent));
        """)

        assert len(found["texts"]) >= len(REPLY_ROW) + 3
        for text in found["texts"]:
            for swapped in LOBE_SWAPS:
                assert swapped not in text, (text, swapped)


# What LobeTheme's `replaceIcon` looks for, by `textContent.includes`, in
# every button (16 px icons), span and link (36 px) of the page -- read from
# lobehub/sd-webui-lobe-theme, src/scripts/replaceIcon.ts, 2026-10-08.
LOBE_SWAPS = (
    # buttons
    "🖌️", "🗃️", "🖼️", "🎨️", "📂", "🔄", "🔁", "♻️", "↙️", "⤴", "↕️", "🗑️", "📋", "💾", "🎲️",
    "🪄", "⚙️", "➡️", "⇅", "⇄", "🎴", "🌀", "💥", "📷", "📝", "📐", "⬇️", "↩", "📒", "📎",
    "📦", "💞", "✨",
    # spans
    "⤡", "⊞", "🖫", "×",
    # links
    "❮", "❯",
)


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


# --------------------------------------------------------------------------- #
# The paperclip opens the browser's own file picker
# --------------------------------------------------------------------------- #

# Conversation's picture chip is a Gradio Image, which is an upload area with a
# file input inside it. Tapping the area opens the picker; the paperclip beside
# it forwards a press to that same input, so there is one way in and it is the
# ordinary one. What used to be there instead was a full-width drop target above
# the composer -- a panel's worth of empty dashed border to say "no picture yet",
# on the one surface that must never grow.

PICKER = """
const opened = [];

function chip(id) {
    const input = {type: "file", click() { opened.push(id); }};
    return {
        id,
        tagName: "DIV",
        dataset: {},
        querySelector: (selector) => (selector.indexOf("file") >= 0 ? input : null),
        querySelectorAll: () => [],
        addEventListener() {},
    };
}

function paperclip(id) {
    const node = {
        id,
        tagName: "BUTTON",
        dataset: {},
        handlers: {},
        addEventListener(kind, fn) { node.handlers[kind] = fn; },
        querySelector: () => null,
        querySelectorAll: () => [],
    };
    return node;
}

const elements = {
    "mc-llm-chat-attach": paperclip("mc-llm-chat-attach"),
    "mc-llm-chat-image": chip("mc-llm-chat-image"),
    "mc-llm-chat-edit-attach": paperclip("mc-llm-chat-edit-attach"),
    "mc-llm-chat-edit-image": chip("mc-llm-chat-edit-image"),
};

const html = {scrollTop: 0, scrollHeight: 0, clientHeight: 0,
              getAttribute: () => null, setAttribute() {}, removeAttribute() {}};
globalThis.document = {
    documentElement: html,
    querySelector: (selector) => elements[selector.replace("#", "")] || null,
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
__PRESS__
console.log(JSON.stringify({opened, wired: Object.keys(elements["mc-llm-chat-attach"].handlers)}));
"""


def picker(press: str = "") -> dict:
    harness = PICKER.replace("SOURCE", SCRIPT.read_text()).replace("__PRESS__", press)
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestThePaperclipOpensThePicker:
    def test_pressing_it_clicks_the_chip_s_own_file_input(self):
        found = picker('elements["mc-llm-chat-attach"].handlers.click();')

        assert found["opened"] == ["mc-llm-chat-image"]

    def test_the_editor_has_one_of_its_own(self):
        """A message that was sent with a picture is edited as a whole, so the
        picture has to be changeable there and not only where it was first
        attached."""
        found = picker('elements["mc-llm-chat-edit-attach"].handlers.click();')

        assert found["opened"] == ["mc-llm-chat-edit-image"]

    def test_nothing_is_opened_until_it_is_pressed(self):
        assert picker()["opened"] == []

    def test_wiring_twice_does_not_press_twice(self):
        """The tab is rebuilt on some updates, so the wiring is re-applied. A
        listener added on every pass would open one picker per rebuild."""
        found = picker('loaded.forEach((fn) => fn());'
                       'elements["mc-llm-chat-attach"].handlers.click();')

        assert found["opened"] == ["mc-llm-chat-image"]


# --------------------------------------------------------------------------- #
# Editing in the message editor's dialog
# --------------------------------------------------------------------------- #
#
# Edit opens the row under the transcript, and that row is still what the
# server knows about. Asked for instead: "a simple pop up ... For both LLM
# studio conversation and our flyout menu." So the script watches the row,
# opens the shared dialog with the box's words when it appears, and finishes
# the press through the row's own Save and Cancel. Python decides everything
# after that, as before.

DIALOG = """
function control(id, tag) {
    const node = {
        id, tagName: tag, dataset: {}, handlers: {}, disabled: false,
        addEventListener(kind, fn) { node.handlers[kind] = fn; },
        querySelector: () => null,
        querySelectorAll: () => [],
        click() { pressed.push(id); },
    };
    return node;
}
const pressed = [];
const events = [];
const field = {tagName: "TEXTAREA", value: "make it top down view",
               dispatchEvent(event) { events.push(event.type + ":" + event.bubbles); }};
const holder = control("mc-llm-chat-editor", "DIV");
holder.querySelector = (selector) => selector === "textarea" ? field : null;
const row = control("mc-llm-chat-edit", "DIV");
row.offsetParent = ROW_SHOWING ? {} : null;
const elements = {
    "mc-llm-chat-edit": row,
    "mc-llm-chat-editor": holder,
    "mc-llm-chat-edit-save": control("mc-llm-chat-edit-save", "BUTTON"),
    "mc-llm-chat-edit-cancel": control("mc-llm-chat-edit-cancel", "BUTTON"),
};
const opened = [];
if (WITH_EDITOR) {
    globalThis.mcMessageEditor = {open(options) { opened.push(options); return true; }};
}
const observers = [];
globalThis.MutationObserver = function (callback) {
    this.callback = callback;
    this.observe = (node, options) => { observers.push({node: node.id, options, fire: callback}); };
};
const html = {scrollTop: 0, scrollHeight: 0, clientHeight: 0,
              getAttribute: () => null, setAttribute() {}, removeAttribute() {}};
globalThis.document = {
    documentElement: html,
    querySelector: (selector) => elements[selector.replace("#", "")] || null,
    addEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
globalThis.addEventListener = () => {};
globalThis.setTimeout = (fn) => { fn(); return 0; };
globalThis.setInterval = () => 0;
globalThis.gradioApp = () => globalThis.document;

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());
const show = () => { row.offsetParent = {}; observers.forEach((o) => o.fire()); };
const hide = () => { row.offsetParent = null; observers.forEach((o) => o.fire()); };
__SCENARIO__
console.log(JSON.stringify({
    opened: opened.map((o) => ({title: o.title, text: o.text, placeholder: o.placeholder})),
    observed: observers.map((o) => [o.node, o.options.attributeFilter]),
    pressed, events, value: field.value,
    __EXTRA__
}));
"""


def dialog(scenario: str = "", extra: str = "", showing: bool = False,
           with_editor: bool = True) -> dict:
    harness = (DIALOG.replace("SOURCE", SCRIPT.read_text())
               .replace("__SCENARIO__", scenario)
               .replace("__EXTRA__", extra)
               .replace("ROW_SHOWING", "true" if showing else "false")
               .replace("WITH_EDITOR", "true" if with_editor else "false"))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestEditOpensTheDialog:
    def test_the_row_is_watched_and_nothing_opens_until_it_shows(self):
        found = dialog()

        assert found["observed"] == [["mc-llm-chat-edit", ["class", "style"]]]
        assert found["opened"] == []

    def test_when_the_row_appears_the_dialog_opens_with_the_box_s_words(self):
        found = dialog("show();")

        assert found["opened"] == [{"title": "Edit message", "text": "make it top down view",
                                    "placeholder": "The message’s new words…"}]
        assert found["pressed"] == []

    def test_a_row_already_showing_when_the_page_is_wired_opens_at_once(self):
        found = dialog(showing=True)

        assert len(found["opened"]) == 1

    def test_a_row_that_stays_showing_does_not_open_a_second_dialog(self):
        """The observer fires on every attribute change; only the change
        from hidden to showing is an Edit."""
        found = dialog("show(); show(); hide(); show();")

        assert len(found["opened"]) == 2

    def test_done_hands_the_words_to_the_row_and_presses_its_save(self):
        found = dialog("""
            show();
            const answer = opened[0].done("make it a top down view, at dusk");
        """, extra="answer,")

        assert found["value"] == "make it a top down view, at dusk"
        assert found["events"] == ["input:true"], (
            "Gradio reads the box on `input`, not on assignment")
        assert found["pressed"] == ["mc-llm-chat-edit-save"]
        assert found["answer"] == {"ok": True}

    def test_a_save_that_cannot_be_pressed_is_a_refusal_the_dialog_shows(self):
        found = dialog("""
            show();
            elements["mc-llm-chat-edit-save"].disabled = true;
            const answer = opened[0].done("edited");
        """, extra="answer,")

        assert found["pressed"] == []
        assert found["answer"] == {"ok": False, "message": "Save is not available right now."}

    def test_cancel_presses_the_row_s_cancel(self):
        found = dialog("""
            show();
            opened[0].cancel();
        """)

        assert found["pressed"] == ["mc-llm-chat-edit-cancel"]
        assert found["value"] == "make it top down view", "the words are left alone"

    def test_without_the_dialog_s_script_the_row_is_the_editor_as_before(self):
        found = dialog("show();", with_editor=False)

        assert found["observed"] == [] and found["opened"] == []

    def test_wiring_twice_watches_once(self):
        found = dialog("loaded.forEach((fn) => fn()); show();")

        assert len(found["observed"]) == 1
        assert len(found["opened"]) == 1

    def test_the_ids_are_the_panel_s(self):
        """Two halves in two languages with nothing between them but these
        strings, so they are compared rather than trusted."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import mc_llm_ui as ui

        script = SCRIPT.read_text(encoding="utf-8")
        for name in ("edit", "editor", "edit-save", "edit-cancel"):
            assert f'"{ui.ident("chat", name)}"' in script, name
