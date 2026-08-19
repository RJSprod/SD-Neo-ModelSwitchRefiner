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
    documentElement: {scrollTop: SCROLLED},
    querySelector: (selector) => elements[selector.replace("#", "")] || null,
    addEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.innerHeight = WINDOW_HEIGHT;
globalThis.scrollY = SCROLLED;
globalThis.addEventListener = () => {};
globalThis.setTimeout = (fn) => { fn(); return 0; };
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
