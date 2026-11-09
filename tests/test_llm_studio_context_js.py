"""LLM Studio's context meter, executed under node.

Python draws the meter into a hidden box (``mc-llm-chat-context``);
``javascript/llm_studio.js`` shows it at the right end of the status line and
asks for it again, through a hidden button, whenever that line changes -- one
request a change, never a poll, and never for the script's own writes to the
line (the meter itself, the seconds a busy line counts). A press on it opens
the Conversation pop-up at its Context section (2026-11-09: "make it something
subtle, like where the status message is").
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "javascript" / "llm_studio.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

PAGE = """
const elements = {};
const observers = [];
const timers = [];
class ClassList {
    constructor(node) { this.node = node; this.names = new Set(); }
    add(name) { this.names.add(name); }
    remove(name) { this.names.delete(name); }
    contains(name) { return this.names.has(name); }
}
function make(tag, id) {
    const node = {
        tagName: (tag || "div").toUpperCase(), id: id || "", nodeType: 1, dataset: {},
        children: [], parentElement: null, attrs: {}, listeners: {}, clicks: 0, shown: true,
        get className() { return Array.from(this.classList.names).join(" "); },
        set className(value) { this.classList.names = new Set(String(value).split(/\\s+/).filter(Boolean)); },
        textContent: "", title: "", type: "",
        setAttribute(name, value) { this.attrs[name] = String(value); },
        getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; },
        appendChild(child) { child.parentElement = this; this.children.push(child); return child; },
        remove() { if (this.parentElement) { const p = this.parentElement;
            p.children = p.children.filter((c) => c !== this); this.parentElement = null; } },
        addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
        removeEventListener() {},
        click() { this.clicks += 1; (this.listeners.click || []).forEach((fn) => fn({preventDefault() {}})); },
        getClientRects() { return this.shown ? [1] : []; },
        querySelector(selector) { return find(this, selector); },
        querySelectorAll() { return []; },
        contains(other) { let n = other; while (n) { if (n === this) return true; n = n.parentElement; } return false; },
    };
    node.classList = new ClassList(node);
    if (id) elements[id] = node;
    return node;
}
function matches(node, selector) {
    if (selector.startsWith("#")) return node.id === selector.slice(1);
    if (selector === "button.label-wrap") return node.tagName === "BUTTON" && node.classList.contains("label-wrap");
    if (selector === "button") return node.tagName === "BUTTON";
    if (selector.startsWith(".")) return node.classList.contains(selector.slice(1));
    return false;
}
function find(root, selector) {
    for (const child of root.children) {
        if (matches(child, selector)) return child;
        const deeper = find(child, selector);
        if (deeper) return deeper;
    }
    return null;
}
const page = make("div", "page");
globalThis.document = {
    documentElement: {scrollTop: 0, clientWidth: 1280, clientHeight: 900,
                      getAttribute: () => null, setAttribute() {}, removeAttribute() {}},
    querySelector: (selector) => selector.startsWith("#") ? (elements[selector.slice(1)] || null) : null,
    querySelectorAll: () => [],
    getElementById: (id) => elements[id] || null,
    createElement: (tag) => make(tag),
    addEventListener() {}, removeEventListener() {},
    readyState: "complete",
};
globalThis.window = globalThis;
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.getComputedStyle = () => ({visibility: "visible", getPropertyValue: () => ""});
globalThis.setTimeout = (fn, ms) => { timers.push(fn); return timers.length; };
globalThis.clearTimeout = (handle) => { if (handle) timers[handle - 1] = null; };
globalThis.setInterval = () => 0;
globalThis.MutationObserver = function (callback) {
    this.observe = (target) => observers.push({target, callback});
    this.disconnect = () => {};
};
globalThis.gradioApp = () => globalThis.document;
const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

// The tab: the status line, the meter's hidden box and its button, the chip,
// the pop-up and its Context section.
const status = page.appendChild(make("div", "mc-llm-chat-status"));
const notice = status.appendChild(make("div"));
notice.className = "mc-llm-notice";
notice.textContent = "Ready.";
const source = page.appendChild(make("div", "mc-llm-chat-context"));
function reading(text, state, title) {
    source.children = [];
    if (!text) return;
    const span = source.appendChild(make("span"));
    span.className = "mc-llm-context";
    span.textContent = text;
    span.setAttribute("title", title || "");
    span.setAttribute("data-state", state || "ok");
}
const nowHolder = page.appendChild(make("div", "mc-llm-chat-context-now"));
const now = nowHolder.appendChild(make("button"));
const chipHolder = page.appendChild(make("div", "mc-llm-chat-to-conversation"));
const chip = chipHolder.appendChild(make("button"));
const sheet = page.appendChild(make("div", "mc-llm-chat-conversation"));
sheet.shown = false;
const section = sheet.appendChild(make("div", "mc-llm-chat-context-section"));
const heading = section.appendChild(make("button"));
heading.classList.add("label-wrap");
let headingOpened = 0;
heading.addEventListener("click", () => { headingOpened += 1; heading.classList.add("open"); });
chip.addEventListener("click", () => { sheet.shown = true; });

reading(READING_TEXT, READING_STATE, "4,401 of 8,192 tokens in context");
SOURCE
loaded.forEach((fn) => fn());

const shown = () => status.querySelector(".mc-llm-context-shown");
const fire = (target, mutations) => observers.filter((o) => o.target === target)
    .forEach((o) => o.callback(mutations));
const flush = () => { const due = timers.splice(0); due.forEach((fn) => fn && fn()); };
const describe = () => { const meter = shown();
    return meter ? {text: meter.textContent, state: meter.getAttribute("data-state"),
                    title: meter.title} : null; };
SCENARIO
"""


def page(scenario: str, text: str = "4,401 / 8,192", state: str = "ok"):
    harness = (PAGE.replace("SOURCE", SCRIPT.read_text())
               .replace("READING_TEXT", json.dumps(text))
               .replace("READING_STATE", json.dumps(state))
               .replace("SCENARIO", scenario))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTheMeterOnTheStatusLine:
    def test_it_is_shown_at_the_end_of_the_status_line(self):
        found = page("console.log(JSON.stringify(describe()));")

        assert found == {"text": "4,401 / 8,192", "state": "ok",
                         "title": "4,401 of 8,192 tokens in context"}

    def test_it_follows_what_python_drew(self):
        found = page("""
            reading("7,300 / 8,192", "full", "past the room");
            fire(source, [{target: source, addedNodes: [], removedNodes: []}]);
            console.log(JSON.stringify(describe()));
        """)

        assert found == {"text": "7,300 / 8,192", "state": "full", "title": "past the room"}

    def test_nothing_for_no_thread(self):
        found = page("""
            reading("");
            fire(source, [{target: source, addedNodes: [], removedNodes: []}]);
            console.log(JSON.stringify(describe()));
        """)

        assert found is None

    def test_it_is_put_back_when_the_line_is_drawn_again(self):
        """Gradio draws the line's box again; the meter is appended to the
        line itself, and a redraw that took it is answered at once."""
        found = page("""
            const meter = shown();
            meter.remove();
            fire(status, [{target: status, addedNodes: [], removedNodes: [meter]},
                          {target: notice, addedNodes: [], removedNodes: []}]);
            console.log(JSON.stringify(describe()));
        """)

        assert found["text"] == "4,401 / 8,192"


class TestWhenItIsAskedForAgain:
    def test_once_when_it_is_wired(self):
        found = page("flush(); console.log(JSON.stringify(now.clicks));")

        assert found == 1

    def test_when_the_line_changes_once_for_a_burst(self):
        found = page("""
            flush();
            notice.textContent = "Replying…";
            fire(status, [{target: notice, addedNodes: [], removedNodes: []}]);
            notice.textContent = "Reply complete.";
            fire(status, [{target: notice, addedNodes: [], removedNodes: []}]);
            flush();
            console.log(JSON.stringify(now.clicks));
        """)

        assert found == 2

    def test_never_for_its_own_writes_or_the_seconds_a_busy_line_counts(self):
        found = page("""
            flush();
            const meter = shown();
            fire(status, [{target: meter, addedNodes: [], removedNodes: []}]);
            fire(status, [{target: status, addedNodes: [meter], removedNodes: []}]);
            const seconds = make("span"); seconds.className = "mc-llm-busy-elapsed";
            fire(status, [{target: seconds, addedNodes: [], removedNodes: []}]);
            flush();
            console.log(JSON.stringify(now.clicks));
        """)

        assert found == 1


class TestAPressOnIt:
    def test_opens_the_pop_up_at_its_context_section(self):
        found = page("""
            flush();
            shown().click();
            for (let round = 0; round < 5; round += 1) flush();
            console.log(JSON.stringify({chip: chip.clicks, opened: headingOpened,
                                        open: heading.classList.contains("open")}));
        """)

        assert found == {"chip": 1, "opened": 1, "open": True}

    def test_with_the_pop_up_open_it_only_opens_the_section(self):
        found = page("""
            flush();
            sheet.shown = true;
            shown().click();
            for (let round = 0; round < 5; round += 1) flush();
            console.log(JSON.stringify({chip: chip.clicks, opened: headingOpened}));
        """)

        assert found == {"chip": 0, "opened": 1}
