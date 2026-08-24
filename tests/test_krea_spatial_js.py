"""The Spatial Layout editor, executed rather than read.

Two kinds of question, and only one of them can be answered by reading the file.

**Does it draw what somebody drew?** A drag from one corner to another has to
become a box at those normalized coordinates, a drag the other way has to become
the same box, a drag off the edge has to stop at it, and a tap has to become
nothing at all. Each of those has one right answer and several plausible wrong
ones, and every wrong one shows up much later as a subject in the wrong place.

**Can it delay a generation?** This is the question the whole Creative Mode
browser story turns on. The old gate swallowed the Generate click, polled a
hidden textbox on a setInterval, and clicked Generate again once the server had
answered -- so a hidden tab made an image late and a closed one made it never
arrive. The editor here writes into a hidden textbox too, and the tests below are
how "that box is an input, not a channel" is checked rather than asserted: run an
hour forward with nobody touching anything and assert no timer exists, and assert
this file never so much as names the Generate button.

The fake page is built from the real markup, nesting and all: ``spatial_editor()``
is parsed into a tree and rebuilt element for element, so ``closest()`` walks the
same ancestors it would in a browser and a control renamed in Python but not in
JavaScript fails here rather than in somebody's tab.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_spatial_krea.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

VOID = {"input", "br", "hr", "img"}
KEPT = ("id", "class", "type", "hidden", "checked", "value")


class _Tree(HTMLParser):
    """The editor markup as a nest of ``{tag, attrs, children}``.

    Not a general HTML parser and not trying to be: the markup it reads is one
    static block this repository writes, so the only thing it has to get right
    is which element is inside which.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {"tag": "div", "attrs": {}, "children": []}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = {"tag": tag,
                "attrs": {name: ("" if value is None else value)
                          for name, value in attrs
                          if name in KEPT or name.startswith("data-")},
                "children": []}
        self.stack[-1]["children"].append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        for at in range(len(self.stack) - 1, 0, -1):
            if self.stack[at]["tag"] == tag:
                del self.stack[at:]
                return


def _markup() -> str:
    """Both canvases, because both are one file's responsibility.

    The compact canvas in the pipeline and the full workspace behind Edit
    Layout… are two views of one document written by one browser file, so the
    fake page carries both -- a test that moved a box on one and asserted about
    the other would otherwise be asserting about a page that does not exist.
    """
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import model_chain_krea_creative as creative_script

    return creative_script.spatial_editor() + creative_script.spatial_compact()


def editor_tree() -> list:
    """The workspace's element tree, for the harness to rebuild."""
    parser = _Tree()
    parser.feed(_markup())
    return parser.root["children"]


def editor_elements() -> list[tuple[str, str]]:
    """``(tag, id)`` for everything the Python-built editor markup carries."""
    return re.findall(r"<(\w+)[^>]*\bid=\"([^\"]+)\"", _markup())


def editor_elements_ids() -> list[str]:
    """Just the ids, for a test that only asks whether a control still exists."""
    return [found for _tag, found in editor_elements()]


HARNESS = """
// A txt2img page with the Spatial Layout workspace in it, a clock a test can
// move, and just enough DOM for the file under test to be the real file.

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
globalThis.clearTimeout = function (id) { timers = timers.filter((t) => t.id !== id); };
globalThis.clearInterval = globalThis.clearTimeout;
globalThis.Date.now = () => now;

function advance(ms) {
    const target = now + ms;
    for (let guard = 0; guard < 100000; guard += 1) {
        const due = timers.filter((t) => t.at <= target).sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        now = due.at;
        if (due.every) { due.at = now + due.every; }
        else { timers = timers.filter((t) => t.id !== due.id); }
        due.fn();
    }
    now = target;
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

// ------------------------------------------------------------------ DOM ---

function matches(element, selector) {
    return selector.split(",").map((part) => part.trim()).filter(Boolean)
        .some(function (part) {
            if (part.startsWith("#")) return element.id === part.slice(1);
            if (part.startsWith(".")) return element.classList.contains(part.slice(1));
            return element.tagName === part.toUpperCase();
        });
}

class El {
    constructor(tag) {
        this.tagName = String(tag).toUpperCase();
        this.id = "";
        this.children = [];
        this.parentNode = null;
        this.dataset = {};
        this.style = {};
        this.attributes = {};
        this.hidden = false;
        this.disabled = false;
        this.checked = false;
        this.value = "";
        this.title = "";
        this.type = "";
        this.tabIndex = -1;
        this.selected = false;
        this.isContentEditable = false;
        this.focused = false;
        this._text = "";
        this._classes = new Set();
        this._listeners = {};
        this._rect = null;
    }
    get className() { return Array.from(this._classes).join(" "); }
    set className(value) {
        this._classes = new Set(String(value).split(/\\s+/).filter(Boolean));
    }
    get classList() {
        const owner = this;
        return {
            add(name) { owner._classes.add(name); },
            remove(name) { owner._classes.delete(name); },
            contains(name) { return owner._classes.has(name); },
            toggle(name, force) {
                const on = force === undefined ? !owner._classes.has(name) : !!force;
                if (on) owner._classes.add(name); else owner._classes.delete(name);
            },
        };
    }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    getAttribute(name) {
        return this.attributes[name] === undefined ? null : this.attributes[name];
    }
    get textContent() {
        return this._text + this.children.map((child) => child.textContent).join("");
    }
    set textContent(value) { this.children = []; this._text = String(value); }
    appendChild(child) {
        if (child.parentNode) child.parentNode.removeChild(child);
        child.parentNode = this;
        this.children.push(child);
        return child;
    }
    removeChild(child) {
        this.children = this.children.filter((entry) => entry !== child);
        child.parentNode = null;
        return child;
    }
    addEventListener(kind, fn) {
        (this._listeners[kind] = this._listeners[kind] || []).push(fn);
    }
    dispatchEvent(event) {
        event.target = event.target || this;
        (this._listeners[event.type] || []).forEach((fn) => fn(event));
        return true;
    }
    walk() {
        return this.children.reduce(function (found, child) {
            return found.concat([child], child.walk());
        }, []);
    }
    querySelector(selector) {
        return this.walk().filter((el) => matches(el, selector))[0] || null;
    }
    querySelectorAll(selector) {
        return this.walk().filter((el) => matches(el, selector));
    }
    closest(selector) {
        let at = this;
        while (at) {
            if (matches(at, selector)) return at;
            at = at.parentNode;
        }
        return null;
    }
    focus() { this.focused = true; focused = this; }
    getBoundingClientRect() {
        return this._rect || {left: 0, top: 0, width: 400, height: 400};
    }
}

let focused = null;

const body = new El("body");
const page = body.appendChild(new El("div"));

function make(tag, id, into) {
    const element = new El(tag);
    element.id = id;
    (into || page).appendChild(element);
    return element;
}

// The real markup, rebuilt element for element so that closest() walks the same
// ancestors it would in a browser: a resize handle is inside the proxy, the
// proxy is inside the frame, and the file under test relies on both.
function build(node, into) {
    const element = new El(node.tag);
    const attrs = node.attrs || {};
    if (attrs.id) element.id = attrs.id;
    if (attrs["class"]) element.className = attrs["class"];
    if (attrs.type !== undefined) element.type = attrs.type;
    if (attrs.value !== undefined) element.value = attrs.value;
    if (attrs.hidden !== undefined) element.hidden = true;
    if (attrs.checked !== undefined) element.checked = true;
    Object.keys(attrs).forEach(function (name) {
        if (!name.startsWith("data-")) return;
        const key = name.slice(5).replace(/-([a-z])/g,
                                          (_all, letter) => letter.toUpperCase());
        element.dataset[key] = attrs[name];
    });
    into.appendChild(element);
    (node.children || []).forEach((child) => build(child, element));
    return element;
}

function buildWorkspace(into) {
    TREE.forEach((node) => build(node, into || page));
    const frame = (into || page).querySelector("#mc-krea-spatial-canvas");
    // 1000 units of layout across, so a normalized coordinate and a client
    // coordinate are the same number and a test can say what it means.
    if (frame) frame._rect = {left: 0, top: 0, width: 1000, height: 1000};
    const small = (into || page).querySelector("#mc-krea-spatial-compact-frame");
    if (small) small._rect = {left: 0, top: 0, width: 1000, height: 1000};
    return frame;
}

let canvas = buildWorkspace(page);

// What Gradio does when it rebuilds a tab: the HTML component's markup comes
// back, ids and all, while the copy the file already wired is still there.
function rebuildMarkup() {
    canvas = buildWorkspace(page);
    return canvas;
}

// The Gradio side: the hidden state box, the Edit button and the size fields.
const stateBox = new El("textarea");
stateBox.value = INITIAL;
const stateHolder = make("div", "mc-krea-spatial-state");
stateHolder.appendChild(stateBox);

const openButton = make("button", "mc-krea-spatial-open");

// The compact canvas's own Gradio controls: a checkbox in a holder, the way
// Gradio renders one, and two buttons.
const autoSaveBox = new El("input");
autoSaveBox.type = "checkbox";
autoSaveBox.checked = true;
make("div", "mc-krea-spatial-autosave").appendChild(autoSaveBox);
const compactUndoButton = make("button", "mc-krea-spatial-compact-undo");
const compactCommitButton = make("button", "mc-krea-spatial-compact-commit");

const width = new El("input");
width.value = "1024";
make("div", "txt2img_width").appendChild(width);
const height = new El("input");
height.value = "1344";
make("div", "txt2img_height").appendChild(height);

const published = [];
stateBox.addEventListener("input", function () { published.push(stateBox.value); });

// Present so a file that went looking for it would find it. Nothing here may
// touch it: a generation that a browser can delay is the arrangement this whole
// design removed.
const generate = make("button", "txt2img_generate");
let generateListeners = 0;
generate.addEventListener = function () { generateListeners += 1; };
generate.click = function () { throw new Error("the editor pressed Generate"); };

globalThis.document = {
    body: body,
    dataset: {},
    readyState: "complete",
    // Present but empty, the way a browser that offers the Fullscreen API and
    // is not currently in it looks. A test that wants the API to work installs
    // requestFullscreen on the workspace itself; one that wants the fallback
    // simply does not.
    fullscreenElement: null,
    exitFullscreen() {
        globalThis.document.fullscreenElement = null;
        (docListeners.fullscreenchange || []).forEach((fn) => fn({}));
        return Promise.resolve();
    },
    createElement: (tag) => new El(tag),
    querySelector: (selector) => body.querySelector(selector),
    querySelectorAll: (selector) => body.querySelectorAll(selector),
    getElementById: (id) => body.querySelector("#" + id),
    addEventListener(kind, fn) { (docListeners[kind] = docListeners[kind] || []).push(fn); },
};
const docListeners = {};

globalThis.window = globalThis;
if (POINTERS) globalThis.PointerEvent = function () {};
globalThis.gradioApp = () => globalThis.document;
globalThis.Event = function (type) { this.type = type; this.bubbles = true; };

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());

const ks = globalThis.modelChainKreaSpatial;

function el(id) { return body.querySelector("#" + id); }

function fire(kind, event) {
    (docListeners[kind] || []).forEach((fn) => fn(event));
}

function press(id) {
    const element = el(id);
    element.dispatchEvent({type: "click", target: element, preventDefault() {}});
}

// Pointer Events, delivered to the frame with the deep target set, which is what
// a bubbling pointerdown on a region actually looks like: the editor listens on
// the frame and works out what was under the contact from event.target.closest.
// pointermove and pointerup go to the frame too, because that is where a
// captured pointer sends them.
function drag(from, to, target, kind) {
    canvas.dispatchEvent({type: "pointerdown", target: target || canvas, pointerId: 1,
                          isPrimary: true, button: 0,
                          clientX: from[0], clientY: from[1], preventDefault() {}});
    canvas.dispatchEvent({type: "pointermove", target: target || canvas, pointerId: 1,
                          clientX: to[0], clientY: to[1], preventDefault() {}});
    canvas.dispatchEvent({type: kind || "pointerup", target: target || canvas,
                          pointerId: 1, clientX: to[0], clientY: to[1],
                          preventDefault() {}});
}

function draw(from, to) {
    press("mc-krea-spatial-draw");
    drag(from, to);
}

function proxy() { return el("mc-krea-spatial-proxy"); }

function small() { return el("mc-krea-spatial-compact-frame"); }

function compactRegion(id) {
    return body.querySelectorAll(".mc-krea-spatial-compact-region")
        .filter((entry) => entry.dataset.regionId === id)[0] || null;
}

// A drag on the compact canvas, delivered the way a bubbling pointerdown on one
// of its regions actually looks: the file listens on the frame and works out
// what was under the contact from event.target.closest.
function compactDrag(from, to, target, kind) {
    const frame = small();
    frame.dispatchEvent({type: "pointerdown", target: target || frame, pointerId: 1,
                         isPrimary: true, button: 0,
                         clientX: from[0], clientY: from[1], preventDefault() {}});
    frame.dispatchEvent({type: "pointermove", target: target || frame, pointerId: 1,
                         clientX: to[0], clientY: to[1], preventDefault() {}});
    frame.dispatchEvent({type: kind || "pointerup", target: target || frame,
                         pointerId: 1, clientX: to[0], clientY: to[1],
                         preventDefault() {}});
}

// A workspace whose requestFullscreen answers the way `answer` says: "yes"
// resolves and promotes it, "no" rejects so the fallback has to take over.
function fullscreenAnswers(answer) {
    const workspace = el("mc-krea-spatial-workspace");
    workspace.requestFullscreen = function () {
        if (answer === "no") return Promise.reject(new Error("refused"));
        globalThis.document.fullscreenElement = workspace;
        return Promise.resolve();
    };
    return workspace;
}

function fullpage() {
    return el("mc-krea-spatial-workspace").classList.contains("fullpage");
}

function handle(corner) {
    return proxy().querySelector(".mc-krea-spatial-handle-" + corner);
}

function rows() {
    return el("mc-krea-spatial-list").querySelectorAll(".mc-krea-spatial-row");
}

// Delegated: the region list is rebuilt on every paint, so its rows cannot
// carry listeners of their own. One listener on the container hears them all,
// which is what a click on a row actually looks like once it has bubbled.
function inList(target) {
    el("mc-krea-spatial-list").dispatchEvent({type: "click", target: target,
                                              preventDefault() {}});
}

function clickRow(index) {
    inList(rows()[index]);
}

function key(name, target, extra) {
    fire("keydown", Object.assign({key: name, target: target || canvas,
                                   preventDefault() {}}, extra || {}));
}

function field(id, value) {
    const element = el(id);
    element.value = value;
    element.dispatchEvent({type: "input", target: element});
}

function commit(id, value) {
    const element = el(id);
    element.value = value;
    element.dispatchEvent({type: "change", target: element});
}

function saved() {
    return published.length ? JSON.parse(published[published.length - 1]) : null;
}

const report = (extra) => console.log(JSON.stringify(Object.assign({
    timers: timers.length,
    documentListeners: Object.keys(docListeners)
        .reduce((total, kind) => total + docListeners[kind].length, 0),
    workspaces: body.querySelectorAll("#mc-krea-spatial-workspace").length,
    generateListeners: generateListeners,
    open: ks.state.open,
    regions: ks.ordered().map(function (region) {
        return {id: region.id, bbox: region.bbox, type: region.type,
                name: region.name, prompt: region.prompt, z: region.z};
    }),
    selected: ks.state.selected,
    past: ks.state.past.length,
    future: ks.state.future.length,
    zoom: ks.state.zoom,
    confirming: ks.state.confirming,
    published: published.length,
    stateBox: stateBox.value,
    compactRegions: ks.compactRegions().map(function (region) {
        return {id: region.id, bbox: region.bbox, name: region.name};
    }),
    compactPast: ks.compact.past.length,
    compactDirty: ks.compact.dirty,
    compactNote: (el("mc-krea-spatial-compact-note") || {}).textContent || "",
    compactDrawn: body.querySelectorAll(".mc-krea-spatial-compact-region")
        .map((entry) => entry.dataset.regionId),
}, extra || {})));

BODY
"""


def run(script: str, initial: str = "", pointers: bool = True) -> dict:
    harness = (
        HARNESS.replace("SOURCE", SCRIPT.read_text(encoding="utf-8"))
        .replace("BODY", script)
        .replace("TREE", json.dumps(editor_tree()))
        .replace("POINTERS", "true" if pointers else "false")
        .replace("INITIAL", json.dumps(initial))
    )
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


FACE = {"id": "r1", "name": "Face", "type": "obj", "bbox": [35, 55, 315, 360],
        "prompt": "elderly Japanese woman", "framing": "Close-up", "angle": "3/4 left",
        "z": 0}


def document(regions=(FACE,), mode="smart", width=1024, height=1344,
             grid="none") -> str:
    return json.dumps({"version": 1, "canvas": {"width": width, "height": height,
                                                "grid": grid},
                       "compose_mode": mode, "auto_position_hint": True,
                       "regions": list(regions)})


# --------------------------------------------------------------------------- #
# Nothing here can delay a generation
# --------------------------------------------------------------------------- #


class TestTheGenerationDoesNotWaitForThisFile:
    """The property the whole Creative Mode browser story turns on, defended
    one feature further out. A hidden textbox that a page writes into is the
    exact shape of the gate that used to strand generations, and the difference
    is entirely in what waits for it: here, nothing does."""

    def test_no_timer_is_ever_armed(self):
        found = run("""
            ks.open();
            draw([100, 100], [400, 400]);
            ks.save();
            advance(60 * 60 * 1000);
            await flush();
            report();
        """, initial=document())

        assert found["timers"] == 0

    def test_the_file_never_names_the_generate_button(self):
        """Read rather than run, because the honest way to ask "could this ever
        press Generate" is to check that it does not know where Generate is."""
        source = SCRIPT.read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("//"))

        assert "txt2img_generate" not in code
        assert ".click(" not in code

    def test_it_listens_to_nothing_on_the_generate_button(self):
        found = run("""
            ks.open();
            ks.save();
            report();
        """, initial=document())

        assert found["generateListeners"] == 0

    def test_re_wiring_does_not_stack_up_document_listeners(self):
        """``onAfterUiUpdate`` fires on most interactions, and ``document`` has
        no dataset for a flag to live on -- so the guard that makes every other
        listener idempotent cannot cover them. Two are left -- a keystroke, and
        the browser saying it has left full screen. Pointer capture removed the
        rest: a drag that leaves the frame is still delivered to the frame, so
        there is nothing else for the document to hear."""
        found = run("""
            const first = Object.keys(docListeners)
                .reduce((total, kind) => total + docListeners[kind].length, 0);
            ks.wire(); ks.wire(); ks.wire();
            report({first: first});
        """, initial=document())

        assert found["first"] == 2
        assert found["documentListeners"] == 2

    def test_there_is_no_polling_of_any_kind(self):
        found = run("""
            ks.open();
            advance(24 * 60 * 60 * 1000);
            await flush();
            report();
        """, initial=document())

        assert found["timers"] == 0


# --------------------------------------------------------------------------- #
# Pointer input
# --------------------------------------------------------------------------- #


class TestOnePointerPath:
    """§5.1. One implementation for mouse, finger and pen, and the old
    document-wide mousemove/mouseup gone with it."""

    def test_the_frame_listens_for_pointer_events(self):
        found = run("""
            report({kinds: Object.keys(canvas._listeners).sort()});
        """, initial=document())

        assert found["kinds"] == ["pointercancel", "pointerdown", "pointermove",
                                  "pointerup"]

    def test_a_browser_without_pointer_events_still_gets_a_mouse_path(self):
        """Deliberately the old path and not a second maintained one: modern
        WebKit, Blink and Gecko all have Pointer Events, and this is for the one
        embedded browser that does not."""
        found = run("""
            report({kinds: Object.keys(canvas._listeners).sort()});
        """, initial=document(), pointers=False)

        assert "mousedown" in found["kinds"]
        assert "pointerdown" in found["kinds"]

    def test_a_cancelled_pointer_puts_the_region_back(self):
        """A phone call, or a gesture the browser decided was a scroll. The edit
        goes back to where it started rather than stopping halfway."""
        found = run("""
            ks.open();
            drag([100, 100], [300, 300], proxy(), "pointercancel");
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

    def test_a_second_finger_does_not_take_over_the_drag(self):
        """§5.4: one primary pointer edits, and the rest are ignored rather than
        fighting it for the same region."""
        found = run("""
            ks.open();
            canvas.dispatchEvent({type: "pointerdown", target: proxy(), pointerId: 1,
                                  isPrimary: true, button: 0, clientX: 100,
                                  clientY: 100, preventDefault() {}});
            canvas.dispatchEvent({type: "pointerdown", target: proxy(), pointerId: 2,
                                  isPrimary: false, button: 0, clientX: 900,
                                  clientY: 900, preventDefault() {}});
            canvas.dispatchEvent({type: "pointermove", target: proxy(), pointerId: 1,
                                  clientX: 200, clientY: 150, preventDefault() {}});
            canvas.dispatchEvent({type: "pointerup", target: proxy(), pointerId: 1,
                                  preventDefault() {}});
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [135, 105, 415, 410]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


class TestDrawing:
    def test_a_drag_becomes_a_region_at_those_coordinates(self):
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            report();
        """)

        assert found["regions"][0]["bbox"] == [100, 200, 400, 600]

    def test_the_new_region_is_the_selected_one(self):
        """§7.2: the release creates *and* selects, so the next thing typed goes
        into the box that was just drawn."""
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            report({focused: !!el("mc-krea-spatial-prompt").focused});
        """)

        assert found["selected"] == found["regions"][0]["id"]
        assert found["focused"] is True

    def test_a_backwards_drag_is_the_same_region(self):
        found = run("""
            ks.open();
            draw([400, 600], [100, 200]);
            report();
        """)

        assert found["regions"][0]["bbox"] == [100, 200, 400, 600]

    def test_a_drag_off_the_edge_stops_at_it(self):
        found = run("""
            ks.open();
            draw([-300, -300], [1400, 1400]);
            report();
        """)

        assert found["regions"][0]["bbox"] == [0, 0, 1000, 1000]

    def test_a_tap_is_not_a_region(self):
        """And neither is a two-pixel drag, which is a tap with a shaky hand.
        Both are dropped here rather than refused later, where the reason would
        arrive on the finished image instead of under the finger."""
        found = run("""
            ks.open();
            draw([500, 500], [500, 500]);
            draw([500, 500], [503, 504]);
            report();
        """)

        assert found["regions"] == []

    def test_drawing_is_one_region_at_a_time(self):
        """§7.2: the tool disarms itself on release. A mode that stayed on would
        turn the next attempt to move a box into a new box on top of it."""
        found = run("""
            ks.open();
            draw([100, 100], [300, 300]);
            drag([600, 600], [800, 800]);
            report({drawing: ks.state.drawing});
        """)

        assert found["drawing"] is False
        assert len(found["regions"]) == 1

    def test_escape_abandons_the_drag_without_closing_the_editor(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-draw");
            canvas.dispatchEvent({type: "pointerdown", target: canvas, pointerId: 1,
                                  isPrimary: true, button: 0, clientX: 100,
                                  clientY: 100, preventDefault() {}});
            key("Escape");
            report({drawing: ks.state.drawing});
        """)

        assert found["open"] is True
        assert found["drawing"] is False
        assert found["regions"] == []

    def test_add_region_makes_one_without_a_precision_gesture(self):
        """§7.1. The whole point of the button: a touch user does not have to
        drag accurately to get a box at all."""
        found = run("""
            ks.open();
            press("mc-krea-spatial-add");
            report({focused: !!el("mc-krea-spatial-prompt").focused});
        """)

        left, top, right, bottom = found["regions"][0]["bbox"]

        assert 250 <= right - left <= 450
        assert 250 <= bottom - top <= 450
        assert found["selected"] == found["regions"][0]["id"]
        assert found["focused"] is True


# --------------------------------------------------------------------------- #
# Selection, and the region that is buried
# --------------------------------------------------------------------------- #


BURIED = [dict(FACE, id="a", name="A", z=0, bbox=[200, 200, 800, 800]),
          dict(FACE, id="b", name="B", z=1, bbox=[200, 200, 800, 800])]


class TestTheSelectionProxy:
    """§6, and the hard acceptance requirement in it. Hit-testing follows paint
    order, so a region behind another cannot be grabbed where they overlap --
    and raising it on selection would fix the grab by changing the composition,
    which is not a trade the user asked for."""

    def test_selecting_from_the_list_puts_the_handles_on_that_region(self):
        found = run("""
            ks.open();
            ks.select("a");
            report({proxyRegion: proxy().dataset.regionId,
                    proxyHidden: !!proxy().hidden,
                    handles: proxy().querySelectorAll(".mc-krea-spatial-handle").length});
        """, initial=document(BURIED))

        assert found["proxyRegion"] == "a"
        assert found["proxyHidden"] is False
        assert found["handles"] == 8

    def test_dragging_the_overlap_moves_the_buried_region_only(self):
        """Acceptance test A. Two fully overlapping boxes, A behind B, A
        selected from the list, a drag through the middle of both."""
        found = run("""
            ks.open();
            ks.select("a");
            drag([500, 500], [600, 550], proxy());
            report();
        """, initial=document(BURIED))
        boxes = {region["id"]: region["bbox"] for region in found["regions"]}

        assert boxes["a"] == [300, 250, 900, 850]
        assert boxes["b"] == [200, 200, 800, 800]

    def test_selecting_never_changes_the_semantic_z_order(self):
        """§6: the proxy writes the selected region's bbox and nothing else.
        The order the compositor is told about is the order it was drawn in."""
        found = run("""
            ks.open();
            ks.select("a");
            drag([500, 500], [600, 550], proxy());
            report();
        """, initial=document(BURIED))

        assert [region["id"] for region in found["regions"]] == ["a", "b"]
        assert [region["z"] for region in found["regions"]] == [0, 1]

    def test_the_proxy_goes_away_when_nothing_is_selected(self):
        found = run("""
            ks.open();
            ks.select("");
            report({proxyHidden: !!proxy().hidden});
        """, initial=document())

        assert found["proxyHidden"] is True

    def test_selection_reaches_the_list_and_the_inspector_together(self):
        """§6.2: one selection, three surfaces, updated in the same breath."""
        found = run("""
            ks.open();
            ks.select("a");
            report({rowSelected: rows().map((row) =>
                        row.classList.contains("selected") ? row.dataset.regionId : null),
                    inspector: el("mc-krea-spatial-selected-name").textContent,
                    name: el("mc-krea-spatial-name").value});
        """, initial=document(BURIED))

        assert [entry for entry in found["rowSelected"] if entry] == ["a"]
        assert found["inspector"] == "A"
        assert found["name"] == "A"


# --------------------------------------------------------------------------- #
# Moving and resizing
# --------------------------------------------------------------------------- #


class TestMovingAndResizing:
    def test_a_region_can_be_dragged(self):
        found = run("""
            ks.open();
            drag([100, 100], [200, 150], proxy());
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [135, 105, 415, 410]

    def test_an_unselected_region_is_grabbed_by_touching_its_body(self):
        found = run("""
            ks.open();
            ks.select("");
            const box = body.querySelector(".mc-krea-spatial-region");
            drag([100, 100], [200, 150], box);
            report();
        """, initial=document())

        assert found["selected"] == "r1"
        assert found["regions"][0]["bbox"] == [135, 105, 415, 410]

    def test_dragging_a_region_cannot_push_it_off_the_canvas(self):
        found = run("""
            ks.open();
            drag([100, 100], [5000, 5000], proxy());
            report();
        """, initial=document())

        left, top, right, bottom = found["regions"][0]["bbox"]

        assert right == 1000 and bottom == 1000
        assert right - left == 315 - 35
        assert bottom - top == 360 - 55

    def test_a_corner_handle_resizes_rather_than_moves(self):
        found = run("""
            ks.open();
            drag([315, 360], [500, 500], handle("se"));
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [35, 55, 500, 500]

    def test_an_edge_handle_moves_one_edge_only(self):
        """§8.2 asks for eight affordances where practical, because "make this
        wider" is most of what resizing actually is and a corner makes you fix
        the other dimension afterwards."""
        found = run("""
            ks.open();
            drag([315, 200], [500, 900], handle("e"));
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [35, 55, 500, 360]

    def test_an_edge_dragged_past_its_opposite_clamps_instead_of_inverting(self):
        """§8.2. Nothing here may produce an inverted box, so nothing
        downstream has to cope with one."""
        found = run("""
            ks.open();
            drag([315, 360], [10, 10], handle("se"));
            report();
        """, initial=document())
        left, top, right, bottom = found["regions"][0]["bbox"]

        assert right - left == 8
        assert bottom - top == 8
        assert [left, top] == [35, 55]

    def test_a_drag_that_leaves_the_frame_still_belongs_to_the_region(self):
        """§5.2. The contact is captured by the frame, so the coordinates are
        clamped rather than the drag being lost the moment a finger slides off
        the edge."""
        found = run("""
            ks.open();
            canvas.dispatchEvent({type: "pointerdown", target: proxy(), pointerId: 7,
                                  isPrimary: true, button: 0, clientX: 100,
                                  clientY: 100, preventDefault() {}});
            canvas.dispatchEvent({type: "pointermove", target: canvas, pointerId: 7,
                                  clientX: -400, clientY: 2000, preventDefault() {}});
            canvas.dispatchEvent({type: "pointerup", target: canvas, pointerId: 7,
                                  preventDefault() {}});
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [0, 695, 280, 1000]

    def test_arrow_keys_nudge_the_selection(self):
        found = run("""
            ks.open();
            key("ArrowRight");
            key("ArrowDown", canvas, {shiftKey: true});
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [40, 80, 320, 385]


# --------------------------------------------------------------------------- #
# The region list
# --------------------------------------------------------------------------- #


class TestTheRegionList:
    def test_the_list_is_frontmost_first(self):
        """§9: the order a layers panel is read in, and the opposite of the
        order the compositor writes them in."""
        found = run("""
            ks.open();
            report({rows: rows().map((row) => row.dataset.regionId)});
        """, initial=document(BURIED))

        assert found["rows"] == ["b", "a"]
        assert [region["id"] for region in found["regions"]] == ["a", "b"]

    def test_touching_a_row_selects_without_touching_the_frame(self):
        found = run("""
            ks.open();
            clickRow(1);
            report();
        """, initial=document(BURIED))

        assert found["selected"] == "a"

    def test_a_row_carries_its_own_delete(self):
        found = run("""
            ks.open();
            inList(rows()[0].querySelector(".mc-krea-spatial-row-trash"));
            report();
        """, initial=document(BURIED))

        assert [region["id"] for region in found["regions"]] == ["a"]

    def test_delete_removes_the_selected_region(self):
        """Acceptance test E: from the list, with no canvas hit-testing at
        all."""
        found = run("""
            ks.open();
            ks.select("r1");
            press("mc-krea-spatial-delete");
            report();
        """, initial=document())

        assert found["regions"] == []
        assert found["selected"] == ""

    def test_deleting_selects_a_predictable_neighbour(self):
        """§6.2: the next row, otherwise the previous one -- never "whatever
        ends up first", which on a reordered list is a different box each
        time."""
        found = run("""
            ks.open();
            ks.select("b");
            press("mc-krea-spatial-delete");
            report();
        """, initial=document(BURIED))

        assert found["selected"] == "a"

    def test_delete_does_nothing_while_somebody_is_typing(self):
        """The single most annoying way an editor can lose work: backspacing in
        a prompt field and having the region disappear."""
        found = run("""
            ks.open();
            key("Backspace", el("mc-krea-spatial-prompt"));
            report();
        """, initial=document())

        assert len(found["regions"]) == 1

    def test_duplicate_makes_a_second_region_offset_from_the_first(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-duplicate");
            report();
        """, initial=document())

        assert len(found["regions"]) == 2
        assert found["regions"][1]["bbox"] != found["regions"][0]["bbox"]
        assert found["regions"][1]["prompt"] == found["regions"][0]["prompt"]

    def test_z_order_is_renumbered_from_the_visible_order(self):
        """"Forward" means one place forward however the numbers arrived --
        including from a hand-edited document where three regions all claim
        z 0."""
        two = json.dumps([dict(FACE, id="a", z=0),
                          dict(FACE, id="b", z=0),
                          dict(FACE, id="c", z=0)])
        found = run("""
            ks.open();
            ks.select("a");
            press("mc-krea-spatial-raise");
            report();
        """, initial=document(json.loads(two)))

        assert [region["id"] for region in found["regions"]] == ["b", "a", "c"]

    def test_the_front_region_cannot_be_pushed_further_forward(self):
        found = run("""
            ks.open();
            ks.select("r1");
            press("mc-krea-spatial-raise");
            press("mc-krea-spatial-raise");
            report();
        """, initial=document())

        assert [region["id"] for region in found["regions"]] == ["r1"]

    def test_to_front_and_to_back_go_all_the_way(self):
        three = [dict(FACE, id="a", z=0), dict(FACE, id="b", z=1),
                 dict(FACE, id="c", z=2)]
        found = run("""
            ks.open();
            ks.select("a");
            press("mc-krea-spatial-front");
            report({after: ks.ordered().map((r) => r.id)});
        """, initial=document(three))

        assert found["after"] == ["b", "c", "a"]

    def test_reordering_changes_z_and_selecting_does_not(self):
        """§9, stated as two facts about the same list."""
        found = run("""
            ks.open();
            ks.select("a");
            const afterSelect = ks.ordered().map((r) => r.z);
            press("mc-krea-spatial-front");
            report({afterSelect: afterSelect,
                    afterReorder: ks.ordered().map((r) => r.id)});
        """, initial=document(BURIED))

        assert found["afterSelect"] == [0, 1]
        assert found["afterReorder"] == ["b", "a"]


# --------------------------------------------------------------------------- #
# The inspector
# --------------------------------------------------------------------------- #


class TestTheInspector:
    def test_typing_a_prompt_reaches_the_region(self):
        found = run("""
            ks.open();
            field("mc-krea-spatial-prompt", "a red bicycle");
            report();
        """, initial=document())

        assert found["regions"][0]["prompt"] == "a red bicycle"

    def test_editing_a_property_does_not_drop_the_selection(self):
        """§11, and the reason it is called out: an inspector that deselects
        what it is inspecting is one you cannot use."""
        found = run("""
            ks.open();
            field("mc-krea-spatial-prompt", "a red bicycle");
            field("mc-krea-spatial-name", "Bicycle");
            report({proxyRegion: proxy().dataset.regionId});
        """, initial=document())

        assert found["selected"] == "r1"
        assert found["proxyRegion"] == "r1"

    def test_switching_to_a_text_region_reveals_the_text_field(self):
        found = run("""
            ks.open();
            field("mc-krea-spatial-type", "text");
            report({textShown: !el("mc-krea-spatial-text-field").hidden,
                    framingShown: !el("mc-krea-spatial-framing-field").hidden});
        """, initial=document())

        assert found["regions"][0]["type"] == "text"
        assert found["textShown"] is True
        # Framing and angle are about how a subject is shot. A line of type has
        # neither, and offering them would invite a selection that renders
        # nothing.
        assert found["framingShown"] is False

    def test_every_semantic_field_survives_a_round_trip(self):
        """§11 keeps the vocabulary the compositor knows. This is the test that
        a refactor of the panel did not quietly drop one of them."""
        found = run("""
            ks.open();
            field("mc-krea-spatial-type", "text");
            field("mc-krea-spatial-name", "Sign");
            field("mc-krea-spatial-text", "OPEN");
            field("mc-krea-spatial-prompt", "neon, buzzing");
            field("mc-krea-spatial-type", "obj");
            field("mc-krea-spatial-framing", "Wide shot");
            field("mc-krea-spatial-angle", "Low angle");
            press("mc-krea-spatial-save");
            report({document: saved()});
        """, initial=document())
        region = found["document"]["regions"][0]

        assert region["name"] == "Sign"
        assert region["prompt"] == "neon, buzzing"
        assert region["framing"] == "Wide shot"
        assert region["angle"] == "Low angle"

    def test_the_editor_offers_no_coordinates_anywhere(self):
        """Boxes are placed with a finger, a mouse or a pen. A normalized
        coordinate is an implementation detail of the format, and typing one
        into a field is not an interaction this editor has."""
        emitted = {identifier for _tag, identifier in editor_elements()}
        gone = {"mc-krea-spatial-" + name for name in ("x", "y", "w", "h", "bbox")}
        markup = _markup()

        assert emitted & gone == set()
        assert "0\u20131000" not in markup
        assert "0-1000" not in markup

    def test_the_inspector_empties_when_the_selection_goes(self):
        """An inspector still saying "Text" and "How the text should look"
        after the text region it described was deleted is a panel lying about
        what is selected."""
        found = run("""
            ks.open();
            field("mc-krea-spatial-type", "text");
            press("mc-krea-spatial-delete");
            report({name: el("mc-krea-spatial-name").value,
                    type: el("mc-krea-spatial-type").value,
                    label: el("mc-krea-spatial-prompt-label").textContent,
                    title: el("mc-krea-spatial-selected-name").textContent,
                    textShown: !el("mc-krea-spatial-text-field").hidden,
                    framingShown: !el("mc-krea-spatial-framing-field").hidden,
                    promptOff: !!el("mc-krea-spatial-prompt").disabled});
        """, initial=document())

        assert found["name"] == ""
        assert found["type"] == "obj"
        assert found["label"] == "Region prompt"
        assert found["title"] == "nothing selected"
        assert found["textShown"] is False
        assert found["framingShown"] is True
        assert found["promptOff"] is True

# --------------------------------------------------------------------------- #
# Undo, redo, clear all
# --------------------------------------------------------------------------- #


class TestHistory:
    def test_clear_all_is_one_action_and_undo_brings_them_back(self):
        """Acceptance test D, which is also why Clear All needs no confirmation
        dialog standing in front of it."""
        four = [dict(FACE, id="a", z=0), dict(FACE, id="b", z=1),
                dict(FACE, id="c", z=2), dict(FACE, id="d", z=3)]
        found = run("""
            ks.open();
            const before = JSON.stringify(ks.ordered());
            press("mc-krea-spatial-clear");
            const cleared = ks.ordered().length;
            press("mc-krea-spatial-undo");
            report({cleared: cleared, same: JSON.stringify(ks.ordered()) === before});
        """, initial=document(four))

        assert found["cleared"] == 0
        assert found["same"] is True
        assert len(found["regions"]) == 4

    def test_a_completed_drag_is_one_history_entry(self):
        """§10.3, and the difference between a usable undo stack and four
        hundred entries describing one gesture."""
        found = run("""
            ks.open();
            canvas.dispatchEvent({type: "pointerdown", target: proxy(), pointerId: 1,
                                  isPrimary: true, button: 0, clientX: 100,
                                  clientY: 100, preventDefault() {}});
            [120, 140, 160, 180, 200].forEach(function (at) {
                canvas.dispatchEvent({type: "pointermove", target: proxy(), pointerId: 1,
                                      clientX: at, clientY: at, preventDefault() {}});
            });
            canvas.dispatchEvent({type: "pointerup", target: proxy(), pointerId: 1,
                                  preventDefault() {}});
            report();
        """, initial=document())

        assert found["past"] == 1

    def test_a_drag_that_moved_nothing_is_not_an_entry(self):
        found = run("""
            ks.open();
            drag([100, 100], [100, 100], proxy());
            report();
        """, initial=document())

        assert found["past"] == 0

    def test_undo_puts_a_dragged_region_back(self):
        found = run("""
            ks.open();
            drag([100, 100], [200, 150], proxy());
            press("mc-krea-spatial-undo");
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

    def test_redo_puts_it_back_again(self):
        found = run("""
            ks.open();
            drag([100, 100], [200, 150], proxy());
            press("mc-krea-spatial-undo");
            press("mc-krea-spatial-redo");
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [135, 105, 415, 410]

    def test_a_burst_of_typing_is_one_history_entry(self):
        found = run("""
            ks.open();
            field("mc-krea-spatial-prompt", "a");
            field("mc-krea-spatial-prompt", "a r");
            field("mc-krea-spatial-prompt", "a red bicycle");
            report();
        """, initial=document())

        assert found["past"] == 1

    def test_undo_of_a_creation_leaves_no_selection_pointing_at_nothing(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-add");
            press("mc-krea-spatial-undo");
            report({proxyHidden: !!proxy().hidden});
        """)

        assert found["regions"] == []
        assert found["selected"] == ""
        assert found["proxyHidden"] is True

    def test_the_keyboard_shortcuts_undo_and_redo(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-add");
            key("z", canvas, {ctrlKey: true});
            const undone = ks.ordered().length;
            key("z", canvas, {ctrlKey: true, shiftKey: true});
            report({undone: undone});
        """)

        assert found["undone"] == 0
        assert len(found["regions"]) == 1

    def test_ctrl_z_inside_a_text_field_belongs_to_the_text_field(self):
        """Taking it away is how a mistyped prompt costs somebody a region."""
        found = run("""
            ks.open();
            press("mc-krea-spatial-add");
            key("z", el("mc-krea-spatial-prompt"), {ctrlKey: true});
            report();
        """)

        assert len(found["regions"]) == 1

    def test_history_does_not_survive_leaving_the_editor(self):
        """§15: the undo stack is editor-session UI state. It has no business
        outliving the session, and nothing serializes it."""
        found = run("""
            ks.open();
            press("mc-krea-spatial-add");
            ks.save();
            ks.open();
            report();
        """, initial=document())

        assert found["past"] == 0
        assert found["future"] == 0


# --------------------------------------------------------------------------- #
# Saving, leaving
# --------------------------------------------------------------------------- #


class TestSavingAndLeaving:
    def test_save_publishes_the_document_and_tells_gradio(self):
        """Setting .value alone updates the page and tells the server nothing;
        the input event is what carries it, and it is what the state box's
        change handler is wired to."""
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            field("mc-krea-spatial-prompt", "a lighthouse");
            press("mc-krea-spatial-save");
            report({document: saved()});
        """)

        assert found["published"] == 1
        assert found["open"] is False
        assert found["document"]["version"] == 1
        assert found["document"]["regions"][0]["prompt"] == "a lighthouse"
        assert found["document"]["regions"][0]["bbox"] == [100, 200, 400, 600]

    def test_the_saved_canvas_is_the_size_the_image_will_be(self):
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            press("mc-krea-spatial-save");
            report({document: saved()});
        """)

        assert found["document"]["canvas"] == {"width": 1024, "height": 1344,
                                               "grid": "none"}

    def test_a_new_layout_draws_no_grid(self):
        """The guides are an aid, not a default. A frame with lines across it
        is a frame you compose around rather than in."""
        found = run("""
            ks.open();
            report({guides: el("mc-krea-spatial-guides").className,
                    chosen: el("mc-krea-spatial-grid").value});
        """)

        assert "none" in found["guides"].split()
        assert "thirds" not in found["guides"]
        assert found["chosen"] == "none"

    def test_a_grid_somebody_chose_is_kept(self):
        found = run("""
            ks.open();
            commit("mc-krea-spatial-grid", "thirds");
            press("mc-krea-spatial-save");
            report({document: saved(),
                    guides: el("mc-krea-spatial-guides").className});
        """, initial=document())

        assert found["document"]["canvas"]["grid"] == "thirds"
        assert "thirds" in found["guides"].split()

    def test_back_with_nothing_changed_leaves_at_once(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-cancel");
            report();
        """, initial=document())

        assert found["open"] is False
        assert found["confirming"] is False
        assert found["published"] == 0

    def test_back_with_changes_asks_in_the_page(self):
        """§14. A browser confirm() would be a modal dialog put back into an
        editor whose whole point was to stop being one."""
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            press("mc-krea-spatial-cancel");
            report({barShown: !el("mc-krea-spatial-confirm").hidden});
        """, initial=document())

        assert found["confirming"] is True
        assert found["barShown"] is True
        assert found["open"] is True
        assert found["published"] == 0

    def test_keep_editing_returns_to_the_editor_with_the_change_intact(self):
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            press("mc-krea-spatial-cancel");
            press("mc-krea-spatial-keep");
            report();
        """, initial=document())

        assert found["open"] is True
        assert found["confirming"] is False
        assert len(found["regions"]) == 2

    def test_discard_leaves_without_publishing_anything(self):
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            press("mc-krea-spatial-cancel");
            press("mc-krea-spatial-discard");
            report();
        """, initial=document())

        assert found["open"] is False
        assert found["published"] == 0
        assert found["stateBox"] == document()

    def test_escape_with_nothing_in_progress_asks_the_same_question(self):
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            key("Escape");
            report();
        """, initial=document())

        assert found["published"] == 0
        assert found["confirming"] is True

    def test_the_editor_opens_on_what_was_last_saved(self):
        found = run("""
            ks.open();
            report();
        """, initial=document())

        assert [region["id"] for region in found["regions"]] == ["r1"]
        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

    def test_an_existing_layout_opens_untouched(self):
        """Acceptance test G. Opening and saving without editing anything must
        publish the same coordinates, prompts and z-order it was given."""
        three = [dict(FACE, id="a", z=0, prompt="one"),
                 dict(FACE, id="b", z=1, prompt="two", bbox=[400, 400, 700, 900]),
                 dict(FACE, id="c", z=2, prompt="three", bbox=[10, 10, 90, 90])]
        found = run("""
            ks.open();
            press("mc-krea-spatial-save");
            report({document: saved()});
        """, initial=document(three))
        published = found["document"]["regions"]

        assert [region["id"] for region in published] == ["a", "b", "c"]
        assert [region["z"] for region in published] == [0, 1, 2]
        assert [region["prompt"] for region in published] == ["one", "two", "three"]
        assert [region["bbox"] for region in published] == [
            [35, 55, 315, 360], [400, 400, 700, 900], [10, 10, 90, 90]]

    def test_a_document_it_cannot_read_is_not_opened_and_not_overwritten(self):
        """Refused, not repaired. Opening onto an empty canvas would look like
        the layout had been deleted, and saving over it would make that true."""
        found = run("""
            ks.open();
            report();
        """, initial='{"version": 99, "regions": []}')

        assert found["open"] is False
        assert found["published"] == 0
        assert found["stateBox"] == '{"version": 99, "regions": []}'

    def test_new_ids_do_not_collide_with_the_ones_already_in_the_document(self):
        found = run("""
            ks.open();
            draw([600, 600], [800, 800]);
            report();
        """, initial=document([dict(FACE, id="r1"), dict(FACE, id="r2")]))

        identifiers = [region["id"] for region in found["regions"]]

        assert len(set(identifiers)) == len(identifiers)


# --------------------------------------------------------------------------- #
# The frame
# --------------------------------------------------------------------------- #


class TestTheFrame:
    def test_the_canvas_takes_the_shape_of_the_image(self):
        """Acceptance test F: the frame is the generation's aspect ratio, not a
        square and not a layout-specific one."""
        found = run("""
            ks.open();
            report({ratio: canvas.style.aspectRatio,
                    w: canvas.style["--mc-ar-w"], h: canvas.style["--mc-ar-h"]});
        """, initial=document())

        assert found["ratio"] == "1024 / 1344"
        assert [found["w"], found["h"]] == ["1024", "1344"]

    def test_the_label_carries_the_size_the_ratio_and_the_orientation(self):
        """§4.3. The reduced exact ratio, including when it is not a familiar
        one -- a nearest-neighbour "about 3:4" would be a lie about the frame
        somebody is composing inside."""
        found = run("""
            report({square: ks.dimensions(1024, 1024),
                    portrait: ks.dimensions(1024, 1536),
                    landscape: ks.dimensions(1536, 1024),
                    odd: ks.dimensions(1024, 1344)});
        """, initial=document())

        assert found["square"] == "1024 × 1024 · 1:1 · Square"
        assert found["portrait"] == "1024 × 1536 · 2:3 · Portrait"
        assert found["landscape"] == "1536 × 1024 · 3:2 · Landscape"
        assert found["odd"] == "1024 × 1344 · 16:21 · Portrait"

    def test_the_label_is_on_the_page(self):
        found = run("""
            ks.open();
            report({label: el("mc-krea-spatial-size").textContent});
        """, initial=document())

        assert found["label"] == "1024 × 1344 · 16:21 · Portrait"

    def test_a_reshaped_frame_warns_and_changes_nothing(self):
        """§4.4 in the strongest terms it uses: never silently delete layout
        state. Reprojecting would be this file deciding which of somebody's
        boxes deserved to keep its shape."""
        found = run("""
            width.value = "1536";
            height.value = "1024";
            ks.open();
            const warning = el("mc-krea-spatial-warning");
            report({warned: !warning.hidden, text: warning.textContent});
        """, initial=document())

        assert found["warned"] is True
        assert "1024 × 1344 (16:21)" in found["text"]
        assert "1536 × 1024 (3:2)" in found["text"]
        assert "unchanged" in found["text"]
        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

    def test_the_same_shape_at_a_different_size_says_nothing(self):
        found = run("""
            width.value = "1536";
            height.value = "2016";
            ks.open();
            report({warned: !el("mc-krea-spatial-warning").hidden});
        """, initial=document())

        assert found["warned"] is False

    def test_the_frame_follows_a_size_changed_while_the_editor_is_open(self):
        found = run("""
            ks.open();
            width.value = "1536";
            height.value = "1024";
            width.dispatchEvent({type: "change", target: width});
            report({ratio: el("mc-krea-spatial-canvas").style.aspectRatio,
                    label: el("mc-krea-spatial-size").textContent});
        """, initial=document())

        assert found["ratio"] == "1536 / 1024"
        assert found["label"] == "1536 × 1024 · 3:2 · Landscape"

    def test_zoom_never_touches_a_stored_coordinate(self):
        """§4.5: display only, and the frame's CSS width is the whole of it."""
        found = run("""
            ks.open();
            press("mc-krea-spatial-zoom-in");
            press("mc-krea-spatial-zoom-in");
            const zoomed = ks.state.zoom;
            press("mc-krea-spatial-zoom-fit");
            report({zoomed: zoomed,
                    level: el("mc-krea-spatial-zoom-level").textContent});
        """, initial=document())

        assert found["zoomed"] > 1
        assert found["zoom"] == 1
        assert found["level"] == "100%"
        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]


# --------------------------------------------------------------------------- #
# The page it lives in
# --------------------------------------------------------------------------- #


class TestTheWorkspaceIsNotAModal:
    """§3.1. The overlay used to be moved to document.body, because a
    position: fixed modal inside an accordion is one overflow: hidden away from
    being a modal nobody can see. Moving it solved that and bought two copies of
    every id whenever Gradio rebuilt the tab."""

    def test_the_workspace_stays_where_gradio_put_it(self):
        found = run("""
            ks.open();
            report({onBody: body.children.indexOf(
                        el("mc-krea-spatial-workspace")) >= 0,
                    inPage: page.children.indexOf(
                        el("mc-krea-spatial-workspace")) >= 0});
        """, initial=document())

        assert found["onBody"] is False
        assert found["inPage"] is True

    def test_it_is_hidden_until_edit_layout_and_shown_after(self):
        found = run("""
            const before = !!el("mc-krea-spatial-workspace").hidden;
            press("mc-krea-spatial-open");
            report({before: before,
                    after: !!el("mc-krea-spatial-workspace").hidden});
        """, initial=document())

        assert found["before"] is True
        assert found["after"] is False
        assert found["open"] is True

    def test_a_rebuilt_page_does_not_leave_two_of_everything(self):
        """Both copies carry every id, so whichever ``byId`` found first would
        be a coin toss between a wired workspace and an unwired one."""
        found = run("""
            rebuildMarkup();
            ks.wire();
            ks.open();
            draw([100, 200], [400, 600]);
            report();
        """, initial=document())

        assert found["workspaces"] == 1
        assert found["open"] is True
        assert len(found["regions"]) == 2

    def test_a_rebuild_under_an_open_editor_leaves_it_open(self):
        found = run("""
            ks.open();
            rebuildMarkup();
            ks.wire();
            report({shown: !el("mc-krea-spatial-workspace").hidden,
                    rows: rows().length});
        """, initial=document())

        assert found["open"] is True
        assert found["shown"] is True
        assert found["rows"] == 1


class TestFullScreen:
    """The frame is the object being worked on, and on a tablet the page around
    it is most of the screen. Full screen gives the whole display to the editor
    and Back still comes back to txt2img."""

    def test_the_browser_is_asked_first(self):
        """The Fullscreen API is the only mechanism nothing can clip, overlap or
        out-stack -- and the only one that takes the browser's own chrome away,
        which on a tablet is most of what is being asked for."""
        found = run("""
            ks.open();
            fullscreenAnswers("yes");
            press("mc-krea-spatial-full");
            await flush();
            report({promoted: document.fullscreenElement
                        === el("mc-krea-spatial-workspace"),
                    fixed: fullpage(),
                    label: el("mc-krea-spatial-full").textContent});
        """, initial=document())

        assert found["promoted"] is True
        # No fixed-position fallback while the real thing is in force.
        assert found["fixed"] is False
        assert found["label"] == "Exit full screen"

    def test_a_refusal_falls_back_to_a_fixed_block(self):
        """An iframe without allowfullscreen, or a user who said no. §3.1 told
        this editor not to be a fixed overlay, and this is fine for the reason
        the modal was not: it is somewhere the user asked to go."""
        found = run("""
            ks.open();
            fullscreenAnswers("no");
            press("mc-krea-spatial-full");
            await flush();
            report({fixed: fullpage(),
                    label: el("mc-krea-spatial-full").textContent});
        """, initial=document())

        assert found["fixed"] is True
        assert found["label"] == "Exit full screen"

    def test_a_browser_without_the_api_at_all_gets_the_same_fallback(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-full");
            report({fixed: fullpage()});
        """, initial=document())

        assert found["fixed"] is True

    def test_pressing_it_again_comes_back(self):
        found = run("""
            ks.open();
            fullscreenAnswers("yes");
            press("mc-krea-spatial-full");
            await flush();
            press("mc-krea-spatial-full");
            await flush();
            report({promoted: !!document.fullscreenElement,
                    fixed: fullpage(),
                    label: el("mc-krea-spatial-full").textContent});
        """, initial=document())

        assert found["promoted"] is False
        assert found["fixed"] is False
        assert found["label"] == "Full screen"

    def test_leaving_the_editor_leaves_full_screen_with_it(self):
        """Save & Return returns to txt2img, and a txt2img tab still filling the
        screen with a hidden editor would be a tab nobody could use."""
        found = run("""
            ks.open();
            fullscreenAnswers("yes");
            press("mc-krea-spatial-full");
            await flush();
            press("mc-krea-spatial-save");
            await flush();
            report({promoted: !!document.fullscreenElement, fixed: fullpage()});
        """, initial=document())

        assert found["open"] is False
        assert found["promoted"] is False
        assert found["fixed"] is False

    def test_going_full_screen_changes_no_layout_state(self):
        found = run("""
            ks.open();
            const before = ks.serialize();
            press("mc-krea-spatial-full");
            report({same: ks.serialize() === before, past: ks.state.past.length});
        """, initial=document())

        assert found["same"] is True
        assert found["past"] == 0


class TestTheMarkupAndTheFileAgree:
    def test_every_id_the_file_uses_is_an_id_python_emits(self):
        """The one failure mode a fake page cannot catch on its own: a control
        renamed on one side of the wire. Read out of the file's own IDS table,
        so a typo in it is a failure here rather than a dead button in a
        browser."""
        source = SCRIPT.read_text(encoding="utf-8")
        table = re.search(r"const IDS = \{(.*?)\n    \};", source, re.S)
        assert table, "the IDS table could not be found"
        used = {"mc-krea-spatial-" + name for name in
                re.findall(r'P \+ "-([a-z-]+)"', table.group(1))}
        emitted = {identifier for _tag, identifier in editor_elements()}
        # The state box and the Edit button are Gradio's components, not the
        # editor markup's, so they are named by the panel instead.
        external = {"mc-krea-spatial-state", "mc-krea-spatial-open"}

        assert used - external <= emitted

    def test_every_control_the_editor_had_is_still_there(self):
        """A refactor that quietly dropped Framing, or the auto-hint checkbox,
        or Duplicate, would pass every behavioural test above by simply never
        exercising it."""
        emitted = {identifier for _tag, identifier in editor_elements()}
        expected = {
            "name", "type", "text", "text-field", "prompt", "prompt-label",
            "framing", "framing-field", "angle", "angle-field", "auto-hint",
            "list", "draw", "duplicate", "delete", "raise", "lower",
            "save", "cancel", "size", "warning", "canvas", "regions",
            # and everything the refactor added
            "workspace", "proxy", "add", "clear", "undo", "redo", "front",
            "bottom", "grid", "zoom-fit", "zoom-in", "zoom-out", "confirm",
            "keep", "discard", "full",
        }

        assert {"mc-krea-spatial-" + name for name in expected} <= emitted


# --------------------------------------------------------------------------- #
# The compact canvas
# --------------------------------------------------------------------------- #


BACKDROP = {"id": "r2", "name": "Backdrop", "type": "obj", "bbox": [0, 0, 1000, 1000],
            "prompt": "paper sweep", "framing": "", "angle": "", "z": -1}
SIGN = {"id": "r3", "name": "Sign", "type": "text", "bbox": [200, 200, 600, 400],
        "prompt": "a shop sign", "text": "OPEN", "framing": "", "angle": "", "z": 5}


class TestTheCompactCanvas:
    """Position correction in the pipeline, and nothing else.

    Section 6.2 gives this canvas one verb. The tests that matter are therefore
    of two kinds: that the verb works by mouse, pen and finger over whichever
    box is actually on top, and that none of the other verbs is reachable from
    here. The second kind is the one that keeps a shortcut from turning into a
    second editor.
    """

    def test_it_draws_the_regions_the_document_holds(self):
        found = run("report({});", initial=document(regions=(FACE, SIGN)))

        assert found["compactDrawn"] == ["r1", "r3"]

    def test_it_draws_them_in_paint_order_so_the_top_one_is_last(self):
        """Hit testing follows paint order, so the order these are appended in
        *is* the answer to "which box does a pointer land on"."""
        found = run("report({});", initial=document(regions=(SIGN, BACKDROP, FACE)))

        assert found["compactDrawn"] == ["r2", "r1", "r3"]

    def test_dragging_a_region_moves_it(self):
        found = run("""
            compactDrag([100, 100], [200, 250], compactRegion("r1"));
            report({});
        """, initial=document())

        assert found["compactRegions"][0]["bbox"] == [135, 205, 415, 510]

    def test_a_drag_moves_and_never_resizes(self):
        """The size is the one thing a position correction must not change."""
        found = run("""
            compactDrag([100, 100], [400, 90], compactRegion("r1"));
            report({});
        """, initial=document())

        x0, y0, x1, y1 = found["compactRegions"][0]["bbox"]
        assert (x1 - x0, y1 - y0) == (315 - 35, 360 - 55)

    def test_the_topmost_region_under_the_pointer_is_the_one_that_moves(self):
        """Two boxes overlap; the one in front wins, however far back the other
        one's coordinates would also contain the contact point."""
        found = run("""
            compactDrag([300, 300], [350, 300], compactRegion("r3"));
            report({});
        """, initial=document(regions=(BACKDROP, SIGN)))

        moved = {entry["id"]: entry["bbox"] for entry in found["compactRegions"]}
        assert moved["r3"] == [250, 200, 650, 400]
        assert moved["r2"] == [0, 0, 1000, 1000]

    def test_a_drag_that_starts_on_the_frame_moves_nothing(self):
        """Empty canvas is not a region, and a stray press must not pick the
        nearest box and drag it."""
        found = run("""
            compactDrag([900, 900], [500, 500]);
            report({});
        """, initial=document())

        assert found["compactRegions"][0]["bbox"] == [35, 55, 315, 360]

    def test_a_region_cannot_be_dragged_off_the_frame(self):
        found = run("""
            compactDrag([100, 100], [-4000, -4000], compactRegion("r1"));
            report({});
        """, initial=document())

        x0, y0, x1, y1 = found["compactRegions"][0]["bbox"]
        assert (x0, y0) == (0, 0)
        assert (x1 - x0, y1 - y0) == (315 - 35, 360 - 55)

    def test_a_finger_drives_it_by_the_same_path_as_a_mouse(self):
        """One pointer path, so a tablet is not a second implementation with
        its own bugs. The harness runs this one with PointerEvent absent, which
        is the only case that reaches the fallback at all."""
        found = run("""
            compactDrag([100, 100], [200, 250], compactRegion("r1"));
            report({});
        """, initial=document(), pointers=False)

        assert found["compactRegions"][0]["bbox"] == [135, 205, 415, 510]

    # -- Auto Save ---------------------------------------------------------- #

    def test_auto_save_commits_when_the_pointer_is_released(self):
        found = run("""
            compactDrag([100, 100], [200, 250], compactRegion("r1"));
            report({});
        """, initial=document())

        assert found["published"] == 1
        assert json.loads(found["stateBox"])["regions"][0]["bbox"] == [135, 205, 415, 510]
        assert found["compactDirty"] is False

    def test_auto_save_off_changes_the_screen_and_says_it_is_unsaved(self):
        found = run("""
            autoSaveBox.checked = false;
            compactDrag([100, 100], [200, 250], compactRegion("r1"));
            report({});
        """, initial=document())

        assert found["published"] == 0
        assert found["compactDirty"] is True
        assert "Unsaved working layout" in found["compactNote"]
        # The box has moved on screen regardless: an unsaved layout is a real
        # layout somebody is looking at, not a preview of one.
        assert found["compactRegions"][0]["bbox"] == [135, 205, 415, 510]

    def test_save_working_layout_commits_what_is_on_screen(self):
        found = run("""
            autoSaveBox.checked = false;
            compactDrag([100, 100], [200, 250], compactRegion("r1"));
            press("mc-krea-spatial-compact-commit");
            report({});
        """, initial=document())

        assert found["published"] == 1
        assert found["compactDirty"] is False
        assert "Unsaved" not in found["compactNote"]

    def test_a_drag_that_moves_nothing_commits_nothing(self):
        found = run("""
            compactDrag([100, 100], [100, 100], compactRegion("r1"));
            report({});
        """, initial=document())

        assert found["published"] == 0
        assert found["compactPast"] == 0

    def test_a_cancelled_drag_puts_the_region_back(self):
        found = run("""
            compactDrag([100, 100], [400, 400], compactRegion("r1"), "pointercancel");
            report({});
        """, initial=document())

        assert found["compactRegions"][0]["bbox"] == [35, 55, 315, 360]
        assert found["published"] == 0

    # -- Undo --------------------------------------------------------------- #

    def test_undo_reverses_the_last_move(self):
        found = run("""
            compactDrag([100, 100], [200, 250], compactRegion("r1"));
            press("mc-krea-spatial-compact-undo");
            report({});
        """, initial=document())

        assert found["compactRegions"][0]["bbox"] == [35, 55, 315, 360]

    def test_undo_commits_too_while_auto_save_is_on(self):
        """§6.4. Otherwise Undo would leave the screen and the generation
        disagreeing, which is the one thing Auto Save exists to prevent."""
        found = run("""
            compactDrag([100, 100], [200, 250], compactRegion("r1"));
            press("mc-krea-spatial-compact-undo");
            report({});
        """, initial=document())

        assert json.loads(found["stateBox"])["regions"][0]["bbox"] == [35, 55, 315, 360]
        assert found["compactDirty"] is False

    def test_undo_with_auto_save_off_leaves_it_unsaved(self):
        found = run("""
            autoSaveBox.checked = false;
            compactDrag([100, 100], [200, 250], compactRegion("r1"));
            press("mc-krea-spatial-compact-undo");
            report({});
        """, initial=document())

        assert found["published"] == 0
        assert found["compactDirty"] is True

    def test_one_history_entry_per_completed_drag(self):
        found = run("""
            compactDrag([100, 100], [200, 200], compactRegion("r1"));
            compactDrag([200, 200], [300, 300], compactRegion("r1"));
            report({});
        """, initial=document())

        assert found["compactPast"] == 2

    def test_undo_with_nothing_to_undo_does_nothing(self):
        found = run("""
            press("mc-krea-spatial-compact-undo");
            report({});
        """, initial=document())

        assert found["compactRegions"][0]["bbox"] == [35, 55, 315, 360]
        assert found["published"] == 0

    def test_the_history_is_bounded(self):
        found = run("""
            for (let at = 0; at < 60; at += 1) {
                compactDrag([100 + at, 100], [101 + at, 100], compactRegion("r1"));
            }
            report({});
        """, initial=document())

        assert found["compactPast"] == 25

    # -- what it deliberately cannot do ------------------------------------- #

    def test_it_offers_no_way_to_create_or_delete_a_region(self):
        """Section 6.2 lists what the compact level must not have. Asserted
        against the markup, because a control that is not in the page cannot be
        reached by any interaction a test forgot to try."""
        import model_chain_krea_creative as creative_script

        markup = creative_script.spatial_compact()
        for forbidden in ("add", "draw", "clear", "delete", "duplicate",
                          "handle", "framing", "angle", "front", "lower"):
            assert f'id="mc-krea-spatial-compact-{forbidden}"' not in markup

    def test_the_full_editor_still_owns_every_other_operation(self):
        """The compact canvas is a shortcut into the same layout state, not a
        second layout system: the operations it lacks are all still there."""
        found = dict(editor_elements())
        ids = set(editor_elements_ids())

        for kept in ("mc-krea-spatial-add", "mc-krea-spatial-draw",
                     "mc-krea-spatial-delete", "mc-krea-spatial-duplicate",
                     "mc-krea-spatial-raise", "mc-krea-spatial-lower",
                     "mc-krea-spatial-framing", "mc-krea-spatial-angle",
                     "mc-krea-spatial-type", "mc-krea-spatial-save"):
            assert kept in ids
        assert found

    # -- staying in step with the other canvas ------------------------------ #

    def test_saving_the_full_editor_refreshes_the_compact_canvas(self):
        """Two views of one document. The editor writing one must not leave the
        other showing the layout as it was before."""
        found = run("""
            ks.open();
            draw([500, 500], [800, 800]);
            ks.save();
            report({});
        """, initial=document())

        assert len(found["compactRegions"]) == 2

    def test_a_reload_never_lands_on_an_unsaved_move(self):
        """`onAfterUiUpdate` fires for reasons that have nothing to do with this
        canvas. Re-reading the document under somebody's unsaved work would
        throw it away without saying so."""
        found = run("""
            autoSaveBox.checked = false;
            compactDrag([100, 100], [200, 250], compactRegion("r1"));
            ks.wireCompact();
            report({});
        """, initial=document())

        assert found["compactRegions"][0]["bbox"] == [135, 205, 415, 510]
        assert found["compactDirty"] is True

    def test_an_unreadable_document_leaves_the_canvas_alone(self):
        """Refused, not repaired -- the same rule the editor follows. Python
        says why in words on the panel."""
        found = run("report({});", initial='{"version": 99}')

        assert found["compactRegions"] == []
        assert found["published"] == 0


class TestRegionLiteralFields:
    """Two boxes per region for text no language model may rewrite.

    The compact canvas has none of this and must not grow any -- section 6 of
    the Literal Prompts intent puts region literals in the full editor and
    leaves the compact canvas position-only. What is tested here is that the
    editor carries them without ever interpreting them: it stores two strings,
    writes two strings, and merges nothing. Python does the merging, once, at
    generation time.
    """

    REGION = {"id": "r1", "name": "Sub", "type": "obj", "bbox": [100, 100, 500, 500],
              "prompt": "astronaut holding a flower",
              "literal_prefix": "<lora:a:1>", "literal_suffix": "__grain__", "z": 0}

    def test_the_editor_reads_a_region_s_literal_fields(self):
        found = run("""
            ks.open();
            report({prefix: el("mc-krea-spatial-literal-prefix").value,
                    suffix: el("mc-krea-spatial-literal-suffix").value});
        """, initial=document(regions=(self.REGION,)))

        assert found["prefix"] == "<lora:a:1>"
        assert found["suffix"] == "__grain__"

    def test_typing_in_one_reaches_the_saved_document(self):
        found = run("""
            ks.open();
            const box = el("mc-krea-spatial-literal-prefix");
            box.value = "<lora:typed:1>";
            box.dispatchEvent({type: "input", target: box});
            ks.save();
            report({});
        """, initial=document(regions=(self.REGION,)))

        saved = json.loads(found["stateBox"])["regions"][0]
        assert saved["literal_prefix"] == "<lora:typed:1>"

    def test_a_region_with_no_literals_serializes_as_it_always_did(self):
        """An absent key and not an empty string, so a layout drawn before
        these fields existed round-trips to the same bytes."""
        found = run("""
            ks.open();
            ks.save();
            report({});
        """, initial=document())

        saved = json.loads(found["stateBox"])["regions"][0]
        assert "literal_prefix" not in saved
        assert "literal_suffix" not in saved

    def test_a_new_region_starts_with_both_fields_empty(self):
        found = run("""
            ks.open();
            ks.add();
            ks.save();
            report({});
        """, initial=document())

        saved = json.loads(found["stateBox"])["regions"]
        assert all("literal_prefix" not in entry for entry in saved)

    def test_duplicating_a_region_copies_its_literals(self):
        found = run("""
            ks.open();
            ks.select("r1");
            ks.duplicate();
            ks.save();
            report({});
        """, initial=document(regions=(self.REGION,)))

        saved = json.loads(found["stateBox"])["regions"]
        assert len(saved) == 2
        assert all(entry["literal_prefix"] == "<lora:a:1>" for entry in saved)

    def test_selecting_another_region_redraws_both_fields(self):
        """The inspector is one set of controls for whichever region is
        selected, so a field left showing the previous region's text would be
        the fastest way to put a LoRA in the wrong box."""
        plain = {"id": "r2", "name": "Plain", "type": "obj",
                 "bbox": [600, 600, 900, 900], "prompt": "a lamp", "z": 1}
        found = run("""
            ks.open();
            ks.select("r1");
            ks.select("r2");
            report({prefix: el("mc-krea-spatial-literal-prefix").value,
                    suffix: el("mc-krea-spatial-literal-suffix").value});
        """, initial=document(regions=(self.REGION, plain)))

        assert found["prefix"] == ""
        assert found["suffix"] == ""

    def test_the_editor_never_parses_a_bracket_out_of_a_field(self):
        """It stores what was typed. Every decision about what a payload is,
        which side it goes and how it merges with the region prompt belongs to
        Python, and a browser that got a vote would be the second parser the
        design intent forbids."""
        found = run("""
            ks.open();
            const box = el("mc-krea-spatial-literal-prefix");
            box.value = "+[[not parsed here]]";
            box.dispatchEvent({type: "input", target: box});
            ks.save();
            report({});
        """, initial=document(regions=(self.REGION,)))

        saved = json.loads(found["stateBox"])["regions"][0]
        assert saved["literal_prefix"] == "+[[not parsed here]]"
        assert saved["prompt"] == "astronaut holding a flower"

    def test_the_compact_canvas_offers_no_literal_control(self):
        """Section 6 again, asserted against the markup: the compact canvas
        stays position-only."""
        import model_chain_krea_creative as creative_script

        markup = creative_script.spatial_compact()

        assert "literal" not in markup.casefold()
