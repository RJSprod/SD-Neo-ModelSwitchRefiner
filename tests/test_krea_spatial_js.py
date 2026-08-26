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
import tempfile
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
            // [data-mc-spatial-path] -- how the takeover finds what it marked.
            if (part.startsWith("[") && part.endsWith("]")) {
                return element.getAttribute(part.slice(1, -1)) !== null;
            }
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
    removeAttribute(name) { delete this.attributes[name]; }
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
// The takeover marks every ancestor between the workspace and the txt2img tab,
// so the fake page has one: without it the walk runs to the top and marks
// nothing, which is a different thing from the browser does.
const tab = body.appendChild(new El("div"));
tab.id = "tab_txt2img";
const page = tab.appendChild(new El("div"));

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

// The rest of the txt2img tab the workspace shows rather than copies: the
// prompt, the two Literal boxes, the two feature toggles, the composition radio
// and the result gallery. Every one of these is a component that already
// exists, and the point of the tests below is that the workspace writes *these*
// rather than keeping a value of its own.
function hostBox(id, tag, value) {
    const field = new El(tag || "textarea");
    field.value = value === undefined ? "" : value;
    make("div", id).appendChild(field);
    return field;
}

const hostPrompt = hostBox("txt2img_prompt", "textarea", "a lighthouse");
const hostLiteralPlus = hostBox("mc-krea-creative-literal-positive", "textarea", "");
const hostLiteralMinus = hostBox("mc-krea-creative-literal-negative", "textarea", "");

function hostToggle(id, on) {
    const box = new El("input");
    box.type = "checkbox";
    box.checked = !!on;
    make("div", id).appendChild(box);
    return box;
}

const hostSpatial = hostToggle("mc-krea-creative-spatial-toggle", true);
const hostCreative = hostToggle("mc-krea-creative-toggle", false);

// Gradio renders a Radio as inputs inside labels; which of `value` and the
// label text carries the choice has moved between versions, so the fake page
// carries both and the file under test is free to read either.
const composeHolder = make("div", "mc-krea-creative-spatial-compose");
const composeInputs = {};
["smart", "direct"].forEach(function (mode) {
    const label = new El("label");
    const input = new El("input");
    input.type = "radio";
    input.value = mode;
    input.checked = mode === "smart";
    label.appendChild(input);
    label._text = mode === "smart" ? "Smart Spatial Compose" : "Direct BBOX Merge";
    composeHolder.appendChild(label);
    composeInputs[mode] = input;
});

// The result gallery and the area the host draws its progress bar into.
const gallery = make("div", "txt2img_gallery");
const results = make("div", "txt2img_results");
results.appendChild(gallery);

function showImages(sources) {
    gallery.children = [];
    sources.forEach(function (src) {
        const image = new El("img");
        image.src = src;
        gallery.appendChild(image);
    });
    mutate();
}

// What javascript/progressbar.js builds and tears down around a generation.
function showProgress(width, said) {
    const holder = new El("div");
    holder.className = "progressDiv";
    const bar = new El("div");
    bar.className = "progress";
    bar.style.width = width;
    bar.textContent = said;
    holder.appendChild(bar);
    results.appendChild(holder);
    mutate();
}

function clearProgress() {
    results.querySelectorAll(".progressDiv").forEach(function (entry) {
        results.removeChild(entry);
    });
    mutate();
}

// A MutationObserver that observes nothing and fires when a test says the page
// changed. It exists so that the observer path in the file under test is the
// path the tests exercise -- and so that "no timer was armed" stays a true
// statement about a gallery that keeps up with the host.
let observers = [];
globalThis.MutationObserver = function (fn) {
    this.observe = function () { observers.push(fn); };
    this.disconnect = function () {
        observers = observers.filter((entry) => entry !== fn);
    };
};

function mutate() {
    observers.slice().forEach((fn) => fn([]));
}

// Present so a file that went looking for it would find it. Nothing here may
// *listen* to it -- a generation that a browser can delay is the arrangement
// this whole design removed -- and §16.4 is the one thing allowed to press it:
// once, inside the click the user made, with nothing waiting on the result.
const generate = make("button", "txt2img_generate");
let generateListeners = 0;
let generatePresses = 0;
generate.addEventListener = function () { generateListeners += 1; };
generate.click = function () { generatePresses += 1; };

// A page with a scroll position, because the takeover changes how tall the
// page is and the workspace's job on the way in and the way out is to leave
// somebody looking at the right part of it.
const sheet = new El("html");
sheet.scrollTop = 0;

globalThis.document = {
    body: body,
    documentElement: sheet,
    scrollingElement: sheet,
    dataset: {},
    readyState: "complete",
    createElement: (tag) => new El(tag),
    querySelector: (selector) => body.querySelector(selector),
    querySelectorAll: (selector) => body.querySelectorAll(selector),
    getElementById: (id) => body.querySelector("#" + id),
    addEventListener(kind, fn) { (docListeners[kind] = docListeners[kind] || []).push(fn); },
};
const docListeners = {};

globalThis.window = globalThis;
globalThis.scrollTo = (_x, top) => { sheet.scrollTop = top; };
globalThis.pageYOffset = 0;
function pageTop() { return sheet.scrollTop; }
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

// §3.1's takeover, as a test can see it: the class on the tab, and the marks on
// the chain of ancestors between the tab and the workspace.
function taken() {
    return tab.classList.contains("mc-krea-spatial-taken");
}

function marked() {
    return body.querySelectorAll("[data-mc-spatial-path]").length;
}

// A palette press, and whether it moves before it is let go: a tap places the
// shape centred, a drag places it where it is dropped. Delivered to the
// workspace, which is where the file listens and where the capture is taken.
function palette(shape) {
    return body.querySelectorAll(".mc-krea-spatial-shape-button")
        .filter((entry) => entry.dataset.shape === shape)[0] || null;
}

function place(shape, to, palette_) {
    const workspace = el("mc-krea-spatial-workspace");
    const button = palette_ || palette(shape);
    workspace.dispatchEvent({type: "pointerdown", target: button, pointerId: 3,
                             isPrimary: true, button: 0, clientX: 0, clientY: 0,
                             preventDefault() {}});
    if (to) {
        workspace.dispatchEvent({type: "pointermove", target: button, pointerId: 3,
                                 clientX: to[0], clientY: to[1], preventDefault() {}});
        workspace.dispatchEvent({type: "pointerup", target: button, pointerId: 3,
                                 clientX: to[0], clientY: to[1], preventDefault() {}});
        return;
    }
    workspace.dispatchEvent({type: "pointerup", target: button, pointerId: 3,
                             clientX: 0, clientY: 0, preventDefault() {}});
}

function panelBody(key) {
    return el("mc-krea-spatial-panel-" + key + "-body");
}

function panelShown(key) {
    return !el("mc-krea-spatial-panel-" + key).hidden;
}

function panelOpen(key) {
    return !panelBody(key).hidden;
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

// One reorder gesture, on either list. The items are given boxes first --
// 40px tall, stacked from the top -- because the file works out what a contact
// is over from the items' own rectangles rather than from elementFromPoint,
// and a fake DOM hands every element the same default box.
function laid(kind) {
    const items = kind === "row"
        ? rows()
        : el("mc-krea-spatial-rail")
            .querySelectorAll(".mc-krea-spatial-widget")
            .filter((entry) => !entry.hidden);
    items.forEach(function (element, at) {
        element._rect = {left: 0, top: at * 40, width: 300, height: 40};
    });
    return items;
}

function slide(kind, from, to) {
    const items = laid(kind);
    const holder = el(kind === "row" ? "mc-krea-spatial-list" : "mc-krea-spatial-rail");
    const source = kind === "row" ? items[from]
        : items[from].querySelector(".mc-krea-spatial-widget-grip");
    // Past the middle of the target, on the side the move is going, which is
    // where a hand actually lets go.
    const landing = to * 40 + (to > from ? 30 : 10);
    holder.dispatchEvent({type: "pointerdown", target: source, pointerId: 9,
                          button: 0, clientY: from * 40 + 20, preventDefault() {}});
    holder.dispatchEvent({type: "pointermove", target: source, pointerId: 9,
                          clientY: landing, preventDefault() {}});
    holder.dispatchEvent({type: "pointerup", target: source, pointerId: 9,
                          clientY: landing, preventDefault() {}});
    laid(kind);
}

function railOrder() {
    return ks.panelOrder();
}

// The list's own keydown, delegated the same way its clicks are: Alt with an
// arrow is the reorder a hand that cannot drag still has.
function rowKey(index, name, extra) {
    const row = rows()[index];
    el("mc-krea-spatial-list").dispatchEvent(Object.assign(
        {type: "keydown", key: name, target: row, preventDefault() {}}, extra || {}));
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
    generatePresses: generatePresses,
    regions: ks.ordered().map(function (region) {
        return {id: region.id, bbox: region.bbox, type: region.type,
                name: region.name, prompt: region.prompt, z: region.z,
                shape: region.shape, rotation: region.rotation};
    }),
    selected: ks.state.selected,
    past: ks.state.past.length,
    future: ks.state.future.length,
    zoom: ks.state.zoom,
    confirming: ks.state.confirming,
    published: published.length,
    stateBox: stateBox.value,
    taken: taken(),
    marked: marked(),
    popup: ks.state.popup,
    hidden: Object.keys(ks.state.hidden).filter((key) => ks.state.hidden[key]),
    collapsed: Object.keys(ks.state.collapsed).filter((key) => ks.state.collapsed[key]),
    shots: ks.state.shots,
    shotAt: ks.state.shotAt,
    hostPrompt: hostPrompt.value,
    hostLiteralPlus: hostLiteralPlus.value,
    hostLiteralMinus: hostLiteralMinus.value,
    hostSpatial: hostSpatial.checked,
    hostCreative: hostCreative.checked,
    hostMode: composeInputs.direct.checked ? "direct" : "smart",
    hostWidth: width.value,
    hostHeight: height.value,
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
    """The harness plus one test body, executed by node, and its report read back.

    Written to a file rather than passed with ``-e``: the workspace markup and
    the file under test together are comfortably past the length a command line
    will carry, and a test suite that fails with "argument list too long" the
    day the editor grows is a test suite measuring the wrong thing.
    """
    harness = (
        HARNESS.replace("SOURCE", SCRIPT.read_text(encoding="utf-8"))
        .replace("BODY", script)
        .replace("TREE", json.dumps(editor_tree()))
        .replace("POINTERS", "true" if pointers else "false")
        .replace("INITIAL", json.dumps(initial))
    )
    with tempfile.TemporaryDirectory() as room:
        entry = Path(room) / "harness.mjs"
        entry.write_text(harness, encoding="utf-8")
        result = subprocess.run(["node", str(entry)],
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

    def test_it_listens_to_nothing_on_the_generate_button(self):
        """The rule, stated the way it survives §16.

        The file used not to know where Generate was at all, and that was one
        way of guaranteeing it could never wait for one. §16.4 asks for a
        Generate button in the Gallery widget, which means the file now knows
        the id -- so the property is defended where it actually lives: it
        installs no listener on that button, so there is no click for it to
        swallow, hold or replay.
        """
        found = run("""
            ks.open();
            ks.save();
            report();
        """, initial=document())

        assert found["generateListeners"] == 0

    def test_it_presses_generate_only_when_somebody_presses_generate(self):
        """§16.4: the host's own button, once, inside the user's own click.

        Not a second endpoint, not a queued click and not a timer -- and
        nothing in the editor's own life (opening, editing, saving) touches it.
        """
        found = run("""
            ks.open();
            draw([100, 100], [400, 400]);
            ks.save();
            const before = generatePresses;
            press("mc-krea-spatial-generate");
            advance(60 * 60 * 1000);
            await flush();
            report({before: before});
        """, initial=document())

        assert found["before"] == 0
        assert found["generatePresses"] == 1
        assert found["timers"] == 0

    def test_the_gallery_hears_about_results_rather_than_asking(self):
        """§16.5. A MutationObserver, and no interval anywhere near it: the
        difference between being told the gallery changed and asking sixty
        times a minute in a tab nobody is looking at."""
        found = run("""
            ks.open();
            showImages(["one.png", "two.png"]);
            advance(60 * 60 * 1000);
            await flush();
            report();
        """, initial=document())

        assert found["shots"] == ["one.png", "two.png"]
        assert found["timers"] == 0

    def test_re_wiring_does_not_stack_up_document_listeners(self):
        """``onAfterUiUpdate`` fires on most interactions, and ``document`` has
        no dataset for a flag to live on -- so the guard that makes every other
        listener idempotent cannot cover them. One is left: a keystroke, which
        has to be heard with the focus anywhere in the page. Pointer capture
        removed the rest, and §3.1's takeover removed the fullscreenchange
        listener that used to be the second."""
        found = run("""
            const first = Object.keys(docListeners)
                .reduce((total, kind) => total + docListeners[kind].length, 0);
            ks.wire(); ks.wire(); ks.wire();
            report({first: first});
        """, initial=document())

        assert found["first"] == 1
        assert found["documentListeners"] == 1

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
        """Four pointer kinds and a dblclick, which is not a fifth input path:
        a double-click is a gesture Pointer Events does not report, and it is
        read rather than driven -- nothing about moving, resizing or drawing
        depends on it."""
        found = run("""
            report({kinds: Object.keys(canvas._listeners).sort()});
        """, initial=document())

        assert found["kinds"] == ["dblclick", "pointercancel", "pointerdown",
                                  "pointermove", "pointerup"]
        assert "mousedown" not in found["kinds"]
        assert "mousemove" not in found["kinds"]

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

    def test_quick_add_makes_a_box_without_a_precision_gesture(self):
        """§10.2, acceptance test C's rectangle half. The whole point of the
        palette: a touch user does not have to drag accurately to get a box at
        all, and the one they get is selected with the cursor in its prompt."""
        found = run("""
            ks.open();
            press("mc-krea-spatial-quick");
            place("rect", null);
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
    def test_the_list_is_in_the_order_the_prompt_is_written_in(self):
        """It used to be reversed -- frontmost first, the way a layers panel
        reads -- which meant the list and the composed prompt disagreed about
        which region came first, on the list that is the thing somebody drags
        to reorder them."""
        found = run("""
            ks.open();
            report({rows: rows().map((row) => row.dataset.regionId)});
        """, initial=document(BURIED))

        assert found["rows"] == ["a", "b"]
        assert [region["id"] for region in found["regions"]] == ["a", "b"]

    def test_a_row_is_picked_up_by_a_pointer_and_not_by_an_api(self):
        """§2.3. The rows used to be HTML5 `draggable`, which is mouse-only in
        practice -- a finger produces no drag events at all, and the grip's own
        `touch-action: none` suppressed the long-press Android used to
        synthesise one from. The rows looked draggable and were not."""
        found = run("""
            ks.open();
            report({draggable: rows().map((row) => row.draggable),
                    kinds: Object.keys(el("mc-krea-spatial-list")._listeners).sort()});
        """, initial=document(BURIED))

        assert found["draggable"] == [None, None]
        assert found["kinds"] == ["click", "keydown", "pointercancel",
                                  "pointerdown", "pointermove", "pointerup"]

    def test_touching_a_row_selects_without_touching_the_frame(self):
        found = run("""
            ks.open();
            clickRow(1);
            report();
        """, initial=document(BURIED))

        assert found["selected"] == "b"

    def test_a_row_carries_its_own_delete(self):
        found = run("""
            ks.open();
            inList(rows()[0].querySelector(".mc-krea-spatial-row-trash"));
            report();
        """, initial=document(BURIED))

        assert [region["id"] for region in found["regions"]] == ["b"]

    def test_a_row_dropped_higher_moves_earlier_in_the_prompt(self):
        """The whole feature in one assertion: the list is prompt order, so a
        row moved up is a region written earlier."""
        found = run("""
            ks.open();
            ks.reorder("b", "a", true);
            report({rows: rows().map((row) => row.dataset.regionId)});
        """, initial=document(BURIED))

        assert found["rows"] == ["b", "a"]
        assert [region["id"] for region in found["regions"]] == ["b", "a"]

    def test_a_row_dropped_below_the_last_one_goes_last(self):
        found = run("""
            ks.open();
            ks.reorder("a", "b", false);
            report({rows: rows().map((row) => row.dataset.regionId)});
        """, initial=document(BURIED))

        assert found["rows"] == ["b", "a"]

    def test_the_move_renumbers_rather_than_adjusts(self):
        """A layout hand-edited elsewhere can arrive with every region claiming
        z 0, and "one place later" has to mean one place later in what somebody
        is looking at rather than in arithmetic nobody can see."""
        flat = [dict(FACE, id="a", name="A", z=0, bbox=[10, 10, 200, 200]),
                dict(FACE, id="b", name="B", z=0, bbox=[10, 10, 200, 200]),
                dict(FACE, id="c", name="C", z=0, bbox=[10, 10, 200, 200])]
        found = run("""
            ks.open();
            ks.reorder("c", "a", true);
            report({z: ks.ordered().map((region) => region.z)});
        """, initial=document(flat))

        assert found["z"] == [0, 1, 2]
        assert [region["id"] for region in found["regions"]] == ["c", "a", "b"]

    def test_dropping_a_row_on_itself_changes_nothing(self):
        found = run("""
            ks.open();
            const before = ks.serialize();
            const moved = ks.reorder("a", "a", true);
            report({moved: moved, same: before === ks.serialize()});
        """, initial=document(BURIED))

        assert found["moved"] is False
        assert found["same"] is True

    def test_a_move_can_be_undone(self):
        """It goes through the same history every other edit does."""
        found = run("""
            ks.open();
            ks.reorder("b", "a", true);
            ks.undo();
            report({rows: rows().map((row) => row.dataset.regionId)});
        """, initial=document(BURIED))

        assert found["rows"] == ["a", "b"]

    def test_the_order_is_what_the_composed_prompt_sees(self):
        """`serialize()` is what travels with the generation, and the elements
        array is built from the same ordering the list draws."""
        found = run("""
            ks.open();
            ks.reorder("b", "a", true);
            report({saved: ks.serialize()});
        """, initial=document(BURIED))

        import json as _json

        regions = _json.loads(found["saved"])["regions"]
        assert [region["id"] for region in regions] == ["b", "a"]

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
        """One place forward means one place forward however the numbers
        arrived -- including from a hand-edited document where three regions
        all claim z 0.

        §14.2 leaves Front and Back on the panel and makes the single step the
        list's own Alt+Arrow, which is the gesture this drives.
        """
        three = json.dumps([dict(FACE, id="a", z=0),
                            dict(FACE, id="b", z=0),
                            dict(FACE, id="c", z=0)])
        found = run("""
            ks.open();
            ks.select("a");
            rowKey(0, "ArrowDown", {altKey: true});
            report();
        """, initial=document(json.loads(three)))

        assert [region["id"] for region in found["regions"]] == ["b", "a", "c"]

    def test_the_front_region_cannot_be_pushed_further_forward(self):
        found = run("""
            ks.open();
            ks.select("r1");
            press("mc-krea-spatial-front");
            press("mc-krea-spatial-front");
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
        """§8.3 offers numeric X/Y/W/H. They are not here, for the third and
        final time, and the round trip is why this test names them.

        Boxes are placed with a finger, a mouse or a pen. A normalized
        coordinate is an implementation detail of the storage format rather
        than something anybody composes in, and a number box invites people to
        think in a unit the picture does not have. Nothing about validation
        changed -- boxes are still clamped and ordered exactly as before. What
        is gone is the way of typing one, and the way of reading one off a
        layer row.
        """
        markup = _markup()
        emitted = {identifier for _tag, identifier in editor_elements()}
        gone = {"mc-krea-spatial-" + name for name in
                ("x", "y", "w", "h", "bbox", "bbox-x", "bbox-y", "bbox-w",
                 "bbox-h")}

        assert emitted & gone == set()
        assert "0\u20131000" not in markup
        assert "0-1000" not in markup
        assert 'type="number"' not in markup.split("mc-krea-spatial-panel-session")[0]

    def test_a_layer_row_names_a_region_rather_than_measuring_it(self):
        found = run("""
            ks.open();
            report({row: rows()[0].textContent});
        """, initial=document())

        assert "280" not in found["row"]
        assert "×" not in found["row"]
        assert "Face" in found["row"]

    def test_the_inspector_empties_when_the_selection_goes(self):
        """An inspector still saying "Text" and describing a text region after
        that region was deleted is a panel lying about what is selected."""
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
        assert found["label"] == "Prompt"
        assert found["title"] == ""
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
            place("rect", null);
            press("mc-krea-spatial-undo");
            report({proxyHidden: !!proxy().hidden});
        """)

        assert found["regions"] == []
        assert found["selected"] == ""
        assert found["proxyHidden"] is True

    def test_the_keyboard_shortcuts_undo_and_redo(self):
        found = run("""
            ks.open();
            place("rect", null);
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
            place("rect", null);
            key("z", el("mc-krea-spatial-prompt"), {ctrlKey: true});
            report();
        """)

        assert len(found["regions"]) == 1

    def test_history_does_not_survive_leaving_the_editor(self):
        """§15: the undo stack is editor-session UI state. It has no business
        outliving the session, and nothing serializes it."""
        found = run("""
            ks.open();
            place("rect", null);
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
            autoSaveBox.checked = false;
            ks.open();
            draw([100, 200], [400, 600]);
            field("mc-krea-spatial-prompt", "a lighthouse");
            press("mc-krea-spatial-save");
            report({document: saved()});
        """)

        assert found["published"] == 1
        # §3.2: Save commits and stays. §3.3 is why -- the whole point of the
        # workspace is that somebody can save, generate, edit and save again
        # without it closing under them once.
        assert found["open"] is True
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
            autoSaveBox.checked = false;
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
            autoSaveBox.checked = false;
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
            autoSaveBox.checked = false;
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

    def test_the_bar_carries_the_ratio_and_the_session_widget_the_rest(self):
        """§18.2. The ratio beside the canvas, where it costs nothing; the
        long form in the Session widget, which is collapsible and hideable.
        No large informational string anywhere in the action bar."""
        found = run("""
            ks.open();
            report({aspect: el("mc-krea-spatial-aspect").textContent,
                    frame: el("mc-krea-spatial-fact-frame").textContent,
                    ratio: el("mc-krea-spatial-fact-ratio").textContent});
        """, initial=document())

        assert found["aspect"] == "16:21"
        assert found["frame"] == "1024 × 1344"
        assert found["ratio"] == "16:21 · Portrait"

    def test_a_reshaped_frame_says_so_in_session_and_changes_nothing(self):
        """§18.3 in the strongest terms the design uses: never silently delete
        layout state. Reprojecting would be this file deciding which of
        somebody's boxes deserved to keep its shape."""
        found = run("""
            width.value = "1536";
            height.value = "1024";
            ks.open();
            report({said: el("mc-krea-spatial-fact-state").textContent});
        """, initial=document())

        assert "1024 × 1344 (16:21)" in found["said"]
        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

    def test_the_same_shape_at_a_different_size_says_nothing(self):
        found = run("""
            width.value = "1536";
            height.value = "2016";
            ks.open();
            report({said: el("mc-krea-spatial-fact-state").textContent});
        """, initial=document())

        assert found["said"] == "Saved"

    def test_the_frame_follows_a_size_changed_while_the_workspace_is_open(self):
        """Acceptance test L: 1:1 to 2:3 while the workspace is open, with the
        normalized regions untouched."""
        found = run("""
            width.value = "1024"; height.value = "1024";
            ks.open();
            const before = el("mc-krea-spatial-canvas").style.aspectRatio;
            width.value = "1024";
            height.value = "1536";
            height.dispatchEvent({type: "change", target: height});
            report({before: before,
                    ratio: el("mc-krea-spatial-canvas").style.aspectRatio,
                    aspect: el("mc-krea-spatial-aspect").textContent});
        """, initial=document())

        assert found["before"] == "1024 / 1024"
        assert found["ratio"] == "1024 / 1536"
        assert found["aspect"] == "2:3"
        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

    def test_the_session_widget_can_change_the_generation_size(self):
        """§17's optional half, written into the host's own number fields so
        that the change is the change the sliders make."""
        found = run("""
            ks.open();
            commit("mc-krea-spatial-size-height", "1536");
            report({ratio: el("mc-krea-spatial-canvas").style.aspectRatio});
        """, initial=document())

        assert found["hostHeight"] == "1536"
        assert found["ratio"] == "1024 / 1536"

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


class TestTheTakeover:
    """§3.1. "Full Screen" means the Spatial workspace has the txt2img work
    area -- not that the browser has the screen. The Fullscreen API is a
    non-goal (§29), and what it actually bought is bought here without taking
    the browser's own chrome, Back button or address bar away.

    Nothing is moved (§22.4): the workspace stays exactly where Gradio put it,
    and the ancestors between it and the tab are *marked* so the stylesheet can
    hide everything else."""

    def test_opening_marks_the_path_and_claims_the_tab(self):
        found = run("""
            ks.open();
            report({onTab: tab.getAttribute("data-mc-spatial-path"),
                    onWorkspace: el("mc-krea-spatial-workspace")
                        .getAttribute("data-mc-spatial-path")});
        """, initial=document())

        assert found["taken"] is True
        assert found["onTab"] == "1"
        assert found["onWorkspace"] == "1"
        # The workspace, its wrapper and the tab: a path, not the whole page.
        assert found["marked"] == 3

    def test_the_workspace_is_never_moved_to_get_there(self):
        """The overlay days: a fixed block relocated to document.body, and two
        copies of every id whenever Gradio rebuilt the tab."""
        found = run("""
            const before = el("mc-krea-spatial-workspace").parentNode;
            ks.open();
            report({same: el("mc-krea-spatial-workspace").parentNode === before,
                    onBody: body.children.indexOf(
                        el("mc-krea-spatial-workspace")) >= 0});
        """, initial=document())

        assert found["same"] is True
        assert found["onBody"] is False

    def test_closing_gives_the_tab_back(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-cancel");
            report();
        """, initial=document())

        assert found["open"] is False
        assert found["taken"] is False
        assert found["marked"] == 0

    def test_saving_keeps_the_tab_because_saving_does_not_close(self):
        """§3.2 and §3.3: Save commits and stays, so the takeover stays too --
        the whole point is generating without leaving the workspace."""
        found = run("""
            autoSaveBox.checked = false;
            ks.open();
            draw([100, 100], [400, 400]);
            press("mc-krea-spatial-save");
            report();
        """, initial=document())

        assert found["open"] is True
        assert found["taken"] is True
        assert found["published"] == 1

    def test_a_rebuilt_tab_is_taken_over_again(self):
        """Gradio rebuilding the tab replaces the very ancestors the marks were
        on. Leaving it there would show the rest of txt2img through a workspace
        that still believes it is open."""
        found = run("""
            ks.open();
            tab.removeAttribute("data-mc-spatial-path");
            tab.classList.remove("mc-krea-spatial-taken");
            ks.wire();
            report();
        """, initial=document())

        assert found["open"] is True
        assert found["taken"] is True

    def test_the_takeover_changes_no_layout_state(self):
        found = run("""
            ks.open();
            const before = ks.serialize();
            ks.surrender();
            ks.takeover();
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
            # Everything the editor has always had.
            "name", "type", "text", "text-field", "prompt", "prompt-label",
            "framing", "framing-field", "angle", "angle-field", "auto-hint",
            "list", "draw", "duplicate", "delete", "lower", "front",
            "save", "cancel", "canvas", "regions", "workspace", "proxy",
            "clear", "undo", "redo", "grid", "zoom-fit", "zoom-in", "zoom-out",
            "confirm", "keep", "discard", "literal-prefix", "literal-suffix",
            # §4: the action bar.
            "power", "mode-direct", "mode-smart", "quick", "quick-popup",
            "quick-size", "panels", "panels-popup", "collapse-all",
            "expand-visible",
            # §12: the rail, and one id per widget.
            "rail", "panel-prompts", "panel-person", "panel-layers",
            "panel-inspector", "panel-gallery", "panel-session",
            "show-prompts", "show-person", "show-layers", "show-inspector",
            "show-gallery", "show-session",
            # §13, §9, §15, §16 and §17.
            "scene", "literal-plus", "literal-minus", "creative", "person",
            "shape", "rotation", "rotation-field",
            "shot", "shot-previous", "shot-next",
            "generate", "progress", "fact-frame", "fact-ratio", "fact-regions",
            "fact-pipeline", "fact-state", "size-width", "size-height",
        }

        assert {"mc-krea-spatial-" + name for name in expected} <= emitted

    def test_the_shape_vocabulary_is_the_same_on_both_sides(self):
        """Python builds the palettes and JavaScript holds the proportions and
        the default prompts. Two lists that agree until somebody adds to one of
        them is how a Quick Add button starts placing a rectangle."""
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "scripts"))
        import model_chain_krea_creative as creative_script

        source = SCRIPT.read_text(encoding="utf-8")
        table = re.search(r"const SHAPES = \{(.*?)\n    \};", source, re.S)
        assert table, "the SHAPES table could not be found"
        known = set(re.findall(r"^\s{8}(\w+):", table.group(1), re.M))
        named = {name for name, _label in creative_script.SHAPES}

        assert known == named
        assert set(creative_script.QUICK_SHAPES) <= known
        assert set(creative_script.PERSON_SHAPES) <= known
        assert "rect" not in creative_script.PERSON_SHAPES


# --------------------------------------------------------------------------- #
# The action bar
# --------------------------------------------------------------------------- #


class TestTheActionBar:
    """§4. Ten controls, one row, and every one of them something somebody
    reaches for while composing rather than something they set once."""

    def test_the_spatial_switch_is_the_panel_s_own(self):
        """§4.1 and §13.2's rule applied to a checkbox: the bar shows the
        component Gradio is bound to, so turning Spatial off here is turning it
        off, remembered, without the workspace closing."""
        found = run("""
            ks.open();
            const box = el("mc-krea-spatial-power");
            box.checked = false;
            box.dispatchEvent({type: "change", target: box});
            report();
        """, initial=document())

        assert found["hostSpatial"] is False
        assert found["open"] is True

    def test_the_switch_shows_what_the_panel_already_says(self):
        found = run("""
            hostSpatial.checked = false;
            ks.open();
            report({shown: el("mc-krea-spatial-power").checked});
        """, initial=document())

        assert found["shown"] is False

    def test_the_mode_is_two_states_of_one_control(self):
        """§4.2: a compact segmented control, not several stacked rows. Pressing
        one presses the radio the panel is bound to."""
        found = run("""
            ks.open();
            press("mc-krea-spatial-mode-direct");
            report({direct: el("mc-krea-spatial-mode-direct").classList
                        .contains("active"),
                    smart: el("mc-krea-spatial-mode-smart").classList
                        .contains("active"),
                    pipeline: el("mc-krea-spatial-fact-pipeline").textContent});
        """, initial=document())

        assert found["hostMode"] == "direct"
        assert found["direct"] is True
        assert found["smart"] is False
        assert "Direct BBOX" in found["pipeline"]

    def test_the_mode_travels_with_the_layout_too(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-mode-direct");
            press("mc-krea-spatial-save");
            report({document: saved()});
        """, initial=document())

        assert found["document"]["compose_mode"] == "direct"

    def test_the_bar_carries_no_explanation_of_itself(self):
        """§2.2 and §25. Control labels, field labels, button names, live state
        and the selected region's data -- and nothing that describes a control
        somebody is already looking at."""
        markup = _markup()

        assert "title=" not in markup
        for banned in ("you can", "This does", "in your own words",
                       "Press Full Screen to draw one.</p>\\n  <div"):
            assert banned not in markup

    def test_the_workspace_has_no_hover_tooltips_at_all(self):
        """A ``title`` set from JavaScript is the same explanation arriving by
        another route, so the file is read as well as the markup."""
        source = SCRIPT.read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("//"))

        assert ".title =" not in code


# --------------------------------------------------------------------------- #
# Quick Add and the Person palette
# --------------------------------------------------------------------------- #


class TestQuickAdd:
    """§10. Immediate access to the high-frequency primitives without opening
    the rail, and the same gesture whether it is tapped or dragged."""

    def test_the_popup_opens_and_closes_from_its_own_button(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-quick");
            const opened = !el("mc-krea-spatial-quick-popup").hidden;
            press("mc-krea-spatial-quick");
            report({opened: opened,
                    closed: !!el("mc-krea-spatial-quick-popup").hidden});
        """, initial=document())

        assert found["opened"] is True
        assert found["closed"] is True

    def test_one_popup_at_a_time(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-quick");
            press("mc-krea-spatial-panels");
            report({quick: !!el("mc-krea-spatial-quick-popup").hidden,
                    panels: !el("mc-krea-spatial-panels-popup").hidden});
        """, initial=document())

        assert found["quick"] is True
        assert found["panels"] is True

    def test_tapping_head_places_it_centred_with_its_own_prompt(self):
        """Acceptance test C."""
        found = run("""
            ks.open();
            ks.clear();
            press("mc-krea-spatial-quick");
            place("head", null);
            report();
        """, initial=document())

        region = found["regions"][0]
        left, top, right, bottom = region["bbox"]

        assert region["shape"] == "head"
        assert region["prompt"] == "head"
        assert region["rotation"] == 0
        assert abs((left + right) / 2 - 500) < 60
        assert abs((top + bottom) / 2 - 500) < 60
        # Taller than it is wide *on screen*, which is not the same thing as
        # taller than it is wide in normalized units. The frame here is
        # 1024x1344, so a fraction of it is worth more pixels down the page
        # than across it, and a head whose stored numbers looked like a head
        # would arrive as a long oval.
        across = (right - left) / 1000 * 1024
        down = (bottom - top) / 1000 * 1344

        assert down > across
        assert abs(across / down - 0.62 / 0.74) < 0.03

    def test_a_silhouette_keeps_its_proportions_on_any_frame(self):
        """The same head, on a square frame and on a tall one, is the same
        picture. Only the numbers behind it differ."""
        found = run("""
            width.value = "1024"; height.value = "1024";
            ks.open();
            ks.clear();
            place("head", null);
            const square = ks.ordered()[0].bbox.slice();
            ks.close();
            width.value = "1024"; height.value = "1536";
            ks.open();
            ks.clear();
            place("head", null);
            report({square: square, tall: ks.ordered()[0].bbox.slice()});
        """, initial=document())

        def picture(bbox, frame_w, frame_h):
            wide = (bbox[2] - bbox[0]) / 1000 * frame_w
            high = (bbox[3] - bbox[1]) / 1000 * frame_h
            return wide / high

        assert abs(picture(found["square"], 1024, 1024)
                   - picture(found["tall"], 1024, 1536)) < 0.02

    def test_a_tap_does_not_take_the_focus_off_the_canvas_for_a_person_part(self):
        """§10.2: a silhouette arrives with a usable prompt already in it, so
        the cursor stays where the composing is happening."""
        found = run("""
            ks.open();
            place("chest", null);
            report({focused: !!el("mc-krea-spatial-prompt").focused});
        """, initial=document())

        assert found["focused"] is False

    def test_dragging_chest_drops_it_where_it_was_let_go(self):
        """Acceptance test D: dragged to the upper right, and the bbox says so."""
        found = run("""
            ks.open();
            ks.clear();
            press("mc-krea-spatial-quick");
            place("chest", [750, 250]);
            report();
        """, initial=document())

        left, top, right, bottom = found["regions"][0]["bbox"]

        assert found["regions"][0]["shape"] == "chest"
        assert abs((left + right) / 2 - 750) < 40
        assert abs((top + bottom) / 2 - 250) < 40

    def test_a_drop_outside_the_frame_creates_nothing(self):
        """§10.3: an invalid drop creates nothing rather than a box clamped to
        an edge nobody aimed at."""
        found = run("""
            ks.open();
            ks.clear();
            place("head", [1400, 1400]);
            report();
        """, initial=document())

        assert found["regions"] == []

    def test_the_popup_closes_after_a_successful_add(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-quick");
            place("head", null);
            report({popupShut: !!el("mc-krea-spatial-quick-popup").hidden});
        """, initial=document())

        assert found["popupShut"] is True
        assert len(found["regions"]) == 2

    def test_the_size_slider_decides_how_big_a_new_shape_is(self):
        """§10.1: one reference length, and each shape a proportion of it."""
        found = run("""
            ks.open();
            ks.clear();
            const slider = el("mc-krea-spatial-quick-size");
            slider.value = "800";
            slider.dispatchEvent({type: "input", target: slider});
            place("rect", null);
            report();
        """, initial=document())

        left, _top, right, _bottom = found["regions"][0]["bbox"]

        assert right - left > 700


class TestThePersonPalette:
    """§9. The user sees body parts; the engine sees rectangles."""

    def test_every_part_the_design_names_is_grabbable(self):
        """§9.2's eleven, each one its own control."""
        found = run("""
            report({parts: el("mc-krea-spatial-person")
                        .querySelectorAll(".mc-krea-spatial-shape-button")
                        .map((entry) => entry.dataset.shape)});
        """, initial=document())

        assert found["parts"] == ["head", "chest", "waist",
                                  "left_arm", "right_arm",
                                  "left_hand", "right_hand",
                                  "left_leg", "right_leg",
                                  "left_foot", "right_foot"]

    def test_dragging_a_left_arm_out_makes_a_left_arm(self):
        """Acceptance test E."""
        found = run("""
            ks.open();
            ks.clear();
            place("left_arm", [300, 400]);
            report();
        """, initial=document())

        region = found["regions"][0]

        assert region["shape"] == "left_arm"
        assert region["prompt"] == "left arm"

    def test_the_silhouette_is_a_palette_and_not_one_stored_person(self):
        """§9.4: multiple heads, multiple left arms, multiple complete sets."""
        found = run("""
            ks.open();
            ks.clear();
            place("head", [200, 200]);
            place("head", [700, 200]);
            place("left_hand", [400, 700]);
            report();
        """, initial=document())

        assert [region["shape"] for region in found["regions"]] == [
            "head", "head", "left_hand"]
        assert len({region["id"] for region in found["regions"]}) == 3

    def test_a_silhouette_is_still_an_axis_aligned_rectangle(self):
        """§2.5 and §9.5. Whatever is drawn, what is serialized is a box."""
        found = run("""
            ks.open();
            ks.clear();
            place("left_leg", [500, 500]);
            press("mc-krea-spatial-save");
            report({document: saved()});
        """, initial=document())

        entry = found["document"]["regions"][0]

        assert len(entry["bbox"]) == 4
        assert entry["bbox"][0] < entry["bbox"][2]
        assert entry["bbox"][1] < entry["bbox"][3]
        assert "path" not in json.dumps(entry)


class TestRotation:
    """§9.6. The silhouette turns. The box does not."""

    def test_rotating_an_arm_changes_the_drawing_and_not_the_box(self):
        """Acceptance test F."""
        found = run("""
            ks.open();
            ks.clear();
            place("left_arm", [500, 500]);
            const before = ks.ordered()[0].bbox.slice();
            commit("mc-krea-spatial-rotation", "35");
            press("mc-krea-spatial-save");
            report({before: before, document: saved(),
                    turned: body.querySelectorAll(".mc-krea-spatial-shape")
                        .map((entry) => entry.style.transform)
                        .filter(Boolean)});
        """, initial=document())

        entry = found["document"]["regions"][0]

        assert entry["ui_rotation"] == 35
        assert entry["bbox"] == found["before"]
        assert "rotate(35deg)" in found["turned"]

    def test_the_slider_is_offered_for_silhouettes_and_not_for_boxes(self):
        found = run("""
            ks.open();
            ks.clear();
            place("rect", null);
            const forBox = !el("mc-krea-spatial-rotation-field").hidden;
            place("head", null);
            report({forBox: forBox,
                    forHead: !el("mc-krea-spatial-rotation-field").hidden});
        """, initial=document())

        assert found["forBox"] is False
        assert found["forHead"] is True

    def test_rotation_is_clamped_to_half_a_turn_each_way(self):
        found = run("""
            ks.open();
            ks.clear();
            place("head", null);
            commit("mc-krea-spatial-rotation", "900");
            report();
        """, initial=document())

        assert found["regions"][0]["rotation"] == 180


# --------------------------------------------------------------------------- #
# The document the editor writes
# --------------------------------------------------------------------------- #


class TestTheEditorMetadata:
    """§21.2 and §2.6. Two optional fields, in both directions."""

    def test_a_rectangle_layout_serializes_to_the_bytes_it_always_did(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-save");
            report({document: saved()});
        """, initial=document())

        entry = found["document"]["regions"][0]

        assert "ui_shape" not in entry
        assert "ui_rotation" not in entry

    def test_a_legacy_layout_loads_as_rectangles_at_zero_degrees(self):
        """Acceptance test P: old rectangle-only layouts load unchanged."""
        found = run("""
            ks.open();
            report();
        """, initial=document())

        assert found["regions"][0]["shape"] == "rect"
        assert found["regions"][0]["rotation"] == 0
        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

    def test_a_shape_this_build_cannot_draw_becomes_a_rectangle(self):
        """§26. A layout from a later build arrives with its boxes in the right
        places and one silhouette this build does not know."""
        alien = dict(FACE, ui_shape="tail", ui_rotation=20)
        found = run("""
            ks.open();
            report();
        """, initial=document([alien]))

        assert found["regions"][0]["shape"] == "rect"
        assert found["regions"][0]["rotation"] == 20
        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

    def test_undo_gives_a_silhouette_back_as_a_silhouette(self):
        """The bug this test is named after: a snapshot stringified the live
        region objects, and restoring one ran them back through the reader that
        speaks the *document's* field names. `shape` and `rotation` are not
        `ui_shape` and `ui_rotation`, so every silhouette on the canvas turned
        into a plain box the first time anybody pressed Undo."""
        found = run("""
            ks.open();
            ks.clear();
            place("left_arm", [300, 400]);
            commit("mc-krea-spatial-rotation", "35");
            place("rect", null);
            press("mc-krea-spatial-undo");
            report();
        """, initial=document())

        arm = found["regions"][0]

        assert len(found["regions"]) == 1
        assert arm["shape"] == "left_arm"
        assert arm["rotation"] == 35
        assert arm["prompt"] == "left arm"

    def test_undo_gives_a_region_its_literal_fields_back_too(self):
        """The same defect, on the fields that met it first: the snapshot said
        `literalPrefix` and the reader wanted `literal_prefix`, so an undo
        emptied both boxes of every region it touched."""
        found = run("""
            ks.open();
            commit("mc-krea-spatial-literal-prefix", "<lora:krea2:1>");
            commit("mc-krea-spatial-literal-suffix", "__grain__");
            place("rect", null);
            press("mc-krea-spatial-undo");
            press("mc-krea-spatial-save");
            report({document: saved()});
        """, initial=document())

        entry = found["document"]["regions"][0]

        assert entry["literal_prefix"] == "<lora:krea2:1>"
        assert entry["literal_suffix"] == "__grain__"

    def test_the_names_are_numbers(self):
        """§6.1. "Region 4" is the word "region" on every row of a panel called
        Layers, and the number is the half anybody reads."""
        found = run("""
            ks.open();
            ks.clear();
            place("rect", null);
            place("head", null);
            place("chest", null);
            report();
        """, initial=document())

        assert [region["name"] for region in found["regions"]] == ["1", "2", "3"]

    def test_the_label_is_centred_in_whatever_the_region_turned_out_to_be(self):
        """§6.2, as far as a test without a layout engine can see it: one label
        element per region, carrying the name, whatever the shape."""
        found = run("""
            ks.open();
            ks.clear();
            place("head", null);
            report({labels: body.querySelectorAll(".mc-krea-spatial-label")
                        .map((entry) => entry.textContent)});
        """, initial=document())

        assert found["labels"] == ["1"]


# --------------------------------------------------------------------------- #
# The rail
# --------------------------------------------------------------------------- #


class TestThePanels:
    """§12 and §4.6. The rail is optional furniture: nothing in it may cost the
    canvas a pixel it was using."""

    def test_every_widget_the_design_names_is_in_the_rail(self):
        found = run("""
            report({widgets: el("mc-krea-spatial-rail")
                        .querySelectorAll(".mc-krea-spatial-widget")
                        .map((entry) => entry.dataset.panel)});
        """, initial=document())

        assert found["widgets"] == ["prompts", "person", "layers", "inspector",
                                    "gallery", "session"]

    def test_hiding_prompts_takes_the_widget_away(self):
        """Acceptance test N."""
        found = run("""
            ks.open();
            press("mc-krea-spatial-panels");
            const box = el("mc-krea-spatial-show-prompts");
            box.checked = false;
            box.dispatchEvent({type: "change", target: box});
            report({shown: panelShown("prompts"),
                    others: panelShown("layers")});
        """, initial=document())

        assert found["shown"] is False
        assert found["others"] is True
        assert found["hidden"] == ["prompts"]

    def test_a_header_collapses_its_own_widget(self):
        found = run("""
            ks.open();
            const head = el("mc-krea-spatial-panel-layers")
                .querySelector(".mc-krea-spatial-widget-head");
            el("mc-krea-spatial-rail").dispatchEvent(
                {type: "click", target: head, preventDefault() {}});
            report({open: panelOpen("layers"),
                    others: panelOpen("inspector"),
                    said: head.getAttribute("aria-expanded")});
        """, initial=document())

        assert found["open"] is False
        assert found["others"] is True
        assert found["said"] == "false"

    def test_collapse_all_and_expand_visible(self):
        found = run("""
            ks.open();
            press("mc-krea-spatial-panels");
            ks.panelShow("gallery", false);
            press("mc-krea-spatial-collapse-all");
            const shut = ks.panels.filter((key) => !panelOpen(key)).length;
            press("mc-krea-spatial-expand-visible");
            report({shut: shut,
                    open: ks.panels.filter((key) => panelOpen(key)),
                    gallery: panelOpen("gallery")});
        """, initial=document())

        assert found["shut"] == 6
        # Expand Visible expands what is visible, and leaves a hidden widget
        # collapsed rather than quietly bringing it back.
        assert found["gallery"] is False
        assert set(found["open"]) == {"prompts", "person", "layers",
                                      "inspector", "session"}

    def test_panel_state_is_not_layout_history(self):
        """§20: hiding a panel is a session preference. It may not mark a
        layout as changed and Undo may not bring a panel back."""
        found = run("""
            ks.open();
            const before = ks.serialize();
            ks.panelShow("layers", false);
            ks.panelCollapse("inspector", true);
            report({same: ks.serialize() === before});
        """, initial=document())

        assert found["same"] is True
        assert found["past"] == 0
        assert found["published"] == 0

    def test_a_new_rectangle_opens_the_inspector_to_type_into(self):
        """§15.2: a prompt field that is switched off or collapsed cannot be
        typed into, so the panel it lives in is opened before the focus."""
        found = run("""
            ks.open();
            ks.panelShow("inspector", false);
            place("rect", null);
            report({shown: panelShown("inspector"),
                    focused: !!el("mc-krea-spatial-prompt").focused});
        """, initial=document())

        assert found["shown"] is True
        assert found["focused"] is True


# --------------------------------------------------------------------------- #
# Prompts, gallery, session
# --------------------------------------------------------------------------- #


class TestThePromptsWidget:
    """§13. Three boxes that are second views of components that already exist,
    and no saved copy of any of them."""

    def test_opening_shows_what_the_tab_already_holds(self):
        found = run("""
            hostPrompt.value = "a lighthouse at dusk";
            hostLiteralPlus.value = "<lora:krea2:1>";
            ks.open();
            report({prompt: el("mc-krea-spatial-scene").value,
                    plus: el("mc-krea-spatial-literal-plus").value});
        """, initial=document())

        assert found["prompt"] == "a lighthouse at dusk"
        assert found["plus"] == "<lora:krea2:1>"

    def test_typing_here_reaches_the_canonical_prompt(self):
        """Acceptance test I."""
        found = run("""
            ks.open();
            field("mc-krea-spatial-scene", "a red bicycle");
            report();
        """, initial=document())

        assert found["hostPrompt"] == "a red bicycle"

    def test_typing_there_reaches_this_one(self):
        found = run("""
            ks.open();
            hostPrompt.value = "a green bicycle";
            hostPrompt.dispatchEvent({type: "input", target: hostPrompt});
            report({shown: el("mc-krea-spatial-scene").value});
        """, initial=document())

        assert found["shown"] == "a green bicycle"

    def test_both_literal_boxes_are_the_panel_s_own(self):
        found = run("""
            ks.open();
            field("mc-krea-spatial-literal-plus", "__grain__");
            field("mc-krea-spatial-literal-minus", "blurry");
            report();
        """, initial=document())

        assert found["hostLiteralPlus"] == "__grain__"
        assert found["hostLiteralMinus"] == "blurry"

    def test_creative_first_is_the_same_state_as_txt2img(self):
        """§13.3: no independent local Creative value."""
        found = run("""
            ks.open();
            const box = el("mc-krea-spatial-creative");
            box.checked = true;
            box.dispatchEvent({type: "change", target: box});
            report({pipeline: el("mc-krea-spatial-fact-pipeline").textContent});
        """, initial=document())

        assert found["hostCreative"] is True
        assert "Creative first" in found["pipeline"]

    def test_the_mirror_writes_nothing_into_the_layout(self):
        found = run("""
            ks.open();
            const before = ks.serialize();
            field("mc-krea-spatial-scene", "a red bicycle");
            report({same: ks.serialize() === before});
        """, initial=document())

        assert found["same"] is True
        assert found["past"] == 0


class TestTheGalleryWidget:
    """§16. The tab's own results, beside the canvas, so that Generate, look,
    change, Generate again never leaves the workspace."""

    def test_it_shows_the_tab_s_own_results(self):
        found = run("""
            showImages(["one.png", "two.png", "three.png"]);
            ks.open();
            report({at: el("mc-krea-spatial-shot-at").textContent,
                    shown: el("mc-krea-spatial-shot").querySelector("img").src});
        """, initial=document())

        assert found["shots"] == ["one.png", "two.png", "three.png"]
        assert found["shown"] == "three.png"
        assert found["at"] == "3 / 3"

    def test_next_from_the_last_image_is_the_first(self):
        """Acceptance test J."""
        found = run("""
            showImages(["one.png", "two.png", "three.png"]);
            ks.open();
            press("mc-krea-spatial-shot-next");
            report({shown: el("mc-krea-spatial-shot").querySelector("img").src});
        """, initial=document())

        assert found["shotAt"] == 0
        assert found["shown"] == "one.png"

    def test_previous_from_the_first_image_is_the_last(self):
        found = run("""
            showImages(["one.png", "two.png", "three.png"]);
            ks.open();
            press("mc-krea-spatial-shot-next");
            press("mc-krea-spatial-shot-previous");
            report();
        """, initial=document())

        assert found["shotAt"] == 2

    def test_a_new_result_arrives_without_leaving_the_workspace(self):
        """Acceptance test K, minus the part only a browser can do: the
        workspace is still open, the takeover is still in force, and the new
        image is the one on show."""
        found = run("""
            ks.open();
            press("mc-krea-spatial-generate");
            showProgress("40%", "40%");
            const during = !el("mc-krea-spatial-progress").hidden;
            const said = el("mc-krea-spatial-progress-read").textContent;
            clearProgress();
            showImages(["fresh.png"]);
            report({during: during, said: said,
                    after: !!el("mc-krea-spatial-progress").hidden,
                    shown: el("mc-krea-spatial-shot").querySelector("img").src});
        """, initial=document())

        assert found["generatePresses"] == 1
        assert found["during"] is True
        assert found["said"] == "40%"
        assert found["after"] is True
        assert found["shown"] == "fresh.png"
        assert found["open"] is True
        assert found["taken"] is True

    def test_generating_does_not_require_a_save(self):
        """§26's last paragraph, and the contract that makes it true: what the
        next Generate composes is whatever is in the state box, and Auto Save
        put the working layout there when the edit finished."""
        found = run("""
            ks.open();
            draw([100, 100], [400, 400]);
            press("mc-krea-spatial-generate");
            report({document: saved()});
        """, initial=document())

        assert found["published"] == 1
        assert len(found["document"]["regions"]) == 2
        assert found["generatePresses"] == 1


class TestTheSessionWidget:
    """§17. Everything that used to be a permanent strip across the top, in one
    widget that collapses and hides like every other."""

    def test_it_says_what_is_true_right_now(self):
        found = run("""
            ks.open();
            report({frame: el("mc-krea-spatial-fact-frame").textContent,
                    ratio: el("mc-krea-spatial-fact-ratio").textContent,
                    regions: el("mc-krea-spatial-fact-regions").textContent,
                    pipeline: el("mc-krea-spatial-fact-pipeline").textContent,
                    state: el("mc-krea-spatial-fact-state").textContent});
        """, initial=document())

        assert found["frame"] == "1024 × 1344"
        assert found["ratio"] == "16:21 · Portrait"
        assert found["regions"] == "1 of 24"
        assert found["pipeline"] == "Smart Spatial"
        assert found["state"] == "Saved"

    def test_an_unsaved_edit_says_so_and_a_save_clears_it(self):
        found = run("""
            autoSaveBox.checked = false;
            ks.open();
            draw([100, 100], [400, 400]);
            const dirty = el("mc-krea-spatial-fact-state").textContent;
            press("mc-krea-spatial-save");
            report({dirty: dirty,
                    clean: el("mc-krea-spatial-fact-state").textContent});
        """, initial=document())

        assert found["dirty"] == "Unsaved"
        assert found["clean"] == "Saved"

    def test_an_auto_saved_edit_reads_as_saved_straight_away(self):
        """The commit and the line describing it happen in one repaint. Painted
        the other way round, Session says Unsaved about an edit that was
        committed a line later and stays wrong until something else repaints."""
        found = run("""
            ks.open();
            draw([100, 100], [400, 400]);
            report({state: el("mc-krea-spatial-fact-state").textContent});
        """, initial=document())

        assert found["published"] == 1
        assert found["state"] == "Saved"

    def test_spatial_switched_off_is_part_of_the_pipeline_line(self):
        found = run("""
            hostSpatial.checked = false;
            ks.open();
            report({pipeline: el("mc-krea-spatial-fact-pipeline").textContent});
        """, initial=document())

        assert found["pipeline"] == "Smart Spatial · off"


# --------------------------------------------------------------------------- #
# Where the page is, on the way in and on the way out
# --------------------------------------------------------------------------- #


class TestTheScrollPosition:
    """The workspace is the last thing in the extension's accordion, which is
    the last thing on the txt2img tab -- several screens down. So the moment
    before the takeover hides everything above it, the page is scrolled to
    wherever somebody was reading, and the moment after, that offset is measured
    against a page one screen tall: the workspace opens showing its bottom edge
    and the WebUI footer, with the action bar off the top."""

    def test_opening_shows_the_action_bar(self):
        found = run("""
            scrollTo(0, 4000);
            ks.open();
            report({at: pageTop()});
        """, initial=document())

        assert found["at"] == 0

    def test_closing_puts_the_page_back_where_it_was(self):
        """Coming back from Full Screen to a tab scrolled somewhere else is the
        same lost place by another route."""
        found = run("""
            scrollTo(0, 4000);
            ks.open();
            press("mc-krea-spatial-cancel");
            report({at: pageTop()});
        """, initial=document())

        assert found["open"] is False
        assert found["at"] == 4000

    def test_discarding_puts_it_back_too(self):
        found = run("""
            scrollTo(0, 2500);
            ks.open();
            draw([100, 200], [400, 600]);
            press("mc-krea-spatial-cancel");
            press("mc-krea-spatial-discard");
            report({at: pageTop()});
        """, initial=document())

        assert found["open"] is False
        assert found["at"] == 2500

    def test_saving_does_not_move_the_page_because_it_does_not_close(self):
        found = run("""
            scrollTo(0, 4000);
            ks.open();
            draw([100, 200], [400, 600]);
            press("mc-krea-spatial-save");
            report({at: pageTop()});
        """, initial=document())

        assert found["open"] is True
        assert found["at"] == 0


# --------------------------------------------------------------------------- #
# Reordering, by the one pointer path
# --------------------------------------------------------------------------- #


THREE = [dict(FACE, id="a", name="A", z=0), dict(FACE, id="b", name="B", z=1),
         dict(FACE, id="c", name="C", z=2)]


class TestReorderingByPointer:
    """§2.3 and §14.1. The rows used to be HTML5 `draggable`, which a finger
    cannot start and which the grip's own `touch-action: none` suppressed the
    long-press fallback for. One pointer path now, and the threshold between a
    press and a slide is what lets a row be both "select this" and "move this"
    without a modifier."""

    def test_dragging_a_row_down_moves_it_down(self):
        found = run("""
            ks.open();
            slide("row", 0, 2);
            report();
        """, initial=document(THREE))

        assert [region["id"] for region in found["regions"]] == ["b", "c", "a"]

    def test_dragging_a_row_up_moves_it_up(self):
        found = run("""
            ks.open();
            slide("row", 2, 0);
            report();
        """, initial=document(THREE))

        assert [region["id"] for region in found["regions"]] == ["c", "a", "b"]

    def test_a_press_that_does_not_move_selects_instead(self):
        found = run("""
            ks.open();
            const row = rows()[0];
            const list = el("mc-krea-spatial-list");
            list.dispatchEvent({type: "pointerdown", target: row, pointerId: 7,
                                clientY: 10, button: 0, preventDefault() {}});
            list.dispatchEvent({type: "pointerup", target: row, pointerId: 7,
                                clientY: 12, preventDefault() {}});
            inList(row);
            report();
        """, initial=document(THREE))

        assert found["selected"] == "a"
        assert [region["id"] for region in found["regions"]] == ["a", "b", "c"]

    def test_the_click_a_completed_drag_leaves_behind_is_spent(self):
        """A pointerup is followed by a click, and that click would select the
        row that was just moved -- or collapse the widget. Same gesture, already
        answered."""
        found = run("""
            ks.open();
            ks.select("c");
            slide("row", 0, 2);
            const after = ks.state.selected;
            inList(rows()[0]);
            report({after: after});
        """, initial=document(THREE))

        assert found["after"] == "c"
        assert found["selected"] == "c"

    def test_a_reorder_is_one_history_entry_and_undoes(self):
        found = run("""
            ks.open();
            slide("row", 0, 2);
            const moved = ks.ordered().map((r) => r.id);
            press("mc-krea-spatial-undo");
            report({moved: moved});
        """, initial=document(THREE))

        assert found["moved"] == ["b", "c", "a"]
        assert [region["id"] for region in found["regions"]] == ["a", "b", "c"]
        assert found["past"] == 0

    def test_a_cancelled_drag_reorders_nothing(self):
        found = run("""
            ks.open();
            const list = el("mc-krea-spatial-list");
            list.dispatchEvent({type: "pointerdown", target: rows()[0], pointerId: 7,
                                clientY: 10, button: 0, preventDefault() {}});
            list.dispatchEvent({type: "pointermove", target: rows()[0], pointerId: 7,
                                clientY: 250, preventDefault() {}});
            list.dispatchEvent({type: "pointercancel", target: rows()[0], pointerId: 7,
                                preventDefault() {}});
            report({marks: rows().filter((row) =>
                        row.classList.contains("dragging")).length});
        """, initial=document(THREE))

        assert [region["id"] for region in found["regions"]] == ["a", "b", "c"]
        assert found["marks"] == 0

    def test_the_trash_button_is_not_a_drag_handle(self):
        found = run("""
            ks.open();
            const list = el("mc-krea-spatial-list");
            const trash = rows()[0].querySelector(".mc-krea-spatial-row-trash");
            list.dispatchEvent({type: "pointerdown", target: trash, pointerId: 7,
                                clientY: 10, button: 0, preventDefault() {}});
            report({carrying: !!ks.state.carrying});
        """, initial=document(THREE))

        assert found["carrying"] is False


class TestReorderingTheRail:
    """§12 makes the rail a set of optional tools rather than a fixed panel, so
    which of them is nearest the canvas is a preference like collapsing one."""

    def test_a_widget_can_be_dragged_up_the_rail(self):
        found = run("""
            ks.open();
            slide("panel", 4, 0);
            report({order: railOrder()});
        """, initial=document())

        assert found["order"][0] == "gallery"
        assert found["order"] == ["gallery", "prompts", "person", "layers",
                                  "inspector", "session"]

    def test_the_dom_follows_the_order(self):
        found = run("""
            ks.open();
            ks.movePanel("session", "prompts", true);
            report({order: railOrder(),
                    dom: el("mc-krea-spatial-rail")
                        .querySelectorAll(".mc-krea-spatial-widget")
                        .map((entry) => entry.dataset.panel)});
        """, initial=document())

        assert found["order"][0] == "session"
        assert found["dom"] == found["order"]

    def test_only_the_grip_starts_a_drag(self):
        """The header is wide and gets pressed often. A rail that rearranged
        itself whenever a finger slid six pixels on the way to collapsing
        something would be a rail nobody trusted."""
        found = run("""
            ks.open();
            const rail = el("mc-krea-spatial-rail");
            const head = el("mc-krea-spatial-panel-layers")
                .querySelector(".mc-krea-spatial-widget-head");
            rail.dispatchEvent({type: "pointerdown", target: head, pointerId: 8,
                                clientY: 10, button: 0, preventDefault() {}});
            report({carrying: !!ks.state.carrying});
        """, initial=document())

        assert found["carrying"] is False

    def test_the_order_is_a_preference_and_not_layout_history(self):
        found = run("""
            ks.open();
            const before = ks.serialize();
            ks.movePanel("gallery", "prompts", true);
            report({same: ks.serialize() === before});
        """, initial=document())

        assert found["same"] is True
        assert found["past"] == 0
        assert found["published"] == 0

    def test_a_widget_the_rail_grows_appears_for_somebody_who_rearranged(self):
        """The stored order is filled out from the canonical list, so a widget
        added later shows up at the end rather than not at all."""
        found = run("""
            ks.open();
            ks.state.order = ["session", "layers"];
            report({order: ks.panelOrder()});
        """, initial=document())

        assert found["order"][:2] == ["session", "layers"]
        assert sorted(found["order"]) == sorted(
            ["prompts", "person", "layers", "inspector", "gallery", "session"])


# --------------------------------------------------------------------------- #
# Straight from the shape to its words
# --------------------------------------------------------------------------- #


class TestDoubleClickToDescribe:
    def test_double_clicking_a_region_opens_its_prompt(self):
        found = run("""
            ks.open();
            ks.select("");
            ks.panelCollapse("inspector", true);
            const region = body.querySelectorAll(".mc-krea-spatial-region")[0];
            canvas.dispatchEvent({type: "dblclick", target: region,
                                  preventDefault() {}});
            report({open: panelOpen("inspector"),
                    focused: !!el("mc-krea-spatial-prompt").focused});
        """, initial=document())

        assert found["selected"] == "r1"
        assert found["open"] is True
        assert found["focused"] is True

    def test_double_clicking_the_selection_proxy_does_the_same(self):
        found = run("""
            ks.open();
            canvas.dispatchEvent({type: "dblclick", target: proxy(),
                                  preventDefault() {}});
            report({focused: !!el("mc-krea-spatial-prompt").focused});
        """, initial=document())

        assert found["focused"] is True

    def test_double_clicking_bare_canvas_does_nothing(self):
        found = run("""
            ks.open();
            ks.select("");
            canvas.dispatchEvent({type: "dblclick", target: canvas,
                                  preventDefault() {}});
            report({focused: !!el("mc-krea-spatial-prompt").focused});
        """, initial=document())

        assert found["selected"] == ""
        assert found["focused"] is False


# --------------------------------------------------------------------------- #
# The workspace's own stylesheet
# --------------------------------------------------------------------------- #


class TestTheWorkspaceStylesheet:
    """The layout is CSS, so the questions a fake DOM cannot answer are asked
    of the stylesheet directly: does every class the markup emits have a rule,
    does every silhouette have a shape, and is any of it dark-only."""

    @pytest.fixture
    def css(self):
        return (ROOT / "style.css").read_text(encoding="utf-8")

    @pytest.fixture
    def section(self, css):
        """The workspace's block, with the prose taken out.

        Comments are stripped first, because every question below is about what
        the browser is told and a paragraph that happens to name a class is not
        a rule -- reading them together is how a stylesheet passes a test by
        describing what it does not do.
        """
        block = css.split("The Spatial Layout workspace.", 1)[1]
        # Past the end of the header comment this split landed inside, and up
        # to the start of the next section's, so that every /* below has its
        # matching */ and the regex can take the pairs out.
        block = block.split("*/", 1)[1].split("/* ====", 1)[0]
        return re.sub(r"/\*.*?\*/", "", block, flags=re.S)

    @pytest.fixture
    def rules(self, section):
        """``(selector, body)`` for every rule in the block."""
        found = []
        for part in section.split("}"):
            if "{" not in part:
                continue
            selector, body = part.rsplit("{", 1)
            found.append((" ".join(selector.split()), body))
        return found

    def test_every_class_the_markup_emits_has_a_rule(self, section):
        """Dangling the other way round: a control styled by nothing looks like
        a control somebody forgot, and on a theme with opinions it looks like a
        bug."""
        used = set()
        for group in re.findall(r'class="([^"]+)"', _markup()):
            used.update(name for name in group.split()
                        if name.startswith("mc-krea-spatial-"))
        # The compact canvas has its own block further down the stylesheet.
        used = {name for name in used if "-compact" not in name}
        missing = {name for name in used if "." + name not in section}

        assert missing == set()

    def test_every_rule_in_it_selects_something_that_exists(self, section):
        """The other direction, and the one a browser never complains about: a
        rule for a class nothing emits is dead weight that reads as a control
        somebody removed and a style somebody forgot."""
        import model_chain_krea_creative as creative_script

        source = SCRIPT.read_text(encoding="utf-8")
        emitted = set()
        for group in re.findall(r'class="([^"]+)"', _markup()):
            emitted.update(name for name in group.split()
                           if name.startswith("mc-krea-spatial-"))
        # The browser file builds the rest, either as a whole name or as the
        # prefix plus a suffix.
        for suffix in re.findall(r'P \+ "(-[A-Za-z0-9_-]+) ?"', source):
            emitted.add("mc-krea-spatial" + suffix)
        emitted.update(re.findall(r'"(mc-krea-spatial-[A-Za-z0-9_-]+)"', source))
        # ...and the shapes, from the table both sides share.
        for shape, _label in creative_script.SHAPES:
            emitted.add("mc-krea-spatial-shape-" + shape)
            emitted.add("mc-krea-spatial-part-" + shape)

        styled = set(re.findall(r"\.(mc-krea-spatial-[A-Za-z0-9_-]+)", section))

        assert styled - emitted == set()

    def test_every_shape_the_editor_can_draw_has_a_silhouette(self, section):
        """A shape with no clip-path is a rectangle with a different name, and
        the palette would offer eleven identical buttons."""
        import model_chain_krea_creative as creative_script

        for name, _label in creative_script.SHAPES:
            if name == "rect":
                continue          # the one with no silhouette, on purpose
            assert f".mc-krea-spatial-shape-{name}" in section, name
            assert f".mc-krea-spatial-part-{name}" in section, name

    def test_the_silhouette_is_clipped_on_the_art_and_not_on_the_button(self):
        """A clip-path on the Quick Add button would take the label with it."""
        markup = _markup()

        assert 'class="mc-krea-spatial-shape-button mc-krea-spatial-part-head"' \
            in markup
        assert 'class="mc-krea-spatial-shape-art mc-krea-spatial-shape-head"' \
            in markup

    def test_it_selects_nothing_gradio_generated(self, section):
        assert ".svelte" not in section

    def test_the_rail_scrolls_and_the_canvas_does_not(self, section):
        """§12.1, and the failure it names: a rail full of panels pushing the
        frame down the page. A flex child's minimum is its content size unless
        it is told otherwise, so both halves of this are load-bearing."""
        rail = section.split(".mc-krea-spatial-rail {", 1)[1].split("}", 1)[0]

        assert "overflow-y: auto" in rail
        # Bounded, or overflow-y has nothing to do: min-height stops the flex
        # default of "at least as tall as my content", and max-height stops a
        # rail of six open widgets outgrowing the body on a browser that
        # stretches it anyway.
        assert "min-height: 0" in rail
        assert "max-height: 100%" in rail

        body = section.split(".mc-krea-spatial-body {", 1)[1].split("}", 1)[0]

        assert "min-height: 0" in body

    def test_the_frame_is_sized_without_a_viewport_unit(self, section):
        """§5.3. The frame is the shape of the image and the size of the box it
        is in -- and `cqh` is not available under `container-type: inline-size`,
        so a stylesheet that used one would silently size the frame from the
        window instead."""
        frame = section.split(".mc-krea-spatial-canvas {", 1)[1].split("}", 1)[0]

        assert "cqh" not in frame
        assert "vh" not in frame
        assert "aspect-ratio: var(--mc-ar-w" in frame

    def test_a_coarse_pointer_gets_targets_it_can_hit(self, section):
        """§23.2, and the numbers §8.2 asks for."""
        coarse = section.split("@media (pointer: coarse)", 1)[1].split("}", 1)[0]

        assert "--mc-grab: 44px" in coarse
        assert "--mc-touch: 44px" in coarse

    def test_the_takeover_never_hides_the_workspace_s_own_contents(self, rules):
        """The workspace is marked so that it survives as a sibling of the
        things being hidden. Marked and not excluded, it would hide itself."""
        found = [selector for selector, _body in rules
                 if "[data-mc-spatial-path]" in selector
                 and selector.strip().startswith(".mc-krea-spatial-taken [")]

        assert found, "the takeover's descendant rules could not be found"
        for selector in found:
            assert ":not(.mc-krea-spatial-workspace)" in selector, selector

    def test_the_tab_s_own_children_are_hidden_by_a_rule_of_their_own(self, rules):
        """`.mc-krea-spatial-taken [data-mc-spatial-path]` is a descendant
        selector and the class is on the tab, so it never matches the tab --
        which means the tab's own children, the whole of txt2img, need saying
        separately or the takeover hides nothing at all."""
        wanted = ".mc-krea-spatial-taken > *:not([data-mc-spatial-path])"
        found = [body for selector, body in rules if selector == wanted]

        assert found, "the tab's own children are not hidden by anything"
        assert "display: none" in found[0]

    def test_the_takeover_forces_no_display_on_the_tab_itself(self, rules):
        """Gradio hides an inactive tab by setting `display`, so a
        `display: ... !important` matching #tab_txt2img would leave txt2img
        showing on top of img2img the moment somebody changed tab. The class
        the tab carries is only ever used as an ancestor."""
        for selector, body in rules:
            if selector.startswith(".mc-krea-spatial-taken {"):
                assert False, "the tab is styled directly"
            if selector == ".mc-krea-spatial-taken":
                assert "display" not in body, selector
            assert "#tab_txt2img" not in selector, selector

    def test_every_gesture_surface_says_touch_action_none(self, section):
        """§5.7 and §23.3: the page does not scroll under a finger that is
        moving a box, resizing one, drawing one or dragging one out."""
        for surface in (".mc-krea-spatial-canvas {", ".mc-krea-spatial-region {",
                        ".mc-krea-spatial-proxy {", ".mc-krea-spatial-handle {",
                        ".mc-krea-spatial-shape-button {"):
            rule = section.split(surface, 1)[1].split("}", 1)[0]
            assert "touch-action: none" in rule, surface


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

        for kept in ("mc-krea-spatial-quick", "mc-krea-spatial-draw",
                     "mc-krea-spatial-delete", "mc-krea-spatial-duplicate",
                     "mc-krea-spatial-front", "mc-krea-spatial-lower",
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


# --------------------------------------------------------------------------- #
# Auto Save, in both editors
# --------------------------------------------------------------------------- #


class TestAutoSaveInTheFullEditor:
    """One switch on the panel, and until now only one of the two canvases
    obeyed it.

    Auto Save committed a move on the compact canvas the moment the pointer came
    up, and did nothing at all in the full editor, where every edit waited for
    the Save button. A switch that means different things in two places on the
    same panel is a switch nobody can trust, and the one it is easiest to lose
    work to.
    """

    def test_a_finished_edit_is_committed(self):
        found = run("""
            ks.open();
            ks.reorder("b", "a", true);
            report({published: published.length,
                    ids: JSON.parse(published[published.length - 1] || "{}")
                         .regions.map((region) => region.id)});
        """, initial=document(BURIED))

        assert found["published"] == 1
        assert found["ids"] == ["b", "a"]

    def test_with_the_switch_off_nothing_is_written_until_save(self):
        found = run("""
            autoSaveBox.checked = false;
            ks.open();
            ks.reorder("b", "a", true);
            const during = published.length;
            ks.save();
            report({during: during, after: published.length});
        """, initial=document(BURIED))

        assert found["during"] == 0
        assert found["after"] == 1

    def test_opening_the_editor_commits_nothing(self):
        """An Auto Save that fired on open would put a "changed" mark against a
        layout somebody only looked at."""
        found = run("""
            ks.open();
            report({published: published.length});
        """, initial=document(BURIED))

        assert found["published"] == 0

    def test_a_repaint_that_changed_nothing_costs_no_round_trip(self):
        """Selecting a row repaints the whole editor. Gradio treats every
        publish as an input event and a round trip, and "which row is
        highlighted" is not worth one."""
        found = run("""
            ks.open();
            ks.reorder("b", "a", true);
            const after = published.length;
            ks.select("a");
            ks.select("b");
            report({after: after, now: published.length});
        """, initial=document(BURIED))

        assert found["after"] == 1
        assert found["now"] == 1

    def test_typing_does_not_commit_and_leaving_the_field_does(self):
        """§"save when the cursor exits the field": one round trip per field
        rather than one per keystroke."""
        found = run("""
            ks.open();
            ks.select("a");
            field("mc-krea-spatial-prompt", "a lighthouse");
            const typing = published.length;
            commit("mc-krea-spatial-prompt", "a lighthouse");
            report({typing: typing, left: published.length,
                    saved: saved().regions.find((region) => region.id === "a").prompt});
        """, initial=document(BURIED))

        assert found["typing"] == 0
        assert found["left"] == 1
        assert found["saved"] == "a lighthouse"

    def test_a_drawn_region_is_a_finished_edit_like_any_other(self):
        """Drawing one used to be the one edit Auto Save did not hear about:
        the gesture records its own history entry, and recording one is not the
        same as saying an edit finished. What it cost was a box on screen that
        the next Generate did not compose."""
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            report({document: saved()});
        """, initial=document())

        assert found["published"] == 1
        assert len(found["document"]["regions"]) == 2
        assert found["document"]["regions"][1]["bbox"] == [100, 200, 400, 600]

    def test_an_undo_is_committed_too(self):
        """Otherwise Undo would leave the screen and the generation disagreeing,
        which is the one thing Auto Save exists to prevent."""
        found = run("""
            ks.open();
            ks.reorder("b", "a", true);
            ks.undo();
            report({ids: JSON.parse(published[published.length - 1] || "{}")
                         .regions.map((region) => region.id)});
        """, initial=document(BURIED))

        assert found["ids"] == ["a", "b"]
