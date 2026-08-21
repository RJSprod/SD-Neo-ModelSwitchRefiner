"""The Spatial Layout editor, executed rather than read.

Two kinds of question, and only one of them can be answered by reading the file.

**Does it draw what somebody drew?** A drag from one corner to another has to
become a box at those normalized coordinates, a drag the other way has to become
the same box, a drag off the edge has to stop at it, and a click has to become
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

The ids the fake page is built from are read out of ``spatial_editor()``, so a
control renamed in Python and not in JavaScript fails here rather than in a
browser.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_spatial_krea.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def editor_elements() -> list[tuple[str, str]]:
    """``(tag, id)`` for everything the Python-built editor markup carries."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import model_chain_krea_creative as creative_script

    return re.findall(r"<(\w+)[^>]*\bid=\"([^\"]+)\"", creative_script.spatial_editor())


HARNESS = """
// A txt2img page with the Spatial Layout editor in it, a clock a test can move,
// and just enough DOM for the file under test to be the real file.

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
        this.hidden = false;
        this.disabled = false;
        this.value = "";
        this.title = "";
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

ELEMENTS.forEach(function (entry) { make(entry[0], entry[1]); });

// The canvas is 1000 units of layout across, so a normalized coordinate and a
// client coordinate are the same number and a test can say what it means.
const canvas = body.querySelector("#mc-krea-spatial-canvas");
canvas._rect = {left: 0, top: 0, width: 1000, height: 1000};
body.querySelector("#mc-krea-spatial-regions").parentNode = canvas;
canvas.appendChild(body.querySelector("#mc-krea-spatial-regions"));

// The Gradio side: the hidden state box, the Edit button and the size fields.
const stateBox = new El("textarea");
stateBox.value = INITIAL;
const stateHolder = make("div", "mc-krea-spatial-state");
stateHolder.appendChild(stateBox);

const openButton = make("button", "mc-krea-spatial-open");

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
    createElement: (tag) => new El(tag),
    querySelector: (selector) => body.querySelector(selector),
    querySelectorAll: (selector) => body.querySelectorAll(selector),
    getElementById: (id) => body.querySelector("#" + id),
    addEventListener(kind, fn) { (docListeners[kind] = docListeners[kind] || []).push(fn); },
};
const docListeners = {};

// What Gradio does when it rebuilds a tab: the HTML component's markup comes
// back, ids and all, while the copy this file moved to the body is still there.
function rebuildMarkup() {
    const fresh = new El("div");
    ELEMENTS.forEach(function (entry) {
        const element = new El(entry[0]);
        element.id = entry[1];
        fresh.appendChild(element);
    });
    page.appendChild(fresh);
    return fresh;
}
globalThis.window = globalThis;
globalThis.gradioApp = () => globalThis.document;
globalThis.Event = function (type) { this.type = type; this.bubbles = true; };

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

loaded.forEach((fn) => fn());

const ks = globalThis.modelChainKreaSpatial;

function fire(kind, event) {
    (docListeners[kind] || []).forEach((fn) => fn(event));
}

function press(id) {
    const element = body.querySelector("#" + id);
    element.dispatchEvent({type: "click", target: element, preventDefault() {}});
}

// Dispatched at the canvas with the deep target set, which is what a bubbling
// mousedown on a region actually looks like: the editor listens once, on the
// canvas, and works out what was under the cursor from event.target.closest.
function drag(from, to, target) {
    canvas.dispatchEvent({type: "mousedown", target: target || canvas,
                          clientX: from[0], clientY: from[1], preventDefault() {}});
    fire("mousemove", {clientX: to[0], clientY: to[1]});
    fire("mouseup", {});
}

function draw(from, to) {
    press("mc-krea-spatial-draw");
    drag(from, to);
}

function key(name, target) {
    fire("keydown", {key: name, target: target || canvas, preventDefault() {}});
}

function field(id, value) {
    const element = body.querySelector("#" + id);
    element.value = value;
    element.dispatchEvent({type: "input", target: element});
}

function saved() {
    return published.length ? JSON.parse(published[published.length - 1]) : null;
}

const report = (extra) => console.log(JSON.stringify(Object.assign({
    timers: timers.length,
    documentListeners: Object.keys(docListeners)
        .reduce((total, kind) => total + docListeners[kind].length, 0),
    overlays: body.querySelectorAll("#mc-krea-spatial-overlay").length,
    generateListeners: generateListeners,
    open: ks.state.open,
    regions: ks.ordered().map(function (region) {
        return {id: region.id, bbox: region.bbox, type: region.type,
                name: region.name, prompt: region.prompt, z: region.z};
    }),
    selected: ks.state.selected,
    published: published.length,
    stateBox: stateBox.value,
}, extra || {})));

BODY
"""


def run(body: str, initial: str = "") -> dict:
    harness = (
        HARNESS.replace("SOURCE", SCRIPT.read_text(encoding="utf-8"))
        .replace("BODY", body)
        .replace("ELEMENTS", json.dumps(editor_elements()))
        .replace("INITIAL", json.dumps(initial))
    )
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


FACE = {"id": "r1", "name": "Face", "type": "obj", "bbox": [35, 55, 315, 360],
        "prompt": "elderly Japanese woman", "framing": "Close-up", "angle": "3/4 left",
        "z": 0}


def document(regions=(FACE,), mode="smart") -> str:
    return json.dumps({"version": 1, "canvas": {"width": 1024, "height": 1344,
                                                "grid": "thirds"},
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
        listener idempotent cannot cover these three. A version without the
        separate flag adds a mousemove handler per Gradio update, and the page
        gets slower the longer somebody uses it."""
        found = run("""
            const before = Object.keys(docListeners)
                .reduce((total, kind) => total + docListeners[kind].length, 0);
            ks.wire(); ks.wire(); ks.wire();
            report({before: before});
        """, initial=document())

        assert found["before"] == 3          # mousemove, mouseup, keydown
        assert found["documentListeners"] == 3

    def test_there_is_no_polling_of_any_kind(self):
        source = SCRIPT.read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("//"))

        assert "setInterval" not in code
        assert "setTimeout" not in code
        assert "requestAnimationFrame" not in code


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
        """§6.3: the release creates *and* selects, so the next thing typed goes
        into the box that was just drawn."""
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            report({focused: !!body.querySelector("#mc-krea-spatial-prompt").focused});
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

    def test_a_click_is_not_a_region(self):
        """And neither is a two-pixel drag, which is a click with a shaky hand.
        Both are dropped here rather than refused later, where the reason would
        arrive on the finished image instead of under the cursor."""
        found = run("""
            ks.open();
            draw([500, 500], [500, 500]);
            draw([500, 500], [503, 504]);
            report();
        """)

        assert found["regions"] == []

    def test_drawing_is_one_region_at_a_time(self):
        """The tool disarms itself on release. A mode that stayed on would turn
        the next attempt to move a box into a new box on top of it."""
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
            canvas.dispatchEvent({type: "mousedown", target: canvas,
                                  clientX: 100, clientY: 100, preventDefault() {}});
            key("Escape");
            report({drawing: ks.state.drawing});
        """)

        assert found["open"] is True
        assert found["drawing"] is False
        assert found["regions"] == []


class TestMovingAndResizing:
    def test_a_region_can_be_dragged(self):
        found = run("""
            ks.open();
            const box = body.querySelector(".mc-krea-spatial-region");
            drag([100, 100], [200, 150], box);
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [135, 105, 415, 410]

    def test_dragging_a_region_cannot_push_it_off_the_canvas(self):
        found = run("""
            ks.open();
            const box = body.querySelector(".mc-krea-spatial-region");
            drag([100, 100], [5000, 5000], box);
            report();
        """, initial=document())

        left, top, right, bottom = found["regions"][0]["bbox"]

        assert right == 1000 and bottom == 1000
        assert right - left == 315 - 35
        assert bottom - top == 360 - 55

    def test_a_corner_handle_resizes_rather_than_moves(self):
        found = run("""
            ks.open();
            const handle = body.querySelector(".mc-krea-spatial-handle-se");
            drag([315, 360], [500, 500], handle);
            report();
        """, initial=document())

        assert found["regions"][0]["bbox"] == [35, 55, 500, 500]


class TestTheRegionList:
    def test_delete_removes_the_selected_region(self):
        found = run("""
            ks.open();
            key("Delete");
            report();
        """, initial=document())

        assert found["regions"] == []

    def test_delete_does_nothing_while_somebody_is_typing(self):
        """The single most annoying way an editor can lose work: backspacing in
        a prompt field and having the region disappear."""
        found = run("""
            ks.open();
            key("Backspace", body.querySelector("#mc-krea-spatial-prompt"));
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


class TestTheInspector:
    def test_typing_a_prompt_reaches_the_region(self):
        found = run("""
            ks.open();
            field("mc-krea-spatial-prompt", "a red bicycle");
            report();
        """, initial=document())

        assert found["regions"][0]["prompt"] == "a red bicycle"

    def test_switching_to_a_text_region_reveals_the_text_field(self):
        found = run("""
            ks.open();
            field("mc-krea-spatial-type", "text");
            report({textShown: !body.querySelector("#mc-krea-spatial-text-field").hidden,
                    framingShown:
                        !body.querySelector("#mc-krea-spatial-framing-field").hidden});
        """, initial=document())

        assert found["regions"][0]["type"] == "text"
        assert found["textShown"] is True
        # Framing and angle are about how a subject is shot. A line of type has
        # neither, and offering them would invite a selection that renders
        # nothing.
        assert found["framingShown"] is False

    def test_the_box_readout_is_the_normalized_one(self):
        found = run("""
            ks.open();
            report({readout: body.querySelector("#mc-krea-spatial-bbox").textContent});
        """, initial=document())

        assert found["readout"] == "35, 55, 315, 360"


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #


class TestSavingAndCancelling:
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
                                               "grid": "thirds"}

    def test_cancel_publishes_nothing(self):
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            press("mc-krea-spatial-cancel");
            report();
        """, initial=document())

        assert found["published"] == 0
        assert found["stateBox"] == document()
        assert found["open"] is False

    def test_escape_with_nothing_in_progress_cancels(self):
        found = run("""
            ks.open();
            draw([100, 200], [400, 600]);
            key("Escape");
            report();
        """, initial=document())

        assert found["published"] == 0
        assert found["open"] is False

    def test_the_editor_opens_on_what_was_last_saved(self):
        found = run("""
            ks.open();
            report();
        """, initial=document())

        assert [region["id"] for region in found["regions"]] == ["r1"]
        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

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


class TestTheFrame:
    def test_the_canvas_takes_the_shape_of_the_image(self):
        found = run("""
            ks.open();
            report({ratio: canvas.style.aspectRatio,
                    size: body.querySelector("#mc-krea-spatial-size").textContent});
        """, initial=document())

        assert found["ratio"] == "1024 / 1344"
        assert found["size"] == "1024 × 1344"

    def test_a_reshaped_frame_warns_and_changes_nothing(self):
        """§6.4 in the strongest terms it uses: never silently delete layout
        state. Reprojecting would be this file deciding which of somebody's
        boxes deserved to keep its shape."""
        found = run("""
            width.value = "1344";
            height.value = "1024";
            ks.open();
            const warning = body.querySelector("#mc-krea-spatial-warning");
            report({warned: !warning.hidden, text: warning.textContent});
        """, initial=document())

        assert found["warned"] is True
        assert "unchanged" in found["text"]
        assert found["regions"][0]["bbox"] == [35, 55, 315, 360]

    def test_the_same_shape_at_a_different_size_says_nothing(self):
        found = run("""
            width.value = "1536";
            height.value = "2016";
            ks.open();
            report({warned:
                !body.querySelector("#mc-krea-spatial-warning").hidden});
        """, initial=document())

        assert found["warned"] is False


class TestARebuiltPage:
    """Gradio rebuilds parts of a tab on some updates, and the editor is the one
    element in this extension that does not live where Gradio put it."""

    def test_a_rebuilt_editor_does_not_leave_two_of_everything(self):
        """Both copies carry every id, so whichever ``byId`` found first would
        be a coin toss between a wired overlay and an unwired one."""
        found = run("""
            rebuildMarkup();
            ks.wire();
            ks.open();
            draw([100, 200], [400, 600]);
            report();
        """, initial=document())

        assert found["overlays"] == 1
        assert found["open"] is True
        assert len(found["regions"]) == 2

    def test_the_overlay_ends_up_on_the_body(self):
        """A position: fixed modal inside an accordion is one overflow: hidden
        away from being a modal nobody can see."""
        found = run("""
            report({onBody: body.children.indexOf(
                body.querySelector("#mc-krea-spatial-overlay")) >= 0});
        """, initial=document())

        assert found["onBody"] is True


class TestTheMarkupAndTheFileAgree:
    def test_every_id_the_editor_uses_is_an_id_python_emits(self):
        """The one failure mode a fake page cannot catch on its own: a control
        renamed on one side of the wire. The harness is built from
        ``spatial_editor()``, so a mismatch is a null dereference here rather
        than a dead button in a browser."""
        emitted = {identifier for _tag, identifier in editor_elements()}
        used = set(re.findall(r'"(mc-krea-spatial-[a-z-]+)"',
                              SCRIPT.read_text(encoding="utf-8")))
        # The state box and the open button are Gradio's components, not the
        # editor markup's, so they are named by the panel instead.
        external = {"mc-krea-spatial-state", "mc-krea-spatial-open"}

        assert used - external - {"mc-krea-spatial"} <= emitted | {
            "mc-krea-spatial-region", "mc-krea-spatial-handle", "mc-krea-spatial-ghost",
            "mc-krea-spatial-label"}
