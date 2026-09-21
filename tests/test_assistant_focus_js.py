"""Workspace focus, as a transaction, exercised rather than read.

Entering is easy. Leaving is the part that has to be right, on a page whose DOM
another extension may have changed in the meantime -- so the tests are mostly
about exit:

    every inline style that was changed is put back to *the value it had*,
    which for most of them is "no inline style at all", and never to a guess
    like ``overflow: auto``;

    ``inert`` on a sibling that already had it is left set, because a page
    where something else had already made a panel unreachable would come back
    subtly broken and nobody would connect it to this;

    nothing is inerted that contains the assistant, or the focused workspace --
    inerting an ancestor inerts everything inside it, including the control
    that turns focus off;

    an ancestor with a transform is reported as a reason rather than
    discovered as a panel that fills a box instead of the screen, which is the
    silent failure ``position: fixed`` has.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile

import pytest

FOCUS = pathlib.Path(__file__).resolve().parent.parent / "javascript" \
    / "forge_assistant_focus.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


HARNESS = r"""
// A small tree with opinions a test can set: a tab bar holding three panels,
// a header and a footer beside them, and the assistant's own root.

const nodes = [];

function make(id, parent) {
    const node = {
        id,
        parentElement: parent || null,
        children: [],
        attributes: {},
        inertAttribute: false,
        // As faithful as this needs to be: a real CSSStyleDeclaration exposes
        // each property directly *and* through setProperty/removeProperty, and
        // the two are the same storage. A stub with only the methods would let
        // "record the inline value, then restore it" pass while recording
        // nothing at all.
        style: (function () {
            const camel = (name) => name.replace(/-([a-z])/g,
                                                 (m, letter) => letter.toUpperCase());
            const declaration = {
                setProperty(name, value) { declaration[camel(name)] = value; },
                removeProperty(name) { delete declaration[camel(name)]; },
            };
            return declaration;
        })(),
        classList: {
            names: new Set(),
            add(...names) { names.forEach((n) => node.classList.names.add(n)); },
            remove(...names) { names.forEach((n) => node.classList.names.delete(n)); },
            toggle(name, on) {
                if (on) node.classList.add(name);
                else node.classList.remove(name);
            },
            contains(name) { return node.classList.names.has(name); },
        },
        computed: {},
        get inert() { return node.inertAttribute; },
        set inert(value) { node.inertAttribute = !!value; },
        setAttribute(name, value) { node.attributes[name] = value; },
        getAttribute(name) { return node.attributes[name] === undefined ? null
                                                                        : node.attributes[name]; },
        removeAttribute(name) {
            delete node.attributes[name];
            if (name === "inert") node.inertAttribute = false;
        },
        hasAttribute(name) { return node.attributes[name] !== undefined; },
        querySelector() { return null; },
        contains(other) {
            let walk = other;
            while (walk) {
                if (walk === node) return true;
                walk = walk.parentElement;
            }
            return false;
        },
        focus() { node.focused = true; },
    };
    if (parent) parent.children.push(node);
    nodes.push(node);
    return node;
}

const documentElement = make("html", null);
const body = make("body", documentElement);
const header = make("header", body);
const tabs = make("tabs", body);
const footer = make("footer", body);
const assistant = make("forge-assistant-root", body);
const txt2img = make("tab_txt2img", tabs);
const studio = make("tab_llm_studio", tabs);
const paint = make("tab_minipaint", tabs);

// Somebody else got there first: this sibling was already unreachable before
// focus was ever entered, and it has to stay that way afterwards.
footer.inert = true;
footer.setAttribute("inert", "");
// And one that carries an inline style focus will change.
body.style.overflow = "scroll";

globalThis.document = {
    documentElement,
    body,
    activeElement: null,
    getElementById: (id) => nodes.find((node) => node.id === id) || null,
    contains: (node) => nodes.indexOf(node) >= 0,
};
globalThis.window = globalThis;
globalThis.getComputedStyle = (node) => Object.assign(
    {transform: "none", filter: "none", perspective: "none", backdropFilter: "none",
     contain: "none", willChange: "auto"},
    node.computed || {});
globalThis.ResizeObserver = function (fn) {
    globalThis.resizeCallback = fn;
    return {
        observe() { globalThis.observed = true; },
        disconnect() { globalThis.observed = false; },
    };
};
globalThis.console = console;

const HOST = {
    resolveWorkspaceRoot: (id) => nodes.find((node) => node.id === id) || null,
};

__SOURCE__

const NS = globalThis.forgeAssistant;
const focus = NS.focus;

__SCENARIO__
"""


def run(scenario: str) -> dict:
    harness = (HARNESS
               .replace("__SOURCE__", FOCUS.read_text(encoding="utf-8"))
               .replace("__SCENARIO__", scenario))
    with tempfile.TemporaryDirectory() as room:
        entry = pathlib.Path(room) / "scenario.mjs"
        entry.write_text(harness, encoding="utf-8")
        result = subprocess.run(["node", str(entry)], capture_output=True, text=True,
                                timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestEntering:
    def test_the_workspace_takes_the_class_and_the_body_stops_scrolling(self):
        found = run("""
            const answer = focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                ok: answer.ok,
                root: document.getElementById("tab_llm_studio")
                    .classList.contains("forge-assistant-focus-root"),
                body: document.body.classList.contains("forge-assistant-focused"),
                overflow: document.body.style.overflow || null,
                active: focus.activeWorkspace(),
            }));
        """)

        assert found["ok"] is True
        assert found["root"] is True
        assert found["body"] is True
        assert found["overflow"] == "hidden"
        assert found["active"] == "tab_llm_studio"

    def test_the_covered_siblings_become_unreachable(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                header: document.getElementById("header").inert,
                txt2img: document.getElementById("tab_txt2img").inert,
                studio: document.getElementById("tab_llm_studio").inert,
            }));
        """)

        assert found["header"] is True
        assert found["txt2img"] is True
        assert found["studio"] is False, "the workspace inerted itself"

    def test_the_assistant_is_never_made_unreachable(self):
        """It carries the control that turns focus off. A focus mode that
        inerts its own escape hatch is a page somebody has to reload."""
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                assistant: document.getElementById("forge-assistant-root").inert,
            }));
        """)

        assert found["assistant"] is False

    def test_no_ancestor_is_ever_inerted(self):
        """Inerting an ancestor inerts the root inside it, and the failure
        looks exactly like "focus mode does nothing"."""
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                body: document.body.inert,
                tabs: document.getElementById("tabs").inert,
                html: document.documentElement.inert,
            }));
        """)

        assert found["body"] is False
        assert found["tabs"] is False
        assert found["html"] is False

    def test_a_workspace_with_no_panel_is_refused_with_a_reason(self):
        found = run("""
            const answer = focus.enter("a_tab_that_is_not_here", HOST);
            console.log(JSON.stringify(answer));
        """)

        assert found["ok"] is False
        assert found["reason"]

    def test_an_ancestor_with_a_transform_is_reported_by_name(self):
        """`position: fixed` resolves against a transformed ancestor rather
        than the viewport, so the panel fills a box instead of the screen --
        and nothing says so."""
        found = run("""
            document.getElementById("tabs").computed = {transform: "translateZ(0)"};
            const answer = focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify(answer));
        """)

        assert found["ok"] is False
        assert "transform" in found["reason"]

    @pytest.mark.parametrize("prop,value,word", [("filter", "blur(2px)", "filter"),
                                                 ("contain", "paint", "contain"),
                                                 ("perspective", "800px", "perspective")])
    def test_every_containing_block_trap_is_caught(self, prop, value, word):
        found = run("""
            document.getElementById("tabs").computed = {%s: "%s"};
            console.log(JSON.stringify(focus.enter("tab_llm_studio", HOST)));
        """ % (prop, value))

        assert found["ok"] is False
        assert word in found["reason"]


class TestLeaving:
    def test_the_class_and_the_scroll_lock_come_off(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            focus.exit();
            console.log(JSON.stringify({
                root: document.getElementById("tab_llm_studio")
                    .classList.contains("forge-assistant-focus-root"),
                body: document.body.classList.contains("forge-assistant-focused"),
                overflow: document.body.style.overflow || null,
                active: focus.isActive(),
            }));
        """)

        assert found["root"] is False
        assert found["body"] is False
        assert found["overflow"] == "scroll", (
            "the inline style was replaced with a guess rather than restored")
        assert found["active"] is False

    def test_a_sibling_that_was_already_unreachable_stays_that_way(self):
        """Never a blind ``inert = false``. A page where something else had
        already made a panel unreachable would come back subtly broken and
        nobody would connect it to this."""
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            focus.exit();
            console.log(JSON.stringify({
                footer: document.getElementById("footer").inert,
                header: document.getElementById("header").inert,
            }));
        """)

        assert found["footer"] is True
        assert found["header"] is False

    def test_focus_is_returned_to_where_it_was(self):
        found = run("""
            const started = document.getElementById("header");
            document.activeElement = started;
            focus.enter("tab_llm_studio", HOST);
            focus.exit();
            console.log(JSON.stringify({restored: !!started.focused}));
        """)

        assert found["restored"] is True

    def test_exiting_twice_is_harmless(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                first: focus.exit(),
                second: focus.exit(),
            }));
        """)

        assert found["first"] is True
        assert found["second"] is False

    def test_a_root_detached_while_focused_still_releases_the_page(self):
        """A workspace rebuilt by a Gradio update is still a workspace whose
        siblings have to come back."""
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            const root = document.getElementById("tab_llm_studio");
            root.parentElement = null;
            focus.exit();
            console.log(JSON.stringify({
                body: document.body.classList.contains("forge-assistant-focused"),
                header: document.getElementById("header").inert,
                overflow: document.body.style.overflow || null,
            }));
        """)

        assert found["body"] is False
        assert found["header"] is False
        assert found["overflow"] == "scroll"


class TestSwitching:
    def test_the_treatment_moves_with_the_workspace(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            const answer = focus.moveTo("tab_txt2img", HOST);
            console.log(JSON.stringify({
                ok: answer.ok,
                studio: document.getElementById("tab_llm_studio")
                    .classList.contains("forge-assistant-focus-root"),
                txt2img: document.getElementById("tab_txt2img")
                    .classList.contains("forge-assistant-focus-root"),
                studioInert: document.getElementById("tab_llm_studio").inert,
            }));
        """)

        assert found["ok"] is True
        assert found["studio"] is False
        assert found["txt2img"] is True
        assert found["studioInert"] is True

    def test_an_unsupported_destination_restores_the_page_and_says_why(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            document.getElementById("tabs").computed = {transform: "scale(1)"};
            const answer = focus.moveTo("tab_txt2img", HOST);
            console.log(JSON.stringify({
                ok: answer.ok,
                reason: answer.reason,
                active: focus.isActive(),
                body: document.body.classList.contains("forge-assistant-focused"),
                header: document.getElementById("header").inert,
            }));
        """)

        assert found["ok"] is False
        assert found["reason"]
        assert found["active"] is False
        assert found["body"] is False
        assert found["header"] is False, "the page was left half-focused"

    def test_moving_when_focus_is_off_does_nothing(self):
        found = run("""
            console.log(JSON.stringify(focus.moveTo("tab_txt2img", HOST)));
        """)

        assert found["ok"] is False


class TestAdapters:
    def test_a_canvas_editor_gets_its_own_adapter(self):
        found = run("""
            console.log(JSON.stringify({
                minipaint: !!focus.adapterFor("tab_minipaint"),
                registered: focus.adapterFor("minipaint").workspaceId,
                fallback: focus.adapterFor("tab_txt2img") === focus.defaultAdapter,
            }));
        """)

        assert found["registered"] == "minipaint"
        assert found["fallback"] is True

    def test_an_adapter_can_be_added_for_an_installation_s_own_tab(self):
        found = run("""
            let entered = 0;
            focus.registerAdapter({
                workspaceId: "tab_txt2img",
                canFocus: () => ({ok: true, reason: ""}),
                enter() { entered += 1; },
                exit() {},
                onResize() {},
            });
            focus.enter("tab_txt2img", HOST);
            focus.exit();
            console.log(JSON.stringify({entered}));
        """)

        assert found["entered"] == 1

    def test_an_adapter_that_throws_does_not_leave_the_page_focused(self):
        """Polish must never be able to trap somebody in a mode they cannot
        leave."""
        found = run("""
            focus.registerAdapter({
                workspaceId: "tab_txt2img",
                canFocus: () => ({ok: true, reason: ""}),
                enter() {},
                exit() { throw new Error("the editor is on fire"); },
                onResize() {},
            });
            focus.enter("tab_txt2img", HOST);
            focus.exit();
            console.log(JSON.stringify({
                active: focus.isActive(),
                body: document.body.classList.contains("forge-assistant-focused"),
                header: document.getElementById("header").inert,
            }));
        """)

        assert found["active"] is False
        assert found["body"] is False
        assert found["header"] is False

    def test_an_adapter_that_refuses_is_honoured(self):
        found = run("""
            focus.registerAdapter({
                workspaceId: "tab_txt2img",
                canFocus: () => ({ok: false, reason: "this editor is mid-save"}),
                enter() {}, exit() {}, onResize() {},
            });
            console.log(JSON.stringify(focus.enter("tab_txt2img", HOST)));
        """)

        assert found["ok"] is False
        assert found["reason"] == "this editor is mid-save"

    def test_the_workspace_is_watched_for_resizes_while_focused(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            const watching = globalThis.observed;
            focus.exit();
            console.log(JSON.stringify({watching, after: globalThis.observed}));
        """)

        assert found["watching"] is True
        assert found["after"] is False, "an observer outlived the focus it was for"
