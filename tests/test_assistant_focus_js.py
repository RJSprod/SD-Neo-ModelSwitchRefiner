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
    def test_the_workspace_takes_the_class_and_nothing_else_changes(self):
        """The whole mechanism, and the whole of what it is allowed to do.

        The class is what the stylesheet turns into `position: fixed; inset: 0`
        over an opaque background. Everything else about the page is left
        exactly as it was, which is the property that makes leaving safe.
        """
        found = run("""
            const answer = focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                ok: answer.ok,
                root: document.getElementById("tab_llm_studio")
                    .classList.contains("forge-assistant-focus-root"),
                body: document.body.classList.contains("forge-assistant-focused"),
                active: focus.activeWorkspace(),
            }));
        """)

        assert found["ok"] is True
        assert found["root"] is True
        assert found["body"] is True
        assert found["active"] == "tab_llm_studio"

    def test_the_page_is_left_scrollable(self):
        """Reported: on Txt2Img the tab bar stayed exactly where it was and the
        only thing that changed was that the page would no longer scroll.

        Locking the body was the half of the old implementation that people
        could see. It was also unnecessary -- the focus root covers the
        viewport, so there is nothing behind it to scroll to -- and the
        stylesheet contains the wheel with `overscroll-behavior` instead.
        """
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({overflow: document.body.style.overflow || null}));
        """)

        assert found["overflow"] == "scroll", (
            "focus mode changed how the page scrolls")

    def test_the_host_s_own_dom_is_never_touched(self):
        """The tab bar is still there, underneath. Nothing of the host's is
        hidden, moved, restyled or made unreachable -- which is what makes
        leaving one class removal rather than an undo log."""
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                header: document.getElementById("header").inert,
                tabs: document.getElementById("tabs").inert,
                txt2img: document.getElementById("tab_txt2img").inert,
                body: document.body.inert,
                html: document.documentElement.inert,
                assistant: document.getElementById("forge-assistant-root").inert,
                headerClasses: Array.from(
                    document.getElementById("header").classList.names).length,
            }));
        """)

        assert found["header"] is False
        assert found["tabs"] is False
        assert found["txt2img"] is False
        assert found["body"] is False
        assert found["html"] is False
        assert found["assistant"] is False
        assert found["headerClasses"] == 0

    def test_a_sibling_that_was_already_unreachable_is_left_alone(self):
        """The footer in this fixture was inert before focus was ever entered.
        Focus neither sets nor clears it, because focus does not touch it."""
        found = run("""
            const before = document.getElementById("footer").inert;
            focus.enter("tab_llm_studio", HOST);
            const during = document.getElementById("footer").inert;
            focus.exit();
            console.log(JSON.stringify({
                before, during, after: document.getElementById("footer").inert,
            }));
        """)

        assert found["before"] is True
        assert found["during"] is True
        assert found["after"] is True

    def test_a_workspace_with_no_panel_is_refused_with_a_reason(self):
        found = run("""
            const answer = focus.enter("a_tab_that_is_not_here", HOST);
            console.log(JSON.stringify(answer));
        """)

        assert found["ok"] is False
        assert found["reason"]

    def test_an_ancestor_with_a_transform_is_a_note_rather_than_a_refusal(self):
        """`position: fixed` resolves against a transformed ancestor rather than
        the viewport, so the workspace fills that box instead of the screen.

        That used to refuse outright, which meant a press that did nothing at
        all and a sentence nobody could act on. It enters and says so now: the
        chrome is taken out of the layout as well as covered, so a workspace
        that cannot quite escape its ancestor still fills what is left of the
        window -- which is the thing somebody asked for.
        """
        found = run("""
            document.getElementById("tabs").computed = {transform: "translateZ(0)"};
            const answer = focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                ok: answer.ok, note: answer.note, active: focus.isActive(),
                root: document.getElementById("tab_llm_studio")
                    .classList.contains("forge-assistant-focus-root"),
            }));
        """)

        assert found["ok"] is True
        assert found["active"] is True
        assert found["root"] is True
        assert "transform" in found["note"]

    @pytest.mark.parametrize("prop,value,word", [("filter", "blur(2px)", "filter"),
                                                 ("contain", "paint", "contain"),
                                                 ("perspective", "800px", "perspective")])
    def test_every_containing_block_trap_is_named(self, prop, value, word):
        found = run("""
            document.getElementById("tabs").computed = {%s: "%s"};
            console.log(JSON.stringify(focus.enter("tab_llm_studio", HOST)));
        """ % (prop, value))

        assert found["ok"] is True
        assert word in found["note"]

    def test_the_chrome_is_taken_out_of_the_layout_not_merely_covered(self):
        """Reported against the Lobe theme: the workspace was laid over the page
        and the theme's header stayed exactly where it was.

        A z-index wins over ordinary content. It does not win over a header that
        is positioned and has a stacking context of its own, which is what a
        theme that draws its own chrome produces. So the workspace's ancestors
        are marked, and one stylesheet rule hides every child of a marked
        element that is not itself on the path -- the header, the tab bar, the
        sidebars, the footer and the other tabs -- without this code having to
        know what any of them are called.
        """
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            const marked = (id) => document.getElementById(id)
                .classList.contains("forge-assistant-focus-path");
            console.log(JSON.stringify({
                body: marked("body"),
                tabs: marked("tabs"),
                header: marked("header"),
                footer: marked("footer"),
                workspace: marked("tab_llm_studio"),
                assistant: marked("forge-assistant-root"),
            }));
        """)

        # On the path: the workspace's own ancestors, and only those.
        assert found["body"] is True
        assert found["tabs"] is True
        # Not on the path, so the rule hides them.
        assert found["header"] is False
        assert found["footer"] is False
        assert found["assistant"] is False
        # The workspace itself is the root, not a path element.
        assert found["workspace"] is False

    def test_the_marks_come_off_again(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            focus.exit();
            console.log(JSON.stringify({
                left: nodes.filter((n) =>
                    n.classList.contains("forge-assistant-focus-path")).map((n) => n.id),
            }));
        """)

        assert found["left"] == []

    def test_the_stylesheet_hides_what_the_path_leaves_out(self):
        """The class is only half of it; the rule that reads it is the other."""
        css = (pathlib.Path(__file__).resolve().parent.parent
               / "style.css").read_text(encoding="utf-8")

        assert "body.forge-assistant-focused .forge-assistant-focus-path > *" in css
        rule = css.split("body.forge-assistant-focused .forge-assistant-focus-path > *",
                         1)[1].split("}", 1)[0]
        assert "display: none" in rule
        assert ":not(.forge-assistant-focus-path)" in rule
        assert ":not(.forge-assistant-focus-root)" in rule
        assert ":not(#forge-assistant-root)" in rule, (
            "the assistant carries the control that turns focus off")

    def test_the_tab_bar_is_hidden_wherever_the_theme_has_put_it(self):
        """The rule above finds the tab bar as a child of a marked ancestor,
        which is where Forge leaves it. A theme is free to move it, and Lobe
        draws its own header out of it -- which is what was still on screen
        when this was reported a second time."""
        css = (pathlib.Path(__file__).resolve().parent.parent
               / "style.css").read_text(encoding="utf-8")

        assert 'body.forge-assistant-focused .tab-nav' in css
        rule = css.split('body.forge-assistant-focused .tab-nav', 1)[1].split("}", 1)[0]
        assert '[role="tablist"]' in rule, "a themed bar may carry only the role"
        assert "display: none" in rule
        assert "body.forge-assistant-focused" in rule.split("{")[0] \
            or "body.forge-assistant-focused" in css.split(".tab-nav", 1)[0][-80:], (
                "the rule has to be inert while focus is off")


class TestLeaving:
    def test_leaving_is_the_class_coming_off(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            focus.exit();
            console.log(JSON.stringify({
                root: document.getElementById("tab_llm_studio")
                    .classList.contains("forge-assistant-focus-root"),
                body: document.body.classList.contains("forge-assistant-focused"),
                active: focus.isActive(),
            }));
        """)

        assert found["root"] is False
        assert found["body"] is False
        assert found["active"] is False

    def test_the_workspace_comes_back_where_it_was_scrolled_to(self):
        """The one thing worth saving and restoring, and the only one left.

        Focus changes the workspace's box from "a row in the page" to "the whole
        window" and back, which resets its scroll. A focus that returned
        somebody to the top of a long tab is a focus nobody uses twice.
        """
        found = run("""
            const root = document.getElementById("tab_llm_studio");
            root.scrollTop = 420;
            focus.enter("tab_llm_studio", HOST);
            root.scrollTop = 0;               // what the reflow does
            focus.exit();
            console.log(JSON.stringify({scrollTop: root.scrollTop}));
        """)

        assert found["scrollTop"] == 420

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
        """A workspace rebuilt by a Gradio update is still a workspace that has
        to stop being fixed to the viewport."""
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            const root = document.getElementById("tab_llm_studio");
            root.parentElement = null;
            focus.exit();
            console.log(JSON.stringify({
                body: document.body.classList.contains("forge-assistant-focused"),
                root: root.classList.contains("forge-assistant-focus-root"),
                active: focus.isActive(),
            }));
        """)

        assert found["body"] is False
        assert found["root"] is False
        assert found["active"] is False


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
        assert found["studioInert"] is False, "the workspace it left was altered"

    def test_a_destination_with_no_panel_restores_the_page_and_says_why(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            const answer = focus.moveTo("a_tab_that_is_not_here", HOST);
            console.log(JSON.stringify({
                ok: answer.ok,
                reason: answer.reason,
                active: focus.isActive(),
                body: document.body.classList.contains("forge-assistant-focused"),
                left: nodes.filter((n) =>
                    n.classList.contains("forge-assistant-focus-path")).map((n) => n.id),
            }));
        """)

        assert found["ok"] is False
        assert found["reason"]
        assert found["active"] is False
        assert found["body"] is False
        assert found["left"] == [], "the page was left half-focused"

    def test_moving_when_focus_is_off_does_nothing(self):
        found = run("""
            console.log(JSON.stringify(focus.moveTo("tab_txt2img", HOST)));
        """)

        assert found["ok"] is False


class TestTheRootItIsGiven:
    """The bug that made focus mode look like it did nothing.

    The root came from the host adapter, which paired tab buttons with tab
    panels by position among ``#tabs``'s children with ids. On the installed
    page one of those candidates *contained the tab bar* -- so the class went
    onto a box holding the tab bar and the workspace together, that box was
    laid over the window, and what the person saw was the tab bar exactly where
    it had been and a page that would no longer scroll.
    """

    def test_a_candidate_holding_the_tab_bar_is_not_a_workspace(self):
        import json as _json
        import subprocess as _subprocess
        import tempfile as _tempfile

        harness = """
        const nodes = [];
        function make(id, parent, className) {
            const node = {
                id, parentElement: parent || null, children: [],
                className: className || "",
                classList: {names: new Set((className || "").split(" ").filter(Boolean)),
                            contains(n) { return node.classList.names.has(n); }},
                getAttribute() { return null; },
                querySelectorAll(selector) {
                    // Only the two shapes the adapter asks for.
                    if (selector === "button") {
                        return node.children.filter((c) => c.tagName === "BUTTON");
                    }
                    return node.children.filter((c) => c.id);
                },
                querySelector(selector) {
                    if (selector.indexOf("tab-nav") >= 0) {
                        return node.children.find((c) => c.classList.contains("tab-nav"))
                            || null;
                    }
                    return null;
                },
            };
            if (parent) parent.children.push(node);
            nodes.push(node);
            return node;
        }
        const tabs = make("tabs", null);
        // A wrapper with an id that holds the tab bar AND the workspaces. This
        // is the shape that broke it.
        const wrapper = make("tab_wrapper", tabs);
        make("nav", wrapper, "tab-nav");
        make("tab_txt2img", tabs);
        globalThis.document = {querySelector: (s) => (s === "#tabs" ? tabs : null)};
        globalThis.window = globalThis;
        globalThis.getComputedStyle = () => ({display: "block", visibility: "visible"});
        globalThis.MutationObserver = function () {
            return {observe() {}, disconnect() {}};
        };
        globalThis.console = console;
        __SOURCE__
        const host = globalThis.forgeAssistant.host();
        console.log(JSON.stringify({
            panels: host.panels().map((p) => p.id),
        }));
        """
        source = (pathlib.Path(__file__).resolve().parent.parent / "javascript"
                  / "forge_assistant_host.js").read_text(encoding="utf-8")
        with _tempfile.TemporaryDirectory() as room:
            entry = pathlib.Path(room) / "scenario.mjs"
            entry.write_text(harness.replace("__SOURCE__", source), encoding="utf-8")
            result = _subprocess.run(["node", str(entry)], capture_output=True, text=True,
                                     timeout=60)
        assert result.returncode == 0, result.stderr
        found = _json.loads(result.stdout.strip().splitlines()[-1])

        assert found["panels"] == ["tab_txt2img"], (
            "a box holding the tab bar was offered as a workspace to fill the window with")


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
