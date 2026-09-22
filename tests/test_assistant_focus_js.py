"""Workspace focus, as a transaction, exercised rather than read.

Entering is easy. Leaving is the part that has to be right, on a page whose DOM
another extension may have changed in the meantime -- so the tests are mostly
about exit:

    every inline style that was changed is put back to *the value it had*,
    which for most of them is "no inline style at all", and never to a guess
    like ``overflow: auto``;

    every class it put on the page comes off again, including the ones on
    elements it went looking for rather than walked to;

    nothing is hidden that contains the focused workspace, or the assistant --
    hiding an ancestor hides everything inside it, including the control that
    turns focus off, and a rule that could not make that distinction is what
    blanked a page on a real host;

    an ancestor with a transform is reported as a note rather than discovered
    as a panel that fills a box instead of the screen, which is the silent
    failure ``position: fixed`` has;

    and the workspace is measured once the marking is done, because the rule
    that hides the chrome is written against somebody else's theme on somebody
    else's Forge -- so when it gets that shape wrong, focus falls back to
    covering the page rather than emptying it.

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
            const priorities = {};
            const declaration = {
                setProperty(name, value, priority) {
                    declaration[camel(name)] = value;
                    priorities[name] = priority || "";
                },
                removeProperty(name) {
                    delete declaration[camel(name)];
                    delete priorities[name];
                },
                getPropertyValue(name) { return declaration[camel(name)] || ""; },
                getPropertyPriority(name) { return priorities[name] || ""; },
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
        // A box a test can set. `painted()` asks the layout whether the
        // workspace is on screen, so a stub that cannot be measured would let
        // the safety valve pass by never firing.
        box: {width: 1280, height: 800},
        getBoundingClientRect() {
            return Object.assign({top: 0, left: 0}, node.box);
        },
        querySelector(selector) { return within(node).find(match(selector)) || null; },
        querySelectorAll(selector) { return within(node).filter(match(selector)); },
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

// Enough of a selector engine for the three shapes this module uses: a class,
// an attribute with a value, and an id -- comma-separated. Written out rather
// than stubbed to `null`, because the tab bars the script marks are *found*
// with one of these, and a query that always answers nothing would make the
// marking untestable and the test a tautology.
function match(selector) {
    const parts = String(selector).split(",").map((piece) => piece.trim());
    return (node) => parts.some((piece) => {
        if (piece.charAt(0) === ".") return node.classList.contains(piece.slice(1));
        if (piece.charAt(0) === "#") return node.id === piece.slice(1);
        const attribute = /^\[([^=\]]+)=?"?([^\]"]*)"?\]$/.exec(piece);
        if (attribute) {
            return node.getAttribute(attribute[1]) === attribute[2];
        }
        return false;
    });
}

function within(node) {
    const found = [];
    (function walk(parent) {
        parent.children.forEach((child) => {
            found.push(child);
            walk(child);
        });
    })(node);
    return found;
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

// Three tab bars, which is the shape that made the last attempt hide the page.
// Forge's own, inside `#tabs`; one a theme has drawn for itself over in the
// header, which is the one a stylesheet rule was added to catch; and a nested
// `gr.Tabs` inside a workspace, which is the one that rule also caught.
const bar = make("tab-bar", tabs);
bar.classList.add("tab-nav");
const themeBar = make("theme-bar", header);
themeBar.setAttribute("role", "tablist");
const nested = make("nested-tabs", studio);
nested.classList.add("tab-nav");
const nestedTxt = make("nested-txt2img-tabs", txt2img);
nestedTxt.classList.add("tab-nav");

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
    querySelector: (selector) => within(documentElement).find(match(selector)) || null,
    querySelectorAll: (selector) => within(documentElement).filter(match(selector)),
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

    def test_a_tab_bar_the_theme_moved_away_is_hidden_too(self):
        """The path rule finds the tab bar as a child of a marked ancestor,
        which is where Forge leaves it. A theme is free to move it, and Lobe
        draws its own header out of it."""
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            const hidden = (id) => document.getElementById(id)
                .classList.contains("forge-assistant-focus-hidden");
            console.log(JSON.stringify({theme: hidden("theme-bar"),
                                        forge: hidden("tab-bar")}));
        """)

        assert found["theme"] is True
        assert found["forge"] is True

    def test_the_workspace_keeps_its_own_nested_tabs(self):
        """This is why the bar is marked by the script and not selected by a
        stylesheet, and it is the defect that hid the page.

        `.tab-nav` and `role="tablist"` are on every nested `gr.Tabs` as well as
        on the one at the top -- Txt2Img is full of them -- so a rule keyed on
        those names alone reaches inside the workspace it is meant to be
        filling. CSS has no way to say "unless it contains the workspace";
        `enter()` says it.
        """
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                nested: document.getElementById("nested-tabs")
                    .classList.contains("forge-assistant-focus-hidden"),
            }));
        """)

        assert found["nested"] is False

    def test_a_tab_bar_that_contains_the_workspace_is_never_hidden(self):
        """The other half of the same defect, and the one that actually blanked
        the page: on a host where the role sits on a container rather than on
        the strip of buttons, that container is an *ancestor* of the workspace
        and hiding it hides everything."""
        found = run("""
            document.getElementById("tabs").setAttribute("role", "tablist");
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                tabs: document.getElementById("tabs")
                    .classList.contains("forge-assistant-focus-hidden"),
                workspace: document.getElementById("tab_llm_studio")
                    .classList.contains("forge-assistant-focus-root"),
            }));
        """)

        assert found["tabs"] is False
        assert found["workspace"] is True

    def test_the_marks_on_the_tab_bars_come_off_again(self):
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            focus.exit();
            console.log(JSON.stringify({
                left: nodes.filter((n) =>
                    n.classList.contains("forge-assistant-focus-hidden")).map((n) => n.id),
            }));
        """)

        assert found["left"] == []

    def test_a_page_that_goes_blank_undoes_itself(self):
        """Reported in use: the whole page went blank.

        The rule that takes the chrome out of the layout is written against a
        shape this code cannot see -- somebody else's theme, on somebody else's
        Forge. When it gets that shape wrong the cost is not "focus mode looks
        odd", it is a page with nothing on it and nothing to say why.

        So the one thing focus mode exists to show is measured, after the
        marking and before anybody looks at it. If the workspace is not being
        painted, every mark comes off and what is left is the plain overlay:
        imperfect under a theme that draws its own header, and never blank.
        """
        found = run("""
            document.getElementById("tab_llm_studio").box = {width: 0, height: 0};
            const answer = focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                ok: answer.ok,
                note: answer.note,
                active: focus.isActive(),
                root: document.getElementById("tab_llm_studio")
                    .classList.contains("forge-assistant-focus-root"),
                body: document.body.classList.contains("forge-assistant-focused"),
                marked: nodes.filter((n) =>
                    n.classList.contains("forge-assistant-focus-path")
                    || n.classList.contains("forge-assistant-focus-hidden"))
                    .map((n) => n.id),
            }));
        """)

        assert found["ok"] is True
        assert found["active"] is True, "the overlay stays, so Escape still leaves"
        assert found["root"] is True
        assert found["body"] is True
        assert found["marked"] == [], "everything that could hide the page comes off"
        assert found["note"], "a mode that quietly did half its job says so"

    def test_a_page_that_is_painted_keeps_its_marks(self):
        """The other side of the valve. A valve that fires on a working page is
        a valve that has quietly turned the feature off."""
        found = run("""
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                tabs: document.getElementById("tabs")
                    .classList.contains("forge-assistant-focus-path"),
                theme: document.getElementById("theme-bar")
                    .classList.contains("forge-assistant-focus-hidden"),
            }));
        """)

        assert found["tabs"] is True
        assert found["theme"] is True

    def test_a_host_with_no_layout_to_ask_is_given_the_benefit_of_the_doubt(self):
        """Measuring can fail -- a detached node, a host that does not
        implement it. Throwing away a mode that may well be working, on a
        question that could not be answered, is the wrong default."""
        found = run("""
            const root = document.getElementById("tab_llm_studio");
            root.getBoundingClientRect = () => { throw new Error("no layout"); };
            focus.enter("tab_llm_studio", HOST);
            console.log(JSON.stringify({
                tabs: document.getElementById("tabs")
                    .classList.contains("forge-assistant-focus-path"),
            }));
        """)

        assert found["tabs"] is True

    def test_under_the_lobe_theme_the_header_is_a_hidden_sibling(self):
        """sd-webui-lobe-theme, read from its source: a React root appended to
        <gradio-app>, Gradio's container moved inside that root's <main>, and a
        header of its own -- `position: sticky; z-index: 999` -- beside it. A
        z-index does not beat that header. Being the child of a marked
        ancestor that is not itself on the path does, and so it is hidden by
        the same rule as the footer and the sidebars, without this code ever
        having heard of Lobe.
        """
        found = run("""
            const gradioApp = make("gradio-app", body);
            const root = make("root", gradioApp);
            const lobeHeader = make("lobe-header", root);
            const lobeNav = make("lobe-nav", lobeHeader);
            lobeNav.setAttribute("role", "tablist");
            const main = make("lobe-main", root);
            const aside = make("lobe-aside", main);
            const content = make("lobe-content", main);
            const container = make("gradio-container", content);
            const lobeFooter = make("lobe-footer", root);
            // Gradio's #tabs, moved under the theme's content column.
            body.children = body.children.filter((n) => n !== tabs);
            container.children.push(tabs);
            tabs.parentElement = container;

            const answer = focus.enter("tab_txt2img", HOST);
            const path = (n) => n.classList.contains("forge-assistant-focus-path");
            const hidden = (n) => n.classList.contains("forge-assistant-focus-hidden");
            console.log(JSON.stringify({
                ok: answer.ok,
                // On the path, so their children are subject to the rule ...
                rootMarked: path(root), mainMarked: path(main), tabsMarked: path(tabs),
                // ... and these are the children the rule hides.
                headerMarked: path(lobeHeader), asideMarked: path(aside),
                footerMarked: path(lobeFooter),
                headerParentMarked: path(lobeHeader.parentElement),
                asideParentMarked: path(aside.parentElement),
                // The theme's own tablist, outside the workspace: hidden.
                lobeNavHidden: hidden(lobeNav),
                // A nested bar inside the workspace: left alone.
                nestedHidden: hidden(nestedTxt),
                workspace: document.getElementById("tab_txt2img")
                    .classList.contains("forge-assistant-focus-root"),
            }));
        """)

        assert found["ok"] is True
        assert found["workspace"] is True
        assert found["rootMarked"] and found["mainMarked"] and found["tabsMarked"]
        assert not found["headerMarked"] and not found["asideMarked"] \
            and not found["footerMarked"]
        assert found["headerParentMarked"] and found["asideParentMarked"], (
            "hidden by being the unmarked child of a marked ancestor")
        assert found["lobeNavHidden"] is True
        assert found["nestedHidden"] is False

    def test_the_focus_root_rule_outranks_gradio_s_own(self):
        """Gradio styles a tab panel from a Svelte component whose rules are
        scoped with a generated class, so what ships is
        `div.svelte-<hash> { position: relative; ... }`: a tag and a class,
        which outranks a lone class. Without `!important`, the panel stayed a
        row in the page with the focus rule silently losing -- which is the
        whole of why the first three attempts at focus mode changed nothing
        anybody could see."""
        css = (pathlib.Path(__file__).resolve().parent.parent
               / "style.css").read_text(encoding="utf-8")
        rule = css.split(".forge-assistant-focus-root {", 1)[1].split("}", 1)[0]
        declarations = [line.strip() for line in rule.splitlines()
                        if ":" in line and not line.strip().startswith("/*")
                        and not line.strip().startswith("*")]

        for wanted in ("position: fixed", "inset: 0", "z-index:", "overflow: auto",
                       "display: block"):
            matching = [d for d in declarations if d.startswith(wanted)]
            assert matching, wanted + " is missing from the focus root"
            assert all("!important" in d for d in matching), (
                wanted + " has to carry !important to beat Gradio's scoped rule")

    def test_a_sticky_offset_measured_against_the_workspace_is_closed(self):
        """Reported in use: a blank band above the gallery in focus mode, and
        the gallery at the same place on the screen whether focus was on or
        off.

        Lobe's split previewer makes the results column `position: sticky;
        top: 80px` -- its header's height and a margin -- so the picture stays
        on screen while the prompt column scrolls. In focus mode the header is
        gone and the workspace is its own scroll container, so the same 80px
        is a gap. The offset becomes the root's padding, so a stuck column
        keeps the margin it has at rest.
        """
        found = run("""
            const results = make("txt2img_results", txt2img);
            results.computed = {position: "sticky", top: "80px"};
            txt2img.computed = {paddingTop: "8px"};
            focus.enter("tab_txt2img", HOST);
            console.log(JSON.stringify({
                top: results.style.top,
                priority: results.style.getPropertyPriority("top"),
            }));
        """)

        assert found["top"] == "8px"
        assert found["priority"] == "important", (
            "the theme's declaration carries !important on an id; nothing less wins")

    def test_the_offset_is_put_back_exactly_on_the_way_out(self):
        found = run("""
            const results = make("txt2img_results", txt2img);
            results.computed = {position: "sticky", top: "80px"};
            const own = make("own_offset", txt2img);
            own.computed = {position: "sticky", top: "40px"};
            own.style.setProperty("top", "40px", "important");
            focus.enter("tab_txt2img", HOST);
            focus.exit();
            console.log(JSON.stringify({
                results: results.style.getPropertyValue("top"),
                own: own.style.getPropertyValue("top"),
                ownPriority: own.style.getPropertyPriority("top"),
            }));
        """)

        assert found["results"] == "", "it had no inline top, and has none again"
        assert found["own"] == "40px"
        assert found["ownPriority"] == "important"

    def test_a_sticky_element_inside_its_own_scroller_is_left_alone(self):
        """Its offset is measured against that scroller, not against the
        workspace, and whatever it clears is still there."""
        found = run("""
            const scroller = make("inner_scroller", txt2img);
            scroller.computed = {overflowY: "auto"};
            const toolbar = make("inner_toolbar", scroller);
            toolbar.computed = {position: "sticky", top: "24px"};
            focus.enter("tab_txt2img", HOST);
            console.log(JSON.stringify({top: toolbar.style.getPropertyValue("top")}));
        """)

        assert found["top"] == ""

    def test_an_offset_no_larger_than_the_padding_is_not_an_offset_worth_closing(self):
        found = run("""
            const tidy = make("tidy", txt2img);
            tidy.computed = {position: "sticky", top: "8px"};
            txt2img.computed = {paddingTop: "8px"};
            focus.enter("tab_txt2img", HOST);
            console.log(JSON.stringify({top: tidy.style.getPropertyValue("top")}));
        """)

        assert found["top"] == ""

    def test_the_valve_puts_sticky_offsets_back_too(self):
        """Degrading to the plain overlay means every change comes off, not
        only the classes -- an offset left closed on a page the theme still
        draws its header over would put the gallery under that header."""
        found = run("""
            const results = make("txt2img_results", txt2img);
            results.computed = {position: "sticky", top: "80px"};
            txt2img.box = {width: 0, height: 0};
            focus.enter("tab_txt2img", HOST);
            console.log(JSON.stringify({top: results.style.getPropertyValue("top")}));
        """)

        assert found["top"] == ""

    def test_a_hidden_subtree_is_not_walked(self):
        """An inactive nested tab is a thousand elements nobody can see.
        Skipping it whole is what keeps the toggle a toggle."""
        found = run("""
            const hiddenTab = make("txt2img_lora", txt2img);
            hiddenTab.computed = {display: "none"};
            const inside = make("inside_hidden", hiddenTab);
            inside.computed = {position: "sticky", top: "80px"};
            focus.enter("tab_txt2img", HOST);
            console.log(JSON.stringify({top: inside.style.getPropertyValue("top")}));
        """)

        assert found["top"] == ""

    def test_the_stylesheet_has_a_rule_for_what_the_script_marks(self):
        css = (pathlib.Path(__file__).resolve().parent.parent
               / "style.css").read_text(encoding="utf-8")

        assert ".forge-assistant-focus-hidden {" in css
        rule = css.split(".forge-assistant-focus-hidden {", 1)[1].split("}", 1)[0]
        assert "display: none" in rule

    def test_a_stylesheet_rule_may_not_hide_a_tab_bar_by_name(self):
        """The rule this replaced. It is asserted *absent*, because it is the
        kind of thing that looks like a free safety net and reads as an
        improvement in a diff -- and it hid the whole page on a real host.
        """
        css = (pathlib.Path(__file__).resolve().parent.parent
               / "style.css").read_text(encoding="utf-8")
        rules = [piece.split("{", 1)[0]
                 for piece in css.split("}") if "{" in piece]

        for selector in rules:
            naked = selector.split("/*")[-1]
            assert ".tab-nav" not in naked and "tablist" not in naked, (
                "a tab bar is hidden by the script, which can check whether it "
                "contains the workspace, and never by a selector, which cannot: "
                + selector.strip())


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
