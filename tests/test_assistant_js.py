"""The Forge Assistant's browser code, executed rather than read.

Four things in those files are worth running, and all four are things that
cannot be found by reading:

    *the anchor arithmetic*, because a panel that docks to the wrong place on a
    phone with its keyboard up is off-screen rather than merely misplaced, and
    the mistake that does it -- gap before inset, or the layout viewport instead
    of the visual one -- looks correct on a desktop;

    *the nearest-anchor choice*, because a tie between two anchors has to be
    broken the same way every time or the preview flickers under the finger;

    *the Markdown renderer*, which takes text a language model wrote and turns
    it into HTML this page inserts -- the one place in the whole feature where
    a mistake is a cross-site scripting hole rather than a layout bug;

    *the Escape precedence*, because Escape means six different things on this
    page and the wrong order stops a reply somebody was reading instead of
    closing the menu in front of them.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import tempfile

import pytest

JAVASCRIPT = pathlib.Path(__file__).resolve().parent.parent / "javascript"
SHELL = JAVASCRIPT / "forge_assistant.js"
HOST = JAVASCRIPT / "forge_assistant_host.js"
STORE = JAVASCRIPT / "forge_assistant_store.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


HARNESS = r"""
// A document with enough of a DOM for the modules to load. Nothing here has
// opinions; the tests that need opinions set them in the scenario.

const listeners = {};
const elements = {};

function element(id, tag) {
    const node = {
        id,
        tagName: tag || "DIV",
        dataset: {},
        handlers: {},
        style: {values: {},
                setProperty(name, value) { node.style.values[name] = value; },
                removeProperty(name) { delete node.style.values[name]; },
                getPropertyValue(name) { return node.style.values[name] || ""; }},
        hidden: false,
        children: [],
        className: "",
        textContent: "",
        innerHTML: "",
        value: "",
        offsetWidth: 0,
        offsetHeight: 0,
        classList: {names: new Set(),
                    add(...names) { names.forEach((n) => node.classList.names.add(n)); },
                    remove(...names) { names.forEach((n) => node.classList.names.delete(n)); },
                    toggle(name, on) {
                        if (on) node.classList.add(name);
                        else node.classList.remove(name);
                    },
                    contains(name) { return node.classList.names.has(name); }},
        setAttribute(name, value) { node[name] = value; },
        getAttribute(name) { return node[name] === undefined ? null : node[name]; },
        removeAttribute(name) { delete node[name]; },
        hasAttribute(name) { return node[name] !== undefined; },
        appendChild(child) { node.children.push(child); return child; },
        removeChild(child) { node.children = node.children.filter((c) => c !== child); },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        addEventListener(type, fn) {
            (listeners[id + ":" + type] ||= []).push(fn);
            (node.handlers[type] ||= []).push(fn);
        },
        removeEventListener() {},
        focus() { node.focused = true; },
        click() { node.clicks = (node.clicks || 0) + 1; },
        contains() { return false; },
        getBoundingClientRect() { return {top: 0, left: 0, width: node.offsetWidth,
                                          height: node.offsetHeight}; },
    };
    elements[id] = node;
    return node;
}

globalThis.document = {
    body: element("body"),
    documentElement: {clientWidth: 1280, clientHeight: 800},
    readyState: "complete",
    hidden: false,
    activeElement: null,
    createElement: (tag) => element("made-" + Math.random().toString(16).slice(2),
                                   tag.toUpperCase()),
    getElementById: (id) => elements[id] || null,
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener() {},
    removeEventListener() {},
    contains() { return false; },
};

globalThis.window = globalThis;
globalThis.location = {pathname: "/", origin: "https://forge.example"};
const INSET_VALUES = __INSETS__;
globalThis.getComputedStyle = () => ({
    getPropertyValue: (name) => (INSET_VALUES[name] === undefined
        ? "0px" : INSET_VALUES[name] + "px"),
});
globalThis.visualViewport = __VIEWPORT__;
globalThis.requestAnimationFrame = (fn) => { fn(); return 1; };
globalThis.setTimeout = (fn) => { return 1; };
globalThis.clearTimeout = () => {};
globalThis.setInterval = () => 1;
globalThis.clearInterval = () => {};
globalThis.fetch = () => Promise.reject(new Error("no network in this harness"));
globalThis.sessionStorage = {
    store: {},
    getItem(key) { return this.store[key] === undefined ? null : this.store[key]; },
    setItem(key, value) { this.store[key] = String(value); },
};
// node 22 makes `crypto` a getter, so it is defined rather than assigned.
Object.defineProperty(globalThis, "crypto", {
    configurable: true,
    value: {getRandomValues(array) {
        for (let i = 0; i < array.length; i += 1) {
            array[i] = Math.floor(Math.random() * 256);
        }
        return array;
    }},
});
globalThis.MutationObserver = function () {
    return {observe() {}, disconnect() {}};
};
globalThis.ResizeObserver = function () {
    return {observe() {}, disconnect() {}};
};
globalThis.TextDecoder = function () { return {decode: (value) => String(value || "")}; };
globalThis.URL = {createObjectURL: () => "blob:x", revokeObjectURL: () => {}};
globalThis.console = console;

// The mount hook Forge supplies, collecting rather than running: a scenario
// that wants the panel built calls `loaded.forEach((fn) => fn())` itself, and
// one that only wants a pure function does not pay for a DOM it is not
// asserting anything about.
const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);

__SOURCE__

const NS = globalThis.forgeAssistant;

__SCENARIO__
"""


def run(scenario: str, viewport=None, insets=None, sources=("shell",)) -> dict:
    """One scenario, in a real JavaScript engine, against the real files.

    Written to a file rather than passed with ``node -e``: a single argument is
    capped at 128 KiB on Linux and the harness plus the sources is past that.
    """
    order = {"shell": SHELL, "host": HOST, "store": STORE}
    body = "\n".join(order[name].read_text(encoding="utf-8") for name in sources)
    # The scalars first and the sources last, deliberately. The sources contain
    # the word VIEWPORT (in ``NARROW_VIEWPORT``), so substituting them first
    # would rewrite a constant inside the file under test -- which fails as a
    # syntax error a long way from the line that caused it.
    harness = (HARNESS
               .replace("__VIEWPORT__", json.dumps(viewport if viewport is not None
                                                   else {"offsetLeft": 0, "offsetTop": 0,
                                                         "width": 1280, "height": 800}))
               .replace("__INSETS__", json.dumps(insets or {}))
               .replace("__SOURCE__", body)
               .replace("__SCENARIO__", scenario))
    with tempfile.TemporaryDirectory() as room:
        entry = pathlib.Path(room) / "scenario.mjs"
        entry.write_text(harness, encoding="utf-8")
        result = subprocess.run(["node", str(entry)], capture_output=True, text=True,
                                timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


BOX = "{width: 360, height: 480}"


class TestTheAnchorArithmetic:
    def test_all_six_anchors_land_inside_the_viewport(self):
        found = run(f"""
            const view = NS.viewport();
            const out = {{}};
            NS.ANCHORS.forEach((anchor) => {{
                const at = NS.anchorPoint(anchor, {BOX}, view, null);
                out[anchor] = at;
            }});
            console.log(JSON.stringify(out));
        """)

        for anchor, at in found.items():
            assert at["left"] >= 0, anchor
            assert at["top"] >= 0, anchor
            assert at["left"] + at["width"] <= 1280, anchor
            assert at["top"] + at["height"] <= 800, anchor

    def test_the_six_are_where_their_names_say(self):
        found = run(f"""
            const view = NS.viewport();
            const out = {{}};
            NS.ANCHORS.forEach((anchor) => {{
                out[anchor] = NS.anchorPoint(anchor, {BOX}, view, null);
            }});
            console.log(JSON.stringify(out));
        """)

        assert found["top-left"]["left"] == 24
        assert found["top-left"]["top"] == 24
        assert found["top-right"]["left"] == 1280 - 24 - 360
        assert found["bottom-left"]["top"] == 800 - 24 - 480
        assert found["top-center"]["left"] == (1280 - 360) / 2
        assert found["bottom-center"]["left"] == found["top-center"]["left"]
        assert found["bottom-right"]["top"] == found["bottom-left"]["top"]

    def test_a_narrow_viewport_uses_the_smaller_gap(self):
        found = run(f"""
            console.log(JSON.stringify(NS.anchorPoint("top-left", {BOX}, NS.viewport(), null)));
        """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 390, "height": 720})

        assert found["left"] == 16

    def test_the_safe_area_is_added_once_and_then_the_gap(self):
        """The mistake that puts a panel on an iPhone's home indicator is
        adding the gap first and the inset afterwards -- or adding neither."""
        found = run(f"""
            const view = NS.viewport();
            const inset = {{top: 47, right: 0, bottom: 34, left: 0}};
            console.log(JSON.stringify({{
                top: NS.anchorPoint("top-left", {BOX}, view, inset),
                bottom: NS.anchorPoint("bottom-left", {BOX}, view, inset),
            }}));
        """)

        assert found["top"]["top"] == 47 + 24
        assert found["bottom"]["top"] == 800 - 34 - 24 - found["bottom"]["height"]

    def test_the_visual_viewport_is_what_is_measured(self):
        """A phone with its keyboard up has a visual viewport shorter than the
        layout one. A panel positioned against the layout viewport ends up
        under the keyboard."""
        found = run(f"""
            console.log(JSON.stringify(NS.anchorPoint("bottom-left", {BOX}, NS.viewport(),
                                                      null)));
        """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 390, "height": 360})

        assert found["top"] + found["height"] <= 360

    def test_a_panel_taller_than_the_screen_is_clamped_rather_than_pushed_off(self):
        found = run("""
            console.log(JSON.stringify(NS.anchorPoint("bottom-right",
                {width: 360, height: 2000}, NS.viewport(), null)));
        """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 390, "height": 600})

        assert found["top"] >= 0
        assert found["top"] + found["height"] <= 600

    def test_an_offset_visual_viewport_is_respected(self):
        """Pinch-zoom moves the visual viewport's origin; a fixed element is
        positioned in that space, not at the document's origin."""
        found = run(f"""
            console.log(JSON.stringify(NS.anchorPoint("top-left", {BOX}, NS.viewport(),
                                                      null)));
        """, viewport={"offsetLeft": 120, "offsetTop": 60, "width": 800, "height": 500})

        assert found["left"] == 120 + 24
        assert found["top"] == 60 + 24

    def test_an_anchor_it_has_never_heard_of_falls_back_rather_than_breaking(self):
        found = run(f"""
            const view = NS.viewport();
            console.log(JSON.stringify({{
                unknown: NS.anchorPoint("middle-of-nowhere", {BOX}, view, null),
                bottomRight: NS.anchorPoint("bottom-right", {BOX}, view, null),
            }}));
        """)

        assert found["unknown"] == found["bottomRight"]


class TestNearestAnchor:
    def test_a_drag_to_a_corner_snaps_to_that_corner(self):
        found = run(f"""
            const view = NS.viewport();
            const out = {{}};
            out.topLeft = NS.nearestAnchor({{x: 40, y: 40}}, {BOX}, view, null, null);
            out.bottomRight = NS.nearestAnchor({{x: 1240, y: 760}}, {BOX}, view, null, null);
            out.topCenter = NS.nearestAnchor({{x: 640, y: 30}}, {BOX}, view, null, null);
            console.log(JSON.stringify(out));
        """)

        assert found["topLeft"] == "top-left"
        assert found["bottomRight"] == "bottom-right"
        assert found["topCenter"] == "top-center"

    def test_an_exact_tie_keeps_the_candidate_it_already_had(self):
        """Otherwise a drag held exactly between two anchors flickers between
        them under the finger."""
        found = run(f"""
            const view = NS.viewport();
            // The centre of the viewport is equidistant from the two centre
            // anchors' snapped rectangles.
            const centre = {{x: 640, y: 400}};
            console.log(JSON.stringify({{
                fromTop: NS.nearestAnchor(centre, {BOX}, view, null, "top-center"),
                fromBottom: NS.nearestAnchor(centre, {BOX}, view, null, "bottom-center"),
            }}));
        """)

        assert found["fromTop"] == "top-center"
        assert found["fromBottom"] == "bottom-center"

    def test_it_always_answers_with_one_of_the_six(self):
        found = run(f"""
            const view = NS.viewport();
            const seen = [];
            for (let x = -500; x < 2000; x += 137) {{
                for (let y = -500; y < 1500; y += 149) {{
                    seen.push(NS.nearestAnchor({{x, y}}, {BOX}, view, null, null));
                }}
            }}
            console.log(JSON.stringify({{seen: Array.from(new Set(seen))}}));
        """)

        assert set(found["seen"]) <= {"top-left", "top-center", "top-right",
                                      "bottom-left", "bottom-center", "bottom-right"}


class TestTheMarkdownRenderer:
    """Text a language model wrote, turned into HTML this page inserts.

    The one place in the feature where a mistake is a cross-site scripting hole
    rather than a layout bug, so the tests are mostly about what does *not*
    come out.
    """

    def test_html_in_a_reply_is_shown_rather_than_run(self):
        found = run("""
            console.log(JSON.stringify({
                out: NS.renderMarkdown("<script>alert(1)</script>"),
            }));
        """)

        assert "<script>" not in found["out"]
        assert "&lt;script&gt;" in found["out"]

    def test_an_image_tag_cannot_be_smuggled_in(self):
        found = run("""
            console.log(JSON.stringify({
                out: NS.renderMarkdown("<img src=x onerror=alert(1)>"),
            }));
        """)

        assert "<img" not in found["out"]
        assert "onerror" not in found["out"] or "&lt;img" in found["out"]

    @pytest.mark.parametrize("scheme", ["javascript:alert(1)", "data:text/html,<x>",
                                        "vbscript:x", "JaVaScRiPt:alert(1)",
                                        " javascript:alert(1)"])
    def test_a_link_with_an_unsafe_scheme_loses_its_href(self, scheme):
        found = run("""
            console.log(JSON.stringify({
                out: NS.renderMarkdown("[click me](" + %s + ")"),
                url: NS.safeUrl(%s),
            }));
        """ % (json.dumps(scheme), json.dumps(scheme)))

        assert found["url"] == ""
        assert "href" not in found["out"]
        assert "click me" in found["out"]

    @pytest.mark.parametrize("url", ["https://example.com/x",
                                     "http://example.com",
                                     "mailto:someone@example.com",
                                     "/model-chain/conversation/v2/attachment/abc"])
    def test_a_safe_link_survives(self, url):
        found = run("""
            console.log(JSON.stringify({out: NS.renderMarkdown("[go](" + %s + ")")}));
        """ % json.dumps(url))

        assert 'href="' in found["out"]
        assert 'rel="noopener noreferrer"' in found["out"], (
            "a link in a reply must not be able to reach back into this page")
        assert 'target="_blank"' in found["out"]

    def test_a_protocol_relative_url_is_not_treated_as_a_path(self):
        found = run("""
            console.log(JSON.stringify({url: NS.safeUrl("//evil.example/x")}));
        """)

        assert found["url"] == ""

    def test_a_fenced_block_is_not_formatted_inside(self):
        found = run("""
            console.log(JSON.stringify({
                out: NS.renderMarkdown("```\\nnot **bold** here\\n```"),
            }));
        """)

        assert "<strong>" not in found["out"]
        assert "<pre" in found["out"]

    def test_ordinary_formatting_still_works(self):
        found = run("""
            console.log(JSON.stringify({
                bold: NS.renderMarkdown("**loud**"),
                italic: NS.renderMarkdown("a *quiet* word"),
                code: NS.renderMarkdown("call `now()`"),
                paragraphs: NS.renderMarkdown("one\\n\\ntwo"),
                breaks: NS.renderMarkdown("one\\ntwo"),
            }));
        """)

        assert "<strong>loud</strong>" in found["bold"]
        assert "<em>quiet</em>" in found["italic"]
        assert "<code>now()</code>" in found["code"]
        assert found["paragraphs"].count("<p>") == 2
        assert "<br>" in found["breaks"]

    def test_an_ampersand_is_escaped_once(self):
        found = run("""
            console.log(JSON.stringify({out: NS.renderMarkdown("Tom & Jerry")}));
        """)

        assert "Tom &amp; Jerry" in found["out"]
        assert "&amp;amp;" not in found["out"]

    def test_nothing_at_all_renders_to_nothing_dangerous(self):
        found = run("""
            console.log(JSON.stringify({
                empty: NS.renderMarkdown(""),
                nothing: NS.renderMarkdown(null),
                undefinedValue: NS.renderMarkdown(undefined),
            }));
        """)

        assert "undefined" not in found["undefinedValue"]
        assert "null" not in found["nothing"]


class TestEscapePrecedence:
    """One key event, at most one action, in a fixed order.

    Escape means six things on this page. The wrong precedence is how Escape
    stops a reply somebody was reading instead of closing the menu in front of
    them, and it cannot be produced reliably by hand -- six of these cases need
    a dialog, an IME and a running job at once.
    """

    def order(self, context):
        return run("""
            console.log(JSON.stringify({
                action: NS.escapeOrder({}, %s),
            }));
        """ % json.dumps(context), sources=("host",))["action"]

    def test_an_ime_composition_is_never_ours(self):
        assert self.order({"composing": True, "assistantMenuOpen": True,
                           "focusActive": True}) == ""

    def test_a_native_dialog_wins(self):
        assert self.order({"nativeDialog": True, "assistantMenuOpen": True}) == ""

    def test_a_host_dialog_wins_unless_the_assistant_has_focus(self):
        assert self.order({"hostDialogOpen": True, "assistantMenuOpen": True,
                           "insideAssistant": False}) == ""
        assert self.order({"hostDialogOpen": True, "assistantMenuOpen": True,
                           "insideAssistant": True}) == "close-menu"

    def test_the_menu_closes_before_focus_exits(self):
        assert self.order({"assistantMenuOpen": True, "focusActive": True}) \
            == "close-menu"

    def test_an_edit_is_cancelled_before_focus_exits(self):
        assert self.order({"assistantEditing": True, "focusActive": True}) \
            == "cancel-edit"

    def test_focus_exits_when_nothing_else_wants_it(self):
        assert self.order({"focusActive": True}) == "exit-focus"

    def test_otherwise_the_host_keeps_its_escape(self):
        """Interrupt is the host's. An assistant that swallowed Escape would
        stop being able to stop an image generation."""
        assert self.order({}) == ""
        assert self.order({"insideAssistant": True}) == ""


class TestTheSubmitShortcut:
    def submits(self, event, context):
        return run("""
            console.log(JSON.stringify({
                submits: NS.submitShortcut(%s, %s),
            }));
        """ % (json.dumps(event), json.dumps(context)), sources=("host",))["submits"]

    def test_ctrl_enter_in_a_composer_submits(self):
        assert self.submits({"key": "Enter", "ctrlKey": True},
                            {"focusedComposer": True}) is True

    def test_cmd_enter_in_a_composer_submits(self):
        assert self.submits({"key": "Enter", "metaKey": True},
                            {"focusedComposer": True}) is True

    def test_ctrl_enter_outside_a_composer_does_nothing(self):
        """G08: never the image workspaces' Generate, which is what a
        document-wide binding would eventually reach."""
        assert self.submits({"key": "Enter", "ctrlKey": True},
                            {"focusedComposer": False}) is False

    def test_a_plain_enter_is_not_this_shortcut(self):
        assert self.submits({"key": "Enter"}, {"focusedComposer": True}) is False

    def test_an_ime_composition_never_submits(self):
        assert self.submits({"key": "Enter", "ctrlKey": True},
                            {"focusedComposer": True, "composing": True}) is False


class TestTheStylesheetCanHideThings:
    """One missing rule, three bugs.

    `hidden` is how every piece of this panel is shown and hidden, and the
    browser implements it in its *user-agent* stylesheet. Any author
    declaration of `display` beats a user-agent one outright, whatever the
    specificity -- so `.forge-assistant-panel { display: flex }` made the
    panel, the menu and the conversation body unhideable.

    From the outside that was: ✕ did nothing, the workspace menu stayed open
    after a workspace was chosen, and the collapsed launcher sat beside a panel
    that was supposed to be closed. None of it is visible in the JavaScript,
    which is why it is asserted against the stylesheet.
    """

    def stylesheet(self):
        return (pathlib.Path(__file__).resolve().parent.parent
                / "style.css").read_text(encoding="utf-8")

    def test_the_root_carries_a_hidden_rule(self):
        css = self.stylesheet()

        assert "#forge-assistant-root [hidden]" in css
        block = css.split("#forge-assistant-root [hidden]", 1)[1].split("}", 1)[0]
        assert "display: none" in block
        assert "!important" in block, (
            "a plain declaration ties with the display rules below it and loses "
            "on source order")

    def test_every_part_that_is_hidden_from_script_is_inside_that_root(self):
        """The rule is scoped to the assistant's root, so anything the script
        hides has to be a descendant of it or the rule does not reach it."""
        shell = SHELL.read_text(encoding="utf-8")
        hidden = set(re.findall(r"nodes\.(\w+)\.hidden", shell))

        assert hidden, "nothing is hidden from script any more; this test is stale"
        assert hidden <= {"panel", "launcher", "menu", "body", "chip", "jump", "stop",
                          "send", "unread", "suppressed", "selector", "transcript",
                          "composer", "status", "filePicker", "launcherIcon",
                          "launcherLabel", "input", "ghost"}, hidden


class TestOpeningAndClosing:
    def build(self, scenario):
        return run("""
            const shell = new NS.Shell(NEW_STORE, NEW_HOST, NEW_FOCUS);
            shell.mount();
            %s
        """ % scenario, sources=("shell",))

    def test_exactly_one_of_the_launcher_and_the_panel_is_ever_drawn(self):
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            shell.state = {panelOpen: false, conversationExpanded: true};
            shell.nodes = {
                panel: {hidden: false},
                launcher: {hidden: false, setAttribute(n, v) { this[n] = v; },
                           focus() { this.focused = true; }},
            };
            shell._save = () => {};
            shell.placeNow = () => {};
            shell.closeMenu = () => { shell.menuClosed = true; };
            const seen = [];
            shell.showOpen(true);
            seen.push([shell.nodes.panel.hidden, shell.nodes.launcher.hidden]);
            shell.showOpen(false);
            seen.push([shell.nodes.panel.hidden, shell.nodes.launcher.hidden]);
            console.log(JSON.stringify({seen, expanded: shell.nodes.launcher["aria-expanded"],
                                        menuClosed: !!shell.menuClosed}));
        """, sources=("shell",))

        assert found["seen"] == [[False, True], [True, False]]
        assert found["expanded"] == "false"

    def test_closing_the_panel_also_closes_its_menu(self):
        """A menu left open on a panel that is not drawn is a menu that comes
        back with it."""
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            shell.state = {panelOpen: true};
            shell.nodes = {panel: {hidden: false},
                           launcher: {hidden: true, setAttribute() {}, focus() {}}};
            shell._save = () => {};
            shell.placeNow = () => {};
            shell.closeMenu = () => { shell.menuClosed = true; };
            shell.close();
            console.log(JSON.stringify({menuClosed: !!shell.menuClosed,
                                        open: shell.state.panelOpen}));
        """, sources=("shell",))

        assert found["menuClosed"] is True
        assert found["open"] is False

    def test_the_header_carries_no_pin(self):
        """It was a preference nobody asked for, in the corner everybody aims
        at."""
        shell = SHELL.read_text(encoding="utf-8")

        assert "pinned" not in shell
        assert "nodes.pin" not in shell


class TestTheUtilityMenu:
    def test_it_offers_exactly_the_two_actions(self):
        """It used to walk the header and offer whatever it found, which on a
        real installation is "Apply settings", "Reload UI" and a column of
        controls whose only visible text is the word JSON."""
        found = run("""
            const host = NS.host();
            console.log(JSON.stringify({
                items: host.listUtilities().map((u) => [u.label, u.kind, u.scope]),
            }));
        """, sources=("host",))

        assert found["items"] == [["Unload All Models", "unload", "all"],
                                  ["Unload LLM", "unload", "llm"]]

    def test_it_never_reads_the_host_s_header(self):
        host = HOST.read_text(encoding="utf-8")
        utilities = host.split("Host.prototype.listUtilities = function", 1)[1] \
            .split("Host.prototype", 1)[0]

        assert "querySelector" not in utilities
        assert "quicksettings" not in utilities
        assert "settings_submit" not in utilities


PICKER = """
// A shell with just enough around it to press a menu item: the picker calls
// into the host, into focus mode and into the status line, and a test that
// stubbed only the host would pass or fail on whichever of the three it
// happened to reach first.
function picker(host, focus) {
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {panelOpen: true, focusEnabled: false, focusWorkspaceId: null};
    shell.closed = 0;
    shell.said = [];
    shell.closeMenu = () => { shell.closed += 1; };
    shell.say = (text, tone) => shell.said.push({text, tone});
    shell._save = () => {};
    shell.place = () => {};
    shell.nodes = {focusToggle: {setAttribute(name, value) { this[name] = value; }}};
    shell.host = Object.assign({
        getActiveWorkspace: () => "tab_txt2img",
        listWorkspaces: () => [{id: "tab_img2img", label: "Img2Img", available: true}],
        activateWorkspace: () => Promise.resolve("tab_img2img"),
    }, host || {});
    shell.focus = Object.assign({
        calls: [],
        on: "",
        isActive() { return !!this.on; },
        activeWorkspace() { return this.on; },
        exit() { this.calls.push("exit:" + this.on); this.on = ""; return true; },
        enter(id) {
            this.calls.push("enter:" + id);
            this.on = id;
            return {ok: true, reason: "", note: ""};
        },
    }, focus || {});
    return shell;
}
"""


class TestTheWorkspacePicker:
    def test_the_menu_closes_on_the_press_not_on_the_confirmation(self):
        """Held open until the host confirmed, it stayed open for ever whenever
        the confirmation did not arrive -- and it did not, because the watchdog
        rejects after four seconds on any page whose tab bar this adapter reads
        differently from the way it expected to."""
        found = run(PICKER + """
            const shell = picker({activateWorkspace: () => new Promise(() => {})});
            shell.workspaceItems()[0].handlers.click.forEach((fn) => fn());
            console.log(JSON.stringify({closed: shell.closed}));
        """, sources=("shell",))

        assert found["closed"] == 1


class TestTheStore:
    def test_the_base_path_follows_a_reverse_proxy(self):
        found = run("""
            console.log(JSON.stringify({route: NS.route("/snapshot")}));
        """, sources=("store",))

        assert found["route"].endswith("/model-chain/conversation/v2/snapshot")

    def test_a_conversation_key_cannot_be_confused_by_a_name(self):
        """Two threads whose character and id happen to concatenate the same
        way must not share a draft."""
        found = run("""
            console.log(JSON.stringify({
                same: NS.conversationKey("a", "bc") === NS.conversationKey("ab", "c"),
            }));
        """, sources=("store",))

        assert found["same"] is False

    def test_an_event_from_another_epoch_is_refused(self):
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "this-process";
            const applied = store.apply({protocol_version: 2, server_epoch: "another",
                                         stream_cursor: 1, kind: "conversation_changed"});
            console.log(JSON.stringify({applied, error: store.error,
                                        connected: store.connected}));
        """, sources=("store",))

        assert found["applied"] is False
        assert "restarted" in found["error"]
        assert found["connected"] is False

    def test_a_duplicate_or_older_cursor_cannot_regress_the_view(self):
        """T03. Reordered, duplicated and old events are the ordinary weather
        of a reconnect."""
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.cursor = 5;
            const older = store.apply({protocol_version: 2, server_epoch: "e",
                                       stream_cursor: 4, kind: "conversation_changed"});
            const same = store.apply({protocol_version: 2, server_epoch: "e",
                                      stream_cursor: 5, kind: "conversation_changed"});
            console.log(JSON.stringify({older, same, cursor: store.cursor}));
        """, sources=("store",))

        assert found["older"] is False
        assert found["same"] is False
        assert found["cursor"] == 5

    def test_a_reply_patch_with_an_obsolete_sequence_is_dropped(self):
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.apply({protocol_version: 2, server_epoch: "e", stream_cursor: 1,
                         kind: "reply_patch", operation_id: "op", operation_seq: 5,
                         conversation: {character: "Ada", thread_id: "t"},
                         payload: {text: "later"}});
            store.apply({protocol_version: 2, server_epoch: "e", stream_cursor: 2,
                         kind: "reply_patch", operation_id: "op", operation_seq: 2,
                         conversation: {character: "Ada", thread_id: "t"},
                         payload: {text: "earlier"}});
            console.log(JSON.stringify({text: store.operations.get("op").text}));
        """, sources=("store",))

        assert found["text"] == "later"

    def test_an_event_for_another_thread_never_touches_this_one(self):
        """I07. A badge, and nothing else."""
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "mine", epoch: "x"};
            store.drafts.set(NS.conversationKey("Ada", "mine"),
                             {text: "half a message", draftVersion: 1,
                              pendingTranscripts: [], attachment: null, editBuffer: null});
            store.apply({protocol_version: 2, server_epoch: "e", stream_cursor: 1,
                         kind: "reply_patch", operation_id: "op", operation_seq: 1,
                         conversation: {character: "Ada", thread_id: "theirs"},
                         payload: {text: "somebody else's reply"}});
            const view = store.snapshot();
            console.log(JSON.stringify({draft: view.draft.text,
                                        operation: view.operation}));
        """, sources=("store",))

        assert found["draft"] == "half a message"
        assert found["operation"] is None

    def test_a_draft_is_never_evicted_by_the_cache(self):
        """A cache may forget a conversation. It may not forget what somebody
        typed into one."""
        found = run("""
            const store = new NS.Store();
            store.drafts.set(NS.conversationKey("Ada", "t0"), {text: "kept"});
            for (let index = 0; index < 30; index += 1) {
                store.remember(NS.conversationKey("Ada", "t" + index), {messages: []});
            }
            console.log(JSON.stringify({
                kept: store.snapshots.has(NS.conversationKey("Ada", "t0")),
                text: store.drafts.get(NS.conversationKey("Ada", "t0")).text,
            }));
        """, sources=("store",))

        assert found["kept"] is True
        assert found["text"] == "kept"

    def test_storage_that_throws_leaves_the_store_working(self):
        found = run("""
            globalThis.sessionStorage = {
                getItem() { throw new Error("blocked"); },
                setItem() { throw new Error("blocked"); },
            };
            const store = new NS.Store();
            store.setDraftText("still typed");
            store.flushDrafts();
            console.log(JSON.stringify({
                text: store.draft().text,
                available: store.storageAvailable,
            }));
        """, sources=("store",))

        assert found["text"] == "still typed"
        assert found["available"] is False

    def test_a_send_latches_one_operation_id(self):
        """9.3. A duplicate click, a duplicate Enter and a retry after a lost
        response are all the same request -- minting a second id is exactly how
        one message becomes two."""
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "t", epoch: "x"};
            store.setDraftText("hello");
            const seen = [];
            store.send = (envelope) => {
                seen.push(envelope.operation_id);
                return new Promise(() => {});   // never settles: still in flight
            };
            store.submit();
            store.submit();
            store.submit();
            console.log(JSON.stringify({seen}));
        """, sources=("store",))

        assert len(found["seen"]) == 1

    def test_a_retry_reuses_the_very_same_envelope(self):
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "t", epoch: "x"};
            store.setDraftText("hello");
            const seen = [];
            store.send = (envelope) => {
                seen.push(envelope.operation_id);
                return Promise.resolve({ok: false, lost: true, envelope});
            };
            store.submit().then(() => store.retryPending()).then(() => {
                console.log(JSON.stringify({seen, same: seen[0] === seen[1]}));
            });
        """, sources=("store",))

        assert found["same"] is True

    def test_newer_typing_is_not_cleared_by_an_older_acknowledgement(self):
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "t", epoch: "x"};
            store.setDraftText("first");
            store.refresh = () => Promise.resolve(null);
            let settle = null;
            store.send = () => new Promise((resolve) => { settle = resolve; });
            const sending = store.submit();
            store.setDraftText("typed while it was in flight");
            settle({ok: true, operation_id: "op", phase: "completed",
                    resulting_conversation: {character: "Ada", thread_id: "t"}});
            sending.then(() => {
                console.log(JSON.stringify({text: store.draft().text}));
            });
        """, sources=("store",))

        assert found["text"] == "typed while it was in flight"


class TestWhichConversationIsOnScreen:
    """Reported in use: the panel came up on "Reconnecting… Your draft is safe."
    with an empty transcript and never drew anything.

    Nothing was broken in the transport. The panel simply never learned *which*
    conversation to show: nothing called `select()`, so the selection stayed
    empty, `refresh()` returned early because there was nothing to ask about,
    and the transcript was never going to be given anything to draw. The fix is
    two halves -- a seed at startup, and following the tab afterwards -- and
    both halves are here.
    """

    def test_the_bootstrap_seeds_which_conversation_to_show(self):
        found = run("""
            globalThis.fetch = (url) => Promise.resolve({
                ok: true,
                json: () => Promise.resolve(String(url).indexOf("/bootstrap") >= 0
                    ? {server_epoch: "e", page_id: "p", characters: ["Ada"],
                       selection: {character: "Ada", thread_id: "t-42"}}
                    : {}),
            });
            const store = new NS.Store();
            store.connect = () => Promise.resolve();
            store.start().then(() => {
                console.log(JSON.stringify({
                    character: store.selection.character,
                    thread: store.selection.thread,
                    characters: store.characters,
                }));
            });
        """, sources=("store",))

        assert found["character"] == "Ada"
        assert found["thread"] == "t-42"
        assert found["characters"] == ["Ada"]

    def test_a_page_that_already_chose_keeps_its_choice(self):
        """A seed, not a source of truth. The preference it comes from is
        installation-wide, so a second window that has deliberately been put on
        another thread must not be dragged back to the first one."""
        found = run("""
            globalThis.fetch = () => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({server_epoch: "e",
                    selection: {character: "Ada", thread_id: "t-42"}}),
            });
            const store = new NS.Store();
            store.selection = {character: "Bob", thread: "t-9", epoch: "x"};
            store.connect = () => Promise.resolve();
            store.start().then(() => {
                console.log(JSON.stringify({character: store.selection.character,
                                            thread: store.selection.thread}));
            });
        """, sources=("store",))

        assert found["character"] == "Bob"
        assert found["thread"] == "t-9"

    def test_the_panel_follows_the_tab_to_another_conversation(self):
        """Being a second window onto the same work is the whole point. A panel
        that stayed on the thread it was seeded with would show a different
        conversation from the one behind it, with nothing on screen to say so.
        """
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "t-1", epoch: "x"};
            const asked = [];
            store.refresh = () => { asked.push(store.selection.thread);
                                    return Promise.resolve(null); };
            store.apply({protocol_version: 2, server_epoch: "e", stream_cursor: 1,
                         kind: "character_changed",
                         payload: {character: "Bob", thread_id: "t-2"}});
            console.log(JSON.stringify({character: store.selection.character,
                                        thread: store.selection.thread, asked}));
        """, sources=("store",))

        assert found["character"] == "Bob"
        assert found["thread"] == "t-2"
        assert found["asked"] == ["t-2"], "the new conversation has to be fetched"

    def test_following_the_tab_onto_the_thread_already_shown_changes_nothing(self):
        """The tab re-announces its selection on every rebuild. Re-selecting
        would mint a new epoch, clear the unread badge and re-fetch, all for a
        conversation already on screen."""
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "t-1", epoch: "keep-me"};
            let asked = 0;
            store.refresh = () => { asked += 1; return Promise.resolve(null); };
            store.follow({character: "Ada", thread_id: "t-1"});
            console.log(JSON.stringify({epoch: store.selection.epoch, asked}));
        """, sources=("store",))

        assert found["epoch"] == "keep-me"
        assert found["asked"] == 0

    def test_an_announcement_with_no_thread_is_not_a_selection(self):
        """Choosing a character before opening one of its threads announces an
        empty thread id. Following that would clear a conversation somebody was
        reading."""
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "t-1", epoch: "x"};
            store.follow({character: "Bob", thread_id: ""});
            console.log(JSON.stringify({character: store.selection.character,
                                        thread: store.selection.thread}));
        """, sources=("store",))

        assert found["character"] == "Ada"
        assert found["thread"] == "t-1"

    def test_a_feed_that_never_opened_is_connecting_rather_than_reconnecting(self):
        """"Reconnecting" is only true the second time. It was the first thing
        on screen when the panel came up with nothing in it -- a sentence
        pointing at the wrong problem."""
        found = run("""
            const said = [];
            const shell = Object.create(NS.Shell.prototype);
            shell.say = (text, tone) => said.push({text, tone});
            shell.renderStatus({ready: true, connected: false, everConnected: false,
                                selection: {thread: ""}, characters: []});
            shell.renderStatus({ready: true, connected: false, everConnected: true,
                                selection: {thread: "t"}, characters: ["Ada"]});
            console.log(JSON.stringify({said}));
        """)

        assert found["said"][0]["text"] == "Connecting…"
        assert found["said"][0]["tone"] == "info"
        assert "Reconnecting" in found["said"][1]["text"]

    def test_connected_with_nothing_chosen_says_so(self):
        """An empty transcript under the word "Ready" reads like a conversation
        that lost its messages rather than one that was never picked."""
        found = run("""
            const said = [];
            const shell = Object.create(NS.Shell.prototype);
            shell.say = (text) => said.push(text);
            shell.renderStatus({ready: true, connected: true, everConnected: true,
                                selection: {thread: ""}, characters: ["Ada"]});
            shell.renderStatus({ready: true, connected: true, everConnected: true,
                                selection: {thread: ""}, characters: []});
            shell.renderStatus({ready: true, connected: true, everConnected: true,
                                selection: {thread: "t"}, characters: ["Ada"]});
            console.log(JSON.stringify({said}));
        """)

        assert found["said"][0] == "Pick a conversation to begin."
        assert found["said"][1] == "No conversations yet. Start one in LLM Studio."
        assert found["said"][2] == "Ready."


TRANSCRIPT = """
// A transcript that can be scrolled and measured. The scroll arithmetic is the
// whole of what these tests are about, so a stub that merely stored the number
// it was handed would agree with any implementation -- including one that
// scrolls past the end.
function fakeTranscript(rows) {
    const node = {
        children: [],
        clientHeight: 100,
        rowHeight: 40,
        get scrollHeight() { return node.children.length * node.rowHeight; },
        set innerHTML(value) { if (!value) node.children.length = 0; },
        get innerHTML() { return ""; },
        appendChild(child) { node.children.push(child); return child; },
        querySelector() { return node.children.length ? node.children[0] : null; },
    };
    let top = 0;
    Object.defineProperty(node, "scrollTop", {
        get() { return top; },
        set(value) {
            top = Math.max(0, Math.min(value, node.scrollHeight - node.clientHeight));
        },
    });
    return node;
}

function bottom(node) {
    return Math.max(0, node.scrollHeight - node.clientHeight);
}

function shellWith(transcript) {
    const shell = Object.create(NS.Shell.prototype);
    shell.nodes = {transcript, jump: {hidden: true, textContent: ""}};
    shell.settings = {bubbleWidth: 80};
    shell.following = true;
    shell.shownEpoch = "";
    shell.lastRendered = "";
    shell.settled = false;
    shell.settling = false;
    shell.bubble = (row) => ({
        dataset: {index: String(row.index)},
        getBoundingClientRect: () => ({top: -transcript.scrollTop}),
    });
    shell.updateBubble = () => undefined;
    return shell;
}

function conversationOf(count, epoch) {
    const messages = [];
    for (let index = 0; index < count; index += 1) {
        messages.push({index, role: index % 2 ? "assistant" : "user",
                       text: "m" + index, active: 0, versions: ["m" + index]});
    }
    return {conversation: {messages}, selection: {epoch: epoch || "e1"},
            unread: 0, operation: null};
}
"""


class TestTheTranscriptScroll:
    """Reported in use: the thread showed, but not at the end of itself.

    Three behaviours, and they are one rule seen from three places: a
    conversation opens at its latest message; a reader who is at the end stays
    at the end as replies arrive; a reader who has scrolled away is not moved.
    """

    def test_a_conversation_opens_at_its_latest_message(self):
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(20));
            console.log(JSON.stringify({top: transcript.scrollTop,
                                        end: bottom(transcript)}));
        """)

        assert found["top"] == found["end"]
        assert found["end"] > 0, "the fixture has to be long enough to scroll"

    def test_a_reader_at_the_end_is_kept_there_by_a_new_message(self):
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(20));
            shell.renderTranscript(conversationOf(21));
            console.log(JSON.stringify({top: transcript.scrollTop,
                                        end: bottom(transcript),
                                        jump: shell.nodes.jump.hidden}));
        """)

        assert found["top"] == found["end"]
        assert found["jump"] is True

    def test_a_reader_who_scrolled_away_is_not_moved_by_a_new_message(self):
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(20));
            transcript.scrollTop = 120;
            shell.following = false;           // what the scroll listener sets
            const before = transcript.scrollTop;
            shell.renderTranscript(conversationOf(21));
            console.log(JSON.stringify({before, after: transcript.scrollTop,
                                        jump: shell.nodes.jump.hidden}));
        """)

        assert found["after"] == found["before"]
        assert found["jump"] is False, "there has to be a way back to the latest"

    def test_another_conversation_opens_at_its_own_latest_message(self):
        """A thread left scrolled halfway up must not put the next thread
        halfway up -- at an offset measured against a message that is not even
        on the page any more."""
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(20, "e1"));
            transcript.scrollTop = 40;
            shell.following = false;
            shell.renderTranscript(conversationOf(30, "e2"));
            console.log(JSON.stringify({top: transcript.scrollTop,
                                        end: bottom(transcript),
                                        following: shell.following}));
        """)

        assert found["following"] is True
        assert found["top"] == found["end"]

    def test_two_threads_that_read_the_same_are_still_two_threads(self):
        """The redraw is skipped when the content fingerprint is unchanged.
        Without the conversation in that fingerprint, switching between two
        threads whose messages happen to match leaves the first one's DOM up --
        with its bubble actions aimed at the wrong file."""
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            let built = 0;
            const make = shell.bubble;
            shell.bubble = (row) => { built += 1; return make(row); };
            shell.renderTranscript(conversationOf(4, "e1"));
            const first = built;
            shell.renderTranscript(conversationOf(4, "e2"));
            console.log(JSON.stringify({first, second: built - first}));
        """)

        assert found["second"] == found["first"], "the second thread is drawn too"

    def test_a_bubble_is_not_reused_across_a_revision(self):
        """`updateBubble` refreshes the text of a reused node and nothing else.
        The action buttons under it closed over the row and the revision they
        were *built* with, and they send those -- so a bubble reused after the
        thread moved sends a revision the server will refuse, on every action,
        for as long as the node survives.

        A reply arriving token by token does not move the revision, so the
        per-token redraw this keying exists to avoid is still avoided.
        """
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            let built = 0;
            const make = shell.bubble;
            shell.bubble = (row) => { built += 1; return make(row); };
            const at = (revision, text) => ({
                conversation: {conversation: {revision},
                               messages: [{index: 0, role: "user", text,
                                           active: 0, versions: [text]}]},
                selection: {epoch: "e1"}, unread: 0, operation: null});
            shell.renderTranscript(at(4, "first"));
            const start = built;
            shell.renderTranscript(at(4, "edited in place"));
            const sameRevision = built - start;
            shell.renderTranscript(at(5, "edited in place"));
            console.log(JSON.stringify({start, sameRevision,
                                        moved: built - start - sameRevision}));
        """)

        assert found["start"] == 1
        assert found["sameRevision"] == 0, "the same revision reuses the node"
        assert found["moved"] == 1, "a moved thread rebuilds it"

    def test_a_redraw_that_changed_nothing_still_holds_the_bottom(self):
        """A panel that has just opened -- or whose conversation section was
        collapsed -- has a transcript of no height at the moment the messages
        go into it, so scrolling to the end scrolls nothing. The box arrives
        afterwards, and the end has to be found again when it does."""
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            transcript.clientHeight = 0;
            transcript.rowHeight = 0;          // nothing has a size yet
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(20));
            const whileFlat = transcript.scrollTop;
            transcript.rowHeight = 40;         // the panel got its box
            transcript.clientHeight = 100;
            shell.renderTranscript(conversationOf(20));
            console.log(JSON.stringify({whileFlat, after: transcript.scrollTop,
                                        end: bottom(transcript)}));
        """)

        assert found["whileFlat"] == 0
        assert found["after"] == found["end"]
        assert found["end"] > 0

    def test_the_jump_button_only_claims_a_new_response_when_there_is_one(self):
        """It sits there for as long as somebody is scrolled up. A button
        labelled "New response" the whole time is a button nobody believes the
        second time it is right."""
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(20));
            shell.following = false;
            const quiet = Object.assign(conversationOf(21), {unread: 0});
            shell.renderTranscript(quiet);
            const idle = shell.nodes.jump.textContent;
            const loud = Object.assign(conversationOf(22), {unread: 2});
            shell.renderTranscript(loud);
            console.log(JSON.stringify({idle, loud: shell.nodes.jump.textContent}));
        """)

        assert found["idle"] == "Jump to latest"
        assert found["loud"] == "New response"

    def test_the_end_is_found_again_once_the_panel_has_a_size(self):
        """The synchronous scroll covers the ordinary case, where replacing the
        messages forces the layout that answers "how tall is this". It does not
        cover a box that arrives *later* -- a panel opening, a conversation
        section expanding -- and a transcript that scrolled to the end of
        nothing is a transcript at the top."""
        found = run(TRANSCRIPT + """
            let pending = null;
            globalThis.requestAnimationFrame = (fn) => { pending = fn; return 1; };
            const transcript = fakeTranscript();
            transcript.clientHeight = 0;
            transcript.rowHeight = 0;
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(20));
            const flat = transcript.scrollTop;
            transcript.rowHeight = 40;         // the box arrives
            transcript.clientHeight = 100;
            pending();                         // ... and the frame runs
            console.log(JSON.stringify({flat, after: transcript.scrollTop,
                                        end: bottom(transcript)}));
        """)

        assert found["flat"] == 0
        assert found["after"] == found["end"]
        assert found["end"] > 0

    def test_a_later_frame_never_drags_a_reader_who_has_scrolled_away(self):
        """The other half of it. A frame queued while following, running after
        somebody has scrolled up, must not take them back down."""
        found = run(TRANSCRIPT + """
            let pending = null;
            globalThis.requestAnimationFrame = (fn) => { pending = fn; return 1; };
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(20));
            transcript.scrollTop = 40;
            shell.following = false;           // what the scroll listener sets
            pending();
            console.log(JSON.stringify({top: transcript.scrollTop}));
        """)

        assert found["top"] == 40

    def test_an_empty_conversation_leaves_nothing_behind(self):
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(20));
            transcript.scrollTop = 40;
            shell.following = false;
            shell.renderTranscript(conversationOf(21));   // the jump appears
            const shown = shell.nodes.jump.hidden;
            shell.renderTranscript({conversation: null, selection: {epoch: "e2"},
                                    unread: 0, operation: null});
            console.log(JSON.stringify({shown, rows: transcript.children.length,
                                        jump: shell.nodes.jump.hidden}));
        """)

        assert found["shown"] is False
        assert found["rows"] == 0
        assert found["jump"] is True, (
            "a button offering to jump to the latest of nothing is a dead control")

    def test_a_conversation_that_comes_back_is_drawn_again(self):
        """A snapshot can arrive without its conversation -- a refresh in
        flight, a thread being read. What must not happen is the transcript
        staying empty when it returns, because the redraw was skipped on the
        grounds that the messages had not changed since the last time they
        were drawn."""
        found = run(TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            shell.renderTranscript(conversationOf(6, "e1"));
            shell.renderTranscript({conversation: null, selection: {epoch: "e1"},
                                    unread: 0, operation: null});
            const gone = transcript.children.length;
            shell.renderTranscript(conversationOf(6, "e1"));
            console.log(JSON.stringify({gone, back: transcript.children.length}));
        """)

        assert found["gone"] == 0
        assert found["back"] == 6


HOST_TREE = r"""
// A DOM with the shape Gradio actually renders, because the stub above has no
// shape at all and that is how four rounds of focus-mode reports got past
// this file. `gr.Tabs` is
//
//     div#tabs.tabs
//       div.tab-nav[role=tablist] > button[role=tab][aria-controls=tab_x] ...
//       div#tab_x.tabitem[role=tabpanel][style="display: block|none"]
//
// and every NESTED gr.Tabs -- Forge's extra networks inside Txt2Img, the mode
// tabs inside Img2Img -- is rendered with the same classes and roles. Enough of
// a selector engine to ask the questions the adapter asks: a selector list,
// `:scope >`, tag, #id, .class, [attr], [attr=v], [attr^=v], [attr$=v].

function el(tag, attrs, parent) {
    attrs = attrs || {};
    const node = {
        tagName: tag.toUpperCase(),
        id: attrs.id || "",
        attributes: {},
        children: [],
        parentElement: parent || null,
        style: {display: attrs.display || ""},
        computedPosition: attrs.position || "static",
        textContent: attrs.text || "",
        disabled: !!attrs.disabled,
        clicks: 0,
        classList: {
            names: new Set((attrs.className || "").split(" ").filter(Boolean)),
            add(...names) { names.forEach((n) => node.classList.names.add(n)); },
            remove(...names) { names.forEach((n) => node.classList.names.delete(n)); },
            contains(name) { return node.classList.names.has(name); },
        },
        getAttribute(name) {
            if (name === "id") return node.id || null;
            return name in node.attributes ? node.attributes[name] : null;
        },
        hasAttribute(name) { return name === "id" ? !!node.id : name in node.attributes; },
        setAttribute(name, value) {
            if (name === "id") node.id = String(value);
            else node.attributes[name] = String(value);
        },
        contains(other) {
            let walk = other;
            while (walk) {
                if (walk === node) return true;
                walk = walk.parentElement;
            }
            return false;
        },
        querySelector(selector) { return query(node, selector)[0] || null; },
        querySelectorAll(selector) { return query(node, selector); },
        click() { node.clicks += 1; },
        get offsetParent() {
            if (effectiveDisplay(node) === "none") return null;
            if (node.computedPosition === "fixed") return null;
            return node.parentElement;
        },
    };
    Object.keys(attrs).forEach((key) => {
        if (["id", "className", "display", "position", "text", "disabled"].indexOf(key) < 0) {
            node.attributes[key] = String(attrs[key]);
        }
    });
    if (parent) parent.children.push(node);
    return node;
}

function effectiveDisplay(node) {
    let walk = node;
    while (walk) {
        if (walk.style && walk.style.display === "none") return "none";
        walk = walk.parentElement;
    }
    return (node.style && node.style.display) || "block";
}

function descendants(node) {
    const out = [];
    (function walk(parent) {
        parent.children.forEach((child) => {
            out.push(child);
            walk(child);
        });
    })(node);
    return out;
}

function parseCompound(text) {
    const out = {tag: null, id: null, classes: [], attrs: []};
    const re = /([a-zA-Z][\w-]*|\*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:([\^$*]?=)(?:"([^"]*)"|'([^']*)'|([^\]]*)))?\]/g;
    let m;
    while ((m = re.exec(text))) {
        if (m[1]) out.tag = m[1] === "*" ? null : m[1].toUpperCase();
        else if (m[2]) out.id = m[2];
        else if (m[3]) out.classes.push(m[3]);
        else if (m[4]) {
            const value = m[6] !== undefined ? m[6] : (m[7] !== undefined ? m[7] : m[8]);
            out.attrs.push({name: m[4], op: m[5] || null,
                            value: value === undefined ? null : value});
        }
    }
    return out;
}

function matchesCompound(node, c) {
    if (c.tag && node.tagName !== c.tag) return false;
    if (c.id && node.id !== c.id) return false;
    if (!c.classes.every((name) => node.classList.contains(name))) return false;
    return c.attrs.every((a) => {
        const v = node.getAttribute(a.name);
        if (v === null) return false;
        if (!a.op) return true;
        if (a.op === "=") return v === a.value;
        if (a.op === "^=") return v.indexOf(a.value) === 0;
        if (a.op === "$=") return v.slice(-a.value.length) === a.value;
        if (a.op === "*=") return v.indexOf(a.value) >= 0;
        return false;
    });
}

function query(root, selector) {
    const found = [];
    String(selector).split(",").map((s) => s.trim()).filter(Boolean).forEach((s) => {
        let pool;
        let compound;
        if (s.indexOf(":scope >") === 0) {
            pool = root.children;
            compound = parseCompound(s.slice(8).trim());
        } else {
            pool = descendants(root);
            compound = parseCompound(s);
        }
        pool.forEach((node) => {
            if (matchesCompound(node, compound) && found.indexOf(node) < 0) found.push(node);
        });
    });
    const order = descendants(root);
    found.sort((a, b) => order.indexOf(a) - order.indexOf(b));
    return found;
}

function forgeTree(options) {
    options = options || {};
    const body = el("body", {id: "body"});
    const gradioApp = el("gradio-app", {}, body);
    const container = el("div", {className: "gradio-container app"}, gradioApp);
    const contain = el("div", {className: "contain"},
                       el("div", {className: "wrap"}, el("div", {className: "main"}, container)));
    el("div", {id: "quicksettings"}, contain);
    const tabs = el("div", {id: "tabs", className: "tabs"}, contain);
    const nav = el("div", {className: "tab-nav scroll-hide", role: "tablist"}, tabs);
    const button = (name, id, selected) => el("button", {
        id: id + "-button", role: "tab", "aria-controls": id,
        "aria-selected": selected ? "true" : "false",
        className: selected ? "selected" : "", text: name,
    }, nav);
    button("txt2img", "tab_txt2img", true);
    button("img2img", "tab_img2img", false);
    button("LLM Studio", "tab_llm_studio", false);

    const txt2img = el("div", {id: "tab_txt2img", className: "tabitem", role: "tabpanel",
                              display: "block"}, tabs);
    // Forge's extra networks: a nested gr.Tabs holding the whole of the
    // generation UI, so this panel has a tab bar inside it on every install.
    const extra = el("div", {id: "txt2img_extra_tabs", className: "tabs"}, txt2img);
    const extraNav = el("div", {className: "tab-nav", role: "tablist"}, extra);
    el("button", {role: "tab", "aria-controls": "txt2img_generation", text: "Generation",
                  className: "selected"}, extraNav);
    el("button", {role: "tab", "aria-controls": "txt2img_lora", text: "Lora"}, extraNav);
    el("div", {id: "txt2img_generation", className: "tabitem", role: "tabpanel",
               display: "block"}, extra);
    el("div", {id: "txt2img_lora", className: "tabitem", role: "tabpanel",
               display: "none"}, extra);

    const img2img = el("div", {id: "tab_img2img", className: "tabitem", role: "tabpanel",
                              display: "none"}, tabs);
    const modes = el("div", {id: "mode_img2img", className: "tabs"}, img2img);
    const modeNav = el("div", {className: "tab-nav", role: "tablist"}, modes);
    el("button", {role: "tab", text: "img2img", className: "selected"}, modeNav);
    el("button", {role: "tab", text: "Inpaint"}, modeNav);

    const studio = el("div", {id: "tab_llm_studio", className: "tabitem", role: "tabpanel",
                             display: "none"}, tabs);
    el("div", {id: "footer"}, contain);
    const assistant = el("div", {id: "forge-assistant-root"}, body);
    const tree = {body, gradioApp, container, contain, tabs, nav, txt2img, img2img, studio,
                  extraNav, modeNav, assistant};
    if (options.lobe) lobify(tree);
    globalThis.gradioApp = () => tree.gradioApp;
    globalThis.getComputedStyle = (node) => (node && node.tagName ? {
        display: effectiveDisplay(node), visibility: "visible",
        position: node.computedPosition, getPropertyValue() { return "0px"; },
    } : {getPropertyValue() { return "0px"; }});
    return tree;
}

// What sd-webui-lobe-theme does at load, read from its source: a React root
// appended to <gradio-app>; Gradio's `.app` container moved inside that root's
// <main>; the original bar hidden with an inline style; and a header of its
// own -- position: sticky, z-index 999 -- carrying a tablist that is NOT
// under #tabs and whose entries are not <button>s with aria-controls.
function lobify(tree) {
    const root = el("div", {id: "root"}, tree.gradioApp);
    const header = el("header", {id: "lobe-header", position: "sticky"}, root);
    const lobeNav = el("div", {id: "lobe-nav", role: "tablist", className: "ant-tabs-nav"},
                       header);
    ["Txt 2 Img", "Img 2 Img", "LLM Studio"].forEach((label) => {
        el("div", {role: "tab", text: label}, lobeNav);
    });
    el("button", {text: "settings"}, header);
    const main = el("main", {id: "lobe-main"}, root);
    el("div", {id: "lobe-background", className: "background", position: "absolute"}, main);
    const asideLeft = el("aside", {id: "lobe-aside-left", position: "sticky"}, main);
    const content = el("div", {id: "lobe-content", className: "content"}, main);
    const asideRight = el("aside", {id: "lobe-aside-right", position: "sticky"}, main);
    const footer = el("footer", {id: "lobe-footer"}, root);
    tree.gradioApp.children = tree.gradioApp.children.filter((c) => c !== tree.container);
    content.children.push(tree.container);
    tree.container.parentElement = content;
    tree.nav.style.display = "none";
    Object.assign(tree, {root, header, lobeNav, main, asideLeft, content, asideRight, footer});
}
"""


class TestTheHostReadsForgesTabs:
    """The adapter between this panel and the page, against the page's real
    shape -- which is the thing the stub DOM above does not have, and the
    reason every round of "focus mode does not work" got past this file.

    The version these replace collected every tab button on the page as a
    workspace (nested tab groups have the same class and the same role), threw
    away every panel with a nested tab group inside it (which is most of them,
    Txt2Img included), and paired what was left of the two lists by position.
    Txt2Img came out as a hidden panel or as nothing, and focus mode did what
    it was told with that.
    """

    def test_a_workspace_is_a_top_level_tab_and_only_that(self):
        found = run(HOST_TREE + """
            forgeTree();
            const host = NS.host();
            console.log(JSON.stringify({
                ids: host.listWorkspaces().map((w) => w.id),
                labels: host.listWorkspaces().map((w) => w.label),
            }));
        """, sources=("host",))

        assert found["ids"] == ["tab_txt2img", "tab_img2img", "tab_llm_studio"]
        assert found["labels"] == ["txt2img", "img2img", "LLM Studio"]
        assert "Generation" not in found["labels"], "a nested tab is not a workspace"

    def test_a_panel_holding_nested_tabs_is_still_a_workspace(self):
        """Txt2Img has a tab bar inside it on every Forge there is: the extra
        networks are a nested gr.Tabs around the whole generation UI. Excluding
        a panel for that made the one workspace everybody focuses first the one
        that could not be."""
        found = run(HOST_TREE + """
            forgeTree();
            const host = NS.host();
            const root = host.resolveWorkspaceRoot("tab_txt2img");
            console.log(JSON.stringify({root: root ? root.id : null,
                                        panels: host.panels().map((p) => p.id)}));
        """, sources=("host",))

        assert found["root"] == "tab_txt2img"
        assert found["panels"] == ["tab_txt2img", "tab_img2img", "tab_llm_studio"]

    def test_the_active_workspace_is_the_panel_on_screen(self):
        found = run(HOST_TREE + """
            const tree = forgeTree();
            const host = NS.host();
            const first = host.getActiveWorkspace();
            tree.txt2img.style.display = "none";
            tree.img2img.style.display = "block";
            console.log(JSON.stringify({first, second: host.getActiveWorkspace()}));
        """, sources=("host",))

        assert found["first"] == "tab_txt2img"
        assert found["second"] == "tab_img2img"

    def test_a_focused_workspace_is_still_the_active_one(self):
        """Focus makes the panel `position: fixed`, and a fixed element has no
        offsetParent -- the shortcut for "not laid out". Read from the inline
        style, where the class says nothing, the one workspace on screen was
        reported as the one that was not, and the next thing asked for was a
        workspace switch to where the person already was."""
        found = run(HOST_TREE + """
            const tree = forgeTree();
            tree.txt2img.computedPosition = "fixed";
            // A theme that does not keep the button's selected state in step:
            // only the panel can answer.
            tree.nav.children.forEach((b) => {
                b.classList.remove("selected");
                b.attributes["aria-selected"] = "false";
            });
            console.log(JSON.stringify({active: NS.host().getActiveWorkspace()}));
        """, sources=("host",))

        assert found["active"] == "tab_txt2img"

    def test_buttons_are_paired_with_panels_by_aria_controls_not_by_position(self):
        """Gradio renders the panel of a tab that is not offered -- `visible`
        false -- and leaves out its button. Paired by position, every workspace
        after it is one panel off, and "focus Txt2Img" focuses whatever comes
        next."""
        found = run(HOST_TREE + """
            const tree = forgeTree();
            // A panel with no button, ahead of the others in the DOM.
            const ghost = el("div", {id: "tab_hidden_extension", className: "tabitem",
                                     role: "tabpanel", display: "none"});
            tree.tabs.children.splice(1, 0, ghost);
            ghost.parentElement = tree.tabs;
            // Only aria-controls can answer: the buttons carry no id of their own.
            tree.nav.children.forEach((b) => { b.id = ""; });
            const host = NS.host();
            console.log(JSON.stringify({
                pairs: host.listWorkspaces().map((w) => [w.button.getAttribute("aria-controls"),
                                                         w.panel && w.panel.id]),
            }));
        """, sources=("host",))

        for controls, panel in found["pairs"]:
            assert controls == panel, found["pairs"]

    def test_a_button_named_after_its_panel_pairs_without_aria_controls(self):
        """Gradio also names the button `<panel id>-button`. Second in line,
        and enough on its own for the same ghost panel."""
        found = run(HOST_TREE + """
            const tree = forgeTree();
            const ghost = el("div", {id: "tab_hidden_extension", className: "tabitem",
                                     role: "tabpanel", display: "none"});
            tree.tabs.children.splice(1, 0, ghost);
            ghost.parentElement = tree.tabs;
            tree.nav.children.forEach((b) => { delete b.attributes["aria-controls"]; });
            console.log(JSON.stringify({
                ids: NS.host().listWorkspaces().map((w) => w.id),
            }));
        """, sources=("host",))

        assert found["ids"] == ["tab_txt2img", "tab_img2img", "tab_llm_studio"]

    def test_the_bar_is_the_shallowest_one_not_the_first_one(self):
        """A theme may move the strip -- below the panels, into a wrapper --
        and a nested bar can then come first in document order. The top-level
        bar is never inside a panel, so it is always the shallowest; the first
        one found is whichever the theme happened to put on top."""
        found = run(HOST_TREE + """
            const tree = forgeTree();
            tree.tabs.children = tree.tabs.children.filter((c) => c !== tree.nav);
            tree.tabs.children.push(tree.nav);          // now after every panel
            console.log(JSON.stringify({
                labels: NS.host().listWorkspaces().map((w) => w.label),
            }));
        """, sources=("host",))

        assert found["labels"] == ["txt2img", "img2img", "LLM Studio"]

    def test_a_host_writing_no_aria_controls_still_gets_a_positional_pairing(self):
        found = run(HOST_TREE + """
            const tree = forgeTree();
            tree.nav.children.forEach((b) => { delete b.attributes["aria-controls"]; b.id = ""; });
            console.log(JSON.stringify({
                ids: NS.host().listWorkspaces().map((w) => w.id),
            }));
        """, sources=("host",))

        assert found["ids"] == ["tab_txt2img", "tab_img2img", "tab_llm_studio"]

    def test_a_box_wrapped_around_the_bar_is_not_a_workspace(self):
        """The one exclusion that is right, kept and made precise. A candidate
        that has the WORKSPACE bar inside it, focused, is the tab bar laid over
        the window. A candidate with a nested bar inside it is a workspace."""
        found = run(HOST_TREE + """
            const tree = forgeTree();
            const wrapper = el("div", {id: "tab_wrapper"});
            tree.tabs.children = tree.tabs.children.filter((c) => c !== tree.nav);
            tree.tabs.children.unshift(wrapper);
            wrapper.parentElement = tree.tabs;
            wrapper.children.push(tree.nav);
            tree.nav.parentElement = wrapper;
            const host = NS.host();
            console.log(JSON.stringify({
                panels: host.panels().map((p) => p.id),
                ids: host.listWorkspaces().map((w) => w.id),
            }));
        """, sources=("host",))

        assert "tab_wrapper" not in found["panels"]
        assert found["ids"] == ["tab_txt2img", "tab_img2img", "tab_llm_studio"]

    def test_under_the_lobe_theme_the_workspaces_are_still_forge_s(self):
        """sd-webui-lobe-theme, read from its source: it appends a React root
        to <gradio-app>, moves Gradio's `.app` container inside it, hides the
        original bar with an inline style, and draws a header of its own with
        a tablist that is not under #tabs. None of that is a workspace, and
        the hidden bar is still the one that says which tabs there are."""
        found = run(HOST_TREE + """
            forgeTree({lobe: true});
            const host = NS.host();
            console.log(JSON.stringify({
                ids: host.listWorkspaces().map((w) => w.id),
                active: host.getActiveWorkspace(),
                root: (host.resolveWorkspaceRoot("tab_txt2img") || {}).id,
            }));
        """, sources=("host",))

        assert found["ids"] == ["tab_txt2img", "tab_img2img", "tab_llm_studio"]
        assert found["active"] == "tab_txt2img"
        assert found["root"] == "tab_txt2img"

    def test_activating_a_workspace_presses_forge_s_own_button(self):
        """Lobe's own nav does exactly this -- finds the hidden Gradio button
        and clicks it -- so a hidden button is a button that works."""
        found = run(HOST_TREE + """
            const tree = forgeTree({lobe: true});
            const host = NS.host();
            host.activateWorkspace("tab_img2img").catch(() => undefined);
            const pressed = tree.nav.children.map((b) => b.clicks);
            console.log(JSON.stringify({pressed}));
        """, sources=("host",))

        assert found["pressed"] == [0, 1, 0]


class TestEveryMenuHasAWayOut:
    """Reported in use: once a menu was open there was no easy exit unless you
    chose something from it.

    Escape closes them, and so does opening the other one, and so would a press
    on the button that opened it. None of that is any use to a thumb: a phone
    has no Escape key, and the two header buttons are a small target beside a
    menu that is covering them.
    """

    def test_every_menu_ends_with_cancel(self):
        found = run(PICKER + """
            const shell = picker();
            shell.nodes.menu = {hidden: true, dataset: {}, items: [],
                                set innerHTML(v) { if (!v) this.items.length = 0; },
                                get innerHTML() { return ""; },
                                appendChild(node) { this.items.push(node); }};
            shell.nodes.picker = {setAttribute() {}};
            shell.nodes.utilities = {setAttribute() {}};
            shell.closeMenu = () => {};
            shell.host.openOnly = () => {};
            shell.host.listUtilities = () => [{id: "u", label: "Unload LLM",
                                               enabled: true, kind: "unload", scope: "llm"}];
            shell.store = {snapshot: () => ({conversation: {threads: []},
                                             selection: {character: "", thread: ""}})};
            const last = {};
            ["workspaces", "threads", "utilities"].forEach((which) => {
                shell.nodes.menu.dataset.which = "";
                shell.toggleMenu(which);
                const items = shell.nodes.menu.items;
                last[which] = items[items.length - 1].textContent;
            });
            console.log(JSON.stringify(last));
        """, sources=("shell",))

        assert found == {"workspaces": "Cancel", "threads": "Cancel",
                         "utilities": "Cancel"}

    def test_cancel_closes_the_menu_and_chooses_nothing(self):
        found = run(PICKER + """
            const shell = picker();
            let activated = 0;
            shell.host.activateWorkspace = () => { activated += 1;
                                                   return Promise.resolve(""); };
            const cancel = shell.cancelItem();
            cancel.handlers.click.forEach((fn) => fn());
            console.log(JSON.stringify({closed: shell.closed, activated,
                                        role: cancel.getAttribute("role")}));
        """, sources=("shell",))

        assert found["closed"] == 1
        assert found["activated"] == 0
        assert found["role"] == "menuitem"

    def test_the_way_out_is_added_where_a_new_menu_cannot_forget_it(self):
        """In `toggleMenu`, not in each builder. A fourth menu added later gets
        one without anybody remembering to."""
        shell = SHELL.read_text(encoding="utf-8")
        builders = [shell.split("Shell.prototype." + name + " = function", 1)[1]
                    .split("Shell.prototype", 1)[0]
                    for name in ("workspaceItems", "threadItems", "utilityItems")]

        for body in builders:
            assert "cancelItem" not in body, (
                "the way out belongs in toggleMenu, once, not in every builder")
        toggle = shell.split("Shell.prototype.toggleMenu = function", 1)[1] \
            .split("Shell.prototype", 1)[0]
        assert "cancelItem" in toggle


class TestSwitchingWorkspaceWhileFocused:
    """Reported in use: the picker would not switch tabs until focus mode was
    turned off.

    Focus hides every panel but the one it is filling, with `display: none
    !important` -- which beats the inline `display: block` Gradio writes on the
    panel it has just switched to. So the destination stayed hidden, the old
    workspace stayed fixed to the viewport, and `getActiveWorkspace()` went on
    naming the focused one. The switch could not be observed, so the
    confirmation watchdog timed out and said the workspace had not opened.
    """

    def test_focus_comes_off_for_the_switch_and_goes_on_at_the_destination(self):
        found = run(PICKER + """
            const shell = picker();
            shell.focus.on = "tab_txt2img";
            shell.state.focusEnabled = true;
            shell.switchWorkspace("tab_img2img").then(() => {
                console.log(JSON.stringify({
                    calls: shell.focus.calls,
                    on: shell.focus.activeWorkspace(),
                    remembered: shell.state.focusWorkspaceId,
                    pressed: shell.nodes.focusToggle["aria-pressed"],
                }));
            });
        """, sources=("shell",))

        assert found["calls"] == ["exit:tab_txt2img", "enter:tab_img2img"], (
            "off before the host is asked to show a panel this code is hiding")
        assert found["on"] == "tab_img2img"
        assert found["remembered"] == "tab_img2img"
        assert found["pressed"] == "true"

    def test_the_host_is_asked_while_nothing_is_being_hidden(self):
        """The order is the fix. Asked first and unfocused second, the host
        would be switching to a panel that is still `display: none`."""
        found = run(PICKER + """
            const order = [];
            const shell = picker({activateWorkspace: (id) => {
                order.push("activate:" + id + " focus=" + (shell.focus.on || "none"));
                return Promise.resolve(id);
            }});
            shell.focus.on = "tab_txt2img";
            shell.state.focusEnabled = true;
            shell.switchWorkspace("tab_img2img").then(() => {
                console.log(JSON.stringify({order}));
            });
        """, sources=("shell",))

        assert found["order"] == ["activate:tab_img2img focus=none"]

    def test_a_switch_that_fails_does_not_also_cost_focus_mode(self):
        found = run(PICKER + """
            const shell = picker({
                activateWorkspace: () => Promise.reject(new Error("did not open")),
            });
            shell.focus.on = "tab_txt2img";
            shell.state.focusEnabled = true;
            shell.switchWorkspace("tab_img2img").then(() => {
                console.log(JSON.stringify({on: shell.focus.activeWorkspace(),
                                            enabled: shell.state.focusEnabled,
                                            said: shell.said.map((s) => s.text)}));
            });
        """, sources=("shell",))

        assert found["on"] == "tab_txt2img", "put back where it was"
        assert found["enabled"] is True
        assert found["said"] == ["did not open"]

    def test_a_switch_with_focus_off_never_touches_focus(self):
        found = run(PICKER + """
            const shell = picker();
            shell.switchWorkspace("tab_img2img").then(() => {
                console.log(JSON.stringify({calls: shell.focus.calls,
                                            enabled: shell.state.focusEnabled}));
            });
        """, sources=("shell",))

        assert found["calls"] == []
        assert found["enabled"] is False

    def test_a_destination_that_cannot_be_focused_says_so_and_stays_switched(self):
        """The switch happened; only the mode could not follow. Reporting it as
        a failed switch would be a lie about where the person now is."""
        found = run(PICKER + """
            const shell = picker({}, {
                enter(id) { this.calls.push("enter:" + id);
                            return {ok: false, reason: "No panel to fill with."}; },
            });
            shell.focus.on = "tab_txt2img";
            shell.state.focusEnabled = true;
            shell.switchWorkspace("tab_img2img").then(() => {
                console.log(JSON.stringify({enabled: shell.state.focusEnabled,
                                            pressed: shell.nodes.focusToggle["aria-pressed"],
                                            said: shell.said.map((s) => s.text)}));
            });
        """, sources=("shell",))

        assert found["enabled"] is False
        assert found["pressed"] == "false"
        assert found["said"] == ["No panel to fill with."]

    def test_navigation_never_tries_to_move_focus_that_is_off(self):
        """Between the exit and the re-entry above, focus is off. `moveTo`
        refuses with "Focus is not on.", which would turn a working switch into
        a warning and a lost mode."""
        shell = SHELL.read_text(encoding="utf-8")
        subscriber = shell.split("subscribeNavigation((active)", 1)[1] \
            .split("}));", 1)[0]

        assert "this.focus.isActive()" in subscriber, (
            "only focus that is on can be moved")
        assert "moveTo" in subscriber


class TestThePerMessageActions:
    """Reported in use: eight buttons and a version pager under every bubble,
    wrapping onto three rows at panel width.

    What is left is the three somebody wants on the thing they just said or
    just read. Nothing here removes an action from the conversation -- the tab
    keeps the whole set, and it has the room to explain what branching and
    truncating are about to do.
    """

    @staticmethod
    def bar(row, count, role="assistant"):
        return PICKER + """
            const shell = picker();
            shell.settings = {bubbleWidth: 80};
            const messages = [];
            for (let i = 0; i < %d; i += 1) {
                messages.push({index: i, role: "user", text: "m" + i, active: 0,
                               versions: ["m" + i]});
            }
            const row = {index: %d, role: "%s", text: "hello", active: 0,
                         versions: ["hello"]};
            messages[%d] = row;
            const view = {conversation: {conversation: {revision: 3}, messages}};
            const bar = shell.actions(row, view);
            console.log(JSON.stringify({
                glyphs: bar.children.map((b) => b.textContent),
                labels: bar.children.map((b) => b.getAttribute("aria-label")),
                titles: bar.children.map((b) => b.title),
            }));
        """ % (count, row, role, row)

    def test_only_the_last_message_carries_actions(self):
        found = run(self.bar(2, 5), sources=("shell",))

        assert found["glyphs"] == []

    def test_the_last_reply_offers_edit_regenerate_and_delete(self):
        found = run(self.bar(4, 5, "assistant"), sources=("shell",))

        assert found["labels"] == ["Edit", "Regenerate", "Delete"]

    def test_the_last_message_of_yours_has_nothing_to_ask_again(self):
        """Regenerate is a reply's action. On an unanswered message of yours
        there is no reply to replace."""
        found = run(self.bar(4, 5, "user"), sources=("shell",))

        assert found["labels"] == ["Edit", "Delete"]

    def test_each_one_is_a_glyph_with_the_word_kept_for_the_reader(self):
        """An icon with no accessible name is a button only sighted people
        have, and it is the thing that goes wrong when labels come off."""
        found = run(self.bar(4, 5, "assistant"), sources=("shell",))

        for glyph, label, title in zip(found["glyphs"], found["labels"],
                                       found["titles"]):
            assert len(glyph) == 1, f"{label} is still a word: {glyph!r}"
            assert glyph.isascii() is False
            assert label and title == label

    def test_regenerate_is_the_glyph_the_tab_already_draws(self):
        """The same action in two views of one conversation. Two glyphs for it
        would be two things to learn."""
        shell = SHELL.read_text(encoding="utf-8")
        studio = (JAVASCRIPT / "llm_studio.js").read_text(encoding="utf-8")

        assert '"\\u21bb"' in shell
        assert '"\\u21bb"' in studio

    def test_the_actions_nobody_asked_for_are_gone_from_this_view(self):
        """Branching, truncating and version paging are the heavy ones, and
        they are the ones that want a screen to explain themselves on."""
        shell = SHELL.read_text(encoding="utf-8")
        bar = shell.split("Shell.prototype.actions = function", 1)[1] \
            .split("Shell.prototype", 1)[0]

        for gone in ("branch", "delete_from", "resend_from_user", "select_version",
                     "drop_version", "continue", "listen"):
            assert gone not in bar, gone

    def test_nothing_is_left_dispatching_an_action_no_button_sends(self):
        """`act` special-cased Listen, a truncation confirm and the version
        pager. With the buttons gone those branches are unreachable, and dead
        code that looks live is how the next person concludes the panel still
        does all of it."""
        shell = SHELL.read_text(encoding="utf-8")
        act = shell.split("Shell.prototype.act = function", 1)[1] \
            .split("Shell.prototype", 1)[0]

        for gone in ("listen", "delete_from", "select_version", "confirm_count"):
            assert gone not in act, gone
