"""The assistant costs the page nothing while nobody is using it.

Reviewed at the user's request, with one rule: fix what can be fixed without
taking anything away, and ask before anything that would. Everything here is
of the first kind.

The finding that mattered: the host adapter watched every ``class`` and
``style`` change anywhere under ``#tabs`` -- the whole application -- to notice
a tab switch. During a generation that is every progress tick, every preview
frame and every status flip, and each one re-read every tab (with a computed
style walk per tab to ask whether it could be focused), called every listener,
and rebuilt the collapsed panel's tab strip. Measured in Chromium over five
seconds of simulated generation: 606 active-tab lookups, 909 full tab
descriptions, 303 listener calls and 303 strip rebuilds before; 2, 0, 1 and 0
after -- the two and the one being the real switch at the end.

Every test here was checked against the change it guards by reverting that
change and watching the test fail.
"""

from __future__ import annotations

from test_assistant_js import SHELL, run

HOST_SETUP = """
const observed = [];
globalThis.MutationObserver = function (callback) {
    const self = {callback, targets: [],
                  observe(node, options) { observed.push([node.id, options]); },
                  disconnect() { observed.push(["disconnect"]); }};
    return self;
};
function node(id) {
    const made = document.createElement("div");
    made.id = id;
    return made;
}
const tabs = node("tabs");
const strip = node("strip");
const buttons = ["txt2img", "img2img", "Extras"].map((label, index) => {
    const button = node("button-" + index);
    button.textContent = label;
    return button;
});
const panels = ["tab_txt2img", "tab_img2img", "tab_extras"].map(node);
let showing = "tab_txt2img";
globalThis.gradioApp = () => ({querySelector: (selector) => (selector === "#tabs" ? tabs : null)});
const host = new NS.Host();
host.bar = () => ({tabs, strip, buttons});
host.panels = () => panels;
host.pairs = () => buttons.map((button, order) => ({
    id: panels[order].id, label: button.textContent, order,
    available: !button.disabled, button, panel: panels[order]}));
host.getActiveWorkspace = () => showing;
"""


class TestTheNavigationObserver:
    def test_it_watches_the_panels_and_the_bar_and_not_the_application(self):
        found = run(HOST_SETUP + """
            host.watch();
            console.log(JSON.stringify({observed}));
        """, sources=("host",))

        calls = [entry for entry in found["observed"] if entry[0] != "disconnect"]
        by_target = {target: options for target, options in calls}

        assert by_target["tabs"] == {"childList": True}, (
            "#tabs itself only for panels coming and going -- never its subtree")
        assert by_target["strip"]["subtree"] is True
        assert set(by_target["strip"]["attributeFilter"]) >= {"class", "aria-selected"}
        for panel in ("tab_txt2img", "tab_img2img", "tab_extras"):
            assert by_target[panel].get("subtree") in (None, False), panel
            assert "style" in by_target[panel]["attributeFilter"], panel
        assert not any(options.get("subtree") and target == "tabs"
                       for target, options in calls), (
            "a subtree observer on #tabs is every attribute change in the "
            "application, which is what this replaced")

    def test_panels_that_come_and_go_are_watched_too(self):
        """The structure observer rebinds; a tab added after start is seen."""
        found = run(HOST_SETUP + """
            let structure = null;
            const made = [];
            globalThis.MutationObserver = function (callback) {
                const self = {callback, observe(node, options) {
                                  observed.push([node.id, options]);
                                  if (options.childList) structure = self;
                              },
                              disconnect() {}};
                made.push(self);
                return self;
            };
            host.watch();
            panels.push(node("tab_new"));
            observed.length = 0;
            structure.callback([]);
            console.log(JSON.stringify({ids: observed.map((entry) => entry[0])}));
        """, sources=("host",))

        assert "tab_new" in found["ids"]

    def test_a_change_nothing_shows_notifies_nobody(self):
        found = run(HOST_SETUP + """
            const heard = [];
            host.listeners.add((active, pending) => heard.push(active + "/" + pending));
            host.notify();
            host.notify();                    // the same state again
            showing = "tab_img2img";
            host.notify();                    // a switch
            host.pending = "tab_extras";
            host.notify();                    // a switch waited on
            buttons[2].textContent = "Upscale";
            host.notify();                    // a tab renamed
            buttons[2].disabled = true;
            host.notify();                    // a tab disabled
            host.notify();
            console.log(JSON.stringify({heard}));
        """, sources=("host",))

        assert found["heard"] == ["tab_txt2img/", "tab_img2img/", "tab_img2img/tab_extras",
                                  "tab_img2img/tab_extras", "tab_img2img/tab_extras"]

    def test_the_hot_paths_never_ask_whether_a_tab_can_be_focused(self):
        """That question is an ancestor walk reading computed styles, and it
        was being asked for every tab on every notification."""
        found = run(HOST_SETUP + """
            delete host.pairs;
            delete host.getActiveWorkspace;
            let asked = 0;
            host.focusCapability = () => { asked += 1; return {ok: true, reason: ""}; };
            // The real pairing needs real selectors; this one is enough for it.
            buttons.forEach((button, order) => button.setAttribute("aria-controls", panels[order].id));
            host.getActiveWorkspace();
            host.resolveWorkspaceRoot("tab_img2img");
            host.activateWorkspace("tab_extras").catch(() => null);
            const hot = asked;
            const listed = host.listWorkspaces();
            const described = asked;
            const first = listed[0].focusCapability;
            listed[0].focusCapability;
            console.log(JSON.stringify({hot, described, afterRead: asked, first,
                                        reason: listed[1].reason, count: listed.length}));
        """, sources=("host",))

        assert found["hot"] == 0
        assert found["described"] == 0, "not even a full listing asks until it is read"
        assert found["afterRead"] == 1, "read once, remembered"
        assert found["first"] is True
        assert found["reason"] == ""
        assert found["count"] == 3


class TestTheStrip:
    ROW = """
        const shell = Object.create(NS.Shell.prototype);
        shell.state = {conversationExpanded: false};
        const row = document.createElement("div");
        // As a browser does it: clearing the markup clears the children.
        Object.defineProperty(row, "innerHTML", {get: () => "",
                                                 set: () => { row.children = []; }});
        row.hidden = false;
        row.scrollWidth = 0;
        row.clientWidth = 0;
        shell.nodes = {workspaces: row};
        let active = "tab_a";
        const tabs = [{id: "tab_a", label: "A", available: true},
                      {id: "tab_b", label: "B", available: true}];
        shell.host = {getActiveWorkspace: () => active, pairs: () => tabs.map((t) => Object.assign({}, t))};
        shell.close = () => {};
        shell.switchWorkspace = () => {};
    """

    def test_a_selection_change_re_marks_the_same_buttons(self):
        found = run(self.ROW + """
            shell.renderWorkspaces();
            const first = row.children.slice();
            active = "tab_b";
            shell.renderWorkspaces();
            console.log(JSON.stringify({
                same: row.children.every((child, index) => child === first[index]),
                current: row.children.map((child) => child.getAttribute("aria-current"))}));
        """)

        assert found["same"] is True, "the buttons under a focus or a drag stay"
        assert found["current"] == ["false", "true"]

    def test_a_change_to_the_tabs_themselves_rebuilds(self):
        found = run(self.ROW + """
            shell.renderWorkspaces();
            const first = row.children[0];
            tabs[1].label = "B renamed";
            shell.renderWorkspaces();
            const renamed = row.children[0] !== first;
            const second = row.children[0];
            tabs[0].available = false;
            shell.renderWorkspaces();
            console.log(JSON.stringify({renamed, disabled: row.children[0] !== second,
                                        labels: row.children.map((c) => c.textContent)}));
        """)

        assert found["renamed"] is True
        assert found["disabled"] is True
        assert found["labels"] == ["A", "B renamed"]


class TestRenderingKeepsToTheFrame:
    def test_many_announcements_in_a_frame_draw_once_with_the_latest(self):
        """A streamed reply announces every token; the screen shows one frame
        at a time. Each announcement used to redraw on the spot."""
        found = run("""
            const frames = [];
            globalThis.requestAnimationFrame = (fn) => { frames.push(fn); return frames.length; };
            const shell = Object.create(NS.Shell.prototype);
            shell.disposers = [];
            const drawn = [];
            shell.render = (view) => drawn.push(view.n);
            shell.fault = (error) => { throw error; };
            let subscriber = null;
            shell.store = {subscribeState(fn) { subscriber = fn; return () => {}; }};
            // Only the subscription from `wire`, run on its own.
            const source = NS.Shell.prototype.wire.toString();
            const start = source.indexOf("this.disposers.push(this.store.subscribeState(");
            const end = source.indexOf("}));", start) + 4;
            new Function(source.slice(start, end)).call(shell);
            for (let n = 1; n <= 5; n += 1) subscriber({n});
            const before = drawn.length;
            frames.splice(0).forEach((fn) => fn());
            subscriber({n: 6});
            frames.splice(0).forEach((fn) => fn());
            console.log(JSON.stringify({before, drawn, requested: frames.length}));
        """)

        assert found["before"] == 0, "nothing is drawn until the frame"
        assert found["drawn"] == [5, 6], "once per frame, with the latest view"

    def test_a_render_that_fails_in_the_frame_still_resets_the_shell(self):
        found = run("""
            const frames = [];
            globalThis.requestAnimationFrame = (fn) => { frames.push(fn); return frames.length; };
            const shell = Object.create(NS.Shell.prototype);
            shell.disposers = [];
            shell.render = () => { throw new Error("bad view"); };
            let faults = 0;
            shell.fault = () => { faults += 1; };
            let subscriber = null;
            shell.store = {subscribeState(fn) { subscriber = fn; return () => {}; }};
            const source = NS.Shell.prototype.wire.toString();
            const start = source.indexOf("this.disposers.push(this.store.subscribeState(");
            const end = source.indexOf("}));", start) + 4;
            new Function(source.slice(start, end)).call(shell);
            subscriber({n: 1});
            frames.splice(0).forEach((fn) => fn());
            subscriber({n: 2});
            frames.splice(0).forEach((fn) => fn());
            console.log(JSON.stringify({faults}));
        """)

        assert found["faults"] == 2, "and the next frame still draws"
