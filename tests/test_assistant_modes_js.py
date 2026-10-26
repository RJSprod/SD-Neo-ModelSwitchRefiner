"""Normal, Focused, Super Focused: two presses move to the next mode.

Asked for from use (docs/22 §3.39): "a new mode that is just the video gallery
from wangp (currently playing and strip) with playback controls, and the prompt
box with a generation button, and progress bar ... two columns that can be
resized, with prompting on left, and gallery on right ... a wan2gp focused mode.
... if I double tap the flyout button action activation / drag handle, it
cycles through Normal Mode / Focused Mode / Super Focused Wan2gp Mode. Super
Focused mode would just be those elements, and when i return out of it,
everything and all settings return (except updated prompt and gallery)."

WanGP's page is Mini Paint NEO's frame, and its bridge (1.15.0) lays it out as
the studio when the focused panel carries one more class
(`forge-assistant-wangp-studio`, the contract). These hold the panel's side
under node: what two presses move to from where, that super focus is the WanGP
tab's and is offered only by a bridge that can lay the page out, that the panel
and the tab bar are put away and what was put away comes back however super
focus ends, and that the old double press -- back to the previous workspace --
is the workspace menu's first entry. The class on a real panel is held in
test_assistant_focus_js.py; the press in a real browser in
test_assistant_layout.py.
"""

from __future__ import annotations

from test_assistant_js import run


WORLD = """
function fakeFocus(active) {
    return {
        active: active || "", studio: false, bar: "shown", barMode: "shown", exits: 0,
        isActive() { return !!this.active; },
        activeWorkspace() { return this.active; },
        enter(id) { this.active = id; this.studio = false; return {ok: true, reason: "", note: ""}; },
        exit() { if (!this.active) return false; this.active = ""; this.studio = false; this.exits += 1; return true; },
        moveTo(id) { this.exit(); return this.enter(id); },
        setStudio(on) { if (!this.active) return false; this.studio = !!on; return true; },
        setBar(mode) { this.bar = mode; this.barMode = mode; return mode; },
        keptBar() { return this.active ? {} : null; },
    };
}

function fakeWanGP(layout, ready) {
    globalThis.minipaintWanGP = {
        state() { return {ready: ready === undefined ? true : ready, layout: layout}; },
        prompt() { return Promise.resolve({ok: true, prompt: ""}); },
        setPrompt() { return Promise.resolve({ok: true}); },
    };
}

const STUDIO = {offered: true, compact: true, studio: true, asked: "compact", shown: "compact", missing: ""};

function modeShell(workspace, state) {
    const shell = Object.create(NS.Shell.prototype);
    shell.state = Object.assign({panelOpen: true, focusEnabled: false, focusBar: true,
                                 studioReturn: null}, state || {});
    shell.focus = fakeFocus("");
    shell.host = {active: workspace, getActiveWorkspace() { return this.active; }};
    shell.studio = null;
    shell.peekTimer = 0;
    shell.disposers = [];
    shell.said = [];
    shell.saved = [];
    shell.nodes = {panel: {dataset: {}}, menu: null};
    shell.showOpen = (open) => { shell.state.panelOpen = !!open; };
    shell._save = () => { shell.saved.push(JSON.stringify({focus: shell.state.focusEnabled,
                                                           back: shell.state.studioReturn})); };
    shell.place = () => {};
    shell.tell = (text, kind) => shell.said.push([kind, text]);
    shell.say = (text, kind) => shell.said.push([kind, text]);
    return shell;
}

const look = (shell) => ({
    focus: shell.focus.isActive(), studio: shell.focus.studio, mode: shell.studio ? "super" : null,
    open: shell.state.panelOpen, bar: shell.focus.bar, back: shell.state.studioReturn,
    marked: shell.nodes.panel.dataset.studio || "", focused: shell.nodes.panel.dataset.focus || "",
});
"""


class TestTheCycle:
    def test_on_the_wangp_tab_two_presses_go_normal_focused_super_normal(self):
        found = run(WORLD + """
            fakeWanGP(STUDIO);
            const shell = modeShell("tab_wangp");
            const moved = [];
            const seen = [];
            for (let i = 0; i < 4; i += 1) { moved.push(shell.cycleMode()); seen.push(look(shell)); }
            console.log(JSON.stringify({moved, seen}));
        """)

        assert found["moved"] == ["focused", "super", "normal", "focused"]
        focused, superfocused, normal, again = found["seen"]
        assert focused["focus"] is True and focused["studio"] is False and focused["open"] is True
        assert superfocused == {"focus": True, "studio": True, "mode": "super", "open": False,
                                "bar": "hidden", "back": {"panelOpen": True}, "marked": "on",
                                "focused": "on"}
        assert normal["focus"] is False and normal["studio"] is False and normal["mode"] is None
        assert normal["open"] is True and normal["back"] is None and normal["marked"] == "off"
        assert again["focus"] is True and again["studio"] is False

    def test_elsewhere_two_presses_go_normal_focused_normal(self):
        found = run(WORLD + """
            fakeWanGP(STUDIO);
            const shell = modeShell("tab_txt2img");
            const moved = [shell.cycleMode(), shell.cycleMode(), shell.cycleMode()];
            console.log(JSON.stringify({moved, studio: shell.focus.studio, said: shell.said}));
        """)

        assert found == {"moved": ["focused", "normal", "focused"], "studio": False, "said": []}

    def test_without_mini_paint_the_wangp_tab_has_no_third_mode_and_nothing_is_said(self):
        found = run(WORLD + """
            delete globalThis.minipaintWanGP;
            const shell = modeShell("tab_wangp");
            const moved = [shell.cycleMode(), shell.cycleMode()];
            console.log(JSON.stringify({moved, said: shell.said}));
        """)

        assert found == {"moved": ["focused", "normal"], "said": []}

    def test_a_bridge_older_than_the_studio_skips_it_and_says_to_reinstall(self):
        found = run(WORLD + """
            fakeWanGP(Object.assign({}, STUDIO, {studio: false}));
            const shell = modeShell("tab_wangp");
            const moved = [shell.cycleMode(), shell.cycleMode()];
            console.log(JSON.stringify({moved, studio: shell.focus.studio, said: shell.said}));
        """)

        assert found["moved"] == ["focused", "normal"] and found["studio"] is False
        assert found["said"] == [["warn", "Reinstall WanGP's bridge from the WanGP tab for Super Focused mode."]]

    def test_a_wangp_not_loaded_yet_has_it_and_its_page_is_laid_out_when_ready(self):
        """Mini Paint's tab tells the page at its ready: the class is what it
        reads, whenever that is."""
        found = run(WORLD + """
            fakeWanGP(Object.assign({}, STUDIO, {studio: false}), false);
            const shell = modeShell("tab_wangp");
            const moved = [shell.cycleMode(), shell.cycleMode()];
            console.log(JSON.stringify({moved, studio: shell.focus.studio}));
        """)

        assert found == {"moved": ["focused", "super"], "studio": True}

    def test_a_closed_panel_stays_closed_on_the_way_out(self):
        found = run(WORLD + """
            fakeWanGP(STUDIO);
            const shell = modeShell("tab_wangp", {panelOpen: false});
            shell.cycleMode(); shell.cycleMode();
            const inside = look(shell);
            shell.cycleMode();
            console.log(JSON.stringify({inside, out: look(shell)}));
        """)

        assert found["inside"]["back"] == {"panelOpen": False} and found["inside"]["open"] is False
        assert found["out"]["open"] is False and found["out"]["focus"] is False


class TestEveryWayOut:
    """"When i return out of it, everything and all settings return": however
    super focus ends, the panel is back as it was before it began."""

    SETUP = WORLD + """
        fakeWanGP(STUDIO);
        const shell = modeShell("tab_wangp");
        shell.cycleMode(); shell.cycleMode();
        const inside = look(shell);
    """

    def test_escape_and_the_browser_leaving_full_screen_end_it_through_focus_off(self):
        found = run(self.SETUP + """
            // What Escape's exit-focus and screenChanged both do.
            shell.focus.exit();
            shell.focusOff();
            console.log(JSON.stringify({inside: inside.mode, out: look(shell)}))
        """)

        assert found["inside"] == "super"
        out = found["out"]
        assert out["mode"] is None and out["open"] is True and out["back"] is None
        assert out["focus"] is False and out["marked"] == "off"

    def test_a_switch_to_another_tab_ends_it_and_focus_follows_the_switch(self):
        found = run(self.SETUP + """
            ["applyGenerate", "applyDock", "applyPrompt", "applySuppression", "renderWorkspaces"]
                .forEach((name) => { shell[name] = () => {}; });
            shell.store = {check() {}};
            shell.state.focusEnabled = true;
            shell.host.active = "tab_img2img";
            shell.navigated("tab_img2img");
            console.log(JSON.stringify({out: look(shell), on: shell.focus.activeWorkspace()}));
        """)

        out = found["out"]
        assert out["mode"] is None and out["open"] is True and out["back"] is None
        assert out["focus"] is True and out["studio"] is False and found["on"] == "tab_img2img"

    def test_a_blank_read_of_the_workspace_is_not_a_switch(self):
        found = run(self.SETUP + """
            ["applyGenerate", "applyDock", "applyPrompt", "applySuppression", "renderWorkspaces", "noteWorkspace"]
                .forEach((name) => { shell[name] = () => {}; });
            shell.navigated("");
            shell.navigated("tab_wangp");
            console.log(JSON.stringify({mode: shell.studio ? "super" : null, studio: shell.focus.studio}));
        """)

        assert found == {"mode": "super", "studio": True}

    def test_a_reload_in_super_focus_starts_from_what_it_put_away(self):
        """A reload starts outside focus, as it always has; the panel that was
        open before super focus is open after the reload."""
        found = run(WORLD + """
            const key = "forge-assistant:/:ui:v5";
            sessionStorage.setItem(key, JSON.stringify({schemaVersion: 5, panelOpen: false,
                                                        studioReturn: {panelOpen: true}}));
            NS.basePath = NS.basePath || (() => "/");
            const shell = Object.create(NS.Shell.prototype);
            shell.state = {panelOpen: false};
            shell._restore();
            const one = shell.state.panelOpen;
            sessionStorage.setItem(key, JSON.stringify({schemaVersion: 5, panelOpen: true, studioReturn: null}));
            shell._restore();
            console.log(JSON.stringify({one, two: shell.state.panelOpen}));
        """)

        assert found == {"one": True, "two": True}

    def test_what_super_focus_put_away_is_saved_with_the_layout(self):
        source = run(WORLD + """
            console.log(JSON.stringify({save: NS.Shell.prototype._save.toString()}));
        """)["save"]

        assert "studioReturn: this.state.studioReturn" in source


class TestNoBarInSuperFocus:
    def test_the_bar_is_hidden_whatever_the_setting_and_no_tap_peeks_it(self):
        found = run(WORLD + """
            fakeWanGP(STUDIO);
            const shell = modeShell("tab_wangp", {focusBar: true});
            shell.cycleMode();
            const focused = shell.focus.bar;
            shell.cycleMode();
            const inside = shell.focus.bar;
            const peeked = shell.peekBar();
            shell.state.focusBar = false;
            const peekedHidden = shell.peekBar();
            shell.cycleMode();
            shell.cycleMode();
            console.log(JSON.stringify({focused, inside, peeked, peekedHidden, after: shell.focus.bar}));
        """)

        assert found == {"focused": "shown", "inside": "hidden", "peeked": False,
                         "peekedHidden": False, "after": "hidden"}


class TestGoingBackIsInTheWorkspaceMenu:
    def test_the_first_entry_goes_back_to_the_workspace_before_this_one(self):
        found = run(WORLD + """
            const made = [];
            const shell = modeShell("tab_wangp");
            shell.previousWorkspace = "tab_txt2img";
            shell.closeMenu = () => {};
            const went = [];
            shell.switchWorkspace = (id) => { went.push(id); return Promise.resolve(id); };
            const items = shell.backItems([{id: "tab_txt2img", label: "Txt2img", available: true},
                                           {id: "tab_wangp", label: "WanGP", available: true}], "tab_wangp");
            items[0].handlers.click[0]();
            const none = shell.backItems([{id: "tab_txt2img", label: "Txt2img", available: false}], "tab_wangp").length;
            shell.previousWorkspace = "tab_wangp";
            const self = shell.backItems([{id: "tab_wangp", label: "WanGP", available: true}], "tab_wangp").length;
            console.log(JSON.stringify({count: items.length, text: items[0].textContent, went, none, self}));
        """)

        assert found == {"count": 1, "text": "Back to Txt2img", "went": ["tab_txt2img"], "none": 0, "self": 0}

    def test_the_workspace_menu_leads_with_it(self):
        found = run(WORLD + """
            const shell = modeShell("tab_wangp");
            shell.previousWorkspace = "tab_txt2img";
            shell.host.listWorkspaces = () => [{id: "tab_txt2img", label: "Txt2img", available: true},
                                               {id: "tab_wangp", label: "WanGP", available: true}];
            const items = shell.workspaceItems();
            console.log(JSON.stringify(items.map((item) => item.textContent)));
        """)

        assert found == ["Back to Txt2img", "Txt2img", "WanGP"]
