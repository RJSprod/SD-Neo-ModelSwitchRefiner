"""The panel's third state: Txt2Img's settings column, docked in it.

Asked for: "a third state for the flyout menu, that removes the left column
from the text to image tab, and makes it scrollable in the flyout menu. Add
this button to the flyout menu bar only when text to image tab is open."

These hold the panel's side under node -- when the button shows, what each
press does to the view, that the column goes back to the page whenever the
panel stops showing it, and that the dock puts back exactly what it changed on
the column. Where the column is then drawn, and that it scrolls and takes
presses, is a question for a browser: test_assistant_layout.py.
"""

from __future__ import annotations

from test_assistant_js import run


WORLD = """
function dockShell(active, state) {
    const dock = {placed: [], released: 0, docked: false,
                  available(id) { return id === "tab_txt2img"; },
                  place(rect) { this.placed.push(rect); this.docked = true; return true; },
                  release() { if (this.docked) this.released += 1; this.docked = false;
                              return true; }};
    NS.dock = dock;
    const shell = Object.create(NS.Shell.prototype);
    shell.settings = {label: "Forge Assistant", bubbleWidth: 80};
    shell.state = Object.assign({panelOpen: true, conversationExpanded: true,
                                 settingsDocked: false}, state || {});
    shell.nodes = {root: {appendChild() {}}};
    shell.disposers = [];
    shell.host = {registerMenu: () => () => undefined, openOnly() {},
                  getActiveWorkspace: () => shell.activeWorkspace,
                  listWorkspaces: () => [{id: "tab_txt2img", label: "Txt2img", available: true}]};
    shell.activeWorkspace = active;
    shell.saved = 0;
    shell._save = () => { shell.saved += 1; };
    // Placing syncs the dock, as `placeNow` does; the geometry is not this
    // file's business.
    shell.place = () => shell.syncDock();
    shell.buildPanel();
    shell.nodes.dock.getBoundingClientRect = () => ({left: 10, top: 50, width: 340, height: 700});
    return {shell, dock};
}

function view(shell) {
    const nodes = shell.nodes;
    return {button: !nodes.settings.hidden, pressed: nodes.settings["aria-pressed"],
            body: !nodes.body.hidden, row: !nodes.workspaces.hidden, slot: !nodes.dock.hidden,
            picker: !nodes.picker.hidden, chat: nodes.chat["aria-expanded"],
            docked: nodes.panel.classList.contains("forge-assistant-docked"),
            collapsed: nodes.panel.classList.contains("forge-assistant-collapsed")};
}
"""


def test_the_button_sits_beside_chat_and_shows_on_txt2img_only():
    found = run(WORLD + """
        const here = dockShell("tab_txt2img").shell;
        const there = dockShell("tab_img2img").shell;
        const row = here.nodes.header.children;
        console.log(JSON.stringify({
            after: row[row.indexOf(here.nodes.settings) - 1] === here.nodes.chat,
            label: here.nodes.settings.getAttribute("aria-label"),
            glyph: here.nodes.settings.textContent,
            here: view(here).button, there: view(there).button,
            pressed: here.nodes.settings["aria-pressed"],
        }));
    """)

    assert found == {"after": True, "label": "Generation settings", "glyph": "\U0001F39B️",
                     "here": True, "there": False, "pressed": "false"}


def test_a_press_docks_the_column_in_the_conversations_place():
    found = run(WORLD + """
        const {shell, dock} = dockShell("tab_txt2img");
        shell.toggleDock();
        console.log(JSON.stringify({view: view(shell), state: shell.state.settingsDocked,
                                    placed: dock.placed, saved: shell.saved}));
    """)

    assert found["state"] is True
    assert found["view"] == {"button": True, "pressed": "true", "body": False, "row": False,
                             "slot": True, "picker": True, "chat": "false", "docked": True,
                             "collapsed": False}
    # Drawn where the panel's placeholder is.
    assert found["placed"][-1] == {"left": 10, "top": 50, "width": 340, "height": 700}
    assert found["saved"] == 1


def test_a_second_press_goes_back_to_the_view_it_replaced():
    found = run(WORLD + """
        const talking = dockShell("tab_txt2img");
        talking.shell.toggleDock();
        talking.shell.toggleDock();
        const tabs = dockShell("tab_txt2img", {conversationExpanded: false});
        tabs.shell.toggleDock();
        tabs.shell.toggleDock();
        console.log(JSON.stringify({talking: view(talking.shell), released: talking.dock.released,
                                    tabs: view(tabs.shell)}));
    """)

    assert found["talking"]["body"] is True and found["talking"]["docked"] is False
    assert found["talking"]["pressed"] == "false"
    assert found["released"] == 1, "the column goes back to the page"
    # The tab bar state comes back as the tab bar.
    assert found["tabs"]["row"] is True and found["tabs"]["collapsed"] is True
    assert found["tabs"]["body"] is False and found["tabs"]["slot"] is False


def test_chat_while_docked_puts_the_column_back_and_shows_the_conversation():
    found = run(WORLD + """
        const {shell, dock} = dockShell("tab_txt2img", {conversationExpanded: false});
        shell.toggleDock();
        shell.toggleChat();
        console.log(JSON.stringify({view: view(shell), docked: shell.state.settingsDocked,
                                    expanded: shell.state.conversationExpanded,
                                    released: dock.released}));
    """)

    assert found["docked"] is False and found["expanded"] is True
    assert found["view"]["body"] is True and found["view"]["chat"] == "true"
    assert found["view"]["slot"] is False
    assert found["released"] == 1


def test_another_workspace_keeps_the_column_out_of_sight_and_returning_shows_it():
    found = run(WORLD + """
        const {shell, dock} = dockShell("tab_txt2img");
        shell.toggleDock();
        shell.activeWorkspace = "tab_img2img";
        shell.applyDock();
        const away = Object.assign(view(shell), {released: dock.released,
                                                 last: dock.placed[dock.placed.length - 1],
                                                 kept: shell.state.settingsDocked});
        shell.activeWorkspace = "tab_txt2img";
        shell.applyDock();
        console.log(JSON.stringify({away, back: view(shell),
                                    last: dock.placed[dock.placed.length - 1]}));
    """)

    # Away: the conversation, as the panel was before, and no button.
    assert found["away"]["docked"] is False and found["away"]["body"] is True
    assert found["away"]["button"] is False
    # The column is not given back to the page: it is held out of sight.
    assert found["away"]["released"] == 0
    assert found["away"]["last"] is None
    # The choice is the panel's, kept for Txt2Img.
    assert found["away"]["kept"] is True
    assert found["back"]["docked"] is True and found["back"]["slot"] is True
    assert found["last"] == {"left": 10, "top": 50, "width": 340, "height": 700}


def test_closing_the_panel_keeps_the_column_out_of_the_page():
    """Asked for: "if i exit focus mode with the column enabled inside the
    flyout menu, it should remain hidden when focus mode exit. the only way to
    get it back is open the fly out and switch to conversation or tab mode".
    The first build gave the column back whenever the panel closed -- which is
    how a panel closed in focus mode, for the whole gallery, ended focus mode
    with the column in the page."""
    found = run(WORLD + """
        const {shell, dock} = dockShell("tab_txt2img");
        shell.toggleDock();
        shell.state.panelOpen = false;
        shell.syncDock();
        const closed = {released: dock.released, last: dock.placed[dock.placed.length - 1]};
        shell.state.panelOpen = true;
        shell.syncDock();
        console.log(JSON.stringify({closed, last: dock.placed[dock.placed.length - 1]}));
    """)

    assert found["closed"] == {"released": 0, "last": None}, "held, out of sight"
    assert found["last"] == {"left": 10, "top": 50, "width": 340, "height": 700}


def test_only_the_conversation_or_the_tab_bar_gives_the_column_back():
    found = run(WORLD + """
        const {shell, dock} = dockShell("tab_txt2img", {conversationExpanded: false});
        shell.toggleDock();
        // Closed and opened, another workspace and back: still held.
        shell.state.panelOpen = false;
        shell.syncDock();
        shell.state.panelOpen = true;
        shell.activeWorkspace = "tab_img2img";
        shell.applyDock();
        shell.activeWorkspace = "tab_txt2img";
        shell.applyDock();
        const held = dock.released;
        // The tab bar, from the settings button: given back.
        shell.toggleDock();
        const tabs = {released: dock.released, row: view(shell).row};
        // Docked again, then the conversation, from Chat: given back.
        shell.toggleDock();
        shell.toggleChat();
        console.log(JSON.stringify({held, tabs, chat: {released: dock.released,
                                                       body: view(shell).body}}));
    """)

    assert found["held"] == 0
    assert found["tabs"] == {"released": 1, "row": True}
    assert found["chat"] == {"released": 2, "body": True}


def test_a_page_opened_with_the_column_docked_and_the_panel_closed_holds_it():
    """The docked state is kept with the panel's layout for the tab, so a
    reload with the panel closed keeps the column out of the page too."""
    found = run(WORLD + """
        const {shell, dock} = dockShell("tab_txt2img", {panelOpen: false,
                                                        settingsDocked: true});
        console.log(JSON.stringify({placed: dock.placed, released: dock.released}));
    """)

    assert found["placed"] and all(rect is None for rect in found["placed"])
    assert found["released"] == 0


def test_a_menu_takes_the_columns_place_and_gives_it_back():
    found = run(WORLD + """
        const {shell, dock} = dockShell("tab_txt2img");
        shell.toggleDock();
        shell.toggleMenu("workspaces");
        const up = {slot: view(shell).slot, last: dock.placed[dock.placed.length - 1]};
        shell.closeMenu();
        const down = {slot: view(shell).slot, last: dock.placed[dock.placed.length - 1]};
        // With a menu up, the settings button is the way back to the column.
        shell.toggleMenu("workspaces");
        shell.toggleDock();
        console.log(JSON.stringify({up, down, again: view(shell),
                                    menu: !shell.nodes.menu.hidden}));
    """)

    assert found["up"] == {"slot": False, "last": None}, "kept docked, out of sight"
    assert found["down"]["slot"] is True and found["down"]["last"]["width"] == 340
    assert found["again"]["docked"] is True and found["again"]["slot"] is True
    assert found["menu"] is False


def test_the_choice_is_saved_with_the_panels_layout():
    found = run(WORLD + """
        NS.basePath = () => "/";
        const shell = Object.create(NS.Shell.prototype);
        shell.state = {anchorOverride: null, panelWidth: null, panelOpen: true,
                       conversationExpanded: true, settingsDocked: true,
                       focusEnabled: false, focusWorkspaceId: null};
        shell._save();
        const stored = Object.values(sessionStorage.store)[0];
        const next = Object.create(NS.Shell.prototype);
        next.state = {settingsDocked: false};
        next._restore();
        console.log(JSON.stringify({stored: JSON.parse(stored).settingsDocked,
                                    restored: next.state.settingsDocked}));
    """)

    assert found == {"stored": True, "restored": True}


# -- the dock itself ----------------------------------------------------------- #

COLUMN_WORLD = """
function styled(initial) {
    const values = {}, priorities = {};
    Object.entries(initial || {}).forEach(([name, [value, priority]]) => {
        values[name] = value; priorities[name] = priority || "";
    });
    return {values, priorities,
            setProperty(name, value, priority) { values[name] = value;
                                                 priorities[name] = priority || ""; },
            removeProperty(name) { delete values[name]; delete priorities[name]; },
            getPropertyValue(name) { return values[name] || ""; },
            getPropertyPriority(name) { return priorities[name] || ""; }};
}
function classes() {
    const names = new Set();
    return {names, add: (n) => names.add(n), remove: (n) => names.delete(n),
            contains: (n) => names.has(n)};
}
// A transform on an ancestor: whatever is written lands this far off.
let trap = {x: 0, y: 0};
const row = {classList: classes(), parentElement: null};
const column = {classList: classes(), parentElement: row,
                // The guard listens on the column for a field reached by Tab.
                addEventListener() {}, removeEventListener() {},
                querySelectorAll: () => [], scrollTop: 0,
                style: styled({width: ["300px", "important"], "min-width": ["320px", ""]}),
                getBoundingClientRect() {
                    return {left: parseFloat(this.style.values.left) + trap.x,
                            top: parseFloat(this.style.values.top) + trap.y};
                }};
const settings = {parentElement: column, closest: (s) => s === ".resize-handle-row" ? row : null};
document.getElementById = (id) => id === "txt2img_settings" ? settings : null;
"""


def test_the_dock_draws_the_column_where_it_is_told_and_puts_back_what_it_found():
    found = run(COLUMN_WORLD + """
        const dock = NS.dock;
        const available = [dock.available("tab_txt2img"), dock.available("tab_img2img")];
        dock.place({left: 10, top: 50, width: 340.4, height: 700});
        const placed = Object.assign({}, column.style.values);
        const marks = [column.classList.contains("forge-assistant-docked-column"),
                       row.classList.contains("forge-assistant-docked-row")];
        const important = column.style.getPropertyPriority("left");
        dock.place(null);
        const tucked = column.style.values.visibility;
        dock.release();
        console.log(JSON.stringify({available, placed, marks, important, tucked,
                                    after: column.style.values,
                                    priorities: column.style.priorities,
                                    unmarked: [column.classList.names.size,
                                               row.classList.names.size],
                                    docked: dock.isDocked()}));
    """, sources=("layout",))

    assert found["available"] == [True, False]
    assert found["placed"] == {"width": "340px", "min-width": "320px", "height": "700px",
                               "visibility": "visible", "left": "10px", "top": "50px"}
    assert found["marks"] == [True, True]
    assert found["important"] == "important"
    assert found["tucked"] == "hidden", "a hidden placeholder keeps the column out of sight"
    # Exactly what was there: the width and its priority back, the rest gone.
    assert found["after"] == {"width": "300px", "min-width": "320px"}
    assert found["priorities"] == {"width": "important", "min-width": ""}
    assert found["unmarked"] == [0, 0]
    assert found["docked"] is False


def test_the_dock_corrects_for_an_ancestor_that_moves_fixed_boxes():
    found = run(COLUMN_WORLD + """
        trap = {x: 7, y: -12};
        NS.dock.place({left: 100, top: 60, width: 300, height: 500});
        const box = column.getBoundingClientRect();
        console.log(JSON.stringify({left: box.left, top: box.top}));
    """, sources=("layout",))

    assert found == {"left": 100, "top": 60}


def test_without_the_settings_column_there_is_nothing_to_offer():
    found = run(COLUMN_WORLD + """
        document.getElementById = () => null;
        console.log(JSON.stringify({available: NS.dock.available("tab_txt2img"),
                                    placed: NS.dock.place({left: 0, top: 0, width: 1,
                                                           height: 1})}));
    """, sources=("layout",))

    assert found == {"available": False, "placed": False}


def test_the_docked_view_is_the_conversations_height_at_its_fullest():
    """Asked for after the first build, which made the docked panel the
    window's height: "maintain the same height restriction as the fly out menu
    in conversation mode". The placeholder is the conversation body's height
    with the transcript at its stylesheet maximum: the body's other parts as
    measured (the body is shown for the measurement and hidden again), plus
    the transcript's max-height."""
    found = run(WORLD + """
        const {shell} = dockShell("tab_txt2img");
        const body = shell.nodes.body;
        const transcript = shell.nodes.transcript;
        // Laid out only while shown: the measurement has to show the body.
        Object.defineProperty(body, "offsetHeight", {get() { return this.hidden ? 0 : 310; }});
        Object.defineProperty(transcript, "offsetHeight",
                              {get() { return body.hidden ? 0 : 60; }});
        globalThis.getComputedStyle = (node) => ({
            maxHeight: node === transcript ? "199px" : "none",
            getPropertyValue: () => "0px"});
        shell.toggleDock();
        shell.sizeDock();
        const docked = {height: shell.nodes.dock.style.height, body: body.hidden,
                        slot: shell.nodes.dock.hidden};
        shell.toggleDock();
        shell.sizeDock();
        console.log(JSON.stringify({docked, undocked: shell.nodes.dock.style.height || ""}));
    """)

    # 310 - 60 of status line and composer, and 199 of transcript.
    assert found["docked"] == {"height": "449px", "body": True, "slot": False}
    assert found["undocked"] == ""
