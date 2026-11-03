"""Transpose View's rules under node: what is kept, what is written, and when.

Asked for (docs/22 §3.47): "a new mode in flyout menu "..." menu that is
called "transpose view" ... the tab bar of webui ... on a slim strip on the
left of screen ... Same on flyout menu buttons ... a slim strip on the left of
the flyout, including the drag handle ... also need an option to choose right
side for either of these independently", and "i DO want this functionality in
focus mode as well".

What a page shows -- the rails, the page and the workspace beside them, the
menu -- is test_transpose_layout.py's, in Chromium. These hold what that
cannot see: the stored choice read back (and anything else read as the top),
the root, the panel and the theme touched only when a choice changes, and the
part of a floating panel that keeps its place, which is the rail's width and
not its whole height.
"""

from __future__ import annotations

import json

from test_assistant_js import run


WORLD = """
// A shell with nothing built: the store and the parts each rule reads, as the
// test chooses them.
NS.basePath = () => "/forge";
const stored = {};
globalThis.localStorage = {getItem: (k) => stored[k] === undefined ? null : stored[k],
                           setItem: (k, v) => { stored[k] = String(v); }};
const KEY = "forge-assistant-transpose:/forge";
function bare(state) {
    const shell = Object.create(NS.Shell.prototype);
    shell.state = Object.assign({}, state || {});
    shell.nodes = {};
    shell.told = [];
    shell.tell = (text) => { shell.told.push(text); };
    return shell;
}
// An element that counts what is written on it.
function marked() {
    const node = {attributes: {}, writes: 0};
    node.getAttribute = (name) => node.attributes[name] === undefined ? null : node.attributes[name];
    node.hasAttribute = (name) => node.attributes[name] !== undefined;
    node.setAttribute = (name, value) => { node.attributes[name] = String(value); node.writes += 1; };
    node.removeAttribute = (name) => { delete node.attributes[name]; node.writes += 1; };
    return node;
}
"""


def test_the_stored_choice_is_read_back_and_anything_else_is_the_top():
    found = run(WORLD + """
        const read = (value, key) => {
            Object.keys(stored).forEach((k) => delete stored[k]);
            if (value !== undefined) stored[key || KEY] = value;
            const shell = bare();
            shell._restoreTranspose();
            return [shell.state.tabRail, shell.state.panelRail];
        };
        console.log(JSON.stringify({
            both: read(JSON.stringify({tabs: "left", panel: "right"})),
            mirrored: read(JSON.stringify({tabs: "right", panel: "left"})),
            top: read(JSON.stringify({tabs: "top", panel: "top"})),
            nothing: read(undefined),
            broken: read("{not json"),
            unknown: read(JSON.stringify({tabs: "middle", panel: 3})),
            another: read(JSON.stringify({tabs: "left", panel: "left"}), "forge-assistant-transpose:/"),
        }));
    """)

    assert found["both"] == ["left", "right"] and found["mirrored"] == ["right", "left"]
    for name in ("top", "nothing", "broken", "unknown"):
        assert found[name] == ["", ""], name
    assert found["another"] == ["", ""], "another WebUI's choice on the same origin is its own"


def test_each_choice_is_kept_beside_the_other_and_says_what_it_did():
    found = run(WORLD + """
        const shell = bare();
        shell.applied = 0;
        shell.applyTranspose = () => { shell.applied += 1; };
        const steps = [];
        const step = (answer) => steps.push({answer, kept: JSON.parse(stored[KEY])});
        step(shell.setTabRail("left"));
        step(shell.setPanelRail("right"));
        step(shell.setTabRail("sideways"));
        step(shell.setPanelRail(""));
        console.log(JSON.stringify({steps, applied: shell.applied, told: shell.told}));
    """)

    assert found["steps"] == [
        {"answer": "left", "kept": {"tabs": "left", "panel": "top"}},
        {"answer": "right", "kept": {"tabs": "left", "panel": "right"}},
        {"answer": "", "kept": {"tabs": "top", "panel": "right"}},
        {"answer": "", "kept": {"tabs": "top", "panel": "top"}},
    ]
    assert found["applied"] == 4
    assert found["told"] == ["The tab bar runs down the window's left side.",
                             "The panel's buttons run down its right side.",
                             "The tab bar runs along the top.",
                             "The panel's buttons run along its top."]


def test_the_root_the_panel_and_the_theme_are_touched_only_when_a_choice_changes():
    """The theme draws its whole bar again on `apply`, and focus measures its
    bar again on `setBar`: neither is asked on a call that changes nothing --
    `applyTranspose` runs at every build -- and a change of the panel's side
    alone redraws neither, only places the panel again for its new shape."""
    found = run(WORLD + """
        const html = marked();
        globalThis.document.documentElement = html;
        const panel = marked();
        let themed = 0;
        globalThis.mcTheme = {apply: () => { themed += 1; }};
        const shell = bare({tabRail: "", panelRail: ""});
        shell.nodes = {panel};
        const bars = [];
        let active = true;
        shell.focus = {isActive: () => active, barMode: "peek", setBar: (mode) => { bars.push(mode); }};
        let placed = 0;
        shell.place = () => { placed += 1; };
        shell.fitRail = () => {};
        const look = () => ({root: html.getAttribute("data-forge-assistant-rail"),
                             panel: panel.getAttribute("data-rail"),
                             rootWrites: html.writes, panelWrites: panel.writes,
                             themed, bars: bars.length, placed});
        const out = {};
        shell.applyTranspose();
        out.unchanged = look();
        shell.state.tabRail = "left";
        shell.applyTranspose();
        out.tabs = look();
        shell.applyTranspose();
        out.again = look();
        shell.state.panelRail = "right";
        shell.applyTranspose();
        out.panel = look();
        active = false;
        shell.state.tabRail = "";
        shell.applyTranspose();
        out.back = look();
        out.mode = bars[0];
        console.log(JSON.stringify(out));
    """)

    assert found["unchanged"] == {"root": None, "panel": None, "rootWrites": 0, "panelWrites": 0,
                                  "themed": 0, "bars": 0, "placed": 0}
    assert found["tabs"] == {"root": "left", "panel": None, "rootWrites": 1, "panelWrites": 0,
                             "themed": 1, "bars": 1, "placed": 0}
    assert found["mode"] == "peek", "focus keeps the bar's mode and only measures it again"
    assert found["again"] == found["tabs"]
    assert found["panel"] == {"root": "left", "panel": "right", "rootWrites": 1, "panelWrites": 1,
                              "themed": 1, "bars": 1, "placed": 1}
    assert found["back"] == {"root": None, "panel": "right", "rootWrites": 2, "panelWrites": 1,
                             "themed": 2, "bars": 1, "placed": 1}, "focus off: nothing to measure"


def test_a_floating_panel_keeps_the_rails_width_in_place_not_its_whole_height():
    """`floatHold` is what of a floating panel holds its place as the panel
    changes height: the header row's height. A rail is the header too, down
    the panel's whole height; held whole, the panel would stand off the
    window's bottom like the launcher, so the rail's thin side is held."""
    found = run(WORLD + """
        const shell = bare();
        const header = {box: null, getBoundingClientRect: () => header.box};
        const panel = {};
        shell.nodes = {panel, header};
        const hold = (box) => { header.box = box; return shell.floatHold(panel); };
        console.log(JSON.stringify({
            row: hold({width: 360, height: 36}),
            rail: hold({width: 44, height: 600}),
            heightOnly: hold({height: 36}),
            unlaid: hold({width: 44, height: 0}),
            launcher: shell.floatHold({}),
        }));
    """)

    assert found == {"row": 36, "rail": 44, "heightOnly": 36, "unlaid": 0, "launcher": 0}


def test_the_panels_least_height_is_its_rails_and_is_written_once():
    """A rail is laid beside the panel's column, so the panel cannot size
    itself to it: its least height is measured -- the buttons it shows, the
    grip's least height, the gaps, the rail's padding and the panel's borders
    -- and written only when it changes; taken off when the rail goes, and
    nothing written on a panel that never had one."""
    found = run(WORLD + """
        const writes = [];
        const style = {
            values: {},
            setProperty(name, value) { style.values[name] = value; writes.push([name, value]); },
            removeProperty(name) { delete style.values[name]; writes.push([name, null]); },
        };
        const panel = {style, computed: {borderTopWidth: "1px", borderBottomWidth: "1px"}};
        const button = (height, more) => Object.assign({hidden: false, computed: {display: "flex"},
            getBoundingClientRect: () => ({height})}, more || {});
        const grip = button(999, {computed: {display: "flex", minHeight: "49px"}});
        const header = {computed: {paddingTop: "4.2px", paddingBottom: "4.2px", rowGap: "4.2px"},
                        children: [button(36), button(36), button(36, {hidden: true}),
                                   button(36, {computed: {display: "none"}}), button(36), grip]};
        let measured = 0;
        globalThis.getComputedStyle = (node) => { measured += 1; return node.computed || {}; };
        const shell = bare({panelRail: ""});
        shell.nodes = {panel, header, grip};
        const out = {};
        shell.fitRail();
        out.never = {writes: writes.length, measured};
        shell.state.panelRail = "left";
        shell.fitRail();
        shell.fitRail();
        out.railed = {writes: writes.length, value: style.values["--forge-assistant-rail-min"]};
        header.children.push(button(36));
        shell.fitRail();
        out.grown = style.values["--forge-assistant-rail-min"];
        shell.state.panelRail = "";
        shell.fitRail();
        const before = measured;
        shell.fitRail();
        out.gone = {writes: writes.length, last: writes[writes.length - 1], measured: measured - before};
        console.log(JSON.stringify(out));
    """)

    # 4.2 + 4.2 padding, three buttons of 36, the grip's 49 (not its 999),
    # three gaps of 4.2 between the four shown, and the two borders: 180.
    assert found["never"] == {"writes": 0, "measured": 0}, "placed often: nothing measured without a rail"
    assert found["railed"] == {"writes": 1, "value": "180px"}
    assert found["grown"] == "221px", "one more button and its gap: 180 + 36 + 4.2, rounded up"
    assert found["gone"] == {"writes": 3, "last": ["--forge-assistant-rail-min", None], "measured": 0}


def test_the_menu_entry_opens_the_submenu_and_a_choice_closes_the_menu_first():
    found = run(WORLD + """
        const html = marked();
        html.attributes["data-mc-theme"] = "on";
        globalThis.document.documentElement = html;
        const shell = bare({tabRail: "right", panelRail: ""});
        shell.shown = [];
        shell.showMenu = (which) => { shell.shown.push(which); };
        const order = [];
        shell.closeMenu = () => { order.push("close"); };
        shell.setTabRail = (side) => { order.push("tabs:" + side); };
        shell.setPanelRail = (side) => { order.push("panel:" + side); };
        shell.backItem = () => ({backItem: true, textContent: "Back"});
        shell.groupLabel = (text) => ({label: true, textContent: text});
        const entry = shell.transposeItem();
        (entry.handlers.click || []).forEach((fn) => fn({}));
        const items = shell.transposeItems();
        const radios = items.filter((item) => item.getAttribute && item.getAttribute("role") === "menuitemradio");
        radios.forEach((item) => (item.handlers.click || []).forEach((fn) => fn({})));
        html.attributes["data-mc-theme"] = "off";
        const plain = shell.transposeItems().filter((item) => item.dataset && item.dataset.rail);
        console.log(JSON.stringify({
            shown: shell.shown, entry: entry.textContent, popup: entry.getAttribute("aria-haspopup"),
            labels: items.map((item) => item.textContent),
            checked: radios.map((item) => item.getAttribute("aria-checked")),
            sides: radios.map((item) => item.dataset.rail + ":" + item.dataset.side),
            order, disabled: plain.map((item) => !!item.disabled),
        }));
    """)

    assert found["shown"] == ["transpose"] and found["entry"] == "Transpose View"
    assert found["popup"] == "menu"
    assert found["labels"] == ["Back", "Tab Bar", "Top", "Left", "Right",
                               "Panel Buttons", "Top", "Left", "Right"]
    assert found["checked"] == ["false", "false", "true", "true", "false", "false"]
    assert found["sides"] == ["tabs:top", "tabs:left", "tabs:right", "panel:top", "panel:left", "panel:right"]
    assert found["order"] == ["close", "tabs:", "close", "tabs:left", "close", "tabs:right",
                              "close", "panel:", "close", "panel:left", "close", "panel:right"]
    assert found["disabled"] == [False, True, True, False, False, False], \
        "the tab bar's sides wait for the theme, which draws the bar; the panel's never wait"


def test_the_names_are_shared_with_the_theme_and_focus():
    """The root's attribute is written by the shell and read by the theme's
    script, its stylesheet and focus mode: one name, held here."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "javascript"
    found = run(WORLD + """
        console.log(JSON.stringify({name: NS.RAIL_ATTRIBUTE,
                                    sides: ["left", "right", "top", "", "up"].map(NS.railSide)}));
    """)

    assert found == {"name": "data-forge-assistant-rail", "sides": ["left", "right", "", "", ""]}
    for script in ("model_chain_theme.js", "forge_assistant_focus.js"):
        text = (root / script).read_text(encoding="utf-8")
        assert re.search(r'RAIL_ATTRIBUTE = "data-forge-assistant-rail";', text), script
    assert json.dumps(found["name"])[1:-1] in (root.parent / "style.css").read_text(encoding="utf-8")
