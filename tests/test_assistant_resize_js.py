"""The panel as a window you can resize, by its corner.

Asked for: "for the flyout menu, I would like for the conversation view to be
within a resizable flyout window so that I can resize it." The edge on the
panel's left only ever made it wider, and nothing showed it was there. The
corner is drawn, at the corner away from where the panel is held, and a drag
of it makes the panel as wide and its conversation as tall as the hand says;
the docked settings column's view, being the conversation's height, follows.

These hold the arithmetic under node: which corner, which way each edge grows,
the bounds, what is saved and how it is put back. That the corner is drawn
where it says, follows the hand and holds a floating panel's other edges is a
browser's question: test_assistant_layout.py.
"""

from __future__ import annotations

from test_assistant_js import run


WORLD = """
// A shell with the nodes the resize touches. The computed styles a test
// chooses: the transcript's max-height (the stylesheet's, or the corner's)
// and the panel's cap (the window's, as `placePanel` writes it).
globalThis.getComputedStyle = (node) => ({maxHeight: node.maxHeightValue || "none",
                                          getPropertyValue: () => "0px"});
function cornerShell(state) {
    const shell = Object.create(NS.Shell.prototype);
    const panel = document.createElement("section");
    const corner = document.createElement("div");
    const resize = document.createElement("div");
    const transcript = document.createElement("div");
    [corner, resize].forEach((node) => {
        node.captures = [];
        node.setPointerCapture = (id) => node.captures.push("set:" + id);
        node.releasePointerCapture = (id) => node.captures.push("release:" + id);
    });
    panel.offsetWidth = 360;
    panel.offsetHeight = 500;
    panel.maxHeightValue = "752px";
    panel.getBoundingClientRect = () => ({left: 896, top: 276, right: 1256, bottom: 776,
                                          width: 360, height: 500});
    transcript.offsetHeight = 200;
    transcript.maxHeightValue = "167px";
    shell.nodes = {panel, corner, resize, transcript};
    // The panel's own place; the launcher's, elsewhere, is not where the
    // panel is held.
    shell.state = Object.assign({panelOpen: true, freeFloat: false, panelAt: null,
                                 panelAnchor: "bottom-right", panelWidth: null,
                                 floatAt: {x: 0.5, y: 0}, anchorOverride: "top-center",
                                 conversationHeight: null}, state || {});
    shell.settings = {defaultAnchor: "bottom-right"};
    shell.resizing = null;
    shell.placed = 0;
    shell.placedNow = 0;
    shell.place = () => { shell.placed += 1; };
    shell.placeNow = () => { shell.placedNow += 1; };
    shell.saved = 0;
    shell._save = () => { shell.saved += 1; };
    shell._saveFloat = () => {};
    return shell;
}
const press = (over) => Object.assign({pointerId: 3, pointerType: "mouse", isPrimary: true,
                                       button: 0, clientX: 896, clientY: 276,
                                       preventDefault() {}}, over || {});
function sized(shell) {
    return {width: shell.state.panelWidth, height: shell.state.conversationHeight,
            variable: shell.nodes.panel.style.getPropertyValue("--forge-assistant-conversation"),
            sized: shell.nodes.panel.classList.contains("forge-assistant-sized")};
}
"""


def test_the_corner_is_the_one_away_from_where_the_panel_is_held():
    found = run(WORLD + """
        const at = (anchor, extra) => {
            const shell = cornerShell(Object.assign({panelAnchor: anchor}, extra || {}));
            return shell.cornerSides();
        };
        const floating = cornerShell({freeFloat: true, panelAt: {x: 0, y: 0}});
        floating.nodes.panel.getBoundingClientRect = () => ({left: 24, top: 24, right: 384,
                                                             bottom: 524, width: 360, height: 500});
        console.log(JSON.stringify({
            bottomRight: at("bottom-right"), topLeft: at("top-left"),
            bottomCenter: at("bottom-center"), topRight: at("top-right"),
            floatingLow: at("bottom-right", {freeFloat: true, panelAt: {x: 1, y: 1}}),
            floatingHigh: floating.cornerSides(),
            sheet: cornerShell().cornerSides(true),
            sheetTop: cornerShell({panelAnchor: "top-left"}).cornerSides(true),
        }));
    """)

    assert found["bottomRight"] == {"x": "left", "y": "top", "factor": 1}
    assert found["topLeft"] == {"x": "right", "y": "bottom", "factor": 1}
    # Centred across the window it grows both ways: twice what the hand moves.
    assert found["bottomCenter"] == {"x": "left", "y": "top", "factor": 2}
    assert found["topRight"] == {"x": "left", "y": "bottom", "factor": 1}
    # Floating, away from the window's corner it is nearest.
    assert found["floatingLow"] == {"x": "left", "y": "top", "factor": 1}
    assert found["floatingHigh"] == {"x": "right", "y": "bottom", "factor": 1}
    # The phone's sheet is the screen's width: only its free edge.
    assert found["sheet"] == {"x": None, "y": "top", "factor": 1}
    assert found["sheetTop"] == {"x": None, "y": "bottom", "factor": 1}


def test_a_drag_of_the_corner_sizes_the_panel_and_its_conversation():
    """The panel along the bottom right: its top-left corner. Up and to the
    left by (50, 80) is 50 wider and 80 taller -- the corner under the hand --
    once a frame, and saved where the hand lets go."""
    found = run(WORLD + """
        const shell = cornerShell();
        shell.startResize(press(), shell.nodes.corner);
        shell.moveResize(press({clientX: 846, clientY: 196}));
        const moved = sized(shell);
        shell.moveResize(press({pointerId: 9, clientX: 100, clientY: 100}));
        const other = sized(shell);
        shell.endResize(press(), false);
        console.log(JSON.stringify({moved, other, placed: shell.placed, now: shell.placedNow,
                                    saved: shell.saved, captures: shell.nodes.corner.captures,
                                    edge: shell.nodes.resize.captures, resizing: shell.resizing}));
    """)

    assert found["moved"] == {"width": 410, "height": 280, "variable": "280px", "sized": True}
    assert found["other"] == found["moved"], "another pointer's moves resize nothing"
    assert found["placed"] == 1 and found["now"] == 1
    assert found["saved"] == 1
    assert found["captures"] == ["set:3", "release:3"] and found["edge"] == []
    assert found["resizing"] is None


def test_the_conversation_keeps_within_the_window_and_above_its_floor():
    """No taller than the panel's cap leaves for it -- 752 less the 300 of
    everything else -- and no shorter than the stylesheet's floor."""
    found = run(WORLD + """
        const shell = cornerShell();
        shell.startResize(press(), shell.nodes.corner);
        shell.moveResize(press({clientY: -900}));
        const tall = shell.state.conversationHeight;
        shell.moveResize(press({clientY: 900}));
        const short = shell.state.conversationHeight;
        shell.moveResize(press({clientX: -2000}));
        const wide = shell.state.panelWidth;
        shell.moveResize(press({clientX: 2000}));
        console.log(JSON.stringify({tall, short, wide, narrow: shell.state.panelWidth}));
    """)

    assert found == {"tall": 452, "short": 120, "wide": 960, "narrow": 320}


def test_a_panel_centred_across_the_window_grows_both_ways():
    found = run(WORLD + """
        const shell = cornerShell({panelAnchor: "bottom-center"});
        shell.startResize(press(), shell.nodes.corner);
        shell.moveResize(press({clientX: 866}));
        console.log(JSON.stringify(shell.state.panelWidth));
    """)

    assert found == 420


def test_docked_the_height_is_the_docked_views():
    """Docked, the transcript is out of sight; the height the docked view is
    given for it is the transcript's max-height (`sizeDock`), so that is what
    the corner starts from, and what it sets."""
    found = run(WORLD + """
        const shell = cornerShell();
        shell.dockedNow = true;
        shell.nodes.transcript.offsetHeight = 0;
        shell.startResize(press(), shell.nodes.corner);
        const from = shell.resizing.fromHeight;
        shell.moveResize(press({clientY: 226}));
        console.log(JSON.stringify({from, most: shell.resizing.most,
                                    height: shell.state.conversationHeight}));
    """)

    # 500 of panel, 167 of it the conversation's: 752 - 333 at most.
    assert found == {"from": 167, "most": 419, "height": 217}


def test_the_edge_still_only_widens():
    found = run(WORLD + """
        const shell = cornerShell();
        shell.startResize(press({clientX: 500}));
        shell.moveResize(press({clientX: 460, clientY: 0}));
        console.log(JSON.stringify({sized: sized(shell), edge: shell.nodes.resize.captures}));
    """)

    assert found["sized"] == {"width": 400, "height": None, "variable": "", "sized": False}
    assert found["edge"] == ["set:3"]


def test_the_corners_arrows_move_it_and_a_double_press_puts_the_panels_size_back():
    found = run(WORLD + """
        const shell = cornerShell();
        const key = (name) => shell.resizeKey({key: name, preventDefault() {}}, shell.nodes.corner);
        key("ArrowUp");
        key("ArrowLeft");
        const grown = sized(shell);
        key("ArrowRight");
        const narrower = shell.state.panelWidth;
        key("Enter");
        const saves = shell.saved;
        shell.resetSize();
        console.log(JSON.stringify({grown, narrower, saves, reset: sized(shell),
                                    now: shell.placedNow}));
    """)

    assert found["grown"] == {"width": 376, "height": 216, "variable": "216px", "sized": True}
    assert found["narrower"] == 360
    assert found["saves"] == 3, "a key that moves nothing saves nothing"
    assert found["reset"] == {"width": None, "height": None, "variable": "", "sized": False}
    assert found["now"] == 4


def test_an_empty_conversation_follows_the_hand_from_where_it_is_drawn():
    """With no messages the transcript is drawn 0 tall. The corner starts from
    that, not from the floor or the stylesheet's cap -- either would jump it
    at the first pixel -- and dragged back to nothing the conversation is the
    stylesheet's again."""
    found = run(WORLD + """
        const shell = cornerShell();
        shell.nodes.transcript.offsetHeight = 0;
        shell.nodes.panel.offsetHeight = 300;
        shell.startResize(press(), shell.nodes.corner);
        shell.moveResize(press({clientY: 266}));
        const first = sized(shell);
        shell.moveResize(press({clientY: 186}));
        const up = sized(shell);
        shell.moveResize(press({clientY: 400}));
        console.log(JSON.stringify({first, up, back: sized(shell)}));
    """)

    assert found["first"]["height"] == 10
    assert found["up"] == {"width": 360, "height": 90, "variable": "90px", "sized": True}
    assert found["back"] == {"width": 360, "height": None, "variable": "", "sized": False}


def test_the_size_is_kept_with_the_panels_layout():
    found = run(WORLD + """
        NS.basePath = () => "/";
        const shell = cornerShell({panelWidth: 500, conversationHeight: 333});
        shell._save = NS.Shell.prototype._save;
        shell._save();
        const next = Object.create(NS.Shell.prototype);
        next.state = {panelWidth: null, conversationHeight: null};
        next._restore();
        const key = Object.keys(sessionStorage.store)[0];
        const stored = JSON.parse(sessionStorage.store[key]);
        const restore = (value) => {
            sessionStorage.store[key] = JSON.stringify(Object.assign({}, stored,
                                                                     {conversationHeight: value}));
            const again = Object.create(NS.Shell.prototype);
            again.state = {panelWidth: null, conversationHeight: null};
            again._restore();
            return again.state.conversationHeight;
        };
        console.log(JSON.stringify({restored: [next.state.panelWidth, next.state.conversationHeight],
                                    others: [restore(90.4), restore(0), restore(-5), restore("tall"),
                                             restore(null)]}));
    """)

    assert found["restored"] == [500, 333]
    # A height a drag could make is kept; nothing else is a height.
    assert found["others"] == [90, None, None, None, None]
