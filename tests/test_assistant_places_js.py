"""The launcher's place and the open panel's, each its own (docs/22 §3.44).

Asked for:

    "I want the flyout menu open state to stay put when opened and closed, as
    well as the button to stay in its position. They should keep their
    positions independently. So, for example, if I have the button in the
    upper right, and I have the open state in the lower left, closing returns
    the button in the upper right. So it should keep the same position
    separately, independently. This should be the new global behavior for
    open and closing."

These hold the two places under node: each placed at its own, a drag writing
only the one it holds, Free Float carrying each where it is, and a layout
saved while the two shared one place giving the panel that place. That a hand
puts the launcher in one corner and the panel in another and every way of
closing and opening keeps both is test_assistant_layout.py's; that putting the
panel away moves neither is test_assistant_tap_close_js.py's.
"""

from __future__ import annotations

from test_assistant_js import PANEL, run


def test_each_is_placed_at_its_own_and_the_sheet_takes_the_panels_half():
    """Closed, the launcher at the launcher's place; open, the panel at the
    panel's -- anchored and floating. On a phone's sheet the root says the
    panel's anchor, which is the half the sheet is drawn in."""
    found = run(PANEL + """
        const where = (state, width) => {
            sized(width || 1280, 900);
            const shell = panel({state});
            const root = shell.nodes.root;
            shell.placeNow();
            const out = {anchor: root["data-anchor"]};
            out[state.panelOpen ? "panel" : "launcher"] =
                placed(state.panelOpen ? shell.nodes.panel : shell.nodes.launcher);
            return out;
        };
        const anchored = {freeFloat: false, anchorOverride: "top-right", panelAnchor: "bottom-left"};
        const floating = {freeFloat: true, floatAt: {x: 1, y: 0}, panelAt: {x: 0, y: 1},
                          anchorOverride: "bottom-center", panelAnchor: "top-center"};
        console.log(JSON.stringify({
            anchoredClosed: where(Object.assign({panelOpen: false}, anchored)),
            anchoredOpen: where(Object.assign({panelOpen: true}, anchored)),
            floatingClosed: where(Object.assign({panelOpen: false}, floating)),
            floatingOpen: where(Object.assign({panelOpen: true}, floating)),
            sheet: where({panelOpen: true, freeFloat: false, anchorOverride: "bottom-right",
                          panelAnchor: "top-left"}, 420),
        }));
    """, sources=("shell",))

    # The launcher is 140 by 44, the panel 360 by 140; the gap is 24 px.
    assert found["anchoredClosed"] == {"anchor": "top-right",
                                       "launcher": {"left": "1116px", "top": "24px"}}
    assert found["anchoredOpen"]["anchor"] == "bottom-left"
    assert (found["anchoredOpen"]["panel"]["left"], found["anchoredOpen"]["panel"]["top"]) == (
        "24px", "736px")
    assert found["floatingClosed"]["launcher"] == {"left": "1116px", "top": "24px"}, (
        "floating, the launcher at its own free place, not at its anchor")
    assert (found["floatingOpen"]["panel"]["left"], found["floatingOpen"]["panel"]["top"]) == (
        "24px", "736px")
    assert found["sheet"] == {"anchor": "top-left", "panel": {}}, "the sheet in the panel's half"


def test_a_drag_of_the_launcher_moves_the_launcher_alone():
    """The launcher dragged and let go, anchored and floating: the launcher's
    place is where it was let go, and the panel's is as it was -- the open
    panel shows where it was left, wherever the launcher has gone."""
    found = run(PANEL + """
        sized(1280, 900);
        const dragged = (state) => {
            const shell = panel({state: Object.assign({panelOpen: false, panelAnchor: "bottom-left",
                                                       panelAt: {x: 0, y: 1}}, state)});
            shell.preview = () => undefined;
            shell.drag = {node: shell.nodes.launcher, pointerId: 1,
                          startX: 1170, startY: 722, offsetX: 70, offsetY: 22,
                          startLeft: 1100, startTop: 700, width: 140, height: 44,
                          moved: false, anchor: shell.anchor(shell.nodes.launcher),
                          committed: false};
            shell.moveDrag({pointerId: 1, clientX: 1170, clientY: 60});
            shell.endDrag({pointerId: 1}, false);
            const s = shell.state;
            return {launcher: [s.anchorOverride, s.floatAt], panel: [s.panelAnchor, s.panelAt]};
        };
        console.log(JSON.stringify({anchored: dragged({freeFloat: false}),
                                    floating: dragged({freeFloat: true})}));
    """, sources=("shell",))

    assert found["anchored"] == {"launcher": ["top-right", None],
                                 "panel": ["bottom-left", {"x": 0, "y": 1}]}
    launcher_anchor, launcher_at = found["floating"]["launcher"]
    assert launcher_anchor is None and launcher_at["x"] > 0.95 and launcher_at["y"] < 0.1, (
        found["floating"])
    assert found["floating"]["panel"] == ["bottom-left", {"x": 0, "y": 1}]


def test_free_float_carries_each_where_it_is():
    """Turned on, each takes a free place where it is: the one on screen as
    it is measured, the one put away at its own resting place -- a stale free
    place from an earlier turn is not where either is. Turned off, each goes
    to the resting place nearest its own: the one on screen by its box, the
    one put away by its free place."""
    found = run(PANEL + """
        sized(1280, 900);
        const on = panel({state: {freeFloat: false, anchorOverride: "top-center",
                                  floatAt: {x: 0.2, y: 0.9}, panelAt: {x: 0.1, y: 0.1}}});
        on.setFreeFloat(true);
        const off = panel({state: {freeFloat: true, anchorOverride: "bottom-right",
                                   floatAt: {x: 0.1, y: 0.3}, panelAnchor: "top-left",
                                   panelAt: {x: 1, y: 1}}});
        off.setFreeFloat(false);
        const offClosed = panel({state: {panelOpen: false, freeFloat: true,
                                         anchorOverride: "top-left", floatAt: {x: 0.6, y: 0.7},
                                         panelAnchor: "top-left", panelAt: {x: 0.6, y: 0.8}}});
        offClosed.setFreeFloat(false);
        const s = (shell) => ({launcher: [shell.state.anchorOverride, shell.state.floatAt],
                               panel: [shell.state.panelAnchor, shell.state.panelAt]});
        console.log(JSON.stringify({on: s(on), off: s(off), offClosed: s(offClosed)}));
    """, sources=("shell",))

    # On: the launcher, put away at the top centre, is halfway across and at
    # the top; the panel, on screen at (900, 500), is measured.
    assert found["on"]["launcher"] == ["top-center", {"x": 0.5, "y": 0}]
    panel_at = found["on"]["panel"][1]
    assert panel_at["x"] == 1 and abs(panel_at["y"] - (500 - 24) / (900 - 48 - 48)) < 1e-9
    # Off: the panel on screen, its box's middle at (1080, 570), is nearest
    # the bottom right; the launcher, put away at 0.1 across and 0.3 down,
    # nearest the top left. Their free places are kept for the next turn.
    assert found["off"]["panel"][0] == "bottom-right"
    assert found["off"]["launcher"] == ["top-left", {"x": 0.1, "y": 0.3}]
    # Closed, the launcher is on screen (at 1100, 700: the bottom right) and
    # the panel, put away at 0.6 across and 0.8 down, nearest the bottom's
    # middle: the six are at none, half and all of the way across.
    assert found["offClosed"]["launcher"][0] == "bottom-right"
    assert found["offClosed"]["panel"] == ["bottom-center", {"x": 0.6, "y": 0.8}]


def test_a_layout_from_when_the_two_shared_one_place_gives_the_panel_that_place():
    """Saved before the update, a layout holds one resting place and one free
    place, the launcher's and the panel's alike: both start from it, so
    nothing moves on the update. Saved after, each comes back as its own --
    and a panel saved with no place of its own stays without one, rather
    than taking the launcher's."""
    found = run(PANEL + """
        NS.basePath = () => "/";
        const local = {};
        globalThis.localStorage = {getItem: (k) => (k in local ? local[k] : null),
                                   setItem: (k, v) => { local[k] = String(v); }};
        sessionStorage.store["forge-assistant:/:ui:v5"] = JSON.stringify(
            {schemaVersion: 5, anchorOverride: "top-left", panelOpen: false});
        local["forge-assistant-float:/"] = JSON.stringify({freeFloat: true,
                                                           floatAt: {x: 0.25, y: 0.75}});
        const before = panel({state: {freeFloat: false}});
        before._restore();
        before._restoreFloat();
        const old = {launcher: [before.state.anchorOverride, before.state.floatAt],
                     panel: [before.state.panelAnchor, before.state.panelAt]};

        const saver = panel({state: {anchorOverride: "top-right", panelAnchor: "bottom-left",
                                     floatAt: {x: 1, y: 0}, panelAt: {x: 0, y: 1}}});
        saver._save = NS.Shell.prototype._save;
        saver._save();
        saver._saveFloat();
        const after = panel();
        after._restore();
        after._restoreFloat();
        const now = {launcher: [after.state.anchorOverride, after.state.floatAt],
                     panel: [after.state.panelAnchor, after.state.panelAt]};

        const alone = panel({state: {anchorOverride: "top-right", floatAt: {x: 1, y: 0}}});
        alone._save = NS.Shell.prototype._save;
        alone._save();
        alone._saveFloat();
        const back = panel();
        back._restore();
        back._restoreFloat();
        console.log(JSON.stringify({old, now, alone: [back.state.panelAnchor, back.state.panelAt]}));
    """, sources=("shell", "store"))

    assert found["old"] == {"launcher": ["top-left", {"x": 0.25, "y": 0.75}],
                            "panel": ["top-left", {"x": 0.25, "y": 0.75}]}
    assert found["now"] == {"launcher": ["top-right", {"x": 1, "y": 0}],
                            "panel": ["bottom-left", {"x": 0, "y": 1}]}
    assert found["alone"] == [None, None]
