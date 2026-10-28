"""The intelligent drag, one tap on the header, and no ✕ with Tap Outside to Close.

Asked for:

    "when tap out to close is enabled, the x button in the flyout menu should
    be removed. The close action at that point is the tap out gesture."

    "when fly out is open, let's make single tap of the drag handle perform
    the minimize as well. So another way back to the flyout button state."

    "I want intelligent drag when the flyout menu is open. Such case, if I
    drag to the edge of the screen and the flyout menu is now off screen,
    then it should resize so the edge that was going to be off screen is now
    in screen. essentially dynamically size the window so that it always
    appears in the screen view ... This should fix annoying bounce reset back
    to position I get when I drag it to the edge. Also, make the drag handle
    hit are larger. Its too small now."

These hold the arithmetic and the states under node. That a real drag draws
the panel so, lets go of it without a bounce and grows it back, and that a
drag may start on a header's button while a tap stays the button's, is
test_assistant_layout.py's.
"""

from __future__ import annotations

from test_assistant_js import run


def test_a_box_is_drawn_inside_the_room_cut_on_the_side_that_would_cross():
    """`fitBox`: inside, unchanged; across the right or the bottom edge, held
    there and cut; across the left, its left held and its right following
    the hand; across the top, stopped, never cut; cut no further than
    `least`, and past that stopped at the edge."""
    found = run("""
        const room = {minX: 24, maxX: 1256, minY: 24, maxY: 776};
        const least = {width: 320, height: 120};
        const fit = (left, top, width, height, small) => {
            const box = NS.fitBox({left, top, width, height}, room, small || least);
            return [Math.round(box.left), Math.round(box.top), Math.round(box.width),
                    Math.round(box.height), box.cutX, box.cutY];
        };
        console.log(JSON.stringify({
            inside: fit(400, 200, 400, 300),
            right: fit(1000, 200, 400, 300),
            rightFar: fit(1100, 200, 400, 300),
            left: fit(-50, 200, 400, 300),
            bottom: fit(400, 600, 400, 300),
            bottomFar: fit(400, 700, 400, 300),
            top: fit(400, -30, 400, 300),
            corner: fit(1000, 600, 400, 300),
            launcher: fit(1200, 760, 140, 44, {width: 140, height: 44}),
            small: fit(1200, 700, 200, 100),
        }));
    """, sources=("shell",))

    assert found["inside"] == [400, 200, 400, 300, False, False]
    # Across the right: held at 1256; 256 wide would be under 320, so it is
    # 320 and stops against the edge.
    assert found["right"] == [936, 200, 320, 300, True, False]
    assert found["rightFar"] == [936, 200, 320, 300, True, False]
    # Across the left: its left at the edge, its right where the hand has it.
    assert found["left"] == [24, 200, 326, 300, True, False]
    # Across the bottom: cut to the room under its top.
    assert found["bottom"] == [400, 600, 400, 176, False, True]
    # 76 would be under 120: 120, and stopped against the bottom.
    assert found["bottomFar"] == [400, 656, 400, 120, False, True]
    assert found["top"] == [400, 24, 400, 300, False, False], "the top is stopped, never cut"
    assert found["corner"] == [936, 600, 320, 176, True, True]
    assert found["launcher"] == [1116, 732, 140, 44, False, False], "the launcher only stops"
    # A box smaller than `least` is never made larger, only stopped.
    assert found["small"] == [1056, 676, 200, 100, False, False]


DRAG = """
// The open panel, floating, 400 wide and 300 tall at (800, 400) in a 1280 x 800
// window (gap 24): the drag machinery, real, on a stub whose size follows
// what is written on it, as a browser's would.
NS.basePath = () => "";
function dragShell(state) {
    const shell = Object.create(NS.Shell.prototype);
    const values = {};
    const panelStyle = {values};
    ["width", "maxHeight", "transform", "willChange", "left", "top"].forEach((name) => {
        Object.defineProperty(panelStyle, name, {get: () => values[name] || "",
                                                 set: (v) => { values[name] = v; }});
    });
    const own = {width: 400, height: 300};
    const panel = {style: panelStyle, classes: new Set(),
                   classList: {add(n) { panel.classes.add(n); }, remove(n) { panel.classes.delete(n); },
                               contains(n) { return panel.classes.has(n); }},
                   setPointerCapture(id) { panel.captured = id; }, releasePointerCapture() {},
                   setAttribute() {}, removeAttribute() {}, captured: null};
    Object.defineProperty(panel, "offsetWidth", {get: () => parseFloat(values.width) || own.width});
    Object.defineProperty(panel, "offsetHeight", {get: () => Math.min(own.height,
        parseFloat(values.maxHeight) || Infinity)});
    panel.getBoundingClientRect = () => ({left: 800, top: 400, width: panel.offsetWidth,
                                          height: panel.offsetHeight});
    shell.nodes = {panel, root: {classList: {add() {}, remove() {}}, appendChild() {}},
                   header: {contains: () => false}, launcher: {}};
    // The panel's own place, and the launcher's elsewhere: a drag of the
    // panel is the panel's alone.
    shell.state = Object.assign({panelOpen: true, freeFloat: true, panelAt: {x: 0.5, y: 0.5},
                                 panelWidth: 400, panelAnchor: "bottom-right",
                                 floatAt: {x: 0, y: 0}, anchorOverride: "top-left",
                                 fitWidth: null, fitHeight: null}, state || {});
    shell.settings = {defaultAnchor: "bottom-right"};
    shell.placed = 0;
    shell.placeNow = () => { shell.placed += 1; };
    shell._save = () => {};
    shell._saveFloat = () => {};
    shell.syncDock = () => { shell.synced = (shell.synced || 0) + 1; };
    shell.shiftDock = (dx, dy) => { shell.shifted = [dx, dy]; };
    shell.floatHold = () => 44;
    return {shell, panel, values, own};
}
const press = (x, y) => ({pointerId: 1, button: 0, isPrimary: true, pointerType: "mouse",
                          clientX: x, clientY: y, target: null, preventDefault() {}});
"""


def test_a_drag_past_the_right_and_the_bottom_draws_the_panel_cut_and_grows_it_back():
    found = run(DRAG + """
        const {shell, values} = dragShell();
        shell.startDrag(press(900, 410), shell.nodes.panel);
        // Right by 200: its right edge would be at 1400; held at 1256.
        shell.moveDrag(press(1100, 410));
        const right = {width: values.width, cap: values.maxHeight || "",
                       transform: values.transform, synced: shell.synced || 0};
        // And down by 300: its bottom would be at 1000; held at 776.
        shell.moveDrag(press(1100, 710));
        const corner = {width: values.width, cap: values.maxHeight, transform: values.transform};
        // Back where it started: its own size again.
        shell.moveDrag(press(900, 410));
        const back = {width: values.width, cap: values.maxHeight, transform: values.transform};
        console.log(JSON.stringify({right, corner, back}));
    """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 1280, "height": 800})

    # 1100 - 100 = 1000 would leave 256: cut to 320 and stopped at the edge;
    # the docked column, were there one, placed over its slot from now on.
    # The stub starts with nothing written on it: the first move writes the
    # window's cap once (on a page the placement has), and from then on it is
    # rewritten only when it changes -- a max-height written on every move
    # is a layout on every move.
    assert found["right"] == {"width": "320px", "cap": "752px",
                              "transform": "translate(136px, 0px)", "synced": 1}
    # Its top at 700 would leave 76 under it: 120, stopped against the bottom.
    assert found["corner"] == {"width": "320px", "cap": "120px",
                               "transform": "translate(136px, 256px)"}
    # Grown back: its own width, and the window's cap (752) rather than a cut.
    assert found["back"] == {"width": "400px", "cap": "752px", "transform": "translate(0px, 0px)"}


def test_let_go_cut_a_floating_panel_stays_where_and_as_it_was_drawn():
    """No spring back: the width and the height it was cut to are kept
    (`fitWidth`, `fitHeight`), and its place is its header's -- never
    "against the bottom", which would mean its own height grown back up.
    Let go uncut, nothing is kept. Anchored, it snaps to its corner at its
    own size."""
    found = run(DRAG + """
        const cut = dragShell();
        cut.shell.startDrag(press(900, 410), cut.shell.nodes.panel);
        cut.shell.moveDrag(press(1000, 710));
        cut.shell.endDrag(press(1000, 710), false);
        const uncut = dragShell({fitWidth: 333, fitHeight: 222});
        uncut.shell.startDrag(press(900, 410), uncut.shell.nodes.panel);
        uncut.shell.moveDrag(press(700, 300));
        uncut.shell.endDrag(press(700, 300), false);
        const anchored = dragShell({freeFloat: false});
        anchored.shell.preview = () => undefined;
        anchored.shell.startDrag(press(900, 410), anchored.shell.nodes.panel);
        anchored.shell.moveDrag(press(1100, 710));
        anchored.shell.endDrag(press(1100, 710), false);
        const launcher = (s) => [s.state.anchorOverride, s.state.floatAt];
        console.log(JSON.stringify({
            cut: [cut.shell.state.fitWidth, cut.shell.state.fitHeight, cut.shell.state.panelAt],
            uncut: [uncut.shell.state.fitWidth, uncut.shell.state.fitHeight],
            anchored: [anchored.shell.state.fitWidth, anchored.shell.state.fitHeight,
                       anchored.shell.state.panelAnchor],
            launchers: [launcher(cut.shell), launcher(uncut.shell), launcher(anchored.shell)],
        }));
    """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 1280, "height": 800})

    width, height, at = found["cut"]
    assert (width, height) == (356, 120)
    assert at["x"] == 1 and 0 < at["y"] < 1, at
    assert found["uncut"] == [None, None], "dragged inward, its own size again"
    assert found["anchored"] == [None, None, "bottom-right"]
    assert found["launchers"] == [["top-left", {"x": 0, "y": 0}]] * 3, (
        "the launcher keeps its own place")


PLACE = """
// placePanel's own arithmetic, on the PANEL stub of test_assistant_js.py.
"""


def test_the_placement_keeps_a_cut_floating_panel_as_it_was_let_go():
    from test_assistant_js import PANEL  # noqa: E402 -- the shared stub
    found = run(PANEL + """
        sized(1280, 800);
        const floating = panel({state: {freeFloat: true, panelAt: {x: 1, y: 0.5},
                                        panelWidth: 400, fitWidth: 356, fitHeight: 176}});
        floating.placeNow();
        const anchored = panel({state: {freeFloat: false, panelWidth: 400,
                                        fitWidth: 356, fitHeight: 176}});
        anchored.placeNow();
        console.log(JSON.stringify({
            floating: [floating.nodes.panel.style.width, floating.nodes.panel.style.maxHeight],
            anchored: [anchored.nodes.panel.style.width, anchored.nodes.panel.style.maxHeight],
        }));
    """, sources=("shell",))

    assert found["floating"] == ["356px", "176px"]
    assert found["anchored"] == ["400px", "752px"], "anchored, a cut means nothing"


def test_what_forgets_a_cut_and_what_takes_it_as_the_panels_own_size():
    found = run(DRAG + """
        const away = dragShell({fitWidth: 356, fitHeight: 176});
        away.shell.showOpen = () => undefined;
        away.shell.putAway(false);
        const anchored = dragShell({fitWidth: 356, fitHeight: 176});
        anchored.shell.setFreeFloat(false);
        const reset = dragShell({fitWidth: 356, fitHeight: 176});
        reset.shell.resizedOnce = () => undefined;
        reset.shell.cornerSides = () => ({x: "left", y: "top", factor: 1});
        reset.shell.resetSize();
        const adopt = dragShell({fitWidth: 356, fitHeight: 176, conversationHeight: null});
        adopt.shell.nodes.transcript = {offsetHeight: 97};
        adopt.shell.applyConversationHeight = () => undefined;
        const adopted = adopt.shell.adoptFit();
        const fits = (s) => [s.state.fitWidth, s.state.fitHeight];
        console.log(JSON.stringify({away: fits(away.shell), anchored: fits(anchored.shell),
                                    reset: fits(reset.shell), adopted,
                                    adopt: [adopt.shell.state.panelWidth,
                                            adopt.shell.state.conversationHeight].concat(fits(adopt.shell))}));
    """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 1280, "height": 800})

    assert found["away"] == [356, 176], "put away, it keeps where and as it was drawn"
    assert found["anchored"] == [None, None]
    assert found["reset"] == [None, None]
    # The corner takes what is drawn: the cut width as its own, the
    # conversation as tall as it is drawn.
    assert found["adopted"] is True
    assert found["adopt"] == [356, 97, None, None]


def test_a_press_on_a_header_button_is_the_buttons_until_it_moves():
    """The whole header is the handle: a press on one of its buttons starts a
    drag only once it moves, and only then is the pointer captured -- before,
    the press is the button's, never cancelled and never a header press. A
    press on a text box is not a handle at all."""
    found = run(DRAG + """
        const {shell, panel} = dragShell();
        const button = {tagName: "BUTTON", closest: () => button};
        const box = {tagName: "INPUT", closest: () => box};
        shell.nodes.header = {contains: (n) => n === button || n === box};
        const down = Object.assign(press(900, 410), {target: button, cancelled: false,
                                                     preventDefault() { this.cancelled = true; }});
        shell.startDrag(down, panel);
        const pressed = {drag: !!shell.drag, captured: panel.captured, cancelled: down.cancelled,
                         header: shell.headerPress};
        shell.moveDrag(press(903, 412));
        const still = panel.captured;
        shell.moveDrag(press(860, 380));
        const moved = {captured: panel.captured, transform: panel.style.transform};
        shell.endDrag(press(860, 380), false);
        const suppressed = shell.suppressHeaderClick;
        shell.startDrag(Object.assign(press(900, 410), {target: box}), panel);
        console.log(JSON.stringify({pressed, still, moved, suppressed, input: !!shell.drag}));
    """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 1280, "height": 800})

    assert found["pressed"] == {"drag": True, "captured": None, "cancelled": False, "header": False}
    assert found["still"] is None, "not yet past the threshold"
    assert found["moved"] == {"captured": 1, "transform": "translate(-40px, -30px)"}
    assert found["suppressed"] is True, "the drag's click is swallowed"
    assert found["input"] is False


def test_the_cross_shows_only_while_a_tap_outside_does_not_close_the_panel():
    found = run("""
        NS.basePath = () => "";
        globalThis.localStorage = {getItem: () => null, setItem() {}};
        const shell = Object.create(NS.Shell.prototype);
        shell.state = {tapToClose: true, panelOpen: true};
        shell.nodes = {minimize: {hidden: false}};
        shell.tell = () => undefined;
        shell.applyMinimize();
        const on = shell.nodes.minimize.hidden;
        shell.setTapToClose(false);
        const off = shell.nodes.minimize.hidden;
        console.log(JSON.stringify({on, off}));
    """, sources=("shell",))

    assert found == {"on": True, "off": False}


def test_escape_in_the_panel_puts_it_away_when_there_is_no_cross():
    """The ✕ gone, the keyboard still has a way: Escape inside the open panel
    -- after a menu's and focus mode's claims, and never outside the panel,
    where it is Forge's (it interrupts a generation)."""
    found = run("""
        const order = (context) => NS.escapeOrder({key: "Escape"}, Object.assign({
            composing: false, nativeDialog: false, hostDialogOpen: false,
            insideAssistant: true, assistantMenuOpen: false, focusActive: false,
            closesPanel: true}, context));
        console.log(JSON.stringify({
            inside: order({}),
            outside: order({insideAssistant: false}),
            cross: order({closesPanel: false}),
            menu: order({assistantMenuOpen: true}),
            focus: order({focusActive: true}),
        }));
    """, sources=("shell", "host"))

    assert found == {"inside": "close-panel", "outside": "", "cross": "",
                     "menu": "close-menu", "focus": "exit-focus"}


def test_a_panel_as_wide_as_the_window_allows_stays_put_sideways():
    """A phone's floating panel fills the room across: a finger drifting
    sideways on its way down neither moves it across nor cuts it narrower,
    which it would then keep."""
    found = run(DRAG + """
        const {shell, values} = dragShell({panelWidth: 400});
        shell.nodes.panel.getBoundingClientRect = () => ({left: 16, top: 100, width: 358,
                                                          height: shell.nodes.panel.offsetHeight});
        shell.startDrag(press(100, 110), shell.nodes.panel);
        shell.moveDrag(press(130, 300));
        shell.moveDrag(press(60, 500));
        const drawn = {width: values.width, transform: values.transform};
        shell.endDrag(press(60, 500), false);
        console.log(JSON.stringify({drawn, fit: shell.state.fitWidth}));
    """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 390, "height": 844})

    # Its own width (358, all the room there is): not cut, and not moved across.
    assert found["drawn"] == {"width": "358px", "transform": "translate(0px, 390px)"}
    assert found["fit"] is None


def test_the_corner_and_the_edge_take_a_cut_panel_at_the_size_it_is_drawn():
    """Pressed on a panel a drag cut, the corner, the edge and their keys
    resize what is on the screen: the cut is taken as the panel's own size
    first, so nothing springs back under the hand."""
    found = run(DRAG + """
        const make = () => {
            const made = dragShell({fitWidth: 356, fitHeight: 176, conversationHeight: null});
            made.shell.nodes.transcript = {offsetHeight: 97};
            made.shell.nodes.resize = {setPointerCapture() {}, releasePointerCapture() {}};
            made.shell.nodes.corner = {setPointerCapture() {}, releasePointerCapture() {}};
            made.shell.applyConversationHeight = () => undefined;
            made.shell.cornerSides = () => ({x: "left", y: "top", factor: 1});
            made.shell.conversationRange = () => ({fromHeight: 97, least: 97, most: 400});
            made.shell.heldEdges = () => null;
            made.shell.resizedOnce = () => undefined;
            return made.shell;
        };
        const state = (s) => [s.state.panelWidth, s.state.conversationHeight, s.state.fitWidth,
                              s.state.fitHeight];
        const corner = make();
        corner.startResize(press(10, 10), corner.nodes.corner);
        const edge = make();
        edge.resizeKey({key: "ArrowLeft", preventDefault() {}}, edge.nodes.resize);
        const key = make();
        key.resizeKey({key: "ArrowUp", preventDefault() {}}, key.nodes.corner);
        console.log(JSON.stringify({corner: state(corner), edge: state(edge), key: state(key)}));
    """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 1280, "height": 800})

    assert found["corner"] == [356, 97, None, None]
    assert found["edge"] == [372, 97, None, None], "taken, then a step wider"
    assert found["key"] == [356, 113, None, None], "taken, then a step taller"
