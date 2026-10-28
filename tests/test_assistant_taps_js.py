"""Quick actions on the launcher: two presses go back, three toggle focus,
four toggle Super Focused.

Asked for from use, first: with the panel put away to its launcher, a double
press should return to the workspace before this one -- and, pressed again, to
the one just left, so two workspaces can be flipped between -- and a triple
press should turn focus mode on or off. Then (docs/22 §3.39) two presses
became a cycle of Normal, Focused and Super Focused, which took going back
away; and then (§3.40): "I noticed that doube tap flyout to switch to last
tab no longer works ... Let's make double tap switch to last tab, triple tap
open focus and 4 taps open super super focus." So two presses go back once
the window has passed, three turn focus on inside the third press (for the
browser's full screen) or, with focus on, turn it off once the window has
passed -- a fourth may still come -- and four act inside the fourth
(`pressFour`, held in test_assistant_modes_js.py). None should need a pace
that is hard to hit or one that turns two unhurried presses into a gesture.

Timers are a fake clock here: the harness's own `setTimeout` never fires, and
the whole point of these tests is what happens when the window passes.
"""

from __future__ import annotations

import re

from test_assistant_js import SHELL, run


CLOCK = """
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push({fn, ms, live: true}); return timers.length; };
globalThis.clearTimeout = (id) => { if (timers[id - 1]) timers[id - 1].live = false; };
const lapse = () => {
    const due = timers.filter((timer) => timer.live);
    due.forEach((timer) => { timer.live = false; timer.fn(); });
    return due.length;
};

function tapShell(focused) {
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {panelOpen: false, focusEnabled: !!focused};
    shell.nodes = {};
    shell.did = [];
    shell.focus = {on: !!focused, isActive() { return this.on; }};
    shell.open = () => shell.did.push("open");
    shell.toggleFocus = () => {
        shell.focus.on = !shell.focus.on;
        shell.did.push(shell.focus.on ? "focus" : "unfocus");
    };
    shell.pressFour = () => shell.did.push("four");
    shell.switchWorkspace = (id) => { shell.did.push("switch:" + id); return Promise.resolve(id); };
    shell.fault = (error) => shell.did.push("fault:" + error.message);
    shell.host = {active: "txt2img", getActiveWorkspace() { return this.active; }};
    return shell;
}
const press = (detail) => ({detail: detail === undefined ? 1 : detail, preventDefault() {}});
"""


class TestCountingPresses:
    def test_one_press_opens_the_panel_once_the_window_has_passed(self):
        found = run(CLOCK + """
            const shell = tapShell();
            shell.tapLauncher(press());
            const before = shell.did.slice();
            lapse();
            console.log(JSON.stringify({before, after: shell.did}));
        """)

        assert found == {"before": [], "after": ["open"]}

    def test_two_presses_go_back_once_the_window_has_passed_and_do_not_open(self):
        """"Double tap switch to last tab": not inside the second press,
        because a third may still come."""
        found = run(CLOCK + """
            const shell = tapShell();
            shell.previousWorkspace = "img2img";
            shell.tapLauncher(press());
            shell.tapLauncher(press());
            const during = shell.did.slice();
            lapse();
            console.log(JSON.stringify({during, after: shell.did}));
        """)

        assert found == {"during": [], "after": ["switch:img2img"]}

    def test_three_presses_turn_focus_on_inside_the_third_press(self):
        """Entering focus asks for the browser's full screen, which a browser
        grants only while a press is being handled -- so the third press acts
        itself, not a timer after it; the window's end does nothing more."""
        found = run(CLOCK + """
            const shell = tapShell();
            shell.previousWorkspace = "img2img";
            [1, 2, 3].forEach(() => shell.tapLauncher(press()));
            const during = shell.did.slice();
            lapse();
            console.log(JSON.stringify({during, after: shell.did}));
        """)

        assert found == {"during": ["focus"], "after": ["focus"]}

    def test_three_presses_turn_focus_off_once_the_window_has_passed(self):
        """With focus on, a fourth press may still come -- four mean Super
        Focused, not out -- so leaving waits for the window; and leaving the
        browser's full screen needs no press to be handled."""
        found = run(CLOCK + """
            const shell = tapShell(true);
            [1, 2, 3].forEach(() => shell.tapLauncher(press()));
            const during = shell.did.slice();
            lapse();
            console.log(JSON.stringify({during, after: shell.did}));
        """)

        assert found == {"during": [], "after": ["unfocus"]}

    def test_four_presses_act_inside_the_fourth(self):
        """From Normal: focus on at the third, the fourth on top of it."""
        found = run(CLOCK + """
            const shell = tapShell();
            [1, 2, 3, 4].forEach(() => shell.tapLauncher(press()));
            const during = shell.did.slice();
            const fired = lapse();
            console.log(JSON.stringify({during, fired, after: shell.did}));
        """)

        assert found == {"during": ["focus", "four"], "fired": 0, "after": ["focus", "four"]}

    def test_four_presses_in_focus_never_leave_it(self):
        """The third press's leaving waited for the window, and the fourth
        came inside it."""
        found = run(CLOCK + """
            const shell = tapShell(true);
            [1, 2, 3, 4].forEach(() => shell.tapLauncher(press()));
            const fired = lapse();
            console.log(JSON.stringify({fired, did: shell.did, focused: shell.focus.on}));
        """)

        assert found == {"fired": 0, "did": ["four"], "focused": True}

    def test_a_fifth_press_starts_a_new_count(self):
        """Four acted; a fifth is the first of the next gesture, and alone it
        is one press: the panel opens once the window has passed."""
        found = run(CLOCK + """
            const shell = tapShell(true);
            [1, 2, 3, 4, 5].forEach(() => shell.tapLauncher(press()));
            const during = shell.did.slice();
            lapse();
            console.log(JSON.stringify({during, after: shell.did}));
        """)

        assert found == {"during": ["four"], "after": ["four", "open"]}

    def test_presses_further_apart_than_the_window_are_separate(self):
        found = run(CLOCK + """
            const shell = tapShell();
            shell.previousWorkspace = "img2img";
            shell.tapLauncher(press());
            lapse();
            shell.tapLauncher(press());
            lapse();
            console.log(JSON.stringify({did: shell.did}));
        """)

        assert found == {"did": ["open", "open"]}

    def test_a_keyboard_press_opens_at_once(self):
        """Enter or Space on the launcher is a click with detail 0: one press,
        and nobody pressing a key expects to wait for another."""
        found = run(CLOCK + """
            const shell = tapShell();
            shell.tapLauncher(press(0));
            console.log(JSON.stringify({did: shell.did, pending: timers.filter((t) => t.live).length}));
        """)

        assert found == {"did": ["open"], "pending": 0}

    def test_the_window_is_a_double_clicks_pace(self):
        """Not so short a quick pair is missed, not so long that a single press
        feels slow or two unhurried ones become a gesture."""
        found = run(CLOCK + """
            const shell = tapShell();
            shell.tapLauncher(press());
            console.log(JSON.stringify({ms: timers[0].ms}));
        """)

        assert 250 <= found["ms"] <= 400

    def test_a_failing_action_is_contained(self):
        found = run(CLOCK + """
            const shell = tapShell();
            shell.open = () => { throw new Error("boom"); };
            shell.tapLauncher(press());
            lapse();
            console.log(JSON.stringify({did: shell.did, taps: shell.taps}));
        """)

        assert found == {"did": ["fault:boom"], "taps": 0}


class TestWhatEndsACount:
    def test_a_drag_between_presses_starts_the_count_again(self):
        """The click a launcher drag leaves behind is swallowed, and it takes
        any half-counted gesture with it: press, drag, press is one press."""
        found = run(CLOCK + """
            const shell = tapShell();
            shell.previousWorkspace = "img2img";
            const click = (event) => shell.launcherClick(event);
            click(press());
            shell.suppressClick = true;
            click(press());
            click(press());
            lapse();
            console.log(JSON.stringify({did: shell.did}));
        """)

        assert found == {"did": ["open"]}

    def test_leaving_the_window_drops_a_half_counted_gesture(self):
        found = run(CLOCK + """
            const shell = tapShell();
            shell.tapLauncher(press());
            shell.cancelGestures();
            const fired = lapse();
            console.log(JSON.stringify({fired, did: shell.did, taps: shell.taps}));
        """)

        assert found == {"fired": 0, "did": [], "taps": 0}

    def test_disposing_the_shell_cancels_a_pending_press(self):
        found = run(CLOCK + """
            const shell = tapShell();
            shell.disposers = [];
            shell.tapLauncher(press());
            shell.dispose();
            console.log(JSON.stringify({fired: lapse(), did: shell.did}));
        """)

        assert found == {"fired": 0, "did": []}


class TestBackAndForth:
    SETUP = CLOCK + """
        const shell = tapShell();
        const go = (id) => { shell.host.active = id; shell.noteWorkspace(id); };
    """

    def test_the_workspace_left_is_remembered(self):
        found = run(self.SETUP + """
            go("txt2img");
            const first = shell.previousWorkspace || null;
            go("img2img");
            console.log(JSON.stringify({first, then: shell.previousWorkspace}))
        """)

        assert found == {"first": None, "then": "txt2img"}

    def test_pressing_twice_again_returns_to_where_it_came_from(self):
        """Back is a toggle: from B back to A, then from A back to B."""
        found = run(self.SETUP + """
            shell.switchWorkspace = (id) => { shell.did.push(id); go(id); return Promise.resolve(id); };
            go("txt2img");
            go("wangp");
            shell.backToPrevious();
            shell.backToPrevious();
            shell.backToPrevious();
            console.log(JSON.stringify({did: shell.did}));
        """)

        assert found == {"did": ["txt2img", "wangp", "txt2img"]}

    def test_a_blank_read_between_two_workspaces_is_not_somewhere(self):
        found = run(self.SETUP + """
            go("txt2img");
            go("");
            go("img2img");
            console.log(JSON.stringify({back: shell.previousWorkspace}));
        """)

        assert found == {"back": "txt2img"}

    def test_with_nowhere_to_go_back_to_nothing_happens(self):
        found = run(self.SETUP + """
            go("txt2img");
            shell.backToPrevious();
            console.log(JSON.stringify({did: shell.did}));
        """)

        assert found == {"did": []}


def test_the_launcher_and_the_navigation_feed_are_wired_to_them():
    shell = SHELL.read_text(encoding="utf-8")
    wiring = shell.split("Shell.prototype.wire = function", 1)[1] \
        .split("Shell.prototype.grow = function", 1)[0]

    assert re.search(r'this\.on\(nodes\.launcher, "click", \(event\) => this\.launcherClick\(event\)\)',
                     wiring)
    assert "subscribeNavigation((active) => this.navigated(active))" in wiring
    subscriber = shell.split("Shell.prototype.navigated = function (active) {", 1)[1]
    assert subscriber.lstrip(" \n").startswith("this.noteWorkspace(active);")


# --------------------------------------------------------------------------- #
# Presses on the header's own space
# --------------------------------------------------------------------------- #

HEADER = CLOCK + """
// The panel, its header and the space between the header's buttons -- and the
// presses as a browser delivers them: pointerdown on what was pressed, the
// drag machinery taking pointer capture on the panel, and the click landing
// on the PANEL, which is where a captured pointer's click goes (the common
// ancestor of where the pointer went down and where it came up).
const header = {id: "header"};
const panel = {id: "panel", style: {}, releasePointerCapture() {},
               getBoundingClientRect: () => ({left: 0, top: 0, width: 360, height: 400}),
               setPointerCapture() { panel.captured = true; }};
const control = {closest: () => control};
const space = {closest: () => null};
const elsewhere = {closest: () => null};
function headerShell() {
    const shell = tapShell();
    shell.state = {panelOpen: true, focusEnabled: false, freeFloat: false};
    shell.nodes = {header, panel, root: {classList: {remove() {}, add() {}}, appendChild() {}}};
    shell.anchor = () => "top-left";
    shell._save = () => {};
    shell.placeNow = () => {};
    shell.noticeCover = () => {};
    shell.fault = (error) => shell.did.push("fault:" + error.message);
    return shell;
}
// One press, start to finish: down on `target` (through the window's capture
// listener, then the header's own if the press is in the header), up, click.
let pointerId = 0;
function tap(shell, target, options) {
    options = options || {};
    pointerId += 1;
    const down = {button: 0, isPrimary: true, pointerId, clientX: 10, clientY: 10, target};
    shell.supersede(down);
    if (target !== elsewhere) shell.startDrag(down, panel);
    if (options.moved) {
        shell.moveDrag({pointerId, pointerType: "touch", clientX: 60, clientY: 10});
    }
    shell.endDrag({pointerId}, false);
    shell.headerClick({detail: options.detail === undefined ? 1 : options.detail,
                       target: panel});
}
"""


class TestPressesOnTheHeader:
    """Asked for with the Focus button's removal: "triple tapping on the drag
    area to be the toggle focus on / off ... just like the launcher's triple
    tap" -- and then, once it did not work in a real browser: "lets bring the
    double tap to that area. So double tap to last tab, and triple tap to
    toggle focus. Just like the button!" -- and since docs/22 §3.40, like the
    button again: two presses back, three focus, four Super Focused.

    The drag area is the header's own space -- the grip and the padding
    between its buttons. The first version listened for clicks on the header
    and never heard one: the drag takes pointer capture on the panel, and a
    captured pointer's click lands on the panel. These presses go down where
    the browser puts them and come up where capture puts them.
    """

    def test_two_presses_on_the_space_go_back_once_the_window_has_passed(self):
        found = run(HEADER + """
            const shell = headerShell();
            shell.previousWorkspace = "img2img";
            tap(shell, space);
            tap(shell, space);
            const during = shell.did.slice();
            lapse();
            console.log(JSON.stringify({during, after: shell.did, captured: !!panel.captured}));
        """)

        assert found == {"during": [], "after": ["switch:img2img"], "captured": True}

    def test_three_presses_on_the_space_turn_focus_on_inside_the_third(self):
        found = run(HEADER + """
            const shell = headerShell();
            shell.previousWorkspace = "img2img";
            [1, 2, 3].forEach(() => tap(shell, space));
            const during = shell.did.slice();
            lapse();
            console.log(JSON.stringify({during, after: shell.did}));
        """)

        assert found == {"during": ["focus"], "after": ["focus"]}

    def test_four_presses_on_the_space_act_inside_the_fourth(self):
        found = run(HEADER + """
            const shell = headerShell();
            shell.focus.on = true;
            [1, 2, 3, 4].forEach(() => tap(shell, space));
            const during = shell.did.slice();
            const fired = lapse();
            console.log(JSON.stringify({during, fired}));
        """)

        assert found == {"during": ["four"], "fired": 0}

    def test_one_press_puts_the_panel_away_and_is_forgotten(self):
        """Asked for: "when fly out is open, let's make single tap of the drag
        handle perform the minimize as well". It was nothing; it is the panel
        put away now, once the window has passed -- and still forgotten: the
        lone press before the window is not the first of two or three."""
        found = run(HEADER + """
            const shell = headerShell();
            shell.putAway = (focus) => shell.did.push("away:" + focus);
            shell.previousWorkspace = "img2img";
            tap(shell, space);
            lapse();
            tap(shell, space);
            const after = shell.did.slice();
            tap(shell, space);
            lapse();
            console.log(JSON.stringify({after, then: shell.did}));
        """)

        assert found == {"after": ["away:true"], "then": ["away:true", "switch:img2img"]}, \
            "the lone press before the window is not the first of three"

    def test_the_click_is_read_where_the_captured_pointer_puts_it(self):
        """On the panel, not the header: the listener is the panel's, and a
        click that comes with no press on the header's space behind it -- a
        press on a message, say -- is not counted."""
        found = run(HEADER + """
            const shell = headerShell();
            tap(shell, elsewhere);
            tap(shell, elsewhere);
            tap(shell, elsewhere);
            console.log(JSON.stringify({did: shell.did, count: shell.headerTaps || 0}));
        """)

        assert found == {"did": [], "count": 0}
        wiring = SHELL.read_text(encoding="utf-8") \
            .split("Shell.prototype.wire = function", 1)[1] \
            .split("Shell.prototype.grow = function", 1)[0]
        assert 'this.on(nodes.panel, "click"' in wiring
        assert 'this.on(nodes.header, "click"' not in wiring

    def test_a_press_that_never_became_a_click_is_forgotten_at_the_next_press(self):
        """A touch the browser took for a scroll: pointerdown on the space,
        then pointercancel and no click. Without the clearing, the next press
        anywhere in the panel -- on a message, say -- would be counted as a
        press on the space."""
        found = run(HEADER + """
            const shell = headerShell();
            const down = {button: 0, isPrimary: true, pointerId: 7, clientX: 1, clientY: 1, target: space};
            shell.supersede(down);
            shell.startDrag(down, panel);
            shell.endDrag({pointerId: 7}, true);
            const noted = shell.headerPress;
            tap(shell, elsewhere);
            tap(shell, elsewhere);
            tap(shell, elsewhere);
            console.log(JSON.stringify({noted, did: shell.did, count: shell.headerTaps || 0}));
        """)

        assert found == {"noted": True, "did": [], "count": 0}

    def test_the_pace_is_the_launcher_s(self):
        source = SHELL.read_text(encoding="utf-8")
        tap = source.split("Shell.prototype.tapHeader = function", 1)[1] \
            .split("Shell.prototype", 1)[0]

        assert "TAP_WINDOW" in tap
        assert "pressThree" in tap and "pressFour" in tap and "afterPresses" in tap
        assert "cycleMode" not in source, "the two-press cycle is gone"

    def test_a_press_on_a_button_in_the_row_is_that_button_s(self):
        """It never starts a drag, so it is never a press on the space; and it
        neither counts nor breaks a count."""
        found = run(HEADER + """
            const shell = headerShell();
            shell.previousWorkspace = "img2img";
            [1, 2, 3].forEach(() => tap(shell, control));
            const onControls = shell.did.slice();
            tap(shell, space);
            tap(shell, control);
            tap(shell, space);
            lapse();
            console.log(JSON.stringify({onControls, then: shell.did}));
        """)

        assert found == {"onControls": [], "then": ["switch:img2img"]}

    def test_a_synthetic_click_is_not_a_press(self):
        """`detail` is 0 for a keyboard activation, and a header cannot be
        activated from the keyboard, so a click with none is nobody's."""
        found = run(HEADER + """
            const shell = headerShell();
            [1, 2, 3].forEach(() => tap(shell, space, {detail: 0}));
            console.log(JSON.stringify(shell.did));
        """)

        assert found == []

    def test_the_tail_of_a_drag_is_swallowed_and_ends_the_count(self):
        found = run(HEADER + """
            const shell = headerShell();
            shell.previousWorkspace = "img2img";
            tap(shell, space);
            tap(shell, space, {moved: true});
            const afterDrag = {did: shell.did.slice(), flag: shell.suppressHeaderClick,
                               count: shell.headerTaps || 0, at: shell.state.anchorOverride};
            [1, 2].forEach(() => tap(shell, space));
            lapse();
            console.log(JSON.stringify({afterDrag, then: shell.did}));
        """)

        assert found["afterDrag"] == {"did": [], "flag": False, "count": 0, "at": "top-left"}, (
            "the drag moved the panel and was not the second press; the flag is spent")
        assert found["then"] == ["switch:img2img"]

    def test_a_new_press_clears_a_tail_that_never_came(self):
        """With the pointer captured the click may not come at all, so a flag
        left for it would eat the next real press."""
        found = run(HEADER + """
            const shell = headerShell();
            shell.suppressHeaderClick = true;
            const down = {button: 0, isPrimary: true, pointerId: 9, clientX: 1, clientY: 1, target: space};
            shell.startDrag(down, panel);
            console.log(JSON.stringify({flag: shell.suppressHeaderClick, dragging: !!shell.drag,
                                        pressed: shell.headerPress}));
        """)

        assert found == {"flag": False, "dragging": True, "pressed": True}

    def test_a_cancelled_gesture_forgets_the_count(self):
        found = run(HEADER + """
            const shell = headerShell();
            shell.drag = null; shell.strip = null; shell.resizing = null;
            tap(shell, space);
            shell.cancelGestures();
            tap(shell, space);
            console.log(JSON.stringify({did: shell.did, count: shell.headerTaps}));
        """)

        assert found == {"did": [], "count": 1}

    def test_there_is_no_focus_button_left_to_press(self):
        source = SHELL.read_text(encoding="utf-8")

        assert "focusToggle" not in source
        assert re.search(r'"Focus"\)', source) is None, "no button is made with that word"
