"""Quick actions on the launcher: two presses go back, three toggle focus.

Asked for from use: with the panel put away to its launcher, a double press
should return to the workspace before this one -- and, pressed again, to the
one just left, so two workspaces can be flipped between -- and a triple press
should turn focus mode on or off. Neither should need a pace that is hard to
hit or one that turns two unhurried presses into a gesture.

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

function tapShell() {
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {panelOpen: false, focusEnabled: false};
    shell.nodes = {};
    shell.did = [];
    shell.open = () => shell.did.push("open");
    shell.toggleFocus = () => shell.did.push("focus");
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

    def test_two_presses_go_back_and_do_not_open(self):
        found = run(CLOCK + """
            const shell = tapShell();
            shell.previousWorkspace = "img2img";
            shell.tapLauncher(press());
            shell.tapLauncher(press());
            lapse();
            console.log(JSON.stringify({did: shell.did}));
        """)

        assert found == {"did": ["switch:img2img"]}

    def test_three_presses_toggle_focus_inside_the_third_press(self):
        """Entering focus asks for the browser's full screen, which a browser
        grants only while a press is being handled -- so the third press acts
        itself, not a timer after it."""
        found = run(CLOCK + """
            const shell = tapShell();
            shell.previousWorkspace = "img2img";
            shell.tapLauncher(press());
            shell.tapLauncher(press());
            shell.tapLauncher(press());
            const during = shell.did.slice();
            const fired = lapse();
            console.log(JSON.stringify({during, fired, after: shell.did}));
        """)

        assert found == {"during": ["focus"], "fired": 0, "after": ["focus"]}

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
    subscriber = wiring.split("subscribeNavigation((active)", 1)[1]
    assert subscriber.lstrip(" =>{\n").startswith("this.noteWorkspace(active);")


# --------------------------------------------------------------------------- #
# Presses on the header's own space
# --------------------------------------------------------------------------- #

HEADER = CLOCK + """
// The header, a control in it, and the space between the controls.
const header = {id: "header"};
const control = {closest: () => control};
const space = {closest: () => null};
function headerShell() {
    const shell = tapShell();
    shell.nodes = {header, panel: {dataset: {}}};
    return shell;
}
const on = (target, detail) => ({detail: detail === undefined ? 1 : detail,
                                 target: target === undefined ? space : target});
"""


class TestPressesOnTheHeader:
    """Asked for with the Focus button's removal: "triple tapping on the drag
    area to be the toggle focus on / off ... just like the launcher's triple
    tap." The drag area is the header's own space -- the grip and the padding
    between its buttons -- and a press on one of the buttons is that button's.
    """

    def test_three_presses_on_the_space_toggle_focus_inside_the_third(self):
        found = run(HEADER + """
            const shell = headerShell();
            shell.headerClick(on());
            shell.headerClick(on());
            const before = shell.did.slice();
            shell.headerClick(on());
            const during = shell.did.slice();
            const fired = lapse();
            console.log(JSON.stringify({before, during, fired, after: shell.did}));
        """)

        assert found == {"before": [], "during": ["focus"], "fired": 0, "after": ["focus"]}

    def test_a_press_on_the_header_itself_counts_as_the_space(self):
        found = run(HEADER + """
            const shell = headerShell();
            [1, 2, 3].forEach(() => shell.headerClick(on(header)));
            console.log(JSON.stringify(shell.did));
        """)

        assert found == ["focus"]

    def test_one_or_two_do_nothing_and_are_forgotten_when_the_window_passes(self):
        """Nothing waits on the window here -- the header has no single-press
        action -- so a pair that is forgotten is simply two presses that were
        not three."""
        found = run(HEADER + """
            const shell = headerShell();
            shell.headerClick(on());
            shell.headerClick(on());
            const fired = lapse();
            shell.headerClick(on());
            shell.headerClick(on());
            const afterFour = shell.did.slice();
            shell.headerClick(on());
            console.log(JSON.stringify({fired, afterFour, then: shell.did}));
        """)

        assert found == {"fired": 1, "afterFour": [], "then": ["focus"]}

    def test_the_pace_is_the_launcher_s(self):
        source = SHELL.read_text(encoding="utf-8")
        tap = source.split("Shell.prototype.tapHeader = function", 1)[1] \
            .split("Shell.prototype", 1)[0]

        assert "TAP_WINDOW" in tap

    def test_a_press_on_a_button_in_the_row_is_that_button_s(self):
        found = run(HEADER + """
            const shell = headerShell();
            [1, 2, 3].forEach(() => shell.headerClick(on(control)));
            const onControls = shell.did.slice();
            shell.headerClick(on(control));
            shell.headerClick(on());
            shell.headerClick(on());
            shell.headerClick(on());
            console.log(JSON.stringify({onControls, then: shell.did}));
        """)

        assert found == {"onControls": [], "then": ["focus"]}, (
            "a press on a control neither counts nor breaks a count")

    def test_a_synthetic_click_is_not_a_press(self):
        """`detail` is 0 for a keyboard activation, and a header cannot be
        activated from the keyboard, so a click with none is nobody's."""
        found = run(HEADER + """
            const shell = headerShell();
            [1, 2, 3].forEach(() => shell.headerClick(on(space, 0)));
            console.log(JSON.stringify(shell.did));
        """)

        assert found == []

    def test_the_tail_of_a_drag_is_swallowed_once(self):
        found = run(HEADER + """
            const shell = headerShell();
            shell.headerClick(on());
            shell.headerClick(on());
            shell.suppressHeaderClick = true;
            shell.headerClick(on());
            const swallowed = {did: shell.did.slice(), flag: shell.suppressHeaderClick,
                               count: shell.headerTaps};
            [1, 2, 3].forEach(() => shell.headerClick(on()));
            console.log(JSON.stringify({swallowed, then: shell.did}));
        """)

        assert found["swallowed"] == {"did": [], "flag": False, "count": 0}, (
            "the drag ends the count it interrupted, and the flag is spent")
        assert found["then"] == ["focus"]

    def test_a_drag_of_the_panel_by_its_header_leaves_that_tail_to_swallow(self):
        found = run(HEADER + """
            const shell = headerShell();
            const panel = {style: {}, releasePointerCapture() {}};
            shell.nodes = {header, panel, root: {classList: {remove() {}}}};
            shell.state = {freeFloat: false};
            shell._save = () => {};
            shell.placeNow = () => {};
            shell.drag = {node: panel, pointerId: 1, moved: true, anchor: "top-left"};
            shell.endDrag({pointerId: 1}, false);
            const afterPanel = shell.suppressHeaderClick;
            const launcher = {style: {}, releasePointerCapture() {}};
            shell.nodes.launcher = launcher;
            shell.suppressHeaderClick = false;
            shell.drag = {node: launcher, pointerId: 2, moved: true, anchor: "top-left"};
            shell.endDrag({pointerId: 2}, false);
            console.log(JSON.stringify({afterPanel, afterLauncher: shell.suppressHeaderClick,
                                        launcherTail: shell.suppressClick}));
        """)

        assert found == {"afterPanel": True, "afterLauncher": False, "launcherTail": True}

    def test_a_new_press_clears_a_tail_that_never_came(self):
        """With the pointer captured the click may land on the panel instead
        of the header, so a flag left for it would eat the next real press."""
        found = run(HEADER + """
            const shell = headerShell();
            shell.suppressHeaderClick = true;
            const node = {getBoundingClientRect: () => ({left: 0, top: 0, width: 10, height: 10}),
                          setPointerCapture() {}};
            shell.anchor = () => "top-left";
            shell.startDrag({button: 0, isPrimary: true, pointerId: 1, clientX: 1, clientY: 1,
                             target: space}, node);
            console.log(JSON.stringify({flag: shell.suppressHeaderClick, dragging: !!shell.drag}));
        """)

        assert found == {"flag": False, "dragging": True}

    def test_a_cancelled_gesture_forgets_the_count(self):
        found = run(HEADER + """
            const shell = headerShell();
            shell.nodes.root = {classList: {remove() {}}};
            shell.drag = null; shell.strip = null; shell.resizing = null;
            shell.headerClick(on());
            shell.headerClick(on());
            shell.cancelGestures();
            shell.headerClick(on());
            console.log(JSON.stringify({did: shell.did, count: shell.headerTaps}));
        """)

        assert found == {"did": [], "count": 1}

    def test_there_is_no_focus_button_left_to_press(self):
        source = SHELL.read_text(encoding="utf-8")

        assert "focusToggle" not in source
        assert re.search(r'"Focus"\)', source) is None, "no button is made with that word"
