"""A tap outside the open panel puts it away.

Asked for: "if the flyout menu is open, tapping anywhere outside of it will
close it. It should close back to the button state. Additionally, the location
of the flyout button when I tap out to close should be the closest docked
location. The docked locations are those when free float is off, there are six
of them. Dragging or scrolling outside of the open fly out should have no
impact, only tapping / clicking outside of it should make it minimize to the
flyout button. Make the new tap to close function enabled / disabled via a new
"..." menu entry".

And since: "I want the first tap outside of an open flyout menu to do nothing
except close the flyout menu. Essentially, catch all the taps and make flyout
menu minimize action the priority. This means I have to tap twice to do
something outside of an open flyout menu. Drag scroll should still work
though." (docs/22 §3.45).

These hold the rules under node: what is a tap and what is not, what is
outside, that the launcher and the panel stay put, the switch, and what of a
press is taken from the page and what is left to it. That real taps, drags,
wheels, touch scrolls and a frame's taps are told apart in a browser, and
that the first tap presses nothing while a scroll still scrolls, is
test_assistant_layout.py's.
"""

from __future__ import annotations

from test_assistant_js import run


WORLD = """
// A shell with a root, a panel the size and place a test chooses, and the
// open and close the tap uses; a page node outside the root, a docked column
// and a dialog, each with the `closest` the rule reads.
if (!NS.basePath) NS.basePath = () => "";
const stored = {};
globalThis.localStorage = {getItem: (k) => stored[k] === undefined ? null : stored[k],
                           setItem: (k, v) => { stored[k] = String(v); }};
function node(matches) {
    return {nodeType: 1, parentElement: null,
            closest: (selector) => (matches && selector.split(", ").some(
                (part) => matches.indexOf(part) !== -1) ? {} : null)};
}
const page = node(null);
const column = node([".forge-assistant-docked-column"]);
const dialog = node(["dialog"]);
function tapShell(state) {
    const shell = Object.create(NS.Shell.prototype);
    const inside = node(null);
    const root = {contains: (n) => n === inside};
    const panel = {getBoundingClientRect: () => ({left: 1100, top: 300, right: 1460, bottom: 760,
                                                  width: 360, height: 460})};
    shell.nodes = {root, panel, inside};
    shell.state = Object.assign({panelOpen: true, tapToClose: true, freeFloat: false,
                                 floatAt: null, anchorOverride: "bottom-right",
                                 panelAt: null, panelAnchor: "bottom-right"}, state || {});
    shell.settings = {defaultAnchor: "bottom-right"};
    shell.closed = 0;
    shell.focused = 0;
    shell.showOpen = (open) => { if (!open) { shell.closed += 1; shell.state.panelOpen = false; } };
    shell.nodes.launcher = {focus: () => { shell.focused += 1; }};
    shell.floatSaved = 0;
    shell._saveFloat = () => { shell.floatSaved += 1; };
    return shell;
}
const at = (target, x, y, more) => Object.assign({target, clientX: x, clientY: y, isTrusted: true,
                                                   isPrimary: true, button: 0}, more || {});
"""


def test_a_tap_outside_puts_the_panel_away_and_leaves_focus_elsewhere_alone():
    """Focus that was not in the panel stays where it was: the tap, taken
    from the page, gave it to nothing."""
    found = run(WORLD + """
        const shell = tapShell();
        const noted = shell.notePressOutside(at(page, 200, 200));
        const closed = shell.tapOutside(at(page, 210, 195));
        console.log(JSON.stringify({noted, closed, count: shell.closed, focused: shell.focused,
                                    float: shell.state.floatAt, anchor: shell.state.anchorOverride}));
    """)

    assert found == {"noted": True, "closed": True, "count": 1, "focused": 0,
                     "float": None, "anchor": "bottom-right"}


def test_what_is_not_a_tap_outside():
    """A drag that strays, a press inside let go outside, a second finger, a
    button other than the first, the page's own clicks, a key between the
    press and the click, a press the browser cancelled (a touch it took for a
    scroll), a click in another document than the press, a click that comes
    off over the panel, and anything at all while the switch is off or the
    panel closed."""
    found = run(WORLD + """
        const tries = {};
        const attempt = (name, press, click, prepare, state) => {
            const shell = tapShell(state);
            if (prepare) prepare(shell, "before");
            if (press) shell.notePressOutside(press[0], press[1]);
            if (prepare) prepare(shell, "between");
            tries[name] = shell.tapOutside(click[0], click[1]) || shell.closed > 0;
        };
        attempt("tap", [at(page, 200, 200)], [at(page, 200, 200)]);
        attempt("drag", [at(page, 200, 200)], [at(page, 217, 200)]);
        attempt("near", [at(page, 200, 200)], [at(page, 216, 184)]);
        {
            const shell = tapShell();
            shell.notePressOutside(at(shell.nodes.inside, 200, 200));
            tries["from inside"] = shell.tapOutside(at(page, 200, 200)) || shell.closed > 0;
        }
        attempt("second finger", [at(page, 200, 200, {isPrimary: false})], [at(page, 200, 200)]);
        attempt("other button", [at(page, 200, 200, {button: 2})], [at(page, 200, 200)]);
        attempt("scripted press", [at(page, 200, 200, {isTrusted: false})], [at(page, 200, 200)]);
        attempt("scripted click", [at(page, 200, 200)], [at(page, 200, 200, {isTrusted: false})]);
        attempt("no press", null, [at(page, 200, 200)]);
        attempt("cancelled", [at(page, 200, 200)], [at(page, 200, 200)],
                (shell, when) => { if (when === "between") shell.outsidePress = null; });
        attempt("other document", [at(page, 200, 200), {}], [at(page, 200, 200)]);
        attempt("same frame", [at(page, 200, 200), "f"], [at(page, 200, 200), "f"]);
        attempt("ends over the panel", [at(page, 1095, 400)], [at(page, 1105, 400)]);
        attempt("switch off", [at(page, 200, 200)], [at(page, 200, 200)], null, {tapToClose: false});
        attempt("closed", [at(page, 200, 200)], [at(page, 200, 200)], null, {panelOpen: false});
        console.log(JSON.stringify(tries));
    """)

    assert found == {"tap": True, "drag": False, "near": True, "from inside": False,
                     "second finger": False, "other button": False, "scripted press": False,
                     "scripted click": False, "no press": False, "cancelled": False,
                     "other document": False, "same frame": True,
                     "ends over the panel": False, "switch off": False, "closed": False}


def test_the_panel_its_docked_column_and_a_dialog_are_not_outside():
    found = run(WORLD + """
        const shell = tapShell();
        const tap = (target) => { shell.notePressOutside(at(target, 50, 50));
                                  return shell.tapOutside(at(target, 50, 50)); };
        console.log(JSON.stringify({panel: tap(shell.nodes.inside), column: tap(column),
                                    dialog: tap(dialog), page: tap(page)}));
    """)

    assert found == {"panel": False, "column": False, "dialog": False, "page": True}


def test_putting_the_panel_away_moves_neither_it_nor_the_launcher():
    """The launcher and the open panel each keep their own place, and putting
    the panel away -- a tap outside, one tap on the header -- moves neither:
    the launcher shows at its own, the panel keeps its own, and the size a
    drag to an edge cut it to, for the next time it opens. Asked for: "if I
    have the button in the upper right, and I have the open state in the
    lower left, closing returns the button in the upper right." Until then a
    put-away panel took the launcher to the resting place nearest itself
    (docs/22 §3.44)."""
    found = run(WORLD + """
        const places = {};
        const putAway = (state, how) => {
            const shell = tapShell(state);
            shell.nodes.panel.getBoundingClientRect = () => ({left: 40, top: 300, right: 400,
                                                              bottom: 760, width: 360, height: 460});
            if (how === "header") shell.putAway(true);
            else {
                shell.notePressOutside(at(page, 1200, 40));
                shell.tapOutside(at(page, 1200, 40));
            }
            const s = shell.state;
            return {closed: shell.closed, launcher: [s.anchorOverride, s.floatAt],
                    panel: [s.panelAnchor, s.panelAt], cut: [s.fitWidth, s.fitHeight],
                    saved: shell.floatSaved};
        };
        const floating = {freeFloat: true, floatAt: {x: 1, y: 0}, panelAt: {x: 0, y: 1},
                          anchorOverride: "top-right", panelAnchor: "bottom-left",
                          fitWidth: 330, fitHeight: 400};
        const anchored = {freeFloat: false, floatAt: null, panelAt: null,
                          anchorOverride: "top-right", panelAnchor: "bottom-left",
                          fitWidth: null, fitHeight: null};
        places.floatingTap = putAway(floating, "tap");
        places.floatingHeader = putAway(floating, "header");
        places.anchoredTap = putAway(anchored, "tap");
        places.anchoredHeader = putAway(anchored, "header");
        console.log(JSON.stringify(places));
    """)

    for how in ("floatingTap", "floatingHeader"):
        assert found[how] == {"closed": 1, "launcher": ["top-right", {"x": 1, "y": 0}],
                              "panel": ["bottom-left", {"x": 0, "y": 1}], "cut": [330, 400],
                              "saved": 0}, how
    for how in ("anchoredTap", "anchoredHeader"):
        assert found[how] == {"closed": 1, "launcher": ["top-right", None],
                              "panel": ["bottom-left", None], "cut": [None, None],
                              "saved": 0}, how


def test_the_switch_is_in_the_panel_group_and_remembered_in_this_browser():
    found = run(WORLD + """
        const shell = tapShell();
        shell.store = {snapshot: () => ({selection: {character: "Ada", thread: "t1"}})};
        shell.host = {listUtilities: () => []};
        shell.closeMenu = () => undefined;
        shell.told = [];
        shell.tell = (text) => shell.told.push(text);
        const labels = shell.utilityItems().map((item) => item.textContent);
        const item = shell.tapToCloseItem();
        const before = item.getAttribute("aria-checked");
        item.handlers.click.forEach((fn) => fn({}));
        const off = {state: shell.state.tapToClose, stored: Object.values(stored)[0]};
        const next = Object.create(NS.Shell.prototype);
        next.state = {tapToClose: true};
        next._restoreTapToClose();
        const fresh = Object.create(NS.Shell.prototype);
        fresh.state = {};
        for (const k of Object.keys(stored)) delete stored[k];
        fresh._restoreTapToClose();
        console.log(JSON.stringify({labels, before, role: item.getAttribute("role"), off,
                                    restored: next.state.tapToClose, fresh: fresh.state.tapToClose,
                                    told: shell.told}));
    """)

    panel = found["labels"].index("Panel")
    assert found["labels"][panel:panel + 4] == ["Panel", "Free Float", "Tab Bar in Focus",
                                                 "Tap Outside to Close"]
    assert found["before"] == "true" and found["role"] == "menuitemcheckbox"
    assert found["off"] == {"state": False, "stored": "off"}
    assert found["restored"] is False
    assert found["fresh"] is True, "on until turned off"
    assert found["told"] == ["A tap outside the panel leaves it open."]


def test_a_cancelled_press_or_a_key_spends_the_press_through_the_listeners_themselves():
    """Heard in the capture phase on the window (and each frame's): a press
    the browser cancelled -- a touch it took for a scroll -- and a key between
    the press and the release (a key that clicks a button has no press of its
    own) each spend the press, so its release closes nothing. A finger's
    touchend is heard not passive, so that it can be cancelled; nothing that
    decides a scroll -- a touchstart, a touchmove, the wheel -- is heard at
    all."""
    found = run(WORLD + """
        const heard = {};
        const view = {addEventListener: (type, fn, options) => { heard[type] = {fn, options}; },
                      removeEventListener() {}};
        const run = (between) => {
            const shell = tapShell();
            shell.disposers = [];
            shell.listenForTaps(view, null);
            heard.pointerdown.fn(at(page, 200, 200));
            if (between) heard[between].fn({isTrusted: true});
            heard.pointerup.fn(at(page, 200, 200));
            return shell.closed;
        };
        console.log(JSON.stringify({tap: run(null), cancelled: run("pointercancel"),
                                    key: run("keydown"),
                                    options: Object.keys(heard).sort().map((t) => [t, heard[t].options])}));
    """)

    assert found["tap"] == 1 and found["cancelled"] == 0 and found["key"] == 0
    assert found["options"] == [["click", True], ["contextmenu", True],
                                ["keydown", True], ["mousedown", True], ["mouseup", True],
                                ["pointercancel", True], ["pointerdown", True],
                                ["pointerup", True], ["selectstart", True],
                                ["touchend", {"capture": True, "passive": False}]]


# An event that says whether it was taken from the page: cancelled and
# stopped, as `holdBack` takes it.
EVENTS = WORLD + """
const ev = (type, target, x, y, more) => {
    const e = at(target, x, y, Object.assign({type, cancelable: true, pointerId: 7}, more || {}));
    e.taken = [];
    e.preventDefault = () => e.taken.push("cancelled");
    e.stopImmediatePropagation = () => e.taken.push("stopped");
    return e;
};
const taken = (e) => e.taken.join("+") || "passed";
let clock = 1000;
Date.now = () => clock;
"""


def test_the_first_tap_outside_is_the_panels_alone():
    """Asked for: "I want the first tap outside of an open flyout menu to do
    nothing except close the flyout menu. Essentially, catch all the taps and
    make flyout menu minimize action the priority. This means I have to tap
    twice to do something outside of an open flyout menu."

    The press is taken from the page as it goes down -- cancelled, which
    starts no focus, selection or emulated mouse events, and stopped -- and
    so is everything of it after: its mouse events, its release, a finger's
    touchend (cancelled, which sends no click) and the click, which ends what
    is held. The next tap is the page's."""
    found = run(EVENTS + """
        const shell = tapShell();
        const seen = {};
        const step = (name, e, how) => { (how || ((x) => shell.holdTrail(x)))(e);
                                         seen[name] = taken(e); };
        step("pointerdown", ev("pointerdown", page, 200, 200), (e) => shell.notePressOutside(e));
        step("mousedown", ev("mousedown", page, 200, 200));
        seen.openWhileDown = shell.state.panelOpen;
        step("pointerup", ev("pointerup", page, 203, 198), (e) => shell.tapOutside(e));
        seen.closed = shell.closed;
        step("touchend", ev("touchend", page, 203, 198));
        step("mouseup", ev("mouseup", page, 203, 198));
        clock += 900;
        step("click", ev("click", page, 203, 198));
        // The click ended what was held.
        step("anotherClick", ev("click", page, 203, 198));
        // The next tap, the panel closed: nothing of it is taken.
        step("nextDown", ev("pointerdown", page, 200, 200), (e) => shell.notePressOutside(e));
        step("nextClick", ev("click", page, 200, 200));
        console.log(JSON.stringify(seen));
    """)

    held = "cancelled+stopped"
    assert found == {"pointerdown": held, "mousedown": held, "openWhileDown": True,
                     "pointerup": held, "closed": 1, "touchend": held, "mouseup": held,
                     "click": held, "anotherClick": "passed", "nextDown": "passed",
                     "nextClick": "passed"}


def test_a_press_on_what_the_browser_drags_is_stopped_but_not_cancelled():
    """A picture or a link the browser drags of its own: cancelling its press
    would stop the drag -- a gallery's picture dragged onto the settings
    column docked in the panel (§3.41). So its press and its mousedown are
    only stopped; its release and click are taken whole, and a tap on it
    still only puts the panel away."""
    found = run(EVENTS + """
        const picture = {nodeType: 1, parentElement: null,
                         closest: (selector) => (selector === "[draggable], img, a[href]"
                                                 ? {draggable: true} : null)};
        const undraggable = {nodeType: 1, parentElement: null,
                             closest: (selector) => (selector === "[draggable], img, a[href]"
                                                     ? {draggable: false} : null)};
        const press = (target) => {
            const shell = tapShell();
            const down = ev("pointerdown", target, 200, 200);
            shell.notePressOutside(down);
            const mouse = ev("mousedown", target, 200, 200);
            shell.holdTrail(mouse);
            const up = ev("pointerup", target, 200, 200);
            shell.tapOutside(up);
            const click = ev("click", target, 200, 200);
            shell.holdTrail(click);
            return [taken(down), taken(mouse), taken(up), taken(click), shell.closed];
        };
        console.log(JSON.stringify({picture: press(picture), undraggable: press(undraggable)}));
    """)

    held = "cancelled+stopped"
    assert found["picture"] == ["stopped", "stopped", held, held, 1]
    assert found["undraggable"] == [held, held, held, held, 1]


def test_a_scroll_and_whatever_is_not_a_tap_is_left_to_the_page():
    """"Drag scroll should still work though." A finger's press the browser
    takes for a scroll is cancelled by it, and leaves nothing held: its
    touchend and anything after pass. A press on a scroll bar is not taken
    at all, nor one with the panel closed or the switch off, nor the page's
    own events. A press let go further than a tap is no tap -- it closes
    nothing -- but what follows it is still the press's. What is held is
    held for a moment only, and a new press ends it; a frame's is the
    frame's."""
    found = run(EVENTS + """
        const out = {};
        {
            const shell = tapShell();
            const down = ev("pointerdown", page, 200, 200, {pointerType: "touch"});
            shell.notePressOutside(down);
            shell.outsidePress = null;   // what the wired pointercancel does
            const end = ev("touchend", page, 200, 260);
            const click = ev("click", page, 200, 260);
            shell.holdTrail(end);
            shell.holdTrail(click);
            out.scroll = [taken(down), taken(end), taken(click), shell.closed];
        }
        {
            const shell = tapShell();
            const bar = Object.assign(node(null), {
                getBoundingClientRect: () => ({left: 0, top: 0, width: 200, height: 100}),
                clientWidth: 184, clientHeight: 100, scrollHeight: 900, scrollWidth: 184,
                clientLeft: 0, clientTop: 0});
            const down = ev("pointerdown", bar, 190, 50);
            out.scrollbar = [shell.notePressOutside(down), taken(down)];
        }
        const untaken = (state) => {
            const shell = tapShell(state);
            const down = ev("pointerdown", page, 200, 200);
            shell.notePressOutside(down);
            return taken(down);
        };
        out.closed = untaken({panelOpen: false});
        out.switchOff = untaken({tapToClose: false});
        {
            const shell = tapShell();
            const down = ev("pointerdown", page, 200, 200, {isTrusted: false});
            shell.notePressOutside(down);
            out.scripted = taken(down);
        }
        {
            const shell = tapShell();
            shell.notePressOutside(ev("pointerdown", page, 200, 200));
            const up = ev("pointerup", page, 260, 200);
            shell.tapOutside(up);
            const scripted = ev("click", page, 230, 200, {isTrusted: false});
            shell.holdTrail(scripted);
            const click = ev("click", page, 230, 200);
            shell.holdTrail(click);
            out.drag = [taken(up), shell.closed, taken(scripted), taken(click)];
        }
        {
            const shell = tapShell();
            shell.notePressOutside(ev("pointerdown", page, 200, 200));
            shell.tapOutside(ev("pointerup", page, 200, 200));
            clock += 1001;
            const late = ev("click", page, 200, 200);
            shell.holdTrail(late);
            out.late = taken(late);
        }
        {
            const shell = tapShell();
            shell.notePressOutside(ev("pointerdown", page, 200, 200));
            shell.tapOutside(ev("pointerup", page, 200, 200));
            shell.state.panelOpen = false;
            shell.notePressOutside(ev("pointerdown", page, 300, 300));
            const click = ev("click", page, 300, 300);
            shell.holdTrail(click);
            out.newPress = taken(click);
        }
        {
            const shell = tapShell();
            shell.notePressOutside(ev("pointerdown", page, 200, 200), "f");
            shell.tapOutside(ev("pointerup", page, 200, 200), "f");
            const here = ev("click", page, 200, 200);
            const there = ev("click", page, 200, 200);
            shell.holdTrail(here);
            shell.holdTrail(there, "f");
            out.frame = [shell.closed, taken(here), taken(there)];
        }
        console.log(JSON.stringify(out));
    """)

    held = "cancelled+stopped"
    assert found["scroll"] == [held, "passed", "passed", 0]
    assert found["scrollbar"] == [False, "passed"]
    assert found["closed"] == "passed" and found["switchOff"] == "passed"
    assert found["scripted"] == "passed"
    assert found["drag"] == [held, 0, "passed", held]
    assert found["late"] == "passed"
    assert found["newPress"] == "passed"
    assert found["frame"] == [1, "passed", held]


def test_focus_in_the_panel_goes_to_the_launcher():
    """The tap put the keyboard's focus nowhere, so focus that was in the
    panel goes to the launcher, as the ✕ and one tap on the header give it;
    focus elsewhere stays where it was."""
    found = run(EVENTS + """
        const tapWith = (inside) => {
            const shell = tapShell();
            const box = {};
            shell.nodes.panel.contains = (n) => n === box;
            Object.defineProperty(document, "activeElement",
                                  {configurable: true, get: () => (inside ? box : page)});
            shell.notePressOutside(ev("pointerdown", page, 200, 200));
            shell.tapOutside(ev("pointerup", page, 200, 200));
            return [shell.closed, shell.focused];
        };
        console.log(JSON.stringify({inside: tapWith(true), elsewhere: tapWith(false)}));
    """)

    assert found == {"inside": [1, 1], "elsewhere": [1, 0]}
