"""A tap outside the open panel puts it away.

Asked for: "if the flyout menu is open, tapping anywhere outside of it will
close it. It should close back to the button state. Additionally, the location
of the flyout button when I tap out to close should be the closest docked
location. The docked locations are those when free float is off, there are six
of them. Dragging or scrolling outside of the open fly out should have no
impact, only tapping / clicking outside of it should make it minimize to the
flyout button. Make the new tap to close function enabled / disabled via a new
"..." menu entry".

These hold the rules under node: what is a tap and what is not, what is
outside, where the launcher goes, and the switch. That real taps, drags,
wheels, touch scrolls and a frame's taps are told apart in a browser is
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
                                 floatAt: null, anchorOverride: "bottom-right"}, state || {});
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


def test_a_tap_outside_puts_the_panel_away_and_leaves_the_focus_where_the_tap_put_it():
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


def test_floating_the_launcher_goes_to_the_nearest_of_the_six():
    """Floating, the launcher (and the panel, next time) is put at the
    resting place nearest the panel's middle, as a floating place exactly on
    it: against the edges it names, or halfway across. Anchored, it is where
    the panel was: nothing is moved."""
    found = run(WORLD + """
        const places = {};
        const tapFrom = (box, state) => {
            const shell = tapShell(Object.assign({freeFloat: true, floatAt: {x: 0.3, y: 0.3}}, state || {}));
            shell.nodes.panel.getBoundingClientRect = () => Object.assign(
                {right: box.left + 360, bottom: box.top + 460, width: 360, height: 460}, box);
            shell.notePressOutside(at(page, 5, 5));
            shell.tapOutside(at(page, 5, 5));
            return {float: shell.state.floatAt, anchor: shell.state.anchorOverride,
                    saved: shell.floatSaved};
        };
        places.topLeft = tapFrom({left: 60, top: 40});
        places.bottomCentre = tapFrom({left: 470, top: 330});
        places.topRight = tapFrom({left: 880, top: 10});
        places.bottomRight = tapFrom({left: 900, top: 320});
        const anchored = tapShell({freeFloat: false, floatAt: {x: 0.3, y: 0.3}, anchorOverride: "top-left"});
        anchored.notePressOutside(at(page, 5, 5));
        anchored.tapOutside(at(page, 5, 5));
        places.anchored = {float: anchored.state.floatAt, anchor: anchored.state.anchorOverride,
                           saved: anchored.floatSaved};
        console.log(JSON.stringify(places));
    """)

    assert found["topLeft"] == {"float": {"x": 0, "y": 0}, "anchor": "top-left", "saved": 1}
    assert found["bottomCentre"] == {"float": {"x": 0.5, "y": 1}, "anchor": "bottom-center",
                                     "saved": 1}
    assert found["topRight"] == {"float": {"x": 1, "y": 0}, "anchor": "top-right", "saved": 1}
    assert found["bottomRight"] == {"float": {"x": 1, "y": 1}, "anchor": "bottom-right",
                                    "saved": 1}
    assert found["anchored"] == {"float": {"x": 0.3, "y": 0.3}, "anchor": "top-left", "saved": 0}


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
    the browser cancelled -- a touch it took for a scroll, which makes no
    click -- and a key between the press and the click (a key that clicks a
    button has no press of its own) each spend the press, so a later click
    cannot end it."""
    found = run(WORLD + """
        const heard = {};
        const view = {addEventListener: (type, fn, capture) => { heard[type] = {fn, capture}; },
                      removeEventListener() {}};
        const run = (between) => {
            const shell = tapShell();
            shell.disposers = [];
            shell.listenForTaps(view, null);
            heard.pointerdown.fn(at(page, 200, 200));
            if (between) heard[between].fn({isTrusted: true});
            heard.click.fn(at(page, 200, 200));
            return shell.closed;
        };
        console.log(JSON.stringify({tap: run(null), cancelled: run("pointercancel"),
                                    key: run("keydown"),
                                    capture: Object.keys(heard).sort().map((t) => [t, heard[t].capture])}));
    """)

    assert found["tap"] == 1 and found["cancelled"] == 0 and found["key"] == 0
    assert found["capture"] == [["click", True], ["keydown", True], ["pointercancel", True],
                                ["pointerdown", True]]
