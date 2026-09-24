"""The assistant recovers from the states that used to need a reload.

Reported from use: after a remote session and a "disconnected" message from
the WebUI, the launcher stopped responding, and with focus mode on there was
no way to another workspace short of reloading the page. Reproduced in a real
browser: a touch drag whose release never arrives -- the classic way to lose
one is a remote-desktop session changing hands mid-press -- left the drag
preview on screen over the launcher, and the preview took every press, because
its ``pointer-events: none`` lost to the root's own rule on specificity. For a
touch the drag could never end at all: every later touch has a different id.

Alongside it, the conversation's transport had two dead ends of its own: a
stream that died without closing was never torn down, and a server restart --
a new key, a new epoch -- was answered with "Reload the page" for as long as
the tab stayed open.

Every test here was checked against the change it guards by reverting that
change and watching the test fail.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from test_assistant_js import SHELL, STORE, run

ROOT = pathlib.Path(__file__).resolve().parent.parent


def stylesheet() -> str:
    return (ROOT / "style.css").read_text(encoding="utf-8")


# A shell with the nodes the gesture code touches and nothing else. The root's
# `contains` is real enough to answer for the nodes built here.
GESTURES = """
function gestureShell() {
    const shell = Object.create(NS.Shell.prototype);
    const root = document.createElement("div");
    const launcher = document.createElement("button");
    const panel = document.createElement("section");
    const header = document.createElement("header");
    const resize = document.createElement("div");
    const ghost = document.createElement("div");
    const mine = new Set([root, launcher, panel, header, resize, ghost]);
    root.contains = (node) => mine.has(node);
    root.isConnected = true;
    [launcher, panel, resize].forEach((node) => {
        node.captures = [];
        node.setPointerCapture = (id) => node.captures.push("set:" + id);
        node.releasePointerCapture = (id) => node.captures.push("release:" + id);
    });
    launcher.getBoundingClientRect = () => ({left: 100, top: 700, right: 260, bottom: 744,
                                             width: 160, height: 44});
    launcher.offsetWidth = 160;
    launcher.offsetHeight = 44;
    panel.offsetWidth = 360;
    panel.offsetHeight = 500;
    shell.nodes = {root, launcher, panel, header, resize, ghost};
    shell.state = {panelOpen: false, freeFloat: false, floatAt: null,
                   anchorOverride: "bottom-left", panelWidth: 360,
                   conversationExpanded: true, focusEnabled: false,
                   focusWorkspaceId: null};
    shell.settings = {defaultAnchor: "bottom-left"};
    shell.drag = null;
    shell.strip = null;
    shell.resizing = null;
    shell.frame = 0;
    shell.placed = 0;
    shell.placedNow = 0;
    shell.place = () => { shell.placed += 1; };
    shell.placeNow = () => { shell.placedNow += 1; };
    shell.preview = () => { ghost.hidden = false; };
    shell._save = () => {};
    shell._saveFloat = () => {};
    shell.say = (text) => { shell.said = text; };
    shell.focus = {on: false, exits: 0,
                   isActive() { return this.on; },
                   exit() { this.exits += 1; this.on = false; return true; }};
    return shell;
}
const touch = (over) => Object.assign({pointerId: 7, pointerType: "touch", isPrimary: true,
                                       button: 0, buttons: 1, clientX: 180, clientY: 722},
                                      over || {});
const mouse = (over) => Object.assign({pointerId: 1, pointerType: "mouse", isPrimary: true,
                                       button: 0, buttons: 1, clientX: 180, clientY: 722},
                                      over || {});
"""


class TestAGestureAlwaysEnds:
    def test_a_new_press_ends_a_drag_whose_release_never_came(self):
        """The failure as reproduced: a touch drag with no release, then the
        mouse. Before this, the drag outlived the page."""
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.startDrag(touch({target: shell.nodes.launcher}), shell.nodes.launcher);
            shell.moveDrag(touch({clientX: 190, clientY: 710}));
            const stuck = {drag: !!shell.drag, ghost: !shell.nodes.ghost.hidden};
            shell.supersede(mouse({target: shell.nodes.launcher}));
            console.log(JSON.stringify({stuck, drag: shell.drag,
                ghost: !shell.nodes.ghost.hidden,
                dragging: shell.nodes.root.classList.contains("forge-assistant-dragging"),
                released: shell.nodes.launcher.captures}));
        """)

        assert found["stuck"] == {"drag": True, "ghost": True}, "the setup is the bug"
        assert found["drag"] is None
        assert found["ghost"] is False
        assert found["dragging"] is False
        assert "release:7" in found["released"]

    def test_a_second_finger_does_not_end_the_first(self):
        """A non-primary pointer is a finger added to a gesture in progress,
        not the start of a new one."""
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.startDrag(touch({target: shell.nodes.launcher}), shell.nodes.launcher);
            shell.moveDrag(touch({clientX: 190, clientY: 710}));
            shell.supersede(touch({pointerId: 8, isPrimary: false,
                                   target: shell.nodes.launcher}));
            console.log(JSON.stringify({drag: !!shell.drag}));
        """)

        assert found["drag"] is True

    def test_a_mouse_moving_with_no_button_down_is_not_dragging(self):
        """The release happened where this page could not hear it. Without
        this the launcher followed the cursor until the next click."""
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.startDrag(mouse({target: shell.nodes.launcher}), shell.nodes.launcher);
            shell.moveDrag(mouse({clientX: 200, clientY: 700}));
            const held = !!shell.drag;
            shell.moveDrag(mouse({clientX: 220, clientY: 690, buttons: 0}));
            console.log(JSON.stringify({held, after: shell.drag,
                                        left: shell.nodes.launcher.style.left}));
        """)

        assert found["held"] is True
        assert found["after"] is None
        assert found["left"] == "", "cancelled, so it goes back where it was"

    def test_a_touch_drag_is_not_ended_by_its_own_moves(self):
        """`buttons` is only meaningful for a mouse; a finger that is down
        reports it inconsistently across browsers."""
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.startDrag(touch({target: shell.nodes.launcher}), shell.nodes.launcher);
            shell.moveDrag(touch({clientX: 200, clientY: 700, buttons: 0}));
            console.log(JSON.stringify({drag: !!shell.drag}));
        """)

        assert found["drag"] is True

    def test_losing_capture_ends_the_drag_and_our_own_release_is_harmless(self):
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.startDrag(mouse({target: shell.nodes.launcher}), shell.nodes.launcher);
            shell.moveDrag(mouse({clientX: 200, clientY: 700}));
            shell.endDrag(mouse({clientX: 200, clientY: 700}), true);   // lostpointercapture
            const ended = shell.drag;
            // A normal drop, then the capture loss that releasing it causes.
            shell.startDrag(mouse({target: shell.nodes.launcher}), shell.nodes.launcher);
            shell.moveDrag(mouse({clientX: 200, clientY: 700}));
            shell.endDrag(mouse({clientX: 200, clientY: 700}), false);
            const anchor = shell.state.anchorOverride;
            shell.endDrag(mouse(), true);
            console.log(JSON.stringify({ended, anchor, after: shell.state.anchorOverride}));
        """)

        assert found["ended"] is None
        assert found["after"] == found["anchor"], (
            "the late lostpointercapture must not undo a drop that committed")

    def test_only_a_launcher_drag_swallows_the_launchers_click(self):
        """A panel dragged by its header leaves no click on the launcher; a
        flag set for one used to eat the next real press on it."""
        found = run(GESTURES + """
            const shell = gestureShell();
            const flags = {};
            [["panel", shell.nodes.panel], ["launcher", shell.nodes.launcher]]
                .forEach(([name, node]) => {
                    shell.suppressClick = false;
                    shell.startDrag(mouse({target: node}), node);
                    shell.moveDrag(mouse({clientX: 220, clientY: 690}));
                    shell.endDrag(mouse({clientX: 220, clientY: 690}), false);
                    flags[name] = !!shell.suppressClick;
                });
            console.log(JSON.stringify(flags));
        """)

        assert found == {"panel": False, "launcher": True}

    def test_the_next_press_clears_a_click_flag_nobody_consumed(self):
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.suppressClick = true;
            shell.stripMoved = true;
            shell.supersede(mouse({target: shell.nodes.launcher}));
            console.log(JSON.stringify({click: shell.suppressClick, strip: shell.stripMoved}));
        """)

        assert found == {"click": False, "strip": False}

    def test_leaving_the_window_cancels_every_gesture(self):
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.startDrag(mouse({target: shell.nodes.launcher}), shell.nodes.launcher);
            shell.moveDrag(mouse({clientX: 220, clientY: 690}));
            shell.startResize(Object.assign(mouse({pointerId: 2}), {preventDefault() {}}));
            shell.cancelGestures();
            console.log(JSON.stringify({drag: shell.drag, resizing: shell.resizing,
                                        ghost: !shell.nodes.ghost.hidden}));
        """)

        assert found == {"drag": None, "resizing": None, "ghost": False}

    def test_the_hooks_that_cancel_are_wired(self):
        """`cancelGestures` is only as good as the moments that call it."""
        shell = SHELL.read_text(encoding="utf-8")
        wiring = shell.split("Shell.prototype.wire = function", 1)[1] \
            .split("Shell.prototype.grow = function", 1)[0]

        assert re.search(r'window, "pointerdown", \(event\) => this\.supersede\(event\), true',
                         wiring), "capture phase, before anything else sees the press"
        assert re.search(r'window, "blur", \(\) => this\.cancelGestures\(\)', wiring)
        assert "if (document.hidden) {\n                this.cancelGestures();" in wiring
        assert wiring.count('"lostpointercapture"') == 2


class TestTheResizeHandle:
    """It added two window listeners per gesture and removed them only on
    pointerup, so a gesture that ended any other way left every later pointer
    movement on the page resizing the panel."""

    def test_a_resize_adds_no_listeners_of_its_own(self):
        found = run(GESTURES + """
            const shell = gestureShell();
            let added = 0;
            globalThis.addEventListener = () => { added += 1; };
            for (let i = 0; i < 3; i += 1) {
                shell.startResize(Object.assign(mouse({pointerId: 2}), {preventDefault() {}}));
                shell.endResize(null, true);
            }
            console.log(JSON.stringify({added}));
        """)

        assert found["added"] == 0

    def test_a_cancelled_resize_is_over_and_keeps_its_width(self):
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.startResize(Object.assign(mouse({pointerId: 2, clientX: 500}),
                                            {preventDefault() {}}));
            shell.moveResize(mouse({pointerId: 2, clientX: 460}));
            const reached = shell.state.panelWidth;
            shell.endResize(mouse({pointerId: 2}), true);
            shell.moveResize(mouse({pointerId: 2, clientX: 300}));
            console.log(JSON.stringify({reached, after: shell.state.panelWidth,
                                        resizing: shell.resizing,
                                        captures: shell.nodes.resize.captures}));
        """)

        assert found["reached"] == 400
        assert found["after"] == 400, "a move after the end resizes nothing"
        assert found["resizing"] is None
        assert found["captures"] == ["set:2", "release:2"]

    def test_moves_from_another_pointer_are_ignored(self):
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.startResize(Object.assign(mouse({pointerId: 2, clientX: 500}),
                                            {preventDefault() {}}));
            shell.moveResize(mouse({pointerId: 9, clientX: 300}));
            console.log(JSON.stringify({width: shell.state.panelWidth}));
        """)

        assert found["width"] == 360

    def test_a_move_places_once_a_frame_rather_than_once_an_event(self):
        """Every placement reads the panel's size; a pointer reports far more
        often than the screen draws."""
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.startResize(Object.assign(mouse({pointerId: 2, clientX: 500}),
                                            {preventDefault() {}}));
            for (let x = 499; x > 470; x -= 1) shell.moveResize(mouse({pointerId: 2, clientX: x}));
            console.log(JSON.stringify({now: shell.placedNow, framed: shell.placed}));
        """)

        assert found["now"] == 0
        assert found["framed"] > 0

    def test_a_finger_on_the_edge_is_a_resize_and_not_a_pan(self):
        rule = stylesheet().split(".forge-assistant-resize {", 1)[1].split("}", 1)[0]

        assert "touch-action: none" in rule


class TestThePreviewCannotTakeAPress:
    def test_its_rule_carries_the_weight_of_the_rule_it_has_to_beat(self):
        """`#forge-assistant-root > *` is an id selector. A class alone never
        won against it, so `pointer-events: none` was never in force."""
        css = stylesheet()
        rule = css.split("#forge-assistant-root > .forge-assistant-ghost {", 1)[1] \
            .split("}", 1)[0]

        assert "pointer-events: none" in rule
        assert "#forge-assistant-root > * {\n    pointer-events: auto;" in css, (
            "the rule it is written against; if that ever changes this test "
            "is asking the wrong question")


class TestAFailingHandlerResetsTheShell:
    def test_the_exception_is_contained_and_the_shell_recovers(self):
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.disposers = [];
            let recovered = 0;
            shell.recover = () => { recovered += 1; return true; };
            const node = document.createElement("button");
            const errors = [];
            const original = console.error;
            console.error = (...args) => errors.push(String(args[0]));
            shell.on(node, "click", () => { throw new Error("boom"); });
            let escaped = false;
            try {
                node.handlers.click.forEach((fn) => fn({}));
                node.handlers.click.forEach((fn) => fn({}));
            } catch (error) {
                escaped = true;
            }
            console.error = original;
            console.log(JSON.stringify({escaped, recovered, logged: errors.length}));
        """)

        assert found["escaped"] is False
        assert found["recovered"] == 2
        assert found["logged"] == 1, "once per distinct failure"

    def test_recovery_is_rate_limited(self):
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.showOpen = () => {};
            const first = shell.recover();
            const second = shell.recover();
            console.log(JSON.stringify({first, second}));
        """)

        assert found == {"first": True, "second": False}

    def test_recovery_puts_back_a_root_something_else_took_off_the_page(self):
        found = run(GESTURES + """
            const shell = gestureShell();
            shell.showOpen = () => {};
            shell.nodes.root.isConnected = false;
            shell.recover();
            console.log(JSON.stringify({
                back: document.body.children.includes(shell.nodes.root)}));
        """)

        assert found["back"] is True

    def test_recovery_undoes_a_stuck_interaction_and_nothing_chosen(self):
        """Open stays open, focus stays on. It resets what nobody chose."""
        found = run(GESTURES + """
            const shell = gestureShell();
            let shown = null;
            shell.showOpen = (open) => { shown = open; };
            shell.state.panelOpen = true;
            shell.focus.on = true;
            shell.startDrag(touch({target: shell.nodes.launcher}), shell.nodes.launcher);
            shell.moveDrag(touch({clientX: 190, clientY: 710}));
            shell.recover();
            console.log(JSON.stringify({shown, focus: shell.focus.on,
                                        exits: shell.focus.exits, drag: shell.drag}));
        """)

        assert found == {"shown": True, "focus": True, "exits": 0, "drag": None}

    def test_a_failed_render_resets_the_shell_too(self):
        shell = SHELL.read_text(encoding="utf-8")
        subscriber = shell.split("this.store.subscribeState((view) => {", 1)[1] \
            .split("}));", 1)[0]

        assert "this.render(latest);" in subscriber
        assert "this.fault(error);" in subscriber


class TestACoveredLauncher:
    """The belt to `supersede`'s braces: whatever is over the launcher, two
    presses on it that land elsewhere reset the shell, and focus mode is left
    if that did not uncover it."""

    SETUP = GESTURES + """
        const shell = gestureShell();
        shell.recovers = 0;
        shell.recover = () => { shell.recovers += 1; return true; };
        const cover = document.createElement("div");
        cover.closest = () => null;
        const press = (over) => mouse(Object.assign({target: cover}, over || {}));
    """

    def test_two_presses_that_miss_it_reset_the_shell(self):
        found = run(self.SETUP + """
            shell.reachable = () => true;
            shell.supersede(press());
            const once = shell.recovers;
            shell.supersede(press());
            console.log(JSON.stringify({once, twice: shell.recovers}));
        """)

        assert found == {"once": 0, "twice": 1}

    def test_a_press_that_reaches_it_starts_the_count_again(self):
        found = run(self.SETUP + """
            shell.reachable = () => true;
            shell.supersede(press());
            shell.supersede(mouse({target: shell.nodes.launcher}));
            shell.supersede(press());
            console.log(JSON.stringify({recovers: shell.recovers}));
        """)

        assert found["recovers"] == 0

    def test_a_press_somewhere_else_on_the_page_is_not_counted(self):
        found = run(self.SETUP + """
            shell.reachable = () => true;
            shell.supersede(press({clientX: 900, clientY: 100}));
            shell.supersede(press({clientX: 900, clientY: 100}));
            console.log(JSON.stringify({recovers: shell.recovers}));
        """)

        assert found["recovers"] == 0

    def test_a_dialog_over_it_is_meant_to_be_there(self):
        found = run(self.SETUP + """
            shell.reachable = () => false;
            shell.focus.on = true;
            cover.closest = (selector) => (selector.indexOf("dialog") >= 0 ? cover : null);
            shell.supersede(press());
            shell.supersede(press());
            console.log(JSON.stringify({recovers: shell.recovers, focus: shell.focus.on}));
        """)

        assert found == {"recovers": 0, "focus": True}

    def test_focus_is_left_when_the_reset_did_not_uncover_it(self):
        """In focus mode the launcher is the way to another workspace. If it
        cannot be reached, the host's own tab bar has to come back."""
        found = run(self.SETUP + """
            shell.reachable = () => false;
            shell.focus.on = true;
            shell.state.focusEnabled = true;
            shell.nodes.focusToggle = document.createElement("button");
            shell.supersede(press());
            shell.supersede(press());
            console.log(JSON.stringify({exits: shell.focus.exits,
                                        enabled: shell.state.focusEnabled,
                                        pressed: shell.nodes.focusToggle["aria-pressed"]}));
        """)

        assert found == {"exits": 1, "enabled": False, "pressed": "false"}

    def test_focus_is_kept_when_the_reset_uncovered_it(self):
        found = run(self.SETUP + """
            shell.reachable = () => true;
            shell.focus.on = true;
            shell.supersede(press());
            shell.supersede(press());
            console.log(JSON.stringify({exits: shell.focus.exits}));
        """)

        assert found["exits"] == 0

    def test_an_open_panel_is_not_watched(self):
        found = run(self.SETUP + """
            shell.state.panelOpen = true;
            shell.supersede(press());
            shell.supersede(press());
            console.log(JSON.stringify({recovers: shell.recovers}));
        """)

        assert found["recovers"] == 0


# --------------------------------------------------------------------------- #
# The transport
# --------------------------------------------------------------------------- #


STORE_SETUP = """
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push({fn, ms}); return timers.length; };
const calls = [];
let answers = {};
globalThis.fetch = (url, options) => {
    calls.push(String(url));
    const found = Object.keys(answers).find((path) => String(url).indexOf(path) >= 0);
    const [status, body] = found ? answers[found] : [500, {}];
    return Promise.resolve({ok: status < 400, status,
                            json: () => Promise.resolve(body)});
};
const field = document.createElement("textarea");
field.id = "mc-llm-chat-conversation-key";
field.tagName = "TEXTAREA";
field.value = "old-key";
document.getElementById = (id) => (id === field.id ? field : null);
const store = new NS.Store();
const tick = () => new Promise((resolve) => setImmediate(resolve));
async function settle() { for (let i = 0; i < 10; i += 1) await tick(); }
"""


class TestADeadStreamIsReplaced:
    def test_the_watchdog_tears_down_a_stream_gone_silent(self):
        found = run(STORE_SETUP + """
            let aborted = 0;
            // An open feed, which it only is while a reply is on its way.
            store.sleeping = false;
            store.operations.set("op", {id: "op", key: "", terminal: false});
            store.connected = true;
            store.lastTraffic = Date.now() - 50000;
            store._abort = {abort() { aborted += 1; }};
            store.watch();
            timers[timers.length - 1].fn();
            console.log(JSON.stringify({aborted, connected: store.connected,
                                        failures: store.failures,
                                        rearmed: timers.some((t) => t.ms === 15000),
                                        ladder: timers.some((t) => t.ms >= 1000 && t.ms < 1400)}));
        """, sources=("store",))

        assert found["aborted"] == 1, "a stream that never closes has to be let go of"
        assert found["connected"] is False
        assert found["failures"] == 1
        assert found["ladder"] is True, "and the reconnect ladder takes it from there"
        assert found["rearmed"] is True

    def test_a_quiet_stream_inside_the_limit_is_left_alone(self):
        found = run(STORE_SETUP + """
            let aborted = 0;
            store.sleeping = false;
            store.connected = true;
            store.lastTraffic = Date.now() - 30000;
            store._abort = {abort() { aborted += 1; }};
            store.watch();
            timers[timers.length - 1].fn();
            console.log(JSON.stringify({aborted, connected: store.connected}));
        """, sources=("store",))

        assert found == {"aborted": 0, "connected": True}

    def test_coming_back_uses_the_short_limit(self):
        """Online again, or out of the back-forward cache: one missed
        heartbeat is enough, because at those moments the stream usually is
        dead."""
        found = run(STORE_SETUP + """
            let aborted = 0;
            store.sleeping = false;
            store.connected = true;
            store.lastTraffic = Date.now() - 25000;
            store._abort = {abort() { aborted += 1; }};
            store.refresh = () => Promise.resolve(null);
            store.reconcile(false);
            const ordinary = aborted;
            store.reconcile(true);
            console.log(JSON.stringify({ordinary, eager: aborted}));
        """, sources=("store",))

        assert found == {"ordinary": 0, "eager": 1}

    def test_the_page_reconciles_eagerly_at_the_moments_it_comes_back(self):
        shell = SHELL.read_text(encoding="utf-8")
        wiring = shell.split("Shell.prototype.wire = function", 1)[1] \
            .split("Shell.prototype.grow = function", 1)[0]

        assert 'this.on(window, "online", () => this.store.reconcile(true));' in wiring
        assert "this.store.reconcile(!!(event && event.persisted));" in wiring
        assert re.search(r'document, "resume", \(\) => \{\s+this\.heal\(\);\s+'
                         r'this\.store\.reconcile\(true\);', wiring)


class TestTheFeedOpensOnlyWhileSomethingIsComing:
    """Reported from use: a tab left for half an hour or more came back with
    its connections half-dead -- open as far as the page could tell, and
    silent -- and everything on it hung. The rule since: a live connection is
    for a known boundary. The feed opens when a reply is asked for (or a
    snapshot shows one on its way), stays while it is written and while it
    is read aloud, and closes at the end. Everything else is read when the
    panel opens, when the page comes back and when the workspace changes."""

    READY = STORE_SETUP + """
        store.ready = true;
        const paths = () => calls.map((c) => c.split("?")[0].replace(/^.*conversation.v2/, ""));
    """

    def test_the_feed_starts_closed_and_start_opens_none(self):
        found = run(self.READY + """
            answers = {"/bootstrap": [200, {capabilities: {}, characters: [],
                                            selection: {character: "c", thread_id: "t"}}],
                       "/snapshot": [200, {operation: null, messages: []}]};
            store.ready = false;
            store.start().then(settle).then(() => console.log(JSON.stringify({
                paths: paths(), idle: store.snapshot().idle, connected: store.connected})));
        """, sources=("store",))

        assert found["paths"] == ["/bootstrap", "/snapshot"], found["paths"]
        assert found["idle"] is True and found["connected"] is False

    def test_asking_for_a_reply_opens_the_feed_before_the_request(self):
        found = run(self.READY + """
            // The server's snapshot lists every reply that has not finished,
            // so the one just asked for is in it.
            answers = {"/subscribe": [200, {feed: "f", stream_cursor: 1}],
                       "/snapshot": [200, {operation: {operation_id: "op1", status: "queued"}, messages: []}],
                       "/commands": [200, {ok: true, operation_id: "op1", phase: "queued",
                                           resulting_conversation: {character: "c", thread_id: "t"}}]};
            store.selection = {character: "c", thread: "t", epoch: "e"};
            const envelope = store.envelope("send");
            store.send(envelope).then(settle).then(() => console.log(JSON.stringify({
                paths: paths(), idle: store.snapshot().idle})));
        """, sources=("store",))

        assert found["paths"].index("/subscribe") < found["paths"].index("/commands"), found["paths"]
        assert found["idle"] is False, "and it stays open while the reply is being written"

    def test_a_command_that_asks_for_no_reply_opens_nothing(self):
        found = run(self.READY + """
            answers = {"/commands": [200, {ok: true}],
                       "/snapshot": [200, {operation: null, messages: []}]};
            store.selection = {character: "c", thread: "t", epoch: "e"};
            store.send(store.envelope("rename_thread")).then(settle).then(() => console.log(JSON.stringify({
                paths: paths(), idle: store.snapshot().idle})));
        """, sources=("store",))

        assert "/subscribe" not in found["paths"] and "/events" not in found["paths"], found["paths"]
        assert found["idle"] is True

    def test_the_end_of_the_reply_closes_the_feed(self):
        found = run(self.READY + """
            let aborted = 0;
            store.sleeping = false;
            store.connected = true;
            store.cursor = 4;
            store._abort = {abort() { aborted += 1; }};
            store.operations.set("op1", {id: "op1", key: "", terminal: false, seq: 1});
            store.refresh = () => Promise.resolve(null);
            store.apply({protocol_version: 2, stream_cursor: 5, kind: "operation_terminal",
                         operation_id: "op1", operation_seq: 2, conversation: {},
                         payload: {phase: "done"}});
            console.log(JSON.stringify({aborted, idle: store.sleeping, failures: store.failures}));
        """, sources=("store",))

        assert found == {"aborted": 1, "idle": True, "failures": 0}

    def test_read_aloud_keeps_it_open_until_the_speech_ends(self):
        """The reply's text is done, but it is still being spoken: the panel's
        Stop is driven by speech_state, so the feed stays until that says
        playback stopped."""
        found = run(self.READY + """
            let aborted = 0;
            store.sleeping = false;
            store.connected = true;
            store.cursor = 4;
            store._abort = {abort() { aborted += 1; }};
            store.refresh = () => Promise.resolve(null);
            store.apply({protocol_version: 2, stream_cursor: 5, kind: "speech_state",
                         operation_id: "op1", conversation: {}, payload: {playing: true}});
            const during = {aborted, idle: store.sleeping};
            store.apply({protocol_version: 2, stream_cursor: 6, kind: "speech_state",
                         operation_id: "op1", conversation: {}, payload: {playing: false}});
            console.log(JSON.stringify({during, after: {aborted, idle: store.sleeping}}));
        """, sources=("store",))

        assert found["during"] == {"aborted": 0, "idle": False}
        assert found["after"] == {"aborted": 1, "idle": True}

    def test_a_reply_found_in_a_snapshot_opens_the_feed(self):
        """Started in LLM Studio, or in another window: the panel learns of it
        when it looks, and follows it from there."""
        found = run(self.READY + """
            answers = {"/subscribe": [200, {feed: "f", stream_cursor: 1}],
                       "/snapshot": [200, {operation: {operation_id: "op9", status: "generating"},
                                           messages: []}]};
            store.selection = {character: "c", thread: "t", epoch: "e"};
            store.refresh().then(settle).then(() => console.log(JSON.stringify({
                paths: paths(), idle: store.snapshot().idle, replying: store.replying()})));
        """, sources=("store",))

        assert "/subscribe" in found["paths"], found["paths"]
        assert found["idle"] is False and found["replying"] is True

    def test_a_reply_the_server_has_finished_does_not_hold_the_feed(self):
        """A terminal event that never arrived cannot keep the feed open: the
        next snapshot that shows no reply marks it finished."""
        found = run(self.READY + """
            let aborted = 0;
            answers = {"/snapshot": [200, {operation: null, messages: []}]};
            store.selection = {character: "c", thread: "t", epoch: "e"};
            store.sleeping = false;
            store.connected = true;
            store._abort = {abort() { aborted += 1; }};
            store.operations.set("op1", {id: "op1", key: NS.conversationKey("c", "t"), terminal: false});
            store.refresh().then(settle).then(() => console.log(JSON.stringify({
                aborted, idle: store.sleeping, replying: store.replying()})));
        """, sources=("store",))

        assert found == {"aborted": 1, "idle": True, "replying": False}

    def test_a_feed_that_drops_with_nothing_coming_stays_closed(self):
        found = run(self.READY + """
            store.sleeping = false;
            store.connected = true;
            store.dropped();
            console.log(JSON.stringify({idle: store.sleeping, failures: store.failures,
                                        ladder: timers.some((t) => t.ms >= 1000 && t.ms <= 15400)}));
        """, sources=("store",))

        assert found == {"idle": True, "failures": 0, "ladder": False}

    def test_going_to_the_background_mid_reply_holds_it_until_it_ends(self):
        found = run(self.READY + """
            let aborted = 0;
            store.sleeping = false;
            store.connected = true;
            store.cursor = 4;
            store._abort = {abort() { aborted += 1; }};
            store.operations.set("op1", {id: "op1", key: "", terminal: false, seq: 1});
            store.sleep();
            const held = {aborted, idle: store.sleeping, timer: timers.some((t) => t.ms === 600000)};
            store.refresh = () => Promise.resolve(null);
            store.apply({protocol_version: 2, stream_cursor: 5, kind: "operation_terminal",
                         operation_id: "op1", operation_seq: 2, conversation: {},
                         payload: {phase: "done"}});
            console.log(JSON.stringify({held, after: {aborted, idle: store.sleeping}}));
        """, sources=("store",))

        assert found["held"] == {"aborted": 0, "idle": False, "timer": True}
        assert found["after"] == {"aborted": 1, "idle": True}

    def test_a_reply_that_never_ends_is_not_held_for_ever_in_the_background(self):
        found = run(self.READY + """
            let aborted = 0;
            store.sleeping = false;
            store.connected = true;
            store._abort = {abort() { aborted += 1; }};
            store.operations.set("op1", {id: "op1", key: "", terminal: false, seq: 1});
            store.sleep();
            timers.find((t) => t.ms === 600000).fn();
            store.review();
            console.log(JSON.stringify({aborted, idle: store.sleeping, calls}));
        """, sources=("store",))

        assert found == {"aborted": 1, "idle": True, "calls": []}, "and not reopened while away"

    def test_nothing_is_asked_for_while_away(self):
        found = run(self.READY + """
            store.feed = "f";
            store.refresh = () => { calls.push("refresh"); return Promise.resolve(null); };
            store.sleep();
            store.dropped();
            store.stream();
            store.connect();
            store.reconcile(true);
            store.check();
            settle().then(() => console.log(JSON.stringify({calls, failures: store.failures})));
        """, sources=("store",))

        assert found == {"calls": [], "failures": 0}

    def test_the_return_looks_and_opens_a_feed_only_for_a_reply(self):
        found = run(self.READY + """
            answers = {"/bootstrap": [200, {selection: {character: "c", thread_id: "t"}}],
                       "/snapshot": [200, {operation: null, messages: []}]};
            store.selection = {character: "c", thread: "t", epoch: "e"};
            store.sleep();
            store.wake();
            settle().then(() => console.log(JSON.stringify({paths: paths(), idle: store.sleeping})));
        """, sources=("store",))

        assert found["paths"] == ["/bootstrap", "/snapshot"], found["paths"]
        assert found["idle"] is True

    def test_looking_follows_the_conversation_llm_studio_is_on(self):
        """What the feed's character_changed used to carry live, read when the
        panel looks."""
        found = run(self.READY + """
            answers = {"/bootstrap": [200, {selection: {character: "c2", thread_id: "t2"}}],
                       "/snapshot": [200, {operation: null, messages: []}]};
            store.selection = {character: "c", thread: "t", epoch: "e"};
            store.check().then(settle).then(() => console.log(JSON.stringify({
                selection: [store.selection.character, store.selection.thread]})));
        """, sources=("store",))

        assert found["selection"] == ["c2", "t2"]

    def test_the_shell_looks_when_the_panel_opens_and_the_workspace_changes(self):
        shell = SHELL.read_text(encoding="utf-8")
        opened = shell.split("Shell.prototype.open = function", 1)[1].split("};", 1)[0]
        noted = shell.split("Shell.prototype.noteWorkspace = function", 1)[1].split("};", 1)[0]
        wiring = shell.split("Shell.prototype.wire = function", 1)[1] \
            .split("Shell.prototype.grow = function", 1)[0]
        handler = wiring.split('this.on(document, "visibilitychange"', 1)[1].split("});", 1)[0]

        assert "this.store.check();" in opened
        assert "this.store.check();" in noted
        assert "this.store.sleep();" in handler.split("return;", 1)[0]
        assert "this.store.wake();" in handler.split("return;", 1)[1]

    def test_a_closed_feed_is_not_a_connection_problem_on_screen(self):
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            shell.said = [];
            shell.say = (text) => shell.said.push(text);
            shell.renderStatus({ready: true, connected: false, everConnected: false,
                                idle: true, polling: false, error: "",
                                characters: ["x"], selection: {character: "x", thread: "t"},
                                conversation: {}});
            console.log(JSON.stringify({said: shell.said}));
        """, sources=("shell",))

        assert found["said"] and not any("onnect" in line for line in found["said"]), found["said"]


class TestARestartedServerIsRejoined:
    def test_a_refused_key_is_renewed_from_gradios_config(self):
        found = run(STORE_SETUP + """
            answers = {"/config": [200, {components: [
                {props: {elem_id: "something-else", value: "x"}},
                {props: {elem_id: "mc-llm-chat-conversation-key", value: "new-key"}}]}]};
            store.renewKey().then((renewed) => {
                console.log(JSON.stringify({renewed, key: store.key(), field: field.value,
                                            asked: calls}));
            });
        """, sources=("store",))

        assert found["renewed"] is True
        assert found["key"] == "new-key"
        assert found["field"] == "new-key"
        assert found["asked"] == ["/config"], (
            "Gradio's own config, behind Gradio's own login check -- not an "
            "endpoint of ours that would hand the key to anybody")

    def test_a_renewed_key_outlasts_the_field_being_redrawn(self):
        """If Gradio ever redraws the hidden field it writes its own state back
        -- the stale key. The renewal stands in for that value, and only that
        value: a field that changes to something else is newer than both."""
        found = run(STORE_SETUP + """
            answers = {"/config": [200, {components: [
                {props: {elem_id: "mc-llm-chat-conversation-key", value: "new-key"}}]}]};
            store.renewKey().then(() => {
                field.value = "old-key";
                const redrawn = store.key();
                field.value = "newer-still";
                console.log(JSON.stringify({redrawn, newer: store.key()}));
            });
        """, sources=("store",))

        assert found == {"redrawn": "new-key", "newer": "newer-still"}

    def test_a_renewal_that_finds_the_same_key_is_not_repeated(self):
        """Once per episode: a refusal the key does not explain is not worth a
        multi-megabyte config every ten seconds."""
        found = run(STORE_SETUP + """
            answers = {"/config": [200, {components: [
                {props: {elem_id: "mc-llm-chat-conversation-key", value: "old-key"}}]}]};
            (async () => {
                const first = await store.renewKey();
                store._renewedAt = 0;
                const second = await store.renewKey();
                console.log(JSON.stringify({first, second, fetches: calls.length}));
            })();
        """, sources=("store",))

        assert found == {"first": False, "second": False, "fetches": 1}

    def test_a_request_that_succeeds_ends_the_episode(self):
        found = run(STORE_SETUP + """
            answers = {"/config": [200, {components: [
                           {props: {elem_id: "mc-llm-chat-conversation-key", value: "old-key"}}]}],
                       "/snapshot": [200, {conversation: {}, messages: []}]};
            (async () => {
                await store.renewKey();
                const futile = store._renewFutile;
                await store.request("/snapshot?x=1");
                console.log(JSON.stringify({futile, after: store._renewFutile}));
            })();
        """, sources=("store",))

        assert found == {"futile": True, "after": False}

    def test_a_refused_request_renews_and_a_new_key_restarts_the_session(self):
        found = run(STORE_SETUP + """
            answers = {"/model-chain/conversation/v2/snapshot":
                           [401, {ok: false, error: {code: "AUTH_REQUIRED"}}],
                       "/config": [200, {components: [
                           {props: {elem_id: "mc-llm-chat-conversation-key", value: "new-key"}}]}]};
            let restarted = 0;
            store.restarted = () => { restarted += 1; return Promise.resolve(); };
            (async () => {
                await store.request("/snapshot?x=1").catch(() => null);
                await settle();
                console.log(JSON.stringify({restarted, key: store.key()}));
            })();
        """, sources=("store",))

        assert found == {"restarted": 1, "key": "new-key"}

    def test_a_refusal_the_key_does_not_explain_restarts_nothing(self):
        found = run(STORE_SETUP + """
            answers = {"/model-chain/conversation/v2/snapshot":
                           [401, {ok: false, error: {code: "AUTH_REQUIRED"}}],
                       "/config": [200, {components: [
                           {props: {elem_id: "mc-llm-chat-conversation-key", value: "old-key"}}]}]};
            let restarted = 0;
            store.restarted = () => { restarted += 1; return Promise.resolve(); };
            (async () => {
                await store.request("/snapshot?x=1").catch(() => null);
                await settle();
                console.log(JSON.stringify({restarted}));
            })();
        """, sources=("store",))

        assert found["restarted"] == 0

    def test_an_event_from_a_new_process_rejoins_instead_of_ending(self):
        """It used to set "Reload the page to carry on" and stop."""
        found = run(STORE_SETUP + """
            let restarted = 0;
            store.restarted = () => { restarted += 1; return Promise.resolve(); };
            store.serverEpoch = "one";
            store.apply({protocol_version: 2, server_epoch: "two", kind: "mode_changed",
                         payload: {mode: "chat"}});
            console.log(JSON.stringify({restarted, error: store.error}));
        """, sources=("store",))

        assert found["restarted"] == 1
        assert "Reload" not in found["error"]

    def test_a_restart_keeps_drafts_and_forgets_the_old_processs_state(self):
        found = run(STORE_SETUP + """
            let started = 0;
            let aborted = 0;
            store.start = () => { started += 1; return Promise.resolve(); };
            store.serverEpoch = "one";
            store.feed = "f";
            store.cursor = 41;
            store._abort = {abort() { aborted += 1; }};
            store.drafts.set("c\\u0000t", {text: "half a thought", attachment: null,
                                          draftVersion: 3});
            store.operations.set("op", {id: "op", key: "c\\u0000t", terminal: false});
            store._pending = {operationId: "op", envelope: {}, settled: false};
            store.restarted().then(() => {
                console.log(JSON.stringify({
                    started, aborted, epoch: store.serverEpoch, feed: store.feed,
                    cursor: store.cursor, operations: store.operations.size,
                    pending: store._pending,
                    draft: store.drafts.get("c\\u0000t").text,
                    commands: calls.filter((url) => url.indexOf("/commands") >= 0).length}));
            });
        """, sources=("store",))

        assert found["started"] == 1
        assert found["aborted"] == 1
        assert (found["epoch"], found["feed"], found["cursor"]) == ("", "", 0)
        assert found["operations"] == 0
        assert found["pending"] is None
        assert found["draft"] == "half a thought", "a draft was never the server's"
        assert found["commands"] == 0, (
            "an unacknowledged send is not retried into a process that may "
            "already have written it")

    def test_a_restart_already_under_way_is_not_started_twice(self):
        found = run(STORE_SETUP + """
            let started = 0;
            store.start = () => { started += 1; return new Promise(() => {}); };
            store.restarted();
            store.restarted();
            console.log(JSON.stringify({started}));
        """, sources=("store",))

        assert found["started"] == 1

    def test_a_refused_bootstrap_does_not_say_the_host_cannot_converse(self):
        """The key is being renewed; for those few hundred milliseconds the
        honest status is "connecting", not "unavailable on this host"."""
        found = run(STORE_SETUP + """
            answers = {"/bootstrap": [401, {ok: false, error: {code: "AUTH_REQUIRED"}}],
                       "/config": [500, {}]};
            const original = console.warn;
            console.warn = () => {};
            store.start().then(() => {
                console.warn = original;
                console.log(JSON.stringify({error: store.error, ready: store.ready}));
            });
        """, sources=("store",))

        assert found == {"error": "", "ready": False}
