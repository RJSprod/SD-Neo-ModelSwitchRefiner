"""WanGP's Generate, in the assistant's header.

Asked for from use: "When im on the wangp tab and wangp is loaded, i want the
ability to invoke whatever is on the page for generation ... a simple icon
button that essentially is generate when on the page, and if there is
something already being generated, it just adds the current to queue ... in
the flyout bar next to '...'".

Mini Paint NEO's `minipaintWanGP.generate()` is the whole of the mechanism
(its docs/wangp/CONTRACTS.md): the live page, nothing overridden, WanGP's own
Generate when idle and its Add to Queue when busy. These hold this side: where
the button is, when it shows, and what a press says.
"""

from __future__ import annotations

from test_assistant_js import run


WORLD = """
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push({fn, ms, live: true}); return timers.length; };
globalThis.clearTimeout = (id) => { if (timers[id - 1]) timers[id - 1].live = false; };
const settle = () => new Promise((resolve) => setImmediate(resolve));

function generateShell(active, bridge) {
    if (bridge === undefined) {
        bridge = {calls: 0, ready: true, answer: {ok: true, status: "started", route: "generate"},
                  state() { return {ready: this.ready, queue: true}; },
                  generate() { this.calls += 1; return Promise.resolve(this.answer); }};
    }
    globalThis.minipaintWanGP = bridge;
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {panelOpen: true};
    shell.activeWorkspace = active;
    shell.told = [];
    shell.tell = (text, kind) => shell.told.push([text, kind]);
    shell.nodes = {generate: element("generate", "BUTTON")};
    shell.nodes.generate.hidden = true;
    return {shell, bridge, button: shell.nodes.generate};
}
"""


def test_it_sits_next_to_the_more_actions_menu():
    found = run("""
        const shell = Object.create(NS.Shell.prototype);
        shell.settings = {label: "Forge Assistant", bubbleWidth: 80};
        shell.state = {conversationExpanded: false};
        shell.nodes = {root: {appendChild() {}}};
        shell.disposers = [];
        shell.host = {registerMenu: () => () => undefined,
                      getActiveWorkspace: () => "tab_txt2img", listWorkspaces: () => []};
        shell.place = () => undefined;
        shell.buildPanel();
        const row = shell.nodes.header.children;
        console.log(JSON.stringify({
            after: row[row.indexOf(shell.nodes.generate) - 1] === shell.nodes.utilities,
            label: shell.nodes.generate.getAttribute("aria-label"),
            hidden: shell.nodes.generate.hidden,
            glyph: shell.nodes.generate.textContent,
        }));
    """)

    assert found == {"after": True, "label": "Generate in WanGP", "hidden": True,
                     "glyph": "▶︎"}


def test_wangps_shows_on_the_wangp_tab_only():
    """WanGP's ▶: on its tab, with Mini Paint NEO there. (This Txt2img has no
    Generate button of its own, so nothing is offered there either; with one,
    the ▶ is that tab's -- below.)"""
    found = run(WORLD + """
        const there = generateShell("tab_wangp");
        there.shell.applyGenerate();
        const elsewhere = generateShell("tab_txt2img");
        elsewhere.shell.applyGenerate();
        const nobody = generateShell("tab_wangp", null);
        nobody.shell.applyGenerate();
        console.log(JSON.stringify([there.button.hidden, elsewhere.button.hidden,
                                    nobody.button.hidden]));
    """)

    assert found == [False, True, True], "shown on WanGP, hidden elsewhere and without Mini Paint"


def test_it_waits_for_wangp_to_be_loaded():
    found = run(WORLD + """
        const {shell, bridge, button} = generateShell("tab_wangp");
        bridge.ready = false;
        shell.applyGenerate();
        const before = {disabled: button.disabled, title: button.title};
        bridge.ready = true;
        shell.applyGenerate();
        console.log(JSON.stringify({before, after: button.disabled}));
    """)

    assert found == {"before": {"disabled": True, "title": "WanGP is not loaded yet"},
                     "after": False}


def test_a_press_generates_and_says_so_on_the_button():
    found = run(WORLD + """
        const {shell, bridge, button} = generateShell("tab_wangp");
        shell.applyGenerate();
        const pending = shell.generateInWanGP();
        const during = {state: button.dataset.state, disabled: button.disabled};
        await pending;
        console.log(JSON.stringify({calls: bridge.calls, during, state: button.dataset.state,
                                    title: button.title, told: shell.told,
                                    disabled: button.disabled}));
    """)

    assert found["calls"] == 1
    assert found["during"] == {"state": "busy", "disabled": False}, "a press never locks it"
    assert found["state"] == "done"
    assert found["title"] == "Generating in WanGP."
    assert found["told"] == [["Generating in WanGP.", "info"]]
    assert found["disabled"] is False


def test_every_press_counts_and_they_go_one_after_another():
    """Asked for: "I should be able to spam the button." The bridge holds
    WanGP's form for one request at a time (QUEUE_BUSY for a second), so
    presses line up here rather than racing each other."""
    found = run(WORLD + """
        const {shell, bridge, button} = generateShell("tab_wangp");
        shell.applyGenerate();
        let open = 0, most = 0;
        const answers = [];
        bridge.generate = function () {
            this.calls += 1;
            open += 1;
            most = Math.max(most, open);
            return new Promise((resolve) => answers.push(() => {
                open -= 1;
                resolve({ok: true, status: this.calls === 1 ? "started" : "queued"});
            }));
        };
        const presses = [shell.generateInWanGP(), shell.generateInWanGP(),
                         shell.generateInWanGP()];
        await settle();
        // The reading's beat looks at the button again while presses wait.
        shell.applyGenerate();
        const counted = {count: button.dataset.count, disabled: button.disabled,
                         sent: bridge.calls};
        while (answers.length) { answers.shift()(); await settle(); await settle(); }
        const results = await Promise.all(presses);
        console.log(JSON.stringify({counted, calls: bridge.calls, most,
                                    statuses: results.map((r) => r.status),
                                    count: button.dataset.count || "",
                                    state: button.dataset.state,
                                    title: button.title}));
    """)

    assert found["counted"] == {"count": "3", "disabled": False, "sent": 1}
    assert found["calls"] == 3, "every press reached WanGP"
    assert found["most"] == 1, "never two at once: the bridge would refuse the second"
    assert found["statuses"] == ["started", "queued", "queued"]
    assert found["count"] == ""
    assert found["state"] == "done"
    assert found["title"] == "Added to WanGP's queue."


def test_a_refused_press_does_not_stop_the_ones_behind_it():
    found = run(WORLD + """
        const {shell, bridge, button} = generateShell("tab_wangp");
        const answers = [{ok: false, status: "refused", message: "WanGP refused the request."},
                         {ok: true, status: "queued"}];
        bridge.generate = function () { this.calls += 1; return Promise.resolve(answers.shift()); };
        await Promise.all([shell.generateInWanGP(), shell.generateInWanGP()]);
        console.log(JSON.stringify({calls: bridge.calls, state: button.dataset.state,
                                    title: button.title}));
    """)

    assert found == {"calls": 2, "state": "failed", "title": "WanGP refused the request."}


def test_while_wangp_is_busy_it_joins_the_queue():
    found = run(WORLD + """
        const {shell, bridge, button} = generateShell("tab_wangp");
        bridge.answer = {ok: true, status: "queued", route: "queue"};
        await shell.generateInWanGP();
        console.log(JSON.stringify(button.title));
    """)

    assert found == "Added to WanGP's queue."


def test_a_refusal_is_said_in_wangp_s_words():
    found = run(WORLD + """
        const {shell, bridge, button} = generateShell("tab_wangp");
        bridge.answer = {ok: false, status: "refused", code: "WANGP_VALIDATION_REFUSED",
                         message: "WanGP refused the request."};
        await shell.generateInWanGP();
        const refused = {state: button.dataset.state, title: button.title,
                         told: shell.told.slice()};
        bridge.generate = () => { throw new Error("gone"); };
        await shell.generateInWanGP();
        refused.thrown = button.title;
        console.log(JSON.stringify(refused));
    """)

    assert found == {"state": "failed", "title": "WanGP refused the request.",
                     "told": [["WanGP refused the request.", "warn"]],
                     "thrown": "WanGP did not take it."}


def test_the_answer_leaves_the_button_after_a_while():
    found = run(WORLD + """
        const {shell, button} = generateShell("tab_wangp");
        await shell.generateInWanGP();
        const shown = timers.filter((timer) => timer.live);
        shown.forEach((timer) => timer.fn());
        console.log(JSON.stringify({ms: shown.map((timer) => timer.ms),
                                    state: button.dataset.state || "",
                                    title: button.title}));
    """)

    assert found == {"ms": [3000], "state": "", "title": "Generate in WanGP"}


def test_a_press_right_after_the_line_empties_is_not_lost():
    """The sender stops in the same turn it finds the line empty; a press that
    lands just after must start the next one, not wait on one that ended."""
    found = run(WORLD + """
        const {shell, bridge} = generateShell("tab_wangp");
        await shell.generateInWanGP();
        const second = shell.generateInWanGP();
        const timeout = new Promise((resolve) => setImmediate(() => setImmediate(
            () => resolve("lost"))));
        const won = await Promise.race([second.then(() => "answered"), timeout]);
        console.log(JSON.stringify({won, calls: bridge.calls}));
    """)

    assert found == {"won": "answered", "calls": 2}


# -- Txt2Img's and Img2Img's own Generate --------------------------------------- #
#
# Asked for: "for the flyout menu in both the text to image and image to image,
# I would like to have a play button, just like I do for WAN2GP, and its
# function should be to initiate generate". The same ▶ presses the tab's own
# Generate button, as Send to Generate does.

FORGE = """
function forgeTab(name) {
    const generate = element(name + "_generate", "BUTTON");
    const interrupt = element(name + "_interrupt", "BUTTON");
    return {generate, interrupt};
}
"""


def test_on_txt2img_and_img2img_it_is_the_tabs_own_generate():
    found = run(WORLD + FORGE + """
        forgeTab("txt2img");
        const here = generateShell("tab_txt2img", null);
        here.shell.applyGenerate();
        const missing = generateShell("tab_img2img", null);
        missing.shell.applyGenerate();
        const hidden = missing.button.hidden;
        forgeTab("img2img");
        missing.shell.applyGenerate();
        console.log(JSON.stringify({
            txt2img: [here.button.hidden, here.button.disabled,
                      here.button.getAttribute("aria-label"), here.button.title],
            img2img: [hidden, missing.button.hidden, missing.button.getAttribute("aria-label")],
        }));
    """)

    # No Mini Paint needed; a tab without its Generate button offers none.
    assert found["txt2img"] == [False, False, "Generate in Txt2img", "Generate in Txt2img"]
    assert found["img2img"] == [True, False, "Generate in Img2img"]


def test_a_press_on_a_generation_tab_presses_its_generate_button():
    found = run(WORLD + FORGE + """
        const tab = forgeTab("img2img");
        const {shell, bridge, button} = generateShell("tab_img2img");
        shell.applyGenerate();
        const answer = await shell.pressGenerate();
        console.log(JSON.stringify({clicks: tab.generate.clicks || 0, wangp: bridge.calls,
                                    answer, told: shell.told, state: button.dataset.state,
                                    title: button.title}));
    """)

    assert found["clicks"] == 1 and found["wangp"] == 0
    assert found["answer"] == {"ok": True, "message": "Generating in Img2img."}
    assert found["told"] == [["Generating in Img2img.", "info"]]
    assert found["state"] == "done" and found["title"] == "Generating in Img2img."


def test_a_press_while_the_tab_generates_presses_nothing_and_says_so():
    """Forge covers its Generate with Interrupt and Skip while a run is on, so
    a person cannot press it then, and neither does the ▶ -- it never locks,
    and says why it did nothing."""
    found = run(WORLD + FORGE + """
        const tab = forgeTab("txt2img");
        tab.interrupt.style.display = "block";
        const {shell, button} = generateShell("tab_txt2img", null);
        shell.applyGenerate();
        const disabled = button.disabled;
        await shell.pressGenerate();
        const refused = {clicks: tab.generate.clicks || 0, state: button.dataset.state,
                         title: button.title, told: shell.told.slice(), disabled};
        // Once the run is over the next press generates.
        tab.interrupt.style.display = "none";
        await shell.pressGenerate();
        console.log(JSON.stringify({refused, clicks: tab.generate.clicks || 0,
                                    state: button.dataset.state}));
    """)

    assert found["refused"] == {"clicks": 0, "state": "failed",
                                "title": "Txt2img is already generating.",
                                "told": [["Txt2img is already generating.", "warn"]],
                                "disabled": False}
    assert found["clicks"] == 1 and found["state"] == "done"


def test_wangps_line_is_not_painted_on_another_tabs_button():
    """One button for every tab: presses still in line for WanGP go on, but
    their count and their answer are WanGP's, and Txt2Img's ▶ shows neither;
    back on WanGP its line is painted again."""
    found = run(WORLD + FORGE + """
        forgeTab("txt2img");
        const {shell, bridge, button} = generateShell("tab_wangp");
        const answers = [];
        bridge.generate = function () {
            this.calls += 1;
            return new Promise((resolve) => answers.push(() => resolve({ok: true, status: "queued"})));
        };
        shell.applyGenerate();
        const presses = [shell.generateInWanGP(), shell.generateInWanGP(), shell.generateInWanGP()];
        await settle();
        const there = {state: button.dataset.state, count: button.dataset.count};
        shell.activeWorkspace = "tab_txt2img";
        shell.applyGenerate();
        const away = {state: button.dataset.state || "", count: button.dataset.count || "",
                      label: button.getAttribute("aria-label")};
        answers.shift()(); await settle(); await settle();
        const during = {state: button.dataset.state || "", count: button.dataset.count || ""};
        shell.activeWorkspace = "tab_wangp";
        shell.applyGenerate();
        const back = {state: button.dataset.state, count: button.dataset.count,
                      label: button.getAttribute("aria-label")};
        while (answers.length) { answers.shift()(); await settle(); await settle(); }
        await Promise.all(presses);
        console.log(JSON.stringify({there, away, during, back, calls: bridge.calls,
                                    done: button.dataset.state}));
    """)

    assert found["there"] == {"state": "busy", "count": "3"}
    assert found["away"] == {"state": "", "count": "", "label": "Generate in Txt2img"}
    assert found["during"] == {"state": "", "count": ""}
    assert found["back"] == {"state": "busy", "count": "2", "label": "Generate in WanGP"}
    assert found["calls"] == 3 and found["done"] == "done"


# The launcher's right click is ▶. Asked for: "the play /generate button for
# text to image, image to image, and wan2gp works well, I want to make a
# shortcut to this function, so when fly out menu is in button state
# minimized, then a right click should invoke generate."

LAUNCHER = """
function putAway(world) {
    world.shell.state.panelOpen = false;
    world.shell.nodes.launcher = element("forge-assistant-launcher", "BUTTON");
    world.shell.nodes.launcher.title = "Forge Assistant";
    return world.shell.nodes.launcher;
}
function rightClick(extra) {
    return Object.assign({pointerType: "mouse", prevented: false,
                          preventDefault() { this.prevented = true; }}, extra || {});
}
const runTimers = () => { const due = timers.filter((t) => t.live); due.forEach((t) => { t.live = false; t.fn(); }); };
"""


def test_a_right_click_on_the_put_away_launcher_presses_the_tabs_generate():
    found = run(WORLD + FORGE + LAUNCHER + """
        const tab = forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        const launcher = putAway(world);
        const event = rightClick();
        const taken = world.shell.launcherMenu(event);
        await settle();
        const said = {state: launcher.dataset.generated, title: launcher.title};
        runTimers();
        console.log(JSON.stringify({taken, prevented: event.prevented, clicks: tab.generate.clicks || 0,
                                    said, after: [launcher.dataset.generated === undefined, launcher.title],
                                    told: world.shell.told}));
    """)

    assert found["taken"] is True and found["prevented"] is True, "the browser's menu is not shown"
    assert found["clicks"] == 1
    assert found["said"] == {"state": "done", "title": "Generating in Txt2img."}
    assert found["after"] == [True, "Forge Assistant"], "its own title back after GENERATE_SHOWN"
    assert found["told"] == [["Generating in Txt2img.", "info"]]


def test_on_the_wangp_tab_it_presses_wangps_generate():
    found = run(WORLD + LAUNCHER + """
        const world = generateShell("tab_wangp");
        const launcher = putAway(world);
        world.bridge.answer = {ok: true, status: "queued"};
        world.shell.launcherMenu(rightClick());
        const during = launcher.dataset.generated;
        await settle(); await settle(); await settle();
        console.log(JSON.stringify({calls: world.bridge.calls, during,
                                    state: launcher.dataset.generated, title: launcher.title}));
    """)

    assert found == {"calls": 1, "during": "busy", "state": "done", "title": "Added to WanGP's queue."}


def test_a_refused_press_is_said_on_the_launcher():
    found = run(WORLD + FORGE + LAUNCHER + """
        const tab = forgeTab("img2img");
        tab.interrupt.style.display = "block";
        const world = generateShell("tab_img2img", null);
        const launcher = putAway(world);
        world.shell.launcherMenu(rightClick());
        await settle();
        const busy = {state: launcher.dataset.generated, title: launcher.title, clicks: tab.generate.clicks || 0};
        const wangp = generateShell("tab_wangp");
        const away = putAway(wangp);
        wangp.bridge.ready = false;
        const event = rightClick();
        wangp.shell.launcherMenu(event);
        console.log(JSON.stringify({busy, notLoaded: {state: away.dataset.generated, title: away.title,
                                                      prevented: event.prevented, calls: wangp.bridge.calls}}));
    """)

    assert found["busy"] == {"state": "failed", "title": "Img2img is already generating.", "clicks": 0}
    assert found["notLoaded"] == {"state": "failed", "title": "WanGP is not loaded yet",
                                  "prevented": True, "calls": 0}


def test_no_browsers_menu_on_the_launcher_even_where_there_is_nothing_to_generate():
    """"No menu needed either when I long hold or right click, just
    generate": the launcher's context menu is never the browser's. Where ▶
    has nothing to press it does nothing; with the panel open the launcher is
    not on screen and its menu is not asked about."""
    found = run(WORLD + FORGE + LAUNCHER + """
        forgeTab("txt2img");
        const elsewhere = generateShell("tab_settings", null);
        putAway(elsewhere);
        const one = rightClick();
        const a = elsewhere.shell.launcherMenu(one);
        const tab = forgeTab("img2img");
        const open = generateShell("tab_img2img", null);
        putAway(open);
        open.shell.state.panelOpen = true;
        const two = rightClick();
        const b = open.shell.launcherMenu(two);
        await settle();
        console.log(JSON.stringify({elsewhere: [a, one.prevented, elsewhere.shell.told.length],
                                    open: [b, two.prevented, tab.generate.clicks || 0]}));
    """)

    assert found == {"elsewhere": [False, True, 0], "open": [False, False, 0]}


def test_a_fingers_or_a_pens_long_press_is_a_right_click_too():
    """"On remote, that right click is a long press. So I need it to be
    correctly interpreted." A touchscreen's own long press arrives as the
    launcher's context menu -- with a touch or pen pointer, or as a plain
    mouse event from a browser that sends one -- and each is ▶. So is the
    keyboard's menu key."""
    found = run(WORLD + FORGE + LAUNCHER + """
        const tab = forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        putAway(world);
        const menus = [rightClick({pointerType: "touch"}), rightClick({pointerType: "pen"}),
                       rightClick({pointerType: undefined}), rightClick({pointerType: ""})];
        const results = menus.map((event) => world.shell.launcherMenu(event));
        await settle();
        console.log(JSON.stringify({results, prevented: menus.map((event) => event.prevented),
                                    clicks: tab.generate.clicks || 0}));
    """)

    assert found == {"results": [True, True, True, True], "prevented": [True, True, True, True],
                     "clicks": 4}


# The hold: "if I long hold on desktop, then it's the same function as
# generate. All in all, I want to collect as many as possible "long holds" so
# I can generate." A press held HOLD_MS on the put-away launcher without moving
# presses ▶ -- a finger, a pen, a mouse's left button -- and so does the
# context menu a held press brings; whichever comes first, once.

HOLD = """
function down(world, extra) {
    const event = Object.assign({pointerId: 1, pointerType: "touch", isPrimary: true, button: 0,
                                 clientX: 100, clientY: 100}, extra || {});
    world.shell.startHold(event);
    return event;
}
function move(world, dx, dy, extra) {
    world.shell.moveHold(Object.assign({pointerId: 1, pointerType: "touch", buttons: 1,
                                        clientX: 100 + dx, clientY: 100 + dy}, extra || {}));
}
function up(world, extra) {
    world.shell.letGoHold(Object.assign({pointerId: 1}, extra || {}));
}
// The hold's own time: GENERATE_SHOWN's, for the answer on the launcher and
// the button, run beside it.
const liveTimers = () => timers.filter((t) => t.live && t.ms !== 3000).map((t) => t.ms);
// The moment the ring shows, a tap's length into the press.
const ring = () => timers.filter((t) => t.live && t.ms === 120).forEach((t) => { t.live = false; t.fn(); });
"""


def test_a_hold_presses_generate_once_it_has_lasted_whatever_holds_it():
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        const tab = forgeTab("txt2img");
        const buzzed = [];
        Object.defineProperty(globalThis, "navigator", {configurable: true,
                                                        value: {vibrate: (ms) => buzzed.push(ms)}});
        const seen = [];
        for (const pointerType of ["touch", "pen", "mouse"]) {
            const world = generateShell("tab_txt2img", null);
            const launcher = putAway(world);
            const before = tab.generate.clicks || 0;
            down(world, {pointerType});
            const tap = {ring: launcher.dataset.holding !== undefined, timers: liveTimers()};
            ring();
            const holding = {tap, ring: launcher.dataset.holding !== undefined, timers: liveTimers(),
                             clicks: (tab.generate.clicks || 0) - before};
            runTimers();
            await settle();
            const felt = buzzed.splice(0);
            seen.push({pointerType, holding, felt, clicks: (tab.generate.clicks || 0) - before,
                       ring: launcher.dataset.holding !== undefined, said: launcher.dataset.generated,
                       spent: world.shell.suppressClick === true});
            up(world);
            runTimers();
        }
        console.log(JSON.stringify(seen));
    """)

    for seen in found:
        assert seen["holding"] == {"tap": {"ring": False, "timers": [120, 500]},
                                   "ring": True, "timers": [500], "clicks": 0}, seen
        assert seen["clicks"] == 1 and seen["ring"] is False and seen["said"] == "done", seen
        assert seen["spent"] is True, "the click its release brings opens nothing"
        # A finger is over the ring: it feels the hold taken. A mouse has nothing to feel with.
        assert seen["felt"] == ([] if seen["pointerType"] == "mouse" else [12]), seen


def test_the_menu_a_hold_brings_is_the_same_press_and_generates_nothing_more():
    """Whichever comes first: the browser's own long-press menu before the
    time, or after it (Windows sends a press-and-hold's on the release)."""
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        const tab = forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        putAway(world);
        // The browser's menu first: it is the hold, and the time is stopped.
        down(world);
        const early = rightClick({pointerType: "touch"});
        const took = world.shell.launcherMenu(early);
        const afterEarly = {clicks: tab.generate.clicks || 0, timers: liveTimers(), took,
                            prevented: early.prevented, spent: world.shell.suppressClick};
        runTimers();
        up(world);
        const late = rightClick({pointerType: "touch"});
        const second = world.shell.launcherMenu(late);
        // The time first, the menu on the release.
        down(world);
        runTimers();
        up(world);
        const onRelease = rightClick({pointerType: undefined});
        const third = world.shell.launcherMenu(onRelease);
        await settle();
        console.log(JSON.stringify({afterEarly, second: [second, late.prevented],
                                    third: [third, onRelease.prevented],
                                    clicks: tab.generate.clicks || 0}));
    """)

    assert found["afterEarly"] == {"clicks": 1, "timers": [], "took": True, "prevented": True,
                                   "spent": True}
    assert found["second"] == [False, True] and found["third"] == [False, True]
    assert found["clicks"] == 2, "one for each press"


def test_a_press_that_moves_first_is_a_drag_and_generates_nothing():
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        const tab = forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        const launcher = putAway(world);
        down(world);
        ring();
        move(world, 3, 4);
        const still = {timers: liveTimers(), ring: launcher.dataset.holding !== undefined};
        move(world, 0, 7);
        const moved = {timers: liveTimers(), ring: launcher.dataset.holding !== undefined};
        runTimers();
        // The browser's long press can still come under a drag that is short
        // of its own slop: it is the drag's.
        const menu = rightClick({pointerType: "touch"});
        const took = world.shell.launcherMenu(menu);
        up(world);
        await settle();
        console.log(JSON.stringify({still, moved, took, prevented: menu.prevented,
                                    clicks: tab.generate.clicks || 0,
                                    spent: !!world.shell.suppressClick}));
    """)

    assert found["still"] == {"timers": [500], "ring": True}, "a hand is never quite still"
    assert found["moved"] == {"timers": [], "ring": False}
    assert found["took"] is False and found["prevented"] is True
    assert found["clicks"] == 0 and found["spent"] is False


def test_a_press_let_go_in_time_is_a_press():
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        const tab = forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        const launcher = putAway(world);
        down(world, {pointerType: "mouse"});
        ring();
        up(world, {pointerType: "mouse"});
        const after = {timers: liveTimers(), ring: launcher.dataset.holding !== undefined,
                       spent: !!world.shell.suppressClick};
        runTimers();
        // A mouse's button let go where the page never heard it: the next
        // move says so, with no button down.
        down(world, {pointerType: "mouse"});
        move(world, 1, 1, {pointerType: "mouse", buttons: 0});
        const unheard = {timers: liveTimers(), down: world.shell.hold.down};
        runTimers();
        await settle();
        console.log(JSON.stringify({after, unheard, clicks: tab.generate.clicks || 0,
                                    said: launcher.dataset.generated || ""}));
    """)

    assert found == {"after": {"timers": [], "ring": False, "spent": False},
                     "unheard": {"timers": [], "down": False}, "clicks": 0, "said": ""}


def test_a_right_buttons_press_is_its_menus_and_a_second_finger_is_no_hold():
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        const tab = forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        const launcher = putAway(world);
        down(world, {pointerType: "mouse", button: 2});
        const right = {timers: liveTimers(), ring: launcher.dataset.holding !== undefined};
        const took = world.shell.launcherMenu(rightClick());
        up(world);
        await settle();
        const clicked = tab.generate.clicks || 0;
        down(world, {pointerId: 3});
        ring();
        down(world, {pointerId: 4, isPrimary: false});
        const two = {timers: liveTimers(), ring: launcher.dataset.holding !== undefined};
        runTimers();
        await settle();
        console.log(JSON.stringify({right, took, clicked, two, clicks: tab.generate.clicks || 0}));
    """)

    assert found["right"] == {"timers": [], "ring": False}
    assert found["took"] is True and found["clicked"] == 1
    assert found["two"] == {"timers": [], "ring": False}
    assert found["clicks"] == 1


def test_where_there_is_nothing_to_generate_a_hold_is_a_press():
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        forgeTab("txt2img");
        const world = generateShell("tab_settings", null);
        const launcher = putAway(world);
        down(world);
        const held = {timers: liveTimers(), ring: launcher.dataset.holding !== undefined};
        const menu = rightClick({pointerType: "touch"});
        const took = world.shell.launcherMenu(menu);
        console.log(JSON.stringify({held, took, prevented: menu.prevented,
                                    said: launcher.dataset.generated || ""}));
    """)

    assert found == {"held": {"timers": [], "ring": False}, "took": False, "prevented": True, "said": ""}


def test_on_the_wangp_tab_a_hold_is_wangps_generate_and_waits_for_wangp():
    found = run(WORLD + LAUNCHER + HOLD + """
        const world = generateShell("tab_wangp");
        const launcher = putAway(world);
        world.bridge.answer = {ok: true, status: "queued"};
        down(world);
        runTimers();
        await settle(); await settle(); await settle();
        const taken = {calls: world.bridge.calls, said: launcher.dataset.generated, title: launcher.title};
        up(world);
        const waiting = generateShell("tab_wangp");
        const away = putAway(waiting);
        waiting.bridge.ready = false;
        down(waiting);
        ring();
        const ringed = away.dataset.holding !== undefined;
        runTimers();
        console.log(JSON.stringify({taken, notLoaded: {ring: ringed, calls: waiting.bridge.calls,
                                                       said: away.dataset.generated, title: away.title}}));
    """)

    assert found["taken"] == {"calls": 1, "said": "done", "title": "Added to WanGP's queue."}
    assert found["notLoaded"] == {"ring": True, "calls": 0, "said": "failed",
                                  "title": "WanGP is not loaded yet"}


def test_a_hold_the_page_stopped_watching_generates_nothing():
    """The window lost, the page hidden: nobody is still holding."""
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        const tab = forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        const launcher = putAway(world);
        world.shell.resetHeaderTaps = () => undefined;
        down(world);
        ring();
        world.shell.cancelGestures();
        const after = {timers: liveTimers(), ring: launcher.dataset.holding !== undefined,
                       down: world.shell.hold.down};
        runTimers();
        await settle();
        console.log(JSON.stringify({after, clicks: tab.generate.clicks || 0}));
    """)

    assert found == {"after": {"timers": [], "ring": False, "down": False}, "clicks": 0}


def test_a_spent_press_whose_release_was_never_heard_is_over_at_the_next_press():
    """Let go over WanGP's frame, the release is the frame's. The next press
    anywhere ends it, and the keyboard's menu key after that is ▶ again."""
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        const tab = forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        putAway(world);
        down(world);
        runTimers();
        world.shell.supersede({isPrimary: true, target: null, clientX: 500, clientY: 500});
        const stillDown = world.shell.hold.down;
        world.shell.hold.upAt -= 5000;
        const key = rightClick({pointerType: ""});
        const took = world.shell.launcherMenu(key);
        await settle();
        console.log(JSON.stringify({down: stillDown, took, clicks: tab.generate.clicks || 0}));
    """)

    assert found == {"down": False, "took": True, "clicks": 2}


def test_the_keyboard_is_not_swallowed_after_a_hold():
    """A finger's hold leaves no click behind in most browsers to take its
    flag away, and Enter on the launcher afterwards is a press of its own."""
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        putAway(world);
        const taps = [];
        world.shell.tapLauncher = (event) => taps.push(event.detail);
        down(world);
        runTimers();
        up(world);
        const swallowed = [];
        world.shell.launcherClick({detail: 0, preventDefault() { swallowed.push("enter"); }});
        down(world, {pointerType: "mouse"});
        runTimers();
        up(world, {pointerType: "mouse"});
        world.shell.launcherClick({detail: 1, preventDefault() { swallowed.push("release"); }});
        console.log(JSON.stringify({taps, swallowed}));
    """)

    assert found == {"taps": [0], "swallowed": ["release"]}


def test_a_hold_ends_with_the_panel():
    found = run(WORLD + FORGE + LAUNCHER + HOLD + """
        const tab = forgeTab("txt2img");
        const world = generateShell("tab_txt2img", null);
        const launcher = putAway(world);
        Object.assign(world.shell, {disposers: [], stopMemory() {}, resetHeaderTaps() {}});
        down(world);
        ring();
        world.shell.dispose();
        const after = {timers: liveTimers(), ring: launcher.dataset.holding !== undefined};
        runTimers();
        await settle();
        console.log(JSON.stringify({after, clicks: tab.generate.clicks || 0}));
    """)

    assert found == {"after": {"timers": [], "ring": False}, "clicks": 0}
