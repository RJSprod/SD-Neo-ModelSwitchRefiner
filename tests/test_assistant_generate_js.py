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
