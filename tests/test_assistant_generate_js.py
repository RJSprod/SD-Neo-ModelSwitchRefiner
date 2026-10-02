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


def test_it_shows_on_the_wangp_tab_only():
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
        shell.generateInWanGP();
        await pending;
        console.log(JSON.stringify({calls: bridge.calls, during, state: button.dataset.state,
                                    title: button.title, told: shell.told,
                                    disabled: button.disabled}));
    """)

    assert found["calls"] == 1, "a second press while the first is asking adds nothing"
    assert found["during"] == {"state": "busy", "disabled": True}
    assert found["state"] == "done"
    assert found["title"] == "Generating in WanGP."
    assert found["told"] == [["Generating in WanGP.", "info"]]
    assert found["disabled"] is False


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
