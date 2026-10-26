"""WanGP's prompt in the panel: the third state, on the WanGP tab.

Asked for from use: "I like how text to image has an option in the flyout menu
that pulls in the entire configuration into the flyout. I want a very simple
version of this, but for wan2gp. when I'm on that tab for generating content, I
would like the ability to directly modify the prompt from the flyout menu just
like image generation in text to image, but just the prompt."

Mini Paint NEO's `minipaintWanGP.prompt()` and `setPrompt()` (its
docs/wangp/CONTRACTS.md, bridge 1.14.0) are the mechanism: the box WanGP
generates from, read and written as WanGP's own value. These hold the panel's
side under node -- when the button offers it, what each press shows, when the
box reads WanGP's page and when it writes, that ▶ waits for what was typed,
and what the box says when it cannot be used. That a real browser draws the
box in the panel and types into it is test_assistant_wangp_prompt.py.
"""

from __future__ import annotations

from test_assistant_js import run


WORLD = """
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push({fn, ms, live: true}); return timers.length; };
globalThis.clearTimeout = (id) => { if (timers[id - 1]) timers[id - 1].live = false; };
const settle = async () => { for (let i = 0; i < 12; i += 1) await new Promise((r) => setImmediate(r)); };
// Every timer that is due, as the quiet after the last keystroke ending.
const elapse = async () => {
    for (const timer of timers) { if (timer.live) { timer.live = false; timer.fn(); } }
    await settle();
};

function fakeWanGP(over) {
    const bridge = Object.assign({
        ready: true, offered: true, touched: 0, page: "the page's own prompt", wizard: false,
        calls: [], hold: false, held: [], refuse: "",
        state() { return {ready: this.ready, queue: true,
                          prompt: {offered: this.offered, touched: this.touched}}; },
        answer(value) {
            if (!this.hold) return Promise.resolve(value);
            return new Promise((resolve) => this.held.push(() => resolve(value)));
        },
        release() { const waiting = this.held.splice(0); waiting.forEach((go) => go()); },
        prompt() {
            this.calls.push(["prompt"]);
            return this.answer({ok: true, prompt: this.page, wizard: this.wizard, written: false});
        },
        setPrompt(text) {
            this.calls.push(["setPrompt", text]);
            if (this.refuse) {
                return this.answer({ok: false, code: this.refuse,
                                    message: "This WanGP bridge needs to be reinstalled or updated."});
            }
            this.page = text;
            return this.answer({ok: true, prompt: text, wizard: this.wizard, written: true});
        },
        generate() {
            this.calls.push(["generate", this.page]);
            return Promise.resolve({ok: true, status: "started", route: "generate"});
        },
    }, over || {});
    globalThis.minipaintWanGP = bridge;
    return bridge;
}

function promptShell(active, state, bridge) {
    if (bridge === undefined) bridge = fakeWanGP();
    else if (bridge) globalThis.minipaintWanGP = bridge;
    else delete globalThis.minipaintWanGP;
    const dock = {placed: 0, released: 0,
                  available(id) { return id === "tab_txt2img"; },
                  place() { this.placed += 1; return true; },
                  release() { this.released += 1; return true; }};
    NS.dock = dock;
    const shell = Object.create(NS.Shell.prototype);
    shell.settings = {label: "Forge Assistant", bubbleWidth: 80};
    shell.state = Object.assign({panelOpen: true, conversationExpanded: true,
                                 settingsDocked: false, wangpPrompt: false}, state || {});
    shell.nodes = {root: {appendChild() {}}};
    shell.disposers = [];
    shell.host = {registerMenu: () => () => undefined, openOnly() {},
                  getActiveWorkspace: () => shell.activeWorkspace,
                  listWorkspaces: () => [{id: "tab_wangp", label: "WanGP", available: true}]};
    shell.activeWorkspace = active;
    shell.saved = 0;
    shell._save = () => { shell.saved += 1; };
    shell.place = () => undefined;
    shell.told = [];
    shell.tell = (text, kind) => shell.told.push([text, kind]);
    shell.buildPanel();
    return {shell, bridge, dock};
}

function view(shell) {
    const nodes = shell.nodes;
    return {button: !nodes.settings.hidden, label: nodes.settings.getAttribute("aria-label"),
            pressed: nodes.settings["aria-pressed"], box: !nodes.wangp.hidden,
            body: !nodes.body.hidden, row: !nodes.workspaces.hidden, slot: !nodes.dock.hidden,
            picker: !nodes.picker.hidden,
            prompting: nodes.panel.classList.contains("forge-assistant-prompting"),
            docked: nodes.panel.classList.contains("forge-assistant-docked")};
}

function typed(shell, text) {
    shell.nodes.prompt.value = text;
    shell.typedPrompt();
}
"""


def test_on_the_wangp_tab_the_settings_button_is_wangps_prompt():
    found = run(WORLD + """
        const here = promptShell("tab_wangp").shell;
        const old = promptShell("tab_wangp", {}, {state() { return {ready: true}; },
                                                  generate() {}}).shell;
        const none = promptShell("tab_wangp", {}, null).shell;
        const txt = promptShell("tab_txt2img").shell;
        console.log(JSON.stringify({here: view(here), old: view(old).button, none: view(none).button,
                                    txt: [view(txt).button, view(txt).label]}));
    """)

    assert found["here"]["button"] is True
    assert found["here"]["label"] == "WanGP prompt"
    assert found["here"]["pressed"] == "false"
    assert found["old"] is False, "a Mini Paint NEO without prompt() and setPrompt() offers nothing"
    assert found["none"] is False
    assert found["txt"] == [True, "Generation settings"]


def test_a_press_shows_wangps_prompt_in_the_conversations_place_and_reads_it():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge, dock} = promptShell("tab_wangp");
            shell.toggleDock();
            await settle();
            console.log(JSON.stringify({view: view(shell), state: shell.state.wangpPrompt,
                                        docked: shell.state.settingsDocked, saved: shell.saved,
                                        value: shell.nodes.prompt.value, calls: bridge.calls,
                                        disabled: !!shell.nodes.prompt.disabled,
                                        note: shell.nodes.promptNote.hidden, placed: dock.placed}));
        })();
    """)

    assert found["state"] is True and found["docked"] is False
    assert found["view"] == {"button": True, "label": "WanGP prompt", "pressed": "true",
                             "box": True, "body": False, "row": False, "slot": False,
                             "picker": True, "prompting": True, "docked": False}
    assert found["calls"] == [["prompt"]]
    assert found["value"] == "the page's own prompt"
    assert found["disabled"] is False and found["note"] is True
    assert found["placed"] == 0, "nothing is drawn under the panel on the WanGP tab"
    assert found["saved"] == 1


def test_a_second_press_and_chat_go_back_to_the_view_it_replaced():
    found = run(WORLD + """
        (async () => {
            const talking = promptShell("tab_wangp");
            talking.shell.toggleDock(); await settle();
            talking.shell.toggleDock(); await settle();
            const tabs = promptShell("tab_wangp", {conversationExpanded: false});
            tabs.shell.toggleDock(); await settle();
            tabs.shell.toggleDock(); await settle();
            const chat = promptShell("tab_wangp", {conversationExpanded: false});
            chat.shell.toggleDock(); await settle();
            chat.shell.toggleChat();
            console.log(JSON.stringify({talking: view(talking.shell), tabs: view(tabs.shell),
                                        chat: view(chat.shell), chatState: chat.shell.state.wangpPrompt,
                                        expanded: chat.shell.state.conversationExpanded}));
        })();
    """)

    assert found["talking"]["body"] is True and found["talking"]["box"] is False
    assert found["talking"]["pressed"] == "false"
    assert found["tabs"]["row"] is True and found["tabs"]["box"] is False
    assert found["chat"]["body"] is True and found["chat"]["box"] is False
    assert found["chatState"] is False and found["expanded"] is True


def test_the_view_is_the_wangp_tabs_alone_and_txt2imgs_column_is_its_own():
    found = run(WORLD + """
        (async () => {
            const {shell, dock} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            const there = view(shell);
            shell.activeWorkspace = "tab_txt2img";
            shell.applyDock();
            const txt = view(shell);
            shell.toggleDock();
            const dockedThere = {settings: shell.state.settingsDocked, prompt: shell.state.wangpPrompt};
            shell.activeWorkspace = "tab_wangp";
            shell.applyDock();
            const back = view(shell);
            console.log(JSON.stringify({there, txt, dockedThere, back}));
        })();
    """)

    assert found["there"]["box"] is True
    assert found["txt"]["box"] is False and found["txt"]["body"] is True
    assert found["txt"]["label"] == "Generation settings" and found["txt"]["pressed"] == "false"
    assert found["dockedThere"] == {"settings": True, "prompt": True}, \
        "the button on Txt2Img docks Txt2Img's column and leaves WanGP's view alone"
    assert found["back"]["box"] is True and found["back"]["label"] == "WanGP prompt"


def test_typing_is_written_to_wangp_once_the_keys_go_quiet():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            bridge.calls.length = 0;
            typed(shell, "a fox");
            typed(shell, "a fox at dawn");
            await settle();
            const before = bridge.calls.slice();
            const quiet = timers.filter((t) => t.live).map((t) => t.ms);
            await elapse();
            console.log(JSON.stringify({before, quiet, after: bridge.calls, page: bridge.page,
                                        note: shell.nodes.promptNote.hidden}));
        })();
    """)

    assert found["before"] == [], "nothing is sent while the keys are still going"
    assert found["quiet"] == [400]
    assert found["after"] == [["setPrompt", "a fox at dawn"]], "one write, of the whole box"
    assert found["page"] == "a fox at dawn"
    assert found["note"] is True


def test_leaving_the_box_writes_at_once():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            bridge.calls.length = 0;
            typed(shell, "a heron");
            shell.writePrompt();
            await settle();
            const now = bridge.calls.slice();
            await elapse();
            console.log(JSON.stringify({now, later: bridge.calls}));
        })();
    """)

    assert found["now"] == [["setPrompt", "a heron"]]
    assert found["later"] == found["now"], "the quiet's own write finds nothing left to send"


def test_text_typed_while_a_write_is_on_its_way_is_written_after_it():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            bridge.calls.length = 0;
            bridge.hold = true;
            typed(shell, "one");
            shell.writePrompt();
            typed(shell, "one two");
            shell.writePrompt();
            await settle();
            const during = bridge.calls.slice();
            bridge.release(); await settle();
            bridge.release(); await settle();
            console.log(JSON.stringify({during, after: bridge.calls, page: bridge.page}));
        })();
    """)

    assert found["during"] == [["setPrompt", "one"]], "one write at a time"
    assert found["after"] == [["setPrompt", "one"], ["setPrompt", "one two"]]
    assert found["page"] == "one two"


def test_generate_waits_for_what_was_typed_and_then_presses_wangps_generate():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            bridge.calls.length = 0;
            typed(shell, "a crane in the reeds");
            const answer = await shell.generateInWanGP();
            console.log(JSON.stringify({calls: bridge.calls, answer}));
        })();
    """)

    assert found["calls"] == [["setPrompt", "a crane in the reeds"],
                              ["generate", "a crane in the reeds"]]
    assert found["answer"]["ok"] is True


def test_a_prompt_wangp_did_not_take_stops_the_press():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            bridge.calls.length = 0;
            bridge.refuse = "BRIDGE_COMPONENT_INCOMPATIBLE";
            typed(shell, "a crane");
            const answer = await shell.generateInWanGP();
            console.log(JSON.stringify({calls: bridge.calls, answer, told: shell.told,
                                        note: shell.nodes.promptNote.textContent,
                                        dirty: !!shell._promptDirty}));
        })();
    """)

    assert ["generate", "the page's own prompt"] not in found["calls"], \
        "WanGP is not asked to generate with the prompt the box replaced"
    assert [call[0] for call in found["calls"]] == ["setPrompt"]
    assert found["answer"]["ok"] is False
    assert found["told"][-1] == ["WanGP's prompt was not updated: This WanGP bridge needs to be "
                                 "reinstalled or updated.", "warn"]
    assert found["note"] == "This WanGP bridge needs to be reinstalled or updated."
    assert found["dirty"] is True, "kept as unsent, for the next try"


def test_with_nothing_typed_generate_asks_nothing_first():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            bridge.calls.length = 0;
            await shell.generateInWanGP();
            console.log(JSON.stringify(bridge.calls));
        })();
    """)

    assert found == [["generate", "the page's own prompt"]]


def test_a_read_never_lands_over_text_wangp_has_not_got():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp");
            bridge.hold = true;
            shell.toggleDock();                      // the read is on its way
            typed(shell, "what I typed first");     // and the box is typed in
            bridge.release(); await settle();
            const box = shell.nodes.prompt.value;
            console.log(JSON.stringify({box}));
        })();
    """)

    assert found["box"] == "what I typed first"


def test_wangps_page_is_read_again_only_when_its_form_was_touched_by_hand():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            const first = bridge.calls.length;
            shell.applyPrompt(); shell.applyPrompt(); await settle();
            const beats = bridge.calls.length - first;
            bridge.touched += 1;
            bridge.page = "typed on WanGP's own page";
            shell.applyPrompt(); await settle();
            const afterTouch = {asked: bridge.calls.length - first, box: shell.nodes.prompt.value};
            shell.applyPrompt(); await settle();
            const again = bridge.calls.length - first;
            console.log(JSON.stringify({beats, afterTouch, again}));
        })();
    """)

    assert found["beats"] == 0, "a beat with nothing new asks nothing"
    assert found["afterTouch"] == {"asked": 1, "box": "typed on WanGP's own page"}
    assert found["again"] == 1


def test_shown_again_it_reads_afresh():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            shell.activeWorkspace = "tab_txt2img"; shell.applyDock(); await settle();
            bridge.page = "changed while the panel showed something else";
            shell.activeWorkspace = "tab_wangp"; shell.applyDock(); await settle();
            console.log(JSON.stringify({asked: bridge.calls.filter((c) => c[0] === "prompt").length,
                                        box: shell.nodes.prompt.value}));
        })();
    """)

    assert found == {"asked": 2, "box": "changed while the panel showed something else"}


def test_put_away_with_something_typed_it_is_written_first():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            bridge.calls.length = 0;
            typed(shell, "half a thought");
            shell.activeWorkspace = "tab_txt2img"; shell.applyDock(); await settle();
            console.log(JSON.stringify(bridge.calls));
        })();
    """)

    assert found == [["setPrompt", "half a thought"]]


def test_the_box_says_why_it_is_shut_and_asks_nothing_meanwhile():
    found = run(WORLD + """
        (async () => {
            const loading = promptShell("tab_wangp", {wangpPrompt: true}, fakeWanGP({ready: false}));
            await settle();
            const old = promptShell("tab_wangp", {wangpPrompt: true}, fakeWanGP({offered: false}));
            await settle();
            const say = (s) => ({disabled: !!s.shell.nodes.prompt.disabled,
                                 note: s.shell.nodes.promptNote.textContent,
                                 shown: !s.shell.nodes.promptNote.hidden, asked: s.bridge.calls.length});
            const before = say(loading);
            globalThis.minipaintWanGP = loading.bridge;
            loading.bridge.ready = true;
            loading.shell.applyPrompt(); await settle();
            console.log(JSON.stringify({loading: before, old: say(old), ready: say(loading),
                                        value: loading.shell.nodes.prompt.value}));
        })();
    """)

    assert found["loading"] == {"disabled": True, "note": "WanGP is not loaded yet.",
                                "shown": True, "asked": 0}
    assert found["old"] == {"disabled": True, "shown": True, "asked": 0,
                            "note": "Reinstall WanGP's bridge from the WanGP tab to edit its "
                                    "prompt here."}
    assert found["ready"] == {"disabled": False, "note": "", "shown": False, "asked": 1}
    assert found["value"] == "the page's own prompt"


def test_the_wizards_box_is_named_and_a_refused_read_shuts_the_box_until_wangp_is_touched():
    found = run(WORLD + """
        (async () => {
            const wizard = promptShell("tab_wangp", {wangpPrompt: true}, fakeWanGP({wizard: true}));
            await settle();
            const bridge = fakeWanGP();
            bridge.prompt = function () {
                this.calls.push(["prompt"]);
                return Promise.resolve({ok: false, code: "PROMPT_TOO_LONG", message: "too long"});
            };
            const long = promptShell("tab_wangp", {wangpPrompt: true}, bridge);
            await settle();
            long.shell.applyPrompt(); await settle();
            console.log(JSON.stringify({wizard: wizard.shell.nodes.promptNote.textContent,
                                        long: {disabled: !!long.shell.nodes.prompt.disabled,
                                               note: long.shell.nodes.promptNote.textContent,
                                               asked: bridge.calls.length}}));
        })();
    """)

    assert found["wizard"] == "WanGP's prompt wizard is on: this is its box."
    assert found["long"] == {"disabled": True, "asked": 1,
                             "note": "WanGP's prompt is too long to edit here; edit it on the "
                                     "WanGP page."}


def test_ctrl_enter_in_the_box_is_generate():
    found = run(WORLD + """
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp", {wangpPrompt: true});
            await settle();
            shell.nodes.generate.hidden = false;
            bridge.calls.length = 0;
            typed(shell, "a wren");
            let stopped = 0;
            const key = (over) => Object.assign({key: "Enter", ctrlKey: false, metaKey: false,
                                                 preventDefault() { stopped += 1; },
                                                 stopPropagation() {}}, over);
            shell.promptKey(key({}));                 // a new line, for WanGP's one prompt per line
            await settle();
            const plain = bridge.calls.slice();
            shell.promptKey(key({ctrlKey: true}));
            await settle();
            console.log(JSON.stringify({plain, calls: bridge.calls, stopped}));
        })();
    """)

    assert found["plain"] == []
    assert found["calls"] == [["setPrompt", "a wren"], ["generate", "a wren"]]
    assert found["stopped"] == 1


def test_the_view_is_remembered_with_the_rest_of_the_panels_layout():
    found = run(WORLD + """
        // The host's half, which these scenarios do not load.
        if (typeof NS.basePath !== "function") NS.basePath = () => "/";
        const {shell} = promptShell("tab_wangp");
        delete shell._save;
        shell.state.wangpPrompt = true;
        shell._save();
        const key = Object.keys(sessionStorage.store)[0];
        const stored = JSON.parse(sessionStorage.store[key]);
        const again = promptShell("tab_wangp").shell;
        again._restore();
        console.log(JSON.stringify({stored: stored.wangpPrompt, restored: again.state.wangpPrompt}));
    """)

    assert found == {"stored": True, "restored": True}


def test_no_live_connection_nothing_opened_nothing_polled_nothing_left_waiting():
    """Asked for once it was built: "Make sure there is no live stream. No
    live connections!" Every read and write is a call into Mini Paint NEO's
    bundle, which carries it to WanGP's frame as a message; the panel opens
    no stream, makes no request and sets no repeating timer for it, and once
    the box has been typed in, written, generated from, read again and put
    away, no timer of its own is left waiting."""
    found = run(WORLD + """
        const opened = [];
        globalThis.fetch = (url) => { opened.push(["fetch", String(url)]);
                                      return Promise.reject(new Error("no network")); };
        globalThis.EventSource = function (url) { opened.push(["eventsource", String(url)]); };
        globalThis.WebSocket = function (url) { opened.push(["websocket", String(url)]); };
        globalThis.XMLHttpRequest = function () { opened.push(["xhr", ""]); };
        globalThis.setInterval = (fn, ms) => { opened.push(["interval", ms]); return 1; };
        (async () => {
            const {shell, bridge} = promptShell("tab_wangp");
            shell.nodes.generate.hidden = false;
            shell.toggleDock(); await settle();
            typed(shell, "a fox"); await elapse();
            typed(shell, "a fox at dawn"); shell.writePrompt(); await settle();
            typed(shell, "a fox at dawn, snow"); await shell.generateInWanGP();
            shell.promptKey({key: "Enter", ctrlKey: true, preventDefault() {}, stopPropagation() {}});
            await settle();
            bridge.touched += 1; shell.applyPrompt(); await settle();
            shell.applyPrompt(); shell.applyPrompt(); await settle();
            typed(shell, "put away with this"); shell.toggleDock(); await settle();
            // Read before any timer is let run: what is still set now is what
            // the box left behind.
            console.log(JSON.stringify({opened, waiting: timers.filter((t) => t.live).map((t) => t.ms),
                                        calls: bridge.calls.map((c) => c[0])}));
        })();
    """)

    assert found["opened"] == [], found["opened"]
    # ▶'s own three seconds of saying how the press went is the button's, a
    # one-off; nothing of the box's is left.
    assert found["waiting"] == [3000], "no timer of the box's own is left behind"
    assert found["calls"] == ["prompt", "setPrompt", "setPrompt", "setPrompt", "generate",
                              "generate", "prompt", "setPrompt"]
