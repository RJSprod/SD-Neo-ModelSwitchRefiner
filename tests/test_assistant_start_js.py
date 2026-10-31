"""The Forge Assistant's half of Start reply with, executed under node.

A button in the composer's row, beside the others, lit while the thread's next
reply has a start of yours and naming its words; a tap opens a box in the
conversation's place -- the words, then Clear and Set -- and Set, Enter or Clear
keep the change and give the message box the keyboard back (the server's
``mc_llm_start_with``). The reply that starts from the words spends them, and
the next snapshot puts the button out again.
"""

from __future__ import annotations

import json

from test_assistant_js import run

BLUE = "The color is blue"

BOX = """
const made = (tag) => document.createElement(tag);
const shell = Object.create(NS.Shell.prototype);
shell.state = {freeFloat: false, autoAttach: false, sendToGenerate: false,
               conversationExpanded: true, panelOpen: true};
shell.host = {listUtilities: () => [], openOnly() { shell.onlyOpened = true; }};
let view = {selection: {character: "Ada", thread: "t1"},
            conversation: {conversation: {title: "Harbour", start_with: __WORDS__}}};
const sent = [];
let answer = {ok: true};
shell.store = {
    snapshot: () => view,
    setStartWith(text) { sent.push(["set", text]); return Promise.resolve(answer); },
    refresh() { sent.push(["refresh"]); return Promise.resolve(null); },
};
const told = [];
shell.tell = (text, kind) => told.push([text, kind]);
const menu = made("div");
menu.hidden = true;
Object.defineProperty(menu, "innerHTML", {get: () => "", set: () => { menu.children = []; }});
const all = (node) => (node.children || []).flatMap((c) => [c].concat(all(c)));
const find = (name) => all(menu).filter((c) => (c.className || "").split(" ").indexOf(name) >= 0)[0];
menu.querySelector = (selector) => find(selector.replace(/^\\./, "")) || null;
const input = made("textarea");
shell.nodes = {menu, input, picker: made("button"), utilities: made("button"),
               header: made("header"), status: made("p")};
shell.place = () => {};
shell.menuHandle = {};
const press = (node) => node.handlers.click.forEach((fn) => fn());
const settle = async () => { for (let turn = 0; turn < 20; turn += 1) await Promise.resolve(); };
"""


def box(scenario: str, words: str = "") -> dict:
    return run(BOX.replace("__WORDS__", json.dumps(words)) + scenario, sources=("shell",))


class TestTheBox:
    def test_the_button_opens_it_with_the_words_kept_and_focused(self):
        found = box("""
            shell.openStart();
            const field = find("forge-assistant-start-input");
            console.log(JSON.stringify({which: menu.dataset.which, owner: menu.dataset.owner,
                hidden: menu.hidden, only: !!shell.onlyOpened,
                head: menu.children[0].textContent, value: field.value,
                focused: !!field.focused,
                buttons: [find("forge-assistant-start-clear").textContent,
                          find("forge-assistant-start-set").textContent],
                last: menu.children[menu.children.length - 1].textContent}));
        """, words=BLUE)

        assert found["which"] == "start" and found["owner"] == "utilities"
        assert found["hidden"] is False and found["only"] is True
        assert found["head"] == "Start the next reply with · Harbour"
        assert found["value"] == BLUE and found["focused"] is True
        assert found["buttons"] == ["Clear", "Set"]
        assert found["last"] == "Cancel"

    def test_set_keeps_the_words_closes_it_and_gives_the_message_box_the_keyboard(self):
        found = box("""
            shell.openStart();
            find("forge-assistant-start-input").value = "  The color is blue  ";
            press(find("forge-assistant-start-set"));
            await settle();
            console.log(JSON.stringify({sent, hidden: menu.hidden, focused: !!input.focused}));
        """)

        assert found["sent"] == [["set", BLUE], ["refresh"]]
        assert found["hidden"] is True and found["focused"] is True

    def test_enter_sets_it_and_shift_enter_is_a_new_line(self):
        found = box("""
            shell.openStart();
            const field = find("forge-assistant-start-input");
            field.value = "The color";
            let kept = false;
            field.handlers.keydown.forEach((fn) => fn({key: "Enter", shiftKey: true,
                preventDefault() { kept = true; }}));
            const afterShift = sent.length;
            let stopped = false;
            field.handlers.keydown.forEach((fn) => fn({key: "Enter", shiftKey: false,
                preventDefault() { stopped = true; }}));
            await settle();
            console.log(JSON.stringify({afterShift, kept, stopped, sent}));
        """)

        assert found["afterShift"] == 0 and found["kept"] is False
        assert found["stopped"] is True
        assert found["sent"][0] == ["set", "The color"]

    def test_clear_empties_it(self):
        found = box("""
            shell.openStart();
            press(find("forge-assistant-start-clear"));
            await settle();
            console.log(JSON.stringify({sent, hidden: menu.hidden}));
        """, words=BLUE)

        assert found["sent"][0] == ["set", ""]
        assert found["hidden"] is True

    def test_a_refusal_is_said_and_the_box_stays(self):
        found = box("""
            answer = {ok: false, error: {message: "A start is a sentence or two."}};
            shell.openStart();
            press(find("forge-assistant-start-set"));
            await settle();
            console.log(JSON.stringify({told, hidden: menu.hidden, sent}));
        """)

        assert found["told"] == [["A start is a sentence or two.", "warn"]]
        assert found["hidden"] is False
        assert ["refresh"] not in found["sent"]

    def test_without_a_thread_it_says_so_and_opens_nothing(self):
        found = box("""
            view = {selection: {character: "Ada", thread: ""}, conversation: null};
            shell.openStart();
            console.log(JSON.stringify({told, hidden: menu.hidden}));
        """)

        assert found["told"] == [["Choose a conversation first.", "warn"]]
        assert found["hidden"] is True


BUTTON = """
const button = document.createElement("button");
const shell = Object.create(NS.Shell.prototype);
shell.nodes = {start: button};
const look = (words, thread) => {
    shell.renderStart({selection: {thread: thread === undefined ? "t1" : thread},
                       conversation: {conversation: {start_with: words}}});
    return {pressed: button.getAttribute("aria-pressed"), label: button.getAttribute("aria-label"),
            title: button.title};
};
"""


class TestTheButton:
    def test_it_is_out_while_the_next_reply_starts_freely(self):
        found = run(BUTTON + "console.log(JSON.stringify(look('')));", sources=("shell",))

        assert found["pressed"] == "false"
        assert found["label"] == "Start the next reply with…"

    def test_it_is_lit_and_names_the_words_while_a_start_is_kept(self):
        found = run(BUTTON + f"console.log(JSON.stringify(look({json.dumps(BLUE)})));",
                    sources=("shell",))

        assert found == {"pressed": "true", "label": f"The next reply starts with: {BLUE}",
                         "title": f"The next reply starts with: {BLUE}"}

    def test_with_no_thread_it_is_out(self):
        found = run(BUTTON + f"console.log(JSON.stringify(look({json.dumps(BLUE)}, '')));",
                    sources=("shell",))

        assert found["pressed"] == "false"

    def test_it_goes_out_again_once_the_reply_has_spent_the_words(self):
        found = run(BUTTON + f"""
            look({json.dumps(BLUE)});
            console.log(JSON.stringify(look('')));
        """, sources=("shell",))

        assert found["pressed"] == "false"

    def test_it_is_written_only_when_it_changed(self):
        found = run(BUTTON + f"""
            look({json.dumps(BLUE)});
            let writes = 0;
            const set = button.setAttribute.bind(button);
            button.setAttribute = (name, value) => {{ writes += 1; set(name, value); }};
            look({json.dumps(BLUE)}); look({json.dumps(BLUE)});
            console.log(JSON.stringify({{writes}}));
        """, sources=("shell",))

        assert found["writes"] == 0

    def test_it_sits_in_the_composer_s_row_before_send(self):
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
            const row = shell.nodes.toolbar;
            const names = row ? row.children.map((c) => c.className) : [];
            console.log(JSON.stringify({names, glyph: shell.nodes.start
                ? shell.nodes.start.innerHTML.indexOf("<svg") === 0 : null}));
        """)

        names = found["names"]
        at = [index for index, name in enumerate(names) if "forge-assistant-start" in name]
        send = [index for index, name in enumerate(names) if "forge-assistant-send" in name]
        assert at and send and at[0] == send[0] - 1
        assert found["glyph"] is True


class TestTheStore:
    def test_it_sends_set_start_with_against_no_revision(self):
        found = run(f"""
            const store = new NS.Store();
            store.selection = {{character: "Ada", thread: "t1", epoch: "x"}};
            const sent = [];
            store.send = (envelope) => {{ sent.push(envelope); return Promise.resolve({{ok: true}}); }};
            store.setStartWith({json.dumps(BLUE)});
            console.log(JSON.stringify(sent.map((e) => [e.action, e.expected_revision,
                                                        e.payload, e.conversation])));
        """, sources=("store",))

        assert found == [["set_start_with", None, {"start_with": BLUE},
                          {"character": "Ada", "thread_id": "t1"}]]
