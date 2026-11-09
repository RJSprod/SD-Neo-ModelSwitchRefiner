"""The Forge Assistant's ⋯ → Injection, executed under node.

"I like all three options, give it a menu for 'Injection'" (2026-11-09). The
submenu holds the open thread's words -- a box, Save, and the two ways they
steer, each a switch that keeps the box with it so nothing typed is lost to a
press -- how each word reads, and the model's control vector with its strength
and Apply. The store sends both commands against no revision.
"""

from __future__ import annotations

import json

from test_assistant_js import run

INJECTION = {"words": "rain +2\ntapestry -4", "odds": True, "prompt": False,
             "entries": [{"word": "rain", "strength": "+2", "banned": False,
                          "pieces": ["rain"], "note": ""},
                         {"word": "tapestry", "strength": "-4", "banned": False,
                          "pieces": ["tap", "estry"],
                          "note": "Split into tap · estry: only ‘tap’ is changed, "
                                  "and every word that starts with it."}],
             "left_out": ["“word65” is past the first 64 words"], "read": True,
             "note": "", "label": "Injection · 2 words · odds"}

VECTOR = {"model": "gemma.gguf", "width": 8, "layers": 12,
          "vectors": [{"name": "happy.gguf", "fits": True, "why": "", "layers": 11, "hint": ""},
                      {"name": "wide.gguf", "fits": False,
                       "why": "It was made for a model 16 wide; this one is 8.", "layers": 11,
                       "hint": ""}],
          "vector": "happy.gguf", "strength": 0.75, "first_layer": 4, "last_layer": 11,
          "state": "next_start", "said": "Takes effect when the model next starts.",
          "folder": "the control_vectors folder in the language model's data folder"}

SHELL = """
const made = (tag) => document.createElement(tag);
const shell = Object.create(NS.Shell.prototype);
shell.state = {freeFloat: false, autoAttach: false, sendToGenerate: false,
               conversationExpanded: true, panelOpen: true};
shell.host = {listUtilities: () => [], openOnly() {}};
let view = {selection: {character: "Ada", thread: "t1"},
            conversation: {conversation: {title: "Harbour", emotion: ""}, emotions: [],
                           threads: [{thread_id: "t1", title: "Harbour"}],
                           injection: __INJECTION__, control_vector: __VECTOR__}};
const sent = [];
const answers = {};
shell.store = {
    snapshot: () => view,
    setInjection(values) { sent.push(["set_injection", values]);
        return Promise.resolve(answers.set_injection || {ok: true, injection: view.conversation.injection}); },
    setControlVector(name, strength) { sent.push(["set_control_vector", name, strength]);
        return Promise.resolve(answers.set_control_vector || {ok: true, control_vector: {said: "Kept it."}}); },
    refresh() { sent.push(["refresh"]); return Promise.resolve(null); },
};
const told = [];
shell.tell = (text, kind) => told.push([text, kind]);
const menu = made("div");
menu.hidden = true;
Object.defineProperty(menu, "innerHTML", {get: () => "", set: () => { menu.children = []; }});
shell.nodes = {menu, picker: made("button"), utilities: made("button"),
               header: made("header"), status: made("p")};
shell.place = () => {};
shell.menuHandle = {};
const settle = async () => { for (let turn = 0; turn < 20; turn += 1) await Promise.resolve(); };
const press = (node) => (node.handlers.click || []).forEach((fn) => fn({preventDefault() {}}));
const has = (node, cls) => String(node.className || "").split(/\\s+/).indexOf(cls) >= 0;
const find = (cls) => menu.children.filter((c) => has(c, cls));
const deep = (node, cls, found) => {
    found = found || [];
    if (has(node, cls)) found.push(node);
    (node.children || []).forEach((child) => deep(child, cls, found));
    return found;
};
const everywhere = (cls) => menu.children.reduce((all, child) => deep(child, cls, all), []);
"""


def shell(scenario: str, injection=INJECTION, vector=VECTOR) -> dict:
    return run(SHELL.replace("__INJECTION__", json.dumps(injection))
               .replace("__VECTOR__", json.dumps(vector)) + scenario, sources=("shell",))


class TestTheEntry:
    def test_it_follows_context_in_the_chat_group_and_counts_the_words(self):
        found = shell("""
            const items = shell.utilityItems();
            const labels = items.map((item) => item.textContent);
            const entry = items.find((item) => has(item, "forge-assistant-injection-item"));
            console.log(JSON.stringify({labels: labels.slice(0, 7),
                value: entry.children[0].textContent, popup: entry.getAttribute("aria-haspopup"),
                disabled: entry.disabled}));
        """)

        assert found["labels"][4:6] == ["Context", "Injection"]
        assert found["value"] == "2 words"
        assert found["popup"] == "menu" and found["disabled"] is False

    def test_with_neither_way_on_it_says_off_and_with_no_thread_it_is_dark(self):
        found = shell("""
            view.conversation.injection.odds = false;
            const off = shell.injectionItem().children[0].textContent;
            view.selection.thread = "";
            const dark = shell.injectionItem().disabled;
            console.log(JSON.stringify({off, dark, empty: NS.injectionValue({entries: []})}));
        """)

        assert found == {"off": "2 words, off", "dark": True, "empty": ""}


class TestTheWords:
    def test_the_box_holds_the_words_and_each_way_is_a_switch_saying_its_state(self):
        found = shell("""
            shell.showMenu("injection");
            const box = find("forge-assistant-injection-words")[0];
            const ways = find("forge-assistant-injection-way");
            console.log(JSON.stringify({box: box.value, ways: ways.map((way) => [way.textContent,
                way.getAttribute("role"), way.getAttribute("aria-checked")]),
                save: everywhere("forge-assistant-injection-save").length}));
        """)

        assert found["box"] == "rain +2\ntapestry -4"
        assert found["ways"] == [["Weigh the odds", "menuitemcheckbox", "true"],
                                 ["Tell the model", "menuitemcheckbox", "false"]]
        assert found["save"] == 1

    def test_a_switch_keeps_what_was_typed_with_it(self):
        found = shell("""
            shell.showMenu("injection");
            find("forge-assistant-injection-words")[0].value = "ember +3";
            press(find("forge-assistant-injection-way")[1]);
            settle().then(() => console.log(JSON.stringify({sent, told})));
        """)

        assert found["sent"][0] == ["set_injection", {"words": "ember +3", "prompt": True}]
        assert found["sent"][1] == ["refresh"]
        assert found["told"] == [["2 words kept, weighing the odds.", "info"]]

    def test_save_keeps_the_box(self):
        found = shell("""
            shell.showMenu("injection");
            find("forge-assistant-injection-words")[0].value = "rain";
            press(everywhere("forge-assistant-injection-save")[0]);
            settle().then(() => console.log(JSON.stringify(sent)));
        """)

        assert found[0] == ["set_injection", {"words": "rain"}]

    def test_a_refusal_is_said_as_a_warning(self):
        found = shell("""
            answers.set_injection = {ok: false, error: {message: "That list is too long to keep."}};
            shell.showMenu("injection");
            press(everywhere("forge-assistant-injection-save")[0]);
            settle().then(() => console.log(JSON.stringify(told)));
        """)

        assert found == [["That list is too long to keep.", "warn"]]

    def test_each_word_reads_with_its_strength_a_split_one_says_so_as_plain_text(self):
        found = shell("""
            view.conversation.injection.entries[0].word = "<b>rain</b>";
            shell.showMenu("injection");
            const rows = everywhere("forge-assistant-injection-read")[0].children;
            const text = (node) => (node.textContent || "")
                + (node.children || []).map(text).join("");
            console.log(JSON.stringify(rows.map((row) => ({text: text(row),
                parts: (row.children || []).map((part) => part.className)}))));
        """)

        assert found[0] == {"text": "<b>rain</b>+2", "parts": [
            "forge-assistant-injection-word", "forge-assistant-injection-strength"]}
        assert found[1]["parts"][-1] == "forge-assistant-injection-split"
        assert "only ‘tap’ is changed" in found[1]["text"]
        assert found[2]["text"].startswith("Left out: ")

    def test_before_the_model_runs_it_says_the_words_are_read_later(self):
        found = shell("""
            view.conversation.injection.read = false;
            shell.showMenu("injection");
            console.log(JSON.stringify(everywhere("forge-assistant-injection-later")
                .map((row) => row.textContent)));
        """)

        assert found == ["Each word is read against the model's tokenizer once the model "
                         "is running."]

    def test_with_no_thread_open_it_asks_for_one(self):
        found = shell("""
            view.conversation.injection = null;
            shell.showMenu("injection");
            console.log(JSON.stringify(menu.children.map((child) => child.textContent)));
        """)

        assert "Choose a conversation to give it words." in found


class TestTheVector:
    def test_none_then_each_file_one_that_does_not_fit_disabled_and_saying_why(self):
        found = shell("""
            shell.showMenu("injection");
            const select = find("forge-assistant-vector-select")[0];
            console.log(JSON.stringify({value: select.value, options: select.children.map((o) =>
                [o.value, o.textContent, !!o.disabled, o.title || ""])}));
        """)

        assert found["value"] == "happy.gguf"
        assert found["options"] == [
            ["", "None", False, ""], ["happy.gguf", "happy.gguf", False, ""],
            ["wide.gguf", "wide.gguf (does not fit)", True,
             "It was made for a model 16 wide; this one is 8."]]

    def test_apply_sends_the_choice_and_its_strength_and_says_where_it_stands(self):
        found = shell("""
            shell.showMenu("injection");
            const select = find("forge-assistant-vector-select")[0];
            select.value = "";
            const slider = everywhere("forge-assistant-context-slider")[0];
            slider.value = "-1.5";
            press(everywhere("forge-assistant-vector-apply")[0]);
            settle().then(() => console.log(JSON.stringify({sent, told,
                said: everywhere("forge-assistant-vector-state").map((p) => [p.textContent,
                    p.dataset.state])})));
        """)

        assert found["sent"][0] == ["set_control_vector", "", -1.5]
        assert found["told"] == [["Kept it.", "info"]]
        assert found["said"] == [["Takes effect when the model next starts.", "next_start"]]

    def test_with_no_model_apply_is_dark_and_where_vectors_come_from_is_said(self):
        vector = dict(VECTOR, state="unavailable", said="Set up a language model first.")
        found = shell("""
            shell.showMenu("injection");
            console.log(JSON.stringify({dark: everywhere("forge-assistant-vector-apply")[0].disabled,
                notes: menu.children.map((child) => child.textContent)
                    .filter((text) => text.indexOf("llama-cvector-generator") >= 0)}));
        """, vector=vector)

        assert found["dark"] is True
        assert len(found["notes"]) == 1 and "Windows builds" in found["notes"][0]


class TestTheStore:
    def sent(self, call: str) -> list:
        return run("""
            const store = new NS.Store();
            store.selection = {character: "Ada", thread: "t1", epoch: "x"};
            const sent = [];
            store.send = (envelope) => { sent.push(envelope); return Promise.resolve({ok: true}); };
            store.__CALL__;
            Promise.resolve().then(() => console.log(JSON.stringify(sent.map((e) => [e.action,
                e.expected_revision, e.payload, e.conversation]))));
        """.replace("__CALL__", call), sources=("store",))

    def test_the_words_are_sent_against_no_revision_with_only_what_is_given(self):
        assert self.sent('setInjection({words: "rain", prompt: true})') == [
            ["set_injection", None, {"words": "rain", "prompt": True},
             {"character": "Ada", "thread_id": "t1"}]]
        assert self.sent("setInjection({odds: false, prompt: 'yes'})")[0][2] == {"odds": False}
        assert self.sent("setInjection({odds: 'no', prompt: true, words: 3})")[0][2] == {
            "prompt": True}

    def test_the_vector_is_sent_with_its_strength_and_none_is_empty(self):
        assert self.sent('setControlVector("happy.gguf", 0.5)')[0][:3] == [
            "set_control_vector", None, {"vector": "happy.gguf", "strength": 0.5}]
        assert self.sent("setControlVector(null)")[0][2] == {"vector": ""}
