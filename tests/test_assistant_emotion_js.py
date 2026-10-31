"""The Forge Assistant's half of Character Emotion, executed under node.

The ⋯ menu's Chat group gains Emotion, a submenu like Threads: Off and every
line of the shared list, the thread's own checked, each line with Delete and a
box to add one, which is then chosen. While the thread has one, a pill in the
composer's row says its word and opens that submenu in a tap. After a reply
written with an emotion, the status line says *Updating cache…* quietly while
the reply is read again, and the page keeps its feed open until that is done
(the server's ``mc_llm_cache_update``); a message sent meanwhile says it is
waiting in its own progress, which is the operation's status and not this.
"""

from __future__ import annotations

from test_assistant_js import run

HAPPY = "*I am so happy*"

MENU = """
const made = (tag) => document.createElement(tag);
const shell = Object.create(NS.Shell.prototype);
shell.state = {freeFloat: false, autoAttach: false, sendToGenerate: false,
               conversationExpanded: true, panelOpen: true};
shell.host = {listUtilities: () => [], openOnly() { shell.onlyOpened = true; }};
let view = {selection: {character: "Ada", thread: "t1"},
            conversation: {conversation: {title: "Harbour", emotion: __CHOSEN__},
                           emotions: ["*I am so happy*", "*I am so sad*"],
                           threads: [{thread_id: "t1", title: "Harbour"}]}};
const sent = [];
shell.store = {
    snapshot: () => view,
    setEmotion(line) { sent.push(["set", line]); return Promise.resolve({ok: true}); },
    changeEmotions(change) { sent.push(["change", change]); return Promise.resolve({ok: true}); },
    refresh() { sent.push(["refresh"]); return Promise.resolve(null); },
};
const told = [];
shell.tell = (text, kind) => told.push([text, kind]);
const menu = made("div");
menu.hidden = true;
Object.defineProperty(menu, "innerHTML", {get: () => "", set: () => { menu.children = []; }});
menu.querySelectorAll = (selector) => menu.children
    .filter((c) => c.className && c.className.indexOf("forge-assistant-emotion-row") >= 0)
    .map((row) => row.children[0]);
shell.nodes = {menu, picker: made("button"), utilities: made("button"),
               header: made("header"), status: made("p")};
shell.place = () => {};
shell.menuHandle = {};
const rows = () => menu.children.filter((c) => (c.className || "").indexOf("forge-assistant-emotion-row") >= 0);
const describeRow = (row) => ({text: row.children[0].textContent,
                               checked: row.children[0].getAttribute("aria-checked"),
                               role: row.children[0].getAttribute("role"),
                               deletes: row.children.length > 1
                                   ? row.children[1].getAttribute("aria-label") : null});
const settle = async () => { for (let turn = 0; turn < 20; turn += 1) await Promise.resolve(); };
"""


def menu(scenario: str, chosen: str = "") -> dict:
    import json

    return run(MENU.replace("__CHOSEN__", json.dumps(chosen)) + scenario, sources=("shell",))


class TestTheMenuEntry:
    def test_it_is_in_the_chat_group_after_threads_and_says_off(self):
        found = menu("""
            const items = shell.utilityItems();
            const entry = items[3];
            console.log(JSON.stringify({labels: items.slice(0, 4).map((i) => i.textContent),
                value: entry.children[1].textContent, dot: entry.children[0].dataset.on,
                popup: entry.getAttribute("aria-haspopup"), disabled: entry.disabled}));
        """)

        assert found["labels"] == ["Chat · Ada", "New chat", "Threads", "Emotion"]
        assert found["value"] == "Off" and found["dot"] == "false"
        assert found["popup"] == "menu" and found["disabled"] is False

    def test_it_says_the_line_that_is_on(self):
        found = menu("""
            const entry = shell.utilityItems()[3];
            console.log(JSON.stringify({value: entry.children[1].textContent,
                                        dot: entry.children[0].dataset.on}));
        """, chosen=HAPPY)

        assert found == {"value": HAPPY, "dot": "true"}

    def test_without_a_thread_there_is_nothing_to_choose_for(self):
        found = menu("""
            view = {selection: {character: "Ada", thread: ""}, conversation: null};
            const entry = shell.utilityItems()[3];
            console.log(JSON.stringify({disabled: entry.disabled, title: entry.title}));
        """)

        assert found["disabled"] is True
        assert found["title"] == "Choose a conversation first"


class TestTheSubmenu:
    def test_off_and_every_line_with_the_thread_s_own_checked(self):
        found = menu("""
            shell.showMenu("emotion");
            console.log(JSON.stringify({which: menu.dataset.which,
                head: menu.children.slice(0, 2).map((c) => c.textContent),
                rows: rows().map(describeRow),
                last: menu.children[menu.children.length - 1].textContent}));
        """, chosen=HAPPY)

        assert found["which"] == "emotion"
        assert found["head"] == ["‹ Back", "Emotion · Harbour"]
        assert found["rows"] == [
            {"text": "Off", "checked": "false", "role": "menuitemradio", "deletes": None},
            {"text": HAPPY, "checked": "true", "role": "menuitemradio",
             "deletes": f"Delete {HAPPY}"},
            {"text": "*I am so sad*", "checked": "false", "role": "menuitemradio",
             "deletes": "Delete *I am so sad*"},
        ]
        assert found["last"] == "Cancel"

    def test_choosing_a_line_sets_it_and_checks_it_at_once(self):
        found = menu("""
            shell.showMenu("emotion");
            const sad = rows()[2].children[0];
            sad.handlers.click.forEach((fn) => fn());
            const checked = rows().map((r) => r.children[0].getAttribute("aria-checked"));
            await settle();
            console.log(JSON.stringify({sent, checked}));
        """, chosen=HAPPY)

        assert found["checked"] == ["false", "false", "true"]
        assert found["sent"][0] == ["set", "*I am so sad*"]
        assert ["refresh"] in found["sent"]

    def test_off_is_a_choice_too(self):
        found = menu("""
            shell.showMenu("emotion");
            rows()[0].children[0].handlers.click.forEach((fn) => fn());
            await settle();
            console.log(JSON.stringify({sent}));
        """, chosen=HAPPY)

        assert found["sent"][0] == ["set", ""]

    def test_delete_takes_the_line_off_the_list(self):
        found = menu("""
            shell.showMenu("emotion");
            rows()[2].children[1].handlers.click.forEach((fn) => fn());
            await settle();
            console.log(JSON.stringify({sent}));
        """)

        assert found["sent"][0] == ["change", {"remove": "*I am so sad*"}]
        assert ["set", "*I am so sad*"] not in found["sent"]

    def test_a_new_line_is_added_and_chosen(self):
        found = menu("""
            shell.showMenu("emotion");
            const adder = menu.children.find((c) => (c.className || "").indexOf("forge-assistant-emotion-add") >= 0);
            const input = adder.children[0];
            input.value = "  *I am so   nervous*  ";
            adder.children[1].handlers.click.forEach((fn) => fn());
            await settle(); await settle();
            console.log(JSON.stringify({sent}));
        """)

        assert found["sent"][:2] == [["change", {"add": "*I am so nervous*"}],
                                     ["set", "*I am so nervous*"]]

    def test_an_empty_line_is_said_and_nothing_is_sent(self):
        found = menu("""
            shell.addEmotion("   ");
            await settle();
            console.log(JSON.stringify({sent, told}));
        """)

        assert found["sent"] == []
        assert found["told"] == [["Write the emotion first.", "warn"]]

    def test_enter_in_the_box_adds_it(self):
        found = menu("""
            shell.showMenu("emotion");
            const adder = menu.children.find((c) => (c.className || "").indexOf("forge-assistant-emotion-add") >= 0);
            const input = adder.children[0];
            input.value = "*I am so calm*";
            let stopped = false;
            input.handlers.keydown.forEach((fn) => fn({key: "Enter", preventDefault() { stopped = true; }}));
            await settle(); await settle();
            console.log(JSON.stringify({sent, stopped}));
        """)

        assert found["sent"][0] == ["change", {"add": "*I am so calm*"}]
        assert found["stopped"] is True

    def test_the_pill_opens_it_a_level_down_in_the_menu(self):
        found = menu("""
            shell.openEmotion();
            console.log(JSON.stringify({which: menu.dataset.which, owner: menu.dataset.owner,
                hidden: menu.hidden, expanded: shell.nodes.utilities.getAttribute("aria-expanded"),
                only: !!shell.onlyOpened}))
        """)

        assert found == {"which": "emotion", "owner": "utilities", "hidden": False,
                         "expanded": "true", "only": True}


class TestHowItSteers:
    def test_the_submenu_ends_with_the_two_ways_and_the_current_one_checked(self):
        found = menu("""
            view.conversation.emotion_mode = "prefill";
            shell.showMenu("emotion");
            const all = menu.children;
            const modes = all.filter((c) => (c.className || "").indexOf("forge-assistant-emotion-mode") >= 0);
            const first = all.indexOf(modes[0]);
            console.log(JSON.stringify({
                heading: all[first - 1].textContent,
                modes: modes.map((m) => [m.textContent, m.getAttribute("role"),
                                         m.getAttribute("aria-checked"), m.dataset.emotionMode]),
                last: all[all.length - 1].textContent}));
        """)

        assert found["heading"] == "How it steers"
        assert found["modes"] == [["System prompt", "menuitemradio", "false", "system"],
                                  ["Starts the reply", "menuitemradio", "true", "prefill"]]
        assert found["last"] == "Cancel"

    def test_with_no_word_from_the_server_the_system_prompt_is_checked(self):
        found = menu("""
            shell.showMenu("emotion");
            const modes = menu.children.filter((c) => (c.className || "").indexOf("forge-assistant-emotion-mode") >= 0);
            console.log(JSON.stringify(modes.map((m) => m.getAttribute("aria-checked"))));
        """)

        assert found == ["true", "false"]

    def test_choosing_one_sends_it_checks_it_at_once_and_reads_the_thread_again(self):
        found = menu("""
            shell.store.setEmotionMode = (mode) => { sent.push(["mode", mode]);
                                                     return Promise.resolve({ok: true}); };
            menu.querySelectorAll = (selector) => menu.children.filter(
                (c) => (c.className || "").indexOf(selector.replace(/^\\./, "")) >= 0);
            shell.showMenu("emotion");
            const modes = menu.children.filter((c) => (c.className || "").indexOf("forge-assistant-emotion-mode") >= 0);
            modes[1].handlers.click.forEach((fn) => fn());
            const checked = modes.map((m) => m.getAttribute("aria-checked"));
            await settle();
            console.log(JSON.stringify({sent, checked}));
        """)

        assert found["checked"] == ["false", "true"]
        assert found["sent"][0] == ["mode", "prefill"]
        assert ["refresh"] in found["sent"]

    def test_a_refusal_is_said(self):
        found = menu("""
            shell.store.setEmotionMode = () => Promise.resolve({ok: false,
                error: {message: "An emotion steers from the system prompt or as the start of the reply."}});
            shell.showMenu("emotion");
            const modes = menu.children.filter((c) => (c.className || "").indexOf("forge-assistant-emotion-mode") >= 0);
            modes[1].handlers.click.forEach((fn) => fn());
            await settle();
            console.log(JSON.stringify({told}));
        """)

        assert found["told"] == [["An emotion steers from the system prompt or as the start "
                                  "of the reply.", "warn"]]


PILL = """
const pill = document.createElement("button");
const word = document.createElement("span");
const shell = Object.create(NS.Shell.prototype);
shell.nodes = {emotion: pill, emotionWord: word};
const look = (line, thread) => {
    shell.renderEmotion({selection: {thread: thread === undefined ? "t1" : thread},
                         conversation: {conversation: {emotion: line}}});
    return {hidden: pill.hidden, word: word.textContent, label: pill.getAttribute("aria-label")};
};
"""


class TestThePill:
    def test_it_is_not_drawn_while_the_thread_has_none(self):
        found = run(PILL + "console.log(JSON.stringify(look('')));", sources=("shell",))

        assert found["hidden"] is True

    def test_it_says_the_word_and_the_whole_line_for_a_reader(self):
        found = run(PILL + "console.log(JSON.stringify(look('*I am so happy*')));",
                    sources=("shell",))

        assert found == {"hidden": False, "word": "happy", "label": "Emotion: *I am so happy*"}

    def test_a_line_of_another_shape_is_said_whole(self):
        found = run(PILL + "console.log(JSON.stringify(look('*grumbles quietly*')));",
                    sources=("shell",))

        assert found["word"] == "grumbles quietly"

    def test_with_no_thread_it_is_not_drawn(self):
        found = run(PILL + "console.log(JSON.stringify(look('*I am so happy*', '')));",
                    sources=("shell",))

        assert found["hidden"] is True

    def test_it_is_written_only_when_it_changed(self):
        found = run(PILL + """
            look('*I am so happy*');
            let writes = 0;
            const set = pill.setAttribute.bind(pill);
            pill.setAttribute = (name, value) => { writes += 1; set(name, value); };
            look('*I am so happy*'); look('*I am so happy*');
            console.log(JSON.stringify({writes}));
        """, sources=("shell",))

        assert found["writes"] == 0


STATUS = """
const shell = Object.create(NS.Shell.prototype);
const status = document.createElement("p");
shell.nodes = {status};
const said = (extra) => {
    shell.renderStatus(Object.assign({ready: true, connected: true, everConnected: true,
                                      error: "", selection: {character: "Ada", thread: "t1"},
                                      characters: ["Ada"], operation: null, writing: false},
                                     extra || {}));
    return [status.textContent, status.dataset.kind];
};
"""


class TestTheStatusLine:
    def test_quietly_while_the_cache_is_updated(self):
        found = run(STATUS + "console.log(JSON.stringify(said({cacheUpdate: true})));",
                    sources=("shell",))

        assert found == ["Updating cache…", "updating"]

    def test_ready_when_it_is_done(self):
        found = run(STATUS + "console.log(JSON.stringify(said({cacheUpdate: false})));",
                    sources=("shell",))

        assert found == ["Ready.", "info"]

    def test_a_message_on_its_way_says_its_own_progress_instead(self):
        """Which, when it is waiting for the update, is the server's own
        sentence: Waiting for the cache update…"""
        found = run(STATUS + """
            console.log(JSON.stringify(said({cacheUpdate: true, operation: {terminal: false,
                status: "Waiting for the cache update…", since: Date.now()}})));
        """, sources=("shell",))

        assert found == ["Waiting for the cache update…", "info"]


class TestTheStore:
    def test_choosing_sends_set_emotion_against_no_revision(self):
        found = run("""
            const store = new NS.Store();
            store.selection = {character: "Ada", thread: "t1", epoch: "x"};
            const sent = [];
            store.send = (envelope) => { sent.push(envelope); return Promise.resolve({ok: true}); };
            store.setEmotion("*I am so happy*");
            console.log(JSON.stringify(sent.map((e) => [e.action, e.expected_revision,
                                                        e.payload, e.conversation])));
        """, sources=("store",))

        assert found == [["set_emotion", None, {"emotion": HAPPY},
                          {"character": "Ada", "thread_id": "t1"}]]

    def test_the_way_it_steers_is_sent_against_no_revision(self):
        found = run("""
            const store = new NS.Store();
            store.selection = {character: "Ada", thread: "t1", epoch: "x"};
            const sent = [];
            store.send = (envelope) => { sent.push(envelope); return Promise.resolve({ok: true}); };
            store.setEmotionMode("prefill");
            console.log(JSON.stringify(sent.map((e) => [e.action, e.expected_revision,
                                                        e.payload, e.conversation])));
        """, sources=("store",))

        assert found == [["set_emotion_mode", None, {"emotion_mode": "prefill"},
                          {"character": "Ada", "thread_id": "t1"}]]

    def test_the_list_is_changed_through_its_own_route(self):
        found = run("""
            const store = new NS.Store();
            const asked = [];
            store.request = (path, options) => { asked.push([path, options.method,
                JSON.parse(options.body)]); return Promise.resolve({ok: true}); };
            store.changeEmotions({add: "*I am so calm*"});
            console.log(JSON.stringify(asked));
        """, sources=("store",))

        assert found == [["/emotions", "POST", {"add": "*I am so calm*"}]]

    def test_the_feed_is_held_while_the_open_thread_s_cache_is_updated(self):
        found = run("""
            const store = new NS.Store();
            store.selection = {character: "Ada", thread: "t1", epoch: "x"};
            const key = NS.conversationKey("Ada", "t1");
            store.snapshots.set(key, {conversation: {cache_update: false}});
            const before = store.needsFeed();
            store.snapshots.get(key).conversation.cache_update = true;
            console.log(JSON.stringify({before, during: store.needsFeed(),
                                        view: store.snapshot().cacheUpdate}));
        """, sources=("store",))

        assert found == {"before": False, "during": True, "view": True}

    def test_the_feed_says_when_it_starts_and_ends_and_the_page_looks_again(self):
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "t1", epoch: "x"};
            const key = NS.conversationKey("Ada", "t1");
            store.snapshots.set(key, {conversation: {cache_update: false}});
            let reviewed = 0;
            store.review = () => { reviewed += 1; };
            const event = (cursor, updating) => store.apply({protocol_version: 2,
                server_epoch: "e", stream_cursor: cursor, kind: "cache_update",
                conversation: {character: "Ada", thread_id: "t1"}, payload: {updating}});
            event(1, true);
            const during = store.updatingCache();
            event(2, false);
            console.log(JSON.stringify({during, after: store.updatingCache(), reviewed}));
        """, sources=("store",))

        assert found == {"during": True, "after": False, "reviewed": 2}

    def test_another_thread_s_update_is_not_this_one_s(self):
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "t1", epoch: "x"};
            store.snapshots.set(NS.conversationKey("Ada", "t1"), {conversation: {}});
            store.review = () => {};
            store.apply({protocol_version: 2, server_epoch: "e", stream_cursor: 1,
                         kind: "cache_update", conversation: {character: "Ada", thread_id: "t2"},
                         payload: {updating: true}});
            console.log(JSON.stringify({mine: store.updatingCache()}));
        """, sources=("store",))

        assert found == {"mine": False}
