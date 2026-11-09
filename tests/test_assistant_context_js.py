"""The Forge Assistant's half of the context window, executed under node.

"I want the ability to see how full my context window is, make it something
subtle, like where the status message is. I would like the ability to adjust
the context window before my next reply ... keep the 5,000 latest ... a
compacting option as an alternative" (2026-11-09).

The meter sits at the right end of the status line -- quiet numbers, coloured
only as the window nears the point where its oldest messages leave by
themselves -- and a press on it opens the ⋯ menu's Context submenu: how full it
is, Keep the latest, Summary length, Trim / Compact / Clear, the summary if
there is one, and while a compaction runs, its progress and Cancel. The store
sends the four commands against no revision and keeps its feed open while the
thread is compacted.
"""

from __future__ import annotations

import json

from test_assistant_js import run

READING = {"used": 4401, "limit": 8192, "room": 7232, "reserve": 960, "exact": True,
           "front": 12, "sent": 23, "messages": 35, "first": 0, "summary": None,
           "compacting": None, "updating": False}

SHELL = """
const made = (tag) => document.createElement(tag);
const shell = Object.create(NS.Shell.prototype);
shell.state = {freeFloat: false, autoAttach: false, sendToGenerate: false,
               conversationExpanded: true, panelOpen: true};
shell.host = {listUtilities: () => [], openOnly() { shell.onlyOpened = true; }};
let view = {selection: {character: "Ada", thread: "t1"},
            conversation: {conversation: {title: "Harbour", emotion: ""}, emotions: [],
                           threads: [{thread_id: "t1", title: "Harbour"}],
                           context: __CONTEXT__}};
const sent = [];
const answers = {};
shell.store = {
    snapshot: () => view,
    setWindow(keep) { sent.push(["set_window", keep]);
        return Promise.resolve(answers.set_window || {ok: true}); },
    compact(keep, length) { sent.push(["compact", keep, length]);
        return Promise.resolve(answers.compact || {ok: true}); },
    clearWindow() { sent.push(["clear_window"]); return Promise.resolve({ok: true}); },
    cancelCompact() { sent.push(["cancel_compact"]); return Promise.resolve({ok: true}); },
    refresh() { sent.push(["refresh"]); return Promise.resolve(null); },
};
const told = [];
shell.tell = (text, kind) => told.push([text, kind]);
const menu = made("div");
menu.hidden = true;
Object.defineProperty(menu, "innerHTML", {get: () => "", set: () => { menu.children = []; }});
const meter = made("button");
meter.hidden = true;
shell.nodes = {menu, picker: made("button"), utilities: made("button"),
               header: made("header"), status: made("p"), context: meter};
shell.place = () => {};
shell.menuHandle = {};
const settle = async () => { for (let turn = 0; turn < 20; turn += 1) await Promise.resolve(); };
// The harness's click() only counts: a press runs the listeners themselves.
const press = (node) => (node.handlers.click || []).forEach((fn) => fn({preventDefault() {}}));
const has = (node, cls) => String(node.className || "").split(/\s+/).indexOf(cls) >= 0;
const find = (cls) => menu.children.filter((c) => has(c, cls));
const deep = (node, cls, found) => {
    found = found || [];
    if (has(node, cls)) found.push(node);
    (node.children || []).forEach((child) => deep(child, cls, found));
    return found;
};
"""


def shell(scenario: str, context=READING) -> dict:
    return run(SHELL.replace("__CONTEXT__", json.dumps(context)) + scenario,
               sources=("shell",))


class TestTheMeter:
    def test_it_says_the_numbers_quietly(self):
        found = shell("""
            shell.renderContext(view);
            console.log(JSON.stringify({text: meter.textContent, hidden: meter.hidden,
                                        state: meter.dataset.state, title: meter.title}));
        """)

        assert found["text"] == "4,401 / 8,192"
        assert found["hidden"] is False and found["state"] == "ok"
        assert found["title"].startswith("4,401 of 8,192 tokens in context")
        assert "23 of 35 messages sent" in found["title"]
        assert "older messages leave past 7,232" in found["title"]

    def test_an_estimate_says_so(self):
        found = shell("shell.renderContext(view); console.log(JSON.stringify(meter.textContent));",
                      context=dict(READING, exact=False))

        assert found == "≈4,401 / 8,192"

    def test_it_is_coloured_as_the_room_runs_out(self):
        states = {}
        for used, state in ((4401, "ok"), (6600, "near"), (7300, "full")):
            states[used] = shell("shell.renderContext(view); "
                                 "console.log(JSON.stringify(meter.dataset.state));",
                                 context=dict(READING, used=used))
        assert states == {4401: "ok", 6600: "near", 7300: "full"}

    def test_and_pulses_while_the_thread_is_compacted(self):
        found = shell("shell.renderContext(view); console.log(JSON.stringify(meter.dataset.state));",
                      context=dict(READING, compacting={"phase": "writing", "written": 40}))

        assert found == "compacting"

    def test_nothing_without_a_thread(self):
        found = shell("""
            view.selection.thread = "";
            shell.renderContext(view);
            console.log(JSON.stringify({hidden: meter.hidden, text: meter.textContent}));
        """)

        assert found == {"hidden": True, "text": ""}

    def test_it_is_written_only_when_it_changed(self):
        """Every render calls it, a streamed reply once a frame."""
        found = shell("""
            shell.renderContext(view);
            let writes = 0;
            Object.defineProperty(meter, "textContent", {set() { writes += 1; }, get() { return ""; }});
            shell.renderContext(view);
            shell.renderContext(view);
            console.log(JSON.stringify(writes));
        """)

        assert found == 0

    def test_how_a_compaction_ended_is_said_once(self):
        found = shell("""
            view.conversation.context.compacting = {phase: "writing", written: 40};
            shell.renderContext(view);
            view.conversation.context.compacting = {phase: "done",
                status: "Compacted 18 messages (3,120 tokens) into a summary of 296."};
            shell.renderContext(view);
            shell.renderContext(view);
            console.log(JSON.stringify(told));
        """)

        assert found == [["Compacted 18 messages (3,120 tokens) into a summary of 296.", "info"]]

    def test_a_press_opens_the_context_submenu(self):
        found = shell("""
            shell.openContext();
            console.log(JSON.stringify({which: menu.dataset.which, owner: menu.dataset.owner,
                                        hidden: menu.hidden, only: !!shell.onlyOpened}));
        """)

        assert found == {"which": "context", "owner": "utilities", "hidden": False,
                         "only": True}


class TestTheSubmenu:
    def test_its_entry_follows_emotion_in_the_chat_group_with_the_numbers(self):
        found = shell("""
            const items = shell.utilityItems();
            const entry = items[4];
            console.log(JSON.stringify({labels: items.slice(0, 5).map((i) => i.textContent),
                value: entry.children[0].textContent, popup: entry.getAttribute("aria-haspopup"),
                disabled: entry.disabled}));
        """)

        assert found["labels"] == ["Chat · Ada", "New chat", "Threads", "Emotion", "Context"]
        assert found["value"] == "4,401 / 8,192"
        assert found["popup"] == "menu" and found["disabled"] is False

    def test_it_reads_how_full_it_is_and_offers_the_three_actions(self):
        found = shell("""
            shell.showMenu("context");
            const sliders = deep(menu, "forge-assistant-context-slider");
            console.log(JSON.stringify({
                label: menu.children[1].textContent,
                note: find("forge-assistant-context-note")[0].textContent,
                sliders: sliders.map((s) => [s.getAttribute("aria-label"), s.min, s.max, s.step,
                                             s.value]),
                actions: deep(menu, "forge-assistant-context-action").map((b) => b.textContent),
                summary: find("forge-assistant-context-summary").length}));
        """)

        assert found["label"] == "Context · Harbour"
        assert found["note"].startswith("4,401 of 8,192 tokens in context")
        assert found["sliders"] == [
            ["Keep the latest (tokens)", "0", "8100", "100", "4900"],
            ["Summary length (tokens)", "100", "1500", "50", "300"]]
        assert found["actions"] == ["Trim", "Compact", "Clear"]
        assert found["summary"] == 0

    def test_a_summary_is_shown_under_them(self):
        found = shell("""
            shell.showMenu("context");
            console.log(JSON.stringify(find("forge-assistant-context-summary")
                .map((node) => node.textContent)));
        """, context=dict(READING, summary={"tokens": 296, "covers": 12, "text": "They met."}))

        assert found == ["They met."]

    def test_while_it_is_compacted_only_its_progress_and_cancel(self):
        found = shell("""
            shell.showMenu("context");
            console.log(JSON.stringify({
                busy: find("forge-assistant-context-busy").map((n) => n.textContent),
                cancel: find("forge-assistant-context-cancel").map((n) => n.textContent),
                actions: deep(menu, "forge-assistant-context-action").length}));
        """, context=dict(READING, compacting={"phase": "writing", "written": 1234}))

        assert found == {"busy": ["Summarizing… 1,234 characters"],
                         "cancel": ["Cancel compacting"], "actions": 0}

    def test_without_a_reading_it_says_to_choose_a_thread(self):
        found = shell("""
            shell.showMenu("context");
            console.log(JSON.stringify(find("forge-assistant-context-note")
                .map((n) => n.textContent)));
        """, context=None)

        assert found == ["Choose a conversation to see its context."]


class TestTheActions:
    def press(self, action: str, context=READING, extra: str = "") -> dict:
        return shell("""
            shell.showMenu("context");
            const sliders = deep(menu, "forge-assistant-context-slider");
            sliders[0].value = "5000";
            sliders[1].value = "400";
            """ + extra + """
            press(deep(menu, "forge-assistant-context-action")
                .find((b) => b.dataset.action === "__ACTION__"));
            settle().then(() => console.log(JSON.stringify({sent, told})));
        """.replace("__ACTION__", action), context=context)

    def test_trim_keeps_the_latest_chosen(self):
        found = self.press("set_window", extra="""
            answers.set_window = {ok: true, kept: {messages: 14, tokens: 4980}};""")

        assert found["sent"][0] == ["set_window", 5000]
        assert found["sent"][-1] == ["refresh"]
        assert found["told"] == [["Sending the latest 14 messages (4,980 tokens).", "info"]]

    def test_compact_keeps_the_latest_and_asks_for_the_length_chosen(self):
        found = self.press("compact", extra="""
            answers.compact = {ok: true, compacting: {messages: 18, tokens: 3120}};""")

        assert found["sent"][0] == ["compact", 5000, 400]
        assert found["told"] == [["Compacting 18 messages…", "info"]]

    def test_clear(self):
        found = self.press("clear_window")

        assert found["sent"][0] == ["clear_window"]
        assert found["told"] == [["The context sends the newest messages that fit again.",
                                  "info"]]

    def test_a_refusal_is_said(self):
        found = self.press("compact", extra="""
            answers.compact = {ok: false, error: {message: "Nothing older than that is being sent now."}};""")

        assert found["told"] == [["Nothing older than that is being sent now.", "warn"]]

    def test_cancel(self):
        found = shell("""
            shell.showMenu("context");
            press(find("forge-assistant-context-cancel")[0]);
            settle().then(() => console.log(JSON.stringify(sent)));
        """, context=dict(READING, compacting={"phase": "writing", "written": 1}))

        assert found[0] == ["cancel_compact"]


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
    def test_quietly_while_the_thread_is_compacted(self):
        found = run(STATUS + """
            console.log(JSON.stringify(said({conversation: {context: {compacting:
                {phase: "writing", written: 820}}}, cacheUpdate: true})));
        """, sources=("shell",))

        assert found == ["Summarizing… 820 characters", "updating"]

    def test_before_the_summary_starts_what_it_waits_for(self):
        found = run(STATUS + """
            console.log(JSON.stringify(said({conversation: {context: {compacting:
                {phase: "waiting", status: "Waiting for the image job…"}}}})));
        """, sources=("shell",))

        assert found == ["Waiting for the image job…", "updating"]

    def test_ready_once_it_ended(self):
        found = run(STATUS + """
            console.log(JSON.stringify(said({conversation: {context: {compacting:
                {phase: "done", status: "Compacted."}}}})));
        """, sources=("shell",))

        assert found == ["Ready.", "info"]


class TestTheStore:
    def sent(self, call: str) -> list:
        return run("""
            const store = new NS.Store();
            store.selection = {character: "Ada", thread: "t1", epoch: "x"};
            store.review = () => {};
            const sent = [];
            store.send = (envelope) => { sent.push(envelope); return Promise.resolve({ok: true}); };
            store.__CALL__;
            Promise.resolve().then(() => console.log(JSON.stringify(sent.map((e) => [e.action,
                e.expected_revision, e.payload, e.conversation]))));
        """.replace("__CALL__", call), sources=("store",))

    def test_trim_is_sent_against_no_revision(self):
        assert self.sent("setWindow(4999.6)") == [
            ["set_window", None, {"keep_tokens": 5000}, {"character": "Ada", "thread_id": "t1"}]]

    def test_compact_carries_both_numbers(self):
        assert self.sent("compact(5000, 300)") == [
            ["compact", None, {"keep_tokens": 5000, "summary_tokens": 300},
             {"character": "Ada", "thread_id": "t1"}]]

    def test_clear_and_cancel(self):
        assert self.sent("clearWindow()")[0][:2] == ["clear_window", None]
        assert self.sent("cancelCompact()")[0][:2] == ["cancel_compact", None]

    def test_the_feed_is_held_while_the_open_thread_is_compacted(self):
        found = run("""
            const store = new NS.Store();
            store.selection = {character: "Ada", thread: "t1", epoch: "x"};
            const key = NS.conversationKey("Ada", "t1");
            store.snapshots.set(key, {conversation: {}, context: {compacting: null}});
            const before = store.needsFeed();
            store.snapshots.get(key).context.compacting = {phase: "writing"};
            const during = store.needsFeed();
            store.snapshots.get(key).context.compacting = {phase: "done"};
            console.log(JSON.stringify({before, during, after: store.needsFeed()}));
        """, sources=("store",))

        assert found == {"before": False, "during": True, "after": False}

    def test_the_feed_carries_its_progress_and_its_end_reads_the_thread_again(self):
        found = run("""
            const store = new NS.Store();
            store.serverEpoch = "e";
            store.selection = {character: "Ada", thread: "t1", epoch: "x"};
            const key = NS.conversationKey("Ada", "t1");
            store.snapshots.set(key, {conversation: {}, context: {compacting: null}});
            let reviewed = 0, refreshed = 0;
            store.review = () => { reviewed += 1; };
            store.refresh = () => { refreshed += 1; return Promise.resolve(null); };
            const event = (cursor, payload) => store.apply({protocol_version: 2,
                server_epoch: "e", stream_cursor: cursor, kind: "context",
                conversation: {character: "Ada", thread_id: "t1"}, payload});
            event(1, {compacting: {phase: "writing", written: 12}});
            const during = Object.assign({}, store.snapshots.get(key).context.compacting);
            const refreshedDuring = refreshed;
            event(2, {compacting: null, outcome: "done", message: "Compacted."});
            console.log(JSON.stringify({during, refreshedDuring, reviewed, refreshed,
                after: store.snapshots.get(key).context.compacting}));
        """, sources=("store",))

        assert found["during"] == {"phase": "writing", "written": 12}
        assert found["refreshedDuring"] == 0
        assert found["after"] == {"phase": "done", "status": "Compacted."}
        assert found["refreshed"] == 1 and found["reviewed"] == 2
