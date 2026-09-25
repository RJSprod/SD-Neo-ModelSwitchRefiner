"""The flyout's fourth round: three things asked for together.

    1. Editing a message opened the browser's own prompt ("It sucks. I dont
       want the browser doing this, i need UI in our flyout"), and the edit has
       to be unmistakably an edit and not a new message;
    2. the ⋯ menu's switches did not say whether they were on;
    3. a "Send to Generate" switch: the reply's send button replaces the
       prompt *and* presses Generate, as if the person had.

Run against the real browser files in the node harness `test_assistant_js.py`
uses. Every test here was checked against the change it guards by reverting
that change and watching the test fail.
"""

from __future__ import annotations

import re

from test_assistant_js import SHELL, run

CSS = (SHELL.parent.parent / "style.css").read_text(encoding="utf-8")


def rule(selector):
    """The declarations of every rule whose selector list names `selector`."""
    bare = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    found = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", bare):
        names = [piece.strip() for piece in match.group(1).split(",")]
        if selector in names:
            found.append(match.group(2))
    assert found, selector + " has no rule of its own"
    return "\n".join(found)


# --------------------------------------------------------------------------- #
# 1. Editing a message in the panel
# --------------------------------------------------------------------------- #

EDITING = """
// A panel with a composer and a transcript, and a store that answers as the
// scenario says. Only what the edit touches is real; everything is the real
// shell code.
const made = (tag, cls) => { const n = document.createElement(tag); if (cls) n.className = cls; return n; };
const KEY = NS.conversationKey("Ada", "t1");
const drafts = {};
drafts[KEY] = {text: "half a thought", attachment: null};
drafts[NS.conversationKey("Ada", "t2")] = {text: "the other thread's draft", attachment: null};
const MESSAGES = [
    {index: 0, role: "user", text: "make it top down view", versions: ["make it top down view"], active: 0},
    {index: 1, role: "assistant", text: "A harbour seen from above, 35mm", versions: ["a"], active: 0},
];
let view = {ready: true, error: "", operation: null,
            selection: {character: "Ada", thread: "t1", epoch: "e1"},
            conversation: {conversation: {character: "Ada", thread_id: "t1", revision: 7},
                           messages: MESSAGES},
            draft: drafts[KEY], speech: {playing: false}};
const store = {
    sent: [], typed: [], answer: {ok: true},
    snapshot() { return view; },
    draft(key) { return drafts[key || KEY] || {text: ""}; },
    setDraftText(text) { store.typed.push(text); },
    envelope(action, extra) {
        return Object.assign({action, operation_id: "op" + store.sent.length,
                              conversation: {character: view.selection.character,
                                             thread_id: view.selection.thread},
                              payload: {}}, extra || {});
    },
    send(envelope) {
        store.sent.push(JSON.parse(JSON.stringify(envelope)));
        return Promise.resolve(store.answer);
    },
};
const shell = Object.create(NS.Shell.prototype);
shell.store = store;
shell.state = {};
shell.settings = {bubbleWidth: 75};
const composer = made("div", "forge-assistant-composer");
const editBar = made("div", "forge-assistant-edit-bar");
editBar.hidden = true;
const transcript = made("div", "forge-assistant-transcript");
MESSAGES.forEach((row) => {
    const bubble = made("article", "forge-assistant-bubble");
    bubble.dataset.index = String(row.index);
    bubble.scrollIntoView = () => { bubble.scrolledTo = true; };
    transcript.appendChild(bubble);
});
const input = made("textarea", "forge-assistant-input");
input.value = "half a thought";
input.scrollHeight = 40;
input.setSelectionRange = (a, b) => { input.caret = [a, b]; };
input.focus = () => { document.activeElement = input; };
shell.nodes = {composer, editBar, editLabel: made("span"), editCancel: made("button"),
               input, transcript, send: made("button"), attach: made("button"),
               dictate: made("button"), status: made("p"), root: made("div")};
shell.nodes.send.textContent = "Send";
shell.nodes.root.contains = (node) => node === input;
const settle = () => new Promise((resolve) => setImmediate(resolve));
function mode() {
    const n = shell.nodes;
    return {editing: !!shell.editing, value: input.value,
            composer: composer.classList.contains("forge-assistant-editing"),
            bar: !n.editBar.hidden, label: n.editLabel.textContent, send: n.send.textContent,
            attach: !!n.attach.disabled, dictate: !!n.dictate.disabled,
            said: n.status.textContent, kind: n.status.dataset.kind,
            targets: transcript.children.filter((b) => b.classList.contains(
                "forge-assistant-edit-target")).map((b) => b.dataset.index)};
}
"""


def run_edit(scenario, sources=("shell", "store")):
    return run(EDITING + scenario, sources=sources)


class TestEditingHappensInThePanel:
    """"That image is what my browser threw me to edit. It sucks. I dont want
    the browser doing this, i need UI in our flyout.\""""

    def test_the_browser_s_prompt_is_never_asked(self):
        found = run_edit("""
            let asked = 0;
            globalThis.prompt = () => { asked += 1; return "typed into a dialog"; };
            shell.startEdit(MESSAGES[0], 7);
            console.log(JSON.stringify({asked, value: input.value}));
        """)

        assert found == {"asked": 0, "value": "make it top down view"}
        assert not re.search(r"window\.prompt\s*\(", SHELL.read_text(encoding="utf-8"))

    def test_the_message_goes_into_the_box_and_the_box_says_it_is_an_edit(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            console.log(JSON.stringify(Object.assign(mode(), {
                focused: document.activeElement === input, caret: input.caret,
                aria: input["aria-label"]})));
        """)

        assert found["value"] == "make it top down view"
        assert found["composer"] is True and found["bar"] is True
        assert found["label"] == "✎ Editing your message"
        assert found["send"] == "Save"
        assert found["focused"] is True and found["caret"] == [21, 21]
        assert found["aria"] == "Edit the message"

    def test_a_reply_says_whose_it_is(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[1], 7);
            console.log(JSON.stringify(mode()));
        """)

        assert found["label"] == "✎ Editing Ada’s reply"

    def test_the_message_being_edited_is_outlined_and_brought_into_view(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[1], 7);
            const during = mode().targets;
            const shown = !!transcript.children[1].scrolledTo;
            shell.cancelEdit();
            console.log(JSON.stringify({during, shown, after: mode().targets}));
        """)

        assert found == {"during": ["1"], "shown": True, "after": []}

    def test_the_outline_survives_the_thread_being_redrawn(self):
        """Bubbles are rebuilt whenever the thread moves; the mark has to be
        put back on the new one."""
        found = run_edit("""
            Object.defineProperty(transcript, "innerHTML", {
                get() { return ""; }, set() { transcript.children = []; }});
            shell.nodes.jump = made("button");
            shell.startEdit(MESSAGES[1], 7);
            view = Object.assign({}, view, {conversation: {
                conversation: {character: "Ada", thread_id: "t1", revision: 8},
                messages: MESSAGES}});
            shell.renderTranscript(view);
            console.log(JSON.stringify({count: transcript.children.length,
                                        targets: mode().targets}));
        """)

        assert found == {"count": 2, "targets": ["1"]}

    def test_the_tools_that_add_to_a_message_stand_aside(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            const during = [mode().attach, mode().dictate];
            shell.cancelEdit();
            console.log(JSON.stringify({during, after: [mode().attach, mode().dictate]}));
        """)

        assert found == {"during": [True, True], "after": [False, False]}


class TestSavingAnEdit:
    """"i just need the ability to submit the edit and have it replace what
    was there.\""""

    def test_save_replaces_the_message_it_was_aimed_at(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            input.value = "make it a top down view, at dusk";
            shell.send();
            settle().then(() => console.log(JSON.stringify(store.sent)));
        """)

        assert len(found) == 1
        sent = found[0]
        assert sent["action"] == "edit_message"
        assert sent["target"] == {"index": 0, "version": 0}
        assert sent["expected_revision"] == {"kind": "revision", "value": 7}
        assert sent["conversation"] == {"character": "Ada", "thread_id": "t1"}
        assert sent["payload"] == {"text": "make it a top down view, at dusk",
                                   "image_action": "keep"}

    def test_the_save_goes_to_the_conversation_the_edit_began_in(self):
        """The selection can move before the panel redraws and puts the edit
        away; a Save pressed in between is still aimed at the edit's thread."""
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            input.value = "edited";
            view = Object.assign({}, view, {selection: {character: "Ada", thread: "t2",
                                                        epoch: "e2"}});
            shell.saveEdit().then(() => console.log(JSON.stringify(store.sent[0].conversation)));
        """)

        assert found == {"character": "Ada", "thread_id": "t1"}

    def test_a_second_press_on_edit_keeps_what_is_being_typed(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            input.value = "half of my edit";
            shell.startEdit(MESSAGES[0], 7);
            console.log(JSON.stringify(mode()));
        """)

        assert found["editing"] is True
        assert found["value"] == "half of my edit"

    def test_enter_saves_and_asks_for_no_reply(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            input.value = "edited";
            let prevented = false;
            shell.composerKey({key: "Enter", preventDefault() { prevented = true; }});
            settle().then(() => console.log(JSON.stringify(
                {prevented, actions: store.sent.map((e) => e.action)})));
        """)

        assert found == {"prevented": True, "actions": ["edit_message"]}

    def test_afterwards_the_box_is_the_draft_again(self):
        """What was half-typed before Edit comes back, and it never went into
        the store as the edit's words."""
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            input.value = "edited";
            input.handlers = input.handlers || {};
            shell.saveEdit().then(() => console.log(JSON.stringify(Object.assign(mode(),
                {typed: store.typed}))));
        """)

        assert found["editing"] is False
        assert found["value"] == "half a thought"
        assert found["composer"] is False and found["bar"] is False
        assert found["send"] == "Send"
        assert found["said"] == "Message edited."
        assert found["typed"] == []

    def test_typing_an_edit_does_not_write_the_draft(self):
        found = run_edit("""
            shell.typed();
            const before = store.typed.slice();
            shell.startEdit(MESSAGES[0], 7);
            input.value = "edi";
            shell.typed();
            const saveable = !shell.nodes.send.disabled;
            input.value = "";
            shell.typed();
            console.log(JSON.stringify({before, during: store.typed.length - before.length,
                                        saveable, empty: !!shell.nodes.send.disabled}));
        """)

        assert found["before"] == ["half a thought"], "outside an edit it is the draft"
        assert found["during"] == 0
        assert found["saveable"] is True and found["empty"] is True

    def test_a_refused_save_keeps_the_edit_and_says_why(self):
        found = run_edit("""
            store.answer = {ok: false, error: {message: "That message changed in another window."}};
            shell.startEdit(MESSAGES[0], 7);
            input.value = "my careful edit";
            shell.saveEdit().then(() => console.log(JSON.stringify(mode())));
        """)

        assert found["editing"] is True
        assert found["value"] == "my careful edit"
        assert found["said"] == ("That message changed in another window. "
                                 "Your edit is still in the box.")
        assert found["kind"] == "warn"

    def test_a_save_that_throws_is_a_refusal_too(self):
        found = run_edit("""
            store.send = () => Promise.reject(new Error("The server went away."));
            shell.startEdit(MESSAGES[0], 7);
            input.value = "my careful edit";
            shell.saveEdit().then(() => console.log(JSON.stringify(mode())));
        """)

        assert found["editing"] is True and found["value"] == "my careful edit"
        assert found["said"].startswith("The server went away.")

    def test_a_second_press_while_saving_sends_nothing_more(self):
        found = run_edit("""
            let release;
            store.send = (envelope) => { store.sent.push(envelope);
                return new Promise((resolve) => { release = () => resolve({ok: true}); }); };
            shell.startEdit(MESSAGES[0], 7);
            input.value = "edited";
            shell.saveEdit();
            shell.saveEdit();
            shell.send();
            const during = store.sent.length;
            release();
            settle().then(() => console.log(JSON.stringify({during, editing: !!shell.editing})));
        """)

        assert found == {"during": 1, "editing": False}

    def test_nothing_changed_is_not_sent(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            shell.saveEdit().then(() => console.log(JSON.stringify(Object.assign(mode(),
                {sent: store.sent.length}))));
        """)

        assert found["sent"] == 0
        assert found["editing"] is False
        assert found["said"] == "Nothing was changed."

    def test_an_empty_box_is_not_saved(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            input.value = "   ";
            const allowed = shell.canSaveEdit(view);
            shell.saveEdit().then(() => console.log(JSON.stringify(Object.assign(mode(),
                {sent: store.sent.length, allowed}))));
        """)

        assert found["sent"] == 0
        assert found["allowed"] is False
        assert found["editing"] is True
        assert found["kind"] == "warn"

    def test_save_waits_for_a_reply_that_is_being_written(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            input.value = "edited";
            const idle = shell.canSaveEdit(view);
            view = Object.assign({}, view, {operation: {id: "o1", terminal: false}});
            console.log(JSON.stringify({idle, busy: shell.canSaveEdit(view)}));
        """)

        assert found == {"idle": True, "busy": False}


class TestLeavingAnEdit:
    def test_cancel_puts_everything_back_and_sends_nothing(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            input.value = "a change I no longer want";
            shell.cancelEdit();
            console.log(JSON.stringify(Object.assign(mode(), {sent: store.sent.length})));
        """)

        assert found["editing"] is False
        assert found["value"] == "half a thought"
        assert found["sent"] == 0
        assert found["said"] == "Edit cancelled. The message is as it was."

    def test_escape_in_the_panel_cancels_it(self):
        found = run_edit("""
            shell.focus = {isActive: () => true};
            shell.nodes.menu = {hidden: true};
            shell.toggleFocus = () => { shell.leftFocus = true; };
            shell.startEdit(MESSAGES[0], 7);
            let prevented = false;
            shell.documentKey({key: "Escape", preventDefault() { prevented = true; },
                               stopPropagation() {}});
            console.log(JSON.stringify({editing: !!shell.editing, prevented,
                                        leftFocus: !!shell.leftFocus}));
        """, sources=("shell", "host", "store"))

        assert found == {"editing": False, "prevented": True, "leftFocus": False}

    def test_escape_elsewhere_on_the_page_is_not_the_edit_s(self):
        """An Escape in Forge's own prompt box belongs to that box; taking it
        would drop an edit nobody was looking at."""
        found = run_edit("""
            shell.focus = {isActive: () => false};
            shell.nodes.menu = {hidden: true};
            shell.startEdit(MESSAGES[0], 7);
            document.activeElement = made("textarea");
            let prevented = false;
            shell.documentKey({key: "Escape", preventDefault() { prevented = true; },
                               stopPropagation() {}});
            console.log(JSON.stringify({editing: !!shell.editing, prevented}));
        """, sources=("shell", "host", "store"))

        assert found == {"editing": True, "prevented": False}

    def test_another_conversation_puts_the_edit_away_and_shows_its_draft(self):
        """Save would be aimed at a message in a thread that is no longer on
        screen -- and the box, focused, would keep the old thread's draft."""
        found = run_edit("""
            ["renderSelector", "renderChip", "renderAutoAttach", "renderReadAloud",
             "renderTranscript", "renderStatus", "applySuppression"].forEach((name) => {
                shell[name] = () => {};
            });
            shell.nodes.unread = made("span");
            shell.nodes.stop = made("button");
            shell.canSend = () => true;
            shell.startEdit(MESSAGES[0], 7);
            view = Object.assign({}, view, {
                selection: {character: "Ada", thread: "t2", epoch: "e2"},
                draft: drafts[NS.conversationKey("Ada", "t2")]});
            shell.render(view);
            console.log(JSON.stringify(mode()));
        """)

        assert found["editing"] is False
        assert found["value"] == "the other thread's draft"
        assert found["said"] == "The conversation changed, so the edit was put away."

    def test_a_picture_pasted_into_an_edit_is_refused(self):
        found = run_edit("""
            shell.stageFile = () => { shell.staged = true; };
            shell.startEdit(MESSAGES[0], 7);
            let prevented = false;
            shell.paste({preventDefault() { prevented = true; }, clipboardData: {items: [
                {kind: "file", type: "image/png", getAsFile: () => ({name: "p.png"})}]}});
            console.log(JSON.stringify({prevented, staged: !!shell.staged, said: mode().said}));
        """)

        assert found == {"prevented": True, "staged": False,
                         "said": "A picture cannot be added while you edit a message."}

    def test_a_long_edit_gets_a_taller_box(self):
        found = run_edit("""
            globalThis.visualViewport = {height: 800, width: 1280};
            input.scrollHeight = 900;
            shell.grow();
            const message = input.style.height;
            shell.startEdit(MESSAGES[0], 7);
            input.scrollHeight = 900;
            shell.grow();
            console.log(JSON.stringify({message, edit: input.style.height,
                                        scrolls: input.style.overflowY}));
        """)

        assert found == {"message": "132px", "edit": "292px", "scrolls": "auto"}


class TestTheEditLooksLikeAnEdit:
    def test_the_box_is_outlined_and_glows_in_the_accent(self):
        box = rule(".forge-assistant-editing .forge-assistant-input")

        assert "border-color: var(--color-accent" in box
        assert "box-shadow" in box and "var(--color-accent" in box
        assert "accent-soft" not in box

    def test_the_message_is_outlined_the_same_way(self):
        target = rule(".forge-assistant-bubble.forge-assistant-edit-target")

        assert "var(--color-accent" in target and "accent-soft" not in target

    def test_the_strip_and_save_never_take_the_soft_accent(self):
        for selector in (".forge-assistant-edit-bar",
                         ".forge-assistant-editing .forge-assistant-send"):
            assert "accent-soft" not in rule(selector), selector

    def test_cancel_is_a_finger_wide(self):
        assert "min-height: 44px" in rule(".forge-assistant-edit-cancel")
