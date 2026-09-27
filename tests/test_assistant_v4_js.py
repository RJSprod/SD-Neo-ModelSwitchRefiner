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
# 1. Editing a message, in the message editor's dialog
# --------------------------------------------------------------------------- #
#
# The edit first went into the panel's own box (this round's original ask:
# "i need UI in our flyout"), and then into a dialog of the extension's own
# (the round after: "a simple pop up ... current text in an input field,
# cancel, and a done button ... For both LLM studio conversation and our
# flyout menu"). These are the box's tests translated to the dialog: what a
# save carries, where it is aimed, what a refusal leaves, what Cancel and
# Escape do. The dialog is the real `mc_message_editor.js`.

EDITING = """
// A panel with a transcript and a status line, a store that answers as the
// scenario says, and the real message editor over it. Only what the edit
// touches is real; everything is the real shell and editor code.
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
const send = made("button");
send.textContent = "➤";
shell.nodes = {input, transcript, send, attach: made("button"), dictate: made("button"),
               status: made("p"), root: made("div"), composer: made("div")};
const settle = () => new Promise((resolve) => setImmediate(resolve));
const editor = () => window.mcMessageEditor.editor();
const box = () => editor().controls.text;
function mode() {
    const n = shell.nodes;
    const c = editor().controls;
    const open = window.mcMessageEditor.isOpen();
    return {editing: !!shell.editing, open,
            value: open ? c.text.value : null,
            title: c.heading ? c.heading.textContent : "",
            note: c.note ? c.note.textContent : "",
            draft: input.value, send: n.send.textContent,
            attach: !!n.attach.disabled, dictate: !!n.dictate.disabled,
            said: n.status.textContent, kind: n.status.dataset.kind,
            targets: transcript.children.filter((b) => b.classList.contains(
                "forge-assistant-edit-target")).map((b) => b.dataset.index)};
}
"""


def run_edit(scenario, sources=("shell", "store", "editor")):
    return run(EDITING + scenario, sources=sources)


class TestEditingHappensInADialog:
    """"That image is what my browser threw me to edit. It sucks. I dont want
    the browser doing this, i need UI in our flyout." And then: "when i choose
    to edit, i want a simple pop up.\""""

    def test_the_browser_s_prompt_is_never_asked(self):
        found = run_edit("""
            let asked = 0;
            globalThis.prompt = () => { asked += 1; return "typed into a dialog"; };
            shell.startEdit(MESSAGES[0], 7);
            console.log(JSON.stringify({asked, open: mode().open, value: mode().value}));
        """)

        assert found == {"asked": 0, "open": True, "value": "make it top down view"}
        for source in (SHELL, SHELL.parent / "mc_message_editor.js"):
            assert not re.search(r"window\.prompt\s*\(", source.read_text(encoding="utf-8"))

    def test_the_message_goes_into_the_dialog_and_the_dialog_says_whose_it_is(self):
        found = run_edit("""
            editor().build();
            box().setSelectionRange = (a, b) => { box().caret = [a, b]; };
            shell.startEdit(MESSAGES[0], 7);
            console.log(JSON.stringify(Object.assign(mode(), {
                focused: !!box().focused, caret: box().caret,
                placeholder: box().placeholder})));
        """)

        assert found["open"] is True and found["editing"] is True
        assert found["value"] == "make it top down view"
        assert found["title"] == "Edit your message"
        assert found["focused"] is True and found["caret"] == [21, 21]
        assert found["placeholder"] == "The message’s new words…"

    def test_a_reply_says_whose_it_is(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[1], 7);
            console.log(JSON.stringify(mode()));
        """)

        assert found["title"] == "Edit Ada’s reply"

    def test_the_composer_is_left_exactly_as_it_was(self):
        """The box used to hold the edit, so the draft went away and Send read
        Save and the paperclip stood aside. None of that now: the edit is in
        the dialog and the composer is not part of it."""
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            console.log(JSON.stringify(Object.assign(mode(), {typed: store.typed,
                composerClass: shell.nodes.composer.className})));
        """)

        assert found["draft"] == "half a thought"
        assert found["send"] == "➤"
        assert found["attach"] is False and found["dictate"] is False
        assert found["typed"] == []
        assert "editing" not in found["composerClass"]

    def test_the_message_being_edited_is_outlined_and_brought_into_view(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[1], 7);
            const during = mode().targets;
            const shown = !!transcript.children[1].scrolledTo;
            editor().cancel();
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

    def test_the_outline_is_only_on_the_conversation_the_edit_is_in(self):
        """The thread on screen can change under an open dialog, and the same
        index there is somebody else's message."""
        found = run_edit("""
            shell.startEdit(MESSAGES[1], 7);
            view = Object.assign({}, view, {selection: {character: "Ada", thread: "t2",
                                                        epoch: "e2"}});
            shell.markEditTarget();
            const away = mode().targets;
            view = Object.assign({}, view, {selection: {character: "Ada", thread: "t1",
                                                        epoch: "e1"}});
            shell.markEditTarget();
            console.log(JSON.stringify({away, back: mode().targets, open: mode().open}));
        """)

        assert found == {"away": [], "back": ["1"], "open": True}

    def test_a_second_press_on_edit_keeps_what_is_being_typed(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            box().value = "half of my edit";
            shell.startEdit(MESSAGES[0], 7);
            console.log(JSON.stringify(mode()));
        """)

        assert found["editing"] is True and found["open"] is True
        assert found["value"] == "half of my edit"

    def test_edit_on_another_message_moves_the_dialog_to_it(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            box().value = "half of my edit";
            shell.startEdit(MESSAGES[1], 7);
            console.log(JSON.stringify(Object.assign(mode(), {index: shell.editing.index})));
        """)

        assert found["index"] == 1
        assert found["value"] == "A harbour seen from above, 35mm"
        assert found["title"] == "Edit Ada’s reply"
        assert found["targets"] == ["1"]

    def test_without_the_dialog_s_script_the_press_says_so(self):
        found = run_edit("""
            delete window.mcMessageEditor;
            const began = shell.startEdit(MESSAGES[0], 7);
            console.log(JSON.stringify({began, editing: !!shell.editing,
                                        said: shell.nodes.status.textContent}));
        """)

        assert found == {"began": False, "editing": False,
                         "said": "The message editor is not loaded. Reload the page to edit."}


class TestDoneSavesTheEdit:
    """"i just need the ability to submit the edit and have it replace what
    was there.\""""

    def test_done_replaces_the_message_it_was_aimed_at(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            box().value = "make it a top down view, at dusk";
            editor().done();
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
        """The selection can move under an open dialog; Done is still aimed
        at the edit's thread."""
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            box().value = "edited";
            view = Object.assign({}, view, {selection: {character: "Ada", thread: "t2",
                                                        epoch: "e2"}});
            editor().done().then(() => console.log(JSON.stringify(store.sent[0].conversation)));
        """)

        assert found == {"character": "Ada", "thread_id": "t1"}

    def test_enter_is_done_and_shift_enter_is_a_new_line(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            box().value = "edited";
            let prevented = [];
            editor().keyed({key: "Enter", shiftKey: true, preventDefault() { prevented.push("shift"); }});
            const afterShift = store.sent.length;
            editor().keyed({key: "Enter", preventDefault() { prevented.push("plain"); }});
            settle().then(() => console.log(JSON.stringify(
                {prevented, afterShift, actions: store.sent.map((e) => e.action)})));
        """)

        assert found == {"prevented": ["plain"], "afterShift": 0, "actions": ["edit_message"]}

    def test_afterwards_the_dialog_is_gone_and_the_draft_was_never_touched(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            box().value = "edited";
            editor().done().then(() => console.log(JSON.stringify(Object.assign(mode(),
                {typed: store.typed}))));
        """)

        assert found["editing"] is False and found["open"] is False
        assert found["draft"] == "half a thought"
        assert found["said"] == "Message edited."
        assert found["typed"] == []
        assert found["targets"] == []

    def test_a_refused_save_keeps_the_dialog_open_and_says_why_in_it(self):
        found = run_edit("""
            store.answer = {ok: false, error: {message: "That message changed in another window."}};
            shell.startEdit(MESSAGES[0], 7);
            box().value = "my careful edit";
            editor().done().then(() => console.log(JSON.stringify(Object.assign(mode(),
                {doneEnabled: !editor().controls.done.disabled,
                 noteKind: editor().controls.note.dataset.kind}))));
        """)

        assert found["editing"] is True and found["open"] is True
        assert found["value"] == "my careful edit"
        assert found["note"] == "That message changed in another window."
        assert found["noteKind"] == "error"
        assert found["doneEnabled"] is True, "so it can be tried again"
        assert found["said"] == ("That message changed in another window. "
                                 "Your edit is still in the editor.")
        assert found["kind"] == "warn"

    def test_a_save_that_throws_is_a_refusal_too(self):
        found = run_edit("""
            store.send = () => Promise.reject(new Error("The server went away."));
            shell.startEdit(MESSAGES[0], 7);
            box().value = "my careful edit";
            editor().done().then(() => console.log(JSON.stringify(mode())));
        """)

        assert found["editing"] is True and found["open"] is True
        assert found["value"] == "my careful edit"
        assert found["note"] == "The server went away."

    def test_a_second_press_of_done_while_saving_sends_nothing_more(self):
        found = run_edit("""
            let release;
            store.send = (envelope) => { store.sent.push(envelope);
                return new Promise((resolve) => { release = () => resolve({ok: true}); }); };
            shell.startEdit(MESSAGES[0], 7);
            box().value = "edited";
            editor().done();
            const greyed = !!editor().controls.done.disabled;
            editor().done();
            shell.saveEdit("edited again");
            const during = store.sent.length;
            release();
            settle().then(() => settle()).then(() => console.log(JSON.stringify(
                {during, greyed, editing: !!shell.editing, open: mode().open})));
        """)

        assert found == {"during": 1, "greyed": True, "editing": False, "open": False}

    def test_nothing_changed_is_not_sent(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            editor().done().then(() => console.log(JSON.stringify(Object.assign(mode(),
                {sent: store.sent.length}))));
        """)

        assert found["sent"] == 0
        assert found["editing"] is False and found["open"] is False
        assert found["said"] == "Nothing was changed."

    def test_an_empty_box_is_not_saved(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            box().value = "   ";
            editor().done().then(() => console.log(JSON.stringify(Object.assign(mode(),
                {sent: store.sent.length}))));
        """)

        assert found["sent"] == 0
        assert found["editing"] is True and found["open"] is True
        assert found["note"] == "Type the message’s new words, or press Cancel to keep it."

    def test_done_waits_for_a_reply_that_is_being_written(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            box().value = "edited";
            const idle = shell.canSaveEdit(view);
            view = Object.assign({}, view, {operation: {id: "o1", terminal: false}});
            const busy = shell.canSaveEdit(view);
            editor().done().then(() => console.log(JSON.stringify(Object.assign(mode(),
                {idle, busy, sent: store.sent.length}))));
        """)

        assert found["idle"] is True and found["busy"] is False
        assert found["sent"] == 0 and found["open"] is True
        assert found["note"] == ("A reply is being written. Wait for it to finish, "
                                 "then press Done.")


class TestLeavingAnEdit:
    def test_cancel_puts_the_dialog_away_and_sends_nothing(self):
        found = run_edit("""
            shell.startEdit(MESSAGES[0], 7);
            box().value = "a change I no longer want";
            editor().cancel();
            console.log(JSON.stringify(Object.assign(mode(), {sent: store.sent.length})));
        """)

        assert found["editing"] is False and found["open"] is False
        assert found["draft"] == "half a thought"
        assert found["sent"] == 0
        assert found["said"] == "Edit cancelled. The message is as it was."

    def test_escape_is_the_dialog_s_own_and_cancels_the_edit(self):
        """The browser's `cancel` on a modal dialog. Taken over so the panel
        hears the edit was given up -- and never reaches the panel's own
        Escape, which would have left focus mode behind the dialog."""
        found = run_edit("""
            shell.focus = {isActive: () => true};
            shell.nodes.menu = {hidden: true};
            shell.toggleFocus = () => { shell.leftFocus = true; };
            shell.startEdit(MESSAGES[0], 7);
            let prevented = false;
            shell.documentKey({key: "Escape", preventDefault() { prevented = true; },
                               stopPropagation() {}});
            const stillEditing = !!shell.editing;
            let dialogPrevented = false;
            editor().dialog.handlers.cancel.forEach((fn) => fn(
                {preventDefault() { dialogPrevented = true; }}));
            console.log(JSON.stringify({stillEditing, prevented, dialogPrevented,
                                        editing: !!shell.editing, open: mode().open,
                                        leftFocus: !!shell.leftFocus,
                                        said: shell.nodes.status.textContent}));
        """, sources=("shell", "host", "store", "editor"))

        assert found == {"stillEditing": True, "prevented": False, "dialogPrevented": True,
                         "editing": False, "open": False, "leftFocus": False,
                         "said": "Edit cancelled. The message is as it was."}

    def test_another_conversation_leaves_the_dialog_open_and_the_save_aimed(self):
        """The box used to be shared with the draft, so a change of
        conversation put the edit away. The dialog is nobody's but the
        edit's: it stays, the outline moves off the thread that is not the
        edit's, and Done still goes where the edit began."""
        found = run_edit("""
            ["renderChip", "renderAutoAttach", "renderReadAloud",
             "renderStatus", "applySuppression"].forEach((name) => {
                shell[name] = () => {};
            });
            Object.defineProperty(transcript, "innerHTML", {
                get() { return ""; }, set() { transcript.children = []; }});
            shell.nodes.jump = made("button");
            shell.nodes.unread = made("span");
            shell.nodes.stop = made("button");
            shell.canSend = () => true;
            shell.startEdit(MESSAGES[0], 7);
            box().value = "edited";
            view = Object.assign({}, view, {
                selection: {character: "Ada", thread: "t2", epoch: "e2"},
                conversation: {conversation: {character: "Ada", thread_id: "t2", revision: 1},
                               messages: MESSAGES},
                draft: drafts[NS.conversationKey("Ada", "t2")]});
            shell.render(view);
            const during = mode();
            editor().done().then(() => console.log(JSON.stringify(
                {during, conversation: store.sent[0].conversation})));
        """)

        assert found["during"]["editing"] is True and found["during"]["open"] is True
        assert found["during"]["value"] == "edited"
        assert found["during"]["targets"] == []
        assert found["during"]["draft"] == "the other thread's draft"
        assert found["conversation"] == {"character": "Ada", "thread_id": "t1"}

    def test_a_long_edit_gets_a_taller_box_up_to_the_glass(self):
        found = run_edit("""
            globalThis.visualViewport = {offsetLeft: 0, offsetTop: 0, width: 1280, height: 800};
            shell.startEdit(MESSAGES[0], 7);
            box().scrollHeight = 40;
            editor().grow();
            const short = [box().style.height, box().style.overflowY];
            box().scrollHeight = 900;
            editor().grow();
            console.log(JSON.stringify({short, tall: [box().style.height, box().style.overflowY]}));
        """)

        assert found["short"] == ["96px", "hidden"], "four lines at least"
        assert found["tall"] == ["768px", "auto"], "the glass less its margins, then scrolls"


class TestTheEditLooksLikeAnEdit:
    def test_the_message_is_outlined_in_the_accent(self):
        target = rule(".forge-assistant-bubble.forge-assistant-edit-target")

        assert "var(--color-accent" in target and "accent-soft" not in target

    def test_the_dialog_is_not_full_screen(self):
        """"This new pop up should not be full screen, make it reasonable."
        The system prompt editor fills the window with `inset: 0`; this one
        is placed and sized by the script from the glass, and is a box with
        corners on top of the page."""
        dialog = rule(".mc-message-editor")

        assert "position: fixed" in dialog
        assert "inset: 0" not in dialog
        assert "border-radius" in dialog
        assert re.search(r"^\s*display\s*:", dialog, re.M) is None, (
            "an author display would keep a closed dialog on the page")

    def test_done_and_cancel_are_a_finger_tall(self):
        assert "min-height: 44px" in rule(".mc-message-editor-action")

    def test_the_box_does_not_make_a_phone_zoom(self):
        assert "font-size: max(16px, 1em)" in rule(".mc-message-editor-text")

    def test_the_composer_has_no_edit_strip_any_more(self):
        bare = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)

        assert ".forge-assistant-edit-bar" not in bare
        assert ".forge-assistant-editing" not in bare
        shell = SHELL.read_text(encoding="utf-8")
        assert "editBar" not in shell and "applyEditing" not in shell


# --------------------------------------------------------------------------- #
# 2. The ⋯ menu's switches say whether they are on
# --------------------------------------------------------------------------- #

MENU = """
// The store's script supplies this on a real page; the key is all it is for.
if (!NS.basePath) NS.basePath = () => "";
const box = {};
globalThis.localStorage = {getItem: (k) => box[k] === undefined ? null : box[k],
                           setItem: (k, v) => { box[k] = String(v); }};
const shell = Object.create(NS.Shell.prototype);
shell.state = {freeFloat: true, autoAttach: false, sendToGenerate: true};
shell.host = {listUtilities: () => []};
shell.store = {snapshot: () => ({selection: {character: "Ada", thread: "t1"}})};
shell.nodes = {status: {dataset: {}, textContent: ""}};
shell.closeMenu = () => undefined;
"""


class TestTheMenuSaysWhatIsOn:
    """"i want the menu items to be state wise. For example, today when i have
    on free form, the state is not in the menu name. Maybe just add a check.\""""

    def test_every_switch_reports_its_state_and_keeps_its_name(self):
        found = run(MENU + """
            console.log(JSON.stringify(shell.utilityItems()
                .filter((item) => item.getAttribute("role") === "menuitemcheckbox")
                .map((item) => [item.textContent, item.getAttribute("aria-checked")])));
        """)

        # In the menu's order: the Composer group's two, then Free Float in
        # the Panel group.
        assert found == [["Auto Attach", "false"], ["Send to Generate", "true"],
                         ["Free Float", "true"]]

    def test_the_check_is_drawn_from_that_state(self):
        on = rule('.forge-assistant-menu-item[role="menuitemcheckbox"]'
                  '[aria-checked="true"]::before')
        off = rule('.forge-assistant-menu-item[role="menuitemcheckbox"]::before')

        assert 'content: "\\2713";' in on
        # Decoration to a screen reader, which hears aria-checked already --
        # and after the plain form, which is what an engine that cannot read
        # the second keeps.
        assert on.index('content: "\\2713";') < on.index('content: "\\2713" / "";')
        assert 'content: "";' in off and "position: absolute" in off

    def test_the_labels_line_up_whether_or_not_a_switch_is_on(self):
        item = rule('.forge-assistant-menu[data-which="utilities"] .forge-assistant-menu-item')

        assert "padding-left: 2em" in item and "position: relative" in item

    def test_no_switch_has_a_look_of_its_own_any_more(self):
        """Free Float was bordered and bold when on and Auto Attach was drawn
        no differently at all: one rule, for every switch, is the cure."""
        bare = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)

        assert '.forge-assistant-float[aria-checked' not in bare


# --------------------------------------------------------------------------- #
# 3. Send to Generate
# --------------------------------------------------------------------------- #

GENERATE = """
// A Forge-shaped page: each image tab's prompt box and its Generate, with
// Interrupt/Skip/Interrupting beside it the way Forge draws them. A press on
// Generate records the prompt it would have read.
const box = {};
globalThis.localStorage = {getItem: (k) => box[k] === undefined ? null : box[k],
                           setItem: (k, v) => { box[k] = String(v); }};
const log = [];
const page = {};
function control(id, tag) {
    const node = document.createElement(tag || "button");
    node.id = id;
    page[id] = node;
    return node;
}
function tab(name, prompt) {
    const area = control(name + "_prompt_area", "textarea");
    area.value = prompt;
    const holder = control(name + "_prompt", "div");
    holder.querySelector = () => area;
    const generate = control(name + "_generate");
    generate.pressed = [];
    generate.click = () => { log.push("click:" + name); generate.pressed.push(area.value); };
    control(name + "_interrupt");
    control(name + "_skip");
    control(name + "_interrupting");
    return {area, generate};
}
const txt = tab("txt2img", "portrait of a woman, <lora:detail:0.5> blue hat -[[__lighting__]]");
const img = tab("img2img", "img2img words");
const previousLookup = document.getElementById;
document.getElementById = (id) => page[id] || previousLookup(id);
globalThis.updateInput = (target) => log.push("input:" + target.value.split("\\n")[0]);
globalThis.requestAnimationFrame = (fn) => { log.push("frame"); fn(); return 1; };
const settle = () => new Promise((resolve) => setImmediate(resolve));
const REPLY = {index: 1, role: "assistant", text: "A misty harbour at dawn, 35mm", active: 0};
function generateShell(workspace, on) {
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {sendToGenerate: on !== false};
    shell.nodes = {status: {dataset: {}, textContent: ""}};
    shell.host = {getActiveWorkspace: () => workspace};
    shell.switched = [];
    shell.switchWorkspace = (id) => { shell.switched.push(id); log.push("switch:" + id);
                                      return Promise.resolve(id); };
    return shell;
}
"""


def run_generate(scenario):
    return run(GENERATE + scenario, sources=("shell",))


class TestSendToGenerate:
    """"If "Send to Generate" is turned on, pressing the button under a reply
    would send the prompt in to replace the current, and invoke a generation
    as if user pressed the button ... If the send to generate is turned off,
    then it just does what it does today which is prompt replacement.\""""

    def test_it_is_a_switch_in_the_menu_remembered_in_this_browser(self):
        found = run(MENU + """
            shell.state.sendToGenerate = false;
            const item = () => shell.utilityItems().find((i) => i.textContent === "Send to Generate");
            const before = item().getAttribute("aria-checked");
            item().handlers.click.forEach((fn) => fn());
            const fresh = Object.create(NS.Shell.prototype);
            fresh.state = {};
            fresh._restoreSendToGenerate();
            console.log(JSON.stringify({before, on: shell.state.sendToGenerate,
                                        restored: fresh.state.sendToGenerate,
                                        after: item().getAttribute("aria-checked"),
                                        said: shell.nodes.status.textContent}));
        """)

        assert found["before"] == "false"
        assert found["on"] is True and found["restored"] is True
        assert found["after"] == "true"
        assert found["said"].startswith("Send to Generate is on")

    def test_it_starts_off(self):
        found = run(MENU + """
            const fresh = Object.create(NS.Shell.prototype);
            fresh.state = {};
            fresh._restoreSendToGenerate();
            console.log(JSON.stringify({on: fresh.state.sendToGenerate}));
        """)

        assert found == {"on": False}

    def test_off_the_button_only_writes_the_prompt(self):
        found = run_generate("""
            const shell = generateShell("tab_txt2img", false);
            const wrote = shell.sendToPrompt(REPLY);
            settle().then(() => console.log(JSON.stringify({wrote, log, prompt: txt.area.value})));
        """)

        assert found["wrote"] is True
        assert found["log"] == ["input:A misty harbour at dawn, 35mm"]
        assert found["prompt"] == ("A misty harbour at dawn, 35mm\n"
                                   "<lora:detail:0.5> -[[__lighting__]]")

    def test_on_it_writes_the_prompt_waits_for_the_page_and_presses_generate(self):
        found = run_generate("""
            const shell = generateShell("tab_txt2img");
            shell.sendToPrompt(REPLY).then((pressed) => console.log(JSON.stringify({
                pressed, log, generated: txt.generate.pressed,
                said: shell.nodes.status.textContent})));
        """)

        assert found["pressed"] is True
        assert found["log"] == ["input:A misty harbour at dawn, 35mm", "frame", "frame",
                                "click:txt2img"]
        assert found["generated"] == ["A misty harbour at dawn, 35mm\n"
                                      "<lora:detail:0.5> -[[__lighting__]]"], \
            "the reply, with the LoRA and the literal kept"
        assert found["said"] == "Prompt sent to txt2img and generating."

    def test_on_img2img_it_generates_in_img2img(self):
        found = run_generate("""
            const shell = generateShell("tab_img2img");
            shell.sendToPrompt(REPLY).then(() => console.log(JSON.stringify({
                img: img.generate.pressed, txt: txt.generate.pressed,
                switched: shell.switched})));
        """)

        assert found == {"img": ["A misty harbour at dawn, 35mm"], "txt": [], "switched": []}

    def test_from_any_other_tab_it_goes_to_txt2img_first(self):
        found = run_generate("""
            const shell = generateShell("tab_extras");
            shell.sendToPrompt(REPLY).then(() => console.log(JSON.stringify({log,
                switched: shell.switched})));
        """)

        assert found["switched"] == ["tab_txt2img"]
        assert found["log"][0] == "switch:tab_txt2img", "the tab first, then the prompt"
        assert found["log"][-1] == "click:txt2img"

    def test_a_tab_that_would_not_open_still_generates_and_says_where(self):
        found = run_generate("""
            const shell = generateShell("tab_extras");
            shell.switchWorkspace = () => Promise.resolve("");
            shell.sendToPrompt(REPLY).then((pressed) => console.log(JSON.stringify({
                pressed, said: shell.nodes.status.textContent})));
        """)

        assert found["pressed"] is True
        assert found["said"] == ("Prompt sent to txt2img and generating there -- its tab "
                                 "could not be opened.")

    def test_a_run_in_progress_is_not_pressed_again(self):
        """Forge covers Generate with Interrupt and Skip while a run is on; a
        person cannot press it then, and neither does this."""
        found = run_generate("""
            page.txt2img_interrupt.style.display = "block";
            const shell = generateShell("tab_txt2img");
            shell.sendToPrompt(REPLY).then((pressed) => console.log(JSON.stringify({
                pressed, generated: txt.generate.pressed, prompt: txt.area.value,
                said: shell.nodes.status.textContent, kind: shell.nodes.status.dataset.kind})));
        """)

        assert found["pressed"] is False
        assert found["generated"] == []
        assert found["prompt"].startswith("A misty harbour at dawn, 35mm"), "still written"
        assert found["said"] == ("txt2img is already generating. The prompt is in place: "
                                 "press Generate when it finishes.")
        assert found["kind"] == "warn"

    def test_an_interruption_under_way_is_a_run_in_progress(self):
        found = run_generate("""
            page.txt2img_interrupting.style.display = "block";
            const shell = generateShell("tab_txt2img");
            shell.sendToPrompt(REPLY).then((pressed) => console.log(JSON.stringify({pressed})));
        """)

        assert found == {"pressed": False}

    def test_a_hidden_generate_is_a_run_in_progress(self):
        found = run_generate("""
            txt.generate.style.display = "none";
            const shell = generateShell("tab_txt2img");
            shell.sendToPrompt(REPLY).then((pressed) => console.log(JSON.stringify({pressed})));
        """)

        assert found == {"pressed": False}

    def test_a_generate_that_cannot_be_pressed_is_not_pressed(self):
        found = run_generate("""
            txt.generate.disabled = true;
            const shell = generateShell("tab_txt2img");
            shell.sendToPrompt(REPLY).then((pressed) => console.log(JSON.stringify({
                pressed, generated: txt.generate.pressed})));
        """)

        assert found == {"pressed": False, "generated": []}

    def test_a_run_that_starts_during_the_wait_is_not_pressed_again(self):
        found = run_generate("""
            globalThis.requestAnimationFrame = (fn) => {
                page.txt2img_skip.style.display = "block";
                fn();
                return 1;
            };
            const shell = generateShell("tab_txt2img");
            shell.sendToPrompt(REPLY).then((pressed) => console.log(JSON.stringify({
                pressed, generated: txt.generate.pressed})));
        """)

        assert found == {"pressed": False, "generated": []}

    def test_a_prompt_changed_during_the_wait_is_not_generated(self):
        found = run_generate("""
            globalThis.requestAnimationFrame = (fn) => {
                txt.area.value = "somebody typed";
                fn();
                return 1;
            };
            const shell = generateShell("tab_txt2img");
            shell.sendToPrompt(REPLY).then((pressed) => console.log(JSON.stringify({
                pressed, generated: txt.generate.pressed,
                said: shell.nodes.status.textContent})));
        """)

        assert found["pressed"] is False and found["generated"] == []
        assert found["said"] == ("The txt2img prompt changed before Generate was pressed, "
                                 "so it was not pressed.")

    def test_a_page_without_generate_says_so(self):
        found = run_generate("""
            delete page.txt2img_generate;
            const shell = generateShell("tab_txt2img");
            shell.sendToPrompt(REPLY).then((pressed) => console.log(JSON.stringify({
                pressed, said: shell.nodes.status.textContent})));
        """)

        assert found["pressed"] is False
        assert found["said"] == ("Prompt sent to txt2img, but there is no Generate button "
                                 "to press on this page.")

    def test_a_reply_with_no_words_sends_nothing(self):
        found = run_generate("""
            const shell = generateShell("tab_txt2img");
            shell.sendToPrompt(Object.assign({}, REPLY, {text: "  "})).then((pressed) =>
                console.log(JSON.stringify({pressed, log})));
        """)

        assert found == {"pressed": False, "log": []}


class TestTheReplyButtonSaysWhichItDoes:
    """The same press starts a generation when the switch is on, so the button
    says so -- it should never be a surprise."""

    BUBBLE = """
    // A browser keeps className and classList as one; the harness's stub does
    // not, and the code under test reads classList.
    const makeElement = document.createElement;
    document.createElement = (tag) => {
        const node = makeElement(tag);
        let name = "";
        Object.defineProperty(node, "className", {
            get() { return name; },
            set(value) {
                name = String(value || "");
                name.split(/\\s+/).filter(Boolean).forEach((one) => node.classList.add(one));
            },
        });
        return node;
    };
    if (!NS.basePath) NS.basePath = () => "";
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {sendToGenerate: true};
    shell.settings = {bubbleWidth: 80};
    shell.nodes = {transcript: document.createElement("div"),
                   status: {dataset: {}, textContent: ""}};
    const row = {index: 1, role: "assistant", text: "a", active: 0, versions: ["a"]};
    const view = {conversation: {conversation: {revision: 3},
                                 messages: [{index: 0, role: "user", text: "q"}, row]}};
    const bubble = shell.bubble(row, view);
    shell.nodes.transcript.appendChild(bubble);
    const send = () => bubble.children.find((c) => c.classList.contains("forge-assistant-actions"))
        .children.find((b) => b.dataset.action === "send_prompt");
    const drawn = (b) => [b.textContent, b.title, b["aria-label"]];
    """

    def test_on_it_is_a_play_button_named_send_to_generate(self):
        found = run(self.BUBBLE + """
            console.log(JSON.stringify(drawn(send())));
        """)

        assert found == ["▶︎", "Send to Generate", "Send to Generate"]

    def test_the_buttons_already_drawn_follow_the_switch(self):
        """Bubbles are kept across renders, so the switch tells the ones on
        the page rather than waiting for them to be rebuilt."""
        found = run(self.BUBBLE + """
            globalThis.localStorage = {setItem() {}, getItem() { return null; }};
            const on = drawn(send());
            shell.setSendToGenerate(false);
            console.log(JSON.stringify({on, off: drawn(send())}));
        """)

        assert found["on"][1] == "Send to Generate"
        assert found["off"] == ["➤", "Send to prompt", "Send to prompt"]
