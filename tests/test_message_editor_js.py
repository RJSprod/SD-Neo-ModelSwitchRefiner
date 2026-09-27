"""The message editor's dialog, executed rather than read.

One dialog for editing a message, shared by LLM Studio's Conversation tab and
the Forge Assistant's panel: "a simple pop up ... current text in an input
field, cancel, and a done button ... not full screen, make it reasonable, and
responsive to the browser available space so it also looks good on mobile."

What is worth running is the contract with whoever opens it -- what Done and
Cancel hand back, what a refused save leaves, what a second open does to the
first -- and the arithmetic that sizes it from the glass, because a dialog that
is placed against the layout viewport is under a phone's keyboard, and the
mistake looks correct on a desktop.

Run under node in the harness `test_assistant_js.py` uses, with the editor's
file alone. Every test here was checked against the change it guards by
reverting that change and watching the test fail.
"""

from __future__ import annotations

import re

from test_assistant_js import EDITOR, run

CSS = (EDITOR.parent.parent / "style.css").read_text(encoding="utf-8")


def rule(selector):
    bare = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    found = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", bare):
        names = [piece.strip() for piece in match.group(1).split(",")]
        if selector in names:
            found.append(match.group(2))
    assert found, selector + " has no rule of its own"
    return "\n".join(found)


DIALOG = """
const editor = window.mcMessageEditor;
const settle = () => new Promise((resolve) => setImmediate(resolve));
// What the dialog is told and what it is told back.
const log = [];
function request(over) {
    return Object.assign({
        title: "Edit your message",
        text: "make it top down view",
        placeholder: "The message’s new words…",
        done(text) { log.push(["done", text]); return {ok: true}; },
        cancel() { log.push(["cancel"]); },
    }, over || {});
}
const it = () => editor.editor();
const c = () => it().controls;
const state = () => ({
    open: editor.isOpen(), value: c().text.value, title: c().heading.textContent,
    note: c().note.textContent, kind: c().note.dataset.kind,
    doneEnabled: !c().done.disabled, busy: it().dialog.dataset.busy, log,
});
"""


def run_dialog(scenario, viewport=None):
    return run(DIALOG + scenario, viewport=viewport, sources=("editor",))


class TestOpening:
    def test_it_is_one_native_modal_dialog_on_the_body_that_says_it_is_one(self):
        """Focus mode spares what says it is a dialog; the top layer is where
        a modal draws; and a second open must not build a second one."""
        found = run_dialog("""
            editor.open(request());
            editor.open(request({text: "again"}));
            const dialog = it().dialog;
            console.log(JSON.stringify({
                dialogs: document.body.children.filter((n) => n.tagName === "DIALOG").length,
                tag: dialog.tagName, role: dialog.role, modal: dialog["aria-modal"],
                labelled: dialog["aria-labelledby"], heading: c().heading.id,
                opened: dialog.open !== undefined,
            }));
        """)

        assert found["dialogs"] == 1
        assert found["tag"] == "DIALOG" and found["role"] == "dialog"
        assert found["modal"] == "true"
        assert found["labelled"] == found["heading"] == "mc-message-editor-heading"
        assert found["opened"] is True

    def test_the_words_the_title_and_the_placeholder_are_the_caller_s(self):
        found = run_dialog("""
            editor.open(request());
            console.log(JSON.stringify(Object.assign(state(), {
                placeholder: c().text.placeholder, focused: !!c().text.focused})));
        """)

        assert found["open"] is True
        assert found["value"] == "make it top down view"
        assert found["title"] == "Edit your message"
        assert found["placeholder"] == "The message’s new words…"
        assert found["focused"] is True
        assert found["note"] == "" and found["doneEnabled"] is True

    def test_the_caret_goes_to_the_end_of_the_words(self):
        found = run_dialog("""
            it().build();
            c().text.setSelectionRange = (a, b) => { c().text.caret = [a, b]; };
            editor.open(request());
            console.log(JSON.stringify({caret: c().text.caret}));
        """)

        assert found == {"caret": [21, 21]}

    def test_nothing_is_open_until_something_opens_it(self):
        found = run_dialog("""
            console.log(JSON.stringify({open: editor.isOpen(), cancelled: editor.cancel()}));
        """)

        assert found == {"open": False, "cancelled": False}

    def test_a_second_open_replaces_the_first_and_tells_it_so(self):
        found = run_dialog("""
            editor.open(request({cancel() { log.push(["cancel-first"]); }}));
            c().text.value = "half typed";
            editor.open(request({title: "Edit Ada’s reply", text: "the reply"}));
            console.log(JSON.stringify(state()));
        """)

        assert found["log"] == [["cancel-first"]]
        assert found["value"] == "the reply"
        assert found["title"] == "Edit Ada’s reply"
        assert found["open"] is True


class TestDone:
    def test_done_hands_the_words_over_and_closes(self):
        found = run_dialog("""
            editor.open(request());
            c().text.value = "make it a top down view, at dusk";
            it().done().then((closed) => console.log(JSON.stringify(
                Object.assign(state(), {closed}))));
        """)

        assert found["log"] == [["done", "make it a top down view, at dusk"]]
        assert found["closed"] is True and found["open"] is False

    def test_a_refusal_keeps_it_open_with_the_reason_under_the_box(self):
        found = run_dialog("""
            editor.open(request({done(text) { log.push(["done", text]);
                return {ok: false, message: "That message changed in another window."}; }}));
            c().text.value = "my careful edit";
            it().done().then((closed) => console.log(JSON.stringify(
                Object.assign(state(), {closed, focused: !!c().text.focused}))));
        """)

        assert found["closed"] is False and found["open"] is True
        assert found["value"] == "my careful edit"
        assert found["note"] == "That message changed in another window."
        assert found["kind"] == "error"
        assert found["doneEnabled"] is True, "so it can be tried again"
        assert found["focused"] is True

    def test_a_refusal_without_words_still_says_something(self):
        found = run_dialog("""
            editor.open(request({done() { return {ok: false}; }}));
            it().done().then(() => console.log(JSON.stringify(state())));
        """)

        assert found["note"] == "That edit was refused."

    def test_an_answer_on_its_way_greys_done_out_and_is_waited_for(self):
        found = run_dialog("""
            let release;
            editor.open(request({done(text) { log.push(["done", text]);
                return new Promise((resolve) => { release = resolve; }); }}));
            it().done();
            const during = state();
            it().done();
            release({ok: true});
            settle().then(() => settle()).then(() => console.log(JSON.stringify(
                {during: {busy: during.busy, doneEnabled: during.doneEnabled, open: during.open},
                 after: state(), handed: log.length})));
        """)

        assert found["during"] == {"busy": "true", "doneEnabled": False, "open": True}
        assert found["handed"] == 1, "a second press while waiting hands nothing more over"
        assert found["after"]["open"] is False and found["after"]["busy"] == "false"

    def test_a_done_that_throws_or_rejects_is_a_refusal(self):
        found = run_dialog("""
            editor.open(request({done() { throw new Error("The server went away."); }}));
            it().done().then(() => {
                const thrown = state().note;
                editor.open(request({done() { return Promise.reject(new Error("Lost.")); }}));
                return it().done().then(() => console.log(JSON.stringify(
                    {thrown, rejected: state().note, open: state().open})));
            });
        """)

        assert found == {"thrown": "The server went away.", "rejected": "Lost.", "open": True}

    def test_an_answer_to_an_edit_that_was_given_up_is_not_shown(self):
        found = run_dialog("""
            let release;
            editor.open(request({done() { return new Promise((resolve) => { release = resolve; }); }}));
            it().done();
            it().cancel();
            editor.open(request({text: "the next edit"}));
            release({ok: false, message: "Too late."});
            settle().then(() => settle()).then(() => console.log(JSON.stringify(state())));
        """)

        assert found["open"] is True
        assert found["value"] == "the next edit"
        assert found["note"] == ""


class TestCancel:
    def test_cancel_tells_the_caller_and_closes(self):
        found = run_dialog("""
            editor.open(request());
            c().text.value = "a change I no longer want";
            const cancelled = it().cancel();
            console.log(JSON.stringify(Object.assign(state(), {cancelled})));
        """)

        assert found["cancelled"] is True and found["open"] is False
        assert found["log"] == [["cancel"]]

    def test_cancel_is_allowed_while_a_save_is_still_being_answered(self):
        """A save that takes a minute is not a reason to be stuck in front
        of it."""
        found = run_dialog("""
            editor.open(request({done() { return new Promise(() => {}); }}));
            it().done();
            const cancelled = it().cancel();
            console.log(JSON.stringify(Object.assign(state(), {cancelled})));
        """)

        assert found["cancelled"] is True and found["open"] is False
        assert found["log"] == [["cancel"]]

    def test_escape_is_taken_over_so_the_caller_hears_it(self):
        """The browser's `cancel` on a modal dialog, and the key itself for a
        dialog opened without `showModal`, where no `cancel` comes."""
        found = run_dialog("""
            editor.open(request());
            let prevented = 0;
            it().dialog.handlers.cancel.forEach((fn) => fn({preventDefault() { prevented += 1; }}));
            const viaEvent = {open: editor.isOpen(), log: log.slice()};
            editor.open(request());
            let stopped = 0;
            it().dialog.handlers.keydown.forEach((fn) => fn({key: "Escape",
                preventDefault() { prevented += 1; }, stopPropagation() { stopped += 1; }}));
            it().dialog.handlers.keydown.forEach((fn) => fn({key: "a",
                preventDefault() { prevented += 1; }, stopPropagation() { stopped += 1; }}));
            console.log(JSON.stringify({viaEvent, viaKey: {open: editor.isOpen(), log},
                                        prevented, stopped}));
        """)

        assert found["viaEvent"] == {"open": False, "log": [["cancel"]]}
        assert found["viaKey"] == {"open": False, "log": [["cancel"], ["cancel"]]}
        assert found["prevented"] == 2 and found["stopped"] == 1


class TestKeys:
    def test_enter_is_done_and_shift_enter_is_a_new_line(self):
        found = run_dialog("""
            editor.open(request());
            const prevented = [];
            const key = (over) => Object.assign({key: "Enter",
                preventDefault() { prevented.push(over.name); }}, over);
            it().keyed(key({name: "shift", shiftKey: true}));
            it().keyed(key({name: "ime", isComposing: true}));
            it().keyed(key({name: "other", key: "a"}));
            it().keyed(key({name: "plain"}));
            settle().then(() => console.log(JSON.stringify({prevented, log})));
        """)

        assert found == {"prevented": ["plain"], "log": [["done", "make it top down view"]]}

    def test_ctrl_enter_is_done_as_well(self):
        found = run_dialog("""
            editor.open(request());
            it().keyed({key: "Enter", ctrlKey: true, preventDefault() {}});
            settle().then(() => console.log(JSON.stringify(log)));
        """)

        assert found == [["done", "make it top down view"]]


class TestFittingTheGlass:
    """"Responsive to the browser available space so it also looks good on
    mobile": sized from the visual viewport, never wider than 560, never
    closer than 16 to an edge, and near the top so a keyboard leaves it be."""

    @staticmethod
    def placed(found):
        style = found["style"]
        return {name: int(style[name][:-2]) for name in ("left", "top", "width", "maxHeight")}

    def fit(self, viewport, before=""):
        return run_dialog(before + """
            editor.open(request());
            const style = it().dialog.style;
            console.log(JSON.stringify({style: {left: style.left, top: style.top,
                width: style.width, maxHeight: style.maxHeight, right: style.right,
                bottom: style.bottom}}));
        """, viewport=viewport)

    def test_on_a_desktop_it_is_a_centred_card_and_not_the_window(self):
        found = self.placed(self.fit({"offsetLeft": 0, "offsetTop": 0,
                                      "width": 1280, "height": 800}))

        assert found["width"] == 560
        assert found["left"] == (1280 - 560) // 2
        assert found["maxHeight"] == 800 - 32
        assert found["top"] >= 16

    def test_on_a_phone_it_takes_the_width_less_the_margins(self):
        found = self.placed(self.fit({"offsetLeft": 0, "offsetTop": 0,
                                      "width": 390, "height": 720}))

        assert found["width"] == 390 - 32
        assert found["left"] == 16
        assert found["maxHeight"] == 720 - 32

    def test_with_the_keyboard_up_it_fits_what_is_left(self):
        """The visual viewport is what shrinks when a phone's keyboard opens;
        the layout viewport does not, and a dialog sized to it has its
        buttons under the keyboard. A dialog taller than what is left starts
        at the top margin and scrolls inside; a shorter one sits a third of
        the spare room down, and neither reaches past the bottom margin."""
        glass = {"offsetLeft": 0, "offsetTop": 0, "width": 390, "height": 360}
        tall = self.placed(self.fit(glass, before="""
            it().build();
            it().dialog.offsetHeight = 2000;
        """))
        short = self.placed(self.fit(glass, before="""
            it().build();
            it().dialog.offsetHeight = 200;
        """))

        assert tall["maxHeight"] == 360 - 32
        assert tall["top"] == 16
        assert short["top"] == (360 - 200) // 3
        assert short["top"] + 200 <= 360 - 16

    def test_a_pinched_page_is_measured_from_where_the_glass_is(self):
        found = self.placed(self.fit({"offsetLeft": 120, "offsetTop": 60,
                                      "width": 800, "height": 500}))

        assert found["left"] == 120 + (800 - 560) // 2
        assert found["top"] >= 60 + 16

    def test_a_tall_dialog_starts_at_the_top_margin_rather_than_off_the_glass(self):
        found = self.fit({"offsetLeft": 0, "offsetTop": 0, "width": 390, "height": 400},
                         before="""
            it().build();
            it().dialog.offsetHeight = 2000;
        """)

        assert found["style"]["top"] == "16px"
        assert found["style"]["right"] == "auto" and found["style"]["bottom"] == "auto"

    def test_the_box_grows_with_the_words_up_to_the_room_and_then_scrolls(self):
        found = run_dialog("""
            editor.open(request());
            const box = c().text;
            box.scrollHeight = 40;
            it().grow();
            const short = [box.style.height, box.style.overflowY];
            box.scrollHeight = 300;
            it().grow();
            const medium = [box.style.height, box.style.overflowY];
            box.scrollHeight = 5000;
            it().grow();
            console.log(JSON.stringify({short, medium, tall: [box.style.height, box.style.overflowY]}));
        """, viewport={"offsetLeft": 0, "offsetTop": 0, "width": 390, "height": 720})

        assert found["short"] == ["96px", "hidden"]
        assert found["medium"] == ["300px", "hidden"]
        assert found["tall"] == [str(720 - 32) + "px", "auto"]

    def test_it_follows_the_glass_while_open_and_lets_go_when_closed(self):
        found = run_dialog("""
            const listened = [];
            globalThis.visualViewport = {offsetLeft: 0, offsetTop: 0, width: 390, height: 720,
                addEventListener(type) { listened.push("+" + type); },
                removeEventListener(type) { listened.push("-" + type); }};
            editor.open(request());
            const during = listened.slice();
            it().cancel();
            console.log(JSON.stringify({during, after: listened}));
        """)

        assert found["during"] == ["+resize", "+scroll"]
        assert found["after"] == ["+resize", "+scroll", "-resize", "-scroll"]


class TestTheLook:
    def test_it_is_a_card_over_the_page_and_never_full_screen(self):
        dialog = rule(".mc-message-editor")

        assert "position: fixed" in dialog and "border-radius" in dialog
        assert "inset: 0" not in dialog
        assert re.search(r"^\s*display\s*:", dialog, re.M) is None

    def test_the_backdrop_dims_in_the_theme_s_own_ink(self):
        assert "var(--body-text-color" in rule(".mc-message-editor::backdrop")

    def test_the_buttons_are_a_finger_tall_and_done_is_the_primary_one(self):
        assert "min-height: 44px" in rule(".mc-message-editor-action")
        assert "--button-primary-background-fill" in rule(".mc-message-editor-done")

    def test_the_box_does_not_make_a_phone_zoom(self):
        assert "font-size: max(16px, 1em)" in rule(".mc-message-editor-text")

    def test_the_browser_s_prompt_is_never_used(self):
        assert not re.search(r"window\.prompt\s*\(", EDITOR.read_text(encoding="utf-8"))
