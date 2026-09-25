"""The flyout's third round: six clean-ups asked for together.

    1. "start a new thread" from the ⋯ menu;
    2. a full-page system prompt editor, in LLM Studio's character menu and in
       the ⋯ menu, like Mini Paint's;
    3. a message sent with Enter stayed in the box;
    4. the conversation a third shorter;
    5. the lit paperclip and read-aloud buttons were low contrast;
    6. a picture on a message never showed, and its file name was listed twice.

Each class is one of those, run against the real browser files in the node
harness `test_assistant_js.py` uses. Where the answer is a layout or a colour,
the stylesheet is read here and the rendering was checked in Chromium.
"""

from __future__ import annotations

import re

from test_assistant_js import SHELL, run

STORE = SHELL.parent / "forge_assistant_store.js"
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


def run_both(scenario):
    return run(scenario, sources=("shell", "store"))


# --------------------------------------------------------------------------- #
# 6. Pictures on messages
# --------------------------------------------------------------------------- #

PICTURES = """
// The key field Forge's LLM Studio renders, and a fetch that records what it
// was asked and answers as the scenario says.
const keyField = {tagName: "TEXTAREA", value: "the-key"};
const previousLookup = document.getElementById;
document.getElementById = (id) => id === "mc-llm-chat-conversation-key"
    ? keyField : previousLookup(id);
const asked = [];
let answer = (url) => Promise.resolve({ok: true,
    blob: () => Promise.resolve(new Blob(["png"], {type: "image/png"}))});
globalThis.fetch = (url, options) => { asked.push({url, options}); return answer(url); };
let made = 0;
const revoked = [];
globalThis.URL = {createObjectURL: () => "blob:picture-" + (made += 1),
                  revokeObjectURL: (url) => revoked.push(url)};
const settle = () => new Promise((resolve) => setImmediate(resolve));
function pictureShell() {
    const shell = Object.create(NS.Shell.prototype);
    shell.store = new NS.Store();
    return shell;
}
function words(node) {
    // Everything a person could read off the figure: text, alternative text,
    // a tooltip. None of it may be the file's name.
    const found = [];
    const walk = (at) => {
        if (!at) return;
        if (at.textContent) found.push(at.textContent);
        if (at.alt) found.push(at.alt);
        if (at.title) found.push(at.title);
        (at.children || []).forEach(walk);
    };
    walk(node);
    return found.join(" | ");
}
const ATTACHED = {name: "tmpsocax_1o.png", available: true,
                  url: "/model-chain/conversation/v2/attachment/TICKET"};
"""


class TestAPictureOnAMessageShows:
    """Reported with a screenshot: under "make it top down view", no picture and
    "tmpsocax_1o.png" twice. The <img> pointed at a route that refuses a request
    without the page's key -- which an <img> cannot send -- so the browser drew
    its alternative text, the name, above a caption that was the name again."""

    def test_the_picture_is_fetched_with_the_page_s_key(self):
        found = run_both(PICTURES + """
            const shell = pictureShell();
            const figure = shell.figure(ATTACHED);
            settle().then(settle).then(() => {
                const image = figure.children[0];
                console.log(JSON.stringify({
                    url: asked[0].url,
                    key: asked[0].options.headers["x-mc-conversation-key"],
                    src: image.src, alt: image.alt}));
            });
        """)

        assert found["url"] == "/model-chain/conversation/v2/attachment/TICKET"
        assert found["key"] == "the-key"
        assert found["src"] == "blob:picture-1"
        assert found["alt"] == "Attached picture"

    def test_the_file_s_name_is_never_shown(self):
        found = run_both(PICTURES + """
            const shell = pictureShell();
            const shown = shell.figure(ATTACHED);
            answer = () => Promise.resolve({ok: false, status: 401});
            const refused = shell.figure(Object.assign({}, ATTACHED,
                                                       {url: ATTACHED.url + "-2"}));
            const gone = shell.figure(Object.assign({}, ATTACHED, {available: false}));
            settle().then(settle).then(() => {
                console.log(JSON.stringify([words(shown), words(refused), words(gone)]));
            });
        """)

        for said in found:
            assert "tmpsocax" not in said, said

    def test_a_picture_that_cannot_be_shown_is_a_placeholder(self):
        found = run_both(PICTURES + """
            const shell = pictureShell();
            answer = () => Promise.resolve({ok: false, status: 404});
            const refused = shell.figure(ATTACHED);
            const gone = shell.figure(Object.assign({}, ATTACHED, {available: false}));
            const inline = shell.figure({name: "x.png", available: true, inline: true});
            settle().then(settle).then(() => {
                const shape = (figure) => ({
                    missing: figure.classList.contains("forge-assistant-attachment-missing"),
                    images: figure.children.filter((c) => c.tagName === "IMG").length,
                    said: words(figure)});
                console.log(JSON.stringify([shape(refused), shape(gone), shape(inline)]));
            });
        """)

        for shape in found:
            assert shape["missing"] is True
            assert shape["images"] == 0
            assert "Picture unavailable" in shape["said"]

    def test_a_picture_that_will_not_decode_becomes_the_placeholder(self):
        found = run_both(PICTURES + """
            const shell = pictureShell();
            const figure = shell.figure(ATTACHED);
            settle().then(settle).then(() => {
                figure.children[0].handlers.error.forEach((fn) => fn());
                console.log(JSON.stringify(figure.classList.contains(
                    "forge-assistant-attachment-missing")));
            });
        """)

        assert found is True

    def test_a_picture_is_fetched_once_however_often_it_is_drawn(self):
        """A thread is redrawn whenever it moves; a fetch per redraw would be a
        picture per message per word of a streaming reply."""
        found = run_both(PICTURES + """
            const store = new NS.Store();
            Promise.all([store.picture(ATTACHED.url), store.picture(ATTACHED.url)])
                .then((both) => console.log(JSON.stringify({asked: asked.length,
                                                            same: both[0] === both[1]})));
        """)

        assert found == {"asked": 1, "same": True}

    def test_a_failure_is_asked_again_next_time(self):
        found = run_both(PICTURES + """
            const store = new NS.Store();
            answer = () => Promise.resolve({ok: false, status: 500});
            store.picture(ATTACHED.url).catch(() => settle()).then(() => {
                answer = (url) => Promise.resolve({ok: true,
                    blob: () => Promise.resolve(new Blob(["png"]))});
                return store.picture(ATTACHED.url);
            }).then((made) => console.log(JSON.stringify({asked: asked.length, made})));
        """)

        assert found == {"asked": 2, "made": "blob:picture-1"}

    def test_a_long_thread_lets_its_oldest_pictures_go(self):
        found = run_both(PICTURES + """
            const store = new NS.Store();
            const all = [];
            for (let at = 0; at < 50; at += 1) all.push(store.picture("/p/" + at));
            Promise.all(all).then(settle).then(() => {
                console.log(JSON.stringify({kept: store.pictures.size, revoked}));
            });
        """)

        assert found["kept"] == 48
        assert found["revoked"] == ["blob:picture-1", "blob:picture-2"]

    def test_a_bubble_draws_its_picture_through_the_figure(self):
        found = run_both(PICTURES + """
            const shell = pictureShell();
            shell.settings = {bubbleWidth: 80};
            const row = {index: 0, role: "user", text: "make it top down view", active: 0,
                         versions: ["make it top down view"], attachment: ATTACHED};
            const node = shell.bubble(row, {conversation: {conversation: {revision: 1},
                                                           messages: [row, {role: "assistant"}]}});
            settle().then(settle).then(() => {
                const figure = node.children.find((c) => c.tagName === "FIGURE");
                console.log(JSON.stringify({src: figure.children[0].src,
                                            captions: figure.children.filter(
                                                (c) => c.tagName === "FIGCAPTION").length}));
            });
        """)

        assert found == {"src": "blob:picture-1", "captions": 0}

    def test_the_picture_is_held_to_the_bubble(self):
        image = rule(".forge-assistant-attachment img")

        assert "max-width: 100%" in image
        assert "max-height" in image
        assert "min-height" in rule(".forge-assistant-attachment-missing")


# --------------------------------------------------------------------------- #
# 3. Enter-to-send empties the box
# --------------------------------------------------------------------------- #

COMPOSER = """
// The composer with focus in it -- which is what Enter means -- and the real
// store's submit, answered by a stubbed send.
function composerShell(answerWith) {
    const shell = Object.create(NS.Shell.prototype);
    const store = new NS.Store();
    store.selection = {character: "Ada", thread: "t1", epoch: "e1"};
    store.ready = true;
    store.snapshots.set(NS.conversationKey("Ada", "t1"), {conversation: {revision: 1},
                                                           messages: []});
    let settle = null;
    store.send = () => new Promise((resolve) => { settle = () => resolve(answerWith); });
    store.refresh = () => Promise.resolve(null);
    shell.store = store;
    shell.state = {autoAttach: false};
    shell.autoSent = new Map();
    const input = document.createElement("textarea");
    input.style = {};
    shell.nodes = {input, status: {dataset: {}, textContent: ""}};
    shell.grow = () => undefined;
    document.activeElement = input;
    return {shell, input, store, answer: () => settle()};
}
const typeAndEnter = (shell, input, text) => {
    input.value = text;
    shell.store.setDraftText(text);
    shell.composerKey({key: "Enter", shiftKey: false, preventDefault() {}});
};
const later = () => new Promise((resolve) => setImmediate(resolve));
"""


class TestEnterEmptiesTheBox:
    """"if i hit enter on my keyboard when submitting prompt for flyout menu,
    the prompt stays in the input field." Enter leaves focus in the box, and
    the panel never rewrites a focused box -- so the store emptied the draft
    and the box went on showing it."""

    def test_a_message_sent_with_enter_leaves_the_box_empty(self):
        found = run_both(COMPOSER + """
            const {shell, input, answer} = composerShell({ok: true, phase: "completed",
                resulting_conversation: {character: "Ada", thread_id: "t1"}});
            typeAndEnter(shell, input, "make it top down view");
            later().then(() => { answer(); return later(); }).then(later).then(() => {
                console.log(JSON.stringify({box: input.value,
                                            focused: document.activeElement === input}));
            });
        """)

        assert found == {"box": "", "focused": True}

    def test_words_typed_while_it_was_sending_stay(self):
        found = run_both(COMPOSER + """
            const {shell, input, answer} = composerShell({ok: true, phase: "completed",
                resulting_conversation: {character: "Ada", thread_id: "t1"}});
            typeAndEnter(shell, input, "first");
            later().then(() => {
                input.value = "first, and more";
                shell.store.setDraftText("first, and more");
                answer();
                return later();
            }).then(later).then(() => console.log(JSON.stringify(input.value)));
        """)

        assert found == "first, and more"

    def test_a_message_that_was_refused_stays_in_the_box(self):
        found = run_both(COMPOSER + """
            const {shell, input, answer} = composerShell({ok: false,
                error: {message: "Refused."}});
            typeAndEnter(shell, input, "keep me");
            later().then(() => { answer(); return later(); }).then(later).then(() => {
                console.log(JSON.stringify(input.value));
            });
        """)

        assert found == "keep me"


# --------------------------------------------------------------------------- #
# 4. A third shorter
# --------------------------------------------------------------------------- #


class TestTheConversationIsAThirdShorter:
    """"The height of the flyout conversation mode is too tall. Lets reduce it
    by 1/3 but shrinking the conversation view." Measured in Chromium: the
    rest of the panel is 268px at every size and the transcript was 48vh, so
    two thirds of the old panel is a transcript of 32vh - 89px -- 652px became
    435px at 800 tall, 787px became 525px at 1080, 708px became 472px on a
    phone."""

    def test_the_transcript_is_what_gives_up_the_third(self):
        transcript = rule(".forge-assistant-transcript")

        assert "max-height: max(120px, calc(32vh - 89px));" in transcript
        assert "48vh" not in transcript

    def test_it_still_scrolls_and_follows(self):
        """"Keep the same rules for its behavior, just not as tall.\""""
        transcript = rule(".forge-assistant-transcript")

        assert "overflow-y: auto" in transcript
        assert "overscroll-behavior: contain" in transcript
        assert "min-height: 0" in transcript


# --------------------------------------------------------------------------- #
# 5. Lit buttons that stay readable
# --------------------------------------------------------------------------- #


class TestALitButtonKeepsItsGlyphReadable:
    """Reported with a screenshot: the lit paperclip a white tile with nothing
    on it, the lit speaker pale on pale. Both were filled with the theme's soft
    accent, which Lobe makes nearly white. Checked in Chromium under a dark
    theme with a white soft accent: the lit fill is dark indigo beside a
    near-black plain button (1.26:1 between the two fills), and light blue
    beside a light one in a light theme."""

    def test_both_lit_buttons_share_one_rule(self):
        lit = rule('.forge-assistant-read-aloud[data-state="on"]')

        assert lit == rule('.forge-assistant-attach[data-auto="on"]')

    def test_the_fill_is_a_tint_of_the_button_s_own_background(self):
        lit = rule('.forge-assistant-read-aloud[data-state="on"]')

        assert "color-mix(in srgb, var(--color-accent" in lit
        assert "var(--background-fill-secondary" in lit
        assert "border-color: var(--color-accent" in lit

    def test_no_part_of_the_panel_takes_a_fill_from_the_soft_accent(self):
        """The whole class of the defect, not the two buttons: the launcher's
        unread count had the same fill under body text, light on light."""
        bare = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
        offenders = []
        for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", bare):
            if "forge-assistant" not in match.group(1):
                continue
            for line in match.group(2).splitlines():
                if "background" in line and "--color-accent-soft" in line:
                    offenders.append(match.group(1).strip())
        assert offenders == []


# --------------------------------------------------------------------------- #
# 1. New thread from the ⋯ menu
# --------------------------------------------------------------------------- #

THREADS = """
function threadStore(answer) {
    const store = new NS.Store();
    store.selection = {character: "Ada", thread: "t1", epoch: "e1"};
    store.sent = [];
    store.send = (envelope) => { store.sent.push(envelope); return Promise.resolve(answer); };
    store.selected = [];
    store.select = (character, thread) => {
        store.selected.push([character, thread]);
        store.selection = {character, thread, epoch: "e2"};
        return Promise.resolve(null);
    };
    return store;
}
const MADE = {ok: true, operation_id: "op", phase: "completed",
              resulting_conversation: {character: "Ada", thread_id: "t9"}};
"""


class TestANewThreadFromTheMenu:
    """"Add the ability to start a new thread directly from the fly out menu.
    It should start a fresh thread with the character.\""""

    def test_it_asks_the_service_for_a_thread_with_this_character(self):
        found = run_both(THREADS + """
            const store = threadStore(MADE);
            store.createThread().then(() => {
                const sent = store.sent[0];
                console.log(JSON.stringify({action: sent.action,
                                            conversation: sent.conversation,
                                            expected: sent.expected_revision}));
            });
        """)

        assert found == {"action": "create_thread",
                         "conversation": {"character": "Ada", "thread_id": ""},
                         "expected": None}

    def test_the_panel_moves_onto_the_new_thread(self):
        found = run_both(THREADS + """
            const store = threadStore(MADE);
            store.createThread().then(() => console.log(JSON.stringify(store.selected)));
        """)

        assert found == [["Ada", "t9"]]

    def test_a_refusal_moves_nothing(self):
        found = run_both(THREADS + """
            const store = threadStore({ok: false, error: {message: "No."}});
            store.createThread().then((outcome) => console.log(JSON.stringify(
                {selected: store.selected, ok: outcome.ok})));
        """)

        assert found == {"selected": [], "ok": False}

    def test_with_no_character_nothing_is_sent(self):
        found = run_both(THREADS + """
            const store = threadStore(MADE);
            store.selection = {character: "", thread: "", epoch: "e1"};
            store.createThread().then((outcome) => console.log(JSON.stringify(
                {sent: store.sent.length, ok: outcome.ok})));
        """)

        assert found == {"sent": 0, "ok": False}

    def test_the_menu_item_starts_one_and_opens_the_conversation(self):
        found = run_both(THREADS + """
            const shell = Object.create(NS.Shell.prototype);
            shell.store = threadStore(MADE);
            shell.state = {conversationExpanded: false};
            shell.nodes = {status: {dataset: {}, textContent: ""}};
            let closed = 0;
            shell.closeMenu = () => { closed += 1; };
            shell.applyAccordion = () => { shell.accordion = shell.state.conversationExpanded; };
            shell._save = () => undefined;
            const item = shell.newThreadItem();
            item.handlers.click.forEach((fn) => fn());
            setImmediate(() => setImmediate(() => console.log(JSON.stringify({
                label: item.textContent, disabled: item.disabled, closed,
                selected: shell.store.selected, opened: shell.accordion,
                said: shell.nodes.status.textContent}))));
        """)

        assert found["label"] == "New thread" and found["disabled"] is False
        assert found["closed"] == 1
        assert found["selected"] == [["Ada", "t9"]]
        assert found["opened"] is True
        assert found["said"] == "New thread with Ada."

    def test_with_no_conversation_the_item_is_there_and_not_pressable(self):
        found = run_both(THREADS + """
            const shell = Object.create(NS.Shell.prototype);
            shell.store = threadStore(MADE);
            shell.store.selection = {character: "", thread: "", epoch: "e1"};
            const item = shell.newThreadItem();
            console.log(JSON.stringify({disabled: item.disabled, title: item.title}));
        """)

        assert found == {"disabled": True, "title": "Choose a conversation first"}


# --------------------------------------------------------------------------- #
# 2. The system prompt editor
# --------------------------------------------------------------------------- #

SYSTEM_JS = SHELL.parent / "forge_assistant_system.js"

EDITOR = """
// Dialogs that open and close the way the browser's do, a store that answers
// the two routes as the scenario says, and a confirm that says what it is told.
const created = document.createElement;
document.createElement = (tag) => {
    const node = created(tag);
    if (node.tagName === "DIALOG") {
        node.open = false;
        node.showModal = () => { node.open = true; node.modal = true; };
        node.close = () => { node.open = false; };
    }
    return node;
};
const BUILT = "You are Ada. Stay in character as Ada.";
const sent = [];
const held = [];
let hold = false;
let refuse = null;
function view(source, text, message, who) {
    return {ok: true, character: who || "Ada", source, text, default: BUILT, message};
}
function reply(value) {
    if (hold) return new Promise((resolve, reject) => held.push({resolve: () => resolve(value), reject}));
    if (refuse) return Promise.reject(refuse);
    return Promise.resolve(value);
}
let stored = "";
const fake = {
    systemPrompt(who) {
        sent.push({read: who});
        const canonical = who.charAt(0).toUpperCase() + who.slice(1);
        return reply(stored ? view("override", stored, undefined, canonical)
                            : view("default", BUILT, undefined, canonical));
    },
    saveSystemPrompt(who, change) {
        sent.push({save: who, change});
        if (change.restore || !change.text.trim() || change.text.trim() === BUILT) {
            stored = "";
            return reply(view("default", BUILT, "Back to the default for Ada."));
        }
        stored = change.text.trim();
        return reply(view("override", stored, "Override saved for Ada."));
    },
};
NS.store = () => fake;
const asked = [];
let agree = true;
globalThis.confirm = (text) => { asked.push(text); return agree; };
const settle = () => new Promise((resolve) => setImmediate(resolve));
const editor = NS.systemEditor.editor();
function type(text) {
    editor.controls.text.value = text;
    editor.controls.text.handlers.input.forEach((fn) => fn());
}
function press(name) {
    editor.controls[name].handlers.click.forEach((fn) => fn());
}
function seen() {
    const c = editor.controls;
    return {open: editor.isOpen(), text: c.text.value, readOnly: c.text.readOnly,
            apply: !c.apply.disabled, restore: !c.restore.disabled,
            state: c.lead.textContent + c.rest.textContent,
            note: c.note.textContent, kind: c.note.dataset.kind,
            heading: c.heading.textContent};
}
"""


def run_editor(scenario, sources=("system",)):
    return run(EDITOR + scenario, sources=sources)


class TestTheStoreSpeaksTheRoute:
    """The editor's two requests, with the page's key like every other."""

    ROUTE = """
    const keyField = {tagName: "TEXTAREA", value: "the-key"};
    const previousLookup = document.getElementById;
    document.getElementById = (id) => id === "mc-llm-chat-conversation-key"
        ? keyField : previousLookup(id);
    const asked = [];
    globalThis.fetch = (url, options) => {
        asked.push({url, method: (options && options.method) || "GET",
                    key: options.headers["x-mc-conversation-key"],
                    type: options.headers["Content-Type"] || "",
                    body: options.body ? JSON.parse(options.body) : null});
        return Promise.resolve({ok: true, json: () => Promise.resolve({ok: true})});
    };
    const store = new NS.Store();
    """

    def test_reading_names_the_character_in_the_address(self):
        found = run(self.ROUTE + """
            store.systemPrompt("Chiharu Yamada").then(() => console.log(JSON.stringify(asked)));
        """, sources=("store",))

        assert found == [{"url": "/model-chain/conversation/v2/system-prompt"
                                 "?character=Chiharu%20Yamada",
                          "method": "GET", "key": "the-key", "type": "", "body": None}]

    def test_applying_and_restoring_post_the_character_and_the_change(self):
        found = run(self.ROUTE + """
            store.saveSystemPrompt("Ada", {text: "Be brief."})
                .then(() => store.saveSystemPrompt("Ada", {restore: true, character: "Bob"}))
                .then(() => console.log(JSON.stringify(asked)));
        """, sources=("store",))

        assert [entry["method"] for entry in found] == ["POST", "POST"]
        assert {entry["url"] for entry in found} == {"/model-chain/conversation/v2/system-prompt"}
        assert all(entry["key"] == "the-key" for entry in found)
        assert all(entry["type"] == "application/json" for entry in found)
        assert found[0]["body"] == {"text": "Be brief.", "character": "Ada"}
        assert found[1]["body"] == {"restore": True, "character": "Ada"}, \
            "the character asked for, whatever the change carries"


class TestTheSystemPromptEditor:
    """Mini Paint's editor, for a character: "a full page editor for system
    prompt, with option to restore default"."""

    def test_it_opens_on_what_the_character_is_told(self):
        found = run_editor("""
            editor.open("Ada");
            settle().then(() => console.log(JSON.stringify(Object.assign(seen(), {sent}))));
        """)

        assert found["open"] is True
        assert found["heading"] == "System prompt — Ada"
        assert found["text"] == "You are Ada. Stay in character as Ada."
        assert found["state"].startswith("Default for Ada, built from Ada’s Context")
        assert found["apply"] is True and found["restore"] is True
        assert found["readOnly"] is False
        assert found["sent"] == [{"read": "Ada"}]

    def test_the_heading_spells_the_name_the_way_the_file_does(self):
        found = run_editor("""
            editor.open("ada");
            settle().then(() => console.log(JSON.stringify(seen())));
        """)

        assert found["heading"] == "System prompt \u2014 Ada"
        assert found["state"].startswith("Default for Ada,")

    def test_an_override_is_shown_as_one(self):
        found = run_editor("""
            stored = "You are {{char}}. Be brief.";
            editor.open("Ada");
            settle().then(() => console.log(JSON.stringify(seen())));
        """)

        assert found["text"] == "You are {{char}}. Be brief."
        assert found["state"] == "Override saved for Ada. Restore default forgets it."

    def test_nothing_can_be_applied_before_the_prompt_is_read(self):
        """An empty box applied is "go back to the default". A press that
        landed before the read answered would wipe an override."""
        found = run_editor("""
            hold = true;
            editor.open("Ada");
            console.log(JSON.stringify(seen()));
        """)

        assert found["apply"] is False and found["restore"] is False
        assert found["readOnly"] is True
        assert found["note"] == "Reading Ada’s system prompt…"

    def test_a_read_that_failed_leaves_nothing_to_press_but_reload(self):
        found = run_editor("""
            refuse = new Error("There is no character called Ada.");
            editor.open("Ada");
            settle().then(() => {
                const c = editor.controls;
                press("apply");
                console.log(JSON.stringify(Object.assign(seen(), {
                    reload: !c.reload.disabled, sent: sent.length})));
            });
        """)

        assert found["apply"] is False and found["restore"] is False
        assert found["reload"] is True
        assert found["note"] == "There is no character called Ada."
        assert found["kind"] == "error"
        assert found["sent"] == 1, "the read, and nothing after it"

    def test_apply_sends_the_box_and_shows_what_was_kept(self):
        found = run_editor("""
            editor.open("Ada");
            settle().then(() => {
                type("You are {{char}}. Answer in one line.  ");
                press("apply");
                return settle();
            }).then(() => console.log(JSON.stringify(Object.assign(seen(), {sent}))));
        """)

        assert found["sent"][1] == {"save": "Ada",
                                    "change": {"text": "You are {{char}}. Answer in one line.  "}}
        assert found["text"] == "You are {{char}}. Answer in one line.", "what the server kept"
        assert found["note"] == "Override saved for Ada."
        assert found["state"].startswith("Override saved for Ada.")

    def test_restore_default_asks_the_server_to_forget_the_override(self):
        found = run_editor("""
            stored = "Be brief.";
            editor.open("Ada");
            settle().then(() => { press("restore"); return settle(); })
                .then(() => console.log(JSON.stringify(Object.assign(seen(), {sent}))));
        """)

        assert found["sent"][1] == {"save": "Ada", "change": {"restore": True}}
        assert found["text"] == "You are Ada. Stay in character as Ada."
        assert found["state"].startswith("Default for Ada")
        assert found["note"] == "Back to the default for Ada."

    def test_typing_says_it_is_not_applied_yet(self):
        found = run_editor("""
            editor.open("Ada");
            settle().then(() => {
                type(BUILT + " More.");
                const typed = [editor.controls.note.textContent, editor.dialog.dataset.dirty];
                type(BUILT);
                const back = [editor.controls.note.textContent, editor.dialog.dataset.dirty];
                console.log(JSON.stringify({typed, back}));
            });
        """)

        assert found["typed"] == ["Not applied yet.", "true"]
        assert found["back"] == ["", "false"]

    def test_close_with_an_unapplied_edit_asks_first(self):
        found = run_editor("""
            editor.open("Ada");
            settle().then(() => {
                type("Something new.");
                agree = false;
                press("close");
                const kept = editor.isOpen();
                agree = true;
                press("close");
                console.log(JSON.stringify({kept, closed: !editor.isOpen(), asked}));
            });
        """)

        assert found["kept"] is True
        assert found["closed"] is True
        assert found["asked"] == ["Discard your changes to Ada’s system prompt?"] * 2

    def test_close_with_nothing_to_lose_does_not_ask(self):
        found = run_editor("""
            editor.open("Ada");
            settle().then(() => {
                press("dismiss");
                console.log(JSON.stringify({closed: !editor.isOpen(), asked}));
            });
        """)

        assert found == {"closed": True, "asked": []}

    def test_escape_is_taken_over_so_it_can_ask(self):
        found = run_editor("""
            editor.open("Ada");
            settle().then(() => {
                type("Something new.");
                agree = false;
                let prevented = false;
                editor.dialog.handlers.cancel.forEach((fn) => fn({preventDefault() { prevented = true; }}));
                console.log(JSON.stringify({prevented, open: editor.isOpen(), asked: asked.length}));
            });
        """)

        assert found == {"prevented": True, "open": True, "asked": 1}

    def test_reload_with_an_unapplied_edit_asks_before_reading_again(self):
        found = run_editor("""
            editor.open("Ada");
            settle().then(() => {
                type("Something new.");
                agree = false;
                press("reload");
                const declined = sent.length;
                agree = true;
                press("reload");
                return settle().then(() => console.log(JSON.stringify(
                    {declined, accepted: sent.length, text: editor.controls.text.value})));
            });
        """)

        assert found["declined"] == 1
        assert found["accepted"] == 2
        assert found["text"] == "You are Ada. Stay in character as Ada."

    def test_a_second_press_on_the_way_in_keeps_the_edit(self):
        found = run_editor("""
            editor.open("Ada");
            settle().then(() => {
                type("Something new.");
                editor.open("Ada");
                return settle();
            }).then(() => console.log(JSON.stringify(
                {text: editor.controls.text.value, reads: sent.length, asked})));
        """)

        assert found == {"text": "Something new.", "reads": 1, "asked": []}

    def test_an_answer_for_a_page_that_has_moved_on_is_not_drawn(self):
        found = run_editor("""
            hold = true;
            editor.open("Ada");
            editor.close(true);
            hold = false;
            stored = "Bob's own.";
            editor.open("Bob");
            settle().then(() => {
                held[0].resolve();
                return settle();
            }).then(() => console.log(JSON.stringify(
                {text: editor.controls.text.value, heading: editor.controls.heading.textContent})));
        """)

        assert found == {"text": "Bob's own.", "heading": "System prompt — Bob"}

    def test_with_no_character_it_says_so_and_asks_nothing(self):
        found = run_editor("""
            const opened = editor.open("");
            console.log(JSON.stringify(Object.assign(seen(), {opened, sent})));
        """)

        assert found["opened"] is False
        assert found["note"] == "Choose a character first."
        assert found["apply"] is False
        assert found["sent"] == []

    def test_it_is_a_modal_dialog_that_focus_mode_leaves_alone(self):
        """Focus mode hides what sits beside the focused workspace unless it
        says it is a dialog, and a native <dialog> only implies it."""
        found = run_editor("""
            editor.open("Ada");
            const dialog = editor.dialog;
            console.log(JSON.stringify({
                modal: !!dialog.modal, role: dialog.getAttribute("role"),
                ariaModal: dialog.getAttribute("aria-modal"),
                inBody: document.body.children.includes(dialog)}));
        """)

        assert found == {"modal": True, "role": "dialog", "ariaModal": "true", "inBody": True}

    def test_it_fills_what_is_on_the_screen(self):
        """A phone whose page is wider than the screen widens the layout
        viewport to the page, and a fixed dialog filled that: off the right
        edge, its buttons below the fold. The visual viewport is the glass."""
        found = run_editor("""
            const moved = {};
            globalThis.visualViewport = {offsetLeft: 96, offsetTop: 0, width: 412, height: 915,
                addEventListener(kind, fn) { moved[kind] = fn; },
                removeEventListener(kind) { delete moved[kind]; }};
            editor.open("Ada");
            const style = editor.dialog.style;
            const opened = [style.left, style.top, style.width, style.height];
            // The keyboard comes up.
            visualViewport.height = 480;
            moved.resize();
            const typing = style.height;
            const following = Object.keys(moved).sort();
            editor.close(true);
            console.log(JSON.stringify({opened, typing, following,
                                        after: Object.keys(moved)}));
        """)

        assert found["opened"] == ["96px", "0px", "412px", "915px"]
        assert found["typing"] == "480px"
        assert found["following"] == ["resize", "scroll"]
        assert found["after"] == [], "nothing is left listening once it is closed"

    def test_a_name_is_put_on_the_page_as_text(self):
        found = run_editor("""
            editor.open("<img src=x onerror=alert(1)>");
            console.log(JSON.stringify({heading: editor.controls.heading.textContent,
                                        markup: editor.controls.heading.innerHTML}));
        """)

        assert found["heading"] == "System prompt — <img src=x onerror=alert(1)>"
        assert found["markup"] == ""


class TestTheEditorAndLLMStudioAgree:
    """LLM Studio's character editor holds the same field in its override
    box. Open on the same character, it has to hear about a save, or its next
    Save writes the old prompt back."""

    STUDIO = """
    const name = document.createElement("textarea");
    const box = document.createElement("textarea");
    const previousLookup = document.getElementById;
    document.getElementById = (id) => ({"mc-llm-chat-name": name,
                                        "mc-llm-chat-system": box})[id] || previousLookup(id);
    const told = [];
    globalThis.updateInput = (target) => told.push(target === box ? "system" : "other");
    """

    def test_an_override_saved_here_is_written_into_its_box(self):
        found = run_editor(self.STUDIO + """
            name.value = "Ada";
            editor.open("Ada");
            settle().then(() => { type("Be brief."); press("apply"); return settle(); })
                .then(() => console.log(JSON.stringify({box: box.value, told})));
        """)

        assert found == {"box": "Be brief.", "told": ["system"]}

    def test_restoring_the_default_empties_its_box(self):
        found = run_editor(self.STUDIO + """
            name.value = "Ada";
            box.value = "Be brief.";
            stored = "Be brief.";
            editor.open("Ada");
            settle().then(() => { press("restore"); return settle(); })
                .then(() => console.log(JSON.stringify({box: box.value, told})));
        """)

        assert found == {"box": "", "told": ["system"]}

    def test_a_studio_editor_on_another_character_is_left_alone(self):
        found = run_editor(self.STUDIO + """
            name.value = "Bob";
            box.value = "Bob's own.";
            editor.open("Ada");
            settle().then(() => { type("Be brief."); press("apply"); return settle(); })
                .then(() => console.log(JSON.stringify({box: box.value, told})));
        """)

        assert found == {"box": "Bob's own.", "told": []}

    def test_the_ids_are_the_ones_the_tab_gives_its_boxes(self):
        import mc_llm_ui as ui

        source = SYSTEM_JS.read_text(encoding="utf-8")

        assert f'const STUDIO_NAME = "{ui.ident("chat", "name")}";' in source
        assert f'const STUDIO_SYSTEM = "{ui.ident("chat", "system")}";' in source


class TestTheMenuOpensTheEditor:
    """"For flyout, put this option to open this view in the '...' menu.\""""

    MENU = """
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {freeFloat: false};
    shell.host = {listUtilities: () => []};
    shell.store = {snapshot: () => ({selection: {character: "Ada", thread: "t1"}})};
    """

    def test_it_is_in_the_menu_after_new_thread(self):
        found = run(self.MENU + """
            console.log(JSON.stringify(shell.utilityItems().map((item) => item.textContent)));
        """, sources=("shell", "system"))

        assert found[:5] == ["Free Float", "Auto Attach", "Send to Generate", "New thread",
                             "System prompt…"]

    def test_pressing_it_puts_the_menu_away_and_opens_the_character(self):
        found = run(self.MENU + """
            const order = [];
            shell.closeMenu = () => order.push("menu");
            NS.systemEditor.open = (who) => order.push("editor:" + who);
            const item = shell.utilityItems().find((i) => i.textContent === "System prompt\\u2026");
            item.handlers.click.forEach((fn) => fn());
            console.log(JSON.stringify({order, popup: item["aria-haspopup"]}));
        """, sources=("shell", "system"))

        assert found == {"order": ["menu", "editor:Ada"], "popup": "dialog"}

    def test_with_no_conversation_it_is_there_and_not_pressable(self):
        found = run(self.MENU + """
            shell.store = {snapshot: () => ({selection: {character: "", thread: ""}})};
            const item = shell.systemPromptItem();
            console.log(JSON.stringify({disabled: item.disabled, title: item.title}));
        """, sources=("shell", "system"))

        assert found == {"disabled": True, "title": "Choose a conversation first"}

    def test_without_the_editor_the_menu_does_not_offer_it(self):
        found = run(self.MENU + """
            console.log(JSON.stringify(shell.utilityItems().map((item) => item.textContent)));
        """, sources=("shell",))

        assert "System prompt…" not in found

    def test_escape_in_the_editor_does_not_leave_focus_mode(self):
        """The editor's Escape asks before an unapplied edit goes. Taken by
        the panel, it would have left focus mode behind the dialog instead."""
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            let toggled = 0;
            shell.toggleFocus = () => { toggled += 1; };
            shell.nodes = {root: {contains: () => true}, menu: {hidden: true}};
            shell.focus = {isActive: () => true};
            NS.hostInDialog = () => false;
            NS.escapeOrder = () => "exit-focus";
            NS.systemEditor.isOpen = () => true;
            let stopped = false;
            shell.documentKey({key: "Escape", preventDefault() {}, stopPropagation() { stopped = true; }});
            console.log(JSON.stringify({toggled, stopped}));
        """, sources=("shell", "system"))

        assert found == {"toggled": 0, "stopped": False}


class TestTheEditorIsTheWholeWindow:
    def test_nothing_sets_the_dialog_s_own_display(self):
        """An author `display` beats the browser's `dialog:not([open])`, and a
        closed editor would sit on the page. The column is the sheet's."""
        assert "display" not in rule(".forge-assistant-system")
        assert "display: flex" in rule(".forge-assistant-system-sheet")

    def test_the_box_takes_what_the_rest_leaves(self):
        text = rule(".forge-assistant-system-text")

        assert "flex: 1 1 auto" in text
        assert "resize: none" in text
        assert "max(16px" in text, "or a phone zooms the page when the box is tapped"

    def test_it_is_sized_by_its_edges_and_not_by_the_viewport_width(self):
        """100vw reaches under the page's scrollbar, where the way out is."""
        dialog = rule(".forge-assistant-system")

        assert "inset: 0" in dialog
        assert "100vw" not in dialog
        assert "max-width: none" in dialog and "max-height: none" in dialog
