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
