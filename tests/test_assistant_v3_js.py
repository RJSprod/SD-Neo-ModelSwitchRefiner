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
