"""LLM Studio's Conversation pop-up on a real Gradio 4.40 page, driven by a real browser.

"make these all just one "Conversation" button ... make it easy to favorite an
existing thread i have with one of my characters, so i can quickly load and
continue" (2026-11-06). The node harness and the stand-in Gradio in
``tests/conftest.py`` hold the pieces; only a real Gradio says whether its event
graph does what the pieces assume: that a folded section's label is redrawn by
``gr.update(label=...)``, that an HTML list redrawn by one handler is pressed by
the script and answered by another, and above all that a favourite with another
character -- remembered, then that character chosen in the drop-down -- lands on
the thread starred and not on that character's newest, which rests on the
drop-down's own ``change`` running after the press that set it.

This builds the panel itself (``mc_llm_chat_panel.build()``) inside
``#mc-llm-studio`` with the real ``javascript/llm_studio.js`` and ``style.css``,
on the host the tests stand in (``tests/conftest.py``'s modules), over a data
folder of its own with two characters and their threads, and drives it in
Chromium.

It needs Gradio 4.40.0 (with the pins Mini Paint NEO's ``tests/requirements.txt``
names), Playwright and pytest (``tests/conftest.py`` imports it), which the
suite's own interpreter does not have, so ``tests/test_llm_studio_live.py`` runs
it in the interpreter ``MC_GRADIO_PYTHON`` names and skips without one. Run by
hand::

    <venv>/bin/python tests/live/llm_studio_conversation_live.py

It prints one JSON object and exits 0 only when every check held.
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Forge's script.js, as far as this extension leans on it (as in
# llm_studio_row_live.py): gradioApp(), and the two callback lists.
HEAD = """
<script>
window.gradioApp = function () {
    const app = document.getElementsByTagName("gradio-app")[0];
    return app && app.shadowRoot ? app.shadowRoot : document;
};
const __uiLoaded = [], __afterUpdate = [];
let __loaded = false, __queued = false;
window.onUiLoaded = (fn) => __uiLoaded.push(fn);
window.onAfterUiUpdate = (fn) => __afterUpdate.push(fn);
window.onUiUpdate = () => {};
new MutationObserver(() => {
    if (__queued) return;
    __queued = true;
    requestAnimationFrame(() => {
        __queued = false;
        if (!__loaded && document.querySelector("#mc-llm-chat-transcript")) {
            __loaded = true;
            __uiLoaded.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
        }
        if (__loaded) __afterUpdate.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
    });
}).observe(document.documentElement, {childList: true, subtree: true, attributes: true, characterData: true});
</script>
<script>
%(script)s
</script>
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def host(data: Path):
    """The host the tests stand in, with the real Gradio, over ``data``."""
    import inspect

    import gradio  # first, so conftest's stand-in is not installed

    # Stock 4.40.0's Chatbot takes no ``autoscroll``, which the panel passes;
    # the user's Forge Neo builds it on a 4.40-lineage Gradio of its own that
    # does (CLAUDE.md, "Gradio's chat bubble is a button").
    if "autoscroll" not in inspect.signature(gradio.Chatbot.__init__).parameters:
        chatbot = gradio.Chatbot

        def without_autoscroll(*args, autoscroll=None, **kwargs):
            return chatbot(*args, **kwargs)

        gradio.Chatbot = without_autoscroll

    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT))
    import conftest  # noqa: F401  -- installs the host's modules

    import mc_llm_paths

    mc_llm_paths.data_root = lambda: data


def threads(data: Path) -> dict:
    """Ada with three threads, Bob with two -- Bob's oldest starred."""
    import mc_llm_chat_panel
    import mc_llm_state
    from prompt_master.chat.characters import Character
    from prompt_master.chat.history import ASSISTANT, USER, ChatStore

    held = mc_llm_chat_panel._characters()
    for name in ("Ada", "Bob"):
        held.save(Character(name=name, context=f"{name}, who answers briefly."))
    chats = ChatStore.from_paths(mc_llm_paths_app())
    made = {}
    for who, titles in (("Ada", ("harbour at night", "a desert road", "the last train")),
                        ("Bob", ("bob's first walk", "bob's market day"))):
        for title in titles:
            conversation = chats.new(who)
            conversation.title = title
            conversation.append(USER, f"Tell me about {title}.")
            conversation.append(ASSISTANT, f"{title}, briefly.")
            chats.save(conversation)
            made[title] = conversation.identifier
    chats.set_pinned("Bob", made["bob's first walk"], True)
    mc_llm_state.remember(character="Ada", thread=made["the last train"])
    return made


def mc_llm_paths_app():
    import mc_llm_paths

    return mc_llm_paths.app_paths()


def build():
    import gradio as gr

    import mc_llm_chat_panel

    # No faces beside the messages: they are files Forge serves through routes
    # of its own, which stock Gradio's Image cannot resolve (its file-url throws
    # "Invalid URL" for each) -- an error of the stand-in, not of the pop-up.
    mc_llm_chat_panel._faces = lambda who: [None, None]
    script = (ROOT / "javascript" / "llm_studio.js").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    with gr.Blocks(head=HEAD % {"script": script}, css=css, analytics_enabled=False) as demo:
        with gr.Column(elem_id="mc-llm-studio", elem_classes=["mc-llm-studio"]):
            mc_llm_chat_panel.build()
    return demo


def launch_browser(playwright):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llm_studio_row_live import launch_browser as launch

    return launch(playwright)


HEADING = "() => document.querySelector('#mc-llm-chat-title .mc-llm-heading').textContent"
OPEN = "() => getComputedStyle(document.querySelector('#mc-llm-chat-conversation')).display !== 'none'"
LISTED = """(id) => [...document.querySelectorAll('#' + id + ' .mc-llm-thread-open')].map((b) => ({
    who: b.getAttribute('data-character'), title: b.querySelector('.mc-llm-thread-title').textContent,
    current: b.getAttribute('aria-current') === 'true',
    starred: b.parentElement.getAttribute('data-pinned') === 'true'}))"""
LABEL = """(id) => { const box = document.querySelector('#' + id);
    const label = box && box.querySelector('.label-wrap span, button span');
    return label ? label.textContent.trim() : null; }"""
# Whether each list is drawn but folded away: in the page, with no box on screen.
FOLDED = """(ids) => ids.map((id) => { const list = document.querySelector('#' + id);
    return !!list && list.getClientRects().length === 0; })"""
DRY = "#mc-llm-chat-generation-dry-multiplier input[type=number]"
VALUE = "(selector) => { const box = document.querySelector(selector); return box ? box.value : null; }"
# The pop-up's own scroll, and where something in it is: the pop-up put so
# that `selector` stands at `share` of its height, and read back as the
# pop-up's scroll and the thing's distance from the pop-up's top edge.
SHEET_TOP = "() => document.querySelector('#mc-llm-chat-conversation').scrollTop"
PUT = """([selector, share]) => { const sheet = document.querySelector('#mc-llm-chat-conversation');
    const it = document.querySelector(selector).getBoundingClientRect();
    sheet.scrollTop = Math.max(0, sheet.scrollTop + it.top - sheet.getBoundingClientRect().top
                               - sheet.clientHeight * share); }"""
WHERE = """(selector) => { const sheet = document.querySelector('#mc-llm-chat-conversation');
    return {top: sheet.scrollTop, at: Math.round(document.querySelector(selector).getBoundingClientRect().top
                                                 - sheet.getBoundingClientRect().top)}; }"""
# Marks the `n`th match of `selector` (from the end when negative) so it can
# be named; a row a redraw replaces is named by its section instead.
MARK = """([selector, n]) => { const all = document.querySelectorAll(selector);
    all[n < 0 ? all.length + n : n].setAttribute('data-probe', 'pressed'); }"""


def _same(record, scrolled: bool = True) -> bool:
    """Whether the pop-up and the thing in it were where they had been put --
    put, with ``scrolled``, somewhere the pop-up had been scrolled to."""
    return bool(record) and record.get("before") == record.get("after") \
        and (not scrolled or record.get("before", {}).get("top", 0) > 0)


def main() -> int:
    from playwright.sync_api import sync_playwright

    data = Path(tempfile.mkdtemp(prefix="mc-conversation-live-"))
    host(data)
    made = threads(data)
    demo = build()
    port = free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                quiet=True, show_api=False, share=False)
    url = f"http://127.0.0.1:{port}/"
    found: dict = {}
    errors: list = []
    try:
        with sync_playwright() as playwright:
            browser = launch_browser(playwright)
            page = browser.new_context(viewport={"width": 1200, "height": 900}).new_page()
            page.on("pageerror", lambda error: errors.append(
                f"pageerror: {error} | {(getattr(error, 'stack', '') or '').splitlines()[:4]}"))
            page.on("console", lambda message: errors.append(f"console.{message.type}: {message.text}")
                    if message.type == "error" and "Failed to load resource" not in message.text else None)
            page.goto(url)
            page.wait_for_selector("#mc-llm-chat-to-conversation", timeout=30000)

            found["chips"] = page.evaluate("""() => [...document.querySelectorAll(
                '.mc-llm-chat-header button')].map((b) => b.textContent.trim()).filter(Boolean)""")
            found["heading_before"] = page.evaluate(HEADING)

            # Open the pop-up: Favourites (Bob's starred), Ada's threads, the sections.
            page.click("#mc-llm-chat-to-conversation")
            page.wait_for_function(OPEN, timeout=10000)
            page.wait_for_function("document.querySelectorAll('#mc-llm-chat-threads .mc-llm-thread-open').length === 3",
                                   timeout=10000)
            found["opened"] = {
                "folded": page.evaluate(FOLDED, ["mc-llm-chat-favourites", "mc-llm-chat-threads"]),
                "favourites_label": page.evaluate(LABEL, "mc-llm-chat-favourites-section"),
                "threads_label": page.evaluate(LABEL, "mc-llm-chat-threads-section"),
                "new_thread_on_top": page.evaluate("""() => { const button = document.querySelector(
                    '#mc-llm-chat-new-thread');
                    return !!button && !!button.closest('.mc-llm-conversation-head')
                        && button.getClientRects().length > 0; }"""),
                "generation_controls": page.evaluate("""() => document.querySelectorAll(
                    '#mc-llm-chat-generation input[type=range]').length"""),
                "favourites": page.evaluate(LISTED, "mc-llm-chat-favourites"),
                "threads": page.evaluate(LISTED, "mc-llm-chat-threads"),
                "emotion_label": page.evaluate(LABEL, "mc-llm-chat-emotion"),
                "start_label": page.evaluate(LABEL, "mc-llm-chat-start"),
                "sections": page.evaluate("""() => ['emotion', 'start', 'generation', 'voice', 'character',
                    'thread', 'persona'].map((id) => !!document.querySelector('#mc-llm-chat-' + id))"""),
            }

            # Star Ada's "harbour at night" -- its list opened first: the star
            # fills, it goes first, and Favourites gains it beside Bob's.
            page.click("#mc-llm-chat-threads-section .label-wrap")
            page.click("#mc-llm-chat-threads .mc-llm-thread-row:has-text('harbour at night') .mc-llm-thread-star")
            page.wait_for_function("""() => [...document.querySelectorAll('#mc-llm-chat-favourites .mc-llm-thread-open')]
                .length === 2""", timeout=10000)
            page.wait_for_function("""() => { const l = document.querySelector(
                '#mc-llm-chat-favourites-section .label-wrap span');
                return l && l.textContent.trim() === 'Favourites · 2'; }""", timeout=10000)
            found["starred"] = {
                "favourites_label": page.evaluate(LABEL, "mc-llm-chat-favourites-section"),
                "favourites": page.evaluate(LISTED, "mc-llm-chat-favourites"),
                "threads": page.evaluate(LISTED, "mc-llm-chat-threads"),
                "still_open": page.evaluate(OPEN),
            }

            # The emotion: a row chosen; the section's label and the chip's dot follow.
            page.click("#mc-llm-chat-emotion .label-wrap")
            page.wait_for_selector("#mc-llm-chat-emotion .mc-llm-emotion-choice >> nth=1", state="visible",
                                   timeout=10000)
            page.click("#mc-llm-chat-emotion .mc-llm-emotion-choice >> nth=1")
            page.wait_for_function("""() => { const l = document.querySelector('#mc-llm-chat-emotion .label-wrap span');
                return l && l.textContent.trim() !== 'Emotion · Off'; }""", timeout=10000)
            found["emotion"] = {
                "label": page.evaluate(LABEL, "mc-llm-chat-emotion"),
                "dot": page.evaluate("""() => getComputedStyle(document.querySelector(
                    '#mc-llm-chat-to-conversation'), '::before').content"""),
                "heading_on": page.evaluate("""() => document.querySelector(
                    '#mc-llm-chat-title .mc-llm-heading').getAttribute('data-emotion')"""),
            }

            # A generation setting, moved in the thread open ("the last train"):
            # kept with that thread on the server, the section relabelled.
            page.click("#mc-llm-chat-generation .label-wrap")
            page.wait_for_selector(DRY, state="visible", timeout=10000)
            page.fill(DRY, "0.8")
            page.locator(DRY).blur()
            page.wait_for_function("""() => { const l = document.querySelector('#mc-llm-chat-generation .label-wrap span');
                return l && l.textContent.trim() === 'Generation · DRY 0.8'; }""", timeout=10000)
            page.wait_for_timeout(400)      # the release after the last input
            import mc_llm_generation

            found["generation"] = {
                "label": page.evaluate(LABEL, "mc-llm-chat-generation"),
                "kept": mc_llm_generation.for_thread("Ada", made["the last train"]),
            }

            # Back, and the pop-up again: the lists folded again though Threads
            # was left open, and the setting drawn from what is kept -- changed
            # meanwhile behind the page, as another window would change it.
            page.click("#mc-llm-chat-conversation .mc-llm-sheet-back")
            page.wait_for_function("() => !(" + OPEN + ")()", timeout=10000)
            mc_llm_generation.choose("Ada", made["the last train"], "dry_multiplier", 1.25)
            page.click("#mc-llm-chat-to-conversation")
            page.wait_for_function(OPEN, timeout=10000)
            page.wait_for_function("(selector) => document.querySelector(selector).value === '1.25'",
                                   arg=DRY, timeout=10000)
            found["reopened"] = {
                "folded": page.evaluate(FOLDED, ["mc-llm-chat-favourites", "mc-llm-chat-threads"]),
                "dry": page.evaluate(VALUE, DRY),
                "label": page.evaluate(LABEL, "mc-llm-chat-generation"),
            }

            # Another of Ada's threads: its own settings, which are none.
            page.click("#mc-llm-chat-threads-section .label-wrap")
            page.click("#mc-llm-chat-threads .mc-llm-thread-row:has-text('a desert road') .mc-llm-thread-open")
            page.wait_for_function("() => document.querySelector('#mc-llm-chat-title .mc-llm-heading')"
                                   ".textContent.includes('a desert road')", timeout=15000)
            page.click("#mc-llm-chat-to-conversation")
            page.wait_for_function(OPEN, timeout=10000)
            page.wait_for_function("""() => { const l = document.querySelector('#mc-llm-chat-generation .label-wrap span');
                return l && l.textContent.trim() === 'Generation'; }""", timeout=10000)
            found["other_thread"] = {"dry": page.evaluate(VALUE, DRY),
                                     "label": page.evaluate(LABEL, "mc-llm-chat-generation")}

            # "It should be default until I override for a thread": + New thread
            # pressed in the tuned thread starts at the defaults ...
            page.click("#mc-llm-chat-threads-section .label-wrap")
            page.click("#mc-llm-chat-threads .mc-llm-thread-row:has-text('the last train') .mc-llm-thread-open")
            page.wait_for_function("() => document.querySelector('#mc-llm-chat-title .mc-llm-heading')"
                                   ".textContent.includes('the last train')", timeout=15000)
            page.click("#mc-llm-chat-to-conversation")
            page.wait_for_function(OPEN, timeout=10000)
            page.click("#mc-llm-chat-new-thread")
            page.wait_for_function("() => !(" + OPEN + ")()", timeout=10000)
            page.click("#mc-llm-chat-to-conversation")
            page.wait_for_function(OPEN, timeout=10000)
            page.wait_for_function("""() => { const l = document.querySelector('#mc-llm-chat-generation .label-wrap span');
                return l && l.textContent.trim() === 'Generation'; }""", timeout=10000)
            found["new_thread"] = {"dry": page.evaluate(VALUE, DRY),
                                   "label": page.evaluate(LABEL, "mc-llm-chat-generation"),
                                   "heading": page.evaluate(HEADING)}

            # ... "and then kept for that thread through sessions": the page
            # loaded again, the tuned thread opened, its setting still there.
            page.reload()
            page.wait_for_selector("#mc-llm-chat-to-conversation", timeout=30000)
            page.click("#mc-llm-chat-to-conversation")
            page.wait_for_function(OPEN, timeout=10000)
            page.click("#mc-llm-chat-threads-section .label-wrap")
            page.click("#mc-llm-chat-threads .mc-llm-thread-row:has-text('the last train') .mc-llm-thread-open")
            page.wait_for_function("() => document.querySelector('#mc-llm-chat-title .mc-llm-heading')"
                                   ".textContent.includes('the last train')", timeout=15000)
            page.click("#mc-llm-chat-to-conversation")
            page.wait_for_function(OPEN, timeout=10000)
            page.wait_for_function("(selector) => document.querySelector(selector).value === '1.25'",
                                   arg=DRY, timeout=10000)
            found["next_session"] = {"dry": page.evaluate(VALUE, DRY),
                                     "label": page.evaluate(LABEL, "mc-llm-chat-generation")}

            # + New opens the editor under "Talking to".
            page.click("#mc-llm-chat-new-character")
            page.wait_for_selector("#mc-llm-chat-character-editor", state="visible", timeout=10000)
            found["editor"] = page.evaluate("""() => { const sheet = document.querySelector('#mc-llm-chat-conversation');
                const editor = document.querySelector('#mc-llm-chat-character-editor').getBoundingClientRect();
                const box = sheet.getBoundingClientRect();
                return {shown: editor.height > 0, inView: editor.top >= box.top - 1 && editor.top < box.bottom}; }""")
            page.click("#mc-llm-chat-character-editor button:has-text('Cancel')")

            # Bob's favourite, from Ada -- Favourites opened first: the pop-up
            # goes, Bob is chosen, and the thread open is the starred one -- not
            # Bob's newest.
            page.click("#mc-llm-chat-favourites-section .label-wrap")
            page.click("#mc-llm-chat-favourites .mc-llm-thread-row:has-text(\"bob's first walk\") .mc-llm-thread-open")
            page.wait_for_function("() => document.querySelector('#mc-llm-chat-title .mc-llm-heading')"
                                   ".textContent.includes(\"bob's first walk\")", timeout=15000)
            page.wait_for_timeout(600)      # anything chained after the change has run
            found["favourite"] = {
                "heading": page.evaluate(HEADING),
                "open": page.evaluate(OPEN),
                "transcript": page.evaluate("""() => document.querySelector('#mc-llm-chat-transcript')
                    .textContent.includes("Tell me about bob's first walk.")"""),
            }

            # The pop-up holds where it is put while an option in it is used
            # (2026-11-09: "interacting with the options in the conversation
            # menu often makes the menu scroll without me taking action"). The
            # first build moved it after every press in it: the DRY box and the
            # last slider, near the end of Generation -- taller than the pop-up
            # -- sent it from 1189 and 1639 back to Generation's heading at 353.
            # A section's heading still brings what it opened into view.
            page.click("#mc-llm-chat-to-conversation")
            page.wait_for_function(OPEN, timeout=10000)
            page.wait_for_timeout(300)
            heading_before = page.evaluate(SHEET_TOP)
            page.click("#mc-llm-chat-generation .label-wrap")
            page.wait_for_selector(DRY, state="visible", timeout=10000)
            page.wait_for_timeout(400)
            held = {"heading": {"before": heading_before, "after": page.evaluate(SHEET_TOP)}}

            def used(name, selector, act, measured=None):
                page.evaluate(PUT, [selector, 0.5])
                before = page.evaluate(WHERE, measured or selector)
                act()
                page.wait_for_timeout(900)      # the server's answer, drawn
                held[name] = {"before": before, "after": page.evaluate(WHERE, measured or selector)}

            def dry():
                page.click(DRY)
                page.keyboard.press("Control+A")
                page.keyboard.type("0.6")
                page.evaluate("() => document.activeElement && document.activeElement.blur()")

            used("dry box", DRY, dry)
            held["dry label"] = page.evaluate(LABEL, "mc-llm-chat-generation")
            page.evaluate(MARK, ["#mc-llm-chat-generation input[type=range]", -1])
            slider = "input[type=range][data-probe=pressed]"
            used("slider pressed", slider, lambda: page.click(slider))
            used("slider moved by a key", slider, lambda: page.keyboard.press("ArrowRight"))
            # An emotion row near the pop-up's foot, its section running on
            # below the fold -- a window short enough for that, as a phone's is
            # -- and its list drawn again by the server's answer. (The first
            # build happened to leave this one alone: the row it would have
            # scrolled to was gone, redrawn, by the time it looked.)
            page.set_viewport_size({"width": 1200, "height": 520})
            page.wait_for_timeout(300)
            page.click("#mc-llm-chat-emotion .label-wrap")
            page.wait_for_selector("#mc-llm-chat-emotion .mc-llm-emotion-choice >> nth=2",
                                   state="visible", timeout=10000)
            page.wait_for_timeout(400)
            page.evaluate(MARK, ["#mc-llm-chat-emotion .mc-llm-emotion-choice", 2])
            choice = ".mc-llm-emotion-choice[data-probe=pressed]"
            page.evaluate(PUT, [choice, 0.85])
            before = page.evaluate(WHERE, "#mc-llm-chat-emotion")
            below = page.evaluate("""() => { const sheet = document.querySelector('#mc-llm-chat-conversation');
                return document.querySelector('#mc-llm-chat-emotion').getBoundingClientRect().bottom
                    > sheet.getBoundingClientRect().bottom; }""")
            page.click(choice)
            page.wait_for_timeout(900)
            held["emotion row"] = {"before": before, "below the fold": below,
                                   "after": page.evaluate(WHERE, "#mc-llm-chat-emotion")}
            held["emotion label"] = page.evaluate(LABEL, "mc-llm-chat-emotion")
            found["held"] = held
            browser.close()
    finally:
        demo.close()

    opened = found.get("opened", {})
    starred = found.get("starred", {})
    checks = {
        "one Conversation chip, none of the six": (
            "Conversation" in found.get("chips", [])
            and not ({"Threads", "Character", "Emotion", "Start with", "You", "Voice"}
                     & set(found.get("chips", [])))),
        "Favourites lists the starred thread, whose it is": opened.get("favourites") == [
            {"who": "Bob", "title": "bob's first walk", "current": False, "starred": True}],
        "both lists open folded, Favourites counted, + New thread on top": (
            opened.get("folded") == [True, True]
            and opened.get("favourites_label") == "Favourites · 1"
            and opened.get("threads_label") == "Threads"
            and opened.get("new_thread_on_top") is True),
        "the Generation section has every slider": opened.get("generation_controls") == 16,
        "a generation setting is kept with the thread and relabels the section": (
            found.get("generation", {}).get("label") == "Generation · DRY 0.8"
            and found.get("generation", {}).get("kept") == {"dry_multiplier": 0.8}),
        "the pop-up again: lists folded again, the setting drawn from what is kept": (
            found.get("reopened", {}).get("folded") == [True, True]
            and float(found.get("reopened", {}).get("dry") or -1) == 1.25
            and found.get("reopened", {}).get("label") == "Generation · DRY 1.25"),
        "another thread has its own settings": (
            float(found.get("other_thread", {}).get("dry") or -1) == 0.0
            and found.get("other_thread", {}).get("label") == "Generation"),
        "+ New thread from a tuned thread starts at the defaults": (
            float(found.get("new_thread", {}).get("dry") or -1) == 0.0
            and found.get("new_thread", {}).get("label") == "Generation"
            and "the last train" not in (found.get("new_thread", {}).get("heading") or "")),
        "the tuned thread keeps its setting in the next session": (
            float(found.get("next_session", {}).get("dry") or -1) == 1.25
            and found.get("next_session", {}).get("label") == "Generation · DRY 1.25"),
        "Threads lists Ada's, the remembered one marked": (
            [row["title"] for row in opened.get("threads", [])]
            == ["the last train", "a desert road", "harbour at night"]
            and [row["current"] for row in opened.get("threads", [])] == [True, False, False]),
        "the sections are there, folded with what is in force": (
            all(opened.get("sections", []))
            and opened.get("emotion_label") == "Emotion · Off"
            and opened.get("start_label") == "Start the next reply with"),
        "a star puts the thread first and in Favourites, the pop-up open": (
            starred.get("still_open") is True
            and starred.get("favourites_label") == "Favourites · 2"
            and [row["title"] for row in starred.get("threads", [])][0] == "harbour at night"
            and starred.get("threads", [{}])[0].get("starred") is True
            and {row["title"] for row in starred.get("favourites", [])}
            == {"harbour at night", "bob's first walk"}),
        "an emotion chosen relabels its section and dots the chip": (
            (found.get("emotion", {}).get("label") or "").startswith("Emotion · *")
            and found.get("emotion", {}).get("heading_on") == "on"
            and found.get("emotion", {}).get("dot") not in ("none", "normal", None)),
        "+ New opens the editor in view": found.get("editor") == {"shown": True, "inView": True},
        "a favourite with another character opens that thread": (
            "Bob" in (found.get("favourite", {}).get("heading") or "")
            and "bob's first walk" in (found.get("favourite", {}).get("heading") or "")
            and found.get("favourite", {}).get("open") is False
            and found.get("favourite", {}).get("transcript") is True),
        "a section's heading brings what it opened into view": (
            found.get("held", {}).get("heading", {}).get("after", 0)
            > found.get("held", {}).get("heading", {}).get("before", 0)),
        "the pop-up holds while the DRY box is pressed and typed into": (
            _same(found.get("held", {}).get("dry box"))
            and found.get("held", {}).get("dry label") == "Generation · DRY 0.6"),
        "the pop-up holds while a slider is pressed and moved by a key": (
            _same(found.get("held", {}).get("slider pressed"))
            and _same(found.get("held", {}).get("slider moved by a key"))),
        "the pop-up holds while an emotion is chosen and its list drawn again": (
            found.get("held", {}).get("emotion row", {}).get("below the fold") is True
            and _same(found.get("held", {}).get("emotion row"), scrolled=False)
            and (found.get("held", {}).get("emotion label") or "").startswith("Emotion · *")),
        "no script errors": not errors,
    }
    print(json.dumps({"checks": checks, "found": found, "errors": errors, "made": made}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
