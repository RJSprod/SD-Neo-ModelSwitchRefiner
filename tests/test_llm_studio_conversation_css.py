"""The Conversation pop-up's look, in Chromium under the real stylesheet.

"make these all just one "Conversation" button ... I need a star or a pin or
something to keep it at top" (2026-11-06). What only a browser can say about
``style.css``: the one button carries a dot while the thread has an emotion or a
start set -- the two chips' dots, folded into one -- and none otherwise; a
starred thread's star is the accent and filled, an unstarred one hollow; the
thread open is marked; and Favourites' rows say whose thread each is. The rows
are the ones ``mc_llm_chat_panel`` draws, so the stylesheet is held to the
markup it is written for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mc_llm_chat_panel

sync_api = pytest.importorskip("playwright.sync_api")

from test_llm_studio_sheet_taps_js import _launch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ACCENT = "rgb(255, 120, 0)"


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        found = _launch(playwright)
        if found is None:
            pytest.skip("no Chromium to render with")
        yield found
        found.close()


class Info:
    def __init__(self, identifier, title, pinned):
        self.identifier, self.title, self.pinned, self.updated = identifier, title, pinned, 0.0


def page(browser, emotion="off", start="off"):
    rows = "".join(mc_llm_chat_panel._thread_row("Ada", info, info.identifier == "t2")
                   for info in (Info("t1", "harbour at night", True),
                                Info("t2", "a desert road", False)))
    favourite = mc_llm_chat_panel._thread_row("Bob", Info("b1", "bob's walk", True), False,
                                              with_who=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>:root {{ --color-accent: {ACCENT}; --body-text-color-subdued: rgb(120, 120, 120); }}
body {{ margin: 0; font: 14px sans-serif; }}</style>
<style>{(ROOT / "style.css").read_text(encoding="utf-8")}</style></head><body>
<div id="mc-llm-studio"><div class="mc-llm-chat-header">
  <div class="mc-llm-heading" data-emotion="{emotion}" data-start="{start}">Ada</div>
  <button id="chip" class="mc-llm-chip-button mc-llm-conversation-chip">Conversation</button>
</div>
<div class="mc-llm-conversation-sheet">
  <div id="threads" class="mc-llm-thread-list">{rows}</div>
  <div id="favourites" class="mc-llm-thread-list">{favourite}</div>
</div></div></body></html>"""
    tab = browser.new_page(viewport={"width": 800, "height": 600})
    tab.set_content(html)
    return tab


DOT = "() => getComputedStyle(document.getElementById('chip'), '::before').content"


@pytest.mark.parametrize("emotion,start", [("on", "off"), ("off", "on"), ("on", "on")])
def test_the_button_carries_a_dot_while_an_emotion_or_a_start_is_set(browser, emotion, start):
    tab = page(browser, emotion, start)
    try:
        found = tab.evaluate("""() => { const dot = getComputedStyle(document.getElementById('chip'), '::before');
            return {content: dot.content, colour: dot.backgroundColor, width: dot.width}; }""")
    finally:
        tab.close()

    assert found["content"] == '""'
    assert found["colour"] == ACCENT and found["width"] != "auto"


def test_with_neither_set_the_button_has_no_dot(browser):
    tab = page(browser)
    try:
        assert tab.evaluate(DOT) == "none"
    finally:
        tab.close()


def test_a_starred_threads_star_is_the_accent_and_filled_and_the_open_one_is_marked(browser):
    tab = page(browser)
    try:
        found = tab.evaluate("""() => [...document.querySelectorAll('#threads .mc-llm-thread-row')].map((row) => {
            const star = row.querySelector('.mc-llm-thread-star');
            return {colour: getComputedStyle(star).color,
                    fill: star.querySelector('svg').getAttribute('fill'),
                    weight: getComputedStyle(row.querySelector('.mc-llm-thread-title')).fontWeight,
                    edge: getComputedStyle(row).borderColor,
                    wide: star.getBoundingClientRect().width >= 36,
                    title: row.querySelector('.mc-llm-thread-title').textContent}; })""")
    finally:
        tab.close()

    starred, plain = found
    assert (starred["title"], starred["colour"], starred["fill"]) == \
        ("harbour at night", ACCENT, "currentColor")
    assert plain["colour"] != ACCENT and plain["fill"] == "none"
    assert plain["weight"] == "600" and starred["weight"] != "600", "the thread open is marked"
    assert starred["wide"] and plain["wide"], "a star a finger can land on"


def test_a_favourite_says_whose_thread_it_is_beside_its_title(browser):
    tab = page(browser)
    try:
        found = tab.evaluate("""() => { const row = document.querySelector('#favourites .mc-llm-thread-open');
            const title = row.querySelector('.mc-llm-thread-title').getBoundingClientRect();
            const who = row.querySelector('.mc-llm-thread-who');
            return {who: who.textContent, after: who.getBoundingClientRect().left >= title.right - 1,
                    subdued: getComputedStyle(who).color}; }""")
    finally:
        tab.close()

    assert found == {"who": "Bob", "after": True, "subdued": "rgb(120, 120, 120)"}


def test_new_thread_keeps_its_words_on_one_line_beside_the_heading(browser):
    """On a phone the heading took the row and squeezed + New thread into two
    lines of half a word each (found in a screenshot of the real pop-up). On
    the pop-up's head since 2026-11-09, beside ‹ Back and the heading, with
    the thread lists folded under it."""
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>body {{ margin: 0; font: 14px sans-serif; }}
.row {{ display: flex; flex-wrap: wrap; }} .row > * {{ flex: 1 1 0%; min-width: 0; }}</style>
<style>{(ROOT / "style.css").read_text(encoding="utf-8")}</style></head><body>
<div id="mc-llm-studio" style="width: 300px">
<div class="row mc-llm-sheet-head mc-llm-conversation-head">
  <button class="mc-llm-sheet-back" style="flex-grow: 0; min-width: 76px">‹ Back</button>
  <div class="block"><h4>Conversation</h4></div>
  <button id="new" style="flex-grow: 0; min-width: 0px">+ New thread</button>
</div></div></body></html>"""
    tab = browser.new_page(viewport={"width": 390, "height": 400})
    try:
        tab.set_content(html)
        found = tab.evaluate("""() => { const button = document.getElementById('new');
            const line = parseFloat(getComputedStyle(button).lineHeight) || 18;
            return {height: button.getBoundingClientRect().height, line,
                    fits: button.scrollWidth <= button.clientWidth + 1}; }""")
    finally:
        tab.close()

    assert found["height"] < found["line"] * 1.6 and found["fits"], found


def test_a_control_scrolled_under_the_pop_ups_head_stays_under_it(browser):
    """Gradio 4.40 draws every labelled control's title at ``--layer-4``
    (``BlockTitle.svelte``: ``position: relative; z-index: var(--layer-4)``),
    and the sticky head of a sheet sat at 2: the Generation section's labels
    scrolled over ‹ Back on a phone (a screenshot of the real pop-up,
    2026-11-09)."""
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>body {{ margin: 0; font: 14px sans-serif; }} :root {{ --layer-4: 40; }}
.row {{ display: flex; flex-wrap: wrap; }} .row > * {{ flex: 1 1 0%; min-width: 0; }}
.title {{ display: inline-block; position: relative; z-index: var(--layer-4); }}</style>
<style>{(ROOT / "style.css").read_text(encoding="utf-8")}</style></head><body>
<div id="mc-llm-studio" style="position: relative; height: 600px">
<div id="sheet" class="mc-llm-sheet" style="position: absolute; inset: 0; overflow-y: auto;
     z-index: 20; display: block; animation: none">
  <div class="row mc-llm-sheet-head mc-llm-conversation-head">
    <button class="mc-llm-sheet-back" style="flex-grow: 0; min-width: 76px">‹ Back</button>
    <div class="block"><h4>Conversation</h4></div>
  </div>
  <div style="height: 300px"></div>
  <div class="block"><label><span id="title" class="title">Repeat penalty</span></label></div>
  <div style="height: 2000px"></div>
</div></div></body></html>"""
    tab = browser.new_page(viewport={"width": 390, "height": 600})
    try:
        tab.set_content(html)
        found = tab.evaluate("""() => {
            const sheet = document.getElementById('sheet');
            const title = document.getElementById('title');
            const head = sheet.querySelector('.mc-llm-sheet-head');
            // Scroll the title's middle onto the head's middle.
            for (let step = 0; step < 3; step += 1) {
                const at = title.getBoundingClientRect(), on = head.getBoundingClientRect();
                sheet.scrollTop += (at.top + at.height / 2) - (on.top + on.height / 2);
            }
            const box = title.getBoundingClientRect(), over = head.getBoundingClientRect();
            const top = document.elementFromPoint(box.left + 4, box.top + box.height / 2);
            return {under: box.top >= over.top && box.bottom <= over.bottom,
                    head: head.contains(top)}; }""")
    finally:
        tab.close()

    assert found == {"under": True, "head": True}
