"""The panel's formatting in a real browser: KaTeX fetched when needed, quotes coloured.

Node draws the HTML (``tests/test_assistant_markdown_js.py``); only a browser
says whether the vendored KaTeX is fetched through its route, once, and typesets
every formula where it stands; whether the stylesheet gives a quotation
Oobabooga's colours from Gradio's palette by day and at night; and whether a
displayed formula wider than the bubble scrolls inside it. The page is the
Forge-shaped one ``tests/test_assistant_layout.py`` builds, with the assistant's
real scripts and stylesheet, and the KaTeX route answered from ``vendor/katex``
by ``mc_katex`` itself.
"""

from __future__ import annotations

import urllib.parse

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

import mc_katex  # noqa: E402
from test_assistant_layout import ORIGIN, _answer, _launch, page_html, settle  # noqa: E402

# Gradio's own palette, as its base stylesheet defines it on every page.
GRADIO_PALETTE = ":root { --color-blue-600: #2563eb; --color-orange-300: #ffb066; }"
BLUE = "rgb(37, 99, 235)"
AMBER = "rgb(255, 176, 102)"

CONVERSATION = {
    "conversation": {"messages": [
        {"index": 0, "role": "user", "active": 0, "versions": ["q"],
         "text": 'What does $\\rightarrow$ mean, in "plain" words?'},
        {"index": 1, "role": "assistant", "active": 0, "versions": ["a"],
         "text": ("## Arrows\n\n*   $\\rightarrow$ means *to*\n*   It costs $5 and $10\n\n"
                  "$$\\frac{a}{b}$$\n\n| a | b |\n|---|---|\n| \"q\" | `$x$` |")},
    ]},
    "selection": {"epoch": "e1"}, "unread": 0, "operation": None,
}


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        found = _launch(playwright)
        if found is None:
            pytest.skip("no Chromium to render with")
        yield found
        found.close()


def open_page(browser, katex: str = "serve"):
    """The page, with the KaTeX route answered (or refused) and every ask kept."""
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    html = page_html()
    asked = []
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def answer(route):
        path = urllib.parse.urlsplit(route.request.url).path
        if path.startswith("/model-chain/katex/"):
            asked.append(path)
            found = mc_katex.file_for(path[len(mc_katex.PREFIX):]) \
                if path.startswith(mc_katex.PREFIX) and katex == "serve" else None
            if found:
                route.fulfill(status=200, content_type=found[1], body=found[0])
            else:
                route.fulfill(status=404, body="")
            return
        _answer(route, html)

    page.route(ORIGIN + "/**", answer)
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!(window.forgeAssistant && window.forgeAssistant.shell)")
    page.add_style_tag(content=GRADIO_PALETTE)
    page.click(".forge-assistant-launcher")
    page.wait_for_function("forgeAssistant.shell.state.panelOpen === true", timeout=3000)
    settle(page)
    return page, asked, errors


def draw(page, view=None) -> None:
    page.evaluate("(view) => window.forgeAssistant.shell.renderTranscript(view)", view or CONVERSATION)


COUNT = """() => {
    const root = document.querySelector('#forge-assistant-root');
    return {waiting: root.querySelectorAll('.forge-assistant-math[data-tex]').length,
            typeset: root.querySelectorAll('.katex').length};
}"""


def test_the_first_formula_fetches_katex_once_and_every_formula_is_typeset_where_it_stands(browser):
    page, asked, errors = open_page(browser)
    try:
        assert asked == [], "a page with no formula on it fetches nothing"
        draw(page)
        before = page.evaluate(COUNT)
        page.wait_for_function("document.querySelectorAll('#forge-assistant-root .katex').length >= 3",
                               timeout=10000)
        after = page.evaluate("""() => {
            const root = document.querySelector('#forge-assistant-root');
            const formula = root.querySelector('.katex');
            return {waiting: root.querySelectorAll('.forge-assistant-math[data-tex]').length,
                    tex: [...root.querySelectorAll('annotation')].map((note) => note.textContent),
                    displayed: root.querySelectorAll('.katex-display').length,
                    face: getComputedStyle(formula).fontFamily,
                    code: [...root.querySelectorAll('.forge-assistant-text td code')].map((c) => c.textContent),
                    money: root.querySelector('.forge-assistant-text ul').textContent};
        }""")
        # A message drawn after KaTeX is in is typeset at once, with nothing fetched again.
        draw(page, {**CONVERSATION, "selection": {"epoch": "e2"}})
        again = page.evaluate(COUNT)
    finally:
        page.close()

    assert before == {"waiting": 3, "typeset": 0}
    assert after["waiting"] == 0
    assert after["tex"] == ["\\rightarrow", "\\rightarrow", "\\frac{a}{b}"]
    assert after["displayed"] == 1
    assert after["face"].startswith("KaTeX_Main"), "the vendored stylesheet is applied"
    assert after["code"] == ["$x$"]
    assert "It costs $5 and $10" in after["money"]
    assert again == {"waiting": 0, "typeset": 3}
    fonts = [path for path in asked if path.startswith(mc_katex.PREFIX + "fonts/")]
    assert sorted(set(asked) - set(fonts)) == [mc_katex.PREFIX + "katex.min.css",
                                               mc_katex.PREFIX + "katex.min.js"]
    assert len(asked) - len(fonts) == 2, "the script and the stylesheet once each"
    assert fonts and all(mc_katex.file_for(path[len(mc_katex.PREFIX):]) for path in fonts), \
        "the fonts the stylesheet names, through the same route"
    assert not errors


def test_a_quotation_is_amber_at_night_and_blue_by_day(browser):
    page, _asked, _errors = open_page(browser)
    try:
        draw(page)
        quote = "document.querySelector('#forge-assistant-root q.forge-assistant-quote')"
        night = page.evaluate(f"""() => {{ const q = {quote};
            return {{colour: getComputedStyle(q).color, text: q.textContent,
                     before: getComputedStyle(q, '::before').content}}; }}""")
        page.evaluate("document.body.classList.remove('dark')")
        day = page.evaluate(f"getComputedStyle({quote}).color")
    finally:
        page.close()

    assert night == {"colour": AMBER, "text": '"plain"', "before": "none"}, \
        "the message's own quote marks, and no second pair from the browser"
    assert day == BLUE


def test_a_route_that_answers_nothing_leaves_every_formula_as_written(browser):
    page, asked, errors = open_page(browser, katex="refuse")
    try:
        draw(page)
        page.wait_for_function("document.querySelectorAll('link[data-forge-assistant-katex]').length === 0 "
                               "&& performance.getEntriesByType('resource').some((r) => r.name.includes('katex.min.js'))",
                               timeout=10000)
        settle(page, 10)
        shown = page.evaluate("""() => [...document.querySelectorAll('#forge-assistant-root .forge-assistant-math')]
            .map((node) => [node.textContent, getComputedStyle(node).fontFamily])""")
    finally:
        page.close()

    assert [text for text, _face in shown] == ["$\\rightarrow$", "$\\rightarrow$", "$$\\frac{a}{b}$$"]
    assert asked, "it was asked for"
    assert not errors


def test_a_displayed_formula_wider_than_the_bubble_scrolls_inside_it(browser):
    wide = "$$" + " + ".join(f"x_{{{n}}}^{{{n}}}" for n in range(60)) + "$$"
    view = {"conversation": {"messages": [
        {"index": 0, "role": "assistant", "active": 0, "versions": ["w"], "text": wide}]},
        "selection": {"epoch": "w"}, "unread": 0, "operation": None}
    page, _asked, _errors = open_page(browser)
    try:
        draw(page, view)
        page.wait_for_function("!!document.querySelector('#forge-assistant-root .katex-display')", timeout=10000)
        settle(page)
        found = page.evaluate("""() => {
            const display = document.querySelector('#forge-assistant-root .katex-display');
            const bubble = display.closest('.forge-assistant-bubble');
            return {scrolls: display.scrollWidth > display.clientWidth,
                    overflow: getComputedStyle(display).overflowX,
                    inside: display.getBoundingClientRect().right <= bubble.getBoundingClientRect().right + 1};
        }""")
    finally:
        page.close()

    assert found == {"scrolls": True, "overflow": "auto", "inside": True}
