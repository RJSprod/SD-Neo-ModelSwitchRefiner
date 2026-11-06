"""Settings -> Voice Chat, a page at a time (mc_voice_ui.VOICE_PAGES,
javascript/voice_chat.js ``showVoicePage``).

"The voice chat settings menu is a MESS. I want you to look at it and redesign
this settings menu so its easier to navigate" (2026-11-06). It was one column
three to six thousand pixels long. Now a row of pages at its top shows one
purpose at a time -- Engine, Voices, Dictation, Pipeline, Setup -- with All for
the old column, and Forge's settings search showing everything while it has
text. Nothing is moved: what is not on the page is hidden by a class.

Held here: every option the section registers has a page (one added later
cannot fall off all of them), every part of the install row is on one, and in
Chromium the pages show what they say, the choice is remembered, a search shows
everything, and the swap of the install row after an engine change keeps the
page.
"""

from __future__ import annotations

import glob
import html
import json
import os
import re
from pathlib import Path

import pytest

import mc_voice_ui

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "voice_chat.js"
STYLE = ROOT / "style.css"
PAGES = [name for name, _ in mc_voice_ui.VOICE_PAGES]


def section_options():
    import model_chain  # noqa: F401  -- registers the section
    from modules import shared

    return {key: info for key, info in shared.options_templates.items()
            if (getattr(info, "section", None) or (None,))[0] == "model_chain_voice"}


class TestEverythingHasAPage:
    def test_every_option_of_the_section_but_the_install_row_has_a_page(self, host, voice_root):
        options = section_options()
        mapping = mc_voice_ui.option_pages()
        unplaced = [key for key in options
                    if key != "model_chain_voice_status" and key not in mapping]
        assert not unplaced, unplaced
        assert set(mapping.values()) <= set(PAGES) - {"all"}
        assert all(key in options for key in mapping), set(mapping) - set(options)

    def test_the_row_of_pages_heads_the_install_row_and_names_each_page(self, host, voice_root):
        markup = mc_voice_ui.settings_html()
        nav = markup.index('class="mc-voice-pages"')
        assert nav < markup.index("mc-voice-page-part"), "the pages first"
        tabs = re.findall(r'data-mc-voice-page-tab="([a-z]+)" aria-selected="(true|false)"', markup)
        assert [name for name, _ in tabs] == PAGES
        assert [name for name, on in tabs if on == "true"] == [mc_voice_ui.DEFAULT_VOICE_PAGE]
        carried = re.search(r"data-mc-voice-option-pages='([^']*)'", markup).group(1)
        assert json.loads(html.unescape(carried)) == mc_voice_ui.option_pages()

    def test_every_part_of_the_install_row_is_on_a_page(self, host, voice_root):
        markup = mc_voice_ui.settings_html()
        inner = markup[markup.index("</nav>") + len("</nav>"):-len("</div>")]
        depth, position, parts = 0, 0, []
        for match in re.finditer(r"<(/?)div\b[^>]*>", inner):
            if match.group(1):
                depth -= 1
                continue
            if depth == 0:
                assert 'class="mc-voice-page-part"' in match.group(0), (
                    "a part on no page: " + inner[match.start():match.start() + 160])
                parts.append(re.search(r'data-mc-voice-page="([a-z]+)"', match.group(0)).group(1))
            depth += 1
        assert depth == 0
        assert set(parts) == {"engine", "dictation", "pipeline", "setup"}, parts


sync_api = pytest.importorskip("playwright.sync_api")


def _launch(playwright):
    try:
        return playwright.chromium.launch()
    except Exception:
        pass
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "/opt/pw-browsers"
    for pattern in ("chromium-*/chrome-linux/chrome", "chromium-*/chrome-linux64/chrome"):
        for executable in sorted(glob.glob(str(Path(base) / pattern)), reverse=True):
            try:
                return playwright.chromium.launch(executable_path=executable)
            except Exception:
                continue
    return None


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        found = _launch(playwright)
        if found is None:
            pytest.skip("no Chromium to render with")
        yield found
        found.close()


ORIGIN = "http://voice-settings.test"


def page_markup(status: str, voices: str, options: dict) -> str:
    """Forge's Settings -> Voice Chat: the section's column, the install row
    and the voices as two HTML options, each other option a block of its own
    under them, and the settings search above."""
    blocks = []
    for key in options:
        if key == "model_chain_voice_status":
            blocks.append(f'<div id="setting_{key}" class="block settings-info">{status}</div>')
        elif key == "model_chain_voice_voices":
            blocks.append(f'<div id="setting_{key}" class="block settings-info">{voices}</div>')
        else:
            blocks.append(f'<div id="setting_{key}" class="block"><label>{html.escape(key)}</label></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="/style.css">
<script>const hooks = []; function onUiLoaded(f) {{ hooks.push(f); }} function onAfterUiUpdate() {{}}
function gradioApp() {{ return document; }}
window.fetch = () => Promise.reject(new Error("offline"));</script>
</head><body><gradio-app>
<div id="settings_search"><label><input type="text"></label></div>
<div id="settings" class="tabs"><div id="settings_model_chain_voice" class="tabitem">
<div id="column_settings_model_chain_voice" class="column">{''.join(blocks)}</div></div></div>
</gradio-app>
<script src="/javascript/voice_chat.js"></script>
<script>hooks.forEach((f) => {{ try {{ f(); }} catch (e) {{ console.log(e); }} }});</script>
</body></html>"""


SHOWN = """() => {
    const drawn = (node) => !!node && node.getClientRects().length > 0;
    const parts = {};
    document.querySelectorAll('.mc-voice-page-part').forEach((part) => {
        const name = part.getAttribute('data-mc-voice-page');
        parts[name] = (parts[name] || false) || drawn(part);
    });
    const options = {};
    document.querySelectorAll('[id^="setting_"]').forEach((node) => {
        options[node.id.slice(8)] = drawn(node);
    });
    const tab = document.querySelector('.mc-voice-page-tab[aria-selected="true"]');
    return {parts, options, tab: tab && tab.getAttribute('data-mc-voice-page-tab')};
}"""


@pytest.fixture
def served(host, voice_root):
    options = section_options()
    return page_markup(mc_voice_ui.settings_html(), mc_voice_ui.voices_html(), options), \
        mc_voice_ui.option_pages()


def open_page(browser, markup):
    context = browser.new_context(viewport={"width": 412, "height": 915})
    page = context.new_page()

    def serve(route):
        path = route.request.url[len(ORIGIN):]
        if path == "/":
            route.fulfill(status=200, content_type="text/html", body=markup)
        elif path == "/style.css":
            route.fulfill(status=200, content_type="text/css",
                          body=STYLE.read_text(encoding="utf-8", errors="replace"))
        elif path == "/javascript/voice_chat.js":
            route.fulfill(status=200, content_type="text/javascript",
                          body=SCRIPT.read_text(encoding="utf-8"))
        else:
            route.fulfill(status=404, body="")

    page.route(ORIGIN + "/**", serve)
    page.goto(ORIGIN + "/")
    page.wait_for_timeout(100)
    return page


def assert_on(found, page_name, mapping):
    for name, drawn in found["parts"].items():
        assert drawn == (name == page_name), (page_name, name, found["parts"])
    for key, page in mapping.items():
        assert found["options"][key] == (page == page_name), (page_name, key)
    assert found["options"]["model_chain_voice_status"] is True, "the row of pages is always there"


class TestAPageAtATime:
    def test_it_opens_on_the_engine_and_each_page_shows_its_own(self, browser, served):
        markup, mapping = served
        page = open_page(browser, markup)
        try:
            found = page.evaluate(SHOWN)
            assert found["tab"] == "engine"
            assert_on(found, "engine", mapping)
            for name in ("voices", "dictation", "pipeline", "setup"):
                page.click(f'[data-mc-voice-page-tab="{name}"]')
                found = page.evaluate(SHOWN)
                assert found["tab"] == name
                assert_on(found, name, mapping)
            page.click('[data-mc-voice-page-tab="all"]')
            found = page.evaluate(SHOWN)
            assert all(found["parts"].values()) and all(found["options"].values()), found
        finally:
            page.context.close()

    def test_the_page_chosen_is_remembered(self, browser, served):
        markup, mapping = served
        page = open_page(browser, markup)
        try:
            page.click('[data-mc-voice-page-tab="dictation"]')
            page.reload()
            page.wait_for_timeout(100)
            found = page.evaluate(SHOWN)
            assert found["tab"] == "dictation"
            assert_on(found, "dictation", mapping)
        finally:
            page.context.close()

    def test_a_settings_search_shows_every_page_until_it_is_cleared(self, browser, served):
        markup, mapping = served
        page = open_page(browser, markup)
        try:
            page.fill("#settings_search input", "pipeline")
            found = page.evaluate(SHOWN)
            assert all(found["parts"].values()) and all(found["options"].values()), found
            page.fill("#settings_search input", "")
            assert_on(page.evaluate(SHOWN), "engine", mapping)
        finally:
            page.context.close()

    def test_the_install_row_swapped_after_an_engine_change_keeps_the_page(self, browser, served):
        """An engine change redraws the install row whole (``swapSurface``)."""
        markup, mapping = served
        page = open_page(browser, markup)
        try:
            page.click('[data-mc-voice-page-tab="setup"]')
            page.evaluate("""() => {
                const row = document.querySelector('.mc-voice-settings');
                const fresh = row.cloneNode(true);
                delete fresh.dataset.mcVoiceWired;
                fresh.querySelectorAll('.mc-voice-page-hidden').forEach((n) => n.classList.remove('mc-voice-page-hidden'));
                row.outerHTML = fresh.outerHTML;
            }""")
            page.evaluate("hooks.forEach((f) => f())")
            found = page.evaluate(SHOWN)
            assert found["tab"] == "setup"
            assert_on(found, "setup", mapping)
        finally:
            page.context.close()
