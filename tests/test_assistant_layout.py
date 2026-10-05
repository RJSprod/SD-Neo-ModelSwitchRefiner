"""The generation tabs' layout, measured by a real browser.

Two asks, both about Txt2Img's two columns:

    "the right column (generate, gallery, gallery buttons) does not scale all
    the way to fill the space. There is a gap at the bottom ... The right
    column should not scroll, it should just fill the space."

    "a third state for the flyout menu, that removes the left column from the
    text to image tab, and makes it scrollable in the flyout menu."

Node has no layout, so neither can be answered there. This renders Txt2Img the
way Forge Neo frames it under the Lobe theme -- Gradio's column rules (every
column a wrapping flex column), a resize-handle grid row, Lobe's 64 px header
and its sticky results column at 80 px, Lobe's split previewer's Generate box
inside the gallery's container -- with the person's own user.css loaded last,
as Forge loads it, and the real assistant scripts and stylesheet. Then it
measures:

    at rest       the results column ends 8 px above the window's bottom, and
                  without the fill's mark the user.css height is what is drawn
                  (so the page is not passing for want of the rule it beats);
    focused       the same, inside focus mode's root, whose padding is 8 px;
    an infotext   arriving under the buttons shrinks the gallery, and the
                  column still ends where it did;
    resized       the column follows the window;
    docked        the settings column is drawn exactly over the panel's
                  placeholder, scrolls inside itself with none of its blocks
                  squeezed, takes a press, and the gallery has the row's width;
                  with focus on as well; and undocked, it is back in its row.

It was checked once against real Gradio 4.40 markup (a Forge-shaped Blocks app
with Lobe's rules and this user.css) while it was written; this page keeps only
what decides the geometry. Needs the ``playwright`` package and a Chromium, and
skips without either.
"""

from __future__ import annotations

import glob
import os
import urllib.parse
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
JAVASCRIPT = ROOT / "javascript"
STYLE = ROOT / "style.css"
ORIGIN = "http://forge-layout.test"
MARGIN = 8

# The person's user.css, as they sent it: a gallery sized against the window.
USER_CSS = """
.token-counter { pointer-events: none; }
.thumbnail-item { aspect-ratio: auto !important; }
#txt2img_gallery, #img2img_gallery { min-height: 80vh; }
#txt2img_gallery_container, #img2img_gallery_container { height: 85vh !important; }
#txt2img_gallery > .grid-wrap, #txt2img_gallery > .preview,
#img2img_gallery > .grid-wrap, #img2img_gallery > .preview { max-height: 80vh; }
"""

# What Gradio 4.40 and Lobe do to these columns, and no more.
HOST_CSS = """
*, ::before, ::after { box-sizing: border-box; }
body { margin: 0; background: #0b0f19; color: #eee; font: 14px sans-serif;
       --background-fill-primary: #0b0f19; --border-color-primary: #374151; }
.row { display: flex; flex-wrap: wrap; gap: 16px; }
.unequal-height { align-items: flex-start; }
.column { display: flex; flex-direction: column; flex-wrap: wrap; gap: 16px;
          min-width: min(320px, 100%); flex-grow: 1; }
.gr-group, .styler { display: flex; flex-direction: column; }
.block { position: relative; }
.grid-wrap { overflow-y: auto; }
.fixed-height { min-height: 320px; max-height: 55vh; }
.resize-handle { grid-column: 2 / 3; min-width: 16px; max-width: 16px; }
.tab-nav { display: flex; height: 40px; }
#lobe-header { position: sticky; top: 0; height: 64px; z-index: 999; background: #000; }
#txt2img_results, #img2img_results { position: sticky; top: 80px !important; }
.panel { margin: 0 !important; padding: 16px !important; }
[id$='_gallery_container'] { min-height: 470px; }
[id$='_gallery_container'] > div:not([id$='_generate_box']) { flex-grow: 1; }
[id$='img_settings'] { display: flex !important; flex-direction: column !important; }
#txt2img_settings .block { height: 90px; background: #1f2937; }
#txt2img_generate { height: 40px; width: 100%; }
#image_buttons_txt2img button { flex: 1 1 100px; height: 32px; }
.infotext p { margin: 0; line-height: 24px; }
"""

SETTINGS_BLOCKS = 24


def page_html() -> str:
    blocks = "".join(f'<div class="block" id="setting_{n}">Setting {n}</div>'
                     for n in range(SETTINGS_BLOCKS))
    buttons = "".join(f"<button>{n}</button>" for n in range(8))
    scripts = "".join(f'<script src="/{path.name}"></script>'
                      for path in sorted(JAVASCRIPT.glob("forge_assistant*.js")))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>{HOST_CSS}</style>
<link rel="stylesheet" href="/style.css">
<style id="user-css">{USER_CSS}</style>
</head><body class="dark">
<gradio-app><div class="gradio-container">
<div id="lobe-header"></div>
<div id="tabs" class="tabs">
  <div class="tab-nav" role="tablist">
    <button role="tab" id="tab_txt2img-button" aria-controls="tab_txt2img"
            aria-selected="true" class="selected">Txt2img</button>
    <button role="tab" id="tab_img2img-button" aria-controls="tab_img2img"
            aria-selected="false">Img2img</button>
  </div>
  <div id="tab_txt2img" class="tabitem" role="tabpanel" style="display: block">
    <div class="row resize-handle-row unequal-height"
         style="display: grid; gap: 0; grid-template-columns: 1fr 16px 1fr">
      <div id="txt2img_settings" class="column" style="min-width: min(320px, 100%)">
        <div class="block" id="txt2img_prompt"><textarea aria-label="Prompt"></textarea></div>
        {blocks}
      </div>
      <div class="resize-handle"></div>
      <div id="txt2img_results" class="column">
        <div id="txt2img_results_panel" class="column panel">
          <div id="txt2img_gallery_container" class="gr-group">
            <div id="txt2img_generate_box" class="row"><button id="txt2img_generate">Generate</button></div>
            <div class="styler">
              <div id="txt2img_gallery" class="block gradio-gallery">
                <div class="grid-wrap fixed-height"><div class="grid-container"></div></div>
              </div>
            </div>
          </div>
          <div id="image_buttons_txt2img" class="row image-buttons">{buttons}</div>
          <div class="gr-group">
            <div id="html_info_txt2img" class="block infotext"></div>
            <div id="html_log_txt2img" class="block"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div id="tab_img2img" class="tabitem" role="tabpanel" style="display: none"></div>
</div>
</div></gradio-app>
<script>
function gradioApp() {{ return document.querySelector("gradio-app"); }}
const uiLoaded = [];
function onUiLoaded(fn) {{ uiLoaded.push(fn); }}
</script>
{scripts}
<script>uiLoaded.forEach((fn) => fn());</script>
</body></html>"""


def _answer(route, html: str) -> None:
    path = urllib.parse.urlsplit(route.request.url).path
    if path == "/":
        route.fulfill(status=200, content_type="text/html", body=html)
    elif path == "/style.css":
        route.fulfill(status=200, content_type="text/css", body=STYLE.read_text(encoding="utf-8"))
    elif path.endswith(".js") and (JAVASCRIPT / path.lstrip("/")).is_file():
        route.fulfill(status=200, content_type="text/javascript",
                      body=(JAVASCRIPT / path.lstrip("/")).read_text(encoding="utf-8"))
    else:
        # The assistant's store asks the server about conversations; there is
        # none here, and a refusal is what it already copes with.
        route.fulfill(status=404, body="")


def _launch(playwright):
    """Chromium: Playwright's own build, or the one found where browsers are kept."""
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


def settle(page, frames: int = 3) -> None:
    """A few frames: the fill measures in one, and its write is laid out by the next."""
    page.evaluate("(n) => new Promise((done) => { const step = (k) => k ? "
                  "requestAnimationFrame(() => step(k - 1)) : done(); step(n); })", frames)


def open_page(browser, width: int = 1600, height: int = 1000):
    page = browser.new_page(viewport={"width": width, "height": height})
    html = page_html()
    page.route(ORIGIN + "/**", lambda route: _answer(route, html))
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!(window.forgeAssistant && window.forgeAssistant.shell)")
    settle(page)
    return page


MEASURE = """() => {
    const box = (node) => { const r = node.getBoundingClientRect();
                            return {top: r.top, bottom: r.bottom, left: r.left,
                                    width: r.width, height: r.height}; };
    const results = document.getElementById("txt2img_results");
    // The last thing in the column, not the column's own box.
    const last = Math.max(...[...results.querySelectorAll("*")].map(
        (node) => node.getBoundingClientRect().bottom));
    return {inner: innerHeight, results: box(results), last,
            gallery: box(document.getElementById("txt2img_gallery")),
            container: box(document.getElementById("txt2img_gallery_container")),
            buttons: box(document.getElementById("image_buttons_txt2img")),
            marked: results.hasAttribute("data-forge-assistant-fill")};
}"""


def measure(page) -> dict:
    return page.evaluate(MEASURE)


def test_the_results_column_ends_at_the_bottom_of_the_window(browser):
    page = open_page(browser)
    try:
        found = measure(page)
        assert found["marked"] is True
        assert abs(found["last"] - (found["inner"] - MARGIN)) <= 1, found
        assert found["buttons"]["bottom"] <= found["inner"] - MARGIN + 1

        # The user.css is in force on this page: without the fill's mark its
        # 85vh is what the container is drawn at.
        page.evaluate("document.getElementById('txt2img_results')"
                      ".removeAttribute('data-forge-assistant-fill')")
        bare = measure(page)
        assert abs(bare["container"]["height"] - 0.85 * bare["inner"]) <= 1, bare
    finally:
        page.close()


def test_focus_mode_leaves_no_gap_under_the_column(browser):
    page = open_page(browser)
    try:
        entered = page.evaluate("forgeAssistant.focus.enter('tab_txt2img', "
                                "forgeAssistant.shell.host)")
        assert entered["ok"] is True
        settle(page)
        found = measure(page)
        # Focus's root pads its edges by 8 px, and the column sits inside them.
        assert abs(found["last"] - (found["inner"] - MARGIN)) <= 1, found
        assert found["results"]["top"] <= 80, "the header's reservation is gone"

        page.evaluate("forgeAssistant.focus.exit()")
        settle(page)
        back = measure(page)
        assert abs(back["last"] - (back["inner"] - MARGIN)) <= 1, back
    finally:
        page.close()


def test_an_infotext_shrinks_the_gallery_and_not_the_window(browser):
    page = open_page(browser)
    try:
        before = measure(page)
        page.evaluate("document.getElementById('html_info_txt2img').innerHTML = "
                      "'<p>a prompt</p>'.repeat(4)")
        settle(page, 4)
        after = measure(page)
        assert abs(after["last"] - (after["inner"] - MARGIN)) <= 1, after
        assert after["gallery"]["height"] < before["gallery"]["height"] - 80
    finally:
        page.close()


def test_the_column_follows_the_window(browser):
    page = open_page(browser)
    try:
        page.set_viewport_size({"width": 1400, "height": 820})
        settle(page, 4)
        found = measure(page)
        assert found["inner"] == 820
        assert abs(found["last"] - (found["inner"] - MARGIN)) <= 1, found
    finally:
        page.close()


DOCKED = """() => {
    const column = document.getElementById("txt2img_settings");
    const slot = document.querySelector(".forge-assistant-dock");
    const box = (node) => { const r = node.getBoundingClientRect();
                            return [Math.round(r.left), Math.round(r.top),
                                    Math.round(r.width), Math.round(r.height)]; };
    const prompt = column.querySelector("textarea");
    prompt.scrollIntoView({block: "nearest"});
    const p = prompt.getBoundingClientRect();
    const hit = document.elementFromPoint(p.left + p.width / 2, p.top + p.height / 2);
    const row = column.parentElement;
    const results = document.getElementById("txt2img_results");
    return {column: box(column), slot: box(slot),
            scroll: [column.scrollHeight, column.clientHeight],
            squeezed: [...column.querySelectorAll(".block[id^='setting_']")]
                .filter((node) => Math.round(node.getBoundingClientRect().height) !== 90).length,
            pressed: hit === prompt,
            results: results.getBoundingClientRect().width, row: row.getBoundingClientRect().width,
            pressedButton: document.querySelector(".forge-assistant-settings")
                .getAttribute("aria-pressed")};
}"""


def test_the_settings_column_docks_in_the_panel_and_scrolls_there(browser):
    page = open_page(browser, 1600, 900)
    try:
        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        page.click(".forge-assistant-settings")
        settle(page)
        found = page.evaluate(DOCKED)
        assert found["pressedButton"] == "true"
        assert found["column"] == found["slot"], found
        assert found["scroll"][0] > found["scroll"][1] + 500, "the column scrolls inside itself"
        assert found["squeezed"] == 0, "no block is squeezed to fit"
        assert found["pressed"] is True, "a press over the panel reaches the column"
        assert abs(found["results"] - found["row"]) <= 1, "the gallery has the row's width"
        filled = measure(page)
        assert abs(filled["last"] - (filled["inner"] - MARGIN)) <= 1, filled

        # Focus as well: the column is inside focus's own layer now, and is
        # still drawn over the panel's placeholder and still takes a press.
        page.evaluate("forgeAssistant.focus.enter('tab_txt2img', forgeAssistant.shell.host)")
        settle(page)
        focused = page.evaluate(DOCKED)
        assert focused["column"] == focused["slot"] and focused["pressed"] is True, focused

        page.evaluate("forgeAssistant.focus.exit()")
        page.click(".forge-assistant-settings")
        settle(page)
        back = page.evaluate("""() => {
            const column = document.getElementById("txt2img_settings");
            const results = document.getElementById("txt2img_results");
            return {docked: column.classList.contains("forge-assistant-docked-column"),
                    position: getComputedStyle(column).position,
                    beside: column.getBoundingClientRect().right
                        <= results.getBoundingClientRect().left,
                    inline: column.getAttribute("style")};
        }""")
        assert back == {"docked": False, "position": "static", "beside": True,
                        "inline": "min-width: min(320px, 100%);"}, back
    finally:
        page.close()


SWITCH = """(to) => {
    for (const id of ["tab_txt2img", "tab_img2img"]) {
        const on = id === to;
        document.getElementById(id).style.display = on ? "block" : "none";
        const button = document.getElementById(id + "-button");
        button.classList.toggle("selected", on);
        button.setAttribute("aria-selected", String(on));
    }
}"""

WHERE = """() => {
    const column = document.getElementById("txt2img_settings");
    const slot = document.querySelector(".forge-assistant-dock");
    const box = (node) => { const r = node.getBoundingClientRect();
                            return [Math.round(r.left), Math.round(r.top)]; };
    return {docked: column.classList.contains("forge-assistant-docked-column"),
            button: !document.querySelector(".forge-assistant-settings").hidden,
            column: box(column), slot: box(slot)};
}"""


def test_the_docked_column_follows_the_panel_and_the_workspace(browser):
    page = open_page(browser, 1600, 900)
    try:
        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        page.click(".forge-assistant-settings")
        settle(page)

        # Dragged by its header, the panel carries the column with it, frame
        # by frame and not only where it is let go.
        grip = page.evaluate("(() => { const r = document.querySelector('.forge-assistant-grip')"
                             ".getBoundingClientRect(); return [r.left + r.width / 2,"
                             " r.top + r.height / 2]; })()")
        start = page.evaluate(WHERE)
        page.mouse.move(*grip)
        page.mouse.down()
        page.mouse.move(grip[0] - 500, grip[1] + 30, steps=6)
        moving = page.evaluate(WHERE)
        page.mouse.up()
        settle(page)
        assert moving["column"] == moving["slot"], moving
        assert moving["column"][0] < start["column"][0] - 400

        # Another workspace: the column goes back to its page and the button
        # goes; back on Txt2Img, it is docked again.
        page.evaluate(SWITCH, "tab_img2img")
        page.wait_for_function("!document.getElementById('txt2img_settings')"
                               ".classList.contains('forge-assistant-docked-column')")
        away = page.evaluate(WHERE)
        assert away["docked"] is False and away["button"] is False, away
        page.evaluate(SWITCH, "tab_txt2img")
        page.wait_for_function("document.getElementById('txt2img_settings')"
                               ".classList.contains('forge-assistant-docked-column')")
        settle(page)
        back = page.evaluate(WHERE)
        assert back["button"] is True and back["column"] == back["slot"], back
    finally:
        page.close()
