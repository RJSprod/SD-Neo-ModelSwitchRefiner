"""The generation tabs' layout, measured by a real browser.

Two asks, both about Txt2Img's two columns:

    "The gallery need to fill its area, only bounded by the screen border,
    generate / interrupt / cancel, progress bar (when in progress), and gallery
    buttons. Everything else should be off page. The image should scale up to
    fill the maximum allowed space. The padding above generate and below
    gallery buttons should be the same, giving view of gallery in perfect
    center ... All other content below should still be rendered, but off page
    and below the view, thus requiring a scroll to see it."

    "a third state for the flyout menu, that removes the left column from the
    text to image tab, and makes it scrollable in the flyout menu."

Node has no layout, so neither can be answered there. This renders Txt2Img the
way Forge Neo frames it under the Lobe theme -- Gradio's column rules (every
column a wrapping flex column), a resize-handle grid row, Lobe's 64 px header
and its sticky results column at 80 px, Lobe's split previewer's Generate box
inside the gallery's container -- with the person's own user.css loaded last,
as Forge loads it, and the real assistant scripts and stylesheet. Then it
measures:

    at rest       the gap from the header's bottom to Generate is the gap
                  from the gallery's buttons to the window's bottom, and what
                  follows the buttons starts below the window; without the
                  fill's mark the user.css height is what is drawn (so the page
                  is not passing for want of the rule it beats);
    focused       the same, inside focus mode's root;
    an infotext   arriving after the buttons takes nothing from the gallery,
                  and stays below the view until the page is scrolled to it;
    progress      a progress bar above Generate takes its room from the
                  gallery, and the buttons stay where they were;
    resized       the gallery follows the window;
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
/* A fraction of a pixel, as a theme's sizes in rem give: the gallery's height
   is a whole number of pixels, and what follows the buttons must still start
   at the view's bottom, not a fraction above it. */
#txt2img_generate { height: 40.4px; width: 100%; }
.progressDiv { position: relative !important; top: 0 !important; height: 20px; }
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
<meta name="viewport" content="width=device-width, initial-scale=1">
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
            <div id="html_info_txt2img" class="block infotext"><p>food</p><p>Steps: 8, Sampler: Res Multistep, CFG scale: 1, Seed: 2771589119</p></div>
            <div id="html_log_txt2img" class="block"><p>Time taken: 8.1 sec.</p></div>
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
    const q = (selector) => document.querySelector(selector);
    const box = (node) => { const r = node.getBoundingClientRect();
                            return {top: r.top, bottom: r.bottom, height: r.height}; };
    // The view: focus mode's root, inside its border; or the window under the
    // header held over it.
    const root = q(".forge-assistant-focus-root");
    let view;
    if (root) {
        const r = root.getBoundingClientRect();
        const top = r.top + root.clientTop;
        view = {top, bottom: top + root.clientHeight};
    } else {
        const header = q("#lobe-header");
        view = {top: header ? header.getBoundingClientRect().bottom : 0, bottom: innerHeight};
    }
    const results = q("#txt2img_results");
    // The first thing kept in view: Generate's box, or a progress bar in the
    // column's flow above it.
    const kept = [q("#txt2img_generate_box"), ...results.querySelectorAll(".progressDiv")]
        .filter((node) => node && getComputedStyle(node).position !== "absolute")
        .map((node) => node.getBoundingClientRect().top);
    return {view, first: Math.min(...kept), inner: innerHeight,
            buttons: box(q("#image_buttons_txt2img")),
            gallery: box(q("#txt2img_gallery")),
            container: box(q("#txt2img_gallery_container")),
            // What follows the buttons: the infotext and the log.
            next: q("#html_info_txt2img").parentElement.getBoundingClientRect().top,
            marked: results.hasAttribute("data-forge-assistant-fill")};
}"""


def measure(page) -> dict:
    return page.evaluate(MEASURE)


def assert_centred(found) -> None:
    """The same gap above the first thing kept in view as under the gallery's
    buttons, and nothing after the buttons inside the view."""
    above = found["first"] - found["view"]["top"]
    below = found["view"]["bottom"] - found["buttons"]["bottom"]
    assert above > 0 and abs(above - below) <= 1, found
    assert found["next"] >= found["view"]["bottom"], found


def test_the_gallery_fills_the_view_between_generate_and_its_buttons(browser):
    page = open_page(browser)
    try:
        found = measure(page)
        assert found["marked"] is True
        assert found["view"]["top"] == 64, "measured from under the header"
        assert_centred(found)

        # The assistant's own panel, put at the top over the column -- right
        # under the header, where the page is asked what holds the top of the
        # window -- is not a header: the gallery is measured as before.
        page.evaluate("""() => {
            const shell = forgeAssistant.shell;
            shell.state.anchorOverride = "top-right";
            shell.state.panelWidth = 640;
            shell.open();
        }""")
        settle(page)
        covers = page.evaluate("""() => {
            const results = document.getElementById("txt2img_results").getBoundingClientRect();
            const panel = document.getElementById("forge-assistant-panel").getBoundingClientRect();
            const x = results.left + results.width / 2;
            return panel.left <= x && x <= panel.right && panel.top <= 65 && 65 <= panel.bottom;
        }""")
        assert covers is True, "the panel is where the page is asked"
        settle(page)
        page.evaluate("forgeAssistant.fill.now()")
        settle(page)
        under = measure(page)
        assert abs(under["gallery"]["height"] - found["gallery"]["height"]) <= 1, under

        # The user.css is in force on this page: without the fill's mark its
        # 85vh is what the container is drawn at.
        page.evaluate("document.getElementById('txt2img_results')"
                      ".removeAttribute('data-forge-assistant-fill')")
        bare = measure(page)
        assert abs(bare["container"]["height"] - 0.85 * bare["inner"]) <= 1, bare
    finally:
        page.close()


def test_focus_mode_centres_the_gallery_in_its_own_view(browser):
    page = open_page(browser)
    try:
        entered = page.evaluate("forgeAssistant.focus.enter('tab_txt2img', "
                                "forgeAssistant.shell.host)")
        assert entered["ok"] is True
        settle(page)
        found = measure(page)
        assert found["view"]["top"] == 0
        assert_centred(found)
        assert found["first"] <= 80, "the header's reservation is gone"

        page.evaluate("forgeAssistant.focus.exit()")
        settle(page)
        assert_centred(measure(page))
    finally:
        page.close()


def test_what_follows_the_buttons_takes_nothing_from_the_gallery(browser):
    """The first build fitted the whole column into the window, and every
    infotext took its height out of the gallery. Now the infotext is below the
    view, and the page scrolls to it."""
    page = open_page(browser)
    try:
        before = measure(page)
        page.evaluate("document.getElementById('html_info_txt2img').innerHTML += "
                      "'<p>Model: kroma-v0.3, Module 1: Qwen2D_VAE</p>'.repeat(12)")
        settle(page, 4)
        after = measure(page)
        assert abs(after["gallery"]["height"] - before["gallery"]["height"]) <= 1, after
        assert_centred(after)

        # Rendered, and reached by scrolling.
        page.evaluate("document.getElementById('html_log_txt2img')"
                      ".scrollIntoView({block: 'end'})")
        settle(page)
        shown = page.evaluate("document.getElementById('html_info_txt2img')"
                              ".getBoundingClientRect().top < innerHeight")
        assert shown is True
    finally:
        page.close()


PROGRESS = """(position) => {
    const container = document.getElementById("txt2img_gallery_container");
    const bar = document.createElement("div");
    bar.className = "progressDiv";
    bar.id = "progress-under-test";
    // Forge's own rule for the bar, when it is laid over the gallery.
    if (position) {
        bar.style.setProperty("position", position, "important");
        bar.style.setProperty("top", "-14px", "important");
        bar.style.setProperty("left", "0px");
        bar.style.setProperty("width", "100%");
    }
    const fill = document.createElement("div");
    fill.className = "progress";
    bar.appendChild(fill);
    container.parentNode.insertBefore(bar, container);
}"""


def test_a_progress_bar_takes_its_room_from_the_gallery(browser):
    """Forge puts its progress bar just before the gallery's container, and
    Lobe puts it in the column's flow: while a generation runs, Generate and
    the gallery move down by its height, and the buttons must not."""
    page = open_page(browser)
    try:
        before = measure(page)
        page.evaluate(PROGRESS, "")
        settle(page, 4)
        running = measure(page)
        assert abs(running["buttons"]["bottom"] - before["buttons"]["bottom"]) <= 1, running
        assert running["gallery"]["height"] < before["gallery"]["height"] - 19
        assert_centred(running)

        page.evaluate("document.getElementById('progress-under-test').remove()")
        settle(page, 4)
        done = measure(page)
        assert abs(done["gallery"]["height"] - before["gallery"]["height"]) <= 1, done

        # Forge's own bar, laid over the gallery, moves nothing -- also when
        # something else has the gallery measured while it is up.
        page.evaluate(PROGRESS, "absolute")
        settle(page)
        page.evaluate("forgeAssistant.fill.now()")
        settle(page)
        overlaid = measure(page)
        assert abs(overlaid["gallery"]["height"] - before["gallery"]["height"]) <= 1, overlaid
    finally:
        page.close()


def test_the_gallery_follows_the_window(browser):
    page = open_page(browser)
    try:
        page.set_viewport_size({"width": 1400, "height": 820})
        settle(page, 4)
        found = measure(page)
        assert found["inner"] == 820
        assert_centred(found)
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
    // Docked, the text box itself takes no press until a tap engages it (the
    // guard); the press still has to land on its block in the column, and
    // not on the panel drawn over it.
    const block = prompt.closest(".block");
    const row = column.parentElement;
    const results = document.getElementById("txt2img_results");
    return {column: box(column), slot: box(slot),
            scroll: [column.scrollHeight, column.clientHeight],
            squeezed: [...column.querySelectorAll(".block[id^='setting_']")]
                .filter((node) => Math.round(node.getBoundingClientRect().height) !== 90).length,
            pressed: !!hit && (hit === prompt || block.contains(hit)),
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
        assert_centred(measure(page))

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
            shown: getComputedStyle(column).visibility === "visible",
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

        # Another workspace: the button goes and the column is held out of
        # sight -- not given back to its page; back on Txt2Img, it shows in
        # the panel again.
        page.evaluate(SWITCH, "tab_img2img")
        page.wait_for_function("document.querySelector('.forge-assistant-settings').hidden")
        settle(page)
        away = page.evaluate(WHERE)
        assert away["docked"] is True and away["shown"] is False, away
        page.evaluate(SWITCH, "tab_txt2img")
        page.wait_for_function("!document.querySelector('.forge-assistant-settings').hidden")
        settle(page)
        back = page.evaluate(WHERE)
        assert back["shown"] is True and back["column"] == back["slot"], back
    finally:
        page.close()


def press_three_times(page, selector) -> None:
    box = page.evaluate("(s) => { const r = document.querySelector(s).getBoundingClientRect();"
                        " return [r.left + r.width / 2, r.top + r.height / 2]; }", selector)
    for _ in range(3):
        page.mouse.click(*box)
        page.wait_for_timeout(60)
    settle(page, 4)


ROW = """() => {
    const column = document.getElementById("txt2img_settings");
    const results = document.getElementById("txt2img_results");
    return {docked: column.classList.contains("forge-assistant-docked-column"),
            shown: getComputedStyle(column).visibility === "visible",
            full: Math.abs(results.getBoundingClientRect().width
                           - results.parentElement.getBoundingClientRect().width) <= 1,
            beside: column.getBoundingClientRect().right
                <= results.getBoundingClientRect().left + 1,
            focus: forgeAssistant.focus.isActive(),
            open: forgeAssistant.shell.state.panelOpen};
}"""


def test_the_docked_column_stays_out_of_the_page_until_chat_or_the_tab_bar(browser):
    """Reported: the gallery "returns to the left column when i exit focus
    mode. if i exit focus mode with the column enabled inside the flyout menu,
    it should remain hidden when focus mode exit. the only way to get it back
    is open the fly out and switch to conversation or tab mode"."""
    page = open_page(browser, 1600, 900)
    try:
        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        page.click(".forge-assistant-settings")
        settle(page)
        # Focus, from the header; the panel put away for the whole gallery;
        # focus ended from the launcher.
        press_three_times(page, ".forge-assistant-grip")
        assert page.evaluate(ROW)["focus"] is True
        page.click(".forge-assistant-header [aria-label='Minimize the assistant']")
        settle(page)
        closed = page.evaluate(ROW)
        assert closed["docked"] is True and closed["shown"] is False, closed
        assert closed["full"] is True, "the gallery keeps the row"
        press_three_times(page, ".forge-assistant-launcher")
        left = page.evaluate(ROW)
        assert left["focus"] is False and left["open"] is False, left
        assert left["docked"] is True and left["shown"] is False, left
        assert left["full"] is True, left
        assert_centred(measure(page))

        # Opened, it is in the panel again.
        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        opened = page.evaluate(WHERE)
        assert opened["shown"] is True and opened["column"] == opened["slot"], opened

        # Chat is a way back: the column is in its page beside the gallery.
        page.click(".forge-assistant-chat")
        settle(page)
        back = page.evaluate(ROW)
        assert back["docked"] is False and back["beside"] is True and back["full"] is False, back
    finally:
        page.close()


# -- the docked view's height ------------------------------------------------- #

def test_the_docked_view_is_as_tall_as_the_conversation_at_its_fullest(browser):
    """Asked for: "maintain the same height restriction as the fly out menu in
    conversation mode. It should not go to the top and bottom of the page ...
    the entire column's worth of content should be available still. Just need
    to scroll it." """
    page = open_page(browser, 1600, 900)
    try:
        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        # The conversation at its fullest: the transcript as tall as the
        # stylesheet lets it be.
        chat = page.evaluate("""() => {
            const transcript = document.querySelector(".forge-assistant-transcript");
            transcript.style.height = getComputedStyle(transcript).maxHeight;
            const height = document.getElementById("forge-assistant-panel").offsetHeight;
            transcript.style.height = "";
            return height;
        }""")
        page.click(".forge-assistant-settings")
        settle(page)
        docked = page.evaluate("""() => {
            const panel = document.getElementById("forge-assistant-panel");
            const column = document.getElementById("txt2img_settings");
            const slot = document.querySelector(".forge-assistant-dock");
            const box = (node) => { const r = node.getBoundingClientRect();
                                    return [Math.round(r.left), Math.round(r.top),
                                            Math.round(r.width), Math.round(r.height)]; };
            column.scrollTop = column.scrollHeight;
            return {panel: panel.offsetHeight, column: box(column), slot: box(slot),
                    scroll: [column.scrollHeight, column.clientHeight],
                    reached: Math.round(column.scrollTop + column.clientHeight)
                        >= column.scrollHeight - 1,
                    last: column.lastElementChild.getBoundingClientRect().bottom
                        <= column.getBoundingClientRect().bottom + 1};
        }""")
        assert abs(docked["panel"] - chat) <= 1, (docked, chat)
        assert docked["panel"] < 900 / 2, "not the window's height"
        assert docked["column"] == docked["slot"], docked
        assert docked["scroll"][0] > docked["scroll"][1] * 3, "the column scrolls"
        assert docked["reached"] and docked["last"], "all of it can be scrolled to"
    finally:
        page.close()


# -- a phone ------------------------------------------------------------------- #

PHONE = {"width": 390, "height": 844}


def open_phone(browser, free_float: bool):
    context = browser.new_context(viewport=PHONE, is_mobile=True, has_touch=True,
                                  device_scale_factor=2)
    page = context.new_page()
    html = page_html()
    page.route(ORIGIN + "/**", lambda route: _answer(route, html))
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!(window.forgeAssistant && window.forgeAssistant.shell)")
    page.evaluate("(on) => forgeAssistant.shell.setFreeFloat(on)", free_float)
    page.evaluate("forgeAssistant.shell.open()")
    settle(page)
    return context, page


def touch_drag(page, start, end, steps: int = 8) -> None:
    """A finger, through the browser's own touch input rather than a mouse
    standing in for one, at a hand's pace: a frame between moves and a rest
    before it lifts. Moves sent all at once are a flick to the browser, and
    the tap after a flick is the one that stops it, which the browser keeps
    for itself -- a test's artefact, not a page's behaviour."""
    cdp = page.context.new_cdp_session(page)
    cdp.send("Input.dispatchTouchEvent",
             {"type": "touchStart", "touchPoints": [{"x": start[0], "y": start[1]}]})
    for step in range(1, steps + 1):
        x = start[0] + (end[0] - start[0]) * step / steps
        y = start[1] + (end[1] - start[1]) * step / steps
        page.wait_for_timeout(16)
        cdp.send("Input.dispatchTouchEvent",
                 {"type": "touchMove", "touchPoints": [{"x": x, "y": y}]})
    page.wait_for_timeout(120)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    cdp.detach()


PANEL_BOX = """() => {
    const panel = document.getElementById("forge-assistant-panel");
    const r = panel.getBoundingClientRect();
    const grip = document.querySelector(".forge-assistant-grip").getBoundingClientRect();
    return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, height: r.height,
            sheet: panel.classList.contains("forge-assistant-sheet"),
            modal: panel.getAttribute("aria-modal"), role: panel.getAttribute("role"),
            grip: [grip.left + grip.width / 2, grip.top + grip.height / 2],
            at: forgeAssistant.shell.state.floatAt,
            inner: [innerWidth, innerHeight]};
}"""


def test_on_a_phone_free_float_floats_and_stays_where_it_is_put(browser):
    """Reported: "on mobile, the fly-out menu is docking instead of being able
    to move anywhere in free float"."""
    context, page = open_phone(browser, free_float=True)
    try:
        before = page.evaluate(PANEL_BOX)
        assert before["sheet"] is False and before["modal"] is None, before
        assert before["role"] == "complementary"
        assert before["left"] >= 16 - 1 and before["right"] <= PHONE["width"] - 16 + 1, before

        # Up the screen by a finger on the header, and let go.
        grip = before["grip"]
        touch_drag(page, grip, (grip[0], grip[1] - 300))
        settle(page)
        after = page.evaluate(PANEL_BOX)
        assert after["sheet"] is False, after
        assert abs(after["top"] - (before["top"] - 300)) <= 1, (before, after)
        assert after["at"] is not None

        # And down again, part way: where it is let go is where it stays.
        grip = after["grip"]
        touch_drag(page, grip, (grip[0], grip[1] + 120))
        settle(page)
        again = page.evaluate(PANEL_BOX)
        assert abs(again["top"] - (after["top"] + 120)) <= 1, (after, again)

        # Reopened, it comes back where it was put.
        page.evaluate("forgeAssistant.shell.close()")
        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        reopened = page.evaluate(PANEL_BOX)
        assert abs(reopened["top"] - again["top"]) <= 1, (again, reopened)
    finally:
        context.close()


def keyboard(page, scale: float) -> None:
    """Less of the page showing than the layout has: what the keyboard coming
    up does on a phone, where only the visible viewport shrinks and the
    stylesheet's `100vh` stays the whole screen. Pinch zoom is the same shrink
    of the visible viewport, and the one a browser can be asked for."""
    cdp = page.context.new_cdp_session(page)
    cdp.send("Emulation.setPageScaleFactor", {"pageScaleFactor": scale})
    cdp.detach()
    settle(page, 4)


VISIBLE = """() => {
    const view = window.visualViewport;
    const box = (selector) => { const r = document.querySelector(selector).getBoundingClientRect();
                                return {top: r.top, bottom: r.bottom}; };
    return {top: view.offsetTop, bottom: view.offsetTop + view.height,
            panel: box("#forge-assistant-panel"), composer: box(".forge-assistant-composer")};
}"""


def test_on_a_phone_a_floating_panel_fits_what_shows(browser):
    """The keyboard coming up leaves less of the window showing; a floating
    panel taller than that ran under the keyboard, composer and all."""
    context, page = open_phone(browser, free_float=True)
    try:
        grip = page.evaluate(PANEL_BOX)["grip"]
        touch_drag(page, grip, (grip[0], 700))
        settle(page)
        keyboard(page, 2.2)
        found = page.evaluate(VISIBLE)
        assert found["bottom"] - found["top"] < 400, found
        assert found["panel"]["top"] >= found["top"] + 16 - 1, found
        assert found["panel"]["bottom"] <= found["bottom"] - 16 + 1, found
        assert found["composer"]["bottom"] > found["composer"]["top"], "the composer is drawn"
        assert found["composer"]["bottom"] <= found["bottom"] - 16 + 1, found

        # Docked, the column's height is what gives way, and the column is
        # still drawn over the panel's placeholder, inside what shows.
        page.evaluate("forgeAssistant.shell.toggleDock()")
        settle(page, 4)
        docked = page.evaluate(VISIBLE)
        where = page.evaluate(WHERE)
        column = page.evaluate("document.getElementById('txt2img_settings')"
                               ".getBoundingClientRect().bottom")
        assert docked["panel"]["bottom"] <= docked["bottom"] - 16 + 1, docked
        assert where["column"] == where["slot"], where
        assert column <= docked["bottom"] - 16 + 1, (column, docked)
    finally:
        context.close()


def test_on_a_phone_without_free_float_the_panel_is_still_a_sheet(browser):
    context, page = open_phone(browser, free_float=False)
    try:
        found = page.evaluate(PANEL_BOX)
        assert found["sheet"] is True and found["modal"] == "true", found
        assert found["left"] == 0 and found["right"] == PHONE["width"]
    finally:
        context.close()


def test_on_a_phone_the_settings_column_docks_in_a_floating_panel(browser):
    context, page = open_phone(browser, free_float=True)
    try:
        # Put mid-screen first: docking makes the panel taller, and its header
        # -- with the button just pressed in it -- stays where it was put.
        grip = page.evaluate(PANEL_BOX)["grip"]
        touch_drag(page, grip, (grip[0], 200))
        settle(page)
        put = page.evaluate(PANEL_BOX)
        page.tap(".forge-assistant-settings")
        settle(page)
        docked = page.evaluate(PANEL_BOX)
        assert docked["height"] > put["height"] + 100, (put, docked)
        assert abs(docked["top"] - put["top"]) <= 1, (put, docked)
        page.tap(".forge-assistant-chat")
        settle(page)
        back = page.evaluate(PANEL_BOX)
        assert abs(back["top"] - put["top"]) <= 1, (put, back)
        page.tap(".forge-assistant-settings")
        settle(page)
        found = page.evaluate(DOCKED)
        assert found["pressedButton"] == "true"
        assert found["column"] == found["slot"], found
        assert found["scroll"][0] > found["scroll"][1], "the column scrolls"
        assert found["pressed"] is True
        grip = page.evaluate(PANEL_BOX)["grip"]
        touch_drag(page, grip, (grip[0], grip[1] + 150))
        settle(page)
        moved = page.evaluate(WHERE)
        assert moved["column"] == moved["slot"], moved
    finally:
        context.close()


# The guard: docked, a field takes a drag only once a tap has engaged it.

FIELDS = """() => {
    const column = document.getElementById("txt2img_settings");
    const html = `
      <div class="block gradio-slider" id="t_slider" style="height: auto">
        <label for="t_number">Steps</label>
        <input id="t_number" type="number" value="20">
        <input id="t_range" type="range" min="0" max="100" value="20"
               style="display: block; width: 90%">
      </div>
      <div class="block gradio-dropdown" id="t_drop" style="height: auto">
        <input id="t_choice" value="Euler">
      </div>
      <div class="block gradio-textbox" id="t_text" style="height: auto">
        <label>Literal</label>
        <textarea id="t_area" rows="3">${"line\\n".repeat(40)}</textarea>
      </div>`;
    column.insertAdjacentHTML("afterbegin", html);
    column.scrollTop = 0;
}"""

HIT = """(id) => {
    // The middle of what shows of it: a docked text box is as tall as its
    // text, and may run past the column's bottom.
    const node = document.getElementById(id);
    const r = node.getBoundingClientRect();
    const column = document.getElementById("txt2img_settings").getBoundingClientRect();
    const top = Math.max(r.top, column.top, 0);
    const bottom = Math.min(r.bottom, column.bottom, innerHeight);
    const x = r.left + r.width / 2, y = (top + bottom) / 2;
    const hit = document.elementFromPoint(x, y);
    return {self: hit === node, inBlock: !!hit && !!node.closest(".block").contains(hit),
            centre: [x, y]};
}"""

STATE = """() => {
    const engaged = [...document.querySelectorAll(".forge-assistant-engaged")].map((n) => n.id);
    const column = document.getElementById("txt2img_settings");
    const slider = document.getElementById("t_slider");
    return {engaged, active: document.activeElement && document.activeElement.id,
            guarded: column.classList.contains("forge-assistant-guarded"),
            range: document.getElementById("t_range").value,
            area: document.getElementById("t_area").scrollTop,
            column: column.scrollTop,
            outline: getComputedStyle(slider).outlineStyle + " "
                + getComputedStyle(slider).outlineWidth};
}"""


FULL = """() => {
    const area = document.getElementById("t_area");
    return {scrollHeight: area.scrollHeight, clientHeight: area.clientHeight};
}"""

# Which line the caret is on, and which line is under (x, y).
CARET = """([x, y]) => {
    const area = document.getElementById("t_area");
    const r = area.getBoundingClientRect();
    const cs = getComputedStyle(area);
    const lines = area.value.split("\\n").length;
    const inner = area.scrollHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    const lineHeight = inner / lines;
    const under = Math.floor((y - r.top - parseFloat(cs.borderTopWidth)
                              - parseFloat(cs.paddingTop)) / lineHeight);
    const line = area.value.slice(0, area.selectionStart).split("\\n").length - 1;
    return {line, under, selection: area.selectionStart};
}"""


def docked_with_fields(page):
    page.evaluate("forgeAssistant.shell.open()")
    settle(page)
    page.click(".forge-assistant-settings")
    settle(page)
    page.evaluate(FIELDS)
    settle(page)


def test_a_docked_field_takes_no_press_until_a_tap_engages_it(browser):
    page = open_page(browser, 1600, 900)
    try:
        docked_with_fields(page)
        state = page.evaluate(STATE)
        assert state["guarded"] is True and state["engaged"] == [], state

        # Nothing engaged: the slider, its number and the text box give their
        # presses to their blocks; a dropdown keeps its own.
        for name in ("t_range", "t_number", "t_area"):
            hit = page.evaluate(HIT, name)
            assert hit["self"] is False and hit["inBlock"] is True, (name, hit)
        assert page.evaluate(HIT, "t_choice")["self"] is True

        # A drag across the slider moves nothing.
        x, y = page.evaluate(HIT, "t_range")["centre"]
        page.mouse.move(x - 100, y)
        page.mouse.down()
        page.mouse.move(x + 150, y, steps=6)
        page.mouse.up()
        assert page.evaluate(STATE)["range"] == "20"

        # The text box shows all of its text: nothing in it to scroll.
        full = page.evaluate(FULL)
        assert full["scrollHeight"] <= full["clientHeight"] + 1, full
        assert full["clientHeight"] > 400, full

        # The wheel over the text box scrolls the column, not the text.
        x, y = page.evaluate(HIT, "t_area")["centre"]
        page.mouse.move(x, y)
        page.mouse.wheel(0, 120)
        settle(page, 6)
        page.wait_for_timeout(200)
        scrolled = page.evaluate(STATE)
        assert scrolled["area"] == 0 and scrolled["column"] > 0, scrolled
        page.evaluate("document.getElementById('txt2img_settings').scrollTop = 0")
        settle(page)

        # A tap engages the slider: outlined, its presses back, and a drag
        # now slides it. No keyboard: the track was tapped, not the number.
        x, y = page.evaluate(HIT, "t_range")["centre"]
        page.mouse.click(x, y)
        engaged = page.evaluate(STATE)
        assert engaged["engaged"] == ["t_slider"], engaged
        assert engaged["outline"] == "solid 2px", engaged
        assert engaged["active"] != "t_number", engaged
        assert page.evaluate(HIT, "t_range")["self"] is True
        page.mouse.move(x - 100, y)
        page.mouse.down()
        page.mouse.move(x + 150, y, steps=6)
        page.mouse.up()
        assert page.evaluate(STATE)["range"] != "20"

        # A tap on the text box moves the engagement there and focuses it with
        # the caret on the line that was tapped -- and the column stays where
        # it was, so what was on screen is still on screen.
        page.evaluate("document.getElementById('txt2img_settings').scrollTop = 120")
        settle(page)
        x, y = page.evaluate(HIT, "t_area")["centre"]
        page.mouse.click(x, y)
        typed = page.evaluate(STATE)
        assert typed["engaged"] == ["t_text"] and typed["active"] == "t_area", typed
        assert typed["column"] == 120, "engaging a text box does not scroll the column"
        caret = page.evaluate(CARET, [x, y])
        assert caret["line"] > 0 and abs(caret["line"] - caret["under"]) <= 1, caret
        page.keyboard.type("!")
        settle(page)
        after = page.evaluate(STATE)
        assert after["column"] == 120, "typing does not scroll the column either"

        # More lines grow the box rather than scrolling inside it.
        page.keyboard.type("\n\n\n\n\n")
        settle(page)
        grown = page.evaluate(FULL)
        assert grown["clientHeight"] > full["clientHeight"], (full, grown)
        assert grown["scrollHeight"] <= grown["clientHeight"] + 1, grown

        # Nothing but the reader lets it go: a script pressing a button
        # elsewhere on the page, and the panel being placed again with its
        # slot empty for a moment, leave it engaged and focused.
        page.evaluate("""() => {
            const button = document.createElement("button");
            document.body.appendChild(button);
            button.click();
            forgeAssistant.dock.place({left: 0, top: 0, width: 0, height: 0});
        }""")
        page.evaluate("forgeAssistant.shell.syncDock()")
        settle(page)
        kept = page.evaluate(STATE)
        assert kept["engaged"] == ["t_text"] and kept["active"] == "t_area", kept

        # A tap outside it lets go, and takes the keyboard with it.
        page.mouse.click(*page.evaluate(HIT, "setting_3")["centre"])
        released = page.evaluate(STATE)
        assert released["engaged"] == [] and released["active"] != "t_area", released

        # Escape lets go too.
        page.mouse.click(*page.evaluate(HIT, "t_area")["centre"])
        assert page.evaluate(STATE)["engaged"] == ["t_text"]
        page.keyboard.press("Escape")
        assert page.evaluate(STATE)["engaged"] == []

        # Back in the page, the fields are ordinary fields, with the heights
        # they had.
        page.click(".forge-assistant-settings")
        settle(page)
        back = page.evaluate(STATE)
        assert back["guarded"] is False, back
        assert page.evaluate(HIT, "t_range")["self"] is True
        assert page.evaluate("document.getElementById('t_area').style.height") == ""
    finally:
        page.close()


def test_on_a_phone_a_drag_on_a_docked_text_box_scrolls_the_column(browser):
    context, page = open_phone(browser, free_float=False)
    try:
        page.tap(".forge-assistant-settings")
        settle(page)
        page.evaluate(FIELDS)
        settle(page)
        x, y = page.evaluate(HIT, "t_area")["centre"]
        touch_drag(page, (x, y), (x, y - 120))
        settle(page, 4)
        dragged = page.evaluate(STATE)
        assert dragged["area"] == 0 and dragged["column"] > 40, dragged

        page.evaluate("document.getElementById('txt2img_settings').scrollTop = 0")
        settle(page)
        x, y = page.evaluate(HIT, "t_area")["centre"]
        page.touchscreen.tap(x, y)
        settle(page)
        tapped = page.evaluate(STATE)
        assert tapped["engaged"] == ["t_text"] and tapped["active"] == "t_area", tapped
    finally:
        context.close()
