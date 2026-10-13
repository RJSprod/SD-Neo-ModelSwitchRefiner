"""The theme's page work, rendered by a real browser.

Node has no layout, and the header's fit, the drawer's place and the sticky
bar are layout. This renders a Forge-shaped page -- Gradio's `#tabs` with its
tab bar and fourteen tabs, the classic top row with its prompts and its
actions column, the settings and results columns inside Forge's extra-network
tabs, the quick settings row above the tabs, the footer -- with the real
script and stylesheet, Forge's `gradioApp()` and its mount hooks stood in for,
and measures what the theme does to it:

    the bar       sticky at the top, 64 px tall, the title first and the
                  actions last, Forge's tab buttons still its direct children;
    the overflow  tabs that do not fit are hidden and listed under the
                  ellipsis, the selected one never; a menu entry presses the
                  real button; at a wide window nothing is hidden;
    the split     the top row heads the settings column and Generate's box
                  heads the gallery's container, once, however often applied;
    the drawer    the quick settings live in a drawer under the bar, closed
                  until the header's button opens it, Escape closes it, and
                  the choice is remembered;
    the prompts   resizable;
    the icons     a button whose whole text is one of Forge's emoji is drawn,
                  one with words beside it is not;
    standing down a Lobe root on the page means nothing is touched;
    off           every switch off puts its element back where Forge had it,
                  and the whole theme off leaves nothing behind.

Needs the ``playwright`` package and a Chromium; skips without either.
"""

from __future__ import annotations

import glob
import json
import os
import urllib.parse
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_theme.js"
STYLE = ROOT / "style.css"
ORIGIN = "http://forge-theme.test"

TABS = [("tab_txt2img", "txt2img"), ("tab_img2img", "img2img"), ("tab_clipboard", "Clipboard"),
        ("tab_wangp", "WanGP"), ("tab_llm_studio", "LLM Studio"), ("tab_voice_box", "Voice Box"),
        ("tab_minipaint", "Mini Paint"), ("tab_pnginfo", "PNG Info"),
        ("tab_lora_details", "Extended LoRA Details"), ("tab_extras", "Extras"),
        ("tab_modelmerger", "Checkpoint Merger"), ("tab_train", "Train"),
        ("tab_settings", "Settings"), ("tab_extensions", "Extensions")]

# What Gradio 4.40 and Forge give the page, and no more: rows and columns,
# the page's padding, Forge's own tool-button and quick-settings rules.
HOST_CSS = """
*, ::before, ::after { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--body-background-fill); color: var(--body-text-color); font: 14px sans-serif;
       --body-background-fill: #000; --body-text-color: #e6e6e6; --body-text-color-subdued: #8c8c8c;
       --background-fill-secondary: #141414; --border-color-primary: #262626; --block-background-fill: #0d0d0d;
       --block-border-color: #262626; --button-secondary-background-fill: #1a1a1a;
       --button-secondary-border-color: #2b2b2b; --button-secondary-text-color: #d9d9d9;
       --button-cancel-background-fill: #d93025; --button-cancel-text-color: #fff; --button-cancel-border-color: #d93025;
       --color-accent: #fff; --primary-500: #fff; --font-mono: monospace; --size-4: 16px; --shadow-drop-lg: none; }
.app { padding: var(--size-4); }
.row { display: flex; flex-wrap: wrap; gap: 8px; }
.column { display: flex; flex-direction: column; gap: 8px; min-width: 0; flex: 1; }
.tabs { position: relative; }
.tab-nav { display: flex; flex-wrap: wrap; border-bottom: 1px solid var(--border-color-primary); }
.tab-nav button { padding: 4px 16px; border: 1px solid transparent; background: transparent; color: inherit; }
.tabitem[style*="display: none"] { display: none; }
.gradio-button.tool { max-width: 2.2em; min-width: 2.2em !important; height: 2.4em; }
#quicksettings > div { width: fit-content; flex: 0 1 fit-content; }
.block { min-height: 40px; }
textarea { width: 100%; min-height: 60px; }
"""


def page_html(*, opts: dict, lobe: bool = False, settings_blocks: int = 40) -> str:
    buttons = "".join(
        f'<button role="tab" id="{tab}-button" aria-controls="{tab}" '
        f'aria-selected="{"true" if i == 0 else "false"}" class="{"selected" if i == 0 else ""}">{label}</button>'
        for i, (tab, label) in enumerate(TABS))
    panels = "".join(
        f'<div id="{tab}" class="tabitem" role="tabpanel" style="display: none"></div>'
        for tab, _ in TABS[2:] if tab != "tab_settings")
    blocks = "".join(f'<div class="block" id="setting_{n}">Setting {n}</div>' for n in range(settings_blocks))
    lobe_root = ('<div id="root"><header id="lobe-header"><div role="tablist" class="ant-tabs-nav">'
                 '<div role="tab">Txt 2 Img</div></div></header></div>' if lobe else "")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{HOST_CSS}</style>
<link rel="stylesheet" href="/style.css">
</head><body class="dark">
<gradio-app>{lobe_root}<div class="gradio-container app">
<div id="quicksettings" class="row"><div class="block gradio-dropdown" id="setting_sd_model_checkpoint">Checkpoint</div><div class="block" id="setting_sd_vae">VAE</div></div>
<div id="tabs" class="tabs">
  <div class="tab-nav scroll-hide" role="tablist">{buttons}</div>
  <div id="tab_txt2img" class="tabitem" role="tabpanel" style="display: block">
    <div id="txt2img_toprow" class="row">
      <div id="txt2img_prompt_container" class="column">
        <div id="txt2img_prompt" class="block"><label><textarea aria-label="Prompt"></textarea></label></div>
        <div id="txt2img_neg_prompt" class="block"><label><textarea aria-label="Negative prompt"></textarea></label></div>
      </div>
      <div id="txt2img_actions_column" class="column">
        <div id="txt2img_generate_box" class="row generate-box">
          <button id="txt2img_interrupt">Interrupt</button><button id="txt2img_skip">Skip</button>
          <button id="txt2img_generate" class="lg primary gradio-button">Generate</button>
        </div>
        <div id="txt2img_tools" class="row">
          <button id="paste" class="gradio-button tool" title="Read generation parameters">↙️</button>
          <button id="txt2img_clear_prompt" class="gradio-button tool" title="Clear prompt">🗑️</button>
          <button id="txt2img_style_apply" class="gradio-button tool" title="Apply styles">📋</button>
          <button id="mixed" class="gradio-button">🗑️ Clear</button>
        </div>
      </div>
    </div>
    <div id="txt2img_extra_tabs" class="tabs extra-networks">
      <div class="tab-nav" role="tablist"><button role="tab" class="selected" aria-controls="txt2img_generation">Generation</button><button role="tab" aria-controls="txt2img_lora">Lora</button></div>
      <div id="txt2img_generation" class="tabitem" role="tabpanel" style="display: block">
        <div class="row">
          <div id="txt2img_settings" class="column">{blocks}</div>
          <div id="txt2img_results" class="column">
            <div id="txt2img_results_panel" class="column panel">
              <div id="txt2img_gallery_container" class="gr-group">
                <div id="txt2img_gallery" class="block gradio-gallery" style="height: 300px"></div>
              </div>
              <div id="image_buttons_txt2img" class="row image-buttons">
                <button id="txt2img_open_folder" class="gradio-button tool" title="Open images output directory">📂</button>
                <button id="save_txt2img" class="gradio-button tool" title="Save the image">💾</button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div id="txt2img_lora" class="tabitem" role="tabpanel" style="display: none"></div>
    </div>
  </div>
  <div id="tab_img2img" class="tabitem" role="tabpanel" style="display: none">
    <div id="img2img_toprow" class="row"><div id="img2img_prompt" class="block"></div></div>
    <div id="img2img_settings" class="column"><div class="block">Resize</div></div>
    <div id="img2img_results" class="column"><div id="img2img_generate_box" class="row"><button id="img2img_generate">Generate</button></div><div id="img2img_results_panel" class="column panel"><div id="img2img_gallery_container" class="gr-group"><div id="img2img_gallery" class="block"></div></div></div></div>
  </div>
  <div id="tab_settings" class="tabitem" role="tabpanel" style="display: none">
    <div id="settings" class="tabs"><div class="tab-nav" role="tablist"><button role="tab">Model Chain</button><button role="tab" id="theme-section">Model Chain: Theme</button></div></div>
  </div>
  {panels}
</div>
<div id="footer">footer</div>
</div></gradio-app>
<script>
function gradioApp() {{
    const elem = document.querySelector("gradio-app");
    elem.getElementById = function (id) {{ return document.getElementById(id); }};
    return elem;
}}
var opts = {json.dumps(opts)};
const hooks = {{loaded: [], changed: [], after: [], tab: []}};
function onUiLoaded(fn) {{ hooks.loaded.push(fn); }}
function onOptionsChanged(fn) {{ hooks.changed.push(fn); }}
function onAfterUiUpdate(fn) {{ hooks.after.push(fn); }}
function onUiTabChange(fn) {{ hooks.tab.push(fn); }}
window.pressed = [];
document.querySelectorAll('#tabs > .tab-nav > button').forEach((b) => b.addEventListener('click', () => window.pressed.push(b.id)));
</script>
<script src="/javascript/model_chain_theme.js"></script>
</body></html>"""


def _answer(route, html: str) -> None:
    path = urllib.parse.urlsplit(route.request.url).path
    if path == "/":
        route.fulfill(status=200, content_type="text/html", body=html)
    elif path == "/style.css":
        route.fulfill(status=200, content_type="text/css", body=STYLE.read_text(encoding="utf-8"))
    elif path == "/javascript/model_chain_theme.js":
        route.fulfill(status=200, content_type="text/javascript", body=SCRIPT.read_text(encoding="utf-8"))
    else:
        route.fulfill(status=404, body="")


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


def settle(page, frames: int = 3) -> None:
    page.evaluate("(n) => new Promise((done) => { const step = (k) => k ? "
                  "requestAnimationFrame(() => step(k - 1)) : done(); step(n); })", frames)


ON = {"model_chain_theme": True}


def open_page(browser, opts=None, *, width=1600, height=900, lobe=False, loaded=True, context=None):
    """A page of its own context, or one in a context given -- two pages of
    one context share a localStorage, which is what the first paint reads."""
    owner = context or browser.new_context(viewport={"width": width, "height": height})
    page = owner.new_page()
    html = page_html(opts=dict(ON, **(opts or {})), lobe=lobe)
    page.route(ORIGIN + "/**", lambda route: _answer(route, html))
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!window.mcTheme")
    if loaded:
        page.evaluate("hooks.loaded.forEach((fn) => fn())")
        settle(page)
    return page


def changed(page):
    page.evaluate("hooks.changed.forEach((fn) => fn())")
    settle(page)


MEASURE = """() => {
    const q = (s) => document.querySelector(s);
    const nav = q("#tabs > .tab-nav");
    const style = getComputedStyle(nav);
    const rect = nav.getBoundingClientRect();
    const buttons = [...nav.querySelectorAll(":scope > button[role='tab']")];
    const hidden = buttons.filter((b) => b.classList.contains("mc-theme-overflow")).map((b) => b.textContent);
    const more = q(".mc-theme-more");
    const menu = q(".mc-theme-menu");
    return {
        attribute: document.documentElement.getAttribute("data-mc-theme"),
        prompts: document.documentElement.getAttribute("data-mc-theme-prompts"),
        position: style.position, top: rect.top, height: rect.height,
        first: nav.firstElementChild && nav.firstElementChild.className,
        last: nav.lastElementChild && nav.lastElementChild.className,
        title: q(".mc-theme-title") && q(".mc-theme-title").textContent,
        directTabs: buttons.length,
        hidden, moreHidden: more ? more.hidden : null,
        menuHidden: menu ? menu.hidden : null,
        menuItems: menu ? [...menu.querySelectorAll(".mc-theme-menu-item")].map((b) => b.textContent) : [],
        selectedHidden: buttons.some((b) => b.classList.contains("selected") && b.classList.contains("mc-theme-overflow")),
        settingsFirst: q("#txt2img_settings").firstElementChild.id,
        galleryFirst: q("#txt2img_gallery_container").firstElementChild.id,
        img2imgGalleryFirst: q("#img2img_gallery_container").firstElementChild.id,
        img2imgSettingsFirst: q("#img2img_settings").firstElementChild.id,
        toprowParent: q("#txt2img_toprow").parentElement.id,
        boxParent: q("#txt2img_generate_box").parentElement.id,
        quickParent: q("#quicksettings").parentElement.id || q("#quicksettings").parentElement.className,
        drawer: q("#mc-theme-quick") ? {hidden: q("#mc-theme-quick").hidden,
                                       position: getComputedStyle(q("#mc-theme-quick")).position,
                                       rect: (() => { const r = q("#mc-theme-quick").getBoundingClientRect();
                                                      return {top: r.top, left: r.left, width: r.width}; })()} : null,
        resize: getComputedStyle(q("#txt2img_prompt textarea")).resize,
        clearHtml: q("#txt2img_clear_prompt").innerHTML,
        clearLabel: q("#txt2img_clear_prompt").getAttribute("aria-label"),
        pasteHasSvg: !!q("#paste svg"),
        folderHasSvg: !!q("#txt2img_open_folder svg"),
        mixedText: q("#mixed").textContent,
        txt2imgLabel: q("#tab_txt2img-button").textContent,
        modeLabel: q(".mc-theme-mode") && q(".mc-theme-mode").getAttribute("aria-label"),
        scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth,
        generateHeight: q("#txt2img_generate").getBoundingClientRect().height,
        toolWidth: q("#paste").getBoundingClientRect().width,
    };
}"""


class TestTheBar:
    def test_forges_tab_bar_is_a_sticky_header_with_the_title_first_and_the_actions_last(self, browser):
        page = open_page(browser)
        found = page.evaluate(MEASURE)

        assert found["attribute"] == "on"
        assert found["position"] == "sticky"
        assert found["height"] == 64
        assert found["first"] == "mc-theme-title"
        assert found["last"] == "mc-theme-actions"
        assert found["title"] == "Forge"
        assert found["directTabs"] == len(TABS)
        page.close()

    def test_it_stays_at_the_top_while_the_page_scrolls(self, browser):
        page = open_page(browser)
        page.evaluate("window.scrollTo(0, 900)")
        settle(page)
        found = page.evaluate(MEASURE)

        assert page.evaluate("window.scrollY") > 0
        assert abs(found["top"]) < 1
        page.close()

    def test_tabs_that_do_not_fit_go_under_the_ellipsis_and_the_selected_one_never(self, browser):
        page = open_page(browser, width=820)
        found = page.evaluate(MEASURE)

        assert found["hidden"], "fourteen tabs do not fit in 820 px"
        assert found["moreHidden"] is False
        assert found["menuHidden"] is True
        assert found["selectedHidden"] is False
        assert found["menuItems"] == found["hidden"]
        assert found["directTabs"] == len(TABS)
        page.close()

    def test_a_menu_entry_presses_the_real_tab_button(self, browser):
        page = open_page(browser, width=820)
        page.click(".mc-theme-more")
        settle(page)
        assert page.evaluate("document.querySelector('.mc-theme-menu').hidden") is False
        label = page.evaluate("document.querySelector('.mc-theme-menu-item').textContent")
        page.click(".mc-theme-menu-item")
        settle(page)

        pressed = page.evaluate("window.pressed")
        expected = next(tab for tab, text in TABS if text == label) + "-button"
        assert pressed == [expected]
        assert page.evaluate("document.querySelector('.mc-theme-menu').hidden") is True
        page.close()

    def test_a_wide_window_shows_every_tab_and_no_ellipsis(self, browser):
        page = open_page(browser, width=2000)
        found = page.evaluate(MEASURE)

        assert found["hidden"] == []
        assert found["moreHidden"] is True
        page.close()

    def test_the_fit_follows_the_window(self, browser):
        page = open_page(browser, width=2000)
        assert page.evaluate(MEASURE)["hidden"] == []
        page.set_viewport_size({"width": 820, "height": 900})
        settle(page, 6)

        assert page.evaluate(MEASURE)["hidden"]
        page.close()

    def test_the_theme_toggle_offers_the_other_mode(self, browser):
        page = open_page(browser)
        found = page.evaluate(MEASURE)

        assert "light" in found["modeLabel"]
        assert page.evaluate("window.mcTheme.nextThemeUrl(location.href, true)").endswith("__theme=light")
        page.close()

    def test_the_tab_names_can_be_spaced(self, browser):
        page = open_page(browser, {"model_chain_theme_tab_labels": "spaced"})

        assert page.evaluate(MEASURE)["txt2imgLabel"] == "Txt 2 Img"
        page.evaluate("opts.model_chain_theme_tab_labels = 'as_forge'")
        changed(page)
        assert page.evaluate(MEASURE)["txt2imgLabel"] == "txt2img"
        page.close()


class TestTheSplitView:
    def test_the_top_row_heads_the_settings_column_and_generate_heads_the_gallery(self, browser):
        page = open_page(browser)
        found = page.evaluate(MEASURE)

        assert found["settingsFirst"] == "txt2img_toprow"
        assert found["galleryFirst"] == "txt2img_generate_box"
        assert found["img2imgSettingsFirst"] == "img2img_toprow"
        assert found["img2imgGalleryFirst"] == "img2img_generate_box"
        assert found["generateHeight"] >= 44
        page.close()

    def test_applying_again_moves_nothing_twice(self, browser):
        page = open_page(browser)
        before = page.evaluate("document.querySelector('#txt2img_settings').childElementCount")
        page.evaluate("hooks.after.forEach((fn) => fn()); hooks.after.forEach((fn) => fn())")
        settle(page)

        assert page.evaluate("document.querySelector('#txt2img_settings').childElementCount") == before
        assert page.evaluate("document.querySelectorAll('[data-mc-theme-moved]').length") == 5
        page.close()

    def test_the_switch_off_puts_both_back_where_forge_had_them(self, browser):
        page = open_page(browser)
        page.evaluate("opts.model_chain_theme_split = false")
        changed(page)
        found = page.evaluate(MEASURE)

        assert found["toprowParent"] == "tab_txt2img"
        assert found["boxParent"] == "txt2img_actions_column"
        assert page.evaluate("document.querySelector('#tab_txt2img').firstElementChild.id") == "txt2img_toprow"
        assert page.evaluate("document.querySelector('#txt2img_actions_column').firstElementChild.id") == "txt2img_generate_box"
        # And the rest of the theme is still on.
        assert found["attribute"] == "on"
        assert found["first"] == "mc-theme-title"
        page.close()


class TestTheQuickSettingsDrawer:
    def test_the_quick_settings_live_in_a_closed_drawer_the_header_opens(self, browser):
        page = open_page(browser)
        found = page.evaluate(MEASURE)

        assert found["quickParent"] == "mc-theme-quick"
        assert found["drawer"]["hidden"] is True
        page.click(".mc-theme-quick-toggle")
        settle(page)
        found = page.evaluate(MEASURE)
        assert found["drawer"]["hidden"] is False
        assert found["drawer"]["position"] == "fixed"
        assert found["drawer"]["rect"] == {"top": 64, "left": 0, "width": 280}
        assert page.evaluate("document.querySelector('.mc-theme-quick-toggle').getAttribute('aria-expanded')") == "true"
        page.close()

    def test_escape_closes_it_and_the_choice_is_remembered(self, browser):
        page = open_page(browser)
        page.click(".mc-theme-quick-toggle")
        settle(page)
        page.keyboard.press("Escape")
        settle(page)

        assert page.evaluate(MEASURE)["drawer"]["hidden"] is True
        assert page.evaluate("localStorage.getItem('mc-theme-quick')") == "closed"
        page.close()

    def test_open_by_default_and_the_width_follow_the_settings(self, browser):
        page = open_page(browser, {"model_chain_theme_quick": "open", "model_chain_theme_quick_width": 340})
        found = page.evaluate(MEASURE)

        assert found["drawer"]["hidden"] is False
        assert found["drawer"]["rect"]["width"] == 340
        page.close()

    def test_off_leaves_the_row_where_forge_draws_it(self, browser):
        page = open_page(browser, {"model_chain_theme_quick": "off"})
        found = page.evaluate(MEASURE)

        assert found["quickParent"] == "gradio-container app"
        assert found["drawer"] is None
        assert page.evaluate("document.querySelector('.mc-theme-quick-toggle').hidden") is True
        page.close()


class TestThePromptsAndTheIcons:
    def test_the_prompt_boxes_are_resizable(self, browser):
        page = open_page(browser)

        assert page.evaluate(MEASURE)["resize"] == "vertical"
        page.evaluate("opts.model_chain_theme_prompts = false")
        changed(page)
        assert page.evaluate(MEASURE)["resize"] != "vertical"
        page.close()

    def test_a_button_whose_whole_text_is_an_emoji_is_drawn_and_one_with_words_is_not(self, browser):
        page = open_page(browser)
        found = page.evaluate(MEASURE)

        assert "<svg" in found["clearHtml"]
        assert found["clearLabel"] == "Clear prompt"
        assert found["pasteHasSvg"] and found["folderHasSvg"]
        assert found["mixedText"] == "🗑️ Clear"
        assert found["toolWidth"] == 36
        page.close()

    def test_the_icons_switch_off_gives_the_emoji_back(self, browser):
        page = open_page(browser)
        page.evaluate("opts.model_chain_theme_icons = false")
        changed(page)
        found = page.evaluate(MEASURE)

        assert found["clearHtml"] == "🗑️"
        assert found["pasteHasSvg"] is False
        assert found["attribute"] == "on"
        page.close()


class TestStandingDownAndOff:
    def test_a_lobe_root_on_the_page_means_nothing_is_touched(self, browser):
        page = open_page(browser, lobe=True)
        found = page.evaluate(MEASURE)

        assert found["attribute"] == "lobe"
        assert found["title"] is None
        assert found["toprowParent"] == "tab_txt2img"
        assert found["boxParent"] == "txt2img_actions_column"
        assert found["quickParent"] == "gradio-container app"
        assert "<svg" not in found["clearHtml"]
        assert found["position"] != "sticky"
        page.close()

    def test_the_whole_theme_off_leaves_nothing_behind(self, browser):
        page = open_page(browser)
        page.evaluate("opts.model_chain_theme = false")
        changed(page)
        found = page.evaluate(MEASURE)

        assert found["attribute"] is None
        assert found["prompts"] is None
        assert found["title"] is None
        assert found["drawer"] is None
        assert found["toprowParent"] == "tab_txt2img"
        assert found["boxParent"] == "txt2img_actions_column"
        assert found["quickParent"] == "gradio-container app"
        assert found["clearHtml"] == "🗑️"
        assert found["position"] != "sticky"
        assert page.evaluate("document.querySelectorAll('[data-mc-theme-moved], [data-mc-theme-icon]').length") == 0
        page.close()

    def test_the_last_configuration_marks_the_root_before_the_host_is_up(self, browser):
        """The same browser, a later load: the bar is a bar before the host
        has published a single setting, from what was remembered."""
        context = browser.new_context(viewport={"width": 1600, "height": 900})
        page = open_page(browser, context=context)
        page.close()
        again = open_page(browser, loaded=False, context=context)
        found = again.evaluate(MEASURE)

        assert found["attribute"] == "on"
        assert found["position"] == "sticky"
        assert found["prompts"] == "resizable"
        context.close()


class TestAtPhoneWidth:
    def test_nothing_scrolls_sideways_and_the_bar_is_shorter(self, browser):
        page = open_page(browser, width=400, height=800)
        found = page.evaluate(MEASURE)

        assert found["scrollWidth"] <= found["innerWidth"]
        assert found["height"] == 56
        assert found["hidden"]
        page.close()
