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
    the networks  Forge's extra-network pages leave its tabs for a drawer on
                  the right once Forge's own script has wired them, with
                  Generation staying the selected tab; the drawer's strip
                  shows one page and its controls; the drawer follows the
                  main tab; a card's press still reaches the prompt; off puts
                  the pages back beside Generation;
    the info      Forge's generation parameters, drawn as a prompt block, a
                  negative block and chips, with Forge's own paragraph kept
                  out of view for what reads it; a new generation redraws it;
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
import html as html_module
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
.tabitem { border: 1px solid var(--border-color-primary); border-top: none; padding: 10px; }
#txt2img_generate, #img2img_generate { min-height: 4.5em; }
.gradio-button.tool { max-width: 2.2em; min-width: 2.2em !important; height: 2.4em; }
#quicksettings > div { width: fit-content; flex: 0 1 fit-content; }
.block { min-height: 40px; }
textarea { width: 100%; min-height: 60px; }
.extra-networks > div.tab-nav { min-height: 2.7rem; }
.extra-networks-controls-div { align-self: center; margin-left: auto; }
.extra-network-control { display: flex; width: 100%; }
.extra-network-pane { display: flex; height: calc(100vh - 24rem); min-height: 52rem; flex-direction: column; overflow: hidden; }
.extra-network-pane .card { display: inline-block; margin: 0.5rem; width: 16rem; height: 24rem; }
.extra-page-prompts { display: none; }
"""

# Forge's extra-network pages (modules/ui_extra_networks.py: a tab with the
# class `extra-page` per page, after Generation) and their cards.
PAGES = {"txt2img": [("textual_inversion", "Textual Inversion"), ("lora", "Lora")],
         "img2img": [("lora", "Lora")]}
CARDS = {"textual_inversion": [("bad_hands", "bad_hands")],
         "lora": [("detail", "<lora:detail:1>"), ("style", "<lora:style:1>")]}
INFOTEXT_HTML = ('<p>a cat in a hat<br>\nNegative prompt: blurry<br>\n'
                 'Steps: 8, Sampler: Euler a, CFG scale: 1, Seed: 42, Size: 1024x1024</p>')


def controls_html(tabname: str, page: str) -> str:
    """html/extra-networks-pane.html: the page's own controls, hidden until
    Forge's script gathers them into the tab strip and shows the page's."""
    full = f"{tabname}_{page}"
    return (f'<div class="extra-network-control" id="{full}_controls" style="display:none">'
            f'<div class="extra-network-control--search"><input id="{full}_extra_search" '
            f'class="extra-network-control--search-text" type="search" placeholder="Search"></div>'
            f'<div id="{full}_extra_sort_dir" class="extra-network-control--sort-dir" data-sortdir="Ascending"></div>'
            f'<div id="{full}_extra_refresh" class="extra-network-control--refresh"></div></div>')


def pane_html(tabname: str, page: str, *, controls_built: bool) -> str:
    full = f"{tabname}_{page}"
    cards = "".join(
        f'<div class="card" style="width: 256px; height: 384px" data-name="{name}" '
        f'onclick="cardClicked(\'{tabname}\', \'{html_module.escape(added)}\')">'
        f'<div class="actions"><span class="name">{name}</span></div></div>'
        for name, added in CARDS[page])
    return (f'<div id="{full}_pane" class="extra-network-pane">'
            + ("" if controls_built else controls_html(tabname, page))
            + f'<div class="extra-network-pane-content-dirs"><div id="{full}_dirs" class="extra-network-dirs"></div>'
            f'<div id="{full}_cards" class="extra-network-cards">{cards}</div></div></div>')


def extra_tabs_nav(tabname: str, *, controls_built: bool) -> str:
    """Gradio's strip of Forge's extra-network tabs -- Generation first --
    and, once Forge's script has wired the pages, the controls strip it
    builds there (javascript/extraNetworks.js, setupExtraNetworksForTab)."""
    buttons = (f'<button role="tab" class="selected" aria-selected="true" aria-controls="{tabname}_generation">Generation</button>'
               + "".join(f'<button role="tab" aria-selected="false" aria-controls="{tabname}_{page}">{label}</button>'
                         for page, label in PAGES[tabname]))
    strip = ('<div class="extra-networks-controls-div">'
             + "".join(controls_html(tabname, page) for page, _ in PAGES[tabname])
             + "</div>") if controls_built else ""
    return f'<div class="tab-nav scroll-hide" role="tablist">{buttons}{strip}</div>'


def extra_pages(tabname: str, *, controls_built: bool) -> str:
    return "".join(
        f'<div id="{tabname}_{page}" class="tabitem extra-page" role="tabpanel" style="display: none">'
        f'<div id="{tabname}_{page}_prompts" class="extra-page-prompts"></div>'
        f'<div id="{tabname}_{page}_cards_html">{pane_html(tabname, page, controls_built=controls_built)}</div></div>'
        for page, _ in PAGES[tabname])


def page_html(*, opts: dict, lobe: bool = False, settings_blocks: int = 40,
              controls_built: bool = True) -> str:
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
      {extra_tabs_nav("txt2img", controls_built=controls_built)}
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
              <div id="html_info_txt2img" class="block infotext"><div class="prose">{INFOTEXT_HTML}</div></div>
            </div>
          </div>
        </div>
      </div>
      {extra_pages("txt2img", controls_built=controls_built)}
    </div>
  </div>
  <div id="tab_img2img" class="tabitem" role="tabpanel" style="display: none">
    <div id="img2img_toprow" class="row"><div id="img2img_prompt" class="block"><label><textarea aria-label="Prompt"></textarea></label></div></div>
    <div id="img2img_extra_tabs" class="tabs extra-networks">
      {extra_tabs_nav("img2img", controls_built=controls_built)}
      <div id="img2img_generation" class="tabitem" role="tabpanel" style="display: block">
        <div id="img2img_settings" class="column"><div class="block">Resize</div></div>
        <div id="img2img_results" class="column"><div id="img2img_generate_box" class="row"><button id="img2img_generate">Generate</button></div><div id="img2img_results_panel" class="column panel"><div id="img2img_gallery_container" class="gr-group"><div id="img2img_gallery" class="block"></div></div><div id="html_info_img2img" class="block infotext"><div class="prose"></div></div></div></div>
      </div>
      {extra_pages("img2img", controls_built=controls_built)}
    </div>
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
const mainButtons = [...document.querySelectorAll('#tabs > .tab-nav > button')];
mainButtons.forEach((b) => b.addEventListener('click', () => {{
    window.pressed.push(b.id);
    // What Gradio does for a press: the button selected, its panel displayed, and
    // what Forge's script.js does on seeing it: the tab-change callbacks.
    mainButtons.forEach((x) => {{ x.classList.toggle('selected', x === b); x.setAttribute('aria-selected', x === b ? 'true' : 'false'); }});
    document.querySelectorAll('#tabs > .tabitem').forEach((p) => {{ p.style.display = p.id === b.getAttribute('aria-controls') ? 'block' : 'none'; }});
    hooks.tab.forEach((fn) => fn());
}}));
function cardClicked(tabname, text) {{
    const area = document.querySelector('#' + tabname + '_prompt textarea');
    area.value = (area.value ? area.value + ', ' : '') + text;
}}
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


def open_page(browser, opts=None, *, width=1600, height=900, lobe=False, loaded=True, context=None,
              controls_built=True):
    """A page of its own context, or one in a context given -- two pages of
    one context share a localStorage, which is what the first paint reads."""
    owner = context or browser.new_context(viewport={"width": width, "height": height})
    page = owner.new_page()
    html = page_html(opts=dict(ON, **(opts or {})), lobe=lobe, controls_built=controls_built)
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


NETWORKS = """() => {
    const q = (s) => document.querySelector(s);
    const aside = q("#mc-theme-networks");
    const toggle = q(".mc-theme-networks-toggle");
    const nav = q("#txt2img_extra_tabs > .tab-nav");
    const where = (el) => el.parentElement.className || el.parentElement.id;
    const page = (id) => {
        const el = document.getElementById(id);
        return {parent: where(el), display: getComputedStyle(el).display,
                shown: el.classList.contains("mc-theme-networks-shown")};
    };
    const controls = (id) => { const el = document.getElementById(id); return {parent: where(el), display: el.style.display}; };
    const rect = aside ? aside.getBoundingClientRect() : null;
    const strip = q(".extra-networks-controls-div");
    const card = q("#txt2img_lora_cards .card").getBoundingClientRect();
    return {
        aside: aside ? {hidden: aside.hidden, position: getComputedStyle(aside).position,
                        rect: {top: rect.top, right: rect.right, width: rect.width},
                        empty: aside.classList.contains("mc-theme-networks-empty"),
                        note: getComputedStyle(q(".mc-theme-networks-body"), "::before").content} : null,
        toggle: toggle ? {hidden: toggle.hidden, expanded: toggle.getAttribute("aria-expanded"),
                          disabled: toggle.disabled} : null,
        navMarked: nav.classList.contains("mc-theme-networks-nav"),
        navDisplay: getComputedStyle(nav).display,
        generationDisplay: getComputedStyle(q("#txt2img_generation")).display,
        generationSelected: q("#txt2img_extra_tabs > .tab-nav > button.selected").textContent,
        extraTabsChildren: [...q("#txt2img_extra_tabs").children].map((c) => c.id || c.className),
        ti: page("txt2img_textual_inversion"), lora: page("txt2img_lora"), img2imgLora: page("img2img_lora"),
        tiControls: controls("txt2img_textual_inversion_controls"),
        loraControls: controls("txt2img_lora_controls"),
        stripParent: strip ? where(strip) : null,
        tabs: [...document.querySelectorAll(".mc-theme-networks-tab")].map((b) => ({
            text: b.textContent, hidden: b.hidden, active: b.classList.contains("mc-theme-networks-active"),
            selected: b.getAttribute("aria-selected")})),
        groups: [...document.querySelectorAll(".mc-theme-networks-group")].map((g) => ({tab: g.dataset.mcThemeTab, hidden: g.hidden})),
        card: {width: card.width, height: card.height},
        prompt: q("#txt2img_prompt textarea").value,
    };
}"""

# What Forge's setupExtraNetworksForTab does to the page once the pages'
# HTML has arrived: the controls strip in the tab bar, every page's controls
# gathered into it.
FORGE_WIRES = """() => {
    ["txt2img", "img2img"].forEach((tabname) => {
        const nav = document.querySelector("#" + tabname + "_extra_tabs > div.tab-nav");
        const strip = document.createElement("DIV");
        strip.classList.add("extra-networks-controls-div");
        nav.appendChild(strip);
        document.querySelectorAll("#" + tabname + "_extra_tabs > [id^='" + tabname + "_']").forEach((elem) => {
            const controls = document.getElementById(elem.id + "_controls");
            if (controls) strip.appendChild(controls);
        });
    });
}"""

INFO = """() => {
    const q = (s) => document.querySelector(s);
    const source = q("#html_info_txt2img");
    const next = source.nextElementSibling;
    const panel = next && next.classList.contains("mc-theme-info") ? next : null;
    const style = getComputedStyle(source);
    return {
        panels: document.querySelectorAll(".mc-theme-info").length,
        img2imgPanels: document.querySelectorAll("#img2img_results_panel .mc-theme-info").length,
        img2imgMarked: q("#html_info_img2img").classList.contains("mc-theme-info-source"),
        sourceMarked: source.classList.contains("mc-theme-info-source"),
        sourcePosition: style.position, sourceClip: style.clip,
        sourceText: source.textContent.replace(/\\s+/g, " ").trim(),
        blocks: panel ? [...panel.querySelectorAll(".mc-theme-info-block")].map((b) => ({
            label: b.querySelector(".mc-theme-info-label").firstChild.textContent,
            text: b.querySelector(".mc-theme-info-text").textContent,
            copy: !!b.querySelector(".mc-theme-info-copy svg")})) : null,
        chips: panel ? [...panel.querySelectorAll(".mc-theme-info-chip")].map((c) => ({
            key: c.querySelector("b").textContent, text: c.textContent})) : null,
        panelHeight: panel ? panel.getBoundingClientRect().height : null,
        copied: window.copied === undefined ? null : window.copied,
        lit: !!q(".mc-theme-info-copy.mc-theme-info-copied"),
    };
}"""

STUB_CLIPBOARD = ("Object.defineProperty(navigator, 'clipboard', {configurable: true, value: "
                  "{writeText: (text) => { window.copied = text; return Promise.resolve(); }}})")


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
        # The split's two rows and two boxes and the quick settings, and the
        # drawer's three pages and two controls strips: ten, however often.
        assert page.evaluate("document.querySelectorAll('[data-mc-theme-moved]').length") == 10
        page.close()

    def test_the_tab_panels_lose_their_frame_and_generate_is_44_px(self, browser):
        """Gradio frames every tab panel (a border and its block padding), and
        Forge's own tabs nest two of them around the generation UI; Forge
        gives Generate a 4.5 em floor by id. The page the user had put the
        content at the page's margin with a 44 px Generate."""
        page = open_page(browser)
        found = page.evaluate("""() => {
            const box = (id) => { const s = getComputedStyle(document.getElementById(id));
                                  return {padding: s.paddingLeft, border: s.borderLeftWidth}; };
            return {outer: box("tab_txt2img"), inner: box("txt2img_generation"),
                    generate: document.getElementById("txt2img_generate").getBoundingClientRect().height,
                    contentLeft: document.getElementById("txt2img_prompt").getBoundingClientRect().left,
                    pageLeft: document.querySelector(".gradio-container").getBoundingClientRect().left};
        }""")

        assert found["outer"] == {"padding": "0px", "border": "0px"}
        assert found["inner"] == {"padding": "0px", "border": "0px"}
        assert found["generate"] == 44
        # The host's own 16 px page padding is the only margin left.
        assert found["contentLeft"] - found["pageLeft"] == 16
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


class TestTheDrawersPlace:
    def test_the_drawers_live_inside_forges_app_element_where_its_lookups_reach(self, browser):
        """Forge's scripts find things through gradioApp().querySelector,
        which never looks outside the <gradio-app> element, and so does this
        one. A drawer on the body put the quick settings out of both their
        reach: this script then found no row, put it back, found it on the
        next update, took it again -- the row showed and hid at the top of
        the page on every update. And the extra-network pages' cards and
        search boxes, which Forge's sort and card presses look up by id
        through the same call, were out of Forge's reach."""
        page = open_page(browser)
        where = """() => ({
            quick: !!document.getElementById("mc-theme-quick").closest("gradio-app"),
            networks: !!document.getElementById("mc-theme-networks").closest("gradio-app"),
            row: !!gradioApp().querySelector("#quicksettings"),
            cards: !!gradioApp().querySelector("#txt2img_lora_cards"),
            search: !!gradioApp().querySelector("#txt2img_lora_extra_search"),
            rowParent: document.getElementById("quicksettings").parentElement.id,
            drawers: document.querySelectorAll("#mc-theme-quick").length,
        })"""
        found = page.evaluate(where)
        assert found == {"quick": True, "networks": True, "row": True, "cards": True, "search": True,
                         "rowParent": "mc-theme-quick", "drawers": 1}

        # Update after update -- an odd number and an even one -- the row
        # stays in the one drawer.
        for _ in range(3):
            page.evaluate("hooks.after.forEach((fn) => fn())")
            settle(page)
            found = page.evaluate(where)
            assert found["rowParent"] == "mc-theme-quick" and found["drawers"] == 1, found
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


class TestTheExtraNetworksDrawer:
    def test_the_pages_leave_forges_tabs_for_a_drawer_on_the_right_and_generation_stays(self, browser):
        page = open_page(browser)
        found = page.evaluate(NETWORKS)

        assert found["aside"]["hidden"] is True
        assert found["toggle"] == {"hidden": False, "expanded": "false", "disabled": False}
        assert found["navMarked"] is True and found["navDisplay"] == "none"
        assert found["generationDisplay"] == "block" and found["generationSelected"] == "Generation"
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide mc-theme-networks-nav", "txt2img_generation"]
        assert found["ti"]["parent"] == "mc-theme-networks-group"
        assert found["lora"]["parent"] == "mc-theme-networks-group"
        assert found["img2imgLora"]["parent"] == "mc-theme-networks-group"
        assert found["stripParent"] == "mc-theme-networks-group"
        assert [t["text"] for t in found["tabs"]] == ["Textual Inversion", "Lora", "Lora"]
        assert [t["hidden"] for t in found["tabs"]] == [False, False, True]
        assert found["groups"] == [{"tab": "txt2img", "hidden": False}, {"tab": "img2img", "hidden": True}]
        # The first page is the one shown, with its controls and no other's.
        assert found["ti"]["shown"] is True and found["lora"]["shown"] is False
        assert found["tiControls"] == {"parent": "extra-networks-controls-div", "display": ""}
        assert found["loraControls"] == {"parent": "extra-networks-controls-div", "display": "none"}
        # Each tab's group keeps a page of its own: img2img's Lora is its one page.
        assert [t["active"] for t in found["tabs"]] == [True, False, True]

        page.click(".mc-theme-networks-toggle")
        settle(page)
        found = page.evaluate(NETWORKS)
        assert found["aside"]["hidden"] is False
        assert found["aside"]["position"] == "fixed"
        assert found["aside"]["rect"] == {"top": 64, "right": 1600, "width": 340}
        assert found["toggle"]["expanded"] == "true"
        assert found["ti"]["display"] == "block" and found["lora"]["display"] == "none"
        assert found["aside"]["empty"] is False
        page.close()

    def test_a_strip_button_shows_that_page_and_its_controls_and_a_card_still_reaches_the_prompt(self, browser):
        page = open_page(browser)
        page.click(".mc-theme-networks-toggle")
        page.click("[data-mc-theme-tab='txt2img'][data-mc-theme-page='txt2img_lora']")
        settle(page)
        found = page.evaluate(NETWORKS)

        assert found["lora"]["shown"] is True and found["ti"]["shown"] is False
        assert found["lora"]["display"] == "block" and found["ti"]["display"] == "none"
        assert found["loraControls"]["display"] == "" and found["tiControls"]["display"] == "none"
        assert [t["active"] for t in found["tabs"]] == [False, True, True]
        assert [t["selected"] for t in found["tabs"]] == ["false", "true", "true"]
        assert found["generationSelected"] == "Generation"
        # Forge's own sizes, 16 by 24 rem and the inline ones, give way to the setting.
        assert found["card"] == {"width": 120, "height": 180}

        page.click("#txt2img_lora_cards .card")
        assert page.evaluate(NETWORKS)["prompt"] == "<lora:detail:1>"
        page.close()

    def test_escape_closes_it_and_the_choice_is_remembered(self, browser):
        page = open_page(browser)
        page.click(".mc-theme-networks-toggle")
        settle(page)
        page.keyboard.press("Escape")
        settle(page)

        assert page.evaluate(NETWORKS)["aside"]["hidden"] is True
        assert page.evaluate("localStorage.getItem('mc-theme-networks')") == "closed"
        page.click(".mc-theme-networks-toggle")
        settle(page)
        assert page.evaluate("localStorage.getItem('mc-theme-networks')") == "open"
        page.click(".mc-theme-networks-close")
        settle(page)
        assert page.evaluate(NETWORKS)["aside"]["hidden"] is True
        assert page.evaluate("localStorage.getItem('mc-theme-networks')") == "closed"
        page.close()

    def test_open_by_default_the_width_and_the_card_size_follow_the_settings(self, browser):
        page = open_page(browser, {"model_chain_theme_networks": "open",
                                   "model_chain_theme_networks_width": 400,
                                   "model_chain_theme_networks_card": 100})
        page.click("[data-mc-theme-tab='txt2img'][data-mc-theme-page='txt2img_lora']")
        settle(page)
        found = page.evaluate(NETWORKS)

        assert found["aside"]["hidden"] is False
        assert found["aside"]["rect"]["width"] == 400
        assert found["card"] == {"width": 100, "height": 150}
        page.close()

    def test_the_drawer_follows_the_main_tab(self, browser):
        page = open_page(browser)
        page.click("#tab_img2img-button")
        settle(page)
        found = page.evaluate(NETWORKS)

        assert [t["hidden"] for t in found["tabs"]] == [True, True, False]
        assert found["groups"] == [{"tab": "txt2img", "hidden": True}, {"tab": "img2img", "hidden": False}]
        assert found["aside"]["empty"] is False and found["toggle"]["disabled"] is False

        page.click("#tab_clipboard-button")
        settle(page)
        found = page.evaluate(NETWORKS)
        assert [t["hidden"] for t in found["tabs"]] == [True, True, True]
        assert found["groups"] == [{"tab": "txt2img", "hidden": True}, {"tab": "img2img", "hidden": True}]
        assert found["aside"]["empty"] is True and found["toggle"]["disabled"] is True
        assert "No extra networks" in found["aside"]["note"]
        page.close()

    def test_a_page_is_taken_only_once_forge_has_wired_it(self, browser):
        """Forge wires the search, sort and refresh of a page after its HTML
        has arrived, later than the host's loaded hook, and looks for the
        pages as its own tabs' children: a page moved before that would
        never be wired. The strip Forge builds for the controls is the sign."""
        page = open_page(browser, controls_built=False)
        found = page.evaluate(NETWORKS)

        assert found["aside"] is not None
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide", "txt2img_generation",
                                              "txt2img_textual_inversion", "txt2img_lora"]
        assert found["navMarked"] is False and found["navDisplay"] == "flex"
        assert found["tiControls"]["parent"] == "extra-network-pane"
        assert found["tabs"] == [] and found["groups"] == []
        assert found["aside"]["empty"] is True and found["toggle"]["disabled"] is True

        page.evaluate(FORGE_WIRES)
        page.evaluate("hooks.after.forEach((fn) => fn())")
        settle(page)
        found = page.evaluate(NETWORKS)
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide mc-theme-networks-nav", "txt2img_generation"]
        assert found["stripParent"] == "mc-theme-networks-group"
        assert found["tiControls"] == {"parent": "extra-networks-controls-div", "display": ""}
        assert found["loraControls"] == {"parent": "extra-networks-controls-div", "display": "none"}
        assert [t["text"] for t in found["tabs"]] == ["Textual Inversion", "Lora", "Lora"]
        assert found["toggle"]["disabled"] is False
        page.close()

    def test_off_puts_the_pages_back_beside_generation(self, browser):
        page = open_page(browser)
        page.click(".mc-theme-networks-toggle")
        settle(page)
        page.evaluate("opts.model_chain_theme_networks = 'off'")
        changed(page)
        found = page.evaluate(NETWORKS)

        assert found["aside"] is None
        assert found["toggle"]["hidden"] is True
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide", "txt2img_generation",
                                              "txt2img_textual_inversion", "txt2img_lora"]
        assert found["navMarked"] is False and found["navDisplay"] == "flex"
        assert found["stripParent"] == "tab-nav scroll-hide"
        assert found["ti"]["shown"] is False and found["ti"]["display"] == "none"
        assert found["tiControls"]["display"] == "none" and found["loraControls"]["display"] == "none"
        assert page.evaluate("document.querySelector('#img2img_lora').parentElement.id") == "img2img_extra_tabs"
        # And the rest of the theme is still on.
        assert page.evaluate(MEASURE)["attribute"] == "on"
        page.close()


class TestTheGenerationInfo:
    def test_forges_paragraph_is_drawn_as_blocks_and_chips_and_kept_for_what_reads_it(self, browser):
        page = open_page(browser)
        found = page.evaluate(INFO)

        assert found["panels"] == 1
        assert found["blocks"] == [{"label": "Prompt", "text": "a cat in a hat", "copy": True},
                                   {"label": "Negative prompt", "text": "blurry", "copy": True}]
        assert found["chips"] == [{"key": "Steps", "text": "Steps 8"}, {"key": "Sampler", "text": "Sampler Euler a"},
                                  {"key": "CFG scale", "text": "CFG scale 1"}, {"key": "Seed", "text": "Seed 42"},
                                  {"key": "Size", "text": "Size 1024x1024"}]
        assert found["panelHeight"] > 0
        # Forge's own paragraph: still there, with its text, out of view.
        assert found["sourceMarked"] is True
        assert found["sourcePosition"] == "absolute"
        assert found["sourceClip"] == "rect(0px, 0px, 0px, 0px)"
        assert found["sourceText"].startswith("a cat in a hat Negative prompt: blurry Steps: 8")
        # An empty one draws nothing.
        assert found["img2imgPanels"] == 0 and found["img2imgMarked"] is False
        page.close()

    def test_the_copy_buttons_copy_the_prompt_and_the_whole_text(self, browser):
        page = open_page(browser)
        page.evaluate(STUB_CLIPBOARD)
        page.click("#html_info_txt2img + .mc-theme-info .mc-theme-info-block:first-child .mc-theme-info-copy")
        found = page.evaluate(INFO)

        assert found["copied"] == "a cat in a hat"
        assert found["lit"] is True
        page.click("#html_info_txt2img + .mc-theme-info .mc-theme-info-copy-all")
        assert page.evaluate(INFO)["copied"] == ("a cat in a hat\nNegative prompt: blurry\n"
                                                  "Steps: 8, Sampler: Euler a, CFG scale: 1, Seed: 42, Size: 1024x1024")
        page.close()

    def test_a_new_generation_redraws_it_once(self, browser):
        page = open_page(browser)
        page.evaluate("document.querySelector('#html_info_txt2img .prose').innerHTML = "
                      "'<p>a dog<br>\\nSteps: 4, Sampler: DPM, Seed: 7</p>'")
        page.evaluate("hooks.after.forEach((fn) => fn()); hooks.after.forEach((fn) => fn())")
        settle(page)
        found = page.evaluate(INFO)

        assert found["panels"] == 1
        assert found["blocks"] == [{"label": "Prompt", "text": "a dog", "copy": True}]
        assert [c["key"] for c in found["chips"]] == ["Steps", "Sampler", "Seed"]

        page.evaluate("document.querySelector('#html_info_txt2img .prose').innerHTML = '<p></p>'")
        page.evaluate("hooks.after.forEach((fn) => fn())")
        settle(page)
        found = page.evaluate(INFO)
        assert found["panels"] == 0 and found["sourceMarked"] is False
        page.close()

    def test_off_leaves_forges_text_as_it_was(self, browser):
        page = open_page(browser)
        page.evaluate("opts.model_chain_theme_image_info = false")
        changed(page)
        found = page.evaluate(INFO)

        assert found["panels"] == 0
        assert found["sourceMarked"] is False
        assert found["sourcePosition"] == "static"
        assert page.evaluate(MEASURE)["attribute"] == "on"
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
        assert page.evaluate(NETWORKS)["aside"] is None
        assert page.evaluate("document.querySelector('#txt2img_lora').parentElement.id") == "txt2img_extra_tabs"
        assert page.evaluate(INFO)["panels"] == 0
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
        networks = page.evaluate(NETWORKS)
        assert networks["aside"] is None
        assert networks["extraTabsChildren"] == ["tab-nav scroll-hide", "txt2img_generation",
                                                 "txt2img_textual_inversion", "txt2img_lora"]
        assert networks["navDisplay"] == "flex"
        assert networks["stripParent"] == "tab-nav scroll-hide"
        info = page.evaluate(INFO)
        assert info["panels"] == 0 and info["sourceMarked"] is False
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
