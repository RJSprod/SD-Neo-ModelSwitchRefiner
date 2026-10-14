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
# the page's padding, Forge's own tool-button and quick-settings rules. The
# page's padding is Gradio 4.40's own, read from its built stylesheet: 16 px
# above and below and 32 px at the sides (`.app { padding: var(--size-4)
# var(--size-8) }`). A stand-in with 16 px all round passed a bar whose
# ground stopped 16 px short of the user's window on each side.
HOST_CSS = """
*, ::before, ::after { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--body-background-fill); color: var(--body-text-color); font-family: sans-serif;
       --text-md: 14px; --body-text-size: var(--text-md);
       --body-background-fill: #000; --body-text-color: #e6e6e6; --body-text-color-subdued: #8c8c8c;
       --background-fill-secondary: #141414; --border-color-primary: #262626; --block-background-fill: #0d0d0d;
       --block-border-color: #262626; --button-secondary-background-fill: #1a1a1a;
       --button-secondary-border-color: #2b2b2b; --button-secondary-text-color: #d9d9d9;
       --button-cancel-background-fill: #d93025; --button-cancel-text-color: #fff; --button-cancel-border-color: #d93025;
       --color-accent: #fff; --primary-500: #fff; --font-mono: monospace; --size-4: 16px; --size-8: 32px;
       --shadow-drop-lg: none; }
.app { padding: var(--size-4) var(--size-8); font-size: var(--body-text-size); }
.row { display: flex; flex-wrap: wrap; gap: 8px; }
.unequal-height { align-items: flex-start; }
.column { display: flex; flex-direction: column; gap: 8px; min-width: 0; flex: 1; }
.tabs { position: relative; }
.tab-nav { display: flex; flex-wrap: wrap; border-bottom: 1px solid var(--border-color-primary); }
.tab-nav button { padding: 4px 16px; border: 1px solid transparent; background: transparent; color: inherit; }
.tabitem[style*="display: none"] { display: none; }
.tabitem { border: 1px solid var(--border-color-primary); border-top: none; padding: 10px; }
#txt2img_generate, #img2img_generate { min-height: 4.5em; }
.generate-box { position: relative; }
.gradio-button.generate-box-skip, .gradio-button.generate-box-interrupt { position: absolute; width: 50%; height: 100%; display: none; background: #b4c0cc; }
.gradio-button.tool { max-width: 2.2em; min-width: 2.2em !important; height: 2.4em; }
#quicksettings > div { width: fit-content; flex: 0 1 fit-content; }
.block { min-height: 40px; }
textarea { width: 100%; min-height: 60px; }
.extra-networks > div.tab-nav { min-height: 2.7rem; }
.extra-networks-controls-div { align-self: center; margin-left: auto; }
.extra-network-control { display: flex; width: 100%; }
.extra-network-pane { display: flex; height: calc(100vh - 24rem); min-height: 52rem; flex-direction: column; overflow: hidden; }
.extra-network-pane .card { display: inline-block; margin: 0.5rem; width: 16rem; height: 24rem; position: relative; }
.extra-network-pane .card .actions { position: absolute; bottom: 0; left: 0; right: 0; padding: 0.5em; }
.extra-network-pane .card .actions .name { font-size: 1.7em; font-weight: bold; }
.extra-network-control .extra-network-control--search::before { content: "\\1F50E\\FE0E"; position: absolute; margin: 0.5rem; font-size: 1rem; color: #888; }
.extra-network-control .extra-network-control--search { display: inline-flex; position: relative; }
.extra-network-control .extra-network-control--search .extra-network-control--search-text { border: 1px solid #2b2b2b; border-radius: 0.5rem; background-color: transparent; width: 16em; padding-left: 2rem; line-height: 1rem; color: inherit; }
.extra-network-control small { line-height: 2.2rem; margin: 0 0.5rem 0 0.75rem; }
.extra-network-control .extra-network-control--sort, .extra-network-control .extra-network-control--sort-dir,
.extra-network-control .extra-network-control--tree-view, .extra-network-control .extra-network-control--refresh { padding: 0.25rem; display: inline-flex; }
.extra-network-control .extra-network-control--icon { width: 1.5rem; height: 1.5rem; background: #888; display: block; }
.global-popup { display: flex; position: fixed; z-index: 1001; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; }
.global-popup-close { position: fixed; left: 0; top: 0; width: 100%; height: 100%; background: rgba(20, 20, 20, 0.95); }
.global-popup-inner { display: inline-block; margin: auto; padding: 2em; z-index: 1001; max-height: 90%; max-width: 90%; }
.edit-user-metadata { width: 56em; background: var(--body-background-fill) !important; padding: 2em !important; }
.edit-user-metadata .extra-network-name { font-size: 18pt; }
.edit-user-metadata .file-metadata th, .edit-user-metadata .file-metadata td { padding: 0.3em 1em; }
.prose table td, .prose table th { border: 1px solid #555; padding: 8px 12px; }
.gap { gap: var(--layout-gap, 12px); }
button.lg { font-size: 16px; padding: 8px 16px; min-height: 40px; }
.extra-page-prompts { display: none; }
"""

# What Gradio 4.40 and the Forge Assistant's own tests give the gallery and
# its container, for a page the assistant's scripts run on too.
ASSISTANT_HOST_CSS = """
.gr-group, .styler { display: flex; flex-direction: column; overflow: hidden; }
.grid-wrap { position: relative; padding: 8px; overflow-y: scroll; }
.grid-container { display: grid; position: relative; gap: 16px; grid-template-columns: repeat(4, minmax(0, 1fr)); grid-auto-rows: minmax(100px, 1fr); }
.fixed-height { min-height: 320px; max-height: 55vh; }
.panel { margin: 0 !important; padding: 16px !important; }
#image_buttons_txt2img button { flex: 1 1 100px; height: 32px; }
"""

# Forge Neo's own rules for its progress bar (style.css: absolute, 14 px above
# the results panel, 20 px tall) and Gradio 4.40's column, which is positioned
# and so is the box the bar is drawn against.
FORGE_PROGRESS_CSS = """
.column { position: relative; }
.progressDiv { position: absolute; height: 20px; background: #b4c0cc; border-radius: 3px !important; top: -14px; left: 0px; width: 100%; }
.dark .progressDiv { background: #424c5b; }
.progressDiv .progress { width: 0%; height: 20px; background: #0060df; color: white; font-weight: bold; line-height: 20px; text-align: right; border-radius: 3px; overflow: visible; white-space: nowrap; padding: 0 0.5em; }
"""

# The person's user.css, as the assistant's tests carry it: the gallery and its
# container sized against the window. And the container alone, which is what
# a theme that lets the gallery stop short of it turns into half a window of
# nothing between the gallery and its buttons.
USER_CSS_WHOLE = """
#txt2img_gallery, #img2img_gallery { min-height: 80vh; }
#txt2img_gallery_container, #img2img_gallery_container { height: 85vh !important; }
#txt2img_gallery > .grid-wrap, #txt2img_gallery > .preview { max-height: 80vh; }
"""
USER_CSS_CONTAINER = "#txt2img_gallery_container, #img2img_gallery_container { height: 85vh !important; }"

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
    sorts = "".join(f'<div id="{full}_extra_sort_{key}" class="extra-network-control--sort" data-sortkey="{key}">'
                    f'<i class="extra-network-control--icon extra-network-control--sort-icon"></i></div>'
                    for key in ("default", "name", "date_created", "date_modified"))
    return (f'<div class="extra-network-control" id="{full}_controls" style="display:none">'
            f'<div class="extra-network-control--search"><input id="{full}_extra_search" '
            f'class="extra-network-control--search-text" type="search" placeholder="Search"></div>'
            f'<small>Sort: </small>{sorts}<small> </small>'
            f'<div id="{full}_extra_sort_dir" class="extra-network-control--sort-dir" data-sortdir="Ascending">'
            f'<i class="extra-network-control--icon extra-network-control--sort-dir-icon"></i></div><small> </small>'
            f'<div id="{full}_extra_tree_view" class="extra-network-control--tree-view">'
            f'<i class="extra-network-control--icon extra-network-control--tree-view-icon"></i></div>'
            f'<div id="{full}_extra_refresh" class="extra-network-control--refresh">'
            f'<i class="extra-network-control--icon extra-network-control--refresh-icon"></i></div></div>')


# A card's metadata editor, open: Forge's popup (javascript/ui.js, popup())
# around the editor Gradio built (modules/ui_extra_networks_user_metadata.py,
# extensions-builtin/sd_forge_lora/ui_edit_user_metadata.py).
POPUP_HTML = """<div class="global-popup"><div class="global-popup-close"></div><div class="global-popup-inner">
<div id="txt2img_lora_edit_user_metadata" class="edit-user-metadata block gap">
  <div class="row gap"><div class="column gap" style="flex: 2">
    <div class="block"><div class="prose extra-network-name">kreapov</div></div>
    <div class="block"><label><span>Description</span><textarea rows="2"></textarea></label></div>
    <div class="block"><div class="prose"><table class="file-metadata"><tr><th>Filename:</th><td>krea2\\kreapov.safetensors</td></tr>
      <tr><th>File size:</th><td>218MB</td></tr><tr><th>Hash:</th><td>07ba28af4eef</td></tr>
      <tr><th>Model:</th><td>urn:air:krea2:repository:huggingface:krea/Krea-2-Raw@main.tar</td></tr></table></div></div>
  </div><div class="column gap" style="flex: 1; min-width: 0"><div class="block"><div class="prose"><div style="width: 100%; aspect-ratio: 2 / 3; background: #333"></div></div></div></div></div>
  <div class="block"><label><span>Preset</span><input value="Unknown"></label></div>
  <div class="block"><label><span>Activation Text</span><textarea rows="1"></textarea></label></div>
  <div class="block"><label><span>Preferred Weight</span><input value="0.0"></label></div>
  <div class="block"><label><span>Negative Prompt</span><textarea rows="1"></textarea></label></div>
  <div class="block"><label><span>Notes</span><textarea rows="2"></textarea></label></div>
  <div class="row gap edit-user-metadata-buttons"><button class="gradio-button">Cancel</button><button class="gradio-button primary">Replace preview</button><button class="gradio-button primary">Save</button></div>
</div></div></div>"""


def pane_html(tabname: str, page: str, *, controls_built: bool) -> str:
    full = f"{tabname}_{page}"
    cards = "".join(
        f'<div class="card" style="width: 256px; height: 384px" data-name="{name}" '
        f'onclick="cardClicked(\'{tabname}\', \'{html_module.escape(added)}\')">'
        f'<div class="actions"><div class="additional"><span class="search_terms">{name}</span></div>'
        f'<span class="name">{name}</span></div></div>'
        for name, added in CARDS[page])
    dirs = "".join(f'<button class="lg secondary gradio-button custom-button">{name}</button>'
                   for name in ("all", "krea2\\", "kreaCELEB\\", "kreat\\"))
    return (f'<div id="{full}_pane" class="extra-network-pane">'
            + ("" if controls_built else controls_html(tabname, page))
            + f'<div class="extra-network-pane-content-dirs"><div id="{full}_dirs" class="extra-network-dirs">{dirs}</div>'
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
              controls_built: bool = True, assistant: str = "", popup: bool = False,
              progress: bool = False) -> str:
    """`assistant` is a user.css to put on a page that also runs the Forge
    Assistant's scripts, in Forge's order (its folder's scripts load before
    this one's), with Gradio's gallery markup for its fill to measure.
    `progress` adds Forge's own progress bar rules and this extension's
    progress-bar appearance script, for a page that runs a generation."""
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
<style>{HOST_CSS}{ASSISTANT_HOST_CSS if assistant else ""}{FORGE_PROGRESS_CSS if progress else ""}</style>
<link rel="stylesheet" href="/style.css">
<style id="user-css">{assistant}</style>
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
          <button id="txt2img_interrupt" class="gradio-button generate-box-interrupt">Interrupt</button><button id="txt2img_skip" class="gradio-button generate-box-skip">Skip</button>
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
        <div class="row unequal-height">
          <div id="txt2img_settings" class="column">{blocks}</div>
          <div id="txt2img_results" class="column">
            <div id="txt2img_results_panel" class="column panel">
              <div id="txt2img_gallery_container" class="gr-group">
                {'<div class="styler"><div id="txt2img_gallery" class="block gradio-gallery"><div class="grid-wrap fixed-height"><div class="grid-container"></div></div></div></div>' if assistant else '<div id="txt2img_gallery" class="block gradio-gallery" style="height: 300px"></div>'}
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
{POPUP_HTML if popup else ""}
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
{"".join(f'<script src="/javascript/{path.name}"></script>' for path in sorted((ROOT / "javascript").glob("forge_assistant*.js"))) if assistant else ""}
{'<script src="/javascript/model_chain_progress.js"></script>' if progress else ""}
<script src="/javascript/model_chain_theme.js"></script>
</body></html>"""


def _answer(route, html: str) -> None:
    path = urllib.parse.urlsplit(route.request.url).path
    if path == "/":
        route.fulfill(status=200, content_type="text/html", body=html)
    elif path == "/style.css":
        route.fulfill(status=200, content_type="text/css", body=STYLE.read_text(encoding="utf-8"))
    elif path.startswith("/javascript/") and "/" not in path[12:] and (ROOT / path[1:]).is_file():
        route.fulfill(status=200, content_type="text/javascript",
                      body=(ROOT / path[1:]).read_text(encoding="utf-8"))
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
              controls_built=True, assistant="", popup=False, progress=False):
    """A page of its own context, or one in a context given -- two pages of
    one context share a localStorage, which is what the first paint reads."""
    owner = context or browser.new_context(viewport={"width": width, "height": height})
    page = owner.new_page()
    html = page_html(opts=dict(ON, **(opts or {})), lobe=lobe, controls_built=controls_built,
                     assistant=assistant, popup=popup, progress=progress)
    page.route(ORIGIN + "/**", lambda route: _answer(route, html))
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!window.mcTheme" + (" && !!(window.forgeAssistant && window.forgeAssistant.fill)"
                                                 if assistant else ""))
    if loaded:
        page.evaluate("hooks.loaded.forEach((fn) => fn())")
        settle(page)
    return page


def changed(page):
    page.evaluate("hooks.changed.forEach((fn) => fn())")
    settle(page)


# What Forge's own after-update observer would hear: every change to a child
# list inside its app element, by the changed node.
HEARD = """() => {
    window.heard = [];
    new MutationObserver((records) => records.forEach((record) => {
        const node = record.target;
        window.heard.push((node.id || node.className || node.nodeName).toString().slice(0, 40));
    })).observe(document.querySelector("gradio-app"), {childList: true, subtree: true});
}"""


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
        bodyFont: getComputedStyle(document.body).fontSize,
        outsideFont: q("#mc-theme-quick") ? getComputedStyle(q("#mc-theme-quick")).fontSize : null,
        generateHeight: q("#txt2img_generate").getBoundingClientRect().height,
        toolWidth: q("#paste").getBoundingClientRect().width,
    };
}"""


NETWORKS = """() => {
    const q = (s) => document.querySelector(s);
    const host = q("#txt2img_extra_tabs");
    const chrome = q("#mc-theme-networks-txt2img");
    const other = q("#mc-theme-networks-img2img");
    const toggle = q(".mc-theme-networks-toggle");
    const nav = q("#txt2img_extra_tabs > .tab-nav");
    const where = (el) => el.parentElement.id || el.parentElement.className;
    const box = (el) => { const r = el.getBoundingClientRect();
                          return {top: Math.round(r.top), right: Math.round(r.right),
                                  width: Math.round(r.width), height: Math.round(r.height)}; };
    const page = (id) => {
        const el = document.getElementById(id);
        const style = getComputedStyle(el);
        return {parent: where(el), display: style.display, position: style.position,
                shown: el.classList.contains("mc-theme-networks-shown"),
                marked: el.classList.contains("mc-theme-networks-page"), rect: box(el)};
    };
    const controls = (id) => { const el = document.getElementById(id); return {parent: where(el), display: el.style.display}; };
    const strip = q(".extra-networks-controls-div");
    const card = q("#txt2img_lora_cards .card").getBoundingClientRect();
    const drawn = (el) => !!el && el.getClientRects().length > 0;
    return {
        host: {open: host.classList.contains("mc-theme-networks-open"),
               marked: host.classList.contains("mc-theme-networks-host"),
               layout: host.getAttribute("data-mc-theme-networks"), display: getComputedStyle(host).display,
               width: host.style.getPropertyValue("--mc-theme-networks-width"),
               chrome: host.style.getPropertyValue("--mc-theme-networks-chrome")},
        chrome: chrome ? {parent: chrome.parentElement.id, display: getComputedStyle(chrome).display,
                          position: getComputedStyle(chrome).position, rect: box(chrome), drawn: drawn(chrome),
                          handle: !!chrome.querySelector(".mc-theme-networks-handle")} : null,
        other: other ? {parent: other.parentElement.id, drawn: drawn(other)} : null,
        toggle: toggle ? {hidden: toggle.hidden, expanded: toggle.getAttribute("aria-expanded"),
                          disabled: toggle.disabled} : null,
        navMarked: nav.classList.contains("mc-theme-networks-nav"),
        navDisplay: getComputedStyle(nav).display,
        generationDisplay: getComputedStyle(q("#txt2img_generation")).display,
        generationSelected: q("#txt2img_extra_tabs > .tab-nav > button.selected").textContent,
        extraTabsChildren: [...host.children].map((c) => c.id || c.className),
        ti: page("txt2img_textual_inversion"), lora: page("txt2img_lora"), img2imgLora: page("img2img_lora"),
        tiControls: controls("txt2img_textual_inversion_controls"),
        loraControls: controls("txt2img_lora_controls"),
        stripParent: strip ? where(strip) : null,
        tabs: [...document.querySelectorAll(".mc-theme-networks-tab")].map((b) => ({
            text: b.textContent, tab: b.dataset.mcThemeTab, active: b.classList.contains("mc-theme-networks-active"),
            selected: b.getAttribute("aria-selected")})),
        card: {width: card.width, height: card.height},
        prompt: q("#txt2img_prompt textarea").value,
    };
}"""

# What Forge's setupExtraNetworksForTab does to the page once the pages'
# HTML has arrived: the controls strip in the tab bar, every page's controls
# gathered into it, and the page's live search wired as Forge wires it
# (javascript/extraNetworks.js, applyFilter): on every input, the cards found
# under the tab's own element -- `#<tab>_extra_tabs div.card` -- are hidden
# unless every word typed is in their search terms or description. Run on a
# page already wired by the markup, it only adds the search.
FORGE_WIRES = """() => {
    ["txt2img", "img2img"].forEach((tabname) => {
        const tabs = document.querySelector("#" + tabname + "_extra_tabs");
        const nav = tabs.querySelector(":scope > div.tab-nav");
        let strip = tabs.querySelector(".extra-networks-controls-div");
        if (!strip) {
            strip = document.createElement("DIV");
            strip.classList.add("extra-networks-controls-div");
            nav.appendChild(strip);
        }
        tabs.querySelectorAll(":scope > [id^='" + tabname + "_']").forEach((elem) => {
            const full = elem.id;
            const search = document.getElementById(full + "_extra_search");
            if (!search) return;
            if (!search.dataset.forgeWired) {
                search.dataset.forgeWired = "1";
                const applyFilter = () => {
                    const term = search.value.toLowerCase();
                    gradioApp().querySelectorAll("#" + tabname + "_extra_tabs div.card").forEach((card) => {
                        const text = [...card.querySelectorAll(".search_terms, .description")]
                            .map((t) => t.textContent.toLowerCase()).join(" ");
                        let visible = true;
                        term.split(" ").forEach((partial) => { if (text.indexOf(partial) == -1) visible = false; });
                        card.classList.toggle("hidden", !visible);
                    });
                };
                search.addEventListener("input", applyFilter);
            }
            const controls = document.getElementById(full + "_controls");
            if (controls && controls.parentElement !== strip) strip.appendChild(controls);
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

    def test_the_pages_text_size_reaches_what_lives_outside_gradios_container(self, browser):
        """Gradio sets its body text size on its container alone; the Forge
        Assistant's panel and launcher live on the body, this theme's drawers
        beside the container, and all of them were drawn at the browser's
        16 px -- a seventh larger than the page, the launcher wider, the
        panel's header wrapping. The page the user had set the body."""
        page = open_page(browser)
        found = page.evaluate(MEASURE)

        assert found["bodyFont"] == "14px"
        assert found["outsideFont"] == "14px"
        page.evaluate("opts.model_chain_theme = false")
        changed(page)
        assert page.evaluate(MEASURE)["bodyFont"] == "16px"
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

    def test_applied_to_a_settled_page_it_changes_no_child_list(self, browser):
        """Forge runs every after-update callback 250 ms after any change to
        a child list inside its app element, and `apply` is one of them. The
        first build wrote the title's text and rebuilt the ⋯ menu on every
        apply -- the same words, new nodes -- so each apply raised the next:
        the page updated every 300 ms on every tab for as long as it was
        open, with every extension's callbacks running each time (the user's
        drag log showed the stalls, about 50 ms every 300). Applied again to
        a settled page, the theme changes nothing that observer would hear,
        and the menu still holds the tabs that do not fit."""
        page = open_page(browser, width=820)
        assert page.evaluate(MEASURE)["hidden"], "the ⋯ menu holds tabs at this width"
        page.evaluate(HEARD)
        for _ in range(3):
            page.evaluate("hooks.after.forEach((fn) => fn())")
            settle(page, 6)

        assert page.evaluate("window.heard") == []
        found = page.evaluate(MEASURE)
        assert found["hidden"] and found["menuItems"] == found["hidden"]
        assert found["title"] == "Forge"
        page.close()

    def test_a_tab_renamed_or_a_title_changed_still_reaches_the_bar(self, browser):
        """The quiet apply above is not a frozen one: the title follows the
        setting, and a menu entry follows its tab's name."""
        page = open_page(browser, width=820)
        before = page.evaluate(MEASURE)
        assert "Extras" in before["hidden"] and before["title"] == "Forge"
        page.evaluate("opts.model_chain_theme_title = 'Mine'")
        changed(page)
        assert page.evaluate(MEASURE)["title"] == "Mine"

        page.evaluate("document.querySelector('#tab_extras-button').textContent = 'Extras (3)'")
        page.evaluate("hooks.after.forEach((fn) => fn())")
        settle(page, 6)
        found = page.evaluate(MEASURE)
        assert "Extras (3)" in found["hidden"] and found["menuItems"] == found["hidden"]
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
        # drawer's two controls strips (its pages are never moved): seven,
        # however often.
        assert page.evaluate("document.querySelectorAll('[data-mc-theme-moved]').length") == 7
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
        # The page's own padding, focus mode's 8 px under the theme, is the
        # only margin left.
        assert found["contentLeft"] - found["pageLeft"] == 8
        page.close()

    def test_the_page_meets_the_window_as_focus_mode_does(self, browser):
        """Gradio pads the page 16 px above and below and 32 px at the
        sides, and the Forge Assistant's focus mode insets a workspace 8 px:
        the user's screens showed the content 8 px from the display's edge
        under focus and four times that outside it, and asked for the page
        to match. Under the theme the page's padding is focus mode's inset
        on every side, measured here against the window's edges in both
        modes, and the bar's ground reaches those edges through it."""
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        edges = """() => {
            const q = (s) => document.querySelector(s);
            const width = document.documentElement.clientWidth;
            const prompt = q("#txt2img_prompt").getBoundingClientRect();
            const results = q("#txt2img_results").getBoundingClientRect();
            const nav = q("#tabs > .tab-nav");
            const rect = nav.getBoundingClientRect();
            const ground = getComputedStyle(nav, "::before");
            return {left: prompt.left, right: width - results.right,
                    padding: getComputedStyle(q(".gradio-container")).padding,
                    bar: {top: rect.top, display: getComputedStyle(nav).display,
                          groundLeft: rect.left + parseFloat(ground.left),
                          groundRight: width - (rect.right - parseFloat(ground.right))}};
        }"""
        at_rest = page.evaluate(edges)
        assert at_rest["padding"] == "8px", at_rest
        assert at_rest["left"] == 8 and at_rest["right"] == 8, at_rest
        assert at_rest["bar"]["top"] == 8, at_rest
        assert at_rest["bar"]["groundLeft"] == 0 and at_rest["bar"]["groundRight"] == 0, at_rest

        entered = page.evaluate("forgeAssistant.focus.enter('tab_txt2img', forgeAssistant.shell.host)")
        assert entered["ok"] is True
        settle(page, 8)
        focused = page.evaluate(edges)
        assert focused["bar"]["display"] == "none"
        assert focused["left"] == 8 and focused["right"] == 8, focused
        page.evaluate("forgeAssistant.focus.exit()")
        settle(page, 4)
        assert page.evaluate(edges)["left"] == 8

        # Off, the page has Gradio's own padding back, as the stand-in models it.
        page.evaluate("opts.model_chain_theme = false")
        changed(page)
        off = page.evaluate(edges)
        assert off["padding"] == "16px 32px", off
        assert off["left"] == 32 + 1 + 10, "Gradio's 32 px and the tab panel's own frame: a 1 px line, 10 px of padding"
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
        through the same call, were out of Forge's reach. (The pages are
        Forge's own tab panels in place now, and the drawer's head is a
        child of Forge's tabs element: inside twice over.)"""
        page = open_page(browser)
        where = """() => ({
            quick: !!document.getElementById("mc-theme-quick").closest("gradio-app"),
            networks: !!document.getElementById("mc-theme-networks-txt2img").closest("gradio-app"),
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


CONTROLS_ROW = """() => {
    const q = (s) => document.querySelector(s);
    const ids = ["txt2img_lora_extra_sort_default", "txt2img_lora_extra_sort_name", "txt2img_lora_extra_sort_date_created",
                 "txt2img_lora_extra_sort_date_modified", "txt2img_lora_extra_sort_dir", "txt2img_lora_extra_tree_view",
                 "txt2img_lora_extra_refresh"];
    const boxes = ids.map((id) => document.getElementById(id).getBoundingClientRect());
    const search = q("#txt2img_lora_extra_search").getBoundingClientRect();
    const label = q("#txt2img_lora_controls small");
    const spacers = [...document.querySelectorAll("#txt2img_lora_controls small")].slice(1)
        .map((s) => getComputedStyle(s).display);
    return {tops: boxes.map((b) => Math.round(b.top)), searchBottom: search.bottom,
            label: getComputedStyle(label).display, labelTop: label.getBoundingClientRect().top,
            labelHeight: label.getBoundingClientRect().height,
            buttonHeight: Math.max(...boxes.map((b) => b.height)),
            aside: q(".mc-theme-networks").getBoundingClientRect().width,
            icon: q("#txt2img_lora_extra_sort_default i").getBoundingClientRect().width, spacers};
}"""


class TestTheExtraNetworksDrawer:
    def test_the_pages_stay_forges_own_and_one_shows_in_a_drawer_with_generation_in_view(self, browser):
        """The pages are Forge's tab panels, in place: the drawer's head goes
        inside Forge's own tabs element as one more child, and a page is
        drawn under it when shown. The first build moved the pages into an
        aside outside that element, and Forge's search, which filters
        `#<tab>_extra_tabs div.card`, found nothing from then on."""
        page = open_page(browser)
        found = page.evaluate(NETWORKS)

        assert found["host"]["open"] is False and found["host"]["marked"] is True
        assert found["host"]["layout"] == "drawer" and found["host"]["width"] == "340px"
        assert found["chrome"]["parent"] == "txt2img_extra_tabs" and found["chrome"]["display"] == "none"
        assert found["chrome"]["handle"] is True
        assert found["toggle"] == {"hidden": False, "expanded": "false", "disabled": False}
        assert found["navMarked"] is True and found["navDisplay"] == "none"
        assert found["generationDisplay"] == "block" and found["generationSelected"] == "Generation"
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide mc-theme-networks-nav", "txt2img_generation",
                                              "txt2img_textual_inversion", "txt2img_lora", "mc-theme-networks-txt2img"]
        assert found["ti"]["parent"] == "txt2img_extra_tabs" and found["ti"]["marked"] is True
        assert found["lora"]["parent"] == "txt2img_extra_tabs"
        assert found["img2imgLora"]["parent"] == "img2img_extra_tabs"
        assert found["stripParent"] == "mc-theme-networks-txt2img"
        assert [t["text"] for t in found["tabs"]] == ["Textual Inversion", "Lora", "Lora"]
        assert [t["tab"] for t in found["tabs"]] == ["txt2img", "txt2img", "img2img"]
        # The first page is the one shown, with its controls and no other's;
        # closed, nothing of it is drawn.
        assert found["ti"]["shown"] is True and found["lora"]["shown"] is False
        assert found["ti"]["display"] == "none"
        assert found["tiControls"] == {"parent": "extra-networks-controls-div", "display": ""}
        assert found["loraControls"] == {"parent": "extra-networks-controls-div", "display": "none"}
        assert [t["active"] for t in found["tabs"]] == [True, False, True]

        page.click(".mc-theme-networks-toggle")
        settle(page)
        found = page.evaluate(NETWORKS)
        assert found["host"]["open"] is True and found["toggle"]["expanded"] == "true"
        assert found["chrome"]["display"] == "flex" and found["chrome"]["position"] == "fixed"
        assert found["chrome"]["rect"]["top"] == 64 and found["chrome"]["rect"]["right"] == 1600
        assert found["chrome"]["rect"]["width"] == 340
        head = int(found["host"]["chrome"][:-2])
        assert head > 40 and abs(head - found["chrome"]["rect"]["height"]) <= 1, found
        assert found["ti"]["display"] == "block" and found["ti"]["position"] == "fixed"
        assert found["ti"]["rect"] == {"top": 64 + head, "right": 1600, "width": 340,
                                       "height": 900 - 64 - head}
        assert found["lora"]["display"] == "none"
        assert found["other"]["parent"] == "img2img_extra_tabs" and found["other"]["drawn"] is False
        page.close()

    def test_typing_in_the_search_filters_the_cards_the_way_forge_wired_it(self, browser):
        """The live search is Forge's own: an input event filters the cards
        it finds under `#<tab>_extra_tabs`. Typed into the drawer's search
        box, it hides the card that does not match and shows it again when
        cleared -- which it did not while the pages lived outside that
        element."""
        page = open_page(browser)
        page.evaluate(FORGE_WIRES)
        page.click(".mc-theme-networks-toggle")
        page.click("[data-mc-theme-tab='txt2img'][data-mc-theme-page='txt2img_lora']")
        settle(page)
        cards = "[...document.querySelectorAll('#txt2img_lora_cards .card')].map((c) => [c.dataset.name, c.classList.contains('hidden')])"
        assert page.evaluate(cards) == [["detail", False], ["style", False]]

        page.fill("#txt2img_lora_extra_search", "det")
        settle(page)
        assert page.evaluate(cards) == [["detail", False], ["style", True]]
        page.fill("#txt2img_lora_extra_search", "")
        settle(page)
        assert page.evaluate(cards) == [["detail", False], ["style", False]]
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

        assert page.evaluate(NETWORKS)["host"]["open"] is False
        assert page.evaluate("localStorage.getItem('mc-theme-networks')") == "closed"
        page.click(".mc-theme-networks-toggle")
        settle(page)
        assert page.evaluate("localStorage.getItem('mc-theme-networks')") == "open"
        page.click("#mc-theme-networks-txt2img .mc-theme-networks-close")
        settle(page)
        assert page.evaluate(NETWORKS)["host"]["open"] is False
        assert page.evaluate("localStorage.getItem('mc-theme-networks')") == "closed"
        page.close()

    def test_open_by_default_the_width_and_the_card_size_follow_the_settings(self, browser):
        page = open_page(browser, {"model_chain_theme_networks": "open",
                                   "model_chain_theme_networks_width": 400,
                                   "model_chain_theme_networks_card": 100})
        page.click("[data-mc-theme-tab='txt2img'][data-mc-theme-page='txt2img_lora']")
        settle(page)
        found = page.evaluate(NETWORKS)

        assert found["host"]["open"] is True
        assert found["chrome"]["rect"]["width"] == 400 and found["lora"]["rect"]["width"] == 400
        assert found["card"] == {"width": 100, "height": 150}
        page.close()

    def test_each_generation_tab_has_a_drawer_of_its_own_and_other_tabs_have_none(self, browser):
        """"if user navigates away from text to image or image to image,
        extra networks should close! Don't stay open on none relevant
        pages": each drawer lives inside its tab and goes with it. Nothing
        of it shows on Clipboard; back on Txt2Img it is as it was."""
        page = open_page(browser)
        page.click(".mc-theme-networks-toggle")
        settle(page)
        page.click("#tab_img2img-button")
        settle(page)
        found = page.evaluate(NETWORKS)
        assert found["chrome"]["drawn"] is False and found["other"]["drawn"] is True
        assert found["toggle"] == {"hidden": False, "expanded": "true", "disabled": False}
        assert page.evaluate("getComputedStyle(document.getElementById('img2img_lora')).display") == "block"

        page.click("#tab_clipboard-button")
        settle(page)
        found = page.evaluate(NETWORKS)
        assert found["chrome"]["drawn"] is False and found["other"]["drawn"] is False
        assert found["toggle"]["disabled"] is True
        assert page.evaluate("document.querySelectorAll('.mc-theme-networks-page').length") == 3
        assert page.evaluate("[...document.querySelectorAll('.mc-theme-networks-page')]"
                             ".every((p) => p.getClientRects().length === 0)") is True

        page.click("#tab_txt2img-button")
        settle(page)
        found = page.evaluate(NETWORKS)
        assert found["chrome"]["drawn"] is True and found["host"]["open"] is True
        assert found["ti"]["display"] == "block" and found["toggle"]["disabled"] is False
        page.close()

    def test_a_page_is_taken_only_once_forge_has_wired_it(self, browser):
        """Forge wires the search, sort and refresh of a page after its HTML
        has arrived, later than the host's loaded hook, and looks for the
        pages as its own tabs' children: a page taken before that would
        never be wired. The strip Forge builds for the controls is the sign."""
        page = open_page(browser, controls_built=False)
        found = page.evaluate(NETWORKS)

        assert found["chrome"] is None and found["host"]["marked"] is False
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide", "txt2img_generation",
                                              "txt2img_textual_inversion", "txt2img_lora"]
        assert found["navMarked"] is False and found["navDisplay"] == "flex"
        assert found["tiControls"]["parent"] == "txt2img_textual_inversion_pane"
        assert found["tabs"] == [] and found["toggle"]["disabled"] is True

        page.evaluate(FORGE_WIRES)
        page.evaluate("hooks.after.forEach((fn) => fn())")
        settle(page)
        found = page.evaluate(NETWORKS)
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide mc-theme-networks-nav", "txt2img_generation",
                                              "txt2img_textual_inversion", "txt2img_lora", "mc-theme-networks-txt2img"]
        assert found["stripParent"] == "mc-theme-networks-txt2img"
        assert found["tiControls"] == {"parent": "extra-networks-controls-div", "display": ""}
        assert found["loraControls"] == {"parent": "extra-networks-controls-div", "display": "none"}
        assert [t["text"] for t in found["tabs"]] == ["Textual Inversion", "Lora", "Lora"]
        assert found["toggle"]["disabled"] is False
        page.close()

    def test_the_drawer_draws_a_page_at_its_own_density(self, browser):
        """Forge draws a page for the window's width: the search and the sort
        buttons in one row, a directory as a large button, a card's name at
        1.7 em in bold. At a third of that width the user saw a tangle, a
        wall of buttons and names over the pictures -- and "undefined" where
        the drawer's close button should have been, a glyph the table lacked.
        Then the search box: its magnifier drawn low and loose beside the
        words, and the sort row on its heels ("it needs better spacing").
        The glyph sits in the box's left padding, centred on its line; the
        box is as tall as a button and the row under it a row's space away."""
        page = open_page(browser)
        page.click(".mc-theme-networks-toggle")
        page.click("[data-mc-theme-tab='txt2img'][data-mc-theme-page='txt2img_lora']")
        settle(page)
        found = page.evaluate("""() => {
            const q = (s) => document.querySelector(s);
            const close = q("#mc-theme-networks-txt2img .mc-theme-networks-close");
            const controls = q("#txt2img_lora_controls");
            const search = q("#txt2img_lora_extra_search");
            const wrap = search.parentElement;
            const glyph = getComputedStyle(wrap, "::before");
            const sort = q("#txt2img_lora_extra_sort_dir");
            const dir = q("#txt2img_lora_dirs button");
            const name = q("#txt2img_lora_cards .card .actions .name");
            const px = (node, prop) => parseFloat(getComputedStyle(node)[prop]);
            const searchBox = search.getBoundingClientRect();
            return {closeSvg: !!close.querySelector("svg"), closeText: close.textContent.trim(),
                    searchWidth: searchBox.width, controlsWidth: controls.getBoundingClientRect().width,
                    searchHeight: searchBox.height, searchPaddingLeft: px(search, "paddingLeft"),
                    glyph: {position: glyph.position, top: parseFloat(glyph.top), left: parseFloat(glyph.left),
                            margin: glyph.margin, transform: glyph.transform, wrapHeight: wrap.getBoundingClientRect().height},
                    gapUnder: sort.getBoundingClientRect().top - searchBox.bottom,
                    dirFont: px(dir, "fontSize"), dirHeight: dir.getBoundingClientRect().height,
                    nameFont: px(name, "fontSize"), nameWeight: getComputedStyle(name).fontWeight};
        }""")

        assert found["closeSvg"] is True and found["closeText"] == "", found
        assert found["searchWidth"] >= 0.9 * found["controlsWidth"], found
        assert found["searchHeight"] >= 28 and found["searchPaddingLeft"] >= 24, found
        glyph = found["glyph"]
        assert glyph["position"] == "absolute" and glyph["margin"] == "0px", glyph
        assert abs(glyph["top"] - glyph["wrapHeight"] / 2) <= 1 and glyph["transform"] != "none", glyph
        assert 0 < glyph["left"] <= 12, glyph
        assert found["gapUnder"] >= 6, found
        assert found["dirFont"] <= 12 and found["dirHeight"] <= 28, found
        assert found["nameFont"] <= 11.5 and found["nameWeight"] in ("600", "500"), found
        page.close()

    def test_the_pages_controls_sit_on_one_row_under_the_search(self, browser):
        """Forge's 24 px icons with half-rem spacers came to a few pixels more
        than the drawer, and the tree view and refresh rolled onto a third
        row. One row, with the label (Forge's is a 2.2 rem line) no taller
        than the buttons beside it; and one row at the drawer's narrowest,
        240 px, where the label goes before anything wraps."""
        page = open_page(browser)
        page.click(".mc-theme-networks-toggle")
        page.click("[data-mc-theme-tab='txt2img'][data-mc-theme-page='txt2img_lora']")
        settle(page)
        found = page.evaluate(CONTROLS_ROW)

        assert found["aside"] == 340, found
        assert len(set(found["tops"])) == 1, found
        assert found["tops"][0] >= found["searchBottom"] - 1, found
        assert found["label"] != "none" and abs(found["labelTop"] - found["tops"][0]) <= 8, found
        assert found["labelHeight"] <= found["buttonHeight"] + 1, found
        assert found["icon"] <= 20 and all(d == "none" for d in found["spacers"]), found

        page.evaluate("opts.model_chain_theme_networks_width = 240")
        changed(page)
        narrow = page.evaluate(CONTROLS_ROW)

        assert narrow["aside"] == 240, narrow
        assert len(set(narrow["tops"])) == 1, narrow
        assert narrow["label"] == "none", narrow
        page.close()

    def test_off_puts_the_pages_back_beside_generation(self, browser):
        page = open_page(browser)
        page.click(".mc-theme-networks-toggle")
        settle(page)
        page.evaluate("opts.model_chain_theme_networks = 'off'")
        changed(page)
        found = page.evaluate(NETWORKS)

        assert found["chrome"] is None and found["other"] is None
        assert found["host"] == {"open": False, "marked": False, "layout": None, "display": "block",
                                 "width": "", "chrome": ""}
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide", "txt2img_generation",
                                              "txt2img_textual_inversion", "txt2img_lora"]
        assert found["navMarked"] is False and found["navDisplay"] == "flex"
        assert found["stripParent"] == "tab-nav scroll-hide"
        assert found["tiControls"] == {"parent": "extra-networks-controls-div", "display": "none"}
        assert found["ti"]["marked"] is False and found["ti"]["shown"] is False
        assert found["ti"]["display"] == "none", "Gradio's own inline display again"
        assert found["tabs"] == [] and found["toggle"]["hidden"] is True
        page.close()

    def test_the_column_layout_gives_the_pages_room_beside_the_generation_ui(self, browser):
        """"in lobe I had an option to make this menu become a column and
        push the other column": the column layout makes Forge's tabs element
        two columns, Generation in the first and the drawer's head and page
        in the second, so the results column gives up the drawer's width.
        The head sticks under the header and the page under the head while
        the page scrolls, and the page scrolls inside its own box."""
        page = open_page(browser, {"model_chain_theme_networks": "open"})
        drawer = page.evaluate(COLUMN)
        page.evaluate("opts.model_chain_theme_networks_mode = 'column'")
        changed(page)
        settle(page, 4)
        found = page.evaluate(COLUMN)

        assert drawer["layout"] == "drawer" and drawer["hostDisplay"] != "grid"
        assert found["layout"] == "column" and found["hostDisplay"] == "grid"
        assert found["chrome"]["position"] == "sticky" and found["page"]["position"] == "sticky"
        assert found["chrome"]["width"] == 340 and found["page"]["width"] == 340
        assert found["generation"]["right"] <= found["chrome"]["left"] - 8, found
        assert found["generation"]["width"] <= drawer["generation"]["width"] - 340, "the generation UI gave room"
        assert found["results"]["right"] <= found["chrome"]["left"], found
        assert found["page"]["top"] == found["chrome"]["bottom"], "the page starts under the head"
        assert found["page"]["overflow"] == "auto" and found["page"]["height"] <= 900 - 64

        page.evaluate("window.scrollTo(0, 400)")
        settle(page, 4)
        scrolled = page.evaluate(COLUMN)
        assert scrolled["scrollY"] == 400
        assert scrolled["chrome"]["top"] == 64, "the head sticks under the header"
        assert scrolled["page"]["top"] == scrolled["chrome"]["bottom"], "and the page under the head"
        page.close()

    def test_the_edge_drags_the_width_and_remembers_it_until_the_setting_changes(self, browser):
        """"another resizable column on the page": the drawer's edge drags to
        a width within the setting's bounds, the arrow keys move it too, and
        the width is remembered per browser -- until the width setting
        changes, which then wins."""
        context = browser.new_context(viewport={"width": 1600, "height": 900})
        page = open_page(browser, {"model_chain_theme_networks": "open"}, context=context)
        handle = page.evaluate("(() => { const r = document.querySelector('#mc-theme-networks-txt2img "
                               ".mc-theme-networks-handle').getBoundingClientRect(); "
                               "return [r.left + r.width / 2, r.top + 30]; })()")
        assert abs(handle[0] - (1600 - 340)) <= 1, handle
        page.mouse.move(*handle)
        page.mouse.down()
        page.mouse.move(handle[0] - 60, handle[1], steps=4)
        page.mouse.move(handle[0] - 100, handle[1], steps=4)
        during = page.evaluate(NETWORKS)
        page.mouse.up()
        settle(page)

        assert during["chrome"]["rect"]["width"] == 440 and during["ti"]["rect"]["width"] == 440
        assert page.evaluate("document.body.classList.contains('mc-theme-networks-resizing')") is False
        assert json.loads(page.evaluate("localStorage.getItem('mc-theme-networks-width')")) == {"width": 440, "setting": 340}

        page.focus("#mc-theme-networks-txt2img .mc-theme-networks-handle")
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("ArrowRight")
        settle(page)
        assert page.evaluate(NETWORKS)["chrome"]["rect"]["width"] == 450
        page.close()

        # The same browser, a new page: the dragged width is what it opens at.
        again = open_page(browser, {"model_chain_theme_networks": "open"}, context=context)
        assert again.evaluate(NETWORKS)["chrome"]["rect"]["width"] == 450
        again.close()
        # The setting changed: the setting wins.
        changed_page = open_page(browser, {"model_chain_theme_networks": "open",
                                           "model_chain_theme_networks_width": 400}, context=context)
        assert changed_page.evaluate(NETWORKS)["chrome"]["rect"]["width"] == 400
        changed_page.close()
        context.close()

    def test_the_edge_keeps_the_width_within_the_settings_bounds(self, browser):
        page = open_page(browser, {"model_chain_theme_networks": "open"})
        handle = page.evaluate("(() => { const r = document.querySelector('#mc-theme-networks-txt2img "
                               ".mc-theme-networks-handle').getBoundingClientRect(); "
                               "return [r.left + r.width / 2, r.top + 30]; })()")
        page.mouse.move(*handle)
        page.mouse.down()
        page.mouse.move(handle[0] + 300, handle[1], steps=4)
        narrow = page.evaluate(NETWORKS)["chrome"]["rect"]["width"]
        page.mouse.move(handle[0] - 900, handle[1], steps=4)
        wide = page.evaluate(NETWORKS)["chrome"]["rect"]["width"]
        page.mouse.up()
        settle(page)

        assert narrow == 240 and wide == 800
        page.close()

    def test_under_focus_mode_the_drawer_stays_out_of_the_workspace(self, browser):
        """The drawer lives inside the workspace's tab now; focus mode is the
        workspace alone, as it was when the drawer lived outside it."""
        page = open_page(browser, {"model_chain_theme_networks": "open",
                                   "model_chain_theme_networks_mode": "column"})
        assert page.evaluate(COLUMN)["hostDisplay"] == "grid"
        page.evaluate("document.body.classList.add('forge-assistant-focused')")
        settle(page)
        found = page.evaluate(COLUMN)

        assert found["chrome"] is None and found["page"] is None, found
        assert found["hostDisplay"] == "block"
        page.evaluate("document.body.classList.remove('forge-assistant-focused')")
        settle(page)
        assert page.evaluate(COLUMN)["hostDisplay"] == "grid"
        page.close()


COLUMN = """() => {
    const q = (s) => document.querySelector(s);
    const host = q("#txt2img_extra_tabs");
    const chrome = q("#mc-theme-networks-txt2img");
    const shown = q("#txt2img_extra_tabs > .mc-theme-networks-shown");
    const box = (el) => { if (!el || !el.getClientRects().length) return null;
                          const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
                          return {top: Math.round(r.top), bottom: Math.round(r.bottom), left: Math.round(r.left),
                                  right: Math.round(r.right), width: Math.round(r.width), height: Math.round(r.height),
                                  position: s.position, overflow: s.overflowY}; };
    return {layout: host.getAttribute("data-mc-theme-networks"), hostDisplay: getComputedStyle(host).display,
            chrome: box(chrome), page: box(shown), generation: box(q("#txt2img_generation")),
            results: box(q("#txt2img_results")), scrollY: window.scrollY};
}"""


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


FILLED = """() => {
    const q = (s) => document.querySelector(s);
    const box = (s) => { const r = q(s).getBoundingClientRect(); return {top: r.top, bottom: r.bottom, height: r.height}; };
    const results = q("#txt2img_results");
    const found = forgeAssistant.fill.explain()[0];
    const interrupt = getComputedStyle(q("#txt2img_interrupt"));
    const skip = getComputedStyle(q("#txt2img_skip"));
    return {
        marked: results.hasAttribute("data-forge-assistant-fill"), reason: found.reason, height: found.height,
        header: box("#tabs > .tab-nav"), generate: box("#txt2img_generate_box"), gallery: box("#txt2img_gallery"),
        container: box("#txt2img_gallery_container"), buttons: box("#image_buttons_txt2img"),
        inner: window.innerHeight,
        interrupt: interrupt.backgroundColor, skip: skip.backgroundColor, interruptBorder: interrupt.borderTopColor,
    };
}"""


def luminance(colour: str) -> float:
    """Relative brightness, 0 to 1, of a colour Chromium reports as rgb(a) or color(srgb)."""
    import re as _re
    numbers = [float(n) for n in _re.findall(r"[\d.]+", colour)]
    if colour.startswith("color("):
        r, g, b = numbers[:3]
    else:
        r, g, b = [n / 255 for n in numbers[:3]]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


class TestWithTheForgeAssistant:
    """The Forge Assistant fits the results column to the window (docs/22
    §3.33): one space above the panel, the gallery taking what is left, the
    buttons one space over the view's bottom. On the user's page under the
    theme it fitted nothing: a user.css sized the gallery's container against
    the window, the gallery stopped short of it, and the gap between the
    gallery and its buttons -- which the assistant reads as the theme's one
    space -- was half a window. And where it did fit, it put Generate under
    the header: the bar sits inside the page's padding at scroll 0, and the
    assistant asked only the first pixel of the window what was drawn there.
    The assistant's scripts load before this one's, as on Forge, and nothing
    here measures by hand: the fill's own observers have to."""

    @pytest.mark.parametrize("user_css", [USER_CSS_WHOLE, USER_CSS_CONTAINER], ids=["whole", "container"])
    def test_the_gallery_is_fitted_to_the_window_under_the_header(self, browser, user_css):
        page = open_page(browser, assistant=user_css)
        settle(page, 8)
        found = page.evaluate(FILLED)

        assert found["marked"] is True and found["reason"] == "", found
        space = found["buttons"]["top"] - found["gallery"]["bottom"]
        assert 4 <= space <= 24, found
        # The first kept thing is under the header, not under it: one space
        # and the panel's padding below the bar's bottom edge.
        assert found["generate"]["top"] >= found["header"]["bottom"] + space - 1, found
        assert found["generate"]["top"] <= found["header"]["bottom"] + space + 24, found
        assert abs(found["inner"] - found["buttons"]["bottom"] - space) <= 1, found
        assert found["gallery"]["height"] > 0.5 * found["inner"], found
        page.close()

    def test_scrolling_keeps_the_column_in_view_under_the_bar_and_focus_mode_takes_the_whole_window(self, browser):
        """Normal mode: the settings scroll, the results column is held a
        margin under the bar, never under it. Focus mode: the bar is hidden,
        the workspace is the window, the column is held at its padding, where
        the fill measured it, scrolled or not; leaving focus puts it back."""
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        at_rest = page.evaluate(FILLED)
        page.evaluate("window.scrollTo(0, 600)")
        settle(page, 4)
        scrolled = page.evaluate(FILLED)
        assert page.evaluate("window.scrollY") == 600
        assert scrolled["generate"]["top"] >= scrolled["header"]["bottom"] + 4, scrolled
        assert scrolled["generate"]["top"] < at_rest["generate"]["top"], "held under the bar, not scrolled away"
        assert scrolled["buttons"]["bottom"] <= scrolled["inner"], scrolled
        page.evaluate("window.scrollTo(0, 0)")
        settle(page, 4)

        entered = page.evaluate("forgeAssistant.focus.enter('tab_txt2img', forgeAssistant.shell.host)")
        assert entered["ok"] is True
        settle(page, 8)
        focused = page.evaluate(FILLED)
        root = page.evaluate("(() => { const r = document.querySelector('.forge-assistant-focus-root');"
                             " const b = r.getBoundingClientRect(); return {top: b.top, bottom: b.bottom,"
                             " navDisplay: getComputedStyle(document.querySelector('#tabs > .tab-nav')).display}; })()")
        assert root["navDisplay"] == "none", "the bar is not drawn under focus"
        assert focused["marked"] is True and focused["reason"] == "", focused
        space = focused["buttons"]["top"] - focused["gallery"]["bottom"]
        assert abs(focused["generate"]["top"] - (root["top"] + space + 16)) <= 1, (focused, root)
        assert abs(root["bottom"] - focused["buttons"]["bottom"] - space) <= 1, (focused, root)
        assert focused["gallery"]["height"] > at_rest["gallery"]["height"], "the header's room is the gallery's"

        page.evaluate("document.querySelector('.forge-assistant-focus-root').scrollTop = 600")
        settle(page, 4)
        held = page.evaluate(FILLED)
        assert abs(held["generate"]["top"] - focused["generate"]["top"]) <= 1, (held, focused)
        assert abs(held["buttons"]["bottom"] - focused["buttons"]["bottom"]) <= 1, (held, focused)

        page.evaluate("forgeAssistant.focus.exit()")
        settle(page, 8)
        back = page.evaluate(FILLED)
        assert abs(back["generate"]["top"] - at_rest["generate"]["top"]) <= 1, (back, at_rest)
        page.close()

    def test_the_assistants_header_keeps_its_close_button_on_the_first_row(self, browser):
        """The header's grip is what the row has left once the buttons have
        taken their room, and the memory reading inside it is clipped to that:
        counted, a reading one digit wider put the close button on a second
        row, which the user saw on every page after a long generation."""
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        page.evaluate("forgeAssistant.shell.open()")
        settle(page, 4)
        page.evaluate("""() => {
            const reading = document.querySelector(".forge-assistant-memory");
            reading.textContent = "RAM 177.2  5090 31.4  3090 23.5  4090 12.0  2080 7.5";
        }""")
        settle(page, 4)
        found = page.evaluate("""() => {
            const header = document.querySelector(".forge-assistant-header");
            const buttons = [...header.querySelectorAll("button")].filter((b) => b.getBoundingClientRect().height > 0);
            const close = buttons.find((b) => b.textContent.trim() === "\u2715");
            const rows = new Set(buttons.map((b) => Math.round(b.getBoundingClientRect().top)));
            return {buttons: buttons.length, rows: rows.size, closeFound: !!close,
                    closeTop: close && close.getBoundingClientRect().top, firstTop: buttons[0].getBoundingClientRect().top,
                    headerHeight: header.getBoundingClientRect().height, buttonHeight: buttons[0].getBoundingClientRect().height};
        }""")

        assert found["closeFound"] is True and found["buttons"] >= 3, found
        assert found["rows"] == 1, found
        assert abs(found["closeTop"] - found["firstTop"]) <= 1, found
        assert found["headerHeight"] < 2 * found["buttonHeight"], found
        page.close()

    def test_interrupt_and_skip_are_no_bright_spots(self, browser):
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        found = page.evaluate(FILLED)

        # Forge's own light grey (#b4c0cc, 0.55) and the cancel red are gone;
        # both sit near the page, Interrupt tinted apart from Skip.
        assert luminance(found["skip"]) < 0.2, found["skip"]
        assert luminance(found["interrupt"]) < 0.3, found["interrupt"]
        assert found["interrupt"] != found["skip"]
        assert found["interruptBorder"] != found["interrupt"]
        page.close()


class TestForgesPopup:
    def test_a_cards_metadata_editor_is_a_centred_dense_dialog(self, browser):
        """Forge's popup is the window with a 56 em box inside, which on a
        window not much wider runs past its 90% cap and sits off centre (the
        box's auto margins are nothing once it overflows), with no scrolling
        of its own, so a tall one scrolls the whole popup; its name at 18 pt
        and its fields at prose padding. The page the user had drew it as a
        dialog: centred, 24 px inside a window smaller than Forge's box both
        ways, scrolling inside it, dense."""
        width, height = 720, 560
        page = open_page(browser, width=width, height=height, popup=True)
        found = page.evaluate("""() => {
            const q = (s) => document.querySelector(s);
            const popup = q(".global-popup");
            const innerNode = q(".global-popup-inner");
            const inner = innerNode.getBoundingClientRect();
            const editorNode = q(".edit-user-metadata");
            const editor = editorNode.getBoundingClientRect();
            const px = (node, prop) => parseFloat(getComputedStyle(node)[prop]);
            const td = q(".file-metadata td");
            return {inner: {left: inner.left, right: innerWidth - inner.right, top: inner.top,
                            bottom: innerHeight - inner.bottom, width: inner.width, height: inner.height,
                            scrolls: innerNode.scrollHeight > innerNode.clientHeight + 1},
                    editor: {left: editor.left, right: innerWidth - editor.right, width: editor.width,
                             inside: editor.left >= inner.left - 0.5 && editor.right <= inner.right + 0.5},
                    popupScrolls: popup.scrollHeight > popup.clientHeight + 1,
                    name: px(q(".extra-network-name"), "fontSize"),
                    cellPad: px(td, "paddingTop"), cellBorder: px(td, "borderTopWidth"),
                    gap: px(editorNode, "gap")};
        }""")

        inner, editor = found["inner"], found["editor"]
        assert abs(inner["left"] - inner["right"]) <= 2 and abs(editor["left"] - editor["right"]) <= 2, found
        assert abs(inner["top"] - inner["bottom"]) <= 2, found
        assert inner["width"] <= width - 48 + 0.5 and editor["width"] <= width - 48 + 0.5, found
        assert inner["height"] <= height - 48 + 0.5, found
        assert editor["inside"], found
        # Taller than the window, the dialog scrolls inside; the popup does not.
        assert inner["scrolls"] is True and found["popupScrolls"] is False, found
        assert found["name"] <= 16.5, found
        assert found["cellPad"] <= 2.5 and found["cellBorder"] == 0, found
        assert found["gap"] <= 8, found
        page.close()


# -- a generation in the results column ----------------------------------------- #

OOZE = {"model_chain_style_enable": True, "model_chain_style_theme": "Ooze"}

# What Forge's submit() does (javascript/ui.js, progressbar.js): Interrupt and
# Skip shown over Generate by their inline display, and the bar built and put
# before the gallery's container, in the results panel.
RUN = """(shown) => {
    for (const id of ["txt2img_interrupt", "txt2img_skip"]) document.getElementById(id).style.display = "block";
    const container = document.getElementById("txt2img_gallery_container");
    const bar = document.createElement("div");
    bar.className = "progressDiv";
    bar.style.display = shown ? "block" : "none";
    const inner = document.createElement("div");
    inner.className = "progress";
    inner.style.width = "62%";
    inner.textContent = "Sampling step 5/8 62% ETA: 00:03";
    bar.appendChild(inner);
    container.parentNode.insertBefore(bar, container);
}"""

# And what its end does: the bar removed, Interrupt and Skip hidden again.
END = """() => {
    document.querySelectorAll("#txt2img_results_panel > .progressDiv").forEach((bar) => bar.remove());
    for (const id of ["txt2img_interrupt", "txt2img_skip"]) document.getElementById(id).style.display = "none";
}"""

RUNNING = """() => {
    const q = (s) => document.querySelector(s);
    const box = (node) => { if (!node) return null; const r = node.getBoundingClientRect();
                            return {top: r.top, bottom: r.bottom, left: r.left, right: r.right, height: r.height}; };
    const bar = q("#txt2img_results_panel > .progressDiv");
    const overlay = bar ? bar.querySelector(".mc-ooze-overlay") : null;
    const interrupt = q("#txt2img_interrupt");
    const header = q("#tabs > .tab-nav");
    // What is drawn at the top of the ooze, two pixels in: the header, if it
    // covers the bubbles there.
    let covered = null;
    if (overlay) {
        const r = overlay.getBoundingClientRect();
        const hit = document.elementFromPoint(r.left + r.width / 2, r.top + 2);
        covered = !!(hit && header.contains(hit));
    }
    return {header: box(header), bar: box(bar), overlay: box(overlay), covered,
            barPosition: bar ? getComputedStyle(bar).position : null,
            fillHeight: bar ? bar.querySelector(".progress").getBoundingClientRect().height : null,
            generate: box(q("#txt2img_generate_box")),
            generateDisplay: getComputedStyle(q("#txt2img_generate_box")).display,
            interruptDisplay: getComputedStyle(interrupt).display, interruptInline: interrupt.style.display,
            interruptWidth: interrupt.getBoundingClientRect().width,
            gallery: box(q("#txt2img_gallery")), buttons: box(q("#image_buttons_txt2img")),
            reason: forgeAssistant.fill.explain()[0].reason};
}"""


def running_page(browser, opts=None, **kwargs):
    """The user's page: the theme, the Forge Assistant fitting the column to the
    window (their user.css), Forge's own bar and the progress bar's ooze."""
    return open_page(browser, dict(OOZE, **(opts or {})), assistant=USER_CSS_WHOLE, progress=True,
                     width=kwargs.pop("width", 886), height=kwargs.pop("height", 898), **kwargs)


def start(page, shown=True):
    page.evaluate(RUN, shown)
    settle(page, 8)
    return page.evaluate(RUNNING)


def press(page, x, y):
    page.mouse.click(x, y)
    settle(page, 8)
    return page.evaluate(RUNNING)


class TestAGenerationInTheResultsColumn:
    """Asked for, with a screenshot of a run: "I cannot see the full animation
    for my progress bar. I have it set to ooze so the animation should be
    above. How about this, when there is a progress bar, there should be no
    button row for interrupt and skip. Instead, just the progress bar the
    properly sized gallery and gallery buttons to fill below the area needed
    to render the progress bar and 'ooze' above it. I want it to be compact
    still, less space than what is currently dedicated to the row of button
    and progress bar." Forge draws the bar absolutely, 14 px above the
    results panel -- under the theme's header, which hid its top and every
    bubble rising out of it. Nothing here measures by hand: the fill's own
    observers have to follow the bar arriving and the row going."""

    def test_the_bar_and_its_ooze_are_clear_of_the_header_and_the_row_is_not_drawn(self, browser):
        page = running_page(browser)
        rest = page.evaluate(RUNNING)
        assert rest["generateDisplay"] != "none" and rest["bar"] is None, rest
        found = start(page)

        # The row is not drawn; the buttons keep the display Forge wrote, so its
        # Escape and Alt+Enter still press them and the assistant still reads
        # a run from them.
        assert found["generateDisplay"] == "none", found
        assert found["interruptInline"] == "block" and found["interruptDisplay"] == "block", found
        # The bar in the column's flow, 14 px, its ooze whole under the header.
        assert found["barPosition"] == "relative", found
        assert abs(found["bar"]["height"] - 14) <= 0.5 and abs(found["fillHeight"] - 14) <= 0.5, found
        assert found["overlay"] is not None and found["overlay"]["top"] < found["bar"]["top"] - 30, found
        assert found["overlay"]["top"] >= found["header"]["bottom"] - 0.5, found
        assert found["covered"] is False, found
        # One space between the bar and the gallery, the gallery's buttons where
        # they were, and the gallery the taller for it.
        space = found["buttons"]["top"] - found["gallery"]["bottom"]
        assert abs(found["gallery"]["top"] - found["bar"]["bottom"] - space) <= 1, found
        assert abs(found["buttons"]["bottom"] - rest["buttons"]["bottom"]) <= 1, (rest, found)
        assert found["gallery"]["height"] > rest["gallery"]["height"] + 10, (rest, found)
        # Compact: the bar and its ooze take a fifth less than the row did.
        before = rest["gallery"]["top"] - rest["header"]["bottom"]
        now = found["gallery"]["top"] - found["header"]["bottom"]
        assert now <= 0.8 * before, (before, now)

        # Scrolled, the column held under the bar: still clear of it.
        page.evaluate("window.scrollTo(0, 500)")
        settle(page, 6)
        scrolled = page.evaluate(RUNNING)
        assert page.evaluate("window.scrollY") == 500
        assert scrolled["overlay"]["top"] >= scrolled["header"]["bottom"] - 0.5, scrolled
        assert scrolled["covered"] is False, scrolled
        page.close()

    def test_a_press_on_the_bar_or_its_ooze_brings_the_row_back_and_another_puts_it_away(self, browser):
        """On a touchscreen there is no Escape to press: a press on the bar
        brings Interrupt and Skip back for the rest of the run, and the room
        the ooze rises into counts, because 14 px is short of a finger."""
        page = running_page(browser)
        running = start(page)
        bar = running["bar"]
        middle = (bar["left"] + bar["right"]) / 2

        shown = press(page, middle, (bar["top"] + bar["bottom"]) / 2)
        assert shown["generateDisplay"] != "none" and shown["interruptWidth"] > 0, shown
        assert shown["gallery"]["height"] < running["gallery"]["height"] - 30, (running, shown)
        assert abs(shown["buttons"]["bottom"] - running["buttons"]["bottom"]) <= 1, (running, shown)
        assert shown["generate"]["top"] >= shown["bar"]["bottom"], shown

        hidden = press(page, middle, bar["top"] - 20)
        assert hidden["generateDisplay"] == "none", hidden
        assert abs(hidden["gallery"]["height"] - running["gallery"]["height"]) <= 1, (running, hidden)

        # A control is a control, even one drawn over the ooze -- a stacked
        # page's last setting can be, under the room above the bar -- and
        # beside the bar is not the bar.
        page.evaluate("""(at) => {
            const button = document.createElement("button");
            button.id = "over-the-ooze";
            button.textContent = "a control";
            button.style.cssText = `position: fixed; left: ${at[0] - 40}px; top: ${at[1] - 10}px; width: 80px; height: 20px; z-index: 5`;
            document.querySelector("gradio-app").appendChild(button);
        }""", [middle, bar["top"] - 20])
        settle(page)
        assert press(page, middle, bar["top"] - 20)["generateDisplay"] == "none"
        assert press(page, bar["left"] - 20, (bar["top"] + bar["bottom"]) / 2)["generateDisplay"] == "none"
        page.close()

    def test_the_end_of_a_run_brings_generate_back_and_the_next_run_starts_without_the_row(self, browser):
        page = running_page(browser)
        rest = page.evaluate(RUNNING)
        running = start(page)
        bar = running["bar"]
        press(page, (bar["left"] + bar["right"]) / 2, (bar["top"] + bar["bottom"]) / 2)

        page.evaluate(END)
        settle(page, 8)
        done = page.evaluate(RUNNING)
        assert done["generateDisplay"] != "none" and done["bar"] is None, done
        assert abs(done["gallery"]["height"] - rest["gallery"]["height"]) <= 1, (rest, done)
        assert abs(done["generate"]["top"] - rest["generate"]["top"]) <= 1, (rest, done)

        # The press was the last run's: the mark went with its bar.
        again = start(page)
        assert again["generateDisplay"] == "none", again
        page.close()

    def test_a_bar_forge_hides_leaves_the_row(self, browser):
        """Settings, "Show progressbar" off: Forge builds the bar hidden, and a
        run with nothing to show keeps its Interrupt and Skip."""
        page = running_page(browser)
        rest = page.evaluate(RUNNING)
        found = start(page, shown=False)

        assert found["generateDisplay"] != "none", found
        assert abs(found["gallery"]["height"] - rest["gallery"]["height"]) <= 1, (rest, found)
        page.close()

    def test_a_bar_with_nothing_above_it_sits_one_space_under_the_header(self, browser):
        """No ooze, no room: the panel one space under the header and the bar
        at its padding, as Generate sits at rest."""
        page = running_page(browser, {"model_chain_style_theme": "Flat"})
        rest = page.evaluate(RUNNING)
        found = start(page)

        assert found["overlay"] is None and found["generateDisplay"] == "none", found
        assert abs(found["bar"]["top"] - rest["generate"]["top"]) <= 1, (rest, found)
        page.close()

    def test_under_focus_the_ooze_is_inside_the_workspace(self, browser):
        page = running_page(browser)
        start(page)
        entered = page.evaluate("forgeAssistant.focus.enter('tab_txt2img', forgeAssistant.shell.host)")
        assert entered["ok"] is True
        settle(page, 8)
        found = page.evaluate(RUNNING)
        root = page.evaluate("(() => { const r = document.querySelector('.forge-assistant-focus-root')"
                             ".getBoundingClientRect(); return {top: r.top, bottom: r.bottom}; })()")

        assert found["reason"] == "", found
        assert found["overlay"]["top"] >= root["top"] - 0.5, (found, root)
        assert found["generateDisplay"] == "none", found
        space = found["buttons"]["top"] - found["gallery"]["bottom"]
        assert abs(root["bottom"] - found["buttons"]["bottom"] - space) <= 1, (found, root)
        page.evaluate("forgeAssistant.focus.exit()")
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
        assert page.evaluate(NETWORKS)["chrome"] is None
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
        assert networks["chrome"] is None and networks["host"]["marked"] is False
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
