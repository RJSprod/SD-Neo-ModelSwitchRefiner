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
import time
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
.resize-handle { position: relative; cursor: col-resize; grid-column: 2 / 3; min-width: 16px !important; max-width: 16px !important; height: 100%; }
/* Forge Neo's own (style.css): the results columns sticky by id, half an em
   from the top, which outranks any attribute selector; and its rows'
   align-content, which keeps a grid's row track at its content height. The
   first build of the one scroller passed without these and held the column
   under the header on the user's page. */
#img2img_results, #txt2img_results, #extras_results { position: sticky; top: 0.5em; }
.unequal-height { align-content: flex-start; }
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

# The same with one unbroken run a few windows long, as an infotext has
# (a hashes record, a path, a prompt without a space): what the paragraph
# kept out of view is as wide as, if its overflow shows.
WIDE_INFOTEXT = INFOTEXT_HTML.replace("Size: 1024x1024</p>",
                                      'Size: 1024x1024, Hashes: {"model":"' + "0123456789abcdef" * 24 + '"}</p>')


# Each card's sort keys, as Forge's pages write them (`data-sort-*`, from
# ExtraNetworksPage.get_sort_keys and the page's own `default`, its place in
# the listing). Chosen so that every press of a sort control in the drawer's
# test changes the order the previous press left.
SORT_KEYS = {"bad_hands": {"default": 0, "name": "bad_hands", "date_created": 50, "date_modified": 50},
             "detail": {"default": 0, "name": "detail", "date_created": 200, "date_modified": 400},
             "style": {"default": 1, "name": "style", "date_created": 100, "date_modified": 300}}


def controls_html(tabname: str, page: str) -> str:
    """html/extra-networks-pane.html: the page's own controls, hidden until
    Forge's script gathers them into the tab strip and shows the page's.
    Every sort control calls Forge's own handler, and the one Forge's
    *Default sorting method* names (Path, its default) is drawn pressed."""
    full = f"{tabname}_{page}"
    sorts = "".join(f'<div id="{full}_extra_sort_{key}" class="extra-network-control--sort'
                    f'{" extra-network-control--enabled" if key == "default" else ""}" data-sortkey="{key}" '
                    f'onclick="extraNetworksControlSortOnClick(event, \'{tabname}\', \'{page}\');">'
                    f'<i class="extra-network-control--icon extra-network-control--sort-icon"></i></div>'
                    for key in ("default", "name", "date_created", "date_modified"))
    return (f'<div class="extra-network-control" id="{full}_controls" style="display:none">'
            f'<div class="extra-network-control--search"><input id="{full}_extra_search" '
            f'class="extra-network-control--search-text" type="search" placeholder="Search"></div>'
            f'<small>Sort: </small>{sorts}<small> </small>'
            f'<div id="{full}_extra_sort_dir" class="extra-network-control--sort-dir" data-sortdir="Ascending" '
            f'onclick="extraNetworksControlSortDirOnClick(event, \'{tabname}\', \'{page}\');">'
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
    """A page's pane. Forge renders a page twice -- empty when the UI is
    built, whole when the page loads (modules/ui_extra_networks.py,
    `create_html(empty=True)`, then `interface.load`) -- and its script wires
    the first render's controls into the tab strip; the second render brings
    its own controls, hidden, under the same ids. So a page Forge has wired
    still holds a hidden copy of its controls, as the user's page does: the
    copy Forge's sort found first once the drawer's head came after the
    pages."""
    full = f"{tabname}_{page}"
    cards = "".join(
        f'<div class="card" style="width: 256px; height: 384px" data-name="{name}" '
        + "".join(f'data-sort-{key}="{value}" ' for key, value in SORT_KEYS[name].items())
        + f'onclick="cardClicked(\'{tabname}\', \'{html_module.escape(added)}\')">'
        f'<div class="actions"><div class="additional"><span class="search_terms">{name}</span></div>'
        f'<span class="name">{name}</span></div></div>'
        for name, added in CARDS[page])
    dirs = "".join(f'<button class="lg secondary gradio-button custom-button">{name}</button>'
                   for name in ("all", "krea2\\", "kreaCELEB\\", "kreat\\"))
    return (f'<div id="{full}_pane" class="extra-network-pane">'
            + controls_html(tabname, page)
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
              progress: bool = False, grid_row: bool = False, infotext: str = INFOTEXT_HTML) -> str:
    """`assistant` is a user.css to put on a page that also runs the Forge
    Assistant's scripts, in Forge's order (its folder's scripts load before
    this one's), with Gradio's gallery markup for its fill to measure.
    `progress` adds Forge's own progress bar rules and this extension's
    progress-bar appearance script, for a page that runs a generation.
    `grid_row` draws Txt2Img's row as Forge's resize-handle script leaves it:
    a grid of the two columns and the handle between them, its template
    written inline. `infotext` is Forge's generation-info paragraph; its
    block carries the inline style Gradio 4.40's Block writes on every block
    (`overflow` from `allow_overflow`, visible for an HTML component)."""
    buttons = "".join(
        f'<button role="tab" id="{tab}-button" aria-controls="{tab}" '
        f'aria-selected="{"true" if i == 0 else "false"}" class="{"selected" if i == 0 else ""}">{label}</button>'
        for i, (tab, label) in enumerate(TABS))
    panels = "".join(
        f'<div id="{tab}" class="tabitem" role="tabpanel" style="display: none"></div>'
        for tab, _ in TABS[2:] if tab != "tab_settings")
    blocks = "".join(f'<div class="block" id="setting_{n}">Setting {n}</div>' for n in range(settings_blocks))
    row_open = ('<div class="row resize-handle-row unequal-height" style="display: grid; gap: 0; grid-template-columns: 1fr 16px 1fr">'
                if grid_row else '<div class="row unequal-height">')
    handle = '<div class="resize-handle"></div>' if grid_row else ""
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
        {row_open}
          <div id="txt2img_settings" class="column">{blocks}</div>
          {handle}
          <div id="txt2img_results" class="column">
            <div id="txt2img_results_panel" class="column panel">
              <div id="txt2img_gallery_container" class="gr-group">
                {'<div class="styler"><div id="txt2img_gallery" class="block gradio-gallery"><div class="grid-wrap fixed-height"><div class="grid-container"></div></div></div></div>' if assistant else '<div id="txt2img_gallery" class="block gradio-gallery" style="height: 300px"></div>'}
              </div>
              <div id="image_buttons_txt2img" class="row image-buttons">
                <button id="txt2img_open_folder" class="gradio-button tool" title="Open images output directory">📂</button>
                <button id="save_txt2img" class="gradio-button tool" title="Save the image">💾</button>
              </div>
              <div id="html_info_txt2img" class="block infotext" style="border-style: solid; overflow: visible; min-width: min(0px, 100%);"><div class="prose">{infotext}</div></div>
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
        <div id="img2img_results" class="column"><div id="img2img_generate_box" class="row"><button id="img2img_generate">Generate</button></div><div id="img2img_results_panel" class="column panel"><div id="img2img_gallery_container" class="gr-group"><div id="img2img_gallery" class="block"></div></div><div id="html_info_img2img" class="block infotext" style="border-style: solid; overflow: visible; min-width: min(0px, 100%);"><div class="prose"></div></div></div></div>
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
              controls_built=True, assistant="", popup=False, progress=False, grid_row=False,
              settings_blocks=40, infotext=INFOTEXT_HTML):
    """A page of its own context, or one in a context given -- two pages of
    one context share a localStorage, which is what the first paint reads."""
    owner = context or browser.new_context(viewport={"width": width, "height": height})
    page = owner.new_page()
    html = page_html(opts=dict(ON, **(opts or {})), lobe=lobe, controls_built=controls_built,
                     assistant=assistant, popup=popup, progress=progress, grid_row=grid_row,
                     settings_blocks=settings_blocks, infotext=infotext)
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
# HTML has arrived (javascript/extraNetworks.js): the controls strip in the
# tab bar, every page's controls gathered into it, and each page's live
# search and sort wired as Forge wires them. The search (applyFilter): on
# every input, the cards found under the tab's own element --
# `#<tab>_extra_tabs div.card` -- are hidden unless every word typed is in
# their search terms or description. The sort (applySort, and the onclick
# handlers the pane's template names): a sort control marks itself pressed
# among its siblings and asks for the page's sort a moment later, which
# reads the pressed control with `gradioApp().querySelector('#<page>_controls
# .extra-network-control--sort.extra-network-control--enabled')` -- the
# first in the document -- and the direction from the control it found when
# it wired the page, sorts the cards on their `data-sort-*` key (numbers as
# numbers) and puts them back into `#<page>_cards`. And what a page's own
# tab runs when it is selected (applyExtraNetworkFilter, a moment later): the
# page's filter, then its sort, both forced. Run on a page already wired by
# the markup, it only adds the search and the sort, finding the wired
# controls in the strip as Forge did before anything else existed.
FORGE_WIRES = """() => {
    window.extraNetworksApplySort = window.extraNetworksApplySort || {};
    window.extraNetworksApplyFilter = window.extraNetworksApplyFilter || {};
    window.applyExtraNetworkFilter = function (full) {
        setTimeout(() => { const apply = extraNetworksApplyFilter[full]; if (apply) apply(true); }, 1);
    };
    window.extraNetworksControlSortOnClick = function (event, tabname, page) {
        const self = event.currentTarget;
        self.parentElement.querySelectorAll(".extra-network-control--sort")
            .forEach((x) => x.classList.remove("extra-network-control--enabled"));
        self.classList.add("extra-network-control--enabled");
        setTimeout(() => extraNetworksApplySort[tabname + "_" + page](true), 1);
    };
    window.extraNetworksControlSortDirOnClick = function (event, tabname, page) {
        const dir = event.currentTarget;
        dir.dataset.sortdir = dir.dataset.sortdir == "Ascending" ? "Descending" : "Ascending";
        setTimeout(() => extraNetworksApplySort[tabname + "_" + page](true), 1);
    };
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
            const controls = strip.querySelector("#" + full + "_controls")
                || document.getElementById(full + "_controls");
            if (!controls) return;
            const search = controls.querySelector("#" + full + "_extra_search");
            const sortDir = controls.querySelector("#" + full + "_extra_sort_dir");
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
                extraNetworksApplyFilter[full] = (force) => { applyFilter(); extraNetworksApplySort[full](force); };
                extraNetworksApplySort[full] = () => {
                    const cards = gradioApp().querySelectorAll("#" + full + " div.card");
                    const parent = gradioApp().querySelector("#" + full + "_cards");
                    const pressed = gradioApp().querySelector(
                        "#" + full + "_controls .extra-network-control--sort.extra-network-control--enabled");
                    const key = pressed ? pressed.dataset.sortkey : "default";
                    const field = "sort" + key.charAt(0).toUpperCase() + key.slice(1);
                    const sorted = Array.from(cards).sort((a, b) => {
                        const x = a.dataset[field];
                        const y = b.dataset[field];
                        if (!isNaN(x) && !isNaN(y)) return parseInt(x) - parseInt(y);
                        return x < y ? -1 : (x > y ? 1 : 0);
                    });
                    if (sortDir.dataset.sortdir == "Descending") sorted.reverse();
                    parent.innerHTML = "";
                    sorted.forEach((card) => parent.appendChild(card));
                };
            }
            if (controls.parentElement !== strip) strip.appendChild(controls);
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


# The page's controls in the drawer's head -- not the hidden copy Forge's
# second render leaves in the page under the same ids.
CONTROLS_ROW = """() => {
    const head = document.querySelector(".mc-theme-networks .extra-networks-controls-div");
    const q = (s) => head.querySelector(s) || document.querySelector(s);
    const ids = ["txt2img_lora_extra_sort_default", "txt2img_lora_extra_sort_name", "txt2img_lora_extra_sort_date_created",
                 "txt2img_lora_extra_sort_date_modified", "txt2img_lora_extra_sort_dir", "txt2img_lora_extra_tree_view",
                 "txt2img_lora_extra_refresh"];
    const boxes = ids.map((id) => head.querySelector("#" + id).getBoundingClientRect());
    const search = q("#txt2img_lora_extra_search").getBoundingClientRect();
    const label = q("#txt2img_lora_controls small");
    const spacers = [...head.querySelectorAll("#txt2img_lora_controls small")].slice(1)
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
        # The head right after Forge's tab strip and ahead of the pages,
        # where Forge's own controls strip is in the document.
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide mc-theme-networks-nav", "mc-theme-networks-txt2img",
                                              "txt2img_generation", "txt2img_textual_inversion", "txt2img_lora"]
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

    SORTED = """() => [...document.querySelectorAll('#txt2img_lora_cards .card')].map((c) => c.dataset.name)"""

    @pytest.mark.parametrize("layout", ["drawer", "column"])
    def test_the_sort_buttons_sort_the_page_forge_rendered_twice(self, browser, layout):
        """"the sort buttons don't work!" Forge renders a page twice and
        wires the first render's controls into its strip, so every page
        keeps a hidden copy of its controls under the same ids, and Forge's
        sort reads the pressed sort button with querySelector: the first in
        the document. The drawer's head, holding the wired strip, came after
        the pages, so the sort read the hidden copy's button -- the default,
        never pressed -- and every sort button did nothing; only the
        direction, which Forge holds from when it wired the page, still
        turned the list round. The head goes before the pages, where Forge's
        strip was: each press here changes the order the last one left."""
        page = open_page(browser, {"model_chain_theme_networks": "open",
                                   "model_chain_theme_networks_mode": layout})
        page.evaluate(FORGE_WIRES)
        page.click("[data-mc-theme-tab='txt2img'][data-mc-theme-page='txt2img_lora']")
        settle(page)
        assert page.evaluate("document.querySelectorAll('[id=\"txt2img_lora_controls\"]').length") == 2, (
            "the page's hidden second render is there, as on the user's page")
        assert page.evaluate(self.SORTED) == ["detail", "style"]

        def press(control):
            page.click(".mc-theme-networks .extra-networks-controls-div #txt2img_lora_extra_" + control)
            page.wait_for_timeout(20)
            settle(page)
            return page.evaluate(self.SORTED)

        assert press("sort_date_created") == ["style", "detail"], "date created, ascending"
        assert press("sort_name") == ["detail", "style"], "name, ascending"
        assert press("sort_dir") == ["style", "detail"], "name, descending"
        assert press("sort_date_modified") == ["detail", "style"], "date modified, descending"
        assert press("sort_default") == ["style", "detail"], "the listing's order, descending"
        pressed = page.evaluate("[...document.querySelectorAll('.mc-theme-networks #txt2img_lora_controls "
                                ".extra-network-control--sort.extra-network-control--enabled')]"
                                ".map((b) => b.dataset.sortkey)")
        assert pressed == ["default"]
        page.close()

    def test_a_page_rendered_again_is_shown_sorted_as_forges_settings_say(self, browser):
        """"default sort to newest created to oldest": Forge's own settings
        say so now, and Forge sorts a page when its own tab is selected --
        but its load renders the cards in the listing's order, and the
        drawer selects no tab of Forge's, so the page showed the listing's
        order until a search or a press. The drawer asks Forge's own filter
        and sort once for each rendering of the cards it shows, and not on
        every update, where the sort's rewrite of the cards would raise the
        next one."""
        page = open_page(browser, {"model_chain_theme_networks": "open"})
        page.evaluate(FORGE_WIRES)
        page.click("[data-mc-theme-tab='txt2img'][data-mc-theme-page='txt2img_lora']")
        settle(page)
        # Forge's settings, as the user's read now: Date Created, Descending.
        page.evaluate("""() => {
            const controls = document.querySelector('.mc-theme-networks #txt2img_lora_controls');
            controls.querySelectorAll('.extra-network-control--sort').forEach((b) =>
                b.classList.toggle('extra-network-control--enabled', b.dataset.sortkey === 'date_created'));
            controls.querySelector('#txt2img_lora_extra_sort_dir').dataset.sortdir = 'Descending';
            window.asked = [];
            const ask = window.applyExtraNetworkFilter;
            window.applyExtraNetworkFilter = (full) => { window.asked.push(full); ask(full); };
        }""")
        # Forge's load renders the page again, its cards in the listing's order.
        page.evaluate("""(pane) => {
            document.getElementById('txt2img_lora_cards_html').innerHTML = pane;
            const cards = document.getElementById('txt2img_lora_cards');
            [...cards.children].reverse().forEach((card) => cards.appendChild(card));
        }""", pane_html("txt2img", "lora", controls_built=True))
        assert page.evaluate(self.SORTED) == ["style", "detail"], "the listing's order"

        changed(page)
        page.wait_for_timeout(20)
        settle(page)
        assert page.evaluate(self.SORTED) == ["detail", "style"], "newest created first"
        # The page rendered again, and the other tab's drawer's page, seen
        # for the first time since Forge's script could sort it.
        assert page.evaluate("window.asked") == ["txt2img_lora", "img2img_lora"]
        page.evaluate(HEARD)
        changed(page)
        changed(page)
        page.wait_for_timeout(20)
        assert page.evaluate("window.asked") == ["txt2img_lora", "img2img_lora"], "once a rendering"
        assert page.evaluate("window.heard") == [], "and no update raised by it"
        page.close()

    def test_the_head_goes_back_before_the_pages_and_otherwise_stays_put(self, browser):
        """Whatever moves the head after a page, the next update puts it
        back after Forge's tab strip; an update that finds it there moves
        nothing (a move is a change to a child list, and one on every update
        would raise Forge's next update itself)."""
        page = open_page(browser)
        order = "[...document.querySelector('#txt2img_extra_tabs').children].map((c) => c.id || c.className)"
        placed = page.evaluate(order)
        assert placed.index("mc-theme-networks-txt2img") == 1
        page.evaluate("document.querySelector('#txt2img_extra_tabs')"
                      ".appendChild(document.getElementById('mc-theme-networks-txt2img'))")
        assert page.evaluate(order)[-1] == "mc-theme-networks-txt2img"
        changed(page)
        assert page.evaluate(order) == placed
        page.evaluate(HEARD)
        changed(page)
        changed(page)
        assert page.evaluate("window.heard") == [], "nothing moved twice"
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
        # The head right after Forge's tab strip and ahead of the pages,
        # where Forge's own controls strip is in the document.
        assert found["extraTabsChildren"] == ["tab-nav scroll-hide mc-theme-networks-nav", "mc-theme-networks-txt2img",
                                              "txt2img_generation", "txt2img_textual_inversion", "txt2img_lora"]
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
        # The edge runs down the shown page's side, above it: the head comes
        # before the pages in the document, a level above the page.
        over_page = page.evaluate("""(x) => { const shown = document.querySelector('.mc-theme-networks-shown')
            .getBoundingClientRect(); const hit = document.elementFromPoint(x, shown.top + 100);
            return hit && hit.className; }""", handle[0])
        assert over_page == "mc-theme-networks-handle", over_page
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


WIDTH = """() => {
    const d = document.documentElement;
    const source = document.querySelector("#html_info_txt2img");
    let widest = 0;
    let name = "";
    document.querySelectorAll("body *").forEach((node) => {
        // The paragraph's own line is clipped, not laid out narrower; a fixed
        // box is held to the window and widens nothing.
        if (node.closest(".mc-theme-info-source") || getComputedStyle(node).position === "fixed") return;
        const box = node.getBoundingClientRect();
        if (box.width > 0 && box.right > widest) { widest = box.right; name = node.tagName.toLowerCase() + (node.id ? "#" + node.id : ""); }
    });
    return {scrollWidth: d.scrollWidth, clientWidth: d.clientWidth, innerWidth: window.innerWidth,
            sourceOverflow: getComputedStyle(source).overflow, widest, name};
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

    def test_forges_paragraph_out_of_view_widens_nothing(self, browser):
        """Gradio writes every block's `overflow` inline (visible, for an HTML
        component), and that beat the paragraph's own `overflow: hidden`: a
        box one pixel wide whose overflow shows is as wide as its longest
        line -- under `white-space: nowrap`, the whole infotext on one line.
        The page was that much wider than the window: a scrollbar on a
        desktop, and on a phone a layout viewport the page's width (see
        TestOnAPhonesBrowser). The user's second run of the one scroller."""
        page = open_page(browser, infotext=WIDE_INFOTEXT)
        found = page.evaluate(WIDTH)

        assert found["sourceOverflow"] == "hidden", found
        assert found["scrollWidth"] <= found["clientWidth"], found
        assert found["widest"] <= found["clientWidth"] + 1, found
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
        """Normal mode: the settings scroll, the results column is held where
        it rests, a margin under the bar and never under it (it used to creep
        up to that margin first; the one-scroller rule locks it). Focus mode:
        the bar is hidden, the workspace is the window, the column is held at
        its padding, where the fill measured it, scrolled or not; leaving
        focus puts it back."""
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        at_rest = page.evaluate(FILLED)
        page.evaluate("window.scrollTo(0, 600)")
        settle(page, 4)
        scrolled = page.evaluate(FILLED)
        assert page.evaluate("window.scrollY") == 600
        assert scrolled["generate"]["top"] >= scrolled["header"]["bottom"] + 4, scrolled
        assert abs(scrolled["generate"]["top"] - at_rest["generate"]["top"]) <= 0.5, "locked where it rests"
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


SCROLLED = """() => {
    const q = (s) => document.querySelector(s);
    const box = (node) => { const r = node.getBoundingClientRect(); return {top: r.top, bottom: r.bottom, height: r.height}; };
    const results = q("#txt2img_results");
    const settings = q("#txt2img_settings");
    const row = results.parentElement;
    const root = q(".forge-assistant-focus-root");
    const page = q(".gradio-container");
    const viewBottom = root
        ? root.getBoundingClientRect().top + root.clientTop + root.clientHeight - parseFloat(getComputedStyle(root).paddingBottom)
        : document.documentElement.clientHeight - parseFloat(getComputedStyle(page).paddingBottom);
    return {
        scrolled: root ? root.scrollTop : window.scrollY, viewBottom,
        settings: box(settings), results: box(results), generate: box(q("#txt2img_generate_box")), row: box(row),
        attribute: row.getAttribute("data-mc-theme-scroll"),
        rowMin: row.style.getPropertyValue("--mc-theme-scroll-row"),
        hold: row.style.getPropertyValue("--mc-theme-scroll-hold"),
        resultsHold: results.style.getPropertyValue("--mc-theme-results-hold"),
        settingsPosition: getComputedStyle(settings).position,
        settingsHeld: settings.classList.contains("mc-theme-scroll-settings"),
        settingsZ: getComputedStyle(settings).zIndex,
        settingsOverscroll: getComputedStyle(settings).overscrollBehaviorY,
        resultsPosition: getComputedStyle(results).position, resultsTop: getComputedStyle(results).top,
        buttons: [...q("#image_buttons_txt2img").children].map((b) => b.id),
        appended: document.querySelectorAll("#image_buttons_txt2img > :not([id^='txt2img_']):not([id^='save_'])").length,
    };
}"""


def scroll_to(page, y: float, *, root: bool = False) -> dict:
    """Scroll the page (or focus mode's workspace) to `y`, instantly, and read
    the columns where the browser has drawn them."""
    if root:
        page.evaluate("(y) => { document.querySelector('.forge-assistant-focus-root').scrollTop = y; }", y)
    else:
        page.evaluate("(y) => window.scrollTo(0, y)", y)
    settle(page, 2)
    return page.evaluate(SCROLLED)


def phases(rest: dict) -> tuple:
    """From the columns at rest: the scroll at which the settings' bottom edge
    meets the view's bottom, and the results' part under the view -- the
    length of each phase."""
    first = rest["settings"]["bottom"] - rest["viewBottom"]
    second = rest["results"]["bottom"] - rest["viewBottom"]
    return first, second


def one_column_at_a_time(page, rest: dict, *, root: bool = False, step: int = 97) -> None:
    """Sampled every `step` pixels from the start to the results' end, each
    step's movement is one column's or the other's and adds up to the step:
    neither both at once (a sum of two steps) nor a dead stretch where nothing
    moves (a sum short of the step)."""
    first, second = phases(rest)
    before = scroll_to(page, 0, root=root)
    y = 0
    while y < first + second:
        y = min(y + step, first + second)
        after = scroll_to(page, y, root=root)
        moved_settings = before["settings"]["top"] - after["settings"]["top"]
        moved_results = before["generate"]["top"] - after["generate"]["top"]
        actual = after["scrolled"] - before["scrolled"]
        assert moved_settings >= -0.5 and moved_results >= -0.5, (y, before, after)
        assert abs(moved_settings + moved_results - actual) <= 1, (y, moved_settings, moved_results, actual)
        before = after


def finger(cdp, kind, x, y, t):
    """One touch event through the DevTools protocol, with its timestamp: the
    browser reads a fling off the last movements' speed."""
    points = [] if kind == "touchEnd" else [{"x": x, "y": y, "radiusX": 1, "radiusY": 1, "force": 1, "id": 1}]
    cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": points, "timestamp": t})


def flick(page, cdp, x, y, travel, *, steps=6, dt=0.016):
    """A finger put down at (x, y), moved `travel` pixels up the screen (down
    the page; negative for the other way) in `steps` quick moves, and lifted:
    a fling, which Chromium's browser process then scrolls on by itself."""
    t = time.time()
    finger(cdp, "touchStart", x, y, t)
    for i in range(1, steps + 1):
        t += dt
        finger(cdp, "touchMove", x, y - travel * i / steps, t)
    t += dt
    finger(cdp, "touchEnd", x, 0, t)


def drag(page, cdp, x, y, travel, *, steps=20, dt=0.04, rest=0.4):
    """A slow finger, held still before it lifts: a drag, no fling."""
    t = time.time()
    finger(cdp, "touchStart", x, y, t)
    for i in range(1, steps + 1):
        t += dt
        finger(cdp, "touchMove", x, y - travel * i / steps, t)
        page.wait_for_timeout(int(dt * 1000))
    page.wait_for_timeout(int(rest * 1000))
    finger(cdp, "touchEnd", x, 0, t + rest)


def settled_fling(page):
    """Wait for the break to let the page go again (the fling's end), then
    read the columns."""
    page.wait_for_function("() => !mcTheme.explainBrake().flinging", timeout=5000)
    settle(page, 3)
    return page.evaluate(SCROLLED)


BRAKE = """() => ({
    brake: mcTheme.explainBrake(),
    clipped: document.querySelectorAll("[data-mc-theme-scroll-clip]").length,
    locked: document.querySelectorAll("[data-mc-theme-scroll-lock]").length,
    scrollHeight: document.documentElement.scrollHeight,
    htmlOverflow: document.documentElement.style.overflow,
    containerMaxHeight: document.querySelector(".gradio-container").style.maxHeight,
})"""


class TestTheMomentumBreak:
    """Asked for: "If I flick scroll to reach the bottom of column one, I do
    not want it to continue momentum into column two. Instead it should stop
    at the bottom like it hit an edge. From that bottom position, I should be
    able to scroll to see what's above, or scroll to see what's below." A
    fling is the browser's own scrolling after the finger has lifted; the
    page gives it an edge at the boundary for the fling's time (the theme
    script, "the momentum break between the two phases"): downward the
    scroller's range is cut to end at the boundary, upward a fling read
    crossing it is put back and the scroller held, and the page is whole
    again when the scroll ends. Fingers are driven through the DevTools
    protocol with timestamps, from which Chromium's browser process
    synthesizes the fling, exactly as a touch screen's would be."""

    def open(self, browser, **kwargs):
        context = browser.new_context(viewport={"width": 1600, "height": 900}, has_touch=True)
        page = open_page(browser, assistant=USER_CSS_WHOLE, context=context, **kwargs)
        settle(page, 8)
        cdp = context.new_cdp_session(page)
        x = page.evaluate("document.querySelector('#txt2img_settings').getBoundingClientRect().left + 100")
        return page, cdp, x

    def test_the_boundary_is_measured_where_the_settings_phase_ends(self, browser):
        page, cdp, x = self.open(browser)
        rest = scroll_to(page, 0)
        first, second = phases(rest)
        explained = page.evaluate("mcTheme.explainScroll()[0]")
        assert abs(explained["boundary"] - first) <= 1, (explained, first)
        whole = page.evaluate(BRAKE)
        assert whole["brake"] == {"installed": True, "fingers": 0, "flinging": False, "clipped": False, "locked": False}
        assert whole["clipped"] == 0 and whole["locked"] == 0
        page.close()

    def test_a_flick_down_the_settings_stops_at_the_boundary_like_an_edge(self, browser):
        page, cdp, x = self.open(browser)
        rest = scroll_to(page, 0)
        first, second = phases(rest)
        whole = page.evaluate(BRAKE)["scrollHeight"]

        # A flick strong enough to carry well into the results' phase: on a
        # page without the break the same flick ends hundreds of pixels past
        # the boundary (the probe that settled the design: 2818 from 0 for a
        # 400 px flick, against a boundary of 1402).
        flick(page, cdp, x, 800, 700)
        edge = settled_fling(page)
        assert abs(edge["scrolled"] - first) <= 1, (edge["scrolled"], first)
        assert abs(edge["settings"]["bottom"] - rest["viewBottom"]) <= 1, edge
        assert abs(edge["generate"]["top"] - rest["generate"]["top"]) <= 0.5, "Generate never moved"
        # The page is whole again: the clip is gone, the range is the page's own.
        after = page.evaluate(BRAKE)
        assert after["clipped"] == 0 and after["locked"] == 0 and after["containerMaxHeight"] == "", after
        assert after["scrollHeight"] == whole, after
        assert after["brake"]["flinging"] is False

        # From the edge a second flick goes on into the results' phase (as far
        # as the page's end, which stretches the last 16 px of bottom padding).
        flick(page, cdp, x, 800, 500)
        on = settled_fling(page)
        assert on["scrolled"] > first + 100, on["scrolled"]
        assert on["generate"]["top"] < rest["generate"]["top"] - 100, "the results scroll in the second phase"
        assert rest["viewBottom"] - 20 <= on["settings"]["bottom"] <= rest["viewBottom"] + 1, "with the settings held"
        assert page.evaluate(BRAKE)["scrollHeight"] == whole
        page.close()

    def test_a_flick_up_the_results_stops_at_the_boundary_too(self, browser):
        page, cdp, x = self.open(browser)
        rest = scroll_to(page, 0)
        first, second = phases(rest)
        whole = page.evaluate(BRAKE)["scrollHeight"]
        scroll_to(page, first + second - 20)
        page.evaluate("""() => { window.lowest = Infinity; addEventListener('scroll', () => {
            window.lowest = Math.min(window.lowest, Math.round(scrollY)); }, true); }""")

        # A short, fast flick in the results' phase, whose fling crosses the
        # boundary upward (a probe: 1608 to 1213 without the break).
        flick(page, cdp, x, 300, -120)
        edge = settled_fling(page)
        assert abs(edge["scrolled"] - first) <= 1, (edge["scrolled"], first)
        assert abs(edge["settings"]["bottom"] - rest["viewBottom"]) <= 1, edge
        # The results column is where its hold puts it, not pushed by the row's
        # end as it is in the second phase. (Generate's place inside it is the
        # Forge Assistant's fill, which lays the column out again whenever
        # the page changes and reads the sticky bar 8 px higher once the page
        # is scrolled past its padding: not the break's.)
        assert abs(edge["results"]["top"] - float(edge["resultsTop"].replace("px", ""))) <= 1, edge
        # Read crossing, put back within the frame or two the reading took.
        lowest = page.evaluate("window.lowest")
        assert first - 60 <= lowest <= first, lowest
        after = page.evaluate(BRAKE)
        assert after["locked"] == 0 and after["htmlOverflow"] == "" and after["clipped"] == 0, after
        assert after["scrollHeight"] == whole

        # From the edge a flick on goes up into the settings' phase.
        flick(page, cdp, x, 300, -300)
        on = settled_fling(page)
        assert on["scrolled"] < first - 100, on["scrolled"]
        assert abs(on["results"]["top"] - float(on["resultsTop"].replace("px", ""))) <= 1, "the results held in the first phase"
        page.close()

    def test_a_finger_on_the_screen_is_a_drag_and_crosses_as_it_likes(self, browser):
        page, cdp, x = self.open(browser)
        rest = scroll_to(page, 0)
        first, second = phases(rest)
        whole = page.evaluate(BRAKE)["scrollHeight"]

        # A slow drag from before the boundary to after it: the finger decides.
        scroll_to(page, first - 100)
        drag(page, cdp, x, 700, 200)
        page.wait_for_timeout(600)
        crossed = page.evaluate(SCROLLED)
        assert crossed["scrolled"] > first + 50, crossed["scrolled"]
        # A slow drag that stops short of the boundary stays where it stopped:
        # nothing pulls it to the edge, and the page stays whole.
        scroll_to(page, first - 400)
        drag(page, cdp, x, 700, 250)
        page.wait_for_timeout(600)
        short = page.evaluate(SCROLLED)
        assert first - 200 <= short["scrolled"] <= first - 100, short["scrolled"]
        after = page.evaluate(BRAKE)
        assert after["clipped"] == 0 and after["locked"] == 0 and after["scrollHeight"] == whole, after
        page.close()

    def test_the_wheel_is_not_momentum_and_is_never_stopped(self, browser):
        page, cdp, x = self.open(browser)
        rest = scroll_to(page, 0)
        first, second = phases(rest)
        scroll_to(page, first - 50)
        page.mouse.move(x, 500)
        page.mouse.wheel(0, 200)
        page.wait_for_timeout(800)
        on = page.evaluate(SCROLLED)
        assert on["scrolled"] > first + 100, on["scrolled"]
        after = page.evaluate(BRAKE)
        assert after["clipped"] == 0 and after["locked"] == 0 and after["brake"]["flinging"] is False, after
        page.close()

    def test_a_scroll_past_the_boundary_without_a_fling_writes_nothing(self, browser):
        page, cdp, x = self.open(browser)
        rest = scroll_to(page, 0)
        first, second = phases(rest)
        page.evaluate(HEARD)
        scroll_to(page, first + 100)
        scroll_to(page, first - 100)
        settle(page, 4)
        after = page.evaluate(BRAKE)
        assert after["clipped"] == 0 and after["locked"] == 0, after
        assert after["containerMaxHeight"] == "" and after["htmlOverflow"] == ""
        assert page.evaluate("window.heard") == []
        page.close()

    def test_under_focus_mode_the_workspace_gets_the_edge(self, browser):
        page, cdp, x = self.open(browser)
        entered = page.evaluate("forgeAssistant.focus.enter('tab_txt2img', forgeAssistant.shell.host)")
        assert entered["ok"] is True
        settle(page, 8)
        rest = scroll_to(page, 0, root=True)
        first, second = phases(rest)
        assert first > 600, first
        x = page.evaluate("document.querySelector('#txt2img_settings').getBoundingClientRect().left + 100")
        whole = page.evaluate("document.querySelector('.forge-assistant-focus-root').scrollHeight")

        flick(page, cdp, x, 800, 700)
        edge = settled_fling(page)
        assert abs(edge["scrolled"] - first) <= 1, (edge["scrolled"], first)
        assert abs(edge["generate"]["top"] - rest["generate"]["top"]) <= 0.5, edge
        assert page.evaluate("window.scrollY") == 0, "the page under the workspace did not move"
        after = page.evaluate(BRAKE)
        assert after["clipped"] == 0 and after["locked"] == 0, after
        assert page.evaluate("document.querySelector('.forge-assistant-focus-root').scrollHeight") == whole
        assert page.evaluate("document.querySelector('#txt2img_extra_tabs').style.maxHeight") == ""

        flick(page, cdp, x, 800, 500)
        on = settled_fling(page)
        assert on["scrolled"] > first + 100, on["scrolled"]
        page.evaluate("forgeAssistant.focus.exit()")
        page.close()


class TestOneScrollerForThePage:
    """Asked for: "the entire page takes one scroll action ... first move the
    left column until I reach bottom, then the right column until I reach
    bottom ... both columns should never move at the same time ... the
    generation button stays locked in position as I scroll through the left
    column ... a single master control that controls the two columns from one
    master scroller". The page's own scroller is that control, and the
    columns are held by sticky positioning from two measured numbers
    (model_chain_theme.js, "one scroller for the page"). Before this, the
    results column held only while the settings were the taller, both moved
    together over the infotext's height at the end, and Generate crept up to
    the bar's margin before holding. Run with the Forge Assistant's scripts
    on the page, as on Forge: its fill is what puts the generation details
    under the view."""

    @pytest.mark.parametrize("grid_row", [False, True], ids=["flex-row", "forges-grid-row"])
    def test_the_settings_scroll_first_with_generate_locked_then_the_results_with_the_settings_held(self, browser, grid_row):
        page = open_page(browser, assistant=USER_CSS_WHOLE, grid_row=grid_row)
        settle(page, 8)
        rest = scroll_to(page, 0)
        assert rest["attribute"] == "settings-first", rest
        assert rest["settingsPosition"] == "sticky" and rest["settingsHeld"] is True, rest
        assert rest["settingsZ"] == "1", "a step above the results column's stacking context"
        first, second = phases(rest)
        assert first > 600 and second > 100, (first, second)
        assert abs(rest["row"]["height"] - (rest["settings"]["height"] + second)) <= 1, rest
        if grid_row:
            # Forge's divider is as tall as the grid's row track, and the track
            # is the row's height: the rule puts the least height on the track
            # too, since Forge's `.unequal-height` keeps the track at its
            # content height (both engines hold the column inside the
            # container's content box regardless, measured).
            handle = page.evaluate("document.querySelector('.resize-handle').getBoundingClientRect().height")
            assert abs(handle - rest["row"]["height"]) <= 1, (handle, rest["row"])

        # The first phase: the settings move by the scroll, Generate not at all --
        # not even the margin it used to creep up to the bar by.
        half = scroll_to(page, first // 2)
        assert abs(half["generate"]["top"] - rest["generate"]["top"]) <= 0.5, (half["generate"], rest["generate"])
        assert abs(half["settings"]["top"] - (rest["settings"]["top"] - first // 2)) <= 0.5, half
        # The boundary: the settings' bottom edge at the view's bottom, Generate still.
        edge = scroll_to(page, first)
        assert abs(edge["settings"]["bottom"] - rest["viewBottom"]) <= 1, edge
        assert abs(edge["generate"]["top"] - rest["generate"]["top"]) <= 0.5, edge
        # The second phase: the settings held at their end, the results moving by the scroll.
        mid = scroll_to(page, first + second // 2)
        assert abs(mid["settings"]["bottom"] - rest["viewBottom"]) <= 1, mid
        assert abs(mid["generate"]["top"] - (rest["generate"]["top"] - second // 2)) <= 1, mid
        end = scroll_to(page, first + second)
        assert abs(end["results"]["bottom"] - rest["viewBottom"]) <= 1, end
        assert abs(end["settings"]["bottom"] - rest["viewBottom"]) <= 1, end
        one_column_at_a_time(page, rest)
        page.close()

    def test_with_the_settings_docked_in_the_assistants_panel_the_page_scrolls_the_results_alone(self, browser):
        """The Forge Assistant draws the settings column in its panel
        (`position: fixed`, out of the row, scrolling inside itself and
        containing its own scrolling): the row is then the results' height,
        nothing is held, and the page scrolls the results from the first
        pixel. Released, the two phases are back."""
        page = open_page(browser, assistant=USER_CSS_WHOLE, grid_row=True)
        settle(page, 8)
        rest = scroll_to(page, 0)
        assert rest["rowMin"] and rest["hold"], rest

        page.evaluate("forgeAssistant.dock.place({left: 100, top: 100, width: 340, height: 600})")
        settle(page, 8)
        docked = scroll_to(page, 0)
        assert docked["settingsPosition"] == "fixed" and docked["settingsHeld"] is False, docked
        assert docked["rowMin"] == "" and docked["hold"] == "" and docked["resultsHold"] == "", docked
        assert docked["settingsOverscroll"] == "contain", "a wheel at the column's end moves no page"
        assert abs(docked["row"]["height"] - docked["results"]["height"]) <= 1, docked
        moved = scroll_to(page, 150)
        assert abs(moved["generate"]["top"] - (docked["generate"]["top"] - 150)) <= 0.5, "the results from the start"
        assert abs(moved["settings"]["top"] - docked["settings"]["top"]) <= 0.5, "the docked column where the panel put it"
        # The column's own scrolling, in the panel, moves no page.
        page.evaluate("document.querySelector('#txt2img_settings').scrollTop = 300")
        settle(page, 2)
        inner = page.evaluate(SCROLLED)
        assert page.evaluate("document.querySelector('#txt2img_settings').scrollTop") == 300
        assert inner["scrolled"] == 150, inner["scrolled"]

        page.evaluate("forgeAssistant.dock.release()")
        settle(page, 8)
        back = scroll_to(page, 0)
        assert back["settingsPosition"] == "sticky" and back["settingsHeld"] is True, back
        assert back["rowMin"] == rest["rowMin"] and back["hold"] == rest["hold"], (back, rest)
        first, _ = phases(back)
        again = scroll_to(page, first // 2)
        assert abs(again["generate"]["top"] - back["generate"]["top"]) <= 0.5, again
        page.close()

    def test_under_focus_mode_the_workspace_is_the_scroller_and_the_phases_are_the_same(self, browser):
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        entered = page.evaluate("forgeAssistant.focus.enter('tab_txt2img', forgeAssistant.shell.host)")
        assert entered["ok"] is True
        settle(page, 8)
        rest = scroll_to(page, 0, root=True)
        assert rest["attribute"] == "settings-first" and rest["settingsHeld"] is True, rest
        first, second = phases(rest)
        assert first > 600 and second > 100, (first, second)
        half = scroll_to(page, first // 2, root=True)
        assert abs(half["generate"]["top"] - rest["generate"]["top"]) <= 0.5, (half["generate"], rest["generate"])
        edge = scroll_to(page, first, root=True)
        assert abs(edge["settings"]["bottom"] - rest["viewBottom"]) <= 1, edge
        assert abs(edge["generate"]["top"] - rest["generate"]["top"]) <= 0.5, edge
        end = scroll_to(page, first + second, root=True)
        assert abs(end["results"]["bottom"] - rest["viewBottom"]) <= 1, end
        assert abs(end["settings"]["bottom"] - rest["viewBottom"]) <= 1, end
        one_column_at_a_time(page, rest, root=True)
        assert page.evaluate("window.scrollY") == 0, "the page under the workspace did not move"

        page.evaluate("forgeAssistant.focus.exit()")
        settle(page, 8)
        back = scroll_to(page, 0)
        first_back, _ = phases(back)
        again = scroll_to(page, first_back // 2)
        assert abs(again["generate"]["top"] - back["generate"]["top"]) <= 0.5, again
        page.close()

    def test_a_settings_column_shorter_than_the_view_is_held_where_it_rests_while_the_results_scroll(self, browser):
        page = open_page(browser, assistant=USER_CSS_WHOLE, settings_blocks=3)
        settle(page, 8)
        rest = scroll_to(page, 0)
        assert rest["settings"]["bottom"] < rest["viewBottom"], rest
        assert rest["settingsHeld"] is True and rest["hold"] == rest["resultsHold"], rest
        _, second = phases(rest)
        assert second > 100, second
        moved = scroll_to(page, second // 2)
        assert abs(moved["settings"]["top"] - rest["settings"]["top"]) <= 0.5, "the settings where they rest"
        assert abs(moved["generate"]["top"] - (rest["generate"]["top"] - second // 2)) <= 1, moved
        page.close()

    def test_a_view_with_no_room_holds_nothing_until_there_is_room_again(self, browser):
        """A scroller with no height for the moment -- a phone's browser
        between full screen and not, a window shorter than the bar's margin
        -- gets no plan: one for a view below zero was a row of the two
        columns' heights added up, scrolled a long way past both."""
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        page.set_viewport_size({"width": 1600, "height": 40})
        settle(page, 8)
        found = page.evaluate(SCROLLED)
        assert found["rowMin"] == "" and found["hold"] == "" and found["settingsHeld"] is False, found
        assert page.evaluate("mcTheme.explainScroll()[0].reason") == "the view has no room"
        page.set_viewport_size({"width": 1600, "height": 900})
        settle(page, 8)
        back = scroll_to(page, 0)
        assert back["settingsHeld"] is True and back["rowMin"] != "", back
        first, _ = phases(back)
        half = scroll_to(page, first // 2)
        assert abs(half["generate"]["top"] - back["generate"]["top"]) <= 0.5, (half["generate"], back["generate"])
        page.close()

    def test_stacked_under_the_settings_as_on_a_phone_nothing_is_held(self, browser):
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        page.evaluate("document.querySelector('#txt2img_results').parentElement.style.display = 'block'")
        page.evaluate("window.dispatchEvent(new Event('resize'))")
        settle(page, 8)
        rest = scroll_to(page, 0)
        assert rest["results"]["top"] >= rest["settings"]["bottom"] - 1, "the results under the settings"
        assert rest["rowMin"] == "" and rest["hold"] == "" and rest["settingsHeld"] is False, rest
        moved = scroll_to(page, 300)
        assert abs(moved["settings"]["top"] - (rest["settings"]["top"] - 300)) <= 0.5, moved
        assert abs(moved["results"]["top"] - (rest["results"]["top"] - 300)) <= 0.5, moved
        page.close()

    def test_the_gallerys_button_row_is_left_as_forge_drew_it(self, browser):
        """A Details button at the end of the row jumped to the results' end
        and back, the one scroller's answer to the details being a settings
        column away; the user did not want it ("I don't need this scroll jump
        button"), so the row holds Forge's own buttons and nothing of the
        theme's, scrolled or not."""
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        rest = scroll_to(page, 0)
        first, second = phases(rest)
        assert rest["buttons"] == ["txt2img_open_folder", "save_txt2img"], rest["buttons"]
        assert rest["appended"] == 0
        end = scroll_to(page, first + second)
        assert end["buttons"] == ["txt2img_open_folder", "save_txt2img"] and end["appended"] == 0, end["buttons"]
        assert page.evaluate("document.querySelector('[id^=\"mc-theme-jump\"], .mc-theme-jump')") is None
        page.close()

    def test_a_measure_taken_in_the_second_phase_writes_the_same_numbers(self, browser):
        """Any update of the page measures again, at whatever scroll the page
        is at. In the second phase the results column is drawn pushed up by
        the row's end, lower than its hold; the first build read that
        position as the hold and shrank the row at every such measure,
        which Gecko showed on the layout page. The hold is the offset from
        the scroller's edge, and the browser's own addition to it is read
        only while the column is drawn held."""
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        rest = scroll_to(page, 0)
        first, second = phases(rest)
        before = scroll_to(page, first + second // 2)
        page.evaluate("hooks.after.forEach((fn) => fn()); hooks.after.forEach((fn) => fn())")
        settle(page, 6)
        after = page.evaluate(SCROLLED)
        assert (after["rowMin"], after["hold"], after["resultsHold"]) == (rest["rowMin"], rest["hold"], rest["resultsHold"]), (after, rest)
        assert abs(after["generate"]["top"] - before["generate"]["top"]) <= 0.5, (after, before)
        assert abs(after["settings"]["bottom"] - rest["viewBottom"]) <= 1, after
        explained = page.evaluate("mcTheme.explainScroll()[0]")
        assert explained["minHeight"] == int(rest["rowMin"][:-2]) and explained["heldAt"] == rest["results"]["top"], explained
        page.close()

    def test_applied_again_the_button_and_the_numbers_are_written_once(self, browser):
        """The button is appended once and its glyph rewritten only when the
        page's end comes or goes: an append or a glyph on every apply, or on
        every scroll, would be a child-list change that raises Forge's next
        after-update itself."""
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        page.evaluate(HEARD)
        for _ in range(3):
            page.evaluate("hooks.after.forEach((fn) => fn())")
            settle(page, 6)
        scroll_to(page, 200)
        scroll_to(page, 400)
        scroll_to(page, 0)
        assert page.evaluate("window.heard") == []
        page.close()

    def test_off_leaves_the_columns_as_the_split_view_had_them(self, browser):
        page = open_page(browser, assistant=USER_CSS_WHOLE)
        settle(page, 8)
        page.evaluate("opts.model_chain_theme_scroll = false")
        changed(page)
        settle(page, 6)
        found = scroll_to(page, 0)
        assert found["attribute"] is None and found["appended"] == 0, found
        assert found["settingsPosition"] == "static" and found["settingsHeld"] is False, found
        assert found["rowMin"] == "" and found["hold"] == "" and found["resultsHold"] == "", found
        assert found["resultsPosition"] == "sticky" and found["resultsTop"] == "80px", "the split view's own hold"
        assert found["buttons"] == ["txt2img_open_folder", "save_txt2img"], found["buttons"]
        page.close()


class TestOnAPhonesBrowser:
    """Chromium's mobile viewport, as a phone's browser lays the page out. A
    page wider than the window widens the *layout* viewport -- what fixed
    and sticky boxes are held against, what `innerWidth` and `innerHeight`
    report -- to the page's width, and lets the page be zoomed out to it; a
    desktop window only grows a scrollbar. The user's second run of the one
    scroller was that, on a phone: the infotext's paragraph a page's width
    (TestTheGenerationInfo), the page zoomed out to it, the gallery filled
    to the zoomed-out glass, the columns held against a viewport taller
    than the screen. `Page.getLayoutMetrics` reads the viewports."""

    def test_nothing_in_the_page_widens_the_layout_viewport_or_lets_it_zoom_out(self, browser):
        context = browser.new_context(viewport={"width": 720, "height": 870}, device_scale_factor=2,
                                      is_mobile=True, has_touch=True)
        page = open_page(browser, assistant=USER_CSS_WHOLE, grid_row=True, context=context,
                         infotext=WIDE_INFOTEXT)
        settle(page, 8)
        # A stray wide box of somebody's, in the page.
        page.evaluate("() => { const stray = document.createElement('div'); stray.id = 'stray'; "
                      "stray.style.cssText = 'width: 1500px; height: 4px'; "
                      "document.querySelector('.gradio-container').appendChild(stray); }")
        settle(page, 8)
        cdp = context.new_cdp_session(page)
        metrics = cdp.send("Page.getLayoutMetrics")
        assert metrics["cssLayoutViewport"]["clientWidth"] == 720, metrics["cssLayoutViewport"]
        assert metrics["cssContentSize"]["width"] == 720, metrics["cssContentSize"]
        assert page.evaluate("[window.innerWidth, window.innerHeight]") == [720, 870]
        cdp.send("Emulation.setPageScaleFactor", {"pageScaleFactor": 0.3})
        settle(page, 8)
        assert page.evaluate("window.visualViewport.scale") == 1, "nothing to zoom out to"
        cdp.detach()
        # And the columns are held as on a desktop: the sideways clip makes
        # no scroll container of Gradio's container.
        rest = scroll_to(page, 0)
        assert rest["attribute"] == "settings-first" and rest["settingsHeld"] is True, rest
        first, second = phases(rest)
        assert first > 100 and second > 0, (first, second)
        half = scroll_to(page, first // 2)
        assert abs(half["generate"]["top"] - rest["generate"]["top"]) <= 0.5, (half["generate"], rest["generate"])
        page.close()
        context.close()


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
        assert page.evaluate("document.querySelectorAll('[data-mc-theme-scroll], .mc-theme-scroll-settings').length") == 0
        assert page.evaluate("getComputedStyle(document.querySelector('#txt2img_settings')).position") == "static"
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
        # With an infotext whose longest run is a few windows wide: the
        # paragraph kept out of view must not be that wide (TestTheGenerationInfo).
        page = open_page(browser, width=400, height=800, infotext=WIDE_INFOTEXT)
        found = page.evaluate(MEASURE)

        assert found["scrollWidth"] <= found["innerWidth"]
        assert found["height"] == 56
        assert found["hidden"]
        page.close()
