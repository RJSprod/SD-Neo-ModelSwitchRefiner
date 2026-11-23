"""The generation settings as accordions, and the Settings Manager, in Chromium.

``javascript/model_chain_settings_manager.js`` (2026-11-23): "i want everything
that is not prompt (positive, negative, literal, and lora viewer), to be in an
accordion ... generation config like batch and resolution should be collapsible
as well ... a last item added to this list that is a "Settings Manager", which
would have options for me to rearrange or show/hide the other accordions in
real time."

The page is Forge Neo's Txt2Img and Img2Img settings columns in the nesting
Gradio 4.40 draws them, as measured on a real Gradio 4.40 page built from Forge
Neo's own ``modules/ui.py``, ``ui_toprow.py`` and processing scripts: the
prompt container, the compact style's tool row, the sampler's group with its
``.styler``, the accordions row with an InputAccordion and its hidden checkbox
in a hidden form, the dimensions row, the CFG row's form, the seed's row and
its hidden extras, the override row's hidden form, and the script container's
groups -- each script's accordion, a group with nothing drawn, the Script
choice's form and a chosen script's hidden group. Gradio's accordions carry
generated classes (``svelte-…`` here, as on the real page) and the rules that
draw them are scoped to those, and Gradio's own button reset
(``.gradio-container-4-40-0 button { padding: 0 }``) is on the page, because
both decided what the first build looked like.

What is held here is what the user would meet: a block left out of every
accordion, a section that moves or hides and comes back after a reload, a drag
that scrolls the page instead, a Hires. fix switched on because a script
clicked its label, an update loop, and a page that is not Forge's again when
the setting goes off.
"""

from __future__ import annotations

import glob
import os
import re
import urllib.parse
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_settings_manager.js"
STYLE = ROOT / "style.css"
ORIGIN = "http://forge-settings-manager.test"

# Gradio 4.40's rules for what the page holds, at their own weight: the theme's
# variables, the block, the row and the column, the form, and the accordion's
# scoped rules (Accordion.svelte), with Gradio's reset of a button's padding.
GRADIO_CSS = """
:root { --block-border-width: 1px; --block-border-color: #303030; --border-color-primary: #303030;
        --block-background-fill: #111; --block-radius: 8px; --block-padding: 8px; --layout-gap: 12px;
        --background-fill-primary: #000; --background-fill-secondary: #161616;
        --body-text-color: #e6e6e6; --body-text-color-subdued: #888; --color-accent: #f97316;
        --button-secondary-background-fill: #1a1a1a; --button-secondary-background-fill-hover: #262626;
        --button-secondary-border-color: #303030; --button-secondary-text-color: #e6e6e6;
        --section-header-text-size: 15px; --section-header-text-weight: 400; --size-2: 8px; }
*, ::before, ::after { box-sizing: border-box; }
body { margin: 0; background: #000; color: #e6e6e6; font: 14px sans-serif; }
.gradio-container-4-40-0 button, .gradio-container-4-40-0 input, .gradio-container-4-40-0 select {
    padding: 0; font-size: 100%; }
.tabitem { padding: 8px; width: 420px; }
.hidden, .hide { display: none !important; }
div.svelte-col { display: flex; position: relative; flex-direction: column; }
div.svelte-col > *, div.svelte-col > .form > * { width: 100%; }
.gap.svelte-col { gap: var(--layout-gap); }
div.svelte-row { display: flex; flex-wrap: wrap; gap: var(--layout-gap); width: 100%; position: relative; }
div.svelte-row > *, div.svelte-row > .form > * { flex: 1 1 0%; min-width: min(160px, 100%); }
div.svelte-form { display: flex; flex-direction: inherit; flex-wrap: wrap; gap: 0; }
.gr-group.svelte-grp { border-radius: 8px; }
.styler.svelte-grp { display: flex; flex-direction: column; gap: 1px; }
.block.svelte-blk { position: relative; margin: 0; box-shadow: none; border-color: var(--block-border-color);
                    border-radius: var(--block-radius); background: var(--block-background-fill);
                    width: 100%; line-height: 1.5; }
.padded.svelte-blk { padding: var(--block-padding); }
span.svelte-lw { font-weight: var(--section-header-text-weight); font-size: var(--section-header-text-size); }
.label-wrap.svelte-lw { display: flex; justify-content: space-between; cursor: pointer;
                        width: 100%; background: none; border: 0; color: inherit; }
.label-wrap.open.svelte-lw { margin-bottom: var(--size-2); }
.icon.svelte-lw { transition: 150ms; }
.field { display: block; width: 100%; min-height: 30px; }
"""

PROMPTS = """
      <div id="{tab}_prompt_container" class="gradio-column svelte-col gap">
        <div id="{tab}_prompt_row" class="gradio-row svelte-row prompt-row"><div class="form svelte-form">
          <div id="{tab}_prompt" class="block svelte-blk padded"><textarea>a cat</textarea></div></div></div>
        <div id="mc-lora-viewer-launch-{tab}" class="mc-lv-launch"><button>Lora Viewer</button></div>
        <div id="{tab}_neg_prompt_row" class="gradio-row svelte-row prompt-row"><div class="form svelte-form">
          <div id="{tab}_neg_prompt" class="block svelte-blk padded"><textarea></textarea></div></div></div>
      </div>
      <div class="gradio-row svelte-row toprow-compact-stylerow">
        <div class="gradio-column svelte-col toprow-compact-tools"><div id="{tab}_tools" class="gradio-row svelte-row">
          <button id="{p}paste" class="tool">↙</button><button id="{tab}_clear_prompt" class="tool">x</button></div></div>
        <div class="gradio-column svelte-col"><div id="{tab}_styles_row" class="gradio-row svelte-row">
          <div class="form svelte-form"><div id="{tab}_styles" class="block svelte-blk padded"><input class="field"></div></div>
        </div></div>
      </div>"""


def accordion(id_, label, *, open_=False, input_=False, body=None):
    """An accordion as Gradio 4.40 draws one (Accordion.svelte inside a Block),
    with Forge's InputAccordion's class and checkbox when ``input_``. A press on
    its label opens and closes it, as Gradio's handler does."""
    classes = "block gradio-accordion svelte-blk padded" + (" input-accordion" if input_ else "")
    checkbox = ('<input type="checkbox" class="input-accordion-checkbox" '
                f'id="{id_}-visible-checkbox">' if input_ else "")
    body = body if body is not None else f'<div class="block svelte-blk padded"><input class="field" aria-label="{label} setting"></div>'
    ident = f'id="{id_}" ' if id_ else ""
    # Block.svelte writes these inline on every block.
    inline = ("border-style: solid; overflow: visible; min-width: min(0px, 100%); "
              "border-width: var(--block-border-width);")
    return (f'<div {ident}class="{classes}" style="{inline}">'
            f'<div class="wrap svelte-status hide"></div>'
            f'<button class="label-wrap svelte-lw{" open" if open_ else ""}" onclick="gradioToggle(this)">'
            f'<span class="svelte-lw">{checkbox}{label}</span> '
            f'<span class="icon svelte-lw" style="transform: rotate({0 if open_ else 90}deg);">▼</span></button>'
            f'<div style="display: {"block" if open_ else "none"};"><div class="svelte-col gap">{body}</div></div></div>')


def group(inner, *, hidden=False, id_=None):
    ident = f'id="{id_}" ' if id_ else ""
    return (f'<div {ident}class="gr-group gradio-group svelte-grp{" hide" if hidden else ""}">'
            f'<div class="styler svelte-grp">{inner}</div></div>')


def block(id_, label, *, hidden=False):
    return (f'<div id="{id_}" class="block svelte-blk padded{" hidden" if hidden else ""}">'
            f'<span data-testid="block-info">{label}</span><input class="field"></div>')


def scripts(tab, chosen=False):
    """The script container: three scripts' accordions, a group with nothing
    drawn, the Script choice, and the chosen script's controls -- hidden until
    a script is chosen, drawn when one was (``chosen``)."""
    return f"""
      <div id="{tab}_script_container" class="gr-group gradio-group svelte-grp"><div class="styler svelte-grp">
        {group(accordion("controlnet", "ControlNet Integrated"), id_=f"{tab}_controlnet")}
        {group(accordion(None, "Dynamic Prompts"))}
        {group('<div class="form svelte-form hidden"><div class="block svelte-blk hidden"><input type="checkbox"></div></div>'
               + accordion("input-accordion-3", "ImageStitch Integrated", input_=True))}
        {group("")}
        <div class="form svelte-form">{block("script_list", "Script")}</div>
        {group(block(f"{tab}_xyz", "X/Y/Z plot values"), hidden=not chosen)}
      </div></div>"""


def txt2img(chosen=False):
    return f"""
    <div id="txt2img_settings" class="gradio-column svelte-col gap compact">{PROMPTS.format(tab="txt2img", p="")}
      {group(f'<div id="sampler_selection_txt2img" class="gradio-row svelte-row"><div class="form svelte-form">'
             f'{block("txt2img_sampling", "Sampling Method")}{block("txt2img_steps", "Sampling Steps")}</div></div>')}
      <div id="txt2img_accordions" class="gradio-row svelte-row accordions">
        <div class="form svelte-form hidden"><div id="txt2img_hr-checkbox" class="block svelte-blk hidden">
          <input type="checkbox"></div></div>
        {accordion("txt2img_hr", "Hires. fix", input_=True)}
      </div>
      <div class="gradio-row svelte-row">
        <div id="txt2img_column_size" class="gradio-column svelte-col gap">{block("txt2img_width", "Width")}{block("txt2img_height", "Height")}</div>
        <div id="txt2img_column_batch" class="gradio-column svelte-col gap">{block("txt2img_batch_count", "Batch Count")}</div>
      </div>
      <div class="gradio-row svelte-row"><div class="form svelte-form">{block("txt2img_cfg_scale", "CFG Scale")}</div></div>
      <div id="txt2img_seed_row" class="gradio-row svelte-row"><div class="form svelte-form">{block("txt2img_seed", "Seed")}</div></div>
      {group(block("txt2img_subseed", "Variation seed"), hidden=True, id_="txt2img_seed_extras")}
      <div id="txt2img_override_settings_row" class="gradio-row svelte-row"><div class="form svelte-form hidden">
        {block("txt2img_override_settings", "Override settings", hidden=True)}</div></div>
      {scripts("txt2img", chosen)}
    </div>"""


def img2img():
    return f"""
    <div id="img2img_settings" class="gradio-column svelte-col gap compact">{PROMPTS.format(tab="img2img", p="i2i_")}
      <div id="mode_img2img" class="tabs">{block("img2img_image", "Image for img2img")}</div>
      <div class="gradio-row svelte-row"><div class="form svelte-form">{block("resize_mode", "Resize Mode")}</div></div>
      <div id="inpaint_controls" class="gr-group gradio-group svelte-grp hide"><div class="styler svelte-grp">
        {block("img2img_mask_blur", "Mask blur")}</div></div>
      {group(f'<div id="sampler_selection_img2img" class="gradio-row svelte-row"><div class="form svelte-form">'
             f'{block("img2img_sampling", "Sampling Method")}</div></div>')}
      <div class="gradio-row svelte-row">
        <div id="img2img_column_size" class="gradio-column svelte-col gap">{block("img2img_width", "Width")}</div>
      </div>
      <div class="gradio-row svelte-row"><div class="form svelte-form">{block("img2img_cfg_scale", "CFG Scale")}</div></div>
      <div class="form svelte-form">{block("img2img_denoising_strength", "Denoising Strength")}{block("img2img_noise_multiplier", "Noise multiplier")}</div>
      <div id="img2img_seed_row" class="gradio-row svelte-row"><div class="form svelte-form">{block("img2img_seed", "Seed")}</div></div>
      {scripts("img2img")}
    </div>"""


def tight(html):
    """No whitespace between elements, as Svelte writes a page: a block's next
    sibling is the next block, never a text node that stays put."""
    return re.sub(r">\s+<", "><", html)


def page_html(*, option=None, no_accordions=False, storage_throws=False, chosen=False):
    """The page. ``option`` is Forge's opts value for the setting (absent when
    None); ``no_accordions`` is a page with no accordion of Gradio's to read
    the classes off; ``storage_throws`` is a browser whose localStorage refuses;
    ``chosen`` is a page whose Script choice has a script chosen."""
    opts = "{}" if option is None else "{model_chain_settings_manager: %s}" % ("true" if option else "false")
    storage = """
Object.defineProperty(window, "localStorage", {get() { throw new Error("blocked"); }});""" if storage_throws else ""
    body_txt = txt2img(chosen)
    if no_accordions:
        # Every accordion of Gradio's taken out: the script's own look is used.
        body_txt = (body_txt.replace('accordions">', 'accordions-none">')
                    .replace("block gradio-accordion", "block gradio-not-an-accordion"))
        body_img = img2img().replace("block gradio-accordion", "block gradio-not-an-accordion")
    else:
        body_img = img2img()
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{GRADIO_CSS}</style>
<link rel="stylesheet" href="/style.css">
<script>{storage}
var opts = {opts};
function gradioApp() {{
    const elems = document.getElementsByTagName("gradio-app");
    return elems.length ? elems[0] : document;
}}
const hooks = {{loaded: [], after: [], options: []}};
function onUiLoaded(fn) {{ hooks.loaded.push(fn); }}
function onAfterUiUpdate(fn) {{ hooks.after.push(fn); }}
function onOptionsChanged(fn) {{ hooks.options.push(fn); }}
// Gradio's own handler on its accordions' label: opens and closes it, and
// counts, so a test can say the script never pressed one.
window.gradioPresses = 0;
function gradioToggle(button) {{
    window.gradioPresses += 1;
    const open = !button.classList.contains("open");
    button.classList.toggle("open", open);
    button.querySelector(".icon").style.transform = open ? "rotate(0)" : "rotate(90deg)";
    button.nextElementSibling.style.display = open ? "block" : "none";
}}
</script>
</head><body class="dark">
<gradio-app><div class="gradio-container gradio-container-4-40-0">
<div id="tabs">
  <div id="tab_txt2img" class="tabitem">{tight(body_txt)}</div>
  <div id="tab_img2img" class="tabitem">{tight(body_img)}</div>
</div>
</div></gradio-app>
<script src="/javascript/model_chain_settings_manager.js"></script>
<script>hooks.loaded.forEach((fn) => fn()); hooks.after.forEach((fn) => fn());</script>
</body></html>"""


class Served:
    def __init__(self, html: str):
        self.html = html

    def __call__(self, route):
        path = urllib.parse.urlsplit(route.request.url).path
        if path == "/":
            route.fulfill(status=200, content_type="text/html", body=self.html)
        elif path == "/style.css":
            route.fulfill(status=200, content_type="text/css", body=STYLE.read_text(encoding="utf-8"))
        elif path.startswith("/javascript/") and (ROOT / path[1:]).is_file():
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


def open_page(browser, *, width=420, height=900, touch=False, **page):
    context = browser.new_context(viewport={"width": width, "height": height}, has_touch=touch)
    tab = context.new_page()
    errors = []
    tab.on("pageerror", lambda error: errors.append(str(error)))
    tab.route(ORIGIN + "/**", Served(page_html(**page)))
    tab.goto(ORIGIN + "/")
    tab.wait_for_function("!!window.mcSettingsManager")
    tab.errors = errors
    return tab


def reload(tab):
    tab.reload()
    tab.wait_for_function("!!window.mcSettingsManager")


def keys(tab, name="txt2img"):
    return tab.evaluate("(tab) => (mcSettingsManager.describe(tab) || []).map((s) => s.key)", name)


def column_order(tab, name="txt2img"):
    return tab.evaluate("""(tab) => Array.from(document.querySelectorAll('#mc-sm-list-' + tab + ' > .mc-sm-item'))
        .map((item) => item.getAttribute('data-mc-sm-key'))""", name)


def row_order(tab, name="txt2img"):
    return tab.evaluate("""(tab) => Array.from(document.querySelectorAll('#mc-sm-list-' + tab + ' .mc-sm-row'))
        .map((row) => row.getAttribute('data-mc-sm-key'))""", name)


def drawn(tab, selector):
    return tab.evaluate("""(selector) => { const node = document.querySelector(selector);
        return !!node && node.getClientRects().length > 0 && getComputedStyle(node).visibility !== 'hidden'; }""",
                        selector)


# The DOM of a settings column, ids and classes six levels down, for "put back
# exactly as it was".
SIGNATURE = """(tab) => {
    const walk = (node, depth) => {
        if (depth > 6) return '';
        const name = (node.id ? '#' + node.id : '') + '.' + Array.from(node.classList).sort().join('.');
        return name + '(' + Array.from(node.children).map((child) => walk(child, depth + 1)).join(',') + ')';
    };
    return walk(document.getElementById(tab + '_settings'), 0);
}"""

TXT2IMG = ["tools", "sampler", "id:txt2img_hr", "size", "cfg", "seed", "override",
           "id:txt2img_controlnet", "label:Dynamic Prompts", "label:ImageStitch Integrated", "script"]


# --------------------------------------------------------------------------- #
# The sections
# --------------------------------------------------------------------------- #


class TestTheSections:
    def test_every_block_under_the_prompts_is_a_section_in_forges_order(self, browser):
        tab = open_page(browser)
        found = tab.evaluate("mcSettingsManager.describe('txt2img')")
        assert [section["key"] for section in found] == TXT2IMG
        labels = {section["key"]: section["label"] for section in found}
        assert labels["tools"] == "Tools & Styles" and labels["sampler"] == "Sampling"
        assert labels["size"] == "Size & Batch" and labels["cfg"] == "CFG"
        assert labels["seed"] == "Seed" and labels["script"] == "Script"
        assert labels["id:txt2img_hr"] == "Hires. fix"
        kinds = {section["key"]: section["kind"] for section in found}
        assert kinds["id:txt2img_hr"] == "accordion" and kinds["label:Dynamic Prompts"] == "accordion"
        assert kinds["sampler"] == "wrap" and kinds["script"] == "wrap"
        assert column_order(tab) == TXT2IMG + ["manager"]
        assert not tab.errors
        tab.context.close()

    def test_img2img_has_its_image_its_inpaint_and_its_denoising(self, browser):
        """And a form Gradio put round two bare components in the column -- the
        denoising slider and a section script's control after it -- is two
        sections, not one."""
        tab = open_page(browser)
        found = tab.evaluate("mcSettingsManager.describe('img2img')")
        assert [section["key"] for section in found] == [
            "tools", "image", "inpaint", "sampler", "size", "cfg", "denoise",
            "id:img2img_noise_multiplier", "seed",
            "id:img2img_controlnet", "label:Dynamic Prompts", "label:ImageStitch Integrated", "script"]
        labels = {section["key"]: section["label"] for section in found}
        assert labels["image"] == "Image" and labels["denoise"] == "Denoising" and labels["size"] == "Size"
        image = tab.evaluate("""() => {
            const item = document.querySelector('#mc-sm-list-img2img [data-mc-sm-key=image]');
            return [!!item.querySelector('#mode_img2img'), !!item.querySelector('#resize_mode')]; }""")
        assert image == [True, True]
        tab.context.close()

    def test_the_list_follows_the_prompts_and_holds_everything_else(self, browser):
        tab = open_page(browser)
        found = tab.evaluate("""() => {
            const column = document.getElementById('txt2img_settings');
            const children = Array.from(column.children);
            return {first: children[0].id, second: children[1].id,
                    drawnOutsideList: children.slice(2).filter((child) => child.getClientRects().length).length,
                    promptRows: Array.from(document.getElementById('txt2img_prompt_container').children).map((c) => c.id),
                    emptied: ['txt2img_accordions', 'txt2img_script_container'].map((id) =>
                        getComputedStyle(document.getElementById(id)).display)};
        }""")
        assert found["first"] == "txt2img_prompt_container" and found["second"] == "mc-sm-list-txt2img"
        assert found["drawnOutsideList"] == 0
        assert found["promptRows"] == ["txt2img_prompt_row", "mc-lora-viewer-launch-txt2img",
                                       "txt2img_neg_prompt_row"]
        assert found["emptied"] == ["none", "none"]
        tab.context.close()

    def test_a_block_forge_did_not_put_in_an_accordion_is_in_one_that_looks_like_gradios(self, browser):
        """The accordion is built of the classes of one of Gradio's on the
        page, generated ones included, so Gradio's scoped rules draw it."""
        tab = open_page(browser)
        found = tab.evaluate("""() => {
            const item = document.querySelector('#mc-sm-list-txt2img [data-mc-sm-key=sampler]');
            const acc = item.firstElementChild;
            const wrap = acc.querySelector(':scope > .label-wrap');
            const gradio = document.getElementById('txt2img_hr');
            const look = (node) => { const s = getComputedStyle(node);
                return [s.paddingTop, s.borderTopWidth, s.borderTopStyle, s.borderRadius, s.backgroundColor]; };
            return {block: Array.from(acc.classList).sort(), wrap: Array.from(wrap.classList).sort(),
                    holds: !!acc.querySelector('#sampler_selection_txt2img'), label: wrap.firstElementChild.textContent,
                    expanded: wrap.getAttribute('aria-expanded'), same: JSON.stringify(look(acc)) === JSON.stringify(look(gradio)),
                    wrapDisplay: getComputedStyle(wrap).display, wrapJustify: getComputedStyle(wrap).justifyContent};
        }""")
        assert found["block"] == ["block", "gradio-accordion", "mc-sm-acc", "padded", "svelte-blk"]
        assert found["wrap"] == ["label-wrap", "open", "svelte-lw"]
        assert found["holds"] and found["label"] == "Sampling" and found["expanded"] == "true"
        assert found["same"], "drawn as Gradio draws its own accordion"
        assert found["wrapDisplay"] == "flex" and found["wrapJustify"] == "space-between"
        tab.context.close()

    def test_an_accordion_already_there_is_the_same_element_moved_not_copied(self, browser):
        tab = open_page(browser)
        found = tab.evaluate("""() => ({
            hr: document.querySelectorAll('#txt2img_hr').length,
            inItem: !!document.getElementById('txt2img_hr').closest('[data-mc-sm-key="id:txt2img_hr"]'),
            checkboxWithIt: !!document.querySelector('[data-mc-sm-key="id:txt2img_hr"] #txt2img_hr-checkbox'),
            controlnet: document.getElementById('txt2img_controlnet').parentElement.getAttribute('data-mc-sm-key'),
            empty: !!document.querySelector('[data-mc-sm-key=script] .gr-group > .styler:empty'),
        })""")
        assert found == {"hr": 1, "inItem": True, "checkboxWithIt": True,
                         "controlnet": "id:txt2img_controlnet", "empty": True}
        tab.context.close()

    def test_a_script_choice_takes_the_chosen_scripts_own_controls_with_it(self, browser):
        """Hidden until a script is chosen, and drawn when one was as the page
        loaded: either way inside the Script section, never one of its own."""
        for chosen in (False, True):
            tab = open_page(browser, chosen=chosen)
            found = tab.evaluate("""() => !!document.querySelector('[data-mc-sm-key=script] #txt2img_xyz')""")
            assert found is True, chosen
            assert keys(tab) == TXT2IMG, chosen
            tab.context.close()

    def test_a_page_with_no_accordion_to_read_draws_its_own(self, browser):
        tab = open_page(browser, no_accordions=True)
        found = tab.evaluate("""() => {
            const acc = document.querySelector('#mc-sm-list-txt2img [data-mc-sm-key=sampler] > .mc-sm-acc');
            const s = getComputedStyle(acc);
            return {own: acc.classList.contains('mc-sm-own'), border: s.borderTopStyle,
                    background: s.backgroundColor, radius: s.borderTopLeftRadius,
                    wrap: getComputedStyle(acc.querySelector('.label-wrap')).display};
        }""")
        assert found == {"own": True, "border": "solid", "background": "rgb(17, 17, 17)",
                         "radius": "8px", "wrap": "flex"}
        tab.context.close()


# --------------------------------------------------------------------------- #
# The accordions
# --------------------------------------------------------------------------- #


class TestTheAccordions:
    def test_a_press_on_a_label_closes_it_and_the_choice_is_kept(self, browser):
        tab = open_page(browser)
        tab.click("#mc-sm-list-txt2img [data-mc-sm-key=seed] .label-wrap")
        closed = tab.evaluate("""() => {
            const acc = document.querySelector('#mc-sm-list-txt2img [data-mc-sm-key=seed] > .mc-sm-acc');
            const wrap = acc.querySelector('.label-wrap');
            return [wrap.classList.contains('open'), wrap.getAttribute('aria-expanded'),
                    acc.querySelector('.icon').style.transform,
                    document.getElementById('txt2img_seed').getClientRects().length];
        }""")
        assert closed == [False, "false", "rotate(90deg)", 0]
        reload(tab)
        assert tab.evaluate("mcSettingsManager.describe('txt2img').find((s) => s.key == 'seed').open") is False
        assert not drawn(tab, "#txt2img_seed")
        tab.click("#mc-sm-list-txt2img [data-mc-sm-key=seed] .label-wrap")
        assert drawn(tab, "#txt2img_seed")
        tab.context.close()

    def test_gradios_own_accordions_are_never_pressed_or_opened(self, browser):
        """Forge's InputAccordion switches Hires. fix on when its accordion
        opens: the script must never open one, build, move, reset or not."""
        tab = open_page(browser)
        tab.evaluate("mcSettingsManager.move('txt2img', 'id:txt2img_hr', -2); mcSettingsManager.reset('txt2img')")
        found = tab.evaluate("""() => ({presses: window.gradioPresses,
            open: document.querySelector('#txt2img_hr > .label-wrap').classList.contains('open'),
            body: document.querySelector('#txt2img_hr > .label-wrap').nextElementSibling.style.display})""")
        assert found == {"presses": 0, "open": False, "body": "none"}
        tab.context.close()

    def test_the_manager_starts_closed_and_is_last(self, browser):
        tab = open_page(browser)
        found = tab.evaluate("""() => {
            const list = document.getElementById('mc-sm-list-txt2img');
            const last = list.lastElementChild;
            return [last.getAttribute('data-mc-sm-key'), last.querySelector('.label-wrap').textContent.trim(),
                    last.querySelector('.label-wrap').getAttribute('aria-expanded')];
        }""")
        assert found == ["manager", "Settings Manager ▼", "false"]
        tab.context.close()


# --------------------------------------------------------------------------- #
# The Settings Manager
# --------------------------------------------------------------------------- #


def open_manager(tab, name="txt2img"):
    """Open the Settings Manager, unless a kept choice already has it open."""
    selector = f"#mc-sm-list-{name} [data-mc-sm-key=manager] .label-wrap"
    if tab.get_attribute(selector, "aria-expanded") != "true":
        tab.click(selector)
    tab.wait_for_selector(f"#mc-sm-list-{name} .mc-sm-rows")


class TestTheManager:
    def test_a_row_for_every_section_in_the_columns_order(self, browser):
        tab = open_page(browser)
        open_manager(tab)
        assert row_order(tab) == TXT2IMG
        found = tab.evaluate("""() => {
            const rows = Array.from(document.querySelectorAll('#mc-sm-list-txt2img .mc-sm-row'));
            return {names: rows.map((row) => row.querySelector('.mc-sm-name').textContent).slice(0, 3),
                    firstUp: rows[0].querySelector('.mc-sm-up').disabled,
                    lastDown: rows[rows.length - 1].querySelector('.mc-sm-down').disabled,
                    idle: rows.filter((row) => row.classList.contains('mc-sm-idle')).map((row) => row.getAttribute('data-mc-sm-key'))};
        }""")
        assert found["names"] == ["Tools & Styles", "Sampling", "Hires. fix"]
        assert found["firstUp"] and found["lastDown"]
        assert found["idle"] == ["override"]
        tab.context.close()

    def test_up_and_down_move_the_section_at_once_and_it_is_kept(self, browser):
        tab = open_page(browser)
        open_manager(tab)
        tab.click("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=seed] .mc-sm-up")
        tab.click("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=seed] .mc-sm-up")
        expected = ["tools", "sampler", "id:txt2img_hr", "seed", "size", "cfg"]
        assert column_order(tab)[:6] == expected and row_order(tab)[:6] == expected
        tab.click("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=tools] .mc-sm-down")
        assert column_order(tab)[:2] == ["sampler", "tools"]
        seed_top = tab.evaluate("document.querySelector('[data-mc-sm-key=seed]').getBoundingClientRect().top")
        size_top = tab.evaluate("document.querySelector('#mc-sm-list-txt2img [data-mc-sm-key=size]').getBoundingClientRect().top")
        assert seed_top < size_top
        reload(tab)
        assert column_order(tab)[:6] == ["sampler", "tools", "id:txt2img_hr", "seed", "size", "cfg"]
        tab.context.close()

    def test_the_eye_hides_a_section_and_shows_it_again(self, browser):
        tab = open_page(browser)
        open_manager(tab)
        tab.click("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=cfg] .mc-sm-eye")
        assert not drawn(tab, "#txt2img_cfg_scale")
        row = tab.evaluate("""() => { const row = document.querySelector('#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=cfg]');
            return [row.classList.contains('mc-sm-off'), row.querySelector('.mc-sm-eye').getAttribute('aria-pressed'),
                    getComputedStyle(row.querySelector('.mc-sm-name')).textDecorationLine]; }""")
        assert row == [True, "false", "line-through"]
        reload(tab)
        assert not drawn(tab, "#txt2img_cfg_scale")
        open_manager(tab)
        tab.click("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=cfg] .mc-sm-eye")
        assert drawn(tab, "#txt2img_cfg_scale")
        tab.context.close()

    def test_a_mouse_drags_a_row_and_the_column_follows(self, browser):
        tab = open_page(browser)
        open_manager(tab)
        grip = tab.locator("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=seed] .mc-sm-grip")
        grip.scroll_into_view_if_needed()
        start = grip.bounding_box()
        target = tab.locator("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=sampler]").bounding_box()
        x, y0, y1 = start["x"] + start["width"] / 2, start["y"] + start["height"] / 2, target["y"] + 4
        tab.mouse.move(x, y0)
        tab.mouse.down()
        for step in range(1, 11):
            tab.mouse.move(x, y0 + (y1 - y0) * step / 10)
        tab.mouse.up()
        assert column_order(tab)[:3] == ["tools", "seed", "sampler"]
        assert row_order(tab)[:3] == ["tools", "seed", "sampler"]
        reload(tab)
        assert column_order(tab)[:3] == ["tools", "seed", "sampler"]
        tab.context.close()

    def test_a_finger_drags_a_row_and_the_page_stays_still(self, browser):
        tab = open_page(browser, touch=True, height=700)
        open_manager(tab)
        grip = tab.locator("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=seed] .mc-sm-grip")
        grip.scroll_into_view_if_needed()
        start = grip.bounding_box()
        target = tab.locator("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=sampler]").bounding_box()
        x, y0, y1 = start["x"] + start["width"] / 2, start["y"] + start["height"] / 2, target["y"] + 4
        scrolled = tab.evaluate("scrollY")
        cdp = tab.context.new_cdp_session(tab)

        def touch(kind, y):
            cdp.send("Input.dispatchTouchEvent", {
                "type": kind, "touchPoints": [] if kind == "touchEnd" else [{"x": x, "y": y}]})

        touch("touchStart", y0)
        for step in range(1, 11):
            touch("touchMove", y0 + (y1 - y0) * step / 10)
        touch("touchEnd", y1)
        tab.wait_for_timeout(100)
        assert column_order(tab)[:3] == ["tools", "seed", "sampler"]
        assert tab.evaluate("scrollY") == scrolled
        assert tab.evaluate("document.documentElement.style.overflow") == ""
        tab.context.close()

    def test_a_drag_the_browser_cancels_puts_the_order_back(self, browser):
        """A browser that takes the pointer for a gesture of its own cancels
        it: what the drag moved goes back, and nothing is kept."""
        tab = open_page(browser, touch=True, height=700)
        open_manager(tab)
        grip = tab.locator("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=seed] .mc-sm-grip")
        grip.scroll_into_view_if_needed()
        start = grip.bounding_box()
        target = tab.locator("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=sampler]").bounding_box()
        x, y0, y1 = start["x"] + start["width"] / 2, start["y"] + start["height"] / 2, target["y"] + 4
        cdp = tab.context.new_cdp_session(tab)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y0}]})
        for step in range(1, 11):
            cdp.send("Input.dispatchTouchEvent", {"type": "touchMove",
                                                  "touchPoints": [{"x": x, "y": y0 + (y1 - y0) * step / 10}]})
        # Chromium hands a finger's moves to the page a frame at a time.
        tab.wait_for_timeout(100)
        assert column_order(tab)[:3] == ["tools", "seed", "sampler"], "moved while dragged"
        cdp.send("Input.dispatchTouchEvent", {"type": "touchCancel", "touchPoints": []})
        tab.wait_for_timeout(100)
        assert column_order(tab) == TXT2IMG + ["manager"]
        assert row_order(tab) == TXT2IMG
        assert tab.evaluate("localStorage.getItem('mc-settings-manager:txt2img')") is None \
            or tab.evaluate("JSON.parse(localStorage.getItem('mc-settings-manager:txt2img')).order.indexOf('seed')") != 1
        tab.context.close()

    def test_the_arrow_keys_on_a_grip_move_its_row_and_keep_the_focus(self, browser):
        tab = open_page(browser)
        open_manager(tab)
        tab.focus("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=cfg] .mc-sm-grip")
        tab.keyboard.press("ArrowUp")
        tab.keyboard.press("ArrowUp")
        assert column_order(tab)[:5] == ["tools", "sampler", "cfg", "id:txt2img_hr", "size"]
        focused = tab.evaluate("""() => [document.activeElement.classList.contains('mc-sm-grip'),
            document.activeElement.closest('.mc-sm-row').getAttribute('data-mc-sm-key')]""")
        assert focused == [True, "cfg"]
        tab.context.close()

    def test_reset_puts_forges_order_back_everything_shown_and_open(self, browser):
        tab = open_page(browser)
        open_manager(tab)
        tab.evaluate("""() => { mcSettingsManager.move('txt2img', 'script', -5);
            mcSettingsManager.setShown('txt2img', 'sampler', false);
            mcSettingsManager.setOpen('txt2img', 'size', false); }""")
        tab.click("#mc-sm-list-txt2img .mc-sm-reset")
        found = tab.evaluate("mcSettingsManager.describe('txt2img')")
        assert [section["key"] for section in found] == TXT2IMG
        assert not any(section["hidden"] for section in found)
        assert all(section["open"] is not False for section in found)
        assert tab.evaluate("JSON.parse(localStorage.getItem('mc-settings-manager:txt2img'))") == {
            "v": 1, "order": TXT2IMG, "hidden": [], "open": {"manager": True}}
        tab.context.close()

    def test_reset_is_a_button_with_room_for_its_word_under_gradios_reset(self, browser):
        tab = open_page(browser)
        open_manager(tab)
        found = tab.evaluate("""() => { const button = document.querySelector('#mc-sm-list-txt2img .mc-sm-reset');
            const s = getComputedStyle(button);
            return [s.paddingLeft, s.paddingRight, button.scrollWidth <= button.clientWidth]; }""")
        assert found == ["12px", "12px", True]
        tab.context.close()

    def test_hiding_and_showing_move_no_block(self, browser):
        """A section hidden or shown is the same item, where it was: only its
        attribute changes, so nothing the page holds is taken out and put back."""
        tab = open_page(browser)
        moved = tab.evaluate("""() => new Promise((resolve) => {
            const list = document.getElementById('mc-sm-list-txt2img');
            let records = 0;
            const watch = new MutationObserver((found) => { records += found.length; });
            watch.observe(list, {childList: true});
            mcSettingsManager.setShown('txt2img', 'cfg', false);
            mcSettingsManager.setShown('txt2img', 'cfg', true);
            mcSettingsManager.setOpen('txt2img', 'seed', false);
            setTimeout(() => { watch.disconnect(); resolve(records); }, 50);
        })""")
        assert moved == 0
        tab.context.close()

    def test_each_tab_keeps_its_own_arrangement(self, browser):
        tab = open_page(browser)
        tab.evaluate("mcSettingsManager.move('img2img', 'denoise', -4)")
        assert column_order(tab, "txt2img") == TXT2IMG + ["manager"]
        assert column_order(tab, "img2img")[:3] == ["tools", "image", "denoise"]
        tab.context.close()


# --------------------------------------------------------------------------- #
# What Gradio does to the blocks afterwards
# --------------------------------------------------------------------------- #


class TestWhatGradioDoes:
    def test_a_section_with_nothing_to_draw_is_not_drawn_until_it_has(self, browser):
        """Override Settings is a row whose only form Gradio hides until a
        paste fills it."""
        tab = open_page(browser)
        assert not drawn(tab, "#mc-sm-list-txt2img [data-mc-sm-key=override]")
        tab.evaluate("""() => {
            document.querySelector('#txt2img_override_settings_row > .form').classList.remove('hidden');
            document.getElementById('txt2img_override_settings').classList.remove('hidden');
        }""")
        tab.wait_for_function("mcSettingsManager.describe('txt2img').find((s) => s.key == 'override').empty === false")
        assert drawn(tab, "#txt2img_override_settings")
        tab.evaluate("document.querySelector('#txt2img_override_settings_row > .form').classList.add('hidden')")
        tab.wait_for_function("mcSettingsManager.describe('txt2img').find((s) => s.key == 'override').empty === true")
        assert not drawn(tab, "#mc-sm-list-txt2img [data-mc-sm-key=override]")
        tab.context.close()

    def test_img2imgs_inpaint_section_comes_with_the_inpaint_tab(self, browser):
        tab = open_page(browser)
        assert not drawn(tab, "#mc-sm-list-img2img [data-mc-sm-key=inpaint]")
        tab.evaluate("document.getElementById('inpaint_controls').classList.remove('hide')")
        tab.wait_for_function("mcSettingsManager.describe('img2img').find((s) => s.key == 'inpaint').empty === false")
        assert drawn(tab, "#img2img_mask_blur")
        tab.context.close()

    def test_a_block_drawn_later_joins_the_list_before_the_manager(self, browser):
        tab = open_page(browser)
        tab.evaluate("""() => {
            const late = document.createElement('div');
            late.className = 'gr-group gradio-group svelte-grp';
            late.innerHTML = '<div class="styler svelte-grp"><div id="late_ext" class="block svelte-blk padded"><span data-testid="block-info">Late Extension</span><input class="field"></div></div>';
            document.getElementById('txt2img_settings').appendChild(late);
            hooks.after.forEach((fn) => fn());
        }""")
        order = column_order(tab)
        assert order[-2:] == ["label:Late Extension", "manager"]
        assert tab.evaluate("!!document.querySelector('[data-mc-sm-key=\"label:Late Extension\"] .mc-sm-acc #late_ext')")
        tab.context.close()

    def test_a_section_added_later_keeps_the_place_it_was_given(self, browser):
        tab = open_page(browser)
        tab.evaluate("""() => { localStorage.setItem('mc-settings-manager:txt2img', JSON.stringify(
            {v: 1, order: ['id:late_ext', 'tools'], hidden: [], open: {}})); }""")
        reload(tab)
        tab.evaluate("""() => {
            const late = document.createElement('div');
            late.innerHTML = '<div id="late_ext" class="block svelte-blk padded"><span data-testid="block-info">Late Extension</span><input class="field"></div>';
            document.getElementById('txt2img_settings').appendChild(late.firstElementChild);
            hooks.after.forEach((fn) => fn());
        }""")
        assert column_order(tab)[:2] == ["id:late_ext", "tools"]
        tab.context.close()


# --------------------------------------------------------------------------- #
# Off is off
# --------------------------------------------------------------------------- #


class TestOffIsOff:
    def test_turning_the_setting_off_puts_forges_columns_back_exactly(self, browser):
        plain = open_page(browser, option=False)
        assert plain.evaluate("!document.querySelector('.mc-sm-list')")
        untouched = {name: plain.evaluate(SIGNATURE, name) for name in ("txt2img", "img2img")}
        plain.context.close()

        tab = open_page(browser)
        tab.evaluate("""() => { mcSettingsManager.move('txt2img', 'seed', -4);
            mcSettingsManager.setShown('txt2img', 'cfg', false);
            mcSettingsManager.setOpen('txt2img', 'sampler', false); }""")
        tab.evaluate("() => { opts.model_chain_settings_manager = false; hooks.options.forEach((fn) => fn()); }")
        assert tab.evaluate("!document.querySelector('.mc-sm-list')")
        for name in ("txt2img", "img2img"):
            assert tab.evaluate(SIGNATURE, name) == untouched[name], name
        tab.evaluate("() => { opts.model_chain_settings_manager = true; hooks.options.forEach((fn) => fn()); }")
        found = tab.evaluate("mcSettingsManager.describe('txt2img')")
        assert [section["key"] for section in found][:2] == ["tools", "seed"]
        assert {section["key"] for section in found if section["hidden"]} == {"cfg"}
        tab.context.close()

    def test_with_the_setting_off_at_load_nothing_is_touched(self, browser):
        tab = open_page(browser, option=False)
        assert tab.evaluate("!document.querySelector('.mc-sm-list') && !document.querySelector('.mc-sm-emptied')")
        assert tab.evaluate("mcSettingsManager.describe('txt2img')") is None
        tab.context.close()


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #


class TestHousekeeping:
    def test_later_updates_on_a_settled_page_change_no_child_list(self, browser):
        """Forge runs every after-update callback 250 ms after any change to the
        page's child lists: a callback that changed one every time would run
        for ever."""
        tab = open_page(browser)
        open_manager(tab)
        changed = tab.evaluate("""() => new Promise((resolve) => {
            let records = 0;
            const watch = new MutationObserver((list) => { records += list.length; });
            watch.observe(document.querySelector('gradio-app'), {childList: true, subtree: true});
            hooks.after.forEach((fn) => fn()); hooks.after.forEach((fn) => fn());
            hooks.after.forEach((fn) => fn());
            setTimeout(() => { watch.disconnect(); resolve(records); }, 100);
        })""")
        assert changed == 0
        tab.context.close()

    def test_a_browser_that_keeps_nothing_still_lays_the_sections_out(self, browser):
        tab = open_page(browser, storage_throws=True)
        assert keys(tab) == TXT2IMG
        tab.evaluate("mcSettingsManager.move('txt2img', 'seed', -1)")
        assert column_order(tab)[4] == "seed"
        assert not tab.errors
        tab.context.close()

    def test_a_kept_arrangement_that_cannot_be_read_is_forges(self, browser):
        tab = open_page(browser)
        tab.evaluate("localStorage.setItem('mc-settings-manager:txt2img', '{not json')")
        reload(tab)
        assert keys(tab) == TXT2IMG
        tab.context.close()
