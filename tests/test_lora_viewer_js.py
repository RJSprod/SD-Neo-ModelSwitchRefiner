"""The LoRA viewer (javascript/model_chain_lora_viewer.js), run for real in Chromium.

The page is Forge's prompt area in the shape the script reads it: the tool row
with Forge's paste, clear and Apply Styles buttons, the prompt and the two
Literal boxes on Txt2Img, the prompt on Img2Img, and the tab bar above. The
route is the test's, answering as each test plans and keeping what was asked.
The viewer's own stylesheet section is loaded from ``style.css``.

What is held here is what the user would see go wrong: a slider that moves
when a finger only scrolled, an Apply that touches a character it should not,
an Escape that also interrupts a generation, a viewer that never stops waiting
for its details, a tap outside or a change of tab that applies (or reaches the
page under the viewer), a theme rule that stretches a trigger out of its row.
"""

from __future__ import annotations

import glob
import json
import os
import urllib.parse
from pathlib import Path

import pytest

import mc_lora_viewer

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_lora_viewer.js"
STYLE = ROOT / "style.css"
ORIGIN = "http://forge-lora-viewer.test"

PROMPT = ("cinematic portrait of a woman, realfilter, soft rim light "
          "<lora:realfilter_v2:0.85> <lora:contrast_slider:-6> <lora:old_style:0.7>")
LITERAL_POSITIVE = "<lora:detail_tweaker:-0.5>"
LITERAL_NEGATIVE = "film grain <lora:krea2_skin:1.2:0.5>"
IMG2IMG_PROMPT = "a watercolour fox <lora:paper_texture:0.6>"

DETAILS = {
    "realfilter_v2": {"found": True, "picture": "/pictures/realfilter.png", "title": "RealFilter",
                      "card": "realfilter_v2_final",
                      "triggers": ["realfilter", "film still", "35mm photograph",
                                   "natural skin texture", "muted colors"]},
    "contrast_slider": {"found": True, "picture": "", "title": "", "triggers": []},
    "old_style": {"found": False, "picture": "", "title": "", "triggers": []},
    "detail_tweaker": {"found": True, "picture": "/pictures/detail.png", "title": "",
                       "triggers": ["detailed"]},
    "krea2_skin": {"found": True, "picture": "", "title": "",
                   "triggers": ["k2skin", "visible pores"]},
    "paper_texture": {"found": True, "picture": "", "title": "", "triggers": ["paper"]},
}

# The names the add box is given, as the names route answers them: the active
# preset's folders already applied on the server.
NAMES = [
    {"name": "add_detail_xl", "tag": "add_detail_xl"},
    {"name": "Analog Film", "tag": "analog_film"},
    {"name": "detail_slider", "tag": "detail_slider"},
    {"name": "realfilter_v2_final", "tag": "realfilter_v2"},
    {"name": "soft_light_detail", "tag": "soft_light_detail"},
    {"name": "zoom_blur", "tag": "zoom_blur"},
]
DETAILS.update({
    "add_detail_xl": {"found": True, "picture": "/pictures/add.png", "title": "", "triggers": ["detailed"],
                      "weight": 0.6},
    "analog_film": {"found": True, "picture": "", "title": "", "triggers": ["analog"]},
    "detail_slider": {"found": True, "picture": "", "title": "", "triggers": []},
    "zoom_blur": {"found": True, "picture": "", "title": "", "triggers": []},
})

# Forge Neo's own stylesheet ("general gradio fixes") says the block fill is
# transparent, on :root and .dark both; a page that gave it a colour hid a
# viewer that showed the page through.
FORGE_CSS = """
:root, .dark { --block-background-fill: transparent; }
"""

HOST_CSS = """
:root { --background-fill-primary: #000; --border-color-primary: #262626;
        --body-text-color: #e6e6e6; --body-text-color-subdued: #8c8c8c; --input-background-fill: #161616;
        --button-secondary-background-fill: #1a1a1a; --button-secondary-background-fill-hover: #262626;
        --color-accent: #ffffff; }
*, ::before, ::after { box-sizing: border-box; }
body { margin: 0; background: #000; color: #e6e6e6; font: 14px sans-serif; }
#tabs > .tab-nav { position: sticky; top: 0; height: 44px; display: flex; gap: 4px; background: #000;
                   border-bottom: 1px solid #262626; z-index: 100; }
#tabs > .tab-nav button { width: 80px; }
.tabitem { padding: 8px; }
.block textarea { width: 100%; min-height: 60px; display: block; }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 8px 0; width: 100%; }
.row > .block { flex: 1 1 0; min-width: min(160px, 100%); }
.gradio-button.tool { width: 40px; height: 40px; display: grid; place-items: center; }
.spacer { height: 1600px; }
"""


# Forge Neo's own popup and description editor (javascript/extraNetworks.js),
# as it is: the editor is opened by a card's name read off the pressed
# element's grandparent, drawn on a popup appended to `.main`, and put away by
# its backdrop, its Cancel and Save, and Escape on the window.
FORGE_POPUP = """
let globalPopup = null;
let globalPopupInner = null;
function closePopup() {
    if (!globalPopup) return;
    globalPopup.style.display = "none";
}
function popup(contents) {
    if (!globalPopup) {
        globalPopup = document.createElement("div");
        globalPopup.classList.add("global-popup");
        let close = document.createElement("div");
        close.classList.add("global-popup-close");
        close.addEventListener("click", closePopup);
        close.title = "Close";
        globalPopup.appendChild(close);
        globalPopupInner = document.createElement("div");
        globalPopupInner.classList.add("global-popup-inner");
        globalPopup.appendChild(globalPopupInner);
        gradioApp().querySelector(".main").appendChild(globalPopup);
    }
    globalPopupInner.innerHTML = "";
    globalPopupInner.appendChild(contents);
    globalPopup.style.display = "flex";
}
let extraPageUserMetadataEditors = {};
function extraNetworksEditUserMetadata(event, tabname, extraPage) {
    let id = tabname + "_" + extraPage + "_edit_user_metadata";
    let editor = extraPageUserMetadataEditors[id];
    if (!editor) {
        editor = {};
        editor.page = gradioApp().getElementById(id);
        editor.nameTextarea = gradioApp().querySelector("#" + id + "_name" + " textarea");
        editor.button = gradioApp().querySelector("#" + id + "_button");
        extraPageUserMetadataEditors[id] = editor;
    }
    let cardName = event.target.parentElement.parentElement.getAttribute("data-name");
    if (cardName == null) {
        cardName = event.target.parentElement.parentElement.parentElement.getAttribute("data-name");
    }
    editor.nameTextarea.value = cardName;
    updateInput(editor.nameTextarea);
    editor.button.click();
    popup(editor.page);
    event.stopPropagation();
}
window.addEventListener("keydown", function (event) {
    if (event.key == "Escape") {
        closePopup();
    }
});
"""

FORGE_POPUP_CSS = """
.global-popup { display: flex; position: fixed; z-index: 1001; left: 0; top: 0; width: 100%; height: 100%;
                overflow: auto; }
.global-popup-close { position: fixed; left: 0; top: 0; width: 100%; height: 100%;
                      background-color: rgba(20, 20, 20, 0.95); }
.global-popup-inner { display: inline-block; margin: auto; padding: 2em; z-index: 1001; max-height: 90%;
                      max-width: 90%; position: relative; background: #111; }
"""


def editor_html(tab: str) -> str:
    """A tab's description editor for LoRA cards, as Forge builds it hidden in
    the page: the name box and the button its script presses, and Cancel and
    Save, each of which puts the popup away (`_js="closePopup"`)."""
    id_ = f"{tab}_lora_edit_user_metadata"
    return (f'<div id="{id_}" class="edit-user-metadata" style="display: none">'
            f'<div id="{id_}_name"><textarea></textarea></div>'
            f'<button id="{id_}_button" onclick="window.edited.push(document.querySelector('
            f"'#{id_}_name textarea').value); this.parentElement.style.display = 'block'\">open</button>"
            f'<button class="cancel" onclick="closePopup()">Cancel</button>'
            f'<button class="save" onclick="closePopup()">Save</button></div>')


def page_html(theme_css: str = "", cfg_one: bool = False, literals_script: bool = False) -> str:
    """The page. ``cfg_one`` is the Negative Prompt put away as the Literal
    script puts it away at CFG 1; ``literals_script`` loads that script as well,
    with the Literal row where Gradio built it, for the two scripts' placement
    together."""
    tools = """
        <button id="{p}paste" class="lg secondary gradio-button tool svelte-cmf5ev">&#x2199;&#xfe0f;</button>
        <button id="{tab}_clear_prompt" class="lg secondary gradio-button tool svelte-cmf5ev">&#x1f5d1;&#xfe0f;</button>
        <button id="{tab}_style_apply" class="lg secondary gradio-button tool svelte-cmf5ev">&#x1f4cb;</button>
        <button id="{tab}_restore_progress" class="lg secondary gradio-button tool svelte-cmf5ev" style="display: none">&#x1f300;</button>
        <span class="green-dot"></span>"""
    NEGATIVE_CLASS = " mc-literal-cfg-collapsed" if cfg_one else ""
    literal_tag = ('<script src="/javascript/model_chain_literals.js"></script>'
                   if literals_script else "")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{HOST_CSS}{FORGE_CSS}{FORGE_POPUP_CSS}{theme_css}</style>
<link rel="stylesheet" href="/style.css">
</head><body class="dark">
<script>
// Forge Neo's own (script.js): <gradio-app>, with getElementById patched
// through to the document, which Forge's description editor reads by.
function gradioApp() {{
    const elems = document.getElementsByTagName("gradio-app");
    const elem = elems.length == 0 ? document : elems[0];
    if (elem !== document) {{
        elem.getElementById = function (id) {{ return document.getElementById(id); }};
    }}
    return elem.shadowRoot ? elem.shadowRoot : elem;
}}
const hooks = {{loaded: [], after: []}};
function onUiLoaded(fn) {{ hooks.loaded.push(fn); }}
function onAfterUiUpdate(fn) {{ hooks.after.push(fn); }}
function updateInput(target) {{
    const event = new Event("input", {{bubbles: true}});
    Object.defineProperty(event, "target", {{value: target}});
    target.dispatchEvent(event);
}}
window.inputs = []; window.blurs = []; window.overlays = []; window.forgeEscapes = 0; window.copied = [];
window.edited = [];
{FORGE_POPUP}
document.addEventListener("input", (e) => window.inputs.push(e.target.closest("[id]").id), true);
document.addEventListener("blur", (e) => {{
    if (e.target && e.target.closest && e.target.closest(".block")) window.blurs.push(e.target.closest("[id]").id);
}}, true);
// Forge's own Escape listener (script.js): it interrupts a running generation.
document.addEventListener("keydown", (e) => {{ if (e.key === "Escape") window.forgeEscapes += 1; }});
document.addEventListener("minipaint:overlay", (e) => window.overlays.push(e.detail));
</script>
<gradio-app>
<div class="main">
{editor_html("txt2img")}{editor_html("img2img")}
<div id="tabs" class="tabs">
  <div class="tab-nav"><button id="nav-txt2img">Txt2img</button><button id="nav-img2img">Img2img</button></div>
  <div id="tab_txt2img" class="tabitem">
    <div id="txt2img_prompt_container" class="column">
      <div id="txt2img_prompt_row" class="row prompt-row">
        <div id="txt2img_prompt" class="block"><textarea>{PROMPT}</textarea></div>
      </div>
      <div id="txt2img_neg_prompt_row" class="row prompt-row">
        <div id="txt2img_neg_prompt" class="block{NEGATIVE_CLASS}"><textarea>blurry</textarea></div>
        <div id="mc-krea-creative-literal-row" class="row">
          <div id="mc-krea-creative-literal-positive" class="block"><textarea>{LITERAL_POSITIVE}</textarea></div>
          <div id="mc-krea-creative-literal-negative" class="block"><textarea>{LITERAL_NEGATIVE}</textarea></div>
        </div>
      </div>
    </div>
    <div id="txt2img_tools" class="row">{tools.format(p="", tab="txt2img")}</div>
    <div class="spacer"></div>
  </div>
  <div id="tab_img2img" class="tabitem" style="display: none;">
    <div id="img2img_prompt_container" class="column">
      <div id="img2img_prompt_row" class="row prompt-row">
        <div id="img2img_prompt" class="block"><textarea>{IMG2IMG_PROMPT}</textarea></div>
      </div>
      <div id="img2img_neg_prompt_row" class="row prompt-row">
        <div id="img2img_neg_prompt" class="block"><textarea></textarea></div>
      </div>
    </div>
    <div id="img2img_tools" class="row">{tools.format(p="i2i_", tab="img2img")}</div>
  </div>
</div>
</div>
</gradio-app>
{literal_tag}
<script src="/javascript/model_chain_lora_viewer.js"></script>
<script>hooks.loaded.forEach((fn) => fn()); hooks.after.forEach((fn) => fn());</script>
</body></html>"""


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


PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
                    "0000000c4944415408d763f8cfc0f01f0005000201e221bc330000000049454e44ae426082")


def answer(name):
    """The details route's answer for a name: the card is the file's name,
    which is the tag's own unless DETAILS names another."""
    found = dict(DETAILS.get(name, {"found": True, "picture": "", "title": "", "triggers": []}), name=name)
    found.setdefault("card", name if found.get("found") else "")
    return found


class Served:
    """The page, the stylesheet, the script, the pictures and the route."""

    def __init__(self, html: str):
        self.html = html
        self.asked: list[list[str]] = []
        self.hold = False
        self.held = []
        self.status = 200
        self.listed = 0
        self.hold_names = False
        self.omit = set()

    def __call__(self, route):
        request = route.request
        parts = urllib.parse.urlsplit(request.url)
        path = parts.path
        if path == "/":
            route.fulfill(status=200, content_type="text/html", body=self.html)
        elif path == "/style.css":
            route.fulfill(status=200, content_type="text/css", body=STYLE.read_text(encoding="utf-8"))
        elif path.startswith("/javascript/") and (ROOT / path[1:]).is_file():
            route.fulfill(status=200, content_type="text/javascript",
                          body=(ROOT / path[1:]).read_text(encoding="utf-8"))
        elif path.startswith("/pictures/"):
            route.fulfill(status=200, content_type="image/png", body=PNG)
        elif path == mc_lora_viewer.ROUTE:
            names = urllib.parse.parse_qs(parts.query).get("name", [])
            self.asked.append(names)
            if self.hold:
                self.held.append(route)
                return
            body = {"loras": [answer(name) for name in names if name not in self.omit]}
            route.fulfill(status=self.status, content_type="application/json", body=json.dumps(body))
        elif path == mc_lora_viewer.NAMES_ROUTE:
            self.listed += 1
            if self.hold_names:
                self.held.append(route)
                return
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps({"loras": NAMES, "preset": "xl", "filtered": True}))
        else:
            route.fulfill(status=404, body="")


def open_page(browser, width=900, height=800, theme_css="", clock=False, touch=False,
              cfg_one=False, literals_script=False):
    context = browser.new_context(viewport={"width": width, "height": height}, has_touch=touch)
    page = context.new_page()
    if clock:
        page.clock.install()
    served = Served(page_html(theme_css, cfg_one=cfg_one, literals_script=literals_script))
    page.route(ORIGIN + "/**", served)
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!window.mcLoraViewer")
    return page, served


def close_page(page, served):
    for route in served.held:
        try:
            route.abort()
        except Exception:
            pass
    page.context.close()


def open_viewer(page, tab="txt2img", wait_for_details=True):
    if tab == "txt2img":
        page.click("#mc-lora-viewer-txt2img")
    else:
        page.evaluate("(tab) => window.mcLoraViewer.open(tab)", tab)
    page.wait_for_selector(".mc-lv:not([hidden])")
    if wait_for_details:
        page.wait_for_function("!document.querySelector('.mc-lv .mc-lv-skeleton')")


ROWS = """() => [...document.querySelectorAll('.mc-lv-row')].map((row) => ({
    name: row.querySelector('.mc-lv-name').textContent,
    box: row.querySelector('.mc-lv-box').textContent,
    weight: row.querySelector('.mc-lv-weight').value,
    missing: row.hasAttribute('data-missing'),
    fixed: row.hasAttribute('data-fixed'),
    pills: [...row.querySelectorAll('.mc-lv-pill:not([hidden])')].map((p) => p.textContent),
    ticked: [...row.querySelectorAll('.mc-lv-pill[data-in]')].map((p) => p.textContent),
    picture: (row.querySelector('.mc-lv-thumb img') || {}).src || '',
    initials: (row.querySelector('.mc-lv-initials') || {}).textContent || '',
}))"""

BOXES = """() => ({
    prompt: document.querySelector('#txt2img_prompt textarea').value,
    positive: document.querySelector('#mc-krea-creative-literal-positive textarea').value,
    negative: document.querySelector('#mc-krea-creative-literal-negative textarea').value,
})"""


def row(page, index):
    return page.locator(".mc-lv-row").nth(index)


def weight_of(page, index):
    return row(page, index).locator(".mc-lv-weight").input_value()


SUGGESTED = """() => [...document.querySelectorAll('.mc-lv-suggest .mc-lv-option')].map((o) => ({
    name: o.querySelector('.mc-lv-option-name').textContent,
    tag: (o.querySelector('.mc-lv-option-tag') || {}).textContent || '',
    in: o.hasAttribute('data-in'), selected: o.getAttribute('aria-selected') === 'true',
}))"""


def typed(page, text):
    page.fill(".mc-lv-add-input", text)


# --------------------------------------------------------------------------- #
# The button
# --------------------------------------------------------------------------- #


BUTTON_PLACE = """(tab) => {
    const panel = document.getElementById('tab_' + tab);
    panel.style.display = 'block';
    const button = document.getElementById('mc-lora-viewer-' + tab);
    const holder = button.parentElement;
    const box = (id) => document.getElementById(id).getBoundingClientRect();
    const own = button.getBoundingClientRect();
    return {
        holder: holder.id, after: holder.previousElementSibling.id,
        before: holder.nextElementSibling ? holder.nextElementSibling.id : null,
        text: button.textContent.trim(), svg: !!button.querySelector('svg'),
        count: document.querySelectorAll('#tab_' + tab + ' .mc-lv-open').length,
        inTools: !!document.querySelector('#' + tab + '_tools .mc-lv-open'),
        width: Math.round(own.width), promptWidth: Math.round(box(tab + '_prompt').width),
        left: Math.round(own.left), promptLeft: Math.round(box(tab + '_prompt').left),
        top: Math.round(own.top), promptBottom: Math.round(box(tab + '_prompt').bottom),
        negativeBottom: Math.round(box(tab + '_neg_prompt').bottom),
    };
}"""


class TestTheButton:
    """"a button that is named "Lora Viewer" and it should appear below the
    negative prompt box in the image generation tab. Make it as wide as the
    prompt input fields and just below negative (or positive when CFG = 1), and
    above the literals" (2026-11-06)."""

    def test_one_lora_viewer_button_under_the_negative_prompt_as_wide_as_the_boxes(self, browser):
        page, served = open_page(browser)
        try:
            for tab, before in (("txt2img", "mc-krea-creative-literal-row"), ("img2img", None)):
                found = page.evaluate(BUTTON_PLACE, tab)
                assert found["after"] == tab + "_neg_prompt", found
                assert found["before"] == before, found
                assert found["text"] == "Lora Viewer" and found["svg"], found
                assert found["count"] == 1 and not found["inTools"], found
                assert abs(found["width"] - found["promptWidth"]) <= 1, found
                assert abs(found["left"] - found["promptLeft"]) <= 1, found
                assert found["top"] >= found["negativeBottom"], found
        finally:
            close_page(page, served)

    def test_at_cfg_one_it_is_the_line_under_the_prompt(self, browser):
        """The Literal script puts the Negative Prompt away while CFG is 1."""
        page, served = open_page(browser, cfg_one=True)
        try:
            found = page.evaluate(BUTTON_PLACE, "txt2img")
            assert found["after"] == "txt2img_neg_prompt", found
            assert 0 <= found["top"] - found["promptBottom"] <= 20, found
            assert abs(found["width"] - found["promptWidth"]) <= 1, found
        finally:
            close_page(page, served)

    def test_with_the_literal_script_the_row_follows_it_and_neither_moves_again(self, browser):
        """Both scripts place their element after the Negative Prompt on every
        update. Anchored on it alone, each would move its element back in front
        of the other's on the next update -- so an odd number of updates is what
        shows a to-and-fro."""
        page, served = open_page(browser, literals_script=True)
        try:
            order = page.evaluate("""() => Array.from(
                document.getElementById('txt2img_neg_prompt_row').children).map((el) => el.id)""")
            assert order == ["txt2img_neg_prompt", "mc-lora-viewer-launch-txt2img",
                             "mc-krea-creative-literal-row"], order
            changed = page.evaluate("""() => new Promise((resolve) => {
                let records = 0;
                const watch = new MutationObserver((list) => { records += list.length; });
                watch.observe(document.querySelector('gradio-app'), {childList: true, subtree: true});
                hooks.after.forEach((fn) => fn()); hooks.after.forEach((fn) => fn());
                hooks.after.forEach((fn) => fn());
                setTimeout(() => { watch.disconnect(); resolve(records); }, 50);
            })""")
            assert changed == 0
        finally:
            close_page(page, served)

    def test_every_later_update_finds_it_there_and_changes_nothing(self, browser):
        """An after-update callback that changed a child list on every run would
        raise Forge's next update itself (the page's update loop)."""
        page, served = open_page(browser)
        try:
            changed = page.evaluate("""() => new Promise((resolve) => {
                let records = 0;
                const watch = new MutationObserver((list) => { records += list.length; });
                watch.observe(document.querySelector('gradio-app'), {childList: true, subtree: true});
                hooks.after.forEach((fn) => fn()); hooks.after.forEach((fn) => fn());
                hooks.after.forEach((fn) => fn());
                setTimeout(() => { watch.disconnect(); resolve(records); }, 50);
            })""")
            assert changed == 0
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# Reading, and the details
# --------------------------------------------------------------------------- #


class TestWhatItReads:
    def test_every_tag_in_the_prompt_and_both_literal_boxes_in_screen_order(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            rows = page.evaluate(ROWS)
            assert [(r["name"], r["box"], r["weight"]) for r in rows] == [
                ("realfilter_v2", "Prompt", "0.85"), ("contrast_slider", "Prompt", "-6"),
                ("old_style", "Prompt", "0.7"), ("detail_tweaker", "Literal +", "-0.5"),
                ("krea2_skin", "Literal −", "1.2")]
            assert page.locator(".mc-lv-count").text_content() == "5 in this prompt"
            assert served.asked == [["realfilter_v2", "contrast_slider", "old_style",
                                     "detail_tweaker", "krea2_skin"]]
        finally:
            close_page(page, served)

    def test_img2img_reads_its_own_prompt(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page, "img2img")
            rows = page.evaluate(ROWS)
            assert [(r["name"], r["box"], r["weight"]) for r in rows] == [
                ("paper_texture", "Prompt", "0.6")]
        finally:
            close_page(page, served)

    def test_pictures_initials_triggers_and_a_lora_not_on_disk(self, browser):
        page, served = open_page(browser, width=1200)
        try:
            open_viewer(page)
            rows = {r["name"]: r for r in page.evaluate(ROWS)}
            assert rows["realfilter_v2"]["picture"].endswith("/pictures/realfilter.png")
            assert rows["contrast_slider"]["initials"] == "CS"
            assert rows["contrast_slider"]["pills"] == []
            assert rows["old_style"]["missing"] is True
            assert rows["realfilter_v2"]["pills"] == DETAILS["realfilter_v2"]["triggers"]
            note = row(page, 2).locator(".mc-lv-note").text_content()
            assert note == "Not in your LoRA folders"
        finally:
            close_page(page, served)

    def test_a_trigger_already_in_the_prompts_words_is_ticked_and_a_tags_name_is_not_a_word(self, browser):
        """"realfilter" is in the prompt as a word, and "paper" only as the
        name inside its own tag, ``<lora:paper:1>``: a LoRA whose trigger is its
        own name would otherwise always read as already written."""
        page, served = open_page(browser, width=1200)
        try:
            page.evaluate("""() => {
                document.querySelector('#mc-krea-creative-literal-negative textarea').value =
                    'film grain <lora:paper:1> artistic';
            }""")
            DETAILS["paper"] = {"found": True, "picture": "", "title": "", "triggers": ["paper", "art"]}
            try:
                open_viewer(page)
                rows = {r["name"]: r for r in page.evaluate(ROWS)}
            finally:
                del DETAILS["paper"]
            assert rows["realfilter_v2"]["ticked"] == ["realfilter"]
            assert rows["paper"]["ticked"] == [], "a tag's name, and a word inside a longer one, are not it"
        finally:
            close_page(page, served)

    def test_the_details_request_has_a_deadline_and_the_rows_still_work(self, browser):
        page, served = open_page(browser, clock=True)
        served.hold = True
        try:
            open_viewer(page, wait_for_details=False)
            assert page.locator(".mc-lv-status").is_hidden()
            page.clock.run_for(9000)
            page.wait_for_selector(".mc-lv-status:not([hidden])")
            assert page.locator(".mc-lv-status span").first.text_content() == \
                "Pictures and trigger words did not arrive in time."
            assert row(page, 0).locator(".mc-lv-note").text_content() == "No details"
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            assert weight_of(page, 0) == "0.9"
            served.hold = False
            page.click(".mc-lv-retry")
            page.wait_for_function("!document.querySelector('.mc-lv .mc-lv-skeleton')")
            assert page.locator(".mc-lv-status").is_hidden()
            assert len(served.asked) == 2
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# Changing a weight
# --------------------------------------------------------------------------- #


class TestTheSlider:
    def test_a_mouse_drag_moves_it_in_steps_of_five_hundredths(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            knob = row(page, 0).locator(".mc-lv-knob").bounding_box()
            track = row(page, 0).locator(".mc-lv-track").bounding_box()
            x, y = knob["x"] + knob["width"] / 2, knob["y"] + knob["height"] / 2
            page.mouse.move(x, y)
            page.mouse.down()
            inner = track["width"] - 22
            # A tenth of the range (+1) and three pixels more: between two steps.
            page.mouse.move(x + inner * 0.1 + 3, y, steps=4)
            dragging = page.evaluate("document.querySelector('.mc-lv-track').classList.contains('mc-lv-dragging')")
            bubble = row(page, 0).locator(".mc-lv-bubble").text_content()
            page.mouse.up()
            value = float(weight_of(page, 0))
            assert dragging and bubble == weight_of(page, 0)
            assert abs(value - 1.9) <= 0.051
            assert round(value / 0.05, 6) == round(value / 0.05)
        finally:
            close_page(page, served)

    def test_a_vertical_swipe_never_moves_it_and_a_sideways_one_does(self, browser):
        page, served = open_page(browser, touch=True)
        try:
            open_viewer(page)
            moved = page.evaluate("""() => {
                const track = document.querySelectorAll('.mc-lv-track')[0];
                const box = track.getBoundingClientRect();
                const y = box.top + box.height / 2;
                const x = box.left + 40;   // on the track, away from the knob
                const fire = (type, cx, cy, id) => track.dispatchEvent(new PointerEvent(type, {
                    bubbles: true, cancelable: true, pointerId: id, pointerType: 'touch',
                    clientX: cx, clientY: cy, isPrimary: true}));
                const weight = () => document.querySelectorAll('.mc-lv-weight')[0].value;
                fire('pointerdown', x, y, 7);
                fire('pointermove', x + 2, y + 12, 7);
                fire('pointermove', x + 30, y + 40, 7);
                fire('pointerup', x + 30, y + 40, 7);
                const afterSwipe = weight();
                fire('pointerdown', x, y, 8);
                fire('pointermove', x + 20, y + 2, 8);
                fire('pointermove', x + 60, y + 3, 8);
                fire('pointerup', x + 60, y + 3, 8);
                return {afterSwipe, afterDrag: weight(),
                        action: getComputedStyle(track).touchAction};
            }""")
            assert moved["afterSwipe"] == "0.85"
            assert moved["afterDrag"] != "0.85"
            assert moved["action"] == "pan-y"
        finally:
            close_page(page, served)

    def test_a_tap_on_the_track_puts_the_knob_there(self, browser):
        page, served = open_page(browser, touch=True)
        try:
            open_viewer(page)
            value = page.evaluate("""() => {
                const track = document.querySelectorAll('.mc-lv-track')[0];
                const box = track.getBoundingClientRect();
                const x = box.left + 11 + (box.width - 22) * 0.25;   // a quarter: -2.5
                const y = box.top + box.height / 2;
                for (const type of ['pointerdown', 'pointerup']) {
                    track.dispatchEvent(new PointerEvent(type, {bubbles: true, cancelable: true,
                        pointerId: 3, pointerType: 'touch', clientX: x, clientY: y, isPrimary: true}));
                }
                return document.querySelectorAll('.mc-lv-weight')[0].value;
            }""")
            assert value == "-2.5"
        finally:
            close_page(page, served)

    def test_the_keys_step_it(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            track = row(page, 0).locator(".mc-lv-track")
            track.focus()
            page.keyboard.press("ArrowRight")
            assert weight_of(page, 0) == "0.9"
            page.keyboard.press("PageUp")
            assert weight_of(page, 0) == "1.4"
            page.keyboard.press("Home")
            assert weight_of(page, 0) == "-5"
        finally:
            close_page(page, served)

    def test_minus_and_plus_step_to_the_next_twentieth(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            plus = row(page, 0).locator(".mc-lv-step").nth(1)
            minus = row(page, 0).locator(".mc-lv-step").nth(0)
            plus.click()
            assert weight_of(page, 0) == "0.9"
            box = row(page, 0).locator(".mc-lv-weight")
            box.fill("0.123")
            box.press("Enter")
            plus.click()
            assert weight_of(page, 0) == "0.15"
            minus.click()
            minus.click()
            assert weight_of(page, 0) == "0.05"
        finally:
            close_page(page, served)


class TestTheWeightBox:
    def test_any_number_beyond_the_slider_is_kept_and_the_knob_is_hollow_at_the_end(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            box = row(page, 0).locator(".mc-lv-weight")
            box.fill("7.5")
            box.press("Enter")
            state = page.evaluate("""() => {
                const row = document.querySelectorAll('.mc-lv-row')[0];
                const knob = row.querySelector('.mc-lv-knob');
                const track = row.querySelector('.mc-lv-track').getBoundingClientRect();
                const k = knob.getBoundingClientRect();
                return {beyond: knob.classList.contains('mc-lv-beyond'),
                        atEnd: Math.abs((k.left + k.width / 2) - (track.right - 11)) < 1.5};
            }""")
            assert weight_of(page, 0) == "7.5"
            assert state == {"beyond": True, "atEnd": True}
            negative = page.evaluate("""() => {
                const knob = document.querySelectorAll('.mc-lv-row')[1].querySelector('.mc-lv-knob');
                return [knob.classList.contains('mc-lv-beyond'), knob.classList.contains('mc-lv-negative')];
            }""")
            assert negative == [True, True], "contrast_slider opened at -6"
        finally:
            close_page(page, served)

    def test_text_that_is_not_a_number_is_marked_and_put_back_on_leaving(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            box = row(page, 0).locator(".mc-lv-weight")
            box.fill("abc")
            assert box.get_attribute("data-bad") is not None
            box.press("Enter")
            assert weight_of(page, 0) == "0.85"
            assert box.get_attribute("data-bad") is None
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# Apply, Cancel, Escape, a change of tab
# --------------------------------------------------------------------------- #


class TestApply:
    def test_apply_writes_each_changed_weight_into_its_own_tag_and_nothing_else(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()          # 0.85 -> 0.9
            row(page, 4).locator(".mc-lv-step").nth(0).click()          # 1.2 -> 1.15, its :0.5 kept
            assert page.locator(".mc-lv-button-apply").text_content() == "Apply 2 changes"
            page.click(".mc-lv-button-apply")
            boxes = page.evaluate(BOXES)
            assert boxes == {
                "prompt": PROMPT.replace("<lora:realfilter_v2:0.85>", "<lora:realfilter_v2:0.9>"),
                "positive": LITERAL_POSITIVE,
                "negative": "film grain <lora:krea2_skin:1.15:0.5>",
            }
            assert page.evaluate("window.inputs") == ["txt2img_prompt", "mc-krea-creative-literal-negative"]
            assert page.evaluate("window.blurs") == ["mc-krea-creative-literal-negative"]
            assert page.locator(".mc-lv").is_hidden()
        finally:
            close_page(page, served)

    def test_removing_takes_the_tag_out_and_closes_the_gap(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("""() => {
                const box = document.querySelector('#txt2img_prompt textarea');
                box.value = 'a cat, <lora:a:1>, on a mat <lora:b:0.5>';
            }""")
            open_viewer(page)
            row(page, 0).locator(".mc-lv-remove").click()
            row(page, 1).locator(".mc-lv-remove").click()
            row(page, 1).locator(".mc-lv-remove").click()     # pressed again: kept
            assert row(page, 1).locator(".mc-lv-remove").get_attribute("aria-pressed") == "false"
            page.click(".mc-lv-button-apply")
            assert page.evaluate(BOXES)["prompt"] == "a cat, on a mat <lora:b:0.5>"
        finally:
            close_page(page, served)

    def test_a_row_and_every_row_go_back_to_where_they_opened(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            assert page.locator(".mc-lv-reset-all").is_hidden()
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            reset = row(page, 0).locator(".mc-lv-reset")
            assert reset.is_visible()
            reset.click()
            assert weight_of(page, 0) == "0.85" and reset.is_hidden()
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            row(page, 3).locator(".mc-lv-remove").click()
            page.click(".mc-lv-reset-all")
            assert weight_of(page, 0) == "0.85"
            assert page.locator(".mc-lv-row[data-removed]").count() == 0
            assert page.locator(".mc-lv-button-apply").text_content() == "Apply"
            page.click(".mc-lv-button-apply")
            assert page.evaluate(BOXES) == {"prompt": PROMPT, "positive": LITERAL_POSITIVE,
                                            "negative": LITERAL_NEGATIVE}
            assert page.evaluate("window.inputs") == []
        finally:
            close_page(page, served)

    def test_a_box_that_changed_while_open_is_left_alone_and_the_page_says_so(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            row(page, 3).locator(".mc-lv-step").nth(1).click()
            page.evaluate("""() => {
                document.querySelector('#txt2img_prompt textarea').value += ' rain';
            }""")
            page.click(".mc-lv-button-apply")
            boxes = page.evaluate(BOXES)
            assert boxes["prompt"] == PROMPT + " rain"
            assert boxes["positive"] == "<lora:detail_tweaker:-0.45>"
            assert "changed while the viewer was open" in page.locator(".mc-lv-toast").text_content()
        finally:
            close_page(page, served)

    def test_a_tag_that_sets_its_weights_by_name_is_shown_fixed_and_left_as_typed(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("""() => {
                document.querySelector('#txt2img_prompt textarea').value = 'a cat <lora:x:te=0.5:unet=0.8>';
            }""")
            open_viewer(page)
            first = page.evaluate(ROWS)[0]
            assert first["name"] == "x" and first["fixed"] is True and first["weight"] == "named"
            assert row(page, 0).locator(".mc-lv-step").nth(1).is_disabled()
            page.click(".mc-lv-button-apply")
            assert page.evaluate(BOXES)["prompt"] == "a cat <lora:x:te=0.5:unet=0.8>"
        finally:
            close_page(page, served)


class TestClosingWithoutApplying:
    def test_cancel_changes_nothing(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            row(page, 1).locator(".mc-lv-remove").click()
            page.click(".mc-lv-button-cancel")
            assert page.evaluate(BOXES) == {"prompt": PROMPT, "positive": LITERAL_POSITIVE,
                                            "negative": LITERAL_NEGATIVE}
            assert page.evaluate("window.inputs") == []
            assert page.evaluate("document.activeElement.id") == "mc-lora-viewer-txt2img"
        finally:
            close_page(page, served)

    def test_escape_cancels_and_reaches_nothing_else(self, browser):
        """Forge's own Escape interrupts a running generation."""
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            page.evaluate("document.activeElement.blur()")
            page.keyboard.press("Escape")
            assert page.locator(".mc-lv").is_hidden()
            assert page.evaluate("window.forgeEscapes") == 0
            assert page.evaluate(BOXES)["prompt"] == PROMPT
            page.keyboard.press("Escape")
            assert page.evaluate("window.forgeEscapes") == 1, "closed, Escape is Forge's again"
        finally:
            close_page(page, served)

    def test_a_change_of_tab_cancels(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            page.evaluate("""() => {
                document.getElementById('tab_txt2img').style.display = 'none';
                document.getElementById('tab_img2img').style.display = '';
            }""")
            page.wait_for_selector(".mc-lv", state="hidden")
            assert page.evaluate(BOXES)["prompt"] == PROMPT
        finally:
            close_page(page, served)

    def test_it_tells_the_page_when_it_takes_it_and_gives_it_back(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            page.click(".mc-lv-button-cancel")
            assert page.evaluate("window.overlays") == [
                {"name": "lora-viewer", "open": True, "modal": True},
                {"name": "lora-viewer", "open": False, "modal": True}]
        finally:
            close_page(page, served)


BELOW = """() => {
    // What a tap would reach under the viewer: clicks the page's own
    // listeners hear, outside the viewer, after the window's capture phase.
    window.below = [];
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.mc-lv')) window.below.push(e.target.id || e.target.className || e.target.tagName);
    });
}"""

TOAST = """() => { const node = document.querySelector('.mc-lv-toast');
                   return node.hidden ? null : node.textContent; }"""


class TestATapOutside:
    """"Tapping outside of that lora pop up should close it" (2026-11-06): as
    Escape does -- Cancel, nothing written -- and the page says so when that
    threw changes away. Until then a tap outside did nothing."""

    def test_a_tap_outside_closes_it_with_nothing_written_and_says_so(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            assert weight_of(page, 0) == "0.9"
            page.evaluate(BELOW)
            page.mouse.click(4, 700)                     # the scrim, beside the viewer
            assert page.locator(".mc-lv").is_hidden() and page.locator(".mc-lv-scrim").is_hidden()
            assert page.evaluate(BOXES) == {"prompt": PROMPT, "positive": LITERAL_POSITIVE,
                                            "negative": LITERAL_NEGATIVE}
            assert page.evaluate("window.inputs") == []
            assert page.evaluate(TOAST) == "Closed without applying: the change was not written."
            assert page.evaluate("window.below") == [], "the tap is taken, not handed to the page"
            assert page.evaluate("document.activeElement.id") == "mc-lora-viewer-txt2img"
            assert page.evaluate("window.overlays")[-1] == {"name": "lora-viewer", "open": False,
                                                           "modal": True}
            page.keyboard.press("Escape")
            assert page.evaluate("window.forgeEscapes") == 1, "closed, Escape is Forge's again"
        finally:
            close_page(page, served)

    def test_with_nothing_changed_it_closes_and_says_nothing(self, browser):
        page, served = open_page(browser, width=1600, height=900)
        try:
            open_viewer(page)
            page.mouse.click(1500, 500)                  # far right of a centred 880 px viewer
            assert page.locator(".mc-lv").is_hidden()
            assert page.evaluate(TOAST) is None
        finally:
            close_page(page, served)

    def test_a_press_that_moved_or_began_inside_is_no_tap(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            page.mouse.move(4, 600)
            page.mouse.down()
            page.mouse.move(4, 660, steps=4)              # a swipe on the scrim
            page.mouse.up()
            assert page.locator(".mc-lv").is_visible()
            knob = row(page, 0).locator(".mc-lv-knob").bounding_box()
            page.mouse.move(knob["x"] + knob["width"] / 2, knob["y"] + knob["height"] / 2)
            page.mouse.down()
            page.mouse.move(2, knob["y"] + knob["height"] / 2, steps=6)   # dragged past the edge
            page.mouse.up()
            assert page.locator(".mc-lv").is_visible(), "a press begun inside is the viewer's"
            assert weight_of(page, 0) == "-5"
            edge = page.locator(".mc-lv-foot").bounding_box()
            page.mouse.move(edge["x"] + 3, edge["y"] + edge["height"] / 2)
            page.mouse.down()
            page.mouse.move(edge["x"] - 5, edge["y"] + edge["height"] / 2)   # 8 px, onto the scrim
            page.mouse.up()
            assert page.locator(".mc-lv").is_visible(), "begun inside, let go just outside"
        finally:
            close_page(page, served)

    def test_on_a_phone_a_finger_on_the_tab_bar_closes_it(self, browser):
        """On a phone the viewer leaves 8 px of scrim at its sides, and the
        browser's touch adjustment gives a finger there to the row beside it
        (as it would a real one): what is outside is the tab bar above."""
        page, served = open_page(browser, width=390, height=800, touch=True)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).tap()
            page.touchscreen.tap(3, 600)                 # adjusted onto the row: the viewer's
            assert page.locator(".mc-lv").is_visible()
            page.touchscreen.tap(300, 22)                # the bar, where no tab is
            page.wait_for_selector(".mc-lv", state="hidden")
            assert page.evaluate(BOXES)["prompt"] == PROMPT
            assert page.evaluate(TOAST) == "Closed without applying: the change was not written."
        finally:
            close_page(page, served)

    def test_the_tab_bar_still_answers_and_a_tap_on_it_closes_the_viewer(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("""() => {
                window.navPresses = 0;
                document.getElementById('nav-img2img').addEventListener('click', () => {
                    window.navPresses += 1;
                    document.getElementById('tab_txt2img').style.display = 'none';
                    document.getElementById('tab_img2img').style.display = '';
                });
            }""")
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            page.click("#nav-img2img")
            assert page.locator(".mc-lv").is_hidden()
            assert page.evaluate("window.navPresses") == 1, "the tab changes too"
            assert page.evaluate(BOXES)["prompt"] == PROMPT
        finally:
            close_page(page, served)

    def test_stepped_aside_for_forges_description_a_tap_is_forges(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-thumb").click()
            page.wait_for_selector(".global-popup", state="visible")
            page.mouse.click(4, 700)                     # Forge's backdrop puts its popup away
            page.wait_for_selector(".mc-lv:not([hidden])")
            assert page.locator(".mc-lv").is_visible(), "back, not closed"
        finally:
            close_page(page, served)


class TestTheAddBoxHasTheFocus:
    """"When i open the lora panel we created, the text field should be auto
    selected, cursor in the field, ready for me to type a character"
    (2026-11-06)."""

    def test_the_press_that_opens_it_leaves_the_cursor_in_the_add_box(self, browser):
        page, served = open_page(browser)
        try:
            with page.expect_request(lambda request: mc_lora_viewer.NAMES_ROUTE in request.url):
                open_viewer(page)
            assert page.evaluate("document.activeElement.className") == "mc-lv-add-input"
            assert page.locator(".mc-lv-suggest").is_hidden(), "an empty box suggests nothing"
            page.keyboard.type("det")
            page.wait_for_selector(".mc-lv-option")
            assert [o["name"] for o in page.evaluate(SUGGESTED)] == \
                ["detail_slider", "add_detail_xl", "soft_light_detail"]
        finally:
            close_page(page, served)

    def test_on_a_phone_too_and_on_img2img(self, browser):
        page, served = open_page(browser, width=390, height=800, touch=True)
        try:
            page.tap("#mc-lora-viewer-txt2img")
            page.wait_for_selector(".mc-lv:not([hidden])")
            assert page.evaluate("document.activeElement.className") == "mc-lv-add-input"
            page.click(".mc-lv-button-cancel")
            page.evaluate("(tab) => window.mcLoraViewer.open(tab)", "img2img")
            assert page.evaluate("document.activeElement.className") == "mc-lv-add-input"
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# Where it is, and how it looks
# --------------------------------------------------------------------------- #


class TestPlacement:
    def test_under_the_tab_bar_above_the_shared_dialog_layer_and_the_bar_still_answers(self, browser):
        page, served = open_page(browser, width=390, height=800)
        try:
            page.evaluate("window.scrollTo(0, 400)")    # the theme's bar is sticky
            open_viewer(page)
            found = page.evaluate("""() => {
                const box = document.querySelector('.mc-lv').getBoundingClientRect();
                const bar = document.querySelector('#tabs > .tab-nav').getBoundingClientRect();
                const at = document.elementFromPoint(40, bar.top + bar.height / 2);
                return {top: box.top, left: box.left, right: box.right, bottom: box.bottom,
                        bar: bar.bottom, barAnswers: !!(at && at.closest('.tab-nav')),
                        layer: Number(getComputedStyle(document.querySelector('.mc-lv')).zIndex),
                        width: innerWidth, height: innerHeight};
            }""")
            assert found["top"] == found["bar"] + 8
            assert found["left"] == 8 and found["right"] == found["width"] - 8
            assert found["bottom"] == found["height"] - 8
            assert found["layer"] >= 2000
            assert found["barAnswers"] is True
        finally:
            close_page(page, served)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_beside_a_tab_bar_that_runs_down_one_side_and_the_bar_still_answers(self, browser, side):
        """The Forge Assistant's Transpose View puts the tab bar down the
        window's left or right side (the theme lays it out, docs/22 §3.47):
        the viewer then starts at the window's top, beside the rail rather than
        over it or under a bar that is not there."""
        other = "right" if side == "left" else "left"
        rail = (f"#tabs > .tab-nav {{ position: fixed; top: 0; bottom: 0; {side}: 0; {other}: auto; "
                "width: 48px; height: auto; flex-direction: column; border-bottom: 0; }"
                "#tabs > .tab-nav button { width: 40px; height: 40px; }")
        page, served = open_page(browser, width=390, height=800, theme_css=rail)
        try:
            open_viewer(page)
            found = page.evaluate("""() => {
                const box = document.querySelector('.mc-lv').getBoundingClientRect();
                const bar = document.querySelector('#tabs > .tab-nav').getBoundingClientRect();
                const at = document.elementFromPoint(bar.left + bar.width / 2, 20);
                return {top: box.top, left: box.left, right: box.right,
                        barLeft: bar.left, barRight: bar.right,
                        barAnswers: !!(at && at.closest('.tab-nav')), width: innerWidth};
            }""")
            assert found["top"] == 8, "no bar above it"
            if side == "left":
                assert found["barRight"] == 48 and found["left"] == 48 + 8 and found["right"] == found["width"] - 8
            else:
                assert found["barLeft"] == found["width"] - 48
                assert found["left"] == 8 and found["right"] == found["width"] - 48 - 8
            assert found["barAnswers"] is True
        finally:
            close_page(page, served)

    def test_on_a_wide_window_it_is_centred_and_no_wider_than_880(self, browser):
        page, served = open_page(browser, width=1600, height=900)
        try:
            open_viewer(page)
            box = page.locator(".mc-lv").bounding_box()
            assert box["width"] == 880
            assert abs((box["x"] + box["width"] / 2) - 800) < 1
        finally:
            close_page(page, served)

    def test_on_a_phone_the_pills_that_do_not_fit_fold_into_a_count_that_opens(self, browser):
        page, served = open_page(browser, width=390, height=800)
        try:
            open_viewer(page)
            first = page.evaluate(ROWS)[0]
            assert first["pills"][-1].startswith("+")
            shown = len(first["pills"]) - 1
            assert 1 <= shown < 5 and first["pills"][-1] == f"+{5 - shown}"
            row(page, 0).locator(".mc-lv-more").click()
            assert page.evaluate(ROWS)[0]["pills"] == DETAILS["realfilter_v2"]["triggers"]
        finally:
            close_page(page, served)

    def test_a_themes_button_rule_does_not_stretch_a_trigger_out_of_its_row(self, browser):
        """The Lobe theme says `min-width: fit-content !important` about every button."""
        lobe = "button { min-width: fit-content !important; }"
        page, served = open_page(browser, width=1200, theme_css=lobe)
        try:
            page.evaluate("""() => {
                document.querySelector('#txt2img_prompt textarea').value = 'x <lora:paper_texture:1>';
            }""")
            served_long = "a very long trigger phrase that a Civitai page might well carry"
            page.evaluate(f"""() => {{ window.__long = {json.dumps(served_long)}; }}""")
            DETAILS["paper_texture"]["triggers"] = [served_long]
            try:
                open_viewer(page)
                width = row(page, 0).locator(".mc-lv-pill").bounding_box()["width"]
            finally:
                DETAILS["paper_texture"]["triggers"] = ["paper"]
            assert width <= 200
        finally:
            close_page(page, served)


class TestCopying:
    def test_a_tap_copies_the_one_trigger_and_says_so(self, browser):
        page, served = open_page(browser, width=1200)
        try:
            page.evaluate("""() => {
                Object.defineProperty(navigator, 'clipboard', {value: {
                    writeText: (text) => { window.copied.push(text); return Promise.resolve(); }}});
            }""")
            open_viewer(page)
            pill = row(page, 0).locator(".mc-lv-pill").nth(1)
            pill.click()
            page.wait_for_function("window.copied.length === 1")
            assert page.evaluate("window.copied") == ["film still"]
            assert pill.text_content() == "Copied"
            page.wait_for_function(
                "document.querySelectorAll('.mc-lv-row')[0].querySelectorAll('.mc-lv-pill')[1].textContent === 'film still'")
            assert page.evaluate(BOXES)["prompt"] == PROMPT, "copying writes nothing"
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# The assistant leaves Escape to the viewer
# --------------------------------------------------------------------------- #


def test_the_assistant_leaves_escape_to_an_open_viewer():
    from test_assistant_js import run

    found = run("""
        const shell = Object.create(NS.Shell.prototype);
        shell.state = {panelOpen: true, tapToClose: true};
        shell.nodes = {root: {contains: () => false}, menu: {hidden: true}};
        shell.focus = {isActive: () => true};
        shell.lookEditor = null;
        const acted = [];
        shell.toggleFocus = () => acted.push("exit-focus");
        shell.close = () => acted.push("close-panel");
        shell.closeMenu = () => acted.push("close-menu");
        const escape = () => ({key: "Escape", preventDefault() {}, stopPropagation() {}});
        globalThis.window.mcLoraViewer = {isOpen: () => true};
        shell.documentKey(escape());
        const whileOpen = acted.slice();
        globalThis.window.mcLoraViewer = {isOpen: () => false};
        shell.documentKey(escape());
        console.log(JSON.stringify({whileOpen, afterwards: acted}));
    """, sources=("shell", "host"))
    assert found["whileOpen"] == []
    assert found["afterwards"] != [], "closed, the panel's Escape is its own again"


# --------------------------------------------------------------------------- #
# Opaque, and adding one
# --------------------------------------------------------------------------- #


class TestOpaque:
    def test_the_viewer_hides_the_page_under_it_on_forges_own_variables(self, browser):
        """"please make its background opaque, right now it's transparent and I
        see what's behind it when it's open": painted from the page's ground,
        not from Forge's transparent block fill."""
        page, served = open_page(browser)
        try:
            open_viewer(page)
            found = page.evaluate("""() => {
                const alpha = (node) => {
                    const colour = getComputedStyle(node).backgroundColor;
                    const parts = colour.match(/[\\d.]+/g).map(Number);
                    return colour.startsWith('rgba') ? parts[3] : 1;
                };
                return {block: getComputedStyle(document.body).getPropertyValue('--block-background-fill').trim(),
                        dialog: alpha(document.querySelector('.mc-lv'))};
            }""")
            assert found["block"] == "transparent", "the page carries Forge's rule"
            assert found["dialog"] == 1
        finally:
            close_page(page, served)


class TestAddingOne:
    def test_typing_suggests_names_the_ones_that_start_with_it_first(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            # Listed as the viewer opens, the box having the focus from then
            # (2026-11-06); until then, when the box was first used.
            typed(page, "det")
            page.wait_for_selector(".mc-lv-option")
            assert [o["name"] for o in page.evaluate(SUGGESTED)] == \
                ["detail_slider", "add_detail_xl", "soft_light_detail"]
            typed(page, "det sl")
            assert [o["name"] for o in page.evaluate(SUGGESTED)] == ["detail_slider"], "every word"
            typed(page, "analog_f")
            assert page.evaluate(SUGGESTED) == [{"name": "Analog Film", "tag": "analog_film",
                                                 "in": False, "selected": False}], "by its tag's name too"
            assert served.listed == 1, "listed once a viewer"
        finally:
            close_page(page, served)

    def test_one_chosen_is_a_new_row_at_its_preferred_weight_written_at_the_prompts_end(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            typed(page, "add")
            page.wait_for_selector(".mc-lv-option")
            page.click(".mc-lv-option")
            page.wait_for_function("document.querySelectorAll('.mc-lv-row').length === 6")
            page.wait_for_function("!document.querySelector('.mc-lv .mc-lv-skeleton')")
            last = page.evaluate(ROWS)[-1]
            assert (last["name"], last["box"], last["weight"]) == ("add_detail_xl", "New in Prompt", "0.6")
            assert last["picture"].endswith("/pictures/add.png")
            assert served.asked[-1] == ["add_detail_xl"], "its details alone"
            assert page.input_value(".mc-lv-add-input") == "", "empty for the next one"
            assert page.locator(".mc-lv-count").text_content() == "6 in this prompt"
            assert page.locator(".mc-lv-button-apply").text_content() == "Apply 1 change"
            page.click(".mc-lv-button-apply")
            boxes = page.evaluate(BOXES)
            assert boxes == {"prompt": PROMPT + " <lora:add_detail_xl:0.6>",
                             "positive": LITERAL_POSITIVE, "negative": LITERAL_NEGATIVE}
            # The add box's own typing is the viewer's; the page's boxes hear one.
            assert [one for one in page.evaluate("window.inputs") if one != "mc-lora-viewer"] == \
                ["txt2img_prompt"]
        finally:
            close_page(page, served)

    def test_without_a_preferred_weight_forges_default_and_separator(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("""() => { window.opts = {extra_networks_default_multiplier: 0.8,
                                                    extra_networks_add_text_separator: ", "}; }""")
            open_viewer(page, "img2img")
            typed(page, "zoom")
            page.wait_for_selector(".mc-lv-option")
            page.press(".mc-lv-add-input", "Enter")
            page.wait_for_function("document.querySelectorAll('.mc-lv-row').length === 2")
            assert weight_of(page, 1) == "0.8"
            row(page, 1).locator(".mc-lv-step").nth(1).click()        # 0.8 -> 0.85
            page.click(".mc-lv-button-apply")
            assert page.evaluate("document.querySelector('#img2img_prompt textarea').value") == \
                IMG2IMG_PROMPT + ", <lora:zoom_blur:0.85>"
        finally:
            close_page(page, served)

    def test_one_already_in_the_prompt_is_shown_not_added_twice(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            typed(page, "realfilter")
            page.wait_for_selector(".mc-lv-option")
            assert page.evaluate(SUGGESTED) == [{"name": "realfilter_v2_final", "tag": "realfilter_v2",
                                                 "in": True, "selected": False}]
            page.click(".mc-lv-option")
            assert page.locator(".mc-lv-row").count() == 5
            assert row(page, 0).get_attribute("data-flash") is not None
            assert page.locator(".mc-lv-button-apply").text_content() == "Apply"
        finally:
            close_page(page, served)

    def test_the_keys_choose_and_escape_puts_the_list_away_before_the_viewer(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            typed(page, "det")
            page.wait_for_selector(".mc-lv-option")
            page.press(".mc-lv-add-input", "ArrowDown")
            page.press(".mc-lv-add-input", "ArrowDown")
            assert [o["selected"] for o in page.evaluate(SUGGESTED)] == [False, True, False]
            assert page.get_attribute(".mc-lv-add-input", "aria-activedescendant") == "mc-lv-option-1"
            page.press(".mc-lv-add-input", "ArrowUp")
            page.press(".mc-lv-add-input", "ArrowUp")
            assert [o["selected"] for o in page.evaluate(SUGGESTED)] == [False, False, True], "round"
            page.press(".mc-lv-add-input", "Escape")
            assert page.locator(".mc-lv-suggest").is_hidden() and page.locator(".mc-lv").is_visible()
            assert page.input_value(".mc-lv-add-input") == "det"
            page.press(".mc-lv-add-input", "Escape")
            assert page.input_value(".mc-lv-add-input") == "" and page.locator(".mc-lv").is_visible()
            page.press(".mc-lv-add-input", "Escape")
            assert page.locator(".mc-lv").is_hidden()
            assert page.evaluate("window.forgeEscapes") == 0
        finally:
            close_page(page, served)

    def test_nothing_matching_says_so_naming_the_preset(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            typed(page, "qqq")
            page.wait_for_selector(".mc-lv-suggest-note")
            assert page.locator(".mc-lv-suggest-note").text_content() == \
                "No LoRA matches \u201cqqq\u201d in the xl preset's folders."
            page.press(".mc-lv-add-input", "Enter")
            assert page.locator(".mc-lv-row").count() == 5
        finally:
            close_page(page, served)

    def test_a_new_row_can_be_taken_back_and_reset_all_takes_it_out(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            typed(page, "zoom")
            page.wait_for_selector(".mc-lv-option")
            page.press(".mc-lv-add-input", "Enter")
            page.wait_for_function("document.querySelectorAll('.mc-lv-row').length === 6")
            row(page, 5).locator(".mc-lv-remove").click()
            assert page.locator(".mc-lv-button-apply").text_content() == "Apply"
            row(page, 5).locator(".mc-lv-remove").click()
            assert page.locator(".mc-lv-button-apply").text_content() == "Apply 1 change"
            page.click(".mc-lv-reset-all")
            assert page.locator(".mc-lv-row").count() == 5
            page.click(".mc-lv-button-apply")
            assert page.evaluate(BOXES)["prompt"] == PROMPT
        finally:
            close_page(page, served)

    def test_the_names_request_has_a_deadline(self, browser):
        page, served = open_page(browser, clock=True)
        served.hold_names = True
        try:
            open_viewer(page, wait_for_details=False)
            typed(page, "det")
            assert page.locator(".mc-lv-suggest-note").text_content() == "Listing your LoRAs\u2026"
            page.clock.run_for(9000)
            page.wait_for_function("document.querySelector('.mc-lv-suggest-note') && "
                                   "document.querySelector('.mc-lv-suggest-note').textContent.includes('in time')")
            served.hold_names = False
            typed(page, "deta")
            page.wait_for_selector(".mc-lv-option")
            assert served.listed == 2, "typed again, listed again"
        finally:
            close_page(page, served)

    def test_on_a_phone_the_list_is_inside_the_viewer_and_the_window(self, browser):
        page, served = open_page(browser, width=390, height=640, touch=True)
        try:
            open_viewer(page)
            page.tap(".mc-lv-add-input")
            page.keyboard.type("d")
            page.wait_for_selector(".mc-lv-option")
            found = page.evaluate("""() => {
                const box = (n) => n.getBoundingClientRect();
                const dialog = box(document.querySelector('.mc-lv'));
                const list = box(document.querySelector('.mc-lv-suggest'));
                const field = box(document.querySelector('.mc-lv-add-field'));
                const options = [...document.querySelectorAll('.mc-lv-option')].map(box);
                return {dialog: {top: dialog.top, bottom: dialog.bottom, left: dialog.left, right: dialog.right},
                        list: {top: list.top, bottom: list.bottom, left: list.left, right: list.right},
                        field: field.height, option: options[0].height, count: options.length,
                        width: innerWidth, height: innerHeight,
                        wide: document.documentElement.scrollWidth};
            }""")
            assert found["list"]["top"] >= found["dialog"]["top"] and found["list"]["bottom"] <= found["dialog"]["bottom"]
            assert found["list"]["left"] >= found["dialog"]["left"] and found["list"]["right"] <= found["dialog"]["right"]
            assert found["dialog"]["bottom"] <= found["height"] and found["wide"] <= found["width"]
            assert found["count"] == 3 and found["field"] <= 40 and found["option"] >= 36
            page.tap(".mc-lv-option >> nth=2")
            page.wait_for_function("document.querySelectorAll('.mc-lv-row').length === 6")
            assert page.evaluate(ROWS)[-1]["name"] == "soft_light_detail"
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# A LoRA's description, and the suggestions' pictures
# --------------------------------------------------------------------------- #

POPUP = """() => {
    const popup = document.querySelector('.global-popup');
    const editor = document.getElementById('txt2img_lora_edit_user_metadata');
    return {shown: !!popup && popup.style.display !== 'none',
            holds: !!popup && popup.contains(editor),
            name: document.querySelector('#txt2img_lora_edit_user_metadata_name textarea').value,
            viewer: !document.querySelector('.mc-lv').hidden, scrim: !document.querySelector('.mc-lv-scrim').hidden,
            edited: window.edited.slice(), focused: document.activeElement && document.activeElement.className};
}"""


class TestTheDescription:
    """"I want the ability to get to the extra network content description by
    our new lora menu": a row's picture opens Forge's own description editor
    for its card, and the viewer steps aside until Forge puts it away."""

    def test_a_rows_picture_opens_its_cards_description_by_the_cards_own_name(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()          # 0.85 -> 0.9, kept through it
            thumb = row(page, 0).locator(".mc-lv-thumb")
            marks = {"disabled": thumb.is_disabled(), "label": thumb.get_attribute("aria-label"),
                     "badge": thumb.locator(".mc-lv-thumb-badge svg").count()}
            thumb.click()
            opened = page.evaluate(POPUP)
            asked = len(served.asked)
            page.keyboard.press("Escape")
            page.wait_for_selector(".mc-lv:not([hidden])")
            page.wait_for_function("!document.querySelector('.mc-lv .mc-lv-skeleton')")
            back = page.evaluate(POPUP)
            again = served.asked[asked:]
            weight = weight_of(page, 0)
        finally:
            close_page(page, served)

        assert marks == {"disabled": False, "label": "Description of realfilter_v2", "badge": 1}
        assert opened["shown"] and opened["holds"], "Forge's editor on Forge's popup"
        assert opened["name"] == "realfilter_v2_final" and opened["edited"] == ["realfilter_v2_final"], \
            "opened by the card's own name, the file's, where the tag carries its alias"
        assert opened["viewer"] is False and opened["scrim"] is False, "stepped aside, under nothing"
        assert back["shown"] is False and back["viewer"] is True, "Escape was Forge's, and the viewer came back"
        assert again == [["realfilter_v2"]], "its details asked again"
        assert weight == "0.9", "the change made before is still there"
        assert back["focused"] and "mc-lv-thumb" in back["focused"]

    @pytest.mark.parametrize("way", [".global-popup .save", ".global-popup .cancel", ".global-popup-close"])
    def test_save_cancel_and_the_backdrop_bring_it_back_too(self, browser, way):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 1).locator(".mc-lv-thumb").click()
            assert page.evaluate(POPUP)["shown"] is True
            page.click(way, position={"x": 5, "y": 5} if way == ".global-popup-close" else None)
            page.wait_for_selector(".mc-lv:not([hidden])")
            found = page.evaluate(POPUP)
        finally:
            close_page(page, served)

        assert found["shown"] is False and found["viewer"] is True

    def test_escape_while_it_is_open_is_forges_not_the_viewers(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-thumb").click()
            page.keyboard.press("Escape")
            page.wait_for_selector(".mc-lv:not([hidden])")
            open_after = page.evaluate("window.mcLoraViewer.isOpen()")
        finally:
            close_page(page, served)

        assert open_after is True, "the first Escape closed Forge's editor, not the viewer"

    def test_a_lora_not_on_disk_or_a_page_without_the_editor_has_no_description(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            missing = row(page, 2).locator(".mc-lv-thumb")
            gone = {"disabled": missing.is_disabled(), "badge": missing.locator(".mc-lv-thumb-badge").count()}
            page.click(".mc-lv-button-cancel")
            page.evaluate("document.getElementById('txt2img_lora_edit_user_metadata').remove()")
            open_viewer(page)
            plain = row(page, 0).locator(".mc-lv-thumb")
            none = {"disabled": plain.is_disabled(), "badge": plain.locator(".mc-lv-thumb-badge").count()}
        finally:
            close_page(page, served)

        assert gone == {"disabled": True, "badge": 0}
        assert none == {"disabled": True, "badge": 0}

    def test_a_row_whose_card_is_not_known_has_no_description(self, browser):
        """An answer that leaves a name out gives its row no card to open."""
        page, served = open_page(browser)
        served.omit = {"contrast_slider"}
        try:
            open_viewer(page)
            thumb = row(page, 1).locator(".mc-lv-thumb")
            found = {"disabled": thumb.is_disabled(), "badge": thumb.locator(".mc-lv-thumb-badge").count()}
        finally:
            close_page(page, served)

        assert found == {"disabled": True, "badge": 0}

    def test_a_change_of_tab_while_it_is_open_cancels_the_viewer(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            row(page, 0).locator(".mc-lv-thumb").click()
            page.evaluate("""() => { document.getElementById('tab_txt2img').style.display = 'none';
                                     document.getElementById('tab_img2img').style.display = 'block'; }""")
            page.wait_for_function("!window.mcLoraViewer.isOpen()")
            page.evaluate("closePopup()")
            page.wait_for_timeout(100)
            found = page.evaluate(POPUP)
            prompt = page.evaluate(BOXES)["prompt"]
        finally:
            close_page(page, served)

        assert found["viewer"] is False, "Forge putting its editor away later brings back no viewer"
        assert prompt == PROMPT, "nothing applied"

    def test_a_themes_button_rule_does_not_widen_the_picture(self, browser):
        page, served = open_page(browser, width=1200, theme_css="button { min-width: fit-content !important; }")
        try:
            open_viewer(page)
            box = row(page, 0).locator(".mc-lv-thumb").bounding_box()
        finally:
            close_page(page, served)

        assert box["width"] == 44 and box["height"] == 44


SUGGESTED_PICTURES = """() => [...document.querySelectorAll('.mc-lv-suggest .mc-lv-option')].map((o) => {
    const thumb = o.querySelector('.mc-lv-option-thumb');
    const img = thumb && thumb.querySelector('img');
    return {name: o.querySelector('.mc-lv-option-name').textContent, first: o.firstElementChild === thumb,
            picture: img ? img.getAttribute('src') : '',
            initials: (thumb && thumb.querySelector('.mc-lv-initials') || {}).textContent || '',
            waiting: !!(thumb && thumb.querySelector('.mc-lv-skeleton')),
            box: thumb ? (({width, height}) => ({width, height}))(thumb.getBoundingClientRect()) : null};
})"""


class TestTheSuggestionsPictures:
    """"Please show the images of the lora as well in the predictive typing
    area. I need to see the image next to the lora name in that list.\""""

    def test_each_suggestion_shows_its_loras_picture_beside_its_name(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            before = len(served.asked)
            typed(page, "det")
            page.wait_for_selector(".mc-lv-option")
            first = page.evaluate(SUGGESTED_PICTURES)
            page.wait_for_function("""() => ![...document.querySelectorAll('.mc-lv-option-thumb')]
                .some((t) => t.querySelector('.mc-lv-skeleton'))""")
            drawn = page.evaluate(SUGGESTED_PICTURES)
            asked = served.asked[before:]
            typed(page, "deta")
            page.wait_for_timeout(400)
            again = served.asked[before:]
        finally:
            close_page(page, served)

        assert [one["waiting"] for one in first] == [True, True, True], "a shimmer while asked for"
        assert all(one["first"] for one in drawn), "the picture before the name"
        pictures = {one["name"]: (one["picture"], one["initials"]) for one in drawn}
        assert pictures["add_detail_xl"] == ("/pictures/add.png", "")
        assert pictures["detail_slider"] == ("", "DS"), "its initials without one"
        assert asked == [["detail_slider", "add_detail_xl", "soft_light_detail"]], "one request for the names shown"
        assert again == asked, "a name seen once is not asked again"
        assert all(one["box"] == {"width": 32, "height": 32} for one in drawn)

    def test_a_lora_added_from_the_list_needs_no_request_of_its_own(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            typed(page, "add")
            page.wait_for_selector(".mc-lv-option img")
            asked = len(served.asked)
            page.press(".mc-lv-add-input", "Enter")
            page.wait_for_function("document.querySelectorAll('.mc-lv-row').length === 6")
            last = page.evaluate(ROWS)[-1]
            after = len(served.asked)
            thumb = row(page, 5).locator(".mc-lv-thumb")
            describable = not thumb.is_disabled()
        finally:
            close_page(page, served)

        assert after == asked
        assert last["picture"].endswith("/pictures/add.png") and last["weight"] == "0.6"
        assert describable, "an added LoRA's description is a press away too"

    def test_the_pictures_request_has_a_deadline_and_is_not_repeated(self, browser):
        page, served = open_page(browser, clock=True)
        try:
            open_viewer(page, wait_for_details=False)
            served.hold = True
            typed(page, "det")
            page.clock.run_for(200)
            page.clock.run_for(9000)
            page.wait_for_function("""() => document.querySelectorAll('.mc-lv-option-thumb .mc-lv-initials').length === 3""")
            held = len(served.held)
            served.hold = False
            typed(page, "deta")
            page.clock.run_for(400)
            page.wait_for_timeout(200)
            found = page.evaluate(SUGGESTED_PICTURES)
            asked = [names for names in served.asked if "add_detail_xl" in names]
        finally:
            close_page(page, served)

        assert held >= 1
        assert [one["initials"] for one in found] == ["DS", "AD", "SL"]
        assert len(asked) == 1, "a name that did not arrive in time is not asked again in this viewer"
