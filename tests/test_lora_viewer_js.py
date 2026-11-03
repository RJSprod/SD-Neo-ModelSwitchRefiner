"""The LoRA viewer (javascript/model_chain_lora_viewer.js), run for real in Chromium.

The page is Forge's prompt area in the shape the script reads it: the tool row
with Forge's paste, clear and Apply Styles buttons, the prompt and the two
Literal boxes on Txt2Img, the prompt on Img2Img, and the tab bar above. The
route is the test's, answering as each test plans and keeping what was asked.
The viewer's own stylesheet section is loaded from ``style.css``.

What is held here is what the user would see go wrong: a slider that moves
when a finger only scrolled, an Apply that touches a character it should not,
an Escape that also interrupts a generation, a viewer that never stops waiting
for its details, a stray tap or a change of tab that applies, a theme rule that
stretches a trigger out of its row.
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

HOST_CSS = """
:root { --block-background-fill: #0d0d0d; --background-fill-primary: #000; --border-color-primary: #262626;
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
.row { display: flex; gap: 8px; align-items: center; margin: 8px 0; }
.gradio-button.tool { width: 40px; height: 40px; display: grid; place-items: center; }
.spacer { height: 1600px; }
"""


def page_html(theme_css: str = "") -> str:
    tools = """
        <button id="{p}paste" class="lg secondary gradio-button tool svelte-cmf5ev">&#x2199;&#xfe0f;</button>
        <button id="{tab}_clear_prompt" class="lg secondary gradio-button tool svelte-cmf5ev">&#x1f5d1;&#xfe0f;</button>
        <button id="{tab}_style_apply" class="lg secondary gradio-button tool svelte-cmf5ev">&#x1f4cb;</button>
        <button id="{tab}_restore_progress" class="lg secondary gradio-button tool svelte-cmf5ev" style="display: none">&#x1f300;</button>
        <span class="green-dot"></span>"""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{HOST_CSS}{theme_css}</style>
<link rel="stylesheet" href="/style.css">
</head><body class="dark">
<script>
function gradioApp() {{ return document.querySelector("gradio-app") || document; }}
const hooks = {{loaded: [], after: []}};
function onUiLoaded(fn) {{ hooks.loaded.push(fn); }}
function onAfterUiUpdate(fn) {{ hooks.after.push(fn); }}
function updateInput(target) {{
    const event = new Event("input", {{bubbles: true}});
    Object.defineProperty(event, "target", {{value: target}});
    target.dispatchEvent(event);
}}
window.inputs = []; window.blurs = []; window.overlays = []; window.forgeEscapes = 0; window.copied = [];
document.addEventListener("input", (e) => window.inputs.push(e.target.closest("[id]").id), true);
document.addEventListener("blur", (e) => {{
    if (e.target && e.target.closest && e.target.closest(".block")) window.blurs.push(e.target.closest("[id]").id);
}}, true);
// Forge's own Escape listener (script.js): it interrupts a running generation.
document.addEventListener("keydown", (e) => {{ if (e.key === "Escape") window.forgeEscapes += 1; }});
document.addEventListener("minipaint:overlay", (e) => window.overlays.push(e.detail));
</script>
<gradio-app>
<div id="tabs" class="tabs">
  <div class="tab-nav"><button id="nav-txt2img">Txt2img</button><button id="nav-img2img">Img2img</button></div>
  <div id="tab_txt2img" class="tabitem">
    <div id="txt2img_prompt" class="block"><textarea>{PROMPT}</textarea></div>
    <div id="mc-krea-creative-literal-positive" class="block"><textarea>{LITERAL_POSITIVE}</textarea></div>
    <div id="mc-krea-creative-literal-negative" class="block"><textarea>{LITERAL_NEGATIVE}</textarea></div>
    <div id="txt2img_tools" class="row">{tools.format(p="", tab="txt2img")}</div>
    <div class="spacer"></div>
  </div>
  <div id="tab_img2img" class="tabitem" style="display: none;">
    <div id="img2img_prompt" class="block"><textarea>{IMG2IMG_PROMPT}</textarea></div>
    <div id="img2img_tools" class="row">{tools.format(p="i2i_", tab="img2img")}</div>
  </div>
</div>
</gradio-app>
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


class Served:
    """The page, the stylesheet, the script, the pictures and the route."""

    def __init__(self, html: str):
        self.html = html
        self.asked: list[list[str]] = []
        self.hold = False
        self.held = []
        self.status = 200

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
            body = {"loras": [dict(DETAILS.get(name, {"found": True, "picture": "", "title": "",
                                                       "triggers": []}), name=name)
                              for name in names]}
            route.fulfill(status=self.status, content_type="application/json", body=json.dumps(body))
        else:
            route.fulfill(status=404, body="")


def open_page(browser, width=900, height=800, theme_css="", clock=False, touch=False):
    context = browser.new_context(viewport={"width": width, "height": height}, has_touch=touch)
    page = context.new_page()
    if clock:
        page.clock.install()
    served = Served(page_html(theme_css))
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


# --------------------------------------------------------------------------- #
# The button
# --------------------------------------------------------------------------- #


class TestTheButton:
    def test_one_button_after_apply_styles_on_each_tab_with_its_neighbours_classes(self, browser):
        page, served = open_page(browser)
        try:
            found = page.evaluate("""() => ['txt2img', 'img2img'].map((tab) => {
                const button = document.getElementById('mc-lora-viewer-' + tab);
                return {after: button.previousElementSibling.id, cls: button.className,
                        count: document.querySelectorAll('#' + tab + '_tools .mc-lv-open').length,
                        label: button.getAttribute('aria-label'), svg: !!button.querySelector('svg'),
                        text: button.textContent.trim()};
            })""")
            assert found == [
                {"after": "txt2img_style_apply", "cls": "lg secondary gradio-button tool svelte-cmf5ev mc-lv-open",
                 "count": 1, "label": "LoRAs in this prompt", "svg": True, "text": ""},
                {"after": "img2img_style_apply", "cls": "lg secondary gradio-button tool svelte-cmf5ev mc-lv-open",
                 "count": 1, "label": "LoRAs in this prompt", "svg": True, "text": ""},
            ]
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

    def test_a_tap_outside_does_nothing(self, browser):
        page, served = open_page(browser)
        try:
            open_viewer(page)
            row(page, 0).locator(".mc-lv-step").nth(1).click()
            page.mouse.click(4, 700)
            assert page.locator(".mc-lv").is_visible()
            assert weight_of(page, 0) == "0.9"
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
