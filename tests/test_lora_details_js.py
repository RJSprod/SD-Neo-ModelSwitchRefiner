"""The LoRA dialog's script (javascript/model_chain_lora_details.js), run for
real in Chromium on the fragments mc_lora_render writes.

What the page does without a round trip to Python: the trigger-word chips
edit Forge's activation text and light up to match it, Copy puts a section's
exact text on the clipboard, Append and Replace write a prompt into the
generation's own boxes, a card's view and download reach Python through the
hidden textbox-and-button pair, a long text folds and unfolds, and a clamped
text that turns out to fit loses its fade. Clicks on the same attributes
outside the dialog are nobody's business.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

import mc_lora_render as render

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_lora_details.js"
ORIGIN = "https://lora-js.test"  # a secure context, which the clipboard needs
ACTIVATION = "txt2img_lora_edit_user_metadata_activation_text"
BRIDGE = "txt2img_lora_edit_user_metadata_mc_lora_pick"

STYLE = ROOT / "style.css"

HOST_CSS = """
body { margin: 0; font: 14px sans-serif; background: #111; color: #eee; }
textarea { width: 30em; height: 4em; }
"""


def dialog_fragments() -> str:
    long_text = " ".join(f"sentence {index} of the version notes." for index in range(400))
    return "".join([
        render.triggers_html(["rim light", "backlit"], target=ACTIVATION, active=set()),
        render.text_section_html("Version notes", long_text),
        '<section class="mc-lora-section mc-lora-text-section" id="short">'
        '<div class="mc-lora-text mc-lora-clamp">short enough</div>'
        '<button type="button" data-mc-lora-action="expand">Show more</button></section>',
        render.negative_html(["flat lighting, (blurry)"], tab="txt2img"),
        render.prompt_cards_html(
            [
                {"index": 0, "prompt": "first (one)", "thumb": "", "saved": False, "video": False, "url": "https://x/1.jpeg"},
                {"index": 1, "prompt": "second", "thumb": "data:image/webp;base64,AAAA", "saved": True, "video": False, "url": "https://x/2.jpeg"},
            ],
            total=2, tab="txt2img", bridge=BRIDGE,
        ),
    ])


def page_html() -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{HOST_CSS}</style>
<link rel="stylesheet" href="/style.css"></head><body>
<script>
function gradioApp() {{ return document; }}
var opts = {{extra_networks_add_text_separator: ", "}};
window.after = []; function onAfterUiUpdate(fn) {{ window.after.push(fn); }}
window.popupClosed = 0; function closePopup() {{ window.popupClosed += 1; }}
window.updated = [];
function updateInput(target) {{
    const holder = target.closest("[id]");
    window.updated.push(holder ? holder.id : "");
    target.dispatchEvent(new Event("input", {{ bubbles: true }}));
}}
window.picked = [];
</script>
<div id="txt2img_prompt"><label><textarea></textarea></label></div>
<div id="txt2img_neg_prompt"><label><textarea></textarea></label></div>
<button id="outside" type="button" data-mc-lora-action="copy">not ours</button>
<div class="global-popup"><div class="global-popup-inner">
<div class="edit-user-metadata mc-lora-dialog">
  <div id="{ACTIVATION}"><label><textarea></textarea></label></div>
  <div class="mc-lora-side-html"><div class="prose">{dialog_fragments()}</div></div>
  <div class="mc-lora-picker">picker</div>
  <div id="{BRIDGE}"><label><textarea></textarea></label></div>
  <button id="{BRIDGE}_button" type="button"
    onclick="window.picked.push(document.querySelector('#{BRIDGE} textarea').value)">pick</button>
</div></div></div>
<script src="/javascript/model_chain_lora_details.js"></script>
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


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 1200, "height": 900})
    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=ORIGIN)
    page = context.new_page()
    html = page_html()
    script = SCRIPT.read_text(encoding="utf-8")
    style = STYLE.read_text(encoding="utf-8")

    def serve(route):
        url = route.request.url
        if url.endswith("/javascript/model_chain_lora_details.js"):
            route.fulfill(status=200, content_type="text/javascript", body=script)
        elif url.endswith("/style.css"):
            route.fulfill(status=200, content_type="text/css", body=style)
        elif url.rstrip("/") == ORIGIN:
            route.fulfill(status=200, content_type="text/html", body=html)
        else:
            route.fulfill(status=404, body="")

    page.route(ORIGIN + "/**", serve)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(ORIGIN + "/")
    page.wait_for_function("typeof mcLoraHandleAction === 'function'")
    page.errors = errors
    yield page
    assert not errors, errors
    context.close()


def activation(page) -> str:
    return page.eval_on_selector(f"#{ACTIVATION} textarea", "el => el.value")


def prompt(page, negative=False) -> str:
    return page.eval_on_selector(f"#txt2img_{'neg_' if negative else ''}prompt textarea", "el => el.value")


def lit(page) -> list:
    return page.eval_on_selector_all(".mc-lora-word.mc-lora-word-on", "els => els.map(e => e.dataset.word)")


def clipboard(page) -> str:
    return page.evaluate("navigator.clipboard.readText()")


class TestTriggerWords:
    def test_a_chip_toggles_its_word_in_the_activation_text(self, page):
        page.click('.mc-lora-word[data-word="rim light"]')
        assert activation(page) == "rim light" and lit(page) == ["rim light"]
        page.click('.mc-lora-word[data-word="backlit"]')
        assert activation(page) == "rim light, backlit" and lit(page) == ["rim light", "backlit"]
        page.click('.mc-lora-word[data-word="rim light"]')
        assert activation(page) == "backlit" and lit(page) == ["backlit"]
        assert page.evaluate("window.updated") == [ACTIVATION] * 3, "Gradio is told of every write"

    def test_use_and_append_and_typing_keep_the_chips_honest(self, page):
        page.click('.mc-lora-triggers [data-mc-lora-action="use-triggers"]')
        assert activation(page) == "rim light, backlit"
        page.fill(f"#{ACTIVATION} textarea", "backlit, my own word")
        assert lit(page) == ["backlit"], "typing re-syncs the chips"
        page.click('.mc-lora-triggers [data-mc-lora-action="append-triggers"]')
        assert activation(page) == "backlit, my own word, rim light", "only the missing word is added"

    def test_copy_all_puts_the_words_on_the_clipboard(self, page):
        page.click('.mc-lora-triggers [data-mc-lora-action="copy"]')
        page.wait_for_function("navigator.clipboard.readText().then(t => t === 'rim light, backlit')")
        assert clipboard(page) == "rim light, backlit"
        assert page.inner_text('.mc-lora-triggers [data-mc-lora-action="copy"]') == "Copied"


class TestTexts:
    def test_a_long_text_folds_and_unfolds(self, page):
        page.evaluate("window.after.forEach(fn => fn())")  # what Gradio runs after an update
        section = page.locator(".mc-lora-text-section").first
        assert section.locator(".mc-lora-clamp").count() == 1, "long: the fade stays"
        text = section.locator(".mc-lora-clamp")
        folded = text.evaluate("el => el.getBoundingClientRect().height")
        assert text.evaluate("el => el.scrollHeight") > folded * 3, "the fade hides most of it"
        section.locator('[data-mc-lora-action="expand"]').click()
        assert "mc-lora-expanded" in section.get_attribute("class")
        assert section.locator('[data-mc-lora-action="expand"]').inner_text() == "Show less"
        assert text.evaluate("el => el.getBoundingClientRect().height") > folded * 3
        section.locator('[data-mc-lora-action="expand"]').click()
        assert "mc-lora-expanded" not in section.get_attribute("class")

    def test_a_clamped_text_that_fits_loses_its_fade_and_its_toggle(self, page):
        assert page.locator("#short .mc-lora-clamp").count() == 1
        page.evaluate("window.after.forEach(fn => fn())")
        assert page.locator("#short .mc-lora-clamp").count() == 0
        assert page.locator('#short [data-mc-lora-action="expand"]').count() == 0

    def test_copy_copies_the_whole_text_and_the_negative_goes_to_its_box(self, page):
        page.locator(".mc-lora-text-section").first.locator('[data-mc-lora-action="copy"]').click()
        page.wait_for_function("navigator.clipboard.readText().then(t => t.startsWith('sentence 0'))")
        assert clipboard(page).endswith("sentence 399 of the version notes.")
        page.click('[data-mc-lora-action="append-negative"]')
        assert prompt(page, negative=True) == "flat lighting, (blurry)"
        page.click('[data-mc-lora-action="append-negative"]')
        assert prompt(page, negative=True) == "flat lighting, (blurry), flat lighting, (blurry)", "appended with the separator"


class TestTheCards:
    def test_view_and_download_reach_python_through_the_bridge(self, page):
        cards = page.locator(".mc-lora-card")
        assert cards.count() == 2
        cards.nth(1).locator('[data-mc-lora-action="view"]').first.click()
        assert page.evaluate("window.picked") == ["view:1"]
        download = cards.nth(0).locator('[data-mc-lora-action="download"]')
        download.click()
        assert page.evaluate("window.picked") == ["view:1", "download:0"]
        assert download.is_disabled() and download.inner_text() == "Downloading…"

    def test_append_and_replace_write_the_exact_prompt(self, page):
        cards = page.locator(".mc-lora-card")
        cards.nth(0).locator('[data-mc-lora-action="append-prompt"]').click()
        assert prompt(page) == "first (one)"
        cards.nth(1).locator('[data-mc-lora-action="append-prompt"]').click()
        assert prompt(page) == "first (one), second"
        cards.nth(0).locator('[data-mc-lora-action="replace-prompt"]').click()
        assert prompt(page) == "first (one)"
        assert page.evaluate("window.popupClosed") == 1, "a replaced prompt is what the dialog was opened for"
        assert page.evaluate("window.updated").count("txt2img_prompt") == 3

    def test_copy_copies_the_prompt_with_its_parentheses(self, page):
        page.locator(".mc-lora-card").nth(0).locator('[data-mc-lora-action="copy"]').click()
        page.wait_for_function("navigator.clipboard.readText().then(t => t === 'first (one)')")


class TestWhatIsNotOurs:
    def test_a_button_outside_the_dialog_is_left_alone(self, page):
        page.evaluate("navigator.clipboard.writeText('untouched')")
        page.click("#outside")
        assert clipboard(page) == "untouched"
        assert page.inner_text("#outside") == "not ours"
