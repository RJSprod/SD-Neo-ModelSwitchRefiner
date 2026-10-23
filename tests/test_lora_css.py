"""The LoRA Details section of style.css, rendered for real in Chromium on a
page in Forge's and Gradio 4.40's shape.

Two of the user's three asks on 2026-10-23 are the stylesheet's: the dialog
"full page (full width) ... instead of the current narrow strip", and "show /
hide folders doesn't work in the extra networks drawer". The first failed
because the theme holds Forge's popup to min(56em, 100vw - 48px) with a rule
more specific than any widening the standalone extension wrote; the second
because the theme draws the folder row as a flex box with a rule more
specific than Forge's hiding one. Both are measured here, with the theme on
and off, against Forge's own popup rules and Gradio's row and block rules.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STYLE = ROOT / "style.css"
SECTION_BANNER = "Model Chain -- LoRA Details"

# Forge's style.css, the rules that size the popup and the folder row.
FORGE_CSS = """
.global-popup { display: flex; position: fixed; z-index: 1001; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; }
.global-popup * { box-sizing: border-box; }
.global-popup-close { position: fixed; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(20, 20, 20, 0.95); }
.global-popup-inner { display: inline-block; margin: auto; padding: 2em; z-index: 1001; max-height: 90%; max-width: 90%; }
.edit-user-metadata { width: 56em; background: #111 !important; padding: 2em !important; }
.edit-user-metadata .extra-network-name { font-size: 18pt; }
.extra-network-dirs-hidden .extra-network-dirs { display: none; }
"""

# Gradio 4.40: every block is as wide as its parent, a row is a wrapping flex
# box whose children share it equally, a group stacks what it holds.
GRADIO_CSS = """
*, ::before, ::after { box-sizing: border-box; }
html { font-size: 16px; }
body { margin: 0; font: 14px sans-serif; background: #000; color: #eee; }
.block { position: relative; width: 100%; }
.gradio-row { display: flex; flex-wrap: wrap; gap: 8px; width: 100%; }
.gradio-row > * { flex: 1 1 0%; min-width: min(160px, 100%); }
.gradio-column { display: flex; flex-direction: column; gap: 8px; }
.gr-group, .styler { display: flex; flex-direction: column; gap: 0; }
.hidden { display: none !important; }
"""


def column(classes: str, scale: int, min_width: int, body: str, hidden: bool = False) -> str:
    return (
        f'<div class="gradio-column mc-lora-col {classes}{" hidden" if hidden else ""}" '
        f'style="min-width: min({min_width}px, 100%); flex-grow: {scale};">{body}</div>'
    )


def dialog_html(*, widened: bool, gallery: bool) -> str:
    """The popup as Forge opens it with the box mc_lora_details laid out."""
    box_classes = "gr-group gradio-group edit-user-metadata" + (" mc-lora-dialog" if widened else "")
    return f"""
<div class="global-popup"><div class="global-popup-close"></div><div class="global-popup-inner">
  <div id="txt2img_lora_edit_user_metadata" class="{box_classes}"><div class="styler">
    <div class="gradio-row mc-lora-head-row unequal-height">
      <div class="block gradio-html extra-network-name"><div class="prose">mossLightingSlider_v1</div></div>
      <div class="block gradio-html mc-lora-header"><div class="prose">
        <div class="mc-lora-head"><div class="mc-lora-head-text"><div class="mc-lora-head-title">Moss lighting slider</div>
        <div class="mc-lora-pills"><span class="mc-lora-pill">v1.0</span></div></div>
        <a class="mc-lora-open-link" href="#">Open on Civitai</a></div>
      </div></div>
    </div>
    <div class="gradio-row mc-lora-body unequal-height">
      {column("mc-lora-col-file", 3, 280, '<div class="block gradio-html mc-lora-col-head"><div class="prose"><div class="mc-lora-col-title">File</div></div></div><div class="block gradio-html"><div class="prose"><div class="card standalone-card-preview"></div></div></div>')}
      {column("mc-lora-col-about", 4, 320, '<div class="block gradio-html mc-lora-col-head"><div class="prose"><div class="mc-lora-col-title">Civitai</div></div></div>')}
      {column("mc-lora-col-gallery", 6, 480, '<div class="block gradio-html"><div class="prose">gallery</div></div>', hidden=not gallery)}
    </div>
    <div class="gradio-row edit-user-metadata-buttons"><button>Cancel</button><button>Save</button></div>
  </div></div>
</div></div>
"""


DRAWER_HTML = """
<div class="mc-theme-networks-host mc-theme-networks-open">
  <div class="tabitem mc-theme-networks-page mc-theme-networks-shown">
    <div id="txt2img_lora_pane" class="extra-network-pane extra-network-dirs-hidden">
      <div class="extra-network-pane-content-dirs">
        <div id="txt2img_lora_dirs" class="extra-network-dirs"><button class="search-all">all</button><button>SDXL/</button></div>
        <div id="txt2img_lora_cards" class="extra-network-cards"></div>
      </div>
    </div>
  </div>
</div>
"""


def page_html(body: str, *, theme: bool) -> str:
    root = ' data-mc-theme="on"' if theme else ""
    return f"""<!doctype html><html{root}><head><meta charset="utf-8">
<style>{GRADIO_CSS}</style><style>{FORGE_CSS}</style>
<link rel="stylesheet" href="/style.css">
</head><body>{body}</body></html>"""


sync_api = pytest.importorskip("playwright.sync_api")


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


def serve(html: str):
    css = STYLE.read_text(encoding="utf-8")

    def handler(route):
        path = route.request.url.split("://", 1)[1].split("/", 1)[1] if "/" in route.request.url.split("://", 1)[1] else ""
        if path == "style.css":
            route.fulfill(status=200, content_type="text/css", body=css)
        elif path == "":
            route.fulfill(status=200, content_type="text/html", body=html)
        else:
            route.fulfill(status=404, body="")

    return handler


def open_page(browser, html: str, *, width: int = 1600, height: int = 1000):
    page = browser.new_page(viewport={"width": width, "height": height})
    page.route("http://lora-css.test/**", serve(html))
    page.goto("http://lora-css.test/")
    page.wait_for_load_state("load")
    return page


def lora_section(css: str) -> str:
    """The section from its banner comment to the next banner (the Voice Box's,
    which must stay last for its own test's sake)."""
    start = css.rfind("/*", 0, css.index(SECTION_BANNER))
    end = css.find("/* ==========", start + 2)
    return css[start:] if end < 0 else css[start:end]


def rect(page, selector: str) -> dict:
    return page.eval_on_selector(selector, "el => { const r = el.getBoundingClientRect(); return {left: r.left, top: r.top, width: r.width, height: r.height}; }")


def display(page, selector: str) -> str:
    return page.eval_on_selector(selector, "el => getComputedStyle(el).display")


# ------------------------------------------------------------------ the dialog


class TestTheDialogsWidth:
    @pytest.mark.parametrize("theme", [False, True], ids=["forge", "theme"])
    def test_the_laid_out_dialog_is_the_window_less_the_gutters(self, browser, theme):
        page = open_page(browser, page_html(dialog_html(widened=True, gallery=True), theme=theme))
        try:
            dialog = rect(page, ".edit-user-metadata")
            assert dialog["width"] == pytest.approx(1600 - 48, abs=1), dialog
            assert dialog["left"] == pytest.approx(24, abs=1), dialog
        finally:
            page.close()

    @pytest.mark.parametrize("theme", [False, True], ids=["forge", "theme"])
    def test_a_dialog_the_layout_could_not_build_keeps_forges_width(self, browser, theme):
        """Without the class the layout puts on the box -- the fallback, and
        the checkpoint and embedding editors -- Forge's 56em (the theme's
        min(56em, 100vw - 48px)) stands."""
        page = open_page(browser, page_html(dialog_html(widened=False, gallery=True), theme=theme))
        try:
            em = float(page.eval_on_selector(".edit-user-metadata", "el => parseFloat(getComputedStyle(el).fontSize)"))
            assert rect(page, ".edit-user-metadata")["width"] == pytest.approx(56 * em, abs=1)
        finally:
            page.close()

    @pytest.mark.parametrize("theme", [False, True], ids=["forge", "theme"])
    def test_the_name_and_the_title_strip_share_the_head_row(self, browser, theme):
        """Gradio gives every block the row's width; the name takes what its
        text needs, so the Civitai title sits beside it and not under it."""
        page = open_page(browser, page_html(dialog_html(widened=True, gallery=True), theme=theme))
        try:
            name = rect(page, ".mc-lora-head-row > .extra-network-name")
            header = rect(page, ".mc-lora-head-row > .mc-lora-header")
            row = rect(page, ".mc-lora-head-row")
            assert header["left"] >= name["left"] + name["width"], (name, header)
            assert header["top"] < name["top"] + name["height"], "beside the name, not under it"
            assert name["width"] < row["width"] * 0.4 and header["width"] > row["width"] * 0.5
            # Gradio's `width: 100%` on the block would make the name a line of
            # its own in a wrapping row; with it the two share one line.
            assert name["width"] < 300, name
        finally:
            page.close()

    @pytest.mark.parametrize("theme", [False, True], ids=["forge", "theme"])
    def test_three_columns_side_by_side_that_wrap_when_the_window_is_narrow(self, browser, theme):
        page = open_page(browser, page_html(dialog_html(widened=True, gallery=True), theme=theme))
        try:
            tops = [rect(page, f".mc-lora-col-{name}")["top"] for name in ("file", "about", "gallery")]
            assert max(tops) - min(tops) < 2, tops
            widths = [rect(page, f".mc-lora-col-{name}")["width"] for name in ("file", "about", "gallery")]
            assert widths[0] < widths[1] < widths[2], "3 : 4 : 6"
            assert display(page, ".standalone-card-preview") == "none", "an empty preview card takes no room"
            # 700 px: the file and the Civitai columns still fit beside each
            # other (280 + 320 and the gaps); the gallery, 480 at least, wraps
            page.set_viewport_size({"width": 700, "height": 1000})
            tops = [rect(page, f".mc-lora-col-{name}")["top"] for name in ("file", "about", "gallery")]
            assert tops[0] == tops[1] < tops[2], tops
            # 600 px: one column under the other
            page.set_viewport_size({"width": 600, "height": 1000})
            tops = [rect(page, f".mc-lora-col-{name}")["top"] for name in ("file", "about", "gallery")]
            assert tops[0] < tops[1] < tops[2], tops
        finally:
            page.close()

    def test_without_prompts_the_two_columns_take_the_width(self, browser):
        page = open_page(browser, page_html(dialog_html(widened=True, gallery=False), theme=True))
        try:
            assert display(page, ".mc-lora-col-gallery") == "none"
            file_column, about = rect(page, ".mc-lora-col-file"), rect(page, ".mc-lora-col-about")
            assert about["left"] + about["width"] > 1600 - 48 - 40, (file_column, about)
        finally:
            page.close()


# --------------------------------------------------------------- the drawer


class TestTheFolderRow:
    def test_forges_toggle_hides_the_folder_row_under_the_theme(self, browser):
        page = open_page(browser, page_html(DRAWER_HTML, theme=True))
        try:
            assert display(page, ".extra-network-dirs") == "none", "hidden: the pane carries extra-network-dirs-hidden"
            page.eval_on_selector("#txt2img_lora_pane", "el => el.classList.toggle('extra-network-dirs-hidden', false)")
            assert display(page, ".extra-network-dirs") == "flex", "shown: the theme's row of small buttons"
            page.eval_on_selector("#txt2img_lora_pane", "el => el.classList.toggle('extra-network-dirs-hidden', true)")
            assert display(page, ".extra-network-dirs") == "none"
        finally:
            page.close()

    def test_and_without_the_theme_forges_own_rule_does(self, browser):
        page = open_page(browser, page_html(DRAWER_HTML, theme=False))
        try:
            assert display(page, ".extra-network-dirs") == "none"
        finally:
            page.close()


# --------------------------------------------------------------- the section


class TestTheSection:
    def test_the_stylesheet_carries_the_section_and_nothing_of_the_standalones_names(self):
        css = STYLE.read_text(encoding="utf-8")
        assert SECTION_BANNER in css
        section = lora_section(css)
        assert ".eld-" not in css
        for selector in (
            ".global-popup-inner:has(> .edit-user-metadata.mc-lora-dialog)",
            ':root[data-mc-theme="on"] .edit-user-metadata.mc-lora-dialog',
            ".mc-lora-dialog .mc-lora-head-row",
            ".mc-lora-dialog .mc-lora-col",
            ':root[data-mc-theme="on"] .mc-theme-networks-page .extra-network-dirs-hidden .extra-network-dirs',
        ):
            assert selector in css, selector
        # every fragment class mc_lora_render writes for the dialog has a rule
        for name in ("head-title", "pills", "pill", "facts", "fact-key", "section", "section-title", "words", "word",
                     "clamp", "cards", "card", "card-media", "notice", "table", "job", "bar", "layout", "muted"):
            assert re.search(rf"\.mc-lora-{name}\b", section), name

    def test_the_sections_rules_are_scoped(self):
        """LLM Studio's rule: nothing Gradio generated is styled on its own.
        A rule on a Gradio class or an element names the dialog, the tab or
        the settings block first."""
        css = STYLE.read_text(encoding="utf-8")
        section = re.sub(r"/\*.*?\*/", "", lora_section(css), flags=re.S)
        selectors = re.findall(r"^([^@{}\n][^{}]*?)\s*\{", section, re.M)
        scopes = (".edit-user-metadata", "#mc-lora-details", ".mc-lora-settings-block", ".mc-lora-dialog",
                  ".global-popup-inner:has(> .edit-user-metadata.mc-lora-dialog)")
        for selector_list in selectors:
            for selector in selector_list.split(","):
                selector = selector.strip()
                if not selector:
                    continue
                own = selector.startswith(".mc-lora-") or selector.startswith("details.mc-lora-")
                scoped = any(scope in selector for scope in scopes)
                assert own or scoped, selector
