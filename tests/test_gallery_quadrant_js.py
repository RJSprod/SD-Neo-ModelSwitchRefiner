"""Gallery Quadrant's script (javascript/model_chain_gallery_quadrant.js), run
for real in Chromium on a gallery in Gradio 4.40's shape.

The standalone extension's behaviour, carried in: five zones over the preview
(download, leave the preview, Forge's Save, previous, next), an overlay of
five cells drawn once, Gradio's own icon buttons hidden, nothing taken from a
thumbnail or from Gradio's own controls, nothing at all with the setting off
or the lightbox on. And the one change: the top-left zone's destination, a
folder on this machine, posts the picture's URL to the extension's route and
shows the answer in the zone.

Gradio builds the preview as a button inside a button, which the HTML parser
would never allow, so the page builds its gallery through the DOM. Forge's
`saveImage` and `modalImageSwitch` are recorders here; the browser's download
is read off the anchor the script clicks.
"""

from __future__ import annotations

import glob
import json
import os
import urllib.parse
from pathlib import Path

import pytest

import mc_gallery_quadrant as quadrant

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_gallery_quadrant.js"
STYLE = ROOT / "style.css"
ORIGIN = "http://forge-quadrant.test"
IMAGE = "/file=/data/outputs/txt2img-images/2026-10-17/00001-42.png"

HOST_CSS = """
*, ::before, ::after { box-sizing: border-box; }
body { margin: 0; background: #111; color: #eee; font: 14px sans-serif; }
.gradio-gallery { position: relative; width: 640px; margin: 20px; }
.gradio-gallery > button.preview { display: block; width: 600px; padding: 0; border: 0; background: #222; }
.gradio-gallery .image-button { display: block; width: 600px; height: 400px; padding: 0; border: 0; background: #333; }
.gradio-gallery .image-button img { display: block; width: 600px; height: 400px; }
.gradio-gallery .icon-buttons { position: absolute; top: 4px; right: 4px; z-index: 30; }
.gradio-gallery .thumbnails { display: flex; height: 60px; }
.gradio-gallery .thumbnail-item { width: 60px; height: 60px; }
"""

# The gallery in Gradio 4.40's preview shape: button.preview > (.icon-buttons,
# button.image-button > img, .thumbnails). Built through the DOM, because
# the parser closes a button at the next button.
BUILD_GALLERY = """
function buildGallery(id, preview) {
    const gallery = document.createElement("div");
    gallery.id = id;
    gallery.className = "block gradio-gallery";
    if (preview) {
        const button = document.createElement("button");
        button.className = "preview";
        const icons = document.createElement("div");
        icons.className = "icon-buttons";
        ["Download", "Clear"].forEach((label) => {
            const icon = document.createElement("button");
            icon.setAttribute("aria-label", label);
            icon.textContent = label[0];
            icon.addEventListener("click", () => { window.iconClicks.push(label); });
            icons.appendChild(icon);
        });
        const image = document.createElement("button");
        image.className = "image-button";
        const img = document.createElement("img");
        img.src = %(image)s;
        img.alt = "";
        image.appendChild(img);
        const thumbs = document.createElement("div");
        thumbs.className = "thumbnails";
        [1, 2].forEach((n) => {
            const thumb = document.createElement("button");
            thumb.className = "thumbnail-item";
            thumb.textContent = String(n);
            thumb.addEventListener("click", () => { window.thumbClicks.push(n); });
            thumbs.appendChild(thumb);
        });
        button.appendChild(icons);
        button.appendChild(image);
        button.appendChild(thumbs);
        gallery.appendChild(button);
    } else {
        const grid = document.createElement("div");
        grid.className = "grid-wrap";
        const thumb = document.createElement("button");
        thumb.className = "thumbnail-item";
        thumb.textContent = "g";
        grid.appendChild(thumb);
        gallery.appendChild(grid);
    }
    document.body.appendChild(gallery);
}
"""


def page_html(opts: dict) -> str:
    build = BUILD_GALLERY % {"image": json.dumps(ORIGIN + IMAGE)}
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>{HOST_CSS}</style>
<link rel="stylesheet" href="/style.css">
</head><body>
<script>
function gradioApp() {{ return document; }}
var opts = {json.dumps(opts)};
const hooks = {{after: []}};
function onAfterUiUpdate(fn) {{ hooks.after.push(fn); }}
window.calls = []; window.downloads = []; window.documentClicks = 0;
window.iconClicks = []; window.thumbClicks = [];
function saveImage() {{ window.calls.push("saveImage"); }}
function modalImageSwitch(n) {{ window.calls.push("modalImageSwitch:" + n); }}
HTMLAnchorElement.prototype.click = function () {{ window.downloads.push(this.download); }};
document.addEventListener("click", (event) => {{
    // Only a click on the picture: the script's own press of Gradio's Clear
    // is a click that bubbles here too.
    if (event.target.closest && event.target.closest(".image-button")) window.documentClicks += 1;
}});
{build}
buildGallery("txt2img_gallery", true);
buildGallery("img2img_gallery", false);
</script>
<script src="/javascript/model_chain_gallery_quadrant.js"></script>
</body></html>"""


ON = {quadrant.OPT_ENABLED: True, quadrant.OPT_DESTINATION: quadrant.DESTINATION_BROWSER,
      quadrant.OPT_FOLDER: "", "js_modal_lightbox": False}

# A tiny PNG, for the browser download's fetch.
PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)


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


class Served:
    """The page, the stylesheet, the script, the picture, and the route --
    the route answering as the test plans, and keeping what was posted."""

    def __init__(self, html: str):
        self.html = html
        self.answer = {"status": 200, "body": {"ok": True, "name": "00001-42.png", "path": "/keep/00001-42.png"}}
        self.posted = []

    def __call__(self, route):
        request = route.request
        path = urllib.parse.urlsplit(request.url).path
        if path == "/":
            route.fulfill(status=200, content_type="text/html", body=self.html)
        elif path == "/style.css":
            route.fulfill(status=200, content_type="text/css", body=STYLE.read_text(encoding="utf-8"))
        elif path.startswith("/javascript/") and (ROOT / path[1:]).is_file():
            route.fulfill(status=200, content_type="text/javascript",
                          body=(ROOT / path[1:]).read_text(encoding="utf-8"))
        elif path.startswith("/file="):
            route.fulfill(status=200, content_type="image/png", body=PNG)
        elif path == quadrant.ROUTE:
            self.posted.append({"method": request.method, "body": request.post_data,
                                "type": request.headers.get("content-type", "")})
            route.fulfill(status=self.answer["status"], content_type="application/json",
                          body=json.dumps(self.answer["body"]))
        else:
            route.fulfill(status=404, body="")


def open_page(browser, opts=None):
    page = browser.new_page(viewport={"width": 900, "height": 700})
    served = Served(page_html(dict(ON, **(opts or {}))))
    page.route(ORIGIN + "/**", served)
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!window.mcGalleryQuadrant")
    update(page)
    return page, served


def update(page, times: int = 1):
    for _ in range(times):
        page.evaluate("hooks.after.forEach((fn) => fn())")


STATE = """() => {
    const gallery = document.querySelector("#txt2img_gallery");
    const preview = gallery.querySelector("button.preview");
    const overlays = [...document.querySelectorAll(".gqa-overlay")];
    const cells = overlays.length ? [...overlays[0].children] : [];
    const icons = gallery.querySelector(".icon-buttons");
    return {
        overlays: overlays.length,
        cells: cells.map((cell) => ({cls: cell.className, text: cell.textContent})),
        attached: gallery.dataset.gqaAttached || "", marked: preview.dataset.gqaOverlay || "",
        iconsDisplay: getComputedStyle(icons).display,
        overlayPointer: overlays.length ? getComputedStyle(overlays[0]).pointerEvents : "",
        calls: window.calls, downloads: window.downloads, documentClicks: window.documentClicks,
        iconClicks: window.iconClicks, thumbClicks: window.thumbClicks,
    };
}"""

ZONES = """() => {
    const image = document.querySelector("#txt2img_gallery .image-button").getBoundingClientRect();
    const preview = document.querySelector("#txt2img_gallery button.preview").getBoundingClientRect();
    const overlay = document.querySelector(".gqa-overlay").getBoundingClientRect();
    const cells = [...document.querySelector(".gqa-overlay").children].map((cell) => {
        const r = cell.getBoundingClientRect();
        return {cls: cell.className.split(" ")[1], left: r.left - preview.left, top: r.top - preview.top,
                right: r.right - preview.left, bottom: r.bottom - preview.top,
                zone: window.mcGalleryQuadrant.zoneAt((r.left + r.right) / 2 - image.left,
                                                      (r.top + r.bottom) / 2 - image.top,
                                                      image.width, image.height)};
    });
    return {width: preview.width, height: preview.height, image: {width: image.width, height: image.height},
            overlay: {left: overlay.left - preview.left, top: overlay.top - preview.top,
                      width: overlay.width, height: overlay.height}, cells};
}"""


def click_image(page, x, y):
    page.click("#txt2img_gallery .image-button", position={"x": x, "y": y})


class TestTheOverlay:
    def test_drawn_once_with_the_five_zones_in_order_and_gradios_icons_hidden(self, browser):
        page, _ = open_page(browser)
        update(page, 2)
        found = page.evaluate(STATE)

        assert found["overlays"] == 1, "one overlay, however many updates"
        assert [cell["cls"] for cell in found["cells"]] == [
            "gqa-cell gqa-tl", "gqa-cell gqa-tm", "gqa-cell gqa-tr", "gqa-cell gqa-bl", "gqa-cell gqa-br"]
        assert [cell["text"] for cell in found["cells"]] == [
            "⬇ Download", "✕ Unfocus", "💾 Save", "◀ Prev", "Next ▶"]
        assert found["attached"] == "true" and found["marked"] == "true"
        assert found["iconsDisplay"] == "none", "Gradio's own Download and Clear, hidden"
        assert found["overlayPointer"] == "none", "the overlay takes no pointer; the clicks reach the preview"
        page.close()

    def test_the_cells_cover_the_zones_the_clicks_are_read_by(self, browser):
        """The overlay is the preview button's grid, as the standalone drew it:
        thirds and halves of the preview's width, two rows of its height --
        which, with the thumbnail strip inside the preview under the picture,
        is a little more than the picture's. The clicks are read against the
        picture; each cell's centre lands in its own zone."""
        page, _ = open_page(browser)
        found = page.evaluate(ZONES)
        w, h = found["width"], found["height"]
        assert (w, h) == (600, 460) and found["image"] == {"width": 600, "height": 400}
        assert found["overlay"] == {"left": 0, "top": 0, "width": w, "height": h}, "the whole preview"
        boxes = {cell["cls"]: cell for cell in found["cells"]}
        expected = {"gqa-tl": (0, 0, w / 3, h / 2), "gqa-tm": (w / 3, 0, 2 * w / 3, h / 2),
                    "gqa-tr": (2 * w / 3, 0, w, h / 2), "gqa-bl": (0, h / 2, w / 2, h),
                    "gqa-br": (w / 2, h / 2, w, h)}
        for cls, (left, top, right, bottom) in expected.items():
            box = boxes[cls]
            assert abs(box["left"] - left) <= 1 and abs(box["top"] - top) <= 1, (cls, box)
            assert abs(box["right"] - right) <= 1 and abs(box["bottom"] - bottom) <= 1, (cls, box)
        assert [cell["zone"] for cell in found["cells"]] == [0, 1, 2, 3, 4], "each cell's centre is its zone"
        page.close()


class TestTheZones:
    def test_each_zone_does_its_action_and_the_click_goes_no_further(self, browser):
        page, _ = open_page(browser)
        click_image(page, 100, 100)                       # top-left: the browser download
        page.wait_for_function("window.downloads.length === 1")
        click_image(page, 300, 100)                       # top-middle: leave the preview
        click_image(page, 500, 100)                       # top-right: Forge's Save
        click_image(page, 150, 300)                       # bottom-left: previous
        click_image(page, 450, 300)                       # bottom-right: next
        found = page.evaluate(STATE)

        assert found["downloads"] == ["00001-42.png"]
        assert found["iconClicks"] == ["Clear"], "the preview left through Gradio's own Clear"
        assert found["calls"] == ["saveImage", "modalImageSwitch:-1", "modalImageSwitch:1"]
        assert found["documentClicks"] == 0, "a click in a zone reaches nothing behind the gallery"
        page.close()

    def test_a_thumbnail_and_the_grid_are_left_to_gradio(self, browser):
        page, _ = open_page(browser)
        page.click("#txt2img_gallery .thumbnail-item >> nth=1")
        page.click("#img2img_gallery .thumbnail-item")
        found = page.evaluate(STATE)

        assert found["thumbClicks"] == [2]
        assert found["calls"] == [] and found["downloads"] == []
        assert page.evaluate("window.thumbClicks.length") == 1 and found["documentClicks"] == 0
        page.close()

    def test_off_or_with_the_lightbox_nothing_is_drawn_or_taken(self, browser):
        for opts in ({quadrant.OPT_ENABLED: False}, {"js_modal_lightbox": True}):
            page, _ = open_page(browser, opts)
            update(page, 2)
            click_image(page, 100, 100)
            found = page.evaluate(STATE)

            assert found["overlays"] == 0 and found["marked"] == "", opts
            assert found["calls"] == [] and found["downloads"] == [], opts
            assert found["documentClicks"] == 1, "the click went on to the page"
            page.close()


class TestTheFolderDestination:
    FOLDER = {quadrant.OPT_DESTINATION: quadrant.DESTINATION_FOLDER, quadrant.OPT_FOLDER: "keep"}

    def test_the_top_left_zone_posts_the_picture_and_shows_the_answer(self, browser):
        page, served = open_page(browser, self.FOLDER)
        assert page.evaluate(STATE)["cells"][0]["text"] == "⬇ Folder", "the zone says where it saves"

        click_image(page, 100, 100)
        page.wait_for_function("document.querySelector('.gqa-tl').classList.contains('gqa-result-visible')")
        shown = page.evaluate("document.querySelector('.gqa-tl').textContent")

        assert served.posted and served.posted[0]["method"] == "POST"
        assert json.loads(served.posted[0]["body"]) == {"src": ORIGIN + IMAGE}
        assert served.posted[0]["type"].startswith("application/json")
        assert shown == "Saved 00001-42.png"
        assert page.evaluate("window.downloads") == [], "no browser download"
        # The result gives way to the title again.
        page.wait_for_function("document.querySelector('.gqa-tl').textContent === '⬇ Folder'", timeout=4000)
        assert page.evaluate("document.querySelector('.gqa-tl').classList.contains('gqa-result-visible')") is False
        page.close()

    def test_a_refusal_is_shown_in_the_zone_in_the_servers_words(self, browser):
        page, served = open_page(browser, self.FOLDER)
        served.answer = {"status": 400, "body": {"ok": False, "error": "No folder is set."}}

        click_image(page, 100, 100)
        page.wait_for_function("document.querySelector('.gqa-tl').classList.contains('gqa-result-visible')")

        assert page.evaluate("document.querySelector('.gqa-tl').textContent") == "No folder is set."
        page.close()

    def test_the_title_follows_the_setting_without_a_reload(self, browser):
        page, _ = open_page(browser)
        assert page.evaluate(STATE)["cells"][0]["text"] == "⬇ Download"
        page.evaluate("(name) => { opts[name] = %s; }" % json.dumps(quadrant.DESTINATION_FOLDER),
                      quadrant.OPT_DESTINATION)
        update(page)
        assert page.evaluate(STATE)["cells"][0]["text"] == "⬇ Folder"
        page.evaluate("(name) => { opts[name] = %s; }" % json.dumps(quadrant.DESTINATION_BROWSER),
                      quadrant.OPT_DESTINATION)
        update(page)
        assert page.evaluate(STATE)["cells"][0]["text"] == "⬇ Download"
        page.close()


class TestBesideTheStandalone:
    def test_a_second_copy_of_the_script_skips_a_gallery_already_marked(self, browser):
        """The standalone extension, still enabled, loads the same script under
        its own name: whichever reaches a gallery first marks it, and the
        other skips it, so one overlay is drawn and one action taken."""
        page, _ = open_page(browser)
        page.add_script_tag(content=SCRIPT.read_text(encoding="utf-8"))
        page.wait_for_function("hooks.after.length === 2")
        update(page)
        click_image(page, 500, 100)
        found = page.evaluate(STATE)

        assert found["overlays"] == 1
        assert found["calls"] == ["saveImage"], "one action, not two"
        page.close()


class TestTheContract:
    def test_the_script_and_python_name_the_same_options_and_route(self, browser):
        page, _ = open_page(browser)
        names = page.evaluate("window.mcGalleryQuadrant")

        assert names["OPT_ENABLED"] == quadrant.OPT_ENABLED
        assert names["OPT_DESTINATION"] == quadrant.OPT_DESTINATION
        assert names["DESTINATION_FOLDER"] == quadrant.DESTINATION_FOLDER
        assert names["ROUTE"] == quadrant.ROUTE
        page.close()

    def test_the_stylesheet_carries_the_standalones_section(self):
        css = STYLE.read_text(encoding="utf-8")
        start = css.index("Model Chain -- Gallery Quadrant")
        section = css[start:]
        for cls in (".gqa-overlay", ".gqa-tl", ".gqa-tm", ".gqa-tr", ".gqa-bl", ".gqa-br", ".gqa-cell",
                    ".gqa-titles-visible", ".gqa-highlight-visible", ".gqa-result-visible"):
            assert cls in section, cls
        assert ".gradio-gallery > button.preview .icon-buttons" in section
