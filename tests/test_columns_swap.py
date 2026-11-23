"""Txt2Img's and Img2Img's two columns swapped by a double tap on Forge's
divider, in Chromium, on a page carrying Forge Neo's own scripts.

Asked: "an option to be able to switch the two main columns on text to image
and image to image tab ... double tab the drag area that i can drag to resize
the columns should switch them left and right."

The page is Forge-shaped where it matters: Gradio 4.40's row (a positioned
flex row, so its children's ``offsetLeft`` counts from its edge), column, tabs
and tab item rules at their real values, Forge Neo's own rules for the divider
and the sticky results column, and Forge Neo's own ``javascript/resizeHandle.js``
and ``extensions-builtin/mobile/javascript/mobile.js``, as they are at Forge
Neo's 6d5aa3b, run in Forge's order before this extension's scripts. What is
measured:

    the swap      a double click, and two touches, on the divider put the
                  results on the left and the settings on the right, each as
                  wide as it was, and again put them back; Forge's own double
                  tap (its first widths back) does not run;
    the drag      Forge's drag sizes the settings while they are first; while
                  swapped this extension's sizes the results, both ways, a
                  mouse's and a finger's, in Forge's own template;
    kept          a reload draws the swap from the start, per tab; a narrow
                  window stacks the row in Forge's order and a wide one swaps
                  it again;
    Generate      Forge's mobile script never takes the swapped page for a
                  phone's, so Generate stays where it was;
    the theme     on and off; with it on, its one scroller still holds the
                  columns, swapped;
    off           the setting off draws Forge's order and leaves the double
                  click to Forge.

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
SCRIPT = ROOT / "javascript" / "model_chain_columns.js"
STYLE = ROOT / "style.css"
ORIGIN = "http://forge-columns.test"

# Forge Neo's javascript/resizeHandle.js, at 6d5aa3b, as it is.
FORGE_RESIZE_HANDLE_JS = r"""
(function () {
    const GRADIO_MIN_WIDTH = 320;
    const PAD = 16;
    const DEBOUNCE_TIME = 100;
    const DOUBLE_TAP_DELAY = 200; //ms

    const R = {
        tracking: false,
        parent: null,
        parentWidth: null,
        leftCol: null,
        leftColStartWidth: null,
        screenX: null,
        lastTapTime: null,
    };

    let resizeTimer;
    let parents = [];

    function setLeftColGridTemplate(el, width) {
        el.style.gridTemplateColumns = `${width}px 16px 1fr`;
    }

    function displayResizeHandle(parent) {
        if (!parent.needHideOnMoblie) {
            return true;
        }
        if (window.innerWidth < GRADIO_MIN_WIDTH * 2 + PAD * 4) {
            parent.style.display = "flex";
            parent.resizeHandle.style.display = "none";
            return false;
        } else {
            parent.style.display = "grid";
            parent.resizeHandle.style.display = "block";
            return true;
        }
    }

    function afterResize(parent) {
        if (displayResizeHandle(parent) && parent.style.gridTemplateColumns != parent.style.originalGridTemplateColumns) {
            const oldParentWidth = R.parentWidth;
            const newParentWidth = parent.offsetWidth;
            const widthL = parseInt(parent.style.gridTemplateColumns.split(" ")[0]);

            const ratio = newParentWidth / oldParentWidth;

            const newWidthL = Math.max(Math.floor(ratio * widthL), parent.minLeftColWidth);
            setLeftColGridTemplate(parent, newWidthL);

            R.parentWidth = newParentWidth;
        }
    }

    function setup(parent) {
        function onDoubleClick(evt) {
            evt.preventDefault();
            evt.stopPropagation();

            parent.style.gridTemplateColumns = parent.style.originalGridTemplateColumns;
        }

        const leftCol = parent.firstElementChild;
        const rightCol = parent.lastElementChild;

        parents.push(parent);

        parent.style.display = "grid";
        parent.style.gap = "0";
        let leftColTemplate = "";
        if (parent.children[0].style.flexGrow) {
            leftColTemplate = `${parent.children[0].style.flexGrow}fr`;
            parent.minLeftColWidth = GRADIO_MIN_WIDTH;
            parent.minRightColWidth = GRADIO_MIN_WIDTH;
            parent.needHideOnMoblie = true;
        } else {
            leftColTemplate = parent.children[0].style.flexBasis;
            parent.minLeftColWidth = parent.children[0].style.flexBasis.slice(0, -2) / 2;
            parent.minRightColWidth = 0;
            parent.needHideOnMoblie = false;
        }

        if (!leftColTemplate) {
            leftColTemplate = "1fr";
        }

        const gridTemplateColumns = `${leftColTemplate} ${PAD}px ${parent.children[1].style.flexGrow}fr`;
        parent.style.gridTemplateColumns = gridTemplateColumns;
        parent.style.originalGridTemplateColumns = gridTemplateColumns;

        const resizeHandle = document.createElement("div");
        resizeHandle.classList.add("resize-handle");
        parent.insertBefore(resizeHandle, rightCol);
        parent.resizeHandle = resizeHandle;

        ["mousedown", "touchstart"].forEach((eventType) => {
            resizeHandle.addEventListener(eventType, (evt) => {
                if (eventType.startsWith("mouse")) {
                    if (evt.button !== 0) return;
                } else {
                    if (evt.changedTouches.length !== 1) return;

                    const currentTime = new Date().getTime();
                    if (R.lastTapTime && currentTime - R.lastTapTime <= DOUBLE_TAP_DELAY) {
                        onDoubleClick(evt);
                        return;
                    }

                    R.lastTapTime = currentTime;
                }

                evt.preventDefault();
                evt.stopPropagation();

                document.body.classList.add("resizing");

                R.tracking = true;
                R.parent = parent;
                R.parentWidth = parent.offsetWidth;
                R.leftCol = leftCol;
                R.leftColStartWidth = leftCol.offsetWidth;
                if (eventType.startsWith("mouse")) {
                    R.screenX = evt.screenX;
                } else {
                    R.screenX = evt.changedTouches[0].screenX;
                }
            });
        });

        resizeHandle.addEventListener("dblclick", onDoubleClick);

        afterResize(parent);
    }

    ["mousemove", "touchmove"].forEach((eventType) => {
        window.addEventListener(eventType, (evt) => {
            if (eventType.startsWith("mouse")) {
                if (evt.button !== 0) return;
            } else {
                if (evt.changedTouches.length !== 1) return;
            }

            if (R.tracking) {
                if (eventType.startsWith("mouse")) {
                    evt.preventDefault();
                }
                evt.stopPropagation();

                let delta = 0;
                if (eventType.startsWith("mouse")) {
                    delta = R.screenX - evt.screenX;
                } else {
                    delta = R.screenX - evt.changedTouches[0].screenX;
                }
                const leftColWidth = Math.max(
                    Math.min(R.leftColStartWidth - delta, R.parent.offsetWidth - R.parent.minRightColWidth - PAD),
                    R.parent.minLeftColWidth,
                );
                setLeftColGridTemplate(R.parent, leftColWidth);
            }
        });
    });

    ["mouseup", "touchend"].forEach((eventType) => {
        window.addEventListener(eventType, (evt) => {
            if (eventType.startsWith("mouse")) {
                if (evt.button !== 0) return;
            } else {
                if (evt.changedTouches.length !== 1) return;
            }

            if (R.tracking) {
                evt.preventDefault();
                evt.stopPropagation();

                R.tracking = false;

                document.body.classList.remove("resizing");
            }
        });
    });

    window.addEventListener("resize", () => {
        clearTimeout(resizeTimer);

        resizeTimer = setTimeout(function () {
            for (const parent of parents) {
                afterResize(parent);
            }
        }, DEBOUNCE_TIME);
    });

    setupResizeHandle = setup;
})();

function setupAllResizeHandles() {
    for (let elem of gradioApp().querySelectorAll(".resize-handle-row")) {
        if (!elem.querySelector(".resize-handle") && !elem.children[0].classList.contains("hidden")) {
            setupResizeHandle(elem);
        }
    }
}

onUiLoaded(setupAllResizeHandles);
"""

# Forge Neo's extensions-builtin/mobile/javascript/mobile.js, at 6d5aa3b, as it is.
FORGE_MOBILE_JS = r"""
(function () {
    let isSetupForMobile = false;

    function isMobile() {
        for (const tab of ["txt2img", "img2img"]) {
            const imageTab = gradioApp().getElementById(tab + "_results");
            if (imageTab && imageTab.offsetParent && imageTab.offsetLeft === 0) {
                return true;
            }
        }

        return false;
    }

    function reportWindowSize() {
        // not applicable for compact prompt layout
        if (gradioApp().querySelector(".toprow-compact-tools")) return;

        const currentlyMobile = isMobile();
        if (currentlyMobile === isSetupForMobile) return;
        isSetupForMobile = currentlyMobile;

        for (const tab of ["txt2img", "img2img"]) {
            const button = gradioApp().getElementById(tab + "_generate_box");
            const target = gradioApp().getElementById(currentlyMobile ? tab + "_results" : tab + "_actions_column");
            target.insertBefore(button, target.firstElementChild);

            gradioApp()
                .getElementById(tab + "_results")
                .classList.toggle("mobile", currentlyMobile);
        }
    }

    window.addEventListener("resize", reportWindowSize);

    onUiLoaded(reportWindowSize);
})();
"""

# Gradio 4.40's own rules for what the page is made of (its built stylesheet:
# Row, Column, Tabs and TabItem), at their values. The row is positioned:
# that is why a results column on the row's left edge has `offsetLeft` 0,
# which Forge's mobile script reads as a phone.
GRADIO_CSS = """
*, ::before, ::after { box-sizing: border-box; }
html, body { margin: 0; }
body { background: #0b0f19; color: #e5e7eb; font-family: sans-serif; --layout-gap: 16px; --block-padding: 10px;
       --border-color-primary: #374151; --size-4: 16px; --size-8: 32px; }
.app { padding: var(--size-4) var(--size-8); }
.row { display: flex; flex-wrap: wrap; gap: var(--layout-gap); width: 100%; position: relative; }
.row.unequal-height { align-items: flex-start; }
.row > * { flex: 1 1 0%; flex-wrap: wrap; min-width: min(160px, 100%); }
.column { display: flex; position: relative; flex-direction: column; gap: var(--layout-gap); }
.column > * { width: 100%; }
.tabs { position: relative; }
.tab-nav { display: flex; position: relative; flex-wrap: wrap; border-bottom: 1px solid var(--border-color-primary); }
.tab-nav button { padding: 4px 16px; border: 1px solid transparent; background: transparent; color: inherit; }
.tabitem { display: flex; position: relative; border: 1px solid var(--border-color-primary); border-top: none;
           padding: var(--block-padding); }
.block { min-height: 40px; }
textarea { width: 100%; min-height: 60px; }
"""

# Forge Neo's own (style.css): the divider, its drag, the sticky results, its
# rows' align-content, and what its mobile script's class does.
FORGE_CSS = """
.unequal-height { align-content: flex-start; }
#img2img_results, #txt2img_results, #extras_results { position: sticky; top: 0.5em; }
[id$="_results"].mobile { margin-top: 28px; }
body.resizing { cursor: col-resize !important; }
body.resizing * { pointer-events: none !important; }
body.resizing .resize-handle { pointer-events: initial !important; }
.resize-handle { position: relative; cursor: col-resize; grid-column: 2 / 3; min-width: 16px !important;
                 max-width: 16px !important; height: 100%; }
.resize-handle::after { content: ""; position: absolute; top: 0; bottom: 0; left: 7.5px;
                        border-left: 1px dashed var(--border-color-primary); }
"""


def tab_html(tab: str, shown: bool, blocks: int) -> str:
    settings = "".join(f'<div class="block" id="{tab}_setting_{n}">Setting {n}</div>' for n in range(blocks))
    return f"""
  <div id="tab_{tab}" class="tabitem" role="tabpanel" style="display: {'block' if shown else 'none'}">
    <div class="column" style="flex-grow: 1">
    <div id="{tab}_toprow" class="row">
      <div id="{tab}_prompt_container" class="column">
        <div id="{tab}_prompt" class="block"><label><textarea aria-label="Prompt"></textarea></label></div>
      </div>
      <div id="{tab}_actions_column" class="column">
        <div id="{tab}_generate_box" class="row generate-box"><button id="{tab}_generate" class="lg primary gradio-button">Generate</button></div>
      </div>
    </div>
    <div id="{tab}_extra_tabs" class="tabs extra-networks">
      <div class="tab-nav" role="tablist"><button role="tab" class="selected" aria-controls="{tab}_generation">Generation</button></div>
      <div id="{tab}_generation" class="tabitem" role="tabpanel" style="display: block">
        <div class="row resize-handle-row unequal-height">
          <div id="{tab}_settings" class="column" style="flex-grow: 1; min-width: min(320px, 100%);">{settings}</div>
          <div id="{tab}_results" class="column" style="flex-grow: 1; min-width: min(320px, 100%);">
            <div id="{tab}_results_panel" class="column panel">
              <div id="{tab}_gallery_container" class="gr-group"><div id="{tab}_gallery" class="block" style="height: 300px"></div></div>
              <div id="image_buttons_{tab}" class="row image-buttons"><button id="{tab}_open_folder" class="gradio-button tool">Open</button></div>
              <div id="html_info_{tab}" class="block infotext"><div class="prose"></div></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    </div>
  </div>"""


def page_html(opts: dict, *, theme: bool, blocks: int, assistant: bool = False) -> str:
    """`assistant` puts the Forge Assistant's scripts on the page, in Forge's
    order: this extension's folder is loaded alphabetically, so they come
    before the columns' script."""
    helpers = "".join(f'<script src="/javascript/{path.name}"></script>'
                      for path in sorted((ROOT / "javascript").glob("forge_assistant*.js"))) if assistant else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{GRADIO_CSS}{FORGE_CSS}</style>
<link rel="stylesheet" href="/style.css">
</head><body class="dark">
<gradio-app><div class="gradio-container app">
<div id="tabs" class="tabs">
  <div class="tab-nav scroll-hide" role="tablist">
    <button role="tab" id="tab_txt2img-button" aria-controls="tab_txt2img" aria-selected="true" class="selected">txt2img</button>
    <button role="tab" id="tab_img2img-button" aria-controls="tab_img2img" aria-selected="false">img2img</button>
  </div>
  {tab_html("txt2img", True, blocks)}
  {tab_html("img2img", False, blocks)}
</div>
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
function showTab(id) {{
    document.querySelectorAll('#tabs > .tabitem').forEach((p) => {{ p.style.display = p.id === id ? 'block' : 'none'; }});
    hooks.tab.forEach((fn) => fn());
}}
// What the root carried when this page's own scripts had run and nothing had
// been built from them yet: the first paint's attribute.
window.firstPaint = null;
</script>
<script>{FORGE_RESIZE_HANDLE_JS}</script>
<script>{FORGE_MOBILE_JS}</script>
{helpers}
<script src="/javascript/model_chain_columns.js"></script>
<script>window.firstPaint = document.documentElement.getAttribute("data-mc-columns-swapped");</script>
{'<script src="/javascript/model_chain_theme.js"></script>' if theme else ""}
</body></html>"""


def _answer(route, html: str) -> None:
    path = urllib.parse.urlsplit(route.request.url).path
    if path == "/":
        route.fulfill(status=200, content_type="text/html", body=html)
    elif path == "/style.css":
        route.fulfill(status=200, content_type="text/css", body=STYLE.read_text(encoding="utf-8"))
    elif path.startswith("/javascript/") and "/" not in path[12:] and (ROOT / path[1:]).is_file():
        route.fulfill(status=200, content_type="text/javascript", body=(ROOT / path[1:]).read_text(encoding="utf-8"))
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


def open_page(browser, opts=None, *, theme=False, width=1400, height=900, context=None, touch=False, blocks=30,
              loaded=True, assistant=False):
    """A page of its own context, or one in a context given -- two pages of one
    context share a localStorage, which is what the first paint reads."""
    owner = context or browser.new_context(viewport={"width": width, "height": height}, has_touch=touch)
    page = owner.new_page()
    settings = dict({"model_chain_theme": theme}, **(opts or {}))
    html = page_html(settings, theme=theme, blocks=blocks, assistant=assistant)
    page.route(ORIGIN + "/**", lambda route: _answer(route, html))
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!window.mcColumns" + (" && !!window.mcTheme" if theme else ""))
    if loaded:
        load(page)
        if assistant:
            page.wait_for_function("!!(window.forgeAssistant && window.forgeAssistant.shell)")
            settle(page)
    return page


def load(page) -> None:
    """What Forge does once the page is built: its onUiLoaded callbacks, in the
    order the scripts registered them (Forge's divider, its mobile script,
    then this extension's), then its options."""
    page.evaluate("hooks.loaded.forEach((fn) => fn())")
    page.evaluate("hooks.changed.forEach((fn) => fn())")
    settle(page)


LAYOUT = """(tab) => {
    const q = (s) => document.querySelector(s);
    const box = (node) => { const r = node.getBoundingClientRect(); return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width}; };
    const results = q("#" + tab + "_results");
    const row = results.parentElement;
    const handle = row.querySelector(":scope > .resize-handle");
    return {
        settings: box(q("#" + tab + "_settings")), results: box(results), handle: handle ? box(handle) : null,
        display: getComputedStyle(row).display, template: row.style.gridTemplateColumns,
        original: row.style.originalGridTemplateColumns || "",
        attribute: document.documentElement.getAttribute("data-mc-columns-swapped"),
        kept: localStorage.getItem("mc-columns-swapped:" + tab),
        offsetLeft: results.offsetLeft, mobile: results.classList.contains("mobile"),
        generateIn: q("#" + tab + "_generate_box").parentElement.id || q("#" + tab + "_generate_box").parentElement.className,
        resizing: document.body.classList.contains("resizing"),
    };
}"""


# Every grid template written on Txt2Img's row from now on, by its first track.
WRITES = """() => {
    const row = document.querySelector('#txt2img_results').parentElement;
    window.written = [];
    new MutationObserver(() => window.written.push(parseFloat(row.style.gridTemplateColumns)))
        .observe(row, {attributes: true, attributeFilter: ['style']});
}"""


def layout(page, tab="txt2img") -> dict:
    return page.evaluate(LAYOUT, tab)


def handle_point(page, tab="txt2img", y_fraction=0.25) -> tuple:
    box = page.evaluate("""([tab, f]) => { const r = document.querySelector('#' + tab + '_results').parentElement
        .querySelector(':scope > .resize-handle').getBoundingClientRect();
        return [r.left + r.width / 2, Math.min(r.top + 40 + (r.height - 80) * f, innerHeight - 20)]; }""", [tab, y_fraction])
    return box[0], box[1]


def swapped(info: dict) -> bool:
    return info["results"]["right"] <= info["handle"]["left"] + 1 and info["handle"]["right"] <= info["settings"]["left"] + 1


def in_order(info: dict) -> bool:
    return info["settings"]["right"] <= info["handle"]["left"] + 1 and info["handle"]["right"] <= info["results"]["left"] + 1


def mouse_drag(page, x, y, travel, steps=8):
    page.mouse.move(x, y)
    page.mouse.down()
    for k in range(1, steps + 1):
        page.mouse.move(x + travel * k / steps, y)
    page.mouse.up()
    settle(page)


def touch_event(cdp, kind, x, y):
    points = [] if kind == "touchEnd" else [{"x": x, "y": y, "radiusX": 1, "radiusY": 1, "force": 1, "id": 1}]
    cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": points})


def finger_drag(page, cdp, x, y, travel, steps=8, down=0):
    """A finger on the divider, `travel` pixels across and `down` pixels down
    -- the browser pans the page by a finger's travel wherever it may."""
    touch_event(cdp, "touchStart", x, y)
    for k in range(1, steps + 1):
        touch_event(cdp, "touchMove", x + travel * k / steps, y + down * k / steps)
    touch_event(cdp, "touchEnd", x + travel, y + down)
    settle(page)


def finger_tap(cdp, x, y):
    touch_event(cdp, "touchStart", x, y)
    touch_event(cdp, "touchEnd", x, y)


def past_the_guard(page):
    """One gesture, one swap: the script ignores a second swap within half a
    second of the first, so a test that swaps twice waits that out."""
    page.wait_for_timeout(560)


# --------------------------------------------------------------------------- #
# The swap                                                                     #
# --------------------------------------------------------------------------- #


class TestTheSwap:

    @pytest.mark.parametrize("theme", [False, True], ids=["forges-page", "the-theme"])
    def test_a_double_click_on_the_divider_swaps_the_columns_and_another_puts_them_back(self, browser, theme):
        page = open_page(browser, theme=theme)
        before = layout(page)
        assert before["display"] == "grid" and in_order(before), before
        assert before["template"] == before["original"] == "1fr 16px 1fr", before
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        after = layout(page)
        assert swapped(after), after
        assert after["attribute"] == "txt2img" and after["kept"] == "1", after
        # Equal halves stay Forge's first template, and each column its width.
        assert after["template"] == "1fr 16px 1fr", after
        assert abs(after["settings"]["width"] - before["settings"]["width"]) <= 1.5, (before, after)
        assert abs(after["results"]["width"] - before["results"]["width"]) <= 1.5, (before, after)
        past_the_guard(page)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        back = layout(page)
        assert in_order(back) and back["attribute"] is None and back["kept"] is None, back
        page.close()

    def test_each_column_keeps_its_width_and_forges_reset_does_not_run(self, browser):
        """Forge's double click put its first widths back; the swap keeps the
        widths a drag left, the settings' 500 px now on the right."""
        page = open_page(browser)
        x, y = handle_point(page)
        start = layout(page)
        mouse_drag(page, x, y, 500 - (x - start["settings"]["left"] + 8))
        dragged = layout(page)
        assert in_order(dragged) and dragged["template"].endswith("px 16px 1fr"), dragged
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        after = layout(page)
        assert swapped(after), after
        assert abs(after["settings"]["width"] - dragged["settings"]["width"]) <= 1.5, (dragged, after)
        # The results lose the pixel of margin they are given on the left.
        assert abs(after["results"]["width"] - (dragged["results"]["width"] - 1)) <= 1.5, (dragged, after)
        assert after["template"] == f"{round(dragged['results']['width'])}px 16px 1fr", after
        past_the_guard(page)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        back = layout(page)
        assert in_order(back) and abs(back["settings"]["width"] - dragged["settings"]["width"]) <= 1.5, (dragged, back)
        page.close()

    def test_two_touches_on_the_divider_swap_the_columns_and_one_does_not(self, browser):
        page = open_page(browser, touch=True)
        cdp = page.context.new_cdp_session(page)
        x, y = handle_point(page)
        finger_tap(cdp, x, y)
        settle(page)
        assert in_order(layout(page)), "one touch is not a double tap"
        page.wait_for_timeout(450)
        finger_tap(cdp, x, y)
        settle(page)
        assert in_order(layout(page)), "two touches further apart than a double tap are two taps"
        # Forge's own double tap would put its first widths back: a drag first,
        # so its reset would show -- out of the last tap's double-tap time.
        page.wait_for_timeout(450)
        finger_drag(page, cdp, x, y, -150)
        dragged = layout(page)
        assert in_order(dragged) and dragged["template"].endswith("px 16px 1fr"), dragged
        page.wait_for_timeout(450)
        x, y = handle_point(page)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(120)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(400)
        after = layout(page)
        assert swapped(after) and after["kept"] == "1", after
        assert abs(after["settings"]["width"] - dragged["settings"]["width"]) <= 1.5, (dragged, after)
        past_the_guard(page)
        x, y = handle_point(page)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(120)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(400)
        assert in_order(layout(page)), "a second double tap puts them back"
        page.close()

    def test_the_double_taps_own_mouse_events_press_nothing_that_moved_under_the_finger(self, browser):
        """Nothing cancels a double tap's touches -- the window's listener is
        passive -- so the browser makes mouse events of them after the
        second: a mousedown, a click and, Chromium, a dblclick for the pair,
        all where the finger was. The swap has put the other column there,
        here the results' gallery; none of them reaches it, and the dblclick
        does not put the columns back."""
        page = open_page(browser, touch=True)
        cdp = page.context.new_cdp_session(page)
        x, y = handle_point(page)
        mouse_drag(page, x, y, -150)
        page.wait_for_timeout(450)
        x, y = handle_point(page, y_fraction=0)
        page.evaluate("""() => {
            window.heard = [];
            const results = document.querySelector('#txt2img_results');
            ['mousedown', 'mouseup', 'click', 'dblclick'].forEach((type) => results.addEventListener(type, () => window.heard.push(type)));
        }""")
        finger_tap(cdp, x, y)
        page.wait_for_timeout(120)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(450)
        info = layout(page)
        assert swapped(info), info
        under = page.evaluate("([x, y]) => document.elementFromPoint(x, y).closest('#txt2img_results') !== null", [x, y])
        assert under, "the results are under the finger now -- what would have been pressed"
        assert page.evaluate("window.heard") == [], page.evaluate("window.heard")
        page.close()

    def test_a_second_swap_asked_for_at_once_is_the_same_gestures(self, browser):
        """A browser that reports one gesture twice -- here two double clicks
        in a row, inside half a second -- swaps the columns once."""
        page = open_page(browser)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        page.mouse.dblclick(x, y)
        settle(page)
        assert swapped(layout(page))
        past_the_guard(page)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        assert in_order(layout(page)), "a double click after that is a gesture of its own"
        page.close()

    def test_a_drag_then_a_touch_soon_after_is_not_a_double_tap(self, browser):
        page = open_page(browser, touch=True)
        cdp = page.context.new_cdp_session(page)
        x, y = handle_point(page)
        finger_drag(page, cdp, x, y, -80, steps=3)
        x, y = handle_point(page)
        finger_tap(cdp, x, y)
        settle(page)
        assert in_order(layout(page)), "the first touch travelled: a drag, not a tap"
        page.close()

    def test_img2img_keeps_a_choice_of_its_own(self, browser):
        page = open_page(browser)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        page.evaluate("showTab('tab_img2img')")
        settle(page)
        img = layout(page, "img2img")
        assert in_order(img) and img["kept"] is None, img
        past_the_guard(page)
        x, y = handle_point(page, "img2img")
        page.mouse.dblclick(x, y)
        settle(page)
        img = layout(page, "img2img")
        assert swapped(img) and img["kept"] == "1" and img["attribute"] == "txt2img img2img", img
        past_the_guard(page)
        page.mouse.dblclick(x, y)
        settle(page)
        assert in_order(layout(page, "img2img")) and layout(page, "img2img")["attribute"] == "txt2img"
        page.close()


# --------------------------------------------------------------------------- #
# The drag                                                                     #
# --------------------------------------------------------------------------- #


class TestTheDrag:

    def test_forges_drag_sizes_the_settings_while_they_are_first(self, browser):
        page = open_page(browser)
        before = layout(page)
        x, y = handle_point(page)
        mouse_drag(page, x, y, 100)
        after = layout(page)
        assert in_order(after), after
        assert abs(after["settings"]["width"] - (before["settings"]["width"] + 100)) <= 1.5, (before, after)
        assert after["template"] == f"{round(after['settings']['width'])}px 16px 1fr", after
        page.close()

    @pytest.mark.parametrize("travel", [120, -120], ids=["right", "left"])
    def test_swapped_the_drag_sizes_the_results_under_the_pointer(self, browser, travel):
        page = open_page(browser)
        # Unequal widths first, so a drag that started from the wrong
        # column's width would jump.
        x, y = handle_point(page)
        mouse_drag(page, x, y, -100)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        before = layout(page)
        assert before["results"]["width"] - before["settings"]["width"] > 150, before
        assert swapped(before), before
        page.evaluate(WRITES)
        x, y = handle_point(page)
        mouse_drag(page, x, y, travel)
        after = layout(page)
        # Every width written on the way, not only the last: Forge's drag,
        # had it heard the press, would write the settings' width first in
        # each move, and a frame could show it.
        written = page.evaluate("window.written")
        low = min(before["results"]["width"], after["results"]["width"]) - 1
        high = max(before["results"]["width"], after["results"]["width"]) + 3
        assert written and all(low <= width <= high for width in written), (low, high, written)
        assert swapped(after) and not after["resizing"], after
        # The divider follows the pointer: the column on its left -- the
        # results -- grows by a drag to the right, and the settings give way.
        assert abs(after["handle"]["left"] - (before["handle"]["left"] + travel)) <= 1.5, (before, after)
        assert abs(after["results"]["width"] - (before["results"]["width"] + travel)) <= 1.5, (before, after)
        assert abs(after["settings"]["width"] - (before["settings"]["width"] - travel)) <= 1.5, (before, after)
        # Forge's own template, the first track the results', so Forge's
        # scaling of it on a window resize goes on working.
        assert after["template"] == f"{round(after['results']['width'] + 1)}px 16px 1fr", after
        page.close()

    def test_swapped_the_page_takes_no_pointer_while_the_divider_is_dragged(self, browser):
        """Forge's own class for its drag, on the body for the drag's length:
        its stylesheet takes the pointer from everything but the divider, and
        the Forge Assistant's fill waits for the class to go."""
        page = open_page(browser)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        x, y = handle_point(page)
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 40, y)
        during = layout(page)
        page.mouse.up()
        after = layout(page)
        assert swapped(during) and during["resizing"] is True, during
        assert after["resizing"] is False, after
        page.close()

    def test_swapped_a_finger_drags_the_results_too(self, browser):
        page = open_page(browser, touch=True)
        cdp = page.context.new_cdp_session(page)
        # Unequal widths first, so Forge's drag -- from the settings' width --
        # would jump the results.
        x, y = handle_point(page)
        mouse_drag(page, x, y, -100)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        page.wait_for_timeout(350)
        before = layout(page)
        assert swapped(before) and before["results"]["width"] - before["settings"]["width"] > 150, before
        x, y = handle_point(page)
        finger_drag(page, cdp, x, y, 90, down=-200)
        after = layout(page)
        assert swapped(after) and not after["resizing"], after
        assert abs(after["results"]["width"] - (before["results"]["width"] + 90)) <= 1.5, (before, after)
        assert abs(after["settings"]["width"] - (before["settings"]["width"] - 90)) <= 1.5, (before, after)
        # The finger also travelled up the divider: the page did not pan.
        assert page.evaluate("[window.scrollX, window.scrollY]") == [0, 0], "the page panned under the finger"
        page.close()

    def test_swapped_the_drag_stops_at_forges_minimum(self, browser):
        page = open_page(browser)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        x, y = handle_point(page)
        mouse_drag(page, x, y, -2000)
        narrow = layout(page)
        assert abs(narrow["results"]["width"] + 1 - 320) <= 1.5, narrow
        x, y = handle_point(page)
        mouse_drag(page, x, y, 4000)
        wide = layout(page)
        assert abs(wide["settings"]["width"] - 320) <= 1.5, wide
        page.close()


# --------------------------------------------------------------------------- #
# Kept, and the narrow window                                                  #
# --------------------------------------------------------------------------- #


class TestKept:

    def test_a_reload_draws_the_swap_from_the_first_paint(self, browser):
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = open_page(browser, context=context)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        page.close()
        again = open_page(browser, context=context, loaded=False)
        # The root says so before Forge has built anything.
        assert again.evaluate("window.firstPaint") == "txt2img"
        load(again)
        info = layout(again)
        assert swapped(info) and info["template"] == "1fr 16px 1fr", info
        again.close()
        context.close()

    def test_a_narrow_window_stacks_in_forges_order_and_a_wide_one_swaps_again(self, browser):
        page = open_page(browser)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        page.set_viewport_size({"width": 600, "height": 900})
        page.wait_for_timeout(250)   # Forge's divider waits 100 ms after a resize
        settle(page)
        narrow = layout(page)
        assert narrow["display"] == "flex", narrow
        assert narrow["results"]["top"] >= narrow["settings"]["bottom"] - 1, "stacked, the settings on top, as Forge stacks"
        assert narrow["offsetLeft"] == 0, "no margin while stacked: the results are the phone's column"
        page.set_viewport_size({"width": 1400, "height": 900})
        page.wait_for_timeout(250)
        settle(page)
        wide = layout(page)
        assert wide["display"] == "grid" and swapped(wide), wide
        page.close()

    def test_a_window_resize_scales_the_swapped_drag_as_forge_scales_its_own(self, browser):
        page = open_page(browser)
        # A drag of Forge's first, which is when Forge learns the row's width.
        x, y = handle_point(page)
        mouse_drag(page, x, y, -60)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        x, y = handle_point(page)
        mouse_drag(page, x, y, 100)
        before = layout(page)
        page.set_viewport_size({"width": 1200, "height": 900})
        page.wait_for_timeout(250)
        settle(page)
        after = layout(page)
        assert swapped(after), after
        ratio = after["results"]["width"] / before["results"]["width"]
        assert 0.75 < ratio < 0.95, (before, after)
        page.close()


# --------------------------------------------------------------------------- #
# Forge's mobile script                                                        #
# --------------------------------------------------------------------------- #


class TestGenerateStays:

    @pytest.mark.parametrize("theme", [False, True], ids=["forges-page", "the-theme"])
    def test_forges_mobile_script_never_takes_the_swapped_page_for_a_phone(self, browser, theme):
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = open_page(browser, context=context, theme=theme)
        before = layout(page)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        page.close()
        # Loaded swapped: Forge's mobile script looks at the results column
        # once the page is built, and at every window resize.
        page = open_page(browser, context=context, theme=theme)
        info = layout(page)
        assert swapped(info) and info["offsetLeft"] >= 1, info
        assert not info["mobile"], info
        assert info["generateIn"] == before["generateIn"], (before, info)
        for width in (1300, 1500, 1400):
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(150)
        settle(page)
        info = layout(page)
        assert swapped(info) and not info["mobile"] and info["generateIn"] == before["generateIn"], (before, info)
        page.close()
        context.close()


# --------------------------------------------------------------------------- #
# With the theme                                                               #
# --------------------------------------------------------------------------- #


class TestWithTheTheme:

    def test_the_one_scroller_still_holds_the_swapped_columns(self, browser):
        page = open_page(browser, theme=True, blocks=40)
        settle(page, 8)
        held = page.evaluate("mcTheme.explainScroll()[0]")
        assert held["reason"] == "", held
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page, 8)
        info = layout(page)
        assert swapped(info), info
        held = page.evaluate("mcTheme.explainScroll()[0]")
        assert held["reason"] == "", "side by side either way round, not stacked: " + json.dumps(held)
        assert page.evaluate("document.querySelector('#txt2img_settings').classList.contains('mc-theme-scroll-settings')")
        page.close()


# --------------------------------------------------------------------------- #
# The Forge Assistant's dock                                                   #
# --------------------------------------------------------------------------- #

# Where Txt2Img's columns are with the Forge Assistant's dock on the page.
DOCKED = """() => {
    const results = document.getElementById("txt2img_results");
    const settings = document.getElementById("txt2img_settings");
    const row = results.parentElement;
    const r = results.getBoundingClientRect(), w = row.getBoundingClientRect();
    const handle = row.querySelector(":scope > .resize-handle");
    return {left: r.left - w.left, width: r.width, rowWidth: w.width,
            docked: forgeAssistant.dock.isDocked("tab_txt2img"),
            settingsPosition: getComputedStyle(settings).position,
            beside: results.classList.contains("forge-assistant-docked-beside"),
            handleShown: getComputedStyle(handle).display !== "none",
            offsetLeft: results.offsetLeft, mobile: results.classList.contains("mobile"),
            generateIn: document.getElementById("txt2img_generate_box").parentElement.id,
            attribute: document.documentElement.getAttribute("data-mc-columns-swapped"),
            kept: localStorage.getItem("mc-columns-swapped:txt2img")};
}"""


def dock(page):
    """The settings column into the Forge Assistant's panel, as its settings
    button does it: the panel opened, the button pressed."""
    page.evaluate("forgeAssistant.shell.open(); forgeAssistant.shell.toggleDock()")
    settle(page, 4)


def undock(page):
    """Back into the page: the panel opened (a tap outside puts it away and
    leaves the column docked, out of sight), the button pressed again."""
    page.evaluate("forgeAssistant.shell.open(); forgeAssistant.shell.toggleDock(); forgeAssistant.shell.close()")
    settle(page, 4)


class TestWithTheAssistantsDock:
    """The Forge Assistant draws a tab's settings column in its panel
    (docs/22 §3.29 and §3.41): the column out of the row, fixed over the
    panel, the row given one track and Forge's divider hidden, and the
    results a pixel off the row's edge for Forge's mobile script
    (`forge-assistant-docked-beside`, decided by where the results were
    before the dock). Swapped, the results were on the left before the dock
    -- a pixel off the edge, by the swap's own margin -- so the dock finds
    them beside the settings, as unswapped."""

    def test_a_swapped_tab_docks_and_undocks_as_an_unswapped_one(self, browser):
        page = open_page(browser, assistant=True, touch=True)
        cdp = page.context.new_cdp_session(page)
        at_rest = layout(page)
        generate = at_rest["generateIn"]

        # Unswapped, docked: the results fill the row a pixel off its edge.
        dock(page)
        plain = page.evaluate(DOCKED)
        assert plain["docked"] and plain["settingsPosition"] == "fixed" and not plain["handleShown"], plain
        assert plain["beside"] and plain["left"] == 1 and abs(plain["width"] - (plain["rowWidth"] - 1)) <= 1, plain
        undock(page)
        assert in_order(layout(page))

        # Swapped, by a double click on the divider, with a width of the
        # user's own: the settings narrower than the results.
        x, y = handle_point(page)
        mouse_drag(page, x, y, -120)
        page.wait_for_timeout(450)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        swapped_at_rest = layout(page)
        assert swapped(swapped_at_rest) and swapped_at_rest["kept"] == "1", swapped_at_rest
        assert swapped_at_rest["generateIn"] == generate, swapped_at_rest

        # Docked: the results fill the row exactly as they did unswapped.
        dock(page)
        docked = page.evaluate(DOCKED)
        assert docked["docked"] and docked["settingsPosition"] == "fixed" and not docked["handleShown"], docked
        assert docked["beside"] and docked["offsetLeft"] >= 1, docked
        assert abs(docked["left"] - plain["left"]) <= 1 and abs(docked["width"] - plain["width"]) <= 1, (plain, docked)
        assert docked["generateIn"] == generate and not docked["mobile"], docked

        # Forge's mobile script, at every window resize: never a phone.
        for width in (1300, 1500, 1400):
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(150)
        settle(page, 4)
        resized = page.evaluate(DOCKED)
        assert resized["docked"] and resized["generateIn"] == generate and not resized["mobile"], resized

        # A double tap where the divider was -- a mouse's and a finger's -- with
        # the divider hidden by the dock: nothing to swap, the dock as it was.
        page.wait_for_timeout(600)
        page.mouse.dblclick(x, y)
        page.wait_for_timeout(600)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(120)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(450)
        tapped = page.evaluate(DOCKED)
        assert tapped["docked"] and tapped["settingsPosition"] == "fixed" and not tapped["handleShown"], tapped
        assert tapped["attribute"] == "txt2img" and tapped["kept"] == "1", "no swap while docked: " + json.dumps(tapped)
        assert abs(tapped["left"] - resized["left"]) <= 1 and abs(tapped["width"] - resized["width"]) <= 1, (resized, tapped)
        assert tapped["generateIn"] == generate and not tapped["mobile"], tapped

        # Undocked: the swapped layout back, each column as wide as before
        # (the window went round three widths and back; Forge scales the
        # first track by each change and rounds down).
        undock(page)
        back = layout(page)
        assert swapped(back) and back["kept"] == "1", back
        assert abs(back["settings"]["width"] - swapped_at_rest["settings"]["width"]) <= 4, (swapped_at_rest, back)
        assert abs(back["results"]["width"] - swapped_at_rest["results"]["width"]) <= 4, (swapped_at_rest, back)
        assert back["generateIn"] == generate and not back["mobile"], back
        assert page.evaluate("forgeAssistant.dock.isDocked('tab_txt2img')") is False
        page.close()


class TestWithTheAssistantsPanelOpen:
    """The user's rule for the Forge Assistant's panel: "the first tap outside
    of an open flyout menu to do nothing except close the flyout menu ... I
    have to tap twice to do something outside". A double click or a double
    tap on the divider whose first press landed while the panel was open is
    that closing tap and one plain tap: no swap, and no reset of Forge's
    either. The next double tap, the panel closed, swaps as usual."""

    @staticmethod
    def panel_open(page) -> bool:
        return page.evaluate("forgeAssistant.shell.state.panelOpen")

    @staticmethod
    def on_the_divider(page, x, y) -> bool:
        return page.evaluate("([x, y]) => document.elementFromPoint(x, y).classList.contains('resize-handle')", [x, y])

    def test_the_first_press_only_closes_the_panel_and_the_next_double_tap_swaps(self, browser):
        page = open_page(browser, assistant=True, touch=True)
        cdp = page.context.new_cdp_session(page)
        # Widths of the user's own, so that Forge's reset would show.
        x, y = handle_point(page, y_fraction=0)
        mouse_drag(page, x, y, -120)
        page.wait_for_timeout(450)
        before = layout(page)
        assert in_order(before) and before["template"].endswith("px 16px 1fr"), before
        x, y = handle_point(page, y_fraction=0)

        # A mouse's double click, the panel open: closed, and nothing else.
        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        assert self.panel_open(page) is True and self.on_the_divider(page, x, y)
        page.mouse.dblclick(x, y)
        page.wait_for_timeout(300)
        clicked = layout(page)
        assert self.panel_open(page) is False
        assert in_order(clicked) and clicked["attribute"] is None and clicked["kept"] is None, clicked
        assert clicked["template"] == before["template"], "Forge's reset did not run either: " + json.dumps(clicked)
        assert abs(clicked["settings"]["width"] - before["settings"]["width"]) <= 1, (before, clicked)
        assert not clicked["resizing"], clicked

        # A finger's double tap, the panel open again: the same, and no drag of
        # Forge's left running from the touch that closed it.
        page.wait_for_timeout(1100)
        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        assert self.panel_open(page) is True and self.on_the_divider(page, x, y)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(120)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(450)
        tapped = layout(page)
        assert self.panel_open(page) is False
        assert in_order(tapped) and tapped["attribute"] is None and tapped["kept"] is None, tapped
        assert tapped["template"] == before["template"], "Forge's reset did not run either: " + json.dumps(tapped)
        assert not tapped["resizing"], tapped

        # The panel closed: the next double click swaps, as usual, and a
        # finger's double tap puts them back.
        page.wait_for_timeout(1100)
        page.mouse.dblclick(x, y)
        settle(page)
        assert swapped(layout(page)), layout(page)
        page.wait_for_timeout(600)
        x, y = handle_point(page, y_fraction=0)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(120)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(450)
        assert in_order(layout(page)), layout(page)
        page.close()

    def test_a_tap_elsewhere_that_closed_the_panel_leaves_the_next_double_click_its_own(self, browser):
        """The closing tap is the pair's first click only where the pair is:
        the panel put away by a tap on the settings, then at once a double
        click on the divider -- both its clicks made with the panel closed --
        swaps."""
        page = open_page(browser, assistant=True)
        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        box = page.evaluate("(() => { const r = document.getElementById('txt2img_setting_3').getBoundingClientRect();"
                            " return [r.left + 20, r.top + r.height / 2]; })()")
        page.mouse.click(box[0], box[1])
        page.wait_for_timeout(100)
        assert self.panel_open(page) is False
        x, y = handle_point(page, y_fraction=0)
        page.mouse.dblclick(x, y)
        settle(page)
        assert swapped(layout(page)), layout(page)
        page.close()

    def test_with_tap_outside_to_close_off_the_panel_stays_open_and_a_double_tap_swaps(self, browser):
        """The assistant's own switch: off, a tap outside leaves the panel
        open and is the page's, so the double tap is a double tap."""
        page = open_page(browser, assistant=True, touch=True)
        cdp = page.context.new_cdp_session(page)
        page.evaluate("forgeAssistant.shell.setTapToClose(false); forgeAssistant.shell.open()")
        settle(page)
        x, y = handle_point(page, y_fraction=0)
        assert self.on_the_divider(page, x, y)
        page.mouse.dblclick(x, y)
        settle(page)
        assert self.panel_open(page) is True and swapped(layout(page))
        page.wait_for_timeout(600)
        x, y = handle_point(page, y_fraction=0)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(120)
        # The second touch swaps as it lands, before the mouse events the
        # browser makes of the pair afterwards (a dblclick among them).
        touch_event(cdp, "touchStart", x, y)
        landed = layout(page)
        touch_event(cdp, "touchEnd", x, y)
        page.wait_for_timeout(450)
        assert in_order(landed), landed
        assert self.panel_open(page) is True and in_order(layout(page)), layout(page)
        page.close()

    def test_swapped_a_press_the_assistant_takes_starts_no_drag(self, browser):
        """Swapped, the divider's drag is this extension's; a press the open
        panel takes starts none of it -- a finger's drag (no tap, so the panel
        stays open, as a mouse's drag there does nothing either), and a
        mouse's even where a browser hands the page the mousedown of a press
        the assistant took (Chromium does not); a finger's tap only closes
        the panel."""
        page = open_page(browser, assistant=True, touch=True)
        cdp = page.context.new_cdp_session(page)
        x, y = handle_point(page, y_fraction=0)
        page.mouse.dblclick(x, y)
        settle(page)
        before = layout(page)
        assert swapped(before), before
        page.wait_for_timeout(600)

        page.evaluate("forgeAssistant.shell.open()")
        settle(page)
        x, y = handle_point(page, y_fraction=0)
        finger_drag(page, cdp, x, y, 120)
        dragged = layout(page)
        assert self.panel_open(page) is True, "a drag is no tap"
        assert dragged["template"] == before["template"] and not dragged["resizing"], dragged

        page.evaluate("""([x, y]) => {
            const handle = document.elementFromPoint(x, y);
            handle.dispatchEvent(new MouseEvent("mousedown", {bubbles: true, cancelable: true, button: 0, clientX: x, clientY: y}));
            window.dispatchEvent(new MouseEvent("mousemove", {bubbles: true, button: 0, clientX: x + 120, clientY: y}));
        }""", [x, y])
        pressed = layout(page)
        page.evaluate("([x, y]) => window.dispatchEvent(new MouseEvent('mouseup', {bubbles: true, button: 0, clientX: x, clientY: y}))", [x + 120, y])
        assert pressed["template"] == before["template"] and not pressed["resizing"], pressed

        page.wait_for_timeout(450)
        finger_tap(cdp, x, y)
        page.wait_for_timeout(450)
        tapped = layout(page)
        assert self.panel_open(page) is False
        assert swapped(tapped) and tapped["template"] == before["template"] and not tapped["resizing"], tapped
        page.close()


# --------------------------------------------------------------------------- #
# Off                                                                          #
# --------------------------------------------------------------------------- #


class TestOff:

    def test_off_draws_forges_order_and_leaves_the_double_click_to_forge(self, browser):
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = open_page(browser, context=context)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        page.close()
        page = open_page(browser, {"model_chain_swap_columns": False}, context=context)
        info = layout(page)
        assert in_order(info) and info["attribute"] is None and info["kept"] == "1", info
        x, y = handle_point(page)
        mouse_drag(page, x, y, 120)
        assert layout(page)["template"].endswith("px 16px 1fr")
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        reset = layout(page)
        assert in_order(reset) and reset["template"] == "1fr 16px 1fr", "Forge's own double click: its first widths"
        page.close()
        # The setting as last read is the next first paint's.
        again = open_page(browser, context=context, loaded=False)
        assert again.evaluate("window.firstPaint") is None
        again.close()
        context.close()

    def test_turned_off_and_on_again_on_the_page(self, browser):
        page = open_page(browser)
        x, y = handle_point(page)
        page.mouse.dblclick(x, y)
        settle(page)
        page.evaluate("opts.model_chain_swap_columns = false; hooks.changed.forEach((fn) => fn())")
        settle(page)
        assert in_order(layout(page))
        page.evaluate("opts.model_chain_swap_columns = true; hooks.changed.forEach((fn) => fn())")
        settle(page)
        assert swapped(layout(page))
        page.close()
