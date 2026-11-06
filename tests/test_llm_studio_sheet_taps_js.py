"""A tap outside an LLM Studio sheet puts it away (javascript/llm_studio.js).

"I need the pop ups in LLM studio to close when i tap outside of them"
(2026-11-06). Run for real in Chromium, with a mouse and with a finger, on a
page shaped like LLM Studio's: the studio root, the shell bar, Conversation's
workspace with its header of openers and a transcript, the Threads screen open
over the workspace's left (as on a wide display) with its own ‹ Back, and the
Forge Assistant's panel fixed over the page. ‹ Back hides the sheet here, as its
Gradio handler does there.

What is held: a tap outside presses the sheet's own close and nothing else -- the
transcript under it gets no click and no focus, as the Forge Assistant's first
tap outside does nothing but put its panel away (docs/22 §3.45); a tap inside,
on an opener or in the assistant is left alone; a drag is no tap; with no sheet
open nothing is taken; and the tap after the one that closed lands.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "llm_studio.js"

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*, ::before, ::after { box-sizing: border-box; }
body { margin: 0; font: 14px sans-serif; }
.hide { display: none !important; }
#mc-llm-shellbar { height: 40px; }
#mc-llm-chat { position: relative; height: 560px; }
.mc-llm-chat-header { height: 44px; display: flex; gap: 8px; justify-content: flex-end; }
#transcript { height: 500px; overflow: auto; }
#transcript button { display: block; width: 100%; height: 60px; }
#mc-llm-chat-threads { position: absolute; top: 0; bottom: 0; left: 0; width: 300px;
                       background: #eee; z-index: 20; }
#assistant { position: fixed; right: 0; bottom: 0; width: 120px; height: 120px; z-index: 100; }
#assistant button { width: 100%; height: 100%; }
</style></head><body>
<gradio-app>
<div id="tabs"><div class="tab-nav"><button id="nav-other">Other tab</button></div>
<div id="tab_llm" class="tabitem">
  <div id="mc-llm-studio">
    <div id="mc-llm-shellbar"><button id="shell-menu">menu</button></div>
    <div id="mc-llm-chat" class="mc-llm-workspace mc-llm-chat-workspace">
      <div class="mc-llm-chat-header">
        <button id="to-threads" class="mc-llm-chip-button">Threads</button>
      </div>
      <div id="transcript"><button id="bubble">a message</button><div style="height: 1200px"></div></div>
      <div id="mc-llm-chat-threads" class="mc-llm-sheet mc-llm-sheet-screen">
        <div class="mc-llm-sheet-head"><button id="back" class="mc-llm-sheet-back">Back</button><h4>Threads</h4></div>
        <input id="search" placeholder="Find a thread">
      </div>
    </div>
  </div>
</div>
</div>
<div id="assistant" class="forge-assistant-panel"><button id="assistant-button">assistant</button></div>
</gradio-app>
<script>
window.log = [];
const sheet = document.getElementById("mc-llm-chat-threads");
document.getElementById("back").addEventListener("click", function () {
    window.log.push("back"); sheet.classList.add("hide");
});
document.getElementById("to-threads").addEventListener("click", function () {
    window.log.push("threads"); sheet.classList.toggle("hide");
});
["bubble", "shell-menu", "nav-other", "assistant-button"].forEach(function (id) {
    document.getElementById(id).addEventListener("click", function () { window.log.push(id); });
});
// What a press does before it is a click: a slider's drag, a canvas stroke, a
// Gradio control reacting as the finger goes down. Taken with the press.
document.getElementById("bubble").addEventListener("pointerdown", function () {
    window.log.push("bubble-down");
});
window.focused = function () { return document.activeElement ? document.activeElement.id : ""; };
</script>
<script src="/javascript/llm_studio.js"></script>
</body></html>"""

ORIGIN = "http://llm-studio-sheets.test"


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


def open_page(browser, touch=False, sheet_open=True):
    context = browser.new_context(viewport={"width": 1000, "height": 760}, has_touch=touch)
    page = context.new_page()

    def serve(route):
        path = route.request.url[len(ORIGIN):]
        if path == "/":
            route.fulfill(status=200, content_type="text/html", body=PAGE)
        elif path == "/javascript/llm_studio.js":
            route.fulfill(status=200, content_type="text/javascript",
                          body=SCRIPT.read_text(encoding="utf-8"))
        else:
            route.fulfill(status=404, body="")

    page.route(ORIGIN + "/**", serve)
    page.goto(ORIGIN + "/")
    if not sheet_open:
        page.evaluate("document.getElementById('mc-llm-chat-threads').classList.add('hide')")
    page.wait_for_timeout(50)
    return page


def state(page):
    return page.evaluate("""() => ({log: window.log.slice(), focused: window.focused(),
        open: !document.getElementById('mc-llm-chat-threads').classList.contains('hide')})""")


class TestATapOutsideClosesTheSheet:
    def test_a_click_beside_the_sheet_presses_its_back_and_nothing_under_it(self, browser):
        page = open_page(browser)
        try:
            page.mouse.click(600, 150)      # the transcript's message, beside the sheet
            page.wait_for_timeout(50)
            found = state(page)
            assert found["log"] == ["back"], found
            assert found["open"] is False and found["focused"] != "bubble", found
        finally:
            page.context.close()

    def test_a_finger_tap_beside_it_does_the_same(self, browser):
        page = open_page(browser, touch=True)
        try:
            page.touchscreen.tap(600, 150)
            page.wait_for_timeout(80)
            found = state(page)
            assert found["log"] == ["back"] and found["open"] is False, found
        finally:
            page.context.close()

    def test_on_a_phone_the_page_around_a_sheet_is_outside_too(self, browser):
        """Below 901 px a sheet covers the whole workspace: what is left to tap
        is the page around it, Forge's tab bar included."""
        page = open_page(browser)
        try:
            page.click("#nav-other")
            page.wait_for_timeout(50)
            found = state(page)
            assert found["log"] == ["back"], found
        finally:
            page.context.close()

    def test_the_tap_after_the_one_that_closed_lands(self, browser):
        page = open_page(browser)
        try:
            page.mouse.click(600, 150)
            page.wait_for_timeout(50)
            page.mouse.click(600, 150)
            page.wait_for_timeout(50)
            assert state(page)["log"] == ["back", "bubble-down", "bubble"], state(page)
        finally:
            page.context.close()


class TestWhatIsNotATapOutside:
    def test_a_tap_inside_the_sheet_is_the_sheets(self, browser):
        page = open_page(browser)
        try:
            page.click("#search")
            page.wait_for_timeout(50)
            found = state(page)
            assert found["log"] == [] and found["open"] is True and found["focused"] == "search", found
        finally:
            page.context.close()

    def test_an_opener_on_the_bar_is_left_to_its_own_handler(self, browser):
        page = open_page(browser)
        try:
            page.click("#to-threads")
            page.wait_for_timeout(50)
            assert state(page)["log"] == ["threads"], state(page)
        finally:
            page.context.close()

    def test_the_forge_assistant_is_exempt(self, browser):
        page = open_page(browser)
        try:
            page.click("#assistant-button")
            page.wait_for_timeout(50)
            found = state(page)
            assert found["log"] == ["assistant-button"] and found["open"] is True, found
        finally:
            page.context.close()

    def test_a_drag_is_no_tap(self, browser):
        page = open_page(browser)
        try:
            page.mouse.move(600, 150)
            page.mouse.down()
            page.mouse.move(660, 230, steps=6)
            page.mouse.up()
            page.wait_for_timeout(50)
            found = state(page)
            assert "back" not in found["log"] and found["open"] is True, found
        finally:
            page.context.close()

    def test_with_no_sheet_open_nothing_is_taken(self, browser):
        page = open_page(browser, sheet_open=False)
        try:
            page.mouse.click(600, 150)
            page.wait_for_timeout(50)
            found = state(page)
            assert found["log"] == ["bubble-down", "bubble"] and found["focused"] == "bubble", found
        finally:
            page.context.close()

    def test_a_tab_not_on_screen_takes_nothing(self, browser):
        page = open_page(browser)
        try:
            page.evaluate("document.getElementById('tab_llm').style.display = 'none'")
            page.click("#nav-other")
            page.wait_for_timeout(50)
            assert state(page)["log"] == ["nav-other"], state(page)
        finally:
            page.context.close()
