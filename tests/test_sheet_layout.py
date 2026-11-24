"""The pop-over every large editor opens (javascript/mc_sheet.js), for real in Chromium.

Asked of Mini Paint NEO on 2026-11-23 -- "a pop up that covers 90% of the
screen, leaving a hint of [what is] behind it around the edge ... a
comfortable typing view with focus on text field ... these should look the
same" -- and moved here the next day with the editors it draws ("those are
related LLM studio IMO"): the gear's editor on Forge's prompt boxes, the
writers' system prompt editor, the character's system prompt editor
(forge_assistant_system.js), and Mini Paint's Send to WanGP prompt editor,
whose Enhance callbacks are stood in for here in the shape Mini Paint sends.

The page is Forge Neo's prompt boxes inside Txt2Img's and Img2Img's settings
columns, with Forge's own hooks collected and run as Forge runs them, this
extension's whole stylesheet, and the instructions route answered by the
route's own Python (`mc_llm_instructions_api.answer`) against a store in a
temporary folder, behind the page's key as the real route is.
"""

from __future__ import annotations

import glob
import json
import os
import urllib.parse
from pathlib import Path

import pytest

import mc_llm_instructions as store
import mc_llm_instructions_api as route
from prompt_master.minimax import enhancer

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
STYLE = ROOT / "style.css"
ORIGIN = "http://forge-sheet.test"
KEY = "the-page-key"

HOST_CSS = """
:root { --background-fill-primary: #0b0f19; --background-fill-secondary: #111827;
        --border-color-primary: #374151; --body-text-color: #e5e7eb; --body-text-color-subdued: #9ca3af;
        --input-background-fill: #1f2937; --button-secondary-background-fill: #1f2937;
        --button-secondary-background-fill-hover: #374151; --color-accent: #ff7c00; }
*, ::before, ::after { box-sizing: border-box; }
html, body { margin: 0; }
body { background: #0b0f19; color: #e5e7eb; font: 14px sans-serif; }
.row { display: flex; flex-wrap: wrap; gap: 8px; width: 100%; position: relative; }
.column { display: flex; flex-direction: column; gap: 8px; position: relative; }
.block { position: relative; }
.block textarea { width: 100%; min-height: 72px; display: block; padding: 8px; }
.tabitem { padding: 8px; }
.spacer { height: 2400px; }
"""


def tab_html(tab: str, shown: bool) -> str:
    return f"""
  <div id="tab_{tab}" class="tabitem" role="tabpanel" style="display: {'block' if shown else 'none'}">
    <div class="row resize-handle-row">
      <div id="{tab}_settings" class="column" style="flex: 1 1 0; min-width: 320px">
        <div id="{tab}_prompt_container" class="column">
          <div id="{tab}_prompt_row" class="row">
            <div id="{tab}_prompt" class="block"><label><textarea aria-label="Prompt" placeholder="Prompt"></textarea></label></div>
          </div>
          <div id="{tab}_neg_prompt_row" class="row">
            <div id="{tab}_neg_prompt" class="block"><label><textarea aria-label="Negative prompt"></textarea></label></div>
          </div>
        </div>
        <div class="block">Sampler</div>
      </div>
      <div id="{tab}_results" class="column" style="flex: 1 1 0; min-width: 320px">
        <button id="{tab}_generate">Generate</button>
      </div>
    </div>
  </div>"""


def page_html(*, assistant: bool = False, key: str = KEY) -> str:
    helpers = "".join(f'<script src="/javascript/{path.name}"></script>'
                      for path in sorted((ROOT / "javascript").glob("forge_assistant*.js"))) if assistant else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{HOST_CSS}</style>
<link rel="stylesheet" href="/style.css">
</head><body class="dark">
<gradio-app><div class="gradio-container app">
<div id="tabs" class="tabs">
  <div class="tab-nav" role="tablist">
    <button role="tab" id="tab_txt2img-button" aria-controls="tab_txt2img" class="selected">txt2img</button>
    <button role="tab" id="tab_img2img-button" aria-controls="tab_img2img">img2img</button>
  </div>
  {tab_html("txt2img", True)}
  {tab_html("img2img", False)}
</div>
<div id="mc-llm-chat-conversation-key" class="hidden"><textarea>{key}</textarea></div>
<div class="spacer"></div>
</div></gradio-app>
<script>
function gradioApp() {{
    const elem = document.querySelector("gradio-app");
    elem.getElementById = function (id) {{ return document.getElementById(id); }};
    return elem;
}}
var opts = {{}};
const hooks = {{loaded: [], after: [], changed: [], tab: []}};
function onUiLoaded(fn) {{ hooks.loaded.push(fn); }}
function onAfterUiUpdate(fn) {{ hooks.after.push(fn); }}
function onOptionsChanged(fn) {{ hooks.changed.push(fn); }}
function onUiTabChange(fn) {{ hooks.tab.push(fn); }}
// Forge's own Generate shortcut (script.js): Ctrl+Enter heard on the document.
window.forgeGenerated = 0;
document.addEventListener("keydown", (e) => {{
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) window.forgeGenerated += 1;
}});
window.documentEscapes = 0;
document.addEventListener("keydown", (e) => {{ if (e.key === "Escape") window.documentEscapes += 1; }});
window.overlays = [];
document.addEventListener("minipaint:overlay", (e) => window.overlays.push(e.detail));
</script>
{helpers}
<script src="/javascript/mc_sheet.js"></script>
</body></html>"""


class Served:
    """The page, the stylesheet, the scripts, and the instructions route."""

    def __init__(self, html: str):
        self.html = html
        self.asked: list = []
        self.keys: list = []
        self.hold = False
        self.held: list = []

    def __call__(self, request_route):
        request = request_route.request
        path = urllib.parse.urlsplit(request.url).path
        if path == "/":
            request_route.fulfill(status=200, content_type="text/html", body=self.html)
        elif path == "/style.css":
            request_route.fulfill(status=200, content_type="text/css", body=STYLE.read_text(encoding="utf-8"))
        elif path.startswith("/javascript/") and (ROOT / path[1:]).is_file():
            request_route.fulfill(status=200, content_type="text/javascript",
                                  body=(ROOT / path[1:]).read_text(encoding="utf-8"))
        elif path == route.ROUTE:
            body = json.loads(request.post_data or "{}")
            self.asked.append(body.get("action"))
            self.keys.append(request.headers.get("x-mc-conversation-key"))
            if self.hold:
                self.held.append(request_route)
                return
            if request.headers.get("x-mc-conversation-key") != KEY:
                request_route.fulfill(status=401, content_type="application/json",
                                      body=json.dumps({"ok": False, "code": "auth_required",
                                                       "message": "Reload the page — this tab's key is from an earlier session."}))
                return
            status, payload = route.answer(body)
            request_route.fulfill(status=status, content_type="application/json", body=json.dumps(payload))
        else:
            request_route.fulfill(status=404, body="")


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


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch, host):
    import mc_llm_paths

    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(route, "availability",
                        lambda: {"state": "ready", "text": "LLM Studio writes these prompts with m.gguf."})
    return tmp_path


def open_page(browser, *, width=1200, height=800, assistant=False, key=KEY, touch=False, loaded=True):
    context = browser.new_context(viewport={"width": width, "height": height}, has_touch=touch)
    page = context.new_page()
    served = Served(page_html(assistant=assistant, key=key))
    page.route(ORIGIN + "/**", served)
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!window.mcSheet")
    if loaded:
        page.evaluate("hooks.loaded.forEach((fn) => fn())")
        if assistant:
            page.wait_for_function("!!(window.forgeAssistant && window.forgeAssistant.shell)")
    return page, served


def close_page(page, served):
    for held in served.held:
        try:
            held.abort()
        except Exception:
            pass
    page.context.close()


BOXES = """() => {
    const layers = Array.from(document.querySelectorAll('.mc-sheet-layer'));
    return {w: innerWidth, h: innerHeight, names: mcSheet.state().open,
            sheets: layers.map((layer) => {
                const r = layer.querySelector('.mc-sheet').getBoundingClientRect();
                return {left: r.left, top: r.top, width: r.width, height: r.height,
                        z: Number(getComputedStyle(layer).zIndex), role: layer.getAttribute('role'),
                        modal: layer.getAttribute('aria-modal'), display: getComputedStyle(layer).display};
            }),
            active: document.activeElement ? document.activeElement.className : ''};
}"""


def boxes(page):
    return page.evaluate(BOXES)


def type_into_prompt(page, text, tab="txt2img"):
    page.locator(f"#{tab}_prompt textarea").click()
    page.keyboard.type(text)


# Mini Paint's Send to WanGP editor, as its intercept route answers: a plan, a
# start with an id, a status read a step at a time, a cancel. The page's
# answers come from `window.writer`, which a test sets as it goes.
MINI_PAINT_PROMPT = """(text) => {
    window.writer = {plan: {ok: true, available: true, variant: "ref2va", mode: "image",
                            variant_label: "MiniMax H3 Ref2VA", described: ["the reference"]},
                     state: "running", prompt: "", started: [], statuses: 0, cancelled: [], done: []};
    window.promptSheet = mcSheet.prompt({
        name: "wangp-prompt", title: "Prompt", subtitle: "Send to WanGP", text,
        done: (value, info) => window.writer.done.push({value, info}),
        cancel: () => window.writer.done.push({cancelled: true}),
        enhance: {
            plan: () => Promise.resolve(window.writer.plan),
            start: (words) => { window.writer.started.push(words);
                                return Promise.resolve({ok: true, llm_id: "llm-1", variant_label: "MiniMax H3 Ref2VA",
                                                        described: ["the reference"], override: false}); },
            status: () => { window.writer.statuses += 1;
                            return Promise.resolve({ok: true, state: window.writer.state, prompt: window.writer.prompt,
                                                    stage: "Writing…"}); },
            cancel: (id) => { window.writer.cancelled.push(id); return Promise.resolve({ok: true}); },
        },
        systemPrompt: (plan) => mcSheet.systemPrompt({name: "system-prompt", variant: plan.variant, mode: plan.mode}),
    });
    return true;
}"""


# --------------------------------------------------------------------------- #
# The sheet
# --------------------------------------------------------------------------- #


class TestTheSheet:
    def test_it_covers_ninety_percent_of_the_window_in_its_middle(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("() => mcSheet.open({name: 'one', title: 'One'})")
            found = boxes(page)
            one = found["sheets"][0]
            assert found["names"] == ["one"]
            assert abs(one["width"] - found["w"] * 0.9) <= 2 and abs(one["height"] - found["h"] * 0.9) <= 2
            assert abs(one["left"] - found["w"] * 0.05) <= 2 and abs(one["top"] - found["h"] * 0.05) <= 2
            assert one["z"] == 2010, "a step over the dialog layer, over everything both extensions draw"
            assert one["role"] == "dialog" and one["modal"] == "true"
        finally:
            close_page(page, served)

    def test_the_page_shows_round_it_dimmed(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("() => mcSheet.open({name: 'one'})")
            dim = page.evaluate("""() => getComputedStyle(document.querySelector('.mc-sheet-backdrop')).backgroundColor""")
            assert dim in ("rgba(0, 0, 0, 0.5)", "color(srgb 0 0 0 / 0.5)"), dim
        finally:
            close_page(page, served)

    def test_a_second_sheet_stacks_a_little_smaller_and_escape_closes_the_top_one(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("() => { mcSheet.open({name: 'lower'}); mcSheet.open({name: 'upper'}); }")
            found = boxes(page)
            lower, upper = found["sheets"]
            assert found["names"] == ["lower", "upper"]
            assert upper["z"] > lower["z"]
            assert upper["width"] < lower["width"] and upper["height"] < lower["height"]
            assert upper["left"] > lower["left"] and upper["top"] > lower["top"]
            page.keyboard.press("Escape")
            assert boxes(page)["names"] == ["lower"]
            assert page.evaluate("documentEscapes") == 0, "Escape is the top sheet's, not the page's"
            page.mouse.click(4, 4)
            assert boxes(page)["names"] == [], "a tap on the dimmed edge is the top sheet's too"
        finally:
            close_page(page, served)

    def test_it_is_announced_as_a_sheet_and_the_page_is_told_when_it_goes(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("() => mcSheet.open({name: 'one'})")
            assert page.evaluate("document.documentElement.classList.contains('mc-sheet-open')")
            page.keyboard.press("Escape")
            assert page.evaluate("overlays") == [
                {"name": "sheet:one", "open": True, "modal": True, "sheet": True},
                {"name": "sheet:one", "open": False, "modal": True, "sheet": True}]
            assert not page.evaluate("document.documentElement.classList.contains('mc-sheet-open')")
        finally:
            close_page(page, served)

    def test_tab_goes_round_inside_the_top_sheet(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate(MINI_PAINT_PROMPT, "a lighthouse")
            seen = set()
            for _ in range(12):
                page.keyboard.press("Tab")
                seen.add(page.evaluate("!!document.activeElement.closest('.mc-sheet')"))
            assert seen == {True}
        finally:
            close_page(page, served)

    def test_it_follows_the_visual_viewport(self, browser):
        """A phone's keyboard makes the visual viewport shorter; the sheet's
        buttons stay above it (the character editor's old test, kept here)."""
        page, served = open_page(browser, width=412, height=915)
        try:
            page.evaluate("() => mcSheet.open({name: 'one'})")
            first = page.evaluate("() => document.querySelector('.mc-sheet-layer').style.height")
            page.set_viewport_size({"width": 412, "height": 480})
            page.wait_for_function("() => document.querySelector('.mc-sheet-layer').style.height === '480px'")
            foot = page.evaluate("() => document.querySelector('.mc-sheet-foot').getBoundingClientRect().bottom")
            assert first == "915px" and foot <= 480
        finally:
            close_page(page, served)

    def test_a_wheel_on_the_edge_does_not_scroll_the_page_under_it(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("() => mcSheet.open({name: 'one'})")
            page.mouse.move(6, 400)
            page.mouse.wheel(0, 600)
            page.wait_for_timeout(200)
            assert page.evaluate("scrollY") == 0
        finally:
            close_page(page, served)

    def test_focus_mode_leaves_it_on_screen(self, browser):
        """Focus mode takes out what sits beside the focused workspace unless
        it says it is a dialog (this stylesheet's path rule, loaded whole)."""
        page, served = open_page(browser)
        try:
            page.evaluate("""() => {
                document.body.classList.add('forge-assistant-focused', 'forge-assistant-focus-path');
                document.querySelector('gradio-app').classList.add('forge-assistant-focus-path');
                mcSheet.open({name: 'one'});
            }""")
            found = boxes(page)
            assert found["sheets"][0]["display"] != "none" and found["sheets"][0]["width"] > 0
            spare = page.evaluate("""() => { const node = document.createElement('div');
                document.querySelector('gradio-app').appendChild(node);
                return getComputedStyle(node).display; }""")
            assert spare == "none", "the rule is on this page: it takes out an ordinary block on the path"
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# The prompt editor
# --------------------------------------------------------------------------- #


class TestThePromptEditor:
    def test_the_cursor_is_in_the_text_at_its_end(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate(MINI_PAINT_PROMPT, "a lighthouse at dusk")
            caret = page.evaluate("""() => { const t = document.activeElement;
                return {cls: t.className, value: t.value, start: t.selectionStart, end: t.selectionEnd}; }""")
            assert "mc-sheet-text" in caret["cls"] and caret["value"] == "a lighthouse at dusk"
            assert caret["start"] == caret["end"] == len("a lighthouse at dusk")
        finally:
            close_page(page, served)

    @pytest.mark.parametrize("how", ["escape", "edge", "cross", "done", "ctrl-enter"])
    def test_every_way_out_but_cancel_keeps_what_was_typed(self, browser, how):
        page, served = open_page(browser)
        try:
            page.evaluate(MINI_PAINT_PROMPT, "a lighthouse")
            page.keyboard.type(", fog")
            if how == "escape":
                page.keyboard.press("Escape")
            elif how == "edge":
                page.mouse.click(4, 4)
            elif how == "cross":
                page.locator(".mc-sheet-close").click()
            elif how == "done":
                page.locator(".mc-sheet-done").click()
            else:
                page.keyboard.press("Control+Enter")
            done = page.evaluate("writer.done")
            assert done == [{"value": "a lighthouse, fog", "info": {"prewritten": None, "changed": True}}]
            assert boxes(page)["names"] == []
            assert page.evaluate("forgeGenerated") == 0, "Forge's Generate never hears Ctrl+Enter"
        finally:
            close_page(page, served)

    def test_cancel_puts_the_text_back(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate(MINI_PAINT_PROMPT, "a lighthouse")
            page.keyboard.type(" THROWN AWAY")
            page.locator(".mc-sheet-cancel").click()
            assert page.evaluate("writer.done") == [{"cancelled": True}]
        finally:
            close_page(page, served)

    def test_enhance_runs_there_and_then_and_its_result_can_be_used(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate(MINI_PAINT_PROMPT, "a lighthouse")
            page.wait_for_function("() => { const i = document.querySelector('.mc-sheet-info'); "
                                   "return i && !i.hidden && i.dataset.state === 'ready'; }")
            info = page.evaluate("document.querySelector('.mc-sheet-info').textContent")
            assert "MiniMax H3 Ref2VA" in info and "the reference" in info
            page.locator(".mc-sheet-enhance").click()
            page.wait_for_function("() => writer.statuses >= 1", timeout=10000)
            assert page.evaluate("writer.started") == ["a lighthouse"]
            page.evaluate("() => { writer.state = 'done'; writer.prompt = 'A lighthouse in rolling fog.'; }")
            page.wait_for_function("() => mcSheet.top().state().state === 'done'", timeout=10000)
            assert page.evaluate("document.querySelector('.mc-sheet-result-text').textContent") == \
                "A lighthouse in rolling fog."
            assert page.evaluate("document.querySelector('.mc-sheet-text').value") == "a lighthouse", \
                "the words typed stay in the box until the result is used"
            reads = page.evaluate("writer.statuses")
            page.wait_for_timeout(2600)
            assert page.evaluate("writer.statuses") == reads, "nothing more is asked once it is written"
            page.locator(".mc-sheet-use").click()
            page.locator(".mc-sheet-done").click()
            assert page.evaluate("writer.done") == [{"value": "A lighthouse in rolling fog.", "info": {
                "prewritten": {"typed": "a lighthouse", "written": "A lighthouse in rolling fog."}, "changed": True}}]
        finally:
            close_page(page, served)

    @pytest.mark.parametrize("how", ["escape", "from outside"])
    def test_closing_while_it_enhances_stops_the_writer(self, browser, how):
        """However it goes: Mini Paint closes every sheet its popup opened when
        the popup goes (closeAll), which is not the editor's own way out."""
        page, served = open_page(browser)
        try:
            page.evaluate(MINI_PAINT_PROMPT, "a lighthouse")
            page.locator(".mc-sheet-enhance").click()
            page.wait_for_function("() => mcSheet.top().state().id === 'llm-1'", timeout=10000)
            if how == "escape":
                page.keyboard.press("Escape")
            else:
                page.evaluate("() => mcSheet.closeAll('popup closed')")
            assert page.evaluate("writer.cancelled") == ["llm-1"]
            assert boxes(page)["names"] == []
        finally:
            close_page(page, served)

    def test_system_prompt_opens_the_set_the_plan_names_stacked_over_it(self, browser, data):
        page, served = open_page(browser)
        try:
            page.evaluate(MINI_PAINT_PROMPT, "a lighthouse")
            page.wait_for_function("() => mcSheet.top().state().plan")
            page.locator(".mc-sheet-system-open").click()
            page.wait_for_function("() => mcSheet.state().depth === 2 && mcSheet.top().state().view")
            view = page.evaluate("mcSheet.top().state().view")
            assert view["variant"] == "ref2va" and view["mode"] == "image"
            assert view["text"] == enhancer.instructions("ref2va", True)
            assert boxes(page)["names"] == ["wangp-prompt", "system-prompt"]
            page.keyboard.press("Escape")
            assert boxes(page)["names"] == ["wangp-prompt"], "the editor under it is still there"
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# The writers' system prompt editor
# --------------------------------------------------------------------------- #


def open_instructions(page, **options):
    page.evaluate("(o) => { window.sp = mcSheet.systemPrompt(o); }", dict({"name": "system-prompt"}, **options))
    page.wait_for_function("() => window.sp.state().view || document.querySelector('.mc-sheet-message:not([hidden])')")


class TestTheSystemPromptEditor:
    def test_it_opens_on_the_set_asked_for_with_its_default(self, browser):
        page, served = open_page(browser)
        try:
            open_instructions(page, variant="ltx25", mode="text")
            found = page.evaluate("""() => ({view: sp.state().view,
                sets: Array.from(document.querySelectorAll('.mc-sheet-set option')).map((o) => [o.value, o.textContent]),
                modes: Array.from(document.querySelectorAll('.mc-sheet-mode option')).map((o) => o.value),
                line: document.querySelector('.mc-sheet-availability').textContent,
                subtitle: document.querySelector('.mc-sheet-sub').textContent,
                key: null})""")
            assert found["view"]["variant"] == "ltx23" and found["view"]["source"] == "default"
            import mc_llm_ltx

            assert found["view"]["text"] == mc_llm_ltx.instructions(False)
            assert found["sets"] == [["fl2va", "MiniMax H3 FL2VA"], ["ref2va", "MiniMax H3 Ref2VA"],
                                     ["ltx23", "LTX 2.3 and 2.5"]]
            assert found["modes"] == ["text", "image"]
            assert found["line"] == "LLM Studio writes these prompts with m.gguf."
            assert found["subtitle"] == "LTX 2.3 and 2.5, without a picture"
            assert served.keys == [KEY], "the page's key, in its header"
        finally:
            close_page(page, served)

    def test_save_override_keeps_it_for_every_request(self, browser):
        page, served = open_page(browser)
        try:
            open_instructions(page, variant="fl2va", mode="image")
            page.evaluate("() => { const box = document.querySelector('.mc-sheet-text'); box.value = 'Write haiku.'; "
                          "box.dispatchEvent(new Event('input')); }")
            assert page.evaluate("document.querySelector('.mc-sheet-save').classList.contains('mc-sheet-unsaved')")
            page.keyboard.press("Control+Enter")
            page.wait_for_function("() => sp.state().view.source === 'override'")
            assert store.saved("fl2va", has_image=True) == "Write haiku."
            assert page.evaluate("forgeGenerated") == 0
            page.locator(".mc-sheet-restore").click()
            page.wait_for_function("() => sp.state().view.source === 'default'")
            assert store.saved("fl2va", has_image=True) is None
        finally:
            close_page(page, served)

    def test_blank_is_refused_before_anything_is_sent(self, browser):
        page, served = open_page(browser)
        try:
            open_instructions(page, variant="fl2va", mode="text")
            page.evaluate("() => { const box = document.querySelector('.mc-sheet-text'); box.value = '  '; "
                          "box.dispatchEvent(new Event('input')); }")
            page.locator(".mc-sheet-save").click()
            message = page.evaluate("document.querySelector('.mc-sheet-message').textContent")
            assert "cannot be blank" in message and served.asked == ["describe"]
        finally:
            close_page(page, served)

    @pytest.mark.parametrize("answer, closed, saved", [("keep", False, None), ("discard", True, None),
                                                       ("save-then", True, "Mine.")])
    def test_an_unsaved_change_is_asked_about_before_it_closes(self, browser, answer, closed, saved):
        page, served = open_page(browser)
        try:
            open_instructions(page, variant="ref2va", mode="text")
            page.evaluate("() => { const box = document.querySelector('.mc-sheet-text'); box.value = 'Mine.'; "
                          "box.dispatchEvent(new Event('input')); }")
            page.keyboard.press("Escape")
            question = page.evaluate("() => { const c = document.querySelector('.mc-sheet-confirm'); "
                                     "return c.hidden ? '' : c.textContent; }")
            assert question.startswith("This change to the system prompt is not saved.")
            page.locator(f".mc-sheet-{answer}").click()
            page.wait_for_timeout(300)
            assert (boxes(page)["names"] == []) is closed
            assert store.saved("ref2va", has_image=False) == saved
        finally:
            close_page(page, served)

    def test_escape_while_it_asks_is_keep_editing(self, browser):
        page, served = open_page(browser)
        try:
            open_instructions(page, variant="ref2va", mode="text")
            page.evaluate("() => { const box = document.querySelector('.mc-sheet-text'); box.value = 'Mine.'; "
                          "box.dispatchEvent(new Event('input')); }")
            page.keyboard.press("Escape")
            page.keyboard.press("Escape")
            assert boxes(page)["names"] == ["system-prompt"]
            assert page.evaluate("document.querySelector('.mc-sheet-confirm').hidden")
        finally:
            close_page(page, served)

    def test_moving_to_another_set_with_an_unsaved_change_asks_first(self, browser):
        page, served = open_page(browser)
        try:
            open_instructions(page, variant="fl2va", mode="text")
            page.evaluate("() => { const box = document.querySelector('.mc-sheet-text'); box.value = 'Mine.'; "
                          "box.dispatchEvent(new Event('input')); }")
            page.select_option(".mc-sheet-set", "ltx23")
            assert page.evaluate("sp.state().view.variant") == "fl2va", "still on the set being edited"
            assert not page.evaluate("document.querySelector('.mc-sheet-confirm').hidden")
            page.locator(".mc-sheet-discard").click()
            page.wait_for_function("() => sp.state().view.variant === 'ltx23'")
        finally:
            close_page(page, served)

    def test_it_opens_where_the_opener_chose_and_the_set_shown_last_after(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("""() => { window.sp = mcSheet.systemPrompt({name: 'system-prompt',
                choose: () => new Promise((done) => setTimeout(() => done({variant: 'ref2va', mode: 'image'}), 50))}); }""")
            page.wait_for_function("() => sp.state().view")
            assert page.evaluate("[sp.state().view.variant, sp.state().view.mode]") == ["ref2va", "image"]
            page.keyboard.press("Escape")
            open_instructions(page)
            assert page.evaluate("sp.state().view.variant") == "ref2va", "where this page left it"
            page.keyboard.press("Escape")
            page.evaluate("""() => { window.told = null; window.sp = mcSheet.systemPrompt({name: 'system-prompt',
                choose: (info) => { window.told = info; return Promise.resolve({}); }}); }""")
            page.wait_for_function("() => sp.state().view")
            assert page.evaluate("told") == {"fallback": "ref2va"}, "an opener is told the set shown last"
            assert page.evaluate("sp.state().view.variant") == "ref2va", "and an opener that names none gets it"
        finally:
            close_page(page, served)

    def test_an_openers_switch_is_drawn_and_set_through_the_opener(self, browser):
        page, served = open_page(browser)
        try:
            page.evaluate("""() => { window.flips = [];
                window.sp = mcSheet.systemPrompt({name: 'system-prompt', variant: 'fl2va', mode: 'text',
                    toggle: {label: 'Enhance prompts before they reach WanGP', on: false,
                             set: (flag) => { flips.push(flag); return Promise.resolve({ok: true, on: flag, status: 'On.'}); }}}); }""")
            page.wait_for_function("() => sp.state().view")
            assert page.evaluate("document.querySelector('.mc-sheet-switch-label').textContent") == \
                "Enhance prompts before they reach WanGP"
            page.locator(".mc-sheet-switch").click()
            page.wait_for_function("() => sp.state().switched === true")
            assert page.evaluate("flips") == [True]
            assert page.evaluate("document.querySelector('.mc-sheet-switch').getAttribute('aria-checked')") == "true"
        finally:
            close_page(page, served)

    def test_without_an_openers_switch_there_is_none(self, browser):
        page, served = open_page(browser)
        try:
            open_instructions(page, variant="fl2va", mode="text")
            assert page.evaluate("!document.querySelector('.mc-sheet-switch')")
            assert page.evaluate("sp.state().switched") is None
        finally:
            close_page(page, served)

    def test_a_stale_key_says_to_reload(self, browser):
        page, served = open_page(browser, key="a-key-from-before-a-restart")
        try:
            open_instructions(page, variant="fl2va", mode="text")
            message = page.evaluate("document.querySelector('.mc-sheet-message').textContent")
            assert "Reload the page" in message
            assert page.evaluate("sp.state().view") is None
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# The gear on Forge's prompt boxes
# --------------------------------------------------------------------------- #


GEAR = """(id) => {
    const host = document.getElementById(id);
    const gear = host && host.querySelector('.mc-prompt-gear');
    const area = host && host.querySelector('textarea');
    if (!gear || !area) return null;
    const g = gear.getBoundingClientRect(), a = area.getBoundingClientRect();
    return {tag: gear.tagName, text: gear.textContent, svg: !!gear.querySelector('svg'),
            label: gear.getAttribute('aria-label'), inside: g.left >= a.left && g.right <= a.right && g.top >= a.top,
            fromRight: a.right - g.right, fromTop: g.top - a.top, room: parseFloat(getComputedStyle(area).paddingRight)};
}"""


class TestTheGear:
    def test_every_one_of_forges_four_prompt_boxes_has_one_after_the_page_loads(self, browser):
        page, served = open_page(browser, loaded=False)
        try:
            assert page.evaluate("mcSheet.state().gears") == []
            page.evaluate("hooks.loaded.forEach((fn) => fn())")
            assert page.evaluate("mcSheet.state().gears") == [
                "txt2img_prompt", "txt2img_neg_prompt", "img2img_prompt", "img2img_neg_prompt"]
        finally:
            close_page(page, served)

    def test_it_is_drawn_in_the_boxs_top_right_corner_with_room_kept_under_it(self, browser):
        page, served = open_page(browser)
        try:
            gear = page.evaluate(GEAR, "txt2img_prompt")
            assert gear["tag"] == "BUTTON" and gear["text"] == "" and gear["svg"] is True
            assert gear["label"] == "Edit the prompt in a large view"
            assert gear["inside"] and gear["fromRight"] <= 24 and gear["fromTop"] <= 12, gear
            assert gear["room"] >= 36, "no word sits under it"
        finally:
            close_page(page, served)

    def test_after_forges_updates_it_writes_nothing(self, browser):
        """Forge runs every onAfterUiUpdate after any change to the page; a
        callback that changed the page each time would raise the next itself."""
        page, served = open_page(browser)
        try:
            changed = page.evaluate("""() => new Promise((done) => {
                let records = 0;
                const watch = new MutationObserver((list) => { records += list.length; });
                watch.observe(document.querySelector('gradio-app'), {childList: true, subtree: true, attributes: true});
                hooks.after.forEach((fn) => fn()); hooks.after.forEach((fn) => fn());
                requestAnimationFrame(() => { watch.disconnect(); done(records); });
            })""")
            assert changed == 0
        finally:
            close_page(page, served)

    def test_a_tap_in_the_box_types_and_the_gear_opens_the_editor_on_its_text(self, browser):
        page, served = open_page(browser)
        try:
            type_into_prompt(page, "a castle")
            assert boxes(page)["names"] == []
            page.locator("#txt2img_prompt .mc-prompt-gear").click()
            opened = page.evaluate("""() => ({names: mcSheet.state().open, value: document.activeElement.value,
                title: document.querySelector('.mc-sheet-title').textContent,
                sub: document.querySelector('.mc-sheet-sub').textContent,
                enhance: !!document.querySelector('.mc-sheet-enhance'),
                system: !!document.querySelector('.mc-sheet-system-open')})""")
            assert opened == {"names": ["forge-prompt"], "value": "a castle", "title": "Prompt", "sub": "txt2img",
                              "enhance": False, "system": False}
        finally:
            close_page(page, served)

    def test_done_writes_it_back_as_typing_would_and_ctrl_enter_is_the_editors(self, browser):
        page, served = open_page(browser)
        try:
            type_into_prompt(page, "a castle")
            page.evaluate("""() => { window.heard = 0;
                document.querySelector('#txt2img_prompt textarea').addEventListener('input', () => { window.heard += 1; }); }""")
            page.locator("#txt2img_prompt .mc-prompt-gear").click()
            page.keyboard.type(" at dusk")
            page.keyboard.press("Control+Enter")
            assert page.evaluate("document.querySelector('#txt2img_prompt textarea').value") == "a castle at dusk"
            assert page.evaluate("heard") >= 1
            assert page.evaluate("forgeGenerated") == 0
            page.locator("#txt2img_prompt .mc-prompt-gear").click()
            page.keyboard.type(" THROWN AWAY")
            page.locator(".mc-sheet-cancel").click()
            assert page.evaluate("document.querySelector('#txt2img_prompt textarea').value") == "a castle at dusk"
        finally:
            close_page(page, served)

    def test_the_negative_prompts_gear_opens_its_own_box(self, browser):
        page, served = open_page(browser)
        try:
            page.locator("#txt2img_neg_prompt .mc-prompt-gear").click()
            assert page.evaluate("document.querySelector('.mc-sheet-title').textContent") == "Negative prompt"
            page.keyboard.type("blurry")
            page.locator(".mc-sheet-done").click()
            assert page.evaluate("document.querySelector('#txt2img_neg_prompt textarea').value") == "blurry"
            assert page.evaluate("document.querySelector('#txt2img_prompt textarea').value") == ""
        finally:
            close_page(page, served)

    def test_it_keeps_clear_of_forges_token_counter(self, browser):
        page, served = open_page(browser)
        try:
            clear = page.evaluate("""() => {
                const host = document.getElementById('txt2img_prompt');
                const area = host.querySelector('textarea');
                const parent = host.parentElement;
                parent.style.position = 'relative';
                const counter = document.createElement('div');
                counter.id = 'txt2img_token_counter';
                counter.textContent = '0/75';
                parent.insertBefore(counter, host);
                const a = area.getBoundingClientRect(), p = parent.getBoundingClientRect();
                counter.style.cssText = 'position:absolute; z-index:100; padding:2px 10px; border:2px solid #999; line-height:16px; '
                    + 'top:' + (a.top - p.top - 10) + 'px; right:' + (p.right - a.right + 16) + 'px';
                const overlaps = () => { const g = host.querySelector('.mc-prompt-gear').getBoundingClientRect(),
                    c = counter.getBoundingClientRect();
                    return !(g.right <= c.left || g.left >= c.right || g.bottom <= c.top || g.top >= c.bottom); };
                const before = overlaps();
                mcSheet.placeGears();
                const after = overlaps();
                const g = host.querySelector('.mc-prompt-gear').getBoundingClientRect();
                return {before, after, inside: g.top >= a.top && g.right <= a.right};
            }""")
            assert clear == {"before": True, "after": False, "inside": True}
        finally:
            close_page(page, served)


class TestTheGearInTheAssistantsDock:
    """The settings column, prompt boxes and all, drawn in the Forge Assistant's
    panel: the gear is pressed inside the panel, and the panel must still be
    there when the editor closes (the assistant does not yield to a sheet)."""

    def test_the_panel_stays_open_under_the_editor_and_gets_the_text_back(self, browser):
        page, served = open_page(browser, assistant=True)
        try:
            page.evaluate("forgeAssistant.shell.open(); forgeAssistant.shell.toggleDock()")
            page.wait_for_function("() => document.getElementById('txt2img_settings').classList.contains('forge-assistant-docked-column')")
            gear = page.locator("#txt2img_prompt .mc-prompt-gear")
            gear.click()
            assert boxes(page)["names"] == ["forge-prompt"]
            assert page.evaluate("forgeAssistant.shell.state.panelOpen") is True
            page.keyboard.type("a harbour at dawn")
            page.mouse.click(4, 4)
            assert boxes(page)["names"] == []
            assert page.evaluate("forgeAssistant.shell.state.panelOpen") is True, \
                "a tap on the sheet's edge is the sheet's, not a tap outside the panel"
            assert page.evaluate("document.querySelector('#txt2img_prompt textarea').value") == "a harbour at dawn"
        finally:
            close_page(page, served)


# --------------------------------------------------------------------------- #
# The character's system prompt editor, on the same sheet
# --------------------------------------------------------------------------- #


CHARACTER_STORE = """() => {
    window.characterSaves = [];
    let stored = "";
    const built = "You are Ada. Stay in character as Ada.";
    const view = (source, text, message) => ({ok: true, character: "Ada", source, text, default: built, message});
    window.forgeAssistant = window.forgeAssistant || {};
    forgeAssistant.store = () => ({
        key: () => "the-page-key",
        systemPrompt: () => Promise.resolve(stored ? view("override", stored) : view("default", built)),
        saveSystemPrompt: (who, change) => { characterSaves.push(change);
            stored = change.restore ? "" : change.text.trim();
            return Promise.resolve(stored ? view("override", stored, "Override saved for Ada.")
                                          : view("default", built, "Back to the default for Ada.")); },
    });
}"""


class TestTheCharacterEditorLooksTheSame:
    def test_it_is_the_writers_editors_sheet(self, browser):
        page, served = open_page(browser, assistant=True)
        try:
            page.evaluate(CHARACTER_STORE)
            page.evaluate("forgeAssistant.systemEditor.open('Ada')")
            page.wait_for_function("() => document.querySelector('.mc-sheet-text').value.startsWith('You are Ada')")
            character = page.evaluate("""() => ({box: (() => { const r = document.querySelector('.mc-sheet').getBoundingClientRect();
                    return {width: r.width, height: r.height}; })(),
                title: document.querySelector('.mc-sheet-title').textContent,
                foot: Array.from(document.querySelectorAll('.mc-sheet-foot button')).map((b) => b.textContent),
                mono: document.querySelector('.mc-sheet-text').classList.contains('mc-sheet-mono'),
                focused: document.activeElement === document.querySelector('.mc-sheet-text')})""")
            page.keyboard.press("Escape")
            open_instructions(page, variant="fl2va", mode="text")
            writers = page.evaluate("""() => ({box: (() => { const r = document.querySelector('.mc-sheet').getBoundingClientRect();
                    return {width: r.width, height: r.height}; })(),
                foot: Array.from(document.querySelectorAll('.mc-sheet-foot button')).map((b) => b.textContent)})""")
            assert character["title"] == "System prompt — Ada"
            assert character["box"] == writers["box"]
            assert character["foot"] == writers["foot"] == ["Restore default", "Reload", "Close", "Save override"]
            assert character["mono"] is True and character["focused"] is True
        finally:
            close_page(page, served)

    def test_an_unsaved_change_is_asked_about_the_same_way(self, browser):
        page, served = open_page(browser, assistant=True)
        try:
            page.evaluate(CHARACTER_STORE)
            page.evaluate("forgeAssistant.systemEditor.open('Ada')")
            page.wait_for_function("() => document.querySelector('.mc-sheet-text').value.startsWith('You are Ada')")
            page.keyboard.type("Be brief. ")
            page.mouse.click(4, 4)
            question = page.evaluate("document.querySelector('.mc-sheet-confirm').textContent")
            assert question.startswith("Your change to Ada’s system prompt is not saved.")
            page.locator(".mc-sheet-save-then").click()
            page.wait_for_function("() => !document.querySelector('.mc-sheet-layer')")
            assert page.evaluate("characterSaves") == [{"text": "Be brief. You are Ada. Stay in character as Ada."}]
            assert page.evaluate("forgeAssistant.systemEditor.isOpen()") is False
        finally:
            close_page(page, served)
