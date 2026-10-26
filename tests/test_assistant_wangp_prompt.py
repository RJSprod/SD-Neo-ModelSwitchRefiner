"""WanGP's prompt in the panel, in a real browser.

The panel's side of the WanGP tab's third state -- when it reads WanGP's page
and when it writes -- is test_assistant_prompt_js.py, under node. This is what
node cannot say: that on a Forge-shaped page with the real stylesheet the box
is drawn in the panel in the conversation's place, takes the keyboard and the
pointer (the docked Txt2Img view lets presses through the panel; this one must
not), that typing reaches Mini Paint NEO's `setPrompt()` through the real
event wiring, and that ▶ and Ctrl+Enter press WanGP's Generate only once what
was typed is there. Mini Paint NEO's bundle is a stand-in with its contract's
shape (its docs/wangp/CONTRACTS.md, bridge 1.14.0).
"""

from __future__ import annotations

from test_assistant_layout import ORIGIN, _answer, browser, page_html, settle  # noqa: F401

WANGP_TAB = """
    <button role="tab" id="tab_wangp-button" aria-controls="tab_wangp"
            aria-selected="false">WanGP</button>
  </div>"""

WANGP_PANEL = """
  <div id="tab_wangp" class="tabitem" role="tabpanel" style="display: none">
    <div id="wangp_iframe_root"><div class="frame-stand-in" style="height: 700px"></div></div>
  </div>
</div>
</div></gradio-app>"""

# Mini Paint NEO's bundle as its contract shapes it: the page's prompt read and
# written after a Gradio round trip's worth of delay, and WanGP's Generate
# noting what the page's prompt was when it was pressed.
STAND_IN = """<script>
window.wangpCalls = [];
window.minipaintWanGP = {
    page: "a fox at dawn", touched: 0,
    state() { return {ready: true, queue: true, prompt: {offered: true, touched: this.touched}}; },
    prompt() {
        wangpCalls.push(["prompt"]);
        return new Promise((done) => setTimeout(() => done(
            {ok: true, prompt: this.page, wizard: false, written: false}), 30));
    },
    setPrompt(text) {
        wangpCalls.push(["setPrompt", text]);
        return new Promise((done) => setTimeout(() => {
            this.page = text;
            done({ok: true, prompt: text, wizard: false, written: true});
        }, 60));
    },
    generate() {
        wangpCalls.push(["generate", this.page]);
        return Promise.resolve({ok: true, status: "started", route: "generate"});
    }
};
</script>
"""

SWITCH = """(to) => {
    for (const id of ["tab_txt2img", "tab_img2img", "tab_wangp"]) {
        const on = id === to;
        document.getElementById(id).style.display = on ? "block" : "none";
        const button = document.getElementById(id + "-button");
        button.classList.toggle("selected", on);
        button.setAttribute("aria-selected", String(on));
    }
}"""

DRAWN = """() => {
    const q = (selector) => document.querySelector(selector);
    const panel = q(".forge-assistant-panel");
    const box = q(".forge-assistant-wangp-prompt");
    const r = box.getBoundingClientRect();
    const p = panel.getBoundingClientRect();
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return {shown: !q(".forge-assistant-wangp").hidden && r.height > 40,
            inside: r.left >= p.left && r.right <= p.right && r.top >= p.top && r.bottom <= p.bottom,
            conversation: !q(".forge-assistant-conversation").hidden,
            hit: hit === box,
            surface: getComputedStyle(panel).backgroundColor,
            pointer: getComputedStyle(panel).pointerEvents,
            value: box.value,
            label: q(".forge-assistant-settings").getAttribute("aria-label"),
            pressed: q(".forge-assistant-settings").getAttribute("aria-pressed")};
}"""


def open_wangp(browser, watch=None):
    page = browser.new_page(viewport={"width": 1600, "height": 900})
    if watch is not None:
        # Every WebSocket, every event stream and every request still open,
        # from before the page loads.
        sockets, streams, live = watch
        page.on("websocket", lambda socket: sockets.append(socket.url))
        page.on("request", lambda request: (
            streams.append(request.url) if request.resource_type == "eventsource" else None,
            live.__setitem__(id(request), request.url)))
        page.on("requestfinished", lambda request: live.pop(id(request), None))
        page.on("requestfailed", lambda request: live.pop(id(request), None))
    html = page_html()
    html = html.replace("""aria-selected="false">Img2img</button>
  </div>""", """aria-selected="false">Img2img</button>""" + WANGP_TAB, 1)
    html = html.replace("""
</div>
</div></gradio-app>""", WANGP_PANEL, 1)
    html = html.replace('<script src="/forge_assistant', STAND_IN + '<script src="/forge_assistant', 1)
    assert 'id="tab_wangp"' in html and "minipaintWanGP" in html
    page.route(ORIGIN + "/**", lambda route: _answer(route, html))
    page.goto(ORIGIN + "/")
    page.wait_for_function("!!(window.forgeAssistant && window.forgeAssistant.shell)")
    page.evaluate(SWITCH, "tab_wangp")
    page.evaluate("forgeAssistant.shell.open()")
    page.wait_for_function("!document.querySelector('.forge-assistant-settings').hidden")
    settle(page)
    return page


def calls(page):
    return page.evaluate("wangpCalls.slice()")


def test_the_prompt_is_drawn_in_the_panel_and_takes_the_keyboard(browser):
    page = open_wangp(browser)
    try:
        page.click(".forge-assistant-settings")
        page.wait_for_function("document.querySelector('.forge-assistant-wangp-prompt').value")
        found = page.evaluate(DRAWN)
        assert found["shown"] and found["inside"], found
        assert found["conversation"] is False, "the box takes the conversation's place"
        assert found["hit"] is True, "a press on the box reaches the box"
        assert found["pointer"] != "none", "the panel is not see-through here, as docked it is"
        assert found["surface"] not in ("rgba(0, 0, 0, 0)", "transparent"), found
        assert found["value"] == "a fox at dawn"
        assert found["label"] == "WanGP prompt" and found["pressed"] == "true"

        page.click(".forge-assistant-wangp-prompt")
        page.keyboard.press("End")
        page.keyboard.type(" in the snow")
        page.wait_for_function("wangpCalls.some((c) => c[0] === 'setPrompt')")
        page.wait_for_timeout(200)
        written = [call for call in calls(page) if call[0] == "setPrompt"]
        assert written[-1] == ["setPrompt", "a fox at dawn in the snow"], written
        assert len(written) <= 2, "written once the keys went quiet, not once a key"
    finally:
        page.close()


def test_generate_right_after_typing_presses_wangps_generate_with_what_was_typed(browser):
    page = open_wangp(browser)
    try:
        page.click(".forge-assistant-settings")
        page.wait_for_function("document.querySelector('.forge-assistant-wangp-prompt').value")
        page.click(".forge-assistant-wangp-prompt")
        page.keyboard.press("Control+A")
        page.keyboard.type("a crane in the reeds")
        page.click(".forge-assistant-generate")
        page.wait_for_function("wangpCalls.some((c) => c[0] === 'generate')")
        found = [call for call in calls(page) if call[0] != "prompt"]
        assert found[-1] == ["generate", "a crane in the reeds"], found
        assert found.index(["setPrompt", "a crane in the reeds"]) < found.index(found[-1])

        page.click(".forge-assistant-wangp-prompt")
        page.keyboard.press("Control+A")
        page.keyboard.type("a heron at dusk")
        page.keyboard.press("Control+Enter")
        page.wait_for_function("wangpCalls.filter((c) => c[0] === 'generate').length === 2")
        assert calls(page)[-1] == ["generate", "a heron at dusk"]
        assert page.evaluate("document.querySelector('.forge-assistant-wangp-prompt').value") \
            == "a heron at dusk", "Ctrl+Enter is Generate, not a new line"
    finally:
        page.close()


def test_on_txt2img_the_same_button_docks_its_own_column_and_wangps_view_waits(browser):
    page = open_wangp(browser)
    try:
        page.click(".forge-assistant-settings")
        page.wait_for_function("document.querySelector('.forge-assistant-wangp-prompt').value")
        page.evaluate("window.minipaintWanGP.page = 'changed while Txt2Img showed'")
        page.evaluate(SWITCH, "tab_txt2img")
        page.wait_for_function("document.querySelector('.forge-assistant-wangp').hidden")
        assert page.get_attribute(".forge-assistant-settings", "aria-label") == "Generation settings"
        assert page.get_attribute(".forge-assistant-settings", "aria-pressed") == "false"
        assert page.evaluate("!document.querySelector('.forge-assistant-conversation').hidden")
        page.evaluate(SWITCH, "tab_wangp")
        page.wait_for_function("!document.querySelector('.forge-assistant-wangp').hidden")
        # Shown again, it reads WanGP's page again.
        page.wait_for_function("document.querySelector('.forge-assistant-wangp-prompt').value"
                               " === 'changed while Txt2Img showed'")
        assert [call for call in calls(page) if call[0] == "prompt"] == [["prompt"], ["prompt"]]
        assert page.evaluate("document.getElementById('txt2img_settings')"
                             ".classList.contains('forge-assistant-docked-column')") is False
    finally:
        page.close()



def test_no_live_connection_the_box_opens_no_stream_and_leaves_no_request_open(browser):
    """"Make sure there is no live stream. No live connections!" -- in the
    browser: from the page's load through showing the box, typing, ▶ and
    Ctrl+Enter, no WebSocket and no event stream is opened, and every request
    the page made has finished. (The page's own requests are the panel's
    ordinary ones -- the free-memory reading's plain request, the
    conversation's -- each with its deadline; the box adds none.)"""
    sockets, streams, live = [], [], {}
    page = open_wangp(browser, watch=(sockets, streams, live))
    try:
        page.click(".forge-assistant-settings")
        page.wait_for_function("document.querySelector('.forge-assistant-wangp-prompt').value")
        page.click(".forge-assistant-wangp-prompt")
        page.keyboard.press("Control+A")
        page.keyboard.type("a lynx in the pines")
        page.click(".forge-assistant-generate")
        page.wait_for_function("wangpCalls.some((c) => c[0] === 'generate')")
        page.click(".forge-assistant-wangp-prompt")
        page.keyboard.type(", at dusk")
        page.keyboard.press("Control+Enter")
        page.wait_for_function("wangpCalls.filter((c) => c[0] === 'generate').length === 2")
        page.wait_for_timeout(600)
        settle(page)
        assert sockets == [], sockets
        assert streams == [], streams
        assert list(live.values()) == [], "every request has finished: " + repr(list(live.values()))
    finally:
        page.close()
