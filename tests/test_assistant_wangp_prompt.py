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


def open_wangp(browser, watch=None, css=""):
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
    if css:
        html = html.replace("</head>", "<style>" + css + "</style></head>", 1)
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


# --------------------------------------------------------------------------- #
# Its lines, and its size (docs/22 §3.39)
# --------------------------------------------------------------------------- #

#: A theme with opinions about every text area on the page, and a page line
#: height set as a length for small text -- which `font: inherit` would bring
#: into the box at the panel's own size.
HOSTILE_LINES = ("body { line-height: 11px !important; }"
                 " textarea { white-space: normal !important; line-height: 11px !important; }")

LINES = """() => {
    const box = document.querySelector(".forge-assistant-wangp-prompt");
    const style = getComputedStyle(box);
    const one = box.value;
    // As short as it can be, so what it scrolls is what its text needs.
    box.style.setProperty("min-height", "0px", "important");
    box.style.setProperty("height", "0px", "important");
    const measure = (text) => { box.value = text; return box.scrollHeight; };
    const single = measure("a fox at dawn");
    const four = measure("a fox at dawn\\na crane in the reeds\\n\\na heron at dusk");
    box.value = one;
    box.style.removeProperty("min-height");
    box.style.removeProperty("height");
    return {white: style.whiteSpace, line: parseFloat(style.lineHeight), size: parseFloat(style.fontSize),
            single, four};
}"""


def test_wangps_lines_are_lines_in_the_box_and_do_not_touch(browser):
    """Reported: "in wan2gp prompt box i see new lines, but in the flyout there
    is no space at all between lines." WanGP's prompt is lines, each one a
    video: here each is its own line, with room between them, whatever a
    theme says about text areas or the page says about its line height."""
    page = open_wangp(browser, css=HOSTILE_LINES)
    try:
        page.evaluate("window.minipaintWanGP.page = 'a fox at dawn\\na crane in the reeds'")
        page.click(".forge-assistant-settings")
        page.wait_for_function("document.querySelector('.forge-assistant-wangp-prompt').value.includes('crane')")
        assert page.evaluate("document.querySelector('.forge-assistant-wangp-prompt').value") \
            == "a fox at dawn\na crane in the reeds"
        found = page.evaluate(LINES)
        assert found["white"] == "pre-wrap", found
        assert found["line"] >= 1.4 * found["size"], found
        assert found["four"] - found["single"] >= 3 * 1.4 * found["size"] - 2, (
            "three more lines, the blank one among them, each a line's height", found)
    finally:
        page.close()


def test_the_box_starts_tall_and_a_dragged_height_is_kept_for_the_next_page(browser):
    """Asked for: "i really like that i can resize the flyout menu for the text
    field ... i was worried at first when i opened the flyout prompt view for
    wan2gp and it was really small." It starts eight lines tall, and the
    height its corner is dragged to is the height it has from then on."""
    page = open_wangp(browser)
    try:
        page.click(".forge-assistant-settings")
        page.wait_for_function("document.querySelector('.forge-assistant-wangp-prompt').value")
        box = page.evaluate("(() => { const r = document.querySelector('.forge-assistant-wangp-prompt')"
                            ".getBoundingClientRect(); return [r.right, r.bottom, r.height]; })()")
        size = page.evaluate("parseFloat(getComputedStyle(document.querySelector('.forge-assistant-wangp-prompt')).fontSize)")
        assert box[2] >= 8 * 1.4 * size, "eight lines to start with"
        page.mouse.move(box[0] - 3, box[1] - 3)
        page.mouse.down()
        page.mouse.move(box[0] - 3, box[1] + 120, steps=6)
        page.mouse.up()
        settle(page)
        dragged = page.evaluate("document.querySelector('.forge-assistant-wangp-prompt').getBoundingClientRect().height")
        assert dragged > box[2] + 90, (box, dragged)
        kept = page.evaluate("Object.keys(localStorage).filter((k) => k.startsWith('forge-assistant-wangp-prompt-height'))"
                             ".map((k) => localStorage.getItem(k))")
        assert kept == [str(round(dragged)) + "px"], kept

        page.reload()
        page.wait_for_function("!!(window.forgeAssistant && window.forgeAssistant.shell)")
        page.evaluate(SWITCH, "tab_wangp")
        page.evaluate("forgeAssistant.shell.open()")
        page.wait_for_function("!document.querySelector('.forge-assistant-wangp').hidden"
                               " || !document.querySelector('.forge-assistant-settings').hidden")
        if page.evaluate("document.querySelector('.forge-assistant-wangp').hidden"):
            page.click(".forge-assistant-settings")
        settle(page)
        again = page.evaluate("document.querySelector('.forge-assistant-wangp-prompt').getBoundingClientRect().height")
        assert abs(again - dragged) <= 1, (dragged, again)
    finally:
        page.close()


# --------------------------------------------------------------------------- #
# Two presses: Normal, Focused, Super Focused (docs/22 §3.39)
# --------------------------------------------------------------------------- #

MODE = """() => {
    const root = document.getElementById("tab_wangp");
    const shell = forgeAssistant.shell;
    return {focus: root.classList.contains("forge-assistant-focus-root"),
            studio: root.classList.contains("forge-assistant-wangp-studio"),
            open: shell.state.panelOpen, panelHidden: document.getElementById("forge-assistant-panel").hidden,
            launcher: !document.getElementById("forge-assistant-launcher").hidden,
            bar: document.body.getAttribute("data-forge-assistant-focus-bar")};
}"""

GRIP = """() => { const g = document.querySelector(".forge-assistant-grip").getBoundingClientRect();
                  return [g.left + g.width / 2, g.top + g.height / 2]; }"""
LAUNCHER = """() => { const b = document.getElementById("forge-assistant-launcher").getBoundingClientRect();
                      return [b.left + b.width / 2, b.top + b.height / 2]; }"""


def test_two_presses_cycle_normal_focused_super_focused_and_back(browser):
    """On the WanGP tab, from an open panel: two presses on the header's
    space are Focused (the panel stays), two more are Super Focused -- the
    panel put away, the tab bar with it, the studio's class on the focused
    panel for Mini Paint's tab to hear -- and two on the launcher, the one
    thing left, are Normal again with the panel back."""
    page = open_wangp(browser)
    try:
        start = page.evaluate(MODE)
        assert start["open"] and not start["focus"] and not start["studio"], start
        page.mouse.dblclick(*page.evaluate(GRIP))
        settle(page)
        focused = page.evaluate(MODE)
        assert focused["focus"] and not focused["studio"] and focused["open"], focused
        page.wait_for_timeout(350)
        page.mouse.dblclick(*page.evaluate(GRIP))
        settle(page)
        superfocused = page.evaluate(MODE)
        assert superfocused["focus"] and superfocused["studio"], superfocused
        assert superfocused["open"] is False and superfocused["panelHidden"] and superfocused["launcher"], superfocused
        assert superfocused["bar"] == "hidden", "the tab bar goes with the panel"
        page.wait_for_timeout(350)
        page.mouse.dblclick(*page.evaluate(LAUNCHER))
        settle(page)
        normal = page.evaluate(MODE)
        assert not normal["focus"] and not normal["studio"], normal
        assert normal["open"] and not normal["panelHidden"], "the panel is back as it was"
    finally:
        page.close()


def test_super_focus_is_left_by_escape_with_everything_back(browser):
    page = open_wangp(browser)
    try:
        page.mouse.dblclick(*page.evaluate(GRIP))
        page.wait_for_timeout(350)
        page.mouse.dblclick(*page.evaluate(GRIP))
        settle(page)
        assert page.evaluate(MODE)["studio"] is True
        page.keyboard.press("Escape")
        settle(page)
        found = page.evaluate(MODE)
        assert not found["focus"] and not found["studio"] and found["open"], found
    finally:
        page.close()
