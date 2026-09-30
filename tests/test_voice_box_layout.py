"""The Voice Box tab's layout, measured by a real browser.

The Node harness in ``test_voice_box_js.py`` proves what the page script writes
-- the height in pixels, ``data-layout`` -- but it has no layout, so it cannot
say whether the stylesheet then fits four stages into that height, whether a
long list scrolls inside its stage or lengthens the page, or whether a phone's
cards snap. Those were the user's words ("nothing should render taller than
browser view and force a scroll of the entire page"; "horizontal card
scrolling makes a lot of sense"), and only a browser answers them.

So this renders the tab the way Forge frames it -- a 160 px header above the
root and a 60 px footer below it, the root exactly as ``mc_voice_box_ui``
paints it -- with the real Voice Box section of ``style.css`` and the real
``javascript/voice_box.js``, every route answered from fixtures with many
samples, history entries and outputs, and measures:

    at 1440 x 900   the page does not scroll; the root is 900 - 160 - 60 px,
                    written in pixels on itself; the four stages sit side by
                    side inside the window and there is no stage bar; the
                    sample library, the history and the output lanes each
                    scroll inside their stage; Render is in the Configuration
                    stage's header, on screen, and nothing covers it; a
                    selected lane's transport is square icon buttons;
    at 390 x 844    the stages are cards side by side, each exactly as wide
                    and as tall as the row they scroll in, which snaps to a
                    card's start; the stage bar marks the card in view and a
                    press on it goes to a card; each card scrolls up and down
                    inside itself while the page does not, and nothing in it
                    is a scroll area of its own; the Configuration card's
                    header (Render and the status line) stays in view while
                    the card scrolls;
    resized         the height follows the window, and the layout its width.

It needs the ``playwright`` package and a Chromium, and skips without either
(the WebUI needs neither). Chromium is looked for where Playwright keeps it
(``PLAYWRIGHT_BROWSERS_PATH``, ``/opt/pw-browsers`` by default here); a
Playwright that pins another build is pointed at the ``chrome`` binary found
there.
"""

from __future__ import annotations

import glob
import json
import math
import os
import urllib.parse
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

import mc_voice_box_ui  # noqa: E402  (after the skip: nothing to import it for otherwise)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "voice_box.js"
STYLE = ROOT / "style.css"
ORIGIN = "http://voice-box.test"
PREFIX = mc_voice_box_ui.PREFIX
TOKEN = "LAYOUT-TOKEN"
HEADER, FOOTER = 160, 60

SAMPLES, HISTORY, FAVOURITES, OUTPUTS = 40, 60, 6, 40


def voice_box_section() -> str:
    """The stylesheet's Voice Box section, from its banner to the end of the file."""
    text = STYLE.read_text(encoding="utf-8")
    marker = "/* ==========================================================================\n   Voice Box\n"
    assert text.count(marker) == 1, "the Voice Box section's banner moved"
    return text[text.index(marker):]


def _identifier(kind: str, index: int) -> str:
    return f"{kind}{index:015x}"[-16:]


def _peaks(seed: int) -> list[float]:
    return [round(0.12 + 0.6 * abs(math.sin(index * 0.37 + seed)), 3) for index in range(240)]


def fixtures() -> dict:
    """Every route's answer, with lists long enough to scroll."""
    samples = [{"id": _identifier("a", index), "title": f"Voice {index + 1}", "seconds": 8.0 + index % 5,
                "rate": 24000, "peaks": _peaks(index), "source": "file", "created": 1 + index}
               for index in range(SAMPLES)]
    configuration = {"id": _identifier("c", 1), "name": "Default", "model_id": "vibevoice-7b",
                     "card_uuid": "GPU-a", "steps": 10, "cfg_scale": 1.3, "seed": None,
                     "max_new_tokens": None,
                     "speakers": {"1": samples[0]["id"], "2": samples[1]["id"]},
                     "created": 1, "updated": 1}
    pipeline = {"id": _identifier("p", 1), "name": "Trailer", "configuration_id": configuration["id"],
                "prompt": "Speaker 1: Welcome back to the show.\nSpeaker 2: Thanks for having me.",
                "outputs": [], "created": 1, "updated": 1}
    outputs = []
    for index in range(OUTPUTS):
        first, second = samples[index % SAMPLES], samples[(index + 7) % SAMPLES]
        speakers = [{"n": 1, "sample_id": first["id"], "title": first["title"]}]
        if index % 2 == 1:
            # Two speakers on every other render, the newest (the top lane) among them.
            speakers.append({"n": 2, "sample_id": second["id"], "title": second["title"]})
        outputs.append({
            "id": _identifier("o", index), "name": f"Trailer {index + 1}",
            "pipeline_id": pipeline["id"], "seconds": 6.5, "rate": 24000, "peaks": _peaks(100 + index),
            "loop": False, "created": 1_790_000_000 + index * 60, "format": "mp3",
            "infotext": pipeline["prompt"] + "\nSteps: 10, CFG scale: 1.3, Seed: 1234, Model: vibevoice-7b",
            "render": {"model_id": "vibevoice-7b", "card": "GPU-a", "seed": 1234, "seed_drawn": True,
                       "steps": 10, "cfg_scale": 1.3, "max_new_tokens": None, "speakers": speakers,
                       "prompt": pipeline["prompt"], "render_seconds": 9.0, "peak_bytes": 1,
                       "sections": 1,
                       "configuration": dict(configuration, seed=1234)}})
    history = [{"id": _identifier("h", index), "favourite": index < FAVOURITES,
                "text": f"Speaker 1: Line number {index + 1} of a script long enough to be cut.",
                "used": 1} for index in range(HISTORY)]
    status = {
        "ok": True,
        "engine": {"installed": True, "ready": True, "supported": True, "message": "Installed",
                   "label": "VibeVoice", "model_id": "vibevoice-7b", "model_label": "VibeVoice 7B",
                   "models": [{"id": "vibevoice-7b", "label": "VibeVoice 7B"}],
                   "download_bytes": 0, "parts": []},
        "progress": {"running": False, "text": "", "fraction": 0.0, "failed": False, "model": ""},
        "engine_settings": {"card_uuid": "GPU-a", "model_id": "vibevoice-7b", "steps": 10,
                            "cfg_scale": 1.3, "seed": None, "max_new_tokens": None,
                            "keep_warm": True},
        "settings": {"save_folder": "", "card_uuid": "GPU-a", "model_id": "vibevoice-7b",
                     "keep_warm": True},
        "cards": [{"uuid": "GPU-a", "key": "a", "index": 0, "name": "NVIDIA GeForce RTX 3090",
                   "image_card": False, "wangp_card": False}],
        "runtime": {"cards": {}, "last_error": ""},
        "turns": [],
        "jobs": [],
    }
    return {
        "/status": status,
        "/samples": {"ok": True, "samples": samples},
        "/prompts": {"ok": True, "history": history,
                     "favourites": [entry for entry in history if entry["favourite"]]},
        "/configurations": {"ok": True, "configurations": [configuration]},
        "/pipelines": {"ok": True, "pipelines": [pipeline]},
        "/outputs": {"ok": True, "outputs": outputs},
    }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Box</title>
<style>
/* What every Forge page has around the tab: Gradio's box sizing and a theme's
   variables. The colours are the fixture's, not the stylesheet's. */
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { background: #ffffff; color: #1f2937; font: 14px/1.4 system-ui, sans-serif;
       --body-text-color: #1f2937; --body-text-color-subdued: #6b7280;
       --border-color-primary: #d1d5db; --background-fill-primary: #ffffff;
       --background-fill-secondary: #f9fafb; --color-accent: #f97316;
       --button-primary-background-fill: #f97316; --button-primary-text-color: #ffffff;
       --button-secondary-background-fill: #f3f4f6; --input-background-fill: #ffffff;
       --error-text-color: #b91c1c; --text-sm: 0.875em; --text-xs: 0.8em; --text-lg: 1.15em; }
body.dark { background: #0b0f19; color: #e5e7eb;
            --body-text-color: #e5e7eb; --body-text-color-subdued: #9ca3af;
            --border-color-primary: #374151; --background-fill-primary: #111827;
            --background-fill-secondary: #1f2937; --button-secondary-background-fill: #374151;
            --input-background-fill: #111827; }
#forge-header { background: #e5e7eb; height: %(header)dpx; }
#forge-footer { background: #e5e7eb; height: %(footer)dpx; }
body.dark #forge-header, body.dark #forge-footer { background: #1f2937; }
</style>
<style>%(section)s</style>
</head><body>
<div id="forge-header">Forge header</div>
<div id="tab"><div class="prose">%(root)s</div></div>
<div id="forge-footer">Forge footer</div>
<script src="/voice_box.js"></script>
</body></html>
"""


# A theme that does not play fair, as a second variant of the page: its rules
# come after the Voice Box section, as a theme's do. "display" gives every
# button, div and span a display of its own, !important -- which beats the
# browser's own `[hidden] { display: none }`; "inputs" squeezes every input to
# a sliver, the way the user's theme drew the Sampling checkbox.
HOSTILE = {
    "display": ("button { display: inline-flex !important; }\n"
                "div { display: inline-flex !important; }\n"
                "span { display: inline-flex !important; }\n"),
    "inputs": ("input { -webkit-appearance: none !important; appearance: none !important;"
               " width: 2px !important; }\n"),
}

# LobeTheme's icon option (its `replaceIcon`), as it runs once the page is
# built: every <span> whose text holds "×" gets its whole content replaced by
# a 36 px X of Lobe's own. It is what put a large X beside "Ready".
LOBE_ICON_SWAP = """() => {
    let swapped = 0;
    for (const span of document.querySelectorAll("span")) {
        if (!span.textContent || span.textContent.indexOf("\u00d7") === -1) continue;
        span.innerHTML = '<svg class="lobe-x" width="36" height="36" viewBox="0 0 24 24">'
            + '<path d="M18 6 6 18M6 6l12 12" stroke="currentColor" stroke-width="2"/></svg>';
        swapped += 1;
    }
    return swapped;
}"""


def page_html(hostile=()) -> str:
    html = PAGE % {"header": HEADER, "footer": FOOTER, "section": voice_box_section(),
                   "root": mc_voice_box_ui.root_markup(TOKEN)}
    if hostile:
        rules = "".join(HOSTILE[name] for name in hostile)
        html = html.replace("</head>", "<style>/* a hostile theme */\n%s</style>\n</head>" % rules, 1)
        # Outside the tab, what the theme does where nothing stops it: a hidden
        # button it shows, and a span with a cross for Lobe's swap to take.
        html = html.replace('<div id="forge-header">Forge header</div>',
                            '<div id="forge-header">Forge header <button class="probe" hidden>probe</button>'
                            ' <span class="probe" hidden>probe</span> <div class="probe" hidden>probe</div>'
                            ' <span id="probe-cross">\u00d7</span></div>', 1)
    return html


def _launch(playwright):
    """Chromium: Playwright's own build, or the one found where browsers are kept."""
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


def _answer(route, answers: dict, html: str, script: str) -> None:
    path = urllib.parse.urlsplit(route.request.url).path
    if path == "/":
        route.fulfill(status=200, content_type="text/html", body=html)
    elif path == "/voice_box.js":
        route.fulfill(status=200, content_type="text/javascript", body=script)
    elif path.startswith(PREFIX + "/"):
        name = path[len(PREFIX):]
        if name in ("/outputs/audio", "/samples/audio"):
            route.fulfill(status=200, content_type="audio/mpeg", body=b"\x00" * 64)
        elif name == "/settings":
            # As the server does: the setting is kept, and the answer carries them all.
            settings = answers["/status"]["settings"]
            settings.update(json.loads(route.request.post_data or "{}"))
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps({"ok": True, "settings": settings,
                                           "engine_settings": answers["/status"]["engine_settings"]}))
        else:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(answers.get(name, {"ok": True})))
    else:
        route.fulfill(status=404, body="")


def open_tab(browser, width: int, height: int, *, dark: bool = False, hostile=()):
    """A page at ``width`` x ``height`` with the tab booted, its lists drawn and fitted;
    ``hostile`` names the parts of HOSTILE the page's theme has."""
    page = browser.new_page(viewport={"width": width, "height": height})
    answers, html, script = fixtures(), page_html(hostile), SCRIPT.read_text(encoding="utf-8")
    if dark:
        html = html.replace("<body>", '<body class="dark">', 1)
    page.route(ORIGIN + "/**", lambda route: _answer(route, answers, html, script))
    page.goto(ORIGIN + "/")
    page.wait_for_function(
        "document.querySelectorAll('#mc-voice-box .mc-voice-box-lane').length === %d"
        " && document.querySelectorAll('#mc-voice-box .mc-voice-box-sample').length === %d"
        % (OUTPUTS, SAMPLES))
    settle(page)
    return page


def settle(page) -> None:
    """Two frames: the fit runs in one, and what it wrote is laid out by the next."""
    page.evaluate("new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))")


MEASURE = """() => {
    const root = document.getElementById("mc-voice-box");
    const box = (node) => { const r = node.getBoundingClientRect();
                            return {top: r.top, left: r.left, right: r.right, bottom: r.bottom,
                                    width: r.width, height: r.height}; };
    const stages = [...root.querySelectorAll(".mc-voice-box-stage")];
    const lists = {};
    for (const [name, selector] of [["samples", ".mc-voice-box-samples"],
                                    ["history", ".mc-voice-box-history-list"],
                                    ["favourites", ".mc-voice-box-favourites-list"],
                                    ["form", ".mc-voice-box-configuration-form"],
                                    ["lanes", ".mc-voice-box-lanes"]]) {
        const node = root.querySelector(selector);
        const stage = node.closest(".mc-voice-box-stage");
        // A row squashed to fit is a row whose own content is cut off.
        const clipped = [...node.children].filter((row) => row.scrollHeight > row.clientHeight + 1).length;
        lists[name] = {scroll: node.scrollHeight, client: node.clientHeight,
                       overflowY: getComputedStyle(node).overflowY,
                       box: box(node), stage: box(stage), rows: node.children.length, clipped};
    }
    const render = root.querySelector(".mc-voice-box-render");
    const r = render.getBoundingClientRect();
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    const container = root.querySelector(".mc-voice-box-stages");
    const rowStyle = getComputedStyle(container);
    const bar = root.querySelector(".mc-voice-box-stagebar");
    return {
        innerWidth, innerHeight, scrollY, scrollX,
        pageScroll: document.documentElement.scrollHeight,
        pageClient: document.documentElement.clientHeight,
        pageWide: document.documentElement.scrollWidth,
        layout: root.getAttribute("data-layout"),
        inlineHeight: root.style.height,
        root: box(root),
        head: box(root.querySelector(".mc-voice-box-head")),
        container: {box: box(container), client: container.clientHeight, clientWidth: container.clientWidth,
                    scrollTop: container.scrollTop, scrollLeft: container.scrollLeft,
                    snap: rowStyle.scrollSnapType, overflowX: rowStyle.overflowX,
                    overflowY: rowStyle.overflowY},
        stagebar: {display: getComputedStyle(bar).display, box: box(bar),
                   buttons: [...bar.querySelectorAll("button")].map((b) => ({
                       text: b.textContent, current: b.getAttribute("aria-current"),
                       fits: b.scrollWidth <= b.clientWidth + 1}))},
        stages: stages.map((stage) => {
            const body = stage.querySelector(".mc-voice-box-stage-body");
            const style = getComputedStyle(stage);
            return Object.assign(box(stage), {
                key: stage.dataset.stage, top0: stage.offsetTop,
                snapAlign: style.scrollSnapAlign, snapStop: style.scrollSnapStop,
                body: {overflowY: getComputedStyle(body).overflowY, scroll: body.scrollHeight,
                       client: body.clientHeight}});
        }),
        connectors: [...root.querySelectorAll(".mc-voice-box-connector")].map((c) => getComputedStyle(c).display),
        lists,
        render: {box: box(render), hit: !!hit && (hit === render || render.contains(hit)),
                 inStage: render.closest(".mc-voice-box-stage").dataset.stage,
                 inHead: !!render.closest(".mc-voice-box-stage-head")},
        renders: root.querySelectorAll(".mc-voice-box-render").length,
        line: {inStage: root.querySelector(".mc-voice-box-status-line").closest(".mc-voice-box-stage").dataset.stage,
               inHead: !!root.querySelector(".mc-voice-box-status-line").closest(".mc-voice-box-stage-head")},
        status: root.querySelector(".mc-voice-box-status").textContent,
    };
}"""


def measure(page) -> dict:
    return page.evaluate(MEASURE)


def _near(a: float, b: float, slack: float = 1.0) -> bool:
    return abs(a - b) <= slack


class TestDesktop:
    def test_the_tab_fits_the_window_and_the_page_does_not_scroll(self, browser):
        page = open_tab(browser, 1440, 900)
        found = measure(page)
        page.close()

        assert found["layout"] == "columns"
        assert found["pageScroll"] <= found["innerHeight"] + 1, found
        assert found["scrollY"] == 0
        assert found["inlineHeight"] == f"{900 - HEADER - FOOTER}px"
        assert _near(found["root"]["top"], HEADER)
        assert _near(found["root"]["height"], 900 - HEADER - FOOTER)
        assert found["root"]["bottom"] <= found["innerHeight"] - FOOTER + 1

    def test_the_four_stages_sit_side_by_side_inside_the_window(self, browser):
        page = open_tab(browser, 1440, 900)
        found = measure(page)
        page.close()

        stages = found["stages"]
        assert [stage["key"] for stage in stages] == ["input", "prompt", "configuration", "outputs"]
        assert all(_near(stage["top"], stages[0]["top"]) for stage in stages)
        assert all(stages[index]["left"] >= stages[index - 1]["right"] for index in range(1, 4))
        for stage in stages:
            assert stage["left"] >= 0 and stage["right"] <= found["innerWidth"], stage
            assert stage["bottom"] <= found["root"]["bottom"] + 1, stage
            assert stage["width"] > 200, stage
        # The stage bar is a phone's; side by side it takes no room at all.
        assert found["stagebar"]["display"] == "none"
        assert found["stagebar"]["box"]["height"] == 0
        assert set(found["connectors"]) != {"none"}

    def test_every_long_list_scrolls_inside_its_stage(self, browser):
        page = open_tab(browser, 1440, 900)
        found = measure(page)
        page.close()

        for name in ("samples", "history", "lanes"):
            entry = found["lists"][name]
            assert entry["client"] > 60, (name, entry)
            assert entry["scroll"] > entry["client"], (name, entry)
            assert entry["overflowY"] == "auto", (name, entry)
            # The list scrolls; its rows are never squashed to fit it instead.
            assert entry["rows"] > 10 and entry["clipped"] == 0, (name, entry)
        # Everything with a list in it, long or not, ends inside its stage.
        for name, entry in found["lists"].items():
            assert entry["box"]["bottom"] <= entry["stage"]["bottom"] + 1, (name, entry)
        assert found["lists"]["form"]["overflowY"] == "auto"

    def test_the_render_header_is_on_screen_and_nothing_covers_it(self, browser):
        page = open_tab(browser, 1440, 900)
        found = measure(page)
        page.close()

        render = found["render"]
        assert render["inStage"] == "configuration" and render["inHead"] is True
        assert found["line"] == {"inStage": "configuration", "inHead": True}
        assert found["renders"] == 1
        assert render["box"]["top"] >= found["root"]["top"]
        assert render["box"]["bottom"] <= found["innerHeight"]
        assert render["box"]["right"] <= found["innerWidth"]
        assert render["hit"] is True
        assert found["status"] == "Ready"

    def test_a_selected_lane_opens_inside_the_list_and_the_page_still_does_not_scroll(self, browser):
        page = open_tab(browser, 1440, 900)
        page.locator("#mc-voice-box .mc-voice-box-lane").nth(2).click()
        settle(page)
        found = page.evaluate("""() => {
            const list = document.querySelector("#mc-voice-box .mc-voice-box-lanes");
            const lane = list.querySelectorAll(".mc-voice-box-lane")[2];
            const l = lane.getBoundingClientRect(), b = list.getBoundingClientRect();
            return {expanded: lane.getAttribute("aria-expanded"),
                    details: getComputedStyle(lane.querySelector(".mc-voice-box-lane-details")).display,
                    inView: l.top >= b.top - 1 && Math.min(l.bottom, l.top + b.height) <= b.bottom + 1,
                    pageScroll: document.documentElement.scrollHeight, innerHeight};
        }""")
        page.close()

        assert found["expanded"] == "true"
        assert found["details"] != "none"
        assert found["inView"] is True
        assert found["pageScroll"] <= found["innerHeight"] + 1

    def test_a_selected_lanes_transport_is_square_icon_buttons_as_tall_as_the_others(self, browser):
        page = open_tab(browser, 1440, 900)
        page.locator("#mc-voice-box .mc-voice-box-lane").nth(0).click()
        settle(page)
        found = page.evaluate("""() => {
            const lane = document.querySelector("#mc-voice-box .mc-voice-box-lane");
            const size = (node) => { const r = node.getBoundingClientRect();
                                     return {width: r.width, height: r.height}; };
            return {icons: [...lane.querySelectorAll(".mc-voice-box-icon")].map((b) => Object.assign(size(b), {
                        label: b.getAttribute("aria-label"), svg: size(b.querySelector("svg")),
                        ink: getComputedStyle(b.querySelector("svg")).fill,
                        colour: getComputedStyle(b).color})),
                    trim: size(lane.querySelector(".mc-voice-box-lane-trim"))};
        }""")
        page.close()

        assert [icon["label"] for icon in found["icons"]] == ["Play", "Play from the start", "Stop", "Loop"]
        for icon in found["icons"]:
            assert _near(icon["width"], icon["height"]), icon
            assert _near(icon["height"], found["trim"]["height"]), (icon, found["trim"])
            assert (round(icon["svg"]["width"]), round(icon["svg"]["height"])) == (16, 16), icon
            # Drawn in the button's own colour.
            assert icon["ink"] == icon["colour"], icon

    def test_the_tint_is_an_edge_of_equal_segments_and_quieter_in_the_dark(self, browser):
        colours = {}
        for dark in (False, True):
            page = open_tab(browser, 1440, 900, dark=dark)
            colours[dark] = page.evaluate("""() => {
                const lane = document.querySelector("#mc-voice-box .mc-voice-box-lane");
                const segments = [...lane.querySelectorAll(".mc-voice-box-edge-segment")];
                const lane_box = lane.getBoundingClientRect();
                const sample = document.querySelector("#mc-voice-box .mc-voice-box-sample .mc-voice-box-edge");
                return {segments: segments.map((s) => {
                            const r = s.getBoundingClientRect();
                            return {width: r.width, height: r.height,
                                    colour: getComputedStyle(s).backgroundColor}; }),
                        lane: lane_box.height, sampleWidth: sample.getBoundingClientRect().width};
            }""")
            page.close()

        light, dark = colours[False], colours[True]
        assert len(light["segments"]) == 2
        assert all(_near(segment["width"], 4) for segment in light["segments"])
        assert _near(light["segments"][0]["height"], light["segments"][1]["height"])
        assert _near(sum(segment["height"] for segment in light["segments"]), light["lane"], 3)
        assert _near(light["sampleWidth"], 4)

        def lightness(colour: str) -> float:
            red, green, blue = (int(part) for part in colour[colour.index("(") + 1:].split(")")[0]
                                .split(",")[:3])
            return (max(red, green, blue) + min(red, green, blue)) / 510

        for segment in light["segments"] + dark["segments"]:
            assert segment["colour"] not in ("", "transparent", "rgba(0, 0, 0, 0)"), segment
        assert all(lightness(d["colour"]) < lightness(l["colour"]) - 0.15
                   for l, d in zip(light["segments"], dark["segments"]))


class TestPhone:
    def test_the_stages_are_cards_side_by_side_at_full_width_and_the_page_does_not_scroll(self, browser):
        page = open_tab(browser, 390, 844)
        found = measure(page)
        page.close()

        assert found["layout"] == "stack"
        assert found["pageScroll"] <= found["innerHeight"] + 1, found
        assert found["pageWide"] <= found["innerWidth"] + 1, found
        assert found["inlineHeight"] == f"{844 - HEADER - FOOTER}px"
        container = found["container"]
        assert container["snap"].startswith("x") and "mandatory" in container["snap"], container
        assert (container["overflowX"], container["overflowY"]) == ("auto", "hidden"), container
        stages = found["stages"]
        assert [stage["key"] for stage in stages] == ["input", "prompt", "configuration", "outputs"]
        for stage in stages:
            assert _near(stage["width"], container["clientWidth"]), (stage, container)
            assert _near(stage["height"], container["client"]), (stage, container)
            assert _near(stage["top"], stages[0]["top"]), stage
            assert (stage["snapAlign"], stage["snapStop"]) == ("start", "always"), stage
        # In pipeline order, left to right, the first filling the row's view
        # and the rest beyond its right edge until they are swiped in.
        assert _near(stages[0]["left"], container["box"]["left"])
        assert all(stages[index]["left"] >= stages[index - 1]["right"] for index in range(1, 4))
        assert all(stage["left"] >= container["box"]["right"] - 1 for stage in stages[1:])
        assert set(found["connectors"]) == {"none"}

    def test_the_stage_bar_sits_between_the_pipeline_bar_and_the_cards_and_its_words_fit(self, browser):
        seen = {}
        for width in (390, 360):
            page = open_tab(browser, width, 844)
            seen[width] = measure(page)
            page.close()

        for width, found in seen.items():
            bar = found["stagebar"]
            assert bar["display"] == "flex", (width, bar)
            assert bar["box"]["top"] >= found["head"]["bottom"] - 1, (width, bar, found["head"])
            assert bar["box"]["bottom"] <= found["container"]["box"]["top"] + 1, (width, bar)
            assert [button["text"] for button in bar["buttons"]] == ["Input", "Prompt", "Config", "Outputs"]
            assert all(button["fits"] for button in bar["buttons"]), (width, bar)
            assert [button["current"] for button in bar["buttons"]] == ["true", None, None, None]

    def test_the_row_snaps_to_a_cards_start(self, browser):
        page = open_tab(browser, 390, 844)
        found = page.evaluate("""async () => {
            const container = document.querySelector("#mc-voice-box .mc-voice-box-stages");
            const starts = [...container.querySelectorAll(".mc-voice-box-stage")].map((stage) =>
                stage.getBoundingClientRect().left - container.getBoundingClientRect().left + container.scrollLeft);
            const settled = [];
            for (const aim of [starts[1] + 40, starts[3] - 30, starts[2] + 25]) {
                container.scrollTo({left: aim});
                await new Promise((done) => setTimeout(done, 250));
                settled.push(container.scrollLeft);
            }
            return {starts, settled, scrollY};
        }""")
        page.close()

        starts = found["starts"]
        assert len({round(start) for start in starts}) == 4
        assert [round(value) for value in found["settled"]] == [round(starts[1]), round(starts[3]),
                                                                round(starts[2])], found
        assert found["scrollY"] == 0

    def test_the_stage_bar_marks_the_card_in_view_and_a_press_moves_to_it(self, browser):
        page = open_tab(browser, 390, 844)
        script = """async (index) => {
            const container = document.querySelector("#mc-voice-box .mc-voice-box-stages");
            const buttons = [...document.querySelectorAll("#mc-voice-box .mc-voice-box-stagebar button")];
            const cards = [...container.querySelectorAll(".mc-voice-box-stage")];
            const marked = () => buttons.filter((b) => b.getAttribute("aria-current") === "true")
                .map((b) => b.textContent);
            const frames = () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
            const wait = (ms) => new Promise((done) => setTimeout(done, ms));
            const start = (card) => card.getBoundingClientRect().left - container.getBoundingClientRect().left
                + container.scrollLeft;
            if (index === null) {
                // Swiped -- scrolled -- to the third card: the bar follows.
                container.scrollTo({left: start(cards[2])});
                await wait(250);
                await frames();
                return {marked: marked(), at: container.scrollLeft, want: start(cards[2])};
            }
            buttons[index].click();
            const at = container.scrollLeft;
            const want = start(cards[index]);
            for (let tries = 0; tries < 80 && Math.abs(container.scrollLeft - want) > 1; tries += 1) await wait(50);
            await frames();
            return {at, landed: container.scrollLeft, want, marked: marked(),
                    left: cards[index].getBoundingClientRect().left - container.getBoundingClientRect().left};
        }"""
        swiped = page.evaluate(script, None)
        pressed = page.evaluate(script, 3)
        page.emulate_media(reduced_motion="reduce")
        reduced = page.evaluate(script, 1)
        page.close()

        assert swiped["marked"] == ["Config"], swiped
        assert _near(pressed["landed"], pressed["want"]), pressed
        assert _near(pressed["left"], 0), pressed
        assert pressed["marked"] == ["Outputs"], pressed
        # Under reduced motion the press is a jump: there at once.
        assert _near(reduced["at"], reduced["want"]), reduced
        assert reduced["marked"] == ["Prompt"], reduced

    def test_a_stage_scrolls_up_and_down_inside_itself_while_the_page_does_not(self, browser):
        page = open_tab(browser, 390, 844)
        before = measure(page)
        page.mouse.move(195, 560)
        page.mouse.wheel(0, 700)
        page.wait_for_timeout(400)
        found = page.evaluate("""() => {
            const body = document.querySelector("#mc-voice-box .mc-voice-box-stage-input .mc-voice-box-stage-body");
            const container = document.querySelector("#mc-voice-box .mc-voice-box-stages");
            return {body: body.scrollTop, row: container.scrollLeft, rowTop: container.scrollTop, scrollY,
                    pageScroll: document.documentElement.scrollHeight, innerHeight};
        }""")
        page.close()

        # Every card is taller inside than the room it has, and scrolls itself.
        for stage in before["stages"]:
            assert stage["body"]["overflowY"] == "auto", stage
            assert stage["body"]["scroll"] > stage["body"]["client"] + 1, stage
        assert found["body"] > 100, found
        assert (found["row"], found["rowTop"], found["scrollY"]) == (0, 0, 0), found
        assert found["pageScroll"] <= found["innerHeight"] + 1, found

    def test_nothing_inside_a_stage_is_a_scroll_area_of_its_own(self, browser):
        """The sample library, the history and favourites, the configuration
        form and the lanes take the height they need; so do a selected lane's
        infotext and the script box, which grows with a long script."""
        page = open_tab(browser, 390, 844)
        page.evaluate("""() => {
            document.querySelector("#mc-voice-box .mc-voice-box-lane").click();
            const box = document.querySelector("#mc-voice-box .mc-voice-box-prompt");
            box.value = Array.from({length: 30}, (_, index) =>
                "Speaker " + (index % 2 + 1) + ": line " + (index + 1) + " of a long script.").join("\\n");
            box.dispatchEvent(new Event("input", {bubbles: true}));
        }""")
        settle(page)
        found = measure(page)
        extra = page.evaluate("""() => {
            const root = document.getElementById("mc-voice-box");
            const scrollers = [];
            for (const body of root.querySelectorAll(".mc-voice-box-stage-body")) {
                for (const node of body.querySelectorAll("*")) {
                    const style = getComputedStyle(node);
                    if (/(auto|scroll)/.test(style.overflowY) || /(auto|scroll)/.test(style.overflowX)) {
                        scrollers.push(node.className || node.tagName);
                    }
                }
            }
            const box = root.querySelector(".mc-voice-box-prompt");
            const infotext = root.querySelector(".mc-voice-box-lane[aria-expanded='true'] .mc-voice-box-infotext");
            return {scrollers,
                    prompt: {overflowY: getComputedStyle(box).overflowY, scroll: box.scrollHeight,
                             client: box.clientHeight, resize: getComputedStyle(box).resize},
                    infotext: {overflowY: getComputedStyle(infotext).overflowY, scroll: infotext.scrollHeight,
                               client: infotext.clientHeight}};
        }""")
        page.close()

        assert extra["scrollers"] == [], extra
        for name, entry in found["lists"].items():
            assert entry["overflowY"] == "visible", (name, entry)
            assert entry["scroll"] <= entry["client"] + 1, (name, entry)
            assert entry["clipped"] == 0, (name, entry)
        for name in ("samples", "history", "lanes"):
            assert found["lists"][name]["rows"] > 10, name
        prompt = extra["prompt"]
        assert prompt["overflowY"] == "hidden" and prompt["resize"] == "none", prompt
        assert prompt["scroll"] <= prompt["client"] + 1, prompt
        assert prompt["client"] > 300, prompt
        assert extra["infotext"]["overflowY"] == "visible", extra
        assert extra["infotext"]["scroll"] <= extra["infotext"]["client"] + 1, extra

    def test_a_lane_selected_below_the_fold_is_brought_into_view_by_its_card(self, browser):
        """The Outputs card's body is what scrolls on a phone, so that is what
        brings a selected lane into view; the lanes list, the row of cards and
        the page stay where they are."""
        page = open_tab(browser, 390, 844)
        found = page.evaluate("""async () => {
            const container = document.querySelector("#mc-voice-box .mc-voice-box-stages");
            const stage = container.querySelector(".mc-voice-box-stage-outputs");
            const body = stage.querySelector(".mc-voice-box-stage-body");
            container.scrollTo({left: stage.getBoundingClientRect().left - container.getBoundingClientRect().left
                                      + container.scrollLeft});
            await new Promise((done) => setTimeout(done, 250));
            const rowLeft = container.scrollLeft;
            const lane = stage.querySelectorAll(".mc-voice-box-lane")[30];
            const before = body.scrollTop;
            lane.click();
            await new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
            const l = lane.getBoundingClientRect(), b = body.getBoundingClientRect();
            return {before, after: body.scrollTop, expanded: lane.getAttribute("aria-expanded"),
                    inView: l.top >= b.top - 1 && l.bottom <= b.bottom + 1, height: l.height, room: b.height,
                    rowLeft, rowAfter: container.scrollLeft, scrollY,
                    list: stage.querySelector(".mc-voice-box-lanes").scrollTop};
        }""")
        page.close()

        assert found["expanded"] == "true"
        assert found["height"] < found["room"], found
        assert found["before"] == 0 and found["after"] > 0, found
        assert found["inView"] is True, found
        assert found["rowAfter"] == found["rowLeft"]
        assert (found["scrollY"], found["list"]) == (0, 0)

    def test_the_configuration_header_stays_in_view_while_its_stage_scrolls(self, browser):
        page = open_tab(browser, 390, 844)
        found = page.evaluate("""async () => {
            const container = document.querySelector("#mc-voice-box .mc-voice-box-stages");
            const stage = container.querySelector(".mc-voice-box-stage-configuration");
            const body = stage.querySelector(".mc-voice-box-stage-body");
            const head = stage.querySelector(".mc-voice-box-stage-head");
            const render = head.querySelector(".mc-voice-box-render");
            const wait = (ms) => new Promise((done) => setTimeout(done, ms));
            container.scrollTo({left: stage.getBoundingClientRect().left - container.getBoundingClientRect().left
                                      + container.scrollLeft});
            await wait(250);
            const look = () => {
                const r = render.getBoundingClientRect();
                const line = head.querySelector(".mc-voice-box-status-line").getBoundingClientRect();
                const s = stage.getBoundingClientRect();
                const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
                return {render: {top: r.top, bottom: r.bottom, left: r.left, right: r.right},
                        line: {top: line.top, bottom: line.bottom, height: line.height},
                        stage: {top: s.top, bottom: s.bottom, left: s.left, right: s.right},
                        hit: !!hit && (hit === render || render.contains(hit)),
                        bodyTop: body.getBoundingClientRect().top,
                        firstTop: body.firstElementChild.getBoundingClientRect().top,
                        scrollTop: body.scrollTop, scrolls: body.scrollHeight > body.clientHeight + 1};
            };
            const before = look();
            body.scrollTop = body.scrollHeight;
            await new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
            return {before, after: look(), status: head.querySelector(".mc-voice-box-status").textContent};
        }""")
        page.close()

        before, after = found["before"], found["after"]
        assert before["scrolls"] is True, before
        assert after["scrollTop"] > 0, after
        # The configuration went up under the header ...
        assert after["firstTop"] < after["bodyTop"] - 20, after
        # ... and the header -- Render and the status line -- stayed.
        for look in (before, after):
            assert look["hit"] is True, look
            assert look["render"]["top"] >= look["stage"]["top"], look
            assert look["render"]["bottom"] <= look["bodyTop"] + 1, look
            assert look["render"]["left"] >= look["stage"]["left"] - 1, look
            assert look["render"]["right"] <= look["stage"]["right"] + 1, look
            assert look["line"]["height"] > 10, look
            assert look["line"]["top"] >= look["stage"]["top"] and look["line"]["bottom"] <= look["bodyTop"] + 1, look
        assert _near(before["render"]["top"], after["render"]["top"])
        assert _near(before["line"]["top"], after["line"]["top"])
        assert found["status"] == "Ready"


class TestTheConfigurationManager:
    def test_it_is_one_row_at_every_width_the_stage_takes(self, browser):
        """The select, Save, Save as and Delete share one row -- in the desktop
        column and on the phone's card at 390 and 360 px -- and nothing in it
        runs out of the stage."""
        seen = {}
        for width, height in ((1440, 900), (390, 844), (360, 780)):
            page = open_tab(browser, width, height)
            seen[width] = page.evaluate("""() => {
                const stage = document.querySelector("#mc-voice-box .mc-voice-box-stage-configuration");
                const bar = stage.querySelector(".mc-voice-box-configuration-bar");
                const s = stage.getBoundingClientRect();
                return {controls: [...bar.children].map((node) => {
                            const r = node.getBoundingClientRect();
                            return {name: node.className.split(" ").pop(), top: r.top, left: r.left,
                                    right: r.right, width: r.width, height: r.height};
                        }),
                        stage: {left: s.left, right: s.right}, overflow: bar.scrollWidth - bar.clientWidth};
            }""")
            page.close()

        for width, found in seen.items():
            controls = found["controls"]
            assert [control["name"] for control in controls] == [
                "mc-voice-box-configuration-select", "mc-voice-box-configuration-save",
                "mc-voice-box-configuration-save-as", "mc-voice-box-configuration-delete"], width
            tops = [control["top"] for control in controls]
            assert max(tops) - min(tops) <= 2, (width, controls)
            for control in controls:
                assert control["left"] >= found["stage"]["left"] - 1, (width, control)
                assert control["right"] <= found["stage"]["right"] + 1, (width, control)
            assert controls[0]["width"] >= 60, (width, controls[0])
            for icon in controls[1:]:
                assert _near(icon["width"], icon["height"]), (width, icon)
            assert found["overflow"] <= 1, (width, found)

    def test_unsaved_changes_put_an_accent_dot_on_save(self, browser):
        page = open_tab(browser, 1440, 900)
        dot = """() => {
            const save = document.querySelector("#mc-voice-box .mc-voice-box-configuration-save");
            const after = getComputedStyle(save, "::after");
            return {content: after.content, background: after.backgroundColor, width: after.width,
                    label: save.getAttribute("aria-label")};
        }"""
        clean = page.evaluate(dot)
        page.locator('#mc-voice-box [data-field="steps"]').fill("20")
        settle(page)
        dirty = page.evaluate(dot)
        page.close()

        assert clean["content"] in ("none", "normal") and clean["label"] == "Save"
        assert dirty["content"] == '""' and dirty["width"] == "7px"
        assert dirty["background"] == "rgb(249, 115, 22)"
        assert dirty["label"] == "Save — unsaved changes"


class TestDeselecting:
    def test_a_tap_outside_puts_the_selected_lane_away_and_its_own_download_does_not(self, browser):
        """In a real browser, where a download's link is clicked by the page
        itself (a synthetic click, which is no tap) and Escape comes from the
        keyboard."""
        page = open_tab(browser, 1440, 900)
        opened = """() => [...document.querySelectorAll("#mc-voice-box .mc-voice-box-lane")]
            .map((lane) => lane.getAttribute("aria-expanded")).indexOf("true")"""
        lane = page.locator("#mc-voice-box .mc-voice-box-lane").nth(1)
        lane.locator(".mc-voice-box-lane-head").click()
        settle(page)
        selected = page.evaluate(opened)
        page.locator("#mc-voice-box .mc-voice-box-prompt-summary").click()
        settle(page)
        tapped = page.evaluate(opened)
        lane.locator(".mc-voice-box-lane-head").click()
        settle(page)
        with page.expect_download():
            lane.locator(".mc-voice-box-lane-download").click()
        settle(page)
        downloaded = page.evaluate(opened)
        # Escape said to a rename box gives up the rename and nothing more.
        lane.locator(".mc-voice-box-lane-name").dblclick()
        page.locator("#mc-voice-box .mc-voice-box-rename").press("Escape")
        settle(page)
        renaming = (page.evaluate(opened), page.locator("#mc-voice-box .mc-voice-box-rename").count())
        page.keyboard.press("Escape")
        settle(page)
        escaped = page.evaluate(opened)
        page.close()

        assert (selected, tapped, downloaded, renaming, escaped) == (1, -1, 1, (1, 0), -1)


class TestUnderAHostileTheme:
    """The same page with a theme that does not play fair (HOSTILE), and with
    LobeTheme's icon swap run over it once it is built (LOBE_ICON_SWAP)."""

    HIDDEN = [".mc-voice-box-status-dismiss", ".mc-voice-box-status-cancel", ".mc-voice-box-status-clear",
              ".mc-voice-box-status-unloads", ".mc-voice-box-install", ".mc-voice-box-trimmer",
              ".mc-voice-box-lane-details", ".mc-voice-box-stage-outputs .mc-voice-box-empty"]

    def test_nothing_hidden_shows_when_a_theme_gives_everything_a_display(self, browser):
        page = open_tab(browser, 1440, 900, hostile=("display",))
        found = page.evaluate("""(selectors) => {
            const root = document.getElementById("mc-voice-box");
            const boxes = {};
            for (const selector of selectors) {
                boxes[selector] = [...root.querySelectorAll(selector)].map((node) => {
                    const r = node.getBoundingClientRect();
                    return {hidden: node.hidden, parent: node.offsetParent === null,
                            width: r.width, height: r.height};
                });
            }
            // The theme's rules are in force: outside the tab, where nothing
            // stops them, a hidden button, span and div are all back.
            const probes = [...document.querySelectorAll("#forge-header .probe")].map((node) => {
                const r = node.getBoundingClientRect();
                return [node.tagName, node.hidden, r.width * r.height > 0];
            });
            return {boxes, probes, status: root.querySelector(".mc-voice-box-status").textContent};
        }""", self.HIDDEN)
        page.close()

        assert found["status"] == "Ready"
        assert found["probes"] == [["BUTTON", True, True], ["SPAN", True, True], ["DIV", True, True]]
        for selector, entries in found["boxes"].items():
            assert entries, selector
            for entry in entries:
                assert entry == {"hidden": True, "parent": True, "width": 0, "height": 0}, (selector, entry)

    def test_lobes_icon_swap_finds_nothing_to_take_in_the_tab(self, browser):
        """Run over the whole page, the swap takes the probe's cross outside
        the tab and nothing inside it: the status line keeps Cancel, Clear
        queue, Dismiss and the Unload holder, and no X of Lobe's sits there."""
        page = open_tab(browser, 1440, 900, hostile=("display",))
        before = page.evaluate("() => document.getElementById('mc-voice-box').innerHTML")
        swapped = page.evaluate(LOBE_ICON_SWAP)
        found = page.evaluate("""() => {
            const line = document.querySelector("#mc-voice-box .mc-voice-box-status-line");
            const dismiss = line.querySelector(".mc-voice-box-status-dismiss");
            return {kept: [".mc-voice-box-status-cancel", ".mc-voice-box-status-clear",
                           ".mc-voice-box-status-dismiss", ".mc-voice-box-status-unloads"]
                        .map((selector) => !!line.querySelector(selector)),
                    lobe: document.querySelectorAll("#mc-voice-box .lobe-x").length,
                    probe: !!document.querySelector("#probe-cross .lobe-x"),
                    dismiss: dismiss.getBoundingClientRect().width,
                    after: document.getElementById("mc-voice-box").innerHTML};
        }""")
        page.close()

        assert swapped == 1 and found["probe"] is True
        assert found["after"] == before
        assert found["kept"] == [True, True, True, True]
        assert found["lobe"] == 0
        assert found["dismiss"] == 0

    def test_the_switches_are_controls_when_a_theme_squeezes_inputs(self, browser):
        """Each at least 32 x 24, its knob a real box that moves to the other
        side when pressed, its track's fill and its border changing with it;
        Space and Enter work it, and the focus shows."""
        page = open_tab(browser, 1440, 900, hostile=("inputs",))
        found = page.evaluate("""async () => {
            const root = document.getElementById("mc-voice-box");
            const wait = () => new Promise((done) => setTimeout(done, 150));
            const look = (node) => {
                const r = node.getBoundingClientRect();
                const knob = node.querySelector(".mc-voice-box-switch-thumb").getBoundingClientRect();
                const picture = node.querySelector(".mc-voice-box-switch").getBoundingClientRect();
                const words = node.querySelector(".mc-voice-box-switch-label").getBoundingClientRect();
                return {checked: node.getAttribute("aria-checked"), width: r.width, height: r.height,
                        knob: {left: knob.left - r.left, width: knob.width, height: knob.height},
                        // The picture has room of its own inside the button,
                        // and the words start after it.
                        picture: {width: picture.width, height: picture.height,
                                  inside: knob.left >= r.left && knob.right <= r.right
                                      && picture.left >= r.left && picture.right <= r.right,
                                  before: words.left >= picture.right - 0.5},
                        fill: getComputedStyle(node.querySelector(".mc-voice-box-switch-track")).fill,
                        border: getComputedStyle(node).borderTopColor};
            };
            const out = {squeezed: root.querySelector('[data-field="seed"]').getBoundingClientRect().width,
                         switches: {}};
            for (const [name, selector] of [["sampling", '[data-field="sampling"]'],
                                            ["warm", ".mc-voice-box-keep-warm"]]) {
                const node = root.querySelector(selector);
                const first = look(node);
                node.click();
                await wait();
                const second = look(node);
                node.click();
                await wait();
                out.switches[name] = {first, second, third: look(node)};
            }
            return out;
        }""")
        sampling = page.locator('#mc-voice-box [data-field="sampling"]')
        sampling.focus()
        page.keyboard.press("Space")
        spaced = sampling.get_attribute("aria-checked")
        page.keyboard.press("Enter")
        entered = sampling.get_attribute("aria-checked")
        ring = sampling.evaluate("(node) => ({visible: node.matches(':focus-visible'),"
                                 " outline: getComputedStyle(node).outlineStyle})")
        page.close()

        assert found["squeezed"] < 20, found
        for name, states in found["switches"].items():
            first, second, third = states["first"], states["second"], states["third"]
            assert first["width"] >= 32 and first["height"] >= 24, (name, first)
            assert first["knob"]["width"] > 0 and first["knob"]["height"] > 0, (name, first)
            for state in (first, second):
                picture = state["picture"]
                assert picture["width"] >= 30 and picture["height"] >= 16, (name, state)
                assert picture["inside"] and picture["before"], (name, state)
            assert first["checked"] != second["checked"], (name, states)
            assert second["knob"]["left"] != first["knob"]["left"], (name, states)
            on, off = (second, first) if second["checked"] == "true" else (first, second)
            assert on["knob"]["left"] > off["knob"]["left"] + 4, (name, states)
            assert on["fill"] != off["fill"], (name, states)
            assert on["border"] != off["border"], (name, states)
            assert third == first, (name, states)
        assert (spaced, entered) == ("true", "false")
        assert ring == {"visible": True, "outline": "solid"}


class TestBelowTheFloor:
    def test_a_window_too_small_for_the_floor_still_reaches_everything_in_each_stage(self, browser):
        """Below 420 px the root keeps the floor and the page may scroll; a stage
        whose fixed parts and list floor no longer fit scrolls its body whole,
        and nothing in it spills past the stage where it could not be reached."""
        seen = {}
        for width, height in ((1440, 560), (390, 560)):
            page = open_tab(browser, width, height)
            # On the desktop the Prompt stage's fixed parts and list floor fit
            # the floor as they come, so the script box is made taller, as a
            # user dragging its corner makes it, until they no longer do.
            page.evaluate("""() => {
                const root = document.getElementById("mc-voice-box");
                if (root.dataset.layout !== "stack") {
                    root.querySelector(".mc-voice-box-prompt").style.height = "14em";
                }
            }""")
            settle(page)
            seen[width] = page.evaluate("""() => ({
                height: document.getElementById("mc-voice-box").style.height,
                stages: [...document.querySelectorAll("#mc-voice-box .mc-voice-box-stage")].map((stage) => {
                    const body = stage.querySelector(".mc-voice-box-stage-body");
                    const s = stage.getBoundingClientRect(), b = body.getBoundingClientRect();
                    return {key: stage.dataset.stage, inside: b.bottom <= s.bottom + 1,
                            scrolls: body.scrollHeight > body.clientHeight + 1,
                            overflowY: getComputedStyle(body).overflowY};
                })})""")
            page.close()

        for width, found in seen.items():
            assert found["height"] == "420px", (width, found)
            for stage in found["stages"]:
                assert stage["inside"], (width, stage)
                assert stage["overflowY"] == "auto", (width, stage)
            prompt = [stage for stage in found["stages"] if stage["key"] == "prompt"][0]
            assert prompt["scrolls"], (width, prompt)


class TestResizing:
    def test_the_height_follows_the_window_and_the_layout_its_width(self, browser):
        page = open_tab(browser, 1440, 900)
        seen = []
        for width, height in ((1440, 700), (1000, 760), (899, 760), (390, 844), (1440, 900)):
            page.set_viewport_size({"width": width, "height": height})
            settle(page)
            found = measure(page)
            seen.append((width, height, found["layout"], found["inlineHeight"],
                         found["pageScroll"] <= found["innerHeight"] + 1))
        # A window that leaves less than the floor gives the root the floor.
        page.set_viewport_size({"width": 1440, "height": 560})
        settle(page)
        floor = measure(page)
        page.close()

        assert seen == [(1440, 700, "columns", "480px", True),
                        (1000, 760, "columns", "540px", True),
                        (899, 760, "stack", "540px", True),
                        (390, 844, "stack", "624px", True),
                        (1440, 900, "columns", "680px", True)]
        assert floor["inlineHeight"] == "420px"
