"""The Voice Box tab's layout, measured by a real browser.

The Node harness in ``test_voice_box_js.py`` proves what the page script writes
-- the height in pixels, ``data-layout`` -- but it has no layout, so it cannot
say whether the stylesheet then fits four stages into that height, whether a
long list scrolls inside its stage or lengthens the page, or whether the
stacked stages snap. Those were the user's words ("nothing should render
taller than browser view and force a scroll of the entire page"; "swipe up to
get to next stage"), and only a browser answers them.

So this renders the tab the way Forge frames it -- a 160 px header above the
root and a 60 px footer below it, the root exactly as ``mc_voice_box_ui``
paints it -- with the real Voice Box section of ``style.css`` and the real
``javascript/voice_box.js``, every route answered from fixtures with many
samples, history entries and outputs, and measures:

    at 1440 x 900   the page does not scroll; the root is 900 - 160 - 60 px,
                    written in pixels on itself; the four stages sit side by
                    side inside the window; the sample library, the history
                    and the output lanes each scroll inside their stage; the
                    Render header is on screen and nothing covers it;
    at 390 x 844    the stages stack, each as tall and as wide as the column
                    they scroll in, which snaps to a stage's start; the page
                    again does not scroll;
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


def page_html() -> str:
    return PAGE % {"header": HEADER, "footer": FOOTER, "section": voice_box_section(),
                   "root": mc_voice_box_ui.root_markup(TOKEN)}


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
        else:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(answers.get(name, {"ok": True})))
    else:
        route.fulfill(status=404, body="")


def open_tab(browser, width: int, height: int, *, dark: bool = False):
    """A page at ``width`` x ``height`` with the tab booted, its lists drawn and fitted."""
    page = browser.new_page(viewport={"width": width, "height": height})
    answers, html, script = fixtures(), page_html(), SCRIPT.read_text(encoding="utf-8")
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
    return {
        innerWidth, innerHeight, scrollY,
        pageScroll: document.documentElement.scrollHeight,
        pageClient: document.documentElement.clientHeight,
        layout: root.getAttribute("data-layout"),
        inlineHeight: root.style.height,
        root: box(root),
        container: {box: box(container), client: container.clientHeight, clientWidth: container.clientWidth,
                    scrollTop: container.scrollTop, snap: getComputedStyle(container).scrollSnapType},
        stages: stages.map((stage) => Object.assign(box(stage), {key: stage.dataset.stage,
                                                                  top0: stage.offsetTop})),
        lists,
        render: {box: box(render), hit: !!hit && (hit === render || render.contains(hit)),
                 inStage: render.closest(".mc-voice-box-stage").dataset.stage},
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
        assert render["inStage"] == "outputs"
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
    def test_the_stages_stack_one_screen_each_and_the_page_does_not_scroll(self, browser):
        page = open_tab(browser, 390, 844)
        found = measure(page)
        page.close()

        assert found["layout"] == "stack"
        assert found["pageScroll"] <= found["innerHeight"] + 1, found
        assert found["inlineHeight"] == f"{844 - HEADER - FOOTER}px"
        container = found["container"]
        assert container["snap"].startswith("y") and "mandatory" in container["snap"]
        for stage in found["stages"]:
            assert _near(stage["height"], container["client"]), (stage, container)
            assert _near(stage["width"], container["clientWidth"]), (stage, container)
        tops = [stage["top"] for stage in found["stages"]]
        assert tops == sorted(tops) and len(set(round(top) for top in tops)) == 4
        assert [stage["key"] for stage in found["stages"]] == ["input", "prompt", "configuration",
                                                               "outputs"]

    def test_the_column_snaps_to_a_stages_start(self, browser):
        page = open_tab(browser, 390, 844)
        found = page.evaluate("""async () => {
            const container = document.querySelector("#mc-voice-box .mc-voice-box-stages");
            const starts = [...container.querySelectorAll(".mc-voice-box-stage")].map((stage) =>
                stage.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop);
            const settled = [];
            for (const aim of [starts[1] + 40, starts[3] - 30, starts[2] + 25]) {
                container.scrollTo({top: aim});
                await new Promise((done) => setTimeout(done, 250));
                settled.push(container.scrollTop);
            }
            return {starts, settled};
        }""")
        page.close()

        starts = found["starts"]
        assert [round(value) for value in found["settled"]] == [round(starts[1]), round(starts[3]),
                                                                round(starts[2])], found

    def test_a_list_inside_a_stage_still_scrolls_inside_it(self, browser):
        page = open_tab(browser, 390, 844)
        found = measure(page)
        page.close()

        for name in ("samples", "lanes"):
            entry = found["lists"][name]
            # The list is the scroller, not the stage around it.
            assert entry["overflowY"] == "auto", (name, entry)
            assert entry["scroll"] > entry["client"] > 40, (name, entry)
            assert entry["box"]["bottom"] <= entry["stage"]["bottom"] + 1, (name, entry)
            assert entry["clipped"] == 0, (name, entry)


class TestBelowTheFloor:
    def test_a_window_too_small_for_the_floor_still_reaches_everything_in_each_stage(self, browser):
        """Below 420 px the root keeps the floor and the page may scroll; a stage
        whose fixed parts and list floor no longer fit scrolls its body whole,
        and nothing in it spills past the stage where it could not be reached."""
        seen = {}
        for width, height in ((1440, 560), (390, 560)):
            page = open_tab(browser, width, height)
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
