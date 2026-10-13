"""The quieter night mode: one setting, one attribute, one stylesheet section.

The ask: "reduce bright spots in night mode, like the generate button, it is
super bright on white background". Under the Lobe theme's night mode the accent
is white, so Gradio's primary buttons -- Generate first, 44 pixels tall and the
width of its column -- are solid white blocks on a black page.

Three things hold it together, and nothing but the spelling links them:

* ``scripts/model_chain.py`` registers ``model_chain_quiet_night``, on by
  default;
* ``javascript/model_chain_night.js`` reads it off the host's ``opts`` and puts
  ``data-mc-night="quiet"`` on the root while it is on, and takes it off again
  while it is off -- live, so turning it off needs no reload;
* ``style.css`` draws primary buttons as a dim, accent-tinted surface wherever
  that attribute and ``.dark`` are both present, and nowhere else.

The script is run in node against a stub document, the way LLM Studio's footer
switch is. The stylesheet's rules are read as text for their contract, and then
rendered for real: Chromium computes what a Generate button looks like under a
Lobe-shaped night page with and without the attribute, which a stub cannot say.
A node stub has no cascade; a text check cannot see that Gradio's own rule
would have won.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_night.js"
STYLE = ROOT / "style.css"

OPTION = "model_chain_quiet_night"
ATTRIBUTE = "data-mc-night"
BANNER = "Model Chain -- a quieter night mode"


def section() -> str:
    """The stylesheet's night-mode section, banner to banner."""
    text = STYLE.read_text(encoding="utf-8")
    start = text.rfind("/* =====", 0, text.index(BANNER))
    end = text.index("/* =====", start + len(BANNER))
    return text[start:end]


def rules(css: str) -> list:
    """(selector, body) for every rule in ``css``, comments removed, @-blocks
    opened so their rules count like any other."""
    bare = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    bare = re.sub(r"@supports[^{]*\{", "", bare)
    found = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", bare):
        selector = " ".join(match.group(1).split())
        if selector:
            found.append((selector, match.group(2)))
    return found


# --------------------------------------------------------------------------- #
# The setting
# --------------------------------------------------------------------------- #


class TestTheSetting:
    def test_it_is_registered_and_defaults_on(self, host):
        """On by default because that was the ask; the script's fallback below
        has to match, or the buttons flash bright before the options load."""
        import model_chain

        registered = host.shared.options_templates

        assert model_chain.OPT_QUIET_NIGHT == OPTION
        assert OPTION in registered
        assert registered[OPTION].default is True

    def test_the_script_reads_it_by_the_name_python_registers(self):
        import model_chain

        assert f'"{model_chain.OPT_QUIET_NIGHT}"' in SCRIPT.read_text(encoding="utf-8")

    def test_the_stylesheet_is_keyed_on_what_the_script_writes(self):
        """Two files, one attribute, nothing but the spelling between them."""
        script = SCRIPT.read_text(encoding="utf-8")

        assert f'"{ATTRIBUTE}"' in script
        assert '"quiet"' in script
        assert f'[{ATTRIBUTE}="quiet"]' in section()


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #

HARNESS = """
const html = {
    attributes: {},
    writes: 0,
    getAttribute: (name) => (name in html.attributes ? html.attributes[name] : null),
    setAttribute(name, value) { html.attributes[name] = value; html.writes += 1; },
    removeAttribute(name) { delete html.attributes[name]; html.writes += 1; },
};
globalThis.document = {documentElement: html};
globalThis.window = globalThis;
__OPTS__

const loaded = [];
const changed = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onOptionsChanged = (fn) => changed.push(fn);
globalThis.onAfterUiUpdate = () => {};

SOURCE

const atOnce = html.getAttribute("data-mc-night");
loaded.forEach((fn) => fn());
const first = html.getAttribute("data-mc-night");
const settled = html.writes;
__AFTER__

console.log(JSON.stringify({
    atOnce,
    first,
    night: html.getAttribute("data-mc-night"),
    writes: html.writes,
    settled,
    hooks: {loaded: loaded.length, changed: changed.length},
}));
"""


def run(options="{}", after=""):
    """What the script put on the root, given a settings object (``None`` for
    a host that has not published one yet)."""
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    declared = f"globalThis.opts = {options};" if options is not None else ""
    harness = (HARNESS.replace("SOURCE", SCRIPT.read_text(encoding="utf-8"))
               .replace("__OPTS__", declared)
               .replace("__AFTER__", after))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTheScript:
    def test_it_is_on_by_default(self):
        assert run(f'{{"{OPTION}": true}}')["night"] == "quiet"

    def test_the_setting_turning_it_off_takes_the_attribute_off(self):
        assert run(f'{{"{OPTION}": false}}')["night"] is None

    def test_no_settings_at_all_still_turns_it_on(self):
        """``opts`` does not exist until the host has loaded the settings; the
        fallback is the option's default, so nothing flashes."""
        assert run(None)["night"] == "quiet"
        assert run("{}")["night"] == "quiet"

    def test_it_marks_the_root_before_the_page_is_up(self):
        """The first paint is already the quiet one: the attribute goes on as
        the script runs, not only once the host says the page is loaded."""
        assert run(f'{{"{OPTION}": true}}')["atOnce"] == "quiet"

    def test_it_writes_the_attribute_once_and_not_on_every_update(self):
        """Setting an attribute invalidates style whether or not the value
        moved, and this runs after every update the host makes."""
        settled = run(f'{{"{OPTION}": true}}',
                      after="loaded.forEach((fn) => fn());"
                            "changed.forEach((fn) => fn());"
                            "loaded.forEach((fn) => fn());")

        assert settled["settled"] == 1
        assert settled["writes"] == 1

    def test_turning_it_off_while_the_page_is_open_takes_effect(self):
        """No reload: the rule is CSS keyed on this attribute, the value is read
        live, and the host's own settings hook is what re-reads it."""
        found = run(f'{{"{OPTION}": true}}',
                    after=f"opts.{OPTION} = false;"
                          "changed.forEach((fn) => fn());")

        assert found["first"] == "quiet"
        assert found["night"] is None
        assert found["hooks"]["changed"] == 1

    def test_turning_it_back_on_puts_the_attribute_back(self):
        found = run(f'{{"{OPTION}": false}}',
                    after=f"opts.{OPTION} = true;"
                          "changed.forEach((fn) => fn());")

        assert found["first"] is None
        assert found["night"] == "quiet"

    def test_a_host_without_the_hooks_does_not_break_the_script(self):
        """Nothing here may throw: a page with a host that publishes no hook
        at all still gets the attribute, and the script's own apply() swallows
        a root it cannot reach."""
        if shutil.which("node") is None:
            pytest.skip("node is not installed")
        harness = ("globalThis.document = {documentElement: null};"
                   + SCRIPT.read_text(encoding="utf-8")
                   + "console.log('survived');")
        result = subprocess.run(["node", "--input-type=module", "-e", harness],
                                capture_output=True, text=True, timeout=30)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == "survived"


# --------------------------------------------------------------------------- #
# The stylesheet, as text
# --------------------------------------------------------------------------- #


class TestTheStylesheetContract:
    def test_every_rule_asks_for_the_attribute_and_the_dark_theme(self):
        """Inert without the setting, and never the light theme: Lobe's day
        mode has a black accent, and a black Generate is not a bright spot."""
        found = rules(section())

        assert found, "the section has rules"
        for selector, _ in found:
            for part in selector.split(","):
                assert f'[{ATTRIBUTE}="quiet"]' in part, part
                assert ".dark" in part, part

    def test_it_reaches_the_generate_button_and_the_rest_of_the_primaries(self):
        """Gradio's primary variant is the class `primary` on `button.gradio-button`
        -- Generate, Send, Apply settings, Mini Paint's Add to Queue -- and
        Lobe's own dialogs use antd's `.ant-btn-primary`."""
        selectors = " ".join(selector for selector, _ in rules(section()))

        assert "button.gradio-button.primary" in selectors
        assert ".ant-btn-primary:not(.ant-btn-dangerous)" in selectors
        assert "ul.options li.active:not(.selected)" in selectors

    def test_it_names_no_generated_class(self):
        """A `.svelte-` hash is regenerated on every Gradio build."""
        for selector, _ in rules(section()):
            assert ".svelte" not in selector, selector

    def test_it_states_no_colour_of_its_own(self):
        """Colours only ever come from the host's variables: a literal would be
        right for one theme and wrong for the next."""
        for selector, body in rules(section()):
            for declaration in body.split(";"):
                if ":" not in declaration:
                    continue
                value = declaration.split(":", 1)[1]
                assert not re.search(r"#[0-9a-fA-F]{3,8}\b", value), (selector, declaration)
                assert not re.search(r"\brgba?\(|\bhsla?\(", value), (selector, declaration)
                assert "var(--" in value, (selector, declaration)

    def test_a_solid_surface_is_declared_before_any_mix(self):
        """color-mix is recent; a browser without it must still get a dim
        button, which the theme's own secondary surface is."""
        css = re.sub(r"/\*.*?\*/", "", section(), flags=re.S)

        assert css.index("var(--button-secondary-background-fill)") < css.index("color-mix")
        mixes = [m.start() for m in re.finditer(r"color-mix", css)]
        assert mixes and all(css.rfind("@supports", 0, m) != -1 for m in mixes)

    def test_every_mix_lets_the_themes_accent_through(self):
        """What keeps a blue theme's Generate blue-tinted and a white one's grey."""
        for selector, body in rules(section()):
            for declaration in body.split(";"):
                if "color-mix" in declaration:
                    assert "var(--color-accent)" in declaration, (selector, declaration)
                    assert "transparent" in declaration, (selector, declaration)

    def test_it_sets_no_geometry(self):
        """Lobe fixes Generate at 44 px and the Forge Assistant measures that
        box; nothing here may move anything when the toggle does."""
        forbidden = {"height", "min-height", "max-height", "width", "min-width",
                     "max-width", "padding", "margin", "border-radius", "position",
                     "top", "left", "right", "bottom", "font-size", "display",
                     "border-width", "border", "line-height", "flex", "gap"}
        for selector, body in rules(section()):
            for declaration in body.split(";"):
                name = declaration.split(":", 1)[0].strip()
                assert name not in forbidden, (selector, declaration)
                assert not name.startswith(("padding-", "margin-", "border-radius")), \
                    (selector, declaration)

    def test_only_what_lobe_marks_important_is_marked_important_here(self):
        """A user stylesheet can still override the Generate rule the ordinary
        way; `!important` is said only where Lobe says it first."""
        for selector, body in rules(section()):
            if "!important" not in body:
                continue
            assert ".ant-btn-primary" in selector or "ul.options" in selector, selector

    def test_the_generate_rule_says_nothing_important(self):
        for selector, body in rules(section()):
            if "button.gradio-button.primary" in selector:
                assert "!important" not in body, selector

    def test_the_section_comes_after_the_progress_bars_solid_fill(self):
        """``tests/test_progress.py`` holds that the bar's solid fill is declared
        before the first color-mix in the file; this section mixes too, so it
        has to sit after that line."""
        text = STYLE.read_text(encoding="utf-8")

        assert text.index("background-color: var(--mc-progress-fill)") < text.index(BANNER)


# --------------------------------------------------------------------------- #
# The stylesheet, rendered
# --------------------------------------------------------------------------- #

ORIGIN = "http://forge-night.test"

# A Forge page the way Gradio 4.40 and the Lobe theme leave it at night: the
# dark class on the body, Lobe's variables (its accent is white in night mode,
# and it hands Gradio's primary-button variables that accent), Gradio's own
# primary-button rule with its generated class, and Lobe's yellow on a
# dropdown's active row. Then this extension's stylesheet, linked last, as
# Forge links it.
PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
:root, .dark {
    --color-accent: #ffffff;
    --body-text-color: rgba(255, 255, 255, 0.85);
    --body-background-fill: #000000;
    --button-primary-background-fill: #ffffff;
    --button-primary-background-fill-hover: #e6e6e6;
    --button-primary-border-color: #ffffff;
    --button-primary-text-color: #000000;
    --button-secondary-background-fill: rgba(255, 255, 255, 0.08);
    --button-secondary-background-fill-hover: rgba(255, 255, 255, 0.18);
    --button-secondary-border-color: #333333;
    --button-secondary-border-color-hover: #444444;
    --button-border-width: 1px;
}
body { margin: 0; background: var(--body-background-fill); color: var(--body-text-color); }
button.svelte-cmf5ev { display: inline-flex; border: var(--button-border-width) solid transparent; }
.primary.svelte-cmf5ev { border-color: var(--button-primary-border-color);
                         background: var(--button-primary-background-fill);
                         color: var(--button-primary-text-color); }
.primary.svelte-cmf5ev:hover { background: var(--button-primary-background-fill-hover); }
button[id$='_generate'] { height: 44px !important; min-height: 44px !important; }
.ant-btn-primary:not(.ant-btn-dangerous) { color: #000; background: #fff; }
ul.options li.active:not(.selected) { color: black !important; background: #fadb14 !important; }
</style>
<script>
    var opts = __OPTS__;
    var loaded = [], changed = [];
    function onUiLoaded(fn) { loaded.push(fn); }
    function onOptionsChanged(fn) { changed.push(fn); }
    function onAfterUiUpdate() {}
</script>
<script src="/javascript/model_chain_night.js"></script>
</head>
<body class="__BODY__">
<div id="txt2img_generate_box">
    <button id="txt2img_generate" class="lg primary gradio-button svelte-cmf5ev">Generate</button>
</div>
<button id="send" class="sm primary gradio-button svelte-cmf5ev">Send</button>
<button id="other" class="lg secondary gradio-button svelte-cmf5ev">Interrupt</button>
<button id="antd" class="ant-btn ant-btn-primary">Save</button>
<ul class="options"><li class="item active">Euler</li><li class="item selected">DPM++</li></ul>
<link rel="stylesheet" href="/style.css">
</body>
</html>"""


def _answer(route, html: str) -> None:
    path = urllib.parse.urlsplit(route.request.url).path
    if path == "/":
        route.fulfill(status=200, content_type="text/html", body=html)
    elif path == "/style.css":
        route.fulfill(status=200, content_type="text/css", body=STYLE.read_text(encoding="utf-8"))
    elif path == "/javascript/model_chain_night.js":
        route.fulfill(status=200, content_type="text/javascript",
                      body=SCRIPT.read_text(encoding="utf-8"))
    else:
        route.fulfill(status=404, body="")


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
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        found = _launch(playwright)
        if found is None:
            pytest.skip("no Chromium to render with")
        yield found
        found.close()


PAINT = """() => {
    const paint = (node) => {
        const style = getComputedStyle(node);
        return {background: style.backgroundColor, color: style.color,
                border: style.borderTopColor, height: node.getBoundingClientRect().height};
    };
    return {
        attribute: document.documentElement.getAttribute("data-mc-night"),
        generate: paint(document.getElementById("txt2img_generate")),
        send: paint(document.getElementById("send")),
        secondary: paint(document.getElementById("other")),
        antd: paint(document.getElementById("antd")),
        active: paint(document.querySelector("ul.options li.active")),
    };
}"""


def rgba(text: str) -> tuple:
    """(r, g, b, a) on 0-255 and 0-1 from a computed colour.

    Chromium answers `rgb(...)` and `rgba(...)` for the ordinary ones, and
    `color(srgb r g b / a)` with the channels on 0-1 for what color-mix made."""
    numbers = [float(n) for n in re.findall(r"[\d.]+", text.replace("srgb", ""))]
    if text.startswith("color("):
        numbers[:3] = [n * 255 for n in numbers[:3]]
    if len(numbers) == 3:
        numbers.append(1.0)
    return tuple(numbers)


def lit(text: str) -> float:
    """How bright a computed colour is over black, 0 to 1: the alpha times the
    mean of the channels, which is what a white at 14% looks like on a black
    page and all this needs to tell a block from a tint."""
    r, g, b, a = rgba(text)
    return a * (r + g + b) / (3 * 255)


def render(browser, options: str, body: str = "dark") -> dict:
    page = browser.new_page(viewport={"width": 1200, "height": 800})
    html = PAGE.replace("__OPTS__", options).replace("__BODY__", body)
    page.route(ORIGIN + "/**", lambda route: _answer(route, html))
    page.goto(ORIGIN + "/")
    page.wait_for_function("document.styleSheets.length >= 2")
    page.evaluate("loaded.forEach((fn) => fn())")
    found = page.evaluate(PAINT)
    page.close()
    return found


class TestWhatChromiumPaints:
    def test_at_rest_generate_is_a_solid_white_block_under_lobes_night_mode(self, browser):
        """The page as it was: with the setting off, Gradio's own rule wins and
        the button is the theme's white. This is the measurement the fix is
        against, and what the rest of this class would be hollow without."""
        found = render(browser, f'{{"{OPTION}": false}}')

        assert found["attribute"] is None
        assert lit(found["generate"]["background"]) > 0.95
        assert lit(found["generate"]["color"]) < 0.05

    def test_with_the_setting_generate_is_a_dim_tinted_surface(self, browser):
        """The accent let through at a low opacity: a trace of white over
        black, with the page's own text on it."""
        found = render(browser, f'{{"{OPTION}": true}}')

        assert found["attribute"] == "quiet"
        assert 0.05 < lit(found["generate"]["background"]) < 0.25
        assert lit(found["generate"]["color"]) > 0.7
        # The line around it is stronger than the surface and weaker than
        # the accent.
        assert lit(found["generate"]["background"]) < lit(found["generate"]["border"]) < 0.7

    def test_every_primary_button_and_lobes_own_go_the_same_way(self, browser):
        found = render(browser, f'{{"{OPTION}": true}}')

        assert 0.05 < lit(found["send"]["background"]) < 0.25
        assert 0.05 < lit(found["antd"]["background"]) < 0.25
        assert lit(found["antd"]["color"]) > 0.7

    def test_a_secondary_button_is_left_alone(self, browser):
        """Interrupt, Skip and every tool button keep the theme's own paint."""
        with_setting = render(browser, f'{{"{OPTION}": true}}')["secondary"]
        without = render(browser, f'{{"{OPTION}": false}}')["secondary"]

        assert with_setting == without

    def test_the_dropdowns_yellow_row_is_the_same_dim_surface(self, browser):
        """Lobe's yellow with black text, which said `!important`."""
        before = render(browser, f'{{"{OPTION}": false}}')["active"]
        after = render(browser, f'{{"{OPTION}": true}}')["active"]

        assert rgba(before["background"])[:3] == (250.0, 219.0, 20.0)
        assert lit(after["background"]) < 0.25
        assert lit(after["color"]) > 0.7

    def test_nothing_moves(self, browser):
        """Lobe's 44 px Generate, and the Forge Assistant measuring that box:
        the toggle changes paint and nothing else."""
        before = render(browser, f'{{"{OPTION}": false}}')
        after = render(browser, f'{{"{OPTION}": true}}')

        assert after["generate"]["height"] == before["generate"]["height"] == 44
        assert after["send"]["height"] == before["send"]["height"]

    def test_the_light_theme_is_untouched(self, browser):
        """No `.dark`, no rule: the day theme's black Generate stays as the
        theme drew it, attribute or no attribute."""
        found = render(browser, f'{{"{OPTION}": true}}', body="light")

        assert found["attribute"] == "quiet"
        assert lit(found["generate"]["background"]) > 0.95

    def test_turning_it_off_on_the_open_page_puts_the_white_back(self, browser):
        """The host's settings hook, with no reload between the two paints."""
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        html = PAGE.replace("__OPTS__", f'{{"{OPTION}": true}}').replace("__BODY__", "dark")
        page.route(ORIGIN + "/**", lambda route: _answer(route, html))
        page.goto(ORIGIN + "/")
        page.wait_for_function("document.styleSheets.length >= 2")
        page.evaluate("loaded.forEach((fn) => fn())")
        quiet = page.evaluate(PAINT)
        page.evaluate(f"opts.{OPTION} = false; changed.forEach((fn) => fn())")
        loud = page.evaluate(PAINT)
        page.close()

        assert lit(quiet["generate"]["background"]) < 0.25
        assert loud["attribute"] is None
        assert lit(loud["generate"]["background"]) > 0.95
