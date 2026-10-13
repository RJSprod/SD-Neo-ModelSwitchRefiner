"""The theme's page script, as far as node can hold it, and its stylesheet
section as text.

Node has no layout, so the header's fit, the drawer and the moves are checked
in Chromium (``tests/test_theme_layout.py``). What node holds is what is pure:
the tab-name spacing, the plan of which tabs fit, the emoji-to-icon match, the
settings' normalisation, the theme toggle's address, the first paint from the
last configuration, and the hooks the script asks the host for.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import mc_theme

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_theme.js"
STYLE = ROOT / "style.css"
BANNER = "Model Chain -- the theme"

HARNESS = """
const attributes = {};
const html = {
    attributes,
    getAttribute: (name) => (name in attributes ? attributes[name] : null),
    setAttribute(name, value) { attributes[name] = String(value); },
    removeAttribute(name) { delete attributes[name]; },
    hasAttribute(name) { return name in attributes; },
};
const listeners = [];
globalThis.document = {
    documentElement: html,
    body: {classList: {contains: () => __DARK__}, appendChild() {}},
    head: {appendChild() {}},
    querySelector: () => null,
    querySelectorAll: () => [],
    getElementById: () => null,
    createElement: () => ({setAttribute() {}, appendChild() {}, addEventListener() {},
                           classList: {add() {}, remove() {}}, style: {setProperty() {}}}),
    addEventListener(type) { listeners.push("document:" + type); },
};
globalThis.window = globalThis;
globalThis.location = {href: "https://forge.test/?x=1"};
globalThis.addEventListener = (type) => listeners.push("window:" + type);
globalThis.requestAnimationFrame = (fn) => { fn(); return 1; };
globalThis.setTimeout = (fn) => { fn(); return 1; };
const store = __STORE__;
globalThis.localStorage = {
    getItem: (key) => (key in store ? store[key] : null),
    setItem: (key, value) => { store[key] = String(value); },
};
__OPTS__
const hooks = {loaded: [], changed: [], after: [], tab: []};
globalThis.onUiLoaded = (fn) => hooks.loaded.push(fn);
globalThis.onOptionsChanged = (fn) => hooks.changed.push(fn);
globalThis.onAfterUiUpdate = (fn) => hooks.after.push(fn);
globalThis.onUiTabChange = (fn) => hooks.tab.push(fn);

SOURCE

const T = globalThis.mcTheme;
const out = {};
__SCENARIO__
console.log(JSON.stringify(out));
"""


def run(scenario: str, store=None, opts=None, dark=True) -> dict:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    harness = (HARNESS
               .replace("__STORE__", json.dumps(store or {}))
               .replace("__OPTS__", f"globalThis.opts = {json.dumps(opts)};" if opts is not None else "")
               .replace("__DARK__", "true" if dark else "false")
               .replace("SOURCE", SCRIPT.read_text(encoding="utf-8"))
               .replace("__SCENARIO__", scenario))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# The pure parts
# --------------------------------------------------------------------------- #


class TestTheTabNames:
    def test_a_space_before_a_capital_and_around_digits_with_acronyms_kept(self):
        found = run("""
            out.names = ["txt2img", "img2img", "LLM Studio", "Extended LoRA Details", "PNG Info",
                         "WanGP", "Voice Box", "Clipboard", "MiniPaint", "Checkpoint Merger", "",
                         "  spaced  "].map(T.spaced);
        """)

        assert found["names"] == ["Txt 2 Img", "Img 2 Img", "LLM Studio", "Extended LoRA Details",
                                  "PNG Info", "WanGP", "Voice Box", "Clipboard", "Mini Paint",
                                  "Checkpoint Merger", "", "Spaced"]


class TestThePlanOfWhatFits:
    def test_the_first_tabs_that_fit_are_shown_in_order(self):
        found = run("out.plan = T.plan([100, 100, 100, 100], 250, 0);")

        assert found["plan"] == {"shown": [0, 1], "hidden": [2, 3]}

    def test_the_selected_tab_is_always_shown_at_the_last_ones_expense(self):
        found = run("out.plan = T.plan([100, 100, 100, 100], 250, 3);")

        assert found["plan"] == {"shown": [0, 3], "hidden": [1, 2]}

    def test_everything_fits_when_there_is_room(self):
        found = run("out.plan = T.plan([100, 100, 100], 1000, 1);")

        assert found["plan"] == {"shown": [0, 1, 2], "hidden": []}

    def test_a_bar_too_narrow_for_anything_hides_everything_into_the_menu(self):
        found = run("out.plan = T.plan([100, 100], 50, 1);")

        assert found["plan"] == {"shown": [], "hidden": [0, 1]}


class TestTheIcons:
    def test_only_a_button_whose_whole_text_is_the_emoji_gets_an_icon(self):
        """A span with a cross in it lost its contents to the Lobe theme's
        swap once (handoff 36); this one matches the whole text or nothing."""
        found = run("""
            out.clear = T.iconFor("🗑️");
            out.clearBare = T.iconFor("🗑");
            out.padded = T.iconFor("  ↙️ ");
            out.mixed = T.iconFor("🗑️ Clear");
            out.word = T.iconFor("Generate");
            out.cross = T.iconFor("×");
            out.every = T.icons.map((key) => !!T.iconFor(key));
        """)

        assert found["clear"].startswith("<svg") and "<path" in found["clear"]
        assert found["clearBare"] == found["clear"]
        assert found["padded"].startswith("<svg")
        assert found["mixed"] is None
        assert found["word"] is None
        assert found["cross"] is None
        assert all(found["every"])

    def test_the_icons_cover_forges_own_tool_buttons(self):
        """modules/ui.py, ui_prompt_styles.py and ui_common.py, by their emoji."""
        found = run("out.icons = T.icons;")

        for emoji in ("↙", "🗑", "📋", "🔄", "🌀", "🎲", "♻", "📂", "💾", "🗃", "🖼", "🎨", "📐", "✨",
                      "⇅", "🖌", "📝"):
            assert emoji in found["icons"], emoji


class TestTheSettingsAsRead:
    def test_nothing_set_is_the_defaults(self):
        found = run("out.config = T.normalise({});")

        assert found["config"] == T_DEFAULTS()

    def test_a_value_out_of_range_or_out_of_vocabulary_is_the_default(self):
        found = run("""
            out.quick = T.normalise({quick: "bogus"}).quick;
            out.narrow = T.normalise({quickWidth: 50}).quickWidth;
            out.wide = T.normalise({quickWidth: 10000}).quickWidth;
            out.text = T.normalise({quickWidth: "abc"}).quickWidth;
            out.labels = T.normalise({labels: "spaced"}).labels;
            out.badLabels = T.normalise({labels: "camel"}).labels;
            out.title = T.normalise({title: "  SD  "}).title;
            out.enabled = T.normalise({enabled: 1}).enabled;
        """)

        assert found["quick"] == "collapsed"
        assert found["narrow"] == 200
        assert found["wide"] == 600
        assert found["text"] == 280
        assert found["labels"] == "spaced"
        assert found["badLabels"] == "as_forge"
        assert found["title"] == "SD"
        assert found["enabled"] is True

    def test_the_script_reads_the_options_by_the_names_python_registers(self):
        script = SCRIPT.read_text(encoding="utf-8")

        for name in (mc_theme.OPT_ENABLED, mc_theme.OPT_SPLIT, mc_theme.OPT_QUICK,
                     mc_theme.OPT_QUICK_WIDTH, mc_theme.OPT_PROMPTS, mc_theme.OPT_ICONS,
                     mc_theme.OPT_TITLE, mc_theme.OPT_TAB_LABELS, mc_theme.OPT_CONFIRM_UNLOAD,
                     mc_theme.OPT_FONT):
            assert f'"{name}"' in script, name

    def test_the_defaults_here_match_pythons(self):
        found = run("out.defaults = T.DEFAULTS;")

        assert found["defaults"]["quick"] == mc_theme.QUICK_DEFAULT
        assert found["defaults"]["quickWidth"] == mc_theme.QUICK_WIDTH_DEFAULT
        assert found["defaults"]["labels"] == mc_theme.TAB_LABEL_DEFAULT
        assert found["defaults"]["title"] == mc_theme.TITLE_DEFAULT
        assert found["defaults"]["enabled"] is False


def T_DEFAULTS():
    return {"enabled": False, "split": True, "quick": "collapsed", "quickWidth": 280,
            "prompts": True, "icons": True, "title": "Forge", "labels": "as_forge",
            "confirm": False, "font": ""}


class TestTheThemeToggle:
    def test_it_rewrites_the_theme_query_rather_than_adding_a_second(self):
        """Forge's set_theme() adds the query only when there is none."""
        found = run("""
            out.toLight = T.nextThemeUrl("https://f.test/?__theme=dark&x=1", true);
            out.toDark = T.nextThemeUrl("https://f.test/", false);
        """)

        assert found["toLight"] == "https://f.test/?__theme=light&x=1"
        assert found["toDark"] == "https://f.test/?__theme=dark"

    def test_a_typeface_is_asked_of_google_fonts_by_family(self):
        found = run('out.url = T.fontUrl("Noto Sans");')

        assert found["url"].startswith("https://fonts.googleapis.com/css2?family=Noto+Sans")
        assert "display=swap" in found["url"]


class TestTheFirstPaint:
    def test_the_last_configuration_marks_the_root_before_the_host_is_up(self):
        found = run("out.attributes = Object.assign({}, html.attributes);",
                    store={"mc-theme": json.dumps({"enabled": True, "prompts": True})})

        assert found["attributes"] == {"data-mc-theme": "on", "data-mc-theme-prompts": "resizable"}

    def test_no_configuration_marks_nothing(self):
        found = run("out.attributes = Object.assign({}, html.attributes);")

        assert found["attributes"] == {}

    def test_a_configuration_that_was_off_marks_nothing(self):
        found = run("out.attributes = Object.assign({}, html.attributes);",
                    store={"mc-theme": json.dumps({"enabled": False, "prompts": True})})

        assert found["attributes"] == {}

    def test_it_asks_the_host_for_every_moment_it_can_change(self):
        found = run("""
            out.hooks = {loaded: hooks.loaded.length, changed: hooks.changed.length,
                         after: hooks.after.length, tab: hooks.tab.length};
            out.listeners = listeners;
        """)

        assert found["hooks"] == {"loaded": 1, "changed": 1, "after": 1, "tab": 1}
        assert "window:resize" in found["listeners"]
        assert "document:keydown" in found["listeners"]

    def test_off_in_the_settings_takes_the_mark_off_again(self):
        found = run("""
            hooks.loaded.forEach((fn) => fn());
            out.attributes = Object.assign({}, html.attributes);
            out.stored = JSON.parse(store["mc-theme"]).enabled;
        """, store={"mc-theme": json.dumps({"enabled": True, "prompts": True})},
           opts={"model_chain_theme": False})

        assert found["attributes"] == {}
        assert found["stored"] is False


# --------------------------------------------------------------------------- #
# The stylesheet, as text
# --------------------------------------------------------------------------- #


def section() -> str:
    text = STYLE.read_text(encoding="utf-8")
    start = text.rfind("/* =====", 0, text.index(BANNER))
    end = text.index("/* =====", start + len(BANNER) + 20)
    return text[start:end]


def rules(css: str) -> list:
    bare = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    bare = re.sub(r"@media[^{]*\{", "", bare)
    found = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", bare):
        selector = " ".join(match.group(1).split())
        if selector:
            found.append((selector, match.group(2)))
    return found


class TestTheStylesheetSection:
    def test_every_rule_is_under_the_attribute_the_script_writes(self):
        """Inert without the theme, and gone the moment it is turned off."""
        found = rules(section())

        assert found
        for selector, _ in found:
            for part in selector.split(","):
                assert '[data-mc-theme="on"]' in part or '[data-mc-theme-prompts=' in part, part

    def test_the_header_is_forges_own_tab_bar_in_place(self):
        """Forge reads the current tab from `#tabs > .tab-nav > button.selected`,
        a direct-child selector; the bar is styled where it is, never wrapped."""
        selectors = [selector for selector, _ in rules(section())]

        assert any("#tabs > .tab-nav" in s and "sticky" in body
                   for (s, body) in rules(section()))
        assert all("#tabs > .tab-nav" in s or ".tab-nav" not in s for s in selectors)

    def test_it_states_no_colour_of_its_own(self):
        for selector, body in rules(section()):
            for declaration in body.split(";"):
                if ":" not in declaration:
                    continue
                value = declaration.split(":", 1)[1]
                assert not re.search(r"#[0-9a-fA-F]{3,8}\b", value), (selector, declaration)
                assert not re.search(r"\brgba?\(|\bhsla?\(", value), (selector, declaration)

    def test_it_names_no_generated_class(self):
        for selector, _ in rules(section()):
            assert ".svelte" not in selector, selector

    def test_important_only_where_the_host_says_it_first(self):
        allowed = ("[hidden]", "mc-theme-overflow", ".gradio-button.tool", "token-counter",
                   "#quicksettings", "gradio-accordion", "_toprow", ".mc-theme-action")
        for selector, body in rules(section()):
            if "!important" not in body:
                continue
            assert any(mark in selector for mark in allowed), selector

    def test_the_hidden_attribute_wins_over_any_display_rule(self):
        """The Lobe trap: a theme's display rule beats the browser's own
        [hidden]. The drawer and the menu are hidden with the attribute."""
        css = section()

        assert ':root[data-mc-theme="on"] [hidden]' in css
        assert "display: none !important" in css

    def test_the_script_and_the_sheet_agree_on_the_names(self):
        script = SCRIPT.read_text(encoding="utf-8")
        css = section()

        for css_name, script_name in (("mc-theme-title", "mc-theme-title"),
                                      ("mc-theme-actions", "mc-theme-actions"),
                                      ("mc-theme-action", "mc-theme-action"),
                                      ("mc-theme-menu", "mc-theme-menu"),
                                      ("mc-theme-menu-item", "mc-theme-menu-item"),
                                      ("mc-theme-overflow", "mc-theme-overflow"),
                                      ("mc-theme-drawer", "mc-theme-drawer"),
                                      ("data-mc-theme-prompts", "data-mc-theme-prompts"),
                                      ("data-mc-theme-icon", "mcThemeIcon")):
            assert script_name in script, script_name
            assert css_name in css, css_name
