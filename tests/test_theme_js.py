"""The theme's page script, as far as node can hold it, and its stylesheet
section as text.

Node has no layout, so the header's fit, the drawer and the moves are checked
in Chromium (``tests/test_theme_layout.py``). What node holds is what is pure:
the tab-name spacing, the plan of which tabs fit, the emoji-to-icon match, the
settings' normalisation, the theme toggle's address, the reading of Forge's
generation parameters, the first paint from the last configuration, and the
hooks the script asks the host for.
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
    addEventListener(type, handler, options) { listeners.push("document:" + type + (options && options.passive ? ":passive" : "")); },
};
globalThis.window = globalThis;
globalThis.location = {href: "https://forge.test/?x=1"};
globalThis.addEventListener = (type) => listeners.push("window:" + type);
globalThis.visualViewport = {addEventListener(type) { listeners.push("visualViewport:" + type); }};
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


class TestTheGlyphs:
    def test_every_glyph_the_script_asks_for_is_drawn(self):
        """A header or drawer button is made by name from the glyph table; a
        name the table lacks is the text "undefined" on the page -- the
        drawer's close button was, on the user's machine."""
        script = SCRIPT.read_text(encoding="utf-8")
        asked = set(re.findall(r'action\("[^"]*",\s*"([a-z0-9-]+)"', script))
        asked |= set(re.findall(r'GLYPH\["([a-z0-9-]+)"\]', script))
        found = run("out.glyphs = T.glyphs;")

        assert asked, "the script asks for glyphs by name"
        assert asked <= set(found["glyphs"]), asked - set(found["glyphs"])
        assert {"x", "captions", "captions-off", "columns-3"} <= asked


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
            out.networks = T.normalise({networks: "sideways"}).networks;
            out.networksOpen = T.normalise({networks: "open"}).networks;
            out.networksNarrow = T.normalise({networksWidth: 10}).networksWidth;
            out.networksWide = T.normalise({networksWidth: 5000}).networksWidth;
            out.cardSmall = T.normalise({networksCard: 1}).networksCard;
            out.cardLarge = T.normalise({networksCard: 999}).networksCard;
            out.cardText = T.normalise({networksCard: "big"}).networksCard;
            out.layout = T.normalise({networksMode: "column"}).networksMode;
            out.badLayout = T.normalise({networksMode: "sideways"}).networksMode;
            out.infoOff = T.normalise({info: false}).info;
            out.infoUnset = T.normalise({}).info;
        """)

        assert found["quick"] == "collapsed"
        assert found["narrow"] == 200
        assert found["wide"] == 600
        assert found["text"] == 280
        assert found["labels"] == "spaced"
        assert found["badLabels"] == "as_forge"
        assert found["title"] == "SD"
        assert found["enabled"] is True
        assert found["networks"] == "collapsed"
        assert found["networksOpen"] == "open"
        assert found["networksNarrow"] == mc_theme.NETWORKS_WIDTH_MIN
        assert found["networksWide"] == mc_theme.NETWORKS_WIDTH_MAX
        assert found["cardSmall"] == mc_theme.NETWORKS_CARD_MIN
        assert found["cardLarge"] == mc_theme.NETWORKS_CARD_MAX
        assert found["cardText"] == mc_theme.NETWORKS_CARD_DEFAULT
        assert found["layout"] == "column" and found["badLayout"] == "drawer"
        assert found["infoOff"] is False
        assert found["infoUnset"] is True

    def test_the_script_reads_the_options_by_the_names_python_registers(self):
        script = SCRIPT.read_text(encoding="utf-8")

        for name in (mc_theme.OPT_ENABLED, mc_theme.OPT_SPLIT, mc_theme.OPT_QUICK,
                     mc_theme.OPT_QUICK_WIDTH, mc_theme.OPT_PROMPTS, mc_theme.OPT_ICONS,
                     mc_theme.OPT_TITLE, mc_theme.OPT_TAB_LABELS, mc_theme.OPT_CONFIRM_UNLOAD,
                     mc_theme.OPT_FONT, mc_theme.OPT_NETWORKS, mc_theme.OPT_NETWORKS_WIDTH,
                     mc_theme.OPT_NETWORKS_CARD, mc_theme.OPT_NETWORKS_MODE, mc_theme.OPT_IMAGE_INFO,
                     mc_theme.OPT_SCROLL):
            assert f'"{name}"' in script, name

    def test_the_defaults_here_match_pythons(self):
        found = run("out.defaults = T.DEFAULTS;")

        assert found["defaults"]["quick"] == mc_theme.QUICK_DEFAULT
        assert found["defaults"]["quickWidth"] == mc_theme.QUICK_WIDTH_DEFAULT
        assert found["defaults"]["labels"] == mc_theme.TAB_LABEL_DEFAULT
        assert found["defaults"]["title"] == mc_theme.TITLE_DEFAULT
        assert found["defaults"]["enabled"] is False
        assert found["defaults"]["networks"] == mc_theme.NETWORKS_DEFAULT
        assert found["defaults"]["networksWidth"] == mc_theme.NETWORKS_WIDTH_DEFAULT
        assert found["defaults"]["networksCard"] == mc_theme.NETWORKS_CARD_DEFAULT
        assert found["defaults"]["networksMode"] == mc_theme.NETWORKS_LAYOUT_DEFAULT
        assert found["defaults"]["info"] is True
        assert found["defaults"]["scroll"] is True

    def test_the_one_scroller_switch_reads_as_a_boolean_and_is_on_unset(self):
        found = run("""
            out.off = T.normalise({scroll: false}).scroll;
            out.on = T.normalise({scroll: 1}).scroll;
            out.unset = T.normalise({}).scroll;
        """)

        assert found == {"off": False, "on": True, "unset": True}


class TestOneScrollerForThePage:
    """The plan behind the two phases (model_chain_theme.js, `scrollPlan`):
    the settings column's sticky offset and the row's least height, from the
    columns' heights, the view's height under where the results are held,
    and the offset that holds them. Chromium holds the layout that follows
    (tests/test_theme_layout.py); node holds the arithmetic."""

    def test_a_tall_settings_column_is_held_once_its_bottom_meets_the_views(self):
        # The user's page: 2206 px of settings, 1035 of results, the results
        # held 88 px down with 804 px of view under that. The settings' top
        # is held at 88 + 804 - 2206, their bottom at the view's bottom; the
        # row is the settings plus the 231 px of results under the view.
        found = run("out.plan = T.scrollPlan(2206, 1035, 804, 88);")

        assert found["plan"] == {"top": -1314, "minHeight": 2437}

    def test_a_short_settings_column_is_held_where_the_results_are(self):
        found = run("out.plan = T.scrollPlan(256, 1035, 804, 88);")

        assert found["plan"] == {"top": 88, "minHeight": 487}

    def test_results_that_fit_the_view_leave_the_row_the_settings_height(self):
        found = run("out.plan = T.scrollPlan(2206, 700, 804, 88);")

        assert found["plan"] == {"top": -1314, "minHeight": 2206}

    def test_the_numbers_are_whole_pixels(self):
        found = run("out.plan = T.scrollPlan(2206.4, 1035.3, 803.6, 88);")

        assert found["plan"] == {"top": -1315, "minHeight": 2438}

    def test_only_the_momentum_break_listens_to_the_scroll_and_passively(self):
        """The columns are held by sticky positioning from measured numbers;
        the Details button's marking, which listened to the scroll, went with
        the button at the user's word. The momentum break reads the scroll
        from a passive listener in the capture phase and writes nothing
        unless a fling crosses the boundary (Chromium holds that); nothing
        else listens."""
        found = run("hooks.loaded.forEach((fn) => fn()); out.listeners = listeners;", opts={"model_chain_theme": True})

        assert "document:scroll:passive" in found["listeners"]
        assert "document:scrollend:passive" in found["listeners"]
        assert "document:scroll" not in found["listeners"], "no listener that could block the scroll"
        assert "window:scroll" not in found["listeners"]
        assert found["listeners"].count("document:scroll:passive") == 1

    def test_the_boundary_is_where_the_settings_bottom_edge_meets_the_views(self):
        """The scroll at which the first phase ends (`scrollBoundary`): the
        settings column's natural top in the scroller's coordinates plus its
        height (margins in), less the view's bottom edge. The user's page: the
        row's content edge 110 px down, 2206 px of settings, the view's
        bottom at 892."""
        found = run("out.boundary = T.scrollBoundary(110, 2206, 892); out.fits = T.scrollBoundary(110, 700, 892);"
                    " out.whole = T.scrollBoundary(110.4, 2206.3, 891.6); out.zero = T.scrollBoundary(0, 892, 892);")

        assert found["boundary"] == 1424
        assert found["fits"] is None, "settings that fit the view have no first phase"
        assert found["whole"] == 1425
        assert found["zero"] is None


class TestTheCardsPerRow:
    """The arithmetic behind the extra-network drawer's slider
    (model_chain_theme.js, `perRowRange`, `perRowNearest`): the counts a row
    of cards may hold, cards no wider than the card size setting's top and
    no narrower than its bottom, and the count a browser starts at. Chromium
    holds the grid that follows (tests/test_theme_layout.py)."""

    def test_the_counts_offered_keep_every_card_within_the_settings_bounds(self):
        # The drawer at 340 px: 290 px of row, cards 8 px apart.
        found = run("out.range = T.perRowRange(290, 8); out.wide = T.perRowRange(776, 8);")

        assert found["range"] == {"min": 1, "max": 4}
        assert found["wide"] == {"min": 3, "max": 11}
        for room, key in ((290, "range"), (776, "wide")):
            low, high = found[key]["min"], found[key]["max"]

            def card(count):
                return (room - (count - 1) * 8) / count

            assert card(low) <= mc_theme.NETWORKS_CARD_MAX and (low == 1 or card(low - 1) > mc_theme.NETWORKS_CARD_MAX)
            assert card(high) >= mc_theme.NETWORKS_CARD_MIN and card(high + 1) < mc_theme.NETWORKS_CARD_MIN

    def test_a_row_too_narrow_for_two_small_cards_still_holds_one(self):
        found = run("out.narrow = T.perRowRange(50, 8); out.none = T.perRowRange(0, 8);")

        assert found["narrow"] == {"min": 1, "max": 1}
        assert found["none"] == {"min": 1, "max": 1}

    def test_a_browser_starts_at_the_count_whose_cards_come_nearest_the_setting(self):
        found = run("""
            const range = T.perRowRange(290, 8);
            out.default = T.perRowNearest(290, 8, 120, range);
            out.lobe = T.perRowNearest(290, 8, 86, range);
            out.tiny = T.perRowNearest(290, 8, 20, range);
            out.huge = T.perRowNearest(290, 8, 900, range);
        """)

        # 2 cards of 141 px for 120; 3 of 91 for the 86 the Lobe import gave.
        assert found == {"default": 2, "lobe": 3, "tiny": 4, "huge": 1}


def T_DEFAULTS():
    return {"enabled": False, "split": True, "scroll": True, "quick": "collapsed", "quickWidth": 280,
            "prompts": True, "icons": True, "title": "Forge", "labels": "as_forge",
            "confirm": False, "font": "", "networks": "collapsed", "networksWidth": 340,
            "networksCard": 120, "networksMode": "drawer", "info": True}


# What Forge writes under the gallery (modules/ui_common.py, plaintext_to_html:
# the lines of the infotext joined by <br>), as the text of that paragraph.
INFOTEXT = (
    'a cat in a hat, <lora:detail:0.8>\n'
    'Negative prompt: blurry, (low quality:1.2)\n'
    'Steps: 8, Sampler: Euler a, Schedule type: Automatic, CFG scale: 1, Seed: 42, '
    'Size: 1024x1024, Model hash: abc123, Model: krea2, '
    'Lora hashes: "detail: deadbeef, other: cafe", Version: neo'
)


class TestTheGenerationInfo:
    def test_the_prompt_the_negative_and_the_pairs_are_read_as_forge_reads_them(self):
        """modules/infotext_utils.parse_generation_parameters: the last line
        is the parameters when it holds three pairs or more, a quoted value
        keeps its commas, and 'Negative prompt:' starts the negative."""
        found = run(f"out.parsed = T.parseInfotext({json.dumps(INFOTEXT)});")

        assert found["parsed"]["prompt"] == "a cat in a hat, <lora:detail:0.8>"
        assert found["parsed"]["negative"] == "blurry, (low quality:1.2)"
        assert found["parsed"]["pairs"] == [
            ["Steps", "8"], ["Sampler", "Euler a"], ["Schedule type", "Automatic"],
            ["CFG scale", "1"], ["Seed", "42"], ["Size", "1024x1024"], ["Model hash", "abc123"],
            ["Model", "krea2"], ["Lora hashes", "detail: deadbeef, other: cafe"], ["Version", "neo"]]

    def test_a_prompt_of_several_lines_and_no_negative(self):
        found = run("""
            out.lines = T.parseInfotext("one\\ntwo\\n\\nSteps: 1, Sampler: a, Seed: 3");
            out.negativeLines = T.parseInfotext("one\\nNegative prompt: bad\\nworse\\nSteps: 1, Sampler: a, Seed: 3");
        """)

        assert found["lines"] == {"prompt": "one\ntwo", "negative": "", "pairs": [["Steps", "1"], ["Sampler", "a"], ["Seed", "3"]]}
        assert found["negativeLines"]["prompt"] == "one"
        assert found["negativeLines"]["negative"] == "bad\nworse"

    def test_a_last_line_with_fewer_than_three_pairs_is_prompt(self):
        found = run("""
            out.short = T.parseInfotext("a prompt\\nSteps: 8");
            out.only = T.parseInfotext("just a prompt");
        """)

        assert found["short"] == {"prompt": "a prompt\nSteps: 8", "negative": "", "pairs": []}
        assert found["only"] == {"prompt": "just a prompt", "negative": "", "pairs": []}

    def test_nothing_is_nothing(self):
        found = run("""
            out.empty = T.parseInfotext("");
            out.blank = T.parseInfotext("  \\n\\r\\n ");
            out.none = T.parseInfotext(null);
        """)

        assert found["empty"] is None and found["blank"] is None and found["none"] is None


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
        # The quick settings drawer's tap outside: a press, and its click.
        for kind in ("document:pointerdown", "document:click"):
            assert kind in found["listeners"], kind
        # What changes what the columns are held against without a resize
        # of the window: a pinch, the page shown again, the browser's full
        # screen coming or going under focus mode.
        for kind in ("visualViewport:resize", "window:pageshow", "document:visibilitychange",
                     "document:fullscreenchange", "document:webkitfullscreenchange"):
            assert kind in found["listeners"], kind

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
        # Each is a place where the host, Gradio's inline style on a tab panel
        # or a card, or a theme's rule about every button says !important
        # first; a drawer's own rule needs none.
        # Forge's `.edit-user-metadata` says `padding` and `background`
        # !important itself, so the dialog's own padding and surface must too.
        # Gradio writes every block's `overflow` inline, so the paragraph
        # kept out of view clips its line !important; and Forge's own
        # `div.gradio-container { overflow: visible !important }` is why the
        # container's sideways clip is.
        allowed = ("[hidden]", "mc-theme-overflow", ".gradio-button.tool", "token-counter",
                   "#quicksettings", "gradio-accordion", "_toprow", ".mc-theme-action",
                   ".mc-theme-networks-nav", ".mc-theme-networks-page",
                   ".mc-theme-networks-page .extra-network-pane .card", ".mc-theme-info-copy",
                   ".edit-user-metadata", "forge-assistant-focused", ".mc-theme-info-source",
                   "div.gradio-container.app")
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
                                      ("mc-theme-networks", "mc-theme-networks"),
                                      ("mc-theme-networks-nav", "mc-theme-networks-nav"),
                                      ("mc-theme-networks-shown", "mc-theme-networks-shown"),
                                      ("mc-theme-networks-tab", "mc-theme-networks-tab"),
                                      ("mc-theme-networks-active", "mc-theme-networks-active"),
                                      ("mc-theme-networks-host", "mc-theme-networks-host"),
                                      ("mc-theme-networks-open", "mc-theme-networks-open"),
                                      ("mc-theme-networks-page", "mc-theme-networks-page"),
                                      ("mc-theme-networks-handle", "mc-theme-networks-handle"),
                                      ("mc-theme-networks-resizing", "mc-theme-networks-resizing"),
                                      ("mc-theme-networks-view", "mc-theme-networks-view"),
                                      ("mc-theme-networks-untitled", "mc-theme-networks-untitled"),
                                      ("mc-theme-networks-per-row", "mc-theme-networks-per-row"),
                                      ("mc-theme-networks-per-row-icon", "mc-theme-networks-per-row-icon"),
                                      ("mc-theme-networks-per-row-count", "mc-theme-networks-per-row-count"),
                                      ("--mc-theme-networks-cols", "--mc-theme-networks-cols"),
                                      ("data-mc-theme-networks", "data-mc-theme-networks"),
                                      ("mc-theme-info-source", "mc-theme-info-source"),
                                      ("mc-theme-info-block", "mc-theme-info-block"),
                                      ("mc-theme-info-chip", "mc-theme-info-chip"),
                                      ("mc-theme-info-copy", "mc-theme-info-copy"),
                                      ("mc-theme-info-copied", "mc-theme-info-copied"),
                                      ("data-mc-theme-prompts", "data-mc-theme-prompts"),
                                      ("data-mc-theme-scroll", "data-mc-theme-scroll"),
                                      ("mc-theme-scroll-settings", "mc-theme-scroll-settings"),
                                      ("--mc-theme-results-hold", "--mc-theme-results-hold"),
                                      ("--mc-theme-scroll-hold", "--mc-theme-scroll-hold"),
                                      ("--mc-theme-scroll-row", "--mc-theme-scroll-row"),
                                      ("data-mc-theme-icon", "mcThemeIcon")):
            assert script_name in script, script_name
            assert css_name in css, css_name
