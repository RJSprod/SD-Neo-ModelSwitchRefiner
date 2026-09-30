"""The Voice Box tab is one HTML root and nothing else of Gradio's.

Mini Paint NEO's Gradio 4.40 traps are why the tab has this shape: a component
of another tab in an outputs list fails silently and totally, an equal string
handed to a component is no change, and a page that owns its own DOM avoids
both. So what these tests hold is the shape itself -- the tuple ``on_ui_tabs``
returns, the root's attributes (the token the routes check, their prefix, the
boot flag the script looks for) and that no component beyond the HTML is ever
built, which is what makes "no other tab's component in an outputs list" true
by construction: there is no outputs list.
"""

from __future__ import annotations

import ast
import html
import re
from pathlib import Path

import gradio as gr
import pytest

import mc_voice_api
import mc_voice_box_ui

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "mc_voice_box_ui.py"


def _component_bases() -> tuple:
    """Every class a Gradio component or layout block descends from, on the
    fake gradio the suite runs under and on the real one alike."""
    found = []
    components = getattr(gr, "components", None)
    if components is not None and isinstance(getattr(components, "Component", None), type):
        found.append(components.Component)
    blocks = getattr(gr, "blocks", None)
    if blocks is not None and isinstance(getattr(blocks, "BlockContext", None), type):
        found.append(blocks.BlockContext)
    return tuple(found)


class TestTheTab:
    def test_the_tab_tuple(self):
        tabs = mc_voice_box_ui.on_ui_tabs()

        assert len(tabs) == 1
        blocks, label, ident = tabs[0]
        assert isinstance(blocks, gr.Blocks)
        assert label == "Voice Box"
        assert ident == "mc_voice_box"

    def test_the_tab_never_raises_into_the_host(self, monkeypatch):
        """A tab that throws takes the whole WebUI's UI down with it."""
        monkeypatch.setattr(mc_voice_box_ui, "build",
                            lambda: (_ for _ in ()).throw(RuntimeError("no gradio here")))

        assert mc_voice_box_ui.on_ui_tabs() == []

    def test_nothing_but_the_html_is_built(self, monkeypatch):
        """One component, and it is the root. No Markdown heading, no hidden
        Textbox for the token, no button: everything else is the script's."""
        made = []
        bases = _component_bases()
        assert bases, "the gradio in use has to expose its component base"
        for name, value in list(vars(gr).items()):
            if not isinstance(value, type) or not issubclass(value, bases):
                continue
            if name in ("Blocks",):
                continue
            original = value.__init__

            def record(self, *args, _name=name, _original=original, **kwargs):
                made.append((_name, self))
                return _original(self, *args, **kwargs)

            monkeypatch.setattr(value, "__init__", record)

        mc_voice_box_ui.on_ui_tabs()

        assert [name for name, _ in made] == ["HTML"]
        _, component = made[0]
        assert getattr(component, "elem_id", None) == "mc_voice_box_root"
        assert getattr(component, "value", None) == mc_voice_box_ui.root_markup()


class TestTheRoot:
    def test_the_root_carries_the_token_the_prefix_and_the_boot_flag(self):
        token = mc_voice_api.session_token()

        markup = mc_voice_box_ui.root_markup()

        assert markup.startswith('<div id="mc-voice-box" class="mc-voice-box" ')
        assert f'data-mc-voice-key="{html.escape(token, quote=True)}"' in markup
        assert 'data-mc-voice-box-prefix="/model-chain/voice-box"' in markup
        assert 'data-mc-voice-box-boot="1"' in markup
        assert markup.endswith("></div>")

    def test_the_root_is_one_empty_element(self):
        """Empty on purpose: the script clears and fills it, and a placeholder
        would be a second copy of the page's first words to keep in step."""
        markup = mc_voice_box_ui.root_markup()

        assert markup.count("<") == 2
        assert "></div>" in markup
        assert ">Loading" not in markup

    def test_the_token_is_the_one_the_routes_check(self):
        """The same token as Voice Chat's routes: minted per WebUI process, so a
        tab left open across a restart fails with "reload" rather than half-working."""
        assert mc_voice_api.session_token() in mc_voice_box_ui.root_markup()
        assert 'data-mc-voice-key' in mc_voice_box_ui.root_markup()

    def test_the_token_is_escaped_for_an_attribute(self, monkeypatch):
        monkeypatch.setattr(mc_voice_api, "session_token", lambda: 'a"b<c>&d')

        markup = mc_voice_box_ui.root_markup()

        assert 'data-mc-voice-key="a&quot;b&lt;c&gt;&amp;d"' in markup
        assert '"a"b' not in markup

    def test_the_prefix_is_the_routes_prefix(self):
        import mc_voice_box_api

        assert mc_voice_box_ui.PREFIX == mc_voice_box_api.PREFIX == "/model-chain/voice-box"


class TestWhatTheModuleReaches:
    def test_it_imports_only_voice_modules_gradio_and_the_standard_library(self):
        """Invariant I-3, stated for this module in particular: the tab is a
        voice module and never reaches the memory planner, the broker, the
        plan, the language model runtime or the turns -- and, being static
        markup, needs none of the host's ``modules`` either."""
        names = set()
        for node in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split(".")[0])

        allowed = {"__future__", "html", "logging", "gradio"}
        for name in names:
            assert name in allowed or name.startswith("mc_voice_"), name
        assert "mc_voice_api" in names
        assert not any(name in names for name in ("mc_memory", "mc_broker", "mc_plan",
                                                  "mc_llm_runtime", "mc_turns", "modules"))

    def test_the_markup_names_are_the_ones_the_script_and_stylesheet_use(self):
        """Three files agree on one id: Python paints it, the script boots on
        it, and every rule of the stylesheet's Voice Box section is scoped to it."""
        script = (ROOT / "javascript" / "voice_box.js").read_text(encoding="utf-8")
        css = (ROOT / "style.css").read_text(encoding="utf-8")

        assert f'const ROOT_ID = "{mc_voice_box_ui.ROOT_ID}";' in script
        assert '"data-mc-voice-key"' in script
        assert '"data-mc-voice-box-prefix"' in script
        assert "/* Voice Box */" in css or "   Voice Box\n" in css
        section = css.split("Voice Box", 1)[1]
        rules = [line for line in section.splitlines()
                 if line and not line[0].isspace() and line.rstrip().endswith("{")
                 and not line.startswith(("@", "/*", "}"))]
        assert rules
        for rule in rules:
            assert rule.startswith(f"#{mc_voice_box_ui.ROOT_ID}"), rule


class TestTheStylesheetSection:
    """What the section promises in its own header: the host's colours, bar one
    tint, and nothing drawn above the host's dialogs."""

    @staticmethod
    def _rules() -> str:
        """The section's rules without its comments (the split lands inside the
        section's banner, so the rest of the banner goes first)."""
        css = (ROOT / "style.css").read_text(encoding="utf-8")
        section = css.split("Voice Box", 1)[1].split("*/", 1)[1]
        return re.sub(r"/\*.*?\*/", "", section, flags=re.S)

    def test_its_one_colour_of_its_own_is_the_tint_and_the_dark_theme_takes_it_darker(self):
        """Every colour is a var() reference to the host's theme except the tint a
        sample carries onto its outputs: a hue the script derives from the
        sample's id, at a saturation and lightness declared once on the root --
        low, and redeclared darker under Forge's dark theme (`.dark`)."""
        rules = self._rules()

        functions = re.findall(
            r"(?<![\w-])((?:hsla?|rgba?|hwb|lab|lch|oklab|oklch|color)\([^;]*);", rules)
        assert functions == ["hsl(var(--mc-voice-box-hue, 0) var(--mc-voice-box-tint-saturation) "
                             "var(--mc-voice-box-tint-lightness))"]

        light = re.search(r"#mc-voice-box \{([^}]*)\}", rules).group(1)
        dark = re.search(r"#mc-voice-box:is\(\.dark \*\) \{([^}]*)\}", rules).group(1)

        def percent(block: str, name: str) -> float:
            return float(re.search(re.escape(name) + r":\s*([\d.]+)%", block).group(1))

        for block in (light, dark):
            assert 0 < percent(block, "--mc-voice-box-tint-saturation") <= 50, "quiet, not colourful"
        assert (percent(dark, "--mc-voice-box-tint-lightness")
                < percent(light, "--mc-voice-box-tint-lightness"))

    def test_it_sets_no_z_index(self):
        """Mini Paint's dialog layer (2000) and the assistant's panel (1200) stay
        above everything the tab draws."""
        assert "z-index" not in self._rules()

    def test_nothing_hidden_can_be_shown_by_a_theme(self):
        """The script hides with the `hidden` attribute, which only the
        browser's own stylesheet turns into `display: none`, and any author
        rule about `display` beats that. The section says it again, under the
        root's id and `!important`, so a theme's `button { display: ...
        !important }` does not bring back a hidden Dismiss."""
        rules = self._rules()

        found = re.search(r"#mc-voice-box \[hidden\],\s*#mc-voice-box\[hidden\] \{([^}]*)\}", rules)
        assert found, "the section's [hidden] rule went"
        assert re.fullmatch(r"\s*display:\s*none\s*!important;\s*", found.group(1)), found.group(1)

    def test_the_page_has_no_native_checkbox_left(self):
        """Sampling and Keep warm are switches the script draws: a theme
        squeezed the checkboxes they were to a sliver nobody could tell was a
        control. The script makes no checkbox and the section styles none."""
        script = (ROOT / "javascript" / "voice_box.js").read_text(encoding="utf-8")
        code = re.sub(r"//[^\n]*", "", script)

        assert not re.search(r"""["']checkbox["']""", code)
        assert "checkbox" not in self._rules()
        assert "mc-voice-box-field-check" not in self._rules()

    def test_only_the_trimmer_and_a_range_input_keep_their_sideways_drags(self):
        """On a phone a sideways swipe changes the card everywhere except on a
        control whose drag is sideways: the trimmer's waveform (its handles)
        and a range input, here -- the active player's waveform gets the same
        `pan-y` from the script. No other rule names a touch-action, so every
        other waveform leaves the swipe to the cards."""
        declared = re.findall(r"([^{}]+)\{[^}]*?touch-action:\s*([^;]+);", self._rules())

        assert sorted((selector.strip(), value.strip()) for selector, value in declared) == [
            ("#mc-voice-box .mc-voice-box-trimmer-wave", "pan-y"),
            ('#mc-voice-box input[type="range"]', "pan-y"),
        ]


@pytest.mark.parametrize("attribute", ["data-mc-voice-key", "data-mc-voice-box-prefix",
                                       "data-mc-voice-box-boot"])
def test_each_attribute_appears_exactly_once(attribute):
    assert mc_voice_box_ui.root_markup().count(attribute + "=") == 1
