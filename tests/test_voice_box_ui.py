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


@pytest.mark.parametrize("attribute", ["data-mc-voice-key", "data-mc-voice-box-prefix",
                                       "data-mc-voice-box-boot"])
def test_each_attribute_appears_exactly_once(attribute):
    assert mc_voice_box_ui.root_markup().count(attribute + "=") == 1
