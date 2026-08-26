"""The Image Pipeline: one surface, six rows, and the contracts that hold it up.

Three kinds of question here, and they fail in three different ways.

**Is it one panel?** The pipeline is built by whichever of the two feature
scripts Forge happens to run first, and filled by both. That is the load-
bearing trick of this refactor, and the way it breaks is silent: two shells,
each with half the stages, on a page nobody looked at in a test.

**Does a row describe the generation it is next to?** Stage 1 is Forge's, read
and never written, so every number on its row comes from a native control this
extension does not own. The failure is a panel that quotes the width and height
sliders at somebody whose Hires pass makes them the wrong numbers.

**Do the labels still line up?** The phase animation matches the progress bar's
own text, which is a real coupling between a Python constant and a JavaScript
table. It is tested rather than hoped for: rename a phase on either side and
this file says so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import mc_llm_progress
import mc_pipeline_panel
import mc_profile_state
import mc_progress

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_JS = ROOT / "javascript" / "model_chain_pipeline.js"


def _panel():
    """The Creative panel, built outside any surface, for a handler test.

    Every caller takes the ``store`` fixture first. Without it the profile
    store writes into the repository, which is how a stray
    ``krea_creative_profiles.json`` once ended up in a commit.
    """
    import mc_creative_panel

    return mc_creative_panel.build(
        lambda *parts: "test-" + "-".join(str(part) for part in parts),
        lambda text, kind="info": text, None)


def _clicked(panel, marker):
    """The click handler wired to the button whose elem_id ends with ``marker``."""
    for button in panel.buttons:
        if str(button.elem_id or "").endswith(marker):
            for kind, kwargs in button._callbacks:
                if kind == "click":
                    return kwargs["fn"]
    raise AssertionError(f"no click handler for {marker}")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point every preferences and history file at a throwaway directory.

    Both roots, because the extension has two. LLM Studio's files hang off
    ``mc_llm_paths.data_root``; the Creative profile store and the Spatial
    layout store hang off the host's own ``paths.data_path``, which falls back
    to the working directory -- which is how a test that saved a profile once
    left a ``krea_creative_profiles.json`` in the repository.
    """
    from modules import paths

    import mc_llm_paths

    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "data_path", str(tmp_path), raising=False)
    yield tmp_path


@pytest.fixture(autouse=True)
def fresh_shell():
    """One shell per test, the way one page gets one shell."""
    mc_pipeline_panel.forget()
    yield
    mc_pipeline_panel.forget()


# --------------------------------------------------------------------------- #
# The shell
# --------------------------------------------------------------------------- #


class TestTheShell:
    def test_the_order_is_the_order_a_generation_runs_in(self):
        """The one ordering in the extension a user can see. A panel that drew
        Spatial above Creative while the code ran them the other way round
        would be a diagram of a different program."""
        assert mc_pipeline_panel.ORDER == ("creative", "spatial", "stage2")

    def test_every_row_is_one_this_extension_runs(self):
        """The panel used to draw Prompt, Stage 1 and Output as well: Forge's
        own, muted and uneditable, so that the path had no holes in it. What
        that produced was three rows restating what the page already said --
        the prompt box is directly above, the sliders directly below, and the
        output is the picture."""
        assert set(mc_pipeline_panel.ORDER) == set(mc_pipeline_panel.OWNED)
        assert not hasattr(mc_pipeline_panel, "CONTEXT")

    def test_it_builds_one_shell_and_hands_the_same_one_back(self, host):
        """The whole mechanism. The second script to call host() must get the
        first script's shell, or the page ends up with two half-filled
        pipelines and no error anywhere."""
        first = mc_pipeline_panel.host()
        second = mc_pipeline_panel.host()

        assert first is second

    def test_every_owned_stage_offers_a_switch_slot_and_an_editor_slot(self, host):
        pipeline = mc_pipeline_panel.host()

        for stage in mc_pipeline_panel.OWNED:
            assert pipeline.head(stage) is not None
            assert pipeline.body(stage) is not None

    def test_the_rows_forge_owned_are_not_built_at_all(self, host):
        """Not hidden, not muted: absent. A row that is present and refuses to
        work is worse than no row."""
        pipeline = mc_pipeline_panel.host()

        for stage in ("prompt", "stage1", "output"):
            assert stage not in pipeline.rows
            assert stage not in pipeline.heads
            assert stage not in pipeline.summaries

    def test_every_stage_has_a_live_second_line(self, host):
        """A row that said only its own name would be decoration."""
        pipeline = mc_pipeline_panel.host()

        for stage in mc_pipeline_panel.ORDER:
            assert pipeline.summary(stage) is not None

    def test_forgetting_it_builds_a_fresh_one(self, host):
        """Forge rebuilds the whole tab on a UI reload, and a shell remembered
        across that would hand the new page slots belonging to a page that no
        longer exists."""
        first = mc_pipeline_panel.host()
        mc_pipeline_panel.forget()

        assert mc_pipeline_panel.host() is not first

    def test_the_handoff_says_the_size_when_it_knows_it(self):
        assert mc_pipeline_panel.handoff_note(1536, 2304) == "1536 × 2304 pixel handoff"
        assert mc_pipeline_panel.handoff_note(0, 0) == "pixel handoff"


# --------------------------------------------------------------------------- #
# One disclosure per card, and every drawer closed
# --------------------------------------------------------------------------- #


SURFACES = ("scripts/model_chain_krea_creative.py", "scripts/model_chain.py",
            "mc_creative_panel.py", "mc_pipeline_panel.py", "mc_plan_panel.py")
"""Every file that draws something on the txt2img tab."""


class TestTheStageCard:
    """§1.3: one large disclosure surface, one switch, one live summary, one
    attached body, and no redundant second "open me" row.

    The card used to be four things stacked -- a title row, a switch, a summary
    line, and then an accordion labelled "Creative direction" that had to be
    opened to reach the settings. Two rows saying the same word, one above the
    other, and only the second one opening anything."""

    def test_the_disclosure_is_the_stage_s_own_name(self, host):
        pipeline = mc_pipeline_panel.host()

        for stage in mc_pipeline_panel.ORDER:
            assert pipeline.editors[stage].label == mc_pipeline_panel.TITLES[stage]

    def test_there_is_no_second_open_me_label(self, host):
        """The table of labels the inner accordion used to carry. Checked as an
        absence rather than left to the eye, because the way this regresses is
        somebody re-adding a drawer inside the card with a title of its own."""
        assert not hasattr(mc_pipeline_panel, "EDITOR_LABELS")

        pipeline = mc_pipeline_panel.host()

        # One disclosure per card: the editor. The body is a plain Column.
        for stage in mc_pipeline_panel.ORDER:
            body = pipeline.bodies[stage]
            assert not hasattr(body, "open")

    def test_the_card_opens_closed(self, host):
        pipeline = mc_pipeline_panel.host()

        for stage in mc_pipeline_panel.ORDER:
            assert pipeline.editors[stage].open is False

    def test_a_bypassed_card_says_bypassed(self):
        """§1: a bypassed card stays visible and configurable, so the summary is
        the only thing between "this setting exists" and "this setting will
        run". It has to say which without being read twice, and every stage has
        to say it the same way."""
        for stage in mc_pipeline_panel.ORDER:
            assert mc_pipeline_panel.PLACEHOLDERS[stage].startswith("Bypassed")

    def test_every_stage_says_bypassed_when_it_is_off(self, store):
        import scripts.model_chain as chain_script
        import scripts.model_chain_krea_creative as creative_script

        assert creative_script._creative_line(enabled=False).startswith("Bypassed")
        assert creative_script._spatial_line("", enabled=False).startswith("Bypassed")
        assert chain_script._stage2_summary(False, "", 0.35, 1.0).startswith("Bypassed")


class TestEveryDrawerStartsClosed:
    """The user-facing half of this refactor: a tab that unfolds four levels of
    settings the moment it is drawn is a tab nobody can see the top of."""

    def test_nothing_on_txt2img_builds_a_bare_accordion(self):
        """`drawer()` is the only way this extension makes a disclosure, which
        is what lets one rule style them all, one file remember which are open,
        and this test count them. A bare `gr.Accordion` gets none of that."""
        for name in SURFACES:
            source = (ROOT / name).read_text(encoding="utf-8")
            if name == "mc_pipeline_panel.py":
                # The one that wraps it.
                assert source.count("gr.Accordion(") == 1
                continue
            assert "gr.Accordion(" not in source, name

    def test_a_drawer_cannot_be_opened_at_build_time(self):
        """Not a default that a caller may override: there is no parameter."""
        import inspect

        signature = inspect.signature(mc_pipeline_panel.drawer)

        assert "open" not in signature.parameters

    def test_a_drawer_is_marked_as_this_extension_s(self, host):
        made = mc_pipeline_panel.drawer("Anything")

        assert made.open is False
        assert mc_pipeline_panel.DRAWER in made.elem_classes

    def test_a_caller_s_own_classes_survive(self, host):
        made = mc_pipeline_panel.drawer("Anything", elem_classes=["mine"])

        assert mc_pipeline_panel.DRAWER in made.elem_classes
        assert "mine" in made.elem_classes

    def test_the_shell_itself_starts_closed(self, host):
        pipeline = mc_pipeline_panel.host()

        assert pipeline.accordion.open is False
        assert mc_pipeline_panel.DRAWER in pipeline.accordion.elem_classes


class TestTheStagesShipOff:
    """A fresh Forge with this extension installed generates exactly as it would
    without it, until somebody says otherwise."""

    def test_creative_is_off_before_anybody_touches_it(self, store):
        import mc_creative_krea

        assert mc_creative_krea.settings()["enabled"] is False

    def test_spatial_is_off_before_anybody_touches_it(self, store):
        import mc_spatial

        assert mc_spatial.settings()["enabled"] is False

    def test_the_shipped_library_does_not_arm_creative_mode(self):
        """The one default that is not in this repository's Python: the
        creativity package can ask for Creative Mode to start on, and the
        settings honour it. The package this build ships must not."""
        found = json.loads(
            (ROOT / "prompt_master" / "krea" / "creativity"
             / "defaults.json").read_text(encoding="utf-8"))

        assert found.get("creative_mode_enabled") is False

    def test_stage_2_is_off_before_anybody_touches_it(self):
        source = (ROOT / "scripts" / "model_chain.py").read_text(encoding="utf-8")
        head = source.split('with pipeline.head("stage2"):', 1)[1].split(
            "with pipeline.body", 1)[0]

        assert "value=False" in head


# --------------------------------------------------------------------------- #
# Creative: one level, four drawers
# --------------------------------------------------------------------------- #


class TestTheCreativeHierarchy:
    """§1's Creative hierarchy change, and the final acceptance statement's
    first clause: the Profile is top-level, and Create a profile, Directions,
    Advanced settings and Recovery & diagnostics are same-level sibling drawers
    with nesting only inside their opened contents.

    Directions used to *be* the body of the panel -- twenty axis rows, a
    heading, a cost line and an Add dropdown, all unfolded the moment Creative
    was expanded, with Settings tucked in an accordion underneath. So the first
    screen of the stage was a list of axes nobody had asked about yet."""

    @pytest.fixture
    def source(self):
        return (ROOT / "mc_creative_panel.py").read_text(encoding="utf-8")

    def test_the_four_drawers_are_the_four_the_intent_names(self, source):
        found = re.findall(r'drawer\(\s*\n?\s*"([^"]+)"', source)
        surface = (ROOT / "scripts" / "model_chain_krea_creative.py").read_text(
            encoding="utf-8")
        found += re.findall(r'drawer\(\s*\n?\s*"([^"]+)"', surface)[:1]

        assert "Create a profile" in found
        assert "Advanced settings" in found

    def test_they_are_built_at_one_level(self, source):
        """Same left edge, same treatment. The way this regresses is a drawer
        opened inside another drawer's `with` block, which indents it."""
        body = source.split("def build(", 1)[1]
        opens = [line for line in body.splitlines()
                 if "mc_pipeline_panel.drawer(" in line]

        assert len(opens) >= 3
        indents = {len(line) - len(line.lstrip()) for line in opens}

        assert indents == {4}

    def test_the_profile_and_creativity_are_above_all_of_them(self, source):
        body = source.split("def build(", 1)[1]

        assert body.index("panel.profile = gr.Dropdown") \
            < body.index("panel.creativity = ") \
            < body.index("mc_pipeline_panel.drawer(")

    def test_directions_says_how_many_are_active(self):
        import mc_creative_panel

        assert mc_creative_panel.directions_label(()) == "Directions"
        assert mc_creative_panel.directions_label(("medium",)) == "Directions — 1 active"
        assert mc_creative_panel.directions_label(
            ("medium", "lighting")) == "Directions — 2 active"

    def test_the_count_is_derived_and_not_written(self, store, host):
        """§4's rule about summaries, applied to a drawer label: it is a
        selector over canonical state, not a value somebody's click set."""
        import mc_creative_krea
        import mc_creative_panel

        panel = mc_creative_panel.build(
            lambda *parts: "test-" + "-".join(str(part) for part in parts),
            lambda text, kind="info": text, None,
            stored=dict(mc_creative_krea.settings(), directions=["medium"]))
        if panel is None:
            pytest.skip("the creativity library did not load")

        assert panel.directions.label == "Directions — 1 active"

        updates = panel.render(stored=dict(mc_creative_krea.settings(),
                                           directions=["medium", "lighting"]))
        at = panel.outputs().index(panel.directions)

        assert updates[at]["label"] == "Directions — 2 active"

    def test_the_recovery_drawer_holds_what_the_intent_lists(self):
        surface = (ROOT / "scripts" / "model_chain_krea_creative.py").read_text(
            encoding="utf-8")
        block = surface.split('drawer(\n                        "Recovery & diagnostics"',
                              1)[1].split("# -- Spatial", 1)[0]

        for held in ("Continue from a pasted image", "Last creative roll",
                     "How Creative Mode reads your prompt"):
            assert held in block


# --------------------------------------------------------------------------- #
# What was opened stays opened
# --------------------------------------------------------------------------- #


class TestTheDrawerMemory:
    """A default is the right answer exactly once. After that, a tab that folds
    everything away again on every reload is a tab somebody has to re-open four
    drawers in before they can carry on."""

    @pytest.fixture
    def code(self):
        return PIPELINE_JS.read_text(encoding="utf-8")

    def test_it_remembers_per_browser_and_not_per_generation(self, code):
        """Furniture, and stored where furniture belongs. Nothing about a
        generation is kept here and nothing here is ever sent anywhere."""
        assert "localStorage" in code
        assert "modelChainOpenDrawers" in code

    def test_a_blocked_store_is_not_an_error(self, code):
        """Private browsing, a blocked store, a value somebody edited by hand.
        Every one of them means "no preferences", which is what this shipped
        with -- so every read and every write is inside a try."""
        block = code.split("function opened()", 1)[1].split("function remember(", 1)[0]

        assert "try {" in block
        assert "catch" in block

        block = code.split("function remember(", 1)[1].split("function isOpen(", 1)[0]

        assert "try {" in block
        assert "catch" in block

    def test_restoring_is_a_press_and_not_a_class(self, code):
        """Gradio owns whether an accordion is open and re-renders it from its
        own state. Setting the class would leave the two disagreeing at the
        first update; pressing the header is what the user would have done."""
        block = code.split("function watchDrawers()", 1)[1]

        assert "header.click()" in block
        assert "classList.add" not in block

    def test_it_only_finds_drawers_this_extension_built(self, code):
        """The class comes from mc_pipeline_panel.drawer(), so this cannot pick
        up a host accordion, a theme's own furniture, or a Column that happens
        to start with a button."""
        block = code.split("function drawers()", 1)[1].split("function watchDrawers",
                                                             1)[0]

        assert "DRAWER" in block

    def test_it_binds_once_however_often_gradio_rebuilds(self, code):
        block = code.split("function watchDrawers()", 1)[1]

        assert "dataset.mcPipelineDrawer" in block


# --------------------------------------------------------------------------- #
# Two presses for something that cannot be undone
# --------------------------------------------------------------------------- #


class TestDeleteAsksFirst:
    """§3: a destructive action should require an explicit confirmation where
    the loss is irreversible. Deleting a saved profile, a Stage 2 preset or a
    named Spatial layout removes a file, and nothing brings it back."""

    def test_the_first_press_arms_and_the_second_deletes(self):
        go, armed, button = mc_pipeline_panel.confirmed(False)

        assert go is False
        assert armed is True
        assert button["value"] == mc_pipeline_panel.CONFIRM

        go, armed, button = mc_pipeline_panel.confirmed(True)

        assert go is True
        assert armed is False
        assert button["value"] == "Delete"

    def test_the_confirmation_is_the_button_and_not_a_dialog(self):
        """A modal would be a second thing to dismiss on a tab that has enough
        of them, and a browser confirm() is not styleable, not themeable and
        not touch-friendly."""
        for name in ("mc_creative_panel.py", "scripts/model_chain.py",
                     "scripts/model_chain_krea_creative.py"):
            source = (ROOT / name).read_text(encoding="utf-8")

            assert "window.confirm" not in source
            assert "gr.Warning" not in source

    def test_a_creative_profile_survives_one_press(self, store, host):
        import mc_creative_profiles as profiles

        profiles.save("Keep me", profiles.from_settings())
        panel = _panel()
        if panel is None:
            pytest.skip("the creativity library did not load")
        handler = _clicked(panel, "profile-delete")

        armed, _button, *_rest = handler("Keep me", False)

        assert armed is True
        assert "Keep me" in profiles.choices()

        armed, _button, *_rest = handler("Keep me", armed)

        assert armed is False
        assert "Keep me" not in profiles.choices()

    def test_the_armed_flag_is_per_browser_and_not_per_process(self):
        """A module-level flag would be shared by every tab open on the server,
        so one person's half-finished gesture would arm somebody else's button."""
        source = (ROOT / "mc_creative_panel.py").read_text(encoding="utf-8")

        assert "gr.State(False)" in source
        assert "panel.arm_delete" in source

    def test_every_delete_on_the_tab_goes_through_it(self):
        """Three files, three delete buttons, one guard. A fourth added without
        it is a file somebody loses to a mis-tap."""
        for name in ("mc_creative_panel.py", "scripts/model_chain.py",
                     "scripts/model_chain_krea_creative.py"):
            source = (ROOT / name).read_text(encoding="utf-8")

            assert "mc_pipeline_panel.confirmed(" in source, name


# --------------------------------------------------------------------------- #
# The panel under somebody else's theme
# --------------------------------------------------------------------------- #


class TestThePipelineStylesheet:
    """The bug this class exists for: the first pass gave the stage cards a
    background taken from `--panel-background-fill`, and on stock Gradio that
    painted every card white while the text stayed the light colour a dark page
    asks for. A card nobody could read, on the theme most people use.

    The lesson is not "pick a better variable". A fill and a text colour taken
    from two different host variables are two guesses that have to agree, and an
    extension has no standing to make either of them."""

    @pytest.fixture
    def section(self):
        css = (ROOT / "style.css").read_text(encoding="utf-8")
        block = css.split("The semantic token adapter.", 1)[1]
        block = block.split("*/", 1)[1].split("/* -- the treatment rows", 1)[0]
        return re.sub(r"/\*.*?\*/", "", block, flags=re.S)

    @pytest.fixture
    def rules(self, section):
        found = []
        for part in section.split("}"):
            if "{" not in part:
                continue
            selector, body = part.rsplit("{", 1)
            found.append((" ".join(selector.split()), body))
        return found

    def test_no_host_fill_variable_is_read_at_all(self, section):
        """The whole class of bug, named. Any one of these can be light while
        `--body-text-color` is light, and an extension has no way to know which
        way round a given theme has them."""
        for banned in ("--panel-background-fill", "--block-background-fill",
                       "--background-fill-secondary", "--background-fill-primary",
                       "--body-background-fill", "--input-background-fill"):
            assert banned not in section, banned

    def test_the_surfaces_are_outlines_and_not_fills(self, rules):
        """Cards, drawers and bodies paint nothing. What is left painting a
        background is the rail, the elbow and the node -- one or two pixels
        each, in a border colour, with no text on top of them."""
        surfaces = (".mc-pipeline-stage", ".mc-pipeline-drawer",
                    ".mc-pipeline-editor", ".mc-pipeline-body",
                    ".mc-krea-creative-direction")
        for selector, body in rules:
            if "::" in selector:
                continue
            if not any(part in selector for part in surfaces):
                continue
            for line in body.splitlines():
                said = line.strip()
                if said.startswith("background"):
                    assert "none" in said, (selector, said)

    def test_it_states_no_colour_of_its_own(self, rules):
        """Every colour is one of the host's custom properties, which is what
        lets a theme -- Lobe, stock Gradio, or anything else -- decide what all
        of this looks like."""
        for selector, body in rules:
            for line in body.splitlines():
                said = line.strip()
                if ":" not in said:
                    continue
                assert not re.search(r":\s*#[0-9a-fA-F]{3,8}\s*[;!]", said), \
                    (selector, said)
                assert not re.search(r":\s*rgba?\(", said), (selector, said)

    def test_the_header_is_found_structurally(self, section):
        """A Gradio Accordion is two elements: the thing you press and the thing
        it shows. `> :first-child` is the header under every theme and every
        version, and it names no class of Gradio's.

        The first pass had the browser file move the switch and the summary into
        whatever header it could recognise, which worked under Lobe -- where the
        header is a <button> -- and found nothing at all under stock Gradio."""
        assert ".mc-pipeline-editor > :first-child" in section
        assert ".mc-pipeline-drawer > :first-child" in section
        for generated in ("label-wrap", "svelte-", "gradio-"):
            assert generated not in section

    def test_the_summary_and_the_switch_are_painted_into_the_header_band(self,
                                                                        section):
        """Both are siblings of the accordion in the page, so when the card is
        open the body would otherwise push them below everything it contains."""
        for name in ("summary", "switch"):
            rule = section.split(f".mc-pipeline-stage > .mc-pipeline-{name} {{",
                                 1)[1].split("}", 1)[0]
            assert "position: absolute" in rule, name

    def test_the_band_reserves_room_for_both(self, section):
        """The switch is painted over the header, so the header's own padding is
        what stops the stage's name running underneath it."""
        rule = section.split(".mc-pipeline-editor > :first-child {", 1)[1].split(
            "}", 1)[0]

        assert "--mc-pipe-lane" in rule
        assert "--mc-pipe-head" in rule

    def test_the_nesting_rail_has_its_elbow(self, section):
        """§2's fifth invariant, and the half the first pass left out."""
        assert ".mc-pipeline-body::before" in section
        assert ".mc-pipeline-body::after" in section

    def test_sibling_sections_are_separated(self, section):
        """§2's third invariant: a drawer's contents sit under its own header
        with a rule between them, so a long body is not one undivided run."""
        rule = section.split(
            ".mc-pipeline-drawer > :last-child:not(:first-child) {", 1)[1].split(
            "}", 1)[0]

        assert "border-top" in rule

    def test_the_enable_switch_is_a_target_and_not_a_tick(self, section):
        """It is the one control on a collapsed pipeline. The default checkbox
        is thirteen pixels of it in the corner of a card."""
        rule = section.split(".mc-pipeline-switch label {", 1)[1].split("}", 1)[0]

        assert "min-height: var(--mc-pipe-tap)" in rule
        assert "min-width" in rule

    def test_a_coarse_pointer_gets_forty_four_pixels(self, section):
        coarse = section.split("@media (pointer: coarse)", 1)[1].split("}", 1)[0]

        assert "--mc-pipe-tap: 44px" in coarse

    def test_the_composition_mode_is_two_large_targets(self, section):
        """§3 asks for large segmented targets rather than two radio dots. It is
        still the same stock Radio underneath, wearing the shape."""
        rule = section.split(".mc-pipeline-segmented label {", 1)[1].split("}", 1)[0]

        assert "min-height: var(--mc-pipe-tap)" in rule

        surface = (ROOT / "scripts" / "model_chain_krea_creative.py").read_text(
            encoding="utf-8")

        assert 'mc_pipeline_panel.classes("segmented")' in surface

    def test_an_armed_stage_says_so_without_relying_on_colour(self, section):
        """§2: never communicate enabled/bypassed status using only colour. The
        pill is outlined *and* the box inside it is ticked."""
        rule = section.split(".mc-pipeline-switch label:has(input:checked) {",
                             1)[1].split("}", 1)[0]

        assert "box-shadow: inset" in rule


class TestTheHeaderLookup:
    """What is left of the browser file's interest in a disclosure's header:
    pressing it to restore a drawer somebody had open."""

    @pytest.fixture
    def code(self):
        return PIPELINE_JS.read_text(encoding="utf-8")

    def test_it_counts_children_rather_than_naming_a_class(self, code):
        block = code.split("function headerOf(", 1)[1].split("\n    }", 1)[0]

        assert "children.length !== 2" in block
        assert "BUTTON" not in block
        assert "aria-expanded" not in block

    def test_nothing_is_moved_in_the_page_any_more(self, code):
        """The layout is the stylesheet's job. It can select "the accordion's
        first child" without any of this, and the move that could not was what
        broke the panel on stock Gradio."""
        assert "appendChild" not in code
        assert "furnish" not in code


class TestBothScriptsFillOneShell:
    """The property that makes the shared shell safe: it does not matter which
    script Forge builds first.

    Forge sorts its alwayson scripts by a key this extension does not choose,
    and a refactor that only worked in one of the two orders would work until
    somebody installed an extension whose name sorted between them.
    """

    def build(self, order):
        import model_chain
        import model_chain_krea_creative

        made = {"stage2": model_chain.ScriptModelChain,
                "creative": model_chain_krea_creative.ScriptKreaCreative}
        built = {}
        for name in order:
            instance = made[name]()
            built[name] = (instance, instance.ui(is_img2img=False))
        return mc_pipeline_panel.host(), built

    @pytest.mark.parametrize("order", [("stage2", "creative"),
                                       ("creative", "stage2")])
    def test_every_owned_stage_is_filled_whichever_script_runs_first(self, order,
                                                                     host, store):
        pipeline, _built = self.build(order)

        assert pipeline.filled == set(mc_pipeline_panel.OWNED)

    @pytest.mark.parametrize("order", [("stage2", "creative"),
                                       ("creative", "stage2")])
    def test_there_is_only_ever_one_shell(self, order, host, store):
        pipeline, _built = self.build(order)

        assert mc_pipeline_panel.host() is pipeline

    def test_the_switch_on_a_row_is_the_control_the_generation_reads(self, host,
                                                                     store):
        """Not a copy of it. A second checkbox mirroring the real one is the
        duplicate source-of-truth section 2.5 forbids, and it would be the one
        a user reached for first.

        Checked by identity against the argument lists, which are what the
        processing hooks actually receive: Stage 2's switch is the first
        argument of one, and Creative's and Spatial's are the first and the
        last-but-two of the other.
        """
        _pipeline, built = self.build(("stage2", "creative"))
        _stage2, stage2_arguments = built["stage2"]
        creative, _creative_arguments = built["creative"]

        assert creative.arguments[0] is creative.components["enabled"]
        assert creative.arguments[-3] is creative.components["spatial_enabled"]

        # The three switches are three distinct components, one per owned
        # stage, and none of them is any other stage's.
        switches = {id(stage2_arguments[0]),
                    id(creative.arguments[0]),
                    id(creative.arguments[-3])}
        assert len(switches) == 3


# --------------------------------------------------------------------------- #
# The Stage 1 context row
# --------------------------------------------------------------------------- #


class TestTheHandoffLine:
    """Section 7, and all that is left of the Stage 1 row: right about Hires.

    Stage 2 refines *finished Stage 1 pixels*, so a Hires pass changes what it
    is handed, and nothing else on the page states that number -- which is the
    whole reason this line outlived the row it used to sit under.
    """

    def helpers(self):
        import model_chain

        return model_chain

    def test_without_hires_the_handoff_is_the_requested_size(self):
        mc = self.helpers()

        assert mc._stage1_size(1024, 1536) == (1024, 1536)

    def test_with_hires_the_handoff_is_the_upscaled_size(self):
        mc = self.helpers()

        assert mc._stage1_size(1024, 1536, True, 1.5) == (1536, 2304)

    def test_a_checkpoint_is_named_without_its_path_or_its_hash(self):
        mc = self.helpers()

        assert mc._short_checkpoint("SD/krea2.safetensors [abc123]") == "krea2"

    def test_the_handoff_line_says_the_size_even_with_stage_2_off(self):
        """It is what Stage 2 *would* be handed, and the number somebody needs
        in order to decide whether to arm it."""
        mc = self.helpers()

        said = mc._handoff_summary(1024, 1536, True, 1.5, enabled=False)

        assert "1536 × 2304 pixel handoff" in said
        assert "Stage 2 is bypassed" in said

    def test_a_size_nobody_has_set_yet_says_nothing_rather_than_zero(self):
        mc = self.helpers()

        assert "pixel handoff" in mc._handoff_summary(0, 0)

    def test_the_rows_that_described_forges_own_work_are_gone(self):
        """Deleted rather than left unused: a builder nothing calls is a
        builder somebody wires back up by accident."""
        mc = self.helpers()

        assert not hasattr(mc, "_stage1_summary")
        assert not hasattr(mc, "_output_summary")


# --------------------------------------------------------------------------- #
# Loaded, modified, not saved
# --------------------------------------------------------------------------- #


class TestTheProfileStateContract:
    """Section 8, and 8.3 is the one that matters.

    "Not saved" is one word away from "not applied", and the two readings are
    opposite in consequence: a user who reads the second one presses Save
    before every generation, or does not, and believes the image came from
    settings it did not.
    """

    def test_nothing_loaded_says_nothing(self):
        assert mc_profile_state.describe("", False) == ""

    def test_a_loaded_profile_is_named(self):
        assert mc_profile_state.describe("Editorial", False) == "**Loaded:** Editorial"

    def test_an_edited_profile_says_modified_and_not_saved(self):
        said = mc_profile_state.describe("Editorial", True)

        assert "Editorial" in said
        assert "Modified" in said
        assert "not saved" in said

    def test_the_panel_says_that_unsaved_settings_are_still_the_active_ones(self):
        """Section 8.3, said on the panel rather than in documentation -- the
        moment somebody needs it is the moment they are reading the word
        "unsaved" and deciding whether to trust the screen."""
        said = mc_profile_state.explain(True)

        assert "next Generate will use" in said
        assert mc_profile_state.explain(False) == ""

    def test_a_stage_that_is_off_still_names_what_it_has_loaded(self):
        said = mc_profile_state.describe("Editorial", True, active=False)

        assert "Editorial" in said and "stage is off" in said

    def test_nothing_loaded_is_not_the_same_as_everything_modified(self):
        """A freshly opened tab has diverged from nothing, because it came from
        nothing."""
        assert mc_profile_state.changed({"a": 1}, "") is False

    def test_a_value_that_survived_json_compares_equal(self):
        """Gradio's transport turns tuples into lists. A dirty flag that
        reported a round trip as an edit would always be on."""
        assert mc_profile_state.changed({"a": [1, 2]},
                                        mc_profile_state.snapshot({"a": (1, 2)})) is False

    def test_a_real_edit_is_reported(self):
        baseline = mc_profile_state.snapshot({"denoise": 0.35})

        assert mc_profile_state.changed({"denoise": 0.4}, baseline) is True

    def test_key_order_is_not_an_edit(self):
        baseline = mc_profile_state.snapshot({"a": 1, "b": 2})

        assert mc_profile_state.changed({"b": 2, "a": 1}, baseline) is False


class TestSpatialLayoutsAreNotTheWorkingLayout:
    """Section 8.5, which is the one place the contract needs saying twice.

    Auto Save commits the *working* layout -- what the next Generate composes.
    Save updates a *named* layout. "Loaded: Studio thirds, Modified, not saved"
    is an ordinary state to sit in for as long as you like.
    """

    @pytest.fixture
    def data(self, tmp_path, monkeypatch, host):
        from modules import paths

        monkeypatch.setattr(paths, "data_path", str(tmp_path), raising=False)
        return tmp_path

    def layout(self, regions=(), mode="smart"):
        return json.dumps({"version": 1,
                           "canvas": {"width": 1024, "height": 1344, "grid": "none"},
                           "compose_mode": mode, "auto_position_hint": True,
                           "regions": list(regions)})

    def test_a_saved_layout_comes_back(self, data):
        import mc_spatial_profiles

        mc_spatial_profiles.save("Studio thirds", self.layout())

        assert mc_spatial_profiles.get("Studio thirds") == self.layout()

    def test_an_untouched_layout_matches_the_name_it_came_from(self, data):
        import mc_spatial_profiles

        mc_spatial_profiles.save("Studio thirds", self.layout())

        assert mc_spatial_profiles.matches("Studio thirds", self.layout()) is True

    def test_a_moved_box_no_longer_matches(self, data):
        import mc_spatial_profiles

        region = {"id": "r1", "name": "Face", "type": "obj",
                  "bbox": [10, 10, 200, 200], "prompt": "", "z": 0}
        mc_spatial_profiles.save("Studio thirds", self.layout([region]))
        moved = dict(region, bbox=[40, 10, 230, 200])

        assert mc_spatial_profiles.matches("Studio thirds", self.layout([moved])) is False

    def test_opening_a_layout_at_a_different_size_is_not_an_edit(self, data):
        """The frame is a fact about txt2img rather than part of the
        composition, so changing the generation size must not report every
        saved layout as modified."""
        import mc_spatial_profiles

        mc_spatial_profiles.save("Studio thirds", self.layout())
        wider = json.loads(self.layout())
        wider["canvas"]["width"] = 1536

        assert mc_spatial_profiles.matches("Studio thirds",
                                           json.dumps(wider)) is True

    def test_deleting_a_named_layout_leaves_the_working_one_alone(self, data):
        """Deleting a saved copy of a composition is not a request to stop
        using it."""
        import mc_spatial
        import mc_spatial_profiles

        mc_spatial_profiles.save("Studio thirds", self.layout())
        mc_spatial.remember(**{mc_spatial.LAYOUT: self.layout()})
        mc_spatial_profiles.delete("Studio thirds")

        assert mc_spatial.settings()["layout"] == self.layout()

    def test_a_layout_cannot_be_saved_without_a_name(self, data):
        import mc_spatial_profiles

        with pytest.raises(mc_spatial_profiles.LayoutError):
            mc_spatial_profiles.save("   ", self.layout())

    def test_the_reserved_name_is_refused(self, data):
        import mc_spatial_profiles

        with pytest.raises(mc_spatial_profiles.LayoutError):
            mc_spatial_profiles.save(mc_spatial_profiles.NONE, self.layout())

    def test_a_damaged_store_reads_as_empty_rather_than_raising(self, data):
        import mc_spatial_profiles

        Path(mc_spatial_profiles.path()).write_text("{not json", encoding="utf-8")

        assert mc_spatial_profiles.names() == []


# --------------------------------------------------------------------------- #
# The phase animation's one coupling
# --------------------------------------------------------------------------- #


def _phase_table() -> list[tuple[str, str]]:
    """The JS file's label table, read out of the JS file.

    Parsed rather than duplicated. A copy of the table here would agree with
    the one in the browser until somebody edited one of them, which is the
    failure this test exists to catch.
    """
    source = PIPELINE_JS.read_text(encoding="utf-8")
    block = re.search(r"const PHASES = \[(.*?)\];", source, re.S)
    assert block, "the pipeline JS no longer has a PHASES table"
    return re.findall(r'match:\s*"([^"]+)",\s*stage:\s*"([^"]+)"', block.group(1))


def _ambiguous() -> str:
    source = PIPELINE_JS.read_text(encoding="utf-8")
    found = re.search(r'const AMBIGUOUS = "([^"]+)"', source)
    assert found, "the pipeline JS no longer names its ambiguous label"
    return found.group(1)


def stage_for(label: str) -> str:
    """The browser's rule, in Python: lowercase, first substring match wins."""
    said = str(label or "").strip().lower()
    if not said:
        return ""
    if said.startswith(_ambiguous()):
        return "ambiguous"
    for match, stage in _phase_table():
        if match in said:
            return stage
    return ""


class TestEveryPhaseLightsARow:
    """The animation matches the progress bar's own text, which couples a
    Python constant to a JavaScript table.

    So the coupling is tested rather than hoped for. Every label the extension
    can put on the bar is listed here with the row it should light, and the
    matching is done by re-implementing the browser's rule over the browser's
    own table -- read out of the file, not copied into this one.
    """

    @pytest.mark.parametrize("label,stage", [
        ("Waiting for Stage 1 preload", "stage1"),
        ("Preparing Stage 1", "stage1"),
        ("Stage 1", "stage1"),
        ("Stage 1 1/2", "stage1"),
        ("Loading Klein 9B", "stage2"),
        ("Stage 2 model", "stage2"),
        ("Freeing VRAM for Stage 2", "stage2"),
        ("Stage 2", "stage2"),
        ("Stage 2 1/2", "stage2"),
        ("Finishing", "output"),
    ])
    def test_an_image_phase_lights_its_own_stage(self, label, stage):
        assert stage_for(label) == stage

    @pytest.mark.parametrize("label,stage", [
        (mc_llm_progress.READING, "creative"),
        (mc_llm_progress.WRITING, "creative"),
        (mc_llm_progress.COMPOSING_READ, "spatial"),
        (mc_llm_progress.COMPOSING_WRITE, "spatial"),
    ])
    def test_a_language_model_phase_lights_the_stage_that_asked_for_it(self, label,
                                                                      stage):
        """The writer is Creative's and the composer is Spatial's. They share a
        progress vocabulary and belong to different rows."""
        assert stage_for(label) == stage

    def test_the_shared_waiting_label_is_treated_as_ambiguous(self):
        """Both passes say it, so on its own it names no stage. The browser
        keeps whatever was already running rather than guessing."""
        assert stage_for(mc_llm_progress.WAITING) == "ambiguous"

    def test_every_label_the_planner_can_emit_is_matched(self):
        """Built from the real plan rather than from a list written here, so a
        phase added to mc_progress without a rule in the browser fails."""
        job = mc_progress.build(
            stage1_arch="flux", stage1_passes=[(20, 1.5)], batch_size=1,
            stage2_arch="flux", stage2_passes=[(20, 1.5)], transition="ram",
            move_gigabytes=6.0, free_gigabytes=2.0, target_label="Klein 9B")

        unmatched = [phase.label for phase in job.phases if not stage_for(phase.label)]

        assert unmatched == []

    def test_every_language_model_label_is_matched(self):
        for pass_ in (mc_llm_progress.WRITER, mc_llm_progress.COMPOSER):
            for label in pass_.labels().values():
                assert stage_for(label) != "", label

    def test_a_label_naming_no_stage_lights_nothing(self):
        """Silence is the right answer for a phase this file does not know
        about -- lighting the wrong row would be worse than lighting none."""
        assert stage_for("Doing something else entirely") == ""
        assert stage_for("") == ""

    def test_every_stage_the_table_names_is_one_this_file_knows(self):
        """A row, or one of the two phases that no longer has a row. What must
        never happen is a third answer: a stage name in the table that the
        panel has never heard of is a table that has gone stale."""
        known = set(mc_pipeline_panel.ORDER) | set(mc_pipeline_panel.PHASES_WITHOUT_A_ROW)

        for _match, stage in _phase_table():
            assert stage in known

    def test_a_phase_with_no_row_lights_nothing_and_is_still_recognised(self):
        """Stage 1 still runs; the panel simply has nothing to light for it.
        Recognised-and-silent and never-heard-of are different failures, and
        only the second one means the table needs updating."""
        assert stage_for("Stage 1 sampling") == "stage1"
        assert "stage1" not in mc_pipeline_panel.ORDER

    def test_the_browser_knows_the_same_six_stages_in_the_same_order(self):
        source = PIPELINE_JS.read_text(encoding="utf-8")
        found = re.search(r"const ORDER = \[(.*?)\];", source, re.S)
        assert found
        order = re.findall(r'"([^"]+)"', found.group(1))

        assert tuple(order) == mc_pipeline_panel.ORDER


class TestTheBrowserFileCannotAffectAGeneration:
    """The property the whole Creative Mode browser story turns on, defended
    one file further out.

    This file reflects two things and decides nothing. The way that could stop
    being true is the way it stopped being true before: something in the page
    that a generation has to wait for. So the two shapes of that are asserted
    against the source rather than argued about in a comment.
    """

    def source(self) -> str:
        return PIPELINE_JS.read_text(encoding="utf-8")

    def code(self) -> str:
        """The file with its comments removed.

        These tests are about what the file *does*, and the file explains
        itself at length -- including by naming the things it is careful not to
        use. Asserting over the prose would make a comment saying "never
        innerHTML" fail a test that innerHTML is never used.
        """
        return "\n".join(re.sub(r"//.*$", "", line)
                          for line in self.source().splitlines())

    def test_it_never_names_the_generate_button(self):
        """The old Creative gate swallowed the Generate click, polled a hidden
        textbox, and clicked Generate again once the server answered -- so a
        hidden tab made an image late and a closed one made it never arrive.
        Nothing here may go near that button."""
        assert "txt2img_generate" not in self.code()
        assert "img2img_generate" not in self.code()

    def test_it_arms_no_repeating_timer(self):
        """A one-shot that clears a finished state is a cosmetic settle. An
        interval is a poll, and a poll is the shape of something waiting."""
        assert "setInterval" not in self.code()

    def test_it_writes_no_gradio_component_value(self):
        """It reads the prompt box and the progress bar's text. Writing to a
        Gradio input from here would put a second author on a value the server
        believes it owns."""
        code = self.code()

        assert "updateInput" not in code
        assert ".dispatchEvent" not in code

    def test_the_prompt_is_echoed_as_text_and_never_as_markup(self):
        """A prompt is the string on the page most likely to contain angle
        brackets, and the echo is written straight into the panel."""
        code = self.code()

        assert "innerHTML" not in code
        # The echo assigns textContent and nothing else assigns into that
        # element at all, so the prompt cannot become markup on its way to the
        # panel however many angle brackets it contains.
        assert "target.textContent =" in code
        assert "target.innerHTML" not in code

    def test_no_gradio_generated_class_is_used_to_find_anything(self):
        """Every hook is an id this extension put in the page, or one of the
        host's own progress-bar classes. A theme is allowed to rearrange
        Gradio's internals; this has to keep working when it does."""
        code = self.code()

        for generated in ("gradio-", "svelte-", "label-wrap", "gr-button",
                          "block-label"):
            assert generated not in code
