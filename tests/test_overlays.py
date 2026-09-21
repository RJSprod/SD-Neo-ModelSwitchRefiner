"""One overlay at a time, across all seven surfaces and in both directions.

LLM Studio had three independent ideas of "only one at a time": the shell's two
sheets knew about each other, Conversation's four screens knew about each other,
and the message action sheet knew about nothing at all. Each was correct on its
own, which is exactly why the failure was so easy to reach -- opening the
character editor left the workspace chooser underneath it, and tapping a bubble
opened an action sheet over whichever of those happened to be showing.

Specification O01 asks for all 21 ordered pairs. That is what
:meth:`TestExclusivity.test_every_ordered_pair_of_surfaces` does, from the one
function that decides, rather than from seven handlers that each have to
remember.
"""

from __future__ import annotations

import itertools

import pytest

import mc_llm_chat_panel
import mc_llm_overlays
import mc_llm_studio


@pytest.fixture(autouse=True)
def registry():
    mc_llm_overlays.reset()
    yield
    mc_llm_overlays.reset()


class Component:
    """Stands in for a Gradio Column. Only its identity is read."""

    def __init__(self, name):
        self.name = name


def build_registry():
    for name in mc_llm_overlays.OVERLAYS:
        mc_llm_overlays.register(name, Component(name))
    mc_llm_overlays.register_state("shell", Component("shell-state"))
    mc_llm_overlays.register_state("chat", Component("chat-state"))


class TestTheDecision:
    def test_opening_one_closes_every_other(self):
        for surface in mc_llm_overlays.OVERLAYS:
            showing = mc_llm_overlays.showing(surface)

            assert showing[surface] is True
            assert sum(1 for open_now in showing.values() if open_now) == 1, surface

    def test_a_name_it_has_never_heard_of_opens_nothing(self):
        """Better a menu that did not open than a screen drawn over another."""
        showing = mc_llm_overlays.showing("a surface nobody built")

        assert not any(showing.values())

    def test_closing_closes_everything(self):
        assert not any(mc_llm_overlays.showing("").values())

    def test_the_seven_are_the_seven(self):
        """A pop surface added without joining this tuple is a pop surface that
        opens over the others, which is the whole defect this replaced."""
        assert mc_llm_overlays.OVERLAYS == (
            "mode", "model", "threads", "character", "persona", "voice", "actions")
        assert set(mc_llm_studio.SHEETS) | set(mc_llm_chat_panel.SCREENS) | {"actions"} \
            == set(mc_llm_overlays.OVERLAYS)


class TestExclusivity:
    def test_every_ordered_pair_of_surfaces(self):
        """All 21, both ways round. O01."""
        pairs = list(itertools.permutations(mc_llm_overlays.OVERLAYS, 2))

        assert len(pairs) == 42, "seven surfaces make 42 ordered pairs"
        for opened, other in pairs:
            showing = mc_llm_overlays.showing(opened)

            assert showing[opened] is True
            assert showing[other] is False, f"{opened} left {other} open"

    def test_the_action_sheet_closes_the_others_and_is_closed_by_them(self):
        """The seventh surface, and the one that knew about none of the rest."""
        assert mc_llm_overlays.showing("actions")["threads"] is False
        assert mc_llm_overlays.showing("threads")["actions"] is False
        assert mc_llm_overlays.showing("model")["actions"] is False
        assert mc_llm_overlays.showing("actions")["model"] is False


class TestReachingGradio:
    def test_a_handler_is_given_every_surface_it_does_not_own(self):
        build_registry()

        others = mc_llm_overlays.closes(mc_llm_studio.SHEETS)

        assert set(others) == set(mc_llm_chat_panel.SCREENS) | {"actions"}

    def test_a_build_that_registered_nothing_asks_for_nothing(self):
        """A panel built on its own -- in a test, or on a host where one mode
        failed to build -- must not leave every handler in the other panel
        expecting an output that is not there."""
        assert mc_llm_overlays.closes(mc_llm_studio.SHEETS) == ()
        assert mc_llm_overlays.components(("mode", "threads")) == []

    def test_the_foreign_values_are_all_closed_and_of_two_shapes(self):
        build_registry()

        parts, values = mc_llm_overlays.foreign(mc_llm_studio.SHEETS, "shell")

        assert len(parts) == len(values)
        surfaces = [value for value in values if not isinstance(value, str)]
        states = [value for value in values if isinstance(value, str)]
        assert all(update.get("visible") is False for update in surfaces)
        assert states == [""], "the other group's State is closed too"

    def test_a_handler_does_not_close_its_own_state(self):
        """Each handler writes the *other* group's State and never its own --
        its own is the first value it already returns, and writing it twice
        would be two answers to one question in one list."""
        build_registry()
        shell_state = mc_llm_overlays.registered
        shell_parts, _ = mc_llm_overlays.foreign(mc_llm_studio.SHEETS, "shell")
        chat_parts, _ = mc_llm_overlays.foreign(mc_llm_chat_panel.SCREENS, "chat")

        shell_states = [part for part in shell_parts if part.name.endswith("-state")]
        chat_states = [part for part in chat_parts if part.name.endswith("-state")]

        assert [part.name for part in shell_states] == ["chat-state"]
        assert [part.name for part in chat_states] == ["shell-state"]
        assert shell_state is mc_llm_overlays.registered

    def test_a_handler_is_given_one_component_per_value_it_answers_with(self):
        """The two halves of one contract. A list one value short would put a
        visibility into a State and nothing would raise."""
        build_registry()

        for own, owner in ((mc_llm_studio.SHEETS, "shell"),
                           (mc_llm_chat_panel.SCREENS, "chat")):
            parts, values = mc_llm_overlays.foreign(own, owner)

            assert len(parts) == len(values)
            assert len(parts) == len(mc_llm_overlays.OVERLAYS) - len(own) + 1

    def test_the_registry_is_cleared_for_a_second_build(self):
        """A UI reload builds a second set of components, and a handler writing
        into the first set's would be writing into a page nobody is looking
        at."""
        build_registry()
        mc_llm_overlays.reset()

        assert mc_llm_overlays.registered("mode") is None
        assert mc_llm_overlays.closes(()) == ()


class TestThePanelsAgree:
    def test_the_chat_panel_answers_in_the_shape_its_outputs_expect(self):
        build_registry()

        answered = mc_llm_chat_panel._screens("threads")
        components = mc_llm_overlays.components(mc_llm_chat_panel.SCREENS) \
            + mc_llm_overlays.foreign(mc_llm_chat_panel.SCREENS,
                                      mc_llm_chat_panel.OVERLAY_OWNER)[0]

        assert len(answered) == 1 + len(components)
        assert answered[0] == "threads"

    def test_the_shell_answers_in_the_shape_its_outputs_expect(self):
        build_registry()

        answered = mc_llm_studio._sheet("model")
        components = mc_llm_overlays.components(mc_llm_studio.SHEETS) \
            + mc_llm_overlays.foreign(mc_llm_studio.SHEETS,
                                      mc_llm_studio.OVERLAY_OWNER)[0]

        assert len(answered) == 1 + len(components)
        assert answered[0] == "model"

    def test_opening_a_chat_screen_closes_the_shell_sheets(self):
        build_registry()
        order = list(mc_llm_chat_panel.SCREENS) \
            + list(mc_llm_overlays.closes(mc_llm_chat_panel.SCREENS))

        answered = mc_llm_chat_panel._screens("character")
        visibilities = answered[1:1 + len(order)]

        for name, update in zip(order, visibilities):
            assert update.get("visible") is (name == "character"), name

    def test_opening_a_shell_sheet_closes_the_chat_screens(self):
        build_registry()
        order = list(mc_llm_studio.SHEETS) \
            + list(mc_llm_overlays.closes(mc_llm_studio.SHEETS))

        answered = mc_llm_studio._sheet("mode")
        visibilities = answered[1:1 + len(order)]

        for name, update in zip(order, visibilities):
            assert update.get("visible") is (name == "mode"), name


class TestWhatIsDeliberatelyNotAnOverlay:
    def test_inline_disclosures_are_not_in_the_list(self):
        """The pipeline drawer, the creative drawers, the browse pickers, the
        character voice delivery group, the reference panel, the edit row and
        the rename row all push content down rather than covering it. Two of
        them can sensibly be open at once, and closing one because another
        opened would be a rule nobody asked for."""
        for name in ("pipeline", "drawer", "browse", "delivery", "reference", "edit",
                     "rename", "ooze"):
            assert name not in mc_llm_overlays.OVERLAYS

    def test_the_assistant_is_exempt_in_both_directions(self):
        """It is a second window onto the conversation, not another sheet over
        it. Opening a sheet does not close it and it does not close a sheet."""
        for surface in mc_llm_overlays.OVERLAYS:
            assert "assistant" not in surface

        assert "forge-assistant" not in " ".join(mc_llm_overlays.OVERLAYS)


class TestTheSpatialPopupStylesAreNotOrphans:
    """The specification's overlay audit calls the Krea spatial popup rules
    orphaned -- "real popup CSS, zero Python/JS references, left by the
    reverted Klein region geometry" -- and says to delete them separately.

    They are not orphaned. ``model_chain_krea_creative.spatial_editor()``
    emits all three classes today, and the reason a search for them finds
    nothing is that the markup builds the names in pieces rather than writing
    them out. Deleting the rules leaves live controls unstyled -- which is
    exactly what the existing spatial stylesheet test catches, and it is why
    this one is here as well: a second reader following the audit would
    otherwise delete them again.
    """

    def test_the_markup_still_emits_them(self):
        import re

        import model_chain_krea_creative as creative

        markup = creative.spatial_editor() + creative.spatial_compact()
        used = set()
        for group in re.findall(r'class="([^"]+)"', markup):
            used.update(name for name in group.split())

        assert "mc-krea-spatial-menu" in used
        assert "mc-krea-spatial-popup" in used

    def test_the_rules_are_still_there_for_them(self):
        from pathlib import Path

        css = (Path(__file__).resolve().parent.parent / "style.css").read_text(
            encoding="utf-8")

        assert ".mc-krea-spatial-menu {" in css
        assert ".mc-krea-spatial-popup {" in css
