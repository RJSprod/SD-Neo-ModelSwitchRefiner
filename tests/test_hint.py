"""Descriptions that cost no space until somebody wants one.

Every panel in this extension explained itself in a paragraph, and every one of
those paragraphs was written for the first time somebody met it -- which is
once. The rule this module applies is: a line that says what is true right now
stays on the panel; a line that says what a mode *means* goes behind an "i".

The tests are mostly about the boundary. A badge is HTML built from this
extension's own words, and it is going into lines that also carry a layout name
somebody typed -- so the one thing that must not happen is text from a user
arriving in a place that renders it as markup.
"""

from __future__ import annotations

import mc_hint


class TestTheBadge:
    def test_it_carries_the_text_in_an_attribute_and_a_title(self):
        """The attribute is what the stylesheet reads; `title` is what a build
        with no stylesheet, and a phone, still show."""
        mark = mc_hint.badge("What Direct BBOX Merge does")

        assert 'data-mc-hint="What Direct BBOX Merge does"' in mark
        assert 'title="What Direct BBOX Merge does"' in mark

    def test_it_is_reachable_without_a_mouse(self):
        mark = mc_hint.badge("something")

        assert 'tabindex="0"' in mark

    def test_it_names_what_it_explains_for_a_screen_reader(self):
        mark = mc_hint.badge("something", label="Spatial Layout")

        assert 'aria-label="About Spatial Layout"' in mark

    def test_nothing_at_all_is_no_badge(self):
        """A control with no explanation gets no "i" -- an empty bubble is a
        worse answer than no bubble."""
        assert mc_hint.badge("") == ""
        assert mc_hint.badge("   ") == ""
        assert mc_hint.badge(None) == ""

    def test_a_paragraph_becomes_one_line_of_attribute(self):
        mark = mc_hint.badge("one line\n   and   another")

        assert 'data-mc-hint="one line and another"' in mark


class TestItCannotBecomeMarkup:
    def test_quotes_and_brackets_are_escaped(self):
        mark = mc_hint.badge('a "quoted" <b>thing</b>')

        assert "<b>" not in mark
        assert '&lt;b&gt;' in mark
        assert '&quot;quoted&quot;' in mark

    def test_an_attribute_cannot_be_closed_early(self):
        """The failure this stands in front of: a description that ends the
        attribute and starts an event handler."""
        mark = mc_hint.badge('" onmouseover="alert(1)')

        assert 'onmouseover="alert' not in mark


class TestBeside:
    def test_the_heading_keeps_its_markdown(self):
        said = mc_hint.beside("**Spatial Layout**", "what it does")

        assert said.startswith("**Spatial Layout**")
        assert "mc-hint" in said

    def test_the_accessible_name_comes_from_the_heading(self):
        said = mc_hint.beside("**Spatial Layout**", "what it does")

        assert 'aria-label="About Spatial Layout"' in said

    def test_a_heading_with_nothing_to_explain_is_left_alone(self):
        assert mc_hint.beside("**Spatial Layout**", "") == "**Spatial Layout**"


class TestLine:
    """The half of the rule that is easy to get wrong."""

    def test_the_data_stays_on_the_line(self):
        said = mc_hint.line("Spatial Layout: 7 regions", "and what that means")

        assert said.startswith("Spatial Layout: 7 regions")

    def test_the_description_does_not(self):
        """It is inside the badge -- in both of its attributes -- and nowhere in
        what the panel draws."""
        said = mc_hint.line("Spatial Layout: 7 regions", "and what that means")
        visible = said.split("<span")[0]

        assert "and what that means" not in visible
        assert "and what that means" in said

    def test_a_line_with_no_data_is_just_the_badge(self):
        said = mc_hint.line("", "an explanation")

        assert said.startswith("<span")

    def test_a_line_with_no_explanation_is_just_the_data(self):
        assert mc_hint.line("7 regions", "") == "7 regions"
