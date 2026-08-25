"""Explanations that cost no space until somebody wants one.

This extension explains itself. Every panel carried a sentence or a paragraph
saying what the control below it does, and each one was written for the first
time somebody met it -- which is exactly once. After that they are prose
standing between a user and the settings they came for, on a tab where the
settings are the point and vertical space is what everything competes for.

    Spatial Layout: 7 regions. Region prompts are used exactly as typed.
    Direct BBOX Merge: your prompt is used exactly as typed as the global scene
    and your regions are applied deterministically. No language-model request is
    made — the fastest and most predictable Spatial option.

One number in that. The rest is a description of a mode that has not changed
since it was chosen, repeated on every render.

So the rule this module exists to apply: **live data stays on the panel,
description goes behind an "i"**. A line that answers "what is true right now"
-- 7 regions, 2 literals active, 1536 × 1536 pixel handoff -- is worth its
space. A line that answers "what does this mode mean" is worth a hover.

What it is
----------
A ``<span>`` with the explanation in an attribute, and CSS that shows it on
hover or focus. No JavaScript, no popup library, no Gradio-generated class: a
theme restyles it with everything else, and a build where the CSS did not load
still shows the text through the browser's own ``title`` tooltip.

    gr.Markdown(f"**Composition** {mc_hint.badge('what the two modes do')}")

Keyboard and touch reach it the same way: ``tabindex`` makes it focusable and
the CSS answers to ``:focus`` as well as ``:hover``, and ``title`` is what a
long-press shows on a phone.
"""

from __future__ import annotations

import html

PREFIX = "mc-hint"
"""Element-class stem for everything this module puts in the page."""

SYMBOL = "i"
"""What the badge says. A letter and not an emoji: an emoji is a font away from
being a coloured square, and this one has to line up with a text baseline."""


def badge(text: str, label: str = "") -> str:
    """One "i" carrying ``text``, as HTML to drop into any Markdown or HTML.

    ``label`` names what is being explained, for a screen reader that would
    otherwise read out "i". It never appears on screen.
    """
    said = " ".join(str(text or "").split())
    if not said:
        return ""
    described = str(label or "").strip()
    aria = f"About {described}" if described else "More about this"
    quoted = html.escape(said, quote=True)
    return (f'<span class="{PREFIX}" tabindex="0" role="note"'
            f' aria-label="{html.escape(aria, quote=True)}"'
            f' title="{quoted}" data-{PREFIX}="{quoted}">{SYMBOL}</span>')


def beside(heading: str, text: str, label: str = "") -> str:
    """``heading`` with an "i" after it, as one Markdown string.

    The heading is Markdown and the badge is HTML, which Markdown passes
    through -- so a caller writes ``beside("**Spatial Layout**", "...")`` and
    gets a heading that has not moved with an explanation that no longer takes
    a line of its own.
    """
    mark = badge(text, label or _plain(heading))
    return f"{heading} {mark}" if mark else heading


def line(data: str, text: str, label: str = "") -> str:
    """A live line, with the description that used to be part of it behind an "i".

    The half of the rule that is easy to get wrong: ``data`` is what changed
    since the last render and belongs on the panel, ``text`` is what has been
    true since the feature was written and does not.
    """
    said = str(data or "").strip()
    mark = badge(text, label)
    if not said:
        return mark
    return f"{said} {mark}" if mark else said


def _plain(heading: str) -> str:
    """A Markdown heading as the words in it, for the accessible name."""
    return " ".join(str(heading or "").replace("*", "").replace("#", "").split())
