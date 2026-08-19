"""Shared helpers for the LLM Studio panels.

Small on purpose. The three modes are separate products (section 4) and are
built by separate modules; what they genuinely have in common is the handful of
conversions below, plus one rule about element identifiers that section 5 turns
into a compatibility requirement.

Element identifiers and theming
-------------------------------
Section 5 asks for stable, extension-owned ids and classes, extension-scoped
CSS, and no selectors that depend on Gradio's generated DOM. :func:`ident` and
:func:`classes` are how that is kept honest: every id this extension puts in
the page starts with ``mc-llm-``, every class does too, and ``style.css``
scopes all of it under ``#mc-llm-studio``. A theme that restyles Gradio's own
containers -- Lobe restyles a great many of them -- changes how these panels
look and cannot change what they *are*, because nothing here is selected by
what Gradio happened to call it.

Colours are the other half of that. Nothing below, and nothing in the CSS,
hard-codes a palette: the panels inherit ``var(--body-text-color)``,
``var(--background-fill-primary)``, ``var(--border-color-primary)`` and the
rest of Gradio's own custom properties, which is what makes a theme switch --
or a light/dark toggle -- flow through without this extension being told.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

PREFIX = "mc-llm"

_GB = 1024**3
_MB = 1024**2


def ident(*parts: str) -> str:
    """A stable, extension-owned element id: ``mc-llm-<part>-<part>``."""
    return "-".join((PREFIX,) + tuple(str(part).strip("-") for part in parts if part))


def classes(*names: str) -> list[str]:
    """Extension-owned class names, for CSS that a theme cannot accidentally match."""
    return [f"{PREFIX}-{name}" for name in names]


def choices(options) -> list[tuple[str, str]]:
    """``prompt_engine.options`` pairs as Gradio choices.

    The two libraries order the pair oppositely -- upstream keeps
    ``(value, label)`` because the value is the engine's key and the label is
    display text, and Gradio takes ``(label, value)``. Converting here rather
    than restating the lists is section 5 of the port brief carried over: the
    UI must not carry its own copies of the engine's option lists.
    """
    return [(label, value) for value, label in options]


def grouped_choices(groups) -> list[tuple[str, str]]:
    """Grouped style options flattened, with the group name kept in the label.

    Gradio 4's Dropdown has no optgroup, so upstream's grouping survives as a
    prefix rather than as structure. Losing the heading entirely would lose
    real information -- several style keys only make sense read under their
    group -- and inventing a nested control to keep it would be a worse trade.
    """
    flattened: list[tuple[str, str]] = []
    for group, options in groups:
        for value, label in options:
            flattened.append((f"{group} · {label}" if group else label, value))
    return flattened


def data_url(path) -> str | None:
    """A picked image as the data URL the prompt engine wants, or ``None``.

    Gradio hands back a temporary file path; the vendored preprocessor does the
    EXIF transpose, the RGB conversion, the 768px thumbnail and the JPEG
    encoding -- all of which the engine's vision policy depends on, and none of
    which is re-implemented here.
    """
    if not path:
        return None
    from prompt_master.imaging.preprocess import image_data_url

    return image_data_url(Path(path))


def gigabytes(value: int | float) -> str:
    return f"{(value or 0) / _GB:.1f} GB"


def megabytes(value: int | float) -> str:
    return f"{(value or 0) / _MB:.0f} MB"


def tokens(value: int) -> str:
    return f"{int(value or 0):,}"


def escape(text: str) -> str:
    """HTML-escape text that is about to go into a status panel.

    Everything the panels put in HTML is either written by this extension or
    read out of a model file's metadata, and the second kind is somebody else's
    bytes. A GGUF's ``general.name`` is a free-text field.
    """
    import html

    return html.escape(str(text or ""))


def notice(text: str, kind: str = "info") -> str:
    """One line of status, as scoped HTML.

    ``kind`` is one of ``info``, ``warn`` or ``error`` and picks a class, never
    a colour: the colours live in ``style.css`` and are expressed against the
    host's own custom properties so a theme decides what "warn" looks like.
    """
    return (f'<div class="{PREFIX}-notice {PREFIX}-notice-{kind}">'
            f'{escape(text)}</div>')


def working(text: str, kind: str = "info") -> str:
    """A status line for something that is still happening.

    The same line :func:`notice` draws, plus one class. Everything the class
    turns on is drawn by ``style.css`` in space the line already occupies -- a
    two-pixel bar along its bottom edge -- because the one thing a progress
    indicator in a chat window must not do is push the conversation down every
    time a reply starts. It is deliberately indeterminate: nothing here knows
    how many tokens a reply will be, and a bar that fills to a number somebody
    invented is a worse answer than one that only claims the request is alive.

    Which lines get it is a decision about *truth*, not about decoration: a
    busy line is one a run is still working behind. "Reply complete." and
    "Cancelled." are not, and neither is an error -- a bar still sweeping under
    a finished run says the opposite of what the sentence beside it says.

    The text is wrapped rather than left bare so that the elapsed-time readout
    ``llm_studio.js`` adds has somewhere to sit that is not inside the
    sentence. Without that file the line reads exactly as it does here.

    The bar is an element rather than a ``::after`` on the line, for the reason
    the stylesheet gives about the host's own progress bar and its tests then
    hold the whole file to: a pseudo-element is somewhere a theme may already
    be drawing, and an element this extension created is somewhere no theme
    has ever heard of.
    """
    return (f'<div class="{PREFIX}-notice {PREFIX}-notice-{kind} {PREFIX}-busy" role="status">'
            f'<span class="{PREFIX}-busy-text">{escape(text)}</span>'
            f'<span class="{PREFIX}-busy-bar" aria-hidden="true"></span></div>')


def state(label: str, kind: str = "info", detail: str = "") -> str:
    """The runtime's state as one word, with the detail on hover.

    A chip rather than a sentence. This sits in the model sheet above the
    chooser, and it is also what the state control in the shell bar and in
    Conversation's header say in one word; the sentence it replaced --
    "Model: Q4_K_M · Device: NVIDIA GeForce RTX 3090 (24575 MiB, 23304 MiB
    free) · Server: stopped" -- had to be read to learn one thing anybody
    glances at it for, which is whether the model is up. Worse, it was long
    enough to wrap the whole bar into ten lines and push the conversation off
    the bottom of the window.

    The detail is not thrown away, it is moved: ``title`` puts it a hover away,
    Setup's residency view has all of it, and the console has every state
    change with a timestamp.
    """
    tooltip = f' title="{escape(detail)}"' if detail else ""
    return (f'<div class="{PREFIX}-state {PREFIX}-state-{kind}"{tooltip}>'
            f'<span class="{PREFIX}-state-dot"></span>{escape(label)}</div>')


def failure(exc: BaseException) -> str:
    """An exception as something a user can act on.

    Deliberately not a traceback. The vendored layers raise sentences -- "this
    request carries an image and the model running has no vision projector",
    "configured model is missing: ..." -- and those sentences are the useful
    output. The traceback still goes to the console, where it belongs.
    """
    logger.debug("Model Chain: LLM Studio operation failed", exc_info=True)
    text = str(exc).strip()
    return text or exc.__class__.__name__
