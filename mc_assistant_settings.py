"""The Forge Assistant's settings, and the reading of them.

Registered on the host's Settings page beside this extension's other options,
validated here as well as there. Both, and not either: a Gradio radio stores
whatever string it displayed, an older config can hold a value that is no longer
offered, and a hand-edited one can hold anything at all -- so what reaches the
browser is always one of the values this module names, never whatever was in the
file.

Nothing here is about a conversation. Turning the assistant off removes its
panel, its listeners and its focus treatment; it does not touch chats, models or
voice, and it deliberately does not turn off the guarded store or the
server-owned replies underneath. There is no "unsafe legacy writer" to fall back
to, which is the point of not having one.
"""

from __future__ import annotations

import logging

import mc_broker

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_ENABLED = "forge_assistant_enabled"
OPT_APPEARANCE = "forge_assistant_appearance"
OPT_LABEL = "forge_assistant_label"
OPT_ICON = "forge_assistant_icon"
OPT_ANCHOR = "forge_assistant_default_anchor"
OPT_COLUMN = "forge_assistant_column_width"
OPT_BUBBLE = "forge_assistant_bubble_width"
OPT_TEXT_SIZE = "forge_assistant_text_size"
OPT_DENSITY = "forge_assistant_density"
OPT_AVATARS = "forge_assistant_avatars"

APPEARANCES = ("text", "icon", "icon_text")
APPEARANCE_DEFAULT = "text"

ANCHORS = ("top-left", "top-center", "top-right",
           "bottom-left", "bottom-center", "bottom-right")
ANCHOR_DEFAULT = "bottom-right"
"""The six resting places, and no seventh.

Stored by name rather than as coordinates, which is the whole reason a window
resized between sessions cannot restore the launcher to somewhere off-screen.
"""

COLUMNS = (("narrow", 640), ("medium", 820), ("wide", 1040), ("full", 0))
COLUMN_DEFAULT = "medium"
"""How wide the conversation column is in the *tab*, in CSS pixels of content.

``full`` is no limit. The column is what makes focus mode an improvement for
conversation on a wide monitor rather than a regression: a transcript stretched
to 2,560 pixels is a transcript nobody can follow from one line to the next.
Never applied to the flyout, which is 360 pixels wide and has no such problem.
"""

BUBBLE_MIN, BUBBLE_MAX, BUBBLE_STEP = 60, 100, 5
BUBBLE_DEFAULT = 75
"""The widest a bubble may be, as a percentage of the thread's content width.

Below 480 CSS pixels at least 90 is used whatever this says, because a bubble
three-quarters of a phone screen wide with a quarter of empty gutter beside it
is a phone screen three-quarters used.
"""

LABEL_DEFAULT = "Forge Assistant"
MAX_LABEL = 256
"""A label, and a bound. Rendered as text and never as markup: it is a name for
a control, not a place to put HTML."""

ICONS = ("chat", "assistant", "spark", "compass", "dot")
ICON_DEFAULT = "chat"
"""Bundled icon names. Never a user-supplied SVG or HTML -- an icon somebody can
write is an icon somebody can put a script in."""

TEXT_SIZE_MIN, TEXT_SIZE_MAX = 14, 22
TEXT_SIZE_DEFAULT = 0
"""Zero means "inherit the host", which is what almost everybody wants."""

DENSITIES = ("comfortable", "compact")
DENSITY_DEFAULT = "comfortable"


def _one_of(value, allowed, default: str) -> str:
    found = str(value or "").strip().lower().replace(" ", "_")
    return found if found in allowed else default


def enabled() -> bool:
    return bool(mc_broker.option(OPT_ENABLED, True))


def appearance() -> str:
    return _one_of(mc_broker.option(OPT_APPEARANCE, APPEARANCE_DEFAULT), APPEARANCES,
                   APPEARANCE_DEFAULT)


def label() -> str:
    """The launcher's name. Empty means the default, never an empty button."""
    found = str(mc_broker.option(OPT_LABEL, LABEL_DEFAULT) or "").strip()
    return (found or LABEL_DEFAULT)[:MAX_LABEL]


def icon() -> str:
    return _one_of(mc_broker.option(OPT_ICON, ICON_DEFAULT), ICONS, ICON_DEFAULT)


def anchor() -> str:
    found = str(mc_broker.option(OPT_ANCHOR, ANCHOR_DEFAULT) or "").strip().lower()
    return found if found in ANCHORS else ANCHOR_DEFAULT


def column() -> str:
    return _one_of(mc_broker.option(OPT_COLUMN, COLUMN_DEFAULT),
                   tuple(name for name, _ in COLUMNS), COLUMN_DEFAULT)


def column_pixels() -> int:
    """The chosen column as a number. ``0`` is "no limit"."""
    return dict(COLUMNS).get(column(), 820)


def bubble_width() -> int:
    """The bubble maximum, clamped and snapped to the step it is offered in."""
    try:
        found = int(float(mc_broker.option(OPT_BUBBLE, BUBBLE_DEFAULT)))
    except (TypeError, ValueError):
        return BUBBLE_DEFAULT
    found = max(BUBBLE_MIN, min(BUBBLE_MAX, found))
    return int(round(found / BUBBLE_STEP) * BUBBLE_STEP)


def text_size() -> int:
    try:
        found = int(float(mc_broker.option(OPT_TEXT_SIZE, TEXT_SIZE_DEFAULT)))
    except (TypeError, ValueError):
        return TEXT_SIZE_DEFAULT
    if found <= 0:
        return TEXT_SIZE_DEFAULT
    return max(TEXT_SIZE_MIN, min(TEXT_SIZE_MAX, found))


def density() -> str:
    return _one_of(mc_broker.option(OPT_DENSITY, DENSITY_DEFAULT), DENSITIES,
                   DENSITY_DEFAULT)


def avatars() -> bool:
    return bool(mc_broker.option(OPT_AVATARS, True))


def describe() -> dict:
    """Every setting, validated, as the page is given them.

    One function so that "what the browser was told" is one thing to read and
    one thing to test -- and so a setting added here cannot be forgotten by the
    handler that publishes them.
    """
    return {"enabled": enabled(), "appearance": appearance(), "label": label(),
            "icon": icon(), "anchor": anchor(), "column": column(),
            "column_pixels": column_pixels(), "bubble_width": bubble_width(),
            "text_size": text_size(), "density": density(), "avatars": avatars()}


def style() -> str:
    """The settings that are presentation, as a stylesheet the page can carry.

    A handful of custom properties rather than a class per value: changing the
    column reloads nothing, resets no scroll and interrupts no reply, because
    all it does is give one variable a different number.
    """
    width = column_pixels()
    size = text_size()
    rules = [f"--forge-assistant-bubble: {bubble_width()}%;",
             f"--mc-llm-column: {width}px;" if width else "--mc-llm-column: none;"]
    if size:
        rules.append(f"--forge-assistant-text: {size}px;")
    return ":root{" + "".join(rules) + "}"
