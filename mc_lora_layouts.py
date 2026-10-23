"""MOSS-Image layout prompts.

MOSS-Image takes a plain-text prompt, or a *layout*: a JSON object whose summary
caption describes the whole picture and whose items place objects, characters and
lettering on the canvas. Forge Neo's JSON Builder compiles a layout into a single
line of JSON with its parentheses escaped - so Forge's prompt parser leaves them
alone - and that line is the prompt. Gallery prompts on Civitai arrive the same
way.

That line is what has to reach the prompt box, character for character, so
nothing here rewrites a prompt the library holds. This module only reads one, to
show what a layout depicts instead of a wall of JSON, and turns a layout the API
handed over as a parsed object back into the compiled line.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from mc_lora_common import truncate

ITEM_LABELS = ("object", "character", "text")

_ANY_ESCAPED_PAREN_RE = re.compile(r"\\*([()])")


@dataclass(frozen=True)
class LayoutItem:
    label: str
    caption: str
    text: str = ""


@dataclass(frozen=True)
class Layout:
    caption: str
    items: tuple[LayoutItem, ...] = ()
    meta: tuple[str, ...] = ()

    def describe(self) -> str:
        """One line naming what the layout places, e.g. ``2 characters, 1 text``."""
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.label] = counts.get(item.label, 0) + 1
        parts = []
        for label in (*ITEM_LABELS, *sorted(set(counts) - set(ITEM_LABELS))):
            count = counts.get(label, 0)
            if count:
                parts.append(f"{count} {label}{'s' if count != 1 and label != 'text' else ''}")
        return ", ".join(parts)


def escape_parens(text: str) -> str:
    """What Forge's JSON Builder does to a compiled layout: ``(`` becomes ``\\(``."""
    return _ANY_ESCAPED_PAREN_RE.sub(lambda match: "\\" + match.group(1), str(text))


def _text(value) -> str:
    return " ".join(str(value).split()) if isinstance(value, str) else ""


def from_object(value) -> Layout | None:
    """Read a layout - compiled, or still a draft - from parsed JSON."""
    if not isinstance(value, dict):
        return None

    summary = value.get("summary")
    caption = _text(summary.get("caption")) if isinstance(summary, dict) else ""
    if not caption and set(value) == {"caption"}:
        caption = _text(value.get("caption"))  # a "caption draft"

    items: list[LayoutItem] = []
    raw_items = value.get("items")
    if isinstance(raw_items, list):
        for raw in raw_items:
            if isinstance(raw, dict) and _text(raw.get("caption")):
                items.append(LayoutItem(_text(raw.get("label")) or "object", _text(raw.get("caption")), _text(raw.get("text"))))

    if not caption and not items:
        return None

    meta: tuple[str, ...] = ()
    if isinstance(summary, dict) and isinstance(summary.get("meta"), list):
        meta = tuple(text for text in (_text(entry) for entry in summary["meta"]) if text)
    return Layout(caption, tuple(items), meta)


def parse(prompt) -> Layout | None:
    """The layout a prompt holds, or ``None`` for an ordinary text prompt."""
    text = str(prompt or "").strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    # The escaped parentheses are not valid JSON; for reading, plain ones do.
    try:
        value = json.loads(_ANY_ESCAPED_PAREN_RE.sub(r"\1", text))
    except ValueError:
        return None
    return from_object(value)


def compile_object(value) -> str:
    """A layout handed over as an object, as the one-line prompt Forge uses.

    Returns ``""`` for anything that is not a layout.
    """
    if from_object(value) is None:
        return ""
    if set(value) == {"caption"}:
        return escape_parens(_text(value["caption"]))
    try:
        return escape_parens(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):
        return ""


def label(prompt, width: int = 90) -> str:
    """How a prompt is named in a list: a layout by its caption."""
    layout = parse(prompt)
    if layout is None:
        return truncate(prompt, width)
    return truncate(f"[layout] {layout.caption or layout.describe()}", width)
