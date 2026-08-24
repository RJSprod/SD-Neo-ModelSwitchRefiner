"""Named Spatial layouts: a composition you can put down and pick up again.

A layout is minutes of work with a mouse. Until now there was exactly one of
them -- the working canvas, persisted so a restart could not throw it away --
which meant that trying a different arrangement cost you the one you had.

This is the smallest thing that fixes that: a name, a serialized document, and
a file beside the Stage 2 presets. It deliberately does *not* introduce a
second layout system. What is stored is the same version-1 document the editor
writes and :mod:`prompt_master.krea.spatial` parses, byte for byte, so a saved
layout is a copy of the working one rather than a translation of it.

The distinction section 8.5 draws
---------------------------------
Two different saves live near each other here and must never be confused:

**The working layout** is what the next Generate composes. Dragging a box on
the compact canvas with Auto Save on commits it immediately, because a position
correction is not a decision worth a dialog.

**A named layout** is a copy somebody asked for by name. Nothing writes to it
except an explicit Save.

So ``Loaded: Studio thirds · Modified · not saved`` is an ordinary, correct
state to sit in for as long as you like: the boxes you just moved are the boxes
that will be composed, and *Studio thirds* still holds what it held. That is
:mod:`mc_profile_state`'s rule, applied to the one feature where the two saves
genuinely differ.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

FILENAME = "model_chain_spatial_layouts.json"
"""Beside the Stage 2 presets, in the WebUI data directory rather than in the
extension folder, so reinstalling or updating does not throw layouts away."""

SCHEMA_VERSION = 1

NONE = "None"
"""Shown in the dropdown when no named layout is loaded."""


class LayoutError(RuntimeError):
    """A save or delete that cannot be carried out.

    A distinct type because the panel turns it into one line on the page rather
    than a traceback in the console.
    """


def path() -> str:
    try:
        from modules import paths

        base = paths.data_path
    except Exception:
        base = os.getcwd()
    return os.path.join(base, FILENAME)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def _read() -> dict:
    """The whole store, tolerating a missing or damaged file."""
    file = path()
    if not os.path.exists(file):
        return {}

    try:
        with open(file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        logger.warning("Model Chain: could not read spatial layouts from %s; "
                       "treating it as empty", file, exc_info=True)
        return {}

    if not isinstance(data, dict):
        return {}
    layouts = data.get("layouts", {})
    if not isinstance(layouts, dict):
        return {}
    # Strings only. A layout is a serialized document; anything else in this
    # file is damage, and passing it to the editor would be passing damage on.
    return {name: value for name, value in layouts.items() if isinstance(value, str)}


def _write(layouts: dict) -> None:
    file = path()
    payload = {"version": SCHEMA_VERSION, "layouts": layouts}

    directory = os.path.dirname(file) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory,
            prefix=".model_chain_spatial_layouts", suffix=".tmp", delete=False)
        try:
            with handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(handle.name, file)
        except Exception:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise
    except Exception as exc:
        raise LayoutError(f"Could not save spatial layouts to {file}: {exc}") from exc


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


def names() -> list[str]:
    """Saved layout names, case-insensitively sorted for a stable dropdown."""
    return sorted(_read(), key=str.casefold)


def choices() -> list[str]:
    """Dropdown choices, with the no-selection entry first."""
    return [NONE] + names()


def get(name: str) -> str | None:
    """One layout's serialized document, or ``None`` when there is no such name."""
    if not name or name == NONE:
        return None
    return _read().get(name)


def save(name: str, serialized: str) -> list[str]:
    """Create or overwrite a named layout. Returns the refreshed name list."""
    name = (name or "").strip()
    if not name:
        raise LayoutError("Give the layout a name before saving.")
    if name == NONE:
        raise LayoutError(f'"{NONE}" is reserved and cannot be used as a layout name.')

    layouts = _read()
    existing = name in layouts
    layouts[name] = str(serialized or "")
    _write(layouts)

    logger.info("Model Chain: %s spatial layout %r",
                "updated" if existing else "saved", name)
    return names()


def delete(name: str) -> list[str]:
    """Remove a named layout. Returns the refreshed name list.

    The working layout is left exactly as it is. Deleting a saved copy of a
    composition is not a request to stop using it, and a delete that quietly
    emptied the canvas would be a destructive undo of work nobody asked to
    undo.
    """
    if not name or name == NONE:
        raise LayoutError("Select a layout to delete.")

    layouts = _read()
    if name not in layouts:
        raise LayoutError(f'No spatial layout named "{name}".')

    layouts.pop(name)
    _write(layouts)

    logger.info("Model Chain: deleted spatial layout %r", name)
    return names()


def matches(name: str, serialized: str) -> bool:
    """Whether the working layout still equals the named one it came from.

    Compared as parsed documents rather than as strings, because the editor
    rewrites key order and whitespace every time it serializes: a byte
    comparison would report a layout as modified for having been opened.
    """
    stored = get(name)
    if stored is None:
        return False
    return _document(stored) == _document(serialized)


def _document(serialized):
    """The comparable half of a layout, or the raw string if it will not parse."""
    try:
        parsed = json.loads(str(serialized or "") or "{}")
    except Exception:
        return str(serialized or "")
    if not isinstance(parsed, dict):
        return str(serialized or "")
    # The frame is a fact about txt2img rather than part of the composition, so
    # opening a layout at a different size is not a modification of it.
    return {
        "compose_mode": parsed.get("compose_mode"),
        "auto_position_hint": parsed.get("auto_position_hint"),
        "regions": parsed.get("regions"),
    }
