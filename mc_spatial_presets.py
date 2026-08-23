"""Named Spatial Layouts: the boxes and the words in them, kept to be reused.

A composition is minutes of work with a mouse. Drawing seven boxes, naming them,
typing a prompt into each and choosing a framing for two of them is a document,
and until now the only copy of it was the one currently loaded -- draw the next
picture and it was gone. This is the store that makes the previous one
recallable.

What a preset is, and what it deliberately is not
-------------------------------------------------
A preset is **the regions**: each box, its name, its Object/Text type, the
visible text, the region prompt, the framing and camera-angle selections, and
the stacking order. That is the whole of it.

It is not the canvas size, because the frame belongs to txt2img and to the
generation somebody is about to make rather than to the layout -- coordinates
are normalized 0..1000 precisely so a composition drawn at 1024x1344 is the same
composition at 1536x2016 (``prompt_master.krea.spatial``, "Coordinates are
normalized"). Recalling a preset must never quietly change the size of the image
that is about to be generated.

It is not the compose mode, the auto-position-hint toggle or the grid either.
Those are panel and editor settings -- how this installation behaves right now --
and a preset that reached out and flipped Smart to Direct would be changing what
the Generate button does, which is the same line ``mc_creative_profiles`` draws
around Creative Mode's own enabled flag.

And it is not a Creative profile. The two stores are deliberately separate and
say so in both directions: a Creative profile describes how art direction
behaves and carries no layout; a layout describes one picture and carries no
Creativity position. Somebody who wants both keeps one of each.

Recall does not apply
---------------------
:func:`get` hands back regions. Nothing here writes to preferences, touches the
hidden state box or changes any generation. The browser loads a recalled preset
into the *editor*, where it is visible as boxes and prompts and can be adjusted,
undone or abandoned -- and it reaches a generation only when the user presses
Save & Return, exactly as anything else they drew would. That is the whole
difference between "recall a layout" and "replace my layout".

Why a file of its own
---------------------
The same reasoning as ``mc_creative_profiles``, which this follows closely
enough to be read beside it: one JSON file under the WebUI's data directory so
that updating the extension does not throw saved work away, written through a
temporary file and an atomic replace so an interrupted save leaves the previous
file intact rather than a truncated one. ``mc_llm_state.preferences()`` holds
what the panel is set to *now*; a list of complete named documents is a
different shape, and growing one inside a preferences file would make every drag
of a box rewrite everybody's saved compositions.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

FILENAME = "krea_spatial_presets.json"

SCHEMA_VERSION = 1
"""Bumped when the *shape* below changes, never when a field is added.

Regions are read through ``prompt_master.krea.spatial``'s own parser, which
already fills in what an older document does not carry, so adding a per-region
field is a change that costs nobody their saved compositions.
"""

MAX_PRESETS = 200
"""How many named layouts one installation may keep.

Not a limit anybody should reach by hand. It is here because this file is
written from a browser control, and a store with no ceiling is a store that a
stuck key or a loop in somebody's userscript can grow without bound.
"""

MAX_NAME = 80
"""Longest preset name. Long enough to describe a composition, short enough to
sit in a dropdown in the editor's top bar without becoming the top bar."""


class PresetError(RuntimeError):
    """A save, rename or delete that cannot be carried out.

    A distinct type for the same reason ``mc_creative_profiles.ProfileError`` is
    one: the editor turns it into a line of text in the workspace rather than a
    traceback. "That name is already taken", "there is no layout called that"
    and "the file could not be written" are answers, not bugs.
    """


def path() -> str:
    """Where presets are stored."""
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
    """The whole store, tolerating a missing, damaged or foreign file.

    A file that will not parse is treated as empty and said so in the log,
    rather than raised out of. The alternative is an editor that refuses to open
    because of a stray byte in a settings file -- and a user who can see their
    saved layouts are missing can say so, where a user looking at a workspace
    that will not appear cannot.
    """
    file = path()
    if not os.path.exists(file):
        return {"presets": {}}

    try:
        with open(file, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception:
        logger.warning("Model Chain: could not read Spatial Layout presets from %s; "
                       "treating it as empty", file, exc_info=True)
        return {"presets": {}}

    if not isinstance(document, dict):
        return {"presets": {}}

    presets = document.get("presets")
    if not isinstance(presets, dict):
        presets = {}
    return {"presets": {name: values for name, values in presets.items()
                        if isinstance(values, dict)}}


def _write(document: dict) -> None:
    file = path()
    payload = {"version": SCHEMA_VERSION, "presets": document.get("presets") or {}}

    directory = os.path.dirname(file) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        # Written beside the target so the replace stays on one filesystem.
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".krea_spatial_presets",
            suffix=".tmp", delete=False)
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
        raise PresetError(f"Could not save Spatial Layout presets to {file}: {exc}") from exc


# --------------------------------------------------------------------------- #
# What a preset is
# --------------------------------------------------------------------------- #


def regions_of(layout) -> list[dict]:
    """The reusable half of ``layout``: its regions, validated, in stacking order.

    ``layout`` is a serialized editor document -- the same string the hidden
    state box carries -- and it is read through the compositor's own parser
    rather than by picking keys out of the JSON here. That parser is where a
    bbox is clamped and oriented, a framing this build does not know is dropped
    with a note, and a region with no area stops being a region; a second reader
    in this file would be a second, quietly different opinion about what a
    layout is.
    """
    from prompt_master.krea import spatial

    parsed = spatial.parse(layout)
    if parsed.unreadable:
        raise PresetError("That layout could not be read, so there was nothing to save.")
    return [region.state() for region in parsed.ordered]


def normalise(values) -> dict:
    """One stored preset, cleaned against the current compositor.

    Applied on the way in and on the way out, for the reason
    ``mc_creative_profiles.normalise`` is: this is a file a user may edit and a
    file a future version may have written. A preset whose regions no longer
    parse is not an exception -- it is a preset with fewer boxes in it, and
    saying so with the boxes that survived beats refusing to open the editor.
    """
    from prompt_master.krea import spatial

    values = dict(values or {})
    document = json.dumps({"version": spatial.VERSION,
                           "regions": values.get("regions") or []})
    try:
        regions = regions_of(document)
    except Exception:
        logger.debug("Model Chain: a stored Spatial preset could not be read",
                     exc_info=True)
        regions = []
    return {"regions": regions, "saved": float(values.get("saved") or 0.0)}


def _clean_name(name: str) -> str:
    """A preset name, or a :class:`PresetError` saying why it is not one.

    Whitespace-collapsed rather than merely stripped: "Portrait  triptych" and
    "Portrait triptych" being two entries in a dropdown is a bug report waiting
    to be written, and the user cannot see the difference to fix it.
    """
    cleaned = " ".join(str(name or "").split())
    if not cleaned:
        raise PresetError("Give the layout a name.")
    if len(cleaned) > MAX_NAME:
        raise PresetError(f"That name is longer than {MAX_NAME} characters.")
    return cleaned


# --------------------------------------------------------------------------- #
# The store, as the editor sees it
# --------------------------------------------------------------------------- #


def names() -> list[str]:
    """Every saved layout, in the order somebody reading a list would expect."""
    return sorted(_read()["presets"], key=lambda name: name.casefold())


def exists(name: str) -> bool:
    return str(name or "") in _read()["presets"]


def get(name: str) -> dict | None:
    """One preset by name, cleaned, or ``None`` when there is no such layout."""
    stored = _read()["presets"].get(str(name or ""))
    if stored is None:
        return None
    return normalise(stored)


def save(name: str, layout: str, *, overwrite: bool = False) -> list[str]:
    """Store ``layout``'s regions under ``name``; return the names afterwards.

    ``overwrite`` false refuses an existing name rather than replacing it. Saving
    over a composition somebody spent ten minutes drawing is exactly the sort of
    loss the design intent forbids doing silently, so the editor asks first and
    passes the answer back in.
    """
    cleaned = _clean_name(name)
    regions = regions_of(layout)
    if not regions:
        raise PresetError("There are no regions to save. Draw a box first.")

    document = _read()
    presets = dict(document["presets"])
    if cleaned in presets and not overwrite:
        raise PresetError(f"There is already a layout called “{cleaned}”.")
    if cleaned not in presets and len(presets) >= MAX_PRESETS:
        raise PresetError(f"There are already {MAX_PRESETS} saved layouts. "
                          "Delete one before saving another.")

    presets[cleaned] = {"regions": regions, "saved": time.time()}
    _write({"presets": presets})
    return names()


def rename(name: str, to: str) -> list[str]:
    """Rename a preset, keeping what is in it. Returns the names afterwards."""
    cleaned = _clean_name(to)
    document = _read()
    presets = dict(document["presets"])
    if str(name or "") not in presets:
        raise PresetError(f"There is no saved layout called “{name}”.")
    if cleaned != name and cleaned in presets:
        raise PresetError(f"There is already a layout called “{cleaned}”.")

    presets[cleaned] = presets.pop(str(name))
    _write({"presets": presets})
    return names()


def delete(name: str) -> list[str]:
    """Remove a preset. Returns the names afterwards."""
    document = _read()
    presets = dict(document["presets"])
    if str(name or "") not in presets:
        raise PresetError(f"There is no saved layout called “{name}”.")
    presets.pop(str(name))
    _write({"presets": presets})
    return names()


def summarise(name: str) -> str:
    """One line describing a saved layout, for the editor to show beside it."""
    found = get(name)
    if found is None:
        return ""
    regions = found["regions"]
    counted = f"{len(regions)} region{'' if len(regions) == 1 else 's'}"
    written = sum(1 for region in regions if str(region.get("prompt") or "").strip())
    if not written:
        return f"{counted}, none with a prompt"
    if written == len(regions):
        return f"{counted}, all with prompts"
    return f"{counted}, {written} with prompts"


# --------------------------------------------------------------------------- #
# The browser's half: one request in, one payload out
# --------------------------------------------------------------------------- #
#
# The editor is a block of static HTML driven by one JavaScript file, and this
# store is a file on disk. Between them there has to be exactly one round trip,
# and it is shaped the way the rest of this feature is shaped: the browser
# *asks*, the server answers, and no generation waits for either.
#
# The request travels in a hidden textbox and a hidden button, which is how the
# layout itself already travels. The answer travels as the contents of a hidden
# ``gr.HTML`` -- not a textbox -- because a Textbox whose value the server
# changes fires no event in the page, while an HTML component's contents being
# replaced is a DOM mutation the browser can watch. The panel's own status line
# is the same mechanism, so this adds no new assumption about Gradio.

ACTIONS = ("list", "load", "save", "rename", "delete")


def handle(request) -> dict:
    """One editor request; the reply to put in the page.

    ``request`` is the JSON the browser wrote into its hidden box. Everything in
    it is treated as text from a page: an unknown action is a listing, a missing
    name is an error with a sentence in it, and nothing in here can raise past
    the caller -- a store that cannot be written is a line in the workspace, not
    a Gradio exception page over somebody's half-drawn composition.

    The reply always carries the current names, whatever the action was and
    whether or not it succeeded. That is what keeps the dropdown correct after a
    save that collided, a delete that raced another tab, or a store that was
    edited on disk while the editor was open.
    """
    try:
        asked = json.loads(str(request or "") or "{}")
    except Exception:
        asked = {}
    if not isinstance(asked, dict):
        asked = {}

    action = str(asked.get("action") or "list").strip().casefold()
    name = str(asked.get("name") or "")
    # Echoed straight back. The browser uses it to tell a fresh answer from the
    # one already in the page: Gradio rebuilding the tab re-runs the editor's
    # wiring, which re-reads this payload, and a recall that applied itself a
    # second time would silently undo whatever was drawn in between.
    reply: dict = {"action": action, "ok": True, "message": "", "names": [],
                   "n": asked.get("n")}

    try:
        if action == "save":
            reply["names"] = save(name, str(asked.get("layout") or ""),
                                  overwrite=bool(asked.get("overwrite")))
            reply["name"] = _clean_name(name)
            reply["message"] = f"Saved “{reply['name']}”."
        elif action == "load":
            found = get(name)
            if found is None:
                raise PresetError(f"There is no saved layout called “{name}”.")
            reply["names"] = names()
            reply["name"] = name
            reply["regions"] = found["regions"]
            reply["message"] = f"Loaded “{name}” — {summarise(name)}. " \
                               "Press Save & Return to use it."
        elif action == "rename":
            reply["names"] = rename(name, str(asked.get("to") or ""))
            reply["name"] = _clean_name(str(asked.get("to") or ""))
            reply["message"] = f"Renamed to “{reply['name']}”."
        elif action == "delete":
            reply["names"] = delete(name)
            reply["message"] = f"Deleted “{name}”."
        else:
            reply["action"] = "list"
            reply["names"] = names()
    except PresetError as exc:
        reply["ok"] = False
        reply["message"] = str(exc)
        reply["names"] = _safe_names()
        # The one error the editor can offer a second press for. Said as a flag
        # rather than parsed back out of the sentence, because a message is
        # wording and a flag is a fact.
        reply["exists"] = bool(action == "save" and exists(_soft_name(name)))
    except Exception as exc:
        logger.warning("Model Chain: a Spatial Layout preset request failed",
                       exc_info=True)
        reply["ok"] = False
        reply["message"] = f"That could not be done: {exc}"
        reply["names"] = _safe_names()

    reply["summaries"] = _summaries(reply["names"])
    return reply


def _summaries(found) -> dict:
    """A line for each name, or none at all. Never raises.

    Decoration on a dropdown: a store that has stopped being summarisable is a
    store whose *names* still have to reach the page, and an editor that would
    not build because a tooltip could not be computed is a poor trade.
    """
    try:
        return {name: summarise(name) for name in found or ()}
    except Exception:
        logger.debug("Model Chain: the Spatial presets could not be summarised",
                     exc_info=True)
        return {}


def _safe_names() -> list[str]:
    """The names, or none, when even listing them has stopped working."""
    try:
        return names()
    except Exception:
        logger.debug("Model Chain: the Spatial preset store could not be listed",
                     exc_info=True)
        return []


def _soft_name(name: str) -> str:
    """``name`` cleaned, or "" when it is not a name at all. Never raises."""
    try:
        return _clean_name(name)
    except PresetError:
        return ""


def payload(reply=None) -> str:
    """One reply as the hidden element the editor watches.

    Escaped and read back through ``textContent`` rather than written into an
    attribute, because a region prompt may legitimately contain
    ``[[<lora:name:1>]]`` and an unescaped ``<`` in either place is a broken
    page rather than a broken preset.
    """
    import html

    if reply is None:
        reply = {"action": "list", "ok": True, "message": "", "n": None,
                 "names": _safe_names()}
        reply["summaries"] = _summaries(reply["names"])
    try:
        body = json.dumps(reply, ensure_ascii=False)
    except Exception:
        logger.debug("Model Chain: a Spatial preset reply could not be encoded",
                     exc_info=True)
        body = '{"ok":false,"message":"That could not be read.","names":[]}'
    return (f'<div class="mc-krea-spatial-presets-payload" '
            f'data-version="{SCHEMA_VERSION}" hidden>{html.escape(body)}</div>')
