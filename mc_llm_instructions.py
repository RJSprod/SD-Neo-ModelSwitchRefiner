"""The writers' instructions, kept here: the system prompts a MiniMax H3 or LTX prompt is written under.

Asked on 2026-11-24, of the system prompt editor Mini Paint NEO had drawn the
day before: "those are related LLM studio IMO ... should it be removed from
mini paint and put in modelswitchrefiner?" They are. The instructions are the
writers' own -- :mod:`prompt_master.minimax.enhancer` and :mod:`mc_llm_ltx`
ship the defaults -- and until now an edit to one lived in Mini Paint, which
added it to its own presses alone: LLM Studio's MiniMax H3 workspace went on
writing under the defaults, so the same writer wrote under two sets of
instructions depending on which door the request came through.

So the edits are kept here, beside the writers, and every request that does
not bring instructions of its own runs under them (:func:`saved`, read where a
run starts, in :mod:`mc_llm_sessions`): LLM Studio's MiniMax workspace, Mini
Paint's Clipboard tab and its Send to WanGP popup, any other caller of the
external API. A caller's own ``system_prompt`` still wins for its one request.

Six sets, as there were in Mini Paint: three writers' -- MiniMax H3's FL2VA and
Ref2VA, and the LTX writer's one set for LTX 2.3 and 2.5 (LTX 2.5 is written
for under LTX 2.3's instructions, the user's choice of 2026-10-04) -- each with
a picture and without one. Saving replaces one of the six; Restore default
forgets it.

Mini Paint hands over what it had kept once (:func:`adopt`), and an edit made
here first is never overwritten by the hand-over.

Kept in the LLM data folder and read on every request -- it is one small file
-- and never logged with its text: instructions are the user's writing.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("model_chain")

FILE_NAME = "writer-instructions.json"
"""Under the LLM data root, beside ``emotions.json``."""

VERSION = 1

FL2VA = "fl2va"
REF2VA = "ref2va"
LTX23 = "ltx23"
LTX25 = "ltx25"

SETS = (FL2VA, REF2VA, LTX23)
"""The instruction sets, in the order the editor offers them."""

SET_OF = {LTX25: LTX23}
"""A variant that runs under another's set: LTX 2.5 under LTX 2.3's."""

SET_LABELS = {FL2VA: "MiniMax H3 FL2VA", REF2VA: "MiniMax H3 Ref2VA", LTX23: "LTX 2.3 and 2.5"}

TEXT = "text"
IMAGE = "image"
MODES = (TEXT, IMAGE)
MODE_LABELS = {TEXT: "without a picture", IMAGE: "with a picture"}

OVERRIDE = "override"
DEFAULT = "default"

MAX_LENGTH = 20000
"""Characters in one set: the longest default is about six thousand."""

_lock = threading.Lock()
_warned = False


class Refused(ValueError):
    """Something the editor or a caller asked for that cannot be kept, in a
    sentence for the screen, with a code a caller can switch on."""

    def __init__(self, reason: str, code: str = "rejected"):
        super().__init__(reason)
        self.reason, self.code = reason, code


def path() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / FILE_NAME


def set_of(variant) -> str:
    """The set ``variant`` is written under. Raises :class:`Refused` for a name
    that is none of the writers'."""
    name = str(variant or "").strip().casefold()
    name = SET_OF.get(name, name)
    if name not in SETS:
        raise Refused(f"There are no instructions called {str(variant)[:40]!r} here. "
                      f"Use one of {', '.join(SETS)}.", "unknown_variant")
    return name


def mode_of(mode=None, *, has_image=None) -> str:
    """``"image"`` or ``"text"``, from a mode's name or from whether a picture is sent."""
    if has_image is not None:
        return IMAGE if has_image else TEXT
    name = str(mode or "").strip().casefold()
    if name not in MODES:
        raise Refused(f"Instructions are kept {' or '.join(MODE_LABELS.values())}, "
                      f"not {str(mode)[:40]!r}.", "unknown_mode")
    return name


def clean(text) -> str:
    """``text`` with its line endings made one kind. Nothing else is touched: the
    instructions are written as the user wrote them."""
    return str(text if text is not None else "").replace("\r\n", "\n").replace("\r", "\n")


def _read() -> dict:
    """``{set: {mode: text}}`` as kept, with anything unusable left out."""
    global _warned
    where = path()
    if not where.is_file():
        return {}
    try:
        from prompt_master.core.config import read_json

        data = read_json(where)
    except Exception:
        if not _warned:
            _warned = True
            logger.warning("Model Chain: the saved writer instructions could not be read; "
                           "the defaults are used until they are saved again")
        return {}
    found = data.get("sets") if isinstance(data, dict) else None
    kept: dict = {}
    for name in SETS:
        block = found.get(name) if isinstance(found, dict) else None
        if not isinstance(block, dict):
            continue
        for mode in MODES:
            text = block.get(mode)
            if isinstance(text, str) and text.strip():
                kept.setdefault(name, {})[mode] = clean(text)[:MAX_LENGTH]
    return kept


def _write(sets: dict, handed: dict | None = None) -> None:
    from prompt_master.core.config import atomic_write_json

    record = {"version": VERSION, "sets": sets}
    if handed:
        record["handed_over"] = handed
    atomic_write_json(path(), record)


def _handed() -> dict:
    where = path()
    if not where.is_file():
        return {}
    try:
        from prompt_master.core.config import read_json

        data = read_json(where)
    except Exception:
        return {}
    handed = data.get("handed_over") if isinstance(data, dict) else None
    return dict(handed) if isinstance(handed, dict) else {}


def saved(variant, *, has_image: bool) -> str | None:
    """The instructions saved for this request's set, or ``None`` for the default.

    Never raises: a name that is no writer's, or a file that cannot be read,
    is the default -- a run is never failed by the file that changes its
    instructions."""
    try:
        name = set_of(variant)
    except Refused:
        return None
    try:
        with _lock:
            return _read().get(name, {}).get(mode_of(has_image=has_image))
    except Exception:
        logger.debug("Model Chain: could not read the saved writer instructions", exc_info=True)
        return None


def default(variant, mode) -> str:
    """The instructions the writer ships for this set."""
    name = set_of(variant)
    picture = mode_of(mode) == IMAGE
    if name == LTX23:
        import mc_llm_ltx

        return mc_llm_ltx.instructions(picture)
    from prompt_master.minimax import enhancer

    return enhancer.instructions(name, picture)


def label(variant, mode) -> str:
    """``"MiniMax H3 FL2VA, with a picture"``."""
    name = set_of(variant)
    return f"{SET_LABELS[name]}, {MODE_LABELS[mode_of(mode)]}"


def view(variant, mode) -> dict:
    """One set as the editor shows it: the instructions a request would run under
    now, where they come from, and the line that says so."""
    name = set_of(variant)
    chosen = mode_of(mode)
    with _lock:
        text = _read().get(name, {}).get(chosen)
    shipped = default(name, chosen)
    called = label(name, chosen)
    if text is not None:
        line = (f"Your instructions for {called}. Every request that brings none of its own "
                "uses them: LLM Studio's MiniMax workspace, Mini Paint's Clipboard and Send to "
                "WanGP. Restore default forgets them.")
        return {"variant": name, "mode": chosen, "text": text, "source": OVERRIDE,
                "default": shipped, "label": called, "line": line}
    line = (f"Default for {called}, as ModelSwitchRefiner ships it. Edit it and save an "
            "override to replace it for every request.")
    return {"variant": name, "mode": chosen, "text": shipped, "source": DEFAULT,
            "default": shipped, "label": called, "line": line}


def save(variant, mode, text) -> dict:
    """Keep ``text`` as this set's instructions from now on. Answers with :func:`view`.

    Blank is refused rather than kept (Restore default is how to go back), and
    so is text past :data:`MAX_LENGTH` -- refused, never cut. The default saved
    unchanged is no override: it would stop following the writer's own when an
    update changes it, and nothing on screen would say so."""
    name = set_of(variant)
    chosen = mode_of(mode)
    written = clean(text)
    if not written.strip():
        raise Refused("Instructions cannot be blank. Restore default puts the shipped ones back.",
                      "empty_system_prompt")
    if len(written) > MAX_LENGTH:
        raise Refused(f"Instructions are {MAX_LENGTH} characters at most; these are {len(written)}.",
                      "system_prompt_too_long")
    shipped = default(name, chosen)
    with _lock:
        sets = _read()
        block = sets.get(name, {})
        if written.strip() == shipped.strip():
            if chosen in block:
                del block[chosen]
                if not block:
                    sets.pop(name, None)
                _write(sets, _handed())
        elif block.get(chosen) != written:
            sets.setdefault(name, {})[chosen] = written
            _write(sets, _handed())
    logger.info("Model Chain: writer instructions saved for %s %s (%d characters)",
                name, chosen, len(written))
    return view(name, chosen)


def restore(variant, mode) -> bool:
    """Forget this set's saved instructions. True when there were some."""
    name = set_of(variant)
    chosen = mode_of(mode)
    with _lock:
        sets = _read()
        block = sets.get(name, {})
        if chosen not in block:
            return False
        del block[chosen]
        if not block:
            sets.pop(name, None)
        _write(sets, _handed())
    logger.info("Model Chain: writer instructions for %s %s back to the default", name, chosen)
    return True


def saved_sets() -> dict:
    """Which of the six sets have instructions of the user's: ``{set: {mode: bool}}``."""
    with _lock:
        sets = _read()
    return {name: {mode: mode in sets.get(name, {}) for mode in MODES} for name in SETS}


def adopt(offered, *, origin: str = "") -> dict:
    """Take over instructions another extension kept: Mini Paint's, once.

    ``offered`` is ``{set: {mode: text}}`` -- LTX 2.5 under either name. A set
    this store already has is kept as it is: an edit made here is newer than
    anything handed over. Answers ``{"adopted": n, "kept": n, "refused": n}``.
    """
    taken = kept = refused = 0
    block_in = offered if isinstance(offered, dict) else {}
    with _lock:
        sets = _read()
        for variant, modes in block_in.items():
            try:
                name = set_of(variant)
            except Refused:
                refused += 1
                continue
            for mode, text in (modes.items() if isinstance(modes, dict) else ()):
                try:
                    chosen = mode_of(mode)
                except Refused:
                    refused += 1
                    continue
                written = clean(text) if isinstance(text, str) else ""
                if not written.strip() or len(written) > MAX_LENGTH:
                    refused += 1
                    continue
                if chosen in sets.get(name, {}):
                    kept += 1
                    continue
                if written.strip() == default(name, chosen).strip():
                    kept += 1
                    continue
                sets.setdefault(name, {})[chosen] = written
                taken += 1
        handed = _handed()
        who = str(origin or "another extension")[:80]
        handed[who] = {"adopted": taken, "kept": kept, "refused": refused,
                       "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _write(sets, handed)
    logger.info("Model Chain: took over %d set(s) of writer instructions from %s "
                "(%d already here, %d unusable)", taken, who, kept, refused)
    return {"adopted": taken, "kept": kept, "refused": refused}


def sets() -> list:
    """The sets the editor offers: ``[{"id", "label"}]``."""
    return [{"id": name, "label": SET_LABELS[name]} for name in SETS]


def modes() -> list:
    """The two modes the editor offers: ``[{"id", "label"}]``."""
    return [{"id": mode, "label": MODE_LABELS[mode]} for mode in MODES]


__all__ = ["DEFAULT", "FILE_NAME", "IMAGE", "MAX_LENGTH", "MODES", "MODE_LABELS", "OVERRIDE",
           "Refused", "SETS", "SET_LABELS", "SET_OF", "TEXT", "adopt", "clean", "default",
           "label", "mode_of", "modes", "path", "restore", "save", "saved", "saved_sets",
           "set_of", "sets", "view"]
