"""Character Emotion: a short line that starts the character's next reply.

Oobabooga's *character bias*, brought across. The line -- ``*I am so happy*``
-- is put on the wire as the opening of the character's turn, after the
conversation and after your message, and the model writes its reply as if it
had already begun with it. llama-server continues an assistant message that
ends the request (b10621 does by default) and sends back only what it wrote,
so the line is never in a reply, never on screen, never spoken and never
saved: the history holds the reply alone, and the next turn sees the reply
alone. That is also why the character never starts copying the line.

Two things are kept, in one file in the LLM data folder:

* **The list**: one for every character and thread, Oobabooga's five until it
  is changed (:data:`DEFAULTS`). Lines are added and deleted from the
  Emotion picker in either view.
* **Each thread's choice**: by character and thread, empty for Off. A new
  thread has none, so it starts Off; a branch is given its source's
  (:func:`carry`); a deleted thread's is forgotten (:func:`forget`).

The choice is kept here and not in the conversation's own file, because a
write to that file is a new revision of the conversation: the other window's
next Send would be refused as "changed in another window", and a choice made
while a reply streams would make that reply's save refuse as a conflict. A
setting is not a change to what was said.

A thread whose line has been deleted from the list reads as Off
(:func:`for_thread`), which is "deleting an emotion that threads are using
turns those threads back to Off" without rewriting anything.

Nothing here is logged with its text: the lines are the user's.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger("model_chain")

FILE_NAME = "emotions.json"
"""Under the LLM data root, beside ``logs`` and ``data``."""

VERSION = 1

DEFAULTS = ("*I am so happy*", "*I am so sad*", "*I am so excited*", "*I am so bored*",
            "*I am so angry*")
"""Oobabooga's own ``bias_options.txt``, in its order."""

MAX_LENGTH = 200
"""Characters in one line. A bias is a few words; a paragraph is a system
prompt, and belongs in the character."""

MAX_COUNT = 50
"""Lines in the list. Enough for any taste, few enough to scroll through."""

_lock = threading.Lock()


class Refused(ValueError):
    """A change to the list that cannot be made, in a sentence for the screen."""


def path() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / FILE_NAME


def clean(text) -> str:
    """``text`` as one line: whitespace and line breaks folded to single spaces."""
    return " ".join(str(text or "").split())


def _read() -> dict:
    """The file's contents, or what an installation that never saved one has."""
    found = {"emotions": list(DEFAULTS), "threads": {}}
    where = path()
    if not where.is_file():
        return found
    try:
        from prompt_master.core.config import read_json

        data = read_json(where)
    except Exception:
        logger.warning("Model Chain: the emotion list could not be read; the default list "
                       "is used until it is saved again")
        return found
    emotions = data.get("emotions")
    if isinstance(emotions, list):
        seen: list[str] = []
        for item in emotions:
            line = clean(item)[:MAX_LENGTH]
            if line and line not in seen:
                seen.append(line)
        found["emotions"] = seen[:MAX_COUNT]
    threads = data.get("threads")
    if isinstance(threads, dict):
        found["threads"] = {str(key): clean(value) for key, value in threads.items()
                            if isinstance(key, str) and clean(value)}
    return found


def _write(data: dict) -> None:
    from prompt_master.core.config import atomic_write_json

    atomic_write_json(path(), {"version": VERSION, "emotions": list(data["emotions"]),
                               "threads": dict(data["threads"])})


def _thread_key(character, thread_id) -> str:
    return f"{str(character or '').strip()}/{str(thread_id or '').strip()}"


# -- the list ----------------------------------------------------------------- #


def listed() -> list[str]:
    """Every line, in the order they were added."""
    with _lock:
        return list(_read()["emotions"])


def add(text) -> list[str]:
    """Add ``text`` to the end of the list, unless it is there. Returns the list."""
    line = clean(text)
    if not line:
        raise Refused("Write the emotion first.")
    if len(line) > MAX_LENGTH:
        raise Refused(f"An emotion is a short line: {MAX_LENGTH} characters at most.")
    with _lock:
        data = _read()
        if line not in data["emotions"]:
            if len(data["emotions"]) >= MAX_COUNT:
                raise Refused(f"The list is full at {MAX_COUNT}. Delete one first.")
            data["emotions"].append(line)
            _write(data)
        return list(data["emotions"])


def remove(text) -> list[str]:
    """Take ``text`` off the list. Returns the list.

    Every thread that had it reads as Off from now on (:func:`for_thread`), and
    its record is dropped with it, so adding the same line again later does not
    quietly turn those threads back on.
    """
    line = clean(text)
    with _lock:
        data = _read()
        if line in data["emotions"]:
            data["emotions"].remove(line)
            data["threads"] = {key: value for key, value in data["threads"].items()
                               if value != line}
            _write(data)
        return list(data["emotions"])


# -- each thread's choice ------------------------------------------------------ #


def for_thread(character, thread_id) -> str:
    """The line this thread's next reply starts with, or ``""`` for Off."""
    with _lock:
        data = _read()
    chosen = data["threads"].get(_thread_key(character, thread_id), "")
    return chosen if chosen in data["emotions"] else ""


def choose(character, thread_id, text) -> str:
    """Set this thread's emotion: a line from the list, or ``""`` for Off."""
    line = clean(text)
    key = _thread_key(character, thread_id)
    with _lock:
        data = _read()
        if line and line not in data["emotions"]:
            raise Refused("That emotion is not in the list any more.")
        if line:
            if data["threads"].get(key) == line:
                return line
            data["threads"][key] = line
        elif key in data["threads"]:
            del data["threads"][key]
        else:
            return ""
        _write(data)
    return line


def carry(character, source_thread, made_thread) -> None:
    """Give a branch its source thread's emotion. Never fails a caller."""
    try:
        chosen = for_thread(character, source_thread)
        if chosen:
            choose(character, made_thread, chosen)
    except Exception:
        logger.debug("Model Chain: could not carry a thread's emotion into its branch",
                     exc_info=True)


def forget(character, thread_id) -> None:
    """Drop a deleted thread's emotion. Never fails a caller."""
    try:
        key = _thread_key(character, thread_id)
        with _lock:
            data = _read()
            if key in data["threads"]:
                del data["threads"][key]
                _write(data)
    except Exception:
        logger.debug("Model Chain: could not forget a deleted thread's emotion", exc_info=True)


__all__ = ["DEFAULTS", "FILE_NAME", "MAX_COUNT", "MAX_LENGTH", "Refused", "add", "carry",
           "choose", "clean", "for_thread", "forget", "listed", "path", "remove"]
