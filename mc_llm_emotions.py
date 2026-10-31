"""Character Emotion: how the character feels, steering its replies one of two ways.

Oobabooga's *character bias*, brought across, with a switch between two roads
(:func:`mode`, one choice for every thread):

* **The system prompt** (:data:`SYSTEM`, the default). The line is put at the
  end of the thread's system prompt, in one shape for every character
  (:func:`system_note`): the character's own prompt, override or built, stays
  as it is in the editor, and every request in the thread -- Continue included
  -- is sent with the two together. The reply is written from the feeling
  rather than after it, so it starts as a reply starts. The front of every
  prompt is the same while the emotion is, so llama-server keeps the
  conversation cached from one message to the next; a change of emotion
  changes the front, and sets the conversation being read again in the
  background at once (``mc_llm_conversation_ops.rewarm``,
  :mod:`mc_llm_cache_update`).
* **The start of the reply** (:data:`PREFILL`), Oobabooga's own way: its
  ``bot_prefix_modifier`` writes the line after the character's turn opening,
  so the model reads it as the first words of its own reply and the page shows
  only what follows. Here the line is a trailing assistant message for Send,
  Regenerate and Send again after a message of yours -- never Continue --
  ending on an empty line (:func:`opening`), which llama-server continues and
  sends back at the front of the reply, where :class:`Unechoed` takes it off;
  the user found such replies starting "like there is hidden response that's
  been cropped out", which is why the system prompt became the default, and
  asked to keep both. The cache disagrees with
  the next prompt from the line onward, so the reply is read again in the
  background after it is saved.

Switching between them changes the front of every thread that has an emotion,
and the thread the switch is made in is read again at once.

Replies saved while the line leaked into them were cleaned once at start-up
(:func:`clean_saved_replies`). :class:`Unechoed` and :func:`without` serve the
start-of-reply way and Start reply with (:mod:`mc_llm_start_with`).

Three things are kept, in one file in the LLM data folder:

* **The list**: one for every character and thread, Oobabooga's five until it
  is changed (:data:`DEFAULTS`). Lines are added and deleted from the
  Emotion picker in either view.
* **Each thread's choice**: by character and thread, empty for Off. A new
  thread has none, so it starts Off; a branch is given its source's
  (:func:`carry`); a deleted thread's is forgotten (:func:`forget`).
* **How they steer**: :data:`SYSTEM` or :data:`PREFILL`, for every thread.

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

SYSTEM = "system"
"""The way a thread's emotion steers: at the end of the thread's system prompt,
the same for every request in it (:func:`system_note`). The default."""

PREFILL = "prefill"
"""Oobabooga's way: the line written after the character's turn opening, as the
first words of a new reply, and taken off what comes back (:class:`Unechoed`)."""

BREAK = "\n\n"
"""What follows the line at the start of a reply (:data:`PREFILL`, :func:`opening`):
an empty line, so the reply -- or a start of yours -- begins on a line of its own
and not straight after the line's last character. Oobabooga's modifier ends on a
space; the user asked for the empty line. Hidden with the line."""

MODES = (SYSTEM, PREFILL)
"""One choice for every thread, kept with the list."""

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
    found = {"emotions": list(DEFAULTS), "threads": {}, "mode": SYSTEM}
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
    if data.get("mode") in MODES:
        found["mode"] = data["mode"]
    return found


def _write(data: dict) -> None:
    from prompt_master.core.config import atomic_write_json

    atomic_write_json(path(), {"version": VERSION, "emotions": list(data["emotions"]),
                               "threads": dict(data["threads"]),
                               "mode": data.get("mode", SYSTEM)})


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


# -- how it steers -------------------------------------------------------------- #


def mode() -> str:
    """How every thread's emotion steers: :data:`SYSTEM` or :data:`PREFILL`."""
    with _lock:
        return _read()["mode"]


def set_mode(chosen) -> str:
    """Choose how every thread's emotion steers. Returns the mode kept."""
    if chosen not in MODES:
        raise Refused("An emotion steers from the system prompt or as the start of the reply.")
    with _lock:
        data = _read()
        if data["mode"] != chosen:
            data["mode"] = chosen
            _write(data)
    return chosen


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


# -- the system prompt ------------------------------------------------------------ #


def system_note(line, name, you="you") -> str:
    """What the thread's system prompt ends with while ``line`` is its emotion.

    One shape for every character and every line, so the character's own prompt
    -- its override, or the one built from its Context -- is never edited and
    reads the same in the editor: this is put after it on the wire
    (``prompt.build``'s ``system_extra``), for every request in the thread.
    ``{{char}}`` and ``{{user}}`` in the line are the character's and yours, as
    everywhere else in a prompt.
    """
    from prompt_master.chat.prompt import substitute

    who = str(name or "").strip() or "the character"
    feeling = substitute(clean(line), who, str(you or "you"))
    if not feeling:
        return ""
    return (f"[Current emotion]\n"
            f"{who} feels this right now, in {who}'s own words: {feeling}\n"
            f"Let it colour {who}'s tone, words and actions in every reply until it "
            f"changes. Do not quote this note, and do not say that you were told how to feel.")


# -- the reply ------------------------------------------------------------------ #


def opening(line) -> str:
    """The start of a new reply under :data:`PREFILL`: the line, then an empty
    line (:data:`BREAK`); ``""`` for Off. A start of yours follows it as it is."""
    line = clean(line)
    return line + BREAK if line else ""


class Unechoed:
    """A reply streamed from a line, with the line taken off its front.

    llama-server b10621 reads a continued reply from the start of the turn --
    the generation prompt it parses with ends on the line -- and leaves the
    line out of the first piece it streams only for a request marked as a
    continuation, which nothing in its chat endpoint marks. So the reply comes
    back as the line, then what the model wrote after it. The pieces are held
    while they could still be the line; the line and the spaces and line breaks
    after it are dropped when they are there, and anything else passes as it
    came: a reply that opens with an action of its own in asterisks, the line
    written again further on. Only the very front is compared, and exactly:
    what was sent is the emotion's line, its empty line (:func:`opening`) and a
    start of yours, whose own line breaks are its own, so nothing inside it is
    folded.

    ``trim`` false keeps the space after the line: the line ends on words the
    caller shows as the reply's start (:mod:`mc_llm_start_with`), and what the
    model wrote after them is carried on from them exactly as it was written.
    """

    def __init__(self, line: str, trim: bool = True):
        self.line = str(line or "").strip()
        self.trim = trim
        self._held = ""
        self._settled = not self.line
        self._trimming = False

    def feed(self, text: str) -> str:
        """What of ``text`` is shown now; ``""`` while it is held or dropped."""
        if self._settled:
            if self._trimming:
                text = text.lstrip()
                self._trimming = not text
            return text
        self._held += text
        front = self._held.lstrip()
        if not front or (len(front) < len(self.line) and self.line.startswith(front)):
            return ""
        self._settled = True
        held, self._held = self._held, ""
        if not front.startswith(self.line):
            return held
        rest = front[len(self.line):]
        if self.trim:
            rest = rest.lstrip()
            self._trimming = not rest
        return rest

    def finish(self) -> str:
        """What was still held when the reply ended: the model's, not the line."""
        held, self._held = self._held, ""
        self._settled = True
        return held


def without(line, text, trim: bool = True) -> str:
    """``text`` with ``line`` taken off its front, as :class:`Unechoed` takes it."""
    line, text = str(line or "").strip(), str(text or "")
    front = text.lstrip()
    if line and front.startswith(line):
        rest = front[len(line):]
        return rest.lstrip() if trim else rest
    return text


# -- the one-time cleanup ------------------------------------------------------- #


CLEANUP_FILE = "emotion-cleanup.json"
"""Under the LLM data root: written when the cleanup has run, so it runs once."""

LEAK_BEGAN = 1793318400.0
"""2026-10-30 00:00 UTC, the day the first build with emotions shipped (#310).
No reply before it was started from a line, so no thread last written before it
is looked at."""


def clean_saved_replies(store) -> int:
    """Once: take a listed line off the front of the saved replies that start with one.

    Until :class:`Unechoed`, llama-server's echo of the line was saved as the
    start of every reply written with an emotion, and the history then showed
    the model its own replies opening with the line. Every reply in a thread
    written since emotions shipped is read; a version that starts with a line
    from the list, or with one of Oobabooga's five, has it taken off with the
    space after it (:func:`without`), through the guarded store, so the
    revision moves and a page is told. A reply the character really started
    with such a line cannot be told from an echo and is cleaned too, which is
    why this runs once and never again: the record is written when a pass ends,
    or at once when there is no chats folder (nothing was ever saved, so nothing
    leaked), and not when the folder is there and cannot be read, which is
    looked at again next time. Returns how many versions were cleaned. Never
    raises.
    """
    try:
        import mc_llm_paths

        record = mc_llm_paths.data_root() / CLEANUP_FILE
        if record.is_file():
            return 0
        lines = sorted({*listed(), *DEFAULTS}, key=len, reverse=True)
        folder = Path(store.directory)
        keys = list(_threads_since(store, LEAK_BEGAN)) if folder.exists() else []
    except Exception:
        logger.debug("Model Chain: could not look for emotion lines in saved replies",
                     exc_info=True)
        return 0
    cleaned = sum(_clean_thread(store, key, lines) for key in keys)
    try:
        from prompt_master.core.config import atomic_write_json

        atomic_write_json(record, {"version": 1, "cleaned": cleaned})
    except Exception:
        logger.debug("Model Chain: could not record the emotion cleanup", exc_info=True)
    if cleaned:
        logger.info("Model Chain: took an emotion's line off the front of %d saved "
                    "reply/replies, once", cleaned)
    return cleaned


def _threads_since(store, since: float):
    """The key of every saved thread last written at or after ``since``."""
    from prompt_master.core.config import read_json

    import mc_llm_conversation_store as store_module

    for folder in sorted(Path(store.directory).iterdir()):
        if not folder.is_dir():
            continue
        for found in sorted(folder.glob("*.json")):
            if found.name.startswith("."):
                continue
            try:
                data = read_json(found)
                character, identifier = str(data["character"]), str(data["id"])
                written = float(data.get("updated", 0.0) or 0.0)
            except Exception:
                continue
            if character and identifier and written >= since:
                yield store_module.Key(character=character, thread_id=identifier)


def _without_any(lines, text: str) -> str:
    for line in lines:
        kept = without(line, text)
        if kept != text:
            return kept
    return text


def _clean_thread(store, key, lines) -> int:
    """How many of this thread's reply versions lost a line; 0 when it is left alone."""
    from prompt_master.chat.history import ASSISTANT

    import mc_llm_conversation_service as service
    import mc_llm_conversation_store as store_module

    count = 0

    def change(copy):
        nonlocal count
        count = 0
        for message in copy.messages:
            if message.role != ASSISTANT:
                continue
            for index, version in enumerate(message.versions):
                kept = _without_any(lines, version)
                if kept != version:
                    message.versions[index] = kept
                    count += 1
        return None if count else False

    try:
        committed = store_module.transaction(store, key, store_module.UNCHECKED, change)
    except Exception:
        logger.debug("Model Chain: could not clean a thread's saved replies", exc_info=True)
        return 0
    if not committed.written:
        return 0
    try:
        service.publish(service.CONVERSATION_CHANGED, key, revision=committed.revision)
    except Exception:
        logger.debug("Model Chain: could not announce a cleaned thread", exc_info=True)
    return count


__all__ = ["BREAK", "CLEANUP_FILE", "DEFAULTS", "FILE_NAME", "LEAK_BEGAN", "MAX_COUNT",
           "MAX_LENGTH", "MODES", "PREFILL", "Refused", "SYSTEM", "Unechoed", "add", "carry",
           "choose", "clean", "clean_saved_replies", "for_thread", "forget", "listed", "mode",
           "opening", "path", "remove", "set_mode", "system_note", "without"]
