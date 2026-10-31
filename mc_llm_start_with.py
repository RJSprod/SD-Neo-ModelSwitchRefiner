"""Start reply with: the words the character's very next reply begins with.

Oobabooga's *Start reply with*, brought across for the next reply alone. You
write the start -- ``The color is blue`` -- and the next reply in the thread
begins with exactly those words, on screen, spoken and saved as part of it,
and the model has to carry on from them, whatever you ask next. It is put on
the wire as the opening of the character's turn, the way a thread's emotion is
(:mod:`mc_llm_emotions`), and llama-server continues it.

Unlike an emotion it is used once. The reply that starts from it -- Send,
Regenerate or Send again from here; never Continue, which carries on a reply
already written -- spends it the moment it is accepted, and the thread is back
to nothing until a start is written again (:func:`spend`). A start written
while that reply is on its way is a new one and is kept.

Kept beside the threads in the LLM data folder, by character and thread, for
the reason the emotion is: a write to the conversation's own file is a new
revision, which would make the other window's next Send stale and a reply
streaming meanwhile refuse to save. (The vendored ``response_prefix`` in the
conversation file is something else: a start for *every* reply, kept until
written over, and still unused here.)

Nothing here is logged with its text: the start is the user's.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger("model_chain")

FILE_NAME = "start_with.json"
"""Under the LLM data root, beside ``emotions.json``."""

VERSION = 1

MAX_LENGTH = 2000
"""Characters in one start. A sentence or a paragraph; a page is a message."""

_lock = threading.Lock()


class Refused(ValueError):
    """A start that cannot be kept, in a sentence for the screen."""


def path() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / FILE_NAME


def clean(text) -> str:
    """``text`` with its ends trimmed. Line breaks inside it are the user's and stay."""
    return str(text or "").strip()


def _key(character, thread_id) -> str:
    return f"{str(character or '').strip()}/{str(thread_id or '').strip()}"


def _read() -> dict:
    where = path()
    if not where.is_file():
        return {}
    try:
        from prompt_master.core.config import read_json

        data = read_json(where)
    except Exception:
        logger.warning("Model Chain: the reply starts could not be read; none is used until "
                       "one is written again")
        return {}
    threads = data.get("threads") if isinstance(data, dict) else None
    if not isinstance(threads, dict):
        return {}
    return {str(key): clean(value)[:MAX_LENGTH] for key, value in threads.items()
            if isinstance(key, str) and isinstance(value, str) and clean(value)}


def _write(threads: dict) -> None:
    from prompt_master.core.config import atomic_write_json

    atomic_write_json(path(), {"version": VERSION, "threads": dict(threads)})


def for_thread(character, thread_id) -> str:
    """The start this thread's next reply begins with, or ``""`` for none."""
    with _lock:
        return _read().get(_key(character, thread_id), "")


def write(character, thread_id, text) -> str:
    """Set this thread's next start, or clear it with ``""``. Returns what is kept."""
    start = clean(text)
    if len(start) > MAX_LENGTH:
        raise Refused(f"A start is a sentence or two: {MAX_LENGTH} characters at most.")
    key = _key(character, thread_id)
    with _lock:
        threads = _read()
        if threads.get(key, "") == start:
            return start
        if start:
            threads[key] = start
        else:
            threads.pop(key, None)
        _write(threads)
    return start


def spend(character, thread_id, start) -> bool:
    """Clear this thread's start if it is still ``start``: the reply it began is under
    way. A start written since is a new one and stays. Never fails a caller."""
    try:
        key = _key(character, thread_id)
        with _lock:
            threads = _read()
            if not start or threads.get(key) != clean(start):
                return False
            del threads[key]
            _write(threads)
        return True
    except Exception:
        logger.debug("Model Chain: could not spend a thread's reply start", exc_info=True)
        return False


def forget(character, thread_id) -> None:
    """Drop a deleted thread's start. Never fails a caller."""
    try:
        key = _key(character, thread_id)
        with _lock:
            threads = _read()
            if key in threads:
                del threads[key]
                _write(threads)
    except Exception:
        logger.debug("Model Chain: could not forget a deleted thread's reply start",
                     exc_info=True)


__all__ = ["FILE_NAME", "MAX_LENGTH", "Refused", "clean", "for_thread", "forget", "path",
           "spend", "write"]
