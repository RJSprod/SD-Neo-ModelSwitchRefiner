"""Persistence for LLM Studio: shared preferences, and three separate histories.

Section 16 draws one line and this module exists to hold it:

* **shared runtime preferences** -- which GGUF, which projector, which device,
  context and buffer settings -- belong to the installation. All three modes
  read them and any of them may change them.
* **mode content** -- what was generated, asked, or enhanced -- belongs to its
  mode. Prompt Studio sessions, Conversation threads and MiniMax sessions are
  three stores and stay three stores, because one LLM serving all three is a
  fact about the runtime and not a reason to merge a user's writing into a
  single stream.

Conversation is the exception in implementation only: it already has stores
worth keeping in the vendored ``prompt_master.chat`` package -- characters as
copyable files, chats as documents, both in the layouts other tools expect --
so it keeps them, and this module does not shadow them.

Everything here writes through a temporary file and an atomic replace, which is
the convention ``mc_presets`` established for the same reason: an interrupted
write should leave the previous file intact rather than a truncated one
(section 16).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

SCHEMA_VERSION = 1

PREFERENCES_FILE = "preferences.json"
PROMPT_HISTORY_FILE = "prompt-studio-history.json"
MINIMAX_HISTORY_FILE = "minimax-history.json"

HISTORY_LIMIT = 200
"""Entries kept per mode history.

A cap rather than unbounded growth: these files are read whole on every panel
refresh, and a session that ran for a year should not make opening a tab slow.
Oldest go first.
"""

_lock = threading.RLock()


def _directory() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / "data"


def _path(filename: str) -> Path:
    return _directory() / filename


# --------------------------------------------------------------------------- #
# Shared preferences (section 16)
# --------------------------------------------------------------------------- #


DEFAULTS: dict = {
    "mode": "prompt",
    # Runtime placement. "auto" sizes the context buffer from what is free;
    # "fixed" honours context_buffer_gb exactly (section 11).
    "context_mode": "auto",
    "context_buffer_gb": 4.0,
    "context_size": 8192,
    "kv_type_k": "f16",
    "kv_type_v": "f16",
    # Prompt Studio's last controls, so the panel opens where it was left.
    "prompt_defaults": {},
    # Conversation's last character and thread.
    "character": "",
    "thread": "",
    # MiniMax's last variant.
    "minimax_variant": "fl2va",
}


def preferences() -> dict:
    """Shared runtime preferences, with every default filled in."""
    stored = _read(PREFERENCES_FILE, {})
    merged = dict(DEFAULTS)
    for key, value in stored.items():
        if key in merged or key == "version":
            merged[key] = value
    return merged


def remember(**values) -> dict:
    """Update shared preferences. Returns the stored result."""
    with _lock:
        current = preferences()
        current.update({k: v for k, v in values.items() if v is not None})
        current["version"] = SCHEMA_VERSION
        _write(PREFERENCES_FILE, current)
        return current


# --------------------------------------------------------------------------- #
# Prompt Studio history (section 4.2: "may keep a history of prompt sessions")
# --------------------------------------------------------------------------- #


@dataclass
class PromptSession:
    """One Prompt Studio generation, kept whole.

    The controls are stored beside the output rather than only the output,
    because the useful thing to do with a prompt you generated last week is to
    load the settings back and change one of them.
    """

    identifier: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    title: str = ""
    intent: str = ""
    positive: str = ""
    negative: str = ""
    seed: int = 0
    image_name: str = ""
    controls: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created))
        return f"{stamp} — {self.title or self.intent[:48] or 'untitled'}"


@dataclass
class MinimaxSession:
    """One MiniMax H3 enhancement, kept separately from everything else."""

    identifier: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    variant: str = "fl2va"
    prompt: str = ""
    caption: str = ""
    result: str = ""
    seed: int = 0
    image_name: str = ""

    @property
    def label(self) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created))
        return f"{stamp} — {self.prompt[:48] or 'untitled'}"


def prompt_sessions() -> list[PromptSession]:
    return [PromptSession(**_fields(PromptSession, row))
            for row in _read(PROMPT_HISTORY_FILE, {}).get("sessions", [])]


def save_prompt_session(session: PromptSession) -> PromptSession:
    _append(PROMPT_HISTORY_FILE, session)
    return session


def delete_prompt_session(identifier: str) -> None:
    _remove(PROMPT_HISTORY_FILE, identifier)


def minimax_sessions() -> list[MinimaxSession]:
    return [MinimaxSession(**_fields(MinimaxSession, row))
            for row in _read(MINIMAX_HISTORY_FILE, {}).get("sessions", [])]


def save_minimax_session(session: MinimaxSession) -> MinimaxSession:
    _append(MINIMAX_HISTORY_FILE, session)
    return session


def delete_minimax_session(identifier: str) -> None:
    _remove(MINIMAX_HISTORY_FILE, identifier)


def clear_history(filename: str) -> None:
    with _lock:
        _write(filename, {"version": SCHEMA_VERSION, "sessions": []})


# --------------------------------------------------------------------------- #
# File handling
# --------------------------------------------------------------------------- #


def _fields(kind, row: dict) -> dict:
    """Only the keys ``kind`` declares.

    A history file written by a newer version of the extension carries keys
    this one has never heard of, and the useful behaviour there is to read what
    is recognised and ignore the rest -- not to raise on somebody's saved work.
    """
    known = set(getattr(kind, "__dataclass_fields__", {}))
    return {k: v for k, v in row.items() if k in known}


def _append(filename: str, session) -> None:
    with _lock:
        document = _read(filename, {})
        sessions = list(document.get("sessions", []))
        identifier = getattr(session, "identifier", "")
        sessions = [row for row in sessions if row.get("identifier") != identifier]
        sessions.append(asdict(session))
        document["sessions"] = sessions[-HISTORY_LIMIT:]
        document["version"] = SCHEMA_VERSION
        _write(filename, document)


def _remove(filename: str, identifier: str) -> None:
    with _lock:
        document = _read(filename, {})
        document["sessions"] = [row for row in document.get("sessions", [])
                                if row.get("identifier") != identifier]
        document["version"] = SCHEMA_VERSION
        _write(filename, document)


def _read(filename: str, default):
    try:
        loaded = json.loads(_path(filename).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
    return loaded if isinstance(loaded, dict) else default


def _write(filename: str, document: dict) -> None:
    path = _path(filename)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
        os.replace(temporary, path)
    except OSError:
        logger.warning("Model Chain: could not write %s", filename, exc_info=True)
