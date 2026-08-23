"""Persistence for LLM Studio: shared preferences, and separate histories.

Section 16 draws one line and this module exists to hold it:

* **shared runtime preferences** -- which GGUF, which projector, which device,
  context and buffer settings -- belong to the installation. Every mode reads
  them and any of them may change them.
* **mode content** -- what was generated, asked, or enhanced -- belongs to its
  mode. Prompt Studio sessions, Conversation threads, MiniMax sessions and
  Krea sessions are four stores and stay four stores, because one LLM serving
  all of them is a fact about the runtime and not a reason to merge a user's
  writing into a single stream.

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
KREA_HISTORY_FILE = "krea-history.json"

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
# Shared preferences (section 16), and the half of them the Settings page owns
# --------------------------------------------------------------------------- #

OPT_CONTEXT_MODE = "model_chain_llm_context_mode"
OPT_CONTEXT_BUFFER = "model_chain_llm_context_buffer_gb"
OPT_CONTEXT_SIZE = "model_chain_llm_context_size"
OPT_KV_TYPE_K = "model_chain_llm_kv_type_k"
OPT_KV_TYPE_V = "model_chain_llm_kv_type_v"

CONTEXT_MODES = (
    ("auto", "Automatic — fill what is free"),
    ("fixed", "Fixed buffer"),
)
"""How the key/value cache is budgeted (section 11), as ``(value, label)``.

The same shape ``mc_broker.MODES`` uses, and for the same reason: what a Gradio
radio on the Settings page stores is the string it displayed, so the stored
value is a label and everything downstream wants the value.
"""

HOSTED = (OPT_CONTEXT_MODE, OPT_CONTEXT_BUFFER, OPT_CONTEXT_SIZE,
          OPT_KV_TYPE_K, OPT_KV_TYPE_V)
"""Preference keys the WebUI's Settings page is the front end for.

These five describe the installation rather than anything a mode is doing, so
they were moved out of the tab and on to the Settings page, where the host
persists them in ``config.json`` and shows them beside the residency settings
they belong with.

The file below is still written and still read: it is what answers on a
headless install, during a test, and before the host has registered anything.
:func:`preferences` layers the host's value over the file's when there is one,
so the two cannot disagree about which is authoritative.
"""


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
    # Krea Creative Mode. One set of settings for both surfaces on purpose: the
    # axes, the Creativity position and the seed describe how this installation
    # does art direction, and somebody who has spent five minutes configuring
    # ten axes in LLM Studio should not have to do it again in txt2img.
    #
    # These are the *current* settings, not the saved ones. Named configurations
    # live in mc_creative_profiles, in their own file under the WebUI data
    # directory, because a list of complete configurations is a different shape
    # from one flat mapping of what the panel is set to right now -- and because
    # every save of a slider position would otherwise rewrite somebody's saved
    # work.
    #
    # Every default here is a fallback for a headless read; the authoritative
    # ones are the creativity package's own defaults.json, which
    # mc_creative_krea.settings() layers over these.
    "krea_creative_enabled": False,
    "krea_creativity": 5,
    "krea_creative_seed": -1,
    "krea_creative_anti_repetition": True,
    "krea_creative_axis_modes": {},
    "krea_creative_fixed": {},
    # Treatments a Vary axis must never choose, by axis. A modifier of Vary and
    # not a fourth mode: "vary the lighting, but never harsh noon" is a statement
    # about how to vary, and making it a mode would force somebody who wants two
    # treatments gone to stop varying altogether.
    "krea_creative_excluded": {},
    # The named profile the settings above were last loaded from, for the panel
    # to open showing. A label on the settings rather than a source of them.
    "krea_creative_profile": "",
    # The variant ids of the last few rolls, newest last. Ids, never prompts:
    # what anti-repetition needs to know is "did we just use the impasto
    # medium", and storing the prompts to answer that would be keeping a
    # transcript of everything anybody asked for in a preferences file.
    "krea_creative_history": [],
    # Spatial Layout. Five keys, and the third of them is the only place in this
    # file that holds a *document* rather than a setting.
    #
    # The canvas is persisted deliberately. Boxes are minutes of work with a
    # mouse, and a WebUI restart that quietly emptied them would be exactly the
    # silent loss of layout state the design intent forbids -- so the last saved
    # layout is what the editor opens onto and what an API request with no page
    # behind it composes. It is not part of a Creative profile: a profile says
    # how art direction behaves, and a composition is about one picture.
    "krea_spatial_enabled": False,
    "krea_spatial_compose_mode": "smart",
    "krea_spatial_layout": "",
    "krea_spatial_record_scenes": True,
    # The fifth is FLUX.2 Klein's, and is named for its backend rather than for
    # Krea because it means nothing to Krea 2. One canvas, one enabled switch,
    # and two questions about what happens to the boxes: Krea's is whether a
    # language model reconciles the scene around them, Klein's is what the source
    # image is and how much of it survives. Both are remembered so that switching
    # checkpoints back and forth does not lose either answer.
    "klein_spatial_mode": "auto",
    # And which backend the panel shows. Auto follows the loaded checkpoint,
    # which is right whenever the host announces one in a way this extension can
    # read; the override is for when it does not.
    "spatial_backend": "auto",
}


def label_for_context_mode(value: str) -> str:
    """:data:`CONTEXT_MODES`' display text for ``value``.

    What a Gradio radio on the Settings page stores is the string it displayed,
    so the option's *default* has to be a label too -- registering ``"auto"``
    there would make the page show a radio with nothing selected until it was
    touched.
    """
    for candidate, label in CONTEXT_MODES:
        if candidate == value:
            return label
    return value


def preferences() -> dict:
    """Shared runtime preferences, with every default filled in.

    Read in three layers, most specific last: the defaults above, the
    preferences file, and then whatever the Settings page holds for the five
    keys in :data:`HOSTED`. The last layer is what makes the Settings page
    authoritative for them without any caller having to know that it is --
    ``mc_llm_runtime.config()`` and the estimator ask this function the same
    question they always asked.
    """
    stored = _read(PREFERENCES_FILE, {})
    merged = dict(DEFAULTS)
    for key, value in stored.items():
        if key in merged or key == "version":
            merged[key] = value
    merged.update(_hosted(merged))
    return merged


def _hosted(current: dict) -> dict:
    """The Settings page's answer for :data:`HOSTED`, where it has one.

    A setting the host has never heard of, or a host that is not there at all,
    contributes nothing -- which is what leaves the file in charge on a
    headless install rather than replacing it with a default.
    """
    found: dict = {}
    context_mode = _option(OPT_CONTEXT_MODE)
    if context_mode is not None:
        found["context_mode"] = _resolve(context_mode, CONTEXT_MODES,
                                         str(current.get("context_mode", "auto")))
    for key, name, cast in (("context_buffer_gb", OPT_CONTEXT_BUFFER, float),
                            ("context_size", OPT_CONTEXT_SIZE, int)):
        raw = _option(name)
        if raw is None:
            continue
        try:
            found[key] = cast(float(raw))
        except (TypeError, ValueError):
            continue
    for key, name in (("kv_type_k", OPT_KV_TYPE_K), ("kv_type_v", OPT_KV_TYPE_V)):
        raw = _option(name)
        if raw is not None:
            found[key] = _kv_type(raw, str(current.get(key, "f16")))
    return found


def _option(name: str):
    """One Forge setting, or ``None`` when there is no host or no answer."""
    try:
        from modules import shared

        value = getattr(shared.opts, name, None)
    except Exception:
        return None
    return None if value in (None, "") else value


def _resolve(raw, table, default: str) -> str:
    """``raw`` as one of ``table``'s values, accepting either half of the pair.

    Restated here rather than imported from :mod:`mc_broker`: this module is
    the bottom of the LLM half's import graph -- the panels, the runtime and
    the estimator all read it -- and giving it a dependency on the broker would
    make the preferences file unreadable on an installation where the broker
    will not import.
    """
    text = str(raw or "").strip()
    if not text:
        return default
    folded = text.casefold()
    for value, label in table:
        if folded in (value.casefold(), label.casefold()):
            return value
    return default


def _kv_type(raw, default: str) -> str:
    """A cache type as ``llama.cpp`` spells it, from whatever the box holds."""
    try:
        import mc_llm_context

        table = tuple(mc_llm_context.KV_TYPE_LABELS)
    except Exception:
        return str(raw or default)
    return _resolve(raw, table, default)


def remember(**values) -> dict:
    """Update shared preferences. Returns the stored result.

    Anything in :data:`HOSTED` is written to the Settings page as well as to
    the file, because the Settings page is what :func:`preferences` reads back
    for those keys: writing only the file would make ``remember`` look like it
    had worked and change nothing anybody could observe.
    """
    with _lock:
        current = preferences()
        current.update({k: v for k, v in values.items() if v is not None})
        current["version"] = SCHEMA_VERSION
        _write(PREFERENCES_FILE, current)
        _publish(values)
        return current


# What each hosted preference is called on the Settings page, and how a value
# has to be spelled to survive the round trip through a Gradio control there.
_PUBLISHED = (
    ("context_mode", OPT_CONTEXT_MODE, label_for_context_mode),
    ("context_buffer_gb", OPT_CONTEXT_BUFFER, float),
    ("context_size", OPT_CONTEXT_SIZE, int),
    ("kv_type_k", OPT_KV_TYPE_K, str),
    ("kv_type_v", OPT_KV_TYPE_V, str),
)


def _publish(values: dict) -> None:
    """Write the hosted half of ``values`` back to the Settings page.

    Best-effort in every direction: no host, an option the host has not
    registered, or a value it will not take are all reasons to leave the
    setting alone and let the file answer, never reasons to lose the write that
    has already succeeded.
    """
    wanted = [(name, cast(values[key])) for key, name, cast in _PUBLISHED
              if values.get(key) is not None]
    if not wanted:
        return
    try:
        from modules import shared

        for name, value in wanted:
            shared.opts.set(name, value)
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: could not persist LLM settings to the host", exc_info=True)


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


@dataclass
class KreaSession:
    """One Krea 2 prompt, and the ordered references it was written about.

    What is *not* here is the point. No image bytes, no data URLs, no temporary
    upload paths -- section 14 rules all three out, and a history file that
    quietly grew a base64 JPEG per entry would be a history file nobody could
    open. What is kept is the two things that stay useful once the files have
    gone: the names, so a loaded session can say which pictures it was about,
    and the captions, so it can say what they contained.

    The two lists are parallel and ordered, and that order is the user's
    visible one: ``reference_names[0]`` and ``reference_captions[0]`` are Image
    1. Loading a session restores them as information -- it does not pretend
    the files are still attached, because they are not, and re-generating a
    reference-aware prompt means re-uploading them.

    ``creativity`` defaults to 1 rather than to 0, and the default is what a
    session written before the slider existed reads back as. That is not a
    guess: 1 is defined as the configuration the Krea writer used before there
    was anything to choose, so an old entry is being labelled with the value it
    actually ran at.

    ``creative_seed`` and ``recipe`` are Creative Mode's, and are what make a
    saved prompt something you can go back to rather than only read. The recipe
    is stored as its compact ``axis=variant_id`` form: the ids are stable across
    library versions by contract, and storing the rendered sentences instead
    would be storing a paragraph of English that a later package could no longer
    explain. Both are empty on a session written without Creative Mode.
    """

    identifier: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    prompt: str = ""
    result: str = ""
    seed: int = 0
    creativity: int = 1
    creative_seed: int = -1
    recipe: str = ""
    reference_names: list = field(default_factory=list)
    reference_captions: list = field(default_factory=list)

    @property
    def label(self) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created))
        counted = len(self.reference_names)
        suffix = ""
        if counted:
            suffix = f" · {counted} ref{'' if counted == 1 else 's'}"
        return f"{stamp} — {self.prompt[:48] or 'untitled'}{suffix}"


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


def krea_sessions() -> list[KreaSession]:
    return [KreaSession(**_fields(KreaSession, row))
            for row in _read(KREA_HISTORY_FILE, {}).get("sessions", [])]


def save_krea_session(session: KreaSession) -> KreaSession:
    _append(KREA_HISTORY_FILE, session)
    return session


def delete_krea_session(identifier: str) -> None:
    _remove(KREA_HISTORY_FILE, identifier)


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


REPLACE_ATTEMPTS = 5
"""How many times an atomic replace is retried before the write is given up on.

Windows refuses ``os.replace`` with ``[WinError 5] Access is denied`` while
anything at all holds the destination open -- an anti-virus scanner reading a
file that was just written, a backup agent, a search indexer -- and every one
of those holds it for a moment rather than for good. The first attempt is the
one that fails; the second, a few milliseconds later, succeeds. POSIX renames
never hit this, which is why the loop was not there to begin with, and why the
only symptom was a Windows user losing a preference every so often.
"""

REPLACE_BACKOFF_SECONDS = 0.05


def _write(filename: str, document: dict) -> None:
    path = _path(filename)
    temporary = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(REPLACE_BACKOFF_SECONDS * (attempt + 1))
    except OSError:
        logger.warning("Model Chain: could not write %s", filename, exc_info=True)
    finally:
        # A replace that never happened leaves the temporary file behind, and
        # the directory is one the user opens to read their settings.
        if temporary and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
