"""A thread's generation settings: how its replies are sampled, kept with the thread.

"It seems for LLM studio that my generation settings (like temperature) are not
being kept. I have to keep setting them again when i return to the
conversation. These should be kept per conversation. I would also like more
controls there ... Im not able to get random enough output, and infact, i get a
lot of repetition. Lets surface controls that can help with that" (2026-11-09).

**What there was.** Four controls -- Temperature, Top-p, Reply tokens, Seed --
written to the *character's* file whenever one moved, and drawn at the
package's defaults whenever Forge built the page: the character's values were
put back only by a change of character. So a page loaded later, or a tab
returned to, showed the defaults and sent them with every reply -- the choice
was in the file and not in force. And a seed typed once stayed in the file
while the box showed -1 again, which the request read as "no preference" and
answered with the file's seed: the same reply on every Regenerate.

**What is built.** Every thread keeps its own choices, by character and thread
(:func:`for_thread`), in one file beside the threads -- not in the
conversation's own file, for the emotion's reason (:mod:`mc_llm_emotions`): a
write there is a new revision of the conversation, which would make the other
window's next Send stale. Every request in the thread is made from them,
whichever view sends it (``mc_llm_conversation_ops._capture``).

A setting a thread has not chosen follows what applies without a choice
(:func:`base`): the character's, for the four it has always had, and for the
rest what llama-server uses for a request that does not say -- the managed
profile's sampler fields (sent with every request), else the model file's own
recommendation (``general.sampling.*``, which llama-server takes as its
default), else llama-server's built-in default. That is what is drawn for it,
and nothing more than the four is sent for it (:func:`request_fields`), so a
thread nobody has touched is sampled exactly as it was. What a thread chose is
sent after the profile's fields, so the thread's choice wins.

A new thread starts with the choices of the thread it was made from, as a
branch does (:func:`carry`); **Reset** puts a thread back to what applies
without a choice (:func:`reset`); a deleted thread's are forgotten
(:func:`forget`).

**The controls** (:data:`SETTINGS`) are what llama-server b10621 reads from a
request (``tools/server/server-schema.cpp``) that bear on variety and
repetition, with its own defaults (``common/common.h``,
``common_params_sampling``). Two things read from its source decide the ranges.
Its repetition penalties and DRY are given every token of the prompt
(``server-context.cpp``, ``init_sampler``), but each looks back only as far as
its range -- 64 tokens by default, less than one reply -- so phrases repeated
from earlier replies go unpenalised until a range is raised (**Penalty range**,
**DRY range**). And both ranges are refused below zero there (no -1 for "the
whole context"), so their sliders start at 0, which turns them off. Left out:
Mirostat (it replaces the rest of the chain, which would make every other
control here do nothing), adaptive-p (it acts only when named in a request's
``samplers`` list), typical-p and top-nσ (neither is a repetition or a variety
control the others do not already give).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("model_chain")

FILE_NAME = "generation.json"
"""Under the LLM data root, beside ``emotions.json`` and ``start_with.json``."""

VERSION = 1

VARIETY = "variety"
REPETITION = "repetition"
LENGTH = "length"

GROUPS = ((VARIETY, "Variety"), (REPETITION, "Repetition"), (LENGTH, "Length and seed"))
"""The sections of the Generation controls, in order, with their headings."""


@dataclass(frozen=True)
class Setting:
    """One control: what a request and this file call it, and its range.

    ``default`` is llama-server b10621's own for a request that does not say.
    ``model_key`` is the name the model file may recommend a value under
    (``general.sampling.<model_key>``). ``character`` names the attribute of
    the character the setting has always been read from; those four are sent
    with every request whether a thread chose them or not.
    """

    key: str
    label: str
    group: str
    minimum: float
    maximum: float
    step: float
    default: float = 0.0
    whole: bool = False
    info: str = ""
    model_key: str = ""
    character: str = ""


SETTINGS: tuple[Setting, ...] = (
    # Drawn two to a row, in this order: each pair is two halves of one idea.
    # -- variety ----------------------------------------------------------- #
    Setting("temperature", "Temperature", VARIETY, 0.0, 2.0, 0.05, 0.8,
            info="Higher is more varied", character="temperature"),
    Setting("dynatemp_range", "Dynamic temperature", VARIETY, 0.0, 1.5, 0.05, 0.0,
            info="0 = off; ± this, higher where the model is unsure"),
    Setting("min_p", "Min-p", VARIETY, 0.0, 1.0, 0.01, 0.05, info="0 = off",
            model_key="min_p"),
    Setting("top_p", "Top-p", VARIETY, 0.0, 1.0, 0.01, 0.95, info="1 = off",
            character="top_p"),
    Setting("xtc_probability", "XTC", VARIETY, 0.0, 1.0, 0.05, 0.0,
            info="0 = off; how often the likeliest words are dropped",
            model_key="xtc_probability"),
    Setting("xtc_threshold", "XTC threshold", VARIETY, 0.0, 0.5, 0.01, 0.1,
            info="Words likelier than this can be dropped", model_key="xtc_threshold"),
    Setting("top_k", "Top-k", VARIETY, 0, 200, 1, 40, whole=True, info="0 = off",
            model_key="top_k"),
    # -- repetition -------------------------------------------------------- #
    Setting("repeat_penalty", "Repeat penalty", REPETITION, 1.0, 2.0, 0.01, 1.0,
            info="1 = off", model_key="penalty_repeat"),
    Setting("repeat_last_n", "Penalty range", REPETITION, 0, 4096, 16, 64, whole=True,
            info="Tokens the penalties look back over; 0 = off",
            model_key="penalty_last_n"),
    Setting("presence_penalty", "Presence penalty", REPETITION, 0.0, 2.0, 0.05, 0.0,
            info="0 = off"),
    Setting("frequency_penalty", "Frequency penalty", REPETITION, 0.0, 2.0, 0.05, 0.0,
            info="0 = off"),
    Setting("dry_multiplier", "DRY", REPETITION, 0.0, 5.0, 0.05, 0.0,
            info="0 = off; 0.8 is a usual start"),
    Setting("dry_penalty_last_n", "DRY range", REPETITION, 0, 8192, 64, 64, whole=True,
            info="Tokens DRY looks back over; 0 = off"),
    Setting("dry_base", "DRY base", REPETITION, 1.0, 4.0, 0.05, 1.75,
            info="How fast it grows with the run"),
    Setting("dry_allowed_length", "DRY allowed length", REPETITION, 0, 20, 1, 2, whole=True,
            info="Tokens repeated before it starts"),
    # -- length and seed --------------------------------------------------- #
    Setting("reply_tokens", "Reply tokens", LENGTH, 64, 4096, 64, 512, whole=True,
            character="max_reply_tokens"),
    Setting("seed", "Seed", LENGTH, -1, 2 ** 31 - 2, 1, -1, whole=True,
            info="-1: a new seed each reply", character="seed"),
)
"""Every control, in the order they are drawn."""

BY_KEY = {setting.key: setting for setting in SETTINGS}

KEYS = tuple(setting.key for setting in SETTINGS)

SENT_ALWAYS = tuple(setting.key for setting in SETTINGS if setting.character)
"""The four every request carries, chosen or not: they have their own fields on
the request (``ChatRequest.temperature`` and the rest)."""

SENT_IF_CHOSEN = tuple(setting.key for setting in SETTINGS if not setting.character)
"""The rest, sent only for a thread that chose them (:func:`request_fields`).
Their names are llama-server's own; ``prompt_master.inference.llama_client``
lets exactly these through (``REQUEST_SAMPLER_FIELDS``)."""

_lock = threading.Lock()


class Refused(ValueError):
    """A setting that cannot be kept, in a sentence for the screen."""


def path() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / FILE_NAME


def _key(character, thread_id) -> str:
    return f"{str(character or '').strip()}/{str(thread_id or '').strip()}"


def value_of(setting: Setting, value):
    """``value`` as ``setting`` keeps it -- a number in its range, whole where it
    is whole -- or ``None`` for something that is not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    number = min(max(number, float(setting.minimum)), float(setting.maximum))
    if setting.whole:
        return int(round(number))
    return round(number, 4)


def _clean(record) -> dict:
    if not isinstance(record, dict):
        return {}
    found = {}
    for key in KEYS:
        if key in record:
            kept = value_of(BY_KEY[key], record[key])
            if kept is not None:
                found[key] = kept
    return found


def _read() -> dict:
    where = path()
    if not where.is_file():
        return {}
    try:
        from prompt_master.core.config import read_json

        data = read_json(where)
    except Exception:
        logger.warning("Model Chain: the threads' generation settings could not be read; "
                       "each thread follows its defaults until one is changed again")
        return {}
    threads = data.get("threads") if isinstance(data, dict) else None
    if not isinstance(threads, dict):
        return {}
    found = {}
    for key, record in threads.items():
        kept = _clean(record)
        if isinstance(key, str) and kept:
            found[key] = kept
    return found


def _write(threads: dict) -> None:
    from prompt_master.core.config import atomic_write_json

    atomic_write_json(path(), {"version": VERSION, "threads": dict(threads)})


# -- each thread's choices ------------------------------------------------------ #


def for_thread(character, thread_id) -> dict:
    """What this thread chose, by key; ``{}`` for a thread that chose nothing."""
    with _lock:
        return dict(_read().get(_key(character, thread_id), {}))


def choose(character, thread_id, key, value) -> dict:
    """Keep ``value`` for one setting of this thread, or forget it with ``None``.

    Returns what the thread has chosen now. A value outside the setting's range
    is held at its end, as llama-server holds one; a key that is no setting, or
    a value that is no number, is refused.
    """
    setting = BY_KEY.get(str(key or ""))
    if setting is None:
        raise Refused("That is not a generation setting.")
    if not str(character or "").strip() or not str(thread_id or "").strip():
        raise Refused("Open a thread first: its settings are kept with it.")
    kept = None if value is None else value_of(setting, value)
    if value is not None and kept is None:
        raise Refused(f"{setting.label} takes a number.")
    name = _key(character, thread_id)
    with _lock:
        threads = _read()
        record = dict(threads.get(name, {}))
        if kept is None:
            if setting.key not in record:
                return record
            del record[setting.key]
        else:
            if record.get(setting.key) == kept:
                return record
            record[setting.key] = kept
        if record:
            threads[name] = record
        else:
            threads.pop(name, None)
        _write(threads)
    return record


def reset(character, thread_id) -> None:
    """Forget every choice of this thread: it follows its defaults again."""
    name = _key(character, thread_id)
    with _lock:
        threads = _read()
        if name in threads:
            del threads[name]
            _write(threads)


def carry(character, source_thread, made_thread) -> None:
    """Give a thread made from another -- a branch, or a new thread started
    from one -- the choices of the one it came from. Never fails a caller."""
    try:
        if not str(source_thread or "").strip() or not str(made_thread or "").strip():
            return
        with _lock:
            threads = _read()
            record = threads.get(_key(character, source_thread))
            if not record:
                return
            threads[_key(character, made_thread)] = dict(record)
            _write(threads)
    except Exception:
        logger.debug("Model Chain: could not carry a thread's generation settings",
                     exc_info=True)


def forget(character, thread_id) -> None:
    """Drop a deleted thread's choices. Never fails a caller."""
    try:
        reset(character, thread_id)
    except Exception:
        logger.debug("Model Chain: could not forget a deleted thread's generation settings",
                     exc_info=True)


# -- what applies ------------------------------------------------------------------ #


def _configuration():
    try:
        import mc_llm_runtime

        return mc_llm_runtime.config()
    except Exception:
        logger.debug("Model Chain: no LLM configuration to read sampling defaults from",
                     exc_info=True)
        return None


def model_recommendations(configuration=None) -> dict:
    """What the model file recommends, by key: llama-server's default for it.

    Read from the file's own metadata (``general.sampling.*``), which
    llama-server applies to every request that does not say otherwise --
    ``common_init_sampler_from_model``, unless a flag on its command line set
    the value, and this extension sets none. ``{}`` for a model with none, or
    none configured.
    """
    found: dict = {}
    model = getattr(configuration, "model", None)
    if not model:
        return found
    try:
        import mc_gguf

        described = mc_gguf.describe(model)
    except Exception:
        logger.debug("Model Chain: could not read the model's sampling recommendations",
                     exc_info=True)
        return found
    metadata = getattr(described, "metadata", None) or {}
    for setting in SETTINGS:
        if not setting.model_key:
            continue
        kept = value_of(setting, metadata.get(f"general.sampling.{setting.model_key}"))
        if kept is not None:
            found[setting.key] = kept
    return found


def profile_fields(configuration=None) -> dict:
    """The managed profile's sampler fields, by key: sent with every request."""
    profile = getattr(configuration, "profile", None)
    if profile is None:
        return {}
    try:
        from prompt_master.models import managed_profiles

        fields = managed_profiles.sampler_arguments(profile)
    except Exception:
        logger.debug("Model Chain: could not read the profile's sampler fields",
                     exc_info=True)
        return {}
    found = {}
    for key, value in fields.items():
        setting = BY_KEY.get(key)
        kept = value_of(setting, value) if setting is not None else None
        if kept is not None:
            found[key] = kept
    return found


def base(character=None, configuration=None) -> dict:
    """Every setting's value for a thread that chose none, by key.

    In llama-server's own order of precedence for the settings beyond the four:
    its built-in default, then the model file's recommendation, then the
    managed profile's field. The four come from the character, as they always
    have -- or the package's defaults, for no character.
    """
    found = {setting.key: value_of(setting, setting.default) for setting in SETTINGS
             if not setting.character}
    if configuration is None:
        configuration = _configuration()
    found.update(model_recommendations(configuration))
    found.update(profile_fields(configuration))
    if character is None:
        from prompt_master.chat.characters import Character

        character = Character(name="")
    for setting in SETTINGS:
        if setting.character:
            kept = value_of(setting, getattr(character, setting.character, None))
            found[setting.key] = (kept if kept is not None
                                  else value_of(setting, setting.default))
    return found


def in_force(record, defaults: dict) -> dict:
    """Every setting's value for a thread: its choice, else what applies without one."""
    found = dict(defaults)
    found.update(_clean(record))
    return found


def request_fields(record) -> dict:
    """What a request carries beyond the four for this thread: what it chose."""
    chosen = _clean(record)
    return {key: chosen[key] for key in SENT_IF_CHOSEN if key in chosen}


def shown(setting: Setting, value) -> str:
    """``value`` as a person reads it: no trailing zeros, -1 as random."""
    if setting.key == "seed" and value == -1:
        return "random"
    if setting.whole:
        return str(int(value))
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def label(record) -> str:
    """The section's label: what this thread chose, in a few words."""
    chosen = _clean(record)
    if not chosen:
        return "Generation"
    parts = [f"{BY_KEY[key].label} {shown(BY_KEY[key], chosen[key])}"
             for key in KEYS if key in chosen]
    more = len(parts) - 2
    return "Generation · " + " · ".join(parts[:2]) + (f" · +{more}" if more > 0 else "")


__all__ = ["BY_KEY", "FILE_NAME", "GROUPS", "KEYS", "LENGTH", "REPETITION", "Refused",
           "SENT_ALWAYS", "SENT_IF_CHOSEN", "SETTINGS", "Setting", "VARIETY", "base",
           "carry", "choose", "for_thread", "forget", "in_force", "label",
           "model_recommendations", "path", "profile_fields", "request_fields", "reset",
           "shown", "value_of"]
