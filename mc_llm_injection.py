"""A thread's injected words: what its replies lean towards, two ways.

"Is there a way to make a list of words the the LLM becomes more likely to
include in its response? Like a word injection into the mind of the LLM that
can be used to subconsciously steer it." Of the three ways put to the user --
the odds, a note in the prompt, a control vector -- "I like all three options,
give it a menu for 'Injection'" (2026-11-09). This module is the first two; the
third is :mod:`mc_llm_control_vector`, which is the model's rather than a
thread's.

**The words.** One list per thread (:func:`for_thread`), kept beside the threads
in ``injection.json`` for the emotion's reason (:mod:`mc_llm_emotions`): a write
to the conversation's own file is a new revision of it. A word or a phrase a
line -- commas part a line too -- each with a strength after it if it wants one
other than :data:`DEFAULT_STRENGTH`: ``rain +2``, ``tapestry -4``,
``delve ban`` (:func:`parse`). A strength needs its sign, a ``:`` or an ``=``
before it, so ``Route 66`` stays a phrase. A branch keeps its thread's list
(:func:`carry`); a deleted thread's is forgotten (:func:`forget`).

**Weigh the odds** (``odds``, on unless turned off): llama-server's own
``logit_bias``, sent with every reply in the thread
(``mc_llm_sessions._conversation``). Each time the model picks its next token a
listed word's token gets that much added to its score -- about 2.7 times as
likely at +1, 20 times at +3 -- and a banned one is never picked. The model
never reads the list: once a nudged word is out, it carries on from it as if it
had meant to say it, which is the steering. It is no part of the prompt, so a
change of the list makes llama-server read nothing again.

It works on the model's tokens, not on words. A word mid-sentence is a
different token from the same word after a full stop, so each word is sent as
it is written and with a capital, a space before each (:func:`forms`); a word
made rarer or banned is sent without the space too, which also catches it at
the start of a line or after an asterisk. A boosted word is not sent without
it: that token joins onto the end of other words. Each form is read by the
running server's tokenizer (``mc_llm_tokens.Counter.pieces``), and a form it
splits into pieces has its **first piece** changed alone -- llama-server would
change every piece of it, and a piece such as ``cent`` is inside other words.
The first piece starts other words too, which the read-back says
(:func:`described`).

**Tell the model** (``prompt``, off unless turned on): a note at the end of the
thread's system prompt (:func:`note`), after any summary and before the emotion
(``mc_llm_conversation_ops._window_extra``), in every request in the thread --
the words to use where they fit, the words to avoid. The model reads it, so it
uses the words where they make sense. A change of the note is a new front to
every later prompt, so the conversation is read again under it in the
background at once, as a change of emotion is.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("model_chain")

FILE_NAME = "injection.json"
VERSION = 1

DEFAULT_STRENGTH = 2.0
"""What a word with no strength of its own is given: about seven times as likely
at each place it could go."""

STRENGTH_LIMIT = 10.0
"""The furthest a strength goes either way. Past this a word is all but forced,
or all but gone, which is what ``ban`` is for."""

MAX_WORDS = 64
"""Words used from a list; the rest are named as left out."""

MAX_WORD_CHARS = 64
MAX_TEXT_BYTES = 16 * 1024

BAN = "ban"
NOTE_HEADING = "[Word choice]"

_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
_TRAILING = re.compile(
    rf"^(?P<word>.+?)\s*(?:[:=]\s*(?P<given>[+-]?{_NUMBER}|{BAN})"
    rf"|\s(?P<signed>[+-]{_NUMBER})|\s(?P<ban>{BAN}))\s*$",
    re.IGNORECASE)
_QUOTES = ('"', "'", "“”", "‘’")
_SPACES = re.compile(r"\s+")

_lock = threading.Lock()


class Refused(ValueError):
    """Something that cannot be kept, in a sentence for the screen."""


@dataclass(frozen=True)
class Entry:
    """One listed word and what it does."""

    word: str
    strength: float = DEFAULT_STRENGTH
    banned: bool = False

    @property
    def shown(self) -> str:
        if self.banned:
            return BAN
        return f"{self.strength:+g}"

    @property
    def avoided(self) -> bool:
        return self.banned or self.strength < 0


@dataclass(frozen=True)
class Record:
    """A thread's injection as kept."""

    words: str = ""
    odds: bool = True
    prompt: bool = False

    def to_dict(self) -> dict:
        return {"words": self.words, "odds": self.odds, "prompt": self.prompt}

    @property
    def default(self) -> bool:
        return self == Record()


# -- reading a list ------------------------------------------------------------ #


def _unquoted(word: str) -> str:
    word = word.strip()
    for pair in _QUOTES:
        opening, closing = (pair, pair) if len(pair) == 1 else (pair[0], pair[1])
        if len(word) >= 2 and word[0] == opening and word[-1] == closing:
            return word[1:-1].strip()
    return word


def _strength(text: str) -> tuple[float, bool]:
    if text.casefold() == BAN:
        return -math.inf, True
    number = float(text)
    return min(max(number, -STRENGTH_LIMIT), STRENGTH_LIMIT), False


def _items(text: str):
    for line in str(text or "").splitlines():
        for item in line.split(","):
            item = item.strip()
            if item:
                yield item


def parse(text: str) -> tuple[list[Entry], list[str]]:
    """The entries ``text`` lists, and what was left out of them, in sentences.

    A word listed twice is kept once, where it was first listed, at the strength
    it was last given. A word longer than :data:`MAX_WORD_CHARS`, and every word
    after the first :data:`MAX_WORDS`, are left out and named.
    """
    found: dict[str, Entry] = {}
    left_out: list[str] = []
    for item in _items(text):
        matched = _TRAILING.match(item)
        word, strength, banned = item, DEFAULT_STRENGTH, False
        if matched:
            given = matched.group("given") or matched.group("signed") or matched.group("ban")
            word = matched.group("word")
            strength, banned = _strength(given)
        word = _SPACES.sub(" ", _unquoted(word))
        if not word:
            continue
        if len(word) > MAX_WORD_CHARS:
            left_out.append(f"“{word[:24]}…” is longer than "
                            f"{MAX_WORD_CHARS} characters")
            continue
        entry = Entry(word=word, strength=0.0 if banned else round(strength, 3),
                      banned=banned)
        name = word.casefold()
        if name in found:
            found[name] = entry
            continue
        if len(found) >= MAX_WORDS:
            left_out.append(f"“{word}” is past the first {MAX_WORDS} words")
            continue
        found[name] = entry
    return list(found.values()), left_out


def forms(entry: Entry) -> tuple[str, ...]:
    """The texts of ``entry`` that are sent: as written and with a capital, a
    space before each; without the space too for a word made rarer or banned."""
    word = entry.word
    capital = word[:1].upper() + word[1:]
    found = [" " + word, " " + capital]
    if entry.avoided:
        found += [word, capital]
    return tuple(dict.fromkeys(found))


# -- what is sent ----------------------------------------------------------------- #


def _clamped(value: float) -> float:
    return round(min(max(value, -STRENGTH_LIMIT), STRENGTH_LIMIT), 3)


def logit_bias(entries, counter) -> list:
    """``entries`` as llama-server's ``logit_bias``: ``[[token, strength], ...]``,
    ``False`` for a banned token.

    ``counter`` is a :class:`mc_llm_tokens.Counter` on the server the request
    goes to. Each form's first piece is the token changed; two words that share
    one add up, within :data:`STRENGTH_LIMIT`, and a ban wins over any strength.
    A form the tokenizer could not be asked about is sent as its text, which the
    server reads itself -- every piece of it, which for the one-piece words most
    words are is the same thing.
    """
    strengths: dict = {}
    banned: list = []
    texts: dict[str, float | bool] = {}
    for entry in entries or ():
        if not entry.banned and not entry.strength:
            continue
        for form in forms(entry):
            pieces = counter.pieces(form) if counter is not None else None
            if pieces is None:
                if entry.banned:
                    texts[form] = False
                elif texts.get(form) is not False:
                    texts[form] = _clamped(float(texts.get(form) or 0.0) + entry.strength)
                continue
            if not pieces:
                continue
            token = pieces[0][0]
            if entry.banned:
                if token not in banned:
                    banned.append(token)
            else:
                strengths[token] = _clamped(strengths.get(token, 0.0) + entry.strength)
    found: list = [[token, False] for token in banned]
    found += [[token, value] for token, value in strengths.items()
              if token not in banned and value]
    found += [[text, value] for text, value in texts.items() if value is False or value]
    return found


def note(entries) -> str:
    """The note at the end of the thread's system prompt for ``entries``, or
    ``""`` when there is nothing to say. Words at no strength are left out."""
    use = [entry.word for entry in entries or () if not entry.avoided and entry.strength]
    avoid = [entry.word for entry in entries or () if entry.avoided]
    if not use and not avoid:
        return ""
    lines = [NOTE_HEADING]
    if use:
        lines.append("Where they fit naturally, use some of these words in your replies: "
                     + ", ".join(use) + ".")
    if avoid:
        lines.append("Avoid these words: " + ", ".join(avoid) + ".")
    lines.append("Do not mention this note.")
    return "\n".join(lines)


def described(entries, counter) -> list[dict]:
    """How each entry is read, for the menus: its word, its strength as shown,
    the pieces the model's tokenizer reads it in, and a sentence when that is
    not one piece. ``pieces`` is ``None`` with no tokenizer to ask."""
    found = []
    for entry in entries or ():
        form = " " + entry.word
        pieces = counter.pieces(form) if counter is not None else None
        texts = [str(piece).strip() or str(piece) for _, piece in pieces] if pieces else None
        said = ""
        if texts and len(texts) > 1:
            joined = " · ".join(texts)
            said = (f"Split into {joined}: only ‘{texts[0]}’ is changed, and "
                    "every word that starts with it.")
        found.append({"word": entry.word, "strength": entry.shown, "banned": entry.banned,
                      "pieces": texts, "note": said})
    return found


# -- each thread's list ------------------------------------------------------------- #


def path() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / FILE_NAME


def _key(character, thread_id) -> str:
    return f"{str(character or '').strip()}/{str(thread_id or '').strip()}"


def _clean(raw) -> Record | None:
    if not isinstance(raw, dict):
        return None
    words = raw.get("words")
    words = words if isinstance(words, str) else ""
    odds = raw.get("odds")
    prompt = raw.get("prompt")
    record = Record(words=words[:MAX_TEXT_BYTES],
                    odds=odds if isinstance(odds, bool) else True,
                    prompt=prompt if isinstance(prompt, bool) else False)
    return None if record.default else record


def _read() -> dict:
    where = path()
    if not where.is_file():
        return {}
    try:
        from prompt_master.core.config import read_json

        data = read_json(where)
    except Exception:
        logger.warning("Model Chain: the threads' injected words could not be read; each "
                       "thread has none until its list is written again")
        return {}
    threads = data.get("threads") if isinstance(data, dict) else None
    if not isinstance(threads, dict):
        return {}
    found = {}
    for name, raw in threads.items():
        record = _clean(raw)
        if isinstance(name, str) and record is not None:
            found[name] = record
    return found


def _write(threads: dict) -> None:
    from prompt_master.core.config import atomic_write_json

    atomic_write_json(path(), {"version": VERSION,
                               "threads": {name: record.to_dict()
                                           for name, record in threads.items()}})


def for_thread(character, thread_id) -> Record:
    """What this thread keeps; the defaults for a thread that keeps nothing."""
    with _lock:
        return _read().get(_key(character, thread_id), Record())


def save(character, thread_id, *, words=None, odds=None, prompt=None) -> Record:
    """Keep what is given for this thread; what is not given stays as it was.

    Returns what the thread keeps now. Raises :class:`Refused` for no thread or
    a list longer than :data:`MAX_TEXT_BYTES`.
    """
    if not str(character or "").strip() or not str(thread_id or "").strip():
        raise Refused("Open a thread first: its words are kept with it.")
    if words is not None:
        words = str(words)
        if len(words.encode("utf-8", "replace")) > MAX_TEXT_BYTES:
            raise Refused("That list is too long to keep.")
    name = _key(character, thread_id)
    with _lock:
        threads = _read()
        before = threads.get(name, Record())
        record = Record(words=before.words if words is None else words,
                        odds=before.odds if odds is None else bool(odds),
                        prompt=before.prompt if prompt is None else bool(prompt))
        if record == before:
            return record
        if record.default:
            threads.pop(name, None)
        else:
            threads[name] = record
        _write(threads)
    return record


def carry(character, source_thread, made_thread) -> None:
    """Give a branch the list of the thread it was made from. Never fails a caller."""
    try:
        if not str(source_thread or "").strip() or not str(made_thread or "").strip():
            return
        with _lock:
            threads = _read()
            record = threads.get(_key(character, source_thread))
            if record is None:
                return
            threads[_key(character, made_thread)] = record
            _write(threads)
    except Exception:
        logger.debug("Model Chain: could not carry a thread's injected words", exc_info=True)


def forget(character, thread_id) -> None:
    """Drop a deleted thread's list. Never fails a caller."""
    try:
        name = _key(character, thread_id)
        with _lock:
            threads = _read()
            if name in threads:
                del threads[name]
                _write(threads)
    except Exception:
        logger.debug("Model Chain: could not forget a deleted thread's injected words",
                     exc_info=True)


# -- what a thread's requests carry --------------------------------------------- #


def entries_of(record: Record) -> list[Entry]:
    return parse(record.words)[0]


def odds_for(character, thread_id) -> tuple[Entry, ...]:
    """The entries a reply in this thread weighs the odds with; none when off."""
    record = for_thread(character, thread_id)
    if not record.odds:
        return ()
    return tuple(entries_of(record))


def note_for(character, thread_id) -> str:
    """The note this thread's system prompt ends with, or ``""``."""
    record = for_thread(character, thread_id)
    if not record.prompt:
        return ""
    return note(entries_of(record))


def label(record: Record) -> str:
    """The Injection section's label: what is in force while it is folded."""
    entries = entries_of(record)
    if not entries:
        return "Injection"
    count = f"{len(entries)} word" + ("" if len(entries) == 1 else "s")
    ways = [name for name, on in (("odds", record.odds), ("prompt", record.prompt)) if on]
    if not ways:
        return f"Injection · {count}, off"
    return f"Injection · {count} · " + " and ".join(ways)


def view(character, thread_id, counter=None) -> dict:
    """What the menus show of this thread's injection: what is kept, how each
    word is read, what was left out, and the note as the model reads it."""
    record = for_thread(character, thread_id)
    entries, left_out = parse(record.words)
    return {"words": record.words, "odds": record.odds, "prompt": record.prompt,
            "entries": described(entries, counter), "left_out": left_out,
            "read": bool(counter is not None and getattr(counter, "exact", False)),
            "note": note(entries) if record.prompt else "", "label": label(record)}


__all__ = ["BAN", "DEFAULT_STRENGTH", "Entry", "FILE_NAME", "MAX_TEXT_BYTES", "MAX_WORDS",
           "MAX_WORD_CHARS", "NOTE_HEADING", "Record", "Refused", "STRENGTH_LIMIT", "carry",
           "described", "entries_of", "for_thread", "forget", "forms", "label", "logit_bias",
           "note", "note_for", "odds_for", "parse", "path", "save", "view"]
