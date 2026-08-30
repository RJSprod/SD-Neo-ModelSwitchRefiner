"""Which voices exist, what they are called, and which number each one is.

The registry is the single source of truth for voice identity, and it exists
because the three things a voice has -- a name, a file and a number -- were the
same thing in V1 and must not be.

    display name     "Alice", which a user may change at any time
    stable id        ``official:af_heart`` or ``clone:<uuid>``, which nothing
                     changes, ever
    runtime slot     a numeric sherpa speaker id, which is an address in a
                     block of floats

V1 stored one voice label and one speaker id in the model manifest, and they
disagreed: the label said ``af_heart`` and the id said 0, which in the upstream
sherpa Kokoro map is ``af_alloy``. Every reply this feature has ever spoken has
been spoken by Alloy. That is section 113's migration in one sentence, and it
is the reason nothing here derives an identity from a name or a name from an
identity.

What is stored where
--------------------
Official voices are not stored at all. They are *derived*, in SID order, from
the installed bundle's own speaker map -- pinned in ``managed-voice-models.json``
beside the checksum of the archive those names came from -- and validated
against the speaker count the installed model file actually declares. A bundle
whose count disagrees with the map produces no official list and a visible
warning rather than a list of names against the wrong numbers.

Custom voices live in ``clones/registry.json``: id, display name, language,
slot, and the canonical ``.bin`` the bank is built from. The slot is allocated
once and never reused by compaction (section 50), because a SID is what a
user's saved default points at.

The asterisk
------------
Section 41: a custom voice is shown as ``* Alice``. The asterisk is presentation
-- it is not in the display name, not in the id, and not in any filename -- so
renaming a clone to "Alice" cannot collide with an official "Alice" in
anything except how the two look side by side, which is exactly what the mark
is for.

The default
-----------
A stable id, never a number (section 44), stored in this registry's own file
rather than in the host's options. It lived in an option once, and that was a
real defect: an option is a component on the settings page as well as a stored
value, so Forge's "Apply settings" wrote the page's stamped-at-build-time copy
back over whatever "Set as Default" had just chosen. An older installation's
option is still read once, so nobody loses the default they had.

A default that no longer resolves falls back to a known-good official voice and
says so, because "Voice Chat stopped speaking" is a much worse answer to a
deleted clone than "the voice you had chosen is gone; using Heart".
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid

import mc_voice_bank as bank
import mc_voice_models as models
import mc_voice_paths as paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_VOICE = "model_chain_voice_tts_voice_id"
OPT_TEST_TEXT = "model_chain_voice_test_text"

DEFAULT_VOICE = "official:af_heart"
"""Where a fresh installation starts, and where a broken default lands.

Heart rather than Alloy, and this is the migration: an installation that has
been speaking as speaker 0 all along gets Heart, which is what its own manifest
has claimed the voice was since the beginning.
"""

DEFAULT_TEST_TEXT = "This is a test of voice cloning."
"""Section 45's exact default. Editable, and short on purpose -- an audition is
a second of speech, not a paragraph."""

MAX_TEST_CHARS = 400
MAX_NAME_CHARS = 48

SCHEMA = 1

ACCENTS = {
    "a": ("en-US", "American English"),
    "b": ("en-GB", "British English"),
}
"""The two Kokoro English families this release manages.

The second letter of a Kokoro speaker name is the gender and the first is the
language: ``af_`` and ``am_`` are American, ``bf_`` and ``bm_`` British.
Everything else in the bundle -- Spanish, French, Hindi, Italian, Japanese,
Portuguese and Chinese speakers -- keeps its slot in the bank untouched
(section 43) and is not offered by this release's English-only management.
"""

_NAME_OK = re.compile(r"^[\w][\w '\-.()&]*$", re.UNICODE)


class RegistryError(RuntimeError):
    """A voice operation that could not be completed. Never fatal."""


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _read() -> dict:
    try:
        found = json.loads(paths.registry_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": SCHEMA, "voices": []}
    if not isinstance(found, dict) or not isinstance(found.get("voices"), list):
        logger.warning("Model Chain: the Voice registry could not be read and was ignored")
        return {"schema": SCHEMA, "voices": []}
    return found


def _write(found: dict) -> None:
    """Replace the registry atomically. A half-written registry is a lost bank.

    Written beside the real file and renamed, on the same filesystem, for the
    same reason the bank is: a registry truncated by a power cut would leave
    clones on disk that nothing can name.
    """
    path = paths.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.new")
    staging.write_text(json.dumps(found, indent=2), encoding="utf-8")
    os.replace(staging, path)


def _custom_entries() -> list:
    base = bank.custom_base()
    found = []
    for item in _read().get("voices") or ():
        if not isinstance(item, dict) or not item.get("id"):
            continue
        slot = int(item.get("slot") or 0)
        found.append({
            "id": str(item["id"]),
            "display_name": str(item.get("display_name") or "Voice"),
            "label": f"* {item.get('display_name') or 'Voice'}",
            "type": "clone",
            "official": False,
            "language": str(item.get("language") or "en-US"),
            "accent": _accent_label(str(item.get("language") or "en-US")),
            "editable": True,
            "deletable": True,
            "slot": slot,
            "sid": base + slot,
            "created_at": str(item.get("created_at") or ""),
        })
    return sorted(found, key=lambda entry: entry["slot"])


def _accent_label(language: str) -> str:
    return "British English" if str(language).endswith("GB") else "American English"


def official() -> list:
    """Every installed official English voice, in SID order.

    Empty, with a warning, when the installed bundle's speaker count does not
    match the pinned map: showing names against numbers that might not be
    theirs is how a user ends up selecting Heart and hearing Michael.
    """
    count = _official_count()
    names = bank.official_names(count)
    found = []
    for sid, name in enumerate(names):
        family = ACCENTS.get(str(name)[:1] or "")
        if not family:
            continue
        language, accent = family
        found.append({
            "id": f"official:{name}",
            "display_name": _pretty(name),
            "label": _pretty(name),
            "type": "official",
            "official": True,
            "language": language,
            "accent": accent,
            "editable": False,
            "deletable": False,
            "slot": None,
            "sid": sid,
            "created_at": "",
        })
    return found


def _official_count() -> int:
    """How many speakers the *installed* model says it has.

    From the model file when it is there and from the bank manifest otherwise,
    and never from the pinned map -- the map is the thing being checked.
    """
    found = bank.manifest()
    if found.get("official_count"):
        return int(found["official_count"])
    try:
        return bank.speaker_capacity(models.bundle_paths("tts")["model"])
    except Exception:
        return 0


def _pretty(name: str) -> str:
    """``af_heart`` as ``Heart``. The prefix is the accent, shown as a group."""
    tail = str(name).split("_", 1)[-1] or str(name)
    return tail.replace("_", " ").strip().title()


def custom() -> list:
    return _custom_entries()


def entries() -> list:
    """Everything Settings shows: official first, then custom."""
    return official() + _custom_entries()


def lookup(voice_id: str):
    wanted = str(voice_id or "")
    for entry in entries():
        if entry["id"] == wanted:
            return entry
    return None


def highest_sid():
    """The largest SID any registered voice needs, or ``None`` for none.

    Read by the runtime handshake: a bank with fewer speakers than this is a
    bank in which a registered clone would be silently spoken by somebody else,
    and the honest response to that is to refuse the worker.
    """
    found = [entry["sid"] for entry in _custom_entries()]
    return max(found) if found else None


def warnings() -> list:
    """What is wrong that a user can see and act on. Section 75.

    Recovery rather than repair: nothing here rewrites anything. It reports the
    two states that make voices untrustworthy -- a speaker map that does not
    match the installed bundle, and a clone whose canonical file has gone -- so
    Settings can say so and typed Conversation carries on regardless.
    """
    found = []
    count = _official_count()
    if count and not bank.official_names(count):
        found.append(f"The installed Kokoro bundle reports {count} voices, which is not the "
                     f"number this version of Voice Chat has a name list for. Official voices "
                     f"are hidden until the text-to-speech model is reinstalled.")
    for entry in _custom_entries():
        try:
            source = paths.clone_file(_uuid_of(entry["id"]))
        except ValueError:
            found.append(f"The custom voice {entry['display_name']!r} has an id Voice Chat "
                         f"cannot use and was ignored.")
            continue
        if not source.is_file():
            found.append(f"The custom voice {entry['display_name']!r} is missing its voice "
                         f"file and cannot be spoken. Delete it in Settings → Voice Chat.")
    if _stored_default() and lookup(_stored_default()) is None:
        found.append("The voice you had chosen is no longer installed, so Voice Chat is using "
                     f"{_pretty(DEFAULT_VOICE.split(':', 1)[-1])}.")
    return found


def _uuid_of(voice_id: str) -> str:
    return str(voice_id or "").split(":", 1)[-1]


# --------------------------------------------------------------------------- #
# The default
# --------------------------------------------------------------------------- #


def _stored_default() -> str:
    """The default voice id, out of this registry's own file.

    It used to live in the Forge option alone, and that is where a real defect
    was. An option is a *component* on the settings page as well as a stored
    value, and "Apply settings" writes every component on that page back into
    the store -- including this one, whose browser-side value was stamped when
    the page was built and knows nothing about the "Set as Default" pressed
    since. Setting a default and then changing anything else on that page put
    the old value quietly back.

    A default voice is voice-library state rather than something anybody types,
    so it belongs beside the voices, in a file written atomically here that no
    settings form can reach. The option is still *read*, once, so an
    installation that set its default under an older build keeps it.
    """
    found = str(_read().get("default") or "").strip()
    if found:
        return found
    try:
        from modules import shared

        return str(getattr(shared.opts, OPT_VOICE, "") or "")
    except Exception:
        return ""


def default_id() -> str:
    """The configured default, or a known-good official voice.

    Never raises and never returns something that does not resolve: a caller of
    this is about to speak, and "the setting says a voice that is not there" is
    this function's problem rather than theirs.
    """
    wanted = _stored_default()
    if wanted and lookup(wanted) is not None:
        return wanted
    if lookup(DEFAULT_VOICE) is not None:
        return DEFAULT_VOICE
    found = official() or _custom_entries()
    return found[0]["id"] if found else DEFAULT_VOICE


def default_entry():
    return lookup(default_id())


def set_default(voice_id: str) -> dict:
    """Commit a new default. Section 44: selecting a row does not do this.

    Only an explicit "Set as Default" reaches here, which is what stops a user
    who is auditioning six voices from ending up with whichever one they
    clicked last.
    """
    entry = lookup(voice_id)
    if entry is None:
        raise RegistryError("That voice is not installed.")
    found = _read()
    found["default"] = entry["id"]
    found["schema"] = SCHEMA
    _write(found)
    logger.info("Model Chain: Voice default is now %s (%s)", entry["id"], entry["type"])
    return entry


def resolve(voice_id: str = "") -> tuple:
    """``(sid, entry)`` for a stable id, or the default. Validated, always.

    The only path from a name to a number in this feature, and the reason
    section 56 says a browser-supplied raw SID must never be trusted: a number
    that did not come through here is a number that can address any speaker in
    the bank, including a reserved slot that is not a registered voice.
    """
    entry = lookup(voice_id) if voice_id else None
    if entry is None:
        entry = default_entry()
    if entry is None:
        raise RegistryError("No voice is installed.")
    return int(entry["sid"]), entry


def test_text() -> str:
    try:
        from modules import shared

        found = str(getattr(shared.opts, OPT_TEST_TEXT, "") or "")
    except Exception:
        found = ""
    return found.strip() or DEFAULT_TEST_TEXT


def set_test_text(text: str) -> str:
    wanted = str(text or "").strip()[:MAX_TEST_CHARS] or DEFAULT_TEST_TEXT
    _remember(OPT_TEST_TEXT, wanted)
    return wanted


def _remember(name: str, value) -> None:
    try:
        from modules import shared

        shared.opts.set(name, value)
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: could not persist a Voice setting", exc_info=True)


# --------------------------------------------------------------------------- #
# Custom voices
# --------------------------------------------------------------------------- #


def check_name(name: str) -> str:
    """A display name that is a name, and cannot be anything else.

    Section 77 -- never build a path or a command from a display name. Nothing
    here does, and this check exists anyway: a name is going into HTML, into a
    JSON file and onto a button, and "safe because nobody uses it dangerously"
    is a property that survives exactly one refactor.
    """
    wanted = str(name or "").strip()
    if not wanted:
        raise RegistryError("Give the voice a name.")
    if len(wanted) > MAX_NAME_CHARS:
        raise RegistryError(f"That name is longer than {MAX_NAME_CHARS} characters.")
    if not _NAME_OK.match(wanted):
        raise RegistryError("A voice name can contain letters, numbers, spaces and "
                            "- ' . ( ) &.")
    return wanted


def free_slot() -> int:
    """The lowest unoccupied custom slot. Raises when the bank is full.

    Lowest rather than next: slots are never compacted, so deleting the clone
    in slot 1 and creating another gives the new one slot 1 and leaves every
    other SID exactly where it was.
    """
    taken = {entry["slot"] for entry in _custom_entries()}
    for slot in range(bank.CUSTOM_CAPACITY):
        if slot not in taken:
            return slot
    raise RegistryError("Custom voice capacity is full. Delete a custom voice before "
                        "creating another.")


def capacity() -> dict:
    used = len(_custom_entries())
    return {"used": used, "total": bank.CUSTOM_CAPACITY,
            "free": max(0, bank.CUSTOM_CAPACITY - used)}


def slot_sources(extra: dict = None) -> dict:
    """``{slot: canonical .bin path}`` for a bank build.

    ``extra`` is a slot the caller is proposing but has not committed, which is
    what makes a candidate bank buildable before its registry entry exists --
    section 55, step 4.
    """
    found = {}
    for entry in _custom_entries():
        try:
            source = paths.clone_file(_uuid_of(entry["id"]))
        except ValueError:
            continue
        if source.is_file():
            found[entry["slot"]] = str(source)
    for slot, source in (extra or {}).items():
        found[int(slot)] = str(source)
    return found


def rename(voice_id: str, display_name: str) -> dict:
    """Change what a custom voice is called, and nothing else.

    Section 46: the id, the slot, the SID and the canonical filename are all
    untouched, so a rename cannot invalidate a saved default, cannot require a
    bank rebuild, and cannot change what anybody hears.
    """
    entry = lookup(voice_id)
    if entry is None:
        raise RegistryError("That voice is not installed.")
    if not entry["editable"]:
        raise RegistryError("Official voices cannot be renamed.")
    wanted = check_name(display_name)
    found = _read()
    for item in found.get("voices") or ():
        if str(item.get("id")) == entry["id"]:
            item["display_name"] = wanted
            break
    _write(found)
    logger.info("Model Chain: a custom voice was renamed")
    return lookup(voice_id)


def delete(voice_id: str) -> dict:
    """Remove a custom voice, transactionally. Section 47.

    The order is the whole design. The default moves first, so that a deleted
    voice can never be the one the next reply resolves to; the bank is rebuilt
    and validated before anything is committed, so a failed rebuild leaves the
    voice installed rather than leaving a registry entry for a voice the bank
    no longer contains; and the canonical ``.bin`` is removed last, after the
    commit, because it is the one step that cannot be undone.

    Other custom slots are not compacted (T-BANK-9).
    """
    entry = lookup(voice_id)
    if entry is None:
        raise RegistryError("That voice is not installed.")
    if not entry["deletable"]:
        raise RegistryError("Official voices cannot be deleted.")

    if default_id() == entry["id"]:
        # Section 44: switch the default *before* the deletion commits, and
        # switch it to the known-good one rather than to whatever happens to be
        # first -- somebody who deletes a clone should land on Heart, not on
        # alphabetical order.
        others = [item for item in entries() if item["id"] != entry["id"]]
        fallback = next((item for item in others if item["id"] == DEFAULT_VOICE), None)
        fallback = fallback or (others[0] if others else None)
        if fallback is None:
            raise RegistryError("That is the only voice installed, so it cannot be deleted.")
        set_default(fallback["id"])

    remaining = {slot: source for slot, source in slot_sources().items()
                 if slot != entry["slot"]}
    _rebuild(remaining, validate_sid=None)

    found = _read()
    found["voices"] = [item for item in (found.get("voices") or ())
                       if str(item.get("id")) != entry["id"]]
    _write(found)
    bank.forget_voicepack(_uuid_of(entry["id"]))
    logger.info("Model Chain: a custom voice was deleted — slot %s stays reserved",
                entry["slot"])
    return entry


def register(display_name: str, language: str, source_bin, identifier: str = "") -> dict:
    """Install a finished clone into the bank, then into the registry.

    Section 55, in order and with the failure of every step meaning the same
    thing -- nothing is registered:

        1  the voicepack is validated and normalized
        2  the lowest free slot is proposed, not committed
        3  a complete candidate bank is built
        4  the bank is promoted and the worker restarted onto it
        5  the proposed SID is asked to say a short phrase through the ordinary
           production runtime
        6  only then is the registry entry written

    Step 5 is not ceremony. sherpa answers an out-of-range speaker by using
    speaker 0, so a bank whose metadata did not take would produce a clone that
    works perfectly and sounds like somebody else. The only way to know is to
    listen, and the only way to do that in software is to synthesize.
    """
    name = check_name(display_name)
    wanted = "en-GB" if str(language).upper().endswith("GB") else "en-US"
    internal = str(identifier or uuid.uuid4().hex)
    slot = free_slot()

    stored = bank.store_voicepack(internal, source_bin)
    try:
        _rebuild(slot_sources({slot: str(stored)}), validate_sid=bank.sid_for_slot(slot))
    except Exception:
        bank.forget_voicepack(internal)
        raise

    found = _read()
    found.setdefault("voices", []).append({
        "id": f"clone:{internal}",
        "display_name": name,
        "type": "clone",
        "official": False,
        "language": wanted,
        "editable": True,
        "deletable": True,
        "slot": slot,
        "sid": bank.sid_for_slot(slot),
        "source_bin": f"clones/{internal}.bin",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    found["schema"] = SCHEMA
    _write(found)
    logger.info("Model Chain: a custom voice was registered — slot %d, sid %d",
                slot, bank.sid_for_slot(slot))
    return lookup(f"clone:{internal}")


def rebuild() -> dict:
    """Rebuild the bank from the registry as it stands. Recovery, on demand."""
    return _rebuild(slot_sources(), validate_sid=None)


def _rebuild(slots: dict, validate_sid) -> dict:
    """Build, promote, restart and prove -- or put everything back.

    ``validate_sid`` of ``None`` still validates: an official voice is
    synthesized instead, because a bank that cannot speak at all is the failure
    a delete could otherwise cause and the one nobody would attribute to the
    delete.
    """
    import mc_voice_runtime as runtime
    import mc_voice_turn as turns

    built = bank.build(slots)
    was_loaded = runtime.status().get("running")

    turns.forget_all("voice bank rebuild")
    runtime.stop("installing a voice bank")
    bank.promote()
    try:
        sid = validate_sid if validate_sid is not None else _validation_sid()
        audio = runtime.synthesize("Voice check.", sid=int(sid))
        if not audio or len(audio) <= 44:
            raise RegistryError("The new voice bank produced no audio, so it was not kept.")
    except Exception as exc:
        logger.warning("Model Chain: the new Voice bank failed its check and was rolled back",
                       exc_info=True)
        runtime.stop("rolling back a voice bank")
        bank.rollback()
        if was_loaded:
            try:
                runtime.ensure_started()
            except Exception:
                logger.debug("Model Chain: could not restart Voice after a rollback",
                             exc_info=True)
        raise RegistryError(
            "The new voice bank did not pass its check, so nothing was changed. "
            + _short(exc)) from None
    if not was_loaded:
        # Restore the residency the user had. Loading a worker as a side effect
        # of managing a voice would be a Settings page that quietly allocates
        # four hundred megabytes.
        runtime.stop("voice bank installed")
    return built


def _validation_sid() -> int:
    try:
        return int(resolve()[0])
    except Exception:
        return 0


def _short(exc: BaseException) -> str:
    text = str(exc or "").strip()
    return text[:200] if text else exc.__class__.__name__
