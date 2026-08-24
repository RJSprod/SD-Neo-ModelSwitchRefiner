"""Named Creative Mode configurations, and the one the user opens with.

A profile is every Creative Mode decision except one: the Creativity position,
the Creative seed, anti-repetition, each axis's mode, its pinned value and its
exclusions. What it deliberately does not carry is whether
Creative Mode is *on*. A profile describes how the feature behaves when it runs;
switching it on is a decision somebody makes at the moment they press Generate,
and a preset that could flip it would be a preset that changes what the button
does.

Why a file of its own
---------------------
``mc_llm_state.preferences()`` holds the *current* Creative settings and should
go on holding them: it is one flat mapping of "what is this installation set to
right now", read on every panel build. Named configurations are a different
shape -- many of them, each complete, one of them nominated as the default -- and
growing a list of them inside a preferences file would make every save of a
slider position rewrite everybody's saved work.

So this follows ``mc_presets`` exactly: one JSON file under the WebUI's data
directory, so reinstalling or updating the extension does not throw profiles
away, written through a temporary file and an atomic replace so an interrupted
save leaves the previous file intact rather than a truncated one.

The Factory profile
-------------------
:data:`FACTORY` is not stored and cannot be deleted or overwritten. It is the
neutral configuration -- every axis Natural, nothing pinned, nothing excluded --
and it is built from the creativity package's own ``defaults.json`` rather than
written out here, so a package that ships different defaults ships a different
Factory without this file being edited.

It exists so that "put it back" is always answerable. A user whose default
profile has been deleted, corrupted, or written by a version of the extension
that is no longer installed still gets a panel that opens, on a configuration
that directs nothing, with a line saying what happened.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import mc_creative_krea

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

FILENAME = "krea_creative_profiles.json"

SCHEMA_VERSION = 1
"""Bumped when the *shape* below changes, never when a field is added.

Readers here fill in anything a stored profile does not carry, so a profile
written by an older version loads with the current defaults in the gaps. That is
what makes adding an axis, or adding exclusions to a schema that had none, a
change that costs nobody their saved work.
"""

FACTORY = "Factory"
"""The built-in neutral profile. Always present, never stored, never deletable."""

SPREAD = "Everything varies"
"""The built-in opposite: every axis on Vary, nothing pinned, nothing excluded.

Here because the neutral default took something away that people had, and one
click is the honest way to give it back. The creativity package shipped nine of
its ten axes on Vary, so anybody who used Creative Mode before the panel was
rebuilt had been running something close to this without choosing it -- and
after the rebuild the same installation directs nothing until a direction is
added, which is correct and is also a surprise.

It is not the default and should not be: this is a configuration to *choose*,
and choosing it is exactly what makes it different from the old behaviour.
"""

BUILT_IN = (FACTORY, SPREAD)
"""The profiles that are computed rather than stored. Never deletable."""

FIELDS = ("creativity", "seed", "anti_repetition", "axis_modes", "fixed_values",
          "excluded_values", "directions")
"""Every field a profile carries, by the name :func:`mc_creative_krea.settings`
uses for it, so a profile reads like the settings it restores.

``loras`` was one of them and is not any more. The Pinned LoRAs control is gone
-- ``[[<lora:name:weight>]]`` in the prompt replaced it -- and a profile written
by an older build simply has one key nobody reads: :func:`normalise` keeps the
fields it knows and drops the rest, so an existing profile still loads and still
restores everything it can."""

EXCLUDED_FIELDS = ("enabled",)
"""Settings a profile intentionally does not carry. See the module docstring."""


class ProfileError(RuntimeError):
    """A save, rename or delete that cannot be carried out.

    A distinct type because the panel turns it into one line on the page rather
    than a traceback: "that name is reserved", "there is no profile called that",
    "the file could not be written". None of those is a bug to report.
    """


def path() -> str:
    """Where profiles are stored."""
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

    A file that will not parse is treated as empty rather than raised out of, and
    said so in the log. The alternative is a Creative panel that refuses to build
    because of a stray byte in a settings file, which is a worse answer to the
    same problem: the user can see their profiles are gone and can say so.
    """
    file = path()
    if not os.path.exists(file):
        return {"profiles": {}, "default": ""}

    try:
        with open(file, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception:
        logger.warning("Model Chain: could not read Creative profiles from %s; "
                       "treating it as empty", file, exc_info=True)
        return {"profiles": {}, "default": ""}

    if not isinstance(document, dict):
        return {"profiles": {}, "default": ""}

    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    return {
        "profiles": {name: values for name, values in profiles.items()
                     if isinstance(values, dict)},
        "default": str(document.get("default") or ""),
    }


def _write(document: dict) -> None:
    file = path()
    payload = {"version": SCHEMA_VERSION,
               "profiles": document.get("profiles") or {},
               "default": str(document.get("default") or "")}

    directory = os.path.dirname(file) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        # Written beside the target so the replace stays on one filesystem.
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".krea_creative_profiles",
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
        raise ProfileError(f"Could not save Creative profiles to {file}: {exc}") from exc


# --------------------------------------------------------------------------- #
# What a profile is
# --------------------------------------------------------------------------- #


def factory() -> dict:
    """The neutral configuration, from the creativity package's own defaults.

    Read rather than written out, so the package decides what neutral means. A
    library that will not load still answers: every axis it knows about is
    Natural, and it knows about none, which is the same silence by a shorter
    route.
    """
    from prompt_master.krea import director, variation

    try:
        from prompt_master.krea import library as library_module

        lib = library_module.library()
        defaults = dict(lib.defaults)
        keys = lib.axis_keys
    except Exception:
        logger.debug("Model Chain: the creativity library could not be read for the "
                     "Factory profile", exc_info=True)
        defaults, keys = {}, ()

    modes = defaults.get("axis_modes") or {}
    return {
        "creativity": variation.clamp(defaults.get("creativity", variation.DEFAULT)),
        "seed": mc_creative_krea._seed(defaults.get("creative_seed", director.RANDOM_SEED)),
        "anti_repetition": bool(defaults.get("anti_repetition", True)),
        "axis_modes": {key: str(modes.get(key, director.NATURAL)).casefold()
                       for key in keys},
        "fixed_values": dict(defaults.get("fixed_values") or {}),
        "excluded_values": {key: list(values) for key, values
                            in (defaults.get("excluded_values") or {}).items()},
        # Whatever the package itself directs gets a row, and nothing else
        # does. Factory is the neutral profile, so on a stock library this is
        # empty -- which is the panel saying "no directions" rather than the
        # panel having forgotten some.
        "directions": [key for key in keys
                       if str(modes.get(key, director.NATURAL)).casefold()
                       in (director.VARY, director.FIXED)],
    }


def spread() -> dict:
    """Every axis varying, at the package's own Creativity position."""
    from prompt_master.krea import director

    values = factory()
    values["axis_modes"] = {key: director.VARY for key in values["axis_modes"]}
    # Every axis varying is every axis with a row: the rows are what "varies"
    # looks like on the panel, and a profile that set the modes without them
    # would restore a configuration the user could not see.
    values["directions"] = list(values["axis_modes"])
    return values


def built_in(name: str) -> dict | None:
    """One built-in profile by name, or ``None`` when the name is not one."""
    name = str(name or "")
    if name == FACTORY:
        return factory()
    if name == SPREAD:
        return spread()
    return None


def normalise(values) -> dict:
    """One profile's fields, cleaned against the current library.

    Applied on the way in and on the way out. A profile is a file a user may edit
    and a file a future version may have written, so an axis mode that is not a
    mode, a pinned id the package no longer has, or an exclusion list that is
    really a string all have to mean something sensible rather than reach the
    Director.
    """
    from prompt_master.krea import director, variation

    values = dict(values or {})
    modes = {}
    for key, mode in (values.get("axis_modes") or {}).items():
        folded = str(mode).casefold()
        modes[str(key)] = folded if folded in director.MODES else director.NATURAL

    return {
        "creativity": variation.clamp(values.get("creativity", variation.DEFAULT)),
        "seed": mc_creative_krea._seed(values.get("seed", director.RANDOM_SEED)),
        "anti_repetition": bool(values.get("anti_repetition", True)),
        "axis_modes": modes,
        "fixed_values": mc_creative_krea.known_fixed(values.get("fixed_values")),
        "excluded_values": mc_creative_krea.known_excluded(values.get("excluded_values")),
        # Which axes have a row, including the ones that have a row and no
        # treatments yet. A profile that carried the settings but not the rows
        # would load as a panel that had silently forgotten half the directions
        # somebody was in the middle of making.
        "directions": mc_creative_krea.known_directions(
            values.get("directions"), modes,
            mc_creative_krea._axes() or tuple(modes)),
    }


def from_settings(stored=None) -> dict:
    """The current Creative settings as a profile, minus the enabled flag."""
    stored = stored or mc_creative_krea.settings()
    return normalise({key: stored.get(key) for key in FIELDS})


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


def names() -> list[str]:
    """Saved profile names, case-insensitively sorted for a stable dropdown."""
    return sorted(_read()["profiles"], key=str.casefold)


def choices() -> list[str]:
    """Dropdown choices: the built-ins first, then the user's own."""
    return list(BUILT_IN) + names()


def exists(name: str) -> bool:
    return str(name or "") in BUILT_IN or str(name or "") in _read()["profiles"]


def get(name: str) -> dict | None:
    """One profile's values, or ``None`` when there is no such profile."""
    name = str(name or "")
    fixed = built_in(name)
    if fixed is not None:
        return fixed
    stored = _read()["profiles"].get(name)
    return None if stored is None else normalise(stored)


def save(name: str, values: dict) -> list[str]:
    """Create or overwrite a named profile. Returns the refreshed name list."""
    name = (name or "").strip()
    if not name:
        raise ProfileError("Give the profile a name before saving.")
    if name in BUILT_IN:
        raise ProfileError(f'"{name}" is built in and cannot be overwritten. Save '
                           "under another name.")

    document = _read()
    existed = name in document["profiles"]
    document["profiles"][name] = normalise(values)
    _write(document)

    logger.info("Model Chain: %s Creative profile %r", "updated" if existed else "saved",
                name)
    return names()


def delete(name: str) -> list[str]:
    """Remove a named profile. Returns the refreshed name list.

    The Factory profile is not deletable and neither is a name that is not there.
    If the profile being deleted was the chosen default, the default goes back to
    Factory in the same write -- a stored default naming a profile that no longer
    exists is exactly the corruption :func:`default_profile` has to recover from,
    and leaving one behind on purpose would be careless.
    """
    name = str(name or "")
    if not name or name in BUILT_IN:
        raise ProfileError(f'"{name or FACTORY}" is built in and cannot be deleted.')

    document = _read()
    if name not in document["profiles"]:
        raise ProfileError(f'No Creative profile named "{name}".')

    document["profiles"].pop(name)
    if document.get("default") == name:
        document["default"] = ""
    _write(document)

    logger.info("Model Chain: deleted Creative profile %r", name)
    return names()


def default_name() -> str:
    """The profile the panel opens on, as stored. May name nothing."""
    return _read().get("default") or FACTORY


def set_default(name: str) -> str:
    """Nominate the profile to open with. Returns the name that was stored."""
    name = str(name or "").strip() or FACTORY
    if not exists(name):
        raise ProfileError(f'No Creative profile named "{name}".')

    document = _read()
    document["default"] = "" if name == FACTORY else name
    _write(document)
    logger.info("Model Chain: Creative profile %r is now the default", name)
    return name


def selected() -> str:
    """The profile name the panel opens showing.

    The one the live settings were last loaded from, if it still exists; failing
    that the nominated default; failing that Factory. Nothing is *applied* by
    reading this -- a panel that reapplied a profile every time a tab opened
    would silently discard whatever was adjusted in the last one, which is the
    opposite of what a settings file is for.
    """
    remembered = _remembered_name()
    if remembered and exists(remembered):
        return remembered
    wanted = default_name()
    return wanted if exists(wanted) else FACTORY


def _remembered_name() -> str:
    import mc_llm_state

    try:
        return str(mc_llm_state.preferences().get(mc_creative_krea.PROFILE) or "")
    except Exception:
        return ""


def remember_selection(name: str) -> str:
    """Note which profile the live settings came from. Never fatal."""
    name = str(name or "")
    mc_creative_krea.remember(**{mc_creative_krea.PROFILE:
                                 "" if name == FACTORY else name})
    return name


def default_profile() -> tuple[str, dict, str]:
    """``(name, values, complaint)`` for the configured default.

    Never raises and never returns nothing. A default naming a profile that has
    been deleted, or a store that will not parse, falls back to Factory and says
    so in the third value -- because a panel that refused to build over a missing
    preset would be a Creative Mode nobody can turn on to fix it.
    """
    wanted = default_name()
    if wanted == FACTORY:
        return FACTORY, factory(), ""
    values = get(wanted)
    if values is None:
        return FACTORY, factory(), (f'The default Creative profile "{wanted}" is no '
                                    "longer there, so the neutral Factory profile was "
                                    "used instead.")
    return wanted, values, ""


def apply(name: str) -> tuple[dict, str]:
    """Write one profile into the live Creative settings.

    Returns ``(settings, complaint)``: the settings as
    :func:`mc_creative_krea.settings` would now answer them, and a sentence when
    the profile asked for could not be used. The enabled flag is untouched, which
    is the whole point of it not being in :data:`FIELDS`.
    """
    values = get(name)
    complaint = ""
    if values is None:
        name, values, complaint = FACTORY, factory(), (
            f'There is no Creative profile called "{name}", so nothing was changed '
            "except that the panel now shows Factory.")

    remember_selection(name)
    mc_creative_krea.remember(**{
        mc_creative_krea.CREATIVITY: values["creativity"],
        mc_creative_krea.SEED: values["seed"],
        mc_creative_krea.ANTI_REPETITION: values["anti_repetition"],
        mc_creative_krea.AXIS_MODES: values["axis_modes"],
        mc_creative_krea.FIXED_VALUES: values["fixed_values"],
        mc_creative_krea.EXCLUDED_VALUES: values["excluded_values"],
        mc_creative_krea.DIRECTIONS: values["directions"],
    })
    return mc_creative_krea.settings(), complaint
