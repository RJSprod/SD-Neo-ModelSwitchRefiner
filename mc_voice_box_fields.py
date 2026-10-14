"""An engine's configuration fields, as data, and the one way they are checked.

Every Voice Box engine describes the settings of its own render -- VibeVoice's
diffusion steps and solver, Breeze's line gap -- as a list of fields
(``adapter.FIELDS``). The page draws the Configuration stage from that list
(docs/25-voice-box-engines.md, section 3), and this module validates a
configuration's values against it, so that a third engine is a third list and
not a hunt through the render service.

A field is a dict:

    key        the option's name in ``configuration.options[<engine>]``
    label      its caption, and the subject of every sentence about it
    kind       ``int``, ``float``, ``bool`` or ``choice``
    min, max   an int's or a float's bounds; ``step`` the page's step
    blank      an int that may be left blank (``None``); the placeholder
    default    a new configuration's value
    options    a choice's ``[{id, name, label}]``, in listed order
    noun       a choice's noun with its article, for "… is not a solver X has"
    needs      the bool field this one means nothing without
    place      ``head`` (drawn before Seed and Batch) or ``tail`` (after)
    wide       a bool drawn across its row
    hint       a tooltip
    infotext   the label an output's infotext gives it

The sentences are the Voice Box's own, unchanged from when they were written
for VibeVoice alone: "Diffusion steps must be between 1 and 50.", "CFG must be
a number.", "euler is not a solver VibeVoice has.". Standard library only.
"""

from __future__ import annotations

import copy

KINDS = ("int", "float", "bool", "choice")
PLACES = ("head", "tail")


class FieldError(ValueError):
    """A value a field does not take, in the sentence the page shows."""


def defaults(fields) -> dict:
    """``{key: default}`` for every field."""
    return {field["key"]: copy.deepcopy(field.get("default")) for field in fields}


def by_key(fields) -> dict:
    return {field["key"]: field for field in fields}


def _number(value, low, high, label: str, whole: bool):
    if whole:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise FieldError(f"{label} must be a whole number.") from None
        if not low <= number <= high:
            raise FieldError(f"{label} must be between {low} and {high}.")
        return number
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise FieldError(f"{label} must be a number.") from None
    if not low <= number <= high:
        raise FieldError(f"{label} must be between {low:g} and {high:g}.")
    return round(number, 3)


def _held(value, field: dict):
    """A number a switched-off field keeps: brought into range, never refused.

    Off, the page greys the field out, so a value typed out of range before
    the switch went off could not be corrected there, and refusing it would
    refuse every Save and Render until the switch was turned on again to fix a
    number that is not being used. A value that is not a number at all goes
    back to the default.
    """
    fallback = field.get("default")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:
        return fallback
    number = min(max(number, field["min"]), field["max"])
    return int(round(number)) if field["kind"] == "int" else round(number, 3)


def clean(fields, values: dict, previous: "dict | None" = None, engine: str = "") -> dict:
    """``values`` checked field by field, with ``previous`` (or the defaults) for the rest.

    Keys that are no field's are dropped. A bool is on only for a real
    ``True``, as the workers read it. A field whose ``needs`` switch is off is
    held in range rather than refused (:func:`_held`). Raises
    :class:`FieldError` with a sentence for anything else that does not fit.
    """
    table = list(fields)
    merged = defaults(table)
    for key, value in dict(previous or {}).items():
        if key in merged:
            merged[key] = value
    for key, value in dict(values or {}).items():
        if key in merged:
            merged[key] = value
    found = {}
    switches = {field["key"]: merged.get(field["key"]) is True
                for field in table if field["kind"] == "bool"}
    for field in table:
        key, kind, label = field["key"], field["kind"], field.get("label") or field["key"]
        value = merged.get(key)
        if kind == "bool":
            found[key] = value is True
            continue
        if kind == "choice":
            text = str(value or "").strip()
            ids = [str(option.get("id")) for option in field.get("options") or ()]
            if text not in ids:
                noun = field.get("noun") or "a choice"
                raise FieldError(f"{text or 'Nothing'} is not {noun} {engine or 'this engine'} has.")
            found[key] = text
            continue
        if field.get("blank") is not None and field.get("blank") != "" \
                and value in (None, ""):
            found[key] = None
            continue
        needs = field.get("needs")
        if needs and not switches.get(needs, False):
            found[key] = _held(value, field)
            continue
        found[key] = _number(value, field["min"], field["max"], label, kind == "int")
    return found


def effective(fields, values: dict) -> dict:
    """The values as a render uses them: a field whose switch is off is ``None``.

    What a worker is sent and what an output records. The configuration keeps
    the field's own value for when the switch is on again.
    """
    found = dict(values or {})
    for field in fields:
        needs = field.get("needs")
        if needs and found.get(needs) is not True:
            found[field["key"]] = None
    return found


def check(fields) -> list:
    """What is wrong with a field list, as sentences. For the engines' own tests."""
    problems = []
    seen = set()
    for field in fields:
        key = field.get("key")
        if not key or key in seen:
            problems.append(f"{key!r} is missing or listed twice")
        seen.add(key)
        if field.get("kind") not in KINDS:
            problems.append(f"{key}: kind {field.get('kind')!r}")
        if field.get("place") not in PLACES:
            problems.append(f"{key}: place {field.get('place')!r}")
        if field.get("kind") in ("int", "float"):
            if not (field.get("min") is not None and field.get("max") is not None
                    and field["min"] <= field["max"]):
                problems.append(f"{key}: bounds")
            default = field.get("default")
            if default is not None and not field["min"] <= default <= field["max"]:
                problems.append(f"{key}: default outside its bounds")
            if default is None and not field.get("blank"):
                problems.append(f"{key}: no default and not blankable")
        if field.get("kind") == "choice":
            ids = [option.get("id") for option in field.get("options") or ()]
            if field.get("default") not in ids:
                problems.append(f"{key}: default is not one of its options")
        needs = field.get("needs")
        if needs and needs not in {other.get("key") for other in fields
                                   if other.get("kind") == "bool"}:
            problems.append(f"{key}: needs {needs!r}, which is no switch")
    return problems
