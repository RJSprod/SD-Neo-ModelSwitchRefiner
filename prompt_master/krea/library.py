"""The creativity library: data in, typed vocabulary out. No UI, no inference.

``creativity/`` is a vendored data package -- ten axis files, 164 variant
families, an activation and sampling policy, compatibility rules and an
anti-repetition policy -- and this module is the only thing that reads it. Its
whole job is to turn that JSON into objects the Director can select from, and to
refuse to hand over a package that has stopped describing what it promises.

Why the validation is strict
----------------------------
Every failure this module can catch is a failure that would otherwise be
invisible. An entry missing its ``extreme`` expression is a Creativity slider
that quietly stops scaling on one axis. A ``min_creativity`` of 11 is a variant
nobody can ever draw. A duplicate id is a saved Fixed selection that means two
different things depending on file order. None of those raise anywhere else;
they just make the feature slightly wrong forever. So they raise here, once, at
load, naming the file and the entry.

The one thing that is deliberately *not* strict is the set of axes. The manifest
lists them and the loader reads what the manifest lists, so adding an axis to a
later package is a data edit rather than a code edit -- which is the whole point
of the package being versioned separately.

What this module refuses to do
------------------------------
It does not choose anything. Selection, seeds, tiers and briefs are the
Director's, and keeping the two apart is what makes the Director testable
against a fake library and this testable against the real one. It also holds no
UI text beyond :data:`AXIS_LABELS`, which is not UI: those strings are the
labels that go *into the Creative Direction brief*, so they are part of what the
model reads and belong beside the data they name.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

DIRECTORY = Path(__file__).with_name("creativity")
"""Where the vendored package lives. See CREATIVITY_LIBRARY_SOURCE.txt."""

TIERS = ("light", "moderate", "strong", "extreme")
"""The four expression strengths, weakest first.

Order is load-bearing: :func:`tier_at_or_below` walks it backwards to find the
strongest tier an entry can express at a given position, and a test asserts the
sequence rather than trusting the tuple literal to stay sorted.
"""

AXIS_LABELS = {
    "medium": "Medium",
    "style": "Style",
    "lighting": "Lighting",
    "composition": "Composition",
    "viewpoint": "Viewpoint",
    "lens_zoom": "Lens / Zoom",
    "palette": "Palette",
    "texture": "Texture",
    "mood": "Mood",
    "detail_emphasis": "Detail emphasis",
}
"""What each axis is called in the Creative Direction brief.

Brief text, not interface text -- these words are read by the language model.
An axis the package adds without a label here is labelled from its own key
(``lens_zoom`` -> ``Lens zoom``), so a new axis works without this dictionary
being edited; it just reads slightly less well until somebody adds a line.
"""

_AXIS_KEY_ALIASES = {"lens": "lens_zoom"}
"""Names ``compatibility.json`` uses that are not the axis's own key.

One entry today: the rules say ``avoid_lens_tags`` about the ``lens_zoom`` axis.
Resolved here rather than in the Director so that the mismatch is a fact about
the data format, recorded next to the data, rather than a special case buried in
selection code.
"""


class LibraryError(RuntimeError):
    """The data package is missing, unreadable, or does not describe itself.

    A distinct type because the callers do different things with it: the panels
    turn it into a sentence on the page, and the Director lets it out to say
    that no creative direction is possible, which is a very different thing
    from "the model said nothing".
    """


# --------------------------------------------------------------------------- #
# The vocabulary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Variant:
    """One variant family: an id, what it is, and how hard to say it.

    Frozen, and shared between every roll in the process. A Director that could
    edit a variant in place would be a Director whose second roll draws from a
    library its first one rewrote.
    """

    axis: str
    identifier: str
    label: str
    family: str
    aliases: tuple[str, ...]
    min_creativity: int
    weight: float
    tags: frozenset[str]
    expressions: MappingProxyType

    def expression(self, tier: str) -> str:
        """This variant said at ``tier`` strength.

        Raises rather than falling back to a neighbouring tier. A missing tier
        is caught at load, so reaching here with an unknown one means a caller
        invented a tier name, and quietly handing back the ``light`` text would
        turn that bug into prompts that are mysteriously mild.
        """
        try:
            return self.expressions[tier]
        except KeyError:
            raise LibraryError(
                f"{self.axis}/{self.identifier} has no {tier!r} expression") from None

    def eligible_at(self, creativity: int) -> bool:
        return int(creativity) >= self.min_creativity


@dataclass(frozen=True)
class Axis:
    """One axis of art direction, and everything that can be said along it."""

    key: str
    label: str
    variants: tuple[Variant, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id",
                           {variant.identifier: variant for variant in self.variants})

    def variant(self, identifier: str) -> Variant | None:
        """One variant by its stable id, or ``None`` if the package no longer has it.

        ``None`` and not an exception: a saved Fixed selection outlives the
        library version it was made against, and a user whose pinned variant was
        renamed in a package update should get their axis quietly falling back to
        Vary with a note, not a panel that will not build.
        """
        return getattr(self, "_by_id").get(str(identifier or ""))

    def eligible(self, creativity: int) -> tuple[Variant, ...]:
        return tuple(v for v in self.variants if v.eligible_at(creativity))

    @property
    def families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for variant in self.variants:
            if variant.family not in seen:
                seen.append(variant.family)
        return tuple(seen)


@dataclass(frozen=True)
class Rule:
    """One compatibility rule: what else makes sense once ``tag`` is in play.

    The JSON keys are ``prefer_<axis>_families``, ``avoid_<axis>_families`` and
    ``avoid_<axis>_tags``, and they are parsed rather than enumerated so that a
    package adding ``prefer_palette_tags`` needs no code change here.
    """

    tag: str
    prefer: MappingProxyType
    avoid: MappingProxyType

    @staticmethod
    def _split(key: str):
        match = re.fullmatch(r"(prefer|avoid)_(.+)_(families|tags)", key)
        if match is None:
            return None
        direction, axis, kind = match.groups()
        return direction, _AXIS_KEY_ALIASES.get(axis, axis), kind


@dataclass(frozen=True)
class Policy:
    """The activation, tier and sampling policy, as read from the package."""

    activation: MappingProxyType
    sampling: MappingProxyType
    precedence: tuple[str, ...]

    def tier(self, creativity: int) -> str:
        """The expression tier at this position, or ``"none"`` below 2."""
        return str(self.activation[int(creativity)].get("tier", "none"))

    def activation_range(self, creativity: int, eligible: int) -> tuple[int, int]:
        """How many Vary axes may activate, as ``(minimum, maximum)``.

        ``all_eligible`` at Creativity 10 is resolved against the number of axes
        actually available rather than against the ten the package ships, so a
        user who has set six axes to Natural still gets "all of them" and not a
        range that cannot be satisfied.
        """
        row = self.activation[int(creativity)]
        if row.get("all_eligible"):
            return eligible, eligible
        low = min(int(row.get("min", 0)), eligible)
        high = min(int(row.get("max", 0)), eligible)
        return low, max(low, high)


@dataclass(frozen=True)
class AntiRepetition:
    """How hard to push away from what the last few rolls already used."""

    history_length: int
    strength: MappingProxyType
    generic_treatments: tuple[str, ...]

    def strength_at(self, creativity: int) -> float:
        return float(self.strength.get(int(creativity), 0.0))


@dataclass(frozen=True)
class Library:
    """One loaded creativity package."""

    version: str
    schema_version: int
    axis_keys: tuple[str, ...]
    axes: MappingProxyType
    rules: tuple[Rule, ...]
    policy: Policy
    anti_repetition: AntiRepetition
    defaults: MappingProxyType
    priority: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def axis(self, key: str) -> Axis | None:
        return self.axes.get(str(key or ""))

    @property
    def variant_count(self) -> int:
        return sum(len(axis.variants) for axis in self.axes.values())

    def aliases(self) -> dict[str, tuple[str, ...]]:
        """Every alias in the package, mapped to the axes it constrains.

        An alias can name more than one axis and that is not a defect: "direct
        flash" is a lighting entry *and* a medium entry, and somebody who typed
        it has constrained both. The Director locks every axis an alias names.
        """
        found: dict[str, list[str]] = {}
        for axis in self.axes.values():
            for variant in axis.variants:
                for alias in variant.aliases:
                    keys = found.setdefault(alias, [])
                    if axis.key not in keys:
                        keys.append(axis.key)
        return {alias: tuple(keys) for alias, keys in found.items()}

    def priority_of(self, axis_key: str) -> float:
        """How readily an axis activates when only a few slots are available.

        Not in the package, and defaulted here rather than left flat because a
        Creativity-2 roll that activates one axis should usually activate the
        one that changes the picture most. Medium first, then style and light;
        a lens note on its own is a nudge nobody would notice.

        A package that grows an ``axis_priority`` map in its manifest overrides
        this without a code change, which is where this belongs long-term.
        """
        if axis_key in self.priority:
            return float(self.priority[axis_key])
        return float(_DEFAULT_PRIORITY.get(axis_key, 1.0))


_DEFAULT_PRIORITY = {
    "medium": 5.0,
    "style": 4.0,
    "lighting": 4.0,
    "palette": 3.0,
    "composition": 3.0,
    "mood": 3.0,
    "viewpoint": 2.5,
    "texture": 2.0,
    "lens_zoom": 2.0,
    "detail_emphasis": 2.0,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LibraryError(f"the creativity library is missing {path.name}") from None
    except (OSError, ValueError) as exc:
        raise LibraryError(f"the creativity library's {path.name} could not be read: "
                           f"{exc}") from None


def _variant(axis_key: str, row, seen: set) -> Variant:
    if not isinstance(row, dict):
        raise LibraryError(f"{axis_key}.json contains an entry that is not an object")
    identifier = str(row.get("id") or "").strip()
    if not identifier:
        raise LibraryError(f"{axis_key}.json contains an entry with no id")
    if identifier in seen:
        raise LibraryError(f"{axis_key}.json defines {identifier!r} twice; ids are "
                           "what saved Fixed selections are stored as and must be unique")
    seen.add(identifier)

    expressions = row.get("expressions")
    if not isinstance(expressions, dict):
        raise LibraryError(f"{axis_key}/{identifier} has no expressions")
    missing = [tier for tier in TIERS if not str(expressions.get(tier) or "").strip()]
    if missing:
        # Refused rather than filled in from a neighbour: a substituted tier is
        # a Creativity slider that stops scaling on one axis and says nothing.
        raise LibraryError(f"{axis_key}/{identifier} is missing its "
                           f"{', '.join(missing)} expression")

    try:
        minimum = int(row.get("min_creativity", 2))
        weight = float(row.get("weight", 1.0))
    except (TypeError, ValueError):
        raise LibraryError(f"{axis_key}/{identifier} has a non-numeric "
                           "min_creativity or weight") from None
    if not 0 <= minimum <= 10:
        raise LibraryError(f"{axis_key}/{identifier} has min_creativity {minimum}, "
                           "which no position on the slider can reach")
    if weight <= 0:
        raise LibraryError(f"{axis_key}/{identifier} has weight {weight}; a variant "
                           "that can never be drawn should be deleted, not weighted out")

    return Variant(
        axis=axis_key,
        identifier=identifier,
        label=str(row.get("label") or identifier),
        family=str(row.get("family") or ""),
        aliases=tuple(str(alias).strip().casefold() for alias in row.get("aliases") or ()
                      if str(alias).strip()),
        min_creativity=minimum,
        weight=weight,
        tags=frozenset(str(tag) for tag in row.get("tags") or ()),
        expressions=MappingProxyType({tier: str(expressions[tier]).strip()
                                      for tier in TIERS}),
    )


def _axis(directory: Path, key: str) -> Axis:
    document = _read(directory / "axes" / f"{key}.json")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LibraryError(f"{key}.json lists no entries")
    seen: set = set()
    variants = tuple(_variant(key, row, seen) for row in entries)
    return Axis(key=key, label=AXIS_LABELS.get(key, key.replace("_", " ").capitalize()),
                variants=variants)


def _rules(document) -> tuple[Rule, ...]:
    rules = []
    for row in document.get("rules") or ():
        if not isinstance(row, dict):
            continue
        tag = str(row.get("tag") or "").strip()
        if not tag:
            continue
        prefer: dict = {}
        avoid: dict = {}
        for key, value in row.items():
            parsed = Rule._split(key)
            if parsed is None:
                continue
            direction, axis, kind = parsed
            target = prefer if direction == "prefer" else avoid
            target.setdefault(axis, {})[kind] = frozenset(str(item) for item in value or ())
        rules.append(Rule(tag=tag,
                          prefer=MappingProxyType({k: MappingProxyType(v)
                                                   for k, v in prefer.items()}),
                          avoid=MappingProxyType({k: MappingProxyType(v)
                                                  for k, v in avoid.items()})))
    return tuple(rules)


def _policy(document) -> Policy:
    activation = document.get("axis_activation") or {}
    sampling = document.get("sampling") or {}
    rows: dict = {}
    samples: dict = {}
    for position in range(0, 11):
        key = str(position)
        if key not in activation:
            raise LibraryError(f"creativity_policy.json has no activation row for {position}")
        if key not in sampling:
            raise LibraryError(f"creativity_policy.json has no sampling row for {position}")
        rows[position] = MappingProxyType(dict(activation[key]))
        samples[position] = MappingProxyType(dict(sampling[key]))
    return Policy(activation=MappingProxyType(rows), sampling=MappingProxyType(samples),
                  precedence=tuple(str(step) for step in document.get("precedence") or ()))


def _anti_repetition(document) -> AntiRepetition:
    strength = {}
    for position in range(0, 11):
        try:
            strength[position] = float((document.get("strength_by_creativity") or {})
                                       .get(str(position), 0.0))
        except (TypeError, ValueError):
            strength[position] = 0.0
    return AntiRepetition(
        history_length=max(int(document.get("history_length") or 8), 1),
        strength=MappingProxyType(strength),
        generic_treatments=tuple(
            str(item) for item in
            document.get("generic_treatments_to_discourage_when_not_explicit") or ()))


def load(directory: Path = None) -> Library:
    """Read and validate one creativity package. Raises :class:`LibraryError`."""
    directory = Path(directory) if directory is not None else DIRECTORY
    manifest = _read(directory / "library_manifest.json")

    axis_keys = tuple(str(key) for key in manifest.get("axes") or ())
    if not axis_keys:
        raise LibraryError("library_manifest.json lists no axes")

    axes = {key: _axis(directory, key) for key in axis_keys}
    policy = _policy(_read(directory / "creativity_policy.json"))
    anti = _anti_repetition(_read(directory / "anti_repetition.json"))
    defaults = _read(directory / "defaults.json")
    rules = _rules(_read(directory / "compatibility.json"))

    priority = manifest.get("axis_priority")
    return Library(
        version=str(manifest.get("library_version") or "unknown"),
        schema_version=int(manifest.get("schema_version") or 0),
        axis_keys=axis_keys,
        axes=MappingProxyType(axes),
        rules=rules,
        policy=policy,
        anti_repetition=anti,
        defaults=MappingProxyType(dict(defaults)),
        priority=MappingProxyType(dict(priority) if isinstance(priority, dict) else {}),
    )


# --------------------------------------------------------------------------- #
# The one that is actually used
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_loaded: Library | None = None


def library() -> Library:
    """The vendored package, read once per process.

    Cached because it is a fifth of a megabyte of JSON and it cannot change
    under a running WebUI -- it is vendored, not user data. The lock is not
    paranoia: a Gradio handler thread and a Forge generation thread can both be
    the first to ask, and parsing it twice would be a waste rather than a bug,
    but handing back two different :class:`Library` objects would make identity
    comparisons in the Director quietly meaningless.
    """
    global _loaded
    with _lock:
        if _loaded is None:
            _loaded = load()
        return _loaded


def reload() -> Library:
    """Forget the cached package and read it again. For tests and for editing data."""
    global _loaded
    with _lock:
        _loaded = None
    return library()


# --------------------------------------------------------------------------- #
# Helpers the Director and the panels both want
# --------------------------------------------------------------------------- #


def tier_at_or_below(tier: str) -> str:
    """``tier`` if it names one of :data:`TIERS`, else the weakest.

    Used where a caller may hand over ``"none"`` -- Creativity 0 and 1 have no
    tier at all, and a Fixed axis still has to say something there. The weakest
    expression is the honest answer: the user asked for that value explicitly,
    and the position they are on says "as little amplification as possible".
    """
    return tier if tier in TIERS else TIERS[0]


def alias_pattern(alias: str) -> re.Pattern:
    """``alias`` as a whole-word search, tolerant of the punctuation in the data.

    Word boundaries by hand rather than ``\\b`` because the aliases include
    ``24mm``, ``children's book`` and ``ink-and-wash``: ``\\b`` around a string
    that starts or ends with punctuation matches in places nobody means. The
    lookarounds below only refuse a match that runs into more word characters,
    which is what stops ``riso`` matching ``risotto`` and lets ``top-down``
    match inside ``a top-down shot``.
    """
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
