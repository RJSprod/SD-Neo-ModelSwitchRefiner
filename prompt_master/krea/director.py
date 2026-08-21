"""The local Creative Director: art direction chosen in Python, never by a model.

One roll in, one recipe out:

    source prompt + Creativity + Creative seed + axis settings + recent history
        -> one CreativeRecipe
        -> one Creative Direction brief
        -> one derived LLM seed

and then exactly one Krea writer call, made by somebody else.

**Nothing in this module performs inference or touches a network.** That is the
whole reason it exists. The obvious way to make a prompt writer more varied is
to ask a model what to vary, which costs a second completion per roll, a second
thing to wait for, and a second thing that can be wrong. Choosing the art
direction from a vocabulary with a seeded PRNG costs microseconds, reproduces
exactly, and can be shown to a user as a list they can argue with.

The three modes
---------------
**Natural** contributes nothing. Not a hedged line, not a "your choice" line --
the axis is simply absent from the brief, and the model decides as it would have
without Creative Mode at all. A brief that says "Texture: whatever you think" is
not the same as silence; it puts texture in the model's foreground.

**Vary** lets this module choose. The master Creativity position decides four
separate things about that: whether the axis activates, which variants are
eligible, how strongly the chosen one is expressed, and how hard recent choices
are pushed away. All four scale, which is why Creativity 2 with one Vary axis
still differs from Creativity 10 with the same one axis.

Vary takes one modifier and it is not a fourth mode: **excluded ids**. "Vary the
lighting, but never harsh noon" is a statement about *how* to vary, and making it
a mode would force a user who wants two treatments gone to give up varying
altogether. The excluded ids come out of the pool before anything is weighed, and
if that empties the pool the axis is skipped with a note rather than being handed
back the value it was told not to use.

**Fixed** repeats a chosen variant every roll. It sits *below* the source prompt
and *above* Vary: a user who pinned a texture gets it until they say otherwise,
unless their own words that roll say something else.

Source text beats all three. :func:`explicit_locks` catches what the library's
aliases can catch, and the brief always carries the rule in words for everything
they cannot -- which is most of English. No second model call is needed to
enforce it because the model doing the expanding is the one being told.

Determinism
-----------
Everything random here comes from one seeded ``random.Random``. Given the same
seed, source, Creativity, axis settings, history and library version, the recipe
is the same recipe, on any machine, in any process. The seeds derive through
SHA-256 rather than :func:`hash`, which is salted per process and would make
"reproducible" mean "until you restart the WebUI".
"""

from __future__ import annotations

import hashlib
import random
import secrets
from dataclasses import dataclass, field
from typing import Literal

from . import library as library_module

RANDOM_SEED = -1
"""What "draw me a new one" looks like in the seed box, as everywhere else here."""

SEED_LIMIT = 2 ** 32
"""Seeds are unsigned 32-bit, which is what llama.cpp and Forge both take."""

NATURAL = "natural"
VARY = "vary"
FIXED = "fixed"
MODES = (NATURAL, VARY, FIXED)

REPLAY = "replay"
"""Where a replayed line came from. Not a mode -- no axis can be set to it.

:attr:`CreativeRecipeItem.source` says which of the user's decisions produced a
line, and a replayed recipe's lines were produced by none of them: they came out
of a record. Labelling them ``vary`` would make the diagnostics view claim the
Director chose something it did not choose.
"""

AxisMode = Literal["natural", "vary", "fixed"]

SOURCE_PRIORITY_RULE = (
    "Preserve every explicit constraint in the user's source prompt. If any "
    "Creative Direction below conflicts with the source prompt, the source prompt wins.")
"""The line every brief carries, and the reason one call is enough.

Alias matching finds "oil painting" and "top-down". It will never find "shot the
way Saul Leiter would have", and no list of aliases ever will. So the brief does
not rely on having found everything: it tells the model, in the same breath as
the direction, which of the two to drop when they disagree. The model is already
reading the source prompt; this costs one sentence and no second request.
"""

BRIEF_HEADING = "creative_direction:"
"""Labelled the way this package labels every other block in a user turn.

``user_prompt:`` and ``reference_images:`` are ``enhancer``'s existing
convention, and a third block shouting in capitals would read as though it came
from somewhere else. The design intent's sample uses upper case; the labels are
this extension's to choose and consistency wins.
"""


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


def stable_hash(seed: int, purpose: str) -> int:
    """A 32-bit sub-seed from ``seed`` and ``purpose``, identical everywhere.

    SHA-256 and not :func:`hash`: Python salts string hashing per process, so a
    Creative seed that reproduced a recipe this morning would reproduce a
    different one after a restart, and "fixed seed reproduces" would be true
    only within a session. This is slower by an amount nobody can measure once
    per roll.
    """
    material = f"{int(seed)}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def resolve_seed(creative_seed) -> int:
    """The concrete Creative seed for this roll.

    ``-1`` draws a fresh one. Every other value is used as given, so that a seed
    written down in a metadata field can be typed back in. Drawn with
    :mod:`secrets` rather than :mod:`random` so that the process-wide PRNG's
    state -- which a caller may have seeded for something else entirely -- has no
    bearing on which recipes a user sees.
    """
    try:
        value = int(creative_seed)
    except (TypeError, ValueError):
        value = RANDOM_SEED
    if value == RANDOM_SEED:
        return secrets.randbelow(SEED_LIMIT)
    return value % SEED_LIMIT


def derive(creative_seed: int) -> tuple[int, int]:
    """``(director_seed, llm_seed)`` for one resolved Creative seed."""
    return stable_hash(creative_seed, "director"), stable_hash(creative_seed, "llm")


# --------------------------------------------------------------------------- #
# What a roll is made of
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AxisSetting:
    """What the user asked for on one axis.

    Natural by default, and that default is load-bearing. A missing axis -- one
    the package added after a user's settings were written, one a profile from
    an older schema does not mention -- has to fail *neutral*: contributing
    nothing to the brief is a thing the user can see the absence of, whereas
    silently varying an axis nobody configured is art direction arriving from
    nowhere.

    ``excluded_ids`` modifies Vary and only Vary. It is the answer to "vary this,
    but never that": the Director removes those ids from the eligible pool before
    it weighs anything. Fixed ignores it, because a pin *is* the decision and an
    exclusion that could cancel one would leave the axis meaning two things at
    once.
    """

    mode: AxisMode = NATURAL
    fixed_id: str | None = None
    excluded_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        # Coerced rather than trusted: these arrive from a JSON file, a Gradio
        # multiselect and an infotext line, and one of the three hands over a
        # list every time. A frozenset of str is what selection wants to ask
        # ``in`` of, and building it here is what stops each caller doing it.
        object.__setattr__(self, "excluded_ids", frozenset(
            str(identifier) for identifier in self.excluded_ids or () if str(identifier)))

    @property
    def valid(self) -> bool:
        return self.mode in MODES

    def excludes(self, identifier: str) -> bool:
        """Whether Vary on this axis is forbidden from choosing ``identifier``."""
        return str(identifier) in self.excluded_ids


@dataclass(frozen=True)
class CreativeRecipeItem:
    """One line of art direction, and where it came from.

    ``source`` is ``"vary"``, ``"fixed"`` or ``"replay"`` and is not decoration:
    it is what lets the diagnostic view show a user which of their pins held,
    which lines the Director chose, and which came back out of a recorded recipe
    rather than being chosen at all.
    """

    axis: str
    label: str
    variant_id: str
    variant_label: str
    expression: str
    strength: str
    source: str

    @property
    def line(self) -> str:
        """This item as it appears in the brief."""
        return f"{self.label}: {self.expression}."


@dataclass(frozen=True)
class CreativeRecipe:
    """One roll's complete art direction, plus everything needed to repeat it."""

    creative_seed: int
    llm_seed: int
    creativity: int
    items: tuple[CreativeRecipeItem, ...] = ()
    avoid: tuple[str, ...] = ()
    locked: tuple[str, ...] = ()
    library_version: str = ""
    brief: str = ""
    notes: tuple[str, ...] = ()
    """Anything the roll decided *not* to do, in words, for a person to read.

    Exclusions are the reason this exists. An axis whose whole eligible pool has
    been excluded cannot be directed, and the two wrong answers are both silent:
    choosing an excluded treatment anyway, or dropping the axis with nothing said
    and letting somebody conclude that exclusions break Vary. The axis is skipped
    and the skip is written down, here, where the diagnostics view and the log
    both read it.
    """
    replayed: bool = False
    """Whether these items were replayed from a record rather than chosen.

    A replayed recipe is not a roll and must never be described as one: the
    Director drew nothing, the history weighted nothing, and the seed on it is
    the seed the original roll resolved. See :func:`replay`.
    """

    def __bool__(self) -> bool:
        """Whether this roll has anything to say.

        Creativity 0 and 1 with no Fixed axes produce a recipe with no items,
        and the truth of that is load-bearing further up: an empty recipe adds
        no block to the user turn at all, which is what keeps Creativity 1
        byte-identical to the request the writer made before Creative Mode
        existed.
        """
        return bool(self.items)

    @property
    def variant_ids(self) -> tuple[str, ...]:
        """The stable ids used, for the history and for compact metadata."""
        return tuple(item.variant_id for item in self.items)

    @property
    def compact(self) -> str:
        """The recipe as one short line: ``medium=oil_impasto, mood=monumental``."""
        return ", ".join(f"{item.axis}={item.variant_id}" for item in self.items)


# --------------------------------------------------------------------------- #
# What the source prompt already settled
# --------------------------------------------------------------------------- #


def explicit_locks(source: str, lib=None) -> frozenset[str]:
    """Axes the user's own words have already decided.

    Alias matching, and deliberately nothing cleverer. It catches the phrases
    people actually type -- "oil painting", "top-down", "macro", "anime" -- and
    it is wrong in exactly one safe direction: an axis it fails to lock is one
    the model is still told to keep, by the rule at the top of every brief. An
    axis it locks in error costs one line of direction the user did not get.

    An alias that names two axes locks both. "direct flash photo of a car" has
    settled the medium *and* the light, and pretending otherwise would have the
    Director cheerfully directing golden-hour lighting over it.
    """
    lib = lib or library_module.library()
    text = str(source or "")
    if not text.strip():
        return frozenset()

    locked: set[str] = set()
    for alias, axes in lib.aliases().items():
        if alias in locked:
            continue
        if library_module.alias_pattern(alias).search(text):
            locked.update(axes)
    return frozenset(locked)


# --------------------------------------------------------------------------- #
# Choosing
# --------------------------------------------------------------------------- #


def _weighted_choice(rng: random.Random, candidates, weights):
    """One item, in proportion to its weight. Total zero means nothing is eligible."""
    total = sum(weights)
    if total <= 0:
        return None
    cut = rng.random() * total
    running = 0.0
    for candidate, weight in zip(candidates, weights):
        running += weight
        if cut < running:
            return candidate
    return candidates[-1]


def _rules_for(lib, tags) -> list:
    return [rule for rule in lib.rules if rule.tag in tags]


def _violates(constraint, variant) -> bool:
    """Whether ``variant`` falls foul of one axis constraint from a rule."""
    families = constraint.get("families")
    if families and variant.family in families:
        return True
    tags = constraint.get("tags")
    return bool(tags and (variant.tags & tags))


def _satisfies(constraint, variant) -> bool:
    """Whether ``variant`` matches a rule's preference for its axis."""
    families = constraint.get("families")
    if families and variant.family in families:
        return True
    tags = constraint.get("tags")
    return bool(tags and (variant.tags & tags))


def _compatible(lib, variant, chosen) -> bool:
    """Whether ``variant`` can sit beside what has already been chosen.

    Rules point both ways and both are checked. A rule attached to the candidate
    constrains what may already be in the recipe -- impasto texture refuses a
    photographic medium. A rule attached to something already chosen constrains
    the candidate -- a fisheye lens already picked refuses a telephoto-tagged
    partner. Checking only one direction would make coherence depend on the
    order the axes happened to be visited in.
    """
    for rule in _rules_for(lib, variant.tags):
        for axis_key, constraint in rule.avoid.items():
            existing = chosen.get(axis_key)
            if existing is not None and _violates(constraint, existing):
                return False
    for existing in chosen.values():
        for rule in _rules_for(lib, existing.tags):
            constraint = rule.avoid.get(variant.axis)
            if constraint is not None and _violates(constraint, variant):
                return False
    return True


def _preference_boost(lib, variant, chosen) -> float:
    """How much a rule likes ``variant`` beside what is already chosen.

    A boost and not a requirement, because the axes are visited in one order and
    a hard preference would empty the pool whenever the preferred partner had
    not been picked yet. Choosing impasto texture after a photographic medium is
    incoherent and is refused above; choosing a painting medium *because* the
    texture already leans that way is merely nice, and nice belongs in a weight.
    """
    boost = 1.0
    for existing in chosen.values():
        for rule in _rules_for(lib, existing.tags):
            constraint = rule.prefer.get(variant.axis)
            if constraint is not None and _satisfies(constraint, variant):
                boost *= 2.0
    for rule in _rules_for(lib, variant.tags):
        for axis_key, constraint in rule.prefer.items():
            existing = chosen.get(axis_key)
            if existing is not None and _satisfies(constraint, existing):
                boost *= 2.0
    return boost


def _choose_variant(lib, axis, creativity, rng, chosen, recent, penalty, excluded=()):
    """One variant for ``axis`` as ``(variant, note)``; ``variant`` may be ``None``.

    The four filters are applied in order of how badly they should be allowed to
    fail. Exclusions are absolute and come first, because a user who said "never
    this" has said something about every roll rather than about this one.
    ``min_creativity`` is absolute too -- an extreme-only variant at Creativity 2
    would break the scale -- and so is compatibility; an incoherent pair is worse
    than a missing line. Anti-repetition is only a weight, because "avoid what you
    used last time" must never become "produce nothing".

    The note is the difference between the two ways this returns ``None``. An
    axis with nothing eligible at this position is ordinary and says nothing; an
    axis whose every eligible treatment the user excluded is a configuration that
    cannot do what it looks like it does, and saying so is the whole point of
    letting exclusions be absolute.
    """
    excluded = frozenset(excluded or ())
    eligible = [v for v in axis.eligible(creativity) if _compatible(lib, v, chosen)]
    if not eligible:
        return None, ""

    if excluded:
        kept = [v for v in eligible if v.identifier not in excluded]
        if not kept:
            # Reached only when compatibility has already narrowed the pool --
            # a wholly excluded axis is dropped before the draw, in roll(). So
            # the sentence names the combination rather than the exclusions
            # alone, which would send somebody looking in the wrong place.
            return None, (f"{axis.label}: everything that fits the rest of this roll is "
                          "excluded, so the axis was left out of the brief.")
        eligible = kept

    weights = [v.weight * _preference_boost(lib, v, chosen) for v in eligible]
    if penalty > 0 and recent:
        damped = [weight * (1.0 - penalty) if variant.identifier in recent else weight
                  for variant, weight in zip(eligible, weights)]
        # At Creativity 10 the penalty is total, so a pool made entirely of
        # recent choices sums to zero. Falling back to the unpenalised weights
        # is the difference between "strongly avoid repeats" and "refuse to
        # direct this axis at all once the library runs short".
        if sum(damped) > 0:
            weights = damped

    return _weighted_choice(rng, eligible, weights), ""


def _active_axes(lib, eligible, creativity, rng) -> list[str]:
    """Which Vary axes get a line this roll, in the library's own axis order.

    The count comes from the package's activation policy. *Which* ones is a
    weighted draw without replacement, so at Creativity 2 the single active axis
    is usually the medium rather than, say, detail emphasis -- one axis worth of
    direction should be the axis that changes the picture.

    The survivors are then sorted back into the package's axis order, because
    the brief reads as a list somebody wrote and a list that arrives in draw
    order reads as a list somebody shuffled.
    """
    low, high = lib.policy.activation_range(creativity, len(eligible))
    if high <= 0:
        return []
    count = rng.randint(low, high) if high > low else high

    pool = list(eligible)
    weights = [lib.priority_of(key) for key in pool]
    picked: list[str] = []
    for _ in range(min(count, len(pool))):
        chosen = _weighted_choice(rng, pool, weights)
        if chosen is None:
            break
        index = pool.index(chosen)
        pool.pop(index)
        weights.pop(index)
        picked.append(chosen)

    order = {key: position for position, key in enumerate(lib.axis_keys)}
    return sorted(picked, key=lambda key: order.get(key, len(order)))


# --------------------------------------------------------------------------- #
# The brief
# --------------------------------------------------------------------------- #


def _discouraged(lib, source: str, creativity: int) -> tuple[str, ...]:
    """Visual clichés to steer away from, minus any the user actually asked for.

    Only at the top of the scale, where the package's own anti-repetition
    strength turns on. At Creativity 3 somebody is asking for a nudge, and a
    paragraph of prohibitions is not a nudge.

    The filter matters more than the list. "ultra detailed" is a cliché until
    the user types it, at which point it is a request, and a brief that told the
    model to avoid the user's own words would be the Director overruling the
    person it works for.
    """
    if lib.anti_repetition.strength_at(creativity) < 0.5:
        return ()
    text = str(source or "").casefold()
    return tuple(phrase for phrase in lib.anti_repetition.generic_treatments
                 if phrase.split(" as ")[0].casefold() not in text)


def assemble(items, avoid=()) -> str:
    """The Creative Direction block, or an empty string when there is nothing to say.

    Empty is a real answer and the caller depends on it: no items means no block,
    which means the user turn is exactly the user turn the writer has always
    been given. That is how Creativity 1 stays a compatibility guarantee rather
    than a claim.
    """
    items = tuple(items)
    if not items:
        return ""
    lines = [BRIEF_HEADING, SOURCE_PRIORITY_RULE]
    lines.extend(item.line for item in items)
    if avoid:
        lines.append("Avoid falling back on " + ", ".join(avoid)
                     + " unless the user's own words ask for them.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# One roll
# --------------------------------------------------------------------------- #


def default_settings(lib=None) -> dict:
    """The package's own fresh-install axis settings.

    Natural for anything the package does not name, which on a fresh install is
    every axis: a new configuration should contain no art direction the user did
    not ask for. An axis missing from ``defaults.json`` is therefore silent
    rather than varied.
    """
    lib = lib or library_module.library()
    modes = lib.defaults.get("axis_modes") or {}
    fixed = lib.defaults.get("fixed_values") or {}
    excluded = lib.defaults.get("excluded_values") or {}
    return {key: AxisSetting(mode=str(modes.get(key, NATURAL)).casefold(),
                             fixed_id=fixed.get(key),
                             excluded_ids=frozenset(excluded.get(key) or ()))
            for key in lib.axis_keys}


def roll(source: str, creativity: int, creative_seed=RANDOM_SEED, settings=None,
         history=(), lib=None) -> CreativeRecipe:
    """One creative roll. No inference, no network, no second anything.

    ``history`` is the recent rolls' variant ids, newest first or oldest first --
    order is not used, only membership, because "did we just do this" is the
    only question anti-repetition asks.

    Everything about the result is decided here, including the seed the single
    Krea writer call will run at. That the writer's seed is *derived* from the
    Creative seed rather than drawn separately is what makes one number enough
    to reproduce a roll: the same Creative seed gives the same recipe and asks
    the model the same way about it.
    """
    lib = lib or library_module.library()
    creativity = max(0, min(10, int(creativity)))
    resolved = resolve_seed(creative_seed)
    director_seed, llm_seed = derive(resolved)
    rng = random.Random(director_seed)

    settings = dict(settings or default_settings(lib))
    locked = explicit_locks(source, lib)
    tier = lib.policy.tier(creativity)
    recent = frozenset(str(identifier) for identifier in history or ())
    penalty = lib.anti_repetition.strength_at(creativity)

    chosen: dict = {}
    items: list[CreativeRecipeItem] = []
    notes: list[str] = []

    def record(axis, variant, strength, origin):
        chosen[axis.key] = variant
        items.append(CreativeRecipeItem(
            axis=axis.key, label=axis.label, variant_id=variant.identifier,
            variant_label=variant.label, expression=variant.expression(strength),
            strength=strength, source=origin))

    # Fixed first, and before any Vary axis is drawn: a pinned value is closer
    # to the user's intent than anything this module chooses, so the choices
    # have to be made compatible with it rather than the other way round.
    # Fixed survives Creativity 0 and 1 for the same reason -- it is explicit
    # configuration, not variation, and the scale governs variation.
    fixed_strength = library_module.tier_at_or_below(tier)
    for key in lib.axis_keys:
        setting = settings.get(key) or AxisSetting()
        if setting.mode != FIXED or key in locked:
            continue
        axis = lib.axis(key)
        variant = axis.variant(setting.fixed_id) if axis else None
        if variant is None:
            # The pinned id is gone -- a library update, or a hand-edited
            # preferences file. The axis falls silent for this roll rather than
            # the panel refusing to build or a substitute being invented.
            continue
        record(axis, variant, fixed_strength, FIXED)

    if tier in library_module.TIERS:
        # An axis every one of whose eligible treatments has been excluded is
        # dropped here rather than in the draw. Left in, it would consume one of
        # the activation slots the Creativity position allows and then produce
        # nothing -- so a user who excluded a whole small axis would find their
        # other axes quietly directed less often, with no visible cause.
        eligible = []
        for key in lib.axis_keys:
            setting = settings.get(key) or AxisSetting()
            axis = lib.axis(key)
            if setting.mode != VARY or key in locked or key in chosen or axis is None:
                continue
            available = axis.eligible(creativity)
            if not available:
                continue
            if setting.excluded_ids and all(variant.identifier in setting.excluded_ids
                                            for variant in available):
                notes.append(f"{axis.label}: every treatment available at Creativity "
                             f"{creativity} is excluded, so the axis was left out of "
                             "the brief.")
                continue
            eligible.append(key)
        for key in _active_axes(lib, eligible, creativity, rng):
            axis = lib.axis(key)
            setting = settings.get(key) or AxisSetting()
            variant, note = _choose_variant(lib, axis, creativity, rng, chosen, recent,
                                            penalty, setting.excluded_ids)
            if variant is not None:
                record(axis, variant, tier, VARY)
            elif note:
                notes.append(note)

    order = {key: position for position, key in enumerate(lib.axis_keys)}
    items.sort(key=lambda item: order.get(item.axis, len(order)))
    avoid = _discouraged(lib, source, creativity) if items else ()

    return CreativeRecipe(
        creative_seed=resolved, llm_seed=llm_seed, creativity=creativity,
        items=tuple(items), avoid=avoid, locked=tuple(sorted(locked)),
        library_version=lib.version, brief=assemble(items, avoid),
        notes=tuple(notes))


def replay(source: str, creativity: int, creative_seed, llm_seed, recipe_ids,
           lib=None) -> CreativeRecipe:
    """The recipe a record describes, rebuilt exactly, with nothing drawn.

    This is the second of the two honest ways to reproduce a creative image, and
    it is the one for continuing from an old idea rather than re-making an old
    picture. The first -- take the final expanded prompt out of the file and skip
    the writer entirely -- is what an ordinary infotext paste does.

    Nothing here is random and nothing here consults history. That is the whole
    point: a fresh roll at the recorded seed would re-derive the same *draw*, but
    the draw is weighted by recent choices, and the recent choices of a machine
    six months later are not the recent choices the original roll saw. So the
    recorded ids are used as ids, in the library's own axis order, at the tier
    the recorded position calls for.

    ``recipe_ids`` is the ``axis=variant_id`` form :attr:`CreativeRecipe.compact`
    writes. Ids the current package no longer has are dropped with a note rather
    than substituted: a replay that quietly swapped a treatment would be a
    reproduction that is not one, which is exactly the failure this exists to
    avoid.
    """
    lib = lib or library_module.library()
    creativity = max(0, min(10, int(creativity)))
    resolved = resolve_seed(creative_seed)
    try:
        writer_seed = int(llm_seed)
    except (TypeError, ValueError):
        writer_seed = derive(resolved)[1]

    strength = library_module.tier_at_or_below(lib.policy.tier(creativity))
    order = {key: position for position, key in enumerate(lib.axis_keys)}

    items: list[CreativeRecipeItem] = []
    notes: list[str] = []
    for axis_key, variant_id in parse_recipe(recipe_ids):
        axis = lib.axis(axis_key)
        variant = axis.variant(variant_id) if axis is not None else None
        if variant is None:
            notes.append(f"{axis_key}={variant_id} is not in creativity library "
                         f"{lib.version}, so that line could not be replayed.")
            continue
        items.append(CreativeRecipeItem(
            axis=axis.key, label=axis.label, variant_id=variant.identifier,
            variant_label=variant.label, expression=variant.expression(strength),
            strength=strength, source=REPLAY))

    items.sort(key=lambda item: order.get(item.axis, len(order)))
    avoid = _discouraged(lib, source, creativity) if items else ()
    return CreativeRecipe(
        creative_seed=resolved, llm_seed=writer_seed, creativity=creativity,
        items=tuple(items), avoid=avoid,
        locked=tuple(sorted(explicit_locks(source, lib))),
        library_version=lib.version, brief=assemble(items, avoid),
        notes=tuple(notes), replayed=True)


def parse_recipe(recipe_ids) -> tuple[tuple[str, str], ...]:
    """``"medium=oil_impasto, mood=monumental"`` as ordered ``(axis, id)`` pairs.

    Accepts the compact string a metadata field holds and the pairs a caller
    already has, because both spellings turn up: one comes out of a PNG, the
    other out of a recipe still in memory. Anything that is not a pair is
    dropped -- this parses somebody's file, and a malformed line is a line to
    ignore rather than an exception to raise into a paste.
    """
    if not recipe_ids:
        return ()
    if isinstance(recipe_ids, str):
        entries = [part.strip() for part in recipe_ids.split(",")]
    else:
        entries = list(recipe_ids)

    parsed: list[tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            axis_key, variant_id = entry
        elif isinstance(entry, str) and "=" in entry:
            axis_key, variant_id = entry.split("=", 1)
        else:
            continue
        axis_key, variant_id = str(axis_key).strip(), str(variant_id).strip()
        if axis_key and variant_id:
            parsed.append((axis_key, variant_id))
    return tuple(parsed)
