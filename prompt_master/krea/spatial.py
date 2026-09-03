"""The Spatial Layout: what the user drew, and the prompt code builds from it.

One layout in, one Krea 2 structured prompt out:

    serialized layout (from the browser)
        -> parse and validate            (this module, no model)
        -> immutable regions
        -> deterministic hints           (framing, angle, position, size)
        -> one compact structured prompt (this module, no model)

**Nothing in this module performs inference or touches a network**, and nothing
in it is allowed to invent a coordinate. That is the whole reason it exists as
its own file. The tempting way to turn a scene and five boxes into a structured
prompt is to hand both to a language model and ask for the JSON, and every
failure mode of that is silent: a coordinate drifts, two regions merge, a user's
own wording is "improved", the reply is not valid JSON, or the same layout
produces a different prompt tomorrow and the image cannot be reproduced.

So the division is the one :mod:`prompt_master.krea.director` already draws, one
layer further down. A model may write *text fields* -- the global scene, the
background -- and code supplies the *structure*. The elements array is built
here, from what the user drew, in the order they stacked it, with their words in
it verbatim.

Literal commands stay in their box
----------------------------------
A region prompt may carry ``[[...]]`` commands -- a reference instruction, a
wildcard, an extra-network tag -- and those are lifted out here, at parse time,
by :mod:`prompt_master.krea.literals`. What the Composer is shown and what this
module rewrites is the cleaned text; the payloads are held on the Region and
written back into that element's ``desc`` and nowhere else. A command written
inside Region 1 reaches Region 1, or it reaches nothing: it never lands in the
global scene, in the background, or in another region.

What is authoritative, and what is not
-------------------------------------
These are the user's and no pass may change them: the bbox, the raw region
prompt, the Object/Text type, the visible text string, the framing selection,
the camera angle and the z-order. They arrive as a serialized layout, they are
validated here, and the validated values are what reaches the prompt. A language
model may be *shown* them for context -- that is what
:mod:`prompt_master.krea.composer` is for -- and what it returns is two strings,
neither of which is any of the above.

Coordinates are normalized 0..1000
----------------------------------
Not pixels. A layout drawn at 1024x1344 and generated at 1536x2016 is the same
composition, and normalized coordinates are what makes that true without any
reprojection step that could round a box somewhere the user did not put it. It
also means the stored layout survives a resolution change untouched, which §6.4
of the design intent asks for in the strongest terms it uses anywhere: *never
silently delete layout state*.

Spatial guidance, not a mask
----------------------------
Nothing here confines anything. A bounding box in a text prompt is a strong,
repeated, multi-signal hint about where a subject belongs -- structural
separation, numbers, the user's own words, a framing hint, an angle hint and a
plain-English position hint, all saying the same thing -- and it is still a
hint. The words this module produces are chosen accordingly, and the UI says so
in the same terms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import gcd

from . import extra_networks, literals

VERSION = 1
"""The editable-state version this module reads and writes.

A layout from the future is *refused*, visibly, rather than read as far as it
parses: a v2 that moved a field would otherwise be interpreted as a v1 with
several regions missing, and a user would see boxes disappear with no
explanation. §7 asks for exactly that -- a warning rather than a silent
reinterpretation.
"""

PROMPT_VERSION = 1
"""The version of the *model-facing* structure this module emits.

Recorded in infotext beside the layout, and deliberately a separate number. The
editable state and the prompt built from it change for different reasons, and an
image made with prompt structure 1 stays reproducible from its own Prompt line
whatever happens to either.
"""

OBJECT = "obj"
TEXT = "text"
TYPES = (OBJECT, TEXT)

SMART = "smart"
DIRECT = "direct"
COMPOSE_MODES = (SMART, DIRECT)

SCALE = 1000
"""Normalized coordinate space: 0..1000 on both axes, as §4 specifies."""

MAX_REGIONS = 24
"""How many regions one layout may carry.

Not a Krea limit -- Krea documents none. It is the point past which the elements
array is longer than the scene it decorates, every region's share of the model's
attention is negligible, and the structured prompt is mostly punctuation. A
layout with more is truncated with a note rather than refused, because the
first twenty-four boxes are still what the user drew.
"""


# --------------------------------------------------------------------------- #
# The deterministic hints
# --------------------------------------------------------------------------- #

FRAMINGS = {
    "": "",
    "Extreme close-up": "shown as an extreme close-up",
    "Close-up": "shown in a close-up view",
    "Medium close-up": "shown in a medium close-up",
    "Medium shot": "shown as a medium shot",
    "Cowboy shot": "shown as a cowboy shot, from mid-thigh up",
    "Full body": "shown full body, head to feet",
    "Wide shot": "shown in a wide shot",
    "Headshot": "shown as a headshot",
}
"""Framing selections, and the phrase each one contributes. Fixed text.

A dictionary and not a template, because every one of these is a phrase a
prompt-writing human would use and none of them is derivable from the others.
The empty key is the default and contributes nothing at all -- the same rule
Natural follows one layer up: an unmade decision is silence, not a hedge.
"""

ANGLES = {
    "": "",
    "Front": "seen straight on from the front",
    "3/4 left": "in a three-quarter view from the left",
    "3/4 right": "in a three-quarter view from the right",
    "Profile left": "seen in profile, facing left",
    "Profile right": "seen in profile, facing right",
    "From behind": "seen from behind",
    "Low angle": "seen from a low angle, looking up",
    "High angle": "seen from a high angle, looking down",
    "Top-down": "seen from directly overhead",
    "Eye level": "seen at eye level",
}
"""Camera-angle selections, and their phrases. Fixed text, same rules."""

_ROWS = ("upper", "center", "lower")
_COLUMNS = ("left", "center", "right")

SIZE_HINTS = (
    (0.02, "occupying a very small area"),
    (0.08, "occupying a small area"),
    (0.25, "occupying a medium-sized area"),
    (0.50, "occupying a large area"),
)
"""Area fraction thresholds, ascending, and what each one is called.

Anything above the last threshold is :data:`DOMINANT`. Thresholds rather than a
formula because the words are what the model reads, and there are five useful
ones -- a continuous "occupying 8.54% of the frame" is a number a text-to-image
model has no calibration for.
"""

DOMINANT = "occupying most of the frame"

CENTRE_HINT = "positioned in the center of the frame"


def clamp(value) -> int:
    """One coordinate, as an integer inside 0..``SCALE``.

    Clamped rather than refused: a box dragged past the edge of the canvas is a
    box the user meant to touch the edge, and the browser has already stopped it
    at the boundary in every case this can see. What is refused, elsewhere, is a
    box with no area -- because that is not a truncated intent, it is a click.
    """
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(SCALE, number))


def normalise_bbox(values) -> list[int] | None:
    """``[x0, y0, x1, y1]`` clamped and ordered, or ``None`` if it has no area.

    A drag from bottom-right to top-left arrives with its coordinates reversed,
    which is not an error and is not the browser's to fix silently either -- the
    canonical form is defined here, so that a layout hand-edited into the state
    box behaves the same as one drawn with a mouse.
    """
    try:
        x0, y0, x1, y1 = (clamp(value) for value in tuple(values)[:4])
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def position_hint(bbox) -> str:
    """Where the box sits, in the words a person would use.

    The nine-cell grid is the thirds grid the canvas draws, which is the point:
    the sentence the model reads and the guides the user drew against describe
    the same division of the frame.
    """
    x0, y0, x1, y1 = bbox
    column = _COLUMNS[min(2, int(((x0 + x1) / 2) * 3 // SCALE))]
    row = _ROWS[min(2, int(((y0 + y1) / 2) * 3 // SCALE))]
    if row == "center" and column == "center":
        return CENTRE_HINT
    return f"positioned in the {row}-{column} area"


def size_hint(bbox) -> str:
    """How much of the frame the box covers, in five bands."""
    x0, y0, x1, y1 = bbox
    fraction = ((x1 - x0) * (y1 - y0)) / float(SCALE * SCALE)
    for threshold, words in SIZE_HINTS:
        if fraction < threshold:
            return words
    return DOMINANT


def aspect_ratio(width, height) -> str:
    """``"4:3"`` for the frame this generation is actually making.

    Read from the generation's own width and height rather than from the canvas
    the layout was drawn on, because those can differ -- somebody draws at 1024
    square and generates at 832x1216 -- and the number the model is told should
    describe the picture it is about to make.

    Exact when the reduced ratio is small enough to be recognisable, and snapped
    to a common ratio when it is not and one is within three per cent. 832x1216
    reduces to 13:19, which is true and useless; it is 2:3 to within two and a
    half per cent, and "2:3" is a form the model has seen.
    """
    try:
        wide, high = int(width), int(height)
    except (TypeError, ValueError):
        return ""
    if wide <= 0 or high <= 0:
        return ""
    divisor = gcd(wide, high)
    reduced = (wide // divisor, high // divisor)
    if max(reduced) <= 16:
        return f"{reduced[0]}:{reduced[1]}"
    target = wide / high
    near = min(COMMON_RATIOS, key=lambda ratio: abs(ratio[0] / ratio[1] - target))
    if abs(near[0] / near[1] - target) / target <= 0.03:
        return f"{near[0]}:{near[1]}"
    return f"{reduced[0]}:{reduced[1]}"


COMMON_RATIOS = ((1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16), (5, 4),
                 (4, 5), (16, 10), (10, 16), (21, 9), (9, 21), (7, 5), (5, 7),
                 (2, 1), (1, 2))
"""Ratios worth snapping to. Every one of them is a form a prompt commonly names."""


# --------------------------------------------------------------------------- #
# One region
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Region:
    """One bounded piece of user intent. Every field here is the user's.

    Constructed only by :func:`parse`, which is where validation lives, so a
    Region that exists is a Region that is safe to put in a prompt.
    """

    identifier: str
    name: str
    kind: str
    bbox: tuple[int, int, int, int]
    prompt: str = ""
    """The region's transformable text: what the user typed, minus its literals.

    This is the string every other pass sees -- the Composer's context line, the
    element's ``desc``, the panel's summary -- because a literal command inside a
    box is content for the image model and for nothing before it.
    """

    text: str = ""
    framing: str = ""
    angle: str = ""
    z: int = 0
    index: int = 0
    """Draw order. Not user intent and not in the prompt -- it is the tie-break
    that keeps two regions at the same z from swapping places between runs."""

    raw_prompt: str = ""
    """Exactly what the user typed, ``[[...]]`` included, or "" when they are the
    same thing.

    Kept so :meth:`state` can write the canvas back as it was drawn. A layout
    round-trip that returned the *cleaned* prompt would silently eat a user's
    reference instruction the first time they opened the editor after a restore,
    which is the one failure §6.4 of the layout design calls unforgivable.
    """

    literal_prefix: str = ""
    literal_suffix: str = ""
    """This region's two Literal Prompt fields, exactly as typed.

    Kept beside the payloads below rather than only inside them, for the same
    reason :attr:`raw_prompt` is kept beside :attr:`prompt`: the editor has to
    be able to draw the boxes again with what somebody put in them, and a round
    trip that returned only the merged payloads could not tell a field apart
    from a ``[[...]]`` typed in the prompt -- so reopening the editor would move
    a command out of the prompt box and into a field, silently, once.
    """

    ui_shape: str = "rect"
    ui_rotation: int = 0
    """What the editor drew, and how far round it turned it.

    Editor metadata, carried and never read. The layout workspace lets a region
    be laid out as a head, an arm or a foot rather than as an unlabelled
    rectangle, and lets a silhouette be rotated so that a raised arm looks like
    a raised arm; the *model* is handed the same axis-aligned bounding box it
    has always been handed, in the same normalized units, whatever silhouette is
    drawn inside it.

    They exist on the Region for one reason: :meth:`state` has to be able to
    hand the canvas back the way it was drawn. Without them, a round trip
    through here -- a restore from an image, a saved layout reloaded -- would
    silently return every silhouette as a rectangle and every rotation as zero,
    which is the same class of loss ``raw_prompt`` exists to prevent.

    Nothing in :meth:`describe`, :meth:`element` or :func:`compose` reads
    either one, and no test of the composed prompt can be made to change by
    setting them. An unknown shape is kept verbatim rather than rejected: a
    build that predates a shape must round-trip a document that uses it, or
    opening an image's layout in an older Forge would delete the drawing.
    """

    prefix_literals: tuple[str, ...] = ()
    suffix_literals: tuple[str, ...] = ()
    """This region's literal payloads, in source order, split by direction.

    Strings and not :class:`~prompt_master.krea.literals.LiteralCommand`
    objects, because by the time a Region exists the only questions left are
    "which side" and "in what order", and both are answered by two tuples. What
    is *not* stored anywhere is a copy in the global scope: a region's commands
    reach the prompt through this region's ``desc`` or they do not reach it at
    all.
    """

    @property
    def source_prompt(self) -> str:
        """What the user actually typed in the region's prompt box.

        The raw text when a literal command was lifted out of it, and the
        ordinary prompt otherwise -- so a region that never carried one is
        recorded exactly as it always was.
        """
        return self.raw_prompt or self.prompt

    @property
    def literals(self) -> tuple[str, ...]:
        """Every literal payload this region carries, prefixes first."""
        return tuple(self.prefix_literals) + tuple(self.suffix_literals)

    @property
    def has_content(self) -> bool:
        """Whether the user put anything in this region at all.

        Three ways to have, and the second and third are why this is a property
        rather than ``bool(region.prompt)`` at each call site: a region whose
        whole content is ``[[Her shirt from image 1]]`` has an empty
        transformable prompt *by design*, and discarding it for that would
        discard exactly the regions this feature exists to carry.
        """
        return bool(self.prompt or self.prefix_literals or self.suffix_literals)

    @property
    def area(self) -> float:
        """Share of the frame, 0..1."""
        x0, y0, x1, y1 = self.bbox
        return ((x1 - x0) * (y1 - y0)) / float(SCALE * SCALE)

    def hints(self, auto_position: bool = True) -> list[str]:
        """The deterministic phrases this region contributes, in a fixed order.

        Framing, then angle, then position, then size -- and each one omitted
        entirely when it has nothing to say, rather than rendered as "framing:
        default". The order is the order a person describes a shot in, and it is
        fixed so that the same region produces the same string on every machine
        and in every process.
        """
        found = []
        if self.kind == OBJECT:
            framing = FRAMINGS.get(self.framing, "")
            if framing:
                found.append(framing)
            angle = ANGLES.get(self.angle, "")
            if angle:
                found.append(angle)
        if auto_position:
            found.append(position_hint(self.bbox))
            found.append(size_hint(self.bbox))
        return found

    def describe(self, auto_position: bool = True, literals: bool = True) -> str:
        """The element's ``desc``: the user's words first, then the hints.

        The user's own text is present verbatim and first. That is the promise
        this whole feature makes about region prompts, and it is kept by
        concatenation rather than by asking anything to preserve it.

        Literal commands wrap that content rather than joining the end of it:
        prefixes, then the region's own words, then the deterministic hints,
        then suffixes. So ``+[[Her shirt from image 1]]`` reads as the first
        thing said about the element and ``-[[__fabric_detail__]]`` as the last,
        which is what ``+`` and ``-`` mean everywhere else in the syntax, and the
        existing hint order is untouched in between.

        ``literals`` false builds the same description without them, for the
        prompt Stage 2 may inherit. It is not a debugging switch: a Stage 1
        reference instruction is meaningless to a Stage 2 model that has no
        ImageStitch reference behind it, and the only way to be sure none
        travels is to have a representation none was ever put into.
        """
        prefix = tuple(self.prefix_literals) if literals else ()
        suffix = tuple(self.suffix_literals) if literals else ()
        base = self.prompt.strip()
        if not base and self.kind == TEXT and not (prefix or suffix):
            base = DEFAULT_TEXT_DESC
        parts = list(prefix)
        if base:
            parts.append(base)
        parts.extend(self.hints(auto_position))
        parts.extend(suffix)
        return ", ".join(parts)

    def element(self, auto_position: bool = True, literals: bool = True) -> dict:
        """This region as one entry of ``compositional_deconstruction.elements``.

        A text region carries ``text`` *and* ``desc`` and they mean different
        things: only ``text`` is intended to become visible writing, and ``desc``
        is how it should look and where it goes. Merging them would ask the model
        to render the adjectives.
        """
        found: dict = {"type": self.kind, "bbox": list(self.bbox)}
        if self.kind == TEXT:
            found["text"] = self.text
        found["desc"] = self.describe(auto_position, literals)
        return found

    def state(self) -> dict:
        """This region back in editable form, with the keys in a fixed order."""
        found = {"id": self.identifier, "name": self.name, "type": self.kind,
                 "bbox": list(self.bbox), "prompt": self.source_prompt}
        if self.kind == TEXT:
            found["text"] = self.text
        # Written only when they carry something, so a layout drawn before the
        # region literal fields existed round-trips to the same bytes it always
        # did and a document this build writes stays readable by one that
        # predates them.
        if self.literal_prefix:
            found["literal_prefix"] = self.literal_prefix
        if self.literal_suffix:
            found["literal_suffix"] = self.literal_suffix
        found.update({"framing": self.framing, "angle": self.angle, "z": self.z})
        # The same rule as the region literals above, for the same reason: a
        # rectangle layout round-trips to the bytes it always did, so a document
        # this build writes stays readable by one that predates the editor
        # metadata entirely.
        if self.ui_shape and self.ui_shape != "rect":
            found["ui_shape"] = self.ui_shape
        if self.ui_rotation:
            found["ui_rotation"] = self.ui_rotation
        return found


DEFAULT_TEXT_DESC = "readable text"
"""What a text region's description falls back to when the user wrote none.

Not empty: ``desc`` is where the position and size hints live, and a text
element whose description is nothing but "positioned in the upper-center area"
reads as though the *area* is the thing to draw.
"""


# --------------------------------------------------------------------------- #
# One layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Layout:
    """A whole spatial layout, validated. Empty is a perfectly ordinary answer.

    ``notes`` carries every reason a region was skipped or a field dropped.
    Nothing here is thrown away quietly: §10 forbids guessing a replacement, and
    the only honest alternative to guessing is saying what was lost.
    """

    width: int = 0
    height: int = 0
    grid: str = "thirds"
    compose_mode: str = SMART
    auto_position_hint: bool = True
    regions: tuple[Region, ...] = ()
    notes: tuple[str, ...] = ()
    unreadable: bool = False
    """True when the layout could not be read at all -- bad JSON, or a version
    this build does not know. Distinct from "no regions", which is a layout the
    user has simply not drawn in yet."""

    def __bool__(self) -> bool:
        return bool(self.regions)

    @property
    def ordered(self) -> tuple[Region, ...]:
        """The regions in the order they go into the prompt: z, then draw order.

        Stable, and stated: an elements array that reordered itself between two
        generations of the same layout would make an A/B comparison meaningless
        and a replay a coin toss.
        """
        return tuple(sorted(self.regions, key=lambda region: (region.z, region.index)))

    def state(self) -> dict:
        """The editable state, normalised, with keys in a fixed order.

        Re-serialized from what parsed rather than passed through, so that what
        an image records is what this build actually used -- clamped
        coordinates, dropped unknown fields, canonical ordering. A record of the
        raw browser string would preserve junk that had no effect on the image.
        """
        return {"version": VERSION,
                "canvas": {"width": int(self.width), "height": int(self.height),
                           "grid": self.grid},
                "compose_mode": self.compose_mode,
                "auto_position_hint": bool(self.auto_position_hint),
                "regions": [region.state() for region in self.ordered]}

    def serialize(self) -> str:
        """The editable state as one compact line, for the state box and infotext."""
        return json.dumps(self.state(), ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Reading what the browser sent
# --------------------------------------------------------------------------- #


def parse(serialized, *, width=0, height=0) -> Layout:
    """One serialized layout, validated into a :class:`Layout`.

    Never raises. Every way this can go wrong produces a Layout that says what
    went wrong, because the caller is a processing hook with an image generation
    already running behind it: there is no failure here worth refusing a
    generation over, and every one of them is worth a sentence.

    Three outcomes worth telling apart:

    * **nothing** -- an empty box, which is a user who has not opened the editor.
      No regions, no notes, ``unreadable`` false. The generation is an ordinary
      Creative Mode generation and says nothing about layouts.
    * **unreadable** -- malformed JSON, or a ``version`` from a later build.
      No regions, one note, ``unreadable`` true. The caller says so out loud
      rather than generating as though the user had drawn nothing.
    * **read** -- regions, and a note for each one that was skipped.

    ``width`` and ``height`` are the generation's, and are used only to fill in a
    canvas the state does not carry. They never move a coordinate: the stored
    boxes are normalized and a resolution change is not a reprojection.
    """
    text = str(serialized or "").strip()
    if not text:
        return Layout(width=int(width or 0), height=int(height or 0))

    try:
        raw = json.loads(text)
    except Exception:
        return Layout(width=int(width or 0), height=int(height or 0), unreadable=True,
                      notes=("The spatial layout could not be read as JSON, so it was "
                             "not applied.",))
    if not isinstance(raw, dict):
        return Layout(width=int(width or 0), height=int(height or 0), unreadable=True,
                      notes=("The spatial layout was not an object, so it was not "
                             "applied.",))

    version = raw.get("version", VERSION)
    try:
        version = int(version)
    except (TypeError, ValueError):
        version = -1
    if version != VERSION:
        # Refused rather than read as far as it parses. A later version that
        # renamed one field would otherwise present as this one with several
        # regions missing, and boxes vanishing without explanation is the single
        # worst thing this feature could do to somebody's afternoon.
        return Layout(width=int(width or 0), height=int(height or 0), unreadable=True,
                      notes=(f"The spatial layout is version {raw.get('version')!r}, "
                             f"which this build cannot read (it reads version "
                             f"{VERSION}). It was not applied and it has not been "
                             f"changed.",))

    canvas = raw.get("canvas") if isinstance(raw.get("canvas"), dict) else {}
    notes: list[str] = []
    regions: list[Region] = []
    listed = raw.get("regions")
    listed = listed if isinstance(listed, list) else []
    if len(listed) > MAX_REGIONS:
        notes.append(f"The layout carries {len(listed)} regions; the first "
                     f"{MAX_REGIONS} were used.")
        listed = listed[:MAX_REGIONS]

    for position, entry in enumerate(listed):
        region, why = _region(entry, position)
        if region is None:
            notes.append(why)
            continue
        if why:
            notes.append(why)
        regions.append(region)

    mode = str(raw.get("compose_mode") or SMART).strip().casefold()
    if mode not in COMPOSE_MODES:
        mode = SMART

    return Layout(
        width=_positive(canvas.get("width"), width),
        height=_positive(canvas.get("height"), height),
        grid=str(canvas.get("grid") or "thirds"),
        compose_mode=mode,
        auto_position_hint=bool(raw.get("auto_position_hint", True)),
        regions=tuple(regions),
        notes=tuple(notes))


def _positive(value, fallback) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    if number > 0:
        return number
    try:
        return max(0, int(fallback))
    except (TypeError, ValueError):
        return 0


def _region(entry, position: int) -> tuple[Region | None, str]:
    """One entry of ``regions``, or ``None`` and the reason it was skipped.

    The second half of the tuple is a note even when the region survives: a
    framing this build does not know is dropped rather than passed through to
    the prompt, and dropping a user's selection silently is exactly the failure
    §10 is about.
    """
    label = f"Region {position + 1}"
    if not isinstance(entry, dict):
        return None, f"{label} was not an object and was skipped."

    identifier = str(entry.get("id") or f"r{position + 1}").strip() or f"r{position + 1}"
    name = str(entry.get("name") or "").strip() or identifier
    label = f"Region {position + 1} ({name})"

    bbox = normalise_bbox(entry.get("bbox"))
    if bbox is None:
        # Never a guess. A box with no area is a click, a bad paste or a bug,
        # and there is no coordinate this module is entitled to invent for it.
        return None, (f"{label} has no usable box (a region needs a positive width "
                      f"and height) and was skipped.")

    notes = []
    kind = str(entry.get("type") or OBJECT).strip().casefold()
    if kind not in TYPES:
        notes.append(f"{label} has an unknown type {entry.get('type')!r} and was "
                     f"treated as an object.")
        kind = OBJECT

    raw_prompt = str(entry.get("prompt") or "").strip()
    # A LoRA tag in a region's own prompt is no more rewritable than one in the
    # global scene, and the Composer is handed this text exactly as the Writer is
    # handed that one. Bracketed before the parse so a region carries it out on
    # the path a typed ``[[...]]`` already takes, with the same scope and the
    # same restore -- see :mod:`~prompt_master.krea.extra_networks`.
    parsed = literals.parse(extra_networks.protect(raw_prompt),
                            scope=literals.REGION, region_id=identifier)
    prompt = parsed.clean_text.strip()
    for warning in parsed.warnings:
        notes.append(f"{label}: {warning}")

    # This region's own Literal Prompt fields, folded in by the same function
    # the global ones use and at the same priority: what the user typed with
    # brackets sits further from the description than what they typed in a
    # field, on both sides. One merge, one ordering rule, two scopes.
    literal_prefix = str(entry.get("literal_prefix") or "").strip()
    literal_suffix = str(entry.get("literal_suffix") or "").strip()
    parsed = literals.merge(parsed, literal_prefix, literal_suffix,
                            scope=literals.REGION, region_id=identifier)

    text = str(entry.get("text") or "").strip()
    if kind == TEXT and not text:
        return None, (f"{label} is a text region with no text to render and was "
                      f"skipped.")
    if kind == OBJECT and not (prompt or parsed.commands):
        # "No prompt" now means "nothing of the user's, in any form". A region
        # holding only [[Her shirt from image 1]] has an empty transformable
        # prompt on purpose, and skipping it would delete the box for having
        # said exactly what this feature was built to carry.
        return None, f"{label} has no prompt and was skipped."

    framing = str(entry.get("framing") or "").strip()
    if framing and framing not in FRAMINGS:
        notes.append(f"{label} asks for a framing this build does not know "
                     f"({framing!r}); no framing hint was added.")
        framing = ""
    angle = str(entry.get("angle") or "").strip()
    if angle and angle not in ANGLES:
        notes.append(f"{label} asks for a camera angle this build does not know "
                     f"({angle!r}); no angle hint was added.")
        angle = ""

    try:
        z = int(entry.get("z", position))
    except (TypeError, ValueError):
        z = position

    # Editor metadata, carried through untouched and never validated against a
    # list of known shapes. This module has no opinion about what the canvas
    # draws -- only about the rectangle it is drawn in -- and refusing a shape
    # it has not heard of would delete somebody's drawing on the way past.
    ui_shape = str(entry.get("ui_shape") or "rect").strip() or "rect"
    try:
        ui_rotation = int(entry.get("ui_rotation") or 0)
    except (TypeError, ValueError):
        ui_rotation = 0
    ui_rotation = max(-180, min(180, ui_rotation))

    return (Region(identifier=identifier, name=name, kind=kind, bbox=tuple(bbox),
                   prompt=prompt, text=text, framing=framing, angle=angle, z=z,
                   index=position,
                   raw_prompt=raw_prompt if raw_prompt != prompt else "",
                   ui_shape=ui_shape, ui_rotation=ui_rotation,
                   literal_prefix=literal_prefix, literal_suffix=literal_suffix,
                   prefix_literals=parsed.prefixes,
                   suffix_literals=parsed.suffixes),
            " ".join(notes))


# --------------------------------------------------------------------------- #
# The deterministic compositor
# --------------------------------------------------------------------------- #


def compose(layout: Layout, scene: str, background: str = "",
            ratio: str = "", literals: bool = True) -> str:
    """The final model-facing prompt: one compact structured Krea 2 prompt.

    This is the function the whole feature is arranged around, and it is worth
    saying what it is *not*: it is not a template a model fills in, and it is not
    a reformatting of something a model returned. ``scene`` and ``background``
    are two strings -- whatever wrote them, they are strings by the time they get
    here -- and everything else is built from the validated layout.

    The shape follows the design intent §4 exactly, in a fixed key order, with
    compact separators. Fixed order because a prompt that reorders its own keys
    between runs is a prompt whose A/B comparison means nothing; compact because
    every space in here is a token the image model's text encoder pays for and
    none of them is read by a person.

    ``literals`` false builds the same document with every region's literal
    payload left out, which is the representation Stage 2 may inherit. Two
    strings are built rather than one string edited, because the alternative --
    finding the payloads in the finished prompt and deleting them -- cannot tell
    a user's ``[[red hat]]`` from a red hat the Creative Writer thought of by
    itself, and would delete both.
    """
    elements = [region.element(layout.auto_position_hint, literals)
                for region in layout.ordered]
    payload = {
        "aspect_ratio": str(ratio or ""),
        "high_level_description": str(scene or "").strip(),
        "compositional_deconstruction": {
            "background": str(background or "").strip(),
            "elements": elements,
        },
    }
    if not payload["aspect_ratio"]:
        # Absent rather than empty. An empty ratio is a claim about the frame,
        # and this module would be making it up.
        payload.pop("aspect_ratio")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def summarise(layout: Layout) -> str:
    """One line naming what is in the layout, for the panel and the log."""
    if layout.unreadable:
        return "Spatial Layout: unreadable"
    if not layout.regions:
        return "Spatial Layout: no regions"
    objects = sum(1 for region in layout.regions if region.kind == OBJECT)
    texts = len(layout.regions) - objects
    parts = []
    if objects:
        parts.append(f"{objects} region{'' if objects == 1 else 's'}")
    if texts:
        parts.append(f"{texts} text region{'' if texts == 1 else 's'}")
    return "Spatial Layout: " + ", ".join(parts)


# --------------------------------------------------------------------------- #
# What pass 1 is told, and the one line it is told
# --------------------------------------------------------------------------- #

PLACEMENT_NOTE = (
    "Exact subject placement is supplied separately as a spatial layout, so "
    "describe the scene, its treatment and its mood without stating rigid "
    "screen positions for the subjects.")
"""The only thing the Creative Writer is ever told about the layout.

§2.2 of the design intent is unusually specific about this and it is worth
saying why in the code as well. Handing pass 1 the region prompts would put the
user's own words through a rewriter -- it would come back "improved", with the
elderly Japanese woman turned into an elderly woman with silver hair in a
different sentence -- and then the compositor would put the *original* words in
the elements array beside the rewrite, and the image model would be asked for
two of her.

So pass 1 is told that placement is handled and nothing else. One sentence, no
coordinates, no region text. It costs about thirty tokens of prompt evaluation
and it is what stops the first pass writing "centred in the frame" over a layout
that says otherwise.
"""


def directed(brief: str) -> str:
    """The Creative brief with the placement note on the end.

    Added only when there is a layout to justify it, which is what keeps the
    Creativity-1 compatibility guarantee intact: with no regions the brief is
    returned untouched, so a Creative Mode generation with Spatial Layout off is
    byte-identical to one made before this feature existed -- the same user turn,
    the same prompt-cache prefix, the same seconds.

    With no brief at all -- every axis Natural, or Creativity below 2 -- the note
    still needs the heading above it, because an unlabelled sentence dropped into
    a labelled turn is the one thing :mod:`prompt_master.krea.enhancer` is careful
    never to do.
    """
    from .director import BRIEF_HEADING

    brief = str(brief or "").strip()
    if not brief:
        return f"{BRIEF_HEADING}\n{PLACEMENT_NOTE}"
    return f"{brief}\n{PLACEMENT_NOTE}"
