"""The Spatial Layout, as the server sees it, with no model in the question.

One editor, one document, two backends. :mod:`prompt_master.krea.spatial` owns
the document -- what a region *is*, how a browser's JSON becomes a validated
:class:`~prompt_master.krea.spatial.Layout`, and how it is written back. This
module sits one layer above that and answers the questions neither backend
should be asking on its own:

    serialized layout (from the browser)
        -> Layout                    (prompt_master.krea.spatial)
        -> SpatialRequest            (this module)
             |
             +-- Krea 2    -> structured BBOX prompt      (prompt_master.krea.spatial)
             +-- Klein 9B  -> regional conditioning       (mc_spatial_klein)

**Nothing in here performs inference, touches a network, or imports the WebUI.**
It is arithmetic and a small state machine, which is what makes the availability
matrix and the Auto resolver testable without a host, a checkpoint or a card --
and those two are exactly the pieces that must never be decided in a browser.

Why the request object exists
-----------------------------
Because the alternative is a backend reading ``p.script_args``, a panel reading
``shared.opts``, and a test reading neither. The layout arrives as one string,
the source arrives from ImageStitch, and the mode arrives from a radio button;
parsing all three once into plain data means the rules below -- which modes are
selectable, what Auto resolves to, which boxes survive -- are stated once and
read three times rather than reimplemented per caller.

What is *not* here
------------------
No host state, no PIL work, no reference encoding, no sampler. A
:class:`SpatialSource` carries images because the modes are defined in terms of
them, but nothing in this module opens one. The mapping from a normalized box to
a concrete conditioning grid *is* here, because it is arithmetic; the tensor
whose shape it is given comes from the caller, which is the only place that can
know it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor

from .krea import spatial as document

SCALE = document.SCALE
"""The normalized coordinate space, 0..1000, shared with the editor.

Re-exported rather than redefined. There is one coordinate system in this
feature and a second constant naming it would be a second chance to disagree.
"""


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

AUTO = "auto"
REGIONAL_GENERATE = "regional_generate"
REGIONAL_IMG2IMG = "regional_img2img"
REFERENCE_REGIONS = "reference_regions"
STRICT_REGIONAL_EDIT = "strict_regional_edit"

MODES = (AUTO, REGIONAL_GENERATE, REGIONAL_IMG2IMG, REFERENCE_REGIONS,
         STRICT_REGIONAL_EDIT)
"""Every mode the panel offers, in the order it offers them."""

RESOLVED_MODES = (REGIONAL_GENERATE, REGIONAL_IMG2IMG, REFERENCE_REGIONS,
                  STRICT_REGIONAL_EDIT)
"""The modes a generation can actually run in. Auto is a question, not an answer."""

IMAGE_REQUIRED_MODES = (REGIONAL_IMG2IMG, REFERENCE_REGIONS, STRICT_REGIONAL_EDIT)
"""Modes that cannot run without a usable source image.

The compatibility matrix of §26, as a tuple, because it drives five separate
things -- the panel's enabled states, the Auto resolver, the runtime validation,
the tests and the diagnostics -- and five copies of a matrix is four too many.
"""

IMPLEMENTED_MODES = (AUTO, REGIONAL_GENERATE, REFERENCE_REGIONS)
"""The modes whose sampling behaviour this build actually implements.

The other two are offered, resolved and validated correctly and then have
nothing behind them. They are named here rather than quietly left out because
half of the design intent's compatibility matrix is about them, and a mode that
vanished would be indistinguishable from one that never existed.

What is *not* acceptable is running one of them as an ordinary generation. §10's
rule -- do not silently turn an explicit strict edit into a new txt2img -- says
nothing about whether the reason is a missing image or a missing implementation,
and the user's experience of the two is identical: they asked for their source
preserved and got a different picture with no error. So an unimplemented mode is
refused with a sentence, exactly as an unavailable one is.
"""

SOURCE_PRESERVING_MODES = (REGIONAL_IMG2IMG, STRICT_REGIONAL_EDIT)
"""Modes in which image #1 is a canvas rather than a reference.

These are the ones whose output geometry is the source image's, and the ones
where an aspect mismatch between the canvas and the source is worth warning
about rather than silently resolving.
"""

MODE_LABELS = {
    AUTO: "Auto",
    REGIONAL_GENERATE: "Regional Generate",
    REGIONAL_IMG2IMG: "Regional Img2Img",
    REFERENCE_REGIONS: "Reference + Regions",
    STRICT_REGIONAL_EDIT: "Strict Regional Edit",
}
"""What each mode is called, in the UI and in infotext.

Infotext records the *label* rather than the key, because a person reading a PNG
should not have to know that ``reference_regions`` is what the radio button
called "Reference + Regions". :func:`mode_from_label` reads them back.
"""

MODE_HELP = {
    AUTO: ("Generate regionally from noise when ImageStitch is empty; use "
           "ImageStitch as Klein reference conditioning when an image is present."),
    REGIONAL_GENERATE: ("Start from noise. ImageStitch is ignored. Region prompts "
                        "guide where concepts appear."),
    REGIONAL_IMG2IMG: ("Use ImageStitch image 1 as the starting image latent. Region "
                       "prompts guide changes; denoise controls how much the whole "
                       "image may move."),
    REFERENCE_REGIONS: ("Start the output from noise and use ImageStitch images as "
                        "Klein reference conditioning while region prompts guide "
                        "composition."),
    STRICT_REGIONAL_EDIT: ("Edit ImageStitch image 1 only inside the union of spatial "
                           "regions. Preserve the source outside the edit mask."),
}
"""§11 verbatim. Written here rather than in the panel for the same reason the
matrix is: the API has to be able to say the same thing the radio button says."""

NO_SOURCE_REASON = "Requires an enabled ImageStitch image."
"""Why a mode is disabled, when it is. One sentence, and always the same one."""

_LABEL_TO_MODE = {label.casefold(): key for key, label in MODE_LABELS.items()}


def normalise_mode(value, fallback: str = AUTO) -> str:
    """One mode key, from whatever a control, a preference or a paste supplied.

    Liberal on the way in and strict on the way out. A stored preference from a
    build that offered a mode this one does not, a hand-edited API argument, a
    label where a key was expected -- all of them land on ``fallback`` rather
    than reaching the state machine as a string nothing matches.
    """
    text = str(value or "").strip().casefold()
    if text in MODES:
        return text
    return _LABEL_TO_MODE.get(text, fallback)


def mode_from_label(label, fallback: str = AUTO) -> str:
    """The mode a recorded infotext label names."""
    return normalise_mode(label, fallback)


def label_for(mode) -> str:
    """The human-readable name of ``mode``."""
    return MODE_LABELS.get(normalise_mode(mode, ""), "")


def available_modes(has_source: bool) -> tuple[str, ...]:
    """Which modes are selectable with, or without, a usable source image.

    §4, and the whole of it: Auto and Regional Generate always; the three
    image-required modes only when there is genuinely an image behind them. The
    panel keeps the rest *visible* and disabled, with :data:`NO_SOURCE_REASON`
    beside them, so that a mode does not appear to have been removed by an
    upgrade when what actually happened is that a gallery is empty.
    """
    if has_source:
        return MODES
    return tuple(mode for mode in MODES if mode not in IMAGE_REQUIRED_MODES)


def is_available(mode, has_source: bool) -> bool:
    return normalise_mode(mode, "") in available_modes(bool(has_source))


def is_implemented(mode) -> bool:
    """Whether ``mode`` has sampling behaviour behind it in this build."""
    return normalise_mode(mode, "") in IMPLEMENTED_MODES


class ModeNotImplemented(ValueError):
    """A mode this build offers and cannot yet run.

    Separate from :class:`ModeUnavailable` because the remedies are different --
    one is answered by adding an image, the other by a later build -- and a
    message that conflated them would send somebody to their ImageStitch gallery
    to fix something that is not there.
    """

    def __init__(self, mode: str):
        self.mode = mode
        super().__init__(
            f"Spatial mode {label_for(mode)!r} is not implemented yet. Choose "
            f"{MODE_LABELS[REGIONAL_GENERATE]} or {MODE_LABELS[REFERENCE_REGIONS]} "
            f"(or Auto, which picks between them), or turn Spatial Layout off.")


class ModeUnavailable(ValueError):
    """An explicitly chosen image-required mode, with no image behind it.

    Raised rather than resolved. §10 draws the line here and it is the sharpest
    line in the design: **Auto is allowed to adapt and an explicit choice is
    not.** Somebody who picked Strict Regional Edit picked it because they want
    their source preserved, and turning that into a fresh txt2img because a
    gallery emptied between the click and the press is not a fallback, it is a
    different picture with the same button.
    """

    def __init__(self, mode: str):
        self.mode = mode
        super().__init__(f"Spatial mode {label_for(mode)!r} requires an enabled "
                         f"ImageStitch image.")


def resolve(requested, has_source: bool) -> str:
    """The mode this generation will actually run in.

    Deterministic, and short enough to state completely::

        Auto + source      -> Reference + Regions
        Auto + no source   -> Regional Generate
        anything else      -> itself, or ModeUnavailable

    Auto never selects Regional Img2Img or Strict Regional Edit (§5). Both of
    those change how strongly the source survives, which is a decision about the
    picture rather than about what is available, and a resolver that made it
    would be choosing how much of somebody's image to keep on their behalf.
    """
    mode = normalise_mode(requested, AUTO)
    if mode == AUTO:
        return REFERENCE_REGIONS if has_source else REGIONAL_GENERATE
    if mode in IMAGE_REQUIRED_MODES and not has_source:
        raise ModeUnavailable(mode)
    return mode


def uses_source(mode) -> bool:
    """Whether ``mode`` reads the ImageStitch images at all.

    False for Regional Generate even with a full gallery, which is §6: an unused
    image elsewhere in the UI must not silently change what a mode means. The
    user chose "start from noise"; the gallery is not a vote.
    """
    return normalise_mode(mode, "") in IMAGE_REQUIRED_MODES


def starts_from_noise(mode) -> bool:
    """Whether the output latent begins as noise rather than as an encoded image.

    True for Regional Generate and for Reference + Regions -- the second one is
    the distinction §22 exists to protect. Reference conditioning is not an init
    canvas, and a mode that quietly started from the reference would be
    "preserve my image except in the box" wearing the wrong name.
    """
    return normalise_mode(mode, "") in (REGIONAL_GENERATE, REFERENCE_REGIONS)


# --------------------------------------------------------------------------- #
# One region, one source, one request
# --------------------------------------------------------------------------- #

DEFAULT_STRENGTH = 1.0
MIN_STRENGTH = 0.0
MAX_STRENGTH = 2.0
"""§30. 0 contributes nothing, 1 is normal, above 1 is a stronger bias and still
not a hard mask. Clamped rather than refused, like every other coordinate in
this feature."""


def clamp_strength(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DEFAULT_STRENGTH
    if number != number:  # NaN, which compares false against every bound below.
        return DEFAULT_STRENGTH
    return max(MIN_STRENGTH, min(MAX_STRENGTH, number))


@dataclass(frozen=True)
class SpatialRegion:
    """One region, in the form a sampling backend can read.

    A narrower thing than :class:`prompt_master.krea.spatial.Region`, on purpose.
    That one carries everything the editor and the Krea prompt composer need --
    raw text, literal payloads, the object/text distinction. This one carries
    what a regional-conditioning pass needs and nothing else, so a backend cannot
    accidentally depend on a field that only means something to the other one.
    """

    identifier: str
    bbox_norm: tuple[int, int, int, int]
    """``(x0, y0, x1, y1)``, integers 0..1000, already ordered and clamped."""

    prompt: str = ""
    strength: float = DEFAULT_STRENGTH
    kind: str = document.OBJECT
    framing: str = ""
    angle: str = ""
    z: int = 0
    index: int = 0

    @property
    def fractions(self) -> tuple[float, float, float, float]:
        """The box as fractions of the frame, 0.0..1.0.

        The one conversion that is safe to do without knowing anything about the
        model: it divides by a constant. Everything past this point needs a
        tensor shape, which is why it is a separate step.
        """
        x0, y0, x1, y1 = self.bbox_norm
        return (x0 / float(SCALE), y0 / float(SCALE),
                x1 / float(SCALE), y1 / float(SCALE))

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.fractions
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def qualified_prompt(self) -> str:
        """The region's prompt with its framing and angle appended, if any.

        §12: framing and angle are useful to Klein as concise natural-language
        qualifiers, and they must not change the bbox. So they join the *text*,
        after the user's own words, and the geometry is untouched -- which is the
        same order and the same phrases the Krea composer uses, taken from the
        same table rather than reworded here.

        Deliberately without the position and size hints the Krea path appends.
        Telling a regional condition in words that it is "positioned in the
        upper-left area" while also conditioning it on the upper-left area is
        saying the same thing twice, and the second copy is the one that leaks
        into the parts of the frame the box does not cover.
        """
        parts = [self.prompt.strip()]
        if self.kind == document.OBJECT:
            parts.append(document.FRAMINGS.get(self.framing, ""))
            parts.append(document.ANGLES.get(self.angle, ""))
        return ", ".join(part for part in parts if part)

    def grid(self, width: int, height: int) -> tuple[int, int, int, int] | None:
        """This box on a concrete ``width`` x ``height`` conditioning grid.

        ``(gx0, gy0, gx1, gy1)``, half-open, or ``None`` when the region cannot
        survive the rounding. See :func:`to_grid` for why the rounding goes the
        way it does.
        """
        return to_grid(self.fractions, width, height)


@dataclass(frozen=True)
class SpatialSource:
    """The images this generation may use, and where they came from.

    Ordered, and never rearranged: §17 makes image #1 meaningful in three of the
    five modes, and a set that sorted itself would move somebody's init canvas.
    """

    images: tuple = ()
    max_dim: int = 0
    origin: str = "imagestitch"
    enabled: bool = True

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def usable(self) -> bool:
        """Whether this counts as "a source image exists" for §3.

        Enabled *and* non-empty. Both halves are load-bearing: a full gallery
        with the script switched off is not a source, and neither is an enabled
        script with nothing in it.
        """
        return bool(self.enabled and self.images)

    @property
    def primary(self):
        """Image #1 -- the init canvas in the source-preserving modes."""
        return self.images[0] if self.images else None

    @property
    def supplemental(self) -> tuple:
        """Images #2..#N, in order.

        §17 and §45: in the modes where image #1 becomes the init latent, this is
        what may additionally be registered as references -- and image #1 is
        deliberately not in it, because Klein inserts ``ini_latent`` ahead of
        ``ref_latents`` itself and registering it twice would hand the model the
        same picture as reference 1 and reference 2.
        """
        return tuple(self.images[1:])

    def references_for(self, mode) -> tuple:
        """The images to register as Klein references under ``mode``.

        The whole of §17's ordering table, in one place:

        =====================  ==========================================
        Regional Generate      nothing at all, however full the gallery is
        Reference + Regions    every image, in order
        Regional Img2Img       images 2..N; image 1 is the init latent
        Strict Regional Edit   images 2..N; image 1 is the source canvas
        =====================  ==========================================
        """
        resolved = normalise_mode(mode, "")
        if not self.usable or not uses_source(resolved):
            return ()
        if resolved == REFERENCE_REGIONS:
            return tuple(self.images)
        return self.supplemental


NO_SOURCE = SpatialSource(images=(), enabled=False)
"""What "there is no ImageStitch image behind this generation" looks like.

A source object rather than ``None`` so that every caller can ask
``source.usable`` without first asking whether there is a source to ask.
"""


@dataclass(frozen=True)
class SpatialRequest:
    """One press of Generate, as far as Spatial Layout is concerned.

    Assembled once, by the host glue, out of the panel's controls and the live
    ImageStitch arguments. Every backend reads it and none of them re-derives any
    of it, which is what makes "the UI is advisory and the runtime is
    authoritative" (§9, §40) a structural property rather than a rule somebody
    has to remember.
    """

    enabled: bool = False
    requested_mode: str = AUTO
    resolved_mode: str = ""
    compose_mode: str = ""
    """Whether a language model reconciles the global prompt with the layout.

    ``smart`` or ``direct``, and it is a different question from
    :attr:`resolved_mode` in the way that matters: the mode decides what the
    *source image* is and the compose mode decides what the *text* is. Both
    backends ask it, which is why it lives on the shared request rather than
    inside either of them -- Krea reconciles the scene it is about to build a
    structured prompt from, and Klein reconciles the prompt the model reads
    directly.
    """
    canvas_width: int = 0
    canvas_height: int = 0
    regions: tuple[SpatialRegion, ...] = ()
    source: SpatialSource | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    """Every reason a region was skipped or a field dropped, from the parse.

    Carried rather than logged here so the caller decides where it goes: a hook
    puts it on the result, a test asserts on it, the panel shows it under the
    canvas.
    """

    unreadable: bool = False
    """True when the document could not be read at all -- bad JSON, or a version
    this build does not know. Distinct from "no regions drawn"."""

    @property
    def has_source(self) -> bool:
        return bool(self.source is not None and self.source.usable)

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def references(self) -> tuple:
        """The images this generation will register, in order. Possibly none."""
        if self.source is None:
            return ()
        return self.source.references_for(self.resolved_mode or self.requested_mode)

    @property
    def uses_source(self) -> bool:
        return uses_source(self.resolved_mode or self.requested_mode)

    def summary(self) -> str:
        """The one log line §42 asks for, minus the backend name.

        The caller appends the backend, because the backend is chosen after this
        object exists and naming it here would mean either building the request
        twice or leaving a hole in it.
        """
        source = "none"
        if self.source is not None and self.source.images:
            source = str(self.source.image_count)
        return (f"requested={label_for(self.requested_mode)} "
                f"resolved={label_for(self.resolved_mode)} "
                f"regions={self.region_count} ImageStitch={source}")


# --------------------------------------------------------------------------- #
# From the editor's document to the request
# --------------------------------------------------------------------------- #


def regions_from(layout) -> tuple[SpatialRegion, ...]:
    """Every usable region of a parsed :class:`Layout`, in prompt order.

    ``layout.ordered`` rather than ``layout.regions``: z then draw order, the
    same sequence the Krea composer uses. §13 is explicit that z does not create
    occlusion in a diffusion model and equally explicit that it stays meaningful
    to the editor, so it decides the *order* here and nothing else -- an
    overlapping pair contributes both conditions either way.
    """
    found = []
    for region in getattr(layout, "ordered", ()) or ():
        bbox = document.normalise_bbox(getattr(region, "bbox", None))
        if bbox is None:
            # Already refused once, by the document parser. Asked again because
            # a Layout can be built in a test without going through parse(), and
            # a zero-area box reaching a grid mapper is an empty tensor slice.
            continue
        found.append(SpatialRegion(
            identifier=str(getattr(region, "identifier", "") or ""),
            bbox_norm=tuple(bbox),
            prompt=str(getattr(region, "prompt", "") or ""),
            strength=clamp_strength(getattr(region, "strength", DEFAULT_STRENGTH)),
            kind=str(getattr(region, "kind", document.OBJECT) or document.OBJECT),
            framing=str(getattr(region, "framing", "") or ""),
            angle=str(getattr(region, "angle", "") or ""),
            z=int(getattr(region, "z", 0) or 0),
            index=int(getattr(region, "index", 0) or 0),
        ))
    return tuple(found)


def request_from(serialized, *, enabled: bool, requested_mode=AUTO,
                 source: SpatialSource | None = None, width: int = 0,
                 height: int = 0, compose_mode: str = "") -> SpatialRequest:
    """One serialized layout and one panel state, as a :class:`SpatialRequest`.

    Never raises, and never resolves. Parsing is separate from deciding on
    purpose: this can run on a UI repaint, where "what would Auto do" is a
    preview, and :func:`resolved_request` runs at generation time, where it is a
    commitment. ``resolved_mode`` is therefore empty here until somebody settles
    it.

    ``width`` and ``height`` fill in a canvas the document does not carry. They
    never move a coordinate -- the boxes are normalized, and a resolution change
    is not a reprojection (§15).
    """
    layout = document.parse(serialized, width=width, height=height)
    return SpatialRequest(
        enabled=bool(enabled),
        requested_mode=normalise_mode(requested_mode, AUTO),
        resolved_mode="",
        canvas_width=int(getattr(layout, "width", 0) or 0),
        canvas_height=int(getattr(layout, "height", 0) or 0),
        compose_mode=str(compose_mode or "").strip().casefold(),
        regions=regions_from(layout),
        source=source if source is not None else NO_SOURCE,
        notes=tuple(getattr(layout, "notes", ()) or ()),
        unreadable=bool(getattr(layout, "unreadable", False)),
    )


def resolved_request(request: SpatialRequest) -> SpatialRequest:
    """``request`` with :attr:`SpatialRequest.resolved_mode` settled.

    Raises :class:`ModeUnavailable` for an explicitly chosen image-required mode
    whose source has gone. That exception is the §10 behaviour in full: the
    caller turns it into a generation error *before sampling*, rather than
    running the pass with stale references or quietly making a different picture.
    """
    from dataclasses import replace

    return replace(request, resolved_mode=resolve(request.requested_mode,
                                                  request.has_source))


# --------------------------------------------------------------------------- #
# Normalized boxes on a concrete grid
# --------------------------------------------------------------------------- #


def to_grid(fractions, width: int, height: int) -> tuple[int, int, int, int] | None:
    """A fractional box on a ``width`` x ``height`` grid, half-open, or ``None``.

    §29's five steps, and the rounding in it is not arbitrary. The near corner
    floors and the far corner ceils, so a box always covers *at least* the cells
    it visually touches: a region that rounded inwards on both sides would lose
    a whole row of tokens at small latent sizes, and the smallest boxes -- the
    ones most in need of every cell they can get -- would lose the largest share.

    The grid is whatever the caller measured on the tensor about to be sampled.
    It is emphatically not derived from the canvas dimensions stored in the
    layout, which are editor metadata from whenever the boxes were drawn, and it
    is not derived from a fixed VAE ratio either: §15 is blunt about not
    hard-coding ComfyUI's ``/8``, and a Flux.2 build that changed its patchify
    factor would silently move every box of every saved layout.

    ``None`` when the region cannot survive -- a zero-sized grid, or a box so
    thin that it lands between two cells. A caller that gets ``None`` drops the
    region and says so; it does not widen the box to make one fit, because a
    coordinate this module invented is a coordinate the user did not draw.
    """
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None

    try:
        x0, y0, x1, y1 = (float(value) for value in tuple(fractions)[:4])
    except (TypeError, ValueError):
        return None

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    gx0 = max(0, min(width - 1, floor(x0 * width)))
    gy0 = max(0, min(height - 1, floor(y0 * height)))
    gx1 = max(gx0 + 1, min(width, ceil(x1 * width)))
    gy1 = max(gy0 + 1, min(height, ceil(y1 * height)))

    if gx1 <= gx0 or gy1 <= gy0:
        return None
    return gx0, gy0, gx1, gy1


def compile_grid(regions, width: int, height: int) -> tuple[tuple, tuple[str, ...]]:
    """Every region mapped onto one grid: ``(pairs, notes)``.

    ``pairs`` is ``((region, (gx0, gy0, gx1, gy1)), ...)`` in the order given,
    with anything that could not survive the rounding left out and named in
    ``notes``. Named rather than dropped quietly: a box that vanished because the
    latent was small is a box the user drew and will look for in the result.
    """
    pairs = []
    notes = []
    for region in regions or ():
        cell = to_grid(region.fractions, width, height)
        if cell is None:
            notes.append(f"Region {region.identifier or '?'} is too small to cover a "
                         f"single cell of the {width}x{height} conditioning grid and "
                         f"was left out.")
            continue
        pairs.append((region, cell))
    return tuple(pairs), tuple(notes)
