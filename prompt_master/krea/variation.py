"""Creativity 0-10, as the Krea writer's sampling settings and nothing else.

This module used to be the whole Creativity feature. It is now one third of it,
and the smaller third: :mod:`prompt_master.krea.director` decides *what* to ask
for and :mod:`prompt_master.krea.library` holds the vocabulary it asks from,
while this decides only how loosely the model is allowed to phrase the answer.

That split is the point of Creative Mode. Making a prompt writer more varied by
raising its temperature makes it phrase the same idea more loosely; making it
more varied by handing it a different art-direction brief makes it write about
something else. The second is what a creativity control should do, so the
sampling here stays modest -- it climbs from 0.60 to 0.96 across the whole
scale, where the previous design reached 1.24 and leaned on the sampler to
produce difference it could not produce.

The two fixed points
--------------------
**1 is today.** Not "about today" -- the exact request the Krea writer has always
made: temperature 0.6, top_p 0.9, and no other sampler field. The scale also
adds no art direction at 1 (the Director emits nothing below 2), so Creativity 1
is byte-identical to the request made before Creative Mode existed, in the
payload *and* in the messages.

**0 is the deterministic end.** Temperature 0 and top_p 1. "Deterministic"
describes the intent -- greedy decoding over a fixed prompt -- and not a promise
of identical bytes across llama.cpp builds, model revisions, offload splits or
kernels, none of which this module can see.

Why the table is here and also in the data package
--------------------------------------------------
``creativity/creativity_policy.json`` carries the same eleven rows, and a test
asserts the two agree. That is deliberate duplication, not a cache: Creativity 0
and 1 are compatibility guarantees, and a data package that failed to load, or
was edited by somebody tuning the creative vocabulary, must not be able to
change what the writer samples at those positions. The library can go missing
and the sampler still does what it promised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

MINIMUM = 0
MAXIMUM = 10
DEFAULT = 5
"""Where the slider sits on a fresh install, from the package's defaults.json.

5 and not 1, because Creative Mode is off on a fresh install: the slider only
does anything once somebody has deliberately turned the feature on, and
somebody who has just done that wants to see what it does. The compatibility
guarantee lives at 1 and is reachable by anyone who wants it; it no longer has
to be the default, because the default no longer applies to anybody who has not
opted in.
"""

LEGACY = 1
"""The position defined as "exactly what the Krea writer did before all this"."""

LABEL = "Creativity"
HELP = "0 deterministic · 1 legacy · 5 balanced · 10 maximum art direction"

_MEANINGS = {
    0: "most deterministic; no creative direction",
    1: "legacy Krea writer behaviour; no creative direction",
    2: "a light nudge on one axis",
    3: "a light nudge on one or two axes",
    4: "moderate direction on a few axes",
    5: "moderate direction, clearly authored",
    6: "moderate direction across most of the picture",
    7: "strong direction, and recent treatments avoided",
    8: "strong direction on most axes",
    9: "extreme direction on nearly every axis",
    10: "extreme direction on every eligible axis",
}


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

_SAMPLING = {
    0: (0.00, 1.00),
    1: (0.60, 0.90),
    2: (0.64, 0.91),
    3: (0.68, 0.92),
    4: (0.72, 0.93),
    5: (0.76, 0.94),
    6: (0.80, 0.95),
    7: (0.84, 0.96),
    8: (0.88, 0.97),
    9: (0.92, 0.98),
    10: (0.96, 0.98),
}
"""Temperature and top_p per position. Row 1 is the legacy anchor.

A gentle climb, and gentle on purpose. The Director is what makes Creativity 10
different from Creativity 2; the sampler's job is to let the model phrase one
brief in more than one way, not to shake the words loose. A temperature high
enough to do the second produces prompts with broken grammar in them, and the
symptom looks like the model being bad rather than like the slider being wrong.
"""


@dataclass(frozen=True)
class SamplingProfile:
    """How one Krea writing request is sampled. Model settings, nothing else.

    There is no ``candidates``, ``candidate_count``, ``judge`` or
    ``novelty_score`` field, and their absence is the design rather than an
    omission: one creative roll produces one request, and a field that could ask
    for a second one would be the beginning of the "best of N" shape this
    feature exists without.
    """

    creativity: int
    temperature: float
    top_p: float

    @property
    def meaning(self) -> str:
        return _MEANINGS.get(self.creativity, "")

    @property
    def is_legacy(self) -> bool:
        return self.creativity == LEGACY


def clamp(value) -> int:
    """``value`` as a Creativity integer, whatever arrived.

    A Gradio slider hands back a float, a preferences file hands back whatever
    was written into it last, and an infotext hands back a string. All three are
    the same number and none of them should be able to reach the sampler as
    ``6.0000000001`` or off the end of the table.
    """
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return DEFAULT
    return max(MINIMUM, min(MAXIMUM, number))


def resolve(value) -> int:
    """``value`` as a position, where ``None`` means the legacy one.

    The distinction :func:`clamp` cannot make. A number that will not parse is a
    corrupted preference and should land on the install default; ``None`` is a
    caller that never mentioned Creativity at all, and the only safe reading of
    that is "behave as you did before this existed". Every caller that has not
    been taught about the slider keeps legacy sampling and no creative
    direction, which is what makes Creative Mode an addition rather than a
    change.
    """
    return LEGACY if value is None else clamp(value)


def creativity_profile(value) -> SamplingProfile:
    """The sampling settings for one Creativity position.

    Used by both surfaces -- LLM Studio's Krea 2 workspace and txt2img's Creative
    Mode -- so that a position means one thing everywhere. There is deliberately
    no second mapping for the two to drift apart.
    """
    creativity = clamp(value)
    temperature, top_p = _SAMPLING[creativity]
    return SamplingProfile(creativity=creativity, temperature=temperature, top_p=top_p)


def legacy_profile() -> SamplingProfile:
    """The profile that has to match what the Krea writer did before Creative Mode.

    Named separately so the compatibility guarantee has something to be asserted
    against by name -- ``creativity_profile(1) == legacy_profile()`` is a
    sentence a test can say, and a table edit that broke it fails on the row
    that states the promise rather than somewhere downstream.
    """
    return creativity_profile(LEGACY)


def describe(value) -> str:
    """One short line naming the position, for a status line or a tooltip."""
    creativity = clamp(value)
    return f"{LABEL} {creativity} — {_MEANINGS.get(creativity, '')}"


MEANINGS = MappingProxyType(_MEANINGS)
"""Read-only view of the per-value wording, for panels that list the scale."""

SAMPLING = MappingProxyType(_SAMPLING)
"""Read-only view of the table, for the test that checks it against the package."""
