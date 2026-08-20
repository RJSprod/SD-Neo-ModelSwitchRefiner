"""Creativity 0-10, as sampling settings and nothing else.

One control, one meaning, one table. Creativity 7 in LLM Studio and Creativity
7 in Krea Live are the same request because both of them come through
:func:`creativity_profile`, and there is deliberately no second mapping
anywhere for the two to drift apart.

What this module is not
-----------------------
It does not touch the instruction. ``expansion.txt`` is Krea's, vendored
verbatim, and a creativity control implemented by appending "be more
adventurous" to it would be this repository rewriting upstream's text at run
time -- exactly the thing ``prompt_master.krea.enhancer`` exists to make
impossible. So creativity is applied where it belongs, on the sampler, and the
semantic guardrail stays whatever Krea wrote.

It also does not generate candidates. There is no ``candidate_count`` field
below, no judge, and no second request: a higher value makes *one* completion
less tightly bound to its most probable wording, which is what a sampler is
for. A "best of N" design would multiply the LLM calls per prompt, and one call
per new prompt is the invariant the Live workflow is built on.

The two fixed points
--------------------
**1 is today.** Not "about today" -- the exact request the Krea writer has
always made: temperature 0.6, top_p 0.9, and none of the optional sampler
fields this module introduces. That is why :data:`_OPTIONAL` starts at 2 rather
than carrying "neutral" values for 1: a neutral value somebody guessed at is
still a field in the payload, still read by llama.cpp, and still capable of
changing what comes back. Omission is the only version of "unchanged" that can
be checked, and :func:`legacy_profile` is what the test checks it against.

**0 is the deterministic end.** Temperature 0 and top_p 1, with nothing added.
"Deterministic" describes the intent -- greedy decoding over a fixed prompt --
and not a promise of identical bytes across llama.cpp builds, model revisions,
offload splits or kernels, none of which this module can see.

Between them the table only ever climbs. Every value above 1 is more permissive
than 1 on temperature, and no value is ever more restrictive than the one below
it; a test asserts both, because a table maintained by hand is a table that can
be edited into a dip nobody notices until prompts at 6 come back flatter than
prompts at 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

MINIMUM = 0
MAXIMUM = 10
DEFAULT = 1
"""What an upgraded installation gets.

1 and not 5. Installing a feature is not asking for it: somebody who updates
the extension and presses Generate Krea Prompt the way they always have must
get the prompt they always got, and the way to guarantee that is for the
default to be the value that is defined as today's behaviour.
"""

LABEL = "Creativity"
HELP = "0 deterministic · 1 current behaviour · 10 maximum variation"
"""The one line of explanation the control carries.

It names the three positions that mean something specific. Everything between
them is a dial, and a user does not need to know that moving it changes
``top_k`` any more than they need to know what ``top_k`` is.
"""

_MEANINGS = {
    0: "most deterministic Krea expansion",
    1: "today's Krea 2 prompt-writing configuration",
    2: "slightly more creative than today",
    3: "more exploratory than today",
    4: "broader wording and detail choices",
    5: "clearly creative",
    6: "strong variation",
    7: "highly exploratory",
    8: "very high variation",
    9: "near-maximum variation while still prompt-faithful",
    10: "maximum supported creativity based on your prompt",
}


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

_SAMPLING = {
    0: (0.00, 1.00),
    1: (0.60, 0.90),
    2: (0.68, 0.92),
    3: (0.76, 0.94),
    4: (0.84, 0.95),
    5: (0.92, 0.96),
    6: (1.00, 0.97),
    7: (1.06, 0.98),
    8: (1.12, 0.99),
    9: (1.18, 1.00),
    10: (1.24, 1.00),
}
"""Temperature and top_p per value. Row 1 is the legacy anchor and is load-bearing."""

_OPTIONAL: dict[int, dict[str, object]] = {
    2: {"top_k": 50, "min_p": 0.05},
    3: {"top_k": 60, "min_p": 0.05},
    4: {"top_k": 70, "min_p": 0.04},
    5: {"top_k": 80, "min_p": 0.04},
    6: {"top_k": 100, "min_p": 0.03},
    7: {"top_k": 120, "min_p": 0.03},
    8: {"top_k": 140, "min_p": 0.02},
    9: {"top_k": 160, "min_p": 0.02},
    10: {"top_k": 200, "min_p": 0.01},
}
"""The extra sampler relaxation, from 2 upwards. 0 and 1 are absent on purpose.

Two fields and not six. ``top_k`` and ``min_p`` widen the pool a rising
temperature is then allowed to choose from, which is precisely what "more
creative" has to mean when the instruction is fixed -- without them a high
temperature spends most of its extra freedom on the tail llama.cpp's defaults
had already cut off, and the slider does less at 9 than the table implies.

Dynamic temperature and XTC are the obvious next candidates and are
deliberately *not* here. Both change decoding in ways that are worth having
only once somebody has measured them against real Krea prompts on real models,
and a sampler setting shipped on the strength of it sounding adventurous is a
setting that makes value 9 produce broken sentences. The client's whitelist
already accepts them (:data:`prompt_master.inference.llama_client
.SAMPLING_FIELDS`), so adding a row here is the whole change when that
measurement exists.
"""


@dataclass(frozen=True)
class SamplingProfile:
    """How one Krea writing request is sampled. Model settings, nothing else.

    Frozen because a profile is the answer to "what does Creativity 7 mean",
    and the answer is not something a caller between here and the request is
    entitled to adjust on the way past.

    There is no ``candidates``, ``candidate_count``, ``judge`` or
    ``novelty_score`` field, and their absence is the design rather than an
    omission: one prompt-authoring state produces one request, and a field that
    could ask for a second one would be the beginning of the "best of N" shape
    this feature exists without.
    """

    creativity: int
    temperature: float
    top_p: float
    optional_request_fields: dict = field(default_factory=dict)

    @property
    def meaning(self) -> str:
        """What this position on the slider is for, in a few words."""
        return _MEANINGS.get(self.creativity, "")

    @property
    def is_legacy(self) -> bool:
        """True for the one value defined as "unchanged from before this feature"."""
        return self.creativity == DEFAULT


def clamp(value) -> int:
    """``value`` as a Creativity integer, whatever arrived.

    A Gradio slider hands back a float, a preferences file hands back whatever
    was written into it last, and an infotext hands back a string. All three
    are the same number and none of them should be able to reach the sampler as
    ``6.0000000001`` or as a value off the end of the table -- so this is the
    single door, and every entry point below goes through it.
    """
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return DEFAULT
    return max(MINIMUM, min(MAXIMUM, number))


def creativity_profile(value) -> SamplingProfile:
    """The sampling settings for one Creativity position.

    The whole feature, in one function, used by both surfaces. The dictionary
    is copied out of :data:`_OPTIONAL` rather than shared with it, because a
    caller that mutated the profile it was handed would be editing the table
    for every later request in the process.
    """
    creativity = clamp(value)
    temperature, top_p = _SAMPLING[creativity]
    return SamplingProfile(creativity=creativity, temperature=temperature, top_p=top_p,
                           optional_request_fields=dict(_OPTIONAL.get(creativity, {})))


def legacy_profile() -> SamplingProfile:
    """The profile that has to match what the Krea writer did before this feature.

    Named separately so the compatibility guarantee has something to be
    asserted against by name -- ``creativity_profile(1) == legacy_profile()``
    is a sentence a test can say, and a table edit that broke it would fail on
    the row that states the promise rather than somewhere downstream.
    """
    return creativity_profile(DEFAULT)


def describe(value) -> str:
    """One short line naming the position, for a status line or a tooltip."""
    creativity = clamp(value)
    return f"{LABEL} {creativity} — {_MEANINGS.get(creativity, '')}"


MEANINGS = MappingProxyType(_MEANINGS)
"""Read-only view of the per-value wording, for panels that list the scale."""
