"""The Spatial Composer: the second pass, and the four fields it may not touch.

Pass 1 writes a rich scene from a short idea and knows nothing about the boxes.
Pass 2 is shown both and may rewrite *two strings*: the global scene, and the
background. That is the whole of its authority, and it is enforced by reading
only those two keys out of whatever comes back -- not by asking the model
nicely, and not by comparing what it returned against what it was given.

Why there is a second pass at all
---------------------------------
Because the first one writes sentences like *"the woman stands prominently in
the centre of the frame"*, and the user has just put her face in the upper-left.
Merging those two directly sends the image model competing instructions about
the same subject, and what comes back is usually neither: a centred figure with
a second face in the corner, or a composition that splits the difference. The
Composer's job is to keep the first pass's aesthetic direction and remove the
half of it that argues with the layout.

Why it cannot be the thing that builds the prompt
-------------------------------------------------
:mod:`prompt_master.krea.spatial` builds the prompt, from validated user state,
in code. Every failure of "ask the model for the finished JSON" is silent and
several are unrecoverable -- a coordinate that drifts by forty units, two
regions merged into one, a user's own wording paraphrased, an elements array
that parses but is in a different order than last time. The model here is a
copy-editor with two paragraphs in front of it and no access to the filing
cabinet.

Its own instruction, and what that costs
----------------------------------------
The system message here is *not* Krea's expansion instruction. It cannot be:
that file's rule 6 says "no bullets, JSON, or markdown" and its rule 1 says
expand, and this pass has to return two labelled fields and must not expand
anything. Appending an addendum that told the model to disregard two of the
rules above it would be this extension arguing with a vendored file inside the
model's context window.

The cost of that choice is a real one and is worth writing down, because it is
paid in seconds on a processor-only placement. llama.cpp resumes a prompt at its
common prefix with the *previous* one, so a pass with a different system message
prefills its own instruction, and the Krea instruction that the next roll wants
back in the cache has to be read again -- roughly 460 tokens each way, which is
about fifteen seconds each at thirty tokens a second and about half a second
each on a card. So this instruction is kept deliberately short, it runs only
when the user asked for Smart mode *and* drew at least one region, and Direct
mode -- which makes no second request at all -- is one radio button away and is
the A/B control the design intent asks for.
"""

from __future__ import annotations

import json
import re

INSTRUCTION_VERSION = 1
"""Bumped when the wording below changes in a way that changes what comes back.

Recorded beside the image, because "the Composer behaved differently" is
otherwise indistinguishable from "the model was different that day".
"""

MAX_TOKENS = 700
"""Room for two paragraphs and the punctuation around them.

Not 1024. This pass shortens; a Composer that needs a thousand tokens to say
what pass 1 said in six hundred has misunderstood the job, and the ceiling is
one more thing making that visible rather than expensive.
"""

TEMPERATURE = 0.3
TOP_P = 0.9
"""Cool, and deliberately not on the Creativity curve.

Creativity is about how much art direction the *first* pass is given. This pass
is an edit: the same enhanced scene and the same layout should reconcile the
same way twice, and a sampler exploring alternatives here would make the Smart
against Direct comparison a comparison of two draws.
"""

SEED_PURPOSE = "spatial"
"""What the Composer's seed is derived from the Creative seed *for*.

Derived rather than drawn, for the reason every other seed in this package is:
one recorded number reproduces the whole roll, including this pass.
"""

SYSTEM_PROMPT = """\
You are a composition editor for text-to-image prompts. You are given a written
scene description and a spatial layout that a person has drawn by hand.

The layout is final. Its boxes, their coordinates, the words in them, the
visible text, the object/text types, the framing choices and the camera angles
have all been decided by the user and are being placed into the final prompt by
software, not by you. You never see them again after this turn and nothing you
write can change them.

Your only job is to rewrite the scene description so that it agrees with that
layout.

Do this:
- Remove or soften statements about where subjects sit in the frame when they
  conflict with the layout, including "centered", "in the middle of the frame",
  "on the left", "dominates the composition" and any similar phrasing.
- Reduce detailed description of subjects that the layout already places. Name
  them briefly if the scene needs them; do not repeat their full description.
- Keep the setting, the medium, the lighting, the mood, the atmosphere, the
  materials, the colour treatment and the overall style exactly as written.
- Shorten prose that has become redundant. Do not add new subjects, objects,
  props or details that were not already there.

Never do this:
- Do not describe or restate the boxes, their coordinates or their numbers.
- Do not write any visible text, caption or typography.
- Do not output a layout, an elements list, coordinates or any structure other
  than the two fields asked for below.

Reply with one JSON object and nothing else, in exactly this form:

{"scene": "the rewritten scene description", "background": "the environment only"}

"scene" is required. "background" describes the setting behind the placed
subjects and may be an empty string if the scene already covers it. Both are
plain prose. No other keys are read."""

USER_HEADING = "enhanced_scene:"
SOURCE_HEADING = "source_prompt:"
LAYOUT_HEADING = "spatial_layout:"


def region_line(position: int, region) -> str:
    """One region, as the context line pass 2 is allowed to see.

    Deliberately readable rather than JSON. The model is being shown these so it
    knows what the layout already covers; a JSON array invites it to answer with
    one, and the one thing this pass must never do is emit structure.

    The coarse position hint is included and the raw coordinates are not. What
    the Composer needs to know is that a face is in the upper-left, so that it
    can stop the scene saying "centred"; the numbers are the compositor's and
    telling the model about them only gives it something to helpfully adjust.
    """
    from . import spatial

    where = spatial.position_hint(region.bbox)
    size = spatial.size_hint(region.bbox)
    if region.kind == spatial.TEXT:
        body = f'visible text "{region.text}"'
        if region.prompt:
            body = f"{body} — {region.prompt}"
    else:
        body = region.prompt
        extra = [spatial.FRAMINGS.get(region.framing, ""),
                 spatial.ANGLES.get(region.angle, "")]
        extra = [part for part in extra if part]
        if extra:
            body = f"{body} ({'; '.join(extra)})"
    return f"{position}. [{region.identifier}] {where}, {size} — {body}"


def user_content(source: str, scene: str, layout, ratio: str = "") -> str:
    """The Composer's turn: what was asked for, what was written, what was drawn.

    The order is the order of authority, the same as
    :func:`prompt_master.krea.enhancer.user_content` uses: the user's own idea,
    then the pass this one is editing, then the material it has to agree with.
    """
    blocks = []
    source = str(source or "").strip()
    if source:
        blocks.append(f"{SOURCE_HEADING}\n{source}")
    blocks.append(f"{USER_HEADING}\n{str(scene or '').strip()}")

    lines = [LAYOUT_HEADING]
    if ratio:
        lines.append(f"frame aspect ratio: {ratio}")
    lines.extend(region_line(position, region)
                 for position, region in enumerate(layout.ordered, start=1))
    blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def messages(source: str, scene: str, layout, ratio: str = "") -> list[dict[str, str]]:
    """The whole Composer request. One system message, one user turn, no history."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content(source, scene, layout, ratio)}]


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #

_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCED = re.compile(r"```[a-zA-Z]*\s*(?P<body>\{.*?\})\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

MAX_SCENE = 4000
"""Longest scene this pass may return, in characters.

A ceiling and not a target. The pass is defined as one that shortens, so a reply
several times the length of its input has done something other than what it was
asked; truncating it would produce a sentence cut in half, so it is refused and
Direct answers instead.
"""


def _object_text(text: str) -> str | None:
    """The JSON object in a reply, fenced or bare, or ``None`` if there is none.

    The fence is looked for first. A model that wrapped its answer and then
    added a sentence after it would otherwise have the sentence swept into the
    bare-object match, which is a parse failure over formatting the instruction
    already said not to use and the model used anyway.
    """
    fenced = _FENCED.search(text)
    if fenced is not None:
        return fenced.group("body")
    bare = _OBJECT.search(text)
    return bare.group(0) if bare is not None else None


class Refused(ValueError):
    """The reply was not a usable ``{scene, background}``. Direct merge answers."""


def parse(reply: str) -> tuple[str, str]:
    """``(scene, background)`` out of one Composer reply, or :class:`Refused`.

    Lenient about the wrapping and strict about the contents. A local model
    wraps JSON in a fence, prefixes it with "Here you go:", or thinks out loud
    first, and none of those is the model failing to do the job -- so the object
    is found rather than demanded. What is *not* forgiven is the object not
    carrying a usable ``scene``, because at that point there is nothing to use
    and Direct merge is a better answer than a guess.

    Extra keys are ignored rather than refused, and that is the isolation
    property stated as code: only these two names are ever read, so a reply
    carrying an ``elements`` array, a ``bbox`` or a rewritten region prompt
    cannot reach the prompt whatever it contains.
    """
    text = _THINKING.sub("", str(reply or "")).strip()
    if not text:
        raise Refused("the Spatial Composer returned nothing")

    body = _object_text(text)
    if body is None:
        raise Refused("the Spatial Composer did not return a JSON object")
    try:
        payload = json.loads(body)
    except Exception as exc:
        raise Refused(f"the Spatial Composer's reply was not valid JSON ({exc})")
    if not isinstance(payload, dict):
        raise Refused("the Spatial Composer returned something other than an object")

    scene = payload.get("scene")
    scene = str(scene).strip() if isinstance(scene, str) else ""
    if not scene:
        raise Refused("the Spatial Composer returned no scene")
    if len(scene) > MAX_SCENE:
        raise Refused(f"the Spatial Composer returned {len(scene):,} characters of "
                      f"scene, which is longer than this pass is allowed to write")

    background = payload.get("background")
    background = str(background).strip() if isinstance(background, str) else ""
    return scene, background


def overreached(reply: str) -> tuple[str, ...]:
    """Which forbidden keys the reply carried, for the log and the A/B record.

    Reading this changes nothing -- :func:`parse` has already ignored them -- and
    that is the point. A Composer that keeps trying to write the elements array
    is a Composer whose instruction needs work, and the only way anybody finds
    that out is if the attempt is counted somewhere.
    """
    body = _object_text(_THINKING.sub("", str(reply or "")).strip())
    if body is None:
        return ()
    try:
        payload = json.loads(body)
    except Exception:
        return ()
    if not isinstance(payload, dict):
        return ()
    return tuple(key for key in ("elements", "regions", "bbox", "compositional_deconstruction")
                 if key in payload)
