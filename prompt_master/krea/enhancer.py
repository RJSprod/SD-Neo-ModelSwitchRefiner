"""How a Krea 2 prompt is asked for, over the server this application runs.

The base instruction is Krea's, verbatim, in ``expansion.txt`` beside this file
-- the file Krea's own prompting guide tells you to use as a system prompt when
you want an LLM to write longer prompts. It is read, never rewritten, and
:func:`system_prompt` never edits a word of it: what this extension has to say
is *appended*, under a heading that says whose it is, so that the two halves of
the system message can always be told apart by anybody reading a transcript.

That separation is the whole design of this module, and there are four things
on this side of the line worth stating plainly, because none of them is Krea's:

*Reference images are numbered by what the user can see.* The first slot is
Image 1. Krea documents no such convention -- it is this extension's, and
:data:`REFERENCE_ADDENDUM` introduces it to the model as the user's numbering
rather than as Krea syntax. It exists because "the woman from image 2" is how
people actually describe an edit, and because a prompt that loses which
reference was which has lost the entire request.

*The images are described, not shown, to the writer.* Each reference is
captioned on its own by the vision model, and the writer -- the pass that
produces the finished prompt -- is a text-only request over those captions.
This is the same caption-first shape ``prompt_master.minimax`` uses, for the
same two reasons: it needs no multi-image transport, and it makes each image's
description a thing that can be shown, checked and kept.

*The captioner is told to describe and nothing else.* No naming of real people,
no inferring what is out of frame, no deciding how the picture should be
edited. A captioner that starts editing produces a caption the writer then
writes a prompt about, and the user's actual instruction is what loses.

*The user turn is labelled.* ``user_prompt:``, then ``creative_direction:`` when
Creative Mode assembled one, then ``reference_images:`` when there are
references, with one numbered line each. The writer is never handed a row of
unlabelled paragraphs and asked to work out which is which -- that is the
failure §4 spends a page forbidding, arriving through the back door.

*The creative direction is the caller's, and is already written.* Creative Mode
assembles it in :mod:`prompt_master.krea.director`, out of a vendored vocabulary,
with no model involved; this module only places it in the turn under its own
label. That is why :func:`messages` takes a finished block of text rather than a
recipe object -- there is nothing left here to decide, and a function that could
decide something would be a second prompt writer sitting between the Director
and the model.

Nothing here writes prompt text on the user's behalf. :func:`clean` takes
formatting off the end result and is deliberately incapable of rewriting it.
"""

from __future__ import annotations

import re
from pathlib import Path

MAX_REFERENCES = 4
"""How many references one prompt may carry.

An LLM Studio limit -- four captioning passes before the writer even starts,
four pictures' worth of room in the window -- and not a claim about any Krea
backend (§4). Raising it costs nothing here and costs a caption pass per image
at run time.
"""

# --------------------------------------------------------------------------- #
# Krea's own instruction
# --------------------------------------------------------------------------- #

_EXPANSION = Path(__file__).with_name("expansion.txt")


def base_instruction() -> str:
    """The contents of ``expansion.txt``: Krea's prompt expansion instruction.

    Read from the file rather than pasted into a string literal, so that
    re-vendoring the upstream file is a copy and a digest in
    ``UPSTREAM_SOURCE.txt`` rather than an edit to a Python module -- and so
    that a diff of this package can never make it look as though the
    instruction had been quietly reworded.
    """
    return _EXPANSION.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# This extension's addendum (design intent §6)
# --------------------------------------------------------------------------- #

REFERENCE_HEADING = "Reference image handling (additional instructions):"
"""What separates upstream's text from ours in the assembled system message."""

REFERENCE_ADDENDUM = """\
The user has attached reference images. They are numbered by the order the user
sees them in: the first is Image 1, the second is Image 2, and so on. You are
given one short factual description per image, under those same numbers.

- Never swap the image numbers. Image 1 is the one described as Image 1.
- The roles the user gives the images are authoritative. A reference may be the
  source image to edit, an identity reference, a scene reference, a style
  reference, a pose reference, a lighting reference, a product reference, or
  anything else the user says it is.
- Do not infer a role the user has already stated, and do not invent
  relationships between the images that the user did not describe.
- When the user asks for something to be preserved, say so explicitly in the
  prompt wherever leaving it out could let it change.
- A style reference supplies visual treatment only. Do not carry its subject,
  layout, text or background into the image unless the user asked for that.
- An identity reference supplies the person only. Do not carry its clothing,
  pose, background or composition across unless the user asked for that.
- Where the request depends on telling the references apart, keep the words
  "Image 1", "Image 2" and so on in the finished prompt. Removing them to read
  more smoothly, at the cost of making the relationship ambiguous, is wrong.
- Write one final natural-language Krea prompt and nothing else."""
"""The reference rules, kept out of ``expansion.txt`` on purpose.

Not to make the model verbose -- the opposite. Every line is here to stop one
specific way a reference edit loses its meaning between the request and the
prompt: the numbers reversing, a role being guessed at, a style reference
dragging its own subject in behind it, or the numbers being polished out of a
sentence that needed them.
"""


def system_prompt(has_references: bool = False) -> str:
    """The system message: Krea's instruction, plus ours when it applies.

    Text-only generation is exactly what Krea documents, with nothing added --
    which is also why a text-only run works on any model that can hold a
    conversation, with no vision projector anywhere near it.
    """
    base = base_instruction()
    if not has_references:
        return base
    return f"{base}\n\n{REFERENCE_HEADING}\n{REFERENCE_ADDENDUM}"


# --------------------------------------------------------------------------- #
# The captioner
# --------------------------------------------------------------------------- #

CAPTION_INSTRUCTION = (
    "Describe this image factually in one concise paragraph. Cover the visible "
    "subject or subjects, their pose and action, the composition and framing, "
    "the setting, notable objects and clothing, the relevant colours, the "
    "lighting, and the visual medium or style where it is clearly visible. "
    "Where a person is shown, describe the visible features that identify them "
    "-- hair, build, face shape, distinguishing features -- as plainly as you "
    "would describe the rest of the picture. "
    "Do not name real people. Do not infer anything you cannot see, invent "
    "hidden details, or guess at relationships between people. Do not suggest "
    "how the image should be changed or edited. "
    "Output only the description, as one paragraph.")
"""What each reference is described with. This extension's own text.

Factual and non-creative by design (§5). The captioner is never shown the
user's instruction: it is describing a picture, not carrying out an edit, and a
captioner that knows what is about to be asked for starts answering that
instead.
"""

CAPTION_MAX_TOKENS = 256
CAPTION_TEMPERATURE = 0.0
CAPTION_TOP_P = 1.0
"""ModelSwitch defaults, not Krea values -- Krea documents no captioner.

Temperature 0 because a description of what is in a photograph has nothing for
a sampler to be creative about, and 256 tokens because a paragraph that covers
subject, pose, setting, clothing, colour and light is longer than the 128 a
one-line MiniMax caption needs.
"""

# --------------------------------------------------------------------------- #
# The writer
# --------------------------------------------------------------------------- #

TEMPERATURE = 0.6
TOP_P = 0.9
MAX_TOKENS = 1024
"""Also this extension's, and also not Krea's -- the guide sets no sampler.

Low enough that "lightly polish and finalize" (upstream rule 7) is what a
detailed prompt actually gets, and 1024 tokens is comfortably more than the one
paragraph the instruction asks for, with room for a model that thinks out loud
before it writes -- :func:`clean` takes that off.

These two numbers are the *legacy* sampling and are kept here under their
original names because that is what they are: what this module asked for before
Creativity existed. :mod:`prompt_master.krea.variation` is what a run actually
samples at now, and its Creativity-1 row is asserted equal to these.
"""

BUTTON_LABEL = "Generate Krea Prompt"

_LABELS = {0: "Writing the Krea prompt",
           1: "Writing the Krea prompt from the reference"}


def label(reference_count: int = 0) -> str:
    """What the run is doing, for the status line."""
    count = max(int(reference_count or 0), 0)
    return _LABELS.get(count, f"Writing the Krea prompt from {count} references")


def caption_label(position: int, total: int) -> str:
    """What the captioner is doing, by the number the user sees on the slot."""
    return f"Describing image {position} of {total}…"


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


def caption_messages(image_data_url: str) -> list[dict[str, object]]:
    """The captioner's turn: the picture, then what to say about it."""
    return [{"role": "user",
             "content": [{"type": "image_url", "image_url": {"url": image_data_url}},
                         {"type": "text", "text": CAPTION_INSTRUCTION}]}]


def reference_block(captions) -> str:
    """The numbered descriptions, one line each, in the user's own order."""
    return "\n".join(f"Image {position}: {str(caption or '').strip()}"
                     for position, caption in enumerate(captions, start=1))


def user_content(prompt: str, captions=None, direction: str = "") -> str:
    """The user turn: what was asked for, how to treat it, and what the references are.

    The user's wording is passed through untouched -- stripped of surrounding
    whitespace and nothing else. Tidying it before the model sees it would mean
    the prompt was written about a paraphrase, which is the one edit this module
    is least entitled to make.

    ``direction`` is Creative Mode's finished block, already labelled and already
    carrying its own "the source prompt wins" rule. It goes *after* the request
    and *before* the references, which is the order of authority: what the user
    asked for, then how this extension suggests treating it, then the material.
    An empty string -- Creative Mode off, or Creativity 0 or 1, or every axis set
    to Natural -- adds nothing at all, so the turn is exactly the turn this
    function has always built.
    """
    text = str(prompt or "").strip()
    described = [caption for caption in (captions or [])]
    blocks = [f"user_prompt:\n{text}"]
    if str(direction or "").strip():
        blocks.append(str(direction).strip())
    if described:
        blocks.append(f"reference_images:\n{reference_block(described)}")
    return "\n\n".join(blocks)


def messages(prompt: str, captions=None, direction: str = "") -> list[dict[str, str]]:
    """The whole writing request: the instructions, and the request under them.

    The system half is untouched by Creative Mode. Every word of art direction
    travels in the user turn, where it reads as part of this request rather than
    as part of Krea's standing instructions -- which matters the moment somebody
    reads a transcript and asks which half of it Krea wrote.
    """
    described = [caption for caption in (captions or [])]
    return [{"role": "system", "content": system_prompt(bool(described))},
            {"role": "user", "content": user_content(prompt, described, direction)}]


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #

# A reply that leaked its own reasoning, or wrapped the answer in a fence the
# instruction forbids. Both put text in the box that must not be pasted into
# Krea. Upstream's instruction asks the model to think first and write the
# paragraph after, so the first of these is not a misbehaving model -- it is a
# model doing as it was told by a file this package must not edit.
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCED = re.compile(r"\A```[^\n]*\n(?P<body>.*?)\n?```\Z", re.DOTALL)


def clean(text: str) -> str:
    """The finished prompt, with what is not part of it taken off.

    Formatting only. This function removes a thinking block and a single
    enclosing code fence, and it cannot do anything else -- no re-wrapping, no
    rephrasing, no trimming the prompt to a length somebody picked. §7 draws
    that line and it is worth keeping sharp: a cleaner that improves prompts is
    a second prompt writer nobody can see, running after the one whose
    instructions are on disk.
    """
    text = _THINKING.sub("", str(text or "")).strip()
    fenced = _FENCED.match(text)
    return (fenced.group("body") if fenced else text).strip()
