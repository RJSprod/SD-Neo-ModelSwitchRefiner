"""Literal commands: the text the language models are not allowed to see.

Creative Mode rewrites a short idea into a Krea 2 paragraph, and the Spatial
Composer rewrites that paragraph to agree with a layout. Both of them are the
right thing to do to *scene language* and the wrong thing to do to anything
else, and a prompt box holds both:

    [[<lora:krea2_edit:1>]]
    a stylish editorial portrait in an elegant restaurant
    -[[__lighting_wildcard__]]

The LoRA tag is not a sentence to improve. The wildcard is another extension's
syntax and means nothing until that extension expands it. "Her shirt from image
1" is an instruction about an ImageStitch reference that only the image model
can act on. None of the three is transformable text, and asking a writer to
"preserve" them is a probability rather than a guarantee -- one roll in twenty
comes back with the tag reworded, reordered or gone.

So they are not asked to preserve anything. Everything inside ``[[...]]`` is
lifted out of the text *before* the first request and put back *after* the last
one, and what the models see in between simply does not contain it:

    raw user text
        -> parse            (this module)
        -> clean text       -> Director -> Writer -> Composer -> compositor
        -> restore          (this module)
        -> Forge's own prompt processing

The payload is opaque
---------------------
This module does not know what a LoRA tag is, and it must not learn. A payload
is a run of characters between two delimiters; whether it is natural language, a
wildcard, a ``$style``, an extra network or a third-party macro is a question
for whoever owns that syntax, downstream, after restoration. The moment this
module starts telling them apart it acquires an opinion about every extension
the user has installed, and gets one of them wrong.

That is also why restoration happens where it does. A restored payload has to
reach Forge's ordinary prompt path -- the wildcard extension, the styles pass,
``modules.extra_networks`` -- exactly as it would if the user had typed it
straight into the box and never enabled Creative Mode. The feature's promise is
delivery and scope, not interpretation.

Direction, and why there are only two
-------------------------------------
``+[[x]]`` goes in front of the model-written body and ``-[[x]]`` goes after it;
a bare ``[[x]]`` is a prefix, because the common case is a LoRA tag or a
reference instruction that a person would have typed at the top. Two positions
and no more: an "insert at the third sentence" would need this module to know
what a sentence is in a paragraph that has not been written yet.

Within each group the user's own order is kept, which is the only ordering rule
here and is worth stating as one. ``-[[D]] +[[A]] [[B]] -[[E]] +[[C]] scene``
becomes ``A B C <scene> D E``: the classification splits the list in two and
nothing shuffles either half.

No imports
----------
Nothing in this file imports Forge, gradio, or any part of the LLM stack, and
nothing in it performs I/O. It is a string in, a dataclass out, so the whole
feature can be tested without a WebUI -- which matters more here than usual,
because every acceptance test worth writing is a statement about text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SYNTAX_VERSION = 1
"""The literal-command grammar this build reads and writes.

Recorded beside an image that used one. The syntax is small on purpose and will
grow -- escaping, and perhaps nesting, are the two obvious pressures -- and an
image made under version 1 should be readable as version 1 rather than
reinterpreted by whatever came later.
"""

OPEN = "[["
CLOSE = "]]"

PREFIX = "prefix"
SUFFIX = "suffix"
PLACEMENTS = (PREFIX, SUFFIX)

GLOBAL = "global"
REGION = "region"
SCOPES = (GLOBAL, REGION)

_SIGNS = {"+": PREFIX, "-": SUFFIX}
"""The optional character immediately before ``[[``, and what it means.

Immediately: no space. ``2 + 2 [[x]]`` is arithmetic followed by a prefix
command, and only ``+[[x]]`` is a signed one. That rule is what keeps the sign
from being something a user has to escape in ordinary prose.
"""


# --------------------------------------------------------------------------- #
# What one command is
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LiteralCommand:
    """One ``[[...]]`` lifted out of a prompt, and where it came from.

    Frozen, because a command is a record of something the user typed. Nothing
    downstream may edit a payload -- not to add a comma, not to strip a bracket
    -- and a mutable dataclass is an invitation to a helpful normalisation that
    changes what reaches the image model.
    """

    payload: str
    """The characters between the delimiters, with outer whitespace trimmed.

    Trimmed only on the outside, because assembly supplies its own separators
    and ``[[  foo, bar  ]]`` is a person lining up their brackets rather than
    asking for two leading spaces. Whitespace *inside* the payload is the
    user's and is never touched.
    """

    placement: str = PREFIX
    source_order: int = 0
    """Where this command was in the text it came from, counting from zero.

    Kept even after the commands are split into prefixes and suffixes, so that
    "the order the user wrote them in" survives a classification that reorders
    the list as a whole.
    """

    scope: str = GLOBAL
    region_id: str = ""
    """Which region this belongs to, when :attr:`scope` is :data:`REGION`.

    A command written inside a bounding box is content of *that element* and of
    nothing else. The identifier travels with it so that no later assembly step
    has to reconstruct the association from position in a list.
    """


@dataclass(frozen=True)
class LiteralParse:
    """One text, split into the half that may be rewritten and the half that may not."""

    clean_text: str
    """What the language models are given. The source with every command removed."""

    commands: tuple[LiteralCommand, ...] = ()
    warnings: tuple[str, ...] = ()
    """What was odd about the source, in the words a user should read.

    Never a reason to refuse: a malformed command is left in the text exactly as
    typed and reported, because the one thing worse than a bracket reaching the
    image model is a sentence silently disappearing on its way there.
    """

    def __bool__(self) -> bool:
        return bool(self.commands)

    @property
    def count(self) -> int:
        return len(self.commands)

    @property
    def prefix_commands(self) -> tuple[LiteralCommand, ...]:
        return tuple(command for command in self.commands
                     if command.placement == PREFIX)

    @property
    def suffix_commands(self) -> tuple[LiteralCommand, ...]:
        return tuple(command for command in self.commands
                     if command.placement == SUFFIX)

    @property
    def prefixes(self) -> tuple[str, ...]:
        """The prefix payloads, in source order."""
        return tuple(command.payload for command in self.prefix_commands)

    @property
    def suffixes(self) -> tuple[str, ...]:
        """The suffix payloads, in source order."""
        return tuple(command.payload for command in self.suffix_commands)


EMPTY = LiteralParse(clean_text="")
"""A parse of nothing, for callers with no text to hand."""


# --------------------------------------------------------------------------- #
# Reading them out
# --------------------------------------------------------------------------- #


def present(text) -> bool:
    """Whether ``text`` could possibly carry a command.

    One substring search, and the reason it exists: every generation this
    extension does not touch still runs through the hook that calls this, and
    "off is off" has to include "costs nothing". A prompt with no ``[[`` in it
    is returned untouched by :func:`parse` for the same reason -- byte-identical
    is a property worth keeping, not just a nice one.
    """
    return OPEN in str(text or "")


def parse(text, *, scope: str = GLOBAL, region_id: str = "") -> LiteralParse:
    """One prompt, as ``clean_text`` plus the commands that were lifted out of it.

    Never raises, and never deletes text it did not understand. The three ways
    this can be surprising all have the same answer -- say so, and leave the
    source alone:

    * **unterminated** -- ``[[missing close``. The command cannot be delimited,
      so there is no payload to lift; the text stays exactly as typed and the
      warning says which one.
    * **empty** -- ``[[]]``. Nothing to carry. Dropped from the clean text,
      because leaving the brackets in would send them to the writer, and
      reported so that a typo is visible rather than mysterious.
    * **nested** -- ``[[a [[b]] c]]``. The first ``]]`` closes the command, per
      the grammar; what follows is ordinary text. Version 1 has no escape
      syntax and inventing one here would be inventing it everywhere.

    ``scope`` and ``region_id`` are recorded on each command rather than
    inferred later. A region's commands and the global ones are the same type
    and travel through the same helpers, and the only thing keeping a region's
    reference instruction out of the global scene is that it says which region
    it belongs to.
    """
    source = str(text or "")
    if OPEN not in source:
        # Untouched, deliberately: no strip, no tidy, no normalisation. A prompt
        # with no commands in it must reach the writer as the same bytes it
        # would have reached it as before this feature existed.
        return LiteralParse(clean_text=source)

    kept: list[str] = []
    commands: list[LiteralCommand] = []
    warnings: list[str] = []
    position = 0

    while True:
        start = source.find(OPEN, position)
        if start < 0:
            kept.append(source[position:])
            break

        end = source.find(CLOSE, start + len(OPEN))
        if end < 0:
            # Everything from here on is ordinary text, including the brackets.
            kept.append(source[position:])
            warnings.append(
                f"A literal command was opened with {OPEN} and never closed with "
                f"{CLOSE}, so it was left in the prompt exactly as typed.")
            break

        opening, placement = start, PREFIX
        if start > 0 and source[start - 1] in _SIGNS:
            placement = _SIGNS[source[start - 1]]
            opening = start - 1

        kept.append(source[position:opening])
        payload = source[start + len(OPEN):end].strip()
        if payload:
            commands.append(LiteralCommand(payload=payload, placement=placement,
                                           source_order=len(commands), scope=scope,
                                           region_id=str(region_id or "")))
        else:
            warnings.append(f"An empty literal command ({OPEN}{CLOSE}) carried nothing "
                            f"and was ignored.")
        position = end + len(CLOSE)

    return LiteralParse(clean_text=_tidy("".join(kept)), commands=tuple(commands),
                        warnings=tuple(warnings))


_REPEATED_SEPARATORS = re.compile(r"\s*,(?:\s*,)+")
_RUNS_OF_SPACES = re.compile(r"[ \t]{2,}")
_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_LINES = re.compile(r"\n{3,}")


def _tidy(text: str) -> str:
    """Close the gap a lifted command leaves behind.

    ``"a, [[x]], b"`` would otherwise become ``"a, , b"`` and a prompt that was
    nothing but a command would become a lone comma, which is not an empty
    prompt to a text encoder -- and, worse here, is not an empty prompt to the
    Creative Writer either: a comma is a source phrase, and the writer would
    dutifully expand it into a paragraph about nothing.

    Only ever applied to text a command was actually removed from. The
    no-command path above returns before this, so tidying can be as opinionated
    as it needs to be without changing a prompt nobody put a command in.
    """
    text = _REPEATED_SEPARATORS.sub(",", text)
    text = _RUNS_OF_SPACES.sub(" ", text)
    text = _TRAILING_SPACES.sub("", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip().strip(",").strip()


# --------------------------------------------------------------------------- #
# Putting them back
# --------------------------------------------------------------------------- #

SPACE = " "
COMMA = ", "
"""The two separators anything here joins with, and where each one belongs.

A space at the top level of a Forge prompt, because that is where extra-network
tags and macros live and none of them wants a comma in front of it -- and
because a payload that *does* want commas carries its own. A comma inside a
BBOX element's ``desc``, because that field is a list of phrases about one
subject and the existing hints are already joined that way.
"""


def join(fragments, separator: str = SPACE) -> str:
    """Non-empty fragments, trimmed at the edges, in order, joined once.

    Trimmed at the *edges* only, and never inside. Everything this joins is
    either a payload the user typed or a body a model wrote, and neither is this
    function's to reflow.
    """
    found = [str(fragment).strip() for fragment in fragments if fragment is not None]
    return separator.join(fragment for fragment in found if fragment)


def restore(body, parsed, separator: str = SPACE) -> str:
    """``prefixes + body + suffixes``, in source order, exactly once.

    The last thing that happens to a Stage 1 prompt before Forge sees it, and
    the only place a payload is ever written back. Called once per generation on
    one final string -- not per pass, not per fallback -- because "exactly once"
    is easiest to guarantee when there is exactly one call site per prompt.

    ``parsed`` may be a :class:`LiteralParse` or a bare sequence of
    :class:`LiteralCommand`, so a caller holding a region's sidecar and a caller
    holding a whole parse can use the same function.
    """
    prefixes, suffixes = split(parsed)
    return join([*prefixes, body, *suffixes], separator)


def split(parsed) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(prefix payloads, suffix payloads)`` out of a parse or a command list."""
    if isinstance(parsed, LiteralParse):
        return parsed.prefixes, parsed.suffixes
    commands = tuple(parsed or ())
    return (tuple(command.payload for command in commands
                  if command.placement == PREFIX),
            tuple(command.payload for command in commands
                  if command.placement == SUFFIX))


def describe(parsed) -> str:
    """One line naming what was lifted, for a log or a status area.

    The count and the placement, never the payloads. A payload can be a
    paragraph, and a status line that printed two of them would push everything
    else off the panel -- and a log that recorded them would be keeping a copy
    of prompt content in a place nothing else keeps one.
    """
    prefixes, suffixes = split(parsed)
    total = len(prefixes) + len(suffixes)
    if not total:
        return "no literal commands"
    parts = []
    if prefixes:
        parts.append(f"{len(prefixes)} before")
    if suffixes:
        parts.append(f"{len(suffixes)} after")
    return (f"{total} literal command{'' if total == 1 else 's'} "
            f"({', '.join(parts)})")
