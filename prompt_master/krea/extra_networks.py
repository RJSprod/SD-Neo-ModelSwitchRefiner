"""Extra-network tags, protected without anybody having typed brackets.

A LoRA tag is a filename and two numbers. It is not scene language, and the
Creative Writer and the Spatial Composer are rewriters of scene language -- so
what happens to ``<lora:krea2_nickiminaj_v1:0.6>`` when it is left in the text
they are handed is what happens to any other phrase. It comes back reworded,
reordered, or gone.

:mod:`~prompt_master.krea.literals` already has the answer to that and states it
in one sentence: text inside ``[[...]]`` is lifted out before the first request
and put back after the last one. What that module will not do is *notice* which
text needs it, because it must not learn what a LoRA tag is -- its own rule, and
a good one. The moment it starts telling payload syntaxes apart it acquires an
opinion about every extension the user has installed, and is wrong about the
next one they add.

So the knowledge lives here, in the one small module allowed to have it, and it
is applied by the only means that adds no second prompt path::

    protect("portrait <lora:foo:0.6> in a cafe")
        -> "portrait [[<lora:foo:0.6>]] in a cafe"

That string then goes through :func:`literals.parse` exactly as a string
somebody typed brackets into would. There is no second parser, no second
sidecar and no second assembly step, and nothing downstream can tell an
automatic bracket from a typed one -- which is the property that makes this safe
to switch on for everybody rather than hide behind a setting.

Why there is no setting
-----------------------
The cost of the two mistakes is not symmetric.

A tag that should have been protected and was not is a LoRA silently missing
from the generation. On a card sized to its own checkpoint it is also expensive:
Forge bakes a LoRA into the weights when ``on_the_fly`` is false, so dropping
the tag makes the next pass *unpatch* them, and a card holding an 18.4 GB model
in 24 GB has nowhere to swap a 12.8 GB module in place. Everything leaves VRAM
and comes back. One real log paid 83 seconds for it.

A tag that is protected and did not need to be arrives at Stage 1 in the same
place it would have arrived anyway. Forge's own ``extra_networks`` pass strips
these tags out of the prompt before the text encoder is given one, so their
position in the string is not something the image model can observe.

Protecting is close to free and not protecting is expensive, so this is not a
question worth asking the user on every prompt.

What counts as one
------------------
The extra-network prefixes Forge's ``extra_networks`` pass registers. They fail
in exactly the same way for exactly the same reason, so they are treated the
same way; :data:`KINDS` is the whole of the knowledge in this file and adding to
it is the whole of the change if a fourth one appears.

This module still does not interpret a payload. It finds where one starts and
where it ends and puts two brackets around it -- what is *inside* remains as
opaque here as it is everywhere else.
"""

from __future__ import annotations

import re

from . import literals

KINDS = ("lora", "lyco", "hypernet")
"""The tag prefixes treated as extra-network references rather than as prose.

Deliberately a closed list rather than the ``<\\w+:...>`` shape Forge itself
matches. A closed list can only fail by missing something -- which is visible,
and is one line to fix -- while the open shape would also capture anything else
that happens to look like it, and silently lift a piece of somebody's prompt out
of the text the writer was meant to rewrite.
"""

_TAG = re.compile(r"<(?:%s):[^<>]*>" % "|".join(KINDS), re.IGNORECASE)
"""One tag. ``[^<>]*`` because a payload may hold colons, spaces and slashes --
a name, a weight, a second weight, an extension's own suffix -- and may not hold
a bracket of its own. Case-insensitive because ``<LoRA:...>`` is the same
request to Forge and would be a baffling way to lose one here.
"""


def present(text) -> bool:
    """Whether ``text`` carries an extra-network tag at all.

    One regex search, and it exists for the same reason
    :func:`literals.present` does: every generation that has nothing to do with
    this feature still runs through the hook that calls it, and "off is off" has
    to include "costs nothing".
    """
    return _TAG.search(str(text or "")) is not None


def protect(text) -> str:
    """``text`` with every unbracketed extra-network tag wrapped in ``[[...]]``.

    Returns the source string itself when there is nothing to wrap, so a prompt
    with no tags in it reaches the writer as the same bytes it would have
    reached it as before this module existed. That is the same guarantee
    :func:`literals.parse` makes about a prompt with no ``[[`` in it, and for the
    same reason.

    A tag already inside a command is left exactly as it is. Wrapping it again
    would produce ``[[[[<lora:x:1>]]]]``, and version 1 of the grammar closes on
    the first ``]]`` and has no escape syntax -- so the second wrap would not
    nest, it would corrupt the payload and leave two brackets in the prompt.
    """
    source = str(text or "")
    if not present(source):
        return source

    spans = _bracketed(source)
    kept: list[str] = []
    position = 0

    for match in _TAG.finditer(source):
        start, end = match.span()
        if any(begin <= start and end <= finish for begin, finish in spans):
            continue
        kept.append(source[position:start])
        kept.append(f"{literals.OPEN}{match.group(0)}{literals.CLOSE}")
        position = end

    if not kept:
        # Every tag was already protected by hand. Same bytes out as in.
        return source

    kept.append(source[position:])
    return "".join(kept)


def count(text) -> int:
    """How many tags :func:`protect` would wrap. For a log line, and for tests.

    The count and never the payloads, which is :func:`literals.describe`'s rule
    and worth keeping here: a log that recorded them would be keeping a copy of
    prompt content in a place nothing else keeps one.
    """
    source = str(text or "")
    if not present(source):
        return 0
    spans = _bracketed(source)
    return sum(1 for match in _TAG.finditer(source)
               if not any(begin <= match.start() and match.end() <= finish
                          for begin, finish in spans))


def _bracketed(source: str) -> tuple[tuple[int, int], ...]:
    """The half-open ranges :func:`literals.parse` will lift, delimiters included.

    Scanned by exactly the rule the parser uses -- the first ``]]`` after an
    ``[[`` closes it -- because "already inside a command" has to mean the same
    thing in both files. Two scanners that disagreed by a character would
    double-wrap a tag, and there is no escape syntax to recover from that.

    An unterminated ``[[`` protects everything after it, which is also the
    parser's behaviour: it stops there and leaves the remainder of the source as
    ordinary text with a warning. Wrapping a tag inside that remainder would be
    editing text the parser has already promised to hand back exactly as typed.
    """
    spans: list[tuple[int, int]] = []
    position = 0
    while True:
        start = source.find(literals.OPEN, position)
        if start < 0:
            break
        end = source.find(literals.CLOSE, start + len(literals.OPEN))
        if end < 0:
            spans.append((start, len(source)))
            break
        spans.append((start, end + len(literals.CLOSE)))
        position = end + len(literals.CLOSE)
    return tuple(spans)
