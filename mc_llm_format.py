"""How a message is drawn: its math, its quotes and its code, found once.

Asked for in these words: "Sometimes gemma4 responds with syntax like
"$\\rightarrow$", i want for all of the syntax formats to be handled
correctly. Also, when text is in quotes like "this" i want it to be slightly
more highlight. I think oobabooga does well with formatting."

Two surfaces draw a conversation, and they draw it differently. LLM Studio's
transcript is Gradio 4.40's ``Chatbot``: marked turns the Markdown into HTML,
DOMPurify sanitises it, and KaTeX 0.16.10 -- bundled with Gradio -- typesets
whatever lies between the ``latex_delimiters`` it is given, ``$$`` alone unless
told otherwise. That is why ``$\\rightarrow$`` reached the screen as written.
The Forge Assistant's panel draws its own HTML (``javascript/
forge_assistant_markdown.js``). What they must agree on is what a message
*says* -- where its math is, which of its words are quoted, which of its text
is code and so never touched -- and this module is that answer for LLM Studio,
with the panel's script carrying the same rules in JavaScript. The two are held
to one table of cases (``tests/fixtures/chat_format_cases.json``), so a rule
changed on one side and not the other fails a test.

Nothing here changes what is saved. The words a model wrote are the words in
the thread, and the words sent back to it on the next turn -- its prompt cache
reads them exactly (CLAUDE.md, the emotion's trap). This is the display only:
:func:`for_chatbot` is applied by ``mc_llm_chat_panel._body`` to the copy of a
message the transcript draws, and the marker's digest is taken from the words
as saved.

The rules
---------
*Code* is never formatted. Fenced blocks at any indentation (a fence inside a
list item is still a fence, and one left open while a reply streams runs to the
end, as marked's does); a block indented four spaces after a blank line below
flush prose that is not a list item, which is what marked calls code there; and
code spans, a run of backticks closed by a run of the same length.

*Math*, outside code:

- ``$$ … $$`` and ``\\[ … \\]``, displayed on a line of their own;
- ``\\( … \\)`` and ``$ … $``, inline;
- ``\\begin{env} … \\end{env}`` for the environments KaTeX knows that stand on
  their own (``equation``, ``align``, ``gather``, ``pmatrix``, ``cases`` and the
  rest of :data:`ENVIRONMENTS`), displayed.

No math crosses a blank line. A single ``$`` follows Pandoc's rule, which is
what keeps money out of it: the opening ``$`` has a character other than a
space after it, the closing one has a character other than a space before it
and no digit after it, and the first unescaped ``$`` after the opening one must
be that closing one, on the same line, with no backtick between them. So
``$\\rightarrow$`` and ``$x^2$`` are math, and ``$5 and $10``,
``$1,000-$2,000`` and ``($5) vs ($6)`` are money. ``\\$`` is a dollar sign.

*Quotes* are double quotes, straight or typographic -- ``"…"``, ``“…”``,
``„…“``, ``«…»``, ``「…」``, ``『…』`` -- on one line, with something in them,
never inside code, math, a tag or a link's address, and never across a table's
``|``. A straight one opens where the character before it is not a letter or a
digit and the one after it is not a space, and closes where the character
before it is not a space and the one after it is not a letter or a digit, so
``5" tall`` is a measurement and ``"word"s`` is not a quote; a mark that can
only open a quote, met while looking for the close of one, ends the search, so
in ``"word"s here "ok"`` the quote is ``"ok"``. Quotes do not nest: the outer
one is marked. They are drawn the way Oobabooga draws them, the
quote marks kept and the words between them coloured -- blue by day, amber at
night (``style.css``).

What LLM Studio gets
--------------------
:func:`for_chatbot` hands the Chatbot Markdown it already understands, changed
in three ways and no others:

1. Every piece of math is put between one of two pairs of invisible characters
   (:data:`INLINE_MATH`, :data:`DISPLAY_MATH`) in place of the delimiters it was
   written with, and the Chatbot is built with those two pairs as its
   ``latex_delimiters`` (:func:`latex_delimiters`). Gradio then does what it
   always does with a delimiter it was given -- lifts the math out before marked
   runs, puts it back after, and has KaTeX typeset it. Not ``$`` itself: Gradio
   would take every ``$`` on the page for a delimiter, money included, and a
   lone one makes marked's paragraph stop at it (Gradio adds a block-level
   ``latex`` rule whose start is the first left delimiter in the text). Not
   ``\\(`` either: Gradio lifts every pair of a delimiter it was given out of
   the message before marked runs, code blocks included, and puts it back as raw
   HTML, so a regular expression with ``\\(`` and ``<name>`` in a code block
   would lose its ``<name>`` to the sanitiser. Nobody types U+2061 to U+2063,
   and where marked does draw one as text -- math our rules found where marked
   decided on code -- they show as nothing. The math is escaped for HTML on the
   way, because Gradio puts it back into the HTML exactly as it found it.
2. Every quote is wrapped in ``<q class="mc-llm-quote">``, the quote marks
   inside it: marked passes an inline tag through and still formats what is in
   it, and DOMPurify keeps ``q`` and its class.
3. A tilde on its own is escaped (``\\~``). Gradio's marked strikes out
   whatever lies between two single tildes (``/^(~~?)(?=[^\\s~])…\\1/``), so
   ``3~4 hours and 5~6 days`` came out half struck; ``~~`` is still a
   strikethrough.
"""

from __future__ import annotations

import functools
import html
import re

INLINE_MATH = ("\u2061\u2063", "\u2063\u2061")
"""The pair inline math is put between for the Chatbot: FUNCTION APPLICATION
and INVISIBLE SEPARATOR, both default-ignorable, so a browser draws them as
nothing. The two left delimiters are not prefixes of each other, which KaTeX's
auto-render needs to tell them apart."""

DISPLAY_MATH = ("\u2062\u2063", "\u2063\u2062")
"""The pair displayed math is put between: INVISIBLE TIMES and INVISIBLE
SEPARATOR."""

ENVIRONMENTS = frozenset({
    "equation", "equation*", "align", "align*", "alignat", "alignat*", "gather", "gather*",
    "aligned", "alignedat", "gathered", "split", "CD",
    "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
    "matrix*", "pmatrix*", "bmatrix*", "Bmatrix*", "vmatrix*", "Vmatrix*", "smallmatrix",
    "cases", "dcases", "rcases", "drcases", "array", "darray", "subarray",
})
"""The environments taken for displayed math where they stand outside any:
KaTeX's own list, less the ones that only make sense inside another."""

QUOTE_PAIRS = {
    '"': ('"',),
    "\u201c": ("\u201d",),
    "\u201e": ("\u201c", "\u201d"),
    "\u00ab": ("\u00bb",),
    "\u300c": ("\u300d",),
    "\u300e": ("\u300f",),
}
"""Each opening quote mark and the marks that close it."""

QUOTE_CLASS = "mc-llm-quote"

ASCII_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
"""What a backslash escapes in Markdown (CommonMark, 2.4)."""

_FENCE = re.compile(r"[ \t]*(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE = re.compile(r"[ \t]*(`{3,}|~{3,})[ \t]*$")
_LIST_ITEM = re.compile(r"(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_ENVIRONMENT = re.compile(r"\\begin\{([A-Za-z]+\*?)\}")
_TAG = re.compile(
    r"<(?:[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z_:][A-Za-z0-9_.:-]*"
    r"(?:\s*=\s*(?:[^\s\"'=<>`]+|'[^'\n]*'|\"[^\"\n]*\"))?)*\s*/?"
    r"|/[A-Za-z][A-Za-z0-9-]*\s*)>")
"""An HTML open or close tag on one line, CommonMark's shape (6.6)."""
_AUTOLINK = re.compile(r"<(?:[A-Za-z][A-Za-z0-9+.-]{1,31}:[^\s<>]*"
                       r"|[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-.]*[A-Za-z0-9])?)>")
_ATOM_START = re.compile(r"[\\`$<\]]")
_QUOTE_START = re.compile("[~\"\u201c\u201e\u00ab\u300c\u300e\\\\]")

CODE, MATH, VERBATIM, ESCAPE = "code", "math", "verbatim", "escape"


# --------------------------------------------------------------------------- #
# Code lines
# --------------------------------------------------------------------------- #


def _indented(line: str) -> bool:
    width = 0
    for char in line:
        if char == " ":
            width += 1
        elif char == "\t":
            width += 4 - width % 4
        else:
            break
    return width >= 4


def _blank(line: str) -> bool:
    """Spaces and tabs only -- the same test as the blank lines ``_BLANK_LINE`` finds."""
    return not line.strip(" \t")


def code_lines(lines: list[str]) -> list[bool]:
    """Which of ``lines`` are code: fenced blocks and indented blocks.

    A fence opens on a line whose first characters, after any indentation, are
    three or more backticks or tildes (a backtick fence's info string holds no
    backtick), and closes on a line of the same character, at least as many,
    and nothing else; one never closed runs to the end. An indented block starts
    after a blank line, or at the top, below flush prose that is not a list
    item, and holds the indented and blank lines that follow it, less the blank
    ones at its end.
    """
    code = [False] * len(lines)
    previous = None
    after_blank = True
    index = 0
    while index < len(lines):
        line = lines[index]
        opened = _FENCE.match(line)
        if opened and not (opened.group(1)[0] == "`" and "`" in opened.group(2)):
            char, size = opened.group(1)[0], len(opened.group(1))
            code[index] = True
            index += 1
            while index < len(lines):
                code[index] = True
                closed = _FENCE_CLOSE.match(lines[index])
                index += 1
                if closed and closed.group(1)[0] == char and len(closed.group(1)) >= size:
                    break
            previous, after_blank = lines[index - 1], False
            continue
        if _blank(line):
            after_blank = True
            index += 1
            continue
        if (after_blank and _indented(line)
                and (previous is None
                     or (previous[:1] not in (" ", "\t") and not _LIST_ITEM.match(previous)))):
            last = index
            ahead = index
            while ahead < len(lines) and (_blank(lines[ahead]) or _indented(lines[ahead])):
                if not _blank(lines[ahead]):
                    last = ahead
                ahead += 1
            for covered in range(index, last + 1):
                code[covered] = True
            previous, after_blank = lines[last], False
            index = last + 1
            continue
        previous, after_blank = line, False
        index += 1
    return code


# --------------------------------------------------------------------------- #
# Atoms: code spans, math, tags and links' addresses, escapes
# --------------------------------------------------------------------------- #


def _escaped(text: str, index: int) -> bool:
    """Whether the character at ``index`` is escaped by an odd run of backslashes."""
    count = 0
    while index - 1 - count >= 0 and text[index - 1 - count] == "\\":
        count += 1
    return count % 2 == 1


def _closing(text: str, start: int, closer: str) -> int:
    """Where an unescaped ``closer`` is, from ``start``, before a blank line; -1 if nowhere."""
    limit = _BLANK_LINE.search(text, start)
    end = limit.start() if limit else len(text)
    found = text.find(closer, start, end)
    while found >= 0 and _escaped(text, found):
        found = text.find(closer, found + 1, end)
    return found


def _backtick_run(text: str, start: int, end: int, size: int) -> int:
    """The end of a run of exactly ``size`` backticks after ``start``, before
    ``end``; -1 if there is none. A code span's closer (CommonMark, 6.1)."""
    at = text.find("`", start)
    while 0 <= at < end:
        stop = at
        while stop < len(text) and text[stop] == "`":
            stop += 1
        if stop - at == size and stop <= end:
            return stop
        at = text.find("`", stop)
    return -1


def _environment(text: str, index: int):
    """The whole ``\\begin{env}…\\end{env}`` at ``index``, as its end, or None."""
    begun = _ENVIRONMENT.match(text, index)
    if not begun or begun.group(1) not in ENVIRONMENTS:
        return None
    name = re.escape(begun.group(1))
    pattern = re.compile(r"\\(begin|end)\{" + name + r"\}")
    limit = _BLANK_LINE.search(text, begun.end())
    end = limit.start() if limit else len(text)
    depth = 1
    for mark in pattern.finditer(text, begun.end(), end):
        depth += 1 if mark.group(1) == "begin" else -1
        if depth == 0:
            return mark.end()
    return None


def _inline_dollar(text: str, index: int):
    """The end of ``$…$`` math opened at ``index`` (past its closing ``$``), or None."""
    after = index + 1
    if after >= len(text) or text[after].isspace() or text[after] == "$":
        return None
    position = after
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2
            continue
        if char in "\n`":
            return None
        if char == "$":
            if text[position - 1].isspace():
                return None
            following = text[position + 1] if position + 1 < len(text) else ""
            if "0" <= following <= "9":
                return None
            return position + 1
        position += 1
    return None


def _destination(text: str, index: int):
    """The end of a link's ``(address "title")`` after the ``]`` at ``index``, or None."""
    if not text.startswith("](", index):
        return None
    depth = 0
    position = index + 2
    while position < len(text):
        char = text[position]
        if char == "\n":
            return None
        if char == "\\":
            position += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return position + 1
            depth -= 1
        position += 1
    return None


def atoms(text: str) -> list[tuple]:
    """The pieces of prose (no code lines in it) that nothing may reach into.

    Each is ``(start, end, kind, payload)``: ``code`` for a code span, ``math``
    with ``(tex, display)``, ``verbatim`` for a tag, an autolink or a link's
    address, ``escape`` for a backslash and the punctuation it escapes. In the
    order they appear, never overlapping.
    """
    found = []
    index = 0
    length = len(text)
    while True:
        hit = _ATOM_START.search(text, index)
        if not hit:
            break
        index = hit.start()
        char = text[index]
        if char == "\\":
            if text.startswith("\\(", index):
                close = _closing(text, index + 2, "\\)")
                if close >= 0 and text[index + 2:close].strip():
                    found.append((index, close + 2, MATH, (text[index + 2:close], False)))
                    index = close + 2
                    continue
            elif text.startswith("\\[", index):
                close = _closing(text, index + 2, "\\]")
                if close >= 0 and text[index + 2:close].strip():
                    found.append((index, close + 2, MATH, (text[index + 2:close], True)))
                    index = close + 2
                    continue
            elif text.startswith("\\begin{", index):
                end = _environment(text, index)
                if end is not None:
                    found.append((index, end, MATH, (text[index:end], True)))
                    index = end
                    continue
            if index + 1 < length and text[index + 1] in ASCII_PUNCTUATION:
                found.append((index, index + 2, ESCAPE, None))
                index += 2
            else:
                index += 1
            continue
        if char == "`":
            run = index
            while run < length and text[run] == "`":
                run += 1
            limit = _BLANK_LINE.search(text, run)
            stop = _backtick_run(text, run, limit.start() if limit else length, run - index)
            if stop >= 0:
                found.append((index, stop, CODE, None))
                index = stop
            else:
                index = run
            continue
        if char == "$":
            run = index
            while run < length and text[run] == "$":
                run += 1
            size = run - index
            if size == 2 and not _escaped(text, index):
                close = _closing(text, run, "$$")
                if close >= 0 and text[run:close].strip():
                    found.append((index, close + 2, MATH, (text[run:close], True)))
                    index = close + 2
                    continue
            elif size == 1 and not _escaped(text, index):
                end = _inline_dollar(text, index)
                if end is not None:
                    found.append((index, end, MATH, (text[index + 1:end - 1], False)))
                    index = end
                    continue
            index = run
            continue
        if char == "<":
            tag = _TAG.match(text, index) or _AUTOLINK.match(text, index)
            if tag:
                found.append((index, tag.end(), VERBATIM, None))
                index = tag.end()
                continue
            index += 1
            continue
        end = _destination(text, index)  # char == "]"
        if end is not None:
            found.append((index + 1, end, VERBATIM, None))
            index = end
            continue
        index += 1
    return found


# --------------------------------------------------------------------------- #
# Quotes and lone tildes
# --------------------------------------------------------------------------- #


def _letter_or_digit(char: str) -> bool:
    return bool(char) and char.isalnum()


def _opens(text: str, index: int) -> bool:
    mark = text[index]
    after = text[index + 1] if index + 1 < len(text) else ""
    if not after:
        return False
    if mark == "\u00ab":
        return True
    if after.isspace():
        return False
    if mark == '"':
        before = text[index - 1] if index > 0 else ""
        return after != '"' and not _letter_or_digit(before)
    return True


def _closes(text: str, index: int, opener: str) -> bool:
    if opener == "\u00ab":
        return True
    before = text[index - 1] if index > 0 else ""
    if not before or before.isspace():
        return False
    if opener == '"':
        after = text[index + 1] if index + 1 < len(text) else ""
        return not _letter_or_digit(after)
    return True


def marks(text: str, found_atoms: list[tuple]) -> tuple[list[tuple], list[int]]:
    """The quotes in prose and its lone tildes, given its :func:`atoms`.

    Quotes are ``(opening index, closing index)``, the indices of the two quote
    marks; tildes are the index of each tilde with no tilde beside it.
    """
    quotes = []
    tildes = []
    starts = {start: (end, kind, payload) for start, end, kind, payload in found_atoms}
    index = 0
    length = len(text)
    following = 0
    while index < length:
        # The next atom at or after here; one under ``index`` is stepped over.
        while following < len(found_atoms) and found_atoms[following][1] <= index:
            following += 1
        if following < len(found_atoms) and found_atoms[following][0] <= index:
            index = found_atoms[following][1]
            continue
        bound = found_atoms[following][0] if following < len(found_atoms) else length
        hit = _QUOTE_START.search(text, index, bound)
        if not hit:
            index = bound
            continue
        index = hit.start()
        char = text[index]
        if char == "\\":
            index += 1
            continue
        if char == "~":
            run = index
            while run < bound and text[run] == "~":
                run += 1
            if run - index == 1:
                tildes.append(index)
            index = run
            continue
        if not _opens(text, index):
            index += 1
            continue
        closers = QUOTE_PAIRS[char]
        position = index + 1
        closed = -1
        while position < length:
            if position in starts:
                end, kind, payload = starts[position]
                if kind == MATH and payload[1]:
                    break
                position = end
                continue
            here = text[position]
            if here in "\n|":
                break
            if here == "\\":
                position += 1
                continue
            if here in closers and _closes(text, position, char):
                closed = position
                break
            if here == char and _opens(text, position):
                break   # another quote opens: this one was never closed
            position += 1
        if closed > 0 and _has_words(text, index + 1, closed, starts):
            quotes.append((index, closed))
            index = closed + 1
        else:
            index += 1
    return quotes, tildes


def _has_words(text: str, start: int, end: int, starts: dict) -> bool:
    """Whether a quote holds anything but space -- an atom counts."""
    return any(not text[at].isspace() or at in starts for at in range(start, end))


# --------------------------------------------------------------------------- #
# The whole message
# --------------------------------------------------------------------------- #


def _normalised(text) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def chunks(text: str) -> list[tuple[bool, str]]:
    """The message as ``(is_code, text)`` runs of whole lines, newlines kept."""
    lines = text.split("\n")
    flags = code_lines(lines)
    runs = []
    for line, is_code in zip(lines, flags):
        if runs and runs[-1][0] == is_code:
            runs[-1][1].append(line)
        else:
            runs.append((is_code, [line]))
    return [(is_code, "\n".join(body)) for is_code, body in runs]


def describe(text) -> dict:
    """What the rules find in a message: its math and its quotes, in order.

    The shape the shared table of cases is written in, and the JavaScript's
    ``describeMessage`` answers in the same shape.
    """
    math = []
    quotes = []
    for is_code, body in chunks(_normalised(text)):
        if is_code:
            continue
        found = atoms(body)
        math.extend([payload[0], payload[1]] for _s, _e, kind, payload in found if kind == MATH)
        quotes.extend(body[start:end + 1] for start, end in marks(body, found)[0])
    return {"math": math, "quotes": quotes}


@functools.lru_cache(maxsize=1024)
def _for_chatbot(text: str) -> str:
    parts = []
    for is_code, body in chunks(text):
        if is_code:
            parts.append(body)
            continue
        found = atoms(body)
        quotes, tildes = marks(body, found)
        before = {start: f'<q class="{QUOTE_CLASS}">' for start, _end in quotes}
        after = {end: "</q>" for _start, end in quotes}
        lone = set(tildes)
        maths = {start: (end, payload) for start, end, kind, payload in found if kind == MATH}
        out = []
        index = 0
        while index < len(body):
            if index in maths:
                end, (tex, display) = maths[index]
                left, right = DISPLAY_MATH if display else INLINE_MATH
                out.append(left + html.escape(tex, quote=False) + right)
                index = end
                continue
            if index in before:
                out.append(before[index])
            out.append("\\~" if index in lone else body[index])
            if index in after:
                out.append(after[index])
            index += 1
        parts.append("".join(out))
    return "\n".join(parts)


def for_chatbot(text) -> str:
    """A message's words as LLM Studio's transcript is given them.

    The math between the Chatbot's own invisible delimiters and escaped for
    HTML, each quote in ``<q class="mc-llm-quote">``, each lone tilde escaped;
    code and everything else exactly as written. Display only -- see the module
    docstring.
    """
    return _for_chatbot(_normalised(text))


def latex_delimiters() -> list[dict]:
    """The Chatbot's ``latex_delimiters``: the two invisible pairs, and no other."""
    return [{"left": INLINE_MATH[0], "right": INLINE_MATH[1], "display": False},
            {"left": DISPLAY_MATH[0], "right": DISPLAY_MATH[1], "display": True}]
