"""Where a reply may safely be cut so that a sentence of it can be spoken.

Streaming speech asks a question the completed-reply design never had to: given
the first 143 characters of an answer that is still arriving, how much of it is
*finished*? Kokoro is an offline synthesiser -- it is handed a string and it
produces the audio for that string -- so every character passed to it is a
commitment. Hand it a token at a time and it says each word with the falling
intonation of a full stop. Hand it "The answer is 3." when the model was in the
middle of writing "3.14159" and it says "three" and then, a second later, says
"point one four one five nine" as though a new sentence had begun.

So this module is the one that decides, and everything it decides is decided
once. A segment that has been handed over is immutable: the audio for it may
already be in somebody's ear by the time the next chunk arrives, which is the
whole semantic change section 4 of the design intent describes.

What it is not
--------------
Not a sentence tokenizer. Section 116 lists "complex NLP sentence tokenizer"
among the non-goals, and the reason is the dependency rather than the accuracy:
this runs inside the WebUI process, beside an image model, on a machine where
adding a data-file-shipping NLP package to speak a chat reply would be a
strange trade. What is here instead is a deterministic state machine with a
named guard for each way a full stop lies -- and each guard has a test, which
is the honest form of "good enough".

The four boundaries, in order of preference (section 8):

    paragraph        a blank line, which is a pause a reader would take anyway
    sentence         . ? ! followed by space or end, once the guards agree
    clause           ; : and -- only when the unsent text is already long
    word             the last resort, at the hard ceiling, never inside a word

Nothing is ever cut inside a word, inside a number, inside a URL, inside a
markdown fence, or between the dots of an ellipsis.

The label
---------
``prompt_master.chat.prompt.clean_reply`` removes a leading ``Name:`` that a
chat-trained model wrote anyway. It runs at the *end* of a reply. Streaming
would speak "Assistant colon" a second before the panel silently removed it, so
:class:`Segmenter` holds the opening back until it can tell -- which costs the
first segment nothing in practice, because deciding takes at most one short
line of text and the first segment target is longer than that.

Normalization
-------------
Section 10: do not summarize, paraphrase or rewrite. What :func:`speech_text`
removes is markdown *punctuation* that a speech engine pronounces or stumbles
over -- emphasis stars, heading hashes, table pipes, bullet markers -- and what
it keeps is every word, every number, every URL and every line of code. A reply
that says something is a reply that says it aloud.
"""

from __future__ import annotations

import re

FIRST_TARGET = 60
"""Characters before the first segment is worth committing on a weak boundary.

Low on purpose. The first segment is the one the whole feature's latency is
measured against: everything after it is hidden behind playback, and nothing
before it is. A complete short sentence commits below this -- see
:data:`SHORT_SENTENCE` -- so "Yes, that's possible." is spoken as soon as it
is whole rather than waiting for a paragraph that may be four seconds away.
"""

SECOND_TARGET = 50
"""Characters before the *second* segment is worth committing.

The second segment is where the reported gap actually is. The first one commits
early and is spoken almost immediately; the second one used to wait for the
ordinary hundred-character target, and a hundred characters of a reply that is
still being written can be several seconds away -- long enough for playback of
sentence one to run out before sentence two exists. Lower than TARGET and
higher than a phrase, because a target this low applied to *every* segment
would turn one synthesis call into four and put a seam between each of them.
"""

SECOND_SOFT_MAX = 140
SECOND_HARD_MAX = 220
"""How large the second segment is allowed to become.

:data:`SECOND_TARGET` says how *soon* the second segment may be committed. It
says nothing about how big it may get, and the two are not the same question: a
model that writes a hundred and eighty characters before its first full stop
hands the second segment a single enormous sentence, which arrives as text
quickly and then takes as long to synthesise as everything it was supposed to be
covering. Continuity is lost to the size of the unit rather than to the wait for
it.

So the second segment gets the whole envelope early, not just the target: a
clause or a comma will do at :data:`SECOND_SOFT_MAX`, and a word boundary has to
by :data:`SECOND_HARD_MAX`. Both are far below the ordinary 320/480, and both
apply to the final tail of a reply as well -- a long second unit must not escape
the envelope merely because generation happened to end before another segment
arrived.
"""

TARGET = 100
SOFT_MAX = 320
HARD_MAX = 480
"""Section 8's suggested thresholds. TARGET is where an ordinary sentence
boundary starts being taken, SOFT_MAX is where a clause boundary will do, and
HARD_MAX is where a word boundary has to."""

SHORT_SENTENCE = 12
"""A complete sentence shorter than the first-segment target still commits, as
long as it is long enough to be a sentence rather than a stray "Mr." that the
abbreviation guard let through."""

LABEL_SCAN = 96
"""How far into a reply a leading ``Name:`` label may begin.

Beyond this there is no label -- there is a sentence with a colon in it -- and
holding the opening back any longer would cost the first segment for nothing.
"""

MAX_LABEL_WORDS = 6
"""``Ambassador Anne-Marie de la Court:`` is a character name. A colon after
two lines of prose is not, and the word count is what tells them apart."""

ABBREVIATIONS = frozenset("""
mr mrs ms mx dr prof sr jr st rev hon gen col sgt lt capt cmdr
inc ltd co corp dept est fig no vol ch sec art pp ed eds trans
vs etc al cf ca approx est max min avg dept univ
am pm ad bc ce bce
jan feb mar apr jun jul aug sep sept oct nov dec
mon tue tues wed thu thur thurs fri sat sun
""".split())
"""Words whose full stop is part of the word.

Deliberately a set of ordinary abbreviations rather than an attempt at every
one: a missed abbreviation costs one early pause in one sentence, and a
tokenizer dependency costs the feature its ability to run anywhere.
``e.g.`` and ``i.e.`` are not here because the single-letter rule below already
covers them, along with initials like ``J. R. R. Tolkien``.
"""

_TERMINATORS = ".?!"
_CLOSERS = "\"')]}”’»"
"""Quote and bracket characters that belong to the sentence they close, so the
boundary is after them: ``He said "no." Then he left.`` splits after the
second full stop, not between it and the quote mark."""

_WORD_TAIL = re.compile(r"([A-Za-z][A-Za-z'’.-]*)$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_QUOTE_MARK = re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE)
_BULLET = re.compile(r"^[ \t]*([-*+]|\d{1,3}[.)])[ \t]+", re.MULTILINE)
_RULE = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.MULTILINE)
_FENCE_LINE = re.compile(r"^[ \t]{0,3}(?:`{3,}|~{3,})[^\n]*$", re.MULTILINE)
_LINK = re.compile(r"\[([^\]\n]+)\]\((?:[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_SPACES = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


def speech_text(text: str) -> str:
    """One segment as the words to say, with the typography taken out.

    Every rule here removes a *mark*, never a word. ``**important**`` becomes
    ``important`` rather than "star star important star star"; a bullet becomes
    the pause a list item is read with; a link keeps its label and loses its
    address, because reading a URL's punctuation aloud in the middle of a
    sentence is the one case where the address is noise and the label is the
    content. A bare URL in running text is left exactly as it is -- it is what
    the reply said, and section 10 says not to decide otherwise in this release.
    """
    if not text:
        return ""
    found = text.replace("\r\n", "\n").replace("\r", "\n")
    found = _FENCE_LINE.sub("", found)
    found = _LINK.sub(r"\1", found)
    found = _INLINE_CODE.sub(r"\1", found)
    found = _RULE.sub("", found)
    found = _HEADING.sub("", found)
    found = _QUOTE_MARK.sub("", found)
    # A list item is a pause, which is what the design intent asks list
    # separators to become. The full stop is added only when the line does not
    # already end in punctuation that would produce one.
    found = _BULLET.sub("", found)
    for _pass in (1, 2):
        # Twice, because ``**a _b_ c**`` needs the outer pair removed before the
        # inner pair is adjacent to its own delimiters. Two passes is enough for
        # every nesting a reply actually contains and cannot loop.
        found = _EMPHASIS.sub(r"\2", found)
    found = found.replace("|", " ")
    found = _SPACES.sub(" ", found)
    found = _BLANKS.sub("\n\n", found)
    return found.strip()


def leading_label(text: str, labels) -> str:
    """The ``Name:`` at the front of ``text``, if one of ``labels`` is there.

    Matched exactly as ``clean_reply`` matches it -- casefolded, name then
    colon -- so that the two cannot disagree about whether a reply begins with
    a label. Returns what should be removed, including the colon, or "".
    """
    stripped = text.lstrip()
    lowered = stripped.casefold()
    for name in labels or ():
        wanted = str(name or "").strip()
        if not wanted:
            continue
        if lowered.startswith(f"{wanted.casefold()}:"):
            return stripped[:len(wanted) + 1]
    return ""


def label_undecided(text: str, labels) -> bool:
    """Whether more text is needed before the opening can be committed.

    True only while a label is still *possible*: nothing has ruled it out and
    nothing has confirmed it. A newline, a sentence terminator, a colon, more
    than :data:`MAX_LABEL_WORDS` words or :data:`LABEL_SCAN` characters all rule
    it out, and after any of them the opening is spoken as it stands.
    """
    if not labels:
        return False
    stripped = text.lstrip()
    if not stripped:
        return True
    if leading_label(stripped, labels):
        return False
    head = stripped[:LABEL_SCAN]
    if len(stripped) >= LABEL_SCAN:
        return False
    for index, character in enumerate(head):
        if character in "\n" or character in _TERMINATORS:
            return False
        if character == ":":
            # A colon this early is either a label we do not know or an ordinary
            # one. Either way the question is settled and nothing is held back.
            return False
        if index and character == " " and head[:index].count(" ") >= MAX_LABEL_WORDS:
            return False
    return True


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #


def _fences(text: str, position: int = None) -> int:
    """How many ``` fence markers open or close before ``position``.

    Counted rather than parsed: fences come in pairs, so the parity of this
    number is whether a fence is open -- and a boundary inside an open fence is
    a boundary in the middle of a program.
    """
    return len(_FENCE.findall(text, 0, len(text) if position is None else position))


def _decimal(text: str, index: int) -> bool:
    """``3.14`` -- a digit on each side of the dot."""
    if text[index] != ".":
        return False
    before = text[index - 1] if index else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    return before.isdigit() and after.isdigit()


def _abbreviation(text: str, index: int) -> bool:
    """``Dr.`` and ``J.`` -- a full stop that belongs to the word before it."""
    if text[index] != ".":
        return False
    match = _WORD_TAIL.search(text, 0, index)
    if not match:
        return False
    word = match.group(1)
    if len(word) == 1:
        # An initial. ``J. R. R. Tolkien`` and ``e.g.`` both land here, the
        # second because ``e`` is what precedes each of its dots.
        return True
    return word.casefold().strip(".") in ABBREVIATIONS


def _domain(text: str, index: int) -> bool:
    """``example.com`` and ``https://host/path`` -- a dot inside one token.

    The test is what follows: a sentence's full stop is followed by whitespace
    or by end of text, and a domain's is followed immediately by more of the
    same word. That single rule covers hostnames, file extensions and paths
    without this module needing to know what a URL is.
    """
    if text[index] != ".":
        return False
    after = text[index + 1] if index + 1 < len(text) else ""
    return bool(after) and not after.isspace()


def _ellipsis(text: str, index: int) -> bool:
    """A dot with another dot on either side of it: ``...``.

    Only the last dot of the run is a candidate boundary, which is what stops
    "wait..." from being spoken as three sentences.
    """
    if text[index] != ".":
        return False
    if index + 1 < len(text) and text[index + 1] == ".":
        return True
    return index >= 1 and text[index - 1] == "." and (
        index + 1 >= len(text) or text[index + 1] != ".")


def _sentence_end(text: str, index: int) -> int:
    """Where a sentence ending at ``index`` actually stops, or 0.

    Returns the cut position -- after the terminator and after any quote or
    bracket that closes with it -- or 0 when ``index`` is not a sentence end.
    """
    if text[index] not in _TERMINATORS:
        return 0
    if _decimal(text, index) or _domain(text, index) or _ellipsis(text, index):
        return 0
    if _abbreviation(text, index):
        return 0
    cut = index + 1
    while cut < len(text) and text[cut] in _CLOSERS:
        cut += 1
    # Repeated terminators are one ending: "Really?!" ends once.
    while cut < len(text) and text[cut] in _TERMINATORS:
        cut += 1
        while cut < len(text) and text[cut] in _CLOSERS:
            cut += 1
    if cut >= len(text):
        return cut
    return cut if text[cut].isspace() else 0


def _paragraph_end(text: str, limit: int, fence=lambda _index: False) -> int:
    """The last blank line at or before ``limit``, or 0."""
    best = 0
    search = 0
    while True:
        found = text.find("\n\n", search, limit)
        if found < 0:
            break
        after = found + 2
        while after < len(text) and text[after] == "\n":
            after += 1
        if text[:found].strip() and not fence(found):
            best = after
        search = found + 1
    return best


def _clause_end(text: str, limit: int, fence=lambda _index: False) -> int:
    """The last ``;`` ``:`` or dash boundary at or before ``limit``, or 0."""
    best = 0
    for index in range(min(limit, len(text)) - 1, -1, -1):
        character = text[index]
        if character not in ";:—":
            continue
        after = text[index + 1] if index + 1 < len(text) else " "
        if not after.isspace():
            continue
        if fence(index):
            continue
        best = index + 1
        break
    return best


def _comma_end(text: str, limit: int, fence=lambda _index: False) -> int:
    best = 0
    for index in range(min(limit, len(text)) - 1, -1, -1):
        if text[index] != ",":
            continue
        after = text[index + 1] if index + 1 < len(text) else " "
        if not after.isspace() or fence(index):
            continue
        best = index + 1
        break
    return best


def _word_end(text: str, limit: int) -> int:
    """The last space at or before ``limit``. Never inside a word."""
    for index in range(min(limit, len(text)) - 1, 0, -1):
        if text[index].isspace():
            return index
    return 0


def _sentence_at_or_before(text: str, limit: int, fence=lambda _index: False) -> int:
    best = 0
    index = 0
    while index < min(limit, len(text)):
        cut = _sentence_end(text, index)
        if cut and cut <= limit and not fence(index):
            best = cut
        index += 1
    return best


def _first_sentence(text: str, minimum: int, limit: int, fence=lambda _index: False) -> int:
    """The *earliest* sentence boundary at least ``minimum`` characters in, or 0.

    The opposite end of the same scan as :func:`_sentence_at_or_before`, and the
    reason it exists is latency rather than correctness. Taking the last
    boundary is right for an ordinary segment, where a longer piece of text is a
    better synthesis call. It is wrong for the opening of a reply: "Yes, that's
    possible." is the whole answer, and holding it back because the next
    sentence has begun to arrive in the same streamed chunk is exactly the delay
    the first segment exists to avoid.

    ``minimum`` is what stops "1." and a stray "Mr." from becoming a segment;
    every guard :func:`_sentence_end` applies is applied here unchanged.
    """
    index = 0
    while index < min(limit, len(text)):
        cut = _sentence_end(text, index)
        if cut and cut <= limit and cut >= minimum and not fence(index):
            return cut
        index += 1
    return 0


# --------------------------------------------------------------------------- #
# The segmenter
# --------------------------------------------------------------------------- #


class Segmenter:
    """Turns a growing reply into immutable segments, one commitment at a time.

    ``labels`` are the names ``clean_reply`` would strip -- the character, the
    persona and "Assistant" -- and are the only reason this class ever holds
    text back that it could otherwise have spoken.

    Usage is three calls:

        feed(delta)   more of the reply arrived; returns whatever is now whole
        flush(final)  the reply is complete; returns everything still unsaid
        pending       what is held back right now, for a metric or a test

    The class holds text, never audio, and it is not thread-safe on purpose:
    :class:`mc_voice_turn.VoiceTurn` owns exactly one and serializes access to
    it, which is cheaper and easier to reason about than a lock per reply.
    """

    def __init__(self, labels=(), first_target: int = FIRST_TARGET, target: int = TARGET,
                 soft_max: int = SOFT_MAX, hard_max: int = HARD_MAX,
                 second_target: int = SECOND_TARGET,
                 second_soft_max: int = SECOND_SOFT_MAX,
                 second_hard_max: int = SECOND_HARD_MAX):
        self.labels = tuple(str(name or "").strip() for name in (labels or ()) if str(name or "").strip())
        self.first_target = max(1, int(first_target))
        self.target = max(self.first_target, int(target))
        self.second_target = min(self.target, max(1, int(second_target)))
        self.soft_max = max(self.target, int(soft_max))
        self.hard_max = max(self.soft_max, int(hard_max))
        self.second_soft_max = max(self.second_target, int(second_soft_max))
        self.second_hard_max = max(self.second_soft_max, int(second_hard_max))
        self._buffer = ""
        self._committed = 0
        self._segments = 0
        self._label_removed = False
        self._fence_open = 0

    # -- state ------------------------------------------------------------- #

    @property
    def pending(self) -> str:
        """The text held back: arrived, not yet committed to speech."""
        return self._buffer[self._committed:]

    @property
    def spoken(self) -> str:
        """Everything committed so far, as it was before normalization."""
        return self._buffer[:self._committed]

    @property
    def segments(self) -> int:
        return self._segments

    # -- feeding ----------------------------------------------------------- #

    def feed(self, delta: str) -> list:
        """Add newly generated text; return the segments that are now whole."""
        if delta:
            self._buffer += str(delta)
        return self._drain(final=False)

    def replace(self, whole: str) -> list:
        """Adopt an authoritative *cumulative* text and return what is new.

        Used for the final reply, which the panel re-derives with
        ``clean_reply`` and which is therefore the truth about what was said.
        It differs from the concatenated chunks in exactly the ways the panel
        changes it: a leading label this class removed already, and stripped
        whitespace at both ends. Neither of those is a difference in *words*,
        so the two are aligned by their non-space characters rather than by
        byte equality -- one leading space would otherwise look like a reply
        that had been rewritten from its first character.

        Text that has already been committed is never revisited. Audio in
        somebody's ear is not editable, so where the authoritative text really
        does contradict what was spoken, the tail is re-based rather than
        repeated: what is still unsaid gets said, and what was said stands.
        """
        wanted = str(whole or "")
        spoken = self.spoken
        covered, whole_of_it = _covered(wanted, spoken)
        self._buffer = wanted
        self._committed = covered if whole_of_it else max(covered, min(len(wanted), len(spoken)))
        return []

    def flush(self, whole: str = None) -> list:
        """Everything left, as segments. The reply is over.

        ``whole`` is the authoritative final text when the caller has one.
        """
        if whole is not None:
            self.replace(whole)
        return self._drain(final=True)

    # -- the decision ------------------------------------------------------ #

    def _drain(self, final: bool) -> list:
        found = []
        while True:
            segment = self._next(final)
            if not segment:
                break
            found.append(segment)
        return found

    def _next(self, final: bool) -> str:
        text = self._buffer[self._committed:]
        if not text.strip():
            if final:
                self._committed = len(self._buffer)
            return ""

        if self._segments == 0 and not self._label_removed:
            if not final and label_undecided(text, self.labels):
                return ""
            label = leading_label(text, self.labels)
            self._label_removed = True
            if label:
                # Dropped from the buffer entirely rather than skipped over, so
                # that ``spoken`` compares cleanly with the authoritative final
                # text, which will not contain the label either.
                lead = len(text) - len(text.lstrip())
                start = self._committed + lead
                self._buffer = self._buffer[:start] + self._buffer[start + len(label):]
                text = self._buffer[self._committed:]

        cut = self._cut(text, final)
        if cut <= 0:
            return ""
        raw = text[:cut]
        spoken = speech_text(raw)
        # Parity, not a count: a fence opened in a segment that has already been
        # spoken is still open for the segment after it, and the pending buffer
        # on its own cannot know that.
        self._fence_open += _fences(raw)
        self._committed += cut
        if not spoken:
            # Whitespace, a horizontal rule, a lone fence marker: consumed so
            # the buffer moves, but never handed on as an empty segment.
            return self._next(final)
        self._segments += 1
        return spoken

    def _limits(self) -> tuple:
        """``(target, soft_max, hard_max)`` for the segment about to be committed.

        One place, deliberately. Every threshold in :meth:`_cut` comes from here
        rather than from the instance directly, so "which segment is this and
        what is it allowed to be" is answered once instead of being re-derived
        at each of the five boundary rules -- and so that the second segment's
        envelope cannot be applied on one path and forgotten on another.

        Three rows rather than two, and the middle one is the whole of the
        reported first-to-second gap. The opening commits early because nothing
        is playing yet; the second one has to commit early *and* stay small,
        because sentence one is playing and the producer has no lead at all; by
        the third there is audio queued ahead of it and a longer,
        better-sounding segment costs nothing anybody can hear.
        """
        if self._segments == 0:
            return self.first_target, self.soft_max, self.hard_max
        if self._segments == 1:
            return self.second_target, self.second_soft_max, self.second_hard_max
        return self.target, self.soft_max, self.hard_max

    def _cut(self, text: str, final: bool) -> int:
        """How much of ``text`` is safe to speak, or 0 for "not yet"."""
        length = len(text)
        target, soft_max, hard_max = self._limits()
        opened = self._fence_open

        def fence(index: int) -> bool:
            return (opened + _fences(text, index)) % 2 == 1

        if final:
            # The reply is over, so there is no "not yet" -- but there is still
            # a reason to cut. A four-thousand-character tail handed over whole
            # is one synthesis call that cannot be cancelled part-way and one
            # queue entry that cannot start playing until all of it exists, so
            # the ceiling applies to the last segment exactly as to the others
            # -- including the second segment's smaller one, which is the whole
            # point of taking the limits from ``_limits`` on this path too. A
            # long second unit must not escape its envelope merely because
            # generation happened to end before another segment arrived.
            if length <= hard_max:
                # One segment. The tail is what is left of a reply rather than
                # a piece of one, and cutting a 370-character ending into "365
                # characters" and "alpha" buys nothing and costs a synthesis
                # call, a queue entry and an audible seam.
                return length
            for finder in (_paragraph_end, _sentence_at_or_before, _clause_end, _comma_end):
                found = finder(text, min(length, hard_max), fence)
                if found >= target:
                    return found
            word = _word_end(text, min(length, hard_max))
            return word if word >= target else min(length, hard_max)

        # 1. A paragraph break is the strongest boundary there is, and it is
        #    taken as soon as there is one -- a heading followed by a blank line
        #    is a complete thing to say.
        paragraph = _paragraph_end(text, min(length, hard_max), fence)
        if paragraph and len(text[:paragraph].strip()) >= SHORT_SENTENCE:
            return paragraph

        # 2. The opening, before anything else is considered: the first
        #    validated sentence long enough to be one is committed as soon as it
        #    is whole. Not "as soon as it is whole and nothing has arrived
        #    behind it" -- that was the streaming-edge condition this corrects.
        #    A model that emits "Yes, that's possible. Here is the next" in one
        #    delta had written the answer just as early as one that emitted the
        #    sentence on its own, and holding it back for the accident of chunk
        #    boundaries cost the first segment its whole reason for existing.
        if self._segments == 0:
            opening = _first_sentence(text, SHORT_SENTENCE, min(length, hard_max), fence)
            if opening:
                return opening

        # 3. A sentence, once there is enough of a segment to be worth one.
        sentence = _sentence_at_or_before(text, min(length, hard_max), fence)
        if sentence and sentence >= target:
            return sentence

        # 4. A clause, but only once the unsent text is long enough that
        #    waiting for a sentence is the bigger risk.
        if length >= soft_max:
            clause = _clause_end(text, soft_max, fence)
            if clause >= target:
                return clause
            comma = _comma_end(text, soft_max, fence)
            if comma >= target:
                return comma

        # 5. The ceiling. A word boundary, never inside a word, and only when
        #    the text really has run past the hard maximum with no punctuation
        #    in it at all.
        if length >= hard_max:
            if fence(hard_max):
                # Inside a code fence there is no good cut. Wait for the fence
                # to close; the reply's own end will flush it if it never does.
                return 0
            word = _word_end(text, hard_max)
            if word >= target:
                return word
            return hard_max
        return 0


def _covered(wanted: str, spoken: str) -> tuple:
    """How far into ``wanted`` the already-spoken text reaches.

    Compared by non-space characters, so that stripped or collapsed whitespace
    -- the only thing ``clean_reply`` changes about a reply it does not
    relabel -- does not read as a rewrite. Returns the index just past the last
    matched character and whether all of ``spoken`` was accounted for.
    """
    if not spoken.strip():
        return 0, True
    left = right = 0
    matched = 0
    while left < len(wanted) and right < len(spoken):
        if wanted[left].isspace():
            left += 1
            continue
        if spoken[right].isspace():
            right += 1
            continue
        if wanted[left] != spoken[right]:
            return matched, False
        left += 1
        right += 1
        matched = left
    return matched, right >= len(spoken.rstrip())
