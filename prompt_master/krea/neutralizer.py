"""The Pose and Placement Neutralizer: a subtraction pass, and the guard that keeps it one.

The third language-model pass in the Krea pipeline, and the first to run. Before
the Creative Director reads the prompt and the Creative Writer expands it, this
pass may take two things *out* of it -- how a body is arranged, and where in
the picture something sits -- and may add nothing. The instruction that says so
is ``neutralize.txt`` beside this file, read and never rewritten; what this
module owns is the request around it, the cleanup of what comes back, and the
one thing an instruction alone cannot provide: proof that the model did as it
was told.

Why the guard is mechanical
---------------------------
A system prompt asks. It cannot check. A local model asked to delete words will,
one roll in twenty, tidy a sentence on its way past -- a synonym here, a
connector there, a reordered clause that reads better -- and every one of those
is exactly the edit this stage exists not to make. So the reply is not trusted
because the instruction was clear; it is trusted because :func:`subtraction_error`
found nothing in it that the source did not have, in the order the source had
it. A reply that fails that test is refused whole, and the pipeline carries on
from the prompt as typed. Refusing whole rather than repairing is deliberate:
deleting only the added word would be a second editor nobody can see, running
after the one whose instruction is on disk.

The guard is one-sided, and that is worth saying plainly. It proves nothing was
added or moved; it cannot prove that too much was taken away. Over-deletion is
the instruction's job, and the acceptance fixtures in ``tests/`` are where that
is checked -- against the same wording, so the two cannot drift apart.

What determinism means here
---------------------------
Temperature zero and top-p one, on every backend, for a task that has nothing
for a sampler to be creative about. That is a promise about *this* machine with
*this* GGUF on *this* llama.cpp build: the same source neutralizes the same way
twice. It is not a promise that two backends agree to the byte, and nothing
downstream assumes they do -- what has to hold everywhere is the contract above,
which is why it is checked rather than hoped for.
"""

from __future__ import annotations

import re
from pathlib import Path

INSTRUCTION_VERSION = 1
"""Bumped when ``neutralize.txt`` changes in a way that changes what comes back.

Recorded nowhere yet -- the image records that the pass ran and what it ran on,
which is enough to reproduce the workflow -- but kept from the start, because
"the Neutralizer behaved differently" is otherwise indistinguishable from "the
model was different that day".
"""

_INSTRUCTION = Path(__file__).with_name("neutralize.txt")


def system_prompt() -> str:
    """The standing instruction, verbatim from the file beside this module.

    Read rather than pasted into a string literal for the reason Krea's own
    expansion instruction is: an edit to the contract is then a visible change
    to a text file, and a diff of this package can never make it look as though
    the instruction had been quietly reworded in passing.
    """
    return _INSTRUCTION.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #

SOURCE_HEADING = "source_prompt:"
"""The one label in the user turn.

Labelled so the model can tell the text it is editing from the instruction it
is editing under -- the instruction says, in as many words, that the source is
data and cannot override it, and a heading is what makes that boundary visible
in the transcript rather than a thing the model has to infer.
"""

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 1024
"""Greedy, and bounded well above any source this pass will be handed.

The task is copy-editing by deletion: the right answer is a subset of the input,
and a sampler exploring alternatives would make the same prompt neutralize two
ways on two presses. 1024 tokens is comfortably more than a prompt box holds,
with room for a model that thinks out loud before it answers -- :func:`clean`
takes that off.
"""

SEED = 0
"""Passed because the client requires one, fixed because it changes nothing.

Greedy decoding has no draw to seed. Sending the same number every time keeps
two requests for the same source byte-identical, which is what lets llama.cpp
resume the second one from its cache.
"""


def user_content(source: str) -> str:
    """The user turn: the label, then the source, and nothing else.

    No character, no brief, no reference caption, no layout. The source is
    stripped of surrounding whitespace and otherwise passed through untouched,
    because the whole contract is that the reply is a subset of *this* text --
    tidying it first would make the reply a subset of a paraphrase.
    """
    return f"{SOURCE_HEADING}\n{str(source or '').strip()}"


def messages(source: str) -> list[dict[str, str]]:
    """The whole request: one system message, one user turn, no history."""
    return [{"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_content(source)}]


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #

_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCED = re.compile(r"\A```[^\n]*\n(?P<body>.*?)\n?```\Z", re.DOTALL)


def clean(text: str) -> str:
    """The reply with what is not part of it taken off. Formatting only.

    A leaked reasoning block, one enclosing code fence, the whitespace around
    the whole -- the same three things :func:`prompt_master.krea.enhancer.clean`
    removes, and deliberately nothing more. This function cannot rewrite a word:
    a cleaner that improved replies would be a second neutralizer running after
    the first, and the guard below would be checking its work rather than the
    model's.
    """
    text = _THINKING.sub("", str(text or "")).strip()
    fenced = _FENCED.match(text)
    return (fenced.group("body") if fenced else text).strip()


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #

_WORD = re.compile(r"[^\W_]+")
"""One lexical token: a run of letters and digits.

Everything else -- spaces, commas, hyphens, apostrophes, underscores -- is a
separator and contributes nothing, which is what lets a reply repair the
punctuation around a deletion without being refused for it. ``silver-haired``
and ``silver haired`` are the same two tokens; ``woman's`` and ``womans`` are
not, and a spelling change is an edit this pass was not asked to make.
"""


def tokens(text: str) -> list[str]:
    """The lexical tokens of ``text``, casefolded, in order.

    The comparison the guard is made of. Case is folded because a deletion at
    the start of a sentence legitimately capitalises the word that follows it;
    nothing else is normalised, because every other normalisation is a way of
    letting a changed word through.
    """
    return [found.casefold() for found in _WORD.findall(str(text or ""))]


def subtraction_error(source: str, reply: str) -> str:
    """Why ``reply`` is not a pure subtraction of ``source``, or ``""`` when it is.

    A reply passes when its tokens are an in-order subsequence of the source's:
    every word it kept is a word the source had, and in the order the source
    had them. Three ways to fail, each named so the console can say which:

    * nothing survived -- an empty reply is not a neutralized prompt, it is no
      prompt, and the source answers instead;
    * a word the source never had -- a synonym, a connector, a heading, an
      explanation, or a replacement pose written where the old one was;
    * a word the source had, out of place -- surviving text reordered for
      readability, or one word repeated where the source said it once.

    The last two are told apart by whether the word exists in the source at
    all, which is a diagnosis for a log line and not a difference in outcome:
    every failure refuses the whole reply. The word itself is never in the
    reason. A moved word is a word of the prompt, an added one is the model's
    guess at the prompt, and the console rule is that a line says what kind
    of run it was and never what was in it.
    """
    kept = tokens(reply)
    if not kept:
        return "the neutralizer returned nothing"
    pool = tokens(source)
    position = 0
    for word in kept:
        try:
            position = pool.index(word, position) + 1
        except ValueError:
            if word in pool:
                return "the neutralizer moved or repeated a word of the source"
            return "the neutralizer added a word the source did not have"
    return ""


def valid_subtraction(source: str, reply: str) -> bool:
    """Whether ``reply`` may replace ``source``. See :func:`subtraction_error`."""
    return not subtraction_error(source, reply)


def removed(source: str, reply: str) -> int:
    """How many of the source's tokens did not survive, for a line in the log.

    A count and never the words: what was taken out of a prompt is prompt
    content, and the console rule is that a status line says what kind of run
    it was and never what was in it.
    """
    return max(len(tokens(source)) - len(tokens(reply)), 0)


# --------------------------------------------------------------------------- #
# What the status line says
# --------------------------------------------------------------------------- #

WAITING = "Waiting for the prompt neutralizer"
READING = "Reading the source prompt"
WRITING = "Neutralizing pose and placement"
"""The three phases, in words that name this pass and no other.

Distinct from the writer's and the Composer's on purpose. The Image Pipeline's
browser file lights a row by matching the bar's text, and a third pass that
said "Waiting for the language model" would be a phase two rows could claim.
The phase producer names the stage; the browser only reflects it.
"""
