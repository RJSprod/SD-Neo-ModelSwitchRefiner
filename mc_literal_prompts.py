"""The two prompt boxes that protect what you put in them.

    Positive Prompt        portrait of a woman
    Negative Prompt        blurry
    ┌──────────────────────────┐ ┌──────────────────────────┐
    │ Literal Positive         │ │ Literal Negative         │
    │ <lora:realfilter:1>      │ │ blue hat                 │
    └──────────────────────────┘ └──────────────────────────┘

    Stage 1 gets:  <lora:realfilter:1> portrait of a woman blue hat

Literal commands already did this. ``[[<lora:realfilter:1>]] portrait of a
woman -[[blue hat]]`` produces the same prompt and has since the feature was
built. What it also produces is a syntax somebody has to learn, in a box that
looks exactly like every other prompt box in the application, where a stray
bracket is a silent typo rather than an error.

So this is a UX layer and nothing else. The two fields are ordinary text; each
one becomes a single :class:`~prompt_master.krea.literals.LiteralCommand`; the
commands join the sidecar the existing parser already produces; and the
existing assembly step puts them back. There is no second parser, no second
prompt path, and no generated bracket string anywhere -- see
:func:`prompt_master.krea.literals.merge`, which is the whole of the mechanism.

What this module owns
---------------------
The two values, and the sentence that keeps them from being invisible. That is
all. It does not assemble a prompt, does not know what a payload is, and does
not decide when the row is on screen.

Why they persist
----------------
The same reason the Spatial canvas does. These are settings a user configures
once -- a filter LoRA they always want, a phrase they always append -- and a
restart that quietly emptied them would change what the next generation
produces without saying so.

That cuts both ways, which is why :func:`active_note` exists. A value that is
still in effect while its row is off screen is exactly the invisible active
state this extension keeps warning itself about, so the Prompt row of the Image
Pipeline says how many there are whenever the row is hidden and they are not
empty. Section 3.3 of the design intent asks for that sentence, and it is the
price of persistence rather than a nicety on top of it.
"""

from __future__ import annotations

import logging

import mc_llm_state

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

POSITIVE = "krea_literal_positive"
NEGATIVE = "krea_literal_negative"
"""The two preferences this feature owns.

Named for the boxes rather than for what they do to a prompt, because that is
what a user reading a preferences file will be looking for. What they *do* is
prefix and suffix -- see :data:`PLACEMENTS` below, which is the mapping stated
once so no caller has to remember which way round they go.
"""


def placements() -> tuple[str, str]:
    """``(placement of the positive field, placement of the negative field)``.

    Literal Negative is a *suffix of the positive prompt*, and it is the one
    thing about this feature somebody is guaranteed to get wrong at first
    reading: it is not Forge's Negative Prompt and it does not remove anything.
    It is the far side of the same protected run of text, which is what
    ``-[[...]]`` has always meant.
    """
    from prompt_master.krea import literals

    return literals.PREFIX, literals.SUFFIX


def settings() -> dict:
    """Both fields, with defaults filled in. Never raises."""
    try:
        stored = mc_llm_state.preferences()
    except Exception:
        logger.debug("Model Chain: could not read the Literal Prompt preferences",
                     exc_info=True)
        stored = {}
    return {"positive": str(stored.get(POSITIVE) or ""),
            "negative": str(stored.get(NEGATIVE) or "")}


def remember(**values) -> None:
    """Keep a Literal Prompt value. Never fatal: this is a convenience."""
    try:
        mc_llm_state.remember(**values)
    except Exception:
        logger.debug("Model Chain: could not save the Literal Prompt preferences",
                     exc_info=True)


def count(positive="", negative="") -> int:
    """How many of the two fields carry anything.

    A field is one command however much text is in it, so this counts fields
    and not words, phrases or commas. The number a user sees has to be the
    number of things that will be inserted.
    """
    return sum(1 for value in (positive, negative) if str(value or "").strip())


def active_note(positive="", negative="") -> str:
    """``2 literals active``, or "" when neither field carries anything.

    Section 3.3. Shown on the Prompt row of the Image Pipeline while the fields
    are off screen, because a value that still reaches the next generation and
    cannot be seen is the kind of state somebody spends an afternoon on.
    """
    found = count(positive, negative)
    if not found:
        return ""
    return f"{found} literal{'' if found == 1 else 's'} active"
