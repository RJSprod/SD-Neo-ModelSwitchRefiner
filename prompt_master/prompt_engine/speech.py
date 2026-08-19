"""Speech expansion — more of the voice the intent already has.

Standalone-only, and off by default. At 1× the intent goes to the engine exactly
as it was typed, which is the behaviour this application has always had. Above
1× a pass over the intent — before the brief is built, not after — writes extra
lines in the voice already quoted there and mixes them back into the intent, so
what the engine receives is a director's request that simply had more speech in
it. The engine itself is untouched: it reads an intent, as it always did.

Two things follow from where the pass sits:

*The lines are the intent's, not the shot's.* They are written from the intent
alone, before anything else exists, so they can only be in the voice the intent
established — which is the point. A brief with no quoted speech has no voice to
match, so lines are invented from the scene instead, and the pass says which of
the two it did.

*The dialogue budget has to move with them.* Upstream tells the model how many
spoken lines a shot of this length should carry, and at 20% that is about two.
Twenty extra lines in the intent against a budget of two is twenty lines the
model is being told to drop. So the multiplier lifts that budget from the user's
own setting toward 100 — never below it, never overwriting the dial, and only
for the generation it applies to.

Like upstream's second pass, this one never raises. The generation is what the
user pressed the button for; a failed expansion costs them extra speech, not the
shot.
"""

from __future__ import annotations

import re
from dataclasses import replace

from prompt_master.core.models import PromptRequest

# 1 is "exactly as written" and the default. 10 is the far end of the slider:
# ten times the lines the intent quoted.
NONE = 1
MOST = 10

# However far the slider goes, the brief still has to fit a context window, and
# a 12-second shot cannot speak thirty lines. The model is asked for what the
# multiplier says, up to this, and the engine decides what survives.
LINE_CEILING = 30

# Upstream requires five words or more per quoted line — a one-word line
# flickers — and nothing here should hand it a line it will have to throw away.
MIN_WORDS = 5
MAX_WORDS = 24

QUOTED = re.compile(r'"([^"\n]{2,240})"|[“„]([^”\n]{2,240})[”“]')

SYSTEM = """You write additional spoken lines for a video shot brief.

Output ONLY the lines, one per line, each wrapped in double quotes. Nothing \
else — no numbering, no names, no attribution, no stage directions, no \
explanation, no blank lines.

Every line is speech a person says out loud in this scene, five words or more, \
plain English, and different from every line already there. Match what the \
existing lines establish: the same speaker or speakers, the same register, the \
same mood, the same subject. Keep to what the brief already contains — invent \
no new characters, no new location, and no event the brief does not have."""

SYSTEM_UNQUOTED = """You write the spoken lines for a video shot brief that has none yet.

Output ONLY the lines, one per line, each wrapped in double quotes. Nothing \
else — no numbering, no names, no attribution, no stage directions, no \
explanation, no blank lines.

Every line is speech a person in this scene says out loud, five words or more, \
plain English, and different from every other line. They belong to the people \
the brief describes, in the mood the brief sets. Keep to what the brief already \
contains — invent no new characters, no new location, and no event the brief \
does not have."""


def quoted_lines(text: str) -> list[str]:
    """The speech already in a brief, in the order it appears."""
    found = []
    for straight, curly in QUOTED.findall(text or ""):
        line = (straight or curly).strip()
        if line and line not in found:
            found.append(line)
    return found


def wanted(multiplier: int, quoted: int) -> int:
    """How many new lines a multiplier asks for.

    ``multiplier`` is a multiple of what the intent already has, so ten times
    two lines is twenty in total and eighteen of them new. A brief with no
    speech is treated as having one line to multiply, since the alternative is
    to multiply nothing and always get nothing.
    """
    if multiplier <= NONE:
        return 0
    total = min(LINE_CEILING, max(quoted, 1) * int(multiplier))
    return max(0, total - quoted)


def dialogue_floor(multiplier: int, dialogue: int) -> int:
    """The dialogue budget lifted in step with the slider.

    Never below what the user set — the dial is a floor, not a value to
    overwrite — and at the far end it is 100, because that is what asking for
    ten times the speech means for how much of the shot is speech.
    """
    current = max(0, min(100, int(dialogue)))
    if multiplier <= NONE:
        return current
    fraction = (min(int(multiplier), MOST) - NONE) / (MOST - NONE)
    return round(current + (100 - current) * fraction)


def messages(intent: str, quoted: list[str], count: int) -> list[dict]:
    system = SYSTEM if quoted else SYSTEM_UNQUOTED
    lines = "\n".join(f'"{line}"' for line in quoted)
    brief = f"BRIEF: {intent.strip()}"
    if quoted:
        brief += f"\n\nLINES ALREADY IN IT:\n{lines}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{brief}\n\nWrite {count} more lines now."}]


def clean_lines(raw: str, existing: list[str], count: int) -> list[str]:
    """The model's reply as usable lines: quoted, new, and the right length."""
    seen = {line.casefold() for line in existing}
    kept: list[str] = []
    for line in (raw or "").splitlines():
        candidates = quoted_lines(line) or [line.strip().lstrip("-*0123456789. )").strip()]
        for candidate in candidates:
            text = candidate.strip().strip('"“”').strip()
            if not text or text.casefold() in seen:
                continue
            if not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
                continue
            # A model that starts explaining itself mid-list: drop the sentence
            # rather than the rest, since the lines after it are usually fine.
            if text.rstrip(".!?").casefold().startswith(("here are", "note", "these lines")):
                continue
            seen.add(text.casefold())
            kept.append(text)
            if len(kept) >= count:
                return kept
    return kept


def mixed(intent: str, extra: list[str]) -> str:
    """The intent with the new lines added to it, as more of the same request.

    They go in as a director's line rather than as prose, because the intent is
    handed to the engine under "Director's request:" and a list of quoted lines
    is exactly what that reads as.
    """
    if not extra:
        return intent
    quoted = " ".join(f'"{line}"' for line in extra)
    return f"{intent.rstrip()}\nAlso use these spoken lines, in the same voices: {quoted}"


def expand(request: PromptRequest, chat_stream, *, seed=None) -> tuple[PromptRequest, str]:
    """``(request, what happened)``. Never raises — see the module docstring.

    The budget is lifted on every path above 1×, including the ones where no
    lines come back. The slider says more speech; the extra lines are one way of
    delivering that and the budget is the other, and a pass that failed is no
    reason to withhold the half that still works.
    """
    multiplier = int(request.speech or NONE)
    if multiplier <= NONE or not (request.intent or "").strip():
        return request, ""
    lifted = replace(request, dialogue=dialogue_floor(multiplier, request.dialogue))

    quoted = quoted_lines(request.intent)
    count = wanted(multiplier, len(quoted))
    if count <= 0:
        return lifted, "already at the line ceiling"

    try:
        raw = "".join(chat_stream(messages(request.intent, quoted, count),
                                  temperature=0.9, top_p=0.95,
                                  max_tokens=64 + 28 * count, seed=seed))
    except Exception as exc:                      # noqa: BLE001 - never costs the shot
        return lifted, f"extra speech skipped ({exc})"

    extra = clean_lines(raw, quoted, count)
    if not extra:
        return lifted, "no extra speech came back"
    source = "in the same voice" if quoted else "written for the scene"
    return replace(lifted, intent=mixed(request.intent, extra)), \
        f"{len(extra)} extra lines {source}"
