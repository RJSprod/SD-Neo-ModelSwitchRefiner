"""Motion presets — how the shot moves, chosen from three settings.

This is standalone-only. Upstream has one way of writing motion, and **Default
is exactly that**: it appends nothing, adds no negative terms, and samples at
the temperatures ``backend.chat_stream`` uses, so a default generation is
byte-for-byte the generation this application produced before these presets
existed. ``tests/test_prompt_engine.py`` holds that: the system prompt for a
default request must equal ``brain.build_system``'s output exactly.

The other two each pull on the same three levers, and only these three:

* a **directive** appended after upstream's system prompt — never woven into
  it, so removing the preset removes the whole difference;
* **negative terms**, merged into the extra terms the user may already have
  typed, which upstream's ``build_negative`` dedupes against its own banks
  exactly as it does theirs;
* **sampling**, the writer pass only. The smart-negative pass keeps upstream's
  cooler numbers, because that pass is upstream's and its output feeds a bank
  it also owns.

The directives are written in the register of the system prompt they are
appended to — a heading, then imperatives — and neither one asks for more words
than the budget upstream already set. Inertia spends the same words on harder
edges; Flow spends them on how a movement carries rather than on more events.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT = "default"


@dataclass(frozen=True, slots=True)
class Motion:
    key: str
    label: str
    # Appended after upstream's system prompt, or "" for the default.
    directive: str
    # Folded into the user's own extra negative terms.
    negative: str
    # The writer pass only; upstream's own defaults are the ones on default.
    temperature: float
    top_p: float


PRESETS: dict[str, Motion] = {
    DEFAULT: Motion(
        key=DEFAULT,
        label="Default — as the engine writes it",
        directive="",
        negative="",
        # backend.chat_stream's own numbers, which is what "default" means.
        temperature=0.85,
        top_p=0.95,
    ),
    "inertia": Motion(
        key="inertia",
        label="Inertia — abrupt cuts, weight, hard starts and stops",
        directive=(
            "MOTION — INERTIA\n"
            "Weight and abruptness. Movement starts hard and stops hard: a body plants, jolts, "
            "whips around, catches itself late. The camera changes angle by cutting or snapping "
            "to it, never by easing — a hard reframe inside a beat is correct here. Momentum "
            "reads as impact rather than as glide, and what stops, stops on a beat. Spend the "
            "words on the edges of each movement, not on more events.\n"
            "No slow push-ins, no drifting, no dreamlike float, no easing into or out of a move."
        ),
        negative="slow motion, floaty movement, weightless drift, dreamy glide, smooth easing",
        # Hotter, and a narrower nucleus: surprising choices, decisively made.
        temperature=0.95,
        top_p=0.90,
    ),
    "flow": Motion(
        key="flow",
        label="Flow — continuous, smooth, meticulously detailed",
        directive=(
            "MOTION — FLOW\n"
            "Continuity and detail. The camera moves in one unbroken gesture — glide, arc, settle "
            "— and holds the frame rather than cutting. Every action carries: how it begins, how "
            "it travels, how it resolves, each beat handing its momentum to the next. Spend the "
            "words on how the motion carries rather than on more events, so the same beats read "
            "slower and closer.\n"
            "No snap cuts, no jolts, no stutter, no movement that starts or stops without a cause "
            "the eye can see."
        ),
        negative="jerky motion, stutter, jitter, strobing, abrupt cut, snap zoom, camera shake",
        # Cooler and steadier: continuity is consistency, sustained over a page.
        temperature=0.70,
        top_p=0.92,
    ),
}

OPTIONS: list[tuple[str, str]] = [(preset.key, preset.label) for preset in PRESETS.values()]


def preset(key: str | None) -> Motion:
    """The preset for a key, falling back to default rather than raising.

    A state file or a saved request from another version can carry a key this
    build does not know, and the right answer to that is upstream's behaviour.
    """
    return PRESETS.get((key or DEFAULT).strip().lower(), PRESETS[DEFAULT])


def applied(system: str, key: str | None) -> str:
    """Upstream's system prompt, with the preset's directive after it."""
    directive = preset(key).directive
    return f"{system}\n\n{directive}" if directive else system


def with_terms(key: str | None, extra: str) -> str:
    """The user's extra negative terms with the preset's own folded in.

    Concatenation is enough: these go to ``build_negative(extra=…)``, whose
    dedupe is what decides a term stated twice does not get weighted twice.
    """
    parts = [part.strip() for part in (extra or "", preset(key).negative) if part and part.strip()]
    return ", ".join(parts)
