"""
shotscript.py — Claude Prompt LD

Output FORMAT only. Content laws live in brain.py; this module is layout.

  flowing    — blank-line prose beats (default)
  bracket    — prose beats; spoken lines on their own row as (delivery) "line"
  shotscript — one setup paragraph, then timed windows [0-N] of short prose
"""

FORMATS = ["flowing", "bracket", "shotscript"]

FORMAT_LABELS = {
    "flowing": "Flowing prose (default)",
    "bracket": "Bracketed dialogue",
    "shotscript": "Shot script [0-3]",
}


def _windows(seconds: float, n_beats: int):
    """Contiguous whole-second windows covering the clip. Setup is untimed."""
    total = max(2, int(round(float(seconds or 12))))
    n = max(1, int(n_beats))
    n = max(1, min(n, max(1, total - 1)))
    if n == 1:
        return [(0, total)]
    edges, step = [0], total / n
    for i in range(1, n):
        edges.append(int(round(step * i)))
    edges.append(total)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    edges[-1] = max(edges[-1], edges[-2] + 1)
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def contract(fmt: str, *, nb_lo: int, nb_hi: int, lo: int, hi: int,
             seconds: float) -> str:
    f = (fmt or "flowing").strip().lower()

    if f == "bracket":
        return (
            "\nOUTPUT CONTRACT\n"
            f"Return {nb_lo}-{nb_hi} BEATS of present-tense prose, {lo}-{hi} "
            "words total, nothing else. Beats = short paragraphs separated by "
            "one blank line.\n"
            "Spoken lines on their OWN line: (delivery) \"line\". Bracket holds "
            "delivery only — tone, volume, pace — never spoken, never action. "
            "Punctuation inside quotes. Prose around the line is normal.\n"
            "No title, headings, timestamps, beat labels, or notes. First word "
            "is the first word of the prompt."
        )

    if f == "shotscript":
        wins = _windows(seconds, max(nb_lo, min(nb_hi, nb_lo + 1)))
        total = int(round(float(seconds or 12)))
        sample = "  ".join(f"[{a}-{b}]" for a, b in wins)
        last = wins[-1]
        return (
            "\nOUTPUT CONTRACT\n"
            f"Return a SHOT SCRIPT, {lo}-{hi} words total, nothing else. "
            "Two parts only.\n"
            "\n"
            "PART 1 — FOOTPRINT (one paragraph, no time tag).\n"
            "Opening footprint: who, look, clothes, place, light, camera "
            "distance/lens, quiet ambient of the place (no score/radio unless "
            "MUSIC names one), pose already underway. No dialogue. No timed "
            "events. Frame one of the clip; everything after is change.\n"
            "\n"
            "PART 2 — TIMED BEATS (cover every second).\n"
            f"After one blank line: {len(wins)} beats in order: {sample}. "
            f"Do not stop until [{last[0]}-{last[1]}]. Windows span all "
            f"{total} seconds.\n"
            f"Under each tag (e.g. [{wins[0][0]}-{wins[0][1]}]): 2–5 short "
            "present-tense sentences, then blank line, then next tag.\n"
            "Order of attention (guide for the prose, not a form — never pad):\n"
            "  1. Camera / frame — move or hold\n"
            "  2. Movement — body action\n"
            "  3. Reaction — what answers (other body, fabric, light, surface)\n"
            "  4. Sound — noise inside the action (not a score unless MUSIC)\n"
            "  5. Dialogue — at most one short line if earned\n"
            "Plain prose under timestamps. Never print stage labels or field "
            "names (no CAPS captions). Do not re-describe face/wardrobe/place "
            "in timed beats — write only what changes. Spoken words are heard, "
            "not shown. One quote pair per line.\n"
            "HARD: response IS the shot script. No product name, summary, rule "
            "bullets, self-notes, 'Correction:', or headings Footprint:/Timed "
            "Beats:. First word = first word of the footprint."
        )

    return (
        "\nOUTPUT CONTRACT\n"
        f"Return {nb_lo}-{nb_hi} BEATS of flowing present-tense prose — short "
        f"paragraphs separated by one blank line, {lo}-{hi} words total — and "
        "nothing else. Prefer the middle of the beat range when unsure.\n"
        "HARD: the entire response IS the shot. No product name, summary, rule "
        "bullets, plan, self-notes, or labels like Beat 1 / Action / Camera. "
        "No title, headings, timestamps, beat labels, or notes. First word is "
        "the first word of the first beat."
    )


def negatives(fmt: str) -> str:
    """Kept for callers; the vocabulary now lives in negative.py so the caption
    guard is stated in exactly one place."""
    from .negative import format_text
    return format_text(fmt)
