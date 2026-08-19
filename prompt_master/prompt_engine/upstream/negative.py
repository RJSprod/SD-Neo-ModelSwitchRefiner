"""
negative.py — Prompt Master LD

The negative prompt, as its own module so it can be reasoned about (and undone)
without touching the brief.

Two halves:

  STATIC   — condition-gated banks. Rule: every term names something a frame
             could actually contain. "slideshow", "teleporting between poses"
             and "third-person view of the viewer" are descriptions of a
             narrative failure, not of pixels; a sampler has no embedding for
             them. They are replaced here by the artifact they actually look
             like (frozen frame, duplicated limbs, camera in frame).

  AUTO     — an optional second LLM pass over the FINISHED script. The static
             bank can only know what the dropdowns said; the script knows what
             the shot committed to. The pass names the visual opposite of those
             commitments (night shot -> daylight, cel animation -> photoreal
             skin, handheld -> locked tripod) and is filtered against the
             script so it can never negate something the shot asked for.
"""

# ── static banks ─────────────────────────────────────────────────────────────
# Artifact names only. Grouped so the reasoning behind each is visible.

_CORE = (
    "blurry, out of focus, low quality, compression artifacts, "
    "deformed hands, extra fingers, fused fingers, extra limbs, "
    "warped face, melted features, "
    "watermark, logo, on-screen text"
)

# Temporal artifacts. What "slideshow" and "jump between frames" LOOK like.
_CORE_TEMPORAL = "frozen frame, flicker, strobing, ghosting, duplicated limbs"

# POV: the failure is the apparatus or a second body becoming visible, so the
# negative names objects, not the concept "third person".
_POV = ("camera in frame, phone in frame, selfie stick, tripod, "
        "camera operator visible, second pair of hands, disembodied arms")

# LTX 2.3 generates audio, so mouth-shape terms and sound terms both carry.
_SILENT = "moving lips, open mouth mid-speech, talking, speech, singing, voiceover"

_UNDRESS = "fully dressed, clothing back on, outfit intact"

_MOTION = "frozen pose, motion smear, stretched limbs, duplicated limbs"

_TRANS_SMOOTH = "hard cut, black frame, split screen"
_TRANS_CUT = "dissolve, cross-fade, double exposure, ghosted overlap"

_FORMAT_TEXT = ("subtitles, captions, closed captions, burned-in text, "
                "timecode overlay, timestamp")

# Transitions whose whole point IS a sharp change — banning cuts fights them.
CUT_TRANSITIONS = {"hard_cut", "whip_pan", "smash_zoom", "flash", "match_cut"}


def core(temporal: bool = True) -> str:
    return _CORE + (", " + _CORE_TEMPORAL if temporal else "")


def pov() -> str:
    return _POV


def silent() -> str:
    return _SILENT


def undress() -> str:
    return _UNDRESS


def motion() -> str:
    return _MOTION


def transition(key: str) -> str:
    k = (key or "off").strip().lower()
    if k == "off":
        return ""
    return _TRANS_CUT if k in CUT_TRANSITIONS else _TRANS_SMOOTH


def format_text(fmt: str) -> str:
    """Layout formats print brackets and numerals the sampler can render as
    burned-in captions. Flowing never had that problem and stays clean."""
    return _FORMAT_TEXT if (fmt or "").strip().lower() in ("bracket", "shotscript") else ""


def dedupe(terms) -> str:
    """Join comma-separated groups, dropping repeats. Repetition in a negative
    weights the repeated concept, exactly as it does in a positive."""
    seen, out = set(), []
    for group in terms:
        for t in str(group or "").split(","):
            t = " ".join(t.split()).strip().lower()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
    return ", ".join(out)


# ── the auto pass ────────────────────────────────────────────────────────────

AUTO_SYSTEM = """You turn a finished video shot description into a NEGATIVE \
prompt for a video sampler.

Output ONE line: 8 to 14 short comma-separated visual terms. Nothing else — no \
sentences, no numbering, no explanation, no "avoid" or "no".

Each term names something a frame could contain and this shot must NOT contain:
* The visual OPPOSITE of what the shot commits to. Night shot -> daylight, \
overcast. Cel animation -> photorealistic skin, live action. Handheld -> \
locked tripod frame. Tight close-up -> wide master shot. Wet skin -> dry matte \
skin.
* The artifact this specific shot invites. A shot with two bodies close \
together risks merged limbs; a shot with a reflection risks a mismatched \
reflection; a shot with fast motion risks stretched limbs.
* Anything the shot forbids in words (an empty room, an unnamed object).

Never list something the shot actually wants. Never list an emotion, a plot \
event, or a rule — only what a camera could see."""


def auto_user(script: str) -> str:
    return ("Shot:\n" + (script or "").strip()
            + "\n\nNegative terms for this shot, one comma-separated line:")


_BANNED_TOKENS = ("negative", "prompt", "avoid", "should", "must", "the shot",
                  "sampler", "terms")


def clean_auto(raw: str, script: str = "", limit: int = 14) -> str:
    """Parse the pass output into safe terms.

    Guards, in order of how badly each failure would hurt:
      1. A term whose words appear in the script is dropped — negating what the
         shot just asked for is worse than having no auto negative at all.
      2. Sentences, meta-talk and rule-restatements are dropped.
      3. Anything over four words is dropped: a long phrase is a sentence
         fragment, and long phrases are exactly the abstractions this module
         exists to stop shipping.
    """
    import re
    t = (raw or "").strip()
    t = re.sub(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", "", t, flags=re.I)
    t = re.sub(r"^```[\w]*\s*|\s*```$", "", t).strip()
    # Keep only the first paragraph — a chatty model adds commentary after.
    t = t.split("\n\n")[0]
    t = t.replace("\n", ", ")
    low_script = " " + " ".join((script or "").lower().split()) + " "

    out, seen = [], set()
    for piece in t.split(","):
        term = " ".join(piece.split()).strip().strip(".;:-–—*•\"'()[]")
        term = re.sub(r"^(?:no|not|avoid|without|never)\s+", "", term, flags=re.I)
        low = term.lower()
        if not low or low in seen:
            continue
        if len(low.split()) > 4 or len(low) < 3:
            continue
        if any(b in low for b in _BANNED_TOKENS):
            continue
        if not re.search(r"[a-z]", low):
            continue
        # Guard 1: never negate what the shot committed to.
        if f" {low} " in low_script:
            continue
        seen.add(low)
        out.append(low)
        if len(out) >= limit:
            break
    return ", ".join(out)


def auto_messages(script: str):
    return [{"role": "system", "content": AUTO_SYSTEM},
            {"role": "user", "content": auto_user(script)}]


def run_auto(script: str, chat_stream, *, seed=None, max_tokens: int = 220,
             limit: int = 14) -> str:
    """Second pass over the finished script. Returns "" on any failure.

    Deliberately never raises: the shot is already written and freeing VRAM is
    next, so a failed negative must not cost the user the generation. Runs
    cooler than the writer (low temperature) because this wants the obvious
    opposite, not an inventive one.
    """
    text = (script or "").strip()
    if not text:
        return ""
    try:
        out = "".join(chat_stream(auto_messages(text), temperature=0.3,
                                 top_p=0.9, max_tokens=int(max_tokens),
                                 seed=seed))
    except Exception as e:
        print(f"[PromptMasterLD] auto-negative pass failed: {e}")
        return ""
    return clean_auto(out, script=text, limit=limit)
