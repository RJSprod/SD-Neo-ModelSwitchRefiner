"""
brain.py — Claude Prompt LD
============================
Director's brief for a literal video renderer (LTX). Keep it short: physics,
output contract, word budget, mode openers. Conditional modules only when used.
"""

import re as _re

from .accents import (accent_kind, accent_label, accent_note,
                      accent_words, density_rule, strength_law)
from . import negative as _neg
from .hands import hand_law, hand_negative
from .identity import identity_block
from .styles import style_law, style_negative, style_opener
from .shotscript import contract as _fmt_contract
from .cinematics import camera_law, camera_negative, transition_law
from .music import music_block, performance_note
from .wardrobe import UNDRESS, infer_who, underwear_block, wardrobe_block


# ── word budget ───────────────────────────────────────────────────────────────
# Base ~20-32 words per write-second; Talk dial densifies toward ~26-38 so
# mid-length clips can hold ~4 shorter beats + continuous dialogue.

def word_budget(seconds: float, dialogue="some"):
    """Word band scales with write-seconds. Soft caps keep very long clips sane
    without flattening 3s and 50s into the same ~400-word ceiling."""
    s = max(1.0, float(seconds or 12.0))
    pct = talk_pct(dialogue)
    lo = max(100, min(int(s * 20), 1400))
    hi = min(2000, int(s * 32))
    if pct > 0:
        t = pct / 100.0
        lo = max(lo, int(s * (20 + 6 * t)))
        hi = min(2400, max(hi, int(s * (32 + 6 * t))))
    if hi - lo < 60:
        hi = lo + 60
    return lo, hi


def write_seconds(seconds: float) -> float:
    """The script covers the WHOLE clip. No padding.

    This used to subtract 2s (or 4s over 12s) on the theory that the encoder
    clips the ending. That is not a mechanism LTX has, and the cost was real:
    a 16s clip was budgeted, beat-counted and window-timed as 12s, so the last
    quarter of every long shot arrived undirected. Lightricks' own guidance
    runs the other way — a prompt short for its duration makes the model rush
    the described action and coast through what is left.

    Kept as a function (rather than deleted) because every budget in this file
    and in shotscript.py routes through it, and because both formats now agree
    on the same number, which is what makes a format A/B a fair comparison.
    """
    return max(2.0, float(seconds or 12.0))


def max_tokens(seconds: float, dialogue="some", fmt: str = "flowing") -> int:
    """Response ceiling scales with write time and the talk dial.

    fmt is retained for call-site compatibility; both formats now budget on
    full wall seconds, so it no longer changes the answer.
    """
    lo, hi = word_budget(write_seconds(seconds), dialogue)
    base = 280 + hi * 3
    pct = talk_pct(dialogue)
    base = int(base * (1.0 + pct / 100 * 0.5))
    return min(8000, max(500, base))


def clean_script(text: str) -> str:
    """Strip reasoning leaks, fences, and meta-planning the model dumps instead of a shot."""
    import re
    t = (text or "").strip()
    t = re.sub(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", "", t, flags=re.I)
    t = re.sub(r"^```[\w]*\s*|\s*```$", "", t)
    # Drop leading product/echo lines
    t = re.sub(r"^(?:LTX(?:-Video)?(?:\s*2\.3)?|OUTPUT CONTRACT|THE SHOT)\s*\.?\s*\n+",
               "", t, flags=re.I)
    # Drop whole-line meta / planning (common failure: model outlines the brief)
    kill = re.compile(
        r"(?im)^(?:\s*[\*\-•]\s*)?(?:"
        r"Note\s*:|Correction\s*:|Footprint\s*\*?|Timed\s+Beats?\s*\*?|"
        r"Beat\s+\d+\s*:|No speech\.?|No speech\b|"
        r"\d+[-–]\d+\s+beats?\.?|\d+[-–]\d+\s+words\.?|"
        r"One continuous take\.?|from (?:the )?(?:identity|wardrobe|image)\b|"
        r"I (?:will|must|should|need to)\b|I'll adapt\b|"
        r"ground truth|respect the image|"
        r"(?:Camera|Action|Light|Lighting|Sound|Motion|Setting|Wardrobe|"
        r"Woman|Dialogue|Detail|Reaction)\s*:\s|"
        r"Must follow\b|No euphemisms\b"
        r").*$"
    )
    lines = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if kill.match(line):
            continue
        # bullet lines that are pure rule restatements (short, instructional)
        if re.match(r"^[\*\-•]\s+", s) and (
            re.search(r"\b(?:from image|from identity|from wardrobe|beats?\.|words\.|No speech)\b", s, re.I)
            or re.match(r"^[\*\-•]\s*(?:Camera|Action|Light|Sound|Motion|Setting|Wardrobe)\b", s, re.I)
        ):
            continue
        lines.append(line)
    t = "\n".join(lines)
    t = re.sub(r"\*([^*\n]{1,40})\*", r"\1", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # If cleaning wiped everything, return original stripped (better than empty)
    out = t.strip()
    return out if out else (text or "").strip()


def beat_budget(write_secs: float):
    """Blank-line beats: ~one per 3s of write time. Long clips get more beats
    (was hard-capped at 6–7, so 50s looked like a 12s shot)."""
    w = max(2.0, float(write_secs or 10.0))
    lo = max(2, min(14, round(w / 3)))
    hi = min(18, max(lo + 1, round(w / 2.5)))
    return lo, hi


def frame_count(fps: int, seconds: float) -> int:
    """LTX wants 8n+1 frames. Snap to the nearest valid count."""
    raw = max(9, round(max(1, int(fps or 24)) * max(1.0, float(seconds or 12.0))))
    n = round((raw - 1) / 8)
    return int(8 * max(1, n) + 1)


# ── the brief ─────────────────────────────────────────────────────────────────

_CORE = """You write finished shot text for a literal video renderer. \
What you name appears; what you omit does not exist. No plans, outlines, \
checklists, notes, or corrections — only the finished shot the renderer runs."""

_LIGHT = """
LIGHT & OPTICS
Name light: direction, quality, one effect (rakes skin, pools on floor, edges \
a silhouette). Name one optical property (shallow depth, rack focus, grain). \
Unnamed light and optics render flat and deep."""

_SHOT = """
THE SHOT — {sec:g} seconds
One continuous take unless you write a cut. Fit {sec:g}s of picture and sound: one \
or two developments, not a story. Show, don't narrate: heat is shimmer and \
sweat, cold is breath-fog, feeling is what the face and body do. Scent, \
taste and inner thought have no picture.
Beats: short chunks of camera time, 3-5 sentences carrying contact, motion, \
light change, or speech. First beat anchors. Speech gets its own moment, \
never stapled to the end of an action sentence.
Motion: name the part that moves and what it moves against, off, or onto. \
Poses chain — weight shifts, limbs travel, positions are arrived at. Nothing \
teleports.
The living detail is never blank: everyone on screen carries expression, eye \
direction, skin state, hair motion, and weight on a surface. Specifics, not \
adjectives — "sweat beading at her temple", not "she looks intense". Let \
them shift across the shot.
Camera: place it once, then one deliberate move (push, drift, tilt, settle) \
unless a CAMERA law below overrides — including holding still. Eye contact \
means looking into the view; a lens, phone, or camera body only exists if \
the intent names one.
Sound is diegetic — cloth, breath, room, bodies. A score or radio exists \
only if a MUSIC block names it.
Cast only who the intent gives you: solo stays solo. Background people come \
with the setting (club crowd, street shoppers) and stay in the background.
Close on a held image — a look, a pose, a fall of light."""

# Fires when bodies are bare or the intent is explicit. On a clothed shot this
# is dead weight that spends attention on words the model will never need.
_SHOT_ANATOMY = """
ANATOMY
Plain words for what the lens sees. Bare breasts are "breasts" — not chest, \
curves, or figure. Euphemism and romance stand-ins (her core, his length, \
her center, his member) name nothing, so nothing renders; an act whose \
anatomy goes unnamed does not happen. Clothed is garment plus shape; bare \
is body parts."""

# Fires when someone turns, or is positioned relative to the lens.
_SHOT_ORIENT = """
ORIENTATION
Facing is a fact — state which way torso and head point (toward the view, \
away, profile). Action agrees with facing. A turn is rotation you can watch, \
never one figure facing two ways."""

_SPEECH = """
SPEECH
Quoted lines are the only lines that lip-sync — attribution + quote. Stop-beat \
before speech, line, then action resumes. Delivery lives in the attribution. \
Plain English in quotes unless a VOICE law owns spelling.
Between lines: short involuntary vocal sounds (caught breath, throat sound, \
half-laugh) when earned — vary them; they punctuate lines, never replace them.
Every quoted line is five words or more (one-word lines flicker). Occupied \
mouth (kiss, food, drink): hum/muffle only until released."""

_SPEECH_COUNT = """Write about {n} short spoken lines across the shot, each \
in its own beat, spaced so action still breathes. Every line costs a stop-beat."""

_PERFORMANCE = """This is a PERFORMED VOCAL shot — the voice is the through-\
line. Near-continuous quoted delivery (rap/sung/spoken-word) in runs of 2–4 \
lines. Most beats must carry quoted voice; long mute stretches fail this mode. \
The stop-beat rule is relaxed: body performs around the vocal. Break only for \
breath, look, or hit. Short varied ad-libs between bars — never a repeated \
catchphrase. Every sung or spoken word is written for this shot: never a real \
song's title, lyric, or hook."""

# Rule 3 comes in two forms because the accent dial covers two different things.
# For a second-language speaker, native vocabulary is contamination and the cap
# is the fix. For a variety of English it is the opposite: the vocabulary IS the
# voice, and the cap was deleting the only recognisable part of it.
_VOICE_ONE_LANE = (
    "3) ONE LANE. Intent language leads (usually English). At most ONE short "
    "native slip in the shot, zero is fine — never stacked, and never a sound "
    "or endearment from another accent."
)

_VOICE_REGISTER = (
    "3) REGISTER. This is a variety of English, not a foreign accent laid over "
    "it. Its own words, idioms and sentence shapes ARE the voice — a line built "
    "from standard English with a few letters swapped is the weak version of "
    "it. Use this variety's real vocabulary and grammar wherever they fall "
    "naturally, several times across the shot, in ordinary speech and in the "
    "greetings, address terms and throwaway reactions where a dialect shows "
    "most. A word this variety genuinely uses, spelled the way that variety "
    "normally spells it, carries the voice further than a standard word with "
    "letters knocked out of it. Stay inside THIS variety: nothing borrowed from "
    "another accent, and never a whole line in a different language."
)

# Fires only when the request supplied the words. Two separate failures were
# landing on the same shot: the model preserved the given line exactly (correct
# for meaning, wrong for accent), and the given line happened to contain nothing
# that accent bends — so even at thick there was nothing for density to act on
# and it had no permission to add anything either.
_GIVEN_LINE = """
WORDS FROM THE REQUEST
The request supplies what is said, so keep those words' meaning and order — but \
the accent still lands ON them: respell what it bends, and put its tags, \
particles or address terms AROUND the line without changing what it says. If the \
given words bend nothing, it shows in the attribution, in an added particle, in \
the non-words, or in a stressed word set in CAPS inside the quote. Flat standard \
English is a failed line, not a faithful one."""

_VOICE_PLAIN = """
VOICE (no accent chosen)
No accent does not mean no voice. An unnamed voice comes back as flat \
synthetic narration with no breath in it, because unnamed renders as generic in \
sound exactly as it does in picture. So give the voice a body the first time \
someone speaks: register (low, high, mid), grain (smooth, husky, rasping, \
breathy, nasal, thin), and pace. Keep that same voice for the whole shot, and \
let breath be audible where a line runs long — a real voice takes air."""

_SILENT = """
NO SPEECH
Nobody speaks. No quoted lines, mouthed words, or whispers. Soundtrack is \
breath, body, cloth, and room."""

_T2V_OPEN = """
OPENING (text-to-video)
Nothing exists until you write it. The opening sentence establishes place, \
light, and who is in frame — after any opening words a \
first-person or style law below reserves, and those laws own position one \
where they apply. Named venues \
come populated — a club has other dancers and a crowd in the low background, a \
beach has people and water, a street has traffic — plus ambient sound. Empty \
rooms read as voids unless the intent wants solitude."""

# Rewritten. The old version spent nineteen instructions on DESCRIBING the
# still and fourteen on moving it — backwards for the one mode where appearance
# is already free. Frame one is pinned by the image latent, so every word that
# re-describes it is a commitment that can disagree with the pixels, and a wrong
# name is a competing signal (this brief's own "what you name appears" law cuts
# both ways). Lightricks say it plainly for i2v: describe the motion, not the
# static elements already visible, and describe the move out of stillness.
_I2V_OPEN = """
OPENING (image-to-video)
Frame one is the attached image. It already holds who, their clothes, body, \
face and place — you do not have to write any of that into being, and every \
word that re-describes it is a chance to contradict what is actually there.
IDENTIFY MINIMALLY, THEN MOVE. Sentence one names the MEDIUM you see (2D \
cartoon, comic illustration, anime cel, 3D render, live-action photograph, \
painterly art) and identifies each person only as far as you need to refer to \
them again ("the woman in the red coat"). Everything after that is motion: \
sentence one must already move something visible in the still.
Do not inventory the outfit, redesign a face, humanize a drawn character, or \
swap a garment. If the intent asks for something the still does not show, it \
arrives by MOVING into frame, never by respecifying frame one.
ANCHOR — every beat: the world of the still persists. However many people are \
in frame, each stays the same person from first frame to last — same face, \
build, hair, clothes, place, light, and medium. They change by moving, and only \
where the request asks. (A first-person viewer is not one of these people.)"""

# Fires when there is no image on the wire: no mmproj, vision unsupported, or
# the vision-fail retry path. The old code appended the full "Frame one is the
# attached image" block unconditionally — several hundred emphatic characters
# against one contrary line in the user turn — so the model confabulated a still
# it could not see and LTX got a prompt describing a different scene than the
# one in frame one. That is the "describes the image then drifts away" report.
_I2V_OPEN_BLIND = """
OPENING (image-to-video — still NOT visible to you)
Frame one is a still you cannot see. Do not describe it, name its medium, or \
invent its contents: any guess you write becomes a competing signal against the \
real first frame.
Write MOTION ONLY — what moves, in what direction, against what, and what the \
camera does. Refer to people by role, not by appearance ("she", "the man at the \
counter"), and never assign hair, skin, clothing, or a place. The still supplies \
all of that; your job is the change across the clip."""

_I2V_CARTOON = """
MEDIUM (detector: still is cartoon / flat illustration)
Open with 2D cartoon / comic illustration language. Hold drawn medium for \
the whole clip — cel color, outlines, illustrated anatomy. No photoreal \
pores, live-action bodies, or photo lens morph."""

_I2V_LOOK = """
LOOK (image-to-video)
Appearance = the still only. No ethnicity/body CAST seed over the image. \
Accent (if any) is VOICE only — never open "An Australian woman…" (or any \
nationality stock lead) unless the still or intent already says that. Match \
the character design in the image (costume colors, proportions, drawn vs real)."""

_POV_RULES = """Camera = viewer's eyes — a {vg}, never described as a third-\
person body. HARD RULE: no face, torso, legs, or name for the viewer — never \
"a man walks", "he steps", "his chest" as a figure in shot. The hands are the only visible \
part of the viewer and POV HANDS below governs them; any body part beyond hands \
ruins POV, the mirror reflection being the one exception VIEWER LOOK / MIRROR \
allow. The living-detail law covers the people on screen \
only — never invent a face, eyes, or expression for a first-person viewer.
View path = head motion (lean, sway, bob), not a rig. Closing distance grows \
the subject in frame — never a body walking into shot.
Contact: her body fills the frame (arms past the view, cheek, weight tilt). \
Closest contact fills the whole frame; a mouth shows shape (parted/pursed) \
before it lands. She reaches the viewer — never a lens, camera, glass, or \
screen. Eye contact = with the viewer.
Invent no companion unless the intent names one. Viewer-alone intents: hands \
+ world; the environment is the subject. First-person "I do X" belongs to the \
VIEWER — render via hands (and mirror reflection if present)."""

_POV_T2V = """
FIRST PERSON
POV shot. First words exactly: "POV shot, first-person camera". """ + _POV_RULES

_POV_I2V = """
FIRST PERSON
POV shot; still decides frame one. If still is not first-person, reach POV by \
motion toward subject or one written cut — never fight the open. """ + _POV_RULES

_POV_VOICE = """Viewer speech: quotes attributed to "the viewer" — close and \
low at the mic — never to a described man or woman. If a VOICE accent law is \
active, it is the VIEWER's accent first (the person looking through the camera).
WHO SPEAKS: only a mouth on screen can lip-sync. The viewer has no mouth in \
frame, so a viewer line is off-screen voice with nothing to sync to. The person \
on screen therefore carries most of the dialogue: for every line the viewer \
speaks there are at least two from a mouth in frame, and the FIRST line of the \
shot belongs to a mouth in frame, never to the viewer. No accent, style or music \
rule is a reason to hand the viewer another line."""

# The same law with nobody else in the shot. Its own block rather than a caveat,
# because a caveat on a ratio is exactly what got traded away last time.
_POV_VOICE_SOLO = """Viewer speech: quotes attributed to "the viewer" — close \
and low at the mic — never to a described man or woman. If a VOICE accent law is \
active, it is the VIEWER's accent.
WHO SPEAKS: nobody else is in this shot, so the viewer is the ONLY voice. There \
is no ratio to meet here, and NOTHING in this brief is a reason to invent \
someone to speak — no passenger, no companion, no bystander, no voice from off \
frame. Adding a person the request did not ask for is a worse failure than a \
quiet shot.
Alone, a person says little: a word to themselves, a shout at what is in front \
of them, a breath that is not a word at all. Keep it sparse and let beats carry \
no voice — the engine, the wind, the room is the sound."""

_FPS_HIGH = """
PACING
High frame rate — continuous flowing motion over held poses."""

_FPS_LOW = """
PACING
Low frame rate — large deliberate moves over fine flutter."""


def talk_pct(dialogue) -> int:
    """Dialogue dial 0-100. Accepts legacy strings from old workflows."""
    if isinstance(dialogue, str):
        d = dialogue.strip().lower()
        legacy = {"silent": 0, "none": 0, "off": 0, "some": 20, "a lot": 45,
                  "alot": 45, "lots": 45, "talkative": 45}
        if d in legacy:
            return legacy[d]
        try:
            return max(0, min(100, int(float(d))))
        except ValueError:
            return 20
    try:
        return max(0, min(100, int(dialogue)))
    except (TypeError, ValueError):
        return 20


def dialogue_lines(write_secs: float, pct: int, beats: int = 0) -> int:
    """Spoken-line target from the dial, CAPPED BY TIME.

    The dial alone is a density knob with no idea how long a line takes to say.
    A quoted line is five words minimum by the SPEECH law, realistically six to
    eight, which is two to three seconds of speech plus the stop-beat before
    it. So the honest ceiling is about one line per three seconds of clip, and
    the old formula blew straight through it: 12s at the DEFAULT 20% asked for
    four lines — near-continuous talking — and 50% asked for nine, which is
    roughly twenty seconds of speech in a twelve second shot. The model can
    only resolve that by rushing the delivery or dropping half the lines.

    Second cap: the instruction says each line gets its own beat, so a target
    above the low beat count is unsatisfiable as written.

    Performance mode (70%+) is allowed one per two seconds: bars are shorter,
    they ride over the action instead of stopping it, and that mode explicitly
    relaxes the stop-beat rule.
    """
    w = max(2.0, float(write_secs))
    p = max(0, min(100, int(pct or 0)))
    if p == 0:
        return 0
    # Slots = how many lines the clock can actually hold. Performance mode gets
    # more because its bars are shorter and ride over the action rather than
    # stopping it.
    slots = max(1, int(w / (2.0 if p >= 70 else 3.0)))
    # The dial now reads as "how much of the available speech do you want",
    # reaching the ceiling at the point the UI already warns about (50%). The
    # old curve put the DEFAULT (20%) straight at the ceiling and then kept
    # counting past it, which is how 12s ended up asking for nine lines.
    out = min(slots, max(1, round(slots * p / 100.0 * 2.0)))
    if beats:
        out = min(out, max(1, int(beats)))
    return max(1, out)


_SPEECH_VERB = _re.compile(
    r"(?<![A-Za-z])(?:says?|said|tells?|told|shouts?|yells?|whispers?|mutters?|"
    r"asks?|screams?|sings?|calls? out|goes)(?![A-Za-z])\s+\S+\s+\S+", _re.I)


def intent_supplies_line(intent: str) -> bool:
    """True when the request hands over the actual words to be spoken.

    "a man in a car says That lora daddy isnt using image to video either" is a
    supplied line even with no quote marks around it. This matters because the
    model treats those words as fixed text and reproduces them verbatim — which
    is right for the CONTENT and wrong for the accent, so a thick Canadian shot
    came back as standard English with a delivery note and nothing else.
    """
    it = intent or ""
    if any(q in it for q in ('"', "\u201c", "\u201d")):
        return True
    return bool(_SPEECH_VERB.search(it))


_PERSON_CUES = _re.compile(
    r"(?i)(?<![A-Za-z])(?:she|her|hers|he|him|his|they|them|woman|women|man|men|"
    r"girl|girls|boy|boys|lady|ladies|guy|guys|dude|bloke|friend|friends|mate|"
    r"mates|partner|wife|husband|girlfriend|boyfriend|crowd|people|team|group|"
    r"crew|band|couple|stranger|barman|bartender|waitress|driver|passenger|"
    r"dancer|dancers|singer|someone|somebody|another)(?![A-Za-z])")


def intent_names_other_person(intent: str) -> bool:
    """True when the request puts someone besides the viewer on screen.

    WHO SPEAKS demands most dialogue come from a mouth in frame, and it was
    written assuming there IS one. "pov in a f1 car going through new york really
    fast" names nobody — so a ratio the model could not satisfy any other way
    made it invent a passenger, in a car with one seat, and give her three of the
    five lines. The ratio has to know whether anyone is actually there.
    """
    return bool(_PERSON_CUES.search(intent or ""))


def accent_in_intent(accent: str, intent: str) -> bool:
    """True when the intent itself puts this accent in the scene.

    "Jamaican party with Rasta vibes" names the accent as a property of the
    PLACE and the people in it, not of the person holding the camera. Without
    this the POV brief assigned the accent to the viewer alone, told the model
    an unbent line means the accent has vanished, and the only way to satisfy
    both was to give the viewer every line and the woman on screen none.
    """
    it = intent or ""
    if not it.strip():
        return False
    for w in accent_words(accent):
        if _re.search(r"(?<![A-Za-z])" + _re.escape(w) + r"(?![A-Za-z])", it, _re.I):
            return True
    return False


def _name_in_intent(name: str, intent: str) -> bool:
    """True if name appears as its own token in the intent (case-insensitive)."""
    import re
    n = (name or "").strip()
    if not n or not (intent or "").strip():
        return False
    # Word-ish boundary so "Ann" does not match "Hannah"
    pat = r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![A-Za-z0-9])"
    return re.search(pat, intent, flags=re.I) is not None


def _filter_lexicon_text(lx: str, intent: str) -> str:
    """Keep only Name = desc lines whose name is present in the intent."""
    if not (lx or "").strip():
        return ""
    out_sections = []
    current_header = None
    current_lines = []

    def flush():
        nonlocal current_header, current_lines
        if current_header is not None and current_lines:
            out_sections.append(current_header)
            out_sections.extend(current_lines)
        current_header = None
        current_lines = []

    for raw in (lx or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if low in ("triggers:", "characters:") or stripped.endswith(":") and "=" not in stripped:
            flush()
            current_header = stripped if stripped.endswith(":") else stripped + ":"
            current_lines = []
            continue
        # "Name = description"
        if "=" in stripped:
            name = stripped.split("=", 1)[0].strip()
            if _name_in_intent(name, intent):
                if current_header is None:
                    current_header = "Entries:"
                current_lines.append(stripped)
            continue
        # free-form leftover line — drop by default to avoid leakage
    flush()
    return "\n".join(out_sections).strip()


def _lexicon_block(lexicon: str, intent: str = "") -> str:
    """Active lexicon only: entries whose names appear in the director's request."""
    lx = (lexicon or "").strip()
    if not lx:
        return ""
    if (intent or "").strip():
        lx = _filter_lexicon_text(lx, intent)
    if not lx:
        return ""
    return (
        "\nLEXICON (ACTIVE only — named in the director's request)\n"
        "Only entries below are in play; render each exactly as described. "
        "Unlisted names do not exist — never cast or invent other lexicon "
        "entries.\n"
        + lx
    )


def build_system(*, mode: str, pov: str, accent: str, fps: int, seconds: float,
                 dialogue: str = "some", wardrobe: str = "off",
                 undress: bool = False, seed: int = 0, intent: str = "",
                 camera: str = "off", transition: str = "off",
                 music: str = "off", music_bg: bool = False,
                 lexicon: str = "", fmt: str = "flowing",
                 style_hint: str = "", style: str = "",
                 has_image: bool = True,
                 accent_strength: str = "natural") -> str:
    """Assemble the brief. Panel controls apply in BOTH i2v and t2v.
    Only opening (and vision attach) differ by mode. Lexicon applies only
    when names are present in the intent.
    style_hint: optional medium from the still ('cartoon') to harden I2V anchor.
    has_image: i2v only — False when the still is not on the wire (no mmproj,
    vision unsupported, or the vision-fail retry). Must be computed from the
    same value the caller uses to attach the image, or the system prompt and the
    user turn will disagree about whether there is a still."""
    # Both formats budget on full wall seconds. See write_seconds().
    wsec = write_seconds(seconds)
    pct = talk_pct(dialogue)
    lo, hi = word_budget(wsec, dialogue)
    nb_lo, nb_hi = beat_budget(wsec)
    parts = [_CORE,
             _fmt_contract(fmt, nb_lo=nb_lo, nb_hi=nb_hi, lo=lo, hi=hi,
                           seconds=seconds),
             _SHOT.format(sec=wsec)]
    _prio = _priority_block(
        pov=pov, is_i2v=(mode or "i2v").lower() == "i2v", intent=intent,
        undress=undress, dialogue_pct=pct, accent=accent, transition=transition)
    if _prio:
        parts.insert(2, _prio)
    # Gated: only spend these words when the shot can actually break on them.
    # Who is on screen, decided ONCE. wardrobe already read the intent for this
    # ("the man" -> him) but identity never did and hardcoded "she", so a shot
    # about a man got a male wardrobe AND an invented Jamaican woman centre-frame
    # from CAST LOOK. Both seeds now answer to the same detector.
    _subject = infer_who(intent)
    _has_other = intent_names_other_person(intent)
    _explicit = has_anatomy(intent, undress)
    if _explicit:
        parts.append(_SHOT_ANATOMY)
    if has_orientation(intent) or (pov or "off").lower() != "off":
        parts.append(_SHOT_ORIENT)
    parts.append(_LIGHT)

    is_i2v = (mode or "i2v").lower() == "i2v"
    if is_i2v:
        parts.append(_I2V_OPEN if has_image else _I2V_OPEN_BLIND)
    else:
        parts.append(_T2V_OPEN)

    # VISUAL STYLE sits directly under the opener: it is the top-level law and
    # has to be declared in the first words of the script, so it must be read
    # before any of the request-dependent laws below.
    # I2V is excluded — the attached still IS the medium and outranks any
    # dropdown, exactly like wardrobe (the still is ground truth).
    if not is_i2v:
        sl = style_law(style)
        if sl:
            parts.append(sl)
            # FIRST PERSON reserves the exact opening tokens; the style takes
            # position two rather than fighting for position one.
            reserved = ('"POV shot, first-person camera"'
                        if (pov or "off").lower() in ("male", "female") else "")
            parts.append(style_opener(style, after=reserved))

    # I2V: still owns face/body/style. Accent may still seed VIEWER hands in POV.
    # Third-person I2V must not inject "An Australian woman…" over a cartoon still.
    p_now = (pov or "off").lower()
    if is_i2v:
        parts.append(_I2V_LOOK)
        sh = (style_hint or "").strip().lower()
        if sh in ("cartoon", "anime", "comic", "illustration", "2d"):
            parts.append(_I2V_CARTOON)
        if p_now in ("male", "female"):
            idb = identity_block(accent, seed, intent, pov, explicit=_explicit,
                                 subject=_subject)
            if idb:
                parts.append(idb)
    else:
        idb = identity_block(accent, seed, intent, pov, explicit=_explicit,
                                 subject=_subject)
        if idb:
            parts.append(idb)

    lxb = _lexicon_block(lexicon, intent=intent)
    if lxb:
        parts.append(lxb)

    cam = camera_law(camera)
    if cam:
        parts.append("\n" + cam)
    trans = transition_law(transition)
    if trans:
        parts.append("\n" + trans)

    # Always append: off is explicit NO-MUSIC (models invent radios otherwise).
    parts.append(music_block(music, music_bg, accent=accent))

    if has_motion(intent):
        mv = _MOVEMENT
        if (music or "off").strip().lower() != "off":
            mv += _MOVEMENT_BEAT
        parts.append(mv)

    if has_mirror(intent):
        parts.append(_MIRROR)

    # I2V: still is the outfit — no wardrobe seed. Undress only removes still garments.
    if not is_i2v:
        ward_who = wardrobe if wardrobe in ("her", "him", "auto", "off") else "auto"
        uw = underwear_block(ward_who if ward_who in ("her", "him") else "auto",
                             seed, intent)
        if uw:
            parts.append(uw)
        else:
            wb = wardrobe_block(ward_who, seed, intent)
            if wb:
                parts.append(wb)
    if undress:
        parts.append(UNDRESS)
        if is_i2v:
            parts.append(
                "\nUNDRESS + STILL\n"
                "Only remove garments already visible on the still. "
                "Do not introduce a new outfit just to take it off."
            )

    silent = pct == 0
    performance = pct >= 70

    p = (pov or "off").lower()
    if p in ("male", "female"):
        vg = "man" if p == "male" else "woman"
        parts.append((_POV_I2V if is_i2v else _POV_T2V).format(vg=vg))
        parts.append(hand_law(p))
        if not silent:
            parts.append(_POV_VOICE if _has_other else _POV_VOICE_SOLO)

    if silent:
        parts.append(_SILENT)
    else:
        parts.append(_SPEECH)
        if performance:
            parts.append(_PERFORMANCE + performance_note(music, accent))
        else:
            _n = dialogue_lines(wsec, pct, beats=nb_lo)
            # Nobody to talk to. The old count gave a lone driver five lines of
            # monologue; a third of that, floor of one.
            if p in ("male", "female") and not _has_other:
                _n = max(1, _n // 3)
            parts.append(_SPEECH_COUNT.format(n=_n))
        note = accent_note(accent)
        if not note:
            parts.append(_VOICE_PLAIN)
        if note:
            lab = accent_label(accent)
            kind = accent_kind(accent)
            shared = accent_in_intent(accent, intent)
            # POV: accent is the viewer's voice; third-person: the on-screen lead.
            if p in ("male", "female") and shared:
                # The intent named this accent as part of the scene, so the
                # person on screen has it too — and she is the one with a mouth
                # in frame, so she leads.
                first_tag = f"she says, in a {lab} accent, …"
                who = (
                    "The request puts this accent in the scene itself, so BOTH "
                    "voices carry it: the viewer AND the person on screen. She "
                    "speaks it first and most — she is the mouth in frame."
                )
            elif p in ("male", "female"):
                first_tag = f'the viewer says, in a {lab} accent, …'
                who = (
                    "This accent is the VIEWER's first. It is also the voice of "
                    "anyone sharing the viewer's place — people in one room, one "
                    "vehicle, one crew normally sound alike, so an on-screen "
                    "speaker uses it too unless the request marks them as coming "
                    "from somewhere else. Where someone genuinely does not share "
                    "it, their plain-voiced line is CORRECT and not a failure "
                    "of this law. The number of accented lines is not a target: "
                    "never move a line to the viewer to raise it."
                )
            else:
                first_tag = f"she says, in a {lab} accent, …"
                who = (
                    "This accent is the on-screen speaker's (usually she)."
                )
            parts.append(
                "\nVOICE\n"
                f"{note} "
                f"{who} "
                "The accent lives in the SOUND of the words, not in a label "
                "attached to them, and never tapers to neutral.\n"
                # The note above lists this accent's ACTUAL shifts. Pointing
                # rule 1 back at it is the whole New York fix: without the
                # anchor the model reached for a generic dialect spelling that
                # belongs to no accent in particular.
                "1) SPELL THE SOUND. " + density_rule(accent_strength)
                + ", using the shifts NAMED IN THE NOTE ABOVE and no others — a "
                "general dialect spelling that could belong to any accent is "
                "not this accent. An accented line with nothing bent reads as "
                "plain English and the accent vanishes; a word this accent says "
                "like plain English stays plain; hyphens added to look accented "
                "read as a speech impediment. Every respelling stays a word a "
                "person could read aloud — spelling mangled past recognition "
                "renders as text burned across the picture instead of sound.\n"
                "2) VARY THE TAG. Name the accent once, in that speaker's "
                f"first line ({first_tag}); after that a short bracket ({lab} "
                "+ a fresh delivery word) or delivery alone. Never the same "
                "tag twice, and never a sentence after the quote explaining how "
                "it was said — the delivery lives in the bracket and the "
                "emphasis lives in CAPS inside the line.\n"
                + (_VOICE_REGISTER if kind == "variety" else _VOICE_ONE_LANE)
                + strength_law(accent_strength)
            )
            if intent_supplies_line(intent):
                parts.append(_GIVEN_LINE)

    f = int(fps or 24)
    if f >= 40:
        parts.append(_FPS_HIGH)
    elif f <= 16:
        parts.append(_FPS_LOW)

    return "\n".join(parts).strip()


def build_user(*, intent: str, mode: str, has_image: bool,
               style_hint: str = "", style: str = "") -> str:
    it = (intent or "").strip()
    tail = (
        "\nWrite ONLY the finished shot text now. No plan, no bullets of rules, "
        "no self-notes, no 'Correction:'."
    )
    if (mode or "").lower() == "i2v":
        head = ("The attached image is frame one." if has_image
                else "Frame one is a provided still (not attached — infer only from the request).")
        medium = ""
        sh = (style_hint or "").strip().lower()
        if sh in ("cartoon", "anime", "comic", "illustration", "2d"):
            medium = (
                "\nMEDIUM CHECK (locked): the still is cartoon/illustration — "
                "sentence one MUST say so (e.g. \"2D cartoon of…\"). Keep every "
                "beat drawn/cel — never photoreal."
            )
        elif has_image:
            medium = (
                "\nMEDIUM CHECK: LOOK at the attached image before writing. "
                "If it is cartoon, comic, anime, superhero illustration, or "
                "other non-photo art, sentence one MUST name that medium and "
                "the whole shot stays in it — do not write a live-action woman "
                "over a drawn character."
            )
        body = it or "Bring the still to life with motion true to what it shows."
        return f"{head}{medium}\nDirector's request: {body}{tail}"
    return (
        f"Director's request: "
        f"{it or 'An interesting cinematic shot. Your choice of subject and place.'}"
        f"{tail}"
    )


# ── negative prompt ───────────────────────────────────────────────────────────

# The vocabulary now lives in negative.py. These names are kept as aliases
# because older callers and tests import them from here.
_NEG_BASE = _neg.core()
_NEG_POV = _neg.pov()
_NEG_SILENT = _neg.silent()
_NEG_UNDRESS = _neg.undress()
_NEG_MOTION = _neg.motion()
_CUT_TRANSITIONS = _neg.CUT_TRANSITIONS

_MOVEMENT = """
MOVEMENT
Named joints in named directions — not mood. "She dances hard" is nothing; \
"shoulders roll, hips snap left then right, head whips" is a body. Each phase: \
plant feet, drop knees, spin through spine, hand trails. Big moves gather, \
explode, recover, and travel full range."""

# Only valid when a MUSIC block actually named a track. Appending it under the
# NO-MUSIC law told the model to sync to a beat the brief had just deleted.
_MOVEMENT_BEAT = " Move to the named music; land the accents on its beat."

# Substring cues fired on the wrong intents: "bracelet" and "brass" both
# contain "bra", "barefoot" contains "bare", "surface" contains "face", "drunk"
# contains "run", "strolls" contains "roll". A SFW intent was picking up the
# explicit ANATOMY law because of a piece of jewellery. These are the same cue
# lists with word boundaries, and with a suffix wildcard only where the stem is
# genuinely a stem (danc -> dancing, straddl -> straddling).
_MOTION_CUES = (r"danc\w*", r"spins?", r"spinning", r"twirl\w*", r"fight\w*",
                r"runs?", r"running", r"jumps?", r"jumping", r"flips?",
                r"flipping", r"shakes?", r"shaking", r"shimmy\w*",
                r"grinds?", r"grinding", r"thrust\w*", r"sways?", r"swaying",
                r"twerk\w*", r"struts?", r"strutting", r"sprint\w*",
                r"leaps?", r"leaping", r"kicks?", r"kicking", r"rolls?",
                r"rolling", r"bounces?", r"bouncing")

_MIRROR = """
MIRROR
Everything renders twice — real and reflected. Reflection must match real pose \
and position. First-person: the viewer stays invisible on the camera side. When \
the intent is the viewer's OWN act (I undress, I look at myself), the mirror is \
the one sanctioned way to show it — reflection may show the viewer's body \
(matching stated gender) while the camera side stays hands-only. Otherwise \
reflection shows subject and room only. Count hands across BOTH real view and \
reflection; a pair plus its reflection is the limit."""

_MIRROR_CUES = (r"mirrors?", r"mirrored", r"reflections?", r"reflect\w*",
                r"vanity")


# ANATOMY fires on bare bodies / explicit intent; ORIENTATION fires when a body
# turns or is placed relative to the lens. Both were unconditional inside THE
# SHOT, spending attention on every clothed, static shot that never needed them.
# Substring cues, matching the style of _MIRROR_CUES / _MOTION_CUES above.
_ANATOMY_CUES = (r"nude", r"nudity", r"naked", r"topless", r"bare", r"bares",
                 r"undress\w*", r"strips?", r"stripping", r"showers?",
                 r"showering", r"bath", r"bathing", r"lingerie", r"underwear",
                 r"panties", r"thongs?", r"bras?", r"nipples?", r"breasts?",
                 r"sex", r"sexual", r"fuck\w*", r"sucks?", r"sucking",
                 r"grinds?", r"grinding", r"straddl\w*", r"cum", r"cumming",
                 r"orgasms?", r"moans?", r"moaning")
# "skin" was in this list and is dropped: it appears in any number of clothed
# intents ("sun on her skin") and on its own says nothing about whether the
# shot needs plain anatomy words.

_ORIENT_CUES = (r"turns?", r"turning", r"turned", r"spins?", r"twists?",
                r"twisting", r"shoulders?", r"back to", r"backwards?",
                r"behind", r"faces?", r"facing", r"toward", r"towards",
                r"away", r"profile", r"glances?", r"glancing", r"grinds?",
                r"grinding", r"straddl\w*", r"bends?", r"bending", r"leans?",
                r"leaning", r"kneels?", r"kneeling", r"sits?", r"sitting",
                r"lies", r"lying", r"over her", r"over his")


def _priority_block(*, pov, is_i2v, intent, undress, dialogue_pct,
                    accent, transition) -> str:
    """Rank the laws that break THIS shot, in the order they break it.

    Every law below arrives at equal weight, so the model has no way to know
    that a POV leak ruins a render while a missing lens property only makes it
    flat. This names the load-bearing ones up front. It adds no new rule — it
    is a reading order for the rules already here.
    """
    top = []
    if (pov or "off").lower() != "off":
        top.append("FIRST PERSON — the viewer is never on screen.")
    if is_i2v:
        top.append("OPENING — the still is ground truth; animate, never redesign.")
    if has_mirror(intent):
        top.append("MIRROR — count hands across real view and reflection.")
    if undress:
        top.append("UNDRESS — physical stages; what comes off stays off.")
    if (accent or "off").lower() != "off":
        top.append("VOICE — audible in the spelling, tag never repeated.")
    if (transition or "off").lower() != "off":
        top.append("TRANSITION — its own beat, real screen time.")
    if int(dialogue_pct or 0) >= 70:
        top.append("SPEECH — the voice is the through-line.")
    if not top:
        return ""
    return ("\nIF ANYTHING BREAKS, IT BREAKS HERE — obey these first, in "
            "this order:\n  " + "\n  ".join(top[:4]))


def _cue_re(cues):
    """One compiled alternation per cue list, anchored on word boundaries."""
    return _re.compile(r"(?<![A-Za-z])(?:" + "|".join(cues) + r")(?![A-Za-z])",
                       _re.I)


_ANATOMY_RE = _cue_re(_ANATOMY_CUES)
_ORIENT_RE = _cue_re(_ORIENT_CUES)
_MIRROR_RE = _cue_re(_MIRROR_CUES)
_MOTION_RE = _cue_re(_MOTION_CUES)


def has_anatomy(intent: str, undress: bool = False) -> bool:
    """True when the shot will show skin the model must name plainly."""
    if undress:
        return True
    return bool(_ANATOMY_RE.search(intent or ""))


def has_orientation(intent: str) -> bool:
    return bool(_ORIENT_RE.search(intent or ""))


def has_mirror(intent: str) -> bool:
    return bool(_MIRROR_RE.search(intent or ""))


def has_motion(intent: str) -> bool:
    return bool(_MOTION_RE.search(intent or ""))


def build_negative(*, pov: str, dialogue: str = "some", undress: bool = False,
                   fmt: str = "flowing",
                   transition: str = "off", intent: str = "", extra: str = "",
                   camera: str = "off", style: str = "", mode: str = "",
                   auto: str = "") -> str:
    """Static banks + the optional auto pass, deduped.

    auto: output of negative.clean_auto() over the finished script. It goes in
    ahead of the user's `extra` but after the gated banks, and dedupe() drops
    anything it repeats — a term stated twice in a negative weights that concept
    twice, the same way it does in a positive.
    """
    parts = [_NEG_BASE]
    # Push the sampler the same way the brief pushes the writer. T2V only —
    # in I2V the still owns the medium, so a style negative could fight it.
    if (mode or "").lower() != "i2v":
        sneg = style_negative(style)
        if sneg:
            parts.append(sneg)
    if (pov or "off").lower() in ("male", "female"):
        parts.append(_NEG_POV)
        parts.append(hand_negative(pov))
    if talk_pct(dialogue) == 0:
        parts.append(_NEG_SILENT)
    if undress:
        parts.append(_NEG_UNDRESS)
    if has_motion(intent):
        parts.append(_NEG_MOTION)
    tn = _neg.transition(transition)
    if tn:
        parts.append(tn)
    cn = camera_negative(camera)
    if cn:
        parts.append(cn)
    a = (auto or "").strip()
    if a:
        parts.append(a)
    x = (extra or "").strip()
    if x:
        parts.append(x)
    fn = _neg.format_text(fmt)
    if fn:
        parts.append(fn)
    return _neg.dedupe(parts)
