"""
selftest.py — Claude Prompt LD
Interaction tests: for each config, assert the right laws are PRESENT and the
wrong ones are ABSENT. Catches law collisions like face-detail leaking into a
POV shot. Run: python3 selftest.py  (no ComfyUI/torch needed — brain only).
"""
import importlib.util, sys, types

def _load(n):
    s = importlib.util.spec_from_file_location(n, n + ".py")
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m

for n in ["accents", "wardrobe", "cinematics", "music", "identity", "shotscript", "styles", "negative", "hands"]:
    _load(n)
src = open("brain.py").read()
for a, b in [("from .accents", "from accents"), ("from .cinematics", "from cinematics"),
             ("from .music", "from music"), ("from .identity", "from identity"), ("from .shotscript", "from shotscript"), ("from .styles", "from styles"), ("from .wardrobe", "from wardrobe"),
             ("from . import negative as _neg", "import negative as _neg"),
             ("from .hands import", "from hands import")]:
    src = src.replace(a, b)
brain = types.ModuleType("brain"); exec(src, brain.__dict__)
BS = brain.build_system
NEG = brain.build_negative

PASS, FAIL = 0, 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1
    else:
        FAIL += 1; print(f"  ✗ {name}")

def has(s, *subs): return all(x in s for x in subs)
def none(s, *subs): return not any(x in s for x in subs)

# ── POV shot: viewer must never get a face/body ─────────────────────────────
s = BS(mode="i2v", pov="female", accent="off", fps=24, seconds=12, dialogue="some",
       intent="she leans close and looks at me")
check("POV: viewer-invisible rule present",
      "no face, torso" in s or "no pronouns, no body" in s or "never a face" in s.lower())
check("POV: living-detail scoped to on-screen only", "never invent a face, eyes, or expression for a first-person viewer" in s)
check("POV: first-person opener present", "FIRST PERSON" in s)

# ── POV + accent: viewer hands + viewer voice (not "she is Jamaican") ───────
s = BS(mode="t2v", pov="male", accent="jamaican", fps=24, seconds=12, dialogue=40,
       seed=7, intent="i reach for the door and say hello")
check("POV+jamaican: VIEWER LOOK not CAST LOOK she",
      "VIEWER LOOK" in s and "she is Jamaican" not in s)
check("POV+jamaican: hands get regional skin",
      "hands" in s.lower() and ("deep brown" in s or "dark brown" in s or "rich brown" in s
                                or "brown hands" in s))
check("POV+jamaican: viewer is Jamaican", "viewer is a Jamaican" in s)
check("POV+jamaican: accent is viewer's", "VIEWER's" in s or "viewer's accent" in s.lower()
      or "the VIEWER's" in s)
check("POV+jamaican: first tag uses the viewer", "the viewer says, in a Jamaican accent" in s
      or "viewer says, in a Jamaican" in s)
check("POV+jamaican: does not force subject ethnicity",
      "Do not paint an on-screen subject" in s or "not from the accent dial" in s)
# third-person jamaican still seeds cast
s = BS(mode="t2v", pov="off", accent="jamaican", fps=24, seconds=12, dialogue=20,
       seed=7, intent="a woman talks")
check("non-POV jamaican: CAST LOOK she is Jamaican", "she is Jamaican" in s)
check("non-POV jamaican: she says tag", "she says, in a Jamaican accent" in s)

# ── non-POV: living detail should NOT carry the viewer caveat noise... ───────
# (it's fine if present; it's harmless. But POV rules must be ABSENT.)
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="some",
       intent="a woman smiles")
check("non-POV: no FIRST PERSON block", "FIRST PERSON" not in s)
check("non-POV: living detail present", "The living detail is never blank" in s)

# ── silent (dial=0): no speech laws, but living detail + anatomy stay ───────
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=0,
       intent="a woman walks")
check("silent: NO SPEECH present", "NO SPEECH" in s)
check("silent: SPEECH block absent", "Quoted lines are the only lines that lip-sync" not in s)
# legacy strings still map
s2 = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="silent", intent="a woman walks")
check("legacy 'silent' maps to 0", "NO SPEECH" in s2)
s2 = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="some", intent="a woman talks")
check("legacy 'some' gets a line count", "spoken lines across the shot" in s2)
s2 = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=100, intent="she raps to the beat")
check("dial 100 = performance mode", "PERFORMED VOCAL" in s2)
check("performance relaxes stop-beat", "stop-beat rule is relaxed" in s2)
check("performance most beats voice", "Most beats must carry quoted voice" in s2)
check("silent: no VOICE/accent block", "VOICE" not in s)
check("silent: living detail still present", "The living detail is never blank" in s)

# ── accent requires dialogue: silent+accent should NOT emit a VOICE block ────
s = BS(mode="t2v", pov="off", accent="polish", fps=24, seconds=12, dialogue=0,
       intent="a woman walks")
check("silent+accent: accent suppressed (no VOICE)", "Polish-accented" not in s)

# ── some+accent: every-quote accent recipe ─────────────────────────────────
s = BS(mode="t2v", pov="off", accent="polish", fps=24, seconds=12, dialogue="some",
       intent="a woman talks")
check("some+accent: VOICE present", "VOICE" in s and "Polish-accented" in s)
check("some+accent: name-once tag rule",
      "Name the accent once" in s and "Never the same tag twice" in s)
check("some+accent: phonetic spelling", "SPELL THE SOUND" in s and "espell" in s)
check("some+accent: slip strict cap",
      "ONE short native slip" in s and "zero is fine" in s)

# ── i2v: no wardrobe seed — clothes only from the still ────────────────────
s = BS(mode="i2v", pov="off", accent="off", fps=24, seconds=12, dialogue="some",
       wardrobe="her", intent="she dances")
check("i2v: no wardrobe seed invent", "WARDROBE (seed bias" not in s)
check("i2v: clothing from still law", "swap a garment" in s and "Do not inventory the outfit" in s
      or "only what the still shows" in s.lower())
check("i2v: medium-first law", "MEDIUM FIRST" in s or "medium" in s.lower())
check("i2v: no ethnicity cast seed", "she is Australian" not in s
      and "she is Indian" not in s)
check("i2v: still-only look law", "LOOK (image-to-video)" in s
      or "Appearance is the still only" in s)
s = BS(mode="i2v", pov="off", accent="australian", fps=24, seconds=12, dialogue=30,
       intent="she talks", style_hint="cartoon")
check("i2v+cartoon: cartoon medium block",
      "cartoon" in s.lower() and (
          "MEDIUM (detector" in s or "2D cartoon" in s or "cel" in s.lower()))
check("i2v+cartoon: still look not nationality cast", "she is Australian" not in s)
check("i2v+cartoon: accent still for voice", "VOICE" in s and "Australian" in s)
check("i2v+cartoon: no full CAST LOOK block",
      "CAST LOOK (intent always wins" not in s)

# ── t2v+wardrobe her: wardrobe present + physics law ────────────────────────
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="some",
       wardrobe="her", intent="a woman dances")
check("t2v+wardrobe: WARDROBE present", "WARDROBE" in s)
check("wardrobe: clothes-are-objects law", "CLOTHES ARE OBJECTS" in s)
check("wardrobe: fabric travels with body", "Clothes travel with the body" in s
      or "Clothes travel with the body" in s.lower()
      or "travel with the body" in s.lower())
check("wardrobe: re-assert fit later beats", "re-assert fit" in s.lower())
u = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20,
       undress=True, intent="she undresses")
check("undress: four-part physics chain", "four named" in u or "grip" in u.lower())
check("undress: what comes off stays off", "What comes off stays off" in u)

# ── intent names clothing: wardrobe suppressed even in t2v ──────────────────
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="some",
       wardrobe="her", intent="a woman in a red dress dances")
check("named-clothes: wardrobe suppressed", "WARDROBE" not in s)

# ── motion intent: MOVEMENT present; still intent: absent ───────────────────
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="silent",
       intent="a woman dances hard")
check("motion intent: MOVEMENT present", "MOVEMENT" in s)
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="silent",
       intent="a woman sits quietly")
check("still intent: MOVEMENT absent", "MOVEMENT" not in s)

# ── camera/transition/music off: their blocks absent ────────────────────────
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="silent",
       camera="off", transition="off", music="off", intent="a woman stands")
check("camera off: no CAMERA block", "CAMERA —" not in s)
check("transition off: no TRANSITION block", "TRANSITION —" not in s)
check("music off: no-music law present", "no music" in s.lower() or "There is no music" in s)
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="silent",
       camera="slow_push", intent="a woman stands")
check("camera on: enforcement", "MANDATORY: this camera behavior is the law" in s)
n = NEG(pov="off", dialogue="some", camera="locked_off", intent="x")
check("neg locked_off: bans handheld", "handheld shake" in n)
wsec = brain.write_seconds(16)
nb_lo, nb_hi = brain.beat_budget(wsec)
lo100, _ = brain.word_budget(wsec, 100)
check("16s: beat floor is 5 (full duration, no pad)", nb_lo == 5)
check("Talk100 word lo denser", lo100 >= 300)
check("CORE middle beat range", "middle of the beat range" in
      BS(mode="t2v", pov="off", accent="off", fps=24, seconds=16, dialogue="silent", intent="a woman stands"))

# ── negatives: POV arms 3rd-person-body; silent arms speech-neg ─────────────
n = NEG(pov="female", dialogue="some", intent="she dances")
check("neg POV: apparatus banned", "camera in frame" in n and "second pair of hands" in n)
n = NEG(pov="off", dialogue="silent", intent="she walks")
check("neg silent: speech banned", "talking" in n)

# ── transition negatives don't fight cut transitions ────────────────────────
n = NEG(pov="off", dialogue="some", transition="hard_cut", intent="x then y")
check("neg hard_cut: doesn't ban cuts", "hard cut" not in n)
n = NEG(pov="off", dialogue="some", transition="morph", intent="x then y")
check("neg morph: bans cuts", "hard cut" in n)

# ── cast fidelity: solo intent must not invent people ───────────────────────
s = BS(mode="i2v", pov="male", accent="off", fps=24, seconds=12, dialogue="some",
       intent="i'm crawling in the desert looking for water")
check("solo POV: cast-fidelity clause present", "Invent no companion" in s)
check("solo POV: environment-is-subject", "the environment is the subject" in s)
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="silent",
       intent="a man walks alone through an empty field")
check("solo t2v: cast-fidelity present", "Cast only who the intent gives you" in s)
# world-pop and cast-fidelity coexist without contradiction for a venue
s = BS(mode="t2v", pov="off", accent="off", fps=30, seconds=15, dialogue="silent",
       intent="a woman dances at a club")
check("club: world-pop + cast-fidelity both present", "a club has other dancers" in s and "Cast only who the intent gives you" in s)

# ── mirror shot: mirror law fires + hand-count-across-reflection ─────────────
s = BS(mode="i2v", pov="male", accent="off", fps=24, seconds=12, dialogue="some",
       intent="undressing in front of a tall mirror in my bedroom")
check("mirror intent: MIRROR law present", "MIRROR" in s)
check("mirror: counts hands across reflection", "Count hands across BOTH" in s)
check("mirror POV: viewer invisible camera-side", "the viewer stays invisible on the camera side" in s)
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="silent",
       intent="a woman walks in a field")
check("no-mirror intent: MIRROR absent", "MIRROR" not in s)
# ── idle viewer hands rule present in POV ───────────────────────────────────
s = BS(mode="i2v", pov="male", accent="off", fps=24, seconds=12, dialogue="some",
       intent="she undresses in front of me")
check("POV: idle-hands rule present (now in hands.py)",
      "hands resting or parked count the same as reaching" in s)

# ── senses: lens+mic only, visible proxies ─────────────────────────────────
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="silent",
       intent="a woman stands in a room")
check("senses: scent/taste/thought banned", "scent, taste, or" in s.lower()
      or "scent, taste" in s.lower())
check("senses: proxies given", "shimmer" in s.lower() and "sweat" in s.lower()
      and ("breath-fog" in s.lower() or "gooseflesh" in s.lower()))
# ── mirror self-reflection exception ────────────────────────────────────────
s = BS(mode="i2v", pov="male", accent="off", fps=24, seconds=12, dialogue="some",
       intent="im getting undressed in front of a tall mirror")
check("mirror self-act: exception present", "the mirror is the one sanctioned way" in s)
check("mirror self-act: camera side stays hands-only", "the camera side stays hands-only" in s)
check("POV: first-person intent belongs to viewer", "belongs to the VIEWER" in s)

# ── mirror false-positives: wine glass / glasses must NOT fire mirror law ───
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="some",
       intent="she sips a glass of wine on the balcony")
check("wine glass: MIRROR absent", "MIRROR" not in s)
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue="some",
       intent="a man in glasses reads a book")
check("eyeglasses: MIRROR absent", "MIRROR" not in s)

# ── him occasion wardrobe fires; dress-party doesn't suppress ───────────────
import wardrobe as W
check("dress party biases (no false suppress)", W.wants_bias("a guy at a dress party"))
check("red dress still suppresses", not W.wants_bias("a woman in a red dress"))
b = W.wardrobe_block("him", 3, "a guy at a dress party dancing")
check("him club occasion fires", any(p in b for p in W._HIS_OCCASIONS["club"]["normal"]) or "He wears" in b)
b = W.wardrobe_block("him", 2, "he arrives at the gala")
check("him formal occasion fires", "Occasion lean (formal)" in b
      or any(p in b for p in W._HIS_OCCASIONS["formal"]["normal"]))
check("cosplay occasion fires", "Occasion lean (cosplay)" in
      W.wardrobe_block("her", 5, "she cosplays at a convention"))

# ── underwear suite ─────────────────────────────────────────────────────────
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20,
       wardrobe="auto", intent="she dances in her panties")
check("panties intent: UNDERWEAR block fires", "UNDERWEAR" in s)
check("underwear: named garment is law", "named garment is law" in s)
check("underwear replaces wardrobe (no double)", "WARDROBE" not in s)
s = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20,
       wardrobe="auto", intent="a woman in a red dress dances")
check("red dress: still plain suppress, no underwear", "UNDERWEAR" not in s and "WARDROBE" not in s)
# seed variety
import wardrobe as W2
picks = set()
for sd in range(12):
    b = W2.underwear_block("her", sd, "she dances in her panties")
    picks.add(b.split("toward ")[1].split(" in ")[0])
check("underwear seed variety (>=6 styles /12 seeds)", len(picks) >= 6)

check("orientation law present", "Facing is a fact" in BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20, intent="she grinds on his lap"))

check("adlib rule when speaking", "short involuntary vocal" in BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=30,intent="she talks"))
check("adlib absent when silent", "short involuntary vocal" not in BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=0,intent="she walks"))

lex="Triggers:\nthe drop = freeze then slam\n\nCharacters:\nNova = silver bob\nHannah = brown hair"
check("lexicon fires when name in intent", "LEXICON" in BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=20,intent="Nova dances",lexicon=lex))
check("lexicon includes only matched name", "Nova = silver bob" in BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=20,intent="Nova dances",lexicon=lex))
check("lexicon drops unmatched character", "Hannah" not in BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=20,intent="Nova dances",lexicon=lex))
check("lexicon silent when name not in intent", "LEXICON" not in BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=20,intent="she dances on the tram",lexicon=lex))
check("lexicon active-only framing", "ACTIVE only" in BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=20,intent="Nova dances",lexicon=lex))
check("no lexicon = no block", "LEXICON" not in BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=20,intent="she dances",lexicon=""))
check("Ann does not match Hannah", "Hannah" not in BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=20,intent="Ann waves",lexicon=lex))

# ── identity seeding ────────────────────────────────────────────────────────
s = BS(mode="t2v",pov="off",accent="indian",fps=24,seconds=12,dialogue=20,seed=7,intent="a woman with a turtle")
check("identity block fires", "CAST LOOK" in s or "IDENTITY" in s)
check("ethnicity anchored in first sentence", "she is Indian" in s and "FIRST sentence" in s)
s = BS(mode="t2v",pov="off",accent="indian",fps=24,seconds=12,dialogue=20,seed=7,intent="a tall curvy indian woman with long black hair")
check("stated look not overwritten", "intent already describes this person" in s)
s = BS(mode="t2v",pov="off",accent="off",fps=24,seconds=12,dialogue=20,seed=3,intent="a woman dances")
check("no accent = global pool identity", ("CAST LOOK" in s or "IDENTITY" in s) and ("Name who is on screen" in s or "on screen" in s))
import identity as _ID
reds = sum(1 for sd in range(60) if "red hair" in _ID.identity_block("scottish", sd, "a woman"))
check("scottish redhead is rare (<25%)", reds < 15)
blk = sum(1 for sd in range(60) if "black" in _ID.identity_block("indian", sd, "a woman"))
check("indian hair mostly black/dark", blk > 25)
# accent attribution: name-once not label-every-quote
s = BS(mode="t2v",pov="off",accent="indian",fps=24,seconds=12,dialogue=50,intent="she talks")
check("accent names once then varies",
      "Name the accent once" in s and "Never the same tag twice" in s)
check("no label-every-quote rule", "LABEL EVERY QUOTE" not in s)
check("respell only real shifts",
      "stays plain" in s and "SPELL THE SOUND" in s)

import identity as _I2
sa = [_I2.identity_block("indian", sd, "a woman") for sd in range(40)]
check("south asian never coiled/curly hair", not any(("coiled" in b or "curly" in b) for b in sa))
check("south asian feature hint present", "South Asian features" in sa[0])
af = _I2.identity_block("nigerian", 5, "a woman")
check("african region resolves (not global)", "West African features" in af)
check("african hair not fair/red", "fair skin" not in af and "red hair" not in af)

# ── output formats ──────────────────────────────────────────────────────────
fl = BS(mode="t2v",pov="off",accent="off",fps=30,seconds=16,dialogue=40,intent="she talks",fmt="flowing")
br = BS(mode="t2v",pov="off",accent="off",fps=30,seconds=16,dialogue=40,intent="she talks",fmt="bracket")
ss = BS(mode="t2v",pov="off",accent="off",fps=30,seconds=16,dialogue=40,intent="she talks",fmt="shotscript")
check("flowing: prose contract", "flowing present-tense prose" in fl)
check("flowing: no timestamps", "timestamp" in fl.lower())
check("bracket: own-line dialogue rule", "OWN line" in br)
check("bracket: delivery-only bracket", "never spoken" in br and "never action" in br)
check("shotscript: timed windows", "[0-" in ss and ("span all" in ss or "contiguous" in ss
                                                     or "cover every second" in ss.lower()
                                                     or "TIMED BEATS" in ss))
check("shotscript: footprint setup", "PART 1 — FOOTPRINT" in ss or "FOOTPRINT" in ss)
check("shotscript: timed beats part", "PART 2 — TIMED BEATS" in ss or "TIMED BEATS" in ss)
check("shotscript: footprint before windows", ss.find("FOOTPRINT") < ss.find("[0-") or ss.find("PART 1") < ss.find("PART 2"))
check("shotscript: final window named", "[13-16]" in ss)
check("shotscript: every window listed", "in order:" in ss and "[0-" in ss)
_w = lambda t: int(__import__("re").search(r"(\d+)-\d+ words", t).group(1))
# Both formats budget on full wall seconds now. Equality is the point:
# a format A/B is only a comparison if both sides get the same budget.
check("formats budget identically (fair A/B)", _w(ss) == _w(fl))
check("shotscript: only what changes in beats", "only what changes" in ss)
check("shotscript: covers whole duration", "16 seconds" in ss)
check("formats are distinct", fl != br and br != ss)
check("laws survive all formats", all("LIVING DETAIL" in x or "living detail" in x for x in (fl,br,ss)))
import shotscript as _SS
w = _SS._windows(16, 5)
check("windows contiguous+cover", w[0][0] == 0 and w[-1][1] == 16 and all(w[i][1]==w[i+1][0] for i in range(len(w)-1)))
check("action windows split evenly", abs((w[0][1]-w[0][0]) - (w[1][1]-w[1][0])) <= 1)
check("no sub-1s window at any duration", all(min(b-a for a,b in _SS._windows(sec,nb)) >= 1 for sec in (6,8,12,16,24,30) for nb in (2,3,4,5,6)))
n_ss = NEG(pov="off", dialogue=40, fmt="shotscript")
check("shotscript negatives guard overlay", "timecode overlay" in n_ss)
check("flowing negatives unchanged", "timecode overlay" not in NEG(pov="off", dialogue=40, fmt="flowing"))

check("shotscript: guide order camera-first", "Camera / frame" in ss and "Movement" in ss)
check("shotscript: reaction then sound then dialogue", "Reaction" in ss and "Sound" in ss and "Dialogue" in ss)
check("shotscript: order is guide not form", "guide for the prose, not a form" in ss)
check("shotscript: no stage labels in output", "Never print stage labels" in ss)
check("shotscript: no ACTION: recipe", "ACTION:" not in ss and "2. ACTION" not in ss)
check("shotscript: never pad", "never pad" in ss)
check("flowing still untouched", "FOOTPRINT" not in fl and "guide for the prose" not in fl)
check("music off is quiet", "There is no music" in
      BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=0,
         music="off", intent="a woman sits"))
check("core sound does not invite music source", "named music source if one exists" not in
      BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=0, music="off", intent="x"))

# ── VISUAL STYLE ────────────────────────────────────────────────────────────
# A style is a top-level law that claims "everything in frame", so it is the
# single most collision-prone thing in the brief. These pin the four rules that
# matter: it fires in t2v, it is silent when off, the still outranks it in i2v,
# and it ABSORBS rather than replaces the other laws (accent/wardrobe/POV).
_sty = dict(pov="off", accent="off", fps=24, seconds=12, dialogue=20,
            intent="a woman walks through a kitchen")
s_off  = BS(mode="t2v", style="off", **_sty)
s_on   = BS(mode="t2v", style="spongebob", **_sty)
s_i2v  = BS(mode="i2v", style="spongebob", **_sty)

check("style off injects nothing", "VISUAL STYLE" not in s_off)
check("style on injects law", "VISUAL STYLE" in s_on)
check("style declared in first words", "STYLE OPENER" in s_on or "first words" in s_on
      or "sentence one" in s_on.lower())
check("style enforced per beat (anti-drift)", "every beat" in s_on or "drift" in s_on.lower()
      or "slides toward ordinary footage" in s_on)
check("i2v still outranks the dropdown", "VISUAL STYLE" not in s_i2v)

# The absorb clause is what stops a style eating the rest of the request.
check("style absorbs, not replaces", "still happens" in s_on or "still applies" in s_on
      or "rendered in this style" in s_on)

# Collision guard: turning a style on must not delete any law that was there
# without it. This is the check that would have caught a style clause that
# silently overwrote POV/anatomy/living-detail.
for law in ("LIVING DETAIL", "THE SHOT"):
    check(f"style keeps {law}", (law in s_off) <= (law in s_on) or law in s_on)

# POV is the law styles are most likely to fight (both claim the whole frame).
p_off = BS(mode="t2v", style="off", pov="female", accent="off", fps=24,
           seconds=12, dialogue=20, intent="she reaches for me")
p_on  = BS(mode="t2v", style="spongebob", pov="female", accent="off", fps=24,
           seconds=12, dialogue=20, intent="she reaches for me")
check("style does not evict POV law", "POV" in p_on and "viewer" in p_on.lower())
check("POV survives style intact", ("never on screen" in p_off) == ("never on screen" in p_on))

# Negatives: t2v gets the style negation, i2v must not (it would fight the still).
check("style negative fires t2v", NEG(pov="off", dialogue=20, style="spongebob",
                                     mode="t2v") != NEG(pov="off", dialogue=20,
                                                        style="off", mode="t2v"))
check("style negative silent in i2v", NEG(pov="off", dialogue=20, style="spongebob",
                                          mode="i2v") == NEG(pov="off", dialogue=20,
                                                             style="off", mode="i2v"))
check("style off adds no negative", NEG(pov="off", dialogue=20, style="off", mode="t2v")
      == NEG(pov="off", dialogue=20, mode="t2v"))

# Every dropdown key must actually build a law — a key in the UI with no entry
# in styles.py would silently render photoreal.
import styles as _ST
_missing = [k for k in _ST.STYLE_KEYS if k != "off" and not _ST.style_law(k)]
check("every STYLE_KEY builds a law", not _missing)
_noneg = [k for k in _ST.STYLE_KEYS if k != "off" and not _ST.style_negative(k)]
check("every STYLE_KEY builds a negative", not _noneg)

# STYLE FIRST vs identity: both want an early sentence. They coexist only
# because identity is scoped to "the first sentence that INTRODUCES her", not
# the first sentence of the script. If either clause is ever reworded to claim
# the script's opening outright, this fails and the render loses one of them.
_si = BS(mode="t2v", pov="off", accent="indian", fps=24, seconds=12,
         dialogue=20, style="spongebob", intent="a woman dances in a kitchen")
check("style opener claims the script's first words", "opens the script" in _si)
check("identity scoped to her introducing sentence",
      "FIRST sentence that introduces her" in _si)
check("identity does not also claim the script opening",
      "Indian" in _si and _si.count("opens the script") == 1)

# ═══ AUDIT FIXES (fresh session) ═══════════════════════════════════════════

# ── 1. the script covers the whole clip; no pad-down ───────────────────────
check("write_seconds: no pad at 12s", brain.write_seconds(12) == 12.0)
check("write_seconds: no pad at 16s", brain.write_seconds(16) == 16.0)
check("write_seconds: floor at 2s", brain.write_seconds(0) >= 2.0)
_s16 = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=16, dialogue=20,
          intent="a woman stands")
check("THE SHOT states the real duration", "THE SHOT — 16 seconds" in _s16)
_ss16 = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=16, dialogue=20,
           fmt="shotscript", intent="a woman stands")
check("shotscript windows reach the clip end", "[13-16]" in _ss16 or "[12-16]" in _ss16)

# ── 2. spoken lines capped by speech time, then by beat count ──────────────
check("12s @20%: lines fit the clock", brain.dialogue_lines(12, 20) <= 4)
check("12s @50%: nine lines refused", brain.dialogue_lines(12, 50) <= 4)
check("12s @100%: still bounded", brain.dialogue_lines(12, 100) <= 6)
check("30s @50%: scales up", brain.dialogue_lines(30, 50) >= 8)
check("silent: zero lines", brain.dialogue_lines(12, 0) == 0)
check("never more lines than beats", all(
    brain.dialogue_lines(d, pct, beats=brain.beat_budget(d)[0])
    <= brain.beat_budget(d)[0]
    for d in (4, 8, 12, 16, 20, 30, 60) for pct in (10, 20, 45, 60)))
check("line target never exceeds one per 3s (non-performance)", all(
    brain.dialogue_lines(d, pct) <= max(1, int(d / 3))
    for d in (4, 8, 12, 16, 20, 30) for pct in (10, 20, 45, 60)))

# ── 3. cue gates fire on the intent, not on a substring ───────────────────
for _bad, _fn, _label in [
    ("a woman wearing a bracelet smiles", brain.has_anatomy, "bracelet !ANATOMY"),
    ("she picks up a brass lamp", brain.has_anatomy, "brass !ANATOMY"),
    ("a barefoot girl on grass", brain.has_anatomy, "barefoot !ANATOMY"),
    ("sun on her skin", brain.has_anatomy, "skin alone !ANATOMY"),
    ("light on a wet surface", brain.has_orientation, "surface !ORIENT"),
    ("a drunk man at a bar", brain.has_motion, "drunk !MOVEMENT"),
    ("she strolls down the lane", brain.has_motion, "strolls !MOVEMENT"),
]:
    check(_label, not _fn(_bad))
for _good, _fn, _label in [
    ("she undresses slowly", brain.has_anatomy, "undresses ANATOMY"),
    ("topless in the shower", brain.has_anatomy, "topless ANATOMY"),
    ("she takes off her bra", brain.has_anatomy, "bra ANATOMY"),
    ("she turns to face him", brain.has_orientation, "turns ORIENT"),
    ("leaning against the wall", brain.has_orientation, "leaning ORIENT"),
    ("she dances hard", brain.has_motion, "dances MOVEMENT"),
    ("running down the street", brain.has_motion, "running MOVEMENT"),
    ("she looks in the mirror", brain.has_mirror, "mirror MIRROR"),
    ("her reflection wavers", brain.has_mirror, "reflection MIRROR"),
]:
    check(_label, _fn(_good))

# ── 4. position zero is claimed once ──────────────────────────────────────
_ps = BS(mode="t2v", pov="female", accent="off", fps=24, seconds=12, dialogue=20,
         style="anime_cel", intent="she leans toward me")
check("POV+style: POV keeps the first words", 'First words exactly' in _ps)
check("POV+style: style yields to it", "stays first" in _ps
      and "IMMEDIATELY after" in _ps)
check("POV+style: style no longer claims position zero",
      "opens the script — the very first words" not in _ps)
_sonly = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20,
            style="anime_cel", intent="a woman dances")
check("style alone: still owns position zero",
      "opens the script — the very first words" in _sonly)
check("t2v opener defers to reserved first words", "position one" in _sonly)
check("non-POV: no FIRST PERSON block (opener reworded)",
      "FIRST PERSON" not in _sonly)

# ── 5. seed banks stop arguing with the laws ──────────────────────────────
_ex = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20,
         seed=21, undress=True, intent="she undresses slowly")
check("explicit: ANATOMY law present", "Bare breasts are" in _ex)
# The word survives in two legitimate places — the ANATOMY law naming it as
# banned, and wardrobe's "bare chest" on a man, which is the correct word there.
# What must never come back is the SEED handing "chest" to sentence one.
_castline = [l for l in _ex.splitlines() if "lean toward" in l]
check("explicit: seed says breasts, never chest",
      bool(_castline) and "chest" not in _castline[0] and "breasts" in _castline[0])
_cl = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20,
         seed=21, intent="a woman plays guitar")
check("clothed: no bust trait at all",
      "breasts" not in _cl and "chest" not in _cl)
check("clothed: no ANATOMY law", "Bare breasts are" not in _cl)
_svp = BS(mode="t2v", pov="female", accent="off", fps=24, seconds=12, dialogue=20,
          style="pixar_3d", intent="she reaches for me")
check("style law scoped to on-screen", "everything actually in frame" in _svp)
check("style law spares the viewer",
      "never overrides a first-person viewer's invisibility" in _svp)

# ── 6. i2v: motion-first, and honest about a missing still ────────────────
_iv = BS(mode="i2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20,
         intent="she turns to the window", has_image=True)
check("i2v: identify minimally then move", "IDENTIFY MINIMALLY, THEN MOVE" in _iv)
check("i2v: medium still named in sentence one", "MEDIUM" in _iv)
check("i2v: per-beat anchor present", "ANCHOR — every beat" in _iv)
check("i2v: anchor excludes the viewer", "not one of these people" in _iv)
check("i2v: no bullet example list", "  • 2D cartoon" not in _iv)
check("i2v: stops asking for an outfit inventory",
      "Do not inventory the outfit" in _iv)
_ib = BS(mode="i2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20,
         intent="she turns to the window", has_image=False)
check("i2v blind: admits it cannot see", "still you cannot see" in _ib)
check("i2v blind: never claims the image is attached",
      "Frame one is the attached image" not in _ib)
check("i2v blind: demands motion only", "Write MOTION ONLY" in _ib)
check("i2v sighted: does claim the attachment",
      "Frame one is the attached image" in _iv)

# ── 8. negatives are visual terms, deduped, and script-guarded ────────────
import negative as NG
_n = NEG(pov="female", dialogue=0, undress=True, intent="she dances",
         transition="morph", camera="hunting", style="anime_cel",
         fmt="shotscript", mode="t2v")
check("neg: abstractions gone", not any(x in _n for x in (
    "slideshow", "teleporting between poses", "third-person view of the viewer",
    "unmotivated jump cuts", "garment popping back on")))
check("neg: artifact names present", "duplicated limbs" in _n and "frozen frame" in _n)
check("neg: no duplicated terms",
      len([t.strip() for t in _n.split(",")]) == len(set(t.strip() for t in _n.split(","))))
_nauto = NEG(pov="off", dialogue=20, intent="x", auto="daylight, dry matte skin")
check("neg: auto terms merged", "daylight" in _nauto and "dry matte skin" in _nauto)
check("neg: auto terms deduped against banks",
      NEG(pov="off", dialogue=20, intent="x", auto="blurry, daylight").count("blurry") == 1)
# the auto pass must never negate what the shot committed to
_script = "A cel-animated woman dances in a dim club, handheld camera swaying."
check("auto pass: drops terms the script wants",
      "handheld camera" not in NG.clean_auto("daylight, handheld camera, dry skin", _script))
check("auto pass: keeps genuine opposites",
      "daylight" in NG.clean_auto("daylight, handheld camera", _script))
check("auto pass: strips sentences",
      NG.clean_auto("The negative prompt should avoid the following things entirely", "") == "")
check("auto pass: strips leading no/avoid",
      NG.clean_auto("no daylight, avoid film grain", "") == "daylight, film grain")
check("auto pass: caps the list", len(NG.clean_auto(
      ", ".join(f"term{i}" for i in range(40)), "").split(",")) <= 14)
check("auto pass: empty in, empty out", NG.clean_auto("", "") == "")
check("auto pass: survives junk", isinstance(NG.clean_auto("...,,,   ,", ""), str))
check("auto runner: no script, no call",
      NG.run_auto("", lambda *a, **k: ["x"]) == "")
check("auto runner: backend failure is not fatal",
      NG.run_auto("a shot", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))) == "")
check("auto runner: happy path parses",
      NG.run_auto("a shot at night", lambda *a, **k: ["daylight, film grain"])
      == "daylight, film grain")

# ── integrity sweep across the changed surfaces ───────────────────────────
_bad = []
for _mode in ("t2v", "i2v"):
    for _pov in ("off", "female"):
        for _sty in ("off", "anime_cel"):
            for _fmt in ("flowing", "shotscript"):
                for _d in (0, 20, 100):
                    for _sec in (4, 12, 16, 30):
                        _b = BS(mode=_mode, pov=_pov, accent="french", fps=24,
                                seconds=_sec, dialogue=_d, style=_sty, fmt=_fmt,
                                seed=5, wardrobe="auto", camera="hunting",
                                intent="she dances close to me in a club",
                                has_image=(_mode == "i2v"))
                        # no double claim on position zero
                        if _pov != "off" and _sty != "off" and _mode == "t2v":
                            if "opens the script — the very first words" in _b:
                                _bad.append(("pos0", _mode, _pov, _sty, _fmt, _d, _sec))
                        # silent never coexists with a line target
                        if _d == 0 and "short spoken lines across the shot" in _b:
                            _bad.append(("silent", _mode, _pov, _sty, _fmt, _d, _sec))
                        # the seed never hands "chest" to the opening sentence
                        if any("lean toward" in l and "chest" in l
                               for l in _b.splitlines()):
                            _bad.append(("chest", _mode, _pov, _sty, _fmt, _d, _sec))
check(f"sweep: 288 configs clean ({len(_bad)} problems)", not _bad)
if _bad:
    for row in _bad[:6]:
        print("     ", row)


# ═══ ACCENT KIND + WHO SPEAKS ══════════════════════════════════════════════
import accents as AC

# ── the two kinds are classified, and "off" is neither ────────────────────
check("accent kind: jamaican is a variety", AC.accent_kind("jamaican") == "variety")
check("accent kind: french is l2", AC.accent_kind("french") == "l2")
check("accent kind: off is neither", AC.accent_kind("off") == "")
check("accent kind: every key classified",
      all(AC.accent_kind(k) in ("variety", "l2") for k in AC.ACCENT_KEYS if k != "off"))
check("accent kind: english varieties grouped", all(
      AC.accent_kind(k) == "variety" for k in
      ("jamaican", "trinidadian", "nigerian", "ghanaian", "south_african",
       "indian", "filipino", "rp_british", "cockney", "scottish", "irish",
       "scouse", "geordie", "northern_english", "welsh", "australian",
       "new_zealand", "southern_us")))
check("accent kind: second-language grouped", all(
      AC.accent_kind(k) == "l2" for k in
      ("french", "german", "russian", "mandarin", "arabic", "swahili",
       "japanese", "italian", "polish", "greek", "hebrew")))
check("accent words: every accent detectable",
      all(AC.accent_words(k) for k in AC.ACCENT_KEYS if k != "off"))
check("accent words: no short junk tokens",
      all(len(w) > 3 for k in AC.ACCENT_KEYS if k != "off" for w in AC.accent_words(k)))
check("accent words: 'new' cannot false-match", "new" not in AC.accent_words("new_zealand"))

# ── rule 3 branches on kind ───────────────────────────────────────────────
_jv = BS(mode="t2v", pov="off", accent="jamaican", fps=24, seconds=12,
         dialogue=35, intent="two people talk on a porch")
_fr = BS(mode="t2v", pov="off", accent="french", fps=24, seconds=12,
         dialogue=35, intent="two people talk on a porch")
check("variety gets REGISTER", "3) REGISTER" in _jv and "3) ONE LANE" not in _jv)
check("l2 keeps ONE LANE", "3) ONE LANE" in _fr and "3) REGISTER" not in _fr)
check("REGISTER licenses vocabulary and grammar",
      "own words, idioms and sentence shapes ARE the voice" in _jv)
check("REGISTER prefers real words over mangled ones",
      "spelled the way that variety normally spells it" in _jv)
check("REGISTER still forbids another language",
      "never a whole line in a different language" in _jv)
check("REGISTER still forbids cross-accent borrowing",
      "nothing borrowed from another accent" in _jv)
check("ONE LANE cap untouched for l2", "At most ONE short native slip" in _fr)

# ── Han's principle: permission, never a word list ────────────────────────
_LEX = ("irie", "bredren", "wagwan", "yaad", "pickney", "rasta", "yuh", "nuh",
        "bruv", "innit", "yaar", "gwan", "mon chéri", "vant")
_leaks = []
for _acc in ("jamaican", "indian", "cockney", "scottish", "nigerian",
             "australian", "southern_us", "french", "german"):
    for _pv in ("off", "male"):
        _b = BS(mode="t2v", pov=_pv, accent=_acc, fps=24, seconds=12,
                dialogue=35, seed=1, intent="two people talk on a porch")
        for _w in _LEX:
            if __import__("re").search(r"(?<![A-Za-z])" + __import__("re").escape(_w)
                                       + r"(?![A-Za-z])", _b, __import__("re").I):
                _leaks.append((_acc, _pv, _w))
check(f"zero dialect example words in any brief ({len(_leaks)} leaks)", not _leaks)

# ── the render bug: POV viewer spoke every line ───────────────────────────
# Cause was a syllogism: rule 1 demanded EVERY quoted line be bent, the
# who-clause said only the viewer is bent, so every line became the viewer's
# and the woman on screen got none.
_pv = BS(mode="t2v", pov="male", accent="jamaican", fps=24, seconds=14,
         dialogue=35, seed=3, intent="she leans close to me on a porch")
check("rule 1 scoped to accented speakers",
      "BY A SPEAKER WHO HAS THIS ACCENT" in _pv)
check("a plain line from someone else is explicitly fine",
      "plain-voiced line is CORRECT" in _pv)
check("rule 1 no longer demands every quoted line",
      "Respell a few words in every quoted line" not in _pv)
check("POV: moving lines to the viewer is banned",
      "never move a line to the viewer" in _pv)
check("POV: lip-sync reasoning present", "only a mouth on screen can lip-sync" in _pv)
# ── a viewer alone must never get a companion invented for them ────────────
check("no person named -> solo", not brain.intent_names_other_person(
      "pov in a f1 car going through new york really fast"))
check("pronoun names one", brain.intent_names_other_person("she leans toward me"))
check("noun names one", brain.intent_names_other_person("pov at a bar, the bartender pours"))
check("crowd names one", brain.intent_names_other_person("pov driving, the crowd waves"))
check("no false match on partial words",
      not brain.intent_names_other_person("a shed in the manger of history"))
_solo = BS(mode="t2v", pov="male", accent="new_york", fps=24, seconds=15,
           dialogue=40, accent_strength="thick",
           intent="pov in a f1 car going really fast")
check("solo: viewer is the only voice", "the viewer is the ONLY voice" in _solo)
check("solo: inventing a speaker is banned",
      "no passenger, no companion, no bystander" in _solo)
check("solo: named as the worse failure",
      "worse failure than a quiet shot" in _solo)
check("solo: no ratio to game", "There \nis no ratio to meet here" in _solo
      or "There is no ratio to meet here" in " ".join(_solo.split()))
check("solo: line count cut", " 1 short spoken lines" in _solo
      or "about 1 short spoken" in _solo)
check("solo: silence allowed", "let beats carry \nno voice" in _solo
      or "let beats carry no voice" in " ".join(_solo.split()))
check("solo: ratio law absent",
      "at least two from a mouth in frame" not in _solo)
_comp = BS(mode="t2v", pov="male", accent="new_york", fps=24, seconds=15,
           dialogue=40, intent="she leans close to me in the passenger seat")
check("companion: ratio law present",
      "at least two from a mouth in frame" in _comp)
check("companion: solo law absent", "the ONLY voice" not in _comp)
check("companion: full line count kept",
      " 1 short spoken lines" not in _comp)
check("ratio no longer overrides every law",
      "not negotiable against any other law" not in _comp)
check("third person untouched by either",
      "WHO SPEAKS" not in BS(mode="t2v", pov="off", accent="off", fps=24,
                             seconds=15, dialogue=40, intent="a car drives"))

check("POV: accent shared with the same place",
      "anyone sharing the viewer's place" in _pv)
check("POV: outsiders still speak plain", "their plain-voiced line is CORRECT" in _pv)
check("POV: accented-line count is not a target", "not a target" in _pv)
check("POV: the no-trade rule is stated once",
      _pv.count("No accent, style or music") == 1)
check("POV: ratio outranks accent/style/music only",
      "No accent, style or music \nrule is a reason" in _pv
      or "No accent, style or music rule is a reason" in " ".join(_pv.split()))
check("POV: first line belongs on screen",
      "FIRST line of the" in _pv and "never to the viewer" in _pv)
check("POV: viewer lines are outnumbered",
      "at least two from a mouth in frame" in _pv)
check("POV: subject carries the dialogue", "carries most of the dialogue" in _pv)

# ── the intent naming the accent hands it to the on-screen speaker ────────
check("intent naming the accent is detected",
      brain.accent_in_intent("jamaican", "Jamaican party with Rasta vibes"))
check("intent not naming it is not detected",
      not brain.accent_in_intent("jamaican", "she leans close to me on a porch"))
check("detection is word-boundary safe",
      not brain.accent_in_intent("new_zealand", "she wears a new dress"))
check("detection is case-insensitive",
      brain.accent_in_intent("scottish", "a SCOTTISH pub at closing time"))
_sh = BS(mode="t2v", pov="male", accent="jamaican", fps=24, seconds=14,
         dialogue=35, seed=3, intent="Jamaican party, braided hair, a woman dances")
check("shared: both voices carry it", "BOTH voices carry it" in _sh)
check("shared: she leads", "She speaks it first and most" in _sh)
check("shared: first tag is hers", "she says, in a Jamaican accent" in _sh)
check("unshared: first tag is the viewer's",
      "the viewer says, in a Jamaican accent" in _pv)
check("shared clause absent when intent is silent on it",
      "BOTH voices carry it" not in _pv)
check("third person unaffected by the POV branches",
      "WHO SPEAKS" not in _jv and "BOTH voices carry it" not in _jv)

# ── silent still wins over all of it ──────────────────────────────────────
_si2 = BS(mode="t2v", pov="male", accent="jamaican", fps=24, seconds=12,
          dialogue=0, intent="Jamaican party, a woman dances")
check("silent: no VOICE law at all", "3) REGISTER" not in _si2 and "SPELL THE SOUND" not in _si2)
check("silent: no who-speaks law", "WHO SPEAKS" not in _si2)

# ── sweep every accent x pov x shared/unshared ────────────────────────────
_bad2 = []
for _acc in AC.ACCENT_KEYS:
    if _acc == "off":
        continue
    _kind = AC.accent_kind(_acc)
    for _pv2 in ("off", "male", "female"):
        for _named in (False, True):
            _it = ("a party in the street, two people talk"
                   if not _named else
                   f"a {AC.accent_words(_acc)[0]} party, two people talk")
            _b2 = BS(mode="t2v", pov=_pv2, accent=_acc, fps=24, seconds=12,
                     dialogue=35, seed=2, intent=_it)
            want = "3) REGISTER" if _kind == "variety" else "3) ONE LANE"
            dont = "3) ONE LANE" if _kind == "variety" else "3) REGISTER"
            if want not in _b2 or dont in _b2:
                _bad2.append(("rule3", _acc, _pv2, _named))
            # exactly one first-tag example, never both
            if _b2.count("says, in a") != 1:
                _bad2.append(("tag", _acc, _pv2, _named))
            # POV must always carry the who-speaks law
            if _pv2 != "off" and "WHO SPEAKS" not in _b2:
                _bad2.append(("whospeaks", _acc, _pv2, _named))
check(f"sweep: 38 accents x 3 pov x 2 intents ({len(_bad2)} problems)", not _bad2)
if _bad2:
    for row in _bad2[:6]:
        print("     ", row)


# ═══ POV HANDS + lens terminology ══════════════════════════════════════════
_hp = BS(mode="t2v", pov="male", accent="jamaican", fps=24, seconds=14,
         dialogue=35, seed=3, intent="Jamaican party, a woman dances")
_hn = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=14, dialogue=35,
         intent="a woman dances")
check("POV HANDS present in POV", "POV HANDS" in _hp)
check("POV HANDS absent otherwise", "POV HANDS" not in _hn)
check("hands: count is stated and held",
      "COUNT." in _hp and "one hand or both" in _hp
      and "hold that number until they leave" in _hp)
check("hands: one pair at a time survived the move", "Only ONE pair is ever up" in _hp)
check("hands: a hand must have a job", "a hand with no job" in _hp)
check("hands: grazing the air named as the failure", "grazing the air" in _hp)
check("hands: back of hand faces the view", "BACK of the hand faces the" in _hp)
check("hands: splayed-palm pose banned",
      "splayed fingers held up is the pose" in " ".join(_hp.split()))
check("hands: contact hides fingers", "Contact also hides fingers" in _hp)
check("hands: exit is written", "Hands leave by an edge" in _hp)
check("hands carry the viewer's gender", "They are a man's." in _hp)
# the anchor: early, but never at position zero
check("anchor folded into POV HANDS", "WHOSE HANDS" in _hp
      and "VIEWER ANCHOR" not in _hp)
check("anchor refuses position zero",
      "nor in the \nrequired opening words" in _hp
      or "nor in the required opening words" in " ".join(_hp.split()))
check("anchor explains why", "reads as a body in shot" in _hp)
check("anchor binds sex to the hands",
      'a possessive on the hands' in _hp and '"a man\'s hands"' in _hp)
check("anchor demands it early", "inside the opening two sentences" in _hp)
check("anchor: female variant", '"a woman\'s hands"' in
      BS(mode="t2v", pov="female", accent="off", fps=24, seconds=14,
         dialogue=35, intent="she grips the rail"))
check("anchor: consequence still stated",
      "read as nobody's" in _hp and "no a man" not in _hp)
check("anchor: no template left", "{noun}" not in _hp and "{vg}" not in _hp)
check("hands law absent without POV",
      "WHOSE HANDS" not in BS(mode="t2v", pov="off", accent="off", fps=24,
                              seconds=14, dialogue=35, intent="she waves"))
check("proven opening tokens untouched",
      'First words exactly: "POV shot, first-person camera"' in _hp)
check("female viewer gets her own hands",
      "They are a woman's." in BS(mode="t2v", pov="female", accent="off",
                                  fps=24, seconds=14, dialogue=35,
                                  intent="she grips the rail"))
check("hands: gender is physical, not a caption",
      "forearm hair" in _hp and "nail length" in _hp
      and "size against the frame" in _hp)
check("hands: skin tone alone is called out",
      "read as nobody's" in _hp)
check("hands: no unsubstituted template left", "{vg}" not in _hp)
check("who speaks: ratio not vibes", "at least two from a mouth in frame" in _hp)
check("who speaks: first line is hers",
      "FIRST line of the" in _hp and "never to the viewer" in _hp)
check("hands: no duplicate clause left in POV rules",
      _hp.count("Only ONE pair is ever up") == 1)
check("hands negative fires in POV",
      "palm facing camera" in NEG(pov="male", dialogue=20, intent="x")
      and "floating hands" in NEG(pov="male", dialogue=20, intent="x"))
check("hands negative silent otherwise",
      "palm facing camera" not in NEG(pov="off", dialogue=20, intent="x"))
# lens: optics only, never the viewpoint
_orient = [l for l in _hp.splitlines() if "which way torso and head point" in l]
check("ORIENTATION points at the view, not the lens",
      bool(_orient) and "toward the lens" not in _orient[0])
check("lens survives for optics", "Name one optical property" in _hp)
check("the ban still names the object", "never a lens, camera, glass, or screen" in _hp)
check("no 'lens + mic'", "lens + mic" not in _hp)
check("light section renamed", "LIGHT & OPTICS" in _hp and "LIGHT & LENS" not in _hp)
_lens_lines = [l for l in _hp.splitlines() if "lens" in l]
check(f"lens appears only in the ban ({len(_lens_lines)} lines)", len(_lens_lines) <= 2)


# ═══ ctx clamp estimate + opera ════════════════════════════════════════════
import importlib.util as _u
_rs = open("routes.py").read()
_ns = {}
_a = _rs.index("def _prompt_estimate"); _b = _rs.index("def _build_messages")
exec(compile(_rs[_a:_b], "est", "exec"), _ns)
_est = _ns["_prompt_estimate"]
_txt = "x" * 12000
_img = "data:image/jpeg;base64," + ("A" * 120000)
_plain = [{"role": "system", "content": _txt}, {"role": "user", "content": "hello"}]
_vision = [{"role": "system", "content": _txt},
           {"role": "user", "content": [{"type": "text", "text": "hello"},
                                        {"type": "image_url", "image_url": {"url": _img}}]}]
check("estimate: text only, ~4 chars/token", 2800 < _est(_plain) < 3200)
check("estimate: base64 image is not counted as text", _est(_vision) < 4500)
check("estimate: image still costs a flat allowance", _est(_vision) > _est(_plain))
check("estimate: a 32k ctx leaves real room for i2v",
      32000 - _est(_vision) - 256 > 20000)
check("estimate: an 8k ctx still fits a loaded brief",
      8192 - _est(_vision) - 256 > 3000)
check("estimate: empty is zero", _est([]) == 0)
check("estimate: survives junk", isinstance(_est([{"role": "user"}]), int))
import music as MU
check("opera exists", "opera" in MU.MUSIC_KEYS)
check("opera is described by sound, not by name",
      "opera" not in MU.MUSIC["opera"].lower())
_op = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=20,
         music="opera", intent="a woman sings on a stage")
check("opera reaches the brief", "single trained voice" in _op)
check("opera harmonics exists", "opera_harmonics" in MU.MUSIC_KEYS)
check("aria: mouth can hold the note",
      "jaw opens and stays open" in MU.music_block("opera"))
check("harmonics: mouth must not chase the runs",
      "do NOT hold on a mouth trying" in MU.music_block("opera_harmonics"))
check("harmonics: names the artifact it prevents",
      "chattering to match them is the artifact" in MU.music_block("opera_harmonics"))
check("harmonics: still uses the mouth on held notes",
      "Save the mouth for the long held notes" in MU.music_block("opera_harmonics"))
check("vocal note only on vocal tracks",
      "jaw opens" not in MU.music_block("techno")
      and "jaw opens" not in MU.music_block("club_house"))
check("background ducks the vocal note too",
      "jaw opens" not in MU.music_block("opera", background=True))
check("no music entry left as a bare stub",
      all(len(MU.MUSIC[k].split()) > 6 for k in MU.MUSIC_KEYS if k != "off"))
_oh = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=80,
         music="opera_harmonics", intent="a woman sings on a stage")
check("harmonics survives performance mode",
      "do NOT hold on a mouth" in _oh and "PERFORMED VOCAL" in _oh)
# ── the subject's gender comes from the intent, in BOTH seed blocks ─────────
def _cast(intent, **kw):
    b = BS(mode="t2v", pov="off", accent="jamaican", fps=24, seconds=16,
           dialogue=60, seed=1843697, wardrobe="auto", intent=intent, **kw)
    cast = [l for l in b.splitlines() if "Unless the intent says otherwise" in l]
    ward = [l for l in b.splitlines() if " wears something in the territory" in l]
    lean = [l for l in b.splitlines() if "lean toward" in l]
    return b, (cast[0] if cast else ""), (ward[0] if ward else ""), (lean[0] if lean else "")

_b, _c, _w, _l = _cast("the man says he doesn't care, then sings with his group")
check("male intent: CAST LOOK says he", " he is Jamaican" in _c)
check("male intent: introduces him, not her", 'introduces him' in _c)
check("male intent: 'a Jamaican man'", '"a Jamaican man' in _c)
check("male intent: wardrobe agrees", _w.startswith("He wears"))
check("male intent: no female build term",
      not any(x in _l for x in ("hourglass", "curvy", "soft-figured", "petite")))
_b2, _c2, _w2, _l2 = _cast("a woman dances in a club")
check("female intent unchanged: she", " she is Jamaican" in _c2)
check("female intent: wardrobe agrees", _w2.startswith("She wears"))
_b3, _c3, _w3, _l3 = _cast("someone walks down a street")
check("no gender cue still defaults female", " she is Jamaican" in _c3
      and _w3.startswith("She wears"))
check("identity and wardrobe never disagree", all(
      ((" he is " in c) == w.startswith("He wears"))
      for c, w in [(_c, _w), (_c2, _w2), (_c3, _w3)]))
# bust never lands on a male subject, even on an explicit shot
_mb = [l for l in BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12,
                     dialogue=20, seed=3, wardrobe="auto", undress=True,
                     intent="the man undresses").splitlines()
       if "lean toward" in l][0]
check("male + undress: no bust trait", "breasts" not in _mb)
_fb = [l for l in BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12,
                     dialogue=20, seed=3, wardrobe="auto", undress=True,
                     intent="she undresses").splitlines()
       if "lean toward" in l][0]
check("female + undress: bust trait kept", "breasts" in _fb)
# lived-in detail must be wearable by the subject
import wardrobe as WD
_hits = []
for _sd in range(120):
    _wb = WD.wardrobe_block("him", _sd, "a man in a suit")
    if __import__("re").search(
            r"(?i)\b(bra|bralette|slit|stocking|skirt|dress|gown|neckline|she|her)\b",
            _wb.split("One lived-in detail like ", 1)[-1].split(".")[0]):
        _hits.append(_sd)
check(f"male wardrobe detail is never female-coded ({len(_hits)} hits)", not _hits)

# North American accents + labels
# ── accent strength: density, grammar, frequency, non-words — no examples ──
_na = BS(mode="t2v", pov="off", accent="jamaican", fps=24, seconds=14,
         dialogue=35, intent="two people talk", accent_strength="natural")
_st = BS(mode="t2v", pov="off", accent="jamaican", fps=24, seconds=14,
         dialogue=35, intent="two people talk", accent_strength="strong")
_th = BS(mode="t2v", pov="off", accent="jamaican", fps=24, seconds=14,
         dialogue=35, intent="two people talk", accent_strength="thick")
check("natural is the current behaviour", "Respell a few words" in _na)
check("natural adds no extra rule",
      "4)" not in _na and "Lay it on" not in _na)
check("strong raises the density", "Respell most of the words" in _st)
check("strong scales rule 3, adds no rule 4",
      "Lay it on:" in _st and "4)" not in _st)
check("strong: frequency not vocabulary", "belong in MOST lines" in _st)
check("strong: grammar follows the accent", "rather than standard word order" in _st)
check("thick rewrites the whole line", "Write the whole line the way" in _th)
check("thick scales rule 3, adds no rule 4",
      "Lay it on hard:" in _th and "4)" not in _th)
check("thick uses the non-words", "the noise made instead of a word" in _th)
check("thick: stress goes in the quote, not a sentence",
      "Stress goes INSIDE the quote" in _th and "in CAPS" in _th)
check("thick: no longer asks the model to describe delivery",
      "say where the stress lands" not in _th)
check("delivery prose banned in the tag rule",
      all("never a sentence after the quote explaining how" in x
          for x in (_na, _st, _th)))
check("supplied line uses CAPS for emphasis too",
      "CAPS inside the quote" in BS(mode="t2v", pov="off", accent="american",
                                    fps=24, seconds=12, dialogue=40,
                                    intent='he says "watch it" to her'))
check("readability limit applies at every strength",
      all("read aloud" in x for x in (_na, _st, _th)))
check("thick names the real failure", "burned across the picture" in _th)
check("strength does nothing without an accent",
      "4) LEAN IN" not in BS(mode="t2v", pov="off", accent="off", fps=24,
                             seconds=14, dialogue=35, intent="talk",
                             accent_strength="strong"))
check("strength does nothing when silent",
      "4) LAY IT ON" not in BS(mode="t2v", pov="off", accent="jamaican", fps=24,
                               seconds=14, dialogue=0, intent="talk",
                               accent_strength="thick"))
check("junk strength falls back to natural",
      AC.accent_strength("banana") == "natural"
      and "Respell a few words" in BS(mode="t2v", pov="off", accent="boston",
                                      fps=24, seconds=14, dialogue=35,
                                      intent="talk", accent_strength="banana"))
check("rule 1 anchors respelling to this accent's own note",
      all("shifts NAMED IN THE NOTE ABOVE" in x for x in (_na, _st, _th)))
check("generic dialect spelling ruled out",
      "could belong to any accent is not this accent" in " ".join(_th.split()))
check("rule 1 stays scoped at every strength",
      all("BY A SPEAKER WHO HAS THIS ACCENT" in x for x in (_na, _st, _th)))
_slex = ("irie", "bredren", "wagwan", "yaad", "yuh", "nuh", "gwan")
check("no example words leak in at any strength", not any(
      __import__("re").search(r"(?<![A-Za-z])" + w + r"(?![A-Za-z])", x,
                              __import__("re").I)
      for x in (_na, _st, _th) for w in _slex))
check("thick is not the default",
      "4) LAY IT ON" not in BS(mode="t2v", pov="off", accent="jamaican", fps=24,
                               seconds=14, dialogue=35, intent="talk"))

# ── a line supplied by the request still has to carry the accent ───────────
check("quoted line detected", brain.intent_supplies_line('she says "get out"'))
check("unquoted speech verb detected",
      brain.intent_supplies_line("a man in a car says that thing isnt working"))
check("bare speech verb is not a supplied line",
      not brain.intent_supplies_line("she speaks"))
check("plain action intent is not a supplied line",
      not brain.intent_supplies_line("a woman dances in a club"))
_gl = BS(mode="t2v", pov="off", accent="canadian", fps=24, seconds=7,
         dialogue=60, accent_strength="thick",
         intent="a man in a car says That lora daddy isnt using image to video either")
check("supplied-line law fires", "WORDS FROM THE REQUEST" in _gl)
check("supplied line keeps its meaning",
      "keep those words' meaning and order" in _gl)
check("accent still lands on given words", "the accent still lands ON them" in _gl)
check("nothing-to-bend has a fallback",
      "If the \ngiven words bend nothing, it shows in the attribution" in _gl
      or "If the given words bend nothing, it shows in the attribution"
      in " ".join(_gl.split()))
check("flat delivery named as the failure", "is a failed line, not a faithful one" in _gl)
check("law absent when the request supplies no words",
      "WORDS FROM THE REQUEST" not in BS(mode="t2v", pov="off", accent="canadian",
                                         fps=24, seconds=7, dialogue=60,
                                         intent="a man sits in a car"))
check("law absent when silent",
      "WORDS FROM THE REQUEST" not in BS(mode="t2v", pov="off", accent="canadian",
                                         fps=24, seconds=7, dialogue=0,
                                         intent='he says "hello there friend"'))
check("law absent with no accent",
      "WORDS FROM THE REQUEST" not in BS(mode="t2v", pov="off", accent="off",
                                         fps=24, seconds=7, dialogue=60,
                                         intent='he says "hello there friend"'))
check("appalachian exists", "appalachian" in AC.ACCENT_KEYS)
check("appalachian is a variety", AC.accent_kind("appalachian") == "variety")
check("appalachian label carries the searched-for word",
      AC.accent_label("appalachian") == "Appalachian (hillbilly)")
check("appalachian is not southern_us",
      AC.accent_note("appalachian") != AC.accent_note("southern_us"))
check("appalachian pairs to bluegrass, not country",
      MU.music_auto("appalachian") == "bluegrass")
check("bluegrass has no drum kit", "no drum kit" in MU.MUSIC["bluegrass"])
check("appalachian reaches the brief",
      "Appalachian English" in BS(mode="t2v", pov="off", accent="appalachian",
                                  fps=24, seconds=12, dialogue=30,
                                  intent="two men on a porch"))
check("american exists", "american" in AC.ACCENT_KEYS)
check("north american set present", all(
      k in AC.ACCENT_KEYS for k in
      ("american", "new_york", "californian", "midwest_us", "boston",
       "canadian", "aave")))
check("all new ones are varieties", all(
      AC.accent_kind(k) == "variety" for k in
      ("american", "new_york", "californian", "midwest_us", "boston",
       "canadian", "aave")))
check("AAVE label is not title-cased", AC.accent_label("aave") == "AAVE")
check("RP label fixed", AC.accent_label("rp_british") == "RP British")
# The note is now the ONLY authority rule 1 has, so guard it as data.
check("no note smuggles in an example word",
      not any('"' in AC.ACCENTS[k] or "'" in AC.ACCENTS[k]
              for k in AC.ACCENT_KEYS if k != "off"))
check("every note names the accent then its shifts",
      all(" — " in AC.ACCENTS[k] for k in AC.ACCENT_KEYS if k != "off"))
check("every note is substantial but still one line",
      all(6 < len(AC.ACCENTS[k].split()) < 40 and "\n" not in AC.ACCENTS[k]
          for k in AC.ACCENT_KEYS if k != "off"))
check("General American says there is little to respell",
      "little to respell" in AC.ACCENTS["american"])
# Two accents sharing their most spellable feature sound like each other, and
# the more famous one wins: "th hardened toward d" in the New York note produced
# Jamaican "ting". Each note must name what DISTINGUISHES it.
check("new york never flattens th at all",
      "th left \nexactly as it is, never flattened to d" in AC.ACCENTS["new_york"]
      or "th left exactly as it is, never flattened to d" in AC.ACCENTS["new_york"])
check("new york does not carry the patois th phrasing",
      "th toward d" not in AC.ACCENTS["new_york"])
check("patois keeps the unrestricted th", "th toward d" in AC.ACCENTS["jamaican"])
check("aave th stays distinct from both", "th toward d or f" in AC.ACCENTS["aave"])
check("r-dropping notes say how it is spelled",
      all("taking an h in spelling" in AC.ACCENTS[k]
          for k in ("new_york", "boston")))
check("new york keeps only shifts unique to it",
      "r dropped" in AC.ACCENTS["new_york"] and "aw tightened" in AC.ACCENTS["new_york"]
      and "flattened to d" in AC.ACCENTS["new_york"])
check("every accent has a non-empty note",
      all(AC.accent_note(k) for k in AC.ACCENT_KEYS if k != "off"))
check("american reaches the brief",
      "General American" in BS(mode="t2v", pov="off", accent="american", fps=24,
                               seconds=12, dialogue=30, intent="two people talk"))

# accent off must still describe a voice (LTX defaults to flat TTS otherwise)
_novoice = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12,
              dialogue=30, intent="two people talk")
check("accent off: voice still described", "VOICE (no accent chosen)" in _novoice)
check("accent off: register/grain/pace named",
      "register (low, high, mid)" in _novoice and "grain (smooth" in _novoice)
check("accent off: breath required", "a real voice takes air" in _novoice)
check("accent on: plain-voice law absent",
      "VOICE (no accent chosen)" not in BS(mode="t2v", pov="off",
                                           accent="french", fps=24, seconds=12,
                                           dialogue=30, intent="two people talk"))
check("silent: no voice law at all",
      "VOICE (no accent chosen)" not in BS(mode="t2v", pov="off", accent="off",
                                           fps=24, seconds=12, dialogue=0,
                                           intent="two people talk"))

# the request outranks the dropdown when it describes the music itself
_mp = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=12, dialogue=30,
         music="latin", intent="she sings electro-opera over a hard beat")
check("music: request outranks the dropdown", "THAT is the score" in _mp)
check("music: precedence stated in background mode too",
      "that description is the score" in BS(mode="t2v", pov="off", accent="off",
                                            fps=24, seconds=12, dialogue=30,
                                            music="latin", music_bg=True,
                                            intent="she sings electro-opera"))
check("performance: no real lyrics or titles",
      "never a real \nsong's title" in BS(mode="t2v", pov="off", accent="off",
                                          fps=24, seconds=20, dialogue=80,
                                          music="opera", intent="she sings")
      or "never a real song's title" in " ".join(
          BS(mode="t2v", pov="off", accent="off", fps=24, seconds=20,
             dialogue=80, music="opera", intent="she sings").split()))
check("MUSIC and MUSIC_LABELS keys match",
      set(MU.MUSIC) == set(MU.MUSIC_LABELS))
check("every genre has a real description, not a label",
      all(len(MU.MUSIC[k].split()) > 6 for k in MU.MUSIC if k != "off"))
check("every label is short enough to be a label",
      all(len(MU.MUSIC_LABELS[k].split()) < 7 for k in MU.MUSIC_LABELS))
check("auto resolves from the accent",
      MU.music_auto("jamaican") == "dancehall"
      and MU.music_auto("trinidadian") == "soca"
      and MU.music_auto("nigerian") == "afrobeats")
check("auto is off when the accent gives nothing",
      MU.music_auto("off") == "off" and MU.music_auto("") == "off")
check("every accent maps to a real genre",
      all(MU.music_auto(k) in MU.MUSIC_KEYS for k in AC.ACCENT_KEYS))
check("auto in the brief follows the accent",
      "dancehall riddim" in BS(mode="t2v", pov="off", accent="jamaican", fps=24,
                               seconds=12, dialogue=20, music="auto",
                               intent="a party"))
check("auto with no accent writes the NO-MUSIC law",
      "There is no music" in BS(mode="t2v", pov="off", accent="off", fps=24,
                                seconds=12, dialogue=20, music="auto",
                                intent="a party"))
check("resolve rejects junk", MU.resolve("not_a_genre") == "off")
check("resolve passes real keys through", MU.resolve("reggae") == "reggae")
check("region genres reach the brief",
      "one-drop reggae" in BS(mode="t2v", pov="off", accent="off", fps=24,
                              seconds=12, dialogue=20, music="reggae",
                              intent="a porch"))
# performance mode adapts to a vocal track instead of staying rap-shaped
_pa = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=20, dialogue=80,
         music="opera", intent="a woman sings on a stage")
_ph = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=20, dialogue=80,
         music="opera_harmonics", intent="a woman sings on a stage")
_pr = BS(mode="t2v", pov="off", accent="off", fps=24, seconds=20, dialogue=80,
         music="trap", intent="a rapper on a rooftop")
check("perf: aria is sung, not barred", "SUNG, not spoken in bars" in _pa)
check("perf: aria kills the ad-libs", "No ad-libs, no catchphrase" in _pa)
check("perf: aria shows breath", "chest and shoulders lift" in _pa)
check("perf: harmonics quotes sparingly", "quote sparingly" in _ph)
check("perf: harmonics leaves runs voice-only", "VOICE ONLY" in _ph)
check("perf: rap keeps the original shape",
      "ad-libs between bars" in _pr and "SUNG, not spoken" not in _pr)
check("perf note only at 70%+",
      "SUNG, not spoken in bars" not in BS(mode="t2v", pov="off", accent="off",
                                           fps=24, seconds=20, dialogue=40,
                                           music="opera", intent="she sings"))
check("perf note follows auto too",
      "quote sparingly" not in BS(mode="t2v", pov="off", accent="jamaican",
                                  fps=24, seconds=20, dialogue=80, music="auto",
                                  intent="a party"))
check("every music key still builds a law",
      all(MU.music_block(k) for k in MU.MUSIC_KEYS if k != "off"))


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
