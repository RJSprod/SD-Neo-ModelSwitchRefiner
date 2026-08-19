"""
styles.py — Prompt Master LD

Optional VISUAL STYLE law. Off by default (photoreal is LTX's default and the
node's proven behaviour). When a style is chosen it becomes the top-level law:
it owns every person, prop, surface and light in frame, and it is declared in
the FIRST words of the script.

Design notes (from Lightricks' own LTX-2.3 guidance):
  * A style must be NAMED EARLY — style vocabulary at the head of the prompt is
    what makes it hold; buried in the middle it gets treated as scene content.
  * Explicit NEGATION is required. "no gradients, no shadows" style clauses stop
    the model sliding subtle photoreal shading back in.
  * DRIFT is the real failure mode: the model starts stylised and creeps toward
    photorealism across a long clip. Their mitigation is short clips; we run
    12-16s, so every style carries per-beat reinforcement (_STYLE_ENFORCE),
    exactly the way cinematics._CAM_ENFORCE holds the camera mode.
  * Some styles imply camera and timing conventions (anime push-ins, stop-motion
    stepped movement). Those ride in the entry as a motion/camera bias.

Each entry is four things:
  1. declaration - the phrase that opens sentence one
  2. grammar     - how it renders (line, fill, shading, texture, palette)
  3. forbid      - the explicit negation that stops the drift
  4. motion      - optional camera/timing bias when the style implies one

Principle-only, in the accents.py tradition: no scene examples, no sample
sentences. The style says HOW things are made, never what happens.
"""

# Repeated as its own law so a 16s clip cannot drift back to photoreal.
_STYLE_ENFORCE = (
    " MANDATORY: this style is the law for the whole shot and every beat in it. "
    "It owns everything actually in frame — every person on screen, their face, "
    "body and garments, every prop, surface, and the light itself. Nothing in "
    "the shot is rendered any other way, and no element is photographic unless "
    "this style is itself photographic. This governs what is ON SCREEN and "
    "never overrides a first-person viewer's invisibility — do not draw a face "
    "or a body for a viewer who has none. Re-state the look in physical words "
    "every few beats; a shot that starts in this style and slides toward "
    "ordinary footage has failed."
)

# Everything the user chose still applies — it just arrives rendered in the
# style. This is the clause that makes accent/music/wardrobe/POV survive.
_STYLE_ABSORB = (
    " Everything else the request asks for still happens, rendered in this "
    "style rather than replaced by it: the people and what they look like, "
    "wardrobe, the accent and the way lines are spoken, any music and the "
    "motion that rides it, the camera behaviour, the setting. A voice, a song "
    "or an outfit does not become photoreal because it was named — it belongs "
    "to this world and is drawn, built, or shot the way this world is."
)


# ═══ NAMED UNIVERSES ════════════════════════════════════════════════════════
# Style references, not content. "In the visual style of" keeps these a LOOK
# instruction and lets them degrade to nothing if the writer model doesn't hold
# the reference — no examples, no quotes, no plot, no character writing.
_UNIVERSES = {
    "spongebob": (
        "VISUAL STYLE — in the visual style of the SpongeBob SquarePants "
        "cartoon. Everything in frame comes from that universe and is drawn "
        "the way it is drawn: hand-drawn 2D animation, wobbling uneven "
        "outlines, flat saturated fills, bulging expressive eyes, rubbery "
        "squash-and-stretch bodies, an undersea world of soft porous shapes "
        "and bright coral colour, painted backgrounds noticeably softer than "
        "the sharp-outlined characters on top of them. "
        "Never photoreal, never 3D-rendered, never realistic anatomy or "
        "realistic water."
    ),
    "rick_morty": (
        "VISUAL STYLE — in the visual style of the Rick and Morty cartoon. "
        "Everything in frame comes from that universe and is drawn the way it "
        "is drawn: flat 2D animation with thin uniform dark outlines, simple "
        "bean-shaped heads, small dot pupils with heavy brows, minimal cel "
        "shading, a slightly sickly green-and-beige palette against clean "
        "flat backgrounds, sci-fi clutter and portal-glow greens. "
        "Never photoreal, never 3D-rendered, never soft painterly shading."
    ),
    "lego_batman": (
        "VISUAL STYLE — in the visual style of the Lego Batman films. "
        "Everything in frame is built from plastic construction bricks and "
        "minifigures: glossy injection-moulded surfaces with visible moulding "
        "seams and stud texture, cylindrical hands that grip, rigid limbs that "
        "rotate only at shoulder and hip, printed faces that change by "
        "swapping rather than deforming, a whole world of brick-built "
        "architecture, brick fire, brick water. Dense saturated primary "
        "colours under glossy CG lighting. "
        "Never organic anatomy, never bending plastic, never a photoreal "
        "human, never a smoothly deforming face."
    ),
}


# ═══ ANIMATION ══════════════════════════════════════════════════════════════
_ANIMATION = {
    "pixar_3d": (
        "VISUAL STYLE — 3D family animation, the polished feature-film look. "
        "Rounded appealing shapes, slightly oversized eyes and heads, soft "
        "subsurface skin that glows where light passes through it, hair and "
        "cloth simulated in soft clumps rather than single strands, warm "
        "bounce light and gentle rim light on every figure, clean depth of "
        "field. "
        "Never photoreal skin pores, never live-action footage, never flat 2D."
    ),
    "anime_cel": (
        "VISUAL STYLE — cel-shaded anime. Crisp ink linework, colour laid in "
        "flat cels with hard-edged shadow shapes rather than gradients, large "
        "detailed eyes with specular highlights, hair rendered as sculpted "
        "clumps with a glossy band across it, dramatic rim light against a "
        "painted background. Emotion carries in the face and in held poses. "
        "Never photoreal, never soft airbrushed shading, never 3D-rendered "
        "skin. "
        "Camera follows the conventions of the form: sudden push-ins on a "
        "reaction, held wide shots with slow pans, occasional canted angle, "
        "brief speed-lines or impact framing on a hard movement."
    ),
    "cartoon_bold": (
        "VISUAL STYLE — bold-outline television cartoon. Thick black outlines "
        "of even weight around every shape, flat saturated fills with simple "
        "two-tone cel shading, exaggerated proportions and expressions, "
        "elastic squash-and-stretch on every movement, simple graphic "
        "backgrounds well behind the characters. "
        "Never photoreal, never gradient shading, never realistic proportion. "
        "Movement holds and snaps rather than flowing evenly — poses land and "
        "hold for a beat before the next one."
    ),
    "flat_vector": (
        "VISUAL STYLE — flat vector illustration. Solid colour fills and clean "
        "geometric shapes, crisp even outlines or no outline at all, a small "
        "deliberate palette, shapes reading as cut paper laid over each other. "
        "No gradients, no shadows, no texture, no depth cues, no photoreal "
        "detail whatsoever. "
        "Motion is graphic: shapes slide, rotate and scale cleanly rather than "
        "deforming organically."
    ),
    "claymation": (
        "VISUAL STYLE — claymation. Every surface is modelling clay: visible "
        "fingerprints and thumb marks, slightly lumpy asymmetric forms, a "
        "faint sheen where the clay catches light, seams where parts were "
        "pressed together, colours slightly muddy from handling. Eyes are "
        "beads pressed into faces. "
        "Never smooth CG, never photoreal skin, never flat 2D. "
        "Movement is stepped and handmade — small imperfect increments, tiny "
        "jitters between poses, never a smooth continuous glide."
    ),
    "stop_motion": (
        "VISUAL STYLE — stop-motion animation of physical handmade objects. "
        "Real materials with real texture: felt, wool, cardboard, painted "
        "wood, wire, fabric. Everything is a tactile object photographed on a "
        "miniature set, with the shallow focus and practical lighting of a "
        "macro lens over a tabletop. "
        "Never CG, never 2D drawing, never photoreal live-action scale. "
        "Movement is frame-stepped — visible small increments, a slight "
        "stutter, hands and limbs settling a fraction after they stop."
    ),
    "watercolour": (
        "VISUAL STYLE — watercolour storybook illustration. Soft painted "
        "washes with visible paper grain, pigment pooling darker at the edges "
        "of a stroke, colours bleeding gently into each other, loose "
        "hand-drawn linework that does not quite close, generous white space, "
        "a muted pastel palette. "
        "Never photoreal, never hard digital edges, never saturated neon. "
        "Motion is gentle and unhurried, drifting rather than snapping."
    ),
    "comic_panel": (
        "VISUAL STYLE — comic book art. Heavy inked linework with brush-weight "
        "variation, cross-hatching for shadow, colour laid in bold flats with "
        "hard-edged highlights, visible halftone dot texture in the shading, "
        "dramatic low and high angles, strong graphic silhouettes. "
        "Never photoreal, never soft gradient shading. No panel borders, no "
        "gutters, no speech balloons, no captions, no lettering anywhere in "
        "frame — the shot is one continuous moving image, not a page."
    ),
}


# ═══ LIVE ACTION ════════════════════════════════════════════════════════════
_LIVE = {
    "film_noir": (
        "VISUAL STYLE — classic film noir. Black and white with deep crushed "
        "blacks and blown highlights, hard single-source key light throwing "
        "long shadows, venetian-blind and window-frame shadow patterns across "
        "faces and walls, wet streets and low haze catching the light, faces "
        "half in darkness, low and canted angles, visible film grain. "
        "Never flat even lighting, never colour, never soft modern digital "
        "cleanliness."
    ),
    "neo_noir": (
        "VISUAL STYLE — neo-noir. Night, rain and reflection, saturated neon "
        "practicals as the key light in cyan and magenta against deep blue "
        "shadow, wet asphalt doubling every light source, atmospheric haze "
        "with visible light shafts, anamorphic flare streaking across the "
        "frame, faces lit from below and to the side by signage. "
        "Never daylight, never flat white light, never a dry clean street."
    ),
    "film_70s": (
        "VISUAL STYLE — 1970s analog film. Shot on grainy 35mm with warm faded "
        "colour, slightly milky lifted blacks, gentle halation blooming around "
        "highlights, soft edges and low contrast, zoom lens breathing rather "
        "than a dolly, period-correct practical lighting. "
        "Never crisp digital sharpness, never clean modern colour grading, "
        "never pure black."
    ),
    "vhs": (
        "VISUAL STYLE — VHS camcorder footage. Low resolution and soft focus, "
        "chromatic smearing on saturated colour, visible scanlines and tape "
        "noise, occasional tracking distortion across the frame, blown-out "
        "highlights from an on-camera light, slightly wrong white balance, "
        "interlaced motion trailing behind fast movement. "
        "Never sharp, never clean, never cinematic grading. Handheld and "
        "amateur — the operator is a person, not a rig."
    ),
    "documentary": (
        "VISUAL STYLE — observational documentary realism. Available light "
        "only, imperfect and unflattering; handheld camera that reframes and "
        "hunts for the subject, occasionally late to a movement; long lens "
        "from a real distance; no staging, no glamour lighting, ordinary "
        "surfaces and clutter left in frame. "
        "Never stylised grading, never a lit-for-camera look, never a locked "
        "cinematic frame."
    ),
    "music_video": (
        "VISUAL STYLE — music video. High-gloss and high-contrast, saturated "
        "colour and coloured gels, hard specular highlights, haze and "
        "backlight rimming every figure, performance-lit rather than "
        "naturally lit, deliberate lens flare, shallow focus. "
        "Never documentary flatness, never natural available light. The "
        "camera is an active performer — pushing, orbiting, and moving on the "
        "beat rather than observing."
    ),
    "horror": (
        "VISUAL STYLE — horror cinematography. Underexposed with detail "
        "swallowed in shadow, a single cold hard source raking across the "
        "frame, deep pools of blackness that the subject moves through, "
        "desaturated sickly colour, unstable handheld framing that holds too "
        "long on empty space, foreground objects obscuring part of the view. "
        "Never bright, never evenly lit, never warm and reassuring."
    ),
    "western": (
        "VISUAL STYLE — western. Wide anamorphic framing with vast sky and low "
        "horizon, harsh overhead sun and hard-edged shadow, dust and heat "
        "haze in the air, sunbaked ochre and dust-blue palette, faces "
        "weathered and squinting in the glare, long lenses compressing "
        "distance. "
        "Never soft diffused light, never lush green, never cool modern "
        "colour."
    ),
    "fashion": (
        "VISUAL STYLE — high-fashion editorial. Precisely controlled studio "
        "light, a crisp key with sculpted falloff and a clean rim separating "
        "the figure from a seamless background, immaculate glossy skin and "
        "fabric texture rendered in fine detail, deliberate symmetrical "
        "composition, restrained palette, shallow focus with a long lens. "
        "Never candid, never available light, never cluttered."
    ),
}


STYLES = {"off": ""}
STYLES.update(_UNIVERSES)
STYLES.update(_ANIMATION)
STYLES.update(_LIVE)

# Grouped for the panel dropdown (<optgroup>). Order is the display order.
STYLE_GROUPS = [
    ("Off", [("off", "Style — off (photoreal)")]),
    ("Cartoon universes", [
        ("spongebob", "SpongeBob"),
        ("rick_morty", "Rick and Morty"),
        ("lego_batman", "Lego Batman"),
    ]),
    ("Animation", [
        ("pixar_3d", "Pixar / 3D animation"),
        ("anime_cel", "Anime (cel-shaded)"),
        ("cartoon_bold", "Bold-outline cartoon"),
        ("flat_vector", "Flat vector"),
        ("claymation", "Claymation"),
        ("stop_motion", "Stop-motion"),
        ("watercolour", "Watercolour storybook"),
        ("comic_panel", "Comic book"),
    ]),
    ("Live action", [
        ("film_noir", "Film noir"),
        ("neo_noir", "Neo-noir / neon"),
        ("film_70s", "70s analog film"),
        ("vhs", "VHS camcorder"),
        ("documentary", "Documentary realism"),
        ("music_video", "Music video"),
        ("horror", "Horror"),
        ("western", "Western"),
        ("fashion", "Fashion editorial"),
    ]),
]

STYLE_KEYS = [k for _, opts in STYLE_GROUPS for k, _ in opts]
STYLE_LABELS = {k: lab for _, opts in STYLE_GROUPS for k, lab in opts}

# Styles whose medium is drawn/built rather than photographed. Used to gate the
# negative prompt so the sampler is pushed away from photoreal too.
_NON_PHOTO = set(_UNIVERSES) | set(_ANIMATION)


def is_animated(key: str) -> bool:
    return (key or "off").strip().lower() in _NON_PHOTO


def style_law(key: str) -> str:
    """Full STYLE block, or '' when off."""
    body = STYLES.get((key or "off").strip().lower(), "")
    if not body:
        return ""
    return "\n" + body + _STYLE_ABSORB + _STYLE_ENFORCE


def style_opener(key: str, after: str = "") -> str:
    """The ordering rule: the style is declared before anything else.

    Lightricks' guidance is explicit that style vocabulary has to arrive early
    or the model treats it as scene content and reverts to photoreal. This
    inverts the usual 'sentence one is place, light, who' rule for styled shots.

    `after` resolves a real collision. FIRST PERSON reserves the exact opening
    tokens "POV shot, first-person camera" — a finding that cost several
    renders — and this block used to demand the style declaration be "the very
    first words" with no exception. Two absolute claims on position zero, on a
    config (POV + style) that is one dropdown away at all times. When `after` is
    supplied the style yields position one and takes position two, which keeps
    it early enough to do its job.
    """
    if not STYLES.get((key or "off").strip().lower(), ""):
        return ""
    if (after or "").strip():
        head = (
            f"The script opens with the words {after.strip()} — that stays "
            "first. The style declaration comes IMMEDIATELY after it, still "
            "ahead of the place and ahead of anyone in it. "
        )
    else:
        head = (
            "The style declaration opens the script — the very first words "
            "name what this is made of, before the place, before anyone in "
            "it. "
        )
    return (
        "\nSTYLE FIRST\n" + head +
        "Then write the shot inside it. A style named late, or named once and "
        "dropped, is a style the render will ignore."
    )


def style_negative(key: str) -> str:
    """Negative-prompt additions so the sampler is pushed the same way."""
    k = (key or "off").strip().lower()
    if not STYLES.get(k, ""):
        return ""
    if k in _NON_PHOTO:
        return ("photorealistic, photoreal, live action, real human skin, "
                "film grain, photographic")
    # Live-action looks: push away from the drawn side.
    return "cartoon, anime, 3D render, illustration, drawing, CGI"
