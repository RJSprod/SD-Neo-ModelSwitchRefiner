"""
identity.py — Claude Prompt LD

Seeded PEOPLE identity. An accent implies a region; a region has a realistic
spread of hair, eyes, skin and build. Traits are drawn from WEIGHTED pools so
the typical is common and the movie-stereotype is rare (a Scottish redhead
turns up about a tenth of the time, not nine tenths).

Rules of the road:
  * The intent always wins. Anything the author states is never overwritten;
    the seed only fills what was left blank.
  * Weights are (weight, value) — higher weight, more likely.
  * No accent chosen → the GLOBAL pool, a wide mixed-population spread.
"""

import random
import re

# ── shared trait vocabularies ────────────────────────────────────────────────
# Female-coded terms are separated out rather than filtered by string match, so
# adding a term to either pool cannot silently leak into the wrong one.
_PRO = {
    False: {"sub": "she", "obj": "her", "pos": "her", "noun": "woman"},
    True: {"sub": "he", "obj": "him", "pos": "his", "noun": "man"},
}

_BUILD_HIM = [
    (4, "slim"), (4, "athletic"), (3, "average build"), (2, "broad-shouldered"),
    (2, "stocky"), (2, "wiry"), (1, "lanky"), (1, "heavyset"), (1, "barrel-chested"),
]

_BUILD = [
    (4, "slim"), (4, "athletic"), (3, "curvy"), (3, "average build"),
    (2, "petite"), (2, "hourglass"), (2, "soft-figured"), (1, "broad-shouldered"),
    (1, "lanky"), (1, "stocky"),
]
# Plain word, per the ANATOMY law's own rule. The old pool said "full chest" /
# "large chest", so on every undress shot CAST LOOK told the model to write
# "chest" in sentence one while ANATOMY, four sections later, said bare breasts
# are "breasts" — not chest, curves, or figure. The seed was arguing with the
# law on 100% of explicit shots. This pool is also now gated: on a clothed shot
# it does not ship at all, because THE SHOT already says clothed is garment plus
# shape, so a bust trait there is a word spent on something the law will not use.
_BUST = [
    (4, "small breasts"), (5, "average breasts"), (4, "full breasts"),
    (2, "large breasts"),
]
_HEIGHT_MID = [(2, "short"), (5, "average height"), (3, "tall")]
_HEIGHT_TALL = [(1, "short"), (4, "average height"), (5, "tall")]
_HEIGHT_PETITE = [(4, "short"), (5, "average height"), (1, "tall")]

_HAIR_LEN = [
    (4, "long"), (3, "shoulder-length"), (2, "short"), (2, "a bob"),
    (1, "cropped"), (2, "tied back"),
]
_HAIR_TEX = [(4, "straight"), (3, "wavy"), (2, "curly"), (1, "tight curls")]  # global default

# per-region hair texture — a shared pool renders the wrong ethnicity
_TEX_STRAIGHT = [(7, "straight"), (3, "gently wavy"), (1, "loosely wavy")]
_TEX_MIXED = [(4, "straight"), (4, "wavy"), (2, "curly")]
_TEX_COILED = [(5, "tightly coiled"), (3, "in braids"), (2, "in locs"), (2, "natural curls")]

# ── regional pools ───────────────────────────────────────────────────────────
# hair / eyes / skin weighted per region. Stereotype held to a low weight.
_REGIONS = {
    "south_asian": {
        "hair": [(9, "black"), (3, "very dark brown"), (1, "dark brown")],
        "eyes": [(9, "dark brown"), (3, "brown"), (1, "hazel")],
        "skin": [(4, "brown"), (4, "deep brown"), (3, "warm tan"), (2, "light brown")],
        "tex": _TEX_STRAIGHT,
        "features": "warm undertones, defined dark brows, deep-set dark eyes, South Asian features",
        "height": _HEIGHT_PETITE,
    },
    "east_asian": {
        "hair": [(9, "black"), (2, "dark brown"), (1, "dyed light brown")],
        "eyes": [(9, "dark brown"), (2, "black-brown")],
        "skin": [(5, "fair"), (4, "light golden"), (2, "warm tan")],
        "tex": _TEX_STRAIGHT,
        "features": "East Asian features, smooth straight hair, softer brow line",
        "height": _HEIGHT_PETITE,
    },
    "nordic": {
        "hair": [(4, "blonde"), (3, "ash blonde"), (3, "light brown"), (2, "brown"), (1, "red")],
        "eyes": [(5, "blue"), (3, "grey-blue"), (2, "green"), (2, "brown")],
        "skin": [(6, "pale"), (3, "fair"), (1, "lightly freckled")],
        "tex": _TEX_MIXED,
        "features": "Northern European features, fine straight-to-wavy hair, light brows",
        "height": _HEIGHT_TALL,
    },
    "british_isles": {
        "hair": [(5, "brown"), (3, "dark brown"), (2, "dark blonde"), (2, "black"), (1, "red")],
        "eyes": [(4, "blue"), (4, "brown"), (3, "green"), (2, "hazel")],
        "skin": [(5, "fair"), (3, "pale"), (2, "freckled"), (1, "olive")],
        "tex": _TEX_MIXED,
        "features": "Northern European features",
        "height": _HEIGHT_MID,
    },
    "slavic": {
        "hair": [(4, "light brown"), (4, "dark blonde"), (3, "brown"), (2, "blonde"), (1, "black")],
        "eyes": [(4, "blue"), (3, "grey"), (3, "green"), (3, "brown")],
        "skin": [(5, "fair"), (3, "pale"), (2, "light olive")],
        "tex": _TEX_MIXED,
        "features": "Eastern European features, high cheekbones",
        "height": _HEIGHT_MID,
    },
    "mediterranean": {
        "hair": [(6, "dark brown"), (4, "black"), (2, "chestnut"), (1, "auburn")],
        "eyes": [(6, "dark brown"), (3, "hazel"), (2, "green")],
        "skin": [(5, "olive"), (3, "light olive"), (2, "warm tan")],
        "tex": _TEX_MIXED,
        "features": "Southern European features, strong dark brows",
        "height": _HEIGHT_MID,
    },
    "latin": {
        "hair": [(6, "black"), (4, "dark brown"), (2, "brown"), (1, "caramel-highlighted")],
        "eyes": [(6, "dark brown"), (3, "brown"), (1, "green")],
        "skin": [(4, "tan"), (4, "warm brown"), (3, "olive"), (2, "light brown")],
        "tex": _TEX_MIXED,
        "features": "Latin American features, dark defined brows",
        "height": _HEIGHT_PETITE,
    },
    "african": {
        "hair": [(8, "black"), (2, "dark brown"), (1, "auburn-tinted black")],
        "eyes": [(9, "dark brown"), (2, "brown")],
        "skin": [(5, "deep brown"), (4, "dark brown"), (3, "rich brown")],
        "tex": _TEX_COILED,
        "features": "West African features, full lips, broad nose bridge",
        "height": _HEIGHT_MID,
    },
    "middle_eastern": {
        "hair": [(6, "black"), (4, "dark brown"), (1, "chestnut")],
        "eyes": [(6, "dark brown"), (3, "hazel"), (2, "green")],
        "skin": [(4, "olive"), (4, "light brown"), (2, "warm tan")],
        "tex": _TEX_MIXED,
        "features": "Middle Eastern features, strong dark brows, aquiline nose",
        "height": _HEIGHT_MID,
    },
    "north_american": {
        "hair": [(4, "brown"), (3, "dark brown"), (2, "blonde"), (2, "black"), (1, "auburn")],
        "eyes": [(4, "brown"), (3, "blue"), (2, "hazel"), (2, "green")],
        "skin": [(3, "fair"), (3, "tan"), (2, "olive"), (2, "brown"), (1, "deep brown")],
        "tex": _TEX_MIXED,
        "height": _HEIGHT_MID,
    },
    "global": {  # no accent chosen — wide mixed spread
        "hair": [(4, "brown"), (4, "black"), (3, "dark brown"), (2, "blonde"),
                 (2, "light brown"), (1, "red"), (1, "auburn")],
        "eyes": [(5, "brown"), (4, "dark brown"), (3, "blue"), (2, "green"), (2, "hazel")],
        "skin": [(4, "fair"), (3, "olive"), (3, "tan"), (3, "brown"),
                 (2, "deep brown"), (2, "pale")],
        "tex": _HAIR_TEX,
        "height": _HEIGHT_MID,
    },
}

# accent key → region + the ethnicity word used to anchor the opening line
_ACCENT_REGION = {
    # keys matching accents.py exactly — a missing key silently falls to GLOBAL
    "mandarin": ("east_asian", "Chinese"), "portuguese_br": ("latin", "Brazilian"),
    "hebrew": ("middle_eastern", "Israeli"), "swahili": ("african", "East African"),
    "trinidadian": ("african", "Trinidadian"), "rp_british": ("british_isles", "English"),
    "northern_english": ("british_isles", "English"), "southern_us": ("north_american", "American"),
    "american": ("north_american", "American"), "new_york": ("north_american", "American"), "californian": ("north_american", "American"), "midwest_us": ("north_american", "American"), "boston": ("north_american", "American"), "canadian": ("north_american", "American"), "aave": ("north_american", "American"), "appalachian": ("north_american", "American"),
    "indian": ("south_asian", "Indian"), "pakistani": ("south_asian", "Pakistani"),
    "bengali": ("south_asian", "Bengali"), "sri_lankan": ("south_asian", "Sri Lankan"),
    "nepali": ("south_asian", "Nepali"),
    "chinese": ("east_asian", "Chinese"), "japanese": ("east_asian", "Japanese"),
    "korean": ("east_asian", "Korean"), "vietnamese": ("east_asian", "Vietnamese"),
    "thai": ("east_asian", "Thai"), "filipino": ("east_asian", "Filipina"),
    "swedish": ("nordic", "Swedish"), "norwegian": ("nordic", "Norwegian"),
    "danish": ("nordic", "Danish"), "finnish": ("nordic", "Finnish"),
    "icelandic": ("nordic", "Icelandic"), "dutch": ("nordic", "Dutch"),
    "german": ("nordic", "German"),
    "english_rp": ("british_isles", "English"), "cockney": ("british_isles", "English"),
    "scottish": ("british_isles", "Scottish"), "irish": ("british_isles", "Irish"),
    "welsh": ("british_isles", "Welsh"), "geordie": ("british_isles", "English"),
    "scouse": ("british_isles", "English"), "yorkshire": ("british_isles", "English"),
    "australian": ("british_isles", "Australian"), "new_zealand": ("british_isles", "New Zealand"),
    "south_african": ("british_isles", "South African"),
    "polish": ("slavic", "Polish"), "russian": ("slavic", "Russian"),
    "ukrainian": ("slavic", "Ukrainian"), "czech": ("slavic", "Czech"),
    "serbian": ("slavic", "Serbian"), "romanian": ("slavic", "Romanian"),
    "hungarian": ("slavic", "Hungarian"),
    "italian": ("mediterranean", "Italian"), "spanish_castilian": ("mediterranean", "Spanish"),
    "greek": ("mediterranean", "Greek"), "portuguese": ("mediterranean", "Portuguese"),
    "french": ("mediterranean", "French"),
    "spanish_latin": ("latin", "Latina"), "brazilian": ("latin", "Brazilian"),
    "mexican": ("latin", "Mexican"), "argentine": ("latin", "Argentine"),
    "cuban": ("latin", "Cuban"), "colombian": ("latin", "Colombian"),
    "nigerian": ("african", "Nigerian"), "ghanaian": ("african", "Ghanaian"),
    "kenyan": ("african", "Kenyan"), "ethiopian": ("african", "Ethiopian"),
    "jamaican": ("african", "Jamaican"), "caribbean": ("african", "Caribbean"),
    "arabic": ("middle_eastern", "Arab"), "egyptian": ("middle_eastern", "Egyptian"),
    "lebanese": ("middle_eastern", "Lebanese"), "turkish": ("middle_eastern", "Turkish"),
    "persian": ("middle_eastern", "Persian"), "israeli": ("middle_eastern", "Israeli"),
    "american_south": ("north_american", "American"), "american_ny": ("north_american", "American"),
    "american_midwest": ("north_american", "American"), "canadian": ("north_american", "Canadian"),
    "american_valley": ("north_american", "American"),
}

# author already described the person → don't seed over them
_DESCRIBED = re.compile(
    r"(?i)\b(blonde?|brunette|redhead|ginger|black[- ]haired|brown[- ]haired|"
    r"dark[- ]haired|pale|tanned?|olive|freckled|petite|curvy|slim|slender|"
    r"athletic|busty|tall|short|skinny|thick|plump|muscular|"
    r"blue[- ]eyed|green[- ]eyed|brown[- ]eyed|hazel)\b")

# an explicit ethnicity/nationality in the intent anchors it by itself
_ETHNIC_WORD = re.compile(
    r"(?i)\b(indian|pakistani|bengali|chinese|japanese|korean|thai|vietnamese|"
    r"filipina|filipino|swedish|norwegian|danish|finnish|dutch|german|english|"
    r"scottish|irish|welsh|australian|polish|russian|ukrainian|czech|italian|"
    r"spanish|greek|portuguese|french|latina|brazilian|mexican|cuban|colombian|"
    r"nigerian|ghanaian|kenyan|ethiopian|jamaican|caribbean|arab|egyptian|"
    r"lebanese|turkish|persian|israeli|american|canadian|asian|black|white|"
    r"african|european|middle[- ]eastern)\b")


def _pick(rng, pool):
    total = sum(w for w, _ in pool)
    r = rng.uniform(0, total)
    upto = 0.0
    for w, v in pool:
        upto += w
        if r <= upto:
            return v
    return pool[-1][1]


_REGION_ALIAS = {
    "african": ("african", "African"), "south_asian": ("south_asian", "South Asian"),
    "east_asian": ("east_asian", "East Asian"), "latin": ("latin", "Latina"),
    "nordic": ("nordic", "Scandinavian"), "slavic": ("slavic", "Eastern European"),
    "mediterranean": ("mediterranean", "Mediterranean"),
    "middle_eastern": ("middle_eastern", "Middle Eastern"),
}


def region_for(accent: str):
    a = (accent or "").strip().lower()
    if a in _ACCENT_REGION:
        return _ACCENT_REGION[a]
    return _REGION_ALIAS.get(a, ("global", ""))


def identity_block(accent: str, seed: int, intent: str, pov: str = "off",
                   explicit: bool = False, subject: str = "her") -> str:
    """Seed look from accent region + seed.

    Third-person (pov off): CAST LOOK for the on-screen person.
    First-person (male/female): VIEWER LOOK for hands/skin only — the accent
    is *your* voice and *your* hands, not the person in front of the camera.

    explicit: the shot will show bare skin (brain.has_anatomy). Only then does
    the bust trait ship, and only in the plain word the ANATOMY law demands.
    """
    it = intent or ""
    region_key, ethnic = region_for(accent)
    spec = _REGIONS.get(region_key, _REGIONS["global"])
    rng = random.Random(int(seed or 0) ^ 0x9E1D)

    stated_look = bool(_DESCRIBED.search(it))
    stated_ethnic = bool(_ETHNIC_WORD.search(it))

    hair = (f"{_pick(rng, _HAIR_LEN)} {_pick(rng, spec.get('tex', _HAIR_TEX))} "
            f"{_pick(rng, spec['hair'])} hair")
    eyes = f"{_pick(rng, spec['eyes'])} eyes"
    skin = f"{_pick(rng, spec['skin'])} skin"
    # Hand-only phrasing (no "skin" word twice): "deep brown hands"
    skin_adj = _pick(rng, spec["skin"])
    him = (subject or "her").strip().lower() == "him"
    build = (f"{_pick(rng, spec['height'])}, "
             f"{_pick(rng, _BUILD_HIM if him else _BUILD)}")
    bust = _pick(rng, _BUST) if (explicit and not him) else ""

    p = (pov or "off").strip().lower()
    if p in ("male", "female"):
        vg = "man" if p == "male" else "woman"
        # Intent may describe the SUBJECT (she/he on screen) — that does not
        # overwrite the viewer's hands. Only explicit viewer-hand wording does.
        viewer_stated = bool(re.search(
            r"(?i)\b(my hands?|viewer'?s?\s+hands?|hands?\s+are\s+(?:dark|black|brown|pale|fair|olive))\b",
            it,
        ))
        lead = (
            "VIEWER LOOK (first-person — hands/skin only; never a face or body "
            "for the viewer; do not restate as notes in the shot)\n"
        )
        ethnic_bit = (
            f"The viewer is {'an' if ethnic[0] in 'AEIOU' else 'a'} "
            f"{ethnic} {vg}. "
            if ethnic else
            f"The viewer is a {vg}. "
        )
        if viewer_stated:
            hands = (
                "Hands: use exactly the skin/hand look the intent states; keep "
                "it identical every time hands appear."
            )
        else:
            hands = (
                # Skin only. Placement, count and naming belong to POV HANDS —
                # stating them here as well was the same rule in two voices.
                f"The viewer's hands are {skin_adj} — not pale default hands, "
                f"not someone else's skin — and the same {skin_adj} every beat."
            )
        mirror = (
            "Mirror self-acts only: the reflection may show a matching "
            f"{skin_adj}-skinned {vg}"
            + (f" ({ethnic})" if ethnic else "")
            + " doing the intent's action; camera-side stays hands-only."
        )
        subject = (
            "Accent/seed identity is the VIEWER's. Do not paint an on-screen "
            "subject with this ethnicity or skin unless the intent says they "
            "share it — the person in front of the camera is from the intent "
            "or the still, not from the accent dial."
        )
        return "\n" + lead + ethnic_bit + hands + " " + mirror + " " + subject

    # ── third-person: on-screen cast ──────────────────────────────────────
    lead = "CAST LOOK (intent always wins; do not restate as notes in the shot)\n"
    if stated_look and stated_ethnic:
        return (
            "\n" + lead +
            "The intent already describes this person. Keep every stated trait "
            "exactly in the opening sentence; hold hair, eyes, skin, build "
            "identical for the whole shot."
        )

    parts = []
    if ethnic and not stated_ethnic:
        parts.append(
            f"Unless the intent says otherwise, {_PRO[him]['sub']} is "
            f"{ethnic} — write \"{ethnic}\" in the FIRST sentence that "
            f"introduces {_PRO[him]['obj']} (\"a {ethnic} "
            f"{_PRO[him]['noun']}…\"), not merely traits."
        )
    elif stated_ethnic:
        parts.append(
            f"The intent names {_PRO[him]['pos']} background — that exact word "
            f"must appear in the FIRST sentence describing {_PRO[him]['obj']} "
            f"(\"a Polish {_PRO[him]['noun']}…\"). "
            "Ethnicity word and traits both appear."
        )
    else:
        parts.append(
            "Name who is on screen in the FIRST sentence with a specific look — "
            "never open on an unspecified 'a woman'."
        )

    if not stated_look:
        parts.append(
            f"Where appearance is open, lean toward: {build}, "
            + (f"{bust}, " if bust else "")
            + f"{hair}, {eyes}, {skin}"
            + (f", {spec['features']}" if spec.get("features") else "")
            + ". Plain physical description in the opening; hold identical "
            "every beat."
        )
    else:
        parts.append(
            "The intent already fixes her appearance — use exactly that; hold "
            "identical every beat."
        )

    return "\n" + lead + " ".join(parts)
