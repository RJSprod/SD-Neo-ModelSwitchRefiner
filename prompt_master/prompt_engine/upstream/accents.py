"""
accents.py — Claude Prompt LD
One line per accent: rhythm + one or two sound-feel cues + character of the
delivery. Principle-only, no example words — a 26B already knows what
French-accented English sounds like; the note just tells it which dials to turn.
"""

ACCENTS = {
    "off": "",
    # East / Southeast Asia
    "korean": "Korean-accented English — even syllable timing, softened f/v, a bright lift at sentence ends.",
    "japanese": "Japanese-accented English — gentle even rhythm, every vowel fully voiced, r and l blurred softly.",
    "mandarin": "Mandarin-accented English — clipped word endings, tone-colored emphasis, articles sometimes dropped.",
    "thai": "Thai-accented English — soft unreleased endings, a melodic rise, a polite trailing habit.",
    "vietnamese": "Vietnamese-accented English — short dropped finals, quick light rhythm.",
    "filipino": "Filipino English — crisp clear vowels, musical up-down phrasing, warm directness.",
    # Europe
    "french": "French-accented English — dropped h, stress pushed to phrase ends, th softened toward z.",
    "spanish_castilian": "Castilian-accented English — crisp vowels, a soft onset before s-clusters, lisped c and z.",
    "spanish_latin": "Latin-American accented English — warm open vowels, rolling musical rhythm.",
    "italian": "Italian-accented English — open vowel endings, wide melodic swings, doubled emphasis on feeling words.",
    "portuguese_br": "Brazilian-accented English — soft ch color on ti and di, final l rounding toward w, sunny rhythm.",
    "german": "German-accented English — firm consonants, w toward v, front-loaded stress, precise pacing.",
    "dutch": "Dutch-accented English — flat clear tone, slightly hard g, blunt direct phrasing.",
    "swedish": "Swedish-accented English — sing-song pitch swings, long pure vowels, j toward y.",
    "norwegian": "Norwegian-accented English — lilting rise-and-fall melody, clean consonants.",
    "russian": "Russian-accented English — flat deliberate rhythm, dark hard consonants, dropped articles, zero filler.",
    "polish": "Polish-accented English — dense consonant attack, level intonation, sincere weight on verbs.",
    "czech": "Czech-accented English — first-syllable stress throughout, even dry rhythm.",
    "greek": "Greek-accented English — rolling emphatic rhythm, s colored toward sh, hands-in-the-voice energy.",
    # Middle East / Africa
    "arabic": "Arabic-accented English — emphatic consonants, p toward b, generous length on stressed vowels.",
    "hebrew": "Hebrew-accented English — guttural r, punchy stress, fast confident clip.",
    "swahili": "Swahili-accented English — even open syllables, steady warm cadence.",
    "nigerian": "Nigerian English — syllable-timed bounce, strong clear vowels, proverb-ready gravitas.",
    "ghanaian": "Ghanaian English — measured syllable timing, rounded vowels, calm authority.",
    "south_african": "South African English — flattened i, clipped precise vowels, dry level delivery.",
    # Caribbean
    "jamaican": "Jamaican-accented English — patois melody, th toward d, phrase-final stress, easy swagger.",
    "trinidadian": "Trinidadian English — sing-song bounce, quick light vowels, playful phrasing.",
    # South Asia
    "indian": "Indian English — retroflex t and d color, syllable-timed rhythm, formal vocabulary at ease.",
    # British Isles
    "rp_british": "RP British English — non-rhotic, long a, clipped precision, understated delivery.",
    "cockney": "Cockney English — dropped h and t, glottal stops, quick cheeky rhythm.",
    "scottish": "Scottish English — rolled r, short tense vowels, rising-falling burr.",
    "irish": "Irish English — melodic lilt, soft t toward ch, unhurried musical phrasing.",
    "scouse": "Scouse English — nasal color, fricative k endings, fast rising melody.",
    "geordie": "Geordie English — sing-song rise at phrase ends, rounded vowels, warm quick clip.",
    "northern_english": "Northern English — short flat a, dropped g on -ing, plain-spoken warmth.",
    "welsh": "Welsh English — lilting up-down melody, doubled emphasis, breathy warmth.",
    # Oceania / Americas
    "australian": "Australian English — rising ends, stretched flattened vowels, dry laid-back delivery.",
    "new_zealand": "New Zealand English — centralized short i, clipped quick vowels, mild even tone.",
    "southern_us": "Southern US English — drawled long vowels, dropped g, unhurried molasses cadence.",
    "american": "General American — the baseline the others bend away from, so there is little to respell: even stress, full r-coloured vowels, no regional colour at all.",
    "new_york": "New York English — r dropped off the end of a syllable, the vowel stretching and taking an h in spelling; aw tightened and raised; th left exactly as it is, never flattened to d; fast, front-loaded delivery.",
    "californian": "Californian English — g softened off -ing, t swallowed between vowels, vowels drawn long, statements rising at the end into a creaky finish.",
    "midwest_us": "Midwestern US English — a flattened long and nasal, o clipped short, t softening toward d between vowels, plain delivery with little rise or fall.",
    "boston": "Boston English — r dropped off the end of a syllable, the vowel opening wide and taking an h in spelling; a broadened toward ah; stress landing early in the phrase.",
    "canadian": "Canadian English — ou raised toward oo, t kept crisp, and a short tag question hung on the end of a statement.",
    "appalachian": "Appalachian English — r held hard, long vowels broken into two, a-prefixed verbs, older forms kept alive, unhurried mountain cadence.",
    "aave": "African-American Vernacular English — final consonant clusters reduced, th toward d or f, habitual and copula forms of its own, stress riding the content word.",
}

ACCENT_KEYS = list(ACCENTS.keys())

# ── accent KIND: this is the distinction the VOICE law was missing ────────────
# Two different things were sharing one rule.
#
#   l2      — a speaker of another language speaking English. The target is
#             English with that accent, and native vocabulary genuinely IS
#             contamination (the French "mon chéri" + German "vant" render).
#             ONE LANE is correct here and stays.
#
#   variety — a variety or creole of English. The lexicon, idiom and grammar
#             ARE the accent, not a foreign slip that has to yield to English.
#             Capping those at "one native slip, zero is fine" removes the
#             single thing that makes the accent recognisable — which is why
#             Jamaican shots came back as standard English with d-for-th and
#             nothing else.
#
# Judgement call on the borderline ones: Indian and Filipino English are listed
# as varieties because each has its own English vocabulary and sentence shapes,
# but the REGISTER clause still forbids whole lines in another language — that
# is what the Hindi-translation slip was, and it stays banned. "swahili" is the
# LANGUAGE, so it is l2; a Kenyan-English entry would be a variety.
_VARIETIES = {
    "jamaican", "trinidadian",
    "nigerian", "ghanaian", "south_african",
    "indian", "filipino",
    "rp_british", "cockney", "scottish", "irish", "scouse", "geordie",
    "northern_english", "welsh",
    "australian", "new_zealand", "southern_us",
    "american", "new_york", "californian", "midwest_us", "boston", "canadian",
    "aave", "appalachian",
}


def accent_kind(key: str) -> str:
    """"variety" for an English variety/creole, "l2" otherwise. "" when off."""
    k = (key or "off").strip().lower()
    if k == "off" or k not in ACCENTS:
        return ""
    return "variety" if k in _VARIETIES else "l2"


def accent_words(key: str):
    """Words that would appear in an intent naming this accent, so the brief can
    tell "a Jamaican woman dances" from "she dances" and decide whether the
    on-screen speaker shares the viewer's accent.

    Derived from the key rather than a second hand-maintained table, so it
    cannot drift out of sync: short tokens (rp, us, br) are dropped because they
    are not words anyone types in an intent — and "new" (from new_zealand) is
    dropped by the same length rule, which matters because "a new dress" would
    otherwise read as the intent naming the accent.
    """
    k = (key or "off").strip().lower()
    if k == "off" or k not in ACCENTS:
        return ()
    return tuple(t for t in k.split("_") if len(t) > 3)


# Title-casing the key is right for almost every accent and wrong for the few
# that are initialisms or have a fixed spelling.
_LABEL_FIX = {"aave": "AAVE", "appalachian": "Appalachian (hillbilly)", "rp_british": "RP British",
              "new_zealand": "New Zealand", "southern_us": "Southern US",
              "midwest_us": "Midwest US", "new_york": "New York",
              "portuguese_br": "Portuguese (BR)",
              "spanish_castilian": "Spanish (Castilian)",
              "spanish_latin": "Spanish (Latin America)"}


def accent_label(key: str) -> str:
    k = (key or "off").strip().lower()
    if k == "off":
        return "Accent — off"
    return _LABEL_FIX.get(k) or k.replace("_", " ").title()


def accent_note(key: str) -> str:
    return ACCENTS.get((key or "off").strip().lower(), "")


# ── accent STRENGTH ──────────────────────────────────────────────────────────
# How hard to lay it on. The only honest levers without example words are
# DENSITY (how much of each line the accent touches), GRAMMAR (whether sentence
# shape follows the accent or standard English), FREQUENCY (once a shot vs every
# line) and NON-WORDS (the laugh, the hesitation, the noise made instead of a
# word — an accent lives in those too, and they cost no vocabulary at all).
STRENGTHS = ("natural", "strong", "thick")

_DENSITY = {
    "natural": ("Respell a few words in every line spoken BY A SPEAKER WHO HAS "
                "THIS ACCENT"),
    "strong": ("Respell most of the words this accent actually bends, in every "
               "line spoken BY A SPEAKER WHO HAS THIS ACCENT"),
    "thick": ("Write the whole line the way this accent says it — spelling, "
              "elisions, dropped and added consonants throughout — in every "
              "line spoken BY A SPEAKER WHO HAS THIS ACCENT"),
}

# Strength no longer adds a rule 4. It scales rules 1 and 3, because a fourth
# clause telling the model to "write it in the dialect" was the third one saying
# so — and with three overlapping demands and no anchor, a New York shot came
# back with generic "de/dis/ting" spellings and never touched the dropped r the
# note actually names.
_EXTRA = {
    "natural": "",
    "strong": (
        " Lay it on: this accent's own address terms, tags and intensifiers "
        "belong in MOST lines rather than once a shot, and sentence shape follows "
        "this accent rather than standard word order."
    ),
    "thick": (
        " Lay it on hard: this accent's own words, particles, tag questions, "
        "contractions and elisions in EVERY line, consistently. Put it in the "
        "non-words too — the hesitation sound, the laugh, the breath, the noise "
        "made instead of a word. Stress goes INSIDE the quote — the stressed "
        "word in CAPS — never described in a sentence after it."
    ),
}


def accent_strength(level: str) -> str:
    lv = (level or "natural").strip().lower()
    return lv if lv in STRENGTHS else "natural"


def density_rule(level: str) -> str:
    return _DENSITY[accent_strength(level)]


def strength_law(level: str) -> str:
    return _EXTRA[accent_strength(level)]
