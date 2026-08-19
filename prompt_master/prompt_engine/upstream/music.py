"""
music.py — Claude Prompt LD
Optional named-track line. Off by default. Genre injects one MUSIC line so LTX
has a beat to sync motion to. Background flag ducks it under action.
"""

MUSIC = {
    "off": "",
    "club_house": "a driving four-on-the-floor house track around 126 BPM, deep bass and a steady kick, bright synth stabs on the beat, building in eight-bar phrases toward a filtered drop",
    "hip_hop": "a hard hip-hop beat around 90 BPM, heavy 808 bass, crisp snare cracking on the two and four, sparse verses that open up on the hook",
    "trap": "a trap beat around 140 BPM half-time, rolling hi-hats, booming sub-bass, a slow menacing swing that drops out and slams back in",
    "techno": "a relentless techno pulse around 132 BPM, hypnotic repetitive kick, dark industrial textures layering thicker every sixteen bars",
    "dnb": "fast drum-and-bass around 174 BPM, breakneck breakbeat, a rolling reese bassline, tension risers snapping into full-throttle sections",
    "rnb": "a slow sultry R&B groove around 70 BPM, warm bass, finger snaps, smooth muted chords that swell under the chorus",
    "pop": "an upbeat pop track around 118 BPM, bright hooks, a danceable beat, verses lifting into a big glossy chorus",
    "rock": "a driving rock track around 130 BPM, distorted electric guitars, a hard live kit, verses grinding into a wide-open chorus",
    "latin": "a hot reggaeton bounce around 96 BPM, syncopated dembow percussion, brass stabs, hips-first rhythm that never lets up",
    "disco": "a classic disco groove around 116 BPM, walking bassline, four-on-the-floor kick, string swells and wah guitar rising into the chorus",
    "jazz_funk": "a smoky jazz-funk soundtrack around 100 BPM, walking upright bass, brushed drums, muted trumpet trading loose phrases",
    "ambient": "a slow ambient wash with no clear beat, low drones and spacious pads, swelling and receding like slow breathing",
    "classical": "a sweeping orchestral score, strings and swelling brass, cinematic dynamics that build from hushed to full and settle again",
    "opera": (
        "a single trained voice carrying across a hall, long sustained vowels "
        "over strings, wide vibrato, orchestra swelling beneath it and falling "
        "away, the last note held until the room rings"
    ),
    "opera_harmonics": (
        "a voice past what a throat can do — glassy overtones stacked above the "
        "melody, runs faster than breath, interval leaps landing dead centre "
        "with no scoop into them, no breath taken where a breath should be, a "
        "cold synthetic sheen over the strings underneath"
    ),
    # ── region-linked — see _FOR_ACCENT for which accent wants which ──
    "reggae": "a one-drop reggae groove around 74 BPM, deep round bass, guitar skanking on the offbeat, kick and snare landing together on the third beat, organ bubbling underneath",
    "dancehall": "a digital dancehall riddim around 100 BPM, hard syncopated kick pattern, sparse synth stabs, heavy sub-bass, the beat dropping out under the vocal and slamming back",
    "soca": "a fast soca drive around 155 BPM, rolling hand percussion, bright horn lines, a relentless carnival lift that keeps building",
    "afrobeats": "an afrobeats groove around 105 BPM, log-drum bass, shakers riding over a loose syncopated kick, bright guitar plucks, laid-back swing",
    "amapiano": "an amapiano groove around 112 BPM, deep log-drum bass sliding between notes, brushed shakers, wide jazzy keys drifting over a patient beat",
    "bollywood": "a filmi playback track, dhol and tabla under sweeping strings, a bright reed line answering the melody, building into a full percussive chorus",
    "bhangra": "a bhangra beat around 145 BPM, dhol driving hard on the downbeat, tumbi plucking a repeating hook, brass punching over the top",
    "kpop": "a glossy K-pop track around 120 BPM, tight programmed drums, synth hooks stacked in layers, a pre-chorus lift dropping into a hard rhythmic hook",
    "citypop": "an eighties city-pop groove around 108 BPM, slap bass, glassy electric piano, chorused guitar, warm saxophone lifting over the chorus",
    "flamenco": "a flamenco compas, nylon-string guitar rasgueado, palmas clapping counter-rhythm, cajon thumping underneath, tempo surging and holding back",
    "tango": "a tango around 118 BPM, bandoneon breathing the melody, staccato strings snapping on the beat, sudden stops and long dragged phrases",
    "samba": "a samba batucada, surdo thumping the low pulse, tamborim and agogo cutting across it, cavaquinho strumming fast on top",
    "celtic": "a trad session tune, fiddle and tin whistle trading the melody, bodhran driving underneath, tempo climbing as the tune repeats",
    "country": "a country track around 100 BPM, brushed drums and walking bass, telecaster twang, pedal steel bending long under the vocal line",
    "bluegrass": "a bluegrass breakdown around 140 BPM, banjo rolling continuously, fiddle and mandolin trading breaks, upright bass walking underneath, no drum kit at all",
    "grime": "a grime beat around 140 BPM, sparse square-wave bass, clipped drums, an eight-bar loop with a cold minimal menace",
    "shaabi": "a shaabi groove, darbuka and riq driving a tight loop, mizmar reed wailing over it, a hand-clap pattern that stumbles forward",
    "chanson": "a chanson waltz, accordion carrying the melody over brushed drums and upright bass, a torch vocal phrasing behind the beat",
    "cinematic": "a tense cinematic score around 80 BPM, low pulsing strings, a rhythmic ostinato tightening as the scene builds",
    "lofi": "a mellow lo-fi beat around 82 BPM, dusty vinyl crackle, soft jazzy chords, an easy head-nod tempo that never hurries",
}

MUSIC_LABELS = {
    "off": "Music — off",
    "club_house": "House / club",
    "hip_hop": "Hip-hop",
    "trap": "Trap",
    "techno": "Techno",
    "dnb": "Drum & bass",
    "rnb": "R&B",
    "pop": "Pop",
    "rock": "Rock",
    "latin": "Reggaeton (Latin)",
    "disco": "Disco",
    "jazz_funk": "Jazz-funk",
    "ambient": "Ambient",
    "classical": "Classical / orchestral",
    "opera": "Opera — carried voice",
    "opera_harmonics": "Opera — pure harmonics",
    "reggae": "Reggae (Jamaica)",
    "dancehall": "Dancehall (Jamaica)",
    "soca": "Soca (Trinidad)",
    "afrobeats": "Afrobeats (Nigeria / Ghana)",
    "amapiano": "Amapiano (South Africa)",
    "bollywood": "Bollywood / filmi (India)",
    "bhangra": "Bhangra (Punjab)",
    "kpop": "K-pop (Korea)",
    "citypop": "City pop (Japan)",
    "flamenco": "Flamenco (Spain)",
    "tango": "Tango (Argentina)",
    "samba": "Samba (Brazil)",
    "celtic": "Celtic trad (Ireland / Scotland)",
    "country": "Country (Southern US)",
    "bluegrass": "Bluegrass (Appalachia)",
    "grime": "Grime (UK)",
    "shaabi": "Shaabi (Arabic)",
    "chanson": "Chanson (France)",
    "cinematic": "Cinematic score",
    "lofi": "Lo-fi",
}

MUSIC_KEYS = list(MUSIC.keys())

# ── "which music goes with which accent" ─────────────────────────────────────
# The dropdown is a flat genre list, so the pairing a Jamaican shot obviously
# wants (dancehall or reggae, not drum-and-bass) was left to the user to know.
# This is that answer, encoded. Accents with no genre of their own map to the
# nearest sensible existing entry rather than getting one invented for them.
_FOR_ACCENT = {
    "jamaican": "dancehall", "trinidadian": "soca",
    "nigerian": "afrobeats", "ghanaian": "afrobeats", "swahili": "afrobeats",
    "south_african": "amapiano",
    "indian": "bollywood",
    "korean": "kpop", "japanese": "citypop",
    "spanish_castilian": "flamenco", "spanish_latin": "tango",
    "portuguese_br": "samba",
    "irish": "celtic", "scottish": "celtic", "welsh": "celtic",
    "southern_us": "country",
    "cockney": "grime", "scouse": "grime", "geordie": "grime",
    "northern_english": "grime",
    "arabic": "shaabi", "hebrew": "shaabi",
    "french": "chanson",
    "italian": "classical", "greek": "classical", "rp_british": "classical",
    "german": "techno", "dutch": "techno", "czech": "techno",
    "swedish": "techno", "norwegian": "techno", "polish": "techno",
    "russian": "techno",
    "mandarin": "pop", "thai": "pop", "vietnamese": "pop", "filipino": "pop",
    "australian": "rock", "new_zealand": "rock",
    "american": "pop",
    "new_york": "hip_hop",
    "californian": "pop",
    "midwest_us": "country",
    "boston": "rock",
    "canadian": "pop",
    "aave": "hip_hop",
    "appalachian": "bluegrass",
}


def music_auto(accent: str) -> str:
    """Genre that fits this accent, or "off" when there is nothing to go on."""
    return _FOR_ACCENT.get((accent or "off").strip().lower(), "off")


def resolve(key: str, accent: str = "off") -> str:
    """Turn the dropdown value into a real genre key. "auto" reads the accent."""
    k = (key or "off").strip().lower()
    if k == "auto":
        return music_auto(accent)
    return k if k in MUSIC else "off"

# Only for tracks whose VOICE is the instrument. LTX 2.3 generates the audio and
# has to match a mouth to it, so whether that is even possible changes how the
# shot should be framed — the same lip-sync reasoning as POV HANDS and WHO
# SPEAKS, applied to singing.
_VOCAL = {
    "opera": (
        "A sustained open vowel is the easiest thing a mouth can hold, so this "
        "voice CAN be matched: if the singer is on screen, the jaw opens and "
        "stays open through the note, throat and chest working, and the shot can "
        "sit on her face while she holds it."
    ),
    "opera_harmonics": (
        "These runs move faster than any jaw, so do NOT hold on a mouth trying "
        "to keep up — a mouth chattering to match them is the artifact. The "
        "voice carries the shot while the frame stays on the body, the hands, or "
        "the room; a mouth in frame during a run is open and still, not moving "
        "note to note. Save the mouth for the long held notes, where it can "
        "actually land."
    ),
}


# Performance mode (talk 70%+) was written for rap and pop: "runs of 2-4 lines",
# "short varied ad-libs between bars". That is the wrong shape for a sung line,
# and for the harmonic track it actively fights the lip-sync clause — "most
# beats must carry quoted voice" pushes a mouth onto runs no mouth can make.
_PERF = {
    "opera": (
        " This performance is SUNG, not spoken in bars: quote the sung line but "
        "keep it short and HOLD it — a few open vowels carried across several "
        "beats, not a stream of new words. No ad-libs, no catchphrase, no rap "
        "cadence. Breath is visible: chest and shoulders lift before a long note."
    ),
    "opera_harmonics": (
        " This performance is a voice past human range, so quote sparingly — one "
        "held word or one sung phrase per run — and leave the impossible "
        "passages as VOICE ONLY, no quoted words attached and the frame off the "
        "mouth. No ad-libs, no rap cadence. Her body sustains while the voice "
        "moves."
    ),
}


def performance_note(key: str, accent: str = "off") -> str:
    """Extra clause for PERFORMED VOCAL shots whose track is itself a voice."""
    return _PERF.get(resolve(key, accent), "")


def music_block(key: str, background: bool = False, accent: str = "off") -> str:
    """MUSIC law when a genre is chosen; explicit NO-MUSIC law when off.

    accent is only read when key == "auto".
    """
    key = resolve(key, accent)
    desc = MUSIC.get(key, "")
    if not desc:
        return (
            "\nMUSIC\n"
            "There is no music in this shot. Ambient and event sound only — "
            "what the place and bodies already make."
        )
    if background:
        return (
            "\nMUSIC\n"
            f"Playing low under the scene: {desc}. If the request describes the "
            "music itself, that description is the score and this line yields "
            "to it. Name once as ambient in the "
            "opening, quieter than voices/action. Do not re-describe later. "
            "By sound, never song title."
        )
    voc = _VOCAL.get((key or "").strip().lower(), "")
    return (
        "\nMUSIC\n"
        f"Scored by {desc}. Name as diegetic sound in the opening; body moves "
        "to it (steps, hips, gestures on the beat). By sound, never song title. "
        "If the request itself describes the music, THAT is the score and this "
        "line yields to it — use this line only for what the request leaves open."
        + (" " + voc if voc else "")
    )
