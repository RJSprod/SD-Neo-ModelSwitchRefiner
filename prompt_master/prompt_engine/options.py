"""UI option lists, derived from the upstream engine constants.

Section 5 of the port brief: the UI must not carry its own copies of these
lists. Every sequence below is built from ``prompt_engine.upstream`` at import
time, so a control cannot drift away from the engine that consumes its value.

Each entry is ``(value, label)``. The value is the upstream key and is what
travels in a ``PromptRequest``; the label is display text only. They stay
separate because upstream keeps them separate — ``accent_label("rp_british")``
is "RP British", and the engine would not recognise that string as a key.
"""

from __future__ import annotations

from .upstream.accents import ACCENT_KEYS, STRENGTHS, accent_label
from .upstream.cinematics import (CAMERA_KEYS, CAMERA_LABELS, TRANSITION_KEYS,
                                  TRANSITION_LABELS)
from .upstream.music import MUSIC_KEYS, MUSIC_LABELS
from .upstream.shotscript import FORMAT_LABELS, FORMATS
from .upstream.styles import STYLE_GROUPS

Option = tuple[str, str]

# ── from the engine ──────────────────────────────────────────────────────────

ACCENTS: list[Option] = [(k, accent_label(k)) for k in ACCENT_KEYS]

ACCENT_STRENGTHS: list[Option] = [(s, s.title()) for s in STRENGTHS]

# Upstream's node offers "auto" ahead of the genre list; auto resolves through
# music.music_auto(accent), so it only means anything once an accent is set.
MUSIC: list[Option] = ([("auto", "Auto (match the accent)")]
                       + [(k, MUSIC_LABELS[k]) for k in MUSIC_KEYS])

# Grouped exactly as upstream groups them for its own panel, so the optgroup
# headings and their order are the engine's, not ours.
STYLES_GROUPED: list[tuple[str, list[Option]]] = [
    (group, list(opts)) for group, opts in STYLE_GROUPS
]
STYLES: list[Option] = [opt for _, opts in STYLES_GROUPED for opt in opts]

CAMERAS: list[Option] = [(k, CAMERA_LABELS[k]) for k in CAMERA_KEYS]

TRANSITIONS: list[Option] = [(k, TRANSITION_LABELS[k]) for k in TRANSITION_KEYS]

OUTPUT_FORMATS: list[Option] = [(f, FORMAT_LABELS[f]) for f in FORMATS]

# ── from the upstream node's own widget definitions ──────────────────────────
# These four are declared inline in upstream/node.py INPUT_TYPES rather than in
# a constant, so they are transcribed here with their upstream values and
# upstream default ordering. check_upstream_sync.py holds node.py byte-identical,
# which is what keeps this honest.

VIDEO_MODES: list[Option] = [("i2v", "Image to video"), ("t2v", "Text to video")]

POV: list[Option] = [("off", "POV — off"), ("male", "First person — male"),
                     ("female", "First person — female")]

WARDROBE: list[Option] = [("auto", "Auto (from the intent)"), ("off", "Wardrobe — off"),
                          ("her", "Her"), ("him", "Him")]

FIT_MODES: list[Option] = [("crop", "Crop to fill"), ("pad", "Pad (letterbox)"),
                           ("stretch", "Stretch")]

# Upstream node defaults, so the UI opens on the same shot the node would build.
DEFAULTS = {
    "video_mode": "i2v",
    "pov": "off",
    "accent": "off",
    "accent_strength": "natural",
    "dialogue": 20,
    "wardrobe": "auto",
    "undress": False,
    "camera": "off",
    "transition": "off",
    "music": "off",
    "music_bg": False,
    "fmt": "flowing",
    "fps": 24,
    "seconds": 12.0,
    "style": "off",
    "seed": 7,
    "smart_negative": False,
    "output_width": 704,
    "output_height": 1216,
    "fit": "crop",
}


def values(options: list[Option]) -> list[str]:
    return [value for value, _ in options]


def labels(options: list[Option]) -> list[str]:
    return [label for _, label in options]


def label_for(options: list[Option], value: str) -> str:
    for candidate, label in options:
        if candidate == value:
            return label
    return value


def value_for(options: list[Option], label: str) -> str:
    for value, candidate in options:
        if candidate == label:
            return value
    return label
