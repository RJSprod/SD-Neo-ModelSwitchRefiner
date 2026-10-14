"""What the Voice Box hands an engine's runtime, and what it is handed back.

Engine-neutral on purpose. The render service (:mod:`mc_voice_box`) turns a
script and a configuration into one :class:`RenderJob` per section and joins the
:class:`RenderResult` it gets back; what an engine does with a section in between
is its runtime's business. VibeVoice's runtime sends the whole section to one
``generate``; Breeze's renders it line by line in each line's voice and joins
the lines. The Voice Box never knows which.

Standard library only, and nothing from the memory side (invariant I-3): every
runtime imports this module, and so does the service.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Voice:
    """One speaker slot, resolved: the sample's sound and what the slot says about it.

    ``pcm`` is little-endian 32-bit float mono at ``rate``, or ``b""`` for a
    voice that is only a description. ``transcript`` is the sample's exact
    words, which an engine that clones from a transcript needs and VibeVoice
    ignores; ``description`` designs a voice instead of cloning one;
    ``instruction`` is the slot's standing direction, which a line's own
    direction overrides.
    """

    pcm: bytes = b""
    rate: int = 24000
    sample_id: str = ""
    title: str = ""
    transcript: str = ""
    description: str = ""
    instruction: str = ""


@dataclass
class Line:
    """One line of a section: who says it, the words, and how (``Speaker 2 (direction):``)."""

    speaker: int
    text: str
    instruction: str = ""


@dataclass
class RenderJob:
    """One section of a script, as a runtime renders it.

    ``lines`` are :class:`Line` (a ``(speaker, text)`` pair is read as one);
    ``voices`` maps each speaker the section uses to its :class:`Voice` (a
    ``(pcm, rate)`` pair is read as one). ``seed`` is take one's, take k is
    rendered at ``seed + k``, and ``options`` are the engine's own
    configuration fields as its adapter cleaned them.
    """

    id: str
    lines: list = field(default_factory=list)
    voices: dict = field(default_factory=dict)
    seed: "int | None" = None
    takes: int = 1
    options: dict = field(default_factory=dict)

    def line_list(self) -> list:
        """``lines`` as :class:`Line`, whatever shape they were handed in."""
        found = []
        for entry in self.lines or ():
            if isinstance(entry, Line):
                found.append(entry)
            elif isinstance(entry, dict):
                found.append(Line(int(entry.get("speaker") or 0), str(entry.get("text") or ""),
                                  str(entry.get("instruction") or "")))
            else:
                speaker, text = entry[0], entry[1]
                direction = entry[2] if len(entry) > 2 else ""
                found.append(Line(int(speaker), str(text or ""), str(direction or "")))
        return found

    def voice_map(self) -> dict:
        """``voices`` as ``{speaker number: Voice}``, whatever shape they were handed in."""
        found = {}
        for speaker, voice in (self.voices or {}).items():
            if isinstance(voice, Voice):
                found[int(speaker)] = voice
            elif isinstance(voice, dict):
                found[int(speaker)] = Voice(**{key: value for key, value in voice.items()
                                               if key in Voice.__dataclass_fields__})
            else:
                pcm, rate = voice
                found[int(speaker)] = Voice(pcm=bytes(pcm or b""), rate=int(rate or 0))
        return found


@dataclass
class Take:
    """One take's audio: ``pcm`` is its little-endian 32-bit float mono samples."""

    pcm: bytes
    seconds: float
    tokens: int
    capped: bool
    seed: "int | None" = None


@dataclass
class RenderResult:
    """What one render produced: its takes, in order, and what the render cost."""

    takes: list
    sample_rate: int
    render_seconds: float
    peak_bytes: int
    cancelled: bool

    @property
    def seconds(self) -> float:
        return self.takes[0].seconds if self.takes else 0.0

    @property
    def tokens(self) -> int:
        return sum(take.tokens for take in self.takes)

    @property
    def capped(self) -> bool:
        return any(take.capped for take in self.takes)
