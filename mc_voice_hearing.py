"""What a recording actually contained, and when Whisper is describing silence.

This module exists because of one report, and the report is worth writing down
exactly: dictation from an Android phone was good on the handset's own
microphone and bad the moment a Bluetooth headset was connected — and "bad" did
not mean misheard words. It meant ``(music)``, ``(static)``, ``[BLANK_AUDIO]``:
Whisper's own non-speech annotations, produced from audio with no speech in it.

Why a Bluetooth microphone is a different microphone
----------------------------------------------------
None of this is Voice Chat's to fix, and all of it is Voice Chat's to survive.

A Bluetooth headset carries audio *out* over A2DP, which is a one-way profile
with no microphone in it at all. Capturing from the headset therefore needs the
hands-free profile instead, and the phone has to open an SCO link to get it —
the same link a phone call uses. Outside a call, Android decides whether to open
that link at all, and what it opens is narrowband: 8 kHz for plain HFP, 16 kHz
where both ends negotiate mSBC. The result reaching the browser is band-limited
to about 4 kHz, already compressed by a codec designed for intelligibility on a
telephone, and usually much quieter than the handset's own microphone — which is
tuned, gain-staged and beam-formed by the phone's own audio stack.

Whisper was trained on 16 kHz speech. Handed a quiet 8 kHz stream that has been
upsampled to look like one, it does what it was trained to do with audio that
has no speech in it: it emits the annotation tokens its training transcripts
used for non-speech passages. ``(music)`` is not a bug in the recording path. It
is the model reporting, accurately, that it could not hear anybody talking.

So there are two jobs here and they are different
-------------------------------------------------
:func:`measure` looks at the samples *before* inference and answers "was there
anything in this at all". A recording whose peak never leaves the noise floor is
refused with a sentence naming the likely cause, which costs nothing and is the
only response that is actually useful — running a large model over silence to be
told it was silence is a slower way to the same place.

:func:`speech` looks at the transcript *after* inference and answers "is this a
description of the recording rather than a transcription of it". It is
deliberately narrow: a result is discarded only when the **whole** of it is
annotation. ``(laughs) I said no`` keeps its words, and so does anybody who
genuinely dictated the word "music".

Nothing here tries to be a voice-activity detector. It is two conservative
gates, and the honest failure mode of both is to let something through.
"""

from __future__ import annotations

import array
import math
import re
import struct
import sys

SILENCE_PEAK = 0.012
"""Below this peak, nothing worth transcribing happened.

About −38 dBFS. Chosen to sit under the quietest capture that has ever produced
a real transcript here and above the idle noise floor of a narrowband Bluetooth
link, which is the gap this has to fit in. A speaker who is genuinely audible on
a bad microphone peaks an order of magnitude above it; an open microphone in a
quiet room does not reach it.
"""

SILENCE_RMS = 0.0025
"""And a floor on the average, so one pop in an otherwise dead recording is not
mistaken for somebody talking."""

NON_SPEECH = re.compile(
    r"""^\s*(?:
          [\(\[\{<*]\s*[^)\]\}>*]{0,60}\s*[\)\]\}>*]   # (music) [BLANK_AUDIO] *sighs*
        | [♪♫♬♩―—\-\.\,\!\?…\s]+  # a line of musical
                                                                     # notes or punctuation
        )\s*$""",
    re.VERBOSE | re.UNICODE)
"""One bracketed annotation, or nothing but marks, and the whole result.

Anchored at both ends on purpose. This is the difference between discarding
Whisper's description of a silent recording and censoring a stage direction
somebody dictated in the middle of a sentence.
"""

FILLERS = (
    "thank you for watching",
    "thanks for watching",
    "thank you for watching!",
    "thanks for watching!",
    "subtitles by the amara.org community",
    "subscribe to my channel",
    "please subscribe",
    "you",
    "bye",
    "bye.",
    ".",
    "...",
)
"""Whisper's other way of saying nothing was there.

These are artefacts of what the model was trained on -- the closing words of a
great many transcribed videos -- and they appear from silence, in every size of
Whisper, in place of an empty result. They are matched only as the entire
transcript and only case-insensitively, so somebody who says "thanks for
watching" to a character still gets it: what they will not get is that sentence
out of two seconds of Bluetooth hiss.

``you`` is on the list and is the uncomfortable one: it is a real word somebody
may genuinely say alone. It is also, by a wide margin, the single most common
thing Whisper returns for a silent clip. A one-word "you" is discarded and the
microphone is still there to press again, which is the better of the two
mistakes -- and it is only ever reached for a recording that has already passed
the level check.
"""


def measure(data: bytes) -> dict:
    """Peak, RMS and duration of a PCM16 mono WAV, without decoding it twice.

    Reads the ``data`` chunk out of the container this feature's own browser
    half produces -- :func:`mc_voice_api.validate_wav` has already established
    that it is one -- and answers in floats normalised to full scale, so the
    thresholds above are readable numbers rather than sample counts.

    Returns ``{}`` for anything it cannot read. A measurement that failed is not
    a reason to refuse a recording: the transcript gate below still runs, and
    the model is a better judge of a strange container than this is.
    """
    body, rate = _pcm(data)
    if not body or not rate:
        return {}
    samples = array.array("h")
    try:
        samples.frombytes(body[:len(body) - (len(body) % 2)])
    except ValueError:
        return {}
    if sys.byteorder == "big":
        samples.byteswap()
    if not len(samples):
        return {}

    peak = 0
    total = 0.0
    for value in samples:
        magnitude = -value if value < 0 else value
        if magnitude > peak:
            peak = magnitude
        total += float(value) * float(value)
    count = len(samples)
    return {
        "peak": peak / 32768.0,
        "rms": math.sqrt(total / count) / 32768.0,
        "seconds": count / float(rate),
        "frames": count,
        "rate": rate,
    }


def _pcm(data: bytes):
    """The bytes of the ``data`` chunk and the sample rate, or ``(b"", 0)``."""
    if not data or len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return b"", 0
    offset, rate, body = 12, 0, b""
    while offset + 8 <= len(data):
        name = data[offset:offset + 4]
        try:
            (size,) = struct.unpack_from("<I", data, offset + 4)
        except struct.error:
            break
        start = offset + 8
        if start + size > len(data):
            break
        if name == b"fmt " and size >= 16:
            rate = struct.unpack_from("<HHIIHH", data, start)[2]
        elif name == b"data":
            body = data[start:start + size]
        offset = start + size + (size % 2)
    return body, int(rate or 0)


def silent(measurement: dict) -> bool:
    """Whether a measured recording is below both floors.

    Both, not either. A recording can be quiet and still hold speech, and it can
    hold one loud click and no speech; requiring the peak *and* the average to
    be under their floors is what makes this a statement about the whole clip.
    """
    if not measurement:
        return False
    return (measurement.get("peak", 1.0) < SILENCE_PEAK
            and measurement.get("rms", 1.0) < SILENCE_RMS)


def quiet_reason(measurement: dict) -> str:
    """What to tell somebody whose recording was empty.

    Names the microphone rather than the feature, because that is where the
    problem is and because the remedy -- speak up, move the microphone, or take
    the headset out of the equation -- is theirs to apply.
    """
    peak = (measurement or {}).get("peak", 0.0)
    return (f"That recording was almost silent (peak {peak * 100:.1f}% of full scale), so "
            f"nothing was transcribed. If you are on a Bluetooth headset, its microphone is "
            f"often much quieter and lower quality than the phone's own — switching back to "
            f"the phone's microphone, or speaking closer to the headset, usually fixes it.")


def speech(text: str) -> str:
    """``text`` if it is a transcription, or ``""`` if it is a description of one.

    Whole-result matching, always. See :data:`NON_SPEECH` and :data:`FILLERS`
    for why this is narrow and why it is narrow on purpose.
    """
    found = str(text or "").strip()
    if not found:
        return ""
    if NON_SPEECH.match(found):
        return ""
    stripped = found.casefold().strip().strip("♪♫ \t")
    if stripped in FILLERS:
        return ""
    # A result that is nothing but repeated annotations -- "(music) (music)" --
    # which a longer silent clip produces and a single-token match misses.
    parts = [part for part in re.split(r"\s{2,}|\n+", found) if part.strip()]
    if len(parts) > 1 and all(NON_SPEECH.match(part) for part in parts):
        return ""
    return found


def hallucinated_reason() -> str:
    """What to tell somebody whose recording produced only an annotation."""
    return ("No speech was heard in that recording — the transcriber described the sound "
            "rather than transcribing it, which is what it does with silence or noise. If "
            "you are on a Bluetooth headset, its microphone is narrowband and often very "
            "quiet; the phone's own microphone is usually much better.")
