"""Silence, and Whisper describing it.

The report these tests exist for: dictation from an Android phone was good on
the handset's own microphone and produced ``(music)`` and ``(static)`` through a
Bluetooth headset. That is not a transcription bug -- it is the model reporting,
accurately, that it could not hear anybody talking, using the annotation tokens
its training transcripts used for non-speech passages.

Two gates, and they are different jobs. One looks at the samples before
inference and refuses a recording with nothing in it. One looks at the
transcript after and discards a result that is entirely annotation. Both are
conservative, and the honest failure mode of both is to let something through.
"""

from __future__ import annotations

import io
import math
import struct
import wave

import pytest

import mc_voice_hearing as hearing


def wav(level: float = 0.3, seconds: float = 1.0, rate: int = 16000,
        tone: bool = True) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = int(rate * seconds)
        handle.writeframes(b"".join(
            struct.pack("<h", int(32767 * level
                                  * (math.sin(2 * math.pi * 220 * i / rate) if tone else 1)))
            for i in range(frames)))
    return buffer.getvalue()


class TestMeasuring:
    def test_it_reads_peak_rms_and_length(self):
        found = hearing.measure(wav(level=0.5, seconds=0.5))
        assert found["peak"] == pytest.approx(0.5, abs=0.01)
        assert found["rms"] == pytest.approx(0.5 / math.sqrt(2), abs=0.01)
        assert found["seconds"] == pytest.approx(0.5, abs=0.01)

    def test_something_it_cannot_read_is_not_a_verdict(self):
        """A measurement that failed is not a reason to refuse a recording. The
        transcript gate still runs, and the model is a better judge of a strange
        container than this is."""
        assert hearing.measure(b"") == {}
        assert hearing.measure(b"not a wav at all") == {}
        assert hearing.silent({}) is False


class TestTheSilenceGate:
    def test_digital_silence_is_silent(self):
        assert hearing.silent(hearing.measure(wav(level=0.0))) is True

    def test_a_bluetooth_noise_floor_is_silent(self):
        assert hearing.silent(hearing.measure(wav(level=0.002))) is True

    def test_ordinary_speech_is_not(self):
        assert hearing.silent(hearing.measure(wav(level=0.2))) is False

    def test_a_quiet_but_real_capture_is_not(self):
        """The floor has to sit under the quietest capture that has ever
        produced a real transcript and above a narrowband idle noise floor.
        This is the gap it has to fit in."""
        assert hearing.silent(hearing.measure(wav(level=0.03))) is False

    def test_a_recording_with_one_loud_moment_in_it_is_never_refused(self):
        """Both floors have to be under, not either. A mostly-quiet recording
        with one clear word in it is a recording worth transcribing, and it is
        the annotation gate rather than this one that catches the case where
        that moment turns out to have been a door closing."""
        rate = 16000
        body = b"".join(struct.pack("<h", 20000 if i == 100 else 0) for i in range(rate))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(body)
        found = hearing.measure(buffer.getvalue())
        assert found["peak"] > 0.5
        assert hearing.silent(found) is False

    def test_both_floors_have_to_be_under_for_a_refusal(self):
        assert hearing.silent({"peak": 0.9, "rms": 0.0}) is False
        assert hearing.silent({"peak": 0.0, "rms": 0.9}) is False
        assert hearing.silent({"peak": 0.0, "rms": 0.0}) is True

    def test_the_reason_names_the_microphone_rather_than_the_feature(self):
        found = hearing.quiet_reason(hearing.measure(wav(level=0.0)))
        assert "Bluetooth" in found
        assert "microphone" in found


class TestTheAnnotationGate:
    @pytest.mark.parametrize("text", [
        "(music)", "[music]", "(static)", "[BLANK_AUDIO]", "*sighs*", "<inaudible>",
        "♪♪♪", "...", "(wind blowing)", "( Music )", "(music)  (static)",
        "Thanks for watching!", "thank you for watching", "you", "You", "Bye.",
    ])
    def test_a_description_of_the_recording_is_not_a_transcription(self, text):
        assert hearing.speech(text) == ""

    @pytest.mark.parametrize("text", [
        "(laughs) I said no",
        "music is good",
        "Play some music.",
        "Thanks for watching the film with me.",
        "You should try the other one.",
        "Bye for now, I will be back later.",
        "The static was bad on that channel.",
    ])
    def test_words_somebody_said_survive(self, text):
        """Narrow on purpose. A result is discarded only when the *whole* of it
        is annotation -- anything else censors somebody's dictation."""
        assert hearing.speech(text) == text

    def test_the_reason_says_what_the_transcriber_did(self):
        found = hearing.hallucinated_reason()
        assert "described the sound" in found
        assert "Bluetooth" in found
