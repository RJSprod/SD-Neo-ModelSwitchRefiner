"""The Voice Chat sidecar: speech in, speech out, and nothing else at all.

This file runs in the isolated CPU runtime that ``mc_voice_models`` provisions,
under an interpreter that is not Forge's and cannot see Forge's packages. It
reads framed requests from stdin and writes framed responses to stdout. It has
no HTTP client, opens no socket, listens on no port, writes no file, and never
prints a transcript or a reply.

    uint32 big-endian header length
    that many bytes of UTF-8 JSON
    uint32 big-endian payload length
    that many raw bytes

Four operations:

    init        the handshake. The parent sends the verified local model paths
                and the thread caps; this replies READY with the provider it
                actually got, which is what the parent checks before believing
                the runtime is CPU-only.
    stt         payload is a PCM16 mono RIFF/WAVE; the header of the reply
                carries the transcript.
    tts         payload is UTF-8 text; the payload of the reply is a WAV.
    shutdown    acknowledge and exit.

Why a pipe and not a port
-------------------------
A port needs discovery, a local authentication story, a firewall prompt, and
one mistake away from binding to a LAN address. A pipe needs none of that, and
it carries the lifetime contract for free: when the parent disappears its handle
closes, the next read returns EOF, and this process exits. That is the portable
first layer of invariant I-7.

It is only the first layer, and this file says so honestly. EOF is observed at
the moment this process is *waiting to read*, and during a long synthesis it is
not waiting to read -- it is inside native ONNX code. So the operating system is
asked as well: on Linux ``PR_SET_PDEATHSIG`` with SIGKILL, installed before the
first request is served and immediately re-checked against the parent pid we
were told about, which closes the window where the parent dies during our own
start-up and leaves a child whose parent-death signal will now never fire. On
Windows the parent puts this process in a job object with KILL_ON_JOB_CLOSE
before it is given any work, which is the same guarantee from the other side.
:func:`_containment` reports which of those is in force, the parent refuses to
mark the runtime running without one, and ``tests/test_voice_shutdown.py`` kills
a parent mid-inference and asserts nothing survives.

Content and logs
----------------
Invariant I-6. Nothing in this file writes transcript text, reply text, or audio
to stderr, and the diagnostics it does write are model ids, byte counts,
durations and error classes. :func:`_safe` exists because an exception message
from a third-party library is not a string this file gets to assume is
content-free -- so the words that come back to the parent are chosen here.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time
import wave

PROTOCOL_VERSION = 1

MARKER = "--model-chain-voice-worker"
"""On the command line so this process is recognisable in a task manager and by
the stray sweep. Never a model name and never chat text: a command line is world
readable on most systems."""

MAX_HEADER = 1 << 20
MAX_PAYLOAD = 8 << 20
"""A 60-second 16 kHz mono PCM16 WAV is about 1.9 MB. Eight is generous and is
still a number rather than "whatever arrives"."""

SAMPLE_RATE = 16000
MAX_SECONDS = 60.0

_LENGTH = struct.Struct(">I")


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


def read_frame(stream) -> tuple[dict, bytes] | None:
    """One request, or ``None`` at end of input.

    ``None`` is how the parent's death arrives when this process is waiting for
    work, and it is not an error: the loop ends, the models are released, and
    the process exits with 0.
    """
    header_length = _read_exactly(stream, 4)
    if header_length is None:
        return None
    (size,) = _LENGTH.unpack(header_length)
    if size > MAX_HEADER:
        raise ValueError("header too large")
    raw = _read_exactly(stream, size)
    if raw is None:
        return None
    header = json.loads(raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not an object")

    payload_length = _read_exactly(stream, 4)
    if payload_length is None:
        return None
    (size,) = _LENGTH.unpack(payload_length)
    if size > MAX_PAYLOAD:
        raise ValueError("payload too large")
    payload = b"" if size == 0 else _read_exactly(stream, size)
    if payload is None:
        return None
    return header, payload


def write_frame(stream, header: dict, payload: bytes = b"") -> None:
    raw = json.dumps(header).encode("utf-8")
    stream.write(_LENGTH.pack(len(raw)))
    stream.write(raw)
    stream.write(_LENGTH.pack(len(payload)))
    if payload:
        stream.write(payload)
    stream.flush()


def _read_exactly(stream, count: int) -> bytes | None:
    chunks = []
    remaining = count
    while remaining > 0:
        block = stream.read(remaining)
        if not block:
            return None
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


# --------------------------------------------------------------------------- #
# Audio, in stdlib only
# --------------------------------------------------------------------------- #


def decode_wav(data: bytes) -> tuple[list, int]:
    """A validated PCM16 mono WAV as float samples in [-1, 1].

    Validated rather than trusted even though the route in front already
    checked: this is the process that would crash, and a worker that dies on a
    malformed upload is a worker the next dictation has to restart.
    """
    with contextlib_closing(wave.open(io.BytesIO(data), "rb")) as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        if channels != 1:
            raise ValueError("audio is not mono")
        if width != 2:
            raise ValueError("audio is not 16-bit")
        if frames <= 0:
            raise ValueError("audio is empty")
        if frames / float(rate or 1) > MAX_SECONDS + 1.0:
            raise ValueError("audio is too long")
        raw = handle.readframes(frames)

    import array

    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    scale = 1.0 / 32768.0
    return [value * scale for value in samples], rate


def encode_wav(samples, rate: int) -> bytes:
    """Float samples as a PCM16 mono WAV, in memory.

    In memory is the requirement, not a preference: invariant I-5 says generated
    speech never becomes a file, so there is no path argument here and no
    tempfile import in this module.
    """
    import array

    clipped = array.array("h")
    for value in samples:
        scaled = int(float(value) * 32767.0)
        if scaled > 32767:
            scaled = 32767
        elif scaled < -32768:
            scaled = -32768
        clipped.append(scaled)
    if sys.byteorder == "big":
        clipped.byteswap()

    buffer = io.BytesIO()
    with contextlib_closing(wave.open(buffer, "wb")) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(clipped.tobytes())
    return buffer.getvalue()


def contextlib_closing(thing):
    import contextlib

    return contextlib.closing(thing)


# --------------------------------------------------------------------------- #
# Dying with the parent
# --------------------------------------------------------------------------- #


def _containment(parent_pid: int) -> str:
    """Ask the OS to end this process when the parent ends. Reports what it got.

    On Linux ``PR_SET_PDEATHSIG`` is set to SIGKILL and then the parent pid is
    re-read. The re-read is the whole point: if the parent died between its fork
    and this line, the death signal it would have sent has already not been
    sent, and this process would sit here forever holding a gigabyte of Whisper.
    Seeing a different parent than the one we were told about is that race, and
    the answer is to exit immediately.

    On Windows the job object is the parent's to create and it has already done
    it by the time the handshake is answered; the honest word for what this side
    contributed is "job". Everywhere else there is only the pipe, the parent is
    told so, and the parent decides whether that platform is one it is willing
    to run on.
    """
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1
            if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        except Exception as exc:  # noqa: BLE001 - reported, never raised onward
            _note(f"parent-death containment unavailable: {exc.__class__.__name__}")
            return "pipe"
        if parent_pid and os.getppid() != parent_pid:
            _note("parent went away during start-up")
            raise SystemExit(0)
        return "pdeathsig"
    if os.name == "nt":
        return "job"
    return "pipe"


def _lower_priority() -> None:
    """Be the process that yields when an image is rendering. Never fatal."""
    try:
        if os.name == "nt":
            import ctypes

            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(5)
    except Exception:
        _note("could not lower the voice worker's priority")


# --------------------------------------------------------------------------- #
# The engines
# --------------------------------------------------------------------------- #


class Engines:
    """The recognizer and the synthesiser, loaded once and kept warm.

    Warm because the alternative is reading four hundred megabytes of ONNX off
    a disk between "let go of the microphone" and "see the words", every time.
    The design intent's own arithmetic is that the target machine has 96 GB and
    this is the correct trade.
    """

    def __init__(self, config: dict):
        self.config = config
        self.provider = "cpu"
        self.recognizer = None
        self.tts = None
        self.stt_model_id = str((config.get("stt") or {}).get("id") or "")
        self.tts_model_id = str((config.get("tts") or {}).get("id") or "")
        self.stt_threads = int(config.get("stt_threads") or 4)
        self.tts_threads = int(config.get("tts_threads") or 2)
        self.speaker_id = int((config.get("tts") or {}).get("speaker_id") or 0)

    def load(self) -> None:
        import sherpa_onnx

        stt = self.config.get("stt") or {}
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=_required(stt, "encoder"),
            decoder=_required(stt, "decoder"),
            tokens=_required(stt, "tokens"),
            num_threads=self.stt_threads,
            provider="cpu",
            decoding_method="greedy_search",
            language=str(stt.get("language") or ""),
            task="transcribe",
        )

        tts = self.config.get("tts") or {}
        root = tts.get("root") or ""
        kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=_required(tts, "model"),
            voices=_required(tts, "voices"),
            tokens=_required(tts, "tokens"),
            data_dir=_optional(root, "espeak-ng-data"),
            dict_dir=_optional(root, "dict"),
            lexicon=_lexicons(root),
        )
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=kokoro, provider="cpu", num_threads=self.tts_threads)
        self.tts = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1))

    def transcribe(self, samples, rate: int) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        self.recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()

    def synthesize(self, text: str):
        audio = self.tts.generate(text, sid=self.speaker_id, speed=1.0)
        return audio.samples, int(audio.sample_rate)


def _required(section: dict, key: str) -> str:
    value = str(section.get(key) or "")
    if not value or not os.path.isfile(value):
        raise ValueError(f"the installed bundle has no {key}")
    return value


def _optional(root: str, name: str) -> str:
    candidate = os.path.join(root, name) if root else ""
    return candidate if candidate and os.path.isdir(candidate) else ""


def _lexicons(root: str) -> str:
    """Kokoro's lexicons, comma-joined, in the order sherpa-onnx expects.

    Discovered from the installed bundle rather than listed in the manifest,
    because which lexicons a Kokoro release ships is a property of that release
    and the manifest already pins the archive it came out of.
    """
    if not root or not os.path.isdir(root):
        return ""
    found = sorted(name for name in os.listdir(root)
                   if name.startswith("lexicon") and name.endswith(".txt"))
    return ",".join(os.path.join(root, name) for name in found)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def _note(text: str) -> None:
    """One diagnostic line on stderr. Never content -- see the module docstring."""
    try:
        sys.stderr.write(f"voice-worker: {text}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _safe(exc: BaseException) -> str:
    """What the parent is told a failure was.

    Deliberately the exception's *class* and nothing else for anything this file
    did not raise itself. A third-party library is entitled to put whatever it
    likes in a message, including the input it was given, and this feature's
    invariant is that the input never leaves this process.
    """
    if isinstance(exc, ValueError):
        return str(exc)
    return exc.__class__.__name__


def serve(stdin, stdout, engines_factory=None) -> int:
    """Read frames until end of input. Returns the process exit status."""
    engines = None
    factory = engines_factory or (lambda config: Engines(config))
    parent_death = "pipe"

    while True:
        try:
            frame = read_frame(stdin)
        except Exception as exc:
            _note(f"malformed frame: {_safe(exc)}")
            return 2
        if frame is None:
            _note("input closed; stopping")
            return 0
        header, payload = frame
        operation = str(header.get("op") or "")
        request_id = header.get("id")

        if operation == "shutdown":
            _reply(stdout, request_id, {"ok": True})
            _note("stopping on request")
            return 0

        if operation == "init":
            try:
                parent_death = _containment(int(header.get("parent_pid") or 0))
                _lower_priority()
                engines = factory(header.get("config") or {})
                started = time.monotonic()
                engines.load()
                _reply(stdout, request_id, {
                    "ok": True,
                    "op": "ready",
                    "protocol_version": PROTOCOL_VERSION,
                    "runtime_version": _runtime_version(),
                    "provider": engines.provider,
                    "parent_death": parent_death,
                    "stt_model_id": engines.stt_model_id,
                    "tts_model_id": engines.tts_model_id,
                    "stt_threads": engines.stt_threads,
                    "tts_threads": engines.tts_threads,
                    "load_seconds": round(time.monotonic() - started, 3),
                })
                _note(f"ready — CPU provider, STT threads {engines.stt_threads}, "
                      f"TTS threads {engines.tts_threads}, containment {parent_death}")
            except SystemExit:
                raise
            except Exception as exc:
                engines = None
                _reply(stdout, request_id, {"ok": False, "error": _safe(exc)})
                _note(f"could not load the speech models: {_safe(exc)}")
            continue

        if engines is None:
            _reply(stdout, request_id, {"ok": False, "error": "runtime is not initialised"})
            continue

        try:
            if operation == "ping":
                _reply(stdout, request_id, {"ok": True})
            elif operation == "stt":
                started = time.monotonic()
                samples, rate = decode_wav(payload)
                text = engines.transcribe(samples, rate)
                elapsed = time.monotonic() - started
                seconds = len(samples) / float(rate or 1)
                _reply(stdout, request_id, {"ok": True, "text": text,
                                            "audio_seconds": round(seconds, 2),
                                            "elapsed": round(elapsed, 3)})
                _note(f"stt finished — {seconds:.1f} s audio in {elapsed:.1f} s")
            elif operation == "tts":
                text = payload.decode("utf-8", "replace")
                started = time.monotonic()
                samples, rate = engines.synthesize(text)
                audio = encode_wav(samples, rate)
                elapsed = time.monotonic() - started
                seconds = len(audio) / float(max(rate, 1) * 2)
                _reply(stdout, request_id, {"ok": True, "sample_rate": rate,
                                            "audio_seconds": round(seconds, 2),
                                            "elapsed": round(elapsed, 3)}, audio)
                _note(f"tts finished — {len(text)} characters, {seconds:.1f} s audio "
                      f"in {elapsed:.1f} s")
            else:
                _reply(stdout, request_id, {"ok": False, "error": "unknown operation"})
        except Exception as exc:
            _reply(stdout, request_id, {"ok": False, "error": _safe(exc)})
            _note(f"request failed: {_safe(exc)}")


def _reply(stdout, request_id, header: dict, payload: bytes = b"") -> None:
    header = dict(header)
    header["id"] = request_id
    write_frame(stdout, header, payload)


def _runtime_version() -> str:
    try:
        import sherpa_onnx

        return str(getattr(sherpa_onnx, "__version__", "") or "")
    except Exception:
        return ""


def selftest() -> int:
    """Prove the staged runtime imports and can be asked for CPU. One JSON line.

    Run by the installer against the *staging* environment, before anything is
    promoted. It deliberately builds a real config object rather than only
    importing: an ONNX Runtime whose CPU provider is missing imports perfectly
    well and fails at the first model.
    """
    report = {"ok": False, "provider": "", "runtime_version": ""}
    try:
        import sherpa_onnx

        report["runtime_version"] = str(getattr(sherpa_onnx, "__version__", "") or "")
        config = sherpa_onnx.OfflineTtsModelConfig(provider="cpu", num_threads=1)
        report["provider"] = str(getattr(config, "provider", "") or "cpu")
        report["ok"] = report["provider"] == "cpu"
    except Exception as exc:
        report["error"] = exc.__class__.__name__
    sys.stdout.write(json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0 if report["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(MARKER, dest="marker", action="store_true")
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--session", default="")
    parser.add_argument("--selftest", action="store_true")
    known, _unknown = parser.parse_known_args(argv)

    if known.selftest:
        return selftest()

    # Binary on both sides: the protocol is length-prefixed bytes, and a text
    # wrapper would translate newlines on Windows and corrupt every WAV.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    return serve(stdin, stdout)


if __name__ == "__main__":
    raise SystemExit(main())
