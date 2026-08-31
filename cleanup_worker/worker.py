"""DeepFilterNet in a process of its own, speaking one operation.

Launched by ``mc_voice_cleanup_runtime`` on the cp311 interpreter that
``mc_voice_cleanup`` installed, and never by Forge's. It is the only file in
this repository that imports DeepFilterNet, and the import happens after the
handshake so that a parent which refuses the handshake never pays for it.

The protocol is byte-identical to the Sopro and Kokoro workers -- a big-endian
length, a JSON header, a big-endian length, a payload -- by agreement rather
than by import, because the three run on three different interpreters and
sharing a module between them would be sharing a Python version too.

One operation
-------------
``clean`` takes mono PCM16 at any rate this build resamples, and gives back mono
PCM16 at the same rate, denoised. There is no streaming, no model choice and no
state that outlives a request beyond the loaded model itself: this runs while
somebody is tidying a recording and stops when they are not.

What it never does
------------------
It has no HTTP client and no hub client, it is told a local model directory
rather than a name, and its environment has no visible graphics device. It
writes nothing except its replies and its stderr notes, and it never writes what
anybody said -- numbers, enums and durations only.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

PROTOCOL_VERSION = 1
MARKER = "--model-chain-cleanup-worker"

MAX_HEADER = 1 << 20
MAX_PAYLOAD = 64 * 1024 * 1024
"""Sixty-four megabytes is minutes of PCM16 and far more than the route in front
of this will pass. It is a number rather than "whatever arrives"."""

MODEL_RATE = 48000
"""What DeepFilterNet works at. Anything else is resampled in and back out
again, here, so the caller never has to know."""

IDLE_SECONDS = 120.0
"""How long the model stays loaded with nothing to do before this process ends
itself. The engine is meant to be running only while it is being used, and a
worker holding a Torch runtime for an afternoon because somebody cleaned one
clip is the thing that promise is about."""

ATTENUATION_LIMIT_DB = 20.0
"""How much noise this cleaner is allowed to take out, in decibels.

DeepFilterNet's own default is *no* limit: the mask it predicts is applied in
full, and everything it does not model as speech goes to nothing. That is the
right default for a recording somebody is going to listen to and the wrong one
for a cloning reference, which is not a recording at all -- the model conditions
on it, so whatever comes out of the timbre comes back under every sentence that
voice ever says. A voice cleaned to nothing but its speech clones as a voice
with nothing but its speech in it.

Twenty decibels is not a taste. It is the same ceiling the in-page cleaner has
had since it was written: its ``NOISE_FLOOR_GAIN`` of 0.10 means a bin it
empties still keeps a tenth of what was there, which is -20 dB. Matching it
leaves the two cleaners taking out the same *amount* and differing only in how
well they choose what to take -- which is the comparison the two Play buttons in
the panel are for, and the one worth having.

Upstream applies it as a mix rather than a different mask
(``enhanced = original * lim + enhanced * (1 - lim)``, ``lim = 10 ** -(dB/20)``),
so a tenth of the original spectrum survives untouched. That is the part a
learned denoiser cannot be talked out of removing, and it is the part a cloning
model needs.
"""

INTRAOP_THREADS = 4
INTEROP_THREADS = 1

_LENGTH = struct.Struct(">I")


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


def _read_exactly(stream, count: int):
    found = b""
    while len(found) < count:
        chunk = stream.read(count - len(found))
        if not chunk:
            return None
        found += chunk
    return found


def read_frame(stream):
    """One request, or ``None`` at end of input.

    ``None`` is how the parent's death arrives while this process is waiting
    for work, and it is not an error: the loop ends and the process exits 0.
    """
    raw_length = _read_exactly(stream, 4)
    if raw_length is None:
        return None
    (size,) = _LENGTH.unpack(raw_length)
    if size > MAX_HEADER:
        raise ValueError("header too large")
    raw = _read_exactly(stream, size)
    if raw is None:
        return None
    header = json.loads(raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not an object")

    raw_length = _read_exactly(stream, 4)
    if raw_length is None:
        return None
    (size,) = _LENGTH.unpack(raw_length)
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


def _note(text: str) -> None:
    """One line to stderr, which the parent drains into the log at info.

    Never anything anybody said: numbers, enums and durations only.
    """
    try:
        sys.stderr.write(f"[cleanup] {text}\n")
        sys.stderr.flush()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Containment
# --------------------------------------------------------------------------- #


def containment(parent_pid: int) -> str:
    """Ask the OS to end this process when the parent ends, and say what it got.

    On Linux ``PR_SET_PDEATHSIG`` is set here, inside the child, because that is
    the only place it can be, and the parent pid is re-read afterwards: if the
    parent died between the fork and this line the signal has already not been
    sent, and this process would sit here forever holding a Torch runtime.

    On Windows the job object is the parent's to create and the parent proves
    it, against its own job handle, before this runs. What this reports is
    corroboration and the parent treats it as such -- ``unknown`` means the
    question could not be put, which is not the same as ``none`` and must not be
    read as one.
    """
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        except Exception as exc:  # noqa: BLE001 - reported, never raised onward
            _note(f"parent-death containment unavailable: {exc.__class__.__name__}")
            return "pipe"
        if parent_pid and os.getppid() != parent_pid:
            _note("parent went away during start-up")
            raise SystemExit(0)
        return "pdeathsig"
    if os.name == "nt":
        return _in_a_job()
    return "pipe"


def _in_a_job() -> str:
    """``job``, ``none`` or ``unknown``. The third is not the second.

    The argument types are declared because a HANDLE is not a C ``int`` on
    64-bit Windows and ``GetCurrentProcess`` returns the pseudo-handle -1, which
    is the one value where getting that wrong matters most.
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                            ctypes.POINTER(wintypes.BOOL)]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        inside = wintypes.BOOL(0)
        if not kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), None,
                                       ctypes.byref(inside)):
            raise OSError(ctypes.get_last_error(), "IsProcessInJob failed")
    except Exception as exc:  # noqa: BLE001 - reported, never raised onward
        _note(f"could not ask whether this process is in a job: {exc.__class__.__name__}")
        return "unknown"
    return "job" if inside.value else "none"


# --------------------------------------------------------------------------- #
# The engine. Everything below this line imports Torch.
# --------------------------------------------------------------------------- #


def _attenuation(enhance) -> dict:
    """``atten_lim_db`` for this build's ``enhance``, or nothing if it has none.

    Asked of the signature rather than found out by calling, because a
    ``TypeError`` raised *inside* ``enhance`` and one raised by handing it a
    keyword it does not take are indistinguishable from out here and want
    opposite handling. A build without the parameter cleans the way it always
    did, which is worse for this job and much better than not cleaning at all.
    """
    import inspect

    try:
        if "atten_lim_db" in inspect.signature(enhance).parameters:
            return {"atten_lim_db": ATTENUATION_LIMIT_DB}
    except (TypeError, ValueError):
        pass
    return {}


class Engine:
    """DeepFilterNet, loaded once, from a directory this extension verified.

    ``init_df`` is given a path rather than a name on purpose: handed a name it
    would go and fetch one, and a worker that can reach the Internet is a worker
    whose behaviour depends on the Internet. The closure has no hub client and
    the environment has no visible graphics device, so neither is a promise --
    both are absences.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.state = None
        self.torch = None
        self.versions = {}

    def load(self) -> None:
        import torch

        torch.set_num_threads(INTRAOP_THREADS)
        try:
            torch.set_num_interop_threads(INTEROP_THREADS)
        except RuntimeError:
            # Already set by an earlier call in this process; not an error, and
            # not worth failing a load over.
            pass
        self.torch = torch

        from df.enhance import init_df

        if not os.path.isdir(self.model_dir):
            raise RuntimeError(f"the model directory {self.model_dir!r} is not there")
        model, state, _suffix = init_df(model_base_dir=self.model_dir,
                                        log_file=None, config_allow_defaults=True)
        model.eval()
        self.model = model
        self.state = state

        import df
        self.versions = {
            "torch": str(getattr(torch, "__version__", "")),
            "deepfilternet": str(getattr(df, "__version__", "")
                                 or getattr(getattr(df, "version", None), "version", "")),
            "python": "%d.%d.%d" % sys.version_info[:3],
            "device": str(next(model.parameters()).device),
        }

    def clean(self, pcm: bytes, rate: int) -> bytes:
        """Mono PCM16 in, mono PCM16 out, at the rate it came in at.

        Resampled to 48 kHz and back because that is what the model works at,
        and doing it here rather than asking the caller keeps one rate
        conversion in one place with one set of rounding.

        Enhanced with a limit on how much it may take out -- see
        :data:`ATTENUATION_LIMIT_DB`. Without one this call suppresses in full,
        which is what a listener wants and not what a voice clone wants.
        """
        import torchaudio
        from df.enhance import enhance

        torch = self.torch
        samples = torch.frombuffer(bytearray(pcm), dtype=torch.int16).to(torch.float32)
        samples = (samples / 32768.0).unsqueeze(0)
        if rate != MODEL_RATE:
            samples = torchaudio.functional.resample(samples, rate, MODEL_RATE)
        with torch.no_grad():
            found = enhance(self.model, self.state, samples, **_attenuation(enhance))
        if rate != MODEL_RATE:
            found = torchaudio.functional.resample(found, MODEL_RATE, rate)
        found = found.squeeze(0).clamp(-1.0, 1.0)
        return (found * 32767.0).to(torch.int16).numpy().tobytes()


def _tone(seconds: float, rate: int) -> bytes:
    """A second of noisy tone for the self-test, built without numpy."""
    import math

    frames = int(seconds * rate)
    out = bytearray()
    seed = 1
    for index in range(frames):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        noise = (seed / 0x7FFFFFFF) * 2 - 1
        value = 0.3 * math.sin(2 * math.pi * 220 * index / rate) + 0.05 * noise
        out += struct.pack("<h", max(-32768, min(32767, int(value * 32767))))
    return bytes(out)


def selftest(model_dir: str) -> int:
    """Prove this build runs, imports, is on the CPU, and denoises.

    Printed as one JSON line, which is what the installer reads. A load that
    succeeds proves the wheels landed and nothing about whether the model works,
    so a second of audio actually goes through it.
    """
    report = {"ok": False}
    try:
        started = time.monotonic()
        engine = Engine(model_dir)
        engine.load()
        report.update(engine.versions)
        if not str(engine.versions.get("device", "")).startswith("cpu"):
            raise RuntimeError(f"DeepFilterNet loaded on {engine.versions.get('device')!r} "
                               f"rather than the CPU")
        seconds = 1.0
        cleaned = engine.clean(_tone(seconds, MODEL_RATE), MODEL_RATE)
        if len(cleaned) < MODEL_RATE:  # at least half a second of PCM16 came back
            raise RuntimeError("the model returned less audio than it was given")
        report["seconds"] = seconds
        report["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the report is the point
        report["error"] = f"{exc.__class__.__name__}: {exc}"
    print(json.dumps(report))
    return 0 if report.get("ok") else 1


def serve(model_dir: str, parent_pid: int) -> int:
    """The request loop. One model, one operation, and an idle clock."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    engine = None
    while True:
        try:
            frame = read_frame(stdin)
        except Exception as exc:  # noqa: BLE001 - a bad frame ends the session
            _note(f"unreadable request: {exc.__class__.__name__}")
            return 1
        if frame is None:
            return 0
        header, payload = frame
        request = header.get("id")
        operation = str(header.get("op") or "")
        try:
            if operation == "start":
                death = containment(int(header.get("parent_pid") or parent_pid or 0))
                engine = Engine(model_dir)
                engine.load()
                write_frame(stdout, {"id": request, "ok": True, "op": "ready",
                                     "protocol_version": PROTOCOL_VERSION,
                                     "backend": "deepfilternet",
                                     "parent_death": death,
                                     "idle_seconds": IDLE_SECONDS,
                                     "intraop_threads": INTRAOP_THREADS,
                                     **engine.versions})
            elif operation == "clean":
                if engine is None:
                    raise RuntimeError("the model is not loaded")
                rate = int(header.get("rate") or MODEL_RATE)
                if rate < 8000 or rate > 192000:
                    raise ValueError(f"unsupported rate {rate}")
                started = time.monotonic()
                cleaned = engine.clean(payload, rate)
                write_frame(stdout, {"id": request, "ok": True, "op": "cleaned",
                                     "rate": rate,
                                     "elapsed_ms": int((time.monotonic() - started) * 1000)},
                            cleaned)
            elif operation == "stop":
                write_frame(stdout, {"id": request, "ok": True, "op": "stopping"})
                return 0
            else:
                raise ValueError(f"unknown operation {operation!r}")
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            _note(f"{operation or 'request'} failed: {exc.__class__.__name__}")
            try:
                write_frame(stdout, {"id": request, "ok": False,
                                     "error": f"{exc.__class__.__name__}: {exc}"})
            except Exception:
                return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(MARKER, action="store_true", dest="marker")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--parent-pid", type=int, default=0)
    found, _rest = parser.parse_known_args(argv)
    if found.selftest:
        return selftest(found.model)
    return serve(found.model, found.parent_pid)


if __name__ == "__main__":
    raise SystemExit(main())
