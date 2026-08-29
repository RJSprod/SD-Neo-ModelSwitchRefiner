"""Storytime, borrowed to manufacture one voicepack, and never left running.

Cloning is the only part of this feature that runs somebody else's program, and
the whole module is arranged around one sentence from section 58: Storytime is
a manufacturing tool, not a second speech server. It is started for one job, it
produces one file, and when that file has been validated and installed into the
Kokoro voice bank the process is gone. Nothing on the speech path ever calls
it. A user who clones a voice and then talks for an hour is talking to the same
sherpa-onnx worker they were talking to before.

What this module owns
---------------------
    installation     is a validated cloning bundle present, and where
    one job          at most, ever, with its state readable while it runs
    the process      started, watched, aborted, and *contained*
    the transaction  validated .bin -> bank -> runtime proof -> registry

What it deliberately does not own
---------------------------------
Speech. There is no synthesis in this file and no path from Conversation to it.
Cloning failing, cloning being absent, and cloning being mid-run are all states
in which Voice Chat works exactly as it does today -- section 84.

Containment is the release blocker
----------------------------------
Section 70 and release blocker nine: a clone worker must not outlive the WebUI.
Storytime's own default is a run of hours, so this is not a theoretical
concern -- a user who closes Forge during a clone would otherwise be left with
a CPU-saturating process and no window to stop it from.

Two mechanisms, one per supported platform, and neither is "we call terminate
and hope":

    POSIX     ``start_new_session=True`` puts the child in a process group of
              its own, and every stop signals the *group* -- so a Storytime
              that has shelled out to anything takes its children with it.
    Windows   a job object with ``KILL_ON_JOB_CLOSE``, created here and closed
              when this process ends however it ends. Not offered yet, because
              section 59 is explicit that Windows cloning waits for a pinned
              CPU bundle that has actually been tested.

The escalation is polite first because Storytime asks for that: SIGINT stops it
after the current optimization step and leaves a resumable checkpoint. Then
SIGTERM, then SIGKILL to the group, each bounded. Abort is never allowed to
depend on cooperation.

macOS is not a supported cloning platform and this file says so rather than
letting it half-work: Storytime's ONNX backend uses the CoreML execution
provider there, so ``--backend onnx`` is not a CPU-only claim anybody should
make on that platform (section 59).

Nothing is built from a display name
------------------------------------
Section 77. The clone's name on disk is a UUID this process generated; the
display name goes in the registry and nowhere else. The command is an argument
array built from a manifest template with only generated values substituted,
and it is never a string handed to a shell.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import struct
import subprocess
import threading
import time
import uuid
from pathlib import Path

import mc_voice_bank as bank
import mc_voice_models as models
import mc_voice_paths as paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

SUPPORTED = ("linux",)
"""Where cloning is offered at all. Section 59.

Not a guess about where the binary would run -- a statement about where the
CPU-only claim and the process-tree containment have both been thought through.
Everywhere else, Settings says cloning is unavailable and Voice Chat is
otherwise identical.
"""

MAX_REFERENCE_BYTES = 32 * 1024 * 1024
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 120.0
TARGET_RATE = 24000
"""Storytime's own recommendation is a 10-20 second mono 24 kHz recording. The
bounds are wider than the recommendation and are still bounds."""

LOG_TAIL = 40
"""Lines of Storytime's own output kept for diagnostics.

Bounded, and it is bounded because it is *shown*: an unbounded buffer of
another program's stdout is a memory leak with a long run time and an audience.
"""

STOP_GRACE = 20.0
TERMINATE_GRACE = 5.0
KILL_GRACE = 2.0
"""SIGINT lets Storytime finish its current step and checkpoint, which can take
a few seconds on a CPU. Then the escalation stops asking."""

IDLE = "idle"
PREPARING = "preparing"
CLONING = "cloning"
VALIDATING_SOURCE = "validating_source"
BUILDING_BANK = "building_bank"
VALIDATING_RUNTIME = "validating_runtime"
COMPLETE = "complete"
FAILED = "failed"
ABORTING = "aborting"
ABORTED = "aborted"
INTERRUPTED = "interrupted"

OPT_ROOT = "model_chain_voice_cloning_root"
"""Where a prepared cloning bundle is, when it is not under the voice root.

A setting because a 350 MB bundle is exactly the kind of thing somebody already
has unpacked somewhere, and copying it to be tidy is not a service.
"""


class CloneError(RuntimeError):
    """A cloning request that could not be served. Never fatal to Voice Chat."""


# --------------------------------------------------------------------------- #
# The installed bundle
# --------------------------------------------------------------------------- #


def _spec() -> dict:
    found = models.manifest().get("cloning")
    return found if isinstance(found, dict) else {}


def supported() -> bool:
    system, _machine, _python = models.current_platform()
    return system in tuple(_spec().get("supported_systems") or SUPPORTED)


def root() -> Path:
    """Where the cloning bundle is looked for."""
    try:
        from modules import shared

        configured = getattr(shared.opts, OPT_ROOT, None)
    except Exception:
        configured = None
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return paths.cloning_root()


def executable(base: Path = None):
    """The Storytime binary inside a bundle, or ``None``.

    Chosen from the manifest's allowlist and required to sit under the
    validated root (section 61 and 77): "point Voice Chat at a folder" must not
    become "point Voice Chat at an executable".
    """
    base = Path(base or root())
    for name in (_spec().get("layout") or {}).get("executable") or ():
        candidate = base / name
        try:
            candidate.resolve().relative_to(base.resolve())
        except (OSError, ValueError):
            continue
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def validate(base=None) -> dict:
    """Report every required part of a bundle separately. Section 61.

    Separately is the requirement: "cloning is not installed" sends somebody
    back to the beginning, and "the speaker encoder is missing" tells them the
    one file to fetch.
    """
    base = Path(base or root())
    layout = _spec().get("layout") or {}
    checks = []

    binary = executable(base)
    checks.append({"item": "Storytime executable", "ok": bool(binary),
                   "detail": "" if binary else "bin/storytime is missing or not executable"})
    for name in layout.get("required_files") or ():
        checks.append({"item": name, "ok": (base / name).is_file(),
                       "detail": "" if (base / name).is_file() else "missing"})
    for name in layout.get("required_dirs") or ():
        checks.append({"item": name, "ok": (base / name).is_dir(),
                       "detail": "" if (base / name).is_dir() else "missing"})
    voices = sorted((base / "assets" / "voices").glob("*.bin")) if (
        base / "assets" / "voices").is_dir() else []
    wanted = int(layout.get("min_voices") or 0)
    checks.append({"item": "Base voices", "ok": len(voices) >= wanted,
                   "detail": f"{len(voices)} found, {wanted} needed"})
    return {"root": str(base), "ok": all(item["ok"] for item in checks), "checks": checks}


def _record() -> dict:
    try:
        return json.loads((root() / "installed.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remember_install(found: dict) -> None:
    path = root() / "installed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=2), encoding="utf-8")


def installation() -> dict:
    """What Settings draws for the cloning row. Never raises, never installs.

    Revalidated on every call rather than trusted from ``installed.json``
    (section 63): a bundle whose assets were moved or half-deleted is a bundle
    that would otherwise report Installed and fail at the first clone.
    """
    if not supported():
        system, _machine, _python = models.current_platform()
        return {"state": "unsupported", "supported": False, "message":
                f"Voice cloning is not offered on {system}. Voice Chat itself is unaffected, "
                f"and voices cloned elsewhere still work here.",
                "sources": _sources(), "capacity": _capacity()}
    found = validate()
    stored = _record()
    if not found["ok"]:
        started = any(item["ok"] for item in found["checks"])
        return {"state": "invalid" if started else "not_installed",
                "supported": True, "checks": found["checks"], "root": "",
                "message": ("Voice cloning is not installed."
                            if not started else
                            "The cloning folder is missing some of its parts."),
                "sources": _sources(), "capacity": _capacity()}
    return {"state": "installed", "supported": True, "checks": found["checks"], "root": "",
            "version": str(stored.get("version") or ""),
            "cpu_validated": bool(stored.get("cpu_validated")),
            "message": "Installed — CPU only." if stored.get("cpu_validated")
                       else "Installed. Checking the CPU backend…",
            "sources": _sources(), "capacity": _capacity()}


def _capacity() -> dict:
    try:
        import mc_voice_registry as registry

        return registry.capacity()
    except Exception:
        return {"used": 0, "total": bank.CUSTOM_CAPACITY, "free": bank.CUSTOM_CAPACITY}


def _sources() -> dict:
    """The addresses the manual instructions show. From the manifest, not here."""
    found = _spec()
    return {"upstream": str(found.get("upstream") or ""),
            "kokoro": str(found.get("kokoro") or ""),
            "note": str(found.get("note") or ""),
            "pinned": bool(found.get("platforms") or [])}


def adopt(folder: str) -> dict:
    """Use a bundle somebody prepared. The manual path, and the only one today.

    One-click installation is not offered because no pinned bundle exists to
    offer: section 78 requires a version, a URL, a size and a checksum before
    anything is downloaded, and a "one-click" that resolved a package at
    install time would be exactly the trust model the rest of this feature
    refuses. When a pinned bundle is published, ``cloning.platforms`` in the
    manifest gains an entry and :func:`install` stops saying so.
    """
    if not supported():
        raise CloneError(installation()["message"])
    wanted = str(folder or "").strip()
    if not wanted:
        raise CloneError("Give Voice Chat the folder your cloning bundle is in.")
    base = Path(wanted).expanduser()
    if not base.is_dir():
        raise CloneError("That folder does not exist.")
    base = base.resolve()
    found = validate(base)
    if not found["ok"]:
        missing = ", ".join(item["item"] for item in found["checks"] if not item["ok"])
        raise CloneError(f"That folder is missing: {missing}.")
    _remember(OPT_ROOT, str(base))
    report = self_check(base)
    _remember_install({"root": str(base), "version": report.get("version", ""),
                       "cpu_validated": bool(report.get("ok")),
                       "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if not report.get("ok"):
        raise CloneError("That Storytime build did not answer a version check, so it was not "
                         "marked usable. " + str(report.get("error") or ""))
    logger.info("Model Chain: a Voice cloning bundle was adopted — %s",
                report.get("version") or "no version reported")
    return installation()


def install(folder: str = "") -> dict:
    """One click, when there is something pinned to click. Otherwise honest."""
    if folder:
        return adopt(folder)
    if not _spec().get("platforms"):
        raise CloneError(
            "There is no pinned cloning bundle published for this build yet, so there is "
            "nothing for Voice Chat to download and verify. Use Manual Setup to point it at "
            "a Storytime folder you have prepared.")
    raise CloneError("Voice Chat could not install the cloning bundle.")


def self_check(base=None) -> dict:
    """Run the binary once, with GPUs hidden, and see whether it answers.

    Section 60, step 8. It is a version query rather than a synthesis because
    what is being established is that this file is the program it claims to be
    and that it starts on this machine -- and because a clone is hours and a
    self-check must not be.
    """
    binary = executable(base)
    if binary is None:
        return {"ok": False, "error": "no executable"}
    argv = [str(binary)] + [str(part) for part in
                            ((_spec().get("command") or {}).get("version_argv") or ["--version"])]
    try:
        found = subprocess.run(  # noqa: S603 - an allowlisted path with fixed arguments
            argv, capture_output=True, timeout=60, env=_environment(),
            cwd=str(Path(binary).parent))
    except Exception as exc:
        return {"ok": False, "error": exc.__class__.__name__}
    text = (found.stdout or b"").decode("utf-8", "replace").strip().splitlines()
    return {"ok": found.returncode == 0, "version": (text[0] if text else "")[:120],
            "error": "" if found.returncode == 0 else f"exit {found.returncode}"}


def _environment() -> dict:
    """The child's environment: offline, CPU, and nothing about this WebUI.

    Section 67. Hiding the GPU devices is belt and braces on top of using a
    CPU-only build -- the point is that a bundle which *did* have an
    accelerated provider compiled in still cannot reach a device, so the
    "cloning never touches the graphics card" claim does not rest on trusting
    somebody else's build flags.
    """
    environ = dict(os.environ)
    environ.update({
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "GPU_DEVICE_ORDINAL": "",
        "ORT_DISABLE_CUDA": "1",
        "HF_HUB_OFFLINE": "1",
        "NO_PROXY": "*",
        "OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 4) // 2)),
    })
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        environ.pop(name, None)
    return environ


def _remember(name: str, value) -> None:
    try:
        from modules import shared

        shared.opts.set(name, value)
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: could not persist a cloning setting", exc_info=True)


# --------------------------------------------------------------------------- #
# The reference recording
# --------------------------------------------------------------------------- #


def normalize_wav(data: bytes) -> tuple:
    """A user's WAV as mono 24 kHz PCM16, or a refusal that says why.

    Storytime's own instructions are ``ffmpeg -ar 24000 -ac 1``, and asking
    somebody to run ffmpeg before they can press a button is the difference
    between a feature and a procedure. Done here, in the standard library, with
    a linear resampler -- which is adequate because what follows is a
    style-space hill climb scored on speaker similarity, not a mastering chain.
    """
    if not data:
        raise CloneError("No recording was received.")
    if len(data) > MAX_REFERENCE_BYTES:
        raise CloneError("That recording is too large.")
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise CloneError("That file is not a WAV recording.")

    offset, fmt, body = 12, None, b""
    while offset + 8 <= len(data):
        name = data[offset:offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        start = offset + 8
        if start + size > len(data):
            raise CloneError("That recording's header is malformed.")
        if name == b"fmt " and size >= 16:
            fmt = struct.unpack_from("<HHIIHH", data, start)
        elif name == b"data":
            body = data[start:start + size]
        offset = start + size + (size % 2)
    if fmt is None or not body:
        raise CloneError("That recording is not a complete WAV.")

    encoding, channels, rate, _bps, _align, bits = fmt
    if encoding != 1 or bits != 16:
        raise CloneError("Voice cloning accepts uncompressed 16-bit PCM WAV recordings.")
    if channels not in (1, 2):
        raise CloneError("Voice cloning accepts mono or stereo recordings.")
    if rate < 8000 or rate > 192000:
        raise CloneError("That recording's sample rate is not one Voice Chat can use.")

    import array

    samples = array.array("h")
    samples.frombytes(body[: len(body) - (len(body) % (2 * channels))])
    import sys as _sys

    if _sys.byteorder == "big":
        samples.byteswap()
    if channels == 2:
        samples = array.array("h", [(samples[i] + samples[i + 1]) // 2
                                    for i in range(0, len(samples) - 1, 2)])
    seconds = len(samples) / float(rate or 1)
    if seconds < MIN_REFERENCE_SECONDS:
        raise CloneError(f"That recording is {seconds:.1f} seconds long. Record at least "
                         f"{MIN_REFERENCE_SECONDS:.0f} seconds — ten to twenty is ideal.")
    if seconds > MAX_REFERENCE_SECONDS:
        raise CloneError(f"That recording is {seconds:.0f} seconds long and the limit is "
                         f"{MAX_REFERENCE_SECONDS:.0f}.")

    if rate != TARGET_RATE:
        wanted = int(len(samples) * TARGET_RATE / float(rate))
        ratio = len(samples) / float(max(1, wanted))
        resampled = array.array("h", bytes(2 * wanted))
        for index in range(wanted):
            position = index * ratio
            left = int(position)
            right = min(left + 1, len(samples) - 1)
            weight = position - left
            resampled[index] = int(samples[left] * (1.0 - weight) + samples[right] * weight)
        samples = resampled
    if _sys.byteorder == "big":
        samples.byteswap()

    raw = samples.tobytes()
    header = (b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, TARGET_RATE, TARGET_RATE * 2, 2, 16)
              + b"data" + struct.pack("<I", len(raw)))
    return header + raw, len(raw) / float(TARGET_RATE * 2)


# --------------------------------------------------------------------------- #
# One job
# --------------------------------------------------------------------------- #


_lock = threading.RLock()
_job: dict = {}
_process = None
_group = 0
"""The process *group* of the running clone, captured at launch.

Captured rather than looked up when it is needed, because by the time it is
needed the process may be a zombie that has already been waited for -- and
``os.getpgid`` on a reaped pid raises. A group id kept from the moment
``Popen`` returned is the only handle that stays valid for the whole job.
"""
_job_handle = None


def state() -> dict:
    """The clone job, safe to show and safe to poll. Never any content."""
    with _lock:
        found = dict(_job)
    found.pop("reference", None)
    found.pop("output", None)
    found.pop("abort_as", None)
    found.setdefault("status", IDLE)
    found["active"] = found["status"] in (PREPARING, CLONING, VALIDATING_SOURCE,
                                          BUILDING_BANK, VALIDATING_RUNTIME, ABORTING)
    return found


def start(display_name: str, language: str, wav: bytes) -> dict:
    """Begin one clone. Everything that can be refused is refused here.

    Before the reference is even written: the platform, the installation, the
    capacity (section 76 -- a full bank must not be discovered after hours of
    optimization), the name and the recording. What is left after all of that
    is a job that can only fail for reasons nobody could have checked in
    advance.
    """
    import mc_voice_registry as registry

    if not supported():
        raise CloneError(installation()["message"])
    found = installation()
    if found["state"] != "installed":
        raise CloneError("Voice cloning is not installed. Use Settings → Voice Chat.")
    with _lock:
        if state()["active"]:
            raise CloneError("A voice is already being cloned. Wait for it or abort it.")
    name = registry.check_name(display_name)
    slot = registry.free_slot()
    audio, seconds = normalize_wav(wav)

    internal = uuid.uuid4().hex
    reference = paths.reference_file(internal)
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(audio)

    job = {
        "job_id": uuid.uuid4().hex[:12],
        "voice_id": internal,
        "display_name": name,
        "label": f"* {name}",
        "language": "en-GB" if str(language).upper().endswith("GB") else "en-US",
        "status": PREPARING,
        "step": 0,
        "total_steps": int((_spec().get("command") or {}).get("default_steps") or 2000),
        "percent": 0,
        "score": None,
        "reference": str(reference),
        "reference_seconds": round(seconds, 1),
        "proposed_slot": slot,
        "started_at": time.time(),
        "updated_at": time.time(),
        "error": "",
        "log": [],
    }
    with _lock:
        _job.clear()
        _job.update(job)
    threading.Thread(target=_run, args=(dict(job),), name="mc-voice-clone", daemon=True).start()
    logger.info("Model Chain: a voice clone started — job %s, slot %d, %.1f s reference",
                job["job_id"], slot, seconds)
    return state()


ABORT_CLEANUP = 5.0
"""How long Abort waits for the supervisor to finish clearing up.

Bounded and not zero, and the reason is a race worth naming: killing the
process is quick, and deleting the reference recording and the checkpoints
happens on the supervisor thread when it comes back from ``wait()``. Reporting
"aborted" before that had happened would mean a UI that says the job is gone
while its files are still on disk -- and a test that looked would sometimes see
them and sometimes not.
"""


def abort(reason: str = ABORTED) -> dict:
    """Stop and discard. Section 69, including the hard-kill fallback.

    The status stays ``aborting`` until the process is dead *and* its artifacts
    are gone; only then does it become ``aborted`` (or ``interrupted``, when the
    WebUI is what stopped it). So "the job is no longer active" means the whole
    of section 69 has happened rather than only its first step.
    """
    with _lock:
        if not state()["active"]:
            return state()
        _job["status"] = ABORTING
        _job["abort_as"] = reason
        _job["updated_at"] = time.time()
        supervised = _process is not None
    _kill_tree()
    if supervised:
        deadline = time.monotonic() + ABORT_CLEANUP
        while time.monotonic() < deadline:
            with _lock:
                if _job.get("status") != ABORTING:
                    break
            time.sleep(0.05)
    with _lock:
        if _job.get("status") == ABORTING:
            # No supervisor to finish the job -- it never got as far as a
            # process. Finish it here rather than leaving a job that says it is
            # still stopping for ever.
            _job["status"] = _job.pop("abort_as", reason)
            _job["updated_at"] = time.time()
    logger.info("Model Chain: a voice clone was %s", state()["status"])
    return state()


def shutdown() -> None:
    """Door A for cloning: the WebUI is going. Idempotent, never raises.

    A clone in progress is *interrupted*, not failed, and the difference is
    section 74: telling somebody their clone failed when what happened is that
    they closed the WebUI is a lie about their own action.
    """
    try:
        with _lock:
            active = state()["active"]
        if active:
            abort(INTERRUPTED)
        _clean_reference()
    except Exception:
        logger.debug("Model Chain: the Voice cloning shutdown hook failed", exc_info=True)


# --------------------------------------------------------------------------- #
# The supervisor
# --------------------------------------------------------------------------- #


def _run(job: dict) -> None:
    """One clone from launch to registry, on a thread of its own."""
    global _process, _group

    binary = executable()
    base = root()
    command = _spec().get("command") or {}
    name = job["voice_id"]
    output = base / str(command.get("output") or "assets/voices/{name}.bin").format(name=name)
    argv = [str(binary)] + [
        str(part).format(assets=str(base / "assets"), reference=job["reference"],
                         name=name, steps=job["total_steps"], seed=0)
        for part in (command.get("clone_argv") or [])]

    try:
        _set(job["job_id"], status=CLONING)
        started = subprocess.Popen(  # noqa: S603 - an allowlisted path, an argument array
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(base), env=_environment(), bufsize=1, text=True,
            errors="replace", **_containment())
        with _lock:
            _process = started
            _group = _group_of(started)
            _job["worker_pid"] = started.pid
        _own(started)
        # The output is read on a thread and the exit is waited for here, and
        # that is not symmetry for its own sake. Storytime may start children of
        # its own -- espeak-ng, on some paths -- and a child inherits the write
        # end of this pipe. So end-of-output does not mean end-of-process: a
        # loop that read until EOF would sit there after Storytime had exited,
        # holding a finished clone hostage to a grandchild that is still alive.
        # Waiting for the *process* is the truth, and closing the pipe
        # afterwards is what lets the reader go.
        reader = threading.Thread(target=_read_output, args=(job["job_id"], started),
                                  name="mc-voice-clone-log", daemon=True)
        reader.start()
        code = started.wait()
        # Storytime has exited; anything left in its group is a child of its
        # own that should not outlive the job either (section 58). Ending the
        # group is what releases the write end of the pipe this reader is
        # blocked on, which is why it happens before the join rather than in
        # some later cleanup.
        _reap_group()
        reader.join(timeout=2.0)
    except FileNotFoundError:
        _fail(job, "The Storytime program could not be started. Check Settings → Voice Chat.")
        return
    except Exception:
        logger.warning("Model Chain: a voice clone could not be started", exc_info=True)
        _fail(job, "The clone could not be started.")
        return
    finally:
        with _lock:
            _process = None
            _group = 0
            _job.pop("worker_pid", None)

    with _lock:
        stopping = _job.get("status") in (ABORTING, ABORTED, INTERRUPTED)
    if stopping:
        _discard(job)
        return
    if code != 0:
        _fail(job, f"Storytime stopped with exit code {code}.")
        _discard(job, keep_status=True)
        return
    if not output.is_file():
        _fail(job, "Storytime finished without producing a voicepack.")
        _discard(job, keep_status=True)
        return
    _install(job, output)


def _install(job: dict, output: Path) -> None:
    """Validate, build, prove, commit. The transaction of section 55."""
    import mc_voice_registry as registry

    try:
        _set(job["job_id"], status=VALIDATING_SOURCE)
        bank.read_voicepack(output)
        _set(job["job_id"], status=BUILDING_BANK)
        _wait_for_quiet()
        _set(job["job_id"], status=VALIDATING_RUNTIME)
        entry = registry.register(job["display_name"], job["language"], output,
                                  identifier=job["voice_id"])
    except Exception as exc:
        logger.warning("Model Chain: a finished clone was not installed", exc_info=True)
        _fail(job, _short(exc))
        _discard(job, keep_status=True)
        return
    _set(job["job_id"], status=COMPLETE, percent=100, voice=entry["id"], error="")
    _discard(job, keep_status=True)
    logger.info("Model Chain: a voice clone completed and was registered at sid %d",
                entry["sid"])


def _wait_for_quiet(timeout: float = 30.0) -> None:
    """Let an active reply finish before the bank is swapped. Section 88.

    Bounded, and then it cancels: waiting forever for a conversation to stop
    would make "install my new voice" depend on the user stopping talking, and
    replacing ``voices.bin`` under a loaded ``OfflineTts`` is the one thing
    section 88 says must not happen.
    """
    try:
        import mc_voice_turn as turns
    except Exception:
        return
    deadline = time.monotonic() + timeout
    while turns.busy() and time.monotonic() < deadline:
        time.sleep(0.25)
    if turns.busy():
        turns.cancel_active("voice bank reload")


def _read_output(job_id: str, started) -> None:
    """Storytime's own lines, until the pipe closes. Never content, always bounded."""
    try:
        for line in iter(started.stdout.readline, ""):
            _progress(job_id, line)
    except Exception:
        # A closed pipe is how this thread is asked to stop.
        logger.debug("Model Chain: the cloning log reader ended", exc_info=True)


def _progress(job_id: str, line: str) -> None:
    """Turn one line of Storytime's output into numbers, or keep it as a tail.

    Deliberately forgiving about the shape: an exact format string here would
    be a parser that silently stops working when the tested build changes its
    progress line, and the failure mode of *that* is a progress bar frozen at
    zero for four hours.
    """
    text = str(line or "").rstrip()
    if not text:
        return
    found = {}
    step = re.search(r"\bstep\D{0,3}(\d+)\s*(?:/|of)\s*(\d+)", text, re.I) \
        or re.search(r"\b(\d+)\s*/\s*(\d+)\b", text)
    if step:
        found["step"] = int(step.group(1))
        found["total_steps"] = max(1, int(step.group(2)))
        found["percent"] = min(100, int(100 * found["step"] / found["total_steps"]))
    percent = re.search(r"(\d{1,3})\s?%", text)
    if percent and "percent" not in found:
        found["percent"] = min(100, int(percent.group(1)))
    score = re.search(r"\b(?:best|score|similarity)\D{0,3}([01]?\.\d+)", text, re.I)
    if score:
        found["score"] = float(score.group(1))
    with _lock:
        if _job.get("job_id") != job_id:
            return
        _job.update(found)
        _job["updated_at"] = time.time()
        tail = _job.setdefault("log", [])
        tail.append(text[:200])
        del tail[:-LOG_TAIL]


def _set(job_id: str, **values) -> None:
    with _lock:
        if _job.get("job_id") != job_id:
            return
        _job.update(values)
        _job["updated_at"] = time.time()


def _fail(job: dict, message: str) -> None:
    _set(job["job_id"], status=FAILED, error=str(message or "The clone failed.")[:300])


def _short(exc: BaseException) -> str:
    text = str(exc or "").strip()
    return text[:300] if text else "The clone could not be installed."


def _discard(job: dict, keep_status: bool = False) -> None:
    """Remove this job's reference recording and checkpoints. Section 69.

    Runs on completion, on abort and on failure -- the three ends a job has --
    because the reference is somebody's voice and the default policy for it is
    deletion (section 79).
    """
    _clean_reference(job.get("reference"))
    base, command = root(), _spec().get("command") or {}
    for template in command.get("checkpoints") or ():
        try:
            (base / str(template).format(name=job["voice_id"])).unlink()
        except (OSError, KeyError, ValueError):
            pass
    if not keep_status:
        with _lock:
            if _job.get("job_id") == job.get("job_id") and _job.get("status") == ABORTING:
                _job["status"] = _job.pop("abort_as", ABORTED)
                _job["updated_at"] = time.time()
    try:
        # Storytime's own copy goes too. What a registered clone is spoken from
        # is the canonical ``clones/<uuid>.bin`` the bank builder wrote, so a
        # second copy inside somebody else's assets directory is a stale file
        # that would confuse the next ``--list-voices``.
        output = base / str(command.get("output") or "").format(name=job["voice_id"])
        output.unlink(missing_ok=True)
    except (OSError, KeyError, ValueError, IndexError):
        pass


def _clean_reference(path=None) -> None:
    try:
        if path:
            Path(path).unlink(missing_ok=True)
            return
        for item in paths.reference_root().glob("*.wav"):
            item.unlink(missing_ok=True)
    except OSError:
        logger.debug("Model Chain: could not remove a clone reference recording",
                     exc_info=True)


# --------------------------------------------------------------------------- #
# Containment
# --------------------------------------------------------------------------- #


def _containment() -> dict:
    """Popen keywords that put the child somewhere it can be killed as a whole."""
    if os.name == "nt":
        return {"creationflags": 0x00000200 | 0x00004000}  # NEW_PROCESS_GROUP, BELOW_NORMAL
    return {"start_new_session": True, "preexec_fn": _child_priority}


def _child_priority() -> None:  # pragma: no cover - runs in the forked child
    """Be the process that yields. A clone is hours; a reply is now."""
    try:
        os.nice(10)
    except Exception:
        pass


def _own(started) -> None:
    """Windows: put the child in a job object that dies with this process."""
    global _job_handle

    if os.name != "nt":
        return
    handle = getattr(started, "_handle", None)
    if handle is None:
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    if _job_handle is None:
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        class _Limits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _Limits),
                        ("IoInfo", ctypes.c_byte * 48),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        information = _Extended()
        information.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(information),
                                              ctypes.sizeof(information)):
            raise ctypes.WinError(ctypes.get_last_error())
        _job_handle = job
    kernel.AssignProcessToJobObject(_job_handle, int(handle))


def _group_of(started) -> int:
    if os.name == "nt":
        return 0
    try:
        return os.getpgid(started.pid)
    except Exception:
        return 0


def _reap_group() -> None:
    """Make sure nothing survives a finished clone. Never raises."""
    with _lock:
        group = _group
    if not group or os.name == "nt":
        return
    try:
        os.killpg(group, getattr(signal, "SIGKILL", signal.SIGTERM))
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _kill_tree() -> None:
    """Ask, then insist, then stop asking -- to the group, never one pid.

    Storytime stops on SIGINT after its current step, which is why the polite
    request comes first and is given real time. Everything after it is bounded,
    and the last step does not depend on the child agreeing to anything.
    """
    with _lock:
        started, group = _process, _group
    if started is None or started.poll() is not None:
        _reap_group()
        return
    for wanted, grace in ((signal.SIGINT, STOP_GRACE), (signal.SIGTERM, TERMINATE_GRACE),
                          (getattr(signal, "SIGKILL", signal.SIGTERM), KILL_GRACE)):
        _signal_group(started, group, wanted)
        try:
            started.wait(timeout=grace)
            _reap_group()
            return
        except Exception:
            if started.poll() is not None:
                _reap_group()
                return
    logger.warning("Model Chain: a cloning process did not stop when it was asked to")


def _signal_group(started, group: int, number) -> None:
    """Signal the whole group, and fall back to the one pid only if that fails.

    The group is the point (section 69): terminating the pid Storytime was
    given would leave anything it started running, and "the clone stopped"
    would be a claim about one process rather than about the work.
    """
    try:
        if os.name == "nt":
            if number == signal.SIGINT:
                started.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                started.terminate()
            return
        os.killpg(group or os.getpgid(started.pid), number)
    except Exception:
        try:
            started.send_signal(number)
        except Exception:
            pass
