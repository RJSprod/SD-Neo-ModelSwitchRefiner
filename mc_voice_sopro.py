"""Sopro V2: what is installed, which voices exist, and how one is created.

The Sopro half of the engine facade. Kokoro's equivalent is three modules that
grew separately -- :mod:`mc_voice_models` installs, :mod:`mc_voice_registry`
names, :mod:`mc_voice_clone` clones -- and this is one, because for Sopro they
are one transaction: a voice is *created by the runtime that will speak it*
(section 23), so there is no cloning bundle to install separately, no speaker
bank to rebuild, and no slot to allocate.

Cloning here is reference conditioning, not training
-----------------------------------------------------
Sopro's largest product advantage is that creating a voice is part of the normal
model capability. There is no Storytime, no long optimisation job, no
platform-specific manufacturing tool and no hill climb: a short recording is
validated, normalized, handed to the Sopro worker, and comes back as a few
hundred kilobytes of canonical conditioning. On a Windows CPU that is seconds
rather than hours, and it is the same code path that will later speak it.

What this module does *not* do
------------------------------
It does not import Torch, Sopro, NumPy or anything from the isolated closure. It
holds paths, JSON, options and a transaction; every tensor is the worker's
(I-6). It also does not touch Kokoro: nothing here reads or writes the Kokoro
registry, the voice bank, the STT model choice or the shared voice options
(I-3), and ``tests/test_voice_sopro.py`` asserts that a full Sopro lifecycle
leaves the Kokoro side byte-for-byte unchanged.

The download primitives are shared with :mod:`mc_voice_models`
--------------------------------------------------------------
:func:`_download`, ``_resolve``, ``_unpack_wheel``, ``_promote`` and the
progress plumbing are imported from there rather than written twice. That is not
a violation of I-6 -- those functions run in the *WebUI* process and know nothing
about either engine -- and the alternative is a second, less-tested downloader
whose refusal semantics would drift from the first one's. What is emphatically
not shared is the manifest, the trust root, the staging tree, the self-test or
the installed marker: Sopro's install can fail without touching a working
Kokoro, and Kokoro's can fail without touching a working Sopro.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import struct
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import mc_voice_engines as engines
import mc_voice_models as models
import mc_voice_paths as paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

ENGINE = engines.SOPRO
LABEL = "Sopro V2"
FAMILY = "sopro-v2-turbo"

SCHEMA = 1
"""The manifest schema this code understands. A file claiming a newer one is
refused rather than guessed at."""

REGISTRY_SCHEMA = 1

OPT_VOICE = "model_chain_voice_sopro_voice_id"
OPT_PRECISION = "model_chain_voice_sopro_precision"
OPT_STEPS = "model_chain_voice_sopro_steps"
OPT_CHUNK = "model_chain_voice_sopro_chunk_frames"

PRECISIONS = ("full", "int8")
"""What the pinned Sopro release supports. ``from_pretrained`` takes
``quantization=None`` or ``"int8"`` and refuses int8 anywhere but the CPU, which
is the whole set."""

STEP_CHOICES = (2, 4, 8)
"""Acoustic solver steps. The reviewed V2 default is 2 and upstream documents
larger values as a quality/compute trade. Only a tested set is offered, because
a free integer here is a control that can make Sopro slower than real time."""

CHUNK_CHOICES = (32, 64, 128)
"""Streaming chunk sizes, in frames. The reviewed default is 64 and the value
must be a multiple of the model's hop ratio -- which the worker checks against
the loaded model rather than against a constant here. Smaller chunks may improve
first-audio at the cost of throughput; larger may do the reverse."""

LANGUAGES = (("", "Auto"), ("en", "English"), ("pt", "European Portuguese"),
             ("fr", "French"), ("de", "German"))
"""Sopro's own language tags, and the fifth option that is the absence of one.

A pronunciation hint, not a translation feature, and the help text says so
(section 33). The list is checked against the pinned ``sopro.text.LANGUAGE_TAGS``
by ``tests/test_voice_sopro.py`` so a model revision that added a language
cannot leave the UI silently offering four.
"""

MIN_REFERENCE_SECONDS = 5.0
MAX_REFERENCE_SECONDS = 20.0
"""Sopro V2's documented cloning envelope, and what the UI asks for.

Five to twenty seconds of one clear speaker. Deliberately not a scoring system:
section 24 says to encourage a natural speaking voice and low background noise
rather than invent a quality metric before any measurement justifies one.
"""

MAX_REFERENCE_BYTES = 32 * 1024 * 1024
TARGET_RATE = 24000
"""Sopro's own sample rate, which is what a reference is normalized to.

Sopro would resample anything it is given, and normalizing here means the WAV
that is *retained* is already the canonical one -- so a later rebuild reads the
same bytes the first preparation did.
"""

MIN_PEAK = 0.01
"""Below this a recording is silence with a microphone left on.

A conservative level check rather than a quality judgement (section 25): it
catches a muted input and a wrong device, which are the two ways somebody
records twenty seconds of nothing and cannot tell why the clone came out wrong.
"""

MAX_NAME_CHARS = 48
MAX_TEST_CHARS = 400
DEFAULT_TEST_TEXT = "This is a test of the Sopro voice."

_NAME_OK = re.compile(r"^[\w][\w '\-.()&]*$", re.UNICODE)

_lock = threading.RLock()
_manifest_cache: "dict | None" = None


class SoproError(RuntimeError):
    """Something a user can read and act on, rather than a traceback."""


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Bundle:
    """The one installable Sopro model bundle, as the manifest declares it."""

    identifier: str
    label: str
    repo: str
    revision: str
    license: str
    attribution: str
    summary: str
    notes: str
    about_bytes: int
    ram_bytes: int
    sample_rate: int
    style_ctrl_dim: int
    languages: tuple
    artifacts: tuple
    required_paths: tuple

    @property
    def download_bytes(self) -> int:
        total = sum(int(item.size or 0) for item in self.artifacts)
        return total or int(self.about_bytes or 0)


def manifest(refresh: bool = False) -> dict:
    """The parsed, validated Sopro manifest. Read once and kept.

    Every failure below is a broken *extension*, not a broken installation, so
    the message says so: a user cannot fix a malformed trust root by pressing
    Install again.
    """
    global _manifest_cache

    with _lock:
        if _manifest_cache is not None and not refresh:
            return _manifest_cache
        path = paths.sopro_manifest_path()
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SoproError(f"The Sopro manifest at {path} could not be read ({exc}). This "
                             f"is a problem with the extension rather than with your "
                             f"installation.") from None
        if not isinstance(found, dict) or int(found.get("schema") or 0) != SCHEMA:
            raise SoproError("The Sopro manifest is not in a layout this build understands.")
        runtime = found.get("runtime")
        if not isinstance(runtime, dict) or not runtime.get("platforms"):
            raise SoproError("The Sopro manifest declares no runtime.")
        _manifest_cache = found
        return found


def _manifest_cache_clear() -> None:
    global _manifest_cache

    with _lock:
        _manifest_cache = None


def _artifact(entry: dict) -> models.Artifact:
    """One manifest artifact as the shared downloader's own dataclass.

    Built rather than re-implemented, so an artifact this module fetches goes
    through exactly the checks -- pinned hash, publisher digest, byte ceiling,
    ``.part`` rename -- that a Kokoro artifact goes through.
    """
    return models.Artifact(
        filename=str(entry.get("filename") or ""),
        local_name=str(entry.get("local_name") or entry.get("filename") or ""),
        url=str(entry.get("url") or ""),
        size=(int(entry["bytes"]) if entry.get("bytes") else None),
        sha256=(str(entry["sha256"]).casefold() if entry.get("sha256") else None))


def bundle(identifier: str = "") -> Bundle:
    """The Sopro model bundle this build installs. One, for now, and named."""
    found = manifest()
    wanted = str(identifier or (found.get("defaults") or {}).get("tts") or "")
    entry = (found.get("models") or {}).get(wanted)
    if not isinstance(entry, dict):
        raise SoproError(f"{wanted!r} is not a Sopro model this build knows.")
    return Bundle(
        identifier=wanted,
        label=str(entry.get("label") or wanted),
        repo=str(entry.get("repo") or ""),
        revision=str(entry.get("revision") or "main"),
        license=str(entry.get("license") or ""),
        attribution=str(entry.get("attribution") or ""),
        summary=str(entry.get("summary") or ""),
        notes=str(entry.get("notes") or ""),
        about_bytes=int(entry.get("about_bytes") or 0),
        ram_bytes=int(entry.get("ram_bytes") or 0),
        sample_rate=int(entry.get("sample_rate") or 24000),
        style_ctrl_dim=int(entry.get("style_ctrl_dim") or 8),
        languages=tuple(entry.get("languages") or ()),
        artifacts=tuple(_artifact(item) for item in (entry.get("files") or ())),
        required_paths=tuple(str(name) for name in (entry.get("required_paths") or ())))


def platform() -> "models.RuntimePlatform | None":
    """The wheel closure for this machine, or ``None`` if it is not on the list.

    Section 17: supported combinations are an explicit allowlist, and general
    PyTorch Windows support is not permission to advertise every Python minor.
    A machine that is not in the manifest gets a sentence saying so rather than
    an install that half works.
    """
    system, machine, python_version = models.current_platform()
    for entry in manifest()["runtime"].get("platforms") or ():
        if (str(entry.get("system")) == system
                and machine in tuple(entry.get("machines") or ())
                and str(entry.get("python")) == python_version):
            return models.RuntimePlatform(
                identifier=str(entry.get("id") or ""),
                system=system,
                machines=tuple(entry.get("machines") or ()),
                python=str(entry.get("python") or ""),
                artifacts=tuple(_artifact(item) for item in (entry.get("artifacts") or ())))
    return None


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


@dataclass
class Status:
    """What is installed, in one object the panel and the runtime both read."""

    platform_supported: bool = False
    runtime_ready: bool = False
    model_ready: bool = False
    runtime_message: str = ""
    model_message: str = ""
    label: str = LABEL
    fingerprint: str = ""
    download_bytes: int = 0
    ram_bytes: int = 0
    closure: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return bool(self.runtime_ready and self.model_ready)

    @property
    def message(self) -> str:
        if self.ready:
            return "Installed."
        if not self.platform_supported:
            return self.runtime_message
        missing = [name for name, ok in (("runtime", self.runtime_ready),
                                         ("model", self.model_ready)) if not ok]
        return f"Setup required — the Sopro {' and '.join(missing)} still to install."


def status() -> Status:
    """Read from disk and from verification metadata. Starts nothing.

    Section 17 again: reading status must never begin a download or a model
    load, so every branch below is a file-system question. Never raises -- a
    manifest this build cannot parse is reported as an uninstallable platform
    rather than as an exception on a status route a browser polls.
    """
    try:
        return _status()
    except SoproError as exc:
        return Status(runtime_message=str(exc), model_message=str(exc))
    except Exception:
        logger.debug("Model Chain: could not read the Sopro installation", exc_info=True)
        return Status(runtime_message="Sopro's installation could not be read.",
                      model_message="Sopro's installation could not be read.")


def _status() -> Status:
    chosen = platform()
    entry = bundle()
    found = Status(label=entry.label, download_bytes=entry.download_bytes,
                   ram_bytes=entry.ram_bytes)
    if chosen is None:
        system, machine, python_version = models.current_platform()
        found.platform_supported = False
        found.runtime_message = (
            f"Sopro has no tested CPU runtime for {system}/{machine} on Python "
            f"{python_version}. This release ships the Sopro closure for 64-bit Windows on "
            f"Python 3.10 to 3.13; Kokoro is unaffected and still works here.")
        found.model_message = found.runtime_message
        return found

    found.platform_supported = True
    installed = _read_json(paths.sopro_runtime_manifest())
    closure = str((installed or {}).get("closure") or "")
    if not installed:
        found.runtime_message = ("Not installed — about "
                                 f"{models._bytes_label(sum(int(item.size or 0) for item in chosen.artifacts))} "
                                 "of PyTorch and Sopro, on the CPU.")
    elif closure != chosen.closure_id:
        # The closure id is derived from the platform id and every wheel's
        # hash, so this is true exactly when the pinned closure has changed --
        # which is the moment a saved voice's fingerprint stops meaning what it
        # meant. Reported rather than silently re-used.
        found.runtime_message = ("Installed, but this build pins a different Sopro runtime. "
                                 "Install it again to update.")
    else:
        found.runtime_ready = True
        found.runtime_message = (f"Installed — Sopro {installed.get('sopro_version') or '?'}, "
                                 f"Torch {installed.get('torch_version') or '?'}, CPU only.")
    found.closure = dict(installed or {})

    model = _read_json(paths.sopro_model_root(entry.identifier) / paths.INSTALLED_FILENAME)
    root = paths.sopro_model_root(entry.identifier)
    missing = [name for name in entry.required_paths if not (root / name).exists()]
    if not model or missing:
        found.model_message = (f"Not installed — {entry.label}, about "
                               f"{models._bytes_label(entry.download_bytes)}.")
    else:
        found.model_ready = True
        found.model_message = f"Installed — {entry.label}."

    found.fingerprint = _fingerprint(chosen, installed, model)
    return found


def _fingerprint(chosen, installed: dict, model: dict) -> str:
    """What "this voice was prepared by this build" actually means. Section 15.

    Content and closure rather than a version string, because ``sopro==2.0.5``
    is a name and this has to be an identity. Five inputs:

        the pinned wheel closure, itself a hash of every artifact's digest;
        the Sopro package version actually installed;
        the Torch version actually installed;
        the digests of the model artifacts that were actually verified;
        the adapter's prepared-reference schema version.

    Precision is deliberately *not* one of them. int8 quantizes the
    autoregressive blocks after loading; the speaker encoder, the semantic
    encoder and the vocoder that produce ``cond_vec``, ``semantic_tokens`` and
    ``mel`` are untouched by it -- so a voice prepared at full precision is
    correct at int8 and the reverse. What precision *does* invalidate is the
    warmed ``PromptState``, and that is keyed separately inside the worker
    (section 29).

    Sixteen hex characters: this is compared against a value this installer
    wrote itself rather than defended against a forger.
    """
    from sopro_worker import worker as protocol

    parts = [
        f"closure:{chosen.closure_id}",
        f"sopro:{(installed or {}).get('sopro_version') or ''}",
        f"torch:{(installed or {}).get('torch_version') or ''}",
        f"schema:{protocol.PREPARATION_SCHEMA}",
    ]
    for name, digest in sorted(((model or {}).get("digests") or {}).items()):
        parts.append(f"{name}:{digest}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def runtime_python() -> "Path | None":
    """The isolated Sopro interpreter, or ``None`` when it is not installed."""
    root = paths.sopro_runtime_root() / "env"
    interpreter = ((root / "Scripts" / "python.exe") if os.name == "nt"
                   else (root / "bin" / "python"))
    return interpreter if interpreter.exists() else None


def worker_environment() -> dict:
    """The environment the Sopro worker runs in. I-9, where it is enforced.

    ``CUDA_VISIBLE_DEVICES=""`` is the blunt instrument and the important one: a
    Torch build that would happily have found a GPU finds no devices to
    enumerate, so Sopro cannot claim VRAM an image generation is using. The
    thread caps catch the libraries that read the environment before any of our
    code runs -- Torch's own ``set_num_threads`` is applied inside the worker,
    but OpenMP has usually already decided by then.

    ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` are belt rather than braces:
    the closure has no ``huggingface_hub`` in it at all, so there is nothing to
    turn off. They are set anyway so that a future closure which acquired one
    fails rather than succeeds (section 54).
    """
    from sopro_worker import worker as protocol

    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        "OMP_NUM_THREADS": str(protocol.INTRAOP_THREADS),
        "MKL_NUM_THREADS": str(protocol.INTRAOP_THREADS),
        "OPENBLAS_NUM_THREADS": str(protocol.INTRAOP_THREADS),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def worker_config() -> dict:
    """Everything the worker is told at ``init``. Paths this process built.

    None of it can be influenced from a browser: the model root comes from the
    manifest and the data root, the settings come from the host's options store,
    and the catalogue is built from the registry by :func:`catalog`.
    """
    entry = bundle()
    found = status()
    return {
        "model_root": str(paths.sopro_model_root(entry.identifier)),
        "model_id": entry.identifier,
        "precision": precision(),
        "steps": steps(),
        "chunk_frames": chunk_frames(),
        "fingerprint": found.fingerprint,
        "voices": catalog(),
    }


# --------------------------------------------------------------------------- #
# Engine settings
# --------------------------------------------------------------------------- #


def precision() -> str:
    found = str(_setting(OPT_PRECISION) or "").strip().lower()
    return found if found in PRECISIONS else "full"


def steps() -> int:
    try:
        found = int(_setting(OPT_STEPS) or 0)
    except (TypeError, ValueError):
        found = 0
    return found if found in STEP_CHOICES else STEP_CHOICES[0]


def chunk_frames() -> int:
    try:
        found = int(_setting(OPT_CHUNK) or 0)
    except (TypeError, ValueError):
        found = 0
    return found if found in CHUNK_CHOICES else 64


def engine_settings() -> dict:
    """The three global Sopro runtime controls, as the panel draws them.

    Global rather than per character on purpose (section 13): they change
    compute, RAM and warm-cache compatibility for the whole worker, and a
    character setting that quietly reloaded the model would be a character
    setting nobody could reason about.
    """
    return {
        "precision": precision(),
        "precisions": [{"id": "full", "label": "Full CPU precision",
                        "help": "Sopro exactly as it was released. Slower and larger in "
                                "memory than INT8, and the one to compare against."},
                       {"id": "int8", "label": "INT8 (faster, CPU only)",
                        "help": "Quantizes the autoregressive blocks. Faster and lighter; "
                                "the acoustic solver and the vocoder are unchanged. Saved "
                                "voices stay valid — only the warmed streaming caches are "
                                "rebuilt."}],
        "steps": steps(),
        "step_choices": list(STEP_CHOICES),
        "chunk_frames": chunk_frames(),
        "chunk_choices": list(CHUNK_CHOICES),
    }


def set_engine_settings(precision_id: str = "", solver_steps=None, chunk=None) -> dict:
    """Change a global Sopro runtime setting and stop the worker if it matters.

    Stopped rather than reconfigured: precision is chosen at model load, and the
    solver steps and chunk size decide which warmed ``PromptState`` entries are
    valid at all. Restarting is one cold load; keeping a worker whose caches no
    longer match its settings is a class of bug nobody can hear the start of.

    It never changes Kokoro (I-3), and the next reply starts the new worker
    lazily, so nothing is loaded as a side effect of moving a control.
    """
    changed = []
    wanted = str(precision_id or "").strip().lower()
    if wanted and wanted in PRECISIONS and wanted != precision():
        _remember(OPT_PRECISION, wanted)
        changed.append("precision")
    if solver_steps is not None:
        try:
            value = int(solver_steps)
        except (TypeError, ValueError):
            value = 0
        if value in STEP_CHOICES and value != steps():
            _remember(OPT_STEPS, value)
            changed.append("solver steps")
    if chunk is not None:
        try:
            value = int(chunk)
        except (TypeError, ValueError):
            value = 0
        if value in CHUNK_CHOICES and value != chunk_frames():
            _remember(OPT_CHUNK, value)
            changed.append("streaming chunk size")
    if changed:
        try:
            import mc_voice_sopro_runtime as runtime

            runtime.stop(f"the Sopro {' and '.join(changed)} changed")
        except Exception:
            logger.debug("Model Chain: could not stop Sopro after a settings change",
                         exc_info=True)
        logger.info("Model Chain: Sopro %s changed", " and ".join(changed))
    return engine_settings()


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #


def refusal(manual: bool = False) -> str:
    """Why Sopro cannot be installed right now, or an empty string.

    Asked before anything is started, and separately from starting it, because
    the two questions have different audiences: this one answers a browser that
    needs a sentence to put on screen, and the transaction below answers a log.
    """
    try:
        chosen = platform()
    except SoproError as exc:
        return str(exc)
    if chosen is None:
        system, machine, python_version = models.current_platform()
        return (f"Sopro has no tested CPU runtime for {system}/{machine} on Python "
                f"{python_version}, so it cannot be installed here.")
    with models._lock:
        for kind in ("sopro", "sopro-runtime"):
            if (models._progress.get(kind) or {}).get("running"):
                return "Sopro is already being installed."
    if not manual:
        unpinned = [item.filename for item in chosen.artifacts if not item.pinned]
        if unpinned:
            # The runtime closure is pinned in this repository and must stay
            # pinned: an unpinned wheel is a wheel nobody reviewed, and the
            # whole preparation fingerprint rests on knowing which bytes ran.
            return (f"This build's Sopro runtime is missing a checksum for "
                    f"{unpinned[0]}, so it will not be downloaded. This is a problem with "
                    f"the extension rather than with your installation.")
    return ""


def sources(part: str = "runtime") -> list:
    """Where a person would go to fetch Sopro's files by hand.

    The manifest already names every URL, so the manual panel can print them
    rather than asking somebody to find the right PyTorch wheel themselves --
    which for a hundred-and-forty-megabyte closure across four Python minors is
    not a search anybody should be asked to do.
    """
    if part == "runtime":
        chosen = platform()
        artifacts = chosen.artifacts if chosen else ()
    else:
        artifacts = bundle().artifacts
    return [{"filename": item.filename, "url": item.url, "save_as": item.local_name,
             "archive": False} for item in artifacts]


def install(on_status=None, on_progress=None) -> Status:
    """Install the Sopro runtime and then the Sopro model. One button.

    A transaction, twice over. Nothing outside a staging directory is touched
    until every declared byte has arrived and matched its hash, each half is
    promoted by a directory rename, and a failure anywhere leaves the
    installation exactly as it was -- which for somebody who already had a
    working Kokoro means Kokoro still works, because this never touches it
    (section 60).

    Installing Sopro does not modify or remove Kokoro, and installing Kokoro
    does not install Sopro (section 17). There is no shared staging tree, no
    shared marker file and no shared runtime.
    """
    say = models._narrator("sopro", on_status)
    tick = models._ticker("sopro", on_progress)
    with models._claim("sopro", say, bundle().identifier):
        say("Checking the Sopro runtime…")
        install_runtime(on_status=say, on_progress=lambda f: tick(f * 0.35))
        install_model(on_status=say, on_progress=lambda f: tick(0.35 + f * 0.65))
        tick(1.0)
        say("Sopro installed.")
        return status()


def install_runtime(on_status=None, on_progress=None, folder=None) -> None:
    """The isolated Torch/Sopro closure: an interpreter of its own and the wheels.

    Built exactly the way Kokoro's is -- a virtual environment without pip, and
    the verified wheels *unpacked* rather than installed by a package manager.
    That is the correction to a real failure: pip can report success, exit zero
    and have installed into somebody's ``PIP_TARGET`` or user site, leaving the
    runtime empty. There is no package manager in this path, so there is nothing
    left to redirect it -- and no index resolution can happen inside a user's
    WebUI at runtime (section 16, section 54).
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    chosen = platform()
    if chosen is None:
        raise SoproError(refusal() or "Sopro has no runtime for this platform.")

    installed = _read_json(paths.sopro_runtime_manifest())
    if installed and str(installed.get("closure") or "") == chosen.closure_id \
            and runtime_python() is not None:
        say("The Sopro runtime is already installed.")
        tick(1.0)
        return

    staging = paths.sopro_staging_for("runtime", uuid.uuid4().hex[:8])
    wheels = staging / "wheels"
    shutil.rmtree(staging, ignore_errors=True)
    wheels.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            _adopt(chosen.artifacts, Path(str(folder)).expanduser(), wheels, say,
                   "Sopro runtime wheel")
        else:
            expectations = models._expectations(chosen.artifacts, say)
            models._make_room(chosen.artifacts, staging, expectations)
            models._fetch_all(chosen.artifacts, wheels, say, tick, 0.7, expectations)
        say("Building the isolated Sopro runtime…")
        _build_environment(staging, wheels, chosen)
        tick(0.85)
        say("Checking that Sopro runs on this machine…")
        report = _smoke_test(staging)
        tick(0.95)
        shutil.rmtree(wheels, ignore_errors=True)
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "closure": chosen.closure_id,
            "platform": chosen.identifier,
            "sopro_version": report.get("sopro_version") or "",
            "torch_version": report.get("torch_version") or "",
            "numpy_version": report.get("numpy_version") or "",
            "intraop_threads": report.get("intraop_threads") or 0,
            "interop_threads": report.get("interop_threads") or 0,
            "attention_stable": bool(report.get("attention_stable")),
            "license": manifest()["runtime"].get("license") or "",
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        models._promote(staging, paths.sopro_runtime_root())
        _manifest_cache_clear()
        say("The Sopro runtime is installed.")
        tick(1.0)
        logger.info("Model Chain: the Sopro runtime is installed — %s, Sopro %s, Torch %s, "
                    "attention check passed", chosen.identifier,
                    report.get("sopro_version"), report.get("torch_version"))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_model(on_status=None, on_progress=None, folder=None) -> None:
    """The pinned Sopro V2 Turbo artifacts, verified and promoted by rename."""
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    entry = bundle()
    target = paths.sopro_model_root(entry.identifier)
    marker = _read_json(target / paths.INSTALLED_FILENAME)
    missing = [name for name in entry.required_paths if not (target / name).exists()]
    if marker and not missing:
        say(f"{entry.label} is already installed.")
        tick(1.0)
        return

    staging = paths.sopro_staging_for(entry.identifier, uuid.uuid4().hex[:8])
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            digests = _adopt(entry.artifacts, Path(str(folder)).expanduser(), staging, say,
                             entry.label)
        else:
            expectations = models._expectations(entry.artifacts, say)
            models._make_room(entry.artifacts, staging, expectations)
            digests = models._fetch_all(entry.artifacts, staging, say, tick, 0.9,
                                        expectations)
        gone = [name for name in entry.required_paths if not (staging / name).exists()]
        if gone:
            raise SoproError(f"{entry.label} is missing {gone[0]} after the download. "
                             f"Nothing was installed.")
        _sanity_check(staging, entry)
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "id": entry.identifier,
            "repo": entry.repo,
            "revision": entry.revision,
            "license": entry.license,
            "attribution": entry.attribution,
            "digests": {name: digest for name, digest in sorted(digests.items())},
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        models._promote(staging, target)
        say(f"{entry.label} is installed.")
        tick(1.0)
        logger.info("Model Chain: the Sopro model %s is installed at %s", entry.identifier,
                    target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_from(part: str, folder: str, on_status=None, on_progress=None) -> Status:
    """Install from files already on this machine. The escape hatch.

    Earns its place beyond the situation that prompted it: a machine with no
    route to pypi.org or huggingface.co, a corporate proxy that refuses a
    hundred-megabyte binary, an air-gapped install, or somebody who already has
    these wheels for another application.

    What is different from the managed path is stated rather than glossed. The
    runtime wheels *are* pinned in this repository, so what is supplied is
    checked against a hash committed here and a file under the right name with
    the wrong contents is refused exactly as a bad download is. The model
    artifacts are not pinned here, so their digests are recorded at install time
    and become the constant the *next* install is checked against.
    """
    say = models._narrator("sopro", on_status)
    tick = models._ticker("sopro", on_progress)
    with models._claim("sopro", say, bundle().identifier):
        if part == "runtime":
            install_runtime(on_status=say, on_progress=tick, folder=folder)
        elif part == "model":
            install_model(on_status=say, on_progress=tick, folder=folder)
        else:
            raise SoproError("Sopro installs its runtime or its model.")
        return status()


def _adopt(artifacts, folder: Path, destination: Path, say, what: str) -> dict:
    """Copy named files out of a folder somebody filled themselves, checking each.

    Named files only: nothing is guessed from an extension, nothing is renamed
    from something similar, and a file that is present under the right name with
    the wrong contents is refused. What comes back is the digest of every file
    that was accepted, which is what the installed marker records.
    """
    if not folder.is_dir():
        raise SoproError(f"{folder} is not a folder this machine can read.")
    destination.mkdir(parents=True, exist_ok=True)
    digests = {}
    for item in artifacts:
        source = folder / item.filename
        if not source.is_file():
            source = folder / item.local_name
        if not source.is_file():
            raise SoproError(f"{item.filename} is not in {folder}. Nothing was installed.")
        say(f"Checking {item.filename}…")
        digest = _digest(source)
        if item.sha256 and digest != item.sha256:
            raise SoproError(
                f"{item.filename} is in that folder, but its contents are not what this "
                f"extension's manifest says they should be. Nothing was installed.")
        if item.size and source.stat().st_size != item.size:
            raise SoproError(
                f"{item.filename} is {source.stat().st_size} bytes and this extension "
                f"expects {item.size}. Nothing was installed.")
        target = destination / item.local_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digests[item.filename] = digest
    logger.info("Model Chain: Sopro adopted %d file(s) for the %s from %s",
                len(digests), what, folder)
    return digests


SAFETENSORS_MIN = 16
"""A safetensors file is an eight-byte little-endian header length followed by
that much JSON. Anything shorter is not one, whatever it is called."""


def _sanity_check(staging: Path, entry: Bundle) -> None:
    """Prove the downloaded model files are the shapes of files they claim to be.

    Structural rather than semantic, and that is the point: a proxy that answers
    every request with an HTML error page produces files of plausible size under
    the right names, and "Sopro will not load" three minutes later is a much
    worse report than "that file is not a safetensors".
    """
    for name in entry.required_paths:
        path = staging / name
        size = path.stat().st_size
        if size <= 0:
            raise SoproError(f"{name} downloaded as an empty file. Nothing was installed.")
        if name.endswith(".safetensors"):
            with open(path, "rb") as handle:
                head = handle.read(8)
            if len(head) < 8:
                raise SoproError(f"{name} is not a safetensors file. Nothing was installed.")
            (declared,) = struct.unpack("<Q", head)
            if declared <= 0 or declared + 8 > size or declared > 64 * 1024 * 1024:
                raise SoproError(f"{name} is not a readable safetensors file. Nothing was "
                                 f"installed.")
        elif name.endswith(".json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise SoproError(f"{name} is not readable JSON. Nothing was "
                                 f"installed.") from None
        elif size < SAFETENSORS_MIN:
            raise SoproError(f"{name} is too small to be what it claims. Nothing was "
                             f"installed.")


def _build_environment(staging: Path, wheels: Path, chosen) -> None:
    """An interpreter of its own, and the verified wheels unpacked into it."""
    import venv

    environment = staging / "env"
    builder = venv.EnvBuilder(with_pip=False, clear=True, symlinks=(os.name != "nt"))
    try:
        builder.create(environment)
    except Exception as exc:
        logger.warning("Model Chain: Sopro could not create a virtual environment at %s "
                       "using %s — %s: %s", environment, sys.executable,
                       exc.__class__.__name__, exc)
        raise SoproError(f"The isolated Sopro runtime could not be created "
                         f"({exc.__class__.__name__}: {exc}).") from None

    interpreter = ((environment / "Scripts" / "python.exe") if os.name == "nt"
                   else (environment / "bin" / "python"))
    if not interpreter.exists():
        raise SoproError("The isolated Sopro runtime was created without an interpreter.")

    target = models.site_packages(environment)
    for item in chosen.artifacts:
        wheel = wheels / item.local_name
        if not wheel.is_file():
            raise SoproError(f"{item.filename} is missing from the staged download. Nothing "
                             f"was installed.")
        added = models._unpack_wheel(wheel, target)
        logger.info("Model Chain: Sopro unpacked %s (%s)", item.filename,
                    ", ".join(added[:6]) or "nothing")
    # Checked rather than assumed, because "the installer said it worked" is
    # precisely the claim that turned out to be worthless the last time.
    for name in ("sopro", "torch"):
        if not (target / name).exists():
            raise SoproError(f"The Sopro wheels were unpacked but {name} is not in {target}. "
                             f"Nothing was installed.")


def _smoke_test(staging: Path) -> dict:
    """Prove the staged runtime runs, imports, is CPU-only, and computes correctly.

    Before promotion, so an environment that builds and cannot run never becomes
    the installed one. Run through the Sopro worker's own ``--selftest``, which
    is the same file that will do the inference: a smoke test that imports
    something else is a smoke test of something else.

    The attention check is Gate S-1 and it is the reason this refuses rather
    than warns. A Torch build whose CPU SDPA is not repeatable at the released
    thread policy does not produce slightly worse speech; it produces speech
    that differs between two identical requests, which is a bug nobody can
    reproduce and everybody blames on the model.
    """
    interpreter = ((staging / "env" / "Scripts" / "python.exe") if os.name == "nt"
                   else (staging / "env" / "bin" / "python"))
    if not interpreter.exists():
        raise SoproError(f"The isolated Sopro runtime has no interpreter at {interpreter}.")

    alive = _run_staged(interpreter, ["-c", "import sys; print(sys.version)"],
                        "check that the isolated Sopro interpreter runs", timeout=180)
    if alive.returncode != 0:
        raise SoproError(
            "The isolated Sopro runtime was created but its interpreter will not run. That "
            "usually means this WebUI is running on an embedded or relocated Python that "
            "cannot make a working virtual environment. The interpreter's own error is in "
            "the console. Nothing was installed.")

    result = _run_staged(interpreter, [paths.sopro_worker_script(), "--selftest"],
                         "Sopro self-test", timeout=600)
    try:
        report = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        logger.warning("Model Chain: the Sopro self-test printed %s",
                       models._quote(result.stdout) or "(nothing)")
        raise SoproError("The staged Sopro runtime did not report what it is. Nothing was "
                         "installed.") from None
    if not report.get("ok"):
        raise SoproError(f"The staged Sopro runtime did not pass its self-test: "
                         f"{report.get('error') or 'no reason reported'}. Nothing was "
                         f"installed and Voice Chat is unchanged.")
    if report.get("device") != "cpu":
        raise SoproError(f"The staged Sopro runtime reports device "
                         f"{report.get('device')!r} rather than the CPU, so it was not "
                         f"installed.")
    logger.info("Model Chain: the Sopro self-test passed — Sopro %s, Torch %s, NumPy %s, "
                "%s intra-op / %s inter-op threads, attention stable",
                report.get("sopro_version"), report.get("torch_version"),
                report.get("numpy_version"), report.get("intraop_threads"),
                report.get("interop_threads"))
    return report


def _run_staged(interpreter: Path, arguments: list, what: str, timeout: float = 300):
    """Run something in the staged Sopro runtime and say what happened when it fails.

    A captured stderr is the difference between one round trip and four: an
    install that failed on somebody else's machine has to be diagnosable from
    what they can paste, and "could not import its engine" without the
    interpreter's own words is not.
    """
    import subprocess

    try:
        result = subprocess.run(  # noqa: S603 - a path this module built
            [str(interpreter)] + [str(item) for item in arguments],
            capture_output=True, text=True,
            env={**os.environ, **worker_environment()}, timeout=timeout)
    except Exception as exc:
        logger.warning("Model Chain: Sopro could not run %s (%s: %s) — interpreter %s",
                       what, exc.__class__.__name__, exc, interpreter)
        raise SoproError(f"The staged Sopro runtime could not be started "
                         f"({exc.__class__.__name__}). Nothing was installed.") from None
    if result.returncode != 0:
        logger.warning("Model Chain: Sopro's %s failed with exit code %s.\n"
                       "  interpreter: %s\n  stderr: %s\n  stdout: %s",
                       what, result.returncode, interpreter,
                       models._quote(result.stderr) or "(nothing)",
                       models._quote(result.stdout) or "(nothing)")
    return result


def uninstall() -> Status:
    """Remove Sopro's runtime and model, and keep every saved voice.

    The voices are deliberately kept. A user who uninstalls a hundred and forty
    megabytes of Torch to free space has not asked to throw away the recordings
    they made of themselves, and the retained WAVs are what a later reinstall
    rebuilds from (section 30). Deleting a voice is its own explicit action.
    """
    if engines.active() == ENGINE:
        raise SoproError("Select Kokoro as the text-to-speech engine before removing Sopro.")
    try:
        import mc_voice_sopro_runtime as runtime

        runtime.stop("Sopro is being removed")
    except Exception:
        logger.debug("Model Chain: could not stop Sopro before removing it", exc_info=True)
    for root in (paths.sopro_runtime_root(), paths.sopro_models_root(),
                 paths.sopro_staging_root()):
        if paths.sopro_inside(root):
            shutil.rmtree(root, ignore_errors=True)
    logger.info("Model Chain: Sopro's runtime and model were removed; saved voices were kept")
    return status()


# --------------------------------------------------------------------------- #
# The voice library
# --------------------------------------------------------------------------- #


def _read() -> dict:
    try:
        found = json.loads(paths.sopro_registry_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": REGISTRY_SCHEMA, "voices": []}
    if not isinstance(found, dict) or not isinstance(found.get("voices"), list):
        logger.warning("Model Chain: the Sopro voice registry could not be read and was "
                       "ignored")
        return {"schema": REGISTRY_SCHEMA, "voices": []}
    return found


def _write(found: dict) -> None:
    """Replace the registry atomically. A half-written registry is a lost library.

    Written beside the real file and renamed, on the same filesystem: a registry
    truncated by a power cut would leave prepared voices on disk that nothing can
    name.
    """
    path = paths.sopro_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.new")
    staging.write_text(json.dumps(found, indent=2), encoding="utf-8")
    os.replace(staging, path)


def _out(item: dict) -> dict:
    """One registry record as the facade and the UI see it.

    Qualified id, no paths, no fingerprint internals beyond a short opaque
    string. A display name is metadata and never a filename (section 57), and
    nothing here can be turned back into one.
    """
    identifier = str(item.get("uuid") or "")
    fingerprint = str(item.get("fingerprint") or "")
    return {
        "id": f"{ENGINE}:clone:{identifier}",
        "display_name": str(item.get("display_name") or "Voice"),
        "label": f"* {item.get('display_name') or 'Voice'}",
        "type": "clone",
        "engine": ENGINE,
        "official": False,
        "editable": True,
        "deletable": True,
        "language": str(item.get("language") or ""),
        "accent": _language_label(str(item.get("language") or "")),
        "source_seconds": round(float(item.get("source_seconds") or 0.0), 1),
        "created_at": str(item.get("created_at") or ""),
        "fingerprint": fingerprint[:12],
        "compatible": bool(fingerprint) and fingerprint == status().fingerprint,
        "has_source": _voice_file(identifier, paths.SOPRO_REFERENCE_FILENAME).is_file(),
        "has_lab": _voice_file(identifier, paths.SOPRO_LAB_FILENAME).is_file(),
        "slot": None,
    }


def _language_label(code: str) -> str:
    for identifier, label in LANGUAGES:
        if identifier == str(code or ""):
            return label
    return "Auto"


def _voice_file(identifier: str, name: str) -> Path:
    return paths.sopro_voice_file(identifier, name)


def _records() -> list:
    found = []
    for item in _read().get("voices") or ():
        if not isinstance(item, dict) or not item.get("uuid"):
            continue
        found.append(item)
    return sorted(found, key=lambda item: str(item.get("created_at") or ""))


def entries() -> list:
    """Every Sopro voice this installation has. Qualified ids, no paths."""
    return [_out(item) for item in _records()]


def custom() -> list:
    return entries()


def official() -> list:
    """None, and that is not a placeholder.

    Sopro's library is user-created reference voices. Section 8 is explicit that
    bundled example voices, if ever shipped, must be explicitly licensed and
    represented separately from user clones -- so this returns an empty list
    rather than inventing a category the manifest does not declare.
    """
    return []


def lookup(voice_id: str):
    if not engines.belongs(voice_id, ENGINE):
        return None
    wanted = engines.native(voice_id)
    for item in _records():
        if f"clone:{item['uuid']}" == wanted:
            return _out(item)
    return None


def _uuid_of(voice_id: str) -> str:
    return str(voice_id or "").split(":")[-1]


def default_id() -> str:
    """The configured Sopro default, or the first usable voice, or ``""``.

    Never crosses engines and never invents one. An installation with no Sopro
    voice at all answers with an empty string, and the caller says "create or
    select a Sopro voice" rather than speaking through Kokoro (I-2, section 53).
    """
    wanted = str(_setting(OPT_VOICE) or "").strip()
    if wanted and lookup(wanted) is not None:
        return wanted
    found = entries()
    return found[0]["id"] if found else ""


def default_entry():
    found = default_id()
    return lookup(found) if found else None


def set_default(voice_id: str) -> dict:
    """Commit a new Sopro default. Stores only the stable id (section 28)."""
    entry = lookup(voice_id)
    if entry is None:
        raise SoproError("That Sopro voice does not exist.")
    _remember(OPT_VOICE, entry["id"])
    logger.info("Model Chain: the Sopro default voice is now %s", entry["id"])
    return entry


def resolve(voice_id: str = "") -> tuple:
    """``(qualified id, entry)`` for a stable id, or the Sopro default.

    Raises when there is no Sopro voice at all, which is a different failure
    from "that voice is gone" and is reported differently: one asks the user to
    create a voice, the other quietly falls back to the default and warns.
    """
    entry = lookup(voice_id) if voice_id else None
    if entry is None:
        entry = default_entry()
    if entry is None:
        raise SoproError("No Sopro voice has been created yet.")
    return entry["id"], entry


def rename(voice_id: str, display_name: str) -> dict:
    """Change what a Sopro voice is called, and nothing else.

    The id, the UUID, the directory and every prepared asset are untouched, so a
    rename cannot invalidate a saved default, cannot require a rebuild, and
    cannot change what anybody hears.
    """
    entry = lookup(voice_id)
    if entry is None:
        raise SoproError("That Sopro voice does not exist.")
    wanted = check_name(display_name)
    found = _read()
    for item in found.get("voices") or ():
        if str(item.get("uuid")) == _uuid_of(voice_id):
            item["display_name"] = wanted
            break
    _write(found)
    logger.info("Model Chain: a Sopro voice was renamed")
    return lookup(voice_id)


def delete(voice_id: str) -> dict:
    """Remove a Sopro voice, transactionally. Section 28.

    The order is the design. The default moves first, so a deleted voice can
    never be what the next reply resolves to; the registry entry is removed
    next, so a failure while deleting files leaves a *missing voice* rather than
    a registry entry pointing at nothing; the worker's caches are invalidated;
    and the directory is removed last, after every path in it has been proved to
    resolve under the Sopro voice root (section 57).

    A directory that cannot be removed leaves a "voice unavailable, cleanup
    needed" state rather than a silent inconsistency -- which is what section 28
    asks for and what makes a half-failed delete recoverable.
    """
    entry = lookup(voice_id)
    if entry is None:
        raise SoproError("That Sopro voice does not exist.")
    identifier = _uuid_of(voice_id)

    if default_id() == entry["id"]:
        others = [item for item in entries() if item["id"] != entry["id"]]
        _remember(OPT_VOICE, others[0]["id"] if others else "")

    found = _read()
    found["voices"] = [item for item in (found.get("voices") or ())
                       if str(item.get("uuid")) != identifier]
    _write(found)

    try:
        import mc_voice_sopro_runtime as runtime

        runtime.refresh_catalog(catalog(), forget=[entry["id"]])
    except Exception:
        logger.debug("Model Chain: could not refresh the Sopro catalogue after a delete",
                     exc_info=True)

    root = paths.sopro_voice_root(identifier)
    if not paths.sopro_inside(root):
        raise SoproError("That voice's files are not where Voice Chat keeps them, so "
                         "nothing was deleted.")
    shutil.rmtree(root, ignore_errors=True)
    if root.exists():
        logger.warning("Model Chain: a deleted Sopro voice left files at %s", root)
    logger.info("Model Chain: a Sopro voice was deleted")
    return entry


def capacity() -> dict:
    """How many voices exist. Sopro has no bank and therefore no slot limit.

    Reported anyway because the surfaces ask both engines the same question, and
    a Sopro answer of "no limit" is a different and more useful sentence than a
    missing key.
    """
    return {"used": len(_records()), "total": None, "free": None}


def warnings() -> list:
    """What is wrong that a user can see and act on. Never repairs anything.

    Three states make a Sopro voice untrustworthy and each has a different
    remedy, so each gets its own sentence: a stale preparation with a retained
    recording can be rebuilt, a stale one without cannot, and a missing
    production asset is a voice that has to go.
    """
    found = []
    current = status().fingerprint
    for entry in entries():
        if not entry["compatible"]:
            if entry["has_source"]:
                found.append(f"The Sopro voice {entry['display_name']!r} was prepared by a "
                             f"different Sopro build. Rebuild it in Settings → Voice Chat; "
                             f"its recording is still here.")
            else:
                found.append(f"The Sopro voice {entry['display_name']!r} was prepared by a "
                             f"different Sopro build and its recording is gone, so it has "
                             f"to be created again.")
            continue
        if not _voice_file(_uuid_of(entry["id"]),
                           paths.SOPRO_PRODUCTION_FILENAME).is_file():
            found.append(f"The Sopro voice {entry['display_name']!r} is missing its prepared "
                         f"data and cannot be spoken. Delete it in Settings → Voice Chat.")
    if current and not entries():
        found.append("No Sopro voice has been created yet, so Voice Chat has nothing to "
                     "speak with. Use Clone voice to make one.")
    return found


def catalog() -> dict:
    """What the worker is told exists: id, directory and fingerprint.

    Only voices whose preparation matches the installed build. A stale voice is
    never guessed compatible (section 15), so it is simply absent from the
    worker's catalogue and speaking it produces "that voice is not in this
    worker's catalogue" rather than conditioning read against the wrong schema.
    """
    current = status().fingerprint
    found = {}
    for item in _records():
        fingerprint = str(item.get("fingerprint") or "")
        if current and fingerprint and fingerprint != current:
            continue
        root = paths.sopro_voice_root(str(item["uuid"]))
        if not (root / paths.SOPRO_PRODUCTION_FILENAME).is_file():
            continue
        found[f"{ENGINE}:clone:{item['uuid']}"] = {
            "root": str(root),
            "fingerprint": fingerprint,
            "language": str(item.get("language") or ""),
        }
    return found


def check_name(name: str) -> str:
    """A display name that is a name, and cannot be anything else.

    Nothing here builds a path or a command from a display name -- voice
    directories are server-generated UUIDs -- and this check exists anyway: a
    name is going into HTML, into a JSON file and onto a button, and "safe
    because nobody uses it dangerously" is a property that survives exactly one
    refactor.
    """
    wanted = str(name or "").strip()
    if not wanted:
        raise SoproError("Give the voice a name.")
    if len(wanted) > MAX_NAME_CHARS:
        raise SoproError(f"That name is longer than {MAX_NAME_CHARS} characters.")
    if not _NAME_OK.match(wanted):
        raise SoproError("A voice name can contain letters, numbers, spaces and - ' . ( ) &.")
    return wanted


def test_text() -> str:
    """The audition text, shared with Kokoro because it is one preference.

    Deliberately the same option: "what should a test voice say" is a property
    of the person, not of the engine, and two boxes holding two sentences would
    be a setting somebody has to keep in step for no reason. It is the one
    genuinely engine-neutral value in this module.
    """
    import mc_voice_registry as registry

    return registry.test_text()


def set_test_text(text: str) -> str:
    import mc_voice_registry as registry

    return registry.set_test_text(text)


# --------------------------------------------------------------------------- #
# Creating a voice
# --------------------------------------------------------------------------- #


def normalize_reference(data: bytes) -> tuple:
    """A recording as canonical mono 24 kHz PCM16, or a refusal that says why.

    Section 25's checks, in the order that produces the most useful sentence:
    is it a WAV at all, is it a size this build accepts, is it an encoding this
    build decodes, is it long enough to condition on, is it short enough, and is
    there actually a voice in it.

    Its own decoder rather than :mod:`mc_voice_clone`'s, and that is I-6 rather
    than duplication for its own sake: that module is Kokoro's Storytime path,
    with Storytime's three-to-a-hundred-and-twenty-second window and Storytime's
    error text. Sopro's window is five to twenty seconds because that is what
    Sopro V2 documents, and an engine whose validation lived in the other
    engine's module would be an engine that could not change its own rules.
    """
    if not data:
        raise SoproError("No recording was received.")
    if len(data) > MAX_REFERENCE_BYTES:
        raise SoproError("That recording is too large.")
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SoproError("That file is not a WAV recording.")

    offset, fmt, body = 12, None, b""
    while offset + 8 <= len(data):
        name = data[offset:offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        start = offset + 8
        if start + size > len(data):
            raise SoproError("That recording's header is malformed.")
        if name == b"fmt " and size >= 16:
            fmt = struct.unpack_from("<HHIIHH", data, start)
        elif name == b"data":
            body = data[start:start + size]
        offset = start + size + (size % 2)
    if fmt is None or not body:
        raise SoproError("That recording is not a complete WAV.")

    encoding, channels, rate, _bps, _align, bits = fmt
    if encoding != 1 or bits != 16:
        raise SoproError("Sopro accepts uncompressed 16-bit PCM WAV recordings.")
    if channels not in (1, 2):
        raise SoproError("Sopro accepts mono or stereo recordings.")
    if rate < 8000 or rate > 192000:
        raise SoproError("That recording's sample rate is not one Voice Chat can use.")

    import array

    samples = array.array("h")
    samples.frombytes(body[: len(body) - (len(body) % (2 * channels))])
    import sys as _sys

    if _sys.byteorder == "big":
        samples.byteswap()
    if channels == 2:
        samples = array.array("h", [(samples[index] + samples[index + 1]) // 2
                                    for index in range(0, len(samples) - 1, 2)])
    seconds = len(samples) / float(rate or 1)
    if seconds < MIN_REFERENCE_SECONDS:
        raise SoproError(f"That recording is {seconds:.1f} seconds long. Sopro clones from "
                         f"{MIN_REFERENCE_SECONDS:.0f} to {MAX_REFERENCE_SECONDS:.0f} "
                         f"seconds of one clear speaker.")
    if seconds > MAX_REFERENCE_SECONDS + 0.5:
        raise SoproError(f"That recording is {seconds:.0f} seconds long and Sopro clones "
                         f"from up to {MAX_REFERENCE_SECONDS:.0f} seconds. Record a shorter "
                         f"one.")
    peak = max((abs(value) for value in samples), default=0) / 32768.0
    if peak < MIN_PEAK:
        raise SoproError("That recording is silent. Check that the right microphone is "
                         "selected and that it is not muted.")

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


def create(display_name: str, wav: bytes, language: str = "") -> dict:
    """Make a Sopro voice from a reference recording. The whole transaction.

    Section 23's flow, and every step of it can fail without leaving anything
    behind:

        1  the name and the language are validated here;
        2  the recording is validated and normalized here;
        3  a server-generated UUID becomes a directory -- never the display
           name, never anything the browser sent (section 57);
        4  the normalized WAV is retained beside the voice, so a later
           compatible Sopro can rebuild without asking for a new recording;
        5  the worker prepares, writes both assets, reads them back *from the
           files it wrote*, and streams a production audition;
        6  and only then is a registry entry written.

    Step 5 is not ceremony (section 27). A preparation that returned without an
    exception proves nothing about whether the file it wrote can be read back
    after a restart, and reading it back is the only way to know.
    """
    name = check_name(display_name)
    wanted = str(language or "").strip().lower()
    if wanted not in {item for item, _label in LANGUAGES}:
        raise SoproError("That is not a language Sopro offers.")
    found = status()
    if not found.ready:
        raise SoproError("Sopro is not installed, so it cannot create a voice.")

    identifier = uuid.uuid4().hex
    normalized, seconds = normalize_reference(wav)
    root = paths.sopro_voice_root(identifier)
    root.mkdir(parents=True, exist_ok=True)
    made = None
    try:
        reference = root / paths.SOPRO_REFERENCE_FILENAME
        reference.write_bytes(normalized)

        import mc_voice_sopro_runtime as runtime

        made = runtime.prepare_voice(
            root=str(root), voice_id=f"{ENGINE}:clone:{identifier}",
            wav_bytes=normalized, seconds=min(seconds, MAX_REFERENCE_SECONDS),
            audition=test_text())
    except Exception:
        # Nothing is registered, so the only thing to undo is the directory.
        # Removed rather than left, because a directory with a WAV in it and no
        # registry entry is a recording of somebody that nothing will ever
        # delete.
        if paths.sopro_inside(root):
            shutil.rmtree(root, ignore_errors=True)
        raise

    record = {
        "uuid": identifier,
        "display_name": name,
        "language": wanted,
        "fingerprint": found.fingerprint,
        "source_seconds": round(float(seconds), 2),
        "sample_rate": int(made.get("sample_rate") or TARGET_RATE),
        "schema": REGISTRY_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stored = _read()
    stored.setdefault("voices", []).append(record)
    stored["schema"] = REGISTRY_SCHEMA
    _write(stored)

    try:
        import mc_voice_sopro_runtime as runtime

        runtime.refresh_catalog(catalog())
    except Exception:
        logger.debug("Model Chain: could not refresh the Sopro catalogue after a create",
                     exc_info=True)

    if not _setting(OPT_VOICE):
        # The first voice becomes the default, because an installation with one
        # voice and no default is one where Auto Speak silently does nothing.
        _remember(OPT_VOICE, f"{ENGINE}:clone:{identifier}")
    logger.info("Model Chain: a Sopro voice was created — %.1f s reference, audition in "
                "%d ms", seconds, int(made.get("audition_ms") or 0))
    return {"voice": lookup(f"{ENGINE}:clone:{identifier}"), "audio": made.get("audio") or b""}


def rebuild(voice_id: str) -> dict:
    """Prepare a stale voice again from its retained recording. Section 55.

    Transactional in the way that matters: the new assets are written into a
    staging directory, validated by a production audition, and only then does
    the metadata switch -- so a rebuild that fails leaves the voice exactly as
    stale as it was rather than leaving it broken. The only working copy is
    never overwritten first (section 60).
    """
    entry = lookup(voice_id)
    if entry is None:
        raise SoproError("That Sopro voice does not exist.")
    identifier = _uuid_of(voice_id)
    source = _voice_file(identifier, paths.SOPRO_REFERENCE_FILENAME)
    if not source.is_file():
        raise SoproError(f"{entry['display_name']} has no retained recording, so it has to "
                         f"be created again from a new one.")
    found = status()
    if not found.ready:
        raise SoproError("Sopro is not installed, so it cannot rebuild a voice.")

    root = paths.sopro_voice_root(identifier)
    staging = root.parent / f"{identifier}.rebuilding"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        import mc_voice_sopro_runtime as runtime

        made = runtime.prepare_voice(
            root=str(staging), voice_id=f"{ENGINE}:clone:{identifier}",
            wav_bytes=source.read_bytes(), audition=test_text())
        for name in (paths.SOPRO_PRODUCTION_FILENAME, paths.SOPRO_PRODUCTION_META,
                     paths.SOPRO_LAB_FILENAME):
            candidate = staging / name
            if candidate.is_file():
                os.replace(candidate, root / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    stored = _read()
    for item in stored.get("voices") or ():
        if str(item.get("uuid")) == identifier:
            item["fingerprint"] = found.fingerprint
            item["rebuilt_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            break
    _write(stored)
    try:
        import mc_voice_sopro_runtime as runtime

        runtime.refresh_catalog(catalog(), forget=[entry["id"]])
    except Exception:
        logger.debug("Model Chain: could not refresh the Sopro catalogue after a rebuild",
                     exc_info=True)
    logger.info("Model Chain: a Sopro voice was rebuilt for the current build")
    return {"voice": lookup(voice_id), "audio": made.get("audio") or b""}


# --------------------------------------------------------------------------- #
# Small shared things
# --------------------------------------------------------------------------- #


def _read_json(path: Path):
    try:
        found = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _digest(path: Path) -> str:
    found = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 256), b""):
            found.update(block)
    return found.hexdigest()


def _setting(name: str):
    try:
        from modules import shared

        return getattr(shared.opts, name, None)
    except Exception:
        return None


def _remember(name: str, value) -> None:
    try:
        from modules import shared

        shared.opts.set(name, value)
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: could not persist a Sopro setting", exc_info=True)
