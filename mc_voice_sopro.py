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
import math
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
OPT_THREADS = "model_chain_voice_sopro_intraop_threads"

PRECISIONS = ("full", "int8")
"""What the pinned Sopro release supports. ``from_pretrained`` takes
``quantization=None`` or ``"int8"`` and refuses int8 anywhere but the CPU, which
is the whole set."""

STEP_CHOICES = (2, 4, 8)
"""Acoustic solver steps. The reviewed V2 default is 2 and upstream documents
larger values as a quality/compute trade. Only a tested set is offered, because
a free integer here is a control that can make Sopro slower than real time."""

CHUNK_CHOICES = (32, 64, 128)

THREAD_CHOICES = (2, 4, 6, 8, 12, 16)
"""The intra-op counts the validation sweep measures, and the ones offerable.

A setting rather than an environment variable, which is where this started and
was the wrong place for it: Precision is already a user control that changes
compute, RAM and which warmed caches survive, and there is no principle that
makes thread count different in kind. What I-12 forbids is the *code* choosing
a number from a measured real-time factor. A person reading a table and picking
a row is exactly what "one measured, fixed policy" asks for -- and asking that
person to set MC_SOPRO_INTRAOP_THREADS on Windows is asking them not to bother.

The released default stays :data:`sopro_worker.worker.INTRAOP_THREADS`. This
list only says which values may be *offered*, and :func:`thread_choices` cuts it
to what the machine actually has, because offering eight threads on a four-core
laptop is offering a slower configuration with a faster-looking number.
"""
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

    # The effective count rather than the constant. When the benchmark override
    # is set these caps have to track it, or OpenMP sizes its pool at the
    # released four while Torch is asked for eight and the measurement belongs
    # to neither number.
    intraop, _interop, overridden = protocol.effective_policy(note=False)
    if not overridden:
        # The chosen setting, when there is no benchmark override in force. The
        # child reads the same value back out of this environment, so the pool
        # OpenMP sizes and the count Torch is given are one number.
        intraop = intraop_threads()
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        # The child reads this back and hands it to Torch, so the count OpenMP
        # sized its pool for and the count Torch was given are the same number
        # by construction rather than by two functions agreeing. When it equals
        # the released policy the child reports "released"; when it does not,
        # every line it writes says so.
        protocol.OVERRIDE_INTRAOP: str(intraop),
        "OMP_NUM_THREADS": str(intraop),
        "MKL_NUM_THREADS": str(intraop),
        "OPENBLAS_NUM_THREADS": str(intraop),
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


def intraop_threads() -> int:
    """The intra-op thread count in force: the setting, or the released policy.

    The environment override still wins over both, because that is what the
    sweep uses to measure a configuration this installation has not chosen.
    """
    from sopro_worker import worker as protocol

    asked, _interop, overridden = protocol.effective_policy(note=False)
    if overridden:
        return asked
    try:
        found = int(_setting(OPT_THREADS) or 0)
    except (TypeError, ValueError):
        found = 0
    return found if found in thread_choices() else protocol.INTRAOP_THREADS


def thread_choices() -> tuple:
    """The counts worth offering here. Never more than the machine has.

    Oversubscribing is not a neutral mistake -- more threads than cores is more
    time in barriers, so it would put a row in the dropdown that is reliably
    worse than the one above it. Clamping the *choices* is not auto-tuning: no
    measurement is consulted and nothing is selected, the list simply stops
    where the hardware does.
    """
    from sopro_worker import worker as protocol

    cores = os.cpu_count() or protocol.INTRAOP_THREADS
    found = tuple(value for value in THREAD_CHOICES if value <= cores)
    # The released policy is always offerable, whatever the machine reports --
    # a dropdown that could not express the shipped configuration would be a
    # dropdown somebody could not get back to.
    return found or (protocol.INTRAOP_THREADS,)


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
        # "INT8 (faster, CPU only)" is what this said, and on the first machine
        # it was measured on it was wrong by 40% in the other direction: full
        # precision fitted at 0.80 x real time and INT8 at 1.12, with the first
        # block 450 ms slower. Quantization shrinks the weights; whether it also
        # shrinks the *time* depends on whether this Torch build has int8
        # kernels for these shapes on this CPU, and on a small autoregressive
        # model the dequantize-requantize traffic can cost more than the
        # narrower multiply saves. So neither option claims a speed here. The
        # sweep is the only thing that can answer it for a given machine.
        "precisions": [{"id": "full", "label": "Full CPU precision",
                        "help": "Sopro exactly as it was released, and the one to compare "
                                "against. Larger in memory than INT8, and on some CPUs "
                                "also the faster of the two."},
                       {"id": "int8", "label": "INT8 (smaller, CPU only)",
                        "help": "Quantizes the autoregressive blocks, so the model is "
                                "lighter in RAM; the acoustic solver and the vocoder are "
                                "unchanged. Whether it is faster depends on the machine, "
                                "and it is measurably slower on some — the turn summary "
                                "in model_chain.log reports the real-time factor either "
                                "way. Saved voices stay valid; only the warmed streaming "
                                "caches are rebuilt."}],
        "steps": steps(),
        "step_choices": list(STEP_CHOICES),
        "chunk_frames": chunk_frames(),
        "chunk_choices": list(CHUNK_CHOICES),
        "threads": intraop_threads(),
        "thread_choices": list(thread_choices()),
        "released_threads": _released_threads(),
    }


def _released_threads() -> int:
    from sopro_worker import worker as protocol

    return int(protocol.INTRAOP_THREADS)


def set_engine_settings(precision_id: str = "", solver_steps=None, chunk=None,
                        threads=None) -> dict:
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
    if threads is not None:
        try:
            value = int(threads)
        except (TypeError, ValueError):
            value = 0
        # Against the offerable list rather than the constant, so a number a
        # browser invented cannot become a thread count.
        if value in thread_choices() and value != intraop_threads():
            _remember(OPT_THREADS, value)
            changed.append("CPU thread count")
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
    wanted = str(_read().get("default") or "").strip()
    if not wanted:
        # Migration, once: the value an older build stored in a Forge option.
        wanted = str(_legacy_default() or "").strip()
    if wanted and lookup(wanted) is not None:
        return wanted
    found = entries()
    return found[0]["id"] if found else ""


def _legacy_default() -> str:
    """What an older build left in ``model_chain_voice_sopro_voice_id``.

    Read only; nothing writes there again, which is what makes the settings
    page harmless.
    """
    try:
        from modules import shared

        return str(getattr(shared.opts, OPT_VOICE, "") or "")
    except Exception:
        return ""


def default_entry():
    found = default_id()
    return lookup(found) if found else None


def set_default(voice_id: str) -> dict:
    """Commit a new Sopro default. Stores only the stable id (section 28)."""
    entry = lookup(voice_id)
    if entry is None:
        raise SoproError("That Sopro voice does not exist.")
    found = _read()
    found["default"] = entry["id"]
    found["schema"] = REGISTRY_SCHEMA
    _write(found)
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
        found = _read()
        found["default"] = others[0]["id"] if others else ""
        _write(found)

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


WAVE_PCM, WAVE_FLOAT, WAVE_EXTENSIBLE = 0x0001, 0x0003, 0xFFFE

ENCODING_NAMES = {0x0002: "ADPCM", 0x0006: "A-law", 0x0007: "u-law",
                  0x0011: "IMA ADPCM", 0x0031: "GSM", 0x0055: "MP3-in-WAV"}


def _encoding_label(encoding: int, bits: int) -> str:
    """What a refusal should call the thing it is refusing.

    "Sopro accepts uncompressed 16-bit PCM" told somebody what was wanted and
    not what they had, which is the half of the sentence that would have let
    them fix it.
    """
    if encoding == WAVE_PCM:
        return f"{bits}-bit PCM"
    if encoding == WAVE_FLOAT:
        return f"{bits}-bit floating point"
    named = ENCODING_NAMES.get(int(encoding))
    return named or f"encoding 0x{int(encoding):04X}"


def _clipped(value: float) -> int:
    # NaN compares unequal to itself, and reaches here from a float WAV whose
    # producer wrote one. Silence is the only honest sample to put in its place.
    if value != value:
        return 0
    return max(-32768, min(32767, int(value * 32767.0)))


def _decode_pcm16(body: bytes, encoding: int, bits: int):
    """Interleaved samples as signed 16-bit, whatever the file stored them as.

    A reference recording arrives from whatever produced it, and that is rarely
    plain 16-bit PCM. An editor that cleans up a clip writes 24-bit or 32-bit
    float as a matter of course, and a great many writers wrap even ordinary
    16-bit PCM in ``WAVE_FORMAT_EXTENSIBLE``. Refusing all of those -- as this
    did, for one user, on a file that was fine -- was refusing the recording
    somebody actually has over a property the rest of this function is about to
    normalise away. It already downmixes and resamples; narrowing a sample is
    the same kind of work and no more of a judgement call.

    ``None`` for an encoding that is genuinely not decodable here, which after
    this is a compressed one: those need a codec, and a codec is a dependency
    this module does not have and should not grow.
    """
    import array
    import sys as _sys

    swap = _sys.byteorder == "big"
    if encoding == WAVE_PCM and bits == 8:
        # The one unsigned width the format has, with 128 as silence.
        return array.array("h", [(value - 128) << 8 for value in body])
    if encoding == WAVE_PCM and bits == 16:
        found = array.array("h")
        found.frombytes(body[: len(body) - (len(body) % 2)])
        if swap:
            found.byteswap()
        return found
    if encoding == WAVE_PCM and bits == 24:
        usable = len(body) - (len(body) % 3)
        # The top two bytes of each little-endian triple, which is the 16-bit
        # sample: truncation rather than rounding, because a reference is about
        # to be peak-normalised and a half-LSB is not audible in it.
        return array.array("h", [
            int.from_bytes(body[index + 1:index + 3], "little", signed=True)
            for index in range(0, usable, 3)])
    if encoding == WAVE_PCM and bits == 32:
        found = array.array("i")
        found.frombytes(body[: len(body) - (len(body) % 4)])
        if swap:
            found.byteswap()
        return array.array("h", [value >> 16 for value in found])
    if encoding == WAVE_FLOAT and bits in (32, 64):
        found = array.array("f" if bits == 32 else "d")
        width = bits // 8
        found.frombytes(body[: len(body) - (len(body) % width)])
        if swap:
            found.byteswap()
        # Clamped rather than scaled to the peak: a float WAV is allowed to
        # exceed unity and a reference that did would otherwise wrap around.
        return array.array("h", [_clipped(value) for value in found])
    return None


RESAMPLE_ZEROS = 24
RESAMPLE_ROLLOFF = 0.945
RESAMPLE_BETA = 8.6
RESAMPLE_BLOCK = 32768
"""The anti-aliasing filter, and why a resampler needs one at all.

What was here before was linear interpolation with no filter, and the damage it
did is not subtle. Measured on this build: a 15 kHz tone in a 48 kHz recording
came back at 9 kHz -- folded into the middle of the speech band -- at **0.0 dB**,
which is to say at full amplitude, entirely unattenuated. Everything a recorder
captured between 12 and 24 kHz was mirrored back down on top of the voice: mic
self-noise, room hiss, and above all sibilance, which is exactly the band where
"s" and "sh" and "t" live. Almost every recording anybody clones from is 44.1 or
48 kHz, so almost every clone was conditioned on a reference with a mirror image
of its own top end laid over it.

A cloned voice inherits that permanently, because the conditioning is computed
from these samples. It is the difference between a clone that sounds like the
speaker and one that sounds like the speaker through a broken microphone.

The replacement is an ordinary windowed-sinc: a Kaiser window at beta 8.6 over
24 zero crossings, with the cutoff pulled to 94.5% of the new Nyquist to leave a
transition band. Same measurement, same tone: **-102 dB**. A 1 kHz and 3 kHz
speech-band pair comes through at 0.00 dB, so nothing that should survive is
touched.

Numbers, not adjectives: ``tests/test_voice_sopro.py`` measures both.
"""

_KAISER_TABLE = None
_KAISER_POINTS = 4097


def _bessel_i0(value: float) -> float:
    """The modified Bessel function of the first kind, order zero, by series.

    Written out rather than imported because this module runs in the WebUI
    process, where SciPy is not a dependency and NumPy is only an optional
    accelerator -- and the window has to be identical on both paths or the two
    would resample differently.
    """
    total, term, step = 1.0, 1.0, 1
    while step < 200:
        term *= (value / (2.0 * step)) ** 2
        total += term
        if term < 1e-12 * total:
            break
        step += 1
    return total


def _kaiser_table():
    """The Kaiser window sampled on [-1, 1], built once.

    A table rather than a call per tap: the window is evaluated tens of millions
    of times in a twenty-second reference, and two Bessel series per tap is the
    difference between a resample that is imperceptible and one that is a
    visible pause in the interface.
    """
    global _KAISER_TABLE

    if _KAISER_TABLE is None:
        scale = _bessel_i0(RESAMPLE_BETA)
        _KAISER_TABLE = [
            _bessel_i0(RESAMPLE_BETA * math.sqrt(max(0.0, 1.0 - position * position)))
            / scale
            for position in (
                -1.0 + 2.0 * index / (_KAISER_POINTS - 1)
                for index in range(_KAISER_POINTS))
        ]
    return _KAISER_TABLE


def _resample(samples, rate: int, target: int):
    """Band-limited resampling to ``target``, as PCM16 in an ``array``.

    NumPy when it is importable, which in Forge it always is; the pure-Python
    path exists so that this module never *requires* it, and narrows the filter
    because the same 24 zero crossings would take most of a minute in a loop.
    Even the narrow one is a different universe from no filter at all.
    """
    import array as _array

    if rate == target or not len(samples):
        return samples
    count = int(len(samples) * target / float(rate))
    if count <= 0:
        return _array.array("h")
    try:
        import numpy
    except Exception:  # noqa: BLE001 - an optional accelerator, never required
        return _resample_slowly(samples, rate, target, count)
    return _resample_with(numpy, samples, rate, target, count)


def _filter_shape(rate: int, target: int, zeros: int):
    """Cutoff and half-width for a source-rate filter. Shared by both paths.

    The half-width widens by the decimation factor because the filter has to be
    band-limited to the *new* Nyquist while being applied at the *old* rate: a
    fixed number of taps would be a fixed fraction of the source spectrum, which
    is the wrong thing to hold constant.
    """
    down = max(1.0, rate / float(target))
    return 0.5 * RESAMPLE_ROLLOFF / down, int(math.ceil(zeros * down))


def _resample_with(numpy, samples, rate: int, target: int, count: int):
    import array as _array

    source = numpy.asarray(samples, dtype=numpy.float64)
    cutoff, half = _filter_shape(rate, target, RESAMPLE_ZEROS)
    taps = numpy.arange(-half, half + 1, dtype=numpy.float64)
    window = numpy.asarray(_kaiser_table(), dtype=numpy.float64)
    positions = numpy.linspace(-1.0, 1.0, _KAISER_POINTS)
    padded = numpy.concatenate([numpy.zeros(half), source, numpy.zeros(half + 2)])
    out = numpy.empty(count, dtype=numpy.float64)
    step = rate / float(target)
    # Blocked, because the tap matrix is one row per output sample: twenty
    # seconds at 48 kHz against a 97-tap filter is 93 million doubles in one
    # allocation, and this runs in the WebUI's own process.
    for begin in range(0, count, RESAMPLE_BLOCK):
        end = min(count, begin + RESAMPLE_BLOCK)
        centre = numpy.arange(begin, end, dtype=numpy.float64) * step
        base = numpy.floor(centre).astype(numpy.int64)
        delta = taps[None, :] - (centre - base)[:, None]
        argument = 2.0 * cutoff * delta
        sinc = numpy.where(numpy.abs(argument) < 1e-9, 1.0,
                           numpy.sin(numpy.pi * argument)
                           / (numpy.pi * argument + 1e-30))
        weights = sinc * numpy.interp(numpy.clip(delta / half, -1.0, 1.0),
                                      positions, window)
        # Normalised per output sample so the passband is flat and DC is exact.
        weights /= weights.sum(axis=1, keepdims=True)
        index = base[:, None] + taps[None, :].astype(numpy.int64) + half
        out[begin:end] = (padded[index] * weights).sum(axis=1)
    clipped = numpy.clip(numpy.rint(out), -32768, 32767).astype(numpy.int16)
    return _array.array("h", clipped.tobytes())


def _resample_slowly(samples, rate: int, target: int, count: int):
    """The same filter in a loop, narrowed so it finishes this decade."""
    import array as _array

    cutoff, half = _filter_shape(rate, target, 8)
    window = _kaiser_table()
    last = len(samples) - 1
    out = _array.array("h", bytes(2 * count))
    step = rate / float(target)
    for index in range(count):
        centre = index * step
        base = int(math.floor(centre))
        total, weight_sum = 0.0, 0.0
        for tap in range(-half, half + 1):
            delta = tap - (centre - base)
            argument = 2.0 * cutoff * delta
            if abs(argument) < 1e-9:
                sinc = 1.0
            else:
                sinc = math.sin(math.pi * argument) / (math.pi * argument)
            position = delta / half
            if position < -1.0:
                position = -1.0
            elif position > 1.0:
                position = 1.0
            shaped = window[int((position + 1.0) * 0.5 * (_KAISER_POINTS - 1))]
            weight = sinc * shaped
            at = base + tap
            if 0 <= at <= last:
                total += samples[at] * weight
            weight_sum += weight
        value = int(round(total / weight_sum)) if weight_sum else 0
        out[index] = max(-32768, min(32767, value))
    return out


def normalize_reference(data: bytes) -> tuple:
    """A recording as canonical mono 24 kHz PCM16, or a refusal that says why.

    Section 25's checks, in the order that produces the most useful sentence:
    is it a WAV at all, is it a size this build accepts, is it an encoding this
    build decodes, is it long enough to condition on, is it short enough, and is
    there actually a voice in it.

    What it decodes is deliberately wider than what Sopro wants, because the two
    are different questions. Sopro wants mono 24 kHz PCM16; a person has
    whatever their recorder or editor produced, which is routinely 24-bit,
    32-bit float, or 16-bit PCM wrapped in ``WAVE_FORMAT_EXTENSIBLE``. This
    already downmixes and resamples, so narrowing a sample is the same kind of
    work -- and refusing a good recording for its container was, for one user,
    the whole of "I could not create a voice".

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
    fmt_start, fmt_size = 0, 0
    while offset + 8 <= len(data):
        name = data[offset:offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        start = offset + 8
        if start + size > len(data):
            raise SoproError("That recording's header is malformed.")
        if name == b"fmt " and size >= 16:
            fmt = struct.unpack_from("<HHIIHH", data, start)
            fmt_start, fmt_size = start, size
        elif name == b"data":
            body = data[start:start + size]
        offset = start + size + (size % 2)
    if fmt is None or not body:
        raise SoproError("That recording is not a complete WAV.")

    encoding, channels, rate, _bps, _align, bits = fmt
    if encoding == WAVE_EXTENSIBLE:
        # The wrapper a writer reaches for as soon as a file has more than two
        # channels or more than sixteen bits -- and, in practice, for plenty of
        # ordinary 16-bit stereo too. The real format tag is the first two bytes
        # of the SubFormat GUID, and reading it is the difference between
        # accepting a perfectly good recording and refusing it for its wrapper.
        if fmt_size < 40:
            raise SoproError("That recording's header is malformed.")
        (encoding,) = struct.unpack_from("<H", data, fmt_start + 24)
    if channels not in (1, 2):
        raise SoproError("Sopro accepts mono or stereo recordings.")
    if rate < 8000 or rate > 192000:
        raise SoproError("That recording's sample rate is not one Voice Chat can use.")

    import array

    samples = _decode_pcm16(body, int(encoding), int(bits))
    if samples is None:
        raise SoproError(
            f"That recording is {_encoding_label(int(encoding), int(bits))}, which Sopro "
            f"cannot read here. Save it as uncompressed PCM or floating-point WAV.")
    if not len(samples):
        raise SoproError("That recording is not a complete WAV.")
    import sys as _sys

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
        samples = _resample(samples, int(rate), TARGET_RATE)
    if _sys.byteorder == "big":
        samples.byteswap()

    raw = samples.tobytes()
    header = (b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, TARGET_RATE, TARGET_RATE * 2, 2, 16)
              + b"data" + struct.pack("<I", len(raw)))
    return header + raw, len(raw) / float(TARGET_RATE * 2)


STARTER_VOICES = (
    ("af_heart", "Heart (starter)"),
    ("am_michael", "Michael (starter)"),
    ("bf_emma", "Emma (starter)"),
    ("bm_george", "George (starter)"),
)
"""Kokoro speakers to seed Sopro from, and what the seeded voices are called.

Two American and two British, two of each timbre, so the four are actually
different from each other rather than four takes of one voice.
"""

STARTER_TEXT = (
    "This is a reference recording, made on this computer, for a starter voice. "
    "It runs for about fifteen seconds so that there is enough of it to work "
    "from. The quick brown fox jumps over the lazy dog, and the judge asked "
    "whether five or six of them had really done so. Numbers like one, seven "
    "and thirty come out here too, along with a question, and a pause."
)
"""Roughly fifteen seconds of read speech, which is what Sopro conditions on.

Phonetically spread on purpose -- a reference made of one flat sentence gives
the model one intonation to copy. It says what it is, so a starter voice that
somebody plays back is not a sentence they have to wonder about.
"""


def starter_names() -> list:
    """The starter voices that do not exist yet, in the order they are made."""
    made = {str(entry.get("display_name") or "").casefold() for entry in entries()}
    return [(sid, name) for sid, name in STARTER_VOICES
            if name.casefold() not in made]


def create_starter_voice() -> dict:
    """Make the next starter voice, by cloning one of Kokoro's own speakers.

    Sopro has no speaker bank: every voice it has is conditioned on a reference
    recording, so a fresh installation has nothing to say anything with and the
    only way in was to record yourself. That is a real wall in front of somebody
    who just wants to hear the engine work, and it is unnecessary, because a
    perfectly good reference recording can be made on the spot by the engine
    that is already installed.

    So the reference is Kokoro reading :data:`STARTER_TEXT`. This is the one
    place the two engines touch, and it is a *creation* step rather than a
    dependency: what it leaves behind is an ordinary Sopro voice with its own
    retained reference, which rebuilds, renames, deletes and speaks like any
    other, and which keeps working if Kokoro is removed afterwards.

    It is also the one clone where consent is not a question anybody has to
    think about. A Kokoro speaker is synthetic and Apache-2.0; no person's
    identity is being copied, which is exactly what makes it the right thing to
    put in front of the record button rather than behind it.

    One per call. Four prepares in one request is a request that takes minutes
    and a browser that gives up halfway; the caller asks again for the next one
    and gets to say which voice it is on.
    """
    import mc_voice_models as models
    import mc_voice_registry as registry
    import mc_voice_runtime as kokoro

    found = status()
    if not found.ready:
        raise SoproError("Sopro is not installed, so it cannot make a starter voice.")
    if not models.status().tts_ready:
        raise SoproError(
            "Starter voices are read by Kokoro, which is not installed. Install the "
            "text-to-speech model in Settings, or record a voice of your own.")

    wanted = starter_names()
    if not wanted:
        return {"created": "", "remaining": 0}
    sid_name, display = wanted[0]
    try:
        sid, _entry = registry.resolve(f"official:{sid_name}")
    except Exception as exc:
        raise SoproError(f"Kokoro has no speaker called {sid_name}, so that starter voice "
                         f"could not be made.") from None
    logger.info("Model Chain: making the Sopro starter voice %r from Kokoro's %s",
                display, sid_name)
    try:
        reference = kokoro.synthesize(STARTER_TEXT, int(sid))
    except Exception as exc:
        raise SoproError(f"Kokoro could not read the reference for that starter voice "
                         f"({exc}).") from None
    made = create(display, reference, "en")
    return {"created": display, "remaining": len(starter_names()), "voice": made["voice"]}


_preview_lock = threading.Lock()
_preview = {}
"""The one voice that has been built but not yet kept.

Section 23 already had the seam this needs and was not using it: everything up
to the registry write leaves nothing anybody can see, which is why the failure
path here has always been a single ``rmtree``. A voice with a directory, a
retained recording and prepared conditioning, but no registry entry, is not a
half-saved voice -- it is a voice that does not exist yet and costs one
directory to stop existing.

So creating and keeping are now two steps with a person in between. The
audition somebody hears is the one this exact preparation produced, played back
from the same bytes, rather than a re-synthesis from a voice that has already
been written down -- which is the only version of "preview before you save"
that is actually a preview.

One at a time, deliberately. A second preview discards the first: two
unregistered directories is two things to reason about and nobody asked for the
second one.
"""


def preview_state() -> dict:
    """What is pending, without the audio. Safe to log and to poll."""
    with _preview_lock:
        if not _preview:
            return {"pending": False}
        return {"pending": True, "name": _preview.get("name", ""),
                "seconds": _preview.get("seconds", 0.0),
                "audition_ms": _preview.get("audition_ms", 0)}


def discard_preview(token: str = "") -> bool:
    """Throw away the pending preview, and the directory it built.

    An empty token discards whatever is pending, which is what the shutdown and
    supersede paths want; a non-empty one has to match, so a stale browser tab
    cannot delete the preview a newer one is looking at.
    """
    with _preview_lock:
        if not _preview:
            return False
        if token and token != _preview.get("token"):
            return False
        identifier = _preview.get("uuid") or ""
        name = _preview.get("name") or ""
        _preview.clear()
    if identifier:
        root = paths.sopro_voice_root(identifier)
        # The same guard the failure path uses: never remove a directory this
        # module did not build inside its own root.
        if paths.sopro_inside(root):
            shutil.rmtree(root, ignore_errors=True)
        logger.info("Model Chain: a Sopro voice preview was discarded — %s", name or "?")
    return True


def prepare_preview(display_name: str, wav: bytes, language: str = "") -> dict:
    """Build a voice and audition it, without writing it down. Steps 1 to 5.

    Section 23's flow, and every step of it can fail without leaving anything
    behind:

        1  the name and the language are validated here;
        2  the recording is validated and normalized here;
        3  a server-generated UUID becomes a directory -- never the display
           name, never anything the browser sent (section 57);
        4  the normalized WAV is retained beside the voice, so a later
           compatible Sopro can rebuild without asking for a new recording;
        5  the worker prepares, writes both assets, reads them back *from the
           files it wrote*, and streams a production audition.

    Step 5 is not ceremony (section 27). A preparation that returned without an
    exception proves nothing about whether the file it wrote can be read back
    after a restart, and reading it back is the only way to know.

    :func:`save_preview` is step 6.
    """
    name = check_name(display_name)
    wanted = str(language or "").strip().lower()
    if wanted not in {item for item, _label in LANGUAGES}:
        raise SoproError("That is not a language Sopro offers.")
    found = status()
    if not found.ready:
        raise SoproError("Sopro is not installed, so it cannot create a voice.")

    # Before the work, not after: a preview that failed halfway would otherwise
    # leave the previous one pending and the panel showing a Save button for a
    # voice the user has already replaced.
    discard_preview()

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

    token = uuid.uuid4().hex
    with _preview_lock:
        _preview.update({
            "token": token,
            "uuid": identifier,
            "name": name,
            "language": wanted,
            "fingerprint": found.fingerprint,
            "seconds": round(float(seconds), 2),
            "sample_rate": int(made.get("sample_rate") or TARGET_RATE),
            "audition_ms": int(made.get("audition_ms") or 0),
        })
    logger.info("Model Chain: a Sopro voice was prepared for preview — %.1f s reference, "
                "audition in %d ms; it is not saved yet",
                seconds, int(made.get("audition_ms") or 0))
    return {"token": token, "name": name, "seconds": round(float(seconds), 2),
            "audio": made.get("audio") or b""}


def save_preview(token: str) -> dict:
    """Keep the pending preview. Step 6, and the only step that writes anything
    a user can afterwards see.

    The token has to match. Without it a browser that had been sitting on an
    old panel could save a voice the user had already replaced with another
    preview, which is the one way this split could produce a voice nobody chose.
    """
    with _preview_lock:
        if not _preview:
            raise SoproError("There is no voice waiting to be saved. Create one first.")
        if str(token or "") != _preview.get("token"):
            raise SoproError("That preview is no longer the one waiting to be saved. "
                             "Create the voice again.")
        pending = dict(_preview)
        _preview.clear()

    identifier = pending["uuid"]
    record = {
        "uuid": identifier,
        "display_name": pending["name"],
        "language": pending["language"],
        "fingerprint": pending["fingerprint"],
        "source_seconds": pending["seconds"],
        "sample_rate": pending["sample_rate"],
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

    if not str(_read().get("default") or "").strip():
        # The first voice becomes the default, because an installation with one
        # voice and no default is one where Auto Speak silently does nothing.
        stored = _read()
        stored["default"] = f"{ENGINE}:clone:{identifier}"
        _write(stored)
    logger.info("Model Chain: a Sopro voice was saved — %s, %.1f s reference",
                pending["name"], pending["seconds"])
    return {"voice": lookup(f"{ENGINE}:clone:{identifier}")}


def create(display_name: str, wav: bytes, language: str = "") -> dict:
    """Prepare and keep in one call, for callers with nobody to ask.

    Starter voices come through here: they are made from Kokoro reading a fixed
    script, so there is no recording anybody chose and nothing to audition
    before deciding. Everything a person supplies goes through
    :func:`prepare_preview` and :func:`save_preview` instead.
    """
    made = prepare_preview(display_name, wav, language)
    kept = save_preview(made["token"])
    return {"voice": kept["voice"], "audio": made.get("audio") or b""}


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


def _settings_read() -> dict:
    try:
        found = json.loads(paths.sopro_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _settings_write(found: dict) -> None:
    """Replace Sopro's settings file atomically, like every other file here."""
    path = paths.sopro_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.new")
    staging.write_text(json.dumps(found, indent=2), encoding="utf-8")
    os.replace(staging, path)


def _setting(name: str):
    """One global Sopro setting, out of Sopro's own file.

    These used to be host options, and every one of them had the same defect the
    default voice had: an option is a *component* on the settings page as well as
    a stored value, and "Apply settings" writes every component on that page back
    into the store. The page's copy is stamped when the page is built, so it
    knows nothing about a slider moved in the delivery panel since -- and putting
    the old value back is exactly what it did.

    They were also, for the same reason, two controls for one value: one in the
    panel that is meant to be used and one further down the settings page, each
    able to overwrite the other. There is one now.

    An option is still read when the file has nothing to say, so an installation
    configured under an older build keeps what it chose.
    """
    found = _settings_read()
    if name in found:
        return found[name]
    try:
        from modules import shared

        return getattr(shared.opts, name, None)
    except Exception:
        return None


def _remember(name: str, value) -> None:
    found = _settings_read()
    found[name] = value
    try:
        _settings_write(found)
    except OSError:
        logger.debug("Model Chain: could not persist a Sopro setting", exc_info=True)
