"""PocketTTS: what is installed, which voices exist, and the clone transaction.

The third text-to-speech engine's product side. Everything a person can see,
choose, name or delete lives here; everything that needs a tensor lives in
:mod:`pocket_worker.worker`, behind :mod:`mc_voice_pocket_runtime`, in a process
of its own with a PyTorch closure of its own (I-PKT-6).

Why this is not a copy of :mod:`mc_voice_sopro`
-----------------------------------------------
It has the same shape on purpose -- a manifest that is the trust root, a staged
install that promotes by rename, a registry of voices with server-minted UUIDs,
and a preview transaction where the audition somebody hears is the one their
recording actually produced. Those are the shapes Sopro proved and there is no
reason to invent different ones.

What differs is not cosmetic:

    Official voices exist here.      Sopro has no speaker bank at all; Pocket
                                     ships precomputed safetensors voice states
                                     with reviewed attribution, installed
                                     locally and never fetched at speech time
                                     (I-PKT-22, section 10).

    Cloning is gated separately.     The public repository has the model and the
                                     official states; the cloning-capable
                                     weights are a *different* upstream
                                     repository behind an access gate. So
                                     "Pocket can speak" and "Pocket can clone"
                                     are two states rather than one boolean, and
                                     the panel says which it is in (section 23.3,
                                     section 24).

    A derived state is per model.    A custom voice's stable id is product
                                     identity and is model-independent; the
                                     safetensors state it speaks from is model-
                                     specific, and lives at
                                     ``states/<fingerprint>.safetensors``. A
                                     model switch must not overwrite the only
                                     state that worked, and must not load one
                                     into a model it does not fit (I-PKT-18,
                                     section 39).

    Engine settings are a file.      Precision, sampler steps and the model id
                                     are engine-global and are persisted here
                                     rather than as Forge options, because an
                                     option is a component on the settings page
                                     as well as a stored value and "Apply
                                     settings" writes the page's build-time copy
                                     back over whatever the panel just set
                                     (I-PKT-19, section 28).

    There is no thread control.      Released PocketTTS 3.0.2 calls
                                     ``torch.set_num_threads(1)`` and gets its
                                     parallelism from its own generation and
                                     decoder threads. Offering a slider that set
                                     ``OMP_NUM_THREADS=8`` and calling it "8
                                     Pocket threads" would be dishonest, so
                                     there is a sentence instead (section 16.4,
                                     section 35).

    There is no Voice Lab and no     Sopro's Lab explores eight learned numbers
    starter voice.                   its encoder produces. Pocket's noise clamp
                                     and EOS threshold are not style axes, and
                                     inventing an equivalent would be inventing
                                     a product claim (section 29.5). Starter
                                     voices exist because Sopro starts with an
                                     empty library; Pocket starts with official
                                     voices, so the wall they were built for is
                                     not there.

Credentials, and where they are not
-----------------------------------
The cloning-capable weights are gated. If a token is available to *this
process*, the installer uses it to ask the publisher and to fetch, and
:mod:`mc_voice_models` removes it before following the signed delivery redirect.
It is never written to a settings file, a registry, a status payload, a log, a
subprocess command line or the worker's environment (I-PKT-21, section 23.4).
Nothing in this module stores one, and there is no route that accepts one.

What is provisional, and says so
--------------------------------
The manifest ships with its runtime closure written down and not yet resolved:
no filenames, no byte counts, no digests. That is a state rather than an
oversight -- section 23.1 requires exact pins before release and
``tools/pin_pocket_models.py`` is how they are produced -- and until it is done
the managed install refuses with a sentence naming the tool, while the
install-from-a-folder path still works and records the digests of whatever it
was given.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import mc_voice_engines as engines
import mc_voice_models as models
import mc_voice_paths as paths
import mc_voice_reference as reference

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

ENGINE = engines.POCKET

LABEL = "PocketTTS"

FAMILY = "pocket-tts"
"""What the handshake and the telemetry call this backend's model family."""

BACKEND = "pocket-tts-native"
"""Which implementation of Pocket this build speaks to.

Named rather than assumed, because there is more than one: upstream's native
PyTorch package is what this integration targets, and a sherpa-onnx export of
the same model family exists and is behaviourally different. A handshake that
did not say which one answered would be a handshake that could not refuse the
wrong one (section 6).
"""

SCHEMA = 1
"""The manifest and installed-marker layout this build understands."""

REGISTRY_SCHEMA = 1
"""The voice registry layout. Bumped only for a change that needs a migration."""

STATE_SCHEMA = 1
"""The exported voice-state layout, and part of the model fingerprint.

A saved safetensors state is only meaningful against the model that produced it
*and* against the shape this build writes it in, so both are in the fingerprint
(I-PKT-18).
"""

OPT_VOICE = "model_chain_voice_pocket_voice_id"
"""The legacy home of the Pocket default voice.

Read when the registry has nothing, never written. It exists so that an
installation configured under an earlier build does not lose its default when
the registry becomes the authority for it (I-PKT-19).
"""

SETTING_PRECISION = "pocket_precision"
SETTING_STEPS = "pocket_sampler_steps"
SETTING_MODEL = "pocket_model_id"
"""Keys in Pocket's own ``settings.json``. Not Forge options: see the module
docstring, and :func:`_setting` for why that distinction was worth making."""

PRECISIONS = ("full", "int8")
"""What the model may be loaded at.

Upstream 3.0.2 supports ``quantize=True`` for dynamic INT8 at load time and its
own documentation reports lower memory and faster x86 inference. Model Chain
does not repeat that claim: GATE P-5 measures both on the release machine, and
until it has, neither choice below says which is faster (I-PKT-26, section 16.1).
"""

PRECISION_DEFAULT = "full"
"""Where an installation starts, and deliberately the unquantized one.

Full precision is the model as released and the thing to compare against. A
default chosen for a speed nobody measured would be a default that had to be
defended later.
"""

STEP_CHOICES = (1, 2, 3, 5)
"""Sampler decode steps this build offers.

Upstream's default is 1 and it documents extra steps as a quality/compute
trade. Four tested values rather than an integer box, because an arbitrary
number is a number nobody has listened to (section 16.2).
"""

STEP_DEFAULT = 1

STEP_LABELS = {1: "Fast (default)", 2: "Balanced", 3: "Quality", 5: "Maximum"}
"""What each step count is called on screen. The number stays visible beside it:
a label is a shorthand and this setting is a compute trade, so hiding the value
would hide the thing being traded."""

MIN_REFERENCE_SECONDS = 5.0
IDEAL_REFERENCE_SECONDS = 10.0
MAX_REFERENCE_SECONDS = 15.0
"""Pocket's reference window, and deliberately below upstream's ceiling.

Upstream's voice export truncates at thirty seconds. Fifteen is below that on
purpose: a longer reference is a longer preparation and there is no evidence yet
that it is a better voice. GATE P-CLONE-1 compares five, ten, fifteen and thirty
seconds across several speakers, and :func:`clone_hints` is why the browser can
be told the answer without a release changing any JavaScript (section 26.1).
"""

MAX_REFERENCE_BYTES = 32 * 1024 * 1024
TARGET_RATE = 24000
MIN_PEAK = 0.01

MAX_NAME_CHARS = 48
MAX_TEST_CHARS = 400
DEFAULT_TEST_TEXT = "This is a test of the PocketTTS voice."

_NAME_OK = re.compile(r"^[\w][\w '\-.()&]*$", re.UNICODE)
"""What a voice may be called. A display name, never a path component: the
directory a voice lives in is a UUID this process minted (section 45)."""

_lock = threading.RLock()
_manifest_cache = None


class PocketError(RuntimeError):
    """A Pocket operation that was refused or could not be completed.

    One class, because every ordinary Pocket refusal -- not installed, no
    cloning access, a reference too short, a state prepared by a different
    build -- is a state a user can act on rather than a fault. The facade reads
    this list through :func:`refusals` so that those states are logged as states
    (section 8).
    """


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OfficialVoice:
    """One precomputed voice state, as the manifest declares it.

    A manifest entry rather than a discovery: section 10 is explicit that the
    official list must not be scraped from whatever upstream's voice dictionary
    currently holds. Adding a voice is a reviewed change to this repository,
    with its source, licence and attribution written down beside it.
    """

    identifier: str
    display_name: str
    language: str
    accent: str
    license: str
    attribution: str
    source: str
    artifact: "models.Artifact | None"


@dataclass(frozen=True)
class Bundle:
    """One installable Pocket model, as the manifest declares it."""

    identifier: str
    label: str
    language: str
    public_repo: str
    cloning_repo: str
    revision: str
    cloning_revision: str
    voice_revision: str
    license: str
    attribution: str
    summary: str
    notes: str
    about_bytes: int
    ram_bytes: int
    sample_rate: int
    recommended_temperature: "float | None"
    config: dict
    artifacts: tuple
    required_paths: tuple
    cloning_artifacts: tuple
    cloning_required_paths: tuple
    voices: tuple

    @property
    def download_bytes(self) -> int:
        total = sum(int(item.size or 0) for item in self.artifacts)
        return total or int(self.about_bytes or 0)

    @property
    def voice_artifacts(self) -> tuple:
        return tuple(voice.artifact for voice in self.voices if voice.artifact is not None)


def manifest(refresh: bool = False) -> dict:
    """The parsed, validated Pocket manifest. Read once and kept.

    Every failure below is a broken *extension*, not a broken installation, so
    the message says so: a user cannot fix a malformed trust root by pressing
    Install again.
    """
    global _manifest_cache

    with _lock:
        if _manifest_cache is not None and not refresh:
            return _manifest_cache
        path = paths.pocket_manifest_path()
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PocketError(f"The PocketTTS manifest at {path} could not be read ({exc}). "
                              f"This is a problem with the extension rather than with your "
                              f"installation.") from None
        if not isinstance(found, dict) or int(found.get("schema") or 0) != SCHEMA:
            raise PocketError("The PocketTTS manifest is not in a layout this build "
                              "understands.")
        runtime = found.get("runtime")
        if not isinstance(runtime, dict) or not runtime.get("platforms"):
            raise PocketError("The PocketTTS manifest declares no runtime.")
        _manifest_cache = found
        return found


def _manifest_cache_clear() -> None:
    global _manifest_cache

    with _lock:
        _manifest_cache = None


def _artifact(entry: dict, authorized: bool = False) -> models.Artifact:
    """One manifest artifact as the shared downloader's own dataclass.

    Built rather than re-implemented, so an artifact this module fetches goes
    through exactly the checks -- pinned hash, publisher digest, byte ceiling,
    ``.part`` rename, credential dropped at a cross-host hop -- that every other
    artifact in this feature goes through.
    """
    return models.Artifact(
        filename=str(entry.get("filename") or ""),
        local_name=str(entry.get("local_name") or entry.get("filename") or ""),
        url=str(entry.get("url") or ""),
        size=(int(entry["bytes"]) if entry.get("bytes") else None),
        sha256=(str(entry["sha256"]).casefold() if entry.get("sha256") else None),
        authorized=bool(authorized or entry.get("authorized")))


def _official_voice(entry: dict) -> OfficialVoice:
    artifact = entry.get("artifact")
    return OfficialVoice(
        identifier=str(entry.get("id") or ""),
        display_name=str(entry.get("display_name") or entry.get("id") or "Voice"),
        language=str(entry.get("language") or ""),
        accent=str(entry.get("accent") or ""),
        license=str(entry.get("license") or ""),
        attribution=str(entry.get("attribution") or ""),
        source=str(entry.get("source") or ""),
        artifact=_artifact(artifact) if isinstance(artifact, dict) else None)


def model_ids() -> tuple:
    """Every Pocket model this build knows, in manifest order.

    A tuple rather than a single name from day one. V1 release scope may expose
    only English, and the architecture must not encode "Pocket means English" in
    stable storage (section 16.3, section 38).
    """
    try:
        return tuple((manifest().get("models") or {}).keys())
    except PocketError:
        return ()


def bundle(identifier: str = "") -> Bundle:
    """The Pocket model this build installs, by id or by the selected one."""
    found = manifest()
    wanted = str(identifier or model_id() or (found.get("defaults") or {}).get("model") or "")
    entry = (found.get("models") or {}).get(wanted)
    if not isinstance(entry, dict):
        raise PocketError(f"{wanted!r} is not a PocketTTS model this build knows.")
    return Bundle(
        identifier=wanted,
        label=str(entry.get("label") or wanted),
        language=str(entry.get("language") or ""),
        public_repo=str(entry.get("public_repo") or ""),
        cloning_repo=str(entry.get("cloning_repo") or ""),
        revision=str(entry.get("revision") or "main"),
        # Upstream pins its two repositories at different commits -- the public
        # weights and the tokenizer at one, the cloning-capable weights at
        # another -- so this is a field of its own rather than one revision
        # standing in for both. It falls back to ``revision`` for a manifest
        # written before the difference was noticed.
        cloning_revision=str(entry.get("cloning_revision") or entry.get("revision")
                             or "main"),
        # A third one, and upstream's again: the voice embeddings were added to
        # the public repository after the weights, so its own configuration
        # names a later commit for them than for the model.
        voice_revision=str(entry.get("voice_revision") or entry.get("revision") or "main"),
        license=str(entry.get("license") or ""),
        attribution=str(entry.get("attribution") or ""),
        summary=str(entry.get("summary") or ""),
        notes=str(entry.get("notes") or ""),
        about_bytes=int(entry.get("about_bytes") or 0),
        ram_bytes=int(entry.get("ram_bytes") or 0),
        sample_rate=int(entry.get("sample_rate") or TARGET_RATE),
        recommended_temperature=(float(entry["recommended_temperature"])
                                 if entry.get("recommended_temperature") is not None
                                 else None),
        config=dict(entry.get("config") or {}),
        artifacts=tuple(_artifact(item) for item in (entry.get("files") or ())),
        required_paths=tuple(str(name) for name in (entry.get("required_paths") or ())),
        # Authorized by construction rather than by a flag somebody remembered
        # to set: these are the cloning-capable weights, they are the gated
        # repository, and there is no other reason for this list to exist.
        cloning_artifacts=tuple(_artifact(item, authorized=True)
                                for item in (entry.get("cloning_files") or ())),
        cloning_required_paths=tuple(str(name) for name
                                     in (entry.get("cloning_required_paths") or ())),
        voices=tuple(_official_voice(item) for item in (entry.get("voices") or ())))


def platform() -> "models.RuntimePlatform | None":
    """The wheel closure for this machine, or ``None`` if it is not on the list.

    An explicit allowlist, for the reason section 23.1 gives: upstream declaring
    Python 3.10 through 3.14 establishes feasibility, and a *combination*
    becomes supported once the exact closure below has been installed and
    self-tested on that class. A machine that is not in the manifest gets a
    sentence rather than an install that half works.

    A platform that is listed but whose artifacts have not been resolved yet is
    still returned. "This build has not pinned the closure" and "this machine is
    not supported" are different sentences with different remedies, and
    collapsing them into ``None`` would tell a supported machine it was not
    (see :func:`pinned`).
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


def pinned() -> bool:
    """Whether this build has resolved its PocketTTS runtime closure.

    False in a fresh checkout, and that is the honest state rather than a bug:
    section 23.1 requires exact filenames, byte counts and digests before
    release, and ``tools/pin_pocket_models.py`` is how they are produced on a
    machine that can reach the publishers. Until then nothing is downloaded,
    because an artifact this repository makes no claim about is an artifact it
    will not fetch.
    """
    try:
        chosen = platform()
    except PocketError:
        return False
    if chosen is None or not chosen.artifacts:
        return False
    return all(item.pinned for item in chosen.artifacts)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


@dataclass
class Status:
    """What is installed, in one object the panel and the runtime both read.

    Five readiness fields rather than one, because Pocket genuinely has five
    states and section 24 asks for them by name. "Installed" is not a useful
    answer to somebody whose speech works and whose Clone button does not.
    """

    platform_supported: bool = False
    runtime_ready: bool = False
    speech_model_ready: bool = False
    official_voices_ready: bool = False
    cloning_ready: bool = False
    label: str = LABEL
    fingerprint: str = ""
    model_id: str = ""
    download_bytes: int = 0
    ram_bytes: int = 0
    runtime_message: str = ""
    model_message: str = ""
    cloning_message: str = ""
    closure: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Whether ordinary speech can be started.

        Cloning is deliberately not in it. A Pocket that can speak with an
        official voice is a Pocket that works, and gating speech on an upstream
        access acceptance nobody has made would be refusing the feature over
        the half of it that is optional (section 23.3).
        """
        return bool(self.runtime_ready and self.speech_model_ready
                    and self.official_voices_ready)

    @property
    def message(self) -> str:
        if self.ready:
            return "Installed." if self.cloning_ready else \
                "Installed — official voices. Cloning needs upstream access."
        if not self.platform_supported:
            return self.runtime_message
        missing = [name for name, ok in (("runtime", self.runtime_ready),
                                         ("model", self.speech_model_ready),
                                         ("official voices", self.official_voices_ready))
                   if not ok]
        return f"Setup required — the PocketTTS {' and '.join(missing)} still to install."


def status() -> Status:
    """Read from disk and from verification metadata. Starts nothing.

    Section 17 again: reading status must never begin a download or a model
    load, so every branch below is a file-system question. Never raises -- a
    manifest this build cannot parse is reported as an uninstallable platform
    rather than as an exception on a status route a browser polls.
    """
    try:
        return _status()
    except PocketError as exc:
        return Status(runtime_message=str(exc), model_message=str(exc),
                      cloning_message=str(exc))
    except Exception:
        logger.debug("Model Chain: could not read the PocketTTS installation", exc_info=True)
        message = "PocketTTS's installation could not be read."
        return Status(runtime_message=message, model_message=message,
                      cloning_message=message)


def _status() -> Status:
    chosen = platform()
    entry = bundle()
    found = Status(label=entry.label, download_bytes=entry.download_bytes,
                   ram_bytes=entry.ram_bytes, model_id=entry.identifier)
    if chosen is None:
        system, machine, python_version = models.current_platform()
        found.platform_supported = False
        found.runtime_message = (
            f"PocketTTS has no tested CPU runtime for {system}/{machine} on Python "
            f"{python_version}. This release ships the PocketTTS closure for 64-bit "
            f"Windows on Python 3.10 to 3.13; Kokoro and Sopro are unaffected and still "
            f"work here.")
        found.model_message = found.runtime_message
        found.cloning_message = found.runtime_message
        return found

    found.platform_supported = True
    installed = _read_json(paths.pocket_runtime_manifest())
    closure = str((installed or {}).get("closure") or "")
    if not installed and not pinned():
        found.runtime_message = (
            "Not installed — this build has not pinned a PocketTTS runtime closure yet, so "
            "it will not download one. A maintainer runs tools/pin_pocket_models.py; you "
            "can install from a folder you filled yourself in the meantime.")
    elif not installed:
        found.runtime_message = (
            "Not installed — about "
            f"{models._bytes_label(sum(int(item.size or 0) for item in chosen.artifacts))} "
            "of PyTorch and PocketTTS, on the CPU.")
    elif chosen.artifacts and closure != chosen.closure_id:
        # The closure id is derived from the platform id and every wheel's hash,
        # so this is true exactly when the pinned closure has changed -- which
        # is the moment a saved voice's fingerprint stops meaning what it meant.
        # Reported rather than silently re-used (I-PKT-18).
        found.runtime_message = ("Installed, but this build pins a different PocketTTS "
                                 "runtime. Install it again to update.")
    else:
        found.runtime_ready = True
        found.runtime_message = (
            f"Installed — PocketTTS {installed.get('pocket_version') or '?'}, "
            f"Torch {installed.get('torch_version') or '?'}, CPU only.")
    found.closure = dict(installed or {})

    root = paths.pocket_model_root(entry.identifier)
    model = _read_json(root / paths.INSTALLED_FILENAME)
    missing = [name for name in entry.required_paths if not (root / name).exists()]
    if not entry.required_paths:
        found.model_message = ("Not installed — this build has not recorded the PocketTTS "
                               "model artifacts yet. A maintainer runs "
                               "tools/pin_pocket_models.py.")
    elif not model or missing:
        found.model_message = (f"Not installed — {entry.label}, about "
                               f"{models._bytes_label(entry.download_bytes)}.")
    else:
        found.speech_model_ready = True
        found.model_message = f"Installed — {entry.label}."

    found.official_voices_ready = _official_voices_ready(entry)
    if not entry.voices:
        found.model_message = found.model_message or ""
    found.fingerprint = _fingerprint(chosen, installed, model, entry)

    cloning = _read_json(root / CLONING_MARKER)
    gone = [name for name in entry.cloning_required_paths if not (root / name).exists()]
    if not entry.cloning_artifacts:
        found.cloning_message = ("Voice cloning needs PocketTTS's gated weights, which this "
                                 "build has not recorded yet. A maintainer runs "
                                 "tools/pin_pocket_models.py.")
    elif cloning and not gone:
        found.cloning_ready = bool(found.runtime_ready and found.speech_model_ready)
        found.cloning_message = "Installed — you can clone a voice from a recording."
    else:
        found.cloning_message = (
            f"Not installed. PocketTTS's voice-cloning weights are a gated repository: "
            f"accept the conditions at huggingface.co/{entry.cloning_repo} with your own "
            f"account, then save an access token under Settings → Voice Chat → Access "
            f"token and press Install. Official PocketTTS voices work without this.")
    return found


CLONING_MARKER = "installed-cloning.json"
"""The marker that says the gated half arrived.

Its own file beside the model's, rather than a flag inside it, so that a failed
cloning install cannot make a working official-voice installation look broken:
the two halves are written separately and read separately (T-INSTALL-P7).
"""


def _official_voices_ready(entry: Bundle) -> bool:
    """Whether every declared official voice state is on disk.

    All of them rather than any: a catalogue that listed four voices and could
    speak two would be a catalogue where choosing the wrong one is silence.
    """
    if not entry.voices:
        return False
    root = paths.pocket_official_root(entry.identifier)
    for voice in entry.voices:
        if not (root / f"{voice.identifier}{paths.POCKET_STATE_SUFFIX}").is_file():
            return False
    return True


def _fingerprint(chosen, installed: dict, model: dict, entry: Bundle) -> str:
    """What "this voice state was prepared by this build" actually means.

    Content and closure rather than a version string, because ``pocket-tts==3.0.2``
    is a name and this has to be an identity. Section 12's list:

        the pinned wheel closure, itself a hash of every artifact's digest;
        the PocketTTS package version actually installed;
        the Torch version actually installed;
        the digests of the model artifacts that were actually verified;
        this build's voice-state schema version;
        the model/language id.

    Precision is in it too, and that is a *conservative* choice with a gate
    against it. Sopro could leave quantization out because its int8 touches only
    the autoregressive blocks and not the encoders that produce the saved
    tensors; nobody has established the equivalent for Pocket. GATE P-VOICE-1
    prepares a state under each precision and loads it under the other, and
    until it passes, changing precision marks a cached state unavailable until
    it is rebuilt rather than loading one that may not mean the same thing
    (section 12).

    Sixteen hex characters: this is compared against a value this installer
    wrote itself rather than defended against a forger.
    """
    parts = [
        f"closure:{chosen.closure_id if chosen is not None else ''}",
        f"pocket:{(installed or {}).get('pocket_version') or ''}",
        f"torch:{(installed or {}).get('torch_version') or ''}",
        f"schema:{STATE_SCHEMA}",
        f"model:{entry.identifier}",
        f"precision:{precision()}",
    ]
    for name, digest in sorted(((model or {}).get("digests") or {}).items()):
        parts.append(f"{name}:{digest}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def runtime_python() -> "Path | None":
    """The isolated PocketTTS interpreter, or ``None`` when it is not installed."""
    root = paths.pocket_runtime_root() / "env"
    interpreter = ((root / "Scripts" / "python.exe") if os.name == "nt"
                   else (root / "bin" / "python"))
    return interpreter if interpreter.exists() else None


def worker_environment() -> dict:
    """The environment the Pocket worker runs in. I-PKT-7, where it is enforced.

    ``CUDA_VISIBLE_DEVICES=""`` is the blunt instrument and the important one: a
    Torch build that would happily have found a GPU finds no devices to
    enumerate, so Pocket cannot claim VRAM an image generation is using. This is
    a Model Chain V1 support decision and not a claim that upstream Pocket can
    never execute elsewhere -- but for this release the worker's handshake has to
    be able to say ``provider=cpu`` and mean it.

    There is deliberately no thread variable here. PocketTTS 3.0.2 calls
    ``torch.set_num_threads(1)`` itself and takes its parallelism from its own
    generation and decoder threads, so setting ``OMP_NUM_THREADS`` to a number
    of our choosing and reporting it as a Pocket thread count would be reporting
    something that is not true (section 16.4, section 35).

    ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` are braces rather than belt.
    ``pocket_tts.utils.utils`` imports ``huggingface_hub`` and ``requests`` at
    module level, so the closure has both and turning them off is a real
    instruction rather than a precaution: nothing in the worker resolves a
    location -- the config it is handed names only local files and it refuses one
    that does not -- and these say so to the libraries as well (I-PKT-20,
    section 25). No credential is here, and there is no branch that could put one
    here.

    ``POCKET_TTS_NO_BEARTYPE`` keeps upstream's runtime type-checking claw from
    wrapping every function in its package. Upstream's own comment says the
    wrapper costs per call and blocks ``torch.compile``; the boundary this
    process actually has to defend is its pipe, and that is checked in the worker
    rather than by decorating a library. It does not remove the dependency --
    ``pocket_tts/data/audio.py`` imports ``beartype.typing`` unconditionally, so
    the closure ships it either way.
    """
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "POCKET_TTS_NO_BEARTYPE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def worker_config() -> dict:
    """Everything the worker is told at ``init``. Paths this process built.

    None of it can be influenced from a browser: the model root comes from the
    manifest and the data root, the settings come from Pocket's own file, and
    the catalogue is built from the registry by :func:`catalog`. The worker
    never receives a URL, a repository id or a credential -- it receives a
    config file whose every path names a file this process verified (I-PKT-20).
    """
    entry = bundle()
    found = status()
    return {
        "model_root": str(paths.pocket_model_root(entry.identifier)),
        "config_path": str(paths.pocket_model_config(entry.identifier)),
        "official_root": str(paths.pocket_official_root(entry.identifier)),
        "clones_root": str(paths.pocket_clones_root()),
        "model_id": entry.identifier,
        "precision": precision(),
        "sampler_steps": steps(),
        "fingerprint": found.fingerprint,
        "state_schema": STATE_SCHEMA,
        "sample_rate": entry.sample_rate,
        "cloning_ready": found.cloning_ready,
        "reference_seconds": IDEAL_REFERENCE_SECONDS,
        "upstream_build_id": str(found.closure.get("upstream_build_id") or ""),
        "voices": catalog(),
    }


# --------------------------------------------------------------------------- #
# Engine settings
# --------------------------------------------------------------------------- #


def precision() -> str:
    found = str(_setting(SETTING_PRECISION) or "").strip().lower()
    return found if found in PRECISIONS else PRECISION_DEFAULT


def steps() -> int:
    try:
        found = int(_setting(SETTING_STEPS) or 0)
    except (TypeError, ValueError):
        found = 0
    return found if found in STEP_CHOICES else STEP_DEFAULT


def model_id() -> str:
    """The selected Pocket model, or the manifest default.

    Persisted from day one even though V1 release scope may expose only English
    (section 16.3). The architecture must not encode "Pocket means English" in
    stable storage: a later release that adds a language should be a manifest
    entry and a selector, not a migration.
    """
    found = str(_setting(SETTING_MODEL) or "").strip()
    known = model_ids()
    if found in known:
        return found
    try:
        offered = str((manifest().get("defaults") or {}).get("model") or "")
    except PocketError:
        return found
    return offered if offered in known else (known[0] if known else "")


def thread_policy() -> str:
    """What this build can honestly say about Pocket's CPU execution.

    A sentence rather than a control. Section 16.4 asks for exactly this, and
    the reason it is worth a function is that the honest answer changes if a
    later Pocket release exposes a supported thread policy -- at which point it
    becomes a new tested engine setting rather than a slider that was already
    there accepting arbitrary integers.
    """
    return ("PocketTTS manages its CPU execution policy internally in this build. "
            "There is no supported thread-count control.")


def engine_settings() -> dict:
    """The global Pocket runtime controls, as the panel draws them.

    Global rather than per character on purpose (I-PKT-23, section 13): they
    change compute, RAM and warm-cache compatibility for the whole worker, and a
    character setting that quietly reloaded the model would be a character
    setting nobody could reason about.
    """
    return {
        "precision": precision(),
        # Neither option claims a speed. Upstream reports lower memory and
        # faster x86 inference for its INT8 path and that may well hold here,
        # but Sopro's identical-looking setting was measured 40% the *other*
        # way on the first machine anybody tried -- so this build measures
        # before it says (GATE P-5, I-PKT-26).
        "precisions": [
            {"id": "full", "label": "Full CPU precision",
             "help": "PocketTTS exactly as it was released, and the one to compare "
                     "against. Larger in memory than INT8."},
            {"id": "int8", "label": "INT8 (smaller, CPU only)",
             "help": "Applies PocketTTS's own dynamic INT8 quantization at load time, so "
                     "the model is lighter in RAM. Whether it is faster depends on the "
                     "machine — the turn summary in model_chain.log reports the real-time "
                     "factor either way. A saved voice may have to be rebuilt after this "
                     "changes, and the voice library says so when it does."},
        ],
        "steps": steps(),
        "step_choices": [{"id": value, "label": STEP_LABELS.get(value, str(value))}
                         for value in STEP_CHOICES],
        "model_id": model_id(),
        "model_choices": [{"id": name, "label": _model_label(name)}
                          for name in model_ids()],
        "thread_policy": thread_policy(),
    }


def _model_label(identifier: str) -> str:
    try:
        return bundle(identifier).label
    except PocketError:
        return str(identifier)


def set_engine_settings(precision_id: str = "", sampler_steps=None,
                        wanted_model: str = "") -> dict:
    """Change a global Pocket runtime setting and stop the worker if it matters.

    Stopped rather than reconfigured: precision is chosen at model load, the
    sampler step count is the generation policy, and the model is the model. A
    turn must never change any of the three in its middle (I-PKT-24), so the
    setting is written, whatever is speaking is retired, and the *next* request
    starts a worker with the new configuration.

    Terminating the worker is valid here even though ordinary Pocket Stop
    drains: this is a lifecycle boundary, not the user Stop contract
    (section 21.6, section 33). Nothing is downloaded and nothing is restarted;
    the next Pocket speech, test or preview starts it.

    It never changes Kokoro or Sopro (I-PKT-3), which is enforced by the storage
    being Pocket's own file rather than by a check that could be forgotten.

    A value that is not one of the offered ones is *refused* rather than
    dropped. Both are safe -- neither writes the setting -- but only one of them
    is honest: a panel that sent a precision this build does not have and was
    answered with the unchanged settings and no error would show the old value
    with no explanation of why its click did nothing.
    """
    changed = []
    offered = str(precision_id or "").strip().lower()
    if offered:
        if offered not in PRECISIONS:
            raise PocketError(f"{offered!r} is not a precision PocketTTS offers.")
        if offered != precision():
            _remember(SETTING_PRECISION, offered)
            changed.append("precision")
    if sampler_steps is not None:
        try:
            value = int(sampler_steps)
        except (TypeError, ValueError):
            value = None
        # Against the offered list rather than against a range, so a number a
        # browser invented cannot become a generation policy nobody has heard.
        if value not in STEP_CHOICES:
            raise PocketError("That is not a PocketTTS generation quality this build "
                              "offers.")
        if value != steps():
            _remember(SETTING_STEPS, value)
            changed.append("generation quality")
    offered = str(wanted_model or "").strip()
    if offered:
        if offered not in model_ids():
            raise PocketError(f"{offered!r} is not a PocketTTS model this build has "
                              f"recorded.")
        if offered != model_id():
            _remember(SETTING_MODEL, offered)
            changed.append("model")
    if changed:
        _retire(f"the PocketTTS {' and '.join(changed)} changed")
        logger.info("Model Chain: PocketTTS %s changed", " and ".join(changed))
    return engine_settings()


def apply_engine_settings(values: dict = None) -> dict:
    """One engine setting change, in the names the *wire* uses.

    A second entry point beside :func:`set_engine_settings`, and the reason is
    the generic route in front of it: one route serves every engine that has
    settings, so what it forwards has to be a stable vocabulary rather than
    whichever Python parameter names this module happens to use. ``precision``,
    ``steps`` and ``model_id`` are that vocabulary here, and they are the names
    the panel sends.

    An unknown key is refused rather than ignored. A page drawn for a build
    whose panel had a control this one does not is a stale surface, and
    answering it with silence would be answering it with "applied".
    """
    offered = {str(key): value for key, value in dict(values or {}).items()}
    known = {"precision", "steps", "model_id"}
    unknown = sorted(set(offered) - known)
    if unknown:
        raise PocketError(f"{unknown[0]!r} is not a PocketTTS engine setting.")
    return set_engine_settings(precision_id=str(offered.get("precision") or ""),
                               sampler_steps=offered.get("steps"),
                               wanted_model=str(offered.get("model_id") or ""))


def _retire(reason: str) -> None:
    """Silence whatever Pocket is saying, then stop its worker. Never raises.

    In that order, for the reason :func:`mc_voice_engines.select` gives: a worker
    stopped while a turn still believes it is being spoken leaves a browser
    waiting on a stream that will never produce another byte.
    """
    try:
        import mc_voice_turn as turns

        turn = turns.active()
        if turn is not None and turn.engine == ENGINE:
            turn.cancel(reason)
            turn.drain_audio()
    except Exception:
        logger.debug("Model Chain: could not retire the PocketTTS turn", exc_info=True)
    try:
        import mc_voice_pocket_runtime as runtime

        runtime.stop(reason)
    except Exception:
        logger.debug("Model Chain: could not stop PocketTTS after a settings change",
                     exc_info=True)


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #


def refusal(manual: bool = False) -> str:
    """Why Pocket cannot be installed right now, or an empty string.

    Asked before anything is started, and separately from starting it, because
    the two questions have different audiences: this one answers a browser that
    needs a sentence to put on screen, and the transaction below answers a log.
    """
    try:
        chosen = platform()
    except PocketError as exc:
        return str(exc)
    if chosen is None:
        system, machine, python_version = models.current_platform()
        return (f"PocketTTS has no tested CPU runtime for {system}/{machine} on Python "
                f"{python_version}, so it cannot be installed here.")
    with models._lock:
        if (models._progress.get(KIND) or {}).get("running"):
            return "PocketTTS is already being installed."
    if not manual and not pinned():
        # The runtime closure is pinned in this repository and must stay pinned:
        # an unpinned wheel is a wheel nobody reviewed, and the whole voice-state
        # fingerprint rests on knowing which bytes ran. Until the pinning tool
        # has been run this build will not download one, and says which tool.
        return ("This build has not pinned a PocketTTS runtime closure yet, so it will not "
                "be downloaded. A maintainer runs tools/pin_pocket_models.py on a machine "
                "that can reach pypi.org; you can install from a folder you filled yourself "
                "in the meantime.")
    return ""


KIND = "pocket"
"""What the shared installer's progress table calls this engine's work."""


def sources(part: str = "runtime") -> list:
    """Where a person would go to fetch Pocket's files by hand.

    The manifest already names every URL, so the manual panel can print them
    rather than asking somebody to find the right PyTorch wheel themselves. The
    gated half is listed too, because knowing the address is not the thing the
    gate protects -- accepting the conditions is.
    """
    entry = bundle()
    if part == "runtime":
        chosen = platform()
        artifacts = chosen.artifacts if chosen else ()
    elif part == "cloning":
        artifacts = entry.cloning_artifacts
    elif part == "voices":
        artifacts = entry.voice_artifacts
    else:
        artifacts = entry.artifacts
    return [{"filename": item.filename, "url": item.url, "save_as": item.local_name,
             "archive": False} for item in artifacts]


def install(on_status=None, on_progress=None, cloning: bool = True) -> Status:
    """Install the PocketTTS runtime, model, official voices and -- if the gate
    allows it -- the cloning weights. One button.

    A transaction, four times over. Nothing outside a staging directory is
    touched until every declared byte has arrived and matched its hash, each
    part is promoted by a directory rename, and a failure anywhere leaves the
    installation exactly as it was.

    The gated half is deliberately last and deliberately not fatal. "One click"
    means one Voice Chat action *after* the upstream access precondition is
    satisfied (section 23.3); a machine with no token still ends this function
    with working official-voice speech and a panel that says what cloning needs.
    Installing Pocket modifies neither Kokoro nor Sopro, and there is no shared
    staging tree, marker file or runtime.
    """
    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)
    with models._claim(KIND, say, bundle().identifier):
        say("Checking the PocketTTS runtime…")
        install_runtime(on_status=say, on_progress=lambda f: tick(f * 0.35))
        install_model(on_status=say, on_progress=lambda f: tick(0.35 + f * 0.40))
        install_voices(on_status=say, on_progress=lambda f: tick(0.75 + f * 0.15))
        if cloning:
            try:
                install_cloning(on_status=say, on_progress=lambda f: tick(0.90 + f * 0.10))
            except models.Gated as exc:
                # The one failure in this transaction that is a *state* rather
                # than a fault. Official voices are installed and working; what
                # is missing is an acceptance somebody has to make upstream.
                say(str(exc))
                logger.info("Model Chain: PocketTTS installed without voice cloning — %s",
                            exc)
            except (PocketError, models.VoiceError, OSError) as exc:
                # Everything the gated half can fail with, and not just this
                # module's own refusals. The three parts before it have already
                # been promoted and work; a link that dropped or a disk that
                # filled while fetching the cloning weights would otherwise
                # abort an installation that had already succeeded, and the
                # panel would report a failure for speech that speaks.
                say(str(exc) or "The PocketTTS voice-cloning weights were not installed.")
                logger.warning("Model Chain: the PocketTTS cloning weights were not "
                               "installed — %s: %s", exc.__class__.__name__, exc)
        tick(1.0)
        say("PocketTTS installed.")
        return status()


def install_runtime(on_status=None, on_progress=None, folder=None) -> None:
    """The isolated Torch/PocketTTS closure: an interpreter of its own and the wheels.

    Built exactly the way Kokoro's and Sopro's are -- a virtual environment
    without pip, and the verified wheels *unpacked* rather than installed by a
    package manager. That is the correction to a real failure: pip can report
    success, exit zero and have installed into somebody's ``PIP_TARGET`` or user
    site, leaving the runtime empty. There is no package manager in this path,
    so there is nothing left to redirect it -- and no index resolution can happen
    inside a user's WebUI at runtime (section 23.1).
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    chosen = platform()
    if chosen is None:
        raise PocketError(refusal() or "PocketTTS has no runtime for this platform.")
    if not chosen.artifacts:
        raise PocketError(refusal(manual=bool(folder)) or
                          "This build has not recorded a PocketTTS runtime closure.")

    installed = _read_json(paths.pocket_runtime_manifest())
    if installed and str(installed.get("closure") or "") == chosen.closure_id \
            and runtime_python() is not None:
        say("The PocketTTS runtime is already installed.")
        tick(1.0)
        return

    staging = paths.pocket_staging_for("runtime", uuid.uuid4().hex[:8])
    wheels = staging / "wheels"
    shutil.rmtree(staging, ignore_errors=True)
    wheels.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            _adopt(chosen.artifacts, Path(str(folder)).expanduser(), wheels, say,
                   "PocketTTS runtime wheel")
        else:
            expectations = models._expectations(chosen.artifacts, say)
            models._make_room(chosen.artifacts, staging, expectations)
            models._fetch_all(chosen.artifacts, wheels, say, tick, 0.7, expectations)
        say("Building the isolated PocketTTS runtime…")
        _build_environment(staging, wheels, chosen)
        tick(0.85)
        say("Checking that PocketTTS runs on this machine…")
        report = _smoke_test(staging)
        tick(0.95)
        shutil.rmtree(wheels, ignore_errors=True)
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "closure": chosen.closure_id,
            "platform": chosen.identifier,
            "pocket_version": report.get("pocket_version") or "",
            "upstream_build_id": report.get("upstream_build_id") or "",
            "torch_version": report.get("torch_version") or "",
            "numpy_version": report.get("numpy_version") or "",
            "thread_policy": report.get("thread_policy") or "",
            "license": manifest()["runtime"].get("license") or "",
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        models._promote(staging, paths.pocket_runtime_root())
        _manifest_cache_clear()
        say("The PocketTTS runtime is installed.")
        tick(1.0)
        logger.info("Model Chain: the PocketTTS runtime is installed — %s, PocketTTS %s, "
                    "Torch %s", chosen.identifier, report.get("pocket_version"),
                    report.get("torch_version"))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_model(on_status=None, on_progress=None, folder=None) -> None:
    """The public PocketTTS artifacts, verified and promoted by rename.

    The public repository, always. The cloning-capable weights are a separate
    upstream repository behind an access gate and are :func:`install_cloning`'s
    job, which is what makes "Pocket speaks but cannot clone yet" a state this
    installer can actually reach (section 23.3).
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    entry = bundle()
    if not entry.artifacts:
        raise PocketError("This build has not recorded the PocketTTS model artifacts yet. "
                          "A maintainer runs tools/pin_pocket_models.py.")
    target = paths.pocket_model_root(entry.identifier)
    marker = _read_json(target / paths.INSTALLED_FILENAME)
    missing = [name for name in entry.required_paths if not (target / name).exists()]
    if marker and not missing:
        say(f"{entry.label} is already installed.")
        tick(1.0)
        return

    staging = paths.pocket_staging_for(entry.identifier, uuid.uuid4().hex[:8])
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
            raise PocketError(f"{entry.label} is missing {gone[0]} after the download. "
                              f"Nothing was installed.")
        _sanity_check(staging, entry.required_paths, entry.label)
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "id": entry.identifier,
            "repo": entry.public_repo,
            "revision": entry.revision,
            "license": entry.license,
            "attribution": entry.attribution,
            "digests": {name: digest for name, digest in sorted(digests.items())},
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        models._promote(staging, target)
        # After the promote, because the runtime is what reads it and the model
        # root is where it lands. Before the local config, because the local
        # config is this document with three paths replaced.
        say("Reading PocketTTS's model configuration…")
        _write_json(paths.pocket_upstream_config(entry.identifier), _read_recipe(entry))
        _write_local_config(entry)
        say(f"{entry.label} is installed.")
        tick(1.0)
        logger.info("Model Chain: the PocketTTS model %s is installed", entry.identifier)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_voices(on_status=None, on_progress=None, folder=None) -> None:
    """The precomputed official voice states, installed locally once.

    Locally and once, which is I-PKT-22: normal speech does not fetch a
    reference voice over the network, and an official voice is a safetensors
    state on this disk with reviewed attribution beside it rather than a name
    resolved against whatever upstream's voice dictionary holds today.
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    entry = bundle()
    artifacts = entry.voice_artifacts
    if not artifacts:
        say("This build has not recorded any official PocketTTS voices yet.")
        tick(1.0)
        return
    target = paths.pocket_official_root(entry.identifier)
    if _official_voices_ready(entry):
        say("The official PocketTTS voices are already installed.")
        tick(1.0)
        return

    staging = paths.pocket_staging_for(f"{entry.identifier}-voices", uuid.uuid4().hex[:8])
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            _adopt(artifacts, Path(str(folder)).expanduser(), staging, say,
                   "official PocketTTS voice")
        else:
            expectations = models._expectations(artifacts, say)
            models._make_room(artifacts, staging, expectations)
            models._fetch_all(artifacts, staging, say, tick, 0.9, expectations)
        _sanity_check(staging, tuple(item.local_name for item in artifacts),
                      "The official PocketTTS voices")
        models._promote(staging, target)
        say("The official PocketTTS voices are installed.")
        tick(1.0)
        logger.info("Model Chain: %d official PocketTTS voice state(s) are installed",
                    len(artifacts))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_cloning(on_status=None, on_progress=None, folder=None) -> None:
    """The gated voice-cloning weights, if the upstream gate lets this machine have them.

    The supported managed path, in order (section 23.3):

        the user has accepted Kyutai's model conditions upstream
        and a token is available to *this process*
            -> the installer asks huggingface.co with an Authorization header
            -> reads the publisher's digest and size
            -> follows the signed delivery URL **without** forwarding it
            -> verifies and installs

    Voice Chat cannot accept the conditions on somebody's behalf, legally or
    technically, and does not pretend to: a 401 or 403 here is
    :class:`mc_voice_models.Gated`, which is a sentence about access rather than
    a download failure. Nothing about this half can remove a working
    official-voice installation (T-INSTALL-P7): it writes its own marker beside
    the model's rather than rewriting it.
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    entry = bundle()
    if not entry.cloning_artifacts:
        raise PocketError("This build has not recorded PocketTTS's voice-cloning weights "
                          "yet. A maintainer runs tools/pin_pocket_models.py.")
    target = paths.pocket_model_root(entry.identifier)
    if not (target / paths.INSTALLED_FILENAME).is_file():
        raise PocketError("Install the PocketTTS model before its voice-cloning weights.")
    marker = _read_json(target / CLONING_MARKER)
    gone = [name for name in entry.cloning_required_paths if not (target / name).exists()]
    if marker and not gone:
        say("The PocketTTS voice-cloning weights are already installed.")
        tick(1.0)
        return

    staging = paths.pocket_staging_for(f"{entry.identifier}-cloning", uuid.uuid4().hex[:8])
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            digests = _adopt(entry.cloning_artifacts, Path(str(folder)).expanduser(),
                             staging, say, "PocketTTS voice-cloning weight")
        else:
            expectations = models._expectations(entry.cloning_artifacts, say)
            models._make_room(entry.cloning_artifacts, staging, expectations)
            digests = models._fetch_all(entry.cloning_artifacts, staging, say, tick, 0.9,
                                        expectations)
        _sanity_check(staging, entry.cloning_required_paths,
                      "The PocketTTS voice-cloning weights")
        # Moved into the installed model directory one file at a time rather
        # than promoted over it: the official half is already there and working,
        # and a directory rename would take it away for the length of the move.
        #
        # Which means this loop is the one step here that is not a rename, so it
        # is the one step that can stop half way. What it added is undone on the
        # way out -- what it *replaced* is not, because a file that was already
        # there was already this build's and restoring a copy of it would mean
        # keeping a copy of it. Today there is one artifact and the partial state
        # is unreachable; the second one is what this is for.
        added = []
        try:
            for item in entry.cloning_artifacts:
                source = staging / item.local_name
                destination = target / item.local_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    added.append(destination)
                os.replace(source, destination)
        except BaseException:
            for destination in added:
                try:
                    destination.unlink()
                except OSError:
                    logger.debug("Model Chain: could not undo a partial PocketTTS "
                                 "cloning install", exc_info=True)
            raise
        _write_json(target / CLONING_MARKER, {
            "schema": SCHEMA,
            "repo": entry.cloning_repo,
            "revision": entry.cloning_revision,
            "license": entry.license,
            "digests": {name: digest for name, digest in sorted(digests.items())},
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        _write_local_config(entry)
        say("The PocketTTS voice-cloning weights are installed.")
        tick(1.0)
        logger.info("Model Chain: the PocketTTS voice-cloning weights are installed")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_from(part: str, folder: str, on_status=None, on_progress=None) -> Status:
    """Install from files already on this machine. The escape hatch.

    Earns its place beyond the situation that prompted it: a machine with no
    route to pypi.org or huggingface.co, a corporate proxy that refuses a
    hundred-megabyte binary, an air-gapped install, or somebody who already has
    these files for another application. It is also, while this build's closure
    is unpinned, the *only* way in -- which is stated in the panel rather than
    discovered.

    What is different from the managed path is stated rather than glossed. A
    pinned artifact is checked against the hash committed here, so a file under
    the right name with the wrong contents is refused exactly as a bad download
    is. An unpinned one has its digest recorded at install time and becomes the
    constant the *next* install is checked against.
    """
    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)
    with models._claim(KIND, say, bundle().identifier):
        if part == "runtime":
            install_runtime(on_status=say, on_progress=tick, folder=folder)
        elif part == "model":
            install_model(on_status=say, on_progress=tick, folder=folder)
        elif part == "voices":
            install_voices(on_status=say, on_progress=tick, folder=folder)
        elif part == "cloning":
            install_cloning(on_status=say, on_progress=tick, folder=folder)
        else:
            raise PocketError("PocketTTS installs its runtime, its model, its official "
                              "voices or its voice-cloning weights.")
        return status()


def _adopt(artifacts, folder: Path, destination: Path, say, what: str) -> dict:
    """Copy named files out of a folder somebody filled themselves, checking each.

    Named files only: nothing is guessed from an extension, nothing is renamed
    from something similar, and a file that is present under the right name with
    the wrong contents is refused. What comes back is the digest of every file
    that was accepted, which is what the installed marker records.
    """
    if not folder.is_dir():
        raise PocketError(f"{folder} is not a folder this machine can read.")
    destination.mkdir(parents=True, exist_ok=True)
    digests = {}
    for item in artifacts:
        source = folder / item.filename
        if not source.is_file():
            source = folder / item.local_name
        if not source.is_file():
            raise PocketError(f"{item.filename} is not in {folder}. Nothing was installed.")
        say(f"Checking {item.filename}…")
        digest = _digest(source)
        if item.sha256 and digest != item.sha256:
            raise PocketError(
                f"{item.filename} is in that folder, but its contents are not what this "
                f"extension's manifest says they should be. Nothing was installed.")
        if item.size and source.stat().st_size != item.size:
            raise PocketError(
                f"{item.filename} is {source.stat().st_size} bytes and this extension "
                f"expects {item.size}. Nothing was installed.")
        target = destination / item.local_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digests[item.filename] = digest
    logger.info("Model Chain: PocketTTS adopted %d file(s) for the %s from %s",
                len(digests), what, folder)
    return digests


SAFETENSORS_MIN = 16
"""A safetensors file is an eight-byte little-endian header length followed by
that much JSON. Anything shorter is not one, whatever it is called."""


def _sanity_check(staging: Path, required, label: str) -> None:
    """Structural validation before anything is promoted.

    Not a hash -- the hash has already been checked, or recorded. This asks the
    different question of whether the bytes are the *kind of thing* they claim
    to be, because a proxy that answers every request with an HTML error page
    produces a file of exactly the right name and a digest that matches nothing
    this build had pinned.
    """
    for name in required:
        path = staging / name
        try:
            size = path.stat().st_size
        except OSError:
            raise PocketError(f"{label} is missing {name}. Nothing was installed.") from None
        if size <= 0:
            raise PocketError(f"{name} downloaded as an empty file. Nothing was installed.")
        if name.endswith(".safetensors"):
            with open(path, "rb") as handle:
                head = handle.read(8)
            if len(head) < 8:
                raise PocketError(f"{name} is too small to be a safetensors file. Nothing "
                                  f"was installed.")
            (declared,) = struct.unpack("<Q", head)
            if declared <= 0 or declared + 8 > size or declared > 64 * 1024 * 1024:
                raise PocketError(f"{name} does not have a safetensors header. Nothing was "
                                  f"installed.")
        elif name.endswith(".json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise PocketError(f"{name} is not readable JSON. Nothing was "
                                  f"installed.") from None
        elif size < SAFETENSORS_MIN:
            raise PocketError(f"{name} is too small to be what it claims. Nothing was "
                              f"installed.")


def _build_environment(staging: Path, wheels: Path, chosen) -> None:
    """An interpreter of Pocket's own, with the verified wheels unpacked into it.

    No package manager anywhere in this path. ``venv`` without pip, then each
    wheel's contents written into the environment's site-packages -- so there is
    no index to resolve against, no dependency graph to be surprised by, and
    nothing that can be redirected into somebody's user site.
    """
    import venv

    environment = staging / "env"
    try:
        builder = venv.EnvBuilder(with_pip=False, clear=True, symlinks=(os.name != "nt"))
        builder.create(environment)
    except Exception as exc:
        logger.warning("Model Chain: the isolated PocketTTS runtime could not be created "
                       "using %s — %s: %s", environment, exc.__class__.__name__, exc)
        raise PocketError(f"The isolated PocketTTS runtime could not be created "
                          f"({exc.__class__.__name__}: {exc}).") from None
    interpreter = ((environment / "Scripts" / "python.exe") if os.name == "nt"
                   else (environment / "bin" / "python"))
    if not interpreter.exists():
        raise PocketError("The isolated PocketTTS runtime was created without an "
                          "interpreter. Nothing was installed.")
    target = models.site_packages(environment)
    for item in chosen.artifacts:
        added = models._unpack_wheel(wheels / item.local_name, target)
        logger.info("Model Chain: PocketTTS unpacked %s (%s)", item.filename,
                    ", ".join(added[:6]) or "nothing")
    for name in (manifest()["runtime"].get("import_name") or "pocket_tts", "torch"):
        if not (target / name).exists():
            raise PocketError(f"The staged PocketTTS runtime has no {name} in it. Nothing "
                              f"was installed.")


def _smoke_test(staging: Path) -> dict:
    """Prove the staged runtime runs *before* anything is promoted.

    Two staged runs: the interpreter answers at all, and the worker's own
    ``--selftest`` imports PocketTTS and Torch, reports what it got and reports
    the device it is on. A runtime that cannot say ``cpu`` is refused here rather
    than at the first reply somebody wanted spoken.
    """
    interpreter = ((staging / "env" / "Scripts" / "python.exe") if os.name == "nt"
                   else (staging / "env" / "bin" / "python"))
    alive = _run_staged(interpreter, ["-c", "import sys; print(sys.version)"],
                        "check that the isolated PocketTTS interpreter runs", timeout=180)
    if alive.returncode != 0:
        raise PocketError(
            "The isolated PocketTTS interpreter would not start. This usually means the "
            "Python running this WebUI is an embedded or relocated build that cannot make "
            "a virtual environment. Nothing was installed and Voice Chat is unchanged.")
    result = _run_staged(interpreter, [paths.pocket_worker_script(), "--selftest"],
                         "PocketTTS self-test", timeout=600)
    try:
        report = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        logger.warning("Model Chain: the staged PocketTTS runtime said %s",
                       models._quote(result.stdout))
        raise PocketError("The staged PocketTTS runtime did not report what it is. Nothing "
                          "was installed.") from None
    if not report.get("ok"):
        # Logged whole and shown short. The full text is what a maintainer
        # diagnoses from and it is already in ``model_chain.log``; the panel gets
        # the sentence without the path, because a refusal is read on screen,
        # photographed, and pasted into a bug report.
        logger.warning("Model Chain: the staged PocketTTS runtime could not load — %s",
                       report.get("error") or "no reason reported")
        raise PocketError(
            f"The staged PocketTTS runtime could not load ("
            f"{_without_paths(report.get('error')) or 'no reason reported'}). Nothing was "
            f"installed and Voice Chat is unchanged.")
    if str(report.get("device") or "") != "cpu":
        raise PocketError(
            f"The staged PocketTTS runtime reported the device {report.get('device')!r} "
            f"rather than the CPU. This release supports PocketTTS on the CPU only. "
            f"Nothing was installed.")
    logger.info("Model Chain: the staged PocketTTS runtime passed its self-test — "
                "PocketTTS %s, Torch %s, NumPy %s, %s",
                report.get("pocket_version"), report.get("torch_version"),
                report.get("numpy_version"), report.get("thread_policy"))
    return report


_PATHLIKE = re.compile(
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]|(?<![:\w\\/])[\\/])"
    r"[^\s'\"()<>]*[\\/]([^\s'\"()<>\\/]+)")
"""An absolute filesystem path, Windows or POSIX, with its last component captured.

The two lookbehinds are what keep it to *filesystem* paths. Without the first,
the ``s:`` in ``https://`` is a drive letter; without the second, the ``//``
after it is a root. A URL and an ``hf://`` location are left whole, which is
right for a different reason as well: where one of those appears in a refusal it
is the thing the refusal is about.
"""


def _without_paths(text) -> str:
    """One staged-runtime message with its filesystem paths cut down to filenames.

    A refusal from inside the isolated runtime is a library's own sentence, and
    a library is entitled to put the file it was reading into it -- which is how
    ``cannot import name 'Sentinel' from 'typing_extensions'`` arrives carrying
    the whole of somebody's install directory. The sentence is the useful half:
    it is what a user acts on and what a maintainer diagnoses from. Where their
    WebUI lives adds nothing to either, and unlike a log line, a panel is read
    on screen, photographed and pasted into a bug report (section 36).

    The filename is kept rather than the path removed entirely, because
    "typing_extensions" is the part of ``…site-packages/typing_extensions.py``
    that says which package is wrong.
    """
    return _PATHLIKE.sub(lambda found: found.group(1), str(text or "")).strip()


def _run_staged(interpreter: Path, arguments: list, what: str, timeout: float = 300):
    import subprocess

    try:
        return subprocess.run(  # noqa: S603 - a path this module built
            [str(interpreter)] + [str(item) for item in arguments],
            capture_output=True, text=True,
            env={**os.environ, **worker_environment()}, timeout=timeout)
    except Exception as exc:
        logger.warning("Model Chain: could not %s — %s: %s", what, exc.__class__.__name__,
                       exc)
        raise PocketError(f"The staged PocketTTS runtime could not be started "
                          f"({exc.__class__.__name__}). Nothing was installed.") from None


def _read_recipe(entry: Bundle) -> dict:
    """PocketTTS's own configuration for this model, out of the installed wheel.

    Run inside the isolated runtime, because that is the only place the document
    exists: upstream ships ``pocket_tts/config/<language>.yaml`` in its wheel and
    the WebUI's own Python has no PocketTTS to read it from.

    Taken from upstream rather than transcribed into this repository's manifest.
    The document describes the *model*: layer counts, dtypes, the frame rate, the
    tokenizer's kind, the sample rate. A copy here would be a copy somebody has
    to update whenever a model revision changes any of them, and nothing would
    notice if it went stale. What this repository owns is the three locations in
    it, and :func:`_write_local_config` is where those are replaced.
    """
    interpreter = runtime_python()
    if interpreter is None:
        raise PocketError("The isolated PocketTTS runtime is not installed, so its model "
                          "configuration could not be read. Install the runtime first.")
    result = _run_staged(interpreter,
                         [paths.pocket_worker_script(), "--recipe", entry.identifier],
                         "read the PocketTTS model configuration", timeout=300)
    try:
        report = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        logger.warning("Model Chain: the staged PocketTTS runtime said %s",
                       models._quote(result.stdout))
        raise PocketError("The PocketTTS runtime did not report its model configuration. "
                          "Nothing was installed.") from None
    if not report.get("ok") or not isinstance(report.get("recipe"), dict):
        logger.warning("Model Chain: the PocketTTS runtime could not describe its model "
                       "— %s", report.get("error") or "no reason reported")
        why = _without_paths(report.get("error")) or "no reason reported"
        raise PocketError(f"The PocketTTS runtime could not describe its model ({why}). "
                          f"Nothing was installed.")
    return report["recipe"]


LOCATIONS = {
    "weights_path": ("weights_path",),
    "weights_path_without_voice_cloning": ("weights_path_without_voice_cloning",),
    "tokenizer_path": ("flow_lm", "lookup_table", "tokenizer_path"),
}
"""Where each artifact this repository installs belongs in upstream's config.

Three, and the manifest's ``config`` block is keyed by the same three names. The
tokenizer's is three levels down, which is why this is a table of paths rather
than a dictionary update: a merge at the top level would leave upstream's
``hf://`` tokenizer location in place, and the one call in the worker that could
reach the network is the one that resolves it (I-PKT-20).
"""


def _place(found: dict, where: tuple, value) -> None:
    """Set one nested key, building the dictionaries on the way down."""
    node = found
    for name in where[:-1]:
        step = node.get(name)
        if not isinstance(step, dict):
            step = {}
            node[name] = step
        node = step
    node[where[-1]] = value


def _write_local_config(entry: Bundle) -> None:
    """The local-only model config the worker loads. Section 25, as a file.

    Upstream's package accepts ``hf://`` and ``https://`` locations and imports
    hub utilities to resolve them. That behaviour belongs to an installer, not to
    production inference: the worker is handed a config whose every path names a
    file this process already downloaded, hashed and promoted, so there is no
    location in it for anything to resolve and no network for it to resolve
    against (I-PKT-20).

    It is upstream's own document with three paths replaced, and not a document
    of this repository's own design, because upstream is what opens it:
    ``TTSModel.load_model`` takes a path to a YAML file in its schema, validates
    it with a model that forbids unknown keys, and reads the architecture out of
    it. Anything this repository wants to tell the worker that is not part of
    that schema goes over the wire in :func:`worker_config` instead.

    Written as JSON rather than as YAML, because JSON is a subset of YAML,
    ``yaml.safe_load`` reads it either way, and this repository has a JSON writer
    everywhere and a YAML writer nowhere.

    ``weights_path`` is the interesting one. Upstream loads the cloning-capable
    weights from it and falls back to ``weights_path_without_voice_cloning`` when
    it cannot -- but only when the *fetch* fails, and a local path that is not
    there does not fail there, it fails later and fatally. So a machine without
    the gated half gets a config whose ``weights_path`` is the public file: the
    engine speaks, and cloning is refused with a sentence rather than the model
    refusing to load at all (section 23.3).
    """
    root = paths.pocket_model_root(entry.identifier)
    stored = _read_json(paths.pocket_upstream_config(entry.identifier))
    if not stored:
        # Written by :func:`install_model` from the runtime, so the only way to
        # be here is a half-installed tree -- the gated weights adopted from a
        # folder before the model itself, say. Not fatal: the worker refuses a
        # config it cannot find with a sentence naming the remedy, and the next
        # model install writes both files. Loud, because it is a state nothing
        # else reports.
        logger.warning("Model Chain: PocketTTS has no upstream model configuration to "
                       "localise, so its local config was not written; install the "
                       "PocketTTS model")
        return
    found = json.loads(json.dumps(stored))
    names = dict(entry.config or {})
    public = str(names.get("weights_path_without_voice_cloning") or "")
    for key, where in LOCATIONS.items():
        name = str(names.get(key) or "")
        candidate = (root / name) if name else None
        if candidate is not None and candidate.is_file():
            _place(found, where, str(candidate))
            continue
        if key == "weights_path" and public and (root / public).is_file():
            # No gated half on this machine. Pointed at the public weights
            # rather than left naming a file that is not there, because upstream
            # only falls back when a *download* fails.
            _place(found, where, str(root / public))
            continue
        # Refused rather than written as null. Upstream's schema makes the
        # tokenizer's location required and treats a null ``weights_path`` as
        # "load nothing", so a config with a hole in it is a pydantic traceback
        # or a silently uninitialised model -- neither of which says which file
        # is missing.
        raise PocketError(f"PocketTTS's model directory has no "
                          f"{name or key.replace('_', ' ')}, so its local configuration "
                          f"was not written. Install the PocketTTS model again.")
    located = sorted(key for key, value in _flatten(found)
                     if str(value).startswith(("hf://", "http://", "https://")))
    if located:
        # A location this build does not know how to replace, which means the
        # manifest's ``config`` block and upstream's document have drifted apart.
        # A broken extension rather than a broken installation, so the sentence
        # says what is on disk rather than claiming a rollback that did not
        # happen: the artifacts are promoted and only this file is missing.
        raise PocketError(f"PocketTTS's local configuration still names a network location "
                          f"for {located[0]}, so it was not written. The model files are "
                          f"installed; PocketTTS will not start until this build is "
                          f"corrected.")
    try:
        _write_json(paths.pocket_model_config(entry.identifier), found)
    except OSError:
        logger.debug("Model Chain: could not write the local PocketTTS config",
                     exc_info=True)


def _flatten(found, prefix: str = ""):
    """Every leaf in a nested document, as ``(dotted key, value)`` pairs.

    Depth-first and total, because the check it feeds is about the whole
    document rather than the three keys this module replaces: upstream owns
    everything else in it, and a location that appeared somewhere new would be a
    location the worker could resolve (I-PKT-20).
    """
    if isinstance(found, dict):
        for key, value in found.items():
            yield from _flatten(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(found, (list, tuple)):
        for value in found:
            yield from _flatten(value, prefix)
    else:
        yield prefix, found


def uninstall() -> Status:
    """Remove the PocketTTS runtime and model. Saved voices are kept.

    Kept because a custom voice's retained recording is the durable source that
    a later Pocket can rebuild a state from (I-PKT-15), and because removing an
    engine's runtime is not somebody saying they no longer want the voices they
    recorded.
    """
    if engines.active() == ENGINE:
        raise PocketError("Select another text-to-speech engine before removing PocketTTS.")
    try:
        import mc_voice_pocket_runtime as runtime

        runtime.stop("PocketTTS is being removed")
    except Exception:
        logger.debug("Model Chain: could not stop PocketTTS before removing it",
                     exc_info=True)
    for root in (paths.pocket_runtime_root(), paths.pocket_models_root(),
                 paths.pocket_official_root(), paths.pocket_staging_root(),
                 paths.pocket_preview_root()):
        if paths.pocket_inside(root):
            shutil.rmtree(root, ignore_errors=True)
    logger.info("Model Chain: PocketTTS's runtime and model were removed; saved voices "
                "were kept")
    return status()


# --------------------------------------------------------------------------- #
# The voice library
# --------------------------------------------------------------------------- #


def _read() -> dict:
    try:
        found = json.loads(paths.pocket_registry_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": REGISTRY_SCHEMA, "voices": []}
    if not isinstance(found, dict) or not isinstance(found.get("voices"), list):
        logger.warning("Model Chain: the PocketTTS voice registry could not be read and "
                       "was ignored")
        return {"schema": REGISTRY_SCHEMA, "voices": []}
    return found


def _write(found: dict) -> None:
    """Replace the registry atomically. A half-written registry is a lost library.

    Written beside the real file and renamed, on the same filesystem: a registry
    truncated by a power cut would leave prepared voices on disk that nothing can
    name.
    """
    path = paths.pocket_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.new")
    staging.write_text(json.dumps(found, indent=2), encoding="utf-8")
    os.replace(staging, path)


def _records() -> list:
    found = []
    for item in _read().get("voices") or ():
        if not isinstance(item, dict) or not item.get("uuid"):
            continue
        found.append(item)
    return sorted(found, key=lambda item: str(item.get("created_at") or ""))


def _out(item: dict) -> dict:
    """One custom voice record as the facade and the UI see it.

    Qualified id, no paths, no fingerprint internals beyond a short opaque
    string. A display name is metadata and never a filename (section 45), and
    nothing here can be turned back into one.
    """
    identifier = str(item.get("uuid") or "")
    current = status().fingerprint
    states = set(_state_fingerprints(identifier))
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
        "accent": "",
        "source_seconds": round(float(item.get("source_seconds") or 0.0), 1),
        "created_at": str(item.get("created_at") or ""),
        "fingerprint": current[:12],
        # Compatible means "there is a derived state for the model and precision
        # in force right now", not "this voice was made by this build". A clone
        # prepared six model revisions ago whose state was rebuilt this morning
        # is perfectly usable, and a layout with one state per fingerprint is
        # what makes that a question worth asking (I-PKT-18, section 39).
        "compatible": bool(current) and current in states,
        "has_source": _clone_file(identifier, paths.POCKET_REFERENCE_FILENAME).is_file(),
        "states": len(states),
        "slot": None,
    }


def _official_out(voice: OfficialVoice, entry: Bundle) -> dict:
    """One manifest voice as the facade and the UI see it.

    Official voices cannot be renamed or deleted, and they say so here rather
    than being refused later: a Delete button that always fails is worse than no
    Delete button (section 10).
    """
    return {
        "id": f"{ENGINE}:official:{voice.identifier}",
        "display_name": voice.display_name,
        "label": voice.display_name,
        "type": "official",
        "engine": ENGINE,
        "official": True,
        "editable": False,
        "deletable": False,
        "language": voice.language or entry.language,
        "accent": voice.accent,
        "source_seconds": 0.0,
        "created_at": "",
        "fingerprint": "",
        # An official state ships prepared for its model. It is compatible when
        # it is installed for the model in force, which is a file question and
        # not a fingerprint one.
        "compatible": (paths.pocket_official_root(entry.identifier)
                       / f"{voice.identifier}{paths.POCKET_STATE_SUFFIX}").is_file(),
        "has_source": False,
        "states": 1,
        "slot": None,
    }


def _clone_file(identifier: str, name: str) -> Path:
    return paths.pocket_clone_file(identifier, name)


def _state_fingerprints(identifier: str) -> list:
    """Which model fingerprints this custom voice has a derived state for."""
    try:
        root = paths.pocket_clone_states_root(identifier)
    except ValueError:
        return []
    if not root.is_dir():
        return []
    return [path.stem for path in root.glob(f"*{paths.POCKET_STATE_SUFFIX}")]


def official() -> list:
    """The reviewed official voices, from the manifest rather than from upstream.

    A manifest read rather than a network discovery event (section 10). Adding a
    voice is a change to this repository with its source, licence and
    attribution written down; it is not something that happens because upstream
    published one.
    """
    try:
        entry = bundle()
    except PocketError:
        return []
    return [_official_out(voice, entry) for voice in entry.voices]


def custom() -> list:
    return [_out(item) for item in _records()]


def entries() -> list:
    """Every Pocket voice this installation has. Official first, then custom."""
    return official() + custom()


def lookup(voice_id: str):
    if not engines.belongs(voice_id, ENGINE):
        return None
    wanted = engines.native(voice_id)
    for entry in entries():
        if engines.native(entry["id"]) == wanted:
            return entry
    return None


def _uuid_of(voice_id: str) -> str:
    return str(voice_id or "").split(":")[-1]


def default_id() -> str:
    """The configured Pocket default, qualified, or ``""``.

    From Pocket's own registry, falling back to the option an earlier build may
    have written, and then to the manifest's reviewed default voice. It never
    falls back to another engine (I-PKT-2): an installation with no Pocket voice
    at all answers with nothing, and the surface says so.
    """
    stored = str(_read().get("default") or "").strip()
    if stored and lookup(stored) is not None:
        return stored
    legacy = str(_setting(OPT_VOICE) or "").strip()
    if legacy:
        legacy = engines.qualify(legacy, ENGINE)
        if lookup(legacy) is not None:
            return legacy
    try:
        wanted = str((manifest().get("defaults") or {}).get("voice") or "")
    except PocketError:
        wanted = ""
    if wanted:
        candidate = f"{ENGINE}:official:{wanted}"
        if lookup(candidate) is not None:
            return candidate
    found = entries()
    return found[0]["id"] if found else ""


def default_entry():
    wanted = default_id()
    return lookup(wanted) if wanted else None


def set_default(voice_id: str) -> dict:
    """Commit a new Pocket default. Cannot be given another engine's voice.

    Written to Pocket's registry and to nothing else. Section 28: the default
    voice changes immediately, does not require Apply Settings, and has no
    second copy on a Forge settings page that Apply could put back.
    """
    entry = lookup(voice_id)
    if entry is None:
        raise PocketError("That is not a PocketTTS voice.")
    found = _read()
    found["default"] = entry["id"]
    found["schema"] = REGISTRY_SCHEMA
    _write(found)
    logger.info("Model Chain: the PocketTTS default voice is now %s", entry["id"])
    return entry


def resolve(voice_id: str = "") -> tuple:
    """``(qualified id, entry)`` for a stable id, or the Pocket default.

    Raises when Pocket has no usable voice at all, which is a different failure
    from "that voice is gone" and is reported differently: one asks the user to
    install or create a voice, the other quietly falls back to the default and
    warns (I-PKT-2 -- neither ever reaches another engine).
    """
    entry = lookup(voice_id) if voice_id else None
    if entry is None:
        entry = default_entry()
    if entry is None:
        raise PocketError("PocketTTS has no voice to speak with yet. Install its official "
                          "voices, or clone one from a recording.")
    # Pocket's address for a voice is its stable id, so ``_handle`` looks
    # redundant and is not: it is the name every engine's entry answers to, so
    # the shared turn carries one opaque thing rather than knowing which engine
    # uses a number (section 8). Stripped before any payload.
    entry["_handle"] = entry["id"]
    return entry["id"], entry


def rename(voice_id: str, display_name: str) -> dict:
    """Change what a custom Pocket voice is called, and nothing else.

    The id, the UUID, the directory and every derived state are untouched, so a
    rename invalidates no cache and breaks no character: a character stores the
    stable id, and the stable id is not the name (I-PKT-15).
    """
    entry = lookup(voice_id)
    if entry is None:
        raise PocketError("That is not a PocketTTS voice.")
    if entry["official"]:
        raise PocketError("An official PocketTTS voice cannot be renamed.")
    name = check_name(display_name)
    identifier = _uuid_of(voice_id)
    found = _read()
    for item in found.get("voices") or ():
        if str(item.get("uuid")) == identifier:
            item["display_name"] = name
            break
    _write(found)
    logger.info("Model Chain: a PocketTTS voice was renamed")
    return lookup(voice_id)


def delete(voice_id: str) -> dict:
    """Remove a custom Pocket voice, its recording and every derived state.

    All of it, because a retained recording of somebody with no registry entry
    to name it is a recording nothing will ever delete. Contained first: every
    path removed here has to resolve under Pocket's own root, which is
    containment and also I-PKT-3 -- deleting a Pocket voice cannot reach a Sopro
    file, because there is no expression here that names one.
    """
    entry = lookup(voice_id)
    if entry is None:
        raise PocketError("That is not a PocketTTS voice.")
    if entry["official"]:
        raise PocketError("An official PocketTTS voice cannot be deleted.")
    identifier = _uuid_of(voice_id)
    found = _read()
    found["voices"] = [item for item in (found.get("voices") or ())
                       if str(item.get("uuid")) != identifier]
    if str(found.get("default") or "") == entry["id"]:
        found["default"] = ""
    _write(found)

    try:
        import mc_voice_pocket_runtime as runtime

        runtime.refresh_catalog(catalog(), forget=[entry["id"]])
    except Exception:
        logger.debug("Model Chain: could not refresh the PocketTTS catalogue after a "
                     "delete", exc_info=True)

    root = paths.pocket_clone_root(identifier)
    if not paths.pocket_inside(root):
        raise PocketError("That voice's files are not where Voice Chat keeps them, so "
                          "nothing was deleted.")
    shutil.rmtree(root, ignore_errors=True)
    if root.exists():
        logger.warning("Model Chain: a deleted PocketTTS voice left files behind")
    logger.info("Model Chain: a PocketTTS voice was deleted")
    return entry


def capacity() -> dict:
    """How many voices exist. Pocket has no bank and therefore no slot limit."""
    return {"used": len(_records()), "total": None, "free": None}


def warnings() -> list:
    """What is wrong that a user can see and act on. Never repairs anything.

    Three states make a Pocket voice unusable and each has a different remedy,
    so each gets its own sentence: a stale derived state with a retained
    recording can be rebuilt, a stale one without cannot, and an official state
    that is not installed is an install rather than a rebuild. Nothing here
    fixes anything -- a background status poll that rebuilt a voice would be a
    poll that ran a model (I-PKT-26, T-PKT-CLONE-11).
    """
    found = []
    current = status().fingerprint
    for entry in custom():
        if entry["compatible"]:
            continue
        if entry["has_source"]:
            found.append(f"The PocketTTS voice {entry['display_name']!r} has no prepared "
                         f"data for the model and precision now selected. Rebuild it in "
                         f"Settings → Voice Chat; its recording is still here.")
        else:
            found.append(f"The PocketTTS voice {entry['display_name']!r} has no prepared "
                         f"data for the model now selected and its recording is gone, so "
                         f"it has to be created again.")
    try:
        entry = bundle()
    except PocketError:
        return found
    if entry.voices and not _official_voices_ready(entry):
        found.append("PocketTTS's official voices are not installed, so it can only speak "
                     "with voices you cloned. Install them in Settings → Voice Chat.")
    if current and not entries():
        found.append("PocketTTS has no voices yet, so Voice Chat has nothing to speak "
                     "with. Install its official voices, or use Clone voice.")
    return found


def catalog() -> dict:
    """The worker's view of the library: stable id to the local files it needs.

    Built here, in the parent, from the registry and the manifest -- so the
    worker is handed a mapping of verified local paths and never a repository
    id, a URL or anything a browser supplied (I-PKT-20, section 45).
    """
    found = {}
    try:
        entry = bundle()
    except PocketError:
        return found
    root = paths.pocket_official_root(entry.identifier)
    for voice in entry.voices:
        path = root / f"{voice.identifier}{paths.POCKET_STATE_SUFFIX}"
        if path.is_file():
            found[f"{ENGINE}:official:{voice.identifier}"] = {
                "kind": "official", "state": str(path)}
    fingerprint = status().fingerprint
    for item in _records():
        identifier = str(item.get("uuid") or "")
        try:
            state = paths.pocket_clone_state(identifier, fingerprint)
        except ValueError:
            continue
        if state.is_file():
            found[f"{ENGINE}:clone:{identifier}"] = {"kind": "clone", "state": str(state)}
    return found


def check_name(name: str) -> str:
    """A display name, or a refusal that says what is wrong with this one."""
    text = str(name or "").strip()
    if not text:
        raise PocketError("Give the voice a name.")
    if len(text) > MAX_NAME_CHARS:
        raise PocketError(f"That name is longer than {MAX_NAME_CHARS} characters.")
    if not _NAME_OK.match(text):
        raise PocketError("A voice name can hold letters, numbers, spaces, apostrophes, "
                          "hyphens, full stops, brackets and ampersands.")
    existing = {entry["display_name"].casefold() for entry in entries()}
    if text.casefold() in existing:
        raise PocketError(f"There is already a PocketTTS voice called {text!r}.")
    return text


def test_text() -> str:
    """What Test plays. Shared with the other engines because it is a property
    of the control rather than of a model."""
    import mc_voice_registry as registry

    try:
        return registry.test_text() or DEFAULT_TEST_TEXT
    except Exception:
        return DEFAULT_TEST_TEXT


def set_test_text(text: str) -> str:
    import mc_voice_registry as registry

    return registry.set_test_text(text)


# --------------------------------------------------------------------------- #
# What the facade asks this engine about itself
# --------------------------------------------------------------------------- #


def capabilities() -> dict:
    """What Pocket can do, as behaviour rather than decoration. Section 8.

    ``interrupt_mode`` is the one that matters and the one that is not a
    preference. Released PocketTTS 3.0.2 has no safe cooperative cancellation
    for an abandoned stream -- upstream's own pull request for one is open, and
    it says in as many words that draining to completion was the correct
    behaviour before it -- and its model is documented as not thread-safe, so
    starting the next generation while the old one is alive would be wrong. So
    Stop here is ``drain_unit``: silence now, ready shortly, and no claim that
    native compute was aborted (I-PKT-11, I-PKT-14, section 21).

    The declared mode comes from the runtime module rather than from a constant
    here, so that adopting a merged upstream cancellation is one change in one
    place (section 21.7).

    ``clone_preview`` is a capability rather than a readiness: Pocket *can*
    clone. Whether the gated weights are installed on this machine is
    :func:`public_status`'s ``cloning_ready``, and the panel needs both -- one
    decides whether the route exists, the other decides what it says.
    """
    mode = "drain_unit"
    try:
        import mc_voice_pocket_runtime as runtime

        mode = str(runtime.declared_interrupt_mode() or mode)
    except Exception:
        logger.debug("Model Chain: could not read PocketTTS's interrupt mode",
                     exc_info=True)
    return {"clone_preview": True, "rebuild": True, "engine_settings": True,
            "starter_voices": False, "voice_lab": False, "interrupt_mode": mode}


def refusals() -> tuple:
    """The exception types this adapter raises to *refuse* rather than fail."""
    return (PocketError,)


def clone_hints() -> dict:
    """What the clone form should suggest, from the engine rather than a guess.

    Read every time and never hardcoded in the page, because the ideal reference
    length is a release measurement (GATE P-CLONE-1) and a number baked into
    JavaScript is a number that goes stale the first time somebody measures it
    (section 26.1).
    """
    return {"min_seconds": MIN_REFERENCE_SECONDS,
            "ideal_seconds": IDEAL_REFERENCE_SECONDS,
            "max_seconds": MAX_REFERENCE_SECONDS}


def public_status() -> dict:
    """Pocket's operational state, in the shape every engine answers with.

    The common subset section 31 asks for, plus ``block`` -- the engine-owned
    part the status payload publishes under this engine's own id, and the only
    part a page scoped to another engine never sees.

    ``draining`` and ``engine_busy`` are the two fields nothing else in this
    feature has needed before, and they are read by one piece of engine-neutral
    browser code rather than by a Pocket branch in it (I-PKT-28): playback is
    already silent when they matter, and what they decide is whether the
    Play control may start something new yet.
    """
    found = status()
    live = {}
    try:
        import mc_voice_pocket_runtime as runtime

        live = runtime.status() or {}
    except Exception:
        logger.debug("Model Chain: could not read the PocketTTS runtime state",
                     exc_info=True)
    engine_state = {}
    try:
        import mc_voice_pocket_runtime as runtime

        engine_state = runtime.engine() or {}
    except Exception:
        logger.debug("Model Chain: could not read the PocketTTS engine state",
                     exc_info=True)
    mode = str(live.get("interrupt_mode") or capabilities()["interrupt_mode"])
    return {
        "installed": found.ready,
        # Whether this engine *can* speak, and not whether it happens to be
        # speaking. The two were folded together here and the status payload
        # publishes this value as ``ready`` and ``tts_ready`` -- so for the
        # whole of every reply the browser was being told Voice Chat was not
        # set up. ``engine_busy`` is the field for "the lane is occupied", and
        # it is reported separately two lines down.
        "ready": found.ready,
        "tts_ready": found.ready,
        "message": found.message,
        "worker_resident": bool(engine_state.get("loaded")),
        "engine_busy": bool(live.get("busy")),
        "draining": bool(live.get("draining")),
        "interrupt_mode": mode,
        "block": {
            "installed": found.ready,
            "platform_supported": found.platform_supported,
            "runtime_ready": found.runtime_ready,
            "speech_model_ready": found.speech_model_ready,
            "official_voices_ready": found.official_voices_ready,
            "cloning_ready": found.cloning_ready,
            "runtime_message": found.runtime_message,
            "model_message": found.model_message,
            "cloning_message": found.cloning_message,
            "pinned": pinned(),
            "model_id": found.model_id,
            "fingerprint": found.fingerprint,
            "settings": engine_settings(),
            "defaults": engine_state.get("defaults") or {},
            "warnings": warnings(),
            "busy": bool(live.get("busy")),
            "draining": bool(live.get("draining")),
            "interrupt_mode": mode,
            "thread_policy": thread_policy(),
        },
    }


# --------------------------------------------------------------------------- #
# Creating a voice
# --------------------------------------------------------------------------- #


ENVELOPE = reference.Envelope(
    engine=LABEL,
    minimum_seconds=MIN_REFERENCE_SECONDS,
    maximum_seconds=MAX_REFERENCE_SECONDS,
    maximum_bytes=MAX_REFERENCE_BYTES,
    target_rate=TARGET_RATE,
    minimum_peak=MIN_PEAK,
)
"""What Pocket will accept in a recording, and what to call itself when it says no.

Shorter than Sopro's window and deliberately below upstream's own thirty-second
truncation ceiling: fifteen seconds is a bound this build can defend, thirty is
a number upstream happens to cut at. GATE P-CLONE-1 is what would change it, and
:func:`clone_hints` is why changing it changes the form without changing the
page.
"""


def normalize_reference(data: bytes) -> tuple:
    """A recording as canonical mono 24 kHz PCM16, or a refusal that says why.

    Run by :func:`mc_voice_reference.normalize` against :data:`ENVELOPE` and
    refused in Pocket's own voice. The decoding, the downmix and the resampling
    are shared because they are the same arithmetic whichever engine asked;
    the window and the name in the sentence are Pocket's because they are not.
    """
    return reference.normalize(data, ENVELOPE, PocketError)


_preview_lock = threading.Lock()
_preview = {}
"""The one voice that has been built but not yet kept.

Everything up to the registry write leaves nothing anybody can see, which is why
the failure path here is a single ``rmtree``. A directory with a recording and a
derived state in it and no registry entry is not a half-saved voice -- it is a
voice that does not exist yet, and it costs one directory to stop existing
(I-PKT-17).

So creating and keeping are two steps with a person in between. The audition
somebody hears is the one this exact preparation produced, played back from the
state that was written and read *back off the disk*, rather than a re-synthesis
from a voice that has already been written down -- which is the only version of
"preview before you save" that is actually a preview.

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

    An empty token discards whatever is pending, which is what the shutdown, the
    engine-switch and the supersede paths want; a non-empty one has to match, so
    a stale browser tab cannot delete the preview a newer one is looking at.
    """
    with _preview_lock:
        if not _preview:
            return False
        if token and token != _preview.get("token"):
            return False
        directory = _preview.get("directory") or ""
        name = _preview.get("name") or ""
        _preview.clear()
    if directory:
        root = Path(directory)
        # The same guard the failure path uses: never remove a directory this
        # module did not build inside its own root.
        if paths.pocket_inside(root):
            shutil.rmtree(root, ignore_errors=True)
        logger.info("Model Chain: a PocketTTS voice preview was discarded — %s",
                    name or "?")
    return True


def prepare_preview(display_name: str, wav: bytes) -> dict:
    """Build a voice and audition it, without writing it down. Section 26.2.

    Every step can fail without leaving anything behind:

        1  the name is validated here;
        2  the recording is validated and normalized here;
        3  a server-minted token becomes a directory -- never the display name,
           never anything the browser sent (section 45);
        4  the normalized WAV is retained beside it, so a later compatible
           Pocket can rebuild without asking for a new recording (I-PKT-15);
        5  the worker extracts the voice state, exports it to safetensors,
           reads it back *from the file it just wrote*, and synthesises the
           audition text from that exact saved state;
        6  nothing is registered.

    Step 5 is not ceremony. A preparation that returned without an exception
    proves nothing about whether the file it wrote can be read back after a
    restart, and reading it back is the only way to know (T-PKT-CLONE-4).

    :func:`save_preview` is the only step that writes anything a user can
    afterwards see.
    """
    name = check_name(display_name)
    found = status()
    if not found.ready:
        raise PocketError("PocketTTS is not installed, so it cannot create a voice.")
    if not found.cloning_ready:
        raise PocketError(found.cloning_message or
                          "PocketTTS's voice-cloning weights are not installed.")

    # Before the work, not after: a preview that failed halfway would otherwise
    # leave the previous one pending and the panel showing a Save button for a
    # voice the user has already replaced.
    discard_preview()

    token = uuid.uuid4().hex
    normalized, seconds = normalize_reference(wav)
    root = paths.pocket_preview_dir(token)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    made = None
    try:
        (root / paths.POCKET_REFERENCE_FILENAME).write_bytes(normalized)

        import mc_voice_pocket_runtime as runtime

        made = runtime.prepare_voice(
            root=str(root), voice_id=f"{ENGINE}:preview:{token}",
            wav_bytes=normalized, seconds=min(seconds, MAX_REFERENCE_SECONDS),
            audition=test_text())
    except Exception:
        # Nothing is registered, so the only thing to undo is the directory.
        # Removed rather than left, because a directory with a WAV in it and no
        # registry entry is a recording of somebody that nothing will ever
        # delete.
        if paths.pocket_inside(root):
            shutil.rmtree(root, ignore_errors=True)
        raise

    with _preview_lock:
        _preview.update({
            "token": token,
            "directory": str(root),
            "name": name,
            "fingerprint": found.fingerprint,
            "model_id": found.model_id,
            "seconds": round(float(seconds), 2),
            "sample_rate": int(made.get("sample_rate") or TARGET_RATE),
            "audition_ms": int(made.get("audition_ms") or 0),
            "source_digest": hashlib.sha256(normalized).hexdigest()[:16],
        })
    logger.info("Model Chain: a PocketTTS voice was prepared for preview — %.1f s "
                "reference, audition in %d ms; it is not saved yet",
                seconds, int(made.get("audition_ms") or 0))
    return {"token": token, "name": name, "seconds": round(float(seconds), 2),
            "audio": made.get("audio") or b""}


def save_preview(token: str) -> dict:
    """Keep the pending preview. The only step that writes anything visible.

    The token has to match. Without it a browser sitting on an old panel could
    save a voice the user had already replaced with another preview, which is
    the one way this split could produce a voice nobody chose (T-PKT-CLONE-8).

    The move is a rename inside Pocket's own root, so a save is atomic in the
    way that matters: either ``clones/<uuid>`` exists with everything in it, or
    it does not exist at all. The registry write comes after, because a
    registry entry naming a directory that is not there yet is the one ordering
    that produces a voice nothing can speak.
    """
    with _preview_lock:
        if not _preview:
            raise PocketError("There is no voice waiting to be saved. Create one first.")
        if str(token or "") != _preview.get("token"):
            raise PocketError("That preview is no longer the one waiting to be saved. "
                              "Create the voice again.")
        pending = dict(_preview)
        identifier = uuid.uuid4().hex
        source = Path(pending["directory"])
        target = paths.pocket_clone_root(identifier)
        if not (paths.pocket_inside(source) and paths.pocket_inside(target)):
            raise PocketError("That preview is not where Voice Chat keeps them, so it was "
                              "not saved.")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        # Cleared here and not a line earlier. Everything above this can still
        # refuse, and a record cleared before the refusal would leave the
        # recording somebody actually made sitting in a directory nothing knows
        # the name of any more: Discard would say there is nothing pending, and
        # the one ``rmtree`` this module relies on would have nothing to remove.
        # The claim is inside the lock for the same reason -- two Saves racing on
        # one token must not both reach the move, because the loser's rename
        # would fail on a directory the winner had already taken.
        _preview.clear()

    # The state the worker exported is the preview's; it becomes this clone's
    # state for the fingerprint it was prepared under. One file per fingerprint
    # from the first save, so a later model change adds a state rather than
    # overwriting the only one that worked (section 39).
    made = target / paths.POCKET_PREVIEW_STATE_FILENAME
    if made.is_file():
        states = paths.pocket_clone_states_root(identifier)
        states.mkdir(parents=True, exist_ok=True)
        os.replace(made, paths.pocket_clone_state(identifier, pending["fingerprint"]))

    record = {
        "uuid": identifier,
        "schema": REGISTRY_SCHEMA,
        "display_name": pending["name"],
        "engine": ENGINE,
        "language": "",
        "model_id": pending.get("model_id") or "",
        "source_seconds": pending["seconds"],
        "source_rate": pending["sample_rate"],
        "source_digest": pending.get("source_digest") or "",
        "voice_state_schema": STATE_SCHEMA,
        "attribution": "user-provided reference",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(target / paths.POCKET_METADATA_FILENAME, record)
    stored = _read()
    stored.setdefault("voices", []).append(record)
    stored["schema"] = REGISTRY_SCHEMA
    _write(stored)

    try:
        import mc_voice_pocket_runtime as runtime

        runtime.refresh_catalog(catalog())
    except Exception:
        logger.debug("Model Chain: could not refresh the PocketTTS catalogue after a save",
                     exc_info=True)

    # Deliberately *not* made the default. Section 26.3: saving a voice is not
    # the same decision as speaking with it, and a Save that silently changed
    # what every character sounds like would be a Save nobody could undo without
    # knowing what the default used to be.
    logger.info("Model Chain: a PocketTTS voice was saved — %.1f s reference",
                pending["seconds"])
    return {"voice": lookup(f"{ENGINE}:clone:{identifier}")}


def rebuild(voice_id: str) -> dict:
    """Prepare a custom voice's state again, for the model in force now.

    Transactional in the way that matters: the new state is written into a
    staging directory, validated by a production audition, and only then moved
    into ``states/<fingerprint>.safetensors`` -- so a rebuild that fails leaves
    the voice exactly as stale as it was rather than leaving it broken. The
    states that already exist are not touched, so switching the model back makes
    the old one usable again without another rebuild (section 27, section 39).

    Never automatic. A background status poll that rebuilt a voice would be a
    poll that started a model and spent a minute of somebody's CPU on a decision
    they did not make (T-PKT-CLONE-11).
    """
    entry = lookup(voice_id)
    if entry is None:
        raise PocketError("That PocketTTS voice does not exist.")
    if entry["official"]:
        raise PocketError("An official PocketTTS voice is installed rather than rebuilt.")
    identifier = _uuid_of(voice_id)
    source = _clone_file(identifier, paths.POCKET_REFERENCE_FILENAME)
    if not source.is_file():
        raise PocketError(f"{entry['display_name']} has no retained recording, so it has "
                          f"to be created again from a new one.")
    found = status()
    if not found.ready:
        raise PocketError("PocketTTS is not installed, so it cannot rebuild a voice.")
    if not found.cloning_ready:
        raise PocketError(found.cloning_message or
                          "PocketTTS's voice-cloning weights are not installed.")

    root = paths.pocket_clone_root(identifier)
    staging = root.parent / f"{identifier}.rebuilding"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        import mc_voice_pocket_runtime as runtime

        # A namespace of its own, and never this voice's real id. The worker
        # names the preparation in its live catalogue so it can audition from
        # the file it just wrote, and the real id would therefore be repointed
        # at ``staging`` -- a directory the ``finally`` below deletes. A reply
        # spoken in this voice in that window would fail on a path that is gone,
        # for a voice that was perfectly fine before the rebuild started.
        made = runtime.prepare_voice(
            root=str(staging), voice_id=f"{ENGINE}:rebuild:{identifier}",
            wav_bytes=source.read_bytes(), audition=test_text())
        candidate = staging / paths.POCKET_PREVIEW_STATE_FILENAME
        if not candidate.is_file():
            raise PocketError("PocketTTS did not export a voice state. Nothing changed.")
        states = paths.pocket_clone_states_root(identifier)
        states.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, paths.pocket_clone_state(identifier, found.fingerprint))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    stored = _read()
    for item in stored.get("voices") or ():
        if str(item.get("uuid")) == identifier:
            item["rebuilt_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            item["model_id"] = found.model_id
            break
    _write(stored)
    try:
        import mc_voice_pocket_runtime as runtime

        runtime.refresh_catalog(catalog(), forget=[entry["id"]])
    except Exception:
        logger.debug("Model Chain: could not refresh the PocketTTS catalogue after a "
                     "rebuild", exc_info=True)
    logger.info("Model Chain: a PocketTTS voice was rebuilt for the current model")
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
        found = json.loads(paths.pocket_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _settings_write(found: dict) -> None:
    """Replace Pocket's settings file atomically, like every other file here."""
    path = paths.pocket_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.new")
    staging.write_text(json.dumps(found, indent=2), encoding="utf-8")
    os.replace(staging, path)


def _setting(name: str):
    """One global Pocket setting, out of Pocket's own file.

    Pocket's settings were never host options, and this is the reason they will
    not become them: an option is a *component* on the settings page as well as
    a stored value, and "Apply settings" writes every component on that page
    back into the store. The page's copy is stamped when the page is built, so
    it knows nothing about a slider moved in the delivery panel since -- and
    putting the old value back is exactly what it would do. Sopro learned that
    the expensive way; Pocket starts on the other side of it (I-PKT-19).

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
        logger.debug("Model Chain: could not persist a PocketTTS setting", exc_info=True)
