"""The checked-in trust root for Voice Chat, and the only door to the Internet.

Every byte Voice Chat ever fetches is named in ``voice/managed-voice-models.json``
-- a URL, a byte count and a SHA-256 for each -- and this module is the only one
allowed to fetch any of them. Nothing downstream may reach the network: the
worker has no HTTP client, the runtime manager starts a process and talks to it
over a pipe, and the API routes move bytes between a browser and that pipe. So
"once it is installed, speech works with the Internet unplugged" is a property
of the import graph rather than a promise in a README.

Three things are installed, and they are installed in that order:

    runtime   the isolated CPU environment the worker runs in, provisioned from
              a complete pinned wheel closure for this platform and Python
    stt       Whisper Small INT8, three ONNX/text files
    tts       Kokoro-82M v1.0, one verified archive expanded into a bundle

Why the closure is complete
---------------------------
R2-2. A package manager that can reach an index is a package manager that can
install something nobody reviewed, and "the user clicked Download" does not make
a transitive dependency resolved at three in the morning trustworthy. So the
manifest names *every* wheel for each supported platform/Python pair, this module
downloads each one itself and checks its hash, and the installation step runs
``pip install --no-index --no-deps <local wheel> <local wheel>``: no index, no
resolver, no opportunity for a byte to arrive that was not verified here first.
``tests/test_voice_models.py`` puts a recording server in front of a fake index
and asserts it is never asked for anything.

Why some entries ship unpinned
------------------------------
The runtime closure is pinned exactly -- eight platform/Python combinations,
sixteen wheels, real sizes and real hashes read from PyPI at the revision this
was written against. The two *model* bundles are not: their artifacts live on
huggingface.co and on a GitHub release, and pinning a hash means downloading the
artifact and hashing it, which is a maintainer's job on a machine that can reach
those hosts. Until somebody runs ``tools/pin_voice_models.py``, those entries
carry ``"sha256": null`` and this module refuses to install them -- see
:meth:`Artifact.pinned` and :func:`install`. That is Gate 0 of the design intent
enforced in code: an unpinned bundle is not a bundle to download slowly, it is a
bundle nobody may download at all.

What is not here
----------------
No update check, no telemetry, no "fetch the tokenizer if it is missing", no
lazy model-hub call on first dictation. :func:`status` answers from the disk,
:func:`install` runs only because somebody pressed a button, and there is no
third path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import mc_voice_paths as paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


class VoiceError(RuntimeError):
    """Something a user can read and act on, rather than a traceback."""


SCHEMA = 1
"""The manifest schema this code understands. A file claiming a newer one is
refused rather than guessed at: a bundle installed by a misread manifest is a
bundle whose hashes mean nothing."""

CHUNK = 1024 * 256

TIMEOUT = 60.0

USER_AGENT = "ModelChain-VoiceChat"
"""Sent on the only requests this extension makes for speech. Named so a
maintainer reading a proxy log can tell which extension asked."""

SAFETY_MARGIN_BYTES = 256 * 1024 * 1024

_lock = threading.RLock()
_manifest_cache: dict | None = None
_progress: dict[str, dict] = {}
"""What an install is doing, for the Settings row and the status route.

Process-local and deliberately not persisted: "downloading, 41%" is a fact
about a running process, and a settings file that remembered it would describe
a download that stopped when the WebUI did."""


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Artifact:
    """One file this extension may fetch, and everything it must be.

    ``local_name`` is what it is called once installed, which is not always what
    the publisher calls it: ``small-encoder.int8.onnx`` becomes ``encoder.onnx``
    so the worker's paths do not have to know which Whisper tier is installed.
    """

    filename: str
    local_name: str
    url: str
    size: int | None
    sha256: str | None
    archive: str = ""
    strip_root: str = ""

    @property
    def pinned(self) -> bool:
        """Whether this artifact may be downloaded at all.

        A hash and a byte count, or nothing doing. There is no "download it and
        trust it" mode and there is no flag to add one: an artifact without a
        hash is an artifact whose contents this repository makes no claim
        about, and the whole install transaction is built on being able to make
        that claim.
        """
        return bool(self.sha256) and isinstance(self.size, int) and self.size > 0

    @property
    def approximate_bytes(self) -> int:
        return int(self.size or 0)


@dataclass(frozen=True)
class RuntimePlatform:
    """The complete wheel closure for one platform and one Python."""

    identifier: str
    system: str
    machines: tuple[str, ...]
    python: str
    artifacts: tuple[Artifact, ...]

    def matches(self, system: str, machine: str, python: str) -> bool:
        return (self.system == system
                and machine in self.machines
                and self.python == python)


@dataclass(frozen=True)
class VoiceModel:
    """One installable speech bundle."""

    identifier: str
    kind: str
    label: str
    engine: str
    artifacts: tuple[Artifact, ...]
    required_paths: tuple[str, ...] = ()
    voice: str = ""
    speaker_id: int = 0
    language: str = ""
    revision: str = ""
    license: str = ""
    attribution: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def pinned(self) -> bool:
        return all(item.pinned for item in self.artifacts)

    @property
    def total_bytes(self) -> int:
        return sum(item.approximate_bytes for item in self.artifacts)

    @property
    def wanted_paths(self) -> tuple[str, ...]:
        """What must exist inside an installed bundle for it to be one.

        For an ordinary file list that is the local names. For an archive it is
        the manifest's ``required_paths``, because the archive itself is not
        kept -- what is installed is what came out of it.
        """
        if self.required_paths:
            return self.required_paths
        return tuple(item.local_name for item in self.artifacts)


LOCAL_PINS_FILENAME = "managed-voice-models.local.json"
"""Pins a maintainer filled in on a machine that could reach the publishers.

Shipped unpinned is the safe state -- an artifact with no hash is one this
repository makes no claim about, and :meth:`Artifact.pinned` refuses to fetch
it. Pinning means downloading each artifact once and hashing it, which needs a
machine that can reach huggingface.co and github.com.

The pins land in a *separate, untracked* file rather than being written over the
shipped manifest, for a plain reason: a checked-in file edited in place turns
every later ``git pull`` into a merge conflict, and the first thing anybody does
with a merge conflict in a file full of hashes is take one side at random.
Overlaid by filename, and only onto artifacts that have no hash yet -- a pin
file cannot change a hash this repository already committed.
"""


def manifest(refresh: bool = False) -> dict:
    """The parsed, validated manifest. Read once and kept.

    Every failure below is a broken *extension*, not a broken installation, so
    the message says so: a user cannot fix a malformed trust root by pressing
    Download again.
    """
    global _manifest_cache

    with _lock:
        if _manifest_cache is not None and not refresh:
            return _manifest_cache
        _manifest_cache = _read_manifest(paths.manifest_path())
        return _manifest_cache


def local_pins_path() -> Path:
    return paths.manifest_path().with_name(LOCAL_PINS_FILENAME)


def _local_pins() -> dict:
    """``{filename: {"sha256": ..., "bytes": ...}}`` from the overlay, if any."""
    try:
        raw = json.loads(local_pins_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    found = {}
    for name, entry in (raw.get("artifacts") or {}).items():
        digest = str((entry or {}).get("sha256") or "").strip().casefold()
        size = (entry or {}).get("bytes")
        if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) \
                and isinstance(size, int) and size > 0:
            found[str(name)] = {"sha256": digest, "bytes": size}
    return found


def _read_manifest(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VoiceError(f"The Voice Chat manifest is missing from this extension "
                         f"({path}).") from exc
    except (OSError, ValueError) as exc:
        raise VoiceError(f"The Voice Chat manifest cannot be read ({exc}).") from exc
    if not isinstance(raw, dict):
        raise VoiceError("The Voice Chat manifest is not an object.")

    schema = raw.get("schema")
    if schema != SCHEMA:
        raise VoiceError(f"The Voice Chat manifest declares schema {schema!r}; this version "
                         f"of the extension understands {SCHEMA}.")

    runtime = raw.get("runtime")
    if not isinstance(runtime, dict):
        raise VoiceError("The Voice Chat manifest has no runtime section.")
    provider = str(runtime.get("provider") or "")
    if provider != "cpu":
        # I-1 stated where it can be checked. A manifest that named a GPU
        # provider would be a manifest asking this feature to do the one thing
        # it exists not to do.
        raise VoiceError(f"The Voice Chat runtime manifest asks for provider {provider!r}; "
                         f"only 'cpu' is supported.")

    platforms = []
    for entry in runtime.get("platforms") or []:
        platforms.append(_read_platform(entry))
    if not platforms:
        raise VoiceError("The Voice Chat manifest declares no runtime platforms.")

    models = {}
    for identifier, entry in (raw.get("models") or {}).items():
        models[identifier] = _read_model(identifier, entry)
    if not models:
        raise VoiceError("The Voice Chat manifest declares no models.")

    defaults = raw.get("defaults") or {}
    for kind in paths.KINDS:
        chosen = defaults.get(kind)
        if chosen not in models:
            raise VoiceError(f"The Voice Chat manifest's default {kind} model {chosen!r} is "
                             f"not in its own catalogue.")
        if models[chosen].kind != kind:
            raise VoiceError(f"The Voice Chat manifest's default {kind} model {chosen!r} is a "
                             f"{models[chosen].kind} model.")

    return {
        "version": int(raw.get("version") or 1),
        "runtime_version": str(runtime.get("version") or ""),
        "runtime_package": str(runtime.get("package") or ""),
        "runtime_import": str(runtime.get("import_name") or "sherpa_onnx"),
        "runtime_license": str(runtime.get("license") or ""),
        "platforms": tuple(platforms),
        "defaults": {kind: defaults[kind] for kind in paths.KINDS},
        "models": models,
    }


def _read_platform(entry) -> RuntimePlatform:
    if not isinstance(entry, dict):
        raise VoiceError("A Voice Chat runtime platform entry is not an object.")
    identifier = str(entry.get("id") or "").strip()
    system = str(entry.get("system") or "").strip().casefold()
    python_version = str(entry.get("python") or "").strip()
    machines = tuple(str(m).strip().casefold() for m in (entry.get("machines") or ()))
    if not identifier or not system or not python_version or not machines:
        raise VoiceError(f"Voice Chat runtime platform {identifier or '?'} is incomplete.")
    artifacts = tuple(_read_artifact(f"runtime {identifier}", item)
                      for item in (entry.get("artifacts") or ()))
    if not artifacts:
        raise VoiceError(f"Voice Chat runtime platform {identifier} names no artifacts, so its "
                         f"dependency closure is not complete.")
    for item in artifacts:
        if not item.pinned:
            raise VoiceError(f"Voice Chat runtime platform {identifier} has an unpinned "
                             f"artifact ({item.filename}).")
    return RuntimePlatform(identifier=identifier, system=system, machines=machines,
                           python=python_version, artifacts=artifacts)


def _read_model(identifier: str, entry) -> VoiceModel:
    if not isinstance(entry, dict):
        raise VoiceError(f"Voice Chat model {identifier} is not an object.")
    kind = str(entry.get("kind") or "").strip()
    if kind not in paths.KINDS:
        raise VoiceError(f"Voice Chat model {identifier} has kind {kind!r}.")
    artifacts = tuple(_read_artifact(identifier, item) for item in (entry.get("files") or ()))
    if not artifacts:
        raise VoiceError(f"Voice Chat model {identifier} names no files.")
    # Rejected here rather than at install time: a bundle id is used to build a
    # directory path, and the check belongs beside the parse that produced it.
    paths.bundle_root(kind, identifier)
    return VoiceModel(
        identifier=identifier,
        kind=kind,
        label=str(entry.get("label") or identifier),
        engine=str(entry.get("engine") or ""),
        artifacts=artifacts,
        required_paths=tuple(str(p) for p in (entry.get("required_paths") or ())),
        voice=str(entry.get("voice") or ""),
        speaker_id=int(entry.get("speaker_id") or 0),
        language=str(entry.get("language") or ""),
        revision=str(entry.get("revision") or ""),
        license=str(entry.get("license") or ""),
        attribution=str(entry.get("attribution") or ""),
    )


def _read_artifact(owner: str, entry) -> Artifact:
    if not isinstance(entry, dict):
        raise VoiceError(f"{owner}: an artifact entry is not an object.")
    filename = str(entry.get("filename") or "").strip()
    local_name = str(entry.get("local_name") or filename).strip()
    url = str(entry.get("url") or "").strip()
    if not filename or not local_name:
        raise VoiceError(f"{owner}: an artifact has no filename.")
    if not url.startswith("https://"):
        # Plain HTTP is not a download this extension will make. There is no
        # setting for it and no per-artifact exception.
        raise VoiceError(f"{owner}: {filename} is not published over HTTPS.")
    if "/" in local_name or "\\" in local_name or local_name in (".", ".."):
        raise VoiceError(f"{owner}: {filename} has an unsafe local name.")
    sha = entry.get("sha256")
    sha = str(sha).strip().casefold() if sha else None
    size_declared = entry.get("bytes")
    if sha is None or not size_declared:
        # Only ever fills a blank. An overlay that could rewrite a hash this
        # repository committed would be an overlay that defeats the trust root
        # it is extending.
        pinned = (_local_pins() or {}).get(filename)
        if pinned:
            sha = sha or pinned["sha256"]
            entry = dict(entry)
            entry["bytes"] = size_declared or pinned["bytes"]
    if sha is not None and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
        raise VoiceError(f"{owner}: {filename} has a malformed SHA-256.")
    size = entry.get("bytes")
    size = int(size) if isinstance(size, (int, float)) and size else None
    archive = str(entry.get("archive") or "").strip()
    if archive and archive != "tar.bz2":
        raise VoiceError(f"{owner}: {filename} declares an unsupported archive {archive!r}.")
    return Artifact(filename=filename, local_name=local_name, url=url, size=size,
                    sha256=sha, archive=archive,
                    strip_root=str(entry.get("strip_root") or ""))


def default_id(kind: str) -> str:
    """Which bundle a V1 installation uses for ``kind``.

    Asked through here by everything -- the installer, the worker's launch
    paths, the Settings row -- so that a V2 model chooser has exactly one
    function to change.
    """
    return manifest()["defaults"][kind]


def model(identifier: str) -> VoiceModel:
    """One catalogue entry by id, or a refusal.

    The refusal is the security property. An id reaching this function from a
    browser -- and one does, through the install route -- can only ever name
    something the manifest already describes, so there is no URL a caller can
    supply and no file it can ask to have written.
    """
    found = manifest()["models"].get(str(identifier or ""))
    if found is None:
        raise VoiceError(f"{identifier!r} is not a Voice Chat model this extension knows about.")
    return found


def default_model(kind: str) -> VoiceModel:
    return model(default_id(kind))


# --------------------------------------------------------------------------- #
# What is on disk
# --------------------------------------------------------------------------- #


def current_platform() -> tuple[str, str, str]:
    """``(system, machine, python)`` in the manifest's own spelling."""
    system = "windows" if os.name == "nt" else platform.system().casefold()
    machine = platform.machine().casefold() or "unknown"
    if machine in ("x64", "amd64", "x86_64"):
        machine = "amd64" if system == "windows" else "x86_64"
    return system, machine, f"{sys.version_info.major}.{sys.version_info.minor}"


def runtime_platform() -> RuntimePlatform | None:
    """The wheel closure for this machine, or ``None`` if there is not one.

    ``None`` is a supported answer and produces a readable sentence rather than
    an exception: Voice Chat on an unsupported platform is a feature that says
    it is not available here, and a Conversation that works exactly as it did.
    """
    system, machine, python_version = current_platform()
    for candidate in manifest()["platforms"]:
        if candidate.matches(system, machine, python_version):
            return candidate
    return None


@dataclass(frozen=True)
class Status:
    """What Settings, the Voice flyout and the status route all read.

    One object with one ``ready``, because "can I press the microphone" is one
    question and three modules used to be able to answer it differently.
    """

    runtime_ready: bool
    stt_ready: bool
    tts_ready: bool
    runtime_message: str
    stt_message: str
    tts_message: str
    platform_supported: bool
    stt_id: str = ""
    tts_id: str = ""
    tts_voice: str = ""
    busy: str = ""

    @property
    def ready(self) -> bool:
        """I-10: both models, always.

        Dictation needs only STT and the microphone is still not offered until
        Kokoro is installed as well. That is the requested behaviour and it is a
        kindness rather than a restriction: a microphone that transcribes and
        then cannot answer aloud, on a feature whose whole point is speaking, is
        a half-installed feature that looks finished.
        """
        return self.runtime_ready and self.stt_ready and self.tts_ready


def status() -> Status:
    """Read the disk and say what is installed. Never raises, never downloads."""
    try:
        return _status()
    except VoiceError as exc:
        text = str(exc)
        return Status(False, False, False, text, text, text, False)
    except Exception:
        logger.debug("Model Chain: could not read the Voice Chat installation", exc_info=True)
        text = "Voice Chat could not read its own installation."
        return Status(False, False, False, text, text, text, False)


def _status() -> Status:
    spec = manifest()
    chosen = runtime_platform()
    busy = _busy_label()
    if chosen is None:
        system, machine, python_version = current_platform()
        text = (f"Voice Chat has no tested CPU runtime for {system}/{machine} on Python "
                f"{python_version}.")
        return Status(False, False, False, text, text, text, False, busy=busy)

    runtime_ready, runtime_message = _runtime_state(spec, chosen)
    states = {}
    for kind in paths.KINDS:
        entry = spec["models"][spec["defaults"][kind]]
        states[kind] = _model_state(entry)

    stt, tts = spec["defaults"]["stt"], spec["defaults"]["tts"]
    return Status(
        runtime_ready=runtime_ready,
        stt_ready=states["stt"][0],
        tts_ready=states["tts"][0],
        runtime_message=runtime_message,
        stt_message=states["stt"][1],
        tts_message=states["tts"][1],
        platform_supported=True,
        stt_id=stt,
        tts_id=tts,
        tts_voice=spec["models"][tts].voice,
        busy=busy,
    )


def _runtime_state(spec: dict, chosen: RuntimePlatform) -> tuple[bool, str]:
    record = _read_json(paths.runtime_manifest())
    if record is None:
        return False, "Not installed"
    if record.get("runtime_version") != spec["runtime_version"]:
        return False, (f"Installed runtime {record.get('runtime_version')!r} does not match the "
                       f"version this extension is pinned to ({spec['runtime_version']}).")
    if record.get("platform_id") != chosen.identifier:
        return False, ("The installed runtime was built for a different Python or platform.")
    interpreter = runtime_python()
    if interpreter is None or not interpreter.is_file():
        return False, "The installed runtime's interpreter is missing."
    return True, f"Installed — sherpa-onnx {spec['runtime_version']}, CPU"


def _model_state(entry: VoiceModel) -> tuple[bool, str]:
    """Whether ``entry`` is installed, and one sentence saying so.

    What is installed is asked *before* whether it could be downloaded, which is
    the order the two questions actually matter in: a bundle somebody put there
    by hand is installed whatever this build's manifest can or cannot fetch, and
    reporting "not available in this build" over the top of a working
    installation would be reporting on the wrong thing.
    """
    root = paths.bundle_root(entry.kind, entry.identifier)
    record = _read_json(root / paths.INSTALLED_FILENAME)

    if record is not None and record.get("identifier") == entry.identifier:
        missing = [name for name in entry.wanted_paths if not (root / name).exists()]
        if record.get("source") == "local":
            # No committed hash to check against, so the claim made here is the
            # weaker one that is actually true. See :func:`install_from`.
            if missing:
                return False, (f"{entry.label} was installed from your own files, but "
                               f"{missing[0]} is no longer there.")
            return True, f"Installed — {entry.label}, from files you supplied"
        wanted = {item.local_name: item.sha256 for item in entry.artifacts}
        if record.get("artifacts") != wanted:
            return False, (f"Installed {entry.label} does not match this extension's "
                           f"manifest. Download it again.")
        if missing:
            return False, f"Installed {entry.label} is missing {missing[0]}."
        return True, f"Installed — {entry.label}"

    if record is not None:
        return False, "Installed bundle does not match the catalogue."
    if not entry.pinned:
        return False, (f"Not installed. {entry.label} cannot be downloaded automatically by "
                       f"this build — its artifacts are not pinned — but it can be installed "
                       f"from files you download yourself.")
    return False, "Not installed"


def bundle_paths(kind: str) -> dict:
    """Where the worker should look for the installed default bundle.

    Built from the manifest and the disk, never from anything a caller supplied:
    the worker is launched with these paths, and a path that could be influenced
    from a browser would be a path a browser could make the worker read.
    """
    entry = default_model(kind)
    root = paths.bundle_root(kind, entry.identifier)
    found = {"id": entry.identifier, "root": str(root), "engine": entry.engine,
             "voice": entry.voice, "speaker_id": entry.speaker_id,
             "language": entry.language}
    for name in entry.wanted_paths:
        found[Path(name).stem.replace("-", "_")] = str(root / name)
    return found


def runtime_python() -> Path | None:
    """The isolated interpreter, or ``None`` if the runtime is not provisioned."""
    root = paths.runtime_root() / "env"
    candidate = (root / "Scripts" / "python.exe") if os.name == "nt" else (root / "bin" / "python")
    return candidate if candidate.exists() else None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Installing
# --------------------------------------------------------------------------- #


def refusal(kind: str, manual: bool = False) -> str:
    """Why ``kind`` cannot be installed right now, or an empty string.

    ``manual`` asks the question for an install from a folder on this machine,
    which does not need pinned artifacts -- there is nothing to download, so
    there is nothing for a committed hash to be about.

    Asked *before* anything is started, and separately from starting it, because
    the two questions have different audiences. This one answers a browser that
    is waiting for a reply and needs a sentence it can put on screen; the
    transaction below answers a log. Splitting them is what stops a refusal
    happening on a background thread where the only place it can go is a warning
    nobody is reading.
    """
    if kind not in paths.KINDS:
        return "Voice Chat installs a speech-to-text or a text-to-speech model."
    try:
        entry = default_model(kind)
    except VoiceError as exc:
        return str(exc)

    # Ordered by how specific the answer is, not by how cheap the check is. A
    # download that is already running is the most specific thing that can be
    # true, and answering "this build cannot install that" to somebody watching
    # it install would be a true sentence about the wrong thing.
    with _lock:
        if (_progress.get(kind) or {}).get("running"):
            return f"The {kind.upper()} model is already being installed."
    if runtime_platform() is None:
        system, machine, python_version = current_platform()
        return (f"Voice Chat has no tested CPU runtime for {system}/{machine} on Python "
                f"{python_version}, so it cannot be installed here.")
    if not manual and not entry.pinned:
        return (f"{entry.label} cannot be downloaded automatically by this build: its "
                f"artifacts are not pinned in voice/managed-voice-models.json. Either run "
                f"tools/pin_voice_models.py once on a machine that can reach the "
                f"publishers, or download the files yourself and install them from a "
                f"folder below.")
    return ""


def install(kind: str, on_status=None, on_progress=None) -> Status:
    """Provision the runtime if needed, then install ``kind``'s default bundle.

    One button per model and no third button for the engine, because "download
    the voice engine" is not a decision anybody has the information to make:
    the runtime is an implementation detail of both models and is installed by
    whichever one is asked for first.

    The whole thing is a transaction. Nothing outside the staging directory is
    touched until every declared byte has arrived and matched its hash, and the
    last step is a directory rename. A failure anywhere leaves the installation
    exactly as it was, which for somebody who already had a working Voice Chat
    means it still works.

    Every step says what it is doing, twice: into :data:`_progress`, which is
    what the Settings row and the status route read, and into the log, which is
    what somebody reads afterwards when the row has moved on. That is not
    decoration. A download of several hundred megabytes with one static line of
    text in front of it is indistinguishable from a download that has hung, and
    the first version of this shipped exactly that -- ``on_status`` defaulted to
    a function that discarded its argument, so every sentence below was written
    and thrown away.
    """
    if kind not in paths.KINDS:
        raise VoiceError(f"{kind!r} is not a Voice Chat model kind.")
    say = _narrator(kind, on_status)
    tick = _ticker(kind, on_progress)

    with _claim(kind, say):
        entry = default_model(kind)
        if not entry.pinned:
            raise VoiceError(
                f"{entry.label} cannot be installed by this build: its artifacts are not "
                f"pinned in voice/managed-voice-models.json. A maintainer pins them with "
                f"tools/pin_voice_models.py; nothing is downloaded until they do."
            )
        logger.info("Model Chain: Voice Chat is installing the %s bundle %s (%s, about %s)",
                    kind.upper(), entry.identifier, entry.label,
                    _bytes_label(entry.total_bytes))

        say("Checking the voice runtime…")
        install_runtime(on_status=say, on_progress=lambda f: tick(f * 0.5))

        if _model_state(entry)[0]:
            say(f"{entry.label} is already installed.")
            tick(1.0)
            return status()

        _install_model(entry, say, lambda f: tick(0.5 + f * 0.5))
        tick(1.0)
        say(f"{entry.label} installed.")
        logger.info("Model Chain: Voice Chat installed the %s bundle %s at %s",
                    kind.upper(), entry.identifier,
                    paths.bundle_root(entry.kind, entry.identifier))
        return status()


def sources(kind: str) -> list[dict]:
    """Where a person would go to fetch ``kind``'s files by hand.

    The manifest already names every URL, so the Settings row can show them
    rather than asking somebody to find the right Whisper export themselves.
    Returned as data rather than markup because the row is not the only thing
    that will want it.
    """
    entry = default_model(kind)
    return [{"filename": item.filename, "url": item.url, "save_as": item.local_name,
             "archive": bool(item.archive)}
            for item in entry.artifacts]


def install_from(kind: str, folder, on_status=None, on_progress=None) -> Status:
    """Install ``kind`` from files already on this machine.

    The escape hatch, and it earns its place beyond the situation that prompted
    it: a machine with no route to huggingface.co, a corporate proxy that
    refuses large binaries, an air-gapped install, or somebody who already has
    these models for another application.

    What is different from the managed path is stated rather than glossed. There
    is no committed hash to check these against -- this repository makes no
    claim about a file it has never seen -- so the checks are the ones that can
    honestly be made: every required file is present, none of them is empty,
    each ONNX file really is an ONNX file, and the token list really is text.
    A hash is computed and recorded at install time, so later *tampering* is
    still detectable even though the original bytes were never vouched for, and
    :func:`status` says "installed from your own files" rather than claiming a
    verification that did not happen.

    The transaction is the same one: staged, checked, promoted with a rename.
    Nothing that was working stops working because a folder turned out to hold
    the wrong thing.
    """
    if kind not in paths.KINDS:
        raise VoiceError(f"{kind!r} is not a Voice Chat model kind.")
    say = _narrator(kind, on_status)
    tick = _ticker(kind, on_progress)

    source = Path(str(folder or "").strip().strip('"')).expanduser()
    if not str(folder or "").strip():
        raise VoiceError("Give the folder the downloaded files are in.")
    if not source.exists():
        raise VoiceError(f"There is nothing at {source}.")
    if source.is_file():
        # A folder is what is asked for, and a file inside one is what people
        # paste. Taking the parent is kinder than refusing the paste.
        source = source.parent
    if not source.is_dir():
        raise VoiceError(f"{source} is not a folder.")

    with _claim(kind, say):
        entry = default_model(kind)
        logger.info("Model Chain: Voice Chat is installing the %s bundle %s from %s",
                    kind.upper(), entry.identifier, source)
        say(f"Reading {source}…")

        staging = paths.staging_for(entry.identifier, uuid.uuid4().hex[:8])
        target = paths.bundle_root(entry.kind, entry.identifier)
        try:
            staging.mkdir(parents=True, exist_ok=True)
            _gather(entry, source, staging, say)
            tick(0.6)

            say(f"Checking {entry.label} is complete…")
            missing = [name for name in entry.wanted_paths if not (staging / name).exists()]
            if missing:
                raise VoiceError(
                    f"{source} does not have everything {entry.label} needs — {', '.join(missing)} "
                    f"{'is' if len(missing) == 1 else 'are'} missing. Nothing was installed.")
            _sanity_check(entry, staging)
            tick(0.8)

            say("Recording what was installed…")
            digests = {}
            for name in entry.wanted_paths:
                item = staging / name
                if item.is_file():
                    digests[name] = _digest(item)
            _write_json(staging / paths.INSTALLED_FILENAME, {
                "schema": SCHEMA,
                "identifier": entry.identifier,
                "kind": entry.kind,
                "label": entry.label,
                "engine": entry.engine,
                "voice": entry.voice,
                "source": "local",
                "source_folder": str(source),
                "artifacts": digests,
                "installed_at": time.time(),
            })
            say("Installing…")
            _promote(staging, target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        tick(1.0)
        say(f"{entry.label} installed from your own files.")
        logger.info("Model Chain: Voice Chat installed the %s bundle %s from %s at %s",
                    kind.upper(), entry.identifier, source, target)
        return status()


def _gather(entry: VoiceModel, source: Path, staging: Path, say) -> None:
    """Copy what ``entry`` needs out of ``source``, under the names it uses.

    Tolerant about the names on the way in and strict about the names on the
    way out. A publisher calls it ``small-encoder.int8.onnx`` and this feature
    calls it ``encoder.onnx``; somebody who downloaded the file has the first
    name, and telling them to rename it would be this extension making its
    internal spelling somebody else's problem.
    """
    for item in entry.artifacts:
        if item.archive:
            found = _find(source, [item.filename, item.local_name])
            if found is not None:
                say(f"Expanding {found.name}…")
                _expand(found, staging, item)
                continue
            # Already unpacked, which is what somebody who opened the archive
            # to look inside it has. Take the tree instead.
            root = source / item.strip_root if item.strip_root else source
            if not root.is_dir():
                root = source
            _copy_tree(root, staging, entry.wanted_paths, say)
            continue

        found = _find(source, [item.local_name, item.filename])
        if found is None:
            continue
        say(f"Copying {found.name}…")
        shutil.copy2(found, staging / item.local_name)


def _copy_tree(root: Path, staging: Path, wanted, say) -> None:
    for name in wanted:
        origin = root / name
        if not origin.exists():
            continue
        destination = staging / name
        say(f"Copying {name}…")
        if origin.is_dir():
            shutil.copytree(origin, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)


def _find(source: Path, names) -> Path | None:
    """The first of ``names`` in ``source``, matched case-insensitively.

    One level down as well, because an archive extracted with its own top-level
    directory is what a double-click produces and is not somebody's mistake.
    """
    wanted = [name.casefold() for name in names if name]
    for candidate in sorted(source.iterdir()) if source.is_dir() else []:
        if candidate.is_file() and candidate.name.casefold() in wanted:
            return candidate
    for folder in sorted(source.iterdir()) if source.is_dir() else []:
        if not folder.is_dir():
            continue
        for candidate in sorted(folder.iterdir()):
            if candidate.is_file() and candidate.name.casefold() in wanted:
                return candidate
    return None


ONNX_MAGIC = b"\x08"
"""An ONNX file is a protobuf whose first field is ``ir_version``, so it starts
with field 1 as a varint. Two bytes of check, and it is the difference between
"installed" and a worker that dies on its first dictation."""


def _sanity_check(entry: VoiceModel, staging: Path) -> None:
    """The checks that can honestly be made about a file nobody vouched for."""
    for name in entry.wanted_paths:
        item = staging / name
        if item.is_dir():
            if not any(item.iterdir()):
                raise VoiceError(f"{name} is an empty folder. Nothing was installed.")
            continue
        if not item.is_file():
            continue
        size = item.stat().st_size
        if size == 0:
            raise VoiceError(f"{name} is empty. Nothing was installed.")
        if name.endswith(".onnx"):
            if size < 1024 * 1024:
                raise VoiceError(f"{name} is only {_bytes_label(size)}, which is far too "
                                 f"small to be a speech model. Nothing was installed.")
            with open(item, "rb") as handle:
                if handle.read(1) != ONNX_MAGIC:
                    raise VoiceError(f"{name} is not an ONNX model. Nothing was installed.")
        if name.endswith(".txt"):
            try:
                (item).read_text(encoding="utf-8")[:4096]
            except (OSError, UnicodeDecodeError):
                raise VoiceError(f"{name} is not a readable text file. Nothing was "
                                 f"installed.") from None


def _digest(path: Path) -> str:
    found = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            found.update(block)
    return found.hexdigest()


def install_runtime(on_status=None, on_progress=None) -> None:
    """Download the pinned wheel closure and build the isolated environment.

    Returns quietly when the runtime already matches the manifest, which is what
    makes :func:`install` safe to call for the second model.
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)

    spec = manifest()
    chosen = runtime_platform()
    if chosen is None:
        system, machine, python_version = current_platform()
        raise VoiceError(f"Voice Chat has no tested CPU runtime for {system}/{machine} on "
                         f"Python {python_version}, so it cannot be installed here.")
    if _runtime_state(spec, chosen)[0]:
        say("The voice runtime is already installed.")
        tick(1.0)
        return
    logger.info("Model Chain: Voice Chat is provisioning its CPU runtime — sherpa-onnx %s "
                "for %s, %s of wheels", spec["runtime_version"], chosen.identifier,
                _bytes_label(sum(a.approximate_bytes for a in chosen.artifacts)))

    staging = paths.staging_for("runtime", uuid.uuid4().hex[:8])
    wheels = staging / "wheels"
    try:
        _make_room(chosen.artifacts, staging)
        wheels.mkdir(parents=True, exist_ok=True)
        _fetch_all(chosen.artifacts, wheels, say, tick, budget=0.7)

        say("Creating the isolated voice runtime…")
        _build_environment(staging, wheels, chosen)
        tick(0.9)

        say("Checking the runtime can run on the CPU…")
        report = _smoke_test(staging, spec)
        say("Installing the voice runtime…")
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "runtime_version": spec["runtime_version"],
            "platform_id": chosen.identifier,
            "python": chosen.python,
            "provider": report.get("provider", "cpu"),
            "artifacts": {item.local_name: item.sha256 for item in chosen.artifacts},
            "installed_at": time.time(),
        })
        _promote(staging, paths.runtime_root())
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    tick(1.0)
    logger.info("Model Chain: Voice Chat provisioned its CPU runtime — sherpa-onnx %s, %s",
                spec["runtime_version"], chosen.identifier)


def _install_model(entry: VoiceModel, say, tick) -> None:
    staging = paths.staging_for(entry.identifier, uuid.uuid4().hex[:8])
    target = paths.bundle_root(entry.kind, entry.identifier)
    try:
        _make_room(entry.artifacts, staging)
        staging.mkdir(parents=True, exist_ok=True)
        _fetch_all(entry.artifacts, staging, say, tick, budget=0.8)

        for item in entry.artifacts:
            if item.archive:
                say(f"Expanding {item.filename}…")
                _expand(staging / item.local_name, staging, item)
                (staging / item.local_name).unlink(missing_ok=True)
        tick(0.9)

        say(f"Checking {entry.label} is complete…")
        for name in entry.wanted_paths:
            if not (staging / name).exists():
                raise VoiceError(f"{entry.label} downloaded and verified, but {name} is not in "
                                 f"it. Nothing was installed.")
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "identifier": entry.identifier,
            "kind": entry.kind,
            "label": entry.label,
            "engine": entry.engine,
            "voice": entry.voice,
            "revision": entry.revision,
            "license": entry.license,
            "attribution": entry.attribution,
            "artifacts": {item.local_name: item.sha256 for item in entry.artifacts},
            "installed_at": time.time(),
        })
        say("Installing…")
        _promote(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _fetch_all(artifacts, destination: Path, say, tick, budget: float) -> None:
    total = max(sum(item.approximate_bytes for item in artifacts), 1)
    done = 0
    for index, item in enumerate(artifacts, start=1):
        # Named, numbered and sized. "Downloading…" on its own is what a hung
        # download looks like; "Downloading 2 of 3 — decoder.onnx (262 MB)" is
        # something somebody can wait for.
        say(f"Downloading {index} of {len(artifacts)} — {item.filename} "
            f"({_bytes_label(item.approximate_bytes)})")
        base, share = done, max(item.approximate_bytes, 1)

        def report(received, base=base, share=share):
            tick(min((base + min(received, share)) / total, 1.0) * budget)

        started = time.monotonic()
        _download(item, destination / item.local_name, report)
        elapsed = max(time.monotonic() - started, 0.001)
        logger.info("Model Chain: Voice Chat fetched %s — %s in %.1fs (%.1f MB/s), hash "
                    "verified", item.filename, _bytes_label(item.approximate_bytes),
                    elapsed, item.approximate_bytes / elapsed / (1024 * 1024))
        done += share
        tick(min(done / total, 1.0) * budget)


def _download(artifact: Artifact, destination: Path, report) -> None:
    """Fetch one artifact and keep it only if it is exactly what was declared.

    Written into ``.part`` and renamed after the hash matches, so no file with a
    real name is ever a file that has not been verified. There is no resume:
    these are megabytes rather than the gigabytes the LLM catalogue deals in,
    and a restart is cheaper than the class of bug where a stale ``.part`` from a
    different revision is appended to and lands on exactly the right length.
    """
    if not artifact.pinned:
        raise VoiceError(f"{artifact.filename} is not pinned in this extension's manifest.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    received = 0
    request = urllib.request.Request(artifact.url, headers={"User-Agent": USER_AGENT})
    try:
        with contextlib.closing(urllib.request.urlopen(request, timeout=TIMEOUT)) as response:
            with open(partial, "wb") as handle:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
                    received += len(block)
                    if received > artifact.size:
                        raise VoiceError(
                            f"{artifact.filename} is larger than this extension's manifest "
                            f"says it is. Nothing was installed.")
                    report(received)
    except VoiceError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise VoiceError(f"{artifact.filename} could not be downloaded ({exc}). Nothing was "
                         f"installed and Voice Chat is unchanged.") from None

    if received != artifact.size:
        partial.unlink(missing_ok=True)
        raise VoiceError(f"{artifact.filename} is {received} bytes; this extension's manifest "
                         f"says {artifact.size}. Nothing was installed.")
    if digest.hexdigest().casefold() != artifact.sha256:
        partial.unlink(missing_ok=True)
        raise VoiceError(
            f"{artifact.filename} downloaded, but its contents are not what this extension's "
            f"manifest says they should be. The published file has changed since this version "
            f"was released — update the extension rather than retrying. Nothing was installed."
        )
    os.replace(partial, destination)


def _expand(archive: Path, destination: Path, artifact: Artifact) -> None:
    """Unpack a verified archive, refusing every member that escapes.

    The hash says the bytes are the ones the manifest names. It says nothing
    about where the members inside want to be written, and ``tar`` has been able
    to write outside its destination since ``tar`` existed. So each member is
    resolved against the destination and skipped if it lands outside it, and
    links are not extracted at all.
    """
    root = artifact.strip_root.strip("/")
    destination = destination.resolve()
    with tarfile.open(archive, "r:bz2") as bundle:
        for member in bundle.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                continue
            name = member.name.lstrip("./")
            if root and (name == root or name.startswith(root + "/")):
                name = name[len(root):].lstrip("/")
            if not name:
                continue
            where = (destination / name).resolve()
            try:
                where.relative_to(destination)
            except ValueError:
                logger.warning("Model Chain: Voice Chat refused an archive member that would "
                               "have been written outside its bundle")
                continue
            if member.isdir():
                where.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            where.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                continue
            with contextlib.closing(source), open(where, "wb") as handle:
                shutil.copyfileobj(source, handle)


def _build_environment(staging: Path, wheels: Path, chosen: RuntimePlatform) -> None:
    """A venv, and the verified wheels installed into it with no index at all.

    ``--no-index`` removes PyPI. ``--no-deps`` removes the resolver: each wheel
    is named by its path on this disk, in an order the manifest already fixed,
    so pip does no dependency work and has nothing to look anything up for.
    The environment variables restate it for a pip configured through a global
    ``pip.conf`` -- a machine with an index URL in its user configuration should
    not be able to widen what this installs.
    """
    import venv

    environment = staging / "env"
    builder = venv.EnvBuilder(with_pip=True, clear=True, symlinks=(os.name != "nt"))
    try:
        builder.create(environment)
    except Exception as exc:
        raise VoiceError(f"The isolated Voice Chat runtime could not be created ({exc}).") from None

    interpreter = ((environment / "Scripts" / "python.exe") if os.name == "nt"
                   else (environment / "bin" / "python"))
    if not interpreter.exists():
        raise VoiceError("The isolated Voice Chat runtime was created without an interpreter.")

    command = [str(interpreter), "-m", "pip", "install", "--no-index", "--no-deps",
               "--no-cache-dir", "--disable-pip-version-check", "--no-build-isolation"]
    command += [str(wheels / item.local_name) for item in chosen.artifacts]
    environ = dict(os.environ)
    environ.update({"PIP_NO_INDEX": "1", "PIP_NO_CACHE_DIR": "1", "PIP_RETRIES": "0",
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "PIP_INDEX_URL": "", "PIP_EXTRA_INDEX_URL": "",
                    "PIP_CONFIG_FILE": os.devnull})
    result = subprocess.run(command, capture_output=True, text=True, env=environ, timeout=900)
    if result.returncode != 0:
        raise VoiceError("The pinned Voice Chat wheels could not be installed into the isolated "
                         "runtime. Nothing was changed. The installer's own output is in the "
                         "console.")


def _smoke_test(staging: Path, spec: dict) -> dict:
    """Prove the staged runtime imports and can be configured for CPU only.

    Before promotion, so an environment that builds and cannot run never becomes
    the installed one. Run through the worker's own ``--selftest``, which is the
    same file that will do the inference: a smoke test that imports something
    else is a smoke test of something else.
    """
    interpreter = ((staging / "env" / "Scripts" / "python.exe") if os.name == "nt"
                   else (staging / "env" / "bin" / "python"))
    environ = dict(os.environ)
    environ.update(worker_environment())
    try:
        result = subprocess.run(
            [str(interpreter), str(paths.worker_script()), "--selftest"],
            capture_output=True, text=True, env=environ, timeout=300)
    except Exception as exc:
        raise VoiceError(f"The staged Voice Chat runtime could not be started ({exc}). Nothing "
                         f"was installed.") from None
    if result.returncode != 0:
        raise VoiceError("The staged Voice Chat runtime could not import its speech engine, so "
                         "it was not installed. Voice Chat is unchanged.")
    try:
        report = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise VoiceError("The staged Voice Chat runtime did not report what it is. Nothing was "
                         "installed.") from None
    if report.get("provider") != "cpu":
        raise VoiceError(f"The staged Voice Chat runtime reports provider "
                         f"{report.get('provider')!r} rather than CPU, so it was not installed.")
    if report.get("runtime_version") and report["runtime_version"] != spec["runtime_version"]:
        raise VoiceError(f"The staged Voice Chat runtime is sherpa-onnx "
                         f"{report['runtime_version']}, not the pinned "
                         f"{spec['runtime_version']}. Nothing was installed.")
    return report


def worker_environment() -> dict:
    """The environment every voice process runs in. I-1, where it is enforced.

    ``CUDA_VISIBLE_DEVICES=""`` is the blunt instrument and the important one:
    an ONNX Runtime build that would happily have found a GPU finds no devices
    to enumerate. The rest stop a speech process from taking every core of a
    machine that is also rendering an image -- thread caps belong in the config
    the worker builds, and these catch the libraries that read the environment
    before any of our code runs.
    """
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "ONNXRUNTIME_FORCE_CPU": "1",
        "OMP_NUM_THREADS": "4",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        # Not a privacy control on its own -- the worker has no HTTP client --
        # but it makes a library that acquired one fail rather than succeed.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_PROXY": "*",
    }


def _make_room(artifacts, staging: Path) -> None:
    wanted = sum(item.approximate_bytes for item in artifacts)
    if wanted <= 0:
        return
    probe = staging
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    margin = max(SAFETY_MARGIN_BYTES, wanted)
    if free >= wanted + margin:
        return
    raise VoiceError(f"Voice Chat needs about {_bytes_label(wanted)} plus headroom, and "
                     f"{probe} has {_bytes_label(free)} free. Nothing was downloaded.")


def _promote(staging: Path, target: Path) -> None:
    """The verified staging directory becomes the installed one, in one rename.

    A previous install is moved aside rather than deleted and put back if the
    rename fails: ``rmtree`` walks a directory file by file, so deleting a
    runtime whose ONNX libraries are mapped by a worker that has not quite
    exited yet does not leave it alone, it leaves it in pieces.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = target.with_name(target.name + ".previous")
    shutil.rmtree(previous, ignore_errors=True)
    moved = False
    try:
        if target.exists():
            target.rename(previous)
            moved = True
        staging.rename(target)
    except OSError as exc:
        restored = True
        if moved and not target.exists():
            try:
                previous.rename(target)
            except OSError:
                restored = False
                logger.error("Model Chain: Voice Chat could not put %s back after a failed "
                             "promotion; it is at %s", target, previous, exc_info=True)
        # Said out loud rather than suppressed. The filesystem that would not
        # take the new bundle is quite capable of not taking the old one back
        # either, and an installation whose model has silently become
        # ``<name>.previous`` is an installation that reads as "Voice Chat broke
        # itself" with nothing on screen to say where the files went.
        raise VoiceError(
            f"The verified Voice Chat download could not be moved into {target} ({exc}). "
            + ("Nothing about your installation has changed." if restored else
               f"Your previous installation was not damaged — it is at {previous}, and "
               f"renaming that folder back to {target.name} restores it.")) from None
    shutil.rmtree(previous, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bytes_label(value: int) -> str:
    step = 1024.0
    number = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if number < step or unit == "GB":
            return f"{number:.0f} {unit}" if unit in ("B", "KB") else f"{number:.1f} {unit}"
        number /= step
    return f"{number:.1f} GB"


# --------------------------------------------------------------------------- #
# What an install is doing, for a row that has to say something
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _claim(kind: str, say=None):
    """One install per model at a time, and a status line while it runs.

    Two presses of the same button is the ordinary case -- a row that says
    "Downloading…" invites a second click -- and two transactions writing the
    same staging tree is not something atomic promotion protects against,
    because they would each promote half of the other's work.

    The failure branch is the important one. It leaves the reason in
    :data:`_progress` where the Settings row will draw it, and puts it in the
    log at warning level, because "the button went back to how it was" is not
    an answer to "what happened".
    """
    with _lock:
        if kind in _progress and _progress[kind].get("running"):
            raise VoiceError(f"The {kind.upper()} model is already being installed.")
        _progress[kind] = {"running": True, "text": "Starting…", "fraction": 0.0,
                           "failed": False}
    try:
        yield
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        with _lock:
            _progress[kind] = {"running": False, "text": reason, "failed": True,
                               "fraction": 0.0}
        logger.warning("Model Chain: Voice Chat could not install the %s model — %s",
                       kind.upper(), reason)
        raise
    else:
        with _lock:
            _progress[kind] = {"running": False, "text": "Installed.", "fraction": 1.0,
                               "failed": False}


def _narrator(kind: str, on_status=None):
    """A ``say`` that reaches the Settings row, the log, and any caller.

    Three destinations because they answer three different questions and none of
    them substitutes for another: the row is what somebody is watching *now*,
    the log is what they read when it went wrong ten minutes ago, and the
    callback is for a caller that wants to do something else with it.
    """

    def say(text: str) -> None:
        text = str(text or "")
        with _lock:
            current = _progress.setdefault(kind, {"running": True, "fraction": 0.0})
            current["text"] = text
        logger.info("Model Chain: Voice Chat %s — %s", kind.upper(), text)
        if on_status is not None:
            try:
                on_status(text)
            except Exception:
                logger.debug("Model Chain: a Voice Chat status callback failed",
                             exc_info=True)

    return say


def _ticker(kind: str, on_progress=None):
    """A ``tick`` that records the fraction without narrating it.

    Separate from :func:`_narrator` because a percentage moves hundreds of times
    per download and a sentence does not. Logging every tick would bury the
    lines that matter in a progress bar nobody can read.
    """

    def tick(fraction: float) -> None:
        value = min(max(float(fraction or 0.0), 0.0), 1.0)
        with _lock:
            current = _progress.setdefault(kind, {"running": True, "text": ""})
            current["fraction"] = value
        if on_progress is not None:
            try:
                on_progress(value)
            except Exception:
                logger.debug("Model Chain: a Voice Chat progress callback failed",
                             exc_info=True)

    return tick


def _busy_label() -> str:
    running = [f"{kind.upper()} {int(state.get('fraction', 0) * 100)}%"
               for kind, state in sorted(_progress.items()) if state.get("running")]
    return "  ".join(running)


def progress() -> dict:
    """A copy of what each install is doing, for the status route."""
    with _lock:
        return {kind: dict(state) for kind, state in _progress.items()}
