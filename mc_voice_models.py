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
downloads each one itself and checks its hash, and the installation step unpacks
those verified wheels into an interpreter of its own: no index, no resolver, no
package manager at all, and no opportunity for a byte to arrive that was not
verified here first. ``tests/test_voice_models.py`` puts a recording server in
front of a fake index and asserts it is never asked for anything.

Which wheels the closure holds is itself a versioned decision. sherpa's own
version says what the engine is; :func:`closure_id` says what the *environment*
is, and it is derived from the platform and the ordered artifact hashes rather
than remembered by hand -- so adding NumPy to the closure, as build 2 did, makes
every already-installed two-wheel runtime stale without anybody having to think
of bumping a number.

Why some entries ship unpinned
------------------------------
The runtime closure is pinned exactly -- eight platform/Python combinations,
twenty-four wheels, real sizes and real hashes read from PyPI at the revision
this was written against. The two *model* bundles are not: their artifacts live on
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
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
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
    authorized: bool = False
    """Whether this artifact sits behind a publisher's access gate.

    Set by a manifest, never by a browser. It is the only thing that makes this
    module attach a credential to a request, and it attaches one *only* to the
    publisher host named in the URL -- see :func:`_credential` for where the
    token may come from and :class:`_DropCredentialOnHop` for the redirect it
    is removed at (I-PKT-21, section 23.4).
    """

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

    @property
    def closure_id(self) -> str:
        """A fingerprint of exactly which wheels this platform installs.

        Derived rather than declared, which is the whole point. A build number
        is a promise somebody remembered to keep; this is the platform id and
        the ordered ``local_name:sha256`` of every artifact, hashed -- so adding
        a wheel, removing one, reordering the closure or re-pinning any single
        hash all change it without a maintainer having to notice that they
        should.

        Sixteen hex characters, because this is compared against a value this
        installer wrote itself rather than defended against a forger.
        """
        parts = [self.identifier]
        parts += [f"{item.local_name}:{item.sha256 or ''}" for item in self.artifacts]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


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
    tier: str = ""
    summary: str = ""
    notes: str = ""
    about_bytes: int = 0
    ram_bytes: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def pinned(self) -> bool:
        return all(item.pinned for item in self.artifacts)

    @property
    def total_bytes(self) -> int:
        return sum(item.approximate_bytes for item in self.artifacts)

    @property
    def download_bytes(self) -> int:
        """How large this bundle is, for a sentence somebody reads before choosing.

        The pinned total when there is one, and the manifest's own approximation
        otherwise. Approximate is the honest word for it: the model bundles are
        not pinned in this repository, so the exact figure is not known until
        :func:`_resolve` asks the publisher at install time -- which is also the
        number the progress line then counts against.
        """
        return self.total_bytes or int(self.about_bytes or 0)

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
        # The environment's own version, which is not sherpa's. See
        # :func:`closure_id`: this number is for humans and release notes, and
        # the derived fingerprint beside it is what freshness is actually
        # decided on.
        "runtime_build": int(runtime.get("build") or 1),
        "runtime_numpy": str(runtime.get("numpy_version") or ""),
        "runtime_package": str(runtime.get("package") or ""),
        "runtime_import": str(runtime.get("import_name") or "sherpa_onnx"),
        "runtime_license": str(runtime.get("license") or ""),
        "platforms": tuple(platforms),
        "defaults": {kind: defaults[kind] for kind in paths.KINDS},
        "models": models,
        # Optional and passed through rather than parsed into a dataclass. It
        # describes a program this extension does not ship and is only read by
        # ``mc_voice_clone``, which validates every part of it against the disk
        # before anything is run -- so a second schema here would be a second
        # place for the same facts to be wrong. Its absence is the ordinary
        # state of a build with no cloning support at all.
        "cloning": raw.get("cloning") if isinstance(raw.get("cloning"), dict) else {},
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
        # What a chooser needs and an installer does not: which of the offered
        # tiers this is, and the two sentences and two numbers somebody weighs
        # before pressing Download. None of it is verified against anything and
        # none of it can be -- see :attr:`VoiceModel.download_bytes`.
        tier=str(entry.get("tier") or ""),
        summary=str(entry.get("summary") or ""),
        notes=str(entry.get("notes") or ""),
        about_bytes=int(entry.get("about_bytes") or 0),
        ram_bytes=int(entry.get("ram_bytes") or 0),
        # Everything a *particular* bundle carries that is not a property every
        # bundle has. The Kokoro entry's speaker map lives here, pinned beside
        # the checksums of the archive those names came out of -- which is the
        # only place it can be pinned honestly, because which voices exist is a
        # property of that release rather than of this feature.
        extra={key: entry[key] for key in ("speakers",) if key in entry},
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


OPTIONS = {"stt": "model_chain_voice_stt_model_id"}
"""Where a chosen bundle is remembered, per kind.

Only speech-to-text is offered as a choice. Text-to-speech has one bundle and
the thing a user actually varies about it -- which voice, how it is delivered --
is :mod:`mc_voice_registry` and :mod:`mc_voice_profile`, not a second Kokoro.
"""

TIERS = ("low", "medium", "high")
"""Display order, worst-and-fastest first. A tier this build does not know is
still listed, after these, rather than hidden."""

TIER_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}


def _stored_choice(kind: str) -> str:
    option = OPTIONS.get(kind)
    if not option:
        return ""
    try:
        from modules import shared

        return str(getattr(shared.opts, option, "") or "").strip()
    except Exception:
        return ""


def default_id(kind: str) -> str:
    """Which bundle this installation uses for ``kind``.

    Asked through here by everything -- the installer, the worker's launch
    paths, the Settings row, the status the browser reads -- which is what makes
    a model chooser one function rather than a search for every literal.

    A stored choice wins, and only if it still names a bundle of this kind that
    this build's catalogue describes. Everything else -- no choice, a choice
    from a later build, a choice for the wrong kind, a host that will not answer
    -- lands on the manifest's own default. That is deliberately total: this is
    read on the path that launches the worker, and a settings value nobody can
    explain is not a reason for speech to stop working.
    """
    fallback = manifest()["defaults"][kind]
    wanted = _stored_choice(kind)
    if not wanted or wanted == fallback:
        return fallback
    found = manifest()["models"].get(wanted)
    if found is None or found.kind != kind:
        logger.info("Model Chain: the chosen %s model %r is not in this build's catalogue; "
                    "using %s", kind.upper(), wanted, fallback)
        return fallback
    return wanted


def choices(kind: str) -> list:
    """Every bundle of ``kind`` this build offers, in tier order.

    The list a chooser draws. Ordered by :data:`TIERS` rather than by the order
    they happen to appear in the manifest, so "Low, Medium, High" is a property
    of this function and not of whoever last edited the JSON.
    """
    found = [entry for entry in manifest()["models"].values() if entry.kind == kind]

    def rank(entry):
        try:
            return (TIERS.index(entry.tier), entry.label)
        except ValueError:
            return (len(TIERS), entry.label)

    return sorted(found, key=rank)


def select(kind: str, identifier: str) -> str:
    """Remember which bundle of ``kind`` to use. Returns what is chosen after.

    Refuses an id that is not in the catalogue, for the reason :func:`model`
    gives: this is reachable from a browser, and an id that reached the disk
    without passing through the manifest would be a path a caller supplied.

    Choosing does not download anything and does not check that the bundle is
    installed. Those are separate on purpose -- somebody may reasonably pick the
    high tier and then press Download, and a chooser that refused to move until
    the files were there would be a chooser that could never start the download.
    """
    if kind not in OPTIONS:
        raise VoiceError(f"Voice Chat does not offer a choice of {kind} model.")
    entry = model(identifier)
    if entry.kind != kind:
        raise VoiceError(f"{entry.label} is not a {kind} model.")
    _remember(OPTIONS[kind], entry.identifier)
    logger.info("Model Chain: the Voice Chat %s model is now %s (%s)",
                kind.upper(), entry.identifier, entry.label)
    return default_id(kind)


def _remember(name: str, value: str) -> None:
    """Write one option through to the host's store, and save it.

    Best-effort and never fatal, exactly as :mod:`mc_voice_state` is: an
    installation whose options object refuses the write keeps the model it had,
    and Voice Chat keeps working with it.
    """
    try:
        from modules import shared

        shared.opts.set(name, value)
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: could not persist the chosen voice model", exc_info=True)


def catalogue(kind: str) -> list:
    """Every offered bundle of ``kind``, with what is on disk beside it.

    One call rather than a list and a status the caller has to join, because
    joining them in the browser is how a row ends up saying "Installed" against
    the wrong tier.
    """
    chosen = default_id(kind)
    found = []
    for entry in choices(kind):
        ready, message = _model_state(entry)
        found.append({
            "id": entry.identifier,
            "kind": entry.kind,
            "tier": entry.tier,
            "tier_label": TIER_LABELS.get(entry.tier, entry.tier.title() or "Other"),
            "label": entry.label,
            "summary": entry.summary,
            "notes": entry.notes,
            "about_bytes": entry.download_bytes,
            "about_label": _bytes_label(entry.download_bytes),
            "ram_bytes": int(entry.ram_bytes or 0),
            "ram_label": _bytes_label(entry.ram_bytes) if entry.ram_bytes else "",
            "license": entry.license,
            "attribution": entry.attribution,
            "installed": ready,
            "message": message,
            "chosen": entry.identifier == chosen,
            "sources": sources(kind, entry.identifier),
        })
    return found


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


def describe_host() -> str:
    """Everything a failed install would otherwise have to be asked for.

    One line, no paths from anybody's home directory beyond the voice root this
    feature owns, and no content of any kind. It is here so that "share your
    log" is the whole of the next question rather than the first of five.
    """
    system, machine, python_version = current_platform()
    try:
        chosen = runtime_platform()
    except Exception:
        chosen = None
    try:
        spec = manifest()
        wanted = spec["runtime_version"]
        build = f"{spec['runtime_build']}, NumPy {spec['runtime_numpy'] or 'none'}"
    except Exception:
        wanted = "unreadable manifest"
        build = "unknown"
    interpreter = runtime_python()
    return (f"{system}/{machine}, Python {python_version} ({sys.version.split()[0]}), "
            f"runtime platform {chosen.identifier if chosen else 'NONE MATCHED'}, "
            f"closure {chosen.closure_id if chosen else 'none'}, "
            f"pinned sherpa-onnx {wanted}, runtime build {build}, "
            f"isolated interpreter {'present' if interpreter else 'absent'}, "
            f"voice root {paths.data_root()}, "
            f"local pins {'present' if local_pins_path().exists() else 'absent'}")


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
    # Appended rather than slotted in beside ``stt_id``, where they belong by
    # meaning. Several callers build a Status positionally, and a field added
    # in the middle of a positional signature is a value that silently lands in
    # the wrong attribute -- which is a worse bug than an untidy field order.
    stt_label: str = ""
    stt_tier: str = ""

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

    @property
    def summary(self) -> str:
        """One line naming what is still missing, or saying it is ready.

        Because three separate "Not installed" lines and a feature that does not
        work is a puzzle, and this is the sentence that solves it.
        """
        if self.ready:
            return "Voice Chat is ready."
        if not self.platform_supported:
            return self.runtime_message
        missing = [name for name, ok in (("the voice engine", self.runtime_ready),
                                         ("the speech-to-text model", self.stt_ready),
                                         ("the text-to-speech model", self.tts_ready))
                   if not ok]
        return ("Voice Chat is not ready yet — still to install: "
                + ", ".join(missing) + ".")


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
    # ``default_id`` rather than the manifest's own default, so that everything
    # downstream of this -- the microphone's readiness, the Settings row, the
    # handshake check that the worker loaded what this installation verified --
    # is talking about the tier the user actually chose.
    stt, tts = default_id("stt"), default_id("tts")
    states = {}
    for kind, identifier in (("stt", stt), ("tts", tts)):
        states[kind] = _model_state(spec["models"][identifier])

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
        stt_label=spec["models"][stt].label,
        stt_tier=spec["models"][stt].tier,
        tts_voice=spec["models"][tts].voice,
        busy=busy,
    )


def _runtime_state(spec: dict, chosen: RuntimePlatform) -> tuple[bool, str]:
    """Whether the installed runtime is the one this extension now describes.

    Three questions, and the third is the one that was missing. sherpa's version
    says which *engine* is installed and the platform id says which Python it
    was built for, but neither notices that the closure gained a wheel: a
    runtime provisioned before NumPy joined it reports sherpa-onnx 1.13.6 on the
    right platform and is, for the purpose of streaming speech, not the runtime
    this repository ships. :attr:`RuntimePlatform.closure_id` is what tells them
    apart, and an installation from before that field existed has none -- which
    is the same answer as a mismatch and produces the same reprovisioning.
    """
    record = _read_json(paths.runtime_manifest())
    if record is None:
        return False, "Not installed"
    if record.get("runtime_version") != spec["runtime_version"]:
        return False, (f"Installed runtime {record.get('runtime_version')!r} does not match the "
                       f"version this extension is pinned to ({spec['runtime_version']}).")
    if record.get("platform_id") != chosen.identifier:
        return False, ("The installed runtime was built for a different Python or platform.")
    if record.get("runtime_closure_id") != chosen.closure_id:
        return False, ("The installed voice runtime was built from a different set of wheels "
                       "than this extension now pins. Install it again to bring it up to "
                       "date.")
    interpreter = runtime_python()
    if interpreter is None or not interpreter.is_file():
        return False, "The installed runtime's interpreter is missing."
    return True, (f"Installed — sherpa-onnx {spec['runtime_version']}, CPU, "
                  f"runtime build {spec['runtime_build']}")


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
        # Only the artifacts this repository actually pins are compared. A
        # bundle whose digests came from the publisher records what arrived
        # rather than a committed constant, and comparing it to ``None`` would
        # turn every such installation into "download it again" forever.
        recorded = record.get("artifacts") or {}
        for item in entry.artifacts:
            if item.sha256 and recorded.get(item.local_name) != item.sha256:
                return False, (f"Installed {entry.label} does not match this extension's "
                               f"manifest. Download it again.")
        if missing:
            return False, f"Installed {entry.label} is missing {missing[0]}."
        return True, f"Installed — {entry.label}"

    if record is not None:
        return False, "Installed bundle does not match the catalogue."
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


class Gated(VoiceError):
    """A publisher that answered "you may not have this yet".

    Its own class because the answer is a product state rather than a fault: the
    artifact exists, this build knows where it is, and what is missing is an
    acceptance somebody has to make upstream and a token this process has to be
    given. An installer that reported it as "the download failed" would be
    telling a user to retry the one thing that cannot work.
    """

    def __init__(self, filename: str, status: int):
        self.filename = str(filename or "")
        self.status = int(status or 0)
        super().__init__(
            f"{self.filename} is behind the publisher's access gate (HTTP {self.status}). "
            f"Accept the model's conditions on the publisher's page with your own account, "
            f"then give Voice Chat an access token — Settings → Voice Chat → Access "
            f"token — or start this WebUI with HF_TOKEN set in its environment.")


@dataclass(frozen=True)
class Expected:
    """What one artifact should turn out to be, and who said so.

    ``sha256`` from this repository is the strongest answer and the one the
    runtime wheels always have. ``sha256`` from the publisher is the answer for
    a model bundle nobody has pinned yet: a HEAD to the hub returns the LFS
    object's own digest, which is the publisher's attestation fetched over TLS
    rather than a constant somebody reviewed. Weaker, and stated as weaker --
    but the alternative that shipped was a Download button that refused to
    download, which is not a security posture, it is a broken feature.
    """

    size: int | None
    sha256: str | None
    source: str

    @property
    def verified(self) -> bool:
        return bool(self.sha256)


class _StopAtRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect handler that follows nothing, so :func:`_head` can decide."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


CREDENTIAL_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
"""Environment variables a gated publisher's token may be read from.

Nothing writes these. If one is set, somebody set it for this process on
purpose, which is the clearest consent this module can observe -- and it lasts
exactly as long as the process, which is the other half of why it is the
default.
"""

CREDENTIAL_HOSTS = ("huggingface.co", "www.huggingface.co")
"""The one publisher a stored token is ever offered to.

A token is for the host that issued it. This list is why a manifest that grew a
second download host could not quietly start sending somebody's Hugging Face
credential to it, and why :func:`_credential` takes the URL rather than being
asked for "the token".
"""


def stored_credential() -> str:
    """The token somebody asked Voice Chat to remember, or ``""``.

    Read from disk on every call rather than cached. A user who presses Forget
    expects the next download to stop using it, and a module that had it in
    memory would keep using it until something restarted.
    """
    try:
        found = json.loads(paths.credential_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str((found or {}).get("token") or "").strip() if isinstance(found, dict) else ""


def remember_credential(token: str) -> dict:
    """Keep a publisher token for later installs. Returns what may be shown.

    Written with the file created ``0600`` from the start rather than tightened
    afterwards, because a chmod after the write is a window in which the token
    is world readable. On Windows the mode is advisory and the protection is
    that the file is under the user's own data root; that is stated in the panel
    rather than implied.

    What comes back never contains the token. The panel gets "there is one, and
    it ends 1a2b", which is enough for somebody to recognise the key they
    pasted and useless to anybody reading over their shoulder.
    """
    wanted = str(token or "").strip()
    if not wanted:
        raise VoiceError("Paste an access token first.")
    if any(character.isspace() for character in wanted):
        raise VoiceError("That does not look like an access token — it has spaces in it.")
    if len(wanted) < 8 or len(wanted) > 512:
        raise VoiceError("That does not look like an access token.")
    path = paths.credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".new")
    # Opened by descriptor with the mode in the open, so the bytes are never on
    # disk under a broader mode than this, not even for an instant.
    handle = os.open(str(staging), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as writer:
            json.dump({"schema": 1, "host": CREDENTIAL_HOSTS[0], "token": wanted}, writer)
    except BaseException:
        try:
            os.remove(staging)
        except OSError:
            pass
        raise
    os.replace(staging, path)
    logger.info("Model Chain: Voice Chat stored a publisher access token for %s",
                CREDENTIAL_HOSTS[0])
    return credential_state()


def forget_credential() -> dict:
    """Remove the stored token. Idempotent, and never raises for a missing file."""
    path = paths.credential_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return credential_state()
    except OSError as exc:
        raise VoiceError(f"That token could not be removed ({exc.__class__.__name__}). "
                         f"Delete {path.name} in the Voice Chat folder by hand.") from None
    logger.info("Model Chain: Voice Chat forgot its stored publisher access token")
    return credential_state()


def credential_state() -> dict:
    """What the panel may know about the token. Never the token.

    ``ends`` is the last four characters and exists so somebody can tell which
    of their keys is in here without being shown it. Everything else is a
    boolean or a name.
    """
    stored = stored_credential()
    environment = [name for name in CREDENTIAL_VARIABLES
                   if str(os.environ.get(name) or "").strip()]
    return {
        "stored": bool(stored),
        "ends": stored[-4:] if len(stored) >= 8 else "",
        "from_environment": bool(environment),
        "environment_variable": environment[0] if environment else "",
        "host": CREDENTIAL_HOSTS[0],
    }


def _credential(url: str) -> str:
    """The bearer token for ``url``'s publisher, or an empty string.

    Read fresh each time rather than cached, so that a token removed from the
    environment stops being used without a restart, and so that this module
    never holds one longer than the request that needs it.
    """
    parts = urllib.parse.urlsplit(str(url or ""))
    if parts.scheme.casefold() != "https":
        # A bearer token belongs on an encrypted transport and nowhere else.
        # The manifests this reads are this repository's own, so this is not a
        # case that arises -- which is exactly why it is cheap to refuse.
        return ""
    if parts.netloc.casefold() not in CREDENTIAL_HOSTS:
        return ""
    # The stored one first. It is the most recent thing a person did on purpose
    # -- they pasted it into this feature's own settings -- and a saved token
    # that a stale environment variable silently overrode would be a Save that
    # appeared to do nothing.
    found = stored_credential()
    if found:
        return found
    for name in CREDENTIAL_VARIABLES:
        found = str(os.environ.get(name) or "").strip()
        if found:
            return found
    return ""


class _DropCredentialOnHop(urllib.request.HTTPRedirectHandler):
    """Follow a redirect, but never carry a credential across a host boundary.

    This is the whole of T-INSTALL-P5, and it has to be written down because
    the default behaviour is the opposite: :mod:`urllib` copies every header
    except ``Content-Length`` and ``Content-Type`` onto the redirected request,
    so an ``Authorization`` added for huggingface.co would be sent verbatim to
    whichever storage host the 302 names.

    That host does not need it -- a gated hub answers with a *signed* delivery
    URL, which is the point of the redirect -- and sending it anyway would hand
    somebody's Hugging Face token to a CDN operator as a matter of routine.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        made = super().redirect_request(req, fp, code, msg, headers, newurl)
        if made is None or _origin(newurl) == _origin(req.full_url):
            return made
        for name in list(made.headers):
            if str(name).casefold() == "authorization":
                made.headers.pop(name, None)
        made.unredirected_hdrs.pop("Authorization", None)
        return made


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.casefold()


def _origin(url: str) -> str:
    """Scheme and host together, which is what "the same place" means here.

    The host alone is not enough. A 302 from ``https://host/a`` to
    ``http://host/b`` stays on the host and is still a downgrade onto the
    clear, so a credential that followed it would be a bearer token sent in
    plaintext to anything on the path -- which is the failure the redirect
    guard exists to prevent, arriving by a different door.
    """
    parts = urllib.parse.urlsplit(str(url or ""))
    return f"{parts.scheme.casefold()}://{parts.netloc.casefold()}"


def _header_map(answer) -> dict:
    return {str(name).casefold(): value for name, value in answer.headers.items()}


def _head(url: str, hops: int = 5, authorized: bool = False) -> tuple[int, dict]:
    """A HEAD whose answer is the *publisher's*, not a delivery host's.

    Redirects are followed only while they stay on the host that was asked, and
    the last same-host answer is what comes back. That is not fastidiousness --
    it is the difference between a digest and a cache key. Hugging Face answers
    a ``/resolve/`` HEAD with ``X-Linked-Etag``, the LFS object's SHA-256, and a
    302 to a storage host whose own ``ETag`` is an opaque validator that happens
    to be sixty-four hexadecimal characters wide. Following that redirect threw
    the first away and left the second in its place, so every large artifact was
    checked against a value that was never its digest and every correct download
    of one was rejected. ``huggingface_hub`` stops at the same point, for the
    same reason.

    A redirect off the host is not a failure and is not reported as one: the
    publisher has said everything it is going to say, and what it said is what
    the caller wanted.
    """
    opener = urllib.request.build_opener(_StopAtRedirect)
    here = url
    status, found = 0, {}
    for _ in range(max(int(hops), 1)):
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        # Only ever on the host this loop is still on: ``here`` advances only
        # while the redirect stays same-host, so the credential cannot follow a
        # hop it should not.
        token = _credential(here) if authorized else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(here, method="HEAD", headers=headers)
        try:
            with contextlib.closing(opener.open(request, timeout=TIMEOUT)) as answer:
                return int(getattr(answer, "status", 200) or 200), _header_map(answer)
        except urllib.error.HTTPError as answer:
            with contextlib.closing(answer):
                status = int(getattr(answer, "code", 0) or 0)
                found = _header_map(answer)
        if status not in (301, 302, 303, 307, 308):
            return status, found
        target = str(found.get("location") or "").strip()
        if not target:
            return status, found
        target = urllib.parse.urljoin(here, target)
        if _origin(target) != _origin(here):
            return 200, found
        here = target
    return status or 200, found


SHA256_HEADERS = ("x-linked-etag", "x-checksum-sha256", "x-amz-meta-sha256")
"""Headers whose *name* states that the value is a SHA-256 of the content.

``etag`` is deliberately not among them. RFC 9110 defines an entity tag as an
opaque validator: a host may make it an MD5, a storage-layer hash, an upload id
or a random string, and it is only ever obliged to change when the body does. A
value that looks like a SHA-256 is therefore not evidence of being one, and
reading Hugging Face's delivery host that way is exactly the bug this list
exists to prevent. ``x-linked-etag`` is different in kind: Hugging Face
documents it as the LFS object's SHA-256 and its own client reads it as that.
"""


def _as_sha256(raw, prefixed: bool) -> str | None:
    """``raw`` as a lowercase SHA-256, or ``None``.

    ``prefixed`` demands that the value say what it is -- ``sha256:<hex>`` --
    which is how a header that is otherwise opaque can still be trusted.
    """
    value = str(raw or "").strip()
    if value[:2].upper() == "W/":
        value = value[2:].strip()
    value = value.strip('"')
    if value[:7].casefold() == "sha256:":
        value = value[7:]
    elif prefixed:
        return None
    if len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value):
        return value.casefold()
    return None


def _published_digest(headers: dict) -> str | None:
    """A SHA-256 the publisher actually stated, or ``None`` if it stated none.

    ``None`` is a perfectly good answer and not a failure: the download then
    has its byte count checked, and the digest of what arrived is recorded so
    the *next* install of that bundle is checked against a constant.
    """
    for name in SHA256_HEADERS:
        found = _as_sha256(headers.get(name), prefixed=False)
        if found:
            return found
    return _as_sha256(headers.get("etag"), prefixed=True)


def _declared_size(url: str) -> int | None:
    """The length a delivery host declares, following every redirect.

    Only reached when the publisher's own answer carried no length -- GitHub's
    release URLs are a redirect and nothing else. A byte count from a delivery
    host is worth having where a digest from one is not: it is a sanity bound
    and a progress bar, never an attestation.
    """
    request = urllib.request.Request(
        url, method="HEAD",
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    try:
        with contextlib.closing(urllib.request.urlopen(request, timeout=TIMEOUT)) as answer:
            raw = str(answer.headers.get("Content-Length") or "")
    except Exception:
        logger.debug("Model Chain: Voice Chat could not read a length for %s", url,
                     exc_info=True)
        return None
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def _resolve(artifact: Artifact) -> Expected:
    """Ask the publisher what this file is, before fetching a byte of it.

    A HEAD that stops at the publisher (:func:`_head`), so what is read is what
    the publisher said rather than what a cache in front of it happens to
    answer. Hugging Face states an LFS object's SHA-256 in ``x-linked-size`` and
    ``x-linked-etag`` -- the same fact ``tools/pin_managed_models.py`` reads to
    pin the LLM catalogue, read here at install time instead of at review time.
    A release asset on another host answers with a length and an opaque etag,
    and saying so is better than pretending: what comes back then is a size, and
    the checks below it are structural.
    """
    if artifact.sha256 and artifact.size:
        return Expected(artifact.size, artifact.sha256, "this extension's manifest")
    try:
        status, headers = _head(artifact.url, authorized=artifact.authorized)
    except Exception as exc:
        logger.warning("Model Chain: Voice Chat could not ask the publisher about %s — %s: %s",
                       artifact.filename, exc.__class__.__name__, exc)
        return Expected(artifact.size, artifact.sha256, "nothing (the publisher did not answer)")
    if status in (401, 403) and artifact.authorized:
        # A different failure from "the publisher is down", and it has a
        # different remedy: somebody has to accept the model's conditions
        # upstream, or make a token available to this process. Raised rather
        # than degraded to a byte count, so the install stops here with a
        # sentence the panel can show instead of failing later on a hash
        # nobody could have matched (T-INSTALL-P4).
        raise Gated(artifact.filename, status)
    if status >= 400:
        logger.warning("Model Chain: Voice Chat asked the publisher about %s and got HTTP %s",
                       artifact.filename, status)
        return Expected(artifact.size, artifact.sha256, "nothing (the publisher answered %s)"
                        % status)

    size = artifact.size
    for name in ("x-linked-size", "content-length"):
        raw = str(headers.get(name) or "")
        if raw.isdigit() and int(raw) > 0:
            size = int(raw)
            break

    digest = artifact.sha256 or _published_digest(headers)
    if not size:
        size = _declared_size(artifact.url) or artifact.size

    return Expected(size, digest,
                    "this extension's manifest" if artifact.sha256
                    else "the publisher, over HTTPS" if digest
                    else "the publisher's byte count only")


def refusal(kind: str, manual: bool = False, identifier: str = "") -> str:
    """Why ``kind`` cannot be installed right now, or an empty string.

    ``manual`` is kept for the caller's benefit and no longer changes the
    answer: an unpinned artifact stopped being a refusal when resolving it
    against the publisher became part of the download.

    ``identifier`` names one bundle rather than whichever is currently chosen,
    which is what lets a Settings row offer three tiers with a Download button
    each. An id that is not in this build's catalogue is refused here, before a
    thread is started, so the browser waiting for the answer gets a sentence.

    Asked *before* anything is started, and separately from starting it, because
    the two questions have different audiences. This one answers a browser that
    is waiting for a reply and needs a sentence it can put on screen; the
    transaction below answers a log. Splitting them is what stops a refusal
    happening on a background thread where the only place it can go is a warning
    nobody is reading.
    """
    if kind not in paths.KINDS and kind != "runtime":
        return ("Voice Chat installs the voice engine, a speech-to-text model or a "
                "text-to-speech model.")
    try:
        # Not for its value -- for the exception. A catalogue this build cannot
        # read, or an id it does not know, are the two things here that are
        # still a refusal.
        if kind != "runtime":
            _wanted(kind, identifier)
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
    # An artifact this repository has not pinned is no longer a refusal. It is
    # resolved against the publisher at install time instead -- see
    # :func:`_resolve`. A Download button that will not download is not a
    # security posture.
    return ""


def _wanted(kind: str, identifier: str = "") -> VoiceModel:
    """The bundle an install or a listing is about.

    Named rather than repeated at four call sites, because "the one that was
    asked for, or the one in use" is the rule and a call site that got it wrong
    would install the right files under the wrong tier's name.
    """
    if not str(identifier or "").strip():
        return default_model(kind)
    entry = model(identifier)
    if entry.kind != kind:
        raise VoiceError(f"{entry.label} is not a {kind} model.")
    return entry


def install(kind: str, on_status=None, on_progress=None, identifier: str = "") -> Status:
    """Provision the runtime if needed, then install one of ``kind``'s bundles.

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
    entry = _wanted(kind, identifier)

    with _claim(kind, say, entry.identifier):
        logger.info("Model Chain: Voice Chat is installing the %s bundle %s (%s, about %s)",
                    kind.upper(), entry.identifier, entry.label,
                    _bytes_label(entry.download_bytes))

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


def sources(kind: str, identifier: str = "") -> list[dict]:
    """Where a person would go to fetch a bundle's files by hand.

    The manifest already names every URL, so the Settings row can show them
    rather than asking somebody to find the right Whisper export themselves.
    Returned as data rather than markup because the row is not the only thing
    that will want it.

    Per bundle rather than per kind: three Whisper tiers are three different
    sets of addresses, and a manual-install panel that printed the chosen
    tier's links under another tier's heading would send somebody to download
    the wrong two hundred megabytes.
    """
    entry = _wanted(kind, identifier)
    return [{"filename": item.filename, "url": item.url, "save_as": item.local_name,
             "archive": bool(item.archive)}
            for item in entry.artifacts]


def install_from(kind: str, folder, on_status=None, on_progress=None,
                 identifier: str = "") -> Status:
    """Install one of ``kind``'s bundles from files already on this machine.

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

    entry = _wanted(kind, identifier)

    with _claim(kind, say, entry.identifier):
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


def install_engine(on_status=None, on_progress=None, folder=None) -> Status:
    """Install the speech engine on its own, as a button somebody can press.

    Ordinarily the engine is an implementation detail of the two models and is
    provisioned by whichever one is downloaded first, which is why there is no
    third button in the ordinary case. There is one now because that reasoning
    has a hole in it, and somebody fell in: install both models from files you
    already had -- which downloads nothing -- and the engine is still missing,
    both model buttons say "Installed", and there is no longer anything to press.
    A dead end reached by following the instructions is a dead end this feature
    put there.
    """
    say = _narrator("runtime", on_status)
    tick = _ticker("runtime", on_progress)
    with _claim("runtime", say):
        install_runtime(on_status=say, on_progress=tick, folder=folder)
        say("The voice engine is installed.")
        return status()


def runtime_sources() -> list[dict]:
    """The wheels this platform needs, for somebody fetching them by hand."""
    chosen = runtime_platform()
    if chosen is None:
        return []
    return [{"filename": item.filename, "url": item.url, "save_as": item.local_name,
             "archive": False, "bytes": item.approximate_bytes}
            for item in chosen.artifacts]


def _adopt_wheels(chosen: RuntimePlatform, folder: Path, wheels: Path, say) -> None:
    """Take the pinned wheels from a folder instead of downloading them.

    Better verified than the model bundles, not worse: these artifacts *are*
    pinned in this repository, so a wheel somebody downloaded by hand is checked
    against the same committed SHA-256 the download would have been. A file
    under the right name with the wrong contents is refused exactly as a bad
    download is.
    """
    for item in chosen.artifacts:
        found = _find(folder, [item.local_name, item.filename])
        if found is None:
            raise VoiceError(f"{folder} does not have {item.filename}. Nothing was installed.")
        say(f"Checking {found.name}…")
        digest = _digest(found)
        if item.sha256 and digest != item.sha256:
            raise VoiceError(
                f"{found.name} is not the file this extension's manifest describes — its "
                f"contents do not match the committed SHA-256. Nothing was installed.")
        shutil.copy2(found, wheels / item.local_name)
        logger.info("Model Chain: Voice Chat adopted %s from %s; digest matched this "
                    "extension's manifest", item.filename, folder)


def install_runtime(on_status=None, on_progress=None, folder=None) -> None:
    """Provision the isolated CPU runtime from the pinned wheel closure.

    ``folder`` takes the wheels from a directory on this machine instead of
    downloading them -- the same escape hatch the models have, and here the
    stronger one: these artifacts are pinned in this repository, so a
    hand-supplied wheel is checked against a committed hash rather than against
    the publisher.

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
                "for %s, %s of wheels%s", spec["runtime_version"], chosen.identifier,
                _bytes_label(sum(a.approximate_bytes for a in chosen.artifacts)),
                f", from {folder}" if folder else "")

    staging = paths.staging_for("runtime", uuid.uuid4().hex[:8])
    wheels = staging / "wheels"
    try:
        wheels.mkdir(parents=True, exist_ok=True)
        if folder:
            source = Path(str(folder).strip().strip('"')).expanduser()
            if source.is_file():
                source = source.parent
            if not source.is_dir():
                raise VoiceError(f"There is no folder at {source}.")
            _adopt_wheels(chosen, source, wheels, say)
            tick(0.7)
        else:
            _make_room(chosen.artifacts, staging)
            # The wheels are always pinned in this repository, so this resolves
            # to the committed values without a request being made.
            _fetch_all(chosen.artifacts, wheels, say, tick, budget=0.7,
                       expectations=_expectations(chosen.artifacts, say))

        say("Creating the isolated voice runtime…")
        _build_environment(staging, wheels, chosen)
        tick(0.9)

        say("Checking the runtime can run on the CPU…")
        report = _smoke_test(staging, spec)
        say("Installing the voice runtime…")
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "runtime_version": spec["runtime_version"],
            # The engine version above, the environment's version here, and the
            # fingerprint of what is actually unpacked below it. Only the last
            # of the three decides freshness -- see :func:`_runtime_state` --
            # but the other two are what a person reads in a bug report.
            "runtime_build": spec["runtime_build"],
            "runtime_closure_id": chosen.closure_id,
            "platform_id": chosen.identifier,
            "python": chosen.python,
            "provider": report.get("provider", "cpu"),
            "numpy_version": report.get("numpy_version", ""),
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
        expectations = _expectations(entry.artifacts, say)
        _make_room(entry.artifacts, staging, expectations)
        staging.mkdir(parents=True, exist_ok=True)
        digests = _fetch_all(entry.artifacts, staging, say, tick, budget=0.8,
                             expectations=expectations)

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
            "artifacts": {item.local_name: digests.get(item.filename) or item.sha256
                          for item in entry.artifacts},
            "verified_by": {item.filename: expectations[item.local_name].source
                            for item in entry.artifacts},
            "installed_at": time.time(),
        })
        say("Installing…")
        _promote(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # What arrived, written down where the next install will read it. A bundle
    # this repository could not pin is checked against the publisher the first
    # time and against a constant every time after -- so a file that changes
    # under us later is a refusal rather than a silent substitution.
    _record_pins(entry, digests, expectations)


def _record_pins(entry: VoiceModel, digests: dict, expectations: dict) -> None:
    """Write what was downloaded into the local pin overlay. Never fatal.

    Only for artifacts this repository has not pinned -- a committed hash is the
    trust root and nothing written at runtime may touch it. An extension folder
    that is read-only, or on a share, simply does not get the benefit: the next
    install resolves against the publisher again, exactly as this one did.
    """
    if not digests:
        return
    wanted = {item.filename: digests.get(item.filename)
              for item in entry.artifacts if not item.sha256 and digests.get(item.filename)}
    if not wanted:
        return
    path = local_pins_path()
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        current = {}
    recorded = dict(current.get("artifacts") or {})
    for item in entry.artifacts:
        digest = wanted.get(item.filename)
        if not digest:
            continue
        size = (expectations.get(item.local_name) or Expected(None, None, "")).size
        recorded[item.filename] = {"sha256": digest, "bytes": int(size or 0)}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "artifacts": recorded}, indent=2) + "\n",
                        encoding="utf-8")
    except OSError:
        logger.debug("Model Chain: Voice Chat could not record what it downloaded to %s",
                     path, exc_info=True)
        return
    _manifest_cache_clear()
    logger.info("Model Chain: Voice Chat recorded %d artifact digest(s) in %s — the next "
                "install of this bundle is checked against them", len(wanted), path.name)


def _manifest_cache_clear() -> None:
    global _manifest_cache

    with _lock:
        _manifest_cache = None


def _expectations(artifacts, say) -> dict:
    """Ask the publisher about every artifact this repository has not pinned.

    Done once, up front, so the byte counts are known before the progress bar
    starts, and so a host that cannot be reached fails here rather than halfway
    through a download.
    """
    found = {}
    for item in artifacts:
        if not (item.sha256 and item.size):
            say(f"Asking the publisher about {item.filename}…")
        expected = _resolve(item)
        found[item.local_name] = expected
        logger.info("Model Chain: Voice Chat expects %s to be %s with digest %s — according "
                    "to %s", item.filename, _bytes_label(expected.size or 0),
                    expected.sha256 or "(none offered)", expected.source)
    return found


def _fetch_all(artifacts, destination: Path, say, tick, budget: float,
               expectations: dict) -> dict:
    """Download each artifact. Returns the digest of what actually arrived."""
    total = max(sum((expectations[item.local_name].size or item.approximate_bytes)
                    for item in artifacts), 1)
    done = 0
    digests = {}
    for index, item in enumerate(artifacts, start=1):
        expected = expectations[item.local_name]
        size = expected.size or item.approximate_bytes
        # Named, numbered and sized. "Downloading…" on its own is what a hung
        # download looks like; "Downloading 2 of 3 — decoder.onnx (262 MB)" is
        # something somebody can wait for.
        say(f"Downloading {index} of {len(artifacts)} — {item.filename} "
            f"({_bytes_label(size)})")
        base, share = done, max(size, 1)

        def report(received, base=base, share=share):
            tick(min((base + min(received, share)) / total, 1.0) * budget)

        started = time.monotonic()
        digests[item.filename] = _download(item, destination / item.local_name, report,
                                           expected)
        elapsed = max(time.monotonic() - started, 0.001)
        logger.info("Model Chain: Voice Chat fetched %s — %s in %.1fs (%.1f MB/s); %s",
                    item.filename, _bytes_label(size), elapsed,
                    size / elapsed / (1024 * 1024),
                    f"digest matched {expected.source}" if expected.verified
                    else "digest recorded (the publisher offered none)")
        done += share
        tick(min(done / total, 1.0) * budget)
    return digests


def _download(artifact: Artifact, destination: Path, report, expected: "Expected") -> str:
    """Fetch one artifact, keep it only if it is what was expected, and report
    the digest of what actually arrived.

    Written into ``.part`` and renamed after the check passes, so no file with a
    real name is ever a file that was not checked. There is no resume: these are
    megabytes rather than the gigabytes the LLM catalogue deals in, and a
    restart is cheaper than the class of bug where a stale ``.part`` from a
    different revision is appended to and lands on exactly the right length.

    What "expected" means depends on who was able to say, and the difference is
    reported rather than blurred. A digest from this repository or from the
    publisher is checked, and a mismatch throws the download away. A publisher
    that offered only a byte count gets the byte count checked -- and the digest
    of what arrived is returned, so the caller can record it and the *next*
    install of that bundle is checked against a constant.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    received = 0
    # A declared size is a budget as well as a checksum input: a URL that
    # started answering with a hundred gigabytes should not fill somebody's disk
    # before anything disagrees. Twice the expected size, or a flat ceiling when
    # nothing would say.
    ceiling = (expected.size or 0) * 2 or (4 * 1024 * 1024 * 1024)
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    token = _credential(artifact.url) if artifact.authorized else ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(artifact.url, headers=headers)
    # The default opener would carry that ``Authorization`` onto the signed
    # delivery URL a gated hub redirects to. The credentialed path does not,
    # which is the difference between fetching a gated file and handing a
    # credential to a CDN (I-PKT-21, T-INSTALL-P5). An artifact with no token
    # takes ``urlopen`` exactly as it always has, so the unchanged path is
    # unchanged down to the function it calls.
    def _open():
        if token:
            return urllib.request.build_opener(_DropCredentialOnHop).open(
                request, timeout=TIMEOUT)
        return urllib.request.urlopen(request, timeout=TIMEOUT)

    try:
        with contextlib.closing(_open()) as response:
            with open(partial, "wb") as handle:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
                    received += len(block)
                    if received > ceiling:
                        raise VoiceError(
                            f"{artifact.filename} is far larger than expected. Nothing was "
                            f"installed.")
                    report(received)
    except VoiceError:
        partial.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as answer:
        status = int(getattr(answer, "code", 0) or 0)
        with contextlib.closing(answer):
            pass
        partial.unlink(missing_ok=True)
        if status in (401, 403) and artifact.authorized:
            raise Gated(artifact.filename, status) from None
        raise VoiceError(f"{artifact.filename} could not be downloaded (HTTP {status}). "
                         f"Nothing was installed and Voice Chat is unchanged.") from None
    except Exception as exc:
        partial.unlink(missing_ok=True)
        # The class and the message, never the request: a urllib error for a
        # gated URL can carry the URL, and a signed delivery URL is a
        # credential of its own for as long as it is valid (section 36).
        raise VoiceError(f"{artifact.filename} could not be downloaded "
                         f"({exc.__class__.__name__}: {exc}). Nothing was installed and "
                         f"Voice Chat is unchanged.") from None

    found = digest.hexdigest().casefold()
    if received == 0:
        partial.unlink(missing_ok=True)
        raise VoiceError(f"{artifact.filename} downloaded as an empty file. Nothing was "
                         f"installed.")
    if expected.size and received != expected.size:
        partial.unlink(missing_ok=True)
        raise VoiceError(f"{artifact.filename} arrived as {received} bytes; {expected.source} "
                         f"says {expected.size}. Nothing was installed.")
    if expected.sha256 and found != expected.sha256:
        partial.unlink(missing_ok=True)
        raise VoiceError(
            f"{artifact.filename} downloaded, but its contents are not what "
            f"{expected.source} says they should be. Nothing was installed and Voice Chat "
            f"is unchanged."
        )
    os.replace(partial, destination)
    return found


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


def site_packages(environment: Path) -> Path:
    """Where an interpreter in ``environment`` imports third-party code from."""
    if os.name == "nt":
        return environment / "Lib" / "site-packages"
    found = sorted((environment / "lib").glob("python*/site-packages"))
    return found[0] if found else (environment / "lib" / "site-packages")


WHEEL_SKIP = ("scripts", "headers", "data")
"""Parts of a wheel's ``.data`` directory this installer has no use for.

``purelib`` and ``platlib`` are the code and are moved into site-packages, which
is what an installer does with them. The rest are console entry points and C
headers for building against the library -- neither of which a speech worker
that is only ever launched by path will use.
"""


def _unpack_wheel(wheel: Path, destination: Path) -> list[str]:
    """Install one wheel by unpacking it. Returns the top-level names it added.

    A wheel is a zip with a defined layout, and installing one that needs no
    build step is unpacking it into site-packages. Doing that directly rather
    than through pip is the correction to a real failure: pip reported success,
    exited zero, and the module was not importable afterwards -- because a
    machine can carry ``PIP_TARGET``, ``PIP_USER``, ``PIP_PREFIX`` or a
    ``pip.ini`` that sends an install somewhere else entirely, and pip is
    perfectly happy about that. There is no package manager in this path now, so
    there is nothing left to redirect it.

    It also makes R2-2 trivially true rather than carefully arranged: an
    installer that resolves nothing cannot resolve something from an index.
    """
    added: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(wheel) as bundle:
            names = bundle.namelist()
            data_root = next((n.split("/")[0] for n in names
                              if n.split("/")[0].endswith(".data")), "")
            for name in names:
                if name.endswith("/"):
                    continue
                parts = name.split("/")
                if data_root and parts[0] == data_root:
                    # <name>.data/<category>/<path…>
                    if len(parts) < 3 or parts[1] in WHEEL_SKIP:
                        continue
                    relative = "/".join(parts[2:])
                else:
                    relative = name
                if not relative or relative.startswith(("/", "\\")) or ".." in relative.split("/"):
                    logger.warning("Model Chain: Voice Chat refused a wheel member that would "
                                   "have been written outside the runtime")
                    continue
                where = (destination / relative).resolve()
                try:
                    where.relative_to(destination.resolve())
                except ValueError:
                    logger.warning("Model Chain: Voice Chat refused a wheel member that would "
                                   "have been written outside the runtime")
                    continue
                where.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, open(where, "wb") as handle:
                    shutil.copyfileobj(source, handle)
                top = relative.split("/")[0]
                if top not in added:
                    added.append(top)
    except (OSError, zipfile.BadZipFile) as exc:
        raise VoiceError(f"{wheel.name} is not a readable wheel ({exc.__class__.__name__}). "
                         f"Nothing was installed.") from None
    return added


def _build_environment(staging: Path, wheels: Path, chosen: RuntimePlatform,
                       import_name: str = "") -> None:
    """An interpreter of its own, and the verified wheels unpacked into it.

    Two deliberate simplifications, both of them removing a failure this feature
    actually hit.

    The virtual environment is built *without* pip. ``ensurepip`` is one of the
    likelier things to be missing or broken on the embedded and relocated
    Pythons a WebUI is often launched from, and it was being installed only to
    be used once. Skipping it is faster and takes a whole class of failure away.

    The wheels are then unpacked rather than installed by a package manager. See
    :func:`_unpack_wheel`: a pip that exits zero having installed into somebody's
    user site or a ``PIP_TARGET`` directory is a pip that has "succeeded" and
    left the runtime empty, which is exactly what happened and exactly what
    ``ModuleNotFoundError`` in the self-test meant.
    """
    import venv

    environment = staging / "env"
    builder = venv.EnvBuilder(with_pip=False, clear=True, symlinks=(os.name != "nt"))
    try:
        builder.create(environment)
    except Exception as exc:
        logger.warning("Model Chain: Voice Chat could not create a virtual environment at %s "
                       "using %s — %s: %s", environment, sys.executable,
                       exc.__class__.__name__, exc)
        raise VoiceError(f"The isolated Voice Chat runtime could not be created "
                         f"({exc.__class__.__name__}: {exc}).") from None

    interpreter = ((environment / "Scripts" / "python.exe") if os.name == "nt"
                   else (environment / "bin" / "python"))
    if not interpreter.exists():
        raise VoiceError("The isolated Voice Chat runtime was created without an interpreter.")

    target = site_packages(environment)
    for item in chosen.artifacts:
        wheel = wheels / item.local_name
        if not wheel.is_file():
            raise VoiceError(f"{item.filename} is missing from the staged download. Nothing "
                             f"was installed.")
        added = _unpack_wheel(wheel, target)
        logger.info("Model Chain: Voice Chat unpacked %s into the isolated runtime (%s)",
                    item.filename, ", ".join(added[:6]) or "nothing")

    # Checked rather than assumed, because "the installer said it worked" is
    # precisely the claim that turned out to be worthless.
    #
    # ``import_name`` is a parameter rather than always the speech manifest's,
    # because this function has more than one caller now: a second component
    # building a closure of its own would otherwise be checked for *sherpa-onnx*
    # and told nothing was installed, having installed everything.
    engine = import_name or manifest()["runtime_import"]
    if not (target / engine).exists() and not list(target.glob(engine + "*")):
        raise VoiceError(f"The Voice Chat wheels were unpacked but {engine} is not in "
                         f"{target}. Nothing was installed.")


def _quote(output: str, limit: int = 1200) -> str:
    """A subprocess's own words, trimmed, for a log line.

    Trimmed from the *end*, because a Python traceback puts the sentence that
    matters on its last line and the first twenty frames are the ones nobody
    needs.
    """
    text = (output or "").strip()
    if len(text) <= limit:
        return text
    return "…" + text[-limit:]


def _run_staged(interpreter: Path, arguments: list, what: str, timeout: float = 300):
    """Run something in the staged runtime and say what happened when it fails.

    The failure this exists for was mine: the smoke test ran a subprocess,
    checked its return code, and threw the output away -- so an installation
    that failed on somebody else's machine reported "could not import its speech
    engine" and there was no way to learn *why* short of asking them to run
    Python by hand. A captured stderr is the difference between one round trip
    and four.
    """
    try:
        result = subprocess.run([str(interpreter)] + [str(a) for a in arguments],
                                capture_output=True, text=True,
                                env={**os.environ, **worker_environment()}, timeout=timeout)
    except Exception as exc:
        logger.warning("Model Chain: Voice Chat could not run %s (%s: %s) — interpreter %s",
                       what, exc.__class__.__name__, exc, interpreter)
        raise VoiceError(f"The staged Voice Chat runtime could not be started "
                         f"({exc.__class__.__name__}). Nothing was installed.") from None
    if result.returncode != 0:
        logger.warning("Model Chain: Voice Chat's %s failed with exit code %s.\n"
                       "  interpreter: %s\n  stderr: %s\n  stdout: %s",
                       what, result.returncode, interpreter,
                       _quote(result.stderr) or "(nothing)",
                       _quote(result.stdout) or "(nothing)")
    return result


def _smoke_test(staging: Path, spec: dict) -> dict:
    """Prove the staged runtime runs, imports, and can be asked for the CPU.

    Before promotion, so an environment that builds and cannot run never becomes
    the installed one. Run through the worker's own ``--selftest``, which is the
    same file that will do the inference: a smoke test that imports something
    else is a smoke test of something else.

    Two steps, not one, because they fail for entirely different reasons and
    used to produce the same sentence. A venv whose interpreter will not start
    is a broken environment -- which is a real outcome when the WebUI is running
    on an embedded or relocated Python. A venv that starts and cannot import
    sherpa-onnx is a broken package for this platform. Telling them apart is the
    difference between "reinstall Python" and "this build is wrong for your
    machine".
    """
    interpreter = ((staging / "env" / "Scripts" / "python.exe") if os.name == "nt"
                   else (staging / "env" / "bin" / "python"))
    if not interpreter.exists():
        raise VoiceError(f"The isolated Voice Chat runtime has no interpreter at "
                         f"{interpreter}. Nothing was installed.")

    alive = _run_staged(interpreter, ["-c", "import sys; print(sys.version)"],
                        "check that the isolated interpreter runs", timeout=120)
    if alive.returncode != 0:
        raise VoiceError(
            "The isolated Voice Chat runtime was created but its interpreter will not run. "
            "That usually means this WebUI is running on an embedded or relocated Python "
            "that cannot make a working virtual environment. The interpreter's own error is "
            "in the console. Nothing was installed.")
    logger.info("Model Chain: Voice Chat's isolated interpreter reports %s",
                _quote(alive.stdout, 200))

    result = _run_staged(interpreter, [paths.worker_script(), "--selftest"],
                         "speech engine self-test")
    if result.returncode != 0:
        raise VoiceError(
            "The isolated Voice Chat runtime could not import its speech engine. Its own "
            "error is in the console and in model_chain.log. Nothing was installed and "
            "Voice Chat is unchanged.")
    try:
        report = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        logger.warning("Model Chain: the Voice Chat self-test printed %s",
                       _quote(result.stdout) or "(nothing)")
        raise VoiceError("The staged Voice Chat runtime did not report what it is. Nothing was "
                         "installed.") from None
    if report.get("provider") != "cpu":
        raise VoiceError(f"The staged Voice Chat runtime reports provider "
                         f"{report.get('provider')!r} rather than CPU, so it was not installed.")
    if report.get("runtime_version") and report["runtime_version"] != spec["runtime_version"]:
        raise VoiceError(f"The staged Voice Chat runtime is sherpa-onnx "
                         f"{report['runtime_version']}, not the pinned "
                         f"{spec['runtime_version']}. Nothing was installed.")
    # Reported, never enforced. NumPy is what lets sherpa hand back a sentence
    # of audio while it is still computing the next one; without it speech
    # arrives a segment at a time, which is slower and is not a reason to refuse
    # an installation that otherwise works. Section 6.7: no optimization turns a
    # recoverable latency problem into "Voice Chat does not speak".
    if report.get("numpy_version"):
        logger.info("Model Chain: Voice Chat's isolated runtime has NumPy %s, so speech can "
                    "stream a sentence at a time", report["numpy_version"])
    else:
        logger.warning("Model Chain: Voice Chat's isolated runtime could not import NumPy (%s), "
                       "so speech will arrive one segment at a time rather than one sentence "
                       "at a time", report.get("numpy_error") or "no reason reported")
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


def _make_room(artifacts, staging: Path, expectations: dict | None = None) -> None:
    """Refuse before downloading when the disk clearly cannot take it.

    Sized from what the publisher said where this repository has not pinned a
    byte count, so a bundle nobody pinned still gets the check rather than
    silently skipping it for want of a number.
    """
    wanted = sum(((expectations or {}).get(item.local_name) or Expected(None, None, "")).size
                 or item.approximate_bytes for item in artifacts)
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
def _claim(kind: str, say=None, identifier: str = ""):
    """One install per model kind at a time, and a status line while it runs.

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
        # ``model`` rather than only ``kind``, because a kind is now three
        # tiers and a row that knew a download was running but not which of
        # its three buttons started it would put the bar on all of them.
        _progress[kind] = {"running": True, "text": "Starting…", "fraction": 0.0,
                           "failed": False, "model": str(identifier or "")}
    try:
        yield
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        with _lock:
            _progress[kind] = {"running": False, "text": reason, "failed": True,
                               "fraction": 0.0, "model": str(identifier or "")}
        logger.warning("Model Chain: Voice Chat could not install the %s model — %s",
                       kind.upper(), reason)
        # Everything somebody would have to be asked for, written down without
        # being asked. An install that failed on one machine is diagnosed from
        # what that machine is, and a log that omits it costs a round trip.
        logger.warning("Model Chain: Voice Chat install environment — %s", describe_host())
        logger.debug("Model Chain: the Voice Chat install failed here", exc_info=True)
        raise
    else:
        with _lock:
            _progress[kind] = {"running": False, "text": "Installed.", "fraction": 1.0,
                               "failed": False, "model": str(identifier or "")}


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
