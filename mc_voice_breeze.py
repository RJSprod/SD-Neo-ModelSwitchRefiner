"""Breeze TTS 2: what is installed, how its worker is started, and what a render needs.

The Voice Box's second speech engine, product side. Everything a person can see,
choose or install lives here; everything that needs a tensor lives in
``breeze_worker/worker.py``, behind :mod:`mc_voice_breeze_runtime`, in a process
of its own with a CUDA PyTorch closure of its own.

It has :mod:`mc_voice_vibevoice`'s shape on purpose -- a manifest that is the
trust root, a staged install that promotes by rename, a settings file rather
than Forge options, a smoke test before anything is promoted, a calibration of
what a render really cost -- and the Voice Box reads both through the same
adapter contract (``FIELDS``, ``clean``, ``refuse``, ``infotext``,
``capabilities``; docs/25-voice-box-engines.md). What differs:

    Its runtime is four things.     The PyPI closure, pinned byte for byte; torch
                                    and torchaudio 2.9.1, resolved from the
                                    publisher's CUDA 12.8 index on this machine
                                    (:func:`_resolve_wheel`); the community
                                    Flash attention 2 wheel for exactly that
                                    torch and CPython 3.13 (:func:`flash_artifact`,
                                    checked against the digest its host states);
                                    and Breeze's own inference code, which is not
                                    a wheel at all: its repository's files at one
                                    commit, each pinned with its SHA-256 and put
                                    in ``runtime/breeze-tts`` (:func:`source_files`),
                                    which the worker puts on its path.

    Its model is listed, not named. The hub's own file tree, at a commit, says
                                    what the model is: every file with its size
                                    and the digest the hub states -- the LFS
                                    SHA-256 of a weight, the git blob id of a
                                    small file (:func:`_listing`). The first
                                    install records the commit it used, and every
                                    later install lists that same commit, so a
                                    reinstall is the same model until a
                                    maintainer pins another
                                    (``tools/pin_breeze_models.py --model``).

    Its voices carry words.        A Breeze clone needs the sample's exact
                                    transcript, and a voice may be designed from
                                    a description and kept as a sample (the
                                    Voice Box's, :mod:`mc_voice_box`; the
                                    capabilities below say so).

    Its fast path is a setting.    Upstream's CUDA-graph stages, at the cost of
                                    more of the card while warm
                                    (:func:`need_vram_bytes` asks for it, and
                                    the calibration keeps its own figure).

The licence is stated wherever the model is: the inference code is Apache-2.0,
the weights are the BreezeBlue Research and Non-Commercial License, which does
not permit commercial use of the model or of what it renders.
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
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import mc_voice_box_fields as box_fields
import mc_voice_models as models
import mc_voice_paths as paths
import mc_voice_wheelindex as wheelindex
from breeze_worker import worker as protocol

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

ID = "breeze"
"""This engine's id in the Voice Box's registry (:mod:`mc_voice_box_engines`)."""

LABEL = "Breeze TTS 2"

GUEST = "Breeze"
"""The name the lead registers this guest under with ``mc_turns``."""

MODEL_DEFAULT = "breeze-tts-2"

KIND = "breeze"
"""What the shared installer's progress table calls this engine's work."""

SCHEMA = 1
"""The manifest and installed-marker layout this build understands."""

SAMPLE_RATE = protocol.SAMPLE_RATE

MAX_TAKES = protocol.MAX_TAKES
"""How many takes one render may make, each at the next seed (the worker's limit)."""

CFG_MIN = 1.0
CFG_MAX = protocol.CFG_MAX
CFG_DEFAULT = protocol.CFG_DEFAULT
"""Upstream's own default guidance is 1.0, which is the only value its plain
clone accepts. The 4.0 here is the owner's choice for the lines it applies to
-- a line with a direction and a designed voice -- and a plain cloned line is
always spoken at 1.0 whatever this says (the worker's rule, docs/25 §4)."""

GAP_MIN_MS = 0
GAP_MAX_MS = protocol.GAP_MAX_MS
GAP_DEFAULT_MS = protocol.GAP_DEFAULT_MS
MAX_NEW_TOKENS = protocol.MAX_NEW_TOKENS
"""The longest one generation may run: 1500 codec frames, two minutes of
speech at 12.5 frames a second. A longer line is split at its sentences."""

SEED_MAX = 2 ** 31 - 1

WEIGHTS_BYTES_DEFAULT = 7_700_000_000
WORKING_BYTES_DEFAULT = 1024 * 1024 * 1024
FAST_BYTES_DEFAULT = 7_200_000_000
RAM_BYTES_DEFAULT = 8 * 1024 * 1024 * 1024
"""What a render needs when nothing has been measured yet. Upstream states about
7.7 GiB for eager inference and 14.4 GiB with every fast stage on; the weights'
real size replaces the first figure once the model is listed, the working set
is generous on purpose until :func:`note_peak` has a number, and the fast path
is asked for on top while it is on."""

UPSTREAM = {"eager": "upstream"}
"""Which attention upstream's own command line loads, said beside it in the list."""

SETTINGS_DEFAULTS = {
    "card_uuid": "",
    "model_id": MODEL_DEFAULT,
    "fast_path": False,
    "keep_warm": True,
}
"""Breeze's engine settings, and the shape :func:`settings` always answers in.

``fast_path`` defaults off: it is quicker after a warm-up that costs time on
every load, and it holds about seven more gigabytes of the card while warm.
"""

SETTINGS_KEYS = tuple(SETTINGS_DEFAULTS)
"""What the Voice Box's settings route may hand this engine (``/settings``)."""

_lock = threading.RLock()
_manifest_cache = None

_HUB = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")
_SAFE_UUID = re.compile(r"^[A-Za-z0-9,\-]*$")
_HEX = set("0123456789abcdef")

HUB_REVISION = "https://huggingface.co/api/models/{repo}/revision/{revision}"
HUB_TREE = "https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
HUB_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"


class BreezeError(RuntimeError):
    """A Breeze operation that was refused or could not be completed.

    One class, because every ordinary refusal -- not installed, no card, an
    unsupported machine, a setting out of range -- is a state a user can act on
    rather than a fault, and the page shows the sentence.
    """


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def manifest(refresh: bool = False) -> dict:
    """The parsed, validated Breeze manifest. Read once and kept."""
    global _manifest_cache

    with _lock:
        if _manifest_cache is not None and not refresh:
            return _manifest_cache
        path = paths.breeze_manifest_path()
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BreezeError(f"The Breeze manifest at {path} could not be read ({exc}). This "
                              f"is a problem with the extension rather than with your "
                              f"installation.") from None
        if not isinstance(found, dict) or int(found.get("schema") or 0) != SCHEMA:
            raise BreezeError("The Breeze manifest is not in a layout this build understands.")
        runtime = found.get("runtime")
        if not isinstance(runtime, dict) or not runtime.get("platforms"):
            raise BreezeError("The Breeze manifest declares no runtime.")
        if not isinstance(found.get("models"), dict) or not found["models"]:
            raise BreezeError("The Breeze manifest declares no model.")
        _manifest_cache = found
        return found


def _manifest_cache_clear() -> None:
    global _manifest_cache

    with _lock:
        _manifest_cache = None


def _overlay() -> dict:
    """The local overlay, read whole: ``{"artifacts": {...}, "models": {...}}``."""
    try:
        raw = json.loads(paths.breeze_local_pins_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _local_pins() -> dict:
    """``{repo/filename: {"sha256", "bytes"}}`` recorded by earlier installs.

    Only ever fills a blank -- see :func:`_artifact`. The flash-attn wheel is the
    one runtime artifact recorded here; the model is recorded by its listing
    (:func:`_recorded_listing`).
    """
    found = {}
    for name, entry in ((_overlay().get("artifacts") or {}) or {}).items():
        digest = str((entry or {}).get("sha256") or "").strip().casefold()
        size = (entry or {}).get("bytes")
        if len(digest) == 64 and set(digest) <= _HEX and isinstance(size, int) and size > 0:
            found[str(name)] = {"sha256": digest, "bytes": size}
    return found


def _pin_key(repo: str, filename: str) -> str:
    return f"{repo}/{filename}"


def _repo_of(url: str) -> str:
    """``owner/name`` out of a hub URL, or ``""`` for anything else."""
    found = _HUB.match(str(url or ""))
    return found.group(1) if found else ""


def _sha256(value) -> "str | None":
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if len(text) != 64 or not set(text) <= _HEX:
        raise BreezeError("The Breeze manifest gives a file a hash that is not a SHA-256.")
    return text


def _artifact(entry: dict, owner: str = "", repo: str = "", authorized: bool = False
              ) -> models.Artifact:
    """One manifest artifact as the shared downloader's own dataclass.

    A hash committed in the manifest wins; a blank is filled from the local
    overlay, which cannot change a committed one.
    """
    if not isinstance(entry, dict):
        raise BreezeError(f"The Breeze manifest's {owner or 'artifact list'} holds something "
                          f"that is not an artifact.")
    filename = str(entry.get("filename") or "").strip()
    local_name = str(entry.get("local_name") or filename).strip()
    url = str(entry.get("url") or "").strip()
    if not filename or not local_name:
        raise BreezeError(f"The Breeze manifest's {owner} has an artifact with no name.")
    if not url.startswith("https://"):
        raise BreezeError(f"The Breeze manifest's {owner} names {filename} somewhere other "
                          f"than an HTTPS address.")
    if "/" in local_name or "\\" in local_name or local_name in (".", ".."):
        raise BreezeError(f"The Breeze manifest's {owner} names a file ({local_name!r}) that "
                          f"would not stay in its own folder.")
    sha = _sha256(entry.get("sha256"))
    size = entry.get("bytes")
    size = int(size) if isinstance(size, (int, float)) and size > 0 else None
    if sha is None or size is None:
        pinned = _local_pins().get(_pin_key(repo or _repo_of(url), filename))
        if pinned:
            sha = sha or pinned["sha256"]
            size = size or pinned["bytes"]
    return models.Artifact(filename=filename, local_name=local_name, url=url, size=size,
                           sha256=sha, authorized=bool(authorized))


@dataclass(frozen=True)
class ModelFile:
    """One file of the model, as a listing names it."""

    path: str
    bytes: int
    sha256: str = ""
    git_oid: str = ""

    @property
    def weight(self) -> bool:
        return self.path.endswith(".safetensors")


@dataclass(frozen=True)
class Bundle:
    """One installable Breeze model, as the manifest declares it."""

    identifier: str
    label: str
    repo: str
    revision: str
    commit: str
    license: str
    attribution: str
    summary: str
    notes: str
    sample_rate: int
    authorized: bool
    skip: tuple
    skip_suffixes: tuple
    files: tuple
    estimates: dict
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def pinned(self) -> bool:
        """Whether the manifest itself commits the listing: a commit and every file."""
        return bool(self.commit) and bool(self.files) and all(
            item.bytes > 0 and (len(item.sha256) == 64 or len(item.git_oid) == 40)
            for item in self.files)

    @property
    def weights_bytes(self) -> int:
        return sum(item.bytes for item in self.files if item.weight)

    @property
    def estimated_weights_bytes(self) -> int:
        return self.weights_bytes or int(self.estimates.get("weights_bytes")
                                         or WEIGHTS_BYTES_DEFAULT)

    @property
    def download_bytes(self) -> int:
        listed = sum(item.bytes for item in self.files)
        return listed or self.estimated_weights_bytes


def _model_file(row: dict, owner: str) -> ModelFile:
    if not isinstance(row, dict):
        raise BreezeError(f"The Breeze manifest's {owner} lists something that is not a file.")
    path = _safe_relative(row.get("path"), owner)
    sha = str(row.get("sha256") or "").strip().casefold()
    oid = str(row.get("git_oid") or "").strip().casefold()
    if sha and (len(sha) != 64 or not set(sha) <= _HEX):
        raise BreezeError(f"The Breeze manifest's {owner} gives {path} a hash that is not a "
                          f"SHA-256.")
    if oid and (len(oid) != 40 or not set(oid) <= _HEX):
        raise BreezeError(f"The Breeze manifest's {owner} gives {path} a blob id that is not "
                          f"one.")
    return ModelFile(path=path, bytes=max(int(row.get("bytes") or 0), 0), sha256=sha,
                     git_oid=oid)


def _safe_relative(value, owner: str = "listing") -> str:
    """A relative path that stays inside the folder it is put in, or a refusal."""
    text = str(value or "").strip()
    pure = PurePosixPath(text)
    if not text or "\\" in text or text.startswith("/") or ":" in text \
            or any(part in ("", ".", "..") for part in text.split("/")) or pure.is_absolute():
        raise BreezeError(f"The Breeze {owner} names a file ({text!r}) that would not stay in "
                          f"the model's folder.")
    return text


def model_ids() -> tuple:
    """Every Breeze model this build knows, in manifest order."""
    try:
        return tuple((manifest().get("models") or {}).keys())
    except BreezeError:
        return ()


def _default_model_id() -> str:
    known = model_ids()
    try:
        offered = str((manifest().get("defaults") or {}).get("model") or "")
    except BreezeError:
        return ""
    return offered if offered in known else (known[0] if known else "")


def model_id() -> str:
    """The selected model, or the manifest default."""
    found = str(_settings_read().get("model_id") or "").strip()
    return found if found in model_ids() else _default_model_id()


def bundle(identifier: str = "") -> Bundle:
    """The model this build installs, by id or by the selected one."""
    found = manifest()
    wanted = str(identifier or model_id() or _default_model_id())
    entry = (found.get("models") or {}).get(wanted)
    if not isinstance(entry, dict):
        raise BreezeError(f"{wanted!r} is not a Breeze model this build knows.")
    repo = str(entry.get("repo") or "")
    if not re.match(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$", repo):
        raise BreezeError(f"The Breeze manifest names {wanted}'s repository as {repo!r}.")
    commit = str(entry.get("commit") or "").strip().casefold()
    if commit and (len(commit) != 40 or not set(commit) <= _HEX):
        raise BreezeError(f"The Breeze manifest's {wanted} names a commit that is not one.")
    return Bundle(
        identifier=wanted,
        label=str(entry.get("label") or wanted),
        repo=repo,
        revision=str(entry.get("revision") or "main"),
        commit=commit,
        license=str(entry.get("license") or ""),
        attribution=str(entry.get("attribution") or ""),
        summary=str(entry.get("summary") or ""),
        notes=str(entry.get("notes") or ""),
        sample_rate=int(entry.get("sample_rate") or SAMPLE_RATE),
        authorized=bool(entry.get("authorized")),
        skip=tuple(str(name) for name in (entry.get("skip") or ())),
        skip_suffixes=tuple(str(name).casefold() for name in (entry.get("skip_suffixes") or ())),
        files=tuple(_model_file(row, wanted) for row in (entry.get("files") or ())),
        estimates=dict(entry.get("estimates") or {}),
        raw=entry)


# --------------------------------------------------------------------------- #
# The runtime's four parts
# --------------------------------------------------------------------------- #


def _platform_entry():
    system, machine, python_version = models.current_platform()
    for entry in manifest()["runtime"].get("platforms") or ():
        if (str(entry.get("system") or "").casefold() == system
                and machine in tuple(str(m).casefold() for m in (entry.get("machines") or ()))
                and str(entry.get("python")) == python_version):
            return entry
    return None


def platform() -> "models.RuntimePlatform | None":
    """The pinned PyPI closure for this machine, or ``None`` if it is not on the list.

    An allowlist of one: 64-bit Windows, CPython 3.13, an NVIDIA card driven by
    CUDA 12.8 -- the user's own machine. torch, torchaudio and the flash-attn
    wheel are *not* among the artifacts returned: the first two are resolves
    (:func:`resolves`), the third a declared download (:func:`flash_artifact`).
    """
    entry = _platform_entry()
    if entry is None:
        return None
    owner = f"runtime {entry.get('id') or ''}"
    artifacts = tuple(_artifact(item, owner) for item in (entry.get("artifacts") or ())
                      if isinstance(item, dict) and not item.get("resolve")
                      and not item.get("declared"))
    return models.RuntimePlatform(
        identifier=str(entry.get("id") or ""),
        system=str(entry.get("system") or "").casefold(),
        machines=tuple(str(m).casefold() for m in (entry.get("machines") or ())),
        python=str(entry.get("python") or ""),
        artifacts=artifacts)


def resolves() -> list:
    """The wheels resolved from the publisher's index for this platform: torch, torchaudio.

    Each ``{"local_name", "index", "package", "version", "filename", "sha256",
    "bytes"}``, the last three empty until a maintainer runs
    ``tools/pin_breeze_models.py --torch``.
    """
    entry = _platform_entry()
    if entry is None:
        return []
    found = []
    for item in entry.get("artifacts") or ():
        resolve = item.get("resolve") if isinstance(item, dict) else None
        if resolve is None:
            continue
        if not isinstance(resolve, dict):
            raise BreezeError("The Breeze manifest's runtime has a resolve that is not an "
                              "object.")
        base = str(resolve.get("index") or "")
        package = str(resolve.get("package") or "")
        if not base.startswith("https://") or not package:
            raise BreezeError("The Breeze manifest's runtime resolves a wheel from an index "
                              "that is not HTTPS, or names no package.")
        if item.get("url") or item.get("sha256"):
            raise BreezeError(f"The Breeze manifest both pins {package} and resolves it. One "
                              f"or the other.")
        digest = str(resolve.get("sha256") or "").casefold()
        if digest and (len(digest) != 64 or not set(digest) <= _HEX):
            raise BreezeError(f"The Breeze manifest records a hash for {package} that is not "
                              f"a SHA-256.")
        found.append({"local_name": str(item.get("local_name") or package), "index": base,
                      "package": package, "version": str(resolve.get("version") or ""),
                      "filename": str(resolve.get("filename") or ""), "sha256": digest or "",
                      "bytes": int(resolve.get("bytes") or 0)})
    return found


def full_version(resolve: dict) -> str:
    """``2.9.1`` on a ``/whl/cu128/`` index is ``2.9.1+cu128`` on its page."""
    version = str((resolve or {}).get("version") or "")
    if "+" in version or not version:
        return version
    found = re.search(r"/whl/([A-Za-z0-9_]+)/", str((resolve or {}).get("index") or ""))
    return f"{version}+{found.group(1)}" if found else version


def flash_artifact() -> "models.Artifact | None":
    """The community Flash attention 2 wheel this platform installs, or ``None``.

    Declared by URL: built for CPython 3.13, torch 2.9.1 and CUDA 12.8 on
    Windows by Wildminder's AI-windows-whl. Checked against the digest its host
    states at install time (and the overlay's record of it after that) until a
    maintainer commits one with ``--flash-attn``.
    """
    entry = _platform_entry()
    if entry is None:
        return None
    for item in entry.get("artifacts") or ():
        if isinstance(item, dict) and item.get("declared"):
            return _artifact(item, "flash-attn wheel")
    return None


def source() -> dict:
    """Breeze's own code as the manifest pins it: ``{"repo", "commit", "license", "files"}``."""
    found = dict(manifest()["runtime"].get("source") or {})
    commit = str(found.get("commit") or "").strip().casefold()
    if commit and (len(commit) != 40 or not set(commit) <= _HEX):
        raise BreezeError("The Breeze manifest pins its source at something that is not a "
                          "commit.")
    rows = []
    for row in found.get("files") or ():
        if not isinstance(row, dict):
            raise BreezeError("The Breeze manifest's source lists something that is not a "
                              "file.")
        path = _safe_relative(row.get("path"), "source")
        url = str(row.get("url") or "")
        if not url.startswith("https://raw.githubusercontent.com/"):
            raise BreezeError(f"The Breeze manifest's source names {path} somewhere other than "
                              f"the repository's raw host.")
        rows.append({"path": path, "url": url, "bytes": max(int(row.get("bytes") or 0), 0),
                     "sha256": _sha256(row.get("sha256")) or ""})
    return {"repo": str(found.get("repo") or ""), "commit": commit,
            "license": str(found.get("license") or ""), "files": rows}


def source_files() -> list:
    return list(source()["files"])


def pinned() -> bool:
    """Whether this build has pinned its PyPI closure and Breeze's code for this machine.

    torch, torchaudio and flash-attn do not count against it, by design: they
    cannot be pinned from the machine that writes the manifest, and refusing
    every install until they are would be refusing the feature.
    """
    try:
        chosen = platform()
        code = source()
    except BreezeError:
        return False
    if chosen is None or not chosen.artifacts:
        return False
    return all(item.pinned for item in chosen.artifacts) and bool(code["commit"]) \
        and bool(code["files"]) and all(len(row["sha256"]) == 64 for row in code["files"])


def closure_id() -> str:
    """A fingerprint of exactly which runtime this build installs.

    The pinned wheels' digests, torch's and torchaudio's *versions* (not their
    digests: recording those later must not make every installed runtime
    stale), the flash-attn wheel's name, and Breeze's commit with its files'
    digests.
    """
    chosen = platform()
    if chosen is None:
        return ""
    parts = [chosen.closure_id]
    parts += [f"{one['package']}:{full_version(one)}" for one in resolves()]
    flash = flash_artifact()
    parts.append(f"flash:{flash.filename if flash else ''}")
    code = source()
    parts.append(f"source:{code['commit']}")
    parts += [f"{row['path']}:{row['sha256']}" for row in code["files"]]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _runtime_bytes(chosen, picked=None) -> int:
    total = sum(int(item.size or 0) for item in (chosen.artifacts if chosen else ()))
    runtime = {}
    try:
        runtime = manifest()["runtime"]
    except BreezeError:
        pass
    for one in (picked if picked is not None else resolves()):
        about = (runtime.get(one["package"]) or {}).get("about_bytes") \
            if isinstance(runtime.get(one["package"]), dict) else 0
        total += int(one.get("bytes") or about or 0)
    flash = flash_artifact() if chosen is not None else None
    if flash is not None:
        total += int(flash.size or (runtime.get("flash_attn") or {}).get("about_bytes") or 0)
    total += sum(int(row["bytes"]) for row in (source_files() if chosen is not None else ()))
    return total


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


@dataclass
class Status:
    """What is installed, in one object the page, the runtime and the lead read."""

    supported: bool = False
    runtime_installed: bool = False
    model_installed: bool = False
    provisional: bool = True
    stale: bool = False
    stale_message: str = ""
    label: str = LABEL
    model_id: str = ""
    model_label: str = ""
    download_bytes: int = 0
    runtime_bytes: int = 0
    model_bytes: int = 0
    runtime_message: str = ""
    model_message: str = ""
    closure: dict = field(default_factory=dict)
    installed_model: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Whether a render can be attempted: the runtime and the model."""
        return bool(self.supported and self.runtime_installed and self.model_installed)

    @property
    def message(self) -> str:
        if self.ready:
            return "Installed."
        if not self.supported:
            return self.runtime_message
        if self.stale:
            return self.stale_message
        missing = []
        if not self.runtime_installed:
            missing.append("runtime")
        if not self.model_installed:
            missing.append("model")
        return f"Setup required — Breeze's {' and '.join(missing)} still to install."


def status() -> Status:
    """Read from disk. Starts nothing, downloads nothing, never raises."""
    try:
        return _status()
    except BreezeError as exc:
        return Status(runtime_message=str(exc), model_message=str(exc))
    except Exception:
        logger.debug("Model Chain: could not read the Breeze installation", exc_info=True)
        text = "Breeze's installation could not be read."
        return Status(runtime_message=text, model_message=text)


def _unsupported_sentence(tail: str) -> str:
    system, machine, python_version = models.current_platform()
    return (f"Breeze TTS 2 needs Windows with an NVIDIA card on Python 3.13, and this machine "
            f"is {system}/{machine} on Python {python_version}{tail}")


STALE_MESSAGE = ("Breeze's runtime was built by an earlier build of this extension, which "
                 "pinned different packages or a different Breeze — install the runtime again "
                 "to update it.")


def _status() -> Status:
    chosen = platform()
    entry = bundle()
    found = Status(model_id=entry.identifier, model_label=entry.label,
                   provisional=not entry.pinned)
    if chosen is None:
        found.supported = False
        found.runtime_message = _unsupported_sentence(
            ". VibeVoice and Voice Chat's CPU engines are unaffected.")
        found.model_message = found.runtime_message
        return found
    found.supported = True
    found.runtime_bytes = _runtime_bytes(chosen)
    installed = _read_json(paths.breeze_runtime_manifest())
    if not installed and not pinned():
        found.runtime_message = (
            "Not installed — this build has not pinned a Breeze runtime yet, so it will not "
            "download one. A maintainer runs tools/pin_breeze_models.py; you can install from "
            "a folder you filled yourself in the meantime.")
    elif not installed:
        found.runtime_message = (
            f"Not installed — about {models._bytes_label(found.runtime_bytes)} of PyTorch "
            f"(CUDA 12.8), Flash attention 2, transformers, qwen-tts and Breeze's code.")
    elif str(installed.get("closure") or "") != closure_id() or runtime_python() is None:
        found.stale = runtime_python() is not None
        found.stale_message = STALE_MESSAGE if found.stale else ""
        found.runtime_message = (STALE_MESSAGE if found.stale else
                                 "Installed, but its interpreter is missing — install the "
                                 "runtime again.")
    else:
        found.runtime_installed = True
        flash = installed.get("flash_attn") if isinstance(installed.get("flash_attn"),
                                                          dict) else {}
        found.runtime_message = (
            f"Installed — Breeze {str(installed.get('breeze_commit') or '?')[:7]}, torch "
            f"{installed.get('torch_version') or '?'}, transformers "
            f"{installed.get('transformers_version') or '?'}"
            + (f", Flash attention {flash.get('version')}" if flash.get("loads") else
               ", Flash attention 2 would not load" if flash.get("version") else "") + ".")
    found.closure = dict(installed or {})

    root = paths.breeze_model_root(entry.identifier)
    marker = _read_json(root / paths.INSTALLED_FILENAME)
    found.installed_model = dict(marker or {})
    found.model_bytes = _model_bytes(entry, marker)
    missing = _model_missing(root, marker)
    if not marker:
        found.model_message = (f"Not installed — {entry.label}, about "
                               f"{models._bytes_label(found.model_bytes)}. Research and "
                               f"non-commercial use only.")
    elif missing:
        found.model_message = (f"Installed, but {missing[0]} is missing — install the "
                               f"model again.")
    else:
        found.model_installed = True
        size = sum(int(value) for value in ((marker or {}).get("bytes") or {}).values()
                   if isinstance(value, int))
        found.model_message = (f"Installed — {entry.label}"
                               f"{', ' + models._bytes_label(size) if size else ''}, at "
                               f"{str(marker.get('commit') or '?')[:7]}. Research and "
                               f"non-commercial use only.")
    found.download_bytes = ((0 if found.runtime_installed else found.runtime_bytes)
                            + (0 if found.model_installed else found.model_bytes))
    return found


def _model_bytes(entry: Bundle, marker) -> int:
    recorded = (marker or {}).get("bytes") if isinstance((marker or {}).get("bytes"),
                                                         dict) else {}
    total = sum(int(value) for value in recorded.values() if isinstance(value, int))
    if total:
        return total
    listed = _recorded_listing(entry)
    if listed:
        return sum(item.bytes for item in listed[1])
    return entry.download_bytes


def _model_missing(root: Path, marker) -> list:
    """Which installed files are not where the marker says they are."""
    if not marker:
        return []
    wanted = list(((marker or {}).get("bytes") or {}).keys())
    if not wanted:
        return ["config.json"]
    return [name for name in wanted if not (root / name).is_file()]


def _site_packages(environment: Path) -> list:
    candidates = [environment / "Lib" / "site-packages"]
    if (environment / "lib").is_dir():
        candidates += sorted((environment / "lib").glob("python3*/site-packages"))
    return candidates


def flash_attention_installed() -> bool:
    """Whether Breeze's runtime has Flash attention 2, and the install proved it loads.

    From the files and the runtime's own record, not by asking the worker, so
    the page can say so before a worker has ever started: the package is a
    folder in the venv's site-packages, and the install's self-test imported
    it (``flash_attn.loads`` in the runtime's marker). The worker checks again,
    by import, before it ever loads with it.
    """
    environment = paths.breeze_runtime_root() / "env"
    if not any((folder / "flash_attn" / "__init__.py").is_file()
               for folder in _site_packages(environment)):
        return False
    record = _read_json(paths.breeze_runtime_manifest()) or {}
    flash = record.get("flash_attn") if isinstance(record.get("flash_attn"), dict) else {}
    return bool(flash.get("loads", True))


def _flash_reason() -> str:
    record = _read_json(paths.breeze_runtime_manifest()) or {}
    flash = record.get("flash_attn") if isinstance(record.get("flash_attn"), dict) else {}
    if flash.get("version") and not flash.get("loads", True):
        return "installed, but it would not load"
    return "not installed"


# --------------------------------------------------------------------------- #
# The engine's vocabulary, as data the Voice Box reads
# --------------------------------------------------------------------------- #

ATTENTION_NAMES = {"flash_attention_2": "Flash attention 2", "sdpa": "SDPA", "eager": "Eager"}
"""What an output's infotext calls each attention. Flash attention 2 here is the
text encoder's; Breeze's two decoders hand transformers a 4-D attention mask,
which its Flash attention path reads as padding, so the worker runs them eager
under this choice (docs/25 §5)."""

FIELDS = (
    {"key": "cfg_scale", "label": "CFG", "kind": "float", "min": CFG_MIN, "max": CFG_MAX,
     "step": 0.1, "default": CFG_DEFAULT, "place": "head", "infotext": "CFG scale",
     "hint": "Guidance for lines with a direction and for voice design. A line spoken "
             "plainly in a cloned voice always uses 1, which is all upstream allows there."},
    {"key": "attention", "label": "Attention", "kind": "choice", "noun": "an attention",
     "options": [{"id": key, "name": name} for key, name in ATTENTION_NAMES.items()],
     "default": protocol.ATTENTION_DEFAULT, "place": "head", "infotext": "Attention",
     "hint": "Flash attention 2 runs Breeze's text encoder; its decoders run eager."},
    {"key": "max_new_tokens", "label": "Max new tokens", "kind": "int", "min": 1,
     "max": MAX_NEW_TOKENS, "step": 1, "blank": "automatic", "default": None,
     "place": "tail", "infotext": "Max new tokens",
     "hint": "Codec frames per generation, 12.5 a second; automatic sizes each line."},
    {"key": "line_gap_ms", "label": "Line gap (ms)", "kind": "int", "min": GAP_MIN_MS,
     "max": GAP_MAX_MS, "step": 10, "default": GAP_DEFAULT_MS, "place": "tail",
     "infotext": "Line gap", "hint": "Silence between one speaker's line and the next's."},
    {"key": "sampling", "label": "Sampling", "kind": "bool", "default": True,
     "place": "tail", "wide": True},
    {"key": "temperature", "label": "Temperature", "kind": "float", "min": 0.1, "max": 2.0,
     "step": 0.05, "default": protocol.SAMPLING_TEMPERATURE, "needs": "sampling",
     "place": "tail", "infotext": "Temperature"},
    {"key": "top_p", "label": "Top-p", "kind": "float", "min": 0.05, "max": 1.0, "step": 0.01,
     "default": protocol.SAMPLING_TOP_P, "needs": "sampling", "place": "tail",
     "infotext": "Top-p"},
)
"""Breeze's configuration fields, in the order the Configuration stage draws them:
guidance and attention before the seed and the batch; the token cap, the line
gap and sampling after. Sampling is on by default because upstream's own
generation configuration samples (temperature 0.9, top-p 1.0, top-k 50); off is
greedy, which upstream never uses."""


def _attention_options() -> list:
    flash = flash_attention_installed()
    reason = "" if flash else _flash_reason()
    found = []
    for key, name in ATTENTION_NAMES.items():
        available = key != "flash_attention_2" or flash
        label = f"{name} ({UPSTREAM[key]})" if key in UPSTREAM else name
        found.append({"id": key, "name": name, "label": label, "available": available,
                      "reason": "" if available else reason})
    return found


def fields() -> list:
    """:data:`FIELDS` as the page draws them: every choice labelled, and Flash
    attention 2 listed unavailable, with the reason, until the runtime has it."""
    found = []
    for item in FIELDS:
        entry = dict(item)
        if item["key"] == "attention":
            entry["options"] = _attention_options()
        found.append(entry)
    return found


def options() -> dict:
    """The attentions and batch a render may ask for, in VibeVoice's ``options`` shape."""
    return {"attention": _attention_options(), "batch_max": MAX_TAKES,
            "defaults": {"attention": protocol.ATTENTION_DEFAULT, "batch": 1}}


def defaults() -> dict:
    """A new configuration's options for Breeze."""
    return box_fields.defaults(FIELDS)


def clean(values: dict, previous: "dict | None" = None) -> dict:
    """One configuration's Breeze options, checked against :data:`FIELDS`."""
    try:
        return box_fields.clean(FIELDS, values, previous, engine=LABEL)
    except box_fields.FieldError as exc:
        raise BreezeError(str(exc)) from None


def refuse(chosen: dict) -> str:
    """Why a configuration's options cannot render on this machine, or ``""``."""
    if str((chosen or {}).get("attention") or "") == "flash_attention_2" \
            and not flash_attention_installed():
        if _flash_reason() == "not installed":
            return ("Flash attention 2 is not in Breeze's runtime yet — install the runtime, "
                    "or choose SDPA or Eager.")
        return ("Flash attention 2 is in Breeze's runtime but would not load on this machine. "
                "Choose SDPA or Eager.")
    return ""


EVENTS = {"en": ["(laugh)", "(sigh)", "(cough)", "(clears throat)"],
          "zh": ["[笑]", "[叹气]", "[咳嗽]", "[清嗓子]"]}
"""The inline vocal events Breeze performs rather than reads, by language: its
model card's own list, written into a line where the event happens."""


def capabilities() -> dict:
    """What the Voice Box may ask of Breeze beyond a script and its samples.

    A line at a time in its speaker's voice, and takes one after another; a
    voice is a sample *with its exact words*, or a description to design one
    from, with a standing direction; a line may carry its own direction; and
    the vocal events in :data:`EVENTS` are performed.
    """
    return {"speakers": "per_line", "takes": "sequential", "design": True, "directions": True,
            "voice": {"sample": True, "transcript": True, "description": True,
                      "instruction": True},
            "events": {key: list(value) for key, value in EVENTS.items()},
            "languages": ["en", "zh"]}


def infotext(render: dict, shared: dict) -> list:
    """The parameters of a Breeze output's infotext, in order, as ``(label, value)``.

    The Voice Box names the engine first (``Engine: Breeze TTS 2``); then CFG
    scale, Seed, the sampling (Temperature and Top-p, or "greedy"), Attention,
    Model, each speaker and its direction, the line gap and the token cap.
    """
    values = dict(dict(render or {}).get("options") or {})

    def number(value):
        return f"{float(value):g}" if isinstance(value, (int, float)) \
            and not isinstance(value, bool) else value

    parts = [("CFG scale", number(values.get("cfg_scale"))), ("Seed", shared.get("seed"))]
    if values.get("sampling", True):
        parts += [("Temperature", number(values.get("temperature"))),
                  ("Top-p", number(values.get("top_p")))]
    else:
        parts.append(("Sampling", "greedy"))
    parts += [("Attention", ATTENTION_NAMES.get(str(values.get("attention") or ""))),
              ("Model", shared.get("model"))]
    parts += list(shared.get("speakers") or ())
    gap = values.get("line_gap_ms")
    parts.append(("Line gap", f"{int(gap)} ms" if isinstance(gap, (int, float))
                  and not isinstance(gap, bool) else None))
    parts.append(("Max new tokens", values.get("max_new_tokens")))
    return parts


def public_status() -> dict:
    """Everything the page needs, JSON-safe, in one answer."""
    found = status()
    try:
        is_pinned = pinned()
    except Exception:
        is_pinned = False
    try:
        vram, ram = need_vram_bytes(found.model_id), need_ram_bytes(found.model_id)
    except Exception:
        vram, ram = 0, 0
    return {
        "id": ID,
        "installed": found.ready,
        "ready": found.ready,
        "supported": found.supported,
        "runtime_installed": found.runtime_installed,
        "model_installed": found.model_installed,
        "provisional": found.provisional,
        "pinned": is_pinned,
        "stale": found.stale,
        "stale_message": found.stale_message,
        "label": found.label,
        "message": found.message,
        "runtime_message": found.runtime_message,
        "model_message": found.model_message,
        "model_id": found.model_id,
        "model_label": found.model_label,
        "models": [{"id": name, "label": _model_label(name)} for name in model_ids()],
        "download_bytes": int(found.download_bytes),
        "download_label": models._bytes_label(found.download_bytes),
        "parts": [
            {"id": "runtime", "label": "Runtime — PyTorch (CUDA 12.8), Flash attention 2, "
                                       "transformers, qwen-tts and Breeze",
             "installed": found.runtime_installed, "bytes": int(found.runtime_bytes),
             "message": found.runtime_message},
            {"id": "model", "label": found.model_label or LABEL,
             "installed": found.model_installed, "bytes": int(found.model_bytes),
             "message": found.model_message},
        ],
        "license": _license_line(),
        "settings": settings(),
        "options": options(),
        "fields": fields(),
        "capabilities": capabilities(),
        "max_takes": MAX_TAKES,
        "need_vram_bytes": int(vram),
        "need_ram_bytes": int(ram),
        "refusal": refusal(),
        "progress": progress(),
    }


def _license_line() -> str:
    return ("Breeze TTS 2's weights are under the BreezeBlue Research and Non-Commercial "
            "License: research and non-commercial use only, renders included. Its inference "
            "code is Apache-2.0.")


def _model_label(identifier: str) -> str:
    try:
        return bundle(identifier).label
    except BreezeError:
        return str(identifier)


# --------------------------------------------------------------------------- #
# The worker's side: interpreter, environment, script, model directory
# --------------------------------------------------------------------------- #


def runtime_python() -> "Path | None":
    """The isolated Breeze interpreter, or ``None`` when it is not installed."""
    interpreter = _interpreter_in(paths.breeze_runtime_root())
    return interpreter if interpreter.exists() else None


def _interpreter_in(root: Path) -> Path:
    environment = root / "env"
    return ((environment / "Scripts" / "python.exe") if os.name == "nt"
            else (environment / "bin" / "python"))


def worker_environment(card_uuid: str, source_root=None) -> dict:
    """The environment the Breeze worker runs in, on one card.

    ``CUDA_VISIBLE_DEVICES`` is the card's UUID as nvidia-smi spells it, so the
    worker's device 0 is that card and no other (the runtime's handshake proves
    it). ``MODEL_CHAIN_BREEZE_SOURCE`` names Breeze's code at the pinned
    commit, which the worker puts on its path. ``HF_HUB_OFFLINE`` and
    ``TRANSFORMERS_OFFLINE`` make any reach for the hub a refusal: everything
    the model needs is in its folder. No credential is here, and there is no
    branch that could put one here.
    """
    found = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "4",
        protocol.SOURCE_VARIABLE: str(source_root if source_root is not None
                                      else paths.breeze_source_root()),
    }
    wanted = str(card_uuid or "").strip()
    if wanted:
        if not _SAFE_UUID.match(wanted) or len(wanted) > 128:
            raise BreezeError(f"{wanted!r} is not a card identifier.")
        found["CUDA_VISIBLE_DEVICES"] = wanted
    return found


def worker_script() -> Path:
    return paths.breeze_worker_script()


def model_dir(identifier: str = "") -> Path:
    """Where the selected (or named) model lives. Validated, never a caller's path."""
    return paths.breeze_model_root(bundle(identifier).identifier)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def settings() -> dict:
    """Breeze's engine settings, defaults filled in, stored values validated."""
    stored = _settings_read()
    found = dict(SETTINGS_DEFAULTS)
    for key in SETTINGS_DEFAULTS:
        if key in stored:
            try:
                found[key] = _validated(key, stored[key])
            except BreezeError:
                logger.debug("Model Chain: a stored Breeze setting (%s) was ignored", key)
    if found["model_id"] not in model_ids():
        found["model_id"] = _default_model_id() or MODEL_DEFAULT
    return found


def set_settings(values: dict) -> dict:
    """Change engine settings. Validated, refused rather than dropped.

    The fast path takes effect on the next load: a warm worker keeps the path
    it was loaded with until it leaves the card (the page says so).
    """
    offered = {str(key): value for key, value in dict(values or {}).items()}
    unknown = sorted(set(offered) - set(SETTINGS_DEFAULTS))
    if unknown:
        raise BreezeError(f"{unknown[0]!r} is not a Breeze setting.")
    checked = {key: _validated(key, value) for key, value in offered.items()}
    current = _settings_read()
    current.update(checked)
    _settings_write(current)
    if checked:
        logger.info("Model Chain: Breeze settings changed — %s", ", ".join(sorted(checked)))
    return settings()


def _switch(value, what: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes",
                                                            "no", "on", "off"):
        return value.strip().lower() in ("true", "1", "yes", "on")
    raise BreezeError(f"{what} is on or off.")


def _validated(key: str, value):
    if key in ("keep_warm", "fast_path"):
        return _switch(value, "Keep warm" if key == "keep_warm" else "The fast path")
    if key == "card_uuid":
        text = str(value or "").strip()
        if text and (len(text) > 128 or not _SAFE_UUID.match(text) or "," in text):
            raise BreezeError(f"{text!r} is not a card identifier.")
        return text
    if key == "model_id":
        text = str(value or "").strip()
        if not text:
            return _default_model_id() or MODEL_DEFAULT
        if text not in model_ids():
            raise BreezeError(f"{text!r} is not a Breeze model this build has recorded.")
        return text
    raise BreezeError(f"{key!r} is not a Breeze setting.")


def _settings_read() -> dict:
    try:
        found = json.loads(paths.breeze_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _settings_write(found: dict) -> None:
    _write_json(paths.breeze_settings_path(), found)


# --------------------------------------------------------------------------- #
# What a render needs
# --------------------------------------------------------------------------- #


def _calibration_key(identifier: str, fast_path: bool) -> str:
    return f"{identifier}+fast" if fast_path else identifier


def calibration() -> dict:
    """``{key: {"peak_bytes", "rss_bytes", "renders", "updated"}}`` observed here.

    Keyed by model id, and by ``<model id>+fast`` for renders made on the fast
    path, whose graphs hold more of the card.
    """
    found = _read_json(paths.breeze_calibration_path()) or {}
    out = {}
    for name, entry in (found.get("models") or {}).items():
        if isinstance(entry, dict):
            out[str(name)] = {
                "peak_bytes": int(entry.get("peak_bytes") or 0),
                "rss_bytes": int(entry.get("rss_bytes") or 0),
                "renders": int(entry.get("renders") or 0),
                "updated": str(entry.get("updated") or ""),
            }
    return out


def note_peak(identifier: str, peak_bytes: int, rss_bytes: int = 0, takes: int = 1,
              fast_path: bool = False) -> None:
    """Remember the most a render has cost on this machine. Never raises.

    Breeze renders its takes one after another, so a batch's peak is one
    take's and is kept as it is (``takes`` is accepted for the runtime's
    symmetry with VibeVoice's and does not scale anything).
    """
    name = str(identifier or "").strip()
    try:
        peak = max(int(peak_bytes or 0), 0)
        rss = max(int(rss_bytes or 0), 0)
        int(takes or 1)
    except (TypeError, ValueError):
        return
    if not name or len(name) > 64 or not all(c.isalnum() or c in "-_." for c in name) \
            or (peak <= 0 and rss <= 0):
        return
    key = _calibration_key(name, bool(fast_path))
    with _lock:
        found = _read_json(paths.breeze_calibration_path()) or {}
        entries = found.get("models") if isinstance(found.get("models"), dict) else {}
        current = dict(entries.get(key) or {}) if isinstance(entries.get(key), dict) else {}
        current["peak_bytes"] = max(int(current.get("peak_bytes") or 0), peak)
        current["rss_bytes"] = max(int(current.get("rss_bytes") or 0), rss)
        current["renders"] = int(current.get("renders") or 0) + 1
        current["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entries[key] = current
        try:
            _write_json(paths.breeze_calibration_path(), {"schema": SCHEMA, "models": entries})
        except OSError:
            logger.debug("Model Chain: could not record what a Breeze render cost",
                         exc_info=True)


def need_vram_bytes(identifier: str = "", takes: int = 1, fast_path=None) -> int:
    """What to ask the turn for: the estimate, or the peak a render has reached.

    The weights (the installed files' sizes, else the listing's, else the
    estimate) and the working set, with the fast path's graphs on top while it
    is on. Takes do not add: Breeze renders them one after another.
    """
    entry = bundle(identifier)
    fast = bool(settings().get("fast_path")) if fast_path is None else bool(fast_path)
    weights = _weights_bytes(entry)
    estimate = weights + int(entry.estimates.get("working_bytes") or WORKING_BYTES_DEFAULT)
    if fast:
        estimate += int(entry.estimates.get("fast_bytes") or FAST_BYTES_DEFAULT)
    observed = int(calibration().get(_calibration_key(entry.identifier, fast), {})
                   .get("peak_bytes") or 0)
    return max(estimate, observed)


def need_ram_bytes(identifier: str = "") -> int:
    entry = bundle(identifier)
    estimate = int(entry.estimates.get("ram_bytes") or RAM_BYTES_DEFAULT)
    measured = calibration()
    observed = max(int(measured.get(key, {}).get("rss_bytes") or 0)
                   for key in (entry.identifier, f"{entry.identifier}+fast"))
    return max(estimate, observed)


def _weights_bytes(entry: Bundle) -> int:
    """The weights' total: installed sizes, then a listing's, then the estimate."""
    marker = _read_json(paths.breeze_model_root(entry.identifier)
                        / paths.INSTALLED_FILENAME) or {}
    recorded = marker.get("bytes") if isinstance(marker.get("bytes"), dict) else {}
    total = sum(int(size) for name, size in recorded.items()
                if str(name).endswith(".safetensors") and isinstance(size, int))
    if total:
        return total
    if entry.weights_bytes:
        return entry.weights_bytes
    listed = _recorded_listing(entry)
    if listed:
        total = sum(item.bytes for item in listed[1] if item.weight)
        if total:
            return total
    return entry.estimated_weights_bytes


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def _nvidia_present() -> bool:
    """Whether this machine has an NVIDIA driver at all. Cheap, and never CUDA."""
    if shutil.which("nvidia-smi"):
        return True
    if os.name == "nt":
        root = Path(os.environ.get("SystemRoot") or r"C:\Windows")
        return (root / "System32" / "nvcuda.dll").is_file()
    return Path("/proc/driver/nvidia/version").exists()


def _install_refusal(manual: bool = False) -> str:
    """Why Breeze cannot be *installed* right now, or an empty string."""
    try:
        chosen = platform()
    except BreezeError as exc:
        return str(exc)
    if chosen is None:
        return _unsupported_sentence(", so it cannot be installed here.")
    with models._lock:
        if (models._progress.get(KIND) or {}).get("running"):
            return "Breeze is still being installed."
    if not manual and not pinned():
        return ("This build has not pinned a Breeze runtime yet, so it will not be "
                "downloaded. A maintainer runs tools/pin_breeze_models.py on a machine that "
                "can reach pypi.org; you can install from a folder you filled yourself in "
                "the meantime.")
    return ""


def refusal(manual: bool = False) -> str:
    """Why a render cannot run now, in one sentence the page can show, or ``""``."""
    found = _install_refusal(manual)
    if found:
        return found
    try:
        current = _status()
    except BreezeError as exc:
        return str(exc)
    if not _nvidia_present():
        return ("Breeze TTS 2 needs an NVIDIA card, and this machine has no NVIDIA driver "
                "(nvidia-smi was not found).")
    if current.stale:
        return current.stale_message
    if not current.runtime_installed:
        return "Breeze's runtime is not installed — install it below."
    if not current.model_installed:
        return f"The {current.model_label or LABEL} model is not installed — install it below."
    return ""


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #


PARTS = ("runtime", "model")


def sources(part: str = "runtime") -> list:
    """Where a person would go to fetch Breeze's files by hand."""
    if part == "runtime":
        chosen = platform()
        found = [{"filename": item.filename, "url": item.url, "save_as": item.local_name,
                  "archive": False} for item in (chosen.artifacts if chosen else ())]
        for one in resolves():
            name = one.get("filename") or (f"{one['package']}-{full_version(one)}-cp313-"
                                           f"cp313-win_amd64.whl")
            found.append({"filename": name, "url": one["index"], "save_as": name,
                          "archive": False})
        flash = flash_artifact()
        if flash is not None:
            found.append({"filename": flash.filename, "url": flash.url,
                          "save_as": flash.local_name, "archive": False})
        code = source()
        found += [{"filename": row["path"], "url": row["url"],
                   "save_as": f"{paths.BREEZE_SOURCE_DIRNAME}/{row['path']}", "archive": False}
                  for row in code["files"]]
        return found
    entry = bundle()
    listed = _recorded_listing(entry) or ((entry.commit, entry.files) if entry.pinned
                                          else None)
    if listed:
        commit, files = listed
        return [{"filename": item.path, "url": _hub_url(entry.repo, commit, item.path),
                 "save_as": item.path, "archive": False} for item in files]
    return [{"filename": f"{entry.repo} (every file)",
             "url": f"https://huggingface.co/{entry.repo}/tree/{entry.revision}",
             "save_as": "", "archive": False}]


def install(part: str = "", on_status=None, on_progress=None) -> Status:
    """Install what is missing: the runtime, the model, or both.

    A transaction per part. Nothing outside a staging directory is touched until
    every byte has arrived and matched what was expected, each part is promoted
    by a directory rename, and a failure leaves the installation as it was.
    """
    wanted = str(part or "").strip().lower()
    if wanted and wanted not in PARTS:
        raise BreezeError("Breeze installs its runtime or its model.")
    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)
    with models._claim(KIND, say, bundle().identifier):
        if wanted in ("", "runtime"):
            say("Checking the Breeze runtime…")
            share = 0.3 if not wanted else 1.0
            install_runtime(on_status=say, on_progress=lambda f: tick(f * share))
        if wanted in ("", "model"):
            base = 0.3 if not wanted else 0.0
            install_model(on_status=say,
                          on_progress=lambda f: tick(base + f * (1.0 - base)))
        tick(1.0)
        say("Breeze installed.")
        return status()


def install_from(part: str, folder: str, on_status=None, on_progress=None) -> Status:
    """Install from files already on this machine. The escape hatch.

    A folder of the runtime's wheels (the pinned ones, torch, torchaudio and the
    flash-attn wheel) with Breeze's code beside them -- a clone, or the
    repository's files at their paths -- or a folder holding the model's files
    as the hub lists them. A pinned file is checked against the hash committed
    here; one this build could not hash has its digest recorded.
    """
    wanted = str(part or "").strip().lower()
    if wanted not in PARTS:
        raise BreezeError("Breeze installs its runtime or its model.")
    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)
    with models._claim(KIND, say, bundle().identifier):
        if wanted == "runtime":
            install_runtime(on_status=say, on_progress=tick, folder=folder)
        else:
            install_model(on_status=say, on_progress=tick, folder=folder)
        return status()


def _folder(folder) -> Path:
    source_folder = Path(str(folder or "").strip().strip('"')).expanduser()
    if not str(folder or "").strip():
        raise BreezeError("Give the folder the downloaded files are in.")
    if source_folder.is_file():
        source_folder = source_folder.parent
    if not source_folder.is_dir():
        raise BreezeError(f"{source_folder} is not a folder this machine can read.")
    return source_folder


def install_runtime(on_status=None, on_progress=None, folder=None) -> None:
    """The isolated CUDA closure, Flash attention 2 and Breeze's code, in one venv.

    Built the way every voice runtime is -- a virtual environment without pip,
    the verified wheels unpacked rather than installed by a package manager --
    with three wheels more than the PyPI closure (torch, torchaudio and the
    flash-attn wheel) and Breeze's own files, each checked against its pinned
    digest, under ``breeze-tts`` beside the environment.
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    chosen = platform()
    if chosen is None:
        raise BreezeError(_install_refusal() or "Breeze has no runtime for this platform.")
    if not chosen.artifacts:
        raise BreezeError(_install_refusal(manual=bool(folder))
                          or "This build has not recorded a Breeze runtime closure.")
    picked = resolves()
    if {one["package"] for one in picked} != {"torch", "torchaudio"}:
        raise BreezeError("The Breeze manifest must resolve torch and torchaudio for this "
                          "platform.")
    flash = flash_artifact()
    if flash is None:
        raise BreezeError("The Breeze manifest names no Flash attention 2 wheel for this "
                          "platform.")
    code = source()
    if not code["commit"] or not code["files"]:
        raise BreezeError("The Breeze manifest has not pinned Breeze's own code.")
    installed = _read_json(paths.breeze_runtime_manifest())
    if installed and str(installed.get("closure") or "") == closure_id() \
            and runtime_python() is not None:
        say("The Breeze runtime is already installed.")
        tick(1.0)
        return

    _stop_runtime("the Breeze runtime is being installed")
    staging = paths.breeze_staging_for("runtime", uuid.uuid4().hex[:8])
    wheels = staging / "wheels"
    shutil.rmtree(staging, ignore_errors=True)
    wheels.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            origin = _folder(folder)
            _adopt(chosen.artifacts, origin, wheels, say, "Breeze runtime wheel")
            resolved = [_adopt_resolved(one, origin, wheels, say) for one in picked]
            flash_item, flash_digest = _adopt_flash(flash, origin, wheels, say)
            tick(0.5)
        else:
            resolved = []
            for one in picked:
                say(f"Asking the publisher which {one['package']} wheel this machine needs…")
                resolved.append(_resolve_wheel(one, chosen, say))
            flash_item = flash
            everything = tuple(chosen.artifacts) + tuple(resolved) + (flash_item,)
            expectations = models._expectations(tuple(chosen.artifacts) + (flash_item,), say)
            for item in resolved:
                expectations[item.local_name] = models.Expected(
                    item.size, item.sha256, "the publisher's package index, over HTTPS")
            room = dict(expectations)
            for item, one in zip(resolved, picked):
                if not item.size:
                    room[item.local_name] = models.Expected(_runtime_bytes(None, [one]),
                                                            item.sha256, "an estimate")
            models._make_room(everything, staging, room)
            digests = models._fetch_all(everything, wheels, say, tick, 0.5, expectations)
            flash_digest = digests.get(flash_item.filename) or _digest(
                wheels / flash_item.local_name)
        say("Fetching Breeze's own code…" if not folder else "Checking Breeze's own code…")
        source_root = staging / paths.BREEZE_SOURCE_DIRNAME
        _install_source(code, source_root, say, origin=_folder(folder) if folder else None)
        tick(0.6)
        say("Building the isolated Breeze runtime…")
        _build_environment(staging, wheels, tuple(chosen.artifacts) + tuple(resolved)
                           + (flash_item,))
        tick(0.75)
        say("Checking that Breeze imports on this machine…")
        report = _smoke_test(staging, source_root)
        tick(0.95)
        wheel_records = {}
        for item in resolved:
            path = wheels / item.local_name
            wheel_records[item.local_name] = {
                "filename": item.filename, "sha256": _digest(path) if path.exists()
                else str(item.sha256 or ""),
                "bytes": path.stat().st_size if path.exists() else int(item.size or 0)}
        flash_path = wheels / flash_item.local_name
        flash_bytes = flash_path.stat().st_size if flash_path.exists() else 0
        shutil.rmtree(wheels, ignore_errors=True)
        runtime = manifest()["runtime"]
        loads = bool(report.get("flash_attn_loads"))
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "closure": closure_id(),
            "platform": chosen.identifier,
            "python": chosen.python,
            "accelerator": "cuda",
            "breeze_commit": str(report.get("breeze") or code["commit"]),
            "torch_version": str(report.get("torch") or ""),
            "torchaudio_version": str(report.get("torchaudio") or ""),
            "transformers_version": str(report.get("transformers")
                                        or runtime.get("transformers_version") or ""),
            "qwen_tts_version": str(report.get("qwen_tts") or ""),
            "cuda": report.get("cuda"),
            "compile": bool(report.get("compile")),
            "flash_attn": {"version": str(report.get("flash_attn") or ""),
                           "filename": flash_item.filename, "sha256": flash_digest,
                           "bytes": flash_bytes, "loads": loads,
                           "error": _without_paths(report.get("flash_attn_error") or "")},
            "resolved": wheel_records,
            "artifacts": {item.local_name: item.sha256 for item in chosen.artifacts},
            "license": runtime.get("license") or "",
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        models._promote(staging, paths.breeze_runtime_root())
        _record_flash(flash_item, flash_digest, flash_bytes)
        if not loads:
            logger.warning("Model Chain: Breeze's Flash attention 2 wheel is installed but "
                           "would not load — %s; SDPA and Eager still render",
                           _without_paths(report.get("flash_attn_error") or "no reason"))
        say("The Breeze runtime is installed.")
        tick(1.0)
        logger.info("Model Chain: the Breeze runtime is installed — %s, torch %s, Breeze %s, "
                    "flash-attn %s (%s)", chosen.identifier, report.get("torch") or "?",
                    str(code["commit"])[:12], report.get("flash_attn") or "?",
                    "loads" if loads else "does not load")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _resolve_wheel(one: dict, chosen, say) -> models.Artifact:
    """One wheel from the publisher's own index: exactly the pinned version for cp313 Windows.

    A digest the manifest recorded (``--torch``) must agree with the index's; a
    publisher that changed a wheel under the same version is refused.
    """
    version = full_version(one)
    machine = (list(chosen.machines) or ["amd64"])[0]
    tags = wheelindex.platform_tags(chosen.python, chosen.system, machine)
    try:
        page = models.read_index_page(one["index"])
    except models.VoiceError:
        raise
    except Exception as exc:
        raise BreezeError(f"The publisher's index for {one['package']} could not be read "
                          f"({exc.__class__.__name__}). Nothing was installed.") from None
    try:
        found = wheelindex.choose(page, one["index"], one["package"], version, tags)
    except wheelindex.IndexError_ as exc:
        raise BreezeError(str(exc)) from None
    recorded = str(one.get("sha256") or "")
    if recorded and recorded != found["sha256"]:
        raise BreezeError(
            f"The publisher now serves a different {found['filename']} than this build "
            f"recorded — its digest has changed under the same version. Nothing was "
            f"downloaded; a maintainer reviews the change with tools/pin_breeze_models.py "
            f"--torch.")
    size = int(one.get("bytes") or 0) or models._declared_size(found["url"])
    say(f"{one['package']} {version} is {found['filename']}"
        f"{' (' + models._bytes_label(size) + ')' if size else ''}.")
    logger.info("Model Chain: Breeze resolved %s from the publisher's index — %s, sha256 %s",
                one["package"], found["filename"], found["sha256"][:12])
    return models.Artifact(filename=found["filename"], local_name=found["filename"],
                           url=found["url"], size=size or None, sha256=found["sha256"])


def _adopt_resolved(one: dict, origin: Path, wheels: Path, say) -> models.Artifact:
    """torch or torchaudio out of a folder somebody filled themselves."""
    version = full_version(one)
    wanted = [one.get("filename") or "",
              f"{one['package']}-{version}-cp313-cp313-win_amd64.whl"]
    found = models._find(origin, [name for name in wanted if name])
    if found is None:
        raise BreezeError(f"{wanted[-1]} is not in {origin}. It comes from {one['index']}. "
                          f"Nothing was installed.")
    say(f"Checking {found.name}…")
    digest = _digest(found)
    recorded = str(one.get("sha256") or "")
    if recorded and digest != recorded:
        raise BreezeError(f"{found.name} is in that folder, but its contents are not what "
                          f"this extension's manifest recorded. Nothing was installed.")
    shutil.copyfile(found, wheels / found.name)
    return models.Artifact(filename=found.name, local_name=found.name, url=one["index"],
                           size=found.stat().st_size, sha256=digest)


def _adopt_flash(flash: models.Artifact, origin: Path, wheels: Path, say) -> tuple:
    """The flash-attn wheel out of a folder: ``(artifact, digest)``."""
    digests = _adopt([flash], origin, wheels, say, "Flash attention 2 wheel")
    return flash, digests[flash.local_name]


def _record_flash(flash: models.Artifact, digest: str, size: int) -> None:
    """The flash-attn wheel's digest into the overlay. Never fatal, never over a pin."""
    if flash.sha256 and flash.size:
        return
    if len(str(digest or "")) != 64 or int(size or 0) <= 0:
        return
    _overlay_write(lambda found: found.setdefault("artifacts", {}).__setitem__(
        _pin_key(_repo_of(flash.url), flash.filename), {"sha256": digest, "bytes": int(size)}))


def _install_source(code: dict, root: Path, say, origin=None) -> None:
    """Breeze's own files at the pinned commit into ``root``, each checked.

    From a folder somebody filled when it has them (a clone, or an unpacked
    archive of the commit), else from the repository's raw host. A file whose
    bytes are not the pinned bytes is refused either way; an empty file (a
    package's ``__init__.py``) is written rather than fetched. ``COMMIT`` names
    the commit for the worker's handshake.
    """
    rows = list(code["files"])
    base = _source_in(origin, rows) if origin is not None else None
    for row in rows:
        target = root / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if row["bytes"] == 0:
            if row["sha256"] and row["sha256"] != hashlib.sha256(b"").hexdigest():
                raise BreezeError(f"The Breeze manifest pins {row['path']} as empty with a "
                                  f"digest of something else.")
            target.write_bytes(b"")
            continue
        if base is not None and (base / row["path"]).is_file():
            found = base / row["path"]
            digest = _digest(found)
            if digest != row["sha256"]:
                raise BreezeError(f"{row['path']} in {base} is not the file this build pins at "
                                  f"{code['commit'][:12]}. Nothing was installed.")
            shutil.copyfile(found, target)
            continue
        item = models.Artifact(filename=row["path"].rsplit("/", 1)[-1],
                               local_name=row["path"].rsplit("/", 1)[-1], url=row["url"],
                               size=row["bytes"], sha256=row["sha256"])
        models._download(item, target, lambda _received: None,
                         models.Expected(row["bytes"], row["sha256"],
                                         "this extension's manifest"))
    (root / protocol.COMMIT_FILENAME).write_text(code["commit"] + "\n", encoding="utf-8")
    say(f"Breeze's code at {code['commit'][:12]}: {len(rows)} files checked.")


def _source_in(origin: Path, rows: list) -> "Path | None":
    """Where Breeze's files are in a folder somebody filled, or ``None``."""
    probe = next((row["path"] for row in rows if row["path"].startswith("breeze_infer/")),
                 "breeze_infer/runtime.py")
    candidates = [origin, origin / paths.BREEZE_SOURCE_DIRNAME]
    candidates += sorted(child for child in origin.iterdir()
                         if child.is_dir() and child.name.casefold().startswith("breeze"))
    for folder in candidates:
        if (folder / probe).is_file():
            return folder
    return None


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


def _recorded_listing(entry: Bundle) -> "tuple | None":
    """``(commit, files)`` an earlier install of ``entry`` recorded, or ``None``."""
    recorded = ((_overlay().get("models") or {}) or {}).get(entry.identifier)
    if not isinstance(recorded, dict) or str(recorded.get("repo") or "") != entry.repo:
        return None
    commit = str(recorded.get("commit") or "").casefold()
    if len(commit) != 40 or not set(commit) <= _HEX:
        return None
    try:
        files = tuple(_model_file(row, "local record") for row in recorded.get("files") or ())
    except BreezeError:
        return None
    return (commit, files) if files else None


def _hub_url(repo: str, revision: str, path: str) -> str:
    return HUB_RESOLVE.format(repo=repo, revision=urllib.parse.quote(revision),
                              path=urllib.parse.quote(path))


def _listing(entry: Bundle, say) -> tuple:
    """``(commit, files, source)``: what the model is, and who said so.

    The manifest's own listing when a maintainer committed one; else the commit
    and files the first install on this machine recorded, so a reinstall is the
    same model; else the hub's tree at the manifest's revision, read now.
    """
    if entry.pinned:
        return entry.commit, entry.files, "this extension's manifest"
    recorded = _recorded_listing(entry)
    if recorded:
        return recorded[0], recorded[1], "the listing recorded by your first install"
    say(f"Asking the hub which files {entry.label} is…")
    return (*_hub_listing(entry), "the hub's file list, over HTTPS")


def _hub_listing(entry: Bundle) -> tuple:
    """The hub's own tree at the manifest's revision, as ``(commit, files)``."""
    name = f"{entry.label}'s file list"
    document, _ = models.read_json(HUB_REVISION.format(
        repo=entry.repo, revision=urllib.parse.quote(entry.revision, safe="")),
        authorized=entry.authorized, name=name)
    commit = str((document or {}).get("sha") or "").casefold() \
        if isinstance(document, dict) else ""
    if len(commit) != 40 or not set(commit) <= _HEX:
        raise BreezeError(f"The hub did not say which commit {entry.label}'s "
                          f"{entry.revision} is. Nothing was installed.")
    files = []
    url = HUB_TREE.format(repo=entry.repo, revision=commit)
    for _page in range(50):
        document, url = models.read_json(url, authorized=entry.authorized, name=name)
        if not isinstance(document, list):
            raise BreezeError(f"The hub's file list for {entry.label} is not a list. Nothing "
                              f"was installed.")
        for item in document:
            if not isinstance(item, dict) or item.get("type") != "file":
                continue
            path = str(item.get("path") or "")
            if path in entry.skip or path.casefold().endswith(entry.skip_suffixes):
                continue
            lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else None
            row = {"path": path, "bytes": int((lfs or item).get("size") or 0)}
            if lfs:
                row["sha256"] = str(lfs.get("oid") or lfs.get("sha256") or "")
            else:
                row["git_oid"] = str(item.get("oid") or "")
            files.append(_model_file(row, "hub listing"))
        if not url:
            break
    names = {item.path for item in files}
    if "config.json" not in names:
        raise BreezeError(f"The hub lists no config.json for {entry.label} at "
                          f"{commit[:12]}. Nothing was installed.")
    if not any(name.startswith("audio_tokenizer/") for name in names):
        raise BreezeError(f"The hub lists no audio_tokenizer folder for {entry.label} at "
                          f"{commit[:12]}, which Breeze's loader cannot start without. Nothing "
                          f"was installed.")
    if not any(item.weight for item in files):
        raise BreezeError(f"The hub lists no weights for {entry.label} at {commit[:12]}. "
                          f"Nothing was installed.")
    return commit, tuple(sorted(files, key=lambda item: item.path))


def install_model(on_status=None, on_progress=None, folder=None) -> None:
    """Every file of the model, at one commit, into one staging tree and one promote."""
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    entry = bundle()
    target = paths.breeze_model_root(entry.identifier)
    marker = _read_json(target / paths.INSTALLED_FILENAME)
    if marker and not _model_missing(target, marker):
        say(f"{entry.label} is already installed.")
        tick(1.0)
        return

    _stop_runtime("the Breeze model is being installed")
    staging = paths.breeze_staging_for(entry.identifier, uuid.uuid4().hex[:8])
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            record = _adopt_model(entry, _folder(folder), staging, say, tick)
        else:
            record = _download_model(entry, staging, say, tick)
        say(f"Checking {entry.label} is complete…")
        _sanity_check(staging, list(record["sizes"]), entry.label)
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "id": entry.identifier,
            "label": entry.label,
            "repo": entry.repo,
            "revision": entry.revision,
            "commit": record["commit"],
            "license": entry.license,
            "attribution": entry.attribution,
            "source": record["source"],
            "source_folder": record.get("source_folder", ""),
            "digests": dict(sorted(record["digests"].items())),
            "bytes": dict(sorted(record["sizes"].items())),
            "verified_by": dict(sorted(record["verified"].items())),
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        say("Installing…")
        models._promote(staging, target)
        _record_listing(entry, record)
        say(f"{entry.label} is installed.")
        tick(1.0)
        logger.info("Model Chain: the Breeze model %s is installed — %d file(s) at %s from %s",
                    entry.identifier, len(record["sizes"]), record["commit"][:12],
                    record["source"])
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _new_record(source_name: str, commit: str) -> dict:
    return {"digests": {}, "sizes": {}, "verified": {}, "git_oids": {}, "commit": commit,
            "source": source_name, "files": []}


def _download_model(entry: Bundle, staging: Path, say, tick) -> dict:
    """Every listed file from the hub at the listing's commit, each checked as it lands."""
    commit, files, said_by = _listing(entry, say)
    record = _new_record("hub", commit)
    record["files"] = list(files)
    artifacts = []
    for item in files:
        artifacts.append((item, models.Artifact(
            filename=item.path, local_name=item.path,
            url=_hub_url(entry.repo, commit, item.path), size=item.bytes or None,
            sha256=item.sha256 or None, authorized=entry.authorized)))
    expectations = {artifact.local_name: models.Expected(item.bytes or None,
                                                         item.sha256 or None, said_by)
                    for item, artifact in artifacts}
    models._make_room([artifact for _item, artifact in artifacts], staging, expectations)
    total = max(sum(item.bytes for item in files), 1)
    done = 0
    for index, (item, artifact) in enumerate(artifacts, start=1):
        say(f"Downloading {index} of {len(artifacts)} — {item.path} "
            f"({models._bytes_label(item.bytes) if item.bytes else 'size unknown'})")
        offset = done

        def report(received, offset=offset, size=item.bytes):
            tick(0.97 * min((offset + min(received, max(size, 1))) / total, 1.0))

        destination = staging / item.path
        started = time.monotonic()
        digest = models._download(artifact, destination, report,
                                  expectations[artifact.local_name])
        if item.git_oid and _git_oid(destination) != item.git_oid:
            destination.unlink(missing_ok=True)
            raise BreezeError(f"{item.path} downloaded, but it is not the file {said_by} "
                              f"names. Nothing was installed.")
        elapsed = max(time.monotonic() - started, 0.001)
        arrived = destination.stat().st_size
        logger.info("Model Chain: Breeze fetched %s — %s in %.1fs (%.1f MB/s); %s",
                    item.path, models._bytes_label(arrived), elapsed,
                    arrived / elapsed / (1024 * 1024),
                    f"checked against {said_by}")
        record["digests"][item.path] = digest
        record["sizes"][item.path] = arrived
        record["verified"][item.path] = said_by
        if item.git_oid:
            record["git_oids"][item.path] = item.git_oid
        done += item.bytes
        tick(0.97 * min(done / total, 1.0))
    return record


def _adopt_model(entry: Bundle, origin: Path, staging: Path, say, tick) -> dict:
    """The model out of a folder somebody filled themselves.

    Checked against a listing when one can be had -- the manifest's, the one
    this machine recorded, or the hub's -- and taken as it is, with every file's
    digest recorded, when none can (an offline machine): every file under the
    folder but the ones the manifest skips, with config.json, the
    audio_tokenizer folder and a weight required.
    """
    try:
        commit, files, said_by = _listing(entry, say)
    except (BreezeError, models.VoiceError) as exc:
        logger.info("Model Chain: Breeze's model listing could not be had (%s); the folder "
                    "is taken as it is", exc)
        commit, files, said_by = "", (), ""
    root = _model_in(origin)
    record = _new_record("local", commit)
    record["source_folder"] = str(origin)
    if files:
        wanted = list(files)
    else:
        wanted = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in entry.skip or relative.casefold().endswith(entry.skip_suffixes) \
                    or relative.startswith(".") or "/." in relative:
                continue
            wanted.append(ModelFile(path=relative, bytes=path.stat().st_size))
        names = {item.path for item in wanted}
        if "config.json" not in names or not any(name.startswith("audio_tokenizer/")
                                                 for name in names) \
                or not any(item.weight for item in wanted):
            raise BreezeError(f"{origin} does not hold {entry.label}: it needs config.json, "
                              f"the audio_tokenizer folder and the weights. Nothing was "
                              f"installed.")
    for index, item in enumerate(wanted, start=1):
        found = root / item.path
        if not found.is_file():
            raise BreezeError(f"{item.path} is not in {root}. Nothing was installed.")
        say(f"Checking {index} of {len(wanted)} — {item.path}…")
        digest = _digest(found)
        size = found.stat().st_size
        if item.bytes and size != item.bytes:
            raise BreezeError(f"{item.path} is {size} bytes and {said_by} says {item.bytes}. "
                              f"Nothing was installed.")
        if item.sha256 and digest != item.sha256:
            raise BreezeError(f"{item.path} is in that folder, but its contents are not what "
                              f"{said_by} says they should be. Nothing was installed.")
        if item.git_oid and _git_oid(found) != item.git_oid:
            raise BreezeError(f"{item.path} is in that folder, but it is not the file "
                              f"{said_by} names. Nothing was installed.")
        target = staging / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found, target)
        record["digests"][item.path] = digest
        record["sizes"][item.path] = size
        record["verified"][item.path] = said_by or "your own files"
        tick(0.95 * index / max(len(wanted), 1))
    record["files"] = list(files) if files else [
        ModelFile(path=name, bytes=record["sizes"][name], sha256=record["digests"][name])
        for name in sorted(record["sizes"])]
    return record


def _model_in(origin: Path) -> Path:
    """The folder that holds the model's config.json: ``origin`` or one level down."""
    if (origin / "config.json").is_file():
        return origin
    for child in sorted(origin.iterdir()):
        if child.is_dir() and (child / "config.json").is_file():
            return child
    return origin


def _git_oid(path: Path) -> str:
    """A file's git blob id: SHA-1 over ``blob <size>\\0`` and its bytes."""
    found = hashlib.sha1(usedforsecurity=False)
    found.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 256), b""):
            found.update(block)
    return found.hexdigest()


def _record_listing(entry: Bundle, record: dict) -> None:
    """The commit and files an install used, into the overlay. Never fatal.

    Only from the hub, and only when the manifest itself does not commit one:
    the next install of this model on this machine lists this same commit.
    """
    if entry.pinned or record.get("source") != "hub" or not record.get("commit"):
        return
    rows = []
    for item in record.get("files") or ():
        row = {"path": item.path, "bytes": int(record["sizes"].get(item.path) or item.bytes)}
        if item.sha256:
            row["sha256"] = item.sha256
        else:
            row["git_oid"] = item.git_oid
            row["sha256"] = record["digests"].get(item.path, "")
        rows.append(row)

    def write(found: dict) -> None:
        found.setdefault("models", {})[entry.identifier] = {
            "repo": entry.repo, "commit": record["commit"], "files": rows,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    _overlay_write(write)


def _overlay_write(change) -> None:
    path = paths.breeze_local_pins_path()
    with _lock:
        found = _overlay()
        found = dict(found) if isinstance(found, dict) else {}
        found["schema"] = 1
        try:
            change(found)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
        except OSError:
            logger.debug("Model Chain: Breeze could not record what it installed to %s",
                         path, exc_info=True)


def _adopt(artifacts, folder: Path, destination: Path, say, what: str) -> dict:
    """Copy named files out of a folder, checking each against what is known."""
    if not folder.is_dir():
        raise BreezeError(f"{folder} is not a folder this machine can read.")
    destination.mkdir(parents=True, exist_ok=True)
    digests = {}
    for item in artifacts:
        found = models._find(folder, [item.filename, item.local_name])
        if found is None:
            raise BreezeError(f"{item.filename} is not in {folder}. Nothing was installed.")
        say(f"Checking {found.name}…")
        digest = _digest(found)
        if item.sha256 and digest != item.sha256:
            raise BreezeError(f"{item.filename} is in that folder, but its contents are not "
                              f"what this extension's manifest says they should be. Nothing "
                              f"was installed.")
        if item.size and found.stat().st_size != item.size:
            raise BreezeError(f"{item.filename} is {found.stat().st_size} bytes and this "
                              f"extension expects {item.size}. Nothing was installed.")
        shutil.copyfile(found, destination / item.local_name)
        digests[item.local_name] = digest
    logger.info("Model Chain: Breeze adopted %d file(s) for the %s from %s",
                len(digests), what, folder)
    return digests


SAFETENSORS_MIN = 16


def _sanity_check(staging: Path, required, label: str) -> None:
    """Structural validation before anything is promoted: the bytes are the kind
    of thing they claim to be, because a proxy that answers every request with
    an HTML page produces a file of exactly the right name."""
    names = list(required)
    if "config.json" not in names:
        raise BreezeError(f"{label} has no config.json. Nothing was installed.")
    if not any(name.startswith("audio_tokenizer/") for name in names):
        raise BreezeError(f"{label} has no audio_tokenizer folder. Nothing was installed.")
    for name in names:
        path = staging / name
        try:
            size = path.stat().st_size
        except OSError:
            raise BreezeError(f"{label} is missing {name}. Nothing was installed.") from None
        if name.endswith(".safetensors"):
            with open(path, "rb") as handle:
                head = handle.read(8)
            if len(head) < 8:
                raise BreezeError(f"{name} is too small to be a safetensors file. Nothing was "
                                  f"installed.")
            (declared,) = struct.unpack("<Q", head)
            if declared <= 0 or declared + 8 > size or declared > 64 * 1024 * 1024:
                raise BreezeError(f"{name} does not have a safetensors header. Nothing was "
                                  f"installed.")
        elif name.endswith(".json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise BreezeError(f"{name} is not readable JSON. Nothing was installed.") \
                    from None


def _build_environment(staging: Path, wheels: Path, artifacts) -> None:
    """An interpreter of Breeze's own, with the verified wheels unpacked into it."""
    import venv

    environment = staging / "env"
    try:
        builder = venv.EnvBuilder(with_pip=False, clear=True, symlinks=(os.name != "nt"))
        builder.create(environment)
    except Exception as exc:
        logger.warning("Model Chain: the isolated Breeze runtime could not be created using "
                       "%s — %s: %s", environment, exc.__class__.__name__, exc)
        raise BreezeError(f"The isolated Breeze runtime could not be created "
                          f"({exc.__class__.__name__}: {exc}).") from None
    if not _interpreter_in(staging).exists():
        raise BreezeError("The isolated Breeze runtime was created without an interpreter. "
                          "Nothing was installed.")
    target = models.site_packages(environment)
    for item in artifacts:
        wheel = wheels / item.local_name
        if not wheel.is_file():
            raise BreezeError(f"{item.filename} is missing from the staged download. Nothing "
                              f"was installed.")
        added = models._unpack_wheel(wheel, target)
        logger.info("Model Chain: Breeze unpacked %s (%s)", item.filename,
                    ", ".join(added[:6]) or "nothing")
    for name in ("torch", "torchaudio", "transformers", "qwen_tts", "flash_attn"):
        if not (target / name).exists():
            raise BreezeError(f"The staged Breeze runtime has no {name} in it. Nothing was "
                              f"installed.")


def _smoke_test(staging: Path, source_root: Path) -> dict:
    """Prove the staged runtime imports Breeze *before* anything is promoted.

    The worker's own ``--selftest`` against the staged interpreter, with the
    staged code on its path: torch, transformers, qwen_tts and Breeze's entry
    points imported, Flash attention 2 imported and reported. No model is
    loaded and no card is required; whether the worker can prove its card is
    the runtime's handshake, on the first render.
    """
    interpreter = _interpreter_in(staging)
    script = worker_script()
    if not script.is_file():
        raise BreezeError("The Breeze worker script is missing from this extension, so the "
                          "runtime cannot be checked. Nothing was installed.")
    alive = _run_staged(interpreter, ["-c", "import sys; print(sys.version)"],
                        "check that the isolated Breeze interpreter runs", source_root,
                        timeout=180)
    if alive.returncode != 0:
        raise BreezeError(
            "The isolated Breeze interpreter would not start. This usually means the Python "
            "running this WebUI is an embedded or relocated build that cannot make a virtual "
            "environment. Nothing was installed.")
    result = _run_staged(interpreter, [script, "--selftest"], "Breeze self-test",
                         source_root, timeout=900)
    report = {}
    for line in reversed((result.stdout or "").strip().splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            report = candidate
            break
    if result.returncode != 0 or report.get("ok") is False or not report:
        reason = report.get("error") or (result.stderr or "").strip().splitlines()[-1:] or [""]
        reason = reason if isinstance(reason, str) else (reason[0] if reason else "")
        logger.warning("Model Chain: the staged Breeze runtime could not import — %s",
                       _without_paths(reason) or "no reason reported")
        raise BreezeError(f"The staged Breeze runtime could not import its packages "
                          f"({_without_paths(reason) or 'no reason reported'}). Nothing was "
                          f"installed.")
    logger.info("Model Chain: the staged Breeze runtime passed its self-test — %s",
                models._quote(_without_paths(json.dumps(report)), 400))
    return report


_PATHLIKE = re.compile(
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]|(?<![:\w\\/])[\\/])"
    r"[^\s'\"()<>]*[\\/]([^\s'\"()<>\\/]+)")


def _without_paths(text) -> str:
    """A staged-runtime message with its filesystem paths cut down to filenames."""
    return _PATHLIKE.sub(lambda found: found.group(1), str(text or "")).strip()


def _run_staged(interpreter: Path, arguments: list, what: str, source_root: Path,
                timeout: float = 300):
    """Run something in the staged runtime, with no credential in its environment."""
    import subprocess

    environ = dict(os.environ)
    for name in models.CREDENTIAL_VARIABLES:
        environ.pop(name, None)
    environ.update(worker_environment("", source_root=source_root))
    try:
        return subprocess.run(  # noqa: S603 - a path this module built
            [str(interpreter)] + [str(item) for item in arguments],
            capture_output=True, text=True, env=environ, timeout=timeout,
            cwd=str(paths.extension_root()))
    except Exception as exc:
        logger.warning("Model Chain: could not %s — %s: %s", what, exc.__class__.__name__, exc)
        raise BreezeError(f"The staged Breeze runtime could not be started "
                          f"({exc.__class__.__name__}). Nothing was installed.") from None


def uninstall() -> Status:
    """Remove the Breeze runtime and model. Settings, the calibration and the overlay stay.

    Kept because they describe this machine rather than an installation: a
    reinstalled model renders with the same card, needs the same VRAM, and is
    the same commit.
    """
    _stop_runtime("Breeze is being removed")
    for root in (paths.breeze_runtime_root(), paths.breeze_models_root(),
                 paths.breeze_staging_root()):
        if paths.breeze_inside(root):
            shutil.rmtree(root, ignore_errors=True)
    logger.info("Model Chain: Breeze's runtime and model were removed; its settings and "
                "calibration were kept")
    return status()


def _stop_runtime(reason: str) -> None:
    """Stop every Breeze worker before its runtime or model is replaced. Never raises."""
    try:
        import mc_voice_breeze_runtime as runtime
    except Exception:
        return
    try:
        runtime.stop("", reason)
    except TypeError:
        try:
            runtime.stop(reason=reason)
        except Exception:
            logger.debug("Model Chain: could not stop Breeze before %s", reason, exc_info=True)
    except Exception:
        logger.debug("Model Chain: could not stop Breeze before %s", reason, exc_info=True)


def progress() -> dict:
    """What the install is doing, for the page. One flat record for this engine."""
    found = models.progress().get(KIND) or {}
    return {
        "running": bool(found.get("running")),
        "text": str(found.get("text") or ""),
        "fraction": float(found.get("fraction") or 0.0),
        "failed": bool(found.get("failed")),
        "model": str(found.get("model") or ""),
    }


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
    """Write a JSON document atomically: beside the real file, then renamed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".new")
    staging.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(staging, path)


def _digest(path: Path) -> str:
    found = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 256), b""):
            found.update(block)
    return found.hexdigest()
