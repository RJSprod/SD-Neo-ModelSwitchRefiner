"""VibeVoice: what is installed, how its worker is started, and what a render needs.

Voice Box's speech guest, product side. Everything a person can see, choose or
install lives here; everything that needs a tensor lives in
``vibevoice_worker/worker.py``, behind :mod:`mc_voice_vibevoice_runtime`, in a
process of its own with a CUDA PyTorch closure of its own.

Why this is not a copy of :mod:`mc_voice_pocket`
------------------------------------------------
It has the same shape on purpose -- a manifest that is the trust root, a staged
install that promotes by rename, a settings file rather than Forge options, and
a smoke test before anything is promoted -- because those are the shapes the
CPU engines proved. What differs is not cosmetic:

    It takes a card.               Every Voice Chat engine hides every GPU from
                                   its worker. This one is *given* one, by
                                   UUID, through :func:`worker_environment`, and
                                   whose turn it is on that card is
                                   :mod:`mc_turns`' decision, made through the
                                   guest the lead registers. Nothing here
                                   imports the memory side (invariant I-3).

    Its Torch is not on PyPI.      The CUDA 12.8 build lives only on
                                   download.pytorch.org, which the machine that
                                   writes the manifest cannot reach, so the
                                   closure carries a *versioned resolve* for
                                   torch: the publisher's index page is read on
                                   the user's machine, the cp313 win_amd64 wheel
                                   of exactly the pinned version is taken, and
                                   its download is checked against the SHA-256
                                   the index states (:func:`_resolve_torch`).

    Its weights are index-driven.  Nineteen gigabytes in shards whose names are
                                   the model's own statement: the safetensors
                                   index is fetched first, and the shards it
                                   names are downloaded in its order
                                   (:func:`_shards_named`). A maintainer's run
                                   of ``tools/pin_vibevoice_models.py --model``
                                   writes that list with sizes and digests into
                                   the manifest, and until then each shard is
                                   checked against the digest its publisher
                                   states and recorded in the local overlay.

    Its tokenizer is somebody else's. Neither community mirror of the model
                                   ships one; the processor fetches Qwen's from
                                   the hub unless handed a local directory whose
                                   name contains ``qwen``. So the four tokenizer
                                   files are installed into
                                   ``<model>/tokenizer-qwen2.5-7b/`` and the
                                   processor configuration the worker loads is
                                   written here to name it
                                   (:func:`_write_local_config`).

    It remembers what a render cost. The manifest ships an estimate of the
                                   VRAM a render needs; the runtime reports the
                                   peak after every render and
                                   :func:`note_peak` keeps the highest, so
                                   :func:`need_vram_bytes` -- what the turn is
                                   asked for -- stands on a measurement as soon
                                   as there is one.

What is provisional, and says so
--------------------------------
The PyPI wheels are pinned byte for byte. The torch wheel, the model's files,
its shards and the tokenizer are declared and not hashed, because the machine
that wrote the manifest could reach pypi.org and nothing else. Each of those is
checked against the digest its publisher states over HTTPS at install time and
recorded in ``voice/managed-vibevoice-models.local.json``, so the second install
is checked against a constant -- exactly as ``README.md`` "Who vouches for the
bytes" describes, and :attr:`Status.provisional` is how the page says it.
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
from pathlib import Path

import mc_voice_models as models
import mc_voice_paths as paths
import mc_voice_wheelindex as wheelindex

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

LABEL = "VibeVoice"

GUEST = "VibeVoice"
"""The name the lead registers this guest under with ``mc_turns``."""

MODEL_DEFAULT = "vibevoice-7b"

KIND = "vibevoice"
"""What the shared installer's progress table calls this guest's work."""

SCHEMA = 1
"""The manifest and installed-marker layout this build understands."""

SAMPLE_RATE = 24000

STEPS_DEFAULT = 10
STEPS_MIN = 1
STEPS_MAX = 50
CFG_DEFAULT = 1.3
CFG_MIN = 1.0
CFG_MAX = 3.0
SEED_MAX = 2 ** 31 - 1
MAX_NEW_TOKENS_MAX = 32768
"""The bounds :func:`set_settings` holds. Ten diffusion steps and a CFG of 1.3
are upstream's demo defaults; the token cap is the 7B's context."""

WEIGHTS_BYTES_DEFAULT = 18_700_000_000
WORKING_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024
RAM_BYTES_DEFAULT = 3 * 1024 * 1024 * 1024
TOKENIZER_BYTES_DEFAULT = 11_000_000
"""What a render needs when nothing has been measured yet. The weights are the
model card's figure for the bf16 checkpoint; the working set is the KV cache,
the diffusion head's activations and the allocator's slack on a 45-minute
script, generous on purpose until :func:`note_peak` has a number."""

DEFAULT_LOCAL_CONFIG = {
    "processor_class": "VibeVoiceProcessor",
    "speech_tok_compress_ratio": 3200,
    "db_normalize": True,
}
"""What the processor assumes when the mirror ships no preprocessor_config.json,
which is what upstream's own code falls back to."""

SETTINGS_DEFAULTS = {
    "card_uuid": "",
    "model_id": MODEL_DEFAULT,
    "steps": STEPS_DEFAULT,
    "cfg_scale": CFG_DEFAULT,
    "seed": None,
    "max_new_tokens": None,
    "keep_warm": True,
}
"""Voice Box's engine settings, and the shape :func:`settings` always answers in.

``keep_warm`` defaults on: a warm 7B answers the next render in seconds rather
than the minute a cold load costs, and it leaves the card the moment an image
job, WanGP or the language model needs the room (docs/23-voice-box.md §4).
"""

_lock = threading.RLock()
_manifest_cache = None

_HUB = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")
_SAFE_UUID = re.compile(r"^[A-Za-z0-9,\-]*$")


class VibeVoiceError(RuntimeError):
    """A VibeVoice operation that was refused or could not be completed.

    One class, because every ordinary refusal -- not installed, no card, an
    unsupported machine, a setting out of range -- is a state a user can act on
    rather than a fault, and the page shows the sentence.
    """


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Tokenizer:
    """The text tokenizer's files, from a repository of their own."""

    repo: str
    revision: str
    dirname: str
    license: str
    attribution: str
    artifacts: tuple

    @property
    def paths(self) -> tuple:
        """Where each file lands, relative to the model directory."""
        return tuple(f"{self.dirname}/{item.local_name}" for item in self.artifacts)

    @property
    def download_bytes(self) -> int:
        total = sum(int(item.size or 0) for item in self.artifacts)
        return total or TOKENIZER_BYTES_DEFAULT


@dataclass(frozen=True)
class Bundle:
    """One installable VibeVoice model, as the manifest declares it."""

    identifier: str
    label: str
    repo: str
    mirrors: tuple
    revision: str
    license: str
    attribution: str
    summary: str
    notes: str
    sample_rate: int
    artifacts: tuple
    optional: tuple
    shards: tuple
    tokenizer: Tokenizer
    estimates: dict
    local_config: dict
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def required_paths(self) -> tuple:
        """The declared files that must be there, by installed name."""
        return tuple(item.local_name for item in self.artifacts
                     if item.local_name not in self.optional)

    @property
    def weights_bytes(self) -> int:
        """The shards' total when the manifest lists them all with sizes, else 0."""
        sizes = [int(item.size or 0) for item in self.shards]
        return sum(sizes) if sizes and all(sizes) else 0

    @property
    def estimated_weights_bytes(self) -> int:
        return self.weights_bytes or int(self.estimates.get("weights_bytes")
                                         or WEIGHTS_BYTES_DEFAULT)

    @property
    def download_bytes(self) -> int:
        """How large the model part is, for a sentence somebody reads first."""
        heads = sum(int(item.size or 0) for item in self.artifacts)
        return heads + self.estimated_weights_bytes + self.tokenizer.download_bytes


def manifest(refresh: bool = False) -> dict:
    """The parsed, validated VibeVoice manifest. Read once and kept.

    Every failure below is a broken *extension*, not a broken installation, so
    the message says so: a user cannot fix a malformed trust root by pressing
    Install again.
    """
    global _manifest_cache

    with _lock:
        if _manifest_cache is not None and not refresh:
            return _manifest_cache
        path = paths.vibevoice_manifest_path()
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VibeVoiceError(f"The VibeVoice manifest at {path} could not be read ({exc}). "
                                 f"This is a problem with the extension rather than with "
                                 f"your installation.") from None
        if not isinstance(found, dict) or int(found.get("schema") or 0) != SCHEMA:
            raise VibeVoiceError("The VibeVoice manifest is not in a layout this build "
                                 "understands.")
        runtime = found.get("runtime")
        if not isinstance(runtime, dict) or not runtime.get("platforms"):
            raise VibeVoiceError("The VibeVoice manifest declares no runtime.")
        if not isinstance(found.get("models"), dict) or not found["models"]:
            raise VibeVoiceError("The VibeVoice manifest declares no model.")
        _manifest_cache = found
        return found


def _manifest_cache_clear() -> None:
    global _manifest_cache

    with _lock:
        _manifest_cache = None


def _local_pins() -> dict:
    """``{repo/filename: {"sha256", "bytes"}}`` recorded by earlier installs.

    Its own overlay beside its own manifest, keyed by repository *and* filename
    because a shard is called ``model-00001-of-00010.safetensors`` in every
    sharded model there is. Only ever fills a blank -- see :func:`_artifact`.
    """
    try:
        raw = json.loads(paths.vibevoice_local_pins_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    found = {}
    for name, entry in ((raw or {}).get("artifacts") or {}).items():
        digest = str((entry or {}).get("sha256") or "").strip().casefold()
        size = (entry or {}).get("bytes")
        if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) \
                and isinstance(size, int) and size > 0:
            found[str(name)] = {"sha256": digest, "bytes": size}
    return found


def _pin_key(repo: str, filename: str) -> str:
    return f"{repo}/{filename}"


def _repo_of(url: str) -> str:
    """``owner/name`` out of a hub URL, or ``""`` for anything else."""
    found = _HUB.match(str(url or ""))
    return found.group(1) if found else ""


def _artifact(entry: dict, owner: str = "", repo: str = "") -> models.Artifact:
    """One manifest artifact as the shared downloader's own dataclass.

    Built rather than re-implemented, so a file this module fetches goes
    through exactly the checks -- pinned hash, publisher digest, byte ceiling,
    ``.part`` rename -- every other artifact in this feature goes through. A
    hash committed in the manifest wins; a blank is filled from the local
    overlay, which cannot change a committed one.
    """
    if not isinstance(entry, dict):
        raise VibeVoiceError(f"The VibeVoice manifest's {owner or 'artifact list'} holds "
                             f"something that is not an artifact.")
    filename = str(entry.get("filename") or "").strip()
    local_name = str(entry.get("local_name") or filename).strip()
    url = str(entry.get("url") or "").strip()
    if not filename or not local_name:
        raise VibeVoiceError(f"The VibeVoice manifest's {owner} has an artifact with no name.")
    if not url.startswith("https://"):
        raise VibeVoiceError(f"The VibeVoice manifest's {owner} names {filename} somewhere "
                             f"other than an HTTPS address.")
    if "/" in local_name or "\\" in local_name or local_name in (".", ".."):
        raise VibeVoiceError(f"The VibeVoice manifest's {owner} names a file ({local_name!r}) "
                             f"that would not stay in its own folder.")
    sha = str(entry.get("sha256") or "").strip().casefold() or None
    if sha is not None and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
        raise VibeVoiceError(f"The VibeVoice manifest's {owner} gives {filename} a hash that "
                             f"is not a SHA-256.")
    size = entry.get("bytes")
    size = int(size) if isinstance(size, (int, float)) and size > 0 else None
    if sha is None or size is None:
        pinned = _local_pins().get(_pin_key(repo or _repo_of(url), filename))
        if pinned:
            sha = sha or pinned["sha256"]
            size = size or pinned["bytes"]
    return models.Artifact(filename=filename, local_name=local_name, url=url, size=size,
                           sha256=sha)


def _tokenizer(entry: dict, owner: str) -> Tokenizer:
    if not isinstance(entry, dict) or not entry.get("files"):
        raise VibeVoiceError(f"The VibeVoice manifest's {owner} declares no tokenizer files.")
    repo = str(entry.get("repo") or "")
    dirname = str(entry.get("dirname") or paths.VIBEVOICE_TOKENIZER_DIRNAME)
    if "qwen" not in dirname.casefold() or "/" in dirname or "\\" in dirname:
        # Load-bearing rather than cosmetic: the processor picks its tokenizer
        # class by finding ``qwen`` in the location it is handed.
        raise VibeVoiceError(f"The VibeVoice manifest's {owner} puts the tokenizer in "
                             f"{dirname!r}; the directory's name must contain 'qwen'.")
    return Tokenizer(
        repo=repo,
        revision=str(entry.get("revision") or "main"),
        dirname=dirname,
        license=str(entry.get("license") or ""),
        attribution=str(entry.get("attribution") or ""),
        artifacts=tuple(_artifact(item, f"{owner} tokenizer", repo)
                        for item in entry.get("files") or ()))


def model_ids() -> tuple:
    """Every VibeVoice model this build knows, in manifest order."""
    try:
        return tuple((manifest().get("models") or {}).keys())
    except VibeVoiceError:
        return ()


def _default_model_id() -> str:
    known = model_ids()
    try:
        offered = str((manifest().get("defaults") or {}).get("model") or "")
    except VibeVoiceError:
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
        raise VibeVoiceError(f"{wanted!r} is not a VibeVoice model this build knows.")
    repo = str(entry.get("repo") or "")
    artifacts = tuple(_artifact(item, wanted, repo) for item in (entry.get("files") or ()))
    optional = tuple(str(item.get("local_name") or item.get("filename") or "")
                     for item in (entry.get("files") or ()) if item.get("optional"))
    return Bundle(
        identifier=wanted,
        label=str(entry.get("label") or wanted),
        repo=repo,
        mirrors=tuple(str(name) for name in (entry.get("mirrors") or ()) if name),
        revision=str(entry.get("revision") or "main"),
        license=str(entry.get("license") or ""),
        attribution=str(entry.get("attribution") or ""),
        summary=str(entry.get("summary") or ""),
        notes=str(entry.get("notes") or ""),
        sample_rate=int(entry.get("sample_rate") or SAMPLE_RATE),
        artifacts=artifacts,
        optional=optional,
        shards=tuple(_artifact(item, f"{wanted} shards", repo)
                     for item in (entry.get("shards") or ())),
        tokenizer=_tokenizer(entry.get("tokenizer"), wanted),
        estimates=dict(entry.get("estimates") or {}),
        local_config=dict(entry.get("local_config") or DEFAULT_LOCAL_CONFIG),
        raw=entry)


def _platform_entry():
    system, machine, python_version = models.current_platform()
    for entry in manifest()["runtime"].get("platforms") or ():
        if (str(entry.get("system") or "").casefold() == system
                and machine in tuple(str(m).casefold() for m in (entry.get("machines") or ()))
                and str(entry.get("python")) == python_version):
            return entry
    return None


def platform() -> "models.RuntimePlatform | None":
    """The pinned wheel closure for this machine, or ``None`` if it is not on the list.

    An explicit allowlist of one: 64-bit Windows, CPython 3.13, an NVIDIA card
    driven by CUDA 12.8 -- the user's own machine. A machine that is not on it
    gets a sentence rather than an install that half works.

    The torch wheel is *not* among the artifacts returned, because it cannot be
    pinned here: it is a resolve entry, read by :func:`torch_resolve`, answered
    against the publisher's index at install time. A platform whose PyPI wheels
    are listed and unhashed is still returned, so "this build has not pinned
    the closure" and "this machine is not supported" stay different sentences.
    """
    entry = _platform_entry()
    if entry is None:
        return None
    owner = f"runtime {entry.get('id') or ''}"
    artifacts = tuple(_artifact(item, owner) for item in (entry.get("artifacts") or ())
                      if isinstance(item, dict) and not item.get("resolve"))
    return models.RuntimePlatform(
        identifier=str(entry.get("id") or ""),
        system=str(entry.get("system") or "").casefold(),
        machines=tuple(str(m).casefold() for m in (entry.get("machines") or ())),
        python=str(entry.get("python") or ""),
        artifacts=artifacts)


def torch_resolve() -> "dict | None":
    """The one wheel resolved from the publisher's index, for this platform.

    ``{"index", "package", "version"}`` and, once a maintainer has run
    ``tools/pin_vibevoice_models.py --torch`` on a machine that can reach the
    index, ``"filename"``, ``"sha256"`` and ``"bytes"`` as well -- the recorded
    constant a later install is checked against on top of the index's word.
    """
    entry = _platform_entry()
    if entry is None:
        return None
    for item in entry.get("artifacts") or ():
        resolve = item.get("resolve") if isinstance(item, dict) else None
        if resolve is None:
            continue
        if not isinstance(resolve, dict):
            raise VibeVoiceError("The VibeVoice manifest's runtime has a resolve that is "
                                 "not an object.")
        base = str(resolve.get("index") or "")
        package = str(resolve.get("package") or "")
        if not base.startswith("https://") or not package:
            raise VibeVoiceError("The VibeVoice manifest's runtime resolves a wheel from an "
                                 "index that is not HTTPS, or names no package.")
        if item.get("url") or item.get("sha256"):
            raise VibeVoiceError(f"The VibeVoice manifest both pins {package} and resolves "
                                 f"it. One or the other.")
        digest = str(resolve.get("sha256") or "").casefold()
        if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise VibeVoiceError(f"The VibeVoice manifest records a hash for {package} that "
                                 f"is not a SHA-256.")
        return {"local_name": str(item.get("local_name") or package), "index": base,
                "package": package, "version": str(resolve.get("version") or ""),
                "filename": str(resolve.get("filename") or ""), "sha256": digest or "",
                "bytes": int(resolve.get("bytes") or 0)}
    return None


def full_torch_version(resolve: dict) -> str:
    """``2.8.0`` on a ``/whl/cu128/`` index is ``2.8.0+cu128`` on its page."""
    version = str((resolve or {}).get("version") or "")
    if "+" in version or not version:
        return version
    found = re.search(r"/whl/([A-Za-z0-9_]+)/", str((resolve or {}).get("index") or ""))
    return f"{version}+{found.group(1)}" if found else version


def pinned() -> bool:
    """Whether this build has resolved its PyPI closure for this machine.

    The torch resolve does not count against it, by design: it can never be
    pinned from the machine that writes the manifest, and refusing every
    install until it is would be refusing the feature.
    """
    try:
        chosen = platform()
    except VibeVoiceError:
        return False
    if chosen is None or not chosen.artifacts:
        return False
    return all(item.pinned for item in chosen.artifacts)


def closure_id() -> str:
    """A fingerprint of exactly which wheels this platform installs.

    The pinned wheels' digests and the torch *version*, not the torch digest:
    recording the digest later with the pin tool must not make every installed
    runtime stale, since the bytes it names are the bytes already unpacked.
    """
    chosen = platform()
    resolve = torch_resolve()
    if chosen is None:
        return ""
    parts = [chosen.closure_id, f"torch:{full_torch_version(resolve) if resolve else ''}"]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _runtime_bytes(chosen, resolve) -> int:
    total = sum(int(item.size or 0) for item in (chosen.artifacts if chosen else ()))
    torch_bytes = int((resolve or {}).get("bytes") or 0)
    if not torch_bytes:
        try:
            torch_bytes = int((manifest()["runtime"].get("torch") or {}).get("about_bytes") or 0)
        except VibeVoiceError:
            torch_bytes = 0
    return total + torch_bytes


def _provisional(entry: dict) -> bool:
    """Whether the manifest's own claim about the model is declared-not-hashed.

    From the committed entries and not the overlay: the page says "checked
    against the publisher's digest and recorded here" until a maintainer has
    turned that into a committed constant, whatever an earlier install wrote.
    """
    items = [item for item in (entry.get("files") or ()) if not item.get("optional")]
    items += list(entry.get("shards") or ())
    items += list((entry.get("tokenizer") or {}).get("files") or ())
    if not entry.get("shards"):
        return True
    return any(len(str(item.get("sha256") or "")) != 64 or int(item.get("bytes") or 0) <= 0
               for item in items)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


@dataclass
class Status:
    """What is installed, in one object the page, the runtime and the lead read."""

    supported: bool = False
    runtime_installed: bool = False
    model_installed: bool = False
    tokenizer_installed: bool = False
    provisional: bool = True
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
        """Whether a render can be attempted: runtime, model and tokenizer."""
        return bool(self.supported and self.runtime_installed and self.model_installed
                    and self.tokenizer_installed)

    @property
    def message(self) -> str:
        if self.ready:
            return "Installed."
        if not self.supported:
            return self.runtime_message
        missing = []
        if not self.runtime_installed:
            missing.append("runtime")
        if not self.model_installed:
            missing.append("model")
        elif not self.tokenizer_installed:
            missing.append("tokenizer (install the model again)")
        return f"Setup required — VibeVoice's {' and '.join(missing)} still to install."


def status() -> Status:
    """Read from disk. Starts nothing, downloads nothing, never raises."""
    try:
        return _status()
    except VibeVoiceError as exc:
        return Status(runtime_message=str(exc), model_message=str(exc))
    except Exception:
        logger.debug("Model Chain: could not read the VibeVoice installation", exc_info=True)
        text = "VibeVoice's installation could not be read."
        return Status(runtime_message=text, model_message=text)


def _unsupported_sentence(tail: str) -> str:
    system, machine, python_version = models.current_platform()
    return (f"VibeVoice needs Windows with an NVIDIA card on Python 3.13, and this machine "
            f"is {system}/{machine} on Python {python_version}{tail}")


def _status() -> Status:
    chosen = platform()
    entry = bundle()
    found = Status(model_id=entry.identifier, model_label=entry.label,
                   provisional=_provisional(entry.raw))
    if chosen is None:
        found.supported = False
        found.runtime_message = _unsupported_sentence(
            ". Voice Chat's CPU engines are unaffected and still work here.")
        found.model_message = found.runtime_message
        return found
    found.supported = True
    resolve = torch_resolve()
    found.runtime_bytes = _runtime_bytes(chosen, resolve)
    installed = _read_json(paths.vibevoice_runtime_manifest())
    if not installed and not pinned():
        found.runtime_message = (
            "Not installed — this build has not pinned a VibeVoice runtime closure yet, so it "
            "will not download one. A maintainer runs tools/pin_vibevoice_models.py; you can "
            "install from a folder you filled yourself in the meantime.")
    elif not installed:
        found.runtime_message = (
            f"Not installed — about {models._bytes_label(found.runtime_bytes)} of PyTorch "
            f"(CUDA 12.8), transformers and VibeVoice.")
    elif str(installed.get("closure") or "") != closure_id() or runtime_python() is None:
        found.runtime_message = ("Installed, but this build pins a different VibeVoice "
                                 "runtime. Install it again to update.")
    else:
        found.runtime_installed = True
        found.runtime_message = (
            f"Installed — VibeVoice {installed.get('vibevoice_version') or '?'}, torch "
            f"{installed.get('torch_version') or '?'}, transformers "
            f"{installed.get('transformers_version') or '?'}.")
    found.closure = dict(installed or {})

    root = paths.vibevoice_model_root(entry.identifier)
    marker = _read_json(root / paths.INSTALLED_FILENAME)
    found.installed_model = dict(marker or {})
    found.model_bytes = entry.download_bytes
    missing = _model_missing(entry, root, marker)
    if not marker:
        found.model_message = (f"Not installed — {entry.label}, about "
                               f"{models._bytes_label(found.model_bytes)}.")
    elif missing:
        found.model_message = (f"Installed, but {missing[0]} is missing — install the "
                               f"model again.")
    else:
        found.model_installed = True
        shards = list((marker or {}).get("shards") or _installed_shards(root))
        size = sum(int(value) for name, value in ((marker or {}).get("bytes") or {}).items()
                   if name in shards and isinstance(value, int))
        found.model_message = (f"Installed — {entry.label}, {len(shards)} shard"
                               f"{'s' if len(shards) != 1 else ''}"
                               f"{', ' + models._bytes_label(size) if size else ''}.")
    found.tokenizer_installed = _tokenizer_present(entry, root)
    if found.model_installed and not found.tokenizer_installed:
        found.model_message += " Its tokenizer is missing — install the model again."
    found.download_bytes = ((0 if found.runtime_installed else found.runtime_bytes)
                            + (0 if found.model_installed and found.tokenizer_installed
                               else found.model_bytes))
    return found


def _installed_shards(root: Path) -> list:
    """The shards the installed index names, or nothing if it is unreadable."""
    index = _read_json(root / "model.safetensors.index.json")
    try:
        return _shards_named(index) if index else []
    except VibeVoiceError:
        return []


def _model_missing(entry: Bundle, root: Path, marker) -> list:
    """Which installed files are not where the marker says they are."""
    wanted = list(entry.required_paths)
    if marker:
        wanted += list(marker.get("shards") or _installed_shards(root))
        wanted.append(paths.VIBEVOICE_LOCAL_CONFIG)
    return [name for name in wanted if not (root / name).is_file()]


def _tokenizer_present(entry: Bundle, root: Path) -> bool:
    return all((root / name).is_file() for name in entry.tokenizer.paths)


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
        "installed": found.ready,
        "ready": found.ready,
        "supported": found.supported,
        "runtime_installed": found.runtime_installed,
        "model_installed": found.model_installed,
        "tokenizer_installed": found.tokenizer_installed,
        "provisional": found.provisional,
        "pinned": is_pinned,
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
            {"id": "runtime", "label": "Runtime — PyTorch (CUDA 12.8), transformers and "
                                       "VibeVoice",
             "installed": found.runtime_installed, "bytes": int(found.runtime_bytes),
             "message": found.runtime_message},
            {"id": "model", "label": f"{found.model_label or LABEL} and its tokenizer",
             "installed": bool(found.model_installed and found.tokenizer_installed),
             "bytes": int(found.model_bytes), "message": found.model_message},
        ],
        "settings": settings(),
        "need_vram_bytes": int(vram),
        "need_ram_bytes": int(ram),
        "refusal": refusal(),
        "progress": progress(),
    }


def _model_label(identifier: str) -> str:
    try:
        return bundle(identifier).label
    except VibeVoiceError:
        return str(identifier)


# --------------------------------------------------------------------------- #
# The worker's side: interpreter, environment, script, model directory
# --------------------------------------------------------------------------- #


def runtime_python() -> "Path | None":
    """The isolated VibeVoice interpreter, or ``None`` when it is not installed."""
    interpreter = _interpreter_in(paths.vibevoice_runtime_root())
    return interpreter if interpreter.exists() else None


def _interpreter_in(root: Path) -> Path:
    environment = root / "env"
    return ((environment / "Scripts" / "python.exe") if os.name == "nt"
            else (environment / "bin" / "python"))


def worker_environment(card_uuid: str) -> dict:
    """The environment the VibeVoice worker runs in, on one card.

    ``CUDA_VISIBLE_DEVICES`` is the card's UUID as nvidia-smi spells it, so the
    worker's device 0 is that card and no other; the handshake then proves it
    (the runtime refuses a worker that came up elsewhere). An empty card leaves
    the variable alone rather than hiding every GPU, which is what the smoke
    test wants and no render ever asks for.

    ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` are real instructions here,
    not precautions: the processor would fetch a tokenizer from the hub if the
    local configuration did not name one, and these make that a refusal.

    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` keeps the caching
    allocator from fragmenting across a 45-minute generation. What is
    deliberately *absent* is ``PYTORCH_NO_CUDA_MEMORY_CACHING``, which the CPU
    engines set: on a card it would disable the caching allocator altogether.

    No credential is here, and there is no branch that could put one here.
    """
    found = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "4",
    }
    wanted = str(card_uuid or "").strip()
    if wanted:
        if not _SAFE_UUID.match(wanted) or len(wanted) > 128:
            raise VibeVoiceError(f"{wanted!r} is not a card identifier.")
        found["CUDA_VISIBLE_DEVICES"] = wanted
    return found


def worker_script() -> Path:
    return paths.vibevoice_worker_script()


def model_dir(identifier: str = "") -> Path:
    """Where the selected (or named) model lives. Validated, never a caller's path."""
    return paths.vibevoice_model_root(bundle(identifier).identifier)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def settings() -> dict:
    """Voice Box's engine settings, defaults filled in, stored values validated.

    A stored value that no longer validates -- a build that narrowed a range,
    a hand-edited file -- falls back to the default for that key alone rather
    than taking the whole file down with it.
    """
    stored = _settings_read()
    found = dict(SETTINGS_DEFAULTS)
    for key in SETTINGS_DEFAULTS:
        if key in stored:
            try:
                found[key] = _validated(key, stored[key])
            except VibeVoiceError:
                logger.debug("Model Chain: a stored VibeVoice setting (%s) was ignored", key)
    if found["model_id"] not in model_ids():
        found["model_id"] = _default_model_id() or MODEL_DEFAULT
    return found


def set_settings(values: dict) -> dict:
    """Change engine settings. Validated, refused rather than dropped.

    An unknown key is refused rather than ignored, and so is a value out of
    range: a page that sent a step count this build does not accept and was
    answered with the unchanged settings and no error would show the old value
    with no explanation of why its press did nothing.
    """
    offered = {str(key): value for key, value in dict(values or {}).items()}
    unknown = sorted(set(offered) - set(SETTINGS_DEFAULTS))
    if unknown:
        raise VibeVoiceError(f"{unknown[0]!r} is not a VibeVoice setting.")
    checked = {key: _validated(key, value) for key, value in offered.items()}
    current = _settings_read()
    current.update(checked)
    _settings_write(current)
    if checked:
        logger.info("Model Chain: VibeVoice settings changed — %s",
                    ", ".join(sorted(checked)))
    return settings()


def _validated(key: str, value):
    if key == "steps":
        number = _whole(value, "Diffusion steps")
        if number is None or not STEPS_MIN <= number <= STEPS_MAX:
            raise VibeVoiceError(f"Diffusion steps are a whole number from {STEPS_MIN} to "
                                 f"{STEPS_MAX}.")
        return number
    if key == "cfg_scale":
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        if not (CFG_MIN <= number <= CFG_MAX):
            raise VibeVoiceError(f"CFG scale is a number from {CFG_MIN:.1f} to {CFG_MAX:.1f}.")
        return round(number, 3)
    if key == "seed":
        if value is None or (isinstance(value, str) and value.strip().lower()
                             in ("", "random", "none")):
            return None
        number = _whole(value, "The seed")
        if number is None or not 0 <= number <= SEED_MAX:
            raise VibeVoiceError(f"The seed is blank for random, or a whole number from 0 "
                                 f"to {SEED_MAX}.")
        return number
    if key == "max_new_tokens":
        if value is None or (isinstance(value, str) and value.strip().lower()
                             in ("", "auto", "automatic", "none")):
            return None
        number = _whole(value, "Max new tokens")
        if number is None or not 1 <= number <= MAX_NEW_TOKENS_MAX:
            raise VibeVoiceError(f"Max new tokens is blank for automatic, or a whole number "
                                 f"from 1 to {MAX_NEW_TOKENS_MAX}.")
        return number
    if key == "keep_warm":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0",
                                                                "yes", "no", "on", "off"):
            return value.strip().lower() in ("true", "1", "yes", "on")
        raise VibeVoiceError("Keep warm is on or off.")
    if key == "card_uuid":
        text = str(value or "").strip()
        if text and (len(text) > 128 or not _SAFE_UUID.match(text) or "," in text):
            raise VibeVoiceError(f"{text!r} is not a card identifier.")
        return text
    if key == "model_id":
        text = str(value or "").strip()
        if not text:
            return _default_model_id() or MODEL_DEFAULT
        if text not in model_ids():
            raise VibeVoiceError(f"{text!r} is not a VibeVoice model this build has recorded.")
        return text
    raise VibeVoiceError(f"{key!r} is not a VibeVoice setting.")


def _whole(value, what: str):
    """``value`` as an int, or ``None`` when it is not one. A float that is not
    whole is not one either: ``10.5`` steps is not a setting."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def _settings_read() -> dict:
    try:
        found = json.loads(paths.vibevoice_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _settings_write(found: dict) -> None:
    """Replace the settings file atomically, like every other file here."""
    _write_json(paths.vibevoice_settings_path(), found)


# --------------------------------------------------------------------------- #
# What a render needs
# --------------------------------------------------------------------------- #


def calibration() -> dict:
    """``{model id: {"peak_bytes", "rss_bytes", "renders", "updated"}}`` observed here."""
    found = _read_json(paths.vibevoice_calibration_path()) or {}
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


def note_peak(identifier: str, peak_bytes: int, rss_bytes: int = 0) -> None:
    """Remember the most a render has cost on this machine. Never raises.

    Called by the runtime after every render with torch's peak reserved bytes
    and the worker's resident set. The highest of each is kept, and written
    atomically: this file is read before every turn request, and a half-written
    one would be an estimate nobody made.
    """
    name = str(identifier or "").strip()
    try:
        peak = max(int(peak_bytes or 0), 0)
        rss = max(int(rss_bytes or 0), 0)
    except (TypeError, ValueError):
        return
    if not name or len(name) > 64 or not all(c.isalnum() or c in "-_." for c in name) \
            or (peak <= 0 and rss <= 0):
        return
    with _lock:
        found = _read_json(paths.vibevoice_calibration_path()) or {}
        entries = found.get("models") if isinstance(found.get("models"), dict) else {}
        current = dict(entries.get(name) or {}) if isinstance(entries.get(name), dict) else {}
        current["peak_bytes"] = max(int(current.get("peak_bytes") or 0), peak)
        current["rss_bytes"] = max(int(current.get("rss_bytes") or 0), rss)
        current["renders"] = int(current.get("renders") or 0) + 1
        current["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entries[name] = current
        try:
            _write_json(paths.vibevoice_calibration_path(),
                        {"schema": SCHEMA, "models": entries})
        except OSError:
            logger.debug("Model Chain: could not record what a VibeVoice render cost",
                         exc_info=True)


def need_vram_bytes(identifier: str = "") -> int:
    """What to ask the turn for: the estimate, or the peak a render has reached.

    The estimate is the weights -- the shards' committed sizes when the manifest
    lists them, else what earlier installs recorded, else the model card's
    figure -- plus the working set. The calibration's observed peak takes over
    once it is larger, because a measurement of this machine beats an estimate
    of any machine.
    """
    entry = bundle(identifier)
    working = int(entry.estimates.get("working_bytes") or WORKING_BYTES_DEFAULT)
    estimate = _weights_bytes(entry) + working
    observed = int(calibration().get(entry.identifier, {}).get("peak_bytes") or 0)
    return max(estimate, observed)


def need_ram_bytes(identifier: str = "") -> int:
    entry = bundle(identifier)
    estimate = int(entry.estimates.get("ram_bytes") or RAM_BYTES_DEFAULT)
    observed = int(calibration().get(entry.identifier, {}).get("rss_bytes") or 0)
    return max(estimate, observed)


def _weights_bytes(entry: Bundle) -> int:
    """The shards' total: committed sizes, then recorded sizes, then the estimate."""
    if entry.weights_bytes:
        return entry.weights_bytes
    root = paths.vibevoice_model_root(entry.identifier)
    marker = _read_json(root / paths.INSTALLED_FILENAME) or {}
    names = [item.filename for item in entry.shards] or list(marker.get("shards") or ())
    pins = _local_pins()
    if names:
        repos = marker.get("repos") if isinstance(marker.get("repos"), dict) else {}
        sizes = [int((pins.get(_pin_key(str(repos.get(name) or marker.get("repo")
                                                or entry.repo), name)) or {}).get("bytes")
                     or 0) for name in names]
        if all(sizes):
            return sum(sizes)
        recorded = marker.get("bytes") if isinstance(marker.get("bytes"), dict) else {}
        sizes = [int(recorded.get(name) or 0) for name in names]
        if all(sizes):
            return sum(sizes)
    return int(entry.estimates.get("weights_bytes") or WEIGHTS_BYTES_DEFAULT)


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
    """Why VibeVoice cannot be *installed* right now, or an empty string."""
    try:
        chosen = platform()
    except VibeVoiceError as exc:
        return str(exc)
    if chosen is None:
        return _unsupported_sentence(", so it cannot be installed here.")
    with models._lock:
        if (models._progress.get(KIND) or {}).get("running"):
            return "VibeVoice is still being installed."
    if not manual and not pinned():
        return ("This build has not pinned a VibeVoice runtime closure yet, so it will not be "
                "downloaded. A maintainer runs tools/pin_vibevoice_models.py on a machine "
                "that can reach pypi.org; you can install from a folder you filled yourself "
                "in the meantime.")
    return ""


def refusal(manual: bool = False) -> str:
    """Why a render cannot run now, in one sentence the page can show, or ``""``.

    Asked before anything is started and separately from starting it, because
    the two questions have different audiences: this one answers a browser that
    needs a sentence, and the install transaction answers a log. ``manual``
    means the caller can fall back to a folder install, so an unpinned closure
    is not a refusal for them.
    """
    found = _install_refusal(manual)
    if found:
        return found
    try:
        current = _status()
    except VibeVoiceError as exc:
        return str(exc)
    if not _nvidia_present():
        return ("VibeVoice needs an NVIDIA card, and this machine has no NVIDIA driver "
                "(nvidia-smi was not found).")
    if not current.runtime_installed:
        return "VibeVoice's runtime is not installed — install it below."
    if not current.model_installed:
        return f"The {current.model_label or LABEL} model is not installed — install it below."
    if not current.tokenizer_installed:
        return "VibeVoice's tokenizer is not installed — install the model again."
    return ""


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #


PARTS = ("runtime", "model")


def sources(part: str = "runtime") -> list:
    """Where a person would go to fetch VibeVoice's files by hand."""
    if part == "runtime":
        chosen = platform()
        found = [{"filename": item.filename, "url": item.url, "save_as": item.local_name,
                  "archive": False} for item in (chosen.artifacts if chosen else ())]
        resolve = torch_resolve()
        if resolve:
            name = resolve.get("filename") or (
                f"{resolve['package']}-{full_torch_version(resolve)}-cp313-cp313-win_amd64.whl")
            found.append({"filename": name, "url": resolve["index"], "save_as": name,
                          "archive": False})
        return found
    entry = bundle()
    found = [{"filename": item.filename, "url": item.url, "save_as": item.local_name,
              "archive": False} for item in entry.artifacts + entry.shards]
    found += [{"filename": item.filename, "url": item.url,
               "save_as": f"{entry.tokenizer.dirname}/{item.local_name}", "archive": False}
              for item in entry.tokenizer.artifacts]
    return found


def install(part: str = "", on_status=None, on_progress=None) -> Status:
    """Install what is missing: the runtime, the model with its tokenizer, or both.

    A transaction per part. Nothing outside a staging directory is touched until
    every byte has arrived and matched what was expected, each part is promoted
    by a directory rename, and a failure leaves the installation as it was.
    """
    wanted = str(part or "").strip().lower()
    if wanted and wanted not in PARTS:
        raise VibeVoiceError("VibeVoice installs its runtime or its model (with its "
                             "tokenizer).")
    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)
    with models._claim(KIND, say, bundle().identifier):
        if wanted in ("", "runtime"):
            say("Checking the VibeVoice runtime…")
            share = 0.15 if not wanted else 1.0
            install_runtime(on_status=say, on_progress=lambda f: tick(f * share))
        if wanted in ("", "model"):
            base = 0.15 if not wanted else 0.0
            install_model(on_status=say,
                          on_progress=lambda f: tick(base + f * (1.0 - base)))
        tick(1.0)
        say("VibeVoice installed.")
        return status()


def install_from(part: str, folder: str, on_status=None, on_progress=None) -> Status:
    """Install from files already on this machine. The escape hatch.

    A folder of the runtime's wheels (the pinned ones and the torch wheel), or a
    folder holding the mirror's files -- config, index, shards, optionally the
    processor config and a tokenizer subfolder. A pinned artifact is checked
    against the hash committed here; an unpinned one has its digest recorded and
    becomes the constant the next install is checked against.
    """
    wanted = str(part or "").strip().lower()
    if wanted not in PARTS:
        raise VibeVoiceError("VibeVoice installs its runtime or its model (with its "
                             "tokenizer).")
    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)
    with models._claim(KIND, say, bundle().identifier):
        if wanted == "runtime":
            install_runtime(on_status=say, on_progress=tick, folder=folder)
        else:
            install_model(on_status=say, on_progress=tick, folder=folder)
        return status()


def _folder(folder) -> Path:
    source = Path(str(folder or "").strip().strip('"')).expanduser()
    if not str(folder or "").strip():
        raise VibeVoiceError("Give the folder the downloaded files are in.")
    if source.is_file():
        source = source.parent
    if not source.is_dir():
        raise VibeVoiceError(f"{source} is not a folder this machine can read.")
    return source


def install_runtime(on_status=None, on_progress=None, folder=None) -> None:
    """The isolated CUDA closure: an interpreter of its own and the wheels unpacked.

    Built the way every voice runtime is -- a virtual environment without pip,
    the verified wheels unpacked rather than installed by a package manager --
    with one wheel more than the manifest can name: torch, resolved from the
    publisher's index on this machine (:func:`_resolve_torch`).
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    chosen = platform()
    if chosen is None:
        raise VibeVoiceError(_install_refusal() or "VibeVoice has no runtime for this platform.")
    if not chosen.artifacts:
        raise VibeVoiceError(_install_refusal(manual=bool(folder))
                             or "This build has not recorded a VibeVoice runtime closure.")
    resolve = torch_resolve()
    if resolve is None:
        raise VibeVoiceError("The VibeVoice manifest names no torch wheel for this platform.")
    installed = _read_json(paths.vibevoice_runtime_manifest())
    if installed and str(installed.get("closure") or "") == closure_id() \
            and runtime_python() is not None:
        say("The VibeVoice runtime is already installed.")
        tick(1.0)
        return

    _stop_runtime("the VibeVoice runtime is being installed")
    staging = paths.vibevoice_staging_for("runtime", uuid.uuid4().hex[:8])
    wheels = staging / "wheels"
    shutil.rmtree(staging, ignore_errors=True)
    wheels.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            source = _folder(folder)
            _adopt(chosen.artifacts, source, wheels, say, "VibeVoice runtime wheel")
            torch_item = _adopt_torch(resolve, source, wheels, say)
            tick(0.7)
        else:
            say("Asking the publisher which torch wheel this machine needs…")
            torch_item = _resolve_torch(resolve, chosen, say)
            everything = tuple(chosen.artifacts) + (torch_item,)
            expectations = models._expectations(chosen.artifacts, say)
            expectations[torch_item.local_name] = models.Expected(
                torch_item.size, torch_item.sha256, "the publisher's package index, over HTTPS")
            room = dict(expectations)
            if not torch_item.size:
                room[torch_item.local_name] = models.Expected(
                    _runtime_bytes(None, resolve), torch_item.sha256, "an estimate")
            models._make_room(everything, staging, room)
            models._fetch_all(everything, wheels, say, tick, 0.7, expectations)
        digest = _digest(wheels / torch_item.local_name)
        say("Building the isolated VibeVoice runtime…")
        _build_environment(staging, wheels, models.RuntimePlatform(
            identifier=chosen.identifier, system=chosen.system, machines=chosen.machines,
            python=chosen.python, artifacts=tuple(chosen.artifacts) + (torch_item,)))
        tick(0.85)
        say("Checking that VibeVoice imports on this machine…")
        report = _smoke_test(staging)
        tick(0.95)
        shutil.rmtree(wheels, ignore_errors=True)
        runtime = manifest()["runtime"]
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "closure": closure_id(),
            "platform": chosen.identifier,
            "python": chosen.python,
            "accelerator": "cuda",
            "vibevoice_version": str(report.get("vibevoice") or runtime.get("version") or ""),
            "torch_version": str(report.get("torch") or full_torch_version(resolve)),
            "transformers_version": str(report.get("transformers")
                                        or runtime.get("transformers_version") or ""),
            "cuda": report.get("cuda"),
            "torch_wheel": {"filename": torch_item.filename, "sha256": digest,
                            "bytes": (wheels / torch_item.local_name).stat().st_size
                            if (wheels / torch_item.local_name).exists() else
                            int(torch_item.size or 0),
                            "index": resolve["index"]},
            "artifacts": {item.local_name: item.sha256 for item in chosen.artifacts},
            "license": runtime.get("license") or "",
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        models._promote(staging, paths.vibevoice_runtime_root())
        say("The VibeVoice runtime is installed.")
        tick(1.0)
        logger.info("Model Chain: the VibeVoice runtime is installed — %s, torch %s (%s)",
                    chosen.identifier, report.get("torch") or full_torch_version(resolve),
                    torch_item.filename)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _resolve_torch(resolve: dict, chosen, say) -> models.Artifact:
    """The one torch wheel this machine installs, from the publisher's own index.

    Read through the one module allowed to reach the network and chosen by
    :mod:`mc_voice_wheelindex`: the cp313 win_amd64 wheel of exactly the pinned
    version, with the SHA-256 the index states. A digest the manifest recorded
    (``--torch``) must agree with the index's; a publisher that changed the
    wheel under the same version is refused rather than followed.
    """
    version = full_torch_version(resolve)
    machine = (list(chosen.machines) or ["amd64"])[0]
    tags = wheelindex.platform_tags(chosen.python, chosen.system, machine)
    try:
        page = models.read_index_page(resolve["index"])
    except models.VoiceError:
        raise
    except Exception as exc:
        raise VibeVoiceError(f"The publisher's index for torch could not be read "
                             f"({exc.__class__.__name__}). Nothing was installed.") from None
    try:
        found = wheelindex.choose(page, resolve["index"], resolve["package"], version, tags)
    except wheelindex.IndexError_ as exc:
        raise VibeVoiceError(str(exc)) from None
    recorded = str(resolve.get("sha256") or "")
    if recorded and recorded != found["sha256"]:
        raise VibeVoiceError(
            f"The publisher now serves a different {found['filename']} than this build "
            f"recorded — its digest has changed under the same version. Nothing was "
            f"downloaded; a maintainer reviews the change with "
            f"tools/pin_vibevoice_models.py --torch.")
    size = int(resolve.get("bytes") or 0) or models._declared_size(found["url"])
    say(f"torch {version} is {found['filename']}"
        f"{' (' + models._bytes_label(size) + ')' if size else ''}.")
    logger.info("Model Chain: VibeVoice resolved torch from the publisher's index — %s, "
                "sha256 %s", found["filename"], found["sha256"][:12])
    return models.Artifact(filename=found["filename"], local_name=found["filename"],
                           url=found["url"], size=size or None, sha256=found["sha256"])


def _adopt_torch(resolve: dict, source: Path, wheels: Path, say) -> models.Artifact:
    """The torch wheel out of a folder somebody filled themselves."""
    version = full_torch_version(resolve)
    wanted = [resolve.get("filename") or "",
              f"{resolve['package']}-{version}-cp313-cp313-win_amd64.whl"]
    found = models._find(source, [name for name in wanted if name])
    if found is None:
        raise VibeVoiceError(f"{wanted[-1]} is not in {source}. It comes from "
                             f"{resolve['index']}. Nothing was installed.")
    say(f"Checking {found.name}…")
    digest = _digest(found)
    recorded = str(resolve.get("sha256") or "")
    if recorded and digest != recorded:
        raise VibeVoiceError(f"{found.name} is in that folder, but its contents are not what "
                             f"this extension's manifest recorded. Nothing was installed.")
    shutil.copyfile(found, wheels / found.name)
    return models.Artifact(filename=found.name, local_name=found.name,
                           url=resolve["index"], size=found.stat().st_size, sha256=digest)


def install_model(on_status=None, on_progress=None, folder=None) -> None:
    """The model's files, its shards from its own index, and its tokenizer.

    One staging directory and one promote for all three, so a model is either
    there with its tokenizer and its local configuration or not there at all.
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    entry = bundle()
    target = paths.vibevoice_model_root(entry.identifier)
    marker = _read_json(target / paths.INSTALLED_FILENAME)
    missing = _model_missing(entry, target, marker)
    if marker and not missing and _tokenizer_present(entry, target):
        say(f"{entry.label} is already installed.")
        tick(1.0)
        return
    if marker and missing == [paths.VIBEVOICE_LOCAL_CONFIG] and _tokenizer_present(entry,
                                                                                   target):
        # Everything is in place but the one file this module writes itself.
        say(f"Writing {entry.label}'s processor configuration…")
        _write_local_config(entry)
        tick(1.0)
        return

    _stop_runtime("the VibeVoice model is being installed")
    staging = paths.vibevoice_staging_for(entry.identifier, uuid.uuid4().hex[:8])
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            record = _adopt_model(entry, _folder(folder), staging, say, tick)
        else:
            record = _download_model(entry, staging, say, tick)
        shards = list(record["shards"])
        required = tuple(entry.required_paths) + tuple(shards) + tuple(entry.tokenizer.paths)
        say(f"Checking {entry.label} is complete…")
        _sanity_check(staging, required, entry.label)
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "id": entry.identifier,
            "label": entry.label,
            "repo": record["repo"],
            "repos": record["repos"],
            "revision": entry.revision,
            "license": entry.license,
            "attribution": entry.attribution,
            "source": record["source"],
            "source_folder": record.get("source_folder", ""),
            "digests": {name: digest for name, digest in sorted(record["digests"].items())},
            "bytes": {name: size for name, size in sorted(record["sizes"].items())},
            "verified_by": {name: how for name, how in sorted(record["verified"].items())},
            "shards": shards,
            "tokenizer": {"repo": entry.tokenizer.repo, "revision": entry.tokenizer.revision,
                          "dirname": entry.tokenizer.dirname},
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        # Into the staging tree, naming the tokenizer where it will *be*, so the
        # promote below carries a model that is complete rather than one that
        # is complete a moment later.
        _write_local_config(entry, root=staging)
        say("Installing…")
        models._promote(staging, target)
        _record_pins(entry, record)
        say(f"{entry.label} is installed.")
        tick(1.0)
        logger.info("Model Chain: the VibeVoice model %s is installed — %d shard(s) from %s",
                    entry.identifier, len(shards), record["repo"])
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _new_record(entry: Bundle, source: str) -> dict:
    return {"digests": {}, "sizes": {}, "verified": {}, "repos": {}, "shards": [],
            "repo": entry.repo, "source": source}


def _download_model(entry: Bundle, staging: Path, say, tick) -> dict:
    """Config and index first, then every shard the index names, then the tokenizer."""
    record = _new_record(entry, "hub")
    heads = [item for item in entry.artifacts if item.local_name not in entry.optional]
    extras = [item for item in entry.artifacts if item.local_name in entry.optional]
    _fetch(entry, heads, staging, say, tick, 0.0, 0.02, record)
    for item in extras:
        try:
            _fetch(entry, [item], staging, say, tick, 0.02, 0.0, record)
        except models.VoiceError as exc:
            say(f"{item.filename} is not published by the mirror; the defaults will be "
                f"written ({exc}).")
            logger.info("Model Chain: VibeVoice's mirror has no %s; the local processor "
                        "configuration is written from defaults", item.filename)
    shards = _shards_named(_read_index(staging / "model.safetensors.index.json"))
    record["shards"] = list(shards)
    say(f"The index names {len(shards)} shard{'s' if len(shards) != 1 else ''}.")
    _fetch(entry, _shard_artifacts(entry, shards), staging, say, tick, 0.02, 0.93, record)
    tokenizer_root = staging / entry.tokenizer.dirname
    _fetch(entry, list(entry.tokenizer.artifacts), tokenizer_root, say, tick, 0.95, 0.05,
           record, prefix=entry.tokenizer.dirname + "/")
    used = [record["repos"].get(name) for name in shards]
    record["repo"] = next((repo for repo in used if repo), entry.repo)
    return record


def _shard_artifacts(entry: Bundle, shards) -> list:
    """The shards as artifacts: the manifest's when it lists them, else built.

    When the manifest lists shards, the index's names must be exactly those
    names -- a mirror whose index names a shard this build never recorded is a
    mirror that changed under the pin, and a maintainer looks at it.
    """
    if entry.shards:
        listed = {item.filename: item for item in entry.shards}
        unknown = [name for name in shards if name not in listed]
        if unknown or set(listed) - set(shards):
            raise VibeVoiceError(
                f"{entry.label}'s index names shards this build has not recorded "
                f"({', '.join(unknown[:3]) or 'a different set'}). The mirror changed under "
                f"the pin; a maintainer runs tools/pin_vibevoice_models.py --model. Nothing "
                f"was installed.")
        return [listed[name] for name in shards]
    return [_artifact({"filename": name, "local_name": name,
                       "url": _hub_url(entry.repo, entry.revision, name)},
                      f"{entry.identifier} shards", entry.repo) for name in shards]


def _hub_url(repo: str, revision: str, filename: str) -> str:
    return (f"https://huggingface.co/{repo}/resolve/{urllib.parse.quote(revision)}/"
            f"{urllib.parse.quote(filename)}")


def _mirror_artifacts(entry: Bundle, item: models.Artifact) -> list:
    """The same file from each fallback mirror, at its ``main``.

    A committed digest travels with it -- the mirror has to serve the same
    bytes -- and an uncommitted one is looked up under the mirror's own key.
    """
    found = []
    raw = {"filename": item.filename, "local_name": item.local_name}
    committed = _committed_digest(entry, item.filename)
    if committed:
        raw["sha256"], raw["bytes"] = committed
    for mirror in entry.mirrors:
        if mirror == _repo_of(item.url):
            continue
        found.append(_artifact({**raw, "url": _hub_url(mirror, "main", item.filename)},
                               f"{entry.identifier} mirror", mirror))
    return found


def _committed_digest(entry: Bundle, filename: str):
    """``(sha256, bytes)`` the manifest itself commits for ``filename``, or ``None``."""
    rows = list(entry.raw.get("files") or ()) + list(entry.raw.get("shards") or ())
    rows += list((entry.raw.get("tokenizer") or {}).get("files") or ())
    for row in rows:
        if str(row.get("filename") or "") == filename and row.get("sha256") \
                and int(row.get("bytes") or 0) > 0:
            return str(row["sha256"]).casefold(), int(row["bytes"])
    return None


def _fetch(entry: Bundle, artifacts, destination: Path, say, tick, base: float,
           share: float, record: dict, prefix: str = "") -> None:
    """Download a phase of artifacts through the shared, checked downloader.

    Each artifact is asked about first (the publisher's size and digest), the
    disk is checked for the whole phase, and each file is fetched into
    ``.part`` and renamed only when it matched. A mirror is tried for a file
    the primary repository would not serve, and which one served it is
    recorded beside its digest.
    """
    artifacts = list(artifacts)
    if not artifacts:
        return
    expectations = {}
    for item in artifacts:
        if not (item.sha256 and item.size):
            say(f"Asking the publisher about {item.filename}…")
        expectations[item.local_name] = models._resolve(item)
    models._make_room(artifacts, destination, expectations)
    total = max(sum(int(expectations[item.local_name].size or item.approximate_bytes)
                    for item in artifacts), 1)
    done = 0
    for index, item in enumerate(artifacts, start=1):
        expected = expectations[item.local_name]
        size = int(expected.size or item.approximate_bytes)
        say(f"Downloading {index} of {len(artifacts)} — {item.filename} "
            f"({models._bytes_label(size) if size else 'size unknown'})")
        offset = done

        def report(received, offset=offset, size=size):
            tick(base + share * min((offset + min(received, max(size, 1))) / total, 1.0))

        started = time.monotonic()
        used, digest = item, ""
        try:
            digest = models._download(item, destination / item.local_name, report, expected)
        except models.VoiceError as failure:
            fallbacks = _mirror_artifacts(entry, item)
            if not fallbacks:
                raise
            last = failure
            for candidate in fallbacks:
                say(f"{item.filename} could not be fetched from {_repo_of(item.url)} "
                    f"({last}); trying {_repo_of(candidate.url)}…")
                logger.warning("Model Chain: VibeVoice is falling back to %s for %s — %s",
                               _repo_of(candidate.url), item.filename, last)
                try:
                    expected = models._resolve(candidate)
                    digest = models._download(candidate, destination / item.local_name,
                                              report, expected)
                    used = candidate
                    break
                except models.VoiceError as again:
                    last = again
            else:
                raise last
        elapsed = max(time.monotonic() - started, 0.001)
        arrived = (destination / item.local_name).stat().st_size
        logger.info("Model Chain: VibeVoice fetched %s — %s in %.1fs (%.1f MB/s); %s",
                    item.filename, models._bytes_label(arrived), elapsed,
                    arrived / elapsed / (1024 * 1024),
                    f"digest matched {expected.source}" if expected.verified
                    else "digest recorded (the publisher offered none)")
        name = prefix + item.local_name
        record["digests"][name] = digest
        record["sizes"][name] = arrived
        record["verified"][name] = expected.source
        record["repos"][name] = _repo_of(used.url)
        done += size
        tick(base + share * min(done / total, 1.0))


def _adopt_model(entry: Bundle, source: Path, staging: Path, say, tick) -> dict:
    """The model out of a folder somebody filled themselves.

    Config and index by name, the shards the index names, the processor config
    if it is there, and the tokenizer from a subfolder -- fetched from the hub
    when the folder has none, because neither mirror ships it and a person who
    downloaded the mirror's files will not have it either.
    """
    record = _new_record(entry, "local")
    record["source_folder"] = str(source)
    heads = [item for item in entry.artifacts if item.local_name not in entry.optional]
    record_digests(record, _adopt(heads, source, staging, say, entry.label), staging)
    tick(0.05)
    for item in entry.artifacts:
        if item.local_name not in entry.optional:
            continue
        found = models._find(source, [item.filename, item.local_name])
        if found is None:
            say(f"{item.filename} is not in the folder; the defaults will be written.")
            continue
        record_digests(record, _adopt([item], found.parent, staging, say, entry.label),
                       staging)
    shards = _shards_named(_read_index(staging / "model.safetensors.index.json"))
    record["shards"] = list(shards)
    items = _shard_artifacts(entry, shards)
    for index, item in enumerate(items, start=1):
        say(f"Checking shard {index} of {len(items)} — {item.filename}…")
        record_digests(record, _adopt([item], source, staging, say, entry.label), staging)
        tick(0.05 + 0.85 * index / len(items))
    tokenizer_root = staging / entry.tokenizer.dirname
    where = _tokenizer_folder(entry, source)
    if where is not None:
        say(f"Taking the tokenizer from {where}…")
        record_digests(record, _adopt(list(entry.tokenizer.artifacts), where, tokenizer_root,
                                      say, f"{entry.label} tokenizer"),
                       tokenizer_root, prefix=entry.tokenizer.dirname + "/")
    else:
        say("The folder has no tokenizer; fetching Qwen2.5-7B's four tokenizer files…")
        try:
            _fetch(entry, list(entry.tokenizer.artifacts), tokenizer_root, say, tick, 0.9,
                   0.05, record, prefix=entry.tokenizer.dirname + "/")
        except models.VoiceError as exc:
            raise VibeVoiceError(
                f"{entry.label}'s tokenizer could not be fetched ({exc}). Put "
                f"tokenizer.json, tokenizer_config.json, vocab.json and merges.txt from "
                f"huggingface.co/{entry.tokenizer.repo} in {source / entry.tokenizer.dirname} "
                f"and try again. Nothing was installed.") from None
    for name in record["digests"]:
        record["repos"].setdefault(name, "")
        record["verified"].setdefault(name, "your own files")
    tick(0.95)
    return record


def record_digests(record: dict, digests: dict, root: Path, prefix: str = "") -> None:
    for name, digest in digests.items():
        record["digests"][prefix + name] = digest
        try:
            record["sizes"][prefix + name] = (root / name).stat().st_size
        except OSError:
            record["sizes"][prefix + name] = 0
        record["verified"][prefix + name] = "your own files"


def _tokenizer_folder(entry: Bundle, source: Path):
    """Where a tokenizer might be in a folder somebody filled, or ``None``."""
    wanted = [item.filename.casefold() for item in entry.tokenizer.artifacts]

    def complete(folder: Path) -> bool:
        names = {child.name.casefold() for child in folder.iterdir() if child.is_file()}
        return all(name in names for name in wanted)

    candidates = [source / entry.tokenizer.dirname, source / "tokenizer", source]
    candidates += sorted(child for child in source.iterdir() if child.is_dir())
    for folder in candidates:
        if folder.is_dir() and complete(folder):
            return folder
    return None


def _adopt(artifacts, folder: Path, destination: Path, say, what: str) -> dict:
    """Copy named files out of a folder, checking each against what is known.

    Returns ``{local_name: digest}``. Named files only: nothing is guessed from
    an extension, and a file under the right name with the wrong contents is
    refused exactly as a bad download is.
    """
    if not folder.is_dir():
        raise VibeVoiceError(f"{folder} is not a folder this machine can read.")
    destination.mkdir(parents=True, exist_ok=True)
    digests = {}
    for item in artifacts:
        source = models._find(folder, [item.filename, item.local_name])
        if source is None:
            raise VibeVoiceError(f"{item.filename} is not in {folder}. Nothing was installed.")
        say(f"Checking {source.name}…")
        digest = _digest(source)
        if item.sha256 and digest != item.sha256:
            raise VibeVoiceError(
                f"{item.filename} is in that folder, but its contents are not what this "
                f"extension's manifest says they should be. Nothing was installed.")
        if item.size and source.stat().st_size != item.size:
            raise VibeVoiceError(
                f"{item.filename} is {source.stat().st_size} bytes and this extension "
                f"expects {item.size}. Nothing was installed.")
        target = destination / item.local_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digests[item.local_name] = digest
    logger.info("Model Chain: VibeVoice adopted %d file(s) for the %s from %s",
                len(digests), what, folder)
    return digests


def _record_pins(entry: Bundle, record: dict) -> None:
    """Write what arrived into the local overlay, for the files the manifest
    could not hash. Never fatal, and never over a committed digest."""
    if record.get("source") != "hub":
        return
    path = paths.vibevoice_local_pins_path()
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        current = {}
    recorded = dict((current or {}).get("artifacts") or {})
    added = 0
    prefix = entry.tokenizer.dirname + "/"
    for name, digest in record["digests"].items():
        filename = name[len(prefix):] if name.startswith(prefix) else name
        for item in entry.artifacts + entry.shards + entry.tokenizer.artifacts:
            if item.local_name == filename:
                filename = item.filename
                break
        if _committed_digest(entry, filename) or not digest:
            continue
        repo = record["repos"].get(name) or entry.repo
        size = int(record["sizes"].get(name) or 0)
        if size <= 0:
            continue
        recorded[_pin_key(repo, filename)] = {"sha256": digest, "bytes": size}
        added += 1
    if not added:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "artifacts": recorded}, indent=2) + "\n",
                        encoding="utf-8")
    except OSError:
        logger.debug("Model Chain: VibeVoice could not record what it downloaded to %s",
                     path, exc_info=True)
        return
    logger.info("Model Chain: VibeVoice recorded %d artifact digest(s) in %s — the next "
                "install of this model is checked against them", added, path.name)


def _read_index(path: Path) -> dict:
    found = _read_json(path)
    if not found:
        raise VibeVoiceError("model.safetensors.index.json is not readable JSON, so the "
                             "shards cannot be listed. Nothing was installed.")
    return found


def _shards_named(index: dict) -> list:
    """The distinct shard files a safetensors index names, in first-seen order.

    The model's own statement of its weight files, read rather than transcribed.
    A name with a directory in it is refused: the index names files beside
    itself and nothing else.
    """
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise VibeVoiceError("model.safetensors.index.json has no weight_map, so the shards "
                             "cannot be listed. Nothing was installed.")
    found = []
    for name in weights.values():
        text = str(name or "").strip()
        if not text or "/" in text or "\\" in text or text in (".", "..") \
                or not text.endswith(".safetensors"):
            raise VibeVoiceError(f"The index names a shard ({text!r}) that is not a "
                                 f"safetensors file beside it. Nothing was installed.")
        if text not in found:
            found.append(text)
    return found


SAFETENSORS_MIN = 16


def _sanity_check(staging: Path, required, label: str) -> None:
    """Structural validation before anything is promoted.

    Not a hash -- the hash has been checked, or recorded. This asks whether the
    bytes are the *kind of thing* they claim to be, because a proxy that answers
    every request with an HTML page produces a file of exactly the right name.
    """
    for name in required:
        path = staging / name
        try:
            size = path.stat().st_size
        except OSError:
            raise VibeVoiceError(f"{label} is missing {name}. Nothing was installed.") from None
        if size <= 0:
            raise VibeVoiceError(f"{name} arrived as an empty file. Nothing was installed.")
        if name.endswith(".safetensors"):
            with open(path, "rb") as handle:
                head = handle.read(8)
            if len(head) < 8:
                raise VibeVoiceError(f"{name} is too small to be a safetensors file. Nothing "
                                     f"was installed.")
            (declared,) = struct.unpack("<Q", head)
            if declared <= 0 or declared + 8 > size or declared > 64 * 1024 * 1024:
                raise VibeVoiceError(f"{name} does not have a safetensors header. Nothing was "
                                     f"installed.")
        elif name.endswith(".json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise VibeVoiceError(f"{name} is not readable JSON. Nothing was "
                                     f"installed.") from None
        elif size < SAFETENSORS_MIN:
            raise VibeVoiceError(f"{name} is too small to be what it claims. Nothing was "
                                 f"installed.")


def _build_environment(staging: Path, wheels: Path, chosen) -> None:
    """An interpreter of VibeVoice's own, with the verified wheels unpacked into it.

    No package manager anywhere in this path: ``venv`` without pip, then each
    wheel's contents written into site-packages. There is no index to resolve
    against and nothing that can be redirected into somebody's user site.
    """
    import venv

    environment = staging / "env"
    try:
        builder = venv.EnvBuilder(with_pip=False, clear=True, symlinks=(os.name != "nt"))
        builder.create(environment)
    except Exception as exc:
        logger.warning("Model Chain: the isolated VibeVoice runtime could not be created "
                       "using %s — %s: %s", environment, exc.__class__.__name__, exc)
        raise VibeVoiceError(f"The isolated VibeVoice runtime could not be created "
                             f"({exc.__class__.__name__}: {exc}).") from None
    if not _interpreter_in(staging).exists():
        raise VibeVoiceError("The isolated VibeVoice runtime was created without an "
                             "interpreter. Nothing was installed.")
    target = models.site_packages(environment)
    for item in chosen.artifacts:
        wheel = wheels / item.local_name
        if not wheel.is_file():
            raise VibeVoiceError(f"{item.filename} is missing from the staged download. "
                                 f"Nothing was installed.")
        added = models._unpack_wheel(wheel, target)
        logger.info("Model Chain: VibeVoice unpacked %s (%s)", item.filename,
                    ", ".join(added[:6]) or "nothing")
    wanted = (manifest()["runtime"].get("import_name") or "vibevoice", "torch", "transformers")
    for name in wanted:
        if not (target / name).exists():
            raise VibeVoiceError(f"The staged VibeVoice runtime has no {name} in it. Nothing "
                                 f"was installed.")


def _smoke_test(staging: Path) -> dict:
    """Prove the staged runtime imports *before* anything is promoted.

    Two staged runs: the interpreter answers at all, and the worker's own
    ``--selftest`` imports torch, transformers and vibevoice and reports what it
    got. No model is loaded and no card is required here; whether the worker
    can prove its card is the runtime's handshake, on the first render.
    """
    interpreter = _interpreter_in(staging)
    script = worker_script()
    if not script.is_file():
        raise VibeVoiceError("The VibeVoice worker script is missing from this extension, so "
                             "the runtime cannot be checked. Nothing was installed.")
    alive = _run_staged(interpreter, ["-c", "import sys; print(sys.version)"],
                        "check that the isolated VibeVoice interpreter runs", timeout=180)
    if alive.returncode != 0:
        raise VibeVoiceError(
            "The isolated VibeVoice interpreter would not start. This usually means the "
            "Python running this WebUI is an embedded or relocated build that cannot make a "
            "virtual environment. Nothing was installed.")
    result = _run_staged(interpreter, [script, "--selftest"], "VibeVoice self-test",
                         timeout=900)
    report = {}
    for line in reversed((result.stdout or "").strip().splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            report = candidate
            break
    if result.returncode != 0 or report.get("ok") is False:
        reason = report.get("error") or (result.stderr or "").strip().splitlines()[-1:] or [""]
        reason = reason if isinstance(reason, str) else (reason[0] if reason else "")
        logger.warning("Model Chain: the staged VibeVoice runtime could not import — %s",
                       reason or "no reason reported")
        raise VibeVoiceError(
            f"The staged VibeVoice runtime could not import its packages "
            f"({_without_paths(reason) or 'no reason reported'}). Nothing was installed.")
    logger.info("Model Chain: the staged VibeVoice runtime passed its self-test — %s",
                models._quote(json.dumps(report), 400) if report else "no report")
    return report


_PATHLIKE = re.compile(
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]|(?<![:\w\\/])[\\/])"
    r"[^\s'\"()<>]*[\\/]([^\s'\"()<>\\/]+)")


def _without_paths(text) -> str:
    """A staged-runtime message with its filesystem paths cut down to filenames."""
    return _PATHLIKE.sub(lambda found: found.group(1), str(text or "")).strip()


def _run_staged(interpreter: Path, arguments: list, what: str, timeout: float = 300):
    """Run something in the staged runtime, with no credential in its environment."""
    import subprocess

    environ = dict(os.environ)
    for name in models.CREDENTIAL_VARIABLES:
        environ.pop(name, None)
    environ.update(worker_environment(""))
    try:
        return subprocess.run(  # noqa: S603 - a path this module built
            [str(interpreter)] + [str(item) for item in arguments],
            capture_output=True, text=True, env=environ, timeout=timeout,
            cwd=str(paths.extension_root()))
    except Exception as exc:
        logger.warning("Model Chain: could not %s — %s: %s", what, exc.__class__.__name__, exc)
        raise VibeVoiceError(f"The staged VibeVoice runtime could not be started "
                             f"({exc.__class__.__name__}). Nothing was installed.") from None


def _write_local_config(entry: Bundle, root=None) -> None:
    """The processor configuration the worker loads. Section 3 of the contract.

    Upstream's fields when the mirror shipped a ``preprocessor_config.json``
    (kept beside it under its upstream name), the processor's own defaults
    otherwise, and one field of this repository's: ``language_model_pretrained_name``
    naming the local tokenizer directory by absolute path. Without it the
    processor asks the hub for Qwen's tokenizer, which is the one call in the
    load path that could reach the network, and the worker runs offline.

    ``root`` is the staging tree during an install; the path written is always
    the *final* one, because that is where the worker will look.
    """
    final = paths.vibevoice_model_root(entry.identifier)
    where = Path(root) if root is not None else final
    upstream = _read_json(where / paths.VIBEVOICE_UPSTREAM_CONFIG)
    found = dict(upstream) if upstream else dict(entry.local_config or DEFAULT_LOCAL_CONFIG)
    tokenizer = final / entry.tokenizer.dirname
    if not (where / entry.tokenizer.dirname / "tokenizer.json").is_file():
        raise VibeVoiceError(f"{entry.label}'s tokenizer is not in place, so its processor "
                             f"configuration was not written. Install the model again.")
    if "qwen" not in str(tokenizer).casefold():
        raise VibeVoiceError("The tokenizer directory's name must contain 'qwen' for the "
                             "processor to pick its tokenizer class; the configuration was "
                             "not written.")
    found["language_model_pretrained_name"] = str(tokenizer)
    _write_json(where / paths.VIBEVOICE_LOCAL_CONFIG, found)


def uninstall() -> Status:
    """Remove the VibeVoice runtime and model. Settings and the calibration are kept.

    Kept because both describe this machine rather than an installation: a
    reinstalled model renders with the same card and needs the same VRAM.
    """
    _stop_runtime("VibeVoice is being removed")
    for root in (paths.vibevoice_runtime_root(), paths.vibevoice_models_root(),
                 paths.vibevoice_staging_root()):
        if paths.vibevoice_inside(root):
            shutil.rmtree(root, ignore_errors=True)
    logger.info("Model Chain: VibeVoice's runtime and model were removed; its settings and "
                "calibration were kept")
    return status()


def _stop_runtime(reason: str) -> None:
    """Stop the worker before its runtime or model is replaced. Never raises.

    The runtime module is part A's; imported lazily so this module stands
    without it, and a build without it simply has nothing to stop.
    """
    try:
        import mc_voice_vibevoice_runtime as runtime
    except Exception:
        return
    try:
        runtime.stop("", reason)
    except TypeError:
        try:
            runtime.stop(reason=reason)
        except Exception:
            logger.debug("Model Chain: could not stop VibeVoice before %s", reason,
                         exc_info=True)
    except Exception:
        logger.debug("Model Chain: could not stop VibeVoice before %s", reason, exc_info=True)


def progress() -> dict:
    """What the install is doing, for the page. One flat record for this guest."""
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
