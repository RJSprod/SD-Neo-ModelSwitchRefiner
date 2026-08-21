"""The managed backbone catalogue: a short list of models, fetched safely.

LLM Studio has always been able to run any GGUF on the machine, and that is the
right floor. It is a poor *first* experience: somebody who has just installed
the extension has no GGUF, no idea which of the several thousand on Hugging Face
this application was written against, and no way to find out except by
downloading a few and comparing. So this module adds the other half -- a handful
of backbones chosen in advance, each with the settings it should run at already
decided, downloaded and verified by the extension rather than by hand.

What it is not is a downloader. There is no box to paste a URL into and no way
to reach one: the only thing this will fetch is an artifact named by
``prompt_master/models/managed-models.json``, which is checked in, reviewed like
any other source file, and carries a SHA-256 for every byte it names. That file
is the trust root. Nothing here reads a model card at run time to discover what
to download or how to run it, because a model card is somebody else's mutable
web page and this decides what a user's machine executes against.

Two lifecycles, deliberately not merged
---------------------------------------
The llama.cpp runtime is a *program this extension starts* and is installed by
``mc_llm_setup``. A GGUF is *data the program reads*, and is installed here.
They are versioned separately, break separately and are replaced separately, and
the one thing that would guarantee they could not be is a single install
function that does both.

The transaction
---------------
A download is not finished when the bytes arrive; it is finished when a
directory that did not exist becomes one that does, in a single rename. Until
that rename, everything lives under ``managed/.downloads/<id>/`` and the
installed tree does not know it exists. That is what makes the interesting
failures boring: a cancelled download is a ``.part`` file and a sidecar, a
corrupted one is a hash mismatch and a deleted ``.part``, a crash halfway is a
staging directory nothing reads. In every one of them the model that was
selected a minute ago is still selected, still on disk and still startable,
because *nothing outside the staging directory was touched at all.*

Applying is the other transaction, and it has a stricter rule: exactly one
llama-server may hold weights at a time. Downloading eight gigabytes while a
model is loaded is fine and does not disturb it; swapping which model is
resident stops the old server, waits for it to be gone, starts the new one, and
puts the old state back if the new one will not answer.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mc_llm_paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_MB = 1024**2
_GB = 1024**3

MODEL_FILENAME = "model.gguf"
MMPROJ_FILENAME = "mmproj.gguf"
INSTALLED_FILENAME = "installed.json"
SIDECAR_FILENAME = "download-state.json"
"""The four names a managed bundle is allowed to contain.

Fixed rather than taken from the registry, and that is a security decision as
much as a tidiness one: a publisher's filename is a string from a JSON file,
and the set of strings that are safe to join onto a path is much smaller than
the set that look like it. The publisher's name is *recorded* in
``installed.json`` -- so a bundle can always say what it really is -- and is
never what anything here opens.
"""

BUNDLE_SUFFIX = ".previous"
"""Where a bundle being replaced waits until the replacement has succeeded."""

SAFETY_MARGIN_BYTES = 512 * _MB
SAFETY_MARGIN_FRACTION = 0.05
"""Free disk a download insists on beyond the bytes it is about to write.

The larger of the two. A fixed floor alone is meaningless against a 17 GB
model, and a percentage alone is meaningless against a 700 MB projector.
Neither number is precise; what they are for is refusing at the start, with a
sentence, rather than at 94% with an OSError.
"""

SWITCH_LOCK_TIMEOUT = 20.0
"""How long a switch waits for the GPU before saying something else has it.

Short because the answer to "an image generation is running" is a sentence
rather than a queue: a user who pressed Use is watching the panel, and twenty
seconds of a frozen button followed by success is a worse experience than an
immediate "Stable Diffusion is using the GPU — try again when it has finished."
"""

RESIDENT_STOP_TIMEOUT = 30.0
"""Seconds a switch waits for the previous llama-server to actually be gone.

Not a formality. The whole point of a switch is that two models never hold the
card at once, and ``stop()`` returning is a statement about a handle rather
than about a process: on Windows the server can still be unmapping a 17 GB file
for several seconds after it has been asked to exit.
"""

SMOKE_TOKENS = 8
SMOKE_TIMEOUT = 120.0
"""The health check after a switch: one very small completion.

A server that reaches ``/health`` has loaded a file. It has not necessarily
loaded a *chat model*, applied a template, or been given a projector it can
use, and every one of those fails at the first real request instead -- which,
without this, would be somebody's actual generation, several minutes later,
with the previous model long since discarded. Eight tokens is enough to prove
the template runs and costs a fraction of a second.
"""


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ManagedError(RuntimeError):
    """Something a user can act on, phrased as a sentence for the panel.

    The same contract ``mc_llm_setup.SetupError`` keeps: these strings are
    printed as-is, so they say what happened *and* what state the installation
    was left in -- "the model you had is still selected" is the half of the
    message that stops a failed download from reading like a broken install.
    """


class Cancelled(ManagedError):
    """The user pressed Cancel. Not a failure, and never reported as one."""


class Busy(ManagedError):
    """A generation is running, so the resident model must not be swapped."""


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


REGISTRY_PATH = Path(__file__).resolve().parent / "prompt_master" / "models" / "managed-models.json"

_ID = re.compile(r"^[a-z0-9]+(?:[-.][a-z0-9]+)*$")
"""What a model id may look like. It becomes a directory name, so this is the
whole of the traversal defence: no separators, no dots on their own, no spaces,
nothing that means anything to a path parser."""

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_COMMIT = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,63}$")

_SIZE = re.compile(r"^~?\s*([0-9]+(?:\.[0-9]+)?)\s*(TB|GB|MB|KB|B)$", re.IGNORECASE)
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": _MB, "GB": _GB, "TB": 1024 * _GB}


@dataclass(frozen=True)
class Artifact:
    """One file in a bundle: what to fetch, and what it must turn out to be."""

    filename: str
    """The publisher's own name for it. Used to build the URL and to record
    provenance, and never joined onto a local path -- see :data:`MODEL_FILENAME`."""
    sha256: str
    local_name: str
    size: int | None = None
    display_size: str = ""

    @property
    def approximate_bytes(self) -> int:
        """The size to plan with: exact when the registry pins one, else the
        catalogue's own display figure parsed back into a number.

        Used for the disk-space check and for the progress fraction, and for
        nothing else -- an approximation may decide how full a bar looks and
        may never decide whether a file is the right file. That is
        :attr:`sha256`'s job, and it is exact.
        """
        if self.size is not None:
            return int(self.size)
        found = _SIZE.match(str(self.display_size or "").strip())
        if not found:
            return 0
        return int(float(found.group(1)) * _SIZE_UNITS[found.group(2).upper()])

    def url(self, repo_id: str, revision: str) -> str:
        return f"https://huggingface.co/{repo_id}/resolve/{revision}/{self.filename}"


@dataclass(frozen=True)
class ManagedModel:
    """One catalogue entry: a backbone, its artifacts, and how to describe it."""

    identifier: str
    label: str
    role: str
    group: str
    family: str
    profile_id: str
    repo_id: str
    revision: str
    source_url: str
    license_url: str
    model: Artifact
    projector: Artifact | None = None
    multimodal: bool = True
    registry_version: str = ""

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        return (self.model,) if self.projector is None else (self.model, self.projector)

    @property
    def pinned(self) -> bool:
        """Whether ``revision`` is an immutable commit rather than a branch.

        A branch still cannot install the wrong thing -- every byte is checked
        against the SHA-256 in this repository, so a publisher who re-uploads
        gets a refusal rather than a silent substitution. What it costs is the
        *quality* of that refusal: pinned, a moved branch is invisible; on a
        branch, it is a hash mismatch and a sentence asking for an extension
        update. The panel says which it is; ``scripts/pin_managed_models.py``
        turns the second into the first.
        """
        return bool(_COMMIT.match(self.revision))

    @property
    def total_bytes(self) -> int:
        return sum(artifact.approximate_bytes for artifact in self.artifacts)

    def describe(self) -> str:
        """The one line the catalogue shows: role, size, and the family.

        Everything a choice between six models actually turns on, and nothing
        else. No temperature, no top-k, no cache type -- those are decided in
        ``managed_profiles`` and showing them here would turn a choice of
        backbone back into the settings screen this replaces.
        """
        parts = [self.role, self._sizes()]
        if self.family:
            parts.append(self.family)
        return " · ".join(part for part in parts if part)

    def _sizes(self) -> str:
        main = self.model.display_size or _bytes_label(self.model.approximate_bytes)
        if self.projector is None:
            return f"{main} · text only"
        vision = self.projector.display_size or _bytes_label(self.projector.approximate_bytes)
        return f"{main} + {vision} vision"


_registry_cache: tuple[float, dict[str, ManagedModel]] | None = None
_registry_lock = threading.RLock()


def registry(refresh: bool = False) -> dict[str, ManagedModel]:
    """Every catalogue entry, validated, keyed by id and in file order.

    Cached against the registry file's modification time rather than for a
    duration: it is a file in the extension, so it changes when the extension
    is updated and not otherwise, and the panel asks for it several times per
    render.
    """
    global _registry_cache

    with _registry_lock:
        try:
            stamp = REGISTRY_PATH.stat().st_mtime
        except OSError as exc:
            raise ManagedError(
                f"The managed model registry is missing ({REGISTRY_PATH}). This extension's "
                f"files are incomplete — reinstall or update it."
            ) from exc
        if not refresh and _registry_cache is not None and _registry_cache[0] == stamp:
            return dict(_registry_cache[1])
        loaded = _load_registry(REGISTRY_PATH)
        _registry_cache = (stamp, loaded)
        return dict(loaded)


def catalogue() -> list[ManagedModel]:
    """The registry as a list, in the order the file lists them.

    That order is editorial -- recommended first, the large baseline last -- and
    is the order the panel offers them in, so it is preserved rather than
    sorted.
    """
    try:
        return list(registry().values())
    except ManagedError:
        logger.warning("Model Chain: the managed model registry could not be read",
                       exc_info=True)
        return []


def entry(identifier: str) -> ManagedModel:
    """The catalogue entry ``identifier`` names.

    The only way to reach a download, and therefore the only place an id is
    turned into a URL. An id that is not in the checked-in file is refused here
    and no network access happens at all.
    """
    found = registry().get(str(identifier or "").strip())
    if found is None:
        raise ManagedError(f"{identifier!r} is not a model in the managed catalogue.")
    return found


def _load_registry(path: Path) -> dict[str, ManagedModel]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManagedError(f"The managed model registry cannot be read ({exc}).") from exc
    if not isinstance(document, dict):
        raise ManagedError("The managed model registry is not a JSON object.")

    version = str(document.get("registry_version") or document.get("version") or "")
    found: dict[str, ManagedModel] = {}
    for raw in document.get("models") or []:
        model = _read_entry(raw, version)
        if model.identifier in found:
            raise ManagedError(f"The managed model registry lists {model.identifier} twice.")
        found[model.identifier] = model
    if not found:
        raise ManagedError("The managed model registry lists no models.")
    return found


def _read_entry(raw, registry_version: str) -> ManagedModel:
    """One registry row, refused unless every field is what it claims to be.

    Every check here is against a file inside the extension, so none of them
    can be triggered by a user -- which is exactly why they are worth having.
    A registry row is the input that decides what gets downloaded and where it
    is written, and the cost of validating one at load time is nothing next to
    the cost of finding out at write time that a filename contained ``..``.
    """
    if not isinstance(raw, dict):
        raise ManagedError("A managed model entry is not a JSON object.")

    identifier = str(raw.get("id") or "").strip()
    if not _ID.match(identifier):
        raise ManagedError(f"{identifier!r} is not a usable managed model id.")

    repo_id = str(raw.get("repo_id") or "").strip()
    if not _REPO_ID.match(repo_id):
        raise ManagedError(f"{identifier}: {repo_id!r} is not a Hugging Face repository id.")

    revision = str(raw.get("revision") or "").strip()
    if not _REVISION.match(revision) or "latest" in revision.casefold():
        raise ManagedError(f"{identifier}: {revision!r} is not a usable revision.")

    for key in ("source_url", "license_url"):
        url = str(raw.get(key) or "")
        if url and not url.startswith("https://"):
            raise ManagedError(f"{identifier}: {key} must be an HTTPS URL.")

    model = _read_artifact(identifier, raw.get("model"), MODEL_FILENAME, required=True)
    projector = _read_artifact(identifier, raw.get("projector"), MMPROJ_FILENAME,
                               required=False)
    if bool(raw.get("multimodal", True)) and projector is None:
        raise ManagedError(f"{identifier}: is marked multimodal but names no projector.")

    return ManagedModel(
        identifier=identifier,
        label=str(raw.get("label") or identifier),
        role=str(raw.get("role") or ""),
        group=str(raw.get("group") or ""),
        family=str(raw.get("family") or ""),
        profile_id=str(raw.get("profile") or identifier),
        repo_id=repo_id,
        revision=revision,
        source_url=str(raw.get("source_url") or ""),
        license_url=str(raw.get("license_url") or raw.get("source_url") or ""),
        model=model,
        projector=projector,
        multimodal=bool(raw.get("multimodal", True)),
        registry_version=registry_version,
    )


def _read_artifact(identifier: str, raw, local_name: str, required: bool) -> Artifact | None:
    if raw in (None, {}):
        if required:
            raise ManagedError(f"{identifier}: names no model file.")
        return None
    if not isinstance(raw, dict):
        raise ManagedError(f"{identifier}: an artifact entry is not a JSON object.")

    filename = str(raw.get("filename") or "").strip()
    if not _FILENAME.match(filename):
        raise ManagedError(f"{identifier}: {filename!r} is not a usable GGUF filename.")

    sha256 = str(raw.get("sha256") or "").strip()
    if not _SHA256.match(sha256):
        raise ManagedError(f"{identifier}: {filename} has no complete SHA-256.")

    size = raw.get("bytes")
    if size is not None:
        try:
            size = int(size)
        except (TypeError, ValueError):
            raise ManagedError(f"{identifier}: {filename} has an unreadable byte count.") from None
        if size <= 0:
            raise ManagedError(f"{identifier}: {filename} has an impossible byte count.")

    return Artifact(filename=filename, sha256=sha256.casefold(), local_name=local_name,
                    size=size, display_size=str(raw.get("display_size") or ""))


# --------------------------------------------------------------------------- #
# What is on disk
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Installed:
    """A verified bundle on disk, as its own ``installed.json`` describes it."""

    identifier: str
    root: Path
    model: Path
    mmproj: Path | None
    registry_version: str = ""
    revision: str = ""
    profile_id: str = ""
    profile_version: str = ""
    hashes: dict = field(default_factory=dict)
    installed_at: float = 0.0

    @property
    def sees(self) -> bool:
        return self.mmproj is not None

    def matches(self, model: ManagedModel) -> bool:
        """Whether what is on disk is what the registry now describes.

        Compared by hash, because that is the only field that means anything: a
        registry update that renames a file or moves to a pinned revision has
        not changed the bytes, and re-downloading eight gigabytes to obtain the
        same eight gigabytes would be a poor way to celebrate it. A *changed*
        hash is a different model wearing the same id, and the panel says so.
        """
        wanted = {artifact.local_name: artifact.sha256 for artifact in model.artifacts}
        return all(str(self.hashes.get(name, "")).casefold() == sha
                   for name, sha in wanted.items())


def bundle_root(identifier: str) -> Path:
    """Where ``identifier``'s bundle lives, checked to be under the managed root.

    The check is not theatre even though the id has already been matched
    against :data:`_ID`: this is the function every path in the module is built
    from, and a containment rule enforced at the one place paths are made is a
    rule that cannot be forgotten at the twenty places they are used.
    """
    if not _ID.match(str(identifier or "").strip()):
        raise ManagedError(f"{identifier!r} is not a usable managed model id.")
    root = mc_llm_paths.managed_models_root()
    candidate = (root / identifier).resolve()
    if candidate.parent != root.resolve():
        raise ManagedError(f"{identifier!r} does not resolve inside the managed models folder.")
    return candidate


def staging_root(identifier: str) -> Path:
    """Where ``identifier`` is downloaded to before it is promoted."""
    bundle_root(identifier)  # the same containment check, for the same reason
    return mc_llm_paths.managed_staging_root() / identifier


def installed(identifier: str) -> Installed | None:
    """The bundle on disk for ``identifier``, or ``None`` when there is not one.

    Cheap on purpose: it reads a small JSON file and stats two large ones. It
    does *not* re-hash them, because this is called every time the Setup panel
    is drawn and a panel that hashed twenty gigabytes to render itself would
    never finish. The hashes were checked when the bundle was promoted, and
    nothing but the user can have changed the files since.
    """
    try:
        root = bundle_root(identifier)
    except ManagedError:
        return None
    try:
        document = json.loads((root / INSTALLED_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None

    model = root / MODEL_FILENAME
    if not model.is_file():
        return None
    mmproj = root / MMPROJ_FILENAME
    artifacts = document.get("artifacts") or {}
    hashes = {name: str((artifacts.get(key) or {}).get("sha256") or "").casefold()
              for key, name in (("model", MODEL_FILENAME), ("projector", MMPROJ_FILENAME))}

    return Installed(
        identifier=str(document.get("model_id") or identifier),
        root=root,
        model=model,
        mmproj=mmproj if mmproj.is_file() else None,
        registry_version=str(document.get("registry_version") or ""),
        revision=str(document.get("revision") or ""),
        profile_id=str(document.get("profile") or ""),
        profile_version=str(document.get("profile_version") or ""),
        hashes={name: value for name, value in hashes.items() if value},
        installed_at=float(document.get("installed_at") or 0.0),
    )


def identify_path(path) -> str | None:
    """The managed model a path belongs to, if it is inside a managed bundle.

    What makes the catalogue survive contact with the manual chooser. A managed
    bundle lives under the LLM data root, which is also where the model chooser
    scans by default, so a downloaded backbone appears in that list like any
    other GGUF -- and picking it there used to be indistinguishable from
    picking a stranger's file, which would have silently dropped the hidden
    profile that is the whole point of having downloaded it. Recognising the
    path puts it back.
    """
    if not path:
        return None
    try:
        resolved = Path(path).expanduser().resolve()
        root = mc_llm_paths.managed_models_root().resolve()
    except (OSError, ValueError):
        return None
    if root not in resolved.parents:
        return None
    try:
        identifier = resolved.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return None
    return identifier if installed(identifier) is not None else None


# --------------------------------------------------------------------------- #
# Selection state (section 6 of the design intent)
# --------------------------------------------------------------------------- #


SOURCE_MANAGED = "managed"
SOURCE_MANUAL = "manual"

STATE_KEY_SOURCE = "source"
STATE_KEY_ID = "managed_model_id"
STATE_KEY_PROFILE = "managed_profile"
STATE_KEY_PROFILE_VERSION = "managed_profile_version"


@dataclass(frozen=True)
class Selection:
    """Which backbone every LLM mode resolves, and where its settings come from.

    Three states rather than two, and the third is the one worth naming:
    ``managed`` with an id nothing on disk answers to. That is what a user who
    deleted a bundle by hand has, and the honest thing to do with it is to fall
    back to manual behaviour -- run the recorded file with the installation's
    own settings -- rather than to apply a profile to weights that may no
    longer be there.
    """

    source: str = SOURCE_MANUAL
    identifier: str = ""
    profile_id: str = ""
    profile_version: str = ""

    @property
    def managed(self) -> bool:
        return self.source == SOURCE_MANAGED and bool(self.identifier)


def selection(state: dict | None = None) -> Selection:
    """What the state file says is selected. Never raises."""
    if state is None:
        state = _read_state()
    source = str(state.get(STATE_KEY_SOURCE) or SOURCE_MANUAL).strip().casefold()
    identifier = str(state.get(STATE_KEY_ID) or "").strip()
    if source != SOURCE_MANAGED or not identifier:
        return Selection()
    return Selection(SOURCE_MANAGED, identifier,
                     str(state.get(STATE_KEY_PROFILE) or identifier),
                     str(state.get(STATE_KEY_PROFILE_VERSION) or ""))


def active_profile(chosen: Selection | None = None):
    """The :class:`ManagedProfile` in force, or ``None`` on a manual install.

    ``None`` is not a failure state: it is what "the user chose their own GGUF"
    looks like, and it is what makes the Settings page's context size and cache
    types authoritative again. A managed selection whose profile this build has
    never heard of also lands here -- see :func:`managed_profiles.profile`.
    """
    from prompt_master.models import managed_profiles

    picked = selection() if chosen is None else chosen
    if not picked.managed:
        return None
    return managed_profiles.profile(picked.profile_id or picked.identifier)


def follow_path(model_path) -> Selection:
    """Set the source to match a model somebody chose by hand. Returns it.

    Called after the manual boxes and the model sheet's chooser have recorded a
    path, and it is the whole of what keeps those two routes and this one
    telling the same story.

    Both directions matter. Picking a stranger's GGUF while a managed backbone
    was selected has to clear the selection, or the profile written for one
    model would be applied to another -- an 8192 context and a q8_0 cache
    silently imposed on a file that was never measured for them. And picking a
    *managed* bundle's own weights out of the ordinary chooser -- which happens
    the moment one is downloaded, because the managed root is under the folder
    that chooser scans -- has to restore the selection, or a curated backbone
    would quietly run as an anonymous GGUF with the previous model's settings.
    """
    identifier = identify_path(model_path)
    state = _read_state()
    current = selection(state)

    if identifier is not None:
        try:
            model = entry(identifier)
        except ManagedError:
            model = None
        if model is not None:
            if current.managed and current.identifier == identifier:
                return current
            _write_state({**state, STATE_KEY_SOURCE: SOURCE_MANAGED,
                          STATE_KEY_ID: model.identifier,
                          STATE_KEY_PROFILE: model.profile_id,
                          STATE_KEY_PROFILE_VERSION: _profile_version()})
            logger.info("Model Chain: the chosen GGUF is the managed backbone %s; its profile "
                        "applies", model.identifier)
            return selection()

    select_manual()
    return selection()


def select_manual() -> None:
    """Record that the user is running their own GGUF again.

    Called by the manual model boxes rather than by anything here. Without it,
    a state file could say ``managed`` while naming a hand-picked file, and the
    hidden profile for a model that is no longer loaded would be applied to one
    that is.
    """
    state = _read_state()
    if str(state.get(STATE_KEY_SOURCE) or SOURCE_MANUAL) == SOURCE_MANUAL:
        return
    _write_state({**state, STATE_KEY_SOURCE: SOURCE_MANUAL, STATE_KEY_ID: "",
                  STATE_KEY_PROFILE: "", STATE_KEY_PROFILE_VERSION: ""})


def _read_state() -> dict:
    from prompt_master.core.config import read_json

    try:
        state = read_json(mc_llm_paths.app_paths().state_file)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _write_state(state: dict) -> None:
    from prompt_master.core.config import atomic_write_json

    paths = mc_llm_paths.app_paths()
    paths.data.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.state_file, state)


# --------------------------------------------------------------------------- #
# Status, for the panel
# --------------------------------------------------------------------------- #


NOT_DOWNLOADED = "Not downloaded"
INSTALLED = "Installed"
ACTIVE = "Active"
SUPERSEDED = "Installed — older revision"
PARTIAL = "Download interrupted"


@dataclass(frozen=True)
class Status:
    """One catalogue entry, and everything the panel needs to draw it."""

    model: ManagedModel
    state: str
    bundle: Installed | None = None
    detail: str = ""

    @property
    def ready(self) -> bool:
        """Downloaded and current: the button says Use rather than Download."""
        return self.state in (INSTALLED, ACTIVE)

    @property
    def active(self) -> bool:
        return self.state == ACTIVE


def status(model: ManagedModel, chosen: Selection | None = None) -> Status:
    """What state ``model`` is in. Never raises -- the panel renders this."""
    picked = selection() if chosen is None else chosen
    try:
        bundle = installed(model.identifier)
    except Exception:
        logger.debug("Model Chain: could not read the managed bundle for %s",
                     model.identifier, exc_info=True)
        bundle = None

    if bundle is None:
        if _staged_bytes(model) > 0:
            return Status(model, PARTIAL, None,
                          "Part of this download is already on disk; it will carry on from "
                          "there.")
        return Status(model, NOT_DOWNLOADED, None, _pin_note(model))
    if not bundle.matches(model):
        return Status(model, SUPERSEDED, bundle,
                      "The catalogue now names different files for this backbone. Download "
                      "it again to move to them; what is installed keeps working until you "
                      "do.")
    if picked.managed and picked.identifier == model.identifier:
        return Status(model, ACTIVE, bundle, "Every LLM Studio mode is using this backbone.")
    return Status(model, INSTALLED, bundle, "Downloaded and verified. Ready to use.")


def statuses() -> list[Status]:
    """Every catalogue entry's state, read once against one selection."""
    picked = selection()
    return [status(model, picked) for model in catalogue()]


def _pin_note(model: ManagedModel) -> str:
    if model.pinned:
        return ""
    return ("This entry resolves its files from the publisher's branch rather than a pinned "
            "commit. Every byte is still checked against the hash in this extension, so a "
            "changed file is refused rather than installed.")


def _staged_bytes(model: ManagedModel) -> int:
    """How much of ``model`` is already sitting in the staging directory."""
    try:
        staging = staging_root(model.identifier)
    except ManagedError:
        return 0
    total = 0
    for artifact in model.artifacts:
        for name in (artifact.local_name, artifact.local_name + ".part"):
            try:
                total += (staging / name).stat().st_size
            except OSError:
                continue
    return total


# --------------------------------------------------------------------------- #
# The download transaction (section 7 of the design intent)
# --------------------------------------------------------------------------- #


def download(identifier: str, on_status=None, on_progress=None,
             cancel: threading.Event | None = None) -> Installed:
    """Fetch, verify and install one managed bundle. Changes no selection.

    Seven steps, and the order is the whole design: resolve the entry from the
    checked-in registry, check there is room, stage every file under
    ``.downloads``, verify each one against its committed SHA-256 and its GGUF
    header, write the manifest, and only then rename the finished directory
    into place. Applying it is a separate call and a separate press --
    a download that succeeds while a model is loaded must not disturb the model
    that is loaded.

    A bundle already installed and still matching the catalogue is returned
    without a single HTTP request. That check is here rather than only in the
    panel because it is a property of the operation and not of the button:
    nothing should be able to spend eight gigabytes of somebody's connection
    re-fetching a file that is already on their disk, whichever route asked
    for it.
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)

    model = entry(identifier)                                          # STEP 1
    already = installed(model.identifier)
    if already is not None and already.matches(model):
        say(f"{model.label} is already downloaded.")
        tick(1.0)
        return already

    staging = staging_root(model.identifier)
    target = bundle_root(model.identifier)

    _preflight(model, staging)                                         # STEP 2
    _prepare_staging(model, staging)                                   # STEPS 3, 4

    total = max(model.total_bytes, 1)
    done_before = 0
    for artifact in model.artifacts:                                   # STEP 3
        _check_cancelled(cancel)
        say(f"Downloading {artifact.filename}…")
        share = artifact.approximate_bytes or total
        base = done_before

        def report(done, _expected, base=base, share=share):
            _check_cancelled(cancel)
            tick(min((base + min(done, share)) / total, 1.0))

        _fetch(model, artifact, staging / artifact.local_name, report, say)
        done_before += share
        tick(min(done_before / total, 1.0))

    for artifact in model.artifacts:                                   # STEP 5
        say(f"Verifying {artifact.filename}…")
        _check_gguf(staging / artifact.local_name, artifact)

    say("Installing…")
    _write_manifest(model, staging)                                    # STEP 6
    _promote(staging, target)

    bundle = installed(model.identifier)
    if bundle is None:
        raise ManagedError(
            f"{model.label} was downloaded and verified, but the installed bundle cannot be "
            f"read back from {target}. Nothing else was changed."
        )
    tick(1.0)
    logger.info("Model Chain: managed backbone %s installed at %s", model.identifier, target)
    return bundle


def _preflight(model: ManagedModel, staging: Path) -> None:
    """STEP 2 — is there room? Asked before anything is created or stopped.

    Against the *remaining* bytes rather than the whole bundle, so resuming a
    download that is nine tenths done is not refused by a disk that has room
    for the last tenth. The margin is deliberately generous: llama.cpp mmaps
    these files and a filesystem with nothing left is a machine with bigger
    problems than a failed download.
    """
    root = mc_llm_paths.managed_models_root()
    remaining = 0
    for artifact in model.artifacts:
        have = 0
        for name in (artifact.local_name, artifact.local_name + ".part"):
            try:
                have = max(have, (staging / name).stat().st_size)
            except OSError:
                continue
        remaining += max(artifact.approximate_bytes - have, 0)
    if remaining <= 0:
        return

    margin = max(SAFETY_MARGIN_BYTES, int(remaining * SAFETY_MARGIN_FRACTION))
    probe = root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        # A drive that will not answer is not a reason to refuse: the download
        # itself will fail with a real error if there is genuinely no room, and
        # refusing on a failed stat would break the feature on network shares.
        logger.debug("Model Chain: could not measure free space on %s", probe, exc_info=True)
        return
    if free >= remaining + margin:
        return
    raise ManagedError(
        f"{model.label} needs about {_bytes_label(remaining)} plus "
        f"{_bytes_label(margin)} of headroom, and {probe} has {_bytes_label(free)} free. "
        f"Nothing was downloaded and the model you are using is unchanged."
    )


def _prepare_staging(model: ManagedModel, staging: Path) -> None:
    """STEPS 3 and 4 — the staging directory, and the sidecar that guards it.

    The sidecar is what makes resuming safe rather than merely possible. A
    ``.part`` file is a pile of bytes with no memory of what it was going to
    be: if the registry has moved to a different quantisation since it was
    written, appending to it produces a file that is exactly the right length
    and is not any model at all. So the expectations are written down first,
    and a staging directory whose sidecar does not match the entry being
    downloaded now is discarded rather than continued.
    """
    wanted = {
        "registry_version": model.registry_version,
        "revision": model.revision,
        "repo_id": model.repo_id,
        "artifacts": {artifact.local_name: {"filename": artifact.filename,
                                            "sha256": artifact.sha256,
                                            "bytes": artifact.size}
                      for artifact in model.artifacts},
    }
    sidecar = staging / SIDECAR_FILENAME
    try:
        current = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = None
    if current is not None and current.get("expected") != wanted:
        logger.info("Model Chain: discarding a staged download of %s — the catalogue entry "
                    "has changed since it was started", model.identifier)
        shutil.rmtree(staging, ignore_errors=True)

    try:
        staging.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ManagedError(f"The managed models folder cannot be created ({exc}).") from exc
    _write_json(sidecar, {"expected": wanted, "started_at": time.time()})


def _fetch(model: ManagedModel, artifact: Artifact, destination: Path, report, say) -> None:
    """One artifact, resumed where possible and verified before it is kept.

    The transfer itself is the vendored provisioning downloader, unchanged: it
    already does HTTP Range resumption, the retry budget that only counts
    attempts which moved no bytes, the 416-means-start-again rule, and the
    verify-then-rename that keeps a failed hash from ever becoming a file with
    a real name. Re-implementing that here to add a catalogue would have been
    two downloaders to keep correct instead of one.
    """
    from prompt_master.provisioning.downloader import download as fetch
    from prompt_master.provisioning.manifest import Component

    component = Component(
        component_id=f"{model.identifier}:{artifact.local_name}",
        url=artifact.url(model.repo_id, model.revision),
        destination=artifact.local_name,
        size=artifact.size,
        sha256=artifact.sha256,
        version=model.revision,
    )
    try:
        component.validate()
    except ValueError as exc:
        raise ManagedError(f"{model.label}: {exc}") from None

    try:
        fetch(component, destination, report, notice=lambda text: say(f"{artifact.filename}: {text}"))
    except Cancelled:
        raise
    except ValueError as exc:
        # The downloader's own words for a size or hash failure. Restated,
        # because "SHA-256 mismatch for model.gguf" is a true sentence that
        # tells a user nothing about what to do, and this is the one failure
        # that means the extension itself is out of date.
        raise ManagedError(
            f"{artifact.filename} downloaded, but its contents are not what this extension's "
            f"catalogue says they should be ({exc}). The published file has changed since "
            f"this version of the extension was released — update the extension rather than "
            f"retrying. Nothing was installed and your current model is unchanged."
        ) from None
    except OSError as exc:
        raise ManagedError(f"{artifact.filename} could not be written ({exc}). Nothing was "
                           f"installed and your current model is unchanged.") from None
    except Exception as exc:
        raise ManagedError(
            f"{artifact.filename} could not be downloaded ({exc}). What has arrived is kept, "
            f"so pressing Download again carries on from there; your current model is "
            f"unchanged."
        ) from None


def _check_gguf(path: Path, artifact: Artifact) -> None:
    """STEP 5 — the part a hash cannot tell you.

    A file can match its SHA-256 perfectly and still be the wrong *kind* of
    thing: a publisher who uploads an ONNX export under a ``.gguf`` name, or a
    projector where a model should be, produces a bundle that verifies and then
    fails at llama-server startup with a message about tensors. Reading the
    header is a few kilobytes and turns that into a refusal here, before
    anything is promoted.
    """
    import mc_gguf

    try:
        mc_gguf.read(path)
    except Exception as exc:
        raise ManagedError(
            f"{artifact.filename} is not a GGUF file this extension can read ({exc}). It has "
            f"not been installed and your current model is unchanged."
        ) from None


def _write_manifest(model: ManagedModel, staging: Path) -> None:
    """STEP 6a — what this bundle is, written beside it before it becomes one.

    Inside the staging directory rather than after the rename, so the promotion
    is one operation with nothing to finish afterwards: a bundle that exists is
    a bundle that can describe itself, always, including after a power cut
    between the two.
    """
    document = {
        "schema": 1,
        "model_id": model.identifier,
        "label": model.label,
        "registry_version": model.registry_version,
        "source_url": model.source_url,
        "repo_id": model.repo_id,
        "revision": model.revision,
        "profile": model.profile_id,
        "profile_version": _profile_version(),
        "artifacts": {
            key: {"filename": artifact.filename, "stored_as": artifact.local_name,
                  "sha256": artifact.sha256,
                  "bytes": _size_of(staging / artifact.local_name)}
            for key, artifact in (("model", model.model), ("projector", model.projector))
            if artifact is not None
        },
        "installed_at": time.time(),
    }
    _write_json(staging / INSTALLED_FILENAME, document)
    # The sidecar described a download. This directory is about to stop being
    # one, and leaving the file behind would put a stale set of expectations
    # inside an installed bundle for the next resume to find.
    (staging / SIDECAR_FILENAME).unlink(missing_ok=True)


def _promote(staging: Path, target: Path) -> None:
    """STEP 6b — the verified staging directory becomes the installed one.

    One rename, so there is no moment at which ``models/managed/<id>`` holds
    half a bundle. A previous bundle is moved aside rather than deleted and put
    back if the rename fails, which is the pattern ``mc_llm_setup`` established
    for replacing the runtime and for the same reason: ``rmtree`` walks a
    directory file by file, so a bundle whose weights are held open by a server
    is not left alone by a failed delete, it is left in pieces -- and an
    installation that had a working model before the attempt would have none
    after it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = target.with_name(target.name + BUNDLE_SUFFIX)
    shutil.rmtree(previous, ignore_errors=True)
    moved = False
    try:
        if target.exists():
            target.rename(previous)
            moved = True
        staging.rename(target)
    except OSError as exc:
        if moved and not target.exists():
            try:
                previous.rename(target)
            except OSError:
                logger.error("Model Chain: could not put %s back after a failed promotion; "
                             "it is at %s", target, previous, exc_info=True)
        raise ManagedError(
            f"The verified download could not be moved into {target} ({exc}). It is still at "
            f"{staging} and nothing about the model you are using has changed."
        ) from None
    shutil.rmtree(previous, ignore_errors=True)


def _check_cancelled(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled("Download cancelled. What has arrived is kept, so starting it again "
                        "carries on from there.")


# --------------------------------------------------------------------------- #
# Apply / switch (section 8 of the design intent)
# --------------------------------------------------------------------------- #


_switch_lock = threading.Lock()
"""One switch at a time, process-wide.

Coarser than the runtime's own lock and deliberately outside it: a switch stops
a server, writes state and starts another server, and two of them interleaving
would produce exactly the thing this feature must never do, which is two
llama-servers holding weights at once.
"""


def use(identifier: str, on_status=None) -> Selection:
    """Make ``identifier`` the backbone every LLM Studio mode resolves.

    The transaction, in order: refuse if a generation is running, snapshot what
    is selected now, stop the server and wait for it to be *gone*, write the
    new selection, start it, and prove it can answer. Any failure after the
    snapshot puts the snapshot back and restarts what was there before.

    Only a fully verified bundle can be applied. That is not a second check for
    its own sake -- it is what makes a download failure and an apply failure
    two separate events with two separate blast radii, which is the property
    that lets a download run while a model is loaded.
    """
    say = on_status or (lambda _text: None)

    model = entry(identifier)
    bundle = installed(model.identifier)
    if bundle is None:
        raise ManagedError(f"{model.label} is not downloaded yet.")

    from prompt_master.models import managed_profiles

    if managed_profiles.profile(model.profile_id) is None:
        raise ManagedError(
            f"{model.label} names a quality profile ({model.profile_id}) that this version of "
            f"the extension does not have. Update the extension; nothing was changed."
        )

    with _switch_lock:
        return _switch(model, bundle, say)


def _switch(model: ManagedModel, bundle: Installed, say) -> Selection:
    import mc_broker

    running = mc_broker.active()
    if running is not None and running.family == mc_broker.FAMILY_LLM:
        raise Busy(
            f"The LLM is busy ({running.label}). Switching backbones would take the weights "
            f"out from under it — wait for it to finish and press Use again."
        )

    previous = _read_state()                                           # STEP 2
    previous_label = _state_label(previous)

    # Held for the whole switch so nothing can start a generation between the
    # old server stopping and the new one answering. Not required=True: a
    # workload we cannot take is an image generation on the card, and the
    # honest answer is to say so rather than to queue behind it for minutes.
    with mc_broker.workload(mc_broker.FAMILY_LLM, f"switching to {model.label}",
                            timeout=SWITCH_LOCK_TIMEOUT, required=False) as held:
        if not held:
            busy = mc_broker.active()
            raise Busy(
                f"{busy.label if busy else 'Something else'} is using the GPU. The backbone "
                f"was not changed — try again when it has finished."
            )

        say("Stopping the current model…")
        _stop_and_wait()                                               # STEPS 3, 4

        say(f"Starting {model.label}…")
        _record(model, bundle, previous)                               # STEP 5
        try:
            _start_and_smoke_test()                                    # STEPS 6, 7
        except Exception as exc:
            logger.warning("Model Chain: %s could not be started; rolling back to %s",
                           model.label, previous_label, exc_info=True)
            _rollback(previous)
            raise ManagedError(
                f"{model.label} was downloaded but would not start ({_sentence(exc)}). Rolled "
                f"back to {previous_label}; the files that were downloaded are kept."
            ) from None

    logger.info("Model Chain: LLM backbone switched to %s (%s)", model.label, model.identifier)
    return selection()                                                 # STEP 8


def _record(model: ManagedModel, bundle: Installed, previous: dict) -> None:
    """STEP 5 — the new selection, written in one atomic replace.

    Merged onto the previous state rather than built fresh: the runtime, the
    device, the offload and everything else the installation has learned about
    this machine are not decisions this feature gets to make.
    """
    from prompt_master.inference import model_choice

    paths = mc_llm_paths.app_paths()
    _write_state({
        **previous,
        "model": paths.record(bundle.model),
        "mmproj": paths.record(bundle.mmproj) if bundle.mmproj is not None else "",
        "quantization": model_choice.describe(Path(model.model.filename)),
        STATE_KEY_SOURCE: SOURCE_MANAGED,
        STATE_KEY_ID: model.identifier,
        STATE_KEY_PROFILE: model.profile_id,
        STATE_KEY_PROFILE_VERSION: _profile_version(),
    })


def _rollback(previous: dict) -> None:
    """STEP 9 — put the state back, and try to put the model back with it.

    Best-effort on the restart and not on the state: writing the previous
    selection must succeed or the installation is left describing a model it
    could not start, while *starting* it again is a courtesy -- the next
    request starts it anyway, and a second failure here would replace a useful
    error message with an unrelated one.
    """
    import mc_llm_runtime

    try:
        _stop_and_wait()
    except Exception:
        logger.debug("Model Chain: could not stop the failed backbone", exc_info=True)
    _write_state(previous)
    if not previous.get("model"):
        return
    try:
        mc_llm_runtime.runtime.client()
    except Exception:
        logger.warning("Model Chain: the previous model could not be restarted after a failed "
                       "backbone switch; it is still selected and will start on the next "
                       "request", exc_info=True)


def _stop_and_wait() -> None:
    """STEPS 3 and 4 — the old server is stopped, and *observed* to be gone."""
    import mc_llm_runtime

    mc_llm_runtime.runtime.stop()
    deadline = time.monotonic() + RESIDENT_STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not mc_llm_runtime.runtime.running():
            return
        time.sleep(0.25)
    raise ManagedError(
        "The llama-server that was running has not exited. The backbone was not changed — "
        "press Unload in the model sheet and try again."
    )


def _start_and_smoke_test() -> None:
    """STEPS 6 and 7 — start the new backbone, and make it answer once."""
    import mc_llm_runtime

    client = mc_llm_runtime.runtime.client()
    client.stream_chat(
        [{"role": "user", "content": "Reply with the single word: ready."}],
        max_tokens=SMOKE_TOKENS, seed=1, on_text=lambda _chunk: None,
    )


def _state_label(state: dict) -> str:
    """What to call the model a rollback went back to, in one phrase."""
    chosen = selection(state)
    if chosen.managed:
        try:
            return entry(chosen.identifier).label
        except ManagedError:
            return chosen.identifier
    recorded = str(state.get("model") or "")
    return Path(recorded).name if recorded else "no model"


def _sentence(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


# --------------------------------------------------------------------------- #
# Small shared pieces
# --------------------------------------------------------------------------- #


def _profile_version() -> str:
    from prompt_master.models import managed_profiles

    return str(managed_profiles.VERSION)


def _size_of(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _write_json(path: Path, document: dict) -> None:
    from prompt_master.core.config import atomic_write_json

    try:
        atomic_write_json(path, document)
    except OSError as exc:
        raise ManagedError(f"{path.name} could not be written ({exc}).") from None


def _bytes_label(value: int) -> str:
    value = int(value or 0)
    if value >= _GB:
        return f"{value / _GB:.1f} GB"
    if value >= _MB:
        return f"{value / _MB:.0f} MB"
    return f"{value} B"


def cleanup(identifier: str) -> None:
    """Throw away a staged, unfinished download for ``identifier``.

    Offered because a resumable download is only a kindness while the user
    still wants the model. Installed bundles are never touched here: deleting
    twenty gigabytes is not something a panel should do on a button that also
    means "tidy up".
    """
    shutil.rmtree(staging_root(identifier), ignore_errors=True)
