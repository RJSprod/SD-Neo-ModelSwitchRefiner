"""Where Voice Chat keeps its runtime, its two speech models, and nothing else.

Voice is two small I/O adapters bolted to a Conversation application that
already works, and the first thing that keeps it small is refusing to share a
folder with anything. Speech models are not managed backbones: they are a few
hundred megabytes rather than twenty gigabytes, they are installed by a
different button, they are verified against a different manifest, and they must
keep working on an installation where LLM Studio's managed-model feature has
been rewritten twice. So they get a root of their own beside
``model_chain_llm`` rather than a subfolder inside it.

    <WebUI data directory>/model_chain_voice/
        runtime/
            installed.json
            <isolated CPU worker runtime>
        models/
            stt/<bundle id>/...
            tts/<bundle id>/...
        bank/
            manifest.json
            model.onnx          derived, extended speaker metadata
            voices.bin          the live Kokoro voice bank
            .staging/           one build at a time, promoted by rename
        clones/
            <uuid>.bin          canonical custom voicepacks
            registry.json       stable ids, names, slots
        cloning/
            bin/ assets/ manifest.json
        reference/
            <job>.wav           one clone's input, deleted when it is done
        .downloads/
            <staging only>

Nothing in here creates a directory. An installation that never presses
"Download default STT" should not grow a folder for it, which is the same rule
:mod:`mc_llm_paths` states and for the same reason.

What is deliberately *not* here
-------------------------------
No speech audio. Invariant I-5 says microphone audio and generated speech stay
in memory, so there is no ``audio/``, no ``cache/``, and nowhere under this root
that a dictation or a spoken reply could correctly be written to.
:func:`inside` exists to make that enforceable rather than merely intended:
every path this feature writes is built here, and a test can ask whether a
candidate is under the one root the feature owns.

``reference/`` is the one audio directory and it is an exception with a name.
It holds the WAV somebody chose as the input to a voice clone -- their own file,
copied here because a subprocess has to be able to read it, deleted when the
job finishes or is abandoned (section 79). It is never a dictation, never a
generated reply, and never written by anything on the speech path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_ROOT = "model_chain_voice_root"
"""Somewhere other than the data directory to keep the voice runtime.

Registered as a setting for the same reason :data:`mc_llm_paths.OPT_ROOT` is:
the runtime is a Python environment plus most of a gigabyte of ONNX, and the
drive the WebUI was installed on is not always the drive that has room for it.
"""

DIRNAME = "model_chain_voice"

RUNTIME_DIRNAME = "runtime"
MODELS_DIRNAME = "models"
STAGING_DIRNAME = ".downloads"
"""A sibling of the installed bundles, so promoting a verified download is a
rename within one filesystem -- which is what makes it atomic. The leading dot
keeps a half-downloaded model out of the folder listings a user browses."""

INSTALLED_FILENAME = "installed.json"
"""Written last and only after every hash has been checked. Its presence is
what "installed" means; the files being there is not."""

KINDS = ("stt", "tts")
"""The two model roles V1 has, and the two subfolders of :func:`models_root`.

A tuple rather than two functions because every caller that walks one walks
both: readiness is ``stt and tts``, the installer loops over them, and the
manifest declares a default for each.
"""

WORKER_DIRNAME = "voice_worker"
MANIFEST_DIRNAME = "voice"
MANIFEST_FILENAME = "managed-voice-models.json"

BANK_DIRNAME = "bank"
BANK_MANIFEST = "manifest.json"
BANK_VOICES = "voices.bin"
BANK_MODEL = "model.onnx"
BANK_STAGING = ".staging"
"""The Model Chain voice bank.

A directory of its own rather than files written into the installed Kokoro
bundle, and that is a rule rather than tidiness: the bundle is verified against
a manifest hash, and a feature that edited a file inside it would make its own
installation fail its own integrity check the next time anybody looked.
"""

CLONES_DIRNAME = "clones"
REGISTRY_FILENAME = "registry.json"
CLONING_DIRNAME = "cloning"
REFERENCE_DIRNAME = "reference"


def extension_root() -> Path:
    """The extension directory, which is where this file is."""
    return Path(__file__).resolve().parent


def data_root() -> Path:
    """The voice root. Setting first, then ``<data>/model_chain_voice``.

    Deliberately not derived from :func:`mc_llm_paths.data_root`. Pointing the
    LLM root at an existing standalone install is a supported thing to do, and
    it would drag the speech runtime into somebody else's application folder
    as a side effect of a setting that says nothing about speech.
    """
    configured = _setting(OPT_ROOT)
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return (_webui_data_path() / DIRNAME).resolve()


def runtime_root() -> Path:
    """Where the isolated CPU worker runtime lives."""
    return data_root() / RUNTIME_DIRNAME


def runtime_manifest() -> Path:
    """The runtime's own ``installed.json``."""
    return runtime_root() / INSTALLED_FILENAME


def models_root() -> Path:
    return data_root() / MODELS_DIRNAME


def kind_root(kind: str) -> Path:
    """``models/stt`` or ``models/tts``. Any other word is a programming error."""
    if kind not in KINDS:
        raise ValueError(f"unknown voice model kind: {kind!r}")
    return models_root() / kind


def bundle_root(kind: str, identifier: str) -> Path:
    """Where one installed model bundle lives.

    ``identifier`` is checked rather than trusted. It arrives from the checked-in
    manifest today, but the manifest is a file on somebody's disk and a bundle
    id spelled ``../../`` would otherwise resolve to a directory this feature
    has no business writing to.
    """
    return _contained(kind_root(kind), identifier)


def staging_root() -> Path:
    return data_root() / STAGING_DIRNAME


def staging_for(identifier: str, nonce: str) -> Path:
    """A private staging directory for one install attempt.

    The nonce is what makes two presses of Download safe: each attempt assembles
    into a directory of its own and promotes it whole, so a second attempt can
    never find and adopt the first one's half-written bytes.
    """
    return _contained(staging_root(), f"{identifier}-{nonce}")


def worker_script() -> Path:
    """The sidecar entry point, inside the extension rather than the data root.

    It is code, it is reviewed, and it ships with the extension: putting it in
    the data directory would make the file a *user's* file, which is exactly
    the kind of thing that ends up edited, backed up, restored from an old
    version and then blamed on the runtime.
    """
    return extension_root() / WORKER_DIRNAME / "worker.py"


def bank_root() -> Path:
    """Where the built voice bank and its derived model live."""
    return data_root() / BANK_DIRNAME


def bank_manifest() -> Path:
    return bank_root() / BANK_MANIFEST


def bank_voices() -> Path:
    return bank_root() / BANK_VOICES


def bank_model() -> Path:
    return bank_root() / BANK_MODEL


def bank_staging() -> Path:
    """A sibling of the live bank, so promotion is a rename on one filesystem.

    The atomicity of the whole bank transaction rests on this being a sibling:
    ``os.replace`` across filesystems is a copy, and a copy is exactly the
    window in which a half-written ``voices.bin`` can exist -- which release
    blocker six forbids.
    """
    return bank_root() / BANK_STAGING


def clones_root() -> Path:
    """Canonical custom voicepacks, one file per registered clone."""
    return data_root() / CLONES_DIRNAME


def clone_file(identifier: str) -> Path:
    """``clones/<uuid>.bin`` for a generated internal id.

    ``identifier`` is a UUID this process generated, never a display name and
    never anything a browser sent -- section 77. It is still checked, because
    the registry is a file on disk and a corrupted one must not be able to
    address a path outside this root.
    """
    return _contained(clones_root(), f"{_uuid(identifier)}.bin")


def registry_path() -> Path:
    """The voice registry: stable ids, display names, slots and SIDs."""
    return clones_root() / REGISTRY_FILENAME


def cloning_root() -> Path:
    """Where an installed Storytime cloning bundle lives, if there is one."""
    return data_root() / CLONING_DIRNAME


def reference_root() -> Path:
    """The managed copy of a clone's reference recording.

    The one place under this root where audio may be written, and it is the
    documented exception rather than a hole in invariant I-5: a clone needs its
    reference on disk for a subprocess to read, it is the user's own file, and
    section 79 deletes it when the job ends either way.
    """
    return data_root() / REFERENCE_DIRNAME


def reference_file(identifier: str) -> Path:
    return _contained(reference_root(), f"{_uuid(identifier)}.wav")


def _uuid(identifier: str) -> str:
    """``identifier`` if it is a plain hex/uuid word, else a refusal."""
    text = str(identifier or "").strip()
    if not text or len(text) > 64:
        raise ValueError(f"unsafe voice id: {identifier!r}")
    for character in text:
        if not (character.isalnum() or character == "-"):
            raise ValueError(f"unsafe voice id: {identifier!r}")
    return text


def manifest_path() -> Path:
    """The checked-in trust root for every artifact Voice Chat may download."""
    return extension_root() / MANIFEST_DIRNAME / MANIFEST_FILENAME


def inside(candidate) -> bool:
    """Whether ``candidate`` is under the one root this feature owns.

    The privacy tests use it, and so does every write: a feature whose whole
    promise includes "no audio files anywhere" needs one function that answers
    "is this ours" rather than a comparison written out at each call site.
    """
    try:
        Path(candidate).resolve().relative_to(data_root())
    except (OSError, ValueError):
        return False
    return True


def _contained(parent: Path, name: str) -> Path:
    """``parent / name``, refusing anything that escapes ``parent``."""
    text = str(name or "").strip()
    if not text or text in (".", "..") or "/" in text or "\\" in text or os.path.isabs(text):
        raise ValueError(f"unsafe voice bundle name: {name!r}")
    resolved = (parent / text).resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(f"unsafe voice bundle name: {name!r}") from exc
    return resolved


def _webui_data_path() -> Path:
    try:
        from modules import paths

        base = Path(paths.data_path)
    except Exception:
        base = extension_root()
    return base


def _setting(name: str):
    try:
        from modules import shared

        return getattr(shared.opts, name, None)
    except Exception:
        return None
