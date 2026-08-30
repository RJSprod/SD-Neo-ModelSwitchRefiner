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

SOPRO_DIRNAME = "sopro"
CLEANUP_DIRNAME = "cleanup"
CLEANUP_WORKER_DIRNAME = "cleanup_worker"
CLEANUP_MANIFEST_FILENAME = "managed-cleanup-models.json"
CLEANUP_MODEL_DIRNAME = "model"
SOPRO_WORKER_DIRNAME = "sopro_worker"
SOPRO_MANIFEST_FILENAME = "managed-sopro-models.json"
SOPRO_VOICES_DIRNAME = "voices"
SOPRO_REGISTRY_FILENAME = "registry.json"
SOPRO_REFERENCE_FILENAME = "reference.wav"
SOPRO_PRODUCTION_FILENAME = "production.safetensors"
SOPRO_PRODUCTION_META = "production.json"
SOPRO_LAB_FILENAME = "lab-conditioning.safetensors"
"""Sopro's whole subtree, under one directory of its own.

A sibling of ``runtime/`` and ``models/`` rather than a set of files mixed into
them, and section 16's dependency separation is the reason. Kokoro's runtime is
two unpacked wheels; Sopro's is a hundred and forty megabytes of PyTorch, and
the two must be installable, verifiable, upgradable and *deletable* without
either one being able to leave the other half-installed.

    sopro/
        runtime/                        installed.json + the isolated closure
            installed.json
        models/<bundle id>/             config.json, safetensors, tokenizer
        voices/
            registry.json               stable ids, names, fingerprints
            <uuid>/
                reference.wav           the retained normalized recording
                production.safetensors  cond_vec + semantic_tokens + mel
                production.json         level_db, shapes, schema, fingerprint
                lab-conditioning.safetensors   id_emb/style_emb/style_ctrl
        .downloads/                     staging only

``reference.wav`` is the second audio directory under this root, and it is the
same documented exception ``reference/`` is: a clone's own recording, kept so
that a later compatible Sopro can rebuild the prepared tensors without asking
somebody to record themselves again (section 30). It is never a dictation and
never a spoken reply, and deleting the voice deletes it.

The three files in a voice directory are three different lifetimes and that is
why they are three files. ``production.safetensors`` is what Conversation reads.
``lab-conditioning.safetensors`` is what the Voice Lab reads and may be
regenerated or deleted without touching production (section 14). The WAV outlives
both and is what either can be rebuilt from.
"""


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


# --------------------------------------------------------------------------- #
# Sopro
# --------------------------------------------------------------------------- #


def sopro_root() -> Path:
    """Everything Sopro owns, under one directory. Nothing else writes here."""
    return data_root() / SOPRO_DIRNAME


def sopro_runtime_root() -> Path:
    """The isolated Torch/Sopro closure. Never imported by Forge or by Kokoro."""
    return sopro_root() / RUNTIME_DIRNAME


def sopro_runtime_manifest() -> Path:
    return sopro_runtime_root() / INSTALLED_FILENAME


def sopro_models_root() -> Path:
    return sopro_root() / MODELS_DIRNAME


def sopro_model_root(identifier: str) -> Path:
    """One installed Sopro model bundle. ``identifier`` is checked, not trusted."""
    return _contained(sopro_models_root(), identifier)


def sopro_staging_root() -> Path:
    return sopro_root() / STAGING_DIRNAME


def sopro_staging_for(identifier: str, nonce: str) -> Path:
    return _contained(sopro_staging_root(), f"{identifier}-{nonce}")


def sopro_worker_script() -> Path:
    """The Sopro sidecar entry point, inside the extension rather than the data root.

    Its own file rather than a mode of the Kokoro worker, because the two are
    launched by *different interpreters* out of different closures. A single
    script would be one file that has to be importable under both, which is one
    import away from a Torch runtime that reaches for sherpa or the reverse.
    """
    return extension_root() / SOPRO_WORKER_DIRNAME / "worker.py"


def sopro_manifest_path() -> Path:
    """The checked-in trust root for every Sopro artifact this build may fetch."""
    return extension_root() / MANIFEST_DIRNAME / SOPRO_MANIFEST_FILENAME


def sopro_voices_root() -> Path:
    return sopro_root() / SOPRO_VOICES_DIRNAME


def sopro_registry_path() -> Path:
    return sopro_voices_root() / SOPRO_REGISTRY_FILENAME


# --------------------------------------------------------------------------- #
# The cleanup engine
# --------------------------------------------------------------------------- #


def cleanup_root() -> Path:
    """Everything the cleanup engine owns. Nothing else writes here.

    A third tree beside Kokoro's and Sopro's, for the reason those two are
    separate from each other: it carries its own interpreter and its own copy of
    Torch, and an installer that could reach into another engine's closure is an
    installer that can break a working engine while adding an optional one.
    """
    return data_root() / CLEANUP_DIRNAME


def cleanup_runtime_root() -> Path:
    """The cp311 interpreter and the closure unpacked beside it."""
    return cleanup_root() / RUNTIME_DIRNAME


def cleanup_runtime_manifest() -> Path:
    return cleanup_runtime_root() / INSTALLED_FILENAME


def cleanup_model_root() -> Path:
    """Where DeepFilterNet's own archive is unpacked, and what is handed to
    ``init_df`` as a path so nothing resolves anything after installation."""
    return cleanup_root() / CLEANUP_MODEL_DIRNAME


def cleanup_staging_root() -> Path:
    return cleanup_root() / STAGING_DIRNAME


def cleanup_worker_script() -> Path:
    return extension_root() / CLEANUP_WORKER_DIRNAME / "worker.py"


def cleanup_manifest_path() -> Path:
    return extension_root() / MANIFEST_DIRNAME / CLEANUP_MANIFEST_FILENAME


def sopro_settings_path() -> Path:
    """Sopro's global settings, in a file of Sopro's own.

    Not in the host's options, and that is the point. An option is a component
    on the settings page as well as a stored value, so Forge's "Apply settings"
    writes the page's stamped-at-build-time copy back over anything a live panel
    changed since -- which is how a chosen default voice, or a slowed-down
    delivery, quietly went back to what it had been.
    """
    return sopro_root() / "settings.json"


def sopro_voice_root(identifier: str) -> Path:
    """One saved Sopro voice's directory, addressed by server-generated UUID.

    Never by display name, never by anything a browser sent, and checked here
    anyway -- the registry is a file on disk and a corrupted one must not be
    able to address a path outside this root (section 57).
    """
    return _contained(sopro_voices_root(), _uuid(identifier))


def sopro_voice_file(identifier: str, name: str) -> Path:
    """One file inside one voice directory, by a name this module owns.

    ``name`` is compared against the four constants rather than joined, so a
    registry entry that named ``../../config.json`` reaches a refusal instead
    of a path.
    """
    known = (SOPRO_REFERENCE_FILENAME, SOPRO_PRODUCTION_FILENAME,
             SOPRO_PRODUCTION_META, SOPRO_LAB_FILENAME)
    if str(name) not in known:
        raise ValueError(f"unknown Sopro voice file: {name!r}")
    return sopro_voice_root(identifier) / str(name)


def sopro_inside(candidate) -> bool:
    """Whether ``candidate`` is under the Sopro subtree.

    Read before every delete. Section 57: voice deletion verifies that every
    path to be removed resolves under the Sopro voice root before removing it,
    and a function that answers that is better than the comparison written out
    at each call site.
    """
    try:
        Path(candidate).resolve().relative_to(sopro_root())
    except (OSError, ValueError):
        return False
    return True


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
