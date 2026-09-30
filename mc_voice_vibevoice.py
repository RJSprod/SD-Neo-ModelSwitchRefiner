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
                                   :func:`note_peak` keeps the highest, per
                                   model *and precision*, so
                                   :func:`need_vram_bytes` -- what the turn is
                                   asked for -- stands on a measurement as soon
                                   as there is one.

Two models, one runtime
-----------------------
The 7B (``longform``: up to four voices cloned from Voice Box samples, three
precisions, LoRA adapters) and the Realtime 0.5B (``realtime``: one voice from
Microsoft's preset voice prompts, full precision only, no LoRA) are two entries of
one manifest, installed and reported separately (:func:`model_info`), and served by
one runtime: the community ``vibevoice`` 0.0.1 wheel has the long-form code and no
streaming model, Microsoft's own repository has the streaming model and no
long-form code, and their shared modules are the same code. So the runtime is the
wheel plus an *overlay* of five files from Microsoft's repository at one commit,
each pinned here by size and SHA-256 and written over the unpacked package
(:func:`overlay_files`). The overlay's digests are part of :func:`closure_id`, so
a runtime installed before it existed reads as one to install again.

The Realtime model cannot clone a voice -- Microsoft withholds the code that makes
a voice prompt from a recording -- so its voices are the preset ``.pt`` files from
the same commit, pinned here and installed beside it (:func:`presets`). Nothing a
person supplies is ever turned into one.

What is provisional, and says so
--------------------------------
The PyPI wheels, the overlay and the preset voices are pinned byte for byte. The
torch wheel, the models' files, their shards and their tokenizers are declared and
not hashed, because the machine that wrote the manifest could reach pypi.org and
raw.githubusercontent.com and nothing else. Each of those is checked against the
digest its publisher states over HTTPS at install time and recorded in
``voice/managed-vibevoice-models.local.json``, so the second install is checked
against a constant -- exactly as ``README.md`` "Who vouches for the bytes"
describes, and :attr:`Status.provisional` is how the page says it.

The LoRA library
----------------
Adapters for the 7B's language model (and, optionally, its diffusion head and
connectors) are copied from a folder on this PC into a library of their own
(:func:`add_lora`), validated first -- a PEFT LoRA adapter where one is expected,
nothing larger than :data:`LORA_BYTES_MAX` -- and normalised so the language
model's adapter is at the root of its directory, which is the one shape the worker
has to know.
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

MODEL_7B = MODEL_DEFAULT = "vibevoice-7b"
MODEL_REALTIME = "vibevoice-realtime-0.5b"

KIND_LONGFORM = "longform"
KIND_REALTIME = "realtime"
MODEL_KINDS = (KIND_LONGFORM, KIND_REALTIME)
"""What a model is. ``longform`` speaks a script with up to four voices cloned
from samples; ``realtime`` speaks with one preset voice and streams as it goes."""

PRECISIONS = ("bf16", "int8", "nf4")
PRECISION_DEFAULT = "bf16"
PRECISION_LABELS = {"bf16": "Full (bf16)", "int8": "8-bit", "nf4": "4-bit (NF4)"}
"""How the language model's weights are held on the card. Only the language model
is ever quantised (the worker skips the tokenizers, the diffusion head and the
connectors), so the smaller precisions cost the most where the most is."""

QUANTISED_SHARE = {"bf16": 1.0, "int8": 0.62, "nf4": 0.41}
"""The weights at a precision as a share of the bf16 weights, for a model whose
manifest gives no figure for that precision. The 7B's own figures (11.5 and 7.5 GB
of 18.7) are where these come from; a render's measured peak replaces either."""

LORA_SCALE_MIN = 0.0
LORA_SCALE_MAX = 2.0
LORA_SCALE_DEFAULT = 1.0
LORA_BYTES_MAX = 4 * 1024 * 1024 * 1024
LORA_NAME_MAX = 80
LORA_PARTS = ("llm", "diffusion_head", "acoustic_connector", "semantic_connector")
"""What an adapter in the library can carry. ``llm`` -- the language model's PEFT
adapter -- is always there; the other three are folders copied when present."""

LORA_EXTRA_FILES = ("model.safetensors", "pytorch_model.bin", "diffusion_head_full.bin")
"""A full state dict for an optional part, by the names finetuning writes them."""
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")

LANGUAGE_LABELS = {
    "en": "English", "de": "German", "fr": "French", "it": "Italian", "jp": "Japanese",
    "kr": "Korean", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "sp": "Spanish",
}
ACCENT_LABELS = {"in": "India"}
"""How a preset's language reads on the page. The codes are upstream's own file
prefixes (``jp``, ``kr``, ``sp`` rather than ISO's), because they are what the
voices are called; ``in-Samuel`` is an English voice with an Indian accent,
released with the English ones rather than with the experimental languages."""

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

CHAT_DEFAULTS = {
    "card_uuid": "",
    "precision": PRECISION_DEFAULT,
    "lora_id": "",
    "lora_scale": LORA_SCALE_DEFAULT,
}
"""Voice Chat's VibeVoice settings, kept in this file under ``"chat"``: the card
it speaks on (blank means the Voice Box's), and the precision and LoRA it uses
when a reply is spoken by the 7B. The Realtime model is always full precision
and takes no LoRA, whatever is stored here."""

SETTINGS_DEFAULTS = {
    "card_uuid": "",
    "model_id": MODEL_DEFAULT,
    "steps": STEPS_DEFAULT,
    "cfg_scale": CFG_DEFAULT,
    "seed": None,
    "max_new_tokens": None,
    "keep_warm": True,
    "precision": PRECISION_DEFAULT,
    "lora_id": "",
    "lora_scale": LORA_SCALE_DEFAULT,
    "chat": CHAT_DEFAULTS,
}
"""Voice Box's engine settings, and the shape :func:`settings` always answers in.

``keep_warm`` defaults on: a warm 7B answers the next render in seconds rather
than the minute a cold load costs, and it leaves the card the moment an image
job, WanGP or the language model needs the room (docs/23-voice-box.md §4).
``precision``, ``lora_id`` and ``lora_scale`` apply to the model ``model_id``
names, and read as full precision and no LoRA for a model that takes neither.
"""

_lock = threading.RLock()
_lora_lock = threading.RLock()
"""Held while the LoRA library changes. Its own lock, because copying a
four-gigabyte adapter must not hold up a status read of the manifest."""
_manifest_cache = None

_HUB = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")
_SAFE_UUID = re.compile(r"^[A-Za-z0-9,\-]*$")
_SAFE_STEM = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
"""A preset voice's id, exactly the characters the worker accepts."""
_SAFE_LORA = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_OVERLAY_PATH = re.compile(r"^vibevoice/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py$")
"""Where an overlay file may be written: a Python module inside the package, and
nowhere else in the runtime."""


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
class Preset:
    """One of the Realtime model's preset voice prompts, as the manifest pins it."""

    identifier: str
    name: str
    language: str
    accent: str
    gender: str
    experimental: bool
    artifact: models.Artifact

    @property
    def filename(self) -> str:
        return f"{self.identifier}.pt"

    @property
    def language_label(self) -> str:
        label = LANGUAGE_LABELS.get(self.language, self.language.upper() or "Unknown")
        accent = ACCENT_LABELS.get(self.accent, self.accent.upper()) if self.accent else ""
        return f"{label} ({accent})" if accent else label


@dataclass(frozen=True)
class OverlayFile:
    """One file of Microsoft's repository written over the unpacked package."""

    path: str
    artifact: models.Artifact
    repository: str
    commit: str
    why: str

    @property
    def basename(self) -> str:
        return self.path.rsplit("/", 1)[-1]


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
    kind: str = KIND_LONGFORM
    precisions: tuple = (PRECISION_DEFAULT,)
    lora: bool = False
    max_speakers: int = 4
    defaults: dict = field(default_factory=dict, compare=False)
    voices: tuple = ()
    single: "models.Artifact | None" = None
    """The one weights file to take when the model publishes no safetensors index."""

    @property
    def required_paths(self) -> tuple:
        """The declared files that must be there, by installed name."""
        return tuple(item.local_name for item in self.artifacts
                     if item.local_name not in self.optional)

    @property
    def index_optional(self) -> bool:
        """Whether the safetensors index may be absent, the weights being one file."""
        return self.single is not None and "model.safetensors.index.json" in self.optional

    @property
    def realtime(self) -> bool:
        return self.kind == KIND_REALTIME

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
    def voices_bytes(self) -> int:
        return sum(int(item.artifact.size or 0) for item in self.voices)

    @property
    def download_bytes(self) -> int:
        """How large the model part is, for a sentence somebody reads first."""
        heads = sum(int(item.size or 0) for item in self.artifacts)
        return (heads + self.estimated_weights_bytes + self.tokenizer.download_bytes
                + self.voices_bytes)


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
    kind = str(entry.get("kind") or KIND_LONGFORM)
    if kind not in MODEL_KINDS:
        raise VibeVoiceError(f"The VibeVoice manifest calls {wanted} a {kind!r} model, which "
                             f"this build does not know how to run.")
    precisions = tuple(str(name) for name in (entry.get("precisions") or (PRECISION_DEFAULT,)))
    if not precisions or any(name not in PRECISIONS for name in precisions):
        raise VibeVoiceError(f"The VibeVoice manifest gives {wanted} a precision this build "
                             f"does not know ({', '.join(precisions) or 'none'}).")
    voices = tuple(_preset(item, wanted) for item in (entry.get("voices") or ()))
    if len({voice.identifier for voice in voices}) != len(voices):
        raise VibeVoiceError(f"The VibeVoice manifest names one of {wanted}'s preset voices "
                             f"twice.")
    single = entry.get("single_weights")
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
        raw=entry,
        kind=kind,
        precisions=precisions,
        lora=bool(entry.get("lora")) and kind == KIND_LONGFORM,
        max_speakers=max(1, min(int(entry.get("max_speakers")
                                    or (1 if kind == KIND_REALTIME else 4)), 4)),
        defaults=_model_defaults(entry.get("defaults")),
        voices=voices,
        single=(_artifact(single, f"{wanted} weights", repo)
                if isinstance(single, dict) else None))


def _model_defaults(found) -> dict:
    """A model's own diffusion steps and CFG, each inside the bounds settings hold."""
    found = found if isinstance(found, dict) else {}
    steps = found.get("steps")
    cfg = found.get("cfg_scale")
    steps = steps if isinstance(steps, int) and not isinstance(steps, bool) \
        and STEPS_MIN <= steps <= STEPS_MAX else STEPS_DEFAULT
    cfg = float(cfg) if isinstance(cfg, (int, float)) and not isinstance(cfg, bool) \
        and CFG_MIN <= float(cfg) <= CFG_MAX else CFG_DEFAULT
    return {"steps": steps, "cfg_scale": cfg}


def _preset(item, owner: str) -> Preset:
    """One preset voice as the manifest pins it. Its id is its filename's stem, and a
    stem with anything but letters, digits, ``-`` and ``_`` is refused here, before
    anything could turn it into a path."""
    if not isinstance(item, dict):
        raise VibeVoiceError(f"The VibeVoice manifest's {owner} voices hold something that "
                             f"is not a voice.")
    stem = str(item.get("id") or "").strip()
    if not _SAFE_STEM.match(stem):
        raise VibeVoiceError(f"The VibeVoice manifest's {owner} names a preset voice "
                             f"({stem!r}) that is not a plain file name.")
    artifact = _artifact({"filename": f"{stem}.pt", "local_name": f"{stem}.pt",
                          "url": item.get("url"), "bytes": item.get("bytes"),
                          "sha256": item.get("sha256")}, f"{owner} voices")
    return Preset(identifier=stem, name=str(item.get("name") or stem),
                  language=str(item.get("language") or "").strip().lower(),
                  accent=str(item.get("accent") or "").strip().lower(),
                  gender=str(item.get("gender") or "").strip().lower(),
                  experimental=bool(item.get("experimental")), artifact=artifact)


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


def overlay_files() -> tuple:
    """The files of Microsoft's repository written over the unpacked package.

    Five of them, at one commit, each pinned here by size and SHA-256: the
    streaming model, its configuration, its inference class, its processor, and
    a configuration module that replaces the wheel's own with the superset the
    streaming configuration imports from. Code, not data, so an overlay entry
    without a committed digest is a broken manifest rather than a file to check
    against a publisher's word -- raw.githubusercontent.com states none.
    """
    found = []
    seen = set()
    for item in manifest()["runtime"].get("overlay") or ():
        if not isinstance(item, dict):
            raise VibeVoiceError("The VibeVoice manifest's runtime overlay holds something "
                                 "that is not a file.")
        path = str(item.get("path") or "").strip()
        if not _OVERLAY_PATH.match(path) or path in seen:
            raise VibeVoiceError(f"The VibeVoice manifest's runtime overlay names {path!r}, "
                                 f"which is not one module inside the vibevoice package.")
        seen.add(path)
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        artifact = _artifact({"filename": path.rsplit("/", 1)[-1],
                              "local_name": path.rsplit("/", 1)[-1], "url": item.get("url"),
                              "bytes": item.get("bytes"), "sha256": item.get("sha256")},
                             "runtime overlay")
        if not artifact.pinned:
            raise VibeVoiceError(f"The VibeVoice manifest's runtime overlay does not pin "
                                 f"{path} by size and SHA-256. This is a problem with the "
                                 f"extension rather than with your installation.")
        found.append(OverlayFile(path=path, artifact=artifact,
                                 repository=str(source.get("repository") or ""),
                                 commit=str(source.get("commit") or ""),
                                 why=str(item.get("why") or "")))
    return tuple(found)


def pinned() -> bool:
    """Whether this build has resolved its PyPI closure and its overlay for this machine.

    The torch resolve does not count against it, by design: it can never be
    pinned from the machine that writes the manifest, and refusing every
    install until it is would be refusing the feature.
    """
    try:
        chosen = platform()
        overlay = overlay_files()
    except VibeVoiceError:
        return False
    if chosen is None or not chosen.artifacts or not overlay:
        return False
    return all(item.pinned for item in chosen.artifacts)


def closure_id() -> str:
    """A fingerprint of exactly which wheels and overlay files this platform installs.

    The pinned wheels' digests, the overlay's digests, and the torch *version*,
    not the torch digest: recording the digest later with the pin tool must not
    make every installed runtime stale, since the bytes it names are the bytes
    already unpacked. An overlay file that changes -- or a runtime installed
    before there was an overlay -- makes the installed runtime one to install
    again, which is the only way a new streaming model reaches it.
    """
    chosen = platform()
    resolve = torch_resolve()
    if chosen is None:
        return ""
    parts = [chosen.closure_id, f"torch:{full_torch_version(resolve) if resolve else ''}"]
    parts += [f"overlay:{item.path}:{item.artifact.sha256}" for item in overlay_files()]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _runtime_bytes(chosen, resolve) -> int:
    total = sum(int(item.size or 0) for item in (chosen.artifacts if chosen else ()))
    try:
        total += sum(int(item.artifact.size or 0) for item in overlay_files())
    except VibeVoiceError:
        pass
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
    kind: str = KIND_LONGFORM
    runtime_stale: bool = False
    """Installed, but by a build that pinned a different closure or overlay."""
    presets_installed: int = 0
    presets_total: int = 0

    @property
    def presets_ready(self) -> bool:
        """A Realtime model speaks only with a preset voice; the 7B needs none."""
        return self.kind != KIND_REALTIME or self.presets_installed > 0

    @property
    def ready(self) -> bool:
        """Whether a render can be attempted: runtime, model, tokenizer and a voice."""
        return bool(self.supported and self.runtime_installed and self.model_installed
                    and self.tokenizer_installed and self.presets_ready)

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
        elif not self.presets_ready:
            missing.append("preset voices (install the model again)")
        return f"Setup required — VibeVoice's {' and '.join(missing)} still to install."


def status(identifier: str = "") -> Status:
    """Read from disk. Starts nothing, downloads nothing, never raises.

    ``identifier`` names the model; ``""`` is the one the settings name.
    """
    try:
        return _status(identifier)
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


def _status(identifier: str = "") -> Status:
    chosen = platform()
    entry = bundle(identifier)
    found = Status(model_id=entry.identifier, model_label=entry.label,
                   provisional=_provisional(entry.raw), kind=entry.kind,
                   presets_total=len(entry.voices))
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
        found.runtime_stale = True
        found.runtime_message = ("Installed, but this build pins a different VibeVoice "
                                 "runtime. Install it again to update.")
    else:
        found.runtime_installed = True
        found.runtime_message = (
            f"Installed — VibeVoice {installed.get('vibevoice_version') or '?'}, torch "
            f"{installed.get('torch_version') or '?'}, transformers "
            f"{installed.get('transformers_version') or '?'}.")
        features = installed.get("features") if isinstance(installed.get("features"),
                                                           dict) else {}
        if features.get("quantisation") is False:
            found.runtime_message += (" 8-bit and 4-bit are unavailable: bitsandbytes did not "
                                      "import when it was checked.")
        if features.get("lora") is False:
            found.runtime_message += (" LoRA adapters are unavailable: PEFT did not import "
                                      "when it was checked.")
    found.closure = dict(installed or {})

    root = paths.vibevoice_model_root(entry.identifier)
    marker = _read_json(root / paths.INSTALLED_FILENAME)
    found.installed_model = dict(marker or {})
    found.model_bytes = entry.download_bytes
    found.presets_installed = _presets_present(entry, root)
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
        voices = (f", {found.presets_installed} of {len(entry.voices)} preset voices"
                  if entry.voices else "")
        weights = ("one weights file" if (marker or {}).get("weights") == "single"
                   else f"{len(shards)} shard{'s' if len(shards) != 1 else ''}")
        found.model_message = (f"Installed — {entry.label}, {weights}"
                               f"{', ' + models._bytes_label(size) if size else ''}{voices}.")
    found.tokenizer_installed = _tokenizer_present(entry, root)
    if found.model_installed and not found.tokenizer_installed:
        found.model_message += " Its tokenizer is missing — install the model again."
    if found.model_installed and found.presets_installed < found.presets_total:
        absent = found.presets_total - found.presets_installed
        found.model_message += (f" {absent} of its {found.presets_total} preset voices "
                                f"{'is' if absent == 1 else 'are'} missing — install the "
                                f"model again.")
    complete = (found.model_installed and found.tokenizer_installed
                and found.presets_installed == found.presets_total)
    found.download_bytes = ((0 if found.runtime_installed else found.runtime_bytes)
                            + (0 if complete else found.model_bytes))
    return found


def _preset_present(entry: Bundle, preset: Preset, root: "Path | None" = None) -> bool:
    """Whether one preset voice is in place, at the size its pin says. A size, not a
    hash: this is read on every status poll, and the hash was checked on arrival."""
    folder = Path(root) if root is not None else paths.vibevoice_model_root(entry.identifier)
    try:
        size = (folder / paths.VIBEVOICE_VOICES_DIRNAME / preset.filename).stat().st_size
    except OSError:
        return False
    return size > 0 and (not preset.artifact.size or size == int(preset.artifact.size))


def _presets_present(entry: Bundle, root: "Path | None" = None) -> int:
    return sum(1 for preset in entry.voices if _preset_present(entry, preset, root))


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


def model_info(identifier: str = "") -> dict:
    """One model as the pages see it: what it is, what it takes, and what is missing.

    ``identifier`` ``""`` is the model the settings name. A model this build does
    not know is a :class:`VibeVoiceError`. ``installed`` is the model's own files
    and its tokenizer; ``runtime_installed`` is the runtime every model shares;
    ``ready`` is both, plus, for a Realtime model, at least one preset voice --
    the whole of what a render (or, for the 7B, a clone) needs. ``message`` says
    in one sentence what stands between this model and a render, or is ``""``.
    """
    entry = bundle(identifier)
    current = status(entry.identifier)
    vram = {}
    ram = 0
    for precision in entry.precisions:
        try:
            vram[precision] = int(need_vram_bytes(entry.identifier, precision))
            ram = max(ram, int(need_ram_bytes(entry.identifier, precision)))
        except Exception:
            logger.debug("Model Chain: could not size %s at %s", entry.identifier, precision,
                         exc_info=True)
            vram[precision] = 0
    complete = current.model_installed and current.tokenizer_installed
    if complete:
        root = paths.vibevoice_model_root(entry.identifier)
        download = sum(int(item.artifact.size or 0) for item in entry.voices
                       if not _preset_present(entry, item, root))
    else:
        download = entry.download_bytes
    return {
        "id": entry.identifier,
        "label": entry.label,
        "kind": entry.kind,
        "installed": bool(complete),
        "runtime_installed": bool(current.runtime_installed),
        "ready": bool(current.ready),
        "precisions": list(entry.precisions),
        "precision_labels": {name: PRECISION_LABELS[name] for name in entry.precisions},
        "lora": bool(entry.lora),
        "max_speakers": int(entry.max_speakers),
        "voices": "presets" if entry.realtime else "samples",
        "defaults": dict(entry.defaults),
        "need_vram_bytes": vram,
        "need_ram_bytes": int(ram),
        "presets": presets(entry.identifier) if entry.voices else [],
        "presets_installed": int(current.presets_installed),
        "message": _missing_sentence(entry, current),
        "download_bytes": int(download),
        "summary": entry.summary,
        "license": entry.license,
        "provisional": bool(current.provisional),
    }


def models_info() -> list:
    """Every model this build knows, in manifest order, as :func:`model_info` has it."""
    found = []
    for name in model_ids():
        try:
            found.append(model_info(name))
        except VibeVoiceError:
            logger.warning("Model Chain: the VibeVoice manifest's %s could not be read", name,
                           exc_info=True)
    return found


def _missing_sentence(entry: Bundle, current: Status) -> str:
    """What stands between ``entry`` and a render, in one sentence, or ``""``.

    Precise on purpose: this is what a form reads to say whether its model --
    the 7B for a clone, the Realtime model for a preset -- is ready, and "setup
    required" is not something anybody can act on.
    """
    if not current.supported:
        return current.runtime_message
    clauses = []
    if not current.runtime_installed:
        clauses.append("VibeVoice's runtime was installed by an earlier build and has to be "
                       "installed again" if current.runtime_stale
                       else "VibeVoice's runtime is not installed")
    root = paths.vibevoice_model_root(entry.identifier)
    missing = _model_missing(entry, root, current.installed_model)
    if not current.model_installed:
        if current.installed_model and missing:
            clauses.append(f"{entry.label} is missing {missing[0]} and has to be installed "
                           f"again")
        else:
            clauses.append(f"{entry.label} is not installed")
    elif not current.tokenizer_installed:
        clauses.append(f"{entry.label}'s tokenizer is missing and the model has to be "
                       f"installed again")
    elif entry.voices and current.presets_installed == 0:
        clauses.append(f"none of {entry.label}'s {len(entry.voices)} preset voices are "
                       f"installed, so the model has to be installed again")
    elif entry.voices and current.presets_installed < len(entry.voices):
        absent = len(entry.voices) - current.presets_installed
        clauses.append(f"{absent} of {entry.label}'s {len(entry.voices)} preset voices "
                       f"{'is' if absent == 1 else 'are'} missing (installing the model again "
                       f"fetches {'it' if absent == 1 else 'them'})")
    if not clauses:
        return ""
    if len(clauses) == 2 and clauses == ["VibeVoice's runtime is not installed",
                                         f"{entry.label} is not installed"]:
        return f"Neither VibeVoice's runtime nor {entry.label} is installed."
    text = "; ".join(clauses)
    return text[0].upper() + text[1:] + "."


def presets(identifier: str = MODEL_REALTIME) -> list:
    """The preset voices of ``identifier``, in manifest order, English ones first.

    ``installed`` is whether the voice's file is in place at its pinned size. A
    model without presets -- the 7B, whose voices are Voice Box samples -- has
    an empty list; a model this build does not know is a :class:`VibeVoiceError`.
    """
    entry = bundle(identifier)
    root = paths.vibevoice_model_root(entry.identifier)
    return [{"id": item.identifier, "name": item.name, "language": item.language,
             "language_label": item.language_label, "accent": item.accent,
             "gender": item.gender, "experimental": bool(item.experimental),
             "installed": _preset_present(entry, item, root),
             "bytes": int(item.artifact.size or 0)}
            for item in entry.voices]


def voices_dir(identifier: str = MODEL_REALTIME) -> Path:
    """Where ``identifier``'s preset voices are installed. Validated, never a caller's path."""
    return paths.vibevoice_voices_root(bundle(identifier).identifier)


def preset_path(stem: str, identifier: str = MODEL_REALTIME) -> Path:
    """The file of one preset voice, by its id. Only an id the manifest names is
    ever turned into a path; anything else is a :class:`VibeVoiceError`."""
    entry = bundle(identifier)
    wanted = str(stem or "").strip()
    for item in entry.voices:
        if item.identifier == wanted:
            return paths.vibevoice_voices_root(entry.identifier) / item.filename
    if not entry.voices:
        raise VibeVoiceError(f"{entry.label} has no preset voices; its voices are Voice Box "
                             f"samples.")
    raise VibeVoiceError(f"{wanted[:64]!r} is not one of {entry.label}'s preset voices.")


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
    try:
        described = models_info()
    except Exception:
        logger.debug("Model Chain: could not describe the VibeVoice models", exc_info=True)
        described = [{"id": name, "label": _model_label(name)} for name in model_ids()]
    try:
        library = loras()
    except Exception:
        logger.debug("Model Chain: could not read the VibeVoice LoRA library", exc_info=True)
        library = []
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
        "models": described,
        "loras": library,
        "precision_labels": dict(PRECISION_LABELS),
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
    found = {key: (dict(value) if isinstance(value, dict) else value)
             for key, value in SETTINGS_DEFAULTS.items()}
    for key in SETTINGS_DEFAULTS:
        if key == "chat" or key not in stored:
            continue
        try:
            found[key] = _validated(key, stored[key])
        except VibeVoiceError:
            logger.debug("Model Chain: a stored VibeVoice setting (%s) was ignored", key)
    if found["model_id"] not in model_ids():
        found["model_id"] = _default_model_id() or MODEL_DEFAULT
    found["chat"] = _chat_read(stored.get("chat"))
    # What the model the settings name can take. A stored precision it does not
    # run in reads as its own first, and a LoRA as none, without being erased:
    # switching back to the 7B finds the choice that was made for it.
    precisions, takes_lora = _model_takes(found["model_id"])
    if found["precision"] not in precisions:
        found["precision"] = precisions[0]
    if not takes_lora:
        found["lora_id"] = ""
    return found


def set_settings(values: dict) -> dict:
    """Change engine settings. Validated, refused rather than dropped.

    An unknown key is refused rather than ignored, and so is a value out of
    range: a page that sent a step count this build does not accept and was
    answered with the unchanged settings and no error would show the old value
    with no explanation of why its press did nothing. ``"chat"`` is a set of
    its own keys, merged into what is stored and validated the same way. A
    precision or LoRA offered for a model that does not take it is refused
    with the model's name.
    """
    offered = {str(key): value for key, value in dict(values or {}).items()}
    unknown = sorted(set(offered) - set(SETTINGS_DEFAULTS))
    if unknown:
        raise VibeVoiceError(f"{unknown[0]!r} is not a VibeVoice setting.")
    chat_offered = offered.pop("chat", None)
    checked = {key: _validated(key, value) for key, value in offered.items()}
    chat_checked = {}
    if chat_offered is not None:
        if not isinstance(chat_offered, dict):
            raise VibeVoiceError("Voice Chat's VibeVoice settings are a set of named values.")
        # An unknown key is refused by _validated_chat, before anything is written.
        chat_checked = {str(key): _validated_chat(str(key), value)
                        for key, value in chat_offered.items()}
    model = checked.get("model_id") or settings()["model_id"]
    precisions, takes_lora = _model_takes(model)
    if "precision" in checked and checked["precision"] not in precisions:
        raise VibeVoiceError(f"{_model_label(model)} runs in "
                             f"{_precision_words(precisions)} only.")
    if checked.get("lora_id") and not takes_lora:
        raise VibeVoiceError(f"{_model_label(model)} takes no LoRA.")
    current = _settings_read()
    current.update(checked)
    if chat_checked:
        chat = dict(current.get("chat")) if isinstance(current.get("chat"), dict) else {}
        chat.update(chat_checked)
        current["chat"] = chat
    _settings_write(current)
    changed = sorted(checked) + [f"chat.{key}" for key in sorted(chat_checked)]
    if changed:
        logger.info("Model Chain: VibeVoice settings changed — %s", ", ".join(changed))
    return settings()


def _chat_read(stored) -> dict:
    """Voice Chat's block, defaults filled in, each stored value validated alone."""
    found = dict(CHAT_DEFAULTS)
    if not isinstance(stored, dict):
        return found
    for key in CHAT_DEFAULTS:
        if key in stored:
            try:
                found[key] = _validated_chat(key, stored[key])
            except VibeVoiceError:
                logger.debug("Model Chain: a stored Voice Chat VibeVoice setting (%s) was "
                             "ignored", key)
    return found


def _validated_chat(key: str, value):
    """One of Voice Chat's keys. The precision is one some model here runs in; which
    model a reply is spoken by is the voice's to say, and the Realtime model ignores
    precision and LoRA whatever is stored."""
    if key in ("card_uuid", "lora_id", "lora_scale"):
        return _validated(key, value)
    if key == "precision":
        found = _validated("precision", value)
        offered = set()
        for name in model_ids():
            offered.update(_model_takes(name)[0])
        if offered and found not in offered:
            raise VibeVoiceError(f"No VibeVoice model here runs in {PRECISION_LABELS[found]}.")
        return found
    raise VibeVoiceError(f"{key!r} is not one of Voice Chat's VibeVoice settings.")


def _model_takes(identifier: str) -> tuple:
    """``(precisions, takes_lora)`` for ``identifier``; full precision and no LoRA for a
    model this build cannot read, which is the one shape every model has."""
    try:
        entry = bundle(identifier)
    except VibeVoiceError:
        return (PRECISION_DEFAULT,), False
    return tuple(entry.precisions), bool(entry.lora)


def _precision_words(precisions) -> str:
    names = [PRECISION_LABELS.get(name, name) for name in precisions]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


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
    if key == "precision":
        text = str(value or "").strip().lower()
        if text not in PRECISIONS:
            raise VibeVoiceError(f"Precision is {_precision_words(PRECISIONS)} "
                                 f"({', '.join(PRECISIONS)}).")
        return text
    if key == "lora_id":
        text = str(value or "").strip()
        if not text:
            return ""
        if not _lora_known(text):
            raise VibeVoiceError("That LoRA is not in the library.")
        return text
    if key == "lora_scale":
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        if isinstance(value, bool) or not LORA_SCALE_MIN <= number <= LORA_SCALE_MAX:
            raise VibeVoiceError(f"LoRA strength is a number from {LORA_SCALE_MIN:.1f} to "
                                 f"{LORA_SCALE_MAX:.1f}.")
        return round(number, 3)
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
    """``{"<model id>@<precision>": {"peak_bytes", "rss_bytes", "renders", "updated"}}``.

    What renders have cost on this machine, per model *and precision*, because an
    8-bit 7B and a full one are different amounts of card. A key with no
    precision in it is one an earlier build wrote, before there were precisions,
    and is read as bf16 -- the only precision there was.
    """
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


def calibration_key(identifier: str, precision: str = "") -> str:
    """Where a render of ``identifier`` at ``precision`` is recorded."""
    return f"{identifier}@{str(precision or '').strip().lower() or PRECISION_DEFAULT}"


def _observed(identifier: str, precision: str, measure: str) -> int:
    found = calibration()
    values = [int((found.get(calibration_key(identifier, precision)) or {}).get(measure) or 0)]
    if precision == PRECISION_DEFAULT:
        values.append(int((found.get(identifier) or {}).get(measure) or 0))
    return max(values)


def note_peak(identifier: str, peak_bytes: int, rss_bytes: int = 0,
              precision: str = "") -> None:
    """Remember the most a render has cost on this machine. Never raises.

    Called after every render with torch's peak reserved bytes, the worker's
    resident set when there is one, and the precision the model was loaded at
    (blank is bf16). The highest of each is kept under
    :func:`calibration_key`, and written atomically: this file is read before
    every turn request, and a half-written one would be an estimate nobody
    made. A precision this build does not know is ignored rather than recorded
    under a key nothing will ever read.
    """
    name = str(identifier or "").strip()
    wanted = str(precision or "").strip().lower() or PRECISION_DEFAULT
    try:
        peak = max(int(peak_bytes or 0), 0)
        rss = max(int(rss_bytes or 0), 0)
    except (TypeError, ValueError):
        return
    if not name or len(name) > 64 or not all(c.isalnum() or c in "-_." for c in name) \
            or wanted not in PRECISIONS or (peak <= 0 and rss <= 0):
        return
    key = calibration_key(name, wanted)
    with _lock:
        found = _read_json(paths.vibevoice_calibration_path()) or {}
        entries = found.get("models") if isinstance(found.get("models"), dict) else {}
        current = dict(entries.get(key) or {}) if isinstance(entries.get(key), dict) else {}
        current["peak_bytes"] = max(int(current.get("peak_bytes") or 0), peak)
        current["rss_bytes"] = max(int(current.get("rss_bytes") or 0), rss)
        current["renders"] = int(current.get("renders") or 0) + 1
        current["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entries[key] = current
        try:
            _write_json(paths.vibevoice_calibration_path(),
                        {"schema": SCHEMA, "models": entries})
        except OSError:
            logger.debug("Model Chain: could not record what a VibeVoice render cost",
                         exc_info=True)


def _precision_for(entry: Bundle, precision: str = "") -> str:
    """The precision a question about ``entry`` is asked at.

    Blank is the settings' precision when ``entry`` runs in it, else the
    model's own first -- bf16. A precision the model does not run in is a
    refusal with its name, never a silent substitute: a turn asked for too
    little card is the failure this whole estimate exists to prevent.
    """
    wanted = str(precision or "").strip().lower()
    if not wanted:
        stored = str(settings().get("precision") or "")
        return stored if stored in entry.precisions else entry.precisions[0]
    if wanted not in entry.precisions:
        raise VibeVoiceError(f"{entry.label} runs in {_precision_words(entry.precisions)} "
                             f"only.")
    return wanted


def need_vram_bytes(identifier: str = "", precision: str = "") -> int:
    """What to ask the turn for: the estimate, or the peak a render has reached.

    The estimate is the weights at that precision -- for bf16 the shards'
    committed sizes when the manifest lists them, else what earlier installs
    recorded, else the model card's figure; for 8-bit and 4-bit the manifest's
    own figure -- plus the working set. The calibration's observed peak for that
    model at that precision takes over once it is larger, because a measurement
    of this machine beats an estimate of any machine.
    """
    entry = bundle(identifier)
    wanted = _precision_for(entry, precision)
    working = int(entry.estimates.get("working_bytes") or WORKING_BYTES_DEFAULT)
    estimate = _weights_bytes(entry, wanted) + working
    return max(estimate, _observed(entry.identifier, wanted, "peak_bytes"))


def need_ram_bytes(identifier: str = "", precision: str = "") -> int:
    """The system RAM a load of ``identifier`` at ``precision`` is asked to find."""
    entry = bundle(identifier)
    wanted = _precision_for(entry, precision)
    estimate = int(entry.estimates.get("ram_bytes") or RAM_BYTES_DEFAULT)
    return max(estimate, _observed(entry.identifier, wanted, "rss_bytes"))


def _weights_bytes(entry: Bundle, precision: str = PRECISION_DEFAULT) -> int:
    """The weights at ``precision``.

    bf16: the shards' total -- committed sizes, then recorded sizes, then the
    estimate. A quantised precision: the manifest's figure for it, else the
    bf16 total scaled by :data:`QUANTISED_SHARE`.
    """
    if precision != PRECISION_DEFAULT:
        figure = (entry.estimates.get("weights_by_precision") or {}).get(precision)
        if isinstance(figure, (int, float)) and not isinstance(figure, bool) and figure > 0:
            return int(figure)
        return int(_weights_bytes(entry) * QUANTISED_SHARE.get(precision, 1.0))
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
    figure = (entry.estimates.get("weights_by_precision") or {}).get(PRECISION_DEFAULT)
    if isinstance(figure, (int, float)) and not isinstance(figure, bool) and figure > 0:
        return int(figure)
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


def refusal(manual: bool = False, model_id: str = "") -> str:
    """Why a render of ``model_id`` cannot run now, in one sentence, or ``""``.

    ``model_id`` ``""`` is the model the settings name. Asked before anything is
    started and separately from starting it, because the two questions have
    different audiences: this one answers a browser that needs a sentence, and
    the install transaction answers a log. ``manual`` means the caller can fall
    back to a folder install, so an unpinned closure is not a refusal for them.

    Each sentence names the one thing that is missing for *that* model -- the
    runtime every model shares, the model, its tokenizer, or (for the Realtime
    model) its preset voices -- because a form that clones through the 7B and
    one that speaks through the Realtime model ask the same question about
    different things.
    """
    found = _install_refusal(manual)
    if found:
        return found
    try:
        current = _status(model_id)
    except VibeVoiceError as exc:
        return str(exc)
    if not _nvidia_present():
        return ("VibeVoice needs an NVIDIA card, and this machine has no NVIDIA driver "
                "(nvidia-smi was not found).")
    if not current.runtime_installed:
        if current.runtime_stale:
            return ("VibeVoice's runtime was installed by an earlier build and has to be "
                    "installed again — install it below.")
        return "VibeVoice's runtime is not installed — install it below."
    label = current.model_label or LABEL
    if not current.model_installed:
        entry = bundle(current.model_id)
        root = paths.vibevoice_model_root(entry.identifier)
        missing = _model_missing(entry, root, current.installed_model)
        if current.installed_model and missing:
            return f"The {label} model is missing {missing[0]} — install it again below."
        return f"The {label} model is not installed — install it below."
    if not current.tokenizer_installed:
        return f"The {label} model's tokenizer is not installed — install the model again."
    if not current.presets_ready:
        return (f"None of the {label} model's preset voices are installed — install the "
                f"model again.")
    return ""


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #


PARTS = ("runtime", "model")


def sources(part: str = "runtime", model_id: str = "") -> list:
    """Where a person would go to fetch VibeVoice's files by hand.

    The runtime's wheels, its torch wheel and the five overlay files (saved under
    their own names, anywhere in the folder); or one model's files, its weights,
    its tokenizer in a subfolder and, for the Realtime model, its preset voices
    in ``voices/``.
    """
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
        found += [{"filename": item.basename, "url": item.artifact.url,
                   "save_as": item.basename, "archive": False} for item in overlay_files()]
        return found
    entry = bundle(model_id)
    weights = list(entry.shards) or ([entry.single] if entry.single is not None else [])
    found = [{"filename": item.filename, "url": item.url, "save_as": item.local_name,
              "archive": False} for item in tuple(entry.artifacts) + tuple(weights)]
    found += [{"filename": item.filename, "url": item.url,
               "save_as": f"{entry.tokenizer.dirname}/{item.local_name}", "archive": False}
              for item in entry.tokenizer.artifacts]
    found += [{"filename": item.filename, "url": item.artifact.url,
               "save_as": f"{paths.VIBEVOICE_VOICES_DIRNAME}/{item.filename}",
               "archive": False} for item in entry.voices]
    return found


def install(part: str = "", on_status=None, on_progress=None, model_id: str = "") -> Status:
    """Install what is missing: the runtime, a model with its tokenizer, or both.

    ``part`` ``"runtime"``, ``"model"`` or ``""`` for both; ``model_id`` names
    the model (``""`` is the one the settings name). A transaction per part.
    Nothing outside a staging directory is touched until every byte has arrived
    and matched what was expected, each part is promoted by a directory rename,
    and a failure leaves the installation as it was.
    """
    wanted = str(part or "").strip().lower()
    if wanted and wanted not in PARTS:
        raise VibeVoiceError("VibeVoice installs its runtime or a model (with its "
                             "tokenizer).")
    entry = bundle(model_id)
    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)
    with models._claim(KIND, say, entry.identifier):
        if wanted in ("", "runtime"):
            say("Checking the VibeVoice runtime…")
            share = 0.15 if not wanted else 1.0
            install_runtime(on_status=say, on_progress=lambda f: tick(f * share))
        if wanted in ("", "model"):
            base = 0.15 if not wanted else 0.0
            install_model(on_status=say,
                          on_progress=lambda f: tick(base + f * (1.0 - base)),
                          model_id=entry.identifier)
        tick(1.0)
        say("VibeVoice installed.")
        return status(entry.identifier)


def install_from(part: str, folder: str, on_status=None, on_progress=None,
                 model_id: str = "") -> Status:
    """Install from files already on this machine. The escape hatch.

    A folder of the runtime's wheels (the pinned ones and the torch wheel) and
    the overlay files, or a folder holding a model's files -- config, index or
    single weights, shards, optionally the processor config, a tokenizer
    subfolder and, for the Realtime model, its preset voices. A pinned artifact
    is checked against the hash committed here; an unpinned one has its digest
    recorded and becomes the constant the next install is checked against.
    """
    wanted = str(part or "").strip().lower()
    if wanted not in PARTS:
        raise VibeVoiceError("VibeVoice installs its runtime or a model (with its "
                             "tokenizer).")
    entry = bundle(model_id)
    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)
    with models._claim(KIND, say, entry.identifier):
        if wanted == "runtime":
            install_runtime(on_status=say, on_progress=tick, folder=folder)
        else:
            install_model(on_status=say, on_progress=tick, folder=folder,
                          model_id=entry.identifier)
        return status(entry.identifier)


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
    """The isolated CUDA closure: an interpreter of its own, the wheels unpacked, the overlay.

    Built the way every voice runtime is -- a virtual environment without pip,
    the verified wheels unpacked rather than installed by a package manager --
    with one wheel more than the manifest can name: torch, resolved from the
    publisher's index on this machine (:func:`_resolve_torch`). Then the five
    overlay files, each checked against its committed digest, are written over
    the unpacked package (:func:`_apply_overlay`), and only then is the staged
    runtime asked to import itself: the self-test runs against the code the
    worker will run.
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    chosen = platform()
    if chosen is None:
        raise VibeVoiceError(_install_refusal() or "VibeVoice has no runtime for this platform.")
    if not chosen.artifacts:
        raise VibeVoiceError(_install_refusal(manual=bool(folder))
                             or "This build has not recorded a VibeVoice runtime closure.")
    overlay = overlay_files()
    if not overlay:
        raise VibeVoiceError("This build has not recorded the VibeVoice runtime's overlay "
                             "(the streaming model's code), so it installs no runtime. This "
                             "is a problem with the extension rather than your installation.")
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
    patches = staging / "overlay"
    shutil.rmtree(staging, ignore_errors=True)
    wheels.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            source = _folder(folder)
            _adopt(chosen.artifacts, source, wheels, say, "VibeVoice runtime wheel")
            torch_item = _adopt_torch(resolve, source, wheels, say)
            _adopt_overlay(overlay, source, patches, say)
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
            _fetch_overlay(overlay, patches, say)
        digest = _digest(wheels / torch_item.local_name)
        torch_bytes = (wheels / torch_item.local_name).stat().st_size
        say("Building the isolated VibeVoice runtime…")
        _build_environment(staging, wheels, models.RuntimePlatform(
            identifier=chosen.identifier, system=chosen.system, machines=chosen.machines,
            python=chosen.python, artifacts=tuple(chosen.artifacts) + (torch_item,)))
        _apply_overlay(staging, patches, overlay, say)
        tick(0.85)
        say("Checking that VibeVoice imports on this machine…")
        report = _smoke_test(staging)
        tick(0.95)
        shutil.rmtree(wheels, ignore_errors=True)
        shutil.rmtree(patches, ignore_errors=True)
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
                            "bytes": int(torch_bytes or torch_item.size or 0),
                            "index": resolve["index"]},
            "artifacts": {item.local_name: item.sha256 for item in chosen.artifacts},
            "overlay": {item.path: item.artifact.sha256 for item in overlay},
            "overlay_source": {"repository": overlay[0].repository,
                               "commit": overlay[0].commit},
            # What the self-test found it could do. Reported rather than
            # required, as the worker reports them: a runtime whose PEFT or
            # bitsandbytes will not import still renders at full precision.
            "features": {name: report.get(name) for name in ("realtime", "lora",
                                                             "quantisation")},
            "peft_version": str(report.get("peft") or ""),
            "bitsandbytes_version": str(report.get("bitsandbytes") or ""),
            "license": runtime.get("license") or "",
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        models._promote(staging, paths.vibevoice_runtime_root())
        say("The VibeVoice runtime is installed.")
        tick(1.0)
        logger.info("Model Chain: the VibeVoice runtime is installed — %s, torch %s (%s), "
                    "with %d overlay file(s) from %s at %s", chosen.identifier,
                    report.get("torch") or full_torch_version(resolve), torch_item.filename,
                    len(overlay), overlay[0].repository or "Microsoft's repository",
                    (overlay[0].commit or "?")[:12])
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _fetch_overlay(items, destination: Path, say) -> None:
    """Download each overlay file, keeping it only if it is the committed bytes.

    Through the shared downloader, so a file that arrives at the wrong size or
    with the wrong digest is thrown away rather than written anywhere near the
    runtime. raw.githubusercontent.com states no digest of its own; the one
    checked is this repository's.
    """
    for index, item in enumerate(items, start=1):
        say(f"Fetching {item.basename} from Microsoft's VibeVoice repository ({index} of "
            f"{len(items)})…")
        models._download(item.artifact, destination / item.path, lambda _received: None,
                         models.Expected(item.artifact.size, item.artifact.sha256,
                                         "this extension's manifest"))


def _adopt_overlay(items, source: Path, destination: Path, say) -> None:
    """The overlay out of a folder somebody filled themselves.

    Each file is looked for under its path in the package (a copy of the
    repository's tree, or a clone of it), then under its own name anywhere a few
    levels down, and the first candidate whose bytes are the committed ones is
    taken. A file of the right name with other contents -- the wheel's own
    ``configuration_vibevoice.py``, say -- is passed over, and only refused when
    nothing else matches.
    """
    for item in items:
        candidates = _overlay_candidates(source, item)
        chosen = None
        for candidate in candidates:
            try:
                if candidate.stat().st_size == int(item.artifact.size or 0) \
                        and _digest(candidate) == item.artifact.sha256:
                    chosen = candidate
                    break
            except OSError:
                continue
        if chosen is None:
            if candidates:
                raise VibeVoiceError(
                    f"{item.basename} is in that folder, but its contents are not the ones "
                    f"this extension pins (Microsoft's VibeVoice repository at commit "
                    f"{item.commit[:12] or '?'}). Nothing was installed.")
            raise VibeVoiceError(f"{item.basename} is not in {source}. It comes from "
                                 f"{item.artifact.url}. Nothing was installed.")
        say(f"Taking {item.path} from {chosen.parent}…")
        target = destination / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chosen, target)


_OVERLAY_SEARCH_DEPTH = 6
_OVERLAY_SEARCH_FILES = 50000


def _overlay_candidates(source: Path, item) -> list:
    """Every file under ``source`` that could be ``item``, most likely first."""
    found = []
    for candidate in (source / item.path, source / "VibeVoice" / item.path,
                      source / item.basename):
        if candidate.is_file() and candidate not in found:
            found.append(candidate)
    wanted = item.basename.casefold()
    seen = 0
    base = len(source.parts)
    for folder, children, files in os.walk(source):
        depth = len(Path(folder).parts) - base
        children[:] = sorted(name for name in children
                             if not name.startswith(".") and depth < _OVERLAY_SEARCH_DEPTH)
        for name in sorted(files):
            seen += 1
            if name.casefold() == wanted:
                candidate = Path(folder) / name
                if candidate not in found:
                    found.append(candidate)
        if seen > _OVERLAY_SEARCH_FILES:
            break
    return found


def _apply_overlay(staging: Path, patches: Path, items, say) -> None:
    """Write each overlay file over the unpacked package, and check what was written.

    Only ever inside ``site-packages/vibevoice``: a path the manifest could name
    is already held to one module of the package (:data:`_OVERLAY_PATH`), and the
    resolved target is checked again here, because this is the one place this
    installer writes code it did not unpack from a wheel.
    """
    target = models.site_packages(staging / "env")
    package = (target / "vibevoice").resolve()
    if not package.is_dir():
        raise VibeVoiceError("The staged VibeVoice runtime has no vibevoice package to "
                             "complete. Nothing was installed.")
    say(f"Writing {len(items)} files from Microsoft's VibeVoice repository over the "
        f"package…")
    for item in items:
        where = (target / item.path).resolve()
        try:
            where.relative_to(package)
        except ValueError:
            raise VibeVoiceError(f"The overlay file {item.path} would be written outside the "
                                 f"vibevoice package. Nothing was installed.") from None
        origin = patches / item.path
        if not origin.is_file():
            raise VibeVoiceError(f"{item.basename} is missing from the staged download. "
                                 f"Nothing was installed.")
        where.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, where)
        if _digest(where) != item.artifact.sha256:
            raise VibeVoiceError(f"{item.path} was written into the staged runtime and did "
                                 f"not read back as the pinned bytes. Nothing was installed.")
    logger.info("Model Chain: VibeVoice's overlay was written over the package — %s",
                ", ".join(item.path for item in items))


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


def install_model(on_status=None, on_progress=None, folder=None, model_id: str = "") -> None:
    """A model's files, its weights (from its own index, or its one file), its
    tokenizer and, for the Realtime model, its preset voices.

    One staging directory and one promote for all of it, so a model is either
    there with its tokenizer, its voices and its local configuration or not
    there at all. Two things are repaired in place instead: the processor
    configuration this module writes itself, and preset voices that went
    missing from an otherwise complete model -- a few megabytes each, not a
    reason to fetch two gigabytes of weights again.
    """
    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    entry = bundle(model_id)
    target = paths.vibevoice_model_root(entry.identifier)
    marker = _read_json(target / paths.INSTALLED_FILENAME)
    missing = _model_missing(entry, target, marker)
    absent = [item for item in entry.voices if not _preset_present(entry, item, target)]
    tokenizer = _tokenizer_present(entry, target)
    if marker and not missing and tokenizer and not absent:
        say(f"{entry.label} is already installed.")
        tick(1.0)
        return
    if marker and set(missing) <= {paths.VIBEVOICE_LOCAL_CONFIG} and tokenizer:
        # Everything is in place but what this module writes itself, and
        # perhaps some of the preset voices.
        if missing:
            say(f"Writing {entry.label}'s processor configuration…")
            _write_local_config(entry)
        if absent:
            _repair_presets(entry, absent, target, marker, folder, say, tick)
        tick(1.0)
        return

    _stop_workers_holding(entry.identifier, "the VibeVoice model is being installed")
    staging = paths.vibevoice_staging_for(entry.identifier, uuid.uuid4().hex[:8])
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            record = _adopt_model(entry, _folder(folder), staging, say, tick)
        else:
            record = _download_model(entry, staging, say, tick)
        shards = list(record["shards"])
        voices = [f"{paths.VIBEVOICE_VOICES_DIRNAME}/{item.filename}" for item in entry.voices]
        required = (tuple(entry.required_paths) + tuple(shards) + tuple(entry.tokenizer.paths)
                    + tuple(voices))
        say(f"Checking {entry.label} is complete…")
        _sanity_check(staging, required, entry.label)
        _write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "id": entry.identifier,
            "label": entry.label,
            "kind": entry.kind,
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
            "weights": record.get("weights", "index"),
            "tokenizer": {"repo": entry.tokenizer.repo, "revision": entry.tokenizer.revision,
                          "dirname": entry.tokenizer.dirname},
            "voices": [item.identifier for item in entry.voices],
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
        logger.info("Model Chain: the VibeVoice model %s is installed — %d shard(s) from %s%s",
                    entry.identifier, len(shards), record["repo"],
                    f", {len(entry.voices)} preset voices" if entry.voices else "")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _new_record(entry: Bundle, source: str) -> dict:
    return {"digests": {}, "sizes": {}, "verified": {}, "repos": {}, "shards": [],
            "repo": entry.repo, "source": source, "weights": "index"}


INDEX_FILENAME = "model.safetensors.index.json"


def _download_model(entry: Bundle, staging: Path, say, tick) -> dict:
    """Config and index first, then the weights the index names (or the one weights
    file a model without an index publishes), then the tokenizer, then the voices."""
    record = _new_record(entry, "hub")
    heads = [item for item in entry.artifacts if item.local_name not in entry.optional]
    extras = [item for item in entry.artifacts if item.local_name in entry.optional]
    _fetch(entry, heads, staging, say, tick, 0.0, 0.02, record)
    for item in extras:
        try:
            _fetch(entry, [item], staging, say, tick, 0.02, 0.0, record)
        except models.VoiceError as exc:
            if item.filename == INDEX_FILENAME:
                say(f"{entry.label} publishes no {INDEX_FILENAME}; its weights are one file "
                    f"({exc}).")
                logger.info("Model Chain: VibeVoice's %s has no safetensors index; taking its "
                            "single weights file", entry.identifier)
                continue
            say(f"{item.filename} is not published by the mirror; the defaults will be "
                f"written ({exc}).")
            logger.info("Model Chain: VibeVoice's mirror has no %s; the local processor "
                        "configuration is written from defaults", item.filename)
    shards = _weights_named(entry, staging, record)
    if record["weights"] == "single":
        say(f"{entry.label}'s weights are one file, {shards[0]}.")
    else:
        say(f"The index names {len(shards)} shard{'s' if len(shards) != 1 else ''}.")
    _fetch(entry, _shard_artifacts(entry, shards), staging, say, tick, 0.02, 0.88, record)
    tokenizer_root = staging / entry.tokenizer.dirname
    _fetch(entry, list(entry.tokenizer.artifacts), tokenizer_root, say, tick, 0.90, 0.03,
           record, prefix=entry.tokenizer.dirname + "/")
    if entry.voices:
        say(f"Fetching {entry.label}'s {len(entry.voices)} preset voices…")
        _fetch(entry, [item.artifact for item in entry.voices],
               staging / paths.VIBEVOICE_VOICES_DIRNAME, say, tick, 0.93, 0.05, record,
               prefix=paths.VIBEVOICE_VOICES_DIRNAME + "/")
    used = [record["repos"].get(name) for name in shards]
    record["repo"] = next((repo for repo in used if repo), entry.repo)
    return record


def _weights_named(entry: Bundle, staging: Path, record: dict) -> list:
    """The weights files: the index's shards, or the model's one file when it has
    no index and the manifest says that is how it ships."""
    if (staging / INDEX_FILENAME).is_file():
        shards = _shards_named(_read_index(staging / INDEX_FILENAME))
        record["weights"] = "index"
    elif entry.single is not None and entry.index_optional:
        shards = [entry.single.local_name]
        record["weights"] = "single"
    else:
        raise VibeVoiceError(f"{entry.label} arrived without {INDEX_FILENAME}, so its weights "
                             f"cannot be listed. Nothing was installed.")
    record["shards"] = list(shards)
    return list(shards)


def _repair_presets(entry: Bundle, wanted, target: Path, marker: dict, folder, say,
                    tick) -> None:
    """Put back the preset voices an otherwise complete model is missing.

    Each is checked against its committed digest in a staging directory and only
    then moved into place, and the model's record says what was put back.
    """
    staging = paths.vibevoice_staging_for(f"{entry.identifier}-voices", uuid.uuid4().hex[:8])
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    record = _new_record(entry, "local" if folder else "hub")
    try:
        say(f"{len(wanted)} of {entry.label}'s preset voices are missing; putting "
            f"{'it' if len(wanted) == 1 else 'them'} back…")
        _gather_presets(entry, wanted, _folder(folder) if folder else None, staging, say,
                        tick, record)
        names = [f"{paths.VIBEVOICE_VOICES_DIRNAME}/{item.filename}" for item in wanted]
        _sanity_check(staging, names, entry.label)
        (target / paths.VIBEVOICE_VOICES_DIRNAME).mkdir(parents=True, exist_ok=True)
        for name in names:
            os.replace(staging / name, target / name)
        updated = dict(marker or {})
        for key in ("digests", "bytes", "verified_by"):
            updated[key] = dict(updated.get(key) or {})
        for name in names:
            updated["digests"][name] = record["digests"].get(name, "")
            updated["bytes"][name] = record["sizes"].get(name, 0)
            updated["verified_by"][name] = record["verified"].get(name, "")
        updated["voices"] = [item.identifier for item in entry.voices]
        _write_json(target / paths.INSTALLED_FILENAME, updated)
        say(f"{entry.label}'s preset voices are back.")
        logger.info("Model Chain: VibeVoice put back %d preset voice(s) of %s", len(names),
                    entry.identifier)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _gather_presets(entry: Bundle, wanted, source, staging: Path, say, tick,
                    record: dict) -> None:
    """``wanted`` preset voices into ``staging/voices``: from ``source`` when it has
    them (checked against their pins), fetched from Microsoft's repository at the
    pinned commit otherwise."""
    destination = staging / paths.VIBEVOICE_VOICES_DIRNAME
    destination.mkdir(parents=True, exist_ok=True)
    prefix = paths.VIBEVOICE_VOICES_DIRNAME + "/"
    remaining = list(wanted)
    if source is not None:
        found = _preset_files(source)
        taken = []
        for item in remaining:
            origin = found.get(item.filename.casefold())
            if origin is None:
                continue
            digest = _digest(origin)
            if item.artifact.sha256 and digest != item.artifact.sha256:
                raise VibeVoiceError(
                    f"{item.filename} is in that folder, but its contents are not the preset "
                    f"voice this extension pins. Nothing was installed.")
            shutil.copyfile(origin, destination / item.filename)
            record["digests"][prefix + item.filename] = digest
            record["sizes"][prefix + item.filename] = origin.stat().st_size
            record["verified"][prefix + item.filename] = "this extension's manifest"
            record["repos"][prefix + item.filename] = ""
            taken.append(item)
        if taken:
            say(f"Took {len(taken)} preset voice{'s' if len(taken) != 1 else ''} from the "
                f"folder.")
        remaining = [item for item in remaining if item not in taken]
    if not remaining:
        return
    try:
        _fetch(entry, [item.artifact for item in remaining], destination, say, tick, 0.93,
               0.05, record, prefix=prefix)
    except models.VoiceError as exc:
        where = (source / paths.VIBEVOICE_VOICES_DIRNAME) if source is not None else \
            "a voices folder beside the model's files"
        commit = _voices_commit(entry)
        raise VibeVoiceError(
            f"{entry.label}'s preset voices could not be fetched ({exc}). Put the .pt files "
            f"from demo/voices/streaming_model in Microsoft's VibeVoice repository"
            f"{' at commit ' + commit[:12] if commit else ''} in {where} and try again. "
            f"Nothing was installed.") from None


def _voices_commit(entry: Bundle) -> str:
    source = entry.raw.get("voices_source") if isinstance(entry.raw.get("voices_source"),
                                                          dict) else {}
    return str(source.get("commit") or "")


def _preset_files(source: Path) -> dict:
    """``{lower-case file name: path}`` of the ``.pt`` files where a person would have
    put preset voices: a ``voices`` folder, upstream's own tree, or the folder itself."""
    found = {}
    folders = [source / paths.VIBEVOICE_VOICES_DIRNAME,
               source / "demo" / "voices" / "streaming_model",
               source / "VibeVoice" / "demo" / "voices" / "streaming_model",
               source / "streaming_model", source]
    for folder in folders:
        if not folder.is_dir():
            continue
        for child in sorted(folder.iterdir()):
            if child.is_file() and child.suffix.casefold() == ".pt":
                found.setdefault(child.name.casefold(), child)
    return found


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
    if entry.single is not None and list(shards) == [entry.single.local_name]:
        return [entry.single]
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
    if isinstance(entry.raw.get("single_weights"), dict):
        rows.append(entry.raw["single_weights"])
    rows += [{"filename": f"{row.get('id')}.pt", "sha256": row.get("sha256"),
              "bytes": row.get("bytes")}
             for row in (entry.raw.get("voices") or ()) if isinstance(row, dict)]
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
            if item.filename == INDEX_FILENAME:
                say(f"{item.filename} is not in the folder; {entry.label}'s weights are one "
                    f"file.")
            else:
                say(f"{item.filename} is not in the folder; the defaults will be written.")
            continue
        record_digests(record, _adopt([item], found.parent, staging, say, entry.label),
                       staging)
    shards = _weights_named(entry, staging, record)
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
        say(f"The folder has no tokenizer; fetching {entry.tokenizer.repo}'s four tokenizer "
            f"files…")
        try:
            _fetch(entry, list(entry.tokenizer.artifacts), tokenizer_root, say, tick, 0.9,
                   0.03, record, prefix=entry.tokenizer.dirname + "/")
        except models.VoiceError as exc:
            raise VibeVoiceError(
                f"{entry.label}'s tokenizer could not be fetched ({exc}). Put "
                f"tokenizer.json, tokenizer_config.json, vocab.json and merges.txt from "
                f"huggingface.co/{entry.tokenizer.repo} in {source / entry.tokenizer.dirname} "
                f"and try again. Nothing was installed.") from None
    if entry.voices:
        _gather_presets(entry, entry.voices, source, staging, say, tick, record)
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
        if name.startswith(paths.VIBEVOICE_VOICES_DIRNAME + "/"):
            continue  # every preset voice is pinned in the manifest itself
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
        elif name.endswith(".pt"):
            # A voice prompt is a torch.save archive, which is a zip.
            with open(path, "rb") as handle:
                head = handle.read(4)
            if head != b"PK\x03\x04":
                raise VibeVoiceError(f"{name} is not a saved voice prompt. Nothing was "
                                     f"installed.")
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


def _stop_workers_holding(identifier: str, reason: str) -> None:
    """Stop only the workers that hold ``identifier``, before its files are replaced.

    Two models share one runtime now, and installing the Realtime model must not
    end a render the 7B is halfway through on another card. A runtime that can
    say what each worker holds (``loaded(card)``) is asked; one that cannot is
    stopped whole, which is what installing a model always did. Never raises.
    """
    try:
        import mc_voice_vibevoice_runtime as runtime
    except Exception:
        return
    ask = getattr(runtime, "loaded", None)
    if not callable(ask):
        _stop_runtime(reason)
        return
    try:
        cards = dict((runtime.status() or {}).get("cards") or {})
    except Exception:
        logger.debug("Model Chain: could not read which VibeVoice workers are running",
                     exc_info=True)
        cards = {}
    for key, card in cards.items():
        uuid_text = str((card or {}).get("uuid") or key)
        try:
            held = ask(uuid_text)
        except Exception:
            held = None
        if isinstance(held, dict) and str(held.get("model_id") or "") == identifier:
            try:
                runtime.stop(uuid_text, reason)
            except Exception:
                logger.debug("Model Chain: could not stop VibeVoice on %s before %s",
                             uuid_text, reason, exc_info=True)


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
# The LoRA library
# --------------------------------------------------------------------------- #


def loras() -> list:
    """Every adapter in the library, oldest first: ``[{"id", "name", "base", "bytes",
    "parts", "created", "adapter"}]``.

    Read from disk on every call. A directory without its record or without the
    language model's adapter -- an add that was interrupted, a hand-edited
    folder -- is left out rather than offered as something that would fail.
    """
    root = paths.vibevoice_loras_root()
    found = []
    try:
        children = sorted(root.iterdir()) if root.is_dir() else []
    except OSError:
        children = []
    for folder in children:
        if not folder.is_dir() or not _SAFE_LORA.match(folder.name):
            continue
        meta = _read_json(folder / paths.VIBEVOICE_LORA_META)
        if not meta or not _adapter_pair(folder):
            continue
        found.append(_lora_record(folder.name, meta))
    found.sort(key=lambda item: (item["created"], item["name"].casefold()))
    return found


def lora_dir(identifier: str) -> Path:
    """Where one adapter of the library is. Validated, never a caller's path."""
    text = str(identifier or "").strip()
    if not _lora_known(text):
        raise VibeVoiceError("That LoRA is not in the library.")
    return paths.vibevoice_lora_root(text)


def add_lora(folder: str, name: str = "") -> dict:
    """Copy a LoRA from a folder on this PC into the library, after checking it.

    Accepted: a PEFT LoRA adapter for the 7B's language model --
    ``adapter_config.json`` saying ``"peft_type": "LORA"`` and
    ``adapter_model.safetensors`` or ``adapter_model.bin`` -- at the folder's
    root or in its ``lora/`` or ``language_model/`` subfolder, with
    ``diffusion_head/``, ``acoustic_connector/`` and ``semantic_connector/``
    copied when present beside it (each holding a full state dict or a PEFT
    adapter pair). Anything else, or more than :data:`LORA_BYTES_MAX` in all, is
    refused with a sentence and nothing is written. The copy is normalised so
    the language model's adapter is at the root of its directory, and lands by
    one rename, so the library never holds half an adapter.
    """
    source = _lora_source(folder)
    adapter, config = _find_adapter(source)
    files = {name_: adapter / name_ for name_ in _adapter_pair(adapter)}
    parts = ["llm"]
    extras = {}
    for part in LORA_PARTS[1:]:
        found = _find_extra(part, adapter, source)
        if found is not None:
            extras[part] = found
            parts.append(part)
    total = sum(path.stat().st_size for path in files.values())
    total += sum(path.stat().st_size for chosen in extras.values() for path in chosen.values())
    if total > LORA_BYTES_MAX:
        raise VibeVoiceError(f"That LoRA is {models._bytes_label(total)}, and one in the "
                             f"library is at most {models._bytes_label(LORA_BYTES_MAX)}. "
                             f"Nothing was added.")
    wanted = str(name or "").strip()
    derived = not wanted
    if derived:
        # The folder's own name, or its parent's when the folder given is the
        # adapter's conventional subfolder rather than the training run.
        wanted = (source.parent.name if source.name.casefold() in ("lora", "language_model")
                  else source.name) or "LoRA"
    with _lora_lock:
        label = _lora_name(wanted, derived=derived)
        identifier = uuid.uuid4().hex[:12]
        root = paths.vibevoice_loras_root()
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f".adding-{identifier}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            for target, origin in files.items():
                shutil.copyfile(origin, staging / target)
            for part, chosen in extras.items():
                (staging / part).mkdir()
                for target, origin in chosen.items():
                    shutil.copyfile(origin, staging / part / target)
            copied = sum(item.stat().st_size for item in staging.rglob("*") if item.is_file())
            meta = {
                "schema": SCHEMA,
                "id": identifier,
                "name": label,
                "base": MODEL_7B,
                "parts": parts,
                "bytes": copied,
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_folder": str(source),
                "adapter": {key: config.get(key) for key in
                            ("base_model_name_or_path", "r", "lora_alpha", "target_modules")},
            }
            _write_json(staging / paths.VIBEVOICE_LORA_META, meta)
            os.replace(staging, paths.vibevoice_lora_root(identifier))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    logger.info("Model Chain: VibeVoice added the LoRA “%s” (%s, %s) from %s", label,
                ", ".join(parts), models._bytes_label(copied), source)
    return _lora_record(identifier, meta)


def rename_lora(identifier: str, name: str) -> dict:
    """Give an adapter in the library a new name. Names are unique, ignoring case."""
    with _lora_lock:
        folder = lora_dir(identifier)
        meta = _read_json(folder / paths.VIBEVOICE_LORA_META) or {}
        label = _lora_name(name, derived=False, keep=folder.name)
        meta["name"] = label
        _write_json(folder / paths.VIBEVOICE_LORA_META, meta)
    logger.info("Model Chain: VibeVoice renamed a LoRA to “%s”", label)
    return _lora_record(folder.name, meta)


def delete_lora(identifier: str) -> dict:
    """Remove an adapter from the library, and from any setting that chose it.

    Returns the adapter as it was. The settings that named it read as no LoRA
    from then on, and the stored value is cleared too, so a later adapter can
    never inherit a choice somebody made for this one.
    """
    with _lora_lock:
        folder = lora_dir(identifier)
        meta = _read_json(folder / paths.VIBEVOICE_LORA_META) or {}
        record = _lora_record(folder.name, meta)
        if not paths.vibevoice_inside(folder):
            raise VibeVoiceError("That LoRA is not in the library.")
        try:
            shutil.rmtree(folder)
        except OSError as exc:
            raise VibeVoiceError(f"The LoRA “{record['name']}” could not be removed ({exc}). "
                                 f"If VibeVoice is using it, unload it first.") from None
        with _lock:
            # Each scope that named it -- the Voice Box's default and Voice Chat's
            # -- goes back to no LoRA at full strength, in this same call.
            stored = _settings_read()
            changed = False
            if stored.get("lora_id") == record["id"]:
                stored["lora_id"] = ""
                stored["lora_scale"] = LORA_SCALE_DEFAULT
                changed = True
            chat = stored.get("chat")
            if isinstance(chat, dict) and chat.get("lora_id") == record["id"]:
                chat["lora_id"] = ""
                chat["lora_scale"] = LORA_SCALE_DEFAULT
                changed = True
            if changed:
                _settings_write(stored)
    logger.info("Model Chain: VibeVoice removed the LoRA “%s”", record["name"])
    return record


def _lora_known(identifier: str) -> bool:
    text = str(identifier or "").strip()
    if not _SAFE_LORA.match(text):
        return False
    try:
        folder = paths.vibevoice_lora_root(text)
    except ValueError:
        return False
    return (folder / paths.VIBEVOICE_LORA_META).is_file() and bool(_adapter_pair(folder))


def _lora_record(identifier: str, meta: dict) -> dict:
    parts = [part for part in (meta.get("parts") or ["llm"]) if part in LORA_PARTS]
    adapter = meta.get("adapter") if isinstance(meta.get("adapter"), dict) else {}
    return {
        "id": identifier,
        "name": str(meta.get("name") or identifier),
        "base": str(meta.get("base") or MODEL_7B),
        "bytes": int(meta.get("bytes") or 0),
        "parts": parts or ["llm"],
        "created": str(meta.get("created") or ""),
        "adapter": {key: adapter.get(key) for key in
                    ("base_model_name_or_path", "r", "lora_alpha", "target_modules")},
    }


def _lora_name(name, derived: bool, keep: str = "") -> str:
    """A name for the library: one line, not blank, at most :data:`LORA_NAME_MAX`
    characters, and not another adapter's. A name made from the folder is made
    unique with a number; one somebody typed is refused instead."""
    text = " ".join(str(name or "").split())
    if any(ord(character) < 32 for character in str(name or "")) and not derived:
        raise VibeVoiceError("A LoRA's name is one line of text.")
    if not text:
        raise VibeVoiceError("A LoRA needs a name.")
    if len(text) > LORA_NAME_MAX:
        if not derived:
            raise VibeVoiceError(f"A LoRA's name is at most {LORA_NAME_MAX} characters.")
        text = text[:LORA_NAME_MAX].rstrip()
    taken = {item["name"].casefold() for item in loras() if item["id"] != keep}
    if text.casefold() not in taken:
        return text
    if not derived:
        raise VibeVoiceError(f"A LoRA called “{text}” is already in the library.")
    number = 2
    while f"{text} ({number})".casefold() in taken:
        number += 1
    return f"{text} ({number})"


def _lora_source(folder) -> Path:
    text = str(folder or "").strip().strip('"')
    if not text:
        raise VibeVoiceError("Give the folder that holds the LoRA.")
    source = Path(text).expanduser()
    if source.is_file():
        source = source.parent
    if not source.is_dir():
        raise VibeVoiceError(f"{source} is not a folder this machine can read.")
    return source


def _adapter_pair(folder: Path) -> list:
    """``["adapter_config.json", <weights>]`` when ``folder`` holds a PEFT adapter pair."""
    if not (folder / ADAPTER_CONFIG).is_file():
        return []
    weights = next((name for name in ADAPTER_WEIGHTS if (folder / name).is_file()), "")
    return [ADAPTER_CONFIG, weights] if weights else []


def _find_adapter(source: Path) -> tuple:
    """``(folder, config)`` of the language model's adapter, checked, or a refusal."""
    half = None
    for folder in (source, source / "lora", source / "language_model"):
        if not folder.is_dir():
            continue
        if _adapter_pair(folder):
            return folder, _checked_adapter(folder, "the language model's adapter")
        if (folder / ADAPTER_CONFIG).is_file() and half is None:
            half = folder
    if half is not None:
        raise VibeVoiceError(f"{half} has an adapter_config.json but no adapter_model."
                             f"safetensors or adapter_model.bin beside it. Nothing was added.")
    raise VibeVoiceError(f"{source} holds no LoRA adapter: VibeVoice looks for "
                         f"adapter_config.json with adapter_model.safetensors (or .bin) in "
                         f"the folder itself, or in its lora or language_model subfolder. "
                         f"Nothing was added.")


def _checked_adapter(folder: Path, what: str) -> dict:
    """The adapter's configuration, after checking that it is a LoRA and that its
    weights are the kind of file they claim to be."""
    try:
        config = json.loads((folder / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise VibeVoiceError(f"The adapter_config.json of {what} is not readable JSON. "
                             f"Nothing was added.") from None
    if not isinstance(config, dict):
        raise VibeVoiceError(f"The adapter_config.json of {what} is not an adapter "
                             f"configuration. Nothing was added.")
    kind = str(config.get("peft_type") or "")
    if kind.upper() != "LORA":
        raise VibeVoiceError(f"{what[0].upper() + what[1:]} is a PEFT "
                             f"{kind or 'adapter of no stated type'}, and VibeVoice takes LoRA "
                             f"adapters only. Nothing was added.")
    rank = config.get("r")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0):
        raise VibeVoiceError(f"{what[0].upper() + what[1:]} gives a rank ({rank!r}) that is "
                             f"not a positive whole number. Nothing was added.")
    weights = folder / _adapter_pair(folder)[1]
    _check_weights(weights, what)
    return config


def _check_weights(path: Path, what: str) -> None:
    """Whether a weights file is the kind of file its name says: a safetensors file
    with a readable header, or a non-empty torch archive. Not a hash -- nobody has
    vouched for these bytes -- but it catches the saved error page and the LFS
    pointer a plain clone leaves behind."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            head = handle.read(8)
            if path.suffix == ".safetensors":
                if len(head) < 8:
                    raise ValueError("short")
                (declared,) = struct.unpack("<Q", head)
                if declared <= 0 or declared + 8 > size or declared > 64 * 1024 * 1024:
                    raise ValueError("header")
                header = json.loads(handle.read(declared).decode("utf-8"))
                if not isinstance(header, dict):
                    raise ValueError("header")
            elif size < SAFETENSORS_MIN or (head[:4] != b"PK\x03\x04" and head[:1] != b"\x80"):
                # torch.save writes a zip archive; a legacy one is a pickle.
                raise ValueError("torch")
    except (OSError, ValueError, UnicodeDecodeError):
        raise VibeVoiceError(f"{path.name} of {what} is not a readable weights file. Nothing "
                             f"was added.") from None


def _find_extra(part: str, adapter: Path, source: Path):
    """``{file name: path}`` of an optional part beside the adapter (or in the
    folder given), or ``None`` when there is no such folder. A folder by that name
    holding nothing VibeVoice can load is refused rather than skipped: whoever
    trained it meant that part to be used."""
    places = [adapter / part, source / part]
    if source.name.casefold() in ("lora", "language_model"):
        places.append(source.parent / part)  # the training run's folder, one up
    for folder in places:
        if not folder.is_dir():
            continue
        chosen = {}
        pair = _adapter_pair(folder)
        if pair:
            _checked_adapter(folder, f"the {part.replace('_', ' ')}'s adapter")
            chosen.update({name: folder / name for name in pair})
        for name in LORA_EXTRA_FILES:
            if (folder / name).is_file():
                _check_weights(folder / name, f"the {part.replace('_', ' ')}")
                chosen[name] = folder / name
        if not chosen:
            raise VibeVoiceError(
                f"{folder} holds none of {', '.join(LORA_EXTRA_FILES)} or a PEFT adapter, so "
                f"there is nothing of the {part.replace('_', ' ')} to load. Nothing was "
                f"added.")
        return chosen
    return None


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
