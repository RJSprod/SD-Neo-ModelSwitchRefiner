"""Resolve and pin the VibeVoice closure, its model shards and its tokenizer.

Not part of the extension and never imported by it. A maintainer's tool, run on
a machine that can reach the publishers, which is why it lives in ``tools/``
rather than in ``scripts/`` where Forge would import it at start-up. It is the
sixth of the pinners and the first whose closure is *solved* rather than copied
out: the community ``vibevoice`` package declares fifteen dependencies of which
this repository ships nine, and the nine pull in twenty more, so the list is
produced from what every publisher declares about every other package and then
written down in full, where a reviewer can read it.

    python tools/pin_vibevoice_models.py --check          # report, change nothing
    python tools/pin_vibevoice_models.py                  # solve and pin the wheels from pypi.org,
                                                          #   and the overlay and the preset voices
                                                          #   from raw.githubusercontent.com
    python tools/pin_vibevoice_models.py --keep-closure   # re-pin the triples already written
    python tools/pin_vibevoice_models.py --model          # also size and hash every model, its
                                                          #   weights and its tokenizer from
                                                          #   huggingface.co
    python tools/pin_vibevoice_models.py --torch          # also record the digest of the torch
                                                          #   wheel from download.pytorch.org

The overlay and the preset voices
---------------------------------
The runtime is the community wheel plus five files of Microsoft's own repository
at one commit (:data:`OVERLAY`), and the Realtime model's voices are the preset
voice prompts committed at that same commit (:data:`PRESETS`). Both are served by
raw.githubusercontent.com, which states no digest of its own, so every run
downloads each file and hashes what arrived; a file whose bytes no longer match
the digest already checked in is a refusal, exactly as a re-uploaded wheel is.
The commit pins the content -- a raw URL at a commit is immutable -- and the
digest is what makes that a claim this repository can check.

Exit codes: 0 when the manifest is complete, 1 when something is still
unresolved (the model unhashed, the torch digest unrecorded, or a ``--check``
with work outstanding), and 2 when a maintainer has to look at something
before running it again.

How the closure is solved, and what "solved" does not mean
----------------------------------------------------------
The roots are fixed by hand (:data:`ROOTS`): the package itself at the one
release that exists, ``transformers`` and ``accelerate`` at the versions it
pins, and ``diffusers`` at whatever release still allows ``huggingface-hub``
below 1.0 -- because transformers 4.51.3 requires that, and huggingface-hub 1.x
would drag an HTTP client into a worker that must never use one. From those
roots every declared requirement is walked with a Windows/CPython 3.13 marker
environment, extras are ignored, the names in :data:`EXCLUDED` are skipped with
the reason written beside each, and for every package the *newest* release that
satisfies every constraint collected so far and ships a usable wheel is taken.
A constraint that arrives later and is violated by an earlier choice re-chooses
that package; the loop runs to a fixed point, and then every pin is checked
against every other pin's declaration once more (:func:`requirements_met`),
which is the check that refuses a closure that contradicts itself.

That is still not "whatever pip decides today". pip resolves at install time on
the user's machine against whatever the indexes hold then; this resolves once,
here, and writes the answer into a reviewed file with a byte count and a digest
for every wheel. The installer then resolves nothing.

Torch is the exception the whole design has to accommodate. The CUDA 12.8 build
is not on PyPI; it lives only on download.pytorch.org, which the machine that
writes this manifest cannot reach. So the closure carries one *versioned
resolve* entry for it: the installer reads the publisher's own index page on the
user's machine, takes the cp313 win_amd64 wheel of exactly the pinned version
and the SHA-256 the index states, and ``--torch`` run on a machine that can
reach the index records that digest here so a later install is checked against
a committed constant as well. Torch's own declared requirements are read from
PyPI's record of the same version, because the CUDA wheel is built from the
same source and declares the same things -- which is how ``setuptools`` ends up
in the closure: torch 2.8.0 still asks for it on Python 3.12 and later.

What it will not do
-------------------
It will not change a digest that is already checked in. A disagreement between
the manifest and what a publisher serves today fails the run and prints both.

It will not touch another engine's manifest. :data:`SIBLINGS` is refused by
name, and ``--manifest`` exists so a copy can be pinned before it is committed.

It will not accept the model's licence for anybody, and it writes no
credential anywhere. Neither community mirror is gated, so no token is needed
or read.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "voice" / "managed-vibevoice-models.json"

SIBLINGS = ("managed-voice-models.json", "managed-sopro-models.json",
            "managed-cleanup-models.json", "managed-pocket-models.json",
            "managed-pipeline-models.json")
"""The manifests this tool refuses to open, by name."""

SCHEMA = 1
TIMEOUT = 120.0
USER_AGENT = "ModelChain-VibeVoice-Pinner"

PYTHON = "3.13"
PLATFORM_ID = "windows-x86_64-cp313-cu128"
"""The one platform this closure is built for.

An allowlist of one, deliberately: the first VibeVoice release is for the
user's own machine -- Windows, CPython 3.13, an NVIDIA card driven by CUDA 12.8
-- and a *combination* becomes supported when the exact closure has been
installed and self-tested on it, not when a wheel exists for it.
"""

MODEL_ID = "vibevoice-7b"
MODEL_REALTIME = "vibevoice-realtime-0.5b"

VIBEVOICE_REPOSITORY = "https://github.com/microsoft/VibeVoice"
VIBEVOICE_COMMIT = "1541f590c7099820f10ea012f48d2399282df69f"
RAW_URL = "https://raw.githubusercontent.com/microsoft/VibeVoice/{commit}/{path}"

OVERLAY = (
    ("vibevoice/modular/configuration_vibevoice.py",
     "replaces the wheel's own copy with Microsoft's: the same four configuration classes, "
     "VibeVoiceConfig gaining get_text_config() (its decoder configuration) and a to_dict() "
     "that writes a torch dtype as a string, plus _convert_dtype_to_string, which the "
     "streaming configuration imports, and a speech-recognition configuration nothing here "
     "uses"),
    ("vibevoice/modular/configuration_vibevoice_streaming.py",
     "the Realtime model's configuration, VibeVoiceStreamingConfig; the wheel has none"),
    ("vibevoice/modular/modeling_vibevoice_streaming.py",
     "the Realtime model itself, VibeVoiceStreamingModel; the wheel has none"),
    ("vibevoice/modular/modeling_vibevoice_streaming_inference.py",
     "the Realtime model's generation, VibeVoiceStreamingForConditionalGenerationInference, "
     "which streams through the wheel's own AudioStreamer"),
    ("vibevoice/processor/vibevoice_streaming_processor.py",
     "the Realtime model's processor, VibeVoiceStreamingProcessor, which reads a preset voice "
     "prompt (process_input_with_cached_prompt) instead of a recording"),
)
"""The five files of Microsoft's repository written over the unpacked wheel.

Read against both sides at the commit: the wheel (vibevoice 0.0.1) has the
long-form code and no streaming model, Microsoft's repository has the streaming
model and no long-form code, and every module they share is the same code apart
from cosmetics -- except ``configuration_vibevoice.py``, where Microsoft's is a
superset the streaming configuration imports from, and so replaces the wheel's.
"""

PRESET_LANGUAGES = {"en": "", "in": "en", "de": "", "fr": "", "it": "", "jp": "", "kr": "",
                    "nl": "", "pl": "", "pt": "", "sp": ""}
"""A preset file's prefix, and the language it speaks when that is not the prefix.

``in-Samuel`` is English with an Indian accent: it was released with the six
English voices in the commit that added the Realtime model, carries a speaker's
name as they do, and ``in`` is not among the nine experimental languages the
model's documentation names -- German, French, Italian, Japanese, Korean, Dutch,
Polish, Portuguese and Spanish, whose files are ``Spk0``/``Spk1`` and arrived in a
later commit."""

PRESETS = (
    "en-Carter_man", "en-Davis_man", "en-Emma_woman", "en-Frank_man", "en-Grace_woman",
    "en-Mike_man", "in-Samuel_man",
    "de-Spk0_man", "de-Spk1_woman", "fr-Spk0_man", "fr-Spk1_woman", "it-Spk0_woman",
    "it-Spk1_man", "jp-Spk0_man", "jp-Spk1_woman", "kr-Spk0_woman", "kr-Spk1_man",
    "nl-Spk0_man", "nl-Spk1_woman", "pl-Spk0_man", "pl-Spk1_woman", "pt-Spk0_woman",
    "pt-Spk1_man", "sp-Spk0_woman", "sp-Spk1_man",
)
"""The Realtime model's preset voices at :data:`VIBEVOICE_COMMIT`, English first.

Every ``.pt`` file in ``demo/voices/streaming_model/`` there, read from a clone
of that commit (``git ls-files``) because the listing cannot be fetched from
raw.githubusercontent.com. More voices exist behind download links in upstream's
``download_experimental_voices.sh``; they are archives on github.com, outside
this commit, and not offered."""

PRESETS_PATH = "demo/voices/streaming_model"

PURE = "pure"
BINARY = "binary"
ABI3 = "abi3"
WINDOWS = "windows"
KINDS = (PURE, BINARY, ABI3, WINDOWS)
"""How the one wheel that belongs to the platform is recognised.

``BINARY``  -- ``cp313-cp313-win_amd64``, built for exactly this interpreter.
``ABI3``    -- ``cpXY-abi3-win_amd64``, built once against the stable ABI and
               valid for every later minor (tokenizers, safetensors, psutil).
``WINDOWS`` -- ``py3-none-win_amd64``, a platform wheel with no CPython tag.
``PURE``    -- ``py3-none-any`` or ``py2.py3-none-any``, one file for everything.

Preferred in that order when a release offers more than one: a compiled build
of charset-normalizer over its pure fallback, for instance. A free-threaded
``cp313t`` wheel matches none of them and is never taken.
"""

PREFERENCE = (BINARY, ABI3, WINDOWS, PURE)

ROOTS = (
    ("vibevoice", "==0.0.1"),
    ("transformers", "==4.51.3"),
    ("accelerate", "==1.6.0"),
    ("diffusers", ""),
    ("bitsandbytes", "==0.48.2"),
    ("peft", "==0.17.1"),
)
"""Where the solve starts. Everything else is derived.

``vibevoice`` at its one release; ``transformers`` and ``accelerate`` at the
versions that release pins in its own metadata (the community code breaks on
transformers 4.56 and later, where ``DynamicCache.key_cache`` was removed); and
``diffusers`` unpinned, so the solver takes the newest release whose own
declaration fits beside transformers 4.51.3's ``huggingface-hub<1.0`` -- 0.39
at the time of writing, because 0.40 needs huggingface-hub 1.23.

``bitsandbytes`` 0.48.2 quantises the 7B's language model to 8-bit or 4-bit
NF4. Its one Windows wheel (``py3-none-win_amd64``) carries
``libbitsandbytes_cuda128.dll`` with code for sm_86 and sm_120, both of the
user's cards, read out of the downloaded wheel; it asks for torch 2.3 or later.

``peft`` 0.17.1 loads LoRA adapters. Pinned exactly, because what peft declares
says nothing true about transformers: every release from 0.17.1 to 0.21.1 asks
for ``transformers`` unversioned, and 0.18.0 does not import under 4.51.3
(``transformers.modeling_layers`` does not exist there). 0.17.1 is the release
the worker is exercised against; 0.18.1 to 0.21.1 import and pass a LoRA round
trip on a small Qwen2 too, and moving to one is a review decision, not a default.
"""

EXCLUDED = {
    "gradio": "the demo's web interface; the worker has no interface",
    "aiortc": "the demo's WebRTC streaming; the worker answers over a pipe",
    "av": "the demo's media decoding, for aiortc",
    "ml-collections": "the demo's config objects, imported nowhere in the package",
    "absl-py": "the demo's flags and logging, imported nowhere in the package",
    "librosa": "imported lazily, only to read an audio file from a path "
               "(VibeVoiceTokenizerProcessor._load_audio_from_path); the worker is handed "
               "PCM over its pipe and never names a file",
    "soundfile": "imported lazily, only to write an audio file "
                 "(VibeVoiceTokenizerProcessor.save_audio); the worker encodes its own WAV",
    "numba": "librosa's just-in-time compiler, imported nowhere in the package",
    "llvmlite": "numba's LLVM binding, imported nowhere in the package; it comes and goes "
                "with numba",
    "scipy": "declared, imported nowhere in the package; librosa's resampling uses it",
    "torchaudio": "imported nowhere in the package; the pair-pinning trap the pipeline's "
                  "cu128 entry has is not copied",
    "httpx": "declared by diffusers 0.36 and later and imported in exactly two of its "
             "files, both pipeline loaders (diffusers/pipelines/pipeline_utils.py and "
             "pipeline_loading_utils.py), to classify hub download errors; vibevoice "
             "imports diffusers.configuration_utils, diffusers.utils and "
             "diffusers.schedulers.scheduling_utils, none of which reaches them, and the "
             "worker never loads a diffusers pipeline. Read against the 0.39.0 wheel",
    "hf-xet": "huggingface-hub declares it behind a marker that spells the machine "
              "\"amd64\" in lower case while Windows reports \"AMD64\", so pip does not "
              "install it there either; it accelerates downloads the worker never makes",
}
"""Declared dependencies this closure deliberately does not ship, and why.

A table rather than an absence, so that a dependency which quietly stopped
being optional shows up as a name that is not in here instead of as a silent
omission -- :func:`requirements_met` refuses a requirement that is neither
pinned nor in this table.
"""

TORCH = {
    "index": "https://download.pytorch.org/whl/cu128/torch/",
    "package": "torch",
    "version": "2.8.0",
}
"""The one wheel that cannot be pinned from here. See the module docstring."""

MARKERS = {
    "os_name": "nt",
    "sys_platform": "win32",
    "platform_system": "Windows",
    "platform_machine": "AMD64",
    "platform_release": "10",
    "platform_version": "",
    "platform_python_implementation": "CPython",
    "implementation_name": "cpython",
    "implementation_version": f"{PYTHON}.0",
    "python_version": PYTHON,
    "python_full_version": f"{PYTHON}.0",
    "extra": "",
}
"""The marker environment every declaration is evaluated in.

``AMD64`` in upper case because that is what ``platform.machine()`` returns on
Windows and therefore what pip evaluates there; a lower-case spelling here would
pull hf-xet into a closure that pip itself would not install it into.
"""

HUB_API = "https://huggingface.co/api/models/{repo}/revision/{revision}"
HUB_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
SHA256_HEADERS = ("x-linked-etag", "x-checksum-sha256", "x-amz-meta-sha256")

_WHEEL = re.compile(
    r"^(?P<name>[A-Za-z0-9._]+)-(?P<version>[A-Za-z0-9._!+]+)"
    r"(?:-(?P<build>[0-9][A-Za-z0-9._]*))?"
    r"-(?P<python>[A-Za-z0-9._]+)-(?P<abi>[A-Za-z0-9._]+)-(?P<platform>[A-Za-z0-9._]+)"
    r"\.whl$")


class PinError(RuntimeError):
    """Something a maintainer has to look at rather than re-run."""


def _packaging():
    try:
        from packaging.markers import Marker  # noqa: F401
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.utils import canonicalize_name
        from packaging.version import InvalidVersion, Version
    except ImportError:
        raise PinError(
            "this tool needs the 'packaging' library to read what each publisher "
            "declares about the others (pip install packaging). It is not optional: "
            "without it the closure cannot be solved or checked.") from None
    return Requirement, SpecifierSet, canonicalize_name, Version, InvalidVersion


# --------------------------------------------------------------------------- #
# PyPI
# --------------------------------------------------------------------------- #


def _read_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            return json.load(answer)
    except urllib.error.HTTPError as answer:
        with contextlib.closing(answer):
            raise PinError(f"{url} answered HTTP {answer.code}") from None
    except Exception as exc:
        raise PinError(f"{url} could not be read ({exc})") from None


class PyPI:
    """The two reads the solver needs, cached, and replaceable by a test.

    ``project`` is ``/pypi/<name>/json`` -- every release with its files --
    and ``release`` is ``/pypi/<name>/<version>/json``, the only place PyPI
    states a release's own ``requires_dist``.
    """

    def __init__(self, project=None, release=None):
        self._project = project or (lambda name: _read_json(f"https://pypi.org/pypi/{name}/json"))
        self._release = release or (
            lambda name, version: _read_json(f"https://pypi.org/pypi/{name}/{version}/json"))
        self._projects = {}
        self._releases = {}

    def project(self, name: str) -> dict:
        key = name.casefold()
        if key not in self._projects:
            self._projects[key] = self._project(name)
        return self._projects[key]

    def release(self, name: str, version: str) -> dict:
        key = (name.casefold(), str(version))
        if key not in self._releases:
            self._releases[key] = self._release(name, version)
        return self._releases[key]


def classify(filename: str, python: str = PYTHON) -> str:
    """Which :data:`KINDS` a wheel filename is for this platform, or ``""``."""
    found = _WHEEL.match(str(filename or ""))
    if found is None:
        return ""
    tag = "cp" + python.replace(".", "")
    pythons = found.group("python").split(".")
    abi = found.group("abi")
    plat = found.group("platform")
    if plat == "win_amd64":
        if abi == tag and pythons == [tag]:
            return BINARY
        if abi == "abi3" and all(p.startswith("cp") and p[2:].isdigit()
                                 and int(p[2:]) <= int(tag[2:]) for p in pythons):
            return ABI3
        if abi == "none" and any(p in ("py3", tag) for p in pythons):
            return WINDOWS
        return ""
    if plat == "any" and abi == "none" and "py3" in pythons:
        return PURE
    return ""


def usable_wheel(files, python: str = PYTHON):
    """``(kind, file)`` -- the one wheel of a release this platform installs.

    The most specific kind wins. Two files of the winning kind is a refusal
    rather than a pick: quietly taking the first is how a free-threaded build
    or a build-numbered re-upload ends up pinned as the ordinary one.
    """
    kinds = {}
    for item in files or ():
        if item.get("packagetype") != "bdist_wheel" or item.get("yanked"):
            continue
        kind = classify(str(item.get("filename") or ""), python)
        if kind:
            kinds.setdefault(kind, []).append(item)
    for kind in PREFERENCE:
        found = kinds.get(kind)
        if not found:
            continue
        if len(found) > 1:
            names = ", ".join(sorted(str(item.get("filename")) for item in found))
            raise PinError(f"more than one {kind} wheel is offered for Python {python} on "
                           f"Windows x86-64: {names}")
        return kind, found[0]
    return "", None


def wheel_artifact(item: dict) -> dict:
    """One PyPI wheel as the five keys the manifest's artifact lists carry."""
    digests = item.get("digests") or {}
    sha256 = str(digests.get("sha256") or "")
    if len(sha256) != 64:
        raise PinError(f"{item.get('filename')} was served without a SHA-256")
    size = int(item.get("size") or 0)
    if size <= 0:
        raise PinError(f"{item.get('filename')} was served without a byte count")
    return {
        "filename": str(item["filename"]),
        "local_name": str(item["filename"]),
        "url": str(item["url"]),
        "bytes": size,
        "sha256": sha256.casefold(),
    }


# --------------------------------------------------------------------------- #
# The solver
# --------------------------------------------------------------------------- #


@dataclass
class Choice:
    """One package as the solver settled it."""

    name: str
    version: str
    kind: str
    file: dict
    requires: tuple = ()


def _requirements_of(release: dict) -> tuple:
    info = release.get("info") or {}
    return tuple(str(item) for item in (info.get("requires_dist") or ()))


def _applies(need, extra: str = "") -> bool:
    """Whether a declared requirement holds on this platform with no extras."""
    if need.marker is None:
        return True
    environment = dict(MARKERS)
    environment["extra"] = extra
    try:
        return bool(need.marker.evaluate(environment))
    except Exception:
        return False


def _python_allows(requires_python: str, Version, SpecifierSet, python: str = PYTHON) -> bool:
    text = str(requires_python or "").strip()
    if not text:
        return True
    try:
        return SpecifierSet(text).contains(Version(f"{python}.0"), prereleases=True)
    except Exception:
        return False


def _final_versions(project: dict, Version, InvalidVersion) -> list:
    """Every final release with at least one file, newest first."""
    found = []
    for text, files in (project.get("releases") or {}).items():
        if not files:
            continue
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease:
            continue
        if all(item.get("yanked") for item in files):
            continue
        found.append((version, text, files))
    found.sort(key=lambda row: row[0], reverse=True)
    return found


def solve(roots, pypi: PyPI, excluded: dict, say, python: str = PYTHON,
          torch: dict = None) -> dict:
    """The closure as ``{canonical name: Choice}``, or a refusal.

    See the module docstring for the rule. What is worth restating here is the
    stopping condition: a package is (re)chosen whenever a constraint arrives
    that its current choice violates, its own declarations replace the ones its
    previous version contributed, and the loop ends when nothing is queued.
    Packages that were pulled in by a version no longer chosen are pruned by
    walking the graph from the roots afterwards.

    Torch takes part as a *virtual* root: it is never chosen from PyPI -- the
    CUDA wheel comes from the publisher's index at install time -- but what
    PyPI's record of the pinned version declares is walked exactly as a chosen
    package's declarations are, so sympy, networkx, jinja2 and setuptools are
    solved for rather than discovered missing at the end.
    """
    Requirement, SpecifierSet, canonical, Version, InvalidVersion = _packaging()

    constraints: dict = {}
    chosen: dict = {}
    queue: list = []
    root_names = []
    for name, spec in roots:
        key = canonical(name)
        root_names.append(key)
        constraints.setdefault(key, []).append((SpecifierSet(spec or ""), "roots", name))
        queue.append((key, name))
    torch = dict(torch or TORCH)
    torch_key = canonical(str(torch.get("package") or "torch"))
    virtual = {torch_key: _requirements_of(pypi.release(torch["package"], torch["version"]))}
    for raw in virtual[torch_key]:
        try:
            need = Requirement(raw)
        except Exception:
            continue
        if not _applies(need):
            continue
        dep = canonical(need.name)
        if dep in excluded:
            continue
        constraints.setdefault(dep, []).append((need.specifier, torch_key, need.name))
        if all(queued != dep for queued, _ in queue):
            queue.append((dep, need.name))

    def specifier_for(key):
        combined = SpecifierSet("")
        for spec, _source, _name in constraints.get(key, ()):
            combined &= spec
        return combined

    banned: dict = {}
    exact = {canonical(name) for name, spec in roots if str(spec or "").startswith("==")}
    order: list = []

    def pick(key, name):
        wanted = specifier_for(key)
        project = pypi.project(name)
        listed = str((project.get("info") or {}).get("name") or name)
        for version, text, files in _final_versions(project, Version, InvalidVersion):
            if text in banned.get(key, ()):
                continue
            if not wanted.contains(version, prereleases=False):
                continue
            kind, item = usable_wheel(files, python)
            if item is None:
                continue
            if not _python_allows(item.get("requires_python") or "", Version, SpecifierSet,
                                  python):
                continue
            return listed, text, kind, item
        raise PinError(f"{listed} has no release that satisfies {wanted or 'anything'} with "
                       f"a wheel for CPython {python} on Windows x86-64")

    steps = 0
    while queue:
        key, name = queue.pop(0)
        steps += 1
        if steps > 600:
            raise PinError("the closure did not settle after 600 steps; a pair of packages "
                           "is pulling each other's version back and forth, and a person has "
                           "to pin one of them in ROOTS")
        try:
            listed, version, kind, item = pick(key, name)
        except PinError as refusal:
            # Nothing fits the constraints on this package as they stand. The
            # newest choice upstream of it is the one to give way: the most
            # recently chosen package that constrains it and is not a root
            # pinned exactly by hand has its current version banned and is
            # chosen again, older, which is what "the newest diffusers that
            # still allows huggingface-hub below 1.0" means in practice. A
            # constraint that only exact roots impose is a real contradiction,
            # and that is the sentence that comes back.
            sources = [row[1] for row in constraints.get(key, ())
                       if row[1] != "roots" and row[1] not in exact and row[1] in chosen]
            if not sources:
                raise
            culprit = max(sources, key=lambda source: order.index(source))
            gave_way = chosen.pop(culprit)
            banned.setdefault(culprit, set()).add(gave_way.version)
            say(f"  {gave_way.name} {gave_way.version} gives way: nothing satisfies "
                f"{specifier_for(key)} for {name}")
            for other in list(constraints):
                constraints[other] = [row for row in constraints[other] if row[1] != culprit]
            queue.append((culprit, gave_way.name))
            continue
        current = chosen.get(key)
        if current is not None and current.version == version:
            continue
        release = pypi.release(listed, version)
        requires = _requirements_of(release)
        chosen[key] = Choice(name=listed, version=version, kind=kind, file=item,
                             requires=requires)
        if key in order:
            order.remove(key)
        order.append(key)
        say(f"  {listed} {version} ({kind}){' ← re-chosen' if current else ''}")
        # This version's declarations replace the previous version's.
        for other in list(constraints):
            constraints[other] = [row for row in constraints[other] if row[1] != key]
        for raw in requires:
            try:
                need = Requirement(raw)
            except Exception:
                say(f"  ! {listed} {version} declares {raw!r}, which this tool cannot read")
                continue
            if not _applies(need):
                continue
            dep = canonical(need.name)
            if dep in excluded or dep == torch_key:
                continue
            constraints.setdefault(dep, []).append((need.specifier, key, need.name))
            have = chosen.get(dep)
            if have is None or not need.specifier.contains(Version(have.version),
                                                           prereleases=True):
                if all(queued != dep for queued, _ in queue):
                    queue.append((dep, need.name))

    # Prune what only a superseded version wanted.
    reachable = set()
    frontier = list(root_names) + [torch_key]
    while frontier:
        key = frontier.pop()
        if key in reachable or (key not in chosen and key not in virtual):
            continue
        reachable.add(key)
        for raw in (chosen[key].requires if key in chosen else virtual[key]):
            try:
                need = Requirement(raw)
            except Exception:
                continue
            if _applies(need):
                dep = canonical(need.name)
                if dep in chosen and dep not in reachable:
                    frontier.append(dep)
    return {key: chosen[key] for key in chosen if key in reachable}


def requirements_met(chosen: dict, excluded: dict, torch_version: str,
                     torch_requires, roots=ROOTS) -> None:
    """Does every pin satisfy every other pin's declaration? Refuse otherwise.

    The same check ``tools/pin_pocket_models.py`` grew after a closure shipped in
    which every wheel downloaded, every hash matched, and the import then died
    on a typing-extensions that predated the pydantic beside it. Here it runs
    over the solver's own answer, which is belt and braces on purpose: the
    solver is new, and the check is the part that has caught a real failure.

    Torch is checked both ways -- what the closure asks of torch against the
    pinned version, and what torch declares against the closure -- from PyPI's
    record of the same version, since the CUDA wheel is built from the same
    source and declares the same things.
    """
    Requirement, SpecifierSet, canonical, Version, _invalid = _packaging()
    pinned = {key: (choice.name, choice.version) for key, choice in chosen.items()}
    torch_key = canonical(TORCH["package"])
    trouble = []

    def check(owner: str, raw: str):
        try:
            need = Requirement(raw)
        except Exception:
            return
        if not _applies(need):
            return
        dep = canonical(need.name)
        if dep == torch_key:
            if not need.specifier.contains(Version(torch_version), prereleases=True):
                trouble.append(f"{owner} needs {need}, and torch is pinned at {torch_version}.")
            return
        if dep not in pinned:
            if dep in excluded:
                return
            trouble.append(f"{owner} needs {need}, and the closure does not ship it. Add it, "
                           f"or add it to EXCLUDED with the reason it is not needed.")
            return
        if not need.specifier.contains(Version(pinned[dep][1]), prereleases=True):
            trouble.append(f"{owner} needs {need}, and the closure pins {pinned[dep][0]} "
                           f"{pinned[dep][1]}.")

    for key, choice in chosen.items():
        for raw in choice.requires:
            check(f"{choice.name} {choice.version}", raw)
    for name, spec in roots:
        key = canonical(name)
        if key not in pinned:
            trouble.append(f"the root {name} is not in the closure.")
        elif spec and not SpecifierSet(spec).contains(Version(pinned[key][1]),
                                                       prereleases=True):
            trouble.append(f"the root {name}{spec} is pinned at {pinned[key][1]}.")
    for raw in torch_requires or ():
        check(f"torch {torch_version}", raw)
    if trouble:
        raise PinError("this closure cannot install as written:\n    "
                       + "\n    ".join(sorted(set(trouble))))


def triples(chosen: dict, roots=ROOTS) -> list:
    """``[package, version, kind]`` rows, roots first and the rest alphabetical.

    Written into the manifest as well as the artifacts, because the manifest is
    the file somebody opens to see what a machine will be asked to install, and
    thirty wheel filenames are harder to read than thirty names.
    """
    _req, _spec, canonical, _version, _invalid = _packaging()
    ordered = []
    seen = set()
    for name, _spec_text in roots:
        key = canonical(name)
        if key in chosen and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in sorted(chosen, key=lambda item: chosen[item].name.casefold()):
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return [[chosen[key].name, chosen[key].version, chosen[key].kind] for key in ordered]


def resolve_triples(rows, pypi: PyPI, say, python: str = PYTHON) -> dict:
    """``--keep-closure``: the written triples, each read from PyPI and selected.

    For a re-pin that must not move anything: the same wheels, read again for
    their sizes and digests, and refused if the kind written down no longer
    matches what the publisher ships.
    """
    _req, _spec, canonical, _version, _invalid = _packaging()
    chosen = {}
    for row in rows or ():
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise PinError(f"{row!r} in runtime.closure is not a [package, version, kind] "
                           f"triple")
        name, version, kind = (str(part).strip() for part in row)
        if kind not in KINDS:
            raise PinError(f"{name} {version} names the wheel kind {kind!r}, which this tool "
                           f"does not know ({', '.join(KINDS)})")
        release = pypi.release(name, version)
        found_kind, item = usable_wheel(release.get("urls") or (), python)
        if item is None:
            raise PinError(f"{name} {version} has no wheel for CPython {python} on Windows "
                           f"x86-64 any more")
        if found_kind != kind:
            raise PinError(f"{name} {version} is written down as a {kind} wheel and the "
                           f"publisher now ships a {found_kind} one")
        listed = str((release.get("info") or {}).get("name") or name)
        chosen[canonical(name)] = Choice(name=listed, version=version, kind=kind, file=item,
                                         requires=_requirements_of(release))
        say(f"  {listed} {version} ({kind})")
    return chosen


# --------------------------------------------------------------------------- #
# The hub
# --------------------------------------------------------------------------- #


class _StopAtRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.casefold()


def _header_map(answer) -> dict:
    return {str(name).casefold(): value for name, value in answer.headers.items()}


def _get(url: str) -> tuple:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            return int(getattr(answer, "status", 200) or 200), answer.read()
    except urllib.error.HTTPError as answer:
        with contextlib.closing(answer):
            return int(getattr(answer, "code", 0) or 0), b""
    except Exception as exc:
        raise PinError(f"{_host(url)} could not be reached ({exc})") from None


def _head(url: str, hops: int = 5) -> tuple:
    """``(status, headers)`` without leaving the publisher's host.

    The hub answers a ``/resolve/`` HEAD with the LFS object's SHA-256 in
    ``x-linked-etag`` and a redirect to a storage host whose own ``ETag`` is
    not a digest. Stopping at the hop is what makes the number a digest.
    """
    opener = urllib.request.build_opener(_StopAtRedirect)
    here = url
    status, found = 0, {}
    for _ in range(max(int(hops), 1)):
        request = urllib.request.Request(
            here, method="HEAD",
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        try:
            with contextlib.closing(opener.open(request, timeout=TIMEOUT)) as answer:
                return int(getattr(answer, "status", 200) or 200), _header_map(answer)
        except urllib.error.HTTPError as answer:
            with contextlib.closing(answer):
                status = int(getattr(answer, "code", 0) or 0)
                found = _header_map(answer)
        except Exception as exc:
            raise PinError(f"{url.split('?')[0]} could not be read ({exc})") from None
        if status not in (301, 302, 303, 307, 308):
            return status, found
        target = urllib.parse.urljoin(here, str(found.get("location") or "").strip())
        if not target or _host(target) != _host(here):
            return 200, found
        here = target
    return status or 200, found


def _as_sha256(raw, prefixed: bool) -> str:
    text = str(raw or "").strip()
    if text.startswith("W/"):
        text = text[2:]
    text = text.strip('"').strip()
    announced = text.casefold().startswith("sha256:")
    if announced:
        text = text.split(":", 1)[1].strip()
    if prefixed and not announced:
        return ""
    if len(text) != 64:
        return ""
    try:
        int(text, 16)
    except ValueError:
        return ""
    return text.casefold()


def _published_sha256(headers: dict) -> str:
    for name in SHA256_HEADERS:
        found = _as_sha256(headers.get(name), prefixed=False)
        if found:
            return found
    return _as_sha256(headers.get("etag"), prefixed=True)


def _published_size(headers: dict) -> int:
    for name in ("x-linked-size", "content-length"):
        raw = str(headers.get(name) or "")
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return 0


def repository(repo: str, revision: str, say) -> tuple:
    """``(commit, served)`` -- what ``revision`` points at, and every file there."""
    status, body = _get(HUB_API.format(repo=repo, revision=revision))
    if status in (401, 403):
        raise PinError(f"huggingface.co/{repo} answered HTTP {status}; the mirror this "
                       f"manifest names is not public any more")
    if status != 200:
        raise PinError(f"{repo}: the hub answered {status} for revision {revision}")
    try:
        answer = json.loads(body.decode("utf-8", "replace"))
        commit = str(answer["sha"])
        served = tuple(sorted(str(item.get("rfilename") or "")
                              for item in (answer.get("siblings") or ())
                              if isinstance(item, dict) and item.get("rfilename")))
    except (ValueError, KeyError, TypeError) as exc:
        raise PinError(f"{repo}: the hub's answer for {revision} could not be read "
                       f"({exc})") from None
    if len(commit) != 40:
        raise PinError(f"{repo}: {commit!r} is not a commit sha")
    say(f"  {repo} is at {commit[:12]}, serving {len(served)} file(s)")
    return commit, served


def hub_artifact(repo: str, commit: str, path: str, local_name: str) -> dict:
    url = HUB_RESOLVE.format(repo=repo, revision=commit, path=urllib.parse.quote(path))
    status, headers = _head(url)
    if status >= 400:
        raise PinError(f"{repo}: the hub answered {status} for {path} at {commit[:12]}")
    return {
        "filename": path,
        "local_name": local_name,
        "url": url,
        "bytes": _published_size(headers) or None,
        "sha256": _published_sha256(headers) or None,
    }


def shards_named(index: dict) -> list:
    """The distinct shard files a safetensors index names, in first-seen order.

    The same reading ``mc_voice_vibevoice`` makes at install time, so the list
    this tool pins and the list the installer downloads are one reading of one
    document. A name with a directory in it is refused: the index names files
    beside itself and nothing else.
    """
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise PinError("model.safetensors.index.json has no weight_map, so the shards "
                       "cannot be listed")
    found = []
    for name in weights.values():
        text = str(name or "").strip()
        if not text or "/" in text or "\\" in text or text in (".", "..") \
                or not text.endswith(".safetensors"):
            raise PinError(f"the index names a shard ({text!r}) that is not a safetensors "
                           f"file beside it")
        if text not in found:
            found.append(text)
    return found


def _describe(artifact: dict) -> str:
    size = artifact.get("bytes")
    digest = artifact.get("sha256")
    if size and digest:
        return f"{size} bytes, sha256 {digest[:12]}"
    if size:
        return f"{size} bytes, no published digest (attested at install time)"
    return "neither a size nor a digest was published (attested at install time)"


INDEX = "model.safetensors.index.json"


def model(entry: dict, say, identifier: str = MODEL_ID) -> dict:
    """A model's files, its weights from its own index (or, for a model that
    publishes none, its one weights file), and its tokenizer."""
    found = dict(entry)
    repo = str(entry.get("repo") or "")
    if not repo:
        raise PinError(f"{identifier} names no repo")
    commit, served = repository(repo, str(entry.get("revision") or "main"), say)
    files = []
    for declared in entry.get("files") or ():
        path = str(declared.get("filename") or "")
        optional = bool(declared.get("optional"))
        if path not in served:
            if optional:
                say(f"  {path}: not served by {repo}; kept as optional")
                files.append(dict(declared))
                continue
            listing = "\n  ".join(served) or "(nothing)"
            raise PinError(f"{repo} at {commit[:12]} does not serve {path}. It holds:"
                           f"\n  {listing}")
        artifact = hub_artifact(repo, commit, path,
                                str(declared.get("local_name") or path))
        if optional:
            artifact["optional"] = True
        say(f"  {path}: {_describe(artifact)}")
        files.append(artifact)
    single = entry.get("single_weights") if isinstance(entry.get("single_weights"),
                                                       dict) else None
    single_name = str((single or {}).get("filename") or "")
    if INDEX in served:
        index_url = HUB_RESOLVE.format(repo=repo, revision=commit, path=INDEX)
        status, body = _get(index_url)
        if status != 200:
            raise PinError(f"{repo}: the hub answered {status} for the safetensors index")
        try:
            names = shards_named(json.loads(body.decode("utf-8", "replace")))
        except ValueError as exc:
            raise PinError(f"{repo}: the safetensors index is not JSON ({exc})") from None
    elif single_name and single_name in served:
        say(f"  {repo} publishes no {INDEX}; its weights are {single_name}")
        names = [single_name]
    else:
        raise PinError(f"{repo} at {commit[:12]} serves no {INDEX}"
                       + (f" and no {single_name}" if single_name else ""))
    shards = []
    for name in names:
        if name not in served:
            raise PinError(f"{repo} at {commit[:12]} names {name} in its index and does "
                           f"not serve it")
        artifact = hub_artifact(repo, commit, name, name)
        say(f"  {name}: {_describe(artifact)}")
        shards.append(artifact)
    if single is not None and names == [single_name]:
        found["single_weights"] = dict(shards[0])
    tokenizer = dict(entry.get("tokenizer") or {})
    token_repo = str(tokenizer.get("repo") or "")
    if not token_repo:
        raise PinError(f"{identifier} names no tokenizer repo")
    token_commit, token_served = repository(
        token_repo, str(tokenizer.get("revision") or "main"), say)
    token_files = []
    for declared in tokenizer.get("files") or ():
        path = str(declared.get("filename") or "")
        if path not in token_served:
            raise PinError(f"{token_repo} at {token_commit[:12]} does not serve {path}")
        artifact = hub_artifact(token_repo, token_commit, path,
                                str(declared.get("local_name") or path))
        say(f"  {path}: {_describe(artifact)}")
        token_files.append(artifact)
    tokenizer["files"] = token_files
    tokenizer["commit"] = token_commit
    found["files"] = files
    found["shards"] = shards
    found["commit"] = commit
    found["tokenizer"] = tokenizer
    sizes = [int(one.get("bytes") or 0) for one in shards]
    if sizes and all(sizes):
        estimates = dict(found.get("estimates") or {})
        estimates["weights_bytes"] = sum(sizes)
        by_precision = dict(estimates.get("weights_by_precision") or {})
        by_precision["bf16"] = sum(sizes)
        estimates["weights_by_precision"] = by_precision
        found["estimates"] = estimates
    return found


# --------------------------------------------------------------------------- #
# Microsoft's repository, at one commit
# --------------------------------------------------------------------------- #


def _read_bytes(url: str, ceiling: int = 64 * 1024 * 1024) -> bytes:
    """One file from raw.githubusercontent.com, whole. A ceiling, because nothing
    this tool pins from there is larger than a few megabytes."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            body = answer.read(ceiling + 1)
    except urllib.error.HTTPError as answer:
        with contextlib.closing(answer):
            raise PinError(f"{url} answered HTTP {answer.code}") from None
    except Exception as exc:
        raise PinError(f"{_host(url)} could not be reached ({exc})") from None
    if len(body) > ceiling:
        raise PinError(f"{url} is larger than {ceiling} bytes, which nothing pinned from "
                       f"there should be")
    if not body:
        raise PinError(f"{url} answered with nothing")
    return body


class Raw:
    """Files of Microsoft's repository, read whole so they can be hashed. Cached,
    and replaceable by a test, like :class:`PyPI`."""

    def __init__(self, fetch=None):
        self._fetch = fetch or _read_bytes
        self._cache = {}

    def fetch(self, url: str) -> bytes:
        if url not in self._cache:
            self._cache[url] = bytes(self._fetch(url))
        return self._cache[url]


def raw_url(path: str, commit: str = VIBEVOICE_COMMIT) -> str:
    return RAW_URL.format(commit=commit, path=urllib.parse.quote(path))


def overlay_entries(raw: Raw, say, commit: str = VIBEVOICE_COMMIT) -> list:
    """:data:`OVERLAY` as manifest entries: each file downloaded, sized and hashed."""
    found = []
    for path, why in OVERLAY:
        url = raw_url(path, commit)
        body = raw.fetch(url)
        try:
            body.decode("utf-8")
        except UnicodeDecodeError:
            raise PinError(f"{path} at {commit[:12]} is not a Python source file") from None
        digest = hashlib.sha256(body).hexdigest()
        say(f"  {path}: {len(body)} bytes, sha256 {digest[:12]}")
        found.append({"path": path, "url": url, "bytes": len(body), "sha256": digest,
                      "source": {"repository": VIBEVOICE_REPOSITORY, "commit": commit},
                      "why": why})
    return found


def preset_meta(stem: str) -> dict:
    """What a preset's file name says: its speaker, language, accent and gender.

    ``en-Carter_man`` is Carter, English, a man; ``de-Spk1_woman`` is the second
    German speaker, a woman; ``in-Samuel_man`` is Samuel, English with an Indian
    accent (:data:`PRESET_LANGUAGES`). Every language but English is one the
    model's own documentation calls experimental.
    """
    prefix, _, rest = str(stem).partition("-")
    speaker, _, gender = rest.rpartition("_")
    if prefix not in PRESET_LANGUAGES or gender not in ("man", "woman") or not speaker \
            or not re.fullmatch(r"[A-Za-z0-9]+", speaker):
        raise PinError(f"{stem!r} is not a preset voice name this tool can read "
                       f"(<language>-<speaker>_<man|woman>)")
    language = PRESET_LANGUAGES[prefix] or prefix
    accent = prefix if PRESET_LANGUAGES[prefix] else ""
    numbered = re.fullmatch(r"Spk(\d+)", speaker)
    return {"name": f"Speaker {numbered.group(1)}" if numbered else speaker,
            "language": language, "accent": accent, "gender": gender,
            "experimental": language != "en"}


def preset_entries(raw: Raw, say, commit: str = VIBEVOICE_COMMIT) -> list:
    """:data:`PRESETS` as manifest entries: each voice downloaded, sized and hashed."""
    found = []
    for stem in PRESETS:
        meta = preset_meta(stem)
        url = raw_url(f"{PRESETS_PATH}/{stem}.pt", commit)
        body = raw.fetch(url)
        if not body.startswith(b"PK\x03\x04"):
            raise PinError(f"{stem}.pt at {commit[:12]} is not a torch.save archive (an LFS "
                           f"pointer or an error page would look like this)")
        digest = hashlib.sha256(body).hexdigest()
        say(f"  {stem}.pt: {len(body)} bytes, sha256 {digest[:12]}")
        found.append({"id": stem, **meta, "url": url, "bytes": len(body), "sha256": digest})
    return found


# --------------------------------------------------------------------------- #
# Torch, from the publisher's own index
# --------------------------------------------------------------------------- #


def torch_wheel(resolve: dict, say) -> dict:
    """The cp313 win_amd64 wheel of the pinned torch version, from the cu128 index.

    Read through :mod:`mc_voice_wheelindex`, the same reader the installer
    uses, so the wheel this tool records is the wheel the installer will
    choose. The digest comes from the index's ``#sha256=`` fragment and the
    size from a HEAD of the wheel itself.
    """
    sys.path.insert(0, str(ROOT))
    import mc_voice_wheelindex as wheelindex

    base = str(resolve.get("index") or "")
    package = str(resolve.get("package") or "torch")
    version = full_torch_version(resolve)
    request = urllib.request.Request(base, headers={"User-Agent": USER_AGENT,
                                                    "Accept": "text/html"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            page = answer.read(8 * 1024 * 1024).decode("utf-8", "replace")
    except Exception as exc:
        raise PinError(f"{base} could not be read ({exc})") from None
    tags = wheelindex.platform_tags(PYTHON, "windows", "amd64")
    try:
        found = wheelindex.choose(page, base, package, version, tags)
    except wheelindex.IndexError_ as exc:
        raise PinError(str(exc)) from None
    status, headers = _head(found["url"])
    size = _published_size(headers) if status < 400 else 0
    say(f"  {found['filename']}: {size or '?'} bytes, sha256 {found['sha256'][:12]}")
    return {"filename": found["filename"], "url": found["url"],
            "sha256": found["sha256"], "bytes": size or None}


def full_torch_version(resolve: dict) -> str:
    """``2.8.0`` on a ``/whl/cu128/`` index is ``2.8.0+cu128`` on its page."""
    version = str(resolve.get("version") or "")
    if "+" in version or not version:
        return version
    found = re.search(r"/whl/([A-Za-z0-9_]+)/", str(resolve.get("index") or ""))
    return f"{version}+{found.group(1)}" if found else version


# --------------------------------------------------------------------------- #
# The whole manifest
# --------------------------------------------------------------------------- #


@dataclass
class State:
    disagreements: list = field(default_factory=list)
    wheels: int = 0
    closure_complete: bool = False
    model_hashed: bool = False
    """Every model the manifest declares is sized and hashed from the hub."""
    torch_recorded: bool = False
    shards: int = 0
    overlay_pinned: bool = False
    presets_pinned: bool = False
    presets: int = 0
    models_hashed: dict = field(default_factory=dict)

    @property
    def pinned(self) -> bool:
        return bool(self.closure_complete and self.model_hashed and self.torch_recorded
                    and self.overlay_pinned and self.presets_pinned)


_HUB = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")


def _identity(artifact: dict) -> str:
    """What makes an artifact the same artifact, whatever revision its URL names.

    A hub file is its repository and path -- two models' ``config.json``, or two
    tokenizers' ``tokenizer.json``, are different files -- an overlay file its
    path in the package, a preset voice its id, and a wheel its filename.
    """
    if artifact.get("path"):
        return f"overlay:{artifact['path']}"
    url = str(artifact.get("url") or "")
    if artifact.get("id") and url.startswith("https://raw.githubusercontent.com/"):
        return f"voice:{artifact['id']}"
    found = _HUB.match(url)
    if found:
        return f"{found.group(1)}/{urllib.parse.unquote(found.group(3))}"
    return str(artifact.get("filename") or "")


def _committed(existing: dict) -> dict:
    """Every digest already checked in, by identity. One artifact, one digest."""
    found = {}

    def keep(entry):
        if not (isinstance(entry, dict) and entry.get("sha256")
                and (entry.get("filename") or entry.get("path") or entry.get("id"))):
            return
        name, digest = _identity(entry), str(entry["sha256"]).casefold()
        was = found.setdefault(name, digest)
        if was != digest:
            raise PinError(f"the manifest gives {name} two digests, {was} and {digest}. "
                           f"One artifact has one identity, and this tool will not choose.")

    runtime = existing.get("runtime") or {}
    for platform in (runtime.get("platforms") or ()):
        for item in platform.get("artifacts") or ():
            keep(item)
    for item in runtime.get("overlay") or ():
        keep(item)
    for entry in (existing.get("models") or {}).values():
        if not isinstance(entry, dict):
            continue
        for item in entry.get("files") or ():
            keep(item)
        for item in entry.get("shards") or ():
            keep(item)
        for item in (entry.get("tokenizer") or {}).get("files") or ():
            keep(item)
        keep(entry.get("single_weights"))
        for item in entry.get("voices") or ():
            keep(item)
    return found


def _agree(state: State, committed: dict, artifact: dict, where: str) -> None:
    name = _identity(artifact)
    fresh = str(artifact.get("sha256") or "").casefold()
    was = committed.get(name, "")
    if was and fresh and was != fresh:
        state.disagreements.append(f"{where}/{name}: checked in {was}, published {fresh}")


def _hashed(items) -> bool:
    items = list(items or ())
    if not items:
        return False
    for one in items:
        if one.get("optional") and not one.get("url"):
            continue
        if len(str(one.get("sha256") or "")) != 64 or int(one.get("bytes") or 0) <= 0:
            return False
    return True


def _model_hashed(entry: dict) -> bool:
    required = [one for one in (entry.get("files") or ()) if not one.get("optional")]
    return (_hashed(required) and _hashed(entry.get("shards"))
            and _hashed((entry.get("tokenizer") or {}).get("files")))


def _declared_models(models: dict) -> str:
    """The models still declared-not-hashed, each with its weights and tokenizer."""
    described = []
    for identifier, entry in (models or {}).items():
        if not isinstance(entry, dict) or _model_hashed(entry):
            continue
        mirrors = [str(name) for name in (entry.get("mirrors") or ()) if name]
        weights = ("its weights file (model.safetensors, or the shards an index names if "
                   "one is published)" if entry.get("single_weights")
                   else "its shards, listed from model.safetensors.index.json at install time")
        described.append(
            f"{entry.get('label') or identifier} ({entry.get('repo') or '?'}"
            f"{', with ' + ' and '.join(mirrors) + ' as fallback' if mirrors else ''}), "
            f"{weights}, and its {(entry.get('tokenizer') or {}).get('repo') or 'Qwen'} "
            f"tokenizer")
    return "; ".join(described)


def notes_for(state: State, torch: dict, models: dict = None) -> str:
    """The manifest's own account of what is and is not hashed. Always current."""
    version = str(torch.get("version") or "")
    models = models if isinstance(models, dict) else {}
    commit = VIBEVOICE_COMMIT[:12]
    closure = ("resolved: every PyPI wheel is named, sized and hashed from pypi.org for "
               f"{PLATFORM_ID}, solved from the package's own declarations for CPython 3.13 "
               "on Windows x86-64, and tools/pin_vibevoice_models.py refuses a closure "
               "whose pins contradict one another. It includes bitsandbytes 0.48.2, which "
               "quantises the 7B's language model to 8-bit or 4-bit NF4 and whose one Windows "
               "wheel carries libbitsandbytes_cuda128.dll with code for sm_86 and sm_120, and "
               "peft 0.17.1, which loads LoRA adapters, pinned exactly because peft declares "
               "transformers unversioned while 0.18.0 does not import under 4.51.3"
               if state.closure_complete else
               "written down but not yet resolved, so the managed install refuses and says "
               "why")
    if state.overlay_pinned:
        overlay = (f"HASHED as well: the runtime overlay, five files of Microsoft's own "
                   f"repository ({VIBEVOICE_REPOSITORY}) at commit {commit} -- the Realtime "
                   f"model's configuration, model, generation and processor, which the wheel "
                   f"does not have, and a configuration module its own copy is a subset of -- "
                   f"sized and hashed from raw.githubusercontent.com at that commit and written "
                   f"over the unpacked package (runtime.overlay says why for each), so the "
                   f"runtime's closure is the wheels plus exactly those bytes")
    else:
        overlay = ("NOT YET PINNED: the runtime overlay, so the managed install refuses until "
                   "this tool has fetched and hashed it")
    if state.presets_pinned:
        presets = (f"and the Realtime 0.5B's {state.presets} preset voices, the .pt voice "
                   f"prompts in {PRESETS_PATH} at the same commit, sized and hashed the same "
                   f"way (Microsoft withholds the code that makes a voice prompt from a "
                   f"recording, so these are the only voices that model has)")
    else:
        presets = ("and NOT YET PINNED: the Realtime 0.5B's preset voices, which the install "
                   "therefore cannot check")
    if state.torch_recorded:
        torch_text = (f"which this file also records ({str(torch.get('sha256') or '')[:12]}…), "
                      f"so a publisher that changed the wheel is refused")
    else:
        torch_text = ("and the machine that wrote this could not reach download.pytorch.org, "
                      "so no digest is recorded here yet: `python tools/pin_vibevoice_models.py "
                      "--torch` on one that can turns the index's word into a committed "
                      "constant")
    declared = _declared_models(models)
    if state.model_hashed or (models and not declared):
        model_text = ("PINNED: every model's files, weights and tokenizer are sized and hashed "
                      "from the hub at named commits")
    else:
        model_text = ("DECLARED BUT NOT HASHED"
                      + (f" -- {declared}" if declared else "")
                      + ". The repositories, paths and revisions are written down, and every "
                        "file is checked against the digest its publisher states over HTTPS "
                        "(x-linked-etag) and recorded in "
                        "voice/managed-vibevoice-models.local.json, because the machine that "
                        "wrote this could not reach huggingface.co. Neither community mirror of "
                        "the 7B ships a tokenizer, and the Realtime model's own processor fetches "
                        "one too, so each model's four Qwen tokenizer files are declared beside "
                        "it and installed into a directory whose name contains 'qwen', which is "
                        "how the processor picks its tokenizer class")
    return (
        f"{'PINNED' if state.pinned else 'PARTIALLY PINNED'}. The runtime closure is {closure}. "
        f"{overlay}, {presets}. "
        f"torch is NOT a PyPI wheel here: the CUDA 12.8 build lives only on "
        f"download.pytorch.org, so the platform carries a versioned resolve entry (torch "
        f"{version}) that the installer answers against the publisher's own index page at "
        f"install time, taking the cp313 win_amd64 wheel of exactly that version and the "
        f"SHA-256 the index states, {torch_text}. {version} is the line both of the user's "
        f"cards share (sm_86 and sm_120 in one cu128 wheel since 2.7) and the nearest release "
        f"to the community package's own era; the 2.9 and 2.10 cu128 lines also ship cp313 "
        f"Windows wheels and moving to one is a review decision, not a default. No "
        f"torchaudio: nothing imports it. The models are {model_text}. That is a weaker "
        f"claim than the closure's and the difference is worth "
        f"stating: an artifact with a digest here is checked against a number this repository "
        f"committed to, and one without is checked against the digest its publisher reports "
        f"at install time. Both refuse a file that arrives wrong; only the first refuses a "
        f"publisher that changed its mind. `python tools/pin_vibevoice_models.py --model` on "
        f"a machine that can reach the hub records the sizes, the digests and the weights "
        f"list of every model and turns the first claim into the second. Not installed, on "
        f"purpose: gradio, "
        f"aiortc, av, ml-collections and absl-py (the demo and its server); librosa and "
        f"soundfile (imported lazily, only to read an audio file from a path and to write "
        f"one, neither of which the worker does -- it is handed PCM over its pipe and answers "
        f"with WAV bytes it encodes itself); numba, llvmlite and scipy (librosa's); torchaudio; "
        f"and httpx (declared by diffusers 0.36 and later, imported only by its pipeline "
        f"loaders, which vibevoice never touches). runtime.excluded records each reason. "
        f"Installing from a folder you filled yourself works either way, and records the "
        f"digests of what it was given.")


def build(existing: dict, options, say) -> tuple:
    """``(manifest, state)`` -- the closure refreshed and everything else kept."""
    state = State()
    committed = _committed(existing)
    found = json.loads(json.dumps(existing)) if existing else {}
    if int(found.get("schema") or 0) not in (0, SCHEMA):
        raise PinError(f"the manifest is schema {found.get('schema')} and this tool writes "
                       f"schema {SCHEMA}")
    found["schema"] = SCHEMA
    runtime = dict(found.get("runtime") or {})
    torch = dict(runtime.get("torch") or {})
    for key, value in TORCH.items():
        torch.setdefault(key, value)
    pypi = options.pypi

    declared = list(runtime.get("platforms") or ())
    if len(declared) != 1 or str(declared[0].get("id") or "") != PLATFORM_ID:
        raise PinError(f"the manifest must advertise exactly one platform, {PLATFORM_ID}; "
                       f"a second combination is a release decision")
    entry = dict(declared[0])
    if str(entry.get("python") or "") != PYTHON or str(entry.get("system") or "") != "windows" \
            or str(entry.get("accelerator") or "") != "cuda":
        raise PinError(f"{PLATFORM_ID} must be windows, Python {PYTHON}, accelerator cuda")

    if options.keep_closure:
        say(f"\nRe-pinning the {len(runtime.get('closure') or ())} package(s) written down:")
        chosen = resolve_triples(runtime.get("closure") or (), pypi, say)
    else:
        say(f"\nSolving the closure from {len(ROOTS)} roots for {PLATFORM_ID}:")
        chosen = solve(ROOTS, pypi, EXCLUDED, say, torch=torch)
    say("  reading what torch declares, from PyPI's record of the same version")
    torch_requires = _requirements_of(pypi.release(torch["package"], torch["version"]))
    requirements_met(chosen, EXCLUDED, str(torch["version"]), torch_requires)
    say("  every pin satisfies every other pin's declared requirement")

    rows = triples(chosen)
    artifacts = []
    for key in (canonical for canonical in _ordered_keys(chosen, rows)):
        artifact = wheel_artifact(chosen[key].file)
        _agree(state, committed, artifact, PLATFORM_ID)
        artifacts.append(artifact)
    state.wheels = len(artifacts)
    state.closure_complete = bool(artifacts) and all(
        len(str(one["sha256"])) == 64 and one["bytes"] > 0 for one in artifacts)
    total = sum(int(one["bytes"]) for one in artifacts)
    say(f"  {PLATFORM_ID}: {len(artifacts)} wheels, {total / 1e6:.0f} MB, plus torch "
        f"{torch['version']} from {torch['index']}")

    resolve = {"index": torch["index"], "package": torch["package"],
               "version": torch["version"]}
    previous = next((item.get("resolve") for item in (entry.get("artifacts") or ())
                     if isinstance(item, dict) and item.get("resolve")), None) or {}
    for key in ("filename", "sha256", "bytes"):
        if previous.get(key):
            resolve[key] = previous[key]
    if options.torch:
        say("\nReading the torch wheel from the publisher's index:")
        wheel = torch_wheel(resolve, say)
        if resolve.get("sha256") and resolve["sha256"] != wheel["sha256"]:
            state.disagreements.append(
                f"torch/{wheel['filename']}: checked in {resolve['sha256']}, published "
                f"{wheel['sha256']}")
        resolve.update({"filename": wheel["filename"], "sha256": wheel["sha256"],
                        "bytes": wheel["bytes"]})
    state.torch_recorded = len(str(resolve.get("sha256") or "")) == 64
    for key in ("filename", "sha256", "bytes"):
        if resolve.get(key):
            torch[key] = resolve[key]
    entry["artifacts"] = artifacts + [{"local_name": "torch", "resolve": resolve}]

    say(f"\nPinning the runtime overlay from {VIBEVOICE_REPOSITORY} at "
        f"{VIBEVOICE_COMMIT[:12]}:")
    overlay = overlay_entries(options.raw, say)
    for item in overlay:
        _agree(state, committed, item, "runtime.overlay")
    state.overlay_pinned = bool(overlay) and all(
        len(str(one["sha256"])) == 64 and one["bytes"] > 0 for one in overlay)

    runtime["closure"] = rows
    runtime["excluded"] = dict(EXCLUDED)
    runtime["torch"] = torch
    runtime["platforms"] = [entry]
    runtime["overlay"] = overlay
    found["runtime"] = runtime

    models = dict(found.get("models") or {})
    if not isinstance(models.get(MODEL_ID), dict):
        raise PinError(f"the manifest has no model called {MODEL_ID!r}")
    realtime = models.get(MODEL_REALTIME)
    if isinstance(realtime, dict):
        say(f"\nPinning {MODEL_REALTIME}'s preset voices from {PRESETS_PATH}:")
        voices = preset_entries(options.raw, say)
        for item in voices:
            _agree(state, committed, item, MODEL_REALTIME)
        realtime = dict(realtime)
        realtime["voices"] = voices
        realtime["voices_source"] = {"repository": VIBEVOICE_REPOSITORY,
                                     "commit": VIBEVOICE_COMMIT, "path": PRESETS_PATH}
        models[MODEL_REALTIME] = realtime
        state.presets = len(voices)
        state.presets_pinned = bool(voices) and all(
            len(str(one["sha256"])) == 64 and one["bytes"] > 0 for one in voices)
    if options.model:
        for identifier in list(models):
            if not isinstance(models[identifier], dict):
                continue
            say(f"\nResolving {identifier} from the hub:")
            pinned_entry = model(models[identifier], say, identifier)
            for item in (list(pinned_entry.get("files") or ())
                         + list(pinned_entry.get("shards") or ())):
                _agree(state, committed, item, identifier)
            for item in (pinned_entry.get("tokenizer") or {}).get("files") or ():
                _agree(state, committed, item, f"{identifier}/tokenizer")
            models[identifier] = pinned_entry
    found["models"] = models
    state.shards = len(models[MODEL_ID].get("shards") or ())
    state.models_hashed = {identifier: _model_hashed(one) for identifier, one in models.items()
                           if isinstance(one, dict)}
    state.model_hashed = bool(state.models_hashed) and all(state.models_hashed.values())

    found["pinned"] = state.pinned
    found["notes"] = notes_for(state, torch, models)
    found["version"] = int(found.get("version") or 0) + 1
    return found, state


def _ordered_keys(chosen: dict, rows: list) -> list:
    _req, _spec, canonical, _version, _invalid = _packaging()
    return [canonical(row[0]) for row in rows]


def survey(existing: dict, path: Path, say) -> None:
    runtime = existing.get("runtime") or {}
    say(f"{path} as it stands:")
    say(f"  pinned: {'yes' if existing.get('pinned') else 'no'}")
    say(f"  runtime.closure: {len(runtime.get('closure') or ())} package(s) written down")
    torch = runtime.get("torch") or {}
    say(f"  torch: {torch.get('version') or '?'} from {torch.get('index') or '?'}"
        f"{' (digest recorded)' if torch.get('sha256') else ' (no digest recorded)'}")
    for item in runtime.get("platforms") or ():
        artifacts = [one for one in (item.get("artifacts") or ()) if not one.get("resolve")]
        hashed = sum(1 for one in artifacts if one.get("sha256"))
        say(f"  {item.get('id')}: "
            + (f"{len(artifacts)} wheel(s), {hashed} hashed" if artifacts
               else "no wheels resolved"))
    overlay = runtime.get("overlay") or ()
    say(f"  runtime.overlay: {len(overlay)} file(s), "
        f"{sum(1 for one in overlay if one.get('sha256'))} hashed")
    for name, entry in (existing.get("models") or {}).items():
        if not isinstance(entry, dict):
            continue
        say(f"  {name}: {len(entry.get('files') or ())} declared file(s), "
            f"{len(entry.get('shards') or ())} shard(s) listed, "
            f"{len((entry.get('tokenizer') or {}).get('files') or ())} tokenizer file(s)"
            + (f", {len(entry.get('voices') or ())} preset voice(s)"
               if entry.get("voices") is not None else ""))


class Options:
    def __init__(self, keep_closure=False, model=False, torch=False, pypi=None, raw=None):
        self.keep_closure = keep_closure
        self.model = model
        self.torch = torch
        self.pypi = pypi or PyPI()
        self.raw = raw or Raw()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report what would be written and change nothing")
    parser.add_argument("--keep-closure", action="store_true",
                        help="re-pin the triples already written instead of solving")
    parser.add_argument("--model", action="store_true",
                        help="also size and hash the model, its shards and the tokenizer")
    parser.add_argument("--torch", action="store_true",
                        help="also record the torch wheel's digest from the cu128 index")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    arguments = parser.parse_args(argv)

    def say(text):
        print(text, flush=True)

    if arguments.manifest.name in SIBLINGS:
        print(f"{arguments.manifest.name} belongs to another engine. This tool writes "
              f"VibeVoice's manifest and nothing else.", file=sys.stderr)
        return 2
    try:
        existing = (json.loads(arguments.manifest.read_text(encoding="utf-8"))
                    if arguments.manifest.exists() else {})
    except (OSError, ValueError) as exc:
        print(f"{arguments.manifest} could not be read ({exc})", file=sys.stderr)
        return 2
    if not isinstance(existing, dict):
        print(f"{arguments.manifest} is not a manifest", file=sys.stderr)
        return 2

    survey(existing, arguments.manifest, say)
    try:
        found, state = build(existing, Options(keep_closure=arguments.keep_closure,
                                               model=arguments.model,
                                               torch=arguments.torch), say)
    except PinError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    if state.disagreements:
        print("\nThe published artifacts no longer match what is checked in:",
              file=sys.stderr)
        for line in state.disagreements:
            print(f"  {line}", file=sys.stderr)
        print("\nNothing was written. A publisher who has re-uploaded is a review "
              "decision, not a script's.", file=sys.stderr)
        return 2

    say(f"\n{state.wheels} wheel(s) resolved; overlay "
        f"{'pinned' if state.overlay_pinned else 'not pinned'}; {state.presets} preset "
        f"voice(s) pinned; torch digest "
        f"{'recorded' if state.torch_recorded else 'not recorded'}; models "
        + ", ".join(f"{name} {'hashed' if done else 'declared but not hashed'}"
                    for name, done in state.models_hashed.items())
        + f"{f' ({state.shards} shards listed for {MODEL_ID})' if state.shards else ''}.")

    body = json.dumps(found, indent=2, ensure_ascii=False) + "\n"
    if arguments.check:
        current = (arguments.manifest.read_text(encoding="utf-8")
                   if arguments.manifest.exists() else "")
        same = _without_version(current) == _without_version(body)
        say("Nothing was written." + ("" if same else " The manifest would change."))
        return 0 if same and state.pinned else 1

    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(body, encoding="utf-8")
    say(f"Wrote {arguments.manifest} (manifest version {found['version']}, "
        f"pinned {'true' if state.pinned else 'false'}).")
    return 0 if state.pinned else 1


def _without_version(text: str) -> str:
    """A manifest's text with its version counter blanked, for ``--check``."""
    return re.sub(r'"version": \d+,', '"version": 0,', str(text or ""), count=1)


if __name__ == "__main__":
    raise SystemExit(main())
